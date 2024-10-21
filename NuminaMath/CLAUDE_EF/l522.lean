import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_altitude_equation_l522_52257

-- Define the triangle ABC
noncomputable def A : ℝ × ℝ := (1, 6)
noncomputable def B : ℝ × ℝ := (-1, -2)
noncomputable def C : ℝ × ℝ := (6, 3)

-- Define D as the midpoint of BC
noncomputable def D : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Theorem for the equation of the median AD
theorem median_equation : 
  ∀ (x y : ℝ), 11 * x + 3 * y - 29 = 0 ↔ 
  ∃ (t : ℝ), (x, y) = ((1 - t) * A.1 + t * D.1, (1 - t) * A.2 + t * D.2) :=
sorry

-- Theorem for the equation of the altitude on BC
theorem altitude_equation : 
  ∀ (x y : ℝ), 7 * x + 5 * y - 37 = 0 ↔ 
  ∃ (t : ℝ), (x, y) = (A.1 + t * (C.2 - B.2), A.2 - t * (C.1 - B.1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_altitude_equation_l522_52257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_translated_lines_l522_52248

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translation of a point in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a line -/
def applyTranslation (l : Line) (t : Translation) : Line :=
  { slope := l.slope,
    intercept := l.intercept + t.dy - l.slope * t.dx }

/-- Calculate the distance between two parallel lines -/
noncomputable def distanceBetweenParallelLines (l1 l2 : Line) : ℝ :=
  abs (l2.intercept - l1.intercept) / Real.sqrt (1 + l1.slope ^ 2)

/-- The main theorem -/
theorem distance_between_translated_lines (l : Line) :
  let l1 := applyTranslation l ⟨3, 5⟩
  let l2 := applyTranslation l1 ⟨1, -2⟩
  l2 = l →
  distanceBetweenParallelLines l l1 = 11 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_translated_lines_l522_52248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distribution_l522_52212

-- Define the set of teams
inductive Team : Type
  | A | B | C | D | E

-- Define the function that assigns points to each team
def points : Team → ℕ := sorry

-- Define the total number of games played
def total_games : ℕ := 10

-- Define the total points awarded in the tournament
def total_points : ℕ := 24

-- All teams have different points
axiom different_points : ∀ t1 t2 : Team, t1 ≠ t2 → points t1 ≠ points t2

-- Team A has the highest score
axiom A_highest : ∀ t : Team, t ≠ Team.A → points Team.A > points t

-- Team A lost to Team B
axiom A_lost_to_B : points Team.B > points Team.A

-- Teams B and C did not lose any games
axiom B_C_no_loss : points Team.B ≥ 4 ∧ points Team.C ≥ 4

-- Team C scored fewer points than Team D
axiom C_less_than_D : points Team.C < points Team.D

-- The sum of all points is equal to the total points awarded
axiom sum_of_points : points Team.A + points Team.B + points Team.C + points Team.D + points Team.E = total_points

-- Theorem: The only possible point distribution satisfying all conditions
theorem point_distribution :
  points Team.A = 7 ∧
  points Team.B = 6 ∧
  points Team.C = 4 ∧
  points Team.D = 5 ∧
  points Team.E = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distribution_l522_52212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l522_52209

-- Define the line C
def line_C (t : ℝ) : ℝ × ℝ := (2 + t, t + 1)

-- Define the circle P
def circle_P (p : ℝ × ℝ) : Prop := (p.1 - 2)^2 + p.2^2 = 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_C t ∧ circle_P p}

-- Theorem statement
theorem intersection_length :
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧ ‖A - B‖ = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l522_52209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_through_focus_and_vertex_l522_52241

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

/-- The left focus of the hyperbola -/
def left_focus : ℝ × ℝ := (-6, 0)

/-- The right vertex of the hyperbola -/
def right_vertex : ℝ × ℝ := (4, 0)

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 25

/-- The theorem stating that the given circle is the one with smallest area passing through the left focus and right vertex of the hyperbola -/
theorem smallest_circle_through_focus_and_vertex :
  ∀ x y : ℝ, hyperbola x y →
  (∀ c : ℝ × ℝ → Prop, 
    (c left_focus ∧ c right_vertex) →
    (∀ x y : ℝ, c (x, y) → (x + 1)^2 + y^2 ≤ 25)) →
  circle_eq x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_through_focus_and_vertex_l522_52241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_upper_bound_l522_52242

noncomputable def N (a₁ a₂ a₃ : ℕ+) : ℕ :=
  (Finset.filter (fun (x : ℕ × ℕ × ℕ) => 
    (a₁ : ℝ) / x.1 + (a₂ : ℝ) / x.2.1 + (a₃ : ℝ) / x.2.2 = 1 ∧ 
    x.1 > 0 ∧ x.2.1 > 0 ∧ x.2.2 > 0) (Finset.range 10000 ×ˢ Finset.range 10000 ×ˢ Finset.range 10000)).card

theorem N_upper_bound (a₁ a₂ a₃ : ℕ+) (h₁ : a₁ ≥ a₂) (h₂ : a₂ ≥ a₃) :
  (N a₁ a₂ a₃ : ℝ) ≤ 6 * a₁ * a₂ * (3 + Real.log (2 * a₁)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_upper_bound_l522_52242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_cd_l522_52255

noncomputable def cylinder_volume (r : ℝ) (h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

noncomputable def total_volume (r : ℝ) (h : ℝ) : ℝ := cylinder_volume r h + 2 * hemisphere_volume r

theorem length_of_cd (r h : ℝ) :
  r = 4 → total_volume r h = 512 * Real.pi → h = 80/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_cd_l522_52255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_squared_is_49_l522_52210

/-- Represents a triangular pyramid with vertices A, B, C, and D. -/
structure TriangularPyramid where
  AB : ℝ
  CD : ℝ
  AD : ℝ
  BC : ℝ
  angleABC : ℝ

/-- The square of the radius of the smallest sphere enclosing a triangular pyramid. -/
noncomputable def smallestEnclosingSphereRadiusSquared (p : TriangularPyramid) : ℝ :=
  (p.AB^2 + p.BC^2 + 2 * p.AB * p.BC * Real.cos p.angleABC) / 4

/-- Theorem: The square of the radius of the smallest enclosing sphere for the given pyramid is 49. -/
theorem smallest_enclosing_sphere_radius_squared_is_49 :
  let p := TriangularPyramid.mk 6 6 10 10 (2 * Real.pi / 3)
  smallestEnclosingSphereRadiusSquared p = 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_squared_is_49_l522_52210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_with_specific_remainders_l522_52222

theorem three_digit_integers_with_specific_remainders : 
  let S : Set ℕ := {n | 100 ≤ n ∧ n < 1000 ∧ 
                        n % 7 = 3 ∧ 
                        n % 8 = 2 ∧ 
                        n % 13 = 4}
  Finset.card (Finset.filter (fun n => n ∈ S) (Finset.range 1000)) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_with_specific_remainders_l522_52222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_set_size_l522_52276

theorem largest_set_size (M : Finset ℕ) (n : ℕ) :
  (∀ (a b : ℕ), a ∈ M → b ∈ M → a ≠ b → (a^2 + 1) % b = 0) →
  M.card = n →
  n ≥ 2 →
  n ≤ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_set_size_l522_52276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l522_52261

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- Define the domain of f
def dom_f : Set ℝ := {x | -1 ≤ x ∧ x < 0}

-- Define the inverse function
noncomputable def f_inv (x : ℝ) : ℝ := -Real.sqrt (x + 1)

-- Define the domain of f_inv
def dom_f_inv : Set ℝ := Set.Ioc (-1) 0

theorem f_inverse_correct :
  (∀ x ∈ dom_f, f x ∈ dom_f_inv) ∧
  (∀ y ∈ dom_f_inv, f_inv y ∈ dom_f) ∧
  (∀ x ∈ dom_f, f_inv (f x) = x) ∧
  (∀ y ∈ dom_f_inv, f (f_inv y) = y) := by
  sorry

#check f_inverse_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l522_52261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_max_at_6_l522_52267

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 0
  | n + 1 => a n + 2 * n + 1

-- Define the sequence b_n
noncomputable def b (n : ℕ+) : ℝ :=
  n.val * Real.sqrt (a n.val + 1 : ℝ) * (8 / 11) ^ (n.val - 1)

-- Theorem stating that b_6 is the maximum term
theorem b_max_at_6 :
  ∀ n : ℕ+, b 6 ≥ b n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_max_at_6_l522_52267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_l522_52201

/-- Given two points A and B in 3D space, symmetric with respect to the y-axis, 
    prove that their coordinates satisfy specific values. -/
theorem symmetric_points (x y z : ℝ) : 
  let A : ℝ × ℝ × ℝ := (x^2 + 4, 4 - y, 1 + 2*z)
  let B : ℝ × ℝ × ℝ := (-4*x, 9, 7 - z)
  (A.fst = -B.fst ∧ A.snd = B.snd ∧ A.2 = -B.2) → 
  (x = 2 ∧ y = -5 ∧ z = -8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_l522_52201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_street_intersections_l522_52230

/-- Represents the length of Apple Street in kilometers -/
def street_length : ℚ := 3.2

/-- Represents the distance between intersections in kilometers -/
def intersection_distance : ℚ := 0.2

/-- Calculates the number of numbered intersections on Apple Street -/
def numbered_intersections : ℕ :=
  let total_intersections := (street_length / intersection_distance).floor
  (total_intersections - 2).toNat

theorem apple_street_intersections :
  numbered_intersections = 14 := by
  -- Proof goes here
  sorry

#eval numbered_intersections

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_street_intersections_l522_52230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_composition_commutes_l522_52286

/-- Chebyshev polynomial of the first kind -/
noncomputable def T (n : ℕ) : ℝ → ℝ := sorry

/-- Property of Chebyshev polynomials: T_n(cos(φ)) = cos(nφ) -/
axiom T_cos_property (n : ℕ) (φ : ℝ) : T n (Real.cos φ) = Real.cos (n * φ)

/-- Commutativity of composition of Chebyshev polynomials -/
theorem chebyshev_composition_commutes (m n : ℕ) (x : ℝ) : 
  T n (T m x) = T m (T n x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_composition_commutes_l522_52286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_row_10_sum_l522_52218

/-- Pascal's Triangle row -/
def pascal_row (n : ℕ) : List ℕ :=
  match n with
  | 0 => [1]
  | n + 1 => 
    let prev := pascal_row n
    1 :: (List.zipWith (· + ·) prev (prev.tail ++ [0]))

/-- Sum of elements in a list -/
def list_sum (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem pascal_row_10_sum : list_sum (pascal_row 10) = 2^10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_row_10_sum_l522_52218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l522_52273

/-- Given vectors a, b, and c in ℝ², prove that if a + λb is perpendicular to c, then λ = 1 -/
theorem perpendicular_vector_lambda (a b c : ℝ × ℝ) (lambda : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (3, 0))
    (h3 : c = (1, -2))
    (h4 : (a.1 + lambda * b.1, a.2 + lambda * b.2) • c = 0) :
  lambda = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l522_52273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l522_52263

theorem negation_of_proposition :
  (∀ x : ℝ, Real.sin x ≤ 1) ↔ ¬(∃ x : ℝ, Real.sin x > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l522_52263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_in_hexagon_l522_52280

open EuclideanGeometry

-- Define the vertices of the hexagon
variable (A B C D E F M : EuclideanSpace ℝ (Fin 2))

-- Define the property that points lie on a circle
def lie_on_circle (points : List (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ), ∀ p ∈ points, dist p center = radius

-- Define the circumcenter of a triangle
noncomputable def circumcenter (p q r : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  sorry

-- Define the area of a quadrilateral
noncomputable def area_quad (p q r s : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

-- Define ConvexPolygon
def ConvexPolygon (points : List (EuclideanSpace ℝ (Fin 2))) : Prop :=
  sorry

-- Define LineSegment
def LineSegment (p q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

-- The main theorem
theorem equal_areas_in_hexagon 
  (hex_convex : ConvexPolygon [A, B, C, D, E, F])
  (common_point : ∃ M, M ∈ LineSegment A D ∧ M ∈ LineSegment B E ∧ M ∈ LineSegment C F)
  (circumcenters_on_circle : lie_on_circle 
    [circumcenter M A B, circumcenter M B C, circumcenter M C D,
     circumcenter M D E, circumcenter M E F, circumcenter M F A]) :
  area_quad A B D E = area_quad B C E F ∧ 
  area_quad B C E F = area_quad C D F A :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_in_hexagon_l522_52280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_cant_afford_trip_l522_52259

/-- Calculates the future value of an investment -/
noncomputable def future_value (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates the future cost considering inflation -/
noncomputable def future_cost (initial_cost : ℝ) (inflation_rate : ℝ) (time : ℝ) : ℝ :=
  initial_cost * (1 + inflation_rate) ^ time

theorem anna_cant_afford_trip (
  initial_savings : ℝ)
  (initial_cost : ℝ)
  (interest_rate : ℝ)
  (inflation_rate : ℝ)
  (time : ℝ)
  (h1 : initial_savings = 40000)
  (h2 : initial_cost = 45000)
  (h3 : interest_rate = 0.05)
  (h4 : inflation_rate = 0.05)
  (h5 : time = 3) :
  future_value initial_savings interest_rate time < 
  future_cost initial_cost inflation_rate time := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_cant_afford_trip_l522_52259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_example_l522_52252

-- Define the custom operation as noncomputable
noncomputable def custom_op (a b : ℝ) : ℝ := a - (5 * a) / (2 * b)

-- Theorem statement
theorem custom_op_example : custom_op 10 4 = 3.75 := by
  -- Unfold the definition of custom_op
  unfold custom_op
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_example_l522_52252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l522_52269

/-- Two triangles are similar -/
def similar_triangles (ABC FGH : Set (ℝ × ℝ)) : Prop := sorry

def BC (triangle : Set (ℝ × ℝ)) : ℝ := sorry
def FG (triangle : Set (ℝ × ℝ)) : ℝ := sorry
def AC (triangle : Set (ℝ × ℝ)) : ℝ := sorry
def GH (triangle : Set (ℝ × ℝ)) : ℝ := sorry

theorem similar_triangles_side_length 
  (ABC FGH : Set (ℝ × ℝ))
  (h_similar : similar_triangles ABC FGH)
  (h_BC : 24 = BC ABC)
  (h_FG : 15 = FG FGH)
  (h_AC : 18 = AC ABC) :
  11.25 = GH FGH := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l522_52269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l522_52236

/-- For an angle α whose terminal side passes through the point (-6, 8), cos α = -3/5 -/
theorem cos_alpha_for_point (α : ℝ) : 
  (∃ (x y : ℝ), x = -6 ∧ y = 8 ∧ (Real.cos α) * (Real.sqrt (x^2 + y^2)) = x) →
  Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l522_52236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l522_52233

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 3*x - b else -2^(-x) - 3*x + 1

-- State the theorem
theorem odd_function_value (b : ℝ) :
  (∀ x, f b x = -f b (-x)) →  -- f is odd
  f b (-2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l522_52233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joker_not_certain_l522_52226

/-- Represents a standard deck of cards -/
structure Deck where
  cards : Nat
  jokers : Nat

/-- Defines a standard deck of cards -/
def standardDeck : Deck :=
  { cards := 52, jokers := 2 }

/-- Probability of drawing a joker from a deck -/
noncomputable def probDrawJoker (d : Deck) : ℝ :=
  d.jokers / (d.cards + d.jokers : ℝ)

/-- Theorem: Drawing a joker from a standard deck is not a certain event -/
theorem joker_not_certain : probDrawJoker standardDeck ≠ 1 := by
  sorry

#eval standardDeck.cards -- This line is added to ensure the code is executable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joker_not_certain_l522_52226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_perfect_squares_l522_52254

theorem not_all_perfect_squares (d : ℕ) 
  (h_pos : d > 0) 
  (h_neq_2 : d ≠ 2) 
  (h_neq_5 : d ≠ 5) 
  (h_neq_13 : d ≠ 13) : 
  ∃ a b : ℕ, a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ k : ℕ, a * b - 1 = k^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_perfect_squares_l522_52254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l522_52296

noncomputable def f (x : Real) : Real := 2 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    ∃ y ∈ Set.Icc (-1) 2, f x = y ∧
    ∀ z, f x = z → z ∈ Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l522_52296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_abc_l522_52258

theorem unique_solution_abc : 
  ∀ (a b c : ℕ), 
    a > b ∧ b > c → 
    34 - 6 * (a + b + c) + (a * b + b * c + c * a) = 0 → 
    79 - 9 * (a + b + c) + (a * b + b * c + c * a) = 0 → 
    a = 10 ∧ b = 3 ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_abc_l522_52258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_problem_l522_52235

/-- Given vectors satisfying certain conditions, prove their dot product equals 4 -/
theorem vector_dot_product_problem (a b c : ℝ × ℝ × ℝ) : 
  (2 • a + b = (0, -5, 10)) → 
  (c = (1, -2, -2)) → 
  (b • c = -18) → 
  (a • c = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_problem_l522_52235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_work_theorem_l522_52256

/-- Work done by a spring -/
noncomputable def work_done (k : ℝ) (x : ℝ) : ℝ :=
  (1/2) * k * x^2

/-- Spring constant calculation -/
noncomputable def spring_constant (force : ℝ) (displacement : ℝ) : ℝ :=
  force / displacement

theorem spring_work_theorem (force : ℝ) (initial_displacement : ℝ) (final_displacement : ℝ) 
    (h1 : force = 100)
    (h2 : initial_displacement = 0.1)
    (h3 : final_displacement = 0.2) :
  work_done (spring_constant force initial_displacement) final_displacement = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_work_theorem_l522_52256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_value_l522_52206

-- Define the slopes of the two lines
noncomputable def slope1 (a : ℝ) : ℝ := -1 / (Real.log a / Real.log 4)
def slope2 : ℝ := 2

-- State the theorem
theorem parallel_lines_imply_a_value (a : ℝ) (h : a > 0) :
  slope1 a = slope2 → a = 1/2 := by
  sorry

-- Note: The condition a > 0 is added to ensure log a is defined

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_value_l522_52206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_cos_C_l522_52202

/-- Given a triangle ABC where A = B (isosceles), a = 3, and c = 2, prove that cos C = 7/9 -/
theorem isosceles_triangle_cos_C (A B C : ℝ) (a b c : ℝ) : 
  A = B → a = 3 → c = 2 → Real.cos C = 7/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_cos_C_l522_52202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ynm_measure_l522_52293

-- Define the triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real

-- Define the angle measure function
noncomputable def angle_measure (a b c : Real) : Real := sorry

-- State the theorem
theorem angle_ynm_measure (t : Triangle) (M N : Real)
  (h1 : angle_measure t.X t.Y t.Z = 70)
  (h2 : angle_measure t.Z t.X t.Y = 50)
  (h3 : M ∈ Set.Icc t.X t.Y)
  (h4 : N ∈ Set.Icc t.Y t.Z)
  (h5 : |M - t.Y| = |t.Y - N|) :
  angle_measure t.Y N M = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ynm_measure_l522_52293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l522_52288

/-- Given a triangle whose dimensions are doubled to form a new triangle, 
    this theorem relates the areas of the original and new triangles. -/
theorem original_triangle_area 
  (original new : Real) 
  (h_doubled : new = 2 * original) 
  (h_new_area : new^2 = 32) : 
  original^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l522_52288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_week_training_time_l522_52220

/-- Represents the daily training time for a two-week period -/
structure TrainingSchedule where
  week1_daily_max : ℚ
  total_hours : ℚ

/-- Calculates the daily training time for the second week -/
def second_week_daily_time (schedule : TrainingSchedule) : ℚ :=
  (schedule.total_hours - 7 * schedule.week1_daily_max) / 7

/-- Theorem stating that given the conditions, the daily training time in the second week is 3 hours -/
theorem second_week_training_time (schedule : TrainingSchedule) 
  (h1 : schedule.week1_daily_max = 2)
  (h2 : schedule.total_hours = 35) :
  second_week_daily_time schedule = 3 := by
  unfold second_week_daily_time
  simp [h1, h2]
  norm_num

#eval second_week_daily_time { week1_daily_max := 2, total_hours := 35 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_week_training_time_l522_52220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l522_52211

/-- Given a parabola with equation x² = 2y, its directrix has equation y = -1/2 -/
theorem parabola_directrix :
  ∃ k : ℝ, k = -1/2 ∧ ∀ x y : ℝ, x^2 = 2*y → y = k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l522_52211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l522_52250

theorem quadratic_function_properties (a b c : ℝ) 
  (ha : a > 0) (hb : b < 0) (hc : c < 0) :
  let f := fun x => a * x^2 + b * x + c
  let discriminant := b^2 - 4*a*c
  let vertex_x := -b / (2*a)
  ∃ (x₁ x₂ : ℝ), 
    (∀ x, (deriv f) x = 2*a*x + b) ∧  -- Opens upwards
    (discriminant > 0) ∧               -- Two distinct real roots
    (vertex_x > 0) ∧                   -- Vertex on positive x-side
    (f 0 < 0)                          -- Negative y-intercept
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l522_52250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_nested_calculation_l522_52268

-- Define the ♡ operation
noncomputable def heart (x y : ℝ) : ℝ := x - Real.sqrt (1 / y)

-- State the theorem
theorem heart_nested_calculation :
  heart 3 (heart 3 3) = (27 - Real.sqrt (3 * (9 + Real.sqrt 3))) / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_nested_calculation_l522_52268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_tangent_to_x_axis_l522_52216

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- Define the function g
noncomputable def g (x a : ℝ) : ℝ := Real.exp (x - 1) - 1 / x - Real.log x - x + a

-- Theorem for the maximum value of f
theorem f_max_value : ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 1 / Real.exp 1 := by
  sorry

-- Theorem for g being tangent to x-axis when a = 1
theorem g_tangent_to_x_axis : 
  ∃ (t : ℝ), g t 1 = 0 ∧ (deriv (λ x => g x 1)) t = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_tangent_to_x_axis_l522_52216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kilometer_race_time_l522_52283

/-- Represents a race between two runners --/
structure Race where
  distance : ℝ
  time_difference : ℝ
  distance_difference : ℝ

/-- Calculates the time taken by the faster runner to complete the race --/
noncomputable def race_time (r : Race) : ℝ :=
  r.distance * r.time_difference / r.distance_difference

/-- Theorem stating that for the given race conditions, the faster runner completes the race in 100 seconds --/
theorem kilometer_race_time (r : Race) 
  (h1 : r.distance = 1000)
  (h2 : r.time_difference = 10)
  (h3 : r.distance_difference = 100) : 
  race_time r = 100 := by
  -- Unfold the definition of race_time
  unfold race_time
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

#check kilometer_race_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kilometer_race_time_l522_52283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DSO_measure_l522_52237

-- Define the triangle DOG
structure Triangle (D O G : Point) where
  -- Add any necessary conditions for a valid triangle

-- Define the angle measure in degrees
def angle_measure (p q r : Point) : ℝ := sorry

-- State the theorem
theorem angle_DSO_measure
  (D O G S : Point)
  (triangle : Triangle D O G)
  (angle_equality : angle_measure D G O = angle_measure D O G)
  (angle_DOG : angle_measure D O G = 30)
  (OS_bisects : angle_measure D O S = (angle_measure D O G) / 2) :
  angle_measure D S O = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DSO_measure_l522_52237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_average_18_l522_52224

/-- The average of the first n terms of an arithmetic progression -/
noncomputable def arithmeticProgressionAverage (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (a + (a + (n - 1) * d)) / 2

/-- Theorem: The average of the first 18 terms in an arithmetic progression 
    with first term 4 and common difference 6 is equal to 55 -/
theorem arithmetic_progression_average_18 :
  arithmeticProgressionAverage 4 6 18 = 55 := by
  -- Unfold the definition of arithmeticProgressionAverage
  unfold arithmeticProgressionAverage
  -- Simplify the expression
  simp [Nat.cast_sub, Nat.cast_one]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_average_18_l522_52224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l522_52266

/-- The polynomial f(x) = x^12 - x^6 + 1 -/
def f (x : ℂ) : ℂ := x^12 - x^6 + 1

/-- The divisor polynomial x^2 + 1 -/
def divisor (x : ℂ) : ℂ := x^2 + 1

/-- The remainder function ax + b -/
def remainder (a b : ℝ) (x : ℂ) : ℂ := a * x + b

theorem polynomial_division_remainder :
  ∃ (q : ℂ → ℂ) (a b : ℝ),
    (∀ x : ℂ, f x = divisor x * q x + remainder a b x) ∧
    b = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l522_52266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_class_size_l522_52285

/-- Represents the number of students in each of the four equal rows -/
def x : ℕ := 10

/-- Represents the total number of students in the class -/
def class_size : ℕ := 4 * x + (x + 3)

/-- The class size is at least 50 -/
axiom class_size_constraint : class_size ≥ 50

/-- The class size is the smallest possible that satisfies the constraints -/
axiom smallest_class_size : ∀ y : ℕ, y < x → 4 * y + (y + 3) < 50

theorem smallest_possible_class_size : class_size = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_class_size_l522_52285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_inequality_c_half_is_smallest_l522_52249

theorem smallest_c_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  Real.sqrt (x^2 * y^2) + (1/2) * abs (x^2 - y^2) ≥ (x^2 + y^2) / 2 :=
sorry

theorem c_half_is_smallest :
  ∀ c : ℝ, c > 0 → c < 1/2 →
  ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧
  Real.sqrt (x^2 * y^2) + c * abs (x^2 - y^2) < (x^2 + y^2) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_inequality_c_half_is_smallest_l522_52249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l522_52219

noncomputable def f (x : ℝ) := Real.cos (2 * x) + Real.sin (2 * x)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧
    (∀ q : ℝ, q > 0 → (∀ x : ℝ, f (x + q) = f x) → p ≤ q)) ∧
  (∀ k : ℤ, ∀ x y : ℝ,
    x ∈ Set.Icc (k * Real.pi + Real.pi / 8) (k * Real.pi + 5 * Real.pi / 8) →
    y ∈ Set.Icc (k * Real.pi + Real.pi / 8) (k * Real.pi + 5 * Real.pi / 8) →
    x ≤ y → f y ≤ f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l522_52219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_property_sum_of_first_30_terms_l522_52228

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The theorem stating the properties of the geometric sequence sums -/
theorem geometric_sequence_sum_property :
  S 10 = 32 →
  S 20 = 56 →
  S 30 = 74 := by
  sorry

/-- The main theorem proving the sum of the first 30 terms -/
theorem sum_of_first_30_terms :
  S 10 = 32 →
  S 20 = 56 →
  S 30 = 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_property_sum_of_first_30_terms_l522_52228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l522_52204

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  eq : (y : ℝ) → (x : ℝ) → Prop := fun y x => y^2 = 4*a*x

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.eq y x

/-- Focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (p.a, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: Distance from point on parabola to focus -/
theorem distance_to_focus (p : Parabola) (point : PointOnParabola p) 
  (h : distance (point.x, point.y) (point.x, 0) = 2 * Real.sqrt 3) :
  distance (point.x, point.y) (focus p) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l522_52204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_correct_l522_52275

/-- Given a hyperbola with equation (x-4)²/7² - (y-20)²/15² = 1, 
    this function returns the coordinates of the focus with the smaller x-coordinate -/
noncomputable def hyperbola_focus : ℝ × ℝ :=
  (4 - Real.sqrt 274, 20)

/-- Theorem stating that the function hyperbola_focus correctly returns 
    the coordinates of the focus with the smaller x-coordinate for the given hyperbola -/
theorem hyperbola_focus_correct : 
  let (x, y) := hyperbola_focus
  (x - 4)^2 / 7^2 - (y - 20)^2 / 15^2 = 1 ∧ 
  ∀ x' y', (x' - 4)^2 / 7^2 - (y' - 20)^2 / 15^2 = 1 → 
    (x' < 4 → x ≤ x') ∧ 
    (x' ≥ 4 → x < x' ∨ (x = x' ∧ y = y')) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_correct_l522_52275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l522_52207

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x - 2 = -y^2 + 10*y

-- Define the center of the circle
def is_center (c d s : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - c)^2 + (y - d)^2 = s^2

-- Define the radius of the circle
def is_radius (c d s : ℝ) : Prop :=
  s > 0 ∧ ∀ x y : ℝ, circle_equation x y → (x - c)^2 + (y - d)^2 = s^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∀ c d s : ℝ,
  (∃ x y : ℝ, circle_equation x y) →
  is_center c d s →
  is_radius c d s →
  c + d + s = 9 + Real.sqrt 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l522_52207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_p_not_increasing_prob_l522_52227

/-- The probability of teams having equal points after two matches -/
noncomputable def equal_points_prob (p : ℝ) : ℝ := (3 * p^2 - 2 * p + 1) / 4

/-- The derivative of the equal points probability function -/
noncomputable def equal_points_prob_derivative (p : ℝ) : ℝ := (3 * p - 1) / 2

theorem exists_p_not_increasing_prob :
  ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ equal_points_prob_derivative p < 0 := by
  use 0
  constructor
  · exact le_refl 0
  constructor
  · exact zero_le_one
  · simp [equal_points_prob_derivative]
    norm_num

#check exists_p_not_increasing_prob

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_p_not_increasing_prob_l522_52227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l522_52205

def sequence_a : ℕ → ℚ
  | 0 => 2/3  -- Add case for 0
  | 1 => 2/3
  | n + 1 => 2 * sequence_a n / (sequence_a n + 1)

theorem sequence_a_properties :
  (∀ n : ℕ, n ≥ 1 → ∃ r : ℚ, (1 / sequence_a (n + 1) - 1) = r * (1 / sequence_a n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = (2^n : ℚ) / ((2^n : ℚ) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l522_52205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circleplus_power_equality_l522_52291

-- Define the binary operation ⊕ as noncomputable
noncomputable def circleplus (a b : ℝ) : ℝ := a ^ (b ^ 2)

-- State the theorem
theorem circleplus_power_equality (a b n : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  (circleplus a b) ^ n = circleplus a (b * n) ↔ n = 1 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circleplus_power_equality_l522_52291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_a_l522_52279

noncomputable def a (n : ℕ) : ℝ := (n + 1)^2 / (n^2 + 1)

theorem limit_of_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, n ≥ 10000 → |a n - 1| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_a_l522_52279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_is_43_l522_52234

/-- Represents the number of students who borrowed a specific number of books -/
structure BookBorrowers where
  zero : Nat
  one : Nat
  two : Nat

/-- Represents the class of students and their book borrowing data -/
structure ClassData where
  borrowers : BookBorrowers
  totalStudents : Nat
  averageBooks : Rat

/-- Theorem stating that given the conditions, the total number of students is 43 -/
theorem total_students_is_43 (c : ClassData)
  (h1 : c.borrowers.zero = 2)
  (h2 : c.borrowers.one = 12)
  (h3 : c.borrowers.two = 13)
  (h4 : c.averageBooks = 2)
  (h5 : c.totalStudents = c.borrowers.zero + c.borrowers.one + c.borrowers.two + 
        (c.totalStudents - (c.borrowers.zero + c.borrowers.one + c.borrowers.two)))
  (h6 : (0 * c.borrowers.zero + 1 * c.borrowers.one + 2 * c.borrowers.two + 
         3 * (c.totalStudents - (c.borrowers.zero + c.borrowers.one + c.borrowers.two))) / c.totalStudents = c.averageBooks) :
  c.totalStudents = 43 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_is_43_l522_52234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_package_cost_l522_52272

/-- The cost of one package of flour, given the conditions of Claire's cake-making scenario -/
theorem flour_package_cost (packages_per_cake : ℕ) (num_cakes : ℕ) (total_cost : ℚ) : 
  packages_per_cake = 2 →
  num_cakes = 2 →
  total_cost = 12 →
  packages_per_cake * num_cakes > 0 →
  total_cost / (packages_per_cake * num_cakes) = 3 := by
  intros h1 h2 h3 h4
  -- The cost of one package of flour is $3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_package_cost_l522_52272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_time_l522_52270

noncomputable def work_rate (x : ℝ) := 1 / x

noncomputable def condition_A : ℝ := work_rate 4
noncomputable def condition_AC : ℝ := work_rate 2
noncomputable def condition_BC : ℝ := work_rate 3

theorem b_work_time : 
  ∃ (rate_B : ℝ), 
    rate_B = condition_BC - (condition_AC - condition_A) ∧ 
    (1 / rate_B) = 12 := by
  -- Define rate_B
  let rate_B := condition_BC - (condition_AC - condition_A)
  
  -- Prove the first part of the conjunction
  have h1 : rate_B = condition_BC - (condition_AC - condition_A) := rfl
  
  -- Prove the second part of the conjunction
  have h2 : (1 / rate_B) = 12 := by
    -- This is where the actual calculation would go
    -- For now, we'll use sorry to skip the proof
    sorry
  
  -- Combine the proofs
  exact ⟨rate_B, ⟨h1, h2⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_time_l522_52270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_implies_x_greater_than_one_l522_52246

theorem exponential_inequality_implies_x_greater_than_one (x : ℝ) :
  (2 : ℝ)^(3 - 2*x) > (1/2 : ℝ)^(3*x - 4) → x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_implies_x_greater_than_one_l522_52246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_numbers_satisfying_inequality_l522_52214

theorem natural_numbers_satisfying_inequality :
  {x : ℕ | (2 * x : ℤ) > 2 ∧ (2 * x : ℤ) ≤ 4} = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_numbers_satisfying_inequality_l522_52214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_area_l522_52217

theorem right_triangle_max_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  a + b + c = 48 →   -- perimeter condition
  c = 20 →           -- hypotenuse condition
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 
    x^2 + y^2 = z^2 → x + y + z = 48 → z = 20 → 
    Real.sqrt (((x + y + z) / 2) * (((x + y + z) / 2) - x) * (((x + y + z) / 2) - y) * (((x + y + z) / 2) - z)) ≤ 
    Real.sqrt (((a + b + c) / 2) * (((a + b + c) / 2) - a) * (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c))) →
  Real.sqrt (((a + b + c) / 2) * (((a + b + c) / 2) - a) * (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c)) = 96 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_area_l522_52217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greeting_card_profit_l522_52229

/-- Represents the store's greeting card sales problem --/
theorem greeting_card_profit
  (purchase_price : ℚ)
  (total_sales : ℚ)
  (selling_price : ℚ)
  (h1 : selling_price ≤ 2 * purchase_price)
  (h2 : (total_sales / selling_price).den = 1)  -- Ensures integer number of cards
  (h3 : purchase_price = 21/10)  -- in dimes
  (h4 : total_sales = 1457/10)   -- in dimes (14.57 yuan = 1457 fen = 145.7 dimes)
  (h5 : selling_price = 31/10)   -- in dimes (31 fen = 3.1 dimes)
  : total_sales - (total_sales / selling_price) * purchase_price = 47 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greeting_card_profit_l522_52229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_left_handed_fraction_l522_52232

theorem football_team_left_handed_fraction :
  ∀ (total_players throwers right_handed : ℕ),
    total_players = 70 →
    throwers = 40 →
    right_handed = 60 →
    (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_left_handed_fraction_l522_52232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bounds_l522_52225

noncomputable def x : ℕ → ℝ
  | 0 => 2  -- Added case for 0 to cover all natural numbers
  | 1 => 2
  | n + 1 => x n / 2 + 1 / x n

theorem x_bounds (n : ℕ) (hn : n ≥ 1) : Real.sqrt 2 < x n ∧ x n < Real.sqrt 2 + 1 / (n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bounds_l522_52225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_proof_l522_52208

theorem right_triangle_proof (A B C : ℝ) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) 
  (h3 : A + B + C = π) (h4 : Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin C) : C = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_proof_l522_52208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSum_eq_five_l522_52299

/-- The sum of the series 20^k / ((4^k - 3^k)(4^(k+1) - 3^(k+1))) from k=1 to infinity -/
noncomputable def infiniteSum : ℝ := ∑' k, (20^k) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

/-- The theorem stating that the infinite sum converges to 5 -/
theorem infiniteSum_eq_five : infiniteSum = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSum_eq_five_l522_52299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_extended_line_l522_52290

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in a plane -/
structure Line where
  point1 : Point
  point2 : Point

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Check if a point is on a line -/
def onLine (p : Point) (l : Line) : Prop := sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop := sorry

/-- Check if a circle is inscribed in a triangle -/
def isInscribed (c : Circle) (a b d : Point) : Prop := sorry

/-- Check if three points are collinear -/
def collinear (a b d : Point) : Prop := sorry

/-- The intersection point of two lines -/
noncomputable def intersection (l1 l2 : Line) : Point := sorry

theorem point_on_extended_line 
  (c : Circle) (l : Line) (M Q R P T : Point) :
  isTangent l c →
  onLine M l →
  onLine Q l →
  onLine R l →
  isMidpoint M Q R →
  isInscribed c P Q R →
  let N := intersection (Line.mk P T) (Line.mk Q R)
  collinear P N T := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_extended_line_l522_52290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_vector_problem_l522_52213

/-- Given a quadrilateral ABCD on a plane, prove the values of k, m, and n. -/
theorem quadrilateral_vector_problem (a b c : ℝ × ℝ) 
  (ha : a = (4, 1)) 
  (hb : b = (3, 1)) 
  (hc : c = (-1, -2)) 
  (hperp : (a.1 + 2*b.1, a.2 + 2*b.2) • (b.1 - k*c.1, b.2 - k*c.2) = 0) 
  (hdb : ∃ m n : ℝ, (1, 2) - (3, 1) = m • ((-6, 0) : ℝ × ℝ) + n • ((1, 2) : ℝ × ℝ)) :
  k = -33/16 ∧ m = 5/2 ∧ n = 1/2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_vector_problem_l522_52213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_over_complex_trig_l522_52287

/-- The definite integral of (sin x) / ((1 + cos x - sin x)^2) from -π/2 to 0 is equal to 1/2 - ln 2 -/
theorem integral_sin_over_complex_trig :
  ∫ x in Set.Icc (-Real.pi/2) 0, Real.sin x / (1 + Real.cos x - Real.sin x)^2 = 1/2 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_over_complex_trig_l522_52287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l522_52278

-- Define the equations
def equation_A (x : ℝ) : ℝ := (x - 3)^2 + 1
def equation_B (x : ℝ) : ℝ := abs (2*x + 1) - 6
noncomputable def equation_C (x : ℝ) : ℝ := Real.sqrt (5 - x) + 3
noncomputable def equation_D (x : ℝ) : ℝ := Real.sqrt (3*x + 4) - 7
def equation_E (x : ℝ) : ℝ := abs (2*x + 2) + 5

-- Theorem stating which equations have no solution and which have solutions
theorem equation_solutions :
  (∀ x : ℝ, equation_A x ≠ 0) ∧
  (∃ x : ℝ, equation_B x = 0) ∧
  (∀ x : ℝ, equation_C x ≠ 0) ∧
  (∃ x : ℝ, equation_D x = 0) ∧
  (∀ x : ℝ, equation_E x ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l522_52278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_people_probability_valid_arrangements_recursive_valid_arrangements_base_fair_coin_probability_l522_52203

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing. -/
def validArrangements : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => validArrangements (n + 1) + validArrangements n

/-- The probability of no two adjacent people standing when n people flip fair coins in a circular arrangement. -/
def noAdjacentStandingProbability (n : ℕ) : ℚ :=
  validArrangements n / 2^n

theorem ten_people_probability :
  noAdjacentStandingProbability 10 = 123 / 1024 := by
  sorry

theorem valid_arrangements_recursive (n : ℕ) (h : n ≥ 2) :
  validArrangements n = validArrangements (n-1) + validArrangements (n-2) := by
  sorry

theorem valid_arrangements_base :
  validArrangements 0 = 1 ∧ validArrangements 1 = 2 := by
  sorry

theorem fair_coin_probability (n : ℕ) :
  (2 : ℚ)^n > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_people_probability_valid_arrangements_recursive_valid_arrangements_base_fair_coin_probability_l522_52203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_alternating_pattern_l522_52247

/-- Represents the color of a counter -/
inductive Color
  | Black
  | White

/-- Represents a 4x4 grid of counters -/
def Grid := Fin 4 → Fin 4 → Color

/-- Represents a move that flips a 2x2 square of counters -/
structure Move where
  x : Fin 3
  y : Fin 3

/-- Apply a move to a grid -/
def applyMove (g : Grid) (m : Move) : Grid :=
  fun i j =>
    if i.val ∈ [m.x.val, m.x.val + 1] ∧ j.val ∈ [m.y.val, m.y.val + 1] then
      match g i j with
      | Color.Black => Color.White
      | Color.White => Color.Black
    else g i j

/-- Check if a grid has an alternating pattern -/
def isAlternating (g : Grid) : Prop :=
  ∀ i j, g i j = (if (i.val + j.val) % 2 = 0 then Color.Black else Color.White)

/-- Initial grid with all counters black -/
def initialGrid : Grid := fun _ _ => Color.Black

/-- Theorem: At least 6 moves are required to achieve an alternating pattern -/
theorem min_moves_for_alternating_pattern :
  ∀ (moves : List Move),
    isAlternating (moves.foldl applyMove initialGrid) →
    moves.length ≥ 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_alternating_pattern_l522_52247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l522_52221

open Real

-- Define the quadratic function
noncomputable def q (x : ℝ) := x^2 - x + 2

-- Define the proposition function
noncomputable def f (x : ℝ) := 4/x - log x / log 3

-- State the theorem
theorem problem_statement :
  (¬ ∃ x : ℝ, q x < 0) ∧ (∃ y : ℝ, 3 < y ∧ y < 4 ∧ f y = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l522_52221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_circumradius_and_inradius_l522_52284

noncomputable section

-- Define a right triangle ABC with legs AB = 6 and BC = 8
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let ab := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let bc := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let ac := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  ab = 6 ∧ bc = 8 ∧ ab^2 + bc^2 = ac^2

-- Define the circumradius of a triangle
noncomputable def circumradius (A B C : ℝ × ℝ) : ℝ :=
  let a := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let b := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let c := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (a * b * c) / (4 * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)))

-- Define the inradius of a triangle
noncomputable def inradius (A B C : ℝ × ℝ) : ℝ :=
  let a := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let b := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let c := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

-- Theorem statement
theorem sum_of_circumradius_and_inradius (A B C : ℝ × ℝ) :
  triangle_ABC A B C → circumradius A B C + inradius A B C = 7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_circumradius_and_inradius_l522_52284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_probability_l522_52239

/-- The probability that at least 7 out of 8 people stay for an entire concert,
    given that 3 are certain to stay and 5 have a 1/3 chance of staying. -/
theorem concert_probability : ℝ := by
  -- Define the total number of people
  let total_people : ℕ := 8
  -- Define the number of people certain to stay
  let certain_people : ℕ := 3
  -- Define the number of uncertain people
  let uncertain_people : ℕ := 5
  -- Define the probability of an uncertain person staying
  let stay_prob : ℝ := 1/3

  -- Calculate the probability
  have prob : ℝ := (uncertain_people.choose 4 : ℝ) * stay_prob^4 * (1 - stay_prob)^1 +
               (uncertain_people.choose 5 : ℝ) * stay_prob^5

  -- Assert that this probability equals 11/243
  have h : prob = 11/243 := by sorry

  -- Return the result
  exact prob

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_probability_l522_52239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l522_52244

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 9

-- State the theorem
theorem zero_in_interval :
  (∀ x y, 0 < x ∧ x < y → f x < f y) →  -- f is increasing on (0, +∞)
  f 2 < 0 →                             -- f(2) < 0
  f 3 > 0 →                             -- f(3) > 0
  ∃! x, 2 < x ∧ x < 3 ∧ f x = 0 :=      -- Unique zero in (2, 3)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l522_52244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratio_after_doubling_side_l522_52274

theorem square_area_ratio_after_doubling_side (s : ℝ) (h : s > 0) :
  (s^2) / ((2*s)^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratio_after_doubling_side_l522_52274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_after_25_years_l522_52245

/-- Calculates the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Calculates the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem balance_difference_after_25_years : 
  let angela_balance := compound_interest 9000 0.025 50
  let bob_balance := simple_interest 11000 0.06 25
  round_to_nearest (|angela_balance - bob_balance|) = 2977 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_after_25_years_l522_52245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_cos_range_l522_52262

theorem sin_greater_cos_range (α : Real) : 
  α ∈ Set.Ioo 0 (2 * Real.pi) → (Real.sin α > Real.cos α ↔ α ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_cos_range_l522_52262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m3_to_dm3_conversion_cm2_to_m2_conversion_L_to_mL_conversion_l522_52238

-- Define conversion rates
noncomputable def m3_to_dm3_rate : ℝ := 1000
noncomputable def m2_to_cm2_rate : ℝ := 10000
noncomputable def L_to_mL_rate : ℝ := 1000

-- Define the conversion functions
noncomputable def convert_m3_to_dm3 (x : ℝ) : ℝ := x * m3_to_dm3_rate
noncomputable def convert_cm2_to_m2 (x : ℝ) : ℝ := x / m2_to_cm2_rate
noncomputable def convert_L_to_mL (x : ℝ) : ℝ := x * L_to_mL_rate

-- Theorem statements
theorem m3_to_dm3_conversion : convert_m3_to_dm3 4.75 = 4750 := by sorry

theorem cm2_to_m2_conversion : convert_cm2_to_m2 6500 = 0.65 := by sorry

theorem L_to_mL_conversion : convert_L_to_mL 3.05 = 3050 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m3_to_dm3_conversion_cm2_to_m2_conversion_L_to_mL_conversion_l522_52238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l522_52264

-- Define the complex number z
noncomputable def z : ℂ := 5 / (Complex.I * (Complex.I + 2))

-- Theorem statement
theorem imaginary_part_of_z : Complex.im z = -2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l522_52264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_value_f_inverse_is_inverse_f_inverse_is_odd_l522_52260

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

-- Statement 1
theorem f_inverse_value (m : ℝ) : f m = 2 → m = Real.log (2 + Real.sqrt 5) := by sorry

-- Statement 2
noncomputable def f_inverse (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1))

theorem f_inverse_is_inverse : Function.LeftInverse f_inverse f ∧ Function.RightInverse f_inverse f := by sorry

-- Statement 3
theorem f_inverse_is_odd : ∀ x, f_inverse (-x) = -f_inverse x := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_value_f_inverse_is_inverse_f_inverse_is_odd_l522_52260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_min_m_value_l522_52240

/-- The function f(x) as defined in the problem -/
noncomputable def f (x m : ℝ) : ℝ := 
  3 * Real.sqrt 2 * Real.sin (x/4) * Real.cos (x/4) + 
  Real.sqrt 6 * (Real.cos (x/4))^2 - Real.sqrt 6 / 2 + m

/-- The theorem stating the maximum value of m -/
theorem max_m_value (m : ℝ) : 
  (∀ x, -5*π/6 ≤ x ∧ x ≤ π/6 → f x m ≤ 0) →
  m ≤ -Real.sqrt 3 := by
  sorry

/-- The theorem stating the minimum value of m -/
theorem min_m_value (m : ℝ) : 
  m ≤ -Real.sqrt 3 →
  (∃ x, -5*π/6 ≤ x ∧ x ≤ π/6 ∧ f x m = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_min_m_value_l522_52240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_OM_and_min_distance_l522_52215

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Curve defined by parametric equations -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Convert polar coordinates to Cartesian coordinates -/
noncomputable def polarToCartesian (r : ℝ) (θ : ℝ) : Point :=
  { x := r * Real.cos θ, y := r * Real.sin θ }

/-- Define the curve C -/
noncomputable def curveC : ParametricCurve :=
  { x := λ α => 1 + Real.sqrt 2 * Real.cos α,
    y := λ α => Real.sqrt 2 * Real.sin α }

/-- Calculate distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem line_OM_and_min_distance (M : Point) (h : M = polarToCartesian (4 * Real.sqrt 2) (Real.pi / 4)) :
  (∀ x : ℝ, x = M.x → M.y = x) ∧ 
  (∀ p : Point, ∃ α : ℝ, p.x = (curveC.x α) ∧ p.y = (curveC.y α) → 
    distance M p ≥ 5 - Real.sqrt 2 ∧ 
    ∃ q : Point, ∃ β : ℝ, q.x = (curveC.x β) ∧ q.y = (curveC.y β) ∧ distance M q = 5 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_OM_and_min_distance_l522_52215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_range_l522_52297

/-- The system of inequalities has only one integer solution -/
def has_one_integer_solution (k : ℝ) : Prop :=
  ∃! (x : ℤ), (x^2 - 2*x - 8 > 0) ∧ (2*x^2 + (2*k + 7)*x + 7*k < 0)

/-- The range of k for which the system has only one integer solution -/
def k_range : Set ℝ :=
  Set.Icc (-5 : ℝ) 3 ∪ Set.Ioc 4 5

theorem system_solution_range :
  ∀ k, has_one_integer_solution k ↔ k ∈ k_range :=
sorry

#check system_solution_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_range_l522_52297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l522_52200

-- Define the functions
noncomputable def f (x : ℝ) := Real.sqrt (x^2 - 1)
noncomputable def g : ℝ → ℝ := Real.sin

-- State the theorem
theorem inequality_proof 
  (a b c d : ℝ) 
  (h1 : a > b) (h2 : b ≥ 1) 
  (h3 : c > d) (h4 : d > 0) 
  (h5 : f a - f b = π) 
  (h6 : g c - g d = π / 10) : 
  a + d - b - c < 9 * π / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l522_52200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_ordering_l522_52277

theorem sine_ordering : Real.sin 4 < Real.sin 6 ∧ Real.sin 6 < Real.sin 3 ∧ Real.sin 3 < Real.sin 1 ∧ Real.sin 1 < Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_ordering_l522_52277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cars_return_to_start_l522_52295

/-- Represents a car on the circular track -/
structure Car where
  position : ℕ  -- Current position (point) on the track
  direction : Bool  -- true for clockwise, false for counterclockwise

/-- Represents the state of all cars on the track at a given time -/
def TrackState := List Car

/-- The circular track with n points and n cars -/
def CircularTrack (n : ℕ) :=
  { state : TrackState // state.length = n }

/-- Simulates the movement of cars for one hour -/
def moveForOneHour (track : CircularTrack n) : CircularTrack n :=
  sorry

/-- Checks if all cars are at their starting positions -/
def allCarsAtStart (track : CircularTrack n) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem all_cars_return_to_start (n : ℕ) (initialTrack : CircularTrack n) :
  ∃ t : ℕ, allCarsAtStart (Nat.iterate moveForOneHour t initialTrack) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cars_return_to_start_l522_52295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necklace_divisibility_l522_52289

/-- The number of ways to make an even number of necklaces with n beads -/
def D₀ : ℕ → ℕ := sorry

/-- The number of ways to make an odd number of necklaces with n beads -/
def D₁ : ℕ → ℕ := sorry

/-- n - 1 divides D₁(n) - D₀(n) for all n ≥ 2 -/
theorem necklace_divisibility (n : ℕ) (h : n ≥ 2) :
  (n - 1) ∣ (D₁ n - D₀ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necklace_divisibility_l522_52289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_equilateral_triangle_l522_52243

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define properties of triangles
def Triangle.isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.A = t.B ∧ t.B = t.C

-- State the theorems
theorem obtuse_triangle (t : Triangle) :
  Real.cos t.A * Real.cos t.B * Real.cos t.C < 0 → t.isObtuse :=
sorry

theorem equilateral_triangle (t : Triangle) :
  Real.cos (t.A - t.B) * Real.cos (t.B - t.C) * Real.cos (t.C - t.A) = 1 → t.isEquilateral :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_equilateral_triangle_l522_52243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l522_52231

theorem smallest_difference (a b : ℤ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (h1 : ∃ k : ℤ, (a + b) / 2 = k)
  (h2 : ∃ m : ℤ, m * m = a * b)
  (h3 : ∃ n : ℤ, 2 * a * b / (a + b) = n) :
  ∃ p q : ℤ, p > 0 ∧ q > 0 ∧ p ≠ q ∧
    (∃ k : ℤ, (p + q) / 2 = k) ∧
    (∃ m : ℤ, m * m = p * q) ∧
    (∃ n : ℤ, 2 * p * q / (p + q) = n) ∧
    |p - q| = 3 ∧
    ∀ x y : ℤ, x > 0 → y > 0 → x ≠ y →
      (∃ k : ℤ, (x + y) / 2 = k) →
      (∃ m : ℤ, m * m = x * y) →
      (∃ n : ℤ, 2 * x * y / (x + y) = n) →
      |x - y| ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l522_52231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_functions_l522_52294

def is_valid_function (f : ℕ → ℕ) : Prop :=
  ∃ (k : ℕ) (p : ℕ), Nat.Prime p ∧
  (∀ n ≥ k, f (n + p) = f n) ∧
  (∀ m n : ℕ, m ∣ n → f (m + 1) ∣ (f n + 1))

theorem characterize_valid_functions (f : ℕ → ℕ) (h : is_valid_function f) :
  (∃ (k : ℕ) (p : ℕ), Nat.Prime p ∧
    (∀ n : ℕ, n % p ≠ 1 → f n = 1) ∧
    (∀ n ≥ k, n % p = 1 → f n = 2) ∧
    (∀ n : ℕ, 1 < n → n < k → n % p = 1 → f n = 1 ∨ f n = 2) ∧
    (∃ m : ℕ, f 1 = m ∧ f 2 ∣ (m + 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_functions_l522_52294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l522_52265

noncomputable def sequenceA (a : ℝ) (n : ℕ) : ℝ := a * n^2 + n

theorem sequence_range (a : ℝ) :
  (∀ n ∈ Finset.range 5, sequenceA a n < sequenceA a (n + 1)) ∧
  (∀ n ≥ 10, sequenceA a n > sequenceA a (n + 1)) →
  a ∈ Set.Icc (-1/12 : ℝ) (-1/20 : ℝ) := by
  sorry

#check sequence_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l522_52265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_price_reduction_initial_sale_is_ten_percent_l522_52281

/-- Represents the initial sale percentage as a real number between 0 and 1 -/
def initial_sale_percentage : ℝ := sorry

/-- The final price is 81% of the original price after two reductions -/
theorem sale_price_reduction (P : ℝ) (h : P > 0) : 
  0.90 * (1 - initial_sale_percentage) * P = 0.81 * P :=
sorry

/-- The initial sale percentage is 10% -/
theorem initial_sale_is_ten_percent : initial_sale_percentage = 0.1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_price_reduction_initial_sale_is_ten_percent_l522_52281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l522_52282

noncomputable def f (x : ℝ) : ℝ := (1/2) * x + 5/2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 4*a - 3

def a_values : Set ℝ := {1, 3}

def a_range : Set ℝ := Set.Icc (5/6) 2

theorem problem_statement :
  (∀ a ∈ a_values, Set.range (g a) = Set.Ici 0) ∧
  (∀ a ∈ a_range, ∀ x₁ ∈ Set.Icc (-1) 1, ∃ x₂ ∈ Set.Icc (-1) 1, f x₁ = g a x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l522_52282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l522_52223

/-- The circle C with center (1, 1) and radius 1 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

/-- The line l: x/4 + y/3 = 1 -/
def l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 / 4 + p.2 / 3 = 1}

/-- The center of the circle C -/
def center : ℝ × ℝ := (1, 1)

/-- The radius of the circle C -/
def radius : ℝ := 1

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  |A * p.1 + B * p.2 + C| / Real.sqrt (A^2 + B^2)

theorem circle_tangent_to_line :
  distance_point_to_line center (1/4) (1/3) (-1) = radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l522_52223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_from_area_and_radius_l522_52253

/-- Given a circle with radius 12 meters and a sector with an area of 47.77142857142857 square meters,
    the central angle of the sector is approximately 38.197 degrees. -/
theorem sector_angle_from_area_and_radius :
  let radius : ℝ := 12
  let sector_area : ℝ := 47.77142857142857
  let central_angle : ℝ := (sector_area * 360) / (π * radius^2)
  ∃ ε > 0, |central_angle - 38.197| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_from_area_and_radius_l522_52253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_z_value_l522_52271

def A : ℂ := -3 - 2*Complex.I
def B : ℂ := -4 + 5*Complex.I
def C : ℂ := 2 + Complex.I
def D (z : ℂ) : ℂ := z

-- Define what it means for ABCD to be a parallelogram
def is_parallelogram (A B C D : ℂ) : Prop :=
  B - A = D - C

-- Theorem statement
theorem parallelogram_z_value (z : ℂ) :
  is_parallelogram A B (C) (D z) → z = 1 + 8*Complex.I :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_z_value_l522_52271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l522_52292

noncomputable def f (x : ℝ) : ℝ := -1 / x

theorem f_increasing_on_positive_reals :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l522_52292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_is_the_spy_l522_52298

-- Define the types of individuals
inductive Individual : Type
  | A : Individual
  | B : Individual
  | C : Individual

-- Define the roles
inductive Role : Type
  | Knight : Role
  | Liar : Role
  | Spy : Role

-- Define a function to represent accusations
def accuses : Individual → Individual → Prop := sorry

-- Define a function to assign roles to individuals
def hasRole : Individual → Role → Prop := sorry

-- State the conditions of the problem
axiom one_of_each : ∃! k l s, hasRole k Role.Knight ∧ hasRole l Role.Liar ∧ hasRole s Role.Spy

axiom A_accuses_B : accuses Individual.A Individual.B
axiom B_accuses_C : accuses Individual.B Individual.C
axiom C_accuses_someone : accuses Individual.C Individual.A ∨ accuses Individual.C Individual.B

-- Define what it means for an accusation to be true
def accusation_true (accuser accused : Individual) : Prop :=
  (hasRole accuser Role.Knight ∧ hasRole accused Role.Spy) ∨
  (hasRole accuser Role.Spy ∧ hasRole accused Role.Spy)

-- Define what it means for an accusation to be false
def accusation_false (accuser accused : Individual) : Prop :=
  (hasRole accuser Role.Liar) ∨
  (hasRole accuser Role.Spy ∧ ¬hasRole accused Role.Spy)

-- State that each accusation is either true or false
axiom accusations_are_true_or_false :
  ∀ x y, accuses x y → (accusation_true x y ∨ accusation_false x y)

-- State the theorem to be proved
theorem C_is_the_spy : hasRole Individual.C Role.Spy := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_is_the_spy_l522_52298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_product_is_447_l522_52251

/-- Represents a row in the random number table -/
def RandomRow := List Nat

/-- The random number table -/
def randomTable : List RandomRow :=
  [[2839, 3125, 8395, 9524, 7232, 8995],
   [7216, 2884, 3660, 1073, 4366, 7575],
   [9436, 6118, 4479, 5140, 9694, 9592],
   [6017, 4951, 4068, 7516, 3241, 4782]]

/-- Converts a 3-digit number to a product number -/
def toProductNumber (n : Nat) : Nat :=
  n % 1000

/-- Selects numbers from the random table starting from a given position -/
def selectNumbers (startRow : Nat) (startCol : Nat) (count : Nat) : List Nat :=
  sorry

theorem fourth_product_is_447 :
  let selectedNumbers := selectNumbers 1 2 4
  let productNumbers := selectedNumbers.map toProductNumber
  productNumbers.length > 3 ∧ productNumbers[3]! = 447 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_product_is_447_l522_52251
