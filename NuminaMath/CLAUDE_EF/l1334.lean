import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_of_intersecting_lines_l1334_133485

/-- Two lines intersecting at a point -/
structure IntersectingLines :=
  (point : ℝ × ℝ)
  (slope1 : ℝ)
  (slope2 : ℝ)

/-- Calculate the x-intercept of a line given its slope and a point it passes through -/
noncomputable def x_intercept (slope : ℝ) (point : ℝ × ℝ) : ℝ :=
  point.1 - point.2 / slope

/-- Calculate the distance between two points on the x-axis -/
noncomputable def x_axis_distance (x1 x2 : ℝ) : ℝ :=
  |x1 - x2|

/-- The main theorem -/
theorem x_intercept_distance_of_intersecting_lines (lines : IntersectingLines)
  (h1 : lines.point = (8, 20))
  (h2 : lines.slope1 = 4)
  (h3 : lines.slope2 = -3) :
  x_axis_distance (x_intercept lines.slope1 lines.point) (x_intercept lines.slope2 lines.point) = 35/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_of_intersecting_lines_l1334_133485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_b_value_l1334_133461

/-- Proves that for a cubic function y = ax³ + bx + c, if two points (2, y₁) and (-2, y₂) on the graph satisfy y₁ - y₂ = 12, then b = 3 - 4a. -/
theorem cubic_function_b_value (a b c y₁ y₂ : ℝ) :
  y₁ = a * 2^3 + b * 2 + c →
  y₂ = a * (-2)^3 + b * (-2) + c →
  y₁ - y₂ = 12 →
  b = 3 - 4 * a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_b_value_l1334_133461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1334_133471

noncomputable def curve (x : ℝ) : ℝ := Real.log x + 2 * x

def tangent_slope : ℝ := 3

theorem tangent_line_equation (x₀ y₀ : ℝ) (h1 : y₀ = curve x₀) 
  (h2 : (1 / x₀ + 2) = tangent_slope) :
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧ 
                 ∀ x y, a * x + b * y + c = 0 ↔ y - y₀ = tangent_slope * (x - x₀) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1334_133471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_in_W_l1334_133432

noncomputable def sequence1 (n : ℕ) : ℝ := n^2 + 1
noncomputable def sequence2 (n : ℕ) : ℝ := (2*n + 9) / (2*n + 11)
noncomputable def sequence3 (n : ℕ) : ℝ := 2 + 4/n
noncomputable def sequence4 (n : ℕ) : ℝ := 1 - 1/(2^n)

def satisfies_condition1 (a : ℕ → ℝ) : Prop :=
  ∀ n, (a n + a (n+2)) / 2 < a (n+1)

def satisfies_condition2 (a : ℕ → ℝ) : Prop :=
  ∃ M, ∀ n, a n ≤ M

def belongs_to_W (a : ℕ → ℝ) : Prop :=
  satisfies_condition1 a ∧ satisfies_condition2 a

theorem sequences_in_W :
  ¬ belongs_to_W sequence1 ∧
  belongs_to_W sequence2 ∧
  ¬ belongs_to_W sequence3 ∧
  belongs_to_W sequence4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_in_W_l1334_133432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sides_sum_lower_bound_l1334_133446

/-- A convex polygon with pq sides, where p and q are prime numbers and p < q -/
structure ConvexPolygon (p q : ℕ) where
  is_prime_p : Nat.Prime p
  is_prime_q : Nat.Prime q
  p_lt_q : p < q
  sides : Fin (p * q) → ℕ
  angles : Fin (p * q) → ℝ
  all_angles_equal : ∀ i j : Fin (p * q), angles i = angles j
  distinct_side_lengths : ∀ i j : Fin (p * q), i ≠ j → sides i ≠ sides j
  positive_side_lengths : ∀ i : Fin (p * q), sides i > 0

/-- The sum of k consecutive side lengths in the polygon -/
def sum_consecutive_sides {p q : ℕ} (polygon : ConvexPolygon p q) (k : ℕ) : ℕ :=
  (Finset.range k).sum (λ i => polygon.sides ⟨i, by sorry⟩)

/-- The main theorem to be proved -/
theorem consecutive_sides_sum_lower_bound
  {p q : ℕ} (polygon : ConvexPolygon p q) (k : ℕ) (h : 1 ≤ k ∧ k ≤ p) :
  sum_consecutive_sides polygon k ≥ (k^3 + k) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sides_sum_lower_bound_l1334_133446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1334_133454

theorem trig_identities (x : ℝ) 
  (h1 : 0 < x) (h2 : x < π) (h3 : Real.sin x + Real.cos x = Real.sqrt 5 / 5) : 
  (Real.sin x - Real.cos x = 3 * Real.sqrt 5 / 5) ∧ 
  ((Real.sin (2 * x) + 2 * (Real.sin x) ^ 2) / (1 - Real.tan x) = 4 / 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1334_133454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sqrt_count_l1334_133435

theorem integer_sqrt_count : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, ∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) ∧ Finset.card S = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sqrt_count_l1334_133435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1334_133445

noncomputable def f (x : ℝ) : ℝ := min (4 * x + 1) (min (x + 2) (-2 * x + 4))

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1334_133445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_filling_rate_conversion_l1334_133449

/-- Converts barrels per minute to cubic meters per hour -/
noncomputable def barrels_to_cubic_meters_per_hour (barrels_per_minute : ℝ) (liters_per_barrel : ℝ) : ℝ :=
  barrels_per_minute * liters_per_barrel * 60 / 1000

/-- Theorem stating that 5 barrels per minute is equivalent to 47.7 cubic meters per hour -/
theorem filling_rate_conversion :
  barrels_to_cubic_meters_per_hour 5 159 = 47.7 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval barrels_to_cubic_meters_per_hour 5 159

end NUMINAMATH_CALUDE_ERRORFEEDBACK_filling_rate_conversion_l1334_133449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1334_133407

theorem sum_remainder (a b c : ℕ) 
  (ha : a % 53 = 31)
  (hb : b % 53 = 15)
  (hc : c % 53 = 7) :
  (a + b + c) % 53 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1334_133407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_configurations_l1334_133470

def total_seats : ℕ := 360

def is_valid_configuration (seats_per_row : ℕ) : Bool :=
  seats_per_row ≥ 18 &&
  (total_seats / seats_per_row) ≥ 12 &&
  total_seats % seats_per_row = 0

def valid_seat_configurations : List ℕ :=
  (List.range (total_seats + 1)).filter is_valid_configuration

theorem sum_of_valid_configurations :
  valid_seat_configurations.sum = 110 := by
  sorry

#eval valid_seat_configurations
#eval valid_seat_configurations.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_configurations_l1334_133470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_rabbit_game_l1334_133419

/-- Represents the field as an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- Represents the position of an agent (fox or rabbit) in the field -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents the game state -/
structure GameState where
  field : EquilateralTriangle
  fox_pos : Position
  rabbit_pos : Position
  fox_speed : ℝ
  rabbit_speed : ℝ
  fox_speed_positive : fox_speed > 0
  rabbit_speed_positive : rabbit_speed > 0

/-- Predicate indicating that the fox catches the rabbit at time t -/
def fox_catches_rabbit (game : GameState) (t : ℝ) : Prop := sorry

/-- Predicate indicating that the rabbit escapes at time t -/
def rabbit_escapes (game : GameState) (t : ℝ) : Prop := sorry

/-- Theorem stating the conditions for fox catching the rabbit or rabbit escaping -/
theorem fox_rabbit_game (game : GameState) :
  (2 * game.fox_speed > game.rabbit_speed →
    ∃ (t : ℝ), t ≥ 0 ∧ fox_catches_rabbit game t) ∧
  (2 * game.fox_speed ≤ game.rabbit_speed →
    ∀ (t : ℝ), t ≥ 0 → rabbit_escapes game t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_rabbit_game_l1334_133419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_per_large_bottle_price_per_large_bottle_exists_l1334_133424

/-- The price per large bottle given the following conditions:
  * 1375 large bottles were purchased at an unknown price
  * 690 small bottles were purchased at $1.35 each
  * The approximate average price per bottle was $1.6163438256658595
-/
theorem price_per_large_bottle : ℝ → Prop :=
  fun price_large =>
    let num_large : ℕ := 1375
    let num_small : ℕ := 690
    let price_small : ℝ := 1.35
    let avg_price : ℝ := 1.6163438256658595
    (((num_large : ℝ) * price_large + (num_small : ℝ) * price_small) / ((num_large : ℝ) + (num_small : ℝ)) = avg_price) →
    (abs (price_large - 1.74979773148) < 0.00000000001)

theorem price_per_large_bottle_exists : ∃ price : ℝ, price_per_large_bottle price :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_per_large_bottle_price_per_large_bottle_exists_l1334_133424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l1334_133478

/-- The distance between two parallel lines ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distanceParallelLines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- The distance between the lines x + 2y = 5 and x + 2y = 10 -/
theorem distance_between_specific_lines :
  distanceParallelLines 1 2 (-5) (-10) = Real.sqrt 5 := by
  sorry

#check distance_between_specific_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l1334_133478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_identical_cubes_l1334_133406

-- Define a cube as a type with 6 faces, each face being either green or red
def Cube := Fin 6 → Bool

-- Define a function to check if two cubes are identical after rotation
def are_identical_after_rotation (cube1 cube2 : Cube) : Prop :=
  ∃ (rotation : Equiv.Perm (Fin 6)), ∀ i, cube1 i = cube2 (rotation i)

-- Define the sample space of all possible pairs of cubes
def all_cube_pairs : Finset (Cube × Cube) :=
  sorry

-- Define the event of two cubes being identical after rotation
def identical_after_rotation_event : Finset (Cube × Cube) :=
  sorry

-- State the theorem
theorem probability_of_identical_cubes :
  (Finset.card identical_after_rotation_event : ℚ) / (Finset.card all_cube_pairs : ℚ) = 45 / 2048 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_identical_cubes_l1334_133406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_kl_in_similar_triangles_l1334_133498

/-- Two triangles are similar -/
structure SimilarTriangles (T : Type) :=
  (sim : T → T → Prop)

/-- Given triangle -/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Theorem: Length of KL in similar triangles -/
theorem length_kl_in_similar_triangles
  (GHI JKL : Triangle)
  (sim : SimilarTriangles Triangle)
  (h_sim : sim.sim GHI JKL)
  (h_hi : GHI.side1 = 10)
  (h_gh : GHI.side2 = 7)
  (h_jk : JKL.side2 = 4) :
  JKL.side1 = (10 * 4) / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_kl_in_similar_triangles_l1334_133498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_inequality_l1334_133494

theorem smallest_constant_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x*y*(x^2 + y^2) + y*z*(y^2 + z^2) + z*x*(z^2 + x^2) ≤ (1/8)*(x + y + z)^4 ∧
  ∀ k : ℝ, (∀ a b c : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → 
    a*b*(a^2 + b^2) + b*c*(b^2 + c^2) + c*a*(c^2 + a^2) ≤ k*(a + b + c)^4) →
  k ≥ 1/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_inequality_l1334_133494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1334_133482

theorem cos_double_angle_special_case (θ : ℝ) (h : Real.cos θ = 1/3) :
  Real.cos (2 * θ) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1334_133482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_problem_l1334_133460

theorem integer_sum_problem :
  ∃ (a b : ℕ),
    a > 0 ∧ b > 0 ∧
    a * b + a + b = 87 ∧
    Nat.gcd a b = 1 ∧
    a < 15 ∧ b < 15 ∧
    (Even a ∨ Even b) ∧
    a + b = 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_problem_l1334_133460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_reciprocal_at_one_l1334_133408

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem derivative_reciprocal_at_one :
  deriv f 1 = -1 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_reciprocal_at_one_l1334_133408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1334_133427

noncomputable section

/-- Curve C in parametric form -/
def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 + 2 * Real.cos α, 1 + 2 * Real.sin α)

/-- Polar equation of curve C -/
def polar_equation (θ : ℝ) : ℝ := 4 * Real.sin (θ + Real.pi/3)

/-- Area of triangle AOB given angle θ -/
def triangle_area (θ : ℝ) : ℝ := 
  2 * Real.sqrt 3 * Real.cos (2*θ) + Real.sqrt 3

/-- Theorem stating the polar equation of curve C and the maximum area of triangle AOB -/
theorem curve_C_properties : 
  (∀ θ, polar_equation θ = (curve_C (θ - Real.pi/6)).1^2 + (curve_C (θ - Real.pi/6)).2^2) ∧ 
  (∀ θ, triangle_area θ ≤ 3 * Real.sqrt 3) ∧
  (∃ θ, triangle_area θ = 3 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1334_133427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_walk_l1334_133480

/-- The distance from the starting point after walking 7 km along a regular hexagon with side length 3 km -/
theorem distance_after_walk (side_length : ℝ) (walk_distance : ℝ) : 
  side_length = 3 →
  walk_distance = 7 →
  let end_point : ℝ × ℝ := (1, 2 * Real.sqrt 3)
  (end_point.1 ^ 2 + end_point.2 ^ 2).sqrt = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_walk_l1334_133480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composite_increasing_l1334_133475

/-- Given a function f that is decreasing on (2, 8),
    prove that f(4-x) is increasing on (-4, 2) -/
theorem f_composite_increasing (f : ℝ → ℝ) 
    (h : ∀ x y, 2 < x ∧ x < y ∧ y < 8 → f x > f y) :
    ∀ x y, -4 < x ∧ x < y ∧ y < 2 → f (4 - x) < f (4 - y) := by
  sorry

#check f_composite_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composite_increasing_l1334_133475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_possible_l1334_133426

/-- Represents a 10x10 table of natural numbers -/
def Table := Fin 10 → Fin 10 → Nat

/-- Predicate to check if a number is in the table -/
def number_in_table (t : Table) (n : Nat) : Prop :=
  ∃ i j, t i j = n

/-- Predicate to check if all numbers from 1 to 100 are in the table -/
def all_numbers_present (t : Table) : Prop :=
  ∀ n, 1 ≤ n ∧ n ≤ 100 → number_in_table t n

/-- Predicate to check if each number appears only once -/
def each_number_once (t : Table) : Prop :=
  ∀ n, 1 ≤ n ∧ n ≤ 100 → (∃! i j, t i j = n)

/-- Predicate to check if two positions are adjacent -/
def adjacent (i₁ j₁ i₂ j₂ : Fin 10) : Prop :=
  (i₁ = i₂ ∧ (j₁.val + 1 = j₂.val ∨ j₂.val + 1 = j₁.val)) ∨
  (j₁ = j₂ ∧ (i₁.val + 1 = i₂.val ∨ i₂.val + 1 = i₁.val))

/-- Predicate to check if a number is composite -/
def is_composite (n : Nat) : Prop :=
  n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- Function to swap two elements in the table -/
def swap_elements (t : Table) (i₁ j₁ i₂ j₂ : Fin 10) : Table :=
  fun i j => if (i = i₁ ∧ j = j₁) then t i₂ j₂
             else if (i = i₂ ∧ j = j₂) then t i₁ j₁
             else t i j

/-- The main theorem to prove -/
theorem rearrangement_possible (t : Table) 
  (h₁ : all_numbers_present t) 
  (h₂ : each_number_once t) : 
  ∃ (t' : Table), 
    (∀ i j i' j', adjacent i j i' j' → is_composite (t' i j + t' i' j')) ∧
    (∃ (swaps : List (Fin 10 × Fin 10 × Fin 10 × Fin 10)), 
      swaps.length ≤ 35 ∧ 
      t' = swaps.foldl (λ acc (i₁, j₁, i₂, j₂) => 
        swap_elements acc i₁ j₁ i₂ j₂) t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_possible_l1334_133426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l1334_133457

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi - α) * Real.cos (α - Real.pi / 2) * Real.cos (Real.pi + α)) /
  (Real.sin (Real.pi / 2 + α) * Real.cos (Real.pi / 2 + α) * Real.tan (3 * Real.pi + α))

theorem f_simplification_and_value (α : Real) 
  (h1 : Real.pi < α ∧ α < 3 * Real.pi / 2)  -- α is in the third quadrant
  (h2 : Real.sin (Real.pi + α) = 1 / 3) :
  (f α = Real.cos α) ∧ (f α = -2 * Real.sqrt 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l1334_133457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_equation_l1334_133492

/-- An ellipse with given eccentricity and area -/
structure Ellipse where
  eccentricity : ℝ
  area : ℝ

/-- The equation of an ellipse in standard form -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    x^2 / a^2 + y^2 / b^2 = 1 ∧
    e.eccentricity = Real.sqrt (1 - b^2 / a^2) ∧
    e.area = Real.pi * a * b

/-- The specific ellipse in the problem -/
noncomputable def ellipse_C : Ellipse :=
  { eccentricity := Real.sqrt 3 / 2
  , area := 8 * Real.pi }

theorem ellipse_C_equation :
  ellipse_equation ellipse_C = λ x y ↦ x^2 / 16 + y^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_equation_l1334_133492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_theorem_l1334_133431

noncomputable def reflect (v : ℝ × ℝ) (a b : ℝ × ℝ) : ℝ × ℝ :=
  let midpoint := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  let dir := (b.1 - a.1, b.2 - a.2)
  let proj := 
    let dot := v.1 * dir.1 + v.2 * dir.2
    let norm := dir.1 * dir.1 + dir.2 * dir.2
    (dot / norm * dir.1, dot / norm * dir.2)
  (2 * proj.1 - v.1, 2 * proj.2 - v.2)

theorem reflection_theorem :
  let r := reflect (-1, 7) (5, -5)
  r (-4, 3) = (0, -5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_theorem_l1334_133431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_circle_centers_l1334_133493

/-- The maximum distance between two points in a rectangle --/
noncomputable def max_distance_between_centers (width height diameter : ℝ) : ℝ :=
  Real.sqrt ((width - diameter) ^ 2 + (height - diameter) ^ 2)

/-- The greatest possible distance between the centers of two circles in a rectangle --/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 20)
  (h_height : rectangle_height = 16)
  (h_diameter : circle_diameter = 8)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = 4 * Real.sqrt 13 ∧
  d = max_distance_between_centers rectangle_width rectangle_height circle_diameter :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_circle_centers_l1334_133493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1334_133450

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

-- Part I
theorem part_one : ¬(B (1/2) ∩ A (1/2)) = ∅ := by sorry

-- Part II
theorem part_two : ∀ a : ℝ, A a ⊆ B a ↔ 
  a ∈ Set.Icc (-1/2 : ℝ) (1/3) ∪ Set.Ioo (1/3 : ℝ) ((3 - Real.sqrt 5) / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1334_133450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l1334_133467

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 49
def C₂ (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 64

-- Define the tangent line segment
def is_tangent_line_segment (P Q : ℝ × ℝ) : Prop :=
  C₁ P.1 P.2 ∧ C₂ Q.1 Q.2 ∧
  ∀ R : ℝ × ℝ, (R ≠ P ∧ R ≠ Q) → ¬(C₁ R.1 R.2 ∨ C₂ R.1 R.2)

-- Define the length of a line segment
noncomputable def length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, is_tangent_line_segment P Q ∧
    (∀ P' Q' : ℝ × ℝ, is_tangent_line_segment P' Q' →
      length P Q ≤ length P' Q') ∧
    length P Q = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l1334_133467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soldier_arrangement_exists_l1334_133404

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the distance between a point and a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Represents the arrangement of soldiers -/
structure SoldierArrangement where
  rows : List Line
  officerPosition : Point

/-- The main theorem stating the existence of a valid arrangement -/
theorem soldier_arrangement_exists : ∃ (arr : SoldierArrangement), 
  (arr.rows.length = 12) ∧ 
  (∀ l ∈ arr.rows, ∃ (points : List Point), points.length = 10) ∧
  (∀ l₁ l₂, l₁ ∈ arr.rows → l₂ ∈ arr.rows → 
    distancePointToLine arr.officerPosition l₁ = distancePointToLine arr.officerPosition l₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soldier_arrangement_exists_l1334_133404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_intersections_l1334_133428

-- Define a function to calculate the number of intersections between two polygons
def intersections (n m : ℕ) : ℕ := 2 * min n m

-- Define the set of polygons
def polygons : List ℕ := [4, 5, 6, 9]

-- Calculate the total number of intersections
def total_intersections : ℕ :=
  (List.sum (do
    let a ← polygons
    let b ← polygons
    if a < b then
      pure (intersections a b)
    else
      pure 0
  ))

-- Theorem statement
theorem polygon_intersections :
  total_intersections = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_intersections_l1334_133428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_b_time_l1334_133430

/-- Represents the efficiency of a worker in terms of the fraction of the job they can complete in one day -/
structure WorkerEfficiency where
  value : ℝ

/-- Represents the time taken to complete a job in days -/
structure Time where
  value : ℝ

/-- The total amount of work to be done -/
structure TotalWork where
  value : ℝ

theorem worker_b_time 
  (a b : WorkerEfficiency) 
  (total_work : TotalWork) 
  (combined_time : Time) :
  a.value = 2 * b.value →
  combined_time.value = 10 →
  (a.value + b.value) * combined_time.value = total_work.value →
  b.value * 30 = total_work.value :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_b_time_l1334_133430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1334_133490

theorem min_value_of_expression (x : ℝ) : (16 : ℝ)^x - (4 : ℝ)^x + 1 ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1334_133490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_one_sufficient_not_necessary_l1334_133442

noncomputable def f (a x : ℝ) : ℝ := Real.cos (a * x)^2 - Real.sin (a * x)^2

def is_smallest_positive_period (a : ℝ) : Prop :=
  ∀ x, f a (x + Real.pi) = f a x ∧ ∀ p, 0 < p ∧ p < Real.pi → ∃ x, f a (x + p) ≠ f a x

theorem a_one_sufficient_not_necessary :
  (∀ a, a = 1 → is_smallest_positive_period a) ∧
  ¬(∀ a, is_smallest_positive_period a → a = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_one_sufficient_not_necessary_l1334_133442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_deduction_l1334_133455

/-- The percentage of salary deducted as house rent -/
def house_rent_percentage : ℝ → ℝ := sorry

/-- Rahul's salary in Rupees -/
def salary : ℝ := 2125

/-- The amount left after all expenditures in Rupees -/
def amount_left : ℝ := 1377

/-- The percentage spent on children's education -/
def education_percentage : ℝ := 10

/-- The percentage spent on clothes -/
def clothes_percentage : ℝ := 10

theorem house_rent_deduction (h : ℝ) :
  h = house_rent_percentage salary →
  (1 - h / 100) * (1 - education_percentage / 100) * (1 - clothes_percentage / 100) * salary = amount_left →
  ∃ ε > 0, |h - 20| < ε := by
  sorry

#check house_rent_deduction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_deduction_l1334_133455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_defective_products_l1334_133458

def number_of_ways_to_select_at_least_two_defective_products (n : ℕ) (d : ℕ) (s : ℕ) : ℕ :=
  (Nat.choose d 2 * Nat.choose (n - d) (s - 2)) + 
  (Nat.choose d 3 * Nat.choose (n - d) (s - 3))

theorem at_least_two_defective_products (n : ℕ) (d : ℕ) (s : ℕ) 
  (h1 : n = 200) (h2 : d = 3) (h3 : s = 5) :
  number_of_ways_to_select_at_least_two_defective_products n d s = 
  (Nat.choose d 2 * Nat.choose (n - d) (s - 2)) + 
  (Nat.choose d 3 * Nat.choose (n - d) (s - 3)) :=
by
  unfold number_of_ways_to_select_at_least_two_defective_products
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_defective_products_l1334_133458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l1334_133466

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the intersection points
def intersect_points (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
  circle_C A.1 A.2 m ∧ circle_C B.1 B.2 m

-- Define the length of segment AB
noncomputable def segment_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem circle_intersection_theorem :
  ∀ (A B : ℝ × ℝ) (m : ℝ),
    intersect_points A B m →
    segment_length A B = 4 * Real.sqrt 5 / 5 →
    m = 1 ∨ m = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l1334_133466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1334_133469

-- Define the direction vector
noncomputable def direction_vector : ℝ × ℝ := (-1, Real.sqrt 3)

-- Define the inclination angle
noncomputable def inclination_angle : ℝ := 2 * Real.pi / 3

-- Theorem statement
theorem line_inclination_angle :
  let (x, y) := direction_vector
  Real.tan inclination_angle = y / x := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1334_133469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1334_133444

theorem range_of_a (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f = λ x ↦ a * x + 2 * a + 1) →
  (∃ (x y : ℝ), x ∈ Set.Icc (-1 : ℝ) 1 ∧ y ∈ Set.Icc (-1 : ℝ) 1 ∧ 
    (a * x + 2 * a + 1) > 0 ∧ (a * y + 2 * a + 1) < 0) →
  -1 < a ∧ a < -1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1334_133444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_slope_tangent_line_l1334_133418

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x + 1/3

-- The tangent line with the smallest slope
def tangent_line (x y : ℝ) : Prop := x + y - 3 = 0

theorem smallest_slope_tangent_line :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x, (deriv f x) ≥ (deriv f x₀)) ∧ 
    f x₀ = y₀ ∧
    tangent_line x₀ y₀ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_slope_tangent_line_l1334_133418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1334_133488

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (a₁ d : ℚ) (n : ℕ) : ℚ := n * (2 * a₁ + (n - 1) * d) / 2

/-- Given an arithmetic sequence with a₁ = 1 and d = 2, 
    prove that k = 5 when S_{k+2} - S_k = 24 -/
theorem arithmetic_sequence_problem (k : ℕ) :
  S 1 2 (k + 2) - S 1 2 k = 24 → k = 5 := by
  intro h
  -- The proof steps would go here
  sorry

#eval S 1 2 7 - S 1 2 5  -- This should evaluate to 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1334_133488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_decrease_interval_of_decrease_is_4_to_inf_l1334_133497

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4*x) / Real.log (1/2)

-- Define the domain of f
def domain (x : ℝ) : Prop := x < 0 ∨ x > 4

-- Theorem statement
theorem interval_of_decrease :
  ∀ x y : ℝ, domain x → x > 4 → y > 4 → x > y → f x < f y :=
by
  sorry

-- Corollary: The interval of decrease is (4, +∞)
theorem interval_of_decrease_is_4_to_inf :
  ∀ x : ℝ, x > 4 → (∀ y, y > x → f y < f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_decrease_interval_of_decrease_is_4_to_inf_l1334_133497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firstTermOfGeometricSequence_l1334_133400

/-- A geometric sequence {a_n} with first term a and common ratio r -/
def geometricSequence (a r : ℝ) : ℕ → ℝ := λ n ↦ a * r^(n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem firstTermOfGeometricSequence (a r : ℝ) :
  (geometricSum a r 4 = 240) →
  (geometricSequence a r 2 + geometricSequence a r 4 = 180) →
  a = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_firstTermOfGeometricSequence_l1334_133400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1334_133479

/-- The focus coordinates of the parabola 4y = x^2 are (0, 1) -/
theorem parabola_focus_coordinates :
  let parabola := {(x, y) : ℝ × ℝ | 4 * y = x^2}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (0, 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1334_133479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_parabola_l1334_133451

/-- The line on which point A lies -/
noncomputable def line (x : ℝ) : ℝ := 8/15 * x - 6

/-- The parabola on which point B lies -/
def parabola (x : ℝ) : ℝ := x^2

/-- The distance between two points (x₁, y₁) and (x₂, y₂) -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The minimum distance between a point on the line and a point on the parabola -/
theorem min_distance_line_parabola :
  ∃ (x₁ x₂ : ℝ), ∀ (a b : ℝ),
    distance x₁ (line x₁) x₂ (parabola x₂) ≤ distance a (line a) b (parabola b) ∧
    distance x₁ (line x₁) x₂ (parabola x₂) = 1334/255 := by
  sorry

#check min_distance_line_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_parabola_l1334_133451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1334_133465

-- Define the function f(x) = lg x - 2x^2 + 3
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10 - 2 * x^2 + 3

-- Theorem statement
theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1334_133465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1334_133491

/-- The equation of a line parameterized by (x, y) = (3t + 5, 5t - 7) is y = (5x - 46) / 3 -/
theorem line_equation (t : ℝ) :
  (5 * (3 * t + 5) - 46) / 3 = 5 * t - 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1334_133491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_ratio_min_trig_ratio_attained_l1334_133443

open Real

theorem min_trig_ratio (x : ℝ) : 
  (sin x)^8 + (cos x)^8 + 2 ≥ (14/27) * ((sin x)^6 + (cos x)^6 + 2) :=
sorry

theorem min_trig_ratio_attained : 
  ∃ x : ℝ, (sin x)^8 + (cos x)^8 + 2 = (14/27) * ((sin x)^6 + (cos x)^6 + 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_ratio_min_trig_ratio_attained_l1334_133443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1334_133410

theorem sum_remainder (a b c : ℕ) 
  (ha : a % 30 = 7)
  (hb : b % 30 = 11)
  (hc : c % 30 = 23) :
  (a + b + c) % 30 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1334_133410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_10_deg_eq_4cos_10_deg_plus_sqrt3_l1334_133453

/-- Given a regular 18-gon with side length 2 units, prove that cot 10° = 4 cos 10° + √3 -/
theorem cot_10_deg_eq_4cos_10_deg_plus_sqrt3 :
  Real.tan (π / 2 - 10 * π / 180) = 4 * Real.cos (10 * π / 180) + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_10_deg_eq_4cos_10_deg_plus_sqrt3_l1334_133453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_five_consecutive_integers_l1334_133402

/-- The largest integer that divides the product of any 5 consecutive integers is 60 -/
theorem largest_divisor_of_five_consecutive_integers : ∃ d : ℕ, 
  (∀ k : ℕ, d ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ 
  (∀ m : ℕ, (∀ k : ℕ, m ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) → m ≤ d) ∧
  d = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_five_consecutive_integers_l1334_133402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_equals_two_l1334_133462

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem function_derivative_equals_two (x : ℝ) (h1 : x > 0) :
  (deriv f x = 2) → x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_equals_two_l1334_133462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_n_l1334_133456

def point (n : ℝ) : ℝ × ℝ := (3*n - 9, n + 2)

theorem range_of_n (α : ℝ) (n : ℝ) :
  (∃ P, P = point n ∧ P.1 = (3*n - 9) ∧ P.2 = (n + 2)) →
  Real.cos α < 0 →
  Real.sin α > 0 →
  -2 < n ∧ n < 3 :=
by
  intro h1 h2 h3
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_n_l1334_133456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l1334_133487

/-- The area of a sector with radius 1 and central angle 2π/3 is π/3 -/
theorem sector_area (radius : Real) (central_angle : Real) :
  radius = 1 →
  central_angle = 2 * Real.pi / 3 →
  (1 / 2) * central_angle * radius^2 = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l1334_133487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_construction_l1334_133476

/-- Represents a point on a sphere or plane -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point
  radius : ℝ

/-- Represents a circle on a sphere or plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line segment -/
structure LineSegment where
  start : Point
  finish : Point  -- Changed 'end' to 'finish' to avoid keyword conflict

/-- Function to draw a circle on a sphere -/
def drawCircleOnSphere (s : Sphere) (center : Point) : Circle :=
  sorry

/-- Function to measure distance between two points -/
def measureDistance (p1 p2 : Point) : ℝ :=
  sorry

/-- Function to construct a plane triangle from three distances -/
def constructTriangle (d1 d2 d3 : ℝ) : (Point × Point × Point) :=
  sorry

/-- Function to construct circumcircle of a triangle -/
def constructCircumcircle (p1 p2 p3 : Point) : Circle :=
  sorry

/-- Function to construct right-angled triangle -/
def constructRightTriangle (hypotenuse leg : ℝ) : (Point × Point × Point) :=
  sorry

/-- Function to construct perpendicular line -/
def constructPerpendicular (p : Point) (l : LineSegment) : LineSegment :=
  sorry

/-- Main theorem: The radius of the sphere is half the length of the constructed line segment -/
theorem sphere_radius_construction (s : Sphere) :
  ∃ (segment : LineSegment), s.radius = (measureDistance segment.start segment.finish) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_construction_l1334_133476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_of_value_rounded_l1334_133452

theorem percent_of_value_rounded (x y z : ℝ) (hx : x = 4.85) (hy : y = 13.5) (hz : z = 1543) :
  Int.floor ((x / y) * z + 0.5) = 554 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_of_value_rounded_l1334_133452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_favorable_digit_l1334_133413

def is_favorable (d : ℕ) : Bool := d < 3 ∨ d = 7

def total_outcomes : ℕ := 10

def favorable_outcomes : ℕ := (List.range 10).filter is_favorable |>.length

theorem probability_of_favorable_digit : 
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 5 := by
  -- Proof goes here
  sorry

#eval favorable_outcomes -- This will evaluate to 4
#eval (favorable_outcomes : ℚ) / total_outcomes -- This will evaluate to 2/5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_favorable_digit_l1334_133413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_matrix_det_det_multiple_of_10_l1334_133441

/-- Definition of our special matrix -/
def special_matrix (n : ℕ) : Matrix (Fin n) (Fin n) ℤ :=
  Matrix.of (λ i j => if i = j then 8 else 3)

/-- Theorem about the determinant of the special matrix -/
theorem special_matrix_det (n : ℕ) :
  Matrix.det (special_matrix n) = 5^(n-1) * (3*n + 5) := by
  sorry

/-- Theorem about when the determinant is a multiple of 10 -/
theorem det_multiple_of_10 (n : ℕ) :
  10 ∣ Matrix.det (special_matrix n) ↔ Odd n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_matrix_det_det_multiple_of_10_l1334_133441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x20_is_494_l1334_133486

/-- The sum of a geometric series from 1 to x^n -/
noncomputable def geometricSum (x : ℝ) (n : ℕ) : ℝ := (1 - x^(n+1)) / (1 - x)

/-- The coefficient of x^20 in the expansion of the given expression -/
def coefficientX20 : ℕ := 494

/-- The theorem stating that the coefficient of x^20 in the expansion is 494 -/
theorem coefficient_x20_is_494 :
  ∃ (g : ℝ → ℝ), ∀ x, x ≠ 1 → 
    (geometricSum x 19) * (geometricSum x 11)^3 = g x + coefficientX20 * x^20 + x^21 * (g x / x) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x20_is_494_l1334_133486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vectors_with_prime_length_and_divisibility_condition_l1334_133401

/-- A vector in 3D space with integer coordinates -/
structure IntVector where
  x : ℤ
  y : ℤ
  z : ℤ

/-- The theorem statement -/
theorem max_vectors_with_prime_length_and_divisibility_condition
  (p : ℕ) (hp : Nat.Prime p) (n : ℕ) (v : Fin n → IntVector)
  (h_length : ∀ i, (v i).x^2 + (v i).y^2 + (v i).z^2 = p^2)
  (h_divisibility : ∀ j k, j < k → ∃ (ℓ : ℕ) (hℓ : 0 < ℓ ∧ ℓ < p),
    (∃ m : ℤ, (v j).x - ℓ * (v k).x = m * p) ∧
    (∃ m : ℤ, (v j).y - ℓ * (v k).y = m * p) ∧
    (∃ m : ℤ, (v j).z - ℓ * (v k).z = m * p)) :
  n ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vectors_with_prime_length_and_divisibility_condition_l1334_133401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_set_exists_l1334_133416

-- Define the function h(n) as the largest prime divisor of n
noncomputable def h (n : ℕ) : ℕ :=
  if n ≥ 2 then
    (Nat.factors n).maximum.getD 1
  else 1

-- Define the property we want to prove
def satisfiesCondition (n : ℕ) : Prop :=
  n ≥ 2 ∧ h n < h (n + 1) ∧ h (n + 1) < h (n + 2)

-- Theorem statement
theorem infinite_set_exists :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, satisfiesCondition n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_set_exists_l1334_133416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_equals_announced_l1334_133412

/-- Represents a person in the circle with their picked number and announced number -/
structure Person where
  picked : ℚ
  announced : ℚ

/-- The circle of 12 people -/
def Circle := Fin 12 → Person

/-- The rule for calculating the announced number -/
def announcementRule (c : Circle) (i : Fin 12) : ℚ :=
  (c (i - 1)).picked / 2 + (c (i + 1)).picked / 2 + 3

theorem original_number_equals_announced (c : Circle) (i : Fin 12) :
  (c i).picked = 8 ∧ (c i).announced = 11 →
  (c i).picked = 11 :=
by
  sorry

#check original_number_equals_announced

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_equals_announced_l1334_133412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_m_range_l1334_133414

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 1 else m*x + m - 1

theorem monotonic_f_m_range (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) → 0 < m ∧ m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_m_range_l1334_133414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_even_implies_m_equals_negative_one_l1334_133464

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(1 - m)

theorem power_function_even_implies_m_equals_negative_one (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_even_implies_m_equals_negative_one_l1334_133464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_l1334_133417

noncomputable def sample : List ℝ := [-2, -1, 0, 3, 5]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (λ x => (x - mean xs) ^ 2)).sum / xs.length

theorem variance_of_sample : variance sample = 34/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_l1334_133417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_minimum_l1334_133437

-- Define the inequality function
noncomputable def f (m n a b : ℝ) : ℝ := (a^2 / m) + (b^2 / n) - ((a + b)^2 / (m + n))

-- Define the function to be minimized
noncomputable def g (x : ℝ) : ℝ := 1 / x + 4 / (1 - x)

-- Theorem statement
theorem inequality_and_minimum :
  (∀ (m n a b : ℝ), m > 0 → n > 0 → f m n a b ≥ 0) ∧
  (∃ (x : ℝ), x > 0 ∧ x < 1 ∧ ∀ (y : ℝ), y > 0 → y < 1 → g x ≤ g y ∧ g x = 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_minimum_l1334_133437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1334_133421

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (4, 7)
def C : ℝ × ℝ := (-4, 3)

-- Define the area of the triangle
def triangle_area : ℝ := 20

-- Define the center and radius of the circumcircle
def circle_center : ℝ × ℝ := (0, 5)
noncomputable def circle_radius : ℝ := 2 * Real.sqrt 5

-- Theorem statement
theorem triangle_properties :
  let area := triangle_area
  let center := circle_center
  let radius := circle_radius
  (area = 20) ∧
  (∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔
    (x^2 + (y - 5)^2 = 20)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1334_133421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_in_second_quadrant_l1334_133496

theorem trig_values_in_second_quadrant (θ : ℝ) 
  (h1 : Real.cos (θ - π/4) = 1/3) 
  (h2 : π/2 < θ ∧ θ < π) : 
  Real.sin θ = Real.sqrt (23/72) ∧ Real.tan θ = -Real.sqrt (7/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_in_second_quadrant_l1334_133496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_average_l1334_133439

theorem consecutive_numbers_average (a : ℤ) : 
  (a + 6 = (3/2) * a) → 
  ((a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7 = 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_average_l1334_133439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_time_decrease_l1334_133472

/-- Represents the rate of decrease in homework time per semester -/
noncomputable def rate_of_decrease : ℝ := Real.sqrt (1 - 70 / 100)

/-- Theorem stating the relationship between initial time, final time, and rate of decrease -/
theorem homework_time_decrease (initial_time final_time : ℝ) 
  (h1 : initial_time = 100) 
  (h2 : final_time = 70) : 
  initial_time * (1 - rate_of_decrease)^2 = final_time := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_time_decrease_l1334_133472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_and_circle_C_properties_l1334_133403

noncomputable section

-- Define the polar coordinates of points A and B
def point_A : ℝ × ℝ := (2, Real.pi / 2)
def point_B : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

-- Define the parametric equation of circle C
def circle_C (θ : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the intersection condition
def intersects (line : (ℝ → ℝ → Prop)) (circle : ℝ → ℝ × ℝ) : Prop :=
  ∃ θ, line (circle θ).1 (circle θ).2

-- Theorem statement
theorem line_AB_and_circle_C_properties :
  (∀ x y, line_AB x y ↔ x + y - 2 = 0) ∧
  intersects line_AB circle_C := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_and_circle_C_properties_l1334_133403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_β_value_l1334_133484

noncomputable def angle_α : ℝ := Real.arctan (-4/3)

theorem cos_β_value :
  let P : ℝ × ℝ := (3, -4)
  let β : ℝ := angle_α + π
  Real.cos β = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_β_value_l1334_133484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_equals_fraction_l1334_133429

/-- The decimal representation of the number -/
noncomputable def decimal : ℚ := 20 + 396 / 999

/-- The fraction representation of the number -/
def fraction : ℚ := 20376 / 999

/-- Theorem stating that the decimal representation equals the fraction representation -/
theorem decimal_equals_fraction : decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_equals_fraction_l1334_133429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_slope_product_l1334_133448

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: For a hyperbola with eccentricity √3, the product of slopes of AM and AN is -2 -/
theorem hyperbola_slope_product (h : Hyperbola) 
  (h_ecc : eccentricity h = Real.sqrt 3) 
  (M N : ℝ × ℝ) 
  (h_MN : ∃ (k : ℝ), M.2 = k ∧ N.2 = k) 
  (h_on_hyperbola : 
    M.1^2 / h.a^2 - M.2^2 / h.b^2 = 1 ∧ 
    N.1^2 / h.a^2 - N.2^2 / h.b^2 = 1) :
  let A : ℝ × ℝ := (-h.a, 0)
  let slope_AM := (M.2 - A.2) / (M.1 - A.1)
  let slope_AN := (N.2 - A.2) / (N.1 - A.1)
  slope_AM * slope_AN = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_slope_product_l1334_133448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_m_range_l1334_133477

noncomputable def f (x : ℝ) := |2*x - 1|

theorem inequality_solution_and_m_range :
  (∃ S : Set ℝ, S = {x | f x - f (x + 1) ≤ 1} ∧ S = Set.Ici (-1/4)) ∧
  (∃ R : Set ℝ, R = {m | ∃ x, f x < m - f (x + 1)} ∧ R = Set.Ioi 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_m_range_l1334_133477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1334_133436

-- Define the points
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 9)
def C : ℝ × ℝ := (6, 6)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the perimeter function
noncomputable def perimeter (p q r : ℝ × ℝ) : ℝ :=
  distance p q + distance q r + distance r p

-- Theorem statement
theorem triangle_perimeter : perimeter A B C = 16 := by
  -- Unfold definitions
  unfold perimeter
  unfold distance
  unfold A B C
  -- Simplify the expression
  simp
  -- The proof is incomplete, so we use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1334_133436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_odd_implies_all_odd_l1334_133481

theorem product_odd_implies_all_odd (a b c d e f g : ℕ) :
  Odd (a * b * c * d * e * f * g) → Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧ Odd g :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_odd_implies_all_odd_l1334_133481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trees_formula_total_trees_when_A_12_l1334_133468

/-- Represents the number of trees planted by student B -/
def x : ℝ := Real.mk 0  -- We need to define x as a real number

/-- Represents the number of trees planted by student A -/
def trees_A (x : ℝ) : ℝ := 1.2 * x

/-- Represents the number of trees planted by student C -/
def trees_C (x : ℝ) : ℝ := trees_A x - 2

/-- Represents the total number of trees planted by A, B, and C -/
def total_trees (x : ℝ) : ℝ := x + trees_A x + trees_C x

/-- Proves that the total number of trees planted is 3.4x - 2 -/
theorem total_trees_formula (x : ℝ) : total_trees x = 3.4 * x - 2 := by
  sorry

/-- Proves that when A plants 12 trees, the total number of trees planted is 32 -/
theorem total_trees_when_A_12 (x : ℝ) : trees_A x = 12 → total_trees x = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trees_formula_total_trees_when_A_12_l1334_133468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_expansion_150th_digit_l1334_133405

/-- The decimal expansion of 5/37 -/
def decimal_expansion : ℚ := 5 / 37

/-- The length of the repeating block in the decimal expansion of 5/37 -/
def repeat_length : ℕ := 3

/-- The 150th digit after the decimal point in the decimal expansion of 5/37 -/
def digit_150 : ℕ := 5

/-- Theorem: The 150th digit after the decimal point in the decimal expansion of 5/37 is 5 -/
theorem decimal_expansion_150th_digit :
  (decimal_expansion - decimal_expansion.floor) * 10^150 % 1 * 10 = digit_150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_expansion_150th_digit_l1334_133405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_five_million_scientific_notation_l1334_133440

/-- Scientific notation representation -/
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

/-- Predicate to check if a number is in valid scientific notation -/
def is_valid_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10

/-- Theorem: 25,000,000 is equal to 2.5 × 10^7 in scientific notation -/
theorem twenty_five_million_scientific_notation :
  (25000000 : ℝ) = scientific_notation 2.5 7 ∧
  is_valid_scientific_notation 2.5 7 :=
by
  sorry

#check twenty_five_million_scientific_notation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_five_million_scientific_notation_l1334_133440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_304201_is_prime_l1334_133495

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def candidate_numbers : List ℕ := [304201, 304202, 304204, 304206, 304208]

theorem only_304201_is_prime :
  ∃! n, n ∈ candidate_numbers ∧ is_prime n ∧ n = 304201 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_304201_is_prime_l1334_133495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l1334_133438

-- Define the complex number z
noncomputable def z : ℂ := (2 + Complex.I)^2 / (1 - Complex.I)

-- Theorem stating that z is in the second quadrant
theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l1334_133438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_560000_l1334_133489

noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

theorem scientific_notation_560000 :
  ∃ (a : ℝ) (n : ℤ), 
    560000 = scientific_notation a n ∧ 
    1 ≤ |a| ∧ 
    |a| < 10 ∧
    a = 5.6 ∧
    n = 5 := by
  use 5.6, 5
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_560000_l1334_133489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_dependency_l1334_133433

/-- The volume of a sphere as a function of its radius -/
noncomputable def sphere_volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

/-- Theorem stating that the volume of a sphere is a function of its radius,
    and that different radii yield different volumes -/
theorem sphere_volume_dependency :
  ∃ (f : ℝ → ℝ), 
    (∀ R, f R = sphere_volume R) ∧ 
    (∀ R₁ R₂, R₁ ≠ R₂ → f R₁ ≠ f R₂) := by
  -- We use the sphere_volume function as our f
  use sphere_volume
  constructor
  · -- First part: f R = sphere_volume R for all R
    intro R
    rfl
  · -- Second part: if R₁ ≠ R₂, then f R₁ ≠ f R₂
    sorry -- The actual proof would go here, but we use sorry to skip it for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_dependency_l1334_133433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fahrenheit_from_celsius_l1334_133447

/-- Given the relationship between C and F, prove the value of F when C is 26 -/
theorem fahrenheit_from_celsius (C F : ℚ) : 
  C = 26 → C = (7/13) * (F - 40) → F = 88 + 2/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fahrenheit_from_celsius_l1334_133447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1334_133425

-- Define the functions
noncomputable def f (x : ℝ) := Real.log (-x^2 + 5*x - 4)
noncomputable def g (x : ℝ) := 3 / (x + 1)

-- Define the sets A and B
def A : Set ℝ := {x | -x^2 + 5*x - 4 > 0}
def B (m : ℝ) : Set ℝ := {y | ∃ x ∈ Set.Ioo 0 m, y = g x}

-- Theorem for part (1)
theorem part_one : A ∪ B 1 = Set.Ioo 1 4 := by sorry

-- Theorem for part (2)
theorem part_two : 
  (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m) → m ∈ Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1334_133425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divisible_by_2005_l1334_133474

theorem smallest_k_divisible_by_2005 :
  ∃ k : ℕ, k = 401 ∧
  (∀ n : ℕ, 0 < n → n < k → ¬(2005 ∣ n * Nat.factorial n)) ∧
  (2005 ∣ k * Nat.factorial k) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divisible_by_2005_l1334_133474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kangaroo_reach_kangaroo_cannot_reach_l1334_133459

/-- A kangaroo can reach at least 1000 units away from the origin if it starts in Region I -/
theorem kangaroo_reach (x y : ℝ) : 
  x + y > 6 → ∃ (n : ℕ), (x + n)^2 + (y + 2*n)^2 ≥ 1000^2 :=
by
  sorry

/-- The kangaroo cannot reach 1000 units away if it starts in Region II -/
theorem kangaroo_cannot_reach (x y : ℝ) :
  x + y < 5 → ∀ (n : ℕ), (x + n)^2 + (y - n)^2 < 1000^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kangaroo_reach_kangaroo_cannot_reach_l1334_133459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1334_133420

/-- Calculates the length of a train given its speed, the platform length, and the time taken to pass the platform. -/
noncomputable def train_length (train_speed : ℝ) (platform_length : ℝ) (time_to_pass : ℝ) : ℝ :=
  train_speed * time_to_pass / 3600 * 1000 - platform_length

/-- Theorem stating that a train with speed 45 km/hr passing a 340 m long platform in 56 seconds has a length of 360 m. -/
theorem train_length_calculation :
  train_length 45 340 56 = 360 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1334_133420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_and_amplitude_l1334_133499

noncomputable def f (x : ℝ) := 5 * Real.cos (x + Real.pi/4) + 2

theorem phase_shift_and_amplitude :
  (∃ (p : ℝ), ∀ (x : ℝ), f (x + p) = 5 * Real.cos x + 2) ∧
  (∃ (A : ℝ), A > 0 ∧ ∀ (x : ℝ), |f x - 2| ≤ A ∧ ∃ (y : ℝ), |f y - 2| = A) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_and_amplitude_l1334_133499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1334_133423

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 → f x ≤ 2) ∧
  (∀ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 → f x ≥ 0) ∧
  (∃ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 ∧ f x = 2) ∧
  (∃ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 ∧ f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1334_133423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l1334_133409

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + 2*y) = 6*x + f (f y - x)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = 2*x + c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l1334_133409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_is_log_base_two_l1334_133422

noncomputable def exp_a (a : ℝ) (x : ℝ) : ℝ := a^x

def is_inverse_of_exp_a (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (exp_a a x) = x ∧ exp_a a (f x) = x

theorem inverse_function_is_log_base_two (f : ℝ → ℝ) (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : is_inverse_of_exp_a f a) 
  (h4 : f 2 = 1) : 
  f = λ x ↦ Real.log x / Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_is_log_base_two_l1334_133422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_z_plus_2i_l1334_133434

theorem min_abs_z_plus_2i (z : ℂ) (h : Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*Complex.I))) :
  Complex.abs (z + 2*Complex.I) ≥ 7/2 ∧ ∃ w : ℂ, Complex.abs (w^2 + 9) = Complex.abs (w * (w + 3*Complex.I)) ∧ Complex.abs (w + 2*Complex.I) = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_z_plus_2i_l1334_133434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_bought_four_peanut_packs_l1334_133473

/-- Represents the purchase scenario of Linda --/
structure Purchase where
  coloring_book_price : ℚ
  coloring_book_quantity : ℕ
  peanut_pack_price : ℚ
  stuffed_animal_price : ℚ
  total_paid : ℚ

/-- Calculates the number of peanut packs bought --/
def peanut_packs_bought (p : Purchase) : ℚ :=
  (p.total_paid - (p.coloring_book_price * p.coloring_book_quantity + p.stuffed_animal_price)) / p.peanut_pack_price

/-- Theorem stating that Linda bought 4 packs of peanuts --/
theorem linda_bought_four_peanut_packs :
  let p : Purchase := {
    coloring_book_price := 4,
    coloring_book_quantity := 2,
    peanut_pack_price := 3/2,
    stuffed_animal_price := 11,
    total_paid := 25
  }
  peanut_packs_bought p = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_bought_four_peanut_packs_l1334_133473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_power_function_and_decreasing_l1334_133411

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, x > 0 → f x = a * x^b

def is_monotonically_decreasing (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ x y, x ∈ interval → y ∈ interval → x < y → f x > f y

theorem unique_m_for_power_function_and_decreasing :
  ∃! m : ℝ,
    is_power_function (fun x ↦ (m^2 - m - 1) * x^(m^2 - 2*m - 3)) ∧
    is_monotonically_decreasing (fun x ↦ (m^2 - m - 1) * x^(m^2 - 2*m - 3)) (Set.Ioi 0) ∧
    m = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_power_function_and_decreasing_l1334_133411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_species_for_2021_birds_l1334_133483

/-- Represents a bird in the row -/
structure Bird where
  species : Nat

/-- The arrangement of birds in a row -/
def BirdArrangement := List Bird

/-- Checks if the arrangement satisfies the even-spacing condition -/
def satisfiesEvenSpacing (arrangement : BirdArrangement) : Prop :=
  ∀ i j, i < j → arrangement.get? i = arrangement.get? j →
    (j - i - 1) % 2 = 0

/-- The theorem stating the minimum number of species required -/
theorem min_species_for_2021_birds :
  ∀ (arrangement : BirdArrangement),
    arrangement.length = 2021 →
    satisfiesEvenSpacing arrangement →
    (arrangement.map Bird.species).toFinset.card ≥ 1011 := by
  sorry

#check min_species_for_2021_birds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_species_for_2021_birds_l1334_133483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_variance_product_l1334_133415

noncomputable def population (c a b : ℝ) : List ℝ := [c, 3, 3, 8, a, b, 12, 13.7, 18.3, 20]

theorem minimize_variance_product (c a b : ℝ) :
  (population c a b).length = 10 ∧
  List.Sorted (· ≤ ·) (population c a b) ∧
  ((population c a b).get? 4).isSome ∧
  ((population c a b).get? 5).isSome ∧
  (((population c a b).get? 4).getD 0 + ((population c a b).get? 5).getD 0) / 2 = 10 ∧
  (population c a b).sum / (population c a b).length = 10 ∧
  (∀ x y : ℝ, (x - 10)^2 + (y - 10)^2 ≥ (a - 10)^2 + (b - 10)^2 → x + y = 20) →
  c * a * b = 200 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_variance_product_l1334_133415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1334_133463

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real × Real
  b : Real × Real
  c : Real × Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = (Real.sqrt 3, 1) ∧
  t.b = (Real.sin t.A, Real.cos t.A) ∧
  Real.cos (60 * Real.pi / 180) = (t.a.1 * t.b.1 + t.a.2 * t.b.2) / (Real.sqrt (t.a.1^2 + t.a.2^2) * Real.sqrt (t.b.1^2 + t.b.2^2)) ∧
  Real.sin (t.B - t.C) = 2 * Real.cos t.B * Real.sin t.C

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) : 
  t.A = 2 * Real.pi / 3 ∧ 
  (Real.sin t.B / Real.sin t.C) = (3 * Real.sqrt 13 - 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1334_133463
