import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_slope_sum_l1135_113564

/-- Two circles that intersect at (15,10) and have specific properties -/
structure CirclePair where
  C₁ : Set (ℝ × ℝ)
  C₂ : Set (ℝ × ℝ)
  center₁ : ℝ × ℝ
  center₂ : ℝ × ℝ
  intersection : (15, 10) ∈ C₁ ∩ C₂
  radii_product : ∃ r₁ r₂ : ℝ, r₁ * r₂ = 130 ∧ 
    (∀ x y, (x, y) ∈ C₁ → (x - center₁.1)^2 + (y - center₁.2)^2 = r₁^2) ∧
    (∀ x y, (x, y) ∈ C₂ → (x - center₂.1)^2 + (y - center₂.2)^2 = r₂^2)
  x_axis_tangent : ∀ y, (0, y) ∉ C₁ ∧ (0, y) ∉ C₂
  n_line_tangent : ∃ n : ℝ, n > 0 ∧ ∀ x, (x, n*x) ∉ C₁ ∧ (x, n*x) ∉ C₂

/-- The slope n of the tangent line in the specific form -/
noncomputable def slope_form (p q r : ℕ) : ℝ := (p : ℝ) * Real.sqrt q / r

/-- The property that q is not divisible by the square of any prime -/
def squarefree (q : ℕ) : Prop := ∀ p : ℕ, Nat.Prime p → (p^2 ∣ q → False)

/-- The main theorem to prove -/
theorem circle_tangent_slope_sum (cp : CirclePair) 
  (h : ∃ p q r : ℕ, p > 0 ∧ q > 0 ∧ r > 0 ∧ 
       squarefree q ∧ 
       Nat.Coprime p r ∧
       ∃ n : ℝ, n > 0 ∧ ∀ x, (x, n*x) ∉ cp.C₁ ∧ (x, n*x) ∉ cp.C₂ ∧
       n = slope_form p q r) : 
  ∃ p q r : ℕ, p + q + r = 11 ∧ 
    p > 0 ∧ q > 0 ∧ r > 0 ∧ 
    squarefree q ∧ 
    Nat.Coprime p r ∧
    ∃ n : ℝ, n > 0 ∧ ∀ x, (x, n*x) ∉ cp.C₁ ∧ (x, n*x) ∉ cp.C₂ ∧
    n = slope_form p q r :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_slope_sum_l1135_113564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1135_113522

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (t : Triangle) : Prop :=
  t.c * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.C ∧
  t.a^2 - t.c^2 = 2 * t.b^2

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  1/2 * t.a * t.b * Real.sin t.C

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) 
  (h1 : satisfiesConditions t) 
  (h2 : area t = 21 * Real.sqrt 3) : 
  t.C = π/3 ∧ t.b = 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1135_113522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_with_horizontal_asymptote_l1135_113588

/-- A rational function with denominator 3x^6 - 2x^3 + 5 -/
noncomputable def rationalFunction (p : ℝ → ℝ) (x : ℝ) : ℝ :=
  p x / (3 * x^6 - 2 * x^3 + 5)

/-- The degree of a polynomial -/
noncomputable def degree (p : ℝ → ℝ) : ℕ := sorry

/-- A function has a horizontal asymptote -/
def hasHorizontalAsymptote (f : ℝ → ℝ) : Prop := sorry

theorem max_degree_with_horizontal_asymptote 
  (p : ℝ → ℝ) 
  (h : hasHorizontalAsymptote (rationalFunction p)) : 
  degree p ≤ 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_with_horizontal_asymptote_l1135_113588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_intersections_concyclic_l1135_113557

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the main circle
noncomputable def main_circle : Circle := ⟨(0, 0), 1⟩

-- Define the four points on the main circle
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry
noncomputable def C : ℝ × ℝ := sorry
noncomputable def D : ℝ × ℝ := sorry

-- Define the circles through adjacent pairs
noncomputable def circle_AB : Circle := sorry
noncomputable def circle_BC : Circle := sorry
noncomputable def circle_CD : Circle := sorry
noncomputable def circle_DA : Circle := sorry

-- Define the second intersection points
noncomputable def A₁ : ℝ × ℝ := sorry
noncomputable def B₁ : ℝ × ℝ := sorry
noncomputable def C₁ : ℝ × ℝ := sorry
noncomputable def D₁ : ℝ × ℝ := sorry

-- Theorem statement
theorem second_intersections_concyclic :
  ∃ (c : Circle), 
    PointOnCircle c A₁ ∧ 
    PointOnCircle c B₁ ∧ 
    PointOnCircle c C₁ ∧ 
    PointOnCircle c D₁ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_intersections_concyclic_l1135_113557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_points_with_min_distance_l1135_113506

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Definition of the rectangle -/
def rectangle : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 4 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- Theorem: It's impossible to place three points in the rectangle
    with minimum distance 2.5 between any two points -/
theorem no_three_points_with_min_distance
  (p1 p2 p3 : Point)
  (h1 : p1 ∈ rectangle)
  (h2 : p2 ∈ rectangle)
  (h3 : p3 ∈ rectangle) :
  ¬(distance p1 p2 ≥ 2.5 ∧ distance p1 p3 ≥ 2.5 ∧ distance p2 p3 ≥ 2.5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_points_with_min_distance_l1135_113506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l1135_113526

theorem inequality_problem (x m : ℝ) : 
  (∀ x, (Real.rpow 2 (x^2 - 4*x + 3) < 1 ∧ 2/(4-x) ≥ 1) → 2*x^2 - 9*x + m < 0) ↔ m ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l1135_113526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_is_22_l1135_113570

/-- Represents the car rental problem with given parameters -/
structure CarRentalProblem where
  trip_distance : ℚ
  first_option_cost : ℚ
  second_option_cost : ℚ
  gasoline_efficiency : ℚ
  gasoline_cost_per_liter : ℚ

/-- Calculates the savings when choosing the first option over the second option -/
def calculate_savings (problem : CarRentalProblem) : ℚ :=
  let total_distance := 2 * problem.trip_distance
  let gasoline_needed := total_distance / problem.gasoline_efficiency
  let gasoline_cost := gasoline_needed * problem.gasoline_cost_per_liter
  let first_option_total := problem.first_option_cost + gasoline_cost
  problem.second_option_cost - first_option_total

/-- Theorem stating that the savings for the given problem is $22 -/
theorem savings_is_22 (problem : CarRentalProblem) 
  (h1 : problem.trip_distance = 150)
  (h2 : problem.first_option_cost = 50)
  (h3 : problem.second_option_cost = 90)
  (h4 : problem.gasoline_efficiency = 15)
  (h5 : problem.gasoline_cost_per_liter = 9/10) :
  calculate_savings problem = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_is_22_l1135_113570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_c_value_l1135_113555

theorem factor_implies_c_value (c : ℝ) : 
  (∀ x : ℝ, (2 * x + 7) * (4 * x^3 + a * x^2 + b * x + d) = 8 * x^4 + 25 * x^3 + c * x^2 + 2 * x + 49) →
  c = -13.93 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_c_value_l1135_113555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1135_113539

def angle_brackets (x : ℕ) : ℕ :=
  if x ≤ 1 then 0
  else List.sum (List.range (2 * x) |>.map (λ i => 2 * x - i))

theorem solve_equation (n : ℕ) : (angle_brackets 2) * n = 360 → n = 36 := by
  intro h
  have h1 : angle_brackets 2 = 10 := by
    rfl
  rw [h1] at h
  exact Nat.eq_of_mul_eq_mul_left (by norm_num : 10 > 0) h

#eval angle_brackets 2  -- This will output 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1135_113539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_l1135_113587

/-- Represents the distance traveled by a truck in kilometers -/
noncomputable def distance_traveled (b t : ℝ) : ℝ := 25 * b / (1000 * t)

/-- Theorem stating the distance traveled by the truck in 5 minutes -/
theorem truck_distance (b t : ℝ) (hb : b > 0) (ht : t > 0) :
  let meters_per_2t_seconds := b / 6
  let seconds_in_5_minutes := 5 * 60
  let meters_in_yard := 0.9144
  distance_traveled b t = (meters_per_2t_seconds * seconds_in_5_minutes) / (2 * t * 1000) := by
  sorry

#check truck_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_l1135_113587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1135_113559

/-- Profit function for a company's product promotion -/
noncomputable def profit_function (x a : ℝ) : ℝ := 26 - 4 / (x + 1) - x

/-- Theorem stating the maximum profit and corresponding promotional cost -/
theorem max_profit (a : ℝ) (ha : a > 0) :
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ a ∧
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ a → profit_function x a ≥ profit_function y a) ∧
  (if a ≥ 1 then
    (∃ (x : ℝ), x = 1 ∧ profit_function x a = 23)
  else
    (∃ (x : ℝ), x = a ∧ profit_function x a = 26 - 4 / (a + 1) - a)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1135_113559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sequence_theorem_l1135_113500

/-- A parabola with equation y = mx² -/
structure Parabola where
  m : ℝ

/-- Generate the next parabola in the sequence -/
def nextParabola (P : Parabola) : Parabola :=
  ⟨3 * P.m⟩

/-- The n-th parabola in the sequence -/
def nthParabola (P₀ : Parabola) : ℕ → Parabola
  | 0 => P₀
  | n + 1 => nextParabola (nthParabola P₀ n)

/-- The equation of the n-th parabola -/
def nthParabolaEq (P₀ : Parabola) (n : ℕ) (x y : ℝ) : Prop :=
  y = (3^n * P₀.m * x^2) + (1 / (4 * P₀.m)) * (1 - (1/3)^n)

theorem parabola_sequence_theorem (P₀ : Parabola) (n : ℕ) (x y : ℝ) :
  nthParabolaEq P₀ n x y ↔ 
  y = ((nthParabola P₀ n).m * x^2) + (1 / (4 * P₀.m)) * (1 - (1/3)^n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sequence_theorem_l1135_113500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_theorem_l1135_113504

-- Define the triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3
  ab_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4
  bc_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 5

-- Define the inscribed circle
structure InscribedCircle (t : RightTriangle) where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_AC : ∃ D : ℝ × ℝ, (D.1 - t.A.1) * (t.C.2 - t.A.2) = (D.2 - t.A.2) * (t.C.1 - t.A.1) ∧
                   Real.sqrt ((D.1 - center.1)^2 + (D.2 - center.2)^2) = radius
  tangent_to_AB : ∃ E : ℝ × ℝ, (E.1 - t.A.1) * (t.B.2 - t.A.2) = (E.2 - t.A.2) * (t.B.1 - t.A.1) ∧
                   Real.sqrt ((E.1 - center.1)^2 + (E.2 - center.2)^2) = radius
  tangent_to_BC : ∃ F : ℝ × ℝ, (F.1 - t.B.1) * (t.C.2 - t.B.2) = (F.2 - t.B.2) * (t.C.1 - t.B.1) ∧
                   Real.sqrt ((F.1 - center.1)^2 + (F.2 - center.2)^2) = radius

-- Define the extension of DE to G
noncomputable def extendDE (t : RightTriangle) (c : InscribedCircle t) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem inscribed_circle_theorem (t : RightTriangle) (c : InscribedCircle t) :
  let G := extendDE t c
  Real.sqrt ((G.1 - t.B.1)^2 + (G.2 - t.B.2)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_theorem_l1135_113504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_faces_at_vertex_l1135_113545

/-- A face of a polyhedron -/
structure Face where

/-- A vertex of a polyhedron -/
structure Vertex where

/-- An icosahedron is a polyhedron with 20 faces, each of which is an equilateral triangle. -/
structure Icosahedron where
  faces : Finset Face
  is_icosahedron : faces.card = 20

/-- The number of faces meeting at a vertex of a polyhedron -/
def faces_at_vertex (v : Vertex) (i : Icosahedron) : ℕ :=
  sorry

/-- In an icosahedron, 5 faces meet at each vertex -/
theorem icosahedron_faces_at_vertex (i : Icosahedron) (v : Vertex) :
  faces_at_vertex v i = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_faces_at_vertex_l1135_113545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_then_blue_probability_l1135_113519

/-- The probability of drawing a green marble first and a blue marble second from a bag
    containing 4 blue marbles and 6 green marbles, without replacement. -/
theorem green_then_blue_probability : 
  (6 : ℚ) / 10 * (4 : ℚ) / 9 = 4 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_then_blue_probability_l1135_113519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_profit_theorem_l1135_113548

/-- Calculates the profit percentage without discount given the discount percentage and profit percentage with discount -/
noncomputable def profit_without_discount (discount_percent : ℝ) (profit_with_discount_percent : ℝ) : ℝ :=
  let cost_price := 100
  let selling_price_with_discount := cost_price * (1 + profit_with_discount_percent / 100)
  let marked_price := selling_price_with_discount / (1 - discount_percent / 100)
  (marked_price - cost_price) / cost_price * 100

/-- Theorem stating that a 10% discount with 25% profit results in approximately 38.89% profit without discount -/
theorem discount_profit_theorem :
  ∃ ε > 0, |profit_without_discount 10 25 - 38.89| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_profit_theorem_l1135_113548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sums_product_l1135_113510

theorem consecutive_sums_product (n : ℕ) : 
  (3 * n + 3) * (3 * n + 12) ≠ 1111111111 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sums_product_l1135_113510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_implies_a_range_l1135_113586

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 2 * a / x

theorem function_monotonicity_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) < 0) →
  0 < a ∧ a ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_implies_a_range_l1135_113586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_circular_cone_central_angle_l1135_113583

/-- A right circular cone is a cone with a circular base whose axis section is an isosceles right triangle. -/
structure RightCircularCone where
  base_radius : ℝ
  slant_height : ℝ
  axis_section_is_isosceles_right : slant_height = Real.sqrt 2 * base_radius

/-- The radian measure of the central angle of the unfolded side of a right circular cone. -/
noncomputable def central_angle (cone : RightCircularCone) : ℝ :=
  2 * Real.pi * cone.base_radius / cone.slant_height

/-- Theorem: The central angle of the unfolded side of a right circular cone is √2π. -/
theorem right_circular_cone_central_angle (cone : RightCircularCone) :
  central_angle cone = Real.sqrt 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_circular_cone_central_angle_l1135_113583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformation_effects_l1135_113595

-- Define the original triangle
def triangle_PQR (PQ PR QR : ℝ) : Prop :=
  PQ = 15 ∧ PR = 9 ∧ QR = 12

-- Define the transformation
def transform_triangle (PQ PR QR : ℝ) : ℝ × ℝ × ℝ :=
  (1.5 * PQ, PR, 2 * QR)

-- Define the area function using Heron's formula
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the median function
noncomputable def median_p (a b c : ℝ) : ℝ :=
  Real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)

-- State the theorem
theorem triangle_transformation_effects (PQ PR QR : ℝ) :
  triangle_PQR PQ PR QR →
  let (PQ', PR', QR') := transform_triangle PQ PR QR
  triangle_area PQ' PR' QR' > 2 * triangle_area PQ PR QR ∧
  median_p PQ' PR' QR' ≠ median_p PQ PR QR :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformation_effects_l1135_113595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_tan_cos_sin_identity_l1135_113576

-- Define the third quadrant
def third_quadrant (α : Real) : Prop := Real.pi < α ∧ α < 3 * Real.pi / 2

-- Theorem 1
theorem simplify_expression (α : Real) (h : third_quadrant α) :
  Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) - Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) = -2 * Real.tan α :=
by sorry

-- Theorem 2
theorem tan_cos_sin_identity (θ : Real) :
  (1 - Real.tan θ ^ 2) / (1 + Real.tan θ ^ 2) = Real.cos θ ^ 2 - Real.sin θ ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_tan_cos_sin_identity_l1135_113576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_l1135_113542

/-- The length of a boat given specific conditions -/
theorem boat_length : 
  ∀ (breadth : ℝ) (sink_depth : ℝ) (man_mass : ℝ) (water_density : ℝ) (gravity : ℝ),
  breadth = 2 →
  sink_depth = 0.01 →
  man_mass = 80 →
  water_density = 1000 →
  gravity = 9.81 →
  ∃ (length : ℝ), length = 4 := by
  intros breadth sink_depth man_mass water_density gravity h1 h2 h3 h4 h5
  -- The proof would go here, but we'll use sorry for now
  sorry

#check boat_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_l1135_113542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_c_value_l1135_113578

/-- A parabola with equation y = x^2 + bx + c -/
def Parabola (b c : ℝ) : ℝ → ℝ := λ x ↦ x^2 + b*x + c

theorem parabola_c_value :
  ∀ b c : ℝ, 
  (Parabola b c 1 = 4 ∧ Parabola b c 5 = 4) → c = 9 :=
by
  intros b c h
  sorry

#check parabola_c_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_c_value_l1135_113578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l1135_113513

noncomputable def a : ℝ := (6 : ℝ) ^ (7/10)
noncomputable def b : ℝ := (7/10 : ℝ) ^ 6
noncomputable def c : ℝ := Real.log 6 / Real.log (7/10)

theorem abc_relationship : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l1135_113513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_cocktail_can_be_stronger_l1135_113574

-- Define the strength of alcoholic beverages
structure Beverage where
  strength : ℝ
  strength_positive : strength > 0

-- Define the cocktail as a mixture of two beverages
structure Cocktail where
  beverage1 : Beverage
  beverage2 : Beverage
  amount1 : ℝ
  amount2 : ℝ
  amounts_positive : amount1 > 0 ∧ amount2 > 0

-- Calculate the strength of a cocktail
noncomputable def cocktail_strength (c : Cocktail) : ℝ :=
  (c.amount1 * c.beverage1.strength + c.amount2 * c.beverage2.strength) / (c.amount1 + c.amount2)

-- Theorem statement
theorem ivan_cocktail_can_be_stronger 
  (whiskey vodka liqueur beer : Beverage)
  (whiskey_stronger : whiskey.strength > vodka.strength)
  (liqueur_stronger : liqueur.strength > beer.strength) :
  ∃ (john_cocktail ivan_cocktail : Cocktail),
    john_cocktail.beverage1 = whiskey ∧
    john_cocktail.beverage2 = liqueur ∧
    ivan_cocktail.beverage1 = vodka ∧
    ivan_cocktail.beverage2 = beer ∧
    cocktail_strength ivan_cocktail > cocktail_strength john_cocktail := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_cocktail_can_be_stronger_l1135_113574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sin_plus_sqrt3_cos_l1135_113515

theorem period_of_sin_plus_sqrt3_cos : 
  ∃ T : ℝ, T > 0 ∧ 
    (∀ x : ℝ, Real.sin (x + T) + Real.sqrt 3 * Real.cos (x + T) = Real.sin x + Real.sqrt 3 * Real.cos x) ∧
    (∀ T' : ℝ, 0 < T' ∧ T' < T → ∃ x : ℝ, Real.sin (x + T') + Real.sqrt 3 * Real.cos (x + T') ≠ Real.sin x + Real.sqrt 3 * Real.cos x) ∧
    T = 2 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sin_plus_sqrt3_cos_l1135_113515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_prime_in_sequence_l1135_113590

def Q : ℕ := (Finset.filter (λ p => Nat.Prime p ∧ p ≤ 53 ∧ p % 2 = 1) (Finset.range 54)).prod id

theorem at_most_one_prime_in_sequence :
  ∃ (count : ℕ), count ≤ 1 ∧
  count = (Finset.filter (λ n => Nat.Prime (Q + n)) (Finset.range 45 \ {0, 1})).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_prime_in_sequence_l1135_113590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_l1135_113505

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem inequality_system_solution :
  ∀ x : ℝ, 
    (log_base (3/5) ((x^2 + x - 6) / (x^2 - 4)) < 1 ∧
     Real.sqrt (5 - x^2) > x - 1) ↔
    (-2 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_l1135_113505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l1135_113562

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_positive : 0 < r

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The length of the chord formed by the intersection of a circle and a line -/
noncomputable def chord_length (c : Circle) (m : ℝ) : ℝ :=
  let d := |m * c.h - c.k + 0| / Real.sqrt (m^2 + 1)
  2 * Real.sqrt (c.r^2 - d^2)

theorem hyperbola_circle_intersection (h : Hyperbola) (c : Circle) :
  eccentricity h = Real.sqrt 5 →
  c.h = 2 ∧ c.k = 3 ∧ c.r = 1 →
  chord_length c 2 = 4 * Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l1135_113562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_improvement_l1135_113540

noncomputable def old_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def sample_mean (data : List ℝ) : ℝ := (data.sum) / (data.length : ℝ)

noncomputable def sample_variance (data : List ℝ) : ℝ :=
  let mean := sample_mean data
  (data.map (fun x => (x - mean) ^ 2)).sum / (data.length : ℝ)

def significant_improvement (x_bar y_bar s1_sq s2_sq : ℝ) : Prop :=
  y_bar - x_bar ≥ 2 * Real.sqrt ((s1_sq + s2_sq) / 10)

theorem new_device_improvement :
  let x_bar := sample_mean old_data
  let y_bar := sample_mean new_data
  let s1_sq := sample_variance old_data
  let s2_sq := sample_variance new_data
  significant_improvement x_bar y_bar s1_sq s2_sq :=
by sorry

#check new_device_improvement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_improvement_l1135_113540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_sixth_year_l1135_113518

def calculate_students (initial : ℕ) (year : ℕ) : ℕ :=
  let even_increase := 0.5
  let odd_increase := 0.3
  let constant_decrease := 20
  let pandemic_decrease := 0.1
  let max_students := 800

  let year1 := initial
  let year2 := min max_students (year1 + (year1 * 50 / 100))
  let year3 := min max_students (year2 + (year2 * 30 / 100))
  let year4 := min max_students (year3 + (year3 * 50 / 100) - constant_decrease)
  let year5_before_pandemic := min max_students (year4 + (year4 * 30 / 100) - constant_decrease)
  let year5 := year5_before_pandemic - (year5_before_pandemic * 10 / 100)
  let year6_before_decrease := min max_students (year5 + (year5 * 30 / 100))
  year6_before_decrease - constant_decrease

theorem students_in_sixth_year (initial : ℕ) :
  initial = 200 → calculate_students initial 6 = 780 := by
  sorry

#eval calculate_students 200 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_sixth_year_l1135_113518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1135_113593

noncomputable section

open Real

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  C = 2 * A →
  Real.cos A = 3 / 4 →
  a * c * Real.cos B = 27 / 2 →
  a * Real.sin C = c * Real.sin A →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  b = 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1135_113593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juanita_loss_l1135_113592

/-- Represents the drumming contest scenario -/
structure DrummingContest where
  entryCost : ℚ
  minDrums : ℕ
  earningsPerDrum : ℚ
  drumsHit : ℕ

/-- Calculates the total loss for a contestant in the drumming contest -/
def totalLoss (contest : DrummingContest) : ℚ :=
  contest.entryCost - max 0 (contest.earningsPerDrum * (contest.drumsHit - contest.minDrums))

/-- Theorem stating that Juanita's loss in the drumming contest is $7.5 -/
theorem juanita_loss :
  let contest : DrummingContest := {
    entryCost := 10,
    minDrums := 200,
    earningsPerDrum := 1/40,  -- 0.025 as a rational number
    drumsHit := 300
  }
  totalLoss contest = 15/2 := by  -- 7.5 as a rational number
  sorry

#eval totalLoss {
  entryCost := 10,
  minDrums := 200,
  earningsPerDrum := 1/40,
  drumsHit := 300
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juanita_loss_l1135_113592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_properties_and_distances_l1135_113558

/-- Represents a polynomial term -/
structure MyTerm where
  coefficient : ℤ
  x_power : ℕ
  y_power : ℕ

/-- Represents a polynomial -/
def MyPolynomial := List MyTerm

def count_terms (p : MyPolynomial) : ℕ := p.length

def degree (p : MyPolynomial) : ℕ :=
  p.foldl (fun acc term => max acc (term.x_power + term.y_power)) 0

def constant_term (p : MyPolynomial) : ℤ :=
  match p.find? (fun term => term.x_power = 0 ∧ term.y_power = 0) with
  | some term => term.coefficient
  | none => 0

def min_sum_distances (a b c : ℤ) : ℤ :=
  min (min (2 * (b - a)) (2 * (c - b))) (b - a + c - b)

theorem polynomial_properties_and_distances (p : MyPolynomial) (a b c : ℤ) :
  p = [⟨7, 3, 2⟩, ⟨-3, 2, 1⟩, ⟨-5, 0, 0⟩] →
  count_terms p = 3 ∧
  degree p = 5 ∧
  constant_term p = -5 ∧
  a = 3 ∧ b = 5 ∧ c = -5 →
  min_sum_distances a b c = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_properties_and_distances_l1135_113558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_length_squared_l1135_113544

noncomputable def p (x : ℝ) : ℝ := -3/2 * x + 1
noncomputable def q (x : ℝ) : ℝ := 3/2 * x + 1
def r : ℝ → ℝ := Function.const ℝ 2
noncomputable def m (x : ℝ) : ℝ := min (min (p x) (q x)) (r x)

noncomputable def graph_length (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt (1 + (deriv f x)^2)

theorem graph_length_squared (s : ℝ) :
  s = graph_length m (-4) 4 →
  s^2 = 841/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_length_squared_l1135_113544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt8_properties_option_A_incorrect_option_B_correct_option_C_correct_option_D_correct_l1135_113533

-- Define √8 as noncomputable
noncomputable def sqrt8 : ℝ := Real.sqrt 8

-- Theorem stating the properties of √8
theorem sqrt8_properties :
  (sqrt8 = 2 * Real.sqrt 2) ∧
  (¬ ∃ (q : ℚ), (↑q : ℝ) = sqrt8) ∧
  (2 < sqrt8 ∧ sqrt8 < 3) ∧
  (sqrt8 ≠ -2 * Real.sqrt 2) :=
by
  sorry

-- Theorem stating that option A is incorrect
theorem option_A_incorrect : sqrt8 ≠ 2 * Real.sqrt 2 ∧ sqrt8 ≠ -2 * Real.sqrt 2 :=
by
  sorry

-- Theorem stating that option B is correct
theorem option_B_correct : ¬ ∃ (q : ℚ), (↑q : ℝ) = sqrt8 :=
by
  sorry

-- Theorem stating that option C is correct
theorem option_C_correct : 2 < sqrt8 ∧ sqrt8 < 3 :=
by
  sorry

-- Theorem stating that option D is correct
theorem option_D_correct : sqrt8 = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt8_properties_option_A_incorrect_option_B_correct_option_C_correct_option_D_correct_l1135_113533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_l1135_113572

theorem cone_height (s θ : ℝ) (h₁ : s = 3) (h₂ : θ = 2 * Real.pi / 3) : 
  Real.sqrt (s^2 - (s * θ / (2 * Real.pi))^2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_l1135_113572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_distribution_and_expectation_l1135_113523

/-- Binomial distribution probability mass function -/
def binomialPMF (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- Expected value of a binomial distribution -/
def binomialExpectedValue (n : ℕ) (p : ℝ) : ℝ := n * p

/-- Theorem for coin toss probability distribution and expected value -/
theorem coin_toss_distribution_and_expectation :
  let X : ℕ → ℝ := binomialPMF 4 (1/2)
  (X 0 = 1/16 ∧ X 1 = 1/4 ∧ X 2 = 3/8 ∧ X 3 = 1/4 ∧ X 4 = 1/16) ∧
  binomialExpectedValue 4 (1/2) = 2 := by
  sorry

#check coin_toss_distribution_and_expectation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_distribution_and_expectation_l1135_113523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_truncated_cone_theorem_l1135_113517

noncomputable def circular_truncated_cone_surface_area 
  (upper_radius lower_radius : ℝ) 
  (central_angle : ℝ) : ℝ :=
  let lateral_area := Real.pi * (upper_radius + lower_radius) * 
    ((lower_radius - upper_radius) / (central_angle / 360))
  let upper_base_area := Real.pi * upper_radius^2
  let lower_base_area := Real.pi * lower_radius^2
  lateral_area + upper_base_area + lower_base_area

theorem circular_truncated_cone_theorem :
  circular_truncated_cone_surface_area 2 4 90 = 68 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_truncated_cone_theorem_l1135_113517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_l1135_113502

open Real

-- Define the function f(x) = x³ - 1/x
noncomputable def f (x : ℝ) : ℝ := x^3 - 1/x

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 + 1/x^2

-- Theorem statement
theorem min_slope_tangent (x : ℝ) (h : x > 0) : 
  ∀ y > 0, f' y ≥ 2 * Real.sqrt 3 ∧ (∃ z > 0, f' z = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_l1135_113502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_sister_age_difference_l1135_113585

/-- Represents the age difference between a person and their sister -/
def AgeDifference := Int

theorem first_sister_age_difference 
  (person_age : ℕ) 
  (first_sister_age : ℕ) 
  (second_sister_age : ℕ) 
  (h1 : first_sister_age = person_age - 1)
  (h2 : (person_age + first_sister_age + second_sister_age) / 3 = 5) :
  (person_age - first_sister_age : Int) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_sister_age_difference_l1135_113585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_number_l1135_113501

def is_valid_two_digit_selection (n : ℕ) : Prop :=
  let digits := n.digits 10
  let two_digit_selections := 
    List.map (λ (i, j) => 
      10 * (digits.get! i) + (digits.get! j))
      [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
  two_digit_selections.toFinset = {33, 37, 38, 73, 77, 78, 83, 87}

theorem unique_five_digit_number : 
  ∃! n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ is_valid_two_digit_selection n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_number_l1135_113501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l1135_113512

/-- Given points A, B, C, D, and E in a 2D coordinate system,
    where E is the intersection of segments AB and CD,
    prove that the length of AE is approximately 9.16 units. -/
theorem length_of_AE (A B C D E : ℝ × ℝ) : 
  A = (0, 4) →
  B = (7, 0) →
  C = (5, 3) →
  D = (3, 0) →
  E.1 = (203 : ℝ) / 23 →
  E.2 = -4 * E.1 / 7 + 4 →
  abs (Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) - 9.16) < 0.01 := by
  sorry

#check length_of_AE

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l1135_113512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1135_113577

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- Theorem statement
theorem f_minimum_value :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -4/3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1135_113577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2014_odd_not_even_l1135_113573

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => fun x => 1 / x
  | n + 1 => fun x => 1 / (x + f n x)

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem f_2014_odd_not_even :
  is_odd (f 2014) ∧ ¬(is_even (f 2014)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2014_odd_not_even_l1135_113573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_journey_l1135_113584

def batch_distances : List ℝ := [5, 2, -4, -3, 10]
def fuel_consumption_rate : ℝ := 0.2

def final_position (distances : List ℝ) : ℝ :=
  distances.sum

def total_fuel_consumption (distances : List ℝ) (rate : ℝ) : ℝ :=
  (distances.map abs).sum * rate

theorem taxi_journey :
  final_position batch_distances = 10 ∧
  total_fuel_consumption batch_distances fuel_consumption_rate = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_journey_l1135_113584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_implies_a_in_interval_l1135_113568

-- Define the equation
noncomputable def f (x a : ℝ) : ℝ := x * Real.log x + (3 - a) * x + a

-- State the theorem
theorem unique_root_implies_a_in_interval :
  (∃! x : ℝ, x > 1 ∧ f x a = 0) → a ∈ Set.Ioo 5 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_implies_a_in_interval_l1135_113568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_theorem_l1135_113535

/-- Represents a trapezoid with parallel sides a and b, and height m -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  m : ℝ
  h_positive : 0 < m
  h_parallel : a ≠ b

/-- The length of the line that divides a trapezoid into two parts with area ratio p:q -/
noncomputable def dividing_line_length (T : Trapezoid) (p q : ℝ) : ℝ :=
  Real.sqrt ((p * T.a^2 + q * T.b^2) / (p + q))

theorem dividing_line_theorem (T : Trapezoid) (p q : ℝ) 
  (h_positive : 0 < p ∧ 0 < q) :
  let x := dividing_line_length T p q
  let y := (T.m * (T.a - x)) / (T.a - T.b)
  (y * (T.a + x)) / ((T.m - y) * (x + T.b)) = q / p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_theorem_l1135_113535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_theorem_l1135_113553

/-- The number of ways to divide a 2×n rectangle into rectangles with integral sides -/
noncomputable def rectangle_division (n : ℕ) : ℝ :=
  ((3 - Real.sqrt 2) / 4) * (3 + Real.sqrt 2) ^ n + 
  ((3 + Real.sqrt 2) / 4) * (3 - Real.sqrt 2) ^ n

/-- The recurrence relation for the rectangle division problem -/
def recurrence_relation (r : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → r n = 6 * r (n - 1) - 7 * r (n - 2)

theorem rectangle_division_theorem :
  recurrence_relation rectangle_division ∧
  rectangle_division 1 = 2 ∧
  rectangle_division 2 = 8 := by
  sorry

#check rectangle_division_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_theorem_l1135_113553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1135_113549

def f (x : ℤ) : ℤ := x.natAbs - 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f : Set.image f domain = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1135_113549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_AC_length_maximum_area_l1135_113532

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ  -- Internal angle A
  BC : ℝ  -- Side BC
  B : ℝ  -- Internal angle B
  y : ℝ  -- Area of triangle

/-- The specific triangle described in the problem -/
noncomputable def problemTriangle (x y : ℝ) : Triangle where
  A := Real.pi / 3
  BC := 2 * Real.sqrt 3
  B := x
  y := y

theorem side_AC_length (t : Triangle) (h : t.B = Real.pi / 4) :
  ∃ AC : ℝ, AC = 2 * Real.sqrt 2 := by
  sorry

theorem maximum_area (t : Triangle) :
  ∃ max_y : ℝ, max_y = 3 * Real.sqrt 3 ∧ ∀ y : ℝ, t.y ≤ max_y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_AC_length_maximum_area_l1135_113532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_statements_l1135_113566

-- Define the statements
def statement_A (a : ℝ) : Prop :=
  (a = -1) ↔ (∀ x y : ℝ, a^2*x - y + 1 = 0 ↔ x - a*y - 2 = 0)

noncomputable def statement_B : Prop :=
  ∀ α : ℝ, ∃ θ : ℝ, 
    (θ ∈ Set.Icc 0 (Real.pi/4) ∪ Set.Icc (3*Real.pi/4) Real.pi) ∧
    (Real.tan θ = -Real.sin α)

def statement_C (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∀ x y : ℝ, (y - y₁) / (y₂ - y₁) = (x - x₁) / (x₂ - x₁)

def statement_D (k : ℝ) : Prop :=
  ∀ x y : ℝ, (y = k*(x-2)) ↔ (k = y/(x-2))

-- Theorem stating which statements are correct and incorrect
theorem geometric_statements :
  (¬ ∀ a : ℝ, statement_A a) ∧
  statement_B ∧
  (¬ ∀ x₁ y₁ x₂ y₂ : ℝ, statement_C x₁ y₁ x₂ y₂) ∧
  (∀ k : ℝ, statement_D k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_statements_l1135_113566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_widget_production_increase_l1135_113554

/-- The number of widgets produced by Machine X per hour -/
noncomputable def widgets_per_hour_X : ℝ := 18

/-- The total number of widgets produced by both machines -/
noncomputable def total_widgets : ℝ := 1080

/-- The time difference between Machine X and Machine Y to produce the total widgets -/
noncomputable def time_difference : ℝ := 10

/-- The number of widgets produced by Machine Y per hour -/
noncomputable def widgets_per_hour_Y : ℝ := total_widgets / (total_widgets / widgets_per_hour_X - time_difference)

/-- The percentage increase in widgets produced by Machine Y compared to Machine X -/
noncomputable def percentage_increase : ℝ := (widgets_per_hour_Y - widgets_per_hour_X) / widgets_per_hour_X * 100

theorem widget_production_increase :
  percentage_increase = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_widget_production_increase_l1135_113554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_fifth_odd_integer_l1135_113581

theorem ninety_fifth_odd_integer : (λ (n : ℕ) => 2 * n - 1) 95 = 189 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_fifth_odd_integer_l1135_113581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_l1135_113524

def our_sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 3) ∧
  (a 3 = 39) ∧
  ∀ n : ℕ, n > 1 → a n = (a (n-1) + a (n+1)) / 3

theorem sixth_term (a : ℕ → ℕ) (h : our_sequence a) : a 6 = 707 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_l1135_113524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_image_l1135_113534

/-- Dilation of a complex number -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

theorem dilation_image :
  let center : ℂ := 1 - 3*I
  let scale : ℝ := 3
  let original : ℂ := -1 + 2*I
  dilation center scale original = -5 + 12*I :=
by
  -- Unfold the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_image_l1135_113534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_integers_product_l1135_113561

def unit_digit (n : ℕ) : ℕ := n % 10

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem special_integers_product (a b c : ℕ) 
  (ha : a > 1000) (hb : b > 1000) (hc : c > 1000)
  (hab : unit_digit (a + b) = unit_digit c)
  (hbc : unit_digit (b + c) = unit_digit a)
  (hca : unit_digit (c + a) = unit_digit b) :
  last_three_digits (a * b * c) ∈ ({000, 250, 500, 750} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_integers_product_l1135_113561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_exceeding_500000_l1135_113594

-- Define the sequence of functions
def f : ℕ → (ℝ → ℝ)
| 0 => fun x => |x - 1|
| (n + 1) => fun x => f n (|x - (n + 1)|)

-- Define the sum of zeros function
noncomputable def sumOfZeros (n : ℕ) : ℝ := sorry

-- Theorem statement
theorem least_n_exceeding_500000 :
  (∀ k < 101, sumOfZeros k ≤ 500000) ∧
  sumOfZeros 101 > 500000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_exceeding_500000_l1135_113594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1135_113571

/-- Additional cost function -/
noncomputable def c (x : ℝ) : ℝ :=
  if x < 80 then (1/2) * x^2 + 40 * x
  else 101 * x + 8100 / x - 2180

/-- Profit function -/
noncomputable def profit (x : ℝ) : ℝ :=
  if x < 80 then -(1/2) * x^2 + 60 * x - 500
  else 1680 - (x + 8100 / x)

/-- Theorem stating the maximum profit and corresponding output -/
theorem max_profit :
  ∃ (max_profit : ℝ) (max_output : ℝ),
    max_profit = 1500 ∧
    max_output = 90 ∧
    ∀ x, x > 0 → profit x ≤ max_profit :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1135_113571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_2x_l1135_113597

open Real MeasureTheory

theorem definite_integral_exp_2x : 
  ∫ x in Set.Icc 0 1, Real.exp (2 * x) = (1/2) * Real.exp 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_2x_l1135_113597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_zero_platform_length_zero_l1135_113567

-- Define the given parameters
noncomputable def train_length : ℝ := 750
noncomputable def train_speed_kmph : ℝ := 180
noncomputable def crossing_time : ℝ := 15

-- Convert train speed to m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmph * 1000 / 3600

-- Calculate the distance covered by the train
noncomputable def distance_covered : ℝ := train_speed_ms * crossing_time

-- Define the theorem
theorem platform_length_is_zero :
  distance_covered = train_length → 0 = distance_covered - train_length :=
by
  intro h
  rw [h]
  ring

-- The main theorem stating that the platform length is 0
theorem platform_length_zero :
  0 = distance_covered - train_length :=
by
  apply platform_length_is_zero
  -- Here we would prove that distance_covered = train_length
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_zero_platform_length_zero_l1135_113567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_folding_configurations_l1135_113511

/-- Represents a position where an additional square can be attached to the cross configuration -/
inductive Position
| EndOfArm
| Intermediate

/-- Represents a configuration of squares -/
structure Configuration :=
(additional_square_position : Position)

/-- Determines if a configuration can be folded into a cube with one face missing -/
def can_fold_to_cube (config : Configuration) : Bool :=
  match config.additional_square_position with
  | Position.EndOfArm => true
  | Position.Intermediate => false

/-- The total number of possible positions to attach the additional square -/
def total_positions : Nat := 8

/-- The number of end-of-arm positions -/
def end_of_arm_positions : Nat := 4

/-- The number of intermediate positions -/
def intermediate_positions : Nat := 4

/-- The number of configurations that can be folded into a cube -/
def foldable_configurations : Nat := end_of_arm_positions

theorem cube_folding_configurations :
  (∃ (n : Nat), n = total_positions ∧
    n = end_of_arm_positions + intermediate_positions) →
  (∃ (foldable : Nat), foldable = end_of_arm_positions ∧
    foldable = foldable_configurations) :=
by
  intro h
  use end_of_arm_positions
  constructor
  . rfl
  . rfl

#check cube_folding_configurations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_folding_configurations_l1135_113511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_sine_cosine_l1135_113560

theorem tangent_to_sine_cosine (α : Real) 
  (h1 : Real.tan α = -3) 
  (h2 : π / 2 < α ∧ α < π) : 
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ 
  Real.cos α = - Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_sine_cosine_l1135_113560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_proof_l1135_113552

theorem base_number_proof (x : ℝ) (h : (x^(3/10))^(40/3) = 16) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_proof_l1135_113552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_l1135_113538

theorem fraction_sum (m n : ℕ) (h1 : (2013 * 2013) / (2014 * 2014 + 2012) = n / m) 
  (h2 : Nat.Coprime m n) : m + n = 1343 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_l1135_113538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial15_trailing_zeros_l1135_113543

/-- The number of trailing zeros in the base 16 representation of a natural number -/
def trailingZerosBase16 (n : ℕ) : ℕ := sorry

/-- 15 factorial -/
def factorial15 : ℕ := Nat.factorial 15

theorem factorial15_trailing_zeros : trailingZerosBase16 factorial15 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial15_trailing_zeros_l1135_113543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l1135_113536

/-- The eccentricity of the curve represented by ρ = 1 / (1 - cos θ + sin θ) -/
noncomputable def eccentricity : ℝ := Real.sqrt 2

/-- The polar equation of the curve -/
noncomputable def polar_equation (θ : ℝ) : ℝ := 1 / (1 - Real.cos θ + Real.sin θ)

/-- Theorem stating that the curve is a hyperbola -/
theorem curve_is_hyperbola : eccentricity > 1 ∧ 
  ∃ (ℓ θ₀ : ℝ), ∀ θ, polar_equation θ = ℓ / (1 - eccentricity * Real.cos (θ - θ₀)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l1135_113536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_def_l1135_113589

/-- Triangle XYZ with points D, E, F on its sides -/
structure TriangleXYZ where
  /-- Side lengths of triangle XYZ -/
  XY : ℝ
  YZ : ℝ
  XZ : ℝ
  /-- Ratios for points D, E, F on sides XY, YZ, XZ respectively -/
  p : ℝ
  q : ℝ
  r : ℝ

/-- Area of triangle DEF -/
noncomputable def area_triangle_DEF (T : TriangleXYZ) : ℝ := sorry

/-- Area of triangle XYZ -/
noncomputable def area_triangle_XYZ (T : TriangleXYZ) : ℝ := sorry

/-- Theorem stating the ratio of areas of triangles DEF and XYZ -/
theorem area_ratio_def (T : TriangleXYZ) 
  (h1 : T.XY = 12) 
  (h2 : T.YZ = 16) 
  (h3 : T.XZ = 20) 
  (h4 : T.p > 0 ∧ T.q > 0 ∧ T.r > 0) 
  (h5 : T.p + T.q + T.r = 0.9) 
  (h6 : T.p^2 + T.q^2 + T.r^2 = 0.29) : 
  (area_triangle_DEF T) / (area_triangle_XYZ T) = 37 / 100 := by
  sorry

#check area_ratio_def

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_def_l1135_113589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_rate_l1135_113579

/-- Calculates the compound interest rate given initial investment, time, and final amount -/
noncomputable def compound_interest_rate (initial_investment : ℝ) (time : ℝ) (final_amount : ℝ) : ℝ :=
  ((final_amount / initial_investment) ^ (1 / time) - 1) * 100

/-- Theorem: The compound interest rate for the given investment scenario is 5% -/
theorem investment_interest_rate :
  let initial_investment : ℝ := 8000
  let time : ℝ := 2
  let final_amount : ℝ := 8820
  compound_interest_rate initial_investment time final_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_rate_l1135_113579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l1135_113528

def is_valid_number (n : ℕ) : Bool :=
  10 ≤ n ∧ n < 100 ∧
  let tens := n / 10
  let ones := n % 10
  (n % (tens - ones) = 0) ∧ (n % (tens * ones) = 0)

theorem sum_of_special_numbers : 
  (Finset.filter (fun n => is_valid_number n = true) (Finset.range 100)).sum id = 73 := by
  sorry

#eval (Finset.filter (fun n => is_valid_number n = true) (Finset.range 100)).sum id

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l1135_113528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frameable_iff_cos_rational_three_four_six_frameable_not_frameable_ge_seven_five_not_frameable_l1135_113551

/-- A positive integer n ≥ 3 is frameable if it's possible to draw a regular n-gon with vertices
    on infinitely many equidistant parallel lines, with no line containing more than one vertex. -/
def Frameable (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (z : ℂ), ∀ i : Fin n, ∃ k : ℤ, (z * Complex.exp (2 * Real.pi * Complex.I * (i : ℂ) / n)).im = k

theorem frameable_iff_cos_rational (n : ℕ) :
  Frameable n ↔ n ≥ 3 ∧ ∃ (q : ℚ), Real.cos (2 * Real.pi / n) = q := by sorry

theorem three_four_six_frameable :
  Frameable 3 ∧ Frameable 4 ∧ Frameable 6 := by sorry

theorem not_frameable_ge_seven (n : ℕ) (h : n ≥ 7) :
  ¬Frameable n := by sorry

theorem five_not_frameable :
  ¬Frameable 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frameable_iff_cos_rational_three_four_six_frameable_not_frameable_ge_seven_five_not_frameable_l1135_113551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_bread_toast_l1135_113525

/-- Represents a side of a piece of bread -/
inductive BreadSide
| Side1
| Side2
deriving Repr

/-- Represents the toasted state of a bread side -/
inductive ToastedState
| Untoasted
| Toasted
deriving Repr

/-- Represents a piece of bread with two sides -/
structure Bread where
  side1 : ToastedState
  side2 : ToastedState
deriving Repr

/-- Represents the state of the frying pan -/
structure PanState where
  bread1 : Option Bread
  bread2 : Option Bread
deriving Repr

/-- Checks if a bread is fully toasted -/
def isFullyToasted (b : Bread) : Prop :=
  b.side1 = ToastedState.Toasted ∧ b.side2 = ToastedState.Toasted

/-- Represents the toasting process -/
def toast (initialState : PanState) (time : Nat) : Prop :=
  ∃ (finalState : PanState),
    (time ≤ 3) ∧
    (∃ b1 b2 b3 : Bread, isFullyToasted b1 ∧ isFullyToasted b2 ∧ isFullyToasted b3)

/-- The main theorem -/
theorem three_bread_toast :
  ∃ (initialState : PanState),
    toast initialState 3 :=
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_bread_toast_l1135_113525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1135_113537

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  -- Amplitude and period
  (∀ x, |f x| ≤ 2) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ Real.pi) ∧
  -- Range when x ∈ [0, π/2]
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-1) 2) ∧
  (∃ x₁ ∈ Set.Icc 0 (Real.pi / 2), f x₁ = -1) ∧
  (∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), f x₂ = 2) ∧
  -- Intervals of monotonic decrease
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), ∀ y ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), x < y → f x > f y) ∧
  (∀ x ∈ Set.Icc (-5 * Real.pi / 6) (-Real.pi / 3), ∀ y ∈ Set.Icc (-5 * Real.pi / 6) (-Real.pi / 3), x < y → f x > f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1135_113537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diff_seq_is_arithmetic_general_formula_exists_arithmetic_seq_l1135_113547

/-- Define the first-order difference sequence -/
def first_order_diff (a : ℕ → ℝ) : ℕ → ℝ := λ n => a (n + 1) - a n

/-- Part 1: Prove that the first-order difference sequence of n^2 - n is arithmetic -/
theorem diff_seq_is_arithmetic :
  let a : ℕ → ℝ := λ n => (n : ℝ)^2 - n
  ∃ d : ℝ, ∀ n : ℕ, first_order_diff a (n + 1) - first_order_diff a n = d :=
by
  sorry

/-- Part 2: Prove the general formula for a_n given the conditions -/
theorem general_formula :
  ∃ a : ℕ → ℝ, a 1 = 1 ∧ 
    (∀ n : ℕ, first_order_diff a n - a n = (2 : ℝ)^n) ∧
    (∀ n : ℕ, a n = n * (2 : ℝ)^(n-1)) :=
by
  sorry

/-- Part 3: Prove the existence of an arithmetic sequence b_n satisfying the condition -/
theorem exists_arithmetic_seq :
  let a : ℕ → ℝ := λ n => n * (2 : ℝ)^(n-1)
  ∃ b : ℕ → ℝ, (∀ n : ℕ, b (n + 1) - b n = b 2 - b 1) ∧
    (∀ n : ℕ, (Finset.range n).sum (λ k => b (k + 1) * (Nat.choose n (k + 1))) = a n) ∧
    (∀ n : ℕ, b n = n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diff_seq_is_arithmetic_general_formula_exists_arithmetic_seq_l1135_113547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_range_of_a_l1135_113565

noncomputable section

-- Define the inverse proportion function
def inverse_prop (k : ℝ) (x : ℝ) : ℝ := (k - 4) / x

-- Define the conditions for the function passing through first and third quadrants
def passes_first_third_quadrants (k : ℝ) : Prop :=
  ∀ x, x > 0 → inverse_prop k x > 0 ∧ ∀ x, x < 0 → inverse_prop k x < 0

-- Theorem for the range of k
theorem range_of_k (k : ℝ) (h : passes_first_third_quadrants k) : k > 4 := by
  sorry

-- Define the conditions for the function passing through two points in the first quadrant
def passes_two_points (k a : ℝ) (y₁ y₂ : ℝ) : Prop :=
  inverse_prop k (a + 5) = y₁ ∧ inverse_prop k (2 * a + 1) = y₂

-- Theorem for the range of a
theorem range_of_a (k a : ℝ) (y₁ y₂ : ℝ) 
  (h1 : passes_first_third_quadrants k)
  (h2 : a > 0)
  (h3 : passes_two_points k a y₁ y₂)
  (h4 : y₁ < y₂) : 
  0 < a ∧ a < 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_range_of_a_l1135_113565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_seven_percent_l1135_113582

/-- Represents a loan with a principal amount and duration -/
structure Loan where
  principal : ℚ
  duration : ℚ

/-- Calculates the simple interest for a given loan and interest rate -/
def simpleInterest (loan : Loan) (rate : ℚ) : ℚ :=
  loan.principal * loan.duration * rate / 100

/-- Theorem: Given the loan conditions, the interest rate is 7% -/
theorem interest_rate_is_seven_percent 
  (loan_to_b : Loan)
  (loan_to_c : Loan)
  (total_interest : ℚ)
  (h1 : loan_to_b.principal = 5000)
  (h2 : loan_to_b.duration = 2)
  (h3 : loan_to_c.principal = 3000)
  (h4 : loan_to_c.duration = 4)
  (h5 : total_interest = 1540)
  (h6 : ∃ rate, simpleInterest loan_to_b rate + simpleInterest loan_to_c rate = total_interest) :
  ∃ rate, rate = 7 ∧ simpleInterest loan_to_b rate + simpleInterest loan_to_c rate = total_interest :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_seven_percent_l1135_113582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_about_two_l1135_113529

noncomputable def f (x : ℝ) : ℝ := |⌊x⌋| - |⌊3 - x⌋|

theorem f_symmetry_about_two : ∀ x : ℝ, f x = f (2 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_about_two_l1135_113529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l1135_113580

-- Define the functions
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2^x + m - 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log m

-- Define the properties
def has_zero (m : ℝ) : Prop := ∃ x, f m x = 0
def is_decreasing (m : ℝ) : Prop := ∀ x y, 0 < x → x < y → g m y < g m x

-- State the theorem
theorem necessary_not_sufficient :
  (∀ m : ℝ, is_decreasing m → has_zero m) ∧
  (∃ m : ℝ, has_zero m ∧ ¬is_decreasing m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l1135_113580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_problem_l1135_113591

theorem fraction_sum_problem :
  let known_fractions : List ℚ := [1/3, 1/7, 1/9, 1/11, 1/33]
  let unknown_fractions : List ℚ := [1/5, 1/15, 1/45, 1/385]
  (List.sum known_fractions + List.sum unknown_fractions = 1) ∧
  (∀ f ∈ unknown_fractions, ∃ n : ℕ, f = 1 / (10 * n + 5)) →
  unknown_fractions = [1/5, 1/15, 1/45, 1/385] := by
  sorry

#check fraction_sum_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_problem_l1135_113591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution1_solution2_solution3_l1135_113520

-- Part 1
def equation1 (p q : ℝ) (x : ℝ) : Prop := x + p / x = q

theorem solution1 (p q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 p q x₁ ∧ equation1 p q x₂ ∧ x₁ = -2 ∧ x₂ = 3) →
  p = -6 ∧ q = 1 := by sorry

-- Part 2
def equation2 (x : ℝ) : Prop := x - 2 / x = 3

theorem solution2 (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation2 x₁ ∧ equation2 x₂ ∧ x₁ = a ∧ x₂ = b) →
  a^2 + b^2 = 13 := by sorry

-- Part 3
def equation3 (n : ℝ) (x : ℝ) : Prop := 2*x + (2*n^2 + n) / (2*x + 1) = 3*n

theorem solution3 (n : ℝ) (x₁ x₂ : ℝ) :
  (x₁ < x₂ ∧ equation3 n x₁ ∧ equation3 n x₂) →
  (2*x₁ + 1) / (2*x₂ - 2) = n / (2*n - 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution1_solution2_solution3_l1135_113520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1135_113563

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, |2*x - 1| < x + a ↔ -1/3 < x ∧ x < 3

def q (a : ℝ) : Prop := ∀ x : ℝ, ¬(4*x ≥ 4*a*x^2 + 1)

-- Define the main theorem
theorem range_of_a : 
  (∀ a : ℝ, p a ∨ q a) → 
  ∀ a : ℝ, a ∈ Set.Ioi 1 ↔ p a ∨ q a :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1135_113563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_decreasing_on_interval_l1135_113550

-- Define the function
def f (x : ℝ) := x^2 - x + 1

-- State the theorem
theorem f_not_decreasing_on_interval :
  ¬(∀ (x y : ℝ), x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x ≤ y → f x ≥ f y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_decreasing_on_interval_l1135_113550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_is_correct_l1135_113527

/-- The maximum area of the projection of a rectangular parallelepiped onto a plane -/
noncomputable def max_projection_area (a b c : ℝ) : ℝ :=
  Real.sqrt (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)

/-- Theorem: The maximum area of the projection of a rectangular parallelepiped
    with edges a, b, and c onto a plane is √(a²b² + b²c² + c²a²) -/
theorem max_projection_area_is_correct (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ (projected_area : ℝ), projected_area ≤ max_projection_area a b c :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_is_correct_l1135_113527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1135_113546

/-- A three-digit positive integer -/
def A : ℕ := sorry

/-- B is obtained by interchanging the two leftmost digits of A -/
def B : ℕ := 
  let a := A / 100
  let b := (A / 10) % 10
  let c := A % 10
  100 * b + 10 * a + c

/-- C is obtained by doubling B -/
def C : ℕ := 2 * B

/-- D is obtained by subtracting 500 from C -/
def D : ℕ := C - 500

/-- The sum of A, B, C, and D equals 2014 -/
axiom sum_condition : A + B + C + D = 2014

/-- A is a three-digit number -/
axiom A_three_digit : 100 ≤ A ∧ A < 1000

theorem unique_solution : A = 344 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1135_113546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_sum_l1135_113521

def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem quadratic_function_sum : 
  ∃ S, S ≡ 71 [MOD 100] ∧
  ∀ a b c p q : ℤ,
    a > 0 → 
    Nat.Prime p.natAbs → 
    Nat.Prime q.natAbs → 
    p < q → 
    f a b c p = 17 → 
    f a b c q = 17 → 
    f a b c (p + q) = 47 → 
    S = f a b c (p * q) + 
        f a b c (2 * 3) + 
        f a b c (2 * 5) + 
        f a b c (3 * 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_sum_l1135_113521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l1135_113575

noncomputable def f (x : ℝ) : ℝ := (x + 1) / Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x + x^2 - 1

theorem min_value_and_inequality (t : ℝ) :
  (∀ x > -1, g x ≥ 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ = t → f x₂ = t → |x₁ - x₂| > 2 * Real.sqrt (1 - t)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l1135_113575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_red_marbles_l1135_113531

def total_marbles : ℕ := 15
def red_marbles : ℕ := 6
def blue_marbles : ℕ := 3
def white_marbles : ℕ := 6
def marbles_picked : ℕ := 4

theorem probability_three_red_marbles :
  (Nat.choose red_marbles 3 * Nat.choose (blue_marbles + white_marbles) 1) /
  Nat.choose total_marbles marbles_picked = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_red_marbles_l1135_113531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_leq_neg_two_l1135_113509

theorem inequality_holds_iff_a_leq_neg_two (a : ℝ) :
  (a < 0) →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔ (a ≤ -2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_leq_neg_two_l1135_113509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1135_113503

/-- The length of a bridge that a train can cross, given the train's length, speed, and crossing time. -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

/-- Theorem stating the length of the bridge given specific conditions. -/
theorem bridge_length_calculation :
  let train_length := (250 : ℝ)
  let train_speed_kmh := (60 : ℝ)
  let crossing_time_s := (55 : ℝ)
  ∃ ε > 0, |bridge_length train_length train_speed_kmh crossing_time_s - 666.85| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1135_113503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_segments_exist_l1135_113596

/-- Regular 2n-gon -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- Pair of vertices -/
structure VertexPair (n : ℕ) where
  first : Fin (2*n)
  second : Fin (2*n)
  different : first ≠ second

/-- Distance between two vertices -/
noncomputable def distance {n : ℕ} (p : RegularPolygon n) (pair : VertexPair n) : ℝ :=
  let (x₁, y₁) := p.vertices pair.first
  let (x₂, y₂) := p.vertices pair.second
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: In a regular 2n-gon with vertices divided into n pairs, 
    if n = 4m + 2 or n = 4m + 3, then there exist two pairs of vertices 
    that form equal-length segments -/
theorem equal_segments_exist (n : ℕ) (p : RegularPolygon n) 
    (pairs : Fin n → VertexPair n) 
    (h : ∃ m : ℕ, n = 4*m + 2 ∨ n = 4*m + 3) :
  ∃ i j : Fin n, i ≠ j ∧ distance p (pairs i) = distance p (pairs j) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_segments_exist_l1135_113596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_and_surface_ratios_l1135_113541

/-- Represents a triangle with sides a, b, and c, where the angle opposite to side a is obtuse -/
structure ObtusedTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_positive : 0 < a
  b_positive : 0 < b
  c_positive : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b
  obtuse_angle : a^2 > b^2 + c^2

/-- Volumes of solids of revolution formed by rotating the triangle around sides a, b, and c -/
noncomputable def volumes (t : ObtusedTriangle) : ℝ × ℝ × ℝ :=
  (1 / t.a, 1 / t.b, 1 / t.c)

/-- Surface areas of solids of revolution formed by rotating the triangle around sides a, b, and c -/
noncomputable def surfaces (t : ObtusedTriangle) : ℝ × ℝ × ℝ :=
  ((t.b + t.c) / t.a, (t.a + t.c) / t.b, (t.a + t.b) / t.c)

/-- Theorem stating the ratios of volumes and surface areas -/
theorem volume_and_surface_ratios (t : ObtusedTriangle) :
  volumes t = (1 / t.a, 1 / t.b, 1 / t.c) ∧
  surfaces t = ((t.b + t.c) / t.a, (t.a + t.c) / t.b, (t.a + t.b) / t.c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_and_surface_ratios_l1135_113541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_2019_value_l1135_113508

def a (n : ℕ) : ℕ := sorry

def S (n : ℕ) : ℕ := sorry

theorem S_2019_value :
  (a 1 = 1) →
  (∀ n, a n = 1 ∨ a n = 2) →
  (∀ k, ∃ m, (∀ i ∈ Finset.range (2*k-1), a (m + i) = 2) ∧
             a (m - 1) = 1 ∧
             a (m + 2*k-1) = 1) →
  (∀ n, S n = (Finset.range n).sum (fun i => a (i + 1))) →
  S 2019 = 3993 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_2019_value_l1135_113508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_b_inconsistent_l1135_113514

-- Define a type for points in a geometric space
structure Point where
  mk :: -- empty constructor

-- Define a predicate for points on the locus
def OnLocus : Point → Prop := fun _ => True

-- Define a predicate for points satisfying the conditions
def SatisfiesConditions : Point → Prop := fun _ => True

-- Statement B from the problem
def StatementB : Prop :=
  (∀ p : Point, ¬OnLocus p → SatisfiesConditions p) ∧
  (∀ p : Point, SatisfiesConditions p → OnLocus p)

-- Theorem stating that Statement B is inconsistent
theorem statement_b_inconsistent : ¬StatementB := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_b_inconsistent_l1135_113514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_specific_point_l1135_113556

noncomputable def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 then Real.arctan (y / x) + Real.pi
           else if y > 0 then Real.pi / 2
           else if y < 0 then -Real.pi / 2
           else 0
  let θ_normalized := (θ + 2 * Real.pi) % (2 * Real.pi)
  (r, θ_normalized, z)

theorem rectangular_to_cylindrical_specific_point :
  rectangular_to_cylindrical 3 (-3 * Real.sqrt 3) 2 = (6, 5 * Real.pi / 3, 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_specific_point_l1135_113556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_factorial_mod_thirteen_l1135_113598

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

/-- The statement of the problem -/
theorem twelve_factorial_mod_thirteen : factorial 12 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_factorial_mod_thirteen_l1135_113598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_textbook_problem_solution_l1135_113569

def problem_sequence (x : ℕ) (n : ℕ) : ℕ := x - n + 1

def total_problems (x : ℕ) (days : ℕ) : ℕ :=
  Finset.sum (Finset.range days) (fun i => problem_sequence x i)

theorem textbook_problem_solution (x : ℕ) :
  (total_problems x 3 = 45) ∧ (total_problems x 7 = 91) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_textbook_problem_solution_l1135_113569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_greater_than_formula_l1135_113516

/-- The percent by which one number is greater than another -/
noncomputable def percentGreaterThan (M N : ℝ) : ℝ := 100 * (M - N) / N

/-- Theorem: The percent by which M is greater than N is equal to (100(M-N))/N -/
theorem percent_greater_than_formula (M N : ℝ) (h : N ≠ 0) : 
  percentGreaterThan M N = 100 * (M - N) / N := by
  -- Unfold the definition of percentGreaterThan
  unfold percentGreaterThan
  -- The rest of the proof would go here
  sorry

#check percent_greater_than_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_greater_than_formula_l1135_113516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_prime_properties_l1135_113530

theorem sequence_prime_properties (a : ℕ) (h : a > 1) :
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ p ∈ S, Nat.Prime p ∧ ∃ n : ℕ, n ≥ 1 ∧ p ∣ (a^n + 1)) ∧
  (∃ (T : Set ℕ), Set.Infinite T ∧ ∀ q ∈ T, Nat.Prime q ∧ ∀ n : ℕ, n ≥ 1 → ¬(q ∣ (a^n + 1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_prime_properties_l1135_113530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_point_d_l1135_113599

-- Define the complex function g
noncomputable def g (z : ℂ) : ℂ := ((1 + Complex.I) * z + (3 * Complex.I - Real.sqrt 2)) / 2

-- Define the rotation point d
noncomputable def d : ℂ := -3/2 + ((3 + Real.sqrt 2)/2) * Complex.I

-- Theorem statement
theorem rotation_point_d : g d = d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_point_d_l1135_113599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_travel_distance_l1135_113507

/-- The distance between towns -/
noncomputable def town_distance : ℝ := 60

/-- Aaron's speed in km/h -/
noncomputable def aaron_speed : ℝ := 17

/-- Michael's speed in km/h -/
noncomputable def michael_speed : ℝ := 7

/-- Time taken for Aaron and Michael to meet -/
noncomputable def meeting_time : ℝ := (2 * town_distance) / (aaron_speed + michael_speed)

/-- Distance traveled by Michael -/
noncomputable def michael_distance : ℝ := michael_speed * meeting_time

/-- Distance of meeting point from Centerville -/
noncomputable def meeting_point_distance : ℝ := town_distance - michael_distance

/-- Theorem stating that David's travel distance is 65 km -/
theorem david_travel_distance :
  Real.sqrt (town_distance^2 + meeting_point_distance^2) = 65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_travel_distance_l1135_113507
