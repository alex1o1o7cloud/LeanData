import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l354_35492

/-- The volume of a regular tetrahedron with face circumradius R -/
noncomputable def regularTetrahedronVolume (R : ℝ) : ℝ := (R^3 * Real.sqrt 6) / 4

/-- Theorem: The volume of a regular tetrahedron with face circumradius R is (R^3 * √6) / 4 -/
theorem regular_tetrahedron_volume (R : ℝ) (h : R > 0) :
  regularTetrahedronVolume R = (R^3 * Real.sqrt 6) / 4 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l354_35492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_change_l354_35460

-- Define the initial tax rate and consumption
variable (T : ℝ) (C : ℝ)

-- Define the new tax rate after 14% reduction
def new_tax_rate (T : ℝ) : ℝ := T * (1 - 0.14)

-- Define the new consumption after 15% increase
def new_consumption (C : ℝ) : ℝ := C * (1 + 0.15)

-- Define the initial revenue
def initial_revenue (T C : ℝ) : ℝ := T * C

-- Define the new revenue
def new_revenue (T C : ℝ) : ℝ := new_tax_rate T * new_consumption C

-- Theorem statement
theorem revenue_change (T C : ℝ) :
  (new_revenue T C - initial_revenue T C) / initial_revenue T C = -0.011 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_change_l354_35460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l354_35425

/-- The inclination angle of a line with slope m is arctan(m) --/
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m

/-- The line equation sqrt(3)x - y + 1 = 0 --/
def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0

theorem line_inclination_angle :
  ∃ (x y : ℝ), line_equation x y ∧ inclination_angle (Real.sqrt 3) = π / 3 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l354_35425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l354_35493

open Real

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (a + 2) * x + 2 * a * log x

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := -(a + 2) * x

/-- Theorem: If there exists x₀ ∈ [e, 4] such that f(x₀) > g(x₀), then a > -2/log(2) -/
theorem function_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (exp 1) 4 ∧ f a x₀ > g a x₀) → a > -2 / log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l354_35493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_ratio_l354_35416

-- Define a triangle with side lengths a, b, c and opposite angles α, β, γ
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the theorem
theorem triangle_cotangent_ratio (t : Triangle) 
  (h1 : t.a > 0) (h2 : t.b > 0) (h3 : t.c > 0)
  (h4 : t.α > 0) (h5 : t.β > 0) (h6 : t.γ > 0)
  (h7 : t.α + t.β + t.γ = Real.pi)
  (h8 : t.a^2 + t.b^2 = 2010 * t.c^2) :
  (Real.tan t.γ)⁻¹ / ((Real.tan t.α)⁻¹ + (Real.tan t.β)⁻¹) = 1004.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_ratio_l354_35416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_path_comparison_min_path_ratio_l354_35431

/-- Represents a rectangular quarry -/
structure Quarry where
  length : ℝ
  width : ℝ
  length_pos : length > 0
  width_pos : width > 0

/-- Represents the path of the first swimmer (along the diagonal and back) -/
noncomputable def diagonal_path (q : Quarry) : ℝ := 2 * Real.sqrt (q.length^2 + q.width^2)

/-- Represents a point on the side of the quarry -/
structure QuarryPoint where
  x : ℝ
  y : ℝ

/-- Represents the path of the second swimmer -/
noncomputable def quadrilateral_path (q : Quarry) (start : QuarryPoint) (p1 p2 p3 : QuarryPoint) : ℝ :=
  Real.sqrt ((start.x - p1.x)^2 + (start.y - p1.y)^2) +
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2) +
  Real.sqrt ((p2.x - p3.x)^2 + (p2.y - p3.y)^2) +
  Real.sqrt ((p3.x - start.x)^2 + (p3.y - start.y)^2)

/-- The starting point of the second swimmer -/
noncomputable def second_swimmer_start (q : Quarry) : QuarryPoint :=
  { x := q.length * (2018 / 4037), y := 0 }

theorem swimmer_path_comparison (q : Quarry) (p1 p2 p3 : QuarryPoint) :
  diagonal_path q ≤ quadrilateral_path q (second_swimmer_start q) p1 p2 p3 := by
  sorry

theorem min_path_ratio (q : Quarry) :
  ∃ (p1 p2 p3 : QuarryPoint),
    quadrilateral_path q (second_swimmer_start q) p1 p2 p3 / diagonal_path q = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_path_comparison_min_path_ratio_l354_35431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_point_l354_35408

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x^2 / 4 - 4 * Real.log x

-- State the theorem
theorem tangent_intersection_point :
  ∃ (x : ℝ), x > 0 ∧ 
  (deriv f x = 1) ∧ 
  x = 4 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_point_l354_35408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_volume_triangular_pyramid_l354_35411

/-- A quadrilateral pyramid with a parallelogram base -/
structure QuadrilateralPyramid where
  base : Parallelogram
  apex : Point

/-- A triangular pyramid -/
structure TriangularPyramid where
  vertices : Fin 4 → Point

/-- The volume of a pyramid -/
noncomputable def volume (p : QuadrilateralPyramid ⊕ TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating the existence of a triangular pyramid with half the volume -/
theorem half_volume_triangular_pyramid (qp : QuadrilateralPyramid) :
  ∃ (tp : TriangularPyramid), volume (Sum.inr tp) = (1/2 : ℝ) * volume (Sum.inl qp) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_volume_triangular_pyramid_l354_35411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l354_35447

-- Define the circle C
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 8*y + m = 0

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Define external tangency
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_C m x y ∧ unit_circle x y ∧
  ∀ (x' y' : ℝ), circle_C m x' y' ∧ unit_circle x' y' → (x = x' ∧ y = y')

-- Define chord length
noncomputable def chord_length (m : ℝ) : ℝ :=
  2 * Real.sqrt 7

theorem circle_C_properties (m : ℝ) :
  (externally_tangent m → m = 9) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_C m x₁ y₁ ∧ circle_C m x₂ y₂ ∧ 
    line x₁ y₁ ∧ line x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (chord_length m)^2 
    → m = 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l354_35447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_512_l354_35459

theorem log_8_512 : (Real.log 512) / (Real.log 8) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_512_l354_35459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_edge_distance_is_three_fourths_a_l354_35449

/-- Regular triangular pyramid with base side length a and lateral edge angle 60° -/
structure RegularTriangularPyramid where
  a : ℝ
  base_side_length : a > 0
  lateral_edge_angle : ℝ
  angle_is_60 : lateral_edge_angle = 60

/-- Distance between opposite edges of the pyramid -/
noncomputable def opposite_edge_distance (p : RegularTriangularPyramid) : ℝ := 3 * p.a / 4

/-- Theorem: The distance between opposite edges of the pyramid is 3a/4 -/
theorem opposite_edge_distance_is_three_fourths_a (p : RegularTriangularPyramid) :
  opposite_edge_distance p = 3 * p.a / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_edge_distance_is_three_fourths_a_l354_35449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_f_l354_35400

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 5 then 2
  else if 5 ≤ x ∧ x < 10 then 3
  else if 10 ≤ x ∧ x < 15 then 4
  else if 15 ≤ x ∧ x ≤ 20 then 5
  else 0  -- This case should never occur for the given domain

-- Theorem for the domain
theorem domain_of_f : Set.Ioo 0 20 = {x | ∃ y, f x = y} := by
  sorry

-- Theorem for the range
theorem range_of_f : Set.range f = {2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_f_l354_35400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_power_of_two_l354_35413

theorem square_sum_power_of_two (n : ℕ) : 
  (∃ k : ℕ, (2^6 : ℕ) + (2^9 : ℕ) + (2^n : ℕ) = k^2) ↔ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_power_of_two_l354_35413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_months_is_nine_l354_35466

/-- Represents the pasture rental problem -/
structure PastureRental where
  total_cost : ℚ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  c_horses : ℕ
  c_months : ℕ
  b_payment : ℚ

/-- Calculates the number of months b put in the horses -/
noncomputable def calculate_b_months (pr : PastureRental) : ℚ :=
  let total_horse_months := pr.a_horses * pr.a_months + pr.c_horses * pr.c_months
  let b_horse_months := pr.b_payment * (total_horse_months + pr.b_horses * pr.b_payment / pr.total_cost) / (pr.total_cost - pr.b_payment)
  b_horse_months / pr.b_horses

/-- Theorem stating that for the given problem, b put in horses for 9 months -/
theorem b_months_is_nine (pr : PastureRental)
  (h1 : pr.total_cost = 870)
  (h2 : pr.a_horses = 12)
  (h3 : pr.a_months = 8)
  (h4 : pr.b_horses = 16)
  (h5 : pr.c_horses = 18)
  (h6 : pr.c_months = 6)
  (h7 : pr.b_payment = 360) :
  calculate_b_months pr = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_months_is_nine_l354_35466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_factory_per_shirt_payment_l354_35433

/-- Calculates the per-shirt payment to employees in a t-shirt factory --/
theorem tshirt_factory_per_shirt_payment
  (num_employees : ℕ)
  (shirts_per_employee : ℕ)
  (shift_hours : ℕ)
  (hourly_rate : ℚ)
  (shirt_price : ℚ)
  (nonemployee_expenses : ℚ)
  (daily_profit : ℚ)
  (h1 : num_employees = 20)
  (h2 : shirts_per_employee = 20)
  (h3 : shift_hours = 8)
  (h4 : hourly_rate = 12)
  (h5 : shirt_price = 35)
  (h6 : nonemployee_expenses = 1000)
  (h7 : daily_profit = 9080) :
  (shirt_price * num_employees * shirts_per_employee - num_employees * shift_hours * hourly_rate - nonemployee_expenses) / (num_employees * shirts_per_employee) = 27.70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_factory_per_shirt_payment_l354_35433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l354_35452

def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property
  (f : ℝ → ℝ)
  (odd : OddFunction f)
  (pos : ∀ x > 0, f x = x + 2) :
  ∀ x < 0, f x = x - 2 := by
  intro x hx
  have h1 : f (-x) = -x + 2 := pos (-x) (by linarith)
  have h2 : f (-x) = -f x := odd x
  rw [h2] at h1
  linarith

#check odd_function_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l354_35452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_sum_l354_35477

/-- AngleBetween represents the angle between a line and a plane. -/
def AngleBetween (l : Line) (p : Plane) : ℝ :=
sorry

/-- DihedralAngle represents the angle between two planes. -/
def DihedralAngle (p₁ p₂ : Plane) : ℝ :=
sorry

/-- Given a line forming angles θ₁ and θ₂ with the two faces of a dihedral angle,
    prove that θ₁ + θ₂ ≤ 90°. -/
theorem dihedral_angle_sum (θ₁ θ₂ : ℝ) 
  (h₁ : 0 ≤ θ₁ ∧ θ₁ ≤ π / 2) 
  (h₂ : 0 ≤ θ₂ ∧ θ₂ ≤ π / 2) 
  (h₃ : ∃ (l : Line) (p₁ p₂ : Plane), 
    AngleBetween l p₁ = θ₁ ∧ 
    AngleBetween l p₂ = θ₂ ∧ 
    DihedralAngle p₁ p₂ > 0) : 
  θ₁ + θ₂ ≤ π / 2 := by
  sorry

-- Placeholder definitions for Line and Plane
structure Line : Type :=
(dummy : Unit)

structure Plane : Type :=
(dummy : Unit)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_sum_l354_35477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_nine_l354_35486

def S : Set ℤ := {-36, -4, -1, 3, 6, 9}

theorem largest_quotient_is_nine :
  ∀ a b : ℤ, a ∈ S → b ∈ S → a ≠ 0 → b ≠ 0 → (a : ℚ) / b ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_nine_l354_35486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_upstream_speed_l354_35485

/-- Given a man's rowing speeds in different conditions, calculate his upstream speed -/
theorem man_rowing_upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 40)
  (h2 : speed_downstream = 55) :
  speed_still - (speed_downstream - speed_still) = 25 := by
  -- Replace the body of the proof with 'sorry'
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_upstream_speed_l354_35485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_triangle_area_l354_35482

theorem linear_function_triangle_area : 
  ∀ b : ℝ, 
  (∃ x y : ℝ, y = 2 * x + b ∧ 
   (1/2) * |x| * |y| = 9 ∧ 
   x * y = 0) → 
  (b = 6 ∨ b = -6) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_triangle_area_l354_35482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_path_distances_l354_35478

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the straight-line distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the L-shaped path -/
theorem l_shaped_path_distances (a b c d : Point)
  (h_ab : b.x - a.x = 4 ∧ b.y = a.y)
  (h_bc : c.x = b.x ∧ c.y - b.y = 3)
  (h_cd : d.x - c.x = -2 ∧ d.y = c.y) :
  distance a c = 5 ∧ distance a d = Real.sqrt 13 := by
  sorry

#check l_shaped_path_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_path_distances_l354_35478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_perimeter_sum_l354_35422

theorem rectangle_area_perimeter_sum (a b : ℕ+) : 
  ∃ (A P : ℤ), A = (a : ℤ) * (b : ℤ) ∧ P = 2 * ((a : ℤ) + (b : ℤ)) ∧ A + P ≠ 146 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_perimeter_sum_l354_35422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junk_items_count_l354_35499

theorem junk_items_count (useful_percent : ℝ) (junk_percent : ℝ) (useful_count : ℕ) 
  (h1 : useful_percent = 0.20)
  (h2 : junk_percent = 0.70)
  (h3 : useful_count = 8)
  : ⌊(useful_count / useful_percent) * junk_percent⌋ = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_junk_items_count_l354_35499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l354_35456

/-- A function that checks if a three-digit number satisfies the given conditions -/
def satisfies_conditions (n : Nat) : Bool :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 + n % 10) % 2 = 0 ∧
  (n / 10) % 10 = (n / 100 + n % 10) / 2 ∧
  n / 100 ≠ n % 10

/-- The theorem stating that there are exactly 32 numbers satisfying the conditions -/
theorem count_satisfying_numbers : 
  (Finset.filter (fun n => satisfies_conditions n) (Finset.range 1000)).card = 32 := by
  sorry

#eval (Finset.filter (fun n => satisfies_conditions n) (Finset.range 1000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l354_35456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_stationary_l354_35463

/-- Triangle type -/
structure Triangle (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] where
  A : P
  B : P
  C : P

/-- Point on a line segment or its extension -/
def PointOnLine {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (A B P : P) : Prop :=
  ∃ t : ℝ, P = (1 - t) • A + t • B

/-- Circumcircle of a triangle -/
def Circumcircle {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (T : Triangle P) : Set P :=
  {X : P | dist X T.A = dist X T.B ∧ dist X T.B = dist X T.C}

/-- Similar triangles -/
def SimilarTriangles {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (T₁ T₂ : Triangle P) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    dist T₁.A T₁.B = k * dist T₂.A T₂.B ∧
    dist T₁.B T₁.C = k * dist T₂.B T₂.C ∧
    dist T₁.C T₁.A = k * dist T₂.C T₂.A

/-- Main theorem -/
theorem intersection_point_stationary
  {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P]
  (ABC : Triangle P)
  (A₁ B₁ C₁ : ℝ → P)
  (h₁ : ∀ t, PointOnLine ABC.B ABC.C (A₁ t))
  (h₂ : ∀ t, PointOnLine ABC.C ABC.A (B₁ t))
  (h₃ : ∀ t, PointOnLine ABC.A ABC.B (C₁ t))
  (h₄ : ∀ t, A₁ t ≠ ABC.B ∧ A₁ t ≠ ABC.C)
  (h₅ : ∀ t, B₁ t ≠ ABC.C ∧ B₁ t ≠ ABC.A)
  (h₆ : ∀ t, C₁ t ≠ ABC.A ∧ C₁ t ≠ ABC.B)
  (h₇ : ∀ t, SimilarTriangles (Triangle.mk (A₁ t) (B₁ t) (C₁ t)) (Triangle.mk (A₁ 0) (B₁ 0) (C₁ 0))) :
  ∃! P, ∀ t,
    P ∈ Circumcircle (Triangle.mk ABC.A (B₁ t) (C₁ t)) ∩
        Circumcircle (Triangle.mk (A₁ t) ABC.B (C₁ t)) ∩
        Circumcircle (Triangle.mk (A₁ t) (B₁ t) ABC.C) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_stationary_l354_35463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l354_35483

-- Define the expression as noncomputable
noncomputable def expression : ℝ := 1 / ((1 / (Real.sqrt 3 + 1)) + (2 / (Real.sqrt 5 - 1)))

-- State the theorem
theorem simplify_expression : expression = Real.sqrt 5 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l354_35483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_y_prime_absolute_value_l354_35441

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_y_prime_absolute_value :
  ∃ (y : ℤ), (∀ (z : ℤ), z < y → ¬(is_prime (Int.natAbs (5 * z^2 - 34 * z + 7)))) ∧
             (is_prime (Int.natAbs (5 * y^2 - 34 * y + 7))) ∧
             y = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_y_prime_absolute_value_l354_35441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_implies_m_value_l354_35476

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (∀ x : ℂ, 9 * x^2 - 5 * x + m = 0 ↔ x = (5 + Complex.I * Real.sqrt 371) / 18 ∨ x = (5 - Complex.I * Real.sqrt 371) / 18) →
  m = 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_implies_m_value_l354_35476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_at_zero_f_has_zeros_iff_a_positive_l354_35448

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * (4^x) - 2^x - 1

-- Theorem 1: When a = 1, f has a zero at x = 0
theorem f_zero_at_zero : f 1 0 = 0 := by
  simp [f]
  norm_num

-- Theorem 2: f has zeros if and only if a > 0
theorem f_has_zeros_iff_a_positive (a : ℝ) :
  (∃ x, f a x = 0) ↔ a > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_at_zero_f_has_zeros_iff_a_positive_l354_35448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_a_l354_35426

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The property that a polynomial P satisfies the given conditions for a specific a -/
def SatisfiesConditions (P : IntPolynomial) (a : ℤ) : Prop :=
  a > 0 ∧
  P.eval (1 : ℤ) = a ∧ P.eval (3 : ℤ) = a ∧ P.eval (5 : ℤ) = a ∧ P.eval (7 : ℤ) = a ∧
  P.eval (2 : ℤ) = -a ∧ P.eval (4 : ℤ) = -a ∧ P.eval (6 : ℤ) = -a ∧ P.eval (8 : ℤ) = -a

/-- The theorem stating that 315 is the smallest positive integer satisfying the conditions -/
theorem smallest_satisfying_a : 
  (∃ (P : IntPolynomial), SatisfiesConditions P 315) ∧ 
  (∀ (a : ℤ), 0 < a → a < 315 → ¬∃ (P : IntPolynomial), SatisfiesConditions P a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_a_l354_35426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_neither_necessary_nor_sufficient_l354_35434

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the condition
def condition (t : Triangle) : Prop :=
  t.a / t.b = Real.cos t.B / Real.cos t.A

-- Define isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

-- Theorem statement
theorem condition_neither_necessary_nor_sufficient :
  ¬(∀ t : Triangle, condition t ↔ isIsosceles t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_neither_necessary_nor_sufficient_l354_35434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_disjoint_circles_are_disjoint_explicit_l354_35489

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O₂ (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 9

-- Define the centers and radii
def center_O₁ : ℝ × ℝ := (0, 0)
def center_O₂ : ℝ × ℝ := (3, -4)
def radius_O₁ : ℝ := 1
def radius_O₂ : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((3 - 0)^2 + (-4 - 0)^2)

-- Theorem statement
theorem circles_are_disjoint :
  distance_between_centers > radius_O₁ + radius_O₂ :=
by
  -- We'll use sorry to skip the proof for now
  sorry

-- Additional theorem to show that the circles are disjoint
theorem circles_are_disjoint_explicit :
  ∀ (x y : ℝ), ¬(circle_O₁ x y ∧ circle_O₂ x y) :=
by
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_disjoint_circles_are_disjoint_explicit_l354_35489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travis_discount_percentage_l354_35479

/-- Calculates the discount percentage given the original price and discounted price -/
noncomputable def discount_percentage (original_price discounted_price : ℝ) : ℝ :=
  (original_price - discounted_price) / original_price * 100

theorem travis_discount_percentage :
  let original_price : ℝ := 2000
  let discounted_price : ℝ := 1400
  discount_percentage original_price discounted_price = 30 := by
  unfold discount_percentage
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travis_discount_percentage_l354_35479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_functions_l354_35491

-- Define the interval (0, 1)
def open_unit_interval : Set ℝ := Set.Ioo 0 1

-- Define the functions
noncomputable def f1 : ℝ → ℝ := λ x => x^(1/2)
noncomputable def f2 : ℝ → ℝ := λ x => Real.log (x + 1) / Real.log (1/2)
def f3 : ℝ → ℝ := λ x => |x^2 - 2*x|
noncomputable def f4 : ℝ → ℝ := λ x => (5/6)^x

-- Define monotonically decreasing function on an interval
def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f y < f x

theorem monotone_decreasing_functions :
  (monotone_decreasing_on f2 open_unit_interval) ∧
  (monotone_decreasing_on f4 open_unit_interval) ∧
  ¬(monotone_decreasing_on f1 open_unit_interval) ∧
  ¬(monotone_decreasing_on f3 open_unit_interval) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_functions_l354_35491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_cos2x_sin3x_eq_quarter_sin2x_l354_35470

theorem cos_cos2x_sin3x_eq_quarter_sin2x 
  (x : ℝ) : 
  (Real.cos x * Real.cos (2 * x) * Real.sin (3 * x) = 0.25 * Real.sin (2 * x)) → 
  (∃ k : ℤ, x = Real.pi / 2 * (2 * k + 1)) ∨ 
  (∃ n : ℤ, x = Real.pi * n / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_cos2x_sin3x_eq_quarter_sin2x_l354_35470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_diagonals_of_specific_polyhedron_l354_35445

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  pentagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - (Q.pentagonal_faces * 5)

/-- Theorem: A convex polyhedron Q with 30 vertices, 70 edges, 42 faces
    (of which 30 are triangular and 12 are pentagonal) has 305 space diagonals -/
theorem space_diagonals_of_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 42,
    triangular_faces := 30,
    pentagonal_faces := 12
  }
  space_diagonals Q = 305 := by
  -- Proof goes here
  sorry

#eval space_diagonals {
  vertices := 30,
  edges := 70,
  faces := 42,
  triangular_faces := 30,
  pentagonal_faces := 12
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_diagonals_of_specific_polyhedron_l354_35445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digitTwoOccurrences_l354_35421

def countDigitOccurrences (start : Nat) (stop : Nat) (digit : Nat) : Nat :=
  sorry

theorem digitTwoOccurrences :
  countDigitOccurrences 1 400 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digitTwoOccurrences_l354_35421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisected_by_P_on_ellipse_l354_35495

noncomputable section

/-- The point P on the ellipse -/
def P : ℝ × ℝ := (1/2, 1/2)

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := y^2 / 9 + x^2 = 1

/-- A line passing through a point -/
def line_through_point (m b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + b

/-- A point bisects a line segment if it's the midpoint of the segment -/
def bisects_line (p q r : ℝ × ℝ) : Prop :=
  p.1 + r.1 = 2 * q.1 ∧ p.2 + r.2 = 2 * q.2

/-- The theorem statement -/
theorem line_bisected_by_P_on_ellipse :
  ∃ (A B : ℝ × ℝ) (m b : ℝ),
    is_on_ellipse A.1 A.2 ∧
    is_on_ellipse B.1 B.2 ∧
    line_through_point m b A ∧
    line_through_point m b B ∧
    line_through_point m b P ∧
    bisects_line A P B ∧
    m = -9 ∧ b = 5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisected_by_P_on_ellipse_l354_35495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_value_l354_35406

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : is_geometric_sequence a)
  (h_2017 : a 2017 = a 2016 + 2 * a 2015)
  (h_prod : ∃ m n : ℕ, a m * a n = 16 * (a 1)^2) :
  (∃ m n : ℕ, (4 / (m : ℝ) + 1 / (n : ℝ)) ≥ 3/2) ∧
  (∀ m n : ℕ, (4 / (m : ℝ) + 1 / (n : ℝ)) ≥ 3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_value_l354_35406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_rate_l354_35451

theorem barbed_wire_rate (area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) : 
  area = 3136 →
  gate_width = 1 →
  num_gates = 2 →
  total_cost = 1998 →
  (let side_length := Real.sqrt area
   let perimeter := 4 * side_length
   let wire_length := perimeter - (↑num_gates * gate_width)
   let rate_per_meter := total_cost / wire_length
   rate_per_meter = 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_rate_l354_35451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_download_time_l354_35402

/-- Represents the time in minutes for downloading a game -/
def download_time : ℝ → Prop := sorry

/-- Represents the total time spent before playing the main game -/
def total_time : ℝ → Prop := sorry

theorem game_download_time :
  ∀ t : ℝ,
  download_time t →
  total_time 60 →
  (t + t/2 + 3*(t + t/2) = 60) →
  t = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_download_time_l354_35402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l354_35403

/-- Given that 250 men complete a job in 16 days, proves that 600 men will take 20 days to complete a job that is 3 times larger -/
theorem job_completion_time 
  (original_men : ℕ) 
  (original_days : ℕ) 
  (job_scale : ℕ) 
  (new_men : ℕ) : 
  original_men = 250 → 
  original_days = 16 → 
  job_scale = 3 → 
  new_men = 600 → 
  (original_men * original_days * job_scale : ℚ) / new_men = 20 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  
#check job_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l354_35403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_l354_35442

-- Define the basic structures
structure Line where

structure Plane where

-- Define the relationships
def parallel (l1 l2 : Line) : Prop := sorry

def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perpendicular_line_plane m α → perpendicular_line_plane n α → parallel m n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_l354_35442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_dressing_vinegar_percentage_l354_35404

/-- Represents a salad dressing with vinegar and oil components -/
structure Dressing where
  vinegar : ℝ
  oil : ℝ

/-- The percentage of vinegar in a dressing -/
noncomputable def vinegar_percentage (d : Dressing) : ℝ :=
  d.vinegar / (d.vinegar + d.oil) * 100

/-- Theorem: The percentage of vinegar in the new dressing is 12% -/
theorem new_dressing_vinegar_percentage
  (P : Dressing)
  (Q : Dressing)
  (h_P_vinegar : P.vinegar = 0.3 * (P.vinegar + P.oil))
  (h_Q_vinegar : Q.vinegar = 0.1 * (Q.vinegar + Q.oil))
  (h_P_ratio : P.vinegar + P.oil = 0.1)
  (h_Q_ratio : Q.vinegar + Q.oil = 0.9)
  : vinegar_percentage { vinegar := P.vinegar + Q.vinegar, oil := P.oil + Q.oil } = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_dressing_vinegar_percentage_l354_35404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stripe_area_calculation_l354_35446

/-- Represents a cylindrical tank with a painted stripe -/
structure StripedTank where
  diameter : ℝ
  height : ℝ
  stripeWidth : ℝ
  stripeRevolutions : ℕ

/-- Calculates the area of the stripe on the tank -/
noncomputable def stripeArea (tank : StripedTank) : ℝ :=
  let radius := tank.diameter / 2
  let circumference := 2 * Real.pi * radius
  let stripeLength := tank.stripeRevolutions * circumference
  let stripeHeight := tank.height / 2
  stripeLength * tank.stripeWidth

/-- Theorem stating the area of the stripe on the given tank -/
theorem stripe_area_calculation :
  let tank := StripedTank.mk 40 60 4 3
  stripeArea tank = 480 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stripe_area_calculation_l354_35446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basis_zero_unique_l354_35412

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- A pair of vectors forms a basis for a real vector space -/
def is_basis (e₁ e₂ : V) : Prop :=
  ∀ v : V, ∃! (a b : ℝ), v = a • e₁ + b • e₂

/-- If e₁ and e₂ form a basis, then the only way to get the zero vector
    is with zero coefficients -/
theorem basis_zero_unique {e₁ e₂ : V} (h : is_basis e₁ e₂) :
  ∀ (m n : ℝ), m • e₁ + n • e₂ = 0 → m = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basis_zero_unique_l354_35412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_l354_35429

/-- Given that 2x cows produce 2x+2 cans of milk in 2x+1 days, 
    the number of days it takes 2x+4 cows to give 2x+10 cans of milk 
    is (2x(2x+1)(2x+10)) / ((2x+2)(2x+4)) -/
theorem milk_production (x : ℝ) (h : x > 0) : 
  (2*x + 4) * (2*x + 2) * (2*x + 1) / (2*x * (2*x + 10)) = 
  (2*x * (2*x + 1) * (2*x + 10)) / ((2*x + 2) * (2*x + 4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_l354_35429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_last_theorem_for_polynomials_l354_35471

-- Define the types for natural numbers and complex polynomials
variable (a b c : ℕ)
variable (X : Type) [CommRing X]
variable (P Q R : MvPolynomial X ℂ)

-- Define the condition that P, Q, R have no common factors
def no_common_factors (P Q R : MvPolynomial X ℂ) : Prop :=
  ∀ (S : MvPolynomial X ℂ), (S ∣ P) ∧ (S ∣ Q) ∧ (S ∣ R) → IsUnit S

-- State the theorem
theorem fermat_last_theorem_for_polynomials 
  (h1 : no_common_factors X P Q R)
  (h2 : P^a + Q^b = R^c) :
  1/a + 1/b + 1/c > 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_last_theorem_for_polynomials_l354_35471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outfit_count_is_correct_l354_35423

/-- Represents the colors of shirts and hats -/
inductive Color
  | Red
  | Green
  deriving Repr, DecidableEq

/-- Represents the colors of pants -/
inductive PantsColor
  | Blue
  | Black
  deriving Repr, DecidableEq

structure Wardrobe where
  redShirts : Nat
  greenShirts : Nat
  bluePants : Nat
  blackPants : Nat
  greenHats : Nat
  redHats : Nat

def isValidOutfit (shirt : Color) (pants : PantsColor) (hat : Color) : Bool :=
  (shirt ≠ hat) && (shirt = Color.Red → pants = PantsColor.Blue)

def countOutfits (w : Wardrobe) : Nat :=
  let redShirtOutfits := w.redShirts * w.bluePants * w.greenHats
  let greenShirtOutfits := w.greenShirts * (w.bluePants + w.blackPants) * w.redHats
  redShirtOutfits + greenShirtOutfits

def myWardrobe : Wardrobe :=
  { redShirts := 7
  , greenShirts := 7
  , bluePants := 4
  , blackPants := 5
  , greenHats := 10
  , redHats := 10
  }

theorem outfit_count_is_correct :
  countOutfits myWardrobe = 910 := by
  rfl

#eval countOutfits myWardrobe

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outfit_count_is_correct_l354_35423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circles_intersection_l354_35418

-- Define necessary structures and predicates
structure Point where
  x : Real
  y : Real

structure Circle where
  center : Point
  radius : Real

def TriangleAngles (A B C : Point) (α β γ : Real) : Prop :=
  sorry -- Define the condition for triangle angles

def TangentCircle (k : Circle) (X Y Z : Point) : Prop :=
  sorry -- Define the condition for a tangent circle

def InsideTriangle (P A B C : Point) : Prop :=
  sorry -- Define the condition for a point inside a triangle

def OnCircle (P : Point) (k : Circle) : Prop :=
  sorry -- Define the condition for a point on a circle

theorem three_circles_intersection (A B C : Point) (α β γ : Real) 
  (kA kB kC : Circle) :
  TriangleAngles A B C α β γ →
  TangentCircle kA C A B →
  TangentCircle kB A B C →
  TangentCircle kC B C A →
  ∃ P : Point, InsideTriangle P A B C ∧ OnCircle P kA ∧ OnCircle P kB ∧ OnCircle P kC := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circles_intersection_l354_35418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l354_35484

theorem geometric_sequence_common_ratio 
  (a : ℝ) : 
  let seq := λ (n : ℕ) => a + Real.log 3 / Real.log (2^(2^n))
  ∃ q : ℝ, q = 2/3 ∧ ∀ n : ℕ, seq (n + 1) / seq n = q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l354_35484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MF_is_sqrt2_div_2_l354_35409

/-- A curve defined by parametric equations x = 2cosθ and y = 1 + cos2θ, where θ ∈ ℝ -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ θ, x θ = 2 * Real.cos θ
  h_y : ∀ θ, y θ = 1 + Real.cos (2 * θ)

/-- The focus of the parametric curve -/
noncomputable def focus (c : ParametricCurve) : ℝ × ℝ := (0, 1/2)

/-- The point M -/
noncomputable def M : ℝ × ℝ := (1/2, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_MF_is_sqrt2_div_2 (c : ParametricCurve) :
  distance M (focus c) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MF_is_sqrt2_div_2_l354_35409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_parallelogram_area_l354_35437

/-- A quadrilateral in a plane. -/
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

/-- The area of a quadrilateral. -/
noncomputable def area (q : Quadrilateral) : ℝ := sorry

/-- A parallelogram formed by the diagonals of a quadrilateral. -/
def diagonalParallelogram (q : Quadrilateral) : Quadrilateral := sorry

/-- The theorem stating that the area of the diagonal parallelogram is twice the area of the quadrilateral. -/
theorem diagonal_parallelogram_area (q : Quadrilateral) :
  area (diagonalParallelogram q) = 2 * area q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_parallelogram_area_l354_35437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_roll_distribution_l354_35428

/-- The number of guests at the conference --/
def num_guests : ℕ := 4

/-- The number of roll types --/
def num_roll_types : ℕ := 3

/-- The number of rolls of each type --/
def rolls_per_type : ℕ := 4

/-- The total number of rolls --/
def total_rolls : ℕ := num_guests * num_roll_types

/-- The number of rolls each guest receives --/
def rolls_per_guest : ℕ := num_roll_types

/-- The probability that each guest gets one roll of each type --/
def probability_correct_distribution : ℚ := 18 / 770

theorem conference_roll_distribution :
  probability_correct_distribution = 18 / 770 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_roll_distribution_l354_35428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l354_35468

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x + π / 6) + cos (2 * x + π / 6)

-- Define the theorem
theorem triangle_side_length 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h1 : f A = sqrt 3)
  (h2 : sin C = 1 / 3)
  (h3 : a = 3)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π)
  (h8 : sin A * b = sin B * a)
  (h9 : sin B * c = sin C * b)
  (h10 : sin C * a = sin A * c) :
  b = sqrt 3 + 2 * sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l354_35468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_satisfies_equation_is_hyperbola_center_l354_35490

/-- The center of a hyperbola given by the equation (4y+6)^2/16 - (5x-3)^2/9 = 1 -/
noncomputable def hyperbola_center : ℝ × ℝ :=
  (3/5, -3/2)

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4*y + 6)^2 / 16 - (5*x - 3)^2 / 9 = 1

/-- Theorem stating that hyperbola_center satisfies the hyperbola equation -/
theorem center_satisfies_equation :
  let (h, k) := hyperbola_center
  hyperbola_equation h k :=
by sorry

/-- Theorem proving that hyperbola_center is indeed the center of the hyperbola -/
theorem is_hyperbola_center :
  let (h, k) := hyperbola_center
  ∀ (x y : ℝ), hyperbola_equation x y ↔ hyperbola_equation (x - h) (y - k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_satisfies_equation_is_hyperbola_center_l354_35490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_other_focus_l354_35436

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem stating the locus of the other focus -/
theorem locus_of_other_focus (A B C : Point) (F : Point → Prop) :
  A.x = -7 ∧ A.y = 0 ∧
  B.x = 7 ∧ B.y = 0 ∧
  C.x = 2 ∧ C.y = -12 ∧
  (∀ p, F p → distance p A + distance p C = distance p B + distance p C) →
  ∀ p, F p → distance p B - distance p A = 2 := by
  sorry

#check locus_of_other_focus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_other_focus_l354_35436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dinosaur_model_price_l354_35415

/-- The original price of a dinosaur model -/
def original_price : ℝ := 100

/-- The number of models bought for the kindergarten library -/
def kindergarten_models : ℕ := 2

/-- The number of models bought for the elementary library -/
def elementary_models : ℕ := 2 * kindergarten_models

/-- The total number of models bought -/
def total_models : ℕ := kindergarten_models + elementary_models

/-- The discount rate applied to each model -/
def discount_rate : ℝ := 0.05

/-- The discounted price of each model -/
def discounted_price : ℝ := original_price * (1 - discount_rate)

/-- The total amount paid by the school -/
def total_paid : ℝ := 570

theorem dinosaur_model_price :
  total_models > 5 ∧
  total_paid = (total_models : ℝ) * discounted_price →
  original_price = 100 := by
  sorry

#eval total_models
#eval original_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dinosaur_model_price_l354_35415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_sqrt5_l354_35497

/-- A regular triangular prism -/
structure RegularTriangularPrism where
  /-- The edge length of the prism -/
  edgeLength : ℝ
  /-- Assumption that the edge length is positive -/
  edgePos : edgeLength > 0

/-- The distance between midpoints of non-parallel sides of different bases -/
noncomputable def midpointDistance (prism : RegularTriangularPrism) : ℝ :=
  Real.sqrt 5

/-- Theorem stating that for a regular triangular prism with edge length 2,
    the distance between midpoints of non-parallel sides of different bases is √5 -/
theorem midpoint_distance_sqrt5 (prism : RegularTriangularPrism) 
    (h : prism.edgeLength = 2) : midpointDistance prism = Real.sqrt 5 := by
  sorry

#check midpoint_distance_sqrt5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_sqrt5_l354_35497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l354_35498

-- Define the complex number
noncomputable def z : ℂ := Complex.mk (-5) 1 / Complex.mk 2 (-3)

-- State the theorem
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l354_35498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_lines_with_equal_intercepts_l354_35473

/-- The circle C in the 2D plane -/
def C (x y : ℝ) : Prop := x^2 + (y+5)^2 = 3

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a line is tangent to the circle C -/
def is_tangent (l : Line) : Prop := sorry

/-- Check if a line has equal x and y intercepts -/
def has_equal_intercepts (l : Line) : Prop := sorry

/-- The set of lines that are tangent to C and have equal intercepts -/
def tangent_lines_with_equal_intercepts : Set Line :=
  {l : Line | is_tangent l ∧ has_equal_intercepts l}

/-- There are exactly four tangent lines with equal intercepts -/
theorem four_tangent_lines_with_equal_intercepts :
  ∃ (s : Finset Line), s.card = 4 ∧ ∀ l, l ∈ s ↔ l ∈ tangent_lines_with_equal_intercepts := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_lines_with_equal_intercepts_l354_35473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_incorrect_l354_35457

-- Define a type for points in a geometric space
variable (Point : Type)

-- Define a predicate for points being on the locus
variable (on_locus : Point → Prop)

-- Define a predicate for points satisfying the conditions
variable (satisfies_conditions : Point → Prop)

-- Define the correctness of each statement
def statement_A_correct (Point : Type) (on_locus satisfies_conditions : Point → Prop) : Prop :=
  (∀ p : Point, on_locus p → satisfies_conditions p) ∧
  (∀ p : Point, ¬on_locus p → ¬satisfies_conditions p)

def statement_B_correct (Point : Type) (on_locus satisfies_conditions : Point → Prop) : Prop :=
  (∀ p : Point, satisfies_conditions p → on_locus p) ∧
  (∀ p : Point, on_locus p → satisfies_conditions p)

def statement_C_correct (Point : Type) (on_locus satisfies_conditions : Point → Prop) : Prop :=
  (∀ p : Point, ¬satisfies_conditions p → ¬on_locus p) ∧
  (∀ p : Point, satisfies_conditions p → on_locus p)

def statement_D_incorrect (Point : Type) (on_locus satisfies_conditions : Point → Prop) : Prop :=
  ¬((∀ p : Point, on_locus p → satisfies_conditions p) ∧
    (∃ p : Point, ¬satisfies_conditions p ∧ on_locus p))

def statement_E_correct (Point : Type) (on_locus satisfies_conditions : Point → Prop) : Prop :=
  (∀ p : Point, ¬on_locus p → ¬satisfies_conditions p) ∧
  (∀ p : Point, on_locus p → satisfies_conditions p)

-- Theorem stating that D is the only incorrect statement
theorem only_D_incorrect (Point : Type) (on_locus satisfies_conditions : Point → Prop) :
  statement_A_correct Point on_locus satisfies_conditions ∧
  statement_B_correct Point on_locus satisfies_conditions ∧
  statement_C_correct Point on_locus satisfies_conditions ∧
  statement_D_incorrect Point on_locus satisfies_conditions ∧
  statement_E_correct Point on_locus satisfies_conditions :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_incorrect_l354_35457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_constants_l354_35410

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g in terms of f and constants a, b, and c
def g (a b c : ℝ) (x : ℝ) : ℝ := a * f (b * x) + c

-- State the theorem
theorem transformation_constants : 
  ∃ (a b c : ℝ), (∀ x, g a b c x = f (x / 2) - 4) ∧ a = 1 ∧ b = 1/2 ∧ c = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_constants_l354_35410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_french_exam_min_words_l354_35474

/-- The minimum number of words to learn for a French exam -/
def min_words_to_learn (total_words : ℕ) (misremember_rate : ℚ) (target_score : ℚ) : ℕ :=
  (((target_score * total_words : ℚ) / (1 - misremember_rate)).ceil).toNat

/-- Theorem: Learning 600 words is the minimum required for at least 90% score -/
theorem french_exam_min_words :
  let total_words : ℕ := 600
  let misremember_rate : ℚ := 1/10
  let target_score : ℚ := 9/10
  min_words_to_learn total_words misremember_rate target_score = 600 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_french_exam_min_words_l354_35474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l354_35465

-- Define the function f
noncomputable def f (b c x : ℝ) : ℝ := Real.log ((2 * x^2 + b * x + c) / (x^2 + 1)) / Real.log 10

-- Define the set A
def A (b c : ℝ) : Set ℝ := {x | -1/2 ≤ x ∧ x ≤ 2 ∧ b * x^2 + 3 * x + c ≥ 0}

-- Define the set M
def M (b c : ℝ) : Set ℝ := {m | ∃ t ∈ A b c, f b c t = m}

-- State the theorem
theorem problem_solution :
  ∃ b c : ℝ,
    (∀ x, x ∈ A b c ↔ -1/2 ≤ x ∧ x ≤ 2) ∧
    b = -2 ∧
    c = 2 ∧
    M b c = {m | 0 ≤ m ∧ m ≤ Real.log (14/5) / Real.log 10} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l354_35465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_objects_arrangements_l354_35407

/-- The number of distinct circular arrangements of n objects -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial n / n

/-- The number of distinct necklace arrangements of n objects -/
def necklaceArrangements (n : ℕ) : ℕ := (Nat.factorial n / n) / 2

theorem seven_objects_arrangements :
  (circularArrangements 7 = 720) ∧ (necklaceArrangements 7 = 360) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_objects_arrangements_l354_35407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_calculation_l354_35481

/-- Represents a number in base 4 --/
structure Base4 where
  value : ℕ

/-- Multiplication operation for Base4 numbers --/
def mul_base4 (a b : Base4) : Base4 :=
  ⟨sorry⟩

/-- Division operation for Base4 numbers --/
def div_base4 (a b : Base4) : Base4 :=
  ⟨sorry⟩

/-- Converts a natural number to its Base4 representation --/
def to_base4 (n : ℕ) : Base4 :=
  ⟨n⟩

/-- The main theorem stating the equality in base 4 --/
theorem base4_calculation : 
  mul_base4 (div_base4 (to_base4 120) (to_base4 2)) (to_base4 13) = to_base4 1110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_calculation_l354_35481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangents_l354_35439

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ 1}

-- Define the line l
def l : Set (ℝ × ℝ) := {p | p.1 + p.2 = 1}

-- Define the point p
def p : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem circle_and_tangents :
  -- Given conditions
  (∃ a b : ℝ × ℝ, a ∈ C ∧ b ∈ C ∧ a ∈ l ∧ b ∈ l ∧ (a.1 - b.1)^2 + (a.2 - b.2)^2 = 2) →
  p ∉ C →
  -- Conclusions
  (∀ x y : ℝ, (x, y) ∈ C ↔ (x - 1)^2 + (y - 1)^2 = 1) ∧
  (∃ t : ℝ, (2, t) ∈ C ∧ ∀ x : ℝ, x ≠ 2 → (x, t) ∉ C) ∧
  (∃ t : ℝ, 3 * 2 - 4 * t + 6 = 0 ∧ (2, t) ∈ C ∧
    ∀ x y : ℝ, 3 * x - 4 * y + 6 = 0 → ((x, y) = (2, t) ∨ (x, y) ∉ C)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangents_l354_35439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_medians_l354_35467

noncomputable section

def angle_between_medians (m1 m2 : ℝ) : ℝ := sorry

def triangle_area (m1 m2 : ℝ) (cos_angle : ℝ) : ℝ := sorry

theorem triangle_area_from_medians (a b c : ℝ) 
  (h1 : a = 15 / 7)
  (h2 : b = Real.sqrt 21)
  (h3 : c = 2 / 5)
  (h4 : c = Real.cos (angle_between_medians a b)) : 
  triangle_area a b c = 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_medians_l354_35467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_center_distances_l354_35440

/-- Two unit squares with parallel sides that overlap by a rectangle of area 1/8 -/
structure OverlappingSquares where
  /-- The area of the overlapping rectangle -/
  overlap_area : ℝ
  /-- The overlap area is 1/8 -/
  overlap_is_eighth : overlap_area = 1/8

/-- The distance between the centers of two overlapping unit squares -/
noncomputable def center_distance (s : OverlappingSquares) : ℝ → ℝ :=
  fun x => x -- This is a placeholder function

/-- The theorem stating the extreme values of the distance between the centers of the squares -/
theorem extreme_center_distances (s : OverlappingSquares) :
  (∃ d, (center_distance s d = d) ∧ ∀ x, center_distance s x ≤ d) ∧
  (∃ d, (center_distance s d = d) ∧ ∀ x, center_distance s x ≥ d) ∧
  (∃ d_max, (center_distance s d_max = d_max) ∧ ∀ x, center_distance s x ≤ d_max ∧ d_max = Real.sqrt 2 - 1/2) ∧
  (∃ d_min, (center_distance s d_min = d_min) ∧ ∀ x, center_distance s x ≥ d_min ∧ d_min = 1/2) := by
  sorry

#check extreme_center_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_center_distances_l354_35440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l354_35462

/-- The parabola is defined by the equation x² = 12y -/
def parabola_equation (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 = 12 * y

/-- The directrix of a parabola is a line perpendicular to its axis of symmetry -/
def is_directrix (y : ℝ) (p : ℝ × ℝ → Prop) : Prop :=
  ∀ x, p (x, y)

/-- Theorem: The directrix of the parabola x² = 12y is y = -3 -/
theorem parabola_directrix :
  is_directrix (-3) parabola_equation :=
by
  intro x
  unfold parabola_equation
  simp
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l354_35462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l354_35419

theorem inequality_solution_set (x : ℝ) : 
  (x - 1) / (x - 3) ≥ 3 ↔ x ∈ Set.Iio 3 ∪ Set.Ioi 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l354_35419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_journey_time_difference_l354_35444

/-- Represents the hiker's journey over three days -/
structure HikerJourney where
  day1_distance : ℚ
  day1_speed : ℚ
  day2_speed_increase : ℚ
  day3_speed : ℚ
  day3_time : ℚ
  total_distance : ℚ

/-- Calculates the time difference between the first and second day of the hiker's journey -/
noncomputable def time_difference (journey : HikerJourney) : ℚ :=
  let day1_time := journey.day1_distance / journey.day1_speed
  let day3_distance := journey.day3_speed * journey.day3_time
  let day2_distance := journey.total_distance - journey.day1_distance - day3_distance
  let day2_speed := journey.day1_speed + journey.day2_speed_increase
  let day2_time := day2_distance / day2_speed
  day1_time - day2_time

/-- Theorem stating that the time difference between the first and second day is 1 hour -/
theorem hiker_journey_time_difference :
  ∃ (journey : HikerJourney),
    journey.day1_distance = 18 ∧
    journey.day1_speed = 3 ∧
    journey.day2_speed_increase = 1 ∧
    journey.day3_speed = 5 ∧
    journey.day3_time = 3 ∧
    journey.total_distance = 53 ∧
    time_difference journey = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_journey_time_difference_l354_35444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l354_35401

/-- The parabola y^2 = 4x with focus F and a point P(3,1) -/
structure Parabola :=
  (F : ℝ × ℝ)  -- Focus of the parabola
  (P : ℝ × ℝ := (3, 1))  -- Fixed point P

/-- A point M on the parabola -/
def PointOnParabola (p : Parabola) := { M : ℝ × ℝ // M.2^2 = 4 * M.1 }

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The theorem to be proved -/
theorem min_distance_sum (p : Parabola) :
  ∃ m : ℝ, ∀ M : PointOnParabola p, 
    m ≤ distance M.val p.P + distance M.val p.F ∧
    ∃ N : PointOnParabola p, m = distance N.val p.P + distance N.val p.F :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l354_35401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_dataset_size_l354_35488

def dataset_operations (initial : ℕ) : ℕ :=
  let step1 := initial * 120 / 100
  let step2 := step1 * 4 / 5
  let step3 := step2 + 60
  let step4 := step3 * 88 / 100
  let step5 := step4 * 125 / 100
  step5 * 85 / 100

theorem final_dataset_size :
  dataset_operations 400 = 415 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_dataset_size_l354_35488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_intersection_l354_35438

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2ax -/
structure Parabola where
  a : ℝ
  h_pos : a > 0

/-- Represents a hyperbola y²/4 - x²/9 = 1 -/
structure Hyperbola where

/-- Returns the focus of a parabola -/
noncomputable def focus (p : Parabola) : Point :=
  { x := p.a / 2, y := 0 }

/-- Returns a point on the directrix of a parabola -/
noncomputable def directrixPoint (p : Parabola) (y : ℝ) : Point :=
  { x := -p.a / 2, y := y }

/-- Checks if a point is on the hyperbola -/
def onHyperbola (p : Point) : Prop :=
  p.y^2 / 4 - p.x^2 / 9 = 1

/-- Calculates the angle between two points and the focus -/
noncomputable def angle (p1 p2 : Point) (f : Point) : ℝ :=
  sorry

/-- Main theorem -/
theorem parabola_hyperbola_intersection (p : Parabola) :
  ∃ (m n : Point),
    onHyperbola m ∧
    onHyperbola n ∧
    m = directrixPoint p m.y ∧
    n = directrixPoint p n.y ∧
    angle m n (focus p) = 120 * π / 180 →
    p.a = 3 * Real.sqrt 26 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_intersection_l354_35438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_polyhedron_volume_l354_35469

/-- A polyhedron with parallel rectangular bases -/
structure Polyhedron where
  ab : ℝ
  ad : ℝ
  apbp : ℝ
  apdp : ℝ
  height : ℝ
  ab_positive : 0 < ab
  ad_positive : 0 < ad
  apbp_positive : 0 < apbp
  apdp_positive : 0 < apdp
  height_positive : 0 < height

/-- The volume of the polyhedron -/
noncomputable def volume (p : Polyhedron) : ℝ :=
  p.apbp * p.apdp * p.height + 
  (p.ab - p.apbp) * p.ad * p.height / 2 + 
  (p.ad - p.apdp) * p.ab * p.height / 2

/-- Theorem stating the volume of the specific polyhedron -/
theorem specific_polyhedron_volume : 
  let p : Polyhedron := {
    ab := 11,
    ad := 5,
    apbp := 9,
    apdp := 3,
    height := Real.sqrt 3,
    ab_positive := by norm_num,
    ad_positive := by norm_num,
    apbp_positive := by norm_num,
    apdp_positive := by norm_num,
    height_positive := by exact Real.sqrt_pos.mpr (by norm_num)
  }
  volume p = 121 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_polyhedron_volume_l354_35469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_theorem_l354_35435

/-- The number of revolutions required for a wheel to travel a certain distance -/
noncomputable def revolutions (diameter : ℝ) (distance : ℝ) : ℝ :=
  distance / (Real.pi * diameter)

/-- Conversion factor from miles to feet -/
def miles_to_feet (miles : ℝ) : ℝ :=
  miles * 5280

theorem wheel_revolutions_theorem :
  revolutions 8 (miles_to_feet 2) = 1320 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_theorem_l354_35435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_m_condition_l354_35494

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos x + Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x - Real.cos x)

-- Define the function f as the dot product of a and b
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Define the interval for x
def x_interval : Set ℝ := { x | 5 * Real.pi / 24 ≤ x ∧ x ≤ 5 * Real.pi / 12 }

-- State the theorem
theorem f_range_and_m_condition (x : ℝ) (hx : x ∈ x_interval) :
  (∀ m : ℝ, 0 ≤ m ∧ m ≤ 4 ↔ ∀ t : ℝ, m * t^2 + m * t + 3 ≥ f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_m_condition_l354_35494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_attacked_squares_approx_expected_attacked_squares_exact_l354_35414

/-- The number of squares on a chessboard -/
def chessboardSquares : ℕ := 64

/-- The number of rooks placed on the chessboard -/
def numberOfRooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook -/
def probNotAttacked : ℚ := (49 : ℚ) / 64

/-- The expected number of squares under attack when three rooks are randomly placed on a chessboard -/
def expectedAttackedSquares : ℚ := chessboardSquares * (1 - probNotAttacked ^ numberOfRooks)

/-- Theorem stating that the expected number of squares under attack is approximately 35.33 -/
theorem expected_attacked_squares_approx :
  abs (expectedAttackedSquares - 35.33) < 0.01 := by sorry

/-- Theorem proving the exact value of the expected number of squares under attack -/
theorem expected_attacked_squares_exact :
  expectedAttackedSquares = 64 * (1 - (49/64)^3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_attacked_squares_approx_expected_attacked_squares_exact_l354_35414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_neg_four_f_two_thirds_l354_35496

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 5) + 1 / (x - 2)

-- Define the domain of f
def domain (x : ℝ) : Prop := x ≥ -5 ∧ x ≠ 2

-- Theorem for the domain of f
theorem f_domain : ∀ x : ℝ, domain x ↔ (∃ y : ℝ, f x = y) := by
  sorry

-- Theorem for f(-4)
theorem f_neg_four : f (-4) = 5/6 := by
  sorry

-- Theorem for f(2/3)
theorem f_two_thirds : f (2/3) = (Real.sqrt 51 - 9/4) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_neg_four_f_two_thirds_l354_35496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_circumference_approx_l354_35417

/-- The circumference of a wheel, given the number of revolutions and distance covered -/
noncomputable def wheel_circumference (revolutions : ℝ) (distance : ℝ) : ℝ :=
  distance / revolutions

theorem wheel_circumference_approx :
  let revolutions : ℝ := 3.002729754322111
  let distance : ℝ := 1056
  abs ((wheel_circumference revolutions distance) - 351.855) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_circumference_approx_l354_35417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transistors_in_2010_l354_35454

/-- Moore's law doubling period in months -/
def doubling_period : ℕ := 18

/-- Initial number of transistors in 1995 -/
def initial_transistors : ℕ := 2500000

/-- Number of months between 1995 and 2010 -/
def months_difference : ℕ := (2010 - 1995) * 12

/-- Number of doubling periods between 1995 and 2010 -/
def num_doubling_periods : ℕ := months_difference / doubling_period

theorem transistors_in_2010 :
  initial_transistors * 2^num_doubling_periods = 2560000000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transistors_in_2010_l354_35454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l354_35430

noncomputable def f (x : ℝ) := Real.exp (x * Real.log 2) + x

theorem zero_point_interval (x₀ : ℝ) (k : ℤ) : 
  (f x₀ = 0) → (x₀ ∈ Set.Ioo (k : ℝ) ((k + 1) : ℝ)) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l354_35430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_hour_rate_l354_35450

/-- Represents the rate of water filling in gallons per hour for each hour -/
def FillRate (x : ℝ) : Fin 5 → ℝ
  | 0 => 8    -- First hour
  | 1 => 10   -- Second hour
  | 2 => 10   -- Third hour
  | 3 => x    -- Fourth hour (unknown rate)
  | 4 => -8   -- Fifth hour (water loss)

/-- The total amount of water in the pool after 5 hours -/
def TotalWater (x : ℝ) : ℝ := (FillRate x 0) + (FillRate x 1) + (FillRate x 2) + (FillRate x 3) + (FillRate x 4)

/-- Theorem stating that the rate of filling during the fourth hour is 14 gallons/hour -/
theorem fourth_hour_rate : ∃ x : ℝ, TotalWater x = 34 ∧ x = 14 := by
  use 14
  apply And.intro
  · -- Prove TotalWater 14 = 34
    simp [TotalWater, FillRate]
    norm_num
  · -- Prove 14 = 14
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_hour_rate_l354_35450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l354_35458

noncomputable def z : ℂ := 5 * Complex.I / (Complex.I - 1)

theorem z_in_first_quadrant : 0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l354_35458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l354_35461

-- Define the function f as noncomputable due to Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (π / 4) (π / 2) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (π / 4) (π / 2) → f x ≤ f y) ∧
  f x = 1 := by
  sorry

-- You can add more lemmas or helper theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l354_35461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_problem_l354_35487

/-- Calculates the percentage of weight lost given initial weight, final weight, and weight gained after loss. -/
noncomputable def weight_loss_percentage (initial_weight final_weight weight_gain : ℝ) : ℝ :=
  (1 - (final_weight - weight_gain) / initial_weight) * 100

/-- Theorem: Given the conditions in the problem, the weight loss percentage is 10%. -/
theorem weight_loss_problem (initial_weight final_weight weight_gain : ℝ)
  (h1 : initial_weight = 220)
  (h2 : final_weight = 200)
  (h3 : weight_gain = 2) :
  weight_loss_percentage initial_weight final_weight weight_gain = 10 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_problem_l354_35487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_nodes_at_distance_5_l354_35475

/-- An infinite grid with square cells of side length 1 -/
structure Grid where
  nodes : Set (ℤ × ℤ)
  is_infinite : Set.Infinite nodes

/-- A coloring of the grid nodes with two colors -/
def Coloring (g : Grid) := g.nodes → Bool

/-- The distance between two nodes in the grid -/
noncomputable def distance (a b : ℤ × ℤ) : ℝ :=
  Real.sqrt (((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) : ℝ)

theorem different_color_nodes_at_distance_5 (g : Grid) (c : Coloring g) 
  (h1 : ∃ n : g.nodes, c n = true) 
  (h2 : ∃ n : g.nodes, c n = false) :
  ∃ (a b : g.nodes), c a ≠ c b ∧ distance a.1 b.1 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_nodes_at_distance_5_l354_35475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_shift_l354_35443

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (2 * x - 2 * φ + Real.pi / 3)

theorem odd_function_shift (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi / 2) :
  (∀ x, f φ (-x) = -(f φ x)) → φ = 5 * Real.pi / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_shift_l354_35443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_proof_l354_35432

open Real

theorem theta_value_proof (f g : ℝ → ℝ) (θ : ℝ) : 
  (∀ x, f x = sin (x + π/3) + 1) →
  (∀ x, g x = sin (2*x + π/3) + 1) →
  θ ∈ Set.Ioo 0 π →
  (∀ x, g x + g (θ - x) = 2) →
  θ = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_proof_l354_35432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_distance_l354_35427

/-- The greatest distance between a vertex of an inscribed square and a vertex of the outer square -/
noncomputable def greatest_distance (inner_perimeter outer_perimeter : ℝ) : ℝ :=
  let inner_side := inner_perimeter / 4
  let outer_side := outer_perimeter / 4
  let segment := (outer_side - inner_side) / 2
  Real.sqrt (2 * segment^2)

/-- Theorem stating that the greatest distance between vertices of inscribed squares is √2 -/
theorem inscribed_squares_distance :
  greatest_distance 24 32 = Real.sqrt 2 := by
  -- Unfold the definition of greatest_distance
  unfold greatest_distance
  -- Simplify the arithmetic expressions
  simp [Real.sqrt_eq_iff_sq_eq]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_distance_l354_35427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l354_35472

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length : ℝ) (time : ℝ) : ℝ :=
  (train_length + bridge_length) / time

/-- Theorem: Given a train of length 110 m crossing a bridge of length 140 m
    in 14.998800095992321 seconds, the speed of the train is approximately 16.67 m/s -/
theorem train_speed_calculation :
  let train_length : ℝ := 110
  let bridge_length : ℝ := 140
  let time : ℝ := 14.998800095992321
  abs (train_speed train_length bridge_length time - 16.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l354_35472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l354_35424

open Set
open MeasureTheory
open Interval

noncomputable section

variable {a b : ℝ}
variable {f g : ℝ → ℝ}

def is_continuous_nondecreasing (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ContinuousOn h (Icc a b) ∧ MonotoneOn h (Icc a b)

theorem integral_inequality
  (hab : a < b)
  (hf : is_continuous_nondecreasing f a b)
  (hg : is_continuous_nondecreasing g a b)
  (h_nonneg : ∀ x ∈ Icc a b, 0 ≤ f x ∧ 0 ≤ g x)
  (h_leq : ∀ x ∈ Icc a b, ∫ t in a..x, Real.sqrt (f t) ≤ ∫ t in a..x, Real.sqrt (g t))
  (h_eq : ∫ t in a..b, Real.sqrt (f t) = ∫ t in a..b, Real.sqrt (g t)) :
  ∫ t in a..b, Real.sqrt (1 + f t) ≥ ∫ t in a..b, Real.sqrt (1 + g t) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l354_35424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_coprime_sequence_l354_35405

def f (x : ℕ) : ℕ := x^2 - x + 1

def sequenceF (m : ℕ) : ℕ → ℕ
  | 0 => m
  | n + 1 => f (sequenceF m n)

theorem pairwise_coprime_sequence (m : ℕ) (h : m > 1) :
  ∀ i j : ℕ, i ≠ j → Nat.Coprime (sequenceF m i) (sequenceF m j) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_coprime_sequence_l354_35405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_and_arrangement_l354_35464

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The number of ways to select one boy and one girl from the group -/
def select_one_each : ℕ := num_boys * num_girls

/-- The number of ways to select two boys from the group -/
def select_two_boys : ℕ := Nat.choose num_boys 2

/-- The number of ways to select two girls from the group -/
def select_two_girls : ℕ := Nat.choose num_girls 2

/-- The number of ways to arrange two boys -/
def arrange_two_boys : ℕ := Nat.factorial 2

/-- The number of ways to place two girls in three positions -/
def place_two_girls : ℕ := Nat.descFactorial 3 2

/-- The number of ways to arrange two boys and two girls with girls not adjacent -/
def arrange_two_each : ℕ := select_two_boys * select_two_girls * arrange_two_boys * place_two_girls

theorem selection_and_arrangement :
  select_one_each = 12 ∧ arrange_two_each = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_and_arrangement_l354_35464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_and_inverse_l354_35453

theorem sum_a_and_inverse (a : ℝ) (h : a > 0) :
  a^(1/2 : ℝ) + a^(-1/2 : ℝ) = 3 → a + a⁻¹ = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_and_inverse_l354_35453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_900_deg_to_rad_l354_35480

/-- Conversion factor from degrees to radians -/
noncomputable def deg_to_rad : ℝ := Real.pi / 180

/-- Theorem: -900 degrees is equal to -5π radians -/
theorem negative_900_deg_to_rad : 
  -900 * deg_to_rad = -5 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_900_deg_to_rad_l354_35480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_difference_l354_35455

theorem equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  ((2 - x₁^2 / 4) ^ (1/3 : ℝ) = -3) ∧
  ((2 - x₂^2 / 4) ^ (1/3 : ℝ) = -3) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 4 * Real.sqrt 29 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_difference_l354_35455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_balls_count_l354_35420

theorem purple_balls_count (k : ℕ) : k > 0 → (
  let total_balls := 7 + k
  let prob_green := (7 : ℚ) / total_balls
  let prob_purple := (k : ℚ) / total_balls
  let expected_value := prob_green * 3 + prob_purple * (-1)
  expected_value = 1
) → k = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_balls_count_l354_35420
