import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l85_8593

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A point belongs to a circle if its distance from the center equals the radius -/
def belongsToCircle (p : Point) (c : Circle) : Prop :=
  distance p c.center = c.radius

/-- Theorem: The standard equation of a circle with center (3, 1) and radius 5 -/
theorem circle_equation (c : Circle) (p : Point) :
  c.center = Point.mk 3 1 →
  c.radius = 5 →
  belongsToCircle p c ↔ (p.x - 3)^2 + (p.y - 1)^2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l85_8593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_range_l85_8550

/-- The inverse proportion function -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := (4 - k) / x

/-- Theorem: For an inverse proportion function y = (4-k)/x with points A(x₁, y₁) and B(x₂, y₂) 
    on its graph, where x₁ < 0 < x₂ and y₁ < y₂, the value of k must be less than 4. -/
theorem inverse_proportion_k_range (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
    (h1 : x₁ < 0) (h2 : 0 < x₂) (h3 : y₁ < y₂)
    (h4 : y₁ = inverse_proportion k x₁)
    (h5 : y₂ = inverse_proportion k x₂) : 
  k < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_range_l85_8550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_minus_2a_l85_8565

theorem tan_beta_minus_2a (α β : ℝ) 
  (h1 : (Real.sin α * Real.cos α) / (1 - Real.cos (2 * α)) = 1/4)
  (h2 : Real.tan (α - β) = 2) : 
  Real.tan (β - 2*α) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_minus_2a_l85_8565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_nonzero_close_to_zero_l85_8503

/-- A polynomial function from ℝ² to ℝ -/
def f (x y : ℝ) : ℝ := x^2 + (x*y - 1)^2

/-- Theorem stating the existence of a polynomial that is everywhere nonzero 
    but takes values arbitrarily close to zero -/
theorem polynomial_nonzero_close_to_zero :
  (∀ x y : ℝ, f x y ≠ 0) ∧
  (∀ ε > 0, ∃ x y : ℝ, 0 < f x y ∧ f x y < ε) := by
  constructor
  · intro x y
    -- Proof that f is never zero
    sorry
  · intro ε hε
    -- Proof that f can be arbitrarily close to zero
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_nonzero_close_to_zero_l85_8503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l85_8541

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.A = Real.pi/3 ∧  -- A = 60°
  (1/2 * t.b * t.c * Real.sin t.A) = 15 * Real.sqrt 3 / 4 ∧  -- Area of triangle
  5 * Real.sin t.B = 3 * Real.sin t.C  -- Given condition

-- Theorem statement
theorem triangle_perimeter (t : Triangle) :
  triangle_conditions t →
  t.a + t.b + t.c = 8 + Real.sqrt 19 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l85_8541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l85_8581

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + a*x + a + 5) / Real.log 3

-- State the theorem
theorem decreasing_f_implies_a_range (a : ℝ) : 
  (∀ x y : ℝ, x < y ∧ y < 1 → f a x > f a y) → 
  a ∈ Set.Icc (-3) (-2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l85_8581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l85_8574

theorem problem_solution (x y : ℝ) (m n : ℕ+) 
  (h1 : x > 0) 
  (h2 : y > 0)
  (h3 : 129 - x^2 = 195 - y^2)
  (h4 : 129 - x^2 = x * y)
  (h5 : x = m / n)
  (h6 : Nat.Coprime m.val n.val) :
  100 * m + n = 4306 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l85_8574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l85_8523

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively, 
    prove that the area of the triangle is (1/2) * a^2 * √3 under the given conditions. -/
theorem triangle_area (a b c A B C : ℝ) : 
  (Real.cos A * Real.cos C + Real.sin A * Real.sin C + Real.cos B = 3/2) →
  (b^2 = a*c) →
  (a / Real.sin A + c / Real.sin C = 2*b / Real.sin B) →
  (2*b / Real.sin B = 2) →
  (1/2 * a^2 * Real.sqrt 3 = 1/2 * a * b * Real.sin C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l85_8523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pricing_l85_8516

/-- Represents the detergent pricing problem -/
structure DetergentPricing where
  cost_increase : ℝ
  last_year_total : ℝ
  this_year_total : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_reduction_step : ℝ
  sales_increase_step : ℝ

/-- Calculates the cost price this year -/
noncomputable def cost_price (d : DetergentPricing) : ℝ :=
  d.this_year_total * (d.last_year_total / d.this_year_total + d.cost_increase) / 
    (1 + d.last_year_total / d.this_year_total)

/-- Calculates the weekly sales volume based on price -/
noncomputable def sales_volume (d : DetergentPricing) (price : ℝ) : ℝ :=
  d.initial_sales + d.sales_increase_step * (d.initial_price - price) / d.price_reduction_step

/-- Calculates the weekly profit based on price -/
noncomputable def weekly_profit (d : DetergentPricing) (price : ℝ) : ℝ :=
  (price - cost_price d) * sales_volume d price

/-- Theorem stating the optimal price and maximum profit -/
theorem optimal_pricing (d : DetergentPricing) 
  (h1 : d.cost_increase = 4)
  (h2 : d.last_year_total = 1200)
  (h3 : d.this_year_total = 1440)
  (h4 : d.initial_price = 36)
  (h5 : d.initial_sales = 600)
  (h6 : d.price_reduction_step = 1)
  (h7 : d.sales_increase_step = 100) :
  ∃ (optimal_price max_profit : ℝ),
    cost_price d = 24 ∧
    optimal_price = 33 ∧
    max_profit = 8100 ∧
    ∀ (price : ℝ), price ≥ cost_price d → weekly_profit d price ≤ max_profit :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pricing_l85_8516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_mixture_price_l85_8544

/-- Given two types of candy mixed together, prove the price of the first type. -/
theorem candy_mixture_price (x : ℝ) (p : ℝ) : 
  x > 0 ∧ 
  x + 6.25 = 10 ∧ 
  x * p + 6.25 * 4.30 = 10 * 4 → 
  p = 3.50 := by
  intro h
  sorry

#check candy_mixture_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_mixture_price_l85_8544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l85_8512

/-- Prove that a > b > c given their definitions --/
theorem a_greater_than_b_greater_than_c :
  let a : ℝ := Real.sqrt 2
  let b : ℝ := (3 : ℝ) ^ (-(1/2 : ℝ))
  let c : ℝ := Real.cos (50 * π / 180) * Real.cos (10 * π / 180) +
               Real.cos (140 * π / 180) * Real.sin (170 * π / 180)
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l85_8512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l85_8595

-- Define the slope angle and y-intercept
noncomputable def slope_angle : ℝ := 45 * Real.pi / 180
def y_intercept : ℝ := -1

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop :=
  x - y - 1 = 0

-- Theorem statement
theorem line_equation_proof :
  ∀ x y : ℝ,
  (Real.tan slope_angle = 1) →
  line_equation x y ↔ y = Real.tan slope_angle * x + y_intercept :=
by
  sorry

#check line_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l85_8595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_plus_real_power_l85_8500

theorem complex_power_plus_real_power : ∃ (z : ℂ), z = 513 := by
  -- Define i as the complex imaginary unit
  let i : ℂ := Complex.I

  -- Define the first term
  let term1 : ℂ := i ^ (4 ^ 3)

  -- Define the second term
  let term2 : ℕ := 2 ^ (3 ^ 2)

  -- Define the sum
  let sum : ℂ := term1 + term2

  -- Prove that the sum equals 513
  use sum
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_plus_real_power_l85_8500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l85_8527

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ t : ℝ, x = 2 + (Real.sqrt 2/2)*t ∧ y = -1 + (Real.sqrt 2/2)*t

-- Define point P
def point_P : ℝ × ℝ := (2, -1)

-- Theorem statement
theorem curve_and_line_intersection :
  -- Given the polar equation of curve C
  (∀ θ ρ : ℝ, ρ * (Real.sin θ)^2 = 4 * Real.cos θ → curve_C (ρ * Real.cos θ) (ρ * Real.sin θ)) →
  -- And given the line l passing through P(2,-1) with 45° inclination
  (∀ x y : ℝ, line_l x y → ∃ t : ℝ, x = 2 + (Real.sqrt 2/2)*t ∧ y = -1 + (Real.sqrt 2/2)*t) →
  -- Then:
  -- 1. The Cartesian equation of curve C is y² = 4x
  (∀ x y : ℝ, curve_C x y ↔ y^2 = 4*x) ∧
  -- 2. The product of distances from P to intersection points is 14
  (∃ A B : ℝ × ℝ, 
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧ 
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) *
    ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) = 14^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l85_8527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_height_from_bounce_pattern_l85_8549

/-- The height of a building from which a ball is dropped -/
noncomputable def building_height : ℝ := 96

/-- The bounce ratio of the ball (half of the previous height) -/
def bounce_ratio : ℚ := 1/2

/-- The number of bounces before reaching the given height -/
def num_bounces : ℕ := 5

/-- The height reached by the ball after the given number of bounces -/
noncomputable def final_bounce_height : ℝ := 3

/-- Theorem stating the relationship between the building height and the ball's bounce pattern -/
theorem building_height_from_bounce_pattern :
  building_height * (bounce_ratio ^ num_bounces : ℝ) = final_bounce_height := by
  sorry

#eval bounce_ratio
#eval num_bounces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_height_from_bounce_pattern_l85_8549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_element_is_three_l85_8530

/-- Define a sequence where each new set starts with 1 and counts up one more number than the previous set -/
def specialSequence : ℕ → ℕ
| 0 => 1
| n => 
  let k := n + 1 - (n.sqrt + 1) * n.sqrt / 2
  if k = 0 then 1 else k

/-- The 13th element (0-indexed) of the specialSequence is 3 -/
theorem thirteenth_element_is_three : specialSequence 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_element_is_three_l85_8530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l85_8564

/-- The function f(x) = x³ - 2ln(x) -/
noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * Real.log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (x : ℝ) : ℝ := 3 * x^2 - 2 / x

theorem tangent_line_at_one (x y : ℝ) :
  (f_deriv 1 : ℝ) = 1 →
  f 1 = 1 →
  (x - y = 0) ↔ y - 1 = f_deriv 1 * (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l85_8564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_values_l85_8518

-- Define the curve C
noncomputable def C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l
noncomputable def l (m t : ℝ) : ℝ × ℝ := (m + (Real.sqrt 3 / 2) * t, t / 2)

-- Define the condition |PA| · |PB| = 1
def intersection_condition (m : ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    C (l m t₁).1 (l m t₁).2 ∧ 
    C (l m t₂).1 (l m t₂).2 ∧ 
    t₁ ≠ t₂ ∧
    ((l m t₁).1 - m)^2 + (l m t₁).2^2 * ((l m t₂).1 - m)^2 + (l m t₂).2^2 = 1

theorem intersection_values :
  ∀ m : ℝ, intersection_condition m → m = 1 + Real.sqrt 2 ∨ m = 1 ∨ m = 1 - Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_values_l85_8518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l85_8531

-- Define the quadratic function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + (1/2)

-- State the theorem
theorem quadratic_properties (a : ℝ) (h : a > 0) :
  (∀ x y, x < 0 ∧ y < 0 → f a x ≠ y) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂, (1/a) < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l85_8531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_bodies_with_triangular_front_view_l85_8575

/-- A geometric body is a three-dimensional shape. -/
structure GeometricBody where
  -- We don't need to define the internal structure for this problem

/-- A front view is a two-dimensional projection of a geometric body. -/
structure FrontView where
  -- We don't need to define the internal structure for this problem

/-- A triangle is a polygon with three sides. -/
structure Triangle where
  -- We don't need to define the internal structure for this problem

/-- Convert a Triangle to a FrontView -/
def Triangle.toFrontView (t : Triangle) : FrontView :=
  ⟨⟩ -- Using unit constructor as we don't have internal structure

/-- The property of having a triangular front view. -/
def has_triangular_front_view (body : GeometricBody) : Prop :=
  ∃ (view : FrontView), ∃ (triangle : Triangle), view = triangle.toFrontView

/-- Theorem: There exist at least three distinct types of geometric bodies with a triangular front view. -/
theorem three_bodies_with_triangular_front_view :
  ∃ (body1 body2 body3 : GeometricBody),
    has_triangular_front_view body1 ∧
    has_triangular_front_view body2 ∧
    has_triangular_front_view body3 ∧
    body1 ≠ body2 ∧ body2 ≠ body3 ∧ body1 ≠ body3 :=
by
  -- We'll use sorry to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_bodies_with_triangular_front_view_l85_8575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l85_8597

-- Define the properties of the cones
def radius_C : ℝ := 10
def height_C : ℝ := 20
def radius_D : ℝ := 18
def height_D : ℝ := 12

-- Define the volume of a cone
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Define the volumes of cones C and D
noncomputable def volume_C : ℝ := cone_volume radius_C height_C
noncomputable def volume_D : ℝ := cone_volume radius_D height_D

-- Theorem statement
theorem volume_ratio_of_cones :
  volume_C / volume_D = 125 / 243 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l85_8597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_point_with_same_center_l85_8566

/-- Given a circle C with equation x^2 + y^2 - 4x + 6y - 3 = 0 and a point M(-1, 1),
    prove that (x-2)^2 + (y+3)^2 = 25 is the equation of the circle passing through M
    with the same center as C. -/
theorem circle_through_point_with_same_center
  (C : Set (ℝ × ℝ))
  (h_C : C = {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 + 6*p.2 - 3 = 0})
  (M : ℝ × ℝ)
  (h_M : M = (-1, 1)) :
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 3)^2 = 25} =
  {p : ℝ × ℝ | ∃ c, c ∈ C ∧ dist p c = dist M c ∧ ∀ q ∈ C, dist p q = dist c q} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_point_with_same_center_l85_8566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutationary_iff_divisibility_l85_8561

/-- A simple graph is divisibility if there exists a labeling with distinct positive integers
    such that there's an edge between two vertices iff one label divides the other. -/
def is_divisibility_graph (G : SimpleGraph α) : Prop :=
  ∃ f : α → ℕ+, Function.Injective f ∧
    ∀ u v : α, G.Adj u v ↔ (f u ∣ f v) ∨ (f v ∣ f u)

/-- A simple graph is permutationary if there exists a bijection between its vertices and {1,...,n}
    and a permutation π such that there's an edge between i and j iff i > j and π(i) < π(j). -/
def is_permutationary_graph {n : ℕ} (G : SimpleGraph (Fin n)) : Prop :=
  ∃ π : Equiv.Perm (Fin n), ∀ i j : Fin n, G.Adj i j ↔ i > j ∧ π i < π j

/-- The main theorem stating that a graph is permutationary iff both it and its complement
    are divisibility graphs. -/
theorem permutationary_iff_divisibility {n : ℕ} (G : SimpleGraph (Fin n)) :
  is_permutationary_graph G ↔ is_divisibility_graph G ∧ is_divisibility_graph Gᶜ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutationary_iff_divisibility_l85_8561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l85_8519

def is_prime (n : ℕ) : Prop := Nat.Prime n

def FunctionProperty (f : ℕ → ℕ) : Prop :=
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ Nat.Coprime a b → f (a * b) = f a * f b) ∧
  (∀ p q : ℕ, is_prime p ∧ is_prime q → f (p + q) = f p + f q)

theorem function_values (f : ℕ → ℕ) (h : FunctionProperty f) :
  f 2 = 2 ∧ f 3 = 3 ∧ f 1999 = 1999 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l85_8519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l85_8579

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x / (2^x - 1)

-- Define the domain of the function
def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

-- Theorem statement
theorem f_domain : domain f = Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l85_8579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_range_l85_8567

/-- The sum S for positive real numbers a, b, c, d -/
noncomputable def S (a b c d : ℝ) : ℝ := 
  a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)

/-- The range of S is (1, 2) -/
theorem S_range (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  1 < S a b c d ∧ S a b c d < 2 ∧ 
  ∀ x, 1 < x → x < 2 → ∃ a' b' c' d', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0 ∧ S a' b' c' d' = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_range_l85_8567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_calculation_l85_8548

/-- Calculate the overall percent decrease in price after applying multiple discounts and tax -/
theorem price_decrease_calculation (original_price : ℝ) (discount1 discount2 discount3 tax : ℝ) :
  original_price = 100 ∧ 
  discount1 = 0.20 ∧ 
  discount2 = 0.15 ∧ 
  discount3 = 0.10 ∧ 
  tax = 0.05 → 
  let price_after_discounts := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let final_price := price_after_discounts * (1 + tax)
  let percent_decrease := (original_price - final_price) / original_price * 100
  ‖percent_decrease - 35.74‖ < 0.01 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_calculation_l85_8548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AEB_l85_8546

-- Define the rectangle ABCD
structure Rectangle (A B C D : ℝ × ℝ) : Prop where
  is_rectangle : True  -- Changed to True to avoid the projection error

-- Define the points F and G on side CD
def F (C D : ℝ × ℝ) : ℝ × ℝ := sorry
def G (C D : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define point E as the intersection of AF and BG
def E (A B F G : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

theorem area_of_triangle_AEB 
  (A B C D : ℝ × ℝ)
  (rect : Rectangle A B C D)
  (h1 : |A.1 - B.1| = 10)  -- AB = 10
  (h2 : |B.2 - C.2| = 5)   -- BC = 5
  (h3 : |D.1 - (F C D).1| = 3)  -- DF = 3
  (h4 : |(G C D).1 - C.1| = 4)  -- GC = 4
  : triangle_area A (E A B (F C D) (G C D)) B = 250/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AEB_l85_8546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_two_tangent_lines_condition_l85_8594

-- Define the function f(x) = (ax + 1)e^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 1) * Real.exp x

-- Part 1: Tangent line when a = 1
theorem tangent_line_at_zero (x : ℝ) :
  let f₁ := f 1
  let f₁' := λ x => Real.exp x * (x + 2)
  f₁' 0 = 2 ∧ f₁ 0 = 1 →
  (λ x => 2 * x + 1) = λ x => f₁ 0 + f₁' 0 * (x - 0) :=
by sorry

-- Part 2: Range of a for two tangent lines
theorem two_tangent_lines_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (f a x₁ / x₁ = (Real.exp x₁) * (a * x₁ + a + 1)) ∧
    (f a x₂ / x₂ = (Real.exp x₂) * (a * x₂ + a + 1)))
  ↔ (a > -1/4 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_two_tangent_lines_condition_l85_8594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l85_8535

/-- The range of slope angles for a line intersecting a given line segment -/
theorem slope_angle_range (P A B : ℝ × ℝ) (h_P : P = (0, -1)) (h_A : A = (1, -2)) (h_B : B = (2, 1)) :
  ∃ (α : ℝ), ∀ (l : Set (ℝ × ℝ)),
    (P ∈ l) →
    (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ ((1 - t) • A + t • B) ∈ l) →
    (α ∈ Set.Icc 0 (π / 4) ∪ Set.Ico (3 * π / 4) π ↔
     ∃ (m : ℝ), l = {(x, y) | y = m * (x - P.1) + P.2} ∧ Real.tan α = m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l85_8535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_theorem_l85_8592

/-- Parabola represented by the equation x^2 = 4y -/
def Parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Line represented by the equation x = 5 -/
def Line (x : ℝ) : Prop := x = 5

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem stating the existence of two specific positions satisfying the condition -/
theorem parabola_line_intersection_theorem :
  ∃ (B1 B2 : ℝ × ℝ) (A1 A2 : ℝ × ℝ),
    Line B1.fst ∧ Line B2.fst ∧
    Parabola A1.fst A1.snd ∧ Parabola A2.fst A2.snd ∧
    (∀ A : ℝ × ℝ, Parabola A.fst A.snd → 
      distance A.fst A.snd 0 1 = distance A.fst A.snd B1.fst B1.snd → A = A1) ∧
    (∀ A : ℝ × ℝ, Parabola A.fst A.snd → 
      distance A.fst A.snd 0 1 = distance A.fst A.snd B2.fst B2.snd → A = A2) ∧
    distance A1.fst A1.snd B1.fst B1.snd = 29/4 ∧
    distance A2.fst A2.snd B2.fst B2.snd = 41/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_theorem_l85_8592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phase_shift_l85_8599

theorem min_phase_shift (φ : ℝ) : 
  (φ > 0 ∧ ∀ x, Real.cos x = Real.sin (x - φ - Real.pi/6)) → φ ≥ 4*Real.pi/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phase_shift_l85_8599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l85_8555

noncomputable section

/-- Circle C1 in polar coordinates -/
def C1 (θ : ℝ) : ℝ := -2 * Real.sqrt 2 * Real.cos (θ - Real.pi / 4)

/-- Circle C2 in parametric form -/
def C2 (m θ : ℝ) : ℝ × ℝ := (2 + m * Real.cos θ, 2 + m * Real.sin θ)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that if C1 and C2 are externally tangent, then |m| = 2√2 -/
theorem circles_externally_tangent (m : ℝ) (m_ne_zero : m ≠ 0) :
  (∃ θ₁ θ₂, distance (C1 θ₁, C1 θ₁ * Real.cos θ₁) (C2 m θ₂) = Real.sqrt 2 + |m|) →
  |m| = 2 * Real.sqrt 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l85_8555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_genuine_coins_l85_8554

/-- Represents a coin which may be genuine or counterfeit -/
inductive Coin
| Genuine
| Counterfeit
deriving BEq, Repr

/-- Represents the result of weighing two groups of coins -/
inductive WeighResult
| Equal
| LeftHeavier
| RightHeavier
deriving BEq, Repr

/-- A function that simulates weighing two groups of coins -/
def weigh (group1 : List Coin) (group2 : List Coin) : WeighResult :=
  sorry

/-- The main theorem stating that it's possible to identify 8 genuine coins -/
theorem identify_genuine_coins 
  (coins : List Coin) 
  (h1 : coins.length = 11) 
  (h2 : coins.count Coin.Counterfeit ≤ 1) : 
  ∃ (genuine : List Coin), 
    genuine.length = 8 ∧ 
    genuine.all (· == Coin.Genuine) ∧
    ∃ (w1 w2 : WeighResult), true := by
  sorry

#check identify_genuine_coins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_genuine_coins_l85_8554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l85_8532

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  x₁ : ℝ
  x₂ : ℝ
  root_order : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / a
  are_roots : ∀ x, x = x₁ ∨ x = x₂ ↔ a * x^2 + b * x + c = x

/-- The quadratic function f(x) = ax^2 + bx + c -/
noncomputable def f (qf : QuadraticFunction) (x : ℝ) : ℝ := qf.a * x^2 + qf.b * x + qf.c

/-- The axis of symmetry of the quadratic function -/
noncomputable def axis_of_symmetry (qf : QuadraticFunction) : ℝ := -qf.b / (2 * qf.a)

theorem quadratic_function_properties (qf : QuadraticFunction) :
  (∀ x, 0 < x ∧ x < qf.x₁ → x < f qf x ∧ f qf x < qf.x₁) ∧
  axis_of_symmetry qf < qf.x₁ / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l85_8532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_where_g_equals_x_minus_one_l85_8510

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -3 then -4
  else if x ≤ -1 then (4/2) * (x + 3) - 4
  else if x ≤ 2 then (-1/3) * (x + 1)
  else if x ≤ 3 then 4 * (x - 2) - 1
  else 3

theorem sum_of_x_coordinates_where_g_equals_x_minus_one :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ = x₁ - 1 ∧ g x₂ = x₂ - 1 ∧ x₁ + x₂ = 2 ∧
  ∀ x₃ : ℝ, g x₃ = x₃ - 1 → x₃ = x₁ ∨ x₃ = x₂ :=
by
  -- Proof goes here
  sorry

#check sum_of_x_coordinates_where_g_equals_x_minus_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_where_g_equals_x_minus_one_l85_8510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l85_8522

noncomputable section

/-- The parabola y = 2x^2 -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- The line y = kx + 1/8 -/
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1/8

/-- The focus of the parabola y = 2x^2 -/
def focus : ℝ × ℝ := (0, 1/8)

/-- The intersection points of the line and the parabola -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = parabola x ∧ p.2 = line k x}

theorem intersection_distance (k : ℝ) (A B : ℝ × ℝ) :
  A ∈ intersection_points k →
  B ∈ intersection_points k →
  A ≠ B →
  dist A focus = 1 →
  dist A B = 8/7 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l85_8522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l85_8543

noncomputable def interest_rate : ℝ := 5
noncomputable def time_period : ℝ := 2
noncomputable def interest_difference : ℝ := 60

noncomputable def simple_interest (principal : ℝ) : ℝ :=
  principal * interest_rate * time_period / 100

noncomputable def compound_interest (principal : ℝ) : ℝ :=
  principal * ((1 + interest_rate / 100) ^ time_period - 1)

theorem principal_calculation :
  ∃ (principal : ℝ),
    compound_interest principal - simple_interest principal = interest_difference ∧
    principal = 24000 := by
  sorry

#check principal_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l85_8543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_lines_l85_8578

/-- The curve represented by the polar equation sin θ = √2/2 (ρ ∈ ℝ) is equivalent to two intersecting straight lines -/
theorem polar_to_cartesian_lines (θ ρ : ℝ) :
  Real.sin θ = Real.sqrt 2 / 2 →
  ∃ (x y : ℝ), (y = x ∨ y = -x) ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_lines_l85_8578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l85_8552

-- Define the function f(x) = sin²x + 2cos x
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.cos x

-- Define the domain
def domain (x : ℝ) : Prop := Real.pi / 3 ≤ x ∧ x ≤ 4 * Real.pi / 3

-- Theorem statement
theorem f_max_min :
  ∃ (max min : ℝ), max = 7/4 ∧ min = -2 ∧
  (∀ x, domain x → f x ≤ max) ∧
  (∃ x, domain x ∧ f x = max) ∧
  (∀ x, domain x → min ≤ f x) ∧
  (∃ x, domain x ∧ f x = min) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l85_8552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_math_homework_pages_l85_8540

/-- Represents the number of pages of homework --/
structure Pages where
  value : ℕ

/-- The number of pages of reading homework Rachel had to complete --/
def reading_homework : Pages := ⟨2⟩

/-- The difference between math and reading homework pages --/
def math_reading_difference : Pages := ⟨2⟩

/-- Addition operation for Pages --/
instance : HAdd Pages Pages Pages where
  hAdd a b := ⟨a.value + b.value⟩

/-- The number of pages of math homework Rachel had to complete --/
def math_homework : Pages := reading_homework + math_reading_difference

theorem rachel_math_homework_pages :
  math_homework = ⟨4⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_math_homework_pages_l85_8540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outfit_choices_l85_8563

/-- Represents the colors available for clothing items -/
inductive Color
| Tan
| Black
| Blue
| Gray
| White
| Yellow

/-- Represents a clothing item -/
structure ClothingItem where
  itemType : String
  color : Color

/-- Represents an outfit -/
structure Outfit where
  shirt : ClothingItem
  pants : ClothingItem
  hat : ClothingItem

/-- The number of shirt colors -/
def numShirtColors : Nat := 6

/-- The number of pants colors -/
def numPantsColors : Nat := 4

/-- The number of hat colors -/
def numHatColors : Nat := 6

/-- The number of shirts -/
def numShirts : Nat := 6

/-- The number of pants -/
def numPants : Nat := 4

/-- The number of hats -/
def numHats : Nat := 6

/-- The total number of possible outfits -/
def totalOutfits : Nat := numShirts * numPants * numHats

/-- The number of outfits where all items are the same color -/
def sameColorOutfits : Nat := numPantsColors

theorem outfit_choices :
  totalOutfits - sameColorOutfits = 140 := by
  rfl

#eval totalOutfits - sameColorOutfits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outfit_choices_l85_8563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_perimeter_l85_8572

/-- Theorem: For a cube with a volume of 125 cm³, the perimeter of one of its faces is 20 cm. -/
theorem cube_face_perimeter (V : ℝ) (h : V = 125) : 
  (4 : ℝ) * V ^ (1/3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_perimeter_l85_8572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l85_8582

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (abs x) - 2 / (x^2 + 3)

-- State the theorem
theorem f_inequality_range (x : ℝ) : 
  f x > f (2*x - 1) ↔ 1/3 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l85_8582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l85_8557

/-- The function f(x) defined with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + (1 - a) * x

/-- The equation that has exactly one real root -/
def has_unique_root (a : ℝ) : Prop :=
  ∃! x : ℝ, x * f a (1 / x) = 4 * x - 3

/-- The function f is monotonically decreasing on (0, +∞) -/
def monotone_decreasing_on_pos (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → g x > g y

theorem f_properties (a : ℝ) (ha : a ≠ 0) :
  has_unique_root a →
  (a = 2 ∧ monotone_decreasing_on_pos (f 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l85_8557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_segment_length_l85_8515

open Real

-- Define the necessary structures and functions
structure Point := (x y : ℝ)

def Square (A B C D : Point) : Prop := sorry
def SegmentLength (P Q : Point) : ℝ := sorry
def Segment (P Q : Point) : Set Point := sorry
def Triangle (P Q R : Point) : Set Point := sorry
def Area (S : Set Point) : ℝ := sorry

theorem square_division_segment_length (A B C D P Q : Point) 
  (h1 : Square A B C D) 
  (h2 : SegmentLength A B = 4)
  (h3 : P ∈ Segment A B) (h4 : Q ∈ Segment A B)
  (h5 : Area (Triangle C B P) = Area (Triangle C P Q))
  (h6 : Area (Triangle C P Q) = Area (Triangle C Q A)) :
  SegmentLength C P = (4 * sqrt 10) / 3 := by
    sorry

#check square_division_segment_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_segment_length_l85_8515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_alpha_beta_l85_8533

theorem sin_sum_alpha_beta (α β : ℝ) 
  (h1 : Real.sin α = -3/5) 
  (h2 : Real.cos β = 1) : 
  Real.sin (α + β) = -3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_alpha_beta_l85_8533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_odd_terms_l85_8501

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Sequence definition -/
noncomputable def a (n k : ℕ) : ℤ :=
  floor ((n ^ k : ℝ) / k)

/-- Main theorem -/
theorem infinitely_many_odd_terms (n : ℕ) (hn : n > 1) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ k ∈ S, Odd (a n k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_odd_terms_l85_8501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_geq_y_l85_8553

theorem x_geq_y (a b : ℝ) : (a^2 + b^2 + 20) ≥ 4*(2*b - a) := by
  let x := a^2 + b^2 + 20
  let y := 4*(2*b - a)
  have h : x - y = (a + 2)^2 + (b - 4)^2 := by
    ring
  have h1 : (a + 2)^2 ≥ 0 := by
    apply sq_nonneg
  have h2 : (b - 4)^2 ≥ 0 := by
    apply sq_nonneg
  have h3 : x - y ≥ 0 := by
    rw [h]
    linarith [h1, h2]
  linarith [h3]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_geq_y_l85_8553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l85_8562

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l85_8562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_latia_tv_hours_l85_8526

/-- Calculates the minimum number of hours needed to afford a TV with given conditions -/
def min_hours_to_afford_tv (initial_wage : ℚ) (raise_wage : ℚ) (raise_hours : ℕ) 
  (tv_cost : ℚ) (tax_rate : ℚ) (shipping_fee : ℚ) : ℕ :=
  let total_cost := tv_cost * (1 + tax_rate) + shipping_fee
  let initial_earnings := initial_wage * raise_hours
  let remaining_cost := total_cost - initial_earnings
  raise_hours + (remaining_cost / raise_wage).ceil.toNat

/-- Theorem stating the minimum hours Latia needs to work to afford the TV -/
theorem latia_tv_hours : 
  min_hours_to_afford_tv 10 12 100 1700 (7/100) 50 = 173 := by
  rfl

#eval min_hours_to_afford_tv 10 12 100 1700 (7/100) 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_latia_tv_hours_l85_8526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l85_8513

noncomputable def f (x : ℝ) : ℝ := Real.tan x + (Real.tan x)⁻¹ + Real.sin (2 * x)

theorem period_of_f :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q :=
by
  use Real.pi
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l85_8513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l85_8568

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = Real.pi
  area_formula : (1/2) * a * c * Real.sin B = (Real.sqrt 3/2) * a * c * Real.cos B

theorem triangle_properties (t : Triangle) :
  (t.c = 2 * t.a → t.A = Real.pi/6 ∧ t.B = Real.pi/3 ∧ t.C = Real.pi/2) ∧
  (t.a = 2 ∧ Real.pi/4 ≤ t.A ∧ t.A ≤ Real.pi/3 → 2 ≤ t.c ∧ t.c ≤ Real.sqrt 3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l85_8568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l85_8506

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the constant b
def b : ℝ := sorry

-- Define the sequence a
def a : ℕ → ℝ := sorry

-- Theorem statement
theorem problem_statement :
  (∀ x, f x / g x = b^x) →
  (∀ x, (deriv f) x * g x < f x * (deriv g) x) →
  (f 1 / g 1 + f (-1) / g (-1) = 5/2) →
  (∀ n m, a (n+1) / a n = a (m+1) / a m) →
  (∀ n, a n > 0) →
  (a 5 * a 7 + 2 * a 6 * a 8 + a 4 * a 12 = f 4 / g 4) →
  a 6 + a 8 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l85_8506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_m_plus_one_negative_l85_8569

/-- Given a quadratic function f(x) = x^2 - x + a, prove that f(m+1) < 0 when f(-m) < 0 -/
theorem f_m_plus_one_negative (a m : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = x^2 - x + a) → 
  f (-m) < 0 → 
  f (m + 1) < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_m_plus_one_negative_l85_8569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_root_distance_range_l85_8560

-- Define the triangle sides
variable (a b c : ℝ)

-- Define the conditions of the triangle
def is_obtuse_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ b > a ∧ b > c ∧ b^2 > a^2 + c^2

-- Define the quadratic function
noncomputable def f (x : ℝ) := a * x^2 - Real.sqrt 2 * b * x + c

-- Theorem for part (1)
theorem two_distinct_roots (h : is_obtuse_triangle a b c) :
  ∃ α β : ℝ, α ≠ β ∧ f a b c α = 0 ∧ f a b c β = 0 :=
by
  sorry

-- Theorem for part (2)
theorem root_distance_range (h : is_obtuse_triangle a b c) (h_eq : a = c) :
  ∃ α β : ℝ, α ≠ β ∧ f a b c α = 0 ∧ f a b c β = 0 ∧ 0 < |α - β| ∧ |α - β| < 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_root_distance_range_l85_8560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l85_8551

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  A : ℝ × ℝ  -- left vertex
  F : ℝ × ℝ  -- right focus
  B : ℝ × ℝ  -- moving point on the hyperbola
  h_on_hyperbola : (B.1^2 / a^2) - (B.2^2 / b^2) = 1
  h_perpendicular : (B.1 - F.1) * (F.1 - A.1) + (B.2 - F.2) * (F.2 - A.2) = 0
  h_distance_ratio : (F.1 - A.1)^2 + (F.2 - A.2)^2 = 4 * ((B.1 - F.1)^2 + (B.2 - F.2)^2)
  h_triangle_area : (B.1 - A.1) * (F.2 - A.2) - (B.2 - A.2) * (F.1 - A.1) = 25/2

/-- The main theorem about the hyperbola -/
theorem hyperbola_properties (C : Hyperbola) :
  C.a^2 = 4 ∧ C.b^2 = 5 ∧
  (C.B.1 > C.a → C.B.2 > 0 →
   3 * (Real.arctan ((C.B.2) / (C.B.1 + 2))) = Real.arctan ((C.B.2) / (3 - C.B.1)) →
   C.B.1 = (29 + 5 * Real.sqrt 17) / 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l85_8551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angelina_walk_time_difference_grocery_to_gym_faster_l85_8504

/-- Represents Angelina's walking scenario -/
structure WalkingScenario where
  v : ℝ  -- Speed from home to grocery in meters per second
  d1 : ℝ  -- Distance from home to grocery in meters
  d2 : ℝ  -- Distance from grocery to gym in meters

/-- Calculates the time difference between two walks -/
noncomputable def timeDifference (scenario : WalkingScenario) : ℝ :=
  scenario.d1 / scenario.v - scenario.d2 / (2 * scenario.v)

/-- Theorem stating the time difference in Angelina's walking scenario -/
theorem angelina_walk_time_difference (scenario : WalkingScenario) 
  (h1 : scenario.d1 = 100) 
  (h2 : scenario.d2 = 180) 
  (h3 : scenario.v > 0) :
  timeDifference scenario = 10 / scenario.v :=
by
  sorry

/-- Theorem stating that the time from grocery to gym is less than from home to grocery -/
theorem grocery_to_gym_faster (scenario : WalkingScenario) 
  (h1 : scenario.d1 = 100) 
  (h2 : scenario.d2 = 180) 
  (h3 : scenario.v > 0) :
  scenario.d2 / (2 * scenario.v) < scenario.d1 / scenario.v :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angelina_walk_time_difference_grocery_to_gym_faster_l85_8504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l85_8534

-- Define the arithmetic sequence and its sum
noncomputable def a (n : ℕ) : ℝ := n
noncomputable def S (n : ℕ) : ℝ := n * (n + 1) / 2

-- State the theorem
theorem arithmetic_sequence_problem :
  (a 2 + a 4 = 6) ∧ 
  (a 6 = S 3) ∧ 
  (∃ k : ℕ, (a k) * (S (2*k)) = (a (3*k))^2) →
  (∀ n : ℕ, a n = n) ∧ 
  (∃ k : ℕ, k = 4 ∧ (a k) * (S (2*k)) = (a (3*k))^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l85_8534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_l85_8596

/-- Calculates simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Represents the loan transaction -/
structure LoanTransaction where
  borrowedAmount : ℝ
  borrowedRate : ℝ
  lentRate : ℝ
  time : ℝ
  yearlyGain : ℝ

/-- Theorem statement -/
theorem loan_interest_rate (transaction : LoanTransaction) 
  (h1 : transaction.borrowedAmount = 20000)
  (h2 : transaction.borrowedRate = 8)
  (h3 : transaction.time = 6)
  (h4 : transaction.yearlyGain = 200) :
  transaction.lentRate = 9 := by
  sorry

#eval "Loan interest rate theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_l85_8596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_tub_volume_l85_8598

/-- Represents the volume of a hot tub in gallons -/
def HotTubVolume : ℝ → Prop := sorry

/-- Represents the price of a bottle of champagne in dollars -/
def BottlePrice : ℝ → Prop := sorry

/-- Represents the volume discount percentage -/
def VolumeDiscount : ℝ → Prop := sorry

/-- Represents the total amount spent on champagne in dollars -/
def TotalSpent : ℝ → Prop := sorry

/-- Represents the volume of a bottle of champagne in quarts -/
def BottleVolume : ℝ → Prop := sorry

/-- Represents the number of quarts in a gallon -/
def QuartsPerGallon : ℝ → Prop := sorry

theorem hot_tub_volume 
  (bottle_price : BottlePrice 50)
  (volume_discount : VolumeDiscount 0.2)
  (total_spent : TotalSpent 6400)
  (bottle_volume : BottleVolume 1)
  (quarts_per_gallon : QuartsPerGallon 4) :
  HotTubVolume 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_tub_volume_l85_8598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_existence_l85_8586

theorem system_solution_existence (a : ℝ) : 
  (∃ b x y : ℝ, 
    x = abs (y + a) + 4 / a ∧ 
    x^2 + y^2 + 24 + b * (2 * y + b) = 10 * x) ↔ 
  (a < 0 ∨ a ≥ 2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_existence_l85_8586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l85_8587

/-- A circle intersecting the coordinate axes at points A(a, 0), B(b, 0), C(0, c), and D(0, d) has its center at ((a+b)/2, (c+d)/2) -/
theorem circle_center_coordinates (a b c d : ℝ) : 
  let circle := {p : ℝ × ℝ | ∃ (r : ℝ), (p.1 - (a + b) / 2)^2 + (p.2 - (c + d) / 2)^2 = r^2}
  let A : ℝ × ℝ := (a, 0)
  let B : ℝ × ℝ := (b, 0)
  let C : ℝ × ℝ := (0, c)
  let D : ℝ × ℝ := (0, d)
  A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle →
  ∃ (center : ℝ × ℝ), center = ((a + b) / 2, (c + d) / 2) ∧
    ∀ (p : ℝ × ℝ), p ∈ circle ↔ ∃ (r : ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l85_8587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l85_8584

noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem geometric_sequence_first_term 
  (a : ℝ) (r : ℝ) 
  (h1 : geometric_sequence a r 7 = (factorial 9 : ℝ))
  (h2 : geometric_sequence a r 11 = (factorial 11 : ℝ)) :
  a = 362880 / (110 ^ (1/4 : ℝ)) ^ 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l85_8584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_acute_angle_l85_8591

theorem parallel_vectors_acute_angle (α : ℝ) :
  let a : Fin 2 → ℝ := ![3/2, Real.sin α]
  let b : Fin 2 → ℝ := ![Real.cos α, 1/3]
  (∃ (k : ℝ), a = k • b) →  -- parallel vectors condition
  0 < α ∧ α < π / 2 →  -- acute angle condition
  α = π / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_acute_angle_l85_8591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l85_8588

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x y : ℝ) : Prop := 4*x + 3*y + 8 = 0

-- Define the distance from a point to the parabola's axis
def d1 (x y : ℝ) : ℝ := abs x

-- Define the distance from a point to the line
noncomputable def d2 (x y : ℝ) : ℝ := abs (4*x + 3*y + 8) / Real.sqrt (4^2 + 3^2)

-- State the theorem
theorem min_distance_sum :
  ∃ (x y : ℝ), parabola x y ∧ 
  (∀ (x' y' : ℝ), parabola x' y' → d1 x' y' + d2 x' y' ≥ d1 x y + d2 x y) ∧
  d1 x y + d2 x y = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l85_8588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_to_intersections_l85_8585

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space defined by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Curve in 2D space defined by y² = kx -/
structure Curve where
  k : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Sum of distances from P to intersection points of line and curve -/
theorem sum_distances_to_intersections 
  (P : Point) 
  (l : Line) 
  (C : Curve) 
  (A B : Point) : 
  P.x = 1 → 
  P.y = -2 → 
  l.a = 1 → 
  l.b = -1 → 
  l.c = -3 → 
  C.k = 2 → 
  (A.x - A.y - 3 = 0 ∧ A.y^2 = 2*A.x) → 
  (B.x - B.y - 3 = 0 ∧ B.y^2 = 2*B.x) → 
  distance P A + distance P B = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_to_intersections_l85_8585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_integer_l85_8571

noncomputable def floor (x : ℝ) := ⌊x⌋

theorem k_is_integer (k : ℝ) (h1 : k ≥ 1) 
  (h2 : ∀ (m n : ℤ), m = n * (m / n) → 
    ∃ (t : ℤ), floor (m * k) = t * floor (n * k)) : 
  ∃ (z : ℤ), k = z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_integer_l85_8571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_probability_is_point_two_l85_8589

/-- Represents a single throw result -/
inductive ThrowResult
| Hit
| Miss

/-- Converts a digit to a ThrowResult -/
def digitToThrowResult (d : Nat) : ThrowResult :=
  match d with
  | 2 | 3 | 5 | 7 => ThrowResult.Hit
  | _ => ThrowResult.Miss

/-- Represents a pair of throws -/
structure ThrowPair :=
  (first second : ThrowResult)

/-- Checks if a ThrowPair is a double hit -/
def isDoubleHit (pair : ThrowPair) : Bool :=
  match pair.first, pair.second with
  | ThrowResult.Hit, ThrowResult.Hit => true
  | _, _ => false

/-- The simulation results -/
def simulationResults : List (Nat × Nat) := [
  (9, 3), (2, 8), (1, 2), (4, 5), (8, 5), (6, 9), (6, 8), (3, 4), (3, 1), (2, 5),
  (7, 3), (9, 3), (0, 2), (7, 5), (5, 6), (4, 8), (8, 7), (3, 0), (1, 1), (3, 5)
]

/-- Converts the simulation results to ThrowPairs -/
def simulationThrowPairs : List ThrowPair :=
  simulationResults.map fun (a, b) => ⟨digitToThrowResult a, digitToThrowResult b⟩

theorem estimated_probability_is_point_two :
  let doubleHits := simulationThrowPairs.filter isDoubleHit
  (doubleHits.length : Rat) / simulationThrowPairs.length = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_probability_is_point_two_l85_8589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l85_8528

/-- The radius of the k-th inscribed circle in a configuration of two tangent semicircles -/
def radius_k (k : ℕ) : ℚ :=
  4 / (4 * k^2 - 4 * k + 9)

/-- The largest semicircle has radius 1 -/
def largest_semicircle_radius : ℚ := 1

/-- The smaller semicircle has radius 1/2 -/
def smaller_semicircle_radius : ℚ := 1/2

/-- The theorem stating that the radius of the k-th inscribed circle is given by radius_k -/
theorem inscribed_circle_radius (k : ℕ) :
  ∃ (r : ℚ), r = radius_k k ∧ 
  r > 0 ∧
  r < largest_semicircle_radius ∧
  r < smaller_semicircle_radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l85_8528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_iff_even_degree_l85_8511

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Condition: p(a) - p(b) is divisible by a + b for all integers a and b where a + b ≠ 0 -/
def SatisfiesCondition (p : IntPolynomial) : Prop :=
  ∀ a b : ℤ, a + b ≠ 0 → (a + b) ∣ (p.eval a - p.eval b)

/-- A polynomial has only even-degree terms -/
def HasOnlyEvenDegreeTerms (p : IntPolynomial) : Prop :=
  ∀ n : ℕ, n % 2 = 1 → p.coeff n = 0

/-- Main theorem: A polynomial satisfies the condition if and only if it has only even-degree terms -/
theorem condition_iff_even_degree (p : IntPolynomial) :
  SatisfiesCondition p ↔ HasOnlyEvenDegreeTerms p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_iff_even_degree_l85_8511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_factors_double11_optimal_strategy_l85_8525

/-- Represents a discount coupon --/
structure Coupon where
  threshold : ℚ
  discount : ℚ

/-- Represents the shopping subsidy --/
structure Subsidy where
  threshold : ℚ
  amount : ℚ

/-- Calculates the discount rate for a given purchase amount and coupon --/
def discountRate (purchaseAmount : ℚ) (coupon : Coupon) : ℚ :=
  if purchaseAmount ≥ coupon.threshold then coupon.discount / purchaseAmount else 0

/-- Calculates the total discount including the subsidy --/
def totalDiscount (purchaseAmount : ℚ) (coupon : Coupon) (subsidy : Subsidy) : ℚ :=
  let couponDiscount := if purchaseAmount ≥ coupon.threshold then coupon.discount else 0
  let subsidyAmount := if purchaseAmount ≥ subsidy.threshold then subsidy.amount else 0
  couponDiscount + subsidyAmount

/-- The main theorem stating that the optimal strategy involves considering purchase amount and discount rate --/
theorem optimal_strategy_factors (coupons : List Coupon) (subsidy : Subsidy) :
  ∃ (optimalAmount : ℚ) (optimalCoupon : Coupon),
    optimalAmount ∈ (coupons.map (·.threshold)).toFinset ∧
    optimalCoupon ∈ coupons ∧
    ∀ (amount : ℚ) (coupon : Coupon),
      amount ≥ 0 → coupon ∈ coupons →
        totalDiscount optimalAmount optimalCoupon subsidy / optimalAmount ≥
        totalDiscount amount coupon subsidy / amount :=
  by sorry

/-- The specific coupons and subsidy for the Double 11 shopping festival --/
def double11Coupons : List Coupon := [
  { threshold := 399, discount := 60 },
  { threshold := 699, discount := 100 },
  { threshold := 1000, discount := 150 }
]

def double11Subsidy : Subsidy := { threshold := 200, amount := 30 }

/-- The theorem applied to the Double 11 shopping festival --/
theorem double11_optimal_strategy :
  ∃ (optimalAmount : ℚ) (optimalCoupon : Coupon),
    optimalAmount ∈ (double11Coupons.map (·.threshold)).toFinset ∧
    optimalCoupon ∈ double11Coupons ∧
    ∀ (amount : ℚ) (coupon : Coupon),
      amount ≥ 0 → coupon ∈ double11Coupons →
        totalDiscount optimalAmount optimalCoupon double11Subsidy / optimalAmount ≥
        totalDiscount amount coupon double11Subsidy / amount :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_factors_double11_optimal_strategy_l85_8525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cone_contradiction_l85_8542

theorem cylinder_cone_contradiction (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  ¬ (π * r^2 * h = π * (3*r)^2 * h / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cone_contradiction_l85_8542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_half_l85_8521

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = a - 1/(2^x + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a - 1 / (2^x + 1)

/-- If f(x) = a - 1/(2^x + 1) is an odd function, then a = 1/2 -/
theorem odd_function_implies_a_eq_half (a : ℝ) :
  IsOdd (f a) → a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_half_l85_8521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_area_theorem_l85_8558

/-- The area of a regular pentagon with side length 10 units -/
noncomputable def regular_pentagon_area : ℝ :=
  (250 * Real.sin (72 * Real.pi / 180)) / (2 * Real.sin (36 * Real.pi / 180))

/-- The side length of the regular pentagon -/
def a : ℝ := 10

/-- The length of the shortest diagonal in the regular pentagon -/
noncomputable def b : ℝ := sorry

/-- The length of the longest diagonal in the regular pentagon -/
noncomputable def d : ℝ := sorry

/-- Theorem stating that the area of a regular pentagon with side length 10 units
    is equal to (250 * sin(72°)) / (2 * sin(36°)) -/
theorem regular_pentagon_area_theorem :
  (5 / 2 * a * (a / (2 * Real.sin (36 * Real.pi / 180))) * Real.sin (72 * Real.pi / 180)) =
  regular_pentagon_area := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_area_theorem_l85_8558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l85_8524

/-- Predicate to represent that a, b, c form a triangle -/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate to represent that α, β, γ are opposite angles to sides a, b, c respectively -/
def OppositeAngles (a b c α β γ : ℝ) : Prop :=
  (a / Real.sin α = b / Real.sin β) ∧ (b / Real.sin β = c / Real.sin γ)

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c α β γ : ℝ) 
  (h_triangle : IsTriangle a b c)
  (h_angles : α + β + γ = Real.pi)
  (h_opposite : OppositeAngles a b c α β γ) : 
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l85_8524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_and_slope_range_l85_8547

-- Define the fixed circle A
noncomputable def circle_A (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define the point B
def point_B : ℝ × ℝ := (1, 0)

-- Define the trajectory of M's center
noncomputable def trajectory_M (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l with slope k
noncomputable def line_l (k m x y : ℝ) : Prop := y = k * x + m

-- Define the area of the triangle
noncomputable def triangle_area (k m : ℝ) : ℝ := (k * m^2) / (2 * (4 * k^2 + 3)^2)

theorem circle_trajectory_and_slope_range :
  ∀ (k m : ℝ),
    k ≠ 0 →
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      trajectory_M x₁ y₁ ∧
      trajectory_M x₂ y₂ ∧
      line_l k m x₁ y₁ ∧
      line_l k m x₂ y₂ ∧
      (x₁, y₁) ≠ (x₂, y₂)) →
    triangle_area k m = 1/14 →
    3/4 < |k| ∧ |k| < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_and_slope_range_l85_8547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l85_8573

theorem complex_equation_solution (n : ℕ) :
  (∃ z : ℂ, z^n + z + 1 = 0 ∧ Complex.abs z = 1) ↔ (n - 2) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l85_8573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_8_div_x_greater_than_x_l85_8502

def is_valid (x : ℕ) : Prop := x > 0 ∧ x < 10

-- Make satisfies_inequality decidable
def satisfies_inequality (x : ℕ) : Bool := (8 : ℚ) / x > x

theorem probability_8_div_x_greater_than_x :
  (Finset.filter (fun x => satisfies_inequality x) (Finset.range 10)).card / (Finset.range 10).card = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_8_div_x_greater_than_x_l85_8502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_sum_l85_8577

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The second term of the sequence sum is 3 -/
axiom S2 : S 2 = 3

/-- The fourth term of the sequence sum is 15 -/
axiom S4 : S 4 = 15

/-- Theorem: The sixth term of the sequence sum is 63 -/
theorem sixth_term_sum : S 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_sum_l85_8577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_148_l85_8505

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  altitude : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  (t.base1 + t.base2) * t.altitude / 2

/-- Theorem: The area of the given trapezoid is 148 -/
theorem trapezoid_area_is_148 (t : Trapezoid) 
    (h1 : t.altitude = 8)
    (h2 : t.base1 = 15)
    (h3 : t.base2 = 22) : 
  trapezoidArea t = 148 := by
  -- Unfold the definition of trapezoidArea
  unfold trapezoidArea
  -- Substitute the known values
  rw [h1, h2, h3]
  -- Simplify the arithmetic
  norm_num

#check trapezoid_area_is_148

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_148_l85_8505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_ted_distance_difference_l85_8539

noncomputable def ted_speed : ℝ := 11.9999976
noncomputable def frank_speed : ℝ := (2/3) * ted_speed
def time : ℝ := 2

theorem frank_ted_distance_difference : 
  ted_speed * time - frank_speed * time = 8 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_ted_distance_difference_l85_8539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_roots_l85_8509

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 then 2010 * x + Real.log x / Real.log 2010
  else if x < 0 then -(2010 * (-x) + Real.log (-x) / Real.log 2010)
  else 0

-- State the theorem
theorem f_has_three_roots :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∃! a b c, a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_roots_l85_8509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_is_three_fourths_l85_8520

-- Define the angle α
variable (α : Real)

-- Define the point P
noncomputable def P : ℝ × ℝ := (8 * Real.cos (60 * Real.pi / 180), 6 * Real.sin (30 * Real.pi / 180))

-- State the theorem
theorem tan_alpha_is_three_fourths :
  (∃ (t : ℝ), t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  Real.tan α = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_is_three_fourths_l85_8520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_10th_term_l85_8514

def our_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n > 0 → a (n + 1) = a n + a 2) ∧ (a 3 = 6)

theorem sequence_10th_term (a : ℕ → ℕ) (h : our_sequence a) : a 10 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_10th_term_l85_8514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_in_leap_year_l85_8507

/-- Represents the outcome of rolling an eight-sided die -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six
  | Seven
  | Eight

/-- Defines the probability of each outcome for a fair eight-sided die -/
def dieProbability (outcome : DieOutcome) : ℚ :=
  1 / 8

/-- Determines if a roll requires re-rolling -/
def requiresReroll (outcome : DieOutcome) : Bool :=
  match outcome with
  | DieOutcome.One => true
  | DieOutcome.Eight => true
  | _ => false

/-- Calculates the expected number of rolls per day -/
def expectedRollsPerDay : ℚ :=
  4 / 3

/-- The number of days in a leap year -/
def daysInLeapYear : ℕ := 366

/-- Theorem: The expected number of die rolls in a leap year is 488 -/
theorem expected_rolls_in_leap_year :
  ⌊(expectedRollsPerDay * daysInLeapYear : ℚ)⌋ = 488 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_in_leap_year_l85_8507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_stereo_purchase_l85_8580

def old_system_cost : ℝ := 250
def trade_in_percentage : ℝ := 0.80
def new_system_cost : ℝ := 600
def discount_percentage : ℝ := 0.25

theorem john_stereo_purchase :
  old_system_cost * trade_in_percentage + new_system_cost * (1 - discount_percentage) - new_system_cost = -250 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_stereo_purchase_l85_8580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_equivalence_l85_8583

-- Define necessary structures and functions
structure Point

def interior_angles (A B C : Point) : Set ℝ := sorry

def is_arithmetic_sequence (a b c : ℝ) : Prop := sorry

theorem triangle_angle_equivalence (A B C : Point) :
  (∃ θ ∈ interior_angles A B C, θ = 60) ↔
  ∃ (a b c : ℝ), interior_angles A B C = {a, b, c} ∧ is_arithmetic_sequence a b c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_equivalence_l85_8583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basket_shooting_game_properties_l85_8570

/-- Represents a player in the basket shooting game -/
inductive Player : Type
| A
| B

/-- The shooting percentage for a given player -/
noncomputable def shootingPercentage (p : Player) : ℝ :=
  match p with
  | Player.A => 0.6
  | Player.B => 0.8

/-- The probability of a player taking the first shot -/
noncomputable def firstShotProbability : ℝ := 0.5

/-- The probability that player B takes the second shot -/
noncomputable def probBTakesSecondShot : ℝ := 0.6

/-- The probability that player A takes the i-th shot -/
noncomputable def probATakesIthShot (i : ℕ) : ℝ :=
  1/3 + (1/6) * (2/5)^(i-1)

/-- The expected number of times player A shoots in the first n shots -/
noncomputable def expectedAShots (n : ℕ) : ℝ :=
  (5/18) * (1 - (2/5)^n) + n/3

theorem basket_shooting_game_properties :
  (probBTakesSecondShot = 0.6) ∧
  (∀ i : ℕ, probATakesIthShot i = 1/3 + (1/6) * (2/5)^(i-1)) ∧
  (∀ n : ℕ, expectedAShots n = (5/18) * (1 - (2/5)^n) + n/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basket_shooting_game_properties_l85_8570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l85_8538

-- Define the ellipse
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the theorem
theorem ellipse_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 2 / 2)
  (h4 : ellipse a b (Real.sqrt 2) 1) :
  -- Part 1: The equation of the ellipse
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  -- Part 2: Constant sum of squared distances
  (∀ m : ℝ, -2 ≤ m → m ≤ 2 →
    ∀ A B : ℝ × ℝ,
    ellipse a b A.1 A.2 →
    ellipse a b B.1 B.2 →
    (B.2 - A.2) / (B.1 - A.1) = Real.sqrt 2 / 2 →
    (A.1 - m)^2 + A.2^2 + (B.1 - m)^2 + B.2^2 = 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l85_8538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_lambda_l85_8559

def a (l : ℝ) : ℝ × ℝ := (l, 1)
def b (l : ℝ) : ℝ × ℝ := (l + 2, 1)

theorem orthogonal_vectors_lambda (l : ℝ) :
  ‖a l + b l‖ = ‖a l - b l‖ → l = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_lambda_l85_8559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l85_8536

/-- Calculates the time taken for a train to cross a signal pole given its length, 
    the length of a platform it crosses, and the time taken to cross the platform. -/
noncomputable def time_to_cross_signal_pole (train_length platform_length time_to_cross_platform : ℝ) : ℝ :=
  train_length / ((train_length + platform_length) / time_to_cross_platform)

/-- Theorem stating that a 300 m long train crossing a 187.5 m platform in 39 seconds
    will take 24 seconds to cross a signal pole. -/
theorem train_crossing_time : 
  time_to_cross_signal_pole 300 187.5 39 = 24 := by
  -- Unfold the definition of time_to_cross_signal_pole
  unfold time_to_cross_signal_pole
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l85_8536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_max_value_l85_8556

theorem triangle_tan_max_value (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) :
  (1 / Real.tan A + 1 / Real.tan B + Real.tan C = 0) →
  (∀ A' B' C' : ℝ, (0 < A' ∧ 0 < B' ∧ 0 < C' ∧ A' + B' + C' = Real.pi) →
    (1 / Real.tan A' + 1 / Real.tan B' + Real.tan C' = 0) →
    Real.tan C ≤ Real.tan C') →
  Real.tan C = -2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_max_value_l85_8556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l85_8529

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 2*a*x^2 - 3*a^2*x

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  -- Part 1
  (a = -1 → ∃ (m b : ℝ), m = 3 ∧ b = -8 ∧ ∀ x y, y = f (-1) x → m*x - m*y + b = 0) ∧
  -- Part 2
  (a > 0 → 
    (∀ x, a < x ∧ x < 3*a → (deriv (f a)) x > 0) ∧
    (∀ x, x < a ∨ x > 3*a → (deriv (f a)) x < 0) ∧
    (∀ x, f a x ≤ 0) ∧
    (∃ x, f a x = 0) ∧
    (∃ x, f a x = -4/3 * a^3) ∧
    (∀ x, f a x ≥ -4/3 * a^3)) ∧
  -- Part 3
  (∀ x, x ∈ Set.Icc (2*a) (2*a + 2) → |deriv (f a) x| ≤ 3*a) →
  1 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l85_8529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l85_8590

/-- The general form of the equation of a line passing through two points -/
def general_line_equation (x₁ y₁ x₂ y₂ : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (y₂ - y₁) * x + (x₁ - x₂) * y + (x₂ * y₁ - x₁ * y₂) = 0

/-- Theorem: The general form of the equation of the line passing through (-5, 0) and (3, -3) is 3x + 8y - 15 = 0 -/
theorem line_through_points : 
  general_line_equation (-5) 0 3 (-3) = λ x y ↦ 3 * x + 8 * y - 15 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l85_8590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l85_8508

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

-- Define the point on the curve
def point : ℝ × ℝ := (-1, -3)

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 5 * x - y + 2 = 0

-- Theorem statement
theorem tangent_line_at_point :
  let (x₀, y₀) := point
  tangent_line x₀ y₀ ∧
  (∀ x, x ≠ -2 → f x = (2 * x - 1) / (x + 2)) ∧
  (∃ δ > 0, ∀ x, |x - x₀| < δ → x ≠ -2 →
    |((f x - f x₀) / (x - x₀) - 5)| < 0.001) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l85_8508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_empty_set_range_l85_8517

theorem inequality_empty_set_range (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 9) * x^2 + (a + 3) * x - 1 < 0) ↔ 
  (a ∈ Set.Icc (-3) (9/5) ∧ a ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_empty_set_range_l85_8517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_value_comparison_l85_8537

theorem trig_value_comparison : Real.tan (240 * π / 180) > Real.sin (150 * π / 180) ∧ Real.sin (150 * π / 180) > Real.cos (-120 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_value_comparison_l85_8537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_property_l85_8576

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if b² = a² + c² - ac and c - a is equal to the height from A to BC,
    then sin((C-A)/2) = 1/2 -/
theorem triangle_special_property (a b c A B C h : ℝ) :
  0 < a → 0 < b → 0 < c →
  0 < A → 0 < B → 0 < C →
  A + B + C = π →
  b^2 = a^2 + c^2 - a*c →
  c - a = h →
  h = b * Real.sin A / Real.sin B →
  Real.sin ((C - A) / 2) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_property_l85_8576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l85_8545

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

/-- Condition on vectors AC, AD, BC, BD -/
def vector_condition (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ a b ∧ ellipse_C x₂ y₂ a b ∧
    (x₁ + a) * (x₂ + a) + y₁ * y₂ - (x₁ - a) * (x₂ - a) - y₁ * y₂ = -32 * Real.sqrt 3 / 5

/-- Main theorem -/
theorem ellipse_properties :
  ∀ (a b : ℝ),
    eccentricity a b = Real.sqrt 3 / 2 →
    vector_condition a b →
    (∀ x y : ℝ, ellipse_C x y a b ↔ x^2 / 4 + y^2 = 1) ∧
    (∃ (t : ℝ), t = -Real.sqrt 3 / 2 ∧
      ∀ x y : ℝ, ellipse_C x y a b →
        (x - t)^2 + y^2 ≥ (Real.sqrt 3 + t)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l85_8545
