import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_division_problem_l712_71286

/-- Given that the factorial of 9 divided by the factorial of (9 - n) equals 504,
    prove that n = 3. -/
theorem factorial_division_problem (n : ℕ) : (Nat.factorial 9 / Nat.factorial (9 - n) = 504) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_division_problem_l712_71286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_daily_tourism_income_l712_71248

noncomputable def f (t : ℝ) : ℝ := 4 + 1 / t

noncomputable def g (t : ℝ) : ℝ := 115 - |t - 15|

noncomputable def w (t : ℝ) : ℝ := f t * g t

theorem min_daily_tourism_income :
  ∃ (min : ℝ), min = 403 + 1/3 ∧
  ∀ (t : ℝ), 1 ≤ t → t ≤ 30 → w t ≥ min :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_daily_tourism_income_l712_71248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_point_D_existence_l712_71295

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y
def line (x y : ℝ) : Prop := y = x + 4

-- Define the intersection points A and B
def intersection_points (p : ℝ) : Prop := ∃ (x₁ y₁ x₂ y₂ : ℝ),
  parabola p x₁ y₁ ∧ line x₁ y₁ ∧ parabola p x₂ y₂ ∧ line x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define OA ⊥ OB condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem parabola_equation_and_point_D_existence (p : ℝ) 
  (hp : p > 0)
  (h_intersection : intersection_points p)
  (h_perpendicular : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    parabola p x₁ y₁ ∧ line x₁ y₁ ∧ parabola p x₂ y₂ ∧ line x₂ y₂ ∧ perpendicular x₁ y₁ x₂ y₂) :
  (∀ (x y : ℝ), parabola p x y ↔ x^2 = 4 * y) ∧
  (∃ (x : ℝ), x = 4 + 4 * Real.sqrt 2 ∧
    ∃ (cx cy : ℝ), cx^2 = 4 * cy ∧
      (cx - x)^2 + cy^2 = ((4 + 4 * Real.sqrt 2 - 4)^2 + 4^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_point_D_existence_l712_71295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_prism_volume_l712_71270

theorem oblique_prism_volume 
  (a b c : ℝ) 
  (α β : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) :
  let V := a * b * c / Real.sqrt (1 + Real.tan (π/2 - α) ^ 2 + Real.tan (π/2 - β) ^ 2)
  ∃ (volume : ℝ), volume = V ∧ volume > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_prism_volume_l712_71270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_mile_revolutions_l712_71242

/-- The number of revolutions a bicycle tire needs to travel half a mile -/
noncomputable def bicycle_revolutions (tire_diameter : ℝ) (distance_miles : ℝ) : ℝ :=
  (distance_miles * 5280) / (tire_diameter * Real.pi)

theorem half_mile_revolutions :
  bicycle_revolutions 4 0.5 = 660 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_mile_revolutions_l712_71242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_area_l712_71237

/-- Given an oblique triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_angle_and_area 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : c * Real.sin A = Real.sqrt 3 * a * Real.cos C)
  (h_c : c = Real.sqrt 21)
  (h_sin : Real.sin C + Real.sin (B - A) = 5 * Real.sin (2 * A)) :
  C = Real.pi / 3 ∧ 
  (1/2) * a * b * Real.sin C = (5 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_area_l712_71237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l712_71291

/-- The eccentricity of a hyperbola with equation y²/4 - x²/9 = 1 is √13/2 -/
theorem hyperbola_eccentricity (x y : ℝ) : 
  (y^2 / 4 - x^2 / 9 = 1) → 
  ∃ (e : ℝ), e = Real.sqrt 13 / 2 ∧ e^2 = 1 + (9 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l712_71291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l712_71215

/-- The curve function y = ln(x-1) -/
noncomputable def curve (x : ℝ) : ℝ := Real.log (x - 1)

/-- The line function x - y + 2 = 0 -/
def line (x y : ℝ) : Prop := x - y + 2 = 0

/-- The shortest distance from a point on the curve to the line -/
noncomputable def shortestDistance : ℝ := 2 * Real.sqrt 2

/-- Theorem stating the existence of a point on the curve that achieves the shortest distance -/
theorem shortest_distance_proof :
  ∃ (x₀ y₀ : ℝ), y₀ = curve x₀ ∧ 
  (∀ (x y : ℝ), y = curve x → line x y → 
    (x - x₀)^2 + (y - y₀)^2 ≥ shortestDistance^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l712_71215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l712_71207

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ -- semi-major axis
  b : ℝ -- semi-minor axis

/-- Checks if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if two points are symmetric about the origin -/
def areSymmetricAboutOrigin (p1 p2 : Point) : Prop :=
  p1.x = -p2.x ∧ p1.y = -p2.y

/-- Calculates the area of a quadrilateral given its four vertices -/
noncomputable def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  distance p1 p2 * distance p2 p3

/-- The main theorem -/
theorem ellipse_quadrilateral_area 
  (e : Ellipse) 
  (F1 F2 P Q : Point) 
  (h1 : e.a = 4 ∧ e.b = 2) 
  (h2 : isOnEllipse F1 e ∧ isOnEllipse F2 e) 
  (h3 : isOnEllipse P e ∧ isOnEllipse Q e) 
  (h4 : areSymmetricAboutOrigin P Q) 
  (h5 : distance P Q = distance F1 F2) :
  quadrilateralArea P F1 Q F2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l712_71207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_zero_l712_71284

open Real MeasureTheory

/-- The function inside the integral -/
noncomputable def f (x : ℝ) : ℝ := log (sqrt (x^2 + 1) + x) + sin x

/-- The theorem stating that the integral of f from -2023 to 2023 is zero -/
theorem integral_f_zero : 
  ∫ x in (-2023 : ℝ)..2023, f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_zero_l712_71284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_quadrilateral_perimeter_l712_71201

/-- An equilateral triangle with side length 4.5 -/
structure EquilateralTriangle :=
  (side : ℝ)
  (is_equilateral : side = 4.5)

/-- A triangular corner cut from the equilateral triangle -/
structure TriangularCorner :=
  (side : ℝ)
  (is_corner : side = 1.5)

/-- The remaining quadrilateral after cutting the corner -/
def RemainingQuadrilateral (t : EquilateralTriangle) (c : TriangularCorner) : ℝ :=
  t.side + 2 * (t.side - c.side) + c.side

/-- Theorem stating that the perimeter of the remaining quadrilateral is 12 -/
theorem remaining_quadrilateral_perimeter
  (t : EquilateralTriangle)
  (c : TriangularCorner) :
  RemainingQuadrilateral t c = 12 := by
  sorry

#check remaining_quadrilateral_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_quadrilateral_perimeter_l712_71201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_system_solution_l712_71281

theorem exponential_system_solution :
  ∃! (x y : ℝ), (2 : ℝ)^x + (3 : ℝ)^y = 5 ∧ (2 : ℝ)^(x+2) + (3 : ℝ)^(y+1) = 18 ∧ (2 : ℝ)^x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_system_solution_l712_71281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l712_71202

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

-- State the theorem
theorem triangle_ratio (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : t.A = 2 * Real.pi / 3) 
  (h3 : t.a = Real.sqrt 3 * t.c) : 
  t.a / t.b = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l712_71202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cadence_old_salary_l712_71292

/-- Represents Cadence's employment situation and earnings --/
structure CadenceEmployment where
  old_salary : ℝ  -- Monthly salary at old company
  old_months : ℕ  -- Months worked at old company
  new_months : ℕ  -- Months worked at new company
  total_earnings : ℝ  -- Total earnings from both companies

/-- The conditions of Cadence's employment --/
def cadence_conditions (c : CadenceEmployment) : Prop :=
  c.new_months = c.old_months + 5 ∧
  c.old_months = 36 ∧
  c.total_earnings = c.old_salary * c.old_months + (1.2 * c.old_salary) * c.new_months ∧
  c.total_earnings = 426000

/-- Theorem stating that given the conditions, Cadence's old salary was $5000 per month --/
theorem cadence_old_salary (c : CadenceEmployment) :
  cadence_conditions c → c.old_salary = 5000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cadence_old_salary_l712_71292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_weight_l712_71243

theorem class_average_weight (group1_count : ℕ) (group1_avg : ℝ) 
  (group2_count : ℕ) (group2_avg : ℝ) :
  group1_count = 16 →
  group2_count = 8 →
  group1_avg = 50.25 →
  group2_avg = 45.15 →
  (group1_count * group1_avg + group2_count * group2_avg) / (group1_count + group2_count) = 48.55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_weight_l712_71243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l712_71278

-- Define the vectors a and b
def a (t : ℝ) : ℝ × ℝ := (1, t)
def b (t : ℝ) : ℝ × ℝ := (-1, t)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a 2D vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem vector_magnitude (t : ℝ) :
  dot_product (2 • (a t) - b t) (b t) = 0 → magnitude (a t) = 2 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l712_71278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_l712_71280

/-- Represents the distance between two cities -/
def distance (d : ℝ) : Prop := d > 0

/-- Represents the speed of a train -/
def speed (s : ℝ) : Prop := s > 0

/-- Represents the time taken for a journey -/
def time (t : ℝ) : Prop := t > 0

/-- The distance traveled by a train given its speed and time -/
def distanceTraveled (s t : ℝ) : ℝ := s * t

theorem train_problem (d : ℝ) (sA sB : ℝ) :
  distance d →
  speed sA →
  speed sB →
  time 4 →
  time 9 →
  sA * 4 = d →
  sB * 9 = d →
  (∃ t : ℝ, distanceTraveled sA t = distanceTraveled sB t + 120) →
  d = 864 :=
by
  sorry

#check train_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_l712_71280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l712_71235

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + Real.cos x ^ 4

-- State the theorem
theorem function_property (x₁ x₂ : ℝ) 
  (h1 : x₁ ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (h2 : x₂ ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (h3 : f x₁ < f x₂) : 
  x₁^2 > x₂^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l712_71235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_remaining_payment_l712_71226

-- Define the bills and their properties
noncomputable def electricity_bill : ℝ := 120
noncomputable def electricity_paid_percentage : ℝ := 0.8

noncomputable def gas_bill : ℝ := 80
noncomputable def gas_paid_fraction : ℝ := 3/4
noncomputable def gas_additional_payment : ℝ := 10

noncomputable def water_bill : ℝ := 60
noncomputable def water_paid_percentage : ℝ := 0.65

noncomputable def internet_bill : ℝ := 50
noncomputable def internet_payment_amount : ℝ := 5
def internet_payment_count : ℕ := 6
noncomputable def internet_discount_percentage : ℝ := 0.1

noncomputable def phone_bill : ℝ := 45
noncomputable def phone_paid_percentage : ℝ := 0.2

-- Theorem to prove
theorem total_remaining_payment :
  let electricity_remaining := electricity_bill * (1 - electricity_paid_percentage)
  let gas_remaining := gas_bill - (gas_bill * gas_paid_fraction) - gas_additional_payment
  let water_remaining := water_bill * (1 - water_paid_percentage)
  let internet_remaining := (internet_bill - (internet_payment_amount * internet_payment_count)) * (1 - internet_discount_percentage)
  let phone_remaining := phone_bill * (1 - phone_paid_percentage)
  electricity_remaining + gas_remaining + water_remaining + internet_remaining + phone_remaining = 109 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_remaining_payment_l712_71226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debt_difference_l712_71203

/-- The amount Jeff paid initially -/
def jeff_paid : ℚ := 90

/-- The amount Maria paid initially -/
def maria_paid : ℚ := 150

/-- The amount Lee paid initially -/
def lee_paid : ℚ := 210

/-- The total amount paid by all three -/
def total_paid : ℚ := jeff_paid + maria_paid + lee_paid

/-- The amount each person should pay for equal sharing -/
noncomputable def equal_share : ℚ := total_paid / 3

/-- The amount Jeff owes Lee -/
noncomputable def jeff_owes : ℚ := equal_share - jeff_paid

/-- The amount Maria owes Lee -/
noncomputable def maria_owes : ℚ := equal_share - maria_paid

theorem debt_difference : jeff_owes - maria_owes = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debt_difference_l712_71203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_angles_l712_71217

theorem sin_sum_angles (α β : ℝ) : 
  0 < α → α < π / 2 → 0 < β → β < π / 2 →
  Real.cos α = 12 / 13 → Real.cos (2 * α + β) = 3 / 5 →
  Real.sin (α + β) = 33 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_angles_l712_71217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_set_B_l712_71211

-- Define the universal set U
def U : Set ℕ := {x : ℕ | x > 0 ∧ Real.log x < Real.log 10}

-- Define set A (we don't know its exact content, but we know it's a subset of U)
variable (A : Set ℕ)

-- Define set B as a variable (this is what we want to prove)
variable (B : Set ℕ)

-- Define the condition for A ∩ (U \ B)
def condition (A B : Set ℕ) : Prop :=
  A ∩ (U \ B) = {m : ℕ | ∃ n : ℕ, n ≤ 4 ∧ m = 2*n + 1}

-- State the theorem
theorem determine_set_B (hU : U = A ∪ B) (hcond : condition A B) :
  B = {2, 4, 6, 8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_set_B_l712_71211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_a_l712_71288

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
noncomputable def vertex : ℝ × ℝ := (3/5, -25/12)

/-- Condition that 2a + b + c is an integer -/
def is_integer_sum (p : Parabola) : Prop :=
  ∃ n : ℤ, 2 * p.a + p.b + p.c = ↑n

/-- The theorem statement -/
theorem smallest_positive_a :
  ∀ p : Parabola,
    p.a > 0 ∧
    is_integer_sum p ∧
    (∀ x y : ℝ, y = p.a * x^2 + p.b * x + p.c ↔ (x, y) = vertex ∨ y > (vertex.2 + p.a * (x - vertex.1)^2)) →
    ∀ q : Parabola,
      q.a > 0 ∧
      is_integer_sum q ∧
      (∀ x y : ℝ, y = q.a * x^2 + q.b * x + q.c ↔ (x, y) = vertex ∨ y > (vertex.2 + q.a * (x - vertex.1)^2)) →
      p.a ≤ q.a →
      p.a = 925/408 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_a_l712_71288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_sqrt_2x_eq_4x_l712_71267

theorem largest_x_sqrt_2x_eq_4x : 
  ∃ (x : ℝ), x = 1/8 ∧ 
  (∀ (y : ℝ), Real.sqrt (2*y) = 4*y → y ≤ x) ∧
  Real.sqrt (2*x) = 4*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_sqrt_2x_eq_4x_l712_71267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l712_71260

-- Define the hyperbola and parabola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus F and point P
def F : ℝ × ℝ := (2, 0)
def P (p : ℝ × ℝ) : Prop := ∃ x y, p = (x, y) ∧ parabola x y ∧ (x - 2)^2 + y^2 = 25

-- Define the theorem
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (h1 : ∃ x y, hyperbola a b x y ∧ parabola x y)
  (h2 : ∃ p, P p) :
  ∃ k : ℝ, k = Real.sqrt 3 ∧ 
    ∀ x y, (y = k * x ∨ y = -k * x) → 
      ∃ t : ℝ, t ≠ 0 → hyperbola a b (t * x) (t * y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l712_71260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_discount_sequence_optimal_discount_difference_l712_71225

/-- Applies discounts to a price in the given order --/
noncomputable def apply_discounts (initial_price : ℝ) (percent_off : ℝ) (flat_off : ℝ) (extra_off : ℝ) : ℝ :=
  let after_percent := initial_price * (1 - percent_off)
  let after_flat := after_percent - flat_off
  if after_flat > 20 then after_flat - extra_off else after_flat

theorem optimal_discount_sequence (initial_price : ℝ) (percent_off : ℝ) (flat_off : ℝ) (extra_off : ℝ)
    (h1 : initial_price = 30)
    (h2 : percent_off = 0.1)
    (h3 : flat_off = 5)
    (h4 : extra_off = 2) :
  apply_discounts initial_price percent_off flat_off extra_off <
  apply_discounts initial_price flat_off percent_off extra_off :=
by
  sorry

theorem optimal_discount_difference (initial_price : ℝ) (percent_off : ℝ) (flat_off : ℝ) (extra_off : ℝ)
    (h1 : initial_price = 30)
    (h2 : percent_off = 0.1)
    (h3 : flat_off = 5)
    (h4 : extra_off = 2) :
  apply_discounts initial_price flat_off percent_off extra_off -
  apply_discounts initial_price percent_off flat_off extra_off = 0.5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_discount_sequence_optimal_discount_difference_l712_71225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_earnings_l712_71265

/-- Calculate the total earnings of a worker for a week based on survey completion --/
theorem worker_earnings (regular_rate : ℚ) (total_surveys : ℕ) (cellphone_surveys : ℕ) 
  (rate_increase : ℚ) : 
  regular_rate = 10 →
  total_surveys = 50 →
  cellphone_surveys = 35 →
  rate_increase = 3/10 →
  (regular_rate * (total_surveys - cellphone_surveys : ℚ) + 
   (regular_rate * (1 + rate_increase) * cellphone_surveys)) = 605 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_earnings_l712_71265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_60_degrees_l712_71299

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)

-- Define the condition a^2 + c^2 - b^2 = ac
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + t.c^2 - t.b^2 = t.a * t.c

-- Define the measure of angle B in radians
noncomputable def angleBMeasure (t : Triangle) : ℝ :=
  Real.arccos ((t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c))

-- Theorem statement
theorem angle_B_is_60_degrees (t : Triangle) 
  (h : satisfiesCondition t) : 
  angleBMeasure t = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_60_degrees_l712_71299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_function_l712_71238

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (4 * x - 2 * φ + Real.pi / 4)

theorem min_phi_for_odd_function :
  ∃ (φ : ℝ), φ > 0 ∧ 
  (∀ (x : ℝ), g φ (-x) = -(g φ x)) ∧
  (∀ (ψ : ℝ), ψ > 0 ∧ ψ < φ → ¬(∀ (x : ℝ), g ψ (-x) = -(g ψ x))) ∧
  φ = Real.pi / 8 := by
  sorry

#check min_phi_for_odd_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_function_l712_71238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l712_71219

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the values a, b, and c
noncomputable def a : ℝ := Real.log (1 / Real.pi)
noncomputable def b : ℝ := (Real.log Real.pi) ^ 2
noncomputable def c : ℝ := Real.log (Real.sqrt Real.pi)

-- State the theorem
theorem f_inequality (heven : ∀ x, f x = f (-x))
  (hdecr : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → (x₁ - x₂) * (f x₁ - f x₂) < 0) :
  f c > f a ∧ f a > f b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l712_71219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_cosine_problem_l712_71266

/-- Given a point Q in coordinate space with all positive coordinates,
    if the line OQ forms angles α', β', and γ' with the x-, y-, and z-axes respectively,
    and cos α' = 2/5 and cos β' = 1/4, then cos γ' = √(311) / 20. -/
theorem direction_cosine_problem (Q : ℝ × ℝ × ℝ) 
  (h_positive : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0) 
  (α' β' γ' : ℝ) 
  (h_angles : α' = Real.arccos (Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)) ∧
              β' = Real.arccos (Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)) ∧
              γ' = Real.arccos (Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_cos_α' : Real.cos α' = 2/5)
  (h_cos_β' : Real.cos β' = 1/4) :
  Real.cos γ' = Real.sqrt 311 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_cosine_problem_l712_71266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_sum_l712_71261

/-- Represents the repeating decimal 3.71717171... -/
def repeating_decimal : ℚ := 3 + 71 / 99

/-- The sum of the numerator and denominator of the fraction 
    representing the repeating decimal when reduced to lowest terms -/
def sum_of_fraction : ℕ := 467

theorem repeating_decimal_fraction_sum : 
  (repeating_decimal.num.natAbs + repeating_decimal.den) = sum_of_fraction := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_sum_l712_71261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_curve_reflection_point_l712_71293

/-- A cubic curve with real coefficients -/
structure CubicCurve where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the cubic curve -/
noncomputable def CubicCurve.y (curve : CubicCurve) (x : ℝ) : ℝ :=
  x^3 + curve.a * x^2 + curve.b * x + curve.c

/-- The x-coordinate of the reflection point -/
noncomputable def CubicCurve.reflectionPointX (curve : CubicCurve) : ℝ :=
  -curve.a / 3

/-- The y-coordinate of the reflection point -/
noncomputable def CubicCurve.reflectionPointY (curve : CubicCurve) : ℝ :=
  2 * curve.a^3 / 27 - curve.a * curve.b / 3 + curve.c

/-- Theorem stating that the reflection point is on the curve and
    the curve is symmetric about this point -/
theorem cubic_curve_reflection_point (curve : CubicCurve) :
  let x₀ := curve.reflectionPointX
  let y₀ := curve.reflectionPointY
  (curve.y x₀ = y₀) ∧
  (∀ x y, curve.y x = y →
    curve.y (2 * x₀ - x) = 2 * y₀ - y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_curve_reflection_point_l712_71293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l712_71275

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 1 + Real.log x

-- Theorem statement
theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = x - 1 :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l712_71275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_property_l712_71240

theorem product_sum_property :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 2 ∧
  (∀ p : ℕ × ℕ, p ∈ s →
    let (x, y) := p
    10 ≤ x ∧ x < 100 ∧
    10 ≤ y ∧ y < 100 ∧
    x < y ∧
    1000 ≤ x * y ∧ x * y < 10000 ∧
    x * y / 1000 = 2 ∧
    x * y % 1000 = x + y) ∧
  ((30, 70) ∈ s ∧ (24, 88) ∈ s) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_property_l712_71240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelf_filling_l712_71222

theorem shelf_filling (P A G J K R : ℕ) (h1 : P ≠ A ∧ P ≠ G ∧ P ≠ J ∧ P ≠ K ∧ P ≠ R ∧
                                           A ≠ G ∧ A ≠ J ∧ A ≠ K ∧ A ≠ R ∧
                                           G ≠ J ∧ G ≠ K ∧ G ≠ R ∧
                                           J ≠ K ∧ J ≠ R ∧
                                           K ≠ R)
                      (h2 : P > 0 ∧ A > 0 ∧ G > 0 ∧ J > 0 ∧ K > 0 ∧ R > 0) :
  R = 2 * G + A / 2 + P ∨ R = J / 2 + K := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelf_filling_l712_71222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_cone_lateral_area_l712_71241

/-- A cone with an isosceles triangle as its axis section. -/
structure IsoscelesCone where
  /-- The base length of the axis section triangle. -/
  base_length : ℝ
  /-- The height of the axis section triangle. -/
  height : ℝ
  /-- The radius of the cone's base. -/
  radius : ℝ
  /-- The slant height of the cone. -/
  slant_height : ℝ

/-- The lateral area of an isosceles cone. -/
noncomputable def lateral_area (cone : IsoscelesCone) : ℝ :=
  Real.pi * cone.radius * cone.slant_height

/-- Theorem stating the lateral area of a specific isosceles cone. -/
theorem isosceles_cone_lateral_area :
  ∃ (cone : IsoscelesCone),
    cone.base_length = Real.sqrt 2 ∧
    cone.height = 1 ∧
    cone.radius = 1 ∧
    cone.slant_height = Real.sqrt 2 ∧
    lateral_area cone = Real.sqrt 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_cone_lateral_area_l712_71241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_evaluation_l712_71254

-- Define the expressions as noncomputable
noncomputable def expr1 : ℝ := Real.sqrt 4 + Real.sqrt 64 - (27 : ℝ) ^ (1/3) - |(-2)|
noncomputable def expr2 : ℝ := Real.sqrt 16 / ((-1) : ℝ) ^ (1/3) * Real.sqrt (1/4)

-- State the theorem
theorem expressions_evaluation :
  expr1 = 5 ∧ expr2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_evaluation_l712_71254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_prime_roots_l712_71296

theorem quadratic_equation_prime_roots (k : ℕ) : 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p + q = 58 ∧ p * q = k) ↔ 
  k ∈ ({265, 517, 697, 841} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_prime_roots_l712_71296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_101_l712_71234

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- State the properties of g
axiom g_property : ∀ (x y : ℝ), g (x * y) + 2 * x = x * g y + g x
axiom g_neg_one : g (-1) = 3
axiom g_one : g 1 = 1

-- Theorem to prove
theorem g_neg_101 : g (-101) = 103 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_101_l712_71234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_sale_price_l712_71214

def original_price : ℝ := 298
def first_discount : ℝ := 0.12
def second_discount : ℝ := 0.15

def sale_price : ℝ :=
  original_price * (1 - first_discount) * (1 - second_discount)

theorem saree_sale_price : 
  ⌊sale_price⌋ = 223 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_sale_price_l712_71214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l712_71229

/-- The parabola y² = 2px in the Cartesian coordinate system -/
def Parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

/-- The line m passing through C(p, 0) -/
def LineM (p m : ℝ) (x y : ℝ) : Prop := x = m*y + p

/-- The point N(-p, 0) -/
def PointN (p : ℝ) : ℝ × ℝ := (-p, 0)

/-- The dot product of vectors NA and NB -/
def DotProductNANB (p : ℝ) (A B : ℝ × ℝ) : ℝ :=
  let N := PointN p
  (A.1 - N.1) * (B.1 - N.1) + (A.2 - N.2) * (B.2 - N.2)

/-- The theorem stating the minimum value of NA · NB and the existence of line l -/
theorem parabola_intersection_theorem (p : ℝ) (hp : p > 0) :
  (∃ (m : ℝ) (A B : ℝ × ℝ),
    Parabola p A.1 A.2 ∧ Parabola p B.1 B.2 ∧
    LineM p m A.1 A.2 ∧ LineM p m B.1 B.2 ∧
    (∀ (m' : ℝ) (A' B' : ℝ × ℝ),
      Parabola p A'.1 A'.2 → Parabola p B'.1 B'.2 →
      LineM p m' A'.1 A'.2 → LineM p m' B'.1 B'.2 →
      DotProductNANB p A' B' ≥ DotProductNANB p A B)) ∧
  DotProductNANB p A B = 2 * p^2 ∧
  (∃ (a : ℝ), a = p / 2 ∧
    ∀ (A : ℝ × ℝ), Parabola p A.1 A.2 →
      ∃ (P Q : ℝ × ℝ),
        P.1 = a ∧ Q.1 = a ∧
        (P.1 - (A.1 + p) / 2)^2 + (P.2 - A.2 / 2)^2 = ((A.1 - p)^2 + A.2^2) / 4 ∧
        (Q.1 - (A.1 + p) / 2)^2 + (Q.2 - A.2 / 2)^2 = ((A.1 - p)^2 + A.2^2) / 4 ∧
        (P.2 - Q.2)^2 = p^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l712_71229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l712_71297

noncomputable def G (x : ℝ) : ℝ := 15 + 5*x

noncomputable def R (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 5 then -2*x^2 + 21*x + 1 else 56

noncomputable def f (x : ℝ) : ℝ := R x - G x

theorem max_profit :
  ∃ (x_max : ℝ), x_max = 4 ∧ 
  (∀ x, f x ≤ f x_max) ∧
  f x_max = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l712_71297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_fraction_l712_71239

theorem greatest_fraction (w x y z : ℚ) 
  (hw : 0 < w) (hx : w < x) (hy : x < y) (hz : y < z) :
  let a := (w + x + y) / (x + y + z)
  let b := (w + y + z) / (x + w + z)
  let c := (x + y + z) / (w + x + y)
  let d := (x + w + z) / (w + y + z)
  let e := (y + z + w) / (x + y + z)
  max a (max b (max c (max d e))) = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_fraction_l712_71239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l712_71271

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x^3 + Real.sqrt (1 + x^6))

-- Theorem statement
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l712_71271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_drivers_percentage_l712_71272

/-- Given a company with employees, calculate the percentage who drive to work -/
def percentage_employees_drive (total_employees : ℕ) 
  (public_transport_users : ℕ) : ℕ :=
  let non_drivers := 2 * public_transport_users
  let drivers := total_employees - non_drivers
  (drivers * 100) / total_employees

/-- The specific case for the given problem -/
theorem company_drivers_percentage : 
  percentage_employees_drive 100 20 = 60 := by
  -- Unfold the definition and simplify
  unfold percentage_employees_drive
  -- Perform the arithmetic
  norm_num
  -- QED

#eval percentage_employees_drive 100 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_drivers_percentage_l712_71272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l712_71257

/-- Circle C with center (a, 2) and radius 2 -/
def circleC (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - 2)^2 = 4

/-- Line l -/
def lineL (x y : ℝ) : Prop :=
  x - y + 3 = 0

/-- Distance from point (x, y) to line l -/
noncomputable def distanceToLine (x y : ℝ) : ℝ :=
  |x - y + 3| / Real.sqrt 2

/-- Chord length of circle C intercepted by line l -/
noncomputable def chordLength (a : ℝ) : ℝ := 2 * Real.sqrt 3

theorem circle_line_intersection (a : ℝ) (h1 : a > 0) :
  chordLength a = 2 * Real.sqrt 3 → a = Real.sqrt 2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l712_71257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_centroid_l712_71269

/-- The centroid of a tetrahedron -/
theorem tetrahedron_centroid
  (P A B C : EuclideanSpace ℝ (Fin 3))
  (G : EuclideanSpace ℝ (Fin 3))
  (a b c : EuclideanSpace ℝ (Fin 3))
  (h1 : A - P = a)
  (h2 : B - P = b)
  (h3 : C - P = c)
  (h4 : G = (1/4 : ℝ) • (P + A + B + C)) :
  G - P = (1/4 : ℝ) • (a + b + c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_centroid_l712_71269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_vector_l712_71227

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 2:3,
    prove that Q = (3/5)C + (2/5)D -/
theorem point_division_vector (C D Q : ℝ × ℝ × ℝ) : 
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ Q = (1 - t) • C + t • D) →  -- Q is on CD
  (2 * ‖Q - C‖ = 3 * ‖D - Q‖) →                        -- CQ:QD = 2:3
  Q = (3/5 : ℝ) • C + (2/5 : ℝ) • D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_vector_l712_71227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l712_71221

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x + Real.cos x) + 1/2

theorem function_properties :
  ∃ (α : ℝ), 
    (Real.tan α = 1/2) ∧ 
    (f α = 17/10) ∧
    (∀ (x : ℝ), f (x + π) = f x) ∧
    (∀ (k : ℤ), ∀ (x : ℝ), 
      (x ∈ Set.Icc (-3*π/8 + k*π) (π/8 + k*π) → 
        ∀ (y : ℝ), y ∈ Set.Icc (-3*π/8 + k*π) x → f y ≤ f x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l712_71221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_valid_part_number_l712_71218

def is_valid_part_number (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 700

def random_number_sequence : List Nat :=
  [253, 313, 457, 860, 736, 253, 007, 328, 523, 578]

theorem fifth_valid_part_number :
  (random_number_sequence.filter is_valid_part_number).get? 4 = some 328 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_valid_part_number_l712_71218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l712_71233

/-- Given two parallel vectors a and b in R², prove that 3a + 2b = (7, -14) -/
theorem parallel_vectors_sum (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (1, -2))
  (h2 : b = (2, m))
  (h3 : ∃ (k : ℝ), a = k • b) :
  (3 : ℝ) • a + (2 : ℝ) • b = (7, -14) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l712_71233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_apples_weight_l712_71230

-- Define constants for the given information
def monday_apples : Real := 15.5
def tuesday_multiplier : Real := 3.2
def wednesday_percentage : Real := 1.05
def kilo_to_pound : Real := 2.2
def kilo_to_gram : Real := 1000

-- Define the theorem
theorem total_apples_weight 
  (monday : Real := monday_apples)
  (tuesday_factor : Real := tuesday_multiplier)
  (wednesday_factor : Real := wednesday_percentage)
  (kilo_pound_ratio : Real := kilo_to_pound)
  (kilo_gram_ratio : Real := kilo_to_gram) :
  let tuesday := monday * tuesday_factor
  let wednesday := tuesday * wednesday_factor
  monday + tuesday + wednesday = 117.18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_apples_weight_l712_71230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l712_71212

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → (2 : ℝ)^a > (2 : ℝ)^b - 1) ↔ (a ≤ b → (2 : ℝ)^a ≤ (2 : ℝ)^b - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l712_71212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l712_71232

def sample_space : Finset Nat := {1, 2, 3, 4, 5, 6}
def event_A : Finset Nat := {1, 2, 3, 4}
def event_B : Finset Nat := {3, 4, 5, 6}

theorem conditional_probability_B_given_A :
  let P : Finset Nat → Rat := λ E => (E ∩ sample_space).card / sample_space.card
  (P (event_B ∩ event_A)) / (P event_A) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l712_71232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_f_on_interval_l712_71252

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (100 - x^2)

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-6) 8, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-6) 8, f x = max) ∧
    (∀ x ∈ Set.Icc (-6) 8, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-6) 8, f x = min) ∧
    max = 10 ∧ min = 6 := by
  sorry

#check max_min_f_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_f_on_interval_l712_71252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_conclude_dracula_alive_l712_71210

-- Define propositions
variable (P : Prop) -- "I am sane"
variable (Q : Prop) -- "Count Dracula is alive"
variable (R : Prop) -- "I am human"

-- Define the statements
def statement1 (P Q : Prop) : Prop := P → Q
def statement2 (P Q R : Prop) : Prop := (R ∧ P) → Q

-- Theorem: It's not possible to conclude Q from either statement
theorem cannot_conclude_dracula_alive (h1 : statement1 P Q) (h2 : statement2 P Q R) : 
  ¬(Q ∨ ¬Q) → False :=
by
  intro h
  apply h
  exact Classical.em Q

-- This theorem states that we cannot prove Q or ¬Q (the law of excluded middle for Q)
-- from the given statements, which means we cannot conclude whether Q is true or false.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_conclude_dracula_alive_l712_71210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_squared_greater_than_one_l712_71276

theorem tan_squared_greater_than_one (θ : ℝ) 
  (h1 : Real.sin (θ + π / 2) < 0) 
  (h2 : Real.cos (θ - π / 2) > 0) : 
  Real.tan θ ^ 2 > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_squared_greater_than_one_l712_71276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_from_cosine_l712_71273

theorem sine_value_from_cosine (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.cos (π + α) = 3/5) : Real.sin α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_from_cosine_l712_71273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_is_46_repeated_l712_71213

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a number as a list of digits -/
def Number := List Digit

/-- The initial 100-digit number -/
def initialNumber : Number :=
  (List.replicate 25 [⟨2, sorry⟩, ⟨1, sorry⟩, ⟨1, sorry⟩, ⟨6, sorry⟩]).join

/-- The operation of replacing two adjacent digits with their sum if possible -/
def replaceAdjacent (n : Number) : Option Number :=
  sorry

/-- Applies the replaceAdjacent operation until no further operations are possible -/
def applyOperationsUntilFixed (n : Number) : Number :=
  sorry

/-- The final number after all possible operations -/
def finalNumber : Number :=
  applyOperationsUntilFixed initialNumber

/-- Theorem: The final number is a 50-digit number composed of repeating "46" -/
theorem final_number_is_46_repeated : 
  finalNumber = (List.replicate 25 [⟨4, sorry⟩, ⟨6, sorry⟩]).join :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_is_46_repeated_l712_71213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l712_71228

def letters : List Char := ['F', 'R', 'E', 'E', 'D', 'O', 'M']

def valid_sequence (seq : List Char) : Bool :=
  seq.length = 4 &&
  seq.toFinset.card = 4 &&
  seq.head? = some 'F' &&
  seq.getLast? ≠ some 'M' &&
  seq.all (· ∈ letters)

def count_valid_sequences : Nat :=
  (letters.permutations.filter valid_sequence).length

theorem valid_sequences_count : count_valid_sequences = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l712_71228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_example_l712_71216

/-- Volume of a truncated right circular cone -/
noncomputable def truncated_cone_volume (R r h : ℝ) : ℝ :=
  (1/3) * Real.pi * h * (R^2 + R*r + r^2)

/-- The volume of a truncated right circular cone with large base radius 10 cm,
    small base radius 5 cm, and height 8 cm is (1400/3)π cm³ -/
theorem truncated_cone_volume_example :
  truncated_cone_volume 10 5 8 = (1400/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_example_l712_71216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_specific_case_l712_71283

/-- The ratio of the volume of a cone to the volume of a cylinder -/
noncomputable def volume_ratio (cylinder_height cylinder_radius cone_height : ℝ) : ℝ :=
  let cylinder_volume := Real.pi * cylinder_radius^2 * cylinder_height
  let cone_volume := (1/3) * Real.pi * cylinder_radius^2 * cone_height
  cone_volume / cylinder_volume

/-- Theorem stating that the ratio of the volume of a specific cone to a specific cylinder is 1/6 -/
theorem volume_ratio_specific_case :
  volume_ratio 10 5 5 = 1/6 := by
  -- Unfold the definition of volume_ratio
  unfold volume_ratio
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_specific_case_l712_71283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l712_71200

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def pseudo_circle (a b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2 + b^2

noncomputable def focus : ℝ × ℝ := (Real.sqrt 2, 0)

noncomputable def minor_to_focus_dist : ℝ := Real.sqrt 3

noncomputable def dot_product_AB_AD (m n : ℝ) : ℝ := (m - 2)^2 - n^2

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ x y, ellipse a b x y ∧ focus = (Real.sqrt 2, 0) ∧ 
    minor_to_focus_dist = Real.sqrt 3) →
  (∀ x y, ellipse a b x y ↔ ellipse (Real.sqrt 3) 1 x y) ∧
  (∀ x y, pseudo_circle a b x y ↔ x^2 + y^2 = 4) ∧
  (∀ m n, ellipse (Real.sqrt 3) 1 m n → 
    0 ≤ dot_product_AB_AD m n ∧ 
    dot_product_AB_AD m n < 7 + 4 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l712_71200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l712_71220

theorem cos_double_angle_special_case (x : ℝ) :
  x ∈ Set.Ioo (-3 * Real.pi / 4) (Real.pi / 4) →
  Real.cos (Real.pi / 4 - x) = -3 / 5 →
  Real.cos (2 * x) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l712_71220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_volunteer_arrangements_l712_71294

theorem adjacent_volunteer_arrangements :
  let n : ℕ := 6  -- Total number of volunteers and intersections
  let k : ℕ := 2  -- Number of specific volunteers (A and B)
  ∀ (arrangements : ℕ), 
    (arrangements = (n - k + 1) * Nat.factorial (n - k) * Nat.factorial k) →  -- Formula for arrangements
    arrangements = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_volunteer_arrangements_l712_71294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sugar_purchase_l712_71274

theorem min_sugar_purchase (f s : ℝ) (h1 : f ≥ 7 + s/2) (h2 : f ≤ 3*s) (h3 : s ≥ 0) (h4 : ∃ n : ℤ, s = n) :
  s ≥ 3 ∧ ∀ (t : ℝ), (∃ m : ℤ, t = m) → t ≥ 0 → (∃ (g : ℝ), g ≥ 7 + t/2 ∧ g ≤ 3*t) → t ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sugar_purchase_l712_71274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_l712_71268

/-- The number of students in a class where some are enrolled in English and German courses -/
theorem class_size (total both german english_only : ℕ) : 
  both + german + english_only = total ∧
  both + german = 22 ∧
  both = 12 ∧
  english_only = 10 →
  total = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_l712_71268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_groupings_l712_71264

-- Define the number of tourists
def n : ℕ := 7

-- Define the function to calculate the number of ways to split n objects into two non-empty groups
def split_ways (n : ℕ) : ℕ :=
  (List.range (n - 1)).map (λ k => Nat.choose n (k + 1)) |>.sum

-- Theorem statement
theorem tourist_groupings :
  split_ways n = 126 := by
  -- Unfold the definition of split_ways and n
  unfold split_ways n
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_groupings_l712_71264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_term_of_geometric_sequence_l712_71277

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ) (r : ℕ) : ℕ → ℕ
  | 0 => a
  | n + 1 => a * r^(n + 1)

theorem second_term_of_geometric_sequence :
  ∀ (r : ℕ),
  (GeometricSequence 5 r 0 = 5) →
  (GeometricSequence 5 r 4 = 1280) →
  (GeometricSequence 5 r 1 = 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_term_of_geometric_sequence_l712_71277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l712_71231

open Real

theorem triangle_problem (a b c A B C : ℝ) (m n : ℝ × ℝ) :
  0 < A ∧ A < 2 * π / 3 →
  0 < B ∧ B < 2 * π / 3 →
  0 < C ∧ C < π →
  c^2 = a^2 + b^2 - a*b →
  tan A - tan B = (Real.sqrt 3 / 3) * (1 + tan A * tan B) →
  m = (sin A, 1) →
  n = (3, cos (2*A)) →
  B = π / 4 ∧ 
  (∀ A : ℝ, m.1 * n.1 + m.2 * n.2 ≤ 17/8) ∧
  (∃ A : ℝ, m.1 * n.1 + m.2 * n.2 = 17/8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l712_71231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_pattern_l712_71208

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- Define the sequence of functions f_n
noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| n + 1 => deriv (f_n n)

-- State the theorem
theorem f_n_pattern (n : ℕ) (x : ℝ) :
  f_n n x = ((-1)^n * (x - n)) / Real.exp x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_pattern_l712_71208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_properties_l712_71206

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Calculate the x-intercept of a line -/
noncomputable def Line.xIntercept (l : Line) : ℝ := -l.c / l.a

/-- Calculate the y-intercept of a line -/
noncomputable def Line.yIntercept (l : Line) : ℝ := -l.c / l.b

/-- The specific line we're interested in -/
def specialLine : Line := { a := 1, b := 1, c := -1 }

theorem special_line_properties :
  specialLine.contains (-1) 2 ∧
  specialLine.xIntercept = specialLine.yIntercept := by
  sorry

#eval specialLine.a
#eval specialLine.b
#eval specialLine.c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_properties_l712_71206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_false_range_l712_71224

/-- Proposition p: For any x in [1,2], x² - a ≥ 0 -/
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

/-- Proposition q: There exists x in ℝ such that x² + 2ax + 2 - a = 0 -/
def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

/-- The range of a for which at least one of p and q is false -/
def range_a : Set ℝ :=
  Set.Ioo (-2) 1 ∪ Set.Ioi 1

theorem at_least_one_false_range (a : ℝ) :
  a ∈ range_a ↔ ¬(prop_p a ∧ prop_q a) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_false_range_l712_71224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meetup_distance_l712_71223

/-- Represents a car with a speed and start time -/
structure Car where
  speed : ℝ
  startTime : ℝ

/-- The problem setup -/
def problem_setup (distanceAB : ℝ) (carA carB : Car) : Prop :=
  distanceAB = 300 ∧
  carA.startTime = carB.startTime + 1 ∧
  (distanceAB / carA.speed) + carA.startTime = (distanceAB / carB.speed) + carB.startTime - 1

/-- The meetup point of the two cars -/
noncomputable def meetup_point (carA carB : Car) : ℝ :=
  (carA.speed * (carB.startTime - carA.startTime)) / (carA.speed - carB.speed)

/-- The main theorem -/
theorem cars_meetup_distance (distanceAB : ℝ) (carA carB : Car) :
  problem_setup distanceAB carA carB →
  300 - meetup_point carA carB = 150 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meetup_distance_l712_71223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sets_l712_71258

-- Define the sets of numbers
def set1 : List ℕ := [7, 8, 9]
def set2 : List ℕ := [12, 9, 15]
def set3 (a : ℕ) : List ℕ := [a^2, a^2 + 1, a^2 + 2]
def set4 (m n : ℕ) : List ℕ := [m^2 + n^2, m^2 - n^2, 2*m*n]

-- Define a function to check if a set forms a right-angled triangle
def isRightTriangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Theorem statement
theorem right_triangle_sets :
  (∃ (a b c : ℕ), a ∈ set2 ∧ b ∈ set2 ∧ c ∈ set2 ∧ isRightTriangle a b c) ∧
  (∀ m n : ℕ, m > n → ∃ (a b c : ℕ), a ∈ set4 m n ∧ b ∈ set4 m n ∧ c ∈ set4 m n ∧ isRightTriangle a b c) ∧
  (¬ ∃ (a b c : ℕ), a ∈ set1 ∧ b ∈ set1 ∧ c ∈ set1 ∧ isRightTriangle a b c) ∧
  (∀ a : ℕ, ¬ ∃ (x y z : ℕ), x ∈ set3 a ∧ y ∈ set3 a ∧ z ∈ set3 a ∧ isRightTriangle x y z) :=
by sorry

#check right_triangle_sets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sets_l712_71258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_3_l712_71245

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point on the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Focus of the hyperbola -/
noncomputable def focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt (h.a^2 + h.b^2), 0)

/-- Eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: Eccentricity of the hyperbola under specific conditions -/
theorem hyperbola_eccentricity_sqrt_3 (h : Hyperbola) (M : HyperbolaPoint h) :
  let F := focus h
  (∃ (θ : ℝ), θ = π/6 ∧ 
    (M.y - F.2) * Real.cos θ = (M.x - F.1) * Real.sin θ) →  -- Line through F with 30° inclination
  (M.x - F.1) * (M.y - F.2) = 0 →  -- MF perpendicular to x-axis
  eccentricity h = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_3_l712_71245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ganzgenau_theorem_l712_71289

/-- Represents the state of the microwave turntable -/
structure MicrowaveState where
  n : ℕ+  -- Number of positions (positive integer)
  position : Fin n  -- Current position of the tea cup

/-- Represents a strategy for Mr. Ganzgenau -/
def Strategy := MicrowaveState → ℕ

/-- Checks if a number is a power of 2 -/
def isPowerOfTwo (n : ℕ+) : Prop := ∃ k : ℕ, n = 2^k

/-- Represents the game between Mr. Ganzgenau and the microwave -/
def canGuaranteeFront (n : ℕ+) : Prop :=
  ∃ (strategy : Strategy),
    ∀ (initial : Fin n),
    ∃ (k : ℕ),
      (Nat.iterate
        (λ state =>
          let input := strategy state
          let newPos := (state.position + input : Fin n)  -- Clockwise rotation
          ⟨n, newPos⟩)
        k
        ⟨n, initial⟩).position = 0

/-- The main theorem: Mr. Ganzgenau can guarantee positioning the tea cup
    at the front if and only if n is a power of 2 -/
theorem ganzgenau_theorem (n : ℕ+) :
  canGuaranteeFront n ↔ isPowerOfTwo n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ganzgenau_theorem_l712_71289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_radius_l712_71209

/-- Given a sphere of radius R that forms a hemisphere of radius r with the same volume, 
    if r = 4∛2, then R = 4. --/
theorem sphere_to_hemisphere_radius (R r : ℝ) : 
  (4 / 3) * Real.pi * R^3 = (2 / 3) * Real.pi * r^3 →
  r = 4 * (2 : ℝ)^((1 : ℝ)/3) →
  R = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_radius_l712_71209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_perimeter_l712_71290

/-- Represents a rhombus with given diagonal lengths and side length -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  side : ℝ

/-- Calculates the area of a rhombus -/
noncomputable def area (r : Rhombus) : ℝ :=
  (r.diagonal1 * r.diagonal2) / 2

/-- Calculates the perimeter of a rhombus -/
noncomputable def perimeter (r : Rhombus) : ℝ :=
  4 * r.side

theorem rhombus_area_and_perimeter (r : Rhombus) 
  (h1 : r.diagonal1 = 18)
  (h2 : r.diagonal2 = 16)
  (h3 : r.side = 10) :
  area r = 144 ∧ perimeter r = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_perimeter_l712_71290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l712_71249

/-- Represents the lemonade mixture --/
structure Lemonade where
  lemon_juice : ℚ
  sugar : ℚ
  water : ℚ
  lime_juice : ℚ

/-- Calculates the total weight of the lemonade --/
def total_weight (l : Lemonade) : ℚ :=
  l.lemon_juice + l.sugar + l.water + l.lime_juice

/-- Calculates the total calories in the lemonade --/
def total_calories (l : Lemonade) : ℚ :=
  (30 * l.lemon_juice / 100) + (390 * l.sugar / 100) + (10 * l.lime_juice / 100)

/-- Calculates the calories in a given weight of lemonade --/
def calories_in_weight (l : Lemonade) (weight : ℚ) : ℚ :=
  (total_calories l) * weight / (total_weight l)

/-- The main theorem to prove --/
theorem lemonade_calories : ∃ (l : Lemonade),
  l.lemon_juice = 150 ∧
  l.sugar = 150 ∧
  l.water = 200 ∧
  l.lime_juice = 50 ∧
  Int.floor (calories_in_weight l 300) = 346 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l712_71249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_span_r2_iff_m_neq_12_l712_71279

open Matrix
open RealInnerProductSpace

def v1 : Fin 2 → ℝ := ![2, 4]
def v2 (m : ℝ) : Fin 2 → ℝ := ![6, m]

theorem span_r2_iff_m_neq_12 (m : ℝ) :
  Submodule.span ℝ {v1, v2 m} = ⊤ ↔ m ≠ 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_span_r2_iff_m_neq_12_l712_71279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_unique_numerators_l712_71298

/-- The set of rational numbers with repeating decimal expansion 0.ab̅ where 0 < r < 1 -/
def T : Set ℚ :=
  {r | 0 < r ∧ r < 1 ∧ ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ r = (10 * a + b) / 99}

/-- The number of unique numerators required to represent all elements of T in lowest terms -/
def unique_numerators : ℕ := 60

theorem count_unique_numerators :
  (Finset.filter (fun n : ℕ => n < 100 ∧ Nat.Coprime n 99) (Finset.range 100)).card = unique_numerators :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_unique_numerators_l712_71298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l712_71259

/-- The inclination angle of a line with equation y = mx + b is the angle between the line and the positive x-axis. -/
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m

theorem line_inclination_angle :
  let line_equation : ℝ → ℝ := λ x => x - 1
  inclination_angle 1 = 45 * (π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l712_71259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bao_moles_required_l712_71287

/-- Represents the number of moles of a substance -/
structure Moles where
  value : ℝ

/-- Represents the chemical reaction BaO + H2O → Ba(OH)2 -/
def reaction (bao : Moles) (h2o : Moles) (baoh2 : Moles) : Prop :=
  bao.value = h2o.value ∧ bao.value = baoh2.value

/-- The amount of H2O required is 1 mole -/
def h2o_required : Moles := ⟨1⟩

/-- The amount of Ba(OH)2 produced is 1 mole -/
def baoh2_produced : Moles := ⟨1⟩

theorem bao_moles_required : 
  ∃ (bao : Moles), reaction bao h2o_required baoh2_produced ∧ bao.value = 1 := by
  use ⟨1⟩
  constructor
  · simp [reaction, h2o_required, baoh2_produced]
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bao_moles_required_l712_71287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_set_l712_71244

theorem smallest_integer_in_set (n : ℤ) : 
  (∀ k : ℕ, k < 7 → (n + k : ℤ) ≥ n ∧ (n + k : ℤ) ≤ n + 6) →
  ((n + 6 : ℚ) < 2 * ((7 * n + 21) / 7 : ℚ)) →
  n ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_set_l712_71244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l712_71285

/-- The distance between two parallel lines in R^2 --/
noncomputable def distance_parallel_lines (a b d : ℝ × ℝ) : ℝ :=
  let v := (b.1 - a.1, b.2 - a.2)
  let proj_v_d := ((v.1 * d.1 + v.2 * d.2) / (d.1^2 + d.2^2)) • d
  let c := (v.1 - proj_v_d.1, v.2 - proj_v_d.2)
  Real.sqrt (c.1^2 + c.2^2)

/-- Theorem stating that the distance between the given parallel lines is 5/13 --/
theorem distance_specific_lines :
  distance_parallel_lines (4, -2) (3, -1) (2, -3) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l712_71285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_functions_list_contains_valid_functions_l712_71282

/-- A cubic polynomial function -/
def CubicFunction (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a*x^3 + b*x^2 + c*x + d

/-- The condition that f(x)f(-x) = f(x^3) -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x * f (-x) = f (x^3)

/-- The set of all cubic functions satisfying the condition -/
def ValidFunctions : Set (ℝ → ℝ) :=
  {f | ∃ a b c d, f = CubicFunction a b c d ∧ SatisfiesCondition f}

/-- The theorem stating that there are exactly 6 valid functions -/
theorem count_valid_functions :
  ∃ (s : Finset (ℝ → ℝ)), s.card = 6 ∧ ∀ f, f ∈ ValidFunctions ↔ f ∈ s := by
  sorry

/-- An explicit list of the 6 valid functions -/
def list_valid_functions : List (ℝ → ℝ) :=
  [ CubicFunction 0 0 0 0,
    CubicFunction 1 0 0 0,
    CubicFunction 1 1 0 0,
    CubicFunction 1 (-2) 0 0,
    CubicFunction 1 0 1 1,
    CubicFunction 1 0 (-1) 1 ]

/-- Theorem stating that the list contains exactly the valid functions -/
theorem list_contains_valid_functions :
  ∀ f, f ∈ ValidFunctions ↔ f ∈ list_valid_functions := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_functions_list_contains_valid_functions_l712_71282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_l712_71262

/-- Fixed point A -/
def A : ℝ × ℝ := (0, 0)

/-- Fixed point B -/
def B : ℝ × ℝ := (1, 3)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating the maximum value of |PA| * |PB| -/
theorem max_distance_product :
  ∃ (P : ℝ × ℝ), ∀ (Q : ℝ × ℝ),
    (distance P A) * (distance P B) ≥ (distance Q A) * (distance Q B) ∧
    (distance P A) * (distance P B) = 5 := by
  sorry

#check max_distance_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_l712_71262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l712_71256

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (3 * x + Real.pi / 4)

theorem phase_shift_of_f :
  ∃ (p : ℝ), ∀ (x : ℝ), f x = f (x + p) ∧ p = -Real.pi / 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l712_71256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_false_statements_range_l712_71250

/-- Represents the number of candies claimed by each child -/
def claimed_candies : Fin 5 → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5

/-- The total number of candies -/
def total_candies : ℕ := 10

/-- A distribution of candies is valid if it sums to the total number of candies -/
def valid_distribution (d : Fin 5 → ℕ) : Prop :=
  (Finset.univ.sum d) = total_candies

/-- The number of false statements in a given distribution -/
def false_statements (d : Fin 5 → ℕ) : ℕ :=
  Finset.card (Finset.filter (fun i => d i ≠ claimed_candies i) Finset.univ)

/-- Theorem stating that the number of false statements can range from 1 to 5 -/
theorem false_statements_range :
  ∃ (d₁ d₂ d₃ d₄ d₅ : Fin 5 → ℕ),
    valid_distribution d₁ ∧
    valid_distribution d₂ ∧
    valid_distribution d₃ ∧
    valid_distribution d₄ ∧
    valid_distribution d₅ ∧
    false_statements d₁ = 1 ∧
    false_statements d₂ = 2 ∧
    false_statements d₃ = 3 ∧
    false_statements d₄ = 4 ∧
    false_statements d₅ = 5 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_false_statements_range_l712_71250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_part1_monotonicity_of_f_part2_l712_71236

-- Define the function f(x) = x + a/x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

-- Part 1: Range of f(x) when a = 2 and D = (-∞, 0)
theorem range_of_f_part1 :
  ∀ y : ℝ, (∃ x : ℝ, x < 0 ∧ f 2 x = y) ↔ y ≤ -2 * Real.sqrt 2 := by sorry

-- Part 2: Monotonicity of f(x) when a = -1 and D = (0, +∞)
theorem monotonicity_of_f_part2 :
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f (-1) x₁ < f (-1) x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_part1_monotonicity_of_f_part2_l712_71236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l712_71204

-- Define the equation
def equation (a : ℝ) (x : ℝ) : Prop :=
  (2*a - 1) * Real.sin x + (2 - a) * Real.sin (2*x) = Real.sin (3*x)

-- Define the domain
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2*Real.pi

-- Define arithmetic sequence property for solutions
def arithmetic_sequence (a : ℝ) : Prop :=
  ∃ (s : Set ℝ), (∀ x, x ∈ s → domain x ∧ equation a x) ∧
                 (∃ d : ℝ, ∀ x y, x ∈ s → y ∈ s → ∃ n : ℤ, y = x + n * d)

-- Theorem statement
theorem equation_solution_range :
  {a : ℝ | arithmetic_sequence a} = Set.Iic (-2) ∪ {0} ∪ Set.Ici 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l712_71204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_intersection_area_theorem_horizontal_intersection_uniqueness_l712_71255

/-- The value of s for which a horizontal line y=s intersects triangle ABC to form
    triangle APQ with area 18, given the coordinates of A, B, and C. -/
noncomputable def horizontal_intersection_s : ℝ :=
  let A : ℝ × ℝ := (0, 10)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ × ℝ := (9, 0)
  10 - 2 * Real.sqrt 15

/-- Theorem stating that the calculated s value results in a triangle APQ with area 18. -/
theorem horizontal_intersection_area_theorem :
  let A : ℝ × ℝ := (0, 10)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ × ℝ := (9, 0)
  let s := horizontal_intersection_s
  let P := (3/10 * (10 - s), s)
  let Q := (9/10 * (10 - s), s)
  (1/2) * (Q.1 - P.1) * (A.2 - s) = 18 := by
  sorry

/-- Theorem stating that the calculated s is the unique value satisfying the conditions. -/
theorem horizontal_intersection_uniqueness (s : ℝ) :
  let A : ℝ × ℝ := (0, 10)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ × ℝ := (9, 0)
  let P := (3/10 * (10 - s), s)
  let Q := (9/10 * (10 - s), s)
  (1/2) * (Q.1 - P.1) * (A.2 - s) = 18 →
  s = horizontal_intersection_s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_intersection_area_theorem_horizontal_intersection_uniqueness_l712_71255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_count_l712_71263

noncomputable def f (x : ℝ) := 2 * abs (x * Real.sin x)

theorem f_zeros_count :
  ∃ (S : Finset ℝ), S.card = 5 ∧
  (∀ x ∈ S, -2*Real.pi ≤ x ∧ x ≤ 2*Real.pi ∧ f x = 0) ∧
  (∀ x, -2*Real.pi ≤ x ∧ x ≤ 2*Real.pi ∧ f x = 0 → x ∈ S) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_count_l712_71263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_cosine_monotonicity_l712_71205

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 3)

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem shifted_cosine_monotonicity (a : ℝ) :
  (is_monotone_increasing g 0 (a / 3) ∧ 
   is_monotone_increasing g (2 * a) (7 * Real.pi / 6)) ↔ 
  (Real.pi / 3 ≤ a ∧ a ≤ Real.pi / 2) := by
  sorry

#check shifted_cosine_monotonicity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_cosine_monotonicity_l712_71205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_circle_to_circle_l712_71253

-- Define the inversion transformation
def inversion (O : EuclideanSpace ℝ (Fin 2)) (P : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  sorry

-- Define a circle
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the property of a point being on a circle
def on_circle (P : EuclideanSpace ℝ (Fin 2)) (C : Circle) : Prop :=
  sorry

-- Define the property of a line passing through two points
def line_through (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

-- Define the intersection of a line and a circle
def line_circle_intersection (L : Set (EuclideanSpace ℝ (Fin 2))) (C : Circle) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

-- Theorem statement
theorem inversion_circle_to_circle
  (O : EuclideanSpace ℝ (Fin 2)) -- Center of inversion
  (S : Circle) -- Original circle
  (h_not_through_O : ¬ on_circle O S) -- S does not pass through O
  (center_line : Set (EuclideanSpace ℝ (Fin 2))) -- Line through O and center of S
  (A B : EuclideanSpace ℝ (Fin 2)) -- Intersection points of S and center_line
  (h_A_on_S : on_circle A S)
  (h_B_on_S : on_circle B S)
  (h_AB_on_line : A ∈ center_line ∧ B ∈ center_line)
  : ∃ (S' : Circle), -- There exists a circle S'
    ∀ (M : EuclideanSpace ℝ (Fin 2)), -- For all points M
      on_circle M S → -- If M is on S
      on_circle (inversion O M) S' ∧ -- Then the inversion of M is on S'
      (inversion O A) ∈ line_through S'.center (inversion O B) ∧ -- And A* is a point on the diameter of S'
      (inversion O B) ∈ line_through S'.center (inversion O A) -- And B* is a point on the diameter of S'
  :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_circle_to_circle_l712_71253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weeks_not_delivered_l712_71247

/-- Represents the weight of a newspaper in ounces -/
def weekdayPaperWeight : ℚ := 8

/-- Represents the number of papers to be delivered per day -/
def papersPerDay : ℕ := 250

/-- Represents the earnings in dollars for recycling one ton of paper -/
def earningsPerTon : ℚ := 100

/-- Calculates the total weight of papers for one week in ounces -/
def weeklyPaperWeight : ℚ :=
  6 * papersPerDay * weekdayPaperWeight +  -- Monday to Saturday
  papersPerDay * (2 * weekdayPaperWeight)  -- Sunday

/-- Converts ounces to tons -/
def ouncesToTons (ounces : ℚ) : ℚ :=
  ounces / (16 * 2000)

/-- Calculates the earnings for recycling one week's worth of papers -/
def weeklyEarnings : ℚ :=
  earningsPerTon * ouncesToTons weeklyPaperWeight

/-- The main theorem to be proved -/
theorem weeks_not_delivered : 
  ⌊earningsPerTon / weeklyEarnings⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weeks_not_delivered_l712_71247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rotation_fifth_position_l712_71251

/-- Represents a regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- Calculates the interior angle of a regular polygon -/
noncomputable def interior_angle (p : RegularPolygon) : ℝ :=
  (p.n - 2 : ℝ) * 180 / p.n

/-- Represents the square rolling around the octagon -/
structure RollingSquare where
  octagon : RegularPolygon
  octagon_is_octagon : octagon.n = 8
  position : ℕ
  position_le_5 : position ≤ 5

/-- Calculates the rotation of the square after rolling to a given position -/
noncomputable def square_rotation (rs : RollingSquare) : ℝ :=
  (rs.position - 1 : ℝ) * (360 - (interior_angle rs.octagon + 90))

/-- The main theorem: The square rotates 180 degrees when it reaches the fifth position -/
theorem square_rotation_fifth_position (rs : RollingSquare) (h : rs.position = 5) :
  square_rotation rs % 360 = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rotation_fifth_position_l712_71251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_quadrilateral_is_rectangle_l712_71246

/-- Quadrilateral type -/
def Quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  -- Add necessary conditions for a quadrilateral
  True -- Placeholder, replace with actual conditions

/-- Similar triangles (helper definition) -/
def SimilarTriangles' (A B C D E F : ℝ × ℝ) : Prop :=
  -- Add necessary conditions for similar triangles
  True -- Placeholder, replace with actual conditions

/-- Similar triangles in a quadrilateral -/
def SimilarTriangles (A B C D : ℝ × ℝ) : Prop :=
  SimilarTriangles' D A B C D A ∧
  SimilarTriangles' D A B B C D ∧
  SimilarTriangles' D A B A B C

/-- Rectangle type -/
def IsRectangle (A B C D : ℝ × ℝ) : Prop :=
  -- Add necessary conditions for a rectangle
  True -- Placeholder, replace with actual conditions

/-- A quadrilateral ABCD where all four triangles DAB, CDA, BCD, and ABC are similar to one another is a rectangle. -/
theorem similar_triangles_quadrilateral_is_rectangle
  (A B C D : ℝ × ℝ) -- Points in 2D plane
  (h_quad : Quadrilateral A B C D) -- ABCD is a quadrilateral
  (h_similar : SimilarTriangles A B C D) : -- All four triangles are similar
  IsRectangle A B C D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_quadrilateral_is_rectangle_l712_71246
