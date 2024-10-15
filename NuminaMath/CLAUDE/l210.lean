import Mathlib

namespace NUMINAMATH_CALUDE_special_rectangle_area_l210_21052

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  /-- The length of the rectangle -/
  length : ℝ
  /-- The width of the rectangle -/
  width : ℝ
  /-- The distance from the intersection of diagonals to the shorter side -/
  diag_dist : ℝ
  /-- Condition: The distance from the intersection of diagonals to the longer side is 2 cm more than to the shorter side -/
  diag_dist_diff : diag_dist + 2 = length / 2
  /-- Condition: The perimeter of the rectangle is 56 cm -/
  perimeter_cond : 2 * (length + width) = 56

/-- The area of a SpecialRectangle is 192 cm² -/
theorem special_rectangle_area (r : SpecialRectangle) : r.length * r.width = 192 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_area_l210_21052


namespace NUMINAMATH_CALUDE_square_root_problem_l210_21030

theorem square_root_problem (x y z : ℝ) 
  (h1 : Real.sqrt (2 * x + 1) = 0)
  (h2 : Real.sqrt y = 4)
  (h3 : z^3 = -27) :
  {r : ℝ | r^2 = 2*x + y + z} = {2 * Real.sqrt 3, -2 * Real.sqrt 3} := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l210_21030


namespace NUMINAMATH_CALUDE_divisibility_theorem_l210_21038

theorem divisibility_theorem (a b : ℕ+) (h : (7^2009 : ℕ) ∣ (a^2 + b^2)) :
  (7^2010 : ℕ) ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l210_21038


namespace NUMINAMATH_CALUDE_power_product_rule_l210_21049

theorem power_product_rule (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l210_21049


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l210_21025

theorem min_value_x_plus_2y (x y : ℝ) (h : x^2 + 4*y^2 - 2*x + 8*y + 1 = 0) :
  ∃ (m : ℝ), m = -2*Real.sqrt 2 - 1 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 - 2*a + 8*b + 1 = 0 → m ≤ a + 2*b :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l210_21025


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l210_21002

/-- The function f(x) = ln(2x) - 1 has a root in the interval (1, 2) -/
theorem f_has_root_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ Real.log (2 * x) - 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_f_has_root_in_interval_l210_21002


namespace NUMINAMATH_CALUDE_log_two_x_equals_neg_two_l210_21060

theorem log_two_x_equals_neg_two (x : ℝ) : 
  x = (Real.log 4 / Real.log 16) ^ (Real.log 16 / Real.log 4) → Real.log x / Real.log 2 = -2 :=
by sorry

end NUMINAMATH_CALUDE_log_two_x_equals_neg_two_l210_21060


namespace NUMINAMATH_CALUDE_value_of_a_l210_21075

/-- Proves that if 0.5% of a equals 95 paise, then a equals 190 rupees -/
theorem value_of_a (a : ℚ) : (0.5 / 100) * a = 95 / 100 → a = 190 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l210_21075


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l210_21097

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 4 → a^2 > 16) ∧ 
  (∃ a, a^2 > 16 ∧ ¬(a > 4)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l210_21097


namespace NUMINAMATH_CALUDE_austin_to_dallas_passes_three_buses_l210_21091

/-- Represents the time in hours since midnight -/
def Time := ℝ

/-- Represents the distance between Dallas and Austin in arbitrary units -/
def Distance := ℝ

/-- Represents the schedule and movement of buses -/
structure BusSchedule where
  departure_interval : ℝ
  departure_offset : ℝ
  trip_duration : ℝ

/-- Calculates the number of buses passed during a trip -/
def buses_passed (austin_schedule dallas_schedule : BusSchedule) : ℕ :=
  sorry

theorem austin_to_dallas_passes_three_buses 
  (austin_schedule : BusSchedule) 
  (dallas_schedule : BusSchedule) : 
  austin_schedule.departure_interval = 2 ∧ 
  austin_schedule.departure_offset = 0.5 ∧
  austin_schedule.trip_duration = 6 ∧
  dallas_schedule.departure_interval = 2 ∧
  dallas_schedule.departure_offset = 0 ∧
  dallas_schedule.trip_duration = 6 →
  buses_passed austin_schedule dallas_schedule = 3 :=
sorry

end NUMINAMATH_CALUDE_austin_to_dallas_passes_three_buses_l210_21091


namespace NUMINAMATH_CALUDE_cab_driver_income_l210_21050

/-- Proves that given the incomes for days 1, 3, 4, and 5, and the average income for all 5 days, the income for day 2 must be $50. -/
theorem cab_driver_income
  (income_day1 : ℕ)
  (income_day3 : ℕ)
  (income_day4 : ℕ)
  (income_day5 : ℕ)
  (average_income : ℕ)
  (h1 : income_day1 = 45)
  (h3 : income_day3 = 60)
  (h4 : income_day4 = 65)
  (h5 : income_day5 = 70)
  (h_avg : average_income = 58)
  : ∃ (income_day2 : ℕ), income_day2 = 50 ∧ 
    (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / 5 = average_income :=
by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l210_21050


namespace NUMINAMATH_CALUDE_find_b_l210_21039

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 4

-- Define the closed interval [2, 2b]
def interval (b : ℝ) : Set ℝ := Set.Icc 2 (2*b)

-- Theorem statement
theorem find_b : 
  ∃ (b : ℝ), b > 1 ∧ 
  (∀ x ∈ interval b, f x ∈ interval b) ∧
  (∀ y ∈ interval b, ∃ x ∈ interval b, f x = y) ∧
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l210_21039


namespace NUMINAMATH_CALUDE_sum_of_radii_is_14_l210_21000

/-- The sum of all possible radii of a circle that is tangent to both positive x and y-axes
    and externally tangent to another circle centered at (5,0) with radius 2 is equal to 14. -/
theorem sum_of_radii_is_14 : ∃ r₁ r₂ : ℝ,
  (∀ x y : ℝ, x^2 + y^2 = r₁^2 ∧ x ≥ 0 ∧ y ≥ 0 → (x = r₁ ∧ y = 0) ∨ (x = 0 ∧ y = r₁)) ∧
  (∀ x y : ℝ, x^2 + y^2 = r₂^2 ∧ x ≥ 0 ∧ y ≥ 0 → (x = r₂ ∧ y = 0) ∨ (x = 0 ∧ y = r₂)) ∧
  ((r₁ - 5)^2 + r₁^2 = (r₁ + 2)^2) ∧
  ((r₂ - 5)^2 + r₂^2 = (r₂ + 2)^2) ∧
  r₁ + r₂ = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_radii_is_14_l210_21000


namespace NUMINAMATH_CALUDE_polynomial_simplification_l210_21041

theorem polynomial_simplification (x : ℝ) :
  5 - 7*x - 13*x^2 + 10 + 15*x - 25*x^2 - 20 + 21*x + 33*x^2 - 15*x^3 =
  -15*x^3 - 5*x^2 + 29*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l210_21041


namespace NUMINAMATH_CALUDE_evaluate_expression_l210_21031

theorem evaluate_expression : -(16 / 2 * 8 - 72 + 4^2) = -8 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l210_21031


namespace NUMINAMATH_CALUDE_park_journey_distance_sum_l210_21027

/-- Represents the speed and start time of a traveler -/
structure Traveler where
  speed : ℚ
  startTime : ℚ

/-- The problem setup -/
def ParkJourney (d : ℚ) (patrick tanya jose : Traveler) : Prop :=
  patrick.speed > 0 ∧
  patrick.startTime = 0 ∧
  tanya.speed = patrick.speed + 2 ∧
  tanya.startTime = patrick.startTime + 1 ∧
  jose.speed = tanya.speed + 7 ∧
  jose.startTime = tanya.startTime + 1 ∧
  d / patrick.speed = (d / tanya.speed) + 1 ∧
  d / patrick.speed = (d / jose.speed) + 2

theorem park_journey_distance_sum :
  ∀ (d : ℚ) (patrick tanya jose : Traveler),
  ParkJourney d patrick tanya jose →
  ∃ (m n : ℕ), m.Coprime n ∧ d = m / n ∧ m + n = 277 := by
  sorry

end NUMINAMATH_CALUDE_park_journey_distance_sum_l210_21027


namespace NUMINAMATH_CALUDE_right_triangle_trig_l210_21053

theorem right_triangle_trig (D E F : ℝ) (h1 : D = 90) (h2 : E = 8) (h3 : F = 17) :
  let cosF := E / F
  let sinF := Real.sqrt (F^2 - E^2) / F
  cosF = 8 / 17 ∧ sinF = 15 / 17 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_trig_l210_21053


namespace NUMINAMATH_CALUDE_square_difference_of_solutions_l210_21037

theorem square_difference_of_solutions (α β : ℝ) : 
  α^2 = 2*α + 1 → β^2 = 2*β + 1 → α ≠ β → (α - β)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_solutions_l210_21037


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l210_21020

/-- Represents the ages of Somu and his father -/
structure Ages where
  somu : ℕ
  father : ℕ

/-- The condition that Somu's age 8 years ago was one-fifth of his father's age 8 years ago -/
def age_relation (ages : Ages) : Prop :=
  ages.somu - 8 = (ages.father - 8) / 5

/-- The theorem stating the ratio of Somu's age to his father's age -/
theorem somu_father_age_ratio :
  ∀ (ages : Ages),
  ages.somu = 16 →
  age_relation ages →
  (ages.somu : ℚ) / (ages.father : ℚ) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_somu_father_age_ratio_l210_21020


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l210_21046

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 7 = (y + 2)^2

-- Define the intersection points
def intersection_points : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem intersection_sum_zero :
  intersection_points →
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l210_21046


namespace NUMINAMATH_CALUDE_sugar_water_inequality_l210_21034

theorem sugar_water_inequality (a b c d : ℝ) : 
  a > b ∧ b > 0 ∧ c > d ∧ d > 0 → 
  (b + d) / (a + d) < (b + c) / (a + c) ∧
  ∀ (a b : ℝ), a > 0 ∧ b > 0 → 
  a / (1 + a + b) + b / (1 + a + b) < a / (1 + a) + b / (1 + b) := by
sorry

end NUMINAMATH_CALUDE_sugar_water_inequality_l210_21034


namespace NUMINAMATH_CALUDE_polygon_sides_l210_21096

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 → n = 8 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l210_21096


namespace NUMINAMATH_CALUDE_ball_ground_hit_time_l210_21056

/-- The time at which a ball hits the ground when thrown downward -/
theorem ball_ground_hit_time :
  let h (t : ℝ) := -16 * t^2 - 30 * t + 200
  ∃ t : ℝ, h t = 0 ∧ t = (-15 + Real.sqrt 3425) / 16 :=
by sorry

end NUMINAMATH_CALUDE_ball_ground_hit_time_l210_21056


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_inverse_l210_21040

theorem quadratic_roots_sum_inverse (p q : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + p*x1 + q = 0 ∧ x2^2 + p*x2 + q = 0)
  (h2 : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ x3^2 + q*x3 + p = 0 ∧ x4^2 + q*x4 + p = 0) :
  ∃ x1 x2 x3 x4 : ℝ, 
    x1 ≠ x2 ∧ x3 ≠ x4 ∧
    x1^2 + p*x1 + q = 0 ∧ x2^2 + p*x2 + q = 0 ∧
    x3^2 + q*x3 + p = 0 ∧ x4^2 + q*x4 + p = 0 ∧
    1/(x1*x3) + 1/(x1*x4) + 1/(x2*x3) + 1/(x2*x4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_inverse_l210_21040


namespace NUMINAMATH_CALUDE_pure_imaginary_m_l210_21013

/-- A complex number z is pure imaginary if and only if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z as a function of m. -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 2) (m^2 + 4*m - 5)

theorem pure_imaginary_m : ∃! m : ℝ, is_pure_imaginary (z m) ∧ m = -2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_l210_21013


namespace NUMINAMATH_CALUDE_area_of_ABCM_l210_21055

structure Polygon where
  sides : ℕ
  sideLength : ℝ
  rightAngles : Bool

def intersectionPoint (p1 p2 p3 p4 : Point) : Point := sorry

def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ := sorry

theorem area_of_ABCM (poly : Polygon) (A B C G J M : Point) :
  poly.sides = 14 ∧
  poly.sideLength = 3 ∧
  poly.rightAngles = true ∧
  M = intersectionPoint A G C J →
  quadrilateralArea A B C M = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_area_of_ABCM_l210_21055


namespace NUMINAMATH_CALUDE_line_intersection_area_ratio_l210_21077

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ :=
  sorry

/-- Theorem: Given a line y = c - x where 0 < c < 6, intersecting the y-axis at P
    and the line x = 6 at S, if the ratio of the area of triangle QRS to the area
    of triangle QOP is 4:16, then c = 4 -/
theorem line_intersection_area_ratio (c : ℝ) 
  (h1 : 0 < c) (h2 : c < 6) : 
  let l : Line := { m := -1, b := c }
  let P : Point := { x := 0, y := c }
  let S : Point := { x := 6, y := c - 6 }
  let Q : Point := { x := c, y := 0 }
  let R : Point := { x := 6, y := 0 }
  let O : Point := { x := 0, y := 0 }
  triangleArea Q R S / triangleArea Q O P = 4 / 16 →
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_area_ratio_l210_21077


namespace NUMINAMATH_CALUDE_distinct_arrangements_of_six_l210_21022

theorem distinct_arrangements_of_six (n : ℕ) (h : n = 6) : 
  Nat.factorial n = 720 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_of_six_l210_21022


namespace NUMINAMATH_CALUDE_sports_equipment_purchase_l210_21021

/-- Represents the purchase of sports equipment --/
structure Equipment where
  price_a : ℕ  -- price of type A equipment
  price_b : ℕ  -- price of type B equipment
  quantity_a : ℕ  -- quantity of type A equipment purchased
  quantity_b : ℕ  -- quantity of type B equipment purchased

/-- The main theorem about the sports equipment purchase --/
theorem sports_equipment_purchase 
  (e : Equipment) 
  (h1 : e.price_b = e.price_a + 10)  -- price difference condition
  (h2 : e.quantity_a * e.price_a = 300)  -- total cost of A
  (h3 : e.quantity_b * e.price_b = 360)  -- total cost of B
  (h4 : e.quantity_a = e.quantity_b)  -- equal quantities purchased
  : 
  (e.price_a = 50 ∧ e.price_b = 60) ∧  -- correct prices
  (∀ m n : ℕ, 
    50 * m + 60 * n = 1000 ↔ 
    ((m = 14 ∧ n = 5) ∨ (m = 8 ∧ n = 10) ∨ (m = 2 ∧ n = 15))) -- possible scenarios
  := by sorry


end NUMINAMATH_CALUDE_sports_equipment_purchase_l210_21021


namespace NUMINAMATH_CALUDE_not_all_vertices_on_same_branch_coordinates_of_Q_and_R_l210_21057

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define the branches of the hyperbola
def branch1 (x y : ℝ) : Prop := hyperbola x y ∧ x > 0 ∧ y > 0
def branch2 (x y : ℝ) : Prop := hyperbola x y ∧ x < 0 ∧ y < 0

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx)^2 + (py - qy)^2 = (qx - rx)^2 + (qy - ry)^2 ∧
  (qx - rx)^2 + (qy - ry)^2 = (rx - px)^2 + (ry - py)^2

-- Theorem 1: Not all vertices can lie on the same branch
theorem not_all_vertices_on_same_branch 
  (P Q R : ℝ × ℝ) 
  (h_triangle : is_equilateral_triangle P Q R)
  (h_on_hyperbola : hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2 ∧ hyperbola R.1 R.2) :
  ¬(branch1 P.1 P.2 ∧ branch1 Q.1 Q.2 ∧ branch1 R.1 R.2) ∧
  ¬(branch2 P.1 P.2 ∧ branch2 Q.1 Q.2 ∧ branch2 R.1 R.2) :=
sorry

-- Theorem 2: Coordinates of Q and R given P(-1, -1)
theorem coordinates_of_Q_and_R
  (P Q R : ℝ × ℝ)
  (h_triangle : is_equilateral_triangle P Q R)
  (h_on_hyperbola : hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2 ∧ hyperbola R.1 R.2)
  (h_P : P = (-1, -1))
  (h_Q_R_branch1 : branch1 Q.1 Q.2 ∧ branch1 R.1 R.2) :
  (Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) ∨
  (Q = (2 + Real.sqrt 3, 2 - Real.sqrt 3) ∧ R = (2 - Real.sqrt 3, 2 + Real.sqrt 3)) :=
sorry

end NUMINAMATH_CALUDE_not_all_vertices_on_same_branch_coordinates_of_Q_and_R_l210_21057


namespace NUMINAMATH_CALUDE_product_evaluation_l210_21016

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l210_21016


namespace NUMINAMATH_CALUDE_linear_function_inverse_sum_l210_21078

/-- Given a linear function f and its inverse f⁻¹, prove that a + b + c = 0 --/
theorem linear_function_inverse_sum (a b c : ℝ) 
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x + b)
  (h2 : ∀ x, f_inv x = b * x + a + c)
  (h3 : ∀ x, f (f_inv x) = x) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_inverse_sum_l210_21078


namespace NUMINAMATH_CALUDE_sin_alpha_for_given_point_l210_21070

theorem sin_alpha_for_given_point : ∀ α : Real,
  let x : Real := -2
  let y : Real := 2 * Real.sqrt 3
  let r : Real := Real.sqrt (x^2 + y^2)
  (∃ A : ℝ × ℝ, A = (x, y) ∧ A.1 = r * Real.cos α ∧ A.2 = r * Real.sin α) →
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_given_point_l210_21070


namespace NUMINAMATH_CALUDE_fermat_number_large_prime_factor_l210_21018

/-- Fermat number -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Theorem: For n ≥ 3, F_n has a prime factor greater than 2^(n+2)(n+1) -/
theorem fermat_number_large_prime_factor (n : ℕ) (h : n ≥ 3) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ F n ∧ p > 2^(n+2) * (n+1) := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_large_prime_factor_l210_21018


namespace NUMINAMATH_CALUDE_min_payment_bound_l210_21066

/-- Tea set price in yuan -/
def tea_set_price : ℕ := 200

/-- Tea bowl price in yuan -/
def tea_bowl_price : ℕ := 20

/-- Number of tea sets purchased -/
def num_tea_sets : ℕ := 30

/-- Discount factor for Option 2 -/
def discount_factor : ℚ := 95 / 100

/-- Payment for Option 1 given x tea bowls -/
def payment_option1 (x : ℕ) : ℕ := 20 * x + 5400

/-- Payment for Option 2 given x tea bowls -/
def payment_option2 (x : ℕ) : ℕ := 19 * x + 5700

/-- Theorem: The minimum payment is less than or equal to the minimum of Option 1 and Option 2 -/
theorem min_payment_bound (x : ℕ) (hx : x > 30) :
  ∃ (y : ℕ), y ≤ min (payment_option1 x) (payment_option2 x) ∧
  y = num_tea_sets * tea_set_price + x * tea_bowl_price -
      (min num_tea_sets x) * tea_bowl_price +
      ((x - min num_tea_sets x) * tea_bowl_price * discount_factor).floor :=
sorry

end NUMINAMATH_CALUDE_min_payment_bound_l210_21066


namespace NUMINAMATH_CALUDE_no_triangle_solution_l210_21074

theorem no_triangle_solution (a b c : ℝ) (A B C : ℝ) :
  b = 4 →
  c = 2 →
  C = π / 3 →
  ¬ (∃ (a : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (A > 0 ∧ B > 0 ∧ C > 0) ∧
    (A + B + C = π) ∧
    (a / Real.sin A = b / Real.sin B) ∧
    (b / Real.sin B = c / Real.sin C) ∧
    (c / Real.sin C = a / Real.sin A)) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_solution_l210_21074


namespace NUMINAMATH_CALUDE_solve_for_m_l210_21086

/-- Given that x = -2, y = 1, and mx + 3y = 7, prove that m = -2 -/
theorem solve_for_m (x y m : ℝ) 
  (hx : x = -2) 
  (hy : y = 1) 
  (heq : m * x + 3 * y = 7) : 
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_m_l210_21086


namespace NUMINAMATH_CALUDE_vlad_sister_height_l210_21069

/-- Converts feet and inches to total inches -/
def height_to_inches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- Converts total inches to feet (discarding remaining inches) -/
def inches_to_feet (inches : ℕ) : ℕ := inches / 12

theorem vlad_sister_height :
  let vlad_height := height_to_inches 6 3
  let height_diff := 41
  let sister_inches := vlad_height - height_diff
  inches_to_feet sister_inches = 2 := by sorry

end NUMINAMATH_CALUDE_vlad_sister_height_l210_21069


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l210_21092

theorem sqrt_expression_equality : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) + Real.sqrt 20 = 2 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l210_21092


namespace NUMINAMATH_CALUDE_valentine_card_cost_l210_21047

def total_students : ℕ := 30
def valentine_percentage : ℚ := 60 / 100
def initial_money : ℚ := 40
def spending_percentage : ℚ := 90 / 100

theorem valentine_card_cost :
  let students_receiving := total_students * valentine_percentage
  let money_spent := initial_money * spending_percentage
  let cost_per_card := money_spent / students_receiving
  cost_per_card = 2 := by sorry

end NUMINAMATH_CALUDE_valentine_card_cost_l210_21047


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l210_21007

def p (x : ℝ) : ℝ := 4*x^4 - 5*x^3 - 30*x^2 + 40*x + 24

theorem roots_of_polynomial :
  {x : ℝ | p x = 0} = {3, -1, -2, 1} :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l210_21007


namespace NUMINAMATH_CALUDE_min_value_expression_l210_21083

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1 / y^2) * (x + 1 / y^2 - 500) + (y + 1 / x^2) * (y + 1 / x^2 - 500) ≥ -125000 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l210_21083


namespace NUMINAMATH_CALUDE_closest_to_fraction_l210_21005

def fraction : ℚ := 805 / 0.410

def options : List ℚ := [0.4, 4, 40, 400, 4000]

theorem closest_to_fraction :
  ∃ (x : ℚ), x ∈ options ∧ 
  ∀ (y : ℚ), y ∈ options → |fraction - x| ≤ |fraction - y| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l210_21005


namespace NUMINAMATH_CALUDE_original_lines_per_sheet_l210_21079

/-- Represents the number of lines on each sheet in the original report -/
def L : ℕ := 56

/-- The number of sheets in the original report -/
def original_sheets : ℕ := 20

/-- The number of characters per line in the original report -/
def original_chars_per_line : ℕ := 65

/-- The number of lines per sheet in the retyped report -/
def new_lines_per_sheet : ℕ := 65

/-- The number of characters per line in the retyped report -/
def new_chars_per_line : ℕ := 70

/-- The percentage reduction in the number of sheets -/
def reduction_percentage : ℚ := 20 / 100

theorem original_lines_per_sheet :
  L = 56 ∧
  original_sheets * L * original_chars_per_line = 
    (original_sheets * (1 - reduction_percentage)).floor * new_lines_per_sheet * new_chars_per_line :=
by sorry

end NUMINAMATH_CALUDE_original_lines_per_sheet_l210_21079


namespace NUMINAMATH_CALUDE_house_to_school_distance_house_to_school_distance_is_60_l210_21014

/-- The distance between a house and a school, given travel times at different speeds -/
theorem house_to_school_distance : ℝ :=
  let speed_slow : ℝ := 10  -- km/hr
  let speed_fast : ℝ := 20  -- km/hr
  let time_late : ℝ := 2    -- hours
  let time_early : ℝ := 1   -- hours
  let distance : ℝ := 60    -- km

  have h1 : distance = speed_slow * (distance / speed_slow + time_late) := by sorry
  have h2 : distance = speed_fast * (distance / speed_fast - time_early) := by sorry

  distance

/-- The proof that the distance is indeed 60 km -/
theorem house_to_school_distance_is_60 : house_to_school_distance = 60 := by sorry

end NUMINAMATH_CALUDE_house_to_school_distance_house_to_school_distance_is_60_l210_21014


namespace NUMINAMATH_CALUDE_inequality_and_equality_cases_l210_21048

theorem inequality_and_equality_cases (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 ∧
  ((a - b) * (b - c) * (a - c) = 2 ↔ 
    ((a = 2 ∧ b = 1 ∧ c = 0) ∨ 
     (a = 1 ∧ b = 0 ∧ c = 2) ∨ 
     (a = 0 ∧ b = 2 ∧ c = 1))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_cases_l210_21048


namespace NUMINAMATH_CALUDE_gathering_gift_equation_l210_21081

/-- Represents a gathering where gifts are exchanged -/
structure Gathering where
  attendees : ℕ
  gifts_exchanged : ℕ
  gift_exchange_rule : attendees > 0 → gifts_exchanged = attendees * (attendees - 1)

/-- Theorem: In a gathering where each pair of attendees exchanges a different small gift,
    if the total number of gifts exchanged is 56 and the number of attendees is x,
    then x(x-1) = 56 -/
theorem gathering_gift_equation (g : Gathering) (h1 : g.gifts_exchanged = 56) :
  g.attendees * (g.attendees - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_gathering_gift_equation_l210_21081


namespace NUMINAMATH_CALUDE_harmonic_mean_leq_geometric_mean_l210_21051

theorem harmonic_mean_leq_geometric_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2 / (1/a + 1/b) ≤ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_leq_geometric_mean_l210_21051


namespace NUMINAMATH_CALUDE_simplified_fraction_sum_l210_21085

theorem simplified_fraction_sum (a b : ℕ) (h : a = 54 ∧ b = 81) :
  let g := Nat.gcd a b
  (a / g) + (b / g) = 5 := by
sorry

end NUMINAMATH_CALUDE_simplified_fraction_sum_l210_21085


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_proportions_l210_21026

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Calculates the total population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Represents the sample sizes for each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Calculates the total sample size -/
def sampleSize (s : Sample) : ℕ :=
  s.elderly + s.middleAged + s.young

/-- Checks if the sample is proportional to the population -/
def isProportionalSample (p : Population) (s : Sample) : Prop :=
  s.elderly * totalPopulation p = p.elderly * sampleSize s ∧
  s.middleAged * totalPopulation p = p.middleAged * sampleSize s ∧
  s.young * totalPopulation p = p.young * sampleSize s

theorem stratified_sampling_correct_proportions 
  (p : Population) 
  (s : Sample) :
  p.elderly = 28 → 
  p.middleAged = 56 → 
  p.young = 84 → 
  sampleSize s = 36 → 
  isProportionalSample p s → 
  s.elderly = 6 ∧ s.middleAged = 12 ∧ s.young = 18 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_proportions_l210_21026


namespace NUMINAMATH_CALUDE_diamond_three_two_l210_21045

def diamond (a b : ℝ) : ℝ := a * b^3 - b^2 + 1

theorem diamond_three_two : diamond 3 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_two_l210_21045


namespace NUMINAMATH_CALUDE_andy_math_problem_l210_21044

theorem andy_math_problem (last_problem : ℕ) (total_solved : ℕ) (start_problem : ℕ) :
  last_problem = 125 →
  total_solved = 56 →
  start_problem = last_problem - total_solved + 1 →
  start_problem = 70 := by
sorry

end NUMINAMATH_CALUDE_andy_math_problem_l210_21044


namespace NUMINAMATH_CALUDE_tan_product_thirty_degrees_l210_21035

theorem tan_product_thirty_degrees :
  let A : Real := 30 * π / 180
  let B : Real := 30 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_thirty_degrees_l210_21035


namespace NUMINAMATH_CALUDE_podium_cube_count_theorem_l210_21008

/-- Represents a three-step podium made of wooden cubes -/
structure Podium where
  total_cubes : ℕ
  no_white_faces : ℕ
  one_white_face : ℕ
  two_white_faces : ℕ
  three_white_faces : ℕ

/-- The podium is valid if it satisfies the conditions of the problem -/
def is_valid_podium (p : Podium) : Prop :=
  p.total_cubes = 144 ∧
  p.no_white_faces = 40 ∧
  p.one_white_face = 64 ∧
  p.two_white_faces = 32 ∧
  p.three_white_faces = 8

/-- Theorem stating that the sum of cubes with 0, 1, 2, and 3 white faces
    equals the total number of cubes, implying no cubes with 4, 5, or 6 white faces -/
theorem podium_cube_count_theorem (p : Podium) (h : is_valid_podium p) :
  p.no_white_faces + p.one_white_face + p.two_white_faces + p.three_white_faces = p.total_cubes :=
by sorry


end NUMINAMATH_CALUDE_podium_cube_count_theorem_l210_21008


namespace NUMINAMATH_CALUDE_equation_describes_two_lines_l210_21032

/-- The set of points satisfying the equation (x-y)^2 = x^2 + y^2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

/-- The union of x-axis and y-axis -/
def TwoLines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

theorem equation_describes_two_lines : S = TwoLines := by
  sorry

end NUMINAMATH_CALUDE_equation_describes_two_lines_l210_21032


namespace NUMINAMATH_CALUDE_z₂_in_fourth_quadrant_z₂_equals_z₁_times_ni_l210_21019

-- Define complex numbers z₁ and z₂
def z₁ (m : ℝ) : ℂ := m + Complex.I
def z₂ (m : ℝ) : ℂ := m + (m - 2) * Complex.I

-- Theorem 1: If z₂ is in the fourth quadrant, then 0 < m < 2
theorem z₂_in_fourth_quadrant (m : ℝ) :
  (z₂ m).re > 0 ∧ (z₂ m).im < 0 → 0 < m ∧ m < 2 := by sorry

-- Theorem 2: If z₂ = z₁ · ni, then (m = 1 and n = -1) or (m = -2 and n = 2)
theorem z₂_equals_z₁_times_ni (m n : ℝ) :
  z₂ m = z₁ m * (n * Complex.I) →
  (m = 1 ∧ n = -1) ∨ (m = -2 ∧ n = 2) := by sorry

end NUMINAMATH_CALUDE_z₂_in_fourth_quadrant_z₂_equals_z₁_times_ni_l210_21019


namespace NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_l210_21064

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial_sum : ℕ := (List.range 20).foldl (fun acc i => acc + factorial ((i + 1) * 5)) 0

theorem last_two_digits_of_factorial_sum :
  last_two_digits factorial_sum = 20 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_l210_21064


namespace NUMINAMATH_CALUDE_xiaozhao_journey_l210_21058

def movements : List Int := [1000, -900, 700, -1200, 1200, 100, -1100, -200]

def calorie_per_km : Nat := 7000

def final_position (moves : List Int) : Int :=
  moves.sum

def total_distance (moves : List Int) : Nat :=
  moves.map (Int.natAbs) |>.sum

theorem xiaozhao_journey :
  let pos := final_position movements
  let dist := total_distance movements
  (pos < 0 ∧ pos.natAbs = 400) ∧
  (dist * calorie_per_km / 1000 = 44800) := by
  sorry

end NUMINAMATH_CALUDE_xiaozhao_journey_l210_21058


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l210_21082

/-- Represents a repeating decimal with a three-digit repetend -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_subtraction :
  RepeatingDecimal 8 6 4 - RepeatingDecimal 5 7 9 - RepeatingDecimal 1 3 5 = 50 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l210_21082


namespace NUMINAMATH_CALUDE_arrangement_exists_l210_21011

theorem arrangement_exists : ∃ (p : Fin 100 → Fin 100), Function.Bijective p ∧ 
  ∀ i : Fin 99, 
    (((p (i + 1)).val = (p i).val + 2) ∨ ((p (i + 1)).val = (p i).val - 2)) ∨
    ((p (i + 1)).val = 2 * (p i).val) ∨ ((p i).val = 2 * (p (i + 1)).val) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_exists_l210_21011


namespace NUMINAMATH_CALUDE_some_number_value_l210_21084

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 25 * some_number * 63) :
  some_number = 35 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l210_21084


namespace NUMINAMATH_CALUDE_value_range_of_f_l210_21009

-- Define the function
def f (x : ℝ) : ℝ := |x - 3| + 1

-- Define the domain
def domain : Set ℝ := Set.Icc 0 9

-- Theorem statement
theorem value_range_of_f :
  Set.Icc 1 7 = (Set.image f domain) := by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l210_21009


namespace NUMINAMATH_CALUDE_new_sales_tax_percentage_l210_21073

theorem new_sales_tax_percentage
  (original_tax : ℝ)
  (market_price : ℝ)
  (savings : ℝ)
  (h1 : original_tax = 3.5)
  (h2 : market_price = 8400)
  (h3 : savings = 14)
  : ∃ (new_tax : ℝ), new_tax = 10/3 ∧ 
    new_tax / 100 * market_price = original_tax / 100 * market_price - savings :=
sorry

end NUMINAMATH_CALUDE_new_sales_tax_percentage_l210_21073


namespace NUMINAMATH_CALUDE_sector_circumference_l210_21033

/-- The circumference of a sector with central angle 60° and radius 15 cm is 5(6 + π) cm. -/
theorem sector_circumference :
  let θ : ℝ := 60  -- Central angle in degrees
  let r : ℝ := 15  -- Radius in cm
  let arc_length : ℝ := (θ / 360) * (2 * π * r)
  let circumference : ℝ := arc_length + 2 * r
  circumference = 5 * (6 + π) := by sorry

end NUMINAMATH_CALUDE_sector_circumference_l210_21033


namespace NUMINAMATH_CALUDE_correct_stool_height_l210_21095

/-- Calculates the height of a stool needed to reach a light bulb. -/
def stool_height (ceiling_height room_height alice_height alice_reach book_thickness : ℝ) : ℝ :=
  ceiling_height - room_height - (alice_height + alice_reach + book_thickness)

/-- Theorem stating the correct height of the stool needed. -/
theorem correct_stool_height :
  let ceiling_height : ℝ := 300
  let light_bulb_below_ceiling : ℝ := 15
  let alice_height : ℝ := 160
  let alice_reach : ℝ := 50
  let book_thickness : ℝ := 5
  stool_height ceiling_height light_bulb_below_ceiling alice_height alice_reach book_thickness = 70 := by
  sorry

#eval stool_height 300 15 160 50 5

end NUMINAMATH_CALUDE_correct_stool_height_l210_21095


namespace NUMINAMATH_CALUDE_pairball_playing_time_l210_21024

theorem pairball_playing_time (total_time : ℕ) (num_children : ℕ) : 
  total_time = 90 → num_children = 5 → (total_time * 2) / num_children = 36 := by
  sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l210_21024


namespace NUMINAMATH_CALUDE_cost_price_calculation_l210_21004

/-- Calculates the cost price of an article given the final sale price, sales tax rate, and profit rate. -/
theorem cost_price_calculation (final_price : ℝ) (sales_tax_rate : ℝ) (profit_rate : ℝ) :
  final_price = 616 →
  sales_tax_rate = 0.1 →
  profit_rate = 0.16 →
  ∃ (cost_price : ℝ),
    cost_price > 0 ∧
    (cost_price * (1 + profit_rate) * (1 + sales_tax_rate) = final_price) ∧
    (abs (cost_price - 482.76) < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l210_21004


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l210_21036

/-- Given an ellipse with equation x²/m + y²/4 = 1 and focal length 4,
    prove that the length of its major axis is 4√2. -/
theorem ellipse_major_axis_length (m : ℝ) :
  (∀ x y : ℝ, x^2 / m + y^2 / 4 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c = 4 ∧ c^2 = m - 4) →     -- Focal length is 4
  (∃ a : ℝ, a = 2 * Real.sqrt 2 ∧ 2 * a = 4 * Real.sqrt 2) := -- Major axis length is 4√2
by sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l210_21036


namespace NUMINAMATH_CALUDE_angle_relationship_l210_21080

theorem angle_relationship (angle1 angle2 angle3 angle4 : Real) :
  (angle1 + angle2 = 90) →  -- angle1 and angle2 are complementary
  (angle3 + angle4 = 180) →  -- angle3 and angle4 are supplementary
  (angle1 = angle3) →  -- angle1 equals angle3
  (angle2 + 90 = angle4) :=  -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_angle_relationship_l210_21080


namespace NUMINAMATH_CALUDE_gcd_problem_l210_21001

-- Define the operation * as the greatest common divisor
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

-- State the theorem
theorem gcd_problem : gcd_op (gcd_op 20 16) (gcd_op 18 24) = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_problem_l210_21001


namespace NUMINAMATH_CALUDE_water_distribution_l210_21068

theorem water_distribution (total_water : ℕ) (size_8oz : ℕ) (size_5oz : ℕ) (size_4oz : ℕ) 
  (num_8oz : ℕ) (num_5oz : ℕ) :
  total_water = 122 →
  size_8oz = 8 →
  size_5oz = 5 →
  size_4oz = 4 →
  num_8oz = 4 →
  num_5oz = 6 →
  (total_water - (num_8oz * size_8oz + num_5oz * size_5oz)) / size_4oz = 15 := by
sorry

end NUMINAMATH_CALUDE_water_distribution_l210_21068


namespace NUMINAMATH_CALUDE_inverse_function_b_value_l210_21043

/-- Given a function f and its inverse, prove the value of b -/
theorem inverse_function_b_value 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 1 / (2 * x + b)) 
  (h2 : ∀ x, f⁻¹ x = (2 - 3 * x) / (5 * x)) 
  : b = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_b_value_l210_21043


namespace NUMINAMATH_CALUDE_sheila_work_hours_l210_21093

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  mwf_hours : ℝ  -- Hours worked on Monday, Wednesday, and Friday combined
  tt_hours : ℝ   -- Hours worked on Tuesday and Thursday combined
  hourly_rate : ℝ -- Hourly rate in dollars
  weekly_earnings : ℝ -- Total weekly earnings in dollars

/-- Theorem stating Sheila's work hours on Monday, Wednesday, and Friday --/
theorem sheila_work_hours (s : WorkSchedule) 
  (h1 : s.tt_hours = 12)  -- 6 hours each on Tuesday and Thursday
  (h2 : s.hourly_rate = 14)  -- $14 per hour
  (h3 : s.weekly_earnings = 504)  -- $504 per week
  : s.mwf_hours = 24 := by
  sorry


end NUMINAMATH_CALUDE_sheila_work_hours_l210_21093


namespace NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l210_21028

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 3) :
  2*x + y ≥ 8/3 ∧ (2*x + y = 8/3 ↔ x = 2/3 ∧ y = 4/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l210_21028


namespace NUMINAMATH_CALUDE_stratified_sampling_school_l210_21076

/-- Proves that in a stratified sampling of a school, given the total number of students,
    the number of second-year students, and the number of second-year students selected,
    we can determine the total number of students selected. -/
theorem stratified_sampling_school (total : ℕ) (second_year : ℕ) (selected_second_year : ℕ) 
    (h1 : total = 1800) 
    (h2 : second_year = 600) 
    (h3 : selected_second_year = 21) :
    ∃ n : ℕ, n * second_year = selected_second_year * total ∧ n = 63 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_school_l210_21076


namespace NUMINAMATH_CALUDE_difference_is_one_over_1650_l210_21023

/-- The repeating decimal 0.060606... -/
def repeating_decimal : ℚ := 2 / 33

/-- The terminating decimal 0.06 -/
def terminating_decimal : ℚ := 6 / 100

/-- The difference between the repeating decimal and the terminating decimal -/
def difference : ℚ := repeating_decimal - terminating_decimal

theorem difference_is_one_over_1650 : difference = 1 / 1650 := by
  sorry

end NUMINAMATH_CALUDE_difference_is_one_over_1650_l210_21023


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l210_21087

theorem floor_plus_self_unique_solution (r : ℝ) : 
  (⌊r⌋ : ℝ) + r = 18.2 ↔ r = 9.2 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l210_21087


namespace NUMINAMATH_CALUDE_tan_product_equals_two_l210_21098

theorem tan_product_equals_two : 
  (1 + Real.tan (23 * π / 180)) * (1 + Real.tan (22 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_equals_two_l210_21098


namespace NUMINAMATH_CALUDE_angle_difference_equality_l210_21006

/-- Represents a triangle with angles A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

/-- Represents the bisection of angle C into C1 and C2 -/
structure BisectedC (t : Triangle) where
  C1 : ℝ
  C2 : ℝ
  sum_C : C1 + C2 = t.C
  positive : 0 < C1 ∧ 0 < C2

theorem angle_difference_equality (t : Triangle) (bc : BisectedC t) 
    (h_A_B : t.A = t.B - 15) 
    (h_C2_adjacent : True) -- This is just a placeholder for the condition that C2 is adjacent to the side opposite B
    : bc.C1 - bc.C2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_angle_difference_equality_l210_21006


namespace NUMINAMATH_CALUDE_sum_inequality_l210_21065

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  (1 / (b * c + a + 1 / a) + 1 / (a * c + b + 1 / b) + 1 / (a * b + c + 1 / c)) ≤ 27 / 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l210_21065


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l210_21054

theorem sphere_radius_ratio (V_L V_S r_L r_S : ℝ) : 
  V_L = 432 * Real.pi ∧ 
  V_S = 0.08 * V_L ∧ 
  V_L = (4/3) * Real.pi * r_L^3 ∧ 
  V_S = (4/3) * Real.pi * r_S^3 →
  r_S / r_L = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l210_21054


namespace NUMINAMATH_CALUDE_first_purchase_correct_max_profit_correct_l210_21089

/-- Represents the types of dolls -/
inductive DollType
| A
| B

/-- Represents the purchase and selling prices of dolls -/
def price (t : DollType) : ℕ × ℕ :=
  match t with
  | DollType.A => (20, 25)
  | DollType.B => (15, 18)

/-- The total number of dolls purchased -/
def total_dolls : ℕ := 100

/-- The total cost of the first purchase -/
def total_cost : ℕ := 1650

/-- Calculates the number of each type of doll in the first purchase -/
def first_purchase : ℕ × ℕ := sorry

/-- Calculates the profit for a given number of A dolls in the second purchase -/
def profit (x : ℕ) : ℕ := sorry

/-- Finds the maximum profit and corresponding number of dolls for the second purchase -/
def max_profit : ℕ × ℕ × ℕ := sorry

theorem first_purchase_correct :
  first_purchase = (30, 70) := by sorry

theorem max_profit_correct :
  max_profit = (366, 33, 67) := by sorry

end NUMINAMATH_CALUDE_first_purchase_correct_max_profit_correct_l210_21089


namespace NUMINAMATH_CALUDE_find_x_l210_21015

theorem find_x : ∃ x : ℚ, (1/2 * x) = (1/3 * x + 110) ∧ x = 660 := by sorry

end NUMINAMATH_CALUDE_find_x_l210_21015


namespace NUMINAMATH_CALUDE_billy_laundry_loads_l210_21072

/-- Represents the time taken for each chore in minutes -/
structure ChoreTime where
  sweeping : ℕ  -- time to sweep one room
  dishwashing : ℕ  -- time to wash one dish
  laundry : ℕ  -- time to do one load of laundry

/-- Represents the chores done by each child -/
structure Chores where
  rooms_swept : ℕ
  dishes_washed : ℕ
  laundry_loads : ℕ

def total_time (ct : ChoreTime) (c : Chores) : ℕ :=
  ct.sweeping * c.rooms_swept + ct.dishwashing * c.dishes_washed + ct.laundry * c.laundry_loads

theorem billy_laundry_loads (ct : ChoreTime) (anna billy : Chores) :
  ct.sweeping = 3 →
  ct.dishwashing = 2 →
  ct.laundry = 9 →
  anna.rooms_swept = 10 →
  billy.dishes_washed = 6 →
  anna.dishes_washed = 0 →
  anna.laundry_loads = 0 →
  billy.rooms_swept = 0 →
  total_time ct anna = total_time ct billy →
  billy.laundry_loads = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_laundry_loads_l210_21072


namespace NUMINAMATH_CALUDE_specific_cube_unpainted_count_l210_21088

/-- Represents a cube with painted strips on its faces -/
structure PaintedCube where
  size : Nat
  totalUnitCubes : Nat
  verticalStripWidth : Nat
  horizontalStripHeight : Nat

/-- Calculates the number of unpainted unit cubes in the painted cube -/
def unpaintedUnitCubes (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem stating that a 6x6x6 cube with specific painted strips has 160 unpainted unit cubes -/
theorem specific_cube_unpainted_count :
  let cube : PaintedCube := {
    size := 6,
    totalUnitCubes := 216,
    verticalStripWidth := 2,
    horizontalStripHeight := 2
  }
  unpaintedUnitCubes cube = 160 := by
  sorry

end NUMINAMATH_CALUDE_specific_cube_unpainted_count_l210_21088


namespace NUMINAMATH_CALUDE_binary_sum_equality_l210_21063

/-- Prove that the binary sum 1111₂ + 110₂ - 1001₂ + 1110₂ equals 11100₂ --/
theorem binary_sum_equality : 
  (0b1111 : Nat) + 0b110 - 0b1001 + 0b1110 = 0b11100 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equality_l210_21063


namespace NUMINAMATH_CALUDE_odd_decreasing_properties_l210_21017

/-- An odd and decreasing function on ℝ -/
def odd_decreasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x ≤ y → f y ≤ f x)

theorem odd_decreasing_properties
  (f : ℝ → ℝ) (hf : odd_decreasing_function f) (m n : ℝ) (h : m + n ≥ 0) :
  (f m * f (-m) ≤ 0) ∧ (f m + f n ≤ f (-m) + f (-n)) :=
by sorry

end NUMINAMATH_CALUDE_odd_decreasing_properties_l210_21017


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l210_21094

theorem geometric_sequence_common_ratio : 
  let a : ℕ → ℝ := fun n => (4 : ℝ) ^ (2 * n + 1)
  ∀ n : ℕ, a (n + 1) / a n = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l210_21094


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_sum_of_b_values_l210_21099

theorem quadratic_one_solution_sum (b : ℝ) : 
  (∃! x, 3 * x^2 + b * x + 6 * x + 1 = 0) ↔ 
  (b = -6 + 2 * Real.sqrt 3 ∨ b = -6 - 2 * Real.sqrt 3) :=
by sorry

theorem sum_of_b_values : 
  (-6 + 2 * Real.sqrt 3) + (-6 - 2 * Real.sqrt 3) = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_sum_of_b_values_l210_21099


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l210_21061

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (1 - 3*x + x^2)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                               a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l210_21061


namespace NUMINAMATH_CALUDE_area_triangle_DEF_is_seven_l210_21071

/-- The area of triangle DEF in the given configuration --/
def area_triangle_DEF (side_length_PQRS : ℝ) (side_length_small_square : ℝ) : ℝ :=
  sorry

/-- The theorem stating the area of triangle DEF is 7 cm² --/
theorem area_triangle_DEF_is_seven
  (h1 : area_triangle_DEF 6 2 = 7) :
  ∃ (side_length_PQRS side_length_small_square : ℝ),
    side_length_PQRS^2 = 36 ∧
    side_length_small_square = 2 ∧
    area_triangle_DEF side_length_PQRS side_length_small_square = 7 :=
  sorry

end NUMINAMATH_CALUDE_area_triangle_DEF_is_seven_l210_21071


namespace NUMINAMATH_CALUDE_positive_difference_of_roots_l210_21059

theorem positive_difference_of_roots (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 2*x^2 - 10*x + 18 - (2*x + 34)
  let roots := {x : ℝ | f x = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₁ ≠ r₂ ∧ |r₁ - r₂| = 2 * Real.sqrt 17 :=
sorry

end NUMINAMATH_CALUDE_positive_difference_of_roots_l210_21059


namespace NUMINAMATH_CALUDE_sale_increase_percentage_l210_21090

theorem sale_increase_percentage
  (original_fee : ℝ)
  (fee_reduction_percentage : ℝ)
  (visitor_increase_percentage : ℝ)
  (h1 : original_fee = 1)
  (h2 : fee_reduction_percentage = 25)
  (h3 : visitor_increase_percentage = 60) :
  let new_fee := original_fee * (1 - fee_reduction_percentage / 100)
  let visitor_multiplier := 1 + visitor_increase_percentage / 100
  let sale_increase_percentage := (new_fee * visitor_multiplier - 1) * 100
  sale_increase_percentage = 20 :=
by sorry

end NUMINAMATH_CALUDE_sale_increase_percentage_l210_21090


namespace NUMINAMATH_CALUDE_principal_amount_proof_l210_21012

/-- Prove that for a given principal amount, interest rate, and time period,
    if the difference between compound and simple interest is 20,
    then the principal amount is 8000. -/
theorem principal_amount_proof (P : ℝ) :
  let r : ℝ := 0.05  -- 5% annual interest rate
  let t : ℝ := 2     -- 2 years time period
  let compound_interest := P * (1 + r) ^ t - P
  let simple_interest := P * r * t
  compound_interest - simple_interest = 20 →
  P = 8000 := by
sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l210_21012


namespace NUMINAMATH_CALUDE_evaluate_expression_l210_21067

theorem evaluate_expression : 49^2 - 25^2 + 10^2 = 1876 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l210_21067


namespace NUMINAMATH_CALUDE_range_of_x0_l210_21029

/-- The circle C: x^2 + y^2 = 3 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 3

/-- The line l: x + 3y - 6 = 0 -/
def Line (x y : ℝ) : Prop := x + 3*y - 6 = 0

/-- The angle between two vectors is 60 degrees -/
def AngleSixtyDegrees (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1*x2 + y1*y2) / (Real.sqrt (x1^2 + y1^2) * Real.sqrt (x2^2 + y2^2)) = 1/2

theorem range_of_x0 (x0 y0 : ℝ) :
  Line x0 y0 →
  (∃ x y, Circle x y ∧ AngleSixtyDegrees x0 y0 x y) →
  0 ≤ x0 ∧ x0 ≤ 6/5 := by sorry

end NUMINAMATH_CALUDE_range_of_x0_l210_21029


namespace NUMINAMATH_CALUDE_interest_rates_calculation_l210_21003

/-- Represents the interest calculation for a loan -/
structure Loan where
  principal : ℕ  -- Principal amount in rupees
  time : ℕ       -- Time in years
  interest : ℕ   -- Total interest received in rupees

/-- Calculates the annual interest rate given a loan -/
def calculate_rate (l : Loan) : ℚ :=
  (l.interest : ℚ) * 100 / (l.principal * l.time)

theorem interest_rates_calculation 
  (loan_b : Loan) 
  (loan_c : Loan) 
  (loan_d : Loan) 
  (loan_e : Loan) 
  (h1 : loan_b.principal = 5000 ∧ loan_b.time = 2)
  (h2 : loan_c.principal = 3000 ∧ loan_c.time = 4)
  (h3 : loan_d.principal = 7000 ∧ loan_d.time = 3 ∧ loan_d.interest = 2940)
  (h4 : loan_e.principal = 4500 ∧ loan_e.time = 5 ∧ loan_e.interest = 3375)
  (h5 : loan_b.interest + loan_c.interest = 1980)
  (h6 : calculate_rate loan_b = calculate_rate loan_c) :
  calculate_rate loan_d = 14 ∧ calculate_rate loan_e = 15 :=
sorry

end NUMINAMATH_CALUDE_interest_rates_calculation_l210_21003


namespace NUMINAMATH_CALUDE_hotel_profit_maximized_l210_21062

/-- Represents a hotel with pricing and occupancy information -/
structure Hotel where
  totalRooms : ℕ
  basePrice : ℕ
  priceIncrement : ℕ
  occupancyDecrease : ℕ
  expensePerRoom : ℕ

/-- Calculates the profit for a given price increase -/
def profit (h : Hotel) (priceIncrease : ℕ) : ℤ :=
  let price := h.basePrice + priceIncrease * h.priceIncrement
  let occupiedRooms := h.totalRooms - priceIncrease * h.occupancyDecrease
  (price - h.expensePerRoom) * occupiedRooms

/-- Theorem stating that the profit is maximized at a specific price -/
theorem hotel_profit_maximized (h : Hotel) :
  h.totalRooms = 50 ∧
  h.basePrice = 180 ∧
  h.priceIncrement = 10 ∧
  h.occupancyDecrease = 1 ∧
  h.expensePerRoom = 20 →
  ∃ (maxPriceIncrease : ℕ),
    (∀ (x : ℕ), profit h x ≤ profit h maxPriceIncrease) ∧
    h.basePrice + maxPriceIncrease * h.priceIncrement = 350 :=
sorry

end NUMINAMATH_CALUDE_hotel_profit_maximized_l210_21062


namespace NUMINAMATH_CALUDE_train_passing_time_l210_21010

/-- Given a train of length 420 meters traveling at 63 km/hr,
    prove that it takes 24 seconds to pass a stationary point. -/
theorem train_passing_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 420 ∧ train_speed_kmh = 63 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l210_21010


namespace NUMINAMATH_CALUDE_shifted_line_through_origin_l210_21042

/-- A line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shift a line horizontally -/
def shift_line (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + l.slope * d }

/-- Check if a line passes through a point -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem shifted_line_through_origin (b : ℝ) :
  let original_line := Line.mk 2 b
  let shifted_line := shift_line original_line 2
  passes_through shifted_line 0 0 → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_shifted_line_through_origin_l210_21042
