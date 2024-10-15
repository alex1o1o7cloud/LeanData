import Mathlib

namespace NUMINAMATH_CALUDE_achieve_target_average_l959_95998

/-- Represents Gage's skating schedule and target average -/
structure SkatingSchedule where
  days_with_80_min : Nat
  days_with_100_min : Nat
  target_average : Nat
  total_days : Nat

/-- Calculates the total skating time for the given schedule -/
def total_skating_time (schedule : SkatingSchedule) (last_day_minutes : Nat) : Nat :=
  schedule.days_with_80_min * 80 + 
  schedule.days_with_100_min * 100 + 
  last_day_minutes

/-- Theorem stating that skating 140 minutes on the 8th day achieves the target average -/
theorem achieve_target_average (schedule : SkatingSchedule) 
    (h1 : schedule.days_with_80_min = 4)
    (h2 : schedule.days_with_100_min = 3)
    (h3 : schedule.target_average = 95)
    (h4 : schedule.total_days = 8) :
  total_skating_time schedule 140 / schedule.total_days = schedule.target_average := by
  sorry

#eval total_skating_time { days_with_80_min := 4, days_with_100_min := 3, target_average := 95, total_days := 8 } 140

end NUMINAMATH_CALUDE_achieve_target_average_l959_95998


namespace NUMINAMATH_CALUDE_total_sessions_for_patients_l959_95976

theorem total_sessions_for_patients : 
  let num_patients : ℕ := 4
  let first_patient_sessions : ℕ := 6
  let second_patient_sessions : ℕ := first_patient_sessions + 5
  let remaining_patients_sessions : ℕ := 8
  
  num_patients = 4 →
  first_patient_sessions + 
  second_patient_sessions + 
  (num_patients - 2) * remaining_patients_sessions = 33 := by
sorry

end NUMINAMATH_CALUDE_total_sessions_for_patients_l959_95976


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l959_95940

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) :
  (∀ x : ℤ, (2*x + 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₂ + a₄ + a₆ = 364 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l959_95940


namespace NUMINAMATH_CALUDE_cos_alpha_plus_seven_pi_twelfths_l959_95981

theorem cos_alpha_plus_seven_pi_twelfths (α : ℝ) 
  (h : Real.sin (α + π / 12) = 1 / 3) : 
  Real.cos (α + 7 * π / 12) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_seven_pi_twelfths_l959_95981


namespace NUMINAMATH_CALUDE_interior_angles_sum_l959_95908

theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 1800) → (180 * ((n + 4) - 2) = 2520) := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l959_95908


namespace NUMINAMATH_CALUDE_stock_yield_proof_l959_95921

/-- Proves that the calculated yield matches the quoted yield for a given stock --/
theorem stock_yield_proof (quoted_yield : ℚ) (stock_price : ℚ) 
  (h1 : quoted_yield = 8 / 100)
  (h2 : stock_price = 225) : 
  let dividend := quoted_yield * stock_price
  ((dividend / stock_price) * 100 : ℚ) = quoted_yield * 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_yield_proof_l959_95921


namespace NUMINAMATH_CALUDE_square_of_complex_number_l959_95967

theorem square_of_complex_number :
  let z : ℂ := 5 - 2 * Complex.I
  z * z = 21 - 20 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_square_of_complex_number_l959_95967


namespace NUMINAMATH_CALUDE_only_two_valid_plans_l959_95970

/-- Represents a deployment plan for trucks -/
structure DeploymentPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a deployment plan is valid according to the given conditions -/
def isValidPlan (p : DeploymentPlan) : Prop :=
  p.typeA + p.typeB = 70 ∧
  p.typeB ≤ 3 * p.typeA ∧
  25 * p.typeA + 15 * p.typeB ≤ 1245

/-- The set of all valid deployment plans -/
def validPlans : Set DeploymentPlan :=
  {p | isValidPlan p}

/-- The theorem stating that there are only two valid deployment plans -/
theorem only_two_valid_plans :
  validPlans = {DeploymentPlan.mk 18 52, DeploymentPlan.mk 19 51} :=
by
  sorry

end NUMINAMATH_CALUDE_only_two_valid_plans_l959_95970


namespace NUMINAMATH_CALUDE_platform_length_l959_95961

/-- Given a train of length 300 m that crosses a platform in 39 seconds
    and a signal pole in 36 seconds, the length of the platform is 25 m. -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 39)
  (h3 : time_pole = 36) :
  let speed := train_length / time_pole
  let platform_length := speed * time_platform - train_length
  platform_length = 25 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l959_95961


namespace NUMINAMATH_CALUDE_decorative_window_area_ratio_l959_95933

-- Define the window structure
structure DecorativeWindow where
  ab : ℝ  -- Width of the rectangle (diameter of semicircles)
  ad : ℝ  -- Length of the rectangle
  h_ab_positive : ab > 0
  h_ratio : ad / ab = 4 / 3

-- Define the theorem
theorem decorative_window_area_ratio (w : DecorativeWindow) (h_ab : w.ab = 36) :
  (w.ad * w.ab) / (π * (w.ab / 2)^2) = 16 / (3 * π) := by
  sorry

end NUMINAMATH_CALUDE_decorative_window_area_ratio_l959_95933


namespace NUMINAMATH_CALUDE_power_of_product_l959_95960

theorem power_of_product (a : ℝ) : (3 * a) ^ 3 = 27 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l959_95960


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l959_95936

/-- Given a rhombus with side length 51 and shorter diagonal 48, prove that its longer diagonal is 90 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 51 → shorter_diagonal = 48 → longer_diagonal = 90 → 
  side^2 = (shorter_diagonal / 2)^2 + (longer_diagonal / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l959_95936


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l959_95955

theorem rectangle_area_perimeter_relation (x : ℝ) : 
  let length : ℝ := 4 * x
  let width : ℝ := x + 8
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (area = 2 * perimeter) → (x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l959_95955


namespace NUMINAMATH_CALUDE_min_value_of_expression_l959_95926

theorem min_value_of_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let A := (a^2 + b^2)^4 / (c*d)^4 + (b^2 + c^2)^4 / (a*d)^4 + (c^2 + d^2)^4 / (a*b)^4 + (d^2 + a^2)^4 / (b*c)^4
  A ≥ 64 ∧ (A = 64 ↔ a = b ∧ b = c ∧ c = d) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l959_95926


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_for_x_in_1_to_3_l959_95984

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |2*x - 1|

-- Statement for part 1
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x + 3 ≥ 0} = {x : ℝ | -4 ≤ x ∧ x ≤ 2} := by sorry

-- Statement for part 2
theorem range_of_a_for_x_in_1_to_3 :
  (∀ x ∈ Set.Icc 1 3, f a x ≤ 3) → a ∈ Set.Icc (-3) 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_for_x_in_1_to_3_l959_95984


namespace NUMINAMATH_CALUDE_child_ticket_cost_l959_95931

theorem child_ticket_cost
  (adult_price : ℕ)
  (total_tickets : ℕ)
  (total_receipts : ℕ)
  (adult_tickets : ℕ)
  (h1 : adult_price = 12)
  (h2 : total_tickets = 130)
  (h3 : total_receipts = 840)
  (h4 : adult_tickets = 40)
  : ∃ (child_price : ℕ),
    child_price = 4 ∧
    adult_price * adult_tickets + child_price * (total_tickets - adult_tickets) = total_receipts :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l959_95931


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_inequality_system_solution_l959_95991

-- Part 1: Quadratic equation
theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ x = 1 ∨ x = 5 := by sorry

-- Part 2: System of inequalities
theorem inequality_system_solution :
  ∀ x : ℝ, (x + 3 > 0 ∧ 2*(x + 1) < 4) ↔ (-3 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_inequality_system_solution_l959_95991


namespace NUMINAMATH_CALUDE_point_outside_circle_iff_a_in_range_l959_95902

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 - x + y + a = 0

-- Define what it means for a point to be outside the circle
def point_outside_circle (x y a : ℝ) : Prop := x^2 + y^2 - x + y + a > 0

-- Theorem statement
theorem point_outside_circle_iff_a_in_range :
  ∀ a : ℝ, point_outside_circle 2 1 a ↔ -4 < a ∧ a < 1/2 := by sorry

end NUMINAMATH_CALUDE_point_outside_circle_iff_a_in_range_l959_95902


namespace NUMINAMATH_CALUDE_students_in_class_l959_95910

def total_pencils : ℕ := 125
def pencils_per_student : ℕ := 5

theorem students_in_class : total_pencils / pencils_per_student = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_in_class_l959_95910


namespace NUMINAMATH_CALUDE_eggs_per_basket_l959_95966

theorem eggs_per_basket : ∀ (n : ℕ),
  (30 % n = 0) →  -- Yellow eggs are evenly distributed
  (42 % n = 0) →  -- Blue eggs are evenly distributed
  (n ≥ 4) →       -- At least 4 eggs per basket
  (30 / n ≥ 3) →  -- At least 3 purple baskets
  (42 / n ≥ 3) →  -- At least 3 orange baskets
  n = 6 :=
by
  sorry

#check eggs_per_basket

end NUMINAMATH_CALUDE_eggs_per_basket_l959_95966


namespace NUMINAMATH_CALUDE_decrement_value_theorem_l959_95912

theorem decrement_value_theorem (n : ℕ) (original_mean new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : new_mean = 185) :
  let decrement := (n * original_mean - n * new_mean) / n
  decrement = 15 := by
sorry

end NUMINAMATH_CALUDE_decrement_value_theorem_l959_95912


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l959_95944

theorem quadratic_roots_relation (b c : ℚ) : 
  (∃ r s : ℚ, (4 * r^2 - 6 * r - 8 = 0) ∧ 
               (4 * s^2 - 6 * s - 8 = 0) ∧ 
               ((r + 3)^2 + b * (r + 3) + c = 0) ∧ 
               ((s + 3)^2 + b * (s + 3) + c = 0)) →
  c = 23 / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l959_95944


namespace NUMINAMATH_CALUDE_spherical_coordinate_transformation_l959_95968

/-- Given a point with rectangular coordinates (-3, -4, 5) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, -θ, φ) has rectangular coordinates (-3, 4, 5) -/
theorem spherical_coordinate_transformation (ρ θ φ : Real) 
  (h1 : -3 = ρ * Real.sin φ * Real.cos θ)
  (h2 : -4 = ρ * Real.sin φ * Real.sin θ)
  (h3 : 5 = ρ * Real.cos φ) :
  (-3 = ρ * Real.sin φ * Real.cos (-θ)) ∧ 
  (4 = ρ * Real.sin φ * Real.sin (-θ)) ∧ 
  (5 = ρ * Real.cos φ) :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_transformation_l959_95968


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l959_95993

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 6 + a 10 = 20)
  (h_a4 : a 4 = 2) :
  a 12 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l959_95993


namespace NUMINAMATH_CALUDE_simplify_expression_l959_95901

theorem simplify_expression (s : ℝ) : 105 * s - 63 * s = 42 * s := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l959_95901


namespace NUMINAMATH_CALUDE_power_product_squared_l959_95969

theorem power_product_squared (m n : ℝ) : (m * n)^2 = m^2 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l959_95969


namespace NUMINAMATH_CALUDE_factor_a_squared_minus_16_l959_95948

theorem factor_a_squared_minus_16 (a : ℝ) : a^2 - 16 = (a - 4) * (a + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_a_squared_minus_16_l959_95948


namespace NUMINAMATH_CALUDE_ratio_equality_product_l959_95937

theorem ratio_equality_product (x : ℝ) :
  (2 * x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) →
  ∃ y : ℝ, (2 * y + 3) / (3 * y + 3) = (5 * y + 4) / (8 * y + 4) ∧ y ≠ x ∧ x * y = 0 :=
by sorry

end NUMINAMATH_CALUDE_ratio_equality_product_l959_95937


namespace NUMINAMATH_CALUDE_f_equals_g_l959_95974

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (t : ℝ) : ℝ := t^2 - 2*t

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l959_95974


namespace NUMINAMATH_CALUDE_train_speed_l959_95973

/-- The speed of a train given the time it takes to pass a pole and cross a stationary train -/
theorem train_speed
  (pole_pass_time : ℝ)
  (stationary_train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : pole_pass_time = 5)
  (h2 : stationary_train_length = 360)
  (h3 : crossing_time = 25) :
  let train_speed := stationary_train_length / (crossing_time - pole_pass_time)
  train_speed = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l959_95973


namespace NUMINAMATH_CALUDE_specific_profit_calculation_l959_95995

/-- Given an item cost, markup percentage, discount percentage, and number of items sold,
    calculates the total profit. -/
def totalProfit (a : ℝ) (markup discount : ℝ) (m : ℝ) : ℝ :=
  let sellingPrice := a * (1 + markup)
  let discountedPrice := sellingPrice * (1 - discount)
  m * (discountedPrice - a)

/-- Theorem stating that under specific conditions, the total profit is 0.08am -/
theorem specific_profit_calculation (a m : ℝ) :
  totalProfit a 0.2 0.1 m = 0.08 * a * m :=
by sorry

end NUMINAMATH_CALUDE_specific_profit_calculation_l959_95995


namespace NUMINAMATH_CALUDE_cheese_cost_proof_l959_95980

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 2

/-- The number of sandwiches Ted makes -/
def num_sandwiches : ℕ := 10

/-- The cost of bread in dollars -/
def bread_cost : ℚ := 4

/-- The cost of one pack of sandwich meat in dollars -/
def meat_cost : ℚ := 5

/-- The number of packs of sandwich meat needed -/
def num_meat_packs : ℕ := 2

/-- The number of packs of sliced cheese needed -/
def num_cheese_packs : ℕ := 2

/-- The discount on one pack of cheese in dollars -/
def cheese_discount : ℚ := 1

/-- The discount on one pack of meat in dollars -/
def meat_discount : ℚ := 1

/-- The cost of one pack of sliced cheese without the coupon -/
def cheese_cost : ℚ := 4.5

theorem cheese_cost_proof :
  cheese_cost * num_cheese_packs + bread_cost + meat_cost * num_meat_packs - 
  cheese_discount - meat_discount = sandwich_cost * num_sandwiches := by
  sorry

end NUMINAMATH_CALUDE_cheese_cost_proof_l959_95980


namespace NUMINAMATH_CALUDE_final_donut_count_l959_95950

def donutsRemaining (initial : ℕ) : ℕ :=
  let afterBill := initial - 2
  let afterSecretary := afterBill - 4
  let afterManager := afterSecretary - (afterSecretary / 10)
  let afterFirstGroup := afterManager - (afterManager / 3)
  afterFirstGroup - (afterFirstGroup / 2)

theorem final_donut_count :
  donutsRemaining 50 = 14 :=
by sorry

end NUMINAMATH_CALUDE_final_donut_count_l959_95950


namespace NUMINAMATH_CALUDE_smallest_blue_chips_l959_95988

theorem smallest_blue_chips (total : ℕ) (h_total : total = 49) :
  ∃ (blue red prime : ℕ),
    blue + red = total ∧
    red = blue + prime ∧
    Nat.Prime prime ∧
    ∀ (b r p : ℕ), b + r = total → r = b + p → Nat.Prime p → blue ≤ b :=
by sorry

end NUMINAMATH_CALUDE_smallest_blue_chips_l959_95988


namespace NUMINAMATH_CALUDE_doll_problem_l959_95958

theorem doll_problem (S : ℕ+) (D : ℕ) 
  (h1 : 4 * S + 3 = D) 
  (h2 : 5 * S = D + 6) : 
  D = 39 := by
sorry

end NUMINAMATH_CALUDE_doll_problem_l959_95958


namespace NUMINAMATH_CALUDE_circle_tangent_intersection_ratio_l959_95999

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are externally tangent at a point -/
def ExternallyTangent (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Two circles intersect at a point -/
def Intersect (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Distance between two points -/
def Distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem circle_tangent_intersection_ratio
  (Γ₁ Γ₂ Γ₃ Γ₄ : Circle)
  (P A B C D : ℝ × ℝ)
  (h1 : Γ₁ ≠ Γ₂ ∧ Γ₁ ≠ Γ₃ ∧ Γ₁ ≠ Γ₄ ∧ Γ₂ ≠ Γ₃ ∧ Γ₂ ≠ Γ₄ ∧ Γ₃ ≠ Γ₄)
  (h2 : ExternallyTangent Γ₁ Γ₃ P)
  (h3 : ExternallyTangent Γ₂ Γ₄ P)
  (h4 : Intersect Γ₁ Γ₂ A)
  (h5 : Intersect Γ₂ Γ₃ B)
  (h6 : Intersect Γ₃ Γ₄ C)
  (h7 : Intersect Γ₄ Γ₁ D)
  (h8 : A ≠ P ∧ B ≠ P ∧ C ≠ P ∧ D ≠ P) :
  (Distance A B * Distance B C) / (Distance A D * Distance D C) = 
  (Distance P B)^2 / (Distance P D)^2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_intersection_ratio_l959_95999


namespace NUMINAMATH_CALUDE_f_is_linear_l959_95923

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation -x - 3 = 4 -/
def f (x : ℝ) : ℝ := -x - 3

/-- Theorem stating that f is a linear equation -/
theorem f_is_linear : is_linear_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_linear_l959_95923


namespace NUMINAMATH_CALUDE_best_discount_l959_95900

def original_price : ℝ := 100

def discount_a (price : ℝ) : ℝ := price * 0.8

def discount_b (price : ℝ) : ℝ := price * 0.9 * 0.9

def discount_c (price : ℝ) : ℝ := price * 0.85 * 0.95

def discount_d (price : ℝ) : ℝ := price * 0.95 * 0.85

theorem best_discount :
  discount_a original_price < discount_b original_price ∧
  discount_a original_price < discount_c original_price ∧
  discount_a original_price < discount_d original_price :=
sorry

end NUMINAMATH_CALUDE_best_discount_l959_95900


namespace NUMINAMATH_CALUDE_hotel_air_conditioning_l959_95929

theorem hotel_air_conditioning (total_rooms : ℚ) : 
  total_rooms > 0 →
  (3 / 4 : ℚ) * total_rooms + (1 / 4 : ℚ) * total_rooms = total_rooms →
  (3 / 5 : ℚ) * total_rooms = total_rooms * (3 / 5 : ℚ) →
  (2 / 3 : ℚ) * ((3 / 5 : ℚ) * total_rooms) = (2 / 5 : ℚ) * total_rooms →
  let rented_rooms := (3 / 4 : ℚ) * total_rooms
  let non_rented_rooms := total_rooms - rented_rooms
  let ac_rooms := (3 / 5 : ℚ) * total_rooms
  let rented_ac_rooms := (2 / 5 : ℚ) * total_rooms
  let non_rented_ac_rooms := ac_rooms - rented_ac_rooms
  (non_rented_ac_rooms / non_rented_rooms) * 100 = 80 := by
sorry


end NUMINAMATH_CALUDE_hotel_air_conditioning_l959_95929


namespace NUMINAMATH_CALUDE_modulus_of_z_l959_95949

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) : Prop := (1 + Real.sqrt 3 * i) * z = 4

-- State the theorem
theorem modulus_of_z (z : ℂ) (h : given_equation z) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l959_95949


namespace NUMINAMATH_CALUDE_modular_equivalence_123456_l959_95956

theorem modular_equivalence_123456 :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_equivalence_123456_l959_95956


namespace NUMINAMATH_CALUDE_max_students_before_new_year_l959_95975

/-- The maximum number of students before New Year given the conditions -/
theorem max_students_before_new_year
  (N : ℕ) -- Total number of students before New Year
  (M : ℕ) -- Number of boys before New Year
  (k : ℕ) -- Percentage of boys before New Year
  (ℓ : ℕ) -- Percentage of boys after New Year
  (h1 : M = k * N / 100) -- Condition relating M, k, and N
  (h2 : ℓ < 100) -- ℓ is less than 100
  (h3 : 100 * (M + 1) = ℓ * (N + 3)) -- Condition after New Year
  : N ≤ 197 := by
  sorry

end NUMINAMATH_CALUDE_max_students_before_new_year_l959_95975


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_negative_two_l959_95909

theorem sum_of_powers_equals_negative_two :
  -1^2010 + (-1)^2011 + 1^2012 - 1^2013 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_negative_two_l959_95909


namespace NUMINAMATH_CALUDE_f_seven_equals_163_l959_95903

theorem f_seven_equals_163 (f : ℝ → ℝ) 
  (h1 : f 1 = 1)
  (h2 : ∀ x y, f (x + y) = f x + f y + 8 * x * y - 2) : 
  f 7 = 163 := by
sorry

end NUMINAMATH_CALUDE_f_seven_equals_163_l959_95903


namespace NUMINAMATH_CALUDE_sum_of_xyz_equals_sqrt_13_l959_95962

theorem sum_of_xyz_equals_sqrt_13 (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : x^2 + y^2 + x*y = 3)
  (h_eq2 : y^2 + z^2 + y*z = 4)
  (h_eq3 : z^2 + x^2 + z*x = 7) :
  x + y + z = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_equals_sqrt_13_l959_95962


namespace NUMINAMATH_CALUDE_line_equation_l959_95924

/-- A line passing through (1,1) and intersecting the circle (x-2)^2 + (y-3)^2 = 9 at two points A and B -/
def Line : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (k : ℝ), p.2 = k * (p.1 - 1) + 1}

/-- The circle (x-2)^2 + (y-3)^2 = 9 -/
def Circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 9}

/-- The line passes through (1,1) -/
axiom line_passes_through : (1, 1) ∈ Line

/-- The line intersects the circle at two points A and B -/
axiom line_intersects_circle : ∃ (A B : ℝ × ℝ), A ∈ Line ∩ Circle ∧ B ∈ Line ∩ Circle ∧ A ≠ B

/-- The distance between A and B is 4 -/
axiom distance_AB : ∀ (A B : ℝ × ℝ), A ∈ Line ∩ Circle → B ∈ Line ∩ Circle → A ≠ B →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16

/-- The equation of the line is x + 2y - 3 = 0 -/
theorem line_equation : Line = {p : ℝ × ℝ | p.1 + 2 * p.2 - 3 = 0} :=
sorry

end NUMINAMATH_CALUDE_line_equation_l959_95924


namespace NUMINAMATH_CALUDE_james_distance_traveled_l959_95906

/-- Calculates the distance traveled given speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: James' distance traveled -/
theorem james_distance_traveled :
  distance_traveled 80.0 16.0 = 1280.0 := by
  sorry

end NUMINAMATH_CALUDE_james_distance_traveled_l959_95906


namespace NUMINAMATH_CALUDE_inductive_reasoning_correct_l959_95992

-- Define the types of reasoning
inductive ReasoningMethod
| Analogical
| Deductive
| Inductive
| Reasonable

-- Define the direction of reasoning
inductive ReasoningDirection
| IndividualToIndividual
| GeneralToSpecific
| IndividualToGeneral
| Other

-- Define a function that describes the direction of each reasoning method
def reasoningDirection (method : ReasoningMethod) : ReasoningDirection :=
  match method with
  | ReasoningMethod.Analogical => ReasoningDirection.IndividualToIndividual
  | ReasoningMethod.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningMethod.Inductive => ReasoningDirection.IndividualToGeneral
  | ReasoningMethod.Reasonable => ReasoningDirection.Other

-- Define a predicate for whether a reasoning method can be used in a proof
def canBeUsedInProof (method : ReasoningMethod) : Prop :=
  match method with
  | ReasoningMethod.Reasonable => False
  | _ => True

-- Theorem stating that inductive reasoning is the only correct answer
theorem inductive_reasoning_correct :
  (∀ m : ReasoningMethod, m ≠ ReasoningMethod.Inductive →
    (reasoningDirection m ≠ ReasoningDirection.IndividualToGeneral ∨
     ¬canBeUsedInProof m)) ∧
  (reasoningDirection ReasoningMethod.Inductive = ReasoningDirection.IndividualToGeneral ∧
   canBeUsedInProof ReasoningMethod.Inductive) :=
by
  sorry


end NUMINAMATH_CALUDE_inductive_reasoning_correct_l959_95992


namespace NUMINAMATH_CALUDE_largest_whole_number_less_than_100_over_7_l959_95939

theorem largest_whole_number_less_than_100_over_7 : 
  ∀ x : ℕ, x ≤ 14 ↔ 7 * x < 100 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_less_than_100_over_7_l959_95939


namespace NUMINAMATH_CALUDE_division_of_fractions_l959_95989

theorem division_of_fractions : (3 + 1/2) / 7 / (5/3) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l959_95989


namespace NUMINAMATH_CALUDE_range_implies_m_value_subset_implies_m_range_l959_95979

-- Define the function f(x)
def f (x m : ℝ) : ℝ := |x - m| - |x - 2|

-- Define the solution set M
def M (m : ℝ) : Set ℝ := {x | f x m ≥ |x - 4|}

-- Theorem for part (1)
theorem range_implies_m_value (m : ℝ) :
  (∀ y ∈ Set.Icc (-4) 4, ∃ x, f x m = y) →
  (∀ x, f x m ∈ Set.Icc (-4) 4) →
  m = -2 ∨ m = 6 := by sorry

-- Theorem for part (2)
theorem subset_implies_m_range (m : ℝ) :
  Set.Icc 2 4 ⊆ M m →
  m ∈ Set.Iic 0 ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_range_implies_m_value_subset_implies_m_range_l959_95979


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l959_95983

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallelLine m n) 
  (h3 : parallelLinePlane m α) : 
  parallelLinePlane n α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l959_95983


namespace NUMINAMATH_CALUDE_bill_calculation_l959_95920

theorem bill_calculation (a b c : ℤ) 
  (h1 : a - (b - c) = 13)
  (h2 : (b - c) - a = -9)
  (h3 : a - b - c = 1) : 
  b - c = 1 := by
sorry

end NUMINAMATH_CALUDE_bill_calculation_l959_95920


namespace NUMINAMATH_CALUDE_sphere_inscribed_in_all_shapes_l959_95932

/-- Represents a sphere with a given diameter -/
structure Sphere where
  diameter : ℝ

/-- Represents a square-based prism with a given base edge -/
structure SquarePrism where
  baseEdge : ℝ

/-- Represents a triangular prism with an isosceles triangle base -/
structure TriangularPrism where
  base : ℝ
  height : ℝ

/-- Represents a cylinder with a given base circle diameter -/
structure Cylinder where
  baseDiameter : ℝ

/-- Predicate to check if a sphere can be inscribed in a square prism -/
def inscribedInSquarePrism (s : Sphere) (p : SquarePrism) : Prop :=
  s.diameter = p.baseEdge

/-- Predicate to check if a sphere can be inscribed in a triangular prism -/
def inscribedInTriangularPrism (s : Sphere) (p : TriangularPrism) : Prop :=
  s.diameter = p.base ∧ s.diameter ≤ p.height * (Real.sqrt 3) / 2

/-- Predicate to check if a sphere can be inscribed in a cylinder -/
def inscribedInCylinder (s : Sphere) (c : Cylinder) : Prop :=
  s.diameter = c.baseDiameter

/-- Theorem stating that a sphere with diameter a can be inscribed in all three shapes -/
theorem sphere_inscribed_in_all_shapes (a : ℝ) :
  let s : Sphere := ⟨a⟩
  let sp : SquarePrism := ⟨a⟩
  let tp : TriangularPrism := ⟨a, a⟩
  let c : Cylinder := ⟨a⟩
  inscribedInSquarePrism s sp ∧
  inscribedInTriangularPrism s tp ∧
  inscribedInCylinder s c :=
by sorry

end NUMINAMATH_CALUDE_sphere_inscribed_in_all_shapes_l959_95932


namespace NUMINAMATH_CALUDE_vector_subtraction_l959_95952

/-- Given two vectors in ℝ³, prove that their difference is equal to a specific vector. -/
theorem vector_subtraction (a b : ℝ × ℝ × ℝ) :
  a = (1, -2, 1) →
  b = (1, 0, 2) →
  a - b = (0, -2, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l959_95952


namespace NUMINAMATH_CALUDE_tourist_distribution_l959_95978

theorem tourist_distribution (n m : ℕ) (hn : n = 8) (hm : m = 3) :
  (m ^ n : ℕ) - m * ((m - 1) ^ n) + (m.choose 2) * (1 ^ n) = 5796 :=
sorry

end NUMINAMATH_CALUDE_tourist_distribution_l959_95978


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l959_95917

theorem concentric_circles_area_ratio : 
  let d₁ : ℝ := 2  -- diameter of smallest circle
  let d₂ : ℝ := 4  -- diameter of middle circle
  let d₃ : ℝ := 6  -- diameter of largest circle
  let r₁ := d₁ / 2  -- radius of smallest circle
  let r₂ := d₂ / 2  -- radius of middle circle
  let r₃ := d₃ / 2  -- radius of largest circle
  let A₁ := π * r₁^2  -- area of smallest circle
  let A₂ := π * r₂^2  -- area of middle circle
  let A₃ := π * r₃^2  -- area of largest circle
  let blue_area := A₂ - A₁  -- area between smallest and middle circles
  let green_area := A₃ - A₂  -- area between middle and largest circles
  (green_area / blue_area : ℝ) = 5/3
  := by sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l959_95917


namespace NUMINAMATH_CALUDE_correct_factorization_l959_95915

theorem correct_factorization (a b : ℤ) :
  (∃ k : ℤ, (X + 6) * (X - 2) = X^2 + k*X + b) ∧
  (∃ m : ℤ, (X - 8) * (X + 4) = X^2 + a*X + m) →
  (X + 2) * (X - 6) = X^2 + a*X + b :=
by sorry

end NUMINAMATH_CALUDE_correct_factorization_l959_95915


namespace NUMINAMATH_CALUDE_wire_length_difference_l959_95905

theorem wire_length_difference (total_length first_part : ℕ) 
  (h1 : total_length = 180)
  (h2 : first_part = 106) :
  first_part - (total_length - first_part) = 32 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_difference_l959_95905


namespace NUMINAMATH_CALUDE_cells_after_3_hours_l959_95951

/-- Represents the number of cells after a given number of divisions -/
def cells (n : ℕ) : ℕ := 2^n

/-- The number of divisions that occur in 3 hours -/
def divisions_in_3_hours : ℕ := 3 * 2

theorem cells_after_3_hours : cells divisions_in_3_hours = 64 := by
  sorry

#eval cells divisions_in_3_hours

end NUMINAMATH_CALUDE_cells_after_3_hours_l959_95951


namespace NUMINAMATH_CALUDE_system_solution_l959_95914

-- Define the system of equations
def system (x y z w : ℝ) : Prop :=
  (x + y = z^2 + w^2 + 6*z*w) ∧
  (x + z = y^2 + w^2 + 6*y*w) ∧
  (x + w = y^2 + z^2 + 6*y*z) ∧
  (y + z = x^2 + w^2 + 6*x*w) ∧
  (y + w = x^2 + z^2 + 6*x*z) ∧
  (z + w = x^2 + y^2 + 6*x*y)

-- Define the set of solutions
def solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(0, 0, 0, 0), (1/4, 1/4, 1/4, 1/4), (-1/4, -1/4, 3/4, -1/4), (-1/2, -1/2, 5/2, -1/2)}

-- Define cyclic permutations
def cyclic_perm (x y z w : ℝ) : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(x, y, z, w), (y, z, w, x), (z, w, x, y), (w, x, y, z)}

-- Define the full solution set including cyclic permutations
def full_solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  ⋃ (s : ℝ × ℝ × ℝ × ℝ) (hs : s ∈ solution_set), cyclic_perm s.1 s.2.1 s.2.2.1 s.2.2.2

-- Theorem statement
theorem system_solution :
  ∀ (x y z w : ℝ), system x y z w ↔ (x, y, z, w) ∈ full_solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l959_95914


namespace NUMINAMATH_CALUDE_weight_of_b_l959_95987

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 30)
  (h2 : (a + b) / 2 = 25)
  (h3 : (b + c) / 2 = 28) : 
  b = 16 := by sorry

end NUMINAMATH_CALUDE_weight_of_b_l959_95987


namespace NUMINAMATH_CALUDE_polynomial_factorization_l959_95947

theorem polynomial_factorization (y : ℝ) :
  1 + 5*y^2 + 25*y^4 + 125*y^6 + 625*y^8 = 
  (5*y^2 + ((5+Real.sqrt 5)*y)/2 + 1) * 
  (5*y^2 + ((5-Real.sqrt 5)*y)/2 + 1) * 
  (5*y^2 - ((5+Real.sqrt 5)*y)/2 + 1) * 
  (5*y^2 - ((5-Real.sqrt 5)*y)/2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l959_95947


namespace NUMINAMATH_CALUDE_incorrect_calculation_l959_95922

theorem incorrect_calculation (a : ℝ) : (2 * a)^3 ≠ 6 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l959_95922


namespace NUMINAMATH_CALUDE_fourth_root_cube_root_equality_l959_95972

theorem fourth_root_cube_root_equality : 
  (0.000008 : ℝ)^((1/3) * (1/4)) = (2 : ℝ)^(1/4) / (10 : ℝ)^(1/2) :=
sorry

end NUMINAMATH_CALUDE_fourth_root_cube_root_equality_l959_95972


namespace NUMINAMATH_CALUDE_helen_washing_time_l959_95965

/-- The time it takes Helen to wash pillowcases each time -/
def washing_time (weeks_between_washes : ℕ) (minutes_per_year : ℕ) (weeks_per_year : ℕ) : ℕ :=
  minutes_per_year / (weeks_per_year / weeks_between_washes)

/-- Theorem stating that Helen's pillowcase washing time is 30 minutes -/
theorem helen_washing_time :
  washing_time 4 390 52 = 30 := by
  sorry

end NUMINAMATH_CALUDE_helen_washing_time_l959_95965


namespace NUMINAMATH_CALUDE_speedster_convertibles_count_l959_95930

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  speedsterConvertibles : ℕ

/-- Conditions of the inventory -/
def inventoryConditions (i : Inventory) : Prop :=
  i.speedsters = (2 * i.total) / 3 ∧
  i.speedsterConvertibles = (4 * i.speedsters) / 5 ∧
  i.total - i.speedsters = 50

/-- Theorem stating that under the given conditions, there are 80 Speedster convertibles -/
theorem speedster_convertibles_count (i : Inventory) :
  inventoryConditions i → i.speedsterConvertibles = 80 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertibles_count_l959_95930


namespace NUMINAMATH_CALUDE_cindy_envelopes_l959_95927

theorem cindy_envelopes (initial : ℕ) (friend1 friend2 friend3 friend4 friend5 : ℕ) :
  initial = 137 →
  friend1 = 4 →
  friend2 = 7 →
  friend3 = 5 →
  friend4 = 10 →
  friend5 = 3 →
  initial - (friend1 + friend2 + friend3 + friend4 + friend5) = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_cindy_envelopes_l959_95927


namespace NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l959_95945

theorem subset_sum_divisible_by_2n
  (n : ℕ)
  (h_n : n ≥ 4)
  (a : Fin n → ℕ)
  (h_distinct : Function.Injective a)
  (h_bounds : ∀ i : Fin n, 0 < a i ∧ a i < 2*n) :
  ∃ (S : Finset (Fin n)), (S.sum (λ i => a i)) % (2*n) = 0 :=
sorry

end NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l959_95945


namespace NUMINAMATH_CALUDE_orthocentre_constructible_l959_95954

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- A circle in the plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- A triangle in the plane -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Definition of a circumcircle of a triangle -/
def isCircumcircle (c : Circle) (t : Triangle) : Prop :=
  sorry

/-- Definition of a circumcentre of a triangle -/
def isCircumcentre (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Definition of an orthocentre of a triangle -/
def isOrthocentre (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Definition of constructible using only a straightedge -/
def isStraightedgeConstructible (p : Point) (given : Set Point) : Prop :=
  sorry

/-- The main theorem -/
theorem orthocentre_constructible (t : Triangle) (c : Circle) (o : Point) :
  isCircumcircle c t → isCircumcentre o t →
  ∃ h : Point, isOrthocentre h t ∧ 
    isStraightedgeConstructible h {t.A, t.B, t.C, o} :=
  sorry

end NUMINAMATH_CALUDE_orthocentre_constructible_l959_95954


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l959_95985

/-- Given that when a person weighing 87 kg replaces a person weighing 67 kg,
    the average weight increases by 2.5 kg, prove that the number of persons
    initially is 8. -/
theorem initial_number_of_persons : ℕ :=
  let old_weight := 67
  let new_weight := 87
  let average_increase := 2.5
  let n := (new_weight - old_weight) / average_increase
  8

#check initial_number_of_persons

end NUMINAMATH_CALUDE_initial_number_of_persons_l959_95985


namespace NUMINAMATH_CALUDE_marble_weight_sum_l959_95997

theorem marble_weight_sum : 
  let piece1 : ℝ := 0.33
  let piece2 : ℝ := 0.33
  let piece3 : ℝ := 0.08
  piece1 + piece2 + piece3 = 0.74 := by
sorry

end NUMINAMATH_CALUDE_marble_weight_sum_l959_95997


namespace NUMINAMATH_CALUDE_student_a_score_l959_95918

/-- Calculates the score for a test based on the given grading method -/
def calculateScore (totalQuestions : ℕ) (correctResponses : ℕ) : ℕ :=
  let incorrectResponses := totalQuestions - correctResponses
  correctResponses - 2 * incorrectResponses

theorem student_a_score :
  calculateScore 100 90 = 70 := by
  sorry

end NUMINAMATH_CALUDE_student_a_score_l959_95918


namespace NUMINAMATH_CALUDE_coin_game_expected_value_l959_95935

/-- A modified coin game with three outcomes --/
structure CoinGame where
  prob_heads : ℝ
  prob_tails : ℝ
  prob_edge : ℝ
  payoff_heads : ℝ
  payoff_tails : ℝ
  payoff_edge : ℝ

/-- Calculate the expected value of the coin game --/
def expected_value (game : CoinGame) : ℝ :=
  game.prob_heads * game.payoff_heads +
  game.prob_tails * game.payoff_tails +
  game.prob_edge * game.payoff_edge

/-- Theorem stating the expected value of the specific coin game --/
theorem coin_game_expected_value :
  let game : CoinGame := {
    prob_heads := 1/4,
    prob_tails := 1/2,
    prob_edge := 1/4,
    payoff_heads := 4,
    payoff_tails := -3,
    payoff_edge := 0
  }
  expected_value game = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_coin_game_expected_value_l959_95935


namespace NUMINAMATH_CALUDE_subset_P_l959_95986

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_subset_P_l959_95986


namespace NUMINAMATH_CALUDE_tulip_bouquet_combinations_l959_95941

theorem tulip_bouquet_combinations (n : ℕ) (max_tulips : ℕ) (total_money : ℕ) (tulip_cost : ℕ) : 
  n = 11 → 
  max_tulips = 11 → 
  total_money = 550 → 
  tulip_cost = 49 → 
  (Finset.filter (fun k => k % 2 = 1 ∧ k ≤ max_tulips) (Finset.range (n + 1))).card = 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_tulip_bouquet_combinations_l959_95941


namespace NUMINAMATH_CALUDE_longest_chord_implies_a_equals_one_l959_95977

/-- The line equation ax + y - 2 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y - 2 = 0

/-- The circle equation (x-1)^2 + (y-a)^2 = 4 -/
def circle_equation (a x y : ℝ) : Prop := (x - 1)^2 + (y - a)^2 = 4

/-- A point (x, y) is on the circle -/
def point_on_circle (a x y : ℝ) : Prop := circle_equation a x y

/-- A point (x, y) is on the line -/
def point_on_line (a x y : ℝ) : Prop := line_equation a x y

/-- The theorem to be proved -/
theorem longest_chord_implies_a_equals_one (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    point_on_circle a x₁ y₁ ∧
    point_on_circle a x₂ y₂ ∧
    point_on_line a x₁ y₁ ∧
    point_on_line a x₂ y₂ ∧
    ∀ x y : ℝ, point_on_circle a x y → (x₂ - x₁)^2 + (y₂ - y₁)^2 ≥ (x - x₁)^2 + (y - y₁)^2) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_longest_chord_implies_a_equals_one_l959_95977


namespace NUMINAMATH_CALUDE_value_of_k_l959_95913

theorem value_of_k : ∃ k : ℚ, (k + 2) / 4 - (2 * k - 1) / 6 = 1 ∧ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_k_l959_95913


namespace NUMINAMATH_CALUDE_eco_park_cherry_sample_l959_95934

/-- Represents the number of cherry trees in a stratified sample -/
def cherry_trees_in_sample (total_trees : ℕ) (total_cherry_trees : ℕ) (sample_size : ℕ) : ℕ :=
  (total_cherry_trees * sample_size) / total_trees

/-- Theorem stating the number of cherry trees in the sample for the given eco-park -/
theorem eco_park_cherry_sample :
  cherry_trees_in_sample 60000 4000 300 = 20 := by
  sorry

end NUMINAMATH_CALUDE_eco_park_cherry_sample_l959_95934


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l959_95938

open Set

theorem intersection_of_M_and_N :
  let M : Set ℝ := {x | x > 1}
  let N : Set ℝ := {x | x^2 - 2*x < 0}
  M ∩ N = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l959_95938


namespace NUMINAMATH_CALUDE_second_cart_travel_distance_l959_95907

/-- Distance traveled by the first cart in n seconds -/
def first_cart_distance (n : ℕ) : ℕ := n * (6 + (n - 1) * 4)

/-- Distance traveled by the second cart in n seconds -/
def second_cart_distance (n : ℕ) : ℕ := n * (7 + (n - 1) * 9 / 2)

/-- Time taken by the first cart to reach the bottom -/
def total_time : ℕ := 35

/-- Time difference between the start of the two carts -/
def start_delay : ℕ := 2

theorem second_cart_travel_distance :
  second_cart_distance (total_time - start_delay) = 4983 := by
  sorry

end NUMINAMATH_CALUDE_second_cart_travel_distance_l959_95907


namespace NUMINAMATH_CALUDE_socks_cost_prove_socks_cost_l959_95943

def initial_amount : ℕ := 100
def shirt_cost : ℕ := 24
def final_amount : ℕ := 65

theorem socks_cost : ℕ :=
  initial_amount - shirt_cost - final_amount

theorem prove_socks_cost : socks_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_socks_cost_prove_socks_cost_l959_95943


namespace NUMINAMATH_CALUDE_dilution_proof_l959_95904

/-- Proves that adding 6 ounces of water to 12 ounces of a 60% alcohol solution results in a 40% alcohol solution. -/
theorem dilution_proof (initial_volume : ℝ) (initial_concentration : ℝ) (target_concentration : ℝ) 
  (water_added : ℝ) (h1 : initial_volume = 12) (h2 : initial_concentration = 0.6) 
  (h3 : target_concentration = 0.4) (h4 : water_added = 6) : 
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_dilution_proof_l959_95904


namespace NUMINAMATH_CALUDE_modulus_of_complex_power_l959_95911

theorem modulus_of_complex_power :
  Complex.abs ((2 + 2*Complex.I)^6) = 512 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_power_l959_95911


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l959_95971

theorem arithmetic_calculation : 2^2 + 3 * 4 - 5 + (6 - 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l959_95971


namespace NUMINAMATH_CALUDE_fraction_equality_l959_95928

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 5 / 2) :
  (a - b) / a = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l959_95928


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l959_95959

theorem parallelogram_side_length 
  (s : ℝ) 
  (side1 : ℝ) 
  (side2 : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h1 : side1 = 3 * s) 
  (h2 : side2 = s) 
  (h3 : angle = π / 3) -- 60 degrees in radians
  (h4 : area = 9 * Real.sqrt 3) 
  (h5 : area = side1 * side2 * Real.sin angle) : 
  s = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l959_95959


namespace NUMINAMATH_CALUDE_vehicle_distance_after_3_minutes_l959_95925

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

theorem vehicle_distance_after_3_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time : ℝ := 3 / 60
  distance_between_vehicles truck_speed car_speed time = 1 := by sorry

end NUMINAMATH_CALUDE_vehicle_distance_after_3_minutes_l959_95925


namespace NUMINAMATH_CALUDE_smartphone_charge_time_proof_l959_95982

/-- The time in minutes to fully charge a smartphone -/
def smartphone_charge_time : ℝ := 26

/-- The time in minutes to fully charge a tablet -/
def tablet_charge_time : ℝ := 53

/-- The total time in minutes for Ana to charge her devices -/
def ana_charge_time : ℝ := 66

theorem smartphone_charge_time_proof :
  smartphone_charge_time = 26 :=
by
  have h1 : tablet_charge_time = 53 := rfl
  have h2 : tablet_charge_time + (1/2 * smartphone_charge_time) = ana_charge_time :=
    by sorry
  sorry

end NUMINAMATH_CALUDE_smartphone_charge_time_proof_l959_95982


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l959_95919

def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

theorem extreme_values_of_f :
  ∃ (a b : ℝ), (∀ x : ℝ, f x ≤ f a ∨ f x ≥ f b) ∧
               (∀ c : ℝ, (∀ x : ℝ, f x ≤ f c) → c = a) ∧
               (∀ c : ℝ, (∀ x : ℝ, f x ≥ f c) → c = b) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l959_95919


namespace NUMINAMATH_CALUDE_number_puzzle_l959_95953

theorem number_puzzle :
  ∃ x : ℝ, 3 * (2 * x + 9) = 75 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l959_95953


namespace NUMINAMATH_CALUDE_expression_evaluation_l959_95946

theorem expression_evaluation :
  let x : ℚ := -2
  (3 + x * (3 + x) - 3^2) / (x - 3 + x^2) = 8 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l959_95946


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l959_95996

/-- Given a sequence aₙ where {1 + aₙ} is geometric with ratio 2 and a₁ = 1, prove a₅ = 31 -/
theorem geometric_sequence_fifth_term (a : ℕ → ℝ) 
  (h1 : ∀ n, (1 + a (n + 1)) = 2 * (1 + a n))  -- {1 + aₙ} is geometric with ratio 2
  (h2 : a 1 = 1)  -- a₁ = 1
  : a 5 = 31 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l959_95996


namespace NUMINAMATH_CALUDE_division_problem_l959_95994

theorem division_problem (dividend : ℕ) (remainder : ℕ) (quotient : ℕ) (divisor : ℕ) (n : ℕ) :
  dividend = 251 →
  remainder = 8 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + n →
  dividend = divisor * quotient + remainder →
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l959_95994


namespace NUMINAMATH_CALUDE_factorization_of_9a_minus_6b_l959_95942

theorem factorization_of_9a_minus_6b (a b : ℝ) : 9*a - 6*b = 3*(3*a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_9a_minus_6b_l959_95942


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l959_95957

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

theorem twelfth_term_of_specific_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let a2 := (5 : ℚ) / 6
  let d := a2 - a
  arithmeticSequence a d 12 = (25 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l959_95957


namespace NUMINAMATH_CALUDE_greg_savings_l959_95964

theorem greg_savings (scooter_cost : ℕ) (amount_needed : ℕ) (amount_saved : ℕ) : 
  scooter_cost = 90 → amount_needed = 33 → amount_saved = scooter_cost - amount_needed → amount_saved = 57 := by
sorry

end NUMINAMATH_CALUDE_greg_savings_l959_95964


namespace NUMINAMATH_CALUDE_circle_m_range_l959_95916

-- Define the equation of the circle
def circle_equation (x y m : ℝ) : Prop :=
  x + y^2 - x + y + m = 0

-- State the theorem
theorem circle_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → m < (1/2) :=
by sorry

end NUMINAMATH_CALUDE_circle_m_range_l959_95916


namespace NUMINAMATH_CALUDE_C_not_necessary_nor_sufficient_for_A_l959_95990

-- Define the propositions
variable (A B C : Prop)

-- Define the given conditions
axiom C_sufficient_for_B : C → B
axiom B_necessary_for_A : A → B

-- Theorem to prove
theorem C_not_necessary_nor_sufficient_for_A :
  ¬(∀ (h : A), C) ∧ ¬(∀ (h : C), A) :=
by sorry

end NUMINAMATH_CALUDE_C_not_necessary_nor_sufficient_for_A_l959_95990


namespace NUMINAMATH_CALUDE_range_of_f_l959_95963

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x^4)

theorem range_of_f :
  Set.range f = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l959_95963
