import Mathlib

namespace quadratic_one_solution_l1474_147477

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 36 = 0) ↔ n = 36 ∨ n = -36 :=
by sorry

end quadratic_one_solution_l1474_147477


namespace triangle_trig_identity_l1474_147498

theorem triangle_trig_identity (A B C : ℝ) (hABC : A + B + C = π) 
  (hAC : 2 = Real.sqrt ((B - C)^2 + 4 * (Real.sin (A/2))^2)) 
  (hBC : 3 = Real.sqrt ((A - C)^2 + 4 * (Real.sin (B/2))^2))
  (hcosA : Real.cos A = -4/5) :
  Real.sin (2*B + π/6) = (17 + 12 * Real.sqrt 7) / 25 := by
  sorry

end triangle_trig_identity_l1474_147498


namespace rectangle_area_increase_l1474_147453

theorem rectangle_area_increase : 
  let original_length : ℝ := 40
  let original_width : ℝ := 20
  let length_decrease : ℝ := 5
  let width_increase : ℝ := 5
  let new_length : ℝ := original_length - length_decrease
  let new_width : ℝ := original_width + width_increase
  let original_area : ℝ := original_length * original_width
  let new_area : ℝ := new_length * new_width
  new_area - original_area = 75
  := by sorry

end rectangle_area_increase_l1474_147453


namespace max_angle_at_tangent_points_l1474_147437

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Checks if a point is strictly inside a circle -/
def is_inside_circle (p : Point) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

/-- Checks if a point is on a circle -/
def is_on_circle (p : Point) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) = c.radius

/-- Calculates the angle ABC given three points A, B, and C -/
noncomputable def angle (A B C : Point) : ℝ := sorry

/-- Defines the tangent points of circles passing through two points and tangent to a given circle -/
noncomputable def tangent_points (A B : Point) (Ω : Circle) : Point × Point := sorry

theorem max_angle_at_tangent_points (Ω : Circle) (A B : Point) :
  is_inside_circle A Ω →
  is_inside_circle B Ω →
  A ≠ B →
  let (C₁, C₂) := tangent_points A B Ω
  ∀ C : Point, is_on_circle C Ω →
    angle A C B ≤ max (angle A C₁ B) (angle A C₂ B) :=
by sorry

end max_angle_at_tangent_points_l1474_147437


namespace last_remaining_number_l1474_147475

/-- The function that determines the next position of a number after one round of erasure -/
def nextPosition (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 * (n / 3)
  else if n % 3 = 2 then 2 * (n / 3) + 1
  else 0  -- This case (n % 3 = 1) corresponds to erased numbers

/-- The function that determines the original position given a final position -/
def originalPosition (finalPos : ℕ) : ℕ :=
  if finalPos = 1 then 1458 else 0  -- We only care about the winning position

/-- The theorem stating that 1458 is the last remaining number -/
theorem last_remaining_number :
  ∃ (n : ℕ), n ≤ 2002 ∧ 
  (∀ (m : ℕ), m ≤ 2002 → m ≠ n → 
    ∃ (k : ℕ), originalPosition m = 3 * k + 1 ∨ 
    ∃ (j : ℕ), nextPosition (originalPosition m) = originalPosition j ∧ j < m) ∧
  originalPosition n = n ∧ n = 1458 :=
sorry

end last_remaining_number_l1474_147475


namespace triangle_area_l1474_147499

/-- The area of a right triangle with base 2 and height (12 - p) is equal to 12 - p. -/
theorem triangle_area (p : ℝ) : 
  (1 / 2 : ℝ) * 2 * (12 - p) = 12 - p :=
sorry

end triangle_area_l1474_147499


namespace not_q_is_false_l1474_147455

theorem not_q_is_false (p q : Prop) (hp : ¬p) (hq : q) : ¬(¬q) := by
  sorry

end not_q_is_false_l1474_147455


namespace solve_system_and_find_perimeter_l1474_147424

/-- Given a system of equations, prove the values of a and b, and the perimeter of an isosceles triangle with these side lengths. -/
theorem solve_system_and_find_perimeter :
  ∃ (a b : ℝ),
    (4 * a - 3 * b = 22) ∧
    (2 * a + b = 16) ∧
    (a = 7) ∧
    (b = 2) ∧
    (2 * max a b + min a b = 16) := by
  sorry


end solve_system_and_find_perimeter_l1474_147424


namespace abc_value_l1474_147465

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 45 * Real.rpow 3 (1/3))
  (hac : a * c = 63 * Real.rpow 3 (1/3))
  (hbc : b * c = 28 * Real.rpow 3 (1/3)) :
  a * b * c = 630 := by
sorry

end abc_value_l1474_147465


namespace log_equation_solution_l1474_147488

theorem log_equation_solution (x : ℝ) :
  (Real.log x / Real.log 4) + (Real.log (1/6) / Real.log 4) = 1/2 → x = 12 := by
sorry

end log_equation_solution_l1474_147488


namespace jills_salary_l1474_147476

/-- Represents a person's monthly financial allocation --/
structure MonthlyFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  vacationFund : ℝ
  savings : ℝ
  socializing : ℝ
  charitable : ℝ

/-- Conditions for Jill's financial allocation --/
def JillsFinances (m : MonthlyFinances) : Prop :=
  m.discretionaryIncome = m.netSalary / 5 ∧
  m.vacationFund = 0.3 * m.discretionaryIncome ∧
  m.savings = 0.2 * m.discretionaryIncome ∧
  m.socializing = 0.35 * m.discretionaryIncome ∧
  m.charitable = 99

/-- Theorem stating that under the given conditions, Jill's net monthly salary is $3300 --/
theorem jills_salary (m : MonthlyFinances) (h : JillsFinances m) : m.netSalary = 3300 := by
  sorry

end jills_salary_l1474_147476


namespace x_twelve_equals_one_l1474_147490

theorem x_twelve_equals_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end x_twelve_equals_one_l1474_147490


namespace dot_product_range_l1474_147421

/-- Given vectors a and b in a plane such that their magnitudes and the magnitude of their difference
    are between 2 and 6 (inclusive), prove that their dot product is between -14 and 34 (inclusive). -/
theorem dot_product_range (a b : ℝ × ℝ) 
  (ha : 2 ≤ ‖a‖ ∧ ‖a‖ ≤ 6)
  (hb : 2 ≤ ‖b‖ ∧ ‖b‖ ≤ 6)
  (hab : 2 ≤ ‖a - b‖ ∧ ‖a - b‖ ≤ 6) :
  -14 ≤ a • b ∧ a • b ≤ 34 :=
by sorry

end dot_product_range_l1474_147421


namespace geometric_subsequence_contains_342_l1474_147400

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d)
  (h_d_nonzero : d ≠ 0)

/-- A geometric sequence extracted from an arithmetic sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) :=
  (seq : ℕ → ℝ)
  (h_geom : ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, seq (n + 1) = q * seq n)
  (h_sub : ∃ f : ℕ → ℕ, ∀ n : ℕ, seq n = as.a (f n))
  (h_2_6_22 : ∃ k₁ k₂ k₃ : ℕ, seq k₁ = as.a 2 ∧ seq k₂ = as.a 6 ∧ seq k₃ = as.a 22)

/-- The main theorem -/
theorem geometric_subsequence_contains_342 (as : ArithmeticSequence) 
  (gs : GeometricSubsequence as) : 
  ∃ k : ℕ, gs.seq k = as.a 342 := by
  sorry

end geometric_subsequence_contains_342_l1474_147400


namespace circle_diameter_ratio_l1474_147411

theorem circle_diameter_ratio (D C : ℝ) : 
  D = 20 → -- Diameter of circle D is 20 cm
  C > 0 → -- Diameter of circle C is positive
  C < D → -- Circle C is inside circle D
  (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 4 → -- Ratio of shaded area to area of C is 4:1
  C = 8 * Real.sqrt 5 := by sorry

end circle_diameter_ratio_l1474_147411


namespace sum_of_powers_equals_six_to_power_three_l1474_147407

theorem sum_of_powers_equals_six_to_power_three :
  3^3 + 4^3 + 5^3 = 6^3 := by sorry

end sum_of_powers_equals_six_to_power_three_l1474_147407


namespace binomial_20_5_l1474_147482

theorem binomial_20_5 : Nat.choose 20 5 = 11628 := by sorry

end binomial_20_5_l1474_147482


namespace multiplicative_inverse_modulo_million_l1474_147448

theorem multiplicative_inverse_modulo_million : 
  let A : ℕ := 123456
  let B : ℕ := 153846
  let N : ℕ := 500000
  let M : ℕ := 1000000
  (A * B * N) % M = 1 := by
  sorry

end multiplicative_inverse_modulo_million_l1474_147448


namespace speed_conversion_l1474_147413

theorem speed_conversion (speed_ms : ℝ) (speed_kmh : ℝ) : 
  speed_ms = 9/36 → speed_kmh = 0.9 → (1 : ℝ) * 3.6 = speed_kmh / speed_ms :=
by
  sorry

end speed_conversion_l1474_147413


namespace sequence_general_term_l1474_147442

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = 2) : 
    ∀ n : ℕ, a n = 2 * n - 1 := by
  sorry

end sequence_general_term_l1474_147442


namespace zhang_li_age_ratio_l1474_147416

-- Define the ages
def li_age : ℕ := 12
def jung_age : ℕ := 26

-- Define Zhang's age based on Jung's age
def zhang_age : ℕ := jung_age - 2

-- Define the ratio of Zhang's age to Li's age
def age_ratio : ℚ := zhang_age / li_age

-- Theorem statement
theorem zhang_li_age_ratio :
  age_ratio = 2 := by sorry

end zhang_li_age_ratio_l1474_147416


namespace edric_working_days_l1474_147441

/-- Calculates the number of working days per week given monthly salary, hours per day, and hourly rate -/
def working_days_per_week (monthly_salary : ℕ) (hours_per_day : ℕ) (hourly_rate : ℕ) : ℚ :=
  (monthly_salary : ℚ) / 4 / (hours_per_day * hourly_rate)

/-- Theorem: Given Edric's monthly salary, hours per day, and hourly rate, he works 6 days a week -/
theorem edric_working_days : working_days_per_week 576 8 3 = 6 := by
  sorry

end edric_working_days_l1474_147441


namespace square_area_from_diagonal_l1474_147463

theorem square_area_from_diagonal : 
  ∀ (d s A : ℝ), 
  d = 8 * Real.sqrt 2 →  -- diagonal length
  d = s * Real.sqrt 2 →  -- relationship between diagonal and side
  A = s^2 →             -- area formula
  A = 64 := by           
sorry

end square_area_from_diagonal_l1474_147463


namespace alex_and_sam_speeds_l1474_147427

-- Define the variables
def alex_downstream_distance : ℝ := 36
def alex_downstream_time : ℝ := 6
def alex_upstream_time : ℝ := 9
def sam_downstream_distance : ℝ := 48
def sam_downstream_time : ℝ := 8
def sam_upstream_time : ℝ := 12

-- Define the theorem
theorem alex_and_sam_speeds :
  ∃ (alex_speed sam_speed current_speed : ℝ),
    alex_speed > 0 ∧ sam_speed > 0 ∧
    (alex_speed + current_speed) * alex_downstream_time = alex_downstream_distance ∧
    (alex_speed - current_speed) * alex_upstream_time = alex_downstream_distance ∧
    (sam_speed + current_speed) * sam_downstream_time = sam_downstream_distance ∧
    (sam_speed - current_speed) * sam_upstream_time = sam_downstream_distance ∧
    alex_speed = 5 ∧ sam_speed = 5 :=
by
  sorry

end alex_and_sam_speeds_l1474_147427


namespace circle_line_properties_l1474_147467

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 25

-- Define the line l
def line_l (m x y : ℝ) : Prop := (3*m + 1)*x + (m + 1)*y - 5*m - 3 = 0

-- Theorem statement
theorem circle_line_properties :
  -- Line l intersects circle C
  (∃ (m x y : ℝ), circle_C x y ∧ line_l m x y) ∧
  -- The chord length intercepted by circle C on the y-axis is 4√6
  (∃ (y1 y2 : ℝ), circle_C 0 y1 ∧ circle_C 0 y2 ∧ y2 - y1 = 4 * Real.sqrt 6) ∧
  -- When the chord length intercepted by circle C is the shortest, the equation of line l is x=1
  (∃ (m : ℝ), ∀ (x y : ℝ), line_l m x y → x = 1) :=
sorry

end circle_line_properties_l1474_147467


namespace parabola_coefficient_l1474_147433

/-- Given a parabola y = ax^2 + bx + c with vertex (q, 2q) and y-intercept (0, -2q), where q ≠ 0, 
    the value of b is 8/q. -/
theorem parabola_coefficient (a b c q : ℝ) (hq : q ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (2 * q = a * q^2 + b * q + c) →
  (-2 * q = c) →
  b = 8 / q := by
sorry

end parabola_coefficient_l1474_147433


namespace intersection_implies_a_value_l1474_147478

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 3*a-1, a^2+1}
  A ∩ B = {-3} → a = -2/3 := by
  sorry

end intersection_implies_a_value_l1474_147478


namespace third_draw_probability_l1474_147452

/-- Represents the number of balls of each color in the box -/
structure BallCount where
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing a white ball -/
def probWhite (balls : BallCount) : ℚ :=
  balls.white / (balls.white + balls.black)

theorem third_draw_probability :
  let initial := BallCount.mk 8 7
  let after_removal := BallCount.mk (initial.white - 1) (initial.black - 1)
  probWhite after_removal = 7 / 13 := by
  sorry

end third_draw_probability_l1474_147452


namespace kayla_apple_count_l1474_147415

/-- Given that Kylie and Kayla picked a total of 340 apples, and Kayla picked 10 more than 4 times 
    the amount of apples that Kylie picked, prove that Kayla picked 274 apples. -/
theorem kayla_apple_count :
  ∀ (kylie_apples : ℕ),
  kylie_apples + (10 + 4 * kylie_apples) = 340 →
  10 + 4 * kylie_apples = 274 :=
by
  sorry

end kayla_apple_count_l1474_147415


namespace range_of_a_for_line_separating_points_l1474_147495

/-- Given points A and B on opposite sides of the line 3x + 2y + a = 0, 
    prove that the range of a is (-19, -9) -/
theorem range_of_a_for_line_separating_points 
  (A B : ℝ × ℝ) 
  (h_A : A = (1, 3)) 
  (h_B : B = (5, 2)) 
  (h_opposite : (3 * A.1 + 2 * A.2 + a) * (3 * B.1 + 2 * B.2 + a) < 0) :
  ∀ a : ℝ, (a > -19 ∧ a < -9) ↔ 
    ∃ (x y : ℝ), (3 * x + 2 * y + a = 0 ∧ 
      (3 * A.1 + 2 * A.2 + a) * (3 * B.1 + 2 * B.2 + a) < 0) :=
by sorry

end range_of_a_for_line_separating_points_l1474_147495


namespace goldfish_preference_total_l1474_147423

/-- Calculates the total number of students preferring goldfish across three classes -/
theorem goldfish_preference_total (class_size : ℕ) 
  (johnson_fraction : ℚ) (feldstein_fraction : ℚ) (henderson_fraction : ℚ)
  (h1 : class_size = 30)
  (h2 : johnson_fraction = 1 / 6)
  (h3 : feldstein_fraction = 2 / 3)
  (h4 : henderson_fraction = 1 / 5) :
  ⌊class_size * johnson_fraction⌋ + ⌊class_size * feldstein_fraction⌋ + ⌊class_size * henderson_fraction⌋ = 31 :=
by sorry

end goldfish_preference_total_l1474_147423


namespace arithmetic_geometric_condition_l1474_147428

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem arithmetic_geometric_condition (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d ∧ a 1 = 2 →
  (d = 4 → geometric_sequence (a 1) (a 2) (a 5)) ∧
  ¬(geometric_sequence (a 1) (a 2) (a 5) → d = 4) :=
by sorry

end arithmetic_geometric_condition_l1474_147428


namespace sum_of_decimals_l1474_147408

theorem sum_of_decimals : 0.305 + 0.089 + 0.007 = 0.401 := by sorry

end sum_of_decimals_l1474_147408


namespace arithmetic_calculation_l1474_147470

theorem arithmetic_calculation : 
  (1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) : ℝ) = 1200 := by
  sorry

end arithmetic_calculation_l1474_147470


namespace wind_velocity_problem_l1474_147447

/-- Represents the relationship between pressure, area, and wind velocity -/
def pressure_relation (k : ℝ) (A V : ℝ) : ℝ := k * A * V^3

theorem wind_velocity_problem (k : ℝ) :
  let A₁ : ℝ := 1
  let V₁ : ℝ := 10
  let P₁ : ℝ := 1
  let A₂ : ℝ := 1
  let P₂ : ℝ := 64
  pressure_relation k A₁ V₁ = P₁ →
  pressure_relation k A₂ 40 = P₂ :=
by
  sorry

#check wind_velocity_problem

end wind_velocity_problem_l1474_147447


namespace cubic_roots_cube_l1474_147456

theorem cubic_roots_cube (u v w : ℂ) :
  (u^3 + v^3 + w^3 = 54) →
  (u^3 * v^3 + v^3 * w^3 + w^3 * u^3 = -89) →
  (u^3 * v^3 * w^3 = 27) →
  (u + v + w = 5) →
  (u * v + v * w + w * u = 4) →
  (u * v * w = 3) →
  (u^3 - 5 * u^2 + 4 * u - 3 = 0) →
  (v^3 - 5 * v^2 + 4 * v - 3 = 0) →
  (w^3 - 5 * w^2 + 4 * w - 3 = 0) →
  ∀ (x : ℂ), x^3 - 54 * x^2 - 89 * x - 27 = 0 ↔ (x = u^3 ∨ x = v^3 ∨ x = w^3) := by
sorry

end cubic_roots_cube_l1474_147456


namespace other_root_of_complex_quadratic_l1474_147402

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -75 + 100*I ∧ z = 5 + 10*I → (-z)^2 = -75 + 100*I :=
by
  sorry

end other_root_of_complex_quadratic_l1474_147402


namespace special_circle_standard_equation_l1474_147412

/-- A circle passing through two points with its center on a given line -/
structure SpecialCircle where
  -- Center coordinates
  h : ℝ
  k : ℝ
  -- Radius
  r : ℝ
  -- The circle passes through (0,4)
  passes_through_A : h^2 + (k - 4)^2 = r^2
  -- The circle passes through (4,6)
  passes_through_B : (h - 4)^2 + (k - 6)^2 = r^2
  -- The center lies on the line x-2y-2=0
  center_on_line : h - 2*k - 2 = 0

/-- The standard equation of the special circle -/
def special_circle_equation (c : SpecialCircle) : Prop :=
  ∀ (x y : ℝ), (x - 4)^2 + (y - 1)^2 = 25 ↔ (x - c.h)^2 + (y - c.k)^2 = c.r^2

/-- The main theorem: proving the standard equation of the special circle -/
theorem special_circle_standard_equation :
  ∃ (c : SpecialCircle), special_circle_equation c :=
sorry

end special_circle_standard_equation_l1474_147412


namespace min_integer_solution_2x_minus_1_geq_5_l1474_147493

theorem min_integer_solution_2x_minus_1_geq_5 :
  ∀ x : ℤ, (2 * x - 1 ≥ 5) → x ≥ 3 ∧ ∀ y : ℤ, (2 * y - 1 ≥ 5) → y ≥ x :=
by sorry

end min_integer_solution_2x_minus_1_geq_5_l1474_147493


namespace number_percentage_problem_l1474_147403

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 35 → 0.40 * N = 420 := by
  sorry

end number_percentage_problem_l1474_147403


namespace ralphs_cards_l1474_147480

/-- Given Ralph's initial and additional cards, prove the total number of cards. -/
theorem ralphs_cards (initial_cards additional_cards : ℕ) 
  (h1 : initial_cards = 4)
  (h2 : additional_cards = 8) :
  initial_cards + additional_cards = 12 := by
  sorry

end ralphs_cards_l1474_147480


namespace dannys_bottle_caps_l1474_147450

theorem dannys_bottle_caps (park_caps : ℕ) (park_wrappers : ℕ) (collection_wrappers : ℕ) :
  park_caps = 58 →
  park_wrappers = 25 →
  collection_wrappers = 11 →
  ∃ (collection_caps : ℕ), collection_caps = collection_wrappers + 1 ∧ collection_caps = 12 :=
by sorry

end dannys_bottle_caps_l1474_147450


namespace roxy_garden_problem_l1474_147483

def garden_problem (initial_flowering : ℕ) (initial_fruiting_factor : ℕ) 
  (bought_flowering : ℕ) (bought_fruiting : ℕ) 
  (given_fruiting : ℕ) (total_remaining : ℕ) : Prop :=
  let initial_fruiting := initial_flowering * initial_fruiting_factor
  let after_purchase_flowering := initial_flowering + bought_flowering
  let after_purchase_fruiting := initial_fruiting + bought_fruiting
  let remaining_fruiting := after_purchase_fruiting - given_fruiting
  let remaining_flowering := total_remaining - remaining_fruiting
  let given_flowering := after_purchase_flowering - remaining_flowering
  given_flowering = 1

theorem roxy_garden_problem : 
  garden_problem 7 2 3 2 4 21 :=
sorry

end roxy_garden_problem_l1474_147483


namespace bill_denomination_l1474_147492

theorem bill_denomination (total_amount : ℕ) (num_bills : ℕ) (h1 : total_amount = 45) (h2 : num_bills = 9) :
  total_amount / num_bills = 5 := by
sorry

end bill_denomination_l1474_147492


namespace circle_through_points_on_line_equation_l1474_147473

/-- A circle passing through two points with its center on a given line -/
def CircleThroughPointsOnLine (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (a, b) := C
  (x₁ - a)^2 + (y₁ - b)^2 = (x₂ - a)^2 + (y₂ - b)^2 ∧ a + b = 2

/-- The standard equation of a circle -/
def StandardCircleEquation (C : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  let (a, b) := C
  (x - a)^2 + (y - b)^2 = r^2

theorem circle_through_points_on_line_equation :
  ∀ (C : ℝ × ℝ),
  CircleThroughPointsOnLine (1, -1) (-1, 1) C →
  ∃ (x y : ℝ), StandardCircleEquation C 2 x y :=
by sorry

end circle_through_points_on_line_equation_l1474_147473


namespace smallest_four_digit_pascal_l1474_147420

/-- Pascal's triangle function -/
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

/-- Predicate for four-digit numbers -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- Theorem: 1001 is the smallest four-digit number in Pascal's triangle -/
theorem smallest_four_digit_pascal :
  (∃ n k, pascal n k = 1001) ∧
  (∀ n k, pascal n k < 1001 → ¬is_four_digit (pascal n k)) :=
sorry

end smallest_four_digit_pascal_l1474_147420


namespace flag_covering_l1474_147430

theorem flag_covering (grid_height : Nat) (grid_width : Nat) (flag_count : Nat) :
  grid_height = 9 →
  grid_width = 18 →
  flag_count = 18 →
  (∃ (ways_to_place_flag : Nat), ways_to_place_flag = 2) →
  (∃ (total_ways : Nat), total_ways = 2^flag_count) :=
by sorry

end flag_covering_l1474_147430


namespace digit_rearrangement_divisibility_l1474_147459

def is_digit_rearrangement (n m : ℕ) : Prop :=
  ∃ (digits_n digits_m : List ℕ), 
    digits_n.length > 0 ∧
    digits_m.length > 0 ∧
    digits_n.sum = digits_m.sum ∧
    n = digits_n.foldl (λ acc d => acc * 10 + d) 0 ∧
    m = digits_m.foldl (λ acc d => acc * 10 + d) 0

def satisfies_property (d : ℕ) : Prop :=
  d > 0 ∧ ∀ n m : ℕ, n > 0 → is_digit_rearrangement n m → (d ∣ n → d ∣ m)

theorem digit_rearrangement_divisibility :
  {d : ℕ | satisfies_property d} = {1, 3, 9} := by sorry

end digit_rearrangement_divisibility_l1474_147459


namespace leahs_coins_value_l1474_147439

theorem leahs_coins_value :
  ∀ (p d : ℕ),
  p + d = 15 →
  p = d + 1 →
  p * 1 + d * 10 = 87 :=
by sorry

end leahs_coins_value_l1474_147439


namespace tyrones_money_equals_thirteen_dollars_l1474_147464

/-- The value of Tyrone's money in cents -/
def tyrones_money : ℕ :=
  2 * 100 +  -- Two $1 bills
  5 * 100 +  -- One $5 bill
  13 * 25 +  -- 13 quarters
  20 * 10 +  -- 20 dimes
  8 * 5 +    -- 8 nickels
  35 * 1     -- 35 pennies

/-- Theorem stating that Tyrone's money equals $13 -/
theorem tyrones_money_equals_thirteen_dollars :
  tyrones_money = 13 * 100 := by
  sorry

end tyrones_money_equals_thirteen_dollars_l1474_147464


namespace can_determine_coin_type_l1474_147462

/-- Represents the outcome of weighing two groups of coins -/
inductive WeighingResult
  | Even
  | Odd

/-- Represents the type of a coin -/
inductive CoinType
  | Genuine
  | Counterfeit

/-- Function to weigh two groups of coins -/
def weigh (group1 : Finset Nat) (group2 : Finset Nat) : WeighingResult :=
  sorry

theorem can_determine_coin_type 
  (total_coins : Nat)
  (counterfeit_coins : Nat)
  (weight_difference : Nat)
  (h1 : total_coins = 101)
  (h2 : counterfeit_coins = 50)
  (h3 : weight_difference = 1)
  : ∃ (f : Finset Nat → Finset Nat → WeighingResult → CoinType), 
    ∀ (selected_coin : Nat) (group1 group2 : Finset Nat),
    selected_coin ∉ group1 ∧ selected_coin ∉ group2 →
    group1.card = 50 ∧ group2.card = 50 →
    f group1 group2 (weigh group1 group2) = 
      if selected_coin ≤ counterfeit_coins then CoinType.Counterfeit else CoinType.Genuine :=
sorry

end can_determine_coin_type_l1474_147462


namespace distinct_painting_methods_is_catalan_l1474_147474

/-- Represents a ball with a number and color -/
structure Ball where
  number : Nat
  color : Nat

/-- Represents a painting method for n balls -/
def PaintingMethod (n : Nat) := Fin n → Ball

/-- Checks if two painting methods are distinct -/
def is_distinct (n : Nat) (m1 m2 : PaintingMethod n) : Prop :=
  ∃ i : Fin n, (m1 i).color ≠ (m2 i).color

/-- The number of distinct painting methods for n balls -/
def distinct_painting_methods (n : Nat) : Nat :=
  (Nat.choose (2 * n - 2) (n - 1)) / n

/-- Theorem: The number of distinct painting methods is the (n-1)th Catalan number -/
theorem distinct_painting_methods_is_catalan (n : Nat) :
  distinct_painting_methods n = (Nat.choose (2 * n - 2) (n - 1)) / n :=
by sorry

end distinct_painting_methods_is_catalan_l1474_147474


namespace range_of_m_for_necessary_condition_l1474_147479

theorem range_of_m_for_necessary_condition : 
  ∀ m : ℝ, 
  (∀ x : ℝ, x^2 - 2*x - 3 > 0 → (x < m - 1 ∨ x > m + 1)) ∧ 
  (∃ x : ℝ, (x < m - 1 ∨ x > m + 1) ∧ x^2 - 2*x - 3 ≤ 0) ↔ 
  m ∈ Set.Icc 0 2 :=
by sorry

end range_of_m_for_necessary_condition_l1474_147479


namespace complex_sum_equality_l1474_147497

theorem complex_sum_equality : 
  12 * Complex.exp (3 * Real.pi * Complex.I / 13) + 
  12 * Complex.exp (7 * Real.pi * Complex.I / 26) = 
  24 * Real.cos (Real.pi / 26) * Complex.exp (19 * Real.pi * Complex.I / 52) := by
  sorry

end complex_sum_equality_l1474_147497


namespace range_of_m_l1474_147406

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 8 * x / y) > m^2 + 2*m) → 
  -4 < m ∧ m < 2 :=
by sorry

end range_of_m_l1474_147406


namespace simplest_form_iff_odd_l1474_147418

theorem simplest_form_iff_odd (n : ℤ) : 
  (∀ d : ℤ, d ∣ (3*n + 10) ∧ d ∣ (5*n + 16) → d = 1 ∨ d = -1) ↔ Odd n :=
sorry

end simplest_form_iff_odd_l1474_147418


namespace train_crossing_time_l1474_147443

/-- The time it takes for a train to cross a pole -/
theorem train_crossing_time (train_speed : ℝ) (train_length : ℝ) : 
  train_speed = 270 →
  train_length = 375.03 →
  (train_length / (train_speed * 1000 / 3600)) = 5.0004 := by
  sorry

end train_crossing_time_l1474_147443


namespace octagon_diagonal_intersection_probability_l1474_147429

/-- A regular octagon is an 8-sided polygon with all sides equal and all angles equal. -/
def RegularOctagon : Type := Unit

/-- The number of diagonals in a regular octagon. -/
def num_diagonals (octagon : RegularOctagon) : ℕ := 20

/-- The number of pairs of diagonals in a regular octagon. -/
def num_diagonal_pairs (octagon : RegularOctagon) : ℕ := 190

/-- The number of pairs of intersecting diagonals in a regular octagon. -/
def num_intersecting_pairs (octagon : RegularOctagon) : ℕ := 70

/-- The probability that two randomly chosen diagonals in a regular octagon intersect inside the octagon. -/
theorem octagon_diagonal_intersection_probability (octagon : RegularOctagon) :
  (num_intersecting_pairs octagon : ℚ) / (num_diagonal_pairs octagon) = 7 / 19 := by
  sorry

end octagon_diagonal_intersection_probability_l1474_147429


namespace first_group_men_count_l1474_147432

/-- Represents the number of men in the first group -/
def first_group_men : ℕ := 30

/-- Represents the number of days worked by the first group -/
def first_group_days : ℕ := 12

/-- Represents the number of hours worked per day by the first group -/
def first_group_hours_per_day : ℕ := 8

/-- Represents the length of road (in km) asphalted by the first group -/
def first_group_road_length : ℕ := 1

/-- Represents the number of men in the second group -/
def second_group_men : ℕ := 20

/-- Represents the number of days worked by the second group -/
def second_group_days : ℝ := 19.2

/-- Represents the number of hours worked per day by the second group -/
def second_group_hours_per_day : ℕ := 15

/-- Represents the length of road (in km) asphalted by the second group -/
def second_group_road_length : ℕ := 2

/-- Theorem stating that the number of men in the first group is 30 -/
theorem first_group_men_count : 
  first_group_men * first_group_days * first_group_hours_per_day * second_group_road_length = 
  second_group_men * second_group_days * second_group_hours_per_day * first_group_road_length :=
by sorry

end first_group_men_count_l1474_147432


namespace fraction_equality_l1474_147451

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x + 2*y) / (x - 5*y) = -3) : 
  (x + 5*y) / (5*x - y) = 53/57 := by
sorry

end fraction_equality_l1474_147451


namespace leftover_coins_value_l1474_147435

def quarters_per_roll : ℕ := 50
def dimes_per_roll : ℕ := 40
def michael_quarters : ℕ := 95
def michael_dimes : ℕ := 172
def anna_quarters : ℕ := 140
def anna_dimes : ℕ := 287
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftover_coins_value :
  let total_quarters := michael_quarters + anna_quarters
  let total_dimes := michael_dimes + anna_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  let leftover_value := (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value
  leftover_value = 10.65 := by sorry

end leftover_coins_value_l1474_147435


namespace sin_alpha_minus_pi_sixth_l1474_147438

theorem sin_alpha_minus_pi_sixth (α : Real) 
  (h : Real.cos (α - π/3) - Real.cos α = 1/3) : 
  Real.sin (α - π/6) = 1/3 := by
sorry

end sin_alpha_minus_pi_sixth_l1474_147438


namespace parabola_vertex_first_quadrant_l1474_147414

/-- A parabola with equation y = (x-m)^2 + (m-1) has its vertex in the first quadrant if and only if m > 1 -/
theorem parabola_vertex_first_quadrant (m : ℝ) : 
  (∃ x y : ℝ, y = (x - m)^2 + (m - 1) ∧ x = m ∧ y = m - 1 ∧ x > 0 ∧ y > 0) ↔ m > 1 := by
  sorry

end parabola_vertex_first_quadrant_l1474_147414


namespace gecko_insect_consumption_l1474_147405

theorem gecko_insect_consumption (geckos lizards total_insects : ℕ) 
  (h1 : geckos = 5)
  (h2 : lizards = 3)
  (h3 : total_insects = 66) :
  ∃ (gecko_consumption : ℕ), 
    gecko_consumption * geckos + (2 * gecko_consumption) * lizards = total_insects ∧ 
    gecko_consumption = 6 := by
  sorry

end gecko_insect_consumption_l1474_147405


namespace sum_of_repeating_decimals_l1474_147404

-- Define repeating decimals
def repeating_decimal_2 : ℚ := 2/9
def repeating_decimal_03 : ℚ := 1/33

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_2 + repeating_decimal_03 = 25/99 := by
  sorry

end sum_of_repeating_decimals_l1474_147404


namespace fibSum_eq_five_nineteenths_l1474_147481

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of the Fibonacci series divided by powers of 5 -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / 5^n

/-- Theorem stating that the sum of the Fibonacci series divided by powers of 5 equals 5/19 -/
theorem fibSum_eq_five_nineteenths : fibSum = 5 / 19 := by sorry

end fibSum_eq_five_nineteenths_l1474_147481


namespace fruit_salad_weight_l1474_147466

/-- The total weight of fruit in Scarlett's fruit salad is 1.85 pounds. -/
theorem fruit_salad_weight :
  let melon : ℚ := 35/100
  let berries : ℚ := 48/100
  let grapes : ℚ := 29/100
  let pineapple : ℚ := 56/100
  let oranges : ℚ := 17/100
  melon + berries + grapes + pineapple + oranges = 185/100 := by
  sorry

end fruit_salad_weight_l1474_147466


namespace van_speed_for_longer_time_l1474_147468

/-- Given a van that travels 450 km in 5 hours, this theorem proves the speed
    required to cover the same distance in 3/2 of the original time. -/
theorem van_speed_for_longer_time (distance : ℝ) (initial_time : ℝ) (time_factor : ℝ) :
  distance = 450 ∧ initial_time = 5 ∧ time_factor = 3/2 →
  distance / (initial_time * time_factor) = 60 := by
  sorry

end van_speed_for_longer_time_l1474_147468


namespace pet_owners_problem_l1474_147485

theorem pet_owners_problem (total_pet_owners : ℕ) (only_dogs : ℕ) (only_cats : ℕ) (cats_dogs_snakes : ℕ) (total_snakes : ℕ)
  (h1 : total_pet_owners = 79)
  (h2 : only_dogs = 15)
  (h3 : only_cats = 10)
  (h4 : cats_dogs_snakes = 3)
  (h5 : total_snakes = 49) :
  total_pet_owners - only_dogs - only_cats - cats_dogs_snakes - (total_snakes - cats_dogs_snakes) = 5 :=
by sorry

end pet_owners_problem_l1474_147485


namespace cubic_extrema_l1474_147401

-- Define a cubic function
def cubic_function (a b c d : ℝ) : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d

-- Define the derivative of the cubic function
def cubic_derivative (a b c : ℝ) : ℝ → ℝ := λ x => 3 * a * x^2 + 2 * b * x + c

-- State the theorem
theorem cubic_extrema (a b c d : ℝ) :
  let f := cubic_function a b c d
  let f' := cubic_derivative (3*a) (2*b) c
  (∀ x, x * f' x = 0 ↔ x = 0 ∨ x = 2 ∨ x = -2) →
  (∀ x, f x ≤ f (-2)) ∧ (∀ x, f 2 ≤ f x) :=
sorry

end cubic_extrema_l1474_147401


namespace power_division_l1474_147419

theorem power_division (a b c d : ℕ) (h : b = a^2) :
  a^(2*c+1) / b^c = a :=
sorry

end power_division_l1474_147419


namespace school_play_seating_l1474_147454

/-- Given a school play seating arrangement, prove the number of unoccupied seats. -/
theorem school_play_seating (rows : ℕ) (chairs_per_row : ℕ) (occupied_seats : ℕ) 
  (h1 : rows = 40)
  (h2 : chairs_per_row = 20)
  (h3 : occupied_seats = 790) :
  rows * chairs_per_row - occupied_seats = 10 := by
  sorry

end school_play_seating_l1474_147454


namespace unique_perfect_square_solution_l1474_147486

theorem unique_perfect_square_solution (n : ℤ) : 
  (∃ k : ℤ, n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = k^2) ↔ n = 10 := by
sorry

end unique_perfect_square_solution_l1474_147486


namespace inequality_proof_l1474_147444

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : -1 < b ∧ b < 0) : a * b^2 > b^2 := by
  sorry

end inequality_proof_l1474_147444


namespace cubic_roots_sum_of_squares_l1474_147426

theorem cubic_roots_sum_of_squares (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 12*x^2 + 47*x - 30 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  a^2 + b^2 + c^2 = 50 := by
sorry

end cubic_roots_sum_of_squares_l1474_147426


namespace regular_polygon_perimeter_l1474_147410

/-- A regular polygon with exterior angle 90 degrees and side length 7 units has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  n * exterior_angle = 360 →
  n * side_length = 28 := by
  sorry

end regular_polygon_perimeter_l1474_147410


namespace chord_division_ratio_l1474_147484

/-- Given a circle with radius 11 and a chord of length 18 that intersects
    a diameter at a point 7 units from the center, prove that this point
    divides the chord in a ratio of either 2:1 or 1:2. -/
theorem chord_division_ratio (R : ℝ) (chord_length : ℝ) (center_to_intersection : ℝ)
    (h1 : R = 11)
    (h2 : chord_length = 18)
    (h3 : center_to_intersection = 7) :
    ∃ (x y : ℝ), (x + y = chord_length ∧ 
                 ((x / y = 2 ∧ y / x = 1/2) ∨ 
                  (x / y = 1/2 ∧ y / x = 2))) :=
by sorry

end chord_division_ratio_l1474_147484


namespace zeros_of_f_l1474_147472

-- Define the function f(x) = x^3 - 3x + 2
def f (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Theorem stating that 1 and -2 are the zeros of f
theorem zeros_of_f : 
  (∃ x : ℝ, f x = 0) ∧ (∀ x : ℝ, f x = 0 → x = 1 ∨ x = -2) ∧ f 1 = 0 ∧ f (-2) = 0 := by
  sorry

end zeros_of_f_l1474_147472


namespace ceiling_equation_solution_l1474_147471

theorem ceiling_equation_solution :
  ∃! b : ℝ, b + ⌈b⌉ = 14.7 ∧ b = 7.2 := by sorry

end ceiling_equation_solution_l1474_147471


namespace min_value_sum_of_squares_l1474_147489

theorem min_value_sum_of_squares (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end min_value_sum_of_squares_l1474_147489


namespace sara_frosting_cans_l1474_147469

/-- The number of cans of frosting needed to frost the remaining cakes after Sara's baking and Carol's eating -/
def frosting_cans_needed (cakes_per_day : ℕ) (days : ℕ) (cakes_eaten : ℕ) (frosting_cans_per_cake : ℕ) : ℕ :=
  ((cakes_per_day * days - cakes_eaten) * frosting_cans_per_cake)

/-- Theorem stating the number of frosting cans needed in Sara's specific scenario -/
theorem sara_frosting_cans : frosting_cans_needed 10 5 12 2 = 76 := by
  sorry

end sara_frosting_cans_l1474_147469


namespace pokemon_cards_total_l1474_147434

theorem pokemon_cards_total (jason_left : ℕ) (jason_gave : ℕ) (lisa_left : ℕ) (lisa_gave : ℕ) :
  jason_left = 4 → jason_gave = 9 → lisa_left = 7 → lisa_gave = 15 →
  (jason_left + jason_gave) + (lisa_left + lisa_gave) = 35 :=
by sorry

end pokemon_cards_total_l1474_147434


namespace total_children_l1474_147457

theorem total_children (happy : ℕ) (sad : ℕ) (neutral : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neutral_boys : ℕ) :
  happy = 30 →
  sad = 10 →
  neutral = 20 →
  boys = 19 →
  girls = 41 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 7 →
  boys + girls = 60 :=
by sorry

end total_children_l1474_147457


namespace third_year_sample_size_l1474_147446

/-- Calculates the number of students to be sampled from a specific grade in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (grade_population : ℕ) (total_sample_size : ℕ) : ℕ :=
  (grade_population * total_sample_size) / total_population

/-- Proves that the number of third-year students to be sampled is 21 -/
theorem third_year_sample_size :
  let total_population : ℕ := 3000
  let third_year_population : ℕ := 1050
  let total_sample_size : ℕ := 60
  stratified_sample_size total_population third_year_population total_sample_size = 21 := by
  sorry


end third_year_sample_size_l1474_147446


namespace average_speed_two_segment_trip_l1474_147496

/-- Given a trip with two segments, prove that the average speed is as calculated -/
theorem average_speed_two_segment_trip (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 360)
  (h2 : speed1 = 60)
  (h3 : distance2 = 120)
  (h4 : speed2 = 40) :
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 480 / 9 := by
sorry

end average_speed_two_segment_trip_l1474_147496


namespace x_negative_y_positive_l1474_147431

theorem x_negative_y_positive (x y : ℝ) 
  (h1 : 2 * x - y > 3 * x) 
  (h2 : x + 2 * y < 2 * y) : 
  x < 0 ∧ y > 0 := by
  sorry

end x_negative_y_positive_l1474_147431


namespace parallel_vectors_m_value_l1474_147422

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, m + 1)

theorem parallel_vectors_m_value :
  (∃ (k : ℝ), k ≠ 0 ∧ b m = k • a) → m = 1 := by
  sorry

end parallel_vectors_m_value_l1474_147422


namespace min_product_of_prime_sum_l1474_147449

theorem min_product_of_prime_sum (m n p : ℕ) : 
  Prime m → Prime n → Prime p → m ≠ n → n ≠ p → m ≠ p → m + n = p → 
  (∀ m' n' p' : ℕ, Prime m' → Prime n' → Prime p' → m' ≠ n' → n' ≠ p' → m' ≠ p' → m' + n' = p' → m' * n' * p' ≥ m * n * p) →
  m * n * p = 30 := by
sorry

end min_product_of_prime_sum_l1474_147449


namespace exactly_two_b_values_l1474_147461

-- Define the quadratic function
def f (b : ℤ) (x : ℤ) : ℤ := x^2 + b*x + 3

-- Define a predicate for when f(b,x) ≤ 0
def satisfies_inequality (b : ℤ) (x : ℤ) : Prop := f b x ≤ 0

-- Define a predicate for when b gives exactly three integer solutions
def has_three_solutions (b : ℤ) : Prop :=
  ∃ x₁ x₂ x₃ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    satisfies_inequality b x₁ ∧
    satisfies_inequality b x₂ ∧
    satisfies_inequality b x₃ ∧
    ∀ x : ℤ, satisfies_inequality b x → (x = x₁ ∨ x = x₂ ∨ x = x₃)

-- The main theorem
theorem exactly_two_b_values :
  ∃! s : Finset ℤ, s.card = 2 ∧ ∀ b : ℤ, b ∈ s ↔ has_three_solutions b :=
sorry

end exactly_two_b_values_l1474_147461


namespace angle_between_vectors_l1474_147445

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = -1)  -- dot product condition
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 2)  -- magnitude of a
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 1)  -- magnitude of b
  : Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2*π/3 := by
  sorry

end angle_between_vectors_l1474_147445


namespace middle_part_of_proportional_distribution_l1474_147494

theorem middle_part_of_proportional_distribution (total : ℚ) (r1 r2 r3 : ℚ) :
  total = 120 →
  r1 = 1 →
  r2 = 1/4 →
  r3 = 1/8 →
  (r2 * total) / (r1 + r2 + r3) = 240/11 := by
  sorry

end middle_part_of_proportional_distribution_l1474_147494


namespace odd_function_symmetry_l1474_147425

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_symmetry (f : ℝ → ℝ) (a : ℝ) (h : IsOdd f) :
  f (-a) = -f a := by sorry

end odd_function_symmetry_l1474_147425


namespace jog_time_proportional_l1474_147458

/-- Given a constant jogging pace, prove that if 3 miles takes 30 minutes, then 1.5 miles takes 15 minutes. -/
theorem jog_time_proportional (pace : ℝ → ℝ) (h_constant : ∀ x y, pace x = pace y) :
  pace 3 = 30 → pace 1.5 = 15 := by
  sorry

end jog_time_proportional_l1474_147458


namespace sqrt_one_minus_x_domain_l1474_147487

theorem sqrt_one_minus_x_domain : ∀ x : ℝ, 
  (x ≤ 1 ↔ ∃ y : ℝ, y ^ 2 = 1 - x) ∧
  (x = 2 → ¬∃ y : ℝ, y ^ 2 = 1 - x) :=
sorry

end sqrt_one_minus_x_domain_l1474_147487


namespace sum_of_solutions_eq_four_l1474_147409

theorem sum_of_solutions_eq_four :
  let f : ℝ → ℝ := λ N => N * (N - 4)
  let solutions := {N : ℝ | f N = -21}
  (∃ N₁ N₂, N₁ ∈ solutions ∧ N₂ ∈ solutions ∧ N₁ ≠ N₂) →
  (∀ N, N ∈ solutions → N₁ = N ∨ N₂ = N) →
  N₁ + N₂ = 4
  := by sorry

end sum_of_solutions_eq_four_l1474_147409


namespace business_value_l1474_147417

/-- Given a man who owns 2/3 of a business and sells 3/4 of his shares for 75,000 Rs,
    the total value of the business is 150,000 Rs. -/
theorem business_value (owned_fraction : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) :
  owned_fraction = 2/3 →
  sold_fraction = 3/4 →
  sale_price = 75000 →
  (owned_fraction * sold_fraction * (sale_price : ℚ) / (owned_fraction * sold_fraction)) = 150000 :=
by sorry

end business_value_l1474_147417


namespace somu_father_age_ratio_l1474_147491

/-- Represents the ages of Somu and his father -/
structure Ages where
  somu : ℕ
  father : ℕ

/-- The condition that Somu's age 10 years ago was one-fifth of his father's age 10 years ago -/
def age_condition (ages : Ages) : Prop :=
  ages.somu - 10 = (ages.father - 10) / 5

/-- The theorem stating that given Somu's present age is 20 and the age condition,
    the ratio of Somu's present age to his father's present age is 1:3 -/
theorem somu_father_age_ratio (ages : Ages) 
    (h1 : ages.somu = 20) 
    (h2 : age_condition ages) : 
    (ages.somu : ℚ) / ages.father = 1 / 3 := by
  sorry

end somu_father_age_ratio_l1474_147491


namespace fraction_problem_l1474_147460

theorem fraction_problem :
  let f₁ : ℚ := 75 / 34
  let f₂ : ℚ := 70 / 51
  (f₁ - f₂ = 5 / 6) ∧
  (Nat.gcd 75 70 = 75 - 70) ∧
  (Nat.lcm 75 70 = 1050) ∧
  (∀ a b c d : ℕ, (a / b : ℚ) = f₁ ∧ (c / d : ℚ) = f₂ → Nat.gcd a c = 1 ∧ Nat.gcd b d = 1) :=
by sorry

end fraction_problem_l1474_147460


namespace isometric_figure_area_l1474_147440

/-- A horizontally placed figure with an isometric view -/
structure IsometricFigure where
  /-- The isometric view is an isosceles right triangle -/
  isIsoscelesRightTriangle : Prop
  /-- The legs of the isometric view triangle have length 1 -/
  legLength : ℝ
  /-- The area of the isometric view -/
  isometricArea : ℝ
  /-- The area of the original plane figure -/
  originalArea : ℝ

/-- 
  If a horizontally placed figure has an isometric view that is an isosceles right triangle 
  with legs of length 1, then the area of the original plane figure is √2.
-/
theorem isometric_figure_area 
  (fig : IsometricFigure) 
  (h1 : fig.isIsoscelesRightTriangle) 
  (h2 : fig.legLength = 1) 
  (h3 : fig.isometricArea = 1 / 2) : 
  fig.originalArea = Real.sqrt 2 := by
  sorry

end isometric_figure_area_l1474_147440


namespace product_in_N_l1474_147436

def M : Set ℤ := {x | ∃ m : ℤ, x = 3 * m + 1}
def N : Set ℤ := {y | ∃ n : ℤ, y = 3 * n + 2}

theorem product_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : x * y ∈ N := by
  sorry

end product_in_N_l1474_147436
