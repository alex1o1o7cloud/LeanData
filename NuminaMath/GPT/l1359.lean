import Mathlib

namespace NUMINAMATH_GPT_solve_quadratic_l1359_135911

theorem solve_quadratic : ∃ x : ℚ, 3 * x^2 + 11 * x - 20 = 0 ∧ x > 0 ∧ x = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1359_135911


namespace NUMINAMATH_GPT_uniqueFlavors_l1359_135984

-- Definitions for the conditions
def numRedCandies : ℕ := 6
def numGreenCandies : ℕ := 4
def numBlueCandies : ℕ := 5

-- Condition stating each flavor must use at least two candies and no more than two colors
def validCombination (x y z : ℕ) : Prop :=
  (x = 0 ∨ y = 0 ∨ z = 0) ∧ (x + y ≥ 2 ∨ x + z ≥ 2 ∨ y + z ≥ 2)

-- The main theorem statement
theorem uniqueFlavors : 
  ∃ n : ℕ, n = 30 ∧ 
  (∀ x y z : ℕ, validCombination x y z → (x ≤ numRedCandies) ∧ (y ≤ numGreenCandies) ∧ (z ≤ numBlueCandies)) :=
sorry

end NUMINAMATH_GPT_uniqueFlavors_l1359_135984


namespace NUMINAMATH_GPT_wage_difference_seven_l1359_135914

-- Define the parameters and conditions
variables (P Q h : ℝ)

-- Given conditions
def condition1 : Prop := P = 1.5 * Q
def condition2 : Prop := P * h = 420
def condition3 : Prop := Q * (h + 10) = 420

-- Theorem to be proved
theorem wage_difference_seven (h : ℝ) (P Q : ℝ) 
  (h_condition1 : condition1 P Q)
  (h_condition2 : condition2 P h)
  (h_condition3 : condition3 Q h) :
  (P - Q) = 7 :=
  sorry

end NUMINAMATH_GPT_wage_difference_seven_l1359_135914


namespace NUMINAMATH_GPT_average_study_difference_is_6_l1359_135994

def study_time_differences : List ℤ := [15, -5, 25, -10, 40, -30, 10]

def total_sum (lst : List ℤ) : ℤ := lst.foldr (· + ·) 0

def number_of_days : ℤ := 7

def average_difference : ℤ := total_sum study_time_differences / number_of_days

theorem average_study_difference_is_6 : average_difference = 6 :=
by
  unfold average_difference
  unfold total_sum 
  sorry

end NUMINAMATH_GPT_average_study_difference_is_6_l1359_135994


namespace NUMINAMATH_GPT_telethon_total_revenue_l1359_135995

noncomputable def telethon_revenue (first_period_hours : ℕ) (first_period_rate : ℕ) 
  (additional_percent_increase : ℕ) (second_period_hours : ℕ) : ℕ :=
  let first_revenue := first_period_hours * first_period_rate
  let second_period_rate := first_period_rate + (first_period_rate * additional_percent_increase / 100)
  let second_revenue := second_period_hours * second_period_rate
  first_revenue + second_revenue

theorem telethon_total_revenue : 
  telethon_revenue 12 5000 20 14 = 144000 :=
by 
  rfl -- replace 'rfl' with 'sorry' if the proof is non-trivial and longer

end NUMINAMATH_GPT_telethon_total_revenue_l1359_135995


namespace NUMINAMATH_GPT_maximum_mark_for_paper_i_l1359_135971

noncomputable def maximum_mark (pass_percentage: ℝ) (secured_marks: ℝ) (failed_by: ℝ) : ℝ :=
  (secured_marks + failed_by) / pass_percentage

theorem maximum_mark_for_paper_i :
  maximum_mark 0.35 42 23 = 186 :=
by
  sorry

end NUMINAMATH_GPT_maximum_mark_for_paper_i_l1359_135971


namespace NUMINAMATH_GPT_problem_statement_l1359_135976

theorem problem_statement
  (x y : ℝ)
  (h1 : x * y = 1 / 9)
  (h2 : x * (y + 1) = 7 / 9)
  (h3 : y * (x + 1) = 5 / 18) :
  (x + 1) * (y + 1) = 35 / 18 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1359_135976


namespace NUMINAMATH_GPT_scooped_water_amount_l1359_135923

variables (x : ℝ)

def initial_water_amount : ℝ := 10
def total_amount : ℝ := initial_water_amount
def alcohol_concentration : ℝ := 0.75

theorem scooped_water_amount (h : x / total_amount = alcohol_concentration) : x = 7.5 :=
by sorry

end NUMINAMATH_GPT_scooped_water_amount_l1359_135923


namespace NUMINAMATH_GPT_counting_numbers_leave_remainder_6_divide_53_l1359_135990

theorem counting_numbers_leave_remainder_6_divide_53 :
  ∃! n : ℕ, (∃ k : ℕ, 53 = n * k + 6) ∧ n > 6 :=
sorry

end NUMINAMATH_GPT_counting_numbers_leave_remainder_6_divide_53_l1359_135990


namespace NUMINAMATH_GPT_battery_current_l1359_135940

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end NUMINAMATH_GPT_battery_current_l1359_135940


namespace NUMINAMATH_GPT_min_value_f_l1359_135991

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 10)

theorem min_value_f (h : ∀ x > 10, f x ≥ 40) : ∀ x > 10, f x = 40 → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_l1359_135991


namespace NUMINAMATH_GPT_six_digit_number_representation_l1359_135916

-- Defining that a is a two-digit number
def isTwoDigitNumber (a : ℕ) : Prop := a >= 10 ∧ a < 100

-- Defining that b is a four-digit number
def isFourDigitNumber (b : ℕ) : Prop := b >= 1000 ∧ b < 10000

-- The statement that placing a to the left of b forms the number 10000*a + b
theorem six_digit_number_representation (a b : ℕ) 
  (ha : isTwoDigitNumber a) 
  (hb : isFourDigitNumber b) : 
  (10000 * a + b) = (10^4 * a + b) :=
by
  sorry

end NUMINAMATH_GPT_six_digit_number_representation_l1359_135916


namespace NUMINAMATH_GPT_inequality_solution_l1359_135936

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := |2 - 3 * x| ≥ 4

-- Define the solution set
def solution_set (x : ℝ) : Prop := x ≤ -2/3 ∨ x ≥ 2

-- The theorem that we need to prove
theorem inequality_solution : {x : ℝ | inequality_condition x} = {x : ℝ | solution_set x} :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l1359_135936


namespace NUMINAMATH_GPT_inequality_proof_l1359_135915

variable (a b c d : ℝ)

theorem inequality_proof (ha : 0 ≤ a ∧ a ≤ 1)
                       (hb : 0 ≤ b ∧ b ≤ 1)
                       (hc : 0 ≤ c ∧ c ≤ 1)
                       (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1) ^ 2 ≥ 4 * (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1359_135915


namespace NUMINAMATH_GPT_triangle_cos_identity_l1359_135939

variable {A B C : ℝ} -- Angle A, B, C are real numbers representing the angles of the triangle
variable {a b c : ℝ} -- Sides a, b, c are real numbers representing the lengths of the sides of the triangle

theorem triangle_cos_identity (h : 2 * b = a + c) : 5 * (Real.cos A) - 4 * (Real.cos A) * (Real.cos C) + 5 * (Real.cos C) = 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_cos_identity_l1359_135939


namespace NUMINAMATH_GPT_investment_years_l1359_135968

def principal (P : ℝ) := P = 1200
def rate (r : ℝ) := r = 0.10
def interest_diff (P r : ℝ) (t : ℝ) :=
  let SI := P * r * t
  let CI := P * (1 + r)^t - P
  CI - SI = 12

theorem investment_years (P r : ℝ) (t : ℝ) 
  (h_principal : principal P) 
  (h_rate : rate r) 
  (h_diff : interest_diff P r t) : 
  t = 2 := 
sorry

end NUMINAMATH_GPT_investment_years_l1359_135968


namespace NUMINAMATH_GPT_net_salary_change_l1359_135961

variable (S : ℝ)

theorem net_salary_change (h1 : S > 0) : 
  (1.3 * S - 0.3 * (1.3 * S)) - S = -0.09 * S := by
  sorry

end NUMINAMATH_GPT_net_salary_change_l1359_135961


namespace NUMINAMATH_GPT_aftershave_alcohol_concentration_l1359_135996

def initial_volume : ℝ := 12
def initial_concentration : ℝ := 0.60
def desired_concentration : ℝ := 0.40
def water_added : ℝ := 6
def final_volume : ℝ := initial_volume + water_added

theorem aftershave_alcohol_concentration :
  initial_concentration * initial_volume = desired_concentration * final_volume :=
by
  sorry

end NUMINAMATH_GPT_aftershave_alcohol_concentration_l1359_135996


namespace NUMINAMATH_GPT_best_in_district_round_l1359_135934

-- Assume a structure that lets us refer to positions
inductive Position
| first
| second
| third
| last

open Position

-- Definitions of the statements
def Eva (p : Position → Prop) := ¬ (p first) ∧ ¬ (p last)
def Mojmir (p : Position → Prop) := ¬ (p last)
def Karel (p : Position → Prop) := p first
def Peter (p : Position → Prop) := p last

-- The main hypothesis
def exactly_one_lie (p : Position → Prop) :=
  (Eva p ∧ Mojmir p ∧ Karel p ∧ ¬ (Peter p)) ∨
  (Eva p ∧ Mojmir p ∧ ¬ (Karel p) ∧ Peter p) ∨
  (Eva p ∧ ¬ (Mojmir p) ∧ Karel p ∧ Peter p) ∨
  (¬ (Eva p) ∧ Mojmir p ∧ Karel p ∧ Peter p)

theorem best_in_district_round :
  ∃ (p : Position → Prop),
    (Eva p ∧ Mojmir p ∧ ¬ (Karel p) ∧ Peter p) ∧ exactly_one_lie p :=
by
  sorry

end NUMINAMATH_GPT_best_in_district_round_l1359_135934


namespace NUMINAMATH_GPT_quadrilateral_area_l1359_135908

theorem quadrilateral_area (a b c d e f : ℝ) : 
    (a^2 + c^2 - b^2 - d^2) ^ 2 ≤ 4 * e^2 * f^2 :=
    by sorry

noncomputable def quadrilateral_area_formula (a b c d e f : ℝ) : ℝ :=
    if H : (a^2 + c^2 - b^2 - d^2) ^ 2 ≤ 4 * e^2 * f^2 then 
    (1/4) * Real.sqrt (4 * e^2 * f^2 - (a^2 + c^2 - b^2 - d^2) ^ 2)
    else 0

-- Ensure that the computed area matches the expected value
example (a b c d e f : ℝ) (H : (a^2 + c^2 - b^2 - d^2)^2 ≤ 4 * e^2 * f^2) : 
    quadrilateral_area_formula a b c d e f = 
        (1/4) * Real.sqrt (4 * e^2 * f^2 - (a^2 + c^2 - b^2 - d^2) ^ 2) :=
by simp [quadrilateral_area_formula, H]

end NUMINAMATH_GPT_quadrilateral_area_l1359_135908


namespace NUMINAMATH_GPT_michael_weight_loss_in_may_l1359_135918

-- Defining the conditions
def weight_loss_goal : ℕ := 10
def weight_loss_march : ℕ := 3
def weight_loss_april : ℕ := 4

-- Statement of the problem to prove
theorem michael_weight_loss_in_may (weight_loss_goal weight_loss_march weight_loss_april : ℕ) :
  weight_loss_goal - (weight_loss_march + weight_loss_april) = 3 :=
by
  sorry

end NUMINAMATH_GPT_michael_weight_loss_in_may_l1359_135918


namespace NUMINAMATH_GPT_grid_black_probability_l1359_135906

theorem grid_black_probability :
  let p_black_each_cell : ℝ := 1 / 3 
  let p_not_black : ℝ := (2 / 3) * (2 / 3)
  let p_one_black : ℝ := 1 - p_not_black
  let total_pairs : ℕ := 8
  (p_one_black ^ total_pairs) = (5 / 9) ^ 8 :=
sorry

end NUMINAMATH_GPT_grid_black_probability_l1359_135906


namespace NUMINAMATH_GPT_area_ratio_l1359_135974

-- Definitions for the geometric entities
structure Point where
  x : ℚ
  y : ℚ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨0, 4⟩
def C : Point := ⟨2, 4⟩
def D : Point := ⟨2, 0⟩
def E : Point := ⟨1, 2⟩  -- Midpoint of BD
def F : Point := ⟨6 / 5, 0⟩  -- Given DF = 2/5 DA

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P Q R : Point) : ℚ :=
  (1 / 2) * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

-- Function to calculate the sum of the area of two triangles
def quadrilateral_area (P Q R S : Point) : ℚ :=
  triangle_area P Q R + triangle_area P R S

-- Prove the ratio of the areas
theorem area_ratio : 
  triangle_area D F E / quadrilateral_area A B E F = 4 / 13 := 
by {
  sorry
}

end NUMINAMATH_GPT_area_ratio_l1359_135974


namespace NUMINAMATH_GPT_part1_part2_l1359_135992

variable (a b c : ℝ)

-- Conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a^2 + b^2 + 4*c^2 = 3

-- Part 1: Prove that a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 := sorry

-- Part 2: Given b = 2c, prove that 1/a + 1/c ≥ 3
axiom h5 : b = 2*c
theorem part2 : 1/a + 1/c ≥ 3 := sorry

end NUMINAMATH_GPT_part1_part2_l1359_135992


namespace NUMINAMATH_GPT_bear_problem_l1359_135912

variables (w b br : ℕ)

theorem bear_problem 
    (h1 : b = 2 * w)
    (h2 : br = b + 40)
    (h3 : w + b + br = 190) :
    b = 60 :=
by
  sorry

end NUMINAMATH_GPT_bear_problem_l1359_135912


namespace NUMINAMATH_GPT_exists_monomials_l1359_135969

theorem exists_monomials (a b : ℕ) :
  ∃ x y : ℕ → ℕ → ℤ,
  (x 2 1 * y 2 1 = -12) ∧
  (∀ m n : ℕ, m ≠ 2 ∨ n ≠ 1 → x m n = 0 ∧ y m n = 0) ∧
  (∃ k l : ℤ, x 2 1 = k * (a ^ 2 * b ^ 1) ∧ y 2 1 = l * (a ^ 2 * b ^ 1) ∧ k + l = 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_monomials_l1359_135969


namespace NUMINAMATH_GPT_distinct_quadrilateral_areas_l1359_135978

theorem distinct_quadrilateral_areas (A B C D E F : ℝ) 
  (h : A + B + C + D + E + F = 156) :
  ∃ (Q1 Q2 Q3 : ℝ), Q1 = 78 ∧ Q2 = 104 ∧ Q3 = 104 :=
sorry

end NUMINAMATH_GPT_distinct_quadrilateral_areas_l1359_135978


namespace NUMINAMATH_GPT_ac_lt_bd_l1359_135935

theorem ac_lt_bd (a b c d : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) : a * c < b * d :=
by
  sorry

end NUMINAMATH_GPT_ac_lt_bd_l1359_135935


namespace NUMINAMATH_GPT_olivia_spent_amount_l1359_135982

noncomputable def initial_amount : ℕ := 100
noncomputable def collected_amount : ℕ := 148
noncomputable def final_amount : ℕ := 159

theorem olivia_spent_amount :
  initial_amount + collected_amount - final_amount = 89 :=
by
  sorry

end NUMINAMATH_GPT_olivia_spent_amount_l1359_135982


namespace NUMINAMATH_GPT_least_possible_area_of_square_l1359_135975

theorem least_possible_area_of_square :
  (∃ (side_length : ℝ), 3.5 ≤ side_length ∧ side_length < 4.5 ∧ 
    (∃ (area : ℝ), area = side_length * side_length ∧ 
    (∀ (side : ℝ), 3.5 ≤ side ∧ side < 4.5 → side * side ≥ 12.25))) :=
sorry

end NUMINAMATH_GPT_least_possible_area_of_square_l1359_135975


namespace NUMINAMATH_GPT_solution_set_inequality_l1359_135947

theorem solution_set_inequality (x : ℝ) (h1 : 2 < 1 / (x - 1)) (h2 : 1 / (x - 1) < 3) (h3 : x - 1 > 0) :
  4 / 3 < x ∧ x < 3 / 2 :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1359_135947


namespace NUMINAMATH_GPT_retailer_profit_percentage_l1359_135964

theorem retailer_profit_percentage 
  (CP MP SP : ℝ)
  (hCP : CP = 100)
  (hMP : MP = CP + 0.65 * CP)
  (hSP : SP = MP - 0.25 * MP)
  : ((SP - CP) / CP) * 100 = 23.75 := 
sorry

end NUMINAMATH_GPT_retailer_profit_percentage_l1359_135964


namespace NUMINAMATH_GPT_soybeans_to_oil_kg_l1359_135922

-- Define initial data
def kgSoybeansToTofu : ℕ := 3
def kgSoybeansToOil : ℕ := 6
def kgTofuCostPerKg : ℕ := 3
def kgOilCostPerKg : ℕ := 15
def batchSoybeansKg : ℕ := 460
def totalRevenue : ℕ := 1800

-- Define problem statement
theorem soybeans_to_oil_kg (x y : ℕ) (h : x + y = batchSoybeansKg) 
  (hRevenue : 3 * kgTofuCostPerKg * x + (kgOilCostPerKg * y) / (kgSoybeansToOil) = totalRevenue) : 
  y = 360 :=
sorry

end NUMINAMATH_GPT_soybeans_to_oil_kg_l1359_135922


namespace NUMINAMATH_GPT_part1_correct_part2_correct_l1359_135932

-- Definitions for conditions
def total_students := 200
def likes_employment := 140
def dislikes_employment := 60
def p_likes : ℚ := likes_employment / total_students

def male_likes := 60
def male_dislikes := 40
def female_likes := 80
def female_dislikes := 20
def n := total_students
def alpha := 0.005
def chi_squared_critical_value := 7.879

-- Part 1: Estimate the probability of selecting at least 2 students who like employment
def probability_at_least_2_of_3 : ℚ :=
  3 * ((7/10) ^ 2) * (3/10) + ((7/10) ^ 3)

-- Proof goal for Part 1
theorem part1_correct : probability_at_least_2_of_3 = 98 / 125 := by
  sorry

-- Part 2: Chi-squared test for independence between intention and gender
def a := male_likes
def b := male_dislikes
def c := female_likes
def d := female_dislikes
def chi_squared_statistic : ℚ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof goal for Part 2
theorem part2_correct : chi_squared_statistic = 200 / 21 ∧ 200 / 21 > chi_squared_critical_value := by
  sorry

end NUMINAMATH_GPT_part1_correct_part2_correct_l1359_135932


namespace NUMINAMATH_GPT_machines_together_work_time_l1359_135926

theorem machines_together_work_time :
  let rate_A := 1 / 4
  let rate_B := 1 / 12
  let rate_C := 1 / 6
  let rate_D := 1 / 8
  let rate_E := 1 / 18
  let total_rate := rate_A + rate_B + rate_C + rate_D + rate_E
  total_rate ≠ 0 → 
  let total_time := 1 / total_rate
  total_time = 72 / 49 :=
by
  sorry

end NUMINAMATH_GPT_machines_together_work_time_l1359_135926


namespace NUMINAMATH_GPT_store_earnings_correct_l1359_135913

theorem store_earnings_correct :
  let graphics_cards_qty := 10
  let hard_drives_qty := 14
  let cpus_qty := 8
  let rams_qty := 4
  let psus_qty := 12
  let monitors_qty := 6
  let keyboards_qty := 18
  let mice_qty := 24

  let graphics_card_price := 600
  let hard_drive_price := 80
  let cpu_price := 200
  let ram_price := 60
  let psu_price := 90
  let monitor_price := 250
  let keyboard_price := 40
  let mouse_price := 20

  let total_earnings := graphics_cards_qty * graphics_card_price +
                        hard_drives_qty * hard_drive_price +
                        cpus_qty * cpu_price +
                        rams_qty * ram_price +
                        psus_qty * psu_price +
                        monitors_qty * monitor_price +
                        keyboards_qty * keyboard_price +
                        mice_qty * mouse_price
  total_earnings = 12740 :=
by
  -- definitions and calculations here
  sorry

end NUMINAMATH_GPT_store_earnings_correct_l1359_135913


namespace NUMINAMATH_GPT_smallest_area_of_right_triangle_l1359_135972

noncomputable def right_triangle_area (a b : ℝ) : ℝ :=
  if a^2 + b^2 = 6^2 then (1/2) * a * b else 12

theorem smallest_area_of_right_triangle :
  min (right_triangle_area 4 (2 * Real.sqrt 5)) 12 = 4 * Real.sqrt 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_smallest_area_of_right_triangle_l1359_135972


namespace NUMINAMATH_GPT_polynomial_expansion_l1359_135930

theorem polynomial_expansion :
  let x := 1 
  let y := -1 
  let a_0 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_1 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  let a_2 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_3 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  let a_4 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_5 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3 + a_5)^2 = 3125 := by
sorry

end NUMINAMATH_GPT_polynomial_expansion_l1359_135930


namespace NUMINAMATH_GPT_row_trip_time_example_l1359_135981

noncomputable def round_trip_time
    (rowing_speed : ℝ)
    (current_speed : ℝ)
    (total_distance : ℝ) : ℝ :=
  let downstream_speed := rowing_speed + current_speed
  let upstream_speed := rowing_speed - current_speed
  let one_way_distance := total_distance / 2
  let time_to_place := one_way_distance / downstream_speed
  let time_back := one_way_distance / upstream_speed
  time_to_place + time_back

theorem row_trip_time_example :
  round_trip_time 10 2 96 = 10 := by
  sorry

end NUMINAMATH_GPT_row_trip_time_example_l1359_135981


namespace NUMINAMATH_GPT_cos_arith_prog_impossible_l1359_135950

noncomputable def sin_arith_prog (x y z : ℝ) : Prop :=
  (2 * Real.sin y = Real.sin x + Real.sin z) ∧ (Real.sin x < Real.sin y) ∧ (Real.sin y < Real.sin z)

theorem cos_arith_prog_impossible (x y z : ℝ) (h : sin_arith_prog x y z) : 
  ¬(2 * Real.cos y = Real.cos x + Real.cos z) := 
by 
  sorry

end NUMINAMATH_GPT_cos_arith_prog_impossible_l1359_135950


namespace NUMINAMATH_GPT_xy_extrema_l1359_135987

noncomputable def xy_product (a : ℝ) : ℝ := a^2 - 1

theorem xy_extrema (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^2 + y^2 = -a^2 + 2) : 
  -1 ≤ xy_product a ∧ xy_product a ≤ 1/3 :=
by
  sorry

end NUMINAMATH_GPT_xy_extrema_l1359_135987


namespace NUMINAMATH_GPT_committee_count_l1359_135988

-- Definitions based on conditions
def num_males := 15
def num_females := 10

-- Define the binomial coefficient
def binomial (n k : ℕ) := Nat.choose n k

-- Define the total number of committees
def num_committees_with_at_least_two_females : ℕ :=
  binomial num_females 2 * binomial num_males 3 +
  binomial num_females 3 * binomial num_males 2 +
  binomial num_females 4 * binomial num_males 1 +
  binomial num_females 5 * binomial num_males 0

theorem committee_count : num_committees_with_at_least_two_females = 36477 :=
by {
  sorry
}

end NUMINAMATH_GPT_committee_count_l1359_135988


namespace NUMINAMATH_GPT_find_a_l1359_135993

theorem find_a (a : ℝ) (h : a^2 + a^2 / 4 = 5) : a = 2 ∨ a = -2 := 
sorry

end NUMINAMATH_GPT_find_a_l1359_135993


namespace NUMINAMATH_GPT_median_possible_values_l1359_135952

variable {ι : Type} -- Representing the set S as a type
variable (S : Finset ℤ) -- S is a finite set of integers

def conditions (S: Finset ℤ) : Prop :=
  S.card = 9 ∧
  {5, 7, 10, 13, 17, 21} ⊆ S

theorem median_possible_values :
  ∀ S : Finset ℤ, conditions S → ∃ medians : Finset ℤ, medians.card = 7 :=
by
  sorry

end NUMINAMATH_GPT_median_possible_values_l1359_135952


namespace NUMINAMATH_GPT_ducks_in_smaller_pond_l1359_135925

theorem ducks_in_smaller_pond (x : ℝ) (h1 : 50 > 0) 
  (h2 : 0.20 * x > 0) (h3 : 0.12 * 50 > 0) (h4 : 0.15 * (x + 50) = 0.20 * x + 0.12 * 50) 
  : x = 30 := 
sorry

end NUMINAMATH_GPT_ducks_in_smaller_pond_l1359_135925


namespace NUMINAMATH_GPT_sum_of_present_ages_l1359_135945

def Jed_age_future (current_Jed: ℕ) (years: ℕ) : ℕ := 
  current_Jed + years

def Matt_age (current_Jed: ℕ) : ℕ := 
  current_Jed - 10

def sum_ages (jed_age: ℕ) (matt_age: ℕ) : ℕ := 
  jed_age + matt_age

theorem sum_of_present_ages :
  ∃ jed_curr_age matt_curr_age : ℕ, 
  (Jed_age_future jed_curr_age 10 = 25) ∧ 
  (jed_curr_age = matt_curr_age + 10) ∧ 
  (sum_ages jed_curr_age matt_curr_age = 20) :=
sorry

end NUMINAMATH_GPT_sum_of_present_ages_l1359_135945


namespace NUMINAMATH_GPT_bus_A_speed_l1359_135962

variable (v_A v_B : ℝ)
variable (h1 : v_A - v_B = 15)
variable (h2 : v_A + v_B = 75)

theorem bus_A_speed : v_A = 45 := sorry

end NUMINAMATH_GPT_bus_A_speed_l1359_135962


namespace NUMINAMATH_GPT_length_of_first_train_is_correct_l1359_135953

noncomputable def length_of_first_train (speed1_km_hr speed2_km_hr : ℝ) (time_cross_sec : ℝ) (length2_m : ℝ) : ℝ :=
  let speed1_m_s := speed1_km_hr * (5 / 18)
  let speed2_m_s := speed2_km_hr * (5 / 18)
  let relative_speed_m_s := speed1_m_s + speed2_m_s
  let total_distance_m := relative_speed_m_s * time_cross_sec
  total_distance_m - length2_m

theorem length_of_first_train_is_correct : 
  length_of_first_train 60 40 11.879049676025918 160 = 170 := by
  sorry

end NUMINAMATH_GPT_length_of_first_train_is_correct_l1359_135953


namespace NUMINAMATH_GPT_min_value_5_l1359_135937

theorem min_value_5 (x y : ℝ) : ∃ x y : ℝ, (xy - 2)^2 + (x + y + 1)^2 = 5 :=
sorry

end NUMINAMATH_GPT_min_value_5_l1359_135937


namespace NUMINAMATH_GPT_height_of_bottom_step_l1359_135979

variable (h l w : ℝ)

theorem height_of_bottom_step
  (h l w : ℝ)
  (eq1 : l + h - w / 2 = 42)
  (eq2 : 2 * l + h = 38)
  (w_value : w = 4) : h = 34 := by
sorry

end NUMINAMATH_GPT_height_of_bottom_step_l1359_135979


namespace NUMINAMATH_GPT_proof_f_f_f_3_l1359_135941

def f (n : ℤ) : ℤ :=
  if n < 5
  then n^2 + 1
  else 2 * n - 3

theorem proof_f_f_f_3 :
  f (f (f 3)) = 31 :=
by 
  -- Here, we skip the proof as instructed
  sorry

end NUMINAMATH_GPT_proof_f_f_f_3_l1359_135941


namespace NUMINAMATH_GPT_system_of_equations_l1359_135910

theorem system_of_equations (x y z : ℝ) (h1 : 4 * x - 6 * y - 2 * z = 0) (h2 : 2 * x + 6 * y - 28 * z = 0) (hz : z ≠ 0) :
  (x^2 - 6 * x * y) / (y^2 + 4 * z^2) = -5 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_l1359_135910


namespace NUMINAMATH_GPT_prime_product_sum_l1359_135989

theorem prime_product_sum (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (h : (p * q * r = 101 * (p + q + r))) : 
  p = 101 ∧ q = 2 ∧ r = 103 :=
sorry

end NUMINAMATH_GPT_prime_product_sum_l1359_135989


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1359_135900

theorem quadratic_no_real_roots (k : ℝ) : (∀ x : ℝ, x^2 - 3 * x - k ≠ 0) → k < -9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1359_135900


namespace NUMINAMATH_GPT_zoo_structure_l1359_135929

theorem zoo_structure (P : ℕ) (h1 : ∃ (snakes monkeys elephants zebras : ℕ),
  snakes = 3 * P ∧
  monkeys = 6 * P ∧
  elephants = (P + snakes) / 2 ∧
  zebras = elephants - 3 ∧
  monkeys - zebras = 35) : P = 8 :=
sorry

end NUMINAMATH_GPT_zoo_structure_l1359_135929


namespace NUMINAMATH_GPT_percentage_loss_l1359_135967

theorem percentage_loss (SP_loss SP_profit CP : ℝ) 
  (h₁ : SP_loss = 9) 
  (h₂ : SP_profit = 11.8125) 
  (h₃ : SP_profit = CP * 1.05) : 
  (CP - SP_loss) / CP * 100 = 20 :=
by sorry

end NUMINAMATH_GPT_percentage_loss_l1359_135967


namespace NUMINAMATH_GPT_rectangles_perimeter_l1359_135924

theorem rectangles_perimeter : 
  let base := 4
  let top := 4
  let left_side := 4
  let right_side := 6
  base + top + left_side + right_side = 18 := 
by {
  let base := 4
  let top := 4
  let left_side := 4
  let right_side := 6
  sorry
}

end NUMINAMATH_GPT_rectangles_perimeter_l1359_135924


namespace NUMINAMATH_GPT_distinct_triangle_areas_l1359_135980

variables (A B C D E F G : ℝ) (h : ℝ)
variables (AB BC CD EF FG AC BD AD EG : ℝ)

def is_valid_points := AB = 2 ∧ BC = 1 ∧ CD = 3 ∧ EF = 1 ∧ FG = 2 ∧ AC = AB + BC ∧ BD = BC + CD ∧ AD = AB + BC + CD ∧ EG = EF + FG

theorem distinct_triangle_areas (h_pos : 0 < h) (valid : is_valid_points AB BC CD EF FG AC BD AD EG) : 
  ∃ n : ℕ, n = 5 := 
by
  sorry

end NUMINAMATH_GPT_distinct_triangle_areas_l1359_135980


namespace NUMINAMATH_GPT_proof_problem_l1359_135931

noncomputable def p : Prop := ∃ x : ℝ, x^2 + 1 / x^2 ≤ 2
def q : Prop := ¬ p

theorem proof_problem : q ∧ (p ∨ q) :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_proof_problem_l1359_135931


namespace NUMINAMATH_GPT_jill_arrives_15_minutes_before_jack_l1359_135943

theorem jill_arrives_15_minutes_before_jack
  (distance : ℝ) (jill_speed : ℝ) (jack_speed : ℝ) (start_same_time : true)
  (h_distance : distance = 2) (h_jill_speed : jill_speed = 8) (h_jack_speed : jack_speed = 4) :
  (2 / 4 * 60) - (2 / 8 * 60) = 15 :=
by
  sorry

end NUMINAMATH_GPT_jill_arrives_15_minutes_before_jack_l1359_135943


namespace NUMINAMATH_GPT_plan_A_is_cost_effective_l1359_135942

-- Definitions of the costs considering the problem's conditions
def cost_plan_A (days_A : ℕ) (rate_A : ℕ) : ℕ := days_A * rate_A
def cost_plan_C (days_AB : ℕ) (rate_A : ℕ) (rate_B : ℕ) (remaining_B : ℕ) : ℕ :=
  (days_AB * (rate_A + rate_B)) + (remaining_B * rate_B)

-- Specification of the days and rates from the conditions
def days_A := 12
def rate_A := 10000
def rate_B := 6000
def days_AB := 3
def remaining_B := 13

-- Costs for each plan
def A_cost := cost_plan_A days_A rate_A
def C_cost := cost_plan_C days_AB rate_A rate_B remaining_B

-- Theorem stating that Plan A is more cost-effective
theorem plan_A_is_cost_effective : A_cost < C_cost := by
  unfold A_cost
  unfold C_cost
  sorry

end NUMINAMATH_GPT_plan_A_is_cost_effective_l1359_135942


namespace NUMINAMATH_GPT_integral_eval_l1359_135949

noncomputable def integral_problem : ℝ :=
  ∫ x in - (Real.pi / 2)..(Real.pi / 2), (x + Real.cos x)

theorem integral_eval : integral_problem = 2 :=
  by 
  sorry

end NUMINAMATH_GPT_integral_eval_l1359_135949


namespace NUMINAMATH_GPT_power_function_point_l1359_135903

theorem power_function_point (n : ℕ) (hn : 2^n = 8) : n = 3 := 
by
  sorry

end NUMINAMATH_GPT_power_function_point_l1359_135903


namespace NUMINAMATH_GPT_average_sales_six_months_l1359_135985

theorem average_sales_six_months :
  let sales1 := 4000
  let sales2 := 6524
  let sales3 := 5689
  let sales4 := 7230
  let sales5 := 6000
  let sales6 := 12557
  let total_sales_first_five := sales1 + sales2 + sales3 + sales4 + sales5
  let total_sales_six := total_sales_first_five + sales6
  let average_sales := total_sales_six / 6
  average_sales = 7000 :=
by
  let sales1 := 4000
  let sales2 := 6524
  let sales3 := 5689
  let sales4 := 7230
  let sales5 := 6000
  let sales6 := 12557
  let total_sales_first_five := sales1 + sales2 + sales3 + sales4 + sales5
  let total_sales_six := total_sales_first_five + sales6
  let average_sales := total_sales_six / 6
  have h : total_sales_first_five = 29443 := by sorry
  have h1 : total_sales_six = 42000 := by sorry
  have h2 : average_sales = 7000 := by sorry
  exact h2

end NUMINAMATH_GPT_average_sales_six_months_l1359_135985


namespace NUMINAMATH_GPT_greatest_divisor_of_three_consecutive_odds_l1359_135928

theorem greatest_divisor_of_three_consecutive_odds (n : ℕ) : 
  ∃ (d : ℕ), (∀ (k : ℕ), k = 2*n + 1 ∨ k = 2*n + 3 ∨ k = 2*n + 5 → d ∣ (2*n + 1) * (2*n + 3) * (2*n + 5)) ∧ d = 3 :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_three_consecutive_odds_l1359_135928


namespace NUMINAMATH_GPT_percent_calculation_l1359_135970

theorem percent_calculation (x : ℝ) (h : 0.40 * x = 160) : 0.30 * x = 120 :=
by
  sorry

end NUMINAMATH_GPT_percent_calculation_l1359_135970


namespace NUMINAMATH_GPT_range_of_m_l1359_135946

noncomputable def condition_p (x : ℝ) : Prop := -2 < x ∧ x < 10
noncomputable def condition_q (x m : ℝ) : Prop := (x - 1)^2 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) :
  (∀ x, condition_p x → condition_q x m) ∧ (∃ x, ¬ condition_p x ∧ condition_q x m) ↔ 9 ≤ m := sorry

end NUMINAMATH_GPT_range_of_m_l1359_135946


namespace NUMINAMATH_GPT_find_b_l1359_135956

theorem find_b (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 20)
  (h3 : (5 + 4 * 83 + 6 * 83^2 + 3 * 83^3 + 7 * 83^4 + 5 * 83^5 + 2 * 83^6 - b) % 17 = 0) :
  b = 8 :=
sorry

end NUMINAMATH_GPT_find_b_l1359_135956


namespace NUMINAMATH_GPT_dasha_rectangle_l1359_135955

theorem dasha_rectangle:
  ∃ (a b c : ℤ), a * (2 * b + 2 * c - a) = 43 ∧ a = 1 ∧ b + c = 22 :=
by
  sorry

end NUMINAMATH_GPT_dasha_rectangle_l1359_135955


namespace NUMINAMATH_GPT_prove_angle_A_l1359_135944

-- Definitions and conditions in triangle ABC
variables (A B C : ℝ) (a b c : ℝ) (h₁ : a^2 - b^2 = 3 * b * c) (h₂ : sin C = 2 * sin B)

-- Objective: Prove that angle A is 120 degrees
theorem prove_angle_A : A = 120 :=
sorry

end NUMINAMATH_GPT_prove_angle_A_l1359_135944


namespace NUMINAMATH_GPT_sum_of_angles_is_correct_l1359_135954

noncomputable def hexagon_interior_angle : ℝ := 180 * (6 - 2) / 6
noncomputable def pentagon_interior_angle : ℝ := 180 * (5 - 2) / 5
noncomputable def sum_of_hexagon_and_pentagon_angles (A B C D : Type) 
  (hexagon_interior_angle : ℝ) 
  (pentagon_interior_angle : ℝ) : ℝ := 
  hexagon_interior_angle + pentagon_interior_angle

theorem sum_of_angles_is_correct (A B C D : Type) : 
  sum_of_hexagon_and_pentagon_angles A B C D hexagon_interior_angle pentagon_interior_angle = 228 := 
by
  simp [hexagon_interior_angle, pentagon_interior_angle]
  sorry

end NUMINAMATH_GPT_sum_of_angles_is_correct_l1359_135954


namespace NUMINAMATH_GPT_solve_system_of_equations_l1359_135966

theorem solve_system_of_equations (a1 a2 a3 a4 : ℝ) (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (x1 x2 x3 x4 : ℝ)
  (h1 : |a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1)
  (h2 : |a2 - a1| * x1 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1)
  (h3 : |a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a4| * x4 = 1)
  (h4 : |a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 = 1) :
  x1 = 1 / (a4 - a1) ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 1 / (a4 - a1) := 
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1359_135966


namespace NUMINAMATH_GPT_find_abscissa_of_P_l1359_135948

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem find_abscissa_of_P (x_P : ℝ) :
  (x + 2*y - 1 = 0 -> 
  (f' x_P = 2 -> 
  (f x_P - 2) * (x_P^2 - 1) = 0)) := by
  sorry

end NUMINAMATH_GPT_find_abscissa_of_P_l1359_135948


namespace NUMINAMATH_GPT_value_of_a_add_b_l1359_135977

theorem value_of_a_add_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
sorry

end NUMINAMATH_GPT_value_of_a_add_b_l1359_135977


namespace NUMINAMATH_GPT_sum_of_numerator_and_denominator_of_decimal_0_345_l1359_135965

def repeating_decimal_to_fraction_sum (x : ℚ) : ℕ :=
if h : x = 115 / 333 then 115 + 333 else 0

theorem sum_of_numerator_and_denominator_of_decimal_0_345 :
  repeating_decimal_to_fraction_sum 345 / 999 = 448 :=
by {
  -- Given: 0.\overline{345} = 345 / 999, simplified to 115 / 333
  -- hence the sum of numerator and denominator = 115 + 333
  -- We don't need the proof steps here, just conclude with the sum
  sorry }

end NUMINAMATH_GPT_sum_of_numerator_and_denominator_of_decimal_0_345_l1359_135965


namespace NUMINAMATH_GPT_find_abc_l1359_135909

theorem find_abc (a b c : ℝ) (x y : ℝ) :
  (x^2 + y^2 + 2*a*x - b*y + c = 0) ∧
  ((-a, b / 2) = (2, 2)) ∧
  (4 = b^2 / 4 + a^2 - c) →
  a = -2 ∧ b = 4 ∧ c = 4 := by
  sorry

end NUMINAMATH_GPT_find_abc_l1359_135909


namespace NUMINAMATH_GPT_find_f_2018_l1359_135933

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_functional_eq : ∀ x : ℝ, f x = - (1 / f (x + 3))
axiom f_at_4 : f 4 = -2018

theorem find_f_2018 : f 2018 = -2018 :=
  sorry

end NUMINAMATH_GPT_find_f_2018_l1359_135933


namespace NUMINAMATH_GPT_minimize_distance_midpoint_Q5_Q6_l1359_135917

theorem minimize_distance_midpoint_Q5_Q6 
  (Q : ℝ → ℝ)
  (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10 : ℝ)
  (h1 : Q2 = Q1 + 1)
  (h2 : Q3 = Q2 + 1)
  (h3 : Q4 = Q3 + 1)
  (h4 : Q5 = Q4 + 1)
  (h5 : Q6 = Q5 + 2)
  (h6 : Q7 = Q6 + 2)
  (h7 : Q8 = Q7 + 2)
  (h8 : Q9 = Q8 + 2)
  (h9 : Q10 = Q9 + 2) :
  Q ((Q5 + Q6) / 2) = (Q ((Q1 + Q2) / 2) + Q ((Q3 + Q4) / 2) + Q ((Q7 + Q8) / 2) + Q ((Q9 + Q10) / 2)) :=
sorry

end NUMINAMATH_GPT_minimize_distance_midpoint_Q5_Q6_l1359_135917


namespace NUMINAMATH_GPT_integer_with_exactly_12_integers_to_its_left_l1359_135963

theorem integer_with_exactly_12_integers_to_its_left :
  let initial_list := List.range' 1 20
  let first_half := initial_list.take 10
  let second_half := initial_list.drop 10
  let new_list := second_half ++ first_half
  new_list.get! 12 = 3 :=
by
  let initial_list := List.range' 1 20
  let first_half := initial_list.take 10
  let second_half := initial_list.drop 10
  let new_list := second_half ++ first_half
  sorry

end NUMINAMATH_GPT_integer_with_exactly_12_integers_to_its_left_l1359_135963


namespace NUMINAMATH_GPT_larger_triangle_perimeter_is_126_l1359_135951

noncomputable def smaller_triangle_side1 : ℝ := 12
noncomputable def smaller_triangle_side2 : ℝ := 12
noncomputable def smaller_triangle_base : ℝ := 18
noncomputable def larger_triangle_longest_side : ℝ := 54
noncomputable def similarity_ratio : ℝ := larger_triangle_longest_side / smaller_triangle_base
noncomputable def larger_triangle_side1 : ℝ := smaller_triangle_side1 * similarity_ratio
noncomputable def larger_triangle_side2 : ℝ := smaller_triangle_side2 * similarity_ratio
noncomputable def larger_triangle_perimeter : ℝ := larger_triangle_side1 + larger_triangle_side2 + larger_triangle_longest_side

theorem larger_triangle_perimeter_is_126 :
  larger_triangle_perimeter = 126 := by
  sorry

end NUMINAMATH_GPT_larger_triangle_perimeter_is_126_l1359_135951


namespace NUMINAMATH_GPT_question_l1359_135905

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.log x - 7

theorem question (x : ℝ) (n : ℕ) (h1 : 2 < x ∧ x < 3) (h2 : f x = 0) : n = 2 := by
  sorry

end NUMINAMATH_GPT_question_l1359_135905


namespace NUMINAMATH_GPT_intersection_primes_evens_l1359_135901

open Set

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def evens : Set ℕ := {n | n % 2 = 0}
def primes : Set ℕ := {n | is_prime n}

theorem intersection_primes_evens :
  primes ∩ evens = {2} :=
by sorry

end NUMINAMATH_GPT_intersection_primes_evens_l1359_135901


namespace NUMINAMATH_GPT_inequality_problem_l1359_135902

variable {a b c d : ℝ}

theorem inequality_problem (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
    a + c > b + d ∧ ad^2 > bc^2 ∧ (1 / bc) < (1 / ad) :=
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l1359_135902


namespace NUMINAMATH_GPT_g_of_f_of_3_is_217_l1359_135960

-- Definitions of f and g
def f (x : ℝ) : ℝ := x^2 - 4
def g (x : ℝ) : ℝ := x^3 + 3 * x^2 + 3 * x + 2

-- The theorem we need to prove
theorem g_of_f_of_3_is_217 : g (f 3) = 217 := by
  sorry

end NUMINAMATH_GPT_g_of_f_of_3_is_217_l1359_135960


namespace NUMINAMATH_GPT_monotonically_increasing_range_a_l1359_135921

theorem monotonically_increasing_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (x^3 + a * x) ≤ (y^3 + a * y)) → a ≥ 0 := 
by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_range_a_l1359_135921


namespace NUMINAMATH_GPT_wooden_toy_price_l1359_135983

noncomputable def price_of_hat : ℕ := 10
noncomputable def total_money : ℕ := 100
noncomputable def hats_bought : ℕ := 3
noncomputable def change_received : ℕ := 30
noncomputable def total_spent := total_money - change_received
noncomputable def cost_of_hats := hats_bought * price_of_hat

theorem wooden_toy_price :
  ∃ (W : ℕ), total_spent = 2 * W + cost_of_hats ∧ W = 20 := 
by 
  sorry

end NUMINAMATH_GPT_wooden_toy_price_l1359_135983


namespace NUMINAMATH_GPT_girls_ratio_correct_l1359_135997

-- Define the number of total attendees
def total_attendees : ℕ := 100

-- Define the percentage of faculty and staff
def faculty_staff_percentage : ℕ := 10

-- Define the number of boys among the students
def number_of_boys : ℕ := 30

-- Define the function to calculate the number of faculty and staff
def faculty_staff (total_attendees faculty_staff_percentage: ℕ) : ℕ :=
  (faculty_staff_percentage * total_attendees) / 100

-- Define the function to calculate the number of students
def number_of_students (total_attendees faculty_staff_percentage: ℕ) : ℕ :=
  total_attendees - faculty_staff total_attendees faculty_staff_percentage

-- Define the function to calculate the number of girls
def number_of_girls (total_attendees faculty_staff_percentage number_of_boys: ℕ) : ℕ :=
  number_of_students total_attendees faculty_staff_percentage - number_of_boys

-- Define the function to calculate the ratio of girls to the remaining attendees
def ratio_girls_to_attendees (total_attendees faculty_staff_percentage number_of_boys: ℕ) : ℚ :=
  (number_of_girls total_attendees faculty_staff_percentage number_of_boys) / 
  (number_of_students total_attendees faculty_staff_percentage)

-- The theorem statement that needs to be proven (no proof required)
theorem girls_ratio_correct : ratio_girls_to_attendees total_attendees faculty_staff_percentage number_of_boys = 2 / 3 := 
by 
  -- The proof is skipped.
  sorry

end NUMINAMATH_GPT_girls_ratio_correct_l1359_135997


namespace NUMINAMATH_GPT_discount_percentage_is_10_l1359_135919

-- Definitions of the conditions directly translated
def CP (MP : ℝ) : ℝ := 0.7 * MP
def GainPercent : ℝ := 0.2857142857142857
def SP (MP : ℝ) : ℝ := CP MP * (1 + GainPercent)

-- Using the alternative expression for selling price involving discount percentage
def DiscountSP (MP : ℝ) (D : ℝ) : ℝ := MP * (1 - D)

-- The theorem to prove the discount percentage is 10%
theorem discount_percentage_is_10 (MP : ℝ) : ∃ D : ℝ, DiscountSP MP D = SP MP ∧ D = 0.1 := 
by
  use 0.1
  sorry

end NUMINAMATH_GPT_discount_percentage_is_10_l1359_135919


namespace NUMINAMATH_GPT_gain_per_year_is_120_l1359_135973

def principal := 6000
def rate_borrow := 4
def rate_lend := 6
def time := 2

def simple_interest (P R T : Nat) : Nat := P * R * T / 100

def interest_earned := simple_interest principal rate_lend time
def interest_paid := simple_interest principal rate_borrow time
def gain_in_2_years := interest_earned - interest_paid
def gain_per_year := gain_in_2_years / 2

theorem gain_per_year_is_120 : gain_per_year = 120 :=
by
  sorry

end NUMINAMATH_GPT_gain_per_year_is_120_l1359_135973


namespace NUMINAMATH_GPT_smallest_b_l1359_135927

noncomputable def Q (b : ℤ) (x : ℤ) : ℤ := sorry -- Q is a polynomial, will be defined in proof

theorem smallest_b (b : ℤ) 
  (h1 : b > 0) 
  (h2 : ∀ x, x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 → Q b x = b) 
  (h3 : ∀ x, x = 1 ∨ x = 3 ∨ x = 5 ∨ x = 7 → Q b x = -b) 
  : b = 315 := sorry

end NUMINAMATH_GPT_smallest_b_l1359_135927


namespace NUMINAMATH_GPT_f_increasing_f_t_range_l1359_135938

noncomputable def f : Real → Real :=
  sorry

axiom f_prop1 : f 2 = 1
axiom f_prop2 : ∀ x, x > 1 → f x > 0
axiom f_prop3 : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

theorem f_increasing (x1 x2 : Real) (hx1 : x1 > 0) (hx2 : x2 > 0) (h : x1 < x2) : f x1 < f x2 := by
  sorry

theorem f_t_range (t : Real) (ht : t > 0) (ht3 : t - 3 > 0) (hf : f t + f (t - 3) ≤ 2) : 3 < t ∧ t ≤ 4 := by
  sorry

end NUMINAMATH_GPT_f_increasing_f_t_range_l1359_135938


namespace NUMINAMATH_GPT_four_fours_expressions_l1359_135998

theorem four_fours_expressions :
  (4 * 4 + 4) / 4 = 5 ∧
  4 + (4 + 4) / 2 = 6 ∧
  4 + 4 - 4 / 4 = 7 ∧
  4 + 4 + 4 - 4 = 8 ∧
  4 + 4 + 4 / 4 = 9 :=
by
  sorry

end NUMINAMATH_GPT_four_fours_expressions_l1359_135998


namespace NUMINAMATH_GPT_ordered_pairs_count_l1359_135904

theorem ordered_pairs_count :
  ∃ (p : Finset (ℕ × ℕ)), (∀ (a b : ℕ), (a, b) ∈ p → a * b + 45 = 10 * Nat.lcm a b + 18 * Nat.gcd a b) ∧
  p.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_ordered_pairs_count_l1359_135904


namespace NUMINAMATH_GPT_total_weight_all_bags_sold_l1359_135986

theorem total_weight_all_bags_sold (morning_potatoes afternoon_potatoes morning_onions afternoon_onions morning_carrots afternoon_carrots : ℕ)
  (weight_potatoes weight_onions weight_carrots total_weight : ℕ)
  (h_morning_potatoes : morning_potatoes = 29)
  (h_afternoon_potatoes : afternoon_potatoes = 17)
  (h_morning_onions : morning_onions = 15)
  (h_afternoon_onions : afternoon_onions = 22)
  (h_morning_carrots : morning_carrots = 12)
  (h_afternoon_carrots : afternoon_carrots = 9)
  (h_weight_potatoes : weight_potatoes = 7)
  (h_weight_onions : weight_onions = 5)
  (h_weight_carrots : weight_carrots = 4)
  (h_total_weight : total_weight = 591) :
  morning_potatoes + afternoon_potatoes * weight_potatoes +
  morning_onions + afternoon_onions * weight_onions +
  morning_carrots + afternoon_carrots * weight_carrots = total_weight :=
by {
  sorry
}

end NUMINAMATH_GPT_total_weight_all_bags_sold_l1359_135986


namespace NUMINAMATH_GPT_angle_quadrant_l1359_135959

def same_terminal_side (θ α : ℝ) (k : ℤ) : Prop :=
  θ = α + 360 * k

def in_first_quadrant (α : ℝ) : Prop :=
  0 < α ∧ α < 90

theorem angle_quadrant (θ : ℝ) (k : ℤ) (h : same_terminal_side θ 12 k) : in_first_quadrant 12 :=
  by
    sorry

end NUMINAMATH_GPT_angle_quadrant_l1359_135959


namespace NUMINAMATH_GPT_total_nails_l1359_135920

-- Definitions based on the conditions
def Violet_nails : ℕ := 27
def Tickletoe_nails : ℕ := (27 - 3) / 2

-- Theorem to prove the total number of nails
theorem total_nails : Violet_nails + Tickletoe_nails = 39 := by
  sorry

end NUMINAMATH_GPT_total_nails_l1359_135920


namespace NUMINAMATH_GPT_initial_money_l1359_135907

theorem initial_money (spent allowance total initial : ℕ) 
  (h1 : spent = 2) 
  (h2 : allowance = 26) 
  (h3 : total = 29) 
  (h4 : initial - spent + allowance = total) : 
  initial = 5 := 
by 
  sorry

end NUMINAMATH_GPT_initial_money_l1359_135907


namespace NUMINAMATH_GPT_perimeter_of_square_is_64_l1359_135999

noncomputable def side_length_of_square (s : ℝ) :=
  let rect_height := s
  let rect_width := s / 4
  let perimeter_of_rectangle := 2 * (rect_height + rect_width)
  perimeter_of_rectangle = 40

theorem perimeter_of_square_is_64 (s : ℝ) (h1 : side_length_of_square s) : 4 * s = 64 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_is_64_l1359_135999


namespace NUMINAMATH_GPT_find_other_endpoint_l1359_135957

theorem find_other_endpoint 
    (Mx My : ℝ) (x1 y1 : ℝ) 
    (hx_Mx : Mx = 3) (hy_My : My = 1)
    (hx1 : x1 = 7) (hy1 : y1 = -3) : 
    ∃ (x2 y2 : ℝ), Mx = (x1 + x2) / 2 ∧ My = (y1 + y2) / 2 ∧ x2 = -1 ∧ y2 = 5 :=
by
    sorry

end NUMINAMATH_GPT_find_other_endpoint_l1359_135957


namespace NUMINAMATH_GPT_A_work_days_l1359_135958

theorem A_work_days (x : ℝ) :
  (1 / x + 1 / 6 + 1 / 12 = 7 / 24) → x = 24 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_A_work_days_l1359_135958
