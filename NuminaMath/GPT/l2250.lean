import Mathlib

namespace complete_the_square_problem_l2250_225027

theorem complete_the_square_problem :
  ∃ r s : ℝ, (r = -2) ∧ (s = 9) ∧ (r + s = 7) ∧ ∀ x : ℝ, 15 * x ^ 2 - 60 * x - 135 = 0 ↔ (x + r) ^ 2 = s := 
by
  sorry

end complete_the_square_problem_l2250_225027


namespace solve_eq1_solve_eq2_l2250_225071

theorem solve_eq1 : (∃ x : ℚ, (5 * x - 1) / 4 = (3 * x + 1) / 2 - (2 - x) / 3) ↔ x = -1 / 7 :=
sorry

theorem solve_eq2 : (∃ x : ℚ, (3 * x + 2) / 2 - 1 = (2 * x - 1) / 4 - (2 * x + 1) / 5) ↔ x = -9 / 28 :=
sorry

end solve_eq1_solve_eq2_l2250_225071


namespace smallest_square_value_l2250_225023

theorem smallest_square_value (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (r s : ℕ) (hr : 15 * a + 16 * b = r^2) (hs : 16 * a - 15 * b = s^2) :
  min (r^2) (s^2) = 481^2 :=
  sorry

end smallest_square_value_l2250_225023


namespace solve_y_from_expression_l2250_225021

-- Define the conditions
def given_conditions := (784 = 28^2) ∧ (49 = 7^2)

-- Define the equivalency to prove based on the given conditions
theorem solve_y_from_expression (h : given_conditions) : 784 + 2 * 28 * 7 + 49 = 1225 := by
  sorry

end solve_y_from_expression_l2250_225021


namespace binary_multiplication_correct_l2250_225093

theorem binary_multiplication_correct :
  (0b1101 : ℕ) * (0b1011 : ℕ) = (0b10011011 : ℕ) :=
by
  sorry

end binary_multiplication_correct_l2250_225093


namespace find_value_of_a4_plus_a5_l2250_225092

variables {S_n : ℕ → ℕ} {a_n : ℕ → ℕ} {d : ℤ} 

-- Conditions
def arithmetic_sequence_sum (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) (d : ℤ) : Prop :=
∀ n : ℕ, S_n n = n * a_n 1 + (n * (n - 1) / 2) * d

def a_3_S_3_condition (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) : Prop := 
a_n 3 = 3 ∧ S_n 3 = 3

-- Question
theorem find_value_of_a4_plus_a5 (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (d : ℤ):
  arithmetic_sequence_sum S_n a_n d →
  a_3_S_3_condition a_n S_n →
  a_n 4 + a_n 5 = 12 :=
by
  sorry

end find_value_of_a4_plus_a5_l2250_225092


namespace number_of_small_jars_l2250_225019

theorem number_of_small_jars (S L : ℕ) (h1 : S + L = 100) (h2 : 3 * S + 5 * L = 376) : S = 62 :=
by
  sorry

end number_of_small_jars_l2250_225019


namespace find_t_value_l2250_225068

theorem find_t_value (t : ℝ) (a b : ℝ × ℝ) (h₁ : a = (t, 1)) (h₂ : b = (1, 2)) 
  (h₃ : (a.1 + b.1)^2 + (a.2 + b.2)^2 = a.1^2 + a.2^2 + b.1^2 + b.2^2) : 
  t = -2 :=
by 
  sorry

end find_t_value_l2250_225068


namespace vectors_are_perpendicular_l2250_225072

def vector_a : ℝ × ℝ := (-5, 6)
def vector_b : ℝ × ℝ := (6, 5)

theorem vectors_are_perpendicular :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = 0 :=
by
  sorry

end vectors_are_perpendicular_l2250_225072


namespace workshop_average_salary_l2250_225011

theorem workshop_average_salary :
  let technicians := 8
  let rest := 24 - technicians
  let avg_technician_salary := 12000
  let avg_rest_salary := 6000
  let total_workers := 24
  let total_staff_salary := (technicians * avg_technician_salary) + (rest * avg_rest_salary)
  let A := total_staff_salary / total_workers
  A = 8000 :=
by
  -- Definitions according to given conditions
  let technicians := 8
  let rest := 24 - technicians
  let avg_technician_salary := 12000
  let avg_rest_salary := 6000
  let total_workers := 24
  let total_staff_salary := (technicians * avg_technician_salary) + (rest * avg_rest_salary)
  let A := total_staff_salary / total_workers
  -- We need to show that A = 8000
  show A = 8000
  sorry

end workshop_average_salary_l2250_225011


namespace value_of_D_l2250_225051

theorem value_of_D (D : ℤ) (h : 80 - (5 - (6 + 2 * (7 - 8 - D))) = 89) : D = -5 :=
by sorry

end value_of_D_l2250_225051


namespace necessary_for_A_l2250_225088

-- Define the sets A, B, C as non-empty sets
variables {α : Type*} (A B C : Set α)
-- Non-empty sets
axiom non_empty_A : ∃ x, x ∈ A
axiom non_empty_B : ∃ x, x ∈ B
axiom non_empty_C : ∃ x, x ∈ C

-- Conditions
axiom union_condition : A ∪ B = C
axiom subset_condition : ¬ (B ⊆ A)

-- Statement to prove
theorem necessary_for_A (x : α) : (x ∈ C → x ∈ A) ∧ ¬(x ∈ C ↔ x ∈ A) :=
sorry

end necessary_for_A_l2250_225088


namespace possible_values_of_c_l2250_225008

-- Definition of c(S) based on the problem conditions
def c (S : String) (m : ℕ) : ℕ := sorry

-- Condition: m > 1
variable {m : ℕ} (hm : m > 1)

-- Goal: To prove the possible values that c(S) can take
theorem possible_values_of_c (S : String) : ∃ n : ℕ, c S m = 0 ∨ c S m = 2^n :=
sorry

end possible_values_of_c_l2250_225008


namespace total_credit_hours_l2250_225098

def max_courses := 40
def max_courses_per_semester := 5
def max_courses_per_semester_credit := 3
def max_additional_courses_last_semester := 2
def max_additional_course_credit := 4
def sid_courses_multiplier := 4
def sid_additional_courses_multiplier := 2

theorem total_credit_hours (total_max_courses : Nat) 
                           (avg_max_courses_per_semester : Nat) 
                           (max_course_credit : Nat) 
                           (extra_max_courses_last_sem : Nat) 
                           (extra_max_course_credit : Nat) 
                           (sid_courses_mult : Nat) 
                           (sid_extra_courses_mult : Nat) 
                           (max_total_courses : total_max_courses = max_courses)
                           (max_avg_courses_per_semester : avg_max_courses_per_semester = max_courses_per_semester)
                           (max_course_credit_def : max_course_credit = max_courses_per_semester_credit)
                           (extra_max_courses_last_sem_def : extra_max_courses_last_sem = max_additional_courses_last_semester)
                           (extra_max_courses_credit_def : extra_max_course_credit = max_additional_course_credit)
                           (sid_courses_mult_def : sid_courses_mult = sid_courses_multiplier)
                           (sid_extra_courses_mult_def : sid_extra_courses_mult = sid_additional_courses_multiplier) : 
  total_max_courses * max_course_credit + extra_max_courses_last_sem * extra_max_course_credit + 
  (sid_courses_mult * total_max_courses - sid_extra_courses_mult * extra_max_courses_last_sem) * max_course_credit + sid_extra_courses_mult * extra_max_courses_last_sem * extra_max_course_credit = 606 := 
  by 
    sorry

end total_credit_hours_l2250_225098


namespace determine_other_number_l2250_225086

theorem determine_other_number (a b : ℤ) (h₁ : 3 * a + 4 * b = 161) (h₂ : a = 17 ∨ b = 17) : 
(a = 31 ∨ b = 31) :=
by
  sorry

end determine_other_number_l2250_225086


namespace square_side_length_leq_half_l2250_225048

theorem square_side_length_leq_half
    (l : ℝ)
    (h_square_inside_unit : l ≤ 1)
    (h_no_center_contain : ∀ (x y : ℝ), x^2 + y^2 > (l/2)^2 → (0.5 ≤ x ∨ 0.5 ≤ y)) :
    l ≤ 0.5 := 
sorry

end square_side_length_leq_half_l2250_225048


namespace cylinder_volume_ratio_l2250_225084

theorem cylinder_volume_ratio
  (h : ℝ)
  (r1 : ℝ)
  (r3 : ℝ := 3 * r1)
  (V1 : ℝ := 40) :
  let V2 := π * r3^2 * h
  (π * r1^2 * h = V1) → 
  V2 = 360 := by
{
  sorry
}

end cylinder_volume_ratio_l2250_225084


namespace rectangle_to_square_l2250_225075

-- Definitions based on conditions
def rectangle_width : ℕ := 12
def rectangle_height : ℕ := 3
def area : ℕ := rectangle_width * rectangle_height
def parts : ℕ := 3
def part_area : ℕ := area / parts
def square_side : ℕ := Nat.sqrt area

-- Theorem to restate the problem
theorem rectangle_to_square : (area = 36) ∧ (part_area = 12) ∧ (square_side = 6) ∧
  (rectangle_width / parts = 4) ∧ (rectangle_height = 3) ∧ 
  ((rectangle_width / parts * parts) = rectangle_width) ∧ (parts * rectangle_height = square_side ^ 2) := by
  -- Placeholder for proof
  sorry

end rectangle_to_square_l2250_225075


namespace probability_all_letters_SUPERBLOOM_l2250_225038

noncomputable def choose (n k : ℕ) : ℕ := sorry

theorem probability_all_letters_SUPERBLOOM :
  let P1 := 1 / (choose 6 3)
  let P2 := 9 / (choose 8 5)
  let P3 := 1 / (choose 5 4)
  P1 * P2 * P3 = 9 / 1120 :=
by
  sorry

end probability_all_letters_SUPERBLOOM_l2250_225038


namespace solve_a1_solve_a2_l2250_225077

noncomputable def initial_volume := 1  -- in m^3
noncomputable def initial_pressure := 10^5  -- in Pa
noncomputable def initial_temperature := 300  -- in K

theorem solve_a1 (a1 : ℝ) : a1 = -10^5 :=
  sorry

theorem solve_a2 (a2 : ℝ) : a2 = -1.4 * 10^5 :=
  sorry

end solve_a1_solve_a2_l2250_225077


namespace math_problem_l2250_225061

theorem math_problem 
  (a b c : ℕ) 
  (h_primea : Nat.Prime a)
  (h_posa : 0 < a)
  (h_posb : 0 < b)
  (h_posc : 0 < c)
  (h_eq : a^2 + b^2 = c^2) :
  (b % 2 ≠ c % 2) ∧ (∃ k, 2 * (a + b + 1) = k^2) := 
sorry

end math_problem_l2250_225061


namespace find_first_term_l2250_225058

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

variable (a1 a3 a9 d : ℤ)

-- Given conditions
axiom h1 : arithmetic_seq a1 d 2 = 30
axiom h2 : arithmetic_seq a1 d 8 = 60

theorem find_first_term : a1 = 20 :=
by
  -- mathematical proof steps here
  sorry

end find_first_term_l2250_225058


namespace value_calculation_l2250_225020

-- Define the given number
def given_number : ℝ := 93.75

-- Define the percentages as ratios
def forty_percent : ℝ := 0.4
def sixteen_percent : ℝ := 0.16

-- Calculate the intermediate value for 40% of the given number
def intermediate_value := forty_percent * given_number

-- Final value calculation for 16% of the intermediate value
def final_value := sixteen_percent * intermediate_value

-- The theorem to prove
theorem value_calculation : final_value = 6 := by
  -- Expanding definitions to substitute and simplify
  unfold final_value intermediate_value forty_percent sixteen_percent given_number
  -- Proving the correctness by calculating
  sorry

end value_calculation_l2250_225020


namespace number_of_points_max_45_lines_l2250_225012

theorem number_of_points_max_45_lines (n : ℕ) (h : n * (n - 1) / 2 ≤ 45) : n = 10 := 
  sorry

end number_of_points_max_45_lines_l2250_225012


namespace find_m_and_n_l2250_225024

theorem find_m_and_n (m n : ℝ) 
  (h1 : m + n = 6) 
  (h2 : 2 * m - n = 6) : 
  m = 4 ∧ n = 2 := 
by 
  sorry

end find_m_and_n_l2250_225024


namespace f_at_1_l2250_225081

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x^2 + (5 : ℝ) * x
  else if x = 2 then 6
  else  - (x^2 + (5 : ℝ) * x)

theorem f_at_1 : f 1 = 4 :=
by {
  sorry
}

end f_at_1_l2250_225081


namespace solve_for_x_l2250_225090

theorem solve_for_x (x : ℝ) : (2 / 7) * (1 / 8) * x = 14 ↔ x = 392 :=
by {
  sorry
}

end solve_for_x_l2250_225090


namespace range_of_f_l2250_225069

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sqrt (5 + 4 * Real.cos x))

theorem range_of_f :
  Set.range f = Set.Icc (-1/2 : ℝ) (1/2 : ℝ) := 
sorry

end range_of_f_l2250_225069


namespace find_a4_l2250_225065

open Nat

def seq (a : ℕ → ℝ) := (a 1 = 1) ∧ (∀ n : ℕ, a (n + 1) = (2 * a n) / (a n + 2))

theorem find_a4 (a : ℕ → ℝ) (h : seq a) : a 4 = 2 / 5 :=
  sorry

end find_a4_l2250_225065


namespace tank_fill_time_l2250_225035

/-- Given the rates at which pipes fill a tank, prove the total time to fill the tank using all three pipes. --/
theorem tank_fill_time (R_a R_b R_c : ℝ) (T : ℝ)
  (h1 : R_a = 1 / 35)
  (h2 : R_b = 2 * R_a)
  (h3 : R_c = 2 * R_b)
  (h4 : T = 5) :
  1 / (R_a + R_b + R_c) = T := by
  sorry

end tank_fill_time_l2250_225035


namespace staffing_battle_station_l2250_225074

-- Define the qualifications
def num_assistant_engineer := 3
def num_maintenance_1 := 4
def num_maintenance_2 := 4
def num_field_technician := 5
def num_radio_specialist := 5

-- Prove the total number of ways to fill the positions
theorem staffing_battle_station : 
  num_assistant_engineer * num_maintenance_1 * num_maintenance_2 * num_field_technician * num_radio_specialist = 960 := by
  sorry

end staffing_battle_station_l2250_225074


namespace bishop_safe_squares_l2250_225067

def chessboard_size : ℕ := 64
def total_squares_removed_king : ℕ := chessboard_size - 1
def threat_squares : ℕ := 7

theorem bishop_safe_squares : total_squares_removed_king - threat_squares = 30 :=
by
  sorry

end bishop_safe_squares_l2250_225067


namespace parallelogram_area_correct_l2250_225032

def parallelogram_area (b h : ℝ) : ℝ := b * h

theorem parallelogram_area_correct :
  parallelogram_area 15 5 = 75 :=
by
  sorry

end parallelogram_area_correct_l2250_225032


namespace sqrt_225_eq_15_l2250_225050

theorem sqrt_225_eq_15 : Real.sqrt 225 = 15 :=
sorry

end sqrt_225_eq_15_l2250_225050


namespace input_language_is_input_l2250_225018

def is_print_statement (statement : String) : Prop := 
  statement = "PRINT"

def is_input_statement (statement : String) : Prop := 
  statement = "INPUT"

def is_conditional_statement (statement : String) : Prop := 
  statement = "IF"

theorem input_language_is_input :
  is_input_statement "INPUT" := 
by
  -- Here we need to show "INPUT" is an input statement
  sorry

end input_language_is_input_l2250_225018


namespace price_of_third_variety_l2250_225004

theorem price_of_third_variety 
    (price1 price2 price3 : ℝ)
    (mix_ratio1 mix_ratio2 mix_ratio3 : ℝ)
    (mixture_price : ℝ)
    (h1 : price1 = 126)
    (h2 : price2 = 135)
    (h3 : mix_ratio1 = 1)
    (h4 : mix_ratio2 = 1)
    (h5 : mix_ratio3 = 2)
    (h6 : mixture_price = 153) :
    price3 = 175.5 :=
by
  sorry

end price_of_third_variety_l2250_225004


namespace retirement_amount_l2250_225049

-- Define the principal amount P
def P : ℝ := 750000

-- Define the annual interest rate r
def r : ℝ := 0.08

-- Define the time period in years t
def t : ℝ := 12

-- Define the accumulated amount A
def A : ℝ := P * (1 + r * t)

-- Prove that the accumulated amount A equals 1470000
theorem retirement_amount : A = 1470000 := by
  -- The proof will involve calculating the compound interest
  sorry

end retirement_amount_l2250_225049


namespace mod_remainder_of_expression_l2250_225062

theorem mod_remainder_of_expression : (7 * 10^20 + 2^20) % 9 = 2 := by
  sorry

end mod_remainder_of_expression_l2250_225062


namespace part_a_part_b_l2250_225064

def P (m n : ℕ) : ℕ := m^2003 * n^2017 - m^2017 * n^2003

theorem part_a (m n : ℕ) : P m n % 24 = 0 := 
by sorry

theorem part_b : ∃ (m n : ℕ), P m n % 7 ≠ 0 :=
by sorry

end part_a_part_b_l2250_225064


namespace malcolm_initial_white_lights_l2250_225033

-- Definitions based on the conditions
def red_lights : Nat := 12
def blue_lights : Nat := 3 * red_lights
def green_lights : Nat := 6
def total_colored_lights := red_lights + blue_lights + green_lights
def lights_left_to_buy : Nat := 5
def initially_white_lights := total_colored_lights + lights_left_to_buy

-- Proof statement
theorem malcolm_initial_white_lights : initially_white_lights = 59 := by
  sorry

end malcolm_initial_white_lights_l2250_225033


namespace solve_for_m_l2250_225082

theorem solve_for_m (m : ℝ) (h : m + (m + 2) + (m + 4) = 24) : m = 6 :=
by {
  sorry
}

end solve_for_m_l2250_225082


namespace annual_earning_difference_l2250_225028

def old_hourly_wage := 16
def old_weekly_hours := 25
def new_hourly_wage := 20
def new_weekly_hours := 40
def weeks_per_year := 52

def old_weekly_earnings := old_hourly_wage * old_weekly_hours
def new_weekly_earnings := new_hourly_wage * new_weekly_hours

def old_annual_earnings := old_weekly_earnings * weeks_per_year
def new_annual_earnings := new_weekly_earnings * weeks_per_year

theorem annual_earning_difference:
  new_annual_earnings - old_annual_earnings = 20800 := by
  sorry

end annual_earning_difference_l2250_225028


namespace maximize_revenue_l2250_225030

-- Defining the revenue function
def revenue (p : ℝ) : ℝ := 200 * p - 4 * p^2

-- Defining the maximum price constraint
def price_constraint (p : ℝ) : Prop := p ≤ 40

-- Statement to be proven
theorem maximize_revenue : ∃ (p : ℝ), price_constraint p ∧ revenue p = 2500 ∧ (∀ q : ℝ, price_constraint q → revenue q ≤ revenue p) :=
sorry

end maximize_revenue_l2250_225030


namespace find_height_of_cuboid_l2250_225096

variable (A : ℝ) (V : ℝ) (h : ℝ)

theorem find_height_of_cuboid (h_eq : h = V / A) (A_eq : A = 36) (V_eq : V = 252) : h = 7 :=
by
  sorry

end find_height_of_cuboid_l2250_225096


namespace cost_of_apples_l2250_225054

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h_total : total_cost = 42)
  (h_bananas : cost_bananas = 12)
  (h_bread : cost_bread = 9)
  (h_milk : cost_milk = 7)
  (h_combined : cost_apples = total_cost - (cost_bananas + cost_bread + cost_milk)) : 
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l2250_225054


namespace quadratic_equation_root_and_coef_l2250_225002

theorem quadratic_equation_root_and_coef (k x : ℤ) (h1 : x^2 - 3 * x + k = 0)
  (root4 : x = 4) : (x = 4 ∧ k = -4 ∧ ∀ y, y ≠ 4 → y^2 - 3 * y + k = 0 → y = -1) :=
by {
  sorry
}

end quadratic_equation_root_and_coef_l2250_225002


namespace find_a_parallel_lines_l2250_225001

theorem find_a_parallel_lines (a : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, x * a + 2 * y + 2 = 0 ↔ 3 * x - y - 2 = k * (x * a + 2 * y + 2)) ↔ a = -6 := by
  sorry

end find_a_parallel_lines_l2250_225001


namespace combined_age_l2250_225046

theorem combined_age (H : ℕ) (Ryanne : ℕ) (Jamison : ℕ) 
  (h1 : Ryanne = H + 7) 
  (h2 : H + Ryanne = 15) 
  (h3 : Jamison = 2 * H) : 
  H + Ryanne + Jamison = 23 := 
by 
  sorry

end combined_age_l2250_225046


namespace parrot_consumption_l2250_225079

theorem parrot_consumption :
  ∀ (parakeet_daily : ℕ) (finch_daily : ℕ) (num_parakeets : ℕ) (num_parrots : ℕ) (num_finches : ℕ) (weekly_birdseed : ℕ),
    parakeet_daily = 2 →
    finch_daily = parakeet_daily / 2 →
    num_parakeets = 3 →
    num_parrots = 2 →
    num_finches = 4 →
    weekly_birdseed = 266 →
    14 = (weekly_birdseed - ((num_parakeets * parakeet_daily + num_finches * finch_daily) * 7)) / num_parrots / 7 :=
by
  intros parakeet_daily finch_daily num_parakeets num_parrots num_finches weekly_birdseed
  intros hp1 hp2 hp3 hp4 hp5 hp6
  sorry

end parrot_consumption_l2250_225079


namespace hannah_total_payment_l2250_225014

def washing_machine_cost : ℝ := 100
def dryer_cost : ℝ := washing_machine_cost - 30
def total_cost_before_discount : ℝ := washing_machine_cost + dryer_cost
def discount : ℝ := 0.10
def total_cost_after_discount : ℝ := total_cost_before_discount * (1 - discount)

theorem hannah_total_payment : total_cost_after_discount = 153 := by
  sorry

end hannah_total_payment_l2250_225014


namespace remainder_of_3456_div_97_l2250_225007

theorem remainder_of_3456_div_97 :
  3456 % 97 = 61 :=
by
  sorry

end remainder_of_3456_div_97_l2250_225007


namespace bernardo_receives_l2250_225044

theorem bernardo_receives :
  let amount_distributed (n : ℕ) : ℕ := (n * (n + 1)) / 2
  let is_valid (n : ℕ) : Prop := amount_distributed n ≤ 1000
  let bernardo_amount (k : ℕ) : ℕ := (k * (2 + (k - 1) * 3)) / 2
  ∃ k : ℕ, is_valid (15 * 3) ∧ bernardo_amount 15 = 345 :=
sorry

end bernardo_receives_l2250_225044


namespace sixty_first_batch_is_1211_l2250_225053

-- Definitions based on conditions
def total_bags : ℕ := 3000
def total_batches : ℕ := 150
def first_batch_number : ℕ := 11

-- Define the calculation of the 61st batch number
def batch_interval : ℕ := total_bags / total_batches
def sixty_first_batch_number : ℕ := first_batch_number + 60 * batch_interval

-- The statement of the proof
theorem sixty_first_batch_is_1211 : sixty_first_batch_number = 1211 := by
  sorry

end sixty_first_batch_is_1211_l2250_225053


namespace find_ordered_pair_l2250_225055

theorem find_ordered_pair (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (hroots : ∀ x, x^2 + c * x + d = (x - c) * (x - d)) : 
  (c, d) = (1, -2) :=
sorry

end find_ordered_pair_l2250_225055


namespace unit_circle_chords_l2250_225041

theorem unit_circle_chords (
    s t u v : ℝ
) (hs : s = 1) (ht : t = 1) (hu : u = 2) (hv : v = 3) :
    (v - u = 1) ∧ (v * u = 6) ∧ (v^2 - u^2 = 5) :=
by
  have h1 : v - u = 1 := by rw [hv, hu]; norm_num
  have h2 : v * u = 6 := by rw [hv, hu]; norm_num
  have h3 : v^2 - u^2 = 5 := by rw [hv, hu]; norm_num
  exact ⟨h1, h2, h3⟩

end unit_circle_chords_l2250_225041


namespace perfect_square_term_l2250_225052

def seq (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 4 * seq (n - 1) - seq (n - 2)

theorem perfect_square_term : ∀ n, (∃ k, seq n = k * k) ↔ n = 0 := by
  sorry

end perfect_square_term_l2250_225052


namespace number_square_of_digits_l2250_225080

theorem number_square_of_digits (x y : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) (h3 : 0 ≤ y) (h4 : y ≤ 9) :
  ∃ n : ℕ, (∃ (k : ℕ), (1001 * x + 110 * y) = k^2) ↔ (x = 7 ∧ y = 4) :=
by
  sorry

end number_square_of_digits_l2250_225080


namespace consecutive_even_sum_l2250_225005

theorem consecutive_even_sum (n : ℤ) (h : (n - 2) + (n + 2) = 156) : n = 78 :=
by
  sorry

end consecutive_even_sum_l2250_225005


namespace weight_in_one_hand_l2250_225047

theorem weight_in_one_hand (total_weight : ℕ) (h : total_weight = 16) : total_weight / 2 = 8 :=
by
  sorry

end weight_in_one_hand_l2250_225047


namespace num_ways_to_turn_off_lights_l2250_225000

-- Let's define our problem in terms of the conditions given
-- Define the total number of lights
def total_lights : ℕ := 12

-- Define that we need to turn off 3 lights
def lights_to_turn_off : ℕ := 3

-- Define that we have 10 possible candidates for being turned off 
def candidates := total_lights - 2

-- Define the gap consumption statement that effectively reduce choices to 7 lights
def effective_choices := candidates - lights_to_turn_off

-- Define the combination formula for the number of ways to turn off the lights
def num_ways := Nat.choose effective_choices lights_to_turn_off

-- Final statement to prove
theorem num_ways_to_turn_off_lights : num_ways = Nat.choose 7 3 :=
by
  sorry

end num_ways_to_turn_off_lights_l2250_225000


namespace find_multiplier_l2250_225045

theorem find_multiplier (x : ℝ) (h : (9 / 6) * x = 18) : x = 12 := sorry

end find_multiplier_l2250_225045


namespace fence_perimeter_l2250_225025

theorem fence_perimeter 
  (N : ℕ) (w : ℝ) (g : ℝ) 
  (square_posts : N = 36) 
  (post_width : w = 0.5) 
  (gap_length : g = 8) :
  4 * ((N / 4 - 1) * g + (N / 4) * w) = 274 :=
by
  sorry

end fence_perimeter_l2250_225025


namespace official_exchange_rate_l2250_225091

theorem official_exchange_rate (E : ℝ)
  (h1 : 70 = 10 * (7 / 5) * E) :
  E = 5 :=
by
  sorry

end official_exchange_rate_l2250_225091


namespace metallic_weight_problem_l2250_225017

variables {m1 m2 m3 m4 : ℝ}

theorem metallic_weight_problem
  (h_total : m1 + m2 + m3 + m4 = 35)
  (h1 : m1 = 1.5 * m2)
  (h2 : m2 = (3/4) * m3)
  (h3 : m3 = (5/6) * m4) :
  m4 = 105 / 13 :=
sorry

end metallic_weight_problem_l2250_225017


namespace no_roots_one_and_neg_one_l2250_225010

theorem no_roots_one_and_neg_one (a b : ℝ) : ¬ ((1 + a + b = 0) ∧ (-1 + a + b = 0)) :=
by
  sorry

end no_roots_one_and_neg_one_l2250_225010


namespace BANANA_arrangement_l2250_225070

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l2250_225070


namespace find_AC_length_l2250_225015

theorem find_AC_length (AB BC CD DA : ℕ) 
  (hAB : AB = 10) (hBC : BC = 9) (hCD : CD = 19) (hDA : DA = 5) : 
  14 < AC ∧ AC < 19 → AC = 15 := 
by
  sorry

end find_AC_length_l2250_225015


namespace benny_lunch_cost_l2250_225095

theorem benny_lunch_cost :
  let person := 3;
  let cost_per_lunch := 8;
  let total_cost := person * cost_per_lunch;
  total_cost = 24 :=
by
  let person := 3;
  let cost_per_lunch := 8;
  let total_cost := person * cost_per_lunch;
  have h : total_cost = 24 := by
    sorry
  exact h

end benny_lunch_cost_l2250_225095


namespace integer_pairs_solution_l2250_225029

theorem integer_pairs_solution (a b : ℤ) : 
  (a - b - 1 ∣ a^2 + b^2 ∧ (a^2 + b^2) * 19 = (2 * a * b - 1) * 20) ↔
  (a, b) = (22, 16) ∨ (a, b) = (-16, -22) ∨ (a, b) = (8, 6) ∨ (a, b) = (-6, -8) :=
by 
  sorry

end integer_pairs_solution_l2250_225029


namespace scientific_notation_correct_l2250_225078

-- Define the given condition
def average_daily_users : ℝ := 2590000

-- The proof problem
theorem scientific_notation_correct :
  average_daily_users = 2.59 * 10^6 :=
sorry

end scientific_notation_correct_l2250_225078


namespace total_cups_needed_l2250_225037

-- Define the known conditions
def ratio_butter : ℕ := 2
def ratio_flour : ℕ := 3
def ratio_sugar : ℕ := 5
def total_sugar_in_cups : ℕ := 10

-- Define the parts-to-cups conversion
def cup_per_part := total_sugar_in_cups / ratio_sugar

-- Define the amounts of each ingredient in cups
def butter_in_cups := ratio_butter * cup_per_part
def flour_in_cups := ratio_flour * cup_per_part
def sugar_in_cups := ratio_sugar * cup_per_part

-- Define the total number of cups
def total_cups := butter_in_cups + flour_in_cups + sugar_in_cups

-- Theorem to prove
theorem total_cups_needed : total_cups = 20 := by
  sorry

end total_cups_needed_l2250_225037


namespace white_area_of_sign_l2250_225087

theorem white_area_of_sign :
  let total_area : ℕ := 6 * 18
  let black_area_C : ℕ := 11
  let black_area_A : ℕ := 10
  let black_area_F : ℕ := 12
  let black_area_E : ℕ := 9
  let total_black_area : ℕ := black_area_C + black_area_A + black_area_F + black_area_E
  let white_area : ℕ := total_area - total_black_area
  white_area = 66 := by
  sorry

end white_area_of_sign_l2250_225087


namespace David_Marks_in_Mathematics_are_85_l2250_225031

theorem David_Marks_in_Mathematics_are_85
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (h1 : english_marks = 86)
  (h2 : physics_marks = 92)
  (h3 : chemistry_marks = 87)
  (h4 : biology_marks = 95)
  (h5 : average_marks = 89)
  (h6 : num_subjects = 5) : 
  (86 + 92 + 87 + 95 + 85) / 5 = 89 :=
by sorry

end David_Marks_in_Mathematics_are_85_l2250_225031


namespace unique_functional_equation_solution_l2250_225083

theorem unique_functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end unique_functional_equation_solution_l2250_225083


namespace total_students_sampled_l2250_225097

theorem total_students_sampled (freq_ratio : ℕ → ℕ → ℕ) (second_group_freq : ℕ) 
  (ratio_condition : freq_ratio 2 1 = 2 ∧ freq_ratio 2 3 = 3) : 
  (6 + second_group_freq + 18) = 48 := 
by 
  sorry

end total_students_sampled_l2250_225097


namespace percent_increase_l2250_225022

/-- Problem statement: Given (1/2)x = 1, prove that the percentage increase from 1/2 to x is 300%. -/
theorem percent_increase (x : ℝ) (h : (1/2) * x = 1) : 
  ((x - (1/2)) / (1/2)) * 100 = 300 := 
by
  sorry

end percent_increase_l2250_225022


namespace solve_for_cubic_l2250_225003

theorem solve_for_cubic (x y : ℝ) (h₁ : x * (x + y) = 49) (h₂: y * (x + y) = 63) : (x + y)^3 = 448 * Real.sqrt 7 := 
sorry

end solve_for_cubic_l2250_225003


namespace rate_of_current_l2250_225073

theorem rate_of_current (c : ℝ) (h1 : ∀ d : ℝ, d / (3.9 - c) = 2 * (d / (3.9 + c))) : c = 1.3 :=
sorry

end rate_of_current_l2250_225073


namespace ratio_of_abc_l2250_225009

theorem ratio_of_abc {a b c : ℕ} (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) 
                     (h_ratio : ∃ x : ℕ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x)
                     (h_mean : (a + b + c) / 3 = 42) : 
  a = 28 := 
sorry

end ratio_of_abc_l2250_225009


namespace connected_paper_area_l2250_225057

def side_length := 30 -- side of each square paper in cm
def overlap_length := 7 -- overlap length in cm
def num_pieces := 6 -- number of paper pieces

def effective_length (side_length overlap_length : ℕ) := side_length - overlap_length
def total_connected_length (num_pieces : ℕ) (side_length overlap_length : ℕ) :=
  side_length + (num_pieces - 1) * (effective_length side_length overlap_length)

def width := side_length -- width of the connected paper is the side of each square piece of paper

def area (length width : ℕ) := length * width

theorem connected_paper_area : area (total_connected_length num_pieces side_length overlap_length) width = 4350 :=
by
  sorry

end connected_paper_area_l2250_225057


namespace p_arithmetic_square_root_l2250_225089

theorem p_arithmetic_square_root {p : ℕ} (hp : p ≠ 2) (a : ℤ) (ha : a ≠ 0) :
  (∃ x1 x2 : ℤ, x1^2 = a ∧ x2^2 = a ∧ x1 ≠ x2) ∨ ¬ (∃ x : ℤ, x^2 = a) :=
  sorry

end p_arithmetic_square_root_l2250_225089


namespace actors_per_group_l2250_225066

theorem actors_per_group (actors_per_hour : ℕ) (show_time_per_actor : ℕ) (total_show_time : ℕ)
  (h1 : show_time_per_actor = 15) (h2 : actors_per_hour = 20) (h3 : total_show_time = 60) :
  actors_per_hour * show_time_per_actor / total_show_time = 5 :=
by sorry

end actors_per_group_l2250_225066


namespace value_of_expr_l2250_225026

-- Definitions
def operation (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- The proof statement
theorem value_of_expr (a b : ℕ) (h₀ : operation a b = 100) : (a + b) + 6 = 11 := by
  sorry

end value_of_expr_l2250_225026


namespace find_a_if_even_function_l2250_225016

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * 2^x + 2^(-x)

-- Theorem statement
theorem find_a_if_even_function (a : ℝ) (h : is_even_function (f a)) : a = 1 :=
sorry

end find_a_if_even_function_l2250_225016


namespace Elle_practice_time_l2250_225013

variable (x : ℕ)

theorem Elle_practice_time : 
  (5 * x) + (3 * x) = 240 → x = 30 :=
by
  intro h
  sorry

end Elle_practice_time_l2250_225013


namespace triangle_right_triangle_l2250_225099

variable {A B C : Real}  -- Define the angles A, B, and C

theorem triangle_right_triangle (sin_A sin_B sin_C : Real)
  (h : sin_A^2 + sin_B^2 = sin_C^2) 
  (triangle_cond : A + B + C = 180) : 
  (A = 90) ∨ (B = 90) ∨ (C = 90) := 
  sorry

end triangle_right_triangle_l2250_225099


namespace factor_expression_l2250_225034

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l2250_225034


namespace avoid_loss_maximize_profit_max_profit_per_unit_l2250_225042

-- Definitions of the functions as per problem conditions
noncomputable def C (x : ℝ) : ℝ := 2 + x
noncomputable def R (x : ℝ) : ℝ := if x ≤ 4 then 4 * x - (1 / 2) * x^2 - (1 / 2) else 7.5
noncomputable def L (x : ℝ) : ℝ := R x - C x

-- Proof statements

-- 1. Range to avoid loss
theorem avoid_loss (x : ℝ) : 1 ≤ x ∧ x ≤ 5.5 ↔ L x ≥ 0 :=
by
  sorry

-- 2. Production to maximize profit
theorem maximize_profit (x : ℝ) : x = 3 ↔ ∀ y, L y ≤ L 3 :=
by
  sorry

-- 3. Maximum profit per unit selling price
theorem max_profit_per_unit (x : ℝ) : x = 3 ↔ (R 3 / 3 = 2.33) :=
by
  sorry

end avoid_loss_maximize_profit_max_profit_per_unit_l2250_225042


namespace sqrt_polynomial_eq_l2250_225039

variable (a b c : ℝ)

def polynomial := 16 * a * c + 4 * a^2 - 12 * a * b + 9 * b^2 - 24 * b * c + 16 * c^2

theorem sqrt_polynomial_eq (a b c : ℝ) : 
  (polynomial a b c) ^ (1 / 2) = (2 * a - 3 * b + 4 * c) :=
by
  sorry

end sqrt_polynomial_eq_l2250_225039


namespace parabola_directrix_l2250_225056

theorem parabola_directrix (p : ℝ) (hp : p > 0) (H : - (p / 2) = -3) : p = 6 :=
by
  sorry

end parabola_directrix_l2250_225056


namespace marked_price_of_article_l2250_225063

noncomputable def marked_price (discounted_total : ℝ) (num_articles : ℕ) (discount_rate : ℝ) : ℝ :=
  let selling_price_each := discounted_total / num_articles
  let discount_factor := 1 - discount_rate
  selling_price_each / discount_factor

theorem marked_price_of_article :
  marked_price 50 2 0.10 = 250 / 9 :=
by
  unfold marked_price
  -- Instantiate values:
  -- discounted_total = 50
  -- num_articles = 2
  -- discount_rate = 0.10
  sorry

end marked_price_of_article_l2250_225063


namespace probability_is_12_over_2907_l2250_225060

noncomputable def probability_drawing_red_red_green : ℚ :=
  (3 / 19) * (2 / 18) * (4 / 17)

theorem probability_is_12_over_2907 :
  probability_drawing_red_red_green = 12 / 2907 :=
sorry

end probability_is_12_over_2907_l2250_225060


namespace leo_current_weight_l2250_225036

variable (L K : ℝ)

noncomputable def leo_current_weight_predicate :=
  (L + 10 = 1.5 * K) ∧ (L + K = 180)

theorem leo_current_weight : leo_current_weight_predicate L K → L = 104 := by
  sorry

end leo_current_weight_l2250_225036


namespace constant_seq_decreasing_implication_range_of_values_l2250_225059

noncomputable def sequences (a b : ℕ → ℝ) := 
  (∀ n, a (n+1) = (1/2) * a n + (1/2) * b n) ∧
  (∀ n, (1/b (n+1)) = (1/2) * (1/a n) + (1/2) * (1/b n))

theorem constant_seq (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 > 0) (h_b1 : b 1 > 0) :
  ∃ c, ∀ n, a n * b n = c :=
sorry

theorem decreasing_implication (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 > 0) (h_b1 : b 1 > 0) (h_dec : ∀ n, a (n+1) < a n) :
  a 1 > b 1 :=
sorry

theorem range_of_values (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 = 4) (h_b1 : b 1 = 1) :
  ∀ n ≥ 2, 2 < a n ∧ a n ≤ 5/2 :=
sorry

end constant_seq_decreasing_implication_range_of_values_l2250_225059


namespace unique_solution_l2250_225040

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  x > 0 ∧ (x * Real.sqrt (18 - x) + Real.sqrt (24 * x - x^3) ≥ 18)

theorem unique_solution :
  ∀ x : ℝ, satisfies_condition x ↔ x = 6 :=
by
  intro x
  unfold satisfies_condition
  sorry

end unique_solution_l2250_225040


namespace find_second_number_l2250_225085

theorem find_second_number (a b c : ℕ) (h1 : a = 5 * x) (h2 : b = 3 * x) (h3 : c = 4 * x) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end find_second_number_l2250_225085


namespace total_length_of_rope_l2250_225043

theorem total_length_of_rope (x : ℝ) : (∃ r1 r2 : ℝ, r1 / r2 = 2 / 3 ∧ r1 = 16 ∧ x = r1 + r2) → x = 40 :=
by
  intro h
  cases' h with r1 hr
  cases' hr with r2 hs
  sorry

end total_length_of_rope_l2250_225043


namespace find_initial_red_marbles_l2250_225076

theorem find_initial_red_marbles (x y : ℚ) 
  (h1 : 2 * x = 3 * y) 
  (h2 : 5 * (x - 15) = 2 * (y + 25)) 
  : x = 375 / 11 := 
by
  sorry

end find_initial_red_marbles_l2250_225076


namespace prime_square_sum_of_cubes_equals_three_l2250_225006

open Nat

theorem prime_square_sum_of_cubes_equals_three (p : ℕ) (h_prime : p.Prime) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ p^2 = a^3 + b^3) → (p = 3) :=
by
  sorry

end prime_square_sum_of_cubes_equals_three_l2250_225006


namespace number_of_pieces_of_paper_used_l2250_225094

theorem number_of_pieces_of_paper_used
  (P : ℕ)
  (h1 : 1 / 5 > 0)
  (h2 : 2 / 5 > 0)
  (h3 : 1 < (P : ℝ) * (1 / 5) + 2 / 5 ∧ (P : ℝ) * (1 / 5) + 2 / 5 ≤ 2) : 
  P = 8 :=
sorry

end number_of_pieces_of_paper_used_l2250_225094
