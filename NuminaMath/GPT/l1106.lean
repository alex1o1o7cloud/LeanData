import Mathlib

namespace NUMINAMATH_GPT_correct_number_for_question_mark_l1106_110600

def first_row := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 200]
def second_row_no_quest := [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
def question_mark (x : ℕ) := first_row.sum = second_row_no_quest.sum + x

theorem correct_number_for_question_mark : question_mark 155 := 
by sorry -- proof to be completed

end NUMINAMATH_GPT_correct_number_for_question_mark_l1106_110600


namespace NUMINAMATH_GPT_license_plates_count_l1106_110604

theorem license_plates_count :
  let vowels := 5 -- choices for the first vowel
  let other_letters := 25 -- choices for the second and third letters
  let digits := 10 -- choices for each digit
  (vowels * other_letters * other_letters * (digits * digits * digits)) = 3125000 :=
by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_license_plates_count_l1106_110604


namespace NUMINAMATH_GPT_units_digit_fraction_l1106_110680

theorem units_digit_fraction (h1 : 30 = 2 * 3 * 5) (h2 : 31 = 31) (h3 : 32 = 2^5) 
    (h4 : 33 = 3 * 11) (h5 : 34 = 2 * 17) (h6 : 35 = 5 * 7) (h7 : 7200 = 2^4 * 3^2 * 5^2) :
    ((30 * 31 * 32 * 33 * 34 * 35) / 7200) % 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_fraction_l1106_110680


namespace NUMINAMATH_GPT_cards_given_to_Jeff_l1106_110664

theorem cards_given_to_Jeff
  (initial_cards : ℕ)
  (cards_given_to_John : ℕ)
  (remaining_cards : ℕ)
  (cards_left : ℕ)
  (h_initial : initial_cards = 573)
  (h_given_John : cards_given_to_John = 195)
  (h_left_before_Jeff : remaining_cards = initial_cards - cards_given_to_John)
  (h_final : cards_left = 210)
  (h_given_Jeff : remaining_cards - cards_left = 168) :
  (initial_cards - cards_given_to_John - cards_left = 168) :=
by
  sorry

end NUMINAMATH_GPT_cards_given_to_Jeff_l1106_110664


namespace NUMINAMATH_GPT_min_value_expression_l1106_110658

theorem min_value_expression : ∃ x y z : ℝ, (3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + z^2 + 6 * z + 10) = -7 / 2 :=
by sorry

end NUMINAMATH_GPT_min_value_expression_l1106_110658


namespace NUMINAMATH_GPT_total_percentage_failed_exam_l1106_110646

theorem total_percentage_failed_exam :
  let total_candidates := 2000
  let general_candidates := 1000
  let obc_candidates := 600
  let sc_candidates := 300
  let st_candidates := total_candidates - (general_candidates + obc_candidates + sc_candidates)
  let general_pass_percentage := 0.35
  let obc_pass_percentage := 0.50
  let sc_pass_percentage := 0.25
  let st_pass_percentage := 0.30
  let general_failed := general_candidates - (general_candidates * general_pass_percentage)
  let obc_failed := obc_candidates - (obc_candidates * obc_pass_percentage)
  let sc_failed := sc_candidates - (sc_candidates * sc_pass_percentage)
  let st_failed := st_candidates - (st_candidates * st_pass_percentage)
  let total_failed := general_failed + obc_failed + sc_failed + st_failed
  let failed_percentage := (total_failed / total_candidates) * 100
  failed_percentage = 62.25 :=
by
  sorry

end NUMINAMATH_GPT_total_percentage_failed_exam_l1106_110646


namespace NUMINAMATH_GPT_units_digit_N_l1106_110693

def P (n : ℕ) : ℕ := (n / 10) * (n % 10)
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem units_digit_N (N : ℕ) (h1 : 10 ≤ N ∧ N ≤ 99) (h2 : N = P N + S N) : N % 10 = 9 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_N_l1106_110693


namespace NUMINAMATH_GPT_technicians_count_l1106_110697

def avg_salary_all := 9500
def avg_salary_technicians := 12000
def avg_salary_rest := 6000
def total_workers := 12

theorem technicians_count : 
  ∃ (T R : ℕ), 
  (T + R = total_workers) ∧ 
  ((T * avg_salary_technicians + R * avg_salary_rest) / total_workers = avg_salary_all) ∧ 
  (T = 7) :=
by sorry

end NUMINAMATH_GPT_technicians_count_l1106_110697


namespace NUMINAMATH_GPT_max_value_of_expression_l1106_110630

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 8) : 
  (1 + x) * (1 + y) ≤ 25 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l1106_110630


namespace NUMINAMATH_GPT_father_gave_8_candies_to_Billy_l1106_110605

theorem father_gave_8_candies_to_Billy (candies_Billy : ℕ) (candies_Caleb : ℕ) (candies_Andy : ℕ) (candies_father : ℕ) 
  (candies_given_to_Caleb : ℕ) (candies_more_than_Caleb : ℕ) (candies_given_by_father_total : ℕ) :
  (candies_given_to_Caleb = 11) →
  (candies_Caleb = 11) →
  (candies_Andy = 9) →
  (candies_father = 36) →
  (candies_Andy = candies_Caleb + 4) →
  (candies_given_by_father_total = candies_given_to_Caleb + (candies_Andy - 9)) →
  (candies_father - candies_given_by_father_total = 8) →
  candies_Billy = 8 := 
by
  intros
  sorry

end NUMINAMATH_GPT_father_gave_8_candies_to_Billy_l1106_110605


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l1106_110694

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 145) : x^2 + y^2 = 433 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l1106_110694


namespace NUMINAMATH_GPT_symmetric_points_addition_l1106_110631

theorem symmetric_points_addition 
  (m n : ℝ)
  (A : (ℝ × ℝ)) (B : (ℝ × ℝ))
  (hA : A = (2, m)) 
  (hB : B = (n, -1))
  (symmetry : A.1 = B.1 ∧ A.2 = -B.2) : 
  m + n = 3 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_addition_l1106_110631


namespace NUMINAMATH_GPT_sum_of_squares_l1106_110660

theorem sum_of_squares (n : ℕ) (h : n * (n + 1) * (n + 2) = 12 * (3 * n + 3)) :
  n^2 + (n + 1)^2 + (n + 2)^2 = 29 := 
sorry

end NUMINAMATH_GPT_sum_of_squares_l1106_110660


namespace NUMINAMATH_GPT_greatest_of_consecutive_integers_sum_18_l1106_110690

theorem greatest_of_consecutive_integers_sum_18 
  (x : ℤ) 
  (h1 : x + (x + 1) + (x + 2) = 18) : 
  max x (max (x + 1) (x + 2)) = 7 := 
sorry

end NUMINAMATH_GPT_greatest_of_consecutive_integers_sum_18_l1106_110690


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l1106_110674

-- Defining the parameters and assumptions
variables (x y : ℝ)
variables (h : x * y = 30)

-- Stating the theorem
theorem express_y_in_terms_of_x (h : x * y = 30) : y = 30 / x :=
sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l1106_110674


namespace NUMINAMATH_GPT_mike_average_points_per_game_l1106_110626

theorem mike_average_points_per_game (total_points games_played points_per_game : ℕ) 
  (h1 : games_played = 6) 
  (h2 : total_points = 24) 
  (h3 : total_points = games_played * points_per_game) : 
  points_per_game = 4 :=
by
  rw [h1, h2] at h3  -- Substitute conditions h1 and h2 into the equation
  sorry  -- the proof goes here

end NUMINAMATH_GPT_mike_average_points_per_game_l1106_110626


namespace NUMINAMATH_GPT_continuous_at_5_l1106_110687

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 5 then x - 2 else 3 * x + b

theorem continuous_at_5 {b : ℝ} : ContinuousAt (fun x => f x b) 5 ↔ b = -12 := by
  sorry

end NUMINAMATH_GPT_continuous_at_5_l1106_110687


namespace NUMINAMATH_GPT_decimal_to_binary_51_l1106_110618

theorem decimal_to_binary_51 : (51 : ℕ) = 0b110011 := by sorry

end NUMINAMATH_GPT_decimal_to_binary_51_l1106_110618


namespace NUMINAMATH_GPT_combined_work_rate_l1106_110671

theorem combined_work_rate (x_rate y_rate : ℚ) (h1 : x_rate = 1 / 15) (h2 : y_rate = 1 / 45) :
    1 / (x_rate + y_rate) = 11.25 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_combined_work_rate_l1106_110671


namespace NUMINAMATH_GPT_exist_elem_not_in_union_l1106_110621

-- Assume closed sets
def isClosedSet (S : Set ℝ) : Prop :=
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a - b) ∈ S

-- The theorem to prove
theorem exist_elem_not_in_union {S1 S2 : Set ℝ} (hS1 : isClosedSet S1) (hS2 : isClosedSet S2) :
  S1 ⊂ (Set.univ : Set ℝ) → S2 ⊂ (Set.univ : Set ℝ) → ∃ c : ℝ, c ∉ S1 ∪ S2 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_exist_elem_not_in_union_l1106_110621


namespace NUMINAMATH_GPT_profit_percentage_l1106_110675

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 550) (hSP : SP = 715) : 
  ((SP - CP) / CP) * 100 = 30 := sorry

end NUMINAMATH_GPT_profit_percentage_l1106_110675


namespace NUMINAMATH_GPT_time_per_page_l1106_110654

theorem time_per_page 
    (planning_time : ℝ := 3) 
    (fraction : ℝ := 3/4) 
    (pages_read : ℕ := 9) 
    (minutes_per_hour : ℕ := 60) : 
    (fraction * planning_time * minutes_per_hour) / pages_read = 15 := 
by
  sorry

end NUMINAMATH_GPT_time_per_page_l1106_110654


namespace NUMINAMATH_GPT_part_I_part_II_l1106_110610

-- Define the sets A and B for the given conditions
def setA : Set ℝ := {x | -3 ≤ x - 2 ∧ x - 2 ≤ 1}
def setB (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

-- Part (Ⅰ) When a = 1, find A ∩ B
theorem part_I (a : ℝ) (ha : a = 1) :
  (setA ∩ setB a) = {x | 0 ≤ x ∧ x ≤ 3} :=
by
  sorry

-- Part (Ⅱ) If A ∪ B = A, find the range of real number a
theorem part_II : 
  (∀ a : ℝ, setA ∪ setB a = setA → 0 ≤ a ∧ a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1106_110610


namespace NUMINAMATH_GPT_product_remainder_mod_5_l1106_110623

theorem product_remainder_mod_5 : (2024 * 1980 * 1848 * 1720) % 5 = 0 := by
  sorry

end NUMINAMATH_GPT_product_remainder_mod_5_l1106_110623


namespace NUMINAMATH_GPT_find_constants_eq_l1106_110613

theorem find_constants_eq (P Q R : ℚ)
  (h : ∀ x, (x^2 - 5) = P * (x - 4) * (x - 6) + Q * (x - 1) * (x - 6) + R * (x - 1) * (x - 4)) :
  (P = -4 / 15) ∧ (Q = -11 / 6) ∧ (R = 31 / 10) :=
by
  sorry

end NUMINAMATH_GPT_find_constants_eq_l1106_110613


namespace NUMINAMATH_GPT_each_player_gets_seven_l1106_110609

-- Define the total number of dominoes and players
def total_dominoes : Nat := 28
def total_players : Nat := 4

-- Define the question for how many dominoes each player would receive
def dominoes_per_player (dominoes players : Nat) : Nat := dominoes / players

-- The theorem to prove each player gets 7 dominoes
theorem each_player_gets_seven : dominoes_per_player total_dominoes total_players = 7 :=
by
  sorry

end NUMINAMATH_GPT_each_player_gets_seven_l1106_110609


namespace NUMINAMATH_GPT_derivative_of_exp_sin_l1106_110612

theorem derivative_of_exp_sin (x : ℝ) : 
  (deriv (fun x => Real.exp x * Real.sin x)) x = Real.exp x * Real.sin x + Real.exp x * Real.cos x :=
sorry

end NUMINAMATH_GPT_derivative_of_exp_sin_l1106_110612


namespace NUMINAMATH_GPT_train_speed_is_correct_l1106_110683

-- Definitions of the given conditions.
def train_length : ℕ := 250
def bridge_length : ℕ := 150
def time_taken : ℕ := 20

-- Definition of the total distance covered by the train.
def total_distance : ℕ := train_length + bridge_length

-- The speed calculation.
def speed : ℕ := total_distance / time_taken

-- The theorem that we need to prove.
theorem train_speed_is_correct : speed = 20 := by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_train_speed_is_correct_l1106_110683


namespace NUMINAMATH_GPT_purchase_price_is_600_l1106_110677

open Real

def daily_food_cost : ℝ := 20
def num_days : ℝ := 40
def vaccination_cost : ℝ := 500
def selling_price : ℝ := 2500
def profit : ℝ := 600

def total_food_cost : ℝ := daily_food_cost * num_days
def total_expenses : ℝ := total_food_cost + vaccination_cost
def total_cost : ℝ := selling_price - profit
def purchase_price : ℝ := total_cost - total_expenses

theorem purchase_price_is_600 : purchase_price = 600 := by
  sorry

end NUMINAMATH_GPT_purchase_price_is_600_l1106_110677


namespace NUMINAMATH_GPT_quadratic_has_real_roots_l1106_110663

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_l1106_110663


namespace NUMINAMATH_GPT_percentage_of_pushups_l1106_110634

-- Problem conditions as definitions
def jumpingJacks := 12
def pushups := 8
def situps := 20
def totalExercises := jumpingJacks + pushups + situps

-- Question and the proof goal
theorem percentage_of_pushups : 
  (pushups / totalExercises : ℝ) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_of_pushups_l1106_110634


namespace NUMINAMATH_GPT_balance_increase_second_year_l1106_110667

variable (initial_deposit : ℝ) (balance_first_year : ℝ) 
variable (total_percentage_increase : ℝ)

theorem balance_increase_second_year
  (h1 : initial_deposit = 1000)
  (h2 : balance_first_year = 1100)
  (h3 : total_percentage_increase = 0.32) : 
  (balance_first_year + (initial_deposit * total_percentage_increase) - balance_first_year) / balance_first_year * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_balance_increase_second_year_l1106_110667


namespace NUMINAMATH_GPT_vehicle_speeds_l1106_110672

theorem vehicle_speeds (V_A V_B V_C : ℝ) (d_AB d_AC : ℝ) (decel_A : ℝ)
  (V_A_eff : ℝ) (delta_V_A : ℝ) :
  V_A = 70 → V_B = 50 → V_C = 65 →
  decel_A = 5 → V_A_eff = V_A - decel_A → 
  d_AB = 40 → d_AC = 250 →
  delta_V_A = 10 →
  (d_AB / (V_A_eff + delta_V_A - V_B) < d_AC / (V_A_eff + delta_V_A + V_C)) :=
by
  intros hVA hVB hVC hdecel hV_A_eff hdAB hdAC hdelta_V_A
  -- the proof would be filled in here
  sorry

end NUMINAMATH_GPT_vehicle_speeds_l1106_110672


namespace NUMINAMATH_GPT_hh3_eq_6582_l1106_110684

def h (x : ℤ) : ℤ := 3 * x^2 + 5 * x + 4

theorem hh3_eq_6582 : h (h 3) = 6582 :=
by
  sorry

end NUMINAMATH_GPT_hh3_eq_6582_l1106_110684


namespace NUMINAMATH_GPT_no_such_function_l1106_110676

theorem no_such_function : ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2 := 
sorry

end NUMINAMATH_GPT_no_such_function_l1106_110676


namespace NUMINAMATH_GPT_poll_total_l1106_110620

-- Define the conditions
variables (men women : ℕ)
variables (pct_favor : ℝ := 35) (women_opposed : ℕ := 39)
noncomputable def total_people (men women : ℕ) : ℕ := men + women

-- We need to prove the total number of people polled, given the conditions
theorem poll_total (h1 : men = women)
  (h2 : (pct_favor / 100) * women + (39 : ℝ) / (65 / 100) = (women: ℝ)) :
  total_people men women = 120 :=
sorry

end NUMINAMATH_GPT_poll_total_l1106_110620


namespace NUMINAMATH_GPT_problem_statement_l1106_110633

theorem problem_statement (a b c x : ℝ) (h1 : a + x^2 = 2015) (h2 : b + x^2 = 2016)
    (h3 : c + x^2 = 2017) (h4 : a * b * c = 24) :
    (a / (b * c) + b / (a * c) + c / (a * b) - (1 / a) - (1 / b) - (1 / c) = 1 / 8) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1106_110633


namespace NUMINAMATH_GPT_find_time_interval_l1106_110666

-- Definitions for conditions
def birthRate : ℕ := 4
def deathRate : ℕ := 2
def netIncreaseInPopulationPerInterval (T : ℕ) : ℕ := birthRate - deathRate
def totalTimeInOneDay : ℕ := 86400
def netIncreaseInOneDay (T : ℕ) : ℕ := (totalTimeInOneDay / T) * (netIncreaseInPopulationPerInterval T)

-- Theorem statement
theorem find_time_interval (T : ℕ) (h1 : netIncreaseInPopulationPerInterval T = 2) (h2 : netIncreaseInOneDay T = 86400) : T = 2 :=
sorry

end NUMINAMATH_GPT_find_time_interval_l1106_110666


namespace NUMINAMATH_GPT_number_satisfies_equation_l1106_110617

theorem number_satisfies_equation :
  ∃ x : ℝ, (0.6667 * x - 10 = 0.25 * x) ∧ (x = 23.9936) :=
by
  sorry

end NUMINAMATH_GPT_number_satisfies_equation_l1106_110617


namespace NUMINAMATH_GPT_point_distance_l1106_110611

theorem point_distance (x y n : ℝ) 
    (h1 : abs x = 8) 
    (h2 : (x - 3)^2 + (y - 10)^2 = 225) 
    (h3 : y > 10) 
    (hn : n = Real.sqrt (x^2 + y^2)) : 
    n = Real.sqrt (364 + 200 * Real.sqrt 2) := 
sorry

end NUMINAMATH_GPT_point_distance_l1106_110611


namespace NUMINAMATH_GPT_students_only_one_activity_l1106_110653

theorem students_only_one_activity 
  (total : ℕ) (both : ℕ) (neither : ℕ)
  (h_total : total = 317) 
  (h_both : both = 30) 
  (h_neither : neither = 20) : 
  (total - both - neither) = 267 :=
by 
  sorry

end NUMINAMATH_GPT_students_only_one_activity_l1106_110653


namespace NUMINAMATH_GPT_opposite_of_neg_two_is_two_l1106_110638

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_two_is_two_l1106_110638


namespace NUMINAMATH_GPT_calculate_expr_eq_two_l1106_110696

def calculate_expr : ℕ :=
  3^(0^(2^8)) + (3^0^2)^8

theorem calculate_expr_eq_two : calculate_expr = 2 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expr_eq_two_l1106_110696


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l1106_110652

open Real

theorem solve_equation1 (x : ℝ) : (x^2 - 4 * x + 3 = 0) ↔ (x = 1 ∨ x = 3) := by
  sorry

theorem solve_equation2 (x : ℝ) : (x * (x - 2) = 2 * (2 - x)) ↔ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l1106_110652


namespace NUMINAMATH_GPT_evaluate_expression_zero_l1106_110661

-- Define the variables and conditions
def x : ℕ := 4
def z : ℕ := 0

-- State the property to be proved
theorem evaluate_expression_zero : z * (2 * z - 5 * x) = 0 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_zero_l1106_110661


namespace NUMINAMATH_GPT_probability_red_is_two_fifths_l1106_110641

-- Define the durations
def red_light_duration : ℕ := 30
def yellow_light_duration : ℕ := 5
def green_light_duration : ℕ := 40

-- Define total cycle duration
def total_cycle_duration : ℕ :=
  red_light_duration + yellow_light_duration + green_light_duration

-- Define the probability function
def probability_of_red_light : ℚ :=
  red_light_duration / total_cycle_duration

-- The theorem statement to prove
theorem probability_red_is_two_fifths :
  probability_of_red_light = 2/5 := sorry

end NUMINAMATH_GPT_probability_red_is_two_fifths_l1106_110641


namespace NUMINAMATH_GPT_arithmetic_sum_S8_proof_l1106_110688

-- Definitions of variables and constants
variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def a1_condition : a 1 = -40 := sorry
def a6_a10_condition : a 6 + a 10 = -10 := sorry

-- Theorem to prove
theorem arithmetic_sum_S8_proof (a : ℕ → ℝ) (S : ℕ → ℝ)
  (a1 : a 1 = -40)
  (a6a10 : a 6 + a 10 = -10)
  : S 8 = -180 := 
sorry

end NUMINAMATH_GPT_arithmetic_sum_S8_proof_l1106_110688


namespace NUMINAMATH_GPT_find_k_value_l1106_110625

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 + 3 * x + 7
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := 3 * x^3 - k * x^2 + 4

theorem find_k_value : (f 5 - g 5 k = 45) → k = 27 / 25 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_k_value_l1106_110625


namespace NUMINAMATH_GPT_total_votes_l1106_110639

theorem total_votes (A B C D E : ℕ)
  (votes_A : ℕ) (votes_B : ℕ) (votes_C : ℕ) (votes_D : ℕ) (votes_E : ℕ)
  (dist_A : votes_A = 38 * A / 100)
  (dist_B : votes_B = 28 * B / 100)
  (dist_C : votes_C = 11 * C / 100)
  (dist_D : votes_D = 15 * D / 100)
  (dist_E : votes_E = 8 * E / 100)
  (redistrib_A : votes_A' = votes_A + 5 * A / 100)
  (redistrib_B : votes_B' = votes_B + 5 * B / 100)
  (redistrib_D : votes_D' = votes_D + 2 * D / 100)
  (total_A : votes_A' = 7320) :
  A = 17023 := 
sorry

end NUMINAMATH_GPT_total_votes_l1106_110639


namespace NUMINAMATH_GPT_positive_integer_solution_eq_l1106_110665

theorem positive_integer_solution_eq :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (xyz + 2 * x + 3 * y + 6 * z = xy + 2 * xz + 3 * yz) ∧ (x, y, z) = (4, 3, 1) := 
by
  sorry

end NUMINAMATH_GPT_positive_integer_solution_eq_l1106_110665


namespace NUMINAMATH_GPT_interest_rate_is_12_percent_l1106_110607

-- Definitions
def SI : ℝ := 5400
def P : ℝ := 15000
def T : ℝ := 3

-- Theorem to prove the interest rate
theorem interest_rate_is_12_percent :
  SI = (P * 12 * T) / 100 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_is_12_percent_l1106_110607


namespace NUMINAMATH_GPT_peter_son_is_nikolay_l1106_110616

variable (x y : ℕ)

/-- Within the stated scenarios of Nikolai/Peter paired fishes caught -/
theorem peter_son_is_nikolay :
  (∀ n p ns ps : ℕ, (
    n = ns ∧              -- Nikolai caught as many fish as his son
    p = 3 * ps ∧          -- Peter caught three times more fish than his son
    n + ns + p + ps = 25  -- A total of 25 fish were caught
  ) → ("Nikolay" = "Peter's son")) := 
sorry

end NUMINAMATH_GPT_peter_son_is_nikolay_l1106_110616


namespace NUMINAMATH_GPT_range_of_a_l1106_110637

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x + 1| ≤ a) → a ≥ 2 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1106_110637


namespace NUMINAMATH_GPT_total_pears_picked_is_correct_l1106_110698

-- Define the number of pears picked by Sara and Sally
def pears_picked_by_Sara : ℕ := 45
def pears_picked_by_Sally : ℕ := 11

-- The total number of pears picked
def total_pears_picked := pears_picked_by_Sara + pears_picked_by_Sally

-- The theorem statement: prove that the total number of pears picked is 56
theorem total_pears_picked_is_correct : total_pears_picked = 56 := by
  sorry

end NUMINAMATH_GPT_total_pears_picked_is_correct_l1106_110698


namespace NUMINAMATH_GPT_sum_of_squares_of_two_numbers_l1106_110689

theorem sum_of_squares_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 30) :
  x^2 + y^2 = 840 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_two_numbers_l1106_110689


namespace NUMINAMATH_GPT_min_flight_routes_l1106_110635

-- Defining a problem of connecting cities with flight routes such that 
-- every city can be reached from any other city with no more than two layovers.
theorem min_flight_routes (n : ℕ) (h : n = 50) : ∃ (r : ℕ), (r = 49) ∧
  (∀ (c1 c2 : ℕ), c1 ≠ c2 → c1 < n → c2 < n → ∃ (a b : ℕ),
    a < n ∧ b < n ∧ (a = c1 ∨ a = c2) ∧ (b = c1 ∨ b = c2) ∧
    ((c1 = a ∧ c2 = b) ∨ (c1 = a ∧ b = c2) ∨ (a = c2 ∧ b = c1))) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_flight_routes_l1106_110635


namespace NUMINAMATH_GPT_length_of_chord_AB_l1106_110628

noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)
noncomputable def line_eq (x : ℝ) := x - Real.sqrt 3
noncomputable def ellipse_eq (x y : ℝ)  := x^2 / 4 + y^2 = 1

theorem length_of_chord_AB :
  ∀ (A B : ℝ × ℝ), 
  (line_eq A.1 = A.2) → 
  (line_eq B.1 = B.2) → 
  (ellipse_eq A.1 A.2) → 
  (ellipse_eq B.1 B.2) → 
  ∃ d : ℝ, d = 8 / 5 ∧ 
  dist A B = d := 
sorry

end NUMINAMATH_GPT_length_of_chord_AB_l1106_110628


namespace NUMINAMATH_GPT_problem1_l1106_110699

theorem problem1 : 2 * Real.sin (Real.pi / 3) - 3 * Real.tan (Real.pi / 6) = 0 := by
  sorry

end NUMINAMATH_GPT_problem1_l1106_110699


namespace NUMINAMATH_GPT_right_angle_triangle_iff_arithmetic_progression_l1106_110682

noncomputable def exists_right_angle_triangle_with_rational_sides_and_area (d : ℤ) : Prop :=
  ∃ (a b c : ℚ), (a^2 + b^2 = c^2) ∧ (a * b = 2 * d)

noncomputable def rational_squares_in_arithmetic_progression (x y z : ℚ) : Prop :=
  2 * y^2 = x^2 + z^2

theorem right_angle_triangle_iff_arithmetic_progression (d : ℤ) :
  (∃ (a b c : ℚ), (a^2 + b^2 = c^2) ∧ (a * b = 2 * d)) ↔ ∃ (x y z : ℚ), rational_squares_in_arithmetic_progression x y z :=
sorry

end NUMINAMATH_GPT_right_angle_triangle_iff_arithmetic_progression_l1106_110682


namespace NUMINAMATH_GPT_part1_monotonic_intervals_part2_range_of_a_l1106_110681

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x - x + 0.5

theorem part1_monotonic_intervals (x : ℝ) : 
  (f 1 x < (f 1 (x + 1)) ↔ x < 1) ∧ 
  (f 1 x > (f 1 (x - 1)) ↔ x > 1) :=
by sorry

theorem part2_range_of_a (a : ℝ) (x : ℝ) (hx : 1 < x ∧ x ≤ Real.exp 1) 
  (h : (f a x / x) + (1 / (2 * x)) < 0) : 
  a < 1 - (1 / Real.exp 1) :=
by sorry

end NUMINAMATH_GPT_part1_monotonic_intervals_part2_range_of_a_l1106_110681


namespace NUMINAMATH_GPT_intersection_A_B_l1106_110624

def set_A (x : ℝ) : Prop := (x + 1 / 2 ≥ 3 / 2) ∨ (x + 1 / 2 ≤ -3 / 2)
def set_B (x : ℝ) : Prop := x^2 + x < 6
def A_cap_B := { x : ℝ | set_A x ∧ set_B x }

theorem intersection_A_B : A_cap_B = { x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x < 2) } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1106_110624


namespace NUMINAMATH_GPT_trig_identity_l1106_110659

theorem trig_identity (α : ℝ) :
  1 - Real.cos (2 * α - Real.pi) + Real.cos (4 * α - 2 * Real.pi) =
  4 * Real.cos (2 * α) * Real.cos (Real.pi / 6 + α) * Real.cos (Real.pi / 6 - α) :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1106_110659


namespace NUMINAMATH_GPT_unique_chair_arrangement_l1106_110651

theorem unique_chair_arrangement (n : ℕ) (h : n = 49)
  (h1 : ∀ i j : ℕ, (n = i * j) → (i ≥ 2) ∧ (j ≥ 2)) :
  ∃! i j : ℕ, (n = i * j) ∧ (i ≥ 2) ∧ (j ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_chair_arrangement_l1106_110651


namespace NUMINAMATH_GPT_ellipse_hyperbola_foci_l1106_110662

theorem ellipse_hyperbola_foci {a b : ℝ} (h1 : b^2 - a^2 = 25) (h2 : a^2 + b^2 = 49) :
  a = 2 * Real.sqrt 3 ∧ b = Real.sqrt 37 :=
by sorry

end NUMINAMATH_GPT_ellipse_hyperbola_foci_l1106_110662


namespace NUMINAMATH_GPT_paving_cost_l1106_110657

theorem paving_cost (length width rate : ℝ) (h_length : length = 8) (h_width : width = 4.75) (h_rate : rate = 900) :
  length * width * rate = 34200 :=
by
  rw [h_length, h_width, h_rate]
  norm_num

end NUMINAMATH_GPT_paving_cost_l1106_110657


namespace NUMINAMATH_GPT_find_f1_l1106_110601

variable {R : Type*} [LinearOrderedField R]

-- Define function f of the form px + q
def f (p q x : R) : R := p * x + q

-- Given conditions
variables (p q : R)

-- Define the equations from given conditions
def cond1 : Prop := (f p q 3) = 5
def cond2 : Prop := (f p q 5) = 9

theorem find_f1 (hpq1 : cond1 p q) (hpq2 : cond2 p q) : f p q 1 = 1 := sorry

end NUMINAMATH_GPT_find_f1_l1106_110601


namespace NUMINAMATH_GPT_tom_marbles_l1106_110649

def jason_marbles := 44
def marbles_difference := 20

theorem tom_marbles : (jason_marbles - marbles_difference = 24) :=
by
  sorry

end NUMINAMATH_GPT_tom_marbles_l1106_110649


namespace NUMINAMATH_GPT_total_eggs_michael_has_l1106_110622

-- Define the initial number of crates
def initial_crates : ℕ := 6

-- Define the number of crates given to Susan
def crates_given_to_susan : ℕ := 2

-- Define the number of crates bought on Thursday
def crates_bought_thursday : ℕ := 5

-- Define the number of eggs per crate
def eggs_per_crate : ℕ := 30

-- Theorem stating the total number of eggs Michael has now
theorem total_eggs_michael_has :
  (initial_crates - crates_given_to_susan + crates_bought_thursday) * eggs_per_crate = 270 :=
sorry

end NUMINAMATH_GPT_total_eggs_michael_has_l1106_110622


namespace NUMINAMATH_GPT_average_age_of_town_population_l1106_110608

theorem average_age_of_town_population
  (children adults : ℕ)
  (ratio_condition : 3 * adults = 2 * children)
  (avg_age_children : ℕ := 10)
  (avg_age_adults : ℕ := 40) :
  ((10 * children + 40 * adults) / (children + adults) = 22) :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_town_population_l1106_110608


namespace NUMINAMATH_GPT_altitudes_sum_eq_l1106_110695

variables {α : Type*} [LinearOrderedField α]

structure Triangle (α) :=
(A B C : α)
(R : α)   -- circumradius
(r : α)   -- inradius

variables (T : Triangle α)
(A B C : α)
(m n p : α)  -- points on respective arcs
(h1 h2 h3 : α)  -- altitudes of the segments

theorem altitudes_sum_eq (T : Triangle α) (A B C m n p h1 h2 h3 : α) :
  h1 + h2 + h3 = 2 * T.R - T.r :=
sorry

end NUMINAMATH_GPT_altitudes_sum_eq_l1106_110695


namespace NUMINAMATH_GPT_solve_equation_in_nat_l1106_110645

theorem solve_equation_in_nat {x y : ℕ} :
  (x - 1) / (1 + (x - 1) * y) + (y - 1) / (2 * y - 1) = x / (x + 1) →
  x = 2 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_in_nat_l1106_110645


namespace NUMINAMATH_GPT_total_squares_in_4x4_grid_l1106_110647

-- Define the grid size
def grid_size : ℕ := 4

-- Define a function to count the number of k x k squares in an n x n grid
def count_squares (n k : ℕ) : ℕ :=
  (n - k + 1) * (n - k + 1)

-- Total number of squares in a 4 x 4 grid
def total_squares (n : ℕ) : ℕ :=
  count_squares n 1 + count_squares n 2 + count_squares n 3 + count_squares n 4

-- The main theorem asserting the total number of squares in a 4 x 4 grid is 30
theorem total_squares_in_4x4_grid : total_squares grid_size = 30 := by
  sorry

end NUMINAMATH_GPT_total_squares_in_4x4_grid_l1106_110647


namespace NUMINAMATH_GPT_problem_1_problem_2_l1106_110627

-- Definitions for problem (1)
def p (x a : ℝ) := x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) := x^2 - x - 6 ≤ 0 ∧ x^2 + 3 * x - 10 > 0

-- Statement for problem (1)
theorem problem_1 (a : ℝ) (h : p 1 a ∧ q x) : 2 < x ∧ x < 3 :=
by 
  sorry

-- Definitions for problem (2)
def neg_p (x a : ℝ) := ¬ (x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0)
def neg_q (x : ℝ) := ¬ (x^2 - x - 6 ≤ 0 ∧ x^2 + 3 * x - 10 > 0)

-- Statement for problem (2)
theorem problem_2 (a : ℝ) (h : ∀ x, neg_p x a → neg_q x ∧ ¬ (neg_q x → neg_p x a)) : 1 < a ∧ a ≤ 2 :=
by 
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1106_110627


namespace NUMINAMATH_GPT_number_added_at_end_l1106_110636

theorem number_added_at_end :
  (26.3 * 12 * 20) / 3 + 125 = 2229 := sorry

end NUMINAMATH_GPT_number_added_at_end_l1106_110636


namespace NUMINAMATH_GPT_three_digit_numbers_not_multiples_of_3_or_11_l1106_110673

def count_multiples (a b : ℕ) (lower upper : ℕ) : ℕ :=
  (upper - lower) / b + 1

theorem three_digit_numbers_not_multiples_of_3_or_11 : 
  let total := 900
  let multiples_3 := count_multiples 3 3 102 999
  let multiples_11 := count_multiples 11 11 110 990
  let multiples_33 := count_multiples 33 33 132 990
  let multiples_3_or_11 := multiples_3 + multiples_11 - multiples_33
  total - multiples_3_or_11 = 546 := 
by 
  sorry

end NUMINAMATH_GPT_three_digit_numbers_not_multiples_of_3_or_11_l1106_110673


namespace NUMINAMATH_GPT_greendale_points_l1106_110642

theorem greendale_points : 
  let roosevelt_game1 := 30 
  let roosevelt_game2 := roosevelt_game1 / 2
  let roosevelt_game3 := roosevelt_game2 * 3
  let roosevelt_bonus := 50
  let greendale_diff := 10
  let roosevelt_total := roosevelt_game1 + roosevelt_game2 + roosevelt_game3 + roosevelt_bonus
  let greendale_total := roosevelt_total - greendale_diff
  greendale_total = 130 :=
by
  sorry

end NUMINAMATH_GPT_greendale_points_l1106_110642


namespace NUMINAMATH_GPT_tax_rate_correct_l1106_110632

noncomputable def tax_rate (total_payroll : ℕ) (tax_free_payroll : ℕ) (tax_paid : ℕ) : ℚ :=
  if total_payroll > tax_free_payroll 
  then (tax_paid : ℚ) / (total_payroll - tax_free_payroll) * 100
  else 0

theorem tax_rate_correct :
  tax_rate 400000 200000 400 = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_tax_rate_correct_l1106_110632


namespace NUMINAMATH_GPT_parallel_lines_l1106_110668

-- Definitions based on the conditions
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y - 2 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := 5 * x + (m + 3) * y - 5 = 0
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), line1 m x y → line2 m x y

-- The theorem to be proved
theorem parallel_lines (m : ℝ) (h : parallel m) : m = -5 := 
by
  sorry

end NUMINAMATH_GPT_parallel_lines_l1106_110668


namespace NUMINAMATH_GPT_log_2_bounds_l1106_110602

theorem log_2_bounds:
  (2^9 = 512) → (2^8 = 256) → (10^2 = 100) → (10^3 = 1000) → 
  (2 / 9 < Real.log 2 / Real.log 10) ∧ (Real.log 2 / Real.log 10 < 3 / 8) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_log_2_bounds_l1106_110602


namespace NUMINAMATH_GPT_base4_to_base10_conversion_l1106_110615

theorem base4_to_base10_conversion : 
  (1 * 4^3 + 2 * 4^2 + 1 * 4^1 + 2 * 4^0) = 102 :=
by
  sorry

end NUMINAMATH_GPT_base4_to_base10_conversion_l1106_110615


namespace NUMINAMATH_GPT_divide_square_into_equal_parts_l1106_110655

-- Given a square with four shaded smaller squares inside
structure SquareWithShaded (n : ℕ) :=
  (squares : Fin n → Fin n → Prop) -- this models the presence of shaded squares
  (shaded : (Fin 2) → (Fin 2) → Prop)

-- To prove: we can divide the square into four equal parts with each containing one shaded square
theorem divide_square_into_equal_parts :
  ∀ (sq : SquareWithShaded 4),
  ∃ (parts : Fin 2 → Fin 2 → Prop),
  (∀ i j, parts i j ↔ 
    ((i = 0 ∧ j = 0) ∨ (i = 1 ∧ j = 0) ∨ 
    (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 1)) ∧
    (∃! k l, sq.shaded k l ∧ parts i j)) :=
sorry

end NUMINAMATH_GPT_divide_square_into_equal_parts_l1106_110655


namespace NUMINAMATH_GPT_determine_age_l1106_110691

def David_age (D Y : ℕ) : Prop := Y = 2 * D ∧ Y = D + 7

theorem determine_age (D : ℕ) (h : David_age D (D + 7)) : D = 7 :=
by
  sorry

end NUMINAMATH_GPT_determine_age_l1106_110691


namespace NUMINAMATH_GPT_simplify_expression_l1106_110643

variable (x : ℝ)

theorem simplify_expression : 
  (3 * x + 6 - 5 * x) / 3 = (-2 / 3) * x + 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1106_110643


namespace NUMINAMATH_GPT_sqrt_sin_cos_expression_l1106_110692

theorem sqrt_sin_cos_expression (α β : ℝ) : 
  Real.sqrt ((1 - Real.sin α * Real.sin β)^2 - (Real.cos α * Real.cos β)^2) = |Real.sin α - Real.sin β| :=
sorry

end NUMINAMATH_GPT_sqrt_sin_cos_expression_l1106_110692


namespace NUMINAMATH_GPT_gazprom_rd_expense_l1106_110679

theorem gazprom_rd_expense
  (R_and_D_t : ℝ) (ΔAPL_t_plus_1 : ℝ)
  (h1 : R_and_D_t = 3289.31)
  (h2 : ΔAPL_t_plus_1 = 1.55) :
  R_and_D_t / ΔAPL_t_plus_1 = 2122 := 
by
  sorry

end NUMINAMATH_GPT_gazprom_rd_expense_l1106_110679


namespace NUMINAMATH_GPT_solve_for_x_l1106_110685

-- Define the given equation as a hypothesis
def equation (x : ℝ) : Prop :=
  0.05 * x - 0.09 * (25 - x) = 5.4

-- State the theorem that x = 54.6428571 satisfies the given equation
theorem solve_for_x : (x : ℝ) → equation x → x = 54.6428571 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1106_110685


namespace NUMINAMATH_GPT_no_point_in_common_l1106_110670

theorem no_point_in_common (b : ℝ) :
  (∀ (x y : ℝ), y = 2 * x + b → (x^2 / 4) + y^2 ≠ 1) ↔ (b < -2 * Real.sqrt 2 ∨ b > 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_no_point_in_common_l1106_110670


namespace NUMINAMATH_GPT_scientific_notation_570_million_l1106_110669

theorem scientific_notation_570_million:
  (570 * 10^6 : ℝ) = (5.7 * 10^8 : ℝ) :=
sorry

end NUMINAMATH_GPT_scientific_notation_570_million_l1106_110669


namespace NUMINAMATH_GPT_max_value_of_expression_l1106_110640

theorem max_value_of_expression (x y : ℝ) (h : x + y = 4) :
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 ≤ 7225 / 28 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l1106_110640


namespace NUMINAMATH_GPT_otto_knives_l1106_110686

theorem otto_knives (n : ℕ) (cost : ℕ) : 
  cost = 32 → 
  (n ≥ 1 → cost = 5 + ((min (n - 1) 3) * 4) + ((max 0 (n - 4)) * 3)) → 
  n = 9 :=
by
  intros h_cost h_structure
  sorry

end NUMINAMATH_GPT_otto_knives_l1106_110686


namespace NUMINAMATH_GPT_dihedral_angle_of_equilateral_triangle_l1106_110648

theorem dihedral_angle_of_equilateral_triangle (a : ℝ) 
(ABC_eq : ∀ {A B C : ℝ}, (B - A) ^ 2 + (C - A) ^ 2 = a^2 ∧ (C - B) ^ 2 + (A - B) ^ 2 = a^2 ∧ (A - C) ^ 2 + (B - C) ^ 2 = a^2) 
(perpendicular : ∀ A B C D : ℝ, D = (B + C)/2 ∧ (B - D) * (C - D) = 0) : 
∃ θ : ℝ, θ = 60 := 
  sorry

end NUMINAMATH_GPT_dihedral_angle_of_equilateral_triangle_l1106_110648


namespace NUMINAMATH_GPT_blake_bought_six_chocolate_packs_l1106_110629

-- Defining the conditions as hypotheses
variables (lollipops : ℕ) (lollipopCost : ℕ) (packCost : ℕ)
          (cashGiven : ℕ) (changeReceived : ℕ)
          (totalSpent : ℕ) (totalLollipopCost : ℕ) (amountSpentOnChocolates : ℕ)

-- Assertion of the values based on the conditions
axiom h1 : lollipops = 4
axiom h2 : lollipopCost = 2
axiom h3 : packCost = lollipops * lollipopCost
axiom h4 : cashGiven = 6 * 10
axiom h5 : changeReceived = 4
axiom h6 : totalSpent = cashGiven - changeReceived
axiom h7 : totalLollipopCost = lollipops * lollipopCost
axiom h8 : amountSpentOnChocolates = totalSpent - totalLollipopCost
axiom chocolatePacks : ℕ
axiom h9 : chocolatePacks = amountSpentOnChocolates / packCost

-- The statement to be proved
theorem blake_bought_six_chocolate_packs :
    chocolatePacks = 6 :=
by
  subst_vars
  sorry

end NUMINAMATH_GPT_blake_bought_six_chocolate_packs_l1106_110629


namespace NUMINAMATH_GPT_average_gpa_difference_2_l1106_110678

def avg_gpa_6th_grader := 93
def avg_gpa_8th_grader := 91
def school_avg_gpa := 93

noncomputable def gpa_diff (gpa_7th_grader diff : ℝ) (avg6 avg8 school_avg : ℝ) := 
  gpa_7th_grader = avg6 + diff ∧ 
  (avg6 + gpa_7th_grader + avg8) / 3 = school_avg

theorem average_gpa_difference_2 (x : ℝ) : 
  (∃ G : ℝ, gpa_diff G x avg_gpa_6th_grader avg_gpa_8th_grader school_avg_gpa) → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_gpa_difference_2_l1106_110678


namespace NUMINAMATH_GPT_max_value_of_seq_diff_l1106_110606

theorem max_value_of_seq_diff :
  ∀ (a : Fin 2017 → ℝ),
    a 0 = a 2016 →
    (∀ i : Fin 2015, |a i + a (i+2) - 2 * a (i+1)| ≤ 1) →
    ∃ b : ℝ, b = 508032 ∧ ∀ i j, 1 ≤ i → i < j → j ≤ 2017 → |a i - a j| ≤ b :=
  sorry

end NUMINAMATH_GPT_max_value_of_seq_diff_l1106_110606


namespace NUMINAMATH_GPT_num_bikes_l1106_110656

variable (C B : ℕ)

-- The given conditions
def num_cars : ℕ := 10
def num_wheels_total : ℕ := 44
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

-- The mathematical proof problem statement
theorem num_bikes :
  C = num_cars →
  B = ((num_wheels_total - (C * wheels_per_car)) / wheels_per_bike) →
  B = 2 :=
by
  intros hC hB
  rw [hC] at hB
  sorry

end NUMINAMATH_GPT_num_bikes_l1106_110656


namespace NUMINAMATH_GPT_probability_more_ones_than_sixes_l1106_110650

theorem probability_more_ones_than_sixes :
  (∃ (p : ℚ), p = 1673 / 3888 ∧ 
  (∃ (d : Fin 6 → ℕ), 
  (∀ i, d i ≤ 4) ∧ 
  (∃ d1 d6 : ℕ, (1 ≤ d1 + d6 ∧ d1 + d6 ≤ 5 ∧ d1 > d6)))) :=
sorry

end NUMINAMATH_GPT_probability_more_ones_than_sixes_l1106_110650


namespace NUMINAMATH_GPT_simplify_expression_l1106_110619

theorem simplify_expression :
  (Real.sqrt 600 / Real.sqrt 75 - Real.sqrt 243 / Real.sqrt 108) = (4 * Real.sqrt 2 - 3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1106_110619


namespace NUMINAMATH_GPT_race_time_A_l1106_110603

theorem race_time_A (v t : ℝ) (h1 : 1000 = v * t) (h2 : 950 = v * (t - 10)) : t = 200 :=
by
  sorry

end NUMINAMATH_GPT_race_time_A_l1106_110603


namespace NUMINAMATH_GPT_sin4x_eq_sin2x_solution_set_l1106_110644

noncomputable def solution_set (x : ℝ) : Prop :=
  0 < x ∧ x < (3 / 2) * Real.pi ∧ Real.sin (4 * x) = Real.sin (2 * x)

theorem sin4x_eq_sin2x_solution_set :
  { x : ℝ | solution_set x } =
  { (Real.pi / 6), (Real.pi / 2), Real.pi, (5 * Real.pi / 6), (7 * Real.pi / 6) } :=
by
  sorry

end NUMINAMATH_GPT_sin4x_eq_sin2x_solution_set_l1106_110644


namespace NUMINAMATH_GPT_opposite_numbers_l1106_110614

theorem opposite_numbers
  (odot otimes : ℝ)
  (x y : ℝ)
  (h1 : 6 * x + odot * y = 3)
  (h2 : 2 * x + otimes * y = -1)
  (h_add : 6 * x + odot * y + (2 * x + otimes * y) = 2) :
  odot + otimes = 0 := by
  sorry

end NUMINAMATH_GPT_opposite_numbers_l1106_110614
