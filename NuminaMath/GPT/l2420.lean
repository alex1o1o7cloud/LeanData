import Mathlib

namespace NUMINAMATH_GPT_milk_remaining_l2420_242087

def initial_whole_milk := 15
def initial_low_fat_milk := 12
def initial_almond_milk := 8

def jason_buys := 5
def jason_promotion := 2 -- every 2 bottles he gets 1 free

def harry_buys_low_fat := 4
def harry_gets_free_low_fat := 1
def harry_buys_almond := 2

theorem milk_remaining : 
  (initial_whole_milk - jason_buys = 10) ∧ 
  (initial_low_fat_milk - (harry_buys_low_fat + harry_gets_free_low_fat) = 7) ∧ 
  (initial_almond_milk - harry_buys_almond = 6) :=
by
  sorry

end NUMINAMATH_GPT_milk_remaining_l2420_242087


namespace NUMINAMATH_GPT_part1_a_value_part2_solution_part3_incorrect_solution_l2420_242069

-- Part 1: Given solution {x = 1, y = 1}, prove a = 3
theorem part1_a_value (a : ℤ) (h1 : 1 + 2 * 1 = a) : a = 3 := 
by 
  sorry

-- Part 2: Given a = -2, prove the solution is {x = 0, y = -1}
theorem part2_solution (x y : ℤ) (h1 : x + 2 * y = -2) (h2 : 2 * x - y = 1) : x = 0 ∧ y = -1 := 
by 
  sorry

-- Part 3: Given {x = -2, y = -2}, prove that it is not a solution
theorem part3_incorrect_solution (a : ℤ) (h1 : -2 + 2 * (-2) = a) (h2 : 2 * (-2) - (-2) = 1) : False := 
by 
  sorry

end NUMINAMATH_GPT_part1_a_value_part2_solution_part3_incorrect_solution_l2420_242069


namespace NUMINAMATH_GPT_no_zero_sum_of_vectors_l2420_242010

-- Definitions and conditions for the problem
variable {n : ℕ} (odd_n : n % 2 = 1) -- n is odd, representing the number of sides of the polygon

-- The statement of the proof problem
theorem no_zero_sum_of_vectors (odd_n : n % 2 = 1) : false :=
by
  sorry

end NUMINAMATH_GPT_no_zero_sum_of_vectors_l2420_242010


namespace NUMINAMATH_GPT_rectangle_total_area_l2420_242039

-- Let s be the side length of the smaller squares
variable (s : ℕ)

-- Define the areas of the squares
def smaller_square_area := s ^ 2
def larger_square_area := (3 * s) ^ 2

-- Define the total_area
def total_area : ℕ := 2 * smaller_square_area s + larger_square_area s

-- Assert the total area of the rectangle ABCD is 11s^2
theorem rectangle_total_area (s : ℕ) : total_area s = 11 * s ^ 2 := 
by 
  -- the proof is skipped
  sorry

end NUMINAMATH_GPT_rectangle_total_area_l2420_242039


namespace NUMINAMATH_GPT_rational_solution_exists_l2420_242084

theorem rational_solution_exists :
  ∃ (a b : ℚ), (a + b) / a + a / (a + b) = b :=
by
  sorry

end NUMINAMATH_GPT_rational_solution_exists_l2420_242084


namespace NUMINAMATH_GPT_total_number_of_workers_l2420_242077

-- Definitions based on the given conditions
def avg_salary_total : ℝ := 8000
def avg_salary_technicians : ℝ := 12000
def avg_salary_non_technicians : ℝ := 6000
def num_technicians : ℕ := 7

-- Problem statement in Lean
theorem total_number_of_workers
    (W : ℕ) (N : ℕ)
    (h1 : W * avg_salary_total = num_technicians * avg_salary_technicians + N * avg_salary_non_technicians)
    (h2 : W = num_technicians + N) :
    W = 21 :=
sorry

end NUMINAMATH_GPT_total_number_of_workers_l2420_242077


namespace NUMINAMATH_GPT_change_factor_w_l2420_242096

theorem change_factor_w (w d z F_w : Real)
  (h_q : ∀ w d z, q = 5 * w / (4 * d * z^2))
  (h1 : d' = 2 * d)
  (h2 : z' = 3 * z)
  (h3 : F_q = 0.2222222222222222)
  : F_w = 4 :=
by
  sorry

end NUMINAMATH_GPT_change_factor_w_l2420_242096


namespace NUMINAMATH_GPT_find_b_l2420_242075

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 3) : b = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l2420_242075


namespace NUMINAMATH_GPT_molecular_weight_is_122_l2420_242059

noncomputable def molecular_weight_of_compound := 
  let atomic_weight_C := 12.01
  let atomic_weight_H := 1.008
  let atomic_weight_O := 16.00
  7 * atomic_weight_C + 6 * atomic_weight_H + 2 * atomic_weight_O

theorem molecular_weight_is_122 :
  molecular_weight_of_compound = 122 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_is_122_l2420_242059


namespace NUMINAMATH_GPT_divisor_of_subtracted_number_l2420_242066

theorem divisor_of_subtracted_number (n : ℕ) (m : ℕ) (h : n = 5264 - 11) : Nat.gcd n 5264 = 5253 :=
by
  sorry

end NUMINAMATH_GPT_divisor_of_subtracted_number_l2420_242066


namespace NUMINAMATH_GPT_solve_problem_l2420_242050

-- Definitions from the conditions
def is_divisible_by (n k : ℕ) : Prop :=
  ∃ m, k * m = n

def count_divisors (limit k : ℕ) : ℕ :=
  Nat.div limit k

def count_numbers_divisible_by_neither_5_nor_7 (limit : ℕ) : ℕ :=
  let total := limit - 1
  let divisible_by_5 := count_divisors limit 5
  let divisible_by_7 := count_divisors limit 7
  let divisible_by_35 := count_divisors limit 35
  total - (divisible_by_5 + divisible_by_7 - divisible_by_35)

-- The statement to be proved
theorem solve_problem : count_numbers_divisible_by_neither_5_nor_7 1000 = 686 :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l2420_242050


namespace NUMINAMATH_GPT_R_and_D_per_increase_l2420_242026

def R_and_D_t : ℝ := 3013.94
def Delta_APL_t2 : ℝ := 3.29

theorem R_and_D_per_increase :
  R_and_D_t / Delta_APL_t2 = 916 := by
  sorry

end NUMINAMATH_GPT_R_and_D_per_increase_l2420_242026


namespace NUMINAMATH_GPT_fuel_tank_capacity_l2420_242061

def ethanol_content_fuel_A (fuel_A : ℝ) : ℝ := 0.12 * fuel_A
def ethanol_content_fuel_B (fuel_B : ℝ) : ℝ := 0.16 * fuel_B

theorem fuel_tank_capacity (C : ℝ) :
  ethanol_content_fuel_A 122 + ethanol_content_fuel_B (C - 122) = 30 → C = 218 :=
by
  sorry

end NUMINAMATH_GPT_fuel_tank_capacity_l2420_242061


namespace NUMINAMATH_GPT_find_integer_N_l2420_242015

theorem find_integer_N : ∃ N : ℤ, (N ^ 2 ≡ N [ZMOD 10000]) ∧ (N - 2 ≡ 0 [ZMOD 7]) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_N_l2420_242015


namespace NUMINAMATH_GPT_fg_of_3_l2420_242007

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 + 2
def g (x : ℝ) : ℝ := 3 * x + 4

-- Theorem statement to prove f(g(3)) = 2199
theorem fg_of_3 : f (g 3) = 2199 :=
by
  sorry

end NUMINAMATH_GPT_fg_of_3_l2420_242007


namespace NUMINAMATH_GPT_range_of_m_for_false_proposition_l2420_242094

theorem range_of_m_for_false_proposition :
  ¬ (∃ x : ℝ, x^2 - m * x - m ≤ 0) → m ∈ Set.Ioo (-4 : ℝ) 0 :=
sorry

end NUMINAMATH_GPT_range_of_m_for_false_proposition_l2420_242094


namespace NUMINAMATH_GPT_dollar_eval_l2420_242001

def dollar (a b : ℝ) : ℝ := (a^2 - b^2)^2

theorem dollar_eval (x : ℝ) : dollar (x^3 + x) (x - x^3) = 16 * x^8 :=
by
  sorry

end NUMINAMATH_GPT_dollar_eval_l2420_242001


namespace NUMINAMATH_GPT_gym_monthly_revenue_l2420_242098

-- Defining the conditions
def charge_per_session : ℕ := 18
def sessions_per_month : ℕ := 2
def number_of_members : ℕ := 300

-- Defining the question as a theorem statement
theorem gym_monthly_revenue : 
  (number_of_members * (charge_per_session * sessions_per_month)) = 10800 := 
by 
  -- Skip the proof, verifying the statement only
  sorry

end NUMINAMATH_GPT_gym_monthly_revenue_l2420_242098


namespace NUMINAMATH_GPT_number_add_thrice_number_eq_twenty_l2420_242083

theorem number_add_thrice_number_eq_twenty (x : ℝ) (h : x + 3 * x = 20) : x = 5 :=
sorry

end NUMINAMATH_GPT_number_add_thrice_number_eq_twenty_l2420_242083


namespace NUMINAMATH_GPT_ratio_of_supply_to_demand_l2420_242008

theorem ratio_of_supply_to_demand (supply demand : ℕ)
  (hs : supply = 1800000)
  (hd : demand = 2400000) :
  supply / (Nat.gcd supply demand) = 3 ∧ demand / (Nat.gcd supply demand) = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_supply_to_demand_l2420_242008


namespace NUMINAMATH_GPT_original_price_of_movie_ticket_l2420_242040

theorem original_price_of_movie_ticket
    (P : ℝ)
    (new_price : ℝ)
    (h1 : new_price = 80)
    (h2 : new_price = 0.80 * P) :
    P = 100 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_movie_ticket_l2420_242040


namespace NUMINAMATH_GPT_studentsInBandOrSports_l2420_242041

-- conditions definitions
def totalStudents : ℕ := 320
def studentsInBand : ℕ := 85
def studentsInSports : ℕ := 200
def studentsInBoth : ℕ := 60

-- theorem statement
theorem studentsInBandOrSports : studentsInBand + studentsInSports - studentsInBoth = 225 :=
by
  sorry

end NUMINAMATH_GPT_studentsInBandOrSports_l2420_242041


namespace NUMINAMATH_GPT_problem1_problem2_l2420_242043

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x * f a x - x) / Real.exp x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 1 - (f a x - 1) / Real.exp x

theorem problem1 (x : ℝ) (h₁ : x ≥ 5) : g 1 x < 1 :=
sorry

theorem problem2 (a : ℝ) (h₂ : a > Real.exp 2 / 4) : 
∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 ∧ h a x1 = 0 ∧ h a x2 = 0 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2420_242043


namespace NUMINAMATH_GPT_sum_first_last_l2420_242045

theorem sum_first_last (A B C D : ℕ) (h1 : (A + B + C) / 3 = 6) (h2 : (B + C + D) / 3 = 5) (h3 : D = 4) : A + D = 11 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_last_l2420_242045


namespace NUMINAMATH_GPT_sarah_daily_candy_consumption_l2420_242065

def neighbors_candy : ℕ := 66
def sister_candy : ℕ := 15
def days : ℕ := 9

def total_candy : ℕ := neighbors_candy + sister_candy
def average_daily_consumption : ℕ := total_candy / days

theorem sarah_daily_candy_consumption : average_daily_consumption = 9 := by
  sorry

end NUMINAMATH_GPT_sarah_daily_candy_consumption_l2420_242065


namespace NUMINAMATH_GPT_h_h_of_2_l2420_242049

def h (x : ℝ) : ℝ := 4 * x^2 - 8

theorem h_h_of_2 : h (h 2) = 248 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_h_h_of_2_l2420_242049


namespace NUMINAMATH_GPT_directrix_of_parabola_l2420_242002

noncomputable def parabola_directrix (x : ℝ) : ℝ := 4 * x^2 + 4 * x + 1

theorem directrix_of_parabola :
  ∃ (y : ℝ) (x : ℝ), parabola_directrix x = y ∧ y = 4 * (x + 1/2)^2 + 3/4 ∧ y - 1/16 = 11/16 :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l2420_242002


namespace NUMINAMATH_GPT_stepa_and_petya_are_wrong_l2420_242029

-- Define the six-digit number where all digits are the same.
def six_digit_same (a : ℕ) : ℕ := a * 111111

-- Define the sum of distinct prime divisors of 1001 and 111.
def prime_divisor_sum : ℕ := 7 + 11 + 13 + 3 + 37

-- Define the sum of prime divisors when a is considered.
def additional_sum (a : ℕ) : ℕ :=
  if (a = 2) || (a = 6) || (a = 8) then 2
  else if (a = 5) then 5
  else 0

-- Summarize the possible correct sums
def correct_sums (a : ℕ) : ℕ := prime_divisor_sum + additional_sum a

-- The proof statement
theorem stepa_and_petya_are_wrong (a : ℕ) :
  correct_sums a ≠ 70 ∧ correct_sums a ≠ 80 := 
by {
  sorry
}

end NUMINAMATH_GPT_stepa_and_petya_are_wrong_l2420_242029


namespace NUMINAMATH_GPT_area_of_region_l2420_242089

theorem area_of_region (x y : ℝ) :
  x ≤ 2 * y ∧ y ≤ 2 * x ∧ x + y ≤ 60 →
  ∃ (A : ℝ), A = 600 :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_l2420_242089


namespace NUMINAMATH_GPT_A_days_to_complete_job_l2420_242071

noncomputable def time_for_A (x : ℝ) (work_left : ℝ) : ℝ :=
  let work_rate_A := 1 / x
  let work_rate_B := 1 / 30
  let combined_work_rate := work_rate_A + work_rate_B
  let completed_work := 4 * combined_work_rate
  let fraction_work_left := 1 - completed_work
  fraction_work_left

theorem A_days_to_complete_job : ∃ x : ℝ, time_for_A x 0.6 = 0.6 ∧ x = 15 :=
by {
  use 15,
  sorry
}

end NUMINAMATH_GPT_A_days_to_complete_job_l2420_242071


namespace NUMINAMATH_GPT_smallest_number_is_D_l2420_242019

-- Define the given numbers in Lean
def A := 25
def B := 111
def C := 16 + 4 + 2  -- since 10110_{(2)} equals 22 in base 10
def D := 16 + 2 + 1  -- since 10011_{(2)} equals 19 in base 10

-- The Lean statement for the proof problem
theorem smallest_number_is_D : min (min A B) (min C D) = D := by
  sorry

end NUMINAMATH_GPT_smallest_number_is_D_l2420_242019


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l2420_242088

theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 - (y^2 / 4) = 1) ↔ (y = 2 * x ∨ y = -2 * x) := by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l2420_242088


namespace NUMINAMATH_GPT_value_of_m_l2420_242016

theorem value_of_m (m : ℝ) : (∀ x : ℝ, x^2 + m * x + 9 = (x + 3)^2) → m = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_m_l2420_242016


namespace NUMINAMATH_GPT_isosceles_trapezoid_legs_squared_l2420_242031

theorem isosceles_trapezoid_legs_squared
  (A B C D : Type)
  (AB CD AD BC : ℝ)
  (isosceles_trapezoid : AB = 50 ∧ CD = 14 ∧ AD = BC)
  (circle_tangent : ∃ M : ℝ, M = 25 ∧ ∀ x : ℝ, MD = 7 ↔ AD = x ∧ BC = x) :
  AD^2 = 800 := 
by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_legs_squared_l2420_242031


namespace NUMINAMATH_GPT_parallelogram_sides_l2420_242003

theorem parallelogram_sides (a b : ℕ): 
  (a = 3 * b) ∧ (2 * a + 2 * b = 24) → (a = 9) ∧ (b = 3) :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_sides_l2420_242003


namespace NUMINAMATH_GPT_compute_difference_a_b_l2420_242022

-- Define the initial amounts paid by Alex, Bob, and Carol
def alex_paid := 120
def bob_paid := 150
def carol_paid := 210

-- Define the total amount and equal share
def total_costs := alex_paid + bob_paid + carol_paid
def equal_share := total_costs / 3

-- Define the amounts Alex and Carol gave to Bob, satisfying their balances
def a := equal_share - alex_paid
def b := carol_paid - equal_share

-- Lean 4 statement to prove a - b = 30
theorem compute_difference_a_b : a - b = 30 := by
  sorry

end NUMINAMATH_GPT_compute_difference_a_b_l2420_242022


namespace NUMINAMATH_GPT_milk_volume_in_ounces_l2420_242021

theorem milk_volume_in_ounces
  (packets : ℕ)
  (volume_per_packet_ml : ℕ)
  (ml_per_oz : ℕ)
  (total_volume_ml : ℕ)
  (total_volume_oz : ℕ)
  (h1 : packets = 150)
  (h2 : volume_per_packet_ml = 250)
  (h3 : ml_per_oz = 30)
  (h4 : total_volume_ml = packets * volume_per_packet_ml)
  (h5 : total_volume_oz = total_volume_ml / ml_per_oz) :
  total_volume_oz = 1250 :=
by
  sorry

end NUMINAMATH_GPT_milk_volume_in_ounces_l2420_242021


namespace NUMINAMATH_GPT_claudia_coins_l2420_242064

variable (x y : ℕ)

theorem claudia_coins :
  (x + y = 15 ∧ ((145 - 5 * x) / 5) + 1 = 23) → y = 9 :=
by
  intro h
  -- The proof steps would go here, but we'll leave it as sorry for now.
  sorry

end NUMINAMATH_GPT_claudia_coins_l2420_242064


namespace NUMINAMATH_GPT_inequality_holds_for_minimal_a_l2420_242047

theorem inequality_holds_for_minimal_a :
  ∀ (x : ℝ), (1 ≤ x) → (x ≤ 4) → (1 + x) * Real.log x + x ≤ x * 1.725 :=
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_inequality_holds_for_minimal_a_l2420_242047


namespace NUMINAMATH_GPT_eleven_pow_2023_mod_eight_l2420_242014

theorem eleven_pow_2023_mod_eight (h11 : 11 % 8 = 3) (h3 : 3^2 % 8 = 1) : 11^2023 % 8 = 3 :=
by
  sorry

end NUMINAMATH_GPT_eleven_pow_2023_mod_eight_l2420_242014


namespace NUMINAMATH_GPT_problem_c_l2420_242056

theorem problem_c (x y : ℝ) (h : x - 3 = y - 3): x - y = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_c_l2420_242056


namespace NUMINAMATH_GPT_age_difference_l2420_242004

theorem age_difference (M S : ℕ) (h1 : S = 16) (h2 : M + 2 = 2 * (S + 2)) : M - S = 18 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l2420_242004


namespace NUMINAMATH_GPT_parallelogram_height_l2420_242099

theorem parallelogram_height (A B H : ℝ) (hA : A = 462) (hB : B = 22) (hArea : A = B * H) : H = 21 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_height_l2420_242099


namespace NUMINAMATH_GPT_leftmost_square_side_length_l2420_242097

open Real

/-- Given the side lengths of three squares, 
    where the middle square's side length is 17 cm longer than the leftmost square,
    the rightmost square's side length is 6 cm shorter than the middle square,
    and the sum of the side lengths of all three squares is 52 cm,
    prove that the side length of the leftmost square is 8 cm. -/
theorem leftmost_square_side_length
  (x : ℝ)
  (h1 : ∀ m : ℝ, m = x + 17)
  (h2 : ∀ r : ℝ, r = x + 11)
  (h3 : x + (x + 17) + (x + 11) = 52) :
  x = 8 := by
  sorry

end NUMINAMATH_GPT_leftmost_square_side_length_l2420_242097


namespace NUMINAMATH_GPT_chose_number_l2420_242017

theorem chose_number (x : ℝ) (h : (x / 12)^2 - 240 = 8) : x = 24 * Real.sqrt 62 :=
sorry

end NUMINAMATH_GPT_chose_number_l2420_242017


namespace NUMINAMATH_GPT_num_pos_multiples_of_six_is_150_l2420_242081

theorem num_pos_multiples_of_six_is_150 : 
  ∃ (n : ℕ), (∀ k, (n = 150) ↔ (102 + (k - 1) * 6 = 996 ∧ 102 ≤ 6 * k ∧ 6 * k ≤ 996)) :=
sorry

end NUMINAMATH_GPT_num_pos_multiples_of_six_is_150_l2420_242081


namespace NUMINAMATH_GPT_sum_of_powers_of_i_l2420_242054

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 + i^4 + i^5 = i := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_i_l2420_242054


namespace NUMINAMATH_GPT_range_of_k_l2420_242062

theorem range_of_k (k : ℝ) (x y : ℝ) : 
  (y = 2 * x - 5 * k + 7) → 
  (y = - (1 / 2) * x + 2) → 
  (x > 0) → 
  (y > 0) → 
  (1 < k ∧ k < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l2420_242062


namespace NUMINAMATH_GPT_value_of_expression_l2420_242033

theorem value_of_expression : (3 + 2) - (2 + 1) = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2420_242033


namespace NUMINAMATH_GPT_intercepts_l2420_242093

def line_equation (x y : ℝ) : Prop :=
  5 * x + 3 * y - 15 = 0

theorem intercepts (a b : ℝ) : line_equation a 0 ∧ line_equation 0 b → (a = 3 ∧ b = 5) :=
  sorry

end NUMINAMATH_GPT_intercepts_l2420_242093


namespace NUMINAMATH_GPT_smallest_base_for_62_three_digits_l2420_242051

theorem smallest_base_for_62_three_digits: 
  ∃ b : ℕ, (b^2 ≤ 62 ∧ 62 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 62 ∧ 62 < n^3) → n ≥ b :=
by
  sorry

end NUMINAMATH_GPT_smallest_base_for_62_three_digits_l2420_242051


namespace NUMINAMATH_GPT_find_m_l2420_242057

noncomputable def first_series_sum : ℝ := 
  let a1 : ℝ := 18
  let a2 : ℝ := 6
  let r : ℝ := a2 / a1
  a1 / (1 - r)

noncomputable def second_series_sum (m : ℝ) : ℝ := 
  let b1 : ℝ := 18
  let b2 : ℝ := 6 + m
  let s : ℝ := b2 / b1
  b1 / (1 - s)

theorem find_m : 
  (3 : ℝ) * first_series_sum = second_series_sum m → m = 8 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_l2420_242057


namespace NUMINAMATH_GPT_max_product_condition_l2420_242037

theorem max_product_condition (x y : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 12) (h3 : 0 ≤ y) (h4 : y ≤ 12) (h_eq : x * y = (12 - x) ^ 2 * (12 - y) ^ 2) : x * y ≤ 81 :=
sorry

end NUMINAMATH_GPT_max_product_condition_l2420_242037


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2420_242055

theorem hyperbola_eccentricity :
  let a := 2
  let b := 2 * Real.sqrt 2
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (e = Real.sqrt 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_hyperbola_eccentricity_l2420_242055


namespace NUMINAMATH_GPT_same_graphs_at_x_eq_1_l2420_242035

theorem same_graphs_at_x_eq_1 :
  let y1 := 2 - 1
  let y2 := (1^3 - 1) / (1 - 1)
  let y3 := (1^3 - 1) / (1 - 1)
  y2 = 3 ∧ y3 = 3 ∧ y1 ≠ y2 := 
by
  let y1 := 2 - 1
  let y2 := (1^3 - 1) / (1 - 1)
  let y3 := (1^3 - 1) / (1 - 1)
  sorry

end NUMINAMATH_GPT_same_graphs_at_x_eq_1_l2420_242035


namespace NUMINAMATH_GPT_certain_number_eq_l2420_242048

theorem certain_number_eq :
  ∃ y : ℝ, y + (y * 4) = 48 ∧ y = 9.6 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_eq_l2420_242048


namespace NUMINAMATH_GPT_distance_AC_l2420_242011

theorem distance_AC (south_dist : ℕ) (west_dist : ℕ) (north_dist : ℕ) (east_dist : ℕ) :
  south_dist = 50 → west_dist = 70 → north_dist = 30 → east_dist = 40 →
  Real.sqrt ((south_dist - north_dist)^2 + (west_dist - east_dist)^2) = 36.06 :=
by
  intros h_south h_west h_north h_east
  rw [h_south, h_west, h_north, h_east]
  simp
  norm_num
  sorry

end NUMINAMATH_GPT_distance_AC_l2420_242011


namespace NUMINAMATH_GPT_drawing_red_ball_random_drawing_yellow_ball_impossible_probability_black_ball_number_of_additional_black_balls_l2420_242063

-- Definitions for the initial conditions
def initial_white_balls := 2
def initial_black_balls := 3
def initial_red_balls := 5
def total_balls := initial_white_balls + initial_black_balls + initial_red_balls

-- Statement for part 1: Drawing a red ball is a random event
theorem drawing_red_ball_random : (initial_red_balls > 0) := by
  sorry

-- Statement for part 1: Drawing a yellow ball is impossible
theorem drawing_yellow_ball_impossible : (0 = 0) := by
  sorry

-- Statement for part 2: Probability of drawing a black ball
theorem probability_black_ball : (initial_black_balls : ℚ) / total_balls = 3 / 10 := by
  sorry

-- Definitions for the conditions in part 3
def additional_black_balls (x : ℕ) := initial_black_balls + x
def new_total_balls (x : ℕ) := total_balls + x

-- Statement for part 3: Finding the number of additional black balls
theorem number_of_additional_black_balls (x : ℕ)
  (h : (additional_black_balls x : ℚ) / new_total_balls x = 3 / 4) : x = 18 := by
  sorry

end NUMINAMATH_GPT_drawing_red_ball_random_drawing_yellow_ball_impossible_probability_black_ball_number_of_additional_black_balls_l2420_242063


namespace NUMINAMATH_GPT_remainder_zero_l2420_242028

theorem remainder_zero (x : ℤ) :
  (x^5 - 1) * (x^3 - 1) % (x^2 + x + 1) = 0 := by
sorry

end NUMINAMATH_GPT_remainder_zero_l2420_242028


namespace NUMINAMATH_GPT_problem_I_problem_II_l2420_242036

def setA : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≤ 0}

theorem problem_I (a : ℝ) : (setB a ⊆ setA) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

theorem problem_II (a : ℝ) : (setA ∩ setB a = {1}) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l2420_242036


namespace NUMINAMATH_GPT_evaluate_expression_l2420_242046

theorem evaluate_expression : 2 + 0 - 2 * 0 = 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2420_242046


namespace NUMINAMATH_GPT_domain_of_f_l2420_242074

open Set Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 6*x + 9)

theorem domain_of_f :
  {x : ℝ | f x ≠ f (-3)} = Iio (-3) ∪ Ioi (-3) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2420_242074


namespace NUMINAMATH_GPT_fraction_irreducible_gcd_2_power_l2420_242030

-- Proof problem (a)
theorem fraction_irreducible (n : ℕ) : gcd (12 * n + 1) (30 * n + 2) = 1 :=
sorry

-- Proof problem (b)
theorem gcd_2_power (n m : ℕ) : gcd (2^100 - 1) (2^120 - 1) = 2^20 - 1 :=
sorry

end NUMINAMATH_GPT_fraction_irreducible_gcd_2_power_l2420_242030


namespace NUMINAMATH_GPT_wallpaper_expenditure_l2420_242042

structure Room :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

def cost_per_square_meter : ℕ := 75

def total_expenditure (room : Room) : ℕ :=
  let perimeter := 2 * (room.length + room.width)
  let area_of_walls := perimeter * room.height
  let area_of_ceiling := room.length * room.width
  let total_area := area_of_walls + area_of_ceiling
  total_area * cost_per_square_meter

theorem wallpaper_expenditure (room : Room) : 
  room = Room.mk 30 25 10 →
  total_expenditure room = 138750 :=
by 
  intros h
  rw [h]
  sorry

end NUMINAMATH_GPT_wallpaper_expenditure_l2420_242042


namespace NUMINAMATH_GPT_basketball_scores_l2420_242068

theorem basketball_scores :
  (∃ P : Finset ℕ, P = { P | ∃ x : ℕ, x ∈ (Finset.range 8) ∧ P = x + 14 } ∧ P.card = 8) :=
by
  sorry

end NUMINAMATH_GPT_basketball_scores_l2420_242068


namespace NUMINAMATH_GPT_total_money_received_l2420_242085

-- Define the conditions
def total_puppies : ℕ := 20
def fraction_sold : ℚ := 3 / 4
def price_per_puppy : ℕ := 200

-- Define the statement to prove
theorem total_money_received : fraction_sold * total_puppies * price_per_puppy = 3000 := by
  sorry

end NUMINAMATH_GPT_total_money_received_l2420_242085


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l2420_242000

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : (x - y)^2 = 49) (h2 : x * y = -8) : x^2 + y^2 = 33 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l2420_242000


namespace NUMINAMATH_GPT_ratio_chloe_to_max_l2420_242091

/-- Chloe’s wins and Max’s wins -/
def chloe_wins : ℕ := 24
def max_wins : ℕ := 9

/-- The ratio of Chloe's wins to Max's wins is 8:3 -/
theorem ratio_chloe_to_max : (chloe_wins / Nat.gcd chloe_wins max_wins) = 8 ∧ (max_wins / Nat.gcd chloe_wins max_wins) = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_chloe_to_max_l2420_242091


namespace NUMINAMATH_GPT_cupcake_packages_l2420_242079

theorem cupcake_packages (total_cupcakes eaten_cupcakes cupcakes_per_package number_of_packages : ℕ) 
  (h1 : total_cupcakes = 18)
  (h2 : eaten_cupcakes = 8)
  (h3 : cupcakes_per_package = 2)
  (h4 : number_of_packages = (total_cupcakes - eaten_cupcakes) / cupcakes_per_package) :
  number_of_packages = 5 :=
by
  -- The proof goes here, we'll use sorry to indicate it's not needed for now.
  sorry

end NUMINAMATH_GPT_cupcake_packages_l2420_242079


namespace NUMINAMATH_GPT_employee_y_payment_l2420_242086

variable (x y : ℝ)

def total_payment (x y : ℝ) : ℝ := x + y
def x_payment (y : ℝ) : ℝ := 1.20 * y

theorem employee_y_payment : (total_payment x y = 638) ∧ (x = x_payment y) → y = 290 :=
by
  sorry

end NUMINAMATH_GPT_employee_y_payment_l2420_242086


namespace NUMINAMATH_GPT_number_replacement_l2420_242078

theorem number_replacement :
  ∃ x : ℝ, ( (x / (1 / 2) * x) / (x * (1 / 2) / x) = 25 ) ↔ x = 2.5 :=
by 
  sorry

end NUMINAMATH_GPT_number_replacement_l2420_242078


namespace NUMINAMATH_GPT_friends_who_dont_eat_meat_l2420_242053

-- Definitions based on conditions
def number_of_friends : Nat := 10
def burgers_per_friend : Nat := 3
def buns_per_pack : Nat := 8
def packs_of_buns : Nat := 3
def friends_dont_eat_meat : Nat := 1
def friends_dont_eat_bread : Nat := 1

-- Total number of buns Alex plans to buy
def total_buns : Nat := buns_per_pack * packs_of_buns

-- Calculation of friends needing buns
def friends_needing_buns : Nat := number_of_friends - friends_dont_eat_meat - friends_dont_eat_bread

-- Total buns needed
def buns_needed : Nat := friends_needing_buns * burgers_per_friend

theorem friends_who_dont_eat_meat :
  buns_needed = total_buns -> friends_dont_eat_meat = 1 := by
  sorry

end NUMINAMATH_GPT_friends_who_dont_eat_meat_l2420_242053


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2420_242024

-- Definitions of the sets A and B
def A : Set ℝ := { x | x^2 + 2*x - 3 < 0 }
def B : Set ℝ := { x | |x - 1| < 2 }

-- The statement to prove their intersection
theorem intersection_of_A_and_B : A ∩ B = { x | -1 < x ∧ x < 1 } :=
by 
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2420_242024


namespace NUMINAMATH_GPT_centroid_path_is_ellipse_l2420_242020

theorem centroid_path_is_ellipse
  (b r : ℝ)
  (C : ℝ → ℝ × ℝ)
  (H1 : ∃ t θ, C t = (r * Real.cos θ, r * Real.sin θ))
  (G : ℝ → ℝ × ℝ)
  (H2 : ∀ t, G t = (1 / 3 * (b + (C t).fst), 1 / 3 * ((C t).snd))) :
  ∃ a c : ℝ, ∀ t, (G t).fst^2 / a^2 + (G t).snd^2 / c^2 = 1 :=
sorry

end NUMINAMATH_GPT_centroid_path_is_ellipse_l2420_242020


namespace NUMINAMATH_GPT_centroid_distance_l2420_242095

-- Define the given conditions and final goal
theorem centroid_distance (a b c p q r : ℝ) 
  (ha : a ≠ 0)  (hb : b ≠ 0)  (hc : c ≠ 0)
  (centroid : p = a / 3 ∧ q = b / 3 ∧ r = c / 3) 
  (plane_distance : (1 / (1 / a^2 + 1 / b^2 + 1 / c^2).sqrt) = 2) :
  (1 / p^2 + 1 / q^2 + 1 / r^2) = 2.25 := 
by 
  -- Start proof here
  sorry

end NUMINAMATH_GPT_centroid_distance_l2420_242095


namespace NUMINAMATH_GPT_anthony_more_than_mabel_l2420_242018

noncomputable def transactions := 
  let M := 90  -- Mabel's transactions
  let J := 82  -- Jade's transactions
  let C := J - 16  -- Cal's transactions
  let A := (3 / 2) * C  -- Anthony's transactions
  let P := ((A - M) / M) * 100 -- Percentage more transactions Anthony handled than Mabel
  P

theorem anthony_more_than_mabel : transactions = 10 := by
  sorry

end NUMINAMATH_GPT_anthony_more_than_mabel_l2420_242018


namespace NUMINAMATH_GPT_find_a_l2420_242034

theorem find_a (a : ℝ) (p : ℕ → ℝ) (h : ∀ k, k = 1 ∨ k = 2 ∨ k = 3 → p k = a * (1 / 2) ^ k)
  (prob_sum : a * (1 / 2 + (1 / 2) ^ 2 + (1 / 2) ^ 3) = 1) : a = 8 / 7 :=
sorry

end NUMINAMATH_GPT_find_a_l2420_242034


namespace NUMINAMATH_GPT_jasmine_max_stickers_l2420_242006

-- Given conditions and data
def sticker_cost : ℝ := 0.75
def jasmine_budget : ℝ := 10.0

-- Proof statement
theorem jasmine_max_stickers : ∃ n : ℕ, (n : ℝ) * sticker_cost ≤ jasmine_budget ∧ (∀ m : ℕ, (m > n) → (m : ℝ) * sticker_cost > jasmine_budget) :=
sorry

end NUMINAMATH_GPT_jasmine_max_stickers_l2420_242006


namespace NUMINAMATH_GPT_find_number_l2420_242012

theorem find_number (x : ℝ) (h : x - (3/5) * x = 50) : x = 125 := by
  sorry

end NUMINAMATH_GPT_find_number_l2420_242012


namespace NUMINAMATH_GPT_meat_needed_for_40_hamburgers_l2420_242073

theorem meat_needed_for_40_hamburgers (meat_per_10_hamburgers : ℕ) (hamburgers_needed : ℕ) (meat_per_hamburger : ℚ) (total_meat_needed : ℚ) :
  meat_per_10_hamburgers = 5 ∧ hamburgers_needed = 40 ∧
  meat_per_hamburger = meat_per_10_hamburgers / 10 ∧
  total_meat_needed = meat_per_hamburger * hamburgers_needed → 
  total_meat_needed = 20 := by
  sorry

end NUMINAMATH_GPT_meat_needed_for_40_hamburgers_l2420_242073


namespace NUMINAMATH_GPT_donna_soda_crates_l2420_242005

def soda_crates (bridge_limit : ℕ) (truck_empty : ℕ) (crate_weight : ℕ) (dryer_weight : ℕ) (num_dryers : ℕ) (truck_loaded : ℕ) (produce_ratio : ℕ) : ℕ :=
  sorry

theorem donna_soda_crates :
  soda_crates 20000 12000 50 3000 3 24000 2 = 20 :=
sorry

end NUMINAMATH_GPT_donna_soda_crates_l2420_242005


namespace NUMINAMATH_GPT_V3_is_correct_l2420_242070

-- Definitions of the polynomial and Horner's method applied at x = -4
def f (x : ℤ) : ℤ := 3*x^6 + 5*x^5 + 6*x^4 + 79*x^3 - 8*x^2 + 35*x + 12

def V_3_value : ℤ := 
  let v0 := -4
  let v1 := v0 * 3 + 5
  let v2 := v0 * v1 + 6
  v0 * v2 + 79

theorem V3_is_correct : V_3_value = -57 := 
  by sorry

end NUMINAMATH_GPT_V3_is_correct_l2420_242070


namespace NUMINAMATH_GPT_gymnastics_average_people_per_team_l2420_242082

def average_people_per_team (boys girls teams : ℕ) : ℕ :=
  (boys + girls) / teams

theorem gymnastics_average_people_per_team:
  average_people_per_team 83 77 4 = 40 :=
by
  sorry

end NUMINAMATH_GPT_gymnastics_average_people_per_team_l2420_242082


namespace NUMINAMATH_GPT_range_of_k_l2420_242080

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (x - 1) / (x - 2) = k / (x - 2) + 2 ∧ x ≥ 0 ∧ x ≠ 2) ↔ (k ≤ 3 ∧ k ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l2420_242080


namespace NUMINAMATH_GPT_range_of_a_l2420_242038

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 1 then a^x else (4 - a/2)*x + 2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, x1 < x2 → f a x1 ≤ f a x2) ↔ (4 ≤ a ∧ a < 8) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2420_242038


namespace NUMINAMATH_GPT_percentage_of_b_l2420_242032

variable (a b c p : ℝ)

theorem percentage_of_b (h1 : 0.06 * a = 12) (h2 : p * b = 6) (h3 : c = b / a) : 
  p = 6 / (200 * c) := by
  sorry

end NUMINAMATH_GPT_percentage_of_b_l2420_242032


namespace NUMINAMATH_GPT_daily_shampoo_usage_l2420_242060

theorem daily_shampoo_usage
  (S : ℝ)
  (h1 : ∀ t : ℝ, t = 14 → 14 * S + 14 * (S / 2) = 21) :
  S = 1 := by
  sorry

end NUMINAMATH_GPT_daily_shampoo_usage_l2420_242060


namespace NUMINAMATH_GPT_find_OQ_l2420_242023
-- Import the required math libarary

-- Define points on a line with the given distances
def O := 0
def A (a : ℝ) := 2 * a
def B (b : ℝ) := 4 * b
def C (c : ℝ) := 5 * c
def D (d : ℝ) := 7 * d

-- Given P between B and C such that ratio condition holds
def P (a b c d x : ℝ) := 
  B b ≤ x ∧ x ≤ C c ∧ 
  (A a - x) * (x - C c) = (B b - x) * (x - D d)

-- Calculate Q based on given ratio condition
def Q (b c d y : ℝ) := 
  C c ≤ y ∧ y ≤ D d ∧ 
  (C c - y) * (y - D d) = (B b - C c) * (C c - D d)

-- Main Proof Statement to prove OQ
theorem find_OQ (a b c d y : ℝ) 
  (hP : ∃ x, P a b c d x)
  (hQ : ∃ y, Q b c d y) :
  y = (14 * c * d - 10 * b * c) / (5 * c - 7 * d) := by
  sorry

end NUMINAMATH_GPT_find_OQ_l2420_242023


namespace NUMINAMATH_GPT_max_sum_e3_f3_g3_h3_i3_l2420_242067

theorem max_sum_e3_f3_g3_h3_i3 (e f g h i : ℝ) (h_cond : e^4 + f^4 + g^4 + h^4 + i^4 = 5) :
  e^3 + f^3 + g^3 + h^3 + i^3 ≤ 5^(3/4) :=
sorry

end NUMINAMATH_GPT_max_sum_e3_f3_g3_h3_i3_l2420_242067


namespace NUMINAMATH_GPT_thomas_monthly_earnings_l2420_242027

def weekly_earnings : ℕ := 4550
def weeks_in_month : ℕ := 4
def monthly_earnings : ℕ := weekly_earnings * weeks_in_month

theorem thomas_monthly_earnings : monthly_earnings = 18200 := by
  sorry

end NUMINAMATH_GPT_thomas_monthly_earnings_l2420_242027


namespace NUMINAMATH_GPT_geometric_sequence_b_mn_theorem_l2420_242013

noncomputable def geometric_sequence_b_mn (b : ℕ → ℝ) (c d : ℝ) (m n : ℕ) 
  (h_b : ∀ (k : ℕ), b k > 0)
  (h_seq : ∃ q : ℝ, (q ≠ 0) ∧ ∀ k : ℕ, b k = b 1 * q ^ (k - 1))
  (h_m : b m = c)
  (h_n : b n = d)
  (h_cond : n - m ≥ 2) 
  (h_nm_pos : m > 0 ∧ n > 0): Prop :=
  b (m + n) = (d ^ n / c ^ m) ^ (1 / (n - m))

-- We skip the proof using sorry.
theorem geometric_sequence_b_mn_theorem 
  (b : ℕ → ℝ) (c d : ℝ) (m n : ℕ)
  (h_b : ∀ (k : ℕ), b k > 0)
  (h_seq : ∃ q : ℝ, (q ≠ 0) ∧ ∀ k : ℕ, b k = b 1 * q ^ (k - 1))
  (h_m : b m = c)
  (h_n : b n = d)
  (h_cond : n - m ≥ 2)
  (h_nm_pos : m > 0 ∧ n > 0) : 
  b (m + n) = (d ^ n / c ^ m) ^ (1 / (n - m)) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_b_mn_theorem_l2420_242013


namespace NUMINAMATH_GPT_payment_relationship_l2420_242025

noncomputable def payment_amount (x : ℕ) (price_per_book : ℕ) (discount_percent : ℕ) : ℕ :=
  if x > 20 then ((x - 20) * (price_per_book * (100 - discount_percent) / 100) + 20 * price_per_book) else x * price_per_book

theorem payment_relationship (x : ℕ) (h : x > 20) : payment_amount x 25 20 = 20 * x + 100 := 
by
  sorry

end NUMINAMATH_GPT_payment_relationship_l2420_242025


namespace NUMINAMATH_GPT_abs_neg_five_l2420_242009

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_five_l2420_242009


namespace NUMINAMATH_GPT_total_potatoes_l2420_242092

theorem total_potatoes (monday_to_friday_potatoes : ℕ) (double_potatoes : ℕ) 
(lunch_potatoes_mon_fri : ℕ) (lunch_potatoes_weekend : ℕ)
(dinner_potatoes_mon_fri : ℕ) (dinner_potatoes_weekend : ℕ)
(h1 : monday_to_friday_potatoes = 5)
(h2 : double_potatoes = 10)
(h3 : lunch_potatoes_mon_fri = 25)
(h4 : lunch_potatoes_weekend = 20)
(h5 : dinner_potatoes_mon_fri = 40)
(h6 : dinner_potatoes_weekend = 26)
  : monday_to_friday_potatoes * 5 + double_potatoes * 2 + dinner_potatoes_mon_fri * 5 + (double_potatoes + 3) * 2 = 111 := 
sorry

end NUMINAMATH_GPT_total_potatoes_l2420_242092


namespace NUMINAMATH_GPT_circle_symmetric_line_l2420_242052

theorem circle_symmetric_line (a b : ℝ) 
  (h1 : ∃ x y, x^2 + y^2 - 4 * x + 2 * y + 1 = 0)
  (h2 : ∀ x y, (x, y) = (2, -1))
  (h3 : 2 * a + 2 * b - 1 = 0) :
  ab ≤ 1 / 16 := sorry

end NUMINAMATH_GPT_circle_symmetric_line_l2420_242052


namespace NUMINAMATH_GPT_altitude_length_l2420_242076

theorem altitude_length {s t : ℝ} 
  (A B C : ℝ × ℝ) 
  (hA : A = (-s, s^2))
  (hB : B = (s, s^2))
  (hC : C = (t, t^2))
  (h_parabola_A : A.snd = (A.fst)^2)
  (h_parabola_B : B.snd = (B.fst)^2)
  (h_parabola_C : C.snd = (C.fst)^2)
  (hyp_parallel : A.snd = B.snd)
  (right_triangle : (t + s) * (t - s) + (t^2 - s^2)^2 = 0) :
  (s^2 - (t^2)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_altitude_length_l2420_242076


namespace NUMINAMATH_GPT_rabbit_travel_time_l2420_242058

theorem rabbit_travel_time (distance : ℕ) (speed : ℕ) (time_in_minutes : ℕ) 
  (h_distance : distance = 3) 
  (h_speed : speed = 6) 
  (h_time_eqn : time_in_minutes = (distance * 60) / speed) : 
  time_in_minutes = 30 := 
by 
  sorry

end NUMINAMATH_GPT_rabbit_travel_time_l2420_242058


namespace NUMINAMATH_GPT_second_trial_amount_691g_l2420_242072

theorem second_trial_amount_691g (low high : ℝ) (h_range : low = 500) (h_high : high = 1000) (h_method : ∃ x, x = 0.618) : 
  high - 0.618 * (high - low) = 691 :=
by
  sorry

end NUMINAMATH_GPT_second_trial_amount_691g_l2420_242072


namespace NUMINAMATH_GPT_polynomial_roots_l2420_242044

theorem polynomial_roots :
  (∀ x, x^3 - 3 * x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := 
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_l2420_242044


namespace NUMINAMATH_GPT_middle_managers_sample_count_l2420_242090

def employees_total : ℕ := 1000
def managers_middle_total : ℕ := 150
def sample_total : ℕ := 200

theorem middle_managers_sample_count :
  sample_total * managers_middle_total / employees_total = 30 := by
  sorry

end NUMINAMATH_GPT_middle_managers_sample_count_l2420_242090
