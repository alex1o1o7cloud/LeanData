import Mathlib

namespace ellipse_x_intersection_l321_32179

theorem ellipse_x_intersection 
  (F₁ F₂ : ℝ × ℝ)
  (origin : ℝ × ℝ)
  (x_intersect : ℝ × ℝ)
  (h₁ : F₁ = (0, 3))
  (h₂ : F₂ = (4, 0))
  (h₃ : origin = (0, 0))
  (h₄ : ∀ P : ℝ × ℝ, (dist P F₁ + dist P F₂ = 7) ↔ (P = origin ∨ P = x_intersect))
  : x_intersect = (56 / 11, 0) := sorry

end ellipse_x_intersection_l321_32179


namespace distance_between_front_contestants_l321_32161

noncomputable def position_a (pd : ℝ) : ℝ := pd - 10
def position_b (pd : ℝ) : ℝ := pd - 40
def position_c (pd : ℝ) : ℝ := pd - 60
def position_d (pd : ℝ) : ℝ := pd

theorem distance_between_front_contestants (pd : ℝ):
  position_d pd - position_a pd = 10 :=
by
  sorry

end distance_between_front_contestants_l321_32161


namespace students_behind_Yoongi_l321_32131

theorem students_behind_Yoongi 
  (total_students : ℕ) 
  (position_Jungkook : ℕ) 
  (students_between : ℕ) 
  (position_Yoongi : ℕ) : 
  total_students = 20 → 
  position_Jungkook = 1 → 
  students_between = 5 → 
  position_Yoongi = position_Jungkook + students_between + 1 → 
  (total_students - position_Yoongi) = 13 :=
by
  sorry

end students_behind_Yoongi_l321_32131


namespace taxi_fare_proportional_l321_32142

theorem taxi_fare_proportional (fare_50 : ℝ) (distance_50 distance_70 : ℝ) (proportional : Prop) (h_fare_50 : fare_50 = 120) (h_distance_50 : distance_50 = 50) (h_distance_70 : distance_70 = 70) :
  fare_70 = 168 :=
by {
  sorry
}

end taxi_fare_proportional_l321_32142


namespace number_of_true_propositions_l321_32176

theorem number_of_true_propositions : 
  (∃ x y : ℝ, (x * y = 1) ↔ (x = y⁻¹ ∨ y = x⁻¹)) ∧
  (¬(∀ x : ℝ, (x > -3) → x^2 - x - 6 ≤ 0)) ∧
  (¬(∀ a b : ℝ, (a > b) → (a^2 < b^2))) ∧
  (¬(∀ x : ℝ, (x - 1/x > 0) → (x > -1))) →
  True := by
  sorry

end number_of_true_propositions_l321_32176


namespace sum_first_ten_terms_arithmetic_l321_32104

def arithmetic_sequence_sum (a₁ : ℤ) (n : ℕ) (d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_ten_terms_arithmetic (a₁ a₂ a₆ d : ℤ) 
  (h1 : a₁ = -2) 
  (h2 : a₂ + a₆ = 2) 
  (common_diff : d = 1) :
  arithmetic_sequence_sum a₁ 10 d = 25 :=
by
  rw [h1, common_diff]
  sorry

end sum_first_ten_terms_arithmetic_l321_32104


namespace sum_of_cubes_l321_32174

open Real

theorem sum_of_cubes (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
(h_eq : (a^3 + 6) / a = (b^3 + 6) / b ∧ (b^3 + 6) / b = (c^3 + 6) / c) :
  a^3 + b^3 + c^3 = -18 := 
by sorry

end sum_of_cubes_l321_32174


namespace fraction_identity_l321_32134

theorem fraction_identity (a b : ℝ) (h : a ≠ b) (h₁ : (a + b) / (a - b) = 3) : a / b = 2 := by
  sorry

end fraction_identity_l321_32134


namespace min_rounds_needed_l321_32193

-- Defining the number of players
def num_players : ℕ := 10

-- Defining the number of matches each player plays per round
def matches_per_round (n : ℕ) : ℕ := n / 2

-- Defining the scoring system
def win_points : ℝ := 1
def draw_points : ℝ := 0.5
def loss_points : ℝ := 0

-- Defining the total number of rounds needed for a clear winner to emerge
def min_rounds_for_winner : ℕ := 7

-- Theorem stating the minimum number of rounds required
theorem min_rounds_needed :
  ∀ (n : ℕ), n = num_players → (∃ r : ℕ, r = min_rounds_for_winner) :=
by
  intros n hn
  existsi min_rounds_for_winner
  sorry

end min_rounds_needed_l321_32193


namespace max_value_of_3cosx_minus_sinx_l321_32164

noncomputable def max_cosine_expression : ℝ :=
  Real.sqrt 10

theorem max_value_of_3cosx_minus_sinx : 
  ∃ x : ℝ, ∀ x : ℝ, 3 * Real.cos x - Real.sin x ≤ Real.sqrt 10 := 
by {
  sorry
}

end max_value_of_3cosx_minus_sinx_l321_32164


namespace winning_votes_cast_l321_32116

theorem winning_votes_cast (V : ℝ) (h1 : 0.40 * V = 280) : 0.70 * V = 490 :=
by
  sorry

end winning_votes_cast_l321_32116


namespace at_least_one_two_prob_l321_32107

-- Definitions and conditions corresponding to the problem
def total_outcomes (n : ℕ) : ℕ := n * n
def no_twos_outcomes (n : ℕ) : ℕ := (n - 1) * (n - 1)

-- The probability calculation
def probability_at_least_one_two (n : ℕ) : ℚ := 
  let tot_outs := total_outcomes n
  let no_twos := no_twos_outcomes n
  (tot_outs - no_twos : ℚ) / tot_outs

-- Our main theorem to be proved
theorem at_least_one_two_prob : 
  probability_at_least_one_two 6 = 11 / 36 := 
by
  sorry

end at_least_one_two_prob_l321_32107


namespace sum_of_tens_and_units_of_product_is_zero_l321_32185

-- Define the repeating patterns used to create the 999-digit numbers
def pattern1 : ℕ := 400
def pattern2 : ℕ := 606

-- Function to construct a 999-digit number by repeating a 3-digit pattern 333 times
def repeat_pattern (pat : ℕ) (times : ℕ) : ℕ := pat * (10 ^ (3 * times - 3))

-- Define the two 999-digit numbers
def num1 : ℕ := repeat_pattern pattern1 333
def num2 : ℕ := repeat_pattern pattern2 333

-- Function to compute the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Function to compute the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Define the product of the two numbers
def product : ℕ := num1 * num2

-- Function to compute the sum of the tens and units digits of a number
def sum_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

-- The statement to be proven
theorem sum_of_tens_and_units_of_product_is_zero :
  sum_digits product = 0 := 
sorry -- Proof steps are omitted

end sum_of_tens_and_units_of_product_is_zero_l321_32185


namespace rohit_distance_from_start_l321_32149

-- Define Rohit's movements
def rohit_walked_south (d: ℕ) : ℕ := d
def rohit_turned_left_walked_east (d: ℕ) : ℕ := d
def rohit_turned_left_walked_north (d: ℕ) : ℕ := d
def rohit_turned_right_walked_east (d: ℕ) : ℕ := d

-- Rohit's total movement in east direction
def total_distance_moved_east (d1 d2 : ℕ) : ℕ :=
  rohit_turned_left_walked_east d1 + rohit_turned_right_walked_east d2

-- Prove the distance from the starting point is 35 meters
theorem rohit_distance_from_start : 
  total_distance_moved_east 20 15 = 35 :=
by
  sorry

end rohit_distance_from_start_l321_32149


namespace part_whole_ratio_l321_32125

theorem part_whole_ratio (N x : ℕ) (hN : N = 160) (hx : x + 4 = N / 4 - 4) :
  x / N = 1 / 5 :=
  sorry

end part_whole_ratio_l321_32125


namespace find_n_l321_32180

theorem find_n (n : ℕ) (h : 20 * n = Nat.factorial (n - 1)) : n = 6 :=
by {
  sorry
}

end find_n_l321_32180


namespace sum_of_consecutive_integers_with_product_812_l321_32141

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l321_32141


namespace freezer_temperature_l321_32184

theorem freezer_temperature 
  (refrigeration_temp : ℝ)
  (freezer_temp_diff : ℝ)
  (h1 : refrigeration_temp = 4)
  (h2 : freezer_temp_diff = 22)
  : (refrigeration_temp - freezer_temp_diff) = -18 :=
by 
  sorry

end freezer_temperature_l321_32184


namespace expression_equality_l321_32140

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : (1 / x + 1 / y) = 1 :=
by
  sorry

end expression_equality_l321_32140


namespace domain_of_f_l321_32102

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f y = x}

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

theorem domain_of_f : domain f = {x | x > 1} := sorry

end domain_of_f_l321_32102


namespace functionMachine_output_l321_32169

-- Define the function machine according to the specified conditions
def functionMachine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 30 then step1 - 4 else step1
  let step3 := if step2 <= 20 then step2 + 8 else step2 - 5
  step3

-- Statement: Prove that the functionMachine applied to 10 yields 25
theorem functionMachine_output : functionMachine 10 = 25 :=
  by
    sorry

end functionMachine_output_l321_32169


namespace initial_average_age_of_students_l321_32188

theorem initial_average_age_of_students 
(A : ℕ) 
(h1 : 23 * A + 46 = (A + 1) * 24) : 
  A = 22 :=
by
  sorry

end initial_average_age_of_students_l321_32188


namespace evaluate_expression_l321_32119

noncomputable def expression (a b : ℕ) := (a + b)^2 - (a - b)^2

theorem evaluate_expression:
  expression (5^500) (6^501) = 24 * 30^500 := by
sorry

end evaluate_expression_l321_32119


namespace inv_88_mod_89_l321_32189

theorem inv_88_mod_89 : (88 * 88) % 89 = 1 := by
  sorry

end inv_88_mod_89_l321_32189


namespace solve_quadratic_roots_l321_32110

theorem solve_quadratic_roots : ∀ x : ℝ, (x - 1)^2 = 1 → (x = 2 ∨ x = 0) :=
by
  sorry

end solve_quadratic_roots_l321_32110


namespace minimum_value_of_f_l321_32171

noncomputable def f (a b x : ℝ) := (a * x + b) / (x^2 + 4)

theorem minimum_value_of_f (a b : ℝ) (h1 : f a b (-1) = 1)
  (h2 : (deriv (f a b)) (-1) = 0) : 
  ∃ (x : ℝ), f a b x = -1 / 4 := 
sorry

end minimum_value_of_f_l321_32171


namespace condition_for_M_eq_N_l321_32108

theorem condition_for_M_eq_N (a1 b1 c1 a2 b2 c2 : ℝ) 
    (h1 : a1 ≠ 0) (h2 : b1 ≠ 0) (h3 : c1 ≠ 0) 
    (h4 : a2 ≠ 0) (h5 : b2 ≠ 0) (h6 : c2 ≠ 0) :
    (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) → 
    (M = {x : ℝ | a1 * x ^ 2 + b1 * x + c1 > 0} ∧
     N = {x : ℝ | a2 * x ^ 2 + b2 * x + c2 > 0} →
    (¬ (M = N))) ∨ (¬ (N = {} ↔ (M = N))) :=
sorry

end condition_for_M_eq_N_l321_32108


namespace width_of_room_l321_32105

-- Definitions from conditions
def length : ℝ := 8
def total_cost : ℝ := 34200
def cost_per_sqm : ℝ := 900

-- Theorem stating the width of the room
theorem width_of_room : (total_cost / cost_per_sqm) / length = 4.75 := by 
  sorry

end width_of_room_l321_32105


namespace sum_of_numbers_l321_32199

theorem sum_of_numbers : 4.75 + 0.303 + 0.432 = 5.485 :=
by
  -- The proof will be filled here
  sorry

end sum_of_numbers_l321_32199


namespace tens_digit_of_8_pow_1234_l321_32118

theorem tens_digit_of_8_pow_1234 :
  (8^1234 / 10) % 10 = 0 :=
sorry

end tens_digit_of_8_pow_1234_l321_32118


namespace ice_cream_scoops_l321_32145

def scoops_of_ice_cream : ℕ := 1 -- single cone has 1 scoop

def scoops_double_cone : ℕ := 2 * scoops_of_ice_cream -- double cone has two times the scoops of a single cone

def scoops_banana_split : ℕ := 3 * scoops_of_ice_cream -- banana split has three times the scoops of a single cone

def scoops_waffle_bowl : ℕ := scoops_banana_split + 1 -- waffle bowl has one more scoop than banana split

def total_scoops : ℕ := scoops_of_ice_cream + scoops_double_cone + scoops_banana_split + scoops_waffle_bowl

theorem ice_cream_scoops : total_scoops = 10 :=
by
  sorry

end ice_cream_scoops_l321_32145


namespace largest_percentage_increase_l321_32190

def students_2003 := 80
def students_2004 := 88
def students_2005 := 94
def students_2006 := 106
def students_2007 := 130

theorem largest_percentage_increase :
  let incr_03_04 := (students_2004 - students_2003) / students_2003 * 100
  let incr_04_05 := (students_2005 - students_2004) / students_2004 * 100
  let incr_05_06 := (students_2006 - students_2005) / students_2005 * 100
  let incr_06_07 := (students_2007 - students_2006) / students_2006 * 100
  incr_06_07 > incr_03_04 ∧
  incr_06_07 > incr_04_05 ∧
  incr_06_07 > incr_05_06 :=
by
  -- Proof goes here
  sorry

end largest_percentage_increase_l321_32190


namespace not_all_x_ne_1_imp_x2_ne_0_l321_32103

theorem not_all_x_ne_1_imp_x2_ne_0 : ¬ (∀ x : ℝ, x ≠ 1 → x^2 - 1 ≠ 0) :=
sorry

end not_all_x_ne_1_imp_x2_ne_0_l321_32103


namespace trigonometric_order_l321_32147

theorem trigonometric_order :
  (Real.sin 2 > Real.sin 1) ∧
  (Real.sin 1 > Real.sin 3) ∧
  (Real.sin 3 > Real.sin 4) := 
by
  sorry

end trigonometric_order_l321_32147


namespace interval_of_monotonic_increase_l321_32158

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem interval_of_monotonic_increase :
  (∃ α : ℝ, power_function α 2 = 4) →
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → power_function 2 x ≤ power_function 2 y) :=
by
  intro h
  sorry

end interval_of_monotonic_increase_l321_32158


namespace num_sets_of_consecutive_integers_sum_to_30_l321_32152

theorem num_sets_of_consecutive_integers_sum_to_30 : 
  let S_n (n a : ℕ) := (n * (2 * a + n - 1)) / 2 
  ∃! (s : ℕ), s = 3 ∧ ∀ n, n ≥ 2 → ∃ a, S_n n a = 30 :=
by
  sorry

end num_sets_of_consecutive_integers_sum_to_30_l321_32152


namespace toms_balloons_l321_32127

-- Define the original number of balloons that Tom had
def original_balloons : ℕ := 30

-- Define the number of balloons that Tom gave to Fred
def balloons_given_to_Fred : ℕ := 16

-- Define the number of balloons that Tom has now
def balloons_left : ℕ := original_balloons - balloons_given_to_Fred

-- The theorem to prove
theorem toms_balloons : balloons_left = 14 := 
by
  -- The proof steps would go here
  sorry

end toms_balloons_l321_32127


namespace area_of_trapezoid_EFGH_l321_32126

noncomputable def length (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

noncomputable def height_FG : ℝ :=
  6 - 2

noncomputable def area_trapezoid (E F G H : ℝ × ℝ) : ℝ :=
  let base1 := length E F
  let base2 := length G H
  let height := height_FG
  1/2 * (base1 + base2) * height

theorem area_of_trapezoid_EFGH :
  area_trapezoid (0, 0) (2, -3) (6, 0) (6, 4) = 2 * (Real.sqrt 13 + 4) :=
by
  sorry

end area_of_trapezoid_EFGH_l321_32126


namespace max_rubles_l321_32194

theorem max_rubles (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2099) :
  (∃ k, n = 99 * k) → 
  31 ≤
    (if n % 1 = 0 then 1 else 0) +
    (if n % 3 = 0 then 3 else 0) +
    (if n % 5 = 0 then 5 else 0) +
    (if n % 7 = 0 then 7 else 0) +
    (if n % 9 = 0 then 9 else 0) +
    (if n % 11 = 0 then 11 else 0) :=
sorry

end max_rubles_l321_32194


namespace factor_3m2n_12mn_12n_factor_abx2_4y2ba_calculate_result_l321_32168

-- Proof 1: Factorize 3m^2 n - 12mn + 12n
theorem factor_3m2n_12mn_12n (m n : ℤ) : 3 * m^2 * n - 12 * m * n + 12 * n = 3 * n * (m - 2)^2 :=
by sorry

-- Proof 2: Factorize (a-b)x^2 + 4y^2(b-a)
theorem factor_abx2_4y2ba (a b x y : ℤ) : (a - b) * x^2 + 4 * y^2 * (b - a) = (a - b) * (x + 2 * y) * (x - 2 * y) :=
by sorry

-- Proof 3: Calculate 2023 * 51^2 - 2023 * 49^2
theorem calculate_result : 2023 * 51^2 - 2023 * 49^2 = 404600 :=
by sorry

end factor_3m2n_12mn_12n_factor_abx2_4y2ba_calculate_result_l321_32168


namespace total_raisins_l321_32129

noncomputable def yellow_raisins : ℝ := 0.3
noncomputable def black_raisins : ℝ := 0.4
noncomputable def red_raisins : ℝ := 0.5

theorem total_raisins : yellow_raisins + black_raisins + red_raisins = 1.2 := by
  sorry

end total_raisins_l321_32129


namespace difference_between_two_numbers_l321_32182

theorem difference_between_two_numbers (a : ℕ) (b : ℕ)
  (h1 : a + b = 24300)
  (h2 : b = 100 * a) :
  b - a = 23760 :=
by {
  sorry
}

end difference_between_two_numbers_l321_32182


namespace specialist_time_l321_32183

def hospital_bed_charge (days : ℕ) (rate : ℕ) : ℕ := days * rate

def total_known_charges (bed_charge : ℕ) (ambulance_charge : ℕ) : ℕ := bed_charge + ambulance_charge

def specialist_minutes (total_bill : ℕ) (known_charges : ℕ) (spec_rate_per_hour : ℕ) : ℕ := 
  ((total_bill - known_charges) / spec_rate_per_hour) * 60 / 2

theorem specialist_time (days : ℕ) (bed_rate : ℕ) (ambulance_charge : ℕ) (spec_rate_per_hour : ℕ) 
(total_bill : ℕ) (known_charges := total_known_charges (hospital_bed_charge days bed_rate) ambulance_charge)
(hospital_days := 3) (bed_charge_per_day := 900) (specialist_rate := 250) 
(ambulance_cost := 1800) (total_cost := 4625) :
  specialist_minutes total_cost known_charges specialist_rate = 15 :=
sorry

end specialist_time_l321_32183


namespace value_of_b_l321_32148

noncomputable def function_bounds := 
  ∃ (k b : ℝ), (∀ (x : ℝ), (-3 ≤ x ∧ x ≤ 1) → (-1 ≤ k * x + b ∧ k * x + b ≤ 8)) ∧ (b = 5 / 4 ∨ b = 23 / 4)

theorem value_of_b : function_bounds :=
by
  sorry

end value_of_b_l321_32148


namespace balls_initial_count_90_l321_32166

theorem balls_initial_count_90 (n : ℕ) (total_initial_balls : ℕ)
  (initial_green_balls : ℕ := 3 * n)
  (initial_yellow_balls : ℕ := 7 * n)
  (remaining_green_balls : ℕ := initial_green_balls - 9)
  (remaining_yellow_balls : ℕ := initial_yellow_balls - 9)
  (h_ratio_1 : initial_green_balls = 3 * n)
  (h_ratio_2 : initial_yellow_balls = 7 * n)
  (h_ratio_3 : remaining_green_balls * 3 = remaining_yellow_balls * 1)
  (h_total : total_initial_balls = initial_green_balls + initial_yellow_balls)
  : total_initial_balls = 90 := 
by
  sorry

end balls_initial_count_90_l321_32166


namespace math_problem_l321_32132

noncomputable def proof_problem (k : ℝ) (a b k1 k2 : ℝ) : Prop :=
  (a*b) = 7/k ∧ (a + b) = (k-1)/k ∧ (k1^2 - 18*k1 + 1) = 0 ∧ (k2^2 - 18*k2 + 1) = 0 ∧ 
  (a/b + b/a = 3/7) → (k1/k2 + k2/k1 = 322)

theorem math_problem (k a b k1 k2 : ℝ) : proof_problem k a b k1 k2 :=
by
  sorry

end math_problem_l321_32132


namespace hamburgers_left_over_l321_32130

theorem hamburgers_left_over (total_hamburgers served_hamburgers : ℕ) (h1 : total_hamburgers = 9) (h2 : served_hamburgers = 3) :
    total_hamburgers - served_hamburgers = 6 := by
  sorry

end hamburgers_left_over_l321_32130


namespace sum_of_homothety_coeffs_geq_4_l321_32157

theorem sum_of_homothety_coeffs_geq_4 (a : ℕ → ℝ)
  (h_pos : ∀ i, 0 < a i)
  (h_less_one : ∀ i, a i < 1)
  (h_sum_cubes : ∑' i, (a i)^3 = 1) :
  (∑' i, a i) ≥ 4 := sorry

end sum_of_homothety_coeffs_geq_4_l321_32157


namespace fraction_problem_l321_32163

theorem fraction_problem (x : ℝ) (h : (3 / 4) * (1 / 2) * x * 5000 = 750.0000000000001) : 
  x = 0.4 :=
sorry

end fraction_problem_l321_32163


namespace root_division_7_pow_l321_32114

theorem root_division_7_pow : 
  ( (7 : ℝ) ^ (1 / 4) / (7 ^ (1 / 7)) = 7 ^ (3 / 28) ) :=
sorry

end root_division_7_pow_l321_32114


namespace polygon_sides_l321_32144

theorem polygon_sides (n : ℕ) (h1 : ∀ i < n, (n > 2) → (150 * n = (n - 2) * 180)) : n = 12 :=
by
  -- Proof omitted
  sorry

end polygon_sides_l321_32144


namespace find_ab_l321_32172

-- Define the polynomials involved
def poly1 (x : ℝ) (a b : ℝ) : ℝ := a * x^4 + b * x^2 + 1
def poly2 (x : ℝ) : ℝ := x^2 - x - 2

-- Define the roots of the second polynomial
def root1 : ℝ := 2
def root2 : ℝ := -1

-- State the theorem to prove
theorem find_ab (a b : ℝ) :
  poly1 root1 a b = 0 ∧ poly1 root2 a b = 0 → a = 1/4 ∧ b = -5/4 :=
by
  -- Skipping the proof here
  sorry

end find_ab_l321_32172


namespace inequality_solution_sets_l321_32175

theorem inequality_solution_sets (a : ℝ) (h : a > 1) :
  ∀ x : ℝ, ((a = 2 → (x ≠ 1 → (a-1)*x*x - a*x + 1 > 0)) ∧
            (1 < a ∧ a < 2 → (x < 1 ∨ x > 1/(a-1) → (a-1)*x*x - a*x + 1 > 0)) ∧
            (a > 2 → (x < 1/(a-1) ∨ x > 1 → (a-1)*x*x - a*x + 1 > 0))) :=
by
  sorry

end inequality_solution_sets_l321_32175


namespace problem1_problem2_l321_32115

noncomputable def f (x a c : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

-- Problem 1: Prove that for c = 19, the inequality f(1, a, 19) > 0 holds for -2 < a < 8
theorem problem1 (a : ℝ) : f 1 a 19 > 0 ↔ -2 < a ∧ a < 8 :=
by sorry

-- Problem 2: Given that f(x) > 0 has solution set (-1, 3), find a and c
theorem problem2 (a c : ℝ) (hx : ∀ x, -1 < x ∧ x < 3 → f x a c > 0) : 
  (a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9 :=
by sorry

end problem1_problem2_l321_32115


namespace calculate_delta_nabla_l321_32150

-- Define the operations Δ and ∇
def delta (a b : ℤ) : ℤ := 3 * a + 2 * b
def nabla (a b : ℤ) : ℤ := 2 * a + 3 * b

-- Formalize the theorem
theorem calculate_delta_nabla : delta 3 (nabla 2 1) = 23 := 
by 
  -- Placeholder for proof, not required by the question
  sorry

end calculate_delta_nabla_l321_32150


namespace find_k_l321_32123

theorem find_k (a b c k : ℤ) (g : ℤ → ℤ)
  (h₁ : g 1 = 0)
  (h₂ : 10 < g 5 ∧ g 5 < 20)
  (h₃ : 30 < g 6 ∧ g 6 < 40)
  (h₄ : 3000 * k < g 100 ∧ g 100 < 3000 * (k + 1))
  (h_g : ∀ x, g x = a * x^2 + b * x + c) :
  k = 9 :=
by
  sorry

end find_k_l321_32123


namespace inequality_proof_l321_32159

theorem inequality_proof (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z ≥ 1) : 
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by
  sorry

end inequality_proof_l321_32159


namespace count_eligible_three_digit_numbers_l321_32136

def is_eligible_digit (d : Nat) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem count_eligible_three_digit_numbers : 
  (∃ n : Nat, 100 ≤ n ∧ n < 1000 ∧
  (∀ d : Nat, d ∈ [n / 100, (n / 10) % 10, n % 10] → is_eligible_digit d)) →
  ∃ count : Nat, count = 343 :=
by
  sorry

end count_eligible_three_digit_numbers_l321_32136


namespace divisor_is_20_l321_32112

theorem divisor_is_20 (D q1 q2 q3 : ℕ) :
  (242 = D * q1 + 11) ∧
  (698 = D * q2 + 18) ∧
  (940 = D * q3 + 9) →
  D = 20 :=
by
  sorry

end divisor_is_20_l321_32112


namespace product_equals_9_l321_32120

theorem product_equals_9 :
  (1 + (1 / 1)) * (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * 
  (1 + (1 / 5)) * (1 + (1 / 6)) * (1 + (1 / 7)) * (1 + (1 / 8)) = 9 := 
by
  sorry

end product_equals_9_l321_32120


namespace count_valid_third_sides_l321_32135

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end count_valid_third_sides_l321_32135


namespace ice_cream_volume_l321_32156

theorem ice_cream_volume (r_cone h_cone r_hemisphere : ℝ) (h1 : r_cone = 3) (h2 : h_cone = 10) (h3 : r_hemisphere = 5) :
  (1 / 3 * π * r_cone^2 * h_cone + 2 / 3 * π * r_hemisphere^3) = (520 / 3) * π :=
by 
  rw [h1, h2, h3]
  norm_num
  sorry

end ice_cream_volume_l321_32156


namespace hyperbola_real_axis_length_l321_32170

variables {a b : ℝ} (ha : a > 0) (hb : b > 0) (h_asymptote_slope : b = 2 * a) (h_c : (a^2 + b^2) = 5)

theorem hyperbola_real_axis_length : 2 * a = 2 :=
by
  sorry

end hyperbola_real_axis_length_l321_32170


namespace trapezoid_other_side_length_l321_32143

theorem trapezoid_other_side_length (a h : ℕ) (A : ℕ) (b : ℕ) : 
  a = 20 → h = 13 → A = 247 → (1/2:ℚ) * (a + b) * h = A → b = 18 :=
by 
  intros h1 h2 h3 h4 
  rw [h1, h2, h3] at h4
  sorry

end trapezoid_other_side_length_l321_32143


namespace natalies_diaries_l321_32154

theorem natalies_diaries : 
  ∀ (initial_diaries : ℕ) (tripled_diaries : ℕ) (total_diaries : ℕ) (lost_diaries : ℕ) (remaining_diaries : ℕ),
  initial_diaries = 15 →
  tripled_diaries = 3 * initial_diaries →
  total_diaries = initial_diaries + tripled_diaries →
  lost_diaries = 3 * total_diaries / 5 →
  remaining_diaries = total_diaries - lost_diaries →
  remaining_diaries = 24 :=
by
  intros initial_diaries tripled_diaries total_diaries lost_diaries remaining_diaries
  intro h1 h2 h3 h4 h5
  sorry

end natalies_diaries_l321_32154


namespace original_laborers_l321_32101

theorem original_laborers (x : ℕ) : (x * 8 = (x - 3) * 14) → x = 7 :=
by
  intro h
  sorry

end original_laborers_l321_32101


namespace bird_counts_remaining_l321_32195

theorem bird_counts_remaining
  (peregrine_falcons pigeons crows sparrows : ℕ)
  (chicks_per_pigeon chicks_per_crow chicks_per_sparrow : ℕ)
  (peregrines_eat_pigeons_percent peregrines_eat_crows_percent peregrines_eat_sparrows_percent : ℝ)
  (initial_peregrine_falcons : peregrine_falcons = 12)
  (initial_pigeons : pigeons = 80)
  (initial_crows : crows = 25)
  (initial_sparrows : sparrows = 15)
  (chicks_per_pigeon_cond : chicks_per_pigeon = 8)
  (chicks_per_crow_cond : chicks_per_crow = 5)
  (chicks_per_sparrow_cond : chicks_per_sparrow = 3)
  (peregrines_eat_pigeons_percent_cond : peregrines_eat_pigeons_percent = 0.4)
  (peregrines_eat_crows_percent_cond : peregrines_eat_crows_percent = 0.25)
  (peregrines_eat_sparrows_percent_cond : peregrines_eat_sparrows_percent = 0.1)
  : 
  (peregrine_falcons = 12) ∧
  (pigeons = 48) ∧
  (crows = 19) ∧
  (sparrows = 14) :=
by
  sorry

end bird_counts_remaining_l321_32195


namespace find_z_l321_32146

theorem find_z (x y z : ℚ) (hx : x = 11) (hy : y = -8) (h : 2 * x - 3 * z = 5 * y) :
  z = 62 / 3 :=
by
  sorry

end find_z_l321_32146


namespace sets_are_equal_l321_32186

-- Defining sets A and B as per the given conditions
def setA : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 1}
def setB : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4 * b + 5}

-- Proving that set A is equal to set B
theorem sets_are_equal : setA = setB :=
by
  sorry

end sets_are_equal_l321_32186


namespace selling_price_percentage_l321_32181

-- Definitions for conditions
def ratio_cara_janet_jerry (c j je : ℕ) : Prop := 4 * (c + j + je) = 4 * c + 5 * j + 6 * je
def total_money (c j je total : ℕ) : Prop := c + j + je = total
def combined_loss (c j loss : ℕ) : Prop := c + j - loss = 36

-- The theorem statement to be proven
theorem selling_price_percentage (c j je total loss : ℕ) (h1 : ratio_cara_janet_jerry c j je) (h2 : total_money c j je total) (h3 : combined_loss c j loss)
    (h4 : total = 75) (h5 : loss = 9) : (36 * 100 / (c + j) = 80) := by
  sorry

end selling_price_percentage_l321_32181


namespace tires_should_be_swapped_l321_32160

-- Define the conditions
def front_wear_out_distance : ℝ := 25000
def rear_wear_out_distance : ℝ := 15000

-- Define the distance to swap tires
def swap_distance : ℝ := 9375

-- Theorem statement
theorem tires_should_be_swapped :
  -- The distance for both tires to wear out should be the same
  swap_distance + (front_wear_out_distance - swap_distance) * (rear_wear_out_distance / front_wear_out_distance) = rear_wear_out_distance :=
sorry

end tires_should_be_swapped_l321_32160


namespace Gemma_ordered_pizzas_l321_32153

-- Definitions of conditions
def pizza_cost : ℕ := 10
def tip : ℕ := 5
def paid_amount : ℕ := 50
def change : ℕ := 5
def total_spent : ℕ := paid_amount - change

-- Statement of the proof problem
theorem Gemma_ordered_pizzas : 
  ∃ (P : ℕ), pizza_cost * P + tip = total_spent ∧ P = 4 :=
sorry

end Gemma_ordered_pizzas_l321_32153


namespace evaluate_exp_power_l321_32109

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l321_32109


namespace definite_integral_sin8_l321_32128

-- Define the definite integral problem and the expected result in Lean.
theorem definite_integral_sin8:
  ∫ x in (Real.pi / 2)..Real.pi, (2^8 * (Real.sin x)^8) = 32 * Real.pi :=
  sorry

end definite_integral_sin8_l321_32128


namespace probability_of_color_change_l321_32197

theorem probability_of_color_change :
  let cycle_duration := 100
  let green_duration := 45
  let yellow_duration := 5
  let red_duration := 50
  let green_to_yellow_interval := 5
  let yellow_to_red_interval := 5
  let red_to_green_interval := 5
  let total_color_change_duration := green_to_yellow_interval + yellow_to_red_interval + red_to_green_interval
  let observation_probability := total_color_change_duration / cycle_duration
  observation_probability = 3 / 20 := by sorry

end probability_of_color_change_l321_32197


namespace value_of_a_l321_32162

theorem value_of_a (a b c d : ℕ) (h : (18^a) * (9^(4*a-1)) * (27^c) = (2^6) * (3^b) * (7^d)) : a = 6 :=
by
  sorry

end value_of_a_l321_32162


namespace average_price_of_pig_l321_32192

theorem average_price_of_pig :
  ∀ (total_cost : ℕ) (num_pigs num_hens : ℕ) (avg_hen_price avg_pig_price : ℕ),
    total_cost = 2100 →
    num_pigs = 5 →
    num_hens = 15 →
    avg_hen_price = 30 →
    avg_pig_price * num_pigs + avg_hen_price * num_hens = total_cost →
    avg_pig_price = 330 :=
by
  intros total_cost num_pigs num_hens avg_hen_price avg_pig_price
  intros h_total_cost h_num_pigs h_num_hens h_avg_hen_price h_eq
  rw [h_total_cost, h_num_pigs, h_num_hens, h_avg_hen_price] at h_eq
  sorry

end average_price_of_pig_l321_32192


namespace gcf_36_54_81_l321_32122

theorem gcf_36_54_81 : Nat.gcd (Nat.gcd 36 54) 81 = 9 :=
by
  -- The theorem states that the greatest common factor of 36, 54, and 81 is 9.
  sorry

end gcf_36_54_81_l321_32122


namespace problem_l321_32151

noncomputable def f : ℝ → ℝ := sorry

theorem problem :
  (∀ x : ℝ, f (x) + f (x + 2) = 0) →
  (f (1) = -2) →
  (f (2019) + f (2018) = 2) :=
by
  intro h1 h2
  sorry

end problem_l321_32151


namespace determine_a_l321_32133

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 / (3 ^ x + 1)) - a

theorem determine_a (a : ℝ) :
  (∀ x : ℝ, f a (-x) = -f a x) ↔ a = 1 :=
by
  sorry

end determine_a_l321_32133


namespace bella_bracelets_l321_32111

theorem bella_bracelets (h_beads_per_bracelet : Nat)
  (h_initial_beads : Nat) 
  (h_additional_beads : Nat) 
  (h_friends : Nat):
  h_beads_per_bracelet = 8 →
  h_initial_beads = 36 →
  h_additional_beads = 12 →
  h_friends = (h_initial_beads + h_additional_beads) / h_beads_per_bracelet →
  h_friends = 6 :=
by
  intros h_beads_per_bracelet_eq h_initial_beads_eq h_additional_beads_eq h_friends_eq
  subst_vars
  sorry

end bella_bracelets_l321_32111


namespace find_number_l321_32138

theorem find_number (x k : ℕ) (h₁ : x / k = 4) (h₂ : k = 6) : x = 24 := by
  sorry

end find_number_l321_32138


namespace max_area_rectangle_perimeter_156_l321_32173

theorem max_area_rectangle_perimeter_156 (x y : ℕ) 
  (h : 2 * (x + y) = 156) : ∃x y, x * y = 1521 :=
by
  sorry

end max_area_rectangle_perimeter_156_l321_32173


namespace domain_expression_l321_32187

-- Define the conditions for the domain of the expression
def valid_numerator (x : ℝ) : Prop := 3 * x - 6 ≥ 0
def valid_denominator (x : ℝ) : Prop := 7 - 2 * x > 0

-- Proof problem statement
theorem domain_expression (x : ℝ) : valid_numerator x ∧ valid_denominator x ↔ 2 ≤ x ∧ x < 3.5 :=
sorry

end domain_expression_l321_32187


namespace vector_subtraction_identity_l321_32167

variables (a b : ℝ)

theorem vector_subtraction_identity (a b : ℝ) :
  ((1 / 2) * a - b) - ((3 / 2) * a - 2 * b) = b - a :=
by
  sorry

end vector_subtraction_identity_l321_32167


namespace unsold_tomatoes_l321_32113

theorem unsold_tomatoes (total_harvest sold_maxwell sold_wilson : ℝ) 
(h_total_harvest : total_harvest = 245.5)
(h_sold_maxwell : sold_maxwell = 125.5)
(h_sold_wilson : sold_wilson = 78) :
(total_harvest - (sold_maxwell + sold_wilson) = 42) :=
by {
  sorry
}

end unsold_tomatoes_l321_32113


namespace area_enclosed_by_curve_and_line_l321_32191

theorem area_enclosed_by_curve_and_line :
  let f := fun x : ℝ => x^2 + 2
  let g := fun x : ℝ => 3 * x
  let A := ∫ x in (0 : ℝ)..1, (f x - g x) + ∫ x in (1 : ℝ)..2, (g x - f x)
  A = 1 := by
    sorry

end area_enclosed_by_curve_and_line_l321_32191


namespace probability_at_least_one_woman_selected_l321_32124

theorem probability_at_least_one_woman_selected:
  let men := 10
  let women := 5
  let totalPeople := men + women
  let totalSelections := Nat.choose totalPeople 4
  let menSelections := Nat.choose men 4
  let noWomenProbability := (menSelections : ℚ) / (totalSelections : ℚ)
  let atLeastOneWomanProbability := 1 - noWomenProbability
  atLeastOneWomanProbability = 11 / 13 :=
by
  sorry

end probability_at_least_one_woman_selected_l321_32124


namespace sum_of_readings_ammeters_l321_32177

variables (I1 I2 I3 I4 I5 : ℝ)

noncomputable def sum_of_ammeters (I1 I2 I3 I4 I5 : ℝ) : ℝ :=
  I1 + I2 + I3 + I4 + I5

theorem sum_of_readings_ammeters :
  I1 = 2 ∧ I2 = I1 ∧ I3 = 2 * I1 ∧ I5 = I3 + I1 ∧ I4 = (5 / 3) * I5 →
  sum_of_ammeters I1 I2 I3 I4 I5 = 24 :=
by
  sorry

end sum_of_readings_ammeters_l321_32177


namespace range_of_a_l321_32117

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x + 5 > 3 ∧ x > a ∧ x ≤ -2) ↔ a ≤ -2 :=
by
  sorry

end range_of_a_l321_32117


namespace range_of_a_l321_32106

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_l321_32106


namespace equation_of_line_through_point_with_equal_intercepts_l321_32137

-- Define a structure for a 2D point
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the problem-specific points and conditions
def A : Point := {x := 4, y := -1}

-- Define the conditions and the theorem to be proven
theorem equation_of_line_through_point_with_equal_intercepts
  (p : Point)
  (h : p = A) : 
  ∃ (a : ℝ), a ≠ 0 → (∀ (a : ℝ), ((∀ (b : ℝ), b = a → b ≠ 0 → x + y - a = 0)) ∨ (x + 4 * y = 0)) :=
sorry

end equation_of_line_through_point_with_equal_intercepts_l321_32137


namespace maximum_abc_827_l321_32139

noncomputable def maximum_abc (a b c : ℝ) := (a * b * c)

theorem maximum_abc_827 (a b c : ℝ) 
  (h1: a > 0) 
  (h2: b > 0) 
  (h3: c > 0) 
  (h4: (a * b) + c = (a + c) * (b + c)) 
  (h5: a + b + c = 2) : 
  maximum_abc a b c = 8 / 27 := 
by 
  sorry

end maximum_abc_827_l321_32139


namespace find_a_l321_32155

noncomputable def csc (x : ℝ) : ℝ := 1 / (Real.sin x)

theorem find_a (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : a * csc (b * (Real.pi / 6) + c) = 3) : a = 3 := 
sorry

end find_a_l321_32155


namespace find_number_l321_32178

theorem find_number (a : ℤ) (h : a - a + 99 * (a - 99) = 19802) : a = 299 := 
by 
  sorry

end find_number_l321_32178


namespace baseball_game_earnings_l321_32100

theorem baseball_game_earnings (W S : ℝ) 
  (h1 : W + S = 4994.50) 
  (h2 : W = S - 1330.50) : 
  S = 3162.50 := 
by 
  sorry

end baseball_game_earnings_l321_32100


namespace no_real_solutions_for_m_l321_32165

theorem no_real_solutions_for_m (m : ℝ) :
  ∃! m, (4 * m + 2) ^ 2 - 4 * m = 0 → false :=
by 
  sorry

end no_real_solutions_for_m_l321_32165


namespace mike_passing_percentage_l321_32198

theorem mike_passing_percentage (mike_score shortfall max_marks : ℝ)
  (h_mike_score : mike_score = 212)
  (h_shortfall : shortfall = 16)
  (h_max_marks : max_marks = 760) :
  (mike_score + shortfall) / max_marks * 100 = 30 :=
by
  sorry

end mike_passing_percentage_l321_32198


namespace flowers_in_each_basket_l321_32121

theorem flowers_in_each_basket
  (plants_per_daughter : ℕ)
  (num_daughters : ℕ)
  (grown_flowers : ℕ)
  (died_flowers : ℕ)
  (num_baskets : ℕ)
  (h1 : plants_per_daughter = 5)
  (h2 : num_daughters = 2)
  (h3 : grown_flowers = 20)
  (h4 : died_flowers = 10)
  (h5 : num_baskets = 5) :
  (plants_per_daughter * num_daughters + grown_flowers - died_flowers) / num_baskets = 4 :=
by
  sorry

end flowers_in_each_basket_l321_32121


namespace number_of_cases_ordered_in_may_l321_32196

noncomputable def cases_ordered_in_may (ordered_in_april_cases : ℕ) (bottles_per_case : ℕ) (total_bottles : ℕ) : ℕ :=
  let bottles_in_april := ordered_in_april_cases * bottles_per_case
  let bottles_in_may := total_bottles - bottles_in_april
  bottles_in_may / bottles_per_case

theorem number_of_cases_ordered_in_may :
  ∀ (ordered_in_april_cases bottles_per_case total_bottles : ℕ),
  ordered_in_april_cases = 20 →
  bottles_per_case = 20 →
  total_bottles = 1000 →
  cases_ordered_in_may ordered_in_april_cases bottles_per_case total_bottles = 30 := by
  intros ordered_in_april_cases bottles_per_case total_bottles ha hbp htt
  sorry

end number_of_cases_ordered_in_may_l321_32196
