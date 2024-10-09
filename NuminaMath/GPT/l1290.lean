import Mathlib

namespace sum_of_numbers_l1290_129039

theorem sum_of_numbers (avg : ℝ) (count : ℕ) (h_avg : avg = 5.7) (h_count : count = 8) : (avg * count = 45.6) :=
by
  sorry

end sum_of_numbers_l1290_129039


namespace pizza_area_difference_l1290_129074

def hueys_hip_pizza (small_size : ℕ) (small_cost : ℕ) (large_size : ℕ) (large_cost : ℕ) : ℕ :=
  let small_area := small_size * small_size
  let large_area := large_size * large_size
  let individual_money := 30
  let pooled_money := 2 * individual_money

  let individual_small_total_area := (individual_money / small_cost) * small_area * 2
  let pooled_large_total_area := (pooled_money / large_cost) * large_area

  pooled_large_total_area - individual_small_total_area

theorem pizza_area_difference :
  hueys_hip_pizza 6 10 9 20 = 27 :=
by
  sorry

end pizza_area_difference_l1290_129074


namespace geometric_series_sum_eq_l1290_129075

theorem geometric_series_sum_eq :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5
  (∀ S_n, S_n = a * (1 - r^n) / (1 - r) → S_n = 1 / 3) :=
by
  intro a r n S_n
  sorry

end geometric_series_sum_eq_l1290_129075


namespace book_distribution_l1290_129072

theorem book_distribution (a b : ℕ) (h1 : a + b = 282) (h2 : (3 / 4) * a = (5 / 9) * b) : a = 120 ∧ b = 162 := by
  sorry

end book_distribution_l1290_129072


namespace find_principal_amount_l1290_129060

noncomputable def principal_amount (SI R T : ℝ) : ℝ :=
  SI / (R * T / 100)

theorem find_principal_amount :
  principal_amount 4052.25 9 5 = 9005 := by
sorry

end find_principal_amount_l1290_129060


namespace alpha_in_second_quadrant_l1290_129004

variable (α : ℝ)

-- Conditions that P(tan α, cos α) is in the third quadrant
def P_in_third_quadrant (α : ℝ) : Prop := (Real.tan α < 0) ∧ (Real.cos α < 0)

-- Theorem statement
theorem alpha_in_second_quadrant (h : P_in_third_quadrant α) : 
  π/2 < α ∧ α < π :=
sorry

end alpha_in_second_quadrant_l1290_129004


namespace find_second_divisor_l1290_129040

theorem find_second_divisor :
  ∃ y : ℝ, (320 / 2) / y = 53.33 ∧ y = 160 / 53.33 :=
by
  sorry

end find_second_divisor_l1290_129040


namespace profit_difference_l1290_129077

-- Define the initial investments
def investment_A : ℚ := 8000
def investment_B : ℚ := 10000
def investment_C : ℚ := 12000

-- Define B's profit share
def profit_B : ℚ := 1700

-- Prove that the difference between A and C's profit shares is Rs. 680
theorem profit_difference (investment_A investment_B investment_C profit_B: ℚ) (hA : investment_A = 8000) (hB : investment_B = 10000) (hC : investment_C = 12000) (pB : profit_B = 1700) :
    let ratio_A : ℚ := 4
    let ratio_B : ℚ := 5
    let ratio_C : ℚ := 6
    let part_value : ℚ := profit_B / ratio_B
    let profit_A : ℚ := ratio_A * part_value
    let profit_C : ℚ := ratio_C * part_value
    profit_C - profit_A = 680 := 
by
  sorry

end profit_difference_l1290_129077


namespace nn_gt_n1n1_l1290_129012

theorem nn_gt_n1n1 (n : ℕ) (h : n > 1) : n^n > (n + 1)^(n - 1) := 
sorry

end nn_gt_n1n1_l1290_129012


namespace c_share_of_profit_l1290_129097

theorem c_share_of_profit 
  (x : ℝ) -- The amount invested by B
  (total_profit : ℝ := 11000) -- Total profit
  (A_invest : ℝ := 3 * x) -- A's investment
  (C_invest : ℝ := (3/2) * A_invest) -- C's investment
  (total_invest : ℝ := A_invest + x + C_invest) -- Total investment
  (C_share : ℝ := C_invest / total_invest * total_profit) -- C's share of the profit
  : C_share = 99000 / 17 := 
  by sorry

end c_share_of_profit_l1290_129097


namespace nat_values_of_x_l1290_129008

theorem nat_values_of_x :
  (∃ (x : ℕ), 2^(x - 5) = 2 ∧ x = 6) ∧
  (∃ (x : ℕ), 2^x = 512 ∧ x = 9) ∧
  (∃ (x : ℕ), x^5 = 243 ∧ x = 3) ∧
  (∃ (x : ℕ), x^4 = 625 ∧ x = 5) :=
  by {
    sorry
  }

end nat_values_of_x_l1290_129008


namespace value_of_expression_l1290_129066

theorem value_of_expression (x : ℝ) (h : x = 5) : (x^2 + x - 12) / (x - 4) = 18 :=
by 
  sorry

end value_of_expression_l1290_129066


namespace inequality_proof_l1290_129036

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1 / 3) * (a + b + c) ^ 2 :=
sorry

end inequality_proof_l1290_129036


namespace gloria_turtle_time_l1290_129053

theorem gloria_turtle_time (g_time : ℕ) (george_time : ℕ) (gloria_time : ℕ) 
  (h1 : g_time = 6) 
  (h2 : george_time = g_time - 2)
  (h3 : gloria_time = 2 * george_time) : 
  gloria_time = 8 :=
sorry

end gloria_turtle_time_l1290_129053


namespace remainder_43_pow_97_pow_5_plus_109_mod_163_l1290_129014

theorem remainder_43_pow_97_pow_5_plus_109_mod_163 :
    (43 ^ (97 ^ 5) + 109) % 163 = 50 :=
by
  sorry

end remainder_43_pow_97_pow_5_plus_109_mod_163_l1290_129014


namespace no_solution_exists_l1290_129052

theorem no_solution_exists :
  ¬ ∃ a b : ℝ, a^2 + 3 * b^2 + 2 = 3 * a * b :=
by
  sorry

end no_solution_exists_l1290_129052


namespace Nell_cards_difference_l1290_129013

-- Definitions
def initial_baseball_cards : ℕ := 438
def initial_ace_cards : ℕ := 18
def given_ace_cards : ℕ := 55
def given_baseball_cards : ℕ := 178

-- Theorem statement
theorem Nell_cards_difference :
  given_baseball_cards - given_ace_cards = 123 := 
by
  sorry

end Nell_cards_difference_l1290_129013


namespace find_integer_n_l1290_129087

theorem find_integer_n (n : ℤ) :
  (⌊ (n^2 : ℤ) / 9 ⌋ - ⌊ n / 3 ⌋^2 = 3) → (n = 8 ∨ n = 10) :=
  sorry

end find_integer_n_l1290_129087


namespace quadratic_roots_identity_l1290_129099

noncomputable def a := - (2 / 5 : ℝ)
noncomputable def b := (1 / 5 : ℝ)
noncomputable def quadraticRoots := (a, b)

theorem quadratic_roots_identity :
  a + b ^ 2 = - (9 / 25 : ℝ) := 
by 
  rw [a, b]
  sorry

end quadratic_roots_identity_l1290_129099


namespace power_modulo_remainder_l1290_129051

theorem power_modulo_remainder :
  (17 ^ 2046) % 23 = 22 := 
sorry

end power_modulo_remainder_l1290_129051


namespace triangle_sin_a_triangle_area_l1290_129076

theorem triangle_sin_a (B : ℝ) (a b c : ℝ) (hB : B = π / 4)
  (h_bc : b = Real.sqrt 5 ∧ c = Real.sqrt 2 ∨ a = 3 ∧ c = Real.sqrt 2) :
  Real.sin A = (3 * Real.sqrt 10) / 10 :=
sorry

theorem triangle_area (B a b c : ℝ) (hB : B = π / 4) (hb : b = Real.sqrt 5)
  (h_ac : a + c = 3) : 1 / 2 * a * c * Real.sin B = Real.sqrt 2 - 1 :=
sorry

end triangle_sin_a_triangle_area_l1290_129076


namespace number_of_subsets_l1290_129098

theorem number_of_subsets (M : Finset ℕ) (h : M.card = 5) : 2 ^ M.card = 32 := by
  sorry

end number_of_subsets_l1290_129098


namespace find_m_value_l1290_129017

-- Define the quadratic function
def quadratic (x m : ℝ) : ℝ := x^2 - 6 * x + m

-- Define the condition that the quadratic function has a minimum value of 1
def has_minimum_value_of_one (m : ℝ) : Prop := ∃ x : ℝ, quadratic x m = 1

-- The main theorem statement
theorem find_m_value : ∀ m : ℝ, has_minimum_value_of_one m → m = 10 :=
by sorry

end find_m_value_l1290_129017


namespace mark_initial_fries_l1290_129088

variable (Sally_fries_before : ℕ)
variable (Sally_fries_after : ℕ)
variable (Mark_fries_given : ℕ)
variable (Mark_fries_initial : ℕ)

theorem mark_initial_fries (h1 : Sally_fries_before = 14) (h2 : Sally_fries_after = 26) (h3 : Mark_fries_given = Sally_fries_after - Sally_fries_before) (h4 : Mark_fries_given = 1/3 * Mark_fries_initial) : Mark_fries_initial = 36 :=
by sorry

end mark_initial_fries_l1290_129088


namespace division_addition_l1290_129080

theorem division_addition : (-300) / (-75) + 10 = 14 := by
  sorry

end division_addition_l1290_129080


namespace compare_a_b_c_l1290_129009

def a : ℝ := 2^(1/2)
def b : ℝ := 3^(1/3)
def c : ℝ := 5^(1/5)

theorem compare_a_b_c : b > a ∧ a > c :=
  by
  sorry

end compare_a_b_c_l1290_129009


namespace area_of_triangle_is_correct_l1290_129032

def point := ℚ × ℚ

def A : point := (4, -4)
def B : point := (-1, 1)
def C : point := (2, -7)

def vector_sub (p1 p2 : point) : point :=
(p1.1 - p2.1, p1.2 - p2.2)

def determinant (v w : point) : ℚ :=
v.1 * w.2 - v.2 * w.1

def area_of_triangle (A B C : point) : ℚ :=
(abs (determinant (vector_sub C A) (vector_sub C B))) / 2

theorem area_of_triangle_is_correct :
  area_of_triangle A B C = 12.5 :=
by sorry

end area_of_triangle_is_correct_l1290_129032


namespace more_balloons_l1290_129048

theorem more_balloons (you_balloons : ℕ) (friend_balloons : ℕ) (h_you : you_balloons = 7) (h_friend : friend_balloons = 5) : 
  you_balloons - friend_balloons = 2 :=
sorry

end more_balloons_l1290_129048


namespace distributive_laws_none_hold_l1290_129082

def star (a b : ℝ) : ℝ := a + b + a * b

theorem distributive_laws_none_hold (x y z : ℝ) :
  ¬ (x * (y + z) = (x * y) + (x * z)) ∧
  ¬ (x + (y * z) = (x + y) * (x + z)) ∧
  ¬ (x * (y * z) = (x * y) * (x * z)) :=
by
  sorry

end distributive_laws_none_hold_l1290_129082


namespace number_of_yellow_crayons_l1290_129015

theorem number_of_yellow_crayons : 
  ∃ (R B Y : ℕ), 
  R = 14 ∧ 
  B = R + 5 ∧ 
  Y = 2 * B - 6 ∧ 
  Y = 32 :=
by
  sorry

end number_of_yellow_crayons_l1290_129015


namespace marbles_selection_l1290_129026

theorem marbles_selection : 
  ∃ (n : ℕ), n = 1540 ∧ 
  ∃ marbles : Finset ℕ, marbles.card = 15 ∧
  ∃ rgb : Finset ℕ, rgb ⊆ marbles ∧ rgb.card = 3 ∧
  ∃ yellow : ℕ, yellow ∈ marbles ∧ yellow ∉ rgb ∧ 
  ∀ (selection : Finset ℕ), selection.card = 5 →
  (∃ red green blue : ℕ, red ∈ rgb ∧ green ∈ rgb ∧ blue ∈ rgb ∧ 
  (red ∈ selection ∨ green ∈ selection ∨ blue ∈ selection) ∧ yellow ∉ selection) → 
  (selection.card = 5) :=
by
  sorry

end marbles_selection_l1290_129026


namespace maximize_savings_l1290_129002

-- Definitions for the conditions
def initial_amount : ℝ := 15000

def discount_option1 (amount : ℝ) : ℝ := 
  let after_first : ℝ := amount * 0.75
  let after_second : ℝ := after_first * 0.90
  after_second * 0.95

def discount_option2 (amount : ℝ) : ℝ := 
  let after_first : ℝ := amount * 0.70
  let after_second : ℝ := after_first * 0.90
  after_second * 0.90

-- Theorem to compare the final amounts
theorem maximize_savings : discount_option2 initial_amount < discount_option1 initial_amount := 
  sorry

end maximize_savings_l1290_129002


namespace people_per_car_l1290_129035

theorem people_per_car (total_people cars : ℕ) (h1 : total_people = 63) (h2 : cars = 9) :
  total_people / cars = 7 :=
by
  sorry

end people_per_car_l1290_129035


namespace mila_total_distance_l1290_129094

/-- Mila's car consumes a gallon of gas every 40 miles, her full gas tank holds 16 gallons, starting with a full tank, she drove 400 miles, then refueled with 10 gallons, 
and upon arriving at her destination her gas tank was a third full.
Prove that the total distance Mila drove that day is 826 miles. -/
theorem mila_total_distance (consumption_per_mile : ℝ) (tank_capacity : ℝ) (initial_drive : ℝ) (refuel_amount : ℝ) (final_fraction : ℝ)
  (consumption_per_mile_def : consumption_per_mile = 1 / 40)
  (tank_capacity_def : tank_capacity = 16)
  (initial_drive_def : initial_drive = 400)
  (refuel_amount_def : refuel_amount = 10)
  (final_fraction_def : final_fraction = 1 / 3) :
  ∃ total_distance : ℝ, total_distance = 826 :=
by
  sorry

end mila_total_distance_l1290_129094


namespace modulo_remainder_even_l1290_129018

theorem modulo_remainder_even (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2018) : 
  ((2019 - k)^12 + 2018) % 2019 = (k^12 + 2018) % 2019 := 
by
  sorry

end modulo_remainder_even_l1290_129018


namespace ce_length_l1290_129034

noncomputable def CE_in_parallelogram (AB AD BD : ℝ) (AB_eq : AB = 480) (AD_eq : AD = 200) (BD_eq : BD = 625) : ℝ :=
  280

theorem ce_length (AB AD BD : ℝ) (AB_eq : AB = 480) (AD_eq : AD = 200) (BD_eq : BD = 625) :
  CE_in_parallelogram AB AD BD AB_eq AD_eq BD_eq = 280 :=
by
  sorry

end ce_length_l1290_129034


namespace largest_three_digit_multiple_of_six_with_sum_fifteen_l1290_129065

theorem largest_three_digit_multiple_of_six_with_sum_fifteen : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n % 6 = 0) ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ (m % 6 = 0) ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
  sorry

end largest_three_digit_multiple_of_six_with_sum_fifteen_l1290_129065


namespace olympic_high_school_amc10_l1290_129091

/-- At Olympic High School, 2/5 of the freshmen and 4/5 of the sophomores took the AMC-10.
    Given that the number of freshmen and sophomore contestants was the same, there are twice as many freshmen as sophomores. -/
theorem olympic_high_school_amc10 (f s : ℕ) (hf : f > 0) (hs : s > 0)
  (contest_equal : (2 / 5 : ℚ)*f = (4 / 5 : ℚ)*s) : f = 2 * s :=
by
  sorry

end olympic_high_school_amc10_l1290_129091


namespace campers_afternoon_l1290_129044

theorem campers_afternoon (x : ℕ) 
  (h1 : 44 = x + 5) : 
  x = 39 := 
by
  sorry

end campers_afternoon_l1290_129044


namespace num_non_congruent_triangles_with_perimeter_12_l1290_129010

noncomputable def count_non_congruent_triangles_with_perimeter_12 : ℕ :=
  sorry -- This is where the actual proof or computation would go.

theorem num_non_congruent_triangles_with_perimeter_12 :
  count_non_congruent_triangles_with_perimeter_12 = 3 :=
  sorry -- This is the theorem stating the result we want to prove.

end num_non_congruent_triangles_with_perimeter_12_l1290_129010


namespace sum_of_first_n_terms_l1290_129046

theorem sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 + 2 * a 2 = 3)
  (h2 : ∀ n, a (n + 1) = a n + 2) :
  ∀ n, S n = n * (n - 4 / 3) := 
sorry

end sum_of_first_n_terms_l1290_129046


namespace solve_inequality_l1290_129043

variable {c : ℝ}
variable (h_c_ne_2 : c ≠ 2)

theorem solve_inequality :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - (1 + 2) * x + 2 ≤ 0) ∧
  (c > 2 → (∀ x : ℝ, (x - c) * (x - 2) > 0 ↔ x > c ∨ x < 2)) ∧
  (c < 2 → (∀ x : ℝ, (x - c) * (x - 2) > 0 ↔ x < c ∨ x > 2)) :=
by
  sorry

end solve_inequality_l1290_129043


namespace golden_ratio_problem_l1290_129037

noncomputable def m := 2 * Real.sin (Real.pi * 18 / 180)
noncomputable def n := 4 - m^2
noncomputable def target_expression := m * Real.sqrt n / (2 * (Real.cos (Real.pi * 27 / 180))^2 - 1)

theorem golden_ratio_problem :
  target_expression = 2 :=
by
  -- Proof will be placed here
  sorry

end golden_ratio_problem_l1290_129037


namespace largest_piece_length_l1290_129073

theorem largest_piece_length (v : ℝ) (hv : v + (3/2) * v + (9/4) * v = 95) : 
  (9/4) * v = 45 :=
by sorry

end largest_piece_length_l1290_129073


namespace part2_l1290_129062

noncomputable def f (a x : ℝ) : ℝ := a * Real.log (x + 1) - x

theorem part2 (a : ℝ) (h : a > 0) (x : ℝ) : f a x < (a - 1) * Real.log a + a^2 := 
  sorry

end part2_l1290_129062


namespace baker_cakes_l1290_129092

theorem baker_cakes (C : ℕ) (h1 : 154 = 78 + 76) (h2 : C = 78) : C = 78 :=
sorry

end baker_cakes_l1290_129092


namespace simplify_sqrt_eight_l1290_129045

theorem simplify_sqrt_eight : Real.sqrt 8 = 2 * Real.sqrt 2 := sorry

end simplify_sqrt_eight_l1290_129045


namespace total_emails_received_l1290_129067

theorem total_emails_received (E : ℝ)
    (h1 : (3/5) * (3/4) * E = 180) :
    E = 400 :=
sorry

end total_emails_received_l1290_129067


namespace cube_difference_div_l1290_129030

theorem cube_difference_div (a b : ℕ) (h_a : a = 64) (h_b : b = 27) : 
  (a^3 - b^3) / (a - b) = 6553 := by
  sorry

end cube_difference_div_l1290_129030


namespace exists_consecutive_integers_not_sum_of_two_squares_l1290_129003

open Nat

theorem exists_consecutive_integers_not_sum_of_two_squares : 
  ∃ (m : ℕ), ∀ k : ℕ, k < 2017 → ¬(∃ a b : ℤ, (m + k) = a^2 + b^2) := 
sorry

end exists_consecutive_integers_not_sum_of_two_squares_l1290_129003


namespace transformed_roots_l1290_129056

noncomputable def specific_polynomial : Polynomial ℝ :=
  Polynomial.C 1 - Polynomial.C 4 * Polynomial.X + Polynomial.C 6 * Polynomial.X ^ 2 - Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 4

theorem transformed_roots (a b c d : ℝ) :
  (a^4 - b*a - 5 = 0) ∧ (b^4 - b*b - 5 = 0) ∧ (c^4 - b*c - 5 = 0) ∧ (d^4 - b*d - 5 = 0) →
  specific_polynomial.eval ((a + b + c) / d)^2 = 0 ∧
  specific_polynomial.eval ((a + b + d) / c)^2 = 0 ∧
  specific_polynomial.eval ((a + c + d) / b)^2 = 0 ∧
  specific_polynomial.eval ((b + c + d) / a)^2 = 0 :=
  by
    sorry

end transformed_roots_l1290_129056


namespace fewest_printers_l1290_129079

theorem fewest_printers (x y : ℕ) (h1 : 350 * x = 200 * y) : x + y = 11 := 
by
  sorry

end fewest_printers_l1290_129079


namespace girls_on_debate_team_l1290_129007

def number_of_students (groups: ℕ) (group_size: ℕ) : ℕ :=
  groups * group_size

def total_students_debate_team : ℕ :=
  number_of_students 8 9

def number_of_boys : ℕ := 26

def number_of_girls : ℕ :=
  total_students_debate_team - number_of_boys

theorem girls_on_debate_team :
  number_of_girls = 46 :=
by
  sorry

end girls_on_debate_team_l1290_129007


namespace find_m_range_l1290_129058

def proposition_p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m < 0) ∧ (1 > 0)

def proposition_q (m : ℝ) : Prop :=
  16 * (m - 2)^2 - 16 < 0

theorem find_m_range : {m : ℝ // proposition_p m ∧ proposition_q m} = {m : ℝ // 2 < m ∧ m < 3} :=
by
  sorry

end find_m_range_l1290_129058


namespace completing_the_square_l1290_129083

theorem completing_the_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) → ((x - 1)^2 = 6) :=
by
  sorry

end completing_the_square_l1290_129083


namespace range_of_a_l1290_129050

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
by sorry

end range_of_a_l1290_129050


namespace right_triangle_area_is_integer_l1290_129027

theorem right_triangle_area_is_integer (a b : ℕ) (h1 : ∃ (A : ℕ), A = (1 / 2 : ℚ) * ↑a * ↑b) : (a % 2 = 0) ∨ (b % 2 = 0) :=
sorry

end right_triangle_area_is_integer_l1290_129027


namespace prove_remaining_area_is_24_l1290_129081

/-- A rectangular piece of paper with length 12 cm and width 8 cm has four identical isosceles 
right triangles with legs of 6 cm cut from it. Prove that the remaining area is 24 cm². --/
def remaining_area : ℕ := 
  let length := 12
  let width := 8
  let rect_area := length * width
  let triangle_leg := 6
  let triangle_area := (triangle_leg * triangle_leg) / 2
  let total_triangle_area := 4 * triangle_area
  rect_area - total_triangle_area

theorem prove_remaining_area_is_24 : (remaining_area = 24) :=
  by sorry

end prove_remaining_area_is_24_l1290_129081


namespace similar_triangle_shortest_side_l1290_129020

theorem similar_triangle_shortest_side
  (a₁ : ℕ) (c₁ : ℕ) (c₂ : ℕ)
  (h₁ : a₁ = 15) (h₂ : c₁ = 17) (h₃ : c₂ = 68)
  (right_triangle_1 : a₁^2 + b₁^2 = c₁^2)
  (similar_triangles : ∃ k : ℕ, c₂ = k * c₁) :
  shortest_side = 32 := 
sorry

end similar_triangle_shortest_side_l1290_129020


namespace days_gumballs_last_l1290_129085

def pairs_day_1 := 3
def gumballs_per_pair := 9
def gumballs_day_1 := pairs_day_1 * gumballs_per_pair

def pairs_day_2 := pairs_day_1 * 2
def gumballs_day_2 := pairs_day_2 * gumballs_per_pair

def pairs_day_3 := pairs_day_2 - 1
def gumballs_day_3 := pairs_day_3 * gumballs_per_pair

def total_gumballs := gumballs_day_1 + gumballs_day_2 + gumballs_day_3
def gumballs_eaten_per_day := 3

theorem days_gumballs_last : total_gumballs / gumballs_eaten_per_day = 42 :=
by
  sorry

end days_gumballs_last_l1290_129085


namespace simplify_expression_l1290_129055

theorem simplify_expression :
  (123 / 999) * 27 = 123 / 37 :=
by sorry

end simplify_expression_l1290_129055


namespace solution_set_inequality_l1290_129016

theorem solution_set_inequality : {x : ℝ | (x - 2) * (1 - 2 * x) ≥ 0} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 2} :=
by
  sorry  -- Proof to be provided

end solution_set_inequality_l1290_129016


namespace mixture_problem_l1290_129090

theorem mixture_problem :
  ∀ (x P : ℝ), 
    let initial_solution := 70
    let initial_percentage := 0.20
    let final_percentage := 0.40
    let final_amount := 70
    (x = 70) →
    (initial_percentage * initial_solution + P * x = final_percentage * (initial_solution + x)) →
    (P = 0.60) :=
by
  intros x P initial_solution initial_percentage final_percentage final_amount hx h_eq
  sorry

end mixture_problem_l1290_129090


namespace inequality_1_inequality_2_l1290_129022

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

theorem inequality_1 (x : ℝ) : f x > 2 * x ↔ x < -1/2 :=
sorry

theorem inequality_2 (t : ℝ) :
  (∃ x : ℝ, f x > t ^ 2 - t + 1) ↔ (0 < t ∧ t < 1) :=
sorry

end inequality_1_inequality_2_l1290_129022


namespace sqrt_square_l1290_129001

theorem sqrt_square (n : ℝ) : (Real.sqrt 2023) ^ 2 = 2023 :=
by
  sorry

end sqrt_square_l1290_129001


namespace sum_four_digit_integers_ending_in_zero_l1290_129054

def arithmetic_series_sum (a l d : ℕ) : ℕ := 
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_four_digit_integers_ending_in_zero : 
  arithmetic_series_sum 1000 9990 10 = 4945500 :=
by
  sorry

end sum_four_digit_integers_ending_in_zero_l1290_129054


namespace four_digit_not_multiples_of_4_or_9_l1290_129064

theorem four_digit_not_multiples_of_4_or_9 (h1 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 4 ∣ n ↔ (250 ≤ n / 4 ∧ n / 4 ≤ 2499))
                                         (h2 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 9 ∣ n ↔ (112 ≤ n / 9 ∧ n / 9 ≤ 1111))
                                         (h3 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 36 ∣ n ↔ (28 ≤ n / 36 ∧ n / 36 ≤ 277)) :
                                         (9000 - ((2250 : ℕ) + 1000 - 250)) = 6000 :=
by sorry

end four_digit_not_multiples_of_4_or_9_l1290_129064


namespace chad_savings_correct_l1290_129006

variable (earnings_mowing : ℝ := 600)
variable (earnings_birthday : ℝ := 250)
variable (earnings_video_games : ℝ := 150)
variable (earnings_odd_jobs : ℝ := 150)
variable (tax_rate : ℝ := 0.10)

noncomputable def total_earnings : ℝ := 
  earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs

noncomputable def taxes : ℝ := 
  tax_rate * total_earnings

noncomputable def money_after_taxes : ℝ := 
  total_earnings - taxes

noncomputable def savings_mowing : ℝ := 
  0.50 * earnings_mowing

noncomputable def savings_birthday : ℝ := 
  0.30 * earnings_birthday

noncomputable def savings_video_games : ℝ := 
  0.40 * earnings_video_games

noncomputable def savings_odd_jobs : ℝ := 
  0.20 * earnings_odd_jobs

noncomputable def total_savings : ℝ := 
  savings_mowing + savings_birthday + savings_video_games + savings_odd_jobs

theorem chad_savings_correct : total_savings = 465 := by
  sorry

end chad_savings_correct_l1290_129006


namespace KayleeAgeCorrect_l1290_129024

-- Define Kaylee's current age
def KayleeCurrentAge (k : ℕ) : Prop :=
  (3 * 5 + (7 - k) = 7)

-- State the theorem
theorem KayleeAgeCorrect : ∃ k : ℕ, KayleeCurrentAge k ∧ k = 8 := 
sorry

end KayleeAgeCorrect_l1290_129024


namespace remainder_when_divided_by_11_l1290_129025

theorem remainder_when_divided_by_11 {k x : ℕ} (h : x = 66 * k + 14) : x % 11 = 3 :=
by
  sorry

end remainder_when_divided_by_11_l1290_129025


namespace min_value_of_box_l1290_129049

theorem min_value_of_box (a b : ℤ) (h_ab : a * b = 30) : 
  ∃ (m : ℤ), m = 61 ∧ (∀ (c : ℤ), a * b = 30 → a^2 + b^2 = c → c ≥ m) := 
sorry

end min_value_of_box_l1290_129049


namespace average_contribution_increase_l1290_129057

theorem average_contribution_increase
  (average_old : ℝ)
  (num_people_old : ℕ)
  (john_donation : ℝ)
  (increase_percentage : ℝ) :
  average_old = 75 →
  num_people_old = 3 →
  john_donation = 150 →
  increase_percentage = 25 :=
by {
  sorry
}

end average_contribution_increase_l1290_129057


namespace intersection_A_B_l1290_129063

def A : Set ℤ := {-1, 1, 3, 5, 7}
def B : Set ℝ := { x | 2^x > 2 * Real.sqrt 2 }

theorem intersection_A_B :
  A ∩ { x : ℤ | x > 3 / 2 } = {3, 5, 7} :=
by
  sorry

end intersection_A_B_l1290_129063


namespace cassie_nails_claws_total_l1290_129000

theorem cassie_nails_claws_total :
  let dogs := 4
  let parrots := 8
  let cats := 2
  let rabbits := 6
  let lizards := 5
  let tortoises := 3

  let dog_nails := dogs * 4 * 4

  let normal_parrots := 6
  let parrot_with_extra_toe := 1
  let parrot_missing_toe := 1
  let parrot_claws := (normal_parrots * 2 * 3) + (parrot_with_extra_toe * 2 * 4) + (parrot_missing_toe * 2 * 2)

  let normal_cats := 1
  let deformed_cat := 1
  let cat_toes := (1 * 4 * 5) + (1 * 4 * 4) + 1 

  let normal_rabbits := 5
  let deformed_rabbit := 1
  let rabbit_nails := (normal_rabbits * 4 * 9) + (3 * 9 + 2)

  let normal_lizards := 4
  let deformed_lizard := 1
  let lizard_toes := (normal_lizards * 4 * 5) + (deformed_lizard * 4 * 4)
  
  let normal_tortoises := 1
  let tortoise_with_extra_claw := 1
  let tortoise_missing_claw := 1
  let tortoise_claws := (normal_tortoises * 4 * 4) + (3 * 4 + 5) + (3 * 4 + 3)

  let total_nails_claws := dog_nails + parrot_claws + cat_toes + rabbit_nails + lizard_toes + tortoise_claws

  total_nails_claws = 524 :=
by
  sorry

end cassie_nails_claws_total_l1290_129000


namespace union_of_sets_l1290_129021

def A : Set ℝ := { x : ℝ | x * (x + 1) ≤ 0 }
def B : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }
def C : Set ℝ := { x : ℝ | -1 ≤ x ∧ x < 1 }

theorem union_of_sets :
  A ∪ B = C := 
sorry

end union_of_sets_l1290_129021


namespace find_number_in_parentheses_l1290_129023

theorem find_number_in_parentheses :
  ∃ x : ℝ, 3 + 2 * (x - 3) = 24.16 ∧ x = 13.58 :=
by
  sorry

end find_number_in_parentheses_l1290_129023


namespace div_by_240_l1290_129047

theorem div_by_240 (a b c d : ℕ) : 240 ∣ (a ^ (4 * b + d) - a ^ (4 * c + d)) :=
sorry

end div_by_240_l1290_129047


namespace Lizzie_has_27_crayons_l1290_129028

variable (Lizzie Bobbie Billie : ℕ)

axiom Billie_crayons : Billie = 18
axiom Bobbie_crayons : Bobbie = 3 * Billie
axiom Lizzie_crayons : Lizzie = Bobbie / 2

theorem Lizzie_has_27_crayons : Lizzie = 27 :=
by
  sorry

end Lizzie_has_27_crayons_l1290_129028


namespace continuous_function_nondecreasing_l1290_129011

open Set

variable {α : Type*} [LinearOrder ℝ] [Preorder ℝ]

theorem continuous_function_nondecreasing
  (f : (ℝ)→ ℝ) 
  (h_cont : ContinuousOn f (Ioi 0))
  (h_seq : ∀ x > 0, Monotone (fun n : ℕ => f (n*x))):
  ∀ x y, x ≤ y → f x ≤ f y := 
sorry

end continuous_function_nondecreasing_l1290_129011


namespace samantha_probability_l1290_129071

noncomputable def probability_of_selecting_yellow_apples 
  (total_apples : ℕ) (yellow_apples : ℕ) (selection_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_apples selection_size
  let yellow_ways := Nat.choose yellow_apples selection_size
  yellow_ways / total_ways

theorem samantha_probability : 
  probability_of_selecting_yellow_apples 10 5 3 = 1 / 12 := 
by 
  sorry

end samantha_probability_l1290_129071


namespace arun_weight_average_l1290_129029

theorem arun_weight_average :
  ∀ (w : ℝ), (w > 61 ∧ w < 72) ∧ (w > 60 ∧ w < 70) ∧ (w ≤ 64) →
  (w = 62 ∨ w = 63) →
  (62 + 63) / 2 = 62.5 :=
by
  intros w h1 h2
  sorry

end arun_weight_average_l1290_129029


namespace smallest_positive_integer_l1290_129096

theorem smallest_positive_integer (n : ℕ) : 13 * n ≡ 567 [MOD 5] ↔ n = 4 := by
  sorry

end smallest_positive_integer_l1290_129096


namespace train_rate_first_hour_l1290_129095

-- Define the conditions
def rateAtFirstHour (r : ℕ) : Prop :=
  (11 / 2) * (r + (r + 100)) = 660

-- Prove the rate is 10 mph
theorem train_rate_first_hour (r : ℕ) : rateAtFirstHour r → r = 10 :=
by 
  sorry

end train_rate_first_hour_l1290_129095


namespace lily_typing_speed_l1290_129061

-- Define the conditions
def wordsTyped : ℕ := 255
def totalMinutes : ℕ := 19
def breakTime : ℕ := 2
def typingInterval : ℕ := 10
def effectiveMinutes : ℕ := totalMinutes - breakTime

-- Define the number of words typed in effective minutes
def wordsPerMinute (words : ℕ) (minutes : ℕ) : ℕ := words / minutes

-- Statement to be proven
theorem lily_typing_speed : wordsPerMinute wordsTyped effectiveMinutes = 15 :=
by
  -- proof goes here
  sorry

end lily_typing_speed_l1290_129061


namespace inequality_condition_l1290_129069

theorem inequality_condition {a b c : ℝ} :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (Real.sqrt (a^2 + b^2) < c) :=
by
  sorry

end inequality_condition_l1290_129069


namespace dorothy_needs_more_money_l1290_129093

structure Person :=
  (age : ℕ)

def Discount (age : ℕ) : ℝ :=
  if age <= 11 then 0.5 else
  if age >= 65 then 0.8 else
  if 12 <= age && age <= 18 then 0.7 else 1.0

def ticketCost (age : ℕ) : ℝ :=
  (10 : ℝ) * Discount age

def specialExhibitCost : ℝ := 5

def totalCost (family : List Person) : ℝ :=
  (family.map (λ p => ticketCost p.age + specialExhibitCost)).sum

def salesTaxRate : ℝ := 0.1

def finalCost (family : List Person) : ℝ :=
  let total := totalCost family
  total + (total * salesTaxRate)

def dorothy_money_after_trip (dorothy_money : ℝ) (family : List Person) : ℝ :=
  dorothy_money - finalCost family

theorem dorothy_needs_more_money :
  dorothy_money_after_trip 70 [⟨15⟩, ⟨10⟩, ⟨40⟩, ⟨42⟩, ⟨65⟩] = -1.5 := by
  sorry

end dorothy_needs_more_money_l1290_129093


namespace min_n_Sn_greater_1020_l1290_129031

theorem min_n_Sn_greater_1020 : ∃ n : ℕ, (n ≥ 0) ∧ (2^(n+1) - 2 - n > 1020) ∧ ∀ m : ℕ, (m ≥ 0) ∧ (m < n) → (2^(m+1) - 2 - m ≤ 1020) :=
by
  sorry

end min_n_Sn_greater_1020_l1290_129031


namespace sum_equidistant_terms_l1290_129070

def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∀ n m : ℕ, (n < m) → a (n+1) - a n = a (m+1) - a m

variable {a : ℕ → ℤ}

theorem sum_equidistant_terms (h_seq : is_arithmetic_sequence a)
  (h_4 : a 4 = 5) : a 3 + a 5 = 10 :=
sorry

end sum_equidistant_terms_l1290_129070


namespace ellipse_area_l1290_129086

theorem ellipse_area
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (a : { endpoints_major_axis : (ℝ × ℝ) × (ℝ × ℝ) // endpoints_major_axis = ((x1, y1), (x2, y2)) })
  (b : { point_on_ellipse : ℝ × ℝ // point_on_ellipse = (x3, y3) }) :
  (-5 : ℝ) = x1 ∧ (2 : ℝ) = y1 ∧ (15 : ℝ) = x2 ∧ (2 : ℝ) = y2 ∧
  (8 : ℝ) = x3 ∧ (6 : ℝ) = y3 → 
  100 * Real.pi * Real.sqrt (16 / 91) = 100 * Real.pi * Real.sqrt (16 / 91) :=
by
  sorry

end ellipse_area_l1290_129086


namespace deepak_share_l1290_129019

theorem deepak_share (investment_Anand investment_Deepak total_profit : ℕ)
  (h₁ : investment_Anand = 2250) (h₂ : investment_Deepak = 3200) (h₃ : total_profit = 1380) :
  ∃ share_Deepak, share_Deepak = 810 := sorry

end deepak_share_l1290_129019


namespace tom_tim_typing_ratio_l1290_129089

theorem tom_tim_typing_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) : M / T = 5 :=
by
  sorry

end tom_tim_typing_ratio_l1290_129089


namespace sunflower_packets_correct_l1290_129078

namespace ShyneGarden

-- Define the given conditions
def eggplants_per_packet := 14
def sunflowers_per_packet := 10
def eggplant_packets_bought := 4
def total_plants := 116

-- Define the function to calculate the number of sunflower packets bought
def sunflower_packets_bought (eggplants_per_packet sunflowers_per_packet eggplant_packets_bought total_plants : ℕ) : ℕ :=
  (total_plants - (eggplant_packets_bought * eggplants_per_packet)) / sunflowers_per_packet

-- State the theorem to prove the number of sunflower packets
theorem sunflower_packets_correct :
  sunflower_packets_bought eggplants_per_packet sunflowers_per_packet eggplant_packets_bought total_plants = 6 :=
by
  sorry

end ShyneGarden

end sunflower_packets_correct_l1290_129078


namespace joan_socks_remaining_l1290_129042

-- Definitions based on conditions
def total_socks : ℕ := 1200
def white_socks : ℕ := total_socks / 4
def blue_socks : ℕ := total_socks * 3 / 8
def red_socks : ℕ := total_socks / 6
def green_socks : ℕ := total_socks / 12
def white_socks_lost : ℕ := white_socks / 3
def blue_socks_sold : ℕ := blue_socks / 2
def remaining_white_socks : ℕ := white_socks - white_socks_lost
def remaining_blue_socks : ℕ := blue_socks - blue_socks_sold

-- Theorem to prove the total number of remaining socks
theorem joan_socks_remaining :
  remaining_white_socks + remaining_blue_socks + red_socks + green_socks = 725 := by
  sorry

end joan_socks_remaining_l1290_129042


namespace total_non_overlapping_area_of_squares_l1290_129059

theorem total_non_overlapping_area_of_squares 
  (side_length : ℕ) 
  (num_squares : ℕ)
  (overlapping_areas_count : ℕ)
  (overlapping_width : ℕ)
  (overlapping_height : ℕ)
  (total_area_with_overlap: ℕ)
  (final_missed_patch_ratio: ℕ)
  (final_adjustment: ℕ) 
  (total_area: ℕ :=  total_area_with_overlap-final_missed_patch_ratio ):
  side_length = 2 ∧ 
  num_squares = 4 ∧ 
  overlapping_areas_count = 3 ∧ 
  overlapping_width = 1 ∧ 
  overlapping_height = 2 ∧
  total_area_with_overlap = 16- 3  ∧
  final_missed_patch_ratio = 3-> 
  total_area = 13 := 
 by sorry

end total_non_overlapping_area_of_squares_l1290_129059


namespace find_a_values_l1290_129038

def setA (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.snd - 3) / (p.fst - 2) = a + 1}

def setB (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (a^2 - 1) * p.fst + (a - 1) * p.snd = 15}

def sets_disjoint (A B : Set (ℝ × ℝ)) : Prop := ∀ p : ℝ × ℝ, p ∉ A ∪ B

theorem find_a_values (a : ℝ) :
  sets_disjoint (setA a) (setB a) ↔ a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -4 :=
sorry

end find_a_values_l1290_129038


namespace domain_of_sqrt_sin_l1290_129084

open Real Set

noncomputable def domain_sqrt_sine : Set ℝ :=
  {x | ∃ (k : ℤ), 2 * π * k + π / 6 ≤ x ∧ x ≤ 2 * π * k + 5 * π / 6}

theorem domain_of_sqrt_sin (x : ℝ) :
  (∃ y, y = sqrt (2 * sin x - 1)) ↔ x ∈ domain_sqrt_sine :=
sorry

end domain_of_sqrt_sin_l1290_129084


namespace roots_of_quadratic_function_l1290_129033

variable (a b x : ℝ)

theorem roots_of_quadratic_function (h : a + b = 0) : (b * x * x + a * x = 0) → (x = 0 ∨ x = 1) :=
by {sorry}

end roots_of_quadratic_function_l1290_129033


namespace p_or_q_not_necessarily_true_l1290_129005

theorem p_or_q_not_necessarily_true (p q : Prop) (hnp : ¬p) (hpq : ¬(p ∧ q)) : ¬(p ∨ q) ∨ (p ∨ q) :=
by
  sorry

end p_or_q_not_necessarily_true_l1290_129005


namespace birdseed_weekly_consumption_l1290_129041

def parakeets := 3
def parakeet_consumption := 2
def parrots := 2
def parrot_consumption := 14
def finches := 4
def finch_consumption := parakeet_consumption / 2
def canaries := 5
def canary_consumption := 3
def african_grey_parrots := 2
def african_grey_parrot_consumption := 18
def toucans := 3
def toucan_consumption := 25

noncomputable def daily_consumption := 
  parakeets * parakeet_consumption +
  parrots * parrot_consumption +
  finches * finch_consumption +
  canaries * canary_consumption +
  african_grey_parrots * african_grey_parrot_consumption +
  toucans * toucan_consumption

noncomputable def weekly_consumption := 7 * daily_consumption

theorem birdseed_weekly_consumption : weekly_consumption = 1148 := by
  sorry

end birdseed_weekly_consumption_l1290_129041


namespace factorize_problem_1_factorize_problem_2_l1290_129068

theorem factorize_problem_1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x + 2) * (x - 2) := 
by sorry

theorem factorize_problem_2 (x y : ℝ) : 2 * x^3 - 12 * x^2 * y + 18 * x * y^2 = 2 * x * (x - 3 * y)^2 :=
by sorry

end factorize_problem_1_factorize_problem_2_l1290_129068
