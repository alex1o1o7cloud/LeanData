import Mathlib

namespace NUMINAMATH_GPT_inequality_condition_l1623_162307

theorem inequality_condition (a : ℝ) : 
  (∀ x y : ℝ, x^2 + 2 * x + a ≥ -y^2 - 2 * y) → a ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_condition_l1623_162307


namespace NUMINAMATH_GPT_coeff_x3_in_binom_expansion_l1623_162379

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the coefficient function for x^k in the binomial expansion of (x + 1)^n
def binom_coeff (n k : ℕ) : ℕ := binom n k

-- The theorem to prove that the coefficient of x^3 in the expansion of (x + 1)^36 is 7140
theorem coeff_x3_in_binom_expansion : binom_coeff 36 3 = 7140 :=
by
  sorry

end NUMINAMATH_GPT_coeff_x3_in_binom_expansion_l1623_162379


namespace NUMINAMATH_GPT_problem_proof_l1623_162313

noncomputable def binomial (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

noncomputable def probability_ratio_pq : ℕ :=
let p := binomial 10 2 * binomial 30 2 * binomial 28 2
let q := binomial 30 3 * binomial 27 3 * binomial 24 3 * binomial 21 3 * binomial 18 3 * binomial 15 3 * binomial 12 3 * binomial 9 3 * binomial 6 3 * binomial 3 3
p / (q / (binomial 30 3 * binomial 27 3 * binomial 24 3 * binomial 21 3 * binomial 18 3 * binomial 15 3 * binomial 12 3 * binomial 9 3 * binomial 6 3 * binomial 3 3))

theorem problem_proof :
  probability_ratio_pq = 7371 :=
sorry

end NUMINAMATH_GPT_problem_proof_l1623_162313


namespace NUMINAMATH_GPT_restaurant_total_dishes_l1623_162323

noncomputable def total_couscous_received : ℝ := 15.4 + 45
noncomputable def total_chickpeas_received : ℝ := 19.8 + 33

-- Week 1, ratio of 5:3 (couscous:chickpeas)
noncomputable def sets_of_ratio_week1_couscous : ℝ := total_couscous_received / 5
noncomputable def sets_of_ratio_week1_chickpeas : ℝ := total_chickpeas_received / 3
noncomputable def dishes_week1 : ℝ := min sets_of_ratio_week1_couscous sets_of_ratio_week1_chickpeas

-- Week 2, ratio of 3:2 (couscous:chickpeas)
noncomputable def sets_of_ratio_week2_couscous : ℝ := total_couscous_received / 3
noncomputable def sets_of_ratio_week2_chickpeas : ℝ := total_chickpeas_received / 2
noncomputable def dishes_week2 : ℝ := min sets_of_ratio_week2_couscous sets_of_ratio_week2_chickpeas

-- Total dishes rounded down
noncomputable def total_dishes : ℝ := dishes_week1 + dishes_week2

theorem restaurant_total_dishes :
  ⌊total_dishes⌋ = 32 :=
by {
  sorry
}

end NUMINAMATH_GPT_restaurant_total_dishes_l1623_162323


namespace NUMINAMATH_GPT_zero_unique_multiple_prime_l1623_162301

-- Condition: let n be a number
def n : Int := sorry

-- Condition: let p be any prime number
def is_prime (p : Int) : Prop := sorry  -- Predicate definition for prime number

-- Proof problem statement
theorem zero_unique_multiple_prime (n : Int) :
  (∀ p : Int, is_prime p → (∃ k : Int, n * p = k * p)) ↔ (n = 0) := by
  sorry

end NUMINAMATH_GPT_zero_unique_multiple_prime_l1623_162301


namespace NUMINAMATH_GPT_calculate_expression_l1623_162326

/-
We need to prove that the value of 18 * 36 + 54 * 18 + 18 * 9 is equal to 1782.
-/

theorem calculate_expression : (18 * 36 + 54 * 18 + 18 * 9 = 1782) :=
by
  have a1 : Int := 18 * 36
  have a2 : Int := 54 * 18
  have a3 : Int := 18 * 9
  sorry

end NUMINAMATH_GPT_calculate_expression_l1623_162326


namespace NUMINAMATH_GPT_robin_hid_150_seeds_l1623_162321

theorem robin_hid_150_seeds
    (x y : ℕ)
    (h1 : 5 * x = 6 * y)
    (h2 : y = x - 5) : 
    5 * x = 150 :=
by
    sorry

end NUMINAMATH_GPT_robin_hid_150_seeds_l1623_162321


namespace NUMINAMATH_GPT_waiter_tables_l1623_162363

theorem waiter_tables (w m : ℝ) (avg_customers_per_table : ℝ) (total_customers : ℝ) (t : ℝ)
  (hw : w = 7.0)
  (hm : m = 3.0)
  (havg : avg_customers_per_table = 1.111111111)
  (htotal : total_customers = w + m)
  (ht : t = total_customers / avg_customers_per_table) :
  t = 90 :=
by
  -- Proof would be inserted here
  sorry

end NUMINAMATH_GPT_waiter_tables_l1623_162363


namespace NUMINAMATH_GPT_algebra_inequality_l1623_162344

theorem algebra_inequality (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x ^ 2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end NUMINAMATH_GPT_algebra_inequality_l1623_162344


namespace NUMINAMATH_GPT_intersection_A_B_l1623_162395

-- Define set A
def A : Set Int := { x | x^2 - x - 2 ≤ 0 }

-- Define set B
def B : Set Int := { x | x < 1 }

-- Define the intersection set
def intersection_AB : Set Int := { -1, 0 }

-- Formalize the proof statement
theorem intersection_A_B : (A ∩ B) = intersection_AB :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1623_162395


namespace NUMINAMATH_GPT_divisor_of_99_l1623_162364

def reverse_digits (n : ℕ) : ℕ :=
  -- We assume a placeholder definition for reversing the digits of a number
  sorry

theorem divisor_of_99 (k : ℕ) (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : k ∣ 99 :=
  sorry

end NUMINAMATH_GPT_divisor_of_99_l1623_162364


namespace NUMINAMATH_GPT_seating_arrangements_total_l1623_162352

def num_round_tables := 3
def num_rect_tables := 4
def num_square_tables := 2
def num_couches := 2
def num_benches := 3
def num_extra_chairs := 5

def seats_per_round_table := 6
def seats_per_rect_table := 7
def seats_per_square_table := 4
def seats_per_couch := 3
def seats_per_bench := 5

def total_seats : Nat :=
  num_round_tables * seats_per_round_table +
  num_rect_tables * seats_per_rect_table +
  num_square_tables * seats_per_square_table +
  num_couches * seats_per_couch +
  num_benches * seats_per_bench +
  num_extra_chairs

theorem seating_arrangements_total :
  total_seats = 80 :=
by
  simp [total_seats, num_round_tables, seats_per_round_table,
        num_rect_tables, seats_per_rect_table, num_square_tables,
        seats_per_square_table, num_couches, seats_per_couch,
        num_benches, seats_per_bench, num_extra_chairs]
  done

end NUMINAMATH_GPT_seating_arrangements_total_l1623_162352


namespace NUMINAMATH_GPT_largest_divisor_of_even_diff_squares_l1623_162302

theorem largest_divisor_of_even_diff_squares (m n : ℤ) (h_m_even : ∃ k : ℤ, m = 2 * k) (h_n_even : ∃ k : ℤ, n = 2 * k) (h_n_lt_m : n < m) : 
  ∃ d : ℤ, d = 16 ∧ ∀ p : ℤ, (p ∣ (m^2 - n^2)) → p ≤ d :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_even_diff_squares_l1623_162302


namespace NUMINAMATH_GPT_percent_birth_month_in_march_l1623_162377

theorem percent_birth_month_in_march (total_people : ℕ) (march_births : ℕ) (h1 : total_people = 100) (h2 : march_births = 8) : (march_births * 100 / total_people) = 8 := by
  sorry

end NUMINAMATH_GPT_percent_birth_month_in_march_l1623_162377


namespace NUMINAMATH_GPT_percentage_increase_after_decrease_l1623_162317

theorem percentage_increase_after_decrease (P : ℝ) :
  let P_decreased := 0.70 * P
  let P_final := 1.16 * P
  let x := ((P_final / P_decreased) - 1) * 100
  (P_decreased * (1 + x / 100) = P_final) → x = 65.71 := 
by 
  intros
  let P_decreased := 0.70 * P
  let P_final := 1.16 * P
  let x := ((P_final / P_decreased) - 1) * 100
  have h : (P_decreased * (1 + x / 100) = P_final) := by assumption
  sorry

end NUMINAMATH_GPT_percentage_increase_after_decrease_l1623_162317


namespace NUMINAMATH_GPT_range_of_a_l1623_162310

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 3 - x ^ 2 + x - 5

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ∃ x_max x_min : ℝ, x_max ≠ x_min ∧
  f a x_max = max (f a x_max) (f a x_min) ∧ f a x_min = min (f a x_max) (f a x_min)) → 
  a < 1 / 3 ∧ a ≠ 0 := sorry

end NUMINAMATH_GPT_range_of_a_l1623_162310


namespace NUMINAMATH_GPT_evaluate_expression_l1623_162334

theorem evaluate_expression : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1623_162334


namespace NUMINAMATH_GPT_vivian_mail_in_august_l1623_162316

-- Conditions
def april_mail : ℕ := 5
def may_mail : ℕ := 2 * april_mail
def june_mail : ℕ := 2 * may_mail
def july_mail : ℕ := 2 * june_mail

-- Question: Prove that Vivian will send 80 pieces of mail in August.
theorem vivian_mail_in_august : 2 * july_mail = 80 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_vivian_mail_in_august_l1623_162316


namespace NUMINAMATH_GPT_total_items_given_out_l1623_162343

-- Miss Davis gave 15 popsicle sticks and 20 straws to each group.
def popsicle_sticks_per_group := 15
def straws_per_group := 20
def items_per_group := popsicle_sticks_per_group + straws_per_group

-- There are 10 groups in total.
def number_of_groups := 10

-- Prove the total number of items given out equals 350.
theorem total_items_given_out : items_per_group * number_of_groups = 350 :=
by
  sorry

end NUMINAMATH_GPT_total_items_given_out_l1623_162343


namespace NUMINAMATH_GPT_num_employees_is_143_l1623_162365

def b := 143
def is_sol (b : ℕ) := 80 < b ∧ b < 150 ∧ b % 4 = 3 ∧ b % 5 = 3 ∧ b % 7 = 4

theorem num_employees_is_143 : is_sol b :=
by
  -- This is where the proof would be written
  sorry

end NUMINAMATH_GPT_num_employees_is_143_l1623_162365


namespace NUMINAMATH_GPT_tan_alpha_plus_beta_l1623_162388

open Real

theorem tan_alpha_plus_beta (A alpha beta : ℝ) (h1 : sin alpha = A * sin (alpha + beta)) (h2 : abs A > 1) :
  tan (alpha + beta) = sin beta / (cos beta - A) :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_beta_l1623_162388


namespace NUMINAMATH_GPT_total_potatoes_now_l1623_162384

def initial_potatoes : ℕ := 8
def uneaten_new_potatoes : ℕ := 3

theorem total_potatoes_now : initial_potatoes + uneaten_new_potatoes = 11 := by
  sorry

end NUMINAMATH_GPT_total_potatoes_now_l1623_162384


namespace NUMINAMATH_GPT_John_pays_amount_l1623_162373

/-- Prove the amount John pays given the conditions -/
theorem John_pays_amount
  (total_candies : ℕ)
  (candies_paid_by_dave : ℕ)
  (cost_per_candy : ℚ)
  (candies_paid_by_john := total_candies - candies_paid_by_dave)
  (total_cost_paid_by_john := candies_paid_by_john * cost_per_candy) :
  total_candies = 20 →
  candies_paid_by_dave = 6 →
  cost_per_candy = 1.5 →
  total_cost_paid_by_john = 21 := 
by
  intros h1 h2 h3
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_John_pays_amount_l1623_162373


namespace NUMINAMATH_GPT_flour_needed_correct_l1623_162368

-- Define the total flour required and the flour already added
def total_flour : ℕ := 8
def flour_already_added : ℕ := 2

-- Define the equation to determine the remaining flour needed
def flour_needed : ℕ := total_flour - flour_already_added

-- Prove that the flour needed to be added is 6 cups
theorem flour_needed_correct : flour_needed = 6 := by
  sorry

end NUMINAMATH_GPT_flour_needed_correct_l1623_162368


namespace NUMINAMATH_GPT_difference_before_flipping_l1623_162332

-- Definitions based on the conditions:
variables (Y G : ℕ) -- Number of yellow and green papers

-- Condition: flipping 16 yellow papers changes counts as described
def papers_after_flipping (Y G : ℕ) : Prop :=
  Y - 16 = G + 16

-- Condition: after flipping, there are 64 more green papers than yellow papers.
def green_more_than_yellow_after_flipping (G Y : ℕ) : Prop :=
  G + 16 = (Y - 16) + 64

-- Statement: Prove the difference in the number of green and yellow papers before flipping was 32.
theorem difference_before_flipping (Y G : ℕ) (h1 : papers_after_flipping Y G) (h2 : green_more_than_yellow_after_flipping G Y) :
  (Y - G) = 32 :=
by
  sorry

end NUMINAMATH_GPT_difference_before_flipping_l1623_162332


namespace NUMINAMATH_GPT_melanie_mother_dimes_l1623_162312

-- Definitions based on the conditions
variables (initial_dimes : ℕ) (dimes_given_to_dad : ℤ) (current_dimes : ℤ)

-- Conditions
def melanie_conditions := initial_dimes = 7 ∧ dimes_given_to_dad = 8 ∧ current_dimes = 3

-- Question to be proved is equivalent to proving the number of dimes given by her mother
theorem melanie_mother_dimes (initial_dimes : ℕ) (dimes_given_to_dad : ℤ) (current_dimes : ℤ) (dimes_given_by_mother : ℤ) 
  (h : melanie_conditions initial_dimes dimes_given_to_dad current_dimes) : 
  dimes_given_by_mother = 4 :=
by 
  sorry

end NUMINAMATH_GPT_melanie_mother_dimes_l1623_162312


namespace NUMINAMATH_GPT_evaluate_expression_l1623_162394

theorem evaluate_expression (a b c : ℝ)
  (h : a / (25 - a) + b / (65 - b) + c / (60 - c) = 7) :
  5 / (25 - a) + 13 / (65 - b) + 12 / (60 - c) = 2 := 
sorry

end NUMINAMATH_GPT_evaluate_expression_l1623_162394


namespace NUMINAMATH_GPT_find_a_l1623_162354

theorem find_a (a : ℕ) (h_pos : 0 < a)
  (h_cube : ∀ n : ℕ, 0 < n → ∃ k : ℤ, 4 * ((a : ℤ) ^ n + 1) = k^3) :
  a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l1623_162354


namespace NUMINAMATH_GPT_determine_g1_l1623_162309

variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - x^2 * y - x^3 + 1)

theorem determine_g1 : g 1 = 2 := sorry

end NUMINAMATH_GPT_determine_g1_l1623_162309


namespace NUMINAMATH_GPT_three_digit_number_550_l1623_162333

theorem three_digit_number_550 (N : ℕ) (a b c : ℕ) (h1 : N = 100 * a + 10 * b + c)
  (h2 : 1 ≤ a ∧ a ≤ 9) (h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : 11 ∣ N)
  (h6 : N / 11 = a^2 + b^2 + c^2) : N = 550 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_550_l1623_162333


namespace NUMINAMATH_GPT_range_of_k_l1623_162366

theorem range_of_k (k : ℝ) : ((∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0)) :=
sorry

end NUMINAMATH_GPT_range_of_k_l1623_162366


namespace NUMINAMATH_GPT_Saheed_earnings_l1623_162304

theorem Saheed_earnings (Vika_earnings : ℕ) (Kayla_earnings : ℕ) (Saheed_earnings : ℕ)
  (h1 : Vika_earnings = 84) (h2 : Kayla_earnings = Vika_earnings - 30) (h3 : Saheed_earnings = 4 * Kayla_earnings) :
  Saheed_earnings = 216 := 
by
  sorry

end NUMINAMATH_GPT_Saheed_earnings_l1623_162304


namespace NUMINAMATH_GPT_minnie_more_than_week_l1623_162342

-- Define the variables and conditions
variable (M : ℕ) -- number of horses Minnie mounts per day
variable (mickey_daily : ℕ) -- number of horses Mickey mounts per day

axiom mickey_daily_formula : mickey_daily = 2 * M - 6
axiom mickey_total_per_week : mickey_daily * 7 = 98
axiom days_in_week : 7 = 7

-- Theorem: Minnie mounts 3 more horses per day than there are days in a week
theorem minnie_more_than_week (M : ℕ) 
  (h1 : mickey_daily = 2 * M - 6)
  (h2 : mickey_daily * 7 = 98)
  (h3 : 7 = 7) :
  M - 7 = 3 := 
sorry

end NUMINAMATH_GPT_minnie_more_than_week_l1623_162342


namespace NUMINAMATH_GPT_ephraim_keiko_same_heads_probability_l1623_162348

def coin_toss_probability_same_heads : ℚ :=
  let keiko_prob_0 := 1 / 4
  let keiko_prob_1 := 1 / 2
  let keiko_prob_2 := 1 / 4
  let ephraim_prob_0 := 1 / 8
  let ephraim_prob_1 := 3 / 8
  let ephraim_prob_2 := 3 / 8
  let ephraim_prob_3 := 1 / 8
  (keiko_prob_0 * ephraim_prob_0) 
  + (keiko_prob_1 * ephraim_prob_1) 
  + (keiko_prob_2 * ephraim_prob_2)

theorem ephraim_keiko_same_heads_probability : 
  coin_toss_probability_same_heads = 11 / 32 :=
by 
  unfold coin_toss_probability_same_heads
  norm_num
  sorry

end NUMINAMATH_GPT_ephraim_keiko_same_heads_probability_l1623_162348


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1623_162381

-- Given definitions from the problem
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}

-- The theorem to be proven
theorem complement_of_A_in_U : U \ A = {1, 3, 6, 7} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1623_162381


namespace NUMINAMATH_GPT_moles_of_HCl_formed_l1623_162369

theorem moles_of_HCl_formed
  (C2H6_initial : Nat)
  (Cl2_initial : Nat)
  (HCl_expected : Nat)
  (balanced_reaction : C2H6_initial + Cl2_initial = C2H6_initial + HCl_expected):
  C2H6_initial = 2 → Cl2_initial = 2 → HCl_expected = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_moles_of_HCl_formed_l1623_162369


namespace NUMINAMATH_GPT_find_slope_l1623_162383

theorem find_slope (k : ℝ) :
  (∀ x y : ℝ, y = -2 * x + 3 → y = k * x + 4 → (x, y) = (1, 1)) → k = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_slope_l1623_162383


namespace NUMINAMATH_GPT_value_of_y_for_absolute_value_eq_zero_l1623_162360

theorem value_of_y_for_absolute_value_eq_zero :
  ∃ (y : ℚ), |(2:ℚ) * y - 3| ≤ 0 ↔ y = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_for_absolute_value_eq_zero_l1623_162360


namespace NUMINAMATH_GPT_mask_digit_correctness_l1623_162359

noncomputable def elephant_mask_digit : ℕ :=
  6
  
noncomputable def mouse_mask_digit : ℕ :=
  4

noncomputable def guinea_pig_mask_digit : ℕ :=
  8

noncomputable def panda_mask_digit : ℕ :=
  1

theorem mask_digit_correctness :
  (∃ (d1 d2 d3 d4 : ℕ), d1 * d1 = 16 ∧ d2 * d2 = 64 ∧ d3 * d3 = 49 ∧ d4 * d4 = 81) →
  elephant_mask_digit = 6 ∧ mouse_mask_digit = 4 ∧ guinea_pig_mask_digit = 8 ∧ panda_mask_digit = 1 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_mask_digit_correctness_l1623_162359


namespace NUMINAMATH_GPT_mars_colony_cost_l1623_162338

theorem mars_colony_cost :
  let total_cost := 45000000000
  let number_of_people := 300000000
  total_cost / number_of_people = 150 := 
by sorry

end NUMINAMATH_GPT_mars_colony_cost_l1623_162338


namespace NUMINAMATH_GPT_measure_of_angle_Q_l1623_162390

theorem measure_of_angle_Q (a b c d e Q : ℝ)
  (ha : a = 138) (hb : b = 85) (hc : c = 130) (hd : d = 120) (he : e = 95)
  (h_hex : a + b + c + d + e + Q = 720) : 
  Q = 152 :=
by
  rw [ha, hb, hc, hd, he] at h_hex
  linarith

end NUMINAMATH_GPT_measure_of_angle_Q_l1623_162390


namespace NUMINAMATH_GPT_total_games_played_l1623_162346

-- Definition of the conditions
def teams : Nat := 10
def games_per_pair : Nat := 4

-- Statement of the problem
theorem total_games_played (teams games_per_pair : Nat) : 
  teams = 10 → 
  games_per_pair = 4 → 
  ∃ total_games, total_games = 180 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_total_games_played_l1623_162346


namespace NUMINAMATH_GPT_min_h4_for_ahai_avg_ge_along_avg_plus_4_l1623_162328

-- Definitions from conditions
variables (a1 a2 a3 a4 : ℝ)
variables (h1 h2 h3 h4 : ℝ)

-- Conditions from the problem
axiom a1_gt_80 : a1 > 80
axiom a2_gt_80 : a2 > 80
axiom a3_gt_80 : a3 > 80
axiom a4_gt_80 : a4 > 80

axiom h1_eq_a1_plus_1 : h1 = a1 + 1
axiom h2_eq_a2_plus_2 : h2 = a2 + 2
axiom h3_eq_a3_plus_3 : h3 = a3 + 3

-- Lean 4 statement for the problem
theorem min_h4_for_ahai_avg_ge_along_avg_plus_4 : h4 ≥ 99 :=
by
  sorry

end NUMINAMATH_GPT_min_h4_for_ahai_avg_ge_along_avg_plus_4_l1623_162328


namespace NUMINAMATH_GPT_f_bound_l1623_162319

noncomputable def f : ℕ+ → ℝ := sorry

axiom f_1 : f 1 = 3 / 2
axiom f_ineq (x y : ℕ+) : f (x + y) ≥ (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2

theorem f_bound (x : ℕ+) : f x ≥ 1 / 4 * x * (x + 1) * (2 * x + 1) := sorry

end NUMINAMATH_GPT_f_bound_l1623_162319


namespace NUMINAMATH_GPT_inequality1_inequality2_l1623_162340

theorem inequality1 (x : ℝ) : 
  x^2 - 2 * x - 1 > 0 -> x > Real.sqrt 2 + 1 ∨ x < -Real.sqrt 2 + 1 := 
by sorry

theorem inequality2 (x : ℝ) : 
  (2 * x - 1) / (x - 3) ≥ 3 -> 3 < x ∧ x <= 8 := 
by sorry

end NUMINAMATH_GPT_inequality1_inequality2_l1623_162340


namespace NUMINAMATH_GPT_probability_at_least_two_green_l1623_162314

def total_apples := 10
def red_apples := 6
def green_apples := 4
def choose_apples := 3

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_at_least_two_green :
  (binomial green_apples 3 + binomial green_apples 2 * binomial red_apples 1) = 40 ∧ 
  binomial total_apples choose_apples = 120 ∧
  (binomial green_apples 3 + binomial green_apples 2 * binomial red_apples 1) / binomial total_apples choose_apples = 1 / 3 := by
sorry

end NUMINAMATH_GPT_probability_at_least_two_green_l1623_162314


namespace NUMINAMATH_GPT_remaining_subtasks_l1623_162345

def total_problems : ℝ := 72.0
def finished_problems : ℝ := 32.0
def subtasks_per_problem : ℕ := 5

theorem remaining_subtasks :
    (total_problems * subtasks_per_problem - finished_problems * subtasks_per_problem) = 200 := 
by
  sorry

end NUMINAMATH_GPT_remaining_subtasks_l1623_162345


namespace NUMINAMATH_GPT_part1_part2_l1623_162367

open Real

variables {a b c : ℝ}

theorem part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1 / a + 1 / b + 1 / c = 1) :
    a + 4 * b + 9 * c ≥ 36 :=
sorry

theorem part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1 / a + 1 / b + 1 / c = 1) :
    (b + c) / sqrt a + (a + c) / sqrt b + (a + b) / sqrt c ≥ 2 * sqrt (a * b * c) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1623_162367


namespace NUMINAMATH_GPT_property_check_l1623_162303

noncomputable def f (x : ℝ) : ℤ := ⌈x⌉ -- Define the ceiling function

theorem property_check :
  (¬ (∀ x : ℝ, f (2 * x) = 2 * f x)) ∧
  (∀ x1 x2 : ℝ, f x1 = f x2 → |x1 - x2| < 1) ∧
  (∀ x1 x2 : ℝ, f (x1 + x2) ≤ f x1 + f x2) ∧
  (¬ (∀ x : ℝ, f x + f (x + 0.5) = f (2 * x))) :=
by
  sorry

end NUMINAMATH_GPT_property_check_l1623_162303


namespace NUMINAMATH_GPT_complement_M_in_U_l1623_162339

open Set

theorem complement_M_in_U : 
  let U : Set ℕ := {1, 3, 5, 7}
  let M : Set ℕ := {1, 5}
  U \ M = {3, 7} := 
by
  let U : Set ℕ := {1, 3, 5, 7}
  let M : Set ℕ := {1, 5}
  sorry

end NUMINAMATH_GPT_complement_M_in_U_l1623_162339


namespace NUMINAMATH_GPT_recommended_daily_serving_l1623_162336

theorem recommended_daily_serving (mg_per_pill : ℕ) (pills_per_week : ℕ) (total_mg_week : ℕ) (days_per_week : ℕ) 
  (h1 : mg_per_pill = 50) (h2 : pills_per_week = 28) (h3 : total_mg_week = pills_per_week * mg_per_pill) 
  (h4 : days_per_week = 7) : 
  total_mg_week / days_per_week = 200 :=
by
  sorry

end NUMINAMATH_GPT_recommended_daily_serving_l1623_162336


namespace NUMINAMATH_GPT_part1_part2_l1623_162380

-- Part (1)
theorem part1 (B : ℝ) (b : ℝ) (S : ℝ) (a c : ℝ) (B_eq : B = Real.pi / 3) 
  (b_eq : b = Real.sqrt 7) (S_eq : S = (3 * Real.sqrt 3) / 2) :
  a + c = 5 := 
sorry

-- Part (2)
theorem part2 (C : ℝ) (c : ℝ) (dot_BA_BC AB_AC : ℝ) 
  (C_cond : 2 * Real.cos C * (dot_BA_BC + AB_AC) = c^2) :
  C = Real.pi / 3 := 
sorry

end NUMINAMATH_GPT_part1_part2_l1623_162380


namespace NUMINAMATH_GPT_cube_surface_area_l1623_162391

theorem cube_surface_area (side_length : ℝ) (h : side_length = 8) : 6 * side_length^2 = 384 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_cube_surface_area_l1623_162391


namespace NUMINAMATH_GPT_a_pow_11_b_pow_11_l1623_162372

theorem a_pow_11_b_pow_11 (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 :=
by
  sorry

end NUMINAMATH_GPT_a_pow_11_b_pow_11_l1623_162372


namespace NUMINAMATH_GPT_people_in_room_l1623_162399

theorem people_in_room (total_chairs seated_chairs total_people : ℕ) 
  (h1 : 3 * total_people = 5 * seated_chairs)
  (h2 : 4 * total_chairs = 5 * seated_chairs) 
  (h3 : total_chairs - seated_chairs = 8) : 
  total_people = 54 :=
by
  sorry

end NUMINAMATH_GPT_people_in_room_l1623_162399


namespace NUMINAMATH_GPT_pavan_travel_distance_l1623_162393

theorem pavan_travel_distance (t : ℝ) (v1 v2 : ℝ) (D : ℝ) (h₁ : t = 15) (h₂ : v1 = 30) (h₃ : v2 = 25):
  (D / 2) / v1 + (D / 2) / v2 = t → D = 2250 / 11 :=
by
  intro h
  rw [h₁, h₂, h₃] at h
  sorry

end NUMINAMATH_GPT_pavan_travel_distance_l1623_162393


namespace NUMINAMATH_GPT_distribution_ways_l1623_162357

-- Define the conditions
def num_papers : ℕ := 7
def num_friends : ℕ := 10

-- Define the theorem to prove the number of ways to distribute the papers
theorem distribution_ways : (num_friends ^ num_papers) = 10000000 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_distribution_ways_l1623_162357


namespace NUMINAMATH_GPT_sum_of_squares_nonnegative_l1623_162330

theorem sum_of_squares_nonnegative (x y z : ℝ) : x^2 + y^2 + z^2 - x * y - x * z - y * z ≥ 0 :=
  sorry

end NUMINAMATH_GPT_sum_of_squares_nonnegative_l1623_162330


namespace NUMINAMATH_GPT_arithmetic_geometric_sum_l1623_162308

theorem arithmetic_geometric_sum (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h_arith : 2 * b = a + c) (h_geom : a^2 = b * c) 
  (h_sum : a + 3 * b + c = 10) : a = -4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sum_l1623_162308


namespace NUMINAMATH_GPT_sum_of_first_21_terms_l1623_162305

def is_constant_sum_sequence (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n, a n + a (n + 1) = c

theorem sum_of_first_21_terms (a : ℕ → ℕ) (h1 : is_constant_sum_sequence a 5) (h2 : a 1 = 2) : (Finset.range 21).sum a = 52 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_21_terms_l1623_162305


namespace NUMINAMATH_GPT_algebraic_expression_value_l1623_162331

theorem algebraic_expression_value (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1623_162331


namespace NUMINAMATH_GPT_number_of_boxes_l1623_162396

def magazines : ℕ := 63
def magazines_per_box : ℕ := 9

theorem number_of_boxes : magazines / magazines_per_box = 7 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_boxes_l1623_162396


namespace NUMINAMATH_GPT_calc_expression_l1623_162375

theorem calc_expression : (2019 / 2018) - (2018 / 2019) = 4037 / 4036 := 
by sorry

end NUMINAMATH_GPT_calc_expression_l1623_162375


namespace NUMINAMATH_GPT_glass_ball_radius_l1623_162349

theorem glass_ball_radius (x y r : ℝ) (h_parabola : x^2 = 2 * y) (h_touch : y = r) (h_range : 0 ≤ y ∧ y ≤ 20) : 0 < r ∧ r ≤ 1 :=
sorry

end NUMINAMATH_GPT_glass_ball_radius_l1623_162349


namespace NUMINAMATH_GPT_inequality_solution_l1623_162341

theorem inequality_solution (x : ℝ) : 2 * x - 1 ≤ 3 → x ≤ 2 :=
by
  intro h
  -- Here we would perform the solution steps, but we'll skip the proof with sorry.
  sorry

end NUMINAMATH_GPT_inequality_solution_l1623_162341


namespace NUMINAMATH_GPT_distance_between_two_girls_after_12_hours_l1623_162355

theorem distance_between_two_girls_after_12_hours :
  let speed1 := 7 -- speed of the first girl (km/hr)
  let speed2 := 3 -- speed of the second girl (km/hr)
  let time := 12 -- time (hours)
  let distance1 := speed1 * time -- distance traveled by the first girl
  let distance2 := speed2 * time -- distance traveled by the second girl
  distance1 + distance2 = 120 := -- total distance
by
  -- Here, we would provide the proof, but we put sorry to skip it
  sorry

end NUMINAMATH_GPT_distance_between_two_girls_after_12_hours_l1623_162355


namespace NUMINAMATH_GPT_solve_log_equation_l1623_162362

theorem solve_log_equation :
  ∀ x : ℝ, 
  5 * Real.logb x (x / 9) + Real.logb (x / 9) x^3 + 8 * Real.logb (9 * x^2) (x^2) = 2
  → (x = 3 ∨ x = Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_solve_log_equation_l1623_162362


namespace NUMINAMATH_GPT_false_statements_l1623_162300

variable (a b c : ℝ)

theorem false_statements (a b c : ℝ) :
  ¬(a > b → a^2 > b^2) ∧ ¬((a^2 > b^2) → a > b) ∧ ¬(a > b → a * c^2 > b * c^2) ∧ ¬(a > b ↔ |a| > |b|) :=
by
  sorry

end NUMINAMATH_GPT_false_statements_l1623_162300


namespace NUMINAMATH_GPT_area_of_circumscribed_circle_eq_48pi_l1623_162324

noncomputable def side_length := 12
noncomputable def radius := (2/3) * (side_length / 2) * (Real.sqrt 3)
noncomputable def area := Real.pi * radius^2

theorem area_of_circumscribed_circle_eq_48pi :
  area = 48 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_circumscribed_circle_eq_48pi_l1623_162324


namespace NUMINAMATH_GPT_find_a_l1623_162387

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem find_a (a : ℝ) (h : binomial_coefficient 4 2 + 4 * a = 10) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1623_162387


namespace NUMINAMATH_GPT_quadratic_expression_value_l1623_162315

theorem quadratic_expression_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : a + b - 1 = 1) : (1 - a - b) = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_value_l1623_162315


namespace NUMINAMATH_GPT_employees_paid_per_shirt_l1623_162311

theorem employees_paid_per_shirt:
  let num_employees := 20
  let shirts_per_employee_per_day := 20
  let hours_per_shift := 8
  let wage_per_hour := 12
  let price_per_shirt := 35
  let nonemployee_expenses_per_day := 1000
  let profit_per_day := 9080
  let total_shirts_made_per_day := num_employees * shirts_per_employee_per_day
  let total_daily_wages := num_employees * hours_per_shift * wage_per_hour
  let total_revenue := total_shirts_made_per_day * price_per_shirt
  let per_shirt_payment := (total_revenue - (total_daily_wages + nonemployee_expenses_per_day)) / total_shirts_made_per_day
  per_shirt_payment = 27.70 :=
sorry

end NUMINAMATH_GPT_employees_paid_per_shirt_l1623_162311


namespace NUMINAMATH_GPT_dozen_pencils_l1623_162350

-- Define the given conditions
def pencils_total : ℕ := 144
def pencils_per_dozen : ℕ := 12

-- Theorem stating the desired proof
theorem dozen_pencils (h : pencils_total = 144) (hdozen : pencils_per_dozen = 12) : 
  pencils_total / pencils_per_dozen = 12 :=
by
  sorry

end NUMINAMATH_GPT_dozen_pencils_l1623_162350


namespace NUMINAMATH_GPT_area_of_triangle_l1623_162356

-- Define the lines as functions
def line1 : ℝ → ℝ := fun x => 3 * x - 4
def line2 : ℝ → ℝ := fun x => -2 * x + 16

-- Define the vertices of the triangle formed by lines and y-axis
def vertex1 : ℝ × ℝ := (0, -4)
def vertex2 : ℝ × ℝ := (0, 16)
def vertex3 : ℝ × ℝ := (4, 8)

-- Define the proof statement
theorem area_of_triangle : 
  let A := vertex1 
  let B := vertex2 
  let C := vertex3 
  -- Compute the area of the triangle using the determinant formula
  let area := (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))
  area = 40 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1623_162356


namespace NUMINAMATH_GPT_medians_square_sum_l1623_162361

theorem medians_square_sum (a b c : ℝ) (ha : a = 13) (hb : b = 13) (hc : c = 10) :
  let m_a := (1 / 2 * (2 * b^2 + 2 * c^2 - a^2))^(1/2)
  let m_b := (1 / 2 * (2 * c^2 + 2 * a^2 - b^2))^(1/2)
  let m_c := (1 / 2 * (2 * a^2 + 2 * b^2 - c^2))^(1/2)
  m_a^2 + m_b^2 + m_c^2 = 432 :=
by
  sorry

end NUMINAMATH_GPT_medians_square_sum_l1623_162361


namespace NUMINAMATH_GPT_roots_of_quadratic_l1623_162385

theorem roots_of_quadratic (x : ℝ) : x^2 - 5 * x = 0 ↔ (x = 0 ∨ x = 5) := by 
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1623_162385


namespace NUMINAMATH_GPT_umar_age_is_ten_l1623_162329

-- Define variables for Ali, Yusaf, and Umar
variables (ali_age yusa_age umar_age : ℕ)

-- Define the conditions from the problem
def ali_is_eight : Prop := ali_age = 8
def ali_older_than_yusaf : Prop := ali_age - yusa_age = 3
def umar_twice_yusaf : Prop := umar_age = 2 * yusa_age

-- The theorem that uses the conditions to assert Umar's age
theorem umar_age_is_ten 
  (h1 : ali_is_eight ali_age)
  (h2 : ali_older_than_yusaf ali_age yusa_age)
  (h3 : umar_twice_yusaf umar_age yusa_age) : 
  umar_age = 10 :=
by
  sorry

end NUMINAMATH_GPT_umar_age_is_ten_l1623_162329


namespace NUMINAMATH_GPT_simplify_expression_l1623_162325

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 1 - (2*x - 2)/(x + 1)) / ((x^2 - x) / (2*x + 2)) = 2 - Real.sqrt 2 := 
by
  -- Here we should include the proof steps, but we skip it with "sorry"
  sorry

end NUMINAMATH_GPT_simplify_expression_l1623_162325


namespace NUMINAMATH_GPT_multiples_of_231_l1623_162335

theorem multiples_of_231 (h : ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ 99 → i % 2 = 1 → 231 ∣ 10^j - 10^i) :
  ∃ n, n = 416 :=
by sorry

end NUMINAMATH_GPT_multiples_of_231_l1623_162335


namespace NUMINAMATH_GPT_cos_sum_of_arctan_roots_l1623_162306

theorem cos_sum_of_arctan_roots (α β : ℝ) (hα : -π/2 < α ∧ α < 0) (hβ : -π/2 < β ∧ β < 0) 
  (h1 : Real.tan α + Real.tan β = -3 * Real.sqrt 3) 
  (h2 : Real.tan α * Real.tan β = 4) : 
  Real.cos (α + β) = - 1 / 2 :=
sorry

end NUMINAMATH_GPT_cos_sum_of_arctan_roots_l1623_162306


namespace NUMINAMATH_GPT_sin_theta_minus_cos_theta_l1623_162392

theorem sin_theta_minus_cos_theta (θ : ℝ) (b : ℝ) (hθ_acute : 0 < θ ∧ θ < π / 2) (h_cos2θ : Real.cos (2 * θ) = b) :
  ∃ x, x = Real.sin θ - Real.cos θ ∧ (x = Real.sqrt b ∨ x = -Real.sqrt b) := 
by
  sorry

end NUMINAMATH_GPT_sin_theta_minus_cos_theta_l1623_162392


namespace NUMINAMATH_GPT_paula_remaining_money_l1623_162378

theorem paula_remaining_money 
  (M : Int) (C_s : Int) (N_s : Int) (C_p : Int) (N_p : Int)
  (h1 : M = 250) 
  (h2 : C_s = 15) 
  (h3 : N_s = 5) 
  (h4 : C_p = 25) 
  (h5 : N_p = 3) : 
  M - (C_s * N_s + C_p * N_p) = 100 := 
by
  sorry

end NUMINAMATH_GPT_paula_remaining_money_l1623_162378


namespace NUMINAMATH_GPT_new_quadratic_coeff_l1623_162398

theorem new_quadratic_coeff (r s p q : ℚ) 
  (h1 : 3 * r^2 + 4 * r + 2 = 0)
  (h2 : 3 * s^2 + 4 * s + 2 = 0)
  (h3 : r + s = -4 / 3)
  (h4 : r * s = 2 / 3) 
  (h5 : r^3 + s^3 = - p) :
  p = 16 / 27 :=
by
  sorry

end NUMINAMATH_GPT_new_quadratic_coeff_l1623_162398


namespace NUMINAMATH_GPT_tickets_difference_l1623_162376

def number_of_tickets_for_toys := 31
def number_of_tickets_for_clothes := 14

theorem tickets_difference : number_of_tickets_for_toys - number_of_tickets_for_clothes = 17 := by
  sorry

end NUMINAMATH_GPT_tickets_difference_l1623_162376


namespace NUMINAMATH_GPT_part1_part2_l1623_162358

-- Problem 1: Given |x| = 9, |y| = 5, x < 0, y > 0, prove x + y = -4
theorem part1 (x y : ℚ) (h1 : |x| = 9) (h2 : |y| = 5) (h3 : x < 0) (h4 : y > 0) : x + y = -4 :=
sorry

-- Problem 2: Given |x| = 9, |y| = 5, |x + y| = x + y, prove x - y = 4 or x - y = 14
theorem part2 (x y : ℚ) (h1 : |x| = 9) (h2 : |y| = 5) (h3 : |x + y| = x + y) : x - y = 4 ∨ x - y = 14 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1623_162358


namespace NUMINAMATH_GPT_find_polynomial_h_l1623_162382

theorem find_polynomial_h (f h : ℝ → ℝ) (hf : ∀ x, f x = x^2) (hh : ∀ x, f (h x) = 9 * x^2 + 6 * x + 1) : 
  (∀ x, h x = 3 * x + 1) ∨ (∀ x, h x = -3 * x - 1) :=
by
  sorry

end NUMINAMATH_GPT_find_polynomial_h_l1623_162382


namespace NUMINAMATH_GPT_log_x3y2_value_l1623_162386

open Real

noncomputable def log_identity (x y : ℝ) : Prop :=
  log (x * y^4) = 1 ∧ log (x^3 * y) = 1

theorem log_x3y2_value (x y : ℝ) (h : log_identity x y) : log (x^3 * y^2) = 13 / 11 :=
  by
  sorry

end NUMINAMATH_GPT_log_x3y2_value_l1623_162386


namespace NUMINAMATH_GPT_functions_same_l1623_162327

theorem functions_same (x : ℝ) : (∀ x, (y = x) → (∀ x, (y = (x^3 + x) / (x^2 + 1)))) :=
by sorry

end NUMINAMATH_GPT_functions_same_l1623_162327


namespace NUMINAMATH_GPT_Frank_initial_savings_l1623_162351

theorem Frank_initial_savings 
  (cost_per_toy : Nat)
  (number_of_toys : Nat)
  (allowance : Nat)
  (total_cost : Nat)
  (initial_savings : Nat)
  (h1 : cost_per_toy = 8)
  (h2 : number_of_tys = 5)
  (h3 : allowance = 37)
  (h4 : total_cost = number_of_toys * cost_per_toy)
  (h5 : initial_savings + allowance = total_cost)
  : initial_savings = 3 := 
by
  sorry

end NUMINAMATH_GPT_Frank_initial_savings_l1623_162351


namespace NUMINAMATH_GPT_negation_of_p_converse_of_p_inverse_of_p_contrapositive_of_p_original_p_l1623_162371

theorem negation_of_p (π : ℝ) (a b c d : ℚ) (h : a * π + b = c * π + d) : a ≠ c ∨ b ≠ d :=
  sorry

theorem converse_of_p (π : ℝ) (a b c d : ℚ) (h : a = c ∧ b = d) : a * π + b = c * π + d :=
  sorry

theorem inverse_of_p (π : ℝ) (a b c d : ℚ) (h : a * π + b ≠ c * π + d) : a ≠ c ∨ b ≠ d :=
  sorry

theorem contrapositive_of_p (π : ℝ) (a b c d : ℚ) (h : a ≠ c ∨ b ≠ d) : a * π + b ≠ c * π + d :=
  sorry

theorem original_p (π : ℝ) (a b c d : ℚ) (h : a * π + b = c * π + d) : a = c ∧ b = d :=
  sorry

end NUMINAMATH_GPT_negation_of_p_converse_of_p_inverse_of_p_contrapositive_of_p_original_p_l1623_162371


namespace NUMINAMATH_GPT_find_coordinates_of_A_l1623_162397

theorem find_coordinates_of_A (x : ℝ) :
  let A := (x, 1, 2)
  let B := (2, 3, 4)
  (Real.sqrt ((x - 2)^2 + (1 - 3)^2 + (2 - 4)^2) = 2 * Real.sqrt 6) →
  (x = 6 ∨ x = -2) := 
by
  intros
  sorry

end NUMINAMATH_GPT_find_coordinates_of_A_l1623_162397


namespace NUMINAMATH_GPT_a2b2_div_ab1_is_square_l1623_162389

theorem a2b2_div_ab1_is_square (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : (ab + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, (a^2 + b^2) / (ab + 1) = k^2 :=
sorry

end NUMINAMATH_GPT_a2b2_div_ab1_is_square_l1623_162389


namespace NUMINAMATH_GPT_discounted_price_of_russian_doll_l1623_162318

theorem discounted_price_of_russian_doll (original_price : ℕ) (number_of_dolls_original : ℕ) (number_of_dolls_discounted : ℕ) (discounted_price : ℕ) :
  original_price = 4 →
  number_of_dolls_original = 15 →
  number_of_dolls_discounted = 20 →
  (number_of_dolls_original * original_price) = 60 →
  (number_of_dolls_discounted * discounted_price) = 60 →
  discounted_price = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_discounted_price_of_russian_doll_l1623_162318


namespace NUMINAMATH_GPT_find_y_l1623_162353

variable {L B y : ℝ}

theorem find_y (h1 : 2 * ((L + y) + (B + y)) - 2 * (L + B) = 16) : y = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1623_162353


namespace NUMINAMATH_GPT_students_gold_award_freshmen_l1623_162337

theorem students_gold_award_freshmen 
    (total_students total_award_winners : ℕ)
    (students_selected exchange_meeting : ℕ)
    (freshmen_selected gold_award_selected : ℕ)
    (prop1 : total_award_winners = 120)
    (prop2 : exchange_meeting = 24)
    (prop3 : freshmen_selected = 6)
    (prop4 : gold_award_selected = 4) :
    ∃ (gold_award_students : ℕ), gold_award_students = 4 ∧ gold_award_students ≤ freshmen_selected :=
by
  sorry

end NUMINAMATH_GPT_students_gold_award_freshmen_l1623_162337


namespace NUMINAMATH_GPT_flag_covering_proof_l1623_162322

def grid_covering_flag_ways (m n num_flags cells_per_flag : ℕ) :=
  if m * n / cells_per_flag = num_flags then 2^num_flags else 0

theorem flag_covering_proof :
  grid_covering_flag_ways 9 18 18 9 = 262144 := by
  sorry

end NUMINAMATH_GPT_flag_covering_proof_l1623_162322


namespace NUMINAMATH_GPT_square_cookie_cutters_count_l1623_162370

def triangles_sides : ℕ := 6 * 3
def hexagons_sides : ℕ := 2 * 6
def total_sides : ℕ := 46
def sides_from_squares (S : ℕ) : ℕ := S * 4

theorem square_cookie_cutters_count (S : ℕ) :
  triangles_sides + hexagons_sides + sides_from_squares S = total_sides → S = 4 :=
by
  sorry

end NUMINAMATH_GPT_square_cookie_cutters_count_l1623_162370


namespace NUMINAMATH_GPT_m_n_sum_l1623_162374

theorem m_n_sum (m n : ℝ) (h : ∀ x : ℝ, x^2 + m * x + 6 = (x - 2) * (x - n)) : m + n = -2 :=
by
  sorry

end NUMINAMATH_GPT_m_n_sum_l1623_162374


namespace NUMINAMATH_GPT_residue_of_minus_963_plus_100_mod_35_l1623_162347

-- Defining the problem in Lean 4
theorem residue_of_minus_963_plus_100_mod_35 : 
  ((-963 + 100) % 35) = 12 :=
by
  sorry

end NUMINAMATH_GPT_residue_of_minus_963_plus_100_mod_35_l1623_162347


namespace NUMINAMATH_GPT_prime_ge_5_div_24_l1623_162320

theorem prime_ge_5_div_24 (p : ℕ) (hp : Prime p) (hp_ge_5 : p ≥ 5) : 24 ∣ p^2 - 1 := 
sorry

end NUMINAMATH_GPT_prime_ge_5_div_24_l1623_162320
