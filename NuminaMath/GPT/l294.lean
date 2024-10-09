import Mathlib

namespace cannot_contain_point_1997_0_l294_29441

variable {m b : ℝ}

theorem cannot_contain_point_1997_0 (h : m * b > 0) : ¬ (0 = 1997 * m + b) := sorry

end cannot_contain_point_1997_0_l294_29441


namespace horner_method_v2_l294_29490

def f(x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.reverse.foldl (λ acc c => acc * x + c) 0

theorem horner_method_v2 :
  horner_eval [1, 5, 10, 10, 5, 1] 2 = 24 :=
by
  sorry

end horner_method_v2_l294_29490


namespace smallest_n_l294_29424

theorem smallest_n (n : ℕ) : 
  (∃ k : ℕ, 4 * n = k^2) ∧ (∃ l : ℕ, 5 * n = l^5) ↔ n = 625 :=
by sorry

end smallest_n_l294_29424


namespace angle_ACE_is_38_l294_29429

noncomputable def measure_angle_ACE (A B C D E : Type) : Prop :=
  let angle_ABC := 55
  let angle_BCA := 38
  let angle_BAC := 87
  let angle_ABD := 125
  (angle_ABC + angle_ABD = 180) → -- supplementary condition
  (angle_BAC = 87) → -- given angle at BAC
  (let angle_ACB := 180 - angle_BAC - angle_ABC;
   angle_ACB = angle_BCA ∧  -- derived angle at BCA
   angle_ACB = 38) → -- target angle
  (angle_BCA = 38) -- final result that needs to be proven

theorem angle_ACE_is_38 {A B C D E : Type} :
  measure_angle_ACE A B C D E :=
by
  sorry

end angle_ACE_is_38_l294_29429


namespace sequence_general_formula_l294_29493

def sequence_term (n : ℕ) : ℕ :=
  if n = 0 then 3 else 3 + n * 5 

theorem sequence_general_formula (n : ℕ) : n > 0 → sequence_term n = 5 * n - 2 :=
by 
  sorry

end sequence_general_formula_l294_29493


namespace angela_more_marbles_l294_29409

/--
Albert has three times as many marbles as Angela.
Allison has 28 marbles.
Albert and Allison have 136 marbles together.
Prove that Angela has 8 more marbles than Allison.
-/
theorem angela_more_marbles 
  (albert_angela : ℕ) 
  (angela: ℕ) 
  (albert: ℕ) 
  (allison: ℕ) 
  (h_albert_is_three_times_angela : albert = 3 * angela) 
  (h_allison_is_28 : allison = 28) 
  (h_albert_allison_is_136 : albert + allison = 136) 
  : angela - allison = 8 := 
by
  sorry

end angela_more_marbles_l294_29409


namespace simplify_expression_l294_29451

variables {R : Type*} [CommRing R]

theorem simplify_expression (x y : R) : (3 * x^2 * y^3)^4 = 81 * x^8 * y^12 := by
  sorry

end simplify_expression_l294_29451


namespace points_per_touchdown_l294_29477

theorem points_per_touchdown (total_points touchdowns : ℕ) (h1 : total_points = 21) (h2 : touchdowns = 3) :
  total_points / touchdowns = 7 :=
by
  sorry

end points_per_touchdown_l294_29477


namespace ellipse_foci_distance_l294_29423

theorem ellipse_foci_distance (a b : ℝ) (ha : a = 10) (hb : b = 8) :
  2 * Real.sqrt (a^2 - b^2) = 12 :=
by
  rw [ha, hb]
  -- Proof follows here, but we skip it using sorry.
  sorry

end ellipse_foci_distance_l294_29423


namespace pow_congr_mod_eight_l294_29465

theorem pow_congr_mod_eight (n : ℕ) : (5^n + 2 * 3^(n-1) + 1) % 8 = 0 := sorry

end pow_congr_mod_eight_l294_29465


namespace DennisHas70Marbles_l294_29483

-- Definitions according to the conditions
def LaurieMarbles : Nat := 37
def KurtMarbles : Nat := LaurieMarbles - 12
def DennisMarbles : Nat := KurtMarbles + 45

-- The proof problem statement
theorem DennisHas70Marbles : DennisMarbles = 70 :=
by
  sorry

end DennisHas70Marbles_l294_29483


namespace pencils_more_than_200_on_saturday_l294_29415

theorem pencils_more_than_200_on_saturday 
    (p : ℕ → ℕ) 
    (h_start : p 1 = 3)
    (h_next_day : ∀ n, p (n + 1) = (p n + 2) * 2) 
    : p 6 > 200 :=
by
  -- Proof steps can be filled in here.
  sorry

end pencils_more_than_200_on_saturday_l294_29415


namespace smallest_four_digit_product_is_12_l294_29468

theorem smallest_four_digit_product_is_12 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
           (∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧ a * b * c * d = 12 ∧ a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 6) ∧
           (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 →
                     (∃ a' b' c' d' : ℕ, m = 1000 * a' + 100 * b' + 10 * c' + d' ∧ a' * b' * c' * d' = 12) →
                     n ≤ m) :=
by
  sorry

end smallest_four_digit_product_is_12_l294_29468


namespace fraction_simplifies_correctly_l294_29463

variable (a b : ℕ)

theorem fraction_simplifies_correctly (h : a ≠ b) : (1/2 * a) / (1/2 * b) = a / b := 
by sorry

end fraction_simplifies_correctly_l294_29463


namespace average_without_ivan_l294_29431

theorem average_without_ivan
  (total_friends : ℕ := 5)
  (avg_all : ℝ := 55)
  (ivan_amount : ℝ := 43)
  (remaining_friends : ℕ := total_friends - 1)
  (total_amount : ℝ := total_friends * avg_all)
  (remaining_amount : ℝ := total_amount - ivan_amount)
  (new_avg : ℝ := remaining_amount / remaining_friends) :
  new_avg = 58 := 
sorry

end average_without_ivan_l294_29431


namespace trigonometric_identity_l294_29482

theorem trigonometric_identity (α : ℝ) (h : Real.tan (π + α) = 2) :
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π + α) - Real.cos (π + α)) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l294_29482


namespace casey_saving_l294_29461

-- Define the conditions
def cost_per_hour_first_employee : ℝ := 20
def cost_per_hour_second_employee : ℝ := 22
def subsidy_per_hour : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost calculations
def weekly_cost_first_employee := cost_per_hour_first_employee * hours_per_week
def effective_cost_per_hour_second_employee := cost_per_hour_second_employee - subsidy_per_hour
def weekly_cost_second_employee := effective_cost_per_hour_second_employee * hours_per_week

-- State the theorem
theorem casey_saving :
    weekly_cost_first_employee - weekly_cost_second_employee = 160 := 
by
  sorry

end casey_saving_l294_29461


namespace crayons_total_cost_l294_29447

theorem crayons_total_cost :
  let packs_initial := 4
  let packs_to_buy := 2
  let cost_per_pack := 2.5
  let total_packs := packs_initial + packs_to_buy
  let total_cost := total_packs * cost_per_pack
  total_cost = 15 :=
by
  sorry

end crayons_total_cost_l294_29447


namespace marcus_sees_7_l294_29426

variable (marcus humphrey darrel : ℕ)
variable (humphrey_sees : humphrey = 11)
variable (darrel_sees : darrel = 9)
variable (average_is_9 : (marcus + humphrey + darrel) / 3 = 9)

theorem marcus_sees_7 : marcus = 7 :=
by
  -- Needs proof
  sorry

end marcus_sees_7_l294_29426


namespace base8_satisfies_l294_29471

noncomputable def check_base (c : ℕ) : Prop := 
  ((2 * c ^ 2 + 4 * c + 3) + (1 * c ^ 2 + 5 * c + 6)) = (4 * c ^ 2 + 2 * c + 1)

theorem base8_satisfies : check_base 8 := 
by
  -- conditions: (243_c, 156_c, 421_c) translated as provided
  -- proof is skipped here as specified
  sorry

end base8_satisfies_l294_29471


namespace max_pencils_thrown_out_l294_29460

theorem max_pencils_thrown_out (n : ℕ) : (n % 7 ≤ 6) :=
by
  sorry

end max_pencils_thrown_out_l294_29460


namespace radius_of_circle_l294_29422

theorem radius_of_circle (r : ℝ) (C : ℝ) (A : ℝ) (h1 : 3 * C = 2 * A) 
  (h2 : C = 2 * Real.pi * r) (h3 : A = Real.pi * r^2) : 
  r = 3 :=
by 
  sorry

end radius_of_circle_l294_29422


namespace largest_value_fraction_l294_29475

theorem largest_value_fraction (x y : ℝ) (hx : 10 ≤ x ∧ x ≤ 20) (hy : 40 ≤ y ∧ y ≤ 60) :
  ∃ z, z = (x^2 / (2 * y)) ∧ z ≤ 5 :=
by
  sorry

end largest_value_fraction_l294_29475


namespace elaine_earnings_increase_l294_29438

variable (E : ℝ) -- Elaine's earnings last year
variable (P : ℝ) -- Percentage increase in earnings

-- Conditions
variable (rent_last_year : ℝ := 0.20 * E)
variable (earnings_this_year : ℝ := E * (1 + P / 100))
variable (rent_this_year : ℝ := 0.30 * earnings_this_year)
variable (multiplied_rent_last_year : ℝ := 1.875 * rent_last_year)

-- Theorem to be proven
theorem elaine_earnings_increase (h : rent_this_year = multiplied_rent_last_year) : P = 25 :=
by
  sorry

end elaine_earnings_increase_l294_29438


namespace probability_white_black_l294_29414

variable (a b : ℕ)

theorem probability_white_black (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  (2 * a * b) / (a + b) / (a + b - 1) = (2 * (a * b) : ℝ) / ((a + b) * (a + b - 1): ℝ) :=
by sorry

end probability_white_black_l294_29414


namespace michael_truck_meetings_2_times_l294_29456

/-- Michael walks at a rate of 6 feet per second on a straight path. 
Trash pails are placed every 240 feet along the path. 
A garbage truck traveling at 12 feet per second in the same direction stops for 40 seconds at each pail. 
When Michael passes a pail, he sees the truck, which is 240 feet ahead, just leaving the next pail. 
Prove that Michael and the truck will meet exactly 2 times. -/

def michael_truck_meetings (v_michael v_truck d_pail t_stop init_michael init_truck : ℕ) : ℕ := sorry

theorem michael_truck_meetings_2_times :
  michael_truck_meetings 6 12 240 40 0 240 = 2 := 
  sorry

end michael_truck_meetings_2_times_l294_29456


namespace two_digit_sum_reverse_l294_29417

theorem two_digit_sum_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end two_digit_sum_reverse_l294_29417


namespace possible_degrees_of_remainder_l294_29403

theorem possible_degrees_of_remainder (p : Polynomial ℝ) (h : p = 3 * X^3 - 5 * X^2 + 2 * X - 8) :
  ∃ d : Finset ℕ, d = {0, 1, 2} :=
by
  sorry

end possible_degrees_of_remainder_l294_29403


namespace solve_equation_2021_2020_l294_29496

theorem solve_equation_2021_2020 (x : ℝ) (hx : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 :=
by {
  sorry
}

end solve_equation_2021_2020_l294_29496


namespace final_balance_is_103_5_percent_of_initial_l294_29484

/-- Define Megan's initial balance. -/
def initial_balance : ℝ := 125

/-- Define the balance after 25% increase from babysitting. -/
def after_babysitting (balance : ℝ) : ℝ :=
  balance + (balance * 0.25)

/-- Define the balance after 20% decrease from buying shoes. -/
def after_shoes (balance : ℝ) : ℝ :=
  balance - (balance * 0.20)

/-- Define the balance after 15% increase by investing in stocks. -/
def after_stocks (balance : ℝ) : ℝ :=
  balance + (balance * 0.15)

/-- Define the balance after 10% decrease due to medical expenses. -/
def after_medical_expense (balance : ℝ) : ℝ :=
  balance - (balance * 0.10)

/-- Define the final balance. -/
def final_balance : ℝ :=
  let b1 := after_babysitting initial_balance
  let b2 := after_shoes b1
  let b3 := after_stocks b2
  after_medical_expense b3

/-- Prove that the final balance is 103.5% of the initial balance. -/
theorem final_balance_is_103_5_percent_of_initial :
  final_balance / initial_balance = 1.035 :=
by
  unfold final_balance
  unfold initial_balance
  unfold after_babysitting
  unfold after_shoes
  unfold after_stocks
  unfold after_medical_expense
  sorry

end final_balance_is_103_5_percent_of_initial_l294_29484


namespace solution_l294_29480

noncomputable def problem_statement (a b : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)

theorem solution (a b : ℝ) :
  problem_statement a b ↔ (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ a' b' : ℤ, (a : ℝ) = a' ∧ (b : ℝ) = b')) :=
by
  sorry

end solution_l294_29480


namespace cost_of_ground_school_l294_29473

theorem cost_of_ground_school (G : ℝ) (F : ℝ) (h1 : F = G + 625) (h2 : F = 950) :
  G = 325 :=
by
  sorry

end cost_of_ground_school_l294_29473


namespace find_number_l294_29474

theorem find_number (x : ℝ) (h : x * 9999 = 824777405) : x = 82482.5 :=
by
  sorry

end find_number_l294_29474


namespace root_implies_quadratic_eq_l294_29472

theorem root_implies_quadratic_eq (m : ℝ) (h : (m + 2) - 2 + m^2 - 2 * m - 6 = 0) : 
  2 * m^2 - m - 6 = 0 :=
sorry

end root_implies_quadratic_eq_l294_29472


namespace incorrect_statement_l294_29458

def population : ℕ := 13000
def sample_size : ℕ := 500
def academic_performance (n : ℕ) : Type := sorry

def statement_A (ap : Type) : Prop := 
  ap = academic_performance population

def statement_B (ap : Type) : Prop := 
  ∀ (u : ℕ), u ≤ population → ap = academic_performance 1

def statement_C (ap : Type) : Prop := 
  ap = academic_performance sample_size

def statement_D : Prop := 
  sample_size = 500

theorem incorrect_statement : ¬ (statement_B (academic_performance 1)) :=
sorry

end incorrect_statement_l294_29458


namespace max_value_of_f_l294_29435

noncomputable def f (x a b : ℝ) := (1 - x ^ 2) * (x ^ 2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (x : ℝ) 
  (h1 : (∀ x, f x 8 15 = (1 - x^2) * (x^2 + 8*x + 15)))
  (h2 : ∀ x, f x a b = f (-(x + 4)) a b) :
  ∃ m, (∀ x, f x 8 15 ≤ m) ∧ m = 16 :=
by
  sorry

end max_value_of_f_l294_29435


namespace cows_count_l294_29455

theorem cows_count (D C : ℕ) (h_legs : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 := 
  sorry

end cows_count_l294_29455


namespace original_number_l294_29470

theorem original_number (h : 2.04 / 1.275 = 1.6) : 204 / 12.75 = 16 := 
by
  sorry

end original_number_l294_29470


namespace catherine_bottle_caps_l294_29466

-- Definitions from conditions
def friends : ℕ := 6
def caps_per_friend : ℕ := 3

-- Theorem statement from question and correct answer
theorem catherine_bottle_caps : friends * caps_per_friend = 18 :=
by sorry

end catherine_bottle_caps_l294_29466


namespace nextSimultaneousRingingTime_l294_29433

-- Define the intervals
def townHallInterval := 18
def universityTowerInterval := 24
def fireStationInterval := 30

-- Define the start time (in minutes from 00:00)
def startTime := 8 * 60 -- 8:00 AM

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Prove the next simultaneous ringing time
theorem nextSimultaneousRingingTime : 
  let lcmIntervals := lcm (lcm townHallInterval universityTowerInterval) fireStationInterval 
  startTime + lcmIntervals = 14 * 60 := -- 14:00 equals 2:00 PM in minutes
by
  -- You can replace the proof with the actual detailed proof.
  sorry

end nextSimultaneousRingingTime_l294_29433


namespace min_value_sin_cos_l294_29449

open Real

theorem min_value_sin_cos (x : ℝ) : 
  ∃ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 = 2 / 3 ∧ ∀ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 ≥ 2 / 3 :=
sorry

end min_value_sin_cos_l294_29449


namespace inequality_satisfaction_l294_29418

theorem inequality_satisfaction (k n : ℕ) (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1 + y^n / x^k) ≥ ((1 + y)^n / (1 + x)^k) ↔ 
    (k = 0) ∨ (n = 0) ∨ (0 = k ∧ 0 = n) ∨ (k ≥ n - 1 ∧ n ≥ 1) :=
by sorry

end inequality_satisfaction_l294_29418


namespace min_value_alpha_beta_gamma_l294_29467

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

def A (α β γ : ℕ) : ℕ := 2 ^ α * 3 ^ β * 5 ^ γ

def condition_1 (α β γ : ℕ) : Prop :=
  is_square (A α β γ / 2)

def condition_2 (α β γ : ℕ) : Prop :=
  is_cube (A α β γ / 3)

def condition_3 (α β γ : ℕ) : Prop :=
  is_fifth_power (A α β γ / 5)

theorem min_value_alpha_beta_gamma (α β γ : ℕ) :
  condition_1 α β γ → condition_2 α β γ → condition_3 α β γ →
  α + β + γ = 31 :=
sorry

end min_value_alpha_beta_gamma_l294_29467


namespace relations_of_sets_l294_29489

open Set

theorem relations_of_sets {A B : Set ℝ} (h : ∃ x ∈ A, x ∉ B) : 
  ¬(A ⊆ B) ∧ ((A ∩ B ≠ ∅) ∨ (B ⊆ A) ∨ (A ∩ B = ∅)) := sorry

end relations_of_sets_l294_29489


namespace closest_multiple_of_21_to_2023_l294_29491

theorem closest_multiple_of_21_to_2023 : ∃ k : ℤ, k * 21 = 2022 ∧ ∀ m : ℤ, m * 21 = 2023 → (abs (m - 2023)) > (abs (2022 - 2023)) :=
by
  sorry

end closest_multiple_of_21_to_2023_l294_29491


namespace quadratic_distinct_real_roots_l294_29454

open Real

theorem quadratic_distinct_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x ^ 2 - 2 * x - 1 = 0 ∧ k * y ^ 2 - 2 * y - 1 = 0) ↔ k > -1 ∧ k ≠ 0 :=
by
  sorry

end quadratic_distinct_real_roots_l294_29454


namespace expected_value_equals_1_5_l294_29485

noncomputable def expected_value_win (roll : ℕ) : ℚ :=
  if roll = 1 then -1
  else if roll = 4 then -4
  else if roll = 2 ∨ roll = 3 ∨ roll = 5 ∨ roll = 7 then roll
  else 0

noncomputable def expected_value_total : ℚ :=
  (1/8 : ℚ) * ((expected_value_win 1) + (expected_value_win 2) + (expected_value_win 3) +
               (expected_value_win 4) + (expected_value_win 5) + (expected_value_win 6) +
               (expected_value_win 7) + (expected_value_win 8))

theorem expected_value_equals_1_5 : expected_value_total = 1.5 := by
  sorry

end expected_value_equals_1_5_l294_29485


namespace inequality_must_be_true_l294_29406

theorem inequality_must_be_true (a b : ℝ) (h : a > b ∧ b > 0) :
  a + 1 / b > b + 1 / a :=
sorry

end inequality_must_be_true_l294_29406


namespace rational_comparison_correct_l294_29442

-- Definitions based on conditions 
def positive_gt_zero (a : ℚ) : Prop := 0 < a
def negative_lt_zero (a : ℚ) : Prop := a < 0
def positive_gt_negative (a b : ℚ) : Prop := positive_gt_zero a ∧ negative_lt_zero b ∧ a > b
def negative_comparison (a b : ℚ) : Prop := negative_lt_zero a ∧ negative_lt_zero b ∧ abs a > abs b ∧ a < b

-- Theorem to prove
theorem rational_comparison_correct :
  (0 < - (1 / 2)) = false ∧
  ((4 / 5) < - (6 / 7)) = false ∧
  ((9 / 8) > (8 / 9)) = true ∧
  (-4 > -3) = false :=
by
  -- Mark the proof as unfinished.
  sorry

end rational_comparison_correct_l294_29442


namespace compute_expression_l294_29410

variables (a b c : ℝ)

theorem compute_expression (h1 : a - b = 2) (h2 : a + c = 6) : 
  (2 * a + b + c) - 2 * (a - b - c) = 12 :=
by
  sorry

end compute_expression_l294_29410


namespace exercise_serial_matches_year_problem_serial_matches_year_l294_29495

-- Definitions for the exercise
def exercise_initial := 1169
def exercises_per_issue := 8
def issues_per_year := 9
def exercise_year := 1979
def exercises_per_year := exercises_per_issue * issues_per_year

-- Definitions for the problem
def problem_initial := 1576
def problems_per_issue := 8
def problems_per_year := problems_per_issue * issues_per_year
def problem_year := 1973

theorem exercise_serial_matches_year :
  ∃ (issue_number : ℕ) (exercise_number : ℕ),
    (issue_number = 3) ∧
    (exercise_number = 2) ∧
    (exercise_initial + 11 * exercises_per_year + 16 = exercise_year) :=
by {
  sorry
}

theorem problem_serial_matches_year :
  ∃ (issue_number : ℕ) (problem_number : ℕ),
    (issue_number = 5) ∧
    (problem_number = 5) ∧
    (problem_initial + 5 * problems_per_year + 36 = problem_year) :=
by {
  sorry
}

end exercise_serial_matches_year_problem_serial_matches_year_l294_29495


namespace num_three_digit_integers_divisible_by_12_l294_29401

theorem num_three_digit_integers_divisible_by_12 : 
  (∃ (count : ℕ), count = 3 ∧ 
    (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → 
      (∀ d : ℕ, d ∈ [n / 100, (n / 10) % 10, n % 10] → 4 < d) ∧ 
      n % 12 = 0 → 
      count = count + 1)) := 
sorry

end num_three_digit_integers_divisible_by_12_l294_29401


namespace rate_of_current_l294_29439

theorem rate_of_current (c : ℝ) (h1 : 7.5 = (20 + c) * 0.3) : c = 5 :=
by
  sorry

end rate_of_current_l294_29439


namespace point_on_xaxis_equidistant_l294_29488

theorem point_on_xaxis_equidistant :
  ∃ (A : ℝ × ℝ), A.2 = 0 ∧ 
                  dist A (-3, 2) = dist A (4, -5) ∧ 
                  A = (2, 0) :=
by
  sorry

end point_on_xaxis_equidistant_l294_29488


namespace num_girls_l294_29464

-- Define conditions as constants
def ratio (B G : ℕ) : Prop := B = (5 * G) / 8
def total (B G : ℕ) : Prop := B + G = 260

-- State the proof problem
theorem num_girls (B G : ℕ) (h1 : ratio B G) (h2 : total B G) : G = 160 :=
by {
  -- actual proof omitted
  sorry
}

end num_girls_l294_29464


namespace find_x_l294_29434

theorem find_x (x : ℤ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
by
  sorry

end find_x_l294_29434


namespace smallest_possible_value_l294_29432

-- Definitions and conditions provided
def x_plus_4_y_minus_4_eq_zero (x y : ℝ) : Prop := (x + 4) * (y - 4) = 0

-- Main theorem to state
theorem smallest_possible_value (x y : ℝ) (h : x_plus_4_y_minus_4_eq_zero x y) : x^2 + y^2 = 32 :=
sorry

end smallest_possible_value_l294_29432


namespace problem1_problem2_l294_29412

def f (x a : ℝ) := x^2 + 2 * a * x + 2

theorem problem1 (a : ℝ) (h : a = -1) : 
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a ≤ 37) ∧ (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x a = 37) ∧
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a ≥ 1) ∧ (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x a = 1) :=
by
  sorry

theorem problem2 (a : ℝ) : 
  (∀ x1 x2 : ℝ, -5 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 5 → f x1 a > f x2 a) ↔ a ≤ -5 :=
by
  sorry

end problem1_problem2_l294_29412


namespace solve_for_x_l294_29443

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 9) * x = 14) : x = 220.5 :=
by
  sorry

end solve_for_x_l294_29443


namespace optimal_cookies_l294_29487

-- Define the initial state and the game's rules
def initial_blackboard : List Int := List.replicate 2020 1

def erase_two (l : List Int) (x y : Int) : List Int :=
  l.erase x |>.erase y

def write_back (l : List Int) (n : Int) : List Int :=
  n :: l

-- Define termination conditions
def game_ends_condition1 (l : List Int) : Prop :=
  ∃ x ∈ l, x > l.sum - x

def game_ends_condition2 (l : List Int) : Prop :=
  l = List.replicate (l.length) 0

def game_ends (l : List Int) : Prop :=
  game_ends_condition1 l ∨ game_ends_condition2 l

-- Define the number of cookies given to Player A
def cookies (l : List Int) : Int :=
  l.length

-- Prove that if both players play optimally, Player A receives 7 cookies
theorem optimal_cookies : cookies (initial_blackboard) = 7 :=
  sorry

end optimal_cookies_l294_29487


namespace kids_still_awake_l294_29497

theorem kids_still_awake (initial_count remaining_after_first remaining_after_second : ℕ) 
  (h_initial : initial_count = 20)
  (h_first_round : remaining_after_first = initial_count / 2)
  (h_second_round : remaining_after_second = remaining_after_first / 2) : 
  remaining_after_second = 5 := 
by
  sorry

end kids_still_awake_l294_29497


namespace equal_sum_sequence_even_odd_l294_29486

-- Define the sequence a_n
variable {a : ℕ → ℤ}

-- Define the condition of the equal-sum sequence
def equal_sum_sequence (a : ℕ → ℤ) : Prop := ∀ n, a n + a (n + 1) = a (n + 1) + a (n + 2)

-- Statement to prove the odd terms are equal and the even terms are equal
theorem equal_sum_sequence_even_odd (a : ℕ → ℤ) (h : equal_sum_sequence a) : (∀ n, a (2 * n) = a 0) ∧ (∀ n, a (2 * n + 1) = a 1) :=
by
  sorry

end equal_sum_sequence_even_odd_l294_29486


namespace percentage_students_receive_valentine_l294_29440

/-- Given the conditions:
  1. There are 30 students.
  2. Mo wants to give a Valentine to some percentage of them.
  3. Each Valentine costs $2.
  4. Mo has $40.
  5. Mo will spend 90% of his money on Valentines.
Prove that the percentage of students receiving a Valentine is 60%.
-/
theorem percentage_students_receive_valentine :
  let total_students := 30
  let valentine_cost := 2
  let total_money := 40
  let spent_percentage := 0.90
  ∃ (cards : ℕ), 
    let money_spent := total_money * spent_percentage
    let cards_bought := money_spent / valentine_cost
    let percentage_students := (cards_bought / total_students) * 100
    percentage_students = 60 := 
by
  sorry

end percentage_students_receive_valentine_l294_29440


namespace find_S17_l294_29402

-- Definitions based on the conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variables (a1 : ℝ) (d : ℝ)

-- Conditions from the problem restated in Lean
axiom arithmetic_sequence : ∀ n, a n = a1 + (n - 1) * d
axiom sum_of_n_terms : ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)
axiom arithmetic_subseq : 2 * a 7 = a 5 + 3

-- Theorem to prove
theorem find_S17 : S 17 = 51 :=
by sorry

end find_S17_l294_29402


namespace total_spent_on_burgers_l294_29499

def days_in_june := 30
def burgers_per_day := 4
def cost_per_burger := 13

theorem total_spent_on_burgers (total_spent : Nat) :
  total_spent = days_in_june * burgers_per_day * cost_per_burger :=
sorry

end total_spent_on_burgers_l294_29499


namespace problem_statement_l294_29481

theorem problem_statement (a b c : ℝ) 
  (h1 : a - 2 * b + c = 0) 
  (h2 : a + 2 * b + c < 0) : b < 0 ∧ b^2 - a * c ≥ 0 :=
by
  sorry

end problem_statement_l294_29481


namespace no_values_of_expression_l294_29476

theorem no_values_of_expression (x : ℝ) (h : x^2 - 4 * x + 4 < 0) :
  ¬ ∃ y, y = x^2 + 4 * x + 5 :=
by
  sorry

end no_values_of_expression_l294_29476


namespace simplify_expression_l294_29457

theorem simplify_expression : (90 / 150) * (35 / 21) = 1 :=
by
  -- Insert proof here 
  sorry

end simplify_expression_l294_29457


namespace M_inter_N_is_1_2_l294_29452

-- Definitions based on given conditions
def M : Set ℝ := { y | ∃ x : ℝ, x > 0 ∧ y = 2^x }
def N : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Prove intersection of M and N is (1, 2]
theorem M_inter_N_is_1_2 :
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end M_inter_N_is_1_2_l294_29452


namespace possible_m_value_l294_29421

variable (a b m t : ℝ)
variable (h_a : a ≠ 0)
variable (h1 : ∃ t, ∀ x, ax^2 - bx ≥ -1 ↔ (x ≤ t - 1 ∨ x ≥ -3 - t))
variable (h2 : a * m^2 - b * m = 2)

theorem possible_m_value : m = 1 :=
sorry

end possible_m_value_l294_29421


namespace baker_remaining_cakes_l294_29436

def initial_cakes : ℝ := 167.3
def sold_cakes : ℝ := 108.2
def remaining_cakes : ℝ := initial_cakes - sold_cakes

theorem baker_remaining_cakes : remaining_cakes = 59.1 := by
  sorry

end baker_remaining_cakes_l294_29436


namespace balls_left_l294_29420

-- Define the conditions
def initial_balls : ℕ := 10
def removed_balls : ℕ := 3

-- The main statement to prove
theorem balls_left : initial_balls - removed_balls = 7 := by sorry

end balls_left_l294_29420


namespace sphere_triangle_distance_l294_29419

theorem sphere_triangle_distance
  (P X Y Z : Type)
  (radius : ℝ)
  (h1 : radius = 15)
  (dist_XY : ℝ)
  (h2 : dist_XY = 6)
  (dist_YZ : ℝ)
  (h3 : dist_YZ = 8)
  (dist_ZX : ℝ)
  (h4 : dist_ZX = 10)
  (distance_from_P_to_triangle : ℝ)
  (h5 : distance_from_P_to_triangle = 10 * Real.sqrt 2) :
  let a := 10
  let b := 2
  let c := 1
  let result := a + b + c
  result = 13 :=
by
  sorry

end sphere_triangle_distance_l294_29419


namespace cube_volume_in_cubic_yards_l294_29405

def volume_in_cubic_feet := 64
def cubic_feet_per_cubic_yard := 27

theorem cube_volume_in_cubic_yards : 
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 64 / 27 :=
by
  sorry

end cube_volume_in_cubic_yards_l294_29405


namespace problem_l294_29416

noncomputable def trajectory_C (x y : ℝ) : Prop :=
  y^2 = -8 * x

theorem problem (P : ℝ × ℝ) (k : ℝ) (h : -1 < k ∧ k < 0) 
  (H1 : P.1 = -2 ∨ P.1 = 2)
  (H2 : trajectory_C P.1 P.2) :
  ∃ Q : ℝ × ℝ, Q.1 < -6 :=
  sorry

end problem_l294_29416


namespace find_k_l294_29425

-- Given conditions
def p (x : ℝ) : ℝ := 2 * x + 3
def q (k : ℝ) (x : ℝ) : ℝ := k * x + k
def intersection (x y : ℝ) : Prop := y = p x ∧ ∃ k, y = q k x

-- Proof that based on the intersection at (1, 5), k evaluates to 5/2
theorem find_k : ∃ k : ℝ, intersection 1 5 → k = 5 / 2 := by
  sorry

end find_k_l294_29425


namespace people_dislike_both_radio_and_music_l294_29400

theorem people_dislike_both_radio_and_music :
  let total_people := 1500
  let dislike_radio_percent := 0.35
  let dislike_both_percent := 0.20
  let dislike_radio := dislike_radio_percent * total_people
  let dislike_both := dislike_both_percent * dislike_radio
  dislike_both = 105 :=
by
  sorry

end people_dislike_both_radio_and_music_l294_29400


namespace area_of_inscribed_rectangle_l294_29430

open Real

theorem area_of_inscribed_rectangle (r l w : ℝ) (h_radius : r = 7) 
  (h_ratio : l / w = 3) (h_width : w = 2 * r) : l * w = 588 :=
by
  sorry

end area_of_inscribed_rectangle_l294_29430


namespace min_value_l294_29450

theorem min_value (a b c : ℤ) (h : a > b ∧ b > c) :
  ∃ x, x = (a + b + c) / (a - b - c) ∧ 
       x + (a - b - c) / (a + b + c) = 2 := sorry

end min_value_l294_29450


namespace hyperbola_asymptote_value_of_a_l294_29462

-- Define the hyperbola and the conditions given
variables {a : ℝ} (h1 : a > 0) (h2 : ∀ x y : ℝ, 3 * x + 2 * y = 0 ∧ 3 * x - 2 * y = 0)

theorem hyperbola_asymptote_value_of_a :
  a = 2 := by
  sorry

end hyperbola_asymptote_value_of_a_l294_29462


namespace average_apples_per_guest_l294_29444

theorem average_apples_per_guest
  (servings_per_pie : ℕ)
  (pies : ℕ)
  (apples_per_serving : ℚ)
  (total_guests : ℕ)
  (red_delicious_proportion : ℚ)
  (granny_smith_proportion : ℚ)
  (total_servings := pies * servings_per_pie)
  (total_apples := total_servings * apples_per_serving)
  (total_red_delicious := (red_delicious_proportion / (red_delicious_proportion + granny_smith_proportion)) * total_apples)
  (total_granny_smith := (granny_smith_proportion / (red_delicious_proportion + granny_smith_proportion)) * total_apples)
  (average_apples_per_guest := total_apples / total_guests) :
  servings_per_pie = 8 →
  pies = 3 →
  apples_per_serving = 1.5 →
  total_guests = 12 →
  red_delicious_proportion = 2 →
  granny_smith_proportion = 1 →
  average_apples_per_guest = 3 :=
by
  intros;
  sorry

end average_apples_per_guest_l294_29444


namespace A_union_B_subset_B_A_intersection_B_subset_B_l294_29413

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3 * x - 10 <= 0}
def B (m : ℝ) : Set ℝ := {x | m - 4 <= x ∧ x <= 3 * m + 2}

-- Problem 1: Prove the range of m if A ∪ B = B
theorem A_union_B_subset_B (m : ℝ) : (A ∪ B m = B m) → (1 ≤ m ∧ m ≤ 2) :=
by
  sorry

-- Problem 2: Prove the range of m if A ∩ B = B
theorem A_intersection_B_subset_B (m : ℝ) : (A ∩ B m = B m) → (m < -3) :=
by
  sorry

end A_union_B_subset_B_A_intersection_B_subset_B_l294_29413


namespace train_length_approx_90_l294_29459

noncomputable def speed_in_m_per_s := (124 : ℝ) * (1000 / 3600)

noncomputable def time_in_s := (2.61269421026963 : ℝ)

noncomputable def length_of_train := speed_in_m_per_s * time_in_s

theorem train_length_approx_90 : abs (length_of_train - 90) < 1e-9 :=
  by
  sorry

end train_length_approx_90_l294_29459


namespace min_c_for_expression_not_min_abs_c_for_expression_l294_29492

theorem min_c_for_expression :
  ∀ c : ℝ,
  (c - 3)^2 + (c - 4)^2 + (c - 8)^2 ≥ (5 - 3)^2 + (5 - 4)^2 + (5 - 8)^2 := 
by sorry

theorem not_min_abs_c_for_expression :
  ∃ c : ℝ, |c - 3| + |c - 4| + |c - 8| < |5 - 3| + |5 - 4| + |5 - 8| := 
by sorry

end min_c_for_expression_not_min_abs_c_for_expression_l294_29492


namespace B_greater_than_A_l294_29494

def A := (54 : ℚ) / (5^7 * 11^4 : ℚ)
def B := (55 : ℚ) / (5^7 * 11^4 : ℚ)

theorem B_greater_than_A : B > A := by
  sorry

end B_greater_than_A_l294_29494


namespace geometric_sequence_term_6_l294_29437

-- Define the geometric sequence conditions
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

variables 
  (a : ℕ → ℝ) -- the geometric sequence
  (r : ℝ) -- common ratio, which is 2
  (h_r : r = 2)
  (h_pos : ∀ n, 0 < a n)
  (h_condition : a 4 * a 10 = 16)

-- The proof statement
theorem geometric_sequence_term_6 :
  a 6 = 2 :=
sorry

end geometric_sequence_term_6_l294_29437


namespace correct_operator_is_subtraction_l294_29478

theorem correct_operator_is_subtraction :
  (8 - 2) + 5 * (3 - 2) = 11 :=
by
  sorry

end correct_operator_is_subtraction_l294_29478


namespace ratatouille_cost_per_quart_l294_29446

theorem ratatouille_cost_per_quart:
  let eggplant_weight := 5.5
  let eggplant_price := 2.20
  let zucchini_weight := 3.8
  let zucchini_price := 1.85
  let tomatoes_weight := 4.6
  let tomatoes_price := 3.75
  let onions_weight := 2.7
  let onions_price := 1.10
  let basil_weight := 1.0
  let basil_price_per_quarter := 2.70
  let bell_peppers_weight := 0.75
  let bell_peppers_price := 3.15
  let yield_quarts := 4.5
  let eggplant_cost := eggplant_weight * eggplant_price
  let zucchini_cost := zucchini_weight * zucchini_price
  let tomatoes_cost := tomatoes_weight * tomatoes_price
  let onions_cost := onions_weight * onions_price
  let basil_cost := basil_weight * (basil_price_per_quarter * 4)
  let bell_peppers_cost := bell_peppers_weight * bell_peppers_price
  let total_cost := eggplant_cost + zucchini_cost + tomatoes_cost + onions_cost + basil_cost + bell_peppers_cost
  let cost_per_quart := total_cost / yield_quarts
  cost_per_quart = 11.67 :=
by
  sorry

end ratatouille_cost_per_quart_l294_29446


namespace least_pawns_required_l294_29407

theorem least_pawns_required (n k : ℕ) (h1 : n > 0) (h2 : k > 0) (h3 : 2 * k > n) (h4 : 3 * k ≤ 2 * n) : 
  ∃ (m : ℕ), m = 4 * (n - k) :=
sorry

end least_pawns_required_l294_29407


namespace roots_of_equation_in_interval_l294_29469

theorem roots_of_equation_in_interval (f : ℝ → ℝ) (interval : Set ℝ) (n_roots : ℕ) :
  (∀ x ∈ interval, f x = 8 * x * (1 - 2 * x^2) * (8 * x^4 - 8 * x^2 + 1) - 1) →
  (interval = Set.Icc 0 1) →
  (n_roots = 4) :=
by
  intros f_eq interval_eq
  sorry

end roots_of_equation_in_interval_l294_29469


namespace subset_A_inter_B_eq_A_l294_29479

variable {x : ℝ}
def A (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2 * k}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem subset_A_inter_B_eq_A (k : ℝ) : (A k ∩ B = A k) ↔ (k ≤ 3 / 2) := 
sorry

end subset_A_inter_B_eq_A_l294_29479


namespace number_of_rows_seating_exactly_9_students_l294_29453

theorem number_of_rows_seating_exactly_9_students (x : ℕ) : 
  ∀ y z, x * 9 + y * 5 + z * 8 = 55 → x % 5 = 1 ∧ x % 8 = 7 → x = 3 :=
by sorry

end number_of_rows_seating_exactly_9_students_l294_29453


namespace chris_money_l294_29445

-- Define conditions
def grandmother_gift : Nat := 25
def aunt_uncle_gift : Nat := 20
def parents_gift : Nat := 75
def total_after_birthday : Nat := 279

-- Define the proof problem to show Chris had $159 before his birthday
theorem chris_money (x : Nat) (h : x + grandmother_gift + aunt_uncle_gift + parents_gift = total_after_birthday) :
  x = 159 :=
by
  -- Leave the proof blank
  sorry

end chris_money_l294_29445


namespace value_of_m_l294_29411

theorem value_of_m (m : ℝ) :
  (∀ A B : ℝ × ℝ, A = (m + 1, -2) → B = (3, m - 1) → (A.snd = B.snd) → m = -1) :=
by
  intros A B hA hB h_parallel
  -- Apply the given conditions and assumptions to prove the value of m.
  sorry

end value_of_m_l294_29411


namespace percentage_meetings_correct_l294_29428

def work_day_hours : ℕ := 10
def minutes_in_hour : ℕ := 60
def total_work_day_minutes := work_day_hours * minutes_in_hour

def lunch_break_minutes : ℕ := 30
def effective_work_day_minutes := total_work_day_minutes - lunch_break_minutes

def first_meeting_minutes : ℕ := 60
def second_meeting_minutes := 3 * first_meeting_minutes
def total_meeting_minutes := first_meeting_minutes + second_meeting_minutes

def percentage_of_day_spent_in_meetings := (total_meeting_minutes * 100) / effective_work_day_minutes

theorem percentage_meetings_correct : percentage_of_day_spent_in_meetings = 42 := 
by
  sorry

end percentage_meetings_correct_l294_29428


namespace stored_bales_correct_l294_29427

theorem stored_bales_correct :
  let initial_bales := 28
  let new_bales := 54
  let stored_bales := new_bales - initial_bales
  stored_bales = 26 :=
by
  let initial_bales := 28
  let new_bales := 54
  let stored_bales := new_bales - initial_bales
  show stored_bales = 26
  sorry

end stored_bales_correct_l294_29427


namespace closest_ratio_of_adults_to_children_l294_29498

def total_fees (a c : ℕ) : ℕ := 20 * a + 10 * c
def adults_children_equation (a c : ℕ) : Prop := 2 * a + c = 160

theorem closest_ratio_of_adults_to_children :
  ∃ a c : ℕ, 
    total_fees a c = 1600 ∧
    a ≥ 1 ∧ c ≥ 1 ∧
    adults_children_equation a c ∧
    (∀ a' c' : ℕ, total_fees a' c' = 1600 ∧ 
        a' ≥ 1 ∧ c' ≥ 1 ∧ 
        adults_children_equation a' c' → 
        abs ((a : ℝ) / c - 1) ≤ abs ((a' : ℝ) / c' - 1)) :=
  sorry

end closest_ratio_of_adults_to_children_l294_29498


namespace calculate_expression_l294_29404

-- Definitions based on the conditions
def opposite (a b : ℤ) : Prop := a + b = 0
def reciprocal (c d : ℝ) : Prop := c * d = 1
def negative_abs_two (m : ℝ) : Prop := m = -2

-- The main statement to be proved
theorem calculate_expression (a b : ℤ) (c d m : ℝ) 
  (h1 : opposite a b) 
  (h2 : reciprocal c d) 
  (h3 : negative_abs_two m) : 
  m + c * d + a + b + (c * d) ^ 2010 = 0 := 
by
  sorry

end calculate_expression_l294_29404


namespace algebraic_expression_value_l294_29408

noncomputable def a : ℝ := 1 + Real.sqrt 2
noncomputable def b : ℝ := 1 - Real.sqrt 2

theorem algebraic_expression_value :
  let a := 1 + Real.sqrt 2
  let b := 1 - Real.sqrt 2
  a^2 - a * b + b^2 = 7 := by
  sorry

end algebraic_expression_value_l294_29408


namespace at_least_three_double_marked_l294_29448

noncomputable def grid := Matrix (Fin 10) (Fin 20) ℕ -- 10x20 matrix with natural numbers

def is_red_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  ∃ (k₁ k₂ : Fin 20), k₁ ≠ k₂ ∧ (g i k₁) ≤ g i j ∧ (g i k₂) ≤ g i j ∧ ∀ (k : Fin 20), (k ≠ k₁ ∧ k ≠ k₂) → g i k ≤ g i j

def is_blue_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  ∃ (k₁ k₂ : Fin 10), k₁ ≠ k₂ ∧ (g k₁ j) ≤ g i j ∧ (g k₂ j) ≤ g i j ∧ ∀ (k : Fin 10), (k ≠ k₁ ∧ k ≠ k₂) → g k j ≤ g i j

def is_double_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  is_red_marked g i j ∧ is_blue_marked g i j

theorem at_least_three_double_marked (g : grid) :
  (∃ (i₁ i₂ i₃ : Fin 10) (j₁ j₂ j₃ : Fin 20), i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₃ ≠ i₁ ∧ 
    j₁ ≠ j₂ ∧ j₂ ≠ j₃ ∧ j₃ ≠ j₁ ∧ is_double_marked g i₁ j₁ ∧ is_double_marked g i₂ j₂ ∧ is_double_marked g i₃ j₃) :=
sorry

end at_least_three_double_marked_l294_29448
