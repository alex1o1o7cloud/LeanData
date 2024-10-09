import Mathlib

namespace weight_of_B_l2226_222686

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 47) : B = 39 :=
by
  sorry

end weight_of_B_l2226_222686


namespace tangent_line_at_pi_over_4_l2226_222668

noncomputable def tangent_eq (x y : ℝ) : Prop :=
  y = 2 * x * Real.tan x

noncomputable def tangent_line_eq (x y : ℝ) : Prop :=
  (2 + Real.pi) * x - y - (Real.pi^2 / 4) = 0

theorem tangent_line_at_pi_over_4 :
  tangent_eq (Real.pi / 4) (Real.pi / 2) →
  tangent_line_eq (Real.pi / 4) (Real.pi / 2) :=
by
  sorry

end tangent_line_at_pi_over_4_l2226_222668


namespace tom_blue_marbles_l2226_222606

-- Definitions based on conditions
def jason_blue_marbles : Nat := 44
def total_blue_marbles : Nat := 68

-- The problem statement to prove
theorem tom_blue_marbles : (total_blue_marbles - jason_blue_marbles) = 24 :=
by
  sorry

end tom_blue_marbles_l2226_222606


namespace problem_statement_l2226_222648

def y_and (y : ℤ) : ℤ := 9 - y
def and_y (y : ℤ) : ℤ := y - 9

theorem problem_statement : and_y (y_and 15) = -15 := 
by
  sorry

end problem_statement_l2226_222648


namespace circle_regions_division_l2226_222689

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l2226_222689


namespace find_x_l2226_222634

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Definition of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- Theorem statement
theorem find_x (x : ℝ) (h_parallel : parallel a (b x)) : x = 6 :=
sorry

end find_x_l2226_222634


namespace one_over_nine_inv_half_eq_three_l2226_222693

theorem one_over_nine_inv_half_eq_three : (1 / 9 : ℝ) ^ (-1 / 2 : ℝ) = 3 := 
by
  sorry

end one_over_nine_inv_half_eq_three_l2226_222693


namespace solution_l2226_222685

-- Definitions
def equation1 (x y z : ℝ) : Prop := 2 * x + y + z = 17
def equation2 (x y z : ℝ) : Prop := x + 2 * y + z = 14
def equation3 (x y z : ℝ) : Prop := x + y + 2 * z = 13

-- Theorem to prove
theorem solution (x y z : ℝ) (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) : x = 6 :=
by
  sorry

end solution_l2226_222685


namespace find_j_l2226_222671

theorem find_j (n j : ℕ) (h_n_pos : n > 0) (h_j_pos : j > 0) (h_rem : n % j = 28) (h_div : n / j = 142 ∧ (↑n / ↑j : ℝ) = 142.07) : j = 400 :=
by {
  sorry
}

end find_j_l2226_222671


namespace projected_increase_in_attendance_l2226_222679

variable (A P : ℝ)

theorem projected_increase_in_attendance :
  (0.8 * A = 0.64 * (A + (P / 100) * A)) → P = 25 :=
by
  intro h
  -- Proof omitted
  sorry

end projected_increase_in_attendance_l2226_222679


namespace triangle_inequality_l2226_222674

theorem triangle_inequality 
  (a b c R : ℝ) 
  (h1 : a + b > c) 
  (h2 : a + c > b) 
  (h3 : b + c > a) 
  (hR : R = (a * b * c) / (4 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)))) : 
  a^2 + b^2 + c^2 ≤ 9 * R^2 :=
by 
  sorry

end triangle_inequality_l2226_222674


namespace average_of_multiples_l2226_222639

theorem average_of_multiples :
  let sum_of_first_7_multiples_of_9 := 9 + 18 + 27 + 36 + 45 + 54 + 63
  let sum_of_first_5_multiples_of_11 := 11 + 22 + 33 + 44 + 55
  let sum_of_first_3_negative_multiples_of_13 := -13 + -26 + -39
  let total_sum := sum_of_first_7_multiples_of_9 + sum_of_first_5_multiples_of_11 + sum_of_first_3_negative_multiples_of_13
  let average := total_sum / 3
  average = 113 :=
by
  sorry

end average_of_multiples_l2226_222639


namespace word_identification_l2226_222641

theorem word_identification (word : String) :
  ( ( (word = "бал" ∨ word = "баллы")
    ∧ (∃ sport : String, sport = "figure skating" ∨ sport = "rhythmic gymnastics"))
    ∧ (∃ year : Nat, year = 2015 ∧ word = "пенсионные баллы") ) → 
  word = "баллы" :=
by
  sorry

end word_identification_l2226_222641


namespace number_of_arrangements_l2226_222651

theorem number_of_arrangements (teams : Finset ℕ) (sites : Finset ℕ) :
  (∀ team, team ∈ teams → (team ∈ sites)) ∧ ((Finset.card sites = 3) ∧ (Finset.card teams = 6)) ∧ 
  (∃ (a b c : ℕ), a + b + c = 6 ∧ a >= 2 ∧ b >= 1 ∧ c >= 1) →
  ∃ (n : ℕ), n = 360 :=
sorry

end number_of_arrangements_l2226_222651


namespace inequality_not_always_true_l2226_222684

variables {a b c d : ℝ}

theorem inequality_not_always_true
  (h1 : a > b) (h2 : b > 0) (h3 : c > 0) (h4 : d ≠ 0) :
  ¬ ∀ (a b d : ℝ), (a > b) → (d ≠ 0) → (a + d)^2 > (b + d)^2 :=
by
  intro H
  specialize H a b d h1 h4
  sorry

end inequality_not_always_true_l2226_222684


namespace find_other_number_l2226_222677

theorem find_other_number (x y : ℕ) (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) (h3 : x = 7) : y = 3 :=
by
  sorry

end find_other_number_l2226_222677


namespace right_triangle_exists_l2226_222614

-- Define the setup: equilateral triangle ABC, point P, and angle condition
def Point (α : Type*) := α 
def inside {α : Type*} (p : Point α) (A B C : Point α) : Prop := sorry
def angle_at {α : Type*} (p q r : Point α) (θ : ℝ) : Prop := sorry
noncomputable def PA {α : Type*} (P A : Point α) : ℝ := sorry
noncomputable def PB {α : Type*} (P B : Point α) : ℝ := sorry
noncomputable def PC {α : Type*} (P C : Point α) : ℝ := sorry

-- Theorem we need to prove
theorem right_triangle_exists {α : Type*} 
  (A B C P : Point α)
  (h1 : inside P A B C)
  (h2 : angle_at P A B 150) :
  ∃ (Q : Point α), angle_at P Q B 90 :=
sorry

end right_triangle_exists_l2226_222614


namespace workers_in_workshop_l2226_222653

theorem workers_in_workshop (W : ℕ) (h1 : W ≤ 100) (h2 : W % 3 = 0) (h3 : W % 25 = 0)
  : W = 75 ∧ W / 3 = 25 ∧ W * 8 / 100 = 6 :=
by
  sorry

end workers_in_workshop_l2226_222653


namespace number_of_ordered_pairs_l2226_222683

theorem number_of_ordered_pairs :
  ∃ (n : ℕ), n = 99 ∧
  (∀ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ (Int.gcd a b) * a + b^2 = 10000
  → ∃ (k : ℕ), k = 99) :=
sorry

end number_of_ordered_pairs_l2226_222683


namespace unique_positive_integers_sum_l2226_222691

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 77) / 3 + 5 / 3)

theorem unique_positive_integers_sum :
  ∃ (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c),
    x^100 = 3 * x^98 + 17 * x^96 + 13 * x^94 - 2 * x^50 + (a : ℝ) * x^46 + (b : ℝ) * x^44 + (c : ℝ) * x^40
    ∧ a + b + c = 167 := by
  sorry

end unique_positive_integers_sum_l2226_222691


namespace Joey_study_time_l2226_222635

theorem Joey_study_time :
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96 := by
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  show (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96
  -- define study times
  let weekday_hours_per_week := weekday_hours_per_night * nights_per_week
  let weekend_hours_per_week := weekend_hours_per_day * days_per_weekend
  -- sum times per week
  let total_hours_per_week := weekday_hours_per_week + weekend_hours_per_week
  -- multiply by weeks until exam
  let total_study_time := total_hours_per_week * weeks_until_exam
  have h : total_study_time = 96 := by sorry
  exact h

end Joey_study_time_l2226_222635


namespace problem_statement_l2226_222613
noncomputable def f (M : ℝ) (x : ℝ) : ℝ := M * Real.sin (2 * x + Real.pi / 6)
def is_symmetric (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def is_center_of_symmetry (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop := ∀ x, f (2 * c.1 - x) = 2 * c.2 - f x

theorem problem_statement (M : ℝ) (hM : M ≠ 0) : 
    is_symmetric (f M) (2 * Real.pi / 3) ∧ 
    is_periodic (f M) Real.pi ∧ 
    is_center_of_symmetry (f M) (5 * Real.pi / 12, 0) :=
by
  sorry

end problem_statement_l2226_222613


namespace f_value_at_3_l2226_222643

theorem f_value_at_3 (a b : ℝ) (h : (a * (-3)^3 - b * (-3) + 2 = -1)) : a * (3)^3 - b * 3 + 2 = 5 :=
sorry

end f_value_at_3_l2226_222643


namespace solve_for_y_l2226_222698

theorem solve_for_y (y : ℕ) (h1 : 40 = 2^3 * 5) (h2 : 8 = 2^3) :
  40^3 = 8^y ↔ y = 3 :=
by sorry

end solve_for_y_l2226_222698


namespace g_four_times_of_three_l2226_222654

noncomputable def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 4 * x - 1

theorem g_four_times_of_three :
  g (g (g (g 3))) = 3 := by
  sorry

end g_four_times_of_three_l2226_222654


namespace find_income_l2226_222665

noncomputable def income_expenditure_proof : Prop := 
  ∃ (x : ℕ), (5 * x - 4 * x = 3600) ∧ (5 * x = 18000)

theorem find_income : income_expenditure_proof :=
  sorry

end find_income_l2226_222665


namespace increasing_function_a_range_l2226_222678

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4 * a * x else (2 * a + 3) * x - 4 * a + 5

theorem increasing_function_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end increasing_function_a_range_l2226_222678


namespace arith_seq_ratio_l2226_222694

-- Definitions related to arithmetic sequence and sum
def arithmetic_seq (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arith_seq (S a : ℕ → ℝ) := ∀ n : ℕ, S n = (n : ℝ) / 2 * (a 1 + a n)

-- Given condition
def condition (a : ℕ → ℝ) := a 8 / a 7 = 13 / 5

-- Prove statement
theorem arith_seq_ratio (a S : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_sum : sum_of_arith_seq S a)
  (h_cond : condition a) :
  S 15 / S 13 = 3 := 
sorry

end arith_seq_ratio_l2226_222694


namespace slope_of_line_l2226_222637

theorem slope_of_line (x y : ℝ) (h : x / 4 + y / 3 = 1) : ∀ m : ℝ, (y = m * x + 3) → m = -3/4 :=
by
  sorry

end slope_of_line_l2226_222637


namespace stratified_sampling_l2226_222681

theorem stratified_sampling (n : ℕ) : 100 + 600 + 500 = 1200 → 500 ≠ 0 → 40 / 500 = n / 1200 → n = 96 :=
by
  intros total_population nonzero_div divisor_eq
  sorry

end stratified_sampling_l2226_222681


namespace minimum_points_to_guarantee_highest_score_l2226_222666

theorem minimum_points_to_guarantee_highest_score :
  ∃ (score1 score2 score3 : ℕ), 
   (score1 = 7 ∨ score1 = 4 ∨ score1 = 2) ∧ (score2 = 7 ∨ score2 = 4 ∨ score2 = 2) ∧
   (score3 = 7 ∨ score3 = 4 ∨ score3 = 2) ∧ 
   (∀ (score4 : ℕ), 
     (score4 = 7 ∨ score4 = 4 ∨ score4 = 2) → 
     (score1 + score2 + score3 + score4 < 25)) → 
  score1 + score2 + score3 + 7 ≥ 25 :=
   sorry

end minimum_points_to_guarantee_highest_score_l2226_222666


namespace cos_alpha_plus_beta_l2226_222602

variable {α β : ℝ}
variable (sin_alpha : Real.sin α = 3/5)
variable (cos_beta : Real.cos β = 4/5)
variable (α_interval : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
variable (β_interval : β ∈ Set.Ioo 0 (Real.pi / 2))

theorem cos_alpha_plus_beta: Real.cos (α + β) = -1 :=
by
  sorry

end cos_alpha_plus_beta_l2226_222602


namespace sticker_ratio_l2226_222619

theorem sticker_ratio (gold : ℕ) (silver : ℕ) (bronze : ℕ)
  (students : ℕ) (stickers_per_student : ℕ)
  (h1 : gold = 50)
  (h2 : bronze = silver - 20)
  (h3 : students = 5)
  (h4 : stickers_per_student = 46)
  (h5 : gold + silver + bronze = students * stickers_per_student) :
  silver / gold = 2 / 1 :=
by
  sorry

end sticker_ratio_l2226_222619


namespace Carter_cards_l2226_222663

variable (C : ℕ) -- Let C be the number of baseball cards Carter has.

-- Condition 1: Marcus has 210 baseball cards.
def Marcus_cards : ℕ := 210

-- Condition 2: Marcus has 58 more cards than Carter.
def Marcus_has_more (C : ℕ) : Prop := Marcus_cards = C + 58

theorem Carter_cards (C : ℕ) (h : Marcus_has_more C) : C = 152 :=
by
  -- Expand the condition
  unfold Marcus_has_more at h
  -- Simplify the given equation
  rw [Marcus_cards] at h
  -- Solve for C
  linarith

end Carter_cards_l2226_222663


namespace largest_k_l2226_222699

def S : Set ℕ := {x | x > 0 ∧ x ≤ 100}

def satisfies_property (A B : Set ℕ) : Prop :=
  ∃ x ∈ A ∩ B, ∀ y ∈ A ∪ B, x ≠ y

theorem largest_k (k : ℕ) : 
  (∃ subsets : Finset (Set ℕ), 
    (subsets.card = k) ∧ 
    (∀ {A B : Set ℕ}, A ∈ subsets ∧ B ∈ subsets ∧ A ≠ B → 
      ¬(A ∩ B = ∅) ∧ satisfies_property A B)) →
  k ≤ 2^99 - 1 := sorry

end largest_k_l2226_222699


namespace compute_expression_l2226_222695

-- Definition of the expression
def expression := 5 + 4 * (4 - 9)^2

-- Statement of the theorem, asserting the expression equals 105
theorem compute_expression : expression = 105 := by
  sorry

end compute_expression_l2226_222695


namespace fillets_per_fish_l2226_222625

-- Definitions for the conditions
def fish_caught_per_day := 2
def days := 30
def total_fish_caught : Nat := fish_caught_per_day * days
def total_fish_fillets := 120

-- The proof problem statement
theorem fillets_per_fish (h1 : total_fish_caught = 60) (h2 : total_fish_fillets = 120) : 
  (total_fish_fillets / total_fish_caught) = 2 := sorry

end fillets_per_fish_l2226_222625


namespace two_pt_seven_five_as_fraction_l2226_222690

-- Define the decimal value 2.75
def decimal_value : ℚ := 11 / 4

-- Define the question
theorem two_pt_seven_five_as_fraction : 2.75 = decimal_value := by
  sorry

end two_pt_seven_five_as_fraction_l2226_222690


namespace exists_linear_eq_solution_x_2_l2226_222623

theorem exists_linear_eq_solution_x_2 : ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x : ℝ, a * x + b = 0 ↔ x = 2 :=
by
  sorry

end exists_linear_eq_solution_x_2_l2226_222623


namespace distinct_colorings_l2226_222603

def sections : ℕ := 6
def red_count : ℕ := 3
def blue_count : ℕ := 1
def green_count : ℕ := 1
def yellow_count : ℕ := 1

def permutations_without_rotation : ℕ := Nat.factorial sections / 
  (Nat.factorial red_count * Nat.factorial blue_count * Nat.factorial green_count * Nat.factorial yellow_count)

def rotational_symmetry : ℕ := permutations_without_rotation / sections

theorem distinct_colorings (rotational_symmetry) : rotational_symmetry = 20 :=
  sorry

end distinct_colorings_l2226_222603


namespace no_infinite_prime_sequence_l2226_222601

theorem no_infinite_prime_sequence (p : ℕ) (h_prime : Nat.Prime p) :
  ¬(∃ (p_seq : ℕ → ℕ), (∀ n, Nat.Prime (p_seq n)) ∧ (∀ n, p_seq (n + 1) = 2 * p_seq n + 1)) :=
by
  sorry

end no_infinite_prime_sequence_l2226_222601


namespace number_of_men_in_first_group_l2226_222658

-- Define the conditions as hypotheses in Lean
def work_completed_in_25_days (x : ℕ) : Prop := x * 25 * (1 : ℚ) / (25 * x) = (1 : ℚ)
def twenty_men_complete_in_15_days : Prop := 20 * 15 * (1 : ℚ) / 15 = (1 : ℚ)

-- Define the theorem to prove the number of men in the first group
theorem number_of_men_in_first_group (x : ℕ) (h1 : work_completed_in_25_days x)
  (h2 : twenty_men_complete_in_15_days) : x = 20 :=
  sorry

end number_of_men_in_first_group_l2226_222658


namespace b_plus_c_is_square_l2226_222633

-- Given the conditions:
variables (a b c : ℕ)
variable (h1 : a > 0 ∧ b > 0 ∧ c > 0)  -- Condition 1: Positive integers
variable (h2 : Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1)  -- Condition 2: Pairwise relatively prime
variable (h3 : a % 2 = 1 ∧ c % 2 = 1)  -- Condition 3: a and c are odd
variable (h4 : a^2 + b^2 = c^2)  -- Condition 4: Pythagorean triple equation

-- Prove that b + c is the square of an integer
theorem b_plus_c_is_square : ∃ k : ℕ, b + c = k^2 :=
by
  sorry

end b_plus_c_is_square_l2226_222633


namespace sum_of_pqrstu_l2226_222644

theorem sum_of_pqrstu (p q r s t : ℤ) (h : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -72) 
  (hpqrs : p ≠ q) (hnpr : p ≠ r) (hnps : p ≠ s) (hnpt : p ≠ t) (hnqr : q ≠ r) 
  (hnqs : q ≠ s) (hnqt : q ≠ t) (hnrs : r ≠ s) (hnrt : r ≠ t) (hnst : s ≠ t) : 
  p + q + r + s + t = 25 := 
by
  sorry

end sum_of_pqrstu_l2226_222644


namespace trapezoid_area_correct_l2226_222631

noncomputable def trapezoid_area (x : ℝ) : ℝ :=
  let base1 := 3 * x
  let base2 := 5 * x + 2
  (base1 + base2) / 2 * x

theorem trapezoid_area_correct (x : ℝ) : trapezoid_area x = 4 * x^2 + x :=
  by
  sorry

end trapezoid_area_correct_l2226_222631


namespace segment_shadow_ratio_l2226_222605

theorem segment_shadow_ratio (a b a' b' : ℝ) (h : a / b = a' / b') : a / a' = b / b' :=
sorry

end segment_shadow_ratio_l2226_222605


namespace simplify_expression_l2226_222610

theorem simplify_expression : (8 * (15 / 9) * (-45 / 40) = -(1 / 15)) :=
by
  sorry

end simplify_expression_l2226_222610


namespace range_of_a_l2226_222697

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ 1 → 1 + 2^x + 4^x * a > 0) ↔ a > -3/4 := 
sorry

end range_of_a_l2226_222697


namespace solve_for_y_in_terms_of_x_l2226_222630

theorem solve_for_y_in_terms_of_x (x y : ℝ) (h : x - 2 = y + 3 * x) : y = -2 * x - 2 :=
sorry

end solve_for_y_in_terms_of_x_l2226_222630


namespace reflection_coefficient_l2226_222659

theorem reflection_coefficient (I_0 : ℝ) (I_4 : ℝ) (k : ℝ) 
  (h1 : I_4 = I_0 * (1 - k)^4) 
  (h2 : I_4 = I_0 / 256) : 
  k = 0.75 :=
by 
  -- Proof omitted
  sorry

end reflection_coefficient_l2226_222659


namespace range_of_m_l2226_222632

noncomputable def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f m x < 0) ↔ -4 < m ∧ m ≤ 0 := by
  sorry

end range_of_m_l2226_222632


namespace graph_is_hyperbola_l2226_222612

theorem graph_is_hyperbola : ∀ x y : ℝ, (x + y) ^ 2 = x ^ 2 + y ^ 2 + 2 * x + 2 * y ↔ (x - 1) * (y - 1) = 1 := 
by {
  sorry
}

end graph_is_hyperbola_l2226_222612


namespace final_score_eq_l2226_222687

variable (initial_score : ℝ)
def deduction_lost_answer : ℝ := 1
def deduction_error : ℝ := 0.5
def deduction_checks : ℝ := 0

def total_deduction : ℝ := deduction_lost_answer + deduction_error + deduction_checks

theorem final_score_eq : final_score = initial_score - total_deduction := by
  sorry

end final_score_eq_l2226_222687


namespace integer_roots_of_polynomial_l2226_222627

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, x^3 - 4*x^2 - 11*x + 24 = 0 ↔ x = 2 ∨ x = -3 ∨ x = 4 := 
by 
  sorry

end integer_roots_of_polynomial_l2226_222627


namespace compute_expression_l2226_222615

theorem compute_expression : 75 * 1313 - 25 * 1313 = 65650 :=
by
  sorry

end compute_expression_l2226_222615


namespace orange_is_faster_by_l2226_222696

def forest_run_time (distance speed : ℕ) : ℕ := distance / speed
def beach_run_time (distance speed : ℕ) : ℕ := distance / speed
def mountain_run_time (distance speed : ℕ) : ℕ := distance / speed

def total_time_in_minutes (forest_distance forest_speed beach_distance beach_speed mountain_distance mountain_speed : ℕ) : ℕ :=
  (forest_run_time forest_distance forest_speed + beach_run_time beach_distance beach_speed + mountain_run_time mountain_distance mountain_speed) * 60

def apple_total_time := total_time_in_minutes 18 3 6 2 3 1
def mac_total_time := total_time_in_minutes 20 4 8 3 3 1
def orange_total_time := total_time_in_minutes 22 5 10 4 3 2

def combined_time := apple_total_time + mac_total_time
def orange_time_difference := combined_time - orange_total_time

theorem orange_is_faster_by :
  orange_time_difference = 856 := sorry

end orange_is_faster_by_l2226_222696


namespace basket_can_hold_40_fruits_l2226_222682

-- Let us define the number of oranges as 10
def oranges : ℕ := 10

-- There are 3 times as many apples as oranges
def apples : ℕ := 3 * oranges

-- The total number of fruits in the basket
def total_fruits : ℕ := oranges + apples

theorem basket_can_hold_40_fruits (h₁ : oranges = 10) (h₂ : apples = 3 * oranges) : total_fruits = 40 :=
by
  -- We assume the conditions and derive the conclusion
  sorry

end basket_can_hold_40_fruits_l2226_222682


namespace length_of_ST_l2226_222672

theorem length_of_ST (LM MN NL: ℝ) (LR : ℝ) (LT TR LS SR: ℝ) 
  (h1: LM = 8) (h2: MN = 10) (h3: NL = 6) (h4: LR = 6) 
  (h5: LT = 8 / 3) (h6: TR = 10 / 3) (h7: LS = 9 / 4) (h8: SR = 15 / 4) :
  LS - LT = -5 / 12 :=
by
  sorry

end length_of_ST_l2226_222672


namespace red_pencils_in_box_l2226_222645

theorem red_pencils_in_box (B R G : ℕ) 
  (h1 : B + R + G = 20)
  (h2 : B = 6 * G)
  (h3 : R < B) : R = 6 := by
  sorry

end red_pencils_in_box_l2226_222645


namespace shortest_distance_between_circles_l2226_222640

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₂.1 - p₁.1) ^ 2 + (p₂.2 - p₁.2) ^ 2)

theorem shortest_distance_between_circles :
  let c₁ := (4, -3)
  let r₁ := 4
  let c₂ := (-5, 1)
  let r₂ := 1
  distance c₁ c₂ - (r₁ + r₂) = Real.sqrt 97 - 5 :=
by
  sorry

end shortest_distance_between_circles_l2226_222640


namespace g_h_2_eq_2175_l2226_222628

def g (x : ℝ) : ℝ := 2 * x^2 - 3
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem g_h_2_eq_2175 : g (h 2) = 2175 := by
  sorry

end g_h_2_eq_2175_l2226_222628


namespace average_honey_per_bee_per_day_l2226_222611

-- Definitions based on conditions
def num_honey_bees : ℕ := 50
def honey_bee_days : ℕ := 35
def total_honey_produced : ℕ := 75
def expected_avg_honey_per_bee_per_day : ℝ := 2.14

-- Statement of the proof problem
theorem average_honey_per_bee_per_day :
  ((total_honey_produced : ℝ) / (num_honey_bees * honey_bee_days)) = expected_avg_honey_per_bee_per_day := by
  sorry

end average_honey_per_bee_per_day_l2226_222611


namespace quadratic_real_roots_range_k_l2226_222650

-- Define the quadratic function
def quadratic_eq (k x : ℝ) : ℝ := k * x^2 - 6 * x + 9

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the conditions for the quadratic equation to have distinct real roots
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ discriminant a b c > 0

theorem quadratic_real_roots_range_k (k : ℝ) :
  has_two_distinct_real_roots k (-6) 9 ↔ k < 1 ∧ k ≠ 0 := 
by
  sorry

end quadratic_real_roots_range_k_l2226_222650


namespace bus_stops_12_minutes_per_hour_l2226_222676

noncomputable def stopping_time (speed_excluding_stoppages : ℝ) (speed_including_stoppages : ℝ) : ℝ :=
  let distance_lost_per_hour := speed_excluding_stoppages - speed_including_stoppages
  let speed_per_minute := speed_excluding_stoppages / 60
  distance_lost_per_hour / speed_per_minute

theorem bus_stops_12_minutes_per_hour :
  stopping_time 50 40 = 12 :=
by
  sorry

end bus_stops_12_minutes_per_hour_l2226_222676


namespace no_x_squared_term_l2226_222600

theorem no_x_squared_term {m : ℚ} (h : (x+1) * (x^2 + 5*m*x + 3) = x^3 + (5*m + 1)*x^2 + (3 + 5*m)*x + 3) : 
  5*m + 1 = 0 → m = -1/5 := by sorry

end no_x_squared_term_l2226_222600


namespace transport_cost_correct_l2226_222624

-- Defining the weights of the sensor unit and communication module in grams
def weight_sensor_grams : ℕ := 500
def weight_comm_module_grams : ℕ := 1500

-- Defining the transport cost per kilogram
def cost_per_kg_sensor : ℕ := 25000
def cost_per_kg_comm_module : ℕ := 20000

-- Converting weights to kilograms
def weight_sensor_kg : ℚ := weight_sensor_grams / 1000
def weight_comm_module_kg : ℚ := weight_comm_module_grams / 1000

-- Calculating the transport costs
def cost_sensor : ℚ := weight_sensor_kg * cost_per_kg_sensor
def cost_comm_module : ℚ := weight_comm_module_kg * cost_per_kg_comm_module

-- Total cost of transporting both units
def total_cost : ℚ := cost_sensor + cost_comm_module

-- Proving that the total cost is $42500
theorem transport_cost_correct : total_cost = 42500 := by
  sorry

end transport_cost_correct_l2226_222624


namespace D_72_value_l2226_222622

-- Define D(n) as described
def D (n : ℕ) : ℕ := 
  sorry -- Placeholder for the actual function definition

-- Theorem statement
theorem D_72_value : D 72 = 97 :=
by sorry

end D_72_value_l2226_222622


namespace first_group_work_done_l2226_222673

-- Define work amounts with the conditions given
variable (W : ℕ) -- amount of work 3 people can do in 3 days
variable (work_rate : ℕ → ℕ → ℕ) -- work_rate(p, d) is work done by p people in d days

-- Conditions
axiom cond1 : work_rate 3 3 = W
axiom cond2 : work_rate 6 3 = 6 * W

-- The proof statement
theorem first_group_work_done : work_rate 3 3 = 2 * W :=
by
  sorry

end first_group_work_done_l2226_222673


namespace cards_eaten_by_hippopotamus_l2226_222670

-- Defining the initial and remaining number of cards
def initial_cards : ℕ := 72
def remaining_cards : ℕ := 11

-- Statement of the proof problem
theorem cards_eaten_by_hippopotamus (initial_cards remaining_cards : ℕ) : initial_cards - remaining_cards = 61 :=
by
  sorry

end cards_eaten_by_hippopotamus_l2226_222670


namespace root_constraints_between_zero_and_twoR_l2226_222661

variable (R l a : ℝ)
variable (hR : R > 0) (hl : l > 0) (ha_nonzero : a ≠ 0)

theorem root_constraints_between_zero_and_twoR :
  ∀ (x : ℝ), (2 * R * x^2 - (l^2 + 4 * a * R) * x + 2 * R * a^2 = 0) →
  (0 < x ∧ x < 2 * R) ↔
  (a > 0 ∧ a < 2 * R ∧ l^2 < (2 * R - a)^2) ∨
  (a < 0 ∧ -2 * R < a ∧ l^2 < (2 * R - a)^2) :=
sorry

end root_constraints_between_zero_and_twoR_l2226_222661


namespace sum_of_consecutive_numbers_with_lcm_168_l2226_222649

theorem sum_of_consecutive_numbers_with_lcm_168 (n : ℕ) (h_lcm : Nat.lcm (Nat.lcm n (n + 1)) (n + 2) = 168) : n + (n + 1) + (n + 2) = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l2226_222649


namespace son_work_rate_l2226_222608

theorem son_work_rate (M S : ℝ) (hM : M = 1 / 5) (hMS : M + S = 1 / 4) : 1 / S = 20 :=
by
  sorry

end son_work_rate_l2226_222608


namespace solve_inequality_system_l2226_222655

-- Define the inequalities as conditions.
def cond1 (x : ℝ) := 2 * x + 1 < 3 * x - 2
def cond2 (x : ℝ) := 3 * (x - 2) - x ≤ 4

-- Formulate the theorem to prove that these conditions give the solution 3 < x ≤ 5.
theorem solve_inequality_system (x : ℝ) : cond1 x ∧ cond2 x ↔ 3 < x ∧ x ≤ 5 := 
sorry

end solve_inequality_system_l2226_222655


namespace min_value_expression_l2226_222675

theorem min_value_expression (α β : ℝ) : 
  ∃ a b : ℝ, 
    ((2 * Real.cos α + 5 * Real.sin β - 8) ^ 2 + 
    (2 * Real.sin α + 5 * Real.cos β - 15) ^ 2  = 100) :=
sorry

end min_value_expression_l2226_222675


namespace ticket_price_l2226_222617

theorem ticket_price (regular_price : ℕ) (discount_percent : ℕ) (paid_price : ℕ) 
  (h1 : regular_price = 15) 
  (h2 : discount_percent = 40) 
  (h3 : paid_price = regular_price - (regular_price * discount_percent / 100)) : 
  paid_price = 9 :=
by
  sorry

end ticket_price_l2226_222617


namespace final_movie_length_l2226_222618

-- Definitions based on conditions
def original_movie_length : ℕ := 60
def cut_scene_length : ℕ := 3

-- Theorem statement proving the final length of the movie
theorem final_movie_length : original_movie_length - cut_scene_length = 57 :=
by
  -- The proof will go here
  sorry

end final_movie_length_l2226_222618


namespace original_bill_l2226_222652

theorem original_bill (m : ℝ) (h1 : 10 * (m / 10) = m)
                      (h2 : 9 * ((m - 10) / 10 + 3) = m - 10) :
  m = 180 :=
  sorry

end original_bill_l2226_222652


namespace fg_at_3_equals_97_l2226_222657

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_at_3_equals_97 : f (g 3) = 97 := by
  sorry

end fg_at_3_equals_97_l2226_222657


namespace maximize_revenue_l2226_222669

noncomputable def revenue (p : ℝ) : ℝ :=
p * (145 - 7 * p)

theorem maximize_revenue : ∃ p : ℕ, p ≤ 30 ∧ p = 10 ∧ ∀ q ≤ 30, revenue (q : ℝ) ≤ revenue 10 :=
by
  sorry

end maximize_revenue_l2226_222669


namespace maximum_value_x2_add_3xy_add_y2_l2226_222636

-- Define the conditions
variables {x y : ℝ}

-- State the theorem
theorem maximum_value_x2_add_3xy_add_y2 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : 3 * x^2 - 2 * x * y + 5 * y^2 = 12) :
  ∃ e f g h : ℕ,
    x^2 + 3 * x * y + y^2 = (1144 + 204 * Real.sqrt 15) / 91 ∧ e + f + g + h = 1454 :=
sorry

end maximum_value_x2_add_3xy_add_y2_l2226_222636


namespace equations_have_different_graphs_l2226_222664

theorem equations_have_different_graphs :
  (∃ (x : ℝ), ∀ (y₁ y₂ y₃ : ℝ),
    (y₁ = x - 2) ∧
    (y₂ = (x^2 - 4) / (x + 2) ∧ x ≠ -2) ∧
    (y₃ = (x^2 - 4) / (x + 2) ∧ x ≠ -2 ∨ (x = -2 ∧ ∀ y₃ : ℝ, (x+2) * y₃ = x^2 - 4)))
  → (∃ y₁ y₂ y₃ : ℝ, y₁ ≠ y₂ ∨ y₁ ≠ y₃ ∨ y₂ ≠ y₃) := sorry

end equations_have_different_graphs_l2226_222664


namespace normal_price_of_article_l2226_222604

theorem normal_price_of_article (P : ℝ) (h : 0.90 * 0.80 * P = 36) : P = 50 :=
by {
  sorry
}

end normal_price_of_article_l2226_222604


namespace smallest_exponentiated_number_l2226_222621

theorem smallest_exponentiated_number :
  127^8 < 63^10 ∧ 63^10 < 33^12 := 
by 
  -- Proof omitted
  sorry

end smallest_exponentiated_number_l2226_222621


namespace inequality_part1_inequality_part2_l2226_222647

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l2226_222647


namespace divisibility_ac_bd_l2226_222680

-- Conditions definitions
variable (a b c d : ℕ)
variable (hab : a ∣ b)
variable (hcd : c ∣ d)

-- Goal
theorem divisibility_ac_bd : (a * c) ∣ (b * d) :=
  sorry

end divisibility_ac_bd_l2226_222680


namespace complement_A_B_correct_l2226_222656

open Set

-- Given sets A and B
def A : Set ℕ := {0, 2, 4, 6, 8, 10}
def B : Set ℕ := {4, 8}

-- Define the complement of B with respect to A
def complement_A_B : Set ℕ := A \ B

-- Statement to prove
theorem complement_A_B_correct : complement_A_B = {0, 2, 6, 10} :=
  by sorry

end complement_A_B_correct_l2226_222656


namespace solve_equation_l2226_222629

open Real

noncomputable def verify_solution (x : ℝ) : Prop :=
  1 / ((x - 3) * (x - 4)) +
  1 / ((x - 4) * (x - 5)) +
  1 / ((x - 5) * (x - 6)) = 1 / 8

theorem solve_equation (x : ℝ) (h : x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6) :
  verify_solution x ↔ (x = (9 + sqrt 57) / 2 ∨ x = (9 - sqrt 57) / 2) := 
by
  sorry

end solve_equation_l2226_222629


namespace hunting_season_fraction_l2226_222626

noncomputable def fraction_of_year_hunting_season (hunting_times_per_month : ℕ) 
    (deers_per_hunt : ℕ) (weight_per_deer : ℕ) (fraction_kept : ℚ) 
    (total_weight_kept : ℕ) : ℚ :=
  let total_yearly_weight := total_weight_kept * 2
  let weight_per_hunt := deers_per_hunt * weight_per_deer
  let total_hunts_per_year := total_yearly_weight / weight_per_hunt
  let total_months_hunting := total_hunts_per_year / hunting_times_per_month
  let fraction_of_year := total_months_hunting / 12
  fraction_of_year

theorem hunting_season_fraction : 
  fraction_of_year_hunting_season 6 2 600 (1 / 2 : ℚ) 10800 = 1 / 4 := 
by
  simp [fraction_of_year_hunting_season]
  sorry

end hunting_season_fraction_l2226_222626


namespace value_of_expression_l2226_222642

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2006 = 2007 :=
sorry

end value_of_expression_l2226_222642


namespace combined_marbles_l2226_222692

def Rhonda_marbles : ℕ := 80
def Amon_marbles : ℕ := Rhonda_marbles + 55

theorem combined_marbles : Amon_marbles + Rhonda_marbles = 215 :=
by
  sorry

end combined_marbles_l2226_222692


namespace number_of_children_l2226_222616

variables (n : ℕ) (y : ℕ) (d : ℕ)

def sum_of_ages (n : ℕ) (y : ℕ) (d : ℕ) : ℕ :=
  n * y + d * (n * (n - 1) / 2)

theorem number_of_children (H1 : sum_of_ages n 6 3 = 60) : n = 6 :=
by {
  sorry
}

end number_of_children_l2226_222616


namespace find_starting_number_l2226_222646

theorem find_starting_number (num_even_ints: ℕ) (end_num: ℕ) (h_num: num_even_ints = 35) (h_end: end_num = 95) : 
  ∃ start_num: ℕ, start_num = 24 ∧ (∀ n: ℕ, (start_num + 2 * n ≤ end_num ∧ n < num_even_ints)) := by
  sorry

end find_starting_number_l2226_222646


namespace length_of_bridge_l2226_222609

noncomputable def L_train : ℝ := 110
noncomputable def v_train : ℝ := 72 * (1000 / 3600)
noncomputable def t : ℝ := 12.099

theorem length_of_bridge : (v_train * t - L_train) = 131.98 :=
by
  -- The proof should come here
  sorry

end length_of_bridge_l2226_222609


namespace complex_problem_l2226_222667

theorem complex_problem (a b : ℝ) (h : (⟨a, 3⟩ : ℂ) + ⟨2, -1⟩ = ⟨5, b⟩) : a * b = 6 := by
  sorry

end complex_problem_l2226_222667


namespace units_digit_of_first_four_composite_numbers_l2226_222662

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l2226_222662


namespace find_dividend_l2226_222638

variable (Divisor Quotient Remainder Dividend : ℕ)
variable (h₁ : Divisor = 15)
variable (h₂ : Quotient = 8)
variable (h₃ : Remainder = 5)

theorem find_dividend : Dividend = 125 ↔ Dividend = Divisor * Quotient + Remainder := by
  sorry

end find_dividend_l2226_222638


namespace percentage_of_men_in_company_l2226_222660

theorem percentage_of_men_in_company 
  (M W : ℝ) 
  (h1 : 0.60 * M + 0.35 * W = 50) 
  (h2 : M + W = 100) : 
  M = 60 :=
by
  sorry

end percentage_of_men_in_company_l2226_222660


namespace woodworker_tables_count_l2226_222620

/-- A woodworker made a total of 40 furniture legs and has built 6 chairs.
    Each chair requires 4 legs. Prove that the number of tables made is 4,
    assuming each table also requires 4 legs. -/
theorem woodworker_tables_count (total_legs chairs tables : ℕ)
  (legs_per_chair legs_per_table : ℕ)
  (H1 : total_legs = 40)
  (H2 : chairs = 6)
  (H3 : legs_per_chair = 4)
  (H4 : legs_per_table = 4)
  (H5 : total_legs = chairs * legs_per_chair + tables * legs_per_table) :
  tables = 4 := 
  sorry

end woodworker_tables_count_l2226_222620


namespace inequality_correct_l2226_222607

-- Theorem: For all real numbers x and y, if x ≥ y, then x² + y² ≥ 2xy.
theorem inequality_correct (x y : ℝ) (h : x ≥ y) : x^2 + y^2 ≥ 2 * x * y := 
by {
  -- Placeholder for the proof
  sorry
}

end inequality_correct_l2226_222607


namespace wholesale_cost_l2226_222688

variable (W R P : ℝ)

-- Conditions
def retail_price := R = 1.20 * W
def employee_discount := P = 0.95 * R
def employee_payment := P = 228

-- Theorem statement
theorem wholesale_cost (H1 : retail_price R W) (H2 : employee_discount P R) (H3 : employee_payment P) : W = 200 :=
by
  sorry

end wholesale_cost_l2226_222688
