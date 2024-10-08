import Mathlib

namespace jack_initial_yen_l10_10835

theorem jack_initial_yen 
  (pounds yen_per_pound euros pounds_per_euro total_yen : ℕ)
  (h₁ : pounds = 42)
  (h₂ : euros = 11)
  (h₃ : pounds_per_euro = 2)
  (h₄ : yen_per_pound = 100)
  (h₅ : total_yen = 9400) : 
  ∃ initial_yen : ℕ, initial_yen = 3000 :=
by
  sorry

end jack_initial_yen_l10_10835


namespace black_white_tile_ratio_l10_10877

/-- Assume the original pattern has 12 black tiles and 25 white tiles.
    The pattern is extended by attaching a border of black tiles two tiles wide around the square.
    Prove that the ratio of black tiles to white tiles in the new extended pattern is 76/25.-/
theorem black_white_tile_ratio 
  (original_black_tiles : ℕ)
  (original_white_tiles : ℕ)
  (black_border_width : ℕ)
  (new_black_tiles : ℕ)
  (total_new_tiles : ℕ) 
  (total_old_tiles : ℕ) 
  (new_white_tiles : ℕ)
  : original_black_tiles = 12 → 
    original_white_tiles = 25 → 
    black_border_width = 2 → 
    total_old_tiles = 36 →
    total_new_tiles = 100 →
    new_black_tiles = 76 → 
    new_white_tiles = 25 → 
    (new_black_tiles : ℚ) / (new_white_tiles : ℚ) = 76 / 25 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end black_white_tile_ratio_l10_10877


namespace square_binomial_l10_10198

theorem square_binomial (x : ℝ) : (-x - 1) ^ 2 = x^2 + 2 * x + 1 :=
by
  sorry

end square_binomial_l10_10198


namespace frogs_meet_time_proven_l10_10665

-- Define the problem
def frogs_will_meet_at_time : Prop :=
  ∃ (meet_time : Nat),
    let initial_time := 12 * 60 -- 12:00 PM in minutes
    let initial_distance := 2015
    let green_frog_jump := 9
    let blue_frog_jump := 8 
    let combined_reduction := green_frog_jump + blue_frog_jump
    initial_distance % combined_reduction = 0 ∧
    meet_time == initial_time + (2 * (initial_distance / combined_reduction))

theorem frogs_meet_time_proven (h : frogs_will_meet_at_time) : meet_time = 15 * 60 + 56 :=
sorry

end frogs_meet_time_proven_l10_10665


namespace value_of_a_b_c_l10_10757

theorem value_of_a_b_c 
    (a b c : Int)
    (h1 : ∀ x : Int, x^2 + 10*x + 21 = (x + a) * (x + b))
    (h2 : ∀ x : Int, x^2 + 3*x - 88 = (x + b) * (x - c))
    :
    a + b + c = 18 := 
sorry

end value_of_a_b_c_l10_10757


namespace simplify_trig_expression_l10_10948

theorem simplify_trig_expression :
  (Real.cos (72 * Real.pi / 180) * Real.sin (78 * Real.pi / 180) +
   Real.sin (72 * Real.pi / 180) * Real.sin (12 * Real.pi / 180) = 1 / 2) :=
by sorry

end simplify_trig_expression_l10_10948


namespace problem1_problem2_l10_10522

-- Definitions used directly from conditions
def inequality (m x : ℝ) : Prop := m * x ^ 2 - 2 * m * x - 1 < 0

-- Proof problem (1)
theorem problem1 (m : ℝ) (h : ∀ x : ℝ, inequality m x) : -1 < m ∧ m ≤ 0 :=
sorry

-- Proof problem (2)
theorem problem2 (x : ℝ) (h : ∀ m : ℝ, |m| ≤ 1 → inequality m x) :
  (1 - Real.sqrt 2 < x ∧ x < 1) ∨ (1 < x ∧ x < 1 + Real.sqrt 2) :=
sorry

end problem1_problem2_l10_10522


namespace find_p_q_sum_l10_10675

theorem find_p_q_sum (p q : ℝ) :
  (∀ x : ℝ, (x - 3) * (x - 5) = 0 → 3 * x ^ 2 - p * x + q = 0) →
  p = 24 ∧ q = 45 ∧ p + q = 69 :=
by
  intros h
  have h3 := h 3 (by ring)
  have h5 := h 5 (by ring)
  sorry

end find_p_q_sum_l10_10675


namespace calculate_sequences_l10_10577

-- Definitions of sequences and constants
def a (n : ℕ) := 2 * n + 1
def b (n : ℕ) := 3 ^ n
def S (n : ℕ) := n * (n + 2)
def T (n : ℕ) := (3 / 4 : ℚ) - (2 * n + 3) / (2 * (n + 1) * (n + 2))

-- Hypotheses and proofs
theorem calculate_sequences (d : ℕ) (a1 : ℕ) (h_d : d = 2) (h_a1 : a1 = 3) :
  ∀ n, (a n = 2 * n + 1) ∧ (b 1 = a 1) ∧ (b 2 = a 4) ∧ (b 3 = a 13) ∧ (b n = 3 ^ n) ∧
  (S n = n * (n + 2)) ∧ (T n = (3 / 4 : ℚ) - (2 * n + 3) / (2 * (n + 1) * (n + 2))) :=
by
  intros
  -- Skipping proof steps with sorry
  sorry

end calculate_sequences_l10_10577


namespace neg_p_equivalence_l10_10599

theorem neg_p_equivalence:
  (∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) :=
sorry

end neg_p_equivalence_l10_10599


namespace sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog_l10_10411

theorem sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog
  (a r : ℝ)
  (volume_cond : a^3 * r^3 = 288)
  (surface_area_cond : 2 * (a^2 * r^4 + a^2 * r^2 + a^2 * r) = 288)
  (geom_prog : True) :
  4 * (a * r^2 + a * r + a) = 92 := 
sorry

end sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog_l10_10411


namespace probability_from_first_to_last_l10_10717

noncomputable def probability_path_possible (n : ℕ) : ℝ :=
  (2 ^ (n - 1) : ℝ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℝ)

theorem probability_from_first_to_last (n : ℕ) (h : n > 1) :
  probability_path_possible n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_l10_10717


namespace initial_apples_l10_10018

theorem initial_apples (X : ℕ) (h : X - 2 + 3 = 5) : X = 4 :=
sorry

end initial_apples_l10_10018


namespace total_coffee_needed_l10_10211

-- Conditions as definitions
def weak_coffee_amount_per_cup : ℕ := 1
def strong_coffee_amount_per_cup : ℕ := 2 * weak_coffee_amount_per_cup
def cups_of_weak_coffee : ℕ := 12
def cups_of_strong_coffee : ℕ := 12

-- Prove that the total amount of coffee needed equals 36 tablespoons
theorem total_coffee_needed : (weak_coffee_amount_per_cup * cups_of_weak_coffee) + (strong_coffee_amount_per_cup * cups_of_strong_coffee) = 36 :=
by
  sorry

end total_coffee_needed_l10_10211


namespace value_added_to_each_number_is_12_l10_10767

theorem value_added_to_each_number_is_12
    (sum_original : ℕ)
    (sum_new : ℕ)
    (n : ℕ)
    (avg_original : ℕ)
    (avg_new : ℕ)
    (value_added : ℕ) :
  (n = 15) →
  (avg_original = 40) →
  (avg_new = 52) →
  (sum_original = n * avg_original) →
  (sum_new = n * avg_new) →
  (value_added = (sum_new - sum_original) / n) →
  value_added = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end value_added_to_each_number_is_12_l10_10767


namespace integer_part_mod_8_l10_10357

theorem integer_part_mod_8 (n : ℕ) (h : n ≥ 2009) :
  ∃ x : ℝ, x = (3 + Real.sqrt 8)^(2 * n) ∧ Int.floor (x) % 8 = 1 := 
sorry

end integer_part_mod_8_l10_10357


namespace original_number_l10_10723

theorem original_number (x y : ℝ) (h1 : 10 * x + 22 * y = 780) (h2 : y = 34) : x + y = 37.2 :=
sorry

end original_number_l10_10723


namespace rectangle_area_l10_10578

theorem rectangle_area (L W P A : ℕ) (h1 : P = 52) (h2 : L = 11) (h3 : 2 * L + 2 * W = P) : 
  A = L * W → A = 165 :=
by
  sorry

end rectangle_area_l10_10578


namespace root_in_interval_l10_10918

noncomputable def f (m x : ℝ) := m * 3^x - x + 3

theorem root_in_interval (m : ℝ) (h1 : m < 0) (h2 : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f m x = 0) : -3 < m ∧ m < -2/3 :=
by
  sorry

end root_in_interval_l10_10918


namespace impossible_score_53_l10_10167

def quizScoring (total_questions correct_answers incorrect_answers unanswered_questions score: ℤ) : Prop :=
  total_questions = 15 ∧
  correct_answers + incorrect_answers + unanswered_questions = 15 ∧
  score = 4 * correct_answers - incorrect_answers ∧
  unanswered_questions ≥ 0 ∧ correct_answers ≥ 0 ∧ incorrect_answers ≥ 0

theorem impossible_score_53 :
  ¬ ∃ (correct_answers incorrect_answers unanswered_questions : ℤ), quizScoring 15 correct_answers incorrect_answers unanswered_questions 53 := 
sorry

end impossible_score_53_l10_10167


namespace ball_box_distribution_l10_10412

theorem ball_box_distribution :
  ∃ (distinct_ways : ℕ), distinct_ways = 7 :=
by
  let num_balls := 5
  let num_boxes := 4
  sorry

end ball_box_distribution_l10_10412


namespace total_number_of_flowers_is_correct_l10_10619

-- Define the conditions
def number_of_pots : ℕ := 544
def flowers_per_pot : ℕ := 32
def total_flowers : ℕ := number_of_pots * flowers_per_pot

-- State the theorem to be proved
theorem total_number_of_flowers_is_correct :
  total_flowers = 17408 :=
by
  sorry

end total_number_of_flowers_is_correct_l10_10619


namespace pythagorean_theorem_special_cases_l10_10466

open Nat

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

theorem pythagorean_theorem_special_cases (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (is_even a ∨ is_even b) ∧ 
  (is_multiple_of_3 a ∨ is_multiple_of_3 b) ∧ 
  (is_multiple_of_5 a ∨ is_multiple_of_5 b ∨ is_multiple_of_5 c) :=
by
  sorry

end pythagorean_theorem_special_cases_l10_10466


namespace prime_not_divisor_ab_cd_l10_10759

theorem prime_not_divisor_ab_cd {a b c d : ℕ} (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) (hd: 0 < d) 
  (p : ℕ) (hp : p = a + b + c + d) (hprime : Nat.Prime p) : ¬ p ∣ (a * b - c * d) := 
sorry

end prime_not_divisor_ab_cd_l10_10759


namespace equilateral_triangle_l10_10724

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ) (p R : ℝ)
  (h : (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = p / (9 * R)) :
  a = b ∧ b = c ∧ a = c :=
sorry

end equilateral_triangle_l10_10724


namespace no_unhappy_days_l10_10634

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end no_unhappy_days_l10_10634


namespace identify_clothes_l10_10389

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l10_10389


namespace total_cans_collected_l10_10682

theorem total_cans_collected (students_perez : ℕ) (half_perez_collected_20 : ℕ) (two_perez_collected_0 : ℕ) (remaining_perez_collected_8 : ℕ)
                             (students_johnson : ℕ) (third_johnson_collected_25 : ℕ) (three_johnson_collected_0 : ℕ) (remaining_johnson_collected_10 : ℕ)
                             (hp : students_perez = 28) (hc1 : half_perez_collected_20 = 28 / 2) (hc2 : two_perez_collected_0 = 2) (hc3 : remaining_perez_collected_8 = 12)
                             (hj : students_johnson = 30) (jc1 : third_johnson_collected_25 = 30 / 3) (jc2 : three_johnson_collected_0 = 3) (jc3 : remaining_johnson_collected_10 = 18) :
    (half_perez_collected_20 * 20 + two_perez_collected_0 * 0 + remaining_perez_collected_8 * 8
    + third_johnson_collected_25 * 25 + three_johnson_collected_0 * 0 + remaining_johnson_collected_10 * 10) = 806 :=
by
  sorry

end total_cans_collected_l10_10682


namespace circumscribed_sphere_surface_area_l10_10563

theorem circumscribed_sphere_surface_area
  (x y z : ℝ)
  (h1 : x * y = Real.sqrt 6)
  (h2 : y * z = Real.sqrt 2)
  (h3 : z * x = Real.sqrt 3) :
  let l := Real.sqrt (x^2 + y^2 + z^2)
  let R := l / 2
  4 * Real.pi * R^2 = 6 * Real.pi :=
by sorry

end circumscribed_sphere_surface_area_l10_10563


namespace dina_dolls_count_l10_10816

-- Define the conditions
variable (Ivy_dolls : ℕ)
variable (Collectors_Ivy_dolls : ℕ := 20)
variable (Dina_dolls : ℕ)

-- Condition: Ivy has 2/3 of her dolls as collectors editions
def collectors_edition_condition : Prop := (2 / 3 : ℝ) * Ivy_dolls = Collectors_Ivy_dolls

-- Condition: Dina has twice as many dolls as Ivy
def dina_ivy_dolls_relationship : Prop := Dina_dolls = 2 * Ivy_dolls

-- Theorem: Prove that Dina has 60 dolls
theorem dina_dolls_count : collectors_edition_condition Ivy_dolls ∧ dina_ivy_dolls_relationship Ivy_dolls Dina_dolls → Dina_dolls = 60 := by
  sorry

end dina_dolls_count_l10_10816


namespace three_digit_number_l10_10428

theorem three_digit_number (m : ℕ) : (300 * m + 10 * m + (m - 1)) = (311 * m - 1) :=
by 
  sorry

end three_digit_number_l10_10428


namespace purely_imaginary_satisfies_condition_l10_10022

theorem purely_imaginary_satisfies_condition (m : ℝ) (h1 : m^2 + 3 * m - 4 = 0) (h2 : m + 4 ≠ 0) : m = 1 :=
by
  sorry

end purely_imaginary_satisfies_condition_l10_10022


namespace find_number_l10_10983

theorem find_number:
  ∃ x : ℝ, x + 1.35 + 0.123 = 1.794 ∧ x = 0.321 :=
by
  sorry

end find_number_l10_10983


namespace b_integer_iff_a_special_form_l10_10385

theorem b_integer_iff_a_special_form (a : ℝ) (b : ℝ) 
  (h1 : a > 0) 
  (h2 : b = (a + Real.sqrt (a ^ 2 + 1)) ^ (1 / 3) + (a - Real.sqrt (a ^ 2 + 1)) ^ (1 / 3)) : 
  (∃ (n : ℕ), a = 1 / 2 * (n * (n^2 + 3))) ↔ (∃ (n : ℕ), b = n) :=
sorry

end b_integer_iff_a_special_form_l10_10385


namespace simplify_expression_l10_10659

theorem simplify_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 6) + 7) + 2 = -x^4 + 3 * x^3 - 6 * x^2 + 7 * x + 2 := 
by 
  sorry

end simplify_expression_l10_10659


namespace find_W_l10_10683

noncomputable def volume_of_space (r_sphere r_cylinder h_cylinder : ℝ) : ℝ :=
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * h_cylinder
  let V_cone := (1 / 3) * Real.pi * r_cylinder^2 * h_cylinder
  V_sphere - V_cylinder - V_cone

theorem find_W : volume_of_space 6 4 10 = (224 / 3) * Real.pi := by
  sorry

end find_W_l10_10683


namespace count_positive_integers_satisfy_l10_10477

theorem count_positive_integers_satisfy :
  ∃ (S : Finset ℕ), (∀ n ∈ S, (n + 5) * (n - 3) * (n - 12) * (n - 17) < 0) ∧ S.card = 4 :=
by
  sorry

end count_positive_integers_satisfy_l10_10477


namespace compute_expression_l10_10980

theorem compute_expression : (3 + 7)^2 + (3^2 + 7^2 + 5^2) = 183 := by
  sorry

end compute_expression_l10_10980


namespace player_B_wins_in_least_steps_l10_10337

noncomputable def least_steps_to_win (n : ℕ) : ℕ :=
  n

theorem player_B_wins_in_least_steps (n : ℕ) (h_n : n > 0) :
  ∃ k, k = least_steps_to_win n ∧ k = n := by
  sorry

end player_B_wins_in_least_steps_l10_10337


namespace factorize_expression_l10_10552

theorem factorize_expression (a x y : ℤ) : a * x - a * y = a * (x - y) :=
  sorry

end factorize_expression_l10_10552


namespace remainder_of_polynomial_division_l10_10844

theorem remainder_of_polynomial_division
  (x : ℝ)
  (h : 2 * x - 4 = 0) :
  (8 * x^4 - 18 * x^3 + 6 * x^2 - 4 * x + 30) % (2 * x - 4) = 30 := by
  sorry

end remainder_of_polynomial_division_l10_10844


namespace cooler_capacity_l10_10513

theorem cooler_capacity (C : ℝ) (h1 : 3.25 * C = 325) : C = 100 :=
sorry

end cooler_capacity_l10_10513


namespace simplify_and_evaluate_l10_10990

noncomputable def given_expression (a : ℝ) : ℝ :=
  (a-4) / a / ((a+2) / (a^2 - 2 * a) - (a-1) / (a^2 - 4 * a + 4))

theorem simplify_and_evaluate (a : ℝ) (h : a^2 - 4 * a + 3 = 0) : given_expression a = 1 := by
  sorry

end simplify_and_evaluate_l10_10990


namespace break_even_number_of_books_l10_10548

-- Definitions from conditions.
def fixed_cost : ℝ := 50000
def variable_cost_per_book : ℝ := 4
def selling_price_per_book : ℝ := 9

-- Main statement proving the break-even point.
theorem break_even_number_of_books 
  (x : ℕ) : (selling_price_per_book * x = fixed_cost + variable_cost_per_book * x) → (x = 10000) :=
by
  sorry

end break_even_number_of_books_l10_10548


namespace percentage_needed_to_pass_l10_10447

-- Define conditions
def student_score : ℕ := 80
def marks_shortfall : ℕ := 40
def total_marks : ℕ := 400

-- Theorem statement: The percentage of marks required to pass the test.
theorem percentage_needed_to_pass : (student_score + marks_shortfall) * 100 / total_marks = 30 := by
  sorry

end percentage_needed_to_pass_l10_10447


namespace syllogism_sequence_correct_l10_10250

-- Definitions based on conditions
def square_interior_angles_equal : Prop := ∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D
def rectangle_interior_angles_equal : Prop := ∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D
def square_is_rectangle : Prop := ∀ (S : Type), S = S

-- Final Goal
theorem syllogism_sequence_correct : (rectangle_interior_angles_equal → square_is_rectangle → square_interior_angles_equal) :=
by
  sorry

end syllogism_sequence_correct_l10_10250


namespace remainder_of_67_pow_67_plus_67_mod_68_l10_10866

theorem remainder_of_67_pow_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end remainder_of_67_pow_67_plus_67_mod_68_l10_10866


namespace correct_sum_rounded_l10_10523

-- Define the conditions: sum and rounding
def sum_58_46 : ℕ := 58 + 46
def round_to_nearest_hundred (n : ℕ) : ℕ :=
  if n % 100 >= 50 then ((n / 100) + 1) * 100 else (n / 100) * 100

-- state the theorem
theorem correct_sum_rounded :
  round_to_nearest_hundred sum_58_46 = 100 :=
by
  sorry

end correct_sum_rounded_l10_10523


namespace total_number_of_students_l10_10694

namespace StudentRanking

def rank_from_right := 17
def rank_from_left := 5
def total_students (rank_from_right rank_from_left : ℕ) := rank_from_right + rank_from_left - 1

theorem total_number_of_students : total_students rank_from_right rank_from_left = 21 :=
by
  sorry

end StudentRanking

end total_number_of_students_l10_10694


namespace mn_not_equal_l10_10471

-- Define conditions for the problem
def isValidN (N : ℕ) (n : ℕ) : Prop :=
  0 ≤ N ∧ N < 10^n ∧ N % 4 = 0 ∧ ((N.digits 10).sum % 4 = 0)

-- Define the number M_n of integers N satisfying the conditions
noncomputable def countMn (n : ℕ) : ℕ :=
  Nat.card { N : ℕ | isValidN N n }

-- Define the theorem stating the problem's conclusion
theorem mn_not_equal (n : ℕ) (hn : n > 0) : 
  countMn n ≠ 10^n / 16 :=
sorry

end mn_not_equal_l10_10471


namespace a_and_b_together_30_days_l10_10747

variable (R_a R_b : ℝ)

-- Conditions
axiom condition1 : R_a = 3 * R_b
axiom condition2 : R_a * 40 = (R_a + R_b) * 30

-- Question: prove that a and b together can complete the work in 30 days.
theorem a_and_b_together_30_days (R_a R_b : ℝ) (condition1 : R_a = 3 * R_b) (condition2 : R_a * 40 = (R_a + R_b) * 30) : true :=
by
  sorry

end a_and_b_together_30_days_l10_10747


namespace weight_measurement_l10_10117

theorem weight_measurement :
  ∀ (w : Set ℕ), w = {1, 3, 9, 27} → (∀ n ∈ w, ∃ k, k = n ∧ k ∈ w) →
  ∃ (num_sets : ℕ), num_sets = 41 := by
  intros w hw hcomb
  sorry

end weight_measurement_l10_10117


namespace range_of_x_l10_10825

theorem range_of_x 
  (x : ℝ)
  (h1 : 1 / x < 4) 
  (h2 : 1 / x > -6) 
  (h3 : x < 0) : 
  -1 / 6 < x ∧ x < 0 := 
by 
  sorry

end range_of_x_l10_10825


namespace daniel_spent_2290_l10_10296

theorem daniel_spent_2290 (total_games: ℕ) (price_12_games count_price_12: ℕ) 
  (price_7_games frac_price_7: ℕ) (price_3_games: ℕ) 
  (count_price_7: ℕ) (h1: total_games = 346)
  (h2: count_price_12 = 80) (h3: price_12_games = 12)
  (h4: frac_price_7 = 50) (h5: price_7_games = 7)
  (h6: price_3_games = 3) (h7: count_price_7 = (frac_price_7 * (total_games - count_price_12)) / 100):
  (count_price_12 * price_12_games) + (count_price_7 * price_7_games) + ((total_games - count_price_12 - count_price_7) * price_3_games) = 2290 := 
by
  sorry

end daniel_spent_2290_l10_10296


namespace petya_can_force_difference_2014_l10_10272

theorem petya_can_force_difference_2014 :
  ∀ (p q r : ℚ), ∃ (a b c : ℚ), ∀ (x : ℝ), (x^3 + a * x^2 + b * x + c = 0) → 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2014 :=
by sorry

end petya_can_force_difference_2014_l10_10272


namespace alex_class_size_l10_10420

theorem alex_class_size 
  (n : ℕ) 
  (h_top : 30 ≤ n)
  (h_bottom : 30 ≤ n) 
  (h_better : n - 30 > 0)
  (h_worse : n - 30 > 0)
  : n = 59 := 
sorry

end alex_class_size_l10_10420


namespace quadratic_equation_general_form_l10_10032

theorem quadratic_equation_general_form (x : ℝ) (h : 4 * x = x^2 - 8) : x^2 - 4 * x - 8 = 0 :=
sorry

end quadratic_equation_general_form_l10_10032


namespace fraction_of_q_age_l10_10616

theorem fraction_of_q_age (P Q : ℕ) (h1 : P / Q = 3 / 4) (h2 : P + Q = 28) : (P - 0) / (Q - 0) = 3 / 4 :=
by
  sorry

end fraction_of_q_age_l10_10616


namespace min_value_of_sum_squares_l10_10588

theorem min_value_of_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
    x^2 + y^2 + z^2 ≥ 10 :=
sorry

end min_value_of_sum_squares_l10_10588


namespace minimum_value_of_f_l10_10702

noncomputable def f (x : ℝ) := 2 * x + 18 / x

theorem minimum_value_of_f :
  ∃ x > 0, f x = 12 ∧ ∀ y > 0, f y ≥ 12 :=
by
  sorry

end minimum_value_of_f_l10_10702


namespace successful_combinations_l10_10095

def herbs := 4
def gems := 6
def incompatible_combinations := 3

theorem successful_combinations : herbs * gems - incompatible_combinations = 21 := by
  sorry

end successful_combinations_l10_10095


namespace curve_C2_eqn_l10_10737

theorem curve_C2_eqn (p : ℝ) (x y : ℝ) :
  (∃ x y, (x^2 - y^2 = 1) ∧ (y^2 = 2 * p * x) ∧ (2 * p = 3/4)) →
  (y^2 = (3/2) * x) :=
by
  sorry

end curve_C2_eqn_l10_10737


namespace lowest_score_is_C_l10_10230

variable (Score : Type) [LinearOrder Score]
variable (A B C : Score)

-- Translate conditions into Lean
variable (h1 : B ≠ max A (max B C) → A = min A (min B C))
variable (h2 : C ≠ min A (min B C) → A = max A (max B C))

-- Define the proof goal
theorem lowest_score_is_C : min A (min B C) =C :=
by
  sorry

end lowest_score_is_C_l10_10230


namespace multiples_of_7_between_20_and_150_l10_10858

def number_of_multiples_of_7_between (a b : ℕ) : ℕ :=
  (b / 7) - (a / 7) + (if a % 7 = 0 then 1 else 0)

theorem multiples_of_7_between_20_and_150 : number_of_multiples_of_7_between 21 147 = 19 := by
  sorry

end multiples_of_7_between_20_and_150_l10_10858


namespace sum_q_p_evaluation_l10_10046

def p (x : Int) : Int := x^2 - 3
def q (x : Int) : Int := x - 2

def T : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

noncomputable def f (x : Int) : Int := q (p x)

noncomputable def sum_f_T : Int := List.sum (List.map f T)

theorem sum_q_p_evaluation :
  sum_f_T = 15 :=
by
  sorry

end sum_q_p_evaluation_l10_10046


namespace intersection_eq_l10_10651

-- Define sets P and Q
def setP := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2}
def setQ := {y : ℝ | ∃ x : ℝ, y = -x + 2}

-- The main theorem statement
theorem intersection_eq: setP ∩ setQ = {y : ℝ | y ≤ 2} :=
by
  sorry

end intersection_eq_l10_10651


namespace units_digit_7_pow_2023_l10_10696

-- Define a function to compute the units digit
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Main theorem to prove: the units digit of 7^2023 equals 3
theorem units_digit_7_pow_2023 : units_digit (7^2023) = 3 :=
by {
  -- It suffices to show that the units digits of powers of 7 repeat every 4 with the pattern [7, 9, 3, 1]
  sorry
}

end units_digit_7_pow_2023_l10_10696


namespace intersection_of_sets_eq_l10_10915

noncomputable def set_intersection (M N : Set ℝ): Set ℝ :=
  {x | x ∈ M ∧ x ∈ N}

theorem intersection_of_sets_eq :
  let M := {x : ℝ | -2 < x ∧ x < 2}
  let N := {x : ℝ | x^2 - 2 * x - 3 < 0}
  set_intersection M N = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end intersection_of_sets_eq_l10_10915


namespace absolute_value_inequality_solution_set_l10_10926

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 * x - 1| - |x - 2| < 0} = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end absolute_value_inequality_solution_set_l10_10926


namespace f_neg_eq_f_l10_10473

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_identically_zero :
  ∃ x, f x ≠ 0

axiom functional_equation :
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_neg_eq_f (x : ℝ) : f (-x) = f x := 
sorry

end f_neg_eq_f_l10_10473


namespace marcy_votes_correct_l10_10881

-- Definition of variables based on the conditions
def joey_votes : ℕ := 8
def barry_votes : ℕ := 2 * (joey_votes + 3)
def marcy_votes : ℕ := 3 * barry_votes

-- The main statement to prove
theorem marcy_votes_correct : marcy_votes = 66 := 
by 
  sorry

end marcy_votes_correct_l10_10881


namespace fare_for_90_miles_l10_10521

noncomputable def fare_cost (miles : ℕ) (base_fare cost_per_mile : ℝ) : ℝ :=
  base_fare + cost_per_mile * miles

theorem fare_for_90_miles (base_fare : ℝ) (cost_per_mile : ℝ)
  (h1 : base_fare = 30)
  (h2 : fare_cost 60 base_fare cost_per_mile = 150)
  (h3 : cost_per_mile = (150 - base_fare) / 60) :
  fare_cost 90 base_fare cost_per_mile = 210 :=
  sorry

end fare_for_90_miles_l10_10521


namespace subtract_decimal_numbers_l10_10814

theorem subtract_decimal_numbers : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_decimal_numbers_l10_10814


namespace function_properties_l10_10746

noncomputable def f (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end function_properties_l10_10746


namespace problem_solution_l10_10740

/-- Define the repeating decimal 0.\overline{49} as a rational number. --/
def rep49 := 7 / 9

/-- Define the repeating decimal 0.\overline{4} as a rational number. --/
def rep4 := 4 / 9

/-- The main theorem stating that 99 times the difference between 
    the repeating decimals 0.\overline{49} and 0.\overline{4} equals 5. --/
theorem problem_solution : 99 * (rep49 - rep4) = 5 := by
  sorry

end problem_solution_l10_10740


namespace all_iterated_quadratic_eq_have_integer_roots_l10_10174

noncomputable def initial_quadratic_eq_has_integer_roots (p q : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, x1 + x2 = -p ∧ x1 * x2 = q

noncomputable def iterated_quadratic_eq_has_integer_roots (p q : ℤ) : Prop :=
  ∀ i : ℕ, i ≤ 9 → ∃ x1 x2 : ℤ, x1 + x2 = -(p + i) ∧ x1 * x2 = (q + i)

theorem all_iterated_quadratic_eq_have_integer_roots :
  ∃ p q : ℤ, initial_quadratic_eq_has_integer_roots p q ∧ iterated_quadratic_eq_has_integer_roots p q :=
sorry

end all_iterated_quadratic_eq_have_integer_roots_l10_10174


namespace find_angle_B_l10_10448

theorem find_angle_B (A B C : ℝ) (a b c : ℝ) 
  (h1 : A = 45) 
  (h2 : a = 6) 
  (h3 : b = 3 * Real.sqrt 2)
  (h4 : ∀ A' B' C' : ℝ, 
        ∃ a' b' c' : ℝ, 
        (a' = a) ∧ (b' = b) ∧ (A' = A) ∧ 
        (b' < a') → (B' < A') ∧ (A' = 45)) :
  B = 30 :=
by
  sorry

end find_angle_B_l10_10448


namespace fraction_remain_same_l10_10755

theorem fraction_remain_same (x y : ℝ) : (2 * x + y) / (3 * x + y) = (2 * (10 * x) + (10 * y)) / (3 * (10 * x) + (10 * y)) :=
by sorry

end fraction_remain_same_l10_10755


namespace repeating_decimal_fractional_representation_l10_10952

theorem repeating_decimal_fractional_representation :
  (0.36 : ℝ) = (4 / 11 : ℝ) :=
sorry

end repeating_decimal_fractional_representation_l10_10952


namespace total_pencils_is_220_l10_10177

theorem total_pencils_is_220
  (A : ℕ) (B : ℕ) (P : ℕ) (Q : ℕ)
  (hA : A = 50)
  (h_sum : A + B = 140)
  (h_diff : B - A = P/2)
  (h_pencils : Q = P + 60)
  : P + Q = 220 :=
by
  sorry

end total_pencils_is_220_l10_10177


namespace range_of_m_l10_10224

variable {R : Type} [LinearOrderedField R]

def discriminant (a b c : R) : R := b^2 - 4 * a * c

theorem range_of_m (m : R) :
  (discriminant (1:R) m (m + 3) > 0) ↔ (m < -2 ∨ m > 6) :=
by
  sorry

end range_of_m_l10_10224


namespace sum_of_squares_expr_l10_10787

theorem sum_of_squares_expr : 
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 := 
by
  sorry

end sum_of_squares_expr_l10_10787


namespace prove_statements_l10_10453

theorem prove_statements (x y z : ℝ) (h : x + y + z = x * y * z) :
  ( (∀ (x y : ℝ), x + y = 0 → (∃ (z : ℝ), (x + y + z = x * y * z) → z = 0))
  ∧ (∀ (x y : ℝ), x = 0 → (∃ (z : ℝ), (x + y + z = x * y * z) → y = -z))
  ∧ z = (x + y) / (x * y - 1) ) :=
by
  sorry

end prove_statements_l10_10453


namespace rahim_books_l10_10320

/-- 
Rahim bought some books for Rs. 6500 from one shop and 35 books for Rs. 2000 from another. 
The average price he paid per book is Rs. 85. 
Prove that Rahim bought 65 books from the first shop. 
-/
theorem rahim_books (x : ℕ) 
  (h1 : 6500 + 2000 = 8500) 
  (h2 : 85 * (x + 35) = 8500) : 
  x = 65 := 
sorry

end rahim_books_l10_10320


namespace downstream_distance_l10_10255

theorem downstream_distance
  (time_downstream : ℝ) (time_upstream : ℝ)
  (distance_upstream : ℝ) (speed_still_water : ℝ)
  (h1 : time_downstream = 3) (h2 : time_upstream = 3)
  (h3 : distance_upstream = 15) (h4 : speed_still_water = 10) :
  ∃ d : ℝ, d = 45 :=
by
  sorry

end downstream_distance_l10_10255


namespace notebook_cost_l10_10837

theorem notebook_cost {s n c : ℕ}
  (h1 : s > 18)
  (h2 : c > n)
  (h3 : s * n * c = 2275) :
  c = 13 :=
sorry

end notebook_cost_l10_10837


namespace number_increased_by_one_fourth_l10_10674

theorem number_increased_by_one_fourth (n : ℕ) (h : 25 * 80 / 100 = 20) (h1 : 80 - 20 = 60) :
  n + n / 4 = 60 ↔ n = 48 :=
by
  -- Conditions
  have h2 : 80 - 25 * 80 / 100 = 60 := by linarith [h, h1]
  have h3 : n + n / 4 = 60 := sorry
  -- Assertion (Proof to show is omitted)
  sorry

end number_increased_by_one_fourth_l10_10674


namespace compare_logs_l10_10078

noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.logb 2 3
noncomputable def c : ℝ := Real.logb 5 8

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l10_10078


namespace quadratic_inequality_solution_range_l10_10126

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, a*x^2 + 2*a*x - 4 < 0) ↔ -4 < a ∧ a < 0 := 
by
  sorry

end quadratic_inequality_solution_range_l10_10126


namespace factorization_example_l10_10057

theorem factorization_example : 
  ∀ (a : ℝ), a^2 - 6 * a + 9 = (a - 3)^2 :=
by
  intro a
  sorry

end factorization_example_l10_10057


namespace square_side_measurement_error_l10_10964

theorem square_side_measurement_error {S S' : ℝ} (h1 : S' = S * Real.sqrt 1.0816) :
  ((S' - S) / S) * 100 = 4 := by
  sorry

end square_side_measurement_error_l10_10964


namespace a_leq_neg4_l10_10047

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0
def neg_p (a x : ℝ) : Prop := ¬(p a x)
def neg_q (x : ℝ) : Prop := ¬(q x)

theorem a_leq_neg4 (a : ℝ) (h_neg_p : ∀ x, neg_p a x → neg_q x) (h_a_neg : a < 0) :
  a ≤ -4 :=
sorry

end a_leq_neg4_l10_10047


namespace area_of_triangle_ABC_l10_10946

theorem area_of_triangle_ABC 
  (r : ℝ) (R : ℝ) (ACB : ℝ) 
  (hr : r = 2) 
  (hR : R = 4) 
  (hACB : ACB = 120) : 
  let s := (2 * (2 + 4 * Real.sqrt 3)) / Real.sqrt 3 
  let S := s * r 
  S = 56 / Real.sqrt 3 :=
sorry

end area_of_triangle_ABC_l10_10946


namespace divisibility_by_24_l10_10091

theorem divisibility_by_24 (n : ℤ) : 24 ∣ n * (n + 2) * (5 * n - 1) * (5 * n + 1) :=
sorry

end divisibility_by_24_l10_10091


namespace range_of_k_l10_10121

theorem range_of_k (k : ℝ) : 
  (∃ a b : ℝ, x^2 + ky^2 = 2 ∧ a^2 = 2/k ∧ b^2 = 2 ∧ a > b) → 0 < k ∧ k < 1 :=
by {
  sorry
}

end range_of_k_l10_10121


namespace max_ounces_among_items_l10_10632

theorem max_ounces_among_items
  (budget : ℝ)
  (candy_cost : ℝ)
  (candy_ounces : ℝ)
  (candy_stock : ℕ)
  (chips_cost : ℝ)
  (chips_ounces : ℝ)
  (chips_stock : ℕ)
  : budget = 7 → candy_cost = 1.25 → candy_ounces = 12 →
    candy_stock = 5 → chips_cost = 1.40 → chips_ounces = 17 → chips_stock = 4 →
    max (min ((budget / candy_cost) * candy_ounces) (candy_stock * candy_ounces))
        (min ((budget / chips_cost) * chips_ounces) (chips_stock * chips_ounces)) = 68 := 
by
  intros h_budget h_candy_cost h_candy_ounces h_candy_stock h_chips_cost h_chips_ounces h_chips_stock
  sorry

end max_ounces_among_items_l10_10632


namespace log_travel_time_24_l10_10381

noncomputable def time_for_log_to_travel (D u v : ℝ) (h1 : D / (u + v) = 4) (h2 : D / (u - v) = 6) : ℝ :=
  D / v

theorem log_travel_time_24 (D u v : ℝ) (h1 : D / (u + v) = 4) (h2 : D / (u - v) = 6) :
  time_for_log_to_travel D u v h1 h2 = 24 :=
sorry

end log_travel_time_24_l10_10381


namespace problem_statement_l10_10035

theorem problem_statement : 20 * (256 / 4 + 64 / 16 + 16 / 64 + 2) = 1405 := by
  sorry

end problem_statement_l10_10035


namespace math_problem_l10_10827

-- Condition 1: The solution set of the inequality \(\frac{x-2}{ax+b} > 0\) is \((-1,2)\)
def solution_set_condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, (x > -1 ∧ x < 2) ↔ ((x - 2) * (a * x + b) > 0)

-- Condition 2: \(m\) is the geometric mean of \(a\) and \(b\)
def geometric_mean_condition (a b m : ℝ) : Prop :=
  a * b = m^2

-- The mathematical statement to prove: \(\frac{3m^{2}a}{a^{3}+2b^{3}} = 1\)
theorem math_problem (a b m : ℝ) (h1 : solution_set_condition a b) (h2 : geometric_mean_condition a b m) :
  3 * m^2 * a / (a^3 + 2 * b^3) = 1 :=
sorry

end math_problem_l10_10827


namespace Joan_attended_games_l10_10735

def total_games : ℕ := 864
def games_missed_by_Joan : ℕ := 469
def games_attended_by_Joan : ℕ := total_games - games_missed_by_Joan

theorem Joan_attended_games : games_attended_by_Joan = 395 := 
by 
  -- Proof omitted
  sorry

end Joan_attended_games_l10_10735


namespace yard_fraction_occupied_by_flowerbeds_l10_10961

noncomputable def rectangular_yard_area (length width : ℕ) : ℕ :=
  length * width

noncomputable def triangle_area (leg_length : ℕ) : ℕ :=
  2 * (1 / 2 * leg_length ^ 2)

theorem yard_fraction_occupied_by_flowerbeds :
  let length := 30
  let width := 7
  let parallel_side_short := 20
  let parallel_side_long := 30
  let flowerbed_leg := 7
  rectangular_yard_area length width ≠ 0 ∧
  triangle_area flowerbed_leg * 2 = 49 →
  (triangle_area flowerbed_leg * 2) / rectangular_yard_area length width = 7 / 30 :=
sorry

end yard_fraction_occupied_by_flowerbeds_l10_10961


namespace star_area_l10_10771

-- Conditions
def square_ABCD_area (s : ℝ) := s^2 = 72

-- Question and correct answer
theorem star_area (s : ℝ) (h : square_ABCD_area s) : 24 = 24 :=
by sorry

end star_area_l10_10771


namespace inequality_solution_set_l10_10158

theorem inequality_solution_set (x : ℝ) :
  (x - 3)^2 - 2 * Real.sqrt ((x - 3)^2) - 3 < 0 ↔ 0 < x ∧ x < 6 :=
by
  sorry

end inequality_solution_set_l10_10158


namespace determine_k_values_parallel_lines_l10_10005

theorem determine_k_values_parallel_lines :
  ∀ k : ℝ, ((k - 3) * x + (4 - k) * y + 1 = 0 ∧ 2 * (k - 3) * x - 2 * y + 3 = 0)
  → k = 2 ∨ k = 3 ∨ k = 6 :=
by
  sorry

end determine_k_values_parallel_lines_l10_10005


namespace scientific_notation_l10_10631

theorem scientific_notation (n : ℝ) (h : n = 40.9 * 10^9) : n = 4.09 * 10^10 :=
by sorry

end scientific_notation_l10_10631


namespace determine_f_2048_l10_10265

theorem determine_f_2048 (f : ℕ → ℝ)
  (A1 : ∀ a b n : ℕ, a > 0 → b > 0 → a * b = 2^n → f a + f b = n^2)
  : f 2048 = 121 := by
  sorry

end determine_f_2048_l10_10265


namespace existence_of_f_and_g_l10_10358

noncomputable def Set_n (n : ℕ) : Set ℕ := { x | x ≥ 1 ∧ x ≤ n }

theorem existence_of_f_and_g (n : ℕ) (f g : ℕ → ℕ) :
  (∀ x ∈ Set_n n, (f (g x) = x ∨ g (f x) = x) ∧ ¬(f (g x) = x ∧ g (f x) = x)) ↔ Even n := sorry

end existence_of_f_and_g_l10_10358


namespace simplify_expression_l10_10864

open Real

theorem simplify_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (( (x + 2) ^ 2 * (x ^ 2 - 2 * x + 2) ^ 2 / (x ^ 3 + 8) ^ 2 ) ^ 2 *
   ( (x - 2) ^ 2 * (x ^ 2 + 2 * x + 2) ^ 2 / (x ^ 3 - 8) ^ 2 ) ^ 2 = 1) :=
by
  sorry

end simplify_expression_l10_10864


namespace topsoil_cost_l10_10908

theorem topsoil_cost
  (cost_per_cubic_foot : ℕ)
  (volume_cubic_yards : ℕ)
  (conversion_factor : ℕ)
  (volume_cubic_feet : ℕ := volume_cubic_yards * conversion_factor)
  (total_cost : ℕ := volume_cubic_feet * cost_per_cubic_foot)
  (cost_per_cubic_foot_def : cost_per_cubic_foot = 8)
  (volume_cubic_yards_def : volume_cubic_yards = 8)
  (conversion_factor_def : conversion_factor = 27) :
  total_cost = 1728 := by
  sorry

end topsoil_cost_l10_10908


namespace meet_starting_point_together_at_7_40_AM_l10_10745

-- Definitions of the input conditions
def Charlie_time : Nat := 5
def Alex_time : Nat := 8
def Taylor_time : Nat := 10

-- The combined time when they meet again at the starting point
def LCM_time (a b c : Nat) : Nat := Nat.lcm a (Nat.lcm b c)

-- Proving that the earliest time they all coincide again is 40 minutes after the start
theorem meet_starting_point_together_at_7_40_AM :
  LCM_time Charlie_time Alex_time Taylor_time = 40 := 
by
  unfold Charlie_time Alex_time Taylor_time LCM_time
  sorry

end meet_starting_point_together_at_7_40_AM_l10_10745


namespace f_bounded_l10_10733

noncomputable def f : ℝ → ℝ := sorry

axiom f_property : ∀ x : ℝ, f (3 * x) = 3 * f x - 4 * (f x) ^ 3

axiom f_continuous_at_zero : ContinuousAt f 0

theorem f_bounded : ∀ x : ℝ, |f x| ≤ 1 :=
by
  sorry

end f_bounded_l10_10733


namespace original_quantity_l10_10333

theorem original_quantity (x : ℕ) : 
  (532 * x - 325 * x = 1065430) -> x = 5148 := 
by
  intro h
  sorry

end original_quantity_l10_10333


namespace John_has_15_snakes_l10_10803

theorem John_has_15_snakes (S : ℕ)
  (H1 : ∀ M, M = 2 * S)
  (H2 : ∀ M L, L = M - 5)
  (H3 : ∀ L P, P = L + 8)
  (H4 : ∀ P D, D = P / 3)
  (H5 : S + (2 * S) + ((2 * S) - 5) + (((2 * S) - 5) + 8) + (((((2 * S) - 5) + 8) / 3)) = 114) :
  S = 15 :=
by sorry

end John_has_15_snakes_l10_10803


namespace bus_empty_seats_l10_10738

theorem bus_empty_seats : 
  let initial_seats : ℕ := 23 * 4
  let people_at_start : ℕ := 16
  let first_board : ℕ := 15
  let first_alight : ℕ := 3
  let second_board : ℕ := 17
  let second_alight : ℕ := 10
  let seats_after_init : ℕ := initial_seats - people_at_start
  let seats_after_first : ℕ := seats_after_init - (first_board - first_alight)
  let seats_after_second : ℕ := seats_after_first - (second_board - second_alight)
  seats_after_second = 57 :=
by
  sorry

end bus_empty_seats_l10_10738


namespace tan_neq_sqrt3_sufficient_but_not_necessary_l10_10418

-- Definition of the condition: tan(α) ≠ √3
def condition_tan_neq_sqrt3 (α : ℝ) : Prop := Real.tan α ≠ Real.sqrt 3

-- Definition of the statement: α ≠ π/3
def statement_alpha_neq_pi_div_3 (α : ℝ) : Prop := α ≠ Real.pi / 3

-- The theorem to be proven
theorem tan_neq_sqrt3_sufficient_but_not_necessary {α : ℝ} :
  condition_tan_neq_sqrt3 α → statement_alpha_neq_pi_div_3 α :=
sorry

end tan_neq_sqrt3_sufficient_but_not_necessary_l10_10418


namespace gcd_91_49_l10_10934

theorem gcd_91_49 : Int.gcd 91 49 = 7 := by
  sorry

end gcd_91_49_l10_10934


namespace player_b_wins_l10_10283

theorem player_b_wins : 
  ∃ B_strategy : (ℕ → ℕ → Prop), (∀ A_turn : ℕ → Prop, 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → (A_turn i ↔ ¬ A_turn (i + 1))) → 
  ((B_strategy 1 2019) ∨ ∃ k : ℕ, 1 ≤ k ∧ k ≤ 2019 ∧ B_strategy k (k + 1) ∧ ¬ A_turn k)) :=
sorry

end player_b_wins_l10_10283


namespace factorization_of_expression_l10_10243

-- Define variables
variables {a x y : ℝ}

-- State the problem
theorem factorization_of_expression : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) :=
  sorry

end factorization_of_expression_l10_10243


namespace fraction_power_multiply_l10_10282

theorem fraction_power_multiply :
  ((1 : ℚ) / 3)^4 * ((1 : ℚ) / 5) = (1 / 405 : ℚ) :=
by sorry

end fraction_power_multiply_l10_10282


namespace joan_seashells_correct_l10_10393

/-- Joan originally found 70 seashells -/
def joan_original_seashells : ℕ := 70

/-- Sam gave Joan 27 seashells -/
def seashells_given_by_sam : ℕ := 27

/-- The total number of seashells Joan has now -/
def joan_total_seashells : ℕ := joan_original_seashells + seashells_given_by_sam

theorem joan_seashells_correct : joan_total_seashells = 97 :=
by
  unfold joan_total_seashells
  unfold joan_original_seashells seashells_given_by_sam
  sorry

end joan_seashells_correct_l10_10393


namespace quadratic_root_zero_l10_10495

theorem quadratic_root_zero (k : ℝ) :
    (∃ x : ℝ, x = 0 ∧ (k - 1) * x ^ 2 + 6 * x + k ^ 2 - k = 0) → k = 0 :=
by
  sorry

end quadratic_root_zero_l10_10495


namespace water_evaporation_l10_10596

theorem water_evaporation (m : ℝ) 
  (evaporation_day1 : m' = m * (0.1)) 
  (evaporation_day2 : m'' = (m * 0.9) * 0.1) 
  (total_evaporation : total = m' + m'')
  (water_added : 15 = total) 
  : m = 1500 / 19 := by
  sorry

end water_evaporation_l10_10596


namespace evaluate_expression_l10_10154

theorem evaluate_expression : 150 * (150 - 4) - (150 * 150 - 6 + 2) = -596 :=
by
  sorry

end evaluate_expression_l10_10154


namespace ratio_of_cost_to_marked_price_l10_10413

variable (p : ℝ)

theorem ratio_of_cost_to_marked_price :
  let selling_price := (3/4) * p
  let cost_price := (5/8) * selling_price
  cost_price / p = 15 / 32 :=
by
  let selling_price := (3 / 4) * p
  let cost_price := (5 / 8) * selling_price
  sorry

end ratio_of_cost_to_marked_price_l10_10413


namespace shortest_time_between_ships_l10_10064

theorem shortest_time_between_ships 
  (AB : ℝ) (speed_A : ℝ) (speed_B : ℝ) (angle_ABA' : ℝ) : (AB = 10) → (speed_A = 4) → (speed_B = 6) → (angle_ABA' = 60) →
  ∃ t : ℝ, (t = 150/7 / 60) :=
by
  intro hAB hSpeedA hSpeedB hAngle
  sorry

end shortest_time_between_ships_l10_10064


namespace no_integer_solutions_l10_10003

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^4 + y^2 = 6 * y - 3 :=
by
  sorry

end no_integer_solutions_l10_10003


namespace smallest_value_y_l10_10587

theorem smallest_value_y : ∃ y : ℝ, 3 * y ^ 2 + 33 * y - 90 = y * (y + 18) ∧ (∀ z : ℝ, 3 * z ^ 2 + 33 * z - 90 = z * (z + 18) → y ≤ z) ∧ y = -18 := 
sorry

end smallest_value_y_l10_10587


namespace parallel_line_eq_l10_10826

theorem parallel_line_eq (a b c : ℝ) (p1 p2 : ℝ) :
  (∃ m b1 b2, 3 * a + 6 * b * p1 = 12 ∧ p2 = - (1 / 2) * p1 + b1 ∧
    - (1 / 2) * p1 - m * p1 = b2) → 
    (∃ b', p2 = - (1 / 2) * p1 + b' ∧ b' = 0) := 
sorry

end parallel_line_eq_l10_10826


namespace sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares_l10_10843

theorem sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares 
  (n : ℕ)
  (a b c d : ℕ) 
  (h1 : n = 2^a + 2^b) 
  (h2 : a ≠ b) 
  (h3 : n = (2^c - 1) + (2^d - 1)) 
  (h4 : c ≠ d)
  (h5 : Nat.Prime (2^c - 1)) 
  (h6 : Nat.Prime (2^d - 1)) : 
  ∃ x y : ℕ, x ≠ y ∧ n = x^2 + y^2 := 
by
  sorry

end sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares_l10_10843


namespace num_real_roots_of_abs_x_eq_l10_10713

theorem num_real_roots_of_abs_x_eq (k : ℝ) (hk : 6 < k ∧ k < 7) 
  : (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (|x1| * x1 - 2 * x1 + 7 - k = 0) ∧ 
    (|x2| * x2 - 2 * x2 + 7 - k = 0) ∧
    (|x3| * x3 - 2 * x3 + 7 - k = 0)) ∧
  (¬ ∃ x4 : ℝ, x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3 ∧ |x4| * x4 - 2 * x4 + 7 - k = 0) :=
sorry

end num_real_roots_of_abs_x_eq_l10_10713


namespace find_wrongly_written_height_l10_10248

def wrongly_written_height
  (n : ℕ)
  (avg_height_incorrect : ℝ)
  (actual_height : ℝ)
  (avg_height_correct : ℝ) : ℝ :=
  let total_height_incorrect := n * avg_height_incorrect
  let total_height_correct := n * avg_height_correct
  let height_difference := total_height_incorrect - total_height_correct
  actual_height + height_difference

theorem find_wrongly_written_height :
  wrongly_written_height 35 182 106 180 = 176 :=
by
  sorry

end find_wrongly_written_height_l10_10248


namespace determine_c_plus_d_l10_10731

theorem determine_c_plus_d (x : ℝ) (c d : ℤ) (h1 : x^2 + 5*x + (5/x) + (1/(x^2)) = 35) (h2 : x = c + Real.sqrt d) : c + d = 5 :=
sorry

end determine_c_plus_d_l10_10731


namespace length_AB_proof_l10_10579

noncomputable def length_AB (AB BC CA : ℝ) (DEF DE EF DF : ℝ) (angle_BAC angle_DEF : ℝ) : ℝ :=
  if h : (angle_BAC = 120 ∧ angle_DEF = 120 ∧ AB = 5 ∧ BC = 17 ∧ CA = 12 ∧ DE = 9 ∧ EF = 15 ∧ DF = 12) then
    (5 * 15) / 17
  else
    0

theorem length_AB_proof : length_AB 5 17 12 9 15 12 120 120 = 75 / 17 := by
  sorry

end length_AB_proof_l10_10579


namespace arithmetic_geometric_sequence_l10_10073

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_d : d ≠ 0)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_S : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d)
  (h_geo : (a 1 + 2 * d) ^ 2 = a 1 * (a 1 + 3 * d)) :
  (S 4 - S 2) / (S 5 - S 3) = 3 :=
by
  sorry

end arithmetic_geometric_sequence_l10_10073


namespace fraction_ratio_l10_10507

theorem fraction_ratio :
  ∃ (x y : ℕ), y ≠ 0 ∧ (x:ℝ) / (y:ℝ) = 240 / 1547 ∧ ((x:ℝ) / (y:ℝ)) / (2 / 13) = (5 / 34) / (7 / 48) :=
sorry

end fraction_ratio_l10_10507


namespace sheets_borrowed_l10_10655

-- Definitions based on conditions
def total_pages : ℕ := 60  -- Hiram's algebra notes are 60 pages
def total_sheets : ℕ := 30  -- printed on 30 sheets of paper
def average_remaining : ℕ := 23  -- the average of the page numbers on all remaining sheets is 23

-- Let S_total be the sum of all page numbers initially
def S_total := (total_pages * (1 + total_pages)) / 2

-- Let c be the number of consecutive sheets borrowed
-- Let b be the number of sheets before the borrowed sheets
-- Calculate S_borrowed based on problem conditions
def S_borrowed (c b : ℕ) := 2 * c * (b + c) + c

-- Calculate the remaining sum and corresponding mean
def remaining_sum (c b : ℕ) := S_total - S_borrowed c b
def remaining_mean (c : ℕ) := (total_sheets * 2 - 2 * c)

-- The theorem we want to prove
theorem sheets_borrowed (c : ℕ) (h : 1830 - S_borrowed c 10 = 23 * (60 - 2 * c)) : c = 15 :=
  sorry

end sheets_borrowed_l10_10655


namespace line_equation_l10_10238

theorem line_equation (l : ℝ → ℝ → Prop) (a b : ℝ) 
  (h1 : ∀ x y, l x y ↔ y = - (b / a) * x + b) 
  (h2 : l 2 1) 
  (h3 : a + b = 0) : 
  l x y ↔ y = x - 1 ∨ y = x / 2 := 
by
  sorry

end line_equation_l10_10238


namespace tan_alpha_plus_pi_div_four_l10_10233

theorem tan_alpha_plus_pi_div_four
  (α : ℝ)
  (a : ℝ × ℝ := (3, 4))
  (b : ℝ × ℝ := (Real.sin α, Real.cos α))
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) :
  Real.tan (α + Real.pi / 4) = 7 := by
  sorry

end tan_alpha_plus_pi_div_four_l10_10233


namespace symmetry_condition_l10_10042

-- Define grid and initial conditions
def grid : Type := ℕ × ℕ
def is_colored (pos : grid) : Prop := 
  pos = (1,4) ∨ pos = (2,1) ∨ pos = (4,2)

-- Conditions for symmetry: horizontal and vertical line symmetry and 180-degree rotational symmetry
def is_symmetric_line (grid_size : grid) (pos : grid) : Prop :=
  (pos.1 <= grid_size.1 / 2 ∧ pos.2 <= grid_size.2 / 2) ∨ 
  (pos.1 > grid_size.1 / 2 ∧ pos.2 <= grid_size.2 / 2) ∨
  (pos.1 <= grid_size.1 / 2 ∧ pos.2 > grid_size.2 / 2) ∨
  (pos.1 > grid_size.1 / 2 ∧ pos.2 > grid_size.2 / 2)

def grid_size : grid := (4, 5)
def add_squares_needed (num : ℕ) : Prop :=
  ∀ (pos : grid), is_symmetric_line grid_size pos → is_colored pos

theorem symmetry_condition : 
  ∃ n, add_squares_needed n ∧ n = 9
  := sorry

end symmetry_condition_l10_10042


namespace multiplication_correct_l10_10933

theorem multiplication_correct :
  72514 * 99999 = 7250675486 :=
by
  sorry

end multiplication_correct_l10_10933


namespace product_of_solutions_l10_10719

theorem product_of_solutions :
  (∀ y : ℝ, (|y| = 2 * (|y| - 1)) → y = 2 ∨ y = -2) →
  (∀ y1 y2 : ℝ, (y1 = 2 ∧ y2 = -2) → y1 * y2 = -4) :=
by
  intro h
  have h1 := h 2
  have h2 := h (-2)
  sorry

end product_of_solutions_l10_10719


namespace initial_weight_cucumbers_l10_10069

theorem initial_weight_cucumbers (W : ℝ) (h1 : 0.99 * W + 0.01 * W = W) 
                                  (h2 : W = (50 - 0.98 * 50 + 0.01 * W))
                                  (h3 : 50 > 0) : W = 100 := 
sorry

end initial_weight_cucumbers_l10_10069


namespace gcd_seven_factorial_ten_fact_div_5_fact_l10_10367

def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define 7!
def seven_factorial := factorial 7

-- Define 10! / 5!
def ten_fact_div_5_fact := factorial 10 / factorial 5

-- Prove that the GCD of 7! and (10! / 5!) is 2520
theorem gcd_seven_factorial_ten_fact_div_5_fact :
  Nat.gcd seven_factorial ten_fact_div_5_fact = 2520 := by
sorry

end gcd_seven_factorial_ten_fact_div_5_fact_l10_10367


namespace no_integer_solutions_l10_10341

theorem no_integer_solutions (m n : ℤ) (h1 : m ^ 3 + n ^ 4 + 130 * m * n = 42875) (h2 : m * n ≥ 0) :
  false :=
sorry

end no_integer_solutions_l10_10341


namespace OBrien_current_hats_l10_10554

-- Definition of the number of hats that Fire chief Simpson has
def Simpson_hats : ℕ := 15

-- Definition of the number of hats that Policeman O'Brien had before losing one
def OBrien_initial_hats (Simpson_hats : ℕ) : ℕ := 2 * Simpson_hats + 5

-- Final proof statement that Policeman O'Brien now has 34 hats
theorem OBrien_current_hats : OBrien_initial_hats Simpson_hats - 1 = 34 := by
  -- Proof will go here, but is skipped for now
  sorry

end OBrien_current_hats_l10_10554


namespace circulation_ratio_l10_10441

variable (A : ℕ) -- Assuming A to be a natural number for simplicity

theorem circulation_ratio (h : ∀ t : ℕ, t = 1971 → t = 4 * A) : 4 / 13 = 4 / 13 := 
by
  sorry

end circulation_ratio_l10_10441


namespace greatest_odd_factors_under_150_l10_10729

theorem greatest_odd_factors_under_150 : ∃ (n : ℕ), n < 150 ∧ ( ∃ (k : ℕ), n = k * k ) ∧ (∀ m : ℕ, m < 150 ∧ ( ∃ (k : ℕ), m = k * k ) → m ≤ 144) :=
by
  sorry

end greatest_odd_factors_under_150_l10_10729


namespace original_prices_sum_l10_10743

theorem original_prices_sum
  (new_price_candy_box : ℝ)
  (new_price_soda_can : ℝ)
  (increase_candy_box : ℝ)
  (increase_soda_can : ℝ)
  (h1 : new_price_candy_box = 10)
  (h2 : new_price_soda_can = 9)
  (h3 : increase_candy_box = 0.25)
  (h4 : increase_soda_can = 0.50) :
  let original_price_candy_box := new_price_candy_box / (1 + increase_candy_box)
  let original_price_soda_can := new_price_soda_can / (1 + increase_soda_can)
  original_price_candy_box + original_price_soda_can = 19 :=
by
  sorry

end original_prices_sum_l10_10743


namespace no_prime_solution_in_2_to_7_l10_10026

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_solution_in_2_to_7 : ∀ p : ℕ, is_prime p ∧ 2 ≤ p ∧ p ≤ 7 → (2 * p^3 - p^2 - 15 * p + 22) ≠ 0 :=
by
  intros p hp
  have h := hp.left
  sorry

end no_prime_solution_in_2_to_7_l10_10026


namespace determine_value_of_e_l10_10273

theorem determine_value_of_e {a b c d e : ℝ} (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) 
    (h5 : a + b = 32) (h6 : a + c = 36) (h7 : b + c = 37 ∨ a + d = 37) 
    (h8 : c + e = 48) (h9 : d + e = 51) : e = 27.5 :=
sorry

end determine_value_of_e_l10_10273


namespace inequality_abc_sum_one_l10_10699

theorem inequality_abc_sum_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 1) :
  (a^2 + b^2 + c^2 + d) / (a + b + c)^3 +
  (b^2 + c^2 + d^2 + a) / (b + c + d)^3 +
  (c^2 + d^2 + a^2 + b) / (c + d + a)^3 +
  (d^2 + a^2 + b^2 + c) / (d + a + b)^3 > 4 := by
  sorry

end inequality_abc_sum_one_l10_10699


namespace probability_incorrect_pairs_l10_10852

theorem probability_incorrect_pairs 
  (k : ℕ) (h_k : k < 6)
  : let m := 7
    let n := 72
    m + n = 79 :=
by
  sorry

end probability_incorrect_pairs_l10_10852


namespace bridge_length_l10_10950

theorem bridge_length (length_of_train : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℕ) : 
  length_of_train = 110 → train_speed_kmph = 45 → time_seconds = 30 → 
  ∃ length_of_bridge : ℕ, length_of_bridge = 265 := by
  intros h1 h2 h3
  sorry

end bridge_length_l10_10950


namespace rectangle_length_l10_10114

theorem rectangle_length (side_length_square : ℝ) (width_rectangle : ℝ) (area_equal : ℝ) 
  (square_area : side_length_square * side_length_square = area_equal) 
  (rectangle_area : width_rectangle * (width_rectangle * length) = area_equal) : 
  length = 24 :=
by 
  sorry

end rectangle_length_l10_10114


namespace employed_females_percentage_l10_10649

theorem employed_females_percentage (total_population_percent employed_population_percent employed_males_percent : ℝ) :
  employed_population_percent = 70 → employed_males_percent = 21 →
  (employed_population_percent - employed_males_percent) / employed_population_percent * 100 = 70 :=
by
  -- Assume the total population percentage is 100%, which allows us to work directly with percentages.
  let employed_population_percent := 70
  let employed_males_percent := 21
  sorry

end employed_females_percentage_l10_10649


namespace cubic_roots_identity_l10_10397

theorem cubic_roots_identity (p q r : ℝ) 
  (h1 : p + q + r = 0) 
  (h2 : p * q + q * r + r * p = -3) 
  (h3 : p * q * r = -2) : 
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = 0 := 
by
  sorry

end cubic_roots_identity_l10_10397


namespace expression_equal_a_five_l10_10818

noncomputable def a : ℕ := sorry

theorem expression_equal_a_five (a : ℕ) : (a^4 * a) = a^5 := by
  sorry

end expression_equal_a_five_l10_10818


namespace molecular_weight_compound_l10_10104

-- Definitions of atomic weights
def atomic_weight_Cu : ℝ := 63.546
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

-- Definitions of the number of atoms in the compound
def num_Cu : ℝ := 1
def num_C : ℝ := 1
def num_O : ℝ := 3

-- The molecular weight of the compound
def molecular_weight : ℝ := (num_Cu * atomic_weight_Cu) + (num_C * atomic_weight_C) + (num_O * atomic_weight_O)

-- Statement to prove
theorem molecular_weight_compound : molecular_weight = 123.554 := by
  sorry

end molecular_weight_compound_l10_10104


namespace correct_calculation_l10_10949

theorem correct_calculation : 
  ¬(2 * Real.sqrt 3 + 3 * Real.sqrt 2 = 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  ¬(5 * Real.sqrt 3 * 5 * Real.sqrt 2 = 5 * Real.sqrt 6) ∧
  ¬(Real.sqrt (4 + 1 / 2) = 2 * Real.sqrt (1 / 2)) :=
by {
  -- Using the conditions to prove the correct option B
  sorry
}

end correct_calculation_l10_10949


namespace interior_triangle_area_l10_10368

theorem interior_triangle_area (s1 s2 s3 : ℝ) (hs1 : s1 = 15) (hs2 : s2 = 6) (hs3 : s3 = 15) 
  (a1 a2 a3 : ℝ) (ha1 : a1 = 225) (ha2 : a2 = 36) (ha3 : a3 = 225) 
  (h1 : s1 * s1 = a1) (h2 : s2 * s2 = a2) (h3 : s3 * s3 = a3) :
  (1/2) * s1 * s2 = 45 :=
by
  sorry

end interior_triangle_area_l10_10368


namespace cubic_intersection_unique_point_l10_10518

-- Define the cubic functions f and g
def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d
def g (a b c d x : ℝ) : ℝ := -a * x^3 + b * x^2 - c * x + d

-- Translate conditions into Lean conditions
variables (a b c d : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- Lean statement to prove the intersection point
theorem cubic_intersection_unique_point :
  ∀ x y : ℝ, (f a b c d x = y) ↔ (g a b c d x = y) → (x = 0 ∧ y = d) :=
by
  -- Mathematical steps would go here (omitted with sorry)
  sorry

end cubic_intersection_unique_point_l10_10518


namespace melon_weights_l10_10798

-- We start by defining the weights of the individual melons.
variables {D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 : ℝ}

-- Define the weights of the given sets of three melons.
def W1 := D1 + D2 + D3
def W2 := D2 + D3 + D4
def W3 := D1 + D3 + D4
def W4 := D1 + D2 + D4
def W5 := D5 + D6 + D7
def W6 := D8 + D9 + D10

-- State the theorem to be proven.
theorem melon_weights (W1 W2 W3 W4 W5 W6 : ℝ) :
  (W1 + W2 + W3 + W4) / 3 + W5 + W6 = D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 :=
sorry 

end melon_weights_l10_10798


namespace tile_difference_is_11_l10_10222

-- Define the initial number of blue and green tiles
def initial_blue_tiles : ℕ := 13
def initial_green_tiles : ℕ := 6

-- Define the number of additional green tiles added as border
def additional_green_tiles : ℕ := 18

-- Define the total number of green tiles in the new figure
def total_green_tiles : ℕ := initial_green_tiles + additional_green_tiles

-- Define the total number of blue tiles in the new figure (remains the same)
def total_blue_tiles : ℕ := initial_blue_tiles

-- Define the difference between the total number of green tiles and blue tiles
def tile_difference : ℕ := total_green_tiles - total_blue_tiles

-- The theorem stating that the difference between the total number of green tiles 
-- and the total number of blue tiles in the new figure is 11
theorem tile_difference_is_11 : tile_difference = 11 := by
  sorry

end tile_difference_is_11_l10_10222


namespace ellipse_has_correct_equation_l10_10051

noncomputable def ellipse_Equation (a b : ℝ) (eccentricity : ℝ) (triangle_perimeter : ℝ) : Prop :=
  let c := a * eccentricity
  (a > b) ∧ (b > 0) ∧ (eccentricity = (Real.sqrt 3) / 3) ∧ (triangle_perimeter = 4 * (Real.sqrt 3)) ∧
  (a = Real.sqrt 3) ∧ (b^2 = a^2 - c^2) ∧
  (c = 1) ∧
  (b = Real.sqrt 2) ∧
  (∀ x y : ℝ, ((x^2 / a^2) + (y^2 / b^2) = 1) ↔ ((x^2 / 3) + (y^2 / 2) = 1))

theorem ellipse_has_correct_equation : ellipse_Equation (Real.sqrt 3) (Real.sqrt 2) ((Real.sqrt 3) / 3) (4 * (Real.sqrt 3)) := 
sorry

end ellipse_has_correct_equation_l10_10051


namespace first_player_wins_l10_10442

theorem first_player_wins :
  ∀ (sticks : ℕ), (sticks = 1) →
  (∀ (break_rule : ℕ → ℕ → Prop),
  (∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z → break_rule x y → break_rule x z)
  → (∃ n : ℕ, n % 3 = 0 ∧ break_rule n (n + 1) → ∃ t₁ t₂ t₃ : ℕ, t₁ = t₂ ∧ t₂ = t₃ ∧ t₁ + t₂ + t₃ = n))
  → (∃ w : ℕ, w = 1) := sorry

end first_player_wins_l10_10442


namespace sum_of_x_and_y_l10_10009

theorem sum_of_x_and_y 
  (x y : ℤ)
  (h1 : x - y = 36) 
  (h2 : x = 28) : 
  x + y = 20 :=
by 
  sorry

end sum_of_x_and_y_l10_10009


namespace vertex_of_parabola_l10_10101

theorem vertex_of_parabola (c d : ℝ) :
  (∀ x, -2 * x^2 + c * x + d ≤ 0 ↔ x ≥ -7 / 2) →
  ∃ k, k = (-7 / 2 : ℝ) ∧ y = -2 * (x + 7 / 2)^2 + 0 := 
sorry

end vertex_of_parabola_l10_10101


namespace hyperbola_focal_length_range_l10_10517

theorem hyperbola_focal_length_range (m : ℝ) (h1 : m > 0)
    (h2 : ∀ x y, x^2 - y^2 / m^2 ≠ 1 → y ≠ m * x ∧ y ≠ -m * x)
    (h3 : ∀ x y, x^2 + (y + 2)^2 = 1 → x^2 + y^2 / m^2 ≠ 1) :
    ∃ c : ℝ, 2 < 2 * Real.sqrt (1 + m^2) ∧ 2 * Real.sqrt (1 + m^2) < 4 :=
by
  sorry

end hyperbola_focal_length_range_l10_10517


namespace smallest_angle_convex_15_polygon_l10_10867

theorem smallest_angle_convex_15_polygon :
  ∃ (a : ℕ) (d : ℕ), (∀ n : ℕ, n ∈ Finset.range 15 → (a + n * d < 180)) ∧
  15 * (a + 7 * d) = 2340 ∧ 15 * d <= 24 -> a = 135 :=
by
  -- Proof omitted
  sorry

end smallest_angle_convex_15_polygon_l10_10867


namespace last_two_digits_2007_pow_20077_l10_10570

theorem last_two_digits_2007_pow_20077 : (2007 ^ 20077) % 100 = 7 := 
by sorry

end last_two_digits_2007_pow_20077_l10_10570


namespace find_x_l10_10468

-- Define the condition variables
variables (y z x : ℝ) (Y Z X : ℝ)
-- Primary conditions given in the problem
variable (h_y : y = 7)
variable (h_z : z = 6)
variable (h_cosYZ : Real.cos (Y - Z) = 15 / 16)

-- The main theorem to prove
theorem find_x (h_y : y = 7) (h_z : z = 6) (h_cosYZ : Real.cos (Y - Z) = 15 / 16) :
  x = Real.sqrt 22 :=
sorry

end find_x_l10_10468


namespace pages_same_units_digit_l10_10039

theorem pages_same_units_digit (n : ℕ) (H : n = 63) : 
  ∃ (count : ℕ), count = 13 ∧ ∀ x : ℕ, (1 ≤ x ∧ x ≤ n) → 
  (((x % 10) = ((n + 1 - x) % 10)) → (x = 2 ∨ x = 7 ∨ x = 12 ∨ x = 17 ∨ x = 22 ∨ x = 27 ∨ x = 32 ∨ x = 37 ∨ x = 42 ∨ x = 47 ∨ x = 52 ∨ x = 57 ∨ x = 62)) :=
by
  sorry

end pages_same_units_digit_l10_10039


namespace find_a1_l10_10497

variable {q a1 a2 a3 a4 : ℝ}
variable (S : ℕ → ℝ)

axiom common_ratio_pos : q > 0
axiom S2_eq : S 2 = 3 * a2 + 2
axiom S4_eq : S 4 = 3 * a4 + 2

theorem find_a1 (h1 : S 2 = 3 * a2 + 2) (h2 : S 4 = 3 * a4 + 2) (common_ratio_pos : q > 0) : a1 = -1 :=
sorry

end find_a1_l10_10497


namespace inequality_with_equality_condition_l10_10888

variables {a b c d : ℝ}

theorem inequality_with_equality_condition (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
    (habcd : a + b + c + d = 1) : 
    (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) >= 1 / 2) ∧ 
    (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) = 1 / 2 ↔ a = b ∧ b = c ∧ c = d) := 
sorry

end inequality_with_equality_condition_l10_10888


namespace range_of_a_l10_10148
noncomputable def exponential_quadratic (a : ℝ) : Prop :=
  ∃ x : ℝ, 0 < x ∧ (1/4)^x + (1/2)^(x-1) + a = 0

theorem range_of_a (a : ℝ) : exponential_quadratic a ↔ -3 < a ∧ a < 0 :=
sorry

end range_of_a_l10_10148


namespace lcm_9_12_15_l10_10924

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l10_10924


namespace required_speed_l10_10319

-- The car covers 504 km in 6 hours initially.
def distance : ℕ := 504
def initial_time : ℕ := 6
def initial_speed : ℕ := distance / initial_time

-- The time that is 3/2 times the initial time.
def factor : ℚ := 3 / 2
def new_time : ℚ := initial_time * factor

-- The speed required to cover the same distance in the new time.
def new_speed : ℚ := distance / new_time

-- The proof statement
theorem required_speed : new_speed = 56 := by
  sorry

end required_speed_l10_10319


namespace trajectory_equation_l10_10335

noncomputable def A : ℝ × ℝ := (0, -1)
noncomputable def B (x_b : ℝ) : ℝ × ℝ := (x_b, -3)
noncomputable def M (x y : ℝ) : ℝ × ℝ := (x, y)

-- Conditions as definitions in Lean 4
def MB_parallel_OA (x y x_b : ℝ) : Prop :=
  ∃ k : ℝ, (x_b - x) = k * 0 ∧ (-3 - y) = k * (-1)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def condition (x y x_b : ℝ) : Prop :=
  let MA := (0 - x, -1 - y)
  let AB := (x_b - 0, -3 - (-1))
  let MB := (x_b - x, -3 - y)
  let BA := (-x_b, 2)

  dot_product MA AB = dot_product MB BA

theorem trajectory_equation : ∀ x y, (∀ x_b, MB_parallel_OA x y x_b) → condition x y x_b → y = (1 / 4) * x^2 - 2 :=
by
  intros
  sorry

end trajectory_equation_l10_10335


namespace greatest_four_digit_number_l10_10372

theorem greatest_four_digit_number (x : ℕ) :
  x ≡ 1 [MOD 7] ∧ x ≡ 5 [MOD 8] ∧ 1000 ≤ x ∧ x < 10000 → x = 9997 :=
by
  sorry

end greatest_four_digit_number_l10_10372


namespace speed_excluding_stoppages_l10_10589

-- Conditions
def speed_with_stoppages := 33 -- kmph
def stoppage_time_per_hour := 16 -- minutes

-- Conversion of conditions to statements
def running_time_per_hour := 60 - stoppage_time_per_hour -- minutes
def running_time_in_hours := running_time_per_hour / 60 -- hours

-- Proof Statement
theorem speed_excluding_stoppages : 
  (speed_with_stoppages = 33) → (stoppage_time_per_hour = 16) → (75 = 33 / (44 / 60)) :=
by
  intros h1 h2
  sorry

end speed_excluding_stoppages_l10_10589


namespace inequality_proof_l10_10508

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = a * b) : 
  (a / (b^2 + 4) + b / (a^2 + 4) >= 1 / 2) := 
  sorry

end inequality_proof_l10_10508


namespace warmup_puzzle_time_l10_10306

theorem warmup_puzzle_time (W : ℕ) (H : W + 3 * W + 3 * W = 70) : W = 10 :=
by
  sorry

end warmup_puzzle_time_l10_10306


namespace total_food_correct_l10_10830

def max_food_per_guest : ℕ := 2
def min_guests : ℕ := 162
def total_food_cons : ℕ := min_guests * max_food_per_guest

theorem total_food_correct : total_food_cons = 324 := by
  sorry

end total_food_correct_l10_10830


namespace average_six_conseq_ints_l10_10805

theorem average_six_conseq_ints (c d : ℝ) (h₁ : d = c + 2.5) :
  (d - 2 + d - 1 + d + d + 1 + d + 2 + d + 3) / 6 = c + 3 :=
by
  sorry

end average_six_conseq_ints_l10_10805


namespace circle_area_from_circumference_l10_10213

theorem circle_area_from_circumference (r : ℝ) (π : ℝ) (h1 : 2 * π * r = 36) : (π * (r^2) = 324 / π) := by
  sorry

end circle_area_from_circumference_l10_10213


namespace volume_larger_of_cube_cut_plane_l10_10613

/-- Define the vertices and the midpoints -/
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def R : Point := ⟨0, 0, 0⟩
def X : Point := ⟨1, 2, 0⟩
def Y : Point := ⟨2, 0, 1⟩

/-- Equation of the plane passing through R, X and Y -/
def plane_eq (p : Point) : Prop :=
p.x - 2 * p.y - 2 * p.z = 0

/-- The volume of the larger of the two solids formed by cutting the cube with the plane -/
noncomputable def volume_larger_solid : ℝ :=
8 - (4/3 - (1/6))

/-- The statement for the given math problem -/
theorem volume_larger_of_cube_cut_plane :
  volume_larger_solid = 41/6 :=
by
  sorry

end volume_larger_of_cube_cut_plane_l10_10613


namespace sequence_formula_l10_10584

-- Define the problem when n >= 2
theorem sequence_formula (n : ℕ) (h : n ≥ 2) : 
  1 / (n^2 - 1) = (1 / 2) * (1 / (n - 1) - 1 / (n + 1)) := 
by {
  sorry
}

end sequence_formula_l10_10584


namespace xiao_ming_equation_l10_10928

-- Defining the parameters of the problem
def distance : ℝ := 2000
def regular_time (x : ℝ) := x
def increased_speed := 5
def time_saved := 2

-- Problem statement to be proven in Lean 4:
theorem xiao_ming_equation (x : ℝ) (h₁ : x > 2) : 
  (distance / (x - time_saved)) - (distance / regular_time x) = increased_speed :=
by
  sorry

end xiao_ming_equation_l10_10928


namespace correct_relation_l10_10590

open Set

def U : Set ℝ := univ

def A : Set ℝ := { x | x^2 < 4 }

def B : Set ℝ := { x | x > 2 }

def comp_of_B : Set ℝ := U \ B

theorem correct_relation : A ∩ comp_of_B = A := by
  sorry

end correct_relation_l10_10590


namespace jane_paid_cashier_l10_10371

-- Define the conditions in Lean
def skirts_bought : ℕ := 2
def price_per_skirt : ℕ := 13
def blouses_bought : ℕ := 3
def price_per_blouse : ℕ := 6
def change_received : ℤ := 56

-- Calculate the total cost in Lean
def cost_of_skirts : ℕ := skirts_bought * price_per_skirt
def cost_of_blouses : ℕ := blouses_bought * price_per_blouse
def total_cost : ℕ := cost_of_skirts + cost_of_blouses
def amount_paid : ℤ := total_cost + change_received

-- Lean statement to prove the question
theorem jane_paid_cashier :
  amount_paid = 100 :=
by
  sorry

end jane_paid_cashier_l10_10371


namespace grocer_initial_stock_l10_10388

noncomputable def initial_coffee_stock (x : ℝ) : Prop :=
  let initial_decaf := 0.20 * x
  let additional_coffee := 100
  let additional_decaf := 0.50 * additional_coffee
  let total_coffee := x + additional_coffee
  let total_decaf := initial_decaf + additional_decaf
  0.26 * total_coffee = total_decaf

theorem grocer_initial_stock :
  ∃ x : ℝ, initial_coffee_stock x ∧ x = 400 :=
by
  sorry

end grocer_initial_stock_l10_10388


namespace inverse_variation_l10_10082

variable (a b : ℝ)

theorem inverse_variation (h_ab : a * b = 400) :
  (b = 0.25 ∧ a = 1600) ∨ (b = 1.0 ∧ a = 400) :=
  sorry

end inverse_variation_l10_10082


namespace population_ratio_l10_10891

-- Definitions
def population_z (Z : ℕ) : ℕ := Z
def population_y (Z : ℕ) : ℕ := 2 * population_z Z
def population_x (Z : ℕ) : ℕ := 6 * population_y Z

-- Theorem stating the ratio
theorem population_ratio (Z : ℕ) : (population_x Z) / (population_z Z) = 12 :=
  by 
  unfold population_x population_y population_z
  sorry

end population_ratio_l10_10891


namespace watch_correction_l10_10549

def watch_loss_per_day : ℚ := 13 / 4

def hours_from_march_15_noon_to_march_22_9am : ℚ := 7 * 24 + 21

def per_hour_loss : ℚ := watch_loss_per_day / 24

def total_loss_in_minutes : ℚ := hours_from_march_15_noon_to_march_22_9am * per_hour_loss

theorem watch_correction :
  total_loss_in_minutes = 2457 / 96 :=
by
  sorry

end watch_correction_l10_10549


namespace value_of_x_plus_y_l10_10020

theorem value_of_x_plus_y (x y : ℤ) (h1 : x - y = 36) (h2 : x = 20) : x + y = 4 :=
by
  sorry

end value_of_x_plus_y_l10_10020


namespace frac_m_over_q_l10_10403

variable (m n p q : ℚ)

theorem frac_m_over_q (h1 : m / n = 10) (h2 : p / n = 2) (h3 : p / q = 1 / 5) : m / q = 1 :=
by
  sorry

end frac_m_over_q_l10_10403


namespace total_number_of_people_l10_10293

def total_people_at_park(hikers bike_riders : Nat) : Nat :=
  hikers + bike_riders

theorem total_number_of_people 
  (bike_riders : Nat)
  (hikers : Nat)
  (hikers_eq_bikes_plus_178 : hikers = bike_riders + 178)
  (bikes_eq_249 : bike_riders = 249) :
  total_people_at_park hikers bike_riders = 676 :=
by
  sorry

end total_number_of_people_l10_10293


namespace who_next_to_boris_l10_10347

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l10_10347


namespace revenue_decrease_percent_l10_10354

theorem revenue_decrease_percent (T C : ℝ) (hT_pos : T > 0) (hC_pos : C > 0) :
  let new_T := 0.75 * T
  let new_C := 1.10 * C
  let original_revenue := T * C
  let new_revenue := new_T * new_C
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 17.5 := 
by {
  sorry
}

end revenue_decrease_percent_l10_10354


namespace solve_for_m_l10_10275

theorem solve_for_m (m : ℝ) : 
  (∀ x : ℝ, (x = 2) → ((m - 2) * x = 5 * (x + 1))) → (m = 19 / 2) :=
by
  intro h
  have h1 := h 2
  sorry  -- proof can be filled in later

end solve_for_m_l10_10275


namespace thomas_probability_of_two_pairs_l10_10982

def number_of_ways_to_choose_five_socks := Nat.choose 12 5
def number_of_ways_to_choose_two_pairs_of_colors := Nat.choose 4 2
def number_of_ways_to_choose_one_color_for_single_sock := Nat.choose 2 1
def number_of_ways_to_choose_two_socks_from_three := Nat.choose 3 2
def number_of_ways_to_choose_one_sock_from_three := Nat.choose 3 1

theorem thomas_probability_of_two_pairs : 
  number_of_ways_to_choose_five_socks = 792 →
  number_of_ways_to_choose_two_pairs_of_colors = 6 →
  number_of_ways_to_choose_one_color_for_single_sock = 2 →
  number_of_ways_to_choose_two_socks_from_three = 3 →
  number_of_ways_to_choose_one_sock_from_three = 3 →
  6 * 2 * 3 * 3 * 3 = 324 →
  (324 : ℚ) / 792 = 9 / 22 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end thomas_probability_of_two_pairs_l10_10982


namespace distinct_real_roots_l10_10756

theorem distinct_real_roots (p : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2 * |x1| - p = 0) ∧ (x2^2 - 2 * |x2| - p = 0)) → p > -1 :=
by
  intro h
  sorry

end distinct_real_roots_l10_10756


namespace necessary_but_not_sufficient_l10_10958

-- Define that for all x in ℝ, x^2 - 4x + 2m ≥ 0
def proposition_p (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 - 4 * x + 2 * m ≥ 0

-- Main theorem statement
theorem necessary_but_not_sufficient (m : ℝ) : 
  (proposition_p m → m ≥ 2) → (m ≥ 1 → m ≥ 2) :=
by
  intros h1 h2
  sorry

end necessary_but_not_sufficient_l10_10958


namespace find_AC_l10_10088

theorem find_AC (AB DC AD : ℕ) (hAB : AB = 13) (hDC : DC = 20) (hAD : AD = 5) : 
  AC = 24.2 := 
sorry

end find_AC_l10_10088


namespace problem_1_problem_2_l10_10876

theorem problem_1 :
  83 * 87 = 100 * 8 * (8 + 1) + 21 :=
by sorry

theorem problem_2 (n : ℕ) :
  (10 * n + 3) * (10 * n + 7) = 100 * n * (n + 1) + 21 :=
by sorry

end problem_1_problem_2_l10_10876


namespace seats_needed_on_bus_l10_10530

variable (f t tr dr c h : ℕ)

def flute_players := 5
def trumpet_players := 3 * flute_players
def trombone_players := trumpet_players - 8
def drummers := trombone_players + 11
def clarinet_players := 2 * flute_players
def french_horn_players := trombone_players + 3

theorem seats_needed_on_bus :
  f = 5 →
  t = 3 * f →
  tr = t - 8 →
  dr = tr + 11 →
  c = 2 * f →
  h = tr + 3 →
  f + t + tr + dr + c + h = 65 :=
by
  sorry

end seats_needed_on_bus_l10_10530


namespace newborn_members_approximation_l10_10226

-- Defining the conditions
def survival_prob_first_month : ℚ := 7/8
def survival_prob_second_month : ℚ := 7/8
def survival_prob_third_month : ℚ := 7/8
def survival_prob_three_months : ℚ := (7/8) ^ 3
def expected_survivors : ℚ := 133.984375

-- Statement to prove that the number of newborn members, N, approximates to 200
theorem newborn_members_approximation (N : ℚ) : 
  N * survival_prob_three_months = expected_survivors → 
  N = 200 :=
by
  sorry

end newborn_members_approximation_l10_10226


namespace last_fish_in_swamp_l10_10890

noncomputable def final_fish (perches pikes sudaks : ℕ) : String :=
  let p := perches
  let pi := pikes
  let s := sudaks
  if p = 6 ∧ pi = 7 ∧ s = 8 then "Sudak" else "Unknown"

theorem last_fish_in_swamp : final_fish 6 7 8 = "Sudak" := by
  sorry

end last_fish_in_swamp_l10_10890


namespace average_speed_l10_10526

-- Define the given conditions as Lean variables and constants
variables (v : ℕ)

-- The average speed problem in Lean
theorem average_speed (h : 8 * v = 528) : v = 66 :=
sorry

end average_speed_l10_10526


namespace sum_of_a_b_l10_10781

theorem sum_of_a_b (a b : ℝ) (h1 : a * b = 1) (h2 : (3 * a + 2 * b) * (3 * b + 2 * a) = 295) : a + b = 7 :=
by
  sorry

end sum_of_a_b_l10_10781


namespace trig_inequality_l10_10821

theorem trig_inequality (theta : ℝ) (h1 : Real.pi / 4 < theta) (h2 : theta < Real.pi / 2) : 
  Real.cos theta < Real.sin theta ∧ Real.sin theta < Real.tan theta :=
sorry

end trig_inequality_l10_10821


namespace final_statue_weight_l10_10624

-- Define the initial weight of the statue
def initial_weight : ℝ := 250

-- Define the percentage of weight remaining after each week
def remaining_after_week1 (w : ℝ) : ℝ := 0.70 * w
def remaining_after_week2 (w : ℝ) : ℝ := 0.80 * w
def remaining_after_week3 (w : ℝ) : ℝ := 0.75 * w

-- Define the final weight of the statue after three weeks
def final_weight : ℝ := 
  remaining_after_week3 (remaining_after_week2 (remaining_after_week1 initial_weight))

-- Prove the weight of the final statue is 105 kg
theorem final_statue_weight : final_weight = 105 := 
  by
    sorry

end final_statue_weight_l10_10624


namespace determine_velocities_l10_10472

theorem determine_velocities (V1 V2 : ℝ) (h1 : 60 / V2 = 60 / V1 + 5) (h2 : |V1 - V2| = 1)
  (h3 : 0 < V1) (h4 : 0 < V2) : V1 = 4 ∧ V2 = 3 :=
by
  sorry

end determine_velocities_l10_10472


namespace pyramid_base_length_l10_10532

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l10_10532


namespace order_of_abc_l10_10408

noncomputable def a : ℚ := 1 / 2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 6 - 2

theorem order_of_abc : a > c ∧ c > b := by
  sorry

end order_of_abc_l10_10408


namespace product_of_integers_l10_10959

theorem product_of_integers (x y : ℕ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 40) : x * y = 99 :=
by {
  sorry
}

end product_of_integers_l10_10959


namespace problem1_problem2_l10_10566

-- Definitions for the sets and conditions
def setA : Set ℝ := {x | -1 < x ∧ x < 2}
def setB (a : ℝ) : Set ℝ := if a > 0 then {x | x ≤ -2 ∨ x ≥ (1 / a)} else ∅

-- Problem 1: Prove the intersection for a == 1
theorem problem1 : (setB 1) ∩ setA = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

-- Problem 2: Prove the range of a
theorem problem2 (a : ℝ) (h : setB a ⊆ setAᶜ) : 0 < a ∧ a ≤ 1/2 :=
by
  sorry

end problem1_problem2_l10_10566


namespace proportion_solution_l10_10152

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 6) : x = 0.9 := by
  sorry

end proportion_solution_l10_10152


namespace geometric_sequence_x_l10_10496

theorem geometric_sequence_x (x : ℝ) (h : 1 * x = x ∧ x * x = 9) : x = 3 ∨ x = -3 := by
  sorry

end geometric_sequence_x_l10_10496


namespace problem_part_1_problem_part_2_l10_10204

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_b : ℝ × ℝ := (3, -Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * vector_b.1 + (vector_a x).2 * vector_b.2

theorem problem_part_1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) : 
  (vector_a x).1 * vector_b.2 = (vector_a x).2 * vector_b.1 → 
  x = 5 * Real.pi / 6 :=
by
  sorry

theorem problem_part_2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) :
  (∀ t, 0 ≤ t ∧ t ≤ Real.pi → f x ≤ f t) → x = 0 ∧ f 0 = 3 ∧ 
  (∀ t, 0 ≤ t ∧ t ≤ Real.pi → f x ≥ f t) → x = 5 * Real.pi / 6 ∧ f (5 * Real.pi / 6) = -2 * Real.sqrt 3 :=
by
  sorry

end problem_part_1_problem_part_2_l10_10204


namespace intersection_domains_l10_10259

def domain_f := {x : ℝ | x < 1}
def domain_g := {x : ℝ | x ≠ 0}

theorem intersection_domains :
  {x : ℝ | x < 1} ∩ {x : ℝ | x ≠ 0} = {x : ℝ | x < 1 ∧ x ≠ 0} :=
by 
  sorry

end intersection_domains_l10_10259


namespace num_packages_l10_10239

theorem num_packages (total_shirts : ℕ) (shirts_per_package : ℕ) (h1 : total_shirts = 51) (h2 : shirts_per_package = 3) : total_shirts / shirts_per_package = 17 := by
  sorry

end num_packages_l10_10239


namespace total_distance_correct_l10_10017

def d1 : ℕ := 350
def d2 : ℕ := 375
def d3 : ℕ := 275
def total_distance : ℕ := 1000

theorem total_distance_correct : d1 + d2 + d3 = total_distance := by
  sorry

end total_distance_correct_l10_10017


namespace chocolate_flavored_cups_sold_l10_10053

-- Define total sales and fractions
def total_cups_sold : ℕ := 50
def fraction_winter_melon : ℚ := 2 / 5
def fraction_okinawa : ℚ := 3 / 10
def fraction_chocolate : ℚ := 1 - (fraction_winter_melon + fraction_okinawa)

-- Define the number of chocolate-flavored cups sold
def num_chocolate_cups_sold : ℕ := 50 - (50 * 2 / 5 + 50 * 3 / 10)

-- Main theorem statement
theorem chocolate_flavored_cups_sold : num_chocolate_cups_sold = 15 := 
by 
  -- The proof would go here, but we use 'sorry' to skip it
  sorry

end chocolate_flavored_cups_sold_l10_10053


namespace original_side_length_l10_10528

theorem original_side_length (x : ℝ) 
  (h1 : (x - 4) * (x - 3) = 120) : x = 12 :=
sorry

end original_side_length_l10_10528


namespace perimeter_of_ABC_HI_IJK_l10_10936

theorem perimeter_of_ABC_HI_IJK (AB AC AH HI AI AK KI IJ JK : ℝ) 
(H_midpoint : H = AC / 2) (K_midpoint : K = AI / 2) 
(equil_triangle_ABC : AB = AC) (equil_triangle_AHI : AH = HI ∧ HI = AI) 
(equil_triangle_IJK : IJ = JK ∧ JK = KI) 
(AB_eq : AB = 6) : 
  AB + AC + AH + HI + IJ + JK + KI = 22.5 :=
by
  sorry

end perimeter_of_ABC_HI_IJK_l10_10936


namespace compute_expression_value_l10_10941

theorem compute_expression_value (x y : ℝ) (hxy : x ≠ y) 
  (h : 1 / (x^2 + 1) + 1 / (y^2 + 1) = 2 / (xy + 1)) :
  1 / (x^2 + 1) + 1 / (y^2 + 1) + 2 / (xy + 1) = 2 :=
by
  sorry

end compute_expression_value_l10_10941


namespace negation_of_proposition_l10_10016

theorem negation_of_proposition (a : ℝ) :
  (¬ (∀ x : ℝ, (x - a) ^ 2 + 2 > 0)) ↔ (∃ x : ℝ, (x - a) ^ 2 + 2 ≤ 0) :=
by
  sorry

end negation_of_proposition_l10_10016


namespace simplify_expression_l10_10502

theorem simplify_expression :
  (210 / 18) * (6 / 150) * (9 / 4) = 21 / 20 :=
by
  sorry

end simplify_expression_l10_10502


namespace ice_cream_total_volume_l10_10892

/-- 
  The interior of a right, circular cone is 12 inches tall with a 3-inch radius at the opening.
  The interior of the cone is filled with ice cream.
  The cone has a hemisphere of ice cream exactly covering the opening of the cone.
  On top of this hemisphere, there is a cylindrical layer of ice cream of height 2 inches 
  and the same radius as the hemisphere (3 inches).
  Prove that the total volume of ice cream is 72π cubic inches.
-/
theorem ice_cream_total_volume :
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  V_cone + V_hemisphere + V_cylinder = 72 * Real.pi :=
by {
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  sorry
}

end ice_cream_total_volume_l10_10892


namespace plot_area_is_correct_l10_10645

noncomputable def scaled_area_in_acres
  (scale_cm_miles : ℕ)
  (area_conversion_factor_miles_acres : ℕ)
  (bottom_cm : ℕ)
  (top_cm : ℕ)
  (height_cm : ℕ) : ℕ :=
  let area_cm_squared := (1 / 2) * (bottom_cm + top_cm) * height_cm
  let area_in_squared_miles := area_cm_squared * (scale_cm_miles * scale_cm_miles)
  area_in_squared_miles * area_conversion_factor_miles_acres

theorem plot_area_is_correct :
  scaled_area_in_acres 3 640 18 14 12 = 1105920 :=
by
  sorry

end plot_area_is_correct_l10_10645


namespace price_per_pot_l10_10580

-- Definitions based on conditions
def total_pots : ℕ := 80
def proportion_not_cracked : ℚ := 3 / 5
def total_revenue : ℚ := 1920

-- The Lean statement to prove she sold each clay pot for $40
theorem price_per_pot : (total_revenue / (total_pots * proportion_not_cracked)) = 40 := 
by sorry

end price_per_pot_l10_10580


namespace number_of_valid_three_digit_numbers_l10_10425

theorem number_of_valid_three_digit_numbers : 
  (∃ A B C : ℕ, 
      (100 * A + 10 * B + C + 297 = 100 * C + 10 * B + A) ∧ 
      (0 ≤ A ∧ A ≤ 9) ∧ 
      (0 ≤ B ∧ B ≤ 9) ∧ 
      (0 ≤ C ∧ C ≤ 9)) 
    ∧ (number_of_such_valid_numbers = 70) :=
by
  sorry

def number_of_such_valid_numbers : ℕ := 
  sorry

end number_of_valid_three_digit_numbers_l10_10425


namespace largest_inscribed_equilateral_triangle_area_l10_10424

theorem largest_inscribed_equilateral_triangle_area 
  (r : ℝ) (h_r : r = 10) : 
  ∃ A : ℝ, 
    A = 100 * Real.sqrt 3 ∧ 
    (∃ s : ℝ, s = 2 * r ∧ A = (Real.sqrt 3 / 4) * s^2) := 
  sorry

end largest_inscribed_equilateral_triangle_area_l10_10424


namespace minimize_notch_volume_l10_10138

noncomputable def total_volume (theta phi : ℝ) : ℝ :=
  let part1 := (2 / 3) * Real.tan phi
  let part2 := (2 / 3) * Real.tan (theta - phi)
  part1 + part2

theorem minimize_notch_volume :
  ∀ (theta : ℝ), (0 < theta ∧ theta < π) →
  ∃ (phi : ℝ), (0 < phi ∧ phi < θ) ∧
  (∀ ψ : ℝ, (0 < ψ ∧ ψ < θ) → total_volume theta ψ ≥ total_volume theta (theta / 2)) :=
by
  -- Proof is omitted
  sorry

end minimize_notch_volume_l10_10138


namespace point_C_lies_within_region_l10_10542

def lies_within_region (x y : ℝ) : Prop :=
  (x + y - 1 < 0) ∧ (x - y + 1 > 0)

theorem point_C_lies_within_region : lies_within_region 0 (-2) :=
by {
  -- Proof is omitted as per the instructions
  sorry
}

end point_C_lies_within_region_l10_10542


namespace solve_fraction_equation_l10_10169

theorem solve_fraction_equation :
  ∀ x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) → x = -2 / 11 :=
by
  intro x
  intro h
  sorry

end solve_fraction_equation_l10_10169


namespace speed_of_first_train_l10_10972

noncomputable def speed_of_second_train : ℝ := 40 -- km/h
noncomputable def length_of_first_train : ℝ := 125 -- m
noncomputable def length_of_second_train : ℝ := 125.02 -- m
noncomputable def time_to_pass_each_other : ℝ := 1.5 / 60 -- hours (converted from minutes)

theorem speed_of_first_train (V1 V2 : ℝ) 
  (h1 : V2 = speed_of_second_train)
  (h2 : 125 + 125.02 = 250.02) 
  (h3 : 1.5 / 60 = 0.025) :
  V1 - V2 = 10.0008 → V1 = 50 :=
by 
  sorry

end speed_of_first_train_l10_10972


namespace B_join_months_after_A_l10_10850

-- Definitions based on conditions
def capitalA (monthsA : ℕ) : ℕ := 3500 * monthsA
def capitalB (monthsB : ℕ) : ℕ := 9000 * monthsB

-- The condition that profit is in ratio 2:3 implies the ratio of their capitals should equal 2:3
def ratio_condition (x : ℕ) : Prop := 2 * (capitalB (12 - x)) = 3 * (capitalA 12)

-- Main theorem stating that B joined the business 5 months after A started
theorem B_join_months_after_A : ∃ x, ratio_condition x ∧ x = 5 :=
by
  use 5
  -- Proof would go here
  sorry

end B_join_months_after_A_l10_10850


namespace abs_ineq_range_l10_10500

theorem abs_ineq_range (x : ℝ) : |x - 3| + |x + 1| ≥ 4 ↔ -1 ≤ x ∧ x ≤ 3 :=
sorry

end abs_ineq_range_l10_10500


namespace plates_probability_l10_10601

noncomputable def number_of_plates := 12
noncomputable def red_plates := 6
noncomputable def light_blue_plates := 3
noncomputable def dark_blue_plates := 3
noncomputable def total_pairs := number_of_plates * (number_of_plates - 1) / 2
noncomputable def red_pairs := red_plates * (red_plates - 1) / 2
noncomputable def light_blue_pairs := light_blue_plates * (light_blue_plates - 1) / 2
noncomputable def dark_blue_pairs := dark_blue_plates * (dark_blue_plates - 1) / 2
noncomputable def mixed_blue_pairs := light_blue_plates * dark_blue_plates
noncomputable def total_satisfying_pairs := red_pairs + light_blue_pairs + dark_blue_pairs + mixed_blue_pairs
noncomputable def desired_probability := (total_satisfying_pairs : ℚ) / total_pairs

theorem plates_probability :
  desired_probability = 5 / 11 :=
by
  -- Add the proof here
  sorry

end plates_probability_l10_10601


namespace first_divisor_l10_10135

-- Definitions
def is_divisible_by (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

-- Theorem to prove
theorem first_divisor (x : ℕ) (h₁ : ∃ l, l = Nat.lcm x 35 ∧ is_divisible_by 1400 l ∧ 1400 / l = 8) : 
  x = 25 := 
sorry

end first_divisor_l10_10135


namespace final_percentage_of_alcohol_l10_10610

theorem final_percentage_of_alcohol (initial_volume : ℝ) (initial_alcohol_percentage : ℝ)
  (removed_alcohol : ℝ) (added_water : ℝ) :
  initial_volume = 15 → initial_alcohol_percentage = 25 →
  removed_alcohol = 2 → added_water = 3 →
  ( ( (initial_alcohol_percentage / 100 * initial_volume - removed_alcohol) / 
    (initial_volume - removed_alcohol + added_water) ) * 100 = 10.9375) :=
by
  intros
  sorry

end final_percentage_of_alcohol_l10_10610


namespace complement_intersection_M_N_l10_10392

def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x > -1}
def U : Set ℝ := Set.univ

theorem complement_intersection_M_N :
  U \ (M ∩ N) = {x | x ≤ -1} ∪ {x | x ≥ 3} :=
by
  sorry

end complement_intersection_M_N_l10_10392


namespace min_sum_of_factors_l10_10534

theorem min_sum_of_factors (a b : ℤ) (h1 : a * b = 72) : a + b ≥ -73 :=
sorry

end min_sum_of_factors_l10_10534


namespace cost_price_of_watch_l10_10720

theorem cost_price_of_watch (C : ℝ) 
  (h1 : ∃ (SP1 SP2 : ℝ), SP1 = 0.54 * C ∧ SP2 = 1.04 * C ∧ SP2 = SP1 + 140) : 
  C = 280 :=
by
  obtain ⟨SP1, SP2, H1, H2, H3⟩ := h1
  sorry

end cost_price_of_watch_l10_10720


namespace molecular_weight_of_compound_l10_10790

noncomputable def atomic_weight_carbon : ℝ := 12.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.008
noncomputable def atomic_weight_oxygen : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def num_oxygen_atoms : ℕ := 1

noncomputable def molecular_weight (num_C num_H num_O : ℕ) : ℝ :=
  (num_C * atomic_weight_carbon) + (num_H * atomic_weight_hydrogen) + (num_O * atomic_weight_oxygen)

theorem molecular_weight_of_compound :
  molecular_weight num_carbon_atoms num_hydrogen_atoms num_oxygen_atoms = 65.048 :=
by
  sorry

end molecular_weight_of_compound_l10_10790


namespace num_games_played_l10_10786

theorem num_games_played (n : ℕ) (h : n = 14) : (n.choose 2) = 91 :=
by
  sorry

end num_games_played_l10_10786


namespace ken_kept_pencils_l10_10207

def ken_total_pencils := 50
def pencils_given_to_manny := 10
def pencils_given_to_nilo := pencils_given_to_manny + 10
def pencils_given_away := pencils_given_to_manny + pencils_given_to_nilo

theorem ken_kept_pencils : ken_total_pencils - pencils_given_away = 20 := by
  sorry

end ken_kept_pencils_l10_10207


namespace abs_sub_eq_three_l10_10142

theorem abs_sub_eq_three {m n : ℝ} (h1 : m * n = 4) (h2 : m + n = 5) : |m - n| = 3 := 
sorry

end abs_sub_eq_three_l10_10142


namespace max_value_of_f_value_of_f_given_tan_half_alpha_l10_10499

noncomputable def f (x : ℝ) := 2 * (Real.cos (x / 2)) ^ 2 + Real.sqrt 3 * (Real.sin x)

theorem max_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≤ 3) ∧ (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3 ∧ f x = 3) :=
sorry

theorem value_of_f_given_tan_half_alpha (α : ℝ) (h : Real.tan (α / 2) = 1 / 2) :
  f α = (8 + 4 * Real.sqrt 3) / 5 :=
sorry

end max_value_of_f_value_of_f_given_tan_half_alpha_l10_10499


namespace problem1_problem2_l10_10396

noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then 2 * a - (x + 4 / x)
  else x - 4 / x

theorem problem1 (h : ∀ x : ℝ, f 1 x = 3 → x = 4) : ∃ x : ℝ, f 1 x = 3 ∧ x = 4 :=
sorry

theorem problem2 (h : ∀ x1 x2 x3 : ℝ, 
  (x1 < x2 ∧ x2 < x3 ∧ x2 - x1 = x3 - x2) →
  f a x1 = 3 ∧ f a x2 = 3 ∧ f a x3 = 3 ∧ a ≤ -1 → 
  a = -11 / 6) : ∃ a : ℝ, a ≤ -1 ∧ (∃ x1 x2 x3 : ℝ, 
  (x1 < x2 ∧ x2 < x3 ∧ x2 - x1 = x3 - x2) ∧ 
  f a x1 = 3 ∧ f a x2 = 3 ∧ f a x3 = 3 ∧ a = -11 / 6) :=
sorry

end problem1_problem2_l10_10396


namespace closest_point_on_line_y_eq_3x_plus_2_l10_10011

theorem closest_point_on_line_y_eq_3x_plus_2 (x y : ℝ) :
  ∃ (p : ℝ × ℝ), p = (-1 / 2, 1 / 2) ∧ y = 3 * x + 2 ∧ p = (x, y) :=
by
-- We skip the proof steps and provide the statement only
sorry

end closest_point_on_line_y_eq_3x_plus_2_l10_10011


namespace bertha_daughters_no_daughters_l10_10940

theorem bertha_daughters_no_daughters (daughters granddaughters: ℕ) (no_great_granddaughters: granddaughters = 5 * daughters) (total_women: 8 + granddaughters = 48) :
  8 + granddaughters = 48 :=
by {
  sorry
}

end bertha_daughters_no_daughters_l10_10940


namespace work_days_for_c_l10_10401

theorem work_days_for_c (A B C : ℝ)
  (h1 : A + B = 1 / 15)
  (h2 : A + B + C = 1 / 11) :
  1 / C = 41.25 :=
by
  sorry

end work_days_for_c_l10_10401


namespace find_a_l10_10531

noncomputable def center_radius_circle1 (x y : ℝ) := x^2 + y^2 = 16
noncomputable def center_radius_circle2 (x y a : ℝ) := (x - a)^2 + y^2 = 1
def centers_tangent (a : ℝ) : Prop := |a| = 5 ∨ |a| = 3

theorem find_a (a : ℝ) (h1 : center_radius_circle1 x y) (h2 : center_radius_circle2 x y a) : centers_tangent a :=
sorry

end find_a_l10_10531


namespace rings_on_fingers_arrangement_l10_10625

-- Definitions based on the conditions
def rings : ℕ := 5
def fingers : ℕ := 5

-- Theorem statement
theorem rings_on_fingers_arrangement : (fingers ^ rings) = 5 ^ 5 := by
  sorry  -- Proof skipped

end rings_on_fingers_arrangement_l10_10625


namespace ked_ben_eggs_ratio_l10_10904

theorem ked_ben_eggs_ratio 
  (saly_needs_ben_weekly_ratio : ℕ)
  (weeks_in_month : ℕ := 4) 
  (total_production_month : ℕ := 124)
  (saly_needs_weekly : ℕ := 10) 
  (ben_needs_weekly : ℕ := 14)
  (ben_needs_monthly : ℕ := ben_needs_weekly * weeks_in_month)
  (saly_needs_monthly : ℕ := saly_needs_weekly * weeks_in_month)
  (total_saly_ben_monthly : ℕ := saly_needs_monthly + ben_needs_monthly)
  (ked_needs_monthly : ℕ := total_production_month - total_saly_ben_monthly)
  (ked_needs_weekly : ℕ := ked_needs_monthly / weeks_in_month) :
  ked_needs_weekly / ben_needs_weekly = 1 / 2 :=
sorry

end ked_ben_eggs_ratio_l10_10904


namespace AM_GM_inequality_example_l10_10364

theorem AM_GM_inequality_example (a b c d : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_prod : a * b * c * d = 1) :
  a^3 + b^3 + c^3 + d^3 ≥ max (a + b + c + d) (1 / a + 1 / b + 1 / c + 1 / d) :=
by
  sorry

end AM_GM_inequality_example_l10_10364


namespace exists_distinct_ij_l10_10841

theorem exists_distinct_ij (n : ℕ) (a : Fin n → ℤ) (h_distinct : Function.Injective a) (h_n_ge_3 : 3 ≤ n) :
  ∃ (i j : Fin n), i ≠ j ∧ (∀ k, (a i + a j) ∣ 3 * a k → False) :=
by
  sorry

end exists_distinct_ij_l10_10841


namespace chandler_weeks_to_buy_bike_l10_10639

-- Define the given problem conditions as variables/constants
def bike_cost : ℕ := 650
def grandparents_gift : ℕ := 60
def aunt_gift : ℕ := 45
def cousin_gift : ℕ := 25
def weekly_earnings : ℕ := 20
def total_birthday_money : ℕ := grandparents_gift + aunt_gift + cousin_gift

-- Define the total money Chandler will have after x weeks
def total_money_after_weeks (x : ℕ) : ℕ := total_birthday_money + weekly_earnings * x

-- The main theorem states that Chandler needs 26 weeks to save enough money to buy the bike
theorem chandler_weeks_to_buy_bike : ∃ x : ℕ, total_money_after_weeks x = bike_cost :=
by
  -- Since we know x = 26 from the solution:
  use 26
  sorry

end chandler_weeks_to_buy_bike_l10_10639


namespace solve_quadratic_inequality_l10_10146

theorem solve_quadratic_inequality (a x : ℝ) (h : a < 1) : 
  x^2 - (a + 1) * x + a < 0 ↔ (a < x ∧ x < 1) :=
by
  sorry

end solve_quadratic_inequality_l10_10146


namespace smallest_integer_value_l10_10193

theorem smallest_integer_value (x : ℤ) (h : 3 * |x| + 8 < 29) : x = -6 :=
sorry

end smallest_integer_value_l10_10193


namespace cost_of_fruits_l10_10721

-- Definitions based on the conditions
variables (x y z : ℝ)

-- Conditions
axiom h1 : 2 * x + y + 4 * z = 6
axiom h2 : 4 * x + 2 * y + 2 * z = 4

-- Question to prove
theorem cost_of_fruits : 4 * x + 2 * y + 5 * z = 8 :=
sorry

end cost_of_fruits_l10_10721


namespace equilateral_triangle_area_increase_l10_10869

theorem equilateral_triangle_area_increase (A : ℝ) (k : ℝ) (s : ℝ) (s' : ℝ) (A' : ℝ) (ΔA : ℝ) :
  A = 36 * Real.sqrt 3 →
  A = (Real.sqrt 3 / 4) * s^2 →
  s' = s + 3 →
  A' = (Real.sqrt 3 / 4) * s'^2 →
  ΔA = A' - A →
  ΔA = 20.25 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_increase_l10_10869


namespace book_stack_sum_l10_10739

theorem book_stack_sum : 
  let a := 15 -- first term
  let d := -2 -- common difference
  let l := 1 -- last term
  -- n = (l - a) / d + 1
  let n := (l - a) / d + 1
  -- S = n * (a + l) / 2
  let S := n * (a + l) / 2
  S = 64 :=
by
  -- The given conditions
  let a := 15 -- first term
  let d := -2 -- common difference
  let l := 1 -- last term
  -- Calculate the number of terms (n)
  let n := (l - a) / d + 1
  -- Calculate the total sum (S)
  let S := n * (a + l) / 2
  -- Prove the sum is 64
  show S = 64
  sorry

end book_stack_sum_l10_10739


namespace pizza_slices_all_toppings_l10_10889

theorem pizza_slices_all_toppings (x : ℕ) :
  (16 = (8 - x) + (12 - x) + (6 - x) + x) → x = 5 := by
  sorry

end pizza_slices_all_toppings_l10_10889


namespace problem_sum_150_consecutive_integers_l10_10087

theorem problem_sum_150_consecutive_integers : 
  ∃ k : ℕ, 150 * k + 11325 = 5310375 :=
sorry

end problem_sum_150_consecutive_integers_l10_10087


namespace line_tangent_to_ellipse_l10_10760

theorem line_tangent_to_ellipse (k : ℝ) :
  (∃ x : ℝ, 2 * x ^ 2 + 8 * (k * x + 2) ^ 2 = 8 ∧
             ∀ x1 x2 : ℝ, (2 + 8 * k ^ 2) * x1 ^ 2 + 32 * k * x1 + 24 = 0 →
             (2 + 8 * k ^ 2) * x2 ^ 2 + 32 * k * x2 + 24 = 0 → x1 = x2) →
  k^2 = 3 / 4 := by
  sorry

end line_tangent_to_ellipse_l10_10760


namespace max_time_digit_sum_l10_10815

-- Define the conditions
def is_valid_time (h m : ℕ) : Prop :=
  (0 ≤ h ∧ h < 24) ∧ (0 ≤ m ∧ m < 60)

-- Define the function to calculate the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ :=
  n % 10 + n / 10

-- Define the function to calculate the sum of digits in the time display
def time_digit_sum (h m : ℕ) : ℕ :=
  digit_sum h + digit_sum m

-- The theorem to prove
theorem max_time_digit_sum : ∀ (h m : ℕ),
  is_valid_time h m → time_digit_sum h m ≤ 24 :=
by {
  sorry
}

end max_time_digit_sum_l10_10815


namespace model_price_and_schemes_l10_10290

theorem model_price_and_schemes :
  ∃ (x y : ℕ), 3 * x = 2 * y ∧ x + 2 * y = 80 ∧ x = 20 ∧ y = 30 ∧ 
  ∃ (count m : ℕ), 468 ≤ m ∧ m ≤ 480 ∧ 
                   (20 * m + 30 * (800 - m) ≤ 19320) ∧ 
                   (800 - m ≥ 2 * m / 3) ∧ 
                   count = 13 ∧ 
                   800 - 480 = 320 :=
sorry

end model_price_and_schemes_l10_10290


namespace James_weight_after_gain_l10_10359

theorem James_weight_after_gain 
    (initial_weight : ℕ)
    (muscle_gain_perc : ℕ)
    (fat_gain_fraction : ℚ)
    (weight_after_gain : ℕ) :
    initial_weight = 120 →
    muscle_gain_perc = 20 →
    fat_gain_fraction = 1/4 →
    weight_after_gain = 150 :=
by
  intros
  sorry

end James_weight_after_gain_l10_10359


namespace original_board_is_120_l10_10427

-- Define the two given conditions
def S : ℕ := 35
def L : ℕ := 2 * S + 15

-- Define the length of the original board
def original_board_length : ℕ := S + L

-- The theorem we want to prove
theorem original_board_is_120 : original_board_length = 120 :=
by
  -- Skipping the actual proof
  sorry

end original_board_is_120_l10_10427


namespace moles_of_magnesium_l10_10920

-- Assuming the given conditions as hypotheses
variables (Mg CO₂ MgO C : ℕ)

-- Theorem statement
theorem moles_of_magnesium (h1 : 2 * Mg + CO₂ = 2 * MgO + C) 
                           (h2 : MgO = Mg) 
                           (h3 : CO₂ = 1) 
                           : Mg = 2 :=
by sorry  -- Proof to be provided

end moles_of_magnesium_l10_10920


namespace number_divided_by_five_is_same_as_three_added_l10_10231

theorem number_divided_by_five_is_same_as_three_added :
  ∃ x : ℚ, x / 5 = x + 3 ∧ x = -15 / 4 :=
by
  sorry

end number_divided_by_five_is_same_as_three_added_l10_10231


namespace general_term_formula_l10_10172

theorem general_term_formula (n : ℕ) (a : ℕ → ℚ) :
  (∀ n, a n = (-1)^n * (n^2)/(2 * n - 1)) :=
sorry

end general_term_formula_l10_10172


namespace distinct_positive_integers_factors_PQ_RS_l10_10200

theorem distinct_positive_integers_factors_PQ_RS (P Q R S : ℕ) (hP : P > 0) (hQ : Q > 0) (hR : R > 0) (hS : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDistinctPQ : P ≠ Q) (hDistinctRS : R ≠ S) (hPQR_S : P + Q = R - S) :
  P = 4 :=
by
  sorry

end distinct_positive_integers_factors_PQ_RS_l10_10200


namespace cos_value_of_geometric_sequence_l10_10947

theorem cos_value_of_geometric_sequence (a : ℕ → ℝ) (r : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * r)
  (h2 : a 1 * a 13 + 2 * (a 7) ^ 2 = 5 * Real.pi) :
  Real.cos (a 2 * a 12) = 1 / 2 := 
sorry

end cos_value_of_geometric_sequence_l10_10947


namespace inequality_has_solutions_iff_a_ge_4_l10_10375

theorem inequality_has_solutions_iff_a_ge_4 (a x : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) ↔ a ≥ 4 :=
sorry

end inequality_has_solutions_iff_a_ge_4_l10_10375


namespace problem_statement_l10_10058

theorem problem_statement (P : ℝ) (h : P = 1 / (Real.log 11 / Real.log 2) + 1 / (Real.log 11 / Real.log 3) + 1 / (Real.log 11 / Real.log 4) + 1 / (Real.log 11 / Real.log 5)) : 1 < P ∧ P < 2 := 
sorry

end problem_statement_l10_10058


namespace emily_dog_count_l10_10965

theorem emily_dog_count (dogs : ℕ) 
  (food_per_day_per_dog : ℕ := 250) 
  (vacation_days : ℕ := 14)
  (total_food_kg : ℕ := 14)
  (kg_to_grams : ℕ := 1000) 
  (total_food_grams : ℕ := total_food_kg * kg_to_grams)
  (food_needed_per_dog : ℕ := food_per_day_per_dog * vacation_days) 
  (total_food_needed : ℕ := dogs * food_needed_per_dog) 
  (h : total_food_needed = total_food_grams) : 
  dogs = 4 := 
sorry

end emily_dog_count_l10_10965


namespace christen_potatoes_l10_10559

theorem christen_potatoes :
  let total_potatoes := 60
  let homer_rate := 4
  let christen_rate := 6
  let alex_potatoes := 2
  let homer_minutes := 6
  homer_minutes * homer_rate + christen_rate * ((total_potatoes + alex_potatoes - homer_minutes * homer_rate) / (homer_rate + christen_rate)) = 24 := 
sorry

end christen_potatoes_l10_10559


namespace canoe_speed_downstream_l10_10074

theorem canoe_speed_downstream (V_upstream V_s V_c V_downstream : ℝ) 
    (h1 : V_upstream = 6) 
    (h2 : V_s = 2) 
    (h3 : V_upstream = V_c - V_s) 
    (h4 : V_downstream = V_c + V_s) : 
  V_downstream = 10 := 
by 
  sorry

end canoe_speed_downstream_l10_10074


namespace totalSleepIsThirtyHours_l10_10274

-- Define constants and conditions
def recommendedSleep : ℝ := 8
def sleepOnTwoDays : ℝ := 3
def percentageSleepOnOtherDays : ℝ := 0.6
def daysInWeek : ℕ := 7
def daysWithThreeHoursSleep : ℕ := 2
def remainingDays : ℕ := daysInWeek - daysWithThreeHoursSleep

-- Define total sleep calculation
theorem totalSleepIsThirtyHours :
  let sleepOnFirstTwoDays := (daysWithThreeHoursSleep : ℝ) * sleepOnTwoDays
  let sleepOnRemainingDays := (remainingDays : ℝ) * (recommendedSleep * percentageSleepOnOtherDays)
  sleepOnFirstTwoDays + sleepOnRemainingDays = 30 := 
by
  sorry

end totalSleepIsThirtyHours_l10_10274


namespace jaylen_bell_peppers_ratio_l10_10976

theorem jaylen_bell_peppers_ratio :
  ∃ j_bell_p, ∃ k_bell_p, ∃ j_green_b, ∃ k_green_b, ∃ j_carrots, ∃ j_cucumbers, ∃ j_total_veg,
  j_carrots = 5 ∧
  j_cucumbers = 2 ∧
  k_bell_p = 2 ∧
  k_green_b = 20 ∧
  j_green_b = 20 / 2 - 3 ∧
  j_total_veg = 18 ∧
  j_carrots + j_cucumbers + j_green_b + j_bell_p = j_total_veg ∧
  j_bell_p / k_bell_p = 2 :=
sorry

end jaylen_bell_peppers_ratio_l10_10976


namespace arrange_magnitudes_l10_10093

theorem arrange_magnitudes (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  let y := x^x
  let z := x^(x^x)
  x < z ∧ z < y := by
  sorry

end arrange_magnitudes_l10_10093


namespace number_of_weeks_in_a_single_harvest_season_l10_10044

-- Define constants based on conditions
def weeklyEarnings : ℕ := 1357
def totalHarvestSeasons : ℕ := 73
def totalEarnings : ℕ := 22090603

-- Prove the number of weeks in a single harvest season
theorem number_of_weeks_in_a_single_harvest_season :
  (totalEarnings / weeklyEarnings) / totalHarvestSeasons = 223 := 
  by
    sorry

end number_of_weeks_in_a_single_harvest_season_l10_10044


namespace solution_statement_l10_10015

-- Define the set of courses
inductive Course
| Physics | Chemistry | Literature | History | Philosophy | Psychology

open Course

-- Define the condition that a valid program must include Physics and at least one of Chemistry or Literature
def valid_program (program : Finset Course) : Prop :=
  Course.Physics ∈ program ∧
  (Course.Chemistry ∈ program ∨ Course.Literature ∈ program)

-- Define the problem statement
def problem_statement : Prop :=
  ∃ programs : Finset (Finset Course),
    programs.card = 9 ∧ ∀ program ∈ programs, program.card = 5 ∧ valid_program program

theorem solution_statement : problem_statement := sorry

end solution_statement_l10_10015


namespace three_digit_numbers_satisfying_condition_l10_10007

theorem three_digit_numbers_satisfying_condition :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000) →
    ∃ (a b c : ℕ),
      (N = 100 * a + 10 * b + c) ∧ (N = 11 * (a^2 + b^2 + c^2)) 
    ↔ (N = 550 ∨ N = 803) :=
by
  sorry

end three_digit_numbers_satisfying_condition_l10_10007


namespace total_students_l10_10576

theorem total_students (x : ℝ) :
  (x - (1/2)*x - (1/4)*x - (1/8)*x = 3) → x = 24 :=
by
  intro h
  sorry

end total_students_l10_10576


namespace solve_for_question_mark_l10_10212

def cube_root (x : ℝ) := x^(1/3)
def square_root (x : ℝ) := x^(1/2)

theorem solve_for_question_mark : 
  cube_root (5568 / 87) + square_root (72 * 2) = square_root 256 := by
  sorry

end solve_for_question_mark_l10_10212


namespace eval_nested_fractions_l10_10219

theorem eval_nested_fractions : (1 / (1 + 1 / (4 + 1 / 5))) = (21 / 26) :=
by
  sorry

end eval_nested_fractions_l10_10219


namespace count_of_green_hats_l10_10666

-- Defining the total number of hats
def total_hats : ℕ := 85

-- Defining the costs of each hat type
def blue_cost : ℕ := 6
def green_cost : ℕ := 7
def red_cost : ℕ := 8

-- Defining the total cost
def total_cost : ℕ := 600

-- Defining the ratio as 3:2:1
def ratio_blue : ℕ := 3
def ratio_green : ℕ := 2
def ratio_red : ℕ := 1

-- Defining the multiplication factor
def x : ℕ := 14

-- Number of green hats based on the ratio
def G : ℕ := ratio_green * x

-- Proving that we bought 28 green hats
theorem count_of_green_hats : G = 28 := by
  -- proof steps intention: sorry to skip the proof
  sorry

end count_of_green_hats_l10_10666


namespace profit_percent_calculation_l10_10206

variable (SP : ℝ) (CP : ℝ) (Profit : ℝ) (ProfitPercent : ℝ)
variable (h1 : CP = 0.75 * SP)
variable (h2 : Profit = SP - CP)
variable (h3 : ProfitPercent = (Profit / CP) * 100)

theorem profit_percent_calculation : ProfitPercent = 33.33 := 
sorry

end profit_percent_calculation_l10_10206


namespace laundry_loads_l10_10607

-- Definitions based on conditions
def num_families : ℕ := 3
def people_per_family : ℕ := 4
def num_people : ℕ := num_families * people_per_family

def days : ℕ := 7
def towels_per_person_per_day : ℕ := 1
def total_towels : ℕ := num_people * days * towels_per_person_per_day

def washing_machine_capacity : ℕ := 14

-- Statement to prove
theorem laundry_loads : total_towels / washing_machine_capacity = 6 := 
by
  sorry

end laundry_loads_l10_10607


namespace sculpture_exposed_surface_area_l10_10545

theorem sculpture_exposed_surface_area :
  let l₁ := 9
  let l₂ := 6
  let l₃ := 4
  let l₄ := 1

  let exposed_bottom_layer := 9 + 16
  let exposed_second_layer := 6 + 10
  let exposed_third_layer := 4 + 8
  let exposed_top_layer := 5

  l₁ + l₂ + l₃ + l₄ = 20 →
  exposed_bottom_layer + exposed_second_layer + exposed_third_layer + exposed_top_layer = 58 :=
by {
  sorry
}

end sculpture_exposed_surface_area_l10_10545


namespace find_a_l10_10186

theorem find_a (a : ℝ) (i : ℂ) (hi : i = Complex.I) (z : ℂ) (hz : z = a + i) (h : z^2 + z = 1 - 3 * Complex.I) :
  a = -2 :=
by {
  sorry
}

end find_a_l10_10186


namespace star_operation_l10_10963

def new_op (a b : ℝ) : ℝ :=
  a^2 + b^2 - a * b

theorem star_operation (x y : ℝ) : 
  new_op (x + 2 * y) (y + 3 * x) = 7 * x^2 + 3 * y^2 + 3 * (x * y) :=
by
  sorry

end star_operation_l10_10963


namespace find_sphere_volume_l10_10176

noncomputable def sphere_volume (d: ℝ) (V: ℝ) : Prop := d = 3 * (16 / 9) * V

theorem find_sphere_volume :
  sphere_volume (2 / 3) (1 / 6) :=
by
  sorry

end find_sphere_volume_l10_10176


namespace value_range_a_l10_10796

theorem value_range_a (a : ℝ) :
  (∀ (x : ℝ), |x + 2| * |x - 3| ≥ 4 / (a - 1)) ↔ (a < 1 ∨ a = 3) :=
by
  sorry

end value_range_a_l10_10796


namespace train_length_approx_l10_10622

noncomputable def length_of_train (distance_km : ℝ) (time_min : ℝ) (time_sec : ℝ) : ℝ :=
  let distance_m := distance_km * 1000 -- Convert km to meters
  let time_s := time_min * 60 -- Convert min to seconds
  let speed := distance_m / time_s -- Speed in meters/second
  speed * time_sec -- Length of train in meters

theorem train_length_approx :
  length_of_train 10 15 10 = 111.1 :=
by
  sorry

end train_length_approx_l10_10622


namespace number_of_pairs_sold_l10_10967

-- Define the conditions
def total_amount_made : ℝ := 588
def average_price_per_pair : ℝ := 9.8

-- The theorem we want to prove
theorem number_of_pairs_sold : total_amount_made / average_price_per_pair = 60 := 
by sorry

end number_of_pairs_sold_l10_10967


namespace platform_length_l10_10623

theorem platform_length
  (L_train : ℕ) (T_platform : ℕ) (T_pole : ℕ) (P : ℕ)
  (h1 : L_train = 300)
  (h2 : T_platform = 39)
  (h3 : T_pole = 10)
  (h4 : L_train / T_pole * T_platform = L_train + P) :
  P = 870 := 
sorry

end platform_length_l10_10623


namespace simplify_expression_l10_10443

variable (x : ℝ)

theorem simplify_expression :
  (3 * x - 2) * (5 * x ^ 12 - 3 * x ^ 11 + 2 * x ^ 9 - x ^ 6) =
  15 * x ^ 13 - 19 * x ^ 12 - 6 * x ^ 11 + 6 * x ^ 10 - 4 * x ^ 9 - 3 * x ^ 7 + 2 * x ^ 6 :=
by
  sorry

end simplify_expression_l10_10443


namespace smallest_n_terminating_decimal_l10_10313

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (n > 0) ∧
           (∃ (k: ℕ), (n + 150) = 2^k ∧ k < 150) ∨ 
           (∃ (k m: ℕ), (n + 150) = 2^k * 5^m ∧ m < 150) ∧ 
           ∀ m : ℕ, ((m > 0 ∧ (∃ (j: ℕ), (m + 150) = 2^j ∧ j < 150) ∨ 
           (∃ (j l: ℕ), (m + 150) = 2^j * 5^l ∧ l < 150)) → m ≥ n)
:= ⟨10, by {
  sorry
}⟩

end smallest_n_terminating_decimal_l10_10313


namespace Johnson_Smith_tied_end_May_l10_10539

def home_runs_Johnson : List ℕ := [2, 12, 15, 8, 14, 11, 9, 16]
def home_runs_Smith : List ℕ := [5, 9, 10, 12, 15, 12, 10, 17]

def total_without_June (runs: List ℕ) : Nat := List.sum (runs.take 5 ++ runs.drop 5)
def estimated_June (total: Nat) : Nat := total / 8

theorem Johnson_Smith_tied_end_May :
  let total_Johnson := total_without_June home_runs_Johnson;
  let total_Smith := total_without_June home_runs_Smith;
  let estimated_June_Johnson := estimated_June total_Johnson;
  let estimated_June_Smith := estimated_June total_Smith;
  let total_with_June_Johnson := total_Johnson + estimated_June_Johnson;
  let total_with_June_Smith := total_Smith + estimated_June_Smith;
  (List.sum (home_runs_Johnson.take 5) = List.sum (home_runs_Smith.take 5)) :=
by
  sorry

end Johnson_Smith_tied_end_May_l10_10539


namespace jessica_mother_age_l10_10478

theorem jessica_mother_age
  (mother_age_when_died : ℕ)
  (jessica_age_when_died : ℕ)
  (jessica_current_age : ℕ)
  (years_since_mother_died : ℕ)
  (half_age_condition : jessica_age_when_died = mother_age_when_died / 2)
  (current_age_condition : jessica_current_age = 40)
  (years_since_death_condition : years_since_mother_died = 10)
  (age_at_death_condition : jessica_age_when_died = jessica_current_age - years_since_mother_died) :
  mother_age_when_died + years_since_mother_died = 70 :=
by {
  sorry
}

end jessica_mother_age_l10_10478


namespace four_lines_set_l10_10081

-- Define the ⬩ operation
def clubsuit (a b : ℝ) := a^3 * b - a * b^3

-- Define the main theorem
theorem four_lines_set (x y : ℝ) : 
  (clubsuit x y = clubsuit y x) ↔ (y = 0 ∨ x = 0 ∨ y = x ∨ y = -x) :=
by sorry

end four_lines_set_l10_10081


namespace third_character_has_2_lines_l10_10878

-- Define the number of lines characters have
variables (x y z : ℕ)

-- The third character has x lines
-- Condition: The second character has 6 more than three times the number of lines the third character has
def second_character_lines : ℕ := 3 * x + 6

-- Condition: The first character has 8 more lines than the second character
def first_character_lines : ℕ := second_character_lines x + 8

-- The first character has 20 lines
def first_character_has_20_lines : Prop := first_character_lines x = 20

-- Prove that the third character has 2 lines
theorem third_character_has_2_lines (h : first_character_has_20_lines x) : x = 2 :=
by
  -- Skipping the proof
  sorry

end third_character_has_2_lines_l10_10878


namespace no_such_triplets_of_positive_reals_l10_10671

-- Define the conditions that the problem states.
def satisfies_conditions (a b c : ℝ) : Prop :=
  a = b + c ∧ b = c + a ∧ c = a + b

-- The main theorem to prove.
theorem no_such_triplets_of_positive_reals :
  ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) → satisfies_conditions a b c → false :=
by
  intro a b c
  intro ha hb hc
  intro habc
  sorry

end no_such_triplets_of_positive_reals_l10_10671


namespace each_person_received_5_l10_10467

theorem each_person_received_5 (S n : ℕ) (hn₁ : n > 5) (hn₂ : 5 * S = 2 * n * (n - 5)) (hn₃ : 4 * S = n * (n + 4)) :
  S / (n + 4) = 5 :=
by
  sorry

end each_person_received_5_l10_10467


namespace inequality_system_solution_l10_10261

theorem inequality_system_solution (x : ℤ) :
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1 + 2 * x) / 3 ≥ x - 1) ↔ (x = 3 ∨ x = 4) := sorry

end inequality_system_solution_l10_10261


namespace factorization_problem_l10_10955

theorem factorization_problem (x : ℝ) :
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 =
  (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + 2) * (x + 1) * (x - 1) :=
sorry

end factorization_problem_l10_10955


namespace council_revote_l10_10647

theorem council_revote (x y x' y' m : ℝ) (h1 : x + y = 500)
    (h2 : y - x = m) (h3 : x' - y' = 1.5 * m) (h4 : x' + y' = 500) (h5 : x' = 11 / 10 * y) :
    x' - x = 156.25 := by
  -- Proof goes here
  sorry

end council_revote_l10_10647


namespace range_of_y_l10_10156

theorem range_of_y (y : ℝ) (h₁ : y < 0) (h₂ : ⌈y⌉ * ⌊y⌋ = 110) : -11 < y ∧ y < -10 := 
sorry

end range_of_y_l10_10156


namespace find_f_at_one_l10_10689

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 4 * x ^ 2 - m * x + 5

theorem find_f_at_one :
  (∀ x : ℝ, x ≥ -2 → f x (-16) ≥ f (-2) (-16)) ∧
  (∀ x : ℝ, x ≤ -2 → f x (-16) ≤ f (-2) (-16)) →
  f 1 (-16) = 25 :=
sorry

end find_f_at_one_l10_10689


namespace cake_icing_l10_10080

/-- Define the cake conditions -/
structure Cake :=
  (dimension : ℕ)
  (small_cube_dimension : ℕ)
  (total_cubes : ℕ)
  (iced_faces : ℕ)

/-- Define the main theorem to prove the number of smaller cubes with icing on exactly two sides -/
theorem cake_icing (c : Cake) : 
  c.dimension = 5 ∧ c.small_cube_dimension = 1 ∧ c.total_cubes = 125 ∧ c.iced_faces = 4 →
  ∃ n, n = 20 :=
by
  sorry

end cake_icing_l10_10080


namespace log2_bounds_158489_l10_10766

theorem log2_bounds_158489 :
  (2^16 = 65536) ∧ (2^17 = 131072) ∧ (65536 < 158489 ∧ 158489 < 131072) →
  (16 < Real.log 158489 / Real.log 2 ∧ Real.log 158489 / Real.log 2 < 17) ∧ 16 + 17 = 33 :=
by
  intro h
  have h1 : 2^16 = 65536 := h.1
  have h2 : 2^17 = 131072 := h.2.1
  have h3 : 65536 < 158489 := h.2.2.1
  have h4 : 158489 < 131072 := h.2.2.2
  sorry

end log2_bounds_158489_l10_10766


namespace number_of_intersections_l10_10684

def line_eq (x y : ℝ) : Prop := 4 * x + 9 * y = 12
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

theorem number_of_intersections : 
  ∃ (p1 p2 : ℝ × ℝ), 
  (line_eq p1.1 p1.2 ∧ circle_eq p1.1 p1.2) ∧ 
  (line_eq p2.1 p2.2 ∧ circle_eq p2.1 p2.2) ∧ 
  p1 ≠ p2 ∧ 
  ∀ p : ℝ × ℝ, 
    (line_eq p.1 p.2 ∧ circle_eq p.1 p.2) → (p = p1 ∨ p = p2) :=
sorry

end number_of_intersections_l10_10684


namespace problem1_solution_problem2_solution_l10_10660

-- Problem 1
theorem problem1_solution (x : ℝ) : (2 * x - 3) * (x + 1) < 0 ↔ (-1 < x) ∧ (x < 3 / 2) :=
sorry

-- Problem 2
theorem problem2_solution (x : ℝ) : (4 * x - 1) / (x + 2) ≥ 0 ↔ (x < -2) ∨ (x >= 1 / 4) :=
sorry

end problem1_solution_problem2_solution_l10_10660


namespace gcd_polynomial_l10_10886

theorem gcd_polynomial (b : ℤ) (h1 : ∃ k : ℤ, b = 7 * k ∧ k % 2 = 1) : 
  Int.gcd (3 * b ^ 2 + 34 * b + 76) (b + 16) = 7 := 
sorry

end gcd_polynomial_l10_10886


namespace number_of_ways_is_25_l10_10900

-- Define the number of books
def number_of_books : ℕ := 5

-- Define the function to calculate the number of ways
def number_of_ways_to_buy_books : ℕ :=
  number_of_books * number_of_books

-- Define the theorem to be proved
theorem number_of_ways_is_25 : 
  number_of_ways_to_buy_books = 25 :=
by
  sorry

end number_of_ways_is_25_l10_10900


namespace initial_walking_speed_l10_10635

variable (v : ℝ)

theorem initial_walking_speed :
  (13.5 / v - 13.5 / 6 = 27 / 60) → v = 5 :=
by
  intro h
  sorry

end initial_walking_speed_l10_10635


namespace find_x_l10_10312

theorem find_x (h : ℝ → ℝ)
  (H1 : ∀x, h (3*x - 2) = 5*x + 6) :
  (∀x, h x = 2*x - 1) → x = 31 :=
by
  sorry

end find_x_l10_10312


namespace perimeter_of_region_is_70_l10_10336

-- Define the given conditions
def area_of_region (total_area : ℝ) (num_squares : ℕ) : Prop :=
  total_area = 392 ∧ num_squares = 8

def side_length_of_square (area : ℝ) (side_length : ℝ) : Prop :=
  area = side_length^2 ∧ side_length = 7

def perimeter_of_region (num_squares : ℕ) (side_length : ℝ) (perimeter : ℝ) : Prop :=
  perimeter = 8 * side_length + 2 * side_length ∧ perimeter = 70

-- Statement to prove
theorem perimeter_of_region_is_70 :
  ∀ (total_area : ℝ) (num_squares : ℕ), 
    area_of_region total_area num_squares →
    ∃ (side_length : ℝ) (perimeter : ℝ), 
      side_length_of_square (total_area / num_squares) side_length ∧
      perimeter_of_region num_squares side_length perimeter :=
by {
  sorry
}

end perimeter_of_region_is_70_l10_10336


namespace maximum_pizzas_baked_on_Friday_l10_10833

def george_bakes := 
  let total_pizzas : ℕ := 1000
  let monday_pizzas := total_pizzas * 7 / 10
  let tuesday_pizzas := if monday_pizzas * 4 / 5 < monday_pizzas * 9 / 10 
                        then monday_pizzas * 4 / 5 
                        else monday_pizzas * 9 / 10
  let wednesday_pizzas := if tuesday_pizzas * 4 / 5 < tuesday_pizzas * 9 / 10 
                          then tuesday_pizzas * 4 / 5 
                          else tuesday_pizzas * 9 / 10
  let thursday_pizzas := if wednesday_pizzas * 4 / 5 < wednesday_pizzas * 9 / 10 
                         then wednesday_pizzas * 4 / 5 
                         else wednesday_pizzas * 9 / 10
  let friday_pizzas := if thursday_pizzas * 4 / 5 < thursday_pizzas * 9 / 10 
                       then thursday_pizzas * 4 / 5 
                       else thursday_pizzas * 9 / 10
  friday_pizzas

theorem maximum_pizzas_baked_on_Friday : george_bakes = 2 := by
  sorry

end maximum_pizzas_baked_on_Friday_l10_10833


namespace jogger_distance_ahead_l10_10315

def speed_jogger_kmph : ℕ := 9
def speed_train_kmph : ℕ := 45
def length_train_m : ℕ := 120
def time_to_pass_jogger_s : ℕ := 36

theorem jogger_distance_ahead :
  let relative_speed_mps := (speed_train_kmph - speed_jogger_kmph) * 1000 / 3600
  let distance_covered_m := relative_speed_mps * time_to_pass_jogger_s
  let jogger_distance_ahead : ℕ := distance_covered_m - length_train_m
  jogger_distance_ahead = 240 :=
by
  sorry

end jogger_distance_ahead_l10_10315


namespace larger_screen_diagonal_length_l10_10373

theorem larger_screen_diagonal_length :
  (∃ d : ℝ, (∀ a : ℝ, a = 16 → d^2 = 2 * (a^2 + 34)) ∧ d = Real.sqrt 580) :=
by
  sorry

end larger_screen_diagonal_length_l10_10373


namespace proof_problem1_proof_problem2_proof_problem3_proof_problem4_l10_10449

noncomputable def problem1 : Prop := 
  2500 * (1/10000) = 0.25

noncomputable def problem2 : Prop := 
  20 * (1/100) = 0.2

noncomputable def problem3 : Prop := 
  45 * (1/60) = 3/4

noncomputable def problem4 : Prop := 
  1250 * (1/10000) = 0.125

theorem proof_problem1 : problem1 := by
  sorry

theorem proof_problem2 : problem2 := by
  sorry

theorem proof_problem3 : problem3 := by
  sorry

theorem proof_problem4 : problem4 := by
  sorry

end proof_problem1_proof_problem2_proof_problem3_proof_problem4_l10_10449


namespace correct_result_l10_10269

-- Define the conditions
variables (x : ℤ)
axiom condition1 : (x - 27 + 19 = 84)

-- Define the goal
theorem correct_result : x - 19 + 27 = 100 :=
  sorry

end correct_result_l10_10269


namespace games_that_didnt_work_l10_10953

variable (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (good_games : ℕ)

theorem games_that_didnt_work
  (h₁ : games_from_friend = 2)
  (h₂ : games_from_garage_sale = 2)
  (h₃ : good_games = 2) :
  (games_from_friend + games_from_garage_sale - good_games) = 2 :=
by 
  sorry

end games_that_didnt_work_l10_10953


namespace rides_first_day_l10_10151

variable (total_rides : ℕ) (second_day_rides : ℕ)

theorem rides_first_day (h1 : total_rides = 7) (h2 : second_day_rides = 3) : total_rides - second_day_rides = 4 :=
by
  sorry

end rides_first_day_l10_10151


namespace min_S_n_at_24_l10_10062

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 49

noncomputable def S_n (n : ℕ) : ℤ := (n : ℤ) * (2 * n - 48)

theorem min_S_n_at_24 : (∀ n : ℕ, n > 0 → S_n n ≥ S_n 24) ∧ S_n 24 < S_n 25 :=
by 
  sorry

end min_S_n_at_24_l10_10062


namespace elaine_earnings_l10_10808

variable (E P : ℝ)
variable (H1 : 0.30 * E * (1 + P / 100) = 2.025 * 0.20 * E)

theorem elaine_earnings : P = 35 :=
by
  -- We assume the conditions here and the proof is skipped by sorry.
  sorry

end elaine_earnings_l10_10808


namespace work_completion_days_l10_10202

theorem work_completion_days (A B : Type) (A_work_rate B_work_rate : ℝ) :
  (1 / 16 : ℝ) = (1 / 20) + A_work_rate → B_work_rate = (1 / 80) := by
  sorry

end work_completion_days_l10_10202


namespace intersection_eq_l10_10476

def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x < 2}

theorem intersection_eq : M ∩ N = {-1, 0, 1} :=
by
  sorry

end intersection_eq_l10_10476


namespace negation_of_p_l10_10657

open Real

-- Define the original proposition p
def p := ∀ x : ℝ, 0 < x → x^2 > log x

-- State the theorem with its negation
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, 0 < x ∧ x^2 ≤ log x :=
by
  sorry

end negation_of_p_l10_10657


namespace find_x_for_g_l10_10406

noncomputable def g (x : ℝ) : ℝ := (↑((x + 5) / 3) : ℝ)^(1/3 : ℝ)

theorem find_x_for_g :
  ∃ x : ℝ, g (3 * x) = 3 * g x ↔ x = -65 / 12 :=
by
  sorry

end find_x_for_g_l10_10406


namespace rectangle_area_given_perimeter_l10_10661

theorem rectangle_area_given_perimeter (x : ℝ) (h_perim : 8 * x = 160) : (2 * x) * (2 * x) = 1600 := by
  -- Definitions derived from conditions
  let length := 2 * x
  let width := 2 * x
  -- Proof transformed to a Lean statement
  have h1 : length = 40 := by sorry
  have h2 : width = 40 := by sorry
  have h_area : length * width = 1600 := by sorry
  exact h_area

end rectangle_area_given_perimeter_l10_10661


namespace four_digit_numbers_permutations_l10_10560

theorem four_digit_numbers_permutations (a b : ℕ) (h1 : a = 3) (h2 : b = 0) : 
  (if a = 3 ∧ b = 0 then 3 else 0) = 3 :=
by
  sorry

end four_digit_numbers_permutations_l10_10560


namespace cylinder_ratio_l10_10754

theorem cylinder_ratio (h r : ℝ) (h_eq : h = 2 * Real.pi * r) : 
  h / r = 2 * Real.pi := 
by 
  sorry

end cylinder_ratio_l10_10754


namespace cost_price_of_article_l10_10350

theorem cost_price_of_article (M : ℝ) (SP : ℝ) (C : ℝ) 
  (hM : M = 65)
  (hSP : SP = 0.95 * M)
  (hProfit : SP = 1.30 * C) : 
  C = 47.50 :=
by 
  sorry

end cost_price_of_article_l10_10350


namespace arithmetic_expressions_correctness_l10_10289

theorem arithmetic_expressions_correctness :
  ((∀ (a b c : ℚ), (a + b) + c = a + (b + c)) ∧
   (∃ (a b c : ℚ), (a - b) - c ≠ a - (b - c)) ∧
   (∀ (a b c : ℚ), (a * b) * c = a * (b * c)) ∧
   (∃ (a b c : ℚ), a / b / c ≠ a / (b / c))) :=
by
  sorry

end arithmetic_expressions_correctness_l10_10289


namespace polygon_angle_ratio_pairs_count_l10_10281

theorem polygon_angle_ratio_pairs_count :
  ∃ (m n : ℕ), (∃ (k : ℕ), (k > 0) ∧ (180 - 360 / ↑m) / (180 - 360 / ↑n) = 4 / 3
  ∧ Prime n ∧ (m - 6) * (n + 8) = 48 ∧ 
  ∃! (m n : ℕ), (180 - 360 / ↑m = (4 * (180 - 360 / ↑n)) / 3)) :=
sorry  -- Proof omitted, providing only the statement

end polygon_angle_ratio_pairs_count_l10_10281


namespace time_to_cover_length_l10_10861

def speed_escalator : ℝ := 10
def speed_person : ℝ := 4
def length_escalator : ℝ := 112

theorem time_to_cover_length :
  (length_escalator / (speed_escalator + speed_person) = 8) :=
by
  sorry

end time_to_cover_length_l10_10861


namespace total_budget_is_correct_l10_10561

-- Define the costs of TV, fridge, and computer based on the given conditions
def cost_tv : ℕ := 600
def cost_computer : ℕ := 250
def cost_fridge : ℕ := cost_computer + 500

-- Statement to prove the total budget
theorem total_budget_is_correct : cost_tv + cost_computer + cost_fridge = 1600 :=
by
  sorry

end total_budget_is_correct_l10_10561


namespace graph_passes_through_quadrants_l10_10115

theorem graph_passes_through_quadrants :
  ∀ x : ℝ, (4 * x + 2 > 0 → (x > 0)) ∨ (4 * x + 2 > 0 → (x < 0)) ∨ (4 * x + 2 < 0 → (x < 0)) :=
by
  intro x
  sorry

end graph_passes_through_quadrants_l10_10115


namespace total_fuel_two_weeks_l10_10514

def fuel_used_this_week : ℝ := 15
def percentage_less_last_week : ℝ := 0.2
def fuel_used_last_week : ℝ := fuel_used_this_week * (1 - percentage_less_last_week)
def total_fuel_used : ℝ := fuel_used_this_week + fuel_used_last_week

theorem total_fuel_two_weeks : total_fuel_used = 27 := 
by
  -- Placeholder for the proof
  sorry

end total_fuel_two_weeks_l10_10514


namespace avg_salary_difference_l10_10056

theorem avg_salary_difference (factory_payroll : ℕ) (factory_workers : ℕ) (office_payroll : ℕ) (office_workers : ℕ)
  (h1 : factory_payroll = 30000) (h2 : factory_workers = 15)
  (h3 : office_payroll = 75000) (h4 : office_workers = 30) :
  (office_payroll / office_workers) - (factory_payroll / factory_workers) = 500 := by
  sorry

end avg_salary_difference_l10_10056


namespace triangle_minimum_perimeter_l10_10802

/--
In a triangle ABC where sides have integer lengths such that no two sides are equal, let ω be a circle with its center at the incenter of ΔABC. Suppose one excircle is tangent to AB and internally tangent to ω, while excircles tangent to AC and BC are externally tangent to ω.
Prove that the minimum possible perimeter of ΔABC is 12.
-/
theorem triangle_minimum_perimeter {a b c : ℕ} (h1 : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
    (h2 : ∀ (r rA rB rC s : ℝ),
      rA = r * s / (s - a) → rB = r * s / (s - b) → rC = r * s / (s - c) →
      r + rA = rB ∧ r + rA = rC) :
  a + b + c = 12 :=
sorry

end triangle_minimum_perimeter_l10_10802


namespace find_num_3_year_olds_l10_10772

noncomputable def num_4_year_olds := 20
noncomputable def num_5_year_olds := 15
noncomputable def num_6_year_olds := 22
noncomputable def average_class_size := 35
noncomputable def num_students_class1 (num_3_year_olds : ℕ) := num_3_year_olds + num_4_year_olds
noncomputable def num_students_class2 := num_5_year_olds + num_6_year_olds
noncomputable def total_students (num_3_year_olds : ℕ) := num_students_class1 num_3_year_olds + num_students_class2

theorem find_num_3_year_olds (num_3_year_olds : ℕ) : 
  (total_students num_3_year_olds) / 2 = average_class_size → num_3_year_olds = 13 :=
by
  sorry

end find_num_3_year_olds_l10_10772


namespace road_length_l10_10932

theorem road_length 
  (D : ℕ) (N1 : ℕ) (t : ℕ) (d1 : ℝ) (N_extra : ℝ) 
  (h1 : D = 300) (h2 : N1 = 35) (h3 : t = 100) (h4 : d1 = 2.5) (h5 : N_extra = 52.5) : 
  ∃ L : ℝ, L = 3 := 
by {
  sorry
}

end road_length_l10_10932


namespace least_add_to_divisible_least_subtract_to_divisible_l10_10301

theorem least_add_to_divisible (n : ℤ) (d : ℤ) (r : ℤ) (a : ℤ) : 
  n = 1100 → d = 37 → r = n % d → a = d - r → (n + a) % d = 0 :=
by sorry

theorem least_subtract_to_divisible (n : ℤ) (d : ℤ) (r : ℤ) (s : ℤ) : 
  n = 1100 → d = 37 → r = n % d → s = r → (n - s) % d = 0 :=
by sorry

end least_add_to_divisible_least_subtract_to_divisible_l10_10301


namespace general_term_formula_of_arithmetic_seq_l10_10145

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem general_term_formula_of_arithmetic_seq 
  (a : ℕ → ℝ) (h_arith : arithmetic_seq a)
  (h1 : a 3 * a 7 = -16) 
  (h2 : a 4 + a 6 = 0) :
  (∀ n : ℕ, a n = 2 * n - 10) ∨ (∀ n : ℕ, a n = -2 * n + 10) :=
by
  sorry

end general_term_formula_of_arithmetic_seq_l10_10145


namespace identify_vanya_l10_10100

structure Twin :=
(name : String)
(truth_teller : Bool)

def is_vanya_truth_teller (twin : Twin) (vanya vitya : Twin) : Prop :=
  twin = vanya ∧ twin.truth_teller ∨ twin = vitya ∧ ¬twin.truth_teller

theorem identify_vanya
  (vanya vitya : Twin)
  (h_vanya : vanya.name = "Vanya")
  (h_vitya : vitya.name = "Vitya")
  (h_one_truth : ∃ t : Twin, t = vanya ∨ t = vitya ∧ (t.truth_teller = true ∨ t.truth_teller = false))
  (h_one_lie : ∀ t : Twin, t = vanya ∨ t = vitya → ¬(t.truth_teller = true ∧ t = vitya) ∧ ¬(t.truth_teller = false ∧ t = vanya)) :
  ∀ twin : Twin, twin = vanya ∨ twin = vitya →
  (is_vanya_truth_teller twin vanya vitya ↔ (twin = vanya ∧ twin.truth_teller = true)) :=
by
  sorry

end identify_vanya_l10_10100


namespace find_a_l10_10343

theorem find_a (a b c : ℝ) (h1 : ∀ x, x = 2 → y = 5) (h2 : ∀ x, x = 3 → y = 7) :
  a = 2 :=
sorry

end find_a_l10_10343


namespace merge_coins_n_ge_3_merge_coins_n_eq_2_l10_10520

-- For Part 1
theorem merge_coins_n_ge_3 (n : ℕ) (hn : n ≥ 3) :
  ∃ (m : ℕ), m = 1 ∨ m = 2 :=
sorry

-- For Part 2
theorem merge_coins_n_eq_2 (r s : ℕ) :
  ∃ (k : ℕ), r + s = 2^k * Nat.gcd r s :=
sorry

end merge_coins_n_ge_3_merge_coins_n_eq_2_l10_10520


namespace find_a5_l10_10139

-- Define an arithmetic sequence with a given common difference
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Define that three terms form a geometric sequence
def geometric_sequence (x y z : ℝ) := y^2 = x * z

-- Given conditions for the problem
def a₁ : ℝ := 1  -- found from the geometric sequence condition
def d : ℝ := 2

-- The definition of the sequence {a_n} based on the common difference
noncomputable def a_n (n : ℕ) : ℝ := a₁ + n * d

-- Given that a_1, a_2, a_5 form a geometric sequence
axiom geo_progression : geometric_sequence a₁ (a_n 1) (a_n 4)

-- The proof goal
theorem find_a5 : a_n 4 = 9 :=
by
  -- the proof is skipped
  sorry

end find_a5_l10_10139


namespace a_investment_l10_10256

theorem a_investment
  (b_investment : ℝ) (c_investment : ℝ) (c_share_profit : ℝ) (total_profit : ℝ)
  (h1 : b_investment = 45000)
  (h2 : c_investment = 50000)
  (h3 : c_share_profit = 36000)
  (h4 : total_profit = 90000) :
  ∃ A : ℝ, A = 30000 :=
by {
  sorry
}

end a_investment_l10_10256


namespace quadratic_polynomials_perfect_square_l10_10460

variables {x y p q a b c : ℝ}

theorem quadratic_polynomials_perfect_square (h1 : ∃ a, x^2 + p * x + q = (x + a) * (x + a))
  (h2 : ∃ a b, a^2 * x^2 + 2 * b^2 * x * y + c^2 * y^2 = (a * x + b * y) * (a * x + b * y)) :
  q = (p^2 / 4) ∧ b^2 = a * c :=
by
  sorry

end quadratic_polynomials_perfect_square_l10_10460


namespace evaluate_expression_l10_10454

theorem evaluate_expression 
  (a c : ℝ)
  (h : a + c = 9) :
  (a * (-1)^2 + (-1) + c) = 8 := 
by 
  sorry

end evaluate_expression_l10_10454


namespace ratio_w_to_y_l10_10150

variable (w x y z : ℚ)
variable (h1 : w / x = 5 / 4)
variable (h2 : y / z = 5 / 3)
variable (h3 : z / x = 1 / 5)

theorem ratio_w_to_y : w / y = 15 / 4 := sorry

end ratio_w_to_y_l10_10150


namespace calc_g_x_plus_3_l10_10960

def g (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem calc_g_x_plus_3 (x : ℝ) : g (x + 3) = x^2 + 3*x + 2 :=
by
  sorry

end calc_g_x_plus_3_l10_10960


namespace probability_of_female_selection_probability_of_male_host_selection_l10_10209

/-!
In a competition, there are eight contestants consisting of five females and three males.
If three contestants are chosen randomly to progress to the next round, what is the 
probability that all selected contestants are female? Additionally, from those who 
do not proceed, one is selected as a host. What is the probability that this host is male?
-/

noncomputable def number_of_ways_select_3_from_8 : ℕ := Nat.choose 8 3

noncomputable def number_of_ways_select_3_females_from_5 : ℕ := Nat.choose 5 3

noncomputable def probability_all_3_females : ℚ := number_of_ways_select_3_females_from_5 / number_of_ways_select_3_from_8

noncomputable def number_of_remaining_contestants : ℕ := 8 - 3

noncomputable def number_of_males_remaining : ℕ := 3 - 1

noncomputable def number_of_ways_select_1_male_from_2 : ℕ := Nat.choose 2 1

noncomputable def number_of_ways_select_1_from_5 : ℕ := Nat.choose 5 1

noncomputable def probability_host_is_male : ℚ := number_of_ways_select_1_male_from_2 / number_of_ways_select_1_from_5

theorem probability_of_female_selection : probability_all_3_females = 5 / 28 := by
  sorry

theorem probability_of_male_host_selection : probability_host_is_male = 2 / 5 := by
  sorry

end probability_of_female_selection_probability_of_male_host_selection_l10_10209


namespace inequality_solution_range_l10_10012

theorem inequality_solution_range (x : ℝ) : (x^2 + 3*x - 10 < 0) ↔ (-5 < x ∧ x < 2) :=
by
  sorry

end inequality_solution_range_l10_10012


namespace tan_domain_correct_l10_10911

noncomputable def domain_tan : Set ℝ := {x | ∃ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 4}

def is_domain_correct : Prop :=
  ∀ x : ℝ, x ∈ domain_tan ↔ (∃ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 4)

-- Statement of the problem in Lean 4
theorem tan_domain_correct : is_domain_correct :=
  sorry

end tan_domain_correct_l10_10911


namespace Mark_hours_left_l10_10744

theorem Mark_hours_left (sick_days vacation_days : ℕ) (hours_per_day : ℕ) 
  (h1 : sick_days = 10) (h2 : vacation_days = 10) (h3 : hours_per_day = 8) 
  (used_sick_days : ℕ) (used_vacation_days : ℕ) 
  (h4 : used_sick_days = sick_days / 2) (h5 : used_vacation_days = vacation_days / 2) 
  : (sick_days + vacation_days - used_sick_days - used_vacation_days) * hours_per_day = 80 :=
by
  sorry

end Mark_hours_left_l10_10744


namespace greatest_visible_unit_cubes_from_one_point_12_l10_10266

def num_unit_cubes (n : ℕ) : ℕ := n * n * n

def face_count (n : ℕ) : ℕ := n * n

def edge_count (n : ℕ) : ℕ := n

def visible_unit_cubes_from_one_point (n : ℕ) : ℕ :=
  let faces := 3 * face_count n
  let edges := 3 * (edge_count n - 1)
  let corner := 1
  faces - edges + corner

theorem greatest_visible_unit_cubes_from_one_point_12 :
  visible_unit_cubes_from_one_point 12 = 400 :=
  by
  sorry

end greatest_visible_unit_cubes_from_one_point_12_l10_10266


namespace eggs_processed_per_day_l10_10762

/-- In a certain egg-processing plant, every egg must be inspected, and is either accepted for processing or rejected. For every 388 eggs accepted for processing, 12 eggs are rejected.

If, on a particular day, 37 additional eggs were accepted, but the overall number of eggs inspected remained the same, the ratio of those accepted to those rejected would be 405 to 3.

Prove that the number of eggs processed per day, given these conditions, is 125763.
-/
theorem eggs_processed_per_day : ∃ (E : ℕ), (∃ (R : ℕ), 38 * R = 3 * (E - 37) ∧  E = 32 * R + E / 33 ) ∧ (E = 125763) :=
sorry

end eggs_processed_per_day_l10_10762


namespace distance_between_city_A_and_B_is_180_l10_10931

theorem distance_between_city_A_and_B_is_180
  (D : ℝ)
  (h1 : ∀ T_C : ℝ, T_C = D / 30)
  (h2 : ∀ T_D : ℝ, T_D = T_C - 1)
  (h3 : ∀ V_D : ℝ, V_D > 36 → T_D = D / V_D) :
  D = 180 := 
by
  sorry

end distance_between_city_A_and_B_is_180_l10_10931


namespace maggie_earnings_proof_l10_10909

def rate_per_subscription : ℕ := 5
def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_neighbor1 : ℕ := 2
def subscriptions_to_neighbor2 : ℕ := 2 * subscriptions_to_neighbor1
def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_neighbor1 + subscriptions_to_neighbor2
def total_earnings : ℕ := total_subscriptions * rate_per_subscription

theorem maggie_earnings_proof : total_earnings = 55 := by
  sorry

end maggie_earnings_proof_l10_10909


namespace prove_inequality_l10_10292

-- Define the function properties
variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Function properties as given in the problem
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def decreasing_on_nonneg (f : ℝ → ℝ) := ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

-- The main theorem statement
theorem prove_inequality (h_even : even_function f) (h_dec : decreasing_on_nonneg f) :
  f (-3 / 4) ≥ f (a^2 - a + 1) :=
sorry

end prove_inequality_l10_10292


namespace find_f_neg2_l10_10165

theorem find_f_neg2 (a b : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = x^5 + a*x^3 + x^2 + b*x + 2) (h₂ : f 2 = 3) : f (-2) = 9 :=
by
  sorry

end find_f_neg2_l10_10165


namespace determine_identity_l10_10037

-- Define the types for human and vampire
inductive Being
| human
| vampire

-- Define the responses for sanity questions
def claims_sanity (b : Being) : Prop :=
  match b with
  | Being.human   => true
  | Being.vampire => false

-- Proof statement: Given that a human always claims sanity and a vampire always claims insanity,
-- asking "Are you sane?" will determine their identity. 
theorem determine_identity (b : Being) (h : b = Being.human ↔ claims_sanity b = true) : 
  ((claims_sanity b = true) → b = Being.human) ∧ ((claims_sanity b = false) → b = Being.vampire) :=
sorry

end determine_identity_l10_10037


namespace gcd_2023_1991_l10_10383

theorem gcd_2023_1991 : Nat.gcd 2023 1991 = 1 :=
by
  sorry

end gcd_2023_1991_l10_10383


namespace expand_product_l10_10817

theorem expand_product (x : ℝ) :
  (2 * x^2 - 3 * x + 5) * (x^2 + 4 * x + 3) = 2 * x^4 + 5 * x^3 - x^2 + 11 * x + 15 :=
by
  -- Proof to be filled in
  sorry

end expand_product_l10_10817


namespace distance_AD_35_l10_10598

-- Definitions based on conditions
variables (A B C D : Point)
variable (distance : Point → Point → ℝ)
variable (angle : Point → Point → Point → ℝ)
variable (dueEast : Point → Point → Prop)
variable (northOf : Point → Point → Prop)

-- Conditions
def conditions : Prop :=
  dueEast A B ∧
  angle A B C = 90 ∧
  distance A C = 15 * Real.sqrt 3 ∧
  angle B A C = 30 ∧
  northOf D C ∧
  distance C D = 10

-- The question: Proving the distance between points A and D
theorem distance_AD_35 (h : conditions A B C D distance angle dueEast northOf) :
  distance A D = 35 :=
sorry

end distance_AD_35_l10_10598


namespace avg_move_to_california_l10_10979

noncomputable def avg_people_per_hour (total_people : ℕ) (total_days : ℕ) : ℕ :=
  let total_hours := total_days * 24
  let avg_per_hour := total_people / total_hours
  let remainder := total_people % total_hours
  if remainder * 2 < total_hours then avg_per_hour else avg_per_hour + 1

theorem avg_move_to_california : avg_people_per_hour 3500 5 = 29 := by
  sorry

end avg_move_to_california_l10_10979


namespace q_one_eq_five_l10_10693

variable (q : ℝ → ℝ)
variable (h : q 1 = 5)

theorem q_one_eq_five : q 1 = 5 :=
by sorry

end q_one_eq_five_l10_10693


namespace pieces_per_package_l10_10297

-- Define Robin's packages
def numGumPackages := 28
def numCandyPackages := 14

-- Define total number of pieces
def totalPieces := 7

-- Define the total number of packages
def totalPackages := numGumPackages + numCandyPackages

-- Define the expected number of pieces per package as the theorem to prove
theorem pieces_per_package : (totalPieces / totalPackages) = 1/6 := by
  sorry

end pieces_per_package_l10_10297


namespace initial_punch_amount_l10_10732

theorem initial_punch_amount (P : ℝ) (h1 : 16 = (P / 2 + 2 + 12)) : P = 4 :=
by
  sorry

end initial_punch_amount_l10_10732


namespace solve_for_a_l10_10894

theorem solve_for_a (a x : ℝ) (h : x = 1 ∧ 2 * a * x - 2 = a + 3) : a = 5 :=
by
  sorry

end solve_for_a_l10_10894


namespace questions_ratio_l10_10710

theorem questions_ratio (R A : ℕ) (H₁ : R + 6 + A = 24) :
  (R, 6, A) = (R, 6, A) :=
sorry

end questions_ratio_l10_10710


namespace simplify_expression_l10_10603

theorem simplify_expression (x : ℝ) : 120 * x - 72 * x + 15 * x - 9 * x = 54 * x := 
by
  sorry

end simplify_expression_l10_10603


namespace proof_problem_l10_10930

-- Triangle and Point Definitions
variables {A B C P : Type}
variables (BC : ℝ) (a b c : ℝ) (PA PB PC : ℝ)

-- Conditions: Triangle ABC with angle A = 90 degrees and P on BC
def is_right_triangle (A B C : Type) (a b c : ℝ) (BC : ℝ) (angleA : ℝ := 90) : Prop :=
a^2 + b^2 = c^2 ∧ c = BC

def on_hypotenuse (P : Type) (BC : ℝ) (PB PC : ℝ) : Prop :=
PB + PC = BC

-- The proof problem
theorem proof_problem (A B C P : Type) 
  (BC : ℝ) (a b c : ℝ) (PA PB PC : ℝ)
  (h1 : is_right_triangle A B C a b c BC)
  (h2 : on_hypotenuse P BC PB PC) :
  (a^2 / PC + b^2 / PB) ≥ (BC^3 / (PA^2 + PB * PC)) := 
sorry

end proof_problem_l10_10930


namespace find_k_l10_10799

def a : ℕ := 786
def b : ℕ := 74
def c : ℝ := 1938.8

theorem find_k (k : ℝ) : (a * b) / k = c → k = 30 :=
by
  intro h
  sorry

end find_k_l10_10799


namespace B_subscribed_fraction_correct_l10_10611

-- Define the total capital and the shares of A, C
variables (X : ℝ) (profit : ℝ) (A_share : ℝ) (C_share : ℝ)

-- Define the conditions as given in the problem
def A_capital_share := 1 / 3
def C_capital_share := 1 / 5
def total_profit := 2430
def A_profit_share := 810

-- Define the calculation of B's share
def B_capital_share := 1 - (A_capital_share + C_capital_share)

-- Define the expected correct answer for B's share
def expected_B_share := 7 / 15

-- Theorem statement
theorem B_subscribed_fraction_correct :
  B_capital_share = expected_B_share :=
by
  sorry

end B_subscribed_fraction_correct_l10_10611


namespace remainder_of_2n_div_7_l10_10234

theorem remainder_of_2n_div_7 (n : ℤ) (k : ℤ) (h : n = 7 * k + 2) : (2 * n) % 7 = 4 :=
by
  sorry

end remainder_of_2n_div_7_l10_10234


namespace circle_center_and_radius_l10_10304

theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4 * y - 1 = 0 ↔ (x, y) = (0, 2) ∧ 5 = (0 - x)^2 + (2 - y)^2 :=
by sorry

end circle_center_and_radius_l10_10304


namespace john_writes_book_every_2_months_l10_10614

theorem john_writes_book_every_2_months
    (years_writing : ℕ)
    (average_earnings_per_book : ℕ)
    (total_earnings : ℕ)
    (H1 : years_writing = 20)
    (H2 : average_earnings_per_book = 30000)
    (H3 : total_earnings = 3600000) : 
    (years_writing * 12 / (total_earnings / average_earnings_per_book)) = 2 :=
by
    sorry

end john_writes_book_every_2_months_l10_10614


namespace x_minus_y_eq_neg3_l10_10001

theorem x_minus_y_eq_neg3 (x y : ℝ) (i : ℂ) (h1 : x * i + 2 = y - i) (h2 : i^2 = -1) : x - y = -3 := 
  sorry

end x_minus_y_eq_neg3_l10_10001


namespace Alice_more_nickels_l10_10641

-- Define quarters each person has
def Alice_quarters (q : ℕ) : ℕ := 10 * q + 2
def Bob_quarters (q : ℕ) : ℕ := 2 * q + 10

-- Prove that Alice has 40(q - 1) more nickels than Bob
theorem Alice_more_nickels (q : ℕ) : 
  (5 * (Alice_quarters q - Bob_quarters q)) = 40 * (q - 1) :=
by
  sorry

end Alice_more_nickels_l10_10641


namespace tina_mother_took_out_coins_l10_10199

theorem tina_mother_took_out_coins :
  let first_hour := 20
  let next_two_hours := 30 * 2
  let fourth_hour := 40
  let total_coins := first_hour + next_two_hours + fourth_hour
  let coins_left_after_fifth_hour := 100
  let coins_taken_out := total_coins - coins_left_after_fifth_hour
  coins_taken_out = 20 :=
by
  sorry

end tina_mother_took_out_coins_l10_10199


namespace number_of_draws_l10_10923

-- Definition of the competition conditions
def competition_conditions (A B C D E : ℕ) : Prop :=
  A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧
  (A = B ∨ B = C ∨ C = D ∨ D = E) ∧
  15 ∣ (10000 * A + 1000 * B + 100 * C + 10 * D + E)

-- The main theorem stating the number of draws
theorem number_of_draws :
  ∃ (A B C D E : ℕ), competition_conditions A B C D E ∧ 
  (∃ (draws : ℕ), draws = 3) :=
by
  sorry

end number_of_draws_l10_10923


namespace find_n_l10_10582

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (n : ℕ)

def isArithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sumTo (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem find_n 
  (h_arith : isArithmeticSeq a)
  (h_a2 : a 2 = 2) 
  (h_S_diff : ∀ n, n > 3 → S n - S (n - 3) = 54)
  (h_Sn : S n = 100)
  : n = 10 := 
by
  sorry

end find_n_l10_10582


namespace profit_percentage_l10_10544

theorem profit_percentage (SP : ℝ) (CP : ℝ) (hSP : SP = 100) (hCP : CP = 83.33) :
    (SP - CP) / CP * 100 = 20 :=
by
  rw [hSP, hCP]
  norm_num
  sorry

end profit_percentage_l10_10544


namespace polynomial_value_l10_10857

theorem polynomial_value (x : ℝ) (h : 3 * x^2 - x = 1) : 6 * x^3 + 7 * x^2 - 5 * x + 2008 = 2011 :=
by
  sorry

end polynomial_value_l10_10857


namespace field_area_l10_10329

def length : ℝ := 80 -- Length of the uncovered side
def total_fencing : ℝ := 97 -- Total fencing required

theorem field_area : ∃ (W L : ℝ), L = length ∧ 2 * W + L = total_fencing ∧ L * W = 680 := by
  sorry

end field_area_l10_10329


namespace solve_inequality_l10_10414

noncomputable def within_interval (x : ℝ) : Prop :=
  x > -3 ∧ x < 5

theorem solve_inequality (x : ℝ) :
  (x^3 - 125) / (x + 3) < 0 ↔ within_interval x :=
sorry

end solve_inequality_l10_10414


namespace tires_in_parking_lot_l10_10479

theorem tires_in_parking_lot (num_cars : ℕ) (regular_tires_per_car spare_tire : ℕ) (h1 : num_cars = 30) (h2 : regular_tires_per_car = 4) (h3 : spare_tire = 1) :
  num_cars * (regular_tires_per_car + spare_tire) = 150 :=
by
  sorry

end tires_in_parking_lot_l10_10479


namespace largest_value_of_n_l10_10922

theorem largest_value_of_n :
  ∃ (n : ℕ) (X Y Z : ℕ),
    n = 25 * X + 5 * Y + Z ∧
    n = 81 * Z + 9 * Y + X ∧
    X < 5 ∧ Y < 5 ∧ Z < 5 ∧
    n = 121 := by
  sorry

end largest_value_of_n_l10_10922


namespace service_cleaning_fee_percentage_is_correct_l10_10187

noncomputable def daily_rate : ℝ := 125
noncomputable def pet_fee : ℝ := 100
noncomputable def duration : ℕ := 14
noncomputable def security_deposit_percentage : ℝ := 0.5
noncomputable def security_deposit : ℝ := 1110

noncomputable def total_expected_cost : ℝ := (daily_rate * duration) + pet_fee
noncomputable def entire_bill : ℝ := security_deposit / security_deposit_percentage
noncomputable def service_cleaning_fee : ℝ := entire_bill - total_expected_cost

theorem service_cleaning_fee_percentage_is_correct : 
  (service_cleaning_fee / entire_bill) * 100 = 16.67 :=
by 
  sorry

end service_cleaning_fee_percentage_is_correct_l10_10187


namespace solve_for_k_l10_10085

theorem solve_for_k (k : ℝ) (h : 2 * (5:ℝ)^2 + 3 * (5:ℝ) - k = 0) : k = 65 := 
by
  sorry

end solve_for_k_l10_10085


namespace increase_in_cases_second_day_l10_10458

-- Define the initial number of cases.
def initial_cases : ℕ := 2000

-- Define the number of recoveries on the second day.
def recoveries_day2 : ℕ := 50

-- Define the number of new cases on the third day and the recoveries on the third day.
def new_cases_day3 : ℕ := 1500
def recoveries_day3 : ℕ := 200

-- Define the total number of positive cases after the third day.
def total_cases_day3 : ℕ := 3750

-- Lean statement to prove the increase in cases on the second day is 750.
theorem increase_in_cases_second_day : 
  ∃ x : ℕ, initial_cases + x - recoveries_day2 + new_cases_day3 - recoveries_day3 = total_cases_day3 ∧ x = 750 :=
by
  sorry

end increase_in_cases_second_day_l10_10458


namespace inequality_system_solution_l10_10770

theorem inequality_system_solution (x: ℝ) (h1: 5 * x - 2 < 3 * (x + 2)) (h2: (2 * x - 1) / 3 - (5 * x + 1) / 2 <= 1) : 
  -1 ≤ x ∧ x < 4 :=
sorry

end inequality_system_solution_l10_10770


namespace cube_root_1728_simplified_l10_10519

theorem cube_root_1728_simplified :
  let a := 12
  let b := 1
  a + b = 13 :=
by
  sorry

end cube_root_1728_simplified_l10_10519


namespace min_value_4a2_b2_plus_1_div_2a_minus_b_l10_10715

variable (a b : ℝ)

theorem min_value_4a2_b2_plus_1_div_2a_minus_b (h1 : 0 < a) (h2 : 0 < b)
  (h3 : a > b) (h4 : a * b = 1 / 2) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 3 ∧ (∀ x y : ℝ, 0 < x → 0 < y → x > y → x * y = 1 / 2 → (4 * x^2 + y^2 + 1) / (2 * x - y) ≥ c) :=
sorry

end min_value_4a2_b2_plus_1_div_2a_minus_b_l10_10715


namespace property_P_difference_l10_10076

noncomputable def f (n : ℕ) : ℕ :=
  if n % 2 = 0 then 
    6 * 2^(n / 2) - n - 5 
  else 
    4 * 2^((n + 1) / 2) - n - 5

theorem property_P_difference : f 9 - f 8 = 31 := by
  sorry

end property_P_difference_l10_10076


namespace find_x_l10_10228

theorem find_x (x : ℝ) (h : ⌊x⌋ + x = 15/4) : x = 7/4 :=
sorry

end find_x_l10_10228


namespace andy_starting_problem_l10_10839

theorem andy_starting_problem (end_num problems_solved : ℕ) 
  (h_end : end_num = 125) (h_solved : problems_solved = 46) : 
  end_num - problems_solved + 1 = 80 := 
by
  sorry

end andy_starting_problem_l10_10839


namespace shopkeeper_profit_percentage_l10_10386

theorem shopkeeper_profit_percentage
  (C : ℝ) -- The cost price of one article
  (cost_price_50 : ℝ := 50 * C) -- The cost price of 50 articles
  (cost_price_70 : ℝ := 70 * C) -- The cost price of 70 articles
  (selling_price_50 : ℝ := 70 * C) -- Selling price of 50 articles is the cost price of 70 articles
  :
  ∃ (P : ℝ), P = 40 :=
by
  sorry

end shopkeeper_profit_percentage_l10_10386


namespace divisibility_by_1897_l10_10355

theorem divisibility_by_1897 (n : ℕ) : 1897 ∣ (2903 ^ n - 803 ^ n - 464 ^ n + 261 ^ n) :=
sorry

end divisibility_by_1897_l10_10355


namespace distinct_prime_sum_product_l10_10474

open Nat

-- Definitions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- The problem statement
theorem distinct_prime_sum_product (a b c : ℕ) (h1 : is_prime a) (h2 : is_prime b) 
    (h3 : is_prime c) (h4 : a ≠ 1) (h5 : b ≠ 1) (h6 : c ≠ 1) 
    (h7 : a ≠ b) (h8 : b ≠ c) (h9 : a ≠ c) : 

    1994 + a + b + c = a * b * c :=
sorry

end distinct_prime_sum_product_l10_10474


namespace largest_expression_value_l10_10573

-- Definitions of the expressions
def expr_A : ℕ := 3 + 0 + 1 + 8
def expr_B : ℕ := 3 * 0 + 1 + 8
def expr_C : ℕ := 3 + 0 * 1 + 8
def expr_D : ℕ := 3 + 0 + 1^2 + 8
def expr_E : ℕ := 3 * 0 * 1^2 * 8

-- Statement of the theorem
theorem largest_expression_value :
  max expr_A (max expr_B (max expr_C (max expr_D expr_E))) = 12 :=
by
  sorry

end largest_expression_value_l10_10573


namespace Annika_three_times_Hans_in_future_l10_10669

theorem Annika_three_times_Hans_in_future
  (hans_age_now : Nat)
  (annika_age_now : Nat)
  (x : Nat)
  (hans_future_age : Nat)
  (annika_future_age : Nat)
  (H1 : hans_age_now = 8)
  (H2 : annika_age_now = 32)
  (H3 : hans_future_age = hans_age_now + x)
  (H4 : annika_future_age = annika_age_now + x)
  (H5 : annika_future_age = 3 * hans_future_age) :
  x = 4 := 
  by
  sorry

end Annika_three_times_Hans_in_future_l10_10669


namespace opposite_of_neg_abs_opposite_of_neg_abs_correct_l10_10896

theorem opposite_of_neg_abs (x : ℚ) (hx : |x| = 2 / 5) : -|x| = - (2 / 5) := sorry

theorem opposite_of_neg_abs_correct (x : ℚ) (hx : |x| = 2 / 5) : - -|x| = 2 / 5 := by
  rw [opposite_of_neg_abs x hx]
  simp

end opposite_of_neg_abs_opposite_of_neg_abs_correct_l10_10896


namespace train_length_l10_10575

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (conversion_factor : ℝ) (speed_ms : ℝ) (distance_m : ℝ) 
  (h1 : speed_kmh = 36) 
  (h2 : time_s = 28)
  (h3 : conversion_factor = 1000 / 3600) -- convert km/hr to m/s
  (h4 : speed_ms = speed_kmh * conversion_factor)
  (h5 : distance_m = speed_ms * time_s) :
  distance_m = 280 := 
by
  sorry

end train_length_l10_10575


namespace range_of_m_range_of_x_l10_10299

-- Define the function f(x) = m*x^2 - m*x - 6 + m
def f (m x : ℝ) : ℝ := m*x^2 - m*x - 6 + m

-- Proof for the first statement
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f m x < 0) ↔ m < 6 / 7 := 
sorry

-- Proof for the second statement
theorem range_of_x (x : ℝ) :
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → f m x < 0) ↔ -1 < x ∧ x < 2 :=
sorry

end range_of_m_range_of_x_l10_10299


namespace fraction_multiplication_l10_10201

-- Given fractions a and b
def a := (1 : ℚ) / 4
def b := (1 : ℚ) / 8

-- The first product result
def result1 := a * b

-- The final product result when multiplied by 4
def result2 := result1 * 4

-- The theorem to prove
theorem fraction_multiplication : result2 = (1 : ℚ) / 8 := by
  sorry

end fraction_multiplication_l10_10201


namespace first_part_is_13_l10_10503

-- Definitions for the conditions
variables (x y : ℕ)

-- Conditions given in the problem
def condition1 : Prop := x + y = 24
def condition2 : Prop := 7 * x + 5 * y = 146

-- The theorem we need to prove
theorem first_part_is_13 (h1 : condition1 x y) (h2 : condition2 x y) : x = 13 :=
sorry

end first_part_is_13_l10_10503


namespace range_of_a_l10_10113

noncomputable def f (x a : ℝ) : ℝ := (Real.sqrt x) / (x^3 - 3 * x + a)

theorem range_of_a (a : ℝ) :
    (∀ x, 0 ≤ x → x^3 - 3 * x + a ≠ 0) ↔ 2 < a := 
by 
  sorry

end range_of_a_l10_10113


namespace floor_sqrt_17_squared_eq_16_l10_10463

theorem floor_sqrt_17_squared_eq_16 :
  (⌊Real.sqrt 17⌋ : Real)^2 = 16 := by
  sorry

end floor_sqrt_17_squared_eq_16_l10_10463


namespace exist_ints_a_b_for_any_n_l10_10215

theorem exist_ints_a_b_for_any_n (n : ℤ) : ∃ a b : ℤ, n = Int.floor (a * Real.sqrt 2) + Int.floor (b * Real.sqrt 3) := by
  sorry

end exist_ints_a_b_for_any_n_l10_10215


namespace complement_of_65_degrees_l10_10929

def angle_complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_65_degrees : angle_complement 65 = 25 := by
  -- Proof would follow here, but it's omitted since 'sorry' is added.
  sorry

end complement_of_65_degrees_l10_10929


namespace sum_in_Q_l10_10083

open Set

def is_set_P (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k
def is_set_Q (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k - 1
def is_set_M (x : ℤ) : Prop := ∃ k : ℤ, x = 4 * k + 1

variables (a b : ℤ)

theorem sum_in_Q (ha : is_set_P a) (hb : is_set_Q b) : is_set_Q (a + b) := 
sorry

end sum_in_Q_l10_10083


namespace isosceles_triangle_perimeter_l10_10988

theorem isosceles_triangle_perimeter :
  (∃ x y : ℝ, x^2 - 6*x + 8 = 0 ∧ y^2 - 6*y + 8 = 0 ∧ (x = 2 ∧ y = 4) ∧ 2 + 4 + 4 = 10) :=
by
  sorry

end isosceles_triangle_perimeter_l10_10988


namespace product_mnp_l10_10170

theorem product_mnp (a x y b : ℝ) (m n p : ℕ):
  (a ^ 8 * x * y - 2 * a ^ 7 * y - 3 * a ^ 6 * x = 2 * a ^ 5 * (b ^ 5 - 2)) ∧
  (a ^ 8 * x * y - 2 * a ^ 7 * y - 3 * a ^ 6 * x + 6 * a ^ 5 = (a ^ m * x - 2 * a ^ n) * (a ^ p * y - 3 * a ^ 3)) →
  m = 5 ∧ n = 4 ∧ p = 3 ∧ m * n * p = 60 :=
by
  intros h
  sorry

end product_mnp_l10_10170


namespace find_s_l10_10309

theorem find_s (n r s c d : ℚ) 
  (h1 : Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 3 = 0) 
  (h2 : c * d = 3)
  (h3 : Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s = 
        Polynomial.C (c + d⁻¹) * Polynomial.C (d + c⁻¹)) : 
  s = 16 / 3 := 
by
  sorry

end find_s_l10_10309


namespace positive_difference_of_complementary_angles_in_ratio_5_to_4_l10_10748

-- Definitions for given conditions
def is_complementary (a b : ℝ) : Prop :=
  a + b = 90

def ratio_5_to_4 (a b : ℝ) : Prop :=
  ∃ x : ℝ, a = 5 * x ∧ b = 4 * x

-- Theorem to prove the measure of their positive difference is 10 degrees
theorem positive_difference_of_complementary_angles_in_ratio_5_to_4
  {a b : ℝ} (h_complementary : is_complementary a b) (h_ratio : ratio_5_to_4 a b) :
  abs (a - b) = 10 :=
by 
  sorry

end positive_difference_of_complementary_angles_in_ratio_5_to_4_l10_10748


namespace part_a_part_a_rev_l10_10572

variable (x y : ℝ)

theorem part_a (hx : x > 0) (hy : y > 0) : x + y > |x - y| :=
sorry

theorem part_a_rev (h : x + y > |x - y|) : x > 0 ∧ y > 0 :=
sorry

end part_a_part_a_rev_l10_10572


namespace correct_conclusions_l10_10595

open Real

noncomputable def parabola (a b c : ℝ) : ℝ → ℝ :=
  λ x => a*x^2 + b*x + c

theorem correct_conclusions (a b c m n : ℝ)
  (h1 : c < 0)
  (h2 : parabola a b c 1 = 1)
  (h3 : parabola a b c m = 0)
  (h4 : parabola a b c n = 0)
  (h5 : n ≥ 3) :
  (4*a*c - b^2 < 4*a) ∧
  (n = 3 → ∃ t : ℝ, parabola a b c 2 = t ∧ t > 1) ∧
  (∀ x : ℝ, parabola a b (c - 1) x = 0 → (0 < m ∧ m ≤ 1/3)) :=
sorry

end correct_conclusions_l10_10595


namespace greatest_integer_a_exists_l10_10191

theorem greatest_integer_a_exists (a x : ℤ) (h : (x - a) * (x - 7) + 3 = 0) : a ≤ 11 := by
  sorry

end greatest_integer_a_exists_l10_10191


namespace find_x_given_distance_l10_10568

theorem find_x_given_distance (x : ℝ) : abs (x - 4) = 1 → (x = 5 ∨ x = 3) :=
by
  intro h
  sorry

end find_x_given_distance_l10_10568


namespace pizza_slices_with_both_l10_10019

theorem pizza_slices_with_both (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24) (h_pepperoni : pepperoni_slices = 15) (h_mushrooms : mushroom_slices = 14) :
  ∃ n, n = 5 ∧ total_slices = pepperoni_slices + mushroom_slices - n := 
by
  use 5
  sorry

end pizza_slices_with_both_l10_10019


namespace monotonic_intervals_range_of_values_l10_10099

-- Part (1): Monotonic intervals of the function
theorem monotonic_intervals (a : ℝ) (h_a : a = 0) :
  (∀ x, 0 < x ∧ x < 1 → (1 + Real.log x) / x > 0) ∧ (∀ x, 1 < x → (1 + Real.log x) / x < 0) :=
by
  sorry

-- Part (2): Range of values for \(a\)
theorem range_of_values (a : ℝ) (h_f : ∀ x, 0 < x → (1 + Real.log x) / x - a ≤ 0) : 
  1 ≤ a :=
by
  sorry

end monotonic_intervals_range_of_values_l10_10099


namespace geom_sequence_50th_term_l10_10801

theorem geom_sequence_50th_term (a a_2 : ℤ) (n : ℕ) (r : ℤ) (h1 : a = 8) (h2 : a_2 = -16) (h3 : r = a_2 / a) (h4 : n = 50) :
  a * r^(n-1) = -8 * 2^49 :=
by
  sorry

end geom_sequence_50th_term_l10_10801


namespace problem_statement_l10_10540

-- Define the conditions as Lean predicates
def is_odd (n : ℕ) : Prop := n % 2 = 1
def between_400_and_600 (n : ℕ) : Prop := 400 < n ∧ n < 600
def divisible_by_55 (n : ℕ) : Prop := n % 55 = 0

-- Define a function to calculate the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

-- Main theorem to prove
theorem problem_statement (N : ℕ)
  (h_odd : is_odd N)
  (h_range : between_400_and_600 N)
  (h_divisible : divisible_by_55 N) :
  sum_of_digits N = 18 :=
sorry

end problem_statement_l10_10540


namespace average_after_modifications_l10_10553

theorem average_after_modifications (S : ℕ) (sum_initial : S = 1080)
  (sum_after_removals : S - 80 - 85 = 915)
  (sum_after_additions : 915 + 75 + 75 = 1065) :
  (1065 / 12 : ℚ) = 88.75 :=
by sorry

end average_after_modifications_l10_10553


namespace monthly_income_A_l10_10779

theorem monthly_income_A (A B C : ℝ) :
  A + B = 10100 ∧ B + C = 12500 ∧ A + C = 10400 →
  A = 4000 :=
by
  intro h
  have h1 : A + B = 10100 := h.1
  have h2 : B + C = 12500 := h.2.1
  have h3 : A + C = 10400 := h.2.2
  sorry

end monthly_income_A_l10_10779


namespace system_of_equations_solution_l10_10161

theorem system_of_equations_solution (a b x y : ℝ) 
  (h1 : x = 1) 
  (h2 : y = 2)
  (h3 : a * x + y = -1)
  (h4 : 2 * x - b * y = 0) : 
  a + b = -2 := 
sorry

end system_of_equations_solution_l10_10161


namespace minimum_cost_l10_10686

theorem minimum_cost (price_pen_A price_pen_B price_notebook_A price_notebook_B : ℕ) 
  (discount_B : ℚ) (num_pens num_notebooks : ℕ)
  (h_price_pen : price_pen_A = 10) (h_price_notebook : price_notebook_A = 2)
  (h_discount : discount_B = 0.9) (h_num_pens : num_pens = 4) (h_num_notebooks : num_notebooks = 24) :
  ∃ (min_cost : ℕ), min_cost = 76 :=
by
  -- The conditions should be used here to construct the min_cost
  sorry

end minimum_cost_l10_10686


namespace distance_problem_l10_10910

noncomputable def distance_point_to_plane 
  (x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ) : ℝ :=
  -- Equation of the plane passing through three points derived using determinants
  let a := x2 - x1
  let b := y2 - y1
  let c := z2 - z1
  let d := x3 - x1
  let e := y3 - y1
  let f := z3 - z1
  let A := b*f - c*e
  let B := c*d - a*f
  let C := a*e - b*d
  let D := -(A*x1 + B*y1 + C*z1)
  -- Distance from the given point to the above plane
  (|A*x0 + B*y0 + C*z0 + D|) / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_problem :
  distance_point_to_plane 
  3 6 68 
  (-3) (-5) 6 
  2 1 (-4) 
  0 (-3) (-1) 
  = Real.sqrt 573 :=
by sorry

end distance_problem_l10_10910


namespace probability_not_exceeding_40_l10_10361

variable (P : ℝ → Prop)

def less_than_30_grams : Prop := P 0.3
def between_30_and_40_grams : Prop := P 0.5

theorem probability_not_exceeding_40 (h1 : less_than_30_grams P) (h2 : between_30_and_40_grams P) : P 0.8 :=
by
  sorry

end probability_not_exceeding_40_l10_10361


namespace algae_cell_count_at_day_nine_l10_10363

noncomputable def initial_cells : ℕ := 5
noncomputable def division_frequency_days : ℕ := 3
noncomputable def total_days : ℕ := 9

def number_of_cycles (total_days division_frequency_days : ℕ) : ℕ :=
  total_days / division_frequency_days

noncomputable def common_ratio : ℕ := 2

noncomputable def number_of_cells_after_n_days (initial_cells common_ratio number_of_cycles : ℕ) : ℕ :=
  initial_cells * common_ratio ^ (number_of_cycles - 1)

theorem algae_cell_count_at_day_nine : number_of_cells_after_n_days initial_cells common_ratio (number_of_cycles total_days division_frequency_days) = 20 :=
by
  sorry

end algae_cell_count_at_day_nine_l10_10363


namespace train_speed_l10_10267

theorem train_speed (length : ℝ) (time : ℝ) (length_eq : length = 400) (time_eq : time = 16) :
  (length / time) * (3600 / 1000) = 90 :=
by 
  rw [length_eq, time_eq]
  sorry

end train_speed_l10_10267


namespace number_of_distinct_rationals_l10_10423

theorem number_of_distinct_rationals (L : ℕ) :
  L = 26 ↔
  (∃ (k : ℚ), |k| < 100 ∧ (∃ (x : ℤ), 7 * x^2 + k * x + 20 = 0)) :=
sorry

end number_of_distinct_rationals_l10_10423


namespace shaded_area_correct_l10_10914

-- Definitions of the given conditions
def first_rectangle_length : ℕ := 8
def first_rectangle_width : ℕ := 5
def second_rectangle_length : ℕ := 4
def second_rectangle_width : ℕ := 9
def overlapping_area : ℕ := 3

def first_rectangle_area := first_rectangle_length * first_rectangle_width
def second_rectangle_area := second_rectangle_length * second_rectangle_width

-- Problem statement in Lean 4
theorem shaded_area_correct :
  first_rectangle_area + second_rectangle_area - overlapping_area = 73 :=
by
  -- The proof is skipped
  sorry

end shaded_area_correct_l10_10914


namespace range_of_x_l10_10356

theorem range_of_x (x : ℤ) : x^2 < 3 * x → x = 1 ∨ x = 2 :=
by
  sorry

end range_of_x_l10_10356


namespace difference_of_squares_l10_10813

theorem difference_of_squares (n : ℕ) : (n+1)^2 - n^2 = 2*n + 1 :=
by
  sorry

end difference_of_squares_l10_10813


namespace debby_soda_bottles_l10_10984

noncomputable def total_bottles (d t : ℕ) : ℕ := d * t

theorem debby_soda_bottles :
  ∀ (d t: ℕ), d = 9 → t = 40 → total_bottles d t = 360 :=
by
  intros d t h1 h2
  sorry

end debby_soda_bottles_l10_10984


namespace probability_of_prime_number_on_spinner_l10_10687

-- Definitions of conditions
def spinner_sections : List ℕ := [2, 3, 4, 5, 7, 9, 10, 11]
def total_sectors : ℕ := 8
def prime_count : ℕ := List.filter Nat.Prime spinner_sections |>.length

-- Statement of the theorem we want to prove
theorem probability_of_prime_number_on_spinner :
  (prime_count : ℚ) / total_sectors = 5 / 8 := by
  sorry

end probability_of_prime_number_on_spinner_l10_10687


namespace polar_to_cartesian_l10_10194

theorem polar_to_cartesian (ρ : ℝ) (θ : ℝ) (hx : ρ = 3) (hy : θ = π / 6) :
  (ρ * Real.cos θ, ρ * Real.sin θ) = (3 * Real.cos (π / 6), 3 * Real.sin (π / 6)) := by
  sorry

end polar_to_cartesian_l10_10194


namespace triangle_with_altitudes_is_obtuse_l10_10247

theorem triangle_with_altitudes_is_obtuse (h1 h2 h3 : ℝ) (h_pos1 : h1 > 0) (h_pos2 : h2 > 0) (h_pos3 : h3 > 0)
    (h_triangle_ineq1 : 1 / h2 + 1 / h3 > 1 / h1)
    (h_triangle_ineq2 : 1 / h1 + 1 / h3 > 1 / h2)
    (h_triangle_ineq3 : 1 / h1 + 1 / h2 > 1 / h3) : 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a * h1 = b * h2 ∧ b * h2 = c * h3 ∧
    (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2) :=
sorry

end triangle_with_altitudes_is_obtuse_l10_10247


namespace exists_n_for_A_of_non_perfect_square_l10_10285

theorem exists_n_for_A_of_non_perfect_square (A : ℕ) (h : ∀ k : ℕ, k^2 ≠ A) :
  ∃ n : ℕ, A = ⌊ n + Real.sqrt n + 1/2 ⌋ :=
sorry

end exists_n_for_A_of_non_perfect_square_l10_10285


namespace Emily_cleaning_time_in_second_room_l10_10345

/-
Lilly, Fiona, Jack, and Emily are cleaning 3 rooms.
For the first room: Lilly and Fiona together: 1/4 of the time, Jack: 1/3 of the time, Emily: the rest of the time.
In the second room: Jack: 25%, Emily: 25%, Lilly and Fiona: the remaining 50%.
In the third room: Emily: 40%, Lilly: 20%, Jack: 20%, Fiona: 20%.
Total time for all rooms: 12 hours.

Prove that the total time Emily spent cleaning in the second room is 60 minutes.
-/

theorem Emily_cleaning_time_in_second_room :
  let total_time := 12 -- total time in hours
  let time_per_room := total_time / 3 -- time per room in hours
  let time_per_room_minutes := time_per_room * 60 -- time per room in minutes
  let emily_cleaning_percentage := 0.25 -- Emily's cleaning percentage in the second room
  let emily_cleaning_time := emily_cleaning_percentage * time_per_room_minutes -- cleaning time in minutes
  emily_cleaning_time = 60 := by
  sorry

end Emily_cleaning_time_in_second_room_l10_10345


namespace distance_between_cities_l10_10865

def distance_thing 
  (d_A d_B : ℝ) 
  (v_A v_B : ℝ) 
  (t_diff : ℝ) : Prop :=
d_A = (3 / 5) * d_B ∧
v_A = 72 ∧
v_B = 108 ∧
t_diff = (1 / 4) ∧
(d_A + d_B) = 432

theorem distance_between_cities
  (d_A d_B : ℝ)
  (v_A v_B : ℝ)
  (t_diff : ℝ)
  (h : distance_thing d_A d_B v_A v_B t_diff)
  : d_A + d_B = 432 := by
  sorry

end distance_between_cities_l10_10865


namespace least_pos_int_div_by_3_5_7_l10_10327

/-
  Prove that the least positive integer divisible by the primes 3, 5, and 7 is 105.
-/

theorem least_pos_int_div_by_3_5_7 : ∃ (n : ℕ), n > 0 ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ n = 105 :=
by 
  sorry

end least_pos_int_div_by_3_5_7_l10_10327


namespace polynomial_root_s_eq_pm1_l10_10050

theorem polynomial_root_s_eq_pm1
  (b_3 b_2 b_1 : ℤ)
  (s : ℤ)
  (h1 : s^3 ∣ 50)
  (h2 : (s^4 + b_3 * s^3 + b_2 * s^2 + b_1 * s + 50) = 0) :
  s = 1 ∨ s = -1 :=
sorry

end polynomial_root_s_eq_pm1_l10_10050


namespace max_projection_area_of_tetrahedron_l10_10550

/-- 
Two adjacent faces of a tetrahedron are isosceles right triangles with a hypotenuse of 2,
and they form a dihedral angle of 60 degrees. The tetrahedron rotates around the common edge
of these faces. The maximum area of the projection of the rotating tetrahedron onto 
the plane containing the given edge is 1.
-/
theorem max_projection_area_of_tetrahedron (S hypotenuse dihedral max_proj_area : ℝ)
  (is_isosceles_right_triangle : ∀ (a b : ℝ), a^2 + b^2 = hypotenuse^2)
  (hypotenuse_len : hypotenuse = 2)
  (dihedral_angle : dihedral = 60) :
  max_proj_area = 1 :=
  sorry

end max_projection_area_of_tetrahedron_l10_10550


namespace total_rocks_needed_l10_10214

def rocks_already_has : ℕ := 64
def rocks_needed : ℕ := 61

theorem total_rocks_needed : rocks_already_has + rocks_needed = 125 :=
by
  sorry

end total_rocks_needed_l10_10214


namespace num_bases_ending_in_1_l10_10678

theorem num_bases_ending_in_1 : 
  (∃ bases : Finset ℕ, 
  ∀ b ∈ bases, 3 ≤ b ∧ b ≤ 10 ∧ (625 % b = 1) ∧ bases.card = 4) :=
sorry

end num_bases_ending_in_1_l10_10678


namespace time_to_be_apart_l10_10558

noncomputable def speed_A : ℝ := 17.5
noncomputable def speed_B : ℝ := 15
noncomputable def initial_distance : ℝ := 65
noncomputable def final_distance : ℝ := 32.5

theorem time_to_be_apart (x : ℝ) :
  x = 1 ∨ x = 3 ↔ 
  (x * (speed_A + speed_B) = initial_distance - final_distance ∨ 
   x * (speed_A + speed_B) = initial_distance + final_distance) :=
sorry

end time_to_be_apart_l10_10558


namespace linear_system_solution_l10_10999

theorem linear_system_solution (a b : ℝ) 
  (h1 : 3 * a + 2 * b = 5) 
  (h2 : 2 * a + 3 * b = 4) : 
  a - b = 1 := 
by
  sorry

end linear_system_solution_l10_10999


namespace total_savings_correct_l10_10195

theorem total_savings_correct :
  let price_chlorine := 10
  let discount1_chlorine := 0.20
  let discount2_chlorine := 0.10
  let price_soap := 16
  let discount1_soap := 0.25
  let discount2_soap := 0.05
  let price_wipes := 8
  let bogo_discount_wipes := 0.50
  let quantity_chlorine := 4
  let quantity_soap := 6
  let quantity_wipes := 8
  let final_chlorine_price := (price_chlorine * (1 - discount1_chlorine)) * (1 - discount2_chlorine)
  let final_soap_price := (price_soap * (1 - discount1_soap)) * (1 - discount2_soap)
  let final_wipes_price_per_two := price_wipes + price_wipes * bogo_discount_wipes
  let final_wipes_price := final_wipes_price_per_two / 2
  let total_original_price := quantity_chlorine * price_chlorine + quantity_soap * price_soap + quantity_wipes * price_wipes
  let total_final_price := quantity_chlorine * final_chlorine_price + quantity_soap * final_soap_price + quantity_wipes * final_wipes_price
  let total_savings := total_original_price - total_final_price
  total_savings = 55.80 :=
by sorry

end total_savings_correct_l10_10195


namespace value_of_sum_cubes_l10_10794

theorem value_of_sum_cubes (x : ℝ) (hx : x ≠ 0) (h : 47 = x^6 + (1 / x^6)) : (x^3 + (1 / x^3)) = 7 := 
by 
  sorry

end value_of_sum_cubes_l10_10794


namespace average_licks_l10_10793

theorem average_licks 
  (Dan_licks : ℕ := 58)
  (Michael_licks : ℕ := 63)
  (Sam_licks : ℕ := 70)
  (David_licks : ℕ := 70)
  (Lance_licks : ℕ := 39) :
  (Dan_licks + Michael_licks + Sam_licks + David_licks + Lance_licks) / 5 = 60 := 
sorry

end average_licks_l10_10793


namespace distance_between_points_A_and_B_l10_10237

theorem distance_between_points_A_and_B :
  ∃ (d : ℝ), 
    -- Distance must be non-negative
    d ≥ 0 ∧
    -- Condition 1: Car 3 reaches point A at 10:00 AM (3 hours after 7:00 AM)
    (∃ V3 : ℝ, V3 = d / 6) ∧ 
    -- Condition 2: Car 2 reaches point A at 10:30 AM (3.5 hours after 7:00 AM)
    (∃ V2 : ℝ, V2 = 2 * d / 7) ∧ 
    -- Condition 3: When Car 1 and Car 3 meet, Car 2 has traveled exactly 3/8 of d
    (∃ V1 : ℝ, V1 = (d - 84) / 7 ∧ 2 * V1 + 2 * V3 = 8 * V2 / 3) ∧ 
    -- Required: The distance between A and B is 336 km
    d = 336 :=
by
  sorry

end distance_between_points_A_and_B_l10_10237


namespace Jessie_final_weight_l10_10061

variable (initial_weight : ℝ) (loss_first_week : ℝ) (loss_rate_second_week : ℝ)
variable (loss_second_week : ℝ) (total_loss : ℝ) (final_weight : ℝ)

def Jessie_weight_loss_problem : Prop :=
  initial_weight = 92 ∧
  loss_first_week = 5 ∧
  loss_rate_second_week = 1.3 ∧
  loss_second_week = loss_rate_second_week * loss_first_week ∧
  total_loss = loss_first_week + loss_second_week ∧
  final_weight = initial_weight - total_loss ∧
  final_weight = 80.5

theorem Jessie_final_weight : Jessie_weight_loss_problem initial_weight loss_first_week loss_rate_second_week loss_second_week total_loss final_weight :=
by
  sorry

end Jessie_final_weight_l10_10061


namespace inequality_for_positive_integers_l10_10268

theorem inequality_for_positive_integers (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b + b * c + a * c ≤ 3 * a * b * c :=
sorry

end inequality_for_positive_integers_l10_10268


namespace ordinary_eq_of_curve_l10_10791

theorem ordinary_eq_of_curve 
  (t : ℝ) (x : ℝ) (y : ℝ)
  (ht : t > 0) 
  (hx : x = Real.sqrt t - 1 / Real.sqrt t)
  (hy : y = 3 * (t + 1 / t)) :
  3 * x^2 - y + 6 = 0 ∧ y ≥ 6 :=
sorry

end ordinary_eq_of_curve_l10_10791


namespace find_k_for_line_l10_10314

theorem find_k_for_line (k : ℝ) : (2 * k * (-1/2) + 1 = -7 * 3) → k = 22 :=
by
  intro h
  sorry

end find_k_for_line_l10_10314


namespace number_of_solutions_l10_10511

-- Define the main theorem with the correct conditions
theorem number_of_solutions : 
  (∃ (x₁ x₂ x₃ x₄ x₅ : ℕ), 
     x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧ x₁ + x₂ + x₃ + x₄ + x₅ = 10) 
  → 
  (∃ t : ℕ, t = 70) :=
by 
  sorry

end number_of_solutions_l10_10511


namespace chessboard_piece_arrangements_l10_10994

-- Define the problem in Lean
theorem chessboard_piece_arrangements (black_pos white_pos : ℕ)
  (black_pos_neq_white_pos : black_pos ≠ white_pos)
  (valid_position : black_pos < 64 ∧ white_pos < 64) :
  ¬(∀ (move : ℕ → ℕ → Prop), (move black_pos white_pos) → ∃! (p : ℕ × ℕ), move (p.fst) (p.snd)) :=
by sorry

end chessboard_piece_arrangements_l10_10994


namespace max_1x2_rectangles_in_3x3_grid_l10_10369

theorem max_1x2_rectangles_in_3x3_grid : 
  ∀ unit_squares rectangles_1x2 : ℕ, unit_squares + rectangles_1x2 = 9 → 
  (∃ max_rectangles : ℕ, max_rectangles = rectangles_1x2 ∧ max_rectangles = 5) :=
by
  sorry

end max_1x2_rectangles_in_3x3_grid_l10_10369


namespace increasing_iff_a_gt_neg1_l10_10765

noncomputable def increasing_function_condition (a : ℝ) (b : ℝ) (x : ℝ) : Prop :=
  let y := (a + 1) * x + b
  a > -1

theorem increasing_iff_a_gt_neg1 (a : ℝ) (b : ℝ) : (∀ x : ℝ, (a + 1) > 0) ↔ a > -1 :=
by
  sorry

end increasing_iff_a_gt_neg1_l10_10765


namespace mean_of_observations_decreased_l10_10451

noncomputable def original_mean : ℕ := 200

theorem mean_of_observations_decreased (S' : ℕ) (M' : ℕ) (n : ℕ) (d : ℕ)
  (h1 : n = 50)
  (h2 : d = 15)
  (h3 : M' = 185)
  (h4 : S' = M' * n)
  : original_mean = (S' + d * n) / n :=
by
  rw [original_mean]
  sorry

end mean_of_observations_decreased_l10_10451


namespace rowing_upstream_distance_l10_10038

theorem rowing_upstream_distance 
  (b s t d1 d2 : ℝ)
  (h1 : s = 7)
  (h2 : d1 = 72)
  (h3 : t = 3)
  (h4 : d1 = (b + s) * t) :
  d2 = (b - s) * t → d2 = 30 :=
by 
  intros h5
  sorry

end rowing_upstream_distance_l10_10038


namespace sum_of_squares_of_coeffs_l10_10168

theorem sum_of_squares_of_coeffs (a b c : ℕ) : (a = 6) → (b = 24) → (c = 12) → (a^2 + b^2 + c^2 = 756) :=
by
  sorry

end sum_of_squares_of_coeffs_l10_10168


namespace stickers_given_to_sister_l10_10991

variable (initial bought birthday used left given : ℕ)

theorem stickers_given_to_sister :
  (initial = 20) →
  (bought = 12) →
  (birthday = 20) →
  (used = 8) →
  (left = 39) →
  (given = (initial + bought + birthday - used - left)) →
  given = 5 := by
  intros
  sorry

end stickers_given_to_sister_l10_10991


namespace prove_central_angle_of_sector_l10_10183

noncomputable def central_angle_of_sector (R α : ℝ) : Prop :=
  (2 * R + R * α = 8) ∧ (1 / 2 * α * R^2 = 4)

theorem prove_central_angle_of_sector :
  ∃ α R : ℝ, central_angle_of_sector R α ∧ α = 2 :=
sorry

end prove_central_angle_of_sector_l10_10183


namespace sum_of_roots_of_quadratic_eq_l10_10332

theorem sum_of_roots_of_quadratic_eq (x : ℝ) (hx : x^2 = 8 * x + 15) :
  ∃ S : ℝ, S = 8 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l10_10332


namespace min_xy_positive_real_l10_10060

theorem min_xy_positive_real (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 3 / (2 + x) + 3 / (2 + y) = 1) :
  ∃ m : ℝ, m = 16 ∧ ∀ xy : ℝ, (xy = x * y) → xy ≥ m :=
by
  sorry

end min_xy_positive_real_l10_10060


namespace number_of_children_at_matinee_l10_10493

-- Definitions of constants based on conditions
def children_ticket_price : ℝ := 4.50
def adult_ticket_price : ℝ := 6.75
def total_receipts : ℝ := 405
def additional_children : ℕ := 20

-- Variables for number of adults and children
variable (A C : ℕ)

-- Assertions based on conditions
axiom H1 : C = A + additional_children
axiom H2 : children_ticket_price * (C : ℝ) + adult_ticket_price * (A : ℝ) = total_receipts

-- Theorem statement: Prove that the number of children is 48
theorem number_of_children_at_matinee : C = 48 :=
by
  sorry

end number_of_children_at_matinee_l10_10493


namespace find_a2_l10_10688

-- Define the geometric sequence and its properties
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions 
variables (a : ℕ → ℝ) (h_geom : is_geometric a)
variables (h_a1 : a 1 = 1/4)
variables (h_condition : a 3 * a 5 = 4 * (a 4 - 1))

-- The goal is to prove a 2 = 1/2
theorem find_a2 : a 2 = 1/2 :=
by
  sorry

end find_a2_l10_10688


namespace correct_divisor_l10_10636

theorem correct_divisor (dividend incorrect_divisor quotient correct_quotient correct_divisor : ℕ) 
  (h1 : incorrect_divisor = 63) 
  (h2 : quotient = 24) 
  (h3 : correct_quotient = 42) 
  (h4 : dividend = incorrect_divisor * quotient) 
  (h5 : dividend / correct_divisor = correct_quotient) : 
  correct_divisor = 36 := 
by 
  sorry

end correct_divisor_l10_10636


namespace sarah_problem_l10_10855

theorem sarah_problem (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 100 ≤ y ∧ y ≤ 999) 
  (h : 1000 * x + y = 11 * x * y) : x + y = 110 :=
sorry

end sarah_problem_l10_10855


namespace total_revenue_correct_l10_10676

def price_per_book : ℝ := 25
def books_sold_monday : ℕ := 60
def discount_monday : ℝ := 0.10
def books_sold_tuesday : ℕ := 10
def discount_tuesday : ℝ := 0.0
def books_sold_wednesday : ℕ := 20
def discount_wednesday : ℝ := 0.05
def books_sold_thursday : ℕ := 44
def discount_thursday : ℝ := 0.15
def books_sold_friday : ℕ := 66
def discount_friday : ℝ := 0.20

def revenue (books_sold: ℕ) (discount: ℝ) : ℝ :=
  (1 - discount) * price_per_book * books_sold

theorem total_revenue_correct :
  revenue books_sold_monday discount_monday +
  revenue books_sold_tuesday discount_tuesday +
  revenue books_sold_wednesday discount_wednesday +
  revenue books_sold_thursday discount_thursday +
  revenue books_sold_friday discount_friday = 4330 := by 
sorry

end total_revenue_correct_l10_10676


namespace range_of_m_l10_10221

theorem range_of_m (m : ℝ) (h : m ≠ 0) :
  (∀ x : ℝ, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) →
  m < -1 / 2 :=
by
  sorry

end range_of_m_l10_10221


namespace part1_l10_10456

theorem part1 (a x0 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a ^ x0 = 2) : a ^ (3 * x0) = 8 := by
  sorry

end part1_l10_10456


namespace min_value_of_function_l10_10681

noncomputable def func (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.sin (2 * x)

theorem min_value_of_function : ∃ x : ℝ, func x = 1 - Real.sqrt 2 :=
by sorry

end min_value_of_function_l10_10681


namespace minimum_value_a_l10_10112

theorem minimum_value_a (a : ℝ) : (∃ x0 : ℝ, |x0 + 1| + |x0 - 2| ≤ a) → a ≥ 3 :=
by 
  sorry

end minimum_value_a_l10_10112


namespace case_a_case_b_case_c_l10_10059

-- Definitions of game manageable
inductive Player
| First
| Second

def sum_of_dimensions (m n : Nat) : Nat := m + n

def is_winning_position (m n : Nat) : Player :=
  if sum_of_dimensions m n % 2 = 1 then Player.First else Player.Second

-- Theorem statements for the given grid sizes
theorem case_a : is_winning_position 9 10 = Player.First := 
  sorry

theorem case_b : is_winning_position 10 12 = Player.Second := 
  sorry

theorem case_c : is_winning_position 9 11 = Player.Second := 
  sorry

end case_a_case_b_case_c_l10_10059


namespace num_intersections_l10_10887

noncomputable def polar_to_cartesian (r θ: ℝ): ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem num_intersections (θ: ℝ): 
  let c1 := polar_to_cartesian (6 * Real.cos θ) θ
  let c2 := polar_to_cartesian (10 * Real.sin θ) θ
  let (x1, y1) := c1
  let (x2, y2) := c2
  ((x1 - 3)^2 + y1^2 = 9 ∧ x2^2 + (y2 - 5)^2 = 25) →
  (x1, y1) = (x2, y2) ↔ false :=
by
  sorry

end num_intersections_l10_10887


namespace max_tan_beta_l10_10027

theorem max_tan_beta (α β : ℝ) (hαβ : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2) 
  (h : α + β ≠ π / 2) (h_sin_cos : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
  Real.tan β ≤ Real.sqrt 3 / 3 :=
sorry

end max_tan_beta_l10_10027


namespace same_terminal_side_eq_l10_10995

theorem same_terminal_side_eq (α : ℝ) : 
    (∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 3) ↔ α = 5 * Real.pi / 3 :=
by sorry

end same_terminal_side_eq_l10_10995


namespace companyA_sold_bottles_l10_10387

-- Let CompanyA and CompanyB be the prices per bottle for the respective companies
def CompanyA_price : ℝ := 4
def CompanyB_price : ℝ := 3.5

-- Company B sold 350 bottles
def CompanyB_bottles : ℕ := 350

-- Total revenue of Company B
def CompanyB_revenue : ℝ := CompanyB_price * CompanyB_bottles

-- Additional condition that the revenue difference is $25
def revenue_difference : ℝ := 25

-- Define the total revenue equations for both scenarios
def revenue_scenario1 (x : ℕ) : Prop :=
  CompanyA_price * x = CompanyB_revenue + revenue_difference

def revenue_scenario2 (x : ℕ) : Prop :=
  CompanyA_price * x + revenue_difference = CompanyB_revenue

-- The problem translates to finding x such that either of these conditions hold
theorem companyA_sold_bottles : ∃ x : ℕ, revenue_scenario2 x ∧ x = 300 :=
by
  sorry

end companyA_sold_bottles_l10_10387


namespace larger_page_number_l10_10840

theorem larger_page_number (x : ℕ) (h1 : (x + (x + 1) = 125)) : (x + 1 = 63) :=
by
  sorry

end larger_page_number_l10_10840


namespace find_k_all_reals_l10_10188

theorem find_k_all_reals (a b c : ℝ) : 
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c :=
sorry

end find_k_all_reals_l10_10188


namespace find_PS_l10_10819

theorem find_PS 
    (P Q R S : Type)
    (PQ PR : ℝ)
    (h : ℝ) 
    (ratio_QS_SR : ℝ)
    (hyp1 : PQ = 13)
    (hyp2 : PR = 20)
    (hyp3 : ratio_QS_SR = 3/7) :
    h = Real.sqrt (117.025) :=
by
  -- Proof steps would go here, but we are just stating the theorem
  sorry

end find_PS_l10_10819


namespace geometric_sequence_a3_q_l10_10811

theorem geometric_sequence_a3_q (a_5 a_4 a_3 a_2 a_1 : ℝ) (q : ℝ) :
  a_5 - a_1 = 15 →
  a_4 - a_2 = 6 →
  (q = 2 ∧ a_3 = 4) ∨ (q = 1/2 ∧ a_3 = -4) :=
by
  sorry

end geometric_sequence_a3_q_l10_10811


namespace smallest_relatively_prime_to_180_is_7_l10_10935

theorem smallest_relatively_prime_to_180_is_7 :
  ∃ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 ∧ ∀ z : ℕ, z > 1 ∧ Nat.gcd z 180 = 1 → y ≤ z :=
by
  sorry

end smallest_relatively_prime_to_180_is_7_l10_10935


namespace smallest_w_for_factors_l10_10149

theorem smallest_w_for_factors (w : ℕ) (h_pos : 0 < w) :
  (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (13^2 ∣ 936 * w) ↔ w = 156 := 
sorry

end smallest_w_for_factors_l10_10149


namespace cannot_determine_total_movies_l10_10633

def number_of_books : ℕ := 22
def books_read : ℕ := 12
def books_to_read : ℕ := 10
def movies_watched : ℕ := 56

theorem cannot_determine_total_movies (n : ℕ) (h1 : books_read + books_to_read = number_of_books) : n ≠ movies_watched → n = 56 → False := 
by 
  intro h2 h3
  sorry

end cannot_determine_total_movies_l10_10633


namespace jason_total_payment_l10_10305

def total_cost (shorts jacket shoes socks tshirts : ℝ) : ℝ :=
  shorts + jacket + shoes + socks + tshirts

def discount_amount (total : ℝ) (discount_rate : ℝ) : ℝ :=
  total * discount_rate

def total_after_discount (total discount : ℝ) : ℝ :=
  total - discount

def sales_tax_amount (total : ℝ) (tax_rate : ℝ) : ℝ :=
  total * tax_rate

def final_amount (total after_discount tax : ℝ) : ℝ :=
  after_discount + tax

theorem jason_total_payment :
  let shorts := 14.28
  let jacket := 4.74
  let shoes := 25.95
  let socks := 6.80
  let tshirts := 18.36
  let discount_rate := 0.15
  let tax_rate := 0.07
  let total := total_cost shorts jacket shoes socks tshirts
  let discount := discount_amount total discount_rate
  let after_discount := total_after_discount total discount
  let tax := sales_tax_amount after_discount tax_rate
  let final := final_amount total after_discount tax
  final = 63.78 :=
by
  sorry

end jason_total_payment_l10_10305


namespace find_ordered_pair_l10_10895

theorem find_ordered_pair (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hroot : ∀ x : ℝ, 2 * x^2 + a * x + b = 0 → x = a ∨ x = b) :
  (a, b) = (1 / 2, -3 / 4) := 
  sorry

end find_ordered_pair_l10_10895


namespace hoseoks_social_studies_score_l10_10667

theorem hoseoks_social_studies_score 
  (avg_three_subjects : ℕ) 
  (new_avg_with_social_studies : ℕ) 
  (total_score_three_subjects : ℕ) 
  (total_score_four_subjects : ℕ) 
  (S : ℕ)
  (h1 : avg_three_subjects = 89) 
  (h2 : new_avg_with_social_studies = 90) 
  (h3 : total_score_three_subjects = 3 * avg_three_subjects) 
  (h4 : total_score_four_subjects = 4 * new_avg_with_social_studies) :
  S = 93 :=
sorry

end hoseoks_social_studies_score_l10_10667


namespace encode_mathematics_l10_10555

def robotCipherMapping : String → String := sorry

theorem encode_mathematics :
  robotCipherMapping "MATHEMATICS" = "2232331122323323132" := sorry

end encode_mathematics_l10_10555


namespace beth_friends_l10_10055

theorem beth_friends (F : ℝ) (h1 : 4 / F + 6 = 6.4) : F = 10 :=
by
  sorry

end beth_friends_l10_10055


namespace total_apples_correctness_l10_10257

-- Define the number of apples each man bought
def applesMen := 30

-- Define the number of apples each woman bought
def applesWomen := applesMen + 20

-- Define the total number of apples bought by the two men
def totalApplesMen := 2 * applesMen

-- Define the total number of apples bought by the three women
def totalApplesWomen := 3 * applesWomen

-- Define the total number of apples bought by the two men and three women
def totalApples := totalApplesMen + totalApplesWomen

-- Prove that the total number of apples bought by two men and three women is 210
theorem total_apples_correctness : totalApples = 210 := by
  sorry

end total_apples_correctness_l10_10257


namespace adrianna_gum_pieces_l10_10417

-- Definitions based on conditions
def initial_gum_pieces : ℕ := 10
def additional_gum_pieces : ℕ := 3
def friends_count : ℕ := 11

-- Expression to calculate the final pieces of gum
def total_gum_pieces : ℕ := initial_gum_pieces + additional_gum_pieces
def gum_left : ℕ := total_gum_pieces - friends_count

-- Lean statement we want to prove
theorem adrianna_gum_pieces: gum_left = 2 := 
by 
  sorry

end adrianna_gum_pieces_l10_10417


namespace maximum_xyz_l10_10859

theorem maximum_xyz (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (h: x ^ (Real.log x / Real.log y) * y ^ (Real.log y / Real.log z) * z ^ (Real.log z / Real.log x) = 10) : 
  x * y * z ≤ 10 := 
sorry

end maximum_xyz_l10_10859


namespace same_terminal_side_l10_10800

theorem same_terminal_side (α : ℝ) (k : ℤ) : 
  ∃ k : ℤ, α = k * 360 + 60 → α = -300 := 
by
  sorry

end same_terminal_side_l10_10800


namespace hyperbola_constants_sum_l10_10352

noncomputable def hyperbola_asymptotes_equation (x y : ℝ) : Prop :=
  (y = 2 * x + 5) ∨ (y = -2 * x + 1)

noncomputable def hyperbola_passing_through (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 7)

theorem hyperbola_constants_sum
  (a b h k : ℝ) (ha : a > 0) (hb : b > 0)
  (H1 : ∀ x y : ℝ, hyperbola_asymptotes_equation x y)
  (H2 : hyperbola_passing_through 0 7)
  (H3 : h = -1)
  (H4 : k = 3)
  (H5 : a = 2 * b)
  (H6 : b = Real.sqrt 3) :
  a + h = 2 * Real.sqrt 3 - 1 :=
sorry

end hyperbola_constants_sum_l10_10352


namespace cost_price_A_l10_10966

-- Establishing the definitions based on the conditions from a)

def profit_A_to_B (CP_A : ℝ) : ℝ := 1.20 * CP_A
def profit_B_to_C (CP_B : ℝ) : ℝ := 1.25 * CP_B
def price_paid_by_C : ℝ := 222

-- Stating the theorem to be proven:
theorem cost_price_A (CP_A : ℝ) (H : profit_B_to_C (profit_A_to_B CP_A) = price_paid_by_C) : CP_A = 148 :=
by 
  sorry

end cost_price_A_l10_10966


namespace solve_inequality_l10_10672

theorem solve_inequality (x : ℝ) (h : x / 3 - 2 < 0) : x < 6 :=
sorry

end solve_inequality_l10_10672


namespace product_zero_when_a_is_three_l10_10377

theorem product_zero_when_a_is_three (a : ℤ) (h : a = 3) :
  (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  cases h
  sorry

end product_zero_when_a_is_three_l10_10377


namespace nesbitts_inequality_l10_10711

theorem nesbitts_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 :=
sorry

end nesbitts_inequality_l10_10711


namespace parallel_lines_slope_m_l10_10515

theorem parallel_lines_slope_m (m : ℝ) : (∀ (x y : ℝ), (x - 2 * y + 5 = 0) ↔ (2 * x + m * y - 5 = 0)) → m = -4 :=
by
  intros h
  -- Add the necessary calculative steps here
  sorry

end parallel_lines_slope_m_l10_10515


namespace solve_inequality_system_l10_10592

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l10_10592


namespace no_integer_roots_quadratic_l10_10216

theorem no_integer_roots_quadratic (a b : ℤ) : 
  ∀ u : ℤ, ¬(u^2 + 3*a*u + 3*(2 - b^2) = 0) := 
by
  sorry

end no_integer_roots_quadratic_l10_10216


namespace solve_trig_eq_l10_10706

theorem solve_trig_eq (x : ℝ) :
  (0.5 * (Real.cos (5 * x) + Real.cos (7 * x)) - Real.cos (2 * x) ^ 2 + Real.sin (3 * x) ^ 2 = 0) →
  (∃ k : ℤ, x = (Real.pi / 2) * (2 * k + 1) ∨ x = (2 * k * Real.pi / 11)) :=
sorry

end solve_trig_eq_l10_10706


namespace vector_subtraction_l10_10506

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Define the expression to be proven
def expression : ℝ × ℝ := ((2 * a.1 - b.1), (2 * a.2 - b.2))

-- The theorem statement
theorem vector_subtraction : expression = (5, 7) :=
by
  -- The proof will be provided here
  sorry

end vector_subtraction_l10_10506


namespace cube_face_sum_l10_10491

theorem cube_face_sum
  (a d b e c f : ℕ)
  (pos_a : 0 < a) (pos_d : 0 < d) (pos_b : 0 < b) (pos_e : 0 < e) (pos_c : 0 < c) (pos_f : 0 < f)
  (hd : (a + d) * (b + e) * (c + f) = 2107) :
  a + d + b + e + c + f = 57 :=
sorry

end cube_face_sum_l10_10491


namespace repeating_decimal_to_fraction_l10_10836

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l10_10836


namespace probability_penny_nickel_heads_l10_10609

noncomputable def num_outcomes : ℕ := 2^4
noncomputable def num_successful_outcomes : ℕ := 2 * 2

theorem probability_penny_nickel_heads :
  (num_successful_outcomes : ℚ) / num_outcomes = 1 / 4 :=
by
  sorry

end probability_penny_nickel_heads_l10_10609


namespace original_rectangle_area_is_56_l10_10646

-- Conditions
def original_rectangle_perimeter := 30 -- cm
def smaller_rectangle_perimeter := 16 -- cm
def side_length_square := (original_rectangle_perimeter - smaller_rectangle_perimeter) / 2 -- Using the reduction logic

-- Computing the length and width of the original rectangle.
def width_original_rectangle := side_length_square
def length_original_rectangle := smaller_rectangle_perimeter / 2

-- The goal is to prove that the area of the original rectangle is 56 cm^2.

theorem original_rectangle_area_is_56:
  (length_original_rectangle - width_original_rectangle + width_original_rectangle) = 8 -- finding the length
  ∧ (length_original_rectangle * width_original_rectangle) = 56 := by
  sorry

end original_rectangle_area_is_56_l10_10646


namespace solution_set_inequality_l10_10160

theorem solution_set_inequality (a m : ℝ) (h : ∀ x : ℝ, (x > m ∧ x < 1) ↔ 2 * x^2 - 3 * x + a < 0) : m = 1 / 2 :=
by
  -- Insert the proof here
  sorry

end solution_set_inequality_l10_10160


namespace stock_and_bond_value_relation_l10_10122

-- Definitions for conditions
def more_valuable_shares : ℕ := 14
def less_valuable_shares : ℕ := 26
def face_value_bond : ℝ := 1000
def coupon_rate_bond : ℝ := 0.06
def discount_rate_bond : ℝ := 0.03
def total_assets_value : ℝ := 2106

-- Lean statement for the proof problem
theorem stock_and_bond_value_relation (x y : ℝ) 
    (h1 : face_value_bond * (1 - discount_rate_bond) = 970)
    (h2 : 27 * x + y = total_assets_value) :
    y = 2106 - 27 * x :=
by
  sorry

end stock_and_bond_value_relation_l10_10122


namespace intersection_of_M_and_N_l10_10251

-- Define sets M and N
def M := {x : ℝ | (x + 2) * (x - 1) < 0}
def N := {x : ℝ | x + 1 < 0}

-- State the theorem for the intersection M ∩ N
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < -1} :=
sorry

end intersection_of_M_and_N_l10_10251


namespace find_minimum_m_l10_10776

theorem find_minimum_m (m : ℕ) (h1 : 1350 + 36 * m < 2136) (h2 : 1500 + 45 * m ≥ 2365) :
  m = 20 :=
by
  sorry

end find_minimum_m_l10_10776


namespace min_value_expression_l10_10245

theorem min_value_expression (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1/2) :
  a^2 + 4 * a * b + 12 * b^2 + 8 * b * c + 3 * c^2 ≥ 18 :=
sorry

end min_value_expression_l10_10245


namespace trig_identity_l10_10905

theorem trig_identity (α : ℝ) (h : Real.sin (π + α) = 1 / 2) : Real.cos (α - 3 / 2 * π) = 1 / 2 :=
  sorry

end trig_identity_l10_10905


namespace blue_line_length_correct_l10_10653

def white_line_length : ℝ := 7.67
def difference_in_length : ℝ := 4.33
def blue_line_length : ℝ := 3.34

theorem blue_line_length_correct :
  white_line_length - difference_in_length = blue_line_length :=
by
  sorry

end blue_line_length_correct_l10_10653


namespace sum_between_9p5_and_10_l10_10311

noncomputable def sumMixedNumbers : ℚ :=
  (29 / 9) + (11 / 4) + (81 / 20)

theorem sum_between_9p5_and_10 :
  9.5 < sumMixedNumbers ∧ sumMixedNumbers < 10 :=
by
  sorry

end sum_between_9p5_and_10_l10_10311


namespace max_distance_increases_l10_10227

noncomputable def largest_n_for_rearrangement (C : ℕ) (marked_points : ℕ) : ℕ :=
  670

theorem max_distance_increases (C : ℕ) (marked_points : ℕ) (n : ℕ) (dist : ℕ → ℕ → ℕ) :
  ∀ i j, i < marked_points → j < marked_points →
    dist i j ≤ n → 
    (∃ rearrangement : ℕ → ℕ, 
    ∀ i j, i < marked_points → j < marked_points → 
      dist (rearrangement i) (rearrangement j) > dist i j) → 
    n ≤ largest_n_for_rearrangement C marked_points := 
by
  sorry

end max_distance_increases_l10_10227


namespace constant_speed_l10_10125

open Real

def total_trip_time := 50.0
def total_distance := 2790.0
def break_interval := 5.0
def break_duration := 0.5
def hotel_search_time := 0.5

theorem constant_speed :
  let number_of_breaks := total_trip_time / break_interval
  let total_break_time := number_of_breaks * break_duration
  let actual_driving_time := total_trip_time - total_break_time - hotel_search_time
  let constant_speed := total_distance / actual_driving_time
  constant_speed = 62.7 :=
by
  -- Provide proof here
  sorry

end constant_speed_l10_10125


namespace remaining_sweet_cookies_correct_remaining_salty_cookies_correct_remaining_chocolate_cookies_correct_l10_10907

-- Definition of initial conditions
def initial_sweet_cookies := 34
def initial_salty_cookies := 97
def initial_chocolate_cookies := 45

def sweet_cookies_eaten := 15
def salty_cookies_eaten := 56
def chocolate_cookies_given_away := 22
def chocolate_cookies_given_back := 7

-- Calculate remaining cookies
def remaining_sweet_cookies : Nat := initial_sweet_cookies - sweet_cookies_eaten
def remaining_salty_cookies : Nat := initial_salty_cookies - salty_cookies_eaten
def remaining_chocolate_cookies : Nat := (initial_chocolate_cookies - chocolate_cookies_given_away) + chocolate_cookies_given_back

-- Theorem statements
theorem remaining_sweet_cookies_correct : remaining_sweet_cookies = 19 := 
by sorry

theorem remaining_salty_cookies_correct : remaining_salty_cookies = 41 := 
by sorry

theorem remaining_chocolate_cookies_correct : remaining_chocolate_cookies = 30 := 
by sorry

end remaining_sweet_cookies_correct_remaining_salty_cookies_correct_remaining_chocolate_cookies_correct_l10_10907


namespace rationalize_denominator_l10_10937

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  A + B + C + D + E = 22 :=
by
  -- Proof goes here
  sorry

end rationalize_denominator_l10_10937


namespace tangent_lines_from_point_to_circle_l10_10068

theorem tangent_lines_from_point_to_circle : 
  ∀ (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ), 
  P = (2, 3) → C = (1, 1) → r = 1 → 
  (∃ k : ℝ, ((3 : ℝ) * P.1 - (4 : ℝ) * P.2 + 6 = 0) ∨ (P.1 = 2)) :=
by
  intros P C r hP hC hr
  sorry

end tangent_lines_from_point_to_circle_l10_10068


namespace distance_between_circle_center_and_point_l10_10912

theorem distance_between_circle_center_and_point (x y : ℝ) (h : x^2 + y^2 = 8*x - 12*y + 40) : 
  dist (4, -6) (4, -2) = 4 := 
by
  sorry

end distance_between_circle_center_and_point_l10_10912


namespace problem_statement_l10_10295

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

def candidate_function (x : ℝ) : ℝ :=
  x * |x|

theorem problem_statement : is_odd_function candidate_function ∧ is_increasing_function candidate_function :=
by
  sorry

end problem_statement_l10_10295


namespace necessary_and_sufficient_condition_l10_10143

theorem necessary_and_sufficient_condition :
  ∀ a b : ℝ, (a + b > 0) ↔ ((a ^ 3) + (b ^ 3) > 0) :=
by
  sorry

end necessary_and_sufficient_condition_l10_10143


namespace net_effect_sale_value_l10_10175

variable (P Q : ℝ) -- New price and quantity sold

theorem net_effect_sale_value (P Q : ℝ) :
  let new_sale_value := (0.75 * P) * (1.75 * Q)
  let original_sale_value := P * Q
  new_sale_value - original_sale_value = 0.3125 * (P * Q) := 
by
  sorry

end net_effect_sale_value_l10_10175


namespace total_viewing_time_l10_10450

theorem total_viewing_time (video_length : ℕ) (num_videos : ℕ) (lila_speed_factor : ℕ) :
  video_length = 100 ∧ num_videos = 6 ∧ lila_speed_factor = 2 →
  (num_videos * (video_length / lila_speed_factor) + num_videos * video_length) = 900 :=
by
  sorry

end total_viewing_time_l10_10450


namespace problem1_problem2_l10_10557

-- Definitions for the inequalities
def f (x a : ℝ) : ℝ := abs (x - a) - 1

-- Problem 1: Given a = 2, solve the inequality f(x) + |2x - 3| > 0
theorem problem1 (x : ℝ) (h1 : abs (x - 2) + abs (2 * x - 3) > 1) : (x ≥ 2 ∨ x ≤ 4 / 3) := sorry

-- Problem 2: If the inequality f(x) > |x - 3| has solutions, find the range of a
theorem problem2 (a : ℝ) (h2 : ∃ x : ℝ, abs (x - a) - abs (x - 3) > 1) : a < 2 ∨ a > 4 := sorry

end problem1_problem2_l10_10557


namespace greatest_multiple_of_four_cubed_less_than_2000_l10_10901

theorem greatest_multiple_of_four_cubed_less_than_2000 :
  ∃ x, (x > 0) ∧ (x % 4 = 0) ∧ (x^3 < 2000) ∧ ∀ y, (y > x) ∧ (y % 4 = 0) → y^3 ≥ 2000 :=
sorry

end greatest_multiple_of_four_cubed_less_than_2000_l10_10901


namespace laundry_lcm_l10_10254

theorem laundry_lcm :
  Nat.lcm (Nat.lcm 6 9) (Nat.lcm 12 15) = 180 :=
by
  sorry

end laundry_lcm_l10_10254


namespace max_profit_achieved_when_x_is_1_l10_10025

noncomputable def revenue (x : ℕ) : ℝ := 30 * x - 0.2 * x^2
noncomputable def fixed_costs : ℝ := 40
noncomputable def material_cost (x : ℕ) : ℝ := 5 * x
noncomputable def profit (x : ℕ) : ℝ := revenue x - (fixed_costs + material_cost x)
noncomputable def marginal_profit (x : ℕ) : ℝ := profit (x + 1) - profit x

theorem max_profit_achieved_when_x_is_1 :
  marginal_profit 1 = 24.40 :=
by
  -- Skip the proof
  sorry

end max_profit_achieved_when_x_is_1_l10_10025


namespace max_arithmetic_sequence_terms_l10_10276

theorem max_arithmetic_sequence_terms
  (n : ℕ)
  (a1 : ℝ)
  (d : ℝ) 
  (sum_sq_term_cond : (a1 + (n - 1) * d / 2)^2 + (n - 1) * (a1 + d * (n - 1) / 2) ≤ 100)
  (common_diff : d = 4)
  : n ≤ 8 := 
sorry

end max_arithmetic_sequence_terms_l10_10276


namespace number_of_continents_collected_l10_10249

-- Definitions of the given conditions
def books_per_continent : ℕ := 122
def total_books : ℕ := 488

-- The mathematical statement to be proved
theorem number_of_continents_collected :
  total_books / books_per_continent = 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_continents_collected_l10_10249


namespace sum_of_z_values_l10_10105

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 2

theorem sum_of_z_values (z : ℝ) : 
  (f (4 * z) = 13) → (∃ z1 z2 : ℝ, z1 = 1/8 ∧ z2 = -1/4 ∧ z1 + z2 = -1/8) :=
sorry

end sum_of_z_values_l10_10105


namespace incorrect_description_is_A_l10_10973

-- Definitions for the conditions
def description_A := "Increasing the concentration of reactants increases the percentage of activated molecules, accelerating the reaction rate."
def description_B := "Increasing the pressure of a gaseous reaction system increases the number of activated molecules per unit volume, accelerating the rate of the gas reaction."
def description_C := "Raising the temperature of the reaction increases the percentage of activated molecules, increases the probability of effective collisions, and increases the reaction rate."
def description_D := "Catalysts increase the reaction rate by changing the reaction path and lowering the activation energy required for the reaction."

-- Problem Statement
theorem incorrect_description_is_A :
  description_A ≠ correct :=
  sorry

end incorrect_description_is_A_l10_10973


namespace determine_f4_l10_10103

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem determine_f4 (f : ℝ → ℝ) (h_odd : odd_function f) (h_f_neg : ∀ x, x < 0 → f x = x * (2 - x)) : f 4 = 24 :=
by
  sorry

end determine_f4_l10_10103


namespace total_money_made_l10_10006

def num_coffee_customers : ℕ := 7
def price_per_coffee : ℕ := 5
def num_tea_customers : ℕ := 8
def price_per_tea : ℕ := 4

theorem total_money_made (h1 : num_coffee_customers = 7) (h2 : price_per_coffee = 5) 
  (h3 : num_tea_customers = 8) (h4 : price_per_tea = 4) : 
  (num_coffee_customers * price_per_coffee + num_tea_customers * price_per_tea) = 67 :=
by
  sorry

end total_money_made_l10_10006


namespace num_ordered_pairs_eq_1728_l10_10475

theorem num_ordered_pairs_eq_1728 (x y : ℕ) (h1 : 1728 = 2^6 * 3^3) (h2 : x * y = 1728) : 
  ∃ (n : ℕ), n = 28 := 
sorry

end num_ordered_pairs_eq_1728_l10_10475


namespace focus_of_parabola_l10_10400

theorem focus_of_parabola :
  (∃ (x y : ℝ), y = 4 * x ^ 2 - 8 * x - 12 ∧ x = 1 ∧ y = -15.9375) :=
by
  sorry

end focus_of_parabola_l10_10400


namespace trigonometric_identity_l10_10485

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 := 
by {
  sorry
}

end trigonometric_identity_l10_10485


namespace problem_bounds_l10_10225

theorem problem_bounds :
  ∀ (A_0 B_0 C_0 A_1 B_1 C_1 A_2 B_2 C_2 A_3 B_3 C_3 : Point),
    (A_0B_0 + B_0C_0 + C_0A_0 = 1) →
    (A_1B_1 = A_0B_0) →
    (B_1C_1 = B_0C_0) →
    (A_2 = A_1 ∧ B_2 = B_1 ∧ C_2 = C_1 ∨
     A_2 = A_1 ∧ B_2 = C_1 ∧ C_2 = B_1 ∨
     A_2 = B_1 ∧ B_2 = A_1 ∧ C_2 = C_1 ∨
     A_2 = B_1 ∧ B_2 = C_1 ∧ C_2 = A_1 ∨
     A_2 = C_1 ∧ B_2 = A_1 ∧ C_2 = B_1 ∨
     A_2 = C_1 ∧ B_2 = B_1 ∧ C_2 = A_1) →
    (A_3B_3 = A_2B_2) →
    (B_3C_3 = B_2C_2) →
    (A_3B_3 + B_3C_3 + C_3A_3) ≥ 1 / 3 ∧ 
    (A_3B_3 + B_3C_3 + C_3A_3) ≤ 3 :=
by
  -- Proof goes here
  sorry

end problem_bounds_l10_10225


namespace Jacob_has_48_graham_crackers_l10_10829

def marshmallows_initial := 6
def marshmallows_needed := 18
def marshmallows_total := marshmallows_initial + marshmallows_needed
def graham_crackers_per_smore := 2

def smores_total := marshmallows_total
def graham_crackers_total := smores_total * graham_crackers_per_smore

theorem Jacob_has_48_graham_crackers (h1 : marshmallows_initial = 6)
                                     (h2 : marshmallows_needed = 18)
                                     (h3 : graham_crackers_per_smore = 2)
                                     (h4 : marshmallows_total = marshmallows_initial + marshmallows_needed)
                                     (h5 : smores_total = marshmallows_total)
                                     (h6 : graham_crackers_total = smores_total * graham_crackers_per_smore) :
                                     graham_crackers_total = 48 :=
by
  sorry

end Jacob_has_48_graham_crackers_l10_10829


namespace shanghai_mock_exam_problem_l10_10162

noncomputable def a_n : ℕ → ℝ := sorry -- Defines the arithmetic sequence 

theorem shanghai_mock_exam_problem 
  (a_is_arithmetic : ∃ d a₀, ∀ n, a_n n = a₀ + n * d)
  (h₁ : a_n 1 + a_n 3 + a_n 5 = 9)
  (h₂ : a_n 2 + a_n 4 + a_n 6 = 15) :
  a_n 3 + a_n 4 = 8 := 
  sorry

end shanghai_mock_exam_problem_l10_10162


namespace proof_problem_exists_R1_R2_l10_10270

def problem (R1 R2 : ℕ) : Prop :=
  let F1_R1 := (4 * R1 + 5) / (R1^2 - 1)
  let F2_R1 := (5 * R1 + 4) / (R1^2 - 1)
  let F1_R2 := (3 * R2 + 2) / (R2^2 - 1)
  let F2_R2 := (2 * R2 + 3) / (R2^2 - 1)
  F1_R1 = F1_R2 ∧ F2_R1 = F2_R2 ∧ R1 + R2 = 14

theorem proof_problem_exists_R1_R2 : ∃ (R1 R2 : ℕ), problem R1 R2 :=
sorry

end proof_problem_exists_R1_R2_l10_10270


namespace positive_m_for_one_root_l10_10629

theorem positive_m_for_one_root (m : ℝ) (h : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
by
  sorry

end positive_m_for_one_root_l10_10629


namespace central_angle_of_sector_l10_10253

variable (r θ : ℝ)
variable (r_pos : 0 < r) (θ_pos : 0 < θ)

def perimeter_eq : Prop := 2 * r + r * θ = 5
def area_eq : Prop := (1 / 2) * r^2 * θ = 1

theorem central_angle_of_sector :
  perimeter_eq r θ ∧ area_eq r θ → θ = 1 / 2 :=
sorry

end central_angle_of_sector_l10_10253


namespace max_days_for_same_shift_l10_10119

open BigOperators

-- We define the given conditions
def nurses : ℕ := 15
def shifts_per_day : ℕ := 24 / 8
noncomputable def total_pairs : ℕ := (nurses.choose 2)

-- The main statement to prove
theorem max_days_for_same_shift : 
  35 = total_pairs / shifts_per_day := by
  sorry

end max_days_for_same_shift_l10_10119


namespace combination_indices_l10_10697
open Nat

theorem combination_indices (x : ℕ) (h : choose 18 x = choose 18 (3 * x - 6)) : x = 3 ∨ x = 6 :=
by
  sorry

end combination_indices_l10_10697


namespace min_value_of_quadratic_l10_10128

theorem min_value_of_quadratic :
  ∃ (x y : ℝ), 2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x - 6 * y + 1 = -3 :=
sorry

end min_value_of_quadratic_l10_10128


namespace consistent_system_l10_10432

variable (x y : ℕ)

def condition1 := x + y = 40
def condition2 := 2 * 15 * x = 20 * y

theorem consistent_system :
  condition1 x y ∧ condition2 x y ↔ 
  (x + y = 40 ∧ 2 * 15 * x = 20 * y) :=
by
  sorry

end consistent_system_l10_10432


namespace simplify_A_plus_2B_value_A_plus_2B_at_a1_bneg1_l10_10854

variable (a b : ℤ)

def A : ℤ := 3 * a^2 - 6 * a * b + b^2
def B : ℤ := -2 * a^2 + 3 * a * b - 5 * b^2

theorem simplify_A_plus_2B : 
  A a b + 2 * B a b = -a^2 - 9 * b^2 := by
  sorry

theorem value_A_plus_2B_at_a1_bneg1 : 
  let a := 1
  let b := -1
  A a b + 2 * B a b = -10 := by
  sorry

end simplify_A_plus_2B_value_A_plus_2B_at_a1_bneg1_l10_10854


namespace four_letter_list_product_l10_10618

def letter_value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then (c.toNat - 'A'.toNat + 1) else 0

def list_product (s : String) : Nat :=
  s.foldl (λ acc c => acc * letter_value c) 1

def target_product : Nat :=
  list_product "TUVW"

theorem four_letter_list_product : 
  ∀ (s1 s2 : String), s1.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') → s2.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') →
  s1.length = 4 → s2.length = 4 →
  list_product s1 = target_product → s2 = "BEHK" :=
by
  sorry

end four_letter_list_product_l10_10618


namespace best_fitting_model_l10_10229

theorem best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.98)
  (h2 : R2_2 = 0.80)
  (h3 : R2_3 = 0.50)
  (h4 : R2_4 = 0.25) :
  R2_1 = 0.98 ∧ R2_1 > R2_2 ∧ R2_1 > R2_3 ∧ R2_1 > R2_4 :=
by { sorry }

end best_fitting_model_l10_10229


namespace find_number_l10_10509

theorem find_number (X : ℝ) (h : 50 = 0.20 * X + 47) : X = 15 :=
sorry

end find_number_l10_10509


namespace range_a_for_inequality_l10_10925

theorem range_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, (a-2) * x^2 - 2 * (a-2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
by
  sorry

end range_a_for_inequality_l10_10925


namespace preimage_of_43_is_21_l10_10564

def f (x y : ℝ) : ℝ × ℝ := (x + 2 * y, 2 * x - y)

theorem preimage_of_43_is_21 : f 2 1 = (4, 3) :=
by {
  -- Proof omitted
  sorry
}

end preimage_of_43_is_21_l10_10564


namespace difference_is_correct_l10_10066

-- Define the digits
def digits : List ℕ := [9, 2, 1, 5]

-- Define the largest number that can be formed by these digits
def largestNumber : ℕ :=
  1000 * 9 + 100 * 5 + 10 * 2 + 1 * 1

-- Define the smallest number that can be formed by these digits
def smallestNumber : ℕ :=
  1000 * 1 + 100 * 2 + 10 * 5 + 1 * 9

-- Define the correct difference
def difference : ℕ :=
  largestNumber - smallestNumber

-- Theorem statement
theorem difference_is_correct : difference = 8262 :=
by
  sorry

end difference_is_correct_l10_10066


namespace identify_worst_player_l10_10124

-- Define the participants
inductive Participant
| father
| sister
| son
| daughter

open Participant

-- Conditions
def participants : List Participant :=
  [father, sister, son, daughter]

def twins (p1 p2 : Participant) : Prop := 
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father) ∨
  (p1 = son ∧ p2 = daughter) ∨
  (p1 = daughter ∧ p2 = son)

def not_same_sex (p1 p2 : Participant) : Prop :=
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father) ∨
  (p1 = son ∧ p2 = daughter) ∨
  (p1 = daughter ∧ p2 = son)

def older_by_one_year (p1 p2 : Participant) : Prop :=
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father)

-- Question: who is the worst player?
def worst_player : Participant := sister

-- Proof statement
theorem identify_worst_player
  (h_twins : ∃ p1 p2, twins p1 p2)
  (h_not_same_sex : ∀ p1 p2, twins p1 p2 → not_same_sex p1 p2)
  (h_age_diff : ∀ p1 p2, twins p1 p2 → older_by_one_year p1 p2) :
  worst_player = sister :=
sorry

end identify_worst_player_l10_10124


namespace ascending_order_conversion_l10_10764

def convert_base (num : Nat) (base : Nat) : Nat :=
  match num with
  | 0 => 0
  | _ => (num / 10) * base + (num % 10)

theorem ascending_order_conversion :
  let num16 := 12
  let num7 := 25
  let num4 := 33
  let base16 := 16
  let base7 := 7
  let base4 := 4
  convert_base num4 base4 < convert_base num16 base16 ∧ 
  convert_base num16 base16 < convert_base num7 base7 :=
by
  -- Here would be the proof, but we skip it
  sorry

end ascending_order_conversion_l10_10764


namespace ratio_perimeter_triangle_square_l10_10977

/-
  Suppose a square piece of paper with side length 4 units is folded in half diagonally.
  The folded paper is then cut along the fold, producing two right-angled triangles.
  We need to prove that the ratio of the perimeter of one of the triangles to the perimeter of the original square is (1/2) + (sqrt 2 / 4).
-/
theorem ratio_perimeter_triangle_square:
  let side_length := 4
  let triangle_leg := side_length
  let hypotenuse := Real.sqrt (triangle_leg ^ 2 + triangle_leg ^ 2)
  let perimeter_triangle := triangle_leg + triangle_leg + hypotenuse
  let perimeter_square := 4 * side_length
  let ratio := perimeter_triangle / perimeter_square
  ratio = (1 / 2) + (Real.sqrt 2 / 4) :=
by
  sorry

end ratio_perimeter_triangle_square_l10_10977


namespace right_triangle_AB_is_approximately_8point3_l10_10455

noncomputable def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

theorem right_triangle_AB_is_approximately_8point3 :
  ∀ (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (BC AB : ℝ),
  angle_A = 40 ∧ angle_B = 90 ∧ BC = 7 →
  AB = 7 / tan_deg 40 →
  abs (AB - 8.3) < 0.1 :=
by
  intros A B C angle_A angle_B BC AB h_cond h_AB
  sorry

end right_triangle_AB_is_approximately_8point3_l10_10455


namespace circle_radius_on_sphere_l10_10917

theorem circle_radius_on_sphere
  (sphere_radius : ℝ)
  (circle1_radius : ℝ)
  (circle2_radius : ℝ)
  (circle3_radius : ℝ)
  (all_circle_touch_each_other : Prop)
  (smaller_circle_touches_all : Prop)
  (smaller_circle_radius : ℝ) :
  sphere_radius = 2 →
  circle1_radius = 1 →
  circle2_radius = 1 →
  circle3_radius = 1 →
  all_circle_touch_each_other →
  smaller_circle_touches_all →
  smaller_circle_radius = 1 - Real.sqrt (2 / 3) :=
by
  intros h_sphere_radius h_circle1_radius h_circle2_radius h_circle3_radius h_all_circle_touch h_smaller_circle_touch
  sorry

end circle_radius_on_sphere_l10_10917


namespace prove_f_cos_eq_l10_10848

variable (f : ℝ → ℝ)

theorem prove_f_cos_eq :
  (∀ x : ℝ, f (Real.sin x) = 3 - Real.cos (2 * x)) →
  (∀ x : ℝ, f (Real.cos x) = 3 + Real.cos (2 * x)) :=
by
  sorry

end prove_f_cos_eq_l10_10848


namespace can_form_triangle_l10_10029

theorem can_form_triangle (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  (a = 7 ∧ b = 12 ∧ c = 17) → True :=
by
  sorry

end can_form_triangle_l10_10029


namespace cellphone_surveys_l10_10041

theorem cellphone_surveys
  (regular_rate : ℕ)
  (total_surveys : ℕ)
  (higher_rate_multiplier : ℕ)
  (total_earnings : ℕ)
  (higher_rate_bonus : ℕ)
  (x : ℕ) :
  regular_rate = 10 → total_surveys = 100 →
  higher_rate_multiplier = 130 → total_earnings = 1180 →
  higher_rate_bonus = 3 → (10 * (100 - x) + 13 * x = 1180) →
  x = 60 :=
by
  sorry

end cellphone_surveys_l10_10041


namespace books_loaned_out_l10_10989

/-- 
Given:
- There are 75 books in a special collection at the beginning of the month.
- By the end of the month, 70 percent of books that were loaned out are returned.
- There are 60 books in the special collection at the end of the month.
Prove:
- The number of books loaned out during the month is 50.
-/
theorem books_loaned_out (x : ℝ) (h1 : 75 - 0.3 * x = 60) : x = 50 :=
by
  sorry

end books_loaned_out_l10_10989


namespace expected_audience_l10_10096

theorem expected_audience (Sat Mon Wed Fri : ℕ) (extra_people expected_total : ℕ)
  (h1 : Sat = 80)
  (h2 : Mon = 80 - 20)
  (h3 : Wed = Mon + 50)
  (h4 : Fri = Sat + Mon)
  (h5 : extra_people = 40)
  (h6 : expected_total = Sat + Mon + Wed + Fri - extra_people) :
  expected_total = 350 := 
sorry

end expected_audience_l10_10096


namespace cousin_age_result_l10_10897

-- Let define the ages
def rick_age : ℕ := 15
def oldest_brother_age : ℕ := 2 * rick_age
def middle_brother_age : ℕ := oldest_brother_age / 3
def smallest_brother_age : ℕ := middle_brother_age / 2
def youngest_brother_age : ℕ := smallest_brother_age - 2
def cousin_age : ℕ := 5 * youngest_brother_age

-- The theorem stating the cousin's age.
theorem cousin_age_result : cousin_age = 15 := by
  sorry

end cousin_age_result_l10_10897


namespace largest_divisor_of_consecutive_odd_integers_l10_10070

theorem largest_divisor_of_consecutive_odd_integers :
  ∀ (x : ℤ), (∃ (d : ℤ) (m : ℤ), d = 48 ∧ (x * (x + 2) * (x + 4) * (x + 6)) = d * m) :=
by 
-- We assert that for any integer x, 48 always divides the product of
-- four consecutive odd integers starting from x
sorry

end largest_divisor_of_consecutive_odd_integers_l10_10070


namespace geometric_sequence_problem_l10_10535

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ a₁ q : ℝ, ∀ n, a n = a₁ * q^n

axiom a_3_eq_2 : ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 3 = 2
axiom a_4a_6_eq_16 : ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 4 * a 6 = 16

theorem geometric_sequence_problem :
  ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 3 = 2 ∧ a 4 * a 6 = 16 →
  (a 9 - a 11) / (a 5 - a 7) = 4 :=
sorry

end geometric_sequence_problem_l10_10535


namespace remainder_when_divided_by_5_l10_10670

theorem remainder_when_divided_by_5 
  (k : ℕ)
  (h1 : k % 6 = 5)
  (h2 : k < 42)
  (h3 : k % 7 = 3) : 
  k % 5 = 2 := 
by 
  sorry

end remainder_when_divided_by_5_l10_10670


namespace solution_set_of_log_inequality_l10_10434

noncomputable def log_a (a x : ℝ) : ℝ := sorry -- The precise definition of the log base 'a' is skipped for brevity.

theorem solution_set_of_log_inequality (a x : ℝ)
  (ha_pos : a > 0)
  (ha_ne_one : a ≠ 1)
  (h_max : ∃ y, log_a a (y^2 - 2*y + 3) = y):
  log_a a (x - 1) > 0 ↔ (1 < x ∧ x < 2) :=
sorry

end solution_set_of_log_inequality_l10_10434


namespace chairs_stools_legs_l10_10489

theorem chairs_stools_legs (x : ℕ) (h1 : 4 * x + 3 * (16 - x) = 60) : 4 * x + 3 * (16 - x) = 60 :=
by
  exact h1

end chairs_stools_legs_l10_10489


namespace students_with_uncool_family_l10_10650

-- Define the conditions as given in the problem.
variables (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool_parents : ℕ)
          (cool_siblings : ℕ) (cool_siblings_and_dads : ℕ)

-- Provide the known values as conditions.
def problem_conditions := 
  total_students = 50 ∧
  cool_dads = 20 ∧
  cool_moms = 25 ∧
  both_cool_parents = 12 ∧
  cool_siblings = 5 ∧
  cool_siblings_and_dads = 3

-- State the problem: prove the number of students with all uncool family members.
theorem students_with_uncool_family : problem_conditions total_students cool_dads cool_moms 
                                            both_cool_parents cool_siblings cool_siblings_and_dads →
                                    (50 - ((20 - 12) + (25 - 12) + 12 + (5 - 3)) = 15) :=
by intros h; cases h; sorry

end students_with_uncool_family_l10_10650


namespace parallel_vectors_l10_10130

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (2, m)

theorem parallel_vectors (m : ℝ) (h : ∃ k : ℝ, vector_a = (k • vector_b m)) : m = -4 :=
by {
  sorry
}

end parallel_vectors_l10_10130


namespace bus_speed_excluding_stoppages_l10_10298

theorem bus_speed_excluding_stoppages :
  ∀ (S : ℝ), (45 = (3 / 4) * S) → (S = 60) :=
by 
  intros S h
  sorry

end bus_speed_excluding_stoppages_l10_10298


namespace integral_solution_unique_l10_10382

theorem integral_solution_unique (a b c : ℤ) : a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end integral_solution_unique_l10_10382


namespace parameterization_solution_l10_10108

/-- Proof problem statement:
  Given the line equation y = 3x - 11 and its parameterization representation,
  the ordered pair (s, h) that satisfies both conditions is (3, 15).
-/
theorem parameterization_solution : ∃ s h : ℝ, 
  (∀ t : ℝ, (∃ x y : ℝ, (x, y) = (s, -2) + t • (5, h)) ∧ y = 3 * x - 11) → 
  (s = 3 ∧ h = 15) :=
by
  -- introduce s and h 
  use 3
  use 15
  -- skip the proof
  sorry

end parameterization_solution_l10_10108


namespace z_real_iff_m_1_or_2_z_complex_iff_not_m_1_and_2_z_pure_imaginary_iff_m_neg_half_z_in_second_quadrant_l10_10171

variables (m : ℝ)

def z_re (m : ℝ) : ℝ := 2 * m^2 - 3 * m - 2
def z_im (m : ℝ) : ℝ := m^2 - 3 * m + 2

-- Part (Ⅰ) Question 1
theorem z_real_iff_m_1_or_2 (m : ℝ) :
  z_im m = 0 ↔ (m = 1 ∨ m = 2) :=
sorry

-- Part (Ⅰ) Question 2
theorem z_complex_iff_not_m_1_and_2 (m : ℝ) :
  ¬ (m = 1 ∨ m = 2) ↔ (m ≠ 1 ∧ m ≠ 2) :=
sorry

-- Part (Ⅰ) Question 3
theorem z_pure_imaginary_iff_m_neg_half (m : ℝ) :
  z_re m = 0 ∧ z_im m ≠ 0 ↔ (m = -1/2) :=
sorry

-- Part (Ⅱ) Question
theorem z_in_second_quadrant (m : ℝ) :
  z_re m < 0 ∧ z_im m > 0 ↔ -1/2 < m ∧ m < 1 :=
sorry

end z_real_iff_m_1_or_2_z_complex_iff_not_m_1_and_2_z_pure_imaginary_iff_m_neg_half_z_in_second_quadrant_l10_10171


namespace regular_14_gon_inequality_l10_10445

noncomputable def side_length_of_regular_14_gon : ℝ := 2 * Real.sin (Real.pi / 14)

theorem regular_14_gon_inequality (a : ℝ) (h : a = side_length_of_regular_14_gon) :
  (2 - a) / (2 * a) > Real.sqrt (3 * Real.cos (Real.pi / 7)) :=
by
  sorry

end regular_14_gon_inequality_l10_10445


namespace equal_share_payment_l10_10586

theorem equal_share_payment (A B C : ℝ) (h : A < B) (h2 : B < C) :
  (B + C + (A + C - 2 * B) / 3) + (A + C - 2 * B / 3) = 2 * C - A - B / 3 :=
sorry

end equal_share_payment_l10_10586


namespace arithmetic_sequence_n_equals_8_l10_10974

theorem arithmetic_sequence_n_equals_8
  (a : ℕ → ℝ) 
  (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) 
  (h2 : a 2 + a 5 = 18)
  (h3 : a 3 * a 4 = 32)
  (h_n : ∃ n, a n = 128) :
  ∃ n, a n = 128 ∧ n = 8 := 
sorry

end arithmetic_sequence_n_equals_8_l10_10974


namespace matches_needed_eq_l10_10705

def count_matches (n : ℕ) : ℕ :=
  let total_triangles := n * n
  let internal_matches := 3 * total_triangles
  let external_matches := 4 * n
  internal_matches - external_matches + external_matches

theorem matches_needed_eq (n : ℕ) : count_matches 10 = 320 :=
by
  sorry

end matches_needed_eq_l10_10705


namespace marcus_goal_points_value_l10_10002

-- Definitions based on conditions
def marcus_goals_first_type := 5
def marcus_goals_second_type := 10
def second_type_goal_points := 2
def team_total_points := 70
def marcus_percentage_points := 50

-- Theorem statement
theorem marcus_goal_points_value : 
  ∃ (x : ℕ), 5 * x + 10 * 2 = 35 ∧ 35 = 50 * team_total_points / 100 := 
sorry

end marcus_goal_points_value_l10_10002


namespace club_members_addition_l10_10484

theorem club_members_addition
  (current_members : ℕ := 10)
  (desired_members : ℕ := 2 * current_members + 5)
  (additional_members : ℕ := desired_members - current_members) :
  additional_members = 15 :=
by
  -- proof placeholder
  sorry

end club_members_addition_l10_10484


namespace given_conditions_implies_a1d1_a2d2_a3d3_eq_zero_l10_10481

theorem given_conditions_implies_a1d1_a2d2_a3d3_eq_zero
  (a1 a2 a3 d1 d2 d3 : ℝ)
  (h : ∀ x : ℝ, 
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
    (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 - x + 1)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 0 :=
by
  sorry

end given_conditions_implies_a1d1_a2d2_a3d3_eq_zero_l10_10481


namespace largest_difference_l10_10913

theorem largest_difference (P Q R S T U : ℕ) 
    (hP : P = 3 * 2003 ^ 2004)
    (hQ : Q = 2003 ^ 2004)
    (hR : R = 2002 * 2003 ^ 2003)
    (hS : S = 3 * 2003 ^ 2003)
    (hT : T = 2003 ^ 2003)
    (hU : U = 2003 ^ 2002) 
    : max (P - Q) (max (Q - R) (max (R - S) (max (S - T) (T - U)))) = P - Q :=
sorry

end largest_difference_l10_10913


namespace median_interval_60_64_l10_10163

theorem median_interval_60_64 
  (students : ℕ) 
  (f_45_49 f_50_54 f_55_59 f_60_64 : ℕ) :
  students = 105 ∧ 
  f_45_49 = 8 ∧ 
  f_50_54 = 15 ∧ 
  f_55_59 = 20 ∧ 
  f_60_64 = 18 ∧ 
  (8 + 15 + 20 + 18) ≥ (105 + 1) / 2
  → 60 ≤ (105 + 1) / 2  ∧ (105 + 1) / 2 ≤ 64 :=
sorry

end median_interval_60_64_l10_10163


namespace fraction_of_orange_juice_correct_l10_10987

-- Define the capacities of the pitchers
def capacity := 800

-- Define the fractions of orange juice and apple juice in the first pitcher
def orangeJuiceFraction1 := 1 / 4
def appleJuiceFraction1 := 1 / 8

-- Define the fractions of orange juice and apple juice in the second pitcher
def orangeJuiceFraction2 := 1 / 5
def appleJuiceFraction2 := 1 / 10

-- Define the total volumes of the contents in each pitcher
def totalVolume := 2 * capacity -- total volume in the large container after pouring

-- Define the orange juice volumes in each pitcher
def orangeJuiceVolume1 := orangeJuiceFraction1 * capacity
def orangeJuiceVolume2 := orangeJuiceFraction2 * capacity

-- Calculate the total volume of orange juice in the large container
def totalOrangeJuiceVolume := orangeJuiceVolume1 + orangeJuiceVolume2

-- Define the fraction of orange juice in the large container
def orangeJuiceFraction := totalOrangeJuiceVolume / totalVolume

theorem fraction_of_orange_juice_correct :
  orangeJuiceFraction = 9 / 40 :=
by
  sorry

end fraction_of_orange_juice_correct_l10_10987


namespace quadratic_square_binomial_l10_10970

theorem quadratic_square_binomial (d : ℝ) : (∃ b : ℝ, (x : ℝ) -> (x + b)^2 = x^2 + 110 * x + d) ↔ d = 3025 :=
by
  sorry

end quadratic_square_binomial_l10_10970


namespace prime_iff_even_and_power_of_two_l10_10945

theorem prime_iff_even_and_power_of_two (a n : ℕ) (h_pos_a : a > 1) (h_pos_n : n > 0) :
  Nat.Prime (a^n + 1) → (∃ k : ℕ, a = 2 * k) ∧ (∃ m : ℕ, n = 2^m) :=
by 
  sorry

end prime_iff_even_and_power_of_two_l10_10945


namespace relationship_xy_l10_10773

def M (x : ℤ) : Prop := ∃ m : ℤ, x = 3 * m + 1
def N (y : ℤ) : Prop := ∃ n : ℤ, y = 3 * n + 2

theorem relationship_xy (x y : ℤ) (hx : M x) (hy : N y) : N (x * y) ∧ ¬ M (x * y) :=
by
  sorry

end relationship_xy_l10_10773


namespace exponent_multiplication_l10_10541

variable (a x y : ℝ)

theorem exponent_multiplication :
  a^x = 2 →
  a^y = 3 →
  a^(x + y) = 6 :=
by
  intros h1 h2
  sorry

end exponent_multiplication_l10_10541


namespace quadrant_of_P_l10_10223

theorem quadrant_of_P (m n : ℝ) (h1 : m * n > 0) (h2 : m + n < 0) : (m < 0 ∧ n < 0) :=
by
  sorry

end quadrant_of_P_l10_10223


namespace largest_possible_A_l10_10505

-- Define natural numbers
variables (A B C : ℕ)

-- Given conditions
def division_algorithm (A B C : ℕ) : Prop := A = 8 * B + C
def B_equals_C (B C : ℕ) : Prop := B = C

-- The proof statement
theorem largest_possible_A (h1 : division_algorithm A B C) (h2 : B_equals_C B C) : A = 63 :=
by
  -- Proof is omitted
  sorry

end largest_possible_A_l10_10505


namespace find_y_l10_10971

theorem find_y (x y : ℤ) (h₁ : x ^ 2 + x + 4 = y - 4) (h₂ : x = 3) : y = 20 :=
by 
  sorry

end find_y_l10_10971


namespace functional_equation_unique_solution_l10_10071

theorem functional_equation_unique_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, a + f b + f (f c) = 0 → f a ^ 3 + b * f b ^ 2 + c ^ 2 * f c = 3 * a * b * c) →
  (∀ x : ℝ, f x = x ∨ f x = -x ∨ f x = 0) :=
by
  sorry

end functional_equation_unique_solution_l10_10071


namespace average_speed_ratio_l10_10284

theorem average_speed_ratio 
  (jack_marathon_distance : ℕ) (jack_marathon_time : ℕ) 
  (jill_marathon_distance : ℕ) (jill_marathon_time : ℕ)
  (h1 : jack_marathon_distance = 40) (h2 : jack_marathon_time = 45) 
  (h3 : jill_marathon_distance = 40) (h4 : jill_marathon_time = 40) :
  (889 : ℕ) / 1000 = (jack_marathon_distance / jack_marathon_time) / 
                      (jill_marathon_distance / jill_marathon_time) :=
by
  sorry

end average_speed_ratio_l10_10284


namespace find_term_ninth_term_l10_10708

variable (a_1 d a_k a_12 : ℤ)
variable (S_20 : ℤ := 200)

-- Definitions of the given conditions
def term_n (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := a_1 + (n - 1) * d

-- Problem Statement
theorem find_term_ninth_term :
  (∃ k, term_n a_1 d k + term_n a_1 d 12 = 20) ∧ 
  (S_20 = 10 * (2 * a_1 + 19 * d)) → 
  ∃ k, k = 9 :=
by sorry

end find_term_ninth_term_l10_10708


namespace qatar_location_is_accurate_l10_10846

def qatar_geo_location :=
  "The most accurate representation of Qatar's geographical location is latitude 25 degrees North, longitude 51 degrees East."

theorem qatar_location_is_accurate :
  qatar_geo_location = "The most accurate representation of Qatar's geographical location is latitude 25 degrees North, longitude 51 degrees East." :=
sorry

end qatar_location_is_accurate_l10_10846


namespace problem_statement_l10_10921

-- Define the arithmetic sequence and required terms
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
variables (a : ℕ → ℝ) (d : ℝ)
axiom seq_is_arithmetic : arithmetic_seq a d
axiom sum_of_a2_a4_a6_is_3 : a 2 + a 4 + a 6 = 3

-- Goal: Prove a1 + a3 + a5 + a7 = 4
theorem problem_statement : a 1 + a 3 + a 5 + a 7 = 4 :=
by 
  sorry

end problem_statement_l10_10921


namespace calculation_l10_10415

theorem calculation : 2005^2 - 2003 * 2007 = 4 :=
by
  have h1 : 2003 = 2005 - 2 := by rfl
  have h2 : 2007 = 2005 + 2 := by rfl
  sorry

end calculation_l10_10415


namespace find_amplitude_l10_10464

theorem find_amplitude (A D : ℝ) (h1 : D + A = 5) (h2 : D - A = -3) : A = 4 :=
by
  sorry

end find_amplitude_l10_10464


namespace value_added_to_half_is_five_l10_10416

theorem value_added_to_half_is_five (n V : ℕ) (h₁ : n = 16) (h₂ : (1 / 2 : ℝ) * n + V = 13) : V = 5 := 
by 
  sorry

end value_added_to_half_is_five_l10_10416


namespace problem_statement_l10_10459

noncomputable def f (x : ℝ) : ℝ :=
  1 - x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2

theorem problem_statement : f (1 / 2) + f (-1 / 2) = 2 := sorry

end problem_statement_l10_10459


namespace chocolate_candies_total_cost_l10_10244

-- Condition 1: A box of 30 chocolate candies costs $7.50
def box_cost : ℝ := 7.50
def candies_per_box : ℕ := 30

-- Condition 2: The local sales tax rate is 10%
def sales_tax_rate : ℝ := 0.10

-- Total number of candies to be bought
def total_candy_count : ℕ := 540

-- Calculate the number of boxes needed
def number_of_boxes (total_candies : ℕ) (candies_per_box : ℕ) : ℕ :=
  total_candies / candies_per_box

-- Calculate the cost without tax
def cost_without_tax (num_boxes : ℕ) (cost_per_box : ℝ) : ℝ :=
  num_boxes * cost_per_box

-- Calculate the total cost including tax
def total_cost_with_tax (cost : ℝ) (tax_rate : ℝ) : ℝ :=
  cost * (1 + tax_rate)

-- The main statement
theorem chocolate_candies_total_cost :
  total_cost_with_tax 
    (cost_without_tax (number_of_boxes total_candy_count candies_per_box) box_cost)
    sales_tax_rate = 148.50 :=
by
  sorry

end chocolate_candies_total_cost_l10_10244


namespace solve_for_x_l10_10334

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.8) : x = 71.7647 := 
by 
  sorry

end solve_for_x_l10_10334


namespace min_value_ge_54_l10_10605

open Real

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) : ℝ :=
2 * x + 3 * y + 6 * z

theorem min_value_ge_54 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) :
  min_value x y z h1 h2 h3 h4 ≥ 54 :=
sorry

end min_value_ge_54_l10_10605


namespace basketball_game_l10_10144

theorem basketball_game (a r b d : ℕ) (r_gt_1 : r > 1) (d_gt_0 : d > 0)
  (H1 : a = b)
  (H2 : a * (1 + r) * (1 + r^2) = 4 * b + 6 * d + 2)
  (H3 : a * (1 + r) * (1 + r^2) ≤ 100)
  (H4 : 4 * b + 6 * d ≤ 98) :
  (a + a * r) + (b + (b + d)) = 43 := 
sorry

end basketball_game_l10_10144


namespace no_positive_integer_makes_sum_prime_l10_10123

theorem no_positive_integer_makes_sum_prime : ¬ ∃ n : ℕ, 0 < n ∧ Prime (4^n + n^4) :=
by
  sorry

end no_positive_integer_makes_sum_prime_l10_10123


namespace jonathan_fourth_task_completion_l10_10769

-- Conditions
def start_time : Nat := 9 * 60 -- 9:00 AM in minutes
def third_task_completion_time : Nat := 11 * 60 + 30 -- 11:30 AM in minutes
def number_of_tasks : Nat := 4
def number_of_completed_tasks : Nat := 3

-- Calculation of time duration
def total_time_first_three_tasks : Nat :=
  third_task_completion_time - start_time

def duration_of_one_task : Nat :=
  total_time_first_three_tasks / number_of_completed_tasks
  
-- Statement to prove
theorem jonathan_fourth_task_completion :
  (third_task_completion_time + duration_of_one_task) = (12 * 60 + 20) :=
  by
    -- We do not need to provide the proof steps as per instructions
    sorry

end jonathan_fourth_task_completion_l10_10769


namespace selection_including_both_genders_is_34_l10_10307

def count_ways_to_select_students_with_conditions (total_students boys girls select_students : ℕ) : ℕ :=
  if total_students = 7 ∧ boys = 4 ∧ girls = 3 ∧ select_students = 4 then
    (Nat.choose total_students select_students) - 1
  else
    0

theorem selection_including_both_genders_is_34 :
  count_ways_to_select_students_with_conditions 7 4 3 4 = 34 :=
by
  -- The proof would go here
  sorry

end selection_including_both_genders_is_34_l10_10307


namespace total_amount_spent_l10_10777

def cost_of_soft_drink : ℕ := 2
def cost_per_candy_bar : ℕ := 5
def number_of_candy_bars : ℕ := 5

theorem total_amount_spent : cost_of_soft_drink + cost_per_candy_bar * number_of_candy_bars = 27 := by
  sorry

end total_amount_spent_l10_10777


namespace quadratic_two_distinct_real_roots_l10_10031

theorem quadratic_two_distinct_real_roots (k : ℝ) :
  2 * k ≠ 0 → (8 * k + 1)^2 - 64 * k^2 > 0 → k > -1 / 16 ∧ k ≠ 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l10_10031


namespace garbage_bill_problem_l10_10052

theorem garbage_bill_problem
  (R : ℝ)
  (trash_bins : ℝ := 2)
  (recycling_bins : ℝ := 1)
  (weekly_trash_cost_per_bin : ℝ := 10)
  (weeks_per_month : ℝ := 4)
  (discount_rate : ℝ := 0.18)
  (fine : ℝ := 20)
  (final_bill : ℝ := 102) :
  (trash_bins * weekly_trash_cost_per_bin * weeks_per_month + recycling_bins * R * weeks_per_month)
  - discount_rate * (trash_bins * weekly_trash_cost_per_bin * weeks_per_month + recycling_bins * R * weeks_per_month)
  + fine = final_bill →
  R = 5 := 
by
  sorry

end garbage_bill_problem_l10_10052


namespace downstream_speed_l10_10395

variable (V_u V_s V_d : ℝ)

theorem downstream_speed (h1 : V_u = 22) (h2 : V_s = 32) (h3 : V_s = (V_u + V_d) / 2) : V_d = 42 :=
sorry

end downstream_speed_l10_10395


namespace zeros_of_quadratic_l10_10094

theorem zeros_of_quadratic : ∃ x : ℝ, x^2 - x - 2 = 0 -> (x = -1 ∨ x = 2) :=
by
  sorry

end zeros_of_quadratic_l10_10094


namespace similar_inscribed_triangle_exists_l10_10129

variable {α : Type*} [LinearOrderedField α]

-- Representing points and triangles
structure Point (α : Type*) := (x : α) (y : α)
structure Triangle (α : Type*) := (A B C : Point α)

-- Definitions for inscribed triangles and similarity conditions
def isInscribed (inner outer : Triangle α) : Prop :=
  -- Dummy definition, needs correct geometric interpretation
  sorry

def areSimilar (Δ1 Δ2 : Triangle α) : Prop :=
  -- Dummy definition, needs correct geometric interpretation
  sorry

-- Main theorem
theorem similar_inscribed_triangle_exists (Δ₁ Δ₂ : Triangle α) (h_ins : isInscribed Δ₂ Δ₁) :
  ∃ Δ₃ : Triangle α, isInscribed Δ₃ Δ₂ ∧ areSimilar Δ₁ Δ₃ :=
sorry

end similar_inscribed_triangle_exists_l10_10129


namespace cuboid_height_l10_10488

-- Definition of variables
def length := 4  -- in cm
def breadth := 6  -- in cm
def surface_area := 120  -- in cm²

-- The formula for the surface area of a cuboid: S = 2(lb + lh + bh)
def surface_area_formula (l b h : ℝ) : ℝ := 2 * (l * b + l * h + b * h)

-- Given these values, we need to prove that the height h is 3.6 cm
theorem cuboid_height : 
  ∃ h : ℝ, surface_area = surface_area_formula length breadth h ∧ h = 3.6 :=
by
  sorry

end cuboid_height_l10_10488


namespace find_first_term_l10_10716

theorem find_first_term (S_n : ℕ → ℝ) (a d : ℝ) (n : ℕ) (h₁ : ∀ n > 0, S_n n = n * (2 * a + (n - 1) * d) / 2)
  (h₂ : d = 3) (h₃ : ∃ c, ∀ n > 0, S_n (3 * n) / S_n n = c) : a = 3 / 2 :=
by
  sorry

end find_first_term_l10_10716


namespace Rachel_money_left_l10_10291

theorem Rachel_money_left 
  (money_earned : ℕ)
  (lunch_fraction : ℚ)
  (clothes_percentage : ℚ)
  (dvd_cost : ℚ)
  (supplies_percentage : ℚ)
  (money_left : ℚ) :
  money_earned = 200 →
  lunch_fraction = 1 / 4 →
  clothes_percentage = 15 / 100 →
  dvd_cost = 24.50 →
  supplies_percentage = 10.5 / 100 →
  money_left = 74.50 :=
by
  intros h_money h_lunch h_clothes h_dvd h_supplies
  sorry

end Rachel_money_left_l10_10291


namespace ratio_B_to_C_l10_10492

theorem ratio_B_to_C (A_share B_share C_share : ℝ) 
  (total : A_share + B_share + C_share = 510) 
  (A_share_val : A_share = 360) 
  (B_share_val : B_share = 90)
  (C_share_val : C_share = 60)
  (A_cond : A_share = (2 / 3) * B_share) 
  : B_share / C_share = 3 / 2 := 
by 
  sorry

end ratio_B_to_C_l10_10492


namespace maci_pays_total_cost_l10_10217

def cost_blue_pen : ℝ := 0.10
def num_blue_pens : ℕ := 10
def num_red_pens : ℕ := 15
def cost_red_pen : ℝ := 2 * cost_blue_pen

def total_cost : ℝ := num_blue_pens * cost_blue_pen + num_red_pens * cost_red_pen

theorem maci_pays_total_cost : total_cost = 4 := by
  -- Proof goes here
  sorry

end maci_pays_total_cost_l10_10217


namespace intersection_when_a_minus2_range_of_a_if_A_subset_B_l10_10538

namespace ProofProblem

open Set

-- Definitions
def A (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x ≤ a + 3 }
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

-- Theorem (1)
theorem intersection_when_a_minus2 : 
  A (-2) ∩ B = { x : ℝ | -5 ≤ x ∧ x < -1 } :=
by
  sorry

-- Theorem (2)
theorem range_of_a_if_A_subset_B : 
  A a ⊆ B → (a ∈ Iic (-4) ∨ a ∈ Ici 3) :=
by
  sorry

end ProofProblem

end intersection_when_a_minus2_range_of_a_if_A_subset_B_l10_10538


namespace inequality_1_l10_10421

theorem inequality_1 (x : ℝ) : (x - 2) * (1 - 3 * x) > 2 → 1 < x ∧ x < 4 / 3 :=
by sorry

end inequality_1_l10_10421


namespace cousin_cards_probability_l10_10258

variable {Isabella_cards : ℕ}
variable {Evan_cards : ℕ}
variable {total_cards : ℕ}

theorem cousin_cards_probability 
  (h1 : Isabella_cards = 8)
  (h2 : Evan_cards = 2)
  (h3 : total_cards = 10) :
  (8 / 10 * 2 / 9) + (2 / 10 * 8 / 9) = 16 / 45 :=
by
  sorry

end cousin_cards_probability_l10_10258


namespace range_of_a_l10_10321

noncomputable def f (x : ℝ) : ℝ := (1 / (1 + x^2)) - Real.log (abs x)

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) ↔
  (1 / Real.exp 1 ≤ a ∧ a ≤ (2 + Real.log 3) / 3) :=
sorry

end range_of_a_l10_10321


namespace amount_with_r_l10_10951

theorem amount_with_r (p q r T : ℝ) 
  (h1 : p + q + r = 4000)
  (h2 : r = (2/3) * T)
  (h3 : T = p + q) : 
  r = 1600 := by
  sorry

end amount_with_r_l10_10951


namespace division_of_decimals_l10_10120

theorem division_of_decimals : 0.25 / 0.005 = 50 := 
by
  sorry

end division_of_decimals_l10_10120


namespace veranda_area_correct_l10_10023

noncomputable def area_veranda (length_room : ℝ) (width_room : ℝ) (width_veranda : ℝ) (radius_obstacle : ℝ) : ℝ :=
  let total_length := length_room + 2 * width_veranda
  let total_width := width_room + 2 * width_veranda
  let area_total := total_length * total_width
  let area_room := length_room * width_room
  let area_circle := Real.pi * radius_obstacle^2
  area_total - area_room - area_circle

theorem veranda_area_correct :
  area_veranda 18 12 2 3 = 107.726 :=
by sorry

end veranda_area_correct_l10_10023


namespace alyssa_money_after_movies_and_carwash_l10_10494

theorem alyssa_money_after_movies_and_carwash : 
  ∀ (allowance spent earned : ℕ), 
  allowance = 8 → 
  spent = allowance / 2 → 
  earned = 8 → 
  (allowance - spent + earned = 12) := 
by 
  intros allowance spent earned h_allowance h_spent h_earned 
  rw [h_allowance, h_spent, h_earned] 
  simp 
  sorry

end alyssa_money_after_movies_and_carwash_l10_10494


namespace volume_of_red_tetrahedron_in_colored_cube_l10_10591

noncomputable def red_tetrahedron_volume (side_length : ℝ) : ℝ :=
  let cube_volume := side_length ^ 3
  let clear_tetrahedron_volume := (1/3) * (1/2 * side_length * side_length) * side_length
  let red_tetrahedron_volume := (cube_volume - 4 * clear_tetrahedron_volume)
  red_tetrahedron_volume

theorem volume_of_red_tetrahedron_in_colored_cube 
: red_tetrahedron_volume 8 = 512 / 3 := by
  sorry

end volume_of_red_tetrahedron_in_colored_cube_l10_10591


namespace tetrahedron_planes_count_l10_10998

def tetrahedron_planes : ℕ :=
  let vertices := 4
  let midpoints := 6
  -- The total number of planes calculated by considering different combinations
  4      -- planes formed by three vertices
  + 6    -- planes formed by two vertices and one midpoint
  + 12   -- planes formed by one vertex and two midpoints
  + 7    -- planes formed by three midpoints

theorem tetrahedron_planes_count :
  tetrahedron_planes = 29 :=
by
  sorry

end tetrahedron_planes_count_l10_10998


namespace balance_balls_l10_10602

theorem balance_balls (R O G B : ℝ) (h₁ : 4 * R = 8 * G) (h₂ : 3 * O = 6 * G) (h₃ : 8 * G = 6 * B) :
  3 * R + 2 * O + 4 * B = (46 / 3) * G :=
by
  -- Using the given conditions to derive intermediate results (included in the detailed proof, not part of the statement)
  sorry

end balance_balls_l10_10602


namespace total_weight_of_oranges_l10_10157

theorem total_weight_of_oranges :
  let capacity1 := 80
  let capacity2 := 50
  let capacity3 := 60
  let filled1 := 3 / 4
  let filled2 := 3 / 5
  let filled3 := 2 / 3
  let weight_per_orange1 := 0.25
  let weight_per_orange2 := 0.30
  let weight_per_orange3 := 0.40
  let num_oranges1 := capacity1 * filled1
  let num_oranges2 := capacity2 * filled2
  let num_oranges3 := capacity3 * filled3
  let total_weight1 := num_oranges1 * weight_per_orange1
  let total_weight2 := num_oranges2 * weight_per_orange2
  let total_weight3 := num_oranges3 * weight_per_orange3
  total_weight1 + total_weight2 + total_weight3 = 40 := by
  sorry

end total_weight_of_oranges_l10_10157


namespace james_after_paying_debt_l10_10338

variables (L J A : Real)

-- Define the initial conditions
def total_money : Real := 300
def debt : Real := 25
def total_with_debt : Real := total_money + debt

axiom h1 : J = A + 40
axiom h2 : J + A = total_with_debt

-- Prove that James owns $170 after paying off half of Lucas' debt
theorem james_after_paying_debt (h1 : J = A + 40) (h2 : J + A = total_with_debt) :
  (J - (debt / 2)) = 170 :=
  sorry

end james_after_paying_debt_l10_10338


namespace total_players_l10_10512

theorem total_players (K Kho_only Both : Nat) (hK : K = 10) (hKho_only : Kho_only = 30) (hBoth : Both = 5) : 
  (K - Both) + Kho_only + Both = 40 := by
  sorry

end total_players_l10_10512


namespace minimum_sides_of_polygon_l10_10993

theorem minimum_sides_of_polygon (θ : ℝ) (hθ : θ = 25.5) : ∃ n : ℕ, n = 240 ∧ ∀ k : ℕ, (k * θ) % 360 = 0 → k = n := 
by
  -- The proof goes here
  sorry

end minimum_sides_of_polygon_l10_10993


namespace root_implies_value_l10_10178

theorem root_implies_value (b c : ℝ) (h : 2 * b - c = 4) : 4 * b - 2 * c + 1 = 9 :=
by
  sorry

end root_implies_value_l10_10178


namespace solve_for_x_l10_10851

theorem solve_for_x (x : ℝ) (y : ℝ) (z : ℝ) (h1 : y = 1) (h2 : z = 3) (h3 : x^2 * y * z - x * y * z^2 = 6) :
  x = -2 / 3 ∨ x = 3 :=
by sorry

end solve_for_x_l10_10851


namespace combined_height_is_320_cm_l10_10806

-- Define Maria's height in inches
def Maria_height_in_inches : ℝ := 54

-- Define Ben's height in inches
def Ben_height_in_inches : ℝ := 72

-- Define the conversion factor from inches to centimeters
def inch_to_cm : ℝ := 2.54

-- Define the combined height of Maria and Ben in centimeters
def combined_height_in_cm : ℝ := (Maria_height_in_inches + Ben_height_in_inches) * inch_to_cm

-- State and prove that the combined height is 320.0 cm
theorem combined_height_is_320_cm : combined_height_in_cm = 320.0 := by
  sorry

end combined_height_is_320_cm_l10_10806


namespace skateboarder_speed_l10_10440

-- Defining the conditions
def distance_feet : ℝ := 476.67
def time_seconds : ℝ := 25
def feet_per_mile : ℝ := 5280
def seconds_per_hour : ℝ := 3600

-- Defining the expected speed in miles per hour
def expected_speed_mph : ℝ := 13.01

-- The problem statement: Prove that the skateboarder's speed is 13.01 mph given the conditions
theorem skateboarder_speed : (distance_feet / feet_per_mile) / (time_seconds / seconds_per_hour) = expected_speed_mph := by
  sorry

end skateboarder_speed_l10_10440


namespace range_of_2x_minus_y_l10_10730

variable {x y : ℝ}

theorem range_of_2x_minus_y (h1 : 2 < x) (h2 : x < 4) (h3 : -1 < y) (h4 : y < 3) :
  ∃ (a b : ℝ), (1 < a) ∧ (a < 2 * x - y) ∧ (2 * x - y < b) ∧ (b < 9) :=
by
  sorry

end range_of_2x_minus_y_l10_10730


namespace general_formula_arithmetic_sum_of_geometric_terms_l10_10235

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 2 = 2 ∧ a 5 = 8

noncomputable def geometric_sequence (b : ℕ → ℝ) (a : ℕ → ℤ) : Prop :=
  b 1 = 1 ∧ b 2 + b 3 = a 4

noncomputable def sum_of_terms (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, T n = (2:ℝ)^n - 1

theorem general_formula_arithmetic (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  ∀ n, a n = 2 * n - 2 :=
sorry

theorem sum_of_geometric_terms (a : ℕ → ℤ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h : arithmetic_sequence a) (h2 : geometric_sequence b a) :
  sum_of_terms T b :=
sorry

end general_formula_arithmetic_sum_of_geometric_terms_l10_10235


namespace game_points_l10_10677

noncomputable def total_points (total_enemies : ℕ) (red_enemies : ℕ) (blue_enemies : ℕ) 
  (enemies_defeated : ℕ) (points_per_enemy : ℕ) (bonus_points : ℕ) 
  (hits_taken : ℕ) (points_lost_per_hit : ℕ) : ℕ :=
  (enemies_defeated * points_per_enemy + if enemies_defeated > 0 ∧ enemies_defeated < total_enemies then bonus_points else 0) - (hits_taken * points_lost_per_hit)

theorem game_points (h : total_points 6 3 3 4 3 5 2 2 = 13) : Prop := sorry

end game_points_l10_10677


namespace Nick_total_money_l10_10102

variable (nickels : Nat) (dimes : Nat) (quarters : Nat)
variable (value_nickel : Nat := 5) (value_dime : Nat := 10) (value_quarter : Nat := 25)

def total_value (nickels dimes quarters : Nat) : Nat :=
  nickels * value_nickel + dimes * value_dime + quarters * value_quarter

theorem Nick_total_money :
  total_value 6 2 1 = 75 := by
  sorry

end Nick_total_money_l10_10102


namespace find_ab_l10_10077

noncomputable def validate_ab : Prop :=
  let n : ℕ := 8
  let a : ℕ := n^2 - 1
  let b : ℕ := n
  a = 63 ∧ b = 8

theorem find_ab : validate_ab :=
by
  sorry

end find_ab_l10_10077


namespace find_missing_number_l10_10391

theorem find_missing_number (x : ℕ) (h : x * 240 = 173 * 240) : x = 173 :=
sorry

end find_missing_number_l10_10391


namespace carmen_sprigs_left_l10_10446

-- Definitions based on conditions
def initial_sprigs : ℕ := 25
def whole_sprigs_used : ℕ := 8
def half_sprigs_plates : ℕ := 12
def half_sprigs_total_used : ℕ := half_sprigs_plates / 2

-- Total sprigs used
def total_sprigs_used : ℕ := whole_sprigs_used + half_sprigs_total_used

-- Leftover sprigs computation
def sprigs_left : ℕ := initial_sprigs - total_sprigs_used

-- Statement to prove
theorem carmen_sprigs_left : sprigs_left = 11 :=
by
  sorry

end carmen_sprigs_left_l10_10446


namespace exponent_property_l10_10384

theorem exponent_property (a : ℝ) : a^7 = a^3 * a^4 :=
by
  -- The proof statement follows from the properties of exponents:
  -- a^m * a^n = a^(m + n)
  -- Therefore, a^3 * a^4 = a^(3 + 4) = a^7.
  sorry

end exponent_property_l10_10384


namespace roots_range_of_a_l10_10036

theorem roots_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + (a - 2)*|x - 3| + 9 - 2*a = 0) ↔ a > 0 ∨ a = -2 :=
sorry

end roots_range_of_a_l10_10036


namespace obtuse_is_second_quadrant_l10_10419

-- Define the boundaries for an obtuse angle.
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Define the second quadrant condition.
def is_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- The proof problem: Prove that an obtuse angle is a second quadrant angle.
theorem obtuse_is_second_quadrant (θ : ℝ) : is_obtuse θ → is_second_quadrant θ :=
by
  intro h
  sorry

end obtuse_is_second_quadrant_l10_10419


namespace pictures_deleted_l10_10084

theorem pictures_deleted (zoo_pics museum_pics remaining_pics : ℕ) 
  (h1 : zoo_pics = 15) 
  (h2 : museum_pics = 18) 
  (h3 : remaining_pics = 2) : 
  zoo_pics + museum_pics - remaining_pics = 31 :=
by 
  sorry

end pictures_deleted_l10_10084


namespace charley_pencils_lost_l10_10370

theorem charley_pencils_lost :
  ∃ x : ℕ, (30 - x - (1/3 : ℝ) * (30 - x) = 16) ∧ x = 6 :=
by
  -- Since x must be an integer and the equations naturally produce whole numbers,
  -- we work within the context of natural numbers, then cast to real as needed.
  use 6
  -- Express the main condition in terms of x
  have h: (30 - 6 - (1/3 : ℝ) * (30 - 6) = 16) := by sorry
  exact ⟨h, rfl⟩

end charley_pencils_lost_l10_10370


namespace bricks_needed_for_wall_l10_10110

noncomputable def number_of_bricks_needed
    (brick_length : ℕ)
    (brick_width : ℕ)
    (brick_height : ℕ)
    (wall_length_m : ℕ)
    (wall_height_m : ℕ)
    (wall_thickness_cm : ℕ) : ℕ :=
  let wall_length_cm := wall_length_m * 100
  let wall_height_cm := wall_height_m * 100
  let wall_volume := wall_length_cm * wall_height_cm * wall_thickness_cm
  let brick_volume := brick_length * brick_width * brick_height
  (wall_volume + brick_volume - 1) / brick_volume -- This rounds up to the nearest whole number.

theorem bricks_needed_for_wall : number_of_bricks_needed 5 11 6 8 6 2 = 2910 :=
sorry

end bricks_needed_for_wall_l10_10110


namespace eden_bears_count_l10_10975

-- Define the main hypothesis
def initial_bears : Nat := 20
def favorite_bears : Nat := 8
def remaining_bears := initial_bears - favorite_bears

def number_of_sisters : Nat := 3
def bears_per_sister := remaining_bears / number_of_sisters

def eden_initial_bears : Nat := 10
def eden_final_bears := eden_initial_bears + bears_per_sister

theorem eden_bears_count : eden_final_bears = 14 :=
by
  unfold eden_final_bears eden_initial_bears bears_per_sister remaining_bears initial_bears favorite_bears
  norm_num
  sorry

end eden_bears_count_l10_10975


namespace commutative_op_l10_10252

variable {S : Type} (op : S → S → S)

-- Conditions
axiom cond1 : ∀ (a b : S), op a (op a b) = b
axiom cond2 : ∀ (a b : S), op (op a b) b = a

-- Proof problem statement
theorem commutative_op : ∀ (a b : S), op a b = op b a :=
by
  intros a b
  sorry

end commutative_op_l10_10252


namespace present_age_of_father_l10_10516

/-- The present age of the father is 3 years more than 3 times the age of his son, 
    and 3 years hence, the father's age will be 8 years more than twice the age of the son. 
    Prove that the present age of the father is 27 years. -/
theorem present_age_of_father (F S : ℕ) (h1 : F = 3 * S + 3) (h2 : F + 3 = 2 * (S + 3) + 8) : F = 27 :=
by
  sorry

end present_age_of_father_l10_10516


namespace number_of_distinguishable_arrangements_l10_10089

-- Define the conditions
def num_blue_tiles : Nat := 1
def num_red_tiles : Nat := 2
def num_green_tiles : Nat := 3
def num_yellow_tiles : Nat := 2
def total_tiles : Nat := num_blue_tiles + num_red_tiles + num_green_tiles + num_yellow_tiles

-- The goal is to prove the number of distinguishable arrangements
theorem number_of_distinguishable_arrangements : 
  (Nat.factorial total_tiles) / ((Nat.factorial num_green_tiles) * 
                                (Nat.factorial num_red_tiles) * 
                                (Nat.factorial num_yellow_tiles) * 
                                (Nat.factorial num_blue_tiles)) = 1680 := by
  sorry

end number_of_distinguishable_arrangements_l10_10089


namespace smallest_three_digit_perfect_square_l10_10750

theorem smallest_three_digit_perfect_square :
  ∀ (a : ℕ), 100 ≤ a ∧ a < 1000 → (∃ (n : ℕ), 1001 * a + 1 = n^2) → a = 183 :=
by
  sorry

end smallest_three_digit_perfect_square_l10_10750


namespace find_first_number_l10_10008

theorem find_first_number (x : ℤ) (k : ℤ) :
  (29 > 0) ∧ (x % 29 = 8) ∧ (1490 % 29 = 11) → x = 29 * k + 8 :=
by
  intros h
  sorry

end find_first_number_l10_10008


namespace factor_quadratic_expression_l10_10725

theorem factor_quadratic_expression (a b : ℤ) :
  (∃ a b : ℤ, (5 * a + 5 * b = -125) ∧ (a * b = -100) → (a + b = -25)) → (25 * x^2 - 125 * x - 100 = (5 * x + a) * (5 * x + b)) := 
by
  sorry

end factor_quadratic_expression_l10_10725


namespace B_days_to_complete_work_l10_10985

theorem B_days_to_complete_work (A_days : ℕ) (efficiency_less_percent : ℕ) 
  (hA : A_days = 12) (hB_efficiency : efficiency_less_percent = 20) :
  let A_work_rate := 1 / 12
  let B_work_rate := (1 - (20 / 100)) * A_work_rate
  let B_days := 1 / B_work_rate
  B_days = 15 :=
by
  sorry

end B_days_to_complete_work_l10_10985


namespace number_of_balls_l10_10159

noncomputable def totalBalls (frequency : ℚ) (yellowBalls : ℕ) : ℚ :=
  yellowBalls / frequency

theorem number_of_balls (h : totalBalls 0.3 6 = 20) : true :=
by
  sorry

end number_of_balls_l10_10159


namespace system_has_two_distinct_solutions_for_valid_a_l10_10405

noncomputable def log_eq (x y a : ℝ) : Prop := 
  Real.log (a * x + 4 * a) / Real.log (abs (x + 3)) = 
  2 * Real.log (x + y) / Real.log (abs (x + 3))

noncomputable def original_system (x y a : ℝ) : Prop :=
  log_eq x y a ∧ (x + 1 + Real.sqrt (x^2 + 2 * x + y - 4) = 0)

noncomputable def valid_range (a : ℝ) : Prop := 
  (4 < a ∧ a < 4.5) ∨ (4.5 < a ∧ a ≤ 16 / 3)

theorem system_has_two_distinct_solutions_for_valid_a (a : ℝ) :
  valid_range a → 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ original_system x₁ 5 a ∧ original_system x₂ 5 a ∧ (-5 < x₁ ∧ x₁ ≤ -1) ∧ (-5 < x₂ ∧ x₂ ≤ -1) := 
sorry

end system_has_two_distinct_solutions_for_valid_a_l10_10405


namespace true_propositions_for_quadratic_equations_l10_10581

theorem true_propositions_for_quadratic_equations :
  (∀ (a b c : ℤ), a ≠ 0 → (∃ Δ : ℤ, Δ^2 = b^2 - 4 * a * c → ∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0)) ∧
  (∀ (a b c : ℤ), a ≠ 0 → (∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0 → ∃ Δ : ℤ, Δ^2 = b^2 - 4 * a * c)) ∧
  (¬ ∀ (a b c : ℝ), a ≠ 0 → (∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0)) ∧
  (∀ (a b c : ℤ), a ≠ 0 ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 → ¬∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0) :=
by sorry

end true_propositions_for_quadratic_equations_l10_10581


namespace wave_propagation_l10_10316

def accum (s : String) : String :=
  String.join (List.intersperse "-" (s.data.enum.map (λ (i : Nat × Char) =>
    String.mk [i.2.toUpper] ++ String.mk (List.replicate i.1 i.2.toLower))))

theorem wave_propagation (s : String) :
  s = "dremCaheя" → accum s = "D-Rr-Eee-Mmmm-Ccccc-Aaaaaa-Hhhhhhh-Eeeeeeee-Яяяяяяяяя" :=
  by
  intro h
  rw [h]
  sorry

end wave_propagation_l10_10316


namespace number_of_girls_l10_10241

theorem number_of_girls (n : ℕ) (h : 2 * 25 * (n * (n - 1)) = 3 * 25 * 24) : (25 - n) = 16 := by
  sorry

end number_of_girls_l10_10241


namespace average_marks_l10_10810

variable (M P C : ℤ)

-- Conditions
axiom h1 : M + P = 50
axiom h2 : C = P + 20

-- Theorem statement
theorem average_marks : (M + C) / 2 = 35 := by
  sorry

end average_marks_l10_10810


namespace number_of_partners_l10_10090

def total_profit : ℝ := 80000
def majority_owner_share := 0.25 * total_profit
def remaining_profit := total_profit - majority_owner_share
def partner_share := 0.25 * remaining_profit
def combined_share := majority_owner_share + 2 * partner_share

theorem number_of_partners : combined_share = 50000 → remaining_profit / partner_share = 4 := by
  intro h1
  have h_majority : majority_owner_share = 0.25 * total_profit := by sorry
  have h_remaining : remaining_profit = total_profit - majority_owner_share := by sorry
  have h_partner : partner_share = 0.25 * remaining_profit := by sorry
  have h_combined : combined_share = majority_owner_share + 2 * partner_share := by sorry
  calc
    remaining_profit / partner_share = _ := by sorry
    4 = 4 := by sorry

end number_of_partners_l10_10090


namespace cos_135_eq_neg_sqrt2_div_2_sin_135_eq_sqrt2_div_2_l10_10986

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

theorem sin_135_eq_sqrt2_div_2 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by sorry

end cos_135_eq_neg_sqrt2_div_2_sin_135_eq_sqrt2_div_2_l10_10986


namespace number_of_lines_through_focus_intersecting_hyperbola_l10_10775

open Set

noncomputable def hyperbola (x y : ℝ) : Prop := (x^2 / 2) - y^2 = 1

-- The coordinates of the focuses of the hyperbola
def right_focus : ℝ × ℝ := (2, 0)

-- Definition to express that a line passes through the right focus
def line_through_focus (l : ℝ → ℝ) : Prop := l 2 = 0

-- Definition for the length of segment AB being 4
def length_AB_is_4 (A B : ℝ × ℝ) : Prop := dist A B = 4

-- The statement asserting the number of lines satisfying the given condition
theorem number_of_lines_through_focus_intersecting_hyperbola:
  ∃ (n : ℕ), n = 3 ∧ ∀ (l : ℝ → ℝ),
  line_through_focus l →
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ length_AB_is_4 A B :=
sorry

end number_of_lines_through_focus_intersecting_hyperbola_l10_10775


namespace radius_of_larger_circle_l10_10045

theorem radius_of_larger_circle
  (r : ℝ) -- radius of the smaller circle
  (R : ℝ) -- radius of the larger circle
  (ratio : R = 4 * r) -- radii ratio 1:4
  (AC : ℝ) -- diameter of the larger circle
  (BC : ℝ) -- chord of the larger circle
  (AB : ℝ := 16) -- given condition AB = 16
  (diameter_AC : AC = 2 * R) -- AC is diameter of the larger circle
  (tangent : BC^2 = AB^2 + (2 * R)^2) -- Pythagorean theorem for the right triangle ABC
  :
  R = 32 := 
sorry

end radius_of_larger_circle_l10_10045


namespace product_of_place_values_l10_10726

theorem product_of_place_values : 
  let place_value_1 := 800000
  let place_value_2 := 80
  let place_value_3 := 0.08
  place_value_1 * place_value_2 * place_value_3 = 5120000 := 
by 
  -- proof will be provided here 
  sorry

end product_of_place_values_l10_10726


namespace population_doubles_l10_10562

theorem population_doubles (initial_population: ℕ) (initial_year: ℕ) (doubling_period: ℕ) (target_population : ℕ) (target_year : ℕ) : 
  initial_population = 500 → 
  initial_year = 2023 → 
  doubling_period = 20 → 
  target_population = 8000 → 
  target_year = 2103 :=
by 
  sorry

end population_doubles_l10_10562


namespace tangent_line_eq_l10_10797

def perp_eq (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 - 1

theorem tangent_line_eq (x y : ℝ) (h1 : perp_eq x y) (h2 : y = curve x) : 
  ∃ (m : ℝ), y = -3 * x + m ∧ y = -3 * x - 2 := 
sorry

end tangent_line_eq_l10_10797


namespace two_digit_number_l10_10780

theorem two_digit_number (x y : ℕ) (h1 : x + y = 7) (h2 : (x + 2) + 10 * (y + 2) = 2 * (x + 10 * y) - 3) : (10 * y + x) = 25 :=
by
  sorry

end two_digit_number_l10_10780


namespace initial_pencils_l10_10376

theorem initial_pencils (P : ℕ) (h1 : 84 = P - (P - 15) / 4 + 16 - 12 + 23) : P = 71 :=
by
  sorry

end initial_pencils_l10_10376


namespace find_uv_non_integer_l10_10004

def p (b : Fin 14 → ℚ) (x y : ℚ) : ℚ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3 + 
  b 10 * x^4 + b 11 * y^4 + b 12 * x^3 * y^2 + b 13 * y^3 * x^2

variables (b : Fin 14 → ℚ)
variables (u v : ℚ)

def zeros_at_specific_points :=
  p b 0 0 = 0 ∧ p b 1 0 = 0 ∧ p b (-1) 0 = 0 ∧
  p b 0 1 = 0 ∧ p b 0 (-1) = 0 ∧ p b 1 1 = 0 ∧
  p b (-1) (-1) = 0 ∧ p b 2 2 = 0 ∧ 
  p b 2 (-2) = 0 ∧ p b (-2) 2 = 0

theorem find_uv_non_integer
  (h : zeros_at_specific_points b) :
  p b (5/19) (16/19) = 0 :=
sorry

end find_uv_non_integer_l10_10004


namespace exist_monochromatic_equilateral_triangle_l10_10328

theorem exist_monochromatic_equilateral_triangle 
  (color : ℝ × ℝ → ℕ) 
  (h_color : ∀ p : ℝ × ℝ, color p = 0 ∨ color p = 1) : 
  ∃ (A B C : ℝ × ℝ), (dist A B = dist B C) ∧ (dist B C = dist C A) ∧ (color A = color B ∧ color B = color C) :=
sorry

end exist_monochromatic_equilateral_triangle_l10_10328


namespace nina_total_amount_l10_10179

theorem nina_total_amount:
  ∃ (x y z w : ℕ), 
  x + y + z + w = 27 ∧
  y = 2 * z ∧
  z = 2 * x ∧
  7 < w ∧ w < 20 ∧
  10 * x + 5 * y + 2 * z + 3 * w = 107 :=
by 
  sorry

end nina_total_amount_l10_10179


namespace average_temperature_l10_10536

def temperatures : List ℝ := [-36, 13, -15, -10]

theorem average_temperature : (List.sum temperatures) / (temperatures.length) = -12 := by
  sorry

end average_temperature_l10_10536


namespace jeff_bought_6_pairs_l10_10856

theorem jeff_bought_6_pairs (price_of_shoes : ℝ) (num_of_shoes : ℕ) (price_of_jersey : ℝ)
  (h1 : price_of_jersey = (1 / 4) * price_of_shoes)
  (h2 : num_of_shoes * price_of_shoes = 480)
  (h3 : num_of_shoes * price_of_shoes + 4 * price_of_jersey = 560) :
  num_of_shoes = 6 :=
sorry

end jeff_bought_6_pairs_l10_10856


namespace jason_needs_201_grams_l10_10820

-- Define the conditions
def rectangular_patch_length : ℕ := 6
def rectangular_patch_width : ℕ := 7
def square_path_side_length : ℕ := 5
def sand_per_square_inch : ℕ := 3

-- Define the areas
def rectangular_patch_area : ℕ := rectangular_patch_length * rectangular_patch_width
def square_path_area : ℕ := square_path_side_length * square_path_side_length

-- Define the total area
def total_area : ℕ := rectangular_patch_area + square_path_area

-- Define the total sand needed
def total_sand_needed : ℕ := total_area * sand_per_square_inch

-- State the proof problem
theorem jason_needs_201_grams : total_sand_needed = 201 := by
    sorry

end jason_needs_201_grams_l10_10820


namespace inscribed_sphere_surface_area_l10_10180

theorem inscribed_sphere_surface_area (V S : ℝ) (hV : V = 2) (hS : S = 3) : 4 * Real.pi * (3 * V / S)^2 = 16 * Real.pi := by
  sorry

end inscribed_sphere_surface_area_l10_10180


namespace number_solution_l10_10546

theorem number_solution : ∃ x : ℝ, x + 9 = x^2 ∧ x = (1 + Real.sqrt 37) / 2 :=
by
  use (1 + Real.sqrt 37) / 2
  simp
  sorry

end number_solution_l10_10546


namespace quadratic_has_real_root_l10_10153

theorem quadratic_has_real_root (a : ℝ) : 
  ¬(∀ x : ℝ, x^2 + a * x + a - 1 ≠ 0) :=
sorry

end quadratic_has_real_root_l10_10153


namespace weight_of_each_bag_of_planks_is_14_l10_10707

-- Definitions
def crate_capacity : Nat := 20
def num_crates : Nat := 15
def num_bags_nails : Nat := 4
def weight_bag_nails : Nat := 5
def num_bags_hammers : Nat := 12
def weight_bag_hammers : Nat := 5
def num_bags_planks : Nat := 10
def weight_to_leave_out : Nat := 80

-- Total weight calculations
def weight_nails := num_bags_nails * weight_bag_nails
def weight_hammers := num_bags_hammers * weight_bag_hammers
def total_weight_nails_hammers := weight_nails + weight_hammers
def total_crate_capacity := num_crates * crate_capacity
def weight_that_can_be_loaded := total_crate_capacity - weight_to_leave_out
def weight_available_for_planks := weight_that_can_be_loaded - total_weight_nails_hammers
def weight_each_bag_planks := weight_available_for_planks / num_bags_planks

-- Theorem statement
theorem weight_of_each_bag_of_planks_is_14 : weight_each_bag_planks = 14 :=
by {
  sorry
}

end weight_of_each_bag_of_planks_is_14_l10_10707


namespace second_multiple_of_three_l10_10279

theorem second_multiple_of_three (n : ℕ) (h : 3 * (n - 1) + 3 * (n + 1) = 150) : 3 * n = 75 :=
sorry

end second_multiple_of_three_l10_10279


namespace Claire_photos_is_5_l10_10583

variable (Claire_photos : ℕ)
variable (Lisa_photos : ℕ := 3 * Claire_photos)
variable (Robert_photos : ℕ := Claire_photos + 10)

theorem Claire_photos_is_5
  (h1 : Lisa_photos = Robert_photos) :
  Claire_photos = 5 :=
by
  sorry

end Claire_photos_is_5_l10_10583


namespace intersection_M_N_l10_10567

noncomputable def M : Set ℝ := { x | x^2 = x }
noncomputable def N : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_M_N :
  M ∩ N = {1} := by
  sorry

end intersection_M_N_l10_10567


namespace rewrite_neg_multiplication_as_exponent_l10_10690

theorem rewrite_neg_multiplication_as_exponent :
  -2 * 2 * 2 * 2 = - (2^4) :=
by
  sorry

end rewrite_neg_multiplication_as_exponent_l10_10690


namespace sentence_structure_diff_l10_10600

-- Definitions based on sentence structures.
def sentence_A := "得不焚，殆有神护者" -- passive
def sentence_B := "重为乡党所笑" -- passive
def sentence_C := "而文采不表于后也" -- post-positioned prepositional
def sentence_D := "是以见放" -- passive

-- Definition to check if the given sentence is passive
def is_passive (s : String) : Prop :=
  s = sentence_A ∨ s = sentence_B ∨ s = sentence_D

-- Definition to check if the given sentence is post-positioned prepositional
def is_post_positioned_prepositional (s : String) : Prop :=
  s = sentence_C

-- Theorem to prove
theorem sentence_structure_diff :
  (is_post_positioned_prepositional sentence_C) ∧ ¬(is_passive sentence_C) :=
by
  sorry

end sentence_structure_diff_l10_10600


namespace faye_complete_bouquets_l10_10346

theorem faye_complete_bouquets :
  let roses_initial := 48
  let lilies_initial := 40
  let tulips_initial := 76
  let sunflowers_initial := 34
  let roses_wilted := 24
  let lilies_wilted := 10
  let tulips_wilted := 14
  let sunflowers_wilted := 7
  let roses_remaining := roses_initial - roses_wilted
  let lilies_remaining := lilies_initial - lilies_wilted
  let tulips_remaining := tulips_initial - tulips_wilted
  let sunflowers_remaining := sunflowers_initial - sunflowers_wilted
  let bouquets_roses := roses_remaining / 2
  let bouquets_lilies := lilies_remaining
  let bouquets_tulips := tulips_remaining / 3
  let bouquets_sunflowers := sunflowers_remaining
  let bouquets := min (min bouquets_roses bouquets_lilies) (min bouquets_tulips bouquets_sunflowers)
  bouquets = 12 :=
by
  sorry

end faye_complete_bouquets_l10_10346


namespace min_polyline_distance_l10_10533

-- Define the polyline distance between two points P(x1, y1) and Q(x2, y2).
noncomputable def polyline_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

-- Define the circle x^2 + y^2 = 1.
def on_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + P.2 ^ 2 = 1

-- Define the line 2x + y = 2√5.
def on_line (P : ℝ × ℝ) : Prop :=
  2 * P.1 + P.2 = 2 * Real.sqrt 5

-- Statement of the minimum distance problem.
theorem min_polyline_distance : 
  ∀ P Q : ℝ × ℝ, on_circle P → on_line Q → 
  polyline_distance P Q ≥ Real.sqrt 5 / 2 :=
sorry

end min_polyline_distance_l10_10533


namespace min_x_plus_y_l10_10872

theorem min_x_plus_y (x y : ℕ) (hxy : x ≠ y) (h : (1/x : ℝ) + 1/y = 1/24) : x + y = 98 :=
sorry

end min_x_plus_y_l10_10872


namespace digit_possibilities_for_mod4_count_possibilities_is_3_l10_10831

theorem digit_possibilities_for_mod4 (N : ℕ) (h : N < 10): 
  (80 + N) % 4 = 0 → N = 0 ∨ N = 4 ∨ N = 8 → true := 
by
  -- proof is not needed
  sorry

def count_possibilities : ℕ := 
  (if (80 + 0) % 4 = 0 then 1 else 0) +
  (if (80 + 1) % 4 = 0 then 1 else 0) +
  (if (80 + 2) % 4 = 0 then 1 else 0) +
  (if (80 + 3) % 4 = 0 then 1 else 0) +
  (if (80 + 4) % 4 = 0 then 1 else 0) +
  (if (80 + 5) % 4 = 0 then 1 else 0) +
  (if (80 + 6) % 4 = 0 then 1 else 0) +
  (if (80 + 7) % 4 = 0 then 1 else 0) +
  (if (80 + 8) % 4 = 0 then 1 else 0) +
  (if (80 + 9) % 4 = 0 then 1 else 0)

theorem count_possibilities_is_3: count_possibilities = 3 := 
by
  -- proof is not needed
  sorry

end digit_possibilities_for_mod4_count_possibilities_is_3_l10_10831


namespace jihyae_initial_money_l10_10875

variables {M : ℕ}

def spent_on_supplies (M : ℕ) := M / 2 + 200
def left_after_buying (M : ℕ) := M - spent_on_supplies M
def saved (M : ℕ) := left_after_buying M / 2 + 300
def final_leftover (M : ℕ) := left_after_buying M - saved M

theorem jihyae_initial_money : final_leftover M = 350 → M = 3000 :=
by
  sorry

end jihyae_initial_money_l10_10875


namespace solve_equation_1_solve_equation_2_l10_10240

theorem solve_equation_1 :
  ∀ x : ℝ, 2 * x^2 - 4 * x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  intro x
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, x^2 - 6 * x - 6 = 0 ↔ (x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15) :=
by
  intro x
  sorry

end solve_equation_1_solve_equation_2_l10_10240


namespace simplify_expression_l10_10166

theorem simplify_expression :
  2 + 1 / (2 + 1 / (2 + 1 / 2)) = 29 / 12 :=
by
  sorry  -- Proof will be provided here

end simplify_expression_l10_10166


namespace daps_equivalent_to_dips_l10_10232

-- Definitions from conditions
def daps (n : ℕ) : ℕ := n
def dops (n : ℕ) : ℕ := n
def dips (n : ℕ) : ℕ := n

-- Given conditions
def equivalence_daps_dops : daps 8 = dops 6 := sorry
def equivalence_dops_dips : dops 3 = dips 11 := sorry

-- Proof problem
theorem daps_equivalent_to_dips (n : ℕ) (h1 : daps 8 = dops 6) (h2 : dops 3 = dips 11) : daps 24 = dips 66 :=
sorry

end daps_equivalent_to_dips_l10_10232


namespace find_number_l10_10028

theorem find_number (x : ℝ) (h : 20 * (x / 5) = 40) : x = 10 :=
by
  sorry

end find_number_l10_10028


namespace find_m_value_l10_10429

-- Define the points P and Q and the condition of perpendicularity
def points_PQ (m : ℝ) : Prop := 
  let P := (-2, m)
  let Q := (m, 4)
  let slope_PQ := (m - 4) / (-2 - m)
  slope_PQ * (-1) = -1

-- Problem statement: Find the value of m such that the above condition holds
theorem find_m_value : ∃ (m : ℝ), points_PQ m ∧ m = 1 :=
by sorry

end find_m_value_l10_10429


namespace age_difference_l10_10380

theorem age_difference (C D m : ℕ) 
  (h1 : C = D + m)
  (h2 : C - 1 = 3 * (D - 1)) 
  (h3 : C * D = 72) : 
  m = 9 :=
sorry

end age_difference_l10_10380


namespace exists_natural_number_n_l10_10106

theorem exists_natural_number_n (t : ℕ) (ht : t > 0) :
  ∃ n : ℕ, n > 1 ∧ Nat.gcd n t = 1 ∧ ∀ k : ℕ, k > 0 → ∃ m : ℕ, m > 1 → n^k + t ≠ m^m :=
by
  sorry

end exists_natural_number_n_l10_10106


namespace exponent_of_four_l10_10287

theorem exponent_of_four (n : ℕ) (k : ℕ) (h : n = 21) 
  (eq : (↑(4 : ℕ) * 2 ^ (2 * n) = 4 ^ k)) : k = 22 :=
by
  sorry

end exponent_of_four_l10_10287


namespace expand_expression_l10_10525

theorem expand_expression (x : ℝ) : (x - 1) * (4 * x + 5) = 4 * x^2 + x - 5 := 
by
  -- Proof omitted
  sorry

end expand_expression_l10_10525


namespace volleyball_tournament_l10_10893

theorem volleyball_tournament (n m : ℕ) (h : n = m) :
  n = m := 
by
  sorry

end volleyball_tournament_l10_10893


namespace find_nabla_l10_10543

theorem find_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by {
  sorry
}

end find_nabla_l10_10543


namespace trajectory_equation_find_m_value_l10_10834

def point (α : Type) := (α × α)
def fixed_points (α : Type) := point α

noncomputable def slopes (x y : ℝ) : ℝ := y / x

theorem trajectory_equation (x y : ℝ) (P : point ℝ) (A B : fixed_points ℝ)
  (k1 k2 : ℝ) (hk : k1 * k2 = -1/4) :
  A = (-2, 0) → B = (2, 0) →
  P = (x, y) → 
  slopes (x + 2) y * slopes (x - 2) y = -1/4 →
  (x^2 / 4) + y^2 = 1 :=
sorry

theorem find_m_value (m x₁ x₂ y₁ y₂ : ℝ) (k : ℝ) (hx : (4 * k^2) + 1 - m^2 > 0)
  (hroots_sum : x₁ + x₂ = -((8 * k * m) / ((4 * k^2) + 1)))
  (hroots_prod : x₁ * x₂ = (4 * m^2 - 4) / ((4 * k^2) + 1))
  (hperp : x₁ * x₂ + y₁ * y₂ = 0) :
  y₁ = k * x₁ + m → y₂ = k * x₂ + m →
  m^2 = 4/5 * (k^2 + 1) →
  m = 2 ∨ m = -2 :=
sorry

end trajectory_equation_find_m_value_l10_10834


namespace value_range_of_f_l10_10626

-- Define the function f(x) = 2x - x^2
def f (x : ℝ) : ℝ := 2 * x - x^2

-- State the theorem with the given conditions and prove the correct answer
theorem value_range_of_f :
  (∀ y : ℝ, -3 ≤ y ∧ y ≤ 1 → ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = y) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → -3 ≤ f x ∧ f x ≤ 1) :=
by
  sorry

end value_range_of_f_l10_10626


namespace hyperbola_eccentricity_range_l10_10862

theorem hyperbola_eccentricity_range (a b e : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_upper : b / a < 2) :
  e = Real.sqrt (1 + (b / a) ^ 2) → 1 < e ∧ e < Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_range_l10_10862


namespace correct_option_D_l10_10942

theorem correct_option_D : 
  (-3)^2 = 9 ∧ 
  - (x + y) = -x - y ∧ 
  ¬ (3 * a + 5 * b = 8 * a * b) ∧ 
  5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 :=
by { sorry }

end correct_option_D_l10_10942


namespace problem_a_b_c_l10_10898

theorem problem_a_b_c (a b c : ℝ) (h1 : a < b) (h2 : b < c) (h3 : ab + bc + ac = 0) (h4 : abc = 1) : |a + b| > |c| := 
by sorry

end problem_a_b_c_l10_10898


namespace regions_bounded_by_blue_lines_l10_10208

theorem regions_bounded_by_blue_lines (n : ℕ) : 
  (2 * n^2 + 3 * n + 2) -(n - 1) * (2 * n + 1) ≥ 4 * n + 2 :=
by
  sorry

end regions_bounded_by_blue_lines_l10_10208


namespace whole_process_time_is_9_l10_10728

variable (BleachingTime : ℕ)
variable (DyeingTime : ℕ)

-- Conditions
axiom bleachingTime_is_3 : BleachingTime = 3
axiom dyeingTime_is_twice_bleachingTime : DyeingTime = 2 * BleachingTime

-- Question and Proof Problem
theorem whole_process_time_is_9 (BleachingTime : ℕ) (DyeingTime : ℕ)
  (h1 : BleachingTime = 3) (h2 : DyeingTime = 2 * BleachingTime) : 
  (BleachingTime + DyeingTime) = 9 :=
  by
  sorry

end whole_process_time_is_9_l10_10728


namespace equilateral_triangle_side_length_l10_10436
noncomputable def equilateral_triangle_side (r R : ℝ) (h : R > r) : ℝ :=
  r * R * Real.sqrt 3 / (Real.sqrt (r ^ 2 - r * R + R ^ 2))

theorem equilateral_triangle_side_length
  (r R : ℝ) (hRgr : R > r) :
  ∃ a, a = equilateral_triangle_side r R hRgr :=
sorry

end equilateral_triangle_side_length_l10_10436


namespace find_a_l10_10718

theorem find_a (a : ℤ) (h_range : 0 ≤ a ∧ a < 13) (h_div : (51 ^ 2022 + a) % 13 = 0) : a = 12 := 
by
  sorry

end find_a_l10_10718


namespace product_factors_eq_l10_10286

theorem product_factors_eq :
  (1 - 1/2) * (1 - 1/3) * (1 - 1/4) * (1 - 1/5) * (1 - 1/6) * (1 - 1/7) * (1 - 1/8) * (1 - 1/9) * (1 - 1/10) * (1 - 1/11) = 1 / 11 := 
by
  sorry

end product_factors_eq_l10_10286


namespace u_1000_eq_2036_l10_10879

open Nat

def sequence_term (n : ℕ) : ℕ :=
  let sum_to (k : ℕ) := k * (k + 1) / 2
  if n ≤ 0 then 0 else
  let group := (Nat.sqrt (2 * n)) + 1
  let k := n - sum_to (group - 1)
  (group * group) + 4 * (k - 1) - (group % 4)

theorem u_1000_eq_2036 : sequence_term 1000 = 2036 := sorry

end u_1000_eq_2036_l10_10879


namespace jake_third_test_score_l10_10884

theorem jake_third_test_score
  (avg_score_eq_75 : (80 + 90 + third_score + third_score) / 4 = 75)
  (second_score : ℕ := 80 + 10) :
  third_score = 65 :=
by
  sorry

end jake_third_test_score_l10_10884


namespace function_is_linear_l10_10280

theorem function_is_linear (f : ℝ → ℝ) :
  (∀ a b c d : ℝ,
    a ≠ b → b ≠ c → c ≠ d → d ≠ a →
    (a ≠ c ∧ b ≠ d ∧ a ≠ d ∧ b ≠ c) →
    (a - b) / (b - c) + (a - d) / (d - c) = 0 →
    (f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f d ∧ f a ≠ f c ∧ f a ≠ f d ∧ f b ≠ f d) →
    (f a - f b) / (f b - f c) + (f a - f d) / (f d - f c) = 0) →
  ∃ m c : ℝ, ∀ x : ℝ, f x = m * x + c :=
by
  sorry

end function_is_linear_l10_10280


namespace palindromes_between_300_800_l10_10024

def palindrome_count (l u : ℕ) : ℕ :=
  (u / 100 - l / 100 + 1) * 10

theorem palindromes_between_300_800 : palindrome_count 300 800 = 50 :=
by
  sorry

end palindromes_between_300_800_l10_10024


namespace lincoln_one_way_fare_l10_10348

-- Define the given conditions as assumptions
variables (x : ℝ) (days : ℝ) (total_cost : ℝ) (trips_per_day : ℝ)

-- State the conditions
axiom condition1 : days = 9
axiom condition2 : total_cost = 288
axiom condition3 : trips_per_day = 2

-- The theorem we want to prove based on the conditions
theorem lincoln_one_way_fare (h1 : total_cost = days * trips_per_day * x) : x = 16 :=
by
  -- We skip the proof for the sake of this exercise
  sorry

end lincoln_one_way_fare_l10_10348


namespace salad_cost_is_correct_l10_10407

-- Definitions of costs according to the given conditions
def muffin_cost : ℝ := 2
def coffee_cost : ℝ := 4
def soup_cost : ℝ := 3
def lemonade_cost : ℝ := 0.75

def breakfast_cost : ℝ := muffin_cost + coffee_cost
def lunch_cost : ℝ := breakfast_cost + 3

def salad_cost : ℝ := lunch_cost - (soup_cost + lemonade_cost)

-- Statement to prove
theorem salad_cost_is_correct : salad_cost = 5.25 :=
by
  sorry

end salad_cost_is_correct_l10_10407


namespace chord_length_l10_10823

/-- Given two concentric circles with radii R and r, where the area of the annulus between them is 16π,
    a chord of the larger circle that is tangent to the smaller circle has a length of 8. -/
theorem chord_length {R r c : ℝ} 
  (h1 : R^2 - r^2 = 16)
  (h2 : (c / 2)^2 + r^2 = R^2) :
  c = 8 :=
by
  sorry

end chord_length_l10_10823


namespace taylor_one_basket_probability_l10_10547

-- Definitions based on conditions
def not_make_basket_prob : ℚ := 1 / 3
def make_basket_prob : ℚ := 1 - not_make_basket_prob
def trials : ℕ := 3
def successes : ℕ := 1

def binomial_coefficient (n k : ℕ) : ℕ := n.choose k

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem taylor_one_basket_probability : 
  binomial_probability trials successes make_basket_prob = 2 / 9 :=
by
  rw [binomial_probability, binomial_coefficient]
  -- The rest of the proof steps can involve simplifications 
  -- and calculations that were mentioned in the solution.
  sorry

end taylor_one_basket_probability_l10_10547


namespace infinite_series_sum_l10_10339

theorem infinite_series_sum :
  (∑' n : ℕ, (3:ℝ)^n / (1 + (3:ℝ)^n + (3:ℝ)^(n+1) + (3:ℝ)^(2*n+2))) = 1 / 4 :=
by
  sorry

end infinite_series_sum_l10_10339


namespace probability_exactly_one_correct_l10_10501

def P_A := 0.7
def P_B := 0.8

def P_A_correct_B_incorrect := P_A * (1 - P_B)
def P_A_incorrect_B_correct := (1 - P_A) * P_B

theorem probability_exactly_one_correct :
  P_A_correct_B_incorrect + P_A_incorrect_B_correct = 0.38 :=
by
  sorry

end probability_exactly_one_correct_l10_10501


namespace vikki_hourly_pay_rate_l10_10996

-- Define the variables and conditions
def hours_worked : ℝ := 42
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def union_dues : ℝ := 5
def net_pay : ℝ := 310

-- Define Vikki's hourly pay rate (we will solve for this)
variable (hourly_pay : ℝ)

-- Define the gross earnings
def gross_earnings (hourly_pay : ℝ) : ℝ := hours_worked * hourly_pay

-- Define the total deductions
def total_deductions (hourly_pay : ℝ) : ℝ := (tax_rate * gross_earnings hourly_pay) + (insurance_rate * gross_earnings hourly_pay) + union_dues

-- Define the net pay
def calculate_net_pay (hourly_pay : ℝ) : ℝ := gross_earnings hourly_pay - total_deductions hourly_pay

-- Prove the solution
theorem vikki_hourly_pay_rate : calculate_net_pay hourly_pay = net_pay → hourly_pay = 10 := by
  sorry

end vikki_hourly_pay_rate_l10_10996


namespace susan_spent_total_l10_10617

-- Definitions for the costs and quantities
def pencil_cost : ℝ := 0.25
def pen_cost : ℝ := 0.80
def total_items : ℕ := 36
def pencils_bought : ℕ := 16

-- Question: How much did Susan spend?
theorem susan_spent_total : (pencil_cost * pencils_bought + pen_cost * (total_items - pencils_bought)) = 20 :=
by
    -- definition goes here
    sorry

end susan_spent_total_l10_10617


namespace smallest_five_digit_multiple_of_9_starting_with_7_l10_10585

theorem smallest_five_digit_multiple_of_9_starting_with_7 :
  ∃ (n : ℕ), (70000 ≤ n ∧ n < 80000) ∧ (n % 9 = 0) ∧ n = 70002 :=
sorry

end smallest_five_digit_multiple_of_9_starting_with_7_l10_10585


namespace roots_square_sum_l10_10000

theorem roots_square_sum (r s t p q : ℝ) 
  (h1 : r + s + t = p) 
  (h2 : r * s + r * t + s * t = q) : 
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
by 
  -- proof skipped
  sorry

end roots_square_sum_l10_10000


namespace total_students_in_class_l10_10378

def period_length : ℕ := 40
def periods_per_student : ℕ := 4
def time_per_student : ℕ := 5

theorem total_students_in_class :
  ((period_length / time_per_student) * periods_per_student) = 32 :=
by
  sorry

end total_students_in_class_l10_10378


namespace depth_of_melted_ice_cream_l10_10795

theorem depth_of_melted_ice_cream (r_sphere r_cylinder : ℝ) (Vs : ℝ) (Vc : ℝ) :
  r_sphere = 3 →
  r_cylinder = 12 →
  Vs = (4 / 3) * Real.pi * r_sphere^3 →
  Vc = Real.pi * r_cylinder^2 * (1 / 4) →
  Vs = Vc →
  (1 / 4) = 1 / 4 := 
by
  intros hr_sphere hr_cylinder hVs hVc hVs_eq_Vc
  sorry

end depth_of_melted_ice_cream_l10_10795


namespace cakes_sold_correct_l10_10627

def total_cakes_baked_today : Nat := 5
def total_cakes_baked_yesterday : Nat := 3
def cakes_left : Nat := 2

def total_cakes : Nat := total_cakes_baked_today + total_cakes_baked_yesterday
def cakes_sold : Nat := total_cakes - cakes_left

theorem cakes_sold_correct :
  cakes_sold = 6 :=
by
  -- proof goes here
  sorry

end cakes_sold_correct_l10_10627


namespace new_average_daily_production_l10_10438

theorem new_average_daily_production (n : ℕ) (avg_past_n_days : ℕ) (today_production : ℕ) (h1 : avg_past_n_days = 50) (h2 : today_production = 90) (h3 : n = 9) : 
  (avg_past_n_days * n + today_production) / (n + 1) = 54 := 
by
  sorry

end new_average_daily_production_l10_10438


namespace at_most_two_greater_than_one_l10_10131

theorem at_most_two_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  ¬ (2 * a - 1 / b > 1 ∧ 2 * b - 1 / c > 1 ∧ 2 * c - 1 / a > 1) :=
by
  sorry

end at_most_two_greater_than_one_l10_10131


namespace triangle_area_l10_10300

theorem triangle_area (h : ℝ) (hypotenuse : h = 12) (angle : ∃θ : ℝ, θ = 30 ∧ θ = 30) :
  ∃ (A : ℝ), A = 18 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l10_10300


namespace hyperbola_t_square_l10_10969

theorem hyperbola_t_square (t : ℝ)
  (h1 : ∃ a : ℝ, ∀ (x y : ℝ), (y^2 / 4) - (5 * x^2 / 64) = 1 ↔ ((x, y) = (2, t) ∨ (x, y) = (4, -3) ∨ (x, y) = (0, -2))) :
  t^2 = 21 / 4 :=
by
  -- We need to prove t² = 21/4 given the conditions
  sorry

end hyperbola_t_square_l10_10969


namespace find_number_l10_10792

theorem find_number (n : ℤ) 
  (h : (69842 * 69842 - n * n) / (69842 - n) = 100000) : 
  n = 30158 :=
sorry

end find_number_l10_10792


namespace chang_total_apples_l10_10482

def sweet_apple_price : ℝ := 0.5
def sour_apple_price : ℝ := 0.1
def sweet_apple_percentage : ℝ := 0.75
def sour_apple_percentage : ℝ := 1 - sweet_apple_percentage
def total_earnings : ℝ := 40

theorem chang_total_apples : 
  (total_earnings / (sweet_apple_percentage * sweet_apple_price + sour_apple_percentage * sour_apple_price)) = 100 :=
by
  sorry

end chang_total_apples_l10_10482


namespace machine_production_time_difference_undetermined_l10_10394

theorem machine_production_time_difference_undetermined :
  ∀ (machineP_machineQ_440_hours_diff : ℝ)
    (machineQ_production_rate : ℝ)
    (machineA_production_rate : ℝ),
    machineA_production_rate = 4.000000000000005 →
    machineQ_production_rate = machineA_production_rate * 1.1 →
    machineP_machineQ_440_hours_diff > 0 →
    machineQ_production_rate * machineP_machineQ_440_hours_diff = 440 →
    ∃ machineP_production_rate, 
    ¬(∃ hours_diff : ℝ, hours_diff = 440 / machineP_production_rate - 440 / machineQ_production_rate) := sorry

end machine_production_time_difference_undetermined_l10_10394


namespace security_deposit_amount_correct_l10_10127

noncomputable def daily_rate : ℝ := 125.00
noncomputable def pet_fee : ℝ := 100.00
noncomputable def service_cleaning_fee_rate : ℝ := 0.20
noncomputable def security_deposit_rate : ℝ := 0.50
noncomputable def weeks : ℝ := 2
noncomputable def days_per_week : ℝ := 7

noncomputable def number_of_days : ℝ := weeks * days_per_week
noncomputable def total_rental_fee : ℝ := number_of_days * daily_rate
noncomputable def total_rental_fee_with_pet : ℝ := total_rental_fee + pet_fee
noncomputable def service_cleaning_fee : ℝ := service_cleaning_fee_rate * total_rental_fee_with_pet
noncomputable def total_cost : ℝ := total_rental_fee_with_pet + service_cleaning_fee

theorem security_deposit_amount_correct : 
    security_deposit_rate * total_cost = 1110.00 := 
by 
  sorry

end security_deposit_amount_correct_l10_10127


namespace trig_identity_l10_10630

theorem trig_identity : Real.sin (35 * Real.pi / 6) + Real.cos (-11 * Real.pi / 3) = 0 := by
  sorry

end trig_identity_l10_10630


namespace mask_production_rates_l10_10433

theorem mask_production_rates (x : ℝ) (y : ℝ) :
  (280 / x) - (280 / (1.4 * x)) = 2 →
  x = 40 ∧ y = 1.4 * x →
  y = 56 :=
by {
  sorry
}

end mask_production_rates_l10_10433


namespace plane_equation_and_gcd_l10_10642

variable (x y z : ℝ)

theorem plane_equation_and_gcd (A B C D : ℤ) (h1 : A = 8) (h2 : B = -6) (h3 : C = 5) (h4 : D = -125) :
    (A * x + B * y + C * z + D = 0 ↔ x = 8 ∧ y = -6 ∧ z = 5) ∧
    Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 :=
by sorry

end plane_equation_and_gcd_l10_10642


namespace cube_root_3375_l10_10944

theorem cube_root_3375 (c d : ℕ) (h1 : c > 0 ∧ d > 0) (h2 : c * d^3 = 3375) (h3 : ∀ k : ℕ, k > 0 → c * (d / k)^3 ≠ 3375) : 
  c + d = 16 :=
sorry

end cube_root_3375_l10_10944


namespace geometric_progression_solution_l10_10063

-- Definitions and conditions as per the problem
def geometric_progression_first_term (b q : ℝ) : Prop :=
  b * (1 + q + q^2) = 21

def geometric_progression_sum_of_squares (b q : ℝ) : Prop :=
  b^2 * (1 + q^2 + q^4) = 189

-- The main theorem to be proven
theorem geometric_progression_solution (b q : ℝ) :
  (geometric_progression_first_term b q ∧ geometric_progression_sum_of_squares b q) →
  (b = 3 ∧ q = 2) ∨ (b = 12 ∧ q = 1 / 2) := 
by
  intros h
  sorry

end geometric_progression_solution_l10_10063


namespace no_real_solution_l10_10997

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

-- Lean statement: prove that the equation x^2 - 4x + 6 = 0 has no real solution
theorem no_real_solution : ¬ ∃ x : ℝ, f x = 0 :=
sorry

end no_real_solution_l10_10997


namespace number_added_is_8_l10_10593

theorem number_added_is_8
  (x y : ℕ)
  (h1 : x = 265)
  (h2 : x / 5 + y = 61) :
  y = 8 :=
by
  sorry

end number_added_is_8_l10_10593


namespace inequality_system_solution_range_l10_10664

theorem inequality_system_solution_range (x m : ℝ) :
  (∃ x : ℝ, (x + 1) / 2 < x / 3 + 1 ∧ x > 3 * m) → m < 1 :=
by
  sorry

end inequality_system_solution_range_l10_10664


namespace total_cost_after_discount_l10_10868

def num_children : Nat := 6
def num_adults : Nat := 10
def num_seniors : Nat := 4

def child_ticket_price : Real := 12
def adult_ticket_price : Real := 20
def senior_ticket_price : Real := 15

def group_discount_rate : Real := 0.15

theorem total_cost_after_discount :
  let total_cost_before_discount :=
    num_children * child_ticket_price +
    num_adults * adult_ticket_price +
    num_seniors * senior_ticket_price
  let discount := group_discount_rate * total_cost_before_discount
  let total_cost := total_cost_before_discount - discount
  total_cost = 282.20 := by
  sorry

end total_cost_after_discount_l10_10868


namespace least_comic_books_l10_10097

theorem least_comic_books (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 4 = 1) : n = 17 :=
sorry

end least_comic_books_l10_10097


namespace ice_cream_flavors_l10_10529

-- Definition of the problem setup
def number_of_flavors : ℕ :=
  let scoops := 5
  let dividers := 2
  let total_objects := scoops + dividers
  Nat.choose total_objects dividers

-- Statement of the theorem
theorem ice_cream_flavors : number_of_flavors = 21 := by
  -- The proof of the theorem will use combinatorics to show the result.
  sorry

end ice_cream_flavors_l10_10529


namespace dark_chocolate_bars_sold_l10_10410

theorem dark_chocolate_bars_sold (W D : ℕ) (h₁ : 4 * D = 3 * W) (h₂ : W = 20) : D = 15 :=
by
  sorry

end dark_chocolate_bars_sold_l10_10410


namespace find_p_l10_10906

theorem find_p :
  ∀ r s : ℝ, (3 * r^2 + 4 * r + 2 = 0) → (3 * s^2 + 4 * s + 2 = 0) →
  (∀ p q : ℝ, (p = - (1/(r^2)) - (1/(s^2))) → (p = -1)) :=
by 
  intros r s hr hs p q hp
  sorry

end find_p_l10_10906


namespace find_x_l10_10294

theorem find_x (x : ℚ) (h : (3 * x - 6 + 4) / 7 = 15) : x = 107 / 3 :=
by
  sorry

end find_x_l10_10294


namespace max_regular_hours_correct_l10_10712

-- Define the conditions
def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
def total_hours_worked : ℝ := 57
def total_compensation : ℝ := 1116

-- Define the maximum regular hours per week
def max_regular_hours : ℝ := 40

-- Define the compensation equation
def compensation (H : ℝ) : ℝ :=
  regular_rate * H + overtime_rate * (total_hours_worked - H)

-- The theorem that needs to be proved
theorem max_regular_hours_correct :
  compensation max_regular_hours = total_compensation :=
by
  -- skolemize the proof
  sorry

end max_regular_hours_correct_l10_10712


namespace calculation_of_cube_exponent_l10_10709

theorem calculation_of_cube_exponent (a : ℤ) : (-2 * a^3)^3 = -8 * a^9 := by
  sorry

end calculation_of_cube_exponent_l10_10709


namespace color_nat_two_colors_no_sum_power_of_two_l10_10978

theorem color_nat_two_colors_no_sum_power_of_two :
  ∃ (f : ℕ → ℕ), (∀ a b : ℕ, a ≠ b → f a = f b → ∃ c : ℕ, c > 0 ∧ c ≠ 1 ∧ c ≠ 2 ∧ (a + b ≠ 2 ^ c)) :=
sorry

end color_nat_two_colors_no_sum_power_of_two_l10_10978


namespace constant_term_in_expansion_l10_10021

-- Define the binomial expansion general term
def binomial_general_term (x : ℤ) (r : ℕ) : ℤ :=
  (-2)^r * 3^(5 - r) * (Nat.choose 5 r) * x^(10 - 5 * r)

-- Define the condition for the specific r that makes the exponent of x zero
def condition (r : ℕ) : Prop :=
  10 - 5 * r = 0

-- Define the constant term calculation
def const_term : ℤ :=
  4 * 27 * (Nat.choose 5 2)

-- Theorem statement
theorem constant_term_in_expansion : const_term = 1080 :=
by 
  -- The proof is omitted
  sorry

end constant_term_in_expansion_l10_10021


namespace excess_percentage_l10_10155

theorem excess_percentage (A B : ℝ) (x : ℝ) 
  (hA' : A' = A * (1 + x / 100))
  (hB' : B' = B * (1 - 5 / 100))
  (h_area_err : A' * B' = 1.007 * (A * B)) : x = 6 :=
by
  sorry

end excess_percentage_l10_10155


namespace arccos_one_eq_zero_l10_10431

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  -- the proof will go here
  sorry

end arccos_one_eq_zero_l10_10431


namespace ways_to_stand_l10_10662

-- Definitions derived from conditions
def num_steps : ℕ := 7
def max_people_per_step : ℕ := 2

-- Define a function to count the number of different ways
noncomputable def count_ways : ℕ :=
  336

-- The statement to be proven in Lean 4
theorem ways_to_stand : count_ways = 336 :=
  sorry

end ways_to_stand_l10_10662


namespace cost_of_door_tickets_l10_10565

theorem cost_of_door_tickets (x : ℕ) 
  (advanced_purchase_cost : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (advanced_tickets_sold : ℕ)
  (total_revenue_advanced : ℕ := advanced_tickets_sold * advanced_purchase_cost)
  (door_tickets_sold : ℕ := total_tickets - advanced_tickets_sold) : 
  advanced_purchase_cost = 8 ∧
  total_tickets = 140 ∧
  total_revenue = 1720 ∧
  advanced_tickets_sold = 100 →
  door_tickets_sold * x + total_revenue_advanced = total_revenue →
  x = 23 := 
by
  intros h1 h2
  sorry

end cost_of_door_tickets_l10_10565


namespace remainder_sum_15_div_11_l10_10444

theorem remainder_sum_15_div_11 :
  let n := 15 
  let a := 1 
  let l := 15 
  let S := (n * (a + l)) / 2
  S % 11 = 10 :=
by
  let n := 15
  let a := 1
  let l := 15
  let S := (n * (a + l)) / 2
  show S % 11 = 10
  sorry

end remainder_sum_15_div_11_l10_10444


namespace Jurassic_Zoo_Total_l10_10490

theorem Jurassic_Zoo_Total
  (C : ℕ) (A : ℕ)
  (h1 : C = 161)
  (h2 : 8 * A + 4 * C = 964) :
  A + C = 201 := by
  sorry

end Jurassic_Zoo_Total_l10_10490


namespace original_class_strength_l10_10203

theorem original_class_strength 
  (x : ℕ) 
  (h1 : ∀ a_avg n, a_avg = 40 → n = x)
  (h2 : ∀ b_avg m, b_avg = 32 → m = 12)
  (h3 : ∀ new_avg, new_avg = 36 → ((x * 40 + 12 * 32) = ((x + 12) * 36))) : 
  x = 12 :=
by 
  sorry

end original_class_strength_l10_10203


namespace b_greater_than_neg3_l10_10574

def a_n (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem b_greater_than_neg3 (b : ℝ) :
  (∀ (n : ℕ), 0 < n → a_n (n + 1) b > a_n n b) → b > -3 :=
by
  sorry

end b_greater_than_neg3_l10_10574


namespace maximize_q_l10_10048

noncomputable def maximum_q (X Y Z : ℕ) : ℕ :=
X * Y * Z + X * Y + Y * Z + Z * X

theorem maximize_q : ∃ (X Y Z : ℕ), X + Y + Z = 15 ∧ (∀ (A B C : ℕ), A + B + C = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ A * B * C + A * B + B * C + C * A) ∧ maximum_q X Y Z = 200 :=
by
  sorry

end maximize_q_l10_10048


namespace cubic_sum_of_reciprocals_roots_l10_10480

theorem cubic_sum_of_reciprocals_roots :
  ∀ (a b c : ℝ),
  a ≠ b → b ≠ c → c ≠ a →
  0 < a ∧ a < 1 → 0 < b ∧ b < 1 → 0 < c ∧ c < 1 →
  (24 * a^3 - 38 * a^2 + 18 * a - 1 = 0) ∧
  (24 * b^3 - 38 * b^2 + 18 * b - 1 = 0) ∧
  (24 * c^3 - 38 * c^2 + 18 * c - 1 = 0) →
  ((1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 2 / 3) :=
by intros a b c neq_ab neq_bc neq_ca a_range b_range c_range roots_eqns
   sorry

end cubic_sum_of_reciprocals_roots_l10_10480


namespace rabbits_distribution_l10_10374

def num_ways_to_distribute : ℕ :=
  20 + 390 + 150

theorem rabbits_distribution :
  num_ways_to_distribute = 560 := by
  sorry

end rabbits_distribution_l10_10374


namespace determinant_value_l10_10040

-- Define the determinant calculation for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the initial conditions
variables {x : ℝ}
axiom h : x^2 - 3*x + 1 = 0

-- State the theorem to be proved
theorem determinant_value : det2x2 (x + 1) (3 * x) (x - 2) (x - 1) = 1 :=
by
  sorry

end determinant_value_l10_10040


namespace angle_terminal_side_l10_10326

theorem angle_terminal_side (k : ℤ) : ∃ α : ℝ, α = k * 360 - 30 ∧ 0 ≤ α ∧ α < 360 →
  α = 330 :=
by
  sorry

end angle_terminal_side_l10_10326


namespace david_older_than_scott_l10_10134

-- Define the ages of Richard, David, and Scott
variables (R D S : ℕ)

-- Given conditions
def richard_age_eq : Prop := R = D + 6
def richard_twice_scott : Prop := R + 8 = 2 * (S + 8)
def david_current_age : Prop := D = 14

-- Prove the statement
theorem david_older_than_scott (h1 : richard_age_eq R D) (h2 : richard_twice_scott R S) (h3 : david_current_age D) :
  D - S = 8 :=
  sorry

end david_older_than_scott_l10_10134


namespace last_digit_of_N_l10_10782

def sum_of_first_n_natural_numbers (N : ℕ) : ℕ :=
  N * (N + 1) / 2

theorem last_digit_of_N (N : ℕ) (h : sum_of_first_n_natural_numbers N = 3080) :
  N % 10 = 8 :=
by {
  sorry
}

end last_digit_of_N_l10_10782


namespace largest_unrepresentable_l10_10437

theorem largest_unrepresentable (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : Nat.gcd b c = 1) (h3 : Nat.gcd c a = 1)
  : ¬ ∃ (x y z : ℕ), x * b * c + y * c * a + z * a * b = 2 * a * b * c - a * b - b * c - c * a :=
by
  -- The proof is omitted
  sorry

end largest_unrepresentable_l10_10437


namespace each_person_ate_2_cakes_l10_10196

def initial_cakes : ℕ := 8
def number_of_friends : ℕ := 4

theorem each_person_ate_2_cakes (h_initial_cakes : initial_cakes = 8)
  (h_number_of_friends : number_of_friends = 4) :
  initial_cakes / number_of_friends = 2 :=
by sorry

end each_person_ate_2_cakes_l10_10196


namespace intersection_of_M_and_N_l10_10140

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_of_M_and_N_l10_10140


namespace no_values_of_g_g_x_eq_one_l10_10885

-- Define the function g and its properties based on the conditions
variable (g : ℝ → ℝ)
variable (h₁ : g (-4) = 1)
variable (h₂ : g (0) = 1)
variable (h₃ : g (4) = 3)
variable (h₄ : ∀ x, -4 ≤ x ∧ x ≤ 4 → g x ≥ 1)

-- Define the theorem to prove the number of values of x such that g(g(x)) = 1 is zero
theorem no_values_of_g_g_x_eq_one : ∃ n : ℕ, n = 0 ∧ (∀ x, -4 ≤ x ∧ x ≤ 4 → g (g x) = 1 → false) :=
by
  sorry -- proof to be provided later

end no_values_of_g_g_x_eq_one_l10_10885


namespace cyclic_ABCD_l10_10409

variable {Point : Type}
variable {Angle LineCircle : Type → Type}
variable {cyclicQuadrilateral : List (Point) → Prop}
variable {convexQuadrilateral : List (Point) → Prop}
variable {lineSegment : Point → Point → LineCircle Point}
variable {onSegment : Point → LineCircle Point → Prop}
variable {angle : Point → Point → Point → Angle Point}

theorem cyclic_ABCD (A B C D P Q E : Point)
  (h1 : convexQuadrilateral [A, B, C, D])
  (h2 : cyclicQuadrilateral [P, Q, D, A])
  (h3 : cyclicQuadrilateral [Q, P, B, C])
  (h4 : onSegment E (lineSegment P Q))
  (h5 : angle P A E = angle Q D E)
  (h6 : angle P B E = angle Q C E) :
  cyclicQuadrilateral [A, B, C, D] :=
  sorry

end cyclic_ABCD_l10_10409


namespace intersection_of_PQ_RS_correct_l10_10556

noncomputable def intersection_point (P Q R S : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let t := 1/9
  let s := 2/3
  (3 + 10 * t, -4 - 10 * t, 4 + 5 * t)

theorem intersection_of_PQ_RS_correct :
  let P := (3, -4, 4)
  let Q := (13, -14, 9)
  let R := (-3, 6, -9)
  let S := (1, -2, 7)
  intersection_point P Q R S = (40/9, -76/9, 49/9) :=
by {
  sorry
}

end intersection_of_PQ_RS_correct_l10_10556


namespace negative_cube_root_l10_10197

theorem negative_cube_root (a : ℝ) : ∃ x : ℝ, x ^ 3 = -a^2 - 1 ∧ x < 0 :=
by
  sorry

end negative_cube_root_l10_10197


namespace time_to_cross_man_l10_10014

-- Definitions based on the conditions
def speed_faster_train_kmph := 72 -- km per hour
def speed_slower_train_kmph := 36 -- km per hour
def length_faster_train_m := 200 -- meters

-- Convert speeds from km/h to m/s
def speed_faster_train_mps := speed_faster_train_kmph * 1000 / 3600 -- meters per second
def speed_slower_train_mps := speed_slower_train_kmph * 1000 / 3600 -- meters per second

-- Relative speed calculation
def relative_speed_mps := speed_faster_train_mps - speed_slower_train_mps -- meters per second

-- Prove the time to cross the man in the slower train
theorem time_to_cross_man : length_faster_train_m / relative_speed_mps = 20 := by
  -- Placeholder for the actual proof
  sorry

end time_to_cross_man_l10_10014


namespace sugar_already_put_in_l10_10658

-- Define the conditions
def totalSugarRequired : Nat := 14
def sugarNeededToAdd : Nat := 12
def sugarAlreadyPutIn (total : Nat) (needed : Nat) : Nat := total - needed

--State the theorem
theorem sugar_already_put_in :
  sugarAlreadyPutIn totalSugarRequired sugarNeededToAdd = 2 := 
  by
    -- Providing 'sorry' as a placeholder for the actual proof
    sorry

end sugar_already_put_in_l10_10658


namespace perpendicular_d_to_BC_l10_10807

def vector := (ℝ × ℝ)

noncomputable def AB : vector := (1, 1)
noncomputable def AC : vector := (2, 3)

noncomputable def BC : vector := (AC.1 - AB.1, AC.2 - AB.2)

def is_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

noncomputable def d : vector := (-6, 3)

theorem perpendicular_d_to_BC : is_perpendicular d BC :=
by
  sorry

end perpendicular_d_to_BC_l10_10807


namespace fraction_is_half_l10_10527

variable (N : ℕ) (F : ℚ)

theorem fraction_is_half (h1 : N = 90) (h2 : 3 + F * (1/3) * (1/5) * N = (1/15) * N) : F = 1/2 :=
by
  sorry

end fraction_is_half_l10_10527


namespace least_number_subtracted_l10_10362

-- Define the original number and the divisor
def original_number : ℕ := 427398
def divisor : ℕ := 14

-- Define the least number to be subtracted
def remainder := original_number % divisor
def least_number := remainder

-- The statement to be proven
theorem least_number_subtracted : least_number = 6 :=
by
  sorry

end least_number_subtracted_l10_10362


namespace student_percentage_first_subject_l10_10365

theorem student_percentage_first_subject
  (P : ℝ)
  (h1 : (P + 60 + 70) / 3 = 60) : P = 50 :=
  sorry

end student_percentage_first_subject_l10_10365


namespace increasing_interval_of_f_l10_10701

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval_of_f :
  ∀ x, x > 2 → ∀ y, y > x → f x < f y :=
sorry

end increasing_interval_of_f_l10_10701


namespace min_surface_area_base_edge_length_l10_10136

noncomputable def min_base_edge_length (V : ℝ) : ℝ :=
  2 * (V / (2 * Real.pi))^(1/3)

theorem min_surface_area_base_edge_length (V : ℝ) : 
  min_base_edge_length V = (4 * V)^(1/3) :=
by
  sorry

end min_surface_area_base_edge_length_l10_10136


namespace range_of_x_l10_10954

noncomputable def integerPart (x : ℝ) : ℤ := Int.floor x

theorem range_of_x (x : ℝ) (h : integerPart ((1 - 3 * x) / 2) = -1) : (1 / 3) < x ∧ x ≤ 1 :=
by
  sorry

end range_of_x_l10_10954


namespace complex_quadrant_l10_10615

theorem complex_quadrant (i : ℂ) (hi : i * i = -1) (z : ℂ) (hz : z = 1 / (1 - i)) : 
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_quadrant_l10_10615


namespace evaluate_expression_l10_10899

theorem evaluate_expression :
  (3 + 6 + 9 : ℚ) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 :=
by
  sorry

end evaluate_expression_l10_10899


namespace induction_proof_l10_10302

-- Given conditions and definitions
def plane_parts (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

-- The induction hypothesis for k ≥ 2
def induction_step (k : ℕ) (h : 2 ≤ k) : Prop :=
  plane_parts (k + 1) - plane_parts k = k + 1

-- The complete statement we want to prove
theorem induction_proof (k : ℕ) (h : 2 ≤ k) : induction_step k h := by
  sorry

end induction_proof_l10_10302


namespace inequality_solution_l10_10849

theorem inequality_solution (b c x : ℝ) (x1 x2 : ℝ)
  (hb_pos : b > 0) (hc_pos : c > 0) 
  (h_eq1 : x1 * x2 = 1) 
  (h_eq2 : -1 + x2 = 2 * x1) 
  (h_b : b = 5 / 2) 
  (h_c : c = 1) 
  : (1 < x ∧ x ≤ 5 / 2) ↔ (1 < x ∧ x ≤ 5 / 2) :=
sorry

end inequality_solution_l10_10849


namespace sum_smallest_largest_2y_l10_10691

variable (a n y : ℤ)

noncomputable def is_even (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k
noncomputable def is_odd (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k + 1

theorem sum_smallest_largest_2y 
  (h1 : is_odd a) 
  (h2 : n % 2 = 0) 
  (h3 : y = a + n) : 
  a + (a + 2 * n) = 2 * y := 
by 
  sorry

end sum_smallest_largest_2y_l10_10691


namespace linear_function_positive_in_interval_abc_sum_greater_negative_one_l10_10734

-- Problem 1
theorem linear_function_positive_in_interval (f : ℝ → ℝ) (k h m n : ℝ) (hk : k ≠ 0) (hmn : m < n)
  (hf_m : f m > 0) (hf_n : f n > 0) : (∀ x : ℝ, m < x ∧ x < n → f x > 0) :=
sorry

-- Problem 2
theorem abc_sum_greater_negative_one (a b c : ℝ)
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) : a * b + b * c + c * a > -1 :=
sorry

end linear_function_positive_in_interval_abc_sum_greater_negative_one_l10_10734


namespace sum_of_first_15_terms_l10_10242

open scoped BigOperators

-- Define the sequence as an arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + n * d

-- Define the condition given in the problem
def condition (a d : ℤ) : Prop :=
  3 * (arithmetic_sequence a d 2 + arithmetic_sequence a d 4) + 
  2 * (arithmetic_sequence a d 6 + arithmetic_sequence a d 11 + arithmetic_sequence a d 16) = 180

-- Prove that the sum of the first 15 terms is 225
theorem sum_of_first_15_terms (a d : ℤ) (h : condition a d) :
  ∑ i in Finset.range 15, arithmetic_sequence a d i = 225 :=
  sorry

end sum_of_first_15_terms_l10_10242


namespace sample_size_eq_36_l10_10695

def total_population := 27 + 54 + 81
def ratio_elderly_total := 27 / total_population
def selected_elderly := 6
def sample_size := 36

theorem sample_size_eq_36 : 
  (selected_elderly : ℚ) / (sample_size : ℚ) = ratio_elderly_total → 
  sample_size = 36 := 
by 
sorry

end sample_size_eq_36_l10_10695


namespace marble_remainder_l10_10741

theorem marble_remainder
  (r p : ℕ)
  (h_r : r % 5 = 2)
  (h_p : p % 5 = 4) :
  (r + p) % 5 = 1 :=
by
  sorry

end marble_remainder_l10_10741


namespace find_a_values_for_eccentricity_l10_10620

theorem find_a_values_for_eccentricity (a : ℝ) : 
  ( ∃ a : ℝ, ((∀ x y : ℝ, (x^2 / (a+8) + y^2 / 9 = 1)) ∧ (e = 1/2) ) 
  → (a = 4 ∨ a = -5/4)) := 
sorry

end find_a_values_for_eccentricity_l10_10620


namespace tim_kittens_l10_10181

theorem tim_kittens (K : ℕ) (h1 : (3 / 5 : ℚ) * (2 / 3 : ℚ) * K = 12) : K = 30 :=
sorry

end tim_kittens_l10_10181


namespace length_of_escalator_l10_10483

-- Given conditions
def escalator_speed : ℝ := 12 -- ft/sec
def person_speed : ℝ := 8 -- ft/sec
def time : ℝ := 8 -- seconds

-- Length of the escalator
def length : ℝ := 160 -- feet

-- Theorem stating the length of the escalator given the conditions
theorem length_of_escalator
  (h1 : escalator_speed = 12)
  (h2 : person_speed = 8)
  (h3 : time = 8)
  (combined_speed := escalator_speed + person_speed) :
  combined_speed * time = length :=
by
  -- Here the proof would go, but it's omitted as per instructions
  sorry

end length_of_escalator_l10_10483


namespace count_ordered_triples_l10_10981

theorem count_ordered_triples (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a^2 = b^2 + c^2) (h5 : b^2 = a^2 + c^2) (h6 : c^2 = a^2 + b^2) : 
  (a = b ∧ b = c ∧ a ≠ 0) ∨ (a = -b ∧ b = c ∧ a ≠ 0) ∨ (a = b ∧ b = -c ∧ a ≠ 0) ∨ (a = -b ∧ b = -c ∧ a ≠ 0) :=
sorry

end count_ordered_triples_l10_10981


namespace scientific_notation_correct_l10_10774

-- Define the number to be converted
def number : ℕ := 3790000

-- Define the correct scientific notation representation
def scientific_notation : ℝ := 3.79 * (10 ^ 6)

-- Statement to prove that number equals scientific_notation
theorem scientific_notation_correct :
  number = 3790000 → scientific_notation = 3.79 * (10 ^ 6) :=
by
  sorry

end scientific_notation_correct_l10_10774


namespace small_circle_ratio_l10_10606

theorem small_circle_ratio (a b : ℝ) (ha : 0 < a) (hb : a < b) 
  (h : π * b^2 - π * a^2 = 5 * (π * a^2)) :
  a / b = Real.sqrt 6 / 6 :=
by
  sorry

end small_circle_ratio_l10_10606


namespace product_of_roots_abs_eq_l10_10487

theorem product_of_roots_abs_eq (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  x = 5 ∨ x = -5 ∧ ((5 : ℝ) * (-5 : ℝ) = -25) := 
sorry

end product_of_roots_abs_eq_l10_10487


namespace speed_of_current_l10_10132

-- Definitions
def speed_boat_still_water := 60
def speed_downstream := 77
def speed_upstream := 43

-- Theorem statement
theorem speed_of_current : ∃ x, speed_boat_still_water + x = speed_downstream ∧ speed_boat_still_water - x = speed_upstream ∧ x = 17 :=
by
  unfold speed_boat_still_water speed_downstream speed_upstream
  sorry

end speed_of_current_l10_10132


namespace intersection_equality_l10_10654

def setA := {x : ℝ | (x - 1) * (3 - x) < 0}
def setB := {x : ℝ | -3 ≤ x ∧ x ≤ 3}

theorem intersection_equality : setA ∩ setB = {x : ℝ | -3 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_equality_l10_10654


namespace CoinRun_ProcGen_ratio_l10_10903

theorem CoinRun_ProcGen_ratio
  (greg_ppo_reward: ℝ)
  (maximum_procgen_reward: ℝ)
  (ppo_ratio: ℝ)
  (maximum_coinrun_reward: ℝ)
  (coinrun_to_procgen_ratio: ℝ)
  (greg_ppo_reward_eq: greg_ppo_reward = 108)
  (maximum_procgen_reward_eq: maximum_procgen_reward = 240)
  (ppo_ratio_eq: ppo_ratio = 0.90)
  (coinrun_equation: maximum_coinrun_reward = greg_ppo_reward / ppo_ratio)
  (ratio_definition: coinrun_to_procgen_ratio = maximum_coinrun_reward / maximum_procgen_reward) :
  coinrun_to_procgen_ratio = 0.5 :=
sorry

end CoinRun_ProcGen_ratio_l10_10903


namespace sum_of_coefficients_of_y_terms_l10_10205

theorem sum_of_coefficients_of_y_terms: 
  let p := (5 * x + 3 * y + 2) * (2 * x + 5 * y + 3)
  ∃ (a b c: ℝ), p = (10 * x^2 + a * x * y + 19 * x + b * y^2 + c * y + 6) ∧ a + b + c = 65 :=
by
  sorry

end sum_of_coefficients_of_y_terms_l10_10205


namespace sufficient_condition_implies_true_l10_10461

variable {p q : Prop}

theorem sufficient_condition_implies_true (h : p → q) : (p → q) = true :=
by
  sorry

end sufficient_condition_implies_true_l10_10461


namespace train_speed_l10_10111

theorem train_speed (distance time : ℝ) (h₁ : distance = 240) (h₂ : time = 4) : 
  ((distance / time) * 3.6) = 216 := 
by 
  rw [h₁, h₂] 
  sorry

end train_speed_l10_10111


namespace company_C_more_than_A_l10_10714

theorem company_C_more_than_A (A B C D: ℕ) (hA: A = 30) (hB: B = 2 * A)
    (hC: C = A + 10) (hD: D = C - 5) (total: A + B + C + D = 165) : C - A = 10 := 
by 
  sorry

end company_C_more_than_A_l10_10714


namespace jia_winning_strategy_l10_10902

variables {p q : ℝ}
def is_quadratic_real_roots (a b c : ℝ) : Prop := b ^ 2 - 4 * a * c > 0

def quadratic_with_roots (x1 x2 : ℝ) :=
  x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ is_quadratic_real_roots 1 (- (x1 + x2)) (x1 * x2)

def modify_jia (p q x1 : ℝ) : (ℝ × ℝ) := (p + 1, q - x1)

def modify_yi1 (p q : ℝ) : (ℝ × ℝ) := (p - 1, q)

def modify_yi2 (p q x2 : ℝ) : (ℝ × ℝ) := (p - 1, q + x2)

def winning_strategy_jia (x1 x2 : ℝ) : Prop :=
  ∃ n : ℕ, ∀ m ≥ n, ∀ p q, quadratic_with_roots x1 x2 → 
  (¬ is_quadratic_real_roots 1 p q) ∨ (q ≤ 0)

theorem jia_winning_strategy (x1 x2 : ℝ)
  (h: quadratic_with_roots x1 x2) : 
  winning_strategy_jia x1 x2 :=
sorry

end jia_winning_strategy_l10_10902


namespace exterior_angle_of_parallel_lines_l10_10353

theorem exterior_angle_of_parallel_lines (A B C x y : ℝ) (hAx : A = 40) (hBx : B = 90) (hCx : C = 40)
  (h_parallel : true)
  (h_triangle : x = 180 - A - C)
  (h_exterior_angle : y = 180 - x) :
  y = 80 := 
by
  sorry

end exterior_angle_of_parallel_lines_l10_10353


namespace harry_bought_l10_10049

-- Definitions based on the conditions
def initial_bottles := 35
def jason_bought := 5
def final_bottles := 24

-- Theorem stating the number of bottles Harry bought
theorem harry_bought :
  (initial_bottles - jason_bought) - final_bottles = 6 :=
by
  sorry

end harry_bought_l10_10049


namespace units_digit_A_is_1_l10_10768

def units_digit (n : ℕ) : ℕ := n % 10

noncomputable def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

theorem units_digit_A_is_1 : units_digit A = 1 := by
  sorry

end units_digit_A_is_1_l10_10768


namespace trip_duration_l10_10462

/--
Given:
1. The car averages 30 miles per hour for the first 5 hours of the trip.
2. The car averages 42 miles per hour for the rest of the trip.
3. The average speed for the entire trip is 34 miles per hour.

Prove: 
The total duration of the trip is 7.5 hours.
-/
theorem trip_duration (t T : ℝ) (h1 : 150 + 42 * t = 34 * T) (h2 : T = 5 + t) : T = 7.5 :=
by
  sorry

end trip_duration_l10_10462


namespace probability_of_fourth_three_is_correct_l10_10727

noncomputable def p_plus_q : ℚ := 41 + 84

theorem probability_of_fourth_three_is_correct :
  let fair_die_prob := (1 / 6 : ℚ)
  let biased_die_prob := (1 / 2 : ℚ)
  -- Probability of rolling three threes with the fair die:
  let fair_die_three_three_prob := fair_die_prob ^ 3
  -- Probability of rolling three threes with the biased die:
  let biased_die_three_three_prob := biased_die_prob ^ 3
  -- Probability of rolling three threes in total:
  let total_three_three_prob := fair_die_three_three_prob + biased_die_three_three_prob
  -- Probability of using the fair die given three threes
  let fair_die_given_three := fair_die_three_three_prob / total_three_three_prob
  -- Probability of using the biased die given three threes
  let biased_die_given_three := biased_die_three_three_prob / total_three_three_prob
  -- Probability of rolling another three:
  let fourth_three_prob := fair_die_given_three * fair_die_prob + biased_die_given_three * biased_die_prob
  -- Simplifying fraction
  let result_fraction := (41 / 84 : ℚ)
  -- Final answer p + q is 125
  p_plus_q = 125 ∧ fourth_three_prob = result_fraction
:= by
  sorry

end probability_of_fourth_three_is_correct_l10_10727


namespace tourist_growth_rate_l10_10692

theorem tourist_growth_rate (F : ℝ) (x : ℝ) 
    (hMarch : F * 0.6 = 0.6 * F)
    (hApril : F * 0.6 * 0.5 = 0.3 * F)
    (hMay : 2 * F = 2 * F):
    (0.6 * 0.5 * (1 + x) = 2) :=
by
  sorry

end tourist_growth_rate_l10_10692


namespace trig_identity_l10_10628

theorem trig_identity (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
by
  sorry

end trig_identity_l10_10628


namespace find_integers_l10_10943

-- Problem statement rewritten as a Lean 4 definition
theorem find_integers (a b c : ℤ) (H1 : a = 1) (H2 : b = 2) (H3 : c = 1) : 
  a^2 + b^2 + c^2 + 3 < a * b + 3 * b + 2 * c :=
by
  -- The proof will be presented here
  sorry

end find_integers_l10_10943


namespace monotonic_intervals_and_non_negative_f_l10_10504

noncomputable def f (m x : ℝ) : ℝ := m / x - m + Real.log x

theorem monotonic_intervals_and_non_negative_f (m : ℝ) : 
  (∀ x > 0, f m x ≥ 0) ↔ m = 1 :=
by
  sorry

end monotonic_intervals_and_non_negative_f_l10_10504


namespace last_digit_of_7_power_7_power_7_l10_10465

theorem last_digit_of_7_power_7_power_7 : (7 ^ (7 ^ 7)) % 10 = 3 :=
by
  sorry

end last_digit_of_7_power_7_power_7_l10_10465


namespace base6_addition_problem_l10_10426

theorem base6_addition_problem (X Y : ℕ) (h1 : 3 * 6^2 + X * 6 + Y + 24 = 6 * 6^2 + 1 * 6 + X) :
  X = 5 ∧ Y = 1 ∧ X + Y = 6 := by
  sorry

end base6_addition_problem_l10_10426


namespace find_f3_minus_f4_l10_10939

noncomputable def f : ℝ → ℝ := sorry

axiom h_odd : ∀ x : ℝ, f (-x) = - f x
axiom h_periodic : ∀ x : ℝ, f (x + 5) = f x
axiom h_f1 : f 1 = 1
axiom h_f2 : f 2 = 2

theorem find_f3_minus_f4 : f 3 - f 4 = -1 := by
  sorry

end find_f3_minus_f4_l10_10939


namespace find_value_of_expression_l10_10569

theorem find_value_of_expression (x y : ℝ) (h1 : |x| = 2) (h2 : |y| = 3) (h3 : x / y < 0) :
  (2 * x - y = 7) ∨ (2 * x - y = -7) :=
by
  sorry

end find_value_of_expression_l10_10569


namespace intersection_P_Q_equals_P_l10_10612

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := { y | ∃ x ∈ Set.univ, y = Real.cos x }

theorem intersection_P_Q_equals_P : P ∩ Q = P := by
  sorry

end intersection_P_Q_equals_P_l10_10612


namespace distance_from_A_to_B_l10_10317

theorem distance_from_A_to_B (D : ℝ) :
  (∃ D, (∀ tC, tC = D / 30) 
      ∧ (∀ tD, tD = D / 48 ∧ tD < (D / 30 - 1.5))
      ∧ D = 120) :=
by
  sorry

end distance_from_A_to_B_l10_10317


namespace range_of_k_l10_10390

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem range_of_k :
  (∀ x : ℝ, 2 < x → f x > k) →
  k ≤ -Real.exp 2 :=
by
  sorry

end range_of_k_l10_10390


namespace root_of_equation_value_l10_10789

theorem root_of_equation_value (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : 2 * m^2 - 4 * m + 5 = 11 := 
by
  sorry

end root_of_equation_value_l10_10789


namespace coloring_ways_l10_10422

def num_colorings (total_circles blue_circles green_circles red_circles : ℕ) : ℕ :=
  if total_circles = blue_circles + green_circles + red_circles then
    (Nat.choose total_circles (green_circles + red_circles)) * (Nat.factorial (green_circles + red_circles) / (Nat.factorial green_circles * Nat.factorial red_circles))
  else
    0

theorem coloring_ways :
  num_colorings 6 4 1 1 = 30 :=
by sorry

end coloring_ways_l10_10422


namespace prove_d_minus_r_eq_1_l10_10379

theorem prove_d_minus_r_eq_1 
  (d r : ℕ) 
  (h_d1 : d > 1)
  (h1 : 1122 % d = r)
  (h2 : 1540 % d = r)
  (h3 : 2455 % d = r) :
  d - r = 1 :=
by sorry

end prove_d_minus_r_eq_1_l10_10379


namespace max_value_f_l10_10236

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * (4 : ℝ) * x + 2

theorem max_value_f :
  ∃ x : ℝ, -f x = -18 ∧ (∀ y : ℝ, f y ≤ f x) :=
by
  sorry

end max_value_f_l10_10236


namespace find_valid_pairs_l10_10398

theorem find_valid_pairs :
  ∃ (a b c : ℕ), 
    (a = 33 ∧ b = 22 ∧ c = 1111) ∨
    (a = 66 ∧ b = 88 ∧ c = 4444) ∨
    (a = 88 ∧ b = 33 ∧ c = 7777) ∧
    (11 ≤ a ∧ a ≤ 99) ∧ (11 ≤ b ∧ b ≤ 99) ∧ (1111 ≤ c ∧ c ≤ 9999) ∧
    (a % 11 = 0) ∧ (b % 11 = 0) ∧ (c % 1111 = 0) ∧
    (a * a + b = c) := sorry

end find_valid_pairs_l10_10398


namespace Charles_chocolate_milk_total_l10_10164

theorem Charles_chocolate_milk_total (milk_per_glass syrup_per_glass total_milk total_syrup : ℝ) 
(h_milk_glass : milk_per_glass = 6.5) (h_syrup_glass : syrup_per_glass = 1.5) (h_total_milk : total_milk = 130) (h_total_syrup : total_syrup = 60) :
  (min (total_milk / milk_per_glass) (total_syrup / syrup_per_glass) * (milk_per_glass + syrup_per_glass) = 160) :=
by
  sorry

end Charles_chocolate_milk_total_l10_10164


namespace car_trip_distance_l10_10325

theorem car_trip_distance (speed_first_car speed_second_car : ℝ) (time_first_car time_second_car distance_first_car distance_second_car : ℝ) 
  (h_speed_first : speed_first_car = 30)
  (h_time_first : time_first_car = 1.5)
  (h_speed_second : speed_second_car = 60)
  (h_time_second : time_second_car = 1.3333)
  (h_distance_first : distance_first_car = speed_first_car * time_first_car)
  (h_distance_second : distance_second_car = speed_second_car * time_second_car) :
  distance_first_car = 45 :=
by
  sorry

end car_trip_distance_l10_10325


namespace ratio_in_two_years_l10_10537

def son_age : ℕ := 22
def man_age : ℕ := son_age + 24

theorem ratio_in_two_years :
  (man_age + 2) / (son_age + 2) = 2 := 
sorry

end ratio_in_two_years_l10_10537


namespace prove_a_lt_zero_l10_10880

variable (a b c : ℝ)

-- Define the quadratic function
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions:
-- The polynomial has roots at -2 and 3
def has_roots : Prop := 
  a ≠ 0 ∧ (a * (-2)^2 + b * (-2) + c = 0) ∧ (a * 3^2 + b * 3 + c = 0)

-- f(-b/(2*a)) > 0
def vertex_positive : Prop := 
  f a b c (-b / (2 * a)) > 0

-- Target: Prove a < 0
theorem prove_a_lt_zero 
  (h_roots : has_roots a b c)
  (h_vertex : vertex_positive a b c) : a < 0 := 
sorry

end prove_a_lt_zero_l10_10880


namespace large_bottle_water_amount_l10_10351

noncomputable def sport_drink_water_amount (C V : ℝ) (prop_e : ℝ) : ℝ :=
  let F := C / 4
  let W := (C * 15)
  W

theorem large_bottle_water_amount (C V : ℝ) (prop_e : ℝ) (hc : C = 7) (hprop_e : prop_e = 0.05) : sport_drink_water_amount C V prop_e = 105 := by
  sorry

end large_bottle_water_amount_l10_10351


namespace correctProduct_l10_10486

-- Define the digits reverse function
def reverseDigits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

-- Main theorem statement
theorem correctProduct (a b : ℕ) (h1 : 9 < a ∧ a < 100) (h2 : reverseDigits a * b = 143) : a * b = 341 :=
  sorry -- proof to be provided

end correctProduct_l10_10486


namespace min_value_frac_l10_10439

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 2) : 
  (2 / x) + (1 / y) ≥ 9 / 2 :=
by
  sorry

end min_value_frac_l10_10439


namespace intervals_of_decrease_l10_10404

open Real

noncomputable def func (x : ℝ) : ℝ :=
  cos (2 * x) + 2 * sin x

theorem intervals_of_decrease :
  {x | deriv func x < 0 ∧ 0 < x ∧ x < 2 * π} =
  {x | (π / 6 < x ∧ x < π / 2) ∨ (5 * π / 6 < x ∧ x < 3 * π / 2)} :=
by
  sorry

end intervals_of_decrease_l10_10404


namespace picnic_attendance_l10_10700

theorem picnic_attendance (L x : ℕ) (h1 : L + x = 2015) (h2 : L - (x - 1) = 4) : x = 1006 := 
by
  sorry

end picnic_attendance_l10_10700


namespace invest_today_for_future_value_l10_10752

-- Define the given future value, interest rate, and number of years as constants
def FV : ℝ := 600000
def r : ℝ := 0.04
def n : ℕ := 15
def target : ℝ := 333087.66

-- Define the present value calculation
noncomputable def PV : ℝ := FV / (1 + r)^n

-- State the theorem that PV is approximately equal to the target value
theorem invest_today_for_future_value : PV = target := 
by sorry

end invest_today_for_future_value_l10_10752


namespace sum_inequality_l10_10968

open Real

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a * b / (a + b)) + (b * c / (b + c)) + (c * a / (c + a)) + 
             (1 / 2) * ((a * b / c) + (b * c / a) + (c * a / b)) :=
by
  sorry

end sum_inequality_l10_10968


namespace fraction_problem_l10_10399

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20) 
  (h2 : p / n = 5) 
  (h3 : p / q = 1 / 15) : 
  m / q = 4 / 15 :=
sorry

end fraction_problem_l10_10399


namespace coloring_connected_circles_diff_colors_l10_10637

def num_ways_to_color_five_circles : ℕ :=
  36

theorem coloring_connected_circles_diff_colors (A B C D E : Type) (colors : Fin 3) 
  (connected : (A → B → C → D → E → Prop)) : num_ways_to_color_five_circles = 36 :=
by sorry

end coloring_connected_circles_diff_colors_l10_10637


namespace total_pages_read_l10_10210

variable (Jairus_pages : ℕ)
variable (Arniel_pages : ℕ)
variable (J_total : Jairus_pages = 20)
variable (A_total : Arniel_pages = 2 + 2 * Jairus_pages)

theorem total_pages_read : Jairus_pages + Arniel_pages = 62 := by
  rw [J_total, A_total]
  sorry

end total_pages_read_l10_10210


namespace find_triples_l10_10938

theorem find_triples (k : ℕ) (hk : 0 < k) :
  ∃ (a b c : ℕ), 
    (0 < a ∧ 0 < b ∧ 0 < c) ∧
    (a + b + c = 3 * k + 1) ∧ 
    (a * b + b * c + c * a = 3 * k^2 + 2 * k) ∧ 
    (a = k + 1 ∧ b = k ∧ c = k) :=
by
  sorry

end find_triples_l10_10938


namespace ratio_of_juniors_to_seniors_l10_10067

theorem ratio_of_juniors_to_seniors (j s : ℕ) (h : (1 / 3) * j = (2 / 3) * s) : j / s = 2 :=
by
  sorry

end ratio_of_juniors_to_seniors_l10_10067


namespace cryptarithmetic_proof_l10_10043

theorem cryptarithmetic_proof (A B C D : ℕ) 
  (h1 : A * B = 6) 
  (h2 : C = 2) 
  (h3 : A + B + D = 13) 
  (h4 : A + B + C = D) : 
  D = 6 :=
by
  sorry

end cryptarithmetic_proof_l10_10043


namespace variance_of_arithmetic_sequence_common_diff_3_l10_10349

noncomputable def variance (ξ : List ℝ) : ℝ :=
  let n := ξ.length
  let mean := ξ.sum / n
  let var_sum := (ξ.map (fun x => (x - mean) ^ 2)).sum
  var_sum / n

def arithmetic_sequence (a1 : ℝ) (d : ℝ) (n : ℕ) : List ℝ :=
  List.range n |>.map (fun i => a1 + i * d)

theorem variance_of_arithmetic_sequence_common_diff_3 :
  ∀ (a1 : ℝ),
    variance (arithmetic_sequence a1 3 9) = 60 :=
by
  sorry

end variance_of_arithmetic_sequence_common_diff_3_l10_10349


namespace minimum_value_of_f_on_interval_l10_10264

noncomputable def f (a x : ℝ) := Real.log x + a * x

theorem minimum_value_of_f_on_interval (a : ℝ) (h : a < 0) :
  ( ( -Real.log 2 ≤ a ∧ a < 0 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≥ a ) ∧
    ( a < -Real.log 2 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≥ (Real.log 2 + 2 * a) )
  ) :=
by
  sorry

end minimum_value_of_f_on_interval_l10_10264


namespace stacy_paper_shortage_l10_10263

theorem stacy_paper_shortage:
  let bought_sheets : ℕ := 240 + 320
  let daily_mwf : ℕ := 60
  let daily_tt : ℕ := 100
  -- Calculate sheets used in a week
  let used_one_week : ℕ := (daily_mwf * 3) + (daily_tt * 2)
  -- Calculate sheets used in two weeks
  let used_two_weeks : ℕ := used_one_week * 2
  -- Remaining sheets at the end of two weeks
  let remaining_sheets : Int := bought_sheets - used_two_weeks
  remaining_sheets = -200 :=
by sorry

end stacy_paper_shortage_l10_10263


namespace calculate_weight_difference_l10_10804

noncomputable def joe_weight := 43 -- Joe's weight in kg
noncomputable def original_avg_weight := 30 -- Original average weight in kg
noncomputable def new_avg_weight := 31 -- New average weight in kg after Joe joins
noncomputable def final_avg_weight := 30 -- Final average weight after two students leave

theorem calculate_weight_difference :
  ∃ (n : ℕ) (x : ℝ), 
  (original_avg_weight * n + joe_weight) / (n + 1) = new_avg_weight ∧
  (new_avg_weight * (n + 1) - 2 * x) / (n - 1) = final_avg_weight →
  x - joe_weight = -6.5 :=
by
  sorry

end calculate_weight_difference_l10_10804


namespace instantaneous_velocity_at_2_l10_10621

def displacement (t : ℝ) : ℝ := 100 * t - 5 * t^2

noncomputable def instantaneous_velocity_at (s : ℝ → ℝ) (t : ℝ) : ℝ :=
  (deriv s) t

theorem instantaneous_velocity_at_2 : instantaneous_velocity_at displacement 2 = 80 :=
by
  sorry

end instantaneous_velocity_at_2_l10_10621


namespace matrix_eigenvalue_neg7_l10_10470

theorem matrix_eigenvalue_neg7 (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ (v : Fin 2 → ℝ), M.mulVec v = -7 • v) →
  M = !![-7, 0; 0, -7] :=
by
  intro h
  -- proof goes here
  sorry

end matrix_eigenvalue_neg7_l10_10470


namespace probability_of_purple_is_one_fifth_l10_10430

-- Definitions related to the problem
def total_faces : ℕ := 10
def purple_faces : ℕ := 2
def probability_purple := (purple_faces : ℚ) / (total_faces : ℚ)

theorem probability_of_purple_is_one_fifth : probability_purple = 1 / 5 := 
by
  -- Converting the numbers to rationals explicitly ensures division is defined.
  change (2 : ℚ) / (10 : ℚ) = 1 / 5
  norm_num
  -- sorry (if finishing the proof manually isn't desired)

end probability_of_purple_is_one_fifth_l10_10430


namespace part1_part2_l10_10551

def f (x : ℝ) := |x + 4| - |x - 1|
def g (x : ℝ) := |2 * x - 1| + 3

theorem part1 (x : ℝ) : (f x > 3) → x > 0 :=
by sorry

theorem part2 (a : ℝ) : (∃ x, f x + 1 < 4^a - 5 * 2^a) ↔ (a < 0 ∨ a > 2) :=
by sorry

end part1_part2_l10_10551


namespace range_of_a_l10_10638

noncomputable def f (a x : ℝ) := a * x - 1
noncomputable def g (x : ℝ) := -x^2 + 2 * x + 1

theorem range_of_a (a : ℝ) :
  (∀ (x1 : ℝ), x1 ∈ (Set.Icc (-1 : ℝ) 1) → ∃ (x2 : ℝ), x2 ∈ (Set.Icc (0 : ℝ) 2) ∧ f a x1 < g x2) ↔ a ∈ Set.Ioo (-3 : ℝ) 3 :=
sorry

end range_of_a_l10_10638


namespace solution_set_inequality_l10_10749

open Set

variable {a b : ℝ}

/-- Proof Problem Statement -/
theorem solution_set_inequality (h : ∀ x : ℝ, -3 < x ∧ x < -1 ↔ a * x^2 - 1999 * x + b > 0) : 
  ∀ x : ℝ, 1 < x ∧ x < 3 ↔ a * x^2 + 1999 * x + b > 0 :=
sorry

end solution_set_inequality_l10_10749


namespace new_average_after_17th_l10_10722

def old_average (A : ℕ) (n : ℕ) : ℕ :=
  A -- A is the average before the 17th inning

def runs_in_17th : ℕ := 84 -- The score in the 17th inning

def average_increase : ℕ := 3 -- The increase in average after the 17th inning

theorem new_average_after_17th (A : ℕ) (n : ℕ) (h1 : n = 16) (h2 : old_average A n + average_increase = A + 3) :
  (old_average A n) + average_increase = 36 :=
by
  sorry

end new_average_after_17th_l10_10722


namespace notebook_cost_l10_10435

theorem notebook_cost
  (n c : ℝ)
  (h1 : n + c = 2.20)
  (h2 : n = c + 2) :
  n = 2.10 :=
by
  sorry

end notebook_cost_l10_10435


namespace dividend_divisor_quotient_l10_10079

theorem dividend_divisor_quotient (x y z : ℕ) 
  (h1 : x = 6 * y) 
  (h2 : y = 6 * z) 
  (h3 : x = y * z) : 
  x = 216 ∧ y = 36 ∧ z = 6 := 
by
  sorry

end dividend_divisor_quotient_l10_10079


namespace smallest_of_three_l10_10054

noncomputable def A : ℕ := 38 + 18
noncomputable def B : ℕ := A - 26
noncomputable def C : ℕ := B / 3

theorem smallest_of_three : C < A ∧ C < B := by
  sorry

end smallest_of_three_l10_10054


namespace acute_angle_sum_eq_pi_div_two_l10_10847

open Real

theorem acute_angle_sum_eq_pi_div_two (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_eq : sin α ^ 2 + sin β ^ 2 = sin (α + β)) : 
  α + β = π / 2 :=
sorry

end acute_angle_sum_eq_pi_div_two_l10_10847


namespace discount_amount_l10_10809

def tshirt_cost : ℕ := 30
def backpack_cost : ℕ := 10
def blue_cap_cost : ℕ := 5
def total_spent : ℕ := 43

theorem discount_amount : (tshirt_cost + backpack_cost + blue_cap_cost) - total_spent = 2 := by
  sorry

end discount_amount_l10_10809


namespace expansion_of_expression_l10_10644

theorem expansion_of_expression (x : ℝ) :
  let a := 15 * x^2 + 5 - 3 * x
  let b := 3 * x^3
  a * b = 45 * x^5 - 9 * x^4 + 15 * x^3 := by
  sorry

end expansion_of_expression_l10_10644


namespace pure_imaginary_solution_second_quadrant_solution_l10_10956

def isPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

def isSecondQuadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

def complexNumber (m : ℝ) : ℂ :=
  ⟨m^2 - 2*m - 3, m^2 + 3*m + 2⟩

theorem pure_imaginary_solution (m : ℝ) : isPureImaginary (complexNumber m) ↔ m = 3 :=
by sorry

theorem second_quadrant_solution (m : ℝ) : isSecondQuadrant (complexNumber m) ↔ (-1 < m ∧ m < 3) :=
by sorry

end pure_imaginary_solution_second_quadrant_solution_l10_10956


namespace find_a_l10_10698

theorem find_a (a : ℝ) (h : -2 * a + 1 = -1) : a = 1 :=
by sorry

end find_a_l10_10698


namespace coeff_of_z_in_eq2_l10_10173

-- Definitions of the conditions from part a)
def equation1 (x y z : ℤ) := 6 * x - 5 * y + 3 * z = 22
def equation2 (x y z : ℤ) := 4 * x + 8 * y - z = (7 : ℚ) / 11
def equation3 (x y z : ℤ) := 5 * x - 6 * y + 2 * z = 12
def sum_xyz (x y z : ℤ) := x + y + z = 10

-- Theorem stating that the coefficient of z in equation 2 is -1.
theorem coeff_of_z_in_eq2 {x y z : ℤ} (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) (h4 : sum_xyz x y z) :
    -1 = -1 :=
by
  -- This is a placeholder for the proof.
  sorry

end coeff_of_z_in_eq2_l10_10173


namespace point_in_third_quadrant_l10_10109

theorem point_in_third_quadrant (m : ℝ) : 
  (-1 < 0 ∧ -2 + m < 0) ↔ (m < 2) :=
by 
  sorry

end point_in_third_quadrant_l10_10109


namespace edward_toy_cars_l10_10072

def initial_amount : ℝ := 17.80
def cost_per_car : ℝ := 0.95
def cost_of_race_track : ℝ := 6.00
def remaining_amount : ℝ := 8.00

theorem edward_toy_cars : ∃ (n : ℕ), initial_amount - remaining_amount = n * cost_per_car + cost_of_race_track ∧ n = 4 := by
  sorry

end edward_toy_cars_l10_10072


namespace no_lunch_students_l10_10277

variable (total_students : ℕ) (cafeteria_eaters : ℕ) (lunch_bringers : ℕ)

theorem no_lunch_students : 
  total_students = 60 →
  cafeteria_eaters = 10 →
  lunch_bringers = 3 * cafeteria_eaters →
  total_students - (cafeteria_eaters + lunch_bringers) = 20 :=
by
  sorry

end no_lunch_students_l10_10277


namespace arun_gokul_age_subtract_l10_10010

theorem arun_gokul_age_subtract:
  ∃ x : ℕ, (60 - x) / 18 = 3 → x = 6 :=
sorry

end arun_gokul_age_subtract_l10_10010


namespace max_value_of_expression_l10_10919

noncomputable def expression (x : ℝ) : ℝ :=
  x^6 / (x^10 + 3 * x^8 - 5 * x^6 + 15 * x^4 + 25)

theorem max_value_of_expression : ∃ x : ℝ, (expression x) = 1 / 17 :=
sorry

end max_value_of_expression_l10_10919


namespace hundredth_odd_positive_integer_l10_10030

theorem hundredth_odd_positive_integer : 2 * 100 - 1 = 199 := 
by
  sorry

end hundredth_odd_positive_integer_l10_10030


namespace problem_1_problem_2_l10_10137

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.log (x + 1) + Real.log (1 - x) + a * (x + 1)

def mono_intervals (a : ℝ) : Set ℝ × Set ℝ := 
  if a = 1 then ((Set.Ioo (-1) (Real.sqrt 2 - 1)), (Set.Ico (Real.sqrt 2 - 1) 1)) 
  else (∅, ∅)

theorem problem_1 (a : ℝ) (h_pos : a > 0) : 
  mono_intervals a = if a = 1 then ((Set.Ioo (-1) (Real.sqrt 2 - 1)), (Set.Ico (Real.sqrt 2 - 1) 1)) else (∅, ∅) :=
sorry

theorem problem_2 (h_max : f a 0 = 1) (h_pos : a > 0) : 
  a = 1 :=
sorry

end problem_1_problem_2_l10_10137


namespace arithmetic_example_l10_10753

theorem arithmetic_example : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end arithmetic_example_l10_10753


namespace ratio_of_thermometers_to_hotwater_bottles_l10_10784

theorem ratio_of_thermometers_to_hotwater_bottles (T H : ℕ) (thermometer_price hotwater_bottle_price total_sales : ℕ) 
  (h1 : thermometer_price = 2) (h2 : hotwater_bottle_price = 6) (h3 : total_sales = 1200) (h4 : H = 60) 
  (h5 : total_sales = thermometer_price * T + hotwater_bottle_price * H) : 
  T / H = 7 :=
by
  sorry

end ratio_of_thermometers_to_hotwater_bottles_l10_10784


namespace tangent_line_at_point_l10_10133

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 4 * x + 2

def point : ℝ × ℝ := (1, -3)

def tangent_line (x y : ℝ) : Prop := 5 * x + y - 2 = 0

theorem tangent_line_at_point : tangent_line 1 (-3) :=
  sorry

end tangent_line_at_point_l10_10133


namespace stayed_days_calculation_l10_10340

theorem stayed_days_calculation (total_cost : ℕ) (charge_1st_week : ℕ) (charge_additional_week : ℕ) (first_week_days : ℕ) :
  total_cost = 302 ∧ charge_1st_week = 18 ∧ charge_additional_week = 11 ∧ first_week_days = 7 →
  ∃ D : ℕ, D = 23 :=
by {
  sorry
}

end stayed_days_calculation_l10_10340


namespace Kaleb_got_rid_of_7_shirts_l10_10118

theorem Kaleb_got_rid_of_7_shirts (initial_shirts : ℕ) (remaining_shirts : ℕ) 
    (h1 : initial_shirts = 17) (h2 : remaining_shirts = 10) : initial_shirts - remaining_shirts = 7 := 
by
  sorry

end Kaleb_got_rid_of_7_shirts_l10_10118


namespace find_first_number_l10_10342

theorem find_first_number (n : ℝ) (h1 : n / 14.5 = 175) :
  n = 2537.5 :=
by 
  sorry

end find_first_number_l10_10342


namespace arith_seq_sum_proof_l10_10992

open Function

variable (a : ℕ → ℕ) -- Define the arithmetic sequence
variables (S : ℕ → ℕ) -- Define the sum function of the sequence

-- Conditions: S_8 = 9 and S_5 = 6
axiom S8 : S 8 = 9
axiom S5 : S 5 = 6

-- Mathematical equivalence
theorem arith_seq_sum_proof : S 13 = 13 :=
sorry

end arith_seq_sum_proof_l10_10992


namespace systematic_sampling_employee_l10_10288

theorem systematic_sampling_employee {x : ℕ} (h1 : 1 ≤ 6 ∧ 6 ≤ 52) (h2 : 1 ≤ 32 ∧ 32 ≤ 52) (h3 : 1 ≤ 45 ∧ 45 ≤ 52) (h4 : 6 + 45 = x + 32) : x = 19 :=
  by
    sorry

end systematic_sampling_employee_l10_10288


namespace min_colors_needed_l10_10324

theorem min_colors_needed (n : ℕ) (h : n + n.choose 2 ≥ 12) : n = 5 :=
sorry

end min_colors_needed_l10_10324


namespace work_completion_time_l10_10189

theorem work_completion_time 
  (M W : ℝ) 
  (h1 : (10 * M + 15 * W) * 6 = 1) 
  (h2 : M * 100 = 1) 
  : W * 225 = 1 := 
by
  sorry

end work_completion_time_l10_10189


namespace total_number_of_girls_is_13_l10_10075

def number_of_girls (n : ℕ) (B : ℕ) : Prop :=
  ∃ A : ℕ, (A = B - 5) ∧ (A = B + 8)

theorem total_number_of_girls_is_13 (n : ℕ) (B : ℕ) :
  number_of_girls n B → n = 13 :=
by
  intro h
  sorry

end total_number_of_girls_is_13_l10_10075


namespace outlet_pipe_emptying_time_l10_10098

theorem outlet_pipe_emptying_time :
  let rate1 := 1 / 18
  let rate2 := 1 / 20
  let fill_time := 0.08333333333333333
  ∃ x : ℝ, (rate1 + rate2 - 1 / x = 1 / fill_time) → x = 45 :=
by
  intro rate1 rate2 fill_time
  use 45
  intro h
  sorry

end outlet_pipe_emptying_time_l10_10098


namespace at_least_two_solutions_l10_10751

theorem at_least_two_solutions (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x, (x - a) * (x - b) = x - c) ∨ (∃ x, (x - b) * (x - c) = x - a) ∨ (∃ x, (x - c) * (x - a) = x - b) ∨
    (((x - a) * (x - b) = x - c) ∧ ((x - b) * (x - c) = x - a)) ∨ 
    (((x - b) * (x + c) = x - a) ∧ ((x - c) * (x - a) = x - b)) ∨ 
    (((x - c) * (x - a) = x - b) ∧ ((x - a) * (x - b) = x - c)) :=
sorry

end at_least_two_solutions_l10_10751


namespace incorrect_inequality_l10_10366

theorem incorrect_inequality (a b : ℝ) (h : a < b) : ¬ (-4 * a < -4 * b) :=
by sorry

end incorrect_inequality_l10_10366


namespace find_orange_shells_l10_10874

theorem find_orange_shells :
  ∀ (total purple pink yellow blue : ℕ),
    total = 65 → purple = 13 → pink = 8 → yellow = 18 → blue = 12 →
    total - (purple + pink + yellow + blue) = 14 :=
by
  intros total purple pink yellow blue h_total h_purple h_pink h_yellow h_blue
  have h := h_total.symm
  rw [h_purple, h_pink, h_yellow, h_blue]
  simp only [Nat.add_assoc, Nat.add_comm, Nat.add_sub_cancel]
  sorry

end find_orange_shells_l10_10874


namespace proportional_function_decreases_l10_10571

theorem proportional_function_decreases
  (k : ℝ) (h : k ≠ 0) (h_point : ∃ k, (-4 : ℝ) = k * 2) :
  ∀ x1 x2 : ℝ, x1 < x2 → (k * x1) > (k * x2) :=
by
  sorry

end proportional_function_decreases_l10_10571


namespace complement_of_A_in_U_intersection_of_A_and_B_union_of_A_and_B_union_of_complements_of_A_and_B_l10_10086

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- Complement of a set
def C_A : Set ℕ := U \ A
def C_B : Set ℕ := U \ B

-- Questions rephrased as theorem statements
theorem complement_of_A_in_U : C_A = {2, 4, 5} := by sorry
theorem intersection_of_A_and_B : A ∩ B = ∅ := by sorry
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by sorry
theorem union_of_complements_of_A_and_B : C_A ∪ C_B = U := by sorry

end complement_of_A_in_U_intersection_of_A_and_B_union_of_A_and_B_union_of_complements_of_A_and_B_l10_10086


namespace field_trip_cost_l10_10594

def candy_bar_price : ℝ := 1.25
def candy_bars_sold : ℤ := 188
def money_from_grandma : ℝ := 250

theorem field_trip_cost : (candy_bars_sold * candy_bar_price + money_from_grandma) = 485 := 
by
  sorry

end field_trip_cost_l10_10594


namespace cube_surface_area_l10_10271

theorem cube_surface_area (a : ℝ) (h : a = 1) :
    6 * a^2 = 6 := by
  sorry

end cube_surface_area_l10_10271


namespace difference_of_one_third_and_five_l10_10318

theorem difference_of_one_third_and_five (n : ℕ) (h : n = 45) : (n / 3) - 5 = 10 :=
by
  sorry

end difference_of_one_third_and_five_l10_10318


namespace percentage_corresponding_to_120_l10_10308

variable (x p : ℝ)

def forty_percent_eq_160 := (0.4 * x = 160)
def p_times_x_eq_120 := (p * x = 120)

theorem percentage_corresponding_to_120 (h₁ : forty_percent_eq_160 x) (h₂ : p_times_x_eq_120 x p) :
  p = 0.30 :=
sorry

end percentage_corresponding_to_120_l10_10308


namespace fill_time_of_three_pipes_l10_10322

def rate (hours : ℕ) : ℚ := 1 / hours

def combined_rate : ℚ :=
  rate 12 + rate 15 + rate 20

def time_to_fill (rate : ℚ) : ℚ :=
  1 / rate

theorem fill_time_of_three_pipes :
  time_to_fill combined_rate = 5 := by
  sorry

end fill_time_of_three_pipes_l10_10322


namespace minimum_value_expression_l10_10190

theorem minimum_value_expression :
  ∀ (r s t : ℝ), (1 ≤ r ∧ r ≤ s ∧ s ≤ t ∧ t ≤ 4) →
  (r - 1) ^ 2 + (s / r - 1) ^ 2 + (t / s - 1) ^ 2 + (4 / t - 1) ^ 2 = 4 * (Real.sqrt 2 - 1) ^ 2 := 
sorry

end minimum_value_expression_l10_10190


namespace ratio_of_spinsters_to_cats_l10_10652

-- Definitions for the conditions given:
def S : ℕ := 12 -- 12 spinsters
def C : ℕ := S + 42 -- 42 more cats than spinsters
def ratio (a b : ℕ) : ℚ := a / b -- Ratio definition

-- The theorem stating the required equivalence:
theorem ratio_of_spinsters_to_cats :
  ratio S C = 2 / 9 :=
by
  -- This proof has been omitted for the purpose of this exercise.
  sorry

end ratio_of_spinsters_to_cats_l10_10652


namespace value_of_ab_plus_bc_plus_ca_l10_10220

theorem value_of_ab_plus_bc_plus_ca (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end value_of_ab_plus_bc_plus_ca_l10_10220


namespace relationship_between_a_b_c_l10_10330

noncomputable def f (x : ℝ) : ℝ := 2^(abs x) - 1
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5)
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2)
noncomputable def c : ℝ := f (Real.log (1/4) / Real.log 2)

theorem relationship_between_a_b_c : a < c ∧ c < b :=
by
  sorry

end relationship_between_a_b_c_l10_10330


namespace replacement_paint_intensity_l10_10065

theorem replacement_paint_intensity 
  (P_original : ℝ) (P_new : ℝ) (f : ℝ) (I : ℝ) :
  P_original = 50 →
  P_new = 45 →
  f = 0.2 →
  0.8 * P_original + f * I = P_new →
  I = 25 :=
by
  intros
  sorry

end replacement_paint_intensity_l10_10065


namespace system_of_equations_solution_l10_10656

theorem system_of_equations_solution (x y z : ℝ) :
  (x = 6 + Real.sqrt 29 ∧ y = (5 - 2 * (6 + Real.sqrt 29)) / 3 ∧ z = (4 - (6 + Real.sqrt 29)) / 3 ∧
   x + y + z = 3 ∧ x + 2 * y - z = 2 ∧ x + y * z + z * x = 3) ∨
  (x = 6 - Real.sqrt 29 ∧ y = (5 - 2 * (6 - Real.sqrt 29)) / 3 ∧ z = (4 - (6 - Real.sqrt 29)) / 3 ∧
   x + y + z = 3 ∧ x + 2 * y - z = 2 ∧ x + y * z + z * x = 3) :=
sorry

end system_of_equations_solution_l10_10656


namespace inequality_not_hold_l10_10452

theorem inequality_not_hold (x y : ℝ) (h : x > y) : ¬ (1 - x > 1 - y) :=
by
  -- condition and given statements
  sorry

end inequality_not_hold_l10_10452


namespace students_algebra_or_drafting_not_both_not_geography_l10_10498

variables (A D G : Finset ℕ)
-- Condition 1: Fifteen students are taking both algebra and drafting
variable (h1 : (A ∩ D).card = 15)
-- Condition 2: There are 30 students taking algebra
variable (h2 : A.card = 30)
-- Condition 3: There are 12 students taking drafting only
variable (h3 : (D \ A).card = 12)
-- Condition 4: There are eight students taking a geography class
variable (h4 : G.card = 8)
-- Condition 5: Two students are also taking both algebra and drafting and geography
variable (h5 : ((A ∩ D) ∩ G).card = 2)

-- Question: Prove the final count of students taking algebra or drafting but not both, and not taking geography is 25
theorem students_algebra_or_drafting_not_both_not_geography :
  ((A \ D) ∪ (D \ A)).card - ((A ∩ D) ∩ G).card = 25 :=
by
  sorry

end students_algebra_or_drafting_not_both_not_geography_l10_10498


namespace sufficient_drivers_and_ivan_petrovich_departure_l10_10824

/--
One-way trip duration in minutes.
-/
def one_way_trip_duration : ℕ := 160

/--
Round trip duration in minutes is twice the one-way trip duration.
-/
def round_trip_duration : ℕ := 2 * one_way_trip_duration

/--
Rest duration after a trip in minutes.
-/
def rest_duration : ℕ := 60

/--
Departure times and rest periods for drivers A, B, C starting initially, 
with additional driver D.
-/
def driver_a_return_time : ℕ := 12 * 60 + 40  -- 12:40 in minutes
def driver_a_next_departure : ℕ := driver_a_return_time + rest_duration  -- 13:40 in minutes
def driver_d_departure : ℕ := 13 * 60 + 5  -- 13:05 in minutes

/--
A proof that 4 drivers are sufficient and Ivan Petrovich departs at 10:40 AM.
-/
theorem sufficient_drivers_and_ivan_petrovich_departure :
  (4 = 4) ∧ (10 * 60 + 40 = 10 * 60 + 40) :=
by
  sorry

end sufficient_drivers_and_ivan_petrovich_departure_l10_10824


namespace identity_proof_l10_10262

theorem identity_proof (n : ℝ) (h1 : n^2 ≥ 4) (h2 : n ≠ 0) :
    (n^3 - 3*n + (n^2 - 1)*Real.sqrt (n^2 - 4) - 2) / 
    (n^3 - 3*n + (n^2 - 1)*Real.sqrt (n^2 - 4) + 2)
    = ((n + 1) * Real.sqrt (n - 2)) / ((n - 1) * Real.sqrt (n + 2)) := by
  sorry

end identity_proof_l10_10262


namespace arithmetic_contains_geometric_l10_10323

theorem arithmetic_contains_geometric {a b : ℚ} (h : a^2 + b^2 ≠ 0) :
  ∃ (c q : ℚ) (f : ℕ → ℚ), (∀ n, f n = c * q^n) ∧ (∀ n, f n = a + b * n) := 
sorry

end arithmetic_contains_geometric_l10_10323


namespace expression_value_l10_10218

theorem expression_value (x y z : ℤ) (hx : x = 25) (hy : y = 30) (hz : z = 10) :
  (x - (y - z)) - ((x - y) - z) = 20 :=
by
  rw [hx, hy, hz]
  -- After substituting the values, we will need to simplify the expression to reach 20.
  sorry

end expression_value_l10_10218


namespace choose_president_vice_president_and_committee_l10_10680

theorem choose_president_vice_president_and_committee :
  let num_ways : ℕ := 10 * 9 * (Nat.choose 8 2)
  num_ways = 2520 :=
by
  sorry

end choose_president_vice_president_and_committee_l10_10680


namespace bobby_candy_l10_10812

theorem bobby_candy (C G : ℕ) (H : C + G = 36) (Hchoc: (2/3 : ℚ) * C = 12) (Hgummy: (3/4 : ℚ) * G = 9) : 
  (1/3 : ℚ) * C + (1/4 : ℚ) * G = 9 :=
by
  sorry

end bobby_candy_l10_10812


namespace time_to_finish_task_l10_10845

-- Define the conditions
def printerA_rate (total_pages : ℕ) (time_A_alone : ℕ) : ℚ := total_pages / time_A_alone
def printerB_rate (rate_A : ℚ) : ℚ := rate_A + 10

-- Define the combined rate of printers working together
def combined_rate (rate_A : ℚ) (rate_B : ℚ) : ℚ := rate_A + rate_B

-- Define the time taken to finish the task together
def time_to_finish (total_pages : ℕ) (combined_rate : ℚ) : ℚ := total_pages / combined_rate

-- Given conditions
def total_pages : ℕ := 35
def time_A_alone : ℕ := 60

-- Definitions derived from given conditions
def rate_A : ℚ := printerA_rate total_pages time_A_alone
def rate_B : ℚ := printerB_rate rate_A

-- Combined rate when both printers work together
def combined_rate_AB : ℚ := combined_rate rate_A rate_B

-- Lean theorem statement to prove time taken by both printers
theorem time_to_finish_task : time_to_finish total_pages combined_rate_AB = 210 / 67 := 
by
  sorry

end time_to_finish_task_l10_10845


namespace option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l10_10838

variable (a b : ℝ)
variable (h : a < b)

theorem option_A_correct : a + 2 < b + 2 := by
  sorry

theorem option_B_correct : 3 * a < 3 * b := by
  sorry

theorem option_C_correct : (1 / 2) * a < (1 / 2) * b := by
  sorry

theorem option_D_incorrect : ¬(-2 * a < -2 * b) := by
  sorry

end option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l10_10838


namespace second_largest_consecutive_odd_195_l10_10853

theorem second_largest_consecutive_odd_195 :
  ∃ x : Int, (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 195 ∧ (x + 2) = 41 := by
  sorry

end second_largest_consecutive_odd_195_l10_10853


namespace average_infections_l10_10870

theorem average_infections (x : ℝ) (h : 1 + x + x^2 = 121) : x = 10 :=
sorry

end average_infections_l10_10870


namespace sum_of_x_y_l10_10860

theorem sum_of_x_y (x y : ℝ) (h : (x + y + 2) * (x + y - 1) = 0) : x + y = -2 ∨ x + y = 1 :=
by sorry

end sum_of_x_y_l10_10860


namespace leo_score_l10_10278

-- Definitions for the conditions
def caroline_score : ℕ := 13
def anthony_score : ℕ := 19
def winning_score : ℕ := 21

-- Lean statement for the proof problem
theorem leo_score : ∃ (leo_score : ℕ), leo_score = winning_score := by
  have h_caroline := caroline_score
  have h_anthony := anthony_score
  have h_winning := winning_score
  use 21
  sorry

end leo_score_l10_10278


namespace linette_problem_proof_l10_10916

def boxes_with_neither_markers_nor_stickers (total_boxes markers stickers both : ℕ) : ℕ :=
  total_boxes - (markers + stickers - both)

theorem linette_problem_proof : 
  let total_boxes := 15
  let markers := 9
  let stickers := 5
  let both := 4
  boxes_with_neither_markers_nor_stickers total_boxes markers stickers both = 5 :=
by
  sorry

end linette_problem_proof_l10_10916


namespace num_rectangles_grid_l10_10679

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l10_10679


namespace find_m_value_l10_10648

theorem find_m_value (f g : ℝ → ℝ) (m : ℝ)
    (hf : ∀ x, f x = 3 * x ^ 2 - 1 / x + 4)
    (hg : ∀ x, g x = x ^ 2 - m)
    (hfg : f 3 - g 3 = 5) :
    m = -50 / 3 :=
  sorry

end find_m_value_l10_10648


namespace problem1_problem2_problem3_l10_10882

-- Problem 1
theorem problem1 (x : ℝ) : (3 * (x - 1)^2 = 12) ↔ (x = 3 ∨ x = -1) :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (3 * x^2 - 6 * x - 2 = 0) ↔ (x = (3 + Real.sqrt 15) / 3 ∨ x = (3 - Real.sqrt 15) / 3) :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (3 * x * (2 * x + 1) = 4 * x + 2) ↔ (x = -1 / 2 ∨ x = 2 / 3) :=
by
  sorry

end problem1_problem2_problem3_l10_10882


namespace communication_scenarios_l10_10033

theorem communication_scenarios
  (nA : ℕ) (nB : ℕ) (hA : nA = 10) (hB : nB = 20) : 
  (∃ scenarios : ℕ, scenarios = 2 ^ (nA * nB)) :=
by
  use 2 ^ (10 * 20)
  sorry

end communication_scenarios_l10_10033


namespace initial_distance_between_stations_l10_10331

theorem initial_distance_between_stations
  (speedA speedB distanceA : ℝ)
  (rateA rateB : speedA = 40 ∧ speedB = 30)
  (dist_travelled : distanceA = 200) :
  (distanceA / speedA) * speedB + distanceA = 350 := by
  sorry

end initial_distance_between_stations_l10_10331


namespace range_of_2a_minus_b_l10_10013

theorem range_of_2a_minus_b (a b : ℝ) (h1 : a > b) (h2 : 2 * a^2 - a * b - b^2 - 4 = 0) :
  (2 * a - b) ∈ (Set.Ici (8 / 3)) :=
sorry

end range_of_2a_minus_b_l10_10013


namespace committee_size_l10_10663

theorem committee_size (n : ℕ) (h : 2 * n = 6) (p : ℚ) (h_prob : p = 2/5) : n = 3 :=
by
  -- problem conditions
  have h1 : 2 * n = 6 := h
  have h2 : p = 2/5 := h_prob
  -- skip the proof details
  sorry

end committee_size_l10_10663


namespace krishan_money_l10_10785

theorem krishan_money 
  (R G K : ℝ)
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (h3 : R = 490) : K = 2890 :=
sorry

end krishan_money_l10_10785


namespace curves_intersect_exactly_three_points_l10_10763

theorem curves_intersect_exactly_three_points (a : ℝ) :
  (∃! (p : ℝ × ℝ), p.1 ^ 2 + p.2 ^ 2 = a ^ 2 ∧ p.2 = p.1 ^ 2 - a) ↔ a > (1 / 2) :=
by sorry

end curves_intersect_exactly_three_points_l10_10763


namespace max_rectangle_area_l10_10604

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 48) : x * y ≤ 144 :=
by
  sorry

end max_rectangle_area_l10_10604


namespace part1_part2_l10_10832

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x - m|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * f x m - f (x + m) m

theorem part1 (h : ∀ x, g x m ≥ -1) : m = 1 :=
  sorry

theorem part2 {a b m : ℝ} (ha : |a| < m) (hb : |b| < m) (a_ne_zero : a ≠ 0) (hm: m = 1) : 
  f (a * b) m > |a| * f (b / a) m :=
  sorry

end part1_part2_l10_10832


namespace june_time_to_bernard_l10_10344

theorem june_time_to_bernard (distance_Julia : ℝ) (time_Julia : ℝ) (distance_Bernard_June : ℝ) (time_Bernard : ℝ) (distance_June_Bernard : ℝ)
  (h1 : distance_Julia = 2) (h2 : time_Julia = 6) (h3 : distance_Bernard_June = 5) (h4 : time_Bernard = 15) (h5 : distance_June_Bernard = 7) :
  distance_June_Bernard / (distance_Julia / time_Julia) = 21 := by
    sorry

end june_time_to_bernard_l10_10344


namespace gcd_m_l10_10116

def m' : ℕ := 33333333
def n' : ℕ := 555555555

theorem gcd_m'_n' : Nat.gcd m' n' = 3 := by
  sorry

end gcd_m_l10_10116


namespace jimin_yuna_difference_l10_10822

-- Definitions based on the conditions.
def seokjin_marbles : ℕ := 3
def yuna_marbles : ℕ := seokjin_marbles - 1
def jimin_marbles : ℕ := seokjin_marbles * 2

-- Theorem stating the problem we need to prove: the difference in marbles between Jimin and Yuna is 4.
theorem jimin_yuna_difference : jimin_marbles - yuna_marbles = 4 :=
by sorry

end jimin_yuna_difference_l10_10822


namespace geometric_seq_sum_l10_10927

theorem geometric_seq_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * (-1)) 
  (h_a3 : a 3 = 3) 
  (h_sum_cond : a 2016 + a 2017 = 0) : 
  S 101 = 3 := 
by
  sorry

end geometric_seq_sum_l10_10927


namespace product_of_four_consecutive_integers_divisible_by_12_l10_10673

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l10_10673


namespace total_teaching_time_l10_10788

def teaching_times :=
  let eduardo_math_time := 3 * 60
  let eduardo_science_time := 4 * 90
  let eduardo_history_time := 2 * 120
  let total_eduardo_time := eduardo_math_time + eduardo_science_time + eduardo_history_time

  let frankie_math_time := 2 * (3 * 60)
  let frankie_science_time := 2 * (4 * 90)
  let frankie_history_time := 2 * (2 * 120)
  let total_frankie_time := frankie_math_time + frankie_science_time + frankie_history_time

  let georgina_math_time := 3 * (3 * 80)
  let georgina_science_time := 3 * (4 * 100)
  let georgina_history_time := 3 * (2 * 150)
  let total_georgina_time := georgina_math_time + georgina_science_time + georgina_history_time

  total_eduardo_time + total_frankie_time + total_georgina_time

theorem total_teaching_time : teaching_times = 5160 := by
  -- calculations omitted
  sorry

end total_teaching_time_l10_10788


namespace gcf_84_112_210_l10_10703

theorem gcf_84_112_210 : gcd (gcd 84 112) 210 = 14 := by sorry

end gcf_84_112_210_l10_10703


namespace number_of_matches_among_three_players_l10_10962

-- Define the given conditions
variables (n r : ℕ) -- n is the number of participants, r is the number of matches among the 3 players
variables (m : ℕ := 50) -- m is the total number of matches played

-- Given assumptions
def condition1 := m = 50
def condition2 := ∃ (n: ℕ), 50 = Nat.choose (n-3) 2 + r + (6 - 2 * r)

-- The target proof
theorem number_of_matches_among_three_players (n r : ℕ) (m : ℕ := 50)
  (h1 : m = 50)
  (h2 : ∃ (n: ℕ), 50 = Nat.choose (n-3) 2 + r + (6 - 2 * r)) :
  r = 1 :=
sorry

end number_of_matches_among_three_players_l10_10962


namespace range_g_minus_2x_l10_10303

variable (g : ℝ → ℝ)
variable (x : ℝ)

axiom g_values : ∀ x, x ∈ Set.Icc (-4 : ℝ) 4 → 
  (g x = x ∨ g x = x - 1 ∨ g x = x - 2 ∨ g x = x - 3 ∨ g x = x - 4)

axiom g_le_2x : ∀ x, x ∈ Set.Icc (-4 : ℝ) 4 → g x ≤ 2 * x

theorem range_g_minus_2x : 
  Set.range (fun x => g x - 2 * x) = Set.Icc (-5 : ℝ) 0 :=
sorry

end range_g_minus_2x_l10_10303


namespace sum_of_distances_l10_10842

theorem sum_of_distances (AB A'B' AD A'D' x y : ℝ) 
  (h1 : AB = 8)
  (h2 : A'B' = 6)
  (h3 : AD = 3)
  (h4 : A'D' = 1)
  (h5 : x = 2)
  (h6 : x / y = 3 / 2) : 
  x + y = 10 / 3 :=
by
  sorry

end sum_of_distances_l10_10842


namespace terez_farm_pregnant_cows_percentage_l10_10147

theorem terez_farm_pregnant_cows_percentage (total_cows : ℕ) (female_percentage : ℕ) (pregnant_females : ℕ) 
  (ht : total_cows = 44) (hf : female_percentage = 50) (hp : pregnant_females = 11) :
  (pregnant_females * 100 / (female_percentage * total_cows / 100) = 50) :=
by 
  sorry

end terez_farm_pregnant_cows_percentage_l10_10147


namespace parabola_inequality_l10_10192

theorem parabola_inequality (a c y1 y2 : ℝ) (h1 : a < 0)
  (h2 : y1 = a * (-1 - 1)^2 + c)
  (h3 : y2 = a * (4 - 1)^2 + c) :
  y1 > y2 :=
sorry

end parabola_inequality_l10_10192


namespace find_average_age_of_students_l10_10668

-- Given conditions
variables (n : ℕ) (T : ℕ) (A : ℕ)

-- 20 students in the class
def students : ℕ := 20

-- Teacher's age is 42 years
def teacher_age : ℕ := 42

-- When the teacher's age is included, the average age increases by 1
def average_age_increase (A : ℕ) := A + 1

-- Proof problem statement in Lean 4
theorem find_average_age_of_students (A : ℕ) :
  20 * A + 42 = 21 * (A + 1) → A = 21 :=
by
  -- Here should be the proof steps, added sorry to skip the proof
  sorry

end find_average_age_of_students_l10_10668


namespace remainder_polynomial_2047_l10_10469

def f (r : ℤ) : ℤ := r ^ 11 - 1

theorem remainder_polynomial_2047 : f 2 = 2047 :=
by
  sorry

end remainder_polynomial_2047_l10_10469


namespace owen_initial_turtles_l10_10107

variables (O J : ℕ)

-- Conditions
def johanna_turtles := J = O - 5
def owen_final_turtles := 2 * O + J / 2 = 50

-- Theorem statement
theorem owen_initial_turtles (h1 : johanna_turtles O J) (h2 : owen_final_turtles O J) : O = 21 :=
sorry

end owen_initial_turtles_l10_10107


namespace binary_predecessor_l10_10758

def M : ℕ := 84
def N : ℕ := 83
def M_bin : ℕ := 0b1010100
def N_bin : ℕ := 0b1010011

theorem binary_predecessor (H : M = M_bin ∧ N = M - 1) : N = N_bin := by
  sorry

end binary_predecessor_l10_10758


namespace reciprocal_opposite_of_neg_neg_3_is_neg_one_third_l10_10778

theorem reciprocal_opposite_of_neg_neg_3_is_neg_one_third : 
  (1 / (-(-3))) = -1 / 3 :=
by
  sorry

end reciprocal_opposite_of_neg_neg_3_is_neg_one_third_l10_10778


namespace eggs_per_chicken_per_week_l10_10761

-- Define the conditions
def chickens : ℕ := 10
def price_per_dozen : ℕ := 2  -- in dollars
def earnings_in_2_weeks : ℕ := 20  -- in dollars
def weeks : ℕ := 2
def eggs_per_dozen : ℕ := 12

-- Define the question as a theorem to be proved
theorem eggs_per_chicken_per_week : 
  (earnings_in_2_weeks / price_per_dozen) * eggs_per_dozen / (chickens * weeks) = 6 :=
by
  -- proof steps
  sorry

end eggs_per_chicken_per_week_l10_10761


namespace plane_second_trace_line_solutions_l10_10524

noncomputable def num_solutions_second_trace_line
  (first_trace_line : Line)
  (angle_with_projection_plane : ℝ)
  (intersection_outside_paper : Prop) : ℕ :=
2

theorem plane_second_trace_line_solutions
  (first_trace_line : Line)
  (angle_with_projection_plane : ℝ)
  (intersection_outside_paper : Prop) :
  num_solutions_second_trace_line first_trace_line angle_with_projection_plane intersection_outside_paper = 2 := by
sorry

end plane_second_trace_line_solutions_l10_10524


namespace mask_production_l10_10863

theorem mask_production (x : ℝ) :
  24 + 24 * (1 + x) + 24 * (1 + x)^2 = 88 :=
sorry

end mask_production_l10_10863


namespace g_value_at_2_l10_10608

def g (x : ℝ) (d : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + d * x - 8

theorem g_value_at_2 (d : ℝ) (h : g (-2) d = 4) : g 2 d = -84 := by
  sorry

end g_value_at_2_l10_10608


namespace prop3_prop4_l10_10141

-- Definitions to represent planes and lines
variable (Plane Line : Type)

-- Predicate representing parallel planes or lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Predicate representing perpendicular planes or lines
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Distinct planes and a line
variables (α β γ : Plane) (l : Line)

-- Proposition 3: If l ⊥ α and l ∥ β, then α ⊥ β
theorem prop3 : perpendicular_line_plane l α ∧ parallel_line_plane l β → perpendicular α β :=
sorry

-- Proposition 4: If α ∥ β and α ⊥ γ, then β ⊥ γ
theorem prop4 : parallel α β ∧ perpendicular α γ → perpendicular β γ :=
sorry

end prop3_prop4_l10_10141


namespace ratio_of_areas_l10_10510

-- Define the conditions
def angle_Q_smaller_circle : ℝ := 60
def angle_Q_larger_circle : ℝ := 30
def arc_length_equal (C1 C2 : ℝ) : Prop := 
  (angle_Q_smaller_circle / 360) * C1 = (angle_Q_larger_circle / 360) * C2

-- The required Lean statement that proves the ratio of the areas
theorem ratio_of_areas (C1 C2 r1 r2 : ℝ) 
  (arc_eq : arc_length_equal C1 C2) : 
  (π * r1^2) / (π * r2^2) = 1 / 4 := 
by 
  sorry

end ratio_of_areas_l10_10510


namespace lemonade_price_fraction_l10_10871

theorem lemonade_price_fraction :
  (2 / 5) * (L / S) = 0.35714285714285715 → L / S = 0.8928571428571429 :=
by
  intro h
  sorry

end lemonade_price_fraction_l10_10871


namespace factory_minimize_salary_l10_10185

theorem factory_minimize_salary :
  ∃ x : ℕ, ∃ W : ℕ,
    x + (120 - x) = 120 ∧
    800 * x + 1000 * (120 - x) = W ∧
    120 - x ≥ 3 * x ∧
    x = 30 ∧
    W = 114000 :=
  sorry

end factory_minimize_salary_l10_10185


namespace Toms_swimming_speed_is_2_l10_10360

theorem Toms_swimming_speed_is_2
  (S : ℝ)
  (h1 : 2 * S + 4 * S = 12) :
  S = 2 :=
by
  sorry

end Toms_swimming_speed_is_2_l10_10360


namespace part1_solution_part2_solution_part3_solution_l10_10310

-- Part (1): Prove the solution of the system of equations 
theorem part1_solution (x y : ℝ) (h1 : x - y - 1 = 0) (h2 : 4 * (x - y) - y = 5) : 
  x = 0 ∧ y = -1 := 
sorry

-- Part (2): Prove the solution of the system of equations 
theorem part2_solution (x y : ℝ) (h1 : 2 * x - 3 * y - 2 = 0) 
  (h2 : (2 * x - 3 * y + 5) / 7 + 2 * y = 9) : 
  x = 7 ∧ y = 4 := 
sorry

-- Part (3): Prove the range of the parameter m
theorem part3_solution (m : ℕ) (h1 : 2 * (2 : ℝ) * x + y = (-3 : ℝ) * ↑m + 2) 
  (h2 : x + 2 * y = 7) (h3 : x + y > -5 / 6) : 
  m = 1 ∨ m = 2 ∨ m = 3 :=
sorry

end part1_solution_part2_solution_part3_solution_l10_10310


namespace value_of_x_l10_10873

theorem value_of_x (x : ℤ) : (3000 + x) ^ 2 = x ^ 2 → x = -1500 := 
by
  sorry

end value_of_x_l10_10873


namespace range_of_f_l10_10883

open Real

noncomputable def f (x y z w : ℝ) : ℝ :=
  x / (x + y) + y / (y + z) + z / (z + x) + w / (w + x)

theorem range_of_f (x y z w : ℝ) (h1x : 0 < x) (h1y : 0 < y) (h1z : 0 < z) (h1w : 0 < w) :
  1 < f x y z w ∧ f x y z w < 2 :=
  sorry

end range_of_f_l10_10883


namespace a_ge_zero_of_set_nonempty_l10_10184

theorem a_ge_zero_of_set_nonempty {a : ℝ} (h : ∃ x : ℝ, x^2 = a) : a ≥ 0 :=
sorry

end a_ge_zero_of_set_nonempty_l10_10184


namespace range_of_k_l10_10736

theorem range_of_k :
  ∀ (a k : ℝ) (f : ℝ → ℝ),
    (∀ x, f x = if x ≥ 0 then k^2 * x + a^2 - k else x^2 + (a^2 + 4 * a) * x + (2 - a)^2) →
    (∀ x1 x2 : ℝ, x1 ≠ 0 → x2 ≠ 0 → x1 ≠ x2 → f x1 = f x2 → False) →
    -20 ≤ k ∧ k ≤ -4 :=
by
  sorry

end range_of_k_l10_10736


namespace last_two_digits_10_93_10_31_plus_3_eq_08_l10_10246

def last_two_digits_fraction_floor (n m d : ℕ) : ℕ :=
  let x := 10^n
  let y := 10^m + d
  (x / y) % 100

theorem last_two_digits_10_93_10_31_plus_3_eq_08 :
  last_two_digits_fraction_floor 93 31 3 = 08 :=
by
  sorry

end last_two_digits_10_93_10_31_plus_3_eq_08_l10_10246


namespace simplify_expression_l10_10640

theorem simplify_expression : 
  (4 * 6 / (12 * 14)) * (8 * 12 * 14 / (4 * 6 * 8)) = 1 := by
  sorry

end simplify_expression_l10_10640


namespace gcd_gx_x_multiple_of_18432_l10_10685

def g (x : ℕ) : ℕ := (3*x + 5) * (7*x + 2) * (13*x + 7) * (2*x + 10)

theorem gcd_gx_x_multiple_of_18432 (x : ℕ) (h : ∃ k : ℕ, x = 18432 * k) : Nat.gcd (g x) x = 28 :=
by
  sorry

end gcd_gx_x_multiple_of_18432_l10_10685


namespace minimum_value_inequality_l10_10457

theorem minimum_value_inequality (x y z : ℝ) (hx : 2 ≤ x) (hxy : x ≤ y) (hyz : y ≤ z) (hz : z ≤ 5) :
    (x - 2)^2 + (y / x - 2)^2 + (z / y - 2)^2 + (5 / z - 2)^2 ≥ 4 * (Real.sqrt (Real.sqrt 5) - 2)^2 := 
    sorry

end minimum_value_inequality_l10_10457


namespace tammy_speed_on_second_day_l10_10828

-- Definitions of the conditions
variables (t v : ℝ)
def total_hours := 14
def total_distance := 52

-- Distance equation
def distance_eq := v * t + (v + 0.5) * (t - 2) = total_distance

-- Time equation
def time_eq := t + (t - 2) = total_hours

theorem tammy_speed_on_second_day :
  (time_eq t ∧ distance_eq v t) → v + 0.5 = 4 :=
by sorry

end tammy_speed_on_second_day_l10_10828


namespace find_x_l10_10597

theorem find_x (x : ℝ) :
  (1 / 3) * ((3 * x + 4) + (7 * x - 5) + (4 * x + 9)) = (5 * x - 3) → x = 17 :=
by
  sorry

end find_x_l10_10597


namespace spinner_final_direction_north_l10_10957

def start_direction := "north"
def clockwise_revolutions := (7 : ℚ) / 2
def counterclockwise_revolutions := (5 : ℚ) / 2
def net_revolutions := clockwise_revolutions - counterclockwise_revolutions

theorem spinner_final_direction_north :
  net_revolutions = 1 → start_direction = "north" → 
  start_direction = "north" :=
by
  intro h1 h2
  -- Here you would prove that net_revolutions of 1 full cycle leads back to start
  exact h2 -- Skipping proof

end spinner_final_direction_north_l10_10957


namespace find_x_l10_10182

-- Define the vectors and the condition of them being parallel
def vector_a : (ℝ × ℝ) := (3, 1)
def vector_b (x : ℝ) : (ℝ × ℝ) := (x, -1)
def parallel (a b : (ℝ × ℝ)) := ∃ k : ℝ, b = (k * a.1, k * a.2)

-- The theorem to prove
theorem find_x (x : ℝ) (h : parallel (3, 1) (x, -1)) : x = -3 :=
by
  sorry

end find_x_l10_10182


namespace num_children_proof_l10_10643

noncomputable def number_of_children (total_persons : ℕ) (total_revenue : ℕ) (adult_price : ℕ) (child_price : ℕ) : ℕ :=
  let adult_tickets := (child_price * total_persons - total_revenue) / (child_price - adult_price)
  let child_tickets := total_persons - adult_tickets
  child_tickets

theorem num_children_proof : number_of_children 280 14000 60 25 = 80 := 
by
  unfold number_of_children
  sorry

end num_children_proof_l10_10643


namespace triangle_intersect_sum_l10_10034

theorem triangle_intersect_sum (P Q R S T U : ℝ × ℝ) :
  P = (0, 8) →
  Q = (0, 0) →
  R = (10, 0) →
  S = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  T = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) →
  ∃ U : ℝ × ℝ, 
    (U.1 = (P.1 + ((T.2 - P.2) / (T.1 - P.1)) * (U.1 - P.1)) ∧
     U.2 = (R.2 + ((S.2 - R.2) / (S.1 - R.1)) * (U.1 - R.1))) ∧
    (U.1 + U.2) = 6 :=
by
  sorry

end triangle_intersect_sum_l10_10034


namespace minimum_button_presses_to_exit_l10_10402

def arms_after (r y : ℕ) : ℕ := 3 + r - 2 * y
def doors_after (y g : ℕ) : ℕ := 3 + y - 2 * g

theorem minimum_button_presses_to_exit :
  ∃ r y g : ℕ, arms_after r y = 0 ∧ doors_after y g = 0 ∧ r + y + g = 9 :=
sorry

end minimum_button_presses_to_exit_l10_10402


namespace taxi_trip_distance_l10_10092

theorem taxi_trip_distance
  (initial_fee : ℝ)
  (per_segment_charge : ℝ)
  (segment_distance : ℝ)
  (total_charge : ℝ)
  (segments_traveled : ℝ)
  (total_miles : ℝ) :
  initial_fee = 2.25 →
  per_segment_charge = 0.3 →
  segment_distance = 2/5 →
  total_charge = 4.95 →
  total_miles = segments_traveled * segment_distance →
  segments_traveled = (total_charge - initial_fee) / per_segment_charge →
  total_miles = 3.6 :=
by
  intros h_initial_fee h_per_segment_charge h_segment_distance h_total_charge h_total_miles h_segments_traveled
  sorry

end taxi_trip_distance_l10_10092


namespace next_in_sequence_is_80_l10_10742

def seq (n : ℕ) : ℕ := n^2 - 1

theorem next_in_sequence_is_80 :
  seq 9 = 80 :=
by
  sorry

end next_in_sequence_is_80_l10_10742


namespace matching_times_l10_10783

noncomputable def chargeAtTime (t : Nat) : ℚ :=
  100 - t / 6

def isMatchingTime (hh mm : Nat) : Prop :=
  hh * 60 + mm = 100 - (hh * 60 + mm) / 6

theorem matching_times:
  isMatchingTime 4 52 ∨
  isMatchingTime 5 43 ∨
  isMatchingTime 6 35 ∨
  isMatchingTime 7 26 ∨
  isMatchingTime 9 9 :=
by
  repeat { sorry }

end matching_times_l10_10783


namespace part_a_l10_10704

theorem part_a (x : ℝ) (hx : x ≥ 1) : x^3 - 5 * x^2 + 8 * x - 4 ≥ 0 := 
  sorry

end part_a_l10_10704


namespace minimize_t_l10_10260

variable (Q : ℝ) (Q_1 Q_2 Q_3 Q_4 Q_5 Q_6 Q_7 Q_8 Q_9 : ℝ)

-- Definition of the sum of undirected lengths
def t (Q : ℝ) := 
  abs (Q - Q_1) + abs (Q - Q_2) + abs (Q - Q_3) + 
  abs (Q - Q_4) + abs (Q - Q_5) + abs (Q - Q_6) + 
  abs (Q - Q_7) + abs (Q - Q_8) + abs (Q - Q_9)

-- Statement that t is minimized when Q = Q_5
theorem minimize_t : ∀ Q : ℝ, t Q ≥ t Q_5 := 
sorry

end minimize_t_l10_10260
