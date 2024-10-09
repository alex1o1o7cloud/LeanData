import Mathlib

namespace geometric_sequence_product_correct_l1376_137697

noncomputable def geometric_sequence_product (a_1 a_5 : ℝ) (a_2 a_3 a_4 : ℝ) :=
  a_1 = 1 / 2 ∧ a_5 = 8 ∧ a_2 * a_4 = a_1 * a_5 ∧ a_3^2 = a_1 * a_5

theorem geometric_sequence_product_correct:
  ∃ a_2 a_3 a_4 : ℝ, geometric_sequence_product (1 / 2) 8 a_2 a_3 a_4 ∧ (a_2 * a_3 * a_4 = 8) :=
by
  sorry

end geometric_sequence_product_correct_l1376_137697


namespace compute_custom_op_l1376_137627

def custom_op (x y : ℤ) : ℤ := 
  x * y - y * x - 3 * x + 2 * y

theorem compute_custom_op : (custom_op 9 5) - (custom_op 5 9) = -20 := 
by
  sorry

end compute_custom_op_l1376_137627


namespace rationalize_expression_l1376_137614

theorem rationalize_expression :
  (2 * Real.sqrt 3) / (Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5) = 
  (Real.sqrt 6 + 3 - Real.sqrt 15) / 2 :=
sorry

end rationalize_expression_l1376_137614


namespace range_of_a_l1376_137630

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + (1 / 2) * a

theorem range_of_a (a : ℝ) : (∀ x > 0, f a x ≤ 0) → (0 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l1376_137630


namespace find_four_letter_list_with_equal_product_l1376_137668

open Nat

theorem find_four_letter_list_with_equal_product :
  ∃ (L T M W : ℕ), 
  (L * T * M * W = 23 * 24 * 25 * 26) 
  ∧ (1 ≤ L ∧ L ≤ 26) ∧ (1 ≤ T ∧ T ≤ 26) ∧ (1 ≤ M ∧ M ≤ 26) ∧ (1 ≤ W ∧ W ≤ 26) 
  ∧ (L ≠ T) ∧ (T ≠ M) ∧ (M ≠ W) ∧ (W ≠ L) ∧ (L ≠ M) ∧ (T ≠ W)
  ∧ (L * T * M * W) = (12 * 20 * 13 * 23) :=
by
  sorry

end find_four_letter_list_with_equal_product_l1376_137668


namespace johns_age_is_25_l1376_137685

variable (JohnAge DadAge SisterAge : ℕ)

theorem johns_age_is_25
    (h1 : JohnAge = DadAge - 30)
    (h2 : JohnAge + DadAge = 80)
    (h3 : SisterAge = JohnAge - 5) :
    JohnAge = 25 := 
sorry

end johns_age_is_25_l1376_137685


namespace cost_of_batman_game_l1376_137603

noncomputable def footballGameCost : ℝ := 14.02
noncomputable def strategyGameCost : ℝ := 9.46
noncomputable def totalAmountSpent : ℝ := 35.52

theorem cost_of_batman_game :
  totalAmountSpent - (footballGameCost + strategyGameCost) = 12.04 :=
by
  -- The proof is omitted as instructed.
  sorry

end cost_of_batman_game_l1376_137603


namespace g_is_zero_l1376_137652

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (4 * (Real.sin x)^4 + (Real.cos x)^2) - 
  Real.sqrt (4 * (Real.cos x)^4 + (Real.sin x)^2)

theorem g_is_zero (x : ℝ) : g x = 0 := 
  sorry

end g_is_zero_l1376_137652


namespace number_less_than_neg_one_is_neg_two_l1376_137693

theorem number_less_than_neg_one_is_neg_two : ∃ x : ℤ, x = -1 - 1 ∧ x = -2 := by
  sorry

end number_less_than_neg_one_is_neg_two_l1376_137693


namespace value_of_expression_l1376_137683

variable {a b c d e f : ℝ}

theorem value_of_expression :
  a * b * c = 130 →
  b * c * d = 65 →
  c * d * e = 1000 →
  d * e * f = 250 →
  (a * f) / (c * d) = 1 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end value_of_expression_l1376_137683


namespace first_player_can_always_make_A_eq_6_l1376_137601

def maxSum3x3In5x5Board (board : Fin 5 → Fin 5 → ℕ) (i j : Fin 3) : ℕ :=
  (i + 3 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 3 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 3 : Fin 5) * (j + 5 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 5 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 5 : Fin 5)

theorem first_player_can_always_make_A_eq_6 :
  ∀ (board : Fin 5 → Fin 5 → ℕ), 
  (∀ (i j : Fin 3), maxSum3x3In5x5Board board i j = 6)
  :=
by
  intros board i j
  sorry

end first_player_can_always_make_A_eq_6_l1376_137601


namespace sufficient_and_necessary_condition_l1376_137638

def isMonotonicallyIncreasing {R : Type _} [LinearOrderedField R] (f : R → R) :=
  ∀ x y, x < y → f x < f y

def fx {R : Type _} [LinearOrderedField R] (x m : R) :=
  x^3 + 2*x^2 + m*x + 1

theorem sufficient_and_necessary_condition (m : ℝ) :
  (isMonotonicallyIncreasing (λ x => fx x m) ↔ m ≥ 4/3) :=
  sorry

end sufficient_and_necessary_condition_l1376_137638


namespace triangle_angles_and_side_l1376_137640

noncomputable def triangle_properties : Type := sorry

variables {A B C : ℝ}
variables {a b c : ℝ}

theorem triangle_angles_and_side (hA : A = 60)
    (ha : a = 4 * Real.sqrt 3)
    (hb : b = 4 * Real.sqrt 2)
    (habc : triangle_properties)
    : B = 45 ∧ C = 75 ∧ c = 2 * Real.sqrt 2 + 2 * Real.sqrt 6 := 
sorry

end triangle_angles_and_side_l1376_137640


namespace pow_div_simplify_l1376_137650

theorem pow_div_simplify : (((15^15) / (15^14))^3 * 3^3) / 3^3 = 3375 := by
  sorry

end pow_div_simplify_l1376_137650


namespace find_k_values_l1376_137661

theorem find_k_values (k : ℝ) : 
  (∃ (x y : ℝ), x + 2 * y - 1 = 0 ∧ x + 1 = 0 ∧ x + k * y = 0) → 
  (k = 0 ∨ k = 1 ∨ k = 2) ∧
  (k = 0 ∨ k = 1 ∨ k = 2 → ∃ (x y : ℝ), x + 2 * y - 1 = 0 ∧ x + 1 = 0 ∧ x + k * y = 0) :=
by
  sorry

end find_k_values_l1376_137661


namespace find_e_value_l1376_137673

-- Define constants a, b, c, d, and e
variables (a b c d e : ℝ)

-- Theorem statement
theorem find_e_value (h1 : (2 : ℝ)^7 * a + (2 : ℝ)^5 * b + (2 : ℝ)^3 * c + 2 * d + e = 23)
                     (h2 : ((-2) : ℝ)^7 * a + ((-2) : ℝ)^5 * b + ((-2) : ℝ)^3 * c + ((-2) : ℝ) * d + e = -35) :
  e = -6 :=
sorry

end find_e_value_l1376_137673


namespace sequence_increasing_range_l1376_137604

theorem sequence_increasing_range (a : ℝ) (n : ℕ) : 
  (∀ n ≤ 5, (a - 1) ^ (n - 4) < (a - 1) ^ ((n+1) - 4)) ∧
  (∀ n > 5, (7 - a) * n - 1 < (7 - a) * (n + 1) - 1) ∧
  (a - 1 < (7 - a) * 6 - 1) 
  → 2 < a ∧ a < 6 := 
sorry

end sequence_increasing_range_l1376_137604


namespace apples_bought_is_28_l1376_137669

-- Define the initial number of apples, number of apples used, and total number of apples after buying more
def initial_apples : ℕ := 38
def apples_used : ℕ := 20
def total_apples_after_buying : ℕ := 46

-- State the theorem: the number of apples bought is 28
theorem apples_bought_is_28 : (total_apples_after_buying - (initial_apples - apples_used)) = 28 := 
by sorry

end apples_bought_is_28_l1376_137669


namespace perpendicular_line_to_plane_l1376_137684

variables {Point Line Plane : Type}
variables (a b c : Line) (α : Plane) (A : Point)

-- Define the conditions
def line_perpendicular_to (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def lines_intersect_at (l1 l2 : Line) (P : Point) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Given conditions in Lean 4
variables (h1 : line_perpendicular_to c a)
variables (h2 : line_perpendicular_to c b)
variables (h3 : line_in_plane a α)
variables (h4 : line_in_plane b α)
variables (h5 : lines_intersect_at a b A)

-- The theorem statement to prove
theorem perpendicular_line_to_plane : line_perpendicular_to_plane c α :=
sorry

end perpendicular_line_to_plane_l1376_137684


namespace payment_plan_months_l1376_137618

theorem payment_plan_months 
  (M T : ℝ) (r : ℝ) 
  (hM : M = 100)
  (hT : T = 1320)
  (hr : r = 0.10)
  : ∃ t : ℕ, t = 12 ∧ T = (M * t) + (M * t * r) :=
by
  sorry

end payment_plan_months_l1376_137618


namespace water_saving_percentage_l1376_137689

/-- 
Given:
1. The old toilet uses 5 gallons of water per flush.
2. The household flushes 15 times per day.
3. John saved 1800 gallons of water in June.

Prove that the percentage of water saved per flush by the new toilet compared 
to the old one is 80%.
-/
theorem water_saving_percentage 
  (old_toilet_usage_per_flush : ℕ)
  (flushes_per_day : ℕ)
  (savings_in_june : ℕ)
  (days_in_june : ℕ) :
  old_toilet_usage_per_flush = 5 →
  flushes_per_day = 15 →
  savings_in_june = 1800 →
  days_in_june = 30 →
  (old_toilet_usage_per_flush * flushes_per_day * days_in_june - savings_in_june)
  * 100 / (old_toilet_usage_per_flush * flushes_per_day * days_in_june) = 80 :=
by 
  sorry

end water_saving_percentage_l1376_137689


namespace solve_x_eq_l1376_137671

theorem solve_x_eq : ∃ x : ℚ, -3 * x - 12 = 6 * x + 9 ∧ x = -7 / 3 :=
by 
  sorry

end solve_x_eq_l1376_137671


namespace combined_depths_underwater_l1376_137617

theorem combined_depths_underwater :
  let Ron_height := 12
  let Dean_height := Ron_height - 11
  let Sam_height := Dean_height + 2
  let Ron_depth := Ron_height / 2
  let Sam_depth := Sam_height
  let Dean_depth := Dean_height + 3
  Ron_depth + Sam_depth + Dean_depth = 13 :=
by
  let Ron_height := 12
  let Dean_height := Ron_height - 11
  let Sam_height := Dean_height + 2
  let Ron_depth := Ron_height / 2
  let Sam_depth := Sam_height
  let Dean_depth := Dean_height + 3
  show Ron_depth + Sam_depth + Dean_depth = 13
  sorry

end combined_depths_underwater_l1376_137617


namespace find_value_of_question_mark_l1376_137605

theorem find_value_of_question_mark (q : ℕ) : q * 40 = 173 * 240 → q = 1036 :=
by
  intro h
  sorry

end find_value_of_question_mark_l1376_137605


namespace abs_neg_three_l1376_137615

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_l1376_137615


namespace distinct_paths_in_grid_l1376_137619

def number_of_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

theorem distinct_paths_in_grid :
  number_of_paths 7 8 = 6435 :=
by
  sorry

end distinct_paths_in_grid_l1376_137619


namespace log_sum_reciprocals_of_logs_l1376_137698

-- Problem (1)
theorem log_sum (log_two : Real.log 2 ≠ 0) :
    Real.log 4 / Real.log 10 + Real.log 50 / Real.log 10 - Real.log 2 / Real.log 10 = 2 := by
  sorry

-- Problem (2)
theorem reciprocals_of_logs (a b : Real) (h : 1 + Real.log a / Real.log 2 = 2 + Real.log b / Real.log 3 ∧ (1 + Real.log a / Real.log 2) = Real.log (a + b) / Real.log 6) : 
    1 / a + 1 / b = 6 := by
  sorry

end log_sum_reciprocals_of_logs_l1376_137698


namespace sum_of_fractions_l1376_137649

theorem sum_of_fractions : 
  (2/100) + (5/1000) + (5/10000) + 3 * (4/1000) = 0.0375 := 
by 
  sorry

end sum_of_fractions_l1376_137649


namespace hyperbola_shares_focus_with_eccentricity_length_of_chord_AB_l1376_137654

theorem hyperbola_shares_focus_with_eccentricity 
  (a1 b1 : ℝ) (h1 : a1 = 3 ∧ b1 = 2)
  (e : ℝ) (h_eccentricity : e = (Real.sqrt 5) / 2)
  (c : ℝ) (h_focus : c = Real.sqrt (a1^2 - b1^2)) :
  (∃ a b : ℝ, a^2 - b^2 = c^2 ∧ c/a = e ∧ a = 2 ∧ b = 1) :=
sorry

theorem length_of_chord_AB 
  (a b : ℝ) (h_ellipse : a^2 = 4 ∧ b^2 = 1)
  (c : ℝ) (h_focus : c = Real.sqrt (a^2 - b^2))
  (f : ℝ) (h_f : f = Real.sqrt 3)
  (line_eq : ℝ -> ℝ) (h_line_eq : ∀ x, line_eq x = x - f) :
  (∃ x1 x2 : ℝ, 
    x1 + x2 = (8 * Real.sqrt 3) / 5 ∧
    x1 * x2 = 8 / 5 ∧
    Real.sqrt ((x1 - x2)^2 + (line_eq x1 - line_eq x2)^2) = 8 / 5) :=
sorry

end hyperbola_shares_focus_with_eccentricity_length_of_chord_AB_l1376_137654


namespace remainder_of_22_divided_by_3_l1376_137611

theorem remainder_of_22_divided_by_3 : ∃ (r : ℕ), 22 = 3 * 7 + r ∧ r = 1 := by
  sorry

end remainder_of_22_divided_by_3_l1376_137611


namespace ratio_c_to_d_l1376_137632

theorem ratio_c_to_d (a b c d : ℚ) 
  (h1 : a / b = 3 / 4) 
  (h2 : b / c = 7 / 9) 
  (h3 : a / d = 0.4166666666666667) : 
  c / d = 5 / 7 := 
by
  -- Proof not needed
  sorry

end ratio_c_to_d_l1376_137632


namespace seq_formula_l1376_137635

noncomputable def seq {a : Nat → ℝ} (h1 : a 2 - a 1 = 1) (h2 : ∀ n, a (n + 1) - a n = 2 * (n - 1) + 1) : Nat → ℝ :=
sorry

theorem seq_formula {a : Nat → ℝ} 
  (h1 : a 2 - a 1 = 1)
  (h2 : ∀ n, a (n + 1) - a n = 2 * (n - 1) + 1)
  (n : Nat) : a n = 2 ^ n - 1 :=
sorry

end seq_formula_l1376_137635


namespace smallest_k_l1376_137621

def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) : (∀ z : ℂ, z ≠ 0 → f z ∣ z^k - 1) ↔ k = 40 :=
by sorry

end smallest_k_l1376_137621


namespace correct_fraction_subtraction_l1376_137670

theorem correct_fraction_subtraction (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) :
  ((1 / x) - (1 / (x - 1))) = - (1 / (x^2 - x)) :=
by
  sorry

end correct_fraction_subtraction_l1376_137670


namespace certain_event_is_eventC_l1376_137691

-- Definitions for the conditions:
def eventA := "A vehicle randomly arriving at an intersection encountering a red light"
def eventB := "The sun rising from the west in the morning"
def eventC := "Two out of 400 people sharing the same birthday"
def eventD := "Tossing a fair coin with the head facing up"

-- The proof goal: proving that event C is the certain event.
theorem certain_event_is_eventC : eventC = "Two out of 400 people sharing the same birthday" :=
sorry

end certain_event_is_eventC_l1376_137691


namespace remainder_of_5n_mod_11_l1376_137628

theorem remainder_of_5n_mod_11 (n : ℤ) (h : n % 11 = 1) : (5 * n) % 11 = 5 := 
by
  sorry

end remainder_of_5n_mod_11_l1376_137628


namespace trapezium_area_l1376_137677

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 12) :
  (1 / 2 * (a + b) * h = 228) :=
by
  sorry

end trapezium_area_l1376_137677


namespace remainder_when_divided_by_39_l1376_137600

theorem remainder_when_divided_by_39 (N k : ℤ) (h : N = 13 * k + 4) : (N % 39) = 4 :=
sorry

end remainder_when_divided_by_39_l1376_137600


namespace conditional_probability_l1376_137625

def slips : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_odd (n : ℕ) : Prop := n % 2 ≠ 0

def P_A : ℚ := 5/9

def P_A_and_B : ℚ := 5/9 * 4/8

theorem conditional_probability :
  (5 / 18) / (5 / 9) = 1 / 2 :=
by
  sorry

end conditional_probability_l1376_137625


namespace students_behind_Yoongi_l1376_137674

theorem students_behind_Yoongi :
  ∀ (n : ℕ), n = 20 → ∀ (j y : ℕ), j = 1 → y = 2 → n - y = 18 :=
by
  intros n h1 j h2 y h3
  sorry

end students_behind_Yoongi_l1376_137674


namespace will_has_123_pieces_of_candy_l1376_137620

def initial_candy_pieces (chocolate_boxes mint_boxes caramel_boxes : ℕ)
  (pieces_per_chocolate_box pieces_per_mint_box pieces_per_caramel_box : ℕ) : ℕ :=
  chocolate_boxes * pieces_per_chocolate_box + mint_boxes * pieces_per_mint_box + caramel_boxes * pieces_per_caramel_box

def given_away_candy_pieces (given_chocolate_boxes given_mint_boxes given_caramel_boxes : ℕ)
  (pieces_per_chocolate_box pieces_per_mint_box pieces_per_caramel_box : ℕ) : ℕ :=
  given_chocolate_boxes * pieces_per_chocolate_box + given_mint_boxes * pieces_per_mint_box + given_caramel_boxes * pieces_per_caramel_box

def remaining_candy : ℕ :=
  let initial := initial_candy_pieces 7 5 4 12 15 10
  let given_away := given_away_candy_pieces 3 2 1 12 15 10
  initial - given_away

theorem will_has_123_pieces_of_candy : remaining_candy = 123 :=
by
  -- Proof goes here
  sorry

end will_has_123_pieces_of_candy_l1376_137620


namespace folded_paper_perimeter_l1376_137675

theorem folded_paper_perimeter (L W : ℝ) 
  (h1 : 2 * L + W = 34)         -- Condition 1
  (h2 : L * W = 140)            -- Condition 2
  : 2 * W + L = 38 :=           -- Goal
sorry

end folded_paper_perimeter_l1376_137675


namespace total_digits_l1376_137623

theorem total_digits (n S S6 S4 : ℕ) 
  (h1 : S = 80 * n)
  (h2 : S6 = 6 * 58)
  (h3 : S4 = 4 * 113)
  (h4 : S = S6 + S4) : 
  n = 10 :=
by 
  sorry

end total_digits_l1376_137623


namespace at_least_one_female_team_l1376_137686

open Classical

namespace Probability

-- Define the Problem
noncomputable def prob_at_least_one_female (females males : ℕ) (team_size : ℕ) :=
  let total_students := females + males
  let total_ways := Nat.choose total_students team_size
  let ways_all_males := Nat.choose males team_size
  1 - (ways_all_males / total_ways : ℝ)

-- Verify the given problem against the expected answer
theorem at_least_one_female_team :
  prob_at_least_one_female 1 3 2 = 1 / 2 := by
  sorry

end Probability

end at_least_one_female_team_l1376_137686


namespace find_a_l1376_137659

open Classical

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (|f x a| < 1) ↔ (x ∈ Set.Ioo (-2) 0 ∨ x ∈ Set.Ioo 2 4)) → a = 1 :=
by
  intro h
  sorry

end find_a_l1376_137659


namespace expressions_equal_iff_l1376_137643

theorem expressions_equal_iff (a b c: ℝ) : a + 2 * b * c = (a + b) * (a + 2 * c) ↔ a + b + 2 * c = 0 :=
by 
  sorry

end expressions_equal_iff_l1376_137643


namespace union_sets_l1376_137681

noncomputable def M : Set ℤ := {1, 2, 3}
noncomputable def N : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem union_sets : M ∪ N = {0, 1, 2, 3} := by
  sorry

end union_sets_l1376_137681


namespace variance_binom_4_half_l1376_137641

-- Define the binomial variance function
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Define the conditions
def n := 4
def p := 1 / 2

-- The target statement
theorem variance_binom_4_half : binomial_variance n p = 1 :=
by
  -- The proof goes here
  sorry

end variance_binom_4_half_l1376_137641


namespace still_need_more_volunteers_l1376_137662

def total_volunteers_needed : ℕ := 80
def students_volunteering_per_class : ℕ := 4
def number_of_classes : ℕ := 5
def teacher_volunteers : ℕ := 10
def total_student_volunteers : ℕ := students_volunteering_per_class * number_of_classes
def total_volunteers_so_far : ℕ := total_student_volunteers + teacher_volunteers

theorem still_need_more_volunteers : total_volunteers_needed - total_volunteers_so_far = 50 := by
  sorry

end still_need_more_volunteers_l1376_137662


namespace harmful_bacteria_time_l1376_137634

noncomputable def number_of_bacteria (x : ℝ) : ℝ :=
  4000 * 2^x

theorem harmful_bacteria_time :
  ∃ (x : ℝ), number_of_bacteria x > 90000 ∧ x = 4.5 :=
by
  sorry

end harmful_bacteria_time_l1376_137634


namespace initial_candies_l1376_137682

theorem initial_candies (L R : ℕ) (h1 : L + R = 27) (h2 : R - L = 2 * L + 3) : L = 6 ∧ R = 21 :=
by
  sorry

end initial_candies_l1376_137682


namespace Wendy_bouquets_l1376_137646

def num_flowers_before : ℕ := 45
def num_wilted_flowers : ℕ := 35
def flowers_per_bouquet : ℕ := 5

theorem Wendy_bouquets : (num_flowers_before - num_wilted_flowers) / flowers_per_bouquet = 2 := by
  sorry

end Wendy_bouquets_l1376_137646


namespace smallest_value_c_zero_l1376_137651

noncomputable def smallest_possible_c (a b c : ℝ) : ℝ :=
if h : (0 < a) ∧ (0 < b) ∧ (0 < c) then
  0
else
  c

theorem smallest_value_c_zero (a b c : ℝ) (h : (0 < a) ∧ (0 < b) ∧ (0 < c)) :
  smallest_possible_c a b c = 0 :=
by
  sorry

end smallest_value_c_zero_l1376_137651


namespace fraction_addition_l1376_137629

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l1376_137629


namespace twentieth_century_years_as_powers_of_two_diff_l1376_137612

theorem twentieth_century_years_as_powers_of_two_diff :
  ∀ (y : ℕ), (1900 ≤ y ∧ y < 2000) →
    ∃ (n k : ℕ), y = 2^n - 2^k ↔ y = 1984 ∨ y = 1920 := 
by
  sorry

end twentieth_century_years_as_powers_of_two_diff_l1376_137612


namespace min_value_of_a_l1376_137607

theorem min_value_of_a (a : ℝ) :
  (¬ ∃ x0 : ℝ, -1 < x0 ∧ x0 ≤ 2 ∧ x0 - a > 0) → a = 2 :=
by
  sorry

end min_value_of_a_l1376_137607


namespace find_extra_lives_first_level_l1376_137688

-- Conditions as definitions
def initial_lives : ℕ := 2
def extra_lives_second_level : ℕ := 11
def total_lives_after_second_level : ℕ := 19

-- Definition representing the extra lives in the first level
def extra_lives_first_level (x : ℕ) : Prop :=
  initial_lives + x + extra_lives_second_level = total_lives_after_second_level

-- The theorem we need to prove
theorem find_extra_lives_first_level : ∃ x : ℕ, extra_lives_first_level x ∧ x = 6 :=
by
  sorry  -- Placeholder for the proof

end find_extra_lives_first_level_l1376_137688


namespace questionnaire_visitors_l1376_137626

theorem questionnaire_visitors (V E : ℕ) (H1 : 140 = V - E) 
  (H2 : E = (3 * V) / 4) : V = 560 :=
by
  sorry

end questionnaire_visitors_l1376_137626


namespace least_subtraction_divisible_l1376_137655

def least_subtrahend (n m : ℕ) : ℕ :=
n % m

theorem least_subtraction_divisible (n : ℕ) (m : ℕ) (sub : ℕ) :
  n = 13604 → m = 87 → sub = least_subtrahend n m → (n - sub) % m = 0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end least_subtraction_divisible_l1376_137655


namespace total_cats_l1376_137653

theorem total_cats (current_cats : ℕ) (additional_cats : ℕ) (h1 : current_cats = 11) (h2 : additional_cats = 32):
  current_cats + additional_cats = 43 :=
by
  -- We state the given conditions:
  -- current_cats = 11
  -- additional_cats = 32
  -- We need to prove:
  -- current_cats + additional_cats = 43
  sorry

end total_cats_l1376_137653


namespace negation_universal_exists_l1376_137692

open Classical

theorem negation_universal_exists :
  (¬ ∀ x : ℝ, x > 0 → (x^2 - x + 3 > 0)) ↔ ∃ x : ℝ, x > 0 ∧ (x^2 - x + 3 ≤ 0) :=
by
  sorry

end negation_universal_exists_l1376_137692


namespace negation_correct_l1376_137666

variable (x : Real)

def original_proposition : Prop :=
  x > 0 → x^2 > 0

def negation_proposition : Prop :=
  x ≤ 0 → x^2 ≤ 0

theorem negation_correct :
  ¬ original_proposition x = negation_proposition x :=
by 
  sorry

end negation_correct_l1376_137666


namespace solve_equation_nat_numbers_l1376_137633

theorem solve_equation_nat_numbers :
  ∃ (x y z : ℕ), (2 ^ x + 3 ^ y + 7 = z!) ∧ ((x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5)) := 
sorry

end solve_equation_nat_numbers_l1376_137633


namespace addition_belongs_to_Q_l1376_137610

def P : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def R : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1}

theorem addition_belongs_to_Q (a b : ℤ) (ha : a ∈ P) (hb : b ∈ Q) : a + b ∈ Q := by
  sorry

end addition_belongs_to_Q_l1376_137610


namespace sunland_more_plates_than_moonland_l1376_137680

theorem sunland_more_plates_than_moonland : 
  let sunland_plates := 26^4 * 10^2
  let moonland_plates := 26^3 * 10^3
  (sunland_plates - moonland_plates) = 7321600 := 
by
  sorry

end sunland_more_plates_than_moonland_l1376_137680


namespace solution_proof_l1376_137690

noncomputable def f (n : ℕ) : ℝ := Real.logb 143 (n^2)

theorem solution_proof : f 7 + f 11 + f 13 = 2 + 2 * Real.logb 143 7 := by
  sorry

end solution_proof_l1376_137690


namespace projection_identity_l1376_137631

variables (P : ℝ × ℝ × ℝ) (x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ)

-- Define point P as (-1, 3, -4)
def point_P := (-1, 3, -4) = P

-- Define projections on the coordinate planes
def projection_yoz := (x1, y1, z1) = (0, 3, -4)
def projection_zox := (x2, y2, z2) = (-1, 0, -4)
def projection_xoy := (x3, y3, z3) = (-1, 3, 0)

-- Prove that x1^2 + y2^2 + z3^2 = 0 under the given conditions
theorem projection_identity :
  point_P P ∧ projection_yoz x1 y1 z1 ∧ projection_zox x2 y2 z2 ∧ projection_xoy x3 y3 z3 →
  (x1^2 + y2^2 + z3^2 = 0) :=
by
  sorry

end projection_identity_l1376_137631


namespace multiple_of_6_is_multiple_of_3_l1376_137657

theorem multiple_of_6_is_multiple_of_3 (n : ℕ) (h1 : ∀ k : ℕ, n = 6 * k)
  : ∃ m : ℕ, n = 3 * m :=
by sorry

end multiple_of_6_is_multiple_of_3_l1376_137657


namespace interest_difference_l1376_137658

noncomputable def annual_amount (P r t : ℝ) : ℝ :=
P * (1 + r)^t

noncomputable def monthly_amount (P r n t : ℝ) : ℝ :=
P * (1 + r / n)^(n * t)

theorem interest_difference
  (P : ℝ)
  (r : ℝ)
  (n : ℕ)
  (t : ℝ)
  (annual_compounded : annual_amount P r t = 8000 * (1 + 0.10)^3)
  (monthly_compounded : monthly_amount P r 12 3 = 8000 * (1 + 0.10 / 12) ^ (12 * 3)) :
  (monthly_amount P r 12 t - annual_amount P r t) = 142.80 := 
sorry

end interest_difference_l1376_137658


namespace gcd_values_count_l1376_137636

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 392) : ∃ d, d = 11 := 
sorry

end gcd_values_count_l1376_137636


namespace john_pack_count_l1376_137606

-- Defining the conditions
def utensilsInPack : Nat := 30
def knivesInPack : Nat := utensilsInPack / 3
def forksInPack : Nat := utensilsInPack / 3
def spoonsInPack : Nat := utensilsInPack / 3
def requiredKnivesRatio : Nat := 2
def requiredForksRatio : Nat := 3
def requiredSpoonsRatio : Nat := 5
def minimumSpoons : Nat := 50

-- Proving the solution
theorem john_pack_count : 
  ∃ packs : Nat, 
    (packs * spoonsInPack >= minimumSpoons) ∧
    (packs * foonsInPack / packs * knivesInPack = requiredForksRatio / requiredKnivesRatio) ∧
    (packs * spoonsInPack / packs * forksInPack = requiredForksRatio / requiredSpoonsRatio) ∧
    (packs * spoonsInPack / packs * knivesInPack = requiredSpoonsRatio / requiredKnivesRatio) ∧
    packs = 5 :=
sorry

end john_pack_count_l1376_137606


namespace anton_food_cost_l1376_137687

def food_cost_julie : ℝ := 10
def food_cost_letitia : ℝ := 20
def tip_per_person : ℝ := 4
def num_people : ℕ := 3
def tip_percentage : ℝ := 0.20

theorem anton_food_cost (A : ℝ) :
  tip_percentage * (food_cost_julie + food_cost_letitia + A) = tip_per_person * num_people →
  A = 30 :=
by
  intro h
  sorry

end anton_food_cost_l1376_137687


namespace solve_quadratic_roots_l1376_137645

theorem solve_quadratic_roots (b c : ℝ) 
  (h : {1, 2} = {x : ℝ | x^2 + b * x + c = 0}) : 
  b = -3 ∧ c = 2 :=
by
  sorry

end solve_quadratic_roots_l1376_137645


namespace hexagon_colorings_correct_l1376_137660

noncomputable def hexagon_colorings : Nat :=
  let colors := ["blue", "orange", "purple"]
  2 -- As determined by the solution.

theorem hexagon_colorings_correct :
  hexagon_colorings = 2 :=
by
  sorry

end hexagon_colorings_correct_l1376_137660


namespace number_of_new_bottle_caps_l1376_137676

def threw_away := 6
def total_bottle_caps_now := 60
def found_more_bottle_caps := 44

theorem number_of_new_bottle_caps (N : ℕ) (h1 : N = threw_away + found_more_bottle_caps) : N = 50 :=
sorry

end number_of_new_bottle_caps_l1376_137676


namespace tangent_line_at_one_l1376_137644

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x

theorem tangent_line_at_one (a b : ℝ) (h_tangent : ∀ x, f x = a * x + b) : 
  a + b = 1 := 
sorry

end tangent_line_at_one_l1376_137644


namespace solve_eq1_solve_eq2_l1376_137667

-- Definition of the first equation
def eq1 (x : ℝ) : Prop := (1 / 2) * x^2 - 8 = 0

-- Definition of the second equation
def eq2 (x : ℝ) : Prop := (x - 5)^3 = -27

-- Proof statement for the value of x in the first equation
theorem solve_eq1 (x : ℝ) : eq1 x ↔ x = 4 ∨ x = -4 := by
  sorry

-- Proof statement for the value of x in the second equation
theorem solve_eq2 (x : ℝ) : eq2 x ↔ x = 2 := by
  sorry

end solve_eq1_solve_eq2_l1376_137667


namespace radio_loss_percentage_l1376_137622

theorem radio_loss_percentage :
  ∀ (cost_price selling_price : ℝ), 
    cost_price = 1500 → 
    selling_price = 1290 → 
    ((cost_price - selling_price) / cost_price) * 100 = 14 :=
by
  intros cost_price selling_price h_cp h_sp
  sorry

end radio_loss_percentage_l1376_137622


namespace range_of_sum_l1376_137699

theorem range_of_sum (a b : ℝ) (h : a^2 - a * b + b^2 = a + b) :
  0 ≤ a + b ∧ a + b ≤ 4 :=
by
  sorry

end range_of_sum_l1376_137699


namespace correct_operation_l1376_137616

variables {x y : ℝ}

theorem correct_operation : -2 * x * 3 * y = -6 * x * y :=
by
  sorry

end correct_operation_l1376_137616


namespace students_without_pens_l1376_137695

theorem students_without_pens (total_students blue_pens red_pens both_pens : ℕ)
  (h_total : total_students = 40)
  (h_blue : blue_pens = 18)
  (h_red : red_pens = 26)
  (h_both : both_pens = 10) :
  total_students - (blue_pens + red_pens - both_pens) = 6 :=
by
  sorry

end students_without_pens_l1376_137695


namespace line_always_passes_through_fixed_point_l1376_137679

theorem line_always_passes_through_fixed_point :
  ∀ m : ℝ, (m-1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  
  -- Proof would go here
  sorry

end line_always_passes_through_fixed_point_l1376_137679


namespace angle_east_northwest_l1376_137637

def num_spokes : ℕ := 12
def central_angle : ℕ := 360 / num_spokes
def angle_between (start_dir end_dir : ℕ) : ℕ := (end_dir - start_dir) * central_angle

theorem angle_east_northwest : angle_between 3 9 = 90 := sorry

end angle_east_northwest_l1376_137637


namespace find_n_l1376_137613

theorem find_n 
  (num_engineers : ℕ) (num_technicians : ℕ) (num_workers : ℕ)
  (total_population : ℕ := num_engineers + num_technicians + num_workers)
  (systematic_sampling_inclusion_exclusion : ∀ n : ℕ, ∃ k : ℕ, n ∣ total_population ↔ n + 1 ≠ total_population) 
  (stratified_sampling_lcm : ∃ lcm : ℕ, lcm = Nat.lcm (Nat.lcm num_engineers num_technicians) num_workers)
  (total_population_is_36 : total_population = 36)
  (num_engineers_is_6 : num_engineers = 6)
  (num_technicians_is_12 : num_technicians = 12)
  (num_workers_is_18 : num_workers = 18) :
  ∃ n : ℕ, n = 6 :=
by
  sorry

end find_n_l1376_137613


namespace smallest_number_l1376_137696

-- Definitions of the numbers in their respective bases
def num1 := 5 * 9^0 + 8 * 9^1 -- 85_9
def num2 := 0 * 6^0 + 1 * 6^1 + 2 * 6^2 -- 210_6
def num3 := 0 * 4^0 + 0 * 4^1 + 0 * 4^2 + 1 * 4^3 -- 1000_4
def num4 := 1 * 2^0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3 + 1 * 2^4 + 1 * 2^5 -- 111111_2

-- Assert that num4 is the smallest
theorem smallest_number : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 :=
by 
  sorry

end smallest_number_l1376_137696


namespace mike_needs_more_money_l1376_137648

-- We define the conditions given in the problem.
def phone_cost : ℝ := 1300
def mike_fraction : ℝ := 0.40

-- Define the statement to be proven.
theorem mike_needs_more_money : (phone_cost - (mike_fraction * phone_cost) = 780) :=
by
  -- The proof steps would go here
  sorry

end mike_needs_more_money_l1376_137648


namespace find_M_plus_N_l1376_137672

theorem find_M_plus_N (M N : ℕ) (h1 : 3 / 5 = M / 30) (h2 : 3 / 5 = 90 / N) : M + N = 168 := 
by
  sorry

end find_M_plus_N_l1376_137672


namespace average_age_group_l1376_137602

theorem average_age_group (n : ℕ) (T : ℕ) (h1 : T = 15 * n) (h2 : T + 37 = 17 * (n + 1)) : n = 10 :=
by
  sorry

end average_age_group_l1376_137602


namespace find_L_l1376_137647

noncomputable def L_value : ℕ := 3

theorem find_L
  (a b : ℕ)
  (cows : ℕ := 5 * b)
  (chickens : ℕ := 5 * a + 7)
  (insects : ℕ := b ^ (a - 5))
  (legs_cows : ℕ := 4 * cows)
  (legs_chickens : ℕ := 2 * chickens)
  (legs_insects : ℕ :=  6 * insects)
  (total_legs : ℕ := legs_cows + legs_chickens + legs_insects) 
  (h1 : cows = insects)
  (h2 : total_legs = (L_value * 100 + L_value * 10 + L_value) + 1) :
  L_value = 3 := sorry

end find_L_l1376_137647


namespace range_of_a_l1376_137656

variable (a : ℝ)

def set_A (a : ℝ) : Set ℝ := { x | x^2 - 2 * x + a ≥ 0 }

theorem range_of_a (h : 1 ∉ set_A a) : a < 1 := 
by {
  sorry
}

end range_of_a_l1376_137656


namespace constant_function_of_horizontal_tangent_l1376_137609

theorem constant_function_of_horizontal_tangent (f : ℝ → ℝ) (h : ∀ x, deriv f x = 0) : ∃ c : ℝ, ∀ x, f x = c :=
sorry

end constant_function_of_horizontal_tangent_l1376_137609


namespace remainder_of_2x_plus_3uy_l1376_137678

theorem remainder_of_2x_plus_3uy (x y u v : ℤ) (hxy : x = u * y + v) (hv : 0 ≤ v) (hv_ub : v < y) :
  (if 2 * v < y then (2 * v % y) else ((2 * v % y) % -y % y)) = 
  (if 2 * v < y then 2 * v else 2 * v - y) :=
by {
  sorry
}

end remainder_of_2x_plus_3uy_l1376_137678


namespace m_greater_than_one_l1376_137664

variables {x m : ℝ}

def p : Prop := -2 ≤ x ∧ x ≤ 11
def q : Prop := 1 - 3 * m ≤ x ∧ x ≤ 3 + m

theorem m_greater_than_one (h : ¬(x^2 - 2 * x + m ≤ 0)) : m > 1 :=
sorry

end m_greater_than_one_l1376_137664


namespace prove_difference_l1376_137624

theorem prove_difference (x y : ℝ) (h1 : x + y = 500) (h2 : x * y = 22000) : y - x = -402.5 :=
sorry

end prove_difference_l1376_137624


namespace six_digit_product_of_consecutive_even_integers_l1376_137665

theorem six_digit_product_of_consecutive_even_integers :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ a * b * c = 287232 :=
sorry

end six_digit_product_of_consecutive_even_integers_l1376_137665


namespace inequality_problem_l1376_137694

open Real

theorem inequality_problem
  (a b c x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h_condition : 1 / x + 1 / y + 1 / z = 1) :
  a^x + b^y + c^z ≥ 4 * a * b * c * x * y * z / (x + y + z - 3) ^ 2 :=
by
  sorry

end inequality_problem_l1376_137694


namespace closest_to_one_tenth_l1376_137642

noncomputable def p (n : ℕ) : ℚ :=
  1 / (n * (n + 2)) + 1 / ((n + 2) * (n + 4)) + 1 / ((n + 4) * (n + 6)) +
  1 / ((n + 6) * (n + 8)) + 1 / ((n + 8) * (n + 10))

theorem closest_to_one_tenth {n : ℕ} (h₀ : 4 ≤ n ∧ n ≤ 7) : 
  |(5 : ℚ) / (n * (n + 10)) - 1 / 10| ≤ 
  |(5 : ℚ) / (4 * (4 + 10)) - 1 / 10| ∧ n = 4 := 
sorry

end closest_to_one_tenth_l1376_137642


namespace find_first_blend_price_l1376_137663

-- Define the conditions
def first_blend_price (x : ℝ) := x
def second_blend_price : ℝ := 8.00
def total_blend_weight : ℝ := 20
def total_blend_price_per_pound : ℝ := 8.40
def first_blend_weight : ℝ := 8
def second_blend_weight : ℝ := total_blend_weight - first_blend_weight

-- Define the cost calculations
def first_blend_total_cost (x : ℝ) := first_blend_weight * x
def second_blend_total_cost := second_blend_weight * second_blend_price
def total_blend_total_cost (x : ℝ) := first_blend_total_cost x + second_blend_total_cost

-- Prove that the price per pound of the first blend is $9.00
theorem find_first_blend_price : ∃ x : ℝ, total_blend_total_cost x = total_blend_weight * total_blend_price_per_pound ∧ x = 9 :=
by
  sorry

end find_first_blend_price_l1376_137663


namespace roy_is_6_years_older_than_julia_l1376_137639

theorem roy_is_6_years_older_than_julia :
  ∀ (R J K : ℕ) (x : ℕ), 
    R = J + x →
    R = K + x / 2 →
    R + 4 = 2 * (J + 4) →
    (R + 4) * (K + 4) = 108 →
    x = 6 :=
by
  intros R J K x h1 h2 h3 h4
  -- Proof goes here (using sorry to skip the proof)
  sorry

end roy_is_6_years_older_than_julia_l1376_137639


namespace value_of_m_l1376_137608

theorem value_of_m (m : ℝ) : (m + 1, 3) ∈ {p : ℝ × ℝ | p.1 + p.2 + 1 = 0} → m = -5 :=
by
  intro h
  sorry

end value_of_m_l1376_137608
