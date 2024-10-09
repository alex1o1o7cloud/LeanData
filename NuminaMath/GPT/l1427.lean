import Mathlib

namespace cotangent_positives_among_sequence_l1427_142726

def cotangent_positive_count (n : ℕ) : ℕ :=
  if n ≤ 2019 then
    let count := (n / 4) * 3 + if n % 4 ≠ 0 then (3 + 1 - max 0 ((n % 4) - 1)) else 0
    count
  else 0

theorem cotangent_positives_among_sequence :
  cotangent_positive_count 2019 = 1515 := sorry

end cotangent_positives_among_sequence_l1427_142726


namespace sum_geometric_seq_l1427_142700

theorem sum_geometric_seq (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1)
  (h2 : 4 * a 2 = 4 * a 1 + a 3)
  (h3 : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q)) :
  S 3 = 15 :=
by
  sorry

end sum_geometric_seq_l1427_142700


namespace angle_GDA_is_135_l1427_142759

-- Definitions for the geometric entities and conditions mentioned
structure Triangle :=
  (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ)

structure Square :=
  (angle : ℝ := 90)

def BCD : Triangle :=
  { angle_A := 45, angle_B := 45, angle_C := 90 }

def ABCD : Square :=
  {}

def DEFG : Square :=
  {}

-- The proof problem stated in Lean 4
theorem angle_GDA_is_135 :
  ∃ θ : ℝ, θ = 135 ∧ 
  (∀ (BCD : Triangle), BCD.angle_C = 90 ∧ BCD.angle_A = 45 ∧ BCD.angle_B = 45) ∧ 
  (∀ (Square : Square), Square.angle = 90) → 
  θ = 135 :=
by
  sorry

end angle_GDA_is_135_l1427_142759


namespace square_side_length_l1427_142783

theorem square_side_length (P : ℝ) (s : ℝ) (h1 : P = 36) (h2 : P = 4 * s) : s = 9 := 
by sorry

end square_side_length_l1427_142783


namespace correct_answer_l1427_142761

def A : Set ℝ := { x | x^2 + 2 * x - 3 > 0 }
def B : Set ℝ := { -1, 0, 1, 2 }

theorem correct_answer : A ∩ B = { 2 } :=
  sorry

end correct_answer_l1427_142761


namespace problem_statement_l1427_142704

-- Define the basic problem setup
def defect_rate (p : ℝ) := p = 0.01
def sample_size (n : ℕ) := n = 200

-- Define the binomial distribution
noncomputable def binomial_expectation (n : ℕ) (p : ℝ) := n * p
noncomputable def binomial_variance (n : ℕ) (p : ℝ) := n * p * (1 - p)

-- The actual statement that we will prove
theorem problem_statement (p : ℝ) (n : ℕ) (X : ℕ → ℕ) 
  (h_defect_rate : defect_rate p) 
  (h_sample_size : sample_size n) 
  (h_distribution : ∀ k, X k = (n.choose k) * (p ^ k) * ((1 - p) ^ (n - k))) 
  : binomial_expectation n p = 2 ∧ binomial_variance n p = 1.98 :=
by
  sorry

end problem_statement_l1427_142704


namespace sufficient_but_not_necessary_l1427_142767

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 0): (x = 1 → x > 0) ∧ ¬(x > 0 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_l1427_142767


namespace a_minus_b_l1427_142768

theorem a_minus_b (a b : ℚ) :
  (∀ x y, (x = 3 → y = 7) ∨ (x = 10 → y = 19) → y = a * x + b) →
  a - b = -(1/7) :=
by
  sorry

end a_minus_b_l1427_142768


namespace smallest_four_digit_equiv_8_mod_9_l1427_142765

theorem smallest_four_digit_equiv_8_mod_9 :
  ∃ n : ℕ, n % 9 = 8 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 9 = 8 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m :=
sorry

end smallest_four_digit_equiv_8_mod_9_l1427_142765


namespace car_speed_second_hour_l1427_142797

theorem car_speed_second_hour
  (S : ℕ)
  (first_hour_speed : ℕ := 98)
  (avg_speed : ℕ := 79)
  (total_time : ℕ := 2)
  (h_avg_speed : avg_speed = (first_hour_speed + S) / total_time) :
  S = 60 :=
by
  -- Proof steps omitted
  sorry

end car_speed_second_hour_l1427_142797


namespace refrigerator_cost_is_15000_l1427_142776

theorem refrigerator_cost_is_15000 (R : ℝ) 
  (phone_cost : ℝ := 8000)
  (phone_profit : ℝ := 0.10) 
  (fridge_loss : ℝ := 0.03) 
  (overall_profit : ℝ := 350) :
  (0.97 * R + phone_cost * (1 + phone_profit) = (R + phone_cost) + overall_profit) →
  (R = 15000) :=
by
  sorry

end refrigerator_cost_is_15000_l1427_142776


namespace eval_expression_l1427_142798

theorem eval_expression : 5 - 7 * (8 - 12 / 3^2) * 6 = -275 := by
  sorry

end eval_expression_l1427_142798


namespace fill_bathtub_with_drain_open_l1427_142793

theorem fill_bathtub_with_drain_open :
  let fill_rate := 1 / 10
  let drain_rate := 1 / 12
  let net_fill_rate := fill_rate - drain_rate
  fill_rate = 1 / 10 ∧ drain_rate = 1 / 12 → 1 / net_fill_rate = 60 :=
by
  intros
  sorry

end fill_bathtub_with_drain_open_l1427_142793


namespace certain_number_is_65_l1427_142792

-- Define the conditions
variables (N : ℕ)
axiom condition1 : N < 81
axiom condition2 : ∀ k : ℕ, k ≤ 15 → N + k < 81
axiom last_consecutive : N + 15 = 80

-- Prove the theorem
theorem certain_number_is_65 (h1 : N < 81) (h2 : ∀ k : ℕ, k ≤ 15 → N + k < 81) (h3 : N + 15 = 80) : N = 65 :=
sorry

end certain_number_is_65_l1427_142792


namespace no_real_solutions_l1427_142749

theorem no_real_solutions :
  ∀ x : ℝ, (2 * x - 6) ^ 2 + 4 ≠ -(x - 3) :=
by
  intro x
  sorry

end no_real_solutions_l1427_142749


namespace rate_per_kg_of_grapes_l1427_142712

theorem rate_per_kg_of_grapes : 
  ∀ (rate_per_kg_grapes : ℕ), 
    (10 * rate_per_kg_grapes + 9 * 55 = 1195) → 
    rate_per_kg_grapes = 70 := 
by
  intros rate_per_kg_grapes h
  sorry

end rate_per_kg_of_grapes_l1427_142712


namespace correct_sentence_is_D_l1427_142778

-- Define the sentences as strings
def sentence_A : String :=
  "Between any two adjacent integers on the number line, an infinite number of fractions can be inserted to fill the gaps on the number line; mathematicians once thought that with this approach, the entire number line was finally filled."

def sentence_B : String :=
  "With zero as the center, all integers are arranged from right to left at equal distances, and then connected with a horizontal line; this is what we call the 'number line'."

def sentence_C : String :=
  "The vast collection of books in the Beijing Library contains an enormous amount of information, but it is still finite, whereas the number pi contains infinite information, which is awe-inspiring."

def sentence_D : String :=
  "Pi is fundamentally the exact ratio of a circle's circumference to its diameter, but the infinite sequence it produces has the greatest uncertainty; we cannot help but be amazed and shaken by the marvel and mystery of nature."

-- Define the problem statement
theorem correct_sentence_is_D :
  sentence_D ≠ "" := by
  sorry

end correct_sentence_is_D_l1427_142778


namespace solve_for_x_l1427_142771

noncomputable def equation (x : ℝ) := (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2

theorem solve_for_x (h : ∀ x, x ≠ 3) : equation (-7 / 6) :=
by
  sorry

end solve_for_x_l1427_142771


namespace area_of_triangle_POF_l1427_142714

noncomputable def origin : (ℝ × ℝ) := (0, 0)
noncomputable def focus : (ℝ × ℝ) := (Real.sqrt 2, 0)

noncomputable def parabola (x y : ℝ) : Prop :=
  y ^ 2 = 4 * Real.sqrt 2 * x

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  parabola x y

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

noncomputable def PF_eq_4sqrt2 (x y : ℝ) : Prop :=
  distance x y (Real.sqrt 2) 0 = 4 * Real.sqrt 2

theorem area_of_triangle_POF (x y : ℝ) 
  (h1: point_on_parabola x y)
  (h2: PF_eq_4sqrt2 x y) :
   1 / 2 * distance 0 0 (Real.sqrt 2) 0 * |y| = 2 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_POF_l1427_142714


namespace mrs_hilt_remaining_cents_l1427_142799

-- Define the initial amount of money Mrs. Hilt had
def initial_cents : ℕ := 43

-- Define the cost of the pencil
def pencil_cost : ℕ := 20

-- Define the cost of the candy
def candy_cost : ℕ := 5

-- Define the remaining money Mrs. Hilt has after the purchases
def remaining_cents : ℕ := initial_cents - (pencil_cost + candy_cost)

-- Theorem statement to prove that the remaining amount is 18 cents
theorem mrs_hilt_remaining_cents : remaining_cents = 18 := by
  -- Proof omitted
  sorry

end mrs_hilt_remaining_cents_l1427_142799


namespace find_larger_number_l1427_142762

theorem find_larger_number (a b : ℕ) (h_diff : a - b = 3) (h_sum_squares : a^2 + b^2 = 117) (h_pos : 0 < a ∧ 0 < b) : a = 9 :=
by
  sorry

end find_larger_number_l1427_142762


namespace proof_OPQ_Constant_l1427_142750

open Complex

def OPQ_Constant :=
  ∀ (z1 z2 : ℂ) (θ : ℝ), abs z1 = 5 ∧
    (z1^2 - z1 * z2 * Real.sin θ + z2^2 = 0) →
      abs z2 = 5

theorem proof_OPQ_Constant : OPQ_Constant :=
by
  sorry

end proof_OPQ_Constant_l1427_142750


namespace find_difference_l1427_142752

variable (d : ℕ) (A B : ℕ)
open Nat

theorem find_difference (hd : d > 7)
  (hAB : d * A + B + d * A + A = d * d + 7 * d + 4)  (hA_gt_B : A > B):
  A - B = 3 :=
sorry

end find_difference_l1427_142752


namespace position_of_2019_in_splits_l1427_142774

def sum_of_consecutive_odds (n : ℕ) : ℕ :=
  n^2 - (n - 1)

theorem position_of_2019_in_splits : ∃ n : ℕ, sum_of_consecutive_odds n = 2019 ∧ n = 45 :=
by
  sorry

end position_of_2019_in_splits_l1427_142774


namespace christmas_bonus_remainder_l1427_142780

theorem christmas_bonus_remainder (B P R : ℕ) (hP : P = 8 * B + 5) (hR : (4 * P) % 8 = R) : R = 4 :=
by
  sorry

end christmas_bonus_remainder_l1427_142780


namespace solve_eq1_solve_eq2_l1427_142753

-- Define the first proof problem
theorem solve_eq1 (x : ℝ) : 2 * x - 3 = 3 * (x + 1) → x = -6 :=
by
  sorry

-- Define the second proof problem
theorem solve_eq2 (x : ℝ) : (1 / 2) * x - (9 * x - 2) / 6 - 2 = 0 → x = -5 / 3 :=
by
  sorry

end solve_eq1_solve_eq2_l1427_142753


namespace misha_total_shots_l1427_142775

theorem misha_total_shots (x y : ℕ) 
  (h1 : 18 * x + 5 * y = 99) 
  (h2 : 2 * x + y = 15) 
  (h3 : (15 / 0.9375 : ℝ) = 16) : 
  (¬(x = 0) ∧ ¬(y = 24)) ->
  16 = 16 :=
by
  sorry

end misha_total_shots_l1427_142775


namespace geometric_sequence_sum_l1427_142725

-- Let {a_n} be a geometric sequence such that S_2 = 7 and S_6 = 91. Prove that S_4 = 28

-- Define the sum of the first n terms of a geometric sequence
noncomputable def S (n : ℕ) (a1 r : ℝ) : ℝ := a1 * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (a1 r : ℝ) (h1 : S 2 a1 r = 7) (h2 : S 6 a1 r = 91) :
  S 4 a1 r = 28 := 
by 
  sorry

end geometric_sequence_sum_l1427_142725


namespace perfect_square_quotient_l1427_142736

theorem perfect_square_quotient (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h : (a * b + 1) ∣ (a * a + b * b)) : 
  ∃ k : ℕ, (a * a + b * b) = (a * b + 1) * (k * k) := 
sorry

end perfect_square_quotient_l1427_142736


namespace geometric_sequence_11th_term_l1427_142721

theorem geometric_sequence_11th_term (a r : ℕ) :
    a * r^4 = 3 →
    a * r^7 = 24 →
    a * r^10 = 192 := by
    sorry

end geometric_sequence_11th_term_l1427_142721


namespace arcsin_one_half_eq_pi_six_l1427_142732

theorem arcsin_one_half_eq_pi_six :
  Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l1427_142732


namespace janet_needs_9_dog_collars_l1427_142795

variable (D : ℕ)

theorem janet_needs_9_dog_collars (h1 : ∀ d : ℕ, d = 18)
  (h2 : ∀ c : ℕ, c = 10)
  (h3 : (18 * D) + (3 * 10) = 192) :
  D = 9 :=
by
  sorry

end janet_needs_9_dog_collars_l1427_142795


namespace moles_of_water_used_l1427_142763

-- Define the balanced chemical equation's molar ratios
def balanced_reaction (Li3N_moles : ℕ) (H2O_moles : ℕ) (LiOH_moles : ℕ) (NH3_moles : ℕ) : Prop :=
  Li3N_moles = 1 ∧ H2O_moles = 3 ∧ LiOH_moles = 3 ∧ NH3_moles = 1

-- Given 1 mole of lithium nitride and 3 moles of lithium hydroxide produced, 
-- prove that 3 moles of water were used.
theorem moles_of_water_used (Li3N_moles : ℕ) (LiOH_moles : ℕ) (H2O_moles : ℕ) :
  Li3N_moles = 1 → LiOH_moles = 3 → H2O_moles = 3 :=
by
  intros h1 h2
  sorry

end moles_of_water_used_l1427_142763


namespace problem_inequality_l1427_142787

theorem problem_inequality 
  (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) 
  (h8 : a + b + c + d ≥ 1) : 
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := 
sorry

end problem_inequality_l1427_142787


namespace probability_at_least_one_succeeds_l1427_142788

variable (p1 p2 : ℝ)

theorem probability_at_least_one_succeeds : 
  0 ≤ p1 ∧ p1 ≤ 1 → 0 ≤ p2 ∧ p2 ≤ 1 → (1 - (1 - p1) * (1 - p2)) = 1 - (1 - p1) * (1 - p2) :=
by 
  intro h1 h2
  sorry

end probability_at_least_one_succeeds_l1427_142788


namespace impossible_to_maintain_Gini_l1427_142754

variables (X Y G0 Y' Z : ℝ)
variables (G1 : ℝ)

-- Conditions
axiom initial_Gini : G0 = 0.1
axiom proportion_poor : X = 0.5
axiom income_poor_initial : Y = 0.4
axiom income_poor_half : Y' = 0.2
axiom population_split : ∀ a b c : ℝ, (a + b + c = 1) ∧ (a = b ∧ b = c)
axiom Gini_constant : G1 = G0

-- Equation system representation final value post situation
axiom Gini_post_reform : 
  G1 = (1 / 2 - ((1 / 6) * 0.2 + (1 / 6) * (0.2 + Z) + (1 / 6) * (1 - 0.2 - Z))) / (1 / 2)

-- Proof problem: to prove inconsistency or inability to maintain Gini coefficient given the conditions
theorem impossible_to_maintain_Gini : false :=
sorry

end impossible_to_maintain_Gini_l1427_142754


namespace count_rectangles_with_perimeter_twenty_two_l1427_142748

theorem count_rectangles_with_perimeter_twenty_two : 
  (∃! (n : ℕ), n = 11) :=
by
  sorry

end count_rectangles_with_perimeter_twenty_two_l1427_142748


namespace max_ab_is_5_l1427_142766

noncomputable def max_ab : ℝ :=
  sorry

theorem max_ab_is_5 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h : a / 4 + b / 5 = 1) : max_ab = 5 :=
  sorry

end max_ab_is_5_l1427_142766


namespace calculate_expression_l1427_142770

theorem calculate_expression : 2^3 * 2^3 + 2^3 = 72 := by
  sorry

end calculate_expression_l1427_142770


namespace pairs_of_boys_girls_l1427_142756

theorem pairs_of_boys_girls (a_g b_g a_b b_b : ℕ) 
  (h1 : a_b = 3 * a_g)
  (h2 : b_b = 4 * b_g) :
  ∃ c : ℕ, b_b = 7 * b_g :=
sorry

end pairs_of_boys_girls_l1427_142756


namespace ribbon_tape_remaining_l1427_142796

theorem ribbon_tape_remaining 
  (initial_length used_for_ribbon used_for_gift : ℝ)
  (h_initial: initial_length = 1.6)
  (h_ribbon: used_for_ribbon = 0.8)
  (h_gift: used_for_gift = 0.3) : 
  initial_length - used_for_ribbon - used_for_gift = 0.5 :=
by 
  sorry

end ribbon_tape_remaining_l1427_142796


namespace smallest_largest_number_in_list_l1427_142720

theorem smallest_largest_number_in_list :
  ∃ (a b c d e : ℕ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ 
  (a + b + c + d + e = 50) ∧ (e - a = 20) ∧ 
  (c = 6) ∧ (b = 6) ∧ 
  (e = 20) :=
by
  sorry

end smallest_largest_number_in_list_l1427_142720


namespace evaluate_expression_l1427_142715

noncomputable def lg (x : ℝ) : ℝ := Real.log x

theorem evaluate_expression :
  lg 5 * lg 50 - lg 2 * lg 20 - lg 625 = -2 :=
by
  sorry

end evaluate_expression_l1427_142715


namespace common_ratio_is_two_l1427_142746

-- Given a geometric sequence with specific terms
variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions: all terms are positive, a_2 = 3, a_6 = 48
axiom pos_terms : ∀ n, a n > 0
axiom a2_eq : a 2 = 3
axiom a6_eq : a 6 = 48

-- Question: Prove the common ratio q is 2
theorem common_ratio_is_two :
  (∀ n, a n = a 1 * q ^ (n - 1)) → q = 2 :=
by
  sorry

end common_ratio_is_two_l1427_142746


namespace original_number_increased_by_40_percent_l1427_142724

theorem original_number_increased_by_40_percent (x : ℝ) (h : 1.40 * x = 700) : x = 500 :=
by
  sorry

end original_number_increased_by_40_percent_l1427_142724


namespace snack_eaters_left_after_second_newcomers_l1427_142789

theorem snack_eaters_left_after_second_newcomers
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (half_left_1 : ℕ)
  (new_outsiders_2 : ℕ)
  (final_snackers : ℕ)
  (H1 : initial_snackers = 100)
  (H2 : new_outsiders_1 = 20)
  (H3 : half_left_1 = (initial_snackers + new_outsiders_1) / 2)
  (H4 : new_outsiders_2 = 10)
  (H5 : final_snackers = 20)
  : (initial_snackers + new_outsiders_1 - half_left_1 + new_outsiders_2 - (initial_snackers + new_outsiders_1 - half_left_1 + new_outsiders_2 - final_snackers * 2)) = 30 :=
by 
  sorry

end snack_eaters_left_after_second_newcomers_l1427_142789


namespace min_value_f_l1427_142785

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (4 / (1 - 2 * x))

theorem min_value_f : ∃ (x : ℝ), (0 < x ∧ x < 1 / 2) ∧ f x = 6 + 4 * sqrt 2 := by
  sorry

end min_value_f_l1427_142785


namespace integer_solutions_to_system_l1427_142727

theorem integer_solutions_to_system (x y z : ℤ) (h1 : x + y + z = 2) (h2 : x^3 + y^3 + z^3 = -10) :
  (x = 3 ∧ y = 3 ∧ z = -4) ∨
  (x = 3 ∧ y = -4 ∧ z = 3) ∨
  (x = -4 ∧ y = 3 ∧ z = 3) :=
sorry

end integer_solutions_to_system_l1427_142727


namespace candy_problem_l1427_142707

theorem candy_problem
  (x y m : ℤ)
  (hx : x ≥ 0)
  (hy : y ≥ 0)
  (hxy : x + y = 176)
  (hcond : x - m * (y - 16) = 47)
  (hm : m > 1) :
  x ≥ 131 := 
sorry

end candy_problem_l1427_142707


namespace min_value_xy_l1427_142790

theorem min_value_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : x * y ≥ 18 := 
sorry

end min_value_xy_l1427_142790


namespace ordering_l1427_142784

noncomputable def a : ℝ := 1 / (Real.exp 0.6)
noncomputable def b : ℝ := 0.4
noncomputable def c : ℝ := Real.log 1.4 / 1.4

theorem ordering : a > b ∧ b > c :=
by
  have ha : a = 1 / (Real.exp 0.6) := rfl
  have hb : b = 0.4 := rfl
  have hc : c = Real.log 1.4 / 1.4 := rfl
  sorry

end ordering_l1427_142784


namespace integer_solution_count_l1427_142710

theorem integer_solution_count (x : ℤ) : (12 * x - 1) * (6 * x - 1) * (4 * x - 1) * (3 * x - 1) = 330 ↔ x = 1 :=
by
  sorry

end integer_solution_count_l1427_142710


namespace no_solution_iff_a_leq_8_l1427_142734

theorem no_solution_iff_a_leq_8 (a : ℝ) :
  (¬ ∃ x : ℝ, |x - 5| + |x + 3| < a) ↔ a ≤ 8 := 
sorry

end no_solution_iff_a_leq_8_l1427_142734


namespace money_made_arkansas_game_is_8722_l1427_142779

def price_per_tshirt : ℕ := 98
def tshirts_sold_arkansas_game : ℕ := 89
def total_money_made_arkansas_game (price_per_tshirt tshirts_sold_arkansas_game : ℕ) : ℕ :=
  price_per_tshirt * tshirts_sold_arkansas_game

theorem money_made_arkansas_game_is_8722 :
  total_money_made_arkansas_game price_per_tshirt tshirts_sold_arkansas_game = 8722 :=
by
  sorry

end money_made_arkansas_game_is_8722_l1427_142779


namespace emily_can_see_emerson_l1427_142791

theorem emily_can_see_emerson : 
  ∀ (emily_speed emerson_speed : ℝ) 
    (initial_distance final_distance : ℝ), 
  emily_speed = 15 → 
  emerson_speed = 9 → 
  initial_distance = 1 → 
  final_distance = 1 →
  (initial_distance / (emily_speed - emerson_speed) + final_distance / (emily_speed - emerson_speed)) * 60 = 20 :=
by
  intros emily_speed emerson_speed initial_distance final_distance
  sorry

end emily_can_see_emerson_l1427_142791


namespace cos_neg_570_eq_neg_sqrt3_div_2_l1427_142740

theorem cos_neg_570_eq_neg_sqrt3_div_2 :
  Real.cos (-(570 : ℝ) * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_neg_570_eq_neg_sqrt3_div_2_l1427_142740


namespace cost_price_l1427_142758

theorem cost_price (SP MP CP : ℝ) (discount_rate : ℝ) 
  (h1 : MP = CP * 1.15)
  (h2 : SP = MP * (1 - discount_rate))
  (h3 : SP = 459)
  (h4 : discount_rate = 0.2608695652173913) : CP = 540 :=
by
  -- We use the hints given as conditions to derive the statement
  sorry

end cost_price_l1427_142758


namespace determine_radii_l1427_142735

-- Definitions based on conditions from a)
variable (S1 S2 S3 S4 : Type) -- Centers of the circles
variable (dist_S2_S4 : ℝ) (dist_S1_S2 : ℝ) (dist_S2_S3 : ℝ) (dist_S3_S4 : ℝ)
variable (r1 r2 r3 r4 : ℝ) -- Radii of circles k1, k2, k3, and k4
variable (rhombus : Prop) -- Quadrilateral S1S2S3S4 is a rhombus

-- Given conditions
axiom C1 : ∀ t : S1, r1 = 5
axiom C2 : dist_S2_S4 = 24
axiom C3 : rhombus

-- Equivalency to be proven
theorem determine_radii : 
  r2 = 12 ∧ r4 = 12 ∧ r1 = 5 ∧ r3 = 5 :=
sorry

end determine_radii_l1427_142735


namespace zeros_in_Q_l1427_142739

def R_k (k : ℕ) : ℤ := (7^k - 1) / 6

def Q : ℤ := (7^30 - 1) / (7^6 - 1)

def count_zeros (n : ℤ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 470588 :=
by sorry

end zeros_in_Q_l1427_142739


namespace sum_product_smallest_number_l1427_142745

theorem sum_product_smallest_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : min x y = 8 :=
  sorry

end sum_product_smallest_number_l1427_142745


namespace surprise_shop_daily_revenue_l1427_142772

def closed_days_per_year : ℕ := 3
def years_active : ℕ := 6
def total_revenue_lost : ℚ := 90000

def total_closed_days : ℕ :=
  closed_days_per_year * years_active

def daily_revenue : ℚ :=
  total_revenue_lost / total_closed_days

theorem surprise_shop_daily_revenue :
  daily_revenue = 5000 := by
  sorry

end surprise_shop_daily_revenue_l1427_142772


namespace corrected_mean_l1427_142738

open Real

theorem corrected_mean (n : ℕ) (mu_incorrect : ℝ)
                      (x1 y1 x2 y2 x3 y3 : ℝ)
                      (h1 : mu_incorrect = 41)
                      (h2 : n = 50)
                      (h3 : x1 = 48 ∧ y1 = 23)
                      (h4 : x2 = 36 ∧ y2 = 42)
                      (h5 : x3 = 55 ∧ y3 = 28) :
                      ((mu_incorrect * n + (x1 - y1) + (x2 - y2) + (x3 - y3)) / n = 41.92) :=
by
  sorry

end corrected_mean_l1427_142738


namespace sum_of_x_coordinates_l1427_142705

def line1 (x : ℝ) : ℝ := -3 * x - 5
def line2 (x : ℝ) : ℝ := 2 * x - 3

def has_x_intersect (line : ℝ → ℝ) (y : ℝ) : Prop := ∃ x : ℝ, line x = y

theorem sum_of_x_coordinates :
  (∃ x1 x2 : ℝ, line1 x1 = 2.2 ∧ line2 x2 = 2.2 ∧ x1 + x2 = 0.2) :=
  sorry

end sum_of_x_coordinates_l1427_142705


namespace contrapositive_proposition_l1427_142755

theorem contrapositive_proposition (x a b : ℝ) : (x < 2 * a * b) → (x < a^2 + b^2) :=
sorry

end contrapositive_proposition_l1427_142755


namespace picked_clovers_when_one_four_found_l1427_142728

-- Definition of conditions
def total_leaves : ℕ := 100
def leaves_three_leaved_clover : ℕ := 3
def leaves_four_leaved_clover : ℕ := 4
def one_four_leaved_clover : ℕ := 1

-- Proof Statement
theorem picked_clovers_when_one_four_found (three_leaved_count : ℕ) :
  (total_leaves - leaves_four_leaved_clover) / leaves_three_leaved_clover = three_leaved_count → 
  three_leaved_count = 32 :=
by
  sorry

end picked_clovers_when_one_four_found_l1427_142728


namespace train_speed_l1427_142709

theorem train_speed (v t : ℝ) (h1 : 16 * t + v * t = 444) (h2 : v * t = 16 * t + 60) : v = 21 := 
sorry

end train_speed_l1427_142709


namespace trig_identity_l1427_142711

variable {α : Real}

theorem trig_identity (h : Real.tan α = 3) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := 
by
  sorry

end trig_identity_l1427_142711


namespace problem1_problem2a_problem2b_l1427_142702

-- Problem 1: Deriving y in terms of x
theorem problem1 (x y : ℕ) (h1 : 30 * x + 10 * y = 2000) : y = 200 - 3 * x :=
by sorry

-- Problem 2(a): Minimum ingredient B for at least 220 yuan profit with a=3
theorem problem2a (x y a w : ℕ) (h1 : a = 3) 
  (h2 : 3 * x + 2 * y ≥ 220) (h3 : y = 200 - 3 * x) 
  (h4 : w = 15 * x + 20 * y) : w = 1300 :=
by sorry

-- Problem 2(b): Profit per portion of dessert A for 450 yuan profit with 3100 grams of B
theorem problem2b (x : ℕ) (a : ℕ) (B : ℕ) 
  (h1 : B = 3100) (h2 : 15 * x + 20 * (200 - 3 * x) ≤ B) 
  (h3 : a * x + 2 * (200 - 3 * x) = 450) 
  (h4 : x ≥ 20) : a = 8 :=
by sorry

end problem1_problem2a_problem2b_l1427_142702


namespace day_care_center_toddlers_l1427_142782

theorem day_care_center_toddlers (I T : ℕ) (h_ratio1 : 7 * I = 3 * T) (h_ratio2 : 7 * (I + 12) = 5 * T) :
  T = 42 :=
by
  sorry

end day_care_center_toddlers_l1427_142782


namespace Phil_quarters_l1427_142760

theorem Phil_quarters (initial_amount : ℝ)
  (pizza : ℝ) (soda : ℝ) (jeans : ℝ) (book : ℝ) (gum : ℝ) (ticket : ℝ)
  (quarter_value : ℝ) (spent := pizza + soda + jeans + book + gum + ticket)
  (remaining := initial_amount - spent)
  (quarters := remaining / quarter_value) :
  initial_amount = 40 ∧ pizza = 2.75 ∧ soda = 1.50 ∧ jeans = 11.50 ∧
  book = 6.25 ∧ gum = 1.75 ∧ ticket = 8.50 ∧ quarter_value = 0.25 →
  quarters = 31 :=
by
  intros
  sorry

end Phil_quarters_l1427_142760


namespace translated_line_value_m_l1427_142723

theorem translated_line_value_m :
  (∀ x y : ℝ, (y = x → y = x + 3) → y = 2 + 3 → ∃ m : ℝ, y = m) :=
by sorry

end translated_line_value_m_l1427_142723


namespace find_angle_A_find_AB_l1427_142747

theorem find_angle_A (A B C : ℝ) (h1 : 2 * Real.sin B * Real.cos A = Real.sin (A + C)) (h2 : A + B + C = Real.pi) :
  A = Real.pi / 3 := by
  sorry

theorem find_AB (A B C : ℝ) (AB BC AC : ℝ) (h1 : 2 * Real.sin B * Real.cos A = Real.sin (A + C))
  (h2 : BC = 2) (h3 : 1 / 2 * AB * AC * Real.sin (Real.pi / 3) = Real.sqrt 3)
  (h4 : A = Real.pi / 3) :
  AB = 2 := by
  sorry

end find_angle_A_find_AB_l1427_142747


namespace total_avg_donation_per_person_l1427_142741

-- Definition of variables and conditions
variables (avgA avgB : ℝ) (numA numB : ℕ)
variables (h1 : avgB = avgA - 100)
variables (h2 : 2 * numA * avgA = 4 * numB * (avgA - 100))
variables (h3 : numA = numB / 4)

-- Lean 4 statement to prove the total average donation per person is 120
theorem total_avg_donation_per_person (h1 :  avgB = avgA - 100)
    (h2 : 2 * numA * avgA = 4 * numB * (avgA - 100))
    (h3 : numA = numB / 4) : 
    ( (numA * avgA + numB * avgB) / (numA + numB) ) = 120 :=
sorry

end total_avg_donation_per_person_l1427_142741


namespace find_first_month_sales_l1427_142706

noncomputable def avg_sales (sales_1 sales_2 sales_3 sales_4 sales_5 sales_6 : ℕ) : ℕ :=
(sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / 6

theorem find_first_month_sales :
  let sales_2 := 6927
  let sales_3 := 6855
  let sales_4 := 7230
  let sales_5 := 6562
  let sales_6 := 5091
  let avg_sales_needed := 6500
  ∃ sales_1, avg_sales sales_1 sales_2 sales_3 sales_4 sales_5 sales_6 = avg_sales_needed := 
by
  sorry

end find_first_month_sales_l1427_142706


namespace solve_for_x_l1427_142703

theorem solve_for_x :
  ∃ x : ℕ, (12 ^ 3) * (6 ^ x) / 432 = 144 ∧ x = 2 := by
  sorry

end solve_for_x_l1427_142703


namespace value_of_expression_l1427_142786

theorem value_of_expression :
  (10^2 - 10) / 9 = 10 :=
by
  sorry

end value_of_expression_l1427_142786


namespace minimum_n_for_all_columns_l1427_142701

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Function to check if a given number covers all columns from 0 to 9
def covers_all_columns (n : ℕ) : Bool :=
  let columns := (List.range n).map (λ i => triangular_number i % 10)
  List.range 10 |>.all (λ c => c ∈ columns)

theorem minimum_n_for_all_columns : ∃ n, covers_all_columns n ∧ triangular_number n = 253 :=
by 
  sorry

end minimum_n_for_all_columns_l1427_142701


namespace ten_years_less_average_age_l1427_142781

-- Defining the conditions formally
def lukeAge : ℕ := 20
def mrBernardAgeInEightYears : ℕ := 3 * lukeAge

-- Lean statement to prove the problem
theorem ten_years_less_average_age : 
  mrBernardAgeInEightYears - 8 = 52 → (lukeAge + (mrBernardAgeInEightYears - 8)) / 2 - 10 = 26 := 
by
  intros h
  sorry

end ten_years_less_average_age_l1427_142781


namespace lending_period_C_l1427_142716

theorem lending_period_C (P_B P_C : ℝ) (R : ℝ) (T_B I_total : ℝ) (T_C_months : ℝ) :
  P_B = 5000 ∧ P_C = 3000 ∧ R = 0.10 ∧ T_B = 2 ∧ I_total = 2200 ∧ 
  T_C_months = (2 / 3) * 12 → T_C_months = 8 := by
  intros h
  sorry

end lending_period_C_l1427_142716


namespace relationship_of_exponents_l1427_142743

theorem relationship_of_exponents (m p r s : ℝ) (u v w t : ℝ) (h1 : m^u = r) (h2 : p^v = r) (h3 : p^w = s) (h4 : m^t = s) : u * v = w * t :=
by
  sorry

end relationship_of_exponents_l1427_142743


namespace correct_equation_l1427_142722

theorem correct_equation (x y a b : ℝ) :
  ¬ (-(x - 6) = -x - 6) ∧
  ¬ (-y^2 - y^2 = 0) ∧
  ¬ (9 * a^2 * b - 9 * a * b^2 = 0) ∧
  (-9 * y^2 + 16 * y^2 = 7 * y^2) :=
by
  sorry

end correct_equation_l1427_142722


namespace percentage_class_takes_lunch_l1427_142731

theorem percentage_class_takes_lunch (total_students boys girls : ℕ)
  (h_total: total_students = 100)
  (h_ratio: boys = 6 * total_students / (6 + 4))
  (h_girls: girls = 4 * total_students / (6 + 4))
  (boys_lunch_ratio : ℝ)
  (girls_lunch_ratio : ℝ)
  (h_boys_lunch_ratio : boys_lunch_ratio = 0.60)
  (h_girls_lunch_ratio : girls_lunch_ratio = 0.40):
  ((boys_lunch_ratio * boys + girls_lunch_ratio * girls) / total_students) * 100 = 52 :=
by
  sorry

end percentage_class_takes_lunch_l1427_142731


namespace gcd_lcm_product_l1427_142742

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 45) : (Int.gcd a b * Nat.lcm a b) = 1080 := by
  rw [h1, h2]
  sorry

end gcd_lcm_product_l1427_142742


namespace color_fig_l1427_142733

noncomputable def total_colorings (dots : Finset (Fin 9)) (colors : Finset (Fin 4))
  (adj : dots → dots → Prop)
  (diag : dots → dots → Prop) : Nat :=
  -- coloring left triangle
  let left_triangle := 4 * 3 * 2;
  -- coloring middle triangle considering diagonal restrictions
  let middle_triangle := 3 * 2;
  -- coloring right triangle considering same restrictions
  let right_triangle := 3 * 2;
  left_triangle * middle_triangle * middle_triangle

theorem color_fig (dots : Finset (Fin 9)) (colors : Finset (Fin 4))
  (adj : dots → dots → Prop)
  (diag : dots → dots → Prop) :
  total_colorings dots colors adj diag = 864 :=
by
  sorry

end color_fig_l1427_142733


namespace smallest_n_divisible_by_2016_smallest_n_divisible_by_2016_pow_10_l1427_142764

-- Problem (a): Smallest n such that n! is divisible by 2016
theorem smallest_n_divisible_by_2016 : ∃ (n : ℕ), n = 8 ∧ 2016 ∣ n.factorial :=
by
  sorry

-- Problem (b): Smallest n such that n! is divisible by 2016^10
theorem smallest_n_divisible_by_2016_pow_10 : ∃ (n : ℕ), n = 63 ∧ 2016^10 ∣ n.factorial :=
by
  sorry

end smallest_n_divisible_by_2016_smallest_n_divisible_by_2016_pow_10_l1427_142764


namespace intersection_point_a_l1427_142777

-- Definitions for the given conditions 
def f (x : ℤ) (b : ℤ) : ℤ := 3 * x + b
def f_inv (x : ℤ) (b : ℤ) : ℤ := (x - b) / 3 -- Considering that f is invertible for integer b

-- The problem statement
theorem intersection_point_a (a b : ℤ) (h1 : a = f (-3) b) (h2 : a = f_inv (-3)) (h3 : f (-3) b = -3):
  a = -3 := sorry

end intersection_point_a_l1427_142777


namespace tim_income_percentage_less_l1427_142737

theorem tim_income_percentage_less (M T J : ℝ)
  (h₁ : M = 1.60 * T)
  (h₂ : M = 0.96 * J) :
  100 - (T / J) * 100 = 40 :=
by sorry

end tim_income_percentage_less_l1427_142737


namespace solve_for_x_l1427_142729

theorem solve_for_x (x : ℝ) (h : 7 - 2 * x = -3) : x = 5 := by
  sorry

end solve_for_x_l1427_142729


namespace graveling_cost_is_correct_l1427_142769

noncomputable def cost_of_graveling (lawn_length : ℕ) (lawn_breadth : ℕ) 
(road_width : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  let area_road_parallel_to_length := road_width * lawn_breadth
  let area_road_parallel_to_breadth := road_width * lawn_length
  let area_overlap := road_width * road_width
  let total_area := area_road_parallel_to_length + area_road_parallel_to_breadth - area_overlap
  total_area * cost_per_sq_m

theorem graveling_cost_is_correct : cost_of_graveling 90 60 10 3 = 4200 := by
  sorry

end graveling_cost_is_correct_l1427_142769


namespace find_m_l1427_142719

theorem find_m (m : ℝ) (h1 : (∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 12))) (h2 : m ≠ 0) : m = 12 :=
by
  sorry

end find_m_l1427_142719


namespace derivative_y_over_x_l1427_142730

noncomputable def x (t : ℝ) : ℝ := (t^2 * Real.log t) / (1 - t^2) + Real.log (Real.sqrt (1 - t^2))
noncomputable def y (t : ℝ) : ℝ := (t / Real.sqrt (1 - t^2)) * Real.arcsin t + Real.log (Real.sqrt (1 - t^2))

theorem derivative_y_over_x (t : ℝ) (ht : t ≠ 0) (h1 : t ≠ 1) (hneg1 : t ≠ -1) : 
  (deriv y t) / (deriv x t) = (Real.arcsin t * Real.sqrt (1 - t^2)) / (2 * t * Real.log t) :=
by
  sorry

end derivative_y_over_x_l1427_142730


namespace eval_expression_l1427_142744

theorem eval_expression : 
  (520 * 0.43 / 0.26 - 217 * (2 + 3/7)) - (31.5 / (12 + 3/5) + 114 * (2 + 1/3) + (61 + 1/2)) = 0.5 := 
by
  sorry

end eval_expression_l1427_142744


namespace unclaimed_candy_fraction_l1427_142718

-- Definitions for the shares taken by each person.
def al_share (x : ℕ) : ℚ := 3 / 7 * x
def bert_share (x : ℕ) : ℚ := 2 / 7 * (x - al_share x)
def carl_share (x : ℕ) : ℚ := 1 / 7 * ((x - al_share x) - bert_share x)
def dana_share (x : ℕ) : ℚ := 1 / 7 * (((x - al_share x) - bert_share x) - carl_share x)

-- The amount of candy that goes unclaimed.
def remaining_candy (x : ℕ) : ℚ := x - (al_share x + bert_share x + carl_share x + dana_share x)

-- The theorem we want to prove.
theorem unclaimed_candy_fraction (x : ℕ) : remaining_candy x / x = 584 / 2401 :=
by
  sorry

end unclaimed_candy_fraction_l1427_142718


namespace number_of_exercise_books_l1427_142708

theorem number_of_exercise_books (pencils pens exercise_books : ℕ) (h_ratio : (14 * pens = 4 * pencils) ∧ (14 * exercise_books = 3 * pencils)) (h_pencils : pencils = 140) : exercise_books = 30 :=
by
  sorry

end number_of_exercise_books_l1427_142708


namespace find_correct_r_l1427_142713

noncomputable def ellipse_tangent_circle_intersection : Prop :=
  ∃ (E F : ℝ × ℝ) (r : ℝ), E ∈ { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1 } ∧
                             F ∈ { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1 } ∧ 
                             (E ≠ F) ∧
                             ((E.1 - 2)^2 + (E.2 - 3/2)^2 = r^2) ∧
                             ((F.1 - 2)^2 + (F.2 - 3/2)^2 = r^2) ∧
                             r = (Real.sqrt 37) / 37

theorem find_correct_r : ellipse_tangent_circle_intersection :=
sorry

end find_correct_r_l1427_142713


namespace correct_operation_l1427_142717

theorem correct_operation (a : ℝ) (h : a ≠ 0) : a * a⁻¹ = 1 :=
by
  sorry

end correct_operation_l1427_142717


namespace football_starting_lineup_count_l1427_142773

variable (n_team_members n_offensive_linemen : ℕ)
variable (H_team_members : 12 = n_team_members)
variable (H_offensive_linemen : 5 = n_offensive_linemen)

theorem football_starting_lineup_count :
  n_team_members = 12 → n_offensive_linemen = 5 →
  (n_offensive_linemen * (n_team_members - 1) * (n_team_members - 2) * ((n_team_members - 3) * (n_team_members - 4) / 2)) = 19800 := 
by
  intros
  sorry

end football_starting_lineup_count_l1427_142773


namespace tangent_line_at_point_l1427_142757

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then (Real.exp (-(x - 1)) - x) else (Real.exp (x - 1) + x)

theorem tangent_line_at_point (f_even : ∀ x : ℝ, f x = f (-x)) :
    ∀ (x y : ℝ), x = 1 → y = 2 → (∃ m b : ℝ, y = m * x + b ∧ m = 2 ∧ b = 0) := by
  sorry

end tangent_line_at_point_l1427_142757


namespace vertical_asymptote_at_9_over_4_l1427_142751

def vertical_asymptote (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ x', x' ≠ x → abs (x' - x) < δ → abs (y x') > ε)

noncomputable def function_y (x : ℝ) : ℝ :=
  (2 * x + 3) / (4 * x - 9)

theorem vertical_asymptote_at_9_over_4 :
  vertical_asymptote function_y (9 / 4) :=
sorry

end vertical_asymptote_at_9_over_4_l1427_142751


namespace min_value_expr_l1427_142794

theorem min_value_expr (a b : ℝ) (h : a * b > 0) : (a^4 + 4 * b^4 + 1) / (a * b) ≥ 4 := 
sorry

end min_value_expr_l1427_142794
