import Mathlib

namespace part1_part2_l8_862

-- Definitions of sets A and B
def A (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < a + 1 }
def B : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 ≥ 0 }

-- Proving the first condition
theorem part1 (a : ℝ) : (A a ∩ B = ∅) ∧ (A a ∪ B = Set.univ) ↔ a = 2 :=
by
  sorry

-- Proving the second condition
theorem part2 (a : ℝ) : (A a ⊆ B) ↔ (a ≤ 0 ∨ a ≥ 4) :=
by
  sorry

end part1_part2_l8_862


namespace option_d_is_pythagorean_triple_l8_888

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem option_d_is_pythagorean_triple : is_pythagorean_triple 5 12 13 :=
by
  -- This will be the proof part, which is omitted as per the problem's instructions.
  sorry

end option_d_is_pythagorean_triple_l8_888


namespace three_colors_sufficient_l8_807

-- Definition of the tessellation problem with specified conditions.
def tessellation (n : ℕ) (x_divisions : ℕ) (y_divisions : ℕ) : Prop :=
  n = 8 ∧ x_divisions = 2 ∧ y_divisions = 2

-- Definition of the adjacency property.
def no_adjacent_same_color {α : Type} (coloring : ℕ → ℕ → α) : Prop :=
  ∀ (i j : ℕ), i < 8 → j < 8 →
  (i > 0 → coloring i j ≠ coloring (i-1) j) ∧ 
  (j > 0 → coloring i j ≠ coloring i (j-1)) ∧
  (i < 7 → coloring i j ≠ coloring (i+1) j) ∧ 
  (j < 7 → coloring i j ≠ coloring i (j+1)) ∧
  (i > 0 ∧ j > 0 → coloring i j ≠ coloring (i-1) (j-1)) ∧
  (i < 7 ∧ j < 7 → coloring i j ≠ coloring (i+1) (j+1)) ∧
  (i > 0 ∧ j < 7 → coloring i j ≠ coloring (i-1) (j+1)) ∧
  (i < 7 ∧ j > 0 → coloring i j ≠ coloring (i+1) (j-1))

-- The main theorem that needs to be proved.
theorem three_colors_sufficient : ∃ (k : ℕ) (coloring : ℕ → ℕ → ℕ), k = 3 ∧ 
  tessellation 8 2 2 ∧ 
  no_adjacent_same_color coloring := by
  sorry 

end three_colors_sufficient_l8_807


namespace fraction_of_b_equals_4_15_of_a_is_0_4_l8_878

variable (A B : ℤ)
variable (X : ℚ)

def a_and_b_together_have_1210 : Prop := A + B = 1210
def b_has_484 : Prop := B = 484
def fraction_of_b_equals_4_15_of_a : Prop := (4 / 15 : ℚ) * A = X * B

theorem fraction_of_b_equals_4_15_of_a_is_0_4
  (h1 : a_and_b_together_have_1210 A B)
  (h2 : b_has_484 B)
  (h3 : fraction_of_b_equals_4_15_of_a A B X) :
  X = 0.4 := sorry

end fraction_of_b_equals_4_15_of_a_is_0_4_l8_878


namespace minute_hand_moves_180_degrees_l8_825

noncomputable def minute_hand_angle_6_15_to_6_45 : ℝ :=
  let degrees_per_hour := 360
  let hours_period := 0.5
  degrees_per_hour * hours_period

theorem minute_hand_moves_180_degrees :
  minute_hand_angle_6_15_to_6_45 = 180 :=
by
  sorry

end minute_hand_moves_180_degrees_l8_825


namespace number_of_students_in_class_l8_887

theorem number_of_students_in_class :
  ∃ n : ℕ, n > 0 ∧ (∀ avg_age teacher_age total_avg_age, avg_age = 26 ∧ teacher_age = 52 ∧ total_avg_age = 27 →
    (∃ total_student_age total_age_with_teacher, 
      total_student_age = n * avg_age ∧ 
      total_age_with_teacher = total_student_age + teacher_age ∧ 
      (total_age_with_teacher / (n + 1) = total_avg_age) → n = 25)) :=
sorry

end number_of_students_in_class_l8_887


namespace max_vec_diff_magnitude_l8_897

open Real

noncomputable def vec_a (θ : ℝ) : ℝ × ℝ := (1, sin θ)
noncomputable def vec_b (θ : ℝ) : ℝ × ℝ := (1, cos θ)

noncomputable def vec_diff_magnitude (θ : ℝ) : ℝ :=
  let a := vec_a θ
  let b := vec_b θ
  abs ((a.1 - b.1)^2 + (a.2 - b.2)^2)^(1/2)

theorem max_vec_diff_magnitude : ∀ θ : ℝ, vec_diff_magnitude θ ≤ sqrt 2 :=
by
  intro θ
  sorry

end max_vec_diff_magnitude_l8_897


namespace calculate_revolutions_l8_895

noncomputable def number_of_revolutions (diameter distance: ℝ) : ℝ :=
  distance / (Real.pi * diameter)

theorem calculate_revolutions :
  number_of_revolutions 10 5280 = 528 / Real.pi :=
by
  sorry

end calculate_revolutions_l8_895


namespace ben_examined_7_trays_l8_858

open Int

def trays_of_eggs (total_eggs : ℕ) (eggs_per_tray : ℕ) : ℕ := total_eggs / eggs_per_tray

theorem ben_examined_7_trays : trays_of_eggs 70 10 = 7 :=
by
  sorry

end ben_examined_7_trays_l8_858


namespace find_f2_plus_fneg2_l8_805

def f (x a: ℝ) := (x + a)^3

theorem find_f2_plus_fneg2 (a : ℝ)
  (h_cond : ∀ x : ℝ, f (1 + x) a = -f (1 - x) a) :
  f 2 (-1) + f (-2) (-1) = -26 :=
by
  sorry

end find_f2_plus_fneg2_l8_805


namespace range_of_a_l8_836

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * x - 1) / (x - 1) < 0 ↔ x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) → 
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  intro h
  sorry

end range_of_a_l8_836


namespace tap_B_filling_time_l8_819

theorem tap_B_filling_time : 
  ∀ (r_A r_B : ℝ), 
  (r_A + r_B = 1 / 30) → 
  (r_B * 40 = 2 / 3) → 
  (1 / r_B = 60) := 
by
  intros r_A r_B h₁ h₂
  sorry

end tap_B_filling_time_l8_819


namespace false_statement_l8_843

noncomputable def heartsuit (x y : ℝ) := abs (x - y)
noncomputable def diamondsuit (z w : ℝ) := (z + w) ^ 2

theorem false_statement : ∃ (x y : ℝ), (heartsuit x y) ^ 2 ≠ diamondsuit x y := by
  sorry

end false_statement_l8_843


namespace GregPPO_reward_correct_l8_859

-- Define the maximum ProcGen reward
def maxProcGenReward : ℕ := 240

-- Define the maximum CoinRun reward in the more challenging version
def maxCoinRunReward : ℕ := maxProcGenReward / 2

-- Define the percentage reward obtained by Greg's PPO algorithm
def percentageRewardObtained : ℝ := 0.9

-- Calculate the reward obtained by Greg's PPO algorithm
def rewardGregPPO : ℝ := percentageRewardObtained * maxCoinRunReward

-- The theorem to prove the correct answer
theorem GregPPO_reward_correct : rewardGregPPO = 108 := by
  sorry

end GregPPO_reward_correct_l8_859


namespace hyperbola_asymptote_slope_l8_830

theorem hyperbola_asymptote_slope
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c ≠ -a ∧ c ≠ a)
  (H1 : (c ≠ -a ∧ c ≠ a) ∧ (a ≠ 0) ∧ (b ≠ 0))
  (H_perp : (c + a) * (c - a) * (a * a * a * a) + (b * b * b * b) = 0) :
  abs (b / a) = 1 :=
by
  sorry  -- Proof here is not required as per the given instructions

end hyperbola_asymptote_slope_l8_830


namespace find_a_l8_841

open Complex

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the main hypothesis: (ai / (1 - i)) = (-1 + i)
def hypothesis (a : ℂ) : Prop :=
  (a * i) / (1 - i) = -1 + i

-- Now, we state the theorem we need to prove
theorem find_a (a : ℝ) (ha : hypothesis a) : a = 2 := by
  sorry

end find_a_l8_841


namespace intersection_complement_l8_821

open Set

noncomputable def U : Set ℝ := {-1, 0, 1, 4}
def A : Set ℝ := {-1, 1}
def B : Set ℝ := {1, 4}
def C_U_B : Set ℝ := U \ B

theorem intersection_complement :
  A ∩ C_U_B = {-1} :=
by
  sorry

end intersection_complement_l8_821


namespace find_range_of_m_l8_853

noncomputable def p (m : ℝ) : Prop := 1 - Real.sqrt 2 < m ∧ m < 1 + Real.sqrt 2
noncomputable def q (m : ℝ) : Prop := 0 < m ∧ m < 4

theorem find_range_of_m (m : ℝ) (hpq : p m ∨ q m) (hnp : ¬ p m) : 1 + Real.sqrt 2 ≤ m ∧ m < 4 :=
sorry

end find_range_of_m_l8_853


namespace S_not_eq_T_l8_875

def S := {x : ℤ | ∃ n : ℤ, x = 2 * n}
def T := {x : ℤ | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}

theorem S_not_eq_T : S ≠ T := by
  sorry

end S_not_eq_T_l8_875


namespace calculate_lower_profit_percentage_l8_872

theorem calculate_lower_profit_percentage 
  (CP : ℕ) 
  (profitAt18Percent : ℕ) 
  (additionalProfit : ℕ)
  (hCP : CP = 800) 
  (hProfitAt18Percent : profitAt18Percent = 144) 
  (hAdditionalProfit : additionalProfit = 72) 
  (hProfitRelation : profitAt18Percent = additionalProfit + ((9 * CP) / 100)) :
  9 = ((9 * CP) / 100) :=
by
  sorry

end calculate_lower_profit_percentage_l8_872


namespace christina_speed_limit_l8_866

theorem christina_speed_limit :
  ∀ (D total_distance friend_distance : ℝ), 
  total_distance = 210 → 
  friend_distance = 3 * 40 → 
  D = total_distance - friend_distance → 
  D / 3 = 30 :=
by
  intros D total_distance friend_distance 
  intros h1 h2 h3 
  sorry

end christina_speed_limit_l8_866


namespace eugene_cards_in_deck_l8_827

theorem eugene_cards_in_deck 
  (cards_used_per_card : ℕ)
  (boxes_used : ℕ)
  (toothpicks_per_box : ℕ)
  (cards_leftover : ℕ)
  (total_toothpicks_used : ℕ)
  (cards_used : ℕ)
  (total_cards_in_deck : ℕ)
  (h1 : cards_used_per_card = 75)
  (h2 : boxes_used = 6)
  (h3 : toothpicks_per_box = 450)
  (h4 : cards_leftover = 16)
  (h5 : total_toothpicks_used = boxes_used * toothpicks_per_box)
  (h6 : cards_used = total_toothpicks_used / cards_used_per_card)
  (h7 : total_cards_in_deck = cards_used + cards_leftover) :
  total_cards_in_deck = 52 :=
by 
  sorry

end eugene_cards_in_deck_l8_827


namespace ninety_percent_greater_than_eighty_percent_l8_817

-- Define the constants involved in the problem
def ninety_percent (n : ℕ) : ℝ := 0.90 * n
def eighty_percent (n : ℕ) : ℝ := 0.80 * n

-- Define the problem statement
theorem ninety_percent_greater_than_eighty_percent :
  ninety_percent 40 - eighty_percent 30 = 12 :=
by
  sorry

end ninety_percent_greater_than_eighty_percent_l8_817


namespace side_length_of_square_base_l8_852

theorem side_length_of_square_base (area : ℝ) (slant_height : ℝ) (s : ℝ) (h : slant_height = 40) (a : area = 160) : s = 8 :=
by sorry

end side_length_of_square_base_l8_852


namespace minute_hand_distance_l8_880

noncomputable def distance_traveled (length_of_minute_hand : ℝ) (time_duration : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * length_of_minute_hand
  let revolutions := time_duration / 60
  circumference * revolutions

theorem minute_hand_distance :
  distance_traveled 8 45 = 12 * Real.pi :=
by
  sorry

end minute_hand_distance_l8_880


namespace parabola_focus_and_directrix_l8_813

theorem parabola_focus_and_directrix :
  (∀ x y : ℝ, x^2 = 4 * y → ∃ a b : ℝ, (a, b) = (0, 1) ∧ y = -1) :=
by
  -- Here, we would provide definitions and logical steps if we were completing the proof.
  -- For now, we will leave it unfinished.
  sorry

end parabola_focus_and_directrix_l8_813


namespace number_of_distinct_sentences_l8_891

noncomputable def count_distinct_sentences (phrase : String) : Nat :=
  let I_options := 3 -- absent, partially present, fully present
  let II_options := 2 -- absent, present
  let IV_options := 2 -- incomplete or absent
  let III_mandatory := 1 -- always present
  (III_mandatory * IV_options * I_options * II_options) - 1 -- subtract the original sentence

theorem number_of_distinct_sentences :
  count_distinct_sentences "ранним утром на рыбалку улыбающийся Игорь мчался босиком" = 23 :=
by
  sorry

end number_of_distinct_sentences_l8_891


namespace seventh_triangular_number_eq_28_l8_856

noncomputable def triangular_number (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem seventh_triangular_number_eq_28 :
  triangular_number 7 = 28 :=
by
  sorry

end seventh_triangular_number_eq_28_l8_856


namespace sequence_all_integers_l8_854

open Nat

def a : ℕ → ℤ
| 0 => 1
| 1 => 1
| n+2 => (a (n+1))^2 + 2 / a n

theorem sequence_all_integers :
  ∀ n : ℕ, ∃ k : ℤ, a n = k :=
by
  sorry

end sequence_all_integers_l8_854


namespace correct_completion_of_sentence_l8_885

def committee_discussing_problem : Prop := True -- Placeholder for the condition
def problem_expected_to_be_solved_next_week : Prop := True -- Placeholder for the condition

theorem correct_completion_of_sentence 
  (h1 : committee_discussing_problem) 
  (h2 : problem_expected_to_be_solved_next_week) 
  : "hopefully" = "hopefully" :=
by 
  sorry

end correct_completion_of_sentence_l8_885


namespace simplify_expression_l8_855

variable (x y : ℝ)

theorem simplify_expression (A B : ℝ) (hA : A = x^2) (hB : B = y^2) :
  (A + B) / (A - B) + (A - B) / (A + B) = 2 * (x^4 + y^4) / (x^4 - y^4) :=
by {
  sorry
}

end simplify_expression_l8_855


namespace cody_discount_l8_842

theorem cody_discount (initial_cost tax_rate cody_paid total_paid price_before_discount discount: ℝ) 
  (h1 : initial_cost = 40)
  (h2 : tax_rate = 0.05)
  (h3 : cody_paid = 17)
  (h4 : total_paid = 2 * cody_paid)
  (h5 : price_before_discount = initial_cost * (1 + tax_rate))
  (h6 : discount = price_before_discount - total_paid) :
  discount = 8 := by
  sorry

end cody_discount_l8_842


namespace line_through_three_points_l8_834

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Given conditions
def p1 : Point := { x := 1, y := -1 }
def p2 : Point := { x := 3, y := 3 }
def p3 : Point := { x := 2, y := 1 }

-- The line that passes through the points
def line_eq (m b : ℝ) (p : Point) : Prop :=
  p.y = m * p.x + b

-- The condition of passing through the three points
def passes_three_points (m b : ℝ) : Prop :=
  line_eq m b p1 ∧ line_eq m b p2 ∧ line_eq m b p3

-- The statement to prove
theorem line_through_three_points (m b : ℝ) (h : passes_three_points m b) : m + b = -1 :=
  sorry

end line_through_three_points_l8_834


namespace true_propositions_l8_835

theorem true_propositions :
  (∀ x : ℚ, ∃ y : ℚ, y = (1/3 : ℚ) * x^2 + (1/2 : ℚ) * x + 1) ∧
  (∃ x y : ℤ, 3 * x - 2 * y = 10) :=
by {
  sorry
}

end true_propositions_l8_835


namespace not_all_ten_on_boundary_of_same_square_l8_867

open Function

variable (points : Fin 10 → ℝ × ℝ)

def four_points_on_square (A B C D : ℝ × ℝ) : Prop :=
  -- Define your own predicate to check if 4 points A, B, C, D are on the boundary of some square
  sorry 

theorem not_all_ten_on_boundary_of_same_square :
  (∀ A B C D : Fin 10, four_points_on_square (points A) (points B) (points C) (points D)) →
  ¬ (∃ square : ℝ × ℝ → Prop, ∀ i : Fin 10, square (points i)) :=
by
  intro h
  sorry

end not_all_ten_on_boundary_of_same_square_l8_867


namespace paint_liters_needed_l8_893

theorem paint_liters_needed :
  let cost_brushes : ℕ := 20
  let cost_canvas : ℕ := 3 * cost_brushes
  let cost_paint_per_liter : ℕ := 8
  let total_costs : ℕ := 120
  ∃ (liters_of_paint : ℕ), cost_brushes + cost_canvas + cost_paint_per_liter * liters_of_paint = total_costs ∧ liters_of_paint = 5 :=
by
  sorry

end paint_liters_needed_l8_893


namespace cooking_oil_remaining_l8_845

theorem cooking_oil_remaining (initial_weight : ℝ) (fraction_used : ℝ) (remaining_weight : ℝ) :
  initial_weight = 5 → fraction_used = 4 / 5 → remaining_weight = 21 / 5 → initial_weight * (1 - fraction_used) ≠ remaining_weight → initial_weight * (1 - fraction_used) = 1 :=
by 
  intros h_initial_weight h_fraction_used h_remaining_weight h_contradiction
  sorry

end cooking_oil_remaining_l8_845


namespace total_amount_after_interest_l8_896

-- Define the constants
def principal : ℝ := 979.0209790209791
def rate : ℝ := 0.06
def time : ℝ := 2.4

-- Define the formula for interest calculation
def interest (P R T : ℝ) : ℝ := P * R * T

-- Define the formula for the total amount after interest is added
def total_amount (P I : ℝ) : ℝ := P + I

-- State the theorem
theorem total_amount_after_interest : 
    total_amount principal (interest principal rate time) = 1120.0649350649352 :=
by
    -- placeholder for the proof
    sorry

end total_amount_after_interest_l8_896


namespace find_quaterns_l8_868

theorem find_quaterns {
  x y z w : ℝ
} : 
  (x + y = z^2 + w^2 + 6 * z * w) → 
  (x + z = y^2 + w^2 + 6 * y * w) → 
  (x + w = y^2 + z^2 + 6 * y * z) → 
  (y + z = x^2 + w^2 + 6 * x * w) → 
  (y + w = x^2 + z^2 + 6 * x * z) → 
  (z + w = x^2 + y^2 + 6 * x * y) → 
  ( (x, y, z, w) = (0, 0, 0, 0) 
    ∨ (x, y, z, w) = (1/4, 1/4, 1/4, 1/4) 
    ∨ (x, y, z, w) = (-1/4, -1/4, 3/4, -1/4) 
    ∨ (x, y, z, w) = (-1/2, -1/2, 5/2, -1/2)
  ) :=
  sorry

end find_quaterns_l8_868


namespace employee_earnings_l8_810

theorem employee_earnings (regular_rate overtime_rate first3_days_h second2_days_h total_hours overtime_hours : ℕ)
  (h1 : regular_rate = 30)
  (h2 : overtime_rate = 45)
  (h3 : first3_days_h = 6)
  (h4 : second2_days_h = 12)
  (h5 : total_hours = first3_days_h * 3 + second2_days_h * 2)
  (h6 : total_hours = 42)
  (h7 : overtime_hours = total_hours - 40)
  (h8 : overtime_hours = 2) :
  (40 * regular_rate + overtime_hours * overtime_rate) = 1290 := 
sorry

end employee_earnings_l8_810


namespace least_possible_mn_correct_l8_803

def least_possible_mn (m n : ℕ) : ℕ :=
  m + n

theorem least_possible_mn_correct (m n : ℕ) :
  (Nat.gcd (m + n) 210 = 1) →
  (n^n ∣ m^m) →
  ¬(n ∣ m) →
  least_possible_mn m n = 407 :=
by
  sorry

end least_possible_mn_correct_l8_803


namespace average_after_31st_inning_l8_828

-- Define the conditions as Lean definitions
def initial_average (A : ℝ) := A

def total_runs_before_31st_inning (A : ℝ) := 30 * A

def score_in_31st_inning := 105

def new_average (A : ℝ) := A + 3

def total_runs_after_31st_inning (A : ℝ) := total_runs_before_31st_inning A + score_in_31st_inning

-- Define the statement to prove the batsman's average after the 31st inning is 15
theorem average_after_31st_inning (A : ℝ) : total_runs_after_31st_inning A = 31 * (new_average A) → new_average A = 15 := by
  sorry

end average_after_31st_inning_l8_828


namespace tangent_line_at_1_l8_838

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

theorem tangent_line_at_1 :
  let x := (1 : ℝ)
  let y := (f 1)
  ∃ m b : ℝ, (∀ x, y - m * (x - 1) + b = 0)
  ∧ (m = -2)
  ∧ (b = -1) :=
by
  sorry

end tangent_line_at_1_l8_838


namespace max_gcd_14m_plus_4_9m_plus_2_l8_848

theorem max_gcd_14m_plus_4_9m_plus_2 (m : ℕ) (h : m > 0) : ∃ M, M = 8 ∧ ∀ k, gcd (14 * m + 4) (9 * m + 2) = k → k ≤ M :=
by
  sorry

end max_gcd_14m_plus_4_9m_plus_2_l8_848


namespace percent_uni_no_job_choice_l8_831

variable (P_ND_JC P_JC P_UD P_U_NJC P_NJC : ℝ)
variable (h1 : P_ND_JC = 0.18)
variable (h2 : P_JC = 0.40)
variable (h3 : P_UD = 0.37)

theorem percent_uni_no_job_choice :
  (P_UD - (P_JC - P_ND_JC)) / (1 - P_JC) = 0.25 :=
by
  sorry

end percent_uni_no_job_choice_l8_831


namespace negation_proposition_l8_801

theorem negation_proposition : 
  ¬ (∃ x_0 : ℝ, x_0^2 + x_0 + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by {
  sorry
}

end negation_proposition_l8_801


namespace smallest_distance_AB_ge_2_l8_812

noncomputable def A (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 9
noncomputable def B (x y : ℝ) : Prop := y^2 = -8 * x

theorem smallest_distance_AB_ge_2 :
  ∀ (x1 y1 x2 y2 : ℝ), A x1 y1 → B x2 y2 → dist (x1, y1) (x2, y2) ≥ 2 := by
  sorry

end smallest_distance_AB_ge_2_l8_812


namespace unit_prices_min_chess_sets_l8_863

-- Define the conditions and prove the unit prices.
theorem unit_prices (x y : ℝ) 
  (h1 : 6 * x + 5 * y = 190)
  (h2 : 8 * x + 10 * y = 320) : 
  x = 15 ∧ y = 20 :=
by
  sorry

-- Define the conditions for the budget and prove the minimum number of chess sets.
theorem min_chess_sets (x y : ℝ) (m : ℕ)
  (hx : x = 15)
  (hy : y = 20)
  (number_sets : m + (100 - m) = 100)
  (budget : 15 * ↑m + 20 * ↑(100 - m) ≤ 1800) :
  m ≥ 40 :=
by
  sorry

end unit_prices_min_chess_sets_l8_863


namespace minimum_bailing_rate_l8_869

theorem minimum_bailing_rate
  (distance_from_shore : Real := 1.5)
  (rowing_speed : Real := 3)
  (water_intake_rate : Real := 12)
  (max_water : Real := 45) :
  (distance_from_shore / rowing_speed) * 60 * water_intake_rate - max_water / ((distance_from_shore / rowing_speed) * 60) >= 10.5 :=
by
  -- Provide the units are consistent and the calculations agree with the given numerical data
  sorry

end minimum_bailing_rate_l8_869


namespace solution_set_of_fractional_inequality_l8_837

theorem solution_set_of_fractional_inequality :
  {x : ℝ | (x + 1) / (x - 3) < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_of_fractional_inequality_l8_837


namespace sufficient_condition_x_gt_2_l8_861

theorem sufficient_condition_x_gt_2 (x : ℝ) (h : x > 2) : x^2 - 2 * x > 0 := by
  sorry

end sufficient_condition_x_gt_2_l8_861


namespace simplify_fraction_l8_860

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l8_860


namespace train_speed_l8_874

/--A train leaves Delhi at 9 a.m. at a speed of 30 kmph.
Another train leaves at 3 p.m. on the same day and in the same direction.
The two trains meet 720 km away from Delhi.
Prove that the speed of the second train is 120 kmph.-/
theorem train_speed
  (speed_first_train speed_first_kmph : 30 = 30)
  (leave_first_train : Nat)
  (leave_first_9am : 9 = 9)
  (leave_second_train : Nat)
  (leave_second_3pm : 3 = 3)
  (distance_meeting_km : Nat)
  (distance_meeting_720km : 720 = 720) :
  ∃ speed_second_train, speed_second_train = 120 := 
sorry

end train_speed_l8_874


namespace volume_ratio_spheres_l8_832

theorem volume_ratio_spheres (r1 r2 r3 v1 v2 v3 : ℕ)
  (h_rad_ratio : r1 = 1 ∧ r2 = 2 ∧ r3 = 3)
  (h_vol_ratio : v1 = r1^3 ∧ v2 = r2^3 ∧ v3 = r3^3) :
  v3 = 3 * (v1 + v2) := by
  -- main proof goes here
  sorry

end volume_ratio_spheres_l8_832


namespace distance_midpoints_eq_2_5_l8_818

theorem distance_midpoints_eq_2_5 (A B C : ℝ) (hAB : A < B) (hBC : B < C) (hAC_len : C - A = 5) :
    let M1 := (A + B) / 2
    let M2 := (B + C) / 2
    (M2 - M1 = 2.5) :=
by
    let M1 := (A + B) / 2
    let M2 := (B + C) / 2
    sorry

end distance_midpoints_eq_2_5_l8_818


namespace profit_difference_l8_886

-- Define the initial capitals of A, B, and C
def capital_A := 8000
def capital_B := 10000
def capital_C := 12000

-- Define B's profit share
def profit_share_B := 3500

-- Define the total number of parts
def total_parts := 15

-- Define the number of parts for each person
def parts_A := 4
def parts_B := 5
def parts_C := 6

-- Define the total profit
noncomputable def total_profit := profit_share_B * (total_parts / parts_B)

-- Define the profit shares of A and C
noncomputable def profit_share_A := (parts_A / total_parts) * total_profit
noncomputable def profit_share_C := (parts_C / total_parts) * total_profit

-- Define the difference between the profit shares of A and C
noncomputable def profit_share_difference := profit_share_C - profit_share_A

-- The theorem to prove
theorem profit_difference :
  profit_share_difference = 1400 := by
  sorry

end profit_difference_l8_886


namespace david_marks_in_biology_l8_850

theorem david_marks_in_biology (english: ℕ) (math: ℕ) (physics: ℕ) (chemistry: ℕ) (average: ℕ) (biology: ℕ) :
  english = 81 ∧ math = 65 ∧ physics = 82 ∧ chemistry = 67 ∧ average = 76 → (biology = 85) :=
by
  sorry

end david_marks_in_biology_l8_850


namespace decrease_travel_time_l8_804

variable (distance : ℕ) (initial_speed : ℕ) (speed_increase : ℕ)

def original_travel_time (distance initial_speed : ℕ) : ℕ :=
  distance / initial_speed

def new_travel_time (distance new_speed : ℕ) : ℕ :=
  distance / new_speed

theorem decrease_travel_time (h₁ : distance = 600) (h₂ : initial_speed = 50) (h₃ : speed_increase = 25) :
  original_travel_time distance initial_speed - new_travel_time distance (initial_speed + speed_increase) = 4 :=
by
  sorry

end decrease_travel_time_l8_804


namespace no_solution_exists_l8_811

theorem no_solution_exists (x y z : ℕ) (hx : x > 2) (hy : y > 1) (h : x^y + 1 = z^2) : false := 
by
  sorry

end no_solution_exists_l8_811


namespace additional_girls_needed_l8_876

theorem additional_girls_needed (initial_girls initial_boys additional_girls : ℕ)
  (h_initial_girls : initial_girls = 2)
  (h_initial_boys : initial_boys = 6)
  (h_fraction_goal : (initial_girls + additional_girls) = (5 * (initial_girls + initial_boys + additional_girls)) / 8) :
  additional_girls = 8 :=
by
  -- A placeholder for the proof
  sorry

end additional_girls_needed_l8_876


namespace complex_number_coordinates_l8_877

-- Define i as the imaginary unit
def i := Complex.I

-- State the theorem
theorem complex_number_coordinates : (i * (1 - i)).re = 1 ∧ (i * (1 - i)).im = 1 :=
by
  -- Proof would go here
  sorry

end complex_number_coordinates_l8_877


namespace quadratic_solution_condition_sufficient_but_not_necessary_l8_894

theorem quadratic_solution_condition_sufficient_but_not_necessary (m : ℝ) :
  (m < -2) → (∃ x : ℝ, x^2 + m * x + 1 = 0) ∧ ¬(∀ m : ℝ, ∃ x : ℝ, x^2 + m * x + 1 = 0 → m < -2) :=
by 
  sorry

end quadratic_solution_condition_sufficient_but_not_necessary_l8_894


namespace arithmetic_sequence_sum_l8_847

theorem arithmetic_sequence_sum :
  ∀ (x y : ℤ), (∃ (n m : ℕ), (3 + n * 6 = x) ∧ (3 + m * 6 = y) ∧ x + 6 = y ∧ y + 6 = 33) → x + y = 60 :=
by
  intro x y h
  obtain ⟨n, m, hn, hm, hx, hy⟩ := h
  exact sorry

end arithmetic_sequence_sum_l8_847


namespace original_price_of_color_TV_l8_800

theorem original_price_of_color_TV
  (x : ℝ)  -- Let the variable x represent the original price
  (h1 : x * 1.4 * 0.8 - x = 144)  -- Condition as equation
  : x = 1200 := 
sorry  -- Proof to be filled in later

end original_price_of_color_TV_l8_800


namespace new_concentration_of_mixture_l8_809

theorem new_concentration_of_mixture :
  let v1 := 2
  let c1 := 0.25
  let v2 := 6
  let c2 := 0.40
  let V := 10
  let alcohol_amount_v1 := v1 * c1
  let alcohol_amount_v2 := v2 * c2
  let total_alcohol := alcohol_amount_v1 + alcohol_amount_v2
  let new_concentration := (total_alcohol / V) * 100
  new_concentration = 29 := 
by
  sorry

end new_concentration_of_mixture_l8_809


namespace sum_of_first_13_terms_is_39_l8_882

-- Definition of arithmetic sequence and the given condition
def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

-- Given condition
axiom given_condition {a : ℕ → ℤ} (h : arithmetic_sequence a) : a 5 + a 6 + a 7 = 9

-- The main theorem
theorem sum_of_first_13_terms_is_39 {a : ℕ → ℤ} (h : arithmetic_sequence a) (h9 : a 5 + a 6 + a 7 = 9) : sum_of_first_n_terms a 12 = 39 :=
sorry

end sum_of_first_13_terms_is_39_l8_882


namespace billy_weight_l8_881

variable (B Bd C D : ℝ)

theorem billy_weight
  (h1 : B = Bd + 9)
  (h2 : Bd = C + 5)
  (h3 : C = D - 8)
  (h4 : C = 145)
  (h5 : D = 2 * Bd) :
  B = 85.5 :=
by
  sorry

end billy_weight_l8_881


namespace rectangle_area_l8_871

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 56) :
  l * b = 147 := by
  sorry

end rectangle_area_l8_871


namespace number_of_three_digit_numbers_with_123_exactly_once_l8_826

theorem number_of_three_digit_numbers_with_123_exactly_once : 
  (∃ (l : List ℕ), l = [1, 2, 3] ∧ l.permutations.length = 6) :=
by
  sorry

end number_of_three_digit_numbers_with_123_exactly_once_l8_826


namespace part1_part2_l8_892

open Real

def f (x a : ℝ) := abs (x + 2 * a) + abs (x - 1)

section part1

variable (x : ℝ)

theorem part1 (a : ℝ) (h : a = 1) : f x a ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
by
  sorry

end part1

section part2

noncomputable def g (a : ℝ) := abs ((1 : ℝ) / a + 2 * a) + abs ((1 : ℝ) / a - 1)

theorem part2 {a : ℝ} (h : a ≠ 0) : g a ≤ 4 ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end part2

end part1_part2_l8_892


namespace sam_current_yellow_marbles_l8_883

theorem sam_current_yellow_marbles (original_yellow : ℕ) (taken_yellow : ℕ) (current_yellow : ℕ) :
  original_yellow = 86 → 
  taken_yellow = 25 → 
  current_yellow = original_yellow - taken_yellow → 
  current_yellow = 61 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sam_current_yellow_marbles_l8_883


namespace tenth_day_of_month_is_monday_l8_864

theorem tenth_day_of_month_is_monday (Sundays_on_even_dates : ℕ → Prop)
  (h1: Sundays_on_even_dates 2)
  (h2: Sundays_on_even_dates 16)
  (h3: Sundays_on_even_dates 30) :
  ∃ k : ℕ, 10 = k + 2 + 7 * 1 ∧ k.succ.succ.succ.succ.succ.succ.succ.succ.succ.succ = 1 :=
by sorry

end tenth_day_of_month_is_monday_l8_864


namespace domain_of_function_l8_846

theorem domain_of_function : 
  {x : ℝ | x ≠ 1 ∧ x > 0} = {x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x)} :=
by
  sorry

end domain_of_function_l8_846


namespace no_perfect_square_in_range_l8_844

theorem no_perfect_square_in_range :
  ¬∃ (x : ℕ), 99990000 ≤ x ∧ x ≤ 99999999 ∧ ∃ (n : ℕ), x = n * n :=
by
  sorry

end no_perfect_square_in_range_l8_844


namespace solve_ineq_l8_870

noncomputable def inequality (x : ℝ) : Prop :=
  (x^2 / (x+1)) ≥ (3 / (x+1) + 3)

theorem solve_ineq :
  { x : ℝ | inequality x } = { x : ℝ | x ≤ -6 ∨ (-1 < x ∧ x ≤ 3) } := sorry

end solve_ineq_l8_870


namespace Emily_sixth_quiz_score_l8_820

theorem Emily_sixth_quiz_score :
  let scores := [92, 95, 87, 89, 100]
  ∃ s : ℕ, (s + scores.sum : ℚ) / 6 = 93 :=
  by
    sorry

end Emily_sixth_quiz_score_l8_820


namespace faster_train_passes_slower_in_54_seconds_l8_890

-- Definitions of the conditions.
def length_of_train := 75 -- Length of each train in meters.
def speed_faster_train := 46 * 1000 / 3600 -- Speed of the faster train in m/s.
def speed_slower_train := 36 * 1000 / 3600 -- Speed of the slower train in m/s.
def relative_speed := speed_faster_train - speed_slower_train -- Relative speed in m/s.
def total_distance := 2 * length_of_train -- Total distance to cover to pass the slower train.

-- The proof statement.
theorem faster_train_passes_slower_in_54_seconds : total_distance / relative_speed = 54 := by
  sorry

end faster_train_passes_slower_in_54_seconds_l8_890


namespace contradiction_proof_l8_833

theorem contradiction_proof (a b c : ℝ) (h : ¬ (a > 0 ∨ b > 0 ∨ c > 0)) : false :=
by
  sorry

end contradiction_proof_l8_833


namespace mean_of_other_two_numbers_l8_899

theorem mean_of_other_two_numbers (a b c d e f g h : ℕ)
  (h_tuple : a = 1871 ∧ b = 2011 ∧ c = 2059 ∧ d = 2084 ∧ e = 2113 ∧ f = 2167 ∧ g = 2198 ∧ h = 2210)
  (h_mean : (a + b + c + d + e + f) / 6 = 2100) :
  ((g + h) / 2 : ℚ) = 2056.5 :=
by
  sorry

end mean_of_other_two_numbers_l8_899


namespace greatest_product_sum_2006_l8_839

theorem greatest_product_sum_2006 :
  (∃ x y : ℤ, x + y = 2006 ∧ ∀ a b : ℤ, a + b = 2006 → a * b ≤ x * y) → 
  ∃ x y : ℤ, x + y = 2006 ∧ x * y = 1006009 :=
by sorry

end greatest_product_sum_2006_l8_839


namespace increasing_iff_range_a_three_distinct_real_roots_l8_857

noncomputable def f (a x : ℝ) : ℝ :=
  if x >= 2 * a then x^2 + (2 - 2 * a) * x else - x^2 + (2 + 2 * a) * x

theorem increasing_iff_range_a (a : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem three_distinct_real_roots (a t : ℝ) (h_a : -2 ≤ a ∧ a ≤ 2)
  (h_roots : ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧
                           f a x₁ = t * f a (2 * a) ∧
                           f a x₂ = t * f a (2 * a) ∧
                           f a x₃ = t * f a (2 * a)) :
  1 < t ∧ t < 9 / 8 :=
sorry

end increasing_iff_range_a_three_distinct_real_roots_l8_857


namespace combined_weight_of_daughter_and_child_l8_889

variables (M D C : ℝ)
axiom mother_daughter_grandchild_weight : M + D + C = 120
axiom daughter_weight : D = 48
axiom child_weight_fraction_of_grandmother : C = (1 / 5) * M

theorem combined_weight_of_daughter_and_child : D + C = 60 :=
  sorry

end combined_weight_of_daughter_and_child_l8_889


namespace number_minus_45_l8_849

theorem number_minus_45 (x : ℕ) (h1 : (x / 2) / 2 = 85 + 45) : x - 45 = 475 := by
  sorry

end number_minus_45_l8_849


namespace distance_A_beats_B_l8_802

noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem distance_A_beats_B :
  let distance_A := 5 -- km
  let time_A := 10 / 60 -- hours (10 minutes)
  let time_B := 14 / 60 -- hours (14 minutes)
  let speed_A := speed distance_A time_A
  let speed_B := speed distance_A time_B
  let distance_A_in_time_B := speed_A * time_B
  distance_A_in_time_B - distance_A = 2 := -- km
by
  sorry

end distance_A_beats_B_l8_802


namespace max_y_difference_l8_829

theorem max_y_difference : (∃ x, (5 - 2 * x^2 + 2 * x^3 = 1 + x^2 + x^3)) ∧ 
                           (∀ y1 y2, y1 = 5 - 2 * (2^2) + 2 * (2^3) ∧ y2 = 5 - 2 * (1/2)^2 + 2 * (1/2)^3 → 
                           (y1 - y2 = 11.625)) := sorry

end max_y_difference_l8_829


namespace integer_pairs_satisfy_equation_l8_824

theorem integer_pairs_satisfy_equation :
  ∃ (S : Finset (ℤ × ℤ)), S.card = 5 ∧ ∀ (m n : ℤ), (m, n) ∈ S ↔ m^2 + n = m * n + 1 :=
by
  sorry

end integer_pairs_satisfy_equation_l8_824


namespace evaluate_expression_at_x_zero_l8_840

theorem evaluate_expression_at_x_zero (x : ℕ) (h1 : x < 3) (h2 : x ≠ 1) (h3 : x ≠ 2) : ((3 / (x - 1) - x - 1) / (x - 2) / (x^2 - 2 * x + 1)) = 2 :=
by
  -- Here we need to provide our proof, though for now it’s indicated by sorry
  sorry

end evaluate_expression_at_x_zero_l8_840


namespace coconut_grove_nut_yield_l8_823

theorem coconut_grove_nut_yield (x : ℕ) (Y : ℕ) 
  (h1 : (x + 4) * 60 + x * 120 + (x - 4) * Y = 3 * x * 100)
  (h2 : x = 8) : Y = 180 := 
by
  sorry

end coconut_grove_nut_yield_l8_823


namespace problem_expression_value_l8_873

theorem problem_expression_value :
  (100 - (3010 - 301)) + (3010 - (301 - 100)) = 200 :=
by
  sorry

end problem_expression_value_l8_873


namespace additional_amount_needed_l8_814

-- Definitions of the conditions
def shampoo_cost : ℝ := 10.00
def conditioner_cost : ℝ := 10.00
def lotion_cost : ℝ := 6.00
def lotions_count : ℕ := 3
def free_shipping_threshold : ℝ := 50.00

-- Calculating the total amount spent
def total_spent : ℝ :=
  shampoo_cost + conditioner_cost + lotions_count * lotion_cost

-- Required statement for the proof
theorem additional_amount_needed : 
  total_spent + 12.00 = free_shipping_threshold :=
by 
  -- Proof will be here
  sorry

end additional_amount_needed_l8_814


namespace z_is_1_2_decades_younger_than_x_l8_808

variable (x y z w : ℕ) -- Assume ages as natural numbers

def age_equivalence_1 : Prop := x + y = y + z + 12
def age_equivalence_2 : Prop := x + y + w = y + z + w + 12

theorem z_is_1_2_decades_younger_than_x (h1 : age_equivalence_1 x y z) (h2 : age_equivalence_2 x y z w) :
  z = x - 12 := by
  sorry

end z_is_1_2_decades_younger_than_x_l8_808


namespace solve_adult_tickets_l8_879

theorem solve_adult_tickets (A C : ℕ) (h1 : 8 * A + 5 * C = 236) (h2 : A + C = 34) : A = 22 :=
sorry

end solve_adult_tickets_l8_879


namespace toys_per_rabbit_l8_884

theorem toys_per_rabbit 
  (rabbits toys_mon toys_wed toys_fri toys_sat : ℕ) 
  (hrabbits : rabbits = 16) 
  (htoys_mon : toys_mon = 6)
  (htoys_wed : toys_wed = 2 * toys_mon)
  (htoys_fri : toys_fri = 4 * toys_mon)
  (htoys_sat : toys_sat = toys_wed / 2) :
  (toys_mon + toys_wed + toys_fri + toys_sat) / rabbits = 3 :=
by 
  sorry

end toys_per_rabbit_l8_884


namespace condition_of_A_with_respect_to_D_l8_898

variables {A B C D : Prop}

theorem condition_of_A_with_respect_to_D (h1 : A → B) (h2 : ¬ (B → A)) (h3 : B ↔ C) (h4 : C → D) (h5 : ¬ (D → C)) :
  (D → A) ∧ ¬ (A → D) :=
by
  sorry

end condition_of_A_with_respect_to_D_l8_898


namespace average_speed_bike_l8_851

theorem average_speed_bike (t_goal : ℚ) (d_swim r_swim : ℚ) (d_run r_run : ℚ) (d_bike r_bike : ℚ) :
  t_goal = 1.75 →
  d_swim = 1 / 3 ∧ r_swim = 1.5 →
  d_run = 2.5 ∧ r_run = 8 →
  d_bike = 12 →
  r_bike = 1728 / 175 :=
by
  intros h_goal h_swim h_run h_bike
  sorry

end average_speed_bike_l8_851


namespace parabola_properties_l8_815

theorem parabola_properties 
  (p : ℝ) (h_pos : 0 < p) (m : ℝ) 
  (A B : ℝ × ℝ)
  (h_AB_on_parabola : ∀ (P : ℝ × ℝ), P = A ∨ P = B → (P.snd)^2 = 2 * p * P.fst) 
  (h_line_intersection : ∀ (P : ℝ × ℝ), P = A ∨ P = B → P.fst = m * P.snd + 3)
  (h_dot_product : (A.fst * B.fst + A.snd * B.snd) = 6)
  : (exists C : ℝ × ℝ, C = (-3, 0)) ∧
    (∃ k1 k2 : ℝ, 
        k1 = A.snd / (A.fst + 3) ∧ 
        k2 = B.snd / (B.fst + 3) ∧ 
        (1 / k1^2 + 1 / k2^2 - 2 * m^2) = 24) :=
by
  sorry

end parabola_properties_l8_815


namespace oil_price_reduction_l8_816

theorem oil_price_reduction (P P_r : ℝ) (h1 : P_r = 24.3) (h2 : 1080 / P - 1080 / P_r = 8) : 
  ((P - P_r) / P) * 100 = 18.02 := by
  sorry

end oil_price_reduction_l8_816


namespace rhombus_has_perpendicular_diagonals_and_rectangle_not_l8_865

-- Definitions based on conditions (a))
def rhombus (sides_equal : Prop) (diagonals_bisect : Prop) (diagonals_perpendicular : Prop) : Prop :=
  sides_equal ∧ diagonals_bisect ∧ diagonals_perpendicular

def rectangle (sides_equal : Prop) (diagonals_bisect : Prop) (diagonals_equal : Prop) : Prop :=
  sides_equal ∧ diagonals_bisect ∧ diagonals_equal

-- Theorem to prove (c))
theorem rhombus_has_perpendicular_diagonals_and_rectangle_not 
  (rhombus_sides_equal rhombus_diagonals_bisect rhombus_diagonals_perpendicular : Prop)
  (rectangle_sides_equal rectangle_diagonals_bisect rectangle_diagonals_equal : Prop) :
  rhombus rhombus_sides_equal rhombus_diagonals_bisect rhombus_diagonals_perpendicular → 
  rectangle rectangle_sides_equal rectangle_diagonals_bisect rectangle_diagonals_equal → 
  rhombus_diagonals_perpendicular ∧ ¬(rectangle (rectangle_sides_equal) (rectangle_diagonals_bisect) (rhombus_diagonals_perpendicular)) :=
sorry

end rhombus_has_perpendicular_diagonals_and_rectangle_not_l8_865


namespace sarah_likes_digits_l8_822

theorem sarah_likes_digits : ∀ n : ℕ, n % 8 = 0 → (n % 10 = 0 ∨ n % 10 = 4 ∨ n % 10 = 8) :=
by
  sorry

end sarah_likes_digits_l8_822


namespace alice_burgers_each_day_l8_806

theorem alice_burgers_each_day (cost_per_burger : ℕ) (total_spent : ℕ) (days_in_june : ℕ) 
  (h1 : cost_per_burger = 13) (h2 : total_spent = 1560) (h3 : days_in_june = 30) :
  (total_spent / cost_per_burger) / days_in_june = 4 := by
  sorry

end alice_burgers_each_day_l8_806
