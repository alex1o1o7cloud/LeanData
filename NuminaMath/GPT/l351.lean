import Mathlib

namespace Ms_Smiths_Class_Books_Distribution_l351_35132

theorem Ms_Smiths_Class_Books_Distribution :
  ∃ (x : ℕ), (20 * 2 * x + 15 * x + 5 * x = 840) ∧ (20 * 2 * x = 560) ∧ (15 * x = 210) ∧ (5 * x = 70) :=
by
  let x := 14
  have h1 : 20 * 2 * x + 15 * x + 5 * x = 840 := by sorry
  have h2 : 20 * 2 * x = 560 := by sorry
  have h3 : 15 * x = 210 := by sorry
  have h4 : 5 * x = 70 := by sorry
  exact ⟨x, h1, h2, h3, h4⟩

end Ms_Smiths_Class_Books_Distribution_l351_35132


namespace find_k_no_xy_term_l351_35191

theorem find_k_no_xy_term (k : ℝ) :
  (¬ ∃ x y : ℝ, (-x^2 - 3 * k * x * y - 3 * y^2 + 9 * x * y - 8) = (- x^2 - 3 * y^2 - 8)) → k = 3 :=
by
  sorry

end find_k_no_xy_term_l351_35191


namespace mia_spent_per_parent_l351_35157

theorem mia_spent_per_parent (amount_sibling : ℕ) (num_siblings : ℕ) (total_spent : ℕ) 
  (num_parents : ℕ) : 
  amount_sibling = 30 → num_siblings = 3 → total_spent = 150 → num_parents = 2 → 
  (total_spent - num_siblings * amount_sibling) / num_parents = 30 :=
by
  sorry

end mia_spent_per_parent_l351_35157


namespace subset_condition_l351_35124

variable {U : Type}
variables (P Q : Set U)

theorem subset_condition (h : P ∩ Q = P) : ∀ x : U, x ∉ Q → x ∉ P :=
by {
  sorry
}

end subset_condition_l351_35124


namespace plane_hover_central_time_l351_35122

theorem plane_hover_central_time (x : ℕ) (h1 : 3 + x + 2 + 5 + (x + 2) + 4 = 24) : x = 4 := by
  sorry

end plane_hover_central_time_l351_35122


namespace problem_statement_l351_35118
noncomputable def not_divisible (n : ℕ) : Prop := ∃ k : ℕ, (5^n - 3^n) = (2^n + 65) * k
theorem problem_statement (n : ℕ) (h : 0 < n) : ¬ not_divisible n := sorry

end problem_statement_l351_35118


namespace possible_values_of_m_l351_35193

-- Proposition: for all real values of m, if for all real x, x^2 + 2x + 2 - m >= 0 holds, then m must be one of -1, 0, or 1

theorem possible_values_of_m (m : ℝ) 
  (h : ∀ (x : ℝ), x^2 + 2 * x + 2 - m ≥ 0) : m = -1 ∨ m = 0 ∨ m = 1 :=
sorry

end possible_values_of_m_l351_35193


namespace simplify_expression_l351_35135

theorem simplify_expression (x : ℝ) (h1 : x^3 + 2*x + 1 ≠ 0) (h2 : x^3 - 2*x - 1 ≠ 0) : 
  ( ((x + 2)^2 * (x^2 - x + 2)^2 / (x^3 + 2*x + 1)^2 )^3 * ((x - 2)^2 * (x^2 + x + 2)^2 / (x^3 - 2*x - 1)^2 )^3 ) = 1 :=
by sorry

end simplify_expression_l351_35135


namespace triangle_area_l351_35100

theorem triangle_area (A B C : ℝ) (AB BC CA : ℝ) (sinA sinB sinC : ℝ)
    (h1 : sinA * sinB * sinC = 1 / 1000) 
    (h2 : AB * BC * CA = 1000) : 
    (AB * BC * CA / (4 * 50)) = 5 :=
by
  -- Proof is omitted
  sorry

end triangle_area_l351_35100


namespace system_of_equations_value_l351_35116

theorem system_of_equations_value (x y z : ℝ)
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x + 4 * y - 10 * z = 0)
  (hz : z ≠ 0) :
  (x^2 + 4 * x * y) / (y^2 + z^2) = 96 / 13 := 
sorry

end system_of_equations_value_l351_35116


namespace range_of_a_l351_35147

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 1 then x else a * x^2 + 2 * x

theorem range_of_a (R : Set ℝ) :
  (∀ x : ℝ, f x a ∈ R) → (a ∈ Set.Icc (-1 : ℝ) 0) :=
sorry

end range_of_a_l351_35147


namespace division_of_neg_six_by_three_l351_35120

theorem division_of_neg_six_by_three : (-6) / 3 = -2 := by
  sorry

end division_of_neg_six_by_three_l351_35120


namespace true_proposition_l351_35164

theorem true_proposition : 
  (∃ x0 : ℝ, x0 > 0 ∧ 3^x0 + x0 = 2016) ∧ 
  ¬(∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, abs x - a * x = abs (-x) - a * (-x)) := by
  sorry

end true_proposition_l351_35164


namespace arithmetic_sequence_a2_a6_l351_35192

theorem arithmetic_sequence_a2_a6 (a : ℕ → ℕ) (d : ℕ) (h_arith_seq : ∀ n, a (n+1) = a n + d)
  (h_a4 : a 4 = 4) : a 2 + a 6 = 8 :=
by sorry

end arithmetic_sequence_a2_a6_l351_35192


namespace symm_diff_A_B_l351_35144

-- Define sets A and B
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | abs x < 2}

-- Define set difference
def set_diff (S T : Set ℤ) : Set ℤ := {x | x ∈ S ∧ x ∉ T}

-- Define symmetric difference
def symm_diff (S T : Set ℤ) : Set ℤ := (set_diff S T) ∪ (set_diff T S)

-- Define the expression we need to prove
theorem symm_diff_A_B : symm_diff A B = {-1, 0, 2} := by
  sorry

end symm_diff_A_B_l351_35144


namespace sequence_transformation_possible_l351_35159

theorem sequence_transformation_possible 
  (a1 a2 : ℕ) (h1 : a1 ≤ 100) (h2 : a2 ≤ 100) (h3 : a1 ≥ a2) : 
  ∃ (operations : ℕ), operations ≤ 51 :=
by
  sorry

end sequence_transformation_possible_l351_35159


namespace determine_conflicting_pairs_l351_35129

structure EngineerSetup where
  n : ℕ
  barrels : Fin (2 * n) → Reactant
  conflicts : Fin n → (Reactant × Reactant)

def testTubeBurst (r1 r2 : Reactant) (conflicts : Fin n → (Reactant × Reactant)) : Prop :=
  ∃ i, conflicts i = (r1, r2) ∨ conflicts i = (r2, r1)

theorem determine_conflicting_pairs (setup : EngineerSetup) :
  ∃ pairs : Fin n → (Reactant × Reactant),
  (∀ i, pairs i ∈ { p | ∃ j, setup.conflicts j = p ∨ setup.conflicts j = (p.snd, p.fst) }) ∧
  (∀ i j, i ≠ j → pairs i ≠ pairs j) := 
sorry

end determine_conflicting_pairs_l351_35129


namespace ratio_pentagon_area_l351_35139

noncomputable def square_side_length := 1
noncomputable def square_area := (square_side_length : ℝ)^2
noncomputable def total_area := 3 * square_area
noncomputable def area_triangle (base height : ℝ) := 0.5 * base * height
noncomputable def GC := 2 / 3 * square_side_length
noncomputable def HD := 2 / 3 * square_side_length
noncomputable def area_GJC := area_triangle GC square_side_length
noncomputable def area_HDJ := area_triangle HD square_side_length
noncomputable def area_AJKCB := square_area - (area_GJC + area_HDJ)

theorem ratio_pentagon_area :
  (area_AJKCB / total_area) = 1 / 9 := 
sorry

end ratio_pentagon_area_l351_35139


namespace annual_interest_rate_l351_35145

theorem annual_interest_rate (r : ℝ): 
  (1000 * r * 4.861111111111111 + 1400 * r * 4.861111111111111 = 350) → 
  r = 0.03 :=
sorry

end annual_interest_rate_l351_35145


namespace RS_segment_length_l351_35138

theorem RS_segment_length (P Q R S : ℝ) (r1 r2 : ℝ) (hP : P = 0) (hQ : Q = 10) (rP : r1 = 6) (rQ : r2 = 4) :
    (∃ PR QR SR : ℝ, PR = 6 ∧ QR = 4 ∧ SR = 6) → (R - S = 12) :=
by
  sorry

end RS_segment_length_l351_35138


namespace sqrt_of_sixteen_l351_35113

theorem sqrt_of_sixteen (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := 
sorry

end sqrt_of_sixteen_l351_35113


namespace shoes_difference_l351_35106

theorem shoes_difference :
  let pairs_per_box := 20
  let boxes_A := 8
  let boxes_B := 5 * boxes_A
  let total_pairs_A := boxes_A * pairs_per_box
  let total_pairs_B := boxes_B * pairs_per_box
  total_pairs_B - total_pairs_A = 640 :=
by
  sorry

end shoes_difference_l351_35106


namespace true_proposition_l351_35199

-- Define proposition p
def p : Prop := ∀ x : ℝ, Real.log (x^2 + 4) / Real.log 2 ≥ 2

-- Define proposition q
def q : Prop := ∀ x : ℝ, x ≥ 0 → x^(1/2) ≤ x^(1/2)

-- Theorem: true proposition is p ∨ ¬q
theorem true_proposition : p ∨ ¬q :=
by
  sorry

end true_proposition_l351_35199


namespace johns_age_in_8_years_l351_35133

theorem johns_age_in_8_years :
  let current_age := 18
  let age_five_years_ago := current_age - 5
  let twice_age_five_years_ago := 2 * age_five_years_ago
  current_age + 8 = twice_age_five_years_ago :=
by
  let current_age := 18
  let age_five_years_ago := current_age - 5
  let twice_age_five_years_ago := 2 * age_five_years_ago
  sorry

end johns_age_in_8_years_l351_35133


namespace inequality_for_M_cap_N_l351_35126

def f (x : ℝ) := 2 * |x - 1| + x - 1
def g (x : ℝ) := 16 * x^2 - 8 * x + 1

def M := {x : ℝ | 0 ≤ x ∧ x ≤ 4 / 3}
def N := {x : ℝ | -1 / 4 ≤ x ∧ x ≤ 3 / 4}
def M_cap_N := {x : ℝ | 0 ≤ x ∧ x ≤ 3 / 4}

theorem inequality_for_M_cap_N (x : ℝ) (hx : x ∈ M_cap_N) : x^2 * f x + x * (f x)^2 ≤ 1 / 4 := 
by 
  sorry

end inequality_for_M_cap_N_l351_35126


namespace fraction_of_work_left_l351_35158

theorem fraction_of_work_left 
  (A_days : ℝ) (B_days : ℝ) (work_days : ℝ) 
  (A_work_rate : A_days = 15) 
  (B_work_rate : B_days = 30) 
  (work_duration : work_days = 4)
  : (1 - (work_days * ((1 / A_days) + (1 / B_days)))) = 3 / 5 := 
by
  sorry

end fraction_of_work_left_l351_35158


namespace coin_collection_problem_l351_35170

theorem coin_collection_problem (n : ℕ) 
  (quarters : ℕ := n / 2)
  (half_dollars : ℕ := 2 * (n / 2))
  (value_nickels : ℝ := 0.05 * n)
  (value_quarters : ℝ := 0.25 * (n / 2))
  (value_half_dollars : ℝ := 0.5 * (2 * (n / 2)))
  (total_value : ℝ := value_nickels + value_quarters + value_half_dollars) :
  total_value = 67.5 ∨ total_value = 135 :=
sorry

end coin_collection_problem_l351_35170


namespace Morio_age_when_Michiko_was_born_l351_35125

theorem Morio_age_when_Michiko_was_born (Teresa_age_now : ℕ) (Teresa_age_when_Michiko_born : ℕ) (Morio_age_now : ℕ)
  (hTeresa : Teresa_age_now = 59) (hTeresa_born : Teresa_age_when_Michiko_born = 26) (hMorio : Morio_age_now = 71) :
  Morio_age_now - (Teresa_age_now - Teresa_age_when_Michiko_born) = 38 :=
by
  sorry

end Morio_age_when_Michiko_was_born_l351_35125


namespace percent_time_in_meetings_l351_35119

-- Define the conditions
def work_day_minutes : ℕ := 10 * 60  -- Total minutes in a 10-hour work day is 600 minutes
def first_meeting_minutes : ℕ := 60  -- The first meeting took 60 minutes
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes  -- The second meeting took three times as long as the first meeting

-- Total time spent in meetings
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes  -- 60 + 180 = 240 minutes

-- The task is to prove that Makarla spent 40% of her work day in meetings.
theorem percent_time_in_meetings : (total_meeting_minutes / work_day_minutes : ℚ) * 100 = 40 := by
  sorry

end percent_time_in_meetings_l351_35119


namespace jill_study_hours_l351_35172

theorem jill_study_hours (x : ℕ) (h_condition : x + 2*x + (2*x - 1) = 9) : x = 2 :=
by
  sorry

end jill_study_hours_l351_35172


namespace part1_part2_l351_35115

variable {a b c m t y1 y2 : ℝ}

-- Condition: point (2, m) lies on the parabola y = ax^2 + bx + c where axis of symmetry is x = t
def point_lies_on_parabola (a b c m : ℝ) := m = a * 2^2 + b * 2 + c

-- Condition: axis of symmetry x = t
def axis_of_symmetry (a b t : ℝ) := t = -b / (2 * a)

-- Condition: m = c
theorem part1 (a c : ℝ) (h : m = c) (h₀ : point_lies_on_parabola a (-2 * a) c m) :
  axis_of_symmetry a (-2 * a) 1 :=
by sorry

-- Additional Condition: c < m
def c_lt_m (c m : ℝ) := c < m

-- Points (-1, y1) and (3, y2) lie on the parabola y = ax^2 + bx + c
def points_on_parabola (a b c y1 y2 : ℝ) :=
  y1 = a * (-1)^2 + b * (-1) + c ∧ y2 = a * 3^2 + b * 3 + c

-- Comparison result
theorem part2 (a : ℝ) (h₁ : c_lt_m c m) (h₂ : 2 * a + (-2 * a) > 0) (h₂' : points_on_parabola a (-2 * a) c y1 y2) :
  y2 > y1 :=
by sorry

end part1_part2_l351_35115


namespace negation_relation_l351_35197

def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2 ∨ x > 1

def not_p (x : ℝ) : Prop := x ≥ -1 ∧ x ≤ 1
def not_q (x : ℝ) : Prop := x ≥ -2 ∧ x ≤ 1

theorem negation_relation : (∀ x, not_p x → not_q x) ∧ ¬ (∀ x, not_q x → not_p x) :=
by 
  sorry

end negation_relation_l351_35197


namespace base7_to_base10_conversion_l351_35148

theorem base7_to_base10_conversion (n: ℕ) (H: n = 3652) : 
  (3 * 7^3 + 6 * 7^2 + 5 * 7^1 + 2 * 7^0 = 1360) := by
  sorry

end base7_to_base10_conversion_l351_35148


namespace length_of_bridge_l351_35183

theorem length_of_bridge
  (length_of_train : ℕ)
  (speed_km_per_hr : ℕ)
  (crossing_time_sec : ℕ)
  (h_train_length : length_of_train = 100)
  (h_speed : speed_km_per_hr = 45)
  (h_time : crossing_time_sec = 30) :
  ∃ (length_of_bridge : ℕ), length_of_bridge = 275 :=
by
  -- Convert speed from km/hr to m/s
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600
  -- Total distance the train travels in crossing_time_sec
  let total_distance := speed_m_per_s * crossing_time_sec
  -- Length of the bridge
  let bridge_length := total_distance - length_of_train
  use bridge_length
  -- Skip the detailed proof steps
  sorry

end length_of_bridge_l351_35183


namespace prime_N_k_iff_k_eq_2_l351_35194

-- Define the function to generate the number N_k based on k
def N_k (k : ℕ) : ℕ := (10^(2 * k) - 1) / 99

-- Define the main theorem to prove
theorem prime_N_k_iff_k_eq_2 (k : ℕ) : Nat.Prime (N_k k) ↔ k = 2 :=
by
  sorry

end prime_N_k_iff_k_eq_2_l351_35194


namespace infinite_possible_matrices_A_squared_l351_35111

theorem infinite_possible_matrices_A_squared (A : Matrix (Fin 3) (Fin 3) ℝ) (hA : A^4 = 0) :
  ∃ (S : Set (Matrix (Fin 3) (Fin 3) ℝ)), (∀ B ∈ S, B = A^2) ∧ S.Infinite :=
sorry

end infinite_possible_matrices_A_squared_l351_35111


namespace shooting_challenge_sequences_l351_35184

theorem shooting_challenge_sequences : ∀ (A B C : ℕ), 
  A = 4 → B = 4 → C = 2 →
  (A + B + C = 10) →
  (Nat.factorial (A + B + C) / (Nat.factorial A * Nat.factorial B * Nat.factorial C) = 3150) :=
by
  intros A B C hA hB hC hsum
  sorry

end shooting_challenge_sequences_l351_35184


namespace no_non_similar_triangles_with_geometric_angles_l351_35141

theorem no_non_similar_triangles_with_geometric_angles :
  ¬∃ (a r : ℕ), a > 0 ∧ r > 0 ∧ a ≠ a * r ∧ a ≠ a * r * r ∧ a * r ≠ a * r * r ∧
  a + a * r + a * r * r = 180 :=
by
  sorry

end no_non_similar_triangles_with_geometric_angles_l351_35141


namespace cos_135_eq_neg_sqrt2_div_2_l351_35198

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l351_35198


namespace product_form_l351_35149

theorem product_form (b a : ℤ) :
  (10 * b + a) * (10 * b + 10 - a) = 100 * b * (b + 1) + a * (10 - a) := 
sorry

end product_form_l351_35149


namespace first_nonzero_digit_one_div_139_l351_35152

theorem first_nonzero_digit_one_div_139 :
  ∀ n : ℕ, (n > 0 → (∀ m : ℕ, (m > 0 → (m * 10^n) ∣ (10^n * 1 - 1) ∧ n ∣ (139 * 10 ^ (n + 1)) ∧ 10^(n+1 - 1) * 1 - 1 < 10^n))) :=
sorry

end first_nonzero_digit_one_div_139_l351_35152


namespace speed_of_man_in_still_water_l351_35187

variable (v_m v_s : ℝ)

theorem speed_of_man_in_still_water :
  (v_m + v_s) * 4 = 48 →
  (v_m - v_s) * 6 = 24 →
  v_m = 8 :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end speed_of_man_in_still_water_l351_35187


namespace inequality_logarithms_l351_35131

noncomputable def a : ℝ := Real.log 3.6 / Real.log 2
noncomputable def b : ℝ := Real.log 3.2 / Real.log 4
noncomputable def c : ℝ := Real.log 3.6 / Real.log 4

theorem inequality_logarithms : a > c ∧ c > b :=
by
  -- the proof will be written here
  sorry

end inequality_logarithms_l351_35131


namespace average_speed_l351_35146

theorem average_speed (d1 d2 t1 t2 : ℝ) 
  (h1 : d1 = 100) 
  (h2 : d2 = 80) 
  (h3 : t1 = 1) 
  (h4 : t2 = 1) : 
  (d1 + d2) / (t1 + t2) = 90 := 
by 
  sorry

end average_speed_l351_35146


namespace kirsty_initial_models_l351_35155

theorem kirsty_initial_models 
  (x : ℕ)
  (initial_price : ℝ)
  (increased_price : ℝ)
  (models_bought : ℕ)
  (h_initial_price : initial_price = 0.45)
  (h_increased_price : increased_price = 0.5)
  (h_models_bought : models_bought = 27) 
  (h_total_saved : x * initial_price = models_bought * increased_price) :
  x = 30 :=
by 
  sorry

end kirsty_initial_models_l351_35155


namespace bicycle_weight_l351_35109

theorem bicycle_weight (b s : ℝ) (h1 : 9 * b = 5 * s) (h2 : 4 * s = 160) : b = 200 / 9 :=
by
  sorry

end bicycle_weight_l351_35109


namespace fish_to_corn_value_l351_35114

/-- In an island kingdom, five fish can be traded for three jars of honey, 
    and a jar of honey can be traded for six cobs of corn. 
    Prove that one fish is worth 3.6 cobs of corn. -/

theorem fish_to_corn_value (f h c : ℕ) (h1 : 5 * f = 3 * h) (h2 : h = 6 * c) : f = 18 * c / 5 := by
  sorry

end fish_to_corn_value_l351_35114


namespace prosecutor_cases_knight_or_liar_l351_35134

-- Define the conditions as premises
variable (X : Prop)
variable (Y : Prop)
variable (prosecutor : Prop) -- Truthfulness of the prosecutor (true for knight, false for liar)

-- Define the statements made by the prosecutor
axiom statement1 : X  -- "X is guilty."
axiom statement2 : ¬ (X ∧ Y)  -- "Both X and Y cannot both be guilty."

-- Lean 4 statement for the proof problem
theorem prosecutor_cases_knight_or_liar (h1 : prosecutor) (h2 : ¬prosecutor) : 
  (prosecutor ∧ X ∧ ¬Y) :=
by sorry

end prosecutor_cases_knight_or_liar_l351_35134


namespace count_three_digit_numbers_with_digit_sum_24_l351_35168

-- Define the conditions:
def isThreeDigitNumber (a b c : ℕ) : Prop :=
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (100 * a + 10 * b + c ≥ 100)

def digitSumEquals24 (a b c : ℕ) : Prop :=
  a + b + c = 24

-- State the theorem:
theorem count_three_digit_numbers_with_digit_sum_24 :
  (∃ (count : ℕ), count = 10 ∧ 
   ∀ (a b c : ℕ), isThreeDigitNumber a b c ∧ digitSumEquals24 a b c → (count = 10)) :=
sorry

end count_three_digit_numbers_with_digit_sum_24_l351_35168


namespace correct_analogical_reasoning_l351_35162

-- Definitions of the statements in the problem
def statement_A : Prop := ∀ (a b : ℝ), a * 3 = b * 3 → a = b → a * 0 = b * 0 → a = b
def statement_B : Prop := ∀ (a b c : ℝ), (a + b) * c = a * c + b * c → (a * b) * c = a * c * b * c
def statement_C : Prop := ∀ (a b c : ℝ), (a + b) * c = a * c + b * c → c ≠ 0 → (a + b) / c = a / c + b / c
def statement_D : Prop := ∀ (a b : ℝ) (n : ℕ), (a * b)^n = a^n * b^n → (a + b)^n = a^n + b^n

-- The theorem stating that option C is the only correct analogical reasoning
theorem correct_analogical_reasoning : statement_C ∧ ¬statement_A ∧ ¬statement_B ∧ ¬statement_D := by
  sorry

end correct_analogical_reasoning_l351_35162


namespace total_weight_of_watermelons_l351_35140

theorem total_weight_of_watermelons (w1 w2 : ℝ) (h1 : w1 = 9.91) (h2 : w2 = 4.11) :
  w1 + w2 = 14.02 :=
by
  sorry

end total_weight_of_watermelons_l351_35140


namespace prop_p_necessary_but_not_sufficient_for_prop_q_l351_35142

theorem prop_p_necessary_but_not_sufficient_for_prop_q (x y : ℕ) :
  (x ≠ 1 ∨ y ≠ 3) → (x + y ≠ 4) → ((x+y ≠ 4) → (x ≠ 1 ∨ y ≠ 3)) ∧ ¬ ((x ≠ 1 ∨ y ≠ 3) → (x + y ≠ 4)) :=
by
  sorry

end prop_p_necessary_but_not_sufficient_for_prop_q_l351_35142


namespace profit_percentage_l351_35117

-- Define the selling price
def selling_price : ℝ := 900

-- Define the profit
def profit : ℝ := 100

-- Define the cost price as selling price minus profit
def cost_price : ℝ := selling_price - profit

-- Statement of the profit percentage calculation
theorem profit_percentage : (profit / cost_price) * 100 = 12.5 := by
  sorry

end profit_percentage_l351_35117


namespace amy_initial_money_l351_35169

-- Define the conditions
variable (left_fair : ℕ) (spent : ℕ)

-- Define the proof problem statement
theorem amy_initial_money (h1 : left_fair = 11) (h2 : spent = 4) : left_fair + spent = 15 := 
by sorry

end amy_initial_money_l351_35169


namespace problem1_problem2_l351_35123

-- Definitions of sets A and B
def setA : Set ℝ := { x | x^2 - 8 * x + 15 = 0 }
def setB (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

-- Problem 1: If a = 1/5, B is a subset of A.
theorem problem1 : setB (1 / 5) ⊆ setA := sorry

-- Problem 2: If A ∩ B = B, then C = {0, 1/3, 1/5}.
def setC : Set ℝ := { a | a = 0 ∨ a = 1 / 3 ∨ a = 1 / 5 }

theorem problem2 (a : ℝ) : (setA ∩ setB a = setB a) ↔ (a ∈ setC) := sorry

end problem1_problem2_l351_35123


namespace determine_function_l351_35190

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (1/2) * (x^2 + (1/x)) else 0

theorem determine_function (f: ℝ → ℝ) (h : ∀ x ≠ 0, (1/x) * f (-x) + f (1/x) = x ) :
  ∀ x ≠ 0, f x = (1/2) * (x^2 + (1/x)) :=
by
  sorry

end determine_function_l351_35190


namespace base8_arithmetic_l351_35196

def base8_to_base10 (n : Nat) : Nat :=
  sorry -- Placeholder for base 8 to base 10 conversion

def base10_to_base8 (n : Nat) : Nat :=
  sorry -- Placeholder for base 10 to base 8 conversion

theorem base8_arithmetic (n m : Nat) (h1 : base8_to_base10 45 = n) (h2 : base8_to_base10 76 = m) :
  base10_to_base8 ((n * 2) - m) = 14 :=
by
  sorry

end base8_arithmetic_l351_35196


namespace compare_A_B_l351_35177

noncomputable def A (x : ℝ) := x / (x^2 - x + 1)
noncomputable def B (y : ℝ) := y / (y^2 - y + 1)

theorem compare_A_B (x y : ℝ) (hx : x > y) (hx_val : x = 2.00 * 10^1998 + 4) (hy_val : y = 2.00 * 10^1998 + 2) : 
  A x < B y := 
by 
  sorry

end compare_A_B_l351_35177


namespace total_metal_rods_needed_l351_35176

-- Definitions extracted from the conditions
def metal_sheets_per_panel := 3
def metal_beams_per_panel := 2
def panels := 10
def rods_per_sheet := 10
def rods_per_beam := 4

-- Problem statement: Prove the total number of metal rods required is 380
theorem total_metal_rods_needed :
  (panels * ((metal_sheets_per_panel * rods_per_sheet) + (metal_beams_per_panel * rods_per_beam))) = 380 :=
by
  sorry

end total_metal_rods_needed_l351_35176


namespace install_time_per_window_l351_35105

/-- A new building needed 14 windows. The builder had already installed 8 windows.
    It will take the builder 48 hours to install the rest of the windows. -/
theorem install_time_per_window (total_windows installed_windows remaining_install_time : ℕ)
  (h_total : total_windows = 14)
  (h_installed : installed_windows = 8)
  (h_remaining_time : remaining_install_time = 48) :
  (remaining_install_time / (total_windows - installed_windows)) = 8 :=
by
  -- Insert usual proof steps here
  sorry

end install_time_per_window_l351_35105


namespace max_sum_is_1717_l351_35195

noncomputable def max_arithmetic_sum (a d : ℤ) : ℤ :=
  let n := 34
  let S : ℤ := n * (2*a + (n - 1)*d) / 2
  S

theorem max_sum_is_1717 (a d : ℤ) (h1 : a + 16 * d = 52) (h2 : a + 29 * d = 13) (hd : d = -3) (ha : a = 100) :
  max_arithmetic_sum a d = 1717 :=
by
  unfold max_arithmetic_sum
  rw [hd, ha]
  -- Add the necessary steps to prove max_arithmetic_sum 100 (-3) = 1717
  -- Sorry ensures the theorem can be checked syntactically without proof
  sorry

end max_sum_is_1717_l351_35195


namespace eustace_age_in_3_years_l351_35107

variable (E M : ℕ)

theorem eustace_age_in_3_years
  (h1 : E = 2 * M)
  (h2 : M + 3 = 21) :
  E + 3 = 39 :=
sorry

end eustace_age_in_3_years_l351_35107


namespace infinitely_many_n_l351_35186

theorem infinitely_many_n (S : Set ℕ) :
  (∀ n ∈ S, n > 0 ∧ (n ∣ 2 ^ (2 ^ n + 1) + 1) ∧ ¬ (n ∣ 2 ^ n + 1)) ∧ S.Infinite :=
sorry

end infinitely_many_n_l351_35186


namespace kelly_can_buy_ten_pounds_of_mangoes_l351_35104

theorem kelly_can_buy_ten_pounds_of_mangoes (h : 0.5 * 1.2 = 0.60) : 12 / (2 * 0.60) = 10 :=
  by
    sorry

end kelly_can_buy_ten_pounds_of_mangoes_l351_35104


namespace first_shipment_weight_l351_35137

variable (first_shipment : ℕ)
variable (total_dishes_made : ℕ := 13)
variable (couscous_per_dish : ℕ := 5)
variable (second_shipment : ℕ := 45)
variable (same_day_shipment : ℕ := 13)

theorem first_shipment_weight :
  13 * 5 = 65 → second_shipment ≠ first_shipment → 
  first_shipment + same_day_shipment = 65 →
  first_shipment = 65 :=
by
  sorry

end first_shipment_weight_l351_35137


namespace solution_set_f_l351_35143

def f (x a b : ℝ) : ℝ := (x - 2) * (a * x + b)

theorem solution_set_f (a b : ℝ) (h1 : b = 2 * a) (h2 : 0 < a) :
  {x | f (2 - x) a b > 0} = {x | x < 0 ∨ 4 < x} :=
by
  sorry

end solution_set_f_l351_35143


namespace factor_expression_l351_35156

theorem factor_expression (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) :=
by
  sorry

end factor_expression_l351_35156


namespace range_of_a_l351_35161

variable (x a : ℝ)

def p : Prop := x^2 - 2 * x - 3 ≥ 0

def q : Prop := x^2 - (2 * a - 1) * x + a * (a - 1) ≥ 0

def sufficient_but_not_necessary (p q : Prop) : Prop := 
  (p → q) ∧ ¬(q → p)

theorem range_of_a (a : ℝ) : (∃ x, sufficient_but_not_necessary (p x) (q a x)) → (0 ≤ a ∧ a ≤ 3) := 
sorry

end range_of_a_l351_35161


namespace KellyGamesLeft_l351_35112

def initialGames : ℕ := 121
def gamesGivenAway : ℕ := 99

theorem KellyGamesLeft : initialGames - gamesGivenAway = 22 := by
  sorry

end KellyGamesLeft_l351_35112


namespace initial_mean_of_observations_l351_35179

theorem initial_mean_of_observations (M : ℝ) (h1 : 50 * M + 30 = 50 * 40.66) : M = 40.06 := 
sorry

end initial_mean_of_observations_l351_35179


namespace subway_distance_per_minute_l351_35121

theorem subway_distance_per_minute :
  let total_distance := 120 -- kilometers
  let total_time := 110 -- minutes (1 hour and 50 minutes)
  let bus_time := 70 -- minutes (1 hour and 10 minutes)
  let bus_distance := (14 * 40.8) / 6 -- kilometers
  let subway_distance := total_distance - bus_distance -- kilometers
  let subway_time := total_time - bus_time -- minutes
  let distance_per_minute := subway_distance / subway_time
  distance_per_minute = 0.62 := 
by
  sorry

end subway_distance_per_minute_l351_35121


namespace selection_methods_l351_35103

theorem selection_methods :
  ∃ (ways_with_girls : ℕ), ways_with_girls = Nat.choose 6 4 - Nat.choose 4 4 ∧ ways_with_girls = 14 := by
  sorry

end selection_methods_l351_35103


namespace weight_of_new_person_l351_35182

-- Definitions based on conditions
def average_weight_increase : ℝ := 2.5
def number_of_persons : ℕ := 8
def old_weight : ℝ := 65
def total_weight_increase : ℝ := number_of_persons * average_weight_increase

-- Proposition to prove
theorem weight_of_new_person : (old_weight + total_weight_increase) = 85 := by
  -- add the actual proof here
  sorry

end weight_of_new_person_l351_35182


namespace equal_probabilities_l351_35167

-- Definitions based on the conditions in the problem

def total_parts : ℕ := 160
def first_class_parts : ℕ := 48
def second_class_parts : ℕ := 64
def third_class_parts : ℕ := 32
def substandard_parts : ℕ := 16
def sample_size : ℕ := 20

-- Define the probabilities for each sampling method
def p1 : ℚ := sample_size / total_parts
def p2 : ℚ := (6 : ℚ) / first_class_parts  -- Given the conditions, this will hold for all classes
def p3 : ℚ := 1 / 8

theorem equal_probabilities :
  p1 = p2 ∧ p2 = p3 :=
by
  -- This is the end of the statement as no proof is required
  sorry

end equal_probabilities_l351_35167


namespace solve_equation_l351_35127

-- Define the conditions
def satisfies_equation (n m : ℕ) : Prop :=
  n > 0 ∧ m > 0 ∧ n^5 + n^4 = 7^m - 1

-- Theorem statement
theorem solve_equation : ∀ n m : ℕ, satisfies_equation n m ↔ (n = 2 ∧ m = 2) := 
by { sorry }

end solve_equation_l351_35127


namespace max_value_fraction_l351_35136

theorem max_value_fraction (x y : ℝ) : 
  (2 * x + 3 * y + 2) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 17 :=
by
  sorry

end max_value_fraction_l351_35136


namespace find_angle4_l351_35110

noncomputable def angle_1 := 70
noncomputable def angle_2 := 110
noncomputable def angle_3 := 35
noncomputable def angle_4 := 35

theorem find_angle4 (h1 : angle_1 + angle_2 = 180) (h2 : angle_3 = angle_4) :
  angle_4 = 35 :=
by
  have h3: angle_1 + 70 + 40 = 180 := by sorry
  have h4: angle_2 + angle_3 + angle_4 = 180 := by sorry
  sorry

end find_angle4_l351_35110


namespace correct_options_l351_35173

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + 1

def option_2 : Prop := ∃ x : ℝ, f (x) = 0 ∧ x = Real.pi / 3
def option_3 : Prop := ∀ T > 0, (∀ x : ℝ, f (x) = f (x + T)) → T = Real.pi
def option_5 : Prop := ∀ x : ℝ, f (x - Real.pi / 6) = f (-(x - Real.pi / 6))

theorem correct_options :
  option_2 ∧ option_3 ∧ option_5 :=
by
  sorry

end correct_options_l351_35173


namespace train_length_250_meters_l351_35153

open Real

noncomputable def speed_in_ms (speed_km_hr: ℝ): ℝ :=
  speed_km_hr * (1000 / 3600)

noncomputable def length_of_train (speed: ℝ) (time: ℝ): ℝ :=
  speed * time

theorem train_length_250_meters (speed_km_hr: ℝ) (time_seconds: ℝ) :
  speed_km_hr = 40 → time_seconds = 22.5 → length_of_train (speed_in_ms speed_km_hr) time_seconds = 250 :=
by
  intros
  sorry

end train_length_250_meters_l351_35153


namespace arrange_leopards_correct_l351_35188

-- Definitions for conditions
def num_shortest : ℕ := 3
def total_leopards : ℕ := 9
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Calculation of total ways to arrange given conditions
def arrange_leopards (num_shortest : ℕ) (total_leopards : ℕ) : ℕ :=
  let choose2short := (num_shortest * (num_shortest - 1)) / 2
  let arrange2short := 2 * factorial (total_leopards - num_shortest)
  choose2short * arrange2short * factorial (total_leopards - num_shortest)

theorem arrange_leopards_correct :
  arrange_leopards num_shortest total_leopards = 30240 := by
  sorry

end arrange_leopards_correct_l351_35188


namespace price_of_tea_mixture_l351_35163

theorem price_of_tea_mixture 
  (p1 p2 p3 : ℝ) 
  (q1 q2 q3 : ℝ) 
  (h_p1 : p1 = 126) 
  (h_p2 : p2 = 135) 
  (h_p3 : p3 = 173.5) 
  (h_q1 : q1 = 1) 
  (h_q2 : q2 = 1) 
  (h_q3 : q3 = 2) : 
  (p1 * q1 + p2 * q2 + p3 * q3) / (q1 + q2 + q3) = 152 := 
by 
  sorry

end price_of_tea_mixture_l351_35163


namespace isosceles_triangle_sides_l351_35102

theorem isosceles_triangle_sides (a b c : ℝ) (hb : b = 3) (hc : a = 3 ∨ c = 3) (hperim : a + b + c = 7) :
  a = 2 ∨ a = 3 ∨ c = 2 ∨ c = 3 :=
by
  sorry

end isosceles_triangle_sides_l351_35102


namespace sum_of_numbers_l351_35171

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def tens_digit_zero (n : ℕ) : Prop := (n / 10) % 10 = 0
def units_digit_nonzero (n : ℕ) : Prop := n % 10 ≠ 0
def same_units_digits (m n : ℕ) : Prop := m % 10 = n % 10

theorem sum_of_numbers (a b c : ℕ)
  (h1 : is_perfect_square a) (h2 : is_perfect_square b) (h3 : is_perfect_square c)
  (h4 : tens_digit_zero a) (h5 : tens_digit_zero b) (h6 : tens_digit_zero c)
  (h7 : units_digit_nonzero a) (h8 : units_digit_nonzero b) (h9 : units_digit_nonzero c)
  (h10 : same_units_digits b c)
  (h11 : a % 10 % 2 = 0) :
  a + b + c = 14612 :=
sorry

end sum_of_numbers_l351_35171


namespace expression_varies_l351_35128

noncomputable def expr (x : ℝ) : ℝ := (3 * x^2 - 2 * x - 5) / ((x + 2) * (x - 3)) - (5 + x) / ((x + 2) * (x - 3))

theorem expression_varies (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) : ∃ y : ℝ, expr x = y ∧ ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → expr x₁ ≠ expr x₂ :=
by
  sorry

end expression_varies_l351_35128


namespace distance_from_house_to_work_l351_35160

-- Definitions for the conditions
variables (D : ℝ) (speed_to_work speed_back_work : ℝ) (time_to_work time_back_work total_time : ℝ)

-- Specific conditions in the problem
noncomputable def conditions : Prop :=
  (speed_back_work = 20) ∧
  (speed_to_work = speed_back_work / 2) ∧
  (time_to_work = D / speed_to_work) ∧
  (time_back_work = D / speed_back_work) ∧
  (total_time = 6) ∧
  (time_to_work + time_back_work = total_time)

-- The statement to prove the distance D is 40 km given the conditions
theorem distance_from_house_to_work (h : conditions D speed_to_work speed_back_work time_to_work time_back_work total_time) : D = 40 :=
sorry

end distance_from_house_to_work_l351_35160


namespace coplanar_k_values_l351_35180

noncomputable def coplanar_lines_possible_k (k : ℝ) : Prop :=
  ∃ (t u : ℝ), (2 + t = 1 + k * u) ∧ (3 + t = 4 + 2 * u) ∧ (4 - k * t = 5 + u)

theorem coplanar_k_values :
  ∀ k : ℝ, coplanar_lines_possible_k k ↔ (k = 0 ∨ k = -3) :=
by
  sorry

end coplanar_k_values_l351_35180


namespace problem_l351_35166

theorem problem (a b : ℚ) (h : a / b = 6 / 5) : (5 * a + 4 * b) / (5 * a - 4 * b) = 5 := 
by 
  sorry

end problem_l351_35166


namespace area_covered_by_congruent_rectangles_l351_35108

-- Definitions of conditions
def length_AB : ℕ := 12
def width_AD : ℕ := 8
def area_rect (l w : ℕ) : ℕ := l * w

-- Center of the first rectangle
def center_ABCD : ℕ × ℕ := (length_AB / 2, width_AD / 2)

-- Proof statement
theorem area_covered_by_congruent_rectangles 
  (length_ABCD length_EFGH width_ABCD width_EFGH : ℕ)
  (congruent : length_ABCD = length_EFGH ∧ width_ABCD = width_EFGH)
  (center_E : ℕ × ℕ)
  (H_center_E : center_E = center_ABCD) :
  area_rect length_ABCD width_ABCD + area_rect length_EFGH width_EFGH - length_ABCD * width_ABCD / 2 = 168 := by
  sorry

end area_covered_by_congruent_rectangles_l351_35108


namespace cadence_total_earnings_l351_35189

/-- Cadence's total earnings in both companies. -/
def total_earnings (old_salary_per_month new_salary_per_month : ℕ) (old_company_months new_company_months : ℕ) : ℕ :=
  (old_salary_per_month * old_company_months) + (new_salary_per_month * new_company_months)

theorem cadence_total_earnings :
  let old_salary_per_month := 5000
  let old_company_years := 3
  let months_per_year := 12
  let old_company_months := old_company_years * months_per_year
  let new_salary_per_month := old_salary_per_month + (old_salary_per_month * 20 / 100)
  let new_company_extra_months := 5
  let new_company_months := old_company_months + new_company_extra_months
  total_earnings old_salary_per_month new_salary_per_month old_company_months new_company_months = 426000 := by
sorry

end cadence_total_earnings_l351_35189


namespace solution_set_of_inequality_l351_35130

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) / (x - 3) ≤ 0} = {x : ℝ | 1 ≤ x ∧ x < 3} := 
by
  sorry

end solution_set_of_inequality_l351_35130


namespace eggs_in_basket_empty_l351_35175

theorem eggs_in_basket_empty (a : ℕ) : 
  let remaining_after_first := a - (a / 2 + 1 / 2)
  let remaining_after_second := remaining_after_first - (remaining_after_first / 2 + 1 / 2)
  let remaining_after_third := remaining_after_second - (remaining_after_second / 2 + 1 / 2)
  (remaining_after_first = a / 2 - 1 / 2) → 
  (remaining_after_second = remaining_after_first / 2 - 1 / 2) → 
  (remaining_after_third = remaining_after_second / 2 -1 / 2) → 
  (remaining_after_third = 0) → 
  (a = 7) := sorry

end eggs_in_basket_empty_l351_35175


namespace complement_intersection_l351_35174

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3, 4}

-- Define set B
def B : Set ℕ := {2, 3}

-- Define the complement of A with respect to U
def complement_U (s : Set ℕ) : Set ℕ := {x ∈ U | x ∉ s}

-- Define the statement to be proven
theorem complement_intersection :
  (complement_U A ∩ B) = {2} :=
by
  sorry

end complement_intersection_l351_35174


namespace parabola_properties_l351_35101

def parabola (a b x : ℝ) : ℝ :=
  a * x ^ 2 + b * x - 4

theorem parabola_properties :
  ∃ (a b : ℝ), (a = 2) ∧ (b = 2) ∧
  parabola a b (-2) = 0 ∧ 
  parabola a b (-1) = -4 ∧ 
  parabola a b 0 = -4 ∧ 
  parabola a b 1 = 0 ∧ 
  parabola a b 2 = 8 ∧ 
  parabola a b (-3) = 8 ∧ 
  (0, -4) ∈ {(x, y) | ∃ a b, y = parabola a b x} :=
sorry

end parabola_properties_l351_35101


namespace number_of_sheets_l351_35185

theorem number_of_sheets
  (n : ℕ)
  (h₁ : 2 * n + 2 = 74) :
  n / 4 = 9 :=
by
  sorry

end number_of_sheets_l351_35185


namespace steve_assignments_fraction_l351_35150

theorem steve_assignments_fraction (h_sleep: ℝ) (h_school: ℝ) (h_family: ℝ) (total_hours: ℝ) : 
  h_sleep = 1/3 ∧ 
  h_school = 1/6 ∧ 
  h_family = 10 ∧ 
  total_hours = 24 → 
  (2 / total_hours = 1 / 12) :=
by
  intros h
  sorry

end steve_assignments_fraction_l351_35150


namespace max_n_for_factorable_polynomial_l351_35154

theorem max_n_for_factorable_polynomial : 
  ∃ n : ℤ, (∀ A B : ℤ, AB = 108 → n = 6 * B + A) ∧ n = 649 :=
by
  sorry

end max_n_for_factorable_polynomial_l351_35154


namespace intersection_M_N_l351_35181

noncomputable def M : Set ℝ := {x | x^2 + x - 6 < 0}
noncomputable def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N :
  {x : ℝ | M x ∧ N x } = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l351_35181


namespace length_of_second_train_is_correct_l351_35178

noncomputable def convert_kmph_to_mps (speed_kmph: ℕ) : ℝ :=
  speed_kmph * (1000 / 3600)

def train_lengths_and_time
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℕ)
  (speed_second_train_kmph : ℕ)
  (time_to_cross : ℝ)
  (length_second_train : ℝ) : Prop :=
  let speed_first_train_mps := convert_kmph_to_mps speed_first_train_kmph
  let speed_second_train_mps := convert_kmph_to_mps speed_second_train_kmph
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance := relative_speed * time_to_cross
  total_distance = length_first_train + length_second_train

theorem length_of_second_train_is_correct :
  train_lengths_and_time 260 120 80 9 239.95 :=
by
  sorry

end length_of_second_train_is_correct_l351_35178


namespace problem_inequality_l351_35151

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem problem_inequality (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 ≠ x2) :
  (f x2 - f x1) / (x2 - x1) < (1 + Real.log ((x1 + x2) / 2)) :=
sorry

end problem_inequality_l351_35151


namespace find_sum_l351_35165

-- Defining the conditions of the problem
variables (P r t : ℝ) 
theorem find_sum 
  (h1 : (P * r * t) / 100 = 88) 
  (h2 : (P * r * t) / (100 + (r * t)) = 80) 
  : P = 880 := 
sorry

end find_sum_l351_35165
