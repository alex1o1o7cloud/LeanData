import Mathlib

namespace determine_a_l773_77346

def A (a : ℝ) : Set ℝ := {1, a}
def B (a : ℝ) : Set ℝ := {a^2}

theorem determine_a (a : ℝ) (A_union_B_eq_A : A a ∪ B a = A a) : a = -1 ∨ a = 0 := by
  sorry

end determine_a_l773_77346


namespace members_playing_both_sports_l773_77306

theorem members_playing_both_sports 
    (N : ℕ) (B : ℕ) (T : ℕ) (D : ℕ)
    (hN : N = 30) (hB : B = 18) (hT : T = 19) (hD : D = 2) :
    N - D = 28 ∧ B + T = 37 ∧ B + T - (N - D) = 9 :=
by
  sorry

end members_playing_both_sports_l773_77306


namespace tangent_parallel_to_line_l773_77338

def f (x : ℝ) : ℝ := x ^ 3 + x - 2

theorem tangent_parallel_to_line (x y : ℝ) : 
  (y = 4 * x - 1) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) :=
by
  sorry

end tangent_parallel_to_line_l773_77338


namespace digit_difference_is_one_l773_77331

theorem digit_difference_is_one {p q : ℕ} (h : 1 ≤ p ∧ p ≤ 9 ∧ 0 ≤ q ∧ q ≤ 9 ∧ p ≠ q)
  (digits_distinct : ∀ n ∈ [p, q], ∀ m ∈ [p, q], n ≠ m)
  (interchange_effect : 10 * p + q - (10 * q + p) = 9) : p - q = 1 :=
sorry

end digit_difference_is_one_l773_77331


namespace largest_of_a_b_c_l773_77374

noncomputable def a : ℝ := 1 / 2
noncomputable def b : ℝ := Real.log 3 / Real.log 4
noncomputable def c : ℝ := Real.sin (Real.pi / 8)

theorem largest_of_a_b_c : b = max (max a b) c :=
by
  have ha : a = 1 / 2 := rfl
  have hb : b = Real.log 3 / Real.log 4 := rfl
  have hc : c = Real.sin (Real.pi / 8) := rfl
  sorry

end largest_of_a_b_c_l773_77374


namespace measure_of_B_l773_77390

theorem measure_of_B (a b : ℝ) (A B : ℝ) (angleA_nonneg : 0 < A ∧ A < 180) (angleB_nonneg : 0 < B ∧ B < 180)
    (a_eq : a = 1) (b_eq : b = Real.sqrt 3) (A_eq : A = 30) :
    B = 60 :=
by
  sorry

end measure_of_B_l773_77390


namespace sum_of_coeffs_eq_225_l773_77334

/-- The sum of the coefficients of all terms in the expansion
of (C_x + C_x^2 + C_x^3 + C_x^4)^2 is equal to 225. -/
theorem sum_of_coeffs_eq_225 (C_x : ℝ) : 
  (C_x + C_x^2 + C_x^3 + C_x^4)^2 = 225 :=
sorry

end sum_of_coeffs_eq_225_l773_77334


namespace min_value_of_expression_l773_77341

theorem min_value_of_expression
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hlines : (∀ x y : ℝ, x + (a-4) * y + 1 = 0) ∧ (∀ x y : ℝ, 2 * b * x + y - 2 = 0) ∧ (∀ x y : ℝ, (x + (a-4) * y + 1 = 0) ∧ (2 * b * x + y - 2 = 0) → -1 * 1 / (a-4) * -2 * b = 1)) :
  ∃ (min_val : ℝ), min_val = (9/5) ∧ min_val = (a + 2)/(a + 1) + 1/(2 * b) :=
by
  sorry

end min_value_of_expression_l773_77341


namespace net_profit_expression_and_break_even_point_l773_77321

-- Definitions based on the conditions in a)
def investment : ℝ := 600000
def initial_expense : ℝ := 80000
def expense_increase : ℝ := 20000
def annual_income : ℝ := 260000

-- Define the net profit function as given in the solution
def net_profit (n : ℕ) : ℝ :=
  - (n : ℝ)^2 + 19 * n - 60

-- Statement about the function and where the dealer starts making profit
theorem net_profit_expression_and_break_even_point :
  net_profit n = - (n : ℝ)^2 + 19 * n - 60 ∧ ∃ n ≥ 5, net_profit n > 0 :=
sorry

end net_profit_expression_and_break_even_point_l773_77321


namespace distance_from_apex_l773_77364

theorem distance_from_apex (a₁ a₂ : ℝ) (d : ℝ)
  (ha₁ : a₁ = 150 * Real.sqrt 3)
  (ha₂ : a₂ = 300 * Real.sqrt 3)
  (hd : d = 10) :
  ∃ h : ℝ, h = 10 * Real.sqrt 2 :=
by
  sorry

end distance_from_apex_l773_77364


namespace percentage_defective_units_shipped_l773_77328

noncomputable def defective_percent : ℝ := 0.07
noncomputable def shipped_percent : ℝ := 0.05

theorem percentage_defective_units_shipped :
  defective_percent * shipped_percent * 100 = 0.35 :=
by
  -- Proof body here
  sorry

end percentage_defective_units_shipped_l773_77328


namespace det_B_squared_sub_3B_eq_10_l773_77352

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2, 3], ![2, 2]]

theorem det_B_squared_sub_3B_eq_10 : 
  Matrix.det (B * B - 3 • B) = 10 := by
  sorry

end det_B_squared_sub_3B_eq_10_l773_77352


namespace number_of_badminton_players_l773_77323

-- Definitions based on the given conditions
variable (Total_members : ℕ := 30)
variable (Tennis_players : ℕ := 19)
variable (No_sport_players : ℕ := 3)
variable (Both_sport_players : ℕ := 9)

-- The goal is to prove the number of badminton players is 17
theorem number_of_badminton_players :
  ∀ (B : ℕ), Total_members = B + Tennis_players - Both_sport_players + No_sport_players → B = 17 :=
by
  intro B
  intro h
  sorry

end number_of_badminton_players_l773_77323


namespace distance_between_stripes_l773_77365

/-- Given a crosswalk parallelogram with curbs 60 feet apart, a base of 20 feet, 
and each stripe of length 50 feet, show that the distance between the stripes is 24 feet. -/
theorem distance_between_stripes (h : Real) (b : Real) (s : Real) : h = 60 ∧ b = 20 ∧ s = 50 → (b * h) / s = 24 :=
by
  sorry

end distance_between_stripes_l773_77365


namespace milk_leftover_after_milkshakes_l773_77325

theorem milk_leftover_after_milkshakes
  (milk_per_milkshake : ℕ)
  (ice_cream_per_milkshake : ℕ)
  (total_milk : ℕ)
  (total_ice_cream : ℕ)
  (milkshakes_made : ℕ)
  (milk_used : ℕ)
  (milk_left : ℕ) :
  milk_per_milkshake = 4 →
  ice_cream_per_milkshake = 12 →
  total_milk = 72 →
  total_ice_cream = 192 →
  milkshakes_made = total_ice_cream / ice_cream_per_milkshake →
  milk_used = milkshakes_made * milk_per_milkshake →
  milk_left = total_milk - milk_used →
  milk_left = 8 :=
by
  intros
  sorry

end milk_leftover_after_milkshakes_l773_77325


namespace total_play_time_in_hours_l773_77326

def football_time : ℕ := 60
def basketball_time : ℕ := 60

theorem total_play_time_in_hours : (football_time + basketball_time) / 60 = 2 := by
  sorry

end total_play_time_in_hours_l773_77326


namespace least_positive_integer_mod_conditions_l773_77340

theorem least_positive_integer_mod_conditions :
  ∃ N : ℕ, (N % 4 = 3) ∧ (N % 5 = 4) ∧ (N % 6 = 5) ∧ (N % 7 = 6) ∧ (N % 11 = 10) ∧ N = 4619 :=
by
  sorry

end least_positive_integer_mod_conditions_l773_77340


namespace find_T_l773_77397

variable (a b c T : ℕ)

theorem find_T (h1 : a + b + c = 84) (h2 : a - 5 = T) (h3 : b + 9 = T) (h4 : 5 * c = T) : T = 40 :=
sorry

end find_T_l773_77397


namespace part1_part2_l773_77342

-- Part 1: Definition of "consecutive roots quadratic equation"
def consecutive_roots (a b : ℤ) : Prop := a = b + 1 ∨ b = a + 1

-- Statement that for some k and constant term, the roots of the quadratic form consecutive roots
theorem part1 (k : ℤ) : consecutive_roots 7 8 → k = -15 → (∀ x : ℤ, x^2 + k * x + 56 = 0 → x = 7 ∨ x = 8) :=
by
  sorry

-- Part 2: Generalizing to the nth equation
theorem part2 (n : ℕ) : 
  (∀ x : ℤ, x^2 - (2 * n - 1) * x + n * (n - 1) = 0 → x = n ∨ x = n - 1) :=
by
  sorry

end part1_part2_l773_77342


namespace prob_three_cards_in_sequence_l773_77381

theorem prob_three_cards_in_sequence : 
  let total_cards := 52
  let spades_count := 13
  let hearts_count := 13
  let sequence_prob := (spades_count / total_cards) * (hearts_count / (total_cards - 1)) * ((spades_count - 1) / (total_cards - 2))
  sequence_prob = (78 / 5100) :=
by
  sorry

end prob_three_cards_in_sequence_l773_77381


namespace sum_base8_l773_77348

theorem sum_base8 (a b c : ℕ) (h₁ : a = 7*8^2 + 7*8 + 7)
                           (h₂ : b = 7*8 + 7)
                           (h₃ : c = 7) :
  a + b + c = 1*8^3 + 1*8^2 + 0*8 + 5 :=
by
  sorry

end sum_base8_l773_77348


namespace mrs_oaklyn_rugs_l773_77375

theorem mrs_oaklyn_rugs (buying_price selling_price total_profit : ℕ) (h1 : buying_price = 40) (h2 : selling_price = 60) (h3 : total_profit = 400) : 
  ∃ (num_rugs : ℕ), num_rugs = 20 :=
by
  sorry

end mrs_oaklyn_rugs_l773_77375


namespace length_inequality_l773_77387

noncomputable def l_a (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def l_b (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def l_c (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def perimeter (A B C : ℝ) : ℝ :=
  A + B + C

theorem length_inequality (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0) :
  (l_a A B C * l_b A B C * l_c A B C) / (perimeter A B C)^3 ≤ 1 / 64 :=
by
  sorry

end length_inequality_l773_77387


namespace herder_bulls_l773_77393

theorem herder_bulls (total_bulls : ℕ) (herder_fraction : ℚ) (claims : total_bulls = 70) (fraction_claim : herder_fraction = (2/3) * (1/3)) : herder_fraction * (total_bulls : ℚ) = 315 :=
by sorry

end herder_bulls_l773_77393


namespace find_x_l773_77359

theorem find_x (x : ℝ) (h : 0.25 * x = 0.12 * 1500 - 15) : x = 660 :=
by
  -- Proof goes here
  sorry

end find_x_l773_77359


namespace complete_remaining_parts_l773_77369

-- Define the main conditions and the proof goal in Lean 4
theorem complete_remaining_parts :
  ∀ (total_parts processed_parts workers days_off remaining_parts_per_day),
  total_parts = 735 →
  processed_parts = 135 →
  workers = 5 →
  days_off = 1 →
  remaining_parts_per_day = total_parts - processed_parts →
  (workers * 2 - days_off) * 15 = processed_parts →
  remaining_parts_per_day / (workers * 15) = 8 :=
by
  -- Starting the proof
  intros total_parts processed_parts workers days_off remaining_parts_per_day
  intros h_total_parts h_processed_parts h_workers h_days_off h_remaining_parts_per_day h_productivity
  -- Replace given variables with their values
  sorry

end complete_remaining_parts_l773_77369


namespace middle_digit_base_7_of_reversed_base_9_l773_77382

noncomputable def middle_digit_of_number_base_7 (N : ℕ) : ℕ :=
  let x := (N / 81) % 9  -- Extract the first digit in base-9
  let y := (N / 9) % 9   -- Extract the middle digit in base-9
  let z := N % 9         -- Extract the last digit in base-9
  -- Given condition: 81x + 9y + z = 49z + 7y + x
  let eq1 := 81 * x + 9 * y + z
  let eq2 := 49 * z + 7 * y + x
  let condition := eq1 = eq2 ∧ 0 ≤ y ∧ y < 7 -- y is a digit in base-7
  if condition then y else sorry

theorem middle_digit_base_7_of_reversed_base_9 (N : ℕ) :
  (∃ (x y z : ℕ), x < 9 ∧ y < 9 ∧ z < 9 ∧
  N = 81 * x + 9 * y + z ∧ N = 49 * z + 7 * y + x) → middle_digit_of_number_base_7 N = 0 :=
  by sorry

end middle_digit_base_7_of_reversed_base_9_l773_77382


namespace find_larger_number_l773_77301

theorem find_larger_number (x y : ℝ) (h1 : 4 * y = 6 * x) (h2 : x + y = 36) : y = 21.6 :=
by
  sorry

end find_larger_number_l773_77301


namespace equal_partition_of_weights_l773_77366

theorem equal_partition_of_weights 
  (weights : Fin 2009 → ℕ) 
  (h1 : ∀ i : Fin 2008, (weights i + 1 = weights (i + 1)) ∨ (weights i = weights (i + 1) + 1))
  (h2 : ∀ i : Fin 2009, weights i ≤ 1000)
  (h3 : (Finset.univ.sum weights) % 2 = 0) :
  ∃ (A B : Finset (Fin 2009)), (A ∪ B = Finset.univ ∧ A ∩ B = ∅ ∧ A.sum weights = B.sum weights) :=
sorry

end equal_partition_of_weights_l773_77366


namespace max_possible_x_l773_77367

noncomputable section

def tan_deg (x : ℕ) : ℝ := Real.tan (x * Real.pi / 180)

theorem max_possible_x (x y : ℕ) (h₁ : tan_deg x - tan_deg y = 1 + tan_deg x * tan_deg y)
  (h₂ : tan_deg x * tan_deg y = 1) (h₃ : x = 98721) : x = 98721 := sorry

end max_possible_x_l773_77367


namespace sum_first_5_terms_arithmetic_l773_77345

variable {a : ℕ → ℝ} -- Defining a sequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom a2_eq_1 : a 2 = 1
axiom a4_eq_5 : a 4 = 5

-- Theorem statement
theorem sum_first_5_terms_arithmetic (h_arith : is_arithmetic_sequence a) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end sum_first_5_terms_arithmetic_l773_77345


namespace students_taking_one_language_l773_77307

-- Definitions based on the conditions
def french_class_students : ℕ := 21
def spanish_class_students : ℕ := 21
def both_languages_students : ℕ := 6
def total_students : ℕ := french_class_students + spanish_class_students - both_languages_students

-- The theorem we want to prove
theorem students_taking_one_language :
    total_students = 36 :=
by
  -- Add the proof here
  sorry

end students_taking_one_language_l773_77307


namespace BC_length_l773_77330

theorem BC_length (AD BC MN : ℝ) (h1 : AD = 2) (h2 : MN = 6) (h3 : MN = 0.5 * (AD + BC)) : BC = 10 :=
by
  sorry

end BC_length_l773_77330


namespace concert_ticket_cost_l773_77343

theorem concert_ticket_cost :
  ∀ (x : ℝ), 
    (12 * x - 2 * 0.05 * x = 476) → 
    x = 40 :=
by
  intros x h
  sorry

end concert_ticket_cost_l773_77343


namespace find_multiple_of_pages_l773_77335

-- Definitions based on conditions
def beatrix_pages : ℕ := 704
def cristobal_extra_pages : ℕ := 1423
def cristobal_pages (x : ℕ) : ℕ := x * beatrix_pages + 15

-- Proposition to prove the multiple x equals 2
theorem find_multiple_of_pages (x : ℕ) (h : cristobal_pages x = beatrix_pages + cristobal_extra_pages) : x = 2 :=
  sorry

end find_multiple_of_pages_l773_77335


namespace cucumbers_for_20_apples_l773_77303

theorem cucumbers_for_20_apples (A B C : ℝ) (h1 : 10 * A = 5 * B) (h2 : 3 * B = 4 * C) :
  20 * A = 40 / 3 * C :=
by
  sorry

end cucumbers_for_20_apples_l773_77303


namespace wire_length_l773_77350

theorem wire_length (S L : ℝ) (h1 : S = 10) (h2 : S = (2 / 5) * L) : S + L = 35 :=
by
  sorry

end wire_length_l773_77350


namespace chord_division_ratio_l773_77312

theorem chord_division_ratio (R AB PO DP PC x AP PB : ℝ)
  (hR : R = 11)
  (hAB : AB = 18)
  (hPO : PO = 7)
  (hDP : DP = R - PO)
  (hPC : PC = R + PO)
  (hPower : AP * PB = DP * PC)
  (hChord : AP + PB = AB) :
  AP = 12 ∧ PB = 6 ∨ AP = 6 ∧ PB = 12 :=
by
  -- Structure of the theorem is provided.
  -- Proof steps are skipped and marked with sorry.
  sorry

end chord_division_ratio_l773_77312


namespace david_more_pushups_than_zachary_l773_77354

def zacharyPushUps : ℕ := 59
def davidPushUps : ℕ := 78

theorem david_more_pushups_than_zachary :
  davidPushUps - zacharyPushUps = 19 :=
by
  sorry

end david_more_pushups_than_zachary_l773_77354


namespace jett_profit_l773_77319

def initial_cost : ℕ := 600
def vaccination_cost : ℕ := 500
def daily_food_cost : ℕ := 20
def number_of_days : ℕ := 40
def selling_price : ℕ := 2500

def total_expenses : ℕ := initial_cost + vaccination_cost + daily_food_cost * number_of_days
def profit : ℕ := selling_price - total_expenses

theorem jett_profit : profit = 600 :=
by
  -- Completed proof steps
  sorry

end jett_profit_l773_77319


namespace mushroom_collectors_l773_77376

theorem mushroom_collectors :
  ∃ (n m : ℕ), 13 * n - 10 * m = 2 ∧ 9 ≤ n ∧ n ≤ 15 ∧ 11 ≤ m ∧ m ≤ 20 ∧ n = 14 ∧ m = 18 := by sorry

end mushroom_collectors_l773_77376


namespace y_at_x_equals_2sqrt3_l773_77308

theorem y_at_x_equals_2sqrt3 (k : ℝ) (y : ℝ → ℝ)
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
sorry

end y_at_x_equals_2sqrt3_l773_77308


namespace total_goals_scored_l773_77349

-- Definitions based on the problem conditions
def kickers_first_period_goals : ℕ := 2
def kickers_second_period_goals : ℕ := 2 * kickers_first_period_goals
def spiders_first_period_goals : ℕ := kickers_first_period_goals / 2
def spiders_second_period_goals : ℕ := 2 * kickers_second_period_goals

-- The theorem we need to prove
theorem total_goals_scored : 
  kickers_first_period_goals + kickers_second_period_goals +
  spiders_first_period_goals + spiders_second_period_goals = 15 := 
by
  -- proof steps will go here
  sorry

end total_goals_scored_l773_77349


namespace biased_die_sum_is_odd_l773_77302

def biased_die_probabilities : Prop :=
  let p_odd := 1 / 3
  let p_even := 2 / 3
  let scenarios := [
    (1/3) * (2/3)^2,
    (1/3)^3
  ]
  let sum := scenarios.sum
  sum = 13 / 27

theorem biased_die_sum_is_odd :
  biased_die_probabilities := by
    sorry

end biased_die_sum_is_odd_l773_77302


namespace marble_selection_l773_77372

-- Definitions based on the conditions
def total_marbles : ℕ := 15
def special_marbles : ℕ := 4
def other_marbles : ℕ := total_marbles - special_marbles

-- Define combination function for ease of use in the theorem
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Statement of the theorem based on the question and the correct answer
theorem marble_selection : combination other_marbles 4 * special_marbles = 1320 := by
  -- Define specific values based on the problem
  have other_marbles_val : other_marbles = 11 := rfl
  have comb_11_4 : combination 11 4 = 330 := by
    rw [combination]
    rfl
  rw [other_marbles_val, comb_11_4]
  norm_num
  sorry

end marble_selection_l773_77372


namespace value_of_a_even_function_monotonicity_on_interval_l773_77304

noncomputable def f (x : ℝ) := (1 / x^2) + 0 * x

theorem value_of_a_even_function 
  (f : ℝ → ℝ) 
  (h : ∀ x, f (-x) = f x) : 
  (∃ a : ℝ, ∀ x, f x = (1 / x^2) + a * x) → a = 0 := by
  -- Placeholder for the proof
  sorry

theorem monotonicity_on_interval 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (1 / x^2) + 0 * x) 
  (h2 : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f x1 > f x2) : 
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f x1 > f x2 := by
  -- Placeholder for the proof
  sorry

end value_of_a_even_function_monotonicity_on_interval_l773_77304


namespace max_number_of_kids_on_school_bus_l773_77329

-- Definitions based on the conditions from the problem
def totalRowsLowerDeck : ℕ := 15
def totalRowsUpperDeck : ℕ := 10
def capacityLowerDeckRow : ℕ := 5
def capacityUpperDeckRow : ℕ := 3
def reservedSeatsLowerDeck : ℕ := 10
def staffMembers : ℕ := 4

-- The total capacity of the lower and upper decks
def totalCapacityLowerDeck := totalRowsLowerDeck * capacityLowerDeckRow
def totalCapacityUpperDeck := totalRowsUpperDeck * capacityUpperDeckRow
def totalCapacity := totalCapacityLowerDeck + totalCapacityUpperDeck

-- The maximum number of different kids that can ride the bus
def maxKids := totalCapacity - reservedSeatsLowerDeck - staffMembers

theorem max_number_of_kids_on_school_bus : maxKids = 91 := 
by 
  -- Step-by-step proof not required for this task
  sorry

end max_number_of_kids_on_school_bus_l773_77329


namespace test_question_count_l773_77313

theorem test_question_count :
  ∃ (x : ℕ), 
    (20 / x: ℚ) > 0.60 ∧ 
    (20 / x: ℚ) < 0.70 ∧ 
    (4 ∣ x) ∧ 
    x = 32 := 
by
  sorry

end test_question_count_l773_77313


namespace map_distance_to_real_distance_l773_77363

theorem map_distance_to_real_distance (d_map : ℝ) (scale : ℝ) (d_real : ℝ) 
    (h1 : d_map = 7.5) (h2 : scale = 8) : d_real = 60 :=
by
  sorry

end map_distance_to_real_distance_l773_77363


namespace find_c_l773_77392

-- Define the necessary conditions for the circle equation and the radius
variable (c : ℝ)

-- The given conditions
def circle_eq := ∀ (x y : ℝ), x^2 + 8*x + y^2 - 6*y + c = 0
def radius_five := (∀ (h k r : ℝ), r = 5 → ∃ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2)

theorem find_c (h k r : ℝ) (r_eq : r = 5) : c = 0 :=
by {
  sorry
}

end find_c_l773_77392


namespace divisor_of_70th_number_l773_77318

-- Define the conditions
def s (d : ℕ) (n : ℕ) : ℕ := n * d + 5

-- Theorem stating the given problem
theorem divisor_of_70th_number (d : ℕ) (h : s d 70 = 557) : d = 8 :=
by
  -- The proof is to be filled in later. 
  -- Now, just create the structure.
  sorry

end divisor_of_70th_number_l773_77318


namespace volume_of_pool_l773_77386

theorem volume_of_pool :
  let diameter := 60
  let radius := diameter / 2
  let height_shallow := 3
  let height_deep := 15
  let height_total := height_shallow + height_deep
  let volume_cylinder := π * radius^2 * height_total
  volume_cylinder / 2 = 8100 * π :=
by
  sorry

end volume_of_pool_l773_77386


namespace cost_of_pastrami_l773_77336

-- Definitions based on the problem conditions
def cost_of_reuben (R : ℝ) : Prop :=
  ∃ P : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55

-- Theorem stating the solution to the problem
theorem cost_of_pastrami : ∃ P : ℝ, ∃ R : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55 ∧ P = 5 :=
by 
  sorry

end cost_of_pastrami_l773_77336


namespace greatest_possible_value_l773_77337

theorem greatest_possible_value (x : ℝ) : 
  (∃ (k : ℝ), k = (5 * x - 25) / (4 * x - 5) ∧ k^2 + k = 20) → x ≤ 2 := 
sorry

end greatest_possible_value_l773_77337


namespace total_feathers_needed_l773_77383

theorem total_feathers_needed
  (animals_first_group : ℕ := 934)
  (feathers_first_group : ℕ := 7)
  (animals_second_group : ℕ := 425)
  (colored_feathers_second_group : ℕ := 7)
  (golden_feathers_second_group : ℕ := 5)
  (animals_third_group : ℕ := 289)
  (colored_feathers_third_group : ℕ := 4)
  (golden_feathers_third_group : ℕ := 10) :
  (animals_first_group * feathers_first_group) +
  (animals_second_group * (colored_feathers_second_group + golden_feathers_second_group)) +
  (animals_third_group * (colored_feathers_third_group + golden_feathers_third_group)) = 15684 := by
  sorry

end total_feathers_needed_l773_77383


namespace compute_54_mul_46_l773_77333

theorem compute_54_mul_46 : (54 * 46 = 2484) :=
by sorry

end compute_54_mul_46_l773_77333


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l773_77353

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l773_77353


namespace factorize_expression_l773_77360

variable {a b : ℝ} -- define a and b as real numbers

theorem factorize_expression : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l773_77360


namespace donation_to_first_home_l773_77311

theorem donation_to_first_home :
  let total_donation := 700
  let donation_to_second := 225
  let donation_to_third := 230
  total_donation - donation_to_second - donation_to_third = 245 :=
by
  sorry

end donation_to_first_home_l773_77311


namespace pages_written_in_a_year_l773_77398

-- Definitions based on conditions
def pages_per_letter : ℕ := 3
def letters_per_week : ℕ := 2
def friends : ℕ := 2
def weeks_per_year : ℕ := 52

-- Definition to calculate total pages written in a week
def weekly_pages (pages_per_letter : ℕ) (letters_per_week : ℕ) (friends : ℕ) : ℕ :=
  pages_per_letter * letters_per_week * friends

-- Definition to calculate total pages written in a year
def yearly_pages (weekly_pages : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weekly_pages * weeks_per_year

-- Theorem to prove the total pages written in a year
theorem pages_written_in_a_year : yearly_pages (weekly_pages pages_per_letter letters_per_week friends) weeks_per_year = 624 :=
by 
  sorry

end pages_written_in_a_year_l773_77398


namespace train_speed_in_km_per_hr_l773_77339

-- Conditions
def time_in_seconds : ℕ := 9
def length_in_meters : ℕ := 175

-- Conversion factor from m/s to km/hr
def meters_per_sec_to_km_per_hr (speed_m_per_s : ℚ) : ℚ :=
  speed_m_per_s * 3.6

-- Question as statement
theorem train_speed_in_km_per_hr :
  meters_per_sec_to_km_per_hr ((length_in_meters : ℚ) / (time_in_seconds : ℚ)) = 70 := by
  sorry

end train_speed_in_km_per_hr_l773_77339


namespace roots_quadratic_l773_77351

theorem roots_quadratic (a b c d : ℝ) :
  (a + b = 3 * c / 2 ∧ a * b = 4 * d ∧ c + d = 3 * a / 2 ∧ c * d = 4 * b)
  ↔ ( (a = 4 ∧ b = 8 ∧ c = 4 ∧ d = 8) ∨
      (a = -2 ∧ b = -22 ∧ c = -8 ∧ d = 11) ∨
      (a = -8 ∧ b = 2 ∧ c = -2 ∧ d = -4) ) :=
by
  sorry

end roots_quadratic_l773_77351


namespace max_number_of_cubes_l773_77309

theorem max_number_of_cubes (l w h v_cube : ℕ) (h_l : l = 8) (h_w : w = 9) (h_h : h = 12) (h_v_cube : v_cube = 27) :
  (l * w * h) / v_cube = 32 :=
by
  sorry

end max_number_of_cubes_l773_77309


namespace find_value_l773_77358

variable {x y : ℝ}

theorem find_value (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y + x * y = 0) : y / x + x / y = -2 := 
sorry

end find_value_l773_77358


namespace tenth_term_arithmetic_sequence_l773_77320

theorem tenth_term_arithmetic_sequence :
  let a₁ := 3 / 4
  let d := 1 / 4
  let aₙ (n : ℕ) := a₁ + (n - 1) * d
  aₙ 10 = 3 :=
by
  let a₁ := 3 / 4
  let d := 1 / 4
  let aₙ (n : ℕ) := a₁ + (n - 1) * d
  show aₙ 10 = 3
  sorry

end tenth_term_arithmetic_sequence_l773_77320


namespace find_d_l773_77377

theorem find_d :
  ∃ d : ℝ, (∀ x y : ℝ, x^2 + 3 * y^2 + 6 * x - 18 * y + d = 0 → x = -3 ∧ y = 3) ↔ d = -27 :=
by {
  sorry
}

end find_d_l773_77377


namespace units_digit_of_5_to_4_l773_77371

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_5_to_4 : units_digit (5^4) = 5 := by
  -- The definition ensures that 5^4 = 625 and the units digit is 5
  sorry

end units_digit_of_5_to_4_l773_77371


namespace common_ratio_of_sequence_l773_77324

theorem common_ratio_of_sequence 
  (a1 a2 a3 a4 : ℤ)
  (h1 : a1 = 25)
  (h2 : a2 = -50)
  (h3 : a3 = 100)
  (h4 : a4 = -200)
  (is_geometric : ∀ (i : ℕ), a1 * (-2) ^ i = if i = 0 then a1 else if i = 1 then a2 else if i = 2 then a3 else a4) : 
  (-50 / 25 = -2) ∧ (100 / -50 = -2) ∧ (-200 / 100 = -2) :=
by 
  sorry

end common_ratio_of_sequence_l773_77324


namespace value_of_n_l773_77368

def is_3_digit_integer (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

def not_divisible_by (n k : ℕ) : Prop := ¬ (k ∣ n)

def least_common_multiple (a b c : ℕ) : Prop := Nat.lcm a b = c

theorem value_of_n (d n : ℕ) (h1 : least_common_multiple d n 690) 
  (h2 : not_divisible_by n 3) (h3 : not_divisible_by d 2) (h4 : is_3_digit_integer n) : n = 230 :=
by
  sorry

end value_of_n_l773_77368


namespace total_stickers_l773_77355

def stickers_in_first_box : ℕ := 23
def stickers_in_second_box : ℕ := stickers_in_first_box + 12

theorem total_stickers :
  stickers_in_first_box + stickers_in_second_box = 58 := 
by
  sorry

end total_stickers_l773_77355


namespace number_of_arrangements_l773_77314

-- Definitions of the problem's conditions
def student_set : Finset ℕ := {1, 2, 3, 4, 5}

def specific_students : Finset ℕ := {1, 2}

def remaining_students : Finset ℕ := student_set \ specific_students

-- Formalize the problem statement
theorem number_of_arrangements : 
  ∀ (students : Finset ℕ) 
    (specific : Finset ℕ) 
    (remaining : Finset ℕ),
    students = student_set →
    specific = specific_students →
    remaining = remaining_students →
    (specific.card = 2 ∧ students.card = 5 ∧ remaining.card = 3) →
    (∃ (n : ℕ), n = 12) :=
by
  intros
  sorry

end number_of_arrangements_l773_77314


namespace find_b_l773_77385

theorem find_b
  (a b c d : ℝ)
  (h₁ : -a + b - c + d = 0)
  (h₂ : a + b + c + d = 0)
  (h₃ : d = 2) :
  b = -2 := 
by 
  sorry

end find_b_l773_77385


namespace distance_city_A_B_l773_77362

theorem distance_city_A_B (D : ℝ) : 
  (3 : ℝ) + (2.5 : ℝ) = 5.5 → 
  ∃ T_saved, T_saved = 1 →
  80 = (2 * D) / (5.5 - T_saved) →
  D = 180 :=
by
  intros
  sorry

end distance_city_A_B_l773_77362


namespace remainder_when_divided_by_x_minus_2_l773_77332

-- We define the polynomial f(x)
def f (x : ℝ) := x^4 - 6 * x^3 + 11 * x^2 + 20 * x - 8

-- We need to show that the remainder when f(x) is divided by (x - 2) is 44
theorem remainder_when_divided_by_x_minus_2 : f 2 = 44 :=
by {
  -- this is where the proof would go
  sorry
}

end remainder_when_divided_by_x_minus_2_l773_77332


namespace bisection_method_root_interval_l773_77373

noncomputable def f (x : ℝ) : ℝ := 2^x + 3 * x - 7

theorem bisection_method_root_interval :
  f 1 < 0 → f 2 > 0 → f 3 > 0 → ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 :=
by
  sorry

end bisection_method_root_interval_l773_77373


namespace find_c_share_l773_77316

noncomputable def shares (a b c d : ℝ) : Prop :=
  (5 * a = 4 * c) ∧ (7 * b = 4 * c) ∧ (2 * d = 4 * c) ∧ (a + b + c + d = 1200)

theorem find_c_share (A B C D : ℝ) (h : shares A B C D) : C = 275 :=
  by
  sorry

end find_c_share_l773_77316


namespace sum_of_dice_not_in_set_l773_77396

theorem sum_of_dice_not_in_set (a b c : ℕ) (h₁ : 1 ≤ a ∧ a ≤ 6) (h₂ : 1 ≤ b ∧ b ≤ 6) (h₃ : 1 ≤ c ∧ c ≤ 6) 
  (h₄ : a * b * c = 72) (h₅ : a = 4 ∨ b = 4 ∨ c = 4) :
  a + b + c ≠ 12 ∧ a + b + c ≠ 14 ∧ a + b + c ≠ 15 ∧ a + b + c ≠ 16 :=
by
  sorry

end sum_of_dice_not_in_set_l773_77396


namespace remainder_when_divided_by_x_minus_2_l773_77322

noncomputable def f (x : ℝ) : ℝ := x^4 - 3 * x^3 + 2 * x^2 + 11 * x - 6

theorem remainder_when_divided_by_x_minus_2 :
  (f 2) = 16 := by
  sorry

end remainder_when_divided_by_x_minus_2_l773_77322


namespace distance_between_parallel_lines_l773_77399

-- Definitions of lines l_1 and l_2
def line_l1 (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def line_l2 (x y : ℝ) : Prop := 6*x + 8*y - 5 = 0

-- Proof statement that the distance between the two lines is 1/10
theorem distance_between_parallel_lines (x y : ℝ) :
  ∃ d : ℝ, d = 1/10 ∧ ∀ p : ℝ × ℝ,
  (line_l1 p.1 p.2 ∧ line_l2 p.1 p.2 → p = (x, y)) :=
sorry

end distance_between_parallel_lines_l773_77399


namespace parallel_edges_octahedron_l773_77388

-- Definition of a regular octahedron's properties
structure regular_octahedron : Type :=
  (edges : ℕ) -- Number of edges in the octahedron

-- Constant to represent the regular octahedron with 12 edges.
def octahedron : regular_octahedron := { edges := 12 }

-- Definition to count unique pairs of parallel edges
def count_parallel_edge_pairs (o : regular_octahedron) : ℕ :=
  if o.edges = 12 then 12 else 0

-- Theorem to assert the number of pairs of parallel edges in a regular octahedron is 12
theorem parallel_edges_octahedron : count_parallel_edge_pairs octahedron = 12 :=
by
  -- Proof will be inserted here
  sorry

end parallel_edges_octahedron_l773_77388


namespace num_lines_in_grid_l773_77361

theorem num_lines_in_grid (columns rows : ℕ) (H1 : columns = 4) (H2 : rows = 3) 
    (total_points : ℕ) (H3 : total_points = columns * rows) :
    ∃ lines, lines = 40 :=
by
  sorry

end num_lines_in_grid_l773_77361


namespace total_votes_cast_l773_77327

/-- Define the conditions for Elvis's votes and percentage representation -/
def elvis_votes : ℕ := 45
def percentage_representation : ℚ := 1 / 4

/-- The main theorem that proves the total number of votes cast -/
theorem total_votes_cast : (elvis_votes: ℚ) / percentage_representation = 180 := by
  sorry

end total_votes_cast_l773_77327


namespace classroom_gpa_l773_77378

theorem classroom_gpa (n : ℕ) (h1 : 1 ≤ n) : 
  (1/3 : ℝ) * 30 + (2/3 : ℝ) * 33 = 32 :=
by sorry

end classroom_gpa_l773_77378


namespace RandomEvent_Proof_l773_77395

-- Define the events and conditions
def EventA : Prop := ∀ (θ₁ θ₂ θ₃ : ℝ), θ₁ + θ₂ + θ₃ = 360 → False
def EventB : Prop := ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 6) → n < 7
def EventC : Prop := ∃ (factors : ℕ → ℝ), (∃ (uncertainty : ℕ → ℝ), True)
def EventD : Prop := ∀ (balls : ℕ), (balls = 0 ∨ balls ≠ 0) → False

-- The theorem represents the proof problem
theorem RandomEvent_Proof : EventC :=
by
  sorry

end RandomEvent_Proof_l773_77395


namespace evaluate_expression_l773_77344

theorem evaluate_expression (x : ℝ) (h : x = Real.sqrt 3) : 
  ( (x^2 - 2*x + 1) / (x^2 - x) / (x - 1) ) = Real.sqrt 3 / 3 :=
by
  sorry

end evaluate_expression_l773_77344


namespace slopes_angle_l773_77357

theorem slopes_angle (k_1 k_2 : ℝ) (θ : ℝ) 
  (h1 : 6 * k_1^2 + k_1 - 1 = 0)
  (h2 : 6 * k_2^2 + k_2 - 1 = 0) :
  θ = π / 4 ∨ θ = 3 * π / 4 := 
by sorry

end slopes_angle_l773_77357


namespace constant_term_exists_l773_77300

theorem constant_term_exists:
  ∃ (n : ℕ), 2 ≤ n ∧ n ≤ 10 ∧ 
  (∃ r : ℕ, n = 3 * r) ∧ (∃ k : ℕ, n = 2 * k) ∧ 
  n = 6 :=
sorry

end constant_term_exists_l773_77300


namespace garage_has_18_wheels_l773_77384

namespace Garage

def bike_wheels_per_bike : ℕ := 2
def bikes_assembled : ℕ := 9

theorem garage_has_18_wheels
  (b : ℕ := bikes_assembled) 
  (w : ℕ := bike_wheels_per_bike) :
  b * w = 18 :=
by
  sorry

end Garage

end garage_has_18_wheels_l773_77384


namespace combination_sum_l773_77380

-- Definition of combination, also known as binomial coefficient
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem statement
theorem combination_sum :
  (combination 8 2) + (combination 8 3) = 84 :=
by
  sorry

end combination_sum_l773_77380


namespace intersection_sets_l773_77379

noncomputable def set1 (x : ℝ) : Prop := (x - 2) / (x + 1) ≤ 0
noncomputable def set2 (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

theorem intersection_sets :
  { x : ℝ | set1 x } ∩ { x : ℝ | set2 x } = { x | (-1 : ℝ) < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_sets_l773_77379


namespace sam_age_l773_77315

variable (Sam Drew : ℕ)

theorem sam_age :
  (Sam + Drew = 54) →
  (Sam = Drew / 2) →
  Sam = 18 :=
by intros h1 h2; sorry

end sam_age_l773_77315


namespace rita_remaining_money_l773_77347

theorem rita_remaining_money :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let jackets_cost := 4 * 30
  let transport_cost := 5
  let total_expenses := dresses_cost + pants_cost + jackets_cost + transport_cost
  let initial_money := 400
  let remaining_money := initial_money - total_expenses
  remaining_money = 139 := 
by
  sorry

end rita_remaining_money_l773_77347


namespace area_triangle_parabola_l773_77391

noncomputable def area_of_triangle_ABC (d : ℝ) (x : ℝ) : ℝ :=
  let A := (x, x^2)
  let B := (x + d, (x + d)^2)
  let C := (x + 2 * d, (x + 2 * d)^2)
  1 / 2 * abs (x * ((x + 2 * d)^2 - (x + d)^2) + (x + d) * ((x + 2 * d)^2 - x^2) + (x + 2 * d) * (x^2 - (x + d)^2))

theorem area_triangle_parabola (d : ℝ) (h_d : 0 < d) (x : ℝ) : 
  area_of_triangle_ABC d x = d^2 := sorry

end area_triangle_parabola_l773_77391


namespace quadratic_coeff_b_is_4_sqrt_15_l773_77389

theorem quadratic_coeff_b_is_4_sqrt_15 :
  ∃ m b : ℝ, (x^2 + bx + 72 = (x + m)^2 + 12) → (m = 2 * Real.sqrt 15) → (b = 4 * Real.sqrt 15) ∧ b > 0 :=
by
  -- Note: Proof not included as per the instruction.
  sorry

end quadratic_coeff_b_is_4_sqrt_15_l773_77389


namespace treaty_signed_on_tuesday_l773_77356

-- Define a constant for the start date and the number of days
def start_day_of_week : ℕ := 1 -- Monday is represented by 1
def days_until_treaty : ℕ := 1301

-- Function to calculate the resulting day of the week
def day_of_week_after_days (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

-- Theorem statement: Prove that 1301 days after Monday is Tuesday
theorem treaty_signed_on_tuesday :
  day_of_week_after_days start_day_of_week days_until_treaty = 2 :=
by
  -- placeholder for the proof
  sorry

end treaty_signed_on_tuesday_l773_77356


namespace sum_of_roots_l773_77370

theorem sum_of_roots (a b c : ℝ) (h : 6 * a^3 + 7 * a^2 - 12 * a = 0) : 
  - (7 / 6 : ℝ) = -1.17 := 
sorry

end sum_of_roots_l773_77370


namespace bus_driver_hours_worked_last_week_l773_77317

-- Definitions for given conditions
def regular_rate : ℝ := 12
def passenger_rate : ℝ := 0.50
def overtime_rate_1 : ℝ := 1.5 * regular_rate
def overtime_rate_2 : ℝ := 2 * regular_rate
def total_compensation : ℝ := 1280
def total_passengers : ℝ := 350
def earnings_from_passengers : ℝ := total_passengers * passenger_rate
def earnings_from_hourly_rate : ℝ := total_compensation - earnings_from_passengers
def regular_hours : ℝ := 40
def first_tier_overtime_hours : ℝ := 5

-- Theorem to prove the number of hours worked is 67
theorem bus_driver_hours_worked_last_week :
  ∃ (total_hours : ℝ),
    total_hours = 67 ∧
    earnings_from_passengers = total_passengers * passenger_rate ∧
    earnings_from_hourly_rate = total_compensation - earnings_from_passengers ∧
    (∃ (overtime_hours : ℝ),
      (overtime_hours = regular_hours + first_tier_overtime_hours + (earnings_from_hourly_rate - (regular_hours * regular_rate) - (first_tier_overtime_hours * overtime_rate_1)) / overtime_rate_2) ∧
      total_hours = regular_hours + first_tier_overtime_hours + (earnings_from_hourly_rate - (regular_hours * regular_rate) - (first_tier_overtime_hours * overtime_rate_1)) / overtime_rate_2 )
  :=
sorry

end bus_driver_hours_worked_last_week_l773_77317


namespace twelve_months_game_probability_l773_77394

/-- The card game "Twelve Months" involves turning over cards according to a set of rules.
Given the rules, we are asked to find the probability that all 12 columns of cards can be fully turned over. -/
def twelve_months_probability : ℚ :=
  1 / 12

theorem twelve_months_game_probability :
  twelve_months_probability = 1 / 12 :=
by
  -- The conditions and their representations are predefined.
  sorry

end twelve_months_game_probability_l773_77394


namespace circle_equation_l773_77310

theorem circle_equation (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 4) :
    x^2 + y^2 - 2 * x - 3 = 0 :=
sorry

end circle_equation_l773_77310


namespace maximum_value_a_plus_b_cubed_plus_c_fourth_l773_77305

theorem maximum_value_a_plus_b_cubed_plus_c_fourth (a b c : ℝ)
    (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
    (h_sum : a + b + c = 2) : a + b^3 + c^4 ≤ 2 :=
sorry

end maximum_value_a_plus_b_cubed_plus_c_fourth_l773_77305
