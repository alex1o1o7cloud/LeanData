import Mathlib

namespace find_A_l1626_162680

theorem find_A (A B : ℕ) (A_digit : A < 10) (B_digit : B < 10) :
  let fourteenA := 100 * 1 + 10 * 4 + A
  let Bseventy3 := 100 * B + 70 + 3
  fourteenA + Bseventy3 = 418 → A = 5 :=
by
  sorry

end find_A_l1626_162680


namespace measure_of_angle_D_l1626_162606

theorem measure_of_angle_D 
  (A B C D E F : ℝ)
  (h1 : A = B) (h2 : B = C) (h3 : C = F)
  (h4 : D = E) (h5 : A = D - 30) 
  (sum_angles : A + B + C + D + E + F = 720) : 
  D = 140 :=
by
  sorry

end measure_of_angle_D_l1626_162606


namespace average_minutes_run_per_day_l1626_162645

theorem average_minutes_run_per_day (f : ℕ) (h_nonzero : f ≠ 0)
  (third_avg fourth_avg fifth_avg : ℕ)
  (third_avg_eq : third_avg = 14)
  (fourth_avg_eq : fourth_avg = 18)
  (fifth_avg_eq : fifth_avg = 8)
  (third_count fourth_count fifth_count : ℕ)
  (third_count_eq : third_count = 3 * fourth_count)
  (fourth_count_eq : fourth_count = f / 2)
  (fifth_count_eq : fifth_count = f) :
  (third_avg * third_count + fourth_avg * fourth_count + fifth_avg * fifth_count) / (third_count + fourth_count + fifth_count) = 38 / 3 :=
by
  sorry

end average_minutes_run_per_day_l1626_162645


namespace find_m_l1626_162625

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem find_m (m : ℝ) :
  is_parallel (vector_a.1 + 2 * m, vector_a.2 + 2 * 1) (2 * vector_a.1 - m, 2 * vector_a.2 - 1) ↔ m = -1 / 2 := 
by {
  sorry
}

end find_m_l1626_162625


namespace sin_double_alpha_trig_expression_l1626_162629

theorem sin_double_alpha (α : ℝ) (h1 : Real.sin α = -1 / 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin (2 * α) = 4 * Real.sqrt 2 / 9 :=
sorry

theorem trig_expression (α : ℝ) (h1 : Real.sin α = -1 / 3) (h2 : π < α ∧ α < 3 * π / 2) :
  (Real.sin (α - 2 * π) * Real.cos (2 * π - α)) / (Real.sin (α + π / 2) ^ 2) = Real.sqrt 2 / 4 :=
sorry

end sin_double_alpha_trig_expression_l1626_162629


namespace Black_Queen_thought_Black_King_asleep_l1626_162602

theorem Black_Queen_thought_Black_King_asleep (BK_awake : Prop) (BQ_awake : Prop) :
  (∃ t : ℕ, t = 10 * 60 + 55 → 
  ∀ (BK : Prop) (BQ : Prop),
    ((BK_awake ↔ ¬BK) ∧ (BQ_awake ↔ ¬BQ)) ∧
    (BK → BQ → BQ_awake) ∧
    (¬BK → ¬BQ → BK_awake)) →
  ((BQ ↔ BK) ∧ (BQ_awake ↔ ¬BQ)) →
  (∃ (BQ_thought : Prop), BQ_thought ↔ BK) := 
sorry

end Black_Queen_thought_Black_King_asleep_l1626_162602


namespace compare_sqrt_sums_l1626_162654

   noncomputable def a : ℝ := Real.sqrt 8 + Real.sqrt 5
   noncomputable def b : ℝ := Real.sqrt 7 + Real.sqrt 6

   theorem compare_sqrt_sums : a < b :=
   by
     sorry
   
end compare_sqrt_sums_l1626_162654


namespace nesbitts_inequality_l1626_162627

variable (a b c : ℝ)

theorem nesbitts_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + c / (a + b)) >= 3 / 2 := 
sorry

end nesbitts_inequality_l1626_162627


namespace arithmetic_sequence_common_difference_l1626_162677

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
    (h1 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2)
    (h2 : (S 2017) / 2017 - (S 17) / 17 = 100) :
    d = 1/10 := 
by sorry

end arithmetic_sequence_common_difference_l1626_162677


namespace inv_g_of_43_div_16_l1626_162613

noncomputable def g (x : ℚ) : ℚ := (x^3 - 5) / 4

theorem inv_g_of_43_div_16 : g (3 * (↑7)^(1/3) / 2) = 43 / 16 :=
by 
  sorry

end inv_g_of_43_div_16_l1626_162613


namespace range_of_a_l1626_162620

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = abs (x - 2) + abs (x + a) ∧ f x ≥ 3) : a ≤ -5 ∨ a ≥ 1 :=
sorry

end range_of_a_l1626_162620


namespace find_max_m_l1626_162669

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * Real.exp (2 * x) - a * x

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (x - m) * f x 1 - (1/4) * Real.exp (2 * x) + x^2 + x

theorem find_max_m (h_inc : ∀ x > 0, g x m ≥ g x m) : m ≤ 1 :=
by
  sorry

end find_max_m_l1626_162669


namespace part_one_solution_part_two_solution_l1626_162615

-- Definitions and conditions
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part (1): When a = 1, solution set of the inequality f(x) > 1 is (1/2, +∞)
theorem part_one_solution (x : ℝ) :
  f x 1 > 1 ↔ x > 1 / 2 := sorry

-- Part (2): If the inequality f(x) > x holds for x ∈ (0,1), range of values for a is (0, 2]
theorem part_two_solution (a : ℝ) :
  (∀ x, 0 < x ∧ x < 1 → f x a > x) ↔ 0 < a ∧ a ≤ 2 := sorry

end part_one_solution_part_two_solution_l1626_162615


namespace train_length_l1626_162671

theorem train_length (t : ℝ) (v : ℝ) (h1 : t = 13) (h2 : v = 58.15384615384615) : abs (v * t - 756) < 1 :=
by
  sorry

end train_length_l1626_162671


namespace correct_calculation_l1626_162641

/-- Conditions for the given calculations -/
def cond_a : Prop := (-2) ^ 3 = 8
def cond_b : Prop := (-3) ^ 2 = -9
def cond_c : Prop := -(3 ^ 2) = -9
def cond_d : Prop := (-2) ^ 2 = 4

/-- Prove that the correct calculation among the given is -3^2 = -9 -/
theorem correct_calculation : cond_c :=
by sorry

end correct_calculation_l1626_162641


namespace algebraic_identity_l1626_162684

theorem algebraic_identity (a b : ℝ) : a^2 - 2 * a * b + b^2 = (a - b)^2 :=
by
  sorry

end algebraic_identity_l1626_162684


namespace square_side_length_l1626_162612

theorem square_side_length (A : ℝ) (s : ℝ) (hA : A = 64) (h_s : A = s * s) : s = 8 := by
  sorry

end square_side_length_l1626_162612


namespace total_people_veg_l1626_162600

def people_only_veg : ℕ := 13
def people_both_veg_nonveg : ℕ := 8

theorem total_people_veg : people_only_veg + people_both_veg_nonveg = 21 := by
  sorry

end total_people_veg_l1626_162600


namespace boris_can_achieve_7_60_cents_l1626_162693

/-- Define the conditions as constants -/
def penny_value : ℕ := 1
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def quarter_value : ℕ := 25

def penny_to_dimes : ℕ := 69
def dime_to_pennies : ℕ := 5
def nickel_to_quarters : ℕ := 120

/-- Function to determine if a value can be produced by a sequence of machine operations -/
def achievable_value (start: ℕ) (target: ℕ) : Prop :=
  ∃ k : ℕ, target = start + k * penny_to_dimes

theorem boris_can_achieve_7_60_cents : achievable_value penny_value 760 :=
  sorry

end boris_can_achieve_7_60_cents_l1626_162693


namespace interval_of_monotonic_increase_l1626_162692

noncomputable def powerFunction (k n x : ℝ) : ℝ := k * x ^ n

variable {k n : ℝ}

theorem interval_of_monotonic_increase
    (h : ∃ k n : ℝ, powerFunction k n 4 = 2) :
    (∀ x y : ℝ, 0 < x ∧ x < y → powerFunction k n x < powerFunction k n y) ∨
    (∀ x y : ℝ, 0 ≤ x ∧ x < y → powerFunction k n x ≤ powerFunction k n y) := sorry

end interval_of_monotonic_increase_l1626_162692


namespace prob_rain_all_days_l1626_162676

/--
The probability of rain on Friday, Saturday, and Sunday is given by 
0.40, 0.60, and 0.35 respectively.
We want to prove that the combined probability of rain on all three days,
assuming independence, is 8.4%.
-/
theorem prob_rain_all_days :
  let p_friday := 0.40
  let p_saturday := 0.60
  let p_sunday := 0.35
  p_friday * p_saturday * p_sunday = 0.084 :=
by
  sorry

end prob_rain_all_days_l1626_162676


namespace line_equations_l1626_162686

theorem line_equations : 
  ∀ (x y : ℝ), (∃ a b c : ℝ, 2 * x + y - 12 = 0 ∨ 2 * x - 5 * y = 0 ∧ (x, y) = (5, 2) ∧ b = 2 * a) :=
by
  sorry

end line_equations_l1626_162686


namespace min_rooms_needed_l1626_162639

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l1626_162639


namespace stuffed_animals_total_l1626_162630

variable (x y z : ℕ)

theorem stuffed_animals_total :
  let initial := x
  let after_mom := initial + y
  let after_dad := z * after_mom
  let total := after_mom + after_dad
  total = (x + y) * (1 + z) := 
  by 
    let initial := x
    let after_mom := initial + y
    let after_dad := z * after_mom
    let total := after_mom + after_dad
    sorry

end stuffed_animals_total_l1626_162630


namespace min_N_such_that_next_person_sits_next_to_someone_l1626_162649

def circular_table_has_80_chairs : Prop := ∃ chairs : ℕ, chairs = 80
def N_people_seated (N : ℕ) : Prop := N > 0
def next_person_sits_next_to_someone (N : ℕ) : Prop :=
  ∀ additional_person_seated : ℕ, additional_person_seated ≤ N → additional_person_seated > 0 
  → ∃ adjacent_person : ℕ, adjacent_person ≤ N ∧ adjacent_person > 0
def smallest_value_for_N (N : ℕ) : Prop :=
  (∀ k : ℕ, k < N → ¬next_person_sits_next_to_someone k)

theorem min_N_such_that_next_person_sits_next_to_someone :
  circular_table_has_80_chairs →
  smallest_value_for_N 20 :=
by
  intro h
  sorry

end min_N_such_that_next_person_sits_next_to_someone_l1626_162649


namespace right_triangle_area_valid_right_triangle_perimeter_valid_l1626_162624

-- Define the basic setup for the right triangle problem
def hypotenuse : ℕ := 13
def leg1 : ℕ := 5
def leg2 : ℕ := 12  -- Calculated from Pythagorean theorem, but assumed here as condition

-- Define the calculated area and perimeter based on the above definitions
def area (a b : ℕ) : ℕ := (1 / 2) * a * b
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- State the proof goals
theorem right_triangle_area_valid : area leg1 leg2 = 30 :=
  by sorry

theorem right_triangle_perimeter_valid : perimeter leg1 leg2 hypotenuse = 30 :=
  by sorry

end right_triangle_area_valid_right_triangle_perimeter_valid_l1626_162624


namespace problem_statement_l1626_162653

noncomputable def p := Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def q := -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def r := Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7
noncomputable def s := -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7

theorem problem_statement :
  (1 / p + 1 / q + 1 / r + 1 / s)^2 = 112 / 3481 :=
sorry

end problem_statement_l1626_162653


namespace geometric_seq_arithmetic_triplet_l1626_162644

-- Definition of being in a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n * q

-- Condition that a_5, a_4, and a_6 form an arithmetic sequence
def is_arithmetic_triplet (a : ℕ → ℝ) (n : ℕ) : Prop :=
  2 * a n = a (n+1) + a (n+2)

-- Our specific problem translated into a Lean statement
theorem geometric_seq_arithmetic_triplet {a : ℕ → ℝ} (q : ℝ) :
  is_geometric_sequence a q →
  is_arithmetic_triplet a 4 →
  q = 1 ∨ q = -2 :=
by
  intros h_geo h_arith
  -- Proof here is omitted
  sorry

end geometric_seq_arithmetic_triplet_l1626_162644


namespace alice_min_speed_exceeds_45_l1626_162678

theorem alice_min_speed_exceeds_45 
  (distance : ℕ)
  (bob_speed : ℕ)
  (alice_delay : ℕ)
  (alice_speed : ℕ)
  (bob_time : ℕ)
  (expected_speed : ℕ) 
  (distance_eq : distance = 180)
  (bob_speed_eq : bob_speed = 40)
  (alice_delay_eq : alice_delay = 1/2)
  (bob_time_eq : bob_time = distance / bob_speed)
  (expected_speed_eq : expected_speed = distance / (bob_time - alice_delay)) :
  alice_speed > expected_speed := 
sorry

end alice_min_speed_exceeds_45_l1626_162678


namespace evaluate_f_5_minus_f_neg_5_l1626_162679

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by
  sorry

end evaluate_f_5_minus_f_neg_5_l1626_162679


namespace prime_p_satisfies_conditions_l1626_162688

theorem prime_p_satisfies_conditions (p : ℕ) (hp : Nat.Prime p) (h1 : Nat.Prime (4 * p^2 + 1)) (h2 : Nat.Prime (6 * p^2 + 1)) : p = 5 :=
sorry

end prime_p_satisfies_conditions_l1626_162688


namespace cube_inequality_l1626_162670

theorem cube_inequality (a b : ℝ) : a > b ↔ a^3 > b^3 :=
sorry

end cube_inequality_l1626_162670


namespace least_positive_integer_l1626_162607

theorem least_positive_integer (x : ℕ) (h : x + 5600 ≡ 325 [MOD 15]) : x = 5 :=
sorry

end least_positive_integer_l1626_162607


namespace overall_loss_percentage_l1626_162608

theorem overall_loss_percentage
  (cost_price : ℝ)
  (discount : ℝ)
  (sales_tax : ℝ)
  (depreciation : ℝ)
  (final_selling_price : ℝ) :
  cost_price = 1900 →
  discount = 0.15 →
  sales_tax = 0.12 →
  depreciation = 0.05 →
  final_selling_price = 1330 →
  ((cost_price - (discount * cost_price)) * (1 + sales_tax) * (1 - depreciation) - final_selling_price) / cost_price * 100 = 20.44 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end overall_loss_percentage_l1626_162608


namespace compute_value_l1626_162614

theorem compute_value : (142 + 29 + 26 + 14) * 2 = 422 := 
by 
  sorry

end compute_value_l1626_162614


namespace negation_proposition_l1626_162697

theorem negation_proposition:
  (¬ (∀ x : ℝ, (1 ≤ x) → (x^2 - 2*x + 1 ≥ 0))) ↔ (∃ x : ℝ, (1 ≤ x) ∧ (x^2 - 2*x + 1 < 0)) := 
sorry

end negation_proposition_l1626_162697


namespace max_y_value_l1626_162626

noncomputable def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_y_value : ∃ α, (∀ x, -1 ≤ x ∧ x ≤ 2 → y x ≤ α) ∧ α = 3 := by
  sorry

end max_y_value_l1626_162626


namespace dealer_decision_is_mode_l1626_162668

noncomputable def sales_A := 15
noncomputable def sales_B := 22
noncomputable def sales_C := 18
noncomputable def sales_D := 10

def is_mode (sales: List ℕ) (mode_value: ℕ) : Prop :=
  mode_value ∈ sales ∧ ∀ x ∈ sales, x ≤ mode_value

theorem dealer_decision_is_mode : 
  is_mode [sales_A, sales_B, sales_C, sales_D] sales_B :=
by
  sorry

end dealer_decision_is_mode_l1626_162668


namespace rectangle_longer_side_l1626_162650

theorem rectangle_longer_side
  (r : ℝ)
  (A_circle : ℝ)
  (A_rectangle : ℝ)
  (shorter_side : ℝ)
  (longer_side : ℝ) :
  r = 5 →
  A_circle = 25 * Real.pi →
  A_rectangle = 3 * A_circle →
  shorter_side = 2 * r →
  longer_side = A_rectangle / shorter_side →
  longer_side = 7.5 * Real.pi :=
by
  intros
  sorry

end rectangle_longer_side_l1626_162650


namespace ratio_problem_l1626_162681

open Classical 

variables {q r s t u : ℚ}

theorem ratio_problem (h1 : q / r = 8) (h2 : s / r = 5) (h3 : s / t = 1 / 4) (h4 : u / t = 3) :
  u / q = 15 / 2 :=
by
  sorry

end ratio_problem_l1626_162681


namespace one_div_add_one_div_interval_one_div_add_one_div_not_upper_bounded_one_div_add_one_div_in_interval_l1626_162695

theorem one_div_add_one_div_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  2 ≤ (1 / a + 1 / b) := 
sorry

theorem one_div_add_one_div_not_upper_bounded (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  ∀ M > 2, ∃ a' b', 0 < a' ∧ 0 < b' ∧ a' + b' = 2 ∧ (1 / a' + 1 / b') > M := 
sorry

theorem one_div_add_one_div_in_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (2 ≤ (1 / a + 1 / b) ∧ ∀ M > 2, ∃ a' b', 0 < a' ∧ 0 < b' ∧ a' + b' = 2 ∧ (1 / a' + 1 / b') > M) := 
sorry

end one_div_add_one_div_interval_one_div_add_one_div_not_upper_bounded_one_div_add_one_div_in_interval_l1626_162695


namespace num_ways_to_remove_blocks_l1626_162637

-- Definitions based on the problem conditions
def stack_blocks := 85
def block_layers := [1, 4, 16, 64]

-- Theorem statement
theorem num_ways_to_remove_blocks : 
  (∃ f : (ℕ → ℕ), 
    (∀ n, f n = if n = 0 then 1 else if n ≤ 4 then n * f (n - 1) + 3 * (f (n - 1) - 1) else 4^3 * 16) ∧ 
    f 5 = 3384) := sorry

end num_ways_to_remove_blocks_l1626_162637


namespace hydrogen_atoms_in_compound_l1626_162662

theorem hydrogen_atoms_in_compound :
  ∀ (n : ℕ), 98 = 14 + n + 80 → n = 4 :=
by intro n h_eq
   sorry

end hydrogen_atoms_in_compound_l1626_162662


namespace trig_identity_solution_l1626_162622

noncomputable def solve_trig_identity (x : ℝ) : Prop :=
  (∃ k : ℤ, x = (Real.pi / 8 * (4 * k + 1))) ∧
  (Real.sin (2 * x))^4 + (Real.cos (2 * x))^4 = Real.sin (2 * x) * Real.cos (2 * x)

theorem trig_identity_solution (x : ℝ) :
  solve_trig_identity x :=
sorry

end trig_identity_solution_l1626_162622


namespace total_games_l1626_162658

-- Definitions and conditions
noncomputable def num_teams : ℕ := 12

noncomputable def regular_season_games_each : ℕ := 4

noncomputable def knockout_games_each : ℕ := 2

-- Calculate total number of games
theorem total_games : (num_teams * (num_teams - 1) / 2) * regular_season_games_each + 
                      (num_teams * knockout_games_each / 2) = 276 :=
by
  -- This is the statement to be proven
  sorry

end total_games_l1626_162658


namespace pages_wednesday_l1626_162619

-- Given conditions as definitions
def borrow_books := 3
def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51

-- Prove that Nico read 19 pages on Wednesday
theorem pages_wednesday :
  let pages_wednesday := total_pages - (pages_monday + pages_tuesday)
  pages_wednesday = 19 :=
by
  sorry

end pages_wednesday_l1626_162619


namespace sum_of_midpoint_xcoords_l1626_162655

theorem sum_of_midpoint_xcoords (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
  by
    sorry

end sum_of_midpoint_xcoords_l1626_162655


namespace top_card_probability_spades_or_clubs_l1626_162696

-- Definitions
def total_cards : ℕ := 52
def suits : ℕ := 4
def ranks : ℕ := 13
def spades_cards : ℕ := ranks
def clubs_cards : ℕ := ranks
def favorable_outcomes : ℕ := spades_cards + clubs_cards

-- Probability calculation statement
theorem top_card_probability_spades_or_clubs :
  (favorable_outcomes : ℚ) / (total_cards : ℚ) = 1 / 2 :=
  sorry

end top_card_probability_spades_or_clubs_l1626_162696


namespace transistor_count_2010_l1626_162666

-- Define the known constants and conditions
def initial_transistors : ℕ := 2000000
def doubling_period : ℕ := 2
def years_elapsed : ℕ := 2010 - 1995
def number_of_doublings := years_elapsed / doubling_period -- we want floor division

-- The theorem statement we need to prove
theorem transistor_count_2010 : initial_transistors * 2^number_of_doublings = 256000000 := by
  sorry

end transistor_count_2010_l1626_162666


namespace Theorem3_l1626_162643

theorem Theorem3 {f g : ℝ → ℝ} (T1_eq_1 : ∀ x, f (x + 1) = f x)
  (m : ℕ) (h_g_periodic : ∀ x, g (x + 1 / m) = g x) (hm : m > 1) :
  ∃ k : ℕ, k > 0 ∧ (k = 1 ∨ (k ≠ m ∧ ¬(m % k = 0))) ∧ 
    (∀ x, (f x + g x) = (f (x + 1 / k) + g (x + 1 / k))) := 
sorry

end Theorem3_l1626_162643


namespace distance_from_ground_at_speed_25_is_137_5_l1626_162616
noncomputable section

-- Define the initial conditions and givens
def buildingHeight : ℝ := 200
def speedProportionalityConstant : ℝ := 10
def distanceProportionalityConstant : ℝ := 10

-- Define the speed function and distance function
def speed (t : ℝ) : ℝ := speedProportionalityConstant * t
def distance (t : ℝ) : ℝ := distanceProportionalityConstant * (t * t)

-- Define the specific time when speed is 25 m/sec
def timeWhenSpeedIs25 : ℝ := 25 / speedProportionalityConstant

-- Define the distance traveled at this specific time
def distanceTraveledAtTime : ℝ := distance timeWhenSpeedIs25

-- Calculate the distance from the ground
def distanceFromGroundAtSpeed25 : ℝ := buildingHeight - distanceTraveledAtTime

-- State the theorem
theorem distance_from_ground_at_speed_25_is_137_5 :
  distanceFromGroundAtSpeed25 = 137.5 :=
sorry

end distance_from_ground_at_speed_25_is_137_5_l1626_162616


namespace greatest_int_radius_of_circle_l1626_162694

theorem greatest_int_radius_of_circle (r : ℝ) (A : ℝ) :
  (A < 200 * Real.pi) ∧ (A = Real.pi * r^2) →
  ∃k : ℕ, (k : ℝ) = 14 ∧ ∀n : ℕ, (n : ℝ) = r → n ≤ k := by
  sorry

end greatest_int_radius_of_circle_l1626_162694


namespace min_xsq_ysq_zsq_l1626_162660

noncomputable def min_value_x_sq_y_sq_z_sq (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : ℝ :=
  (x^2 + y^2 + z^2)

theorem min_xsq_ysq_zsq (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  min_value_x_sq_y_sq_z_sq x y z h = 40 / 7 :=
  sorry

end min_xsq_ysq_zsq_l1626_162660


namespace train_time_first_platform_correct_l1626_162674

-- Definitions
variables (L_train L_first_plat L_second_plat : ℕ) (T_second : ℕ) (T_first : ℕ)

-- Given conditions
def length_train := 350
def length_first_platform := 100
def length_second_platform := 250
def time_second_platform := 20
def expected_time_first_platform := 15

-- Derived values
def total_distance_second_platform := length_train + length_second_platform
def speed := total_distance_second_platform / time_second_platform
def total_distance_first_platform := length_train + length_first_platform
def time_first_platform := total_distance_first_platform / speed

-- Proof Statement
theorem train_time_first_platform_correct : 
  time_first_platform = expected_time_first_platform :=
  by
  sorry

end train_time_first_platform_correct_l1626_162674


namespace largest_angle_heptagon_l1626_162661

theorem largest_angle_heptagon :
  ∃ (x : ℝ), 4 * x + 4 * x + 4 * x + 5 * x + 6 * x + 7 * x + 8 * x = 900 ∧ 8 * x = (7200 / 38) := 
by 
  sorry

end largest_angle_heptagon_l1626_162661


namespace equalize_expenses_l1626_162682

/-- Problem Statement:
Given the amount paid by LeRoy (A), Bernardo (B), and Carlos (C),
prove that the amount LeRoy must adjust to share the costs equally is (B + C - 2A) / 3.
-/
theorem equalize_expenses (A B C : ℝ) : 
  (B+C-2*A) / 3 = (A + B + C) / 3 - A :=
by
  sorry

end equalize_expenses_l1626_162682


namespace kostya_initially_planted_l1626_162618

def bulbs_after_planting (n : ℕ) (stages : ℕ) : ℕ :=
  match stages with
  | 0 => n
  | k + 1 => 2 * bulbs_after_planting n k - 1

theorem kostya_initially_planted (n : ℕ) (stages : ℕ) :
  bulbs_after_planting n stages = 113 → n = 15 := 
sorry

end kostya_initially_planted_l1626_162618


namespace floor_length_l1626_162685

theorem floor_length (tile_length tile_width : ℕ) (floor_width max_tiles : ℕ)
  (h_tile : tile_length = 25 ∧ tile_width = 16)
  (h_floor_width : floor_width = 120)
  (h_max_tiles : max_tiles = 54) :
  ∃ floor_length : ℕ, 
    (∃ num_cols num_rows : ℕ, 
      num_cols * tile_width = floor_width ∧ 
      num_cols * num_rows = max_tiles ∧ 
      num_rows * tile_length = floor_length) ∧
    floor_length = 175 := 
by
  sorry

end floor_length_l1626_162685


namespace cylinder_sphere_ratio_is_3_2_l1626_162657

noncomputable def cylinder_sphere_surface_ratio (r : ℝ) : ℝ :=
  let cylinder_surface_area := 2 * Real.pi * r^2 + 2 * r * Real.pi * (2 * r)
  let sphere_surface_area := 4 * Real.pi * r^2
  cylinder_surface_area / sphere_surface_area

theorem cylinder_sphere_ratio_is_3_2 (r : ℝ) (h : r > 0) :
  cylinder_sphere_surface_ratio r = 3 / 2 :=
by
  sorry

end cylinder_sphere_ratio_is_3_2_l1626_162657


namespace scale_length_l1626_162623

theorem scale_length (num_parts : ℕ) (part_length : ℕ) (total_length : ℕ) 
  (h1 : num_parts = 5) (h2 : part_length = 16) : total_length = 80 :=
by
  sorry

end scale_length_l1626_162623


namespace cube_painting_l1626_162690

theorem cube_painting (n : ℕ) (h : n > 2) :
  (6 * (n - 2)^2 = (n - 2)^3) ↔ (n = 8) :=
by
  sorry

end cube_painting_l1626_162690


namespace coordinates_of_point_P_l1626_162675

-- Define the function y = x^3
def cubic (x : ℝ) : ℝ := x^3

-- Define the derivative of the function
def derivative_cubic (x : ℝ) : ℝ := 3 * x^2

-- Define the condition for the slope of the tangent line to the function at point P
def slope_tangent_line := 3

-- Prove that the coordinates of point P are (1, 1) or (-1, -1) when the slope of the tangent line is 3
theorem coordinates_of_point_P (x : ℝ) (y : ℝ) 
    (h1 : y = cubic x) 
    (h2 : derivative_cubic x = slope_tangent_line) : 
    (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by
  sorry

end coordinates_of_point_P_l1626_162675


namespace joan_gave_away_kittens_l1626_162652

-- Definitions based on conditions in the problem
def original_kittens : ℕ := 8
def kittens_left : ℕ := 6

-- Mathematical statement to be proved
theorem joan_gave_away_kittens : original_kittens - kittens_left = 2 :=
by
  sorry

end joan_gave_away_kittens_l1626_162652


namespace quadratic_has_one_solution_l1626_162638

theorem quadratic_has_one_solution (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0) ∧ (∀ x₁ x₂ : ℝ, (3 * x₁^2 - 6 * x₁ + m = 0) → (3 * x₂^2 - 6 * x₂ + m = 0) → x₁ = x₂) → m = 3 :=
by
  -- intricate steps would go here
  sorry

end quadratic_has_one_solution_l1626_162638


namespace total_population_correct_l1626_162659

-- Given conditions
def number_of_cities : ℕ := 25
def average_population : ℕ := 3800

-- Statement to prove
theorem total_population_correct : number_of_cities * average_population = 95000 :=
by
  sorry

end total_population_correct_l1626_162659


namespace percent_of_area_triangle_in_pentagon_l1626_162672

-- Defining a structure for the problem statement
structure PentagonAndTriangle where
  s : ℝ -- side length of the equilateral triangle
  side_square : ℝ -- side of the square
  area_triangle : ℝ
  area_square : ℝ
  area_pentagon : ℝ

noncomputable def calculate_areas (s : ℝ) : PentagonAndTriangle :=
  let height_triangle := s * (Real.sqrt 3) / 2
  let area_triangle := Real.sqrt 3 / 4 * s^2
  let area_square := height_triangle^2
  let area_pentagon := area_square + area_triangle
  { s := s, side_square := height_triangle, area_triangle := area_triangle, area_square := area_square, area_pentagon := area_pentagon }

/--
Prove that the percentage of the pentagon's area that is the area of the equilateral triangle is (3 * (Real.sqrt 3 - 1)) / 6 * 100%.
-/
theorem percent_of_area_triangle_in_pentagon 
  (s : ℝ) 
  (pt : PentagonAndTriangle)
  (h₁ : pt = calculate_areas s)
  : pt.area_triangle / pt.area_pentagon = (3 * (Real.sqrt 3 - 1)) / 6 * 100 :=
by
  sorry

end percent_of_area_triangle_in_pentagon_l1626_162672


namespace volunteers_per_class_l1626_162634

theorem volunteers_per_class (total_needed volunteers teachers_needed : ℕ) (classes : ℕ)
    (h_total : total_needed = 50) (h_teachers : teachers_needed = 13) (h_more_needed : volunteers = 7) (h_classes : classes = 6) :
  (total_needed - teachers_needed - volunteers) / classes = 5 :=
by
  -- calculation and simplification
  sorry

end volunteers_per_class_l1626_162634


namespace jinho_total_distance_l1626_162691

theorem jinho_total_distance (bus_distance_km : ℝ) (bus_distance_m : ℝ) (walk_distance_m : ℝ) :
  bus_distance_km = 4 → bus_distance_m = 436 → walk_distance_m = 1999 → 
  (2 * (bus_distance_km + bus_distance_m / 1000 + walk_distance_m / 1000)) = 12.87 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jinho_total_distance_l1626_162691


namespace unique_positive_integer_solutions_l1626_162640

theorem unique_positive_integer_solutions : 
  ∀ (m n : ℕ), 0 < m ∧ 0 < n ∧ 7 ^ m - 3 * 2 ^ n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) :=
by sorry

end unique_positive_integer_solutions_l1626_162640


namespace second_caterer_cheaper_l1626_162611

theorem second_caterer_cheaper (x : ℕ) (h : x > 33) : 200 + 12 * x < 100 + 15 * x := 
by
  sorry

end second_caterer_cheaper_l1626_162611


namespace remainder_when_y_squared_divided_by_30_l1626_162699

theorem remainder_when_y_squared_divided_by_30 (y : ℤ) :
  6 * y ≡ 12 [ZMOD 30] → 5 * y ≡ 25 [ZMOD 30] → y ^ 2 ≡ 19 [ZMOD 30] :=
  by
  intro h1 h2
  sorry

end remainder_when_y_squared_divided_by_30_l1626_162699


namespace price_of_each_bracelet_l1626_162667

-- The conditions
def bike_cost : ℕ := 112
def days_in_two_weeks : ℕ := 14
def bracelets_per_day : ℕ := 8
def total_bracelets := days_in_two_weeks * bracelets_per_day

-- The question and the expected answer
def price_per_bracelet : ℕ := bike_cost / total_bracelets

theorem price_of_each_bracelet :
  price_per_bracelet = 1 := 
by
  sorry

end price_of_each_bracelet_l1626_162667


namespace fraction_division_l1626_162610

theorem fraction_division :
  (3 / 7) / (2 / 5) = (15 / 14) :=
by
  sorry

end fraction_division_l1626_162610


namespace sqrt_7_estimate_l1626_162632

theorem sqrt_7_estimate : (2 : Real) < Real.sqrt 7 ∧ Real.sqrt 7 < 3 → (Real.sqrt 7 - 1) / 2 < 1 := 
by
  intro h
  sorry

end sqrt_7_estimate_l1626_162632


namespace natasha_quarters_l1626_162636

theorem natasha_quarters :
  ∃ n : ℕ, (4 < n) ∧ (n < 40) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) ∧ (n = 2) := sorry

end natasha_quarters_l1626_162636


namespace analyze_a_b_m_n_l1626_162605

theorem analyze_a_b_m_n (a b m n : ℕ) (ha : 1 < a) (hb : 1 < b) (hm : 1 < m) (hn : 1 < n)
  (h1 : Prime (a^n - 1))
  (h2 : Prime (b^m + 1)) :
  n = 2 ∧ ∃ k : ℕ, m = 2^k :=
by
  sorry

end analyze_a_b_m_n_l1626_162605


namespace arithmetic_mean_reciprocals_first_four_primes_l1626_162635

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l1626_162635


namespace tim_books_l1626_162621

def has_some_books (Tim Sam : ℕ) : Prop :=
  Sam = 52 ∧ Tim + Sam = 96

theorem tim_books (Tim : ℕ) :
  has_some_books Tim 52 → Tim = 44 := 
by
  intro h
  obtain ⟨hSam, hTogether⟩ := h
  sorry

end tim_books_l1626_162621


namespace seq_max_min_terms_l1626_162617

noncomputable def a (n: ℕ) : ℝ := 1 / (2^n - 18)

theorem seq_max_min_terms : (∀ (n : ℕ), n > 5 → a 5 > a n) ∧ (∀ (n : ℕ), n ≠ 4 → a 4 < a n) :=
by 
  sorry

end seq_max_min_terms_l1626_162617


namespace student_tickets_sold_l1626_162656

theorem student_tickets_sold
  (A S : ℕ)
  (h1 : A + S = 846)
  (h2 : 6 * A + 3 * S = 3846) :
  S = 410 :=
sorry

end student_tickets_sold_l1626_162656


namespace necessary_but_not_sufficient_condition_l1626_162665

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x > 0) : 
  ((x > 2 ∧ x < 4) ↔ (2 < x ∧ x < 4)) :=
by {
    sorry
}

end necessary_but_not_sufficient_condition_l1626_162665


namespace percent_of_200_is_400_when_whole_is_50_l1626_162647

theorem percent_of_200_is_400_when_whole_is_50 (Part Whole : ℕ) (hPart : Part = 200) (hWhole : Whole = 50) :
  (Part / Whole) * 100 = 400 :=
by {
  -- Proof steps go here.
  sorry
}

end percent_of_200_is_400_when_whole_is_50_l1626_162647


namespace total_cupcakes_needed_l1626_162687

-- Definitions based on conditions
def cupcakes_per_event : ℝ := 96.0
def number_of_events : ℝ := 8.0

-- Theorem based on the question and the correct answer
theorem total_cupcakes_needed : (cupcakes_per_event * number_of_events) = 768.0 :=
by 
  sorry

end total_cupcakes_needed_l1626_162687


namespace smallest_possible_e_l1626_162651

-- Define the polynomial with its roots and integer coefficients
def polynomial (x : ℝ) : ℝ := (x + 4) * (x - 6) * (x - 10) * (2 * x + 1)

-- Define e as the constant term
def e : ℝ := 200 -- based on the final expanded polynomial result

-- The theorem stating the smallest possible value of e
theorem smallest_possible_e : 
  ∃ (e : ℕ), e > 0 ∧ polynomial e = 200 := 
sorry

end smallest_possible_e_l1626_162651


namespace find_original_workers_and_time_l1626_162642

-- Definitions based on the identified conditions
def original_workers (x : ℕ) (y : ℕ) : Prop :=
  (x - 2) * (y + 4) = x * y ∧
  (x + 3) * (y - 2) > x * y ∧
  (x + 4) * (y - 3) > x * y

-- Problem statement to prove
theorem find_original_workers_and_time (x y : ℕ) :
  original_workers x y → x = 6 ∧ y = 8 :=
by
  sorry

end find_original_workers_and_time_l1626_162642


namespace min_handshakes_35_people_l1626_162673

theorem min_handshakes_35_people (n : ℕ) (h1 : n = 35) (h2 : ∀ p : ℕ, p < n → p ≥ 3) : ∃ m : ℕ, m = 51 :=
by
  sorry

end min_handshakes_35_people_l1626_162673


namespace problem_statement_l1626_162633

theorem problem_statement : ∀ (x y : ℝ), |x - 2| + (y + 3)^2 = 0 → (x + y)^2023 = -1 :=
by
  intros x y h
  sorry

end problem_statement_l1626_162633


namespace gcd_282_470_l1626_162689

theorem gcd_282_470 : Nat.gcd 282 470 = 94 :=
by
  sorry

end gcd_282_470_l1626_162689


namespace star_shell_arrangements_l1626_162663

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Conditions
def outward_points : ℕ := 6
def inward_points : ℕ := 6
def total_points : ℕ := outward_points + inward_points
def unique_shells : ℕ := 12

-- The problem statement translated into Lean 4:
theorem star_shell_arrangements : (factorial unique_shells / 12 = 39916800) :=
by
  sorry

end star_shell_arrangements_l1626_162663


namespace max_similar_triangles_five_points_l1626_162664

-- Let P be a finite set of points on a plane with exactly 5 elements.
def max_similar_triangles(P : Finset (ℝ × ℝ)) : ℕ :=
  if h : P.card = 5 then
    8
  else
    0 -- This is irrelevant for the problem statement, but we need to define it.

-- The main theorem statement
theorem max_similar_triangles_five_points {P : Finset (ℝ × ℝ)} (h : P.card = 5) :
  max_similar_triangles P = 8 :=
sorry

end max_similar_triangles_five_points_l1626_162664


namespace find_original_price_l1626_162603

theorem find_original_price (sale_price : ℕ) (discount : ℕ) (original_price : ℕ) 
  (h1 : sale_price = 60) 
  (h2 : discount = 40) 
  (h3 : original_price = sale_price / ((100 - discount) / 100)) : original_price = 100 :=
by
  sorry

end find_original_price_l1626_162603


namespace geom_progression_sum_ratio_l1626_162628

theorem geom_progression_sum_ratio (a : ℝ) (r : ℝ) (m : ℕ) :
  r = 5 →
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^m) / (1 - r)) = 126 →
  m = 3 := by
  sorry

end geom_progression_sum_ratio_l1626_162628


namespace seulgi_stack_higher_l1626_162698

-- Define the conditions
def num_red_boxes : ℕ := 15
def num_yellow_boxes : ℕ := 20
def height_red_box : ℝ := 4.2
def height_yellow_box : ℝ := 3.3

-- Define the total height for each stack
def total_height_hyunjeong : ℝ := num_red_boxes * height_red_box
def total_height_seulgi : ℝ := num_yellow_boxes * height_yellow_box

-- Lean statement to prove the comparison of their heights
theorem seulgi_stack_higher : total_height_seulgi > total_height_hyunjeong :=
by
  -- Proof will be inserted here
  sorry

end seulgi_stack_higher_l1626_162698


namespace negative_values_of_x_l1626_162609

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l1626_162609


namespace sum_of_smallest_x_and_y_for_540_l1626_162648

theorem sum_of_smallest_x_and_y_for_540 (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : ∃ k₁, 540 * x = k₁ * k₁)
  (h2 : ∃ k₂, 540 * y = k₂ * k₂ * k₂) :
  x + y = 65 := 
sorry

end sum_of_smallest_x_and_y_for_540_l1626_162648


namespace find_second_sum_l1626_162646

theorem find_second_sum (x : ℝ) (h : 24 * x / 100 = (2730 - x) * 15 / 100) : 2730 - x = 1680 := by
  sorry

end find_second_sum_l1626_162646


namespace smallest_b_for_perfect_square_l1626_162631

theorem smallest_b_for_perfect_square : ∃ (b : ℕ), b > 4 ∧ (∃ k, (2 * b + 4) = k * k) ∧
                                             ∀ (b' : ℕ), b' > 4 ∧ (∃ k, (2 * b' + 4) = k * k) → b ≤ b' :=
by
  sorry

end smallest_b_for_perfect_square_l1626_162631


namespace daysRequired_l1626_162683

-- Defining the structure of the problem
structure WallConstruction where
  m1 : ℕ    -- Number of men in the first scenario
  d1 : ℕ    -- Number of days in the first scenario
  m2 : ℕ    -- Number of men in the second scenario

-- Given values
def wallConstructionProblem : WallConstruction :=
  WallConstruction.mk 20 5 30

-- The total work constant
def totalWork (wc : WallConstruction) : ℕ :=
  wc.m1 * wc.d1

-- Proving the number of days required for m2 men
theorem daysRequired (wc : WallConstruction) (k : ℕ) : 
  k = totalWork wc → (wc.m2 * (k / wc.m2 : ℚ) = k) → (k / wc.m2 : ℚ) = 3.3 :=
by
  intro h1 h2
  sorry

end daysRequired_l1626_162683


namespace num_ways_choose_officers_8_l1626_162604

def numWaysToChooseOfficers (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem num_ways_choose_officers_8 : numWaysToChooseOfficers 8 = 336 := by
  sorry

end num_ways_choose_officers_8_l1626_162604


namespace deepak_age_l1626_162601

theorem deepak_age
  (A D : ℕ)
  (h1 : A / D = 2 / 5)  -- the ratio condition
  (h2 : A + 10 = 30)   -- Arun’s age after 10 years will be 30
  : D = 50 :=       -- conclusion Deepak is 50 years old
sorry

end deepak_age_l1626_162601
