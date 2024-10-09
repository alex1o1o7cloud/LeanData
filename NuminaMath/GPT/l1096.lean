import Mathlib

namespace convert_and_compute_l1096_109604

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  if n = 231 then 2 * 4^2 + 3 * 4^1 + 1 * 4^0
  else if n = 21 then 2 * 4^1 + 1 * 4^0
  else if n = 3 then 3
  else 0

noncomputable def base10_to_base4 (n : ℕ) : ℕ :=
  if n = 135 then 2 * 4^2 + 1 * 4^1 + 3 * 4^0
  else 0

theorem convert_and_compute :
  base10_to_base4 ((base4_to_base10 231 / base4_to_base10 3) * base4_to_base10 21) = 213 :=
by {
  sorry
}

end convert_and_compute_l1096_109604


namespace percentage_second_question_correct_l1096_109679

theorem percentage_second_question_correct (a b c : ℝ) 
  (h1 : a = 0.75) (h2 : b = 0.20) (h3 : c = 0.50) :
  (1 - b) - (a - c) + c = 0.55 :=
by
  sorry

end percentage_second_question_correct_l1096_109679


namespace find_jessica_almonds_l1096_109642

-- Definitions for j (Jessica's almonds) and l (Louise's almonds)
variables (j l : ℕ)
-- Conditions
def condition1 : Prop := l = j - 8
def condition2 : Prop := l = j / 3

theorem find_jessica_almonds (h1 : condition1 j l) (h2 : condition2 j l) : j = 12 :=
by sorry

end find_jessica_almonds_l1096_109642


namespace find_N_l1096_109624

theorem find_N :
  ∃ N : ℕ,
  (5 + 6 + 7 + 8 + 9) / 5 = (2005 + 2006 + 2007 + 2008 + 2009) / (N : ℝ) ∧ N = 1433 :=
sorry

end find_N_l1096_109624


namespace find_number_l1096_109651

theorem find_number (x : ℤ) (h : x - 7 = 9) : x * 3 = 48 :=
by sorry

end find_number_l1096_109651


namespace percentage_equivalence_l1096_109608

theorem percentage_equivalence (x : ℝ) : 0.3 * 0.6 * 0.7 * x = 0.126 * x :=
by
  sorry

end percentage_equivalence_l1096_109608


namespace thirty_ml_of_one_liter_is_decimal_fraction_l1096_109603

-- We define the known conversion rule between liters and milliliters.
def liter_to_ml := 1000

-- We define the volume in milliliters that we are considering.
def volume_ml := 30

-- We state the main theorem which asserts that 30 ml of a liter is equal to the decimal fraction 0.03.
theorem thirty_ml_of_one_liter_is_decimal_fraction : (volume_ml / (liter_to_ml : ℝ)) = 0.03 := by
  -- insert proof here
  sorry

end thirty_ml_of_one_liter_is_decimal_fraction_l1096_109603


namespace sum_xyz_l1096_109682

theorem sum_xyz (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + 2 * (y - 1) * (z - 1) = 85)
  (h2 : y^2 + 2 * (z - 1) * (x - 1) = 84)
  (h3 : z^2 + 2 * (x - 1) * (y - 1) = 89) :
  x + y + z = 18 := 
by
  sorry

end sum_xyz_l1096_109682


namespace proof_S_squared_l1096_109633

variables {a b c p S r r_a r_b r_c : ℝ}

-- Conditions
axiom cond1 : r * p = r_a * (p - a)
axiom cond2 : r * r_a = (p - b) * (p - c)
axiom cond3 : r_b * r_c = p * (p - a)
axiom heron : S^2 = p * (p - a) * (p - b) * (p - c)

-- Proof statement
theorem proof_S_squared : S^2 = r * r_a * r_b * r_c :=
by sorry

end proof_S_squared_l1096_109633


namespace weights_problem_l1096_109621

theorem weights_problem
  (a b c d : ℕ)
  (h1 : a + b = 280)
  (h2 : b + c = 255)
  (h3 : c + d = 290) 
  : a + d = 315 := 
  sorry

end weights_problem_l1096_109621


namespace possible_pairs_copies_each_key_min_drawers_l1096_109650

-- Define the number of distinct keys
def num_keys : ℕ := 10

-- Define the function to calculate the number of pairs
def num_pairs (n : ℕ) := n * (n - 1) / 2

-- Theorem for the first question
theorem possible_pairs : num_pairs num_keys = 45 :=
by sorry

-- Define the number of copies needed for each key
def copies_needed (n : ℕ) := n - 1

-- Theorem for the second question
theorem copies_each_key : copies_needed num_keys = 9 :=
by sorry

-- Define the minimum number of drawers Fernando needs to open
def min_drawers_to_open (n : ℕ) := num_pairs n - (n - 1) + 1

-- Theorem for the third question
theorem min_drawers : min_drawers_to_open num_keys = 37 :=
by sorry

end possible_pairs_copies_each_key_min_drawers_l1096_109650


namespace find_a_extreme_values_l1096_109694

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + 4 * Real.log (x + 1)
noncomputable def f' (x a : ℝ) : ℝ := 2 * (x - a) + 4 / (x + 1)

-- Given conditions
theorem find_a (a : ℝ) :
  f' 1 a = 0 ↔ a = 2 :=
by
  sorry

theorem extreme_values :
  ∃ x : ℝ, -1 < x ∧ f (0 : ℝ) 2 = 4 ∨ f (1 : ℝ) 2 = 1 + 4 * Real.log 2 :=
by
  sorry

end find_a_extreme_values_l1096_109694


namespace sum_of_series_l1096_109698

theorem sum_of_series : 
  (1 / (1 * 2) + 1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7)) = 6 / 7 := 
 by sorry

end sum_of_series_l1096_109698


namespace find_third_number_l1096_109632

noncomputable def averageFirstSet (x : ℝ) : ℝ := (20 + 40 + x) / 3
noncomputable def averageSecondSet : ℝ := (10 + 70 + 16) / 3

theorem find_third_number (x : ℝ) (h : averageFirstSet x = averageSecondSet + 8) : x = 60 :=
by
  sorry

end find_third_number_l1096_109632


namespace union_A_B_l1096_109658

noncomputable def A := {x : ℝ | Real.log x ≤ 0}
noncomputable def B := {x : ℝ | x^2 - 1 < 0}
def A_union_B := {x : ℝ | (Real.log x ≤ 0) ∨ (x^2 - 1 < 0)}

theorem union_A_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by
  -- proof to be added
  sorry

end union_A_B_l1096_109658


namespace perpendicular_vectors_l1096_109607

def vec := ℝ × ℝ

def dot_product (a b : vec) : ℝ :=
  a.1 * b.1 + a.2 * b.2

variables (m : ℝ)
def a : vec := (1, 2)
def b : vec := (m, 1)

theorem perpendicular_vectors (h : dot_product a (b m) = 0) : m = -2 :=
sorry

end perpendicular_vectors_l1096_109607


namespace roots_inequality_l1096_109663

noncomputable def a : ℝ := Real.sqrt 2020

theorem roots_inequality (x1 x2 x3 : ℝ) (h_roots : ∀ x, (a * x^3 - 4040 * x^2 + 4 = 0) ↔ (x = x1 ∨ x = x2 ∨ x = x3))
  (h_inequality: x1 < x2 ∧ x2 < x3) : x2 * (x1 + x3) = 2 :=
sorry

end roots_inequality_l1096_109663


namespace number_of_cows_l1096_109601

-- Definitions
variables (c h : ℕ)

-- Conditions
def condition1 : Prop := 4 * c + 2 * h = 20 + 2 * (c + h)
def condition2 : Prop := c + h = 12

-- Theorem
theorem number_of_cows : condition1 c h → condition2 c h → c = 10 :=
  by 
  intros h1 h2
  sorry

end number_of_cows_l1096_109601


namespace problem_statement_l1096_109613

theorem problem_statement {m n : ℝ} 
  (h1 : (n + 2 * m) / (1 + m ^ 2) = -1 / 2) 
  (h2 : -(1 + n) + 2 * (m + 2) = 0) : 
  (m / n = -1) := 
sorry

end problem_statement_l1096_109613


namespace unrepresentable_integers_l1096_109660

theorem unrepresentable_integers :
    {n : ℕ | ∀ a b : ℕ, a > 0 → b > 0 → n ≠ (a * (b + 1) + (a + 1) * b) / (b * (b + 1)) } =
    {1} ∪ {n | ∃ m : ℕ, n = 2^m + 2} :=
by
    sorry

end unrepresentable_integers_l1096_109660


namespace rabbit_can_escape_l1096_109675

def RabbitEscapeExists
  (center_x : ℝ)
  (center_y : ℝ)
  (wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 : ℝ)
  (wolf_speed rabbit_speed : ℝ)
  (condition1 : center_x = 0 ∧ center_y = 0)
  (condition2 : wolf_x1 = -1 ∧ wolf_y1 = -1 ∧ wolf_x2 = 1 ∧ wolf_y2 = -1 ∧ wolf_x3 = -1 ∧ wolf_y3 = 1 ∧ wolf_x4 = 1 ∧ wolf_y4 = 1)
  (condition3 : wolf_speed = 1.4 * rabbit_speed) : Prop :=
 ∃ (rabbit_escapes : Bool), rabbit_escapes = true

theorem rabbit_can_escape
  (center_x : ℝ)
  (center_y : ℝ)
  (wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 : ℝ)
  (wolf_speed rabbit_speed : ℝ)
  (condition1 : center_x = 0 ∧ center_y = 0)
  (condition2 : wolf_x1 = -1 ∧ wolf_y1 = -1 ∧ wolf_x2 = 1 ∧ wolf_y2 = -1 ∧ wolf_x3 = -1 ∧ wolf_y3 = 1 ∧ wolf_x4 = 1 ∧ wolf_y4 = 1)
  (condition3 : wolf_speed = 1.4 * rabbit_speed) : RabbitEscapeExists center_x center_y wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 wolf_speed rabbit_speed condition1 condition2 condition3 := 
sorry

end rabbit_can_escape_l1096_109675


namespace maria_cupcakes_l1096_109644

variable (initial : ℕ) (additional : ℕ) (remaining : ℕ)

theorem maria_cupcakes (h_initial : initial = 19) (h_additional : additional = 10) (h_remaining : remaining = 24) : initial + additional - remaining = 5 := by
  sorry

end maria_cupcakes_l1096_109644


namespace max_marks_l1096_109617

theorem max_marks (M : ℝ) (h_pass : 0.30 * M = 231) : M = 770 := sorry

end max_marks_l1096_109617


namespace reasoning_is_invalid_l1096_109671

-- Definitions based on conditions
variables {Line Plane : Type} (is_parallel_to : Line → Plane → Prop) (is_parallel_to' : Line → Line → Prop) (is_contained_in : Line → Plane → Prop)

-- Conditions
axiom major_premise (b : Line) (α : Plane) : is_parallel_to b α → ∀ (a : Line), is_contained_in a α → is_parallel_to' b a
axiom minor_premise1 (b : Line) (α : Plane) : is_parallel_to b α
axiom minor_premise2 (a : Line) (α : Plane) : is_contained_in a α

-- Conclusion
theorem reasoning_is_invalid : ∃ (a : Line) (b : Line) (α : Plane), ¬ (is_parallel_to b α → ∀ (a : Line), is_contained_in a α → is_parallel_to' b a) :=
sorry

end reasoning_is_invalid_l1096_109671


namespace domain_and_monotone_l1096_109631

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem domain_and_monotone :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → ∃ y, f x = y) ∧
  ∀ x1 x2 : ℝ, 1 < x1 ∧ x1 < x2 → f x1 < f x2 :=
by
  sorry

end domain_and_monotone_l1096_109631


namespace evaluate_expression_l1096_109612

theorem evaluate_expression : 3002^3 - 3001 * 3002^2 - 3001^2 * 3002 + 3001^3 + 1 = 6004 :=
by
  sorry

end evaluate_expression_l1096_109612


namespace matching_pairs_less_than_21_in_at_least_61_positions_l1096_109685

theorem matching_pairs_less_than_21_in_at_least_61_positions :
  ∀ (disks : ℕ) (total_sectors : ℕ) (red_sectors : ℕ) (max_overlap : ℕ) (rotations : ℕ),
  disks = 2 →
  total_sectors = 1965 →
  red_sectors = 200 →
  max_overlap = 20 →
  rotations = total_sectors →
  (∃ positions, positions = total_sectors - (red_sectors * red_sectors / (max_overlap + 1)) ∧ positions ≤ rotations) →
  positions = 61 :=
by {
  -- Placeholder to provide the structure of the theorem.
  sorry
}

end matching_pairs_less_than_21_in_at_least_61_positions_l1096_109685


namespace Yvonne_laps_l1096_109648

-- Definitions of the given conditions
def laps_swim_by_Yvonne (l_y : ℕ) : Prop := 
  ∃ l_s l_j, 
  l_s = l_y / 2 ∧ 
  l_j = 3 * l_s ∧ 
  l_j = 15

-- Theorem statement
theorem Yvonne_laps (l_y : ℕ) (h : laps_swim_by_Yvonne l_y) : l_y = 10 :=
sorry

end Yvonne_laps_l1096_109648


namespace find_value_of_expression_l1096_109668

theorem find_value_of_expression (y : ℝ) (h : 6 * y^2 + 7 = 4 * y + 13) : (12 * y - 5)^2 = 161 :=
sorry

end find_value_of_expression_l1096_109668


namespace middle_of_7_consecutive_nat_sum_63_l1096_109639

theorem middle_of_7_consecutive_nat_sum_63 (x : ℕ) (h : 7 * x = 63) : x = 9 :=
by
  sorry

end middle_of_7_consecutive_nat_sum_63_l1096_109639


namespace ratio_of_spinsters_to_cats_l1096_109687

-- Defining the problem in Lean 4
theorem ratio_of_spinsters_to_cats (S C : ℕ) (h₁ : S = 22) (h₂ : C = S + 55) : S / gcd S C = 2 ∧ C / gcd S C = 7 :=
by
  sorry

end ratio_of_spinsters_to_cats_l1096_109687


namespace complement_of_67_is_23_l1096_109673

-- Define complement function
def complement (x : ℝ) : ℝ := 90 - x

-- State the theorem
theorem complement_of_67_is_23 : complement 67 = 23 := 
by
  sorry

end complement_of_67_is_23_l1096_109673


namespace smaller_number_is_72_l1096_109616

theorem smaller_number_is_72
  (x : ℝ)
  (h1 : (3 * x - 24) / (8 * x - 24) = 4 / 9)
  : 3 * x = 72 :=
sorry

end smaller_number_is_72_l1096_109616


namespace sin_of_angle_l1096_109697

theorem sin_of_angle (α : ℝ) (h : Real.cos (π + α) = -(1/3)) : Real.sin ((3 * π / 2) - α) = -(1/3) := 
by
  sorry

end sin_of_angle_l1096_109697


namespace possible_values_of_Q_l1096_109666

theorem possible_values_of_Q (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (x + y) / z = (y + z) / x ∧ (y + z) / x = (z + x) / y) :
  ∃ Q : ℝ, Q = 8 ∨ Q = -1 := 
sorry

end possible_values_of_Q_l1096_109666


namespace minimum_value_l1096_109611

theorem minimum_value : 
  ∀ a b : ℝ, 0 < a → 0 < b → a + 2 * b = 3 → (1 / a + 1 / b) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
by
  sorry

end minimum_value_l1096_109611


namespace range_of_a_h_diff_l1096_109619

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x

theorem range_of_a (a : ℝ) (h : a < 0) : (∀ x, 0 < x ∧ x < Real.log 3 → 
  (a * x - 1) / x < 0 ∧ Real.exp x + a ≠ 0 ∧ (a ≤ -3)) :=
sorry

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + Real.log x

theorem h_diff (a : ℝ) (x1 x2 : ℝ) (hx1 : 0 < x1 ∧ x1 < 1/2) : 
    x1 * x2 = 1/2 ∧ h a x1 - h a x2 > 3/4 - Real.log 2 :=
sorry

end range_of_a_h_diff_l1096_109619


namespace smaller_circle_radius_l1096_109657

noncomputable def radius_of_smaller_circles (R : ℝ) (r1 r2 r3 : ℝ) (OA OB OC : ℝ) : Prop :=
(OA = R + r1) ∧ (OB = R + 3 * r1) ∧ (OC = R + 5 * r1) ∧ 
((OB = OA + 2 * r1) ∧ (OC = OB + 2 * r1))

theorem smaller_circle_radius (r : ℝ) (R : ℝ := 2) :
  radius_of_smaller_circles R r r r (R + r) (R + 3 * r) (R + 5 * r) → r = 1 :=
by
  sorry

end smaller_circle_radius_l1096_109657


namespace percentage_apples_sold_l1096_109614

theorem percentage_apples_sold (A P : ℕ) (h1 : A = 600) (h2 : A * (100 - P) / 100 = 420) : P = 30 := 
by {
  sorry
}

end percentage_apples_sold_l1096_109614


namespace thor_fraction_correct_l1096_109667

-- Define the initial conditions
def moes_money : ℕ := 12
def lokis_money : ℕ := 10
def nicks_money : ℕ := 8
def otts_money : ℕ := 6

def thor_received_from_each : ℕ := 2

-- Calculate total money each time
def total_initial_money : ℕ := moes_money + lokis_money + nicks_money + otts_money
def thor_total_received : ℕ := 4 * thor_received_from_each
def thor_fraction_of_total : ℚ := thor_total_received / total_initial_money

-- The theorem to prove
theorem thor_fraction_correct : thor_fraction_of_total = 2/9 :=
by
  sorry

end thor_fraction_correct_l1096_109667


namespace min_value_expression_geq_twosqrt3_l1096_109600

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  (1/(x-1)) + (3/(y-1))

theorem min_value_expression_geq_twosqrt3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1/x) + (1/y) = 1) : 
  min_value_expression x y >= 2 * Real.sqrt 3 :=
by
  sorry

end min_value_expression_geq_twosqrt3_l1096_109600


namespace problem_solution_l1096_109652

theorem problem_solution 
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h₁ : a = ⌊2 + Real.sqrt 2⌋) 
  (h₂ : b = (2 + Real.sqrt 2) - ⌊2 + Real.sqrt 2⌋)
  (h₃ : c = ⌊4 - Real.sqrt 2⌋)
  (h₄ : d = (4 - Real.sqrt 2) - ⌊4 - Real.sqrt 2⌋) :
  (b + d) / (a * c) = 1 / 6 :=
by
  sorry

end problem_solution_l1096_109652


namespace problem_l1096_109680

noncomputable def f (x a b : ℝ) := x^2 + a*x + b
noncomputable def g (x c d : ℝ) := x^2 + c*x + d

theorem problem (a b c d : ℝ) (h_min_f : f (-a/2) a b = -25) (h_min_g : g (-c/2) c d = -25)
  (h_intersection_f : f 50 a b = -50) (h_intersection_g : g 50 c d = -50)
  (h_root_f_of_g : g (-a/2) c d = 0) (h_root_g_of_f : f (-c/2) a b = 0) :
  a + c = -200 := by
  sorry

end problem_l1096_109680


namespace initial_bushes_l1096_109695

theorem initial_bushes (b : ℕ) (h1 : b + 4 = 6) : b = 2 :=
by {
  sorry
}

end initial_bushes_l1096_109695


namespace compare_f_values_l1096_109610

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - Real.cos x

theorem compare_f_values :
  f 0 < f 0.5 ∧ f 0.5 < f 0.6 :=
by {
  -- proof would go here
  sorry
}

end compare_f_values_l1096_109610


namespace ellipse_condition_l1096_109649

theorem ellipse_condition (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (m - 2) + (y^2) / (6 - m) = 1) →
  (2 < m ∧ m < 6 ∧ m ≠ 4) :=
by
  sorry

end ellipse_condition_l1096_109649


namespace exponent_property_l1096_109686

theorem exponent_property (a : ℝ) (m n : ℝ) (h₁ : a^m = 4) (h₂ : a^n = 8) : a^(m + n) = 32 := 
by 
  sorry

end exponent_property_l1096_109686


namespace sum_of_consecutive_2022_l1096_109662

theorem sum_of_consecutive_2022 (m n : ℕ) (h : m ≤ n - 1) (sum_eq : (n - m + 1) * (m + n) = 4044) :
  (m = 163 ∧ n = 174) ∨ (m = 504 ∧ n = 507) ∨ (m = 673 ∧ n = 675) :=
sorry

end sum_of_consecutive_2022_l1096_109662


namespace carol_optimal_strategy_l1096_109646

-- Definitions of the random variables
def uniform_A (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1
def uniform_B (b : ℝ) : Prop := 0.25 ≤ b ∧ b ≤ 0.75
def winning_condition (a b c : ℝ) : Prop := (a < c ∧ c < b) ∨ (b < c ∧ c < a)

-- Carol's optimal strategy stated as a theorem
theorem carol_optimal_strategy : ∀ (a b c : ℝ), 
  uniform_A a → uniform_B b → (c = 7 / 12) → 
  winning_condition a b c → 
  ∀ (c' : ℝ), uniform_A c' → c' ≠ c → ¬(winning_condition a b c') :=
by
  sorry

end carol_optimal_strategy_l1096_109646


namespace correct_actual_profit_l1096_109618

def profit_miscalculation (calculated_profit actual_profit : ℕ) : Prop :=
  let err1 := 5 * 100  -- Error due to mistaking 3 for 8 in the hundreds place
  let err2 := 3 * 10   -- Error due to mistaking 8 for 5 in the tens place
  actual_profit = calculated_profit - err1 + err2

theorem correct_actual_profit : profit_miscalculation 1320 850 :=
by
  sorry

end correct_actual_profit_l1096_109618


namespace pentagon_area_l1096_109629

variable (a b c d e : ℕ)
variable (r s : ℕ)

-- Given conditions
axiom H₁: a = 14
axiom H₂: b = 35
axiom H₃: c = 42
axiom H₄: d = 14
axiom H₅: e = 35
axiom H₆: r = 21
axiom H₇: s = 28
axiom H₈: r^2 + s^2 = e^2

-- Question: Prove that the area of the pentagon is 1176
theorem pentagon_area : b * c - (1 / 2) * r * s = 1176 := 
by 
  sorry

end pentagon_area_l1096_109629


namespace length_of_goods_train_l1096_109684

-- Define the given conditions
def speed_kmph := 72
def platform_length := 260
def crossing_time := 26

-- Convert speed to m/s
def speed_mps := (speed_kmph * 5) / 18

-- Calculate distance covered
def distance_covered := speed_mps * crossing_time

-- Define the length of the train
def train_length := distance_covered - platform_length

theorem length_of_goods_train : train_length = 260 := by
  sorry

end length_of_goods_train_l1096_109684


namespace unique_games_count_l1096_109665

noncomputable def total_games_played (n : ℕ) (m : ℕ) : ℕ :=
  (n * m) / 2

theorem unique_games_count (students : ℕ) (games_per_student : ℕ) (h1 : students = 9) (h2 : games_per_student = 6) :
  total_games_played students games_per_student = 27 :=
by
  rw [h1, h2]
  -- This partially evaluates total_games_played using the values from h1 and h2.
  -- Performing actual proof steps is not necessary, so we'll use sorry.
  sorry

end unique_games_count_l1096_109665


namespace sum_of_digits_base10_representation_l1096_109691

def digit_sum (n : ℕ) : ℕ := sorry  -- Define a function to calculate the sum of digits

noncomputable def a : ℕ := 7 * (10 ^ 1234 - 1) / 9
noncomputable def b : ℕ := 2 * (10 ^ 1234 - 1) / 9
noncomputable def product : ℕ := 7 * a * b

theorem sum_of_digits_base10_representation : digit_sum product = 11100 := 
by sorry

end sum_of_digits_base10_representation_l1096_109691


namespace arithmetic_seq_a5_l1096_109615

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

theorem arithmetic_seq_a5 (h1 : is_arithmetic_sequence a) (h2 : a 2 + a 8 = 12) :
  a 5 = 6 :=
by
  sorry

end arithmetic_seq_a5_l1096_109615


namespace sin_double_theta_eq_three_fourths_l1096_109677

theorem sin_double_theta_eq_three_fourths (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2)
  (h1 : Real.sin (π * Real.cos θ) = Real.cos (π * Real.sin θ)) :
  Real.sin (2 * θ) = 3 / 4 :=
  sorry

end sin_double_theta_eq_three_fourths_l1096_109677


namespace min_B_minus_A_l1096_109688

noncomputable def S_n (n : ℕ) : ℚ :=
  let a1 : ℚ := 2
  let r : ℚ := -1 / 3
  a1 * (1 - r ^ n) / (1 - r)

theorem min_B_minus_A :
  ∃ A B : ℚ, 
    (∀ n : ℕ, 1 ≤ n → A ≤ 3 * S_n n - 1 / S_n n ∧ 3 * S_n n - 1 / S_n n ≤ B) ∧
    ∀ A' B' : ℚ, 
      (∀ n : ℕ, 1 ≤ n → A' ≤ 3 * S_n n - 1 / S_n n ∧ 3 * S_n n - 1 / S_n n ≤ B') → 
      B' - A' ≥ 9 / 4 ∧ B - A = 9 / 4 :=
sorry

end min_B_minus_A_l1096_109688


namespace part1_part2_part3_l1096_109627

open Real

-- Definitions of points
structure Point :=
(x : ℝ)
(y : ℝ)

def M (m : ℝ) : Point := ⟨m - 2, 2 * m - 7⟩
def N (n : ℝ) : Point := ⟨n, 3⟩

-- Part 1
theorem part1 : 
  (M (7 / 2)).y = 0 ∧ (M (7 / 2)).x = 3 / 2 :=
by
  sorry

-- Part 2
theorem part2 (m : ℝ) : abs (m - 2) = abs (2 * m - 7) → (m = 5 ∨ m = 3) :=
by
  sorry

-- Part 3
theorem part3 (m n : ℝ) : abs ((M m).y - 3) = 2 ∧ (M m).x = n - 2 → (n = 4 ∨ n = 2) :=
by
  sorry

end part1_part2_part3_l1096_109627


namespace odds_against_C_winning_l1096_109659

theorem odds_against_C_winning (prob_A: ℚ) (prob_B: ℚ) (prob_C: ℚ)
    (odds_A: prob_A = 1 / 5) (odds_B: prob_B = 2 / 5) 
    (total_prob: prob_A + prob_B + prob_C = 1):
    ((1 - prob_C) / prob_C) = 3 / 2 :=
by
  sorry

end odds_against_C_winning_l1096_109659


namespace example_equation_l1096_109683

-- Define what it means to be an equation in terms of containing an unknown and being an equality
def is_equation (expr : Prop) (contains_unknown : Prop) : Prop :=
  (contains_unknown ∧ expr)

-- Prove that 4x + 2 = 10 is an equation
theorem example_equation : is_equation (4 * x + 2 = 10) (∃ x : ℝ, true) :=
  by sorry

end example_equation_l1096_109683


namespace oblong_perimeter_182_l1096_109696

variables (l w : ℕ) (x : ℤ)

def is_oblong (l w : ℕ) : Prop :=
l * w = 4624 ∧ l = 4 * x ∧ w = 3 * x

theorem oblong_perimeter_182 (l w x : ℕ) (hlw : is_oblong l w x) : 
  2 * l + 2 * w = 182 :=
by
  sorry

end oblong_perimeter_182_l1096_109696


namespace number_of_pairs_l1096_109638

theorem number_of_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m^2 + n < 50) : 
  ∃! p : ℕ, p = 203 := 
sorry

end number_of_pairs_l1096_109638


namespace amount_saved_by_Dalton_l1096_109676

-- Defining the costs of each item and the given conditions
def jump_rope_cost : ℕ := 7
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def uncle_gift : ℕ := 13
def additional_needed : ℕ := 4

-- Calculated values based on the conditions
def total_cost : ℕ := jump_rope_cost + board_game_cost + playground_ball_cost
def total_money_needed : ℕ := uncle_gift + additional_needed

-- The theorem that needs to be proved
theorem amount_saved_by_Dalton : total_cost - total_money_needed = 6 :=
by
  sorry -- Proof to be filled in

end amount_saved_by_Dalton_l1096_109676


namespace find_power_of_4_l1096_109623

theorem find_power_of_4 (x : Nat) : 
  (2 * x + 5 + 2 = 29) -> 
  (x = 11) :=
by
  sorry

end find_power_of_4_l1096_109623


namespace evaluate_expression_l1096_109625

theorem evaluate_expression (m n : ℝ) (h : 4 * m - 4 + n = 2) : 
  (m * (-2)^2 - 2 * (-2) + n = 10) :=
by
  sorry

end evaluate_expression_l1096_109625


namespace single_intersection_point_l1096_109637

theorem single_intersection_point (k : ℝ) :
  (∃! x : ℝ, x^2 - 2 * x - k = 0) ↔ k = 0 :=
by
  sorry

end single_intersection_point_l1096_109637


namespace sequence_formula_min_value_Sn_min_value_Sn_completion_l1096_109693

-- Define the sequence sum Sn
def Sn (n : ℕ) : ℤ := n^2 - 48 * n

-- General term of the sequence
def an (n : ℕ) : ℤ :=
  match n with
  | 0     => 0 -- Conventionally, sequences start from 1 in these problems
  | (n+1) => 2 * (n + 1) - 49

-- Prove that the general term of the sequence produces the correct sum
theorem sequence_formula (n : ℕ) (h : 0 < n) : an n = 2 * n - 49 := by
  sorry

-- Prove that the minimum value of Sn is -576 and occurs at n = 24
theorem min_value_Sn : ∃ n : ℕ, Sn n = -576 ∧ ∀ m : ℕ, Sn m ≥ -576 := by
  use 24
  sorry

-- Alternative form of the theorem using the square completion form 
theorem min_value_Sn_completion (n : ℕ) : Sn n = (n - 24)^2 - 576 := by
  sorry

end sequence_formula_min_value_Sn_min_value_Sn_completion_l1096_109693


namespace largest_x_exists_largest_x_largest_real_number_l1096_109647

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l1096_109647


namespace initial_oranges_l1096_109674

theorem initial_oranges (X : ℕ) (h1 : X - 37 + 7 = 10) : X = 40 :=
by
  sorry

end initial_oranges_l1096_109674


namespace cone_plane_distance_l1096_109672

theorem cone_plane_distance (H α : ℝ) : 
  (x = 2 * H * (Real.sin (α / 4)) ^ 2) :=
sorry

end cone_plane_distance_l1096_109672


namespace alpha_value_l1096_109655

theorem alpha_value (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.logb 3 (x + 1)) (h2 : f α = 1) : α = 2 := by
  sorry

end alpha_value_l1096_109655


namespace parallel_lines_iff_a_eq_neg3_l1096_109635

theorem parallel_lines_iff_a_eq_neg3 (a : ℝ) :
  (∀ x y : ℝ, a * x + 3 * y + 1 = 0 → 2 * x + (a + 1) * y + 1 ≠ 0) ↔ a = -3 :=
sorry

end parallel_lines_iff_a_eq_neg3_l1096_109635


namespace find_g_l1096_109602

variable (x : ℝ)

theorem find_g :
  ∃ g : ℝ → ℝ, 2 * x ^ 5 + 4 * x ^ 3 - 3 * x + 5 + g x = 3 * x ^ 4 + 7 * x ^ 2 - 2 * x - 4 ∧
                g x = -2 * x ^ 5 + 3 * x ^ 4 - 4 * x ^ 3 + 7 * x ^ 2 - x - 9 :=
sorry

end find_g_l1096_109602


namespace corveus_sleep_deficit_l1096_109622

theorem corveus_sleep_deficit :
  let weekday_sleep := 5 -- 4 hours at night + 1-hour nap
  let weekend_sleep := 5 -- 5 hours at night, no naps
  let total_weekday_sleep := 5 * weekday_sleep
  let total_weekend_sleep := 2 * weekend_sleep
  let total_sleep := total_weekday_sleep + total_weekend_sleep
  let recommended_sleep_per_day := 6
  let total_recommended_sleep := 7 * recommended_sleep_per_day
  let sleep_deficit := total_recommended_sleep - total_sleep
  sleep_deficit = 7 :=
by
  -- Insert proof steps here
  sorry

end corveus_sleep_deficit_l1096_109622


namespace smallest_value_of_Q_l1096_109628

def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - 4*x^2 + 2*x - 3

theorem smallest_value_of_Q :
  min (-10) (min 3 (-2)) = -10 :=
by
  -- Skip the proof
  sorry

end smallest_value_of_Q_l1096_109628


namespace factories_checked_by_second_group_l1096_109605

theorem factories_checked_by_second_group 
(T : ℕ) (G1 : ℕ) (R : ℕ) 
(hT : T = 169) 
(hG1 : G1 = 69) 
(hR : R = 48) : 
T - (G1 + R) = 52 :=
by {
  sorry
}

end factories_checked_by_second_group_l1096_109605


namespace train_speed_in_kmh_l1096_109681

def train_length : ℝ := 250 -- Length of the train in meters
def station_length : ℝ := 200 -- Length of the station in meters
def time_to_pass : ℝ := 45 -- Time to pass the station in seconds

theorem train_speed_in_kmh :
  (train_length + station_length) / time_to_pass * 3.6 = 36 :=
  sorry -- Proof is skipped

end train_speed_in_kmh_l1096_109681


namespace opposite_of_neg_2_l1096_109669

theorem opposite_of_neg_2 : ∃ y : ℝ, -2 + y = 0 ∧ y = 2 := by
  sorry

end opposite_of_neg_2_l1096_109669


namespace remainder_when_add_13_l1096_109653

theorem remainder_when_add_13 (x : ℤ) (h : x % 82 = 5) : (x + 13) % 41 = 18 :=
sorry

end remainder_when_add_13_l1096_109653


namespace chocolates_bought_l1096_109641

theorem chocolates_bought (cost_price selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : cost_price * 24 = selling_price)
  (h2 : gain_percent = 83.33333333333334)
  (h3 : selling_price = cost_price * 24 * (1 + gain_percent / 100)) :
  cost_price * 44 = selling_price :=
by
  sorry

end chocolates_bought_l1096_109641


namespace mulberry_sales_l1096_109678

theorem mulberry_sales (x : ℝ) (p : ℝ) (h1 : 3000 = x * p)
    (h2 : 150 * (p * 1.4) + (x - 150) * (p * 0.8) - 3000 = 750) :
    x = 200 := by sorry

end mulberry_sales_l1096_109678


namespace factorize_3x2_minus_3y2_l1096_109645

theorem factorize_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factorize_3x2_minus_3y2_l1096_109645


namespace proof_part_a_l1096_109636

variable {α : Type} [LinearOrder α]

structure ConvexQuadrilateral (α : Type) :=
(a b c d : α)
(a'b'c'd' : α)
(ab_eq_a'b' : α)
(bc_eq_b'c' : α)
(cd_eq_c'd' : α)
(da_eq_d'a' : α)
(angle_A_gt_angle_A' : Prop)
(angle_B_lt_angle_B' : Prop)
(angle_C_gt_angle_C' : Prop)
(angle_D_lt_angle_D' : Prop)

theorem proof_part_a (Quad : ConvexQuadrilateral ℝ) : 
  Quad.angle_A_gt_angle_A' → 
  Quad.angle_B_lt_angle_B' ∧ Quad.angle_C_gt_angle_C' ∧ Quad.angle_D_lt_angle_D' := sorry

end proof_part_a_l1096_109636


namespace total_blocks_in_pyramid_l1096_109689

-- Define the number of blocks in each layer
def blocks_in_layer (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n+1) => 3 * blocks_in_layer n

-- Prove the total number of blocks in the four-layer pyramid
theorem total_blocks_in_pyramid : 
  (blocks_in_layer 0) + (blocks_in_layer 1) + (blocks_in_layer 2) + (blocks_in_layer 3) = 40 :=
by
  sorry

end total_blocks_in_pyramid_l1096_109689


namespace hyperbola_asymptotes_l1096_109606

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (y^2 / 9 - x^2 / 4 = 1 →
  (y = (3 / 2) * x ∨ y = - (3 / 2) * x)) :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l1096_109606


namespace maximum_quadratic_expr_l1096_109670

noncomputable def quadratic_expr (x : ℝ) : ℝ :=
  -5 * x^2 + 25 * x - 7

theorem maximum_quadratic_expr : ∃ x : ℝ, quadratic_expr x = 53 / 4 :=
by
  sorry

end maximum_quadratic_expr_l1096_109670


namespace tan_expression_value_l1096_109630

noncomputable def sequence_properties (a b : ℕ → ℝ) :=
  (a 0 * a 5 * a 10 = -3 * Real.sqrt 3) ∧
  (b 0 + b 5 + b 10 = 7 * Real.pi) ∧
  (∀ n, a (n + 1) = a n * a 1) ∧
  (∀ n, b (n + 1) = b n + (b 1 - b 0))

theorem tan_expression_value (a b : ℕ → ℝ) (h : sequence_properties a b) :
  Real.tan (b 2 + b 8) / (1 - a 3 * a 7) = -Real.sqrt 3 :=
sorry

end tan_expression_value_l1096_109630


namespace find_R_position_l1096_109661

theorem find_R_position :
  ∀ (P Q R : ℤ), P = -6 → Q = -1 → Q = (P + R) / 2 → R = 4 :=
by
  intros P Q R hP hQ hQ_halfway
  sorry

end find_R_position_l1096_109661


namespace yara_total_earnings_l1096_109699

-- Lean code to represent the conditions and the proof statement

theorem yara_total_earnings
  (x : ℕ)  -- Yara's hourly wage
  (third_week_hours : ℕ := 18)
  (previous_week_hours : ℕ := 12)
  (extra_earnings : ℕ := 36)
  (third_week_earning : ℕ := third_week_hours * x)
  (previous_week_earning : ℕ := previous_week_hours * x)
  (total_earning : ℕ := third_week_earning + previous_week_earning) :
  third_week_earning = previous_week_earning + extra_earnings → 
  total_earning = 180 := 
by
  -- Proof here
  sorry

end yara_total_earnings_l1096_109699


namespace minimize_value_l1096_109690

noncomputable def minimize_y (a b x : ℝ) : ℝ := (x - a) ^ 3 + (x - b) ^ 3

theorem minimize_value (a b : ℝ) : ∃ x : ℝ, minimize_y a b x = minimize_y a b a ∨ minimize_y a b x = minimize_y a b b :=
sorry

end minimize_value_l1096_109690


namespace expected_pairs_correct_l1096_109626

-- Define the total number of cards in the deck.
def total_cards : ℕ := 52

-- Define the number of black cards in the deck.
def black_cards : ℕ := 26

-- Define the number of red cards in the deck.
def red_cards : ℕ := 26

-- Define the expected number of pairs of adjacent cards such that one is black and the other is red.
def expected_adjacent_pairs := 52 * (26 / 51)

-- Prove that the expected_adjacent_pairs is equal to 1352 / 51.
theorem expected_pairs_correct : expected_adjacent_pairs = 1352 / 51 := 
by
  have expected_adjacent_pairs_simplified : 52 * (26 / 51) = (1352 / 51) := 
    by sorry
  exact expected_adjacent_pairs_simplified

end expected_pairs_correct_l1096_109626


namespace coastal_city_spending_l1096_109664

def beginning_of_may_spending : ℝ := 1.2
def end_of_september_spending : ℝ := 4.5

theorem coastal_city_spending :
  (end_of_september_spending - beginning_of_may_spending) = 3.3 :=
by
  -- Proof can be filled in here
  sorry

end coastal_city_spending_l1096_109664


namespace binom_divisibility_by_prime_l1096_109640

-- Given definitions
variable (p k : ℕ) (hp : Nat.Prime p) (hk1 : 2 ≤ k) (hk2 : k ≤ p - 2)

-- Main theorem statement
theorem binom_divisibility_by_prime
  (hp : Nat.Prime p) (hk1 : 2 ≤ k) (hk2 : k ≤ p - 2) :
  Nat.choose (p - k + 1) k - Nat.choose (p - k - 1) (k - 2) ≡ 0 [MOD p] :=
sorry

end binom_divisibility_by_prime_l1096_109640


namespace find_four_numbers_proportion_l1096_109620

theorem find_four_numbers_proportion :
  ∃ (a b c d : ℝ), 
  a + d = 14 ∧
  b + c = 11 ∧
  a^2 + b^2 + c^2 + d^2 = 221 ∧
  a * d = b * c ∧
  a = 12 ∧
  b = 8 ∧
  c = 3 ∧
  d = 2 :=
by
  sorry

end find_four_numbers_proportion_l1096_109620


namespace hammers_ordered_in_october_l1096_109692

theorem hammers_ordered_in_october
  (ordered_in_june : Nat)
  (ordered_in_july : Nat)
  (ordered_in_august : Nat)
  (ordered_in_september : Nat)
  (pattern_increase : ∀ n : Nat, ordered_in_june + n = ordered_in_july ∧ ordered_in_july + (n + 1) = ordered_in_august ∧ ordered_in_august + (n + 2) = ordered_in_september) :
  ordered_in_september + 4 = 13 :=
by
  -- Proof omitted
  sorry

end hammers_ordered_in_october_l1096_109692


namespace ravioli_to_tortellini_ratio_l1096_109656

-- Definitions from conditions
def total_students : ℕ := 800
def ravioli_students : ℕ := 300
def tortellini_students : ℕ := 150

-- Ratio calculation as a theorem
theorem ravioli_to_tortellini_ratio : 2 = ravioli_students / Nat.gcd ravioli_students tortellini_students :=
by
  -- Given the defined values
  have gcd_val : Nat.gcd ravioli_students tortellini_students = 150 := by
    sorry
  have ratio_simp : ravioli_students / 150 = 2 := by
    sorry
  exact ratio_simp

end ravioli_to_tortellini_ratio_l1096_109656


namespace fraction_of_crop_brought_to_BC_l1096_109634

/-- Consider a kite-shaped field with sides AB = 120 m, BC = CD = 80 m, DA = 120 m.
    The angle between sides AB and BC is 120°, and between sides CD and DA is also 120°.
    Prove that the fraction of the crop brought to the longest side BC is 1/2. -/
theorem fraction_of_crop_brought_to_BC :
  ∀ (AB BC CD DA : ℝ) (α β : ℝ),
  AB = 120 ∧ BC = 80 ∧ CD = 80 ∧ DA = 120 ∧ α = 120 ∧ β = 120 →
  ∃ (frac : ℝ), frac = 1 / 2 :=
by
  intros AB BC CD DA α β h
  sorry

end fraction_of_crop_brought_to_BC_l1096_109634


namespace average_speed_l1096_109609

theorem average_speed :
  ∀ (initial_odometer final_odometer total_time : ℕ), 
    initial_odometer = 2332 →
    final_odometer = 2772 →
    total_time = 8 →
    (final_odometer - initial_odometer) / total_time = 55 :=
by
  intros initial_odometer final_odometer total_time h_initial h_final h_time
  sorry

end average_speed_l1096_109609


namespace parabola_vertex_point_l1096_109654

theorem parabola_vertex_point (a b c : ℝ) 
    (h_vertex : ∃ a b c : ℝ, ∀ x : ℝ, y = a * x^2 + b * x + c) 
    (h_vertex_coord : ∃ (h k : ℝ), h = 3 ∧ k = -5) 
    (h_pass : ∃ (x y : ℝ), x = 0 ∧ y = -2) :
    c = -2 := by
  sorry

end parabola_vertex_point_l1096_109654


namespace choose_math_class_representative_l1096_109643

def number_of_boys : Nat := 26
def number_of_girls : Nat := 24

theorem choose_math_class_representative : number_of_boys + number_of_girls = 50 := 
by
  sorry

end choose_math_class_representative_l1096_109643
