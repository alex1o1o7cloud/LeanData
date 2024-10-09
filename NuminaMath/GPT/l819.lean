import Mathlib

namespace chandler_weeks_to_buy_bike_l819_81905

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

end chandler_weeks_to_buy_bike_l819_81905


namespace simplify_expression_l819_81906

theorem simplify_expression : 
  (4 * 6 / (12 * 14)) * (8 * 12 * 14 / (4 * 6 * 8)) = 1 := by
  sorry

end simplify_expression_l819_81906


namespace column_sum_correct_l819_81940

theorem column_sum_correct : 
  -- Define x to be the sum of the first column (which is also the minuend of the second column)
  ∃ x : ℕ, 
  -- x should match the expected valid sum provided:
  (x = 1001) := 
sorry

end column_sum_correct_l819_81940


namespace range_of_a_l819_81960

theorem range_of_a (a : ℝ) (h_a : a > 0) :
  (∃ t : ℝ, (5 * t + 1)^2 + (12 * t - 1)^2 = 2 * a * (5 * t + 1)) ↔ (0 < a ∧ a ≤ 17 / 25) := 
sorry

end range_of_a_l819_81960


namespace cricket_team_members_count_l819_81966

theorem cricket_team_members_count 
(captain_age : ℕ) (wk_keeper_age : ℕ) (whole_team_avg_age : ℕ)
(remaining_players_avg_age : ℕ) (n : ℕ) 
(h1 : captain_age = 28)
(h2 : wk_keeper_age = captain_age + 3)
(h3 : whole_team_avg_age = 25)
(h4 : remaining_players_avg_age = 24)
(h5 : (n * whole_team_avg_age - (captain_age + wk_keeper_age)) / (n - 2) = remaining_players_avg_age) :
n = 11 := 
sorry

end cricket_team_members_count_l819_81966


namespace number_of_lines_through_focus_intersecting_hyperbola_l819_81999

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

end number_of_lines_through_focus_intersecting_hyperbola_l819_81999


namespace medical_team_formation_l819_81955

theorem medical_team_formation (m f : ℕ) (h_m : m = 5) (h_f : f = 4) :
  (m + f).choose 3 - m.choose 3 - f.choose 3 = 70 :=
by
  sorry

end medical_team_formation_l819_81955


namespace new_average_after_17th_l819_81995

def old_average (A : ℕ) (n : ℕ) : ℕ :=
  A -- A is the average before the 17th inning

def runs_in_17th : ℕ := 84 -- The score in the 17th inning

def average_increase : ℕ := 3 -- The increase in average after the 17th inning

theorem new_average_after_17th (A : ℕ) (n : ℕ) (h1 : n = 16) (h2 : old_average A n + average_increase = A + 3) :
  (old_average A n) + average_increase = 36 :=
by
  sorry

end new_average_after_17th_l819_81995


namespace operation_on_b_l819_81961

variables (t b b' : ℝ)
variable (C : ℝ := t * b ^ 4)
variable (e : ℝ := 16 * C)

theorem operation_on_b :
  tb'^4 = 16 * tb^4 → b' = 2 * b := by
  sorry

end operation_on_b_l819_81961


namespace relationship_xy_l819_81989

def M (x : ℤ) : Prop := ∃ m : ℤ, x = 3 * m + 1
def N (y : ℤ) : Prop := ∃ n : ℤ, y = 3 * n + 2

theorem relationship_xy (x y : ℤ) (hx : M x) (hy : N y) : N (x * y) ∧ ¬ M (x * y) :=
by
  sorry

end relationship_xy_l819_81989


namespace meet_starting_point_together_at_7_40_AM_l819_81996

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

end meet_starting_point_together_at_7_40_AM_l819_81996


namespace number_of_unique_products_l819_81959

-- Define the sets a and b
def setA : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}
def setB : Set ℕ := {2, 4, 6, 19, 21, 24, 27, 31, 35}

-- Define the number of unique products
def numUniqueProducts : ℕ := 405

-- Statement that needs to be proved
theorem number_of_unique_products :
  (∀ A1 ∈ setA, ∀ B ∈ setB, ∀ A2 ∈ setA, ∃ p, p = A1 * B * A2) ∧ 
  (∃ count, count = 45 * 9) ∧ 
  (∃ result, result = numUniqueProducts) :=
  by {
    sorry
  }

end number_of_unique_products_l819_81959


namespace inscribed_square_ab_l819_81917

theorem inscribed_square_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^2 + b^2 = 32) : 2 * a * b = -7 :=
by
  sorry

end inscribed_square_ab_l819_81917


namespace min_value_ge_54_l819_81904

open Real

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) : ℝ :=
2 * x + 3 * y + 6 * z

theorem min_value_ge_54 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) :
  min_value x y z h1 h2 h3 h4 ≥ 54 :=
sorry

end min_value_ge_54_l819_81904


namespace neither_sufficient_nor_necessary_condition_l819_81937

noncomputable def p (x : ℝ) : Prop := (x - 2) * (x - 1) > 0

noncomputable def q (x : ℝ) : Prop := x - 2 > 0 ∨ x - 1 > 0

theorem neither_sufficient_nor_necessary_condition (x : ℝ) : ¬(p x → q x) ∧ ¬(q x → p x) :=
by
  sorry

end neither_sufficient_nor_necessary_condition_l819_81937


namespace small_circle_ratio_l819_81912

theorem small_circle_ratio (a b : ℝ) (ha : 0 < a) (hb : a < b) 
  (h : π * b^2 - π * a^2 = 5 * (π * a^2)) :
  a / b = Real.sqrt 6 / 6 :=
by
  sorry

end small_circle_ratio_l819_81912


namespace altered_solution_ratio_l819_81931

variable (b d w : ℕ)
variable (b' d' w' : ℕ)
variable (ratio_orig_bd_ratio_orig_dw_ratio_orig_bw : Rat)
variable (ratio_new_bd_ratio_new_dw_ratio_new_bw : Rat)

noncomputable def orig_ratios (ratio_orig_bd ratio_orig_bw : Rat) (d w : ℕ) : Prop := 
    ratio_orig_bd = 2 / 40 ∧ ratio_orig_bw = 40 / 100

noncomputable def new_ratios (ratio_new_bd : Rat) (d' : ℕ) : Prop :=
    ratio_new_bd = 6 / 40 ∧ d' = 60

noncomputable def new_solution (w' : ℕ) : Prop :=
    w' = 300

theorem altered_solution_ratio : 
    ∀ (orig_ratios: Prop) (new_ratios: Prop) (new_solution: Prop),
    orig_ratios ∧ new_ratios ∧ new_solution →
    (d' / w = 2 / 5) :=
by
    sorry

end altered_solution_ratio_l819_81931


namespace amount_c_is_1600_l819_81975

-- Given conditions
def total_money : ℕ := 2000
def ratio_b_c : (ℕ × ℕ) := (4, 16)

-- Define the total_parts based on the ratio
def total_parts := ratio_b_c.fst + ratio_b_c.snd

-- Define the value of each part
def value_per_part := total_money / total_parts

-- Calculate the amount for c
def amount_c_gets := ratio_b_c.snd * value_per_part

-- Main theorem stating the problem
theorem amount_c_is_1600 : amount_c_gets = 1600 := by
  -- Proof would go here
  sorry

end amount_c_is_1600_l819_81975


namespace complement_union_eq_l819_81942

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_union_eq :
  U = {1, 2, 3, 4, 5, 6, 7, 8} →
  A = {1, 3, 5, 7} →
  B = {2, 4, 5} →
  U \ (A ∪ B) = {6, 8} :=
by
  intros hU hA hB
  -- Proof goes here
  sorry

end complement_union_eq_l819_81942


namespace correct_divisor_l819_81911

theorem correct_divisor (dividend incorrect_divisor quotient correct_quotient correct_divisor : ℕ) 
  (h1 : incorrect_divisor = 63) 
  (h2 : quotient = 24) 
  (h3 : correct_quotient = 42) 
  (h4 : dividend = incorrect_divisor * quotient) 
  (h5 : dividend / correct_divisor = correct_quotient) : 
  correct_divisor = 36 := 
by 
  sorry

end correct_divisor_l819_81911


namespace range_of_a_l819_81902

noncomputable def f (a x : ℝ) := a * x - 1
noncomputable def g (x : ℝ) := -x^2 + 2 * x + 1

theorem range_of_a (a : ℝ) :
  (∀ (x1 : ℝ), x1 ∈ (Set.Icc (-1 : ℝ) 1) → ∃ (x2 : ℝ), x2 ∈ (Set.Icc (0 : ℝ) 2) ∧ f a x1 < g x2) ↔ a ∈ Set.Ioo (-3 : ℝ) 3 :=
sorry

end range_of_a_l819_81902


namespace council_revote_l819_81909

theorem council_revote (x y x' y' m : ℝ) (h1 : x + y = 500)
    (h2 : y - x = m) (h3 : x' - y' = 1.5 * m) (h4 : x' + y' = 500) (h5 : x' = 11 / 10 * y) :
    x' - x = 156.25 := by
  -- Proof goes here
  sorry

end council_revote_l819_81909


namespace quadratic_distinct_real_roots_l819_81916

theorem quadratic_distinct_real_roots (k : ℝ) :
  ((k - 1) * x^2 + 6 * x + 3 = 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k - 1) * x1^2 + 6 * x1 + 3 = 0) ∧ ((k - 1) * x2^2 + 6 * x2 + 3 = 0)) ↔ (k < 4 ∧ k ≠ 1) :=
by {
  sorry
}

end quadratic_distinct_real_roots_l819_81916


namespace Tyler_age_l819_81915

variable (T B S : ℕ)

theorem Tyler_age :
  (T = B - 3) ∧
  (S = B + 2) ∧
  (S = 2 * T) ∧
  (T + B + S = 30) →
  T = 5 := by
  sorry

end Tyler_age_l819_81915


namespace students_with_uncool_family_l819_81913

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

end students_with_uncool_family_l819_81913


namespace range_of_a_l819_81968

noncomputable def quadratic_inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  x^2 - 2 * (a - 2) * x + a > 0

theorem range_of_a :
  (∀ x : ℝ, (x < 1 ∨ x > 5) → quadratic_inequality_condition a x) ↔ (1 < a ∧ a ≤ 5) :=
by
  sorry

end range_of_a_l819_81968


namespace cost_effective_for_3000_cost_equal_at_2500_l819_81971

def cost_company_A (x : Nat) : Nat :=
  2 * x / 10 + 500

def cost_company_B (x : Nat) : Nat :=
  4 * x / 10

theorem cost_effective_for_3000 : cost_company_A 3000 < cost_company_B 3000 := 
by {
  sorry
}

theorem cost_equal_at_2500 : cost_company_A 2500 = cost_company_B 2500 := 
by {
  sorry
}

end cost_effective_for_3000_cost_equal_at_2500_l819_81971


namespace odd_function_strictly_decreasing_inequality_solutions_l819_81923

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom positive_for_neg_x (x : ℝ) : x < 0 → f x > 0

theorem odd_function : ∀ (x : ℝ), f (-x) = -f x := sorry

theorem strictly_decreasing : ∀ (x₁ x₂ : ℝ), x₁ > x₂ → f x₁ < f x₂ := sorry

theorem inequality_solutions (a x : ℝ) :
  (a = 0 ∧ false) ∨ 
  (a > 3 ∧ 3 < x ∧ x < a) ∨ 
  (a < 3 ∧ a < x ∧ x < 3) := sorry

end odd_function_strictly_decreasing_inequality_solutions_l819_81923


namespace number_of_turns_l819_81952

/-
  Given the cyclist's speed v = 5 m/s, time duration t = 5 s,
  and the circumference of the wheel c = 1.25 m, 
  prove that the number of complete turns n the wheel makes is equal to 20.
-/
theorem number_of_turns (v t c : ℝ) (h_v : v = 5) (h_t : t = 5) (h_c : c = 1.25) : 
  (v * t) / c = 20 :=
by
  sorry

end number_of_turns_l819_81952


namespace remainder_of_expression_l819_81920

theorem remainder_of_expression (a b c d : ℕ) (h1 : a = 8) (h2 : b = 20) (h3 : c = 34) (h4 : d = 3) :
  (a * b ^ c + d ^ c) % 7 = 5 := 
by 
  rw [h1, h2, h3, h4]
  sorry

end remainder_of_expression_l819_81920


namespace initial_punch_amount_l819_81993

theorem initial_punch_amount (P : ℝ) (h1 : 16 = (P / 2 + 2 + 12)) : P = 4 :=
by
  sorry

end initial_punch_amount_l819_81993


namespace primes_digit_sum_difference_l819_81957

def is_prime (a : ℕ) : Prop := Nat.Prime a

def sum_digits (n : ℕ) : ℕ := 
  Nat.digits 10 n |>.sum

theorem primes_digit_sum_difference (p q r : ℕ) (n : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (hneq : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (hpqr : p * q * r = 1899 * 10^n + 962) :
  (sum_digits p + sum_digits q + sum_digits r - sum_digits (p * q * r) = 8) := 
sorry

end primes_digit_sum_difference_l819_81957


namespace value_of_fg3_l819_81948

namespace ProofProblem

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2

theorem value_of_fg3 : f (g 3) = 83 := 
by 
  sorry -- Proof not needed

end ProofProblem

end value_of_fg3_l819_81948


namespace find_num_3_year_olds_l819_81988

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

end find_num_3_year_olds_l819_81988


namespace sum_of_b_for_one_solution_l819_81932

theorem sum_of_b_for_one_solution (b : ℝ) (has_single_solution : ∃ x, 3 * x^2 + (b + 12) * x + 11 = 0) :
  ∃ b₁ b₂ : ℝ, (3 * x^2 + (b + 12) * x + 11) = 0 ∧ b₁ + b₂ = -24 := by
  sorry

end sum_of_b_for_one_solution_l819_81932


namespace train_length_l819_81946

theorem train_length (L V : ℝ) (h1 : V = L / 15) (h2 : V = (L + 100) / 40) : L = 60 := by
  sorry

end train_length_l819_81946


namespace cube_problem_l819_81941

-- Define the conditions
def cube_volume (s : ℝ) : ℝ := s^3
def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

theorem cube_problem (x : ℝ) (s : ℝ) :
  cube_volume s = 8 * x ∧ cube_surface_area s = 4 * x → x = 216 :=
by
  intro h
  sorry

end cube_problem_l819_81941


namespace fruit_seller_sp_l819_81936

theorem fruit_seller_sp (CP SP : ℝ)
    (h1 : SP = 0.75 * CP)
    (h2 : 19.93 = 1.15 * CP) :
    SP = 13.00 :=
by
  sorry

end fruit_seller_sp_l819_81936


namespace range_of_values_l819_81930

theorem range_of_values (x : ℝ) (h1 : x - 1 ≥ 0) (h2 : x ≠ 0) : x ≥ 1 := 
sorry

end range_of_values_l819_81930


namespace probability_penny_nickel_heads_l819_81910

noncomputable def num_outcomes : ℕ := 2^4
noncomputable def num_successful_outcomes : ℕ := 2 * 2

theorem probability_penny_nickel_heads :
  (num_successful_outcomes : ℚ) / num_outcomes = 1 / 4 :=
by
  sorry

end probability_penny_nickel_heads_l819_81910


namespace HCF_of_two_numbers_l819_81944

theorem HCF_of_two_numbers (a b : ℕ) (h1 : a * b = 2562) (h2 : Nat.lcm a b = 183) : Nat.gcd a b = 14 := 
by
  sorry

end HCF_of_two_numbers_l819_81944


namespace lance_read_yesterday_l819_81958

-- Definitions based on conditions
def total_pages : ℕ := 100
def pages_tomorrow : ℕ := 35
def pages_yesterday (Y : ℕ) : ℕ := Y
def pages_today (Y : ℕ) : ℕ := Y - 5

-- The statement that we need to prove
theorem lance_read_yesterday (Y : ℕ) (h : pages_yesterday Y + pages_today Y + pages_tomorrow = total_pages) : Y = 35 :=
by sorry

end lance_read_yesterday_l819_81958


namespace Zack_kept_5_marbles_l819_81926

-- Define the initial number of marbles Zack had
def Zack_initial_marbles : ℕ := 65

-- Define the number of marbles each friend receives
def marbles_per_friend : ℕ := 20

-- Define the total number of friends
def friends : ℕ := 3

noncomputable def marbles_given_away : ℕ := friends * marbles_per_friend

-- Define the amount of marbles kept by Zack
noncomputable def marbles_kept_by_Zack : ℕ := Zack_initial_marbles - marbles_given_away

-- The theorem to prove
theorem Zack_kept_5_marbles : marbles_kept_by_Zack = 5 := by
  -- Proof skipped with sorry
  sorry

end Zack_kept_5_marbles_l819_81926


namespace expression_equals_12_l819_81962

-- Define the values of a, b, c, and k
def a : ℤ := 10
def b : ℤ := 15
def c : ℤ := 3
def k : ℤ := 2

-- Define the expression to be evaluated
def expr : ℤ := (a - (b - k * c)) - ((a - b) - k * c)

-- Prove that the expression equals 12
theorem expression_equals_12 : expr = 12 :=
by
  -- The proof will go here, leaving a placeholder for now
  sorry

end expression_equals_12_l819_81962


namespace book_stack_sum_l819_81987

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

end book_stack_sum_l819_81987


namespace new_daily_average_wage_l819_81956

theorem new_daily_average_wage (x : ℝ) : 
  (∀ y : ℝ, 25 - x = y) → 
  (∀ z : ℝ, 20 * (25 - x) = 30 * (10)) → 
  x = 10 :=
by
  intro h1 h2
  sorry

end new_daily_average_wage_l819_81956


namespace daily_rate_first_week_l819_81972

-- Definitions from given conditions
variable (x : ℝ) (h1 : ∀ y : ℝ, 0 ≤ y)
def cost_first_week := 7 * x
def additional_days_cost := 16 * 14
def total_cost := cost_first_week + additional_days_cost

-- Theorem to solve the problem
theorem daily_rate_first_week (h : total_cost = 350) : x = 18 :=
sorry

end daily_rate_first_week_l819_81972


namespace average_blinks_in_normal_conditions_l819_81951

theorem average_blinks_in_normal_conditions (blink_gaming : ℕ) (k : ℚ) (blink_normal : ℚ) 
  (h_blink_gaming : blink_gaming = 10)
  (h_k : k = (3 / 5))
  (h_condition : blink_gaming = blink_normal - k * blink_normal) : 
  blink_normal = 25 := 
by 
  sorry

end average_blinks_in_normal_conditions_l819_81951


namespace roots_rational_l819_81943

/-- Prove that the roots of the equation x^2 + px + q = 0 are always rational,
given the rational numbers p and q, and a rational n where p = n + q / n. -/
theorem roots_rational
  (n p q : ℚ)
  (hp : p = n + q / n)
  : ∃ x y : ℚ, x^2 + p * x + q = 0 ∧ y^2 + p * y + q = 0 ∧ x ≠ y :=
sorry

end roots_rational_l819_81943


namespace range_of_m_l819_81984

theorem range_of_m : 3 < 3 * Real.sqrt 2 - 1 ∧ 3 * Real.sqrt 2 - 1 < 4 :=
by
  have h1 : 3 * Real.sqrt 2 < 5 := sorry
  have h2 : 4 < 3 * Real.sqrt 2 := sorry
  exact ⟨by linarith, by linarith⟩

end range_of_m_l819_81984


namespace rational_cubes_rational_values_l819_81918

theorem rational_cubes_rational_values {a b : ℝ} (ha : 0 < a) (hb : 0 < b) 
  (hab : a + b = 1) (ha3 : ∃ r : ℚ, a^3 = r) (hb3 : ∃ s : ℚ, b^3 = s) : 
  ∃ r s : ℚ, a = r ∧ b = s :=
sorry

end rational_cubes_rational_values_l819_81918


namespace charity_event_fund_raising_l819_81973

theorem charity_event_fund_raising :
  let n := 9
  let I := 2000
  let p := 0.10
  let increased_total := I * (1 + p)
  let amount_per_person := increased_total / n
  amount_per_person = 244.44 := by
  sorry

end charity_event_fund_raising_l819_81973


namespace intersection_P_Q_equals_P_l819_81907

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := { y | ∃ x ∈ Set.univ, y = Real.cos x }

theorem intersection_P_Q_equals_P : P ∩ Q = P := by
  sorry

end intersection_P_Q_equals_P_l819_81907


namespace shoveling_driveways_l819_81981

-- Definitions of the conditions
def cost_of_candy_bars := 2 * 0.75
def cost_of_lollipops := 4 * 0.25
def total_cost := cost_of_candy_bars + cost_of_lollipops
def portion_of_earnings := total_cost * 6
def charge_per_driveway := 1.50
def number_of_driveways := portion_of_earnings / charge_per_driveway

-- The theorem to prove Jimmy shoveled 10 driveways
theorem shoveling_driveways :
  number_of_driveways = 10 := 
by
  sorry

end shoveling_driveways_l819_81981


namespace missing_digit_divisible_by_9_l819_81983

theorem missing_digit_divisible_by_9 (x : ℕ) (h : 0 ≤ x ∧ x < 10) : (3 + 5 + 1 + 9 + 2 + x) % 9 = 0 ↔ x = 7 :=
by
  sorry

end missing_digit_divisible_by_9_l819_81983


namespace find_angle_B_l819_81964

variables {A B C a b c : ℝ} (h1 : 3 * a * Real.cos C = 2 * c * Real.cos A) (h2 : Real.tan A = 1 / 3)

theorem find_angle_B (h1 : 3 * a * Real.cos C = 2 * c * Real.cos A) (h2 : Real.tan A = 1 / 3) : B = 3 * Real.pi / 4 :=
by
  sorry

end find_angle_B_l819_81964


namespace simplify_expression_l819_81978

open Real

theorem simplify_expression (a b : ℝ) 
  (h1 : b ≠ 0) (h2 : b ≠ -3 * a) (h3 : b ≠ a) (h4 : b ≠ -a) : 
  ((2 * b + a - (4 * a ^ 2 - b ^ 2) / a) / (b ^ 3 + 2 * a * b ^ 2 - 3 * a ^ 2 * b)) *
  ((a ^ 3 * b - 2 * a ^ 2 * b ^ 2 + a * b ^ 3) / (a ^ 2 - b ^ 2)) = 
  (a - b) / (a + b) :=
by
  sorry

end simplify_expression_l819_81978


namespace sum_of_integers_l819_81925

theorem sum_of_integers (a b : ℕ) (h1 : a * b + a + b = 103) 
                        (h2 : Nat.gcd a b = 1) 
                        (h3 : a < 20) 
                        (h4 : b < 20) : 
                        a + b = 19 :=
  by sorry

end sum_of_integers_l819_81925


namespace possible_values_of_a_l819_81965

def A (a : ℤ) : Set ℤ := {2, 4, a^3 - 2 * a^2 - a + 7}
def B (a : ℤ) : Set ℤ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem possible_values_of_a (a : ℤ) :
  A a ∩ B a = {2, 5} ↔ a = -1 ∨ a = 2 :=
by
  sorry

end possible_values_of_a_l819_81965


namespace C_increases_as_n_increases_l819_81969

theorem C_increases_as_n_increases (e n R r : ℝ) (he : 0 < e) (hn : 0 < n) (hR : 0 < R) (hr : 0 < r) :
  0 < (2 * e * n * R + e * n^2 * r) / (R + n * r)^2 :=
by
  sorry

end C_increases_as_n_increases_l819_81969


namespace min_value_of_X_l819_81945

theorem min_value_of_X (n : ℕ) (h : n ≥ 2) 
  (X : Finset ℕ) 
  (B : Fin n → Finset ℕ) 
  (hB : ∀ i, (B i).card = 2) :
  ∃ (Y : Finset ℕ), Y.card = n ∧ ∀ i, (Y ∩ (B i)).card ≤ 1 →
  X.card = 2 * n - 1 :=
sorry

end min_value_of_X_l819_81945


namespace prime_not_divisor_ab_cd_l819_81986

theorem prime_not_divisor_ab_cd {a b c d : ℕ} (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) (hd: 0 < d) 
  (p : ℕ) (hp : p = a + b + c + d) (hprime : Nat.Prime p) : ¬ p ∣ (a * b - c * d) := 
sorry

end prime_not_divisor_ab_cd_l819_81986


namespace min_value_of_A_sq_sub_B_sq_l819_81974

noncomputable def A (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 13)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (2 * x + 2) + Real.sqrt (2 * y + 2) + Real.sqrt (2 * z + 2)

theorem min_value_of_A_sq_sub_B_sq (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  A x y z ^ 2 - B x y z ^ 2 ≥ 36 :=
  sorry

end min_value_of_A_sq_sub_B_sq_l819_81974


namespace cakes_sold_correct_l819_81901

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

end cakes_sold_correct_l819_81901


namespace f_bounded_l819_81994

noncomputable def f : ℝ → ℝ := sorry

axiom f_property : ∀ x : ℝ, f (3 * x) = 3 * f x - 4 * (f x) ^ 3

axiom f_continuous_at_zero : ContinuousAt f 0

theorem f_bounded : ∀ x : ℝ, |f x| ≤ 1 :=
by
  sorry

end f_bounded_l819_81994


namespace derivative_at_one_l819_81950

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * Real.log x

theorem derivative_at_one : (deriv f 1) = 5 := 
by 
  sorry

end derivative_at_one_l819_81950


namespace number_of_four_digit_numbers_l819_81927

theorem number_of_four_digit_numbers (digits: Finset ℕ) (h: digits = {1, 1, 2, 0}) :
  ∃ count : ℕ, (count = 9) ∧ 
  (∀ n ∈ digits, n ≠ 0 → n * 1000 + n ≠ 0) := 
sorry

end number_of_four_digit_numbers_l819_81927


namespace minimum_value_of_reciprocals_l819_81970

theorem minimum_value_of_reciprocals {m n : ℝ} 
  (hmn : m > 0 ∧ n > 0 ∧ (m * n > 0)) 
  (hline : 2 * m + 2 * n = 1) : 
  (1 / m + 1 / n) = 8 :=
sorry

end minimum_value_of_reciprocals_l819_81970


namespace farmer_rewards_l819_81934

theorem farmer_rewards (x y : ℕ) (h1 : x + y = 60) (h2 : 1000 * x + 3000 * y = 100000) : x = 40 ∧ y = 20 :=
by {
  sorry
}

end farmer_rewards_l819_81934


namespace problem_statement_l819_81963

noncomputable def a := Real.log 2 / Real.log 14
noncomputable def b := Real.log 2 / Real.log 7
noncomputable def c := Real.log 2 / Real.log 4

theorem problem_statement : (1 / a - 1 / b + 1 / c) = 3 := by
  sorry

end problem_statement_l819_81963


namespace jonathan_fourth_task_completion_l819_81997

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

end jonathan_fourth_task_completion_l819_81997


namespace janet_extra_cost_l819_81976

theorem janet_extra_cost :
  let clarinet_hourly_rate := 40
  let clarinet_hours_per_week := 3
  let clarinet_weeks_per_year := 50
  let clarinet_yearly_cost := clarinet_hourly_rate * clarinet_hours_per_week * clarinet_weeks_per_year

  let piano_hourly_rate := 28
  let piano_hours_per_week := 5
  let piano_weeks_per_year := 50
  let piano_yearly_cost := piano_hourly_rate * piano_hours_per_week * piano_weeks_per_year
  let piano_discount_rate := 0.10
  let piano_discounted_yearly_cost := piano_yearly_cost * (1 - piano_discount_rate)

  let violin_hourly_rate := 35
  let violin_hours_per_week := 2
  let violin_weeks_per_year := 50
  let violin_yearly_cost := violin_hourly_rate * violin_hours_per_week * violin_weeks_per_year
  let violin_discount_rate := 0.15
  let violin_discounted_yearly_cost := violin_yearly_cost * (1 - violin_discount_rate)

  let singing_hourly_rate := 45
  let singing_hours_per_week := 1
  let singing_weeks_per_year := 50
  let singing_yearly_cost := singing_hourly_rate * singing_hours_per_week * singing_weeks_per_year

  let combined_cost := piano_discounted_yearly_cost + violin_discounted_yearly_cost + singing_yearly_cost
  combined_cost - clarinet_yearly_cost = 5525 := 
  sorry

end janet_extra_cost_l819_81976


namespace smallest_integer_n_l819_81924

theorem smallest_integer_n (m n : ℕ) (r : ℝ) :
  (m = (n + r)^3) ∧ (0 < r) ∧ (r < 1 / 2000) ∧ (m = n^3 + 3 * n^2 * r + 3 * n * r^2 + r^3) →
  n = 26 :=
by 
  sorry

end smallest_integer_n_l819_81924


namespace value_of_m_l819_81967

theorem value_of_m (a b m : ℝ)
    (h1: 2 ^ a = m)
    (h2: 5 ^ b = m)
    (h3: 1 / a + 1 / b = 1 / 2) :
    m = 100 :=
sorry

end value_of_m_l819_81967


namespace correct_assignment_statement_l819_81982

theorem correct_assignment_statement (a b : ℕ) : 
  (2 = a → False) ∧ 
  (a = a + 1 → True) ∧ 
  (a * b = 2 → False) ∧ 
  (a + 1 = a → False) :=
by {
  sorry
}

end correct_assignment_statement_l819_81982


namespace intersection_eq_l819_81914

-- Define sets P and Q
def setP := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2}
def setQ := {y : ℝ | ∃ x : ℝ, y = -x + 2}

-- The main theorem statement
theorem intersection_eq: setP ∩ setQ = {y : ℝ | y ≤ 2} :=
by
  sorry

end intersection_eq_l819_81914


namespace sum_of_decimals_as_fraction_l819_81979

axiom decimal_to_fraction :
  0.2 = 2 / 10 ∧
  0.04 = 4 / 100 ∧
  0.006 = 6 / 1000 ∧
  0.0008 = 8 / 10000 ∧
  0.00010 = 10 / 100000 ∧
  0.000012 = 12 / 1000000

theorem sum_of_decimals_as_fraction:
  0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 + 0.000012 = (3858:ℚ) / 15625 :=
by
  have h := decimal_to_fraction
  sorry

end sum_of_decimals_as_fraction_l819_81979


namespace regular_price_coffee_l819_81922

theorem regular_price_coffee (y : ℝ) (h1 : 0.4 * y / 4 = 4) : y = 40 :=
by
  sorry

end regular_price_coffee_l819_81922


namespace correct_equation_l819_81929

theorem correct_equation : -(-5) = |-5| :=
by
  -- sorry is used here to skip the actual proof steps which are not required.
  sorry

end correct_equation_l819_81929


namespace probability_of_fourth_three_is_correct_l819_81998

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

end probability_of_fourth_three_is_correct_l819_81998


namespace total_arrangements_l819_81919

-- Defining the selection and arrangement problem conditions
def select_and_arrange (n m : ℕ) : ℕ :=
  Nat.choose n m * Nat.factorial m

-- Specifying the specific problem's constraints and results
theorem total_arrangements : select_and_arrange 8 2 * select_and_arrange 6 2 = 60 := by
  -- Proof omitted
  sorry

end total_arrangements_l819_81919


namespace star_area_l819_81990

-- Conditions
def square_ABCD_area (s : ℝ) := s^2 = 72

-- Question and correct answer
theorem star_area (s : ℝ) (h : square_ABCD_area s) : 24 = 24 :=
by sorry

end star_area_l819_81990


namespace spherical_distance_between_points_l819_81933

noncomputable def spherical_distance (R : ℝ) (α : ℝ) : ℝ :=
  α * R

theorem spherical_distance_between_points 
  (R : ℝ) 
  (α : ℝ) 
  (hR : R > 0) 
  (hα : α = π / 6) : 
  spherical_distance R α = (π / 6) * R :=
by
  rw [hα]
  unfold spherical_distance
  ring

end spherical_distance_between_points_l819_81933


namespace max_value_of_expression_l819_81947

-- Define the real numbers p, q, r and the conditions
variables {p q r : ℝ}

-- Define the main goal
theorem max_value_of_expression 
(h : 9 * p^2 + 4 * q^2 + 25 * r^2 = 4) : 
  (5 * p + 3 * q + 10 * r) ≤ (10 * Real.sqrt 13 / 3) :=
sorry

end max_value_of_expression_l819_81947


namespace total_amount_spent_l819_81991

def cost_of_soft_drink : ℕ := 2
def cost_per_candy_bar : ℕ := 5
def number_of_candy_bars : ℕ := 5

theorem total_amount_spent : cost_of_soft_drink + cost_per_candy_bar * number_of_candy_bars = 27 := by
  sorry

end total_amount_spent_l819_81991


namespace fraction_to_decimal_equiv_l819_81935

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l819_81935


namespace prob1_part1_prob1_part2_find_integers_x_prob3_min_value_prob4_l819_81921

-- Problem 1
theorem prob1_part1 : |-4 + 6| = 2 := sorry
theorem prob1_part2 : |-2 - 4| = 6 := sorry

-- Problem 2
theorem find_integers_x :
  {x : ℤ | |x + 2| + |x - 1| = 3} = {-2, -1, 0, 1} :=
sorry

-- Problem 3
theorem prob3 (a : ℤ) (h : -4 ≤ a ∧ a ≤ 6) : |a + 4| + |a - 6| = 10 :=
sorry

-- Problem 4
theorem min_value_prob4 :
  ∃ (a : ℤ), |a - 1| + |a + 5| + |a - 4| = 9 ∧ ∀ (b : ℤ), |b - 1| + |b + 5| + |b - 4| ≥ 9 :=
sorry

end prob1_part1_prob1_part2_find_integers_x_prob3_min_value_prob4_l819_81921


namespace determine_c_plus_d_l819_81992

theorem determine_c_plus_d (x : ℝ) (c d : ℤ) (h1 : x^2 + 5*x + (5/x) + (1/(x^2)) = 35) (h2 : x = c + Real.sqrt d) : c + d = 5 :=
sorry

end determine_c_plus_d_l819_81992


namespace binary_predecessor_l819_81985

def M : ℕ := 84
def N : ℕ := 83
def M_bin : ℕ := 0b1010100
def N_bin : ℕ := 0b1010011

theorem binary_predecessor (H : M = M_bin ∧ N = M - 1) : N = N_bin := by
  sorry

end binary_predecessor_l819_81985


namespace day_of_week_after_n_days_l819_81928

theorem day_of_week_after_n_days (birthday : ℕ) (n : ℕ) (day_of_week : ℕ) :
  birthday = 4 → (n % 7) = 2 → day_of_week = 6 :=
by sorry

end day_of_week_after_n_days_l819_81928


namespace crayons_left_l819_81949

-- Define the initial number of crayons and the number taken
def initial_crayons : ℕ := 7
def crayons_taken : ℕ := 3

-- Prove the number of crayons left in the drawer
theorem crayons_left : initial_crayons - crayons_taken = 4 :=
by
  sorry

end crayons_left_l819_81949


namespace num_children_proof_l819_81903

noncomputable def number_of_children (total_persons : ℕ) (total_revenue : ℕ) (adult_price : ℕ) (child_price : ℕ) : ℕ :=
  let adult_tickets := (child_price * total_persons - total_revenue) / (child_price - adult_price)
  let child_tickets := total_persons - adult_tickets
  child_tickets

theorem num_children_proof : number_of_children 280 14000 60 25 = 80 := 
by
  unfold number_of_children
  sorry

end num_children_proof_l819_81903


namespace value_range_of_f_l819_81900

-- Define the function f(x) = 2x - x^2
def f (x : ℝ) : ℝ := 2 * x - x^2

-- State the theorem with the given conditions and prove the correct answer
theorem value_range_of_f :
  (∀ y : ℝ, -3 ≤ y ∧ y ≤ 1 → ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = y) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → -3 ≤ f x ∧ f x ≤ 1) :=
by
  sorry

end value_range_of_f_l819_81900


namespace minimal_blue_chips_value_l819_81953

noncomputable def minimal_blue_chips (r g b : ℕ) : Prop :=
b ≥ r / 3 ∧
b ≤ g / 4 ∧
r + g ≥ 75

theorem minimal_blue_chips_value : ∃ (b : ℕ), minimal_blue_chips 33 44 b ∧ b = 11 :=
by
  have b := 11
  use b
  sorry

end minimal_blue_chips_value_l819_81953


namespace slant_height_of_cone_l819_81938

theorem slant_height_of_cone (r : ℝ) (h : ℝ) (s : ℝ) (unfolds_to_semicircle : s = π) (base_radius : r = 1) : s = 2 :=
by
  sorry

end slant_height_of_cone_l819_81938


namespace minimum_people_correct_answer_l819_81977

theorem minimum_people_correct_answer (people questions : ℕ) (common_correct : ℕ) (h_people : people = 21) (h_questions : questions = 15) (h_common_correct : ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ people → 1 ≤ b ∧ b ≤ people → a ≠ b → common_correct ≥ 1) :
  ∃ (min_correct : ℕ), min_correct = 7 := 
sorry

end minimum_people_correct_answer_l819_81977


namespace highway_extension_l819_81980

def initial_length : ℕ := 200
def final_length : ℕ := 650
def first_day_construction : ℕ := 50
def second_day_construction : ℕ := 3 * first_day_construction
def total_construction : ℕ := first_day_construction + second_day_construction
def total_extension_needed : ℕ := final_length - initial_length
def miles_still_needed : ℕ := total_extension_needed - total_construction

theorem highway_extension : miles_still_needed = 250 := by
  sorry

end highway_extension_l819_81980


namespace odd_expressions_l819_81939

theorem odd_expressions (m n p : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) (hp : p % 2 = 0) : 
  ((2 * m * n + 5) ^ 2 % 2 = 1) ∧ (5 * m * n + p % 2 = 1) := 
by
  sorry

end odd_expressions_l819_81939


namespace original_amount_of_milk_is_720_l819_81954

variable (M : ℝ) -- The original amount of milk in milliliters

theorem original_amount_of_milk_is_720 :
  ((5 / 6) * M) - ((2 / 5) * ((5 / 6) * M)) - ((2 / 3) * (((5 / 6) * M) - ((2 / 5) * ((5 / 6) * M)))) = 120 → 
  M = 720 := by
  sorry

end original_amount_of_milk_is_720_l819_81954


namespace volume_larger_of_cube_cut_plane_l819_81908

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

end volume_larger_of_cube_cut_plane_l819_81908
