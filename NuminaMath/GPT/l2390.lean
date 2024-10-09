import Mathlib

namespace find_f_6_l2390_239029

def f : ℕ → ℕ := sorry

lemma f_equality (x : ℕ) : f (x + 1) = x := sorry

theorem find_f_6 : f 6 = 5 :=
by
  -- the proof would go here
  sorry

end find_f_6_l2390_239029


namespace find_x_if_arithmetic_mean_is_12_l2390_239035

theorem find_x_if_arithmetic_mean_is_12 (x : ℝ) (h : (8 + 16 + 21 + 7 + x) / 5 = 12) : x = 8 :=
by
  sorry

end find_x_if_arithmetic_mean_is_12_l2390_239035


namespace alice_minimum_speed_l2390_239099

noncomputable def minimum_speed_to_exceed (d t_bob t_alice : ℝ) (v_bob : ℝ) : ℝ :=
  d / t_alice

theorem alice_minimum_speed (d : ℝ) (v_bob : ℝ) (t_lag : ℝ) (v_alice : ℝ) :
  d = 30 → v_bob = 40 → t_lag = 0.5 → v_alice = d / (d / v_bob - t_lag) → v_alice > 60 :=
by
  intros hd hv hb ht
  rw [hd, hv, hb] at ht
  simp at ht
  sorry

end alice_minimum_speed_l2390_239099


namespace problem_xy_l2390_239093

theorem problem_xy (x y : ℝ) (h1 : x + y = 25) (h2 : x^2 * y^3 + y^2 * x^3 = 25) : x * y = 1 :=
by
  sorry

end problem_xy_l2390_239093


namespace sqrt_of_16_is_4_l2390_239022

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_16_is_4_l2390_239022


namespace problem_l2390_239033

theorem problem (x a : ℝ) (h : x^5 - x^3 + x = a) : x^6 ≥ 2 * a - 1 := 
by 
  sorry

end problem_l2390_239033


namespace circle_center_radius_l2390_239056

theorem circle_center_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4 * x = 0 ↔ ((x - 2)^2 + y^2 = 4) ∧ (∃ (c_x c_y r : ℝ), c_x = 2 ∧ c_y = 0 ∧ r = 2) :=
by
  sorry

end circle_center_radius_l2390_239056


namespace least_positive_integer_exists_l2390_239094

theorem least_positive_integer_exists :
  ∃ (x : ℕ), 
    (x % 6 = 5) ∧
    (x % 8 = 7) ∧
    (x % 7 = 6) ∧
    x = 167 :=
by {
  sorry
}

end least_positive_integer_exists_l2390_239094


namespace polynomial_identity_l2390_239086

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem polynomial_identity (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h : ∀ g : ℝ → ℝ, ∀ x : ℝ, f (g x) = g (f x)) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end polynomial_identity_l2390_239086


namespace juniors_score_l2390_239049

theorem juniors_score (juniors seniors total_students avg_score avg_seniors_score total_score : ℝ)
  (hj: juniors = 0.2 * total_students)
  (hs: seniors = 0.8 * total_students)
  (ht: total_students = 20)
  (ha: avg_score = 78)
  (hp: (seniors * avg_seniors_score + juniors * c) / total_students = avg_score)
  (havg_seniors: avg_seniors_score = 76)
  (hts: total_score = total_students * avg_score)
  (total_seniors_score : ℝ)
  (hts_seniors: total_seniors_score = seniors * avg_seniors_score)
  (total_juniors_score : ℝ)
  (hts_juniors: total_juniors_score = total_score - total_seniors_score)
  (hjs: c = total_juniors_score / juniors) :
  c = 86 :=
sorry

end juniors_score_l2390_239049


namespace gwen_more_money_from_mom_l2390_239075

def dollars_received_from_mom : ℕ := 7
def dollars_received_from_dad : ℕ := 5

theorem gwen_more_money_from_mom :
  dollars_received_from_mom - dollars_received_from_dad = 2 :=
by
  sorry

end gwen_more_money_from_mom_l2390_239075


namespace initial_number_of_men_l2390_239067

theorem initial_number_of_men (x : ℕ) :
    (50 * x = 25 * (x + 20)) → x = 20 := 
by
  sorry

end initial_number_of_men_l2390_239067


namespace nonnegative_integer_count_l2390_239092

def balanced_quaternary_nonnegative_count : Nat :=
  let base := 4
  let max_index := 6
  let valid_digits := [-1, 0, 1]
  let max_sum := (base ^ (max_index + 1) - 1) / (base - 1)
  max_sum + 1

theorem nonnegative_integer_count : balanced_quaternary_nonnegative_count = 5462 := by
  sorry

end nonnegative_integer_count_l2390_239092


namespace geometric_sequence_first_term_l2390_239028

open Real Nat

theorem geometric_sequence_first_term (a r : ℝ)
  (h1 : a * r^4 = (7! : ℝ))
  (h2 : a * r^7 = (8! : ℝ)) : a = 315 := by
  sorry

end geometric_sequence_first_term_l2390_239028


namespace same_number_written_every_vertex_l2390_239018

theorem same_number_written_every_vertex (a : ℕ → ℝ) (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → a i > 0) 
(h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → (a i) ^ 2 = a (i - 1) + a (i + 1) ) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → a i = 2 :=
by
  sorry

end same_number_written_every_vertex_l2390_239018


namespace mom_foster_dog_food_l2390_239059

theorem mom_foster_dog_food
    (puppy_food_per_meal : ℚ := 1 / 2)
    (puppy_meals_per_day : ℕ := 2)
    (num_puppies : ℕ := 5)
    (total_food_needed : ℚ := 57)
    (days : ℕ := 6)
    (mom_meals_per_day : ℕ := 3) :
    (total_food_needed - (num_puppies * puppy_food_per_meal * ↑puppy_meals_per_day * ↑days)) / (↑days * ↑mom_meals_per_day) = 1.5 :=
by
  -- Definitions translation
  let puppy_total_food := num_puppies * puppy_food_per_meal * ↑puppy_meals_per_day * ↑days
  let mom_total_food := total_food_needed - puppy_total_food
  let mom_meals := ↑days * ↑mom_meals_per_day
  -- Proof starts with sorry to indicate that the proof part is not included
  sorry

end mom_foster_dog_food_l2390_239059


namespace ages_proof_l2390_239078

noncomputable def A : ℝ := 12.1
noncomputable def B : ℝ := 6.1
noncomputable def C : ℝ := 11.3

-- Conditions extracted from the problem
def sum_of_ages (A B C : ℝ) : Prop := A + B + C = 29.5
def specific_age (C : ℝ) : Prop := C = 11.3
def twice_as_old (A B : ℝ) : Prop := A = 2 * B

theorem ages_proof : 
  ∃ (A B C : ℝ), 
    specific_age C ∧ twice_as_old A B ∧ sum_of_ages A B C :=
by
  exists 12.1, 6.1, 11.3
  sorry

end ages_proof_l2390_239078


namespace blue_paint_cans_needed_l2390_239054

-- Definitions of the conditions
def blue_to_green_ratio : ℕ × ℕ := (4, 3)
def total_cans : ℕ := 42
def expected_blue_cans : ℕ := 24

-- Proof statement
theorem blue_paint_cans_needed (r : ℕ × ℕ) (total : ℕ) (expected : ℕ) 
  (h1: r = (4, 3)) (h2: total = 42) : expected = 24 :=
by
  sorry

end blue_paint_cans_needed_l2390_239054


namespace number_of_male_employees_l2390_239041

theorem number_of_male_employees (num_female : ℕ) (x y : ℕ) 
  (h1 : 7 * x = y) 
  (h2 : 8 * x = num_female) 
  (h3 : 9 * (7 * x + 3) = 8 * num_female) :
  y = 189 := by
  sorry

end number_of_male_employees_l2390_239041


namespace squared_sum_of_a_b_l2390_239072

theorem squared_sum_of_a_b (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 3) : (a + b) ^ 2 = 16 :=
by
  sorry

end squared_sum_of_a_b_l2390_239072


namespace find_k_solve_quadratic_l2390_239083

-- Define the conditions
variables (x1 x2 k : ℝ)

-- Given conditions
def quadratic_roots : Prop :=
  x1 + x2 = 6 ∧ x1 * x2 = k

def condition_A (x1 x2 : ℝ) : Prop :=
  x1^2 * x2^2 - x1 - x2 = 115

-- Prove that k = -11 given the conditions
theorem find_k (h1: quadratic_roots x1 x2 k) (h2 : condition_A x1 x2) : k = -11 :=
  sorry

-- Prove the roots of the quadratic equation when k = -11
theorem solve_quadratic (h1 : quadratic_roots x1 x2 (-11)) : 
  x1 = 3 + 2 * Real.sqrt 5 ∧ x2 = 3 - 2 * Real.sqrt 5 ∨ 
  x1 = 3 - 2 * Real.sqrt 5 ∧ x2 = 3 + 2 * Real.sqrt 5 :=
  sorry

end find_k_solve_quadratic_l2390_239083


namespace new_average_contribution_75_l2390_239063

-- Define the conditions given in the problem
def original_contributions : ℝ := 1
def johns_donation : ℝ := 100
def increase_rate : ℝ := 1.5

-- Define a function to calculate the new average contribution size
def new_total_contributions (A : ℝ) := A + johns_donation
def new_average_contribution (A : ℝ) := increase_rate * A

-- Theorem to prove that the new average contribution size is $75
theorem new_average_contribution_75 (A : ℝ) :
  new_total_contributions A / (original_contributions + 1) = increase_rate * A →
  A = 50 →
  new_average_contribution A = 75 :=
by
  intros h1 h2
  rw [new_average_contribution, h2]
  sorry

end new_average_contribution_75_l2390_239063


namespace floor_length_l2390_239031

theorem floor_length (width length : ℕ) 
  (cost_per_square total_cost : ℕ)
  (square_side : ℕ)
  (h1 : width = 64) 
  (h2 : square_side = 8)
  (h3 : cost_per_square = 24)
  (h4 : total_cost = 576) 
  : length = 24 :=
by
  -- Placeholder for the proof, using sorry
  sorry

end floor_length_l2390_239031


namespace number_of_solutions_l2390_239085

-- Defining the conditions for the equation
def isCondition (x : ℝ) : Prop := x ≠ 2 ∧ x ≠ 3

-- Defining the equation
def eqn (x : ℝ) : Prop := (3 * x^2 - 15 * x + 18) / (x^2 - 5 * x + 6) = x - 2

-- Defining the property that we need to prove
def property (x : ℝ) : Prop := eqn x ∧ isCondition x

-- Statement of the proof problem
theorem number_of_solutions : 
  ∃! x : ℝ, property x :=
sorry

end number_of_solutions_l2390_239085


namespace polynomial_inequality_holds_l2390_239098

def polynomial (x : ℝ) : ℝ := x^6 + 4 * x^5 + 2 * x^4 - 6 * x^3 - 2 * x^2 + 4 * x - 1

theorem polynomial_inequality_holds (x : ℝ) :
  (x ≤ -1 - Real.sqrt 2 ∨ x = (-1 - Real.sqrt 5) / 2 ∨ x ≥ -1 + Real.sqrt 2) →
  polynomial x ≥ 0 :=
by
  sorry

end polynomial_inequality_holds_l2390_239098


namespace always_positive_inequality_l2390_239042

theorem always_positive_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
sorry

end always_positive_inequality_l2390_239042


namespace fraction_of_tips_l2390_239000

variable (S T : ℝ) -- assuming S is salary and T is tips
variable (h : T / (S + T) = 0.7142857142857143)

/-- 
If the fraction of the waiter's income from tips is 0.7142857142857143,
then the fraction of his salary that were his tips is 2.5.
-/
theorem fraction_of_tips (h : T / (S + T) = 0.7142857142857143) : T / S = 2.5 :=
sorry

end fraction_of_tips_l2390_239000


namespace rolling_cube_dot_path_l2390_239005

theorem rolling_cube_dot_path (a b c : ℝ) (h_edge : a = 1) (h_dot_top : True):
  c = (1 + Real.sqrt 5) / 2 := by
  sorry

end rolling_cube_dot_path_l2390_239005


namespace area_ratio_l2390_239021

variables {rA rB : ℝ} (C_A C_B : ℝ)

#check C_A = 2 * Real.pi * rA
#check C_B = 2 * Real.pi * rB

theorem area_ratio (h : (60 / 360) * C_A = (40 / 360) * C_B) : (Real.pi * rA^2) / (Real.pi * rB^2) = 4 / 9 := by
  sorry

end area_ratio_l2390_239021


namespace quadratic_three_distinct_solutions_l2390_239058

open Classical

variable (a b c : ℝ) (x1 x2 x3 : ℝ)

-- Conditions:
variables (hx1 : a * x1^2 + b * x1 + c = 0)
          (hx2 : a * x2^2 + b * x2 + c = 0)
          (hx3 : a * x3^2 + b * x3 + c = 0)
          (h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)

-- Proof problem
theorem quadratic_three_distinct_solutions : a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end quadratic_three_distinct_solutions_l2390_239058


namespace solve_inequality_l2390_239089

open Set

theorem solve_inequality (a x : ℝ) : 
  (x - 2) * (a * x - 2) > 0 → 
  (a = 0 ∧ x < 2) ∨ 
  (a < 0 ∧ (2/a) < x ∧ x < 2) ∨ 
  (0 < a ∧ a < 1 ∧ ((x < 2 ∨ x > 2/a))) ∨ 
  (a = 1 ∧ x ≠ 2) ∨ 
  (a > 1 ∧ ((x < 2/a ∨ x > 2)))
  := sorry

end solve_inequality_l2390_239089


namespace final_tree_count_l2390_239061

noncomputable def current_trees : ℕ := 39
noncomputable def trees_planted_today : ℕ := 41
noncomputable def trees_planted_tomorrow : ℕ := 20

theorem final_tree_count : current_trees + trees_planted_today + trees_planted_tomorrow = 100 := by
  sorry

end final_tree_count_l2390_239061


namespace semicircle_problem_l2390_239014

theorem semicircle_problem (N : ℕ) (r : ℝ) (π : ℝ) (hπ : 0 < π) 
  (h1 : ∀ (r : ℝ), ∃ (A B : ℝ), A = N * (π * r^2 / 2) ∧ B = (π * (N^2 * r^2 / 2) - N * (π * r^2 / 2)) ∧ A / B = 1 / 3) :
  N = 4 :=
by
  sorry

end semicircle_problem_l2390_239014


namespace gcd_12012_18018_l2390_239091

theorem gcd_12012_18018 : Int.gcd 12012 18018 = 6006 := 
by
  sorry

end gcd_12012_18018_l2390_239091


namespace girls_on_playground_l2390_239070

variable (total_children : ℕ) (boys : ℕ) (girls : ℕ)

theorem girls_on_playground (h1 : total_children = 117) (h2 : boys = 40) (h3 : girls = total_children - boys) : girls = 77 :=
by
  sorry

end girls_on_playground_l2390_239070


namespace num_distinct_solutions_l2390_239076

theorem num_distinct_solutions : 
  (∃ x : ℝ, |x - 3| = |x + 5|) ∧ 
  (∀ x1 x2 : ℝ, |x1 - 3| = |x1 + 5| → |x2 - 3| = |x2 + 5| → x1 = x2) := 
  sorry

end num_distinct_solutions_l2390_239076


namespace math_marks_l2390_239052

theorem math_marks (english physics chemistry biology total_marks math_marks : ℕ) 
  (h_eng : english = 73)
  (h_phy : physics = 92)
  (h_chem : chemistry = 64)
  (h_bio : biology = 82)
  (h_avg : total_marks = 76 * 5) :
  math_marks = 69 := 
by
  sorry

end math_marks_l2390_239052


namespace tim_score_l2390_239040

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def first_seven_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

theorem tim_score :
  (first_seven_primes.sum = 58) :=
by
  sorry

end tim_score_l2390_239040


namespace problem_I_problem_II_l2390_239043

-- Define the function f as given
def f (x m : ℝ) : ℝ := x^2 + (m-1)*x - m

-- Problem (I)
theorem problem_I (x : ℝ) : -2 < x ∧ x < 1 ↔ f x 2 < 0 := sorry

-- Problem (II)
theorem problem_II (m : ℝ) : ∀ x, f x m + 1 ≥ 0 ↔ -3 ≤ m ∧ m ≤ 1 := sorry

end problem_I_problem_II_l2390_239043


namespace max_tan_B_l2390_239079

theorem max_tan_B (A B : ℝ) (h : Real.sin (2 * A + B) = 2 * Real.sin B) : 
  Real.tan B ≤ Real.sqrt 3 / 3 := sorry

end max_tan_B_l2390_239079


namespace solution_set_of_abs_fraction_eq_fraction_l2390_239050

-- Problem Statement
theorem solution_set_of_abs_fraction_eq_fraction :
  { x : ℝ | |x / (x - 1)| = x / (x - 1) } = { x : ℝ | x ≤ 0 ∨ x > 1 } :=
by
  sorry

end solution_set_of_abs_fraction_eq_fraction_l2390_239050


namespace sum_a_c_e_l2390_239084

theorem sum_a_c_e {a b c d e f : ℝ} 
  (h1 : a / b = 2) 
  (h2 : c / d = 2) 
  (h3 : e / f = 2) 
  (h4 : b + d + f = 5) : 
  a + c + e = 10 :=
by
  -- Proof goes here
  sorry

end sum_a_c_e_l2390_239084


namespace correctness_of_statements_l2390_239007

theorem correctness_of_statements 
  (A B C D : Prop)
  (h1 : A → B) (h2 : ¬(B → A))
  (h3 : C → B) (h4 : B → C)
  (h5 : D → C) (h6 : ¬(C → D)) : 
  (A → (C ∧ ¬(C → A))) ∧ (¬(A → D) ∧ ¬(D → A)) := 
by
  -- Proof will go here.
  sorry

end correctness_of_statements_l2390_239007


namespace system_of_equations_solution_l2390_239002

theorem system_of_equations_solution (x y : ℤ) (h1 : 2 * x + 5 * y = 26) (h2 : 4 * x - 2 * y = 4) : 
    x = 3 ∧ y = 4 :=
by
  sorry

end system_of_equations_solution_l2390_239002


namespace cylinder_volume_from_cone_l2390_239090

/-- Given the volume of a cone, prove the volume of a cylinder with the same base and height. -/
theorem cylinder_volume_from_cone (V_cone : ℝ) (h : V_cone = 3.6) : 
  ∃ V_cylinder : ℝ, V_cylinder = 0.0108 :=
by
  have V_cylinder := 3 * V_cone
  have V_cylinder_meters := V_cylinder / 1000
  use V_cylinder_meters
  sorry

end cylinder_volume_from_cone_l2390_239090


namespace g_five_eq_one_l2390_239097

noncomputable def g : ℝ → ℝ := sorry

axiom g_mul (x z : ℝ) : g (x * z) = g x * g z
axiom g_one_ne_zero : g (1) ≠ 0

theorem g_five_eq_one : g (5) = 1 := 
by
  sorry

end g_five_eq_one_l2390_239097


namespace all_lines_can_be_paired_perpendicular_l2390_239047

noncomputable def can_pair_perpendicular_lines : Prop := 
  ∀ (L1 L2 : ℝ), 
    L1 ≠ L2 → 
      ∃ (m : ℝ), 
        (m * L1 = -1/L2 ∨ L1 = 0 ∧ L2 ≠ 0 ∨ L2 = 0 ∧ L1 ≠ 0)

theorem all_lines_can_be_paired_perpendicular : can_pair_perpendicular_lines :=
sorry

end all_lines_can_be_paired_perpendicular_l2390_239047


namespace calculator_sum_is_large_l2390_239051

-- Definitions for initial conditions and operations
def participants := 50
def initial_calc1 := 2
def initial_calc2 := -2
def initial_calc3 := 0

-- Define the operations
def operation_calc1 (n : ℕ) := initial_calc1 * 2^n
def operation_calc2 (n : ℕ) := (-2) ^ (2^n)
def operation_calc3 (n : ℕ) := initial_calc3 - n

-- Define the final values for each calculator
def final_calc1 := operation_calc1 participants
def final_calc2 := operation_calc2 participants
def final_calc3 := operation_calc3 participants

-- The final sum
def final_sum := final_calc1 + final_calc2 + final_calc3

-- Prove the final result
theorem calculator_sum_is_large :
  final_sum = 2 ^ (2 ^ 50) :=
by
  -- The proof would go here.
  sorry

end calculator_sum_is_large_l2390_239051


namespace find_B_l2390_239020

variable (A B : ℝ)

def condition1 : Prop := A + B = 1210
def condition2 : Prop := (4 / 15) * A = (2 / 5) * B

theorem find_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 484 :=
sorry

end find_B_l2390_239020


namespace next_two_series_numbers_l2390_239096

theorem next_two_series_numbers :
  ∀ (a : ℕ → ℤ), a 1 = 2 → a 2 = 3 →
    (∀ n, 3 ≤ n → a n = a (n - 1) + a (n - 2) - 5) →
    a 7 = -26 ∧ a 8 = -45 :=
by
  intros a h1 h2 h3
  sorry

end next_two_series_numbers_l2390_239096


namespace part1_part2_part3_l2390_239074

-- Part 1: Prove that B = 90° given a=20, b=29, c=21

theorem part1 (a b c : ℝ) (h1 : a = 20) (h2 : b = 29) (h3 : c = 21) : 
  ∃ B : ℝ, B = 90 := 
sorry

-- Part 2: Prove that b = 7 given a=3√3, c=2, B=150°

theorem part2 (a c B b : ℝ) (h1 : a = 3 * Real.sqrt 3) (h2 : c = 2) (h3 : B = 150) : 
  ∃ b : ℝ, b = 7 :=
sorry

-- Part 3: Prove that A = 45° given a=2, b=√2, c=√3 + 1

theorem part3 (a b c A : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 2) (h3 : c = Real.sqrt 3 + 1) : 
  ∃ A : ℝ, A = 45 :=
sorry

end part1_part2_part3_l2390_239074


namespace total_spent_by_pete_and_raymond_l2390_239046

def initial_money_in_cents : ℕ := 250
def pete_spent_in_nickels : ℕ := 4
def nickel_value_in_cents : ℕ := 5
def raymond_dimes_left : ℕ := 7
def dime_value_in_cents : ℕ := 10

theorem total_spent_by_pete_and_raymond : 
  (pete_spent_in_nickels * nickel_value_in_cents) 
  + (initial_money_in_cents - (raymond_dimes_left * dime_value_in_cents)) = 200 := sorry

end total_spent_by_pete_and_raymond_l2390_239046


namespace constants_unique_l2390_239011

theorem constants_unique (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 → (5 * x) / ((x - 4) * (x - 2) ^ 2) = A / (x - 4) + B / (x - 2) + C / (x - 2) ^ 2) ↔
  A = 5 ∧ B = -5 ∧ C = -5 :=
by
  sorry

end constants_unique_l2390_239011


namespace slope_range_of_line_l2390_239027

/-- A mathematical proof problem to verify the range of the slope of a line
that passes through a given point (-1, -1) and intersects a circle. -/
theorem slope_range_of_line (
  k : ℝ
) : (∃ x y : ℝ, (y + 1 = k * (x + 1)) ∧ (x - 2) ^ 2 + y ^ 2 = 1) ↔ (0 < k ∧ k < 3 / 4) := 
by
  sorry  

end slope_range_of_line_l2390_239027


namespace expressible_numbers_count_l2390_239009

theorem expressible_numbers_count : ∃ k : ℕ, k = 2222 ∧ ∀ n : ℕ, n ≤ 2000 → ∃ x : ℝ, n = Int.floor x + Int.floor (3 * x) + Int.floor (5 * x) :=
by sorry

end expressible_numbers_count_l2390_239009


namespace solve_equation_l2390_239008

theorem solve_equation (x : ℝ) : 
  (x - 4)^6 + (x - 6)^6 = 64 → x = 4 ∨ x = 6 :=
by
  sorry

end solve_equation_l2390_239008


namespace triangle_angle_construction_l2390_239034

-- Step d): Lean 4 Statement
theorem triangle_angle_construction (a b c : ℝ) (α β : ℝ) (γ : ℝ) (h1 : γ = 120)
  (h2 : a < c) (h3 : c < a + b) (h4 : b < c)  (h5 : c < a + b) :
    (∃ α' β' γ', α' = 60 ∧ β' = α ∧ γ' = 60 + β) ∧ 
    (∃ α'' β'' γ'', α'' = 60 ∧ β'' = β ∧ γ'' = 60 + α) :=
  sorry

end triangle_angle_construction_l2390_239034


namespace range_and_intervals_of_f_l2390_239073

noncomputable def f (x : ℝ) : ℝ := (1/3)^(x^2 - 2 * x - 3)

theorem range_and_intervals_of_f :
  (∀ y, y > 0 → y ≤ 81 → (∃ x : ℝ, f x = y)) ∧
  (∀ x y, x ≤ y → f x ≥ f y) ∧
  (∀ x y, x ≥ y → f x ≤ f y) :=
by
  sorry

end range_and_intervals_of_f_l2390_239073


namespace digit_inequality_l2390_239045

theorem digit_inequality : ∃ (n : ℕ), n = 9 ∧ ∀ (d : ℕ), d < 10 → (2 + d / 10 + 5 / 1000 > 2 + 5 / 1000) → d > 0 :=
by
  sorry

end digit_inequality_l2390_239045


namespace area_of_region_a_area_of_region_b_area_of_region_c_l2390_239012

-- Definition of regions and their areas
def area_of_square : Real := sorry
def area_of_diamond : Real := sorry
def area_of_hexagon : Real := sorry

-- Define the conditions for the regions
def region_a (x y : ℝ) := abs x ≤ 1 ∧ abs y ≤ 1
def region_b (x y : ℝ) := abs x + abs y ≤ 10
def region_c (x y : ℝ) := abs x + abs y + abs (x + y) ≤ 2020

-- Prove that the areas match the calculated solutions
theorem area_of_region_a : area_of_square = 4 := 
by sorry

theorem area_of_region_b : area_of_diamond = 200 := 
by sorry

theorem area_of_region_c : area_of_hexagon = 3060300 := 
by sorry

end area_of_region_a_area_of_region_b_area_of_region_c_l2390_239012


namespace jon_payment_per_visit_l2390_239077

theorem jon_payment_per_visit 
  (visits_per_hour : ℕ) (operating_hours_per_day : ℕ) (income_in_month : ℚ) (days_in_month : ℕ) 
  (visits_per_hour_eq : visits_per_hour = 50) 
  (operating_hours_per_day_eq : operating_hours_per_day = 24) 
  (income_in_month_eq : income_in_month = 3600) 
  (days_in_month_eq : days_in_month = 30) :
  (income_in_month / (visits_per_hour * operating_hours_per_day * days_in_month) : ℚ) = 0.10 := 
by
  sorry

end jon_payment_per_visit_l2390_239077


namespace part_I_part_II_l2390_239017

-- Part (I) 
theorem part_I (a b : ℝ) : (∀ x : ℝ, x^2 - 5 * a * x + b > 0 ↔ (x > 4 ∨ x < 1)) → 
(a = 1 ∧ b = 4) :=
by { sorry }

-- Part (II) 
theorem part_II (x y : ℝ) (a b : ℝ) (h : x + y = 2 ∧ a = 1 ∧ b = 4) : 
x > 0 → y > 0 → 
(∃ t : ℝ, t = a / x + b / y ∧ t ≥ 9 / 2) :=
by { sorry }

end part_I_part_II_l2390_239017


namespace not_divisible_1961_1963_divisible_1963_1965_l2390_239087

def is_divisible_by_three (n : Nat) : Prop :=
  n % 3 = 0

theorem not_divisible_1961_1963 : ¬ is_divisible_by_three (1961 * 1963) :=
by
  sorry

theorem divisible_1963_1965 : is_divisible_by_three (1963 * 1965) :=
by
  sorry

end not_divisible_1961_1963_divisible_1963_1965_l2390_239087


namespace inequality_holds_l2390_239039

theorem inequality_holds (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * (4 * x + y + 2 * z) * (2 * x + y + 8 * z) ≥ (375 / 2) * x * y * z :=
by
  sorry

end inequality_holds_l2390_239039


namespace pencils_purchased_l2390_239088

theorem pencils_purchased (total_cost : ℝ) (num_pens : ℕ) (pen_price : ℝ) (pencil_price : ℝ) (num_pencils : ℕ) : 
  total_cost = (num_pens * pen_price) + (num_pencils * pencil_price) → 
  num_pens = 30 → 
  pen_price = 20 → 
  pencil_price = 2 → 
  total_cost = 750 →
  num_pencils = 75 :=
by
  sorry

end pencils_purchased_l2390_239088


namespace first_term_arithmetic_sequence_l2390_239026

theorem first_term_arithmetic_sequence (S : ℕ → ℤ) (a : ℤ) (h1 : ∀ n, S n = (n * (2 * a + (n - 1) * 5)) / 2)
    (h2 : ∀ n m, (S (3 * n)) / (S m) = (S (3 * m)) / (S n)) : a = 5 / 2 := 
sorry

end first_term_arithmetic_sequence_l2390_239026


namespace value_of_x_l2390_239048

theorem value_of_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 6 = 3 * y) : x = 108 :=
by
  sorry

end value_of_x_l2390_239048


namespace triangle_inradius_exradius_l2390_239053

-- Define the properties of the triangle
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the inradius
def inradius (a b c : ℝ) (r : ℝ) : Prop :=
  r = (a + b - c) / 2

-- Define the exradius
def exradius (a b c : ℝ) (rc : ℝ) : Prop :=
  rc = (a + b + c) / 2

-- Formalize the Lean statement for the given proof problem
theorem triangle_inradius_exradius (a b c r rc: ℝ) 
  (h_triangle: right_triangle a b c) : 
  inradius a b c r ∧ exradius a b c rc :=
by
  sorry

end triangle_inradius_exradius_l2390_239053


namespace walter_time_at_seals_l2390_239036

theorem walter_time_at_seals 
  (s p e total : ℕ)
  (h1 : p = 8 * s)
  (h2 : e = 13)
  (h3 : total = 130)
  (h4 : s + p + e = total) : s = 13 := 
by 
  sorry

end walter_time_at_seals_l2390_239036


namespace parabola_vertex_calc_l2390_239069

noncomputable def vertex_parabola (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem parabola_vertex_calc 
  (a b c : ℝ) 
  (h_vertex : vertex_parabola a b c 2 = 5)
  (h_point : vertex_parabola a b c 1 = 8) : 
  a - b + c = 32 :=
sorry

end parabola_vertex_calc_l2390_239069


namespace f_nonnegative_when_a_ge_one_l2390_239001

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp (2 * x) + (a - 2) * Real.exp x - x

noncomputable def h (a : ℝ) : ℝ := Real.log a + 1 - (1 / a)

theorem f_nonnegative_when_a_ge_one (a : ℝ) (x : ℝ) (h_a : a ≥ 1) : f a x ≥ 0 := by
  sorry  -- Placeholder for the proof.

end f_nonnegative_when_a_ge_one_l2390_239001


namespace find_params_l2390_239019

theorem find_params (a b c : ℝ) :
    (∀ x : ℝ, x = 2 ∨ x = -2 → x^5 + 4 * x^4 + a * x = b * x^2 + 4 * c) 
    → a = 16 ∧ b = 48 ∧ c = -32 :=
by
  sorry

end find_params_l2390_239019


namespace yogurt_combinations_l2390_239037

-- Definitions based on conditions
def flavors : ℕ := 5
def toppings : ℕ := 8
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The problem statement to be proved
theorem yogurt_combinations :
  flavors * choose toppings 3 = 280 :=
by
  sorry

end yogurt_combinations_l2390_239037


namespace total_fruits_picked_l2390_239080

theorem total_fruits_picked :
  let sara_pears := 6
  let tim_pears := 5
  let lily_apples := 4
  let max_oranges := 3
  sara_pears + tim_pears + lily_apples + max_oranges = 18 :=
by
  -- skip the proof
  sorry

end total_fruits_picked_l2390_239080


namespace find_cost_price_l2390_239082

variable (CP SP1 SP2 : ℝ)

theorem find_cost_price
    (h1 : SP1 = CP * 0.92)
    (h2 : SP2 = CP * 1.04)
    (h3 : SP2 = SP1 + 140) :
    CP = 1166.67 :=
by
  -- Proof would be filled here
  sorry

end find_cost_price_l2390_239082


namespace bryan_bought_4_pairs_of_pants_l2390_239044

def number_of_tshirts : Nat := 5
def total_cost : Nat := 1500
def cost_per_tshirt : Nat := 100
def cost_per_pants : Nat := 250

theorem bryan_bought_4_pairs_of_pants : (total_cost - number_of_tshirts * cost_per_tshirt) / cost_per_pants = 4 := by
  sorry

end bryan_bought_4_pairs_of_pants_l2390_239044


namespace necessary_and_sufficient_condition_l2390_239006

theorem necessary_and_sufficient_condition (a b : ℝ) : a^2 * b > a * b^2 ↔ 1/a < 1/b := 
sorry

end necessary_and_sufficient_condition_l2390_239006


namespace volume_of_revolution_l2390_239066

theorem volume_of_revolution (a : ℝ) (h : 0 < a) :
  let x (θ : ℝ) := a * (1 + Real.cos θ) * Real.cos θ
  let y (θ : ℝ) := a * (1 + Real.cos θ) * Real.sin θ
  V = (8 / 3) * π * a^3 :=
sorry

end volume_of_revolution_l2390_239066


namespace greatest_value_k_l2390_239004

theorem greatest_value_k (k : ℝ) (h : ∀ x : ℝ, (x - 1) ∣ (x^2 + 2*k*x - 3*k^2)) : k ≤ 1 :=
  by
  sorry

end greatest_value_k_l2390_239004


namespace triangle_at_most_one_right_angle_l2390_239095

-- Definition of a triangle with its angles adding up to 180 degrees
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

-- The main theorem stating that a triangle can have at most one right angle.
theorem triangle_at_most_one_right_angle (α β γ : ℝ) 
  (h₁ : triangle α β γ) 
  (h₂ : α = 90 ∨ β = 90 ∨ γ = 90) : 
  (α = 90 → β ≠ 90 ∧ γ ≠ 90) ∧ 
  (β = 90 → α ≠ 90 ∧ γ ≠ 90) ∧ 
  (γ = 90 → α ≠ 90 ∧ β ≠ 90) :=
sorry

end triangle_at_most_one_right_angle_l2390_239095


namespace second_recipe_cup_count_l2390_239030

theorem second_recipe_cup_count (bottle_ounces : ℕ) (ounces_per_cup : ℕ)
  (first_recipe_cups : ℕ) (third_recipe_cups : ℕ) (bottles_needed : ℕ)
  (total_ounces : bottle_ounces = 16)
  (ounce_to_cup : ounces_per_cup = 8)
  (first_recipe : first_recipe_cups = 2)
  (third_recipe : third_recipe_cups = 3)
  (bottles : bottles_needed = 3) :
  (bottles_needed * bottle_ounces) / ounces_per_cup - first_recipe_cups - third_recipe_cups = 1 :=
by
  sorry

end second_recipe_cup_count_l2390_239030


namespace simplify_expression_l2390_239065

variable (a b : ℝ)
variable (h₁ : a = 3 + Real.sqrt 5)
variable (h₂ : b = 3 - Real.sqrt 5)

theorem simplify_expression : (a^2 - 2 * a * b + b^2) / (a^2 - b^2) * (a * b) / (a - b) = 2 / 3 := by
  sorry

end simplify_expression_l2390_239065


namespace blue_chips_count_l2390_239081

variable (T : ℕ) (blue_chips : ℕ) (white_chips : ℕ) (green_chips : ℕ)

-- Conditions
def condition1 : Prop := blue_chips = (T / 10)
def condition2 : Prop := white_chips = (T / 2)
def condition3 : Prop := green_chips = 12
def condition4 : Prop := blue_chips + white_chips + green_chips = T

-- Proof problem
theorem blue_chips_count (h1 : condition1 T blue_chips)
                          (h2 : condition2 T white_chips)
                          (h3 : condition3 green_chips)
                          (h4 : condition4 T blue_chips white_chips green_chips) :
  blue_chips = 3 :=
sorry

end blue_chips_count_l2390_239081


namespace greatest_b_value_for_integer_solution_eq_l2390_239013

theorem greatest_b_value_for_integer_solution_eq : ∀ (b : ℤ), (∃ (x : ℤ), x^2 + b * x = -20) → b > 0 → b ≤ 21 :=
by
  sorry

end greatest_b_value_for_integer_solution_eq_l2390_239013


namespace smallest_perimeter_of_triangle_with_consecutive_odd_integers_l2390_239025

theorem smallest_perimeter_of_triangle_with_consecutive_odd_integers :
  ∃ (a b c : ℕ), (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ 
  (a < b) ∧ (b < c) ∧ (c = a + 4) ∧
  (a + b > c) ∧ (b + c > a) ∧ (a + c > b) ∧ 
  (a + b + c = 15) :=
by
  sorry

end smallest_perimeter_of_triangle_with_consecutive_odd_integers_l2390_239025


namespace k_value_l2390_239010

theorem k_value (m n k : ℤ) (h₁ : m + 2 * n + 5 = 0) (h₂ : (m + 2) + 2 * (n + k) + 5 = 0) : k = -1 :=
by sorry

end k_value_l2390_239010


namespace probability_of_one_failure_l2390_239060

theorem probability_of_one_failure (p1 p2 : ℝ) (h1 : p1 = 0.90) (h2 : p2 = 0.95) :
  (p1 * (1 - p2) + (1 - p1) * p2) = 0.14 :=
by
  rw [h1, h2]
  -- Additional leaning code can be inserted here to finalize the proof if this was complete
  sorry

end probability_of_one_failure_l2390_239060


namespace mul_mod_l2390_239003

theorem mul_mod (n1 n2 n3 : ℤ) (h1 : n1 = 2011) (h2 : n2 = 1537) (h3 : n3 = 450) : 
  (2011 * 1537) % 450 = 307 := by
  sorry

end mul_mod_l2390_239003


namespace exists_consecutive_natural_numbers_satisfy_equation_l2390_239068

theorem exists_consecutive_natural_numbers_satisfy_equation :
  ∃ (n a b c d: ℕ), a = n ∧ b = n+2 ∧ c = n-1 ∧ d = n+1 ∧ n>0 ∧ a * b - c * d = 11 :=
by
  sorry

end exists_consecutive_natural_numbers_satisfy_equation_l2390_239068


namespace estimate_red_balls_l2390_239057

-- Define the conditions
variable (total_balls : ℕ)
variable (prob_red_ball : ℝ)
variable (frequency_red_ball : ℝ := prob_red_ball)

-- Assume total number of balls in the bag is 20
axiom total_balls_eq_20 : total_balls = 20

-- Assume the probability (or frequency) of drawing a red ball
axiom prob_red_ball_eq_0_25 : prob_red_ball = 0.25

-- The Lean statement
theorem estimate_red_balls (H1 : total_balls = 20) (H2 : prob_red_ball = 0.25) : total_balls * prob_red_ball = 5 :=
by
  rw [H1, H2]
  norm_num
  sorry

end estimate_red_balls_l2390_239057


namespace shortest_distance_proof_l2390_239062

noncomputable def shortest_distance (k : ℝ) : ℝ :=
  let p := (k - 6) / 2
  let f_p := -p^2 + (6 - k) * p + 18
  let d := |f_p|
  d / (Real.sqrt (k^2 + 1))

theorem shortest_distance_proof (k : ℝ) :
  shortest_distance k = 
  |(-(k - 6) / 2^2 + (6 - k) * (k - 6) / 2 + 18)| / (Real.sqrt (k^2 + 1)) :=
sorry

end shortest_distance_proof_l2390_239062


namespace solve_equation_l2390_239024

theorem solve_equation (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : 3 / x + 4 / y = 1) : 
  x = 3 * y / (y - 4) :=
sorry

end solve_equation_l2390_239024


namespace percent_relation_l2390_239023

variable (x y z : ℝ)

theorem percent_relation (h1 : x = 1.30 * y) (h2 : y = 0.60 * z) : x = 0.78 * z :=
by sorry

end percent_relation_l2390_239023


namespace bottles_drunk_l2390_239071

theorem bottles_drunk (initial_bottles remaining_bottles : ℕ)
  (h₀ : initial_bottles = 17) (h₁ : remaining_bottles = 14) :
  initial_bottles - remaining_bottles = 3 :=
sorry

end bottles_drunk_l2390_239071


namespace allowance_spent_l2390_239032

variable (A x y : ℝ)
variable (h1 : x = 0.20 * (A - y))
variable (h2 : y = 0.05 * (A - x))

theorem allowance_spent : (x + y) / A = 23 / 100 :=
by 
  sorry

end allowance_spent_l2390_239032


namespace expression_evaluation_l2390_239064

-- Define the variables and the given condition
variables (x y : ℝ)

-- Define the equation condition
def equation_condition : Prop := x - 3 * y = 4

-- State the theorem
theorem expression_evaluation (h : equation_condition x y) : 15 * y - 5 * x + 6 = -14 :=
by
  sorry

end expression_evaluation_l2390_239064


namespace sequence_contains_2017_l2390_239055

theorem sequence_contains_2017 (a1 d : ℕ) (hpos : d > 0)
  (k n m l : ℕ) 
  (hk : 25 = a1 + k * d)
  (hn : 41 = a1 + n * d)
  (hm : 65 = a1 + m * d)
  (h2017 : 2017 = a1 + l * d) : l > 0 :=
sorry

end sequence_contains_2017_l2390_239055


namespace largest_three_digit_number_satisfying_conditions_l2390_239038

def valid_digits (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 
  1 ≤ b ∧ b ≤ 9 ∧ 
  1 ≤ c ∧ c ≤ 9 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def sum_of_two_digit_permutations_eq (a b c : ℕ) : Prop :=
  22 * (a + b + c) = 100 * a + 10 * b + c

theorem largest_three_digit_number_satisfying_conditions (a b c : ℕ) :
  valid_digits a b c →
  sum_of_two_digit_permutations_eq a b c →
  100 * a + 10 * b + c ≤ 396 :=
sorry

end largest_three_digit_number_satisfying_conditions_l2390_239038


namespace solution_set_of_inequality_l2390_239015

theorem solution_set_of_inequality (x : ℝ) :
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end solution_set_of_inequality_l2390_239015


namespace square_side_length_l2390_239016

theorem square_side_length (x y : ℕ) (h_gcd : Nat.gcd x y = 5) (h_area : ∃ a : ℝ, a^2 = (169 / 6) * ↑(Nat.lcm x y)) : ∃ a : ℝ, a = 65 * Real.sqrt 2 :=
by
  sorry

end square_side_length_l2390_239016
