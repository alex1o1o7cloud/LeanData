import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l80_8076

-- Definition of the conditions
variables {a : ℝ} (h₀ : a > 0) (h₁ : a ≠ 1)

-- The Lean 4 statement for the problem
theorem problem_statement (h : 0 < a ∧ a < 1) : 
  (∀ x y : ℝ, x < y → a^x > a^y) → 
  (∀ x : ℝ, (2 - a) * x^3 > 0) ∧ 
  (∀ x : ℝ, (2 - a) * x^3 > 0 → 0 < a ∧ a < 2 ∧ (∀ x y : ℝ, x < y → a^x > a^y) → False) :=
by
  intros
  sorry

end NUMINAMATH_GPT_problem_statement_l80_8076


namespace NUMINAMATH_GPT_perpendicular_lines_a_value_l80_8078

theorem perpendicular_lines_a_value :
  ∀ (a : ℝ), (∀ x y : ℝ, 2 * x - y = 0) -> (∀ x y : ℝ, a * x - 2 * y - 1 = 0) ->    
  (∀ m1 m2 : ℝ, m1 = 2 -> m2 = a / 2 -> m1 * m2 = -1) -> a = -1 :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_a_value_l80_8078


namespace NUMINAMATH_GPT_domain_of_sqrt_fun_l80_8018

theorem domain_of_sqrt_fun : 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 7 → 7 + 6 * x - x^2 ≥ 0) :=
sorry

end NUMINAMATH_GPT_domain_of_sqrt_fun_l80_8018


namespace NUMINAMATH_GPT_a_minus_b_eq_three_l80_8093

theorem a_minus_b_eq_three (a b : ℝ) (h : (a+bi) * i = 1 + 2 * i) : a - b = 3 :=
by
  sorry

end NUMINAMATH_GPT_a_minus_b_eq_three_l80_8093


namespace NUMINAMATH_GPT_equal_probabilities_hearts_clubs_l80_8085

/-- Define the total number of cards in a standard deck including two Jokers -/
def total_cards := 52 + 2

/-- Define the counts of specific card types -/
def num_jokers := 2
def num_spades := 13
def num_tens := 4
def num_hearts := 13
def num_clubs := 13

/-- Define the probabilities of drawing specific card types -/
def prob_joker := num_jokers / total_cards
def prob_spade := num_spades / total_cards
def prob_ten := num_tens / total_cards
def prob_heart := num_hearts / total_cards
def prob_club := num_clubs / total_cards

theorem equal_probabilities_hearts_clubs :
  prob_heart = prob_club :=
by
  sorry

end NUMINAMATH_GPT_equal_probabilities_hearts_clubs_l80_8085


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l80_8006

theorem symmetric_point_coordinates (Q : ℝ × ℝ × ℝ) 
  (P : ℝ × ℝ × ℝ := (-6, 7, -9)) 
  (A : ℝ × ℝ × ℝ := (1, 3, -1)) 
  (B : ℝ × ℝ × ℝ := (6, 5, -2)) 
  (C : ℝ × ℝ × ℝ := (0, -3, -5)) : Q = (2, -5, 7) :=
sorry

end NUMINAMATH_GPT_symmetric_point_coordinates_l80_8006


namespace NUMINAMATH_GPT_proof_PQ_expression_l80_8091

theorem proof_PQ_expression (P Q : ℝ) (h1 : P^2 - P * Q = 1) (h2 : 4 * P * Q - 3 * Q^2 = 2) : 
  P^2 + 3 * P * Q - 3 * Q^2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_proof_PQ_expression_l80_8091


namespace NUMINAMATH_GPT_total_airflow_in_one_week_l80_8025

-- Define the conditions
def airflow_rate : ℕ := 10 -- liters per second
def working_time_per_day : ℕ := 10 -- minutes per day
def days_per_week : ℕ := 7

-- Define the conversion factors
def minutes_to_seconds : ℕ := 60

-- Define the total working time in seconds
def total_working_time_per_week : ℕ := working_time_per_day * days_per_week * minutes_to_seconds

-- Define the expected total airflow in one week
def expected_total_airflow : ℕ := airflow_rate * total_working_time_per_week

-- Prove that the expected total airflow is 42000 liters
theorem total_airflow_in_one_week : expected_total_airflow = 42000 := 
by
  -- assertion is correct given the conditions above 
  -- skip the proof
  sorry

end NUMINAMATH_GPT_total_airflow_in_one_week_l80_8025


namespace NUMINAMATH_GPT_find_e_l80_8020

theorem find_e (a b c d e : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : a + b = 32)
  (h6 : a + c = 36)
  (h7 : b + c = 37)
  (h8 : c + e = 48)
  (h9 : d + e = 51) : e = 55 / 2 :=
  sorry

end NUMINAMATH_GPT_find_e_l80_8020


namespace NUMINAMATH_GPT_quadratic_root_product_l80_8067

theorem quadratic_root_product (a b : ℝ) (m p r : ℝ)
  (h1 : a * b = 3)
  (h2 : ∀ x, x^2 - mx + 3 = 0 → x = a ∨ x = b)
  (h3 : ∀ x, x^2 - px + r = 0 → x = a + 2 / b ∨ x = b + 2 / a) :
  r = 25 / 3 := by
  sorry

end NUMINAMATH_GPT_quadratic_root_product_l80_8067


namespace NUMINAMATH_GPT_seating_arrangements_l80_8056

/-
Given:
1. There are 8 students.
2. Four different classes: (1), (2), (3), and (4).
3. Each class has 2 students.
4. There are 2 cars, Car A and Car B, each with a capacity for 4 students.
5. The two students from Class (1) (twin sisters) must ride in the same car.

Prove:
The total number of ways to seat the students such that exactly 2 students from the same class are in Car A is 24.
-/

theorem seating_arrangements : 
  ∃ (arrangements : ℕ), arrangements = 24 :=
sorry

end NUMINAMATH_GPT_seating_arrangements_l80_8056


namespace NUMINAMATH_GPT_bead_bracelet_problem_l80_8036

-- Define the condition Bead A and Bead B are always next to each other
def adjacent (A B : ℕ) (l : List ℕ) : Prop :=
  ∃ (l1 l2 : List ℕ), l = l1 ++ A :: B :: l2 ∨ l = l1 ++ B :: A :: l2

-- Define the context and translate the problem
def bracelet_arrangements (n : ℕ) : ℕ :=
  if n = 8 then 720 else 0

theorem bead_bracelet_problem : bracelet_arrangements 8 = 720 :=
by {
  -- Place proof here
  sorry 
}

end NUMINAMATH_GPT_bead_bracelet_problem_l80_8036


namespace NUMINAMATH_GPT_total_investment_amount_l80_8074

-- Define the initial conditions
def amountAt8Percent : ℝ := 3000
def interestAt8Percent (amount : ℝ) : ℝ := amount * 0.08
def interestAt10Percent (amount : ℝ) : ℝ := amount * 0.10
def totalAmount (x y : ℝ) : ℝ := x + y

-- State the theorem
theorem total_investment_amount : 
    let x := 2400
    totalAmount amountAt8Percent x = 5400 :=
by
  sorry

end NUMINAMATH_GPT_total_investment_amount_l80_8074


namespace NUMINAMATH_GPT_intersection_points_count_l80_8016

theorem intersection_points_count : 
  ∃ n : ℕ, n = 2 ∧
  (∀ x ∈ (Set.Icc 0 (2 * Real.pi)), (1 + Real.sin x = 3 / 2) → n = 2) :=
sorry

end NUMINAMATH_GPT_intersection_points_count_l80_8016


namespace NUMINAMATH_GPT_largest_additional_license_plates_l80_8079

theorem largest_additional_license_plates :
  let original_first_set := 5
  let original_second_set := 3
  let original_third_set := 4
  let original_total := original_first_set * original_second_set * original_third_set

  let new_set_case1 := original_first_set * (original_second_set + 2) * original_third_set
  let new_set_case2 := original_first_set * (original_second_set + 1) * (original_third_set + 1)

  let new_total := max new_set_case1 new_set_case2

  new_total - original_total = 40 :=
by
  let original_first_set := 5
  let original_second_set := 3
  let original_third_set := 4
  let original_total := original_first_set * original_second_set * original_third_set

  let new_set_case1 := original_first_set * (original_second_set + 2) * original_third_set
  let new_set_case2 := original_first_set * (original_second_set + 1) * (original_third_set + 1)

  let new_total := max new_set_case1 new_set_case2

  sorry

end NUMINAMATH_GPT_largest_additional_license_plates_l80_8079


namespace NUMINAMATH_GPT_cos_double_angle_l80_8041

open Real

theorem cos_double_angle {α β : ℝ} (h1 : sin α = sqrt 5 / 5)
                         (h2 : sin (α - β) = - sqrt 10 / 10)
                         (h3 : 0 < α ∧ α < π / 2)
                         (h4 : 0 < β ∧ β < π / 2) :
  cos (2 * β) = 0 :=
  sorry

end NUMINAMATH_GPT_cos_double_angle_l80_8041


namespace NUMINAMATH_GPT_min_perimeter_l80_8095

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the coordinates of the right focus, point on the hyperbola, and point M
def right_focus (F : ℝ × ℝ) : Prop := F = (2, 0)
def point_on_left_branch (P : ℝ × ℝ) : Prop := P.1 < 0 ∧ hyperbola P.1 P.2
def point_M (M : ℝ × ℝ) : Prop := M = (0, 2)

-- Perimeter of ΔPFM
noncomputable def perimeter (P F M : ℝ × ℝ) : ℝ :=
  let PF := (P.1 - F.1)^2 + (P.2 - F.2)^2
  let PM := (P.1 - M.1)^2 + (P.2 - M.2)^2
  let MF := (M.1 - F.1)^2 + (M.2 - F.2)^2
  PF.sqrt + PM.sqrt + MF.sqrt

-- Theorem statement
theorem min_perimeter (P F M : ℝ × ℝ) 
  (hF : right_focus F)
  (hP : point_on_left_branch P)
  (hM : point_M M) :
  ∃ P, perimeter P F M = 2 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_perimeter_l80_8095


namespace NUMINAMATH_GPT_compute_fractions_product_l80_8069

theorem compute_fractions_product :
  (2 * (2^4 - 1) / (2 * (2^4 + 1))) *
  (2 * (3^4 - 1) / (2 * (3^4 + 1))) *
  (2 * (4^4 - 1) / (2 * (4^4 + 1))) *
  (2 * (5^4 - 1) / (2 * (5^4 + 1))) *
  (2 * (6^4 - 1) / (2 * (6^4 + 1))) *
  (2 * (7^4 - 1) / (2 * (7^4 + 1)))
  = 4400 / 135 := by
sorry

end NUMINAMATH_GPT_compute_fractions_product_l80_8069


namespace NUMINAMATH_GPT_sum_of_discounts_l80_8015

theorem sum_of_discounts
  (price_fox : ℝ)
  (price_pony : ℝ)
  (savings : ℝ)
  (discount_pony : ℝ) :
  (3 * price_fox * (F / 100) + 2 * price_pony * (discount_pony / 100) = savings) →
  (F + discount_pony = 22) :=
sorry


end NUMINAMATH_GPT_sum_of_discounts_l80_8015


namespace NUMINAMATH_GPT_function_is_even_with_period_pi_div_2_l80_8050

noncomputable def f (x : ℝ) : ℝ := (1 + Real.cos (2 * x)) * (Real.sin x) ^ 2

theorem function_is_even_with_period_pi_div_2 : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, f (x + (π / 2)) = f x) :=
by
  sorry

end NUMINAMATH_GPT_function_is_even_with_period_pi_div_2_l80_8050


namespace NUMINAMATH_GPT_baker_cakes_left_l80_8001

theorem baker_cakes_left (cakes_made cakes_bought : ℕ) (h1 : cakes_made = 155) (h2 : cakes_bought = 140) : cakes_made - cakes_bought = 15 := by
  sorry

end NUMINAMATH_GPT_baker_cakes_left_l80_8001


namespace NUMINAMATH_GPT_b_n_geometric_a_n_formula_T_n_sum_less_than_2_l80_8008

section problem

variable {a_n : ℕ → ℝ} {b_n : ℕ → ℝ} {C_n : ℕ → ℝ} {T_n : ℕ → ℝ}

-- Given conditions
axiom seq_a (n : ℕ) : a_n 1 = 1
axiom recurrence (n : ℕ) : 2 * a_n (n + 1) - a_n n = (n - 2) / (n * (n + 1) * (n + 2))
axiom seq_b (n : ℕ) : b_n n = a_n n - 1 / (n * (n + 1))

-- Required proofs
theorem b_n_geometric : ∀ n : ℕ, b_n n = (1 / 2) ^ n := sorry
theorem a_n_formula : ∀ n : ℕ, a_n n = (1 / 2) ^ n + 1 / (n * (n + 1)) := sorry
theorem T_n_sum_less_than_2 : ∀ n : ℕ, T_n n < 2 := sorry

end problem

end NUMINAMATH_GPT_b_n_geometric_a_n_formula_T_n_sum_less_than_2_l80_8008


namespace NUMINAMATH_GPT_dividend_is_correct_l80_8073

-- Definitions of the given conditions.
def divisor : ℕ := 17
def quotient : ℕ := 4
def remainder : ℕ := 8

-- Define the dividend using the given formula.
def dividend : ℕ := (divisor * quotient) + remainder

-- The theorem to prove.
theorem dividend_is_correct : dividend = 76 := by
  -- The following line contains a placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_dividend_is_correct_l80_8073


namespace NUMINAMATH_GPT_fence_remaining_l80_8082

noncomputable def totalFence : Float := 150.0
noncomputable def ben_whitewashed : Float := 20.0

-- Remaining fence after Ben's contribution
noncomputable def remaining_after_ben : Float := totalFence - ben_whitewashed

noncomputable def billy_fraction : Float := 1.0 / 5.0
noncomputable def billy_whitewashed : Float := billy_fraction * remaining_after_ben

-- Remaining fence after Billy's contribution
noncomputable def remaining_after_billy : Float := remaining_after_ben - billy_whitewashed

noncomputable def johnny_fraction : Float := 1.0 / 3.0
noncomputable def johnny_whitewashed : Float := johnny_fraction * remaining_after_billy

-- Remaining fence after Johnny's contribution
noncomputable def remaining_after_johnny : Float := remaining_after_billy - johnny_whitewashed

noncomputable def timmy_percentage : Float := 15.0 / 100.0
noncomputable def timmy_whitewashed : Float := timmy_percentage * remaining_after_johnny

-- Remaining fence after Timmy's contribution
noncomputable def remaining_after_timmy : Float := remaining_after_johnny - timmy_whitewashed

noncomputable def alice_fraction : Float := 1.0 / 8.0
noncomputable def alice_whitewashed : Float := alice_fraction * remaining_after_timmy

-- Remaining fence after Alice's contribution
noncomputable def remaining_fence : Float := remaining_after_timmy - alice_whitewashed

theorem fence_remaining : remaining_fence = 51.56 :=
by
    -- Placeholder for actual proof
    sorry

end NUMINAMATH_GPT_fence_remaining_l80_8082


namespace NUMINAMATH_GPT_problem_equivalence_l80_8081

noncomputable def f (a b x : ℝ) : ℝ := a ^ x + b

theorem problem_equivalence (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : f a b 0 = -2) (h4 : f a b 2 = 0) :
    a = Real.sqrt 3 ∧ b = -3 ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 4, (-8 / 3 : ℝ) ≤ f a b x ∧ f a b x ≤ 6) :=
sorry

end NUMINAMATH_GPT_problem_equivalence_l80_8081


namespace NUMINAMATH_GPT_net_change_in_onions_l80_8090

-- Definitions for the given conditions
def onions_added_by_sara : ℝ := 4.5
def onions_taken_by_sally : ℝ := 5.25
def onions_added_by_fred : ℝ := 9.75

-- Statement of the problem to be proved
theorem net_change_in_onions : 
  onions_added_by_sara - onions_taken_by_sally + onions_added_by_fred = 9 := 
by
  sorry -- hint that proof is required

end NUMINAMATH_GPT_net_change_in_onions_l80_8090


namespace NUMINAMATH_GPT_valentine_count_initial_l80_8075

def valentines_given : ℕ := 42
def valentines_left : ℕ := 16
def valentines_initial := valentines_given + valentines_left

theorem valentine_count_initial :
  valentines_initial = 58 :=
by
  sorry

end NUMINAMATH_GPT_valentine_count_initial_l80_8075


namespace NUMINAMATH_GPT_find_a_if_odd_l80_8011

def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + 1) * (x + a)

theorem find_a_if_odd (a : ℝ) : (∀ x : ℝ, f (-x) a = -f x a) → a = 0 := by
  intro h
  have h0 : f 0 a = 0 := by
    simp [f]
    specialize h 0
    simp [f] at h
    exact h
  sorry

end NUMINAMATH_GPT_find_a_if_odd_l80_8011


namespace NUMINAMATH_GPT_students_in_hollow_square_are_160_l80_8065

-- Define the problem conditions
def hollow_square_formation (outer_layer : ℕ) (inner_layer : ℕ) : Prop :=
  outer_layer = 52 ∧ inner_layer = 28

-- Define the total number of students in the group based on the given condition
def total_students (n : ℕ) : Prop := n = 160

-- Prove that the total number of students is 160 given the hollow square formation conditions
theorem students_in_hollow_square_are_160 : ∀ (outer_layer inner_layer : ℕ),
  hollow_square_formation outer_layer inner_layer → total_students 160 :=
by
  intros outer_layer inner_layer h
  sorry

end NUMINAMATH_GPT_students_in_hollow_square_are_160_l80_8065


namespace NUMINAMATH_GPT_real_solution_exists_l80_8096

theorem real_solution_exists (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) :
  (x^3 - 4*x^2) / (x^2 - 5*x + 6) - x = 9 → x = 9/2 :=
by sorry

end NUMINAMATH_GPT_real_solution_exists_l80_8096


namespace NUMINAMATH_GPT_brownies_count_l80_8058

variable (total_people : Nat) (pieces_per_person : Nat) (cookies : Nat) (candy : Nat) (brownies : Nat)

def total_dessert_needed : Nat := total_people * pieces_per_person

def total_pieces_have : Nat := cookies + candy

def total_brownies_needed : Nat := total_dessert_needed total_people pieces_per_person - total_pieces_have cookies candy

theorem brownies_count (h1 : total_people = 7)
                       (h2 : pieces_per_person = 18)
                       (h3 : cookies = 42)
                       (h4 : candy = 63) :
                       total_brownies_needed total_people pieces_per_person cookies candy = 21 :=
by
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_brownies_count_l80_8058


namespace NUMINAMATH_GPT_radius_for_visibility_l80_8083

theorem radius_for_visibility (r : ℝ) (h₁ : r > 0)
  (h₂ : ∃ o : ℝ, ∀ (s : ℝ), s = 3 → o = 0):
  (∃ p : ℝ, p = 1/3) ∧ (r = 3.6) :=
sorry

end NUMINAMATH_GPT_radius_for_visibility_l80_8083


namespace NUMINAMATH_GPT_bookshelf_prices_purchasing_plans_l80_8097

/-
We are given the following conditions:
1. 3 * x + 2 * y = 1020
2. 4 * x + 3 * y = 1440

From these conditions, we need to prove that:
1. Price of type A bookshelf (x) is 180 yuan.
2. Price of type B bookshelf (y) is 240 yuan.

Given further conditions:
1. The school plans to purchase a total of 20 bookshelves.
2. Type B bookshelves not less than type A bookshelves.
3. Maximum budget of 4320 yuan.

We need to prove that the following plans are valid:
1. 8 type A bookshelves, 12 type B bookshelves.
2. 9 type A bookshelves, 11 type B bookshelves.
3. 10 type A bookshelves, 10 type B bookshelves.
-/

theorem bookshelf_prices (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 1020) 
  (h2 : 4 * x + 3 * y = 1440) : 
  x = 180 ∧ y = 240 :=
by sorry

theorem purchasing_plans (m : ℕ) 
  (h3 : 8 ≤ m ∧ m ≤ 10) 
  (h4 : 180 * m + 240 * (20 - m) ≤ 4320) 
  (h5 : 20 - m ≥ m) : 
  m = 8 ∨ m = 9 ∨ m = 10 :=
by sorry

end NUMINAMATH_GPT_bookshelf_prices_purchasing_plans_l80_8097


namespace NUMINAMATH_GPT_num_of_solutions_eq_28_l80_8004

def num_solutions : Nat :=
  sorry

theorem num_of_solutions_eq_28 : num_solutions = 28 :=
  sorry

end NUMINAMATH_GPT_num_of_solutions_eq_28_l80_8004


namespace NUMINAMATH_GPT_max_value_expr_l80_8013

theorem max_value_expr (x : ℝ) : 
  ( x ^ 6 / (x ^ 12 + 3 * x ^ 8 - 6 * x ^ 6 + 12 * x ^ 4 + 36) <= 1/18 ) :=
by
  sorry

end NUMINAMATH_GPT_max_value_expr_l80_8013


namespace NUMINAMATH_GPT_negative_expressions_l80_8052

-- Define the approximated values for P, Q, R, S, and T
def P : ℝ := 3.5
def Q : ℝ := 1.1
def R : ℝ := -0.1
def S : ℝ := 0.9
def T : ℝ := 1.5

-- State the theorem to be proved
theorem negative_expressions : 
  (R / (P * Q) < 0) ∧ ((S + T) / R < 0) :=
by
  sorry

end NUMINAMATH_GPT_negative_expressions_l80_8052


namespace NUMINAMATH_GPT_ludwig_weekly_salary_is_55_l80_8028

noncomputable def daily_salary : ℝ := 10
noncomputable def full_days : ℕ := 4
noncomputable def half_days : ℕ := 3
noncomputable def half_day_salary := daily_salary / 2

theorem ludwig_weekly_salary_is_55 :
  (full_days * daily_salary + half_days * half_day_salary = 55) := by
  sorry

end NUMINAMATH_GPT_ludwig_weekly_salary_is_55_l80_8028


namespace NUMINAMATH_GPT_x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13_l80_8021

variable {x y : ℝ}

theorem x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13
  (h1 : x + y = 10) 
  (h2 : x * y = 12) : 
  x^3 - y^3 = 176 * Real.sqrt 13 := 
by
  sorry

end NUMINAMATH_GPT_x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13_l80_8021


namespace NUMINAMATH_GPT_speed_of_stream_l80_8080

theorem speed_of_stream (c v : ℝ) (h1 : c - v = 8) (h2 : c + v = 12) : v = 2 :=
by {
  -- proof will go here
  sorry
}

end NUMINAMATH_GPT_speed_of_stream_l80_8080


namespace NUMINAMATH_GPT_sum_of_ages_l80_8042

variable (M E : ℝ)
variable (h1 : M = E + 9)
variable (h2 : M + 5 = 3 * (E - 3))

theorem sum_of_ages : M + E = 32 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l80_8042


namespace NUMINAMATH_GPT_ammonium_iodide_molecular_weight_l80_8032

theorem ammonium_iodide_molecular_weight :
  let N := 14.01
  let H := 1.008
  let I := 126.90
  let NH4I_weight := (1 * N) + (4 * H) + (1 * I)
  NH4I_weight = 144.942 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_ammonium_iodide_molecular_weight_l80_8032


namespace NUMINAMATH_GPT_dot_product_parallel_vectors_l80_8046

variable (x : ℝ)
def a : ℝ × ℝ := (x, x - 1)
def b : ℝ × ℝ := (1, 2)
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2

theorem dot_product_parallel_vectors
  (h_parallel : are_parallel (a x) b)
  (h_x : x = -1) :
  (a x).1 * (b).1 + (a x).2 * (b).2 = -5 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_parallel_vectors_l80_8046


namespace NUMINAMATH_GPT_find_s_at_3_l80_8049

def t (x : ℝ) : ℝ := 4 * x - 9
def s (y : ℝ) : ℝ := y^2 - (y + 12)

theorem find_s_at_3 : s 3 = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_s_at_3_l80_8049


namespace NUMINAMATH_GPT_distance_between_first_and_last_tree_l80_8005

theorem distance_between_first_and_last_tree
  (n : ℕ)
  (trees : ℕ)
  (dist_between_first_and_fourth : ℕ)
  (eq_dist : ℕ):
  trees = 6 ∧ dist_between_first_and_fourth = 60 ∧ eq_dist = dist_between_first_and_fourth / 3 ∧ n = (trees - 1) * eq_dist → n = 100 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_distance_between_first_and_last_tree_l80_8005


namespace NUMINAMATH_GPT_candle_height_problem_l80_8045

/-- Define the height functions of the two candles. -/
def h1 (t : ℚ) : ℚ := 1 - t / 5
def h2 (t : ℚ) : ℚ := 1 - t / 4

/-- The main theorem stating the time t when the first candle is three times the height of the second candle. -/
theorem candle_height_problem : 
  (∀ t : ℚ, h1 t = 3 * h2 t) → t = (40 : ℚ) / 11 :=
by
  sorry

end NUMINAMATH_GPT_candle_height_problem_l80_8045


namespace NUMINAMATH_GPT_triangle_area_is_4_l80_8089

-- Define the lines
def line1 (x : ℝ) : ℝ := 4
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define intersection points
def intersection1 : ℝ × ℝ := (2, 4)
def intersection2 : ℝ × ℝ := (-2, 4)
def intersection3 : ℝ × ℝ := (0, 2)

-- Function to calculate the area of a triangle using its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

-- Statement of the proof problem
theorem triangle_area_is_4 :
  ∀ A B C : ℝ × ℝ, A = intersection1 → B = intersection2 → C = intersection3 →
  triangle_area A B C = 4 := by
  sorry

end NUMINAMATH_GPT_triangle_area_is_4_l80_8089


namespace NUMINAMATH_GPT_time_per_bone_l80_8029

theorem time_per_bone (total_hours : ℕ) (total_bones : ℕ) (h1 : total_hours = 1030) (h2 : total_bones = 206) :
  (total_hours / total_bones = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_time_per_bone_l80_8029


namespace NUMINAMATH_GPT_no_hot_dogs_l80_8086

def hamburgers_initial := 9.0
def hamburgers_additional := 3.0
def hamburgers_total := 12.0

theorem no_hot_dogs (h1 : hamburgers_initial + hamburgers_additional = hamburgers_total) : 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_hot_dogs_l80_8086


namespace NUMINAMATH_GPT_solve_for_x_l80_8038

theorem solve_for_x
  (x y : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h1 : 6 * x^3 + 12 * x * y = 2 * x^4 + 3 * x^3 * y)
  (h2 : y = x^2) :
  x = (-1 + Real.sqrt 55) / 3 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l80_8038


namespace NUMINAMATH_GPT_elliot_storeroom_blocks_l80_8051

def storeroom_volume (length: ℕ) (width: ℕ) (height: ℕ) : ℕ :=
  length * width * height

def inner_volume (length: ℕ) (width: ℕ) (height: ℕ) (thickness: ℕ) : ℕ :=
  (length - 2 * thickness) * (width - 2 * thickness) * (height - thickness)

def blocks_needed (outer_volume: ℕ) (inner_volume: ℕ) : ℕ :=
  outer_volume - inner_volume

theorem elliot_storeroom_blocks :
  let length := 15
  let width := 12
  let height := 8
  let thickness := 2
  let outer_volume := storeroom_volume length width height
  let inner_volume := inner_volume length width height thickness
  let required_blocks := blocks_needed outer_volume inner_volume
  required_blocks = 912 :=
by {
  -- Definitions and calculations as per conditions
  sorry
}

end NUMINAMATH_GPT_elliot_storeroom_blocks_l80_8051


namespace NUMINAMATH_GPT_skaters_total_hours_l80_8044

-- Define the practice hours based on the conditions
def hannah_weekend_hours := 8
def hannah_weekday_extra_hours := 17
def sarah_weekday_hours := 12
def sarah_weekend_hours := 6
def emma_weekday_hour_multiplier := 2
def emma_weekend_hour_extra := 5

-- Hannah's total hours
def hannah_weekday_hours := hannah_weekend_hours + hannah_weekday_extra_hours
def hannah_total_hours := hannah_weekend_hours + hannah_weekday_hours

-- Sarah's total hours
def sarah_total_hours := sarah_weekday_hours + sarah_weekend_hours

-- Emma's total hours
def emma_weekday_hours := emma_weekday_hour_multiplier * sarah_weekday_hours
def emma_weekend_hours := sarah_weekend_hours + emma_weekend_hour_extra
def emma_total_hours := emma_weekday_hours + emma_weekend_hours

-- Total hours for all three skaters combined
def total_hours := hannah_total_hours + sarah_total_hours + emma_total_hours

-- Lean statement version only, no proof required
theorem skaters_total_hours : total_hours = 86 := by
  sorry

end NUMINAMATH_GPT_skaters_total_hours_l80_8044


namespace NUMINAMATH_GPT_exists_mn_coprime_l80_8087

theorem exists_mn_coprime (a b : ℤ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gcd : Int.gcd a b = 1) :
  ∃ (m n : ℕ), 1 ≤ m ∧ 1 ≤ n ∧ (a^m + b^n) % (a * b) = 1 % (a * b) :=
sorry

end NUMINAMATH_GPT_exists_mn_coprime_l80_8087


namespace NUMINAMATH_GPT_sin2θ_over_1pluscos2θ_eq_sqrt3_l80_8043

theorem sin2θ_over_1pluscos2θ_eq_sqrt3 {θ : ℝ} (h : Real.tan θ = Real.sqrt 3) :
  (Real.sin (2 * θ)) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_sin2θ_over_1pluscos2θ_eq_sqrt3_l80_8043


namespace NUMINAMATH_GPT_dad_strawberry_weight_l80_8066

theorem dad_strawberry_weight :
  ∀ (T L M D : ℕ), T = 36 → L = 8 → M = 12 → (D = T - L - M) → D = 16 :=
by
  intros T L M D hT hL hM hD
  rw [hT, hL, hM] at hD
  exact hD

end NUMINAMATH_GPT_dad_strawberry_weight_l80_8066


namespace NUMINAMATH_GPT_linear_function_quadrants_l80_8007

theorem linear_function_quadrants (k b : ℝ) 
  (h1 : k < 0)
  (h2 : b < 0) 
  : k * b > 0 := 
sorry

end NUMINAMATH_GPT_linear_function_quadrants_l80_8007


namespace NUMINAMATH_GPT_ratio_of_spots_to_wrinkles_l80_8072

-- Definitions
def E : ℕ := 3
def W : ℕ := 3 * E
def S : ℕ := E + W - 69

-- Theorem
theorem ratio_of_spots_to_wrinkles : S / W = 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_spots_to_wrinkles_l80_8072


namespace NUMINAMATH_GPT_margaret_mean_score_l80_8030

theorem margaret_mean_score : 
  let all_scores_sum := 832
  let cyprian_scores_count := 5
  let margaret_scores_count := 4
  let cyprian_mean_score := 92
  let cyprian_scores_sum := cyprian_scores_count * cyprian_mean_score
  (all_scores_sum - cyprian_scores_sum) / margaret_scores_count = 93 := by
  sorry

end NUMINAMATH_GPT_margaret_mean_score_l80_8030


namespace NUMINAMATH_GPT_treaty_of_versailles_original_day_l80_8048

-- Define the problem in Lean terms
def treatySignedDay : Nat -> Nat -> String
| 1919, 6 => "Saturday"
| _, _ => "Unknown"

-- Theorem statement
theorem treaty_of_versailles_original_day :
  treatySignedDay 1919 6 = "Saturday" :=
sorry

end NUMINAMATH_GPT_treaty_of_versailles_original_day_l80_8048


namespace NUMINAMATH_GPT_correct_choice_l80_8060

def proposition_p : Prop := ∀ (x : ℝ), 2^x > x^2
def proposition_q : Prop := ∃ (x_0 : ℝ), x_0 - 2 > 0

theorem correct_choice : ¬proposition_p ∧ proposition_q :=
by
  sorry

end NUMINAMATH_GPT_correct_choice_l80_8060


namespace NUMINAMATH_GPT_required_extra_money_l80_8003

theorem required_extra_money 
(Patricia_money Lisa_money Charlotte_money : ℕ) 
(hP : Patricia_money = 6) 
(hL : Lisa_money = 5 * Patricia_money) 
(hC : Lisa_money = 2 * Charlotte_money) 
(cost : ℕ) 
(hCost : cost = 100) : 
  cost - (Patricia_money + Lisa_money + Charlotte_money) = 49 := 
by 
  sorry

end NUMINAMATH_GPT_required_extra_money_l80_8003


namespace NUMINAMATH_GPT_insurance_covers_90_percent_l80_8009

-- We firstly define the variables according to the conditions.
def adoption_fee : ℕ := 150
def training_cost_per_week : ℕ := 250
def training_weeks : ℕ := 12
def certification_cost : ℕ := 3000
def total_out_of_pocket_cost : ℕ := 3450

-- We now compute intermediate results based on the conditions provided.
def total_training_cost : ℕ := training_cost_per_week * training_weeks
def out_of_pocket_cert_cost : ℕ := total_out_of_pocket_cost - adoption_fee - total_training_cost
def insurance_coverage_amount : ℕ := certification_cost - out_of_pocket_cert_cost
def insurance_coverage_percentage : ℕ := (insurance_coverage_amount * 100) / certification_cost

-- Now, we state the theorem that needs to be proven.
theorem insurance_covers_90_percent : insurance_coverage_percentage = 90 := by
  sorry

end NUMINAMATH_GPT_insurance_covers_90_percent_l80_8009


namespace NUMINAMATH_GPT_roll_contains_25_coins_l80_8014

variable (coins_per_roll : ℕ)

def rolls_per_teller := 10
def number_of_tellers := 4
def total_coins := 1000

theorem roll_contains_25_coins : 
  (number_of_tellers * rolls_per_teller * coins_per_roll = total_coins) → 
  (coins_per_roll = 25) :=
by
  sorry

end NUMINAMATH_GPT_roll_contains_25_coins_l80_8014


namespace NUMINAMATH_GPT_cost_of_toilet_paper_roll_l80_8033

-- Definitions of the problem's conditions
def num_toilet_paper_rolls : Nat := 10
def num_paper_towel_rolls : Nat := 7
def num_tissue_boxes : Nat := 3

def cost_per_paper_towel : Real := 2
def cost_per_tissue_box : Real := 2

def total_cost : Real := 35

-- The function to prove
def cost_per_toilet_paper_roll (x : Real) :=
  num_toilet_paper_rolls * x + 
  num_paper_towel_rolls * cost_per_paper_towel + 
  num_tissue_boxes * cost_per_tissue_box = total_cost

-- Statement to prove
theorem cost_of_toilet_paper_roll : 
  cost_per_toilet_paper_roll 1.5 := 
by
  simp [num_toilet_paper_rolls, num_paper_towel_rolls, num_tissue_boxes, cost_per_paper_towel, cost_per_tissue_box, total_cost]
  sorry

end NUMINAMATH_GPT_cost_of_toilet_paper_roll_l80_8033


namespace NUMINAMATH_GPT_max_initial_jars_l80_8040

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end NUMINAMATH_GPT_max_initial_jars_l80_8040


namespace NUMINAMATH_GPT_find_rosy_age_l80_8071

-- Definitions and conditions
def rosy_current_age (R : ℕ) : Prop :=
  ∃ D : ℕ,
    (D = R + 18) ∧ -- David is 18 years older than Rosy
    (D + 6 = 2 * (R + 6)) -- In 6 years, David will be twice as old as Rosy

-- Proof statement: Rosy's current age is 12
theorem find_rosy_age : rosy_current_age 12 :=
  sorry

end NUMINAMATH_GPT_find_rosy_age_l80_8071


namespace NUMINAMATH_GPT_intersection_M_N_l80_8068

def M (x : ℝ) : Prop := x^2 ≥ x

def N (x : ℝ) (y : ℝ) : Prop := y = 3^x + 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | ∃ y : ℝ, N x y ∧ y > 1} = {x : ℝ | x > 1} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_M_N_l80_8068


namespace NUMINAMATH_GPT_ratio_of_distances_l80_8012

-- Define the given conditions
variables (w x y : ℕ)
variables (h1 : w > 0) -- walking speed must be positive
variables (h2 : x > 0) -- distance from home must be positive
variables (h3 : y > 0) -- distance to stadium must be positive

-- Define the two times:
-- Time taken to walk directly to the stadium
def time_walk (w y : ℕ) := y / w

-- Time taken to walk home, then bike to the stadium
def time_walk_bike (w x y : ℕ) := x / w + (x + y) / (5 * w)

-- Given that both times are equal
def times_equal (w x y : ℕ) := time_walk w y = time_walk_bike w x y

-- We want to prove that the ratio of x to y is 2/3
theorem ratio_of_distances (w x y : ℕ) (h_time_eq : times_equal w x y) : x / y = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_distances_l80_8012


namespace NUMINAMATH_GPT_monotonicity_of_f_inequality_f_l80_8024

section
variables {f : ℝ → ℝ}
variables (h_dom : ∀ x, x > 0 → f x > 0)
variables (h_f2 : f 2 = 1)
variables (h_fxy : ∀ x y, f (x * y) = f x + f y)
variables (h_pos : ∀ x, 1 < x → f x > 0)

-- Monotonicity of f(x)
theorem monotonicity_of_f :
  ∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2 :=
sorry

-- Inequality f(x) + f(x-2) ≤ 3 
theorem inequality_f (x : ℝ) :
  2 < x ∧ x ≤ 4 → f x + f (x - 2) ≤ 3 :=
sorry

end

end NUMINAMATH_GPT_monotonicity_of_f_inequality_f_l80_8024


namespace NUMINAMATH_GPT_wind_velocity_l80_8039

theorem wind_velocity (P A V : ℝ) (k : ℝ := 1/200) :
  (P = k * A * V^2) →
  (P = 2) → (A = 1) → (V = 20) →
  ∀ (P' A' : ℝ), P' = 128 → A' = 4 → ∃ V' : ℝ, V'^2 = 6400 :=
by
  intros h1 h2 h3 h4 P' A' h5 h6
  use 80
  linarith

end NUMINAMATH_GPT_wind_velocity_l80_8039


namespace NUMINAMATH_GPT_sequence_a7_l80_8061

/-- 
  Given a sequence {a_n} such that a_1 + a_{2n-1} = 4n - 6, 
  prove that a_7 = 11 
-/
theorem sequence_a7 (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : a 7 = 11 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a7_l80_8061


namespace NUMINAMATH_GPT_smallest_rectangles_to_cover_square_l80_8098

theorem smallest_rectangles_to_cover_square :
  ∃ n : ℕ, 
    (∃ a : ℕ, a = 3 * 4) ∧
    (∃ k : ℕ, k = lcm 3 4) ∧
    (∃ s : ℕ, s = k * k) ∧
    (s / a = n) ∧
    n = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_rectangles_to_cover_square_l80_8098


namespace NUMINAMATH_GPT_inclination_line_eq_l80_8047

theorem inclination_line_eq (l : ℝ → ℝ) (h1 : ∃ x, l x = 2 ∧ ∃ y, l y = 2) (h2 : ∃ θ, θ = 135) :
  ∃ a b c, a = 1 ∧ b = 1 ∧ c = -4 ∧ ∀ x y, y = l x → a * x + b * y + c = 0 :=
by 
  sorry

end NUMINAMATH_GPT_inclination_line_eq_l80_8047


namespace NUMINAMATH_GPT_positive_solution_y_l80_8017

theorem positive_solution_y (x y z : ℝ) 
  (h1 : x * y = 8 - 3 * x - 2 * y) 
  (h2 : y * z = 15 - 5 * y - 3 * z) 
  (h3 : x * z = 40 - 5 * x - 4 * z) : 
  y = 4 := 
sorry

end NUMINAMATH_GPT_positive_solution_y_l80_8017


namespace NUMINAMATH_GPT_janet_clarinet_hours_l80_8094

theorem janet_clarinet_hours 
  (C : ℕ)  -- number of clarinet lessons hours per week
  (clarinet_cost_per_hour : ℕ := 40)
  (piano_cost_per_hour : ℕ := 28)
  (hours_of_piano_per_week : ℕ := 5)
  (annual_extra_piano_cost : ℕ := 1040) :
  52 * (piano_cost_per_hour * hours_of_piano_per_week - clarinet_cost_per_hour * C) = annual_extra_piano_cost → 
  C = 3 :=
by
  sorry

end NUMINAMATH_GPT_janet_clarinet_hours_l80_8094


namespace NUMINAMATH_GPT_expression_for_A_l80_8002

theorem expression_for_A (A k : ℝ)
  (h : ∀ k : ℝ, Ax^2 + 6 * k * x + 2 = 0 → k = 0.4444444444444444 → (6 * k)^2 - 4 * A * 2 = 0) :
  A = 9 * k^2 / 2 := 
sorry

end NUMINAMATH_GPT_expression_for_A_l80_8002


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l80_8088

theorem arithmetic_sequence_sum (x y : ℕ)
  (h₁ : ∃ d, 9 = 3 + d)  -- Common difference exists, d = 6
  (h₂ : ∃ n, 15 = 3 + n * 6)  -- Arithmetic sequence term verification
  (h₃ : y = 33 - 6)
  (h₄ : x = 27 - 6) : x + y = 48 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l80_8088


namespace NUMINAMATH_GPT_betty_total_blue_and_green_beads_l80_8037

theorem betty_total_blue_and_green_beads (r b g : ℕ) (h1 : 5 * b = 3 * r) (h2 : 5 * g = 2 * r) (h3 : r = 50) : b + g = 50 :=
by
  sorry

end NUMINAMATH_GPT_betty_total_blue_and_green_beads_l80_8037


namespace NUMINAMATH_GPT_triangle_inequality_l80_8077

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a / (b + c) + b / (c + a) + c / (a + b) ≤ 2 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l80_8077


namespace NUMINAMATH_GPT_inverse_value_at_2_l80_8057

noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := x / (1 - 2 * x)

theorem inverse_value_at_2 :
  f_inv 2 = -2/3 := by
  sorry

end NUMINAMATH_GPT_inverse_value_at_2_l80_8057


namespace NUMINAMATH_GPT_prob_A_second_day_is_correct_l80_8031

-- Definitions for the problem conditions
def prob_first_day_A : ℝ := 0.5
def prob_A_given_A : ℝ := 0.6
def prob_first_day_B : ℝ := 0.5
def prob_A_given_B : ℝ := 0.5

-- Calculate the probability of going to A on the second day
def prob_A_second_day : ℝ :=
  prob_first_day_A * prob_A_given_A + prob_first_day_B * prob_A_given_B

-- The theorem statement
theorem prob_A_second_day_is_correct : 
  prob_A_second_day = 0.55 :=
by
  unfold prob_A_second_day prob_first_day_A prob_A_given_A prob_first_day_B prob_A_given_B
  sorry

end NUMINAMATH_GPT_prob_A_second_day_is_correct_l80_8031


namespace NUMINAMATH_GPT_math_proof_problem_l80_8019

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

def parallel (x : Line) (y : Plane) : Prop := sorry
def contained_in (x : Line) (y : Plane) : Prop := sorry
def perpendicular (x : Plane) (y : Plane) : Prop := sorry
def perpendicular_line_plane (x : Line) (y : Plane) : Prop := sorry

theorem math_proof_problem :
  (perpendicular α β) ∧ (perpendicular_line_plane m β) ∧ ¬(contained_in m α) → parallel m α :=
by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l80_8019


namespace NUMINAMATH_GPT_probability_of_4_rainy_days_out_of_6_l80_8054

noncomputable def probability_of_rain_on_given_day : ℝ := 0.5

noncomputable def probability_of_rain_on_exactly_k_days (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem probability_of_4_rainy_days_out_of_6 :
  probability_of_rain_on_exactly_k_days 6 4 probability_of_rain_on_given_day = 0.234375 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_4_rainy_days_out_of_6_l80_8054


namespace NUMINAMATH_GPT_min_value_x_plus_y_l80_8027

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 / x + 1 / y = 1 / 2) : x + y ≥ 18 := sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l80_8027


namespace NUMINAMATH_GPT_gcd_20586_58768_l80_8059

theorem gcd_20586_58768 : Int.gcd 20586 58768 = 2 := by
  sorry

end NUMINAMATH_GPT_gcd_20586_58768_l80_8059


namespace NUMINAMATH_GPT_train_speed_l80_8055

theorem train_speed (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_man_km_hr : ℝ) 
  (h_length : length_of_train = 420)
  (h_time : time_to_cross = 62.99496040316775)
  (h_man_speed : speed_of_man_km_hr = 6) :
  ∃ speed_of_train_km_hr : ℝ, speed_of_train_km_hr = 30 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l80_8055


namespace NUMINAMATH_GPT_sweets_distribution_l80_8035

theorem sweets_distribution (S X : ℕ) (h1 : S = 112 * X) (h2 : S = 80 * (X + 6)) :
  X = 15 := 
by
  sorry

end NUMINAMATH_GPT_sweets_distribution_l80_8035


namespace NUMINAMATH_GPT_arithmetic_sequence_first_term_l80_8053

theorem arithmetic_sequence_first_term
  (a : ℕ) -- First term of the arithmetic sequence
  (d : ℕ := 3) -- Common difference, given as 3
  (n : ℕ := 20) -- Number of terms, given as 20
  (S : ℕ := 650) -- Sum of the sequence, given as 650
  (h : S = (n / 2) * (2 * a + (n - 1) * d)) : a = 4 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_first_term_l80_8053


namespace NUMINAMATH_GPT_last_digit_101_pow_100_l80_8084

theorem last_digit_101_pow_100 :
  (101^100) % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_101_pow_100_l80_8084


namespace NUMINAMATH_GPT_largest_a_for_integer_solution_l80_8092

noncomputable def largest_integer_a : ℤ := 11

theorem largest_a_for_integer_solution :
  ∃ (x : ℤ), ∃ (a : ℤ), 
  (∃ (a : ℤ), a ≤ largest_integer_a) ∧
  (a = largest_integer_a → (
    (x^2 - (a + 7) * x + 7 * a)^3 = -3^3)) := 
by 
  sorry

end NUMINAMATH_GPT_largest_a_for_integer_solution_l80_8092


namespace NUMINAMATH_GPT_devin_teaching_years_l80_8010

section DevinTeaching
variable (Calculus Algebra Statistics Geometry DiscreteMathematics : ℕ)

theorem devin_teaching_years :
  Calculus = 4 ∧
  Algebra = 2 * Calculus ∧
  Statistics = 5 * Algebra ∧
  Geometry = 3 * Statistics ∧
  DiscreteMathematics = Geometry / 2 ∧
  (Calculus + Algebra + Statistics + Geometry + DiscreteMathematics) = 232 :=
by
  sorry
end DevinTeaching

end NUMINAMATH_GPT_devin_teaching_years_l80_8010


namespace NUMINAMATH_GPT_equal_real_roots_of_quadratic_eq_l80_8000

theorem equal_real_roots_of_quadratic_eq {k : ℝ} (h : ∃ x : ℝ, (x^2 + 3 * x - k = 0) ∧ ∀ y : ℝ, (y^2 + 3 * y - k = 0) → y = x) : k = -9 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_equal_real_roots_of_quadratic_eq_l80_8000


namespace NUMINAMATH_GPT_length_of_platform_l80_8070

theorem length_of_platform (length_train speed_train time_crossing speed_train_mps distance_train_cross : ℝ)
  (h1 : length_train = 120)
  (h2 : speed_train = 60)
  (h3 : time_crossing = 20)
  (h4 : speed_train_mps = 16.67)
  (h5 : distance_train_cross = speed_train_mps * time_crossing):
  (distance_train_cross = length_train + 213.4) :=
by
  sorry

end NUMINAMATH_GPT_length_of_platform_l80_8070


namespace NUMINAMATH_GPT_daniel_age_is_correct_l80_8063

open Nat

-- Define Uncle Ben's age
def uncleBenAge : ℕ := 50

-- Define Edward's age as two-thirds of Uncle Ben's age
def edwardAge : ℚ := (2 / 3) * uncleBenAge

-- Define that Daniel is 7 years younger than Edward
def danielAge : ℚ := edwardAge - 7

-- Assert that Daniel's age is 79/3 years old
theorem daniel_age_is_correct : danielAge = 79 / 3 := by
  sorry

end NUMINAMATH_GPT_daniel_age_is_correct_l80_8063


namespace NUMINAMATH_GPT_max_isosceles_triangles_l80_8026

theorem max_isosceles_triangles 
  {A B C D P : ℝ} 
  (h_collinear: A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ D)
  (h_non_collinear: P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D)
  : (∀ a b c : ℝ, (a = P ∨ a = A ∨ a = B ∨ a = C ∨ a = D) ∧ (b = P ∨ b = A ∨ b = B ∨ b = C ∨ b = D) ∧ (c = P ∨ c = A ∨ c = B ∨ c = C ∨ c = D) 
    ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ((a - b)^2 + (b - c)^2 = (a - c)^2 ∨ (a - c)^2 + (b - c)^2 = (a - b)^2 ∨ (a - b)^2 + (a - c)^2 = (b - c)^2)) → 
    isosceles_triangle_count = 6 :=
sorry

end NUMINAMATH_GPT_max_isosceles_triangles_l80_8026


namespace NUMINAMATH_GPT_z_in_second_quadrant_l80_8034

open Complex

-- Given the condition
def satisfies_eqn (z : ℂ) : Prop := z * (1 - I) = 4 * I

-- Define the second quadrant condition
def in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (h : satisfies_eqn z) : in_second_quadrant z :=
  sorry

end NUMINAMATH_GPT_z_in_second_quadrant_l80_8034


namespace NUMINAMATH_GPT_area_of_rectangle_l80_8022

variables {group_interval rate : ℝ}

theorem area_of_rectangle (length_of_small_rectangle : ℝ) (height_of_small_rectangle : ℝ) :
  (length_of_small_rectangle = group_interval) → (height_of_small_rectangle = rate / group_interval) →
  length_of_small_rectangle * height_of_small_rectangle = rate :=
by
  intros h_length h_height
  rw [h_length, h_height]
  exact mul_div_cancel' rate (by sorry)

end NUMINAMATH_GPT_area_of_rectangle_l80_8022


namespace NUMINAMATH_GPT_sequence_term_l80_8023

theorem sequence_term (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) (hn : n > 0)
  (hSn : ∀ n, S n = n^2)
  (hrec : ∀ n, n > 1 → a n = S n - S (n-1)) :
  a n = 2 * n - 1 := by
  -- Base case
  cases n with
  | zero => contradiction  -- n > 0 implies n ≠ 0
  | succ n' =>
    cases n' with
    | zero => sorry  -- When n = 0 + 1 = 1, we need to show a 1 = 2 * 1 - 1 = 1 based on given conditions
    | succ k => sorry -- When n = k + 1, we use the provided recursive relation to prove the statement

end NUMINAMATH_GPT_sequence_term_l80_8023


namespace NUMINAMATH_GPT_max_length_of_third_side_of_triangle_l80_8064

noncomputable def max_third_side_length (D E F : ℝ) (a b : ℝ) : ℝ :=
  let c_square := a^2 + b^2 - 2 * a * b * Real.cos (90 * Real.pi / 180)
  Real.sqrt c_square

theorem max_length_of_third_side_of_triangle (D E F : ℝ) (a b : ℝ) (h₁ : Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1)
    (h₂ : a = 8) (h₃ : b = 15) : 
    max_third_side_length D E F a b = 17 := 
by
  sorry

end NUMINAMATH_GPT_max_length_of_third_side_of_triangle_l80_8064


namespace NUMINAMATH_GPT_percentage_third_year_students_l80_8062

-- Define the conditions as given in the problem
variables (T : ℝ) (T_3 : ℝ) (S_2 : ℝ)

-- Conditions
def cond1 : Prop := S_2 = 0.10 * T
def cond2 : Prop := (0.10 * T) / (T - T_3) = 1 / 7

-- Define the proof goal
theorem percentage_third_year_students (h1 : cond1 T S_2) (h2 : cond2 T T_3) : T_3 = 0.30 * T :=
sorry

end NUMINAMATH_GPT_percentage_third_year_students_l80_8062


namespace NUMINAMATH_GPT_det_of_matrix_M_l80_8099

open Matrix

def M : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![2, -4, 4], 
    ![0, 6, -2], 
    ![5, -3, 2]]

theorem det_of_matrix_M : Matrix.det M = -68 :=
by
  sorry

end NUMINAMATH_GPT_det_of_matrix_M_l80_8099
