import Mathlib

namespace gcd_a_b_is_one_l1556_155664

-- Definitions
def a : ℤ := 100^2 + 221^2 + 320^2
def b : ℤ := 101^2 + 220^2 + 321^2

-- Theorem statement
theorem gcd_a_b_is_one : Int.gcd a b = 1 := by
  sorry

end gcd_a_b_is_one_l1556_155664


namespace find_missing_number_l1556_155625

theorem find_missing_number (x y : ℝ) 
  (h1 : (x + 50 + 78 + 104 + y) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 76.4) : 
  y = 28 :=
by
  sorry

end find_missing_number_l1556_155625


namespace perimeter_of_excircle_opposite_leg_l1556_155624

noncomputable def perimeter_of_right_triangle (a varrho_a : ℝ) : ℝ :=
  2 * varrho_a * a / (2 * varrho_a - a)

theorem perimeter_of_excircle_opposite_leg
  (a varrho_a : ℝ) (h_a_pos : 0 < a) (h_varrho_a_pos : 0 < varrho_a) :
  (perimeter_of_right_triangle a varrho_a = 2 * varrho_a * a / (2 * varrho_a - a)) :=
by
  sorry

end perimeter_of_excircle_opposite_leg_l1556_155624


namespace custom_op_diff_l1556_155673

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_diff : custom_op 8 5 - custom_op 5 8 = -12 :=
by
  sorry

end custom_op_diff_l1556_155673


namespace atomic_weight_of_calcium_l1556_155649

theorem atomic_weight_of_calcium (Ca I : ℝ) (h1 : 294 = Ca + 2 * I) (h2 : I = 126.9) : Ca = 40.2 :=
by
  sorry

end atomic_weight_of_calcium_l1556_155649


namespace sum_series_eq_four_ninths_l1556_155612

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l1556_155612


namespace jose_cupcakes_l1556_155617

theorem jose_cupcakes (lemons_needed : ℕ) (tablespoons_per_lemon : ℕ) (tablespoons_per_dozen : ℕ) (target_lemons : ℕ) : 
  (lemons_needed = 12) → 
  (tablespoons_per_lemon = 4) → 
  (target_lemons = 9) → 
  ((target_lemons * tablespoons_per_lemon / lemons_needed) = 3) :=
by
  intros h1 h2 h3
  sorry

end jose_cupcakes_l1556_155617


namespace tracy_initial_balloons_l1556_155608

theorem tracy_initial_balloons (T : ℕ) : 
  (12 + 8 + (T + 24) / 2 = 35) → T = 6 :=
by
  sorry

end tracy_initial_balloons_l1556_155608


namespace sin_double_angle_given_sum_identity_l1556_155674

theorem sin_double_angle_given_sum_identity {α : ℝ} 
  (h : Real.sin (Real.pi / 4 + α) = Real.sqrt 5 / 5) : 
  Real.sin (2 * α) = -3 / 5 := 
by 
  sorry

end sin_double_angle_given_sum_identity_l1556_155674


namespace patients_before_doubling_l1556_155603

theorem patients_before_doubling (C P : ℕ) 
    (h1 : (1 / 4) * C = 13) 
    (h2 : C = 2 * P) : 
    P = 26 := 
sorry

end patients_before_doubling_l1556_155603


namespace S_n_expression_l1556_155659

/-- 
  Given a sequence of positive terms {a_n} with sum of the first n terms represented as S_n,
  and given a_1 = 2, and given the relationship 
  S_{n+1}(S_{n+1} - 3^n) = S_n(S_n + 3^n), prove that S_{2023} = (3^2023 + 1) / 2.
-/
theorem S_n_expression
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (hr : ∀ n, S (n + 1) * (S (n + 1) - 3^n) = S n * (S n + 3^n)) :
  S 2023 = (3^2023 + 1) / 2 :=
sorry

end S_n_expression_l1556_155659


namespace number_of_true_propositions_l1556_155615

-- Definitions based on conditions
def prop1 (x : ℝ) : Prop := x^2 - x + 1 > 0
def prop2 (x : ℝ) : Prop := x^2 + x - 6 < 0 → x ≤ 2
def prop3 (x : ℝ) : Prop := (x^2 - 5*x + 6 = 0) → x = 2

-- Main theorem
theorem number_of_true_propositions : 
  (∀ x : ℝ, prop1 x) ∧ (∀ x : ℝ, prop2 x) ∧ (∃ x : ℝ, ¬ prop3 x) → 
  2 = 2 :=
by sorry

end number_of_true_propositions_l1556_155615


namespace unique_solution_xp_eq_1_l1556_155692

theorem unique_solution_xp_eq_1 (x p q : ℕ) (h1 : x ≥ 2) (h2 : p ≥ 2) (h3 : q ≥ 2):
  ((x + 1)^p - x^q = 1) ↔ (x = 2 ∧ p = 2 ∧ q = 3) :=
by 
  sorry

end unique_solution_xp_eq_1_l1556_155692


namespace pentagonal_faces_count_l1556_155643

theorem pentagonal_faces_count (x y : ℕ) (h : (5 * x + 6 * y) % 6 = 0) (h1 : ∃ v e f, v - e + f = 2 ∧ f = x + y ∧ e = (5 * x + 6 * y) / 2 ∧ v = (5 * x + 6 * y) / 3 ∧ (5 * x + 6 * y) / 3 * 3 = 5 * x + 6 * y) : 
  x = 12 :=
sorry

end pentagonal_faces_count_l1556_155643


namespace product_modulo_6_l1556_155638

theorem product_modulo_6 :
  (2017 * 2018 * 2019 * 2020) % 6 = 0 :=
by
  -- Conditions provided:
  have h1 : 2017 ≡ 5 [MOD 6] := by sorry
  have h2 : 2018 ≡ 0 [MOD 6] := by sorry
  have h3 : 2019 ≡ 1 [MOD 6] := by sorry
  have h4 : 2020 ≡ 2 [MOD 6] := by sorry
  -- Proof of the theorem:
  sorry

end product_modulo_6_l1556_155638


namespace solve_system_of_inequalities_l1556_155611

theorem solve_system_of_inequalities {x : ℝ} :
  (x + 3 ≥ 2) ∧ (2 * (x + 4) > 4 * x + 2) ↔ (-1 ≤ x ∧ x < 3) :=
by
  sorry

end solve_system_of_inequalities_l1556_155611


namespace number_of_math_books_l1556_155601

-- Definitions based on the conditions in the problem
def total_books (M H : ℕ) : Prop := M + H = 90
def total_cost (M H : ℕ) : Prop := 4 * M + 5 * H = 390

-- Proof statement
theorem number_of_math_books (M H : ℕ) (h1 : total_books M H) (h2 : total_cost M H) : M = 60 :=
  sorry

end number_of_math_books_l1556_155601


namespace solve_equation_l1556_155695

def equation_params (a x : ℝ) : Prop :=
  a * (1 / (Real.cos x) - Real.tan x) = 1

def valid_solutions (a x : ℝ) (k : ℤ) : Prop :=
  (a ≠ 0) ∧ (Real.cos x ≠ 0) ∧ (
    (|a| ≥ 1 ∧ x = Real.arccos (a / Real.sqrt (a * a + 1)) + 2 * Real.pi * k) ∨
    ((-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) ∧ x = - Real.arccos (a / Real.sqrt (a * a + 1)) + 2 * Real.pi * k)
  )

theorem solve_equation (a x : ℝ) (k : ℤ) :
  equation_params a x → valid_solutions a x k := by
  sorry

end solve_equation_l1556_155695


namespace fraction_given_to_classmates_l1556_155691

theorem fraction_given_to_classmates
  (total_boxes : ℕ) (pens_per_box : ℕ)
  (percentage_to_friends : ℝ) (pens_left_after_classmates : ℕ) :
  total_boxes = 20 →
  pens_per_box = 5 →
  percentage_to_friends = 0.40 →
  pens_left_after_classmates = 45 →
  (15 / (total_boxes * pens_per_box - percentage_to_friends * total_boxes * pens_per_box)) = 1 / 4 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_given_to_classmates_l1556_155691


namespace product_of_b_l1556_155637

noncomputable def b_product : ℤ :=
  let y1 := 3
  let y2 := 8
  let x1 := 2
  let l := y2 - y1 -- Side length of the square
  let b₁ := x1 - l -- One possible value of b
  let b₂ := x1 + l -- Another possible value of b
  b₁ * b₂ -- Product of possible values of b

theorem product_of_b :
  b_product = -21 := by
  sorry

end product_of_b_l1556_155637


namespace nth_equation_l1556_155686

open Nat

theorem nth_equation (n : ℕ) (hn : 0 < n) :
  (n + 1)/((n + 1) * (n + 1) - 1) - (1/(n * (n + 1) * (n + 2))) = 1/(n + 1) := 
by
  sorry

end nth_equation_l1556_155686


namespace amount_left_after_expenses_l1556_155628

namespace GirlScouts

def totalEarnings : ℝ := 30
def poolEntryCosts : ℝ :=
  5 * 3.5 + 3 * 2.0 + 2 * 1.0
def transportationCosts : ℝ :=
  6 * 1.5 + 4 * 0.75
def snackCosts : ℝ :=
  3 * 3.0 + 4 * 2.5 + 3 * 2.0
def totalExpenses : ℝ :=
  poolEntryCosts + transportationCosts + snackCosts
def amountLeft : ℝ :=
  totalEarnings - totalExpenses

theorem amount_left_after_expenses :
  amountLeft = -32.5 :=
by
  sorry

end GirlScouts

end amount_left_after_expenses_l1556_155628


namespace marie_saves_money_in_17_days_l1556_155635

noncomputable def number_of_days_needed (cash_register_cost revenue tax_rate costs : ℝ) : ℕ := 
  let net_revenue := revenue / (1 + tax_rate) 
  let daily_profit := net_revenue - costs
  Nat.ceil (cash_register_cost / daily_profit)

def marie_problem_conditions : Prop := 
  let bread_daily_revenue := 40 * 2
  let bagels_daily_revenue := 20 * 1.5
  let cakes_daily_revenue := 6 * 12
  let muffins_daily_revenue := 10 * 3
  let daily_revenue := bread_daily_revenue + bagels_daily_revenue + cakes_daily_revenue + muffins_daily_revenue
  let fixed_daily_costs := 20 + 2 + 80 + 30
  fixed_daily_costs = 132 ∧ daily_revenue = 212 ∧ 8 / 100 = 0.08

theorem marie_saves_money_in_17_days : marie_problem_conditions → number_of_days_needed 1040 212 0.08 132 = 17 := 
by 
  intro h
  -- Proof goes here.
  sorry

end marie_saves_money_in_17_days_l1556_155635


namespace ratio_eliminated_to_remaining_l1556_155600

theorem ratio_eliminated_to_remaining (initial_racers : ℕ) (final_racers : ℕ)
  (eliminations_1st_segment : ℕ) (eliminations_2nd_segment : ℕ) :
  initial_racers = 100 →
  final_racers = 30 →
  eliminations_1st_segment = 10 →
  eliminations_2nd_segment = initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3 - final_racers →
  (eliminations_2nd_segment / (initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3)) = 1 / 2 :=
by
  sorry

end ratio_eliminated_to_remaining_l1556_155600


namespace no_solution_in_nat_for_xx_plus_2yy_eq_zz_l1556_155622

theorem no_solution_in_nat_for_xx_plus_2yy_eq_zz :
  ¬∃ (x y z : ℕ), x^x + 2 * y^y = z^z := by
  sorry

end no_solution_in_nat_for_xx_plus_2yy_eq_zz_l1556_155622


namespace cat_food_inequality_l1556_155607

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l1556_155607


namespace f_a_plus_b_eq_neg_one_l1556_155669

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if x ≥ 0 then x * (x - b) else a * x * (x + 2)

theorem f_a_plus_b_eq_neg_one (a b : ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) a b = -f x a b) 
  (ha : a = -1) 
  (hb : b = 2) : 
  f (a + b) a b = -1 :=
by
  sorry

end f_a_plus_b_eq_neg_one_l1556_155669


namespace problem_I_problem_II_l1556_155670

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 3)

theorem problem_I (x : ℝ) : (f x > 7 - x) ↔ (x < -6 ∨ x > 2) := 
by 
  sorry

theorem problem_II (m : ℝ) : (∃ x : ℝ, f x ≤ abs (3 * m - 2)) ↔ (m ≤ -1 ∨ m ≥ 7 / 3) := 
by 
  sorry

end problem_I_problem_II_l1556_155670


namespace find_f_37_5_l1556_155604

noncomputable def f (x : ℝ) : ℝ := sorry

/--
Given that \( f \) is an odd function defined on \( \mathbb{R} \) and satisfies
\( f(x+2) = -f(x) \). When \( 0 \leqslant x \leqslant 1 \), \( f(x) = x \),
prove that \( f(37.5) = 0.5 \).
-/
theorem find_f_37_5 (f : ℝ → ℝ) (odd_f : ∀ x : ℝ, f (-x) = -f x)
  (periodic_f : ∀ x : ℝ, f (x + 2) = -f x)
  (interval_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x) : f 37.5 = 0.5 :=
sorry

end find_f_37_5_l1556_155604


namespace symmetric_circle_equation_l1556_155689

theorem symmetric_circle_equation (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 5 → (x + 1)^2 + (y - 2)^2 = 5 :=
by
  sorry

end symmetric_circle_equation_l1556_155689


namespace sample_size_calculation_l1556_155656

theorem sample_size_calculation : 
  ∀ (high_school_students junior_high_school_students sampled_high_school_students n : ℕ), 
  high_school_students = 3500 →
  junior_high_school_students = 1500 →
  sampled_high_school_students = 70 →
  n = (3500 + 1500) * 70 / 3500 →
  n = 100 :=
by
  intros high_school_students junior_high_school_students sampled_high_school_students n
  intros h1 h2 h3 h4
  sorry

end sample_size_calculation_l1556_155656


namespace find_unknown_number_l1556_155693

theorem find_unknown_number (x : ℝ) (h : (28 + 48 / x) * x = 1980) : x = 69 :=
sorry

end find_unknown_number_l1556_155693


namespace length_of_pond_l1556_155682

-- Define the problem conditions
variables (W L S : ℝ)
variables (h1 : L = 2 * W) (h2 : L = 24) 
variables (A_field A_pond : ℝ)
variables (h3 : A_pond = 1 / 8 * A_field)

-- State the theorem
theorem length_of_pond :
  A_field = L * W ∧ A_pond = S^2 ∧ A_pond = 1 / 8 * A_field ∧ L = 24 ∧ L = 2 * W → 
  S = 6 :=
by
  sorry

end length_of_pond_l1556_155682


namespace geometric_progression_terms_l1556_155620

theorem geometric_progression_terms 
  (q b4 S_n : ℚ) 
  (hq : q = 1/3) 
  (hb4 : b4 = 1/54) 
  (hS : S_n = 121/162) 
  (b1 : ℚ) 
  (hb1 : b1 = b4 * q^3)
  (Sn : ℚ) 
  (hSn : Sn = b1 * (1 - q^5) / (1 - q)) : 
  ∀ (n : ℕ), S_n = Sn → n = 5 :=
by
  intro n hn
  sorry

end geometric_progression_terms_l1556_155620


namespace flagpole_height_l1556_155675

/-
A flagpole is of certain height. It breaks, folding over in half, such that what was the tip of the flagpole is now dangling two feet above the ground. 
The flagpole broke 7 feet from the base. Prove that the height of the flagpole is 16 feet.
-/

theorem flagpole_height (H : ℝ) (h1 : H > 0) (h2 : H - 7 > 0) (h3 : H - 9 = 7) : H = 16 :=
by
  /- the proof is omitted -/
  sorry

end flagpole_height_l1556_155675


namespace men_left_hostel_l1556_155679

variable (x : ℕ)
variable (h1 : 250 * 36 = (250 - x) * 45)

theorem men_left_hostel : x = 50 :=
by
  sorry

end men_left_hostel_l1556_155679


namespace problem1_problem2_l1556_155639

theorem problem1 (α : ℝ) (h : Real.tan α = -2) : 
  (Real.sin α + 5 * Real.cos α) / (-2 * Real.cos α + Real.sin α) = -3 / 4 :=
sorry

theorem problem2 (α : ℝ) (h : Real.tan α = -2) :
  Real.sin (α - 5 * Real.pi) * Real.sin (3 * Real.pi / 2 - α) = -2 / 5 :=
sorry

end problem1_problem2_l1556_155639


namespace factorize_expression_l1556_155676

theorem factorize_expression (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1) ^ 2 :=
by sorry

end factorize_expression_l1556_155676


namespace treasure_chest_coins_l1556_155660

theorem treasure_chest_coins (hours : ℕ) (coins_per_hour : ℕ) (total_coins : ℕ) :
  hours = 8 → coins_per_hour = 25 → total_coins = hours * coins_per_hour → total_coins = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end treasure_chest_coins_l1556_155660


namespace math_problem_l1556_155648

theorem math_problem (x y : ℝ) (h : (x + 2 * y) ^ 3 + x ^ 3 + 2 * x + 2 * y = 0) : x + y - 1 = -1 := 
sorry

end math_problem_l1556_155648


namespace min_dot_product_l1556_155663

noncomputable def vec_a (m : ℝ) : ℝ × ℝ := (1 + 2^m, 1 - 2^m)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ := (4^m - 3, 4^m + 5)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem min_dot_product : ∃ m : ℝ, dot_product (vec_a m) (vec_b m) = -6 := by
  sorry

end min_dot_product_l1556_155663


namespace find_y_l1556_155646

-- Conditions as definitions in Lean 4
def angle_AXB : ℝ := 180
def angle_AX : ℝ := 70
def angle_BX : ℝ := 40
def angle_CY : ℝ := 130

-- The Lean statement for the proof problem
theorem find_y (angle_AXB_eq : angle_AXB = 180)
               (angle_AX_eq : angle_AX = 70)
               (angle_BX_eq : angle_BX = 40)
               (angle_CY_eq : angle_CY = 130) : 
               ∃ y : ℝ, y = 60 :=
by
  sorry -- The actual proof goes here.

end find_y_l1556_155646


namespace find_a_l1556_155668

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 152) : a = 50 := 
by 
  sorry

end find_a_l1556_155668


namespace factorize_expression_l1556_155610

theorem factorize_expression (x : ℝ) : 2 * x - x^2 = x * (2 - x) := sorry

end factorize_expression_l1556_155610


namespace count_odd_numbers_distinct_digits_l1556_155657

theorem count_odd_numbers_distinct_digits : 
  ∃ n : ℕ, (∀ x : ℕ, 200 ≤ x ∧ x ≤ 999 ∧ x % 2 = 1 ∧ (∀ d ∈ [digit1, digit2, digit3], d ≤ 7) ∧ (digit1 ≠ digit2 ∧ digit2 ≠ digit3 ∧ digit1 ≠ digit3) → True) ∧
  n = 120 :=
sorry

end count_odd_numbers_distinct_digits_l1556_155657


namespace square_of_cube_of_smallest_prime_l1556_155630

def smallest_prime : Nat := 2

theorem square_of_cube_of_smallest_prime :
  ((smallest_prime ^ 3) ^ 2) = 64 := by
  sorry

end square_of_cube_of_smallest_prime_l1556_155630


namespace factor_expression_l1556_155694

theorem factor_expression (x : ℝ) : 46 * x^3 - 115 * x^7 = -23 * x^3 * (5 * x^4 - 2) := 
by
  sorry

end factor_expression_l1556_155694


namespace fraction_zero_implies_value_l1556_155667

theorem fraction_zero_implies_value (x : ℝ) (h : (|x| - 2) / (x + 2) = 0) (h_non_zero : x + 2 ≠ 0) : x = 2 :=
sorry

end fraction_zero_implies_value_l1556_155667


namespace problem1_problem3_l1556_155636

-- Define the function f(x)
def f (x : ℚ) : ℚ := (1 - x) / (1 + x)

-- Problem 1: Prove f(1/x) = -f(x), given x ≠ -1, x ≠ 0
theorem problem1 (x : ℚ) (hx1 : x ≠ -1) (hx2 : x ≠ 0) : f (1 / x) = -f x :=
by sorry

-- Problem 2: Comment on graph transformations for f(x)
-- This is a conceptual question about graph translation and is not directly translatable to a Lean theorem.

-- Problem 3: Find the minimum value of M - m such that m ≤ f(x) ≤ M for x ∈ ℤ
theorem problem3 : ∃ (M m : ℤ), (∀ x : ℤ, m ≤ f x ∧ f x ≤ M) ∧ (M - m = 4) :=
by sorry

end problem1_problem3_l1556_155636


namespace opposite_reciprocal_abs_value_l1556_155690

theorem opposite_reciprocal_abs_value (a b c d m : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : abs m = 3) : 
  (a + b) / m + c * d + m = 4 ∨ (a + b) / m + c * d + m = -2 := by 
  sorry

end opposite_reciprocal_abs_value_l1556_155690


namespace solution_set_quadratic_inequality_l1556_155644

theorem solution_set_quadratic_inequality (a b : ℝ) (h1 : a < 0)
    (h2 : ∀ x, ax^2 - bx - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :
    ∀ x, x^2 - b*x - a ≥ 0 ↔ x ≥ 3 ∨ x ≤ 2 := 
by
  sorry

end solution_set_quadratic_inequality_l1556_155644


namespace bernold_wins_game_l1556_155665

/-- A game is played on a 2007 x 2007 grid. Arnold's move consists of taking a 2 x 2 square,
 and Bernold's move consists of taking a 1 x 1 square. They alternate turns with Arnold starting.
  When Arnold can no longer move, Bernold takes all remaining squares. The goal is to prove that 
  Bernold can always win the game by ensuring that Arnold cannot make enough moves to win. --/
theorem bernold_wins_game (N : ℕ) (hN : N = 2007) :
  let admissible_points := (N - 1) * (N - 1)
  let arnold_moves_needed := (N / 2) * (N / 2 + 1) / 2 + 1
  admissible_points < arnold_moves_needed :=
by
  let admissible_points := 2006 * 2006
  let arnold_moves_needed := 1003 * 1004 / 2 + 1
  exact sorry

end bernold_wins_game_l1556_155665


namespace smaller_rectangle_ratio_l1556_155678

theorem smaller_rectangle_ratio
  (length_large : ℝ) (width_large : ℝ) (area_small : ℝ)
  (h_length : length_large = 40)
  (h_width : width_large = 20)
  (h_area : area_small = 200) : 
  ∃ r : ℝ, (length_large * r) * (width_large * r) = area_small ∧ r = 0.5 :=
by
  sorry

end smaller_rectangle_ratio_l1556_155678


namespace solve_eq_l1556_155634

noncomputable def fx (x : ℝ) : ℝ :=
  ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1) * (x - 5)) /
  ((x - 2) * (x - 4) * (x - 2) * (x - 5))

theorem solve_eq (x : ℝ) (h : x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 5) :
  fx x = 1 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
by
  sorry

end solve_eq_l1556_155634


namespace percent_of_div_l1556_155698

theorem percent_of_div (P: ℝ) (Q: ℝ) (R: ℝ) : ( ( P / 100 ) * Q ) / R = 354.2 :=
by
  -- Given P = 168, Q = 1265, R = 6
  let P := 168
  let Q := 1265
  let R := 6
  -- sorry to skip the actual proof.
  sorry

end percent_of_div_l1556_155698


namespace cannot_finish_third_l1556_155653

variable (P Q R S T U : ℕ)
variable (beats : ℕ → ℕ → Prop)
variable (finishes_after : ℕ → ℕ → Prop)
variable (finishes_before : ℕ → ℕ → Prop)

noncomputable def race_conditions (P Q R S T U : ℕ) (beats finishes_after finishes_before : ℕ → ℕ → Prop) : Prop :=
  beats P Q ∧
  beats P R ∧
  beats Q S ∧
  finishes_after T P ∧
  finishes_before T Q ∧
  finishes_after U R ∧
  beats U T

theorem cannot_finish_third (P Q R S T U : ℕ) (beats finishes_after finishes_before : ℕ → ℕ → Prop) :
  race_conditions P Q R S T U beats finishes_after finishes_before →
  ¬ (finishes_before P T ∧ finishes_before T S ∧ finishes_after P R ∧ finishes_after P S) ∧ ¬ (finishes_before S T ∧ finishes_before T P) :=
sorry

end cannot_finish_third_l1556_155653


namespace final_score_is_89_l1556_155677

def final_score (s_e s_l s_b : ℝ) (p_e p_l p_b : ℝ) : ℝ :=
  s_e * p_e + s_l * p_l + s_b * p_b

theorem final_score_is_89 :
  final_score 95 92 80 0.4 0.25 0.35 = 89 := 
by
  sorry

end final_score_is_89_l1556_155677


namespace new_year_season_markup_l1556_155685

variable {C : ℝ} (hC : 0 < C)

theorem new_year_season_markup (h1 : ∀ C, C > 0 → ∃ P1, P1 = 1.20 * C)
                              (h2 : ∀ (P1 M : ℝ), M >= 0 → ∃ P2, P2 = P1 * (1 + M / 100))
                              (h3 : ∀ P2, ∃ P3, P3 = P2 * 0.91)
                              (h4 : ∃ P3, P3 = 1.365 * C) :
  ∃ M, M = 25 := 
by 
  sorry

end new_year_season_markup_l1556_155685


namespace total_number_of_students_l1556_155688

theorem total_number_of_students (girls boys : ℕ) 
  (h_ratio : 8 * girls = 5 * boys) 
  (h_girls : girls = 160) : 
  girls + boys = 416 := 
sorry

end total_number_of_students_l1556_155688


namespace new_avg_weight_l1556_155651

theorem new_avg_weight (A B C D E : ℝ) (h1 : (A + B + C) / 3 = 84) (h2 : A = 78) 
(h3 : (B + C + D + E) / 4 = 79) (h4 : E = D + 6) : 
(A + B + C + D) / 4 = 80 :=
by
  sorry

end new_avg_weight_l1556_155651


namespace undefined_hydrogen_production_l1556_155658

-- Define the chemical species involved as follows:
structure ChemQty where
  Ethane : ℕ
  Oxygen : ℕ
  CarbonDioxide : ℕ
  Water : ℕ

-- Balanced reaction equation
def balanced_reaction : ChemQty :=
  { Ethane := 2, Oxygen := 7, CarbonDioxide := 4, Water := 6 }

-- Given conditions as per problem scenario
def initial_state : ChemQty :=
  { Ethane := 1, Oxygen := 2, CarbonDioxide := 0, Water := 0 }

-- The statement reflecting the unclear result of the reaction under the given conditions.
theorem undefined_hydrogen_production :
  initial_state.Oxygen < balanced_reaction.Oxygen / balanced_reaction.Ethane * initial_state.Ethane →
  ∃ water_products : ℕ, water_products ≤ 6 * initial_state.Ethane / 2 := 
by
  -- Due to incomplete reaction
  sorry

end undefined_hydrogen_production_l1556_155658


namespace find_f_three_l1556_155654

variable {α : Type*} [LinearOrderedField α]

def f (a b c x : α) := a * x^5 - b * x^3 + c * x - 3

theorem find_f_three (a b c : α) (h : f a b c (-3) = 7) : f a b c 3 = -13 :=
by sorry

end find_f_three_l1556_155654


namespace equation_of_ellipse_AN_BM_constant_l1556_155606

noncomputable def a := 2
noncomputable def b := 1
noncomputable def e := (Real.sqrt 3) / 2
noncomputable def c := Real.sqrt 3

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

theorem equation_of_ellipse :
  ellipse a b
:=
by
  sorry

theorem AN_BM_constant (x0 y0 : ℝ) (hx : x0^2 + 4 * y0^2 = 4) :
  let AN := 2 + x0 / (y0 - 1)
  let BM := 1 + 2 * y0 / (x0 - 2)
  abs (AN * BM) = 4
:=
by
  sorry

end equation_of_ellipse_AN_BM_constant_l1556_155606


namespace initial_volume_kola_solution_l1556_155613

-- Initial composition of the kola solution
def initial_composition_sugar (V : ℝ) : ℝ := 0.20 * V

-- Final volume after additions
def final_volume (V : ℝ) : ℝ := V + 3.2 + 12 + 6.8

-- Final amount of sugar after additions
def final_amount_sugar (V : ℝ) : ℝ := initial_composition_sugar V + 3.2

-- Final percentage of sugar in the solution
def final_percentage_sugar (total_sol : ℝ) : ℝ := 0.1966850828729282 * total_sol

theorem initial_volume_kola_solution : 
  ∃ V : ℝ, final_amount_sugar V = final_percentage_sugar (final_volume V) :=
sorry

end initial_volume_kola_solution_l1556_155613


namespace mortgage_payoff_months_l1556_155684

-- Declare the initial payment (P), the common ratio (r), and the total amount (S)
def initial_payment : ℕ := 100
def common_ratio : ℕ := 3
def total_amount : ℕ := 12100

-- Define a function that calculates the sum of a geometric series
noncomputable def geom_series_sum (P : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  P * (1 - r ^ n) / (1 - r)

-- The statement we need to prove
theorem mortgage_payoff_months : ∃ n : ℕ, geom_series_sum initial_payment common_ratio n = total_amount :=
by
  sorry -- Proof to be provided

end mortgage_payoff_months_l1556_155684


namespace valid_elixir_combinations_l1556_155647

theorem valid_elixir_combinations :
  let herbs := 4
  let crystals := 6
  let incompatible_herbs := 3
  let incompatible_crystals := 2
  let total_combinations := herbs * crystals
  let incompatible_combinations := incompatible_herbs * incompatible_crystals
  total_combinations - incompatible_combinations = 18 :=
by
  sorry

end valid_elixir_combinations_l1556_155647


namespace find_k_l1556_155687

theorem find_k (k : ℝ) :
  let a := (3, 1)
  let b := (1, 3)
  let c := (k, 7)
  ((a.1 - c.1) * b.2 - (a.2 - c.2) * b.1 = 0) → k = 5 := 
by
  sorry

end find_k_l1556_155687


namespace max_g_of_15_l1556_155616

noncomputable def g (x : ℝ) : ℝ := x^3  -- Assume the polynomial g(x) = x^3 based on the maximum value found.

theorem max_g_of_15 (g : ℝ → ℝ) (h_coeff : ∀ x, 0 ≤ g x)
  (h3 : g 3 = 3) (h27 : g 27 = 1701) : g 15 = 3375 :=
by
  -- According to the problem's constraint and identified solution,
  -- here is the statement asserting that the maximum value of g(15) is 3375
  sorry

end max_g_of_15_l1556_155616


namespace find_abc_values_l1556_155671

-- Define the problem conditions as lean definitions
def represents_circle (a b c : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x - b * y + c = 0

def circle_center_and_radius_condition (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 3^2

-- Lean 4 statement for the proof problem
theorem find_abc_values (a b c : ℝ) :
  (∀ x y : ℝ, represents_circle a b c x y ↔ circle_center_and_radius_condition x y) →
  a = -2 ∧ b = 6 ∧ c = 4 :=
by
  intro h
  sorry

end find_abc_values_l1556_155671


namespace ab_ac_plus_bc_range_l1556_155605

theorem ab_ac_plus_bc_range (a b c : ℝ) (h : a + b + 2 * c = 0) :
  ∃ (k : ℝ), k ≤ 0 ∧ k = ab + ac + bc :=
sorry

end ab_ac_plus_bc_range_l1556_155605


namespace eval_sqrt4_8_pow12_l1556_155632

-- Define the fourth root of 8
def fourthRootOfEight : ℝ := 8 ^ (1 / 4)

-- Define the original expression
def expr := (fourthRootOfEight) ^ 12

-- The theorem to prove
theorem eval_sqrt4_8_pow12: expr = 512 := by
  sorry

end eval_sqrt4_8_pow12_l1556_155632


namespace count_integer_values_l1556_155633

theorem count_integer_values (x : ℕ) (h : 3 < Real.sqrt x ∧ Real.sqrt x < 5) : 
  ∃! n, (n = 15) ∧ ∀ k, (3 < Real.sqrt k ∧ Real.sqrt k < 5) → (k ≥ 10 ∧ k ≤ 24) :=
by
  sorry

end count_integer_values_l1556_155633


namespace roots_of_f_l1556_155683

noncomputable def f (a x : ℝ) : ℝ := x - Real.log (a * x)

theorem roots_of_f (a : ℝ) :
  (a < 0 → ¬∃ x : ℝ, f a x = 0) ∧
  (0 < a ∧ a < Real.exp 1 → ∃! x : ℝ, f a x = 0) ∧
  (a = Real.exp 1 → ∃! x : ℝ, f a x = 0) ∧
  (a > Real.exp 1 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :=
sorry

end roots_of_f_l1556_155683


namespace toy_playing_dogs_ratio_l1556_155699

theorem toy_playing_dogs_ratio
  (d_t : ℕ) (d_r : ℕ) (d_n : ℕ) (d_b : ℕ) (d_p : ℕ)
  (h1 : d_t = 88)
  (h2 : d_r = 12)
  (h3 : d_n = 10)
  (h4 : d_b = d_t / 4)
  (h5 : d_p = d_t - d_r - d_b - d_n) :
  d_p / d_t = 1 / 2 :=
by sorry

end toy_playing_dogs_ratio_l1556_155699


namespace maria_ann_age_problem_l1556_155631

theorem maria_ann_age_problem
  (M A : ℕ)
  (h1 : M = 7)
  (h2 : M = A - 3) :
  ∃ Y : ℕ, 7 - Y = 1 / 2 * (10 - Y) := by
  sorry

end maria_ann_age_problem_l1556_155631


namespace stickers_on_first_page_l1556_155618

theorem stickers_on_first_page :
  ∀ (a b c d e : ℕ), 
    (b = 16) →
    (c = 24) →
    (d = 32) →
    (e = 40) →
    (b - a = 8) →
    (c - b = 8) →
    (d - c = 8) →
    (e - d = 8) →
    a = 8 :=
by
  intros a b c d e hb hc hd he h1 h2 h3 h4
  -- Proof would go here
  sorry

end stickers_on_first_page_l1556_155618


namespace correct_option_is_B_l1556_155680

-- Define the total number of balls
def total_black_balls : ℕ := 3
def total_red_balls : ℕ := 7
def total_balls : ℕ := total_black_balls + total_red_balls

-- Define the event of drawing balls
def drawing_balls (n : ℕ) : Prop := n = 3

-- Define what a random variable is within this context
def is_random_variable (n : ℕ) : Prop :=
  n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3

-- The main statement to prove
theorem correct_option_is_B (n : ℕ) :
  drawing_balls n → is_random_variable n :=
by
  intro h
  sorry

end correct_option_is_B_l1556_155680


namespace a5_value_S8_value_l1556_155621

-- Definitions based on the conditions
def seq (n : ℕ) : ℕ :=
if n = 0 then 0
else if n = 1 then 1
else 2 * seq (n - 1)

noncomputable def S (n : ℕ) : ℕ :=
(1 - 2^n) / (1 - 2)

-- Proof statements
theorem a5_value : seq 5 = 16 := sorry

theorem S8_value : S 8 = 255 := sorry

end a5_value_S8_value_l1556_155621


namespace kennedy_is_larger_l1556_155627

-- Definitions based on given problem conditions
def KennedyHouse : ℕ := 10000
def BenedictHouse : ℕ := 2350
def FourTimesBenedictHouse : ℕ := 4 * BenedictHouse

-- Goal defined as a theorem to be proved
theorem kennedy_is_larger : KennedyHouse - FourTimesBenedictHouse = 600 :=
by 
  -- these are the conditions translated into Lean format
  let K := KennedyHouse
  let B := BenedictHouse
  let FourB := 4 * B
  let Goal := K - FourB
  -- prove the goal
  sorry

end kennedy_is_larger_l1556_155627


namespace average_number_of_ducks_l1556_155609

def average_ducks (A E K : ℕ) : ℕ :=
  (A + E + K) / 3

theorem average_number_of_ducks :
  ∀ (A E K : ℕ), A = 2 * E → E = K - 45 → A = 30 → average_ducks A E K = 35 :=
by 
  intros A E K h1 h2 h3
  sorry

end average_number_of_ducks_l1556_155609


namespace reflected_rectangle_has_no_point_neg_3_4_l1556_155696

structure Point where
  x : ℤ
  y : ℤ
  deriving DecidableEq, Repr

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def is_not_vertex (pts: List Point) (p: Point) : Prop :=
  ¬ (p ∈ pts)

theorem reflected_rectangle_has_no_point_neg_3_4 :
  let initial_pts := [ Point.mk 1 3, Point.mk 1 1, Point.mk 4 1, Point.mk 4 3 ]
  let reflected_pts := initial_pts.map reflect_y
  is_not_vertex reflected_pts (Point.mk (-3) 4) :=
by
  sorry

end reflected_rectangle_has_no_point_neg_3_4_l1556_155696


namespace dance_relationship_l1556_155642

theorem dance_relationship (b g : ℕ) 
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ b → i = 1 → ∃ m, m = 7)
  (h2 : b = g - 6) 
  : 7 + (b - 1) = g := 
by
  sorry

end dance_relationship_l1556_155642


namespace rise_in_water_level_l1556_155681

-- Define the conditions related to the cube and the vessel
def edge_length := 15 -- in cm
def base_length := 20 -- in cm
def base_width := 15 -- in cm

-- Calculate volumes and areas
def V_cube := edge_length ^ 3
def A_base := base_length * base_width

-- Declare the mathematical proof problem statement
theorem rise_in_water_level : 
  (V_cube / A_base : ℝ) = 11.25 :=
by
  -- edge_length, V_cube, A_base are all already defined
  -- This particularly proves (15^3) / (20 * 15) = 11.25
  sorry

end rise_in_water_level_l1556_155681


namespace vinnie_makes_more_l1556_155619

-- Define the conditions
def paul_tips : ℕ := 14
def vinnie_tips : ℕ := 30

-- Define the theorem to prove
theorem vinnie_makes_more :
  vinnie_tips - paul_tips = 16 := by
  sorry

end vinnie_makes_more_l1556_155619


namespace sum_of_legs_of_larger_triangle_l1556_155602

theorem sum_of_legs_of_larger_triangle (area_small : ℝ) (area_large : ℝ) (hypotenuse_small : ℝ) :
    (area_small = 8 ∧ area_large = 200 ∧ hypotenuse_small = 6) →
    ∃ sum_of_legs : ℝ, sum_of_legs = 41.2 :=
by
  sorry

end sum_of_legs_of_larger_triangle_l1556_155602


namespace period_of_function_is_2pi_over_3_l1556_155652

noncomputable def period_of_f (x : ℝ) : ℝ :=
  4 * (Real.sin x)^3 - Real.sin x + 2 * (Real.sin (x / 2) - Real.cos (x / 2))^2

theorem period_of_function_is_2pi_over_3 : ∀ x, period_of_f (x + (2 * Real.pi) / 3) = period_of_f x :=
by sorry

end period_of_function_is_2pi_over_3_l1556_155652


namespace counterexample_exists_l1556_155655

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := ¬ is_prime n ∧ n > 1

theorem counterexample_exists :
  ∃ n, is_composite n ∧ is_composite (n - 3) ∧ n = 18 := by
  sorry

end counterexample_exists_l1556_155655


namespace ellipse_area_constant_l1556_155645

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def passes_through (x_a y_a x_b y_b : ℝ) (p : ℝ × ℝ) : Prop := 
  p.1 = x_a ∧ p.2 = y_a ∨ p.1 = x_b ∧ p.2 = y_b

def area_ABNM_constant (x y : ℝ) : Prop :=
  let x_0 := x;
  let y_0 := y;
  let y_M := -2 * y_0 / (x_0 - 2);
  let BM := 1 + 2 * y_0 / (x_0 - 2);
  let x_N := - x_0 / (y_0 - 1);
  let AN := 2 + x_0 / (y_0 - 1);
  (1 / 2) * AN * BM = 2

theorem ellipse_area_constant :
  ∀ (a b : ℝ), (a = 2 ∧ b = 1) → 
  (∀ (x y : ℝ), 
    ellipse_equation a b x y → 
    passes_through 2 0 0 1 (x, y) → 
    (x < 0 ∧ y < 0) →
    area_ABNM_constant x y) :=
by
  intros
  sorry

end ellipse_area_constant_l1556_155645


namespace average_weight_is_15_l1556_155666

-- Define the ages of the 10 children
def ages : List ℕ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

-- Define the regression function
def weight (age : ℕ) : ℕ := 2 * age + 7

-- Function to calculate average
def average (l : List ℕ) : ℕ := l.sum / l.length

-- Define the weights of the children based on the regression function
def weights : List ℕ := ages.map weight

-- State the theorem to find the average weight of the children
theorem average_weight_is_15 : average weights = 15 := by
  sorry

end average_weight_is_15_l1556_155666


namespace simplified_expression_evaluation_l1556_155650

def expression (x y : ℝ) : ℝ :=
  3 * (x^2 - 2 * x^2 * y) - 3 * x^2 + 2 * y - 2 * (x^2 * y + y)

def x := 1/2
def y := -3

theorem simplified_expression_evaluation : expression x y = 6 :=
  sorry

end simplified_expression_evaluation_l1556_155650


namespace sum_distances_eq_6sqrt2_l1556_155662

-- Define the curves C1 and C2 in Cartesian coordinates
def curve_C1 := { p : ℝ × ℝ | p.1 + p.2 = 3 }
def curve_C2 := { p : ℝ × ℝ | p.2^2 = 2 * p.1 }

-- Defining the point P in ℝ²
def point_P : ℝ × ℝ := (1, 2)

-- Find the sum of distances |PA| + |PB|
theorem sum_distances_eq_6sqrt2 : 
  ∃ A B : ℝ × ℝ, A ∈ curve_C1 ∧ A ∈ curve_C2 ∧ 
                B ∈ curve_C1 ∧ B ∈ curve_C2 ∧ 
                (dist point_P A) + (dist point_P B) = 6 * Real.sqrt 2 := 
sorry

end sum_distances_eq_6sqrt2_l1556_155662


namespace derivative_at_1_derivative_at_neg_2_derivative_at_x0_l1556_155697

noncomputable def f (x : ℝ) : ℝ := 2 / x + x

theorem derivative_at_1 : (deriv f 1) = -1 :=
sorry

theorem derivative_at_neg_2 : (deriv f (-2)) = 1 / 2 :=
sorry

theorem derivative_at_x0 (x0 : ℝ) : (deriv f x0) = -2 / (x0^2) + 1 :=
sorry

end derivative_at_1_derivative_at_neg_2_derivative_at_x0_l1556_155697


namespace initial_apples_9_l1556_155629

def initial_apple_count (picked : ℕ) (remaining : ℕ) : ℕ :=
  picked + remaining

theorem initial_apples_9 (picked : ℕ) (remaining : ℕ) :
  picked = 2 → remaining = 7 → initial_apple_count picked remaining = 9 := by
sorry

end initial_apples_9_l1556_155629


namespace expression_evaluates_to_47_l1556_155640

theorem expression_evaluates_to_47 : 
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by 
  sorry

end expression_evaluates_to_47_l1556_155640


namespace cats_and_dogs_biscuits_l1556_155614

theorem cats_and_dogs_biscuits 
  (d c : ℕ) 
  (h1 : d + c = 10) 
  (h2 : 6 * d + 5 * c = 56) 
  : d = 6 ∧ c = 4 := 
by 
  sorry

end cats_and_dogs_biscuits_l1556_155614


namespace fifteenth_battery_replacement_month_l1556_155626

theorem fifteenth_battery_replacement_month :
  (98 % 12) + 1 = 4 :=
by
  sorry

end fifteenth_battery_replacement_month_l1556_155626


namespace norris_money_left_l1556_155623

-- Defining the conditions
def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def dec_savings : ℕ := 35
def jan_savings : ℕ := 40

def initial_savings : ℕ := sept_savings + oct_savings + nov_savings + dec_savings + jan_savings
def interest_rate : ℝ := 0.02

def total_interest : ℝ :=
  sept_savings * interest_rate + 
  (sept_savings + oct_savings) * interest_rate + 
  (sept_savings + oct_savings + nov_savings) * interest_rate +
  (sept_savings + oct_savings + nov_savings + dec_savings) * interest_rate

def total_savings_with_interest : ℝ := initial_savings + total_interest
def hugo_owes_norris : ℕ := 20 - 10

-- The final statement to prove Norris' total amount of money
theorem norris_money_left : total_savings_with_interest + hugo_owes_norris = 175.76 := by
  sorry

end norris_money_left_l1556_155623


namespace product_even_if_sum_odd_l1556_155661

theorem product_even_if_sum_odd (a b : ℤ) (h : (a + b) % 2 = 1) : (a * b) % 2 = 0 :=
sorry

end product_even_if_sum_odd_l1556_155661


namespace quadratic_inequality_solution_l1556_155672

theorem quadratic_inequality_solution (a : ℝ) (h1 : ∀ x : ℝ, ax^2 + (a + 1) * x + 1 ≥ 0) : a = 1 := by
  sorry

end quadratic_inequality_solution_l1556_155672


namespace part1_part2_l1556_155641

-- Define m as a positive integer greater than or equal to 2
def m (k : ℕ) := k ≥ 2

-- Part 1: Existential statement for x_i's
theorem part1 (m : ℕ) (h : m ≥ 2) :
  ∃ (x : ℕ → ℤ),
    ∀ i, 1 ≤ i ∧ i ≤ m →
    x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1 := by
  sorry

-- Part 2: Infinite sequence y_k
theorem part2 (x : ℕ → ℤ) (m : ℕ) (h : m ≥ 2) :
  (∀ i, 1 ≤ i ∧ i ≤ m → x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) →
  ∃ (y : ℤ → ℤ),
    (∀ k : ℤ, y k * y (m + k) = y (k + 1) * y (m + k - 1) + 1) ∧
    (∀ i, 1 ≤ i ∧ i ≤ 2 * m → y i = x i) := by
  sorry

end part1_part2_l1556_155641
