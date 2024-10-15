import Mathlib

namespace NUMINAMATH_GPT_total_books_in_school_l2050_205013

theorem total_books_in_school (tables_A tables_B tables_C : ℕ)
  (books_per_table_A books_per_table_B books_per_table_C : ℕ → ℕ)
  (hA : tables_A = 750)
  (hB : tables_B = 500)
  (hC : tables_C = 850)
  (h_books_per_table_A : ∀ n, books_per_table_A n = 3 * n / 5)
  (h_books_per_table_B : ∀ n, books_per_table_B n = 2 * n / 5)
  (h_books_per_table_C : ∀ n, books_per_table_C n = n / 3) :
  books_per_table_A tables_A + books_per_table_B tables_B + books_per_table_C tables_C = 933 :=
by sorry

end NUMINAMATH_GPT_total_books_in_school_l2050_205013


namespace NUMINAMATH_GPT_number_of_connections_l2050_205012

-- Definitions based on conditions
def switches : ℕ := 15
def connections_per_switch : ℕ := 4

-- Theorem statement proving the correct number of connections
theorem number_of_connections : switches * connections_per_switch / 2 = 30 := by
  sorry

end NUMINAMATH_GPT_number_of_connections_l2050_205012


namespace NUMINAMATH_GPT_find_number_l2050_205015

theorem find_number (x : ℤ) (h : x * 9999 = 806006795) : x = 80601 :=
sorry

end NUMINAMATH_GPT_find_number_l2050_205015


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l2050_205002

theorem positive_difference_of_two_numbers :
  ∀ (x y : ℝ), x + y = 8 → x^2 - y^2 = 24 → abs (x - y) = 3 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l2050_205002


namespace NUMINAMATH_GPT_final_net_worth_l2050_205075

noncomputable def initial_cash_A := (20000 : ℤ)
noncomputable def initial_cash_B := (22000 : ℤ)
noncomputable def house_value := (20000 : ℤ)
noncomputable def vehicle_value := (10000 : ℤ)

noncomputable def transaction_1_cash_A := initial_cash_A + 25000
noncomputable def transaction_1_cash_B := initial_cash_B - 25000

noncomputable def transaction_2_cash_A := transaction_1_cash_A - 12000
noncomputable def transaction_2_cash_B := transaction_1_cash_B + 12000

noncomputable def transaction_3_cash_A := transaction_2_cash_A + 18000
noncomputable def transaction_3_cash_B := transaction_2_cash_B - 18000

noncomputable def transaction_4_cash_A := transaction_3_cash_A + 9000
noncomputable def transaction_4_cash_B := transaction_3_cash_B + 9000

noncomputable def final_value_A := transaction_4_cash_A
noncomputable def final_value_B := transaction_4_cash_B + house_value + vehicle_value

theorem final_net_worth :
  final_value_A - initial_cash_A = 40000 ∧ final_value_B - initial_cash_B = 8000 :=
by
  sorry

end NUMINAMATH_GPT_final_net_worth_l2050_205075


namespace NUMINAMATH_GPT_solid_triangle_front_view_l2050_205045

def is_triangle_front_view (solid : ℕ) : Prop :=
  solid = 1 ∨ solid = 2 ∨ solid = 3 ∨ solid = 5

theorem solid_triangle_front_view (s : ℕ) (h : s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 4 ∨ s = 5 ∨ s = 6):
  is_triangle_front_view s ↔ (s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 5) :=
by
  sorry

end NUMINAMATH_GPT_solid_triangle_front_view_l2050_205045


namespace NUMINAMATH_GPT_max_cookie_price_l2050_205021

theorem max_cookie_price (k p : ℕ) :
  8 * k + 3 * p < 200 →
  4 * k + 5 * p > 150 →
  k ≤ 19 :=
sorry

end NUMINAMATH_GPT_max_cookie_price_l2050_205021


namespace NUMINAMATH_GPT_rectangle_dimensions_l2050_205072

theorem rectangle_dimensions (l w : ℝ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 2880) :
  (l = 86.833 ∧ w = 33.167) ∨ (l = 33.167 ∧ w = 86.833) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l2050_205072


namespace NUMINAMATH_GPT_f_comp_g_eq_g_comp_f_has_solution_l2050_205008

variable {R : Type*} [Field R]

def f (a b x : R) : R := a * x + b
def g (c d x : R) : R := c * x ^ 2 + d

theorem f_comp_g_eq_g_comp_f_has_solution (a b c d : R) :
  (∃ x : R, f a b (g c d x) = g c d (f a b x)) ↔ (c = 0 ∨ a * b = 0) ∧ (a * d - c * b ^ 2 + b - d = 0) := by
  sorry

end NUMINAMATH_GPT_f_comp_g_eq_g_comp_f_has_solution_l2050_205008


namespace NUMINAMATH_GPT_trapezium_other_parallel_side_l2050_205016

theorem trapezium_other_parallel_side (a : ℝ) (b d : ℝ) (area : ℝ) 
  (h1 : a = 18) (h2 : d = 15) (h3 : area = 285) : b = 20 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_other_parallel_side_l2050_205016


namespace NUMINAMATH_GPT_minimum_value_l2050_205017

theorem minimum_value (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : a + b = 1) : 
  (∃ (x : ℝ), x = a + 2*b) → (∃ (y : ℝ), y = 2*a + b) → 
  (∀ (x y : ℝ), x + y = 3 → (1/x + 4/y) ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l2050_205017


namespace NUMINAMATH_GPT_min_hypotenuse_of_right_triangle_l2050_205063

theorem min_hypotenuse_of_right_triangle (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a + b + c = 6) : 
  c = 6 * (Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_GPT_min_hypotenuse_of_right_triangle_l2050_205063


namespace NUMINAMATH_GPT_real_part_zero_implies_x3_l2050_205001

theorem real_part_zero_implies_x3 (x : ℝ) : 
  (x^2 - 2*x - 3 = 0) ∧ (x + 1 ≠ 0) → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_real_part_zero_implies_x3_l2050_205001


namespace NUMINAMATH_GPT_efficiency_ratio_l2050_205032

variable {A B : ℝ}

theorem efficiency_ratio (hA : A = 1 / 30) (hAB : A + B = 1 / 20) : A / B = 2 :=
by
  sorry

end NUMINAMATH_GPT_efficiency_ratio_l2050_205032


namespace NUMINAMATH_GPT_arithmetic_seq_sum_a4_a6_l2050_205003

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_sum_a4_a6 (a : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_root1 : a 3 ^ 2 - 3 * a 3 + 1 = 0)
  (h_root2 : a 7 ^ 2 - 3 * a 7 + 1 = 0) :
  a 4 + a 6 = 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_a4_a6_l2050_205003


namespace NUMINAMATH_GPT_factorization_correct_l2050_205096

noncomputable def original_poly (x : ℝ) : ℝ := 12 * x ^ 2 + 18 * x - 24
noncomputable def factored_poly (x : ℝ) : ℝ := 6 * (2 * x - 1) * (x + 4)

theorem factorization_correct (x : ℝ) : original_poly x = factored_poly x :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l2050_205096


namespace NUMINAMATH_GPT_sum_of_slopes_eq_zero_l2050_205023

theorem sum_of_slopes_eq_zero
  (p : ℝ) (a : ℝ) (hp : p > 0) (ha : a > 0)
  (P Q : ℝ × ℝ)
  (hP : P.2 ^ 2 = 2 * p * P.1)
  (hQ : Q.2 ^ 2 = 2 * p * Q.1)
  (hcollinear : ∃ m : ℝ, ∀ (x y : (ℝ × ℝ)), y = P ∨ y = Q ∨ y = (-a, 0) → y.2 = m * (y.1 + a)) :
  let k_AP := (P.2) / (P.1 - a)
  let k_AQ := (Q.2) / (Q.1 - a)
  k_AP + k_AQ = 0 := by
    sorry

end NUMINAMATH_GPT_sum_of_slopes_eq_zero_l2050_205023


namespace NUMINAMATH_GPT_sum_of_three_consecutive_odds_is_69_l2050_205091

-- Definition for the smallest of three consecutive odd numbers
def smallest_consecutive_odd := 21

-- Define the three consecutive odd numbers based on the smallest one
def first_consecutive_odd := smallest_consecutive_odd
def second_consecutive_odd := smallest_consecutive_odd + 2
def third_consecutive_odd := smallest_consecutive_odd + 4

-- Calculate the sum of these three consecutive odd numbers
def sum_consecutive_odds := first_consecutive_odd + second_consecutive_odd + third_consecutive_odd

-- Theorem statement that the sum of these three consecutive odd numbers is 69
theorem sum_of_three_consecutive_odds_is_69 : 
  sum_consecutive_odds = 69 := by
    sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_odds_is_69_l2050_205091


namespace NUMINAMATH_GPT_increasing_function_a_l2050_205065

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≥ 0 then
    x^2
  else
    x^3 - (a-1)*x + a^2 - 3*a - 4

theorem increasing_function_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f x a ≤ f y a) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_increasing_function_a_l2050_205065


namespace NUMINAMATH_GPT_odd_function_fixed_point_l2050_205035

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_fixed_point 
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f) :
  f (0) = 0 → f (-1 + 1) - 2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_fixed_point_l2050_205035


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_evaluate_at_zero_l2050_205092

theorem simplify_and_evaluate_expr (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 2) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1)) = (2 + x) / (2 - x) :=
by
  sorry

theorem evaluate_at_zero :
  (2 + 0 : ℝ) / (2 - 0) = 1 :=
by
  norm_num

end NUMINAMATH_GPT_simplify_and_evaluate_expr_evaluate_at_zero_l2050_205092


namespace NUMINAMATH_GPT_tom_tim_typing_ratio_l2050_205056

variable (T M : ℝ)

theorem tom_tim_typing_ratio (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) : M / T = 5 :=
sorry

end NUMINAMATH_GPT_tom_tim_typing_ratio_l2050_205056


namespace NUMINAMATH_GPT_min_value_xy_expression_l2050_205062

theorem min_value_xy_expression : ∃ x y : ℝ, (xy - 2)^2 + (x^2 + y^2) = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_xy_expression_l2050_205062


namespace NUMINAMATH_GPT_largest_stamps_per_page_l2050_205081

-- Definitions of the conditions
def stamps_book1 : ℕ := 1260
def stamps_book2 : ℕ := 1470

-- Statement to be proven: The largest number of stamps per page (gcd of 1260 and 1470)
theorem largest_stamps_per_page : Nat.gcd stamps_book1 stamps_book2 = 210 :=
by
  sorry

end NUMINAMATH_GPT_largest_stamps_per_page_l2050_205081


namespace NUMINAMATH_GPT_diff_in_set_l2050_205074

variable (A : Set Int)
variable (ha : ∃ a ∈ A, a > 0)
variable (hb : ∃ b ∈ A, b < 0)
variable (h : ∀ {a b : Int}, a ∈ A → b ∈ A → (2 * a) ∈ A ∧ (a + b) ∈ A)

theorem diff_in_set (x y : Int) (hx : x ∈ A) (hy : y ∈ A) : (x - y) ∈ A :=
  sorry

end NUMINAMATH_GPT_diff_in_set_l2050_205074


namespace NUMINAMATH_GPT_sum_of_distinct_integers_l2050_205030

theorem sum_of_distinct_integers (a b c d e : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) 
(h_prod : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120) : 
a + b + c + d + e = 33 := 
sorry

end NUMINAMATH_GPT_sum_of_distinct_integers_l2050_205030


namespace NUMINAMATH_GPT_distance_between_M_and_focus_l2050_205088

theorem distance_between_M_and_focus
  (θ : ℝ)
  (x y : ℝ)
  (M : ℝ × ℝ := (1/2, 0))
  (F : ℝ × ℝ := (0, 1/2))
  (hx : x = 2 * Real.cos θ)
  (hy : y = 1 + Real.cos (2 * θ)) :
  Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_M_and_focus_l2050_205088


namespace NUMINAMATH_GPT_equation_solutions_exist_l2050_205018

theorem equation_solutions_exist (d x y : ℤ) (hx : Odd x) (hy : Odd y)
  (hxy : x^2 - d * y^2 = -4) : ∃ X Y : ℕ, X^2 - d * Y^2 = -1 :=
by
  sorry  -- Proof is omitted as per the instructions

end NUMINAMATH_GPT_equation_solutions_exist_l2050_205018


namespace NUMINAMATH_GPT_equality_of_costs_l2050_205038

variable (x : ℝ)
def C1 : ℝ := 50 + 0.35 * (x - 500)
def C2 : ℝ := 75 + 0.45 * (x - 1000)

theorem equality_of_costs : C1 x = C2 x → x = 2500 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_equality_of_costs_l2050_205038


namespace NUMINAMATH_GPT_a_2005_l2050_205064

noncomputable def a : ℕ → ℤ := sorry 

axiom a3 : a 3 = 5
axiom a5 : a 5 = 8
axiom exists_n : ∃ (n : ℕ), n > 0 ∧ a n + a (n + 1) + a (n + 2) = 7

theorem a_2005 : a 2005 = -6 := by {
  sorry
}

end NUMINAMATH_GPT_a_2005_l2050_205064


namespace NUMINAMATH_GPT_probability_of_6_consecutive_heads_l2050_205007

/-- Define the probability of obtaining at least 6 consecutive heads in 10 flips of a fair coin. -/
def prob_at_least_6_consecutive_heads : ℚ :=
  129 / 1024

/-- Proof statement: The probability of getting at least 6 consecutive heads in 10 flips of a fair coin is 129/1024. -/
theorem probability_of_6_consecutive_heads : 
  prob_at_least_6_consecutive_heads = 129 / 1024 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_6_consecutive_heads_l2050_205007


namespace NUMINAMATH_GPT_translate_parabola_l2050_205010

-- Translating the parabola y = (x-2)^2 - 8 three units left and five units up
theorem translate_parabola (x y : ℝ) :
  y = (x - 2) ^ 2 - 8 →
  y = ((x + 3) - 2) ^ 2 - 8 + 5 →
  y = (x + 1) ^ 2 - 3 := by
sorry

end NUMINAMATH_GPT_translate_parabola_l2050_205010


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l2050_205087

theorem boat_speed_in_still_water:
  ∀ (V_b : ℝ) (V_s : ℝ) (D : ℝ),
    V_s = 3 → 
    (D = (V_b + V_s) * 1) → 
    (D = (V_b - V_s) * 1.5) → 
    V_b = 15 :=
by
  intros V_b V_s D V_s_eq H_downstream H_upstream
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l2050_205087


namespace NUMINAMATH_GPT_Lakers_win_in_7_games_l2050_205094

-- Variables for probabilities given in the problem
variable (p_Lakers_win : ℚ := 1 / 4) -- Lakers' probability of winning a single game
variable (p_Celtics_win : ℚ := 3 / 4) -- Celtics' probability of winning a single game

-- Probabilities and combinations
def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_Lakers_win_game7 : ℚ :=
  let first_6_games := binom 6 3 * (p_Lakers_win ^ 3) * (p_Celtics_win ^ 3)
  let seventh_game := p_Lakers_win
  first_6_games * seventh_game

theorem Lakers_win_in_7_games : probability_Lakers_win_game7 = 540 / 16384 := by
  sorry

end NUMINAMATH_GPT_Lakers_win_in_7_games_l2050_205094


namespace NUMINAMATH_GPT_number_of_rabbits_is_38_l2050_205099

-- Conditions: 
def ducks : ℕ := 52
def chickens : ℕ := 78
def condition (ducks rabbits chickens : ℕ) : Prop := 
  chickens = ducks + rabbits - 12

-- Statement: Prove that the number of rabbits is 38
theorem number_of_rabbits_is_38 : ∃ R : ℕ, condition ducks R chickens ∧ R = 38 := by
  sorry

end NUMINAMATH_GPT_number_of_rabbits_is_38_l2050_205099


namespace NUMINAMATH_GPT_speed_in_kmh_l2050_205050

def distance : ℝ := 550.044
def time : ℝ := 30
def conversion_factor : ℝ := 3.6

theorem speed_in_kmh : (distance / time) * conversion_factor = 66.00528 := 
by
  sorry

end NUMINAMATH_GPT_speed_in_kmh_l2050_205050


namespace NUMINAMATH_GPT_problem_solution_l2050_205004

open Real

/-- If (y / 6) / 3 = 6 / (y / 3), then y is ±18. -/
theorem problem_solution (y : ℝ) (h : (y / 6) / 3 = 6 / (y / 3)) : y = 18 ∨ y = -18 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2050_205004


namespace NUMINAMATH_GPT_product_xyz_l2050_205014

theorem product_xyz (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) : x * y * z = -1 :=
by
  sorry

end NUMINAMATH_GPT_product_xyz_l2050_205014


namespace NUMINAMATH_GPT_john_fixes_8_computers_l2050_205044

theorem john_fixes_8_computers 
  (total_computers : ℕ)
  (unfixable_percentage : ℝ)
  (waiting_percentage : ℝ) 
  (h1 : total_computers = 20)
  (h2 : unfixable_percentage = 0.2)
  (h3 : waiting_percentage = 0.4) :
  let fixed_right_away := total_computers * (1 - unfixable_percentage - waiting_percentage)
  fixed_right_away = 8 :=
by
  sorry

end NUMINAMATH_GPT_john_fixes_8_computers_l2050_205044


namespace NUMINAMATH_GPT_intersection_of_parabola_with_y_axis_l2050_205054

theorem intersection_of_parabola_with_y_axis :
  ∃ y : ℝ, y = - (0 + 2)^2 + 6 ∧ (0, y) = (0, 2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_parabola_with_y_axis_l2050_205054


namespace NUMINAMATH_GPT_indoor_table_chairs_l2050_205000

theorem indoor_table_chairs (x : ℕ) :
  (9 * x) + (11 * 3) = 123 → x = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_indoor_table_chairs_l2050_205000


namespace NUMINAMATH_GPT_smallest_integer_to_make_y_perfect_square_l2050_205098

-- Define y as given in the problem
def y : ℕ :=
  2^33 * 3^54 * 4^45 * 5^76 * 6^57 * 7^38 * 8^69 * 9^10

-- Smallest integer n such that (y * n) is a perfect square
theorem smallest_integer_to_make_y_perfect_square
  : ∃ n : ℕ, (∀ k : ℕ, y * n = k * k) ∧ (∀ m : ℕ, (∀ k : ℕ, y * m = k * k) → n ≤ m) := 
sorry

end NUMINAMATH_GPT_smallest_integer_to_make_y_perfect_square_l2050_205098


namespace NUMINAMATH_GPT_simplify_division_l2050_205060

theorem simplify_division (a b c d : ℕ) (h1 : a = 27) (h2 : b = 10^12) (h3 : c = 9) (h4 : d = 10^4) :
  ((a * b) / (c * d) = 300000000) :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_division_l2050_205060


namespace NUMINAMATH_GPT_expr_simplify_l2050_205078

variable {a b c d m : ℚ}
variable {b_nonzero : b ≠ 0}
variable {m_nat : ℕ}
variable {m_bound : 0 ≤ m_nat ∧ m_nat < 2}

def expr_value (a b c d m : ℚ) : ℚ :=
  m - (c * d) + (a + b) / 2023 + a / b

theorem expr_simplify (h1 : a = -b) (h2 : c * d = 1) (h3 : m = (m_nat : ℚ)) :
  expr_value a b c d m = -1 ∨ expr_value a b c d m = -2 := by
  sorry

end NUMINAMATH_GPT_expr_simplify_l2050_205078


namespace NUMINAMATH_GPT_solve_y_l2050_205031

theorem solve_y (y : ℝ) (h : (4 * y - 2) / (5 * y - 5) = 3 / 4) : y = -7 :=
by
  sorry

end NUMINAMATH_GPT_solve_y_l2050_205031


namespace NUMINAMATH_GPT_find_x_value_l2050_205006

open Real

theorem find_x_value (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
(h3 : tan (150 * π / 180 - x * π / 180) = (sin (150 * π / 180) - sin (x * π / 180)) / (cos (150 * π / 180) - cos (x * π / 180))) :
x = 120 :=
sorry

end NUMINAMATH_GPT_find_x_value_l2050_205006


namespace NUMINAMATH_GPT_number_of_students_l2050_205067

theorem number_of_students (S N : ℕ) (h1 : S = 15 * N)
                           (h2 : (8 * 14) = 112)
                           (h3 : (6 * 16) = 96)
                           (h4 : 17 = 17)
                           (h5 : S = 225) : N = 15 :=
by sorry

end NUMINAMATH_GPT_number_of_students_l2050_205067


namespace NUMINAMATH_GPT_possible_values_of_m_l2050_205084

theorem possible_values_of_m (m : ℝ) (A B : Set ℝ) (hA : A = {-1, 1}) (hB : B = {x | m * x = 1}) (hUnion : A ∪ B = A) : m = 0 ∨ m = 1 ∨ m = -1 :=
sorry

end NUMINAMATH_GPT_possible_values_of_m_l2050_205084


namespace NUMINAMATH_GPT_area_triangle_ABC_correct_l2050_205083

noncomputable def rectangle_area : ℝ := 42

noncomputable def area_triangle_outside_I : ℝ := 9
noncomputable def area_triangle_outside_II : ℝ := 3.5
noncomputable def area_triangle_outside_III : ℝ := 12

noncomputable def area_triangle_ABC : ℝ :=
  rectangle_area - (area_triangle_outside_I + area_triangle_outside_II + area_triangle_outside_III)

theorem area_triangle_ABC_correct : area_triangle_ABC = 17.5 := by 
  sorry

end NUMINAMATH_GPT_area_triangle_ABC_correct_l2050_205083


namespace NUMINAMATH_GPT_f_inequality_l2050_205049

-- Definition of odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- f is an odd function
variable {f : ℝ → ℝ}
variable (h1 : is_odd_function f)

-- f has a period of 4
variable (h2 : ∀ x, f (x + 4) = f x)

-- f is monotonically increasing on [0, 2)
variable (h3 : ∀ x y, 0 ≤ x → x < y → y < 2 → f x < f y)

theorem f_inequality : f 3 < 0 ∧ 0 < f 1 :=
by 
  -- Place proof here
  sorry

end NUMINAMATH_GPT_f_inequality_l2050_205049


namespace NUMINAMATH_GPT_annual_decrease_rate_l2050_205093

theorem annual_decrease_rate (P₀ P₂ : ℝ) (r : ℝ) (h₀ : P₀ = 8000) (h₂ : P₂ = 5120) :
  P₂ = P₀ * (1 - r / 100) ^ 2 → r = 20 :=
by
  intros h
  have h₀' : P₀ = 8000 := h₀
  have h₂' : P₂ = 5120 := h₂
  sorry

end NUMINAMATH_GPT_annual_decrease_rate_l2050_205093


namespace NUMINAMATH_GPT_min_value_of_2x_plus_4y_l2050_205022

noncomputable def minimum_value (x y : ℝ) : ℝ := 2^x + 4^y

theorem min_value_of_2x_plus_4y (x y : ℝ) (h : x + 2 * y = 3) : minimum_value x y = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_2x_plus_4y_l2050_205022


namespace NUMINAMATH_GPT_product_divisible_by_third_l2050_205076

theorem product_divisible_by_third (a b c : Int)
    (h1 : (a + b + c)^2 = -(a * b + a * c + b * c))
    (h2 : a + b ≠ 0) (h3 : b + c ≠ 0) (h4 : a + c ≠ 0) :
    ((a + b) * (a + c) % (b + c) = 0) ∧ ((a + b) * (b + c) % (a + c) = 0) ∧ ((a + c) * (b + c) % (a + b) = 0) :=
  sorry

end NUMINAMATH_GPT_product_divisible_by_third_l2050_205076


namespace NUMINAMATH_GPT_ratio_of_poets_to_novelists_l2050_205071

-- Define the conditions
def total_people : ℕ := 24
def novelists : ℕ := 15
def poets := total_people - novelists

-- Theorem asserting the ratio of poets to novelists
theorem ratio_of_poets_to_novelists (h1 : poets = total_people - novelists) : poets / novelists = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_of_poets_to_novelists_l2050_205071


namespace NUMINAMATH_GPT_quadratic_eq_coeff_l2050_205089

theorem quadratic_eq_coeff (x : ℝ) : 
  (x^2 + 2 = 3 * x) = (∃ a b c : ℝ, a = 1 ∧ b = -3 ∧ c = 2 ∧ (a * x^2 + b * x + c = 0)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_coeff_l2050_205089


namespace NUMINAMATH_GPT_max_naive_number_l2050_205069

-- Define the digits and conditions for a naive number
variable (a b c d : ℕ)
variable (M : ℕ)
variable (h1 : b = c + 2)
variable (h2 : a = d + 6)
variable (h3 : M = 1000 * a + 100 * b + 10 * c + d)

-- Define P(M) and Q(M)
def P (a b c d : ℕ) : ℕ := 3 * (a + b) + c + d
def Q (a : ℕ) : ℕ := a - 5

-- Problem statement: Prove the maximum value of M satisfying the divisibility condition
theorem max_naive_number (div_cond : (P a b c d) % (Q a) = 0) (hq : Q a % 10 = 0) : M = 9313 := 
sorry

end NUMINAMATH_GPT_max_naive_number_l2050_205069


namespace NUMINAMATH_GPT_irrational_roots_of_odd_coeffs_l2050_205057

theorem irrational_roots_of_odd_coeffs (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := 
sorry

end NUMINAMATH_GPT_irrational_roots_of_odd_coeffs_l2050_205057


namespace NUMINAMATH_GPT_mow_lawn_payment_l2050_205041

theorem mow_lawn_payment (bike_cost weekly_allowance babysitting_rate babysitting_hours money_saved target_savings mowing_payment : ℕ) 
  (h1 : bike_cost = 100)
  (h2 : weekly_allowance = 5)
  (h3 : babysitting_rate = 7)
  (h4 : babysitting_hours = 2)
  (h5 : money_saved = 65)
  (h6 : target_savings = 6) :
  mowing_payment = 10 :=
sorry

end NUMINAMATH_GPT_mow_lawn_payment_l2050_205041


namespace NUMINAMATH_GPT_find_ellipse_equation_l2050_205059

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∃ c : ℝ, a > b ∧ b > 0 ∧ 4 * a = 16 ∧ |c| = 2 ∧ a^2 = b^2 + c^2

theorem find_ellipse_equation :
  (∃ (a b : ℝ), ellipse_equation a b) → (∃ b : ℝ, (a = 4) ∧ (b > 0) ∧ (b^2 = 12) ∧ (∀ x y : ℝ, (x^2 / 16) + (y^2 / 12) = 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_ellipse_equation_l2050_205059


namespace NUMINAMATH_GPT_total_cups_of_mushroom_soup_l2050_205061

def cups_team_1 : ℕ := 90
def cups_team_2 : ℕ := 120
def cups_team_3 : ℕ := 70

theorem total_cups_of_mushroom_soup :
  cups_team_1 + cups_team_2 + cups_team_3 = 280 :=
  by sorry

end NUMINAMATH_GPT_total_cups_of_mushroom_soup_l2050_205061


namespace NUMINAMATH_GPT_quadratic_function_vertex_and_comparison_l2050_205009

theorem quadratic_function_vertex_and_comparison
  (a b c : ℝ)
  (A_conds : 4 * a - 2 * b + c = 9)
  (B_conds : c = 3)
  (C_conds : 16 * a + 4 * b + c = 3) :
  (a = 1/2 ∧ b = -2 ∧ c = 3) ∧
  (∀ (m : ℝ) (y₁ y₂ : ℝ),
     y₁ = 1/2 * m^2 - 2 * m + 3 ∧
     y₂ = 1/2 * (m + 1)^2 - 2 * (m + 1) + 3 →
     (m < 3/2 → y₁ > y₂) ∧
     (m = 3/2 → y₁ = y₂) ∧
     (m > 3/2 → y₁ < y₂)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_vertex_and_comparison_l2050_205009


namespace NUMINAMATH_GPT_width_of_door_l2050_205068

theorem width_of_door 
  (L W H : ℕ) 
  (cost_per_sq_ft : ℕ) 
  (door_height window_height window_width : ℕ) 
  (num_windows total_cost : ℕ) 
  (door_width : ℕ) 
  (total_wall_area area_door area_windows area_to_whitewash : ℕ)
  (raw_area_door raw_area_windows total_walls_to_paint : ℕ) 
  (cost_per_sq_ft_eq : cost_per_sq_ft = 9)
  (total_cost_eq : total_cost = 8154)
  (room_dimensions_eq : L = 25 ∧ W = 15 ∧ H = 12)
  (door_dimensions_eq : door_height = 6)
  (window_dimensions_eq : window_height = 3 ∧ window_width = 4)
  (num_windows_eq : num_windows = 3)
  (total_wall_area_eq : total_wall_area = 2 * (L * H) + 2 * (W * H))
  (raw_area_door_eq : raw_area_door = door_height * door_width)
  (raw_area_windows_eq : raw_area_windows = num_windows * (window_width * window_height))
  (total_walls_to_paint_eq : total_walls_to_paint = total_wall_area - raw_area_door - raw_area_windows)
  (area_to_whitewash_eq : area_to_whitewash = 924 - 6 * door_width)
  (total_cost_eq_calc : total_cost = area_to_whitewash * cost_per_sq_ft) :
  door_width = 3 := sorry

end NUMINAMATH_GPT_width_of_door_l2050_205068


namespace NUMINAMATH_GPT_max_xy_l2050_205053

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 9 * y = 12) : xy ≤ 4 :=
by
sorry

end NUMINAMATH_GPT_max_xy_l2050_205053


namespace NUMINAMATH_GPT_consecutive_odd_integers_expressions_l2050_205048

theorem consecutive_odd_integers_expressions
  {p q : ℤ} (hpq : p + 2 = q ∨ p - 2 = q) (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) :
  (2 * p + 5 * q) % 2 = 1 ∧ (5 * p - 2 * q) % 2 = 1 ∧ (2 * p * q + 5) % 2 = 1 :=
  sorry

end NUMINAMATH_GPT_consecutive_odd_integers_expressions_l2050_205048


namespace NUMINAMATH_GPT_range_of_m_l2050_205079

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + |x - 1| ≥ (m + 2) * x - 1) ↔ (-3 - 2 * Real.sqrt 2) ≤ m ∧ m ≤ 0 := 
sorry

end NUMINAMATH_GPT_range_of_m_l2050_205079


namespace NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l2050_205026

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 + n

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then S n else S n - S (n - 1)

noncomputable def b_n (n : ℕ) : ℕ := 2^(n - 1)

noncomputable def T_n (n : ℕ) : ℕ :=
  (4 * n - 5) * 2^n + 5

theorem problem_part1 (n : ℕ) (h : n > 0) : n > 0 → a_n n = 4 * n - 1 := by
  sorry

theorem problem_part2 (n : ℕ) (h : n > 0) : n > 0 → b_n n = 2^(n - 1) := by
  sorry

theorem problem_part3 (n : ℕ) (h : n > 0) : n > 0 → T_n n = (4 * n - 5) * 2^n + 5 := by
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l2050_205026


namespace NUMINAMATH_GPT_spherical_coord_plane_l2050_205066

-- Let's define spherical coordinates and the condition theta = c.
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

def is_plane (c : ℝ) (p : SphericalCoordinates) : Prop :=
  p.θ = c

theorem spherical_coord_plane (c : ℝ) : 
  ∀ p : SphericalCoordinates, is_plane c p → True := 
by
  intros p hp
  sorry

end NUMINAMATH_GPT_spherical_coord_plane_l2050_205066


namespace NUMINAMATH_GPT_geom_seq_sum_l2050_205046

theorem geom_seq_sum (q : ℝ) (a₃ a₄ a₅ : ℝ) : 
  0 < q ∧ 3 * (1 - q^3) / (1 - q) = 21 ∧ a₃ = 3 * q^2 ∧ a₄ = 3 * q^3 ∧ a₅ = 3 * q^4 
  -> a₃ + a₄ + a₅ = 84 := 
by 
  sorry

end NUMINAMATH_GPT_geom_seq_sum_l2050_205046


namespace NUMINAMATH_GPT_cubic_root_sum_eq_constant_term_divided_l2050_205027

theorem cubic_root_sum_eq_constant_term_divided 
  (a b c : ℝ) 
  (h_roots : (24 * a^3 - 36 * a^2 + 14 * a - 1 = 0) 
           ∧ (24 * b^3 - 36 * b^2 + 14 * b - 1 = 0) 
           ∧ (24 * c^3 - 36 * c^2 + 14 * c - 1 = 0))
  (h_bounds : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) 
  : (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = (158 / 73) := 
sorry

end NUMINAMATH_GPT_cubic_root_sum_eq_constant_term_divided_l2050_205027


namespace NUMINAMATH_GPT_tangent_line_ln_curve_l2050_205047

theorem tangent_line_ln_curve (a : ℝ) :
  (∃ x y : ℝ, y = Real.log x + a ∧ x - y + 1 = 0 ∧ (∀ t : ℝ, t = x → (t - (Real.log t + a)) = -(1 - a))) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_ln_curve_l2050_205047


namespace NUMINAMATH_GPT_three_digit_number_divisible_by_eleven_l2050_205037

theorem three_digit_number_divisible_by_eleven
  (x : ℕ) (n : ℕ)
  (units_digit_is_two : n % 10 = 2)
  (hundreds_digit_is_seven : n / 100 = 7)
  (tens_digit : n = 700 + x * 10 + 2)
  (divisibility_condition : (7 - x + 2) % 11 = 0) :
  n = 792 := by
  sorry

end NUMINAMATH_GPT_three_digit_number_divisible_by_eleven_l2050_205037


namespace NUMINAMATH_GPT_correct_choice_is_C_l2050_205036

def is_opposite_number (a b : ℤ) : Prop := a + b = 0

def option_A : Prop := ¬is_opposite_number (2^3) (3^2)
def option_B : Prop := ¬is_opposite_number (-2) (-|-2|)
def option_C : Prop := is_opposite_number ((-3)^2) (-3^2)
def option_D : Prop := ¬is_opposite_number 2 (-(-2))

theorem correct_choice_is_C : option_C ∧ option_A ∧ option_B ∧ option_D :=
by
  sorry

end NUMINAMATH_GPT_correct_choice_is_C_l2050_205036


namespace NUMINAMATH_GPT_trapezium_area_l2050_205043

theorem trapezium_area (a b area h : ℝ) (h1 : a = 20) (h2 : b = 15) (h3 : area = 245) :
  area = 1 / 2 * (a + b) * h → h = 14 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_area_l2050_205043


namespace NUMINAMATH_GPT_fathers_age_multiple_l2050_205070

theorem fathers_age_multiple 
  (Johns_age : ℕ)
  (sum_of_ages : ℕ)
  (additional_years : ℕ)
  (m : ℕ)
  (h1 : Johns_age = 15)
  (h2 : sum_of_ages = 77)
  (h3 : additional_years = 32)
  (h4 : sum_of_ages = Johns_age + (Johns_age * m + additional_years)) :
  m = 2 := 
by 
  sorry

end NUMINAMATH_GPT_fathers_age_multiple_l2050_205070


namespace NUMINAMATH_GPT_find_probability_of_B_l2050_205055

-- Define the conditions and the problem
def system_A_malfunction_prob := 1 / 10
def at_least_one_not_malfunction_prob := 49 / 50

/-- The probability that System B malfunctions given that 
  the probability of at least one system not malfunctioning 
  is 49/50 and the probability of System A malfunctioning is 1/10 -/
theorem find_probability_of_B (p : ℝ) 
  (h1 : system_A_malfunction_prob = 1 / 10) 
  (h2 : at_least_one_not_malfunction_prob = 49 / 50) 
  (h3 : 1 - (system_A_malfunction_prob * p) = at_least_one_not_malfunction_prob) : 
  p = 1 / 5 :=
sorry

end NUMINAMATH_GPT_find_probability_of_B_l2050_205055


namespace NUMINAMATH_GPT_recipe_flour_requirement_l2050_205085

def sugar_cups : ℕ := 9
def salt_cups : ℕ := 40
def flour_initial_cups : ℕ := 4
def additional_flour : ℕ := sugar_cups + 1
def total_flour_cups : ℕ := additional_flour

theorem recipe_flour_requirement : total_flour_cups = 10 := by
  sorry

end NUMINAMATH_GPT_recipe_flour_requirement_l2050_205085


namespace NUMINAMATH_GPT_travel_west_l2050_205082

-- Define the condition
def travel_east (d: ℝ) : ℝ := d

-- Define the distance for east
def east_distance := (travel_east 3 = 3)

-- The theorem to prove that traveling west for 2km should be -2km
theorem travel_west (d: ℝ) (h: east_distance) : travel_east (-d) = -d := 
by
  sorry

-- Applying this theorem to the specific case of 2km travel
example (h: east_distance): travel_east (-2) = -2 :=
by 
  apply travel_west 2 h

end NUMINAMATH_GPT_travel_west_l2050_205082


namespace NUMINAMATH_GPT_fifth_eq_l2050_205028

theorem fifth_eq :
  (1 = 1) ∧
  (2 + 3 + 4 = 9) ∧
  (3 + 4 + 5 + 6 + 7 = 25) ∧
  (4 + 5 + 6 + 7 + 8 + 9 + 10 = 49) →
  5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fifth_eq_l2050_205028


namespace NUMINAMATH_GPT_lego_tower_levels_l2050_205040

theorem lego_tower_levels (initial_pieces : ℕ) (pieces_per_level : ℕ) (pieces_left : ℕ) 
    (h1 : initial_pieces = 100) (h2 : pieces_per_level = 7) (h3 : pieces_left = 23) :
    (initial_pieces - pieces_left) / pieces_per_level = 11 := 
by
  sorry

end NUMINAMATH_GPT_lego_tower_levels_l2050_205040


namespace NUMINAMATH_GPT_cube_edge_length_l2050_205025

def radius := 2
def edge_length (r : ℕ) := 4 + 2 * r

theorem cube_edge_length :
  ∀ r : ℕ, r = radius → edge_length r = 8 :=
by
  intros r h
  rw [h, edge_length]
  rfl

end NUMINAMATH_GPT_cube_edge_length_l2050_205025


namespace NUMINAMATH_GPT_divide_80_into_two_parts_l2050_205052

theorem divide_80_into_two_parts :
  ∃ a b : ℕ, a + b = 80 ∧ b / 2 = a + 10 ∧ a = 20 ∧ b = 60 :=
by
  sorry

end NUMINAMATH_GPT_divide_80_into_two_parts_l2050_205052


namespace NUMINAMATH_GPT_num_aluminum_cans_l2050_205024

def num_glass_bottles : ℕ := 10
def total_litter : ℕ := 18

theorem num_aluminum_cans : total_litter - num_glass_bottles = 8 :=
by
  sorry

end NUMINAMATH_GPT_num_aluminum_cans_l2050_205024


namespace NUMINAMATH_GPT_abs_diff_one_l2050_205039

theorem abs_diff_one (a b : ℤ) (h : |a| + |b| = 1) : |a - b| = 1 := sorry

end NUMINAMATH_GPT_abs_diff_one_l2050_205039


namespace NUMINAMATH_GPT_possible_values_of_f2001_l2050_205034

noncomputable def f : ℕ → ℝ := sorry

theorem possible_values_of_f2001 (f : ℕ → ℝ)
    (H : ∀ a b : ℕ, a > 1 → b > 1 → ∀ d : ℕ, d = Nat.gcd a b → 
           f (a * b) = f d * (f (a / d) + f (b / d))) :
    f 2001 = 0 ∨ f 2001 = 1/2 :=
sorry

end NUMINAMATH_GPT_possible_values_of_f2001_l2050_205034


namespace NUMINAMATH_GPT_intersection_A_B_range_m_l2050_205042

-- Define set A when m = 3 as given
def A_set (x : ℝ) : Prop := 3 - 2 * x - x^2 ≥ 0
def A (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1

-- Define set B when m = 3 as given
def B_set (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0
def B (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 4

-- The intersection of A and B should be: -2 ≤ x ≤ 1
theorem intersection_A_B : ∀ (x : ℝ), A x ∧ B x ↔ (-2 ≤ x ∧ x ≤ 1) := sorry

-- Define A for general m > 0
def A_set_general (x : ℝ) : Prop := 3 - 2 * x - x^2 ≥ 0

-- Define B for general m
def B_set_general (x : ℝ) (m : ℝ) : Prop := (x - 1)^2 ≤ m^2

-- Prove the range for m such that A ⊆ B
theorem range_m (m : ℝ) (h : m > 0) : (∀ x, A_set_general x → B_set_general x m) ↔ m ≥ 4 := sorry

end NUMINAMATH_GPT_intersection_A_B_range_m_l2050_205042


namespace NUMINAMATH_GPT_scientific_notation_l2050_205090

def billion : ℝ := 10^9
def fifteenPointSeventyFiveBillion : ℝ := 15.75 * billion

theorem scientific_notation :
  fifteenPointSeventyFiveBillion = 1.575 * 10^10 :=
  sorry

end NUMINAMATH_GPT_scientific_notation_l2050_205090


namespace NUMINAMATH_GPT_difference_of_squares_l2050_205033

theorem difference_of_squares 
  (x y : ℝ) 
  (optionA := (-x + y) * (x + y))
  (optionB := (-x + y) * (x - y))
  (optionC := (x + 2) * (2 + x))
  (optionD := (2 * x + 3) * (3 * x - 2)) :
  optionA = -(x + y)^2 ∨ optionA = (x + y) * (y - x) :=
sorry

end NUMINAMATH_GPT_difference_of_squares_l2050_205033


namespace NUMINAMATH_GPT_sum_of_squares_diagonals_cyclic_quadrilateral_l2050_205011

theorem sum_of_squares_diagonals_cyclic_quadrilateral 
(a b c d : ℝ) (α : ℝ) 
(hc : c^2 = a^2 + b^2 + 2 * a * b * Real.cos α)
(hd : d^2 = a^2 + b^2 - 2 * a * b * Real.cos α) :
  c^2 + d^2 = 2 * a^2 + 2 * b^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_diagonals_cyclic_quadrilateral_l2050_205011


namespace NUMINAMATH_GPT_free_endpoints_can_be_1001_l2050_205019

variables (initial_segs : ℕ) (total_free_ends : ℕ) (k : ℕ)

-- Initial setup: one initial segment.
def initial_segment : ℕ := 1

-- Each time 5 segments are drawn from a point, the number of free ends increases by 4.
def free_ends_after_k_actions (k : ℕ) : ℕ := initial_segment + 4 * k

-- Question: Can the number of free endpoints be exactly 1001?
theorem free_endpoints_can_be_1001 : free_ends_after_k_actions 250 = 1001 := by
  sorry

end NUMINAMATH_GPT_free_endpoints_can_be_1001_l2050_205019


namespace NUMINAMATH_GPT_triangle_third_side_possibilities_l2050_205080

theorem triangle_third_side_possibilities (x : ℕ) : 
  (6 + 8 > x) ∧ (x + 6 > 8) ∧ (x + 8 > 6) → 
  3 ≤ x ∧ x < 14 → 
  ∃ n, n = 11 :=
by
  sorry

end NUMINAMATH_GPT_triangle_third_side_possibilities_l2050_205080


namespace NUMINAMATH_GPT_bananas_to_oranges_l2050_205095

variables (banana apple orange : Type) 
variables (cost_banana : banana → ℕ) 
variables (cost_apple : apple → ℕ)
variables (cost_orange : orange → ℕ)

-- Conditions given in the problem
axiom cond1 : ∀ (b1 b2 b3 : banana) (a1 a2 : apple), cost_banana b1 = cost_banana b2 → cost_banana b2 = cost_banana b3 → 3 * cost_banana b1 = 2 * cost_apple a1
axiom cond2 : ∀ (a3 a4 a5 a6 : apple) (o1 o2 : orange), cost_apple a3 = cost_apple a4 → cost_apple a4 = cost_apple a5 → cost_apple a5 = cost_apple a6 → 6 * cost_apple a3 = 4 * cost_orange o1

-- Prove that 8 oranges cost as much as 18 bananas
theorem bananas_to_oranges (b1 b2 b3 : banana) (a1 a2 a3 a4 a5 a6 : apple) (o1 o2 : orange) :
    3 * cost_banana b1 = 2 * cost_apple a1 →
    6 * cost_apple a3 = 4 * cost_orange o1 →
    18 * cost_banana b1 = 8 * cost_orange o2 := 
sorry

end NUMINAMATH_GPT_bananas_to_oranges_l2050_205095


namespace NUMINAMATH_GPT_Jason_reroll_exactly_two_dice_probability_l2050_205086

noncomputable def probability_reroll_two_dice : ℚ :=
  let favorable_outcomes := 5 * 3 + 1 * 3 + 5 * 3
  let total_possibilities := 6^3
  favorable_outcomes / total_possibilities

theorem Jason_reroll_exactly_two_dice_probability : probability_reroll_two_dice = 5 / 9 := 
  sorry

end NUMINAMATH_GPT_Jason_reroll_exactly_two_dice_probability_l2050_205086


namespace NUMINAMATH_GPT_total_fruits_purchased_l2050_205058

-- Defining the costs of apples and bananas
def cost_per_apple : ℝ := 0.80
def cost_per_banana : ℝ := 0.70

-- Defining the total cost the customer spent
def total_cost : ℝ := 6.50

-- Defining the total number of fruits purchased as 9
theorem total_fruits_purchased (A B : ℕ) : 
  (cost_per_apple * A + cost_per_banana * B = total_cost) → 
  (A + B = 9) :=
by
  sorry

end NUMINAMATH_GPT_total_fruits_purchased_l2050_205058


namespace NUMINAMATH_GPT_bananas_oranges_equivalence_l2050_205097

theorem bananas_oranges_equivalence :
  (3 / 4) * 12 * banana_value = 9 * orange_value →
  (2 / 3) * 6 * banana_value = 4 * orange_value :=
by
  intros h
  sorry

end NUMINAMATH_GPT_bananas_oranges_equivalence_l2050_205097


namespace NUMINAMATH_GPT_find_M_plus_N_l2050_205029

theorem find_M_plus_N (M N : ℕ) (h1 : (3:ℚ) / 5 = M / 45) (h2 : (3:ℚ) / 5 = 60 / N) : M + N = 127 :=
sorry

end NUMINAMATH_GPT_find_M_plus_N_l2050_205029


namespace NUMINAMATH_GPT_pizza_area_increase_l2050_205077

theorem pizza_area_increase 
  (r : ℝ) 
  (A_medium A_large : ℝ) 
  (h_medium_area : A_medium = Real.pi * r^2)
  (h_large_area : A_large = Real.pi * (1.40 * r)^2) : 
  ((A_large - A_medium) / A_medium) * 100 = 96 := 
by 
  sorry

end NUMINAMATH_GPT_pizza_area_increase_l2050_205077


namespace NUMINAMATH_GPT_new_person_weight_l2050_205020

theorem new_person_weight (w : ℝ) (avg_increase : ℝ) (replaced_person_weight : ℝ) (num_people : ℕ) 
(H1 : avg_increase = 4.8) (H2 : replaced_person_weight = 62) (H3 : num_people = 12) : 
w = 119.6 :=
by
  -- We could provide the intermediate steps as definitions here but for the theorem statement, we just present the goal.
  sorry

end NUMINAMATH_GPT_new_person_weight_l2050_205020


namespace NUMINAMATH_GPT_number_of_animal_books_l2050_205073

variable (A : ℕ)

theorem number_of_animal_books (h1 : 6 * 6 + 3 * 6 + A * 6 = 102) : A = 8 :=
sorry

end NUMINAMATH_GPT_number_of_animal_books_l2050_205073


namespace NUMINAMATH_GPT_simplify_fraction_product_l2050_205005

theorem simplify_fraction_product :
  (2 / 3) * (4 / 7) * (9 / 13) = 24 / 91 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_product_l2050_205005


namespace NUMINAMATH_GPT_f_25_over_11_neg_l2050_205051

variable (f : ℚ → ℚ)
axiom f_mul : ∀ a b : ℚ, a > 0 → b > 0 → f (a * b) = f a + f b
axiom f_prime : ∀ p : ℕ, Prime p → f p = p

theorem f_25_over_11_neg : f (25 / 11) < 0 :=
by
  -- You can prove the necessary steps during interactive proof:
  -- Using primes 25 = 5^2 and 11 itself,
  -- f (25/11) = f 25 - f 11. Thus, f (25) = 2f(5) = 2 * 5 = 10 and f(11) = 11
  -- Therefore, f (25/11) = 10 - 11 = -1 < 0.
  sorry

end NUMINAMATH_GPT_f_25_over_11_neg_l2050_205051
