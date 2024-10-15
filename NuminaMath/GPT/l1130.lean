import Mathlib

namespace NUMINAMATH_GPT_exists_sum_of_two_squares_l1130_113055

theorem exists_sum_of_two_squares (n : ℕ) (h₁ : n > 10000) : 
  ∃ m : ℕ, (∃ a b : ℕ, m = a^2 + b^2) ∧ 0 < m - n ∧ m - n < 3 * Real.sqrt n := 
sorry

end NUMINAMATH_GPT_exists_sum_of_two_squares_l1130_113055


namespace NUMINAMATH_GPT_minimum_w_value_l1130_113005

theorem minimum_w_value : 
  (∀ x y : ℝ, w = 2*x^2 + 3*y^2 - 12*x + 9*y + 35) → 
  ∃ w_min : ℝ, w_min = 41 / 4 ∧ 
  (∀ x y : ℝ, 2*x^2 + 3*y^2 - 12*x + 9*y + 35 ≥ w_min) :=
by
  sorry

end NUMINAMATH_GPT_minimum_w_value_l1130_113005


namespace NUMINAMATH_GPT_delivery_time_is_40_minutes_l1130_113099

-- Define the conditions
def total_pizzas : Nat := 12
def two_pizza_stops : Nat := 2
def pizzas_per_stop_with_two_pizzas : Nat := 2
def time_per_stop_minutes : Nat := 4

-- Define the number of pizzas covered by stops with two pizzas
def pizzas_covered_by_two_pizza_stops : Nat := two_pizza_stops * pizzas_per_stop_with_two_pizzas

-- Define the number of single pizza stops
def single_pizza_stops : Nat := total_pizzas - pizzas_covered_by_two_pizza_stops

-- Define the total number of stops
def total_stops : Nat := two_pizza_stops + single_pizza_stops

-- Total time to deliver all pizzas
def total_delivery_time_minutes : Nat := total_stops * time_per_stop_minutes

theorem delivery_time_is_40_minutes : total_delivery_time_minutes = 40 := by
  sorry

end NUMINAMATH_GPT_delivery_time_is_40_minutes_l1130_113099


namespace NUMINAMATH_GPT_students_with_both_l1130_113094

/-- There are 28 students in a class -/
def total_students : ℕ := 28

/-- Number of students with a cat -/
def students_with_cat : ℕ := 17

/-- Number of students with a dog -/
def students_with_dog : ℕ := 10

/-- Number of students with neither a cat nor a dog -/
def students_with_neither : ℕ := 5

/-- Number of students having both a cat and a dog -/
theorem students_with_both :
  students_with_cat + students_with_dog - (total_students - students_with_neither) = 4 :=
sorry

end NUMINAMATH_GPT_students_with_both_l1130_113094


namespace NUMINAMATH_GPT_sodas_purchasable_l1130_113011

namespace SodaPurchase

variable {D C : ℕ}

theorem sodas_purchasable (D C : ℕ) : (3 * (4 * D) / 5 + 5 * C / 15) = (36 * D + 5 * C) / 15 := 
  sorry

end SodaPurchase

end NUMINAMATH_GPT_sodas_purchasable_l1130_113011


namespace NUMINAMATH_GPT_no_member_of_T_is_divisible_by_4_l1130_113010

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

theorem no_member_of_T_is_divisible_by_4 : ∀ n : ℤ, ¬ (sum_of_squares_of_four_consecutive_integers n % 4 = 0) := by
  intro n
  sorry

end NUMINAMATH_GPT_no_member_of_T_is_divisible_by_4_l1130_113010


namespace NUMINAMATH_GPT_sum_of_coefficients_l1130_113036

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℕ) (h₁ : (1 + x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) : 
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1130_113036


namespace NUMINAMATH_GPT_rod_mass_equilibrium_l1130_113054

variable (g : ℝ) (m1 : ℝ) (l : ℝ) (S : ℝ)

-- Given conditions
axiom m1_value : m1 = 1
axiom l_value  : l = 0.5
axiom S_value  : S = 0.1

-- The goal is to find m2 such that the equilibrium condition holds
theorem rod_mass_equilibrium (m2 : ℝ) :
  (m1 * S = m2 * l) → m2 = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_rod_mass_equilibrium_l1130_113054


namespace NUMINAMATH_GPT_parabola_vertex_example_l1130_113090

-- Definitions based on conditions
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def vertex (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + 3

-- Conditions given in the problem
def condition1 (a b c : ℝ) : Prop := parabola a b c 2 = 5
def condition2 (a : ℝ) : Prop := vertex a 1 = 3

-- Goal statement to be proved
theorem parabola_vertex_example : ∃ (a b c : ℝ), 
  condition1 a b c ∧ condition2 a ∧ a - b + c = 11 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_example_l1130_113090


namespace NUMINAMATH_GPT_system_of_equations_solution_l1130_113000

theorem system_of_equations_solution (x y : ℚ) :
  (x / 3 + y / 4 = 4 ∧ 2 * x - 3 * y = 12) → (x = 10 ∧ y = 8 / 3) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1130_113000


namespace NUMINAMATH_GPT_player2_winning_strategy_l1130_113007

-- Definitions of the game setup
def initial_position_player1 := (1, 1)
def initial_position_player2 := (998, 1998)

def adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 - 1 ∨ p1.2 = p2.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 - 1 ∨ p1.1 = p2.1 + 1))

-- A function defining the winning condition for Player 2
def player2_wins (p1 p2 : ℕ × ℕ) : Prop :=
  p1 = p2 ∨ p1.1 = (initial_position_player2.1)

-- Theorem stating the pair (998, 1998) guarantees a win for Player 2
theorem player2_winning_strategy : player2_wins (998, 0) (998, 1998) :=
sorry

end NUMINAMATH_GPT_player2_winning_strategy_l1130_113007


namespace NUMINAMATH_GPT_base7_subtraction_correct_l1130_113053

-- Define a function converting base 7 number to base 10
def base7_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n2 := n1 / 10
  let d2 := n2 % 10
  let n3 := n2 / 10
  let d3 := n3 % 10
  d3 * 7^3 + d2 * 7^2 + d1 * 7^1 + d0 * 7^0

-- Define the numbers in base 7
def a : Nat := 2456
def b : Nat := 1234

-- Define the expected result in base 7
def result_base7 : Nat := 1222

-- State the theorem: The difference of a and b in base 7 should equal result_base7
theorem base7_subtraction_correct :
  let diff_base10 := (base7_to_base10 a) - (base7_to_base10 b)
  let result_base10 := base7_to_base10 result_base7
  diff_base10 = result_base10 :=
by
  sorry

end NUMINAMATH_GPT_base7_subtraction_correct_l1130_113053


namespace NUMINAMATH_GPT_mass_percentage_Al_in_AlI3_l1130_113075

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_AlI3 : ℝ := molar_mass_Al + 3 * molar_mass_I

theorem mass_percentage_Al_in_AlI3 : 
  (molar_mass_Al / molar_mass_AlI3) * 100 = 6.62 := 
  sorry

end NUMINAMATH_GPT_mass_percentage_Al_in_AlI3_l1130_113075


namespace NUMINAMATH_GPT_fraction_juniors_study_Japanese_l1130_113001

-- Define the size of the junior and senior classes
variable (J S : ℕ)

-- Condition 1: The senior class is twice the size of the junior class
axiom senior_twice_junior : S = 2 * J

-- The fraction of the seniors studying Japanese
noncomputable def fraction_seniors_study_Japanese : ℚ := 3 / 8

-- The total fraction of students in both classes that study Japanese
noncomputable def fraction_total_study_Japanese : ℚ := 1 / 3

-- Define the unknown fraction of juniors studying Japanese
variable (x : ℚ)

-- The proof problem transformed from the questions and the correct answer
theorem fraction_juniors_study_Japanese :
  (fraction_seniors_study_Japanese * ↑S + x * ↑J = fraction_total_study_Japanese * (↑J + ↑S)) → (x = 1 / 4) :=
by
  -- We use the given conditions and solve for x
  sorry

end NUMINAMATH_GPT_fraction_juniors_study_Japanese_l1130_113001


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1130_113050

theorem quadratic_two_distinct_real_roots:
  ∃ (α β : ℝ), α ≠ β ∧ (∀ x : ℝ, x * (x - 2) = x - 2 ↔ x = α ∨ x = β) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1130_113050


namespace NUMINAMATH_GPT_negation_of_existence_l1130_113064

theorem negation_of_existence (x : ℝ) (hx : 0 < x) : ¬ (∃ x_0 : ℝ, 0 < x_0 ∧ Real.log x_0 = x_0 - 1) 
  → ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by sorry

end NUMINAMATH_GPT_negation_of_existence_l1130_113064


namespace NUMINAMATH_GPT_sqrt_eq_cond_l1130_113033

theorem sqrt_eq_cond (a b c : ℕ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
    (not_perfect_square_a : ¬(∃ n : ℕ, n * n = a)) (not_perfect_square_b : ¬(∃ n : ℕ, n * n = b))
    (not_perfect_square_c : ¬(∃ n : ℕ, n * n = c)) :
    (Real.sqrt a + Real.sqrt b = Real.sqrt c) →
    (2 * Real.sqrt (a * b) = c - (a + b) ∧ (∃ k : ℕ, a * b = k * k)) :=
sorry

end NUMINAMATH_GPT_sqrt_eq_cond_l1130_113033


namespace NUMINAMATH_GPT_minimal_APR_bank_A_l1130_113068

def nominal_interest_rate_A : Float := 0.05
def nominal_interest_rate_B : Float := 0.055
def nominal_interest_rate_C : Float := 0.06

def compounding_periods_A : ℕ := 4
def compounding_periods_B : ℕ := 2
def compounding_periods_C : ℕ := 12

def effective_annual_rate (nom_rate : Float) (n : ℕ) : Float :=
  (1 + nom_rate / n.toFloat)^n.toFloat - 1

def APR_A := effective_annual_rate nominal_interest_rate_A compounding_periods_A
def APR_B := effective_annual_rate nominal_interest_rate_B compounding_periods_B
def APR_C := effective_annual_rate nominal_interest_rate_C compounding_periods_C

theorem minimal_APR_bank_A :
  APR_A < APR_B ∧ APR_A < APR_C ∧ APR_A = 0.050945 :=
by
  sorry

end NUMINAMATH_GPT_minimal_APR_bank_A_l1130_113068


namespace NUMINAMATH_GPT_restaurant_hamburgers_l1130_113095

-- Define the conditions
def hamburgers_served : ℕ := 3
def hamburgers_left_over : ℕ := 6

-- Define the total hamburgers made
def hamburgers_made : ℕ := hamburgers_served + hamburgers_left_over

-- State and prove the theorem
theorem restaurant_hamburgers : hamburgers_made = 9 := by
  sorry

end NUMINAMATH_GPT_restaurant_hamburgers_l1130_113095


namespace NUMINAMATH_GPT_find_a_range_l1130_113071

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 1 then (a + 3) * x - 5 else 2 * a / x

theorem find_a_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2) → -2 ≤ a ∧ a < 0 :=
by
  sorry

end NUMINAMATH_GPT_find_a_range_l1130_113071


namespace NUMINAMATH_GPT_solution_l1130_113066

def problem_statement : Prop :=
  (3025 - 2880) ^ 2 / 225 = 93

theorem solution : problem_statement :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_l1130_113066


namespace NUMINAMATH_GPT_max_stickers_one_student_l1130_113046

def total_students : ℕ := 25
def mean_stickers : ℕ := 4
def total_stickers := total_students * mean_stickers
def minimum_stickers_per_student : ℕ := 1
def minimum_stickers_taken_by_24_students := (total_students - 1) * minimum_stickers_per_student

theorem max_stickers_one_student : 
  total_stickers - minimum_stickers_taken_by_24_students = 76 := by
  sorry

end NUMINAMATH_GPT_max_stickers_one_student_l1130_113046


namespace NUMINAMATH_GPT_solve_quad_eq1_solve_quad_eq2_solve_quad_eq3_solve_quad_eq4_l1130_113048

-- Problem 1: Prove the solutions to x^2 = 2
theorem solve_quad_eq1 : ∃ x : ℝ, x^2 = 2 ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) :=
by
  sorry

-- Problem 2: Prove the solutions to 4x^2 - 1 = 0
theorem solve_quad_eq2 : ∃ x : ℝ, 4 * x^2 - 1 = 0 ∧ (x = 1/2 ∨ x = -1/2) :=
by
  sorry

-- Problem 3: Prove the solutions to (x-1)^2 - 4 = 0
theorem solve_quad_eq3 : ∃ x : ℝ, (x - 1)^2 - 4 = 0 ∧ (x = 3 ∨ x = -1) :=
by
  sorry

-- Problem 4: Prove the solutions to 12 * (3 - x)^2 - 48 = 0
theorem solve_quad_eq4 : ∃ x : ℝ, 12 * (3 - x)^2 - 48 = 0 ∧ (x = 1 ∨ x = 5) :=
by
  sorry

end NUMINAMATH_GPT_solve_quad_eq1_solve_quad_eq2_solve_quad_eq3_solve_quad_eq4_l1130_113048


namespace NUMINAMATH_GPT_find_a_g_range_l1130_113028

noncomputable def f (x a : ℝ) : ℝ := x^2 + 4 * a * x + 2 * a + 6
noncomputable def g (a : ℝ) : ℝ := 2 - a * |a - 1|

theorem find_a (x a : ℝ) :
  (∀ x, f x a ≥ 0) ∧ (∀ x, f x a = 0 → x^2 + 4 * a * x + 2 * a + 6 = 0) ↔ (a = -1 ∨ a = 3 / 2) :=
  sorry

theorem g_range :
  (∀ x, f x a ≥ 0) ∧ (-1 ≤ a ∧ a ≤ 3/2) → (∀ a, (5 / 4 ≤ g a ∧ g a ≤ 4)) :=
  sorry

end NUMINAMATH_GPT_find_a_g_range_l1130_113028


namespace NUMINAMATH_GPT_max_value_expression_l1130_113057

variable (x y z : ℝ)

theorem max_value_expression (h : x^2 + y^2 + z^2 = 4) :
  (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2 ≤ 28 :=
sorry

end NUMINAMATH_GPT_max_value_expression_l1130_113057


namespace NUMINAMATH_GPT_infinite_sum_equals_l1130_113043

theorem infinite_sum_equals :
  10 * (79 * (1 / 7)) + (∑' n : ℕ, if n % 2 = 0 then (if n = 0 then 0 else 2 / 7 ^ n) else (1 / 7 ^ n)) = 3 / 16 :=
by
  sorry

end NUMINAMATH_GPT_infinite_sum_equals_l1130_113043


namespace NUMINAMATH_GPT_ratio_of_routes_l1130_113030

-- Definitions of m and n
def m : ℕ := 2 
def n : ℕ := 6

-- Theorem statement
theorem ratio_of_routes (m_positive : m > 0) : n / m = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_routes_l1130_113030


namespace NUMINAMATH_GPT_inequality_gt_zero_l1130_113023

theorem inequality_gt_zero (x y : ℝ) : x^2 + 2*y^2 + 2*x*y + 6*y + 10 > 0 :=
  sorry

end NUMINAMATH_GPT_inequality_gt_zero_l1130_113023


namespace NUMINAMATH_GPT_sum_of_digits_of_triangular_number_2010_l1130_113077

noncomputable def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_triangular_number_2010 (N : ℕ)
  (h₁ : triangular_number N = 2010) :
  sum_of_digits N = 9 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_of_triangular_number_2010_l1130_113077


namespace NUMINAMATH_GPT_cos_double_angle_l1130_113061

theorem cos_double_angle (α : ℝ) (h : Real.cos (π - α) = -3/5) : Real.cos (2 * α) = -7/25 :=
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1130_113061


namespace NUMINAMATH_GPT_solve_x_squared_eq_four_l1130_113080

theorem solve_x_squared_eq_four (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 := 
by sorry

end NUMINAMATH_GPT_solve_x_squared_eq_four_l1130_113080


namespace NUMINAMATH_GPT_inequality_solution_l1130_113079

theorem inequality_solution (x : ℝ) :
  (7 / 36 + (abs (2 * x - (1 / 6)))^2 < 5 / 12) ↔
  (x ∈ Set.Ioo ((1 / 12 - (Real.sqrt 2 / 6))) ((1 / 12 + (Real.sqrt 2 / 6)))) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1130_113079


namespace NUMINAMATH_GPT_random_variable_prob_l1130_113047

theorem random_variable_prob (n : ℕ) (h : (3 : ℝ) / n = 0.3) : n = 10 :=
sorry

end NUMINAMATH_GPT_random_variable_prob_l1130_113047


namespace NUMINAMATH_GPT_required_equation_l1130_113062

-- Define the given lines
def line1 (x y : ℝ) : Prop := 2 * x - y = 0
def line2 (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the equation to be proven for the line through the intersection point and perpendicular to perp_line
def required_line (x y : ℝ) : Prop := x - 2 * y + 6 = 0

-- Define the predicate that states a point (2, 4) lies on line1 and line2
def point_intersect (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- The main theorem to be proven in Lean 4
theorem required_equation : 
  point_intersect 2 4 ∧ perp_line 2 4 → required_line 2 4 := by
  sorry

end NUMINAMATH_GPT_required_equation_l1130_113062


namespace NUMINAMATH_GPT_total_carriages_in_towns_l1130_113002

noncomputable def total_carriages (euston norfolk norwich flyingScotsman victoria waterloo : ℕ) : ℕ :=
  euston + norfolk + norwich + flyingScotsman + victoria + waterloo

theorem total_carriages_in_towns :
  let euston := 130
  let norfolk := euston - (20 * euston / 100)
  let norwich := 100
  let flyingScotsman := 3 * norwich / 2
  let victoria := euston - (15 * euston / 100)
  let waterloo := 2 * norwich
  total_carriages euston norfolk norwich flyingScotsman victoria waterloo = 794 :=
by
  sorry

end NUMINAMATH_GPT_total_carriages_in_towns_l1130_113002


namespace NUMINAMATH_GPT_find_A_in_terms_of_B_and_C_l1130_113045

noncomputable def f (A B : ℝ) (x : ℝ) := A * x - 3 * B^2
noncomputable def g (B C : ℝ) (x : ℝ) := B * x + C

theorem find_A_in_terms_of_B_and_C (A B C : ℝ) (h : B ≠ 0) (h1 : f A B (g B C 1) = 0) : A = 3 * B^2 / (B + C) :=
by sorry

end NUMINAMATH_GPT_find_A_in_terms_of_B_and_C_l1130_113045


namespace NUMINAMATH_GPT_value_of_g_neg3_l1130_113056

def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem value_of_g_neg3 : g (-3) = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_g_neg3_l1130_113056


namespace NUMINAMATH_GPT_find_line_eq_l1130_113051

noncomputable def line_perpendicular (p : ℝ × ℝ) (a b c: ℝ) : Prop :=
  ∃ (m: ℝ) (k: ℝ), k ≠ 0 ∧ (b * m = -a) ∧ p = (m, (c - a * m) / b) ∧
  (∀ x y : ℝ, y = m * x + ((c - a * m) / b) ↔ b * y = -a * x - c)

theorem find_line_eq (p : ℝ × ℝ) (a b c : ℝ) (p_eq : p = (-3, 0)) (perpendicular_eq : a = 2 ∧ b = -1 ∧ c = 3) :
  ∃ (m k : ℝ), (k ≠ 0 ∧ (-1 * (b / a)) = m ∧ line_perpendicular p a b c) ∧ (b * m = -a) ∧ ((k = (-a * m) / b) ∧ (b * k * 0 - (-a * 3)) = c) := sorry

end NUMINAMATH_GPT_find_line_eq_l1130_113051


namespace NUMINAMATH_GPT_original_days_to_finish_work_l1130_113009

theorem original_days_to_finish_work : 
  ∀ (D : ℕ), 
  (∃ (W : ℕ), 15 * D * W = 25 * (D - 3) * W) → 
  D = 8 :=
by
  intros D h
  sorry

end NUMINAMATH_GPT_original_days_to_finish_work_l1130_113009


namespace NUMINAMATH_GPT_black_squares_in_45th_row_l1130_113093

-- Definitions based on the conditions
def number_of_squares_in_row (n : ℕ) : ℕ := 2 * n + 1

def number_of_black_squares (total_squares : ℕ) : ℕ := (total_squares - 1) / 2

-- The theorem statement
theorem black_squares_in_45th_row : number_of_black_squares (number_of_squares_in_row 45) = 45 :=
by sorry

end NUMINAMATH_GPT_black_squares_in_45th_row_l1130_113093


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l1130_113041

/-- Given:
  * 1, a₁, a₂, 4 form an arithmetic sequence
  * 1, b₁, b₂, b₃, 4 form a geometric sequence
Prove that:
  (a₁ + a₂) / b₂ = 5 / 2
-/
theorem arithmetic_geometric_sequence (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (h_arith : 2 * a₁ = 1 + a₂ ∧ 2 * a₂ = a₁ + 4)
  (h_geom : b₁ * b₁ = b₂ ∧ b₁ * b₂ = b₃ ∧ b₂ * b₂ = b₃ * 4) :
  (a₁ + a₂) / b₂ = 5 / 2 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l1130_113041


namespace NUMINAMATH_GPT_product_floor_ceil_sequence_l1130_113013

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem product_floor_ceil_sequence :
    (floor (-6 - 0.5) * ceil (6 + 0.5)) *
    (floor (-5 - 0.5) * ceil (5 + 0.5)) *
    (floor (-4 - 0.5) * ceil (4 + 0.5)) *
    (floor (-3 - 0.5) * ceil (3 + 0.5)) *
    (floor (-2 - 0.5) * ceil (2 + 0.5)) *
    (floor (-1 - 0.5) * ceil (1 + 0.5)) *
    (floor (-0.5) * ceil (0.5)) = -25401600 :=
by
  sorry

end NUMINAMATH_GPT_product_floor_ceil_sequence_l1130_113013


namespace NUMINAMATH_GPT_parabola_focus_distance_l1130_113084

theorem parabola_focus_distance (p : ℝ) (h : 2 * p = 8) : p = 4 :=
  by
  sorry

end NUMINAMATH_GPT_parabola_focus_distance_l1130_113084


namespace NUMINAMATH_GPT_gas_total_cost_l1130_113018

theorem gas_total_cost (x : ℝ) (h : (x/3) - 11 = x/5) : x = 82.5 :=
sorry

end NUMINAMATH_GPT_gas_total_cost_l1130_113018


namespace NUMINAMATH_GPT_intersection_complement_correct_l1130_113016

open Set

def A : Set ℕ := {x | (x + 4) * (x - 5) ≤ 0}
def B : Set ℕ := {x | x < 2}
def U : Set ℕ := { x | True }

theorem intersection_complement_correct :
  (A ∩ (U \ B)) = {x | x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_correct_l1130_113016


namespace NUMINAMATH_GPT_determine_n_l1130_113021

-- All the terms used in the conditions
variables (S C M : ℝ)
variables (n : ℝ)

-- Define the conditions as hypotheses
def condition1 := M = 1 / 3 * S
def condition2 := M = 1 / n * C

-- The main theorem statement
theorem determine_n (S C M : ℝ) (n : ℝ) (h1 : condition1 S M) (h2 : condition2 M n C) : n = 2 :=
by sorry

end NUMINAMATH_GPT_determine_n_l1130_113021


namespace NUMINAMATH_GPT_range_of_m_l1130_113004

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x - m)/2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2)) →
  ∃ x : ℝ, x = 2 ∧ -3 < m ∧ m ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1130_113004


namespace NUMINAMATH_GPT_min_odd_integers_l1130_113039

theorem min_odd_integers :
  ∀ (a b c d e f g h : ℤ),
  a + b + c = 30 →
  a + b + c + d + e + f = 58 →
  a + b + c + d + e + f + g + h = 73 →
  ∃ (odd_count : ℕ), odd_count = 1 :=
by
  sorry

end NUMINAMATH_GPT_min_odd_integers_l1130_113039


namespace NUMINAMATH_GPT_larger_gate_width_is_10_l1130_113015

-- Define the conditions as constants
def garden_length : ℝ := 225
def garden_width : ℝ := 125
def small_gate_width : ℝ := 3
def total_fencing_length : ℝ := 687

-- Define the perimeter function for a rectangle
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

-- Define the width of the larger gate
def large_gate_width : ℝ :=
  let total_perimeter := perimeter garden_length garden_width
  let remaining_fencing := total_perimeter - total_fencing_length
  remaining_fencing - small_gate_width

-- State the theorem
theorem larger_gate_width_is_10 : large_gate_width = 10 := by
  -- skipping proof part
  sorry

end NUMINAMATH_GPT_larger_gate_width_is_10_l1130_113015


namespace NUMINAMATH_GPT_smallest_number_l1130_113089

theorem smallest_number (n : ℕ) :
  (n % 3 = 1) ∧
  (n % 5 = 3) ∧
  (n % 6 = 4) →
  n = 28 :=
sorry

end NUMINAMATH_GPT_smallest_number_l1130_113089


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1130_113038

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem solve_quadratic_eq :
  ∀ a b c x1 x2 : ℝ,
  a = 2 →
  b = -2 →
  c = -1 →
  quadratic_eq a b c x1 ∧ quadratic_eq a b c x2 →
  (x1 = (1 + Real.sqrt 3) / 2 ∧ x2 = (1 - Real.sqrt 3) / 2) :=
by
  intros a b c x1 x2 ha hb hc h
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1130_113038


namespace NUMINAMATH_GPT_correct_figure_is_D_l1130_113083

def option_A : Prop := sorry -- placeholder for option A as a diagram representation
def option_B : Prop := sorry -- placeholder for option B as a diagram representation
def option_C : Prop := sorry -- placeholder for option C as a diagram representation
def option_D : Prop := sorry -- placeholder for option D as a diagram representation
def equilateral_triangle (figure : Prop) : Prop := sorry -- placeholder for the condition representing an equilateral triangle in the oblique projection method

theorem correct_figure_is_D : equilateral_triangle option_D := 
sorry

end NUMINAMATH_GPT_correct_figure_is_D_l1130_113083


namespace NUMINAMATH_GPT_problem_ACD_l1130_113019

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 - x

theorem problem_ACD (a : ℝ) :
  (f a 0 = (2/3) ∧
  ¬(∀ x, f a x ≥ 0 → ((a ≥ 1) ∨ (a ≤ -1))) ∧
  (∃ x1 x2, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ∧
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0)) :=
sorry

end NUMINAMATH_GPT_problem_ACD_l1130_113019


namespace NUMINAMATH_GPT_adam_final_amount_l1130_113073

def initial_savings : ℝ := 1579.37
def money_received_monday : ℝ := 21.85
def money_received_tuesday : ℝ := 33.28
def money_spent_wednesday : ℝ := 87.41

def total_money_received : ℝ := money_received_monday + money_received_tuesday
def new_total_after_receiving : ℝ := initial_savings + total_money_received
def final_amount : ℝ := new_total_after_receiving - money_spent_wednesday

theorem adam_final_amount : final_amount = 1547.09 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_adam_final_amount_l1130_113073


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1130_113034

variable (a : ℕ → ℤ) -- defining the sequence {a_n}
variable (S : ℕ → ℤ) -- defining the sum of the first n terms S_n

theorem arithmetic_sequence_problem (m : ℕ) (h1 : m > 1) 
  (h2 : a (m - 1) + a (m + 1) - a m ^ 2 = 0) 
  (h3 : S (2 * m - 1) = 38) 
  : m = 10 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1130_113034


namespace NUMINAMATH_GPT_totalCostOfAllPuppies_l1130_113069

noncomputable def goldenRetrieverCost : ℕ :=
  let numberOfGoldenRetrievers := 3
  let puppiesPerGoldenRetriever := 4
  let shotsPerPuppy := 2
  let costPerShot := 5
  let vitaminCostPerMonth := 12
  let monthsOfSupplements := 6
  numberOfGoldenRetrievers * puppiesPerGoldenRetriever *
  (shotsPerPuppy * costPerShot + vitaminCostPerMonth * monthsOfSupplements)

noncomputable def germanShepherdCost : ℕ :=
  let numberOfGermanShepherds := 2
  let puppiesPerGermanShepherd := 5
  let shotsPerPuppy := 3
  let costPerShot := 8
  let microchipCost := 25
  let toyCost := 15
  numberOfGermanShepherds * puppiesPerGermanShepherd *
  (shotsPerPuppy * costPerShot + microchipCost + toyCost)

noncomputable def bulldogCost : ℕ :=
  let numberOfBulldogs := 4
  let puppiesPerBulldog := 3
  let shotsPerPuppy := 4
  let costPerShot := 10
  let collarCost := 20
  let chewToyCost := 18
  numberOfBulldogs * puppiesPerBulldog *
  (shotsPerPuppy * costPerShot + collarCost + chewToyCost)

theorem totalCostOfAllPuppies : goldenRetrieverCost + germanShepherdCost + bulldogCost = 2560 :=
by
  sorry

end NUMINAMATH_GPT_totalCostOfAllPuppies_l1130_113069


namespace NUMINAMATH_GPT_prob1_prob2_prob3_prob4_prob5_l1130_113040

theorem prob1 : (1 - 27 + (-32) + (-8) + 27) = -40 := sorry

theorem prob2 : (2 * -5 + abs (-3)) = -2 := sorry

theorem prob3 (x y : Int) (h₁ : -x = 3) (h₂ : abs y = 5) : x + y = 2 ∨ x + y = -8 := sorry

theorem prob4 : ((-1 : Int) * (3 / 2) + (5 / 4) + (-5 / 2) - (-13 / 4) - (5 / 4)) = -3 / 4 := sorry

theorem prob5 (a b : Int) (h : abs (a - 4) + abs (b + 5) = 0) : a - b = 9 := sorry

end NUMINAMATH_GPT_prob1_prob2_prob3_prob4_prob5_l1130_113040


namespace NUMINAMATH_GPT_hua_luogeng_optimal_selection_method_uses_golden_ratio_l1130_113098

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end NUMINAMATH_GPT_hua_luogeng_optimal_selection_method_uses_golden_ratio_l1130_113098


namespace NUMINAMATH_GPT_find_incorrect_observation_l1130_113067

theorem find_incorrect_observation (n : ℕ) (initial_mean new_mean : ℝ) (correct_value incorrect_value : ℝ) (observations_count : ℕ)
  (h1 : observations_count = 50)
  (h2 : initial_mean = 36)
  (h3 : new_mean = 36.5)
  (h4 : correct_value = 44) :
  incorrect_value = 19 :=
by
  sorry

end NUMINAMATH_GPT_find_incorrect_observation_l1130_113067


namespace NUMINAMATH_GPT_width_of_rect_prism_l1130_113029

theorem width_of_rect_prism (w : ℝ) 
  (h : ℝ := 8) (l : ℝ := 5) (diagonal : ℝ := 17) 
  (h_diag : l^2 + w^2 + h^2 = diagonal^2) :
  w = 10 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_width_of_rect_prism_l1130_113029


namespace NUMINAMATH_GPT_number_of_ways_to_read_BANANA_l1130_113092

/-- 
In a 3x3 grid, there are 84 different ways to read the word BANANA 
by moving from one cell to another cell with which it shares an edge,
and cells may be visited more than once.
-/
theorem number_of_ways_to_read_BANANA (grid : Matrix (Fin 3) (Fin 3) Char) (word : String := "BANANA") : 
  ∃! n : ℕ, n = 84 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_read_BANANA_l1130_113092


namespace NUMINAMATH_GPT_natural_number_pairs_lcm_gcd_l1130_113097

theorem natural_number_pairs_lcm_gcd (a b : ℕ) (h1 : lcm a b * gcd a b = a * b)
  (h2 : lcm a b - gcd a b = (a * b) / 5) : 
  (a = 4 ∧ b = 20) ∨ (a = 20 ∧ b = 4) :=
  sorry

end NUMINAMATH_GPT_natural_number_pairs_lcm_gcd_l1130_113097


namespace NUMINAMATH_GPT_monotonic_increasing_on_interval_l1130_113026

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - a * Real.log x

theorem monotonic_increasing_on_interval (a : ℝ) :
  (∀ x > 1, 2 * x - a / x ≥ 0) → a ≤ 2 :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_on_interval_l1130_113026


namespace NUMINAMATH_GPT_sequence_problem_l1130_113065

open Nat

theorem sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h : ∀ n : ℕ, 0 < n → S n + a n = 2 * n) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = 7 / 4 ∧ a 4 = 15 / 8 ∧ ∀ n : ℕ, 0 < n → a n = (2^n - 1) / (2^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_problem_l1130_113065


namespace NUMINAMATH_GPT_second_train_length_is_correct_l1130_113022

noncomputable def length_of_second_train (length_first_train : ℝ) (speed_first_train_kmph : ℝ) (speed_second_train_kmph : ℝ) (time_crossing_seconds : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train_kmph * (1000 / 3600)
  let speed_second_train_mps := speed_second_train_kmph * (1000 / 3600)
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance := relative_speed * time_crossing_seconds
  total_distance - length_first_train

theorem second_train_length_is_correct : length_of_second_train 360 120 80 9 = 139.95 :=
by
  sorry

end NUMINAMATH_GPT_second_train_length_is_correct_l1130_113022


namespace NUMINAMATH_GPT_no_fixed_points_range_l1130_113027

def no_fixed_points (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 ≠ x

theorem no_fixed_points_range (a : ℝ) : no_fixed_points a ↔ -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_GPT_no_fixed_points_range_l1130_113027


namespace NUMINAMATH_GPT_totalInitialAmount_l1130_113088

variable (a j t k x : ℝ)

-- Given conditions
def initialToyAmount : Prop :=
  t = 48

def kimRedistribution : Prop :=
  k = 4 * x - 144

def amyRedistribution : Prop :=
  (a = 3 * x) ∧ (j = 2 * x) ∧ (t = 2 * x)

def janRedistribution : Prop :=
  (a = 3 * x) ∧ (t = 4 * x)

def toyRedistribution : Prop :=
  (a = 6 * x) ∧ (j = -6 * x) ∧ (t = 48) 

def toyFinalAmount : Prop :=
  t = 48

-- Proof Problem
theorem totalInitialAmount
  (h1 : initialToyAmount t)
  (h2 : kimRedistribution k x)
  (h3 : amyRedistribution a j t x)
  (h4 : janRedistribution a t x)
  (h5 : toyRedistribution a j t x)
  (h6 : toyFinalAmount t) :
  a + j + t + k = 192 :=
sorry

end NUMINAMATH_GPT_totalInitialAmount_l1130_113088


namespace NUMINAMATH_GPT_bus_speed_l1130_113031

noncomputable def radius : ℝ := 35 / 100  -- Radius in meters
noncomputable def rpm : ℝ := 500.4549590536851

noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def distance_in_one_minute : ℝ := circumference * rpm
noncomputable def distance_in_km_per_hour : ℝ := (distance_in_one_minute / 1000) * 60

theorem bus_speed :
  distance_in_km_per_hour = 66.037 :=
by
  -- The proof is skipped here as it is not required
  sorry

end NUMINAMATH_GPT_bus_speed_l1130_113031


namespace NUMINAMATH_GPT_mean_of_elements_increased_by_2_l1130_113087

noncomputable def calculate_mean_after_increase (m : ℝ) (median_value : ℝ) (increase_value : ℝ) : ℝ :=
  let set := [m, m + 2, m + 4, m + 7, m + 11, m + 13]
  let increased_set := set.map (λ x => x + increase_value)
  increased_set.sum / increased_set.length

theorem mean_of_elements_increased_by_2 (m : ℝ) (h : (m + 4 + m + 7) / 2 = 10) :
  calculate_mean_after_increase m 10 2 = 38 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_mean_of_elements_increased_by_2_l1130_113087


namespace NUMINAMATH_GPT_cricket_team_members_l1130_113078

theorem cricket_team_members (n : ℕ)
    (captain_age : ℕ) (wicket_keeper_age : ℕ) (average_age : ℕ)
    (remaining_average_age : ℕ) (total_age : ℕ) (remaining_players : ℕ) :
    captain_age = 27 →
    wicket_keeper_age = captain_age + 3 →
    average_age = 24 →
    remaining_average_age = average_age - 1 →
    total_age = average_age * n →
    remaining_players = n - 2 →
    total_age = captain_age + wicket_keeper_age + remaining_average_age * remaining_players →
    n = 11 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_cricket_team_members_l1130_113078


namespace NUMINAMATH_GPT_smallest_integer_equal_costs_l1130_113063

-- Definitions based directly on conditions
def decimal_cost (n : ℕ) : ℕ :=
  (n.digits 10).sum * 2

def binary_cost (n : ℕ) : ℕ :=
  (n.digits 2).sum

-- The main statement to prove
theorem smallest_integer_equal_costs : ∃ n : ℕ, n < 2000 ∧ decimal_cost n = binary_cost n ∧ n = 255 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_integer_equal_costs_l1130_113063


namespace NUMINAMATH_GPT_percentage_of_class_taking_lunch_l1130_113052

theorem percentage_of_class_taking_lunch 
  (total_students : ℕ)
  (boys_ratio : ℕ := 6)
  (girls_ratio : ℕ := 4)
  (boys_percentage_lunch : ℝ := 0.60)
  (girls_percentage_lunch : ℝ := 0.40) :
  total_students = 100 →
  (6 / (6 + 4) * 100) = 60 →
  (4 / (6 + 4) * 100) = 40 →
  (boys_percentage_lunch * 60 + girls_percentage_lunch * 40) = 52 →
  ℝ :=
    by
      intros
      sorry

end NUMINAMATH_GPT_percentage_of_class_taking_lunch_l1130_113052


namespace NUMINAMATH_GPT_more_oranges_than_apples_l1130_113091

-- Definitions based on conditions
def apples : ℕ := 14
def oranges : ℕ := 2 * 12  -- 2 dozen oranges

-- Statement to prove
theorem more_oranges_than_apples : oranges - apples = 10 := by
  sorry

end NUMINAMATH_GPT_more_oranges_than_apples_l1130_113091


namespace NUMINAMATH_GPT_sum_of_reciprocals_eq_six_l1130_113044

theorem sum_of_reciprocals_eq_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  (1 / x + 1 / y) = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_eq_six_l1130_113044


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l1130_113024

theorem no_positive_integer_solutions (x y : ℕ) (h : x > 0 ∧ y > 0) : x^2 + (x+1)^2 ≠ y^4 + (y+1)^4 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l1130_113024


namespace NUMINAMATH_GPT_pastries_sold_value_l1130_113059

-- Define the number of cakes sold and the relationship between cakes and pastries
def number_of_cakes_sold := 78
def pastries_sold (C : Nat) := C + 76

-- State the theorem we want to prove
theorem pastries_sold_value : pastries_sold number_of_cakes_sold = 154 := by
  sorry

end NUMINAMATH_GPT_pastries_sold_value_l1130_113059


namespace NUMINAMATH_GPT_max_profit_at_60_l1130_113035

variable (x : ℕ) (y W : ℝ)

def charter_fee : ℝ := 15000
def max_group_size : ℕ := 75

def ticket_price (x : ℕ) : ℝ :=
  if x ≤ 30 then 900
  else if 30 < x ∧ x ≤ max_group_size then -10 * (x - 30) + 900
  else 0

def profit (x : ℕ) : ℝ :=
  if x ≤ 30 then 900 * x - charter_fee
  else if 30 < x ∧ x ≤ max_group_size then (-10 * x + 1200) * x - charter_fee
  else 0

theorem max_profit_at_60 : x = 60 → profit x = 21000 := by
  sorry

end NUMINAMATH_GPT_max_profit_at_60_l1130_113035


namespace NUMINAMATH_GPT_total_walking_time_l1130_113025

open Nat

def walking_time (distance speed : ℕ) : ℕ :=
distance / speed

def number_of_rests (distance : ℕ) : ℕ :=
(distance / 10) - 1

def resting_time_in_minutes (rests : ℕ) : ℕ :=
rests * 5

def resting_time_in_hours (rest_time : ℕ) : ℚ :=
rest_time / 60

def total_time (walking_time resting_time : ℚ) : ℚ :=
walking_time + resting_time

theorem total_walking_time (distance speed : ℕ) (rest_per_10 : ℕ) (rest_time : ℕ) :
  speed = 10 →
  rest_per_10 = 10 →
  rest_time = 5 →
  distance = 50 →
  total_time (walking_time distance speed) (resting_time_in_hours (resting_time_in_minutes (number_of_rests distance))) = 5 + 1 / 3 :=
sorry

end NUMINAMATH_GPT_total_walking_time_l1130_113025


namespace NUMINAMATH_GPT_alcohol_water_ratio_l1130_113037

variable {r s V1 : ℝ}

theorem alcohol_water_ratio 
  (h1 : r > 0) 
  (h2 : s > 0) 
  (h3 : V1 > 0) :
  let alcohol_in_JarA := 2 * r * V1 / (r + 1) + V1
  let water_in_JarA := 2 * V1 / (r + 1)
  let alcohol_in_JarB := 3 * s * V1 / (s + 1)
  let water_in_JarB := 3 * V1 / (s + 1)
  let total_alcohol := alcohol_in_JarA + alcohol_in_JarB
  let total_water := water_in_JarA + water_in_JarB
  (total_alcohol / total_water) = 
  ((2 * r / (r + 1) + 1 + 3 * s / (s + 1)) / (2 / (r + 1) + 3 / (s + 1))) :=
by
  sorry

end NUMINAMATH_GPT_alcohol_water_ratio_l1130_113037


namespace NUMINAMATH_GPT_dish_heats_up_by_5_degrees_per_minute_l1130_113072

theorem dish_heats_up_by_5_degrees_per_minute
  (final_temperature initial_temperature : ℕ)
  (time_taken : ℕ)
  (h1 : final_temperature = 100)
  (h2 : initial_temperature = 20)
  (h3 : time_taken = 16) :
  (final_temperature - initial_temperature) / time_taken = 5 :=
by
  sorry

end NUMINAMATH_GPT_dish_heats_up_by_5_degrees_per_minute_l1130_113072


namespace NUMINAMATH_GPT_impossible_sum_of_two_smaller_angles_l1130_113006

theorem impossible_sum_of_two_smaller_angles
  {α β γ : ℝ}
  (h1 : α + β + γ = 180)
  (h2 : 0 < α + β ∧ α + β < 180) :
  α + β ≠ 130 :=
sorry

end NUMINAMATH_GPT_impossible_sum_of_two_smaller_angles_l1130_113006


namespace NUMINAMATH_GPT_revenue_after_fall_is_correct_l1130_113070

variable (originalRevenue : ℝ) (percentageDecrease : ℝ)

theorem revenue_after_fall_is_correct :
    originalRevenue = 69 ∧ percentageDecrease = 39.130434782608695 →
    originalRevenue - (originalRevenue * (percentageDecrease / 100)) = 42 := by
  intro h
  rcases h with ⟨h1, h2⟩
  sorry

end NUMINAMATH_GPT_revenue_after_fall_is_correct_l1130_113070


namespace NUMINAMATH_GPT_conference_hall_initial_people_l1130_113032

theorem conference_hall_initial_people (x : ℕ)  
  (h1 : 3 ∣ x) 
  (h2 : 4 ∣ (2 * x / 3))
  (h3 : (x / 2) = 27) : 
  x = 54 := 
by 
  sorry

end NUMINAMATH_GPT_conference_hall_initial_people_l1130_113032


namespace NUMINAMATH_GPT_max_complete_bouquets_l1130_113058

-- Definitions based on conditions
def total_roses := 20
def total_lilies := 15
def total_daisies := 10

def wilted_roses := 12
def wilted_lilies := 8
def wilted_daisies := 5

def roses_per_bouquet := 3
def lilies_per_bouquet := 2
def daisies_per_bouquet := 1

-- Calculation of remaining flowers
def remaining_roses := total_roses - wilted_roses
def remaining_lilies := total_lilies - wilted_lilies
def remaining_daisies := total_daisies - wilted_daisies

-- Proof statement
theorem max_complete_bouquets : 
  min
    (remaining_roses / roses_per_bouquet)
    (min (remaining_lilies / lilies_per_bouquet) (remaining_daisies / daisies_per_bouquet)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_max_complete_bouquets_l1130_113058


namespace NUMINAMATH_GPT_catherine_friends_count_l1130_113082

/-
Definition and conditions:
- An equal number of pencils and pens, totaling 60 each.
- Gave away 8 pens and 6 pencils to each friend.
- Left with 22 pens and pencils.
Proof:
- The number of friends she gave pens and pencils to equals 7.
-/
theorem catherine_friends_count :
  ∀ (pencils pens friends : ℕ),
  pens = 60 →
  pencils = 60 →
  (pens + pencils) - friends * (8 + 6) = 22 →
  friends = 7 :=
sorry

end NUMINAMATH_GPT_catherine_friends_count_l1130_113082


namespace NUMINAMATH_GPT_single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l1130_113003

section transmission_scheme

variables (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1)

-- Part A
theorem single_transmission_probability :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β) ^ 2 :=
by sorry

-- Part B
theorem triple_transmission_probability :
  (1 - β) * β * (1 - β) = β * (1 - β) ^ 2 :=
by sorry

-- Part C
theorem triple_transmission_decoding :
  (3 * β * (1 - β) ^ 2) + (1 - β) ^ 3 = β * (1 - β) ^ 2 + (1 - β) ^ 3 :=
by sorry

-- Part D
theorem decoding_comparison (h : 0 < α ∧ α < 0.5) :
  (1 - α) < (3 * α * (1 - α) ^ 2 + (1 - α) ^ 3) :=
by sorry

end transmission_scheme

end NUMINAMATH_GPT_single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l1130_113003


namespace NUMINAMATH_GPT_min_sum_a1_a2_l1130_113020

-- Define the condition predicate for the sequence
def satisfies_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = (a n + 2009) / (1 + a (n + 1))

-- State the main problem as a theorem in Lean 4
theorem min_sum_a1_a2 (a : ℕ → ℕ) (h_seq : satisfies_seq a) (h_pos : ∀ n, a n > 0) :
  a 1 * a 2 = 2009 → a 1 + a 2 = 90 :=
sorry

end NUMINAMATH_GPT_min_sum_a1_a2_l1130_113020


namespace NUMINAMATH_GPT_maximum_value_of_N_l1130_113060

-- Define J_k based on the conditions given
def J (k : ℕ) : ℕ := 10^(k+3) + 128

-- Define the number of factors of 2 in the prime factorization of J_k
def N (k : ℕ) : ℕ := Nat.factorization (J k) 2

-- The proposition to be proved
theorem maximum_value_of_N (k : ℕ) (hk : k > 0) : N 4 = 7 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_N_l1130_113060


namespace NUMINAMATH_GPT_min_marked_price_l1130_113096

theorem min_marked_price 
  (x : ℝ) 
  (sets : ℝ) 
  (cost_per_set : ℝ) 
  (discount : ℝ) 
  (desired_profit : ℝ) 
  (purchase_cost : ℝ) 
  (total_revenue : ℝ) 
  (cost : ℝ)
  (h1 : sets = 40)
  (h2 : cost_per_set = 80)
  (h3 : discount = 0.9)
  (h4 : desired_profit = 4000)
  (h5 : cost = sets * cost_per_set)
  (h6 : total_revenue = sets * (discount * x))
  (h7 : total_revenue - cost ≥ desired_profit) : x ≥ 200 := by
  sorry

end NUMINAMATH_GPT_min_marked_price_l1130_113096


namespace NUMINAMATH_GPT_range_of_x_plus_y_l1130_113076

theorem range_of_x_plus_y (x y : ℝ) (h : x^2 + 2 * x * y - 1 = 0) : (x + y ≤ -1 ∨ x + y ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_plus_y_l1130_113076


namespace NUMINAMATH_GPT_license_plates_count_l1130_113008

def num_consonants : Nat := 20
def num_vowels : Nat := 6
def num_digits : Nat := 10
def num_symbols : Nat := 3

theorem license_plates_count : 
  num_consonants * num_vowels * num_consonants * num_digits * num_symbols = 72000 :=
by 
  sorry

end NUMINAMATH_GPT_license_plates_count_l1130_113008


namespace NUMINAMATH_GPT_zhiqiang_series_l1130_113085

theorem zhiqiang_series (a b : ℝ) (n : ℕ) (n_pos : 0 < n) (h : a * b = 1) (h₀ : b ≠ 1):
  (1 + a^n) / (1 + b^n) = ((1 + a) / (1 + b)) ^ n :=
by
  sorry

end NUMINAMATH_GPT_zhiqiang_series_l1130_113085


namespace NUMINAMATH_GPT_scallops_cost_l1130_113014

-- define the conditions
def scallops_per_pound : ℝ := 8
def cost_per_pound : ℝ := 24
def scallops_per_person : ℝ := 2
def number_of_people : ℝ := 8

-- the question
theorem scallops_cost : (scallops_per_person * number_of_people / scallops_per_pound) * cost_per_pound = 48 := by 
  sorry

end NUMINAMATH_GPT_scallops_cost_l1130_113014


namespace NUMINAMATH_GPT_determine_b_from_inequality_l1130_113049

theorem determine_b_from_inequality (b : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - b * x + 6 < 0) → b = 5 :=
by
  intro h
  -- Proof can be added here
  sorry

end NUMINAMATH_GPT_determine_b_from_inequality_l1130_113049


namespace NUMINAMATH_GPT_find_k_l1130_113012

-- Define the problem conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 1)
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- Define the dot product for 2D vectors
def dot_prod (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

-- State the theorem
theorem find_k (k : ℝ) (h : dot_prod b (c k) = 0) : k = -3/2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1130_113012


namespace NUMINAMATH_GPT_system_eq_solution_l1130_113042

theorem system_eq_solution (x y c d : ℝ) (hd : d ≠ 0) 
  (h1 : 4 * x - 2 * y = c) 
  (h2 : 6 * y - 12 * x = d) :
  c / d = -1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_system_eq_solution_l1130_113042


namespace NUMINAMATH_GPT_compare_neg5_neg7_l1130_113086

theorem compare_neg5_neg7 : -5 > -7 := 
by
  sorry

end NUMINAMATH_GPT_compare_neg5_neg7_l1130_113086


namespace NUMINAMATH_GPT_arithmetic_sequence_second_term_l1130_113081

theorem arithmetic_sequence_second_term (S₃: ℕ) (a₁: ℕ) (h1: S₃ = 9) (h2: a₁ = 1) : 
∃ d a₂, 3 * a₁ + 3 * d = S₃ ∧ a₂ = a₁ + d ∧ a₂ = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_second_term_l1130_113081


namespace NUMINAMATH_GPT_valid_assignment_statement_l1130_113017

theorem valid_assignment_statement (S a : ℕ) : (S = a + 1) ∧ ¬(a + 1 = S) ∧ ¬(S - 1 = a) ∧ ¬(S - a = 1) := by
  sorry

end NUMINAMATH_GPT_valid_assignment_statement_l1130_113017


namespace NUMINAMATH_GPT_inequality_proof_l1130_113074

open Real

-- Define the conditions
def conditions (a b c : ℝ) := (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a * b * c = 1)

-- Express the inequality we need to prove
def inequality (a b c : ℝ) :=
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1

-- Statement of the theorem
theorem inequality_proof (a b c : ℝ) (h : conditions a b c) : inequality a b c :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_proof_l1130_113074
