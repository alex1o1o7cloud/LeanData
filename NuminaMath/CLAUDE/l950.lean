import Mathlib

namespace NUMINAMATH_CALUDE_sum_sequence_37th_term_l950_95036

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_sequence_37th_term
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : arithmetic_sequence b)
  (ha1 : a 1 = 25)
  (hb1 : b 1 = 75)
  (hab2 : a 2 + b 2 = 100) :
  a 37 + b 37 = 100 := by
sorry

end NUMINAMATH_CALUDE_sum_sequence_37th_term_l950_95036


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l950_95006

def f (x : ℝ) : ℝ := -x^2 + 1

theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l950_95006


namespace NUMINAMATH_CALUDE_polynomial_factorization_l950_95027

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 16 = (x^2 - 2) * (x^2 + 2) * (x^2 - 2*x + 2) * (x^2 + 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l950_95027


namespace NUMINAMATH_CALUDE_sample_data_properties_l950_95049

def median (s : Finset ℝ) : ℝ := sorry

theorem sample_data_properties (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (h : x₁ ≤ x₂ ∧ x₂ ≤ x₃ ∧ x₃ ≤ x₄ ∧ x₄ ≤ x₅ ∧ x₅ ≤ x₆) :
  let s₁ := {x₂, x₃, x₄, x₅}
  let s₂ := {x₁, x₂, x₃, x₄, x₅, x₆}
  (median s₁ = median s₂) ∧ 
  (x₅ - x₂ ≤ x₆ - x₁) := by
  sorry

end NUMINAMATH_CALUDE_sample_data_properties_l950_95049


namespace NUMINAMATH_CALUDE_equal_age_regression_five_year_difference_regression_ten_percent_older_regression_l950_95041

-- Define the type for a couple's ages
structure CoupleAges where
  bride_age : ℝ
  groom_age : ℝ

-- Define the dataset
def dataset : List CoupleAges := sorry

-- Define the regression line
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

-- Function to calculate regression line
def calculate_regression_line (data : List CoupleAges) : RegressionLine := sorry

-- Theorems to prove

theorem equal_age_regression :
  (∀ couple ∈ dataset, couple.bride_age = couple.groom_age) →
  let reg_line := calculate_regression_line dataset
  reg_line.slope = 1 ∧ reg_line.intercept = 0 := by sorry

theorem five_year_difference_regression :
  (∀ couple ∈ dataset, couple.groom_age = couple.bride_age + 5) →
  let reg_line := calculate_regression_line dataset
  reg_line.slope = 1 ∧ reg_line.intercept = 5 := by sorry

theorem ten_percent_older_regression :
  (∀ couple ∈ dataset, couple.groom_age = 1.1 * couple.bride_age) →
  let reg_line := calculate_regression_line dataset
  reg_line.slope = 1.1 ∧ reg_line.intercept = 0 := by sorry

end NUMINAMATH_CALUDE_equal_age_regression_five_year_difference_regression_ten_percent_older_regression_l950_95041


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l950_95013

/-- Given a quadratic function g(x) = x^2 + cx + d, prove that if g(g(x) + x) / g(x) = x^2 + 44x + 50, then g(x) = x^2 + 44x + 50 -/
theorem quadratic_function_proof (c d : ℝ) (g : ℝ → ℝ) 
  (h1 : ∀ x, g x = x^2 + c*x + d)
  (h2 : ∀ x, g (g x + x) / g x = x^2 + 44*x + 50) :
  ∀ x, g x = x^2 + 44*x + 50 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l950_95013


namespace NUMINAMATH_CALUDE_power_of_2016_condition_l950_95088

theorem power_of_2016_condition (a b c : ℤ) : 
  let N := ((a - b) * (b - c) * (c - a)) / 2 + 2
  ∃ (n : ℕ), N = 2016^n → N = 1 := by
sorry

end NUMINAMATH_CALUDE_power_of_2016_condition_l950_95088


namespace NUMINAMATH_CALUDE_median_formulas_l950_95059

/-- Triangle with sides a, b, c and medians ma, mb, mc -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ma : ℝ
  mb : ℝ
  mc : ℝ

/-- Theorem: Median formula and sum of squares of medians -/
theorem median_formulas (t : Triangle) :
  t.ma^2 = (2*t.b^2 + 2*t.c^2 - t.a^2) / 4 ∧
  t.ma^2 + t.mb^2 + t.mc^2 = 3*(t.a^2 + t.b^2 + t.c^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_median_formulas_l950_95059


namespace NUMINAMATH_CALUDE_guaranteed_scores_l950_95025

/-- Represents a player in the card game -/
inductive Player : Type
| One
| Two

/-- The deck of cards for each player -/
def player_deck (p : Player) : List Nat :=
  match p with
  | Player.One => List.range 1000 |>.map (fun n => 2 * n + 2)
  | Player.Two => List.range 1001 |>.map (fun n => 2 * n + 1)

/-- The number of turns in the game -/
def num_turns : Nat := 1000

/-- The result of the game -/
structure GameResult where
  player1_score : Nat
  player2_score : Nat

/-- A strategy for playing the game -/
def Strategy := List Nat → Nat

/-- Play the game with given strategies -/
def play_game (s1 s2 : Strategy) : GameResult :=
  sorry

/-- The theorem stating the guaranteed minimum scores for both players -/
theorem guaranteed_scores :
  ∃ (s1 : Strategy), ∀ (s2 : Strategy), (play_game s1 s2).player1_score ≥ 499 ∧
  ∃ (s2 : Strategy), ∀ (s1 : Strategy), (play_game s1 s2).player2_score ≥ 501 :=
  sorry

end NUMINAMATH_CALUDE_guaranteed_scores_l950_95025


namespace NUMINAMATH_CALUDE_garden_ratio_maintenance_l950_95003

/-- Represents a garden with tulips and daisies -/
structure Garden where
  tulips : ℕ
  daisies : ℕ

/-- Calculates the number of tulips needed to maintain a 3:7 ratio with the given number of daisies -/
def tulipsForRatio (daisies : ℕ) : ℕ :=
  (3 * daisies + 6) / 7

theorem garden_ratio_maintenance (initial : Garden) (added_daisies : ℕ) :
  initial.daisies = 35 →
  added_daisies = 30 →
  (3 : ℚ) / 7 = initial.tulips / initial.daisies →
  tulipsForRatio (initial.daisies + added_daisies) = 28 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_maintenance_l950_95003


namespace NUMINAMATH_CALUDE_cos_30_minus_cos_60_l950_95039

theorem cos_30_minus_cos_60 : Real.cos (30 * π / 180) - Real.cos (60 * π / 180) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_30_minus_cos_60_l950_95039


namespace NUMINAMATH_CALUDE_slope_of_line_l950_95020

theorem slope_of_line (x y : ℝ) : 3 * y + 2 * x = 12 → (y - 4) / x = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l950_95020


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l950_95090

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x^2 + 2*x - 3}

-- Define set B
def B : Set ℝ := {y | ∃ x < 0, y = x + 1/x}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Icc (-4) (-2) := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l950_95090


namespace NUMINAMATH_CALUDE_linda_original_correct_l950_95037

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := 20

/-- The amount of money Lucy would give to Linda -/
def transfer_amount : ℕ := 5

/-- Linda's original amount of money -/
def linda_original : ℕ := 10

/-- Theorem stating that Linda's original amount is correct -/
theorem linda_original_correct : 
  lucy_original - transfer_amount = linda_original + transfer_amount := by
  sorry

end NUMINAMATH_CALUDE_linda_original_correct_l950_95037


namespace NUMINAMATH_CALUDE_total_balls_l950_95080

theorem total_balls (jungkook_balls yoongi_balls : ℕ) : 
  jungkook_balls = 3 → yoongi_balls = 2 → jungkook_balls + yoongi_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_l950_95080


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l950_95052

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|x - 3| = 5 - 2*x) ↔ (x = 2 ∨ x = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l950_95052


namespace NUMINAMATH_CALUDE_pole_wire_distance_l950_95095

/-- Given a vertical pole and three equally spaced wires, calculate the distance between anchor points -/
theorem pole_wire_distance (pole_height : ℝ) (wire_length : ℝ) (anchor_distance : ℝ) : 
  pole_height = 70 →
  wire_length = 490 →
  (pole_height ^ 2 + (anchor_distance / (3 ^ (1/2))) ^ 2 = wire_length ^ 2) →
  anchor_distance = 840 := by
  sorry

end NUMINAMATH_CALUDE_pole_wire_distance_l950_95095


namespace NUMINAMATH_CALUDE_angle_range_theorem_l950_95067

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is monotonically increasing on an interval (a, b) if
    for all x, y in (a, b), x < y implies f(x) < f(y) -/
def MonoIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem angle_range_theorem (f : ℝ → ℝ) (A : ℝ) :
  IsOdd f →
  MonoIncreasing f 0 Real.pi →
  f (1/2) = 0 →
  f (Real.cos A) < 0 →
  (π/3 < A ∧ A < π/2) ∨ (2*π/3 < A ∧ A < π) :=
sorry

end NUMINAMATH_CALUDE_angle_range_theorem_l950_95067


namespace NUMINAMATH_CALUDE_car_cost_calculation_l950_95072

/-- The total cost of a car given an initial payment, monthly payment, and number of months -/
theorem car_cost_calculation (initial_payment monthly_payment num_months : ℕ) :
  initial_payment = 5400 →
  monthly_payment = 420 →
  num_months = 19 →
  initial_payment + monthly_payment * num_months = 13380 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_calculation_l950_95072


namespace NUMINAMATH_CALUDE_triangle_area_l950_95002

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  b = Real.sqrt 3 →
  C = π / 6 →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l950_95002


namespace NUMINAMATH_CALUDE_intended_profit_percentage_l950_95010

/-- Given a cost price, labeled price, and selling price satisfying certain conditions,
    prove that the intended profit percentage is 1/3. -/
theorem intended_profit_percentage
  (C L S : ℝ)  -- Cost price, Labeled price, Selling price
  (P : ℝ)      -- Intended profit percentage (as a decimal)
  (h1 : L = C * (1 + P))        -- Labeled price condition
  (h2 : S = 0.90 * L)           -- 10% discount condition
  (h3 : S = 1.17 * C)           -- 17% actual profit condition
  : P = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_intended_profit_percentage_l950_95010


namespace NUMINAMATH_CALUDE_fourth_number_unit_digit_l950_95070

def unit_digit (n : ℕ) : ℕ := n % 10

theorem fourth_number_unit_digit 
  (a b c : ℕ) 
  (ha : a = 7858) 
  (hb : b = 1086) 
  (hc : c = 4582) : 
  ∃ d : ℕ, unit_digit (a * b * c * d) = 8 ↔ unit_digit d = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_unit_digit_l950_95070


namespace NUMINAMATH_CALUDE_ellipse_b_plus_k_l950_95046

/-- Definition of an ellipse with given foci and a point on the curve -/
def Ellipse (f1 f2 p : ℝ × ℝ) :=
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
    (p.1 - h)^2 / a^2 + (p.2 - k)^2 / b^2 = 1 ∧
    (f1.1 - h)^2 / a^2 + (f1.2 - k)^2 / b^2 = 1 ∧
    (f2.1 - h)^2 / a^2 + (f2.2 - k)^2 / b^2 = 1

/-- Theorem stating the sum of b and k for the given ellipse -/
theorem ellipse_b_plus_k :
  ∀ (a b h k : ℝ),
    Ellipse (2, 3) (2, 7) (6, 5) →
    a > 0 →
    b > 0 →
    (6 - h)^2 / a^2 + (5 - k)^2 / b^2 = 1 →
    b + k = 4 * Real.sqrt 5 + 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_b_plus_k_l950_95046


namespace NUMINAMATH_CALUDE_unique_cubic_function_l950_95017

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

theorem unique_cubic_function (a b : ℝ) (h1 : a > 0) 
  (h2 : ∃ (x : ℝ), f a b x = 5 ∧ ∀ (y : ℝ), f a b y ≤ 5)
  (h3 : ∃ (x : ℝ), f a b x = 1 ∧ ∀ (y : ℝ), f a b y ≥ 1) :
  ∀ (x : ℝ), f a b x = x^3 + 3*x^2 + 1 := by
sorry

end NUMINAMATH_CALUDE_unique_cubic_function_l950_95017


namespace NUMINAMATH_CALUDE_refrigerator_profit_theorem_l950_95062

def refrigerator_profit (cost_price marked_price : ℝ) : Prop :=
  let profit_20_off := 0.8 * marked_price - cost_price
  let profit_margin := profit_20_off / cost_price
  profit_20_off = 200 ∧
  profit_margin = 0.1 ∧
  0.85 * marked_price - cost_price = 337.5

theorem refrigerator_profit_theorem :
  ∃ (cost_price marked_price : ℝ),
    refrigerator_profit cost_price marked_price :=
  sorry

end NUMINAMATH_CALUDE_refrigerator_profit_theorem_l950_95062


namespace NUMINAMATH_CALUDE_purple_shoes_count_l950_95096

/-- Prove the number of purple shoes in a warehouse --/
theorem purple_shoes_count (total : ℕ) (blue : ℕ) (green : ℕ) (purple : ℕ) : 
  total = 1250 →
  blue = 540 →
  green + purple = total - blue →
  green = purple →
  purple = 355 := by
sorry

end NUMINAMATH_CALUDE_purple_shoes_count_l950_95096


namespace NUMINAMATH_CALUDE_existence_of_constant_g_l950_95035

-- Define the necessary types and functions
def Graph : Type := sorry
def circumference (G : Graph) : ℕ := sorry
def chromaticNumber (G : Graph) : ℕ := sorry
def containsSubgraph (G H : Graph) : Prop := sorry
def TK (r : ℕ) : Graph := sorry

-- The main theorem
theorem existence_of_constant_g : 
  ∃ g : ℕ, ∀ (G : Graph) (r : ℕ), 
    circumference G ≥ g → chromaticNumber G ≥ r → containsSubgraph G (TK r) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_constant_g_l950_95035


namespace NUMINAMATH_CALUDE_quadratic_inequality_l950_95075

theorem quadratic_inequality (x : ℝ) : 
  -10 * x^2 + 4 * x + 2 > 0 ↔ (1 - Real.sqrt 6) / 5 < x ∧ x < (1 + Real.sqrt 6) / 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l950_95075


namespace NUMINAMATH_CALUDE_base_b_divisibility_l950_95073

theorem base_b_divisibility (b : ℤ) : 
  let diff := 2 * b^3 + 2 * b - (2 * b^2 + 2 * b + 1)
  (b = 8 ∧ ¬(diff % 3 = 0)) ∨
  ((b = 3 ∨ b = 4 ∨ b = 6 ∨ b = 7) ∧ (diff % 3 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_base_b_divisibility_l950_95073


namespace NUMINAMATH_CALUDE_sequence_formula_smallest_m_bound_l950_95030

def S (n : ℕ) : ℚ := 3/2 * n^2 - 1/2 * n

def a (n : ℕ+) : ℚ := 3 * n - 2

def T (n : ℕ+) : ℚ := 1 - 1 / (3 * n + 1)

theorem sequence_formula (n : ℕ+) : a n = 3 * n - 2 :=
sorry

theorem smallest_m_bound : 
  ∃ m : ℕ, (∀ n : ℕ+, T n < m / 20) ∧ (∀ k : ℕ, k < m → ∃ n : ℕ+, T n ≥ k / 20) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_smallest_m_bound_l950_95030


namespace NUMINAMATH_CALUDE_smith_family_seating_l950_95023

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smith_family_seating (boys girls : ℕ) (h : boys = 4 ∧ girls = 4) : 
  factorial (boys + girls) - (factorial boys * factorial girls) = 39744 :=
by sorry

end NUMINAMATH_CALUDE_smith_family_seating_l950_95023


namespace NUMINAMATH_CALUDE_paperboy_delivery_patterns_l950_95093

def deliverySequences (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 15
  | m + 5 => deliverySequences (m + 4) + deliverySequences (m + 3) + 
             deliverySequences (m + 2) + deliverySequences (m + 1)

theorem paperboy_delivery_patterns : deliverySequences 12 = 2873 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_patterns_l950_95093


namespace NUMINAMATH_CALUDE_triangle_side_equations_l950_95032

/-- Given a triangle ABC with point A at (1,3) and medians from A satisfying specific equations,
    prove that the sides of the triangle have the given equations. -/
theorem triangle_side_equations (B C : ℝ × ℝ) : 
  let A : ℝ × ℝ := (1, 3)
  let median_to_BC (x y : ℝ) := x - 2*y + 1 = 0
  let median_to_AC (x y : ℝ) := y = 1
  (∃ t : ℝ, median_to_BC (C.1 + t*(B.1 - C.1)) (C.2 + t*(B.2 - C.2))) ∧ 
  (∃ s : ℝ, median_to_AC ((1 + C.1)/2) ((3 + C.2)/2)) →
  (B.2 - 3 = (3 - 1)/(1 - B.1) * (B.1 - 1)) ∧ 
  (C.2 - B.2 = (C.2 - B.2)/(C.1 - B.1) * (C.1 - B.1)) ∧ 
  (C.2 - 3 = (C.2 - 3)/(C.1 - 1) * (C.1 - 1)) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_equations_l950_95032


namespace NUMINAMATH_CALUDE_number_with_inserted_zero_l950_95076

def insert_zero (n : ℕ) : ℕ :=
  10000 * (n / 1000) + 1000 * ((n / 100) % 10) + (n % 100)

theorem number_with_inserted_zero (N : ℕ) :
  (insert_zero N = 9 * N) → (N = 225 ∨ N = 450 ∨ N = 675) := by
sorry

end NUMINAMATH_CALUDE_number_with_inserted_zero_l950_95076


namespace NUMINAMATH_CALUDE_average_salary_all_employees_l950_95057

/-- Calculate the average salary of all employees in an office --/
theorem average_salary_all_employees 
  (avg_salary_officers : ℝ) 
  (avg_salary_non_officers : ℝ) 
  (num_officers : ℕ) 
  (num_non_officers : ℕ) 
  (h1 : avg_salary_officers = 440)
  (h2 : avg_salary_non_officers = 110)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 480) :
  (avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers) / (num_officers + num_non_officers) = 120 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_all_employees_l950_95057


namespace NUMINAMATH_CALUDE_peggy_stamp_count_l950_95091

/-- The number of stamps Peggy has -/
def peggy_stamps : ℕ := 75

/-- The number of stamps Ernie has -/
def ernie_stamps : ℕ := 3 * peggy_stamps

/-- The number of stamps Bert has -/
def bert_stamps : ℕ := 4 * ernie_stamps

theorem peggy_stamp_count : 
  bert_stamps = peggy_stamps + 825 ∧ 
  ernie_stamps = 3 * peggy_stamps ∧ 
  bert_stamps = 4 * ernie_stamps →
  peggy_stamps = 75 := by sorry

end NUMINAMATH_CALUDE_peggy_stamp_count_l950_95091


namespace NUMINAMATH_CALUDE_first_player_seeds_l950_95014

/-- Given a sunflower seed eating contest with three players, where:
  * The second player eats 53 seeds
  * The third player eats 30 more seeds than the second
  * The total number of seeds eaten is 214
  This theorem proves that the first player eats 78 seeds. -/
theorem first_player_seeds (second_player : ℕ) (third_player : ℕ) (total_seeds : ℕ) :
  second_player = 53 →
  third_player = second_player + 30 →
  total_seeds = 214 →
  total_seeds = second_player + third_player + 78 :=
by sorry

end NUMINAMATH_CALUDE_first_player_seeds_l950_95014


namespace NUMINAMATH_CALUDE_forall_op_example_l950_95050

-- Define the new operation ∀
def forall_op (a b : ℚ) : ℚ := -a - b^2

-- Theorem statement
theorem forall_op_example : forall_op (forall_op 2022 1) 2 = 2019 := by
  sorry

end NUMINAMATH_CALUDE_forall_op_example_l950_95050


namespace NUMINAMATH_CALUDE_two_digit_special_property_l950_95066

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

theorem two_digit_special_property : 
  {n : ℕ | is_two_digit n ∧ n = 6 * sum_of_digits (n + 7)} = {24, 78} := by
  sorry

end NUMINAMATH_CALUDE_two_digit_special_property_l950_95066


namespace NUMINAMATH_CALUDE_evaluate_expression_l950_95045

theorem evaluate_expression (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l950_95045


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l950_95064

theorem sufficient_condition_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l950_95064


namespace NUMINAMATH_CALUDE_binary_to_decimal_11001101_l950_95016

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary_digits : List Bool) : ℕ :=
  binary_digits.enum.foldl
    (fun acc (i, b) => acc + (if b then 2^i else 0))
    0

/-- The binary representation of the number we want to convert -/
def binary_number : List Bool := [true, false, true, true, false, false, true, true]

/-- Theorem stating that the decimal equivalent of 11001101 (binary) is 205 -/
theorem binary_to_decimal_11001101 :
  binary_to_decimal (binary_number.reverse) = 205 := by
  sorry

#eval binary_to_decimal (binary_number.reverse)

end NUMINAMATH_CALUDE_binary_to_decimal_11001101_l950_95016


namespace NUMINAMATH_CALUDE_more_even_products_l950_95056

def S : Finset Nat := {1, 2, 3, 4, 5}

def pairs : Finset (Nat × Nat) :=
  S.product S |>.filter (λ (a, b) => a ≤ b)

def products : Finset Nat :=
  pairs.image (λ (a, b) => a * b)

def evenProducts : Finset Nat :=
  products.filter (λ x => x % 2 = 0)

def oddProducts : Finset Nat :=
  products.filter (λ x => x % 2 ≠ 0)

theorem more_even_products :
  Finset.card evenProducts > Finset.card oddProducts :=
by sorry

end NUMINAMATH_CALUDE_more_even_products_l950_95056


namespace NUMINAMATH_CALUDE_f_is_quadratic_l950_95077

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation x^2 - 2 = x -/
def f (x : ℝ) : ℝ := x^2 - x - 2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l950_95077


namespace NUMINAMATH_CALUDE_problem_statement_l950_95092

theorem problem_statement (x₁ x₂ : ℝ) 
  (h₁ : |x₁ - 2| < 1) 
  (h₂ : |x₂ - 2| < 1) : 
  (2 < x₁ + x₂ ∧ x₁ + x₂ < 6 ∧ |x₁ - x₂| < 2) ∧ 
  (let f := fun x => x^2 - x + 1
   |x₁ - x₂| < |f x₁ - f x₂| ∧ |f x₁ - f x₂| < 5 * |x₁ - x₂|) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l950_95092


namespace NUMINAMATH_CALUDE_max_earnings_is_250_l950_95086

/-- Represents a plumbing job with counts of toilets, showers, and sinks to be fixed -/
structure PlumbingJob where
  toilets : ℕ
  showers : ℕ
  sinks : ℕ

/-- Calculates the earnings for a given plumbing job -/
def jobEarnings (job : PlumbingJob) : ℕ :=
  job.toilets * 50 + job.showers * 40 + job.sinks * 30

/-- The list of available jobs -/
def availableJobs : List PlumbingJob := [
  { toilets := 3, showers := 0, sinks := 3 },
  { toilets := 2, showers := 0, sinks := 5 },
  { toilets := 1, showers := 2, sinks := 3 }
]

/-- Theorem stating that the maximum earnings from the available jobs is $250 -/
theorem max_earnings_is_250 : 
  (availableJobs.map jobEarnings).maximum? = some 250 := by sorry

end NUMINAMATH_CALUDE_max_earnings_is_250_l950_95086


namespace NUMINAMATH_CALUDE_remainder_512_power_512_mod_13_l950_95065

theorem remainder_512_power_512_mod_13 : 512^512 ≡ 1 [ZMOD 13] := by sorry

end NUMINAMATH_CALUDE_remainder_512_power_512_mod_13_l950_95065


namespace NUMINAMATH_CALUDE_count_hens_and_cows_l950_95097

theorem count_hens_and_cows (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) : 
  total_animals = 44 → 
  total_feet = 128 → 
  hen_feet = 2 → 
  cow_feet = 4 → 
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧ 
    hens = 24 := by
  sorry

end NUMINAMATH_CALUDE_count_hens_and_cows_l950_95097


namespace NUMINAMATH_CALUDE_max_product_of_three_distinct_naturals_l950_95000

theorem max_product_of_three_distinct_naturals (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c → a + b + c = 48 → a * b * c ≤ 4080 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_three_distinct_naturals_l950_95000


namespace NUMINAMATH_CALUDE_total_time_spent_on_pictures_johns_total_time_is_34_hours_l950_95058

/-- Calculates the total time spent on drawing and coloring pictures. -/
theorem total_time_spent_on_pictures 
  (num_pictures : ℕ) 
  (drawing_time : ℝ) 
  (coloring_time_reduction : ℝ) : ℝ :=
  let coloring_time := drawing_time * (1 - coloring_time_reduction)
  let time_per_picture := drawing_time + coloring_time
  num_pictures * time_per_picture

/-- Proves that John spends 34 hours on all pictures given the conditions. -/
theorem johns_total_time_is_34_hours : 
  total_time_spent_on_pictures 10 2 0.3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_time_spent_on_pictures_johns_total_time_is_34_hours_l950_95058


namespace NUMINAMATH_CALUDE_inflation_time_is_20_l950_95015

/-- The time it takes to inflate one soccer ball -/
def inflation_time : ℕ := sorry

/-- The number of balls Alexia inflates -/
def alexia_balls : ℕ := 20

/-- The number of balls Ermias inflates -/
def ermias_balls : ℕ := alexia_balls + 5

/-- The total time taken to inflate all balls -/
def total_time : ℕ := 900

/-- Theorem stating that the inflation time for one ball is 20 minutes -/
theorem inflation_time_is_20 : inflation_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_inflation_time_is_20_l950_95015


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l950_95001

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = Complex.abs (3 - 4*I)) :
  z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l950_95001


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l950_95089

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^3 + 1/a^3 = 2 * Real.sqrt 5 ∨ a^3 + 1/a^3 = -2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l950_95089


namespace NUMINAMATH_CALUDE_gym_income_calculation_l950_95051

/-- A gym charges its members a certain amount twice a month and has a fixed number of members. -/
structure Gym where
  charge_per_half_month : ℕ
  number_of_members : ℕ

/-- Calculate the monthly income of a gym -/
def monthly_income (g : Gym) : ℕ :=
  g.charge_per_half_month * 2 * g.number_of_members

/-- Theorem: A gym that charges $18 twice a month and has 300 members makes $10,800 per month -/
theorem gym_income_calculation :
  let g : Gym := { charge_per_half_month := 18, number_of_members := 300 }
  monthly_income g = 10800 := by
  sorry

end NUMINAMATH_CALUDE_gym_income_calculation_l950_95051


namespace NUMINAMATH_CALUDE_matt_current_age_is_65_l950_95048

def james_age_3_years_ago : ℕ := 27
def years_since_james_27 : ℕ := 3
def years_until_matt_twice_james : ℕ := 5

def james_current_age : ℕ := james_age_3_years_ago + years_since_james_27

def james_age_in_5_years : ℕ := james_current_age + years_until_matt_twice_james

def matt_age_in_5_years : ℕ := 2 * james_age_in_5_years

theorem matt_current_age_is_65 : matt_age_in_5_years - years_until_matt_twice_james = 65 := by
  sorry

end NUMINAMATH_CALUDE_matt_current_age_is_65_l950_95048


namespace NUMINAMATH_CALUDE_no_real_solution_l950_95024

theorem no_real_solution : ¬∃ (x y : ℝ), x^3 + y^2 = 2 ∧ x^2 + x*y + y^2 - y = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l950_95024


namespace NUMINAMATH_CALUDE_fibonacci_sum_convergence_l950_95012

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of F_n / 2^n from n = 0 to infinity -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (2 ^ n)

theorem fibonacci_sum_convergence : fibSum = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_sum_convergence_l950_95012


namespace NUMINAMATH_CALUDE_range_of_m_l950_95081

/-- The probability that two lines l₁: ax + by = 2 and l₂: x + 2y = 2 are parallel,
    where a and b are results of two dice throws. -/
def P₁ : ℚ := 1/18

/-- The probability that two lines l₁: ax + by = 2 and l₂: x + 2y = 2 intersect,
    where a and b are results of two dice throws. -/
def P₂ : ℚ := 11/12

/-- The theorem stating the range of m for which the point (P₁, P₂) is inside
    the circle (x-m)² + y² = 137/144. -/
theorem range_of_m : ∀ m : ℚ, 
  (P₁ - m)^2 + P₂^2 < 137/144 ↔ -5/18 < m ∧ m < 7/18 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l950_95081


namespace NUMINAMATH_CALUDE_min_perimeter_noncongruent_isosceles_triangles_l950_95044

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- The area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

theorem min_perimeter_noncongruent_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    t1.base * 5 = t2.base * 6 ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      s1.base * 5 = s2.base * 6 →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 364 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_noncongruent_isosceles_triangles_l950_95044


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l950_95022

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ∀ x : ℝ, x^2 + b*x + a = 0 ↔ x = a ∨ x = b) : 
  a = -2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l950_95022


namespace NUMINAMATH_CALUDE_x_value_l950_95042

theorem x_value (x : ℝ) (h1 : x > 0) (h2 : Real.sqrt ((8 * x) / 3) = x) : x = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l950_95042


namespace NUMINAMATH_CALUDE_unique_divisible_by_thirteen_l950_95007

theorem unique_divisible_by_thirteen :
  ∀ (B : Nat),
    B < 10 →
    (2000 + 100 * B + 34) % 13 = 0 ↔ B = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_thirteen_l950_95007


namespace NUMINAMATH_CALUDE_binomial_20_10_l950_95055

theorem binomial_20_10 (h1 : Nat.choose 18 9 = 48620) 
                       (h2 : Nat.choose 18 10 = 43758) 
                       (h3 : Nat.choose 18 11 = 24310) : 
  Nat.choose 20 10 = 97240 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_10_l950_95055


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_181_l950_95009

theorem consecutive_squares_sum_181 :
  ∃ k : ℕ, k^2 + (k+1)^2 = 181 ∧ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_181_l950_95009


namespace NUMINAMATH_CALUDE_walking_distance_l950_95071

/-- Given a person who walks at two speeds, prove that the actual distance traveled at the slower speed is 50 km. -/
theorem walking_distance (slow_speed fast_speed extra_distance : ℝ) 
  (h1 : slow_speed = 10)
  (h2 : fast_speed = 14)
  (h3 : extra_distance = 20)
  (h4 : slow_speed > 0)
  (h5 : fast_speed > slow_speed) :
  ∃ (actual_distance : ℝ),
    actual_distance / slow_speed = (actual_distance + extra_distance) / fast_speed ∧
    actual_distance = 50 := by
  sorry


end NUMINAMATH_CALUDE_walking_distance_l950_95071


namespace NUMINAMATH_CALUDE_total_carriages_l950_95038

/-- The number of carriages in each town -/
structure TownCarriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flying_scotsman : ℕ

/-- The conditions given in the problem -/
def problem_conditions (tc : TownCarriages) : Prop :=
  tc.euston = tc.norfolk + 20 ∧
  tc.norwich = 100 ∧
  tc.flying_scotsman = tc.norwich + 20 ∧
  tc.euston = 130

/-- The theorem stating that the total number of carriages is 460 -/
theorem total_carriages (tc : TownCarriages) 
  (h : problem_conditions tc) : 
  tc.euston + tc.norfolk + tc.norwich + tc.flying_scotsman = 460 := by
  sorry

end NUMINAMATH_CALUDE_total_carriages_l950_95038


namespace NUMINAMATH_CALUDE_bobs_work_hours_l950_95054

/-- Given Bob's wage increase, benefit reduction, and net weekly gain, 
    prove that he works 40 hours per week. -/
theorem bobs_work_hours : 
  ∀ (h : ℝ), 
    (0.50 * h - 15 = 5) → 
    h = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_bobs_work_hours_l950_95054


namespace NUMINAMATH_CALUDE_parking_space_per_car_l950_95005

/-- Calculates the area required to park one car given the dimensions of a parking lot,
    the percentage of usable area, and the total number of cars that can be parked. -/
theorem parking_space_per_car
  (length width : ℝ)
  (usable_percentage : ℝ)
  (total_cars : ℕ)
  (h1 : length = 400)
  (h2 : width = 500)
  (h3 : usable_percentage = 0.8)
  (h4 : total_cars = 16000) :
  (length * width * usable_percentage) / total_cars = 10 := by
  sorry

#check parking_space_per_car

end NUMINAMATH_CALUDE_parking_space_per_car_l950_95005


namespace NUMINAMATH_CALUDE_magnitude_of_z_l950_95085

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) : Prop := (1 + i) * z = 3 + i

-- State the theorem
theorem magnitude_of_z (z : ℂ) (h : given_equation z) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l950_95085


namespace NUMINAMATH_CALUDE_power_of_two_expression_l950_95011

theorem power_of_two_expression : 2^3 * 2^3 + 2^3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_expression_l950_95011


namespace NUMINAMATH_CALUDE_log10_graph_property_l950_95083

-- Define the logarithm function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the condition for a point to be on the graph of y = log₁₀ x
def on_log10_graph (p : ℝ × ℝ) : Prop :=
  p.2 = log10 p.1

-- State the theorem
theorem log10_graph_property (a b : ℝ) (h1 : on_log10_graph (a, b)) (h2 : a ≠ 1) :
  on_log10_graph (a^2, 2*b) :=
sorry

end NUMINAMATH_CALUDE_log10_graph_property_l950_95083


namespace NUMINAMATH_CALUDE_division_problem_l950_95068

theorem division_problem (divisor : ℕ) : 
  (109 / divisor = 9) ∧ (109 % divisor = 1) → divisor = 12 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l950_95068


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_A_l950_95053

-- Define the sets
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x^2 < 4}

-- State the theorems
theorem union_of_A_and_B : A ∪ B = {x | -2 < x ∧ x ≤ 3} := by sorry

theorem complement_of_A : (Set.univ \ A) = {x | x < -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_A_l950_95053


namespace NUMINAMATH_CALUDE_sum_integers_neg25_to_55_l950_95079

/-- The sum of integers from a to b, inclusive -/
def sum_integers (a b : ℤ) : ℤ := (b - a + 1) * (a + b) / 2

/-- Theorem: The sum of integers from -25 to 55 is 1215 -/
theorem sum_integers_neg25_to_55 : sum_integers (-25) 55 = 1215 := by
  sorry

end NUMINAMATH_CALUDE_sum_integers_neg25_to_55_l950_95079


namespace NUMINAMATH_CALUDE_smallest_c_for_negative_three_in_range_l950_95028

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

-- State the theorem
theorem smallest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (c' : ℝ), (∃ (x : ℝ), f c' x = -3) → c ≤ c') ∧
  (∃ (x : ℝ), f (-3/4) x = -3) :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_negative_three_in_range_l950_95028


namespace NUMINAMATH_CALUDE_valid_n_gon_values_l950_95019

/-- The set of possible n values for a convex n-gon with n-1 angles of 150° and one angle less than 150° -/
def valid_n_gon : Set ℕ :=
  {n : ℕ | n ≥ 3 ∧ 
           (150 * (n - 1) + (30 * n - 210) = 180 * (n - 2)) ∧
           (30 * n - 210 > 0) ∧
           (30 * n - 210 < 150)}

/-- Theorem stating that the valid n values are 8, 9, 10, and 11 -/
theorem valid_n_gon_values : valid_n_gon = {8, 9, 10, 11} := by
  sorry


end NUMINAMATH_CALUDE_valid_n_gon_values_l950_95019


namespace NUMINAMATH_CALUDE_initial_withdrawal_l950_95099

theorem initial_withdrawal (initial_balance : ℚ) : 
  let remaining_balance := initial_balance - (2/5) * initial_balance
  let final_balance := remaining_balance + (1/4) * remaining_balance
  final_balance = 750 →
  (2/5) * initial_balance = 400 := by
sorry

end NUMINAMATH_CALUDE_initial_withdrawal_l950_95099


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l950_95021

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ (a : ℕ) + b ≤ x + y ∧ (a : ℕ) + b = 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l950_95021


namespace NUMINAMATH_CALUDE_sequence_problem_l950_95026

theorem sequence_problem (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n ≤ 3^n) ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 2) - a n ≥ 4 * 3^n) →
  a 2017 = (3^2017 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l950_95026


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l950_95029

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 789 [ZMOD 11]) → n = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l950_95029


namespace NUMINAMATH_CALUDE_vector_equation_solution_l950_95078

/-- Given vectors e₁ and e₂, and real numbers x and y satisfying the equation,
    prove that x - y = -3 -/
theorem vector_equation_solution (e₁ e₂ : ℝ × ℝ) (x y : ℝ) 
    (h₁ : e₁ = (1, 2))
    (h₂ : e₂ = (3, 4))
    (h₃ : x • e₁ + y • e₂ = (5, 6)) :
  x - y = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l950_95078


namespace NUMINAMATH_CALUDE_charlie_original_price_l950_95087

-- Define the given quantities
def alice_acorns : ℕ := 3600
def bob_acorns : ℕ := 2400
def charlie_acorns : ℕ := 4500
def bob_total_price : ℚ := 6000
def discount_rate : ℚ := 0.1

-- Define the relationships
def bob_price_per_acorn : ℚ := bob_total_price / bob_acorns
def alice_price_per_acorn : ℚ := 9 * bob_price_per_acorn
def average_price_per_acorn : ℚ := (alice_price_per_acorn * alice_acorns + bob_price_per_acorn * bob_acorns) / (alice_acorns + bob_acorns)
def charlie_discounted_price_per_acorn : ℚ := average_price_per_acorn * (1 - discount_rate)

-- State the theorem
theorem charlie_original_price : 
  charlie_acorns * average_price_per_acorn = 65250 := by sorry

end NUMINAMATH_CALUDE_charlie_original_price_l950_95087


namespace NUMINAMATH_CALUDE_sum_of_cubes_l950_95047

theorem sum_of_cubes (a b c d e : ℝ) 
  (sum_zero : a + b + c + d + e = 0)
  (sum_products : a*b*c + a*b*d + a*b*e + a*c*d + a*c*e + a*d*e + b*c*d + b*c*e + b*d*e + c*d*e = 2008) :
  a^3 + b^3 + c^3 + d^3 + e^3 = -12048 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l950_95047


namespace NUMINAMATH_CALUDE_perpendicular_tangents_and_inequality_l950_95074

noncomputable def f (x : ℝ) := x^2 + 4*x + 2

noncomputable def g (t x : ℝ) := t * Real.exp x * ((2*x + 4) - 2)

theorem perpendicular_tangents_and_inequality (t k : ℝ) : 
  (((2 * (-17/8) + 4) * (2 * t * Real.exp 0 * (0 + 2)) = -1) ∧
   (∀ x : ℝ, x ≥ 2 → k * g 1 x ≥ 2 * f x)) ↔ 
  (t = 1 ∧ 2 ≤ k ∧ k ≤ 2 * Real.exp 2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_and_inequality_l950_95074


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l950_95033

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^12 + b^12) / (a + b)^9 = -2 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l950_95033


namespace NUMINAMATH_CALUDE_not_always_true_l950_95061

theorem not_always_true (r p q : ℝ) (hr : r > 0) (hpq : p * q ≠ 0) (hpqr : p * r > q * r) :
  ¬((-p > -q) ∨ (-p > q) ∨ (1 > -q/p) ∨ (1 < q/p)) := by
  sorry

end NUMINAMATH_CALUDE_not_always_true_l950_95061


namespace NUMINAMATH_CALUDE_oldest_babysat_prime_age_l950_95098

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def max_babysit_age (current_age : ℕ) (start_age : ℕ) (stop_age : ℕ) (years_since_stop : ℕ) : ℕ :=
  min (stop_age / 2 + years_since_stop) (current_age - 1)

def satisfies_babysit_criteria (age : ℕ) (max_age : ℕ) (gap : ℕ) : Prop :=
  age ≤ max_age ∧ ∃ n : ℕ, n ≤ gap ∧ max_age - n = age

theorem oldest_babysat_prime_age :
  ∀ (current_age : ℕ) (start_age : ℕ) (stop_age : ℕ) (years_since_stop : ℕ),
    current_age = 32 →
    start_age = 20 →
    stop_age = 22 →
    years_since_stop = 10 →
    ∃ (oldest_age : ℕ),
      is_prime oldest_age ∧
      oldest_age = 19 ∧
      satisfies_babysit_criteria oldest_age (max_babysit_age current_age start_age stop_age years_since_stop) 1 ∧
      ∀ (age : ℕ),
        is_prime age →
        satisfies_babysit_criteria age (max_babysit_age current_age start_age stop_age years_since_stop) 1 →
        age ≤ oldest_age :=
by sorry

end NUMINAMATH_CALUDE_oldest_babysat_prime_age_l950_95098


namespace NUMINAMATH_CALUDE_x_pow_zero_eq_one_f_eq_S_l950_95008

-- Define the functions
def f (x : ℝ) := x^2
def S (t : ℝ) := t^2

-- Theorem statements
theorem x_pow_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem f_eq_S : ∀ x : ℝ, f x = S x := by sorry

end NUMINAMATH_CALUDE_x_pow_zero_eq_one_f_eq_S_l950_95008


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l950_95040

theorem gcd_lcm_sum : Nat.gcd 42 56 + Nat.lcm 24 18 = 86 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l950_95040


namespace NUMINAMATH_CALUDE_bargain_bin_books_l950_95031

def total_books (x y : ℕ) (z : ℚ) : ℚ :=
  (x : ℚ) - (y : ℚ) + z / 100 * (x : ℚ)

theorem bargain_bin_books (x y : ℕ) (z : ℚ) :
  total_books x y z = (x : ℚ) - (y : ℚ) + z / 100 * (x : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_bargain_bin_books_l950_95031


namespace NUMINAMATH_CALUDE_max_uncolored_cubes_l950_95043

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular prism --/
def volume (p : RectangularPrism) : ℕ := p.length * p.width * p.height

/-- Calculates the number of interior cubes in a rectangular prism --/
def interiorCubes (p : RectangularPrism) : ℕ :=
  (p.length - 2) * (p.width - 2) * (p.height - 2)

theorem max_uncolored_cubes (p : RectangularPrism) 
  (h_dim : p.length = 8 ∧ p.width = 8 ∧ p.height = 16) 
  (h_vol : volume p = 1024) :
  interiorCubes p = 504 := by
  sorry


end NUMINAMATH_CALUDE_max_uncolored_cubes_l950_95043


namespace NUMINAMATH_CALUDE_no_base_for_square_l950_95094

theorem no_base_for_square (b : ℤ) : b > 4 → ¬∃ (k : ℤ), b^2 + 4*b + 3 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_square_l950_95094


namespace NUMINAMATH_CALUDE_dihedral_angle_definition_inconsistency_l950_95060

/-- Definition of a half-plane --/
def HalfPlane : Type := sorry

/-- Definition of a straight line --/
def StraightLine : Type := sorry

/-- Definition of a spatial figure --/
def SpatialFigure : Type := sorry

/-- Definition of a planar angle --/
def PlanarAngle : Type := sorry

/-- Incorrect definition of a dihedral angle --/
def IncorrectDihedralAngle : Type :=
  {angle : PlanarAngle // ∃ (hp1 hp2 : HalfPlane) (l : StraightLine),
    angle = sorry }

/-- Correct definition of a dihedral angle --/
def CorrectDihedralAngle : Type :=
  {sf : SpatialFigure // ∃ (hp1 hp2 : HalfPlane) (l : StraightLine),
    sf = sorry }

/-- Theorem stating that the incorrect definition is inconsistent with the 3D nature of dihedral angles --/
theorem dihedral_angle_definition_inconsistency :
  ¬(IncorrectDihedralAngle = CorrectDihedralAngle) :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_definition_inconsistency_l950_95060


namespace NUMINAMATH_CALUDE_max_actors_in_tournament_l950_95084

/-- Represents the result of a chess match -/
inductive MatchResult
  | Win
  | Draw
  | Loss

/-- Calculates the score for a given match result -/
def scoreForResult (result : MatchResult) : Rat :=
  match result with
  | MatchResult.Win => 1
  | MatchResult.Draw => 1/2
  | MatchResult.Loss => 0

/-- Represents a chess tournament -/
structure ChessTournament (n : ℕ) where
  /-- The results of all matches in the tournament -/
  results : Fin n → Fin n → MatchResult
  /-- Each player plays exactly one match against each other player -/
  no_self_play : ∀ i, results i i = MatchResult.Draw
  /-- Matches are symmetric: if A wins against B, B loses against A -/
  symmetry : ∀ i j, results i j = MatchResult.Win ↔ results j i = MatchResult.Loss

/-- Calculates the score of player i against player j -/
def score (tournament : ChessTournament n) (i j : Fin n) : Rat :=
  scoreForResult (tournament.results i j)

/-- The tournament satisfies the "1.5 solido" condition -/
def satisfies_condition (tournament : ChessTournament n) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (score tournament i j + score tournament i k = 3/2) ∨
    (score tournament j i + score tournament j k = 3/2) ∨
    (score tournament k i + score tournament k j = 3/2)

/-- The main theorem: the maximum number of actors in a valid tournament is 5 -/
theorem max_actors_in_tournament :
  (∃ (tournament : ChessTournament 5), satisfies_condition tournament) ∧
  (∀ n > 5, ¬∃ (tournament : ChessTournament n), satisfies_condition tournament) :=
sorry

end NUMINAMATH_CALUDE_max_actors_in_tournament_l950_95084


namespace NUMINAMATH_CALUDE_solve_john_age_problem_l950_95034

def john_age_problem (john_current_age : ℕ) (sister_age_multiplier : ℕ) (sister_future_age : ℕ) : Prop :=
  let sister_current_age := john_current_age * sister_age_multiplier
  let age_difference := sister_current_age - john_current_age
  let john_future_age := sister_future_age - age_difference
  john_future_age = sister_future_age - age_difference

theorem solve_john_age_problem :
  john_age_problem 10 2 60 = true :=
sorry

end NUMINAMATH_CALUDE_solve_john_age_problem_l950_95034


namespace NUMINAMATH_CALUDE_binomial_20_19_l950_95063

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end NUMINAMATH_CALUDE_binomial_20_19_l950_95063


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equals_22_l950_95069

-- Define the triangle
def triangle_side_a : ℝ := 5
def triangle_side_b : ℝ := 12
def triangle_side_c : ℝ := 13

-- Define the rectangle
def rectangle_width : ℝ := 5

-- Theorem statement
theorem rectangle_perimeter_equals_22 :
  let triangle_area := (1/2) * triangle_side_a * triangle_side_b
  let rectangle_length := triangle_area / rectangle_width
  2 * (rectangle_width + rectangle_length) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_equals_22_l950_95069


namespace NUMINAMATH_CALUDE_max_roses_for_680_l950_95082

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : ℚ  -- Price of an individual rose
  dozen : ℚ       -- Price of a dozen roses
  twoDozen : ℚ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def maxRoses (budget : ℚ) (pricing : RosePricing) : ℕ :=
  sorry

/-- The specific pricing for the problem -/
def problemPricing : RosePricing :=
  { individual := 730/100,  -- $7.30
    dozen := 36,            -- $36
    twoDozen := 50 }        -- $50

theorem max_roses_for_680 :
  maxRoses 680 problemPricing = 316 :=
sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l950_95082


namespace NUMINAMATH_CALUDE_graph_intersection_symmetry_l950_95004

/-- Given real numbers a, b, c, and d, if the graphs of 
    y = 2a + 1/(x-b) and y = 2c + 1/(x-d) have exactly one common point, 
    then the graphs of y = 2b + 1/(x-a) and y = 2d + 1/(x-c) 
    also have exactly one common point. -/
theorem graph_intersection_symmetry (a b c d : ℝ) :
  (∃! x : ℝ, 2*a + 1/(x-b) = 2*c + 1/(x-d)) →
  (∃! x : ℝ, 2*b + 1/(x-a) = 2*d + 1/(x-c)) :=
by sorry

end NUMINAMATH_CALUDE_graph_intersection_symmetry_l950_95004


namespace NUMINAMATH_CALUDE_inequality_proof_l950_95018

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + b / 3 + c ≥ Real.sqrt ((a + b) * b * (c + a)) ∧
  Real.sqrt ((a + b) * b * (c + a)) ≥ Real.sqrt (a * b) + (Real.sqrt (b * c) + Real.sqrt (c * a)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l950_95018
