import Mathlib

namespace NUMINAMATH_CALUDE_first_discount_percentage_l3161_316111

/-- Given an initial price and a final price after two discounts, 
    where the second discount is known, calculate the first discount percentage. -/
theorem first_discount_percentage 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) 
  (h1 : initial_price = 528)
  (h2 : final_price = 380.16)
  (h3 : second_discount = 0.1)
  : ∃ (first_discount : ℝ),
    first_discount = 0.2 ∧ 
    final_price = initial_price * (1 - first_discount) * (1 - second_discount) :=
by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3161_316111


namespace NUMINAMATH_CALUDE_expression_equals_hundred_l3161_316154

theorem expression_equals_hundred : (7.5 * 7.5 + 37.5 + 2.5 * 2.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_hundred_l3161_316154


namespace NUMINAMATH_CALUDE_triangle_side_length_l3161_316176

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a = 3 →
  C = 2 * π / 3 →  -- 120° in radians
  S = (15 * Real.sqrt 3) / 4 →
  S = (1 / 2) * a * b * Real.sin C →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l3161_316176


namespace NUMINAMATH_CALUDE_slope_product_negative_one_l3161_316141

/-- Two lines with slopes that differ by 45° and are negative reciprocals have a slope product of -1 -/
theorem slope_product_negative_one (m n : ℝ) : 
  (∃ θ : ℝ, m = Real.tan (θ + π/4) ∧ n = Real.tan θ) →  -- L₁ makes 45° larger angle than L₂
  m = -1/n →                                           -- slopes are negative reciprocals
  m * n = -1 :=                                        -- product of slopes is -1
by sorry

end NUMINAMATH_CALUDE_slope_product_negative_one_l3161_316141


namespace NUMINAMATH_CALUDE_smallest_d_l3161_316102

/-- The smallest positive value of d that satisfies the equation √((4√3)² + (d+4)²) = 2d -/
theorem smallest_d : ∃ d : ℝ, d > 0 ∧ 
  (∀ d' : ℝ, d' > 0 → (4 * Real.sqrt 3)^2 + (d' + 4)^2 = (2 * d')^2 → d ≤ d') ∧
  (4 * Real.sqrt 3)^2 + (d + 4)^2 = (2 * d)^2 ∧
  d = (2 * (2 - Real.sqrt 52)) / 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_l3161_316102


namespace NUMINAMATH_CALUDE_function_is_zero_l3161_316170

/-- A function satisfying the given functional equation is the zero function -/
theorem function_is_zero (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x * f y + 2 * x) = x * y + 2 * f x) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_is_zero_l3161_316170


namespace NUMINAMATH_CALUDE_total_trout_caught_l3161_316151

/-- The number of trout caught by Sara, Melanie, and John -/
def total_trout (sara melanie john : ℕ) : ℕ := sara + melanie + john

/-- Theorem stating the total number of trout caught -/
theorem total_trout_caught :
  ∃ (sara melanie john : ℕ),
    sara = 5 ∧
    melanie = 2 * sara ∧
    john = 3 * melanie ∧
    total_trout sara melanie john = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_trout_caught_l3161_316151


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3161_316127

theorem sufficient_not_necessary (p q : Prop) :
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3161_316127


namespace NUMINAMATH_CALUDE_game_probabilities_l3161_316188

/-- Represents the outcome of a single trial -/
inductive Outcome
  | win
  | loss

/-- Represents the result of 4 trials -/
def GameResult := List Outcome

/-- Counts the number of wins in a game result -/
def countWins : GameResult → Nat
  | [] => 0
  | (Outcome.win :: rest) => 1 + countWins rest
  | (Outcome.loss :: rest) => countWins rest

/-- The sample space of all possible game results -/
def sampleSpace : List GameResult := sorry

/-- The probability of an event occurring -/
def probability (event : GameResult → Bool) : Rat :=
  (sampleSpace.filter event).length / sampleSpace.length

/-- Winning at least once -/
def winAtLeastOnce (result : GameResult) : Bool :=
  countWins result ≥ 1

/-- Winning at most twice -/
def winAtMostTwice (result : GameResult) : Bool :=
  countWins result ≤ 2

theorem game_probabilities :
  probability winAtLeastOnce = 5/16 ∧
  probability winAtMostTwice = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_game_probabilities_l3161_316188


namespace NUMINAMATH_CALUDE_adams_shopping_cost_l3161_316113

/-- Calculates the total cost of Adam's shopping, including discount and sales tax -/
def total_cost (sandwich_price : ℚ) (chips_price : ℚ) (water_price : ℚ) 
                (sandwich_count : ℕ) (chips_count : ℕ) (water_count : ℕ) 
                (tax_rate : ℚ) : ℚ :=
  let sandwich_cost := (sandwich_count - 1) * sandwich_price
  let chips_cost := chips_count * chips_price
  let water_cost := water_count * water_price
  let subtotal := sandwich_cost + chips_cost + water_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that Adam's total shopping cost is $29.15 -/
theorem adams_shopping_cost : 
  total_cost 4 3.5 2 4 3 2 0.1 = 29.15 := by
  sorry

end NUMINAMATH_CALUDE_adams_shopping_cost_l3161_316113


namespace NUMINAMATH_CALUDE_cube_root_of_square_64_l3161_316135

theorem cube_root_of_square_64 (x : ℝ) (h : x^2 = 64) :
  x^(1/3) = 2 ∨ x^(1/3) = -2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_square_64_l3161_316135


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l3161_316122

theorem add_preserves_inequality (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l3161_316122


namespace NUMINAMATH_CALUDE_set_equality_l3161_316108

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}

theorem set_equality : (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end NUMINAMATH_CALUDE_set_equality_l3161_316108


namespace NUMINAMATH_CALUDE_max_grandchildren_l3161_316143

/-- Calculates the number of grandchildren for a person with given conditions -/
def grandchildren_count (num_children : ℕ) (num_same_children : ℕ) (num_five_children : ℕ) (five_children : ℕ) : ℕ :=
  (num_same_children * num_children) + (num_five_children * five_children)

/-- Theorem stating that Max has 58 grandchildren -/
theorem max_grandchildren :
  let num_children := 8
  let num_same_children := 6
  let num_five_children := 2
  let five_children := 5
  grandchildren_count num_children num_same_children num_five_children five_children = 58 := by
  sorry

end NUMINAMATH_CALUDE_max_grandchildren_l3161_316143


namespace NUMINAMATH_CALUDE_work_completion_time_l3161_316190

theorem work_completion_time (x_total_days y_completion_days : ℕ) 
  (x_work_days : ℕ) (h1 : x_total_days = 20) (h2 : x_work_days = 10) 
  (h3 : y_completion_days = 12) : 
  (x_total_days * y_completion_days) / (y_completion_days - x_work_days) = 24 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3161_316190


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_four_to_500_l3161_316121

theorem x_one_minus_f_equals_four_to_500 :
  let x : ℝ := (3 + Real.sqrt 5) ^ 500
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 4 ^ 500 := by
  sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_four_to_500_l3161_316121


namespace NUMINAMATH_CALUDE_least_number_divisibility_l3161_316131

theorem least_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 234 → ¬((3072 + m) % 57 = 0 ∧ (3072 + m) % 29 = 0)) ∧ 
  ((3072 + 234) % 57 = 0 ∧ (3072 + 234) % 29 = 0) := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l3161_316131


namespace NUMINAMATH_CALUDE_original_expenditure_l3161_316137

/-- Represents the hostel mess expenditure problem -/
structure HostelMess where
  initial_students : ℕ
  initial_expenditure : ℕ
  initial_avg_expenditure : ℕ

/-- Represents changes in the hostel mess -/
structure MessChange where
  day : ℕ
  student_change : ℤ
  expense_change : ℕ
  avg_expenditure_change : ℤ

/-- Theorem stating the original expenditure of the mess -/
theorem original_expenditure (mess : HostelMess) 
  (change1 : MessChange) (change2 : MessChange) (change3 : MessChange) : 
  mess.initial_students = 35 →
  change1.day = 10 → change1.student_change = 7 → change1.expense_change = 84 → change1.avg_expenditure_change = -1 →
  change2.day = 15 → change2.student_change = -5 → change2.expense_change = 40 → change2.avg_expenditure_change = 2 →
  change3.day = 25 → change3.student_change = 3 → change3.expense_change = 30 → change3.avg_expenditure_change = 0 →
  mess.initial_expenditure = 630 := by
  sorry

end NUMINAMATH_CALUDE_original_expenditure_l3161_316137


namespace NUMINAMATH_CALUDE_no_solution_exists_l3161_316162

theorem no_solution_exists (a b : ℝ) : a^2 + 3*b^2 + 2 > 3*a*b := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3161_316162


namespace NUMINAMATH_CALUDE_complex_number_opposite_parts_l3161_316155

theorem complex_number_opposite_parts (a : ℝ) : 
  let z : ℂ := a / (1 - 2*I) + Complex.abs I
  (Complex.re z = -Complex.im z) → a = -5/3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposite_parts_l3161_316155


namespace NUMINAMATH_CALUDE_solve_marble_problem_l3161_316171

def marble_problem (katrina_marbles : ℕ) : Prop :=
  let amanda_marbles : ℕ := 2 * katrina_marbles - 12
  let mabel_marbles : ℕ := 5 * katrina_marbles
  let carlos_marbles : ℕ := 3 * katrina_marbles
  mabel_marbles = 85 ∧ mabel_marbles - (amanda_marbles + carlos_marbles) = 12

theorem solve_marble_problem :
  ∃ (katrina_marbles : ℕ), marble_problem katrina_marbles := by
  sorry

end NUMINAMATH_CALUDE_solve_marble_problem_l3161_316171


namespace NUMINAMATH_CALUDE_square_49_using_50_l3161_316138

theorem square_49_using_50 : ∃ x : ℕ, 49^2 = 50^2 - x + 1 ∧ x = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_49_using_50_l3161_316138


namespace NUMINAMATH_CALUDE_number_equation_solution_l3161_316178

theorem number_equation_solution : ∃ x : ℝ, 46 + 3 * x = 109 ∧ x = 21 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3161_316178


namespace NUMINAMATH_CALUDE_bernards_blue_notebooks_l3161_316101

/-- Represents the number of notebooks Bernard had -/
structure BernardsNotebooks where
  red : ℕ
  white : ℕ
  blue : ℕ
  given : ℕ
  left : ℕ

/-- Theorem stating the number of blue notebooks Bernard had -/
theorem bernards_blue_notebooks
  (notebooks : BernardsNotebooks)
  (h_red : notebooks.red = 15)
  (h_white : notebooks.white = 19)
  (h_given : notebooks.given = 46)
  (h_left : notebooks.left = 5)
  (h_total : notebooks.red + notebooks.white + notebooks.blue = notebooks.given + notebooks.left) :
  notebooks.blue = 17 := by
  sorry

end NUMINAMATH_CALUDE_bernards_blue_notebooks_l3161_316101


namespace NUMINAMATH_CALUDE_not_always_possible_within_30_moves_l3161_316112

/-- Represents a move on the board -/
inductive Move
  | add_two : Fin 3 → Fin 3 → Move
  | subtract_all : Move

/-- The state of the board -/
def Board := Fin 3 → ℕ

/-- Apply a move to the board -/
def apply_move (b : Board) (m : Move) : Board :=
  match m with
  | Move.add_two i j => fun k => if k = i ∨ k = j then b k + 1 else b k
  | Move.subtract_all => fun k => if b k > 0 then b k - 1 else 0

/-- Check if all numbers on the board are zero -/
def all_zero (b : Board) : Prop := ∀ i, b i = 0

/-- The main theorem -/
theorem not_always_possible_within_30_moves :
  ∃ (initial : Board),
    (∀ i, 1 ≤ initial i ∧ initial i ≤ 9) ∧
    (∀ i j, i ≠ j → initial i ≠ initial j) ∧
    ¬∃ (moves : List Move),
      moves.length ≤ 30 ∧
      all_zero (moves.foldl apply_move initial) :=
by sorry

end NUMINAMATH_CALUDE_not_always_possible_within_30_moves_l3161_316112


namespace NUMINAMATH_CALUDE_number_problem_l3161_316124

theorem number_problem (A B : ℤ) 
  (h1 : A - B = 144) 
  (h2 : A = 3 * B - 14) : 
  A = 223 := by sorry

end NUMINAMATH_CALUDE_number_problem_l3161_316124


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3161_316181

theorem unique_solution_trigonometric_equation :
  ∃! x : Real,
    0 < x ∧ x < 180 ∧
    Real.tan (120 - x) = (Real.sin 120 - Real.sin x) / (Real.cos 120 - Real.cos x) ∧
    x = 100 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3161_316181


namespace NUMINAMATH_CALUDE_min_abs_diff_sqrt_30_l3161_316103

theorem min_abs_diff_sqrt_30 (x : ℤ) : |x - Real.sqrt 30| ≥ |5 - Real.sqrt 30| := by
  sorry

end NUMINAMATH_CALUDE_min_abs_diff_sqrt_30_l3161_316103


namespace NUMINAMATH_CALUDE_function_value_at_two_l3161_316117

/-- Given a function f(x) = x^5 + px^3 + qx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem function_value_at_two (p q : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + p*x^3 + q*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l3161_316117


namespace NUMINAMATH_CALUDE_true_discount_calculation_l3161_316134

/-- Given a present worth and banker's gain, calculate the true discount. -/
theorem true_discount_calculation (PW BG : ℚ) (h1 : PW = 576) (h2 : BG = 16) :
  ∃ TD : ℚ, TD^2 = BG * PW ∧ TD = 96 := by
  sorry

end NUMINAMATH_CALUDE_true_discount_calculation_l3161_316134


namespace NUMINAMATH_CALUDE_britta_winning_strategy_l3161_316133

-- Define the game
def Game (n : ℕ) :=
  n ≥ 5 ∧ Odd n

-- Define Britta's winning condition
def BrittaWins (n x₁ x₂ y₁ y₂ : ℕ) : Prop :=
  (x₁ * x₂ * (x₁ - y₁) * (x₂ - y₂)) ^ ((n - 1) / 2) % n = 1

-- Define Britta's strategy
def BrittaStrategy (n : ℕ) (h : Game n) : Prop :=
  ∀ (x₁ x₂ : ℕ), x₁ < n ∧ x₂ < n ∧ x₁ ≠ x₂ →
  ∃ (y₁ y₂ : ℕ), y₁ < n ∧ y₂ < n ∧ y₁ ≠ y₂ ∧ BrittaWins n x₁ x₂ y₁ y₂

-- Theorem statement
theorem britta_winning_strategy (n : ℕ) (h : Game n) :
  BrittaStrategy n h ↔ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_britta_winning_strategy_l3161_316133


namespace NUMINAMATH_CALUDE_closest_fraction_is_one_sixth_l3161_316104

def medals_won : ℚ := 17 / 100

def possible_fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction_is_one_sixth :
  ∀ x ∈ possible_fractions, x ≠ 1/6 → |medals_won - 1/6| < |medals_won - x| :=
sorry

end NUMINAMATH_CALUDE_closest_fraction_is_one_sixth_l3161_316104


namespace NUMINAMATH_CALUDE_max_sum_with_lcm_constraint_max_sum_with_lcm_constraint_achievable_l3161_316189

theorem max_sum_with_lcm_constraint (m n : ℕ) : 
  m > 0 → n > 0 → m < 500 → n < 500 → Nat.lcm m n = (m - n)^2 → m + n ≤ 840 := by
  sorry

theorem max_sum_with_lcm_constraint_achievable : 
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m < 500 ∧ n < 500 ∧ Nat.lcm m n = (m - n)^2 ∧ m + n = 840 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_lcm_constraint_max_sum_with_lcm_constraint_achievable_l3161_316189


namespace NUMINAMATH_CALUDE_freelancer_earnings_l3161_316195

def calculate_final_amount (initial_amount : ℚ) : ℚ :=
  let first_client_payment := initial_amount / 2
  let second_client_payment := first_client_payment * (1 + 2/5)
  let third_client_payment := 2 * (first_client_payment + second_client_payment)
  let average_first_three := (first_client_payment + second_client_payment + third_client_payment) / 3
  let fourth_client_payment := average_first_three * (1 + 1/10)
  initial_amount + first_client_payment + second_client_payment + third_client_payment + fourth_client_payment

theorem freelancer_earnings (initial_amount : ℚ) :
  initial_amount = 4000 → calculate_final_amount initial_amount = 23680 :=
by sorry

end NUMINAMATH_CALUDE_freelancer_earnings_l3161_316195


namespace NUMINAMATH_CALUDE_same_terminal_side_as_610_degrees_l3161_316191

theorem same_terminal_side_as_610_degrees :
  ∀ θ : ℝ, (∃ k : ℤ, θ = k * 360 + 250) ↔ (∃ n : ℤ, θ = n * 360 + 610) :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_as_610_degrees_l3161_316191


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_9_and_22_l3161_316184

theorem six_digit_divisible_by_9_and_22 : ∃! n : ℕ, 
  220140 ≤ n ∧ n < 220150 ∧ 
  n % 9 = 0 ∧ 
  n % 22 = 0 ∧
  n = 520146 := by
sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_9_and_22_l3161_316184


namespace NUMINAMATH_CALUDE_alvin_egg_rolls_l3161_316196

/-- Given the egg roll consumption of Matthew, Patrick, and Alvin, prove that Alvin ate 4 egg rolls. -/
theorem alvin_egg_rolls (matthew patrick alvin : ℕ) : 
  matthew = 3 * patrick →  -- Matthew eats three times as many egg rolls as Patrick
  patrick = alvin / 2 →    -- Patrick eats half as many egg rolls as Alvin
  matthew = 6 →            -- Matthew ate 6 egg rolls
  alvin = 4 := by           -- Prove that Alvin ate 4 egg rolls
sorry

end NUMINAMATH_CALUDE_alvin_egg_rolls_l3161_316196


namespace NUMINAMATH_CALUDE_range_of_a_l3161_316199

def f : Set ℝ → Set ℝ := sorry

def A : Set ℝ := {x | ∃ y ∈ Set.Icc 7 15, f {y} = {2 * x + 1}}

def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ A ∨ x ∈ B a) ↔ 3 ≤ a ∧ a < 6 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3161_316199


namespace NUMINAMATH_CALUDE_kg_to_tons_conversion_l3161_316150

theorem kg_to_tons_conversion (kg_per_ton : ℕ) (h : kg_per_ton = 1000) :
  (3600 - 600) / kg_per_ton = 3 := by
  sorry

end NUMINAMATH_CALUDE_kg_to_tons_conversion_l3161_316150


namespace NUMINAMATH_CALUDE_G_n_planarity_l3161_316114

/-- A graph G_n where vertices are integers from 1 to n -/
def G_n (n : ℕ) := {v : ℕ // v ≤ n}

/-- Two vertices are connected if and only if their sum is prime -/
def connected (n : ℕ) (a b : G_n n) : Prop :=
  Nat.Prime (a.val + b.val)

/-- The graph G_n is planar -/
def is_planar (n : ℕ) : Prop :=
  ∃ (f : G_n n → ℝ × ℝ), ∀ (a b c d : G_n n),
    a ≠ b ∧ c ≠ d ∧ connected n a b ∧ connected n c d →
    (f a ≠ f c ∨ f b ≠ f d) ∧ (f a ≠ f d ∨ f b ≠ f c)

/-- The main theorem: G_n is planar if and only if n ≤ 8 -/
theorem G_n_planarity (n : ℕ) : is_planar n ↔ n ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_G_n_planarity_l3161_316114


namespace NUMINAMATH_CALUDE_free_throws_stats_l3161_316105

def free_throws : List ℝ := [20, 12, 22, 25, 10, 16, 15, 12, 30, 10]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem free_throws_stats :
  median free_throws = 15.5 ∧ mean free_throws = 17.2 := by sorry

end NUMINAMATH_CALUDE_free_throws_stats_l3161_316105


namespace NUMINAMATH_CALUDE_solve_clubsuit_equation_l3161_316160

-- Define the ♣ operation
def clubsuit (A B : ℝ) : ℝ := 3 * A^2 + 2 * B + 7

-- State the theorem
theorem solve_clubsuit_equation :
  ∃ A : ℝ, (clubsuit A 7 = 61) ∧ (A = 2 * Real.sqrt 30 / 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_clubsuit_equation_l3161_316160


namespace NUMINAMATH_CALUDE_count_valid_pairs_l3161_316177

def is_valid_pair (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 23 ∧ 1 ≤ y ∧ y ≤ 23 ∧ (x^2 + y^2 + x + y) % 6 = 0

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ S ↔ is_valid_pair p.1 p.2) ∧ S.card = 225 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l3161_316177


namespace NUMINAMATH_CALUDE_cheese_options_correct_l3161_316173

/-- Represents the number of cheese options -/
def cheese_options : ℕ := 3

/-- Represents the number of meat options -/
def meat_options : ℕ := 4

/-- Represents the number of vegetable options -/
def vegetable_options : ℕ := 5

/-- Represents the total number of topping combinations -/
def total_combinations : ℕ := 57

/-- Theorem stating that the number of cheese options is correct -/
theorem cheese_options_correct : 
  cheese_options * (meat_options * (vegetable_options - 1) + 
  (meat_options - 1) * vegetable_options) = total_combinations :=
sorry

end NUMINAMATH_CALUDE_cheese_options_correct_l3161_316173


namespace NUMINAMATH_CALUDE_perpendicular_construction_l3161_316119

-- Define the plane
structure Plane :=
  (Point : Type)
  (Line : Type)
  (on_line : Point → Line → Prop)
  (not_on_line : Point → Line → Prop)
  (draw_line : Point → Point → Line)
  (draw_perpendicular : Point → Line → Line)

-- Define the theorem
theorem perpendicular_construction 
  (P : Plane) (A : P.Point) (l : P.Line) (h : P.not_on_line A l) :
  ∃ (m : P.Line), P.on_line A m ∧ ∀ (X : P.Point), P.on_line X l → P.on_line X m → 
    ∃ (n : P.Line), P.on_line X n ∧ (∀ (Y : P.Point), P.on_line Y n → P.on_line Y m → Y = X) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_construction_l3161_316119


namespace NUMINAMATH_CALUDE_cubic_three_monotonic_intervals_l3161_316159

/-- A cubic function with a linear term -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x

/-- The derivative of f -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem cubic_three_monotonic_intervals (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f_deriv a x = 0 ∧ f_deriv a y = 0) ↔ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_three_monotonic_intervals_l3161_316159


namespace NUMINAMATH_CALUDE_circles_intersection_range_l3161_316164

/-- Two circles C₁ and C₂ defined by their equations -/
def C₁ (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*m*x + m^2 - 4 = 0
def C₂ (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*m*y + 4*m^2 - 8 = 0

/-- The condition for two circles to intersect -/
def circles_intersect (m : ℝ) : Prop :=
  ∃ x y : ℝ, C₁ m x y ∧ C₂ m x y

/-- The theorem stating the range of m for which the circles intersect -/
theorem circles_intersection_range :
  ∀ m : ℝ, circles_intersect m ↔ (-12/5 < m ∧ m < -2/5) ∨ (0 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_range_l3161_316164


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3161_316185

/-- Given a quadratic function y = ax² + bx - 1 where a ≠ 0, 
    if the graph passes through the point (1, 1), then a + b + 1 = 3 -/
theorem quadratic_function_theorem (a b : ℝ) (ha : a ≠ 0) :
  (a * 1^2 + b * 1 - 1 = 1) → (a + b + 1 = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3161_316185


namespace NUMINAMATH_CALUDE_estate_division_valid_l3161_316120

/-- Represents the estate division problem in Ancient Rome --/
structure EstateDivision where
  total_estate : ℕ
  son_share : ℕ
  daughter_share : ℕ
  wife_share : ℕ

/-- Checks if the given division is valid according to the problem constraints --/
def is_valid_division (d : EstateDivision) : Prop :=
  d.total_estate = 210 ∧
  d.son_share + d.daughter_share + d.wife_share = d.total_estate ∧
  d.son_share > d.daughter_share ∧
  d.son_share > d.wife_share ∧
  7 * d.son_share = 4 * d.total_estate ∧
  7 * d.daughter_share = d.total_estate ∧
  7 * d.wife_share = 2 * d.total_estate

/-- The proposed solution satisfies the constraints of the problem --/
theorem estate_division_valid : 
  is_valid_division ⟨210, 120, 30, 60⟩ := by
  sorry

#check estate_division_valid

end NUMINAMATH_CALUDE_estate_division_valid_l3161_316120


namespace NUMINAMATH_CALUDE_sqrt_24_times_sqrt_3_over_2_equals_6_l3161_316140

theorem sqrt_24_times_sqrt_3_over_2_equals_6 :
  Real.sqrt 24 * Real.sqrt (3/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_24_times_sqrt_3_over_2_equals_6_l3161_316140


namespace NUMINAMATH_CALUDE_xyz_value_l3161_316161

theorem xyz_value (x y z s : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 12)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 0)
  (h3 : x + y + z = s) :
  x * y * z = -8 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3161_316161


namespace NUMINAMATH_CALUDE_divisors_of_48n5_l3161_316126

/-- Given a positive integer n where 132n^3 has 132 positive integer divisors,
    48n^5 has 105 positive integer divisors -/
theorem divisors_of_48n5 (n : ℕ+) (h : (Nat.divisors (132 * n ^ 3)).card = 132) :
  (Nat.divisors (48 * n ^ 5)).card = 105 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_48n5_l3161_316126


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l3161_316193

/-- For the equation 7x^2-(m+13)x+m^2-m-2=0 to have one root greater than 1 
    and one root less than 1, m must satisfy -2 < m < 4 -/
theorem quadratic_root_condition (m : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ > 1 ∧ x₂ < 1 ∧ 
    7 * x₁^2 - (m + 13) * x₁ + m^2 - m - 2 = 0 ∧
    7 * x₂^2 - (m + 13) * x₂ + m^2 - m - 2 = 0) ↔
  -2 < m ∧ m < 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l3161_316193


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l3161_316115

theorem polynomial_equation_solution (x : ℝ) : 
  let q : ℝ → ℝ := λ t => 12 * t^3 - 4
  q (x^3) - q (x^3 - 4) = (q x)^2 + 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l3161_316115


namespace NUMINAMATH_CALUDE_problem_solution_l3161_316197

/-- If 2a - b + 3 = 0, then 2(2a + b) - 4b = -6 -/
theorem problem_solution (a b : ℝ) (h : 2*a - b + 3 = 0) : 
  2*(2*a + b) - 4*b = -6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3161_316197


namespace NUMINAMATH_CALUDE_gwen_science_problems_l3161_316106

/-- Given information about Gwen's homework problems, prove that she had 11 science problems. -/
theorem gwen_science_problems
  (math_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_problems : ℕ)
  (h1 : math_problems = 18)
  (h2 : finished_problems = 24)
  (h3 : remaining_problems = 5) :
  finished_problems + remaining_problems - math_problems = 11 :=
by sorry

end NUMINAMATH_CALUDE_gwen_science_problems_l3161_316106


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3161_316149

theorem rectangle_circle_area_ratio :
  ∀ (b r : ℝ),
  b > 0 →
  r > 0 →
  6 * b = 2 * Real.pi * r →
  (2 * b^2) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3161_316149


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l3161_316186

theorem expression_equals_negative_one (a x : ℝ) (ha : a ≠ 0) (hx1 : x ≠ a) (hx2 : x ≠ -2*a) :
  (((a / (2*a + x)) - (x / (a - x))) / ((x / (2*a + x)) + (a / (a - x)))) = -1 ↔ x = a / 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l3161_316186


namespace NUMINAMATH_CALUDE_square_formation_possible_l3161_316116

theorem square_formation_possible (figure_area : ℕ) (h : figure_area = 4) :
  ∃ (n : ℕ), n > 0 ∧ (n * n) % figure_area = 0 :=
sorry

end NUMINAMATH_CALUDE_square_formation_possible_l3161_316116


namespace NUMINAMATH_CALUDE_prime_sum_special_equation_l3161_316132

theorem prime_sum_special_equation (p q : ℕ) : 
  Prime p → Prime q → q^5 - 2*p^2 = 1 → p + q = 14 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_special_equation_l3161_316132


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3161_316123

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3161_316123


namespace NUMINAMATH_CALUDE_cistern_emptying_l3161_316174

/-- If a pipe can empty 3/4 of a cistern in 12 minutes, then it will empty 1/2 of the cistern in 8 minutes. -/
theorem cistern_emptying (empty_rate : ℚ) (empty_time : ℕ) (target_time : ℕ) :
  empty_rate = 3/4 ∧ empty_time = 12 ∧ target_time = 8 →
  (target_time : ℚ) * (empty_rate / empty_time) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cistern_emptying_l3161_316174


namespace NUMINAMATH_CALUDE_selectStudents_eq_30_l3161_316129

/-- The number of ways to select 3 students from 4 boys and 3 girls, ensuring both genders are represented -/
def selectStudents : ℕ :=
  Nat.choose 4 2 * Nat.choose 3 1 + Nat.choose 4 1 * Nat.choose 3 2

/-- Theorem stating that the number of selections is 30 -/
theorem selectStudents_eq_30 : selectStudents = 30 := by
  sorry

end NUMINAMATH_CALUDE_selectStudents_eq_30_l3161_316129


namespace NUMINAMATH_CALUDE_male_response_rate_change_l3161_316109

/- Define the survey data structure -/
structure SurveyData where
  totalCustomers : ℕ
  malePercentage : ℚ
  femalePercentage : ℚ
  totalResponses : ℕ
  maleResponsePercentage : ℚ
  femaleResponsePercentage : ℚ

/- Define the surveys -/
def initialSurvey : SurveyData :=
  { totalCustomers := 100
  , malePercentage := 60 / 100
  , femalePercentage := 40 / 100
  , totalResponses := 10
  , maleResponsePercentage := 50 / 100
  , femaleResponsePercentage := 50 / 100 }

def finalSurvey : SurveyData :=
  { totalCustomers := 90
  , malePercentage := 50 / 100
  , femalePercentage := 50 / 100
  , totalResponses := 27
  , maleResponsePercentage := 30 / 100
  , femaleResponsePercentage := 70 / 100 }

/- Calculate male response rate -/
def maleResponseRate (survey : SurveyData) : ℚ :=
  (survey.maleResponsePercentage * survey.totalResponses) /
  (survey.malePercentage * survey.totalCustomers)

/- Calculate percentage change -/
def percentageChange (initial : ℚ) (final : ℚ) : ℚ :=
  ((final - initial) / initial) * 100

/- Theorem statement -/
theorem male_response_rate_change :
  percentageChange (maleResponseRate initialSurvey) (maleResponseRate finalSurvey) = 113.4 := by
  sorry

end NUMINAMATH_CALUDE_male_response_rate_change_l3161_316109


namespace NUMINAMATH_CALUDE_tan_eq_two_solution_set_l3161_316175

theorem tan_eq_two_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + (-1)^k * Real.arctan 2} =
  {x : ℝ | Real.tan x = 2} := by sorry

end NUMINAMATH_CALUDE_tan_eq_two_solution_set_l3161_316175


namespace NUMINAMATH_CALUDE_flour_weight_relation_l3161_316125

/-- Theorem: Given two equations representing the weight of flour bags, 
    prove that the new combined weight is equal to the original weight plus 33 pounds. -/
theorem flour_weight_relation (x y : ℝ) : 
  y = (16 - 4) + (30 - 6) + (x - 3) → 
  y = 12 + 24 + (x - 3) → 
  y = x + 33 := by
  sorry

end NUMINAMATH_CALUDE_flour_weight_relation_l3161_316125


namespace NUMINAMATH_CALUDE_equation_proof_l3161_316128

theorem equation_proof : 289 + 2 * 17 * 4 + 16 = 441 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3161_316128


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l3161_316166

theorem ratio_equation_solution : 
  let x : ℚ := 7 / 15
  (3 : ℚ) / 5 / ((6 : ℚ) / 7) = x / ((2 : ℚ) / 3) :=
by sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l3161_316166


namespace NUMINAMATH_CALUDE_perfect_square_mod_three_l3161_316148

theorem perfect_square_mod_three (n : ℤ) : 
  (∃ k : ℤ, n = k^2) → (n % 3 = 0 ∨ n % 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_mod_three_l3161_316148


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l3161_316187

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, 
  n = 104 ∧ 
  n % 13 = 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  ∀ m : ℕ, (m % 13 = 0 ∧ 100 ≤ m ∧ m < 1000) → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l3161_316187


namespace NUMINAMATH_CALUDE_inequality_proof_l3161_316147

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3161_316147


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3161_316198

/-- The amount of money Rachel and Sarah had when they left home -/
def initial_money : ℝ := 50

/-- The amount spent on gasoline -/
def gasoline_cost : ℝ := 8

/-- The amount spent on lunch -/
def lunch_cost : ℝ := 15.65

/-- The amount spent on gifts for grandma (per person) -/
def gift_cost : ℝ := 5

/-- The amount received from grandma (per person) -/
def grandma_gift : ℝ := 10

/-- The amount of money they have for the return trip -/
def return_trip_money : ℝ := 36.35

/-- The number of people (Rachel and Sarah) -/
def num_people : ℕ := 2

theorem initial_money_calculation :
  initial_money = 
    return_trip_money + 
    (gasoline_cost + lunch_cost + num_people * gift_cost) - 
    (num_people * grandma_gift) := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l3161_316198


namespace NUMINAMATH_CALUDE_clock_chimes_theorem_l3161_316180

/-- Represents the number of chimes at a given hour -/
def chimes_at_hour (hour : ℕ) : ℕ := hour

/-- Represents the time taken for a given number of chimes -/
def time_for_chimes (chimes : ℕ) : ℕ :=
  if chimes ≤ 1 then chimes else chimes - 1 + 1

/-- The theorem statement -/
theorem clock_chimes_theorem (hour : ℕ) (chimes : ℕ) (time : ℕ) 
  (h1 : hour = 2 → chimes = 2)
  (h2 : hour = 2 → time = 2)
  (h3 : hour = 12 → chimes = 12) :
  hour = 12 → time_for_chimes chimes = 12 := by
  sorry

#check clock_chimes_theorem

end NUMINAMATH_CALUDE_clock_chimes_theorem_l3161_316180


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l3161_316158

theorem power_of_three_mod_five : 3^2040 ≡ 1 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l3161_316158


namespace NUMINAMATH_CALUDE_vector_magnitude_l3161_316118

theorem vector_magnitude (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2 = 5) →
  ((a.1 - b.1)^2 + (a.2 - b.2)^2 = 20) →
  (b.1^2 + b.2^2 = 25) := by
    sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3161_316118


namespace NUMINAMATH_CALUDE_streetlight_distance_l3161_316157

/-- The distance between streetlights in meters -/
def interval : ℝ := 60

/-- The number of streetlights -/
def num_streetlights : ℕ := 45

/-- The distance from the first to the last streetlight in kilometers -/
def distance_km : ℝ := 2.64

theorem streetlight_distance :
  (interval * (num_streetlights - 1)) / 1000 = distance_km := by
  sorry

end NUMINAMATH_CALUDE_streetlight_distance_l3161_316157


namespace NUMINAMATH_CALUDE_prime_pairs_perfect_square_l3161_316182

theorem prime_pairs_perfect_square :
  ∀ a b : ℕ,
  Prime a → Prime b → a > 0 → b > 0 →
  (∃ k : ℕ, 3 * a^2 * b + 16 * a * b^2 = k^2) →
  ((a = 19 ∧ b = 19) ∨ (a = 2 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_prime_pairs_perfect_square_l3161_316182


namespace NUMINAMATH_CALUDE_calcium_bromide_weight_l3161_316146

/-- The atomic weight of calcium in g/mol -/
def calcium_weight : ℝ := 40.08

/-- The atomic weight of bromine in g/mol -/
def bromine_weight : ℝ := 79.904

/-- The number of moles of calcium bromide -/
def moles : ℝ := 4

/-- The molecular weight of calcium bromide (CaBr2) in g/mol -/
def molecular_weight_CaBr2 : ℝ := calcium_weight + 2 * bromine_weight

/-- The total weight of the given number of moles of calcium bromide in grams -/
def total_weight : ℝ := moles * molecular_weight_CaBr2

theorem calcium_bromide_weight : total_weight = 799.552 := by
  sorry

end NUMINAMATH_CALUDE_calcium_bromide_weight_l3161_316146


namespace NUMINAMATH_CALUDE_area_to_paint_dining_room_l3161_316183

/-- The area to be painted on a wall with a painting hanging on it -/
def area_to_paint (wall_height wall_length painting_height painting_length : ℝ) : ℝ :=
  wall_height * wall_length - painting_height * painting_length

/-- Theorem: The area to be painted is 135 square feet -/
theorem area_to_paint_dining_room : 
  area_to_paint 10 15 3 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_dining_room_l3161_316183


namespace NUMINAMATH_CALUDE_stratum_c_sample_size_l3161_316192

/-- Calculates the sample size for a stratum in stratified sampling -/
def stratumSampleSize (stratumSize : ℕ) (totalPopulation : ℕ) (totalSampleSize : ℕ) : ℕ :=
  (stratumSize * totalSampleSize) / totalPopulation

theorem stratum_c_sample_size :
  let stratum_a_size : ℕ := 400
  let stratum_b_size : ℕ := 800
  let stratum_c_size : ℕ := 600
  let total_population : ℕ := stratum_a_size + stratum_b_size + stratum_c_size
  let total_sample_size : ℕ := 90
  stratumSampleSize stratum_c_size total_population total_sample_size = 30 := by
  sorry

#eval stratumSampleSize 600 1800 90

end NUMINAMATH_CALUDE_stratum_c_sample_size_l3161_316192


namespace NUMINAMATH_CALUDE_linear_function_proof_l3161_316142

/-- A linear function passing through three given points -/
def linear_function (x : ℝ) : ℝ := 3 * x + 4

/-- Theorem stating that the linear function passes through the given points and f(40) = 124 -/
theorem linear_function_proof :
  (linear_function 2 = 10) ∧
  (linear_function 6 = 22) ∧
  (linear_function 10 = 34) ∧
  (linear_function 40 = 124) := by
  sorry

#check linear_function_proof

end NUMINAMATH_CALUDE_linear_function_proof_l3161_316142


namespace NUMINAMATH_CALUDE_gcd_and_bezout_identity_l3161_316179

theorem gcd_and_bezout_identity :
  ∃ (d u v : ℤ), Int.gcd 663 182 = d ∧ d = 663 * u + 182 * v ∧ d = 13 :=
by sorry

end NUMINAMATH_CALUDE_gcd_and_bezout_identity_l3161_316179


namespace NUMINAMATH_CALUDE_number_squared_sum_equals_100_l3161_316145

theorem number_squared_sum_equals_100 : ∃ x : ℝ, (7.5 * 7.5) + 37.5 + (x * x) = 100 ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_number_squared_sum_equals_100_l3161_316145


namespace NUMINAMATH_CALUDE_dans_initial_cards_l3161_316165

/-- The number of baseball cards Dan had initially -/
def initial_cards : ℕ := sorry

/-- The number of torn cards -/
def torn_cards : ℕ := 8

/-- The number of cards Sam bought -/
def cards_sold : ℕ := 15

/-- The number of cards Dan has after selling to Sam -/
def remaining_cards : ℕ := 82

theorem dans_initial_cards : initial_cards = 105 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_cards_l3161_316165


namespace NUMINAMATH_CALUDE_complex_power_result_l3161_316153

theorem complex_power_result (n : ℕ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + 3)^n = 256) : (1 + i : ℂ)^n = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_result_l3161_316153


namespace NUMINAMATH_CALUDE_odd_function_theorem_l3161_316144

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_theorem (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_nonneg : ∀ x ≥ 0, f x = x^2 - 2*x) : 
  ∀ x, f x = x * (|x| - 2) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_theorem_l3161_316144


namespace NUMINAMATH_CALUDE_train_length_l3161_316110

theorem train_length (platform1_length platform2_length : ℝ)
                     (platform1_time platform2_time : ℝ)
                     (h1 : platform1_length = 150)
                     (h2 : platform2_length = 250)
                     (h3 : platform1_time = 15)
                     (h4 : platform2_time = 20) :
  ∃ train_length : ℝ,
    train_length = 150 ∧
    (train_length + platform1_length) / platform1_time =
    (train_length + platform2_length) / platform2_time :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_l3161_316110


namespace NUMINAMATH_CALUDE_area_30_60_90_triangle_l3161_316194

theorem area_30_60_90_triangle (a : ℝ) (h : a > 0) :
  let triangle_area := (1/2) * a * (a / Real.sqrt 3)
  triangle_area = (32 * Real.sqrt 3) / 3 ↔ a = 8 := by
sorry

end NUMINAMATH_CALUDE_area_30_60_90_triangle_l3161_316194


namespace NUMINAMATH_CALUDE_dad_took_90_steps_l3161_316156

/-- The number of steps Dad takes for every 5 steps Masha takes -/
def dad_steps : ℕ := 3

/-- The number of steps Masha takes for every 5 steps Yasha takes -/
def masha_steps : ℕ := 3

/-- The total number of steps Masha and Yasha took together -/
def total_steps : ℕ := 400

/-- Theorem stating that Dad took 90 steps -/
theorem dad_took_90_steps : 
  ∃ (d m y : ℕ), 
    d * 5 = m * dad_steps ∧ 
    m * 5 = y * masha_steps ∧ 
    m + y = total_steps ∧ 
    d = 90 := by sorry

end NUMINAMATH_CALUDE_dad_took_90_steps_l3161_316156


namespace NUMINAMATH_CALUDE_apple_count_l3161_316168

theorem apple_count (red : ℕ) (green : ℕ) : 
  red = 16 → green = red + 12 → red + green = 44 := by
  sorry

end NUMINAMATH_CALUDE_apple_count_l3161_316168


namespace NUMINAMATH_CALUDE_z_equals_four_when_x_is_five_l3161_316107

/-- The inverse relationship between 7z and x² -/
def inverse_relation (z x : ℝ) : Prop := ∃ k : ℝ, 7 * z = k / (x ^ 2)

/-- The theorem stating that given the inverse relationship and initial condition, z = 4 when x = 5 -/
theorem z_equals_four_when_x_is_five :
  ∀ z₀ : ℝ, inverse_relation z₀ 2 ∧ z₀ = 25 →
  ∃ z : ℝ, inverse_relation z 5 ∧ z = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_z_equals_four_when_x_is_five_l3161_316107


namespace NUMINAMATH_CALUDE_min_games_for_2015_scores_l3161_316172

/-- Represents the scoring system for a football league -/
structure ScoringSystem where
  a : ℝ  -- Points for a win
  b : ℝ  -- Points for a draw
  h : a > b ∧ b > 0

/-- Calculates the number of possible scores after n games -/
def possibleScores (s : ScoringSystem) (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of games for 2015 possible scores -/
theorem min_games_for_2015_scores (s : ScoringSystem) :
  (∀ m : ℕ, m < 62 → possibleScores s m < 2015) ∧
  possibleScores s 62 = 2015 :=
sorry

end NUMINAMATH_CALUDE_min_games_for_2015_scores_l3161_316172


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l3161_316169

/-- Given a geometric sequence with first term a₁ = 2, 
    the smallest possible value of 6a₂ + 7a₃ is -18/7 -/
theorem min_value_geometric_sequence (a₁ a₂ a₃ : ℝ) : 
  a₁ = 2 → 
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) → 
  (∀ b₂ b₃ : ℝ, (∃ s : ℝ, b₂ = a₁ * s ∧ b₃ = b₂ * s) → 
    6 * a₂ + 7 * a₃ ≤ 6 * b₂ + 7 * b₃) → 
  6 * a₂ + 7 * a₃ = -18/7 :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l3161_316169


namespace NUMINAMATH_CALUDE_oranges_harvested_per_day_l3161_316152

theorem oranges_harvested_per_day :
  let total_sacks : ℕ := 56
  let total_days : ℕ := 14
  let sacks_per_day : ℕ := total_sacks / total_days
  sacks_per_day = 4 := by sorry

end NUMINAMATH_CALUDE_oranges_harvested_per_day_l3161_316152


namespace NUMINAMATH_CALUDE_min_desks_for_arrangements_l3161_316136

/-- The number of students to be seated -/
def num_students : ℕ := 2

/-- The number of different seating arrangements -/
def num_arrangements : ℕ := 2

/-- The minimum number of empty desks between students -/
def min_empty_desks : ℕ := 1

/-- A function that calculates the number of valid seating arrangements
    given the number of desks -/
def valid_arrangements (num_desks : ℕ) : ℕ := sorry

/-- Theorem stating that 5 is the minimum number of desks required -/
theorem min_desks_for_arrangements :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ m : ℕ, m < n → valid_arrangements m < num_arrangements) ∧
  valid_arrangements n = num_arrangements :=
sorry

end NUMINAMATH_CALUDE_min_desks_for_arrangements_l3161_316136


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l3161_316163

/-- The capacity of a tank with specific inlet and outlet pipe characteristics -/
def tank_capacity : ℝ := 1280

/-- The time it takes for the outlet pipe to empty the full tank -/
def outlet_time : ℝ := 10

/-- The rate at which the inlet pipe fills the tank in litres per minute -/
def inlet_rate : ℝ := 8

/-- The additional time it takes to empty the tank when the inlet pipe is open -/
def additional_time : ℝ := 6

theorem tank_capacity_proof :
  tank_capacity = outlet_time * inlet_rate * 60 * (outlet_time + additional_time) / additional_time :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l3161_316163


namespace NUMINAMATH_CALUDE_toy_poodle_height_l3161_316130

/-- Heights of different poodle types -/
structure PoodleHeights where
  standard : ℝ
  miniature : ℝ
  toy : ℝ
  moyen : ℝ

/-- Conditions for poodle heights -/
def valid_poodle_heights (h : PoodleHeights) : Prop :=
  h.standard = h.miniature + 8.5 ∧
  h.miniature = h.toy + 6.25 ∧
  h.standard = h.moyen + 3.75 ∧
  h.moyen = h.toy + 4.75 ∧
  h.standard = 28

/-- Theorem: The toy poodle's height is 13.25 inches -/
theorem toy_poodle_height (h : PoodleHeights) 
  (hvalid : valid_poodle_heights h) : h.toy = 13.25 := by
  sorry

end NUMINAMATH_CALUDE_toy_poodle_height_l3161_316130


namespace NUMINAMATH_CALUDE_cone_volume_over_pi_l3161_316100

/-- Given a cone formed from a 240-degree sector of a circle with radius 16,
    prove that the volume of the cone divided by π is equal to 8192√10 / 81. -/
theorem cone_volume_over_pi (r : ℝ) (h : ℝ) :
  r = 32 / 3 →
  h = 8 * Real.sqrt 10 / 3 →
  (1 / 3 * π * r^2 * h) / π = 8192 * Real.sqrt 10 / 81 := by sorry

end NUMINAMATH_CALUDE_cone_volume_over_pi_l3161_316100


namespace NUMINAMATH_CALUDE_election_votes_l3161_316167

theorem election_votes :
  ∀ (V : ℕ) (geoff_votes : ℕ),
    geoff_votes = V / 100 →                     -- Geoff received 1% of votes
    geoff_votes + 3000 > V * 51 / 100 →         -- With 3000 more votes, Geoff would win
    geoff_votes + 3000 ≤ V * 51 / 100 + 1 →     -- Geoff needed exactly 3000 more votes to win
    V = 6000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l3161_316167


namespace NUMINAMATH_CALUDE_f_unique_zero_and_inequality_l3161_316139

noncomputable section

variable (a : ℝ)

def f (x : ℝ) := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

def g (x : ℝ) := a * Real.exp x + x

theorem f_unique_zero_and_inequality (h : a ≥ 0) :
  (∃! x, f a x = 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ > -1 → x₂ > -1 → f a x₁ = g a x₁ - g a x₂ → x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2) :=
sorry

end

end NUMINAMATH_CALUDE_f_unique_zero_and_inequality_l3161_316139
