import Mathlib

namespace NUMINAMATH_CALUDE_distribute_cards_count_l275_27563

/-- The number of ways to distribute 6 cards into 3 envelopes -/
def distribute_cards : ℕ :=
  let n_cards := 6
  let n_envelopes := 3
  let cards_per_envelope := 2
  let ways_to_place_1_and_2 := n_envelopes
  let remaining_cards := n_cards - cards_per_envelope
  let ways_to_distribute_remaining := 6  -- This is a given fact from the problem
  ways_to_place_1_and_2 * ways_to_distribute_remaining

/-- Theorem stating that the number of ways to distribute the cards is 18 -/
theorem distribute_cards_count : distribute_cards = 18 := by
  sorry

end NUMINAMATH_CALUDE_distribute_cards_count_l275_27563


namespace NUMINAMATH_CALUDE_medium_size_can_be_rational_l275_27523

-- Define the popcorn sizes
structure PopcornSize where
  name : String
  amount : Nat
  price : Nat

-- Define the customer's preferences
structure CustomerPreferences where
  budget : Nat
  wantsDrink : Bool
  preferBalancedMeal : Bool

-- Define the utility function
def utility (choice : PopcornSize) (prefs : CustomerPreferences) : Nat :=
  sorry

-- Define the theorem
theorem medium_size_can_be_rational (small medium large : PopcornSize) 
  (prefs : CustomerPreferences) : 
  small.name = "small" → 
  small.amount = 50 → 
  small.price = 200 →
  medium.name = "medium" → 
  medium.amount = 70 → 
  medium.price = 400 →
  large.name = "large" → 
  large.amount = 130 → 
  large.price = 500 →
  prefs.budget = 500 →
  prefs.wantsDrink = true →
  prefs.preferBalancedMeal = true →
  ∃ (drink_price : Nat), 
    utility medium prefs + utility (PopcornSize.mk "drink" 0 drink_price) prefs ≥ 
    max (utility small prefs) (utility large prefs) :=
  sorry


end NUMINAMATH_CALUDE_medium_size_can_be_rational_l275_27523


namespace NUMINAMATH_CALUDE_residue_calculation_l275_27535

theorem residue_calculation : (230 * 15 - 20 * 9 + 5) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l275_27535


namespace NUMINAMATH_CALUDE_hyperbola_equation_l275_27543

noncomputable section

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 3 * x

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -Real.sqrt 3

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 2 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define points A and B
def A : ℝ × ℝ := (-Real.sqrt 3, 2)
def B : ℝ × ℝ := (-Real.sqrt 3, -2)

-- Define the property of equilateral triangle
def is_equilateral_triangle (F A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2 ∧
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, hyperbola a b x y ↔ parabola x y) →
  (∀ x, directrix x → ∃ y, hyperbola a b x y) →
  (∀ x y, asymptote x y → hyperbola a b x y) →
  is_equilateral_triangle focus A B →
  (∀ x y, hyperbola a b x y ↔ x^2 - y^2 / 2 = 1) :=
sorry

end

end NUMINAMATH_CALUDE_hyperbola_equation_l275_27543


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_one_l275_27525

theorem negation_of_forall_positive_square_plus_one :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_one_l275_27525


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l275_27577

/-- The maximum value of x-2y for points (x,y) on the ellipse x^2/16 + y^2/9 = 1 is 2√13 -/
theorem max_value_on_ellipse :
  (∃ (x y : ℝ), x^2/16 + y^2/9 = 1 ∧ x - 2*y = 2*Real.sqrt 13) ∧
  (∀ (x y : ℝ), x^2/16 + y^2/9 = 1 → x - 2*y ≤ 2*Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l275_27577


namespace NUMINAMATH_CALUDE_min_value_of_x_l275_27549

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 3 + (1/3) * Real.log x) : x ≥ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l275_27549


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_l275_27557

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2*x + 2

theorem tangent_line_at_point_one :
  (f' 1 = 4) ∧
  (∀ x y : ℝ, y = f 1 → (4*x - y - 3 = 0 ↔ y - f 1 = f' 1 * (x - 1))) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_l275_27557


namespace NUMINAMATH_CALUDE_parallel_linear_function_b_value_l275_27573

/-- A linear function y = kx + b whose graph is parallel to y = 3x and passes through (1, -1) has b = -4 -/
theorem parallel_linear_function_b_value (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b) →  -- Definition of linear function
  k = 3 →  -- Parallel to y = 3x
  -1 = k * 1 + b →  -- Passes through (1, -1)
  b = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_linear_function_b_value_l275_27573


namespace NUMINAMATH_CALUDE_population_growth_prediction_l275_27502

/-- Theorem: Population Growth and Prediction --/
theorem population_growth_prediction
  (initial_population : ℝ)
  (current_population : ℝ)
  (future_population : ℝ)
  (h1 : current_population = 3 * initial_population)
  (h2 : future_population = 1.4 * current_population)
  (h3 : future_population = 16800)
  : initial_population = 4000 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_prediction_l275_27502


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l275_27593

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ)^4 * (3 : ℝ)^x = 81 ∧ x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l275_27593


namespace NUMINAMATH_CALUDE_max_correct_answers_l275_27553

theorem max_correct_answers (total_questions : Nat) (correct_score : Int) (incorrect_score : Int)
  (john_score : Int) (min_attempted : Nat) :
  total_questions = 25 →
  correct_score = 4 →
  incorrect_score = -3 →
  john_score = 52 →
  min_attempted = 20 →
  ∃ (correct incorrect unanswered : Nat),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_score + incorrect * incorrect_score = john_score ∧
    correct + incorrect ≥ min_attempted ∧
    correct ≤ 17 ∧
    ∀ (c : Nat), c > 17 →
      ¬(∃ (i u : Nat), c + i + u = total_questions ∧
        c * correct_score + i * incorrect_score = john_score ∧
        c + i ≥ min_attempted) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l275_27553


namespace NUMINAMATH_CALUDE_highest_y_coordinate_zero_is_highest_y_l275_27558

theorem highest_y_coordinate (x y : ℝ) : 
  (x - 4)^2 / 25 + y^2 / 49 = 0 → y ≤ 0 :=
by sorry

theorem zero_is_highest_y (x y : ℝ) : 
  (x - 4)^2 / 25 + y^2 / 49 = 0 → ∃ (x₀ y₀ : ℝ), (x₀ - 4)^2 / 25 + y₀^2 / 49 = 0 ∧ y₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_highest_y_coordinate_zero_is_highest_y_l275_27558


namespace NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l275_27590

theorem solutions_of_quadratic_equation :
  ∀ x : ℝ, x^2 - 64 = 0 ↔ x = 8 ∨ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l275_27590


namespace NUMINAMATH_CALUDE_final_number_lower_bound_board_game_result_l275_27580

/-- 
Given a positive integer n and a real number a ≥ n, 
we define a sequence of operations on a multiset of n real numbers,
initially all equal to a. In each step, we replace any two numbers
x and y in the multiset with (x+y)/4 until only one number remains.
-/
def final_number (n : ℕ+) (a : ℝ) (h : a ≥ n) : ℝ :=
  sorry

/-- 
The final number obtained after performing the operations
is always greater than or equal to a/n.
-/
theorem final_number_lower_bound (n : ℕ+) (a : ℝ) (h : a ≥ n) :
  final_number n a h ≥ a / n :=
  sorry

/--
For the specific case of 2023 numbers, each initially equal to 2023,
the final number is greater than 1.
-/
theorem board_game_result :
  final_number 2023 2023 (by norm_num) > 1 :=
  sorry

end NUMINAMATH_CALUDE_final_number_lower_bound_board_game_result_l275_27580


namespace NUMINAMATH_CALUDE_odd_integer_minus_twenty_l275_27595

theorem odd_integer_minus_twenty : 
  (2 * 53 - 1) - 20 = 85 := by sorry

end NUMINAMATH_CALUDE_odd_integer_minus_twenty_l275_27595


namespace NUMINAMATH_CALUDE_stu_book_count_l275_27511

theorem stu_book_count (stu_books : ℕ) (albert_books : ℕ) : 
  albert_books = 4 * stu_books →
  stu_books + albert_books = 45 →
  stu_books = 9 := by
sorry

end NUMINAMATH_CALUDE_stu_book_count_l275_27511


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l275_27562

theorem quadratic_distinct_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) ↔ 
  (a < 1 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l275_27562


namespace NUMINAMATH_CALUDE_specific_line_equation_l275_27567

/-- A line parameterized by real t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The specific parametric line from the problem -/
def specificLine : ParametricLine where
  x := λ t => 3 * t + 2
  y := λ t => 5 * t - 3

/-- The equation of a line in slope-intercept form -/
structure LineEquation where
  slope : ℝ
  intercept : ℝ

/-- Theorem stating that the specific parametric line has the given equation -/
theorem specific_line_equation :
  ∃ (t : ℝ), specificLine.y t = (5/3) * specificLine.x t - 19/3 := by
  sorry

end NUMINAMATH_CALUDE_specific_line_equation_l275_27567


namespace NUMINAMATH_CALUDE_game_cost_calculation_l275_27530

theorem game_cost_calculation (initial_amount : ℕ) (spent_amount : ℕ) (num_games : ℕ) :
  initial_amount = 42 →
  spent_amount = 10 →
  num_games = 4 →
  num_games > 0 →
  ∃ (game_cost : ℕ), game_cost * num_games = initial_amount - spent_amount ∧ game_cost = 8 :=
by sorry

end NUMINAMATH_CALUDE_game_cost_calculation_l275_27530


namespace NUMINAMATH_CALUDE_larger_number_is_84_l275_27513

theorem larger_number_is_84 (a b : ℕ+) (h1 : Nat.gcd a b = 84) (h2 : Nat.lcm a b = 21) (h3 : b = 4 * a) :
  b = 84 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_84_l275_27513


namespace NUMINAMATH_CALUDE_negative_one_less_than_abs_neg_two_fifths_l275_27514

theorem negative_one_less_than_abs_neg_two_fifths : -1 < |-2/5| := by
  sorry

end NUMINAMATH_CALUDE_negative_one_less_than_abs_neg_two_fifths_l275_27514


namespace NUMINAMATH_CALUDE_problem_solution_l275_27555

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = (1 + Real.sqrt 33) / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l275_27555


namespace NUMINAMATH_CALUDE_crackers_distribution_l275_27507

theorem crackers_distribution
  (initial_crackers : ℕ)
  (num_friends : ℕ)
  (remaining_crackers : ℕ)
  (h1 : initial_crackers = 15)
  (h2 : num_friends = 5)
  (h3 : remaining_crackers = 10)
  (h4 : num_friends > 0) :
  (initial_crackers - remaining_crackers) / num_friends = 1 :=
by sorry

end NUMINAMATH_CALUDE_crackers_distribution_l275_27507


namespace NUMINAMATH_CALUDE_only_exponential_has_multiplicative_property_l275_27533

-- Define the property that we're looking for
def HasMultiplicativeProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (x + y) = f x * f y

-- Define the types of functions we're considering
class FunctionType (f : ℝ → ℝ) where
  isPower : Prop
  isLogarithmic : Prop
  isExponential : Prop
  isLinear : Prop

-- Theorem stating that only exponential functions have the multiplicative property
theorem only_exponential_has_multiplicative_property (f : ℝ → ℝ) [FunctionType f] :
  HasMultiplicativeProperty f ↔ FunctionType.isExponential f := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_only_exponential_has_multiplicative_property_l275_27533


namespace NUMINAMATH_CALUDE_original_number_proof_l275_27512

theorem original_number_proof (x : ℝ) (h : 1 + 1/x = 9/4) : x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l275_27512


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_is_perpendicular_l275_27529

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_parallel_is_perpendicular
  (α β : Plane) (m n : Line)
  (h1 : α ≠ β)
  (h2 : m ≠ n)
  (h3 : perpendicular_line_plane m β)
  (h4 : parallel_line_plane n β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_is_perpendicular_l275_27529


namespace NUMINAMATH_CALUDE_total_sides_l275_27554

/-- The number of dice each person brought -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 6

/-- The number of people who brought dice -/
def num_people : ℕ := 2

/-- Theorem: The total number of sides on all dice is 48 -/
theorem total_sides : num_people * num_dice * sides_per_die = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_sides_l275_27554


namespace NUMINAMATH_CALUDE_rearranged_digits_subtraction_l275_27569

theorem rearranged_digits_subtraction :
  ∀ h t u : ℕ,
  h ≠ t → h ≠ u → t ≠ u →
  h > 0 → t > 0 → u > 0 →
  h > u →
  h * 100 + t * 10 + u - (t * 100 + u * 10 + h) = 179 →
  h = 8 ∧ t = 7 ∧ u = 9 :=
by sorry

end NUMINAMATH_CALUDE_rearranged_digits_subtraction_l275_27569


namespace NUMINAMATH_CALUDE_guppy_angelfish_ratio_l275_27561

/-- Proves that the ratio of guppies to angelfish is 2:1 given the conditions -/
theorem guppy_angelfish_ratio :
  let goldfish : ℕ := 8
  let angelfish : ℕ := goldfish + 4
  let total_fish : ℕ := 44
  let guppies : ℕ := total_fish - (goldfish + angelfish)
  (guppies : ℚ) / angelfish = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_guppy_angelfish_ratio_l275_27561


namespace NUMINAMATH_CALUDE_valid_n_set_l275_27510

theorem valid_n_set (n : ℕ) : (∃ a b : ℤ, n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n ∈ ({0, 1, 2} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_valid_n_set_l275_27510


namespace NUMINAMATH_CALUDE_half_of_a_l275_27508

theorem half_of_a (a : ℝ) : (1 / 2) * a = a / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_of_a_l275_27508


namespace NUMINAMATH_CALUDE_square_of_binomial_l275_27542

theorem square_of_binomial (m n : ℝ) : (3*m - n)^2 = 9*m^2 - 6*m*n + n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l275_27542


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l275_27575

/-- Given a parabola and a line with exactly one intersection point, prove a specific algebraic identity. -/
theorem parabola_line_intersection (m : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + 4 * x + 5 = 8 * m * x + 8 * m) → 
  m^36 + 1155 / m^12 = 39236 :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l275_27575


namespace NUMINAMATH_CALUDE_longs_interest_l275_27586

/-- Calculates the total interest earned after a given number of years with compound interest and an additional deposit -/
def totalInterest (initialInvestment : ℝ) (interestRate : ℝ) (additionalDeposit : ℝ) (depositYear : ℕ) (totalYears : ℕ) : ℝ :=
  let finalAmount := 
    (initialInvestment * (1 + interestRate) ^ depositYear + additionalDeposit) * (1 + interestRate) ^ (totalYears - depositYear)
  finalAmount - initialInvestment - additionalDeposit

/-- The total interest earned by Long after 4 years -/
theorem longs_interest : 
  totalInterest 1200 0.08 500 2 4 = 515.26 := by sorry

end NUMINAMATH_CALUDE_longs_interest_l275_27586


namespace NUMINAMATH_CALUDE_arithmetic_progression_nth_term_l275_27597

theorem arithmetic_progression_nth_term (a d n : ℕ) (Tn : ℕ) 
  (h1 : a = 2) 
  (h2 : d = 8) 
  (h3 : Tn = 90) 
  (h4 : Tn = a + (n - 1) * d) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_nth_term_l275_27597


namespace NUMINAMATH_CALUDE_abs_leq_two_necessary_not_sufficient_l275_27546

theorem abs_leq_two_necessary_not_sufficient :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x| ≤ 2) ∧
  (∃ x : ℝ, |x| ≤ 2 ∧ ¬(0 ≤ x ∧ x ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_abs_leq_two_necessary_not_sufficient_l275_27546


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l275_27571

/-- Calculate the interest rate given principal, final amount, and time -/
theorem interest_rate_calculation (P A t : ℝ) (h1 : P = 1200) (h2 : A = 1344) (h3 : t = 2.4) :
  (A - P) / (P * t) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l275_27571


namespace NUMINAMATH_CALUDE_fraction_problem_l275_27541

theorem fraction_problem (a b : ℤ) : 
  (a + 2 : ℚ) / b = 4 / 7 →
  (a : ℚ) / (b - 2) = 14 / 25 →
  ∃ (k : ℤ), k ≠ 0 ∧ k * a = 6 ∧ k * b = 11 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l275_27541


namespace NUMINAMATH_CALUDE_exchange_calculation_l275_27587

/-- Exchange rate from USD to JPY -/
def exchange_rate : ℚ := 5000 / 45

/-- Amount in USD to be exchanged -/
def usd_amount : ℚ := 15

/-- Theorem stating the correct exchange amount -/
theorem exchange_calculation :
  usd_amount * exchange_rate = 5000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_exchange_calculation_l275_27587


namespace NUMINAMATH_CALUDE_parabola_focus_l275_27589

/-- Represents a parabola with equation y^2 = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun y x => y^2 = 8*x

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (2, 0)

/-- Theorem stating that the focus of the parabola y^2 = 8x is (2, 0) -/
theorem parabola_focus (p : Parabola) : focus p = (2, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l275_27589


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l275_27548

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l275_27548


namespace NUMINAMATH_CALUDE_sum_20_225_base7_l275_27531

/-- Represents a number in base 7 --/
def Base7 : Type := ℕ

/-- Converts a natural number to its base 7 representation --/
def toBase7 (n : ℕ) : Base7 := sorry

/-- Adds two numbers in base 7 --/
def addBase7 (a b : Base7) : Base7 := sorry

/-- Theorem: The sum of 20₇ and 225₇ in base 7 is 245₇ --/
theorem sum_20_225_base7 :
  addBase7 (toBase7 20) (toBase7 225) = toBase7 245 := by sorry

end NUMINAMATH_CALUDE_sum_20_225_base7_l275_27531


namespace NUMINAMATH_CALUDE_figurine_cost_l275_27526

theorem figurine_cost (tv_count : ℕ) (tv_price : ℕ) (figurine_count : ℕ) (total_spent : ℕ) :
  tv_count = 5 →
  tv_price = 50 →
  figurine_count = 10 →
  total_spent = 260 →
  (total_spent - tv_count * tv_price) / figurine_count = 1 :=
by sorry

end NUMINAMATH_CALUDE_figurine_cost_l275_27526


namespace NUMINAMATH_CALUDE_inverse_function_parameter_l275_27565

/-- Given a function f and its inverse, find the value of b -/
theorem inverse_function_parameter (f : ℝ → ℝ) (b : ℝ) : 
  (∀ x, f x = 1 / (2 * x + b)) →
  (∀ x, f⁻¹ x = (2 - 3 * x) / (3 * x)) →
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_parameter_l275_27565


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l275_27528

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) : 
  let total_juice := (2/3) * C
  let cups := 6
  let juice_per_cup := total_juice / cups
  (juice_per_cup / C) * 100 = 100/9 := by sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l275_27528


namespace NUMINAMATH_CALUDE_system_solution_proof_l275_27552

theorem system_solution_proof :
  let x₁ : ℝ := 4
  let x₂ : ℝ := 3
  let x₃ : ℝ := 5
  (x₁ + 2 * x₂ = 10) ∧
  (3 * x₁ + 2 * x₂ + x₃ = 23) ∧
  (x₂ + 2 * x₃ = 13) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l275_27552


namespace NUMINAMATH_CALUDE_divisibility_by_2008_l275_27503

theorem divisibility_by_2008 (k m : ℕ) (h1 : ∃ (u : ℕ), k = 25 * (2 * u + 1)) (h2 : ∃ (v : ℕ), m = 25 * v) :
  2008 ∣ (2^k + 4^m) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_2008_l275_27503


namespace NUMINAMATH_CALUDE_tangent_line_condition_l275_27505

theorem tangent_line_condition (a : ℝ) : 
  (∃ (x : ℝ), a * x + 1 - a = x^2 ∧ ∀ (y : ℝ), y ≠ x → a * y + 1 - a ≠ y^2) ↔ |a| = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_condition_l275_27505


namespace NUMINAMATH_CALUDE_remainder_theorem_l275_27568

/-- A polynomial p(x) satisfying p(2) = 4 and p(5) = 10 -/
def p : Polynomial ℝ :=
  sorry

theorem remainder_theorem (h1 : p.eval 2 = 4) (h2 : p.eval 5 = 10) :
  ∃ q : Polynomial ℝ, p = q * ((X - 2) * (X - 5)) + (2 * X) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l275_27568


namespace NUMINAMATH_CALUDE_debby_museum_pictures_l275_27581

/-- The number of pictures Debby took at the zoo -/
def zoo_pictures : ℕ := 24

/-- The number of pictures Debby deleted -/
def deleted_pictures : ℕ := 14

/-- The number of pictures Debby had remaining after deletion -/
def remaining_pictures : ℕ := 22

/-- The number of pictures Debby took at the museum -/
def museum_pictures : ℕ := zoo_pictures + deleted_pictures - remaining_pictures

theorem debby_museum_pictures : museum_pictures = 12 := by
  sorry

end NUMINAMATH_CALUDE_debby_museum_pictures_l275_27581


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l275_27534

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 5 + a 8 = 15 → a 3 + a 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l275_27534


namespace NUMINAMATH_CALUDE_exists_point_P_trajectory_G_l275_27517

/-- Definition of the ellipse C -/
def is_on_ellipse_C (x y : ℝ) : Prop := x^2/36 + y^2/20 = 1

/-- Definition of point A -/
def point_A : ℝ × ℝ := (-6, 0)

/-- Definition of point F -/
def point_F : ℝ × ℝ := (4, 0)

/-- Definition of vector AP -/
def vector_AP (x y : ℝ) : ℝ × ℝ := (x + 6, y)

/-- Definition of vector FP -/
def vector_FP (x y : ℝ) : ℝ × ℝ := (x - 4, y)

/-- Theorem stating the existence of point P -/
theorem exists_point_P :
  ∃ (x y : ℝ), 
    is_on_ellipse_C x y ∧ 
    y > 0 ∧ 
    (vector_AP x y).1 * (vector_FP x y).1 + (vector_AP x y).2 * (vector_FP x y).2 = 0 ∧
    x = 3/2 ∧ 
    y = 5 * Real.sqrt 3 / 2 :=
sorry

/-- Definition of point M on ellipse C -/
def point_M (x₀ y₀ : ℝ) : Prop := is_on_ellipse_C x₀ y₀

/-- Definition of midpoint G of MF -/
def point_G (x y : ℝ) (x₀ y₀ : ℝ) : Prop :=
  x = (x₀ + 2) / 2 ∧ y = y₀ / 2

/-- Theorem stating the trajectory equation of G -/
theorem trajectory_G :
  ∀ (x y : ℝ), 
    (∃ (x₀ y₀ : ℝ), point_M x₀ y₀ ∧ point_G x y x₀ y₀) ↔ 
    (x - 1)^2 / 9 + y^2 / 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_exists_point_P_trajectory_G_l275_27517


namespace NUMINAMATH_CALUDE_fruit_bowl_total_l275_27583

/-- Represents the number of pieces of each type of fruit in the bowl -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Theorem stating the total number of fruits in the bowl under given conditions -/
theorem fruit_bowl_total (bowl : FruitBowl) 
  (h1 : bowl.pears = bowl.apples + 2)
  (h2 : bowl.bananas = bowl.pears + 3)
  (h3 : bowl.bananas = 9) : 
  bowl.apples + bowl.pears + bowl.bananas = 19 := by
  sorry

end NUMINAMATH_CALUDE_fruit_bowl_total_l275_27583


namespace NUMINAMATH_CALUDE_percent_increase_proof_l275_27515

def original_lines : ℕ := 5600 - 1600
def increased_lines : ℕ := 5600
def line_increase : ℕ := 1600

theorem percent_increase_proof :
  (line_increase : ℝ) / (original_lines : ℝ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_proof_l275_27515


namespace NUMINAMATH_CALUDE_history_book_cost_l275_27540

/-- Given the conditions of a book purchase, this theorem proves the cost of each history book. -/
theorem history_book_cost (total_books : ℕ) (math_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 90 →
  math_book_cost = 4 →
  total_price = 397 →
  math_books = 53 →
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 :=
by sorry

end NUMINAMATH_CALUDE_history_book_cost_l275_27540


namespace NUMINAMATH_CALUDE_line_equation_from_intercepts_l275_27560

theorem line_equation_from_intercepts (x y : ℝ) :
  (x = -2 ∧ y = 0) ∨ (x = 0 ∧ y = 3) → 3 * x - 2 * y + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_from_intercepts_l275_27560


namespace NUMINAMATH_CALUDE_chocolate_bars_per_box_l275_27501

theorem chocolate_bars_per_box (total_bars : ℕ) (num_boxes : ℕ) 
  (h1 : total_bars = 442) (h2 : num_boxes = 17) :
  total_bars / num_boxes = 26 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_per_box_l275_27501


namespace NUMINAMATH_CALUDE_student_age_problem_l275_27537

theorem student_age_problem (total_students : Nat) (avg_age : Nat) 
  (group1_count : Nat) (group1_avg : Nat)
  (group2_count : Nat) (group2_avg : Nat)
  (group3_count : Nat) (group3_avg : Nat) :
  total_students = 25 →
  avg_age = 16 →
  group1_count = 7 →
  group1_avg = 15 →
  group2_count = 12 →
  group2_avg = 16 →
  group3_count = 5 →
  group3_avg = 18 →
  group1_count + group2_count + group3_count = total_students - 1 →
  (total_students * avg_age) - (group1_count * group1_avg + group2_count * group2_avg + group3_count * group3_avg) = 13 := by
  sorry

end NUMINAMATH_CALUDE_student_age_problem_l275_27537


namespace NUMINAMATH_CALUDE_consecutive_pairs_49_6_l275_27578

/-- The number of ways to choose 6 elements among the first 49 positive integers
    with at least two consecutive elements -/
def consecutivePairs (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose (n - k + 1) k

theorem consecutive_pairs_49_6 :
  consecutivePairs 49 6 = Nat.choose 49 6 - Nat.choose 44 6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pairs_49_6_l275_27578


namespace NUMINAMATH_CALUDE_elsa_angus_token_difference_l275_27594

/-- Calculates the difference in total token value between two people -/
def tokenValueDifference (elsa_tokens : ℕ) (angus_tokens : ℕ) (token_value : ℕ) : ℕ :=
  (elsa_tokens * token_value) - (angus_tokens * token_value)

/-- Proves that the difference in token value between Elsa and Angus is $20 -/
theorem elsa_angus_token_difference :
  tokenValueDifference 60 55 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_elsa_angus_token_difference_l275_27594


namespace NUMINAMATH_CALUDE_percentage_difference_difference_is_twelve_l275_27521

theorem percentage_difference : ℝ → Prop :=
  let percent_of_40 := (80 / 100) * 40
  let fraction_of_25 := (4 / 5) * 25
  λ x => percent_of_40 - fraction_of_25 = x

theorem difference_is_twelve : percentage_difference 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_difference_is_twelve_l275_27521


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l275_27539

theorem quadratic_root_difference (m : ℝ) : 
  ∃ (x₁ x₂ : ℂ), x₁^2 + m*x₁ + 3 = 0 ∧ 
                 x₂^2 + m*x₂ + 3 = 0 ∧ 
                 x₁ ≠ x₂ ∧
                 Complex.abs (x₁ - x₂) = 2 → 
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l275_27539


namespace NUMINAMATH_CALUDE_cookie_recipe_total_cups_l275_27591

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio :=
  (butter : ℕ)
  (flour : ℕ)
  (sugar : ℕ)

/-- Calculates the total cups of ingredients given a ratio and amount of sugar -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  (ratio.butter + ratio.flour + ratio.sugar) * partSize

/-- Theorem: Given the specified ratio and sugar amount, the total cups is 40 -/
theorem cookie_recipe_total_cups :
  let ratio : RecipeRatio := ⟨2, 5, 3⟩
  totalCups ratio 12 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cookie_recipe_total_cups_l275_27591


namespace NUMINAMATH_CALUDE_prop_logic_l275_27519

theorem prop_logic (p q : Prop) (h : ¬(¬p ∨ ¬q)) : (p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_prop_logic_l275_27519


namespace NUMINAMATH_CALUDE_horner_method_v2_value_l275_27599

def f (x : ℝ) : ℝ := 4*x^4 + 3*x^3 - 6*x^2 + x - 1

def horner_v2 (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  let v₀ := a₄
  let v₁ := v₀ * x + a₃
  v₁ * x + a₂

theorem horner_method_v2_value :
  horner_v2 4 3 (-6) 1 (-1) (-1) = -5 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v2_value_l275_27599


namespace NUMINAMATH_CALUDE_sam_exchange_probability_l275_27551

/-- Represents the vending machine and Sam's purchasing scenario -/
structure VendingMachine where
  num_toys : Nat
  toy_prices : List Rat
  favorite_toy_price : Rat
  sam_quarters : Nat
  sam_bill : Nat

/-- Calculates the probability of Sam needing to exchange his bill -/
def probability_need_exchange (vm : VendingMachine) : Rat :=
  1 - (Nat.factorial 7 : Rat) / (Nat.factorial vm.num_toys : Rat)

/-- Theorem stating the probability of Sam needing to exchange his bill -/
theorem sam_exchange_probability (vm : VendingMachine) :
  vm.num_toys = 10 ∧
  vm.toy_prices = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5] ∧
  vm.favorite_toy_price = 4 ∧
  vm.sam_quarters = 12 ∧
  vm.sam_bill = 20 →
  probability_need_exchange vm = 719 / 720 := by
  sorry

#eval probability_need_exchange {
  num_toys := 10,
  toy_prices := [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
  favorite_toy_price := 4,
  sam_quarters := 12,
  sam_bill := 20
}

end NUMINAMATH_CALUDE_sam_exchange_probability_l275_27551


namespace NUMINAMATH_CALUDE_mouse_cheese_distance_l275_27544

/-- The point where the mouse starts getting farther from the cheese -/
def mouse_turn_point : ℚ × ℚ := (-33/17, 285/17)

/-- The location of the cheese -/
def cheese_location : ℚ × ℚ := (9, 15)

/-- The equation of the line the mouse is running on: y = -4x + 9 -/
def mouse_path (x : ℚ) : ℚ := -4 * x + 9

theorem mouse_cheese_distance :
  let (a, b) := mouse_turn_point
  -- The point is on the mouse's path
  (mouse_path a = b) ∧
  -- The line from the cheese to the point is perpendicular to the mouse's path
  ((b - 15) / (a - 9) = 1 / 4) ∧
  -- The sum of the coordinates is 252/17
  (a + b = 252 / 17) := by sorry

end NUMINAMATH_CALUDE_mouse_cheese_distance_l275_27544


namespace NUMINAMATH_CALUDE_miss_stevie_payment_l275_27579

def jerry_painting_hours : ℕ := 8
def jerry_painting_rate : ℚ := 15
def jerry_mowing_hours : ℕ := 6
def jerry_mowing_rate : ℚ := 10
def jerry_plumbing_hours : ℕ := 4
def jerry_plumbing_rate : ℚ := 18
def jerry_discount : ℚ := 0.1

def randy_painting_hours : ℕ := 7
def randy_painting_rate : ℚ := 12
def randy_mowing_hours : ℕ := 4
def randy_mowing_rate : ℚ := 8
def randy_electrical_hours : ℕ := 3
def randy_electrical_rate : ℚ := 20
def randy_discount : ℚ := 0.05

def total_payment : ℚ := 394

theorem miss_stevie_payment :
  let jerry_total := (jerry_painting_hours * jerry_painting_rate +
                      jerry_mowing_hours * jerry_mowing_rate +
                      jerry_plumbing_hours * jerry_plumbing_rate) * (1 - jerry_discount)
  let randy_total := (randy_painting_hours * randy_painting_rate +
                      randy_mowing_hours * randy_mowing_rate +
                      randy_electrical_hours * randy_electrical_rate) * (1 - randy_discount)
  jerry_total + randy_total = total_payment := by
    sorry

end NUMINAMATH_CALUDE_miss_stevie_payment_l275_27579


namespace NUMINAMATH_CALUDE_lydias_flowering_plants_fraction_l275_27524

theorem lydias_flowering_plants_fraction (total_plants : ℕ) 
  (flowering_percentage : ℚ) (flowers_per_plant : ℕ) (total_flowers_on_porch : ℕ) :
  total_plants = 80 →
  flowering_percentage = 2/5 →
  flowers_per_plant = 5 →
  total_flowers_on_porch = 40 →
  (total_flowers_on_porch / flowers_per_plant) / (flowering_percentage * total_plants) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_lydias_flowering_plants_fraction_l275_27524


namespace NUMINAMATH_CALUDE_pairball_playing_time_l275_27516

theorem pairball_playing_time (total_time : ℕ) (num_children : ℕ) (players_per_game : ℕ) :
  total_time = 90 ∧ 
  num_children = 6 ∧ 
  players_per_game = 2 →
  (total_time * players_per_game) / num_children = 30 := by
sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l275_27516


namespace NUMINAMATH_CALUDE_estimate_larger_than_actual_l275_27520

theorem estimate_larger_than_actual (x y z : ℝ) 
  (hxy : x > y) (hy : y > 0) (hz : z > 0) : 
  (x + z) - (y - z) > x - y := by
  sorry

end NUMINAMATH_CALUDE_estimate_larger_than_actual_l275_27520


namespace NUMINAMATH_CALUDE_car_dealership_problem_l275_27584

/-- Represents the initial number of cars on the lot -/
def initial_cars : ℕ := 280

/-- Represents the number of cars in the new shipment -/
def new_shipment : ℕ := 80

/-- Represents the initial percentage of silver cars -/
def initial_silver_percent : ℚ := 20 / 100

/-- Represents the percentage of non-silver cars in the new shipment -/
def new_non_silver_percent : ℚ := 35 / 100

/-- Represents the final percentage of silver cars after the new shipment -/
def final_silver_percent : ℚ := 30 / 100

theorem car_dealership_problem :
  let initial_silver := initial_silver_percent * initial_cars
  let new_silver := (1 - new_non_silver_percent) * new_shipment
  let total_cars := initial_cars + new_shipment
  let total_silver := initial_silver + new_silver
  (total_silver : ℚ) / total_cars = final_silver_percent := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_problem_l275_27584


namespace NUMINAMATH_CALUDE_sin_75_degrees_l275_27582

theorem sin_75_degrees : 
  Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  -- Define known values
  have sin_45 : Real.sin (45 * π / 180) = Real.sqrt 2 / 2 := sorry
  have cos_45 : Real.cos (45 * π / 180) = Real.sqrt 2 / 2 := sorry
  have sin_30 : Real.sin (30 * π / 180) = 1 / 2 := sorry
  have cos_30 : Real.cos (30 * π / 180) = Real.sqrt 3 / 2 := sorry

  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sin_75_degrees_l275_27582


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l275_27596

/-- A circle inscribed in a quadrilateral EFGH -/
structure InscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The point where the circle is tangent to EF -/
  R : Point
  /-- The point where the circle is tangent to GH -/
  S : Point
  /-- The length of ER -/
  ER : ℝ
  /-- The length of RF -/
  RF : ℝ
  /-- The length of GS -/
  GS : ℝ
  /-- The length of SH -/
  SH : ℝ

/-- Theorem: The square of the radius of the inscribed circle is 868 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle)
    (h1 : c.ER = 21)
    (h2 : c.RF = 28)
    (h3 : c.GS = 40)
    (h4 : c.SH = 32) :
    c.r^2 = 868 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l275_27596


namespace NUMINAMATH_CALUDE_addington_average_temperature_l275_27559

/-- The average of the daily low temperatures in Addington from September 15th, 2008 through September 19th, 2008, inclusive, is 42.4 degrees Fahrenheit. -/
theorem addington_average_temperature : 
  let temperatures : List ℝ := [40, 47, 45, 41, 39]
  (temperatures.sum / temperatures.length : ℝ) = 42.4 := by
  sorry

end NUMINAMATH_CALUDE_addington_average_temperature_l275_27559


namespace NUMINAMATH_CALUDE_candy_distribution_l275_27509

/-- Candy distribution problem -/
theorem candy_distribution (tabitha stan julie carlos : ℕ) : 
  tabitha = 22 →
  stan = 13 →
  julie = tabitha / 2 →
  tabitha + stan + julie + carlos = 72 →
  carlos / stan = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l275_27509


namespace NUMINAMATH_CALUDE_odd_function_and_inequality_l275_27527

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + a * x^2) + 2 * x)

theorem odd_function_and_inequality (a m : ℝ) : 
  (∀ x, f a x = -f a (-x)) ∧ 
  (∀ x, f a (2 * m - m * Real.sin x) + f a (Real.cos x)^2 ≥ 0) →
  a = 4 ∧ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_and_inequality_l275_27527


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l275_27547

theorem no_solutions_for_equation : 
  ¬∃ (a b : ℕ+), 
    a ≥ b ∧ 
    a * b + 125 = 30 * Nat.lcm a b + 24 * Nat.gcd a b + a % b :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l275_27547


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_l275_27588

theorem smallest_perfect_square_divisible (n : ℕ) (h : n = 14400) :
  (∃ k : ℕ, k * k = n ∧ n / 5 = 2880) ∧
  (∀ m : ℕ, m < n → m / 5 = 2880 → ¬∃ j : ℕ, j * j = m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_l275_27588


namespace NUMINAMATH_CALUDE_xyz_equals_one_l275_27522

theorem xyz_equals_one
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = (2 * b + 3 * c) / (x - 3))
  (eq_b : b = (3 * a + 2 * c) / (y - 3))
  (eq_c : c = (2 * a + 2 * b) / (z - 3))
  (sum_product : x * y + x * z + y * z = -1)
  (sum : x + y + z = 1) :
  x * y * z = 1 := by
sorry


end NUMINAMATH_CALUDE_xyz_equals_one_l275_27522


namespace NUMINAMATH_CALUDE_parallelogram_area_smallest_real_part_l275_27585

theorem parallelogram_area_smallest_real_part (z : ℂ) :
  (z.im > 0) →
  (abs ((z - z⁻¹).re) ≥ 0) →
  (abs (z.im * z⁻¹.re - z.re * z⁻¹.im) = 1) →
  ∃ (w : ℂ), (w.im > 0) ∧ 
             (abs ((w - w⁻¹).re) ≥ 0) ∧ 
             (abs (w.im * w⁻¹.re - w.re * w⁻¹.im) = 1) ∧
             (abs ((w - w⁻¹).re) = 0) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_smallest_real_part_l275_27585


namespace NUMINAMATH_CALUDE_water_in_mixture_l275_27545

theorem water_in_mixture (water_parts syrup_parts total_volume : ℚ) 
  (h1 : water_parts = 5)
  (h2 : syrup_parts = 2)
  (h3 : total_volume = 3) : 
  (water_parts * total_volume) / (water_parts + syrup_parts) = 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_water_in_mixture_l275_27545


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l275_27566

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 4 - 2 * x) ↔ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l275_27566


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l275_27598

theorem arithmetic_calculations :
  (3 * 232 + 456 = 1152) ∧
  (760 * 5 - 2880 = 920) ∧
  (805 / 7 = 115) ∧
  (45 + 255 / 5 = 96) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l275_27598


namespace NUMINAMATH_CALUDE_kitchen_tiles_l275_27538

theorem kitchen_tiles (kitchen_length kitchen_width tile_area : ℝ) 
  (h1 : kitchen_length = 52)
  (h2 : kitchen_width = 79)
  (h3 : tile_area = 7.5) : 
  ⌈(kitchen_length * kitchen_width) / tile_area⌉ = 548 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_tiles_l275_27538


namespace NUMINAMATH_CALUDE_parabola_point_distance_l275_27564

/-- A point on a parabola with a specific distance to the focus has a specific distance to the y-axis -/
theorem parabola_point_distance (P : ℝ × ℝ) : 
  (P.2)^2 = 8 * P.1 →  -- P is on the parabola y^2 = 8x
  ((P.1 - 2)^2 + P.2^2)^(1/2 : ℝ) = 6 →  -- Distance from P to focus (2, 0) is 6
  P.1 = 4 :=  -- Distance from P to y-axis is 4
by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l275_27564


namespace NUMINAMATH_CALUDE_smallest_shift_for_even_function_l275_27574

/-- Given a function f(x) = sin(ωx + π/2) where ω > 0, 
    if the distance between adjacent axes of symmetry is 2π
    and shifting f(x) to the left by m units results in an even function,
    then the smallest positive value of m is π/(2ω). -/
theorem smallest_shift_for_even_function (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x + π / 2)
  (∀ x : ℝ, f (x + 2*π/ω) = f x) →  -- distance between adjacent axes of symmetry is 2π
  (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, f (x + m) = f (-x + m)) →  -- shifting by m results in an even function
  (∃ m : ℝ, m > 0 ∧ 
    (∀ x : ℝ, f (x + m) = f (-x + m)) ∧  -- m shift results in even function
    (∀ m' : ℝ, m' > 0 → (∀ x : ℝ, f (x + m') = f (-x + m')) → m ≤ m') ∧  -- m is the smallest such shift
    m = π / (2 * ω)) :=  -- m equals π/(2ω)
by sorry

end NUMINAMATH_CALUDE_smallest_shift_for_even_function_l275_27574


namespace NUMINAMATH_CALUDE_april_initial_roses_l275_27504

/-- The number of roses April started with, given the price per rose, 
    the number of roses left, and the total earnings from selling roses. -/
def initial_roses (price_per_rose : ℕ) (roses_left : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings / price_per_rose) + roses_left

/-- Theorem stating that April started with 13 roses -/
theorem april_initial_roses : 
  initial_roses 4 4 36 = 13 := by
  sorry

end NUMINAMATH_CALUDE_april_initial_roses_l275_27504


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l275_27506

theorem smallest_integer_with_given_remainders :
  let x : ℕ := 167
  (∀ y : ℕ, y > 0 →
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → y ≥ x) ∧
  (x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l275_27506


namespace NUMINAMATH_CALUDE_problem_line_direction_cosines_l275_27576

/-- The line defined by two equations -/
structure Line where
  eq1 : ℝ → ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ → ℝ

/-- Direction cosines of a line -/
structure DirectionCosines where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- The specific line from the problem -/
def problemLine : Line where
  eq1 := fun x y z => 2*x - 3*y - 3*z - 9
  eq2 := fun x y z => x - 2*y + z + 3

/-- Compute direction cosines for a given line -/
noncomputable def computeDirectionCosines (l : Line) : DirectionCosines :=
  { α := 9 / Real.sqrt 107
  , β := 5 / Real.sqrt 107
  , γ := 1 / Real.sqrt 107 }

/-- Theorem: The direction cosines of the problem line are (9/√107, 5/√107, 1/√107) -/
theorem problem_line_direction_cosines :
  computeDirectionCosines problemLine = 
  { α := 9 / Real.sqrt 107
  , β := 5 / Real.sqrt 107
  , γ := 1 / Real.sqrt 107 } := by
  sorry

end NUMINAMATH_CALUDE_problem_line_direction_cosines_l275_27576


namespace NUMINAMATH_CALUDE_intersection_A_B_l275_27572

-- Define set A
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 ≥ 4}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l275_27572


namespace NUMINAMATH_CALUDE_identity_function_proof_l275_27556

theorem identity_function_proof (f : ℝ → ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 0 1) →
  (∀ x ∈ Set.Icc 0 1, f (2 * x - f x) = x) →
  (∀ x ∈ Set.Icc 0 1, f x = x) := by
sorry

end NUMINAMATH_CALUDE_identity_function_proof_l275_27556


namespace NUMINAMATH_CALUDE_fraction_equality_l275_27592

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 5 / 8) :
  (b - a) / a = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l275_27592


namespace NUMINAMATH_CALUDE_first_discount_percentage_l275_27500

/-- Proves that the first discount percentage is 10% for an article with a given price and two successive discounts -/
theorem first_discount_percentage
  (normal_price : ℝ)
  (first_discount : ℝ)
  (second_discount : ℝ)
  (h1 : normal_price = 174.99999999999997)
  (h2 : first_discount = 0.1)
  (h3 : second_discount = 0.2)
  : first_discount = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l275_27500


namespace NUMINAMATH_CALUDE_evans_books_in_eight_years_l275_27518

def books_six_years_ago : ℕ := 500
def books_reduced : ℕ := 100
def books_given_away_fraction : ℚ := 1/2
def books_replaced_fraction : ℚ := 1/4
def books_increase_fraction : ℚ := 3/2
def books_gifted : ℕ := 30

theorem evans_books_in_eight_years :
  let current_books := books_six_years_ago - books_reduced
  let books_after_giving_away := (current_books : ℚ) * (1 - books_given_away_fraction)
  let books_after_replacing := books_after_giving_away + books_after_giving_away * books_replaced_fraction
  let final_books := books_after_replacing + books_after_replacing * books_increase_fraction + books_gifted
  final_books = 655 := by sorry

end NUMINAMATH_CALUDE_evans_books_in_eight_years_l275_27518


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l275_27532

theorem number_exceeding_fraction : ∃ x : ℚ, x = (3/8) * x + 15 ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l275_27532


namespace NUMINAMATH_CALUDE_symmetry_proof_l275_27570

/-- A point in the 2D Cartesian coordinate system -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Symmetry with respect to the y-axis -/
def symmetric_wrt_y_axis (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = q.y

/-- The given point -/
def given_point : Point :=
  { x := -1, y := -2 }

/-- The symmetric point to be proved -/
def symmetric_point : Point :=
  { x := 1, y := -2 }

theorem symmetry_proof : symmetric_wrt_y_axis given_point symmetric_point := by
  sorry

end NUMINAMATH_CALUDE_symmetry_proof_l275_27570


namespace NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l275_27550

theorem sum_of_special_primes_is_prime (P Q : ℕ+) (h1 : Nat.Prime P)
  (h2 : Nat.Prime Q) (h3 : Nat.Prime (P - Q)) (h4 : Nat.Prime (P + Q)) :
  Nat.Prime (P + Q + (P - Q) + Q) :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l275_27550


namespace NUMINAMATH_CALUDE_circle_diameter_l275_27536

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 9 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l275_27536
