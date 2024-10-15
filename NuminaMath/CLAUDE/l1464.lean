import Mathlib

namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1464_146426

/-- A line with slope 4 passing through (2, -1) has m + b = -5 -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
    (m = 4) →  -- Given slope
    (-1 = 4 * 2 + b) →  -- Line passes through (2, -1)
    (m + b = -5) :=  -- Conclusion to prove
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1464_146426


namespace NUMINAMATH_CALUDE_min_xy_value_l1464_146431

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 10 * x + 2 * y + 60 = x * y) : 
  x * y ≥ 180 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 10 * x₀ + 2 * y₀ + 60 = x₀ * y₀ ∧ x₀ * y₀ = 180 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l1464_146431


namespace NUMINAMATH_CALUDE_water_level_rise_rate_l1464_146466

/-- The water level function with respect to time -/
def water_level (t : ℝ) : ℝ := 0.3 * t + 3

/-- The time domain -/
def time_domain : Set ℝ := { t | 0 ≤ t ∧ t ≤ 5 }

/-- The rate of change of the water level -/
def water_level_rate : ℝ := 0.3

theorem water_level_rise_rate :
  ∀ t ∈ time_domain, 
    (water_level (t + 1) - water_level t) = water_level_rate := by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_rate_l1464_146466


namespace NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_one_l1464_146403

theorem fraction_equality_implies_x_equals_one :
  ∀ x : ℚ, (5 + x) / (7 + x) = (2 + x) / (3 + x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_one_l1464_146403


namespace NUMINAMATH_CALUDE_baking_ingredient_calculation_l1464_146405

/-- Represents the ingredients needed for baking --/
structure BakingIngredients where
  flour_cake : ℝ
  flour_cookies : ℝ
  sugar_cake : ℝ
  sugar_cookies : ℝ

/-- Represents the available ingredients --/
structure AvailableIngredients where
  flour : ℝ
  sugar : ℝ

/-- Calculates the difference between available and needed ingredients --/
def ingredientDifference (needed : BakingIngredients) (available : AvailableIngredients) : 
  ℝ × ℝ :=
  let total_flour_needed := needed.flour_cake + needed.flour_cookies
  let total_sugar_needed := needed.sugar_cake + needed.sugar_cookies
  (available.flour - total_flour_needed, available.sugar - total_sugar_needed)

theorem baking_ingredient_calculation 
  (needed : BakingIngredients) 
  (available : AvailableIngredients) : 
  needed.flour_cake = 6 ∧ 
  needed.flour_cookies = 2 ∧ 
  needed.sugar_cake = 3.5 ∧ 
  needed.sugar_cookies = 1.5 ∧
  available.flour = 8 ∧ 
  available.sugar = 4 → 
  ingredientDifference needed available = (0, -1) := by
  sorry

end NUMINAMATH_CALUDE_baking_ingredient_calculation_l1464_146405


namespace NUMINAMATH_CALUDE_increasing_function_implies_a_bound_l1464_146401

/-- A function f is increasing on an interval [a, +∞) if for any x₁, x₂ in the interval with x₁ < x₂, we have f(x₁) < f(x₂) -/
def IncreasingOnInterval (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The main theorem stating that if f(x) = x^2 - 2ax + 2 is increasing on [3, +∞), then a ≤ 3 -/
theorem increasing_function_implies_a_bound (a : ℝ) :
  IncreasingOnInterval (fun x => x^2 - 2*a*x + 2) 3 → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_implies_a_bound_l1464_146401


namespace NUMINAMATH_CALUDE_expected_value_of_heads_l1464_146415

/-- Represents the different types of coins -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : ℚ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25
  | .HalfDollar => 50

/-- Returns the probability of a coin landing heads -/
def headsProbability (c : Coin) : ℚ :=
  match c with
  | .HalfDollar => 1/3
  | _ => 1/2

/-- The set of all coins -/
def coinSet : List Coin := [Coin.Penny, Coin.Nickel, Coin.Dime, Coin.Quarter, Coin.HalfDollar]

/-- Calculates the expected value for a single coin -/
def expectedValue (c : Coin) : ℚ := (headsProbability c) * (coinValue c)

/-- Theorem: The expected value of the amount of money from coins that come up heads is 223/6 cents -/
theorem expected_value_of_heads : 
  (coinSet.map expectedValue).sum = 223/6 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_heads_l1464_146415


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_40_by_150_percent_l1464_146424

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial + (percentage / 100) * initial = initial * (1 + percentage / 100) := by sorry

theorem increase_40_by_150_percent :
  40 + (150 / 100) * 40 = 100 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_40_by_150_percent_l1464_146424


namespace NUMINAMATH_CALUDE_kishore_savings_l1464_146494

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 3940
def savings_percentage : ℚ := 1 / 10

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous

theorem kishore_savings :
  let monthly_salary := total_expenses / (1 - savings_percentage)
  (monthly_salary * savings_percentage).floor = 2160 := by
  sorry

end NUMINAMATH_CALUDE_kishore_savings_l1464_146494


namespace NUMINAMATH_CALUDE_digits_of_8_power_10_times_3_power_15_l1464_146446

theorem digits_of_8_power_10_times_3_power_15 : ∃ (n : ℕ), 
  (10 ^ (n - 1) ≤ 8^10 * 3^15) ∧ (8^10 * 3^15 < 10^n) ∧ (n = 12) := by
  sorry

end NUMINAMATH_CALUDE_digits_of_8_power_10_times_3_power_15_l1464_146446


namespace NUMINAMATH_CALUDE_marks_remaining_money_l1464_146413

def initial_money : ℕ := 85
def num_books : ℕ := 10
def book_cost : ℕ := 5

theorem marks_remaining_money :
  initial_money - (num_books * book_cost) = 35 := by
  sorry

end NUMINAMATH_CALUDE_marks_remaining_money_l1464_146413


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_zero_l1464_146420

theorem fraction_zero_implies_x_zero (x : ℝ) : 
  (x^2 - x) / (x - 1) = 0 ∧ x - 1 ≠ 0 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_zero_l1464_146420


namespace NUMINAMATH_CALUDE_g_neg_two_l1464_146408

def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem g_neg_two : g (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_g_neg_two_l1464_146408


namespace NUMINAMATH_CALUDE_anayet_driving_time_l1464_146409

/-- Proves that Anayet drove for 2 hours given the conditions of the problem -/
theorem anayet_driving_time 
  (total_distance : ℝ)
  (amoli_speed : ℝ)
  (amoli_time : ℝ)
  (anayet_speed : ℝ)
  (remaining_distance : ℝ)
  (h1 : total_distance = 369)
  (h2 : amoli_speed = 42)
  (h3 : amoli_time = 3)
  (h4 : anayet_speed = 61)
  (h5 : remaining_distance = 121)
  : ∃ (anayet_time : ℝ), anayet_time = 2 ∧ 
    total_distance = amoli_speed * amoli_time + anayet_speed * anayet_time + remaining_distance :=
by
  sorry


end NUMINAMATH_CALUDE_anayet_driving_time_l1464_146409


namespace NUMINAMATH_CALUDE_max_sum_with_constraint_l1464_146484

theorem max_sum_with_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 2) :
  a + b ≤ 3/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_constraint_l1464_146484


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1464_146480

/-- A quadratic function f(x) = 3x^2 + ax + b where f(x-1) is an even function -/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ 3 * x^2 + a * x + b

/-- The property that f(x-1) is an even function -/
def f_even (a b : ℝ) : Prop := ∀ x, f a b (x - 1) = f a b (-x - 1)

theorem quadratic_inequality (a b : ℝ) (h : f_even a b) :
  f a b (-1) < f a b (-3/2) ∧ f a b (-3/2) < f a b (3/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1464_146480


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l1464_146488

theorem logarithm_sum_simplification : 
  1 / (Real.log 3 / Real.log 20 + 1) + 
  1 / (Real.log 5 / Real.log 12 + 1) + 
  1 / (Real.log 7 / Real.log 8 + 1) = 2 := by sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l1464_146488


namespace NUMINAMATH_CALUDE_parallel_vectors_x_equals_one_l1464_146417

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_equals_one :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2*x + 1, 3)
  let b : ℝ × ℝ := (2 - x, 1)
  parallel a b → x = 1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_equals_one_l1464_146417


namespace NUMINAMATH_CALUDE_valid_reasoning_methods_l1464_146470

-- Define the set of reasoning methods
inductive ReasoningMethod
| Method1
| Method2
| Method3
| Method4

-- Define a predicate for valid analogical reasoning
def is_valid_analogical_reasoning (m : ReasoningMethod) : Prop :=
  m = ReasoningMethod.Method1

-- Define a predicate for valid inductive reasoning
def is_valid_inductive_reasoning (m : ReasoningMethod) : Prop :=
  m = ReasoningMethod.Method2 ∨ m = ReasoningMethod.Method4

-- Define a predicate for valid reasoning
def is_valid_reasoning (m : ReasoningMethod) : Prop :=
  is_valid_analogical_reasoning m ∨ is_valid_inductive_reasoning m

-- Theorem statement
theorem valid_reasoning_methods :
  {m : ReasoningMethod | is_valid_reasoning m} =
  {ReasoningMethod.Method1, ReasoningMethod.Method2, ReasoningMethod.Method4} :=
by sorry

end NUMINAMATH_CALUDE_valid_reasoning_methods_l1464_146470


namespace NUMINAMATH_CALUDE_prime_factorization_equality_l1464_146410

theorem prime_factorization_equality : 5 * 13 * 31 - 2 = 3 * 11 * 61 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_equality_l1464_146410


namespace NUMINAMATH_CALUDE_water_fountain_problem_l1464_146400

/-- Represents the number of men needed to build a water fountain -/
def men_needed (length : ℕ) (days : ℕ) (men : ℕ) : Prop :=
  ∃ (k : ℚ), k * (men * days) = length

theorem water_fountain_problem :
  men_needed 56 42 60 ∧ men_needed 7 3 35 →
  (∀ l₁ d₁ m₁ l₂ d₂ m₂,
    men_needed l₁ d₁ m₁ → men_needed l₂ d₂ m₂ →
    (m₁ * d₁ : ℚ) / l₁ = (m₂ * d₂ : ℚ) / l₂) →
  60 = (35 * 3 * 56) / (7 * 42) :=
by sorry

end NUMINAMATH_CALUDE_water_fountain_problem_l1464_146400


namespace NUMINAMATH_CALUDE_bakers_friend_cakes_l1464_146463

/-- Given that Baker made 155 cakes initially and now has 15 cakes remaining,
    prove that Baker's friend bought 140 cakes. -/
theorem bakers_friend_cakes :
  let initial_cakes : ℕ := 155
  let remaining_cakes : ℕ := 15
  let friend_bought : ℕ := initial_cakes - remaining_cakes
  friend_bought = 140 := by sorry

end NUMINAMATH_CALUDE_bakers_friend_cakes_l1464_146463


namespace NUMINAMATH_CALUDE_seashells_count_l1464_146433

theorem seashells_count (mary_shells jessica_shells : ℕ) 
  (h1 : mary_shells = 18) 
  (h2 : jessica_shells = 41) : 
  mary_shells + jessica_shells = 59 := by
  sorry

end NUMINAMATH_CALUDE_seashells_count_l1464_146433


namespace NUMINAMATH_CALUDE_old_card_sale_amount_l1464_146441

def initial_cost : ℕ := 1200
def new_card_cost : ℕ := 500
def total_spent : ℕ := 1400

theorem old_card_sale_amount : 
  initial_cost + new_card_cost - total_spent = 300 :=
by sorry

end NUMINAMATH_CALUDE_old_card_sale_amount_l1464_146441


namespace NUMINAMATH_CALUDE_arithmetic_sequence_from_equation_l1464_146499

theorem arithmetic_sequence_from_equation (a b c : ℝ) :
  (2*b - a)^2 + (2*b - c)^2 = 2*(2*b^2 - a*c) →
  b = (a + c) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_from_equation_l1464_146499


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l1464_146407

theorem greatest_value_quadratic_inequality :
  ∃ (a_max : ℝ), a_max = 9 ∧
  (∀ a : ℝ, a^2 - 14*a + 45 ≤ 0 → a ≤ a_max) ∧
  (a_max^2 - 14*a_max + 45 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l1464_146407


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1464_146497

theorem digit_sum_problem :
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
    a + c = 10 →
    b + c + 1 = 10 →
    a + d + 1 = 11 →
    1000 * a + 100 * b + 10 * c + d + 100 * c + 10 * a = 1100 →
    a + b + c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1464_146497


namespace NUMINAMATH_CALUDE_parabola_line_intersection_property_l1464_146474

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop :=
  ∃ (k : ℝ), y = k * (x - 1) ∨ x = 1

-- Theorem statement
theorem parabola_line_intersection_property 
  (x₁ y₁ x₂ y₂ : ℝ) (h_distinct : (x₁, y₁) ≠ (x₂, y₂)) :
  (∀ x y, line_through_points x₁ y₁ x₂ y₂ x y → line_through_focus x y → 
    parabola x₁ y₁ → parabola x₂ y₂ → x₁ * x₂ = 1) ∧
  (∃ x₁' y₁' x₂' y₂', (x₁', y₁') ≠ (x₂', y₂') ∧
    parabola x₁' y₁' ∧ parabola x₂' y₂' ∧ x₁' * x₂' = 1 ∧
    ¬(∀ x y, line_through_points x₁' y₁' x₂' y₂' x y → line_through_focus x y)) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_property_l1464_146474


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1464_146434

/-- The polynomial function we're analyzing -/
def g (x : ℝ) : ℝ := x^10 + 9*x^9 + 20*x^8 + 2000*x^7 - 1500*x^6

/-- Theorem stating that g(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! x : ℝ, x > 0 ∧ g x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1464_146434


namespace NUMINAMATH_CALUDE_tape_recorder_cost_l1464_146440

theorem tape_recorder_cost : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧
  170 ≤ x * y ∧ x * y ≤ 195 ∧
  y = 2 * x + 2 ∧
  x * y = 180 := by
  sorry

end NUMINAMATH_CALUDE_tape_recorder_cost_l1464_146440


namespace NUMINAMATH_CALUDE_shirley_eggs_theorem_l1464_146469

/-- The number of eggs Shirley started with -/
def initial_eggs : ℕ := 98

/-- The number of eggs Shirley bought -/
def bought_eggs : ℕ := 8

/-- The total number of eggs Shirley ended with -/
def final_eggs : ℕ := 106

/-- Theorem stating that the initial number of eggs plus the bought eggs equals the final number of eggs -/
theorem shirley_eggs_theorem : initial_eggs + bought_eggs = final_eggs := by
  sorry

end NUMINAMATH_CALUDE_shirley_eggs_theorem_l1464_146469


namespace NUMINAMATH_CALUDE_max_n_inequality_l1464_146495

theorem max_n_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (∀ n : ℝ, 1 / (a - b) + 1 / (b - c) ≥ n / (a - c)) →
  (∃ n : ℝ, 1 / (a - b) + 1 / (b - c) = n / (a - c) ∧
            ∀ m : ℝ, 1 / (a - b) + 1 / (b - c) ≥ m / (a - c) → m ≤ n) →
  (∃ n : ℝ, n = 4 ∧
            1 / (a - b) + 1 / (b - c) = n / (a - c) ∧
            ∀ m : ℝ, 1 / (a - b) + 1 / (b - c) ≥ m / (a - c) → m ≤ n) :=
by sorry


end NUMINAMATH_CALUDE_max_n_inequality_l1464_146495


namespace NUMINAMATH_CALUDE_count_even_positive_factors_l1464_146411

/-- The number of even positive factors of n, where n = 2^4 * 3^2 * 5^2 * 7 -/
def evenPositiveFactors (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of even positive factors of n is 72 -/
theorem count_even_positive_factors :
  ∃ n : ℕ, n = 2^4 * 3^2 * 5^2 * 7 ∧ evenPositiveFactors n = 72 :=
sorry

end NUMINAMATH_CALUDE_count_even_positive_factors_l1464_146411


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1464_146454

theorem polynomial_simplification (q : ℝ) :
  (2 * q^3 - 7 * q^2 + 3 * q - 4) + (5 * q^2 - 4 * q + 8) = 2 * q^3 - 2 * q^2 - q + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1464_146454


namespace NUMINAMATH_CALUDE_cost_of_grapes_and_pineapple_l1464_146478

/-- Represents the price of fruits and their combinations -/
structure FruitPrices where
  f : ℚ  -- price of one piece of fruit
  g : ℚ  -- price of a bunch of grapes
  p : ℚ  -- price of a pineapple
  φ : ℚ  -- price of a pack of figs

/-- The conditions given in the problem -/
def satisfiesConditions (prices : FruitPrices) : Prop :=
  3 * prices.f + 2 * prices.g + prices.p + prices.φ = 36 ∧
  prices.φ = 3 * prices.f ∧
  prices.p = prices.f + prices.g

/-- The theorem to be proved -/
theorem cost_of_grapes_and_pineapple (prices : FruitPrices) 
  (h : satisfiesConditions prices) : 
  2 * prices.g + prices.p = (15 * prices.g + 36) / 7 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_grapes_and_pineapple_l1464_146478


namespace NUMINAMATH_CALUDE_veronica_cherry_pie_l1464_146425

/-- Given that:
  - There are 80 cherries in one pound
  - It takes 10 minutes to pit 20 cherries
  - It takes Veronica 2 hours to pit all the cherries
  Prove that Veronica needs 3 pounds of cherries for her pie. -/
theorem veronica_cherry_pie (cherries_per_pound : ℕ) (pit_time : ℕ) (pit_amount : ℕ) (total_time : ℕ) :
  cherries_per_pound = 80 →
  pit_time = 10 →
  pit_amount = 20 →
  total_time = 120 →
  (total_time / pit_time) * pit_amount / cherries_per_pound = 3 :=
by sorry

end NUMINAMATH_CALUDE_veronica_cherry_pie_l1464_146425


namespace NUMINAMATH_CALUDE_opposite_to_83_l1464_146442

/-- Represents a circle with 100 equally spaced points -/
def Circle := Fin 100

/-- A function assigning numbers 1 to 100 to the points on the circle -/
def numbering : Circle → Nat :=
  sorry

/-- Predicate to check if a number is opposite to another on the circle -/
def is_opposite (a b : Circle) : Prop :=
  sorry

/-- Predicate to check if numbers less than k are evenly distributed -/
def evenly_distributed (k : Nat) : Prop :=
  sorry

theorem opposite_to_83 (h : ∀ k, evenly_distributed k) :
  ∃ n : Circle, numbering n = 84 ∧ is_opposite n (⟨82, sorry⟩ : Circle) :=
sorry

end NUMINAMATH_CALUDE_opposite_to_83_l1464_146442


namespace NUMINAMATH_CALUDE_absolute_value_of_w_l1464_146419

theorem absolute_value_of_w (w : ℂ) (h : w^2 = -48 + 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_w_l1464_146419


namespace NUMINAMATH_CALUDE_tim_sweets_multiple_of_four_l1464_146481

/-- The number of grape-flavored sweets Peter has -/
def peter_sweets : ℕ := 44

/-- The largest possible number of sweets in each tray without remainder -/
def tray_size : ℕ := 4

/-- The number of orange-flavored sweets Tim has -/
def tim_sweets : ℕ := sorry

theorem tim_sweets_multiple_of_four :
  ∃ k : ℕ, tim_sweets = k * tray_size ∧ peter_sweets % tray_size = 0 :=
by sorry

end NUMINAMATH_CALUDE_tim_sweets_multiple_of_four_l1464_146481


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1464_146430

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (7 + 3 * z) = 13 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1464_146430


namespace NUMINAMATH_CALUDE_janet_tickets_l1464_146450

/-- The number of tickets needed for Janet's amusement park rides -/
def total_tickets (roller_coaster_tickets_per_ride : ℕ) 
                  (giant_slide_tickets_per_ride : ℕ) 
                  (roller_coaster_rides : ℕ) 
                  (giant_slide_rides : ℕ) : ℕ :=
  roller_coaster_tickets_per_ride * roller_coaster_rides + 
  giant_slide_tickets_per_ride * giant_slide_rides

/-- Theorem: Janet needs 47 tickets for her amusement park rides -/
theorem janet_tickets : 
  total_tickets 5 3 7 4 = 47 := by
  sorry

end NUMINAMATH_CALUDE_janet_tickets_l1464_146450


namespace NUMINAMATH_CALUDE_basketball_game_equations_l1464_146451

/-- Represents a basketball team's game results -/
structure BasketballTeam where
  gamesWon : ℕ
  gamesLost : ℕ

/-- Calculates the total points earned by a basketball team -/
def totalPoints (team : BasketballTeam) : ℕ :=
  2 * team.gamesWon + team.gamesLost

theorem basketball_game_equations (team : BasketballTeam) 
  (h1 : team.gamesWon + team.gamesLost = 12) 
  (h2 : totalPoints team = 20) : 
  (team.gamesWon + team.gamesLost = 12) ∧ (2 * team.gamesWon + team.gamesLost = 20) := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_equations_l1464_146451


namespace NUMINAMATH_CALUDE_abc_product_l1464_146457

theorem abc_product (a b c : ℕ) : 
  Nat.Prime a → 
  Nat.Prime b → 
  Nat.Prime c → 
  a * b * c < 10000 → 
  2 * a + 3 * b = c → 
  4 * a + c + 1 = 4 * b → 
  a * b * c = 1118 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1464_146457


namespace NUMINAMATH_CALUDE_probability_not_greater_than_two_l1464_146421

def card_set : Finset ℕ := {1, 2, 3, 4}

theorem probability_not_greater_than_two :
  (card_set.filter (λ x => x ≤ 2)).card / card_set.card = (1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_greater_than_two_l1464_146421


namespace NUMINAMATH_CALUDE_triangle_area_approx_l1464_146437

/-- The area of a triangle with sides 30, 28, and 10 is approximately 139.94 -/
theorem triangle_area_approx : ∃ (area : ℝ), 
  let a : ℝ := 30
  let b : ℝ := 28
  let c : ℝ := 10
  let s : ℝ := (a + b + c) / 2
  area = Real.sqrt (s * (s - a) * (s - b) * (s - c)) ∧ 
  abs (area - 139.94) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_approx_l1464_146437


namespace NUMINAMATH_CALUDE_parabola_vertex_equation_l1464_146436

/-- A parabola with vertex coordinates (-2, 0) is represented by the equation y = (x+2)^2 -/
theorem parabola_vertex_equation :
  ∀ (x y : ℝ), (∃ (a : ℝ), y = a * (x + 2)^2) ↔ 
  (y = (x + 2)^2 ∧ (∀ (x₀ y₀ : ℝ), y₀ = (x₀ + 2)^2 → y₀ ≥ 0 ∧ (y₀ = 0 → x₀ = -2))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_equation_l1464_146436


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1464_146447

/-- The coefficient of x^n in the expansion of (x-1/x)^m -/
def coeff (m n : ℕ) : ℤ :=
  if (m - n) % 2 = 0 
  then (-1)^((m - n) / 2) * (m.choose ((m - n) / 2))
  else 0

/-- The coefficient of x^6 in the expansion of (x^2+a)(x-1/x)^10 -/
def coeff_x6 (a : ℤ) : ℤ := coeff 10 6 + a * coeff 10 4

theorem expansion_coefficient (a : ℤ) : 
  coeff_x6 a = -30 → a = 2 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1464_146447


namespace NUMINAMATH_CALUDE_three_digit_sum_divisibility_l1464_146458

theorem three_digit_sum_divisibility (a b : ℕ) : 
  (100 * 2 + 10 * a + 3) + 326 = (500 + 10 * b + 9) → 
  (500 + 10 * b + 9) % 9 = 0 →
  a + b = 6 := by sorry

end NUMINAMATH_CALUDE_three_digit_sum_divisibility_l1464_146458


namespace NUMINAMATH_CALUDE_decimal_to_base_k_l1464_146423

/-- Given that the decimal number 26 is equal to the base-k number 32, prove that k = 8 -/
theorem decimal_to_base_k (k : ℕ) (h : 3 * k + 2 = 26) : k = 8 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_base_k_l1464_146423


namespace NUMINAMATH_CALUDE_existence_of_c_l1464_146414

theorem existence_of_c (a b : ℝ) : ∃ c ∈ Set.Icc 0 1, |a * c + b + 1 / (c + 1)| ≥ 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_c_l1464_146414


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l1464_146485

/-- Represents the problem from "The Mathematical Classic of Sunzi" --/
theorem sunzi_wood_measurement (x y : ℝ) :
  y = x + 4.5 ∧ 0.5 * y = x - 1 →
  (y - x = 4.5 ∧ y / 2 - x = 1) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l1464_146485


namespace NUMINAMATH_CALUDE_ticket_sales_l1464_146429

theorem ticket_sales (adult_price children_price senior_price discount : ℕ)
  (total_receipts total_attendance : ℕ)
  (discounted_adults discounted_children : ℕ)
  (h1 : adult_price = 25)
  (h2 : children_price = 15)
  (h3 : senior_price = 20)
  (h4 : discount = 5)
  (h5 : discounted_adults = 50)
  (h6 : discounted_children = 30)
  (h7 : total_receipts = 7200)
  (h8 : total_attendance = 400) :
  ∃ (regular_adults regular_children senior : ℕ),
    regular_adults + discounted_adults = 2 * senior ∧
    regular_adults + discounted_adults + regular_children + discounted_children + senior = total_attendance ∧
    regular_adults * adult_price + discounted_adults * (adult_price - discount) +
    regular_children * children_price + discounted_children * (children_price - discount) +
    senior * senior_price = total_receipts ∧
    regular_adults = 102 ∧
    regular_children = 142 ∧
    senior = 76 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_l1464_146429


namespace NUMINAMATH_CALUDE_multiply_and_add_l1464_146459

theorem multiply_and_add : 45 * 55 + 45 * 45 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_l1464_146459


namespace NUMINAMATH_CALUDE_men_science_majors_percentage_l1464_146465

/-- Represents the composition of a college class -/
structure ClassComposition where
  total_students : ℕ
  women_science_majors : ℕ
  non_science_majors : ℕ
  men : ℕ

/-- Calculates the percentage of men who are science majors -/
def percentage_men_science_majors (c : ClassComposition) : ℚ :=
  let total_science_majors := c.total_students - c.non_science_majors
  let men_science_majors := total_science_majors - c.women_science_majors
  (men_science_majors : ℚ) / (c.men : ℚ) * 100

/-- Theorem stating the percentage of men who are science majors -/
theorem men_science_majors_percentage (c : ClassComposition) 
  (h1 : c.women_science_majors = c.total_students * 30 / 100)
  (h2 : c.non_science_majors = c.total_students * 60 / 100)
  (h3 : c.men = c.total_students * 40 / 100) :
  percentage_men_science_majors c = 25 := by
  sorry

end NUMINAMATH_CALUDE_men_science_majors_percentage_l1464_146465


namespace NUMINAMATH_CALUDE_quadratic_fit_energy_production_l1464_146406

/-- Represents a quadratic function of the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Theorem: There exists a quadratic function that fits the given data points
    and predicts the correct value for 2007 -/
theorem quadratic_fit_energy_production : ∃ f : QuadraticFunction,
  f.evaluate 0 = 8.6 ∧
  f.evaluate 5 = 10.4 ∧
  f.evaluate 10 = 12.9 ∧
  f.evaluate 15 = 16.1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_fit_energy_production_l1464_146406


namespace NUMINAMATH_CALUDE_triangle_property_l1464_146477

/-- Given a triangle ABC with sides a, b, and c satisfying the equation
    a^2 + b^2 + c^2 + 50 = 6a + 8b + 10c, prove that it is a right-angled
    triangle with area 6. -/
theorem triangle_property (a b c : ℝ) (h : a^2 + b^2 + c^2 + 50 = 6*a + 8*b + 10*c) :
  (a = 3 ∧ b = 4 ∧ c = 5) ∧ a^2 + b^2 = c^2 ∧ (1/2) * a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1464_146477


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1464_146473

theorem line_intersects_circle (a : ℝ) (h : a ≥ 0) :
  ∃ (x y : ℝ), (a * x - y + Real.sqrt 2 * a = 0) ∧ (x^2 + y^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1464_146473


namespace NUMINAMATH_CALUDE_weight_replacement_l1464_146491

theorem weight_replacement (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) : 
  initial_count = 5 → 
  replaced_weight = 65 → 
  avg_increase = 1.5 → 
  (initial_count : ℝ) * avg_increase + replaced_weight = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l1464_146491


namespace NUMINAMATH_CALUDE_parabola_parameter_l1464_146444

/-- A parabola with equation y = ax² and latus rectum y = -1/2 has a = 1/2 --/
theorem parabola_parameter (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x^2) ∧  -- Parabola equation
  (∃ (y : ℝ), y = -1/2) →       -- Latus rectum equation
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_parameter_l1464_146444


namespace NUMINAMATH_CALUDE_hypotenuse_length_triangle_area_l1464_146438

-- Define a right triangle with legs 30 and 40
def right_triangle (a b c : ℝ) : Prop :=
  a = 30 ∧ b = 40 ∧ c^2 = a^2 + b^2

-- Theorem for the hypotenuse
theorem hypotenuse_length (a b c : ℝ) (h : right_triangle a b c) : c = 50 := by
  sorry

-- Theorem for the area
theorem triangle_area (a b : ℝ) (h : a = 30 ∧ b = 40) : (1/2) * a * b = 600 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_triangle_area_l1464_146438


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1464_146486

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmeticSequence a →
  (a 2 = 0) →
  (S 3 + S 4 = 6) →
  (a 5 + a 6 = 21) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1464_146486


namespace NUMINAMATH_CALUDE_shift_sin_left_specific_sin_shift_l1464_146482

/-- Shifting a sinusoidal function to the left --/
theorem shift_sin_left (A ω φ δ : ℝ) :
  let f (x : ℝ) := A * Real.sin (ω * x + φ)
  let g (x : ℝ) := A * Real.sin (ω * (x + δ) + φ)
  ∀ x, f (x - δ) = g x := by sorry

/-- The specific shift problem --/
theorem specific_sin_shift :
  let f (x : ℝ) := 3 * Real.sin (2 * x - π / 6)
  let g (x : ℝ) := 3 * Real.sin (2 * x + π / 3)
  ∀ x, f (x - π / 4) = g x := by sorry

end NUMINAMATH_CALUDE_shift_sin_left_specific_sin_shift_l1464_146482


namespace NUMINAMATH_CALUDE_jessica_birth_year_l1464_146456

theorem jessica_birth_year (first_amc8_year : ℕ) (jessica_age : ℕ) :
  first_amc8_year = 1985 →
  jessica_age = 15 →
  (first_amc8_year + 10 - 1) - jessica_age = 1979 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_birth_year_l1464_146456


namespace NUMINAMATH_CALUDE_curve_properties_l1464_146428

-- Define the function f(x) = x^3 - x
def f (x : ℝ) : ℝ := x^3 - x

-- State the theorem
theorem curve_properties :
  -- Part I: f'(1) = 2
  (deriv f) 1 = 2 ∧
  -- Part II: Tangent line equation at P(1, f(1)) is 2x - y - 2 = 0
  (∃ (m b : ℝ), m = 2 ∧ b = -2 ∧ ∀ x y, y = m * (x - 1) + f 1 ↔ 2 * x - y - 2 = 0) ∧
  -- Part III: Extreme values
  (∃ (x1 x2 : ℝ), 
    x1 = -Real.sqrt 3 / 3 ∧ 
    x2 = Real.sqrt 3 / 3 ∧
    f x1 = -2 * Real.sqrt 3 / 9 ∧
    f x2 = -2 * Real.sqrt 3 / 9 ∧
    (∀ x, f x ≥ -2 * Real.sqrt 3 / 9) ∧
    (∀ x, (deriv f) x = 0 → x = x1 ∨ x = x2)) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l1464_146428


namespace NUMINAMATH_CALUDE_percentage_of_students_owning_only_cats_l1464_146416

theorem percentage_of_students_owning_only_cats
  (total_students : ℕ)
  (dog_owners : ℕ)
  (cat_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : dog_owners = 150)
  (h3 : cat_owners = 80)
  (h4 : both_owners = 25) :
  (cat_owners - both_owners) / total_students * 100 = 11 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_students_owning_only_cats_l1464_146416


namespace NUMINAMATH_CALUDE_triangle_angles_l1464_146464

theorem triangle_angles (x y z : ℝ) : 
  (y + 150 + 160 = 360) →
  (z + 150 + 160 = 360) →
  (x + y + z = 180) →
  (x = 80 ∧ y = 50 ∧ z = 50) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_l1464_146464


namespace NUMINAMATH_CALUDE_sin_cos_sum_20_40_l1464_146435

theorem sin_cos_sum_20_40 :
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) +
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_20_40_l1464_146435


namespace NUMINAMATH_CALUDE_spacing_change_at_20th_post_l1464_146475

/-- Represents the fence with its posts and spacings -/
structure Fence where
  initialSpacing : ℝ
  changedSpacing : ℝ
  changePost : ℕ

/-- The fence satisfies the given conditions -/
def satisfiesConditions (f : Fence) : Prop :=
  f.initialSpacing > f.changedSpacing ∧
  f.initialSpacing * 15 = 48 ∧
  f.changedSpacing * (28 - f.changePost) + f.initialSpacing * (f.changePost - 16) = 36 ∧
  f.changePost > 16 ∧ f.changePost ≤ 28

/-- The theorem stating that the 20th post is where the spacing changes -/
theorem spacing_change_at_20th_post (f : Fence) (h : satisfiesConditions f) : f.changePost = 20 := by
  sorry

end NUMINAMATH_CALUDE_spacing_change_at_20th_post_l1464_146475


namespace NUMINAMATH_CALUDE_price_change_calculation_l1464_146468

theorem price_change_calculation (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_decrease := 0.8 * initial_price
  let final_price := 1.04 * initial_price
  ∃ x : ℝ, price_after_decrease * (1 + x / 100) = final_price ∧ x = 30 :=
by sorry

end NUMINAMATH_CALUDE_price_change_calculation_l1464_146468


namespace NUMINAMATH_CALUDE_circle_symmetry_l1464_146487

-- Define the original circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y + 24 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x - 3*y - 5 = 0

-- Define the symmetric circle S
def S (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), S x y ↔ ∃ (x' y' : ℝ), C x' y' ∧
  (∃ (m : ℝ), l m ((y + y')/2) ∧ m = (x + x')/2) ∧
  ((y - y')/(x - x') = -3 ∨ x = x') :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1464_146487


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l1464_146412

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_roots_zero
  (a b c : ℝ)
  (h : ∀ x : ℝ, QuadraticPolynomial a b c (x^3 - x) ≥ QuadraticPolynomial a b c (x^2 - 1)) :
  b / a = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l1464_146412


namespace NUMINAMATH_CALUDE_student_ticket_price_l1464_146404

theorem student_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (general_tickets : ℕ) 
  (general_price : ℕ) 
  (h1 : total_tickets = 525)
  (h2 : total_revenue = 2876)
  (h3 : general_tickets = 388)
  (h4 : general_price = 6) :
  ∃ (student_price : ℕ),
    student_price = 4 ∧
    (total_tickets - general_tickets) * student_price + general_tickets * general_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_student_ticket_price_l1464_146404


namespace NUMINAMATH_CALUDE_mulch_price_per_pound_l1464_146496

/-- Given the cost of mulch in tons, calculate the price per pound -/
theorem mulch_price_per_pound (cost : ℝ) (tons : ℝ) (pounds_per_ton : ℝ) : 
  cost = 15000 → tons = 3 → pounds_per_ton = 2000 →
  cost / (tons * pounds_per_ton) = 2.5 := by sorry

end NUMINAMATH_CALUDE_mulch_price_per_pound_l1464_146496


namespace NUMINAMATH_CALUDE_a2023_coordinates_l1464_146418

def companion_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2) + 1, p.1 + 1)

def sequence_point (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => (2, 4)
  | n + 1 => companion_point (sequence_point n)

theorem a2023_coordinates :
  sequence_point 2022 = (-2, -2) :=
sorry

end NUMINAMATH_CALUDE_a2023_coordinates_l1464_146418


namespace NUMINAMATH_CALUDE_seonhos_wallet_problem_l1464_146443

theorem seonhos_wallet_problem (initial_money : ℚ) : 
  (initial_money / 4) * (1 / 3) = 2500 → initial_money = 10000 := by sorry

end NUMINAMATH_CALUDE_seonhos_wallet_problem_l1464_146443


namespace NUMINAMATH_CALUDE_prob_red_or_black_is_three_fourths_prob_red_or_black_or_white_is_eleven_twelfths_l1464_146462

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Black
  | White
  | Green

/-- Represents the box of balls -/
structure BallBox where
  total : ℕ
  red : ℕ
  black : ℕ
  white : ℕ
  green : ℕ
  sum_constraint : red + black + white + green = total

/-- Calculates the probability of drawing a ball of a specific color -/
def prob_color (box : BallBox) (color : BallColor) : ℚ :=
  match color with
  | BallColor.Red => box.red / box.total
  | BallColor.Black => box.black / box.total
  | BallColor.White => box.white / box.total
  | BallColor.Green => box.green / box.total

/-- The box described in the problem -/
def problem_box : BallBox :=
  { total := 12
    red := 5
    black := 4
    white := 2
    green := 1
    sum_constraint := by simp }

theorem prob_red_or_black_is_three_fourths :
    prob_color problem_box BallColor.Red + prob_color problem_box BallColor.Black = 3/4 := by
  sorry

theorem prob_red_or_black_or_white_is_eleven_twelfths :
    prob_color problem_box BallColor.Red + prob_color problem_box BallColor.Black +
    prob_color problem_box BallColor.White = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_or_black_is_three_fourths_prob_red_or_black_or_white_is_eleven_twelfths_l1464_146462


namespace NUMINAMATH_CALUDE_no_valid_tiling_exists_l1464_146461

/-- Represents a chessboard square --/
inductive Square
| Black
| White

/-- Represents a 2x1 domino --/
structure Domino :=
(first : Square)
(second : Square)

/-- Represents the modified 8x8 chessboard with corners removed --/
def ModifiedChessboard := Fin 62 → Square

/-- A tiling of the modified chessboard using dominos --/
def Tiling := Fin 31 → Domino

/-- Checks if a tiling is valid for the modified chessboard --/
def is_valid_tiling (board : ModifiedChessboard) (tiling : Tiling) : Prop :=
  ∀ i j : Fin 62, i ≠ j → 
    ∃ k : Fin 31, (tiling k).first = board i ∧ (tiling k).second = board j

/-- The main theorem stating that no valid tiling exists --/
theorem no_valid_tiling_exists :
  ¬∃ (board : ModifiedChessboard) (tiling : Tiling), is_valid_tiling board tiling :=
sorry

end NUMINAMATH_CALUDE_no_valid_tiling_exists_l1464_146461


namespace NUMINAMATH_CALUDE_sum_of_composite_function_l1464_146498

def p (x : ℝ) : ℝ := x^2 - 3*x + 2

def q (x : ℝ) : ℝ := -x^2

def eval_points : List ℝ := [0, 1, 2, 3, 4]

theorem sum_of_composite_function :
  (eval_points.map (λ x => q (p x))).sum = -12 := by sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_l1464_146498


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_zero_l1464_146439

theorem factorial_fraction_equals_zero : 
  (5 * Nat.factorial 7 - 35 * Nat.factorial 6) / Nat.factorial 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_zero_l1464_146439


namespace NUMINAMATH_CALUDE_money_distribution_l1464_146483

theorem money_distribution (a b c : ℤ) 
  (total : a + b + c = 500)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 360) :
  c = 60 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1464_146483


namespace NUMINAMATH_CALUDE_original_number_proof_l1464_146455

theorem original_number_proof (increased_number : ℝ) (increase_percentage : ℝ) :
  increased_number = 480 ∧ increase_percentage = 0.2 →
  (1 + increase_percentage) * (increased_number / (1 + increase_percentage)) = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1464_146455


namespace NUMINAMATH_CALUDE_total_turnips_l1464_146402

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139)
  (h2 : benny_turnips = 113) :
  melanie_turnips + benny_turnips = 252 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l1464_146402


namespace NUMINAMATH_CALUDE_reflect_point_across_y_axis_l1464_146453

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

theorem reflect_point_across_y_axis :
  let P : Point := ⟨5, -1⟩
  reflectAcrossYAxis P = ⟨-5, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_reflect_point_across_y_axis_l1464_146453


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1464_146427

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 1/a + 1/b + 1/c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1464_146427


namespace NUMINAMATH_CALUDE_furniture_cost_price_l1464_146448

theorem furniture_cost_price (computer_table_price chair_price bookshelf_price : ℝ)
  (h1 : computer_table_price = 8091)
  (h2 : chair_price = 5346)
  (h3 : bookshelf_price = 11700)
  (computer_table_markup : ℝ)
  (h4 : computer_table_markup = 0.24)
  (chair_markup : ℝ)
  (h5 : chair_markup = 0.18)
  (chair_discount : ℝ)
  (h6 : chair_discount = 0.05)
  (bookshelf_markup : ℝ)
  (h7 : bookshelf_markup = 0.30)
  (sales_tax : ℝ)
  (h8 : sales_tax = 0.045) :
  ∃ (computer_table_cost chair_cost bookshelf_cost : ℝ),
    computer_table_cost = computer_table_price / (1 + computer_table_markup) ∧
    chair_cost = chair_price / ((1 + chair_markup) * (1 - chair_discount)) ∧
    bookshelf_cost = bookshelf_price / (1 + bookshelf_markup) ∧
    computer_table_cost + chair_cost + bookshelf_cost = 20295 :=
by sorry

end NUMINAMATH_CALUDE_furniture_cost_price_l1464_146448


namespace NUMINAMATH_CALUDE_triangle_properties_l1464_146452

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  (2 * Real.sin t.B * Real.cos t.A = Real.sin t.A * Real.cos t.C + Real.cos t.A * Real.sin t.C) →
  (t.A = π / 3) ∧
  (t.A = π / 3 ∧ t.a = 6) →
  ∃ p : Real, 12 < p ∧ p ≤ 18 ∧ p = t.a + t.b + t.c :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1464_146452


namespace NUMINAMATH_CALUDE_toy_average_price_l1464_146479

theorem toy_average_price (n : ℕ) (dhoni_avg : ℚ) (david_price : ℚ) : 
  n = 5 → dhoni_avg = 10 → david_price = 16 → 
  (n * dhoni_avg + david_price) / (n + 1) = 11 := by sorry

end NUMINAMATH_CALUDE_toy_average_price_l1464_146479


namespace NUMINAMATH_CALUDE_largest_divisible_n_l1464_146472

theorem largest_divisible_n : 
  ∀ n : ℕ, n > 882 → ¬(n + 9 ∣ n^3 + 99) ∧ (882 + 9 ∣ 882^3 + 99) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l1464_146472


namespace NUMINAMATH_CALUDE_mike_song_book_price_l1464_146432

/-- The amount Mike received from selling the song book, given the cost of the trumpet and the net amount spent. -/
def song_book_price (trumpet_cost net_spent : ℚ) : ℚ :=
  trumpet_cost - net_spent

/-- Theorem stating that Mike sold the song book for $5.84, given the cost of the trumpet and the net amount spent. -/
theorem mike_song_book_price :
  let trumpet_cost : ℚ := 145.16
  let net_spent : ℚ := 139.32
  song_book_price trumpet_cost net_spent = 5.84 := by
  sorry

#eval song_book_price 145.16 139.32

end NUMINAMATH_CALUDE_mike_song_book_price_l1464_146432


namespace NUMINAMATH_CALUDE_min_value_theorem_l1464_146490

theorem min_value_theorem (a b : ℝ) (h1 : 2*a + 3*b = 6) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2*x + 3*y = 6 → 2/x + 3/y ≥ 25/6) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + 3*y = 6 ∧ 2/x + 3/y = 25/6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1464_146490


namespace NUMINAMATH_CALUDE_power_seven_mod_nine_l1464_146467

theorem power_seven_mod_nine : 7^123 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_nine_l1464_146467


namespace NUMINAMATH_CALUDE_symmetry_about_x_axis_and_origin_l1464_146489

/-- Given point A (2, -3) and point B symmetrical to A about the x-axis,
    prove that the coordinates of point C, which is symmetrical to point B about the origin,
    are (-2, -3). -/
theorem symmetry_about_x_axis_and_origin :
  let A : ℝ × ℝ := (2, -3)
  let B : ℝ × ℝ := (A.1, -A.2)  -- B is symmetrical to A about the x-axis
  let C : ℝ × ℝ := (-B.1, -B.2) -- C is symmetrical to B about the origin
  C = (-2, -3) := by sorry

end NUMINAMATH_CALUDE_symmetry_about_x_axis_and_origin_l1464_146489


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1464_146445

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  arithmetic : ∀ n, a (n + 1) = a n + d
  S : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence, if S₁/S₄ = 1/10, then S₃/S₅ = 2/5 -/
theorem arithmetic_sequence_sum_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 1 / seq.S 4 = 1 / 10) : 
  seq.S 3 / seq.S 5 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1464_146445


namespace NUMINAMATH_CALUDE_factorization_cube_minus_linear_l1464_146492

theorem factorization_cube_minus_linear (a b : ℝ) : a^3 * b - a * b = a * b * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cube_minus_linear_l1464_146492


namespace NUMINAMATH_CALUDE_total_cookies_baked_l1464_146449

/-- Calculates the total number of cookies baked by a baker -/
theorem total_cookies_baked 
  (chocolate_chip_batches : ℕ) 
  (cookies_per_batch : ℕ) 
  (oatmeal_cookies : ℕ) : 
  chocolate_chip_batches * cookies_per_batch + oatmeal_cookies = 10 :=
by
  sorry

#check total_cookies_baked 2 3 4

end NUMINAMATH_CALUDE_total_cookies_baked_l1464_146449


namespace NUMINAMATH_CALUDE_sum_xyz_equals_2014_l1464_146471

theorem sum_xyz_equals_2014 (x y z : ℝ) : 
  Real.sqrt (x - 3) + Real.sqrt (3 - x) + abs (x - y + 2010) + z^2 + 4*z + 4 = 0 → 
  x + y + z = 2014 := by
  sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_2014_l1464_146471


namespace NUMINAMATH_CALUDE_negation_existential_derivative_l1464_146460

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

theorem negation_existential_derivative :
  (¬ ∃ x : ℝ, f' x ≥ 0) ↔ (∀ x : ℝ, f' x < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_existential_derivative_l1464_146460


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l1464_146422

/-- A rectangular garden with length three times its width and width of 15 meters has an area of 675 square meters. -/
theorem rectangular_garden_area :
  ∀ (length width area : ℝ),
  length = 3 * width →
  width = 15 →
  area = length * width →
  area = 675 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l1464_146422


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1464_146493

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + 
    a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + a₉*(x+2)^9 + 
    a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1464_146493


namespace NUMINAMATH_CALUDE_one_fourth_of_6_8_l1464_146476

theorem one_fourth_of_6_8 : (6.8 : ℚ) / 4 = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_6_8_l1464_146476
