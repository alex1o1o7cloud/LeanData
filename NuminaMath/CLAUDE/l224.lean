import Mathlib

namespace NUMINAMATH_CALUDE_rachel_reading_homework_l224_22406

theorem rachel_reading_homework (math_homework : ℕ) (reading_homework : ℕ) 
  (h1 : math_homework = 7)
  (h2 : math_homework = reading_homework + 3) :
  reading_homework = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_reading_homework_l224_22406


namespace NUMINAMATH_CALUDE_max_distinct_substrings_l224_22462

/-- Represents the length of the string -/
def stringLength : ℕ := 66

/-- Represents the number of distinct letters in the string -/
def distinctLetters : ℕ := 4

/-- Calculates the sum of an arithmetic series -/
def arithmeticSum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Theorem: The maximum number of distinct substrings in a string of length 66
    composed of 4 distinct letters is 2100 -/
theorem max_distinct_substrings :
  distinctLetters +
  distinctLetters^2 +
  (arithmeticSum (stringLength - 2) - arithmeticSum (distinctLetters - 1)) = 2100 := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_substrings_l224_22462


namespace NUMINAMATH_CALUDE_present_age_of_b_l224_22423

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →
  (a = b + 7) →
  b = 37 := by
sorry

end NUMINAMATH_CALUDE_present_age_of_b_l224_22423


namespace NUMINAMATH_CALUDE_number_not_perfect_square_l224_22494

theorem number_not_perfect_square (n : ℕ) 
  (h : ∃ k, n = 6 * (10^600 - 1) / 9 + k * 10^600 ∧ k ≥ 0) : 
  ¬∃ m : ℕ, n = m^2 := by
sorry

end NUMINAMATH_CALUDE_number_not_perfect_square_l224_22494


namespace NUMINAMATH_CALUDE_star_seven_three_l224_22457

def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

theorem star_seven_three : star 7 3 = 16 := by sorry

end NUMINAMATH_CALUDE_star_seven_three_l224_22457


namespace NUMINAMATH_CALUDE_trey_kyle_turtle_difference_l224_22454

/-- Proves that Trey has 60 more turtles than Kyle given the conditions in the problem -/
theorem trey_kyle_turtle_difference : 
  ∀ (kristen trey kris layla tim kyle : ℚ),
  kristen = 24.5 →
  kris = kristen / 3 →
  trey = 8.5 * kris →
  layla = 2 * trey →
  tim = 2 / 3 * kristen →
  kyle = tim / 2 →
  trey - kyle = 60 := by
  sorry

end NUMINAMATH_CALUDE_trey_kyle_turtle_difference_l224_22454


namespace NUMINAMATH_CALUDE_min_value_abc_l224_22469

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 + 4*a*b + 9*b^2 + 3*b*c + c^2 ≥ 18 ∧
  (a^2 + 4*a*b + 9*b^2 + 3*b*c + c^2 = 18 ↔ a = 3 ∧ b = 1/3 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l224_22469


namespace NUMINAMATH_CALUDE_calculate_expression_l224_22450

theorem calculate_expression : |(-8 : ℝ)| + (-2011 : ℝ)^0 - 2 * Real.cos (π / 3) + (1 / 2)⁻¹ = 10 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l224_22450


namespace NUMINAMATH_CALUDE_jacoby_lottery_winnings_l224_22417

theorem jacoby_lottery_winnings :
  let trip_cost : ℕ := 5000
  let hourly_wage : ℕ := 20
  let hours_worked : ℕ := 10
  let cookie_price : ℕ := 4
  let cookies_sold : ℕ := 24
  let lottery_ticket_cost : ℕ := 10
  let remaining_needed : ℕ := 3214
  let sister_gift : ℕ := 500
  let num_sisters : ℕ := 2

  let job_earnings := hourly_wage * hours_worked
  let cookie_earnings := cookie_price * cookies_sold
  let total_earnings := job_earnings + cookie_earnings - lottery_ticket_cost
  let total_gifts := sister_gift * num_sisters
  let current_funds := total_earnings + total_gifts
  let lottery_winnings := trip_cost - remaining_needed - current_funds

  lottery_winnings = 500 := by
    sorry

end NUMINAMATH_CALUDE_jacoby_lottery_winnings_l224_22417


namespace NUMINAMATH_CALUDE_binomial_product_integer_l224_22474

theorem binomial_product_integer (m n : ℕ) : 
  ∃ k : ℕ, (Nat.factorial (2 * m) * Nat.factorial (2 * n)) = 
    k * (Nat.factorial m * Nat.factorial n * Nat.factorial (m + n)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_integer_l224_22474


namespace NUMINAMATH_CALUDE_certain_fraction_is_two_fifths_l224_22413

theorem certain_fraction_is_two_fifths :
  ∀ (x y : ℚ),
    (x ≠ 0 ∧ y ≠ 0) →
    ((1 : ℚ) / 7) / (x / y) = ((3 : ℚ) / 7) / ((6 : ℚ) / 5) →
    x / y = (2 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_fraction_is_two_fifths_l224_22413


namespace NUMINAMATH_CALUDE_max_value_of_expression_l224_22434

theorem max_value_of_expression (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ 
    (a + b + c)^2 / (a^2 + b^2 + c^2) = 3) ∧ 
  (∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → p + q + r = 1 → 
    (p + q + r)^2 / (p^2 + q^2 + r^2) ≤ 3) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l224_22434


namespace NUMINAMATH_CALUDE_tank_capacity_l224_22499

theorem tank_capacity (T : ℚ) : 
  (3/4 : ℚ) * T + 4 = (7/8 : ℚ) * T → T = 32 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l224_22499


namespace NUMINAMATH_CALUDE_brenda_lead_after_turn3_l224_22453

/-- Represents the score of a player in a Scrabble game -/
structure ScrabbleScore where
  turn1 : ℕ
  turn2 : ℕ
  turn3 : ℕ

/-- Calculates the total score for a player -/
def totalScore (score : ScrabbleScore) : ℕ :=
  score.turn1 + score.turn2 + score.turn3

/-- Represents the Scrabble game between Brenda and David -/
structure ScrabbleGame where
  brenda : ScrabbleScore
  david : ScrabbleScore
  brenda_lead_before_turn3 : ℕ

/-- The Scrabble game instance based on the given problem -/
def game : ScrabbleGame :=
  { brenda := { turn1 := 18, turn2 := 25, turn3 := 15 }
  , david := { turn1 := 10, turn2 := 35, turn3 := 32 }
  , brenda_lead_before_turn3 := 22 }

/-- Theorem stating that Brenda is ahead by 5 points after the third turn -/
theorem brenda_lead_after_turn3 (g : ScrabbleGame) : 
  totalScore g.brenda - totalScore g.david = 5 :=
sorry

end NUMINAMATH_CALUDE_brenda_lead_after_turn3_l224_22453


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l224_22493

theorem sum_of_two_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 130) : x + y = -7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l224_22493


namespace NUMINAMATH_CALUDE_square_even_implies_even_l224_22473

theorem square_even_implies_even (a : ℤ) (h : Even (a^2)) : Even a := by
  sorry

end NUMINAMATH_CALUDE_square_even_implies_even_l224_22473


namespace NUMINAMATH_CALUDE_probability_standard_deck_l224_22431

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (diamond_cards : Nat)
  (spade_cards : Nat)

/-- A standard deck has 52 cards, 13 diamonds, and 13 spades -/
def standard_deck : Deck :=
  ⟨52, 13, 13⟩

/-- Calculates the probability of drawing a diamond first, then two spades -/
def probability_diamond_then_two_spades (d : Deck) : Rat :=
  (d.diamond_cards : Rat) / d.total_cards *
  (d.spade_cards : Rat) / (d.total_cards - 1) *
  ((d.spade_cards - 1) : Rat) / (d.total_cards - 2)

/-- Theorem: The probability of drawing a diamond first, then two spades from a standard deck is 13/850 -/
theorem probability_standard_deck :
  probability_diamond_then_two_spades standard_deck = 13 / 850 := by
  sorry

end NUMINAMATH_CALUDE_probability_standard_deck_l224_22431


namespace NUMINAMATH_CALUDE_ellipse_equation_l224_22464

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (a b c : ℝ) (h1 : a = 2) (h2 : c = Real.sqrt 3) (h3 : b^2 = a^2 - c^2) :
  ∀ (x y : ℝ), x^2 / 4 + y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l224_22464


namespace NUMINAMATH_CALUDE_log_simplification_l224_22435

theorem log_simplification (a b c d x y : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hx : 0 < x) (hy : 0 < y) : 
  Real.log (a / b) + Real.log (b / c) + Real.log (c / d) - Real.log ((a * y) / (d * x)) = Real.log (x / y) := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l224_22435


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l224_22466

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x + c = 0 ↔ x = (-3 + Real.sqrt 7) / 2 ∨ x = (-3 - Real.sqrt 7) / 2) → 
  c = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l224_22466


namespace NUMINAMATH_CALUDE_sara_lunch_bill_l224_22447

/-- The cost of Sara's lunch given the prices of a hotdog and a salad -/
def lunch_bill (hotdog_price salad_price : ℚ) : ℚ :=
  hotdog_price + salad_price

/-- Theorem stating that Sara's lunch bill is the sum of the hotdog and salad prices -/
theorem sara_lunch_bill :
  lunch_bill 5.36 5.10 = 10.46 :=
by sorry

end NUMINAMATH_CALUDE_sara_lunch_bill_l224_22447


namespace NUMINAMATH_CALUDE_water_remaining_l224_22422

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 11/4 → remaining = initial - used → remaining = 1/4 := by
sorry

end NUMINAMATH_CALUDE_water_remaining_l224_22422


namespace NUMINAMATH_CALUDE_square_root_7396_squared_l224_22419

theorem square_root_7396_squared : (Real.sqrt 7396)^2 = 7396 := by sorry

end NUMINAMATH_CALUDE_square_root_7396_squared_l224_22419


namespace NUMINAMATH_CALUDE_least_valid_number_l224_22414

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d : ℕ) (p : ℕ), 
    d ≥ 1 ∧ d ≤ 9 ∧
    n = 10^p * d + (n % 10^p) ∧
    10^p * d + (n % 10^p) = 17 * (n % 10^p)

theorem least_valid_number : 
  is_valid_number 10625 ∧ 
  ∀ (m : ℕ), m < 10625 → ¬(is_valid_number m) :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l224_22414


namespace NUMINAMATH_CALUDE_tommy_wheel_count_l224_22471

/-- The number of wheels on each truck -/
def truck_wheels : ℕ := 4

/-- The number of wheels on each car -/
def car_wheels : ℕ := 4

/-- The number of trucks Tommy saw -/
def trucks_seen : ℕ := 12

/-- The number of cars Tommy saw -/
def cars_seen : ℕ := 13

/-- The total number of wheels Tommy saw -/
def total_wheels : ℕ := (trucks_seen * truck_wheels) + (cars_seen * car_wheels)

theorem tommy_wheel_count : total_wheels = 100 := by
  sorry

end NUMINAMATH_CALUDE_tommy_wheel_count_l224_22471


namespace NUMINAMATH_CALUDE_choose_three_from_eleven_l224_22402

theorem choose_three_from_eleven (n : ℕ) (k : ℕ) : n = 11 → k = 3 → Nat.choose n k = 165 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_eleven_l224_22402


namespace NUMINAMATH_CALUDE_simplification_condition_l224_22411

theorem simplification_condition (x y k : ℝ) : 
  y = k * x →
  ((x - y) * (2 * x - y) - 3 * x * (2 * x - y) = 5 * x^2) ↔ (k = 3 ∨ k = -3) :=
by sorry

end NUMINAMATH_CALUDE_simplification_condition_l224_22411


namespace NUMINAMATH_CALUDE_range_of_function_l224_22496

theorem range_of_function :
  ∀ (x : ℝ), -2/3 ≤ (Real.sin x - 1) / (2 - Real.sin x) ∧ 
             (Real.sin x - 1) / (2 - Real.sin x) ≤ 0 ∧
  (∃ (y : ℝ), (Real.sin y - 1) / (2 - Real.sin y) = -2/3) ∧
  (∃ (z : ℝ), (Real.sin z - 1) / (2 - Real.sin z) = 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_function_l224_22496


namespace NUMINAMATH_CALUDE_solution_value_l224_22495

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 6*x + 11 = 24

-- Define a and b as solutions to the equation
def a_b_solutions (a b : ℝ) : Prop :=
  quadratic_equation a ∧ quadratic_equation b ∧ a ≥ b

-- Theorem statement
theorem solution_value (a b : ℝ) (h : a_b_solutions a b) :
  3*a - b = 6 + 4*Real.sqrt 22 :=
by sorry

end NUMINAMATH_CALUDE_solution_value_l224_22495


namespace NUMINAMATH_CALUDE_max_red_surface_area_76_l224_22420

/-- Represents a small cube with two red faces -/
inductive SmallCube
| Adjacent : SmallCube  -- Two adjacent faces are red
| Opposite : SmallCube  -- Two opposite faces are red

/-- Configuration of small cubes -/
structure CubeConfiguration where
  total : Nat
  adjacent : Nat
  opposite : Nat

/-- Represents the large cube assembled from small cubes -/
structure LargeCube where
  config : CubeConfiguration
  side_length : Nat

/-- Calculates the maximum red surface area of the large cube -/
def max_red_surface_area (lc : LargeCube) : Nat :=
  sorry

/-- Theorem stating the maximum red surface area for the given configuration -/
theorem max_red_surface_area_76 :
  ∀ (lc : LargeCube),
    lc.config.total = 64 ∧
    lc.config.adjacent = 20 ∧
    lc.config.opposite = 44 ∧
    lc.side_length = 4 →
    max_red_surface_area lc = 76 :=
  sorry

end NUMINAMATH_CALUDE_max_red_surface_area_76_l224_22420


namespace NUMINAMATH_CALUDE_average_problem_l224_22478

theorem average_problem (n₁ n₂ : ℕ) (avg_all avg₂ : ℚ) (h₁ : n₁ = 30) (h₂ : n₂ = 20) 
  (h₃ : avg₂ = 30) (h₄ : avg_all = 24) :
  let sum_all := (n₁ + n₂ : ℚ) * avg_all
  let sum₂ := n₂ * avg₂
  let sum₁ := sum_all - sum₂
  sum₁ / n₁ = 20 := by sorry

end NUMINAMATH_CALUDE_average_problem_l224_22478


namespace NUMINAMATH_CALUDE_power_of_128_l224_22421

theorem power_of_128 : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by sorry

end NUMINAMATH_CALUDE_power_of_128_l224_22421


namespace NUMINAMATH_CALUDE_pinterest_group_average_pins_l224_22441

/-- The average number of pins contributed per day by each member in a Pinterest group. -/
def average_pins_per_day (
  group_size : ℕ
  ) (
  initial_pins : ℕ
  ) (
  final_pins : ℕ
  ) (
  days : ℕ
  ) (
  deleted_pins_per_week_per_person : ℕ
  ) : ℚ :=
  let total_deleted_pins := (group_size * deleted_pins_per_week_per_person * (days / 7) : ℚ)
  let total_new_pins := (final_pins - initial_pins : ℚ) + total_deleted_pins
  total_new_pins / (group_size * days : ℚ)

/-- Theorem stating that the average number of pins contributed per day is 10. -/
theorem pinterest_group_average_pins :
  average_pins_per_day 20 1000 6600 30 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pinterest_group_average_pins_l224_22441


namespace NUMINAMATH_CALUDE_identical_angular_acceleration_l224_22458

/-- Two wheels with identical masses and different radii have identical angular accelerations -/
theorem identical_angular_acceleration (m : ℝ) (R₁ R₂ F₁ F₂ : ℝ) 
  (h_m : m = 1)
  (h_R₁ : R₁ = 0.5)
  (h_R₂ : R₂ = 1)
  (h_F₁ : F₁ = 1)
  (h_positive : m > 0 ∧ R₁ > 0 ∧ R₂ > 0 ∧ F₁ > 0 ∧ F₂ > 0) :
  (F₁ * R₁ / (m * R₁^2) = F₂ * R₂ / (m * R₂^2)) → F₂ = 2 := by
  sorry

#check identical_angular_acceleration

end NUMINAMATH_CALUDE_identical_angular_acceleration_l224_22458


namespace NUMINAMATH_CALUDE_max_value_xyz_l224_22481

theorem max_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : 
  x^3 * y^3 * z^2 ≤ 4782969/390625 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xyz_l224_22481


namespace NUMINAMATH_CALUDE_lg_sqrt5_plus_half_lg20_equals_1_l224_22439

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_sqrt5_plus_half_lg20_equals_1 : lg (Real.sqrt 5) + (1/2) * lg 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_sqrt5_plus_half_lg20_equals_1_l224_22439


namespace NUMINAMATH_CALUDE_no_real_roots_of_composition_l224_22488

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_real_roots_of_composition 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, quadratic a b c x ≠ x) :
  ∀ x : ℝ, quadratic a b c (quadratic a b c x) ≠ x := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_of_composition_l224_22488


namespace NUMINAMATH_CALUDE_m_eq_2_necessary_not_sufficient_l224_22426

def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

theorem m_eq_2_necessary_not_sufficient :
  (∀ m : ℝ, A m ∩ B = {4} → m = 2 ∨ m = -2) ∧
  (∃ m : ℝ, m = 2 ∧ A m ∩ B = {4}) ∧
  (∃ m : ℝ, m = -2 ∧ A m ∩ B = {4}) :=
sorry

end NUMINAMATH_CALUDE_m_eq_2_necessary_not_sufficient_l224_22426


namespace NUMINAMATH_CALUDE_remainder_after_adding_2023_l224_22444

theorem remainder_after_adding_2023 (n : ℤ) (h : n % 7 = 2) : (n + 2023) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2023_l224_22444


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l224_22485

/-- An arithmetic sequence with first term 13 and fourth term 1 has common difference -4. -/
theorem arithmetic_sequence_common_difference :
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence property
  a 1 = 13 →
  a 4 = 1 →
  a 2 - a 1 = -4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l224_22485


namespace NUMINAMATH_CALUDE_symmetric_latin_square_diagonal_property_l224_22463

/-- A square matrix with odd size, filled with numbers 1 to n, where each row and column contains all numbers exactly once, and which is symmetric about the main diagonal. -/
structure SymmetricLatinSquare (n : ℕ) :=
  (matrix : Fin n → Fin n → Fin n)
  (odd : Odd n)
  (latin_square : ∀ (i j : Fin n), ∃! (k : Fin n), matrix i k = j ∧ ∃! (k : Fin n), matrix k j = i)
  (symmetric : ∀ (i j : Fin n), matrix i j = matrix j i)

/-- The main diagonal of a square matrix contains all numbers from 1 to n exactly once. -/
def diagonal_contains_all (n : ℕ) (matrix : Fin n → Fin n → Fin n) : Prop :=
  ∀ (k : Fin n), ∃! (i : Fin n), matrix i i = k

/-- If a SymmetricLatinSquare exists, then its main diagonal contains all numbers from 1 to n exactly once. -/
theorem symmetric_latin_square_diagonal_property {n : ℕ} (sls : SymmetricLatinSquare n) :
  diagonal_contains_all n sls.matrix :=
sorry

end NUMINAMATH_CALUDE_symmetric_latin_square_diagonal_property_l224_22463


namespace NUMINAMATH_CALUDE_absolute_value_equation_l224_22425

theorem absolute_value_equation (x : ℝ) : 
  |(-2 : ℝ)| * (|(-25 : ℝ)| - |x|) = 40 ↔ |x| = 5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l224_22425


namespace NUMINAMATH_CALUDE_greatest_x_under_conditions_l224_22448

theorem greatest_x_under_conditions (x : ℕ) : 
  x % 4 = 0 → 
  x > 0 → 
  x^3 < 8000 → 
  x ≤ 16 ∧ 
  ∀ y : ℕ, y % 4 = 0 → y > 0 → y^3 < 8000 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_under_conditions_l224_22448


namespace NUMINAMATH_CALUDE_equal_expressions_l224_22490

theorem equal_expressions : 
  (2^3 ≠ 2 * 3) ∧ 
  (-(-2)^2 ≠ (-2)^2) ∧ 
  (-3^2 ≠ 3^2) ∧ 
  (-2^3 = (-2)^3) := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_l224_22490


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_25_20_l224_22472

theorem half_abs_diff_squares_25_20 : (1/2 : ℝ) * |25^2 - 20^2| = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_25_20_l224_22472


namespace NUMINAMATH_CALUDE_exponential_above_line_l224_22415

theorem exponential_above_line (k : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.exp x > k * x + 1) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_above_line_l224_22415


namespace NUMINAMATH_CALUDE_taxi_ride_cost_l224_22407

def taxi_cost (initial_charge : ℚ) (additional_charge : ℚ) (passenger_fee : ℚ) (luggage_fee : ℚ)
              (distance : ℚ) (passengers : ℕ) (luggage : ℕ) : ℚ :=
  let distance_quarters := (distance - 1/4).ceil * 4
  let distance_charge := initial_charge + additional_charge * (distance_quarters - 1)
  let passenger_charge := passenger_fee * (passengers - 1)
  let luggage_charge := luggage_fee * luggage
  distance_charge + passenger_charge + luggage_charge

theorem taxi_ride_cost :
  taxi_cost 5 0.6 1 2 12.4 3 2 = 39.8 := by
  sorry

end NUMINAMATH_CALUDE_taxi_ride_cost_l224_22407


namespace NUMINAMATH_CALUDE_count_integers_in_range_l224_22486

theorem count_integers_in_range : 
  (Finset.range (513 - 2)).card = 511 := by sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l224_22486


namespace NUMINAMATH_CALUDE_balance_theorem_l224_22405

/-- Represents the balance of symbols -/
structure Balance :=
  (star : ℚ)
  (square : ℚ)
  (heart : ℚ)
  (club : ℚ)

/-- The balance equations from the problem -/
def balance_equations (b : Balance) : Prop :=
  3 * b.star + 4 * b.square + b.heart = 12 * b.club ∧
  b.star = b.heart + 2 * b.club

/-- The theorem to prove -/
theorem balance_theorem (b : Balance) :
  balance_equations b →
  3 * b.square + 2 * b.heart = (26 / 9) * b.square :=
by sorry

end NUMINAMATH_CALUDE_balance_theorem_l224_22405


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l224_22429

theorem book_arrangement_count : Nat :=
  let num_dictionaries : Nat := 3
  let num_novels : Nat := 2
  let dict_arrangements : Nat := Nat.factorial num_dictionaries
  let novel_arrangements : Nat := Nat.factorial num_novels
  let group_arrangements : Nat := Nat.factorial 2
  dict_arrangements * novel_arrangements * group_arrangements

theorem book_arrangement_proof :
  book_arrangement_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l224_22429


namespace NUMINAMATH_CALUDE_expected_heads_value_l224_22479

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The probability of a coin showing heads after a single flip -/
def prob_heads : ℚ := 1 / 2

/-- The maximum number of flips for each coin -/
def max_flips : ℕ := 4

/-- The probability of a coin showing heads after at most four flips -/
def prob_heads_after_four_flips : ℚ :=
  1 - (1 - prob_heads) ^ max_flips

/-- The expected number of coins showing heads after the series of flips -/
def expected_heads : ℚ := num_coins * prob_heads_after_four_flips

theorem expected_heads_value :
  expected_heads = 93.75 := by sorry

end NUMINAMATH_CALUDE_expected_heads_value_l224_22479


namespace NUMINAMATH_CALUDE_max_students_is_nine_l224_22487

/-- Represents the answer choices for each question -/
inductive Choice
| A
| B
| C

/-- Represents a student's answers to all questions -/
def StudentAnswers := Fin 4 → Choice

/-- The property that for any 3 students, there is at least one question where their answers differ -/
def DifferentAnswersExist (answers : Finset StudentAnswers) : Prop :=
  ∀ s1 s2 s3 : StudentAnswers, s1 ∈ answers → s2 ∈ answers → s3 ∈ answers →
    s1 ≠ s2 → s2 ≠ s3 → s1 ≠ s3 →
    ∃ q : Fin 4, s1 q ≠ s2 q ∧ s2 q ≠ s3 q ∧ s1 q ≠ s3 q

/-- The main theorem stating that the maximum number of students is 9 -/
theorem max_students_is_nine :
  ∃ (answers : Finset StudentAnswers),
    DifferentAnswersExist answers ∧
    answers.card = 9 ∧
    ∀ (larger_set : Finset StudentAnswers),
      larger_set.card > 9 →
      ¬DifferentAnswersExist larger_set :=
sorry

end NUMINAMATH_CALUDE_max_students_is_nine_l224_22487


namespace NUMINAMATH_CALUDE_min_value_expression_l224_22409

theorem min_value_expression (x y : ℝ) : (x^2*y + x*y^2 - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l224_22409


namespace NUMINAMATH_CALUDE_validArrangementCount_l224_22408

/-- Represents a seating arrangement around a rectangular table. -/
structure SeatingArrangement where
  chairs : Fin 15 → Person
  satisfiesConditions : Bool

/-- Represents a person to be seated. -/
inductive Person
  | Man : Person
  | Woman : Person
  | AdditionalPerson : Person

/-- Checks if two positions are adjacent or opposite on the table. -/
def areAdjacentOrOpposite (pos1 pos2 : Fin 15) : Bool := sorry

/-- Checks if the seating arrangement satisfies all conditions. -/
def satisfiesAllConditions (arrangement : SeatingArrangement) : Bool := sorry

/-- Counts the number of valid seating arrangements. -/
def countValidArrangements : Nat := sorry

/-- Theorem stating the number of valid seating arrangements. -/
theorem validArrangementCount : countValidArrangements = 3265920 := by sorry

end NUMINAMATH_CALUDE_validArrangementCount_l224_22408


namespace NUMINAMATH_CALUDE_f_one_upper_bound_l224_22412

/-- A quadratic function f(x) = 2x^2 - mx + 5 where m is a real number -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - m * x + 5

/-- The theorem stating that if f(x) is monotonically decreasing on (-∞, -2],
    then f(1) ≤ 15 -/
theorem f_one_upper_bound (m : ℝ) 
  (h : ∀ x y, x ≤ y → y ≤ -2 → f m x ≥ f m y) : 
  f m 1 ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_f_one_upper_bound_l224_22412


namespace NUMINAMATH_CALUDE_largest_nested_root_l224_22438

theorem largest_nested_root : 
  let a := (7 : ℝ)^(1/4) * 8^(1/12)
  let b := 8^(1/2) * 7^(1/8)
  let c := 7^(1/2) * 8^(1/8)
  let d := 7^(1/3) * 8^(1/6)
  let e := 8^(1/3) * 7^(1/6)
  b > a ∧ b > c ∧ b > d ∧ b > e :=
by sorry

end NUMINAMATH_CALUDE_largest_nested_root_l224_22438


namespace NUMINAMATH_CALUDE_rogers_initial_money_l224_22443

theorem rogers_initial_money (game_cost toy_cost num_toys : ℕ) 
  (h1 : game_cost = 48)
  (h2 : toy_cost = 3)
  (h3 : num_toys = 5)
  (h4 : ∃ (remaining : ℕ), remaining = num_toys * toy_cost) :
  game_cost + num_toys * toy_cost = 63 := by
sorry

end NUMINAMATH_CALUDE_rogers_initial_money_l224_22443


namespace NUMINAMATH_CALUDE_polynomial_Q_value_l224_22461

-- Define the polynomial
def polynomial (P Q R : ℤ) (z : ℝ) : ℝ :=
  z^5 - 15*z^4 + P*z^3 + Q*z^2 + R*z + 64

-- Define the roots
def roots : List ℤ := [8, 4, 1, 1, 1]

-- Theorem statement
theorem polynomial_Q_value (P Q R : ℤ) :
  (∀ r ∈ roots, polynomial P Q R r = 0) →
  (List.sum roots = 15) →
  (List.prod roots = 64) →
  (∀ r ∈ roots, r > 0) →
  Q = -45 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_Q_value_l224_22461


namespace NUMINAMATH_CALUDE_max_leftover_cookies_l224_22465

theorem max_leftover_cookies (n : ℕ) (h : n > 0) : 
  ∃ (total : ℕ), total % n = n - 1 ∧ ∀ (m : ℕ), m % n ≤ n - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_cookies_l224_22465


namespace NUMINAMATH_CALUDE_darwin_remaining_money_l224_22475

/-- Calculates the remaining money after Darwin's expenditures --/
def remaining_money (initial : ℝ) : ℝ :=
  let after_gas := initial * (1 - 0.35)
  let after_food := after_gas * (1 - 0.2)
  let after_clothing := after_food * (1 - 0.25)
  after_clothing * (1 - 0.15)

/-- Theorem stating that Darwin's remaining money is $4,972.50 --/
theorem darwin_remaining_money :
  remaining_money 15000 = 4972.50 := by
  sorry

end NUMINAMATH_CALUDE_darwin_remaining_money_l224_22475


namespace NUMINAMATH_CALUDE_pudong_exemplifies_ideal_pattern_l224_22482

-- Define the characteristics of city cluster development
structure CityClusterDevelopment where
  aggregation : Bool
  radiation : Bool
  mutualInfluence : Bool

-- Define the development pattern of Pudong, Shanghai
def pudongDevelopment : CityClusterDevelopment :=
  { aggregation := true,
    radiation := true,
    mutualInfluence := true }

-- Define the ideal world city cluster development pattern
def idealCityClusterPattern : CityClusterDevelopment :=
  { aggregation := true,
    radiation := true,
    mutualInfluence := true }

-- Theorem statement
theorem pudong_exemplifies_ideal_pattern :
  pudongDevelopment = idealCityClusterPattern :=
by sorry

end NUMINAMATH_CALUDE_pudong_exemplifies_ideal_pattern_l224_22482


namespace NUMINAMATH_CALUDE_parabola_c_value_l224_22480

/-- A parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord (-1) = 3 →  -- vertex condition
  p.x_coord (-2) = 1 →  -- point condition
  p.c = 1 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l224_22480


namespace NUMINAMATH_CALUDE_article_cost_price_l224_22489

/-- The cost price of an article that satisfies the given conditions -/
def cost_price : ℝ := 1600

/-- The selling price of the article with a 5% gain -/
def selling_price (c : ℝ) : ℝ := 1.05 * c

/-- The new cost price if bought at 5% less -/
def new_cost_price (c : ℝ) : ℝ := 0.95 * c

/-- The new selling price if sold for 8 less -/
def new_selling_price (c : ℝ) : ℝ := selling_price c - 8

theorem article_cost_price :
  selling_price cost_price = 1.05 * cost_price ∧
  new_cost_price cost_price = 0.95 * cost_price ∧
  new_selling_price cost_price = selling_price cost_price - 8 ∧
  new_selling_price cost_price = 1.1 * new_cost_price cost_price :=
by sorry

end NUMINAMATH_CALUDE_article_cost_price_l224_22489


namespace NUMINAMATH_CALUDE_c_leq_one_sufficient_not_necessary_l224_22477

def is_increasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n < a (n + 1)

def sequence_a (c : ℝ) (n : ℕ+) : ℝ :=
  |n.val - c|

theorem c_leq_one_sufficient_not_necessary (c : ℝ) :
  (c ≤ 1 → is_increasing (sequence_a c)) ∧
  ¬(is_increasing (sequence_a c) → c ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_c_leq_one_sufficient_not_necessary_l224_22477


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l224_22436

theorem solution_satisfies_system : ∃ (x y : ℝ), 
  (y = 2 - x ∧ 3 * x = 1 + 2 * y) ∧ (x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l224_22436


namespace NUMINAMATH_CALUDE_odd_numbers_between_300_and_700_l224_22430

def count_odd_numbers (lower upper : ℕ) : ℕ :=
  (upper - lower - 1 + (lower % 2)) / 2

theorem odd_numbers_between_300_and_700 :
  count_odd_numbers 300 700 = 200 := by
  sorry

end NUMINAMATH_CALUDE_odd_numbers_between_300_and_700_l224_22430


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l224_22470

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length : ℝ) (cost_per_meter : ℝ) : ℝ :=
  let breadth := length - 10
  let perimeter := 2 * (length + breadth)
  cost_per_meter * perimeter

/-- Theorem: The total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_theorem :
  total_fencing_cost 55 26.50 = 5300 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_theorem_l224_22470


namespace NUMINAMATH_CALUDE_smallest_r_in_special_progression_l224_22403

theorem smallest_r_in_special_progression (p q r : ℤ) : 
  p < q → q < r → 
  q^2 = p * r →  -- Geometric progression condition
  2 * q = p + r →  -- Arithmetic progression condition
  ∀ (p' q' r' : ℤ), p' < q' → q' < r' → q'^2 = p' * r' → 2 * q' = p' + r' → r ≤ r' →
  r = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_r_in_special_progression_l224_22403


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l224_22452

theorem integer_pairs_satisfying_equation :
  ∀ x y : ℤ, y ≥ 0 → (x^2 + 2*x*y + Nat.factorial y.toNat = 131) ↔ ((x = 1 ∧ y = 5) ∨ (x = -11 ∧ y = 5)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l224_22452


namespace NUMINAMATH_CALUDE_line_equation_through_points_l224_22446

/-- Prove that the equation x + 2y - 2 = 0 represents the line passing through points A(0,1) and B(2,0). -/
theorem line_equation_through_points :
  ∀ (x y : ℝ), x + 2*y - 2 = 0 ↔ ∃ (t : ℝ), (x, y) = (1 - t, t) * 2 + (0, 1) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l224_22446


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l224_22418

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 30 ∧ x - y = 10 → x * y = 200 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l224_22418


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l224_22410

/-- Represents the dimensions of a cistern with an elevated platform --/
structure CisternDimensions where
  length : Real
  width : Real
  waterDepth : Real
  platformLength : Real
  platformWidth : Real
  platformHeight : Real

/-- Calculates the total wet surface area of the cistern --/
def totalWetSurfaceArea (d : CisternDimensions) : Real :=
  let wallArea := 2 * (d.length * d.waterDepth) + 2 * (d.width * d.waterDepth)
  let bottomArea := d.length * d.width
  let submergedHeight := d.waterDepth - d.platformHeight
  let platformSideArea := 2 * (d.platformLength * submergedHeight) + 2 * (d.platformWidth * submergedHeight)
  wallArea + bottomArea + platformSideArea

/-- Theorem stating that the total wet surface area of the given cistern is 63.5 square meters --/
theorem cistern_wet_surface_area :
  let d : CisternDimensions := {
    length := 8,
    width := 4,
    waterDepth := 1.25,
    platformLength := 1,
    platformWidth := 0.5,
    platformHeight := 0.75
  }
  totalWetSurfaceArea d = 63.5 := by
  sorry


end NUMINAMATH_CALUDE_cistern_wet_surface_area_l224_22410


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l224_22427

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2017^x + Real.log x / Real.log 2017
  else if x < 0 then -(2017^(-x) + Real.log (-x) / Real.log 2017)
  else 0

theorem f_has_three_zeros :
  (∃! a b c : ℝ, a < 0 ∧ b = 0 ∧ c > 0 ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) ∧
  (∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_f_has_three_zeros_l224_22427


namespace NUMINAMATH_CALUDE_sushil_marks_proof_l224_22437

def total_marks (english science maths : ℕ) : ℕ := english + science + maths

theorem sushil_marks_proof (english science maths : ℕ) :
  english = 3 * science →
  english = maths / 4 →
  science = 17 →
  total_marks english science maths = 272 :=
by
  sorry

end NUMINAMATH_CALUDE_sushil_marks_proof_l224_22437


namespace NUMINAMATH_CALUDE_exp_2pi_3i_in_second_quadrant_l224_22459

-- Define Euler's formula
axiom euler_formula (x : ℝ) : Complex.exp (x * Complex.I) = Complex.mk (Real.cos x) (Real.sin x)

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem exp_2pi_3i_in_second_quadrant :
  second_quadrant (Complex.exp ((2 * Real.pi / 3) * Complex.I)) :=
sorry

end NUMINAMATH_CALUDE_exp_2pi_3i_in_second_quadrant_l224_22459


namespace NUMINAMATH_CALUDE_garden_fencing_l224_22449

theorem garden_fencing (garden_area : ℝ) (extension : ℝ) : 
  garden_area = 784 →
  extension = 10 →
  (4 * (Real.sqrt garden_area + extension)) = 152 := by
  sorry

end NUMINAMATH_CALUDE_garden_fencing_l224_22449


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l224_22451

/-- Given a quadratic equation 3x^2 - 4 = -2x, prove that when rearranged 
    into the standard form ax^2 + bx + c = 0, the coefficients are a = 3, b = 2, and c = -4 -/
theorem quadratic_coefficients : 
  ∀ (x : ℝ), 3 * x^2 - 4 = -2 * x → 
  ∃ (a b c : ℝ), a * x^2 + b * x + c = 0 ∧ a = 3 ∧ b = 2 ∧ c = -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l224_22451


namespace NUMINAMATH_CALUDE_lower_half_plane_inequality_l224_22445

/-- Given a line l passing through points A(2,1) and B(-1,3), 
    the inequality 2x + 3y - 7 ≤ 0 represents the lower half-plane including line l. -/
theorem lower_half_plane_inequality (x y : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (2 - 3*t, 1 + 2*t)}
  (x, y) ∈ l ∨ (∃ p ∈ l, y < p.2) ↔ 2*x + 3*y - 7 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_lower_half_plane_inequality_l224_22445


namespace NUMINAMATH_CALUDE_ways_to_soccer_field_l224_22467

theorem ways_to_soccer_field (walk_ways drive_ways : ℕ) : 
  walk_ways = 3 → drive_ways = 4 → walk_ways + drive_ways = 7 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_soccer_field_l224_22467


namespace NUMINAMATH_CALUDE_inequality_implies_bounds_l224_22428

/-- Custom operation ⊗ defined on ℝ -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the relationship between the inequality and the bounds on a -/
theorem inequality_implies_bounds (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_bounds_l224_22428


namespace NUMINAMATH_CALUDE_square_root_equality_l224_22492

theorem square_root_equality (a b : ℝ) : 
  Real.sqrt (6 + a / b) = 6 * Real.sqrt (a / b) → a = 6 ∧ b = 35 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l224_22492


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_third_roll_l224_22424

-- Define a die roll as a number between 1 and 6
def DieRoll : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of two die rolls
def SumTwoDice (roll1 roll2 : DieRoll) : ℕ := roll1.val + roll2.val

-- Define the probability space
def TotalOutcomes : ℕ := 6 * 6 * 6

-- Define the favorable outcomes
def FavorableOutcomes : ℕ := 51

-- The main theorem
theorem probability_sum_greater_than_third_roll :
  (FavorableOutcomes : ℚ) / TotalOutcomes = 17 / 72 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_third_roll_l224_22424


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l224_22440

theorem smallest_number_with_given_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 3 = 1 ∧ 
  x % 4 = 2 ∧ 
  x % 7 = 3 ∧ 
  ∀ y : ℕ, y > 0 → y % 3 = 1 → y % 4 = 2 → y % 7 = 3 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l224_22440


namespace NUMINAMATH_CALUDE_perpendicular_lines_l224_22416

/-- Two lines are perpendicular if the sum of the products of their corresponding coefficients is zero -/
def perpendicular (a b c e f g : ℝ) : Prop := a * e + b * f = 0

/-- The line equation x + (m^2 - m)y = 4m - 1 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := x + (m^2 - m) * y = 4 * m - 1

/-- The line equation 2x - y - 5 = 0 -/
def line2 (x y : ℝ) : Prop := 2 * x - y - 5 = 0

theorem perpendicular_lines (m : ℝ) : 
  perpendicular 1 (m^2 - m) (1 - 4*m) 2 (-1) (-5) → m = -1 ∨ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l224_22416


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l224_22483

/-- Given real number m and vectors a and b in ℝ², prove that if a ⊥ b, then |a + b| = √34 -/
theorem vector_sum_magnitude (m : ℝ) (a b : ℝ × ℝ) : 
  a = (m + 2, 1) → 
  b = (1, -2*m) → 
  a.1 * b.1 + a.2 * b.2 = 0 → 
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 34 := by
  sorry


end NUMINAMATH_CALUDE_vector_sum_magnitude_l224_22483


namespace NUMINAMATH_CALUDE_no_valid_A_l224_22404

theorem no_valid_A : ¬∃ (A : ℕ), A < 10 ∧ 81 % A = 0 ∧ (456200 + A * 10 + 4) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_A_l224_22404


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l224_22401

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (sum_eq_10 : x + y = 10) (sum_eq_5prod : x + y = 5 * x * y) : 
  1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l224_22401


namespace NUMINAMATH_CALUDE_compute_expression_l224_22432

theorem compute_expression : 12 * (216 / 3 + 36 / 6 + 16 / 8 + 2) = 984 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l224_22432


namespace NUMINAMATH_CALUDE_min_value_product_l224_22498

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 38 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l224_22498


namespace NUMINAMATH_CALUDE_no_such_function_exists_l224_22400

theorem no_such_function_exists :
  ∀ (f : ℤ → Fin 3),
  ∃ (x y : ℤ), (|x - y| = 2 ∨ |x - y| = 3 ∨ |x - y| = 5) ∧ f x = f y :=
by sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l224_22400


namespace NUMINAMATH_CALUDE_total_money_divided_l224_22476

def money_division (maya annie saiji : ℕ) : Prop :=
  maya = annie / 2 ∧ annie = saiji / 2 ∧ saiji = 400

theorem total_money_divided : 
  ∀ maya annie saiji : ℕ, 
  money_division maya annie saiji → 
  maya + annie + saiji = 700 := by
sorry

end NUMINAMATH_CALUDE_total_money_divided_l224_22476


namespace NUMINAMATH_CALUDE_prime_factorization_sum_l224_22497

theorem prime_factorization_sum (w x y z k : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 11^k = 2310 → 2*w + 3*x + 5*y + 7*z + 11*k = 28 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_sum_l224_22497


namespace NUMINAMATH_CALUDE_mile_equals_400_rods_l224_22468

/-- Conversion rate from miles to furlongs -/
def mile_to_furlong : ℚ := 8

/-- Conversion rate from furlongs to rods -/
def furlong_to_rod : ℚ := 50

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_furlong * furlong_to_rod

theorem mile_equals_400_rods : rods_in_mile = 400 := by
  sorry

end NUMINAMATH_CALUDE_mile_equals_400_rods_l224_22468


namespace NUMINAMATH_CALUDE_hundred_squared_plus_201_is_composite_l224_22433

theorem hundred_squared_plus_201_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 100^2 + 201 = a * b := by
  sorry

end NUMINAMATH_CALUDE_hundred_squared_plus_201_is_composite_l224_22433


namespace NUMINAMATH_CALUDE_correct_arrangements_l224_22484

/-- The number of different arrangements of representatives for 7 subjects -/
def num_arrangements (num_boys num_girls num_subjects : ℕ) : ℕ :=
  num_boys * num_girls * (Nat.factorial (num_subjects - 2))

/-- Theorem stating the correct number of arrangements -/
theorem correct_arrangements :
  num_arrangements 4 3 7 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangements_l224_22484


namespace NUMINAMATH_CALUDE_complementary_events_l224_22455

-- Define the sample space for two shots
inductive ShotOutcome
| HH  -- Hit-Hit
| HM  -- Hit-Miss
| MH  -- Miss-Hit
| MM  -- Miss-Miss

-- Define the event of missing both times
def missBoth : Set ShotOutcome := {ShotOutcome.MM}

-- Define the event of hitting at least once
def hitAtLeastOnce : Set ShotOutcome := {ShotOutcome.HH, ShotOutcome.HM, ShotOutcome.MH}

-- Theorem stating that hitAtLeastOnce is the complement of missBoth
theorem complementary_events :
  hitAtLeastOnce = missBoth.compl :=
sorry

end NUMINAMATH_CALUDE_complementary_events_l224_22455


namespace NUMINAMATH_CALUDE_average_of_abc_l224_22442

theorem average_of_abc (A B C : ℚ) : 
  A = 2 → 
  2002 * C - 1001 * A = 8008 → 
  2002 * B + 3003 * A = 7007 → 
  (A + B + C) / 3 = 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_abc_l224_22442


namespace NUMINAMATH_CALUDE_three_cards_same_suit_count_l224_22456

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h1 : total_cards = num_suits * cards_per_suit)

/-- The number of ways to select three cards in order from the same suit -/
def ways_to_select_three_same_suit (d : Deck) : Nat :=
  d.num_suits * (d.cards_per_suit * (d.cards_per_suit - 1) * (d.cards_per_suit - 2))

/-- Theorem stating the number of ways to select three cards from the same suit -/
theorem three_cards_same_suit_count (d : Deck) 
  (h2 : d.total_cards = 52) 
  (h3 : d.num_suits = 4) 
  (h4 : d.cards_per_suit = 13) : 
  ways_to_select_three_same_suit d = 6864 := by
  sorry

#eval ways_to_select_three_same_suit ⟨52, 4, 13, rfl⟩

end NUMINAMATH_CALUDE_three_cards_same_suit_count_l224_22456


namespace NUMINAMATH_CALUDE_probability_of_a_l224_22491

theorem probability_of_a (a b : Set α) (p : Set α → ℝ) 
  (h1 : p b = 2/5)
  (h2 : p (a ∩ b) = p a * p b)
  (h3 : p (a ∩ b) = 0.28571428571428575) :
  p a = 0.7142857142857143 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_a_l224_22491


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l224_22460

/-- The focus of the parabola y² = 4x has coordinates (1, 0) -/
theorem parabola_focus_coordinates :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*x}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (1, 0) ∧ 
    (∀ (p : ℝ × ℝ), p ∈ parabola → (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.1 - 0)^2 + (p.2 - 0)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l224_22460
