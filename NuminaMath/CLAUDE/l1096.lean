import Mathlib

namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l1096_109600

theorem least_positive_integer_with_remainder_one (n : ℕ) : n = 9241 ↔ 
  n > 1 ∧ 
  (∀ d ∈ ({5, 7, 8, 10, 11, 12} : Set ℕ), n % d = 1) ∧
  (∀ m : ℕ, m > 1 → (∀ d ∈ ({5, 7, 8, 10, 11, 12} : Set ℕ), m % d = 1) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l1096_109600


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l1096_109619

/-- Sum of first n positive even integers -/
def sumEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of first n positive odd integers -/
def sumOddIntegers (n : ℕ) : ℕ := n^2

/-- Positive difference between two natural numbers -/
def positiveDifference (a b : ℕ) : ℕ := max a b - min a b

theorem even_odd_sum_difference :
  positiveDifference (sumEvenIntegers 25) (3 * sumOddIntegers 20) = 550 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l1096_109619


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1096_109627

theorem sqrt_equation_solution (x : ℝ) :
  x > 6 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3) ↔
  x ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1096_109627


namespace NUMINAMATH_CALUDE_solve_for_y_l1096_109664

theorem solve_for_y (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -6) : y = 29 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1096_109664


namespace NUMINAMATH_CALUDE_fraction_product_l1096_109622

theorem fraction_product (a b c d e f : ℝ) 
  (h1 : a / b = 5 / 2)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 1)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  a * b * c / (d * e * f) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1096_109622


namespace NUMINAMATH_CALUDE_max_integers_with_pairwise_common_divisor_and_coprime_triples_l1096_109671

theorem max_integers_with_pairwise_common_divisor_and_coprime_triples :
  (∃ (n : ℕ) (a : Fin n → ℕ), n ≥ 3 ∧
    (∀ i, a i < 5000) ∧
    (∀ i j, i ≠ j → ∃ d > 1, d ∣ a i ∧ d ∣ a j) ∧
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → Nat.gcd (a i) (Nat.gcd (a j) (a k)) = 1)) →
  (∀ (n : ℕ) (a : Fin n → ℕ), n ≥ 3 →
    (∀ i, a i < 5000) →
    (∀ i j, i ≠ j → ∃ d > 1, d ∣ a i ∧ d ∣ a j) →
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → Nat.gcd (a i) (Nat.gcd (a j) (a k)) = 1) →
    n ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_max_integers_with_pairwise_common_divisor_and_coprime_triples_l1096_109671


namespace NUMINAMATH_CALUDE_tangent_line_of_cubic_l1096_109614

/-- Given a cubic function f(x) with specific derivative conditions, 
    prove that its tangent line at x = 1 has a specific equation. -/
theorem tangent_line_of_cubic (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + b*x + 1
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 + 2*a*x + b
  (f' 1 = 2*a) → (f' 2 = -b) → 
  ∃ m c : ℝ, m = -3 ∧ c = -5/2 ∧ 
    (∀ x y : ℝ, y - c = m * (x - 1) ↔ 6*x + 2*y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_of_cubic_l1096_109614


namespace NUMINAMATH_CALUDE_amanda_almonds_l1096_109629

theorem amanda_almonds (total_almonds : ℚ) (num_piles : ℕ) (amanda_piles : ℕ) : 
  total_almonds = 66 / 7 →
  num_piles = 6 →
  amanda_piles = 3 →
  amanda_piles * (total_almonds / num_piles) = 33 / 7 := by
  sorry

end NUMINAMATH_CALUDE_amanda_almonds_l1096_109629


namespace NUMINAMATH_CALUDE_wall_area_l1096_109684

/-- Represents the types of tiles used on the wall -/
inductive TileType
  | Small
  | Regular
  | Jumbo

/-- Represents the properties of a tile -/
structure Tile where
  type : TileType
  length : ℝ
  width : ℝ

/-- Represents the wall covered with tiles -/
structure Wall where
  smallTiles : Tile
  regularTiles : Tile
  jumboTiles : Tile
  smallTileProportion : ℝ
  regularTileProportion : ℝ
  jumboTileProportion : ℝ
  regularTileArea : ℝ

/-- Theorem stating that the area of the wall is 300 square feet -/
theorem wall_area (w : Wall) : ℝ :=
  by
  have small_ratio : w.smallTiles.length = 2 * w.smallTiles.width := sorry
  have regular_ratio : w.regularTiles.length = 3 * w.regularTiles.width := sorry
  have jumbo_ratio : w.jumboTiles.length = 3 * w.jumboTiles.width := sorry
  have jumbo_length : w.jumboTiles.length = 3 * w.regularTiles.length := sorry
  have tile_proportions : w.smallTileProportion + w.regularTileProportion + w.jumboTileProportion = 1 := sorry
  have no_overlap : w.smallTileProportion * (300 : ℝ) + w.regularTileProportion * (300 : ℝ) + w.jumboTileProportion * (300 : ℝ) = 300 := sorry
  have regular_area : w.regularTileArea = 90 := sorry
  sorry

#check wall_area

end NUMINAMATH_CALUDE_wall_area_l1096_109684


namespace NUMINAMATH_CALUDE_polynomial_inequality_l1096_109609

theorem polynomial_inequality (a b c d : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |3 * a * x^2 + 2 * b * x + c| ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l1096_109609


namespace NUMINAMATH_CALUDE_perfect_square_sum_l1096_109639

theorem perfect_square_sum (a b : ℕ) :
  (∃ k : ℕ, 2^(2*a) + 2^b + 5 = k^2) → (a + b = 4 ∨ a + b = 5) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l1096_109639


namespace NUMINAMATH_CALUDE_no_prime_roots_sum_65_l1096_109628

theorem no_prime_roots_sum_65 : ¬∃ (p q k : ℕ), Prime p ∧ Prime q ∧ p + q = 65 ∧ p * q = k ∧ p^2 - 65*p + k = 0 ∧ q^2 - 65*q + k = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_roots_sum_65_l1096_109628


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1096_109682

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a + 1) * x + y - 2 = 0
def l₂ (a x y : ℝ) : Prop := a * x + (2 * a + 2) * y + 1 = 0

-- Define perpendicularity of two lines
def perpendicular (a : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ → l₂ a x₂ y₂ → 
    (a + 1) * a = -(2 * a + 2)

-- State the theorem
theorem perpendicular_lines_a_values (a : ℝ) :
  perpendicular a → a = -1 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1096_109682


namespace NUMINAMATH_CALUDE_kay_family_age_difference_l1096_109662

/-- Given Kay's family information, prove the age difference. -/
theorem kay_family_age_difference :
  ∀ (kay_age youngest_age oldest_age : ℕ),
    kay_age = 32 →
    oldest_age = 44 →
    oldest_age = 4 * youngest_age →
    (kay_age / 2 : ℚ) - youngest_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_kay_family_age_difference_l1096_109662


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l1096_109698

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_15 :
  lastTwoDigits (sumFactorials 15) = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l1096_109698


namespace NUMINAMATH_CALUDE_man_business_ownership_l1096_109689

theorem man_business_ownership (total_value : ℝ) (sold_value : ℝ) (sold_fraction : ℝ) :
  total_value = 150000 →
  sold_value = 75000 →
  sold_fraction = 3/4 →
  ∃ original_fraction : ℝ,
    original_fraction * total_value * sold_fraction = sold_value ∧
    original_fraction = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_man_business_ownership_l1096_109689


namespace NUMINAMATH_CALUDE_lowest_true_statement_l1096_109653

def statement201 (s203 : Bool) : Bool := s203
def statement202 (s201 : Bool) : Bool := s201
def statement203 (s206 : Bool) : Bool := ¬s206
def statement204 (s202 : Bool) : Bool := ¬s202
def statement205 (s201 s202 s203 s204 : Bool) : Bool := ¬(s201 ∨ s202 ∨ s203 ∨ s204)
def statement206 : Bool := 1 + 1 = 2

theorem lowest_true_statement :
  let s206 := statement206
  let s203 := statement203 s206
  let s201 := statement201 s203
  let s202 := statement202 s201
  let s204 := statement204 s202
  let s205 := statement205 s201 s202 s203 s204
  (¬s201 ∧ ¬s202 ∧ ¬s203 ∧ s204 ∧ ¬s205 ∧ s206) ∧
  (∀ n : Nat, n < 204 → ¬(n = 201 ∧ s201 ∨ n = 202 ∧ s202 ∨ n = 203 ∧ s203)) :=
by sorry

end NUMINAMATH_CALUDE_lowest_true_statement_l1096_109653


namespace NUMINAMATH_CALUDE_classmates_lateness_l1096_109612

theorem classmates_lateness 
  (charlize_lateness : ℕ) 
  (total_lateness : ℕ) 
  (num_classmates : ℕ) 
  (h1 : charlize_lateness = 20)
  (h2 : total_lateness = 140)
  (h3 : num_classmates = 4) :
  (total_lateness - charlize_lateness) / num_classmates = 30 :=
by sorry

end NUMINAMATH_CALUDE_classmates_lateness_l1096_109612


namespace NUMINAMATH_CALUDE_range_of_a_l1096_109637

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1096_109637


namespace NUMINAMATH_CALUDE_jamal_cart_books_l1096_109603

def books_in_cart (history : ℕ) (fiction : ℕ) (children : ℕ) (children_misplaced : ℕ) 
  (science : ℕ) (science_misplaced : ℕ) (biography : ℕ) (remaining : ℕ) : ℕ :=
  history + fiction + (children - children_misplaced) + (science - science_misplaced) + biography + remaining

theorem jamal_cart_books :
  books_in_cart 15 22 10 5 8 3 12 20 = 79 := by
  sorry

end NUMINAMATH_CALUDE_jamal_cart_books_l1096_109603


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l1096_109672

theorem min_sum_of_squares (a b c d : ℤ) : 
  a + b = 18 →
  a * b + c + d = 85 →
  a * d + b * c = 180 →
  c * d = 104 →
  ∃ (min : ℤ), min = 484 ∧ ∀ (a' b' c' d' : ℤ),
    a' + b' = 18 →
    a' * b' + c' + d' = 85 →
    a' * d' + b' * c' = 180 →
    c' * d' = 104 →
    a' ^ 2 + b' ^ 2 + c' ^ 2 + d' ^ 2 ≥ min :=
by sorry


end NUMINAMATH_CALUDE_min_sum_of_squares_l1096_109672


namespace NUMINAMATH_CALUDE_house_width_calculation_l1096_109631

/-- Given a house with length 20.5 feet, a porch measuring 6 feet by 4.5 feet,
    and a total shingle area of 232 square feet, the width of the house is 10 feet. -/
theorem house_width_calculation (house_length porch_length porch_width total_shingle_area : ℝ)
    (h1 : house_length = 20.5)
    (h2 : porch_length = 6)
    (h3 : porch_width = 4.5)
    (h4 : total_shingle_area = 232) :
    (total_shingle_area - porch_length * porch_width) / house_length = 10 := by
  sorry

#check house_width_calculation

end NUMINAMATH_CALUDE_house_width_calculation_l1096_109631


namespace NUMINAMATH_CALUDE_test_scores_l1096_109668

/-- Represents the score of a test -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  score : Nat

/-- Calculates the score for a given test result -/
def calculateScore (correct unanswered incorrect : Nat) : Nat :=
  6 * correct + unanswered

/-- Checks if a given score is achievable on the test -/
def isAchievableScore (s : Nat) : Prop :=
  ∃ (correct unanswered incorrect : Nat),
    correct + unanswered + incorrect = 25 ∧
    calculateScore correct unanswered incorrect = s

theorem test_scores :
  (isAchievableScore 130) ∧
  (isAchievableScore 131) ∧
  (isAchievableScore 133) ∧
  (isAchievableScore 138) ∧
  ¬(isAchievableScore 139) := by
  sorry

end NUMINAMATH_CALUDE_test_scores_l1096_109668


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l1096_109606

theorem stratified_sampling_sample_size 
  (total_teachers : ℕ) 
  (total_male_students : ℕ) 
  (total_female_students : ℕ) 
  (sampled_female_students : ℕ) 
  (h1 : total_teachers = 100) 
  (h2 : total_male_students = 600) 
  (h3 : total_female_students = 500) 
  (h4 : sampled_female_students = 40) : 
  (sampled_female_students : ℚ) / total_female_students = 
  96 / (total_teachers + total_male_students + total_female_students) := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l1096_109606


namespace NUMINAMATH_CALUDE_birthday_games_increase_l1096_109659

theorem birthday_games_increase (initial_games : ℕ) (increase_percentage : ℚ) : 
  initial_games = 7 → 
  increase_percentage = 30 / 100 → 
  initial_games + Int.floor (increase_percentage * initial_games) = 9 := by
  sorry

end NUMINAMATH_CALUDE_birthday_games_increase_l1096_109659


namespace NUMINAMATH_CALUDE_janice_purchase_l1096_109646

theorem janice_purchase (a b c : ℕ) : 
  a + b + c = 30 →
  30 * a + 200 * b + 300 * c = 3000 →
  a = 20 :=
by sorry

end NUMINAMATH_CALUDE_janice_purchase_l1096_109646


namespace NUMINAMATH_CALUDE_die_roll_probability_l1096_109651

theorem die_roll_probability : 
  let p : ℝ := 1 / 2  -- probability of rolling an even number on a single die
  let n : ℕ := 8      -- number of rolls
  1 - (1 - p) ^ n = 255 / 256 :=
by sorry

end NUMINAMATH_CALUDE_die_roll_probability_l1096_109651


namespace NUMINAMATH_CALUDE_max_y_coordinate_value_l1096_109621

noncomputable def max_y_coordinate (θ : ℝ) : ℝ :=
  let r := Real.sin (3 * θ)
  r * Real.sin θ

theorem max_y_coordinate_value :
  ∃ (θ : ℝ), ∀ (φ : ℝ), max_y_coordinate θ ≥ max_y_coordinate φ ∧
  max_y_coordinate θ = 3 * (3 / 16) ^ (1 / 3) - 4 * 3 ^ (4 / 3) / 16 ^ (4 / 3) :=
sorry

end NUMINAMATH_CALUDE_max_y_coordinate_value_l1096_109621


namespace NUMINAMATH_CALUDE_binomial_18_10_l1096_109676

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 45760 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l1096_109676


namespace NUMINAMATH_CALUDE_expression_equals_one_tenth_l1096_109642

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Define the expression
def expression : ℚ :=
  (ceiling ((21 : ℚ) / 8 - ceiling ((35 : ℚ) / 21))) /
  (ceiling ((35 : ℚ) / 8 + ceiling ((8 * 21 : ℚ) / 35)))

-- Theorem statement
theorem expression_equals_one_tenth : expression = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_tenth_l1096_109642


namespace NUMINAMATH_CALUDE_game_cost_calculation_l1096_109657

theorem game_cost_calculation (total_earnings : ℕ) (blade_cost : ℕ) (num_games : ℕ) 
  (h1 : total_earnings = 101)
  (h2 : blade_cost = 47)
  (h3 : num_games = 9)
  (h4 : (total_earnings - blade_cost) % num_games = 0) :
  (total_earnings - blade_cost) / num_games = 6 := by
  sorry

end NUMINAMATH_CALUDE_game_cost_calculation_l1096_109657


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1096_109640

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x ≤ 24 ∧ (1015 + x) % 25 = 0 ∧ ∀ y : ℕ, y < x → (1015 + y) % 25 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1096_109640


namespace NUMINAMATH_CALUDE_function_inequality_l1096_109663

-- Define the function f(x) = ax - x^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

-- State the theorem
theorem function_inequality (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f a x₂ - f a x₁ > x₂ - x₁) →
  a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1096_109663


namespace NUMINAMATH_CALUDE_billboard_perimeter_l1096_109630

/-- Represents a rectangular billboard --/
structure Billboard where
  length : ℝ
  width : ℝ

/-- The area of a billboard --/
def area (b : Billboard) : ℝ := b.length * b.width

/-- The perimeter of a billboard --/
def perimeter (b : Billboard) : ℝ := 2 * (b.length + b.width)

theorem billboard_perimeter :
  ∀ b : Billboard,
    area b = 91 ∧
    b.width = 7 →
    perimeter b = 40 := by
  sorry

end NUMINAMATH_CALUDE_billboard_perimeter_l1096_109630


namespace NUMINAMATH_CALUDE_impossible_all_black_l1096_109654

/-- Represents the color of a square on the board -/
inductive Color
| White
| Black

/-- Represents a 4x4 board -/
def Board := Fin 4 → Fin 4 → Color

/-- Represents a 1x3 rectangle on the board -/
structure Rectangle :=
  (row : Fin 4)
  (col : Fin 4)
  (horizontal : Bool)

/-- Initial state of the board where all squares are white -/
def initialBoard : Board :=
  λ _ _ => Color.White

/-- Applies a move to the board by flipping colors in a 1x3 rectangle -/
def applyMove (b : Board) (r : Rectangle) : Board :=
  sorry

/-- Checks if all squares on the board are black -/
def allBlack (b : Board) : Prop :=
  ∀ i j, b i j = Color.Black

/-- Theorem stating that it's impossible to make all squares black -/
theorem impossible_all_black :
  ¬ ∃ (moves : List Rectangle), allBlack (moves.foldl applyMove initialBoard) :=
sorry

end NUMINAMATH_CALUDE_impossible_all_black_l1096_109654


namespace NUMINAMATH_CALUDE_log_equation_solution_l1096_109623

theorem log_equation_solution (x : ℝ) (h : Real.log 125 / Real.log (3 * x) = x) :
  (∃ (a b : ℤ), x = a / b ∧ a ≠ 0 ∧ b > 0 ∧ (∀ n : ℕ, n > 1 → (a : ℝ) / b ≠ n^2 ∧ (a : ℝ) / b ≠ n^3) ∧ ¬∃ (n : ℤ), (a : ℝ) / b = n) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1096_109623


namespace NUMINAMATH_CALUDE_most_precise_announcement_l1096_109610

def K_approx : ℝ := 5.72788
def error_margin : ℝ := 0.00625

def is_valid_announcement (x : ℝ) : Prop :=
  ∀ y : ℝ, |y - K_approx| ≤ error_margin → |x - y| < 0.05

theorem most_precise_announcement :
  is_valid_announcement 5.7 ∧
  ∀ z : ℝ, is_valid_announcement z → |z - 5.7| < 0.05 :=
sorry

end NUMINAMATH_CALUDE_most_precise_announcement_l1096_109610


namespace NUMINAMATH_CALUDE_pizza_cost_three_pizzas_cost_l1096_109633

/-- The cost of all pizzas given the number of pizzas, slices per pizza, and the cost of a subset of slices. -/
theorem pizza_cost (num_pizzas : ℕ) (slices_per_pizza : ℕ) (subset_slices : ℕ) (subset_cost : ℚ) : ℚ :=
  let total_slices := num_pizzas * slices_per_pizza
  let cost_per_slice := subset_cost / subset_slices
  total_slices * cost_per_slice

/-- Proof that 3 pizzas with 12 slices each cost $72, given that 5 slices cost $10. -/
theorem three_pizzas_cost : pizza_cost 3 12 5 10 = 72 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cost_three_pizzas_cost_l1096_109633


namespace NUMINAMATH_CALUDE_expression_equality_l1096_109692

theorem expression_equality (x : ℝ) : x * (x * (x * (2 - x) - 4) + 10) + 1 = -x^4 + 2*x^3 - 4*x^2 + 10*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1096_109692


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_range_l1096_109667

theorem parabola_axis_of_symmetry_range 
  (a b c m n t : ℝ) 
  (h_a_pos : a > 0)
  (h_point1 : m = a + b + c)
  (h_point2 : n = 9*a + 3*b + c)
  (h_order : m < n ∧ n < c)
  (h_axis : t = -b / (2*a)) : 
  3/2 < t ∧ t < 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_range_l1096_109667


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_three_digit_product_l1096_109656

theorem smallest_multiplier_for_three_digit_product : 
  (∀ k : ℕ, k < 4 → 27 * k < 100) ∧ (27 * 4 ≥ 100 ∧ 27 * 4 < 1000) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_three_digit_product_l1096_109656


namespace NUMINAMATH_CALUDE_mars_mission_cost_share_l1096_109696

-- Define the given conditions
def cost_in_euros : ℝ := 25e9
def number_of_people : ℝ := 300e6
def exchange_rate : ℝ := 1.2

-- Define the theorem to prove
theorem mars_mission_cost_share :
  (cost_in_euros * exchange_rate) / number_of_people = 100 := by
  sorry

end NUMINAMATH_CALUDE_mars_mission_cost_share_l1096_109696


namespace NUMINAMATH_CALUDE_eight_book_distribution_l1096_109695

/-- The number of ways to distribute identical books between a library and being checked out -/
def distribute_books (total : ℕ) : ℕ :=
  if total < 2 then 0 else total - 1

/-- Theorem: For 8 identical books, there are 7 ways to distribute them between a library and being checked out, with at least one book in each location -/
theorem eight_book_distribution :
  distribute_books 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_eight_book_distribution_l1096_109695


namespace NUMINAMATH_CALUDE_expression_evaluation_l1096_109635

theorem expression_evaluation : 
  let x : ℝ := -2
  3 * (-2 * x^2 + 5 + 4 * x) - (5 * x - 4 - 7 * x^2) = 9 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1096_109635


namespace NUMINAMATH_CALUDE_complex_vector_relation_l1096_109670

theorem complex_vector_relation (z₁ z₂ z₃ : ℂ) (x y : ℝ)
  (h₁ : z₁ = -1 + 2 * Complex.I)
  (h₂ : z₂ = 1 - Complex.I)
  (h₃ : z₃ = 3 - 2 * Complex.I)
  (h₄ : z₃ = x • z₁ + y • z₂) :
  x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_vector_relation_l1096_109670


namespace NUMINAMATH_CALUDE_marble_bag_count_l1096_109658

theorem marble_bag_count :
  ∀ (total blue red white : ℕ),
    blue = 5 →
    red = 9 →
    total = blue + red + white →
    (red + white : ℚ) / total = 5 / 6 →
    total = 30 := by
  sorry

end NUMINAMATH_CALUDE_marble_bag_count_l1096_109658


namespace NUMINAMATH_CALUDE_complex_subtraction_l1096_109688

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 2 + 3*I) :
  a - 3*b = -1 - 12*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1096_109688


namespace NUMINAMATH_CALUDE_perimeter_comparison_l1096_109697

-- Define a structure for rectangular parallelepiped
structure RectangularParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  positive_dimensions : 0 < length ∧ 0 < width ∧ 0 < height

-- Define a function to calculate the perimeter of a rectangular parallelepiped
def perimeter (p : RectangularParallelepiped) : ℝ :=
  4 * (p.length + p.width + p.height)

-- Define what it means for one parallelepiped to be contained within another
def contained_within (p q : RectangularParallelepiped) : Prop :=
  p.length ≤ q.length ∧ p.width ≤ q.width ∧ p.height ≤ q.height

-- Theorem statement
theorem perimeter_comparison 
  (p q : RectangularParallelepiped) 
  (h : contained_within p q) : 
  perimeter p ≤ perimeter q :=
sorry

end NUMINAMATH_CALUDE_perimeter_comparison_l1096_109697


namespace NUMINAMATH_CALUDE_hans_deposit_is_101_l1096_109608

/-- Calculates the deposit for a restaurant reservation --/
def calculate_deposit (num_adults num_children num_seniors : ℕ) 
  (flat_deposit adult_charge child_charge senior_charge service_charge : ℕ) 
  (split_bill : Bool) : ℕ :=
  flat_deposit + 
  num_adults * adult_charge + 
  num_children * child_charge + 
  num_seniors * senior_charge +
  (if split_bill then service_charge else 0)

/-- Theorem: The deposit for Hans' reservation is $101 --/
theorem hans_deposit_is_101 : 
  calculate_deposit 10 2 3 25 5 2 4 10 true = 101 := by
  sorry

end NUMINAMATH_CALUDE_hans_deposit_is_101_l1096_109608


namespace NUMINAMATH_CALUDE_at_least_one_woman_probability_l1096_109650

def num_men : ℕ := 9
def num_women : ℕ := 6
def total_people : ℕ := num_men + num_women
def num_selected : ℕ := 4

theorem at_least_one_woman_probability :
  (1 : ℚ) - (Nat.choose num_men num_selected : ℚ) / (Nat.choose total_people num_selected : ℚ) = 13/15 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_woman_probability_l1096_109650


namespace NUMINAMATH_CALUDE_is_stratified_sampling_l1096_109602

/-- Represents a sampling method -/
structure SamplingMethod where
  name : String
  dividePopulation : Bool
  sampleFromParts : Bool
  proportionalSampling : Bool
  combineSamples : Bool

/-- Definition of stratified sampling -/
def stratifiedSampling : SamplingMethod :=
  { name := "Stratified Sampling",
    dividePopulation := true,
    sampleFromParts := true,
    proportionalSampling := true,
    combineSamples := true }

/-- Theorem stating that a sampling method with specific characteristics is stratified sampling -/
theorem is_stratified_sampling
  (method : SamplingMethod)
  (h1 : method.dividePopulation = true)
  (h2 : method.sampleFromParts = true)
  (h3 : method.proportionalSampling = true)
  (h4 : method.combineSamples = true) :
  method = stratifiedSampling := by
  sorry

#check is_stratified_sampling

end NUMINAMATH_CALUDE_is_stratified_sampling_l1096_109602


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l1096_109673

theorem smallest_whole_number_above_sum : ⌈(10/3 : ℚ) + (17/4 : ℚ) + (26/5 : ℚ) + (37/6 : ℚ)⌉ = 19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l1096_109673


namespace NUMINAMATH_CALUDE_smallest_whole_number_satisfying_inequality_two_satisfies_inequality_two_is_smallest_l1096_109687

theorem smallest_whole_number_satisfying_inequality :
  ∀ x : ℤ, (3 * x + 4 > 11 - 2 * x) → x ≥ 2 :=
by sorry

theorem two_satisfies_inequality :
  3 * 2 + 4 > 11 - 2 * 2 :=
by sorry

theorem two_is_smallest :
  ∀ x : ℤ, x < 2 → (3 * x + 4 ≤ 11 - 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_satisfying_inequality_two_satisfies_inequality_two_is_smallest_l1096_109687


namespace NUMINAMATH_CALUDE_smallest_addend_proof_l1096_109607

/-- The smallest non-negative integer that, when added to 27452, makes the sum divisible by 9 -/
def smallest_addend : ℕ := 7

/-- The original number we're working with -/
def original_number : ℕ := 27452

theorem smallest_addend_proof :
  (∀ k : ℕ, k < smallest_addend → ¬((original_number + k) % 9 = 0)) ∧
  ((original_number + smallest_addend) % 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_addend_proof_l1096_109607


namespace NUMINAMATH_CALUDE_focus_coordinates_l1096_109669

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 9 = 1 ∧ a > 0

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = 3 * x

-- Define the parabola
def parabola (a : ℝ) (x y : ℝ) : Prop :=
  y = 2 * a * x^2

-- State the theorem
theorem focus_coordinates (a : ℝ) :
  (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) →
  (∃ x y : ℝ, parabola a x y ∧ x = 0 ∧ y = 1/8) :=
sorry

end NUMINAMATH_CALUDE_focus_coordinates_l1096_109669


namespace NUMINAMATH_CALUDE_linear_function_proof_l1096_109677

/-- A linear function passing through two points -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

/-- The linear function passes through the point (3,1) -/
def PassesThrough3_1 (k b : ℝ) : Prop := LinearFunction k b 3 = 1

/-- The linear function passes through the point (2,0) -/
def PassesThrough2_0 (k b : ℝ) : Prop := LinearFunction k b 2 = 0

theorem linear_function_proof (k b : ℝ) 
  (h1 : PassesThrough3_1 k b) (h2 : PassesThrough2_0 k b) :
  (∀ x, LinearFunction k b x = x - 2) ∧ (LinearFunction k b 6 = 4) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_proof_l1096_109677


namespace NUMINAMATH_CALUDE_triangle_inequality_new_magnitude_min_magnitude_on_line_max_magnitude_on_circle_l1096_109665

-- Define the new magnitude
def new_magnitude (x y : ℝ) : ℝ := |x + y| + |x - y|

-- Theorem for proposition (1)
theorem triangle_inequality_new_magnitude (x₁ y₁ x₂ y₂ : ℝ) :
  new_magnitude (x₁ - x₂) (y₁ - y₂) ≤ new_magnitude x₁ y₁ + new_magnitude x₂ y₂ := by
  sorry

-- Theorem for proposition (2)
theorem min_magnitude_on_line :
  ∃ (t : ℝ), ∀ (s : ℝ), new_magnitude t (t - 1) ≤ new_magnitude s (s - 1) ∧ new_magnitude t (t - 1) = 1 := by
  sorry

-- Theorem for proposition (3)
theorem max_magnitude_on_circle :
  ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ new_magnitude x y = 2 ∧ 
  ∀ (a b : ℝ), a^2 + b^2 = 1 → new_magnitude a b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_new_magnitude_min_magnitude_on_line_max_magnitude_on_circle_l1096_109665


namespace NUMINAMATH_CALUDE_original_number_proof_l1096_109699

theorem original_number_proof (x : ℕ) : 
  (x + 4) % 23 = 0 → x > 0 → x = 19 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l1096_109699


namespace NUMINAMATH_CALUDE_equivalent_statements_l1096_109605

theorem equivalent_statements :
  (∀ x : ℝ, x ≥ 0 → x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0 → x < 0) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_statements_l1096_109605


namespace NUMINAMATH_CALUDE_on_y_axis_on_x_axis_abscissa_greater_than_ordinate_l1096_109638

-- Define point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (2*m + 4, m - 1)

-- Theorem for condition (1)
theorem on_y_axis (m : ℝ) : 
  P m = (0, -3) ↔ P m = (0, (P m).2) :=
sorry

-- Theorem for condition (2)
theorem on_x_axis (m : ℝ) :
  P m = (6, 0) ↔ P m = ((P m).1, 0) :=
sorry

-- Theorem for condition (3)
theorem abscissa_greater_than_ordinate (m : ℝ) :
  P m = (-4, -5) ↔ (P m).1 = (P m).2 + 1 :=
sorry

end NUMINAMATH_CALUDE_on_y_axis_on_x_axis_abscissa_greater_than_ordinate_l1096_109638


namespace NUMINAMATH_CALUDE_fifth_month_sale_l1096_109634

theorem fifth_month_sale
  (target_average : ℕ)
  (num_months : ℕ)
  (sales : Fin 4 → ℕ)
  (sixth_month_sale : ℕ)
  (h1 : target_average = 6000)
  (h2 : num_months = 5)
  (h3 : sales 0 = 5420)
  (h4 : sales 1 = 5660)
  (h5 : sales 2 = 6200)
  (h6 : sales 3 = 6350)
  (h7 : sixth_month_sale = 5870) :
  ∃ (fifth_month_sale : ℕ),
    fifth_month_sale = target_average * num_months - (sales 0 + sales 1 + sales 2 + sales 3) :=
by sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l1096_109634


namespace NUMINAMATH_CALUDE_office_paper_shortage_l1096_109693

def paper_shortage (pack1 pack2 mon_wed_fri_usage tue_thu_usage : ℕ) (period : ℕ) : ℤ :=
  (pack1 + pack2 : ℤ) - (3 * mon_wed_fri_usage + 2 * tue_thu_usage) * period

theorem office_paper_shortage :
  paper_shortage 240 320 60 100 2 = -200 :=
by sorry

end NUMINAMATH_CALUDE_office_paper_shortage_l1096_109693


namespace NUMINAMATH_CALUDE_train_length_problem_l1096_109647

/-- Represents the problem of calculating the length of a train based on James's jogging --/
theorem train_length_problem (james_speed : ℝ) (train_speed : ℝ) (steps_forward : ℕ) (steps_backward : ℕ) :
  james_speed > train_speed →
  steps_forward = 400 →
  steps_backward = 160 →
  let train_length := (steps_forward * james_speed - steps_forward * train_speed + 
                       steps_backward * james_speed + steps_backward * train_speed) / 2
  train_length = 640 / 7 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l1096_109647


namespace NUMINAMATH_CALUDE_k_value_l1096_109632

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer. -/
def length (k : ℕ) : ℕ := sorry

/-- k is an integer greater than 1 with a length of 4 and prime factors 2, 2, 2, and 3 -/
def k : ℕ := sorry

theorem k_value : k = 24 := by sorry

end NUMINAMATH_CALUDE_k_value_l1096_109632


namespace NUMINAMATH_CALUDE_average_of_eight_thirteen_and_M_l1096_109660

theorem average_of_eight_thirteen_and_M (M : ℝ) (h1 : 12 < M) (h2 : M < 22) :
  (8 + 13 + M) / 3 = 13 := by
sorry

end NUMINAMATH_CALUDE_average_of_eight_thirteen_and_M_l1096_109660


namespace NUMINAMATH_CALUDE_jills_peaches_l1096_109691

theorem jills_peaches (steven_peaches : ℕ) (jake_fewer : ℕ) (jake_more : ℕ) 
  (h1 : steven_peaches = 14)
  (h2 : jake_fewer = 6)
  (h3 : jake_more = 3)
  : steven_peaches - jake_fewer - jake_more = 5 := by
  sorry

end NUMINAMATH_CALUDE_jills_peaches_l1096_109691


namespace NUMINAMATH_CALUDE_sam_recycling_cans_l1096_109666

/-- The number of bags Sam filled on Saturday -/
def saturday_bags : ℕ := 3

/-- The number of bags Sam filled on Sunday -/
def sunday_bags : ℕ := 4

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 9

/-- The total number of cans Sam picked up -/
def total_cans : ℕ := (saturday_bags + sunday_bags) * cans_per_bag

theorem sam_recycling_cans : total_cans = 63 := by
  sorry

end NUMINAMATH_CALUDE_sam_recycling_cans_l1096_109666


namespace NUMINAMATH_CALUDE_correct_probability_order_l1096_109601

/-- Enum representing the five types of phenomena -/
inductive Phenomenon
  | CertainToHappen
  | VeryLikelyToHappen
  | PossibleToHappen
  | ImpossibleToHappen
  | UnlikelyToHappen

/-- Function to compare the probability of two phenomena -/
def probabilityLessThan (a b : Phenomenon) : Prop :=
  match a, b with
  | Phenomenon.ImpossibleToHappen, _ => a ≠ b
  | Phenomenon.UnlikelyToHappen, Phenomenon.ImpossibleToHappen => False
  | Phenomenon.UnlikelyToHappen, _ => a ≠ b
  | Phenomenon.PossibleToHappen, Phenomenon.ImpossibleToHappen => False
  | Phenomenon.PossibleToHappen, Phenomenon.UnlikelyToHappen => False
  | Phenomenon.PossibleToHappen, _ => a ≠ b
  | Phenomenon.VeryLikelyToHappen, Phenomenon.CertainToHappen => True
  | Phenomenon.VeryLikelyToHappen, _ => False
  | Phenomenon.CertainToHappen, _ => False

/-- Theorem stating the correct order of phenomena by probability -/
theorem correct_probability_order :
  probabilityLessThan Phenomenon.ImpossibleToHappen Phenomenon.UnlikelyToHappen ∧
  probabilityLessThan Phenomenon.UnlikelyToHappen Phenomenon.PossibleToHappen ∧
  probabilityLessThan Phenomenon.PossibleToHappen Phenomenon.VeryLikelyToHappen ∧
  probabilityLessThan Phenomenon.VeryLikelyToHappen Phenomenon.CertainToHappen :=
sorry

end NUMINAMATH_CALUDE_correct_probability_order_l1096_109601


namespace NUMINAMATH_CALUDE_right_triangle_trig_identity_l1096_109611

/-- Given a right triangle PQR with hypotenuse PQ = 15 and PR = 9, prove that sin Q = 4/5 and the trigonometric identity sin² Q + cos² Q = 1 holds. -/
theorem right_triangle_trig_identity (PQ PR : ℝ) (hPQ : PQ = 15) (hPR : PR = 9) :
  let sinQ := Real.sqrt (PQ^2 - PR^2) / PQ
  let cosQ := PR / PQ
  sinQ = 4/5 ∧ sinQ^2 + cosQ^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_identity_l1096_109611


namespace NUMINAMATH_CALUDE_board_game_theorem_l1096_109652

/-- Represents the operation of replacing two numbers with their combination -/
def combine (a b : ℚ) : ℚ := a * b + a + b

/-- The set of initial numbers on the board -/
def initial_numbers (n : ℕ) : List ℚ := List.range n |>.map (λ i => 1 / (i + 1))

/-- The invariant product of all numbers on the board increased by 1 -/
def product_plus_one (numbers : List ℚ) : ℚ := numbers.foldl (λ acc x => acc * (x + 1)) 1

/-- The final number after n-1 operations -/
def final_number (n : ℕ) : ℚ := n

theorem board_game_theorem (n : ℕ) (h : n > 0) :
  ∃ (operations : List (ℕ × ℕ)),
    operations.length = n - 1 ∧
    final_number n = product_plus_one (initial_numbers n) - 1 := by
  sorry

#check board_game_theorem

end NUMINAMATH_CALUDE_board_game_theorem_l1096_109652


namespace NUMINAMATH_CALUDE_min_sum_eccentricities_l1096_109675

theorem min_sum_eccentricities (e₁ e₂ : ℝ) (h₁ : e₁ > 0) (h₂ : e₂ > 0) 
  (h : 1 / (e₁^2) + 1 / (e₂^2) = 1) : 
  e₁ + e₂ ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_eccentricities_l1096_109675


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l1096_109617

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (2 * Real.sin θ - Real.cos θ)

-- Define the Cartesian equation of a line
def line_equation (x y : ℝ) : Prop :=
  2 * y - x = 1

-- Theorem statement
theorem polar_to_cartesian_line :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ → 
    x = r * Real.cos θ →
    y = r * Real.sin θ →
    line_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_polar_to_cartesian_line_l1096_109617


namespace NUMINAMATH_CALUDE_product_digits_sum_l1096_109626

def A : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def B : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def C : ℕ := (A * B) / 10000 % 10
def D : ℕ := A * B % 10

theorem product_digits_sum : C + D = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_digits_sum_l1096_109626


namespace NUMINAMATH_CALUDE_point_B_complex_number_l1096_109655

theorem point_B_complex_number 
  (A C : ℂ) 
  (AC BC : ℂ) 
  (h1 : A = 3 + I) 
  (h2 : AC = -2 - 4*I) 
  (h3 : BC = -4 - I) 
  (h4 : C = A + AC) :
  A + AC + BC = 5 - 2*I := by
sorry

end NUMINAMATH_CALUDE_point_B_complex_number_l1096_109655


namespace NUMINAMATH_CALUDE_angle_TSB_closest_to_27_l1096_109649

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The art gallery setup -/
structure ArtGallery where
  B : Point -- Bottom of painting
  T : Point -- Top of painting
  S : Point -- Spotlight position

/-- Definition of the art gallery setup based on given conditions -/
def setupGallery : ArtGallery :=
  { B := ⟨0, 1⟩,    -- Bottom of painting (0, 1)
    T := ⟨0, 3⟩,    -- Top of painting (0, 3)
    S := ⟨3, 4⟩ }   -- Spotlight position (3, 4)

/-- Calculate the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Theorem stating that the angle TSB is closest to 27° -/
theorem angle_TSB_closest_to_27 (g : ArtGallery) :
  let angleTSB := angle g.T g.S g.B
  ∀ x ∈ [27, 63, 34, 45, 18], |angleTSB - 27| ≤ |angleTSB - x| :=
by sorry

end NUMINAMATH_CALUDE_angle_TSB_closest_to_27_l1096_109649


namespace NUMINAMATH_CALUDE_abc_and_fourth_power_sum_l1096_109613

theorem abc_and_fourth_power_sum (a b c : ℝ) 
  (sum_1 : a + b + c = 1)
  (sum_2 : a^2 + b^2 + c^2 = 2)
  (sum_3 : a^3 + b^3 + c^3 = 3) :
  a * b * c = 1/6 ∧ a^4 + b^4 + c^4 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_abc_and_fourth_power_sum_l1096_109613


namespace NUMINAMATH_CALUDE_danivan_initial_inventory_l1096_109679

/-- Represents the inventory and sales data for Danivan Drugstore --/
structure DrugstoreData where
  monday_sales : ℕ
  tuesday_sales : ℕ
  daily_sales_wed_to_sun : ℕ
  saturday_delivery : ℕ
  end_of_week_inventory : ℕ

/-- Calculates the initial inventory of hand sanitizer gel bottles --/
def initial_inventory (data : DrugstoreData) : ℕ :=
  data.end_of_week_inventory + 
  data.monday_sales + 
  data.tuesday_sales + 
  (5 * data.daily_sales_wed_to_sun) - 
  data.saturday_delivery

/-- Theorem stating that the initial inventory is 4500 bottles --/
theorem danivan_initial_inventory : 
  initial_inventory {
    monday_sales := 2445,
    tuesday_sales := 900,
    daily_sales_wed_to_sun := 50,
    saturday_delivery := 650,
    end_of_week_inventory := 1555
  } = 4500 := by
  sorry


end NUMINAMATH_CALUDE_danivan_initial_inventory_l1096_109679


namespace NUMINAMATH_CALUDE_ones_digit_of_3_to_52_l1096_109680

theorem ones_digit_of_3_to_52 : (3^52 : ℕ) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_3_to_52_l1096_109680


namespace NUMINAMATH_CALUDE_fayes_age_l1096_109683

/-- Given the ages of Diana, Eduardo, Chad, Faye, and Greg, prove Faye's age --/
theorem fayes_age 
  (D E C F G : ℕ) -- Ages of Diana, Eduardo, Chad, Faye, and Greg
  (h1 : D = E - 2)
  (h2 : C = E + 3)
  (h3 : F = C - 1)
  (h4 : D = 16)
  (h5 : G = D - 5) :
  F = 20 := by
  sorry

end NUMINAMATH_CALUDE_fayes_age_l1096_109683


namespace NUMINAMATH_CALUDE_mistaken_division_l1096_109641

theorem mistaken_division (n : ℕ) (h : 2 * n = 622) : 
  (n / 12) + (n % 12) = 36 := by
sorry

end NUMINAMATH_CALUDE_mistaken_division_l1096_109641


namespace NUMINAMATH_CALUDE_greatest_n_value_l1096_109643

theorem greatest_n_value (n : ℤ) (h : 102 * n^2 ≤ 8100) : n ≤ 8 ∧ ∃ (m : ℤ), m = 8 ∧ 102 * m^2 ≤ 8100 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_l1096_109643


namespace NUMINAMATH_CALUDE_journey_problem_l1096_109678

theorem journey_problem (total_distance : ℝ) (days : ℕ) (ratio : ℝ) 
  (h1 : total_distance = 378)
  (h2 : days = 6)
  (h3 : ratio = 1/2) :
  let first_day := total_distance * (1 - ratio) / (1 - ratio^days)
  first_day * ratio = 96 := by
  sorry

end NUMINAMATH_CALUDE_journey_problem_l1096_109678


namespace NUMINAMATH_CALUDE_square_area_ratio_l1096_109644

/-- Given three square regions A, B, and C, where the perimeter of A is 20 units and
    the perimeter of B is 40 units, and assuming the side length of C increases
    proportionally from B as B did from A, the ratio of the area of A to the area of C is 1/16. -/
theorem square_area_ratio (A B C : ℝ) : 
  (A * 4 = 20) →  -- Perimeter of A is 20 units
  (B * 4 = 40) →  -- Perimeter of B is 40 units
  (C = 2 * B) →   -- Side length of C increases proportionally
  (A^2 / C^2 = 1/16) :=
by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1096_109644


namespace NUMINAMATH_CALUDE_inequality_of_product_one_l1096_109636

theorem inequality_of_product_one (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_product_one_l1096_109636


namespace NUMINAMATH_CALUDE_smallest_n_with_gcd_conditions_l1096_109645

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem smallest_n_with_gcd_conditions :
  ∃ (n : ℕ), n > 200 ∧ 
  Nat.gcd 70 (n + 150) = 35 ∧ 
  Nat.gcd (n + 70) 150 = 75 ∧
  ∀ (m : ℕ), m > 200 → 
    Nat.gcd 70 (m + 150) = 35 → 
    Nat.gcd (m + 70) 150 = 75 → 
    n ≤ m ∧
  n = 305 ∧
  digit_sum n = 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_with_gcd_conditions_l1096_109645


namespace NUMINAMATH_CALUDE_equality_of_fractions_l1096_109620

theorem equality_of_fractions (x y z k : ℝ) :
  (5 / (x + y) = k / (x - z)) ∧ (k / (x - z) = 9 / (z + y)) → k = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l1096_109620


namespace NUMINAMATH_CALUDE_consecutive_numbers_with_lcm_168_l1096_109616

theorem consecutive_numbers_with_lcm_168 (a b c : ℕ) : 
  (b = a + 1) → (c = b + 1) → Nat.lcm (Nat.lcm a b) c = 168 → a + b + c = 21 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_with_lcm_168_l1096_109616


namespace NUMINAMATH_CALUDE_grape_juice_amount_l1096_109694

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_percent : ℝ
  orange_watermelon_oz : ℝ

/-- The fruit drink satisfies the given conditions -/
def valid_fruit_drink (drink : FruitDrink) : Prop :=
  drink.orange_percent = 0.15 ∧
  drink.watermelon_percent = 0.60 ∧
  drink.orange_watermelon_oz = 120 ∧
  drink.orange_percent + drink.watermelon_percent + drink.grape_percent = 1

/-- Calculate the amount of grape juice in ounces -/
def grape_juice_oz (drink : FruitDrink) : ℝ :=
  drink.grape_percent * drink.total

/-- Theorem stating that the amount of grape juice is 40 ounces -/
theorem grape_juice_amount (drink : FruitDrink) 
  (h : valid_fruit_drink drink) : grape_juice_oz drink = 40 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_amount_l1096_109694


namespace NUMINAMATH_CALUDE_san_diego_zoo_ticket_cost_l1096_109690

/-- Calculates the total cost of zoo tickets for a family -/
def total_cost_zoo_tickets (family_size : ℕ) (adult_price : ℕ) (child_price : ℕ) (adult_tickets : ℕ) : ℕ :=
  let child_tickets := family_size - adult_tickets
  adult_price * adult_tickets + child_price * child_tickets

/-- Theorem: The total cost of zoo tickets for a family of 7 with 4 adult tickets is $126 -/
theorem san_diego_zoo_ticket_cost :
  total_cost_zoo_tickets 7 21 14 4 = 126 := by
  sorry

end NUMINAMATH_CALUDE_san_diego_zoo_ticket_cost_l1096_109690


namespace NUMINAMATH_CALUDE_family_boys_count_l1096_109615

/-- Represents a family with boys and girls -/
structure Family where
  boys : ℕ
  girls : ℕ

/-- A child in the family -/
structure Child where
  brothers : ℕ
  sisters : ℕ

/-- Defines a valid family based on the problem conditions -/
def isValidFamily (f : Family) : Prop :=
  ∃ (c1 c2 : Child),
    c1.brothers = 3 ∧ c1.sisters = 6 ∧
    c2.brothers = 4 ∧ c2.sisters = 5 ∧
    f.boys = c1.brothers + 1 ∧
    f.girls = c1.sisters + 1

theorem family_boys_count (f : Family) :
  isValidFamily f → f.boys = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_family_boys_count_l1096_109615


namespace NUMINAMATH_CALUDE_mod_product_equiv_l1096_109681

theorem mod_product_equiv (m : ℕ) : 
  (264 * 391 ≡ m [ZMOD 100]) → 
  (0 ≤ m ∧ m < 100) → 
  m = 24 := by
  sorry

end NUMINAMATH_CALUDE_mod_product_equiv_l1096_109681


namespace NUMINAMATH_CALUDE_katie_baked_18_cupcakes_l1096_109661

/-- The number of cupcakes Todd ate -/
def todd_ate : ℕ := 8

/-- The number of packages Katie could make after Todd ate some cupcakes -/
def num_packages : ℕ := 5

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 2

/-- The initial number of cupcakes Katie baked -/
def initial_cupcakes : ℕ := todd_ate + num_packages * cupcakes_per_package

theorem katie_baked_18_cupcakes : initial_cupcakes = 18 := by
  sorry

end NUMINAMATH_CALUDE_katie_baked_18_cupcakes_l1096_109661


namespace NUMINAMATH_CALUDE_total_cost_approx_636_38_l1096_109604

def membership_fee (initial_fee : ℝ) (increase_rates : List ℝ) (discount_rates : List ℝ) : ℝ :=
  let fees := List.scanl (λ acc rate => acc * (1 + rate)) initial_fee increase_rates
  let discounted_fees := List.zipWith (λ fee discount => fee * (1 - discount)) fees discount_rates
  discounted_fees.sum

def total_cost : ℝ :=
  membership_fee 80 [0.1, 0.12, 0.14, 0.15, 0.15, 0.15] [0, 0, 0, 0, 0.1, 0.05]

theorem total_cost_approx_636_38 : 
  ∃ ε > 0, abs (total_cost - 636.38) < ε :=
sorry

end NUMINAMATH_CALUDE_total_cost_approx_636_38_l1096_109604


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1096_109624

/-- The longest segment in a cylinder with radius 5 and height 12 is 2√61 -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) :
  Real.sqrt ((2 * r)^2 + h^2) = 2 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1096_109624


namespace NUMINAMATH_CALUDE_factorization_2x2_minus_8_factorization_ax2_minus_2ax_plus_a_l1096_109686

-- Factorization of 2x^2 - 8
theorem factorization_2x2_minus_8 (x : ℝ) :
  2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by sorry

-- Factorization of ax^2 - 2ax + a
theorem factorization_ax2_minus_2ax_plus_a (x a : ℝ) (ha : a ≠ 0) :
  a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_2x2_minus_8_factorization_ax2_minus_2ax_plus_a_l1096_109686


namespace NUMINAMATH_CALUDE_correct_operation_order_l1096_109674

-- Define operation levels
inductive OperationLevel
| FirstLevel
| SecondLevel

-- Define operations
inductive Operation
| Multiplication
| Division
| Subtraction

-- Define the level of each operation
def operationLevel : Operation → OperationLevel
| Operation.Multiplication => OperationLevel.SecondLevel
| Operation.Division => OperationLevel.SecondLevel
| Operation.Subtraction => OperationLevel.FirstLevel

-- Define the rule for operation order
def shouldPerformBefore (op1 op2 : Operation) : Prop :=
  operationLevel op1 = OperationLevel.SecondLevel ∧ 
  operationLevel op2 = OperationLevel.FirstLevel

-- Define the expression
def expression : List Operation :=
  [Operation.Multiplication, Operation.Subtraction, Operation.Division]

-- Theorem to prove
theorem correct_operation_order :
  shouldPerformBefore Operation.Multiplication Operation.Subtraction ∧
  shouldPerformBefore Operation.Division Operation.Subtraction ∧
  (¬ shouldPerformBefore Operation.Multiplication Operation.Division ∨
   ¬ shouldPerformBefore Operation.Division Operation.Multiplication) :=
by sorry

end NUMINAMATH_CALUDE_correct_operation_order_l1096_109674


namespace NUMINAMATH_CALUDE_tangent_and_unique_zero_l1096_109625

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (a * Real.log x) / x

def g (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := f a (f a x) - t

theorem tangent_and_unique_zero (a : ℝ) (h1 : a > 0) :
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ = f a x₀ ∧ x₀ - 2 * y₀ = 0 ∧ 
    (∀ x : ℝ, x > 0 → x - 2 * f a x ≥ 0) ∧
    (∀ x : ℝ, x > 0 → x - 2 * f a x = 0 → x = x₀)) →
  (∃! t : ℝ, ∃! x : ℝ, x > 0 ∧ g a t x = 0) →
  (∀ t : ℝ, (∃! x : ℝ, x > 0 ∧ g a t x = 0) → t = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_unique_zero_l1096_109625


namespace NUMINAMATH_CALUDE_growth_rate_equation_l1096_109618

/-- Represents the average annual growth rate of a company's capital -/
def x : ℝ := sorry

/-- The initial capital of the company in millions of yuan -/
def initial_capital : ℝ := 10

/-- The final capital of the company after two years in millions of yuan -/
def final_capital : ℝ := 14.4

/-- The number of years over which the growth occurred -/
def years : ℕ := 2

/-- Theorem stating that the equation 1000(1+x)^2 = 1440 correctly represents 
    the average annual growth rate of the company's capital -/
theorem growth_rate_equation : 1000 * (1 + x)^years = 1440 := by sorry

end NUMINAMATH_CALUDE_growth_rate_equation_l1096_109618


namespace NUMINAMATH_CALUDE_strip_length_is_four_l1096_109648

/-- The length of each square in the strip -/
def square_length : ℚ := 2/3

/-- The number of squares in the strip -/
def num_squares : ℕ := 6

/-- The total length of the strip -/
def strip_length : ℚ := square_length * num_squares

/-- Theorem: The strip composed of 6 squares, each with length 2/3, has a total length of 4 -/
theorem strip_length_is_four : strip_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_strip_length_is_four_l1096_109648


namespace NUMINAMATH_CALUDE_union_complement_equal_l1096_109685

open Set

def U : Finset ℕ := {0,1,2,4,6,8}
def M : Finset ℕ := {0,4,6}
def N : Finset ℕ := {0,1,6}

theorem union_complement_equal : M ∪ (U \ N) = {0,2,4,6,8} := by sorry

end NUMINAMATH_CALUDE_union_complement_equal_l1096_109685
