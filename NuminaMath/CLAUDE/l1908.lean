import Mathlib

namespace NUMINAMATH_CALUDE_fraction_equality_l1908_190888

theorem fraction_equality (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (1/x - 1/y) / (1/x + 1/y) = 1001 → (x + y) / (x - y) = -(1/1001) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1908_190888


namespace NUMINAMATH_CALUDE_correct_num_cups_l1908_190891

/-- The number of cups of coffee on the tray -/
def num_cups : ℕ := 5

/-- The initial volume of coffee in each cup (in ounces) -/
def initial_volume : ℝ := 8

/-- The shrink factor of the ray -/
def shrink_factor : ℝ := 0.5

/-- The total volume of coffee after shrinking (in ounces) -/
def final_total_volume : ℝ := 20

/-- Theorem stating that the number of cups is correct given the conditions -/
theorem correct_num_cups :
  initial_volume * shrink_factor * num_cups = final_total_volume :=
by sorry

end NUMINAMATH_CALUDE_correct_num_cups_l1908_190891


namespace NUMINAMATH_CALUDE_cubic_polynomial_value_at_5_l1908_190840

/-- A cubic polynomial satisfying specific conditions -/
def cubicPolynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a * x^3 + b * x^2 + c * x + d) ∧
  (p 1 = 1) ∧ (p 2 = 1/8) ∧ (p 3 = 1/27) ∧ (p 4 = 1/64)

/-- Theorem stating that a cubic polynomial satisfying the given conditions has p(5) = 0 -/
theorem cubic_polynomial_value_at_5 (p : ℝ → ℝ) (h : cubicPolynomial p) : p 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_value_at_5_l1908_190840


namespace NUMINAMATH_CALUDE_store_gross_profit_l1908_190896

theorem store_gross_profit (purchase_price : ℝ) (initial_markup_percent : ℝ) (price_decrease_percent : ℝ) : 
  purchase_price = 210 →
  initial_markup_percent = 25 →
  price_decrease_percent = 20 →
  let original_selling_price := purchase_price / (1 - initial_markup_percent / 100)
  let discounted_price := original_selling_price * (1 - price_decrease_percent / 100)
  let gross_profit := discounted_price - purchase_price
  gross_profit = 14 := by
sorry

end NUMINAMATH_CALUDE_store_gross_profit_l1908_190896


namespace NUMINAMATH_CALUDE_g_magnitude_l1908_190837

/-- A quadratic function that is even on a specific interval -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a

/-- The function g defined as a transformation of f -/
def g (a : ℝ) (x : ℝ) : ℝ := f a (x - 1)

/-- Theorem stating the relative magnitudes of g at specific points -/
theorem g_magnitude (a : ℝ) (h : ∀ x ∈ Set.Icc (-a) (a^2), f a x = f a (-x)) :
  g a (3/2) < g a 0 ∧ g a 0 < g a 3 := by
  sorry

end NUMINAMATH_CALUDE_g_magnitude_l1908_190837


namespace NUMINAMATH_CALUDE_overall_loss_percentage_l1908_190829

def purchase_prices : List ℝ := [600, 800, 1000, 1200, 1400]
def selling_prices : List ℝ := [550, 750, 1100, 1000, 1350]

theorem overall_loss_percentage :
  let total_cost_price := purchase_prices.sum
  let total_selling_price := selling_prices.sum
  let loss := total_cost_price - total_selling_price
  let loss_percentage := (loss / total_cost_price) * 100
  loss_percentage = 5 := by sorry

end NUMINAMATH_CALUDE_overall_loss_percentage_l1908_190829


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_zero_implies_zero_l1908_190881

theorem sqrt_sum_squares_zero_implies_zero (a b : ℂ) : 
  Real.sqrt (Complex.abs a ^ 2 + Complex.abs b ^ 2) = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_zero_implies_zero_l1908_190881


namespace NUMINAMATH_CALUDE_smallest_number_l1908_190832

theorem smallest_number (s : Finset ℚ) (hs : s = {-5, 1, -1, 0}) : 
  ∀ x ∈ s, -5 ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1908_190832


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l1908_190812

/-- The quadratic function f(x) = 15x^2 + 75x + 225 -/
def f (x : ℝ) : ℝ := 15 * x^2 + 75 * x + 225

/-- The constants a, b, and c in the form a(x+b)^2+c -/
def a : ℝ := 15
def b : ℝ := 2.5
def c : ℝ := 131.25

/-- The quadratic function g(x) in the form a(x+b)^2+c -/
def g (x : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum_of_constants :
  (∀ x, f x = g x) → a + b + c = 148.75 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l1908_190812


namespace NUMINAMATH_CALUDE_bus_row_capacity_l1908_190851

/-- Represents a bus with a given number of rows and total capacity -/
structure Bus where
  rows : ℕ
  capacity : ℕ

/-- Calculates the number of children each row can accommodate -/
def childrenPerRow (bus : Bus) : ℕ := bus.capacity / bus.rows

/-- Theorem: Given a bus with 9 rows and a capacity of 36 children,
    prove that each row can accommodate 4 children -/
theorem bus_row_capacity (bus : Bus) 
    (h_rows : bus.rows = 9) 
    (h_capacity : bus.capacity = 36) : 
    childrenPerRow bus = 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_row_capacity_l1908_190851


namespace NUMINAMATH_CALUDE_milk_replacement_theorem_l1908_190849

/-- The fraction of original substance remaining after one replacement operation -/
def replacement_fraction : ℝ := 0.8

/-- The number of replacement operations -/
def num_operations : ℕ := 3

/-- The percentage of original substance remaining after multiple replacement operations -/
def remaining_percentage (f : ℝ) (n : ℕ) : ℝ := 100 * f^n

theorem milk_replacement_theorem :
  remaining_percentage replacement_fraction num_operations = 51.2 := by
  sorry

end NUMINAMATH_CALUDE_milk_replacement_theorem_l1908_190849


namespace NUMINAMATH_CALUDE_a_plus_b_equals_seven_l1908_190838

theorem a_plus_b_equals_seven (a b : ℝ) (h : ∀ x, a * (x + b) = 3 * x + 12) : a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_seven_l1908_190838


namespace NUMINAMATH_CALUDE_no_common_points_condition_l1908_190806

theorem no_common_points_condition (d : ℝ) : 
  (∀ x y : ℝ × ℝ, (x.1 - y.1)^2 + (x.2 - y.2)^2 = d^2 → 
    ((x.1^2 + x.2^2 ≤ 4 ∧ y.1^2 + y.2^2 ≤ 9) ∨ 
     (x.1^2 + x.2^2 ≤ 9 ∧ y.1^2 + y.2^2 ≤ 4)) → 
    (x.1^2 + x.2^2 - 4) * (y.1^2 + y.2^2 - 9) > 0) ↔ 
  (0 ≤ d ∧ d < 1) ∨ d > 5 :=
sorry

end NUMINAMATH_CALUDE_no_common_points_condition_l1908_190806


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1908_190811

theorem system_of_equations_solution :
  ∃ (x y : ℝ), 2*x - 3*y = -5 ∧ 5*x - 2*y = 4 :=
by
  use 2, 3
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1908_190811


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_and_60_l1908_190805

theorem smallest_divisible_by_12_and_60 : Nat.lcm 12 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_and_60_l1908_190805


namespace NUMINAMATH_CALUDE_cube_edge_length_l1908_190875

-- Define the volume of the cube in milliliters
def cube_volume : ℝ := 729

-- Define the edge length of the cube in centimeters
def edge_length : ℝ := 9

-- Theorem: The edge length of a cube with volume 729 ml is 9 cm
theorem cube_edge_length : 
  edge_length ^ 3 * 1000 = cube_volume ∧ edge_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1908_190875


namespace NUMINAMATH_CALUDE_chip_consumption_theorem_l1908_190826

/-- Calculates the total number of bags of chips consumed in a week -/
def weekly_chip_consumption (breakfast_bags : ℕ) (lunch_bags : ℕ) (days_in_week : ℕ) : ℕ :=
  let dinner_bags := 2 * lunch_bags
  let daily_consumption := breakfast_bags + lunch_bags + dinner_bags
  daily_consumption * days_in_week

/-- Theorem stating that consuming 1 bag for breakfast, 2 for lunch, and doubling lunch for dinner
    every day for a week results in 49 bags consumed -/
theorem chip_consumption_theorem :
  weekly_chip_consumption 1 2 7 = 49 := by
  sorry

#eval weekly_chip_consumption 1 2 7

end NUMINAMATH_CALUDE_chip_consumption_theorem_l1908_190826


namespace NUMINAMATH_CALUDE_percent_of_percent_l1908_190822

theorem percent_of_percent (y : ℝ) (hy : y ≠ 0) :
  (0.6 * 0.3 * y) / y = 0.18 := by sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1908_190822


namespace NUMINAMATH_CALUDE_power_of_four_equality_l1908_190862

theorem power_of_four_equality (m n : ℕ+) (x y : ℝ) 
  (hx : 2^(m : ℕ) = x) (hy : 2^(2*n : ℕ) = y) : 
  4^((m : ℕ) + 2*(n : ℕ)) = x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_equality_l1908_190862


namespace NUMINAMATH_CALUDE_students_tree_planting_l1908_190894

/-- The number of apple trees planted by students -/
def apple_trees : ℕ := 47

/-- The number of orange trees planted by students -/
def orange_trees : ℕ := 27

/-- The total number of trees planted by students -/
def total_trees : ℕ := apple_trees + orange_trees

theorem students_tree_planting : total_trees = 74 := by
  sorry

end NUMINAMATH_CALUDE_students_tree_planting_l1908_190894


namespace NUMINAMATH_CALUDE_composition_difference_l1908_190834

/-- Given two functions f and g, prove that their composition difference
    f(g(x)) - g(f(x)) equals 6x^2 - 12x + 9 for all real x. -/
theorem composition_difference (x : ℝ) : 
  let f (x : ℝ) := 3 * x^2 - 6 * x + 1
  let g (x : ℝ) := 2 * x - 1
  f (g x) - g (f x) = 6 * x^2 - 12 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_composition_difference_l1908_190834


namespace NUMINAMATH_CALUDE_seventy_fifth_term_of_sequence_l1908_190836

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem seventy_fifth_term_of_sequence :
  arithmetic_sequence 2 4 75 = 298 := by sorry

end NUMINAMATH_CALUDE_seventy_fifth_term_of_sequence_l1908_190836


namespace NUMINAMATH_CALUDE_base_76_minus_b_multiple_of_17_l1908_190877

/-- The value of 528376415 in base 76 -/
def base_76_number : ℕ := 5 + 1*76 + 4*(76^2) + 6*(76^3) + 7*(76^4) + 3*(76^5) + 8*(76^6) + 2*(76^7) + 5*(76^8)

theorem base_76_minus_b_multiple_of_17 (b : ℤ) 
  (h1 : 0 ≤ b) (h2 : b ≤ 20) 
  (h3 : ∃ k : ℤ, base_76_number - b = 17 * k) :
  b = 0 ∨ b = 17 := by
sorry

end NUMINAMATH_CALUDE_base_76_minus_b_multiple_of_17_l1908_190877


namespace NUMINAMATH_CALUDE_smallest_value_3a_plus_2_l1908_190899

theorem smallest_value_3a_plus_2 (a : ℝ) (h : 4 * a^2 + 6 * a + 3 = 2) :
  ∃ (min : ℝ), min = 1/2 ∧ ∀ (x : ℝ), (∃ (b : ℝ), 4 * b^2 + 6 * b + 3 = 2 ∧ x = 3 * b + 2) → x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_3a_plus_2_l1908_190899


namespace NUMINAMATH_CALUDE_set_union_equality_l1908_190817

-- Define the sets M and N
def M : Set ℝ := {x | x^2 < 4*x}
def N : Set ℝ := {x | |x - 1| ≥ 3}

-- Define the union set
def unionSet : Set ℝ := {x | x ≤ -2 ∨ x > 0}

-- Theorem statement
theorem set_union_equality : M ∪ N = unionSet := by
  sorry

end NUMINAMATH_CALUDE_set_union_equality_l1908_190817


namespace NUMINAMATH_CALUDE_root_reciprocal_sum_l1908_190895

theorem root_reciprocal_sum (m n : ℝ) : 
  m^2 + 3*m - 1 = 0 → 
  n^2 + 3*n - 1 = 0 → 
  m ≠ n →
  1/m + 1/n = 3 := by
sorry

end NUMINAMATH_CALUDE_root_reciprocal_sum_l1908_190895


namespace NUMINAMATH_CALUDE_tan_sixty_minus_reciprocal_tan_thirty_equals_zero_l1908_190828

theorem tan_sixty_minus_reciprocal_tan_thirty_equals_zero :
  Real.tan (60 * π / 180) - (1 / Real.tan (30 * π / 180)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_sixty_minus_reciprocal_tan_thirty_equals_zero_l1908_190828


namespace NUMINAMATH_CALUDE_solve_for_y_l1908_190850

theorem solve_for_y (x : ℝ) (y : ℝ) 
  (h1 : x = 101) 
  (h2 : x^3 * y - 2 * x^2 * y + x * y = 101000) : 
  y = 1/10 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l1908_190850


namespace NUMINAMATH_CALUDE_sqrt_expressions_equality_l1908_190897

theorem sqrt_expressions_equality :
  (Real.sqrt 75 - Real.sqrt 54 + Real.sqrt 96 - Real.sqrt 108 = -Real.sqrt 3 + Real.sqrt 6) ∧
  (Real.sqrt 24 / Real.sqrt 3 + Real.sqrt (1/2) * Real.sqrt 18 - Real.sqrt 50 = 3 - 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expressions_equality_l1908_190897


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1908_190864

/-- The number of ordered pairs (m,n) of positive integers satisfying the given conditions -/
def valid_pairs : ℕ := 4

/-- Definition of a valid pair (m,n) -/
def is_valid_pair (m n : ℕ) : Prop :=
  m ≥ n ∧ n % 2 = 1 ∧ m^2 - n^2 = 120

/-- Theorem stating that there are exactly 4 valid pairs -/
theorem count_valid_pairs :
  (∃! (s : Finset (ℕ × ℕ)), s.card = valid_pairs ∧
    ∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_pair p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1908_190864


namespace NUMINAMATH_CALUDE_shadow_boundary_equation_l1908_190883

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The equation of the shadow boundary on the xy-plane -/
def shadowBoundary (s : Sphere) (lightSource : Point3D) : ℝ → ℝ := fun x ↦ -14

/-- Theorem: The shadow boundary of the given sphere with the given light source is y = -14 -/
theorem shadow_boundary_equation (s : Sphere) (lightSource : Point3D) :
  s.center = Point3D.mk 2 0 2 →
  s.radius = 2 →
  lightSource = Point3D.mk 2 (-2) 6 →
  ∀ x : ℝ, shadowBoundary s lightSource x = -14 := by
  sorry

#check shadow_boundary_equation

end NUMINAMATH_CALUDE_shadow_boundary_equation_l1908_190883


namespace NUMINAMATH_CALUDE_max_score_theorem_l1908_190807

/-- Represents a pile of stones -/
structure Pile :=
  (stones : ℕ)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)
  (score : ℕ)

/-- Defines a move in the game -/
def move (state : GameState) (i j : ℕ) : GameState :=
  sorry

/-- Checks if the game is over (all stones removed) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Calculates the maximum score achievable from a given state -/
def maxScore (state : GameState) : ℕ :=
  sorry

/-- The main theorem stating the maximum achievable score -/
theorem max_score_theorem :
  let initialState : GameState := ⟨List.replicate 100 ⟨400⟩, 0⟩
  maxScore initialState = 3920000 := by
  sorry

end NUMINAMATH_CALUDE_max_score_theorem_l1908_190807


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l1908_190825

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 25 years older than his son and the son's current age is 23. -/
theorem age_ratio_in_two_years (son_age : ℕ) (man_age : ℕ) : 
  son_age = 23 →
  man_age = son_age + 25 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l1908_190825


namespace NUMINAMATH_CALUDE_chessboard_circle_area_ratio_l1908_190821

/-- Represents a square chessboard -/
structure Chessboard where
  side_length : ℝ
  dimensions : ℕ × ℕ

/-- Represents a circle placed on the chessboard -/
structure PlacedCircle where
  radius : ℝ

/-- Calculates the sum of areas within the circle for intersected squares -/
def S₁ (board : Chessboard) (circle : PlacedCircle) : ℝ := sorry

/-- Calculates the sum of areas outside the circle for intersected squares -/
def S₂ (board : Chessboard) (circle : PlacedCircle) : ℝ := sorry

/-- The main theorem to be proved -/
theorem chessboard_circle_area_ratio
  (board : Chessboard)
  (circle : PlacedCircle)
  (h_board_side : board.side_length = 8)
  (h_board_dim : board.dimensions = (8, 8))
  (h_circle_radius : circle.radius = 4) :
  Int.floor (S₁ board circle / S₂ board circle) = 3 := by sorry

end NUMINAMATH_CALUDE_chessboard_circle_area_ratio_l1908_190821


namespace NUMINAMATH_CALUDE_smallest_multiple_of_8_no_repeated_digits_remainder_l1908_190802

/-- A function that checks if a natural number has no repeated digits -/
def hasNoRepeatedDigits (n : ℕ) : Prop := sorry

/-- The smallest multiple of 8 with no repeated digits -/
def M : ℕ := sorry

theorem smallest_multiple_of_8_no_repeated_digits_remainder :
  (M % 1000 = 120) ∧
  (∀ k : ℕ, k < M → (k % 8 = 0 → ¬hasNoRepeatedDigits k)) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_8_no_repeated_digits_remainder_l1908_190802


namespace NUMINAMATH_CALUDE_stamps_per_book_is_15_l1908_190884

/-- The number of stamps in each book of the second type -/
def stamps_per_book : ℕ := sorry

/-- The total number of stamps Ruel has -/
def total_stamps : ℕ := 130

/-- The number of books of the first type (10 stamps each) -/
def books_type1 : ℕ := 4

/-- The number of stamps in each book of the first type -/
def stamps_per_book_type1 : ℕ := 10

/-- The number of books of the second type -/
def books_type2 : ℕ := 6

theorem stamps_per_book_is_15 : 
  stamps_per_book = 15 ∧ 
  total_stamps = books_type1 * stamps_per_book_type1 + books_type2 * stamps_per_book :=
by sorry

end NUMINAMATH_CALUDE_stamps_per_book_is_15_l1908_190884


namespace NUMINAMATH_CALUDE_minimal_blue_chips_l1908_190823

theorem minimal_blue_chips (r g b : ℕ) : 
  b ≥ r / 3 →
  b ≤ g / 4 →
  r + g ≥ 75 →
  (∀ b' : ℕ, b' ≥ r / 3 → b' ≤ g / 4 → b' ≥ b) →
  b = 11 := by
  sorry

end NUMINAMATH_CALUDE_minimal_blue_chips_l1908_190823


namespace NUMINAMATH_CALUDE_train_distance_theorem_l1908_190841

/-- Calculates the distance a train can travel given its coal efficiency and remaining coal. -/
def train_distance (miles_per_unit : ℚ) (pounds_per_unit : ℚ) (coal_remaining : ℚ) : ℚ :=
  (coal_remaining / pounds_per_unit) * miles_per_unit

/-- Proves that a train with given efficiency and coal amount can travel the calculated distance. -/
theorem train_distance_theorem (miles_per_unit : ℚ) (pounds_per_unit : ℚ) (coal_remaining : ℚ) :
  miles_per_unit = 5 → pounds_per_unit = 2 → coal_remaining = 160 →
  train_distance miles_per_unit pounds_per_unit coal_remaining = 400 := by
  sorry

#check train_distance_theorem

end NUMINAMATH_CALUDE_train_distance_theorem_l1908_190841


namespace NUMINAMATH_CALUDE_library_books_count_library_books_count_proof_l1908_190800

theorem library_books_count : ℕ → Prop :=
  fun N =>
    let initial_issued := N / 17
    let transferred := 2000
    let new_issued := initial_issued + transferred
    (initial_issued = (N - initial_issued) / 16) ∧
    (new_issued = (N - new_issued) / 15) →
    N = 544000

-- The proof goes here
theorem library_books_count_proof : library_books_count 544000 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_library_books_count_proof_l1908_190800


namespace NUMINAMATH_CALUDE_supremum_of_expression_l1908_190861

theorem supremum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  -1 / (2 * a) - 2 / b ≤ -9 / 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ -1 / (2 * a₀) - 2 / b₀ = -9 / 2 :=
sorry

end NUMINAMATH_CALUDE_supremum_of_expression_l1908_190861


namespace NUMINAMATH_CALUDE_square_of_difference_l1908_190871

theorem square_of_difference (x : ℝ) : (x - 1)^2 = x^2 + 1 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l1908_190871


namespace NUMINAMATH_CALUDE_cube_neg_iff_neg_l1908_190833

theorem cube_neg_iff_neg (x : ℝ) : x^3 < 0 ↔ x < 0 := by sorry

end NUMINAMATH_CALUDE_cube_neg_iff_neg_l1908_190833


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_squared_plus_one_positive_l1908_190843

theorem negation_of_forall_positive (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_squared_plus_one_positive :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_squared_plus_one_positive_l1908_190843


namespace NUMINAMATH_CALUDE_tangent_segment_length_is_3cm_l1908_190863

/-- An isosceles triangle with a base of 12 cm and a height of 8 cm -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  isIsosceles : base = 12 ∧ height = 8

/-- A circle inscribed in the isosceles triangle -/
structure InscribedCircle (t : IsoscelesTriangle) where
  center : ℝ × ℝ
  radius : ℝ
  isInscribed : True  -- This is a placeholder for the inscribed circle condition

/-- A tangent line parallel to the base of the triangle -/
structure ParallelTangent (t : IsoscelesTriangle) (c : InscribedCircle t) where
  point : ℝ × ℝ
  isParallel : True  -- This is a placeholder for the parallel condition
  isTangent : True   -- This is a placeholder for the tangent condition

/-- The length of the segment of the tangent line between the sides of the triangle -/
def tangentSegmentLength (t : IsoscelesTriangle) (c : InscribedCircle t) (l : ParallelTangent t c) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The main theorem stating that the length of the tangent segment is 3 cm -/
theorem tangent_segment_length_is_3cm (t : IsoscelesTriangle) (c : InscribedCircle t) (l : ParallelTangent t c) :
  tangentSegmentLength t c l = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_segment_length_is_3cm_l1908_190863


namespace NUMINAMATH_CALUDE_function_extrema_l1908_190839

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a*x - (a+1) * Real.log x

theorem function_extrema (a : ℝ) (h1 : a < -1) :
  (∀ x : ℝ, x > 0 → (deriv (f a)) 2 = 0) →
  (a = -3 ∧ 
   (∀ x : ℝ, x > 0 → f a x ≤ f a 1) ∧
   (∀ x : ℝ, x > 0 → f a x ≥ f a 2) ∧
   f a 1 = -5/2 ∧
   f a 2 = -4 + 2 * Real.log 2) := by
  sorry

end

end NUMINAMATH_CALUDE_function_extrema_l1908_190839


namespace NUMINAMATH_CALUDE_rachel_milk_consumption_l1908_190876

theorem rachel_milk_consumption (don_milk : ℚ) (rachel_fraction : ℚ) :
  don_milk = 3 / 7 →
  rachel_fraction = 4 / 5 →
  rachel_fraction * don_milk = 12 / 35 :=
by sorry

end NUMINAMATH_CALUDE_rachel_milk_consumption_l1908_190876


namespace NUMINAMATH_CALUDE_symmetric_points_on_parabola_l1908_190853

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define symmetry about the line x + y = m
def symmetric_about_line (x₁ y₁ x₂ y₂ m : ℝ) : Prop :=
  (x₁ + y₁ + x₂ + y₂) / 2 = m

-- Main theorem
theorem symmetric_points_on_parabola (x₁ y₁ x₂ y₂ m : ℝ) :
  is_on_parabola x₁ y₁ →
  is_on_parabola x₂ y₂ →
  symmetric_about_line x₁ y₁ x₂ y₂ m →
  y₁ * y₂ = -1/2 →
  m = 9/4 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_on_parabola_l1908_190853


namespace NUMINAMATH_CALUDE_num_divisors_30_is_8_l1908_190844

/-- The number of positive divisors of 30 -/
def num_divisors_30 : ℕ := sorry

/-- Theorem stating that the number of positive divisors of 30 is 8 -/
theorem num_divisors_30_is_8 : num_divisors_30 = 8 := by sorry

end NUMINAMATH_CALUDE_num_divisors_30_is_8_l1908_190844


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1908_190865

/-- Given a rectangle with area 540 square centimeters, if its length is decreased by 15% and
    its width is increased by 20%, the new area will be 550.8 square centimeters. -/
theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 540) : 
  (L * 0.85) * (W * 1.2) = 550.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1908_190865


namespace NUMINAMATH_CALUDE_equations_hold_l1908_190886

structure BinaryOp (S : Type) where
  op : S → S → S

def satisfies_condition (S : Type) (star : BinaryOp S) :=
  ∀ (a b : S), star.op a (star.op b a) = b

theorem equations_hold (S : Type) [Inhabited S] (star : BinaryOp S) 
  (h : satisfies_condition S star) :
  (∀ (a b : S), (star.op (star.op a (star.op b a)) (star.op a b)) = a) ∧
  (∀ (a b : S), star.op b (star.op a b) = a) ∧
  (∀ (a b : S), star.op (star.op a b) (star.op b (star.op a b)) = b) :=
by sorry

end NUMINAMATH_CALUDE_equations_hold_l1908_190886


namespace NUMINAMATH_CALUDE_two_statements_true_l1908_190820

open Real

-- Define the function f
noncomputable def f (x : ℝ) := 2 * sin x * cos (abs x)

-- Define the sequence a_n
def a (n : ℕ) (k : ℝ) := n^2 + k*n + 2

theorem two_statements_true :
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∃ w : ℝ, w > 0 ∧ w = 1 ∧ ∀ x, f (x + w) = f x) ∧
  (∀ k, (∀ n : ℕ, n > 0 → a (n+1) k > a n k) → k > -3) :=
by sorry

end NUMINAMATH_CALUDE_two_statements_true_l1908_190820


namespace NUMINAMATH_CALUDE_hospital_bill_ambulance_cost_l1908_190874

/-- Given a hospital bill with specified percentages for various services and fixed costs,
    calculate the cost of the ambulance ride. -/
theorem hospital_bill_ambulance_cost 
  (total_bill : ℝ) 
  (medication_percent : ℝ) 
  (imaging_percent : ℝ) 
  (surgical_percent : ℝ) 
  (overnight_percent : ℝ) 
  (food_cost : ℝ) 
  (consultation_cost : ℝ) 
  (h1 : total_bill = 12000)
  (h2 : medication_percent = 0.40)
  (h3 : imaging_percent = 0.15)
  (h4 : surgical_percent = 0.20)
  (h5 : overnight_percent = 0.25)
  (h6 : food_cost = 300)
  (h7 : consultation_cost = 80)
  (h8 : medication_percent + imaging_percent + surgical_percent + overnight_percent = 1) :
  total_bill - (food_cost + consultation_cost) = 11620 := by
  sorry


end NUMINAMATH_CALUDE_hospital_bill_ambulance_cost_l1908_190874


namespace NUMINAMATH_CALUDE_distance_traveled_l1908_190854

theorem distance_traveled (speed : ℝ) (time : ℝ) : 
  speed = 57 → time = 30 / 3600 → speed * time = 0.475 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l1908_190854


namespace NUMINAMATH_CALUDE_celine_change_l1908_190860

/-- The price of a laptop in dollars -/
def laptop_price : ℕ := 600

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 400

/-- The number of laptops Celine buys -/
def laptops_bought : ℕ := 2

/-- The number of smartphones Celine buys -/
def smartphones_bought : ℕ := 4

/-- The amount of money Celine has in dollars -/
def money_available : ℕ := 3000

/-- The change Celine receives after her purchase -/
theorem celine_change : 
  money_available - (laptop_price * laptops_bought + smartphone_price * smartphones_bought) = 200 := by
  sorry

end NUMINAMATH_CALUDE_celine_change_l1908_190860


namespace NUMINAMATH_CALUDE_max_product_constraint_l1908_190804

theorem max_product_constraint (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 2 → (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a * b ≥ x * y) → a * b = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l1908_190804


namespace NUMINAMATH_CALUDE_radius_equals_leg_length_l1908_190887

/-- A circle tangent to coordinate axes and hypotenuse of a 45-45-90 triangle --/
structure TangentCircle where
  /-- Radius of the circle --/
  r : ℝ
  /-- Length of the leg of the 45-45-90 triangle --/
  a : ℝ
  /-- The circle is tangent to the coordinate axes --/
  tangent_to_axes : r > 0
  /-- The circle is tangent to the hypotenuse of the 45-45-90 triangle --/
  tangent_to_hypotenuse : r ≤ a * Real.sqrt 2

/-- The radius of the tangent circle is equal to the leg length of the triangle --/
theorem radius_equals_leg_length (c : TangentCircle) : c.r = c.a := by
  sorry


end NUMINAMATH_CALUDE_radius_equals_leg_length_l1908_190887


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l1908_190857

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l1908_190857


namespace NUMINAMATH_CALUDE_max_xy_value_l1908_190882

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 6) :
  x*y ≤ 3/2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 6 ∧ x₀*y₀ = 3/2 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l1908_190882


namespace NUMINAMATH_CALUDE_shortest_distance_dasha_vasya_l1908_190868

-- Define the friends as vertices in a graph
inductive Friend : Type
| Asya : Friend
| Galia : Friend
| Borya : Friend
| Dasha : Friend
| Vasya : Friend

-- Define the distance function between friends
def distance : Friend → Friend → ℕ
| Friend.Asya, Friend.Galia => 12
| Friend.Galia, Friend.Asya => 12
| Friend.Galia, Friend.Borya => 10
| Friend.Borya, Friend.Galia => 10
| Friend.Asya, Friend.Borya => 8
| Friend.Borya, Friend.Asya => 8
| Friend.Dasha, Friend.Galia => 15
| Friend.Galia, Friend.Dasha => 15
| Friend.Vasya, Friend.Galia => 17
| Friend.Galia, Friend.Vasya => 17
| _, _ => 0  -- Default case for undefined distances

-- Define the shortest path function
def shortest_path (a b : Friend) : ℕ := sorry

-- Theorem statement
theorem shortest_distance_dasha_vasya :
  shortest_path Friend.Dasha Friend.Vasya = 18 := by sorry

end NUMINAMATH_CALUDE_shortest_distance_dasha_vasya_l1908_190868


namespace NUMINAMATH_CALUDE_number_in_scientific_notation_l1908_190808

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 21600

/-- Theorem stating that 21,600 in scientific notation is 2.16 × 10^4 -/
theorem number_in_scientific_notation :
  ∃ (sn : ScientificNotation), (sn.coefficient * (10 : ℝ) ^ sn.exponent = number) ∧
    (sn.coefficient = 2.16 ∧ sn.exponent = 4) :=
sorry

end NUMINAMATH_CALUDE_number_in_scientific_notation_l1908_190808


namespace NUMINAMATH_CALUDE_choose_three_from_thirteen_l1908_190814

theorem choose_three_from_thirteen : Nat.choose 13 3 = 286 := by sorry

end NUMINAMATH_CALUDE_choose_three_from_thirteen_l1908_190814


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1908_190803

/-- 
Given an exam where:
1. 40% of the maximum marks are required to pass
2. A student got 40 marks
3. The student failed by 40 marks
Prove that the maximum marks for the exam are 200.
-/
theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (pass_percentage : ℚ) (student_marks : ℕ) (fail_margin : ℕ),
    pass_percentage = 40 / 100 →
    student_marks = 40 →
    fail_margin = 40 →
    (pass_percentage * max_marks : ℚ) = student_marks + fail_margin →
    max_marks = 200 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1908_190803


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l1908_190846

-- Define variables
variable (a b m : ℝ)

-- Theorem for the first factorization
theorem factorization_1 : a^2 * (a - b) - 4 * b^2 * (a - b) = (a - b) * (a - 2*b) * (a + 2*b) := by
  sorry

-- Theorem for the second factorization
theorem factorization_2 : m^2 - 6*m + 9 = (m - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l1908_190846


namespace NUMINAMATH_CALUDE_pool_capacity_pool_capacity_is_2000_liters_l1908_190867

theorem pool_capacity (water_loss_per_jump : ℝ) (cleaning_threshold : ℝ) (jumps_before_cleaning : ℕ) : ℝ :=
  let total_water_loss := water_loss_per_jump * jumps_before_cleaning
  let water_loss_percentage := 1 - cleaning_threshold
  total_water_loss / water_loss_percentage

#check pool_capacity 0.4 0.8 1000 = 2000

theorem pool_capacity_is_2000_liters :
  pool_capacity 0.4 0.8 1000 = 2000 := by sorry

end NUMINAMATH_CALUDE_pool_capacity_pool_capacity_is_2000_liters_l1908_190867


namespace NUMINAMATH_CALUDE_binomial_11_10_l1908_190824

theorem binomial_11_10 : Nat.choose 11 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_binomial_11_10_l1908_190824


namespace NUMINAMATH_CALUDE_luis_task_completion_l1908_190859

-- Define the start time and end time of the third task
def start_time : Nat := 9 * 60  -- 9:00 AM in minutes
def end_third_task : Nat := 12 * 60 + 30  -- 12:30 PM in minutes

-- Define the number of tasks
def num_tasks : Nat := 4

-- Define the theorem
theorem luis_task_completion :
  ∀ (task_duration : Nat),
  (end_third_task - start_time = 3 * task_duration) →
  (start_time + num_tasks * task_duration = 13 * 60 + 40) :=
by
  sorry


end NUMINAMATH_CALUDE_luis_task_completion_l1908_190859


namespace NUMINAMATH_CALUDE_website_development_time_ratio_l1908_190845

/-- The time Katherine takes to develop a website -/
def katherine_time : ℕ := 20

/-- The number of websites Naomi developed -/
def naomi_websites : ℕ := 30

/-- The total time Naomi took to develop all websites -/
def naomi_total_time : ℕ := 750

/-- The ratio of the time Naomi takes to the time Katherine takes to develop a website -/
def time_ratio : ℚ := (naomi_total_time / naomi_websites : ℚ) / katherine_time

theorem website_development_time_ratio :
  time_ratio = 5/4 := by sorry

end NUMINAMATH_CALUDE_website_development_time_ratio_l1908_190845


namespace NUMINAMATH_CALUDE_log_calculation_l1908_190809

-- Define the common logarithm (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_calculation : log10 5 * log10 20 + (log10 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_calculation_l1908_190809


namespace NUMINAMATH_CALUDE_triangle_inequality_l1908_190848

theorem triangle_inequality (A B C : Real) (h_triangle : A + B + C = π) :
  Real.tan (B / 2) * Real.tan (C / 2)^2 ≥ 4 * Real.tan (A / 2) * (Real.tan (A / 2) * Real.tan (C / 2) - 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1908_190848


namespace NUMINAMATH_CALUDE_rhombus_area_l1908_190818

/-- The area of a rhombus with side length √125 and diagonal difference 8 is 60.5 -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 125 →
  diag_diff = 8 →
  area = (side^2 * Real.sqrt (4 - (diag_diff / side)^2)) / 2 →
  area = 60.5 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l1908_190818


namespace NUMINAMATH_CALUDE_correct_amount_given_to_john_l1908_190827

/-- The amount given to John after one month -/
def amount_given_to_john (held_commission : ℕ) (advance_fees : ℕ) (incentive : ℕ) : ℕ :=
  (held_commission - advance_fees) + incentive

/-- Theorem stating the correct amount given to John -/
theorem correct_amount_given_to_john :
  amount_given_to_john 25000 8280 1780 = 18500 := by
  sorry

end NUMINAMATH_CALUDE_correct_amount_given_to_john_l1908_190827


namespace NUMINAMATH_CALUDE_cricket_innings_count_l1908_190873

-- Define the problem parameters
def current_average : ℝ := 32
def runs_next_innings : ℝ := 116
def average_increase : ℝ := 4

-- Theorem statement
theorem cricket_innings_count :
  ∀ n : ℝ,
  (n > 0) →
  (current_average * n + runs_next_innings) / (n + 1) = current_average + average_increase →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_cricket_innings_count_l1908_190873


namespace NUMINAMATH_CALUDE_third_shiny_penny_probability_l1908_190801

def total_pennies : ℕ := 9
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := 5

def probability_more_than_five_draws : ℚ :=
  37 / 63

theorem third_shiny_penny_probability :
  probability_more_than_five_draws =
    (Nat.choose 5 2 * Nat.choose 4 1 +
     Nat.choose 5 1 * Nat.choose 4 2 +
     Nat.choose 5 0 * Nat.choose 4 3) /
    Nat.choose total_pennies shiny_pennies :=
by sorry

end NUMINAMATH_CALUDE_third_shiny_penny_probability_l1908_190801


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1908_190835

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (4 * x - 3 * y = -14) ∧ (5 * x + 3 * y = -12) ∧ (x = -26/9) ∧ (y = -22/27) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1908_190835


namespace NUMINAMATH_CALUDE_batsman_average_l1908_190898

/-- Calculates the new average score after an additional inning -/
def new_average (prev_avg : ℚ) (prev_innings : ℕ) (new_score : ℕ) : ℚ :=
  (prev_avg * prev_innings + new_score) / (prev_innings + 1)

/-- Theorem: Given the conditions, the batsman's new average is 18 -/
theorem batsman_average : new_average 19 17 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l1908_190898


namespace NUMINAMATH_CALUDE_green_face_probability_l1908_190872

/-- A structure representing a die with colored faces -/
structure ColoredDie where
  sides : ℕ
  green_faces : ℕ
  red_faces : ℕ
  blue_faces : ℕ
  yellow_faces : ℕ
  total_eq_sum : sides = green_faces + red_faces + blue_faces + yellow_faces

/-- The probability of rolling a specific color on a colored die -/
def roll_probability (d : ColoredDie) (color_faces : ℕ) : ℚ :=
  color_faces / d.sides

/-- Theorem: The probability of rolling a green face on our specific 12-sided die is 1/12 -/
theorem green_face_probability :
  let d : ColoredDie := {
    sides := 12,
    green_faces := 1,
    red_faces := 5,
    blue_faces := 4,
    yellow_faces := 2,
    total_eq_sum := by simp
  }
  roll_probability d d.green_faces = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_green_face_probability_l1908_190872


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1908_190878

theorem arithmetic_mean_problem (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 4 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1908_190878


namespace NUMINAMATH_CALUDE_population_decrease_percentage_l1908_190892

/-- Calculates the percentage of population that moved away after a growth spurt -/
def percentage_moved_away (initial_population : ℕ) (growth_rate : ℚ) (final_population : ℕ) : ℚ :=
  let population_after_growth := initial_population * (1 + growth_rate)
  let people_moved_away := population_after_growth - final_population
  people_moved_away / population_after_growth

theorem population_decrease_percentage 
  (initial_population : ℕ) 
  (growth_rate : ℚ) 
  (final_population : ℕ) 
  (h1 : initial_population = 684) 
  (h2 : growth_rate = 1/4) 
  (h3 : final_population = 513) : 
  percentage_moved_away initial_population growth_rate final_population = 2/5 := by
  sorry

#eval percentage_moved_away 684 (1/4) 513

end NUMINAMATH_CALUDE_population_decrease_percentage_l1908_190892


namespace NUMINAMATH_CALUDE_nimathur_prime_l1908_190813

/-- Definition of a-nimathur -/
def is_a_nimathur (a b : ℕ) : Prop :=
  a ≥ 1 ∧ b ≥ 1 ∧ ∀ n : ℕ, n ≥ b / a →
    (a * n + 1) ∣ (Nat.choose (a * n) b - 1)

/-- Main theorem -/
theorem nimathur_prime (a b : ℕ) :
  is_a_nimathur a b ∧ ¬is_a_nimathur a (b + 2) → Nat.Prime (b + 1) :=
by sorry

end NUMINAMATH_CALUDE_nimathur_prime_l1908_190813


namespace NUMINAMATH_CALUDE_prob_queens_or_aces_l1908_190842

def standard_deck : ℕ := 52
def num_aces : ℕ := 4
def num_queens : ℕ := 4

def prob_two_queens : ℚ := (num_queens * (num_queens - 1)) / (standard_deck * (standard_deck - 1))
def prob_one_ace : ℚ := 2 * (num_aces * (standard_deck - num_aces)) / (standard_deck * (standard_deck - 1))
def prob_two_aces : ℚ := (num_aces * (num_aces - 1)) / (standard_deck * (standard_deck - 1))

theorem prob_queens_or_aces :
  prob_two_queens + prob_one_ace + prob_two_aces = 2 / 13 :=
sorry

end NUMINAMATH_CALUDE_prob_queens_or_aces_l1908_190842


namespace NUMINAMATH_CALUDE_star_one_two_l1908_190885

-- Define the * operation
def star (a b : ℝ) : ℝ := a + b + a * b

-- State the theorem
theorem star_one_two (a : ℝ) : star (star a 1) 2 = 6 * a + 5 := by
  sorry

end NUMINAMATH_CALUDE_star_one_two_l1908_190885


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l1908_190880

-- Define the hyperbola equation
def hyperbola_eq (x y a : ℝ) : Prop := x^2 / (a + 3) - y^2 / 3 = 1

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_a_value :
  ∃ (a : ℝ), (∀ (x y : ℝ), hyperbola_eq x y a) ∧ 
  (eccentricity = 2) → a = -2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l1908_190880


namespace NUMINAMATH_CALUDE_triangle_height_to_bc_l1908_190858

/-- In a triangle ABC, given side lengths and an angle, prove the height to a specific side. -/
theorem triangle_height_to_bc (a b c h : ℝ) (B : ℝ) : 
  a = 2 → 
  b = Real.sqrt 7 → 
  B = π / 3 →
  c^2 = a^2 + b^2 - 2*a*c*(Real.cos B) →
  h = (a * c * Real.sin B) / a →
  h = 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_to_bc_l1908_190858


namespace NUMINAMATH_CALUDE_point_on_line_l1908_190869

/-- Given two points (m, n) and (m + p, n + 21) on the line x = (y / 7) - (2 / 5),
    prove that p = 3 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 7 - 2 / 5) ∧ (m + p = (n + 21) / 7 - 2 / 5) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1908_190869


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1908_190819

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4^7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1908_190819


namespace NUMINAMATH_CALUDE_sandwich_counts_l1908_190879

def is_valid_sandwich_count (s : ℕ) : Prop :=
  ∃ (c : ℕ), 
    s + c = 7 ∧ 
    (100 * s + 75 * c) % 100 = 0

theorem sandwich_counts : 
  ∀ s : ℕ, is_valid_sandwich_count s ↔ (s = 3 ∨ s = 7) :=
by sorry

end NUMINAMATH_CALUDE_sandwich_counts_l1908_190879


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1908_190816

/-- Given a line y = kx + b that is parallel to y = 2x - 3 and passes through (1, -5),
    prove that its equation is y = 2x - 7 -/
theorem parallel_line_through_point (k b : ℝ) : 
  (∀ x y, y = k * x + b ↔ y = 2 * x - 3) →  -- parallelism condition
  (-5 : ℝ) = k * 1 + b →                   -- point condition
  ∀ x y, y = k * x + b ↔ y = 2 * x - 7 :=   -- conclusion
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1908_190816


namespace NUMINAMATH_CALUDE_tournament_games_played_l1908_190893

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  num_teams : ℕ
  no_ties : Bool

/-- The number of games played in a single-elimination tournament -/
def games_played (t : SingleEliminationTournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 24 teams and no ties,
    the number of games played to declare a winner is 23 -/
theorem tournament_games_played :
  ∀ (t : SingleEliminationTournament),
    t.num_teams = 24 → t.no_ties = true →
    games_played t = 23 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_played_l1908_190893


namespace NUMINAMATH_CALUDE_square_area_l1908_190890

/-- The area of a square with specific properties -/
theorem square_area (s : ℝ) (h1 : s > 0) 
  (h2 : 3.6 = s * (42/85 + 2/5)) : s^2 = 1156 := by
  sorry

#check square_area

end NUMINAMATH_CALUDE_square_area_l1908_190890


namespace NUMINAMATH_CALUDE_max_volume_is_three_l1908_190855

/-- Represents a rectangular solid with given constraints -/
structure RectangularSolid where
  width : ℝ
  length : ℝ
  height : ℝ
  sum_of_edges : width * 4 + length * 4 + height * 4 = 18
  length_width_ratio : length = 2 * width

/-- The volume of a rectangular solid -/
def volume (r : RectangularSolid) : ℝ := r.width * r.length * r.height

/-- Theorem stating that the maximum volume of the rectangular solid is 3 -/
theorem max_volume_is_three :
  ∃ (r : RectangularSolid), volume r = 3 ∧ ∀ (s : RectangularSolid), volume s ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_is_three_l1908_190855


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1908_190810

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) : 
  z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1908_190810


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l1908_190830

/-- The perimeter of a semicircle with radius 20 is approximately 102.83 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 20
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 102.83) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l1908_190830


namespace NUMINAMATH_CALUDE_intersection_and_union_of_sets_l1908_190866

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_and_union_of_sets :
  ∃ (a : ℝ), (A a ∩ B a = {9}) ∧ (a = -3) ∧ (A a ∪ B a = {-8, -7, -4, 4, 9}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_sets_l1908_190866


namespace NUMINAMATH_CALUDE_roots_of_cosine_equation_l1908_190852

theorem roots_of_cosine_equation :
  let f (t : ℝ) := 32 * t^5 - 40 * t^3 + 10 * t - Real.sqrt 3
  (f (Real.cos (6 * π / 180)) = 0) →
  (f (Real.cos (66 * π / 180)) = 0) ∧
  (f (Real.cos (78 * π / 180)) = 0) ∧
  (f (Real.cos (138 * π / 180)) = 0) ∧
  (f (Real.cos (150 * π / 180)) = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_cosine_equation_l1908_190852


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1908_190856

theorem inequality_solution_set (a : ℝ) (h : a < -1) :
  {x : ℝ | (a * x - 1) / (x + 1) < 0} = {x : ℝ | x < -1 ∨ x > 1/a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1908_190856


namespace NUMINAMATH_CALUDE_total_amount_proof_l1908_190889

/-- Prove that the total amount of money is 3000 given the specified conditions -/
theorem total_amount_proof (T P1 P2 : ℝ) : 
  T = P1 + P2 →
  P1 = 300 →
  0.03 * P1 + 0.05 * P2 = 144 →
  T = 3000 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_proof_l1908_190889


namespace NUMINAMATH_CALUDE_largest_minimum_uniform_output_l1908_190870

def black_box (n : ℕ) : ℕ :=
  if n % 2 = 1 then 4 * n + 1 else n / 2

def series_black_box (n : ℕ) : ℕ :=
  black_box (black_box (black_box n))

def is_valid_input (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < n ∧ b < n ∧ c < n ∧
  series_black_box a = series_black_box b ∧
  series_black_box b = series_black_box c ∧
  series_black_box c = series_black_box n

theorem largest_minimum_uniform_output :
  ∃ (n : ℕ), is_valid_input n ∧
  (∀ m, is_valid_input m → m ≤ n) ∧
  (∀ k, k < n → ¬is_valid_input k) ∧
  n = 680 :=
sorry

end NUMINAMATH_CALUDE_largest_minimum_uniform_output_l1908_190870


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1908_190831

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 2*x + 1) * (x^2 + 8*x + 15) + (x^2 + 6*x + 5) = (x + 1) * (x + 5) * (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1908_190831


namespace NUMINAMATH_CALUDE_optimal_selection_l1908_190847

/-- Represents a 5x5 matrix of integers -/
def Matrix5x5 : Type := Fin 5 → Fin 5 → ℤ

/-- The given matrix -/
def givenMatrix : Matrix5x5 :=
  λ i j => match i, j with
  | ⟨0, _⟩, ⟨0, _⟩ => 11 | ⟨0, _⟩, ⟨1, _⟩ => 17 | ⟨0, _⟩, ⟨2, _⟩ => 25 | ⟨0, _⟩, ⟨3, _⟩ => 19 | ⟨0, _⟩, ⟨4, _⟩ => 16
  | ⟨1, _⟩, ⟨0, _⟩ => 24 | ⟨1, _⟩, ⟨1, _⟩ => 10 | ⟨1, _⟩, ⟨2, _⟩ => 13 | ⟨1, _⟩, ⟨3, _⟩ => 15 | ⟨1, _⟩, ⟨4, _⟩ => 3
  | ⟨2, _⟩, ⟨0, _⟩ => 12 | ⟨2, _⟩, ⟨1, _⟩ => 5  | ⟨2, _⟩, ⟨2, _⟩ => 14 | ⟨2, _⟩, ⟨3, _⟩ => 2  | ⟨2, _⟩, ⟨4, _⟩ => 18
  | ⟨3, _⟩, ⟨0, _⟩ => 23 | ⟨3, _⟩, ⟨1, _⟩ => 4  | ⟨3, _⟩, ⟨2, _⟩ => 1  | ⟨3, _⟩, ⟨3, _⟩ => 8  | ⟨3, _⟩, ⟨4, _⟩ => 22
  | ⟨4, _⟩, ⟨0, _⟩ => 6  | ⟨4, _⟩, ⟨1, _⟩ => 20 | ⟨4, _⟩, ⟨2, _⟩ => 7  | ⟨4, _⟩, ⟨3, _⟩ => 21 | ⟨4, _⟩, ⟨4, _⟩ => 9
  | _, _ => 0

/-- A selection of 5 elements from the matrix -/
def Selection : Type := Fin 5 → (Fin 5 × Fin 5)

/-- Check if a selection is valid (no two elements in same row or column) -/
def isValidSelection (s : Selection) : Prop :=
  ∀ i j, i ≠ j → (s i).1 ≠ (s j).1 ∧ (s i).2 ≠ (s j).2

/-- The claimed optimal selection -/
def claimedOptimalSelection : Selection :=
  λ i => match i with
  | ⟨0, _⟩ => (⟨0, by norm_num⟩, ⟨2, by norm_num⟩)  -- 25
  | ⟨1, _⟩ => (⟨4, by norm_num⟩, ⟨1, by norm_num⟩)  -- 20
  | ⟨2, _⟩ => (⟨3, by norm_num⟩, ⟨0, by norm_num⟩)  -- 23
  | ⟨3, _⟩ => (⟨2, by norm_num⟩, ⟨4, by norm_num⟩)  -- 18
  | ⟨4, _⟩ => (⟨1, by norm_num⟩, ⟨3, by norm_num⟩)  -- 15

/-- The theorem to prove -/
theorem optimal_selection :
  isValidSelection claimedOptimalSelection ∧
  (∀ s : Selection, isValidSelection s →
    (∃ i, givenMatrix (s i).1 (s i).2 ≤ givenMatrix (claimedOptimalSelection 4).1 (claimedOptimalSelection 4).2)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_selection_l1908_190847


namespace NUMINAMATH_CALUDE_passion_fruit_crates_l1908_190815

theorem passion_fruit_crates (total_crates grapes_crates mangoes_crates : ℕ) 
  (h1 : total_crates = 50)
  (h2 : grapes_crates = 13)
  (h3 : mangoes_crates = 20) :
  total_crates - (grapes_crates + mangoes_crates) = 17 := by
  sorry

end NUMINAMATH_CALUDE_passion_fruit_crates_l1908_190815
