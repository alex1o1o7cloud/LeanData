import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3535_353503

theorem imaginary_part_of_z (z : ℂ) : z = Complex.I * (-1 + 2 * Complex.I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3535_353503


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l3535_353508

def base : ℕ := 3 + 4
def exponent : ℕ := 21

def last_two_digits (n : ℕ) : ℕ := n % 100

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_of_power :
  tens_digit (last_two_digits (base ^ exponent)) + ones_digit (last_two_digits (base ^ exponent)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l3535_353508


namespace NUMINAMATH_CALUDE_distance_between_trees_l3535_353522

/-- Given a yard of length 441 meters with 22 equally spaced trees (including one at each end),
    the distance between two consecutive trees is 21 meters. -/
theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (distance : ℕ) :
  yard_length = 441 →
  num_trees = 22 →
  distance * (num_trees - 1) = yard_length →
  distance = 21 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3535_353522


namespace NUMINAMATH_CALUDE_min_score_on_last_two_l3535_353528

/-- The number of tests Shauna takes -/
def num_tests : ℕ := 5

/-- The maximum score possible on each test -/
def max_score : ℕ := 120

/-- The desired average score across all tests -/
def target_average : ℕ := 95

/-- Shauna's scores on the first three tests -/
def first_three_scores : Fin 3 → ℕ
  | 0 => 86
  | 1 => 112
  | 2 => 91

/-- The sum of Shauna's scores on the first three tests -/
def sum_first_three : ℕ := (first_three_scores 0) + (first_three_scores 1) + (first_three_scores 2)

/-- The theorem stating the minimum score needed on one of the last two tests -/
theorem min_score_on_last_two (score : ℕ) :
  (sum_first_three + score + max_score = target_average * num_tests) ∧
  (∀ s, s < score → sum_first_three + s + max_score < target_average * num_tests) →
  score = 66 := by
  sorry

end NUMINAMATH_CALUDE_min_score_on_last_two_l3535_353528


namespace NUMINAMATH_CALUDE_uncle_bradley_money_l3535_353505

theorem uncle_bradley_money (M : ℚ) (F H : ℕ) : 
  F + H = 13 →
  50 * F = (3 / 10) * M →
  100 * H = (7 / 10) * M →
  M = 1300 := by
sorry

end NUMINAMATH_CALUDE_uncle_bradley_money_l3535_353505


namespace NUMINAMATH_CALUDE_quadratic_function_evaluation_l3535_353527

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem quadratic_function_evaluation :
  3 * g 2 + 2 * g (-2) = 85 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_evaluation_l3535_353527


namespace NUMINAMATH_CALUDE_max_value_when_min_ratio_l3535_353520

theorem max_value_when_min_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 - 3*x*y + 4*y^2 - z = 0) :
  ∃ (max_value : ℝ), max_value = 2 ∧
  ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 →
  x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 →
  (z' / (x' * y') ≥ z / (x * y)) →
  x' + 2*y' - z' ≤ max_value :=
sorry

end NUMINAMATH_CALUDE_max_value_when_min_ratio_l3535_353520


namespace NUMINAMATH_CALUDE_prime_pair_product_l3535_353557

theorem prime_pair_product : ∃ p q : ℕ, 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Odd (p + q) ∧ 
  p + q < 100 ∧ 
  (∃ k : ℕ, p + q = 17 * k) ∧ 
  p * q = 166 := by
  sorry

end NUMINAMATH_CALUDE_prime_pair_product_l3535_353557


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3535_353547

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3535_353547


namespace NUMINAMATH_CALUDE_book_pages_calculation_l3535_353541

/-- Given a book where:
  1. The initial ratio of pages read to pages not read is 3:4
  2. After reading 33 more pages, the ratio becomes 5:3
  This theorem states that the total number of pages in the book
  is equal to 33 divided by the difference between 5/8 and 3/7. -/
theorem book_pages_calculation (initial_read : ℚ) (initial_unread : ℚ) 
  (final_read : ℚ) (final_unread : ℚ) :
  initial_read / initial_unread = 3 / 4 →
  (initial_read + 33) / initial_unread = 5 / 3 →
  (initial_read + initial_unread) = 33 / (5/8 - 3/7) := by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l3535_353541


namespace NUMINAMATH_CALUDE_total_pies_count_l3535_353549

/-- The number of miniature pumpkin pies made by Pinky -/
def pinky_pies : ℕ := 147

/-- The number of miniature pumpkin pies made by Helen -/
def helen_pies : ℕ := 56

/-- The total number of miniature pumpkin pies -/
def total_pies : ℕ := pinky_pies + helen_pies

theorem total_pies_count : total_pies = 203 := by
  sorry

end NUMINAMATH_CALUDE_total_pies_count_l3535_353549


namespace NUMINAMATH_CALUDE_least_number_divisible_by_all_l3535_353559

def divisors : List Nat := [24, 32, 36, 54, 72, 81, 100]

theorem least_number_divisible_by_all (n : Nat) :
  (∀ d ∈ divisors, (n + 21) % d = 0) →
  (∀ m < n, ∃ d ∈ divisors, (m + 21) % d ≠ 0) →
  n = 64779 := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_all_l3535_353559


namespace NUMINAMATH_CALUDE_max_min_values_l3535_353535

theorem max_min_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 4 * a + b = 3) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 3 → b - 1 / a ≥ y - 1 / x) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 3 → 1 / (3 * a + 1) + 1 / (a + b) ≤ 1 / (3 * x + 1) + 1 / (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_l3535_353535


namespace NUMINAMATH_CALUDE_intersection_A_B_when_a_is_3_A_subset_B_iff_a_greater_than_5_l3535_353587

-- Define set A
def A : Set ℝ := {x | 1 < x - 1 ∧ x - 1 ≤ 4}

-- Define set B as a function of a
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for part I
theorem intersection_A_B_when_a_is_3 :
  A ∩ B 3 = {x : ℝ | 2 < x ∧ x < 3} := by sorry

-- Theorem for part II
theorem A_subset_B_iff_a_greater_than_5 :
  ∀ a : ℝ, A ⊆ B a ↔ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_a_is_3_A_subset_B_iff_a_greater_than_5_l3535_353587


namespace NUMINAMATH_CALUDE_positive_difference_of_solutions_l3535_353569

theorem positive_difference_of_solutions : ∃ (x₁ x₂ : ℝ), 
  (|2 * x₁ - 3| = 15 ∧ |2 * x₂ - 3| = 15) ∧ |x₁ - x₂| = 15 := by
  sorry

end NUMINAMATH_CALUDE_positive_difference_of_solutions_l3535_353569


namespace NUMINAMATH_CALUDE_is_334th_term_l3535_353500

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem is_334th_term :
  arithmetic_sequence 7 6 334 = 2005 :=
by sorry

end NUMINAMATH_CALUDE_is_334th_term_l3535_353500


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l3535_353511

theorem exponent_equation_solution :
  ∃ x : ℤ, (5 : ℝ)^7 * (5 : ℝ)^x = 125 ∧ x = -4 :=
by sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l3535_353511


namespace NUMINAMATH_CALUDE_inequality_solution_expression_value_l3535_353588

-- Problem 1
theorem inequality_solution (x : ℝ) : 2*x - 3 > x + 1 ↔ x > 4 := by sorry

-- Problem 2
theorem expression_value (a b : ℝ) (h : a^2 + 3*a*b = 5) : 
  (a + b) * (a + 2*b) - 2*b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_expression_value_l3535_353588


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_l3535_353571

theorem sqrt_twelve_minus_sqrt_three : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_l3535_353571


namespace NUMINAMATH_CALUDE_expansion_activity_optimal_time_l3535_353524

theorem expansion_activity_optimal_time :
  ∀ (x y : ℕ),
    x + y = 15 →
    x = 2 * y - 3 →
    ∀ (m : ℕ),
      m ≤ 10 →
      10 - m > (m / 2) →
      6 * m + 8 * (10 - m) ≥ 68 :=
by
  sorry

end NUMINAMATH_CALUDE_expansion_activity_optimal_time_l3535_353524


namespace NUMINAMATH_CALUDE_line_point_order_l3535_353584

/-- Given a line y = mx + n where m < 0 and n > 0, if points A(-2, y₁), B(-3, y₂), and C(1, y₃) 
    are on the line, then y₃ < y₁ < y₂. -/
theorem line_point_order (m n y₁ y₂ y₃ : ℝ) 
    (hm : m < 0) (hn : n > 0)
    (hA : y₁ = m * (-2) + n)
    (hB : y₂ = m * (-3) + n)
    (hC : y₃ = m * 1 + n) :
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_line_point_order_l3535_353584


namespace NUMINAMATH_CALUDE_total_stocking_stuffers_l3535_353592

def num_kids : ℕ := 3
def candy_canes_per_stocking : ℕ := 4
def beanie_babies_per_stocking : ℕ := 2
def books_per_stocking : ℕ := 1
def small_toys_per_stocking : ℕ := 3
def gift_cards_per_stocking : ℕ := 1

def items_per_stocking : ℕ := 
  candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking + 
  small_toys_per_stocking + gift_cards_per_stocking

theorem total_stocking_stuffers : 
  num_kids * items_per_stocking = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_stocking_stuffers_l3535_353592


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l3535_353551

theorem x_eq_2_sufficient_not_necessary :
  (∀ x : ℝ, x = 2 → (x + 1) * (x - 2) = 0) ∧
  (∃ x : ℝ, (x + 1) * (x - 2) = 0 ∧ x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l3535_353551


namespace NUMINAMATH_CALUDE_equation_unique_solution_l3535_353564

theorem equation_unique_solution :
  ∃! x : ℝ, Real.sqrt (3 + Real.sqrt (4 + Real.sqrt x)) = (3 + Real.sqrt x) ^ (1/3) ∧ x = 576 := by
  sorry

end NUMINAMATH_CALUDE_equation_unique_solution_l3535_353564


namespace NUMINAMATH_CALUDE_set_union_implies_a_zero_l3535_353578

theorem set_union_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {2^a, 3}
  let B : Set ℝ := {2, 3}
  A ∪ B = {1, 2, 3} → a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_set_union_implies_a_zero_l3535_353578


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l3535_353585

theorem opposite_of_negative_two_thirds :
  let x : ℚ := -2/3
  let opposite (y : ℚ) := -y
  opposite x = 2/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l3535_353585


namespace NUMINAMATH_CALUDE_ellipse_foci_l3535_353548

/-- The equation of an ellipse in standard form -/
def is_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 25 = 1

/-- The coordinates of a focus of the ellipse -/
def is_focus (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 3) ∨ (x = 0 ∧ y = -3)

/-- Theorem: The foci of the given ellipse are at (0, ±3) -/
theorem ellipse_foci :
  ∀ x y : ℝ, is_ellipse x y → (∃ fx fy : ℝ, is_focus fx fy ∧ 
    (x - fx)^2 + (y - fy)^2 = (x + fx)^2 + (y + fy)^2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_l3535_353548


namespace NUMINAMATH_CALUDE_mojave_population_increase_l3535_353596

/-- The factor by which the population of Mojave has increased over a decade -/
def population_increase_factor (initial_population : ℕ) (future_population : ℕ) (future_increase_percent : ℕ) : ℚ :=
  let current_population := (100 : ℚ) / (100 + future_increase_percent) * future_population
  current_population / initial_population

/-- Theorem stating that the population increase factor is 3 -/
theorem mojave_population_increase : 
  population_increase_factor 4000 16800 40 = 3 := by sorry

end NUMINAMATH_CALUDE_mojave_population_increase_l3535_353596


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l3535_353509

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l3535_353509


namespace NUMINAMATH_CALUDE_baby_tarantula_legs_l3535_353550

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

/-- The number of tarantulas in one egg sac -/
def tarantulas_per_sac : ℕ := 1000

/-- The number of egg sacs being considered -/
def egg_sacs : ℕ := 5 - 1

/-- The total number of baby tarantula legs in one less than 5 egg sacs -/
def total_baby_legs : ℕ := egg_sacs * tarantulas_per_sac * tarantula_legs

theorem baby_tarantula_legs :
  total_baby_legs = 32000 := by sorry

end NUMINAMATH_CALUDE_baby_tarantula_legs_l3535_353550


namespace NUMINAMATH_CALUDE_product_closest_to_640_l3535_353544

def product : ℝ := 0.0000421 * 15864300

def options : List ℝ := [620, 640, 660, 680, 700]

theorem product_closest_to_640 : 
  (options.argmin (fun x => |x - product|)) = some 640 := by sorry

end NUMINAMATH_CALUDE_product_closest_to_640_l3535_353544


namespace NUMINAMATH_CALUDE_geometric_series_relation_l3535_353561

/-- Given real numbers x and y satisfying an infinite geometric series equation,
    prove that another related infinite geometric series has a specific value. -/
theorem geometric_series_relation (x y : ℝ) 
  (h : (x / y) / (1 - 1 / y) = 3) :
  (x / (x + 2 * y)) / (1 - 1 / (x + 2 * y)) = 3 * (y - 1) / (5 * y - 4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l3535_353561


namespace NUMINAMATH_CALUDE_money_distribution_l3535_353512

theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 → 
  a + c = 200 → 
  b + c = 350 → 
  c = 50 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3535_353512


namespace NUMINAMATH_CALUDE_solution_range_l3535_353543

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, 2 * (x + a) = x + 3 ∧ 2 * x - 10 > 8 * a) → 
  a < -1/3 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l3535_353543


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_prime_angles_l3535_353501

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def consecutive_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p < q ∧ ∀ m, p < m → m < q → ¬is_prime m

theorem right_triangle_consecutive_prime_angles (p q : ℕ) :
  p < q →
  consecutive_primes p q →
  p + q = 90 →
  (∀ p' q' : ℕ, p' < q' → consecutive_primes p' q' → p' + q' = 90 → p ≤ p') →
  p = 43 := by sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_prime_angles_l3535_353501


namespace NUMINAMATH_CALUDE_average_problem_l3535_353572

theorem average_problem (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h : (x₁ + x₂ + x₃ + x₄ + x₅ + 3) / 6 = 3) : 
  (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l3535_353572


namespace NUMINAMATH_CALUDE_complex_fraction_problem_l3535_353518

theorem complex_fraction_problem (x y : ℂ) 
  (h : (x - y) / (x + y) - (x + y) / (x - y) = 2) :
  ∃ (result : ℂ), (x^6 + y^6) / (x^6 - y^6) - (x^6 - y^6) / (x^6 + y^6) = result :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_problem_l3535_353518


namespace NUMINAMATH_CALUDE_opposite_of_blue_is_white_l3535_353593

/-- Represents the colors of the squares --/
inductive Color
| Red | Blue | Orange | Purple | Green | Yellow | White

/-- Represents the positions on the cube --/
inductive Position
| Top | Bottom | Front | Back | Left | Right

/-- Represents a cube configuration --/
structure CubeConfig where
  top : Color
  bottom : Color
  front : Color
  back : Color
  left : Color
  right : Color

/-- Defines the property of opposite faces --/
def isOpposite (p1 p2 : Position) : Prop :=
  (p1 = Position.Top ∧ p2 = Position.Bottom) ∨
  (p1 = Position.Bottom ∧ p2 = Position.Top) ∨
  (p1 = Position.Front ∧ p2 = Position.Back) ∨
  (p1 = Position.Back ∧ p2 = Position.Front) ∨
  (p1 = Position.Left ∧ p2 = Position.Right) ∨
  (p1 = Position.Right ∧ p2 = Position.Left)

/-- The main theorem --/
theorem opposite_of_blue_is_white 
  (cube : CubeConfig)
  (top_is_purple : cube.top = Color.Purple)
  (front_is_green : cube.front = Color.Green)
  (blue_on_side : cube.left = Color.Blue ∨ cube.right = Color.Blue) :
  (cube.left = Color.Blue ∧ cube.right = Color.White) ∨ 
  (cube.right = Color.Blue ∧ cube.left = Color.White) :=
sorry

end NUMINAMATH_CALUDE_opposite_of_blue_is_white_l3535_353593


namespace NUMINAMATH_CALUDE_modified_ohara_triple_solution_l3535_353580

/-- Definition of a Modified O'Hara Triple -/
def is_modified_ohara_triple (a b x k : ℕ+) : Prop :=
  k * (a : ℝ).sqrt + (b : ℝ).sqrt = x

/-- Theorem: If (49, 16, x, 2) is a Modified O'Hara Triple, then x = 18 -/
theorem modified_ohara_triple_solution :
  ∀ x : ℕ+, is_modified_ohara_triple 49 16 x 2 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_modified_ohara_triple_solution_l3535_353580


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3535_353591

-- Define the function f(x) = x^2 + px + q
def f (p q x : ℝ) : ℝ := x^2 + p*x + q

-- Theorem statement
theorem min_value_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f p q x_min ≤ f p q x ∧ x_min = -p/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3535_353591


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3535_353586

/-- A geometric sequence of positive real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 3 + a 2 * a 4 + 2 * a 2 * a 3 = 49 →
  a 2 + a 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3535_353586


namespace NUMINAMATH_CALUDE_ellipse_standard_form_l3535_353575

/-- Given an ellipse with equation (x²/a²) + (y²/b²) = 1, where a > b > 0,
    passing through the point (√6/2, 1/2), and having an eccentricity of √2/2,
    prove that the standard form of the ellipse equation is (x²/a²) + (y²/b²) = 1 -/
theorem ellipse_standard_form 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (Real.sqrt 6 / 2)^2 / a^2 + (1/2)^2 / b^2 = 1)
  (h4 : Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2) :
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_form_l3535_353575


namespace NUMINAMATH_CALUDE_cube_of_negative_half_x_squared_y_l3535_353515

theorem cube_of_negative_half_x_squared_y (x y : ℝ) : 
  (-1/2 * x^2 * y)^3 = -1/8 * x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_half_x_squared_y_l3535_353515


namespace NUMINAMATH_CALUDE_row_sum_1008_equals_2015_squared_l3535_353532

/-- Represents the sum of numbers in a row of the given pattern. -/
def row_sum (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The theorem stating that the sum of numbers in the 1008th row equals 2015². -/
theorem row_sum_1008_equals_2015_squared : row_sum 1008 = 2015 ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_row_sum_1008_equals_2015_squared_l3535_353532


namespace NUMINAMATH_CALUDE_books_movies_difference_l3535_353570

theorem books_movies_difference (total_books total_movies : ℕ) 
  (h1 : total_books = 10) 
  (h2 : total_movies = 6) : 
  total_books - total_movies = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_movies_difference_l3535_353570


namespace NUMINAMATH_CALUDE_move_point_left_l3535_353574

/-- Given a point A in a 2D Cartesian coordinate system, moving it
    3 units to the left results in a new point A' -/
def move_left (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 - 3, A.2)

/-- Theorem: Moving point A(1, -1) 3 units to the left results in A'(-2, -1) -/
theorem move_point_left : 
  let A : ℝ × ℝ := (1, -1)
  move_left A = (-2, -1) := by
sorry

end NUMINAMATH_CALUDE_move_point_left_l3535_353574


namespace NUMINAMATH_CALUDE_existence_of_least_t_for_geometric_progression_l3535_353517

open Real

theorem existence_of_least_t_for_geometric_progression :
  ∃ t : ℝ, t > 0 ∧
  ∃ α : ℝ, 0 < α ∧ α < π / 3 ∧
  ∃ r : ℝ, r > 0 ∧
  (arcsin (sin α) = α) ∧
  (arcsin (sin (3 * α)) = r * α) ∧
  (arcsin (sin (8 * α)) = r^2 * α) ∧
  (arcsin (sin (t * α)) = r^3 * α) ∧
  ∀ s : ℝ, s > 0 →
    (∃ β : ℝ, 0 < β ∧ β < π / 3 ∧
    ∃ q : ℝ, q > 0 ∧
    (arcsin (sin β) = β) ∧
    (arcsin (sin (3 * β)) = q * β) ∧
    (arcsin (sin (8 * β)) = q^2 * β) ∧
    (arcsin (sin (s * β)) = q^3 * β)) →
    t ≤ s :=
by sorry

end NUMINAMATH_CALUDE_existence_of_least_t_for_geometric_progression_l3535_353517


namespace NUMINAMATH_CALUDE_bracelet_ratio_l3535_353540

theorem bracelet_ratio : 
  ∀ (x : ℕ), 
  (5 + x : ℚ) - (1/3) * (5 + x) = 6 → 
  (x : ℚ) / 16 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_ratio_l3535_353540


namespace NUMINAMATH_CALUDE_platform_length_l3535_353552

/-- Given a train of length 300 meters that takes 30 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is 200 meters. -/
theorem platform_length (train_length : ℝ) (platform_cross_time : ℝ) (pole_cross_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_cross_time = 30)
  (h3 : pole_cross_time = 18) :
  let platform_length := (train_length * platform_cross_time / pole_cross_time) - train_length
  platform_length = 200 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l3535_353552


namespace NUMINAMATH_CALUDE_laptop_lighter_than_tote_l3535_353531

/-- Represents the weights of various items in pounds -/
structure Weights where
  karens_tote : ℝ
  kevins_empty_briefcase : ℝ
  kevins_umbrella : ℝ
  kevins_laptop : ℝ
  kevins_work_papers : ℝ

/-- Conditions given in the problem -/
def problem_conditions (w : Weights) : Prop :=
  w.karens_tote = 8 ∧
  w.karens_tote = 2 * w.kevins_empty_briefcase ∧
  w.kevins_umbrella = w.kevins_empty_briefcase / 2 ∧
  w.kevins_empty_briefcase + w.kevins_laptop + w.kevins_work_papers + w.kevins_umbrella = 2 * w.karens_tote ∧
  w.kevins_work_papers = (w.kevins_empty_briefcase + w.kevins_laptop + w.kevins_work_papers) / 6

theorem laptop_lighter_than_tote (w : Weights) (h : problem_conditions w) :
  w.kevins_laptop < w.karens_tote ∧ w.karens_tote - w.kevins_laptop = 1/3 := by
  sorry

#check laptop_lighter_than_tote

end NUMINAMATH_CALUDE_laptop_lighter_than_tote_l3535_353531


namespace NUMINAMATH_CALUDE_impossibleArrangement_l3535_353539

/-- Represents a 3x3 grid filled with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- The given fixed grid arrangement -/
def fixedGrid : Grid :=
  fun i j => Fin.mk ((i.val * 3 + j.val) % 9 + 1) (by sorry)

/-- Two positions are adjacent if they share a side -/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ p.1.val = q.1.val + 1))

/-- Two numbers are neighbors in a grid if they are in adjacent positions -/
def neighbors (g : Grid) (x y : Fin 9) : Prop :=
  ∃ (p q : Fin 3 × Fin 3), g p.1 p.2 = x ∧ g q.1 q.2 = y ∧ adjacent p q

theorem impossibleArrangement :
  ¬∃ (g₂ g₃ : Grid),
    (∀ x y : Fin 9, (neighbors fixedGrid x y ∨ neighbors g₂ x y ∨ neighbors g₃ x y) →
      ¬(neighbors fixedGrid x y ∧ neighbors g₂ x y) ∧
      ¬(neighbors fixedGrid x y ∧ neighbors g₃ x y) ∧
      ¬(neighbors g₂ x y ∧ neighbors g₃ x y)) :=
by sorry

end NUMINAMATH_CALUDE_impossibleArrangement_l3535_353539


namespace NUMINAMATH_CALUDE_sun_city_has_12000_people_l3535_353507

/-- The population of Willowdale City -/
def willowdale_population : ℕ := 2000

/-- The population of Roseville City -/
def roseville_population : ℕ := 3 * willowdale_population - 500

/-- The population of Sun City -/
def sun_city_population : ℕ := 2 * roseville_population + 1000

/-- Theorem stating that Sun City has 12000 people -/
theorem sun_city_has_12000_people : sun_city_population = 12000 := by
  sorry

end NUMINAMATH_CALUDE_sun_city_has_12000_people_l3535_353507


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l3535_353589

theorem factor_implies_d_value (d : ℚ) :
  (∀ x : ℚ, (3 * x + 4) ∣ (4 * x^3 + 17 * x^2 + d * x + 28)) →
  d = 155 / 9 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l3535_353589


namespace NUMINAMATH_CALUDE_linda_needs_one_train_l3535_353597

/-- The number of trains Linda currently has -/
def current_trains : ℕ := 31

/-- The number of trains Linda wants in each row -/
def trains_per_row : ℕ := 8

/-- The function to calculate the smallest number of additional trains needed -/
def additional_trains_needed (current : ℕ) (per_row : ℕ) : ℕ :=
  (per_row - current % per_row) % per_row

/-- The theorem stating that Linda needs 1 additional train -/
theorem linda_needs_one_train : 
  additional_trains_needed current_trains trains_per_row = 1 := by
  sorry

end NUMINAMATH_CALUDE_linda_needs_one_train_l3535_353597


namespace NUMINAMATH_CALUDE_girls_count_in_school_l3535_353534

/-- Proves that in a school with a given total number of students and a ratio of boys to girls,
    the number of girls is as calculated. -/
theorem girls_count_in_school (total : ℕ) (boys_ratio girls_ratio : ℕ) 
    (h_total : total = 480) 
    (h_ratio : boys_ratio = 3 ∧ girls_ratio = 5) : 
    (girls_ratio * total) / (boys_ratio + girls_ratio) = 300 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_in_school_l3535_353534


namespace NUMINAMATH_CALUDE_total_bird_wings_l3535_353506

/-- The number of birds in the sky -/
def num_birds : ℕ := 10

/-- The number of wings each bird has -/
def wings_per_bird : ℕ := 2

/-- Theorem: The total number of bird wings in the sky is 20 -/
theorem total_bird_wings : num_birds * wings_per_bird = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_bird_wings_l3535_353506


namespace NUMINAMATH_CALUDE_parabola_axis_l3535_353502

/-- The equation of the axis of the parabola y = x^2 -/
theorem parabola_axis (x y : ℝ) : 
  (y = x^2) → (∃ (axis : ℝ → ℝ), axis y = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_l3535_353502


namespace NUMINAMATH_CALUDE_first_group_size_l3535_353516

/-- Represents the number of questions Cameron answers per tourist -/
def questions_per_tourist : ℕ := 2

/-- Represents the total number of tour groups -/
def total_groups : ℕ := 4

/-- Represents the number of people in the second group -/
def second_group : ℕ := 11

/-- Represents the number of people in the third group -/
def third_group : ℕ := 8

/-- Represents the number of people in the fourth group -/
def fourth_group : ℕ := 7

/-- Represents the total number of questions Cameron answered -/
def total_questions : ℕ := 68

/-- Proves that the number of people in the first tour group is 8 -/
theorem first_group_size : ℕ := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l3535_353516


namespace NUMINAMATH_CALUDE_integer_division_implication_l3535_353579

theorem integer_division_implication (n : ℕ) (m : ℤ) :
  2^n - 2 = m * n →
  ∃ k : ℤ, (2^(2^n - 1) - 2) / (2^n - 1) = 2 * k :=
sorry

end NUMINAMATH_CALUDE_integer_division_implication_l3535_353579


namespace NUMINAMATH_CALUDE_odd_number_as_difference_of_squares_l3535_353583

theorem odd_number_as_difference_of_squares (n : ℤ) : 
  2 * n + 1 = (n + 1)^2 - n^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_as_difference_of_squares_l3535_353583


namespace NUMINAMATH_CALUDE_bus_ride_difference_l3535_353510

theorem bus_ride_difference (vince_ride zachary_ride : ℝ) 
  (h1 : vince_ride = 0.625) 
  (h2 : zachary_ride = 0.5) : 
  vince_ride - zachary_ride = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l3535_353510


namespace NUMINAMATH_CALUDE_g_negative_three_l3535_353599

def g (x : ℝ) : ℝ := 3*x^5 - 5*x^4 + 9*x^3 - 6*x^2 + 15*x - 210

theorem g_negative_three : g (-3) = -1686 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_three_l3535_353599


namespace NUMINAMATH_CALUDE_b_40_mod_49_l3535_353542

def b (n : ℕ) : ℤ := 5^n - 7^n

theorem b_40_mod_49 : b 40 ≡ 2 [ZMOD 49] := by sorry

end NUMINAMATH_CALUDE_b_40_mod_49_l3535_353542


namespace NUMINAMATH_CALUDE_root_product_equality_l3535_353530

theorem root_product_equality : 
  (16 : ℝ) ^ (1/5) * (64 : ℝ) ^ (1/6) = 2 * (16 : ℝ) ^ (1/5) :=
by sorry

end NUMINAMATH_CALUDE_root_product_equality_l3535_353530


namespace NUMINAMATH_CALUDE_min_digits_removal_l3535_353523

def original_number : ℕ := 20162016

def is_valid_removal (n : ℕ) : Prop :=
  ∃ (removed : ℕ),
    removed > 0 ∧
    removed < original_number ∧
    (original_number - removed) % 2016 = 0 ∧
    (String.length (toString removed) + String.length (toString (original_number - removed)) = 8)

theorem min_digits_removal :
  (∀ n : ℕ, n < 3 → ¬(is_valid_removal n)) ∧
  (∃ n : ℕ, n = 3 ∧ is_valid_removal n) :=
sorry

end NUMINAMATH_CALUDE_min_digits_removal_l3535_353523


namespace NUMINAMATH_CALUDE_prob_at_least_one_girl_pair_approx_l3535_353525

/-- The number of boys in the group -/
def num_boys : ℕ := 8

/-- The number of girls in the group -/
def num_girls : ℕ := 8

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The probability of at least one pair consisting of two girls -/
noncomputable def prob_at_least_one_girl_pair : ℝ :=
  1 - (num_boys.factorial * num_girls.factorial * (2^num_pairs) * num_pairs.factorial) / total_people.factorial

/-- Theorem stating that the probability of at least one pair consisting of two girls is approximately 0.98 -/
theorem prob_at_least_one_girl_pair_approx :
  abs (prob_at_least_one_girl_pair - 0.98) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_girl_pair_approx_l3535_353525


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l3535_353521

/-- Calculates the length of the second train given the speeds of both trains,
    the time they take to clear each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (clear_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * (1000 / 3600)
  let total_distance := relative_speed_ms * clear_time
  total_distance - length1

/-- The length of the second train is approximately 1984 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, abs (second_train_length 75 65 7.353697418492236 121 - 1984) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l3535_353521


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3535_353519

def is_valid (n : ℕ) : Prop :=
  n % 10 = 9 ∧
  n % 9 = 8 ∧
  n % 8 = 7 ∧
  n % 7 = 6 ∧
  n % 6 = 5 ∧
  n % 5 = 4 ∧
  n % 4 = 3 ∧
  n % 3 = 2 ∧
  n % 2 = 1

theorem smallest_valid_number :
  is_valid 2519 ∧ ∀ m : ℕ, m < 2519 → ¬ is_valid m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3535_353519


namespace NUMINAMATH_CALUDE_unique_solution_l3535_353594

def clubsuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

theorem unique_solution : ∃! A : ℝ, clubsuit A 5 = 80 ∧ A = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3535_353594


namespace NUMINAMATH_CALUDE_greatest_n_value_exists_greatest_n_l3535_353577

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 8100) : n ≤ 8 :=
sorry

theorem exists_greatest_n :
  ∃ n : ℤ, n = 8 ∧ 101 * n^2 ≤ 8100 ∧ ∀ m : ℤ, 101 * m^2 ≤ 8100 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_n_value_exists_greatest_n_l3535_353577


namespace NUMINAMATH_CALUDE_citrus_grove_orchards_l3535_353555

theorem citrus_grove_orchards (total : ℕ) (lemons : ℕ) (oranges : ℕ) (limes : ℕ) (grapefruits : ℕ) :
  total = 16 →
  lemons = 8 →
  oranges = lemons / 2 →
  limes + grapefruits = total - lemons - oranges →
  limes = grapefruits →
  grapefruits = 2 := by
sorry

end NUMINAMATH_CALUDE_citrus_grove_orchards_l3535_353555


namespace NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_8_times_sqrt_4_l3535_353504

theorem fourth_root_256_times_cube_root_8_times_sqrt_4 : 
  (256 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (4 : ℝ) ^ (1/2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_8_times_sqrt_4_l3535_353504


namespace NUMINAMATH_CALUDE_counterfeiters_payment_range_l3535_353554

/-- Represents a counterfeiter who can pay amounts between 1 and 25 rubles --/
structure Counterfeiter where
  pay : ℕ → ℕ
  pay_range : ∀ n, 1 ≤ pay n ∧ pay n ≤ 25

/-- The theorem states that three counterfeiters can collectively pay any amount from 100 to 200 rubles --/
theorem counterfeiters_payment_range (c1 c2 c3 : Counterfeiter) :
  ∀ n, 100 ≤ n ∧ n ≤ 200 → ∃ (x y z : ℕ), x + y + z = n ∧ 
    (∃ (a b c : ℕ), c1.pay a + c2.pay b + c3.pay c = x) ∧
    (∃ (d e f : ℕ), c1.pay d + c2.pay e + c3.pay f = y) ∧
    (∃ (g h i : ℕ), c1.pay g + c2.pay h + c3.pay i = z) :=
  sorry

end NUMINAMATH_CALUDE_counterfeiters_payment_range_l3535_353554


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3535_353573

/-- Given a quadratic function f(x) = ax^2 + bx + 5 where a ≠ 0,
    if there exist two distinct points (x₁, 2002) and (x₂, 2002) on the graph of f,
    then f(x₁ + x₂) = 5. -/
theorem quadratic_function_property (a b x₁ x₂ : ℝ) (ha : a ≠ 0) :
  let f := λ x : ℝ => a * x^2 + b * x + 5
  (f x₁ = 2002) → (f x₂ = 2002) → (x₁ ≠ x₂) → f (x₁ + x₂) = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3535_353573


namespace NUMINAMATH_CALUDE_proposition_p_q_equivalence_l3535_353566

theorem proposition_p_q_equivalence (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) ∧
  (∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0) ↔
  2 < m ∧ m < 3 :=
sorry

end NUMINAMATH_CALUDE_proposition_p_q_equivalence_l3535_353566


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l3535_353581

/-- Given a quadratic expression x^2 - 24x + 50 that can be rewritten as (x+d)^2 + e,
    this theorem states that d + e = -106. -/
theorem quadratic_rewrite_sum (d e : ℝ) : 
  (∀ x, x^2 - 24*x + 50 = (x + d)^2 + e) → d + e = -106 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l3535_353581


namespace NUMINAMATH_CALUDE_proportion_sum_l3535_353567

theorem proportion_sum (x y : ℝ) : 
  (31.25 : ℝ) / x = 100 / (9.6 : ℝ) ∧ x / 13.75 = (9.6 : ℝ) / y → x + y = 47 := by
  sorry

end NUMINAMATH_CALUDE_proportion_sum_l3535_353567


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l3535_353582

theorem recipe_flour_amount (flour_added : ℕ) (flour_needed : ℕ) : 
  flour_added = 4 → flour_needed = 4 → flour_added + flour_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l3535_353582


namespace NUMINAMATH_CALUDE_suitcase_lock_settings_l3535_353538

/-- The number of digits on each dial -/
def num_digits : ℕ := 10

/-- The number of dials on the lock -/
def num_dials : ℕ := 4

/-- The number of choices for each dial after the first -/
def choices_after_first : ℕ := num_digits - 1

/-- The total number of possible settings for the lock -/
def total_settings : ℕ := num_digits * choices_after_first^(num_dials - 1)

/-- Theorem stating that the total number of settings is 7290 -/
theorem suitcase_lock_settings : total_settings = 7290 := by
  sorry

end NUMINAMATH_CALUDE_suitcase_lock_settings_l3535_353538


namespace NUMINAMATH_CALUDE_max_profit_at_one_MP_decreasing_max_profit_x_eq_one_l3535_353576

-- Define the profit function
def P (x : ℕ) : ℚ := -0.2 * x^2 + 25 * x - 40

-- Define the marginal profit function
def MP (x : ℕ) : ℚ := P (x + 1) - P x

-- State the theorem
theorem max_profit_at_one :
  ∀ x : ℕ, 1 ≤ x → x ≤ 100 → P 1 ≥ P x ∧ P 1 = 24.4 := by
  sorry

-- Prove that MP is decreasing
theorem MP_decreasing :
  ∀ x y : ℕ, x < y → MP x > MP y := by
  sorry

-- Prove that maximum profit occurs at x = 1
theorem max_profit_x_eq_one :
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 100 ∧ ∀ y : ℕ, 1 ≤ y ∧ y ≤ 100 → P x ≥ P y := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_one_MP_decreasing_max_profit_x_eq_one_l3535_353576


namespace NUMINAMATH_CALUDE_milk_remaining_l3535_353598

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) : 
  initial = 4 → given_away = 16/3 → remaining = initial - given_away → remaining = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_milk_remaining_l3535_353598


namespace NUMINAMATH_CALUDE_sector_perimeter_l3535_353595

theorem sector_perimeter (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 8) :
  2 * r + 2 * area / r = 12 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l3535_353595


namespace NUMINAMATH_CALUDE_inequality_proof_l3535_353545

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : |x - y| < 2) (hyz : |y - z| < 2) (hzx : |z - x| < 2) :
  Real.sqrt (x * y + 1) + Real.sqrt (y * z + 1) + Real.sqrt (z * x + 1) > x + y + z :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3535_353545


namespace NUMINAMATH_CALUDE_min_students_for_question_distribution_l3535_353514

theorem min_students_for_question_distribution (total_questions : Nat) 
  (folder_size : Nat) (num_folders : Nat) (max_unsolved : Nat) :
  total_questions = 2010 →
  folder_size = 670 →
  num_folders = 3 →
  max_unsolved = 2 →
  ∃ (min_students : Nat), 
    (∀ (n : Nat), n < min_students → 
      ¬(∀ (folder : Finset Nat), folder.card = folder_size → 
        ∃ (solved_by : Finset Nat), solved_by.card ≥ num_folders ∧ 
          ∀ (q : Nat), q ∈ folder → (n - solved_by.card) ≤ max_unsolved)) ∧
    (∀ (folder : Finset Nat), folder.card = folder_size → 
      ∃ (solved_by : Finset Nat), solved_by.card ≥ num_folders ∧ 
        ∀ (q : Nat), q ∈ folder → (min_students - solved_by.card) ≤ max_unsolved) ∧
    min_students = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_students_for_question_distribution_l3535_353514


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_19_l3535_353562

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is four-digit -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- Theorem stating that 9730 is the largest four-digit number whose digits add up to 19 -/
theorem largest_four_digit_sum_19 : 
  (∀ n : ℕ, is_four_digit n → sum_of_digits n = 19 → n ≤ 9730) ∧ 
  is_four_digit 9730 ∧ 
  sum_of_digits 9730 = 19 := by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_19_l3535_353562


namespace NUMINAMATH_CALUDE_city_rentals_cost_per_mile_l3535_353526

/-- Proves that the cost per mile for City Rentals is $0.16 given the rental rates and equal cost for 48.0 miles. -/
theorem city_rentals_cost_per_mile :
  let sunshine_daily_rate : ℝ := 17.99
  let sunshine_per_mile : ℝ := 0.18
  let city_daily_rate : ℝ := 18.95
  let miles : ℝ := 48.0
  ∀ city_per_mile : ℝ,
    sunshine_daily_rate + sunshine_per_mile * miles = city_daily_rate + city_per_mile * miles →
    city_per_mile = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_city_rentals_cost_per_mile_l3535_353526


namespace NUMINAMATH_CALUDE_min_horizontal_distance_l3535_353558

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - x - 6

/-- Theorem stating the minimum horizontal distance between points P and Q -/
theorem min_horizontal_distance :
  ∃ (xp xq : ℝ),
    f xp = 6 ∧
    f xq = -6 ∧
    ∀ (yp yq : ℝ),
      f yp = 6 → f yq = -6 →
      |xp - xq| ≤ |yp - yq| ∧
      |xp - xq| = 4 :=
sorry

end NUMINAMATH_CALUDE_min_horizontal_distance_l3535_353558


namespace NUMINAMATH_CALUDE_fractional_sum_equality_l3535_353568

theorem fractional_sum_equality (n : ℕ) (h : n > 1) :
  ∃ i j : ℕ, (1 : ℚ) / n = 
    Finset.sum (Finset.range (j - i + 1)) (λ k => 1 / ((i + k) * (i + k + 1))) := by
  sorry

end NUMINAMATH_CALUDE_fractional_sum_equality_l3535_353568


namespace NUMINAMATH_CALUDE_nancy_books_count_l3535_353565

/-- Given that Alyssa has 36 books and Nancy has 7 times more books than Alyssa,
    prove that Nancy has 252 books. -/
theorem nancy_books_count (alyssa_books : ℕ) (nancy_books : ℕ) 
    (h1 : alyssa_books = 36)
    (h2 : nancy_books = 7 * alyssa_books) : 
  nancy_books = 252 := by
  sorry

end NUMINAMATH_CALUDE_nancy_books_count_l3535_353565


namespace NUMINAMATH_CALUDE_log_equation_solution_l3535_353529

theorem log_equation_solution (m n : ℝ) (b : ℝ) (h : m > 0) (h' : n > 0) :
  Real.log m^2 / Real.log 10 = b - Real.log n^3 / Real.log 10 →
  m = (10^b / n^3)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3535_353529


namespace NUMINAMATH_CALUDE_continuous_fraction_identity_l3535_353536

-- Define x as the solution to the equation x = 1 + √3 / x
noncomputable def x : ℝ := Real.sqrt 3 + 1

-- State the theorem
theorem continuous_fraction_identity :
  1 / ((x + 2) * (x - 3)) = (Real.sqrt 3 + 6) / (-33) := by
  sorry

end NUMINAMATH_CALUDE_continuous_fraction_identity_l3535_353536


namespace NUMINAMATH_CALUDE_product_equality_implies_sum_l3535_353563

theorem product_equality_implies_sum (g h a b : ℝ) :
  (∀ d : ℝ, (8 * d^2 - 4 * d + g) * (2 * d^2 + h * d - 7) = 16 * d^4 - 28 * d^3 + a * h^2 * d^2 - b * d + 49) →
  g + h = -3 := by
sorry

end NUMINAMATH_CALUDE_product_equality_implies_sum_l3535_353563


namespace NUMINAMATH_CALUDE_heesu_has_greatest_sum_l3535_353556

-- Define the card numbers for each player
def sora_cards : List Nat := [4, 6]
def heesu_cards : List Nat := [7, 5]
def jiyeon_cards : List Nat := [3, 8]

-- Define a function to calculate the sum of a player's cards
def sum_cards (cards : List Nat) : Nat :=
  cards.sum

-- Theorem statement
theorem heesu_has_greatest_sum :
  sum_cards heesu_cards > sum_cards sora_cards ∧
  sum_cards heesu_cards > sum_cards jiyeon_cards :=
by
  sorry


end NUMINAMATH_CALUDE_heesu_has_greatest_sum_l3535_353556


namespace NUMINAMATH_CALUDE_remainder_theorem_l3535_353533

theorem remainder_theorem (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 19) → 
  (∃ m : ℤ, N = 13 * m + 6) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3535_353533


namespace NUMINAMATH_CALUDE_correct_calculation_l3535_353590

theorem correct_calculation (a b : ℝ) : 3 * a * b + 2 * a * b = 5 * a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3535_353590


namespace NUMINAMATH_CALUDE_alpha_30_sufficient_not_necessary_for_sin_half_l3535_353513

theorem alpha_30_sufficient_not_necessary_for_sin_half :
  (∀ α : Real, α = 30 * π / 180 → Real.sin α = 1 / 2) ∧
  (∃ α : Real, Real.sin α = 1 / 2 ∧ α ≠ 30 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_alpha_30_sufficient_not_necessary_for_sin_half_l3535_353513


namespace NUMINAMATH_CALUDE_sheets_used_for_printing_james_sheets_used_l3535_353553

/-- Calculate the number of sheets of paper used for printing books -/
theorem sheets_used_for_printing (num_books : ℕ) (pages_per_book : ℕ) 
  (pages_per_side : ℕ) (is_double_sided : Bool) : ℕ :=
  let total_pages := num_books * pages_per_book
  let pages_per_sheet := pages_per_side * (if is_double_sided then 2 else 1)
  total_pages / pages_per_sheet

/-- Prove that James uses 150 sheets of paper for printing his books -/
theorem james_sheets_used :
  sheets_used_for_printing 2 600 4 true = 150 := by
  sorry

end NUMINAMATH_CALUDE_sheets_used_for_printing_james_sheets_used_l3535_353553


namespace NUMINAMATH_CALUDE_sector_area_l3535_353546

/-- Given a sector with radius 4 cm and arc length 12 cm, its area is 24 cm². -/
theorem sector_area (radius : ℝ) (arc_length : ℝ) (area : ℝ) : 
  radius = 4 → arc_length = 12 → area = (1/2) * arc_length * radius → area = 24 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3535_353546


namespace NUMINAMATH_CALUDE_additive_function_characterization_l3535_353560

/-- A function satisfying the given functional equation -/
def AdditiveFunctionQ (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) = f x + f y

/-- The main theorem characterizing additive functions on rationals -/
theorem additive_function_characterization :
  ∀ f : ℚ → ℚ, AdditiveFunctionQ f →
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x := by
  sorry

end NUMINAMATH_CALUDE_additive_function_characterization_l3535_353560


namespace NUMINAMATH_CALUDE_hyperbola_construction_equivalence_l3535_353537

/-- The equation of a hyperbola in standard form -/
def is_hyperbola_point (a b x y : ℝ) : Prop :=
  (x / a)^2 - (y / b)^2 = 1

/-- The construction equation for a point on the hyperbola -/
def satisfies_construction (a b x y : ℝ) : Prop :=
  x = (a / b) * Real.sqrt (b^2 + y^2)

/-- Theorem: Any point satisfying the hyperbola equation also satisfies the construction equation -/
theorem hyperbola_construction_equivalence (a b x y : ℝ) (ha : a > 0) (hb : b > 0) :
  is_hyperbola_point a b x y → satisfies_construction a b x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_construction_equivalence_l3535_353537
