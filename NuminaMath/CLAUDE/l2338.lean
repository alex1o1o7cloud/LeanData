import Mathlib

namespace NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l2338_233821

theorem ceiling_fraction_evaluation : 
  (⌈(19 / 8 : ℚ) - ⌈(35 / 19 : ℚ)⌉⌉ : ℚ) / 
  (⌈(35 / 8 : ℚ) + ⌈(8 * 19 / 35 : ℚ)⌉⌉ : ℚ) = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l2338_233821


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_segment_length_l2338_233859

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Determines if a quadrilateral is convex -/
def isConvex (quad : Quadrilateral) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_segment_length 
  (PQRS : Quadrilateral) 
  (T : Point) :
  isConvex PQRS →
  distance PQRS.P PQRS.Q = 15 →
  distance PQRS.R PQRS.S = 20 →
  distance PQRS.P PQRS.R = 25 →
  T = intersection PQRS.P PQRS.R PQRS.Q PQRS.S →
  triangleArea PQRS.P T PQRS.R = triangleArea PQRS.Q T PQRS.S →
  distance PQRS.P T = 75 / 7 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_segment_length_l2338_233859


namespace NUMINAMATH_CALUDE_min_distance_MN_min_distance_is_two_l2338_233886

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2
def g (x : ℝ) : ℝ := x - 1

def M (x₁ : ℝ) : ℝ × ℝ := (x₁, f x₁)
def N (x₂ : ℝ) : ℝ × ℝ := (x₂, g x₂)

theorem min_distance_MN (x₁ x₂ : ℝ) (h₁ : x₁ ≥ 0) (h₂ : x₂ > 0) 
  (h₃ : f x₁ = g x₂) : 
  ∀ y₁ y₂ : ℝ, y₁ ≥ 0 → y₂ > 0 → f y₁ = g y₂ → 
  |x₂ - x₁| ≤ |y₂ - y₁| := by sorry

theorem min_distance_is_two (x₁ x₂ : ℝ) (h₁ : x₁ ≥ 0) (h₂ : x₂ > 0) 
  (h₃ : f x₁ = g x₂) : 
  |x₂ - x₁| = 2 := by sorry

end NUMINAMATH_CALUDE_min_distance_MN_min_distance_is_two_l2338_233886


namespace NUMINAMATH_CALUDE_max_candies_eaten_l2338_233875

theorem max_candies_eaten (n : ℕ) (h : n = 25) : 
  (n.choose 2) = 300 := by
  sorry

#check max_candies_eaten

end NUMINAMATH_CALUDE_max_candies_eaten_l2338_233875


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2338_233854

-- Define the sets A and B
def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2338_233854


namespace NUMINAMATH_CALUDE_problem_statement_l2338_233883

theorem problem_statement (a b c d m : ℝ) 
  (h1 : a + b = 0)
  (h2 : c * d = 1)
  (h3 : |m| = 2) :
  m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2338_233883


namespace NUMINAMATH_CALUDE_ninth_term_value_l2338_233822

/-- A geometric sequence with a₁ = 2 and a₅ = 18 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 5 = 18 ∧ ∀ n m : ℕ, a (n + m) = a n * a m

theorem ninth_term_value (a : ℕ → ℝ) (h : geometric_sequence a) : a 9 = 162 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_value_l2338_233822


namespace NUMINAMATH_CALUDE_divisibility_condition_l2338_233863

theorem divisibility_condition (a b : ℕ+) :
  (a.val^2 + b.val^2 - a.val - b.val + 1) % (a.val * b.val) = 0 ↔ a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2338_233863


namespace NUMINAMATH_CALUDE_soldiers_divisible_by_six_l2338_233835

theorem soldiers_divisible_by_six (b : ℕ+) : 
  ∃ k : ℕ, b + 3 * b ^ 2 + 2 * b ^ 3 = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_soldiers_divisible_by_six_l2338_233835


namespace NUMINAMATH_CALUDE_ken_cycling_distance_l2338_233825

/-- Ken's cycling speed in miles per hour when it's raining -/
def rain_speed : ℝ := 30 * 3

/-- Ken's cycling speed in miles per hour when it's snowing -/
def snow_speed : ℝ := 10 * 3

/-- Number of rainy days in a week -/
def rainy_days : ℕ := 3

/-- Number of snowy days in a week -/
def snowy_days : ℕ := 4

/-- Hours Ken cycles per day -/
def hours_per_day : ℝ := 1

theorem ken_cycling_distance :
  rain_speed * rainy_days * hours_per_day + snow_speed * snowy_days * hours_per_day = 390 := by
  sorry

end NUMINAMATH_CALUDE_ken_cycling_distance_l2338_233825


namespace NUMINAMATH_CALUDE_unique_solution_E_l2338_233804

/-- Definition of the function E -/
def E (a b c : ℝ) : ℝ := a * b^2 + c

/-- Theorem stating that 3/8 is the unique solution to E(a, 3, 12) = E(a, 5, 6) -/
theorem unique_solution_E :
  ∃! a : ℝ, E a 3 12 = E a 5 6 ∧ a = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_E_l2338_233804


namespace NUMINAMATH_CALUDE_rose_price_is_seven_l2338_233858

/-- Calculates the price per rose given the initial number of roses,
    remaining number of roses, and total earnings. -/
def price_per_rose (initial : ℕ) (remaining : ℕ) (earnings : ℕ) : ℚ :=
  earnings / (initial - remaining)

/-- Proves that the price per rose is 7 dollars given the problem conditions. -/
theorem rose_price_is_seven :
  price_per_rose 9 4 35 = 7 := by
  sorry

end NUMINAMATH_CALUDE_rose_price_is_seven_l2338_233858


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2338_233844

/-- The roots of the quadratic equation ax² + 4ax + c = 0 are equal if and only if c = 4a, given that a ≠ 0 -/
theorem quadratic_equal_roots (a c : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, ∀ y : ℝ, a * y^2 + 4 * a * y + c = 0 ↔ y = x) ↔ c = 4 * a :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2338_233844


namespace NUMINAMATH_CALUDE_equilateral_triangle_ratio_l2338_233880

/-- Given two equilateral triangles with side lengths A and a, and altitudes h_A and h_a respectively,
    if h_A = 2h_a, then the ratio of their perimeters is equal to the ratio of their altitudes. -/
theorem equilateral_triangle_ratio (A a h_A h_a : ℝ) 
  (h_positive : h_a > 0)
  (h_eq : h_A = 2 * h_a) :
  3 * A / (3 * a) = h_A / h_a := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_ratio_l2338_233880


namespace NUMINAMATH_CALUDE_workshop_workers_count_l2338_233898

/-- Proves that the total number of workers in a workshop is 21, given specific salary conditions. -/
theorem workshop_workers_count :
  let total_average_salary : ℕ := 8000
  let technician_count : ℕ := 7
  let technician_average_salary : ℕ := 12000
  let non_technician_average_salary : ℕ := 6000
  ∃ (total_workers : ℕ),
    total_workers * total_average_salary = 
      technician_count * technician_average_salary + 
      (total_workers - technician_count) * non_technician_average_salary ∧
    total_workers = 21 := by
sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l2338_233898


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2338_233826

-- Define the repeating decimals
def repeating_2 : ℚ := 2 / 9
def repeating_03 : ℚ := 3 / 99
def repeating_0004 : ℚ := 4 / 9999
def repeating_00005 : ℚ := 5 / 99999

-- State the theorem
theorem sum_of_repeating_decimals :
  repeating_2 + repeating_03 + repeating_0004 + repeating_00005 = 56534 / 99999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2338_233826


namespace NUMINAMATH_CALUDE_discount_savings_difference_l2338_233840

def shoe_price : ℕ := 50
def discount_a_percent : ℕ := 40
def discount_b_amount : ℕ := 15

def cost_with_discount_a : ℕ := shoe_price + (shoe_price - (shoe_price * discount_a_percent / 100))
def cost_with_discount_b : ℕ := shoe_price + (shoe_price - discount_b_amount)

theorem discount_savings_difference : 
  cost_with_discount_b - cost_with_discount_a = 5 := by
  sorry

end NUMINAMATH_CALUDE_discount_savings_difference_l2338_233840


namespace NUMINAMATH_CALUDE_ratio_problem_l2338_233830

theorem ratio_problem (x : ℝ) : 
  (5 : ℝ) * x = 60 → x = 12 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2338_233830


namespace NUMINAMATH_CALUDE_mean_median_difference_l2338_233873

/-- Represents the frequency distribution of days missed --/
def frequency_distribution : List (Nat × Nat) :=
  [(0, 4), (1, 2), (2, 5), (3, 2), (4, 3), (5, 4)]

/-- Total number of students --/
def total_students : Nat := 20

/-- Calculates the median of the dataset --/
def median (data : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- Calculates the mean of the dataset --/
def mean (data : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- The main theorem to prove --/
theorem mean_median_difference :
  (mean frequency_distribution total_students) - 
  (median frequency_distribution total_students) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2338_233873


namespace NUMINAMATH_CALUDE_magic_square_solution_l2338_233807

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a11 : ℚ
  a12 : ℚ
  a13 : ℚ
  a21 : ℚ
  a22 : ℚ
  a23 : ℚ
  a31 : ℚ
  a32 : ℚ
  a33 : ℚ
  sum_property : ∃ s : ℚ,
    a11 + a12 + a13 = s ∧
    a21 + a22 + a23 = s ∧
    a31 + a32 + a33 = s ∧
    a11 + a21 + a31 = s ∧
    a12 + a22 + a32 = s ∧
    a13 + a23 + a33 = s ∧
    a11 + a22 + a33 = s ∧
    a13 + a22 + a31 = s

/-- The theorem stating that y = 168.5 in the given magic square -/
theorem magic_square_solution :
  ∀ (ms : MagicSquare),
    ms.a11 = ms.a11 ∧  -- y (unknown)
    ms.a12 = 25 ∧
    ms.a13 = 81 ∧
    ms.a21 = 4 →
    ms.a11 = 168.5 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_solution_l2338_233807


namespace NUMINAMATH_CALUDE_intersection_complement_equals_three_l2338_233882

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {0, 1, 2}
def N : Set Int := {0, 1, 2, 3}

theorem intersection_complement_equals_three : (U \ M) ∩ N = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_three_l2338_233882


namespace NUMINAMATH_CALUDE_cafe_latte_cost_correct_l2338_233849

/-- Represents the cost of a cafe latte -/
def cafe_latte_cost : ℝ := 1.50

/-- Represents the cost of a cappuccino -/
def cappuccino_cost : ℝ := 2

/-- Represents the cost of an iced tea -/
def iced_tea_cost : ℝ := 3

/-- Represents the cost of an espresso -/
def espresso_cost : ℝ := 1

/-- Represents the number of cappuccinos Sandy ordered -/
def num_cappuccinos : ℕ := 3

/-- Represents the number of iced teas Sandy ordered -/
def num_iced_teas : ℕ := 2

/-- Represents the number of cafe lattes Sandy ordered -/
def num_lattes : ℕ := 2

/-- Represents the number of espressos Sandy ordered -/
def num_espressos : ℕ := 2

/-- Represents the amount Sandy paid -/
def amount_paid : ℝ := 20

/-- Represents the change Sandy received -/
def change_received : ℝ := 3

theorem cafe_latte_cost_correct :
  cafe_latte_cost * num_lattes +
  cappuccino_cost * num_cappuccinos +
  iced_tea_cost * num_iced_teas +
  espresso_cost * num_espressos =
  amount_paid - change_received :=
by sorry

end NUMINAMATH_CALUDE_cafe_latte_cost_correct_l2338_233849


namespace NUMINAMATH_CALUDE_combinatorial_identity_l2338_233869

theorem combinatorial_identity (n k : ℕ) (h : k ≤ n) :
  Nat.choose (n + 1) k = Nat.choose n k + Nat.choose n (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_identity_l2338_233869


namespace NUMINAMATH_CALUDE_bookshelf_count_l2338_233803

theorem bookshelf_count (books_per_shelf : ℕ) (total_books : ℕ) (shelf_count : ℕ) : 
  books_per_shelf = 15 → 
  total_books = 2250 → 
  shelf_count * books_per_shelf = total_books → 
  shelf_count = 150 := by
sorry

end NUMINAMATH_CALUDE_bookshelf_count_l2338_233803


namespace NUMINAMATH_CALUDE_distance_point_to_line_l2338_233843

def vector_AB : Fin 3 → ℝ := ![1, 1, 2]
def vector_AC : Fin 3 → ℝ := ![2, 1, 1]

theorem distance_point_to_line :
  let distance := Real.sqrt (6 - (5 * Real.sqrt 6 / 6) ^ 2)
  distance = Real.sqrt 66 / 6 := by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l2338_233843


namespace NUMINAMATH_CALUDE_quadruple_primes_l2338_233866

theorem quadruple_primes (p q r : Nat) (n : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ n > 0 ∧ p^2 = q^2 + r^n →
  ((p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4)) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_primes_l2338_233866


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_even_sides_l2338_233834

theorem right_triangle_consecutive_even_sides (a b c : ℕ) : 
  (∃ x : ℕ, a = x - 2 ∧ b = x ∧ c = x + 2) →  -- sides are consecutive even numbers
  (a^2 + b^2 = c^2) →                        -- right-angled triangle
  c = 10                                     -- hypotenuse length is 10
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_even_sides_l2338_233834


namespace NUMINAMATH_CALUDE_faye_coloring_books_l2338_233845

def coloring_books_problem (initial_books : ℝ) (first_giveaway : ℝ) (second_giveaway : ℝ) : Prop :=
  initial_books - first_giveaway - second_giveaway = 11.0

theorem faye_coloring_books :
  coloring_books_problem 48.0 34.0 3.0 := by
  sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l2338_233845


namespace NUMINAMATH_CALUDE_number_of_bowls_l2338_233837

theorem number_of_bowls (n : ℕ) : n > 0 → (96 : ℝ) / n = 6 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_bowls_l2338_233837


namespace NUMINAMATH_CALUDE_color_tv_price_l2338_233811

theorem color_tv_price : ∃ (x : ℝ), x > 0 ∧ (1.4 * x * 0.8) - x = 144 ∧ x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_color_tv_price_l2338_233811


namespace NUMINAMATH_CALUDE_two_true_propositions_l2338_233818

theorem two_true_propositions :
  let original := ∀ a : ℝ, a > -3 → a > 0
  let converse := ∀ a : ℝ, a > 0 → a > -3
  let inverse := ∀ a : ℝ, a ≤ -3 → a ≤ 0
  let contrapositive := ∀ a : ℝ, a ≤ 0 → a ≤ -3
  (¬original ∧ converse ∧ inverse ∧ ¬contrapositive) :=
by sorry

end NUMINAMATH_CALUDE_two_true_propositions_l2338_233818


namespace NUMINAMATH_CALUDE_min_value_fraction_l2338_233897

theorem min_value_fraction (a b : ℕ) (h1 : a > b) (h2 : b > 0) :
  (2 * a + b : ℚ) / (a - 2 * b) + (a - 2 * b : ℚ) / (2 * a + b) ≥ 50 / 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2338_233897


namespace NUMINAMATH_CALUDE_game_a_vs_game_b_l2338_233857

def coin_prob_heads : ℚ := 2/3
def coin_prob_tails : ℚ := 1/3

def game_a_win (p : ℚ) : ℚ := p^4 + (1-p)^4

def game_b_win (p : ℚ) : ℚ := p^3 * (1-p) + (1-p)^3 * p

theorem game_a_vs_game_b :
  game_a_win coin_prob_heads - game_b_win coin_prob_heads = 7/81 :=
by sorry

end NUMINAMATH_CALUDE_game_a_vs_game_b_l2338_233857


namespace NUMINAMATH_CALUDE_quaternary_201_is_33_l2338_233810

def quaternary_to_decimal (q : ℕ) : ℕ :=
  (q / 100) * 4^2 + ((q / 10) % 10) * 4^1 + (q % 10) * 4^0

theorem quaternary_201_is_33 : quaternary_to_decimal 201 = 33 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_201_is_33_l2338_233810


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l2338_233889

/-- Given two rectangles with equal areas, where one rectangle has dimensions 5 by 24 inches
    and the other has a length of 3 inches, prove that the width of the second rectangle is 40 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length : ℝ)
    (carol_area jordan_area : ℝ) (h1 : carol_length = 5)
    (h2 : carol_width = 24) (h3 : jordan_length = 3)
    (h4 : carol_area = carol_length * carol_width)
    (h5 : jordan_area = jordan_length * (jordan_area / jordan_length))
    (h6 : carol_area = jordan_area) :
  jordan_area / jordan_length = 40 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l2338_233889


namespace NUMINAMATH_CALUDE_inspection_result_l2338_233806

/-- Given a set of products and a selection for inspection, 
    we define the total number of items and the sample size. -/
def inspection_setup (total_products : ℕ) (selected : ℕ) : 
  (ℕ × ℕ) :=
  (total_products, selected)

/-- Theorem stating that for 50 products with 10 selected,
    the total number of items is 50 and the sample size is 10. -/
theorem inspection_result : 
  inspection_setup 50 10 = (50, 10) := by
  sorry

end NUMINAMATH_CALUDE_inspection_result_l2338_233806


namespace NUMINAMATH_CALUDE_reflection_path_exists_l2338_233829

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a path from A to B with two reflections in a triangle -/
structure ReflectionPath (t : Triangle) where
  P : Point -- Point of reflection on side BC
  Q : Point -- Point of reflection on side CA

/-- Angle at vertex C of a triangle -/
def angle_C (t : Triangle) : ℝ := sorry

/-- Theorem stating the condition for the existence of a reflection path -/
theorem reflection_path_exists (t : Triangle) : 
  (∃ path : ReflectionPath t, True) ↔ (π/4 < angle_C t ∧ angle_C t < π/3) := by sorry

end NUMINAMATH_CALUDE_reflection_path_exists_l2338_233829


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l2338_233850

theorem unique_solution_cube_equation :
  ∀ x y z : ℤ, x^3 - 3*y^3 - 9*z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l2338_233850


namespace NUMINAMATH_CALUDE_largest_number_in_l_pattern_l2338_233842

/-- Represents the L-shaped pattern in the number arrangement --/
structure LPattern where
  largest : ℕ
  second : ℕ
  third : ℕ

/-- The sum of numbers in the L-shaped pattern is 2015 --/
def sum_is_2015 (p : LPattern) : Prop :=
  p.largest + p.second + p.third = 2015

/-- The L-shaped pattern follows the specific arrangement described --/
def valid_arrangement (p : LPattern) : Prop :=
  (p.second = p.largest - 6 ∧ p.third = p.largest - 7) ∨
  (p.second = p.largest - 7 ∧ p.third = p.largest - 8) ∨
  (p.second = p.largest - 1 ∧ p.third = p.largest - 7) ∨
  (p.second = p.largest - 1 ∧ p.third = p.largest - 8)

theorem largest_number_in_l_pattern :
  ∀ p : LPattern, sum_is_2015 p → valid_arrangement p → p.largest = 676 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_l_pattern_l2338_233842


namespace NUMINAMATH_CALUDE_machine_work_time_l2338_233864

theorem machine_work_time (x : ℝ) : x > 0 → 
  (1 / (x + 8) + 2 / (3 * x) + 1 / (2 * x) = 1 / (2 * x)) → 
  x = (-1 + Real.sqrt 97) / 6 :=
by sorry

end NUMINAMATH_CALUDE_machine_work_time_l2338_233864


namespace NUMINAMATH_CALUDE_tomatoes_picked_l2338_233878

/-- Calculates the number of tomatoes picked by a farmer -/
theorem tomatoes_picked (initial_tomatoes : ℕ) (initial_potatoes : ℕ) (final_total : ℕ) : 
  initial_tomatoes - (final_total - initial_potatoes) = 53 :=
by
  sorry

end NUMINAMATH_CALUDE_tomatoes_picked_l2338_233878


namespace NUMINAMATH_CALUDE_first_solution_percentage_l2338_233874

-- Define the volumes and percentages
def volume_first : ℝ := 40
def volume_second : ℝ := 60
def percent_second : ℝ := 0.7
def percent_final : ℝ := 0.5
def total_volume : ℝ := 100

-- Define the theorem
theorem first_solution_percentage :
  ∃ (percent_first : ℝ),
    volume_first * percent_first + volume_second * percent_second = total_volume * percent_final ∧
    percent_first = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_first_solution_percentage_l2338_233874


namespace NUMINAMATH_CALUDE_no_solution_composite_l2338_233802

/-- Two polynomials P and Q that satisfy the given conditions -/
class SpecialPolynomials (P Q : ℝ → ℝ) : Prop where
  commutativity : ∀ x : ℝ, P (Q x) = Q (P x)
  no_solution : ∀ x : ℝ, P x ≠ Q x

/-- Theorem stating that if P and Q satisfy the special conditions,
    then P(P(x)) = Q(Q(x)) has no solutions -/
theorem no_solution_composite 
  (P Q : ℝ → ℝ) [SpecialPolynomials P Q] :
  ∀ x : ℝ, P (P x) ≠ Q (Q x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_composite_l2338_233802


namespace NUMINAMATH_CALUDE_semicircle_curve_length_l2338_233832

open Real

/-- The length of the curve traced by point D in a semicircle configuration --/
theorem semicircle_curve_length (k : ℝ) (h : k > 0) :
  ∃ (curve_length : ℝ),
    (∀ (θ : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 →
      let C : ℝ × ℝ := (cos (2 * θ), sin (2 * θ))
      let D : ℝ × ℝ := (cos (2 * θ) + k * sin (2 * θ), sin (2 * θ) + k * (1 - cos (2 * θ)))
      (D.1 ^ 2 + (D.2 - k) ^ 2 = 1 + k ^ 2) ∧
      (k * D.1 + D.2 ≥ k)) →
    curve_length = π * sqrt (1 + k ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_semicircle_curve_length_l2338_233832


namespace NUMINAMATH_CALUDE_floor_abs_neq_abs_floor_exists_floor_diff_lt_floor_eq_implies_diff_lt_one_floor_inequality_solution_set_l2338_233847

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Proposition A (negation)
theorem floor_abs_neq_abs_floor : ∃ x : ℝ, floor (|x|) ≠ |floor x| :=
sorry

-- Proposition B
theorem exists_floor_diff_lt : ∃ x y : ℝ, floor (x - y) < floor x - floor y :=
sorry

-- Proposition C
theorem floor_eq_implies_diff_lt_one :
  ∀ x y : ℝ, floor x = floor y → x - y < 1 :=
sorry

-- Proposition D
theorem floor_inequality_solution_set :
  {x : ℝ | 2 * (floor x)^2 - floor x - 3 ≥ 0} = {x : ℝ | x < 0 ∨ x ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_floor_abs_neq_abs_floor_exists_floor_diff_lt_floor_eq_implies_diff_lt_one_floor_inequality_solution_set_l2338_233847


namespace NUMINAMATH_CALUDE_marble_distribution_l2338_233885

theorem marble_distribution (total_marbles : ℕ) (initial_group : ℕ) (joining_group : ℕ) : 
  total_marbles = 312 →
  initial_group = 24 →
  (total_marbles / initial_group : ℕ) = ((total_marbles / (initial_group + joining_group)) + 1 : ℕ) →
  joining_group = 2 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l2338_233885


namespace NUMINAMATH_CALUDE_congress_room_arrangement_l2338_233800

/-- A type representing delegates -/
def Delegate : Type := ℕ

/-- A relation representing the ability to communicate directly -/
def CanCommunicate : Delegate → Delegate → Prop := sorry

/-- The total number of delegates -/
def totalDelegates : ℕ := 1000

theorem congress_room_arrangement 
  (delegates : Finset Delegate) 
  (h_count : delegates.card = totalDelegates)
  (h_communication : ∀ (a b c : Delegate), a ∈ delegates → b ∈ delegates → c ∈ delegates → 
    (CanCommunicate a b ∨ CanCommunicate b c ∨ CanCommunicate a c)) :
  ∃ (pairs : List (Delegate × Delegate)), 
    (∀ (pair : Delegate × Delegate), pair ∈ pairs → CanCommunicate pair.1 pair.2) ∧ 
    (pairs.length = totalDelegates / 2) ∧
    (∀ (d : Delegate), d ∈ delegates ↔ (∃ (pair : Delegate × Delegate), pair ∈ pairs ∧ (d = pair.1 ∨ d = pair.2))) :=
sorry

end NUMINAMATH_CALUDE_congress_room_arrangement_l2338_233800


namespace NUMINAMATH_CALUDE_tetrahedron_existence_l2338_233893

-- Define a tetrahedron type
structure Tetrahedron :=
  (edges : Fin 6 → ℝ)

-- Define the conditions for configuration (a)
def config_a (t : Tetrahedron) : Prop :=
  (∃ i j : Fin 6, i ≠ j ∧ t.edges i < 0.01 ∧ t.edges j < 0.01) ∧
  (∀ k : Fin 6, (t.edges k ≤ 0.01) ∨ (t.edges k > 1000))

-- Define the conditions for configuration (b)
def config_b (t : Tetrahedron) : Prop :=
  (∃ i j k l : Fin 6, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    t.edges i < 0.01 ∧ t.edges j < 0.01 ∧ t.edges k < 0.01 ∧ t.edges l < 0.01) ∧
  (∀ m : Fin 6, (t.edges m < 0.01) ∨ (t.edges m > 1000))

-- Theorem statements
theorem tetrahedron_existence :
  (∃ t : Tetrahedron, config_a t) ∧ (¬ ∃ t : Tetrahedron, config_b t) :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_existence_l2338_233893


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2338_233817

def U : Set ℤ := {1, 2, 3, 4, 5, 6, 7}

def M : Set ℤ := {x | x^2 - 6*x + 5 ≤ 0 ∧ x ∈ U}

theorem complement_of_M_in_U :
  U \ M = {6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2338_233817


namespace NUMINAMATH_CALUDE_x_value_proof_l2338_233808

theorem x_value_proof (x : ℝ) (h : x^2 * 8^3 / 256 = 450) : x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2338_233808


namespace NUMINAMATH_CALUDE_candy_jar_problem_l2338_233862

theorem candy_jar_problem (total : ℕ) (red : ℕ) (blue : ℕ) : 
  total = 3409 → red = 145 → blue = total - red → blue = 3264 := by
  sorry

end NUMINAMATH_CALUDE_candy_jar_problem_l2338_233862


namespace NUMINAMATH_CALUDE_point_movement_to_y_axis_l2338_233839

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem statement
theorem point_movement_to_y_axis (a : ℝ) :
  let P : Point := ⟨a + 1, a⟩
  let P₁ : Point := ⟨P.x + 3, P.y⟩
  P₁.x = 0 → P = ⟨-3, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_movement_to_y_axis_l2338_233839


namespace NUMINAMATH_CALUDE_ship_journey_l2338_233887

theorem ship_journey (D : ℝ) (speed : ℝ) (h1 : D > 0) (h2 : speed = 30) :
  D / 2 - 200 = D / 3 →
  D = 1200 ∧ (D / 2) / speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_ship_journey_l2338_233887


namespace NUMINAMATH_CALUDE_smallest_sum_for_equation_l2338_233865

theorem smallest_sum_for_equation : ∃ (a b : ℕ+), 
  (2^10 * 7^4 : ℕ) = a^(b:ℕ) ∧ 
  (∀ (c d : ℕ+), (2^10 * 7^4 : ℕ) = c^(d:ℕ) → a + b ≤ c + d) ∧
  a + b = 1570 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_for_equation_l2338_233865


namespace NUMINAMATH_CALUDE_waddle_hop_difference_l2338_233823

/-- The number of hops Winston takes between consecutive markers -/
def winston_hops : ℕ := 88

/-- The number of waddles Petra takes between consecutive markers -/
def petra_waddles : ℕ := 24

/-- The total number of markers -/
def total_markers : ℕ := 81

/-- The total distance in feet between the first and last marker -/
def total_distance : ℕ := 10560

/-- The length of Petra's waddle in feet -/
def petra_waddle_length : ℚ := total_distance / (petra_waddles * (total_markers - 1))

/-- The length of Winston's hop in feet -/
def winston_hop_length : ℚ := total_distance / (winston_hops * (total_markers - 1))

/-- The difference between Petra's waddle length and Winston's hop length -/
def length_difference : ℚ := petra_waddle_length - winston_hop_length

theorem waddle_hop_difference : length_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_waddle_hop_difference_l2338_233823


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2338_233816

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem seventh_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_positive : ∀ n, a n > 0) 
  (h_fifth : a 5 = 16) 
  (h_ninth : a 9 = 4) : 
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2338_233816


namespace NUMINAMATH_CALUDE_figure_100_cubes_l2338_233892

-- Define the sequence of unit cubes for the first four figures
def cube_sequence : Fin 4 → ℕ
  | 0 => 1
  | 1 => 8
  | 2 => 27
  | 3 => 64

-- Define the general formula for the number of cubes in figure n
def num_cubes (n : ℕ) : ℕ := n^3

-- Theorem statement
theorem figure_100_cubes :
  (∀ k : Fin 4, cube_sequence k = num_cubes k) →
  num_cubes 100 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_figure_100_cubes_l2338_233892


namespace NUMINAMATH_CALUDE_jeff_shelter_cats_l2338_233819

/-- Calculates the number of cats in Jeff's shelter after a series of events -/
def cats_in_shelter (initial : ℕ) (monday_found : ℕ) (tuesday_found : ℕ) (wednesday_adopted : ℕ) : ℕ :=
  initial + monday_found + tuesday_found - wednesday_adopted

/-- Theorem stating the number of cats in Jeff's shelter after the given events -/
theorem jeff_shelter_cats : cats_in_shelter 20 2 1 6 = 17 := by
  sorry

#eval cats_in_shelter 20 2 1 6

end NUMINAMATH_CALUDE_jeff_shelter_cats_l2338_233819


namespace NUMINAMATH_CALUDE_ladder_height_proof_l2338_233820

def ceiling_height : ℝ := 300
def fixture_below_ceiling : ℝ := 15
def alice_height : ℝ := 170
def alice_normal_reach : ℝ := 55
def extra_reach_needed : ℝ := 5

theorem ladder_height_proof :
  let fixture_height := ceiling_height - fixture_below_ceiling
  let total_reach_needed := fixture_height
  let alice_max_reach := alice_height + alice_normal_reach + extra_reach_needed
  let ladder_height := total_reach_needed - alice_max_reach
  ladder_height = 60 := by sorry

end NUMINAMATH_CALUDE_ladder_height_proof_l2338_233820


namespace NUMINAMATH_CALUDE_twentieth_triangular_number_l2338_233836

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 20th triangular number is 210 -/
theorem twentieth_triangular_number : triangular_number 20 = 210 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_triangular_number_l2338_233836


namespace NUMINAMATH_CALUDE_sum_of_ten_numbers_l2338_233884

theorem sum_of_ten_numbers (numbers : Finset ℕ) (group_of_ten : Finset ℕ) (group_of_207 : Finset ℕ) :
  numbers = Finset.range 217 →
  numbers = group_of_ten ∪ group_of_207 →
  group_of_ten.card = 10 →
  group_of_207.card = 207 →
  group_of_ten ∩ group_of_207 = ∅ →
  (Finset.sum group_of_ten id) / 10 = (Finset.sum group_of_207 id) / 207 →
  Finset.sum group_of_ten id = 1090 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_ten_numbers_l2338_233884


namespace NUMINAMATH_CALUDE_factorial_difference_l2338_233890

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2338_233890


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2338_233894

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, x^2 - x < 0 → -1 < x ∧ x < 1) ∧
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ ¬(x^2 - x < 0)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2338_233894


namespace NUMINAMATH_CALUDE_fraction_equality_l2338_233846

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : a / b = (2 * a) / (2 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2338_233846


namespace NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l2338_233809

-- Part 1: System of Equations
theorem solve_system_equations (x y : ℝ) :
  (3 * (y - 2) = x - 1) ∧ (2 * (x - 1) = 5 * y - 8) →
  x = 7 ∧ y = 4 := by sorry

-- Part 2: System of Inequalities
theorem solve_system_inequalities (x : ℝ) :
  (3 * x ≤ 2 * x + 3) ∧ ((x + 1) / 6 - 1 < (2 * x + 2) / 3) ↔
  -3 < x ∧ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l2338_233809


namespace NUMINAMATH_CALUDE_smallest_area_right_triangle_l2338_233876

theorem smallest_area_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area1 := a * b / 2
  let c := Real.sqrt (a^2 + b^2)
  let area2 := a * Real.sqrt (b^2 - a^2) / 2
  area2 < area1 := by sorry

end NUMINAMATH_CALUDE_smallest_area_right_triangle_l2338_233876


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l2338_233877

theorem simplify_and_ratio : ∃ (a b : ℤ), 
  (∀ k, (6 * k + 12) / 6 = a * k + b) ∧ 
  (a : ℚ) / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l2338_233877


namespace NUMINAMATH_CALUDE_min_value_of_f_l2338_233855

theorem min_value_of_f (x : ℝ) : 1 / Real.sqrt (x^2 + 2) + Real.sqrt (x^2 + 2) ≥ 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2338_233855


namespace NUMINAMATH_CALUDE_sphere_hemisphere_cone_volume_ratio_l2338_233856

/-- The ratio of the volume of a sphere to the combined volume of a hemisphere and a cone -/
theorem sphere_hemisphere_cone_volume_ratio (r : ℝ) (hr : r > 0) : 
  (4 / 3 * π * r^3) / ((1 / 2 * 4 / 3 * π * (3 * r)^3) + (1 / 3 * π * r^2 * (2 * r))) = 1 / 14 := by
  sorry

#check sphere_hemisphere_cone_volume_ratio

end NUMINAMATH_CALUDE_sphere_hemisphere_cone_volume_ratio_l2338_233856


namespace NUMINAMATH_CALUDE_game_winning_strategy_l2338_233861

def game_move (k : ℕ) : Set ℕ := {k + 1, 2 * k}

def is_winning_position (n : ℕ) : Prop :=
  ∃ (k c : ℕ), n = 2^(2*k+1) + 2*c ∧ c < 2^k

theorem game_winning_strategy (n : ℕ) (h : n > 1) :
  (∀ k, k ∈ game_move 2 → k ≤ n) →
  (is_winning_position n ↔ 
    ∃ (strategy : ℕ → ℕ), 
      (∀ m, m < n → strategy m ∈ game_move m) ∧
      (∀ m, m < n → strategy (strategy m) > n)) :=
sorry

end NUMINAMATH_CALUDE_game_winning_strategy_l2338_233861


namespace NUMINAMATH_CALUDE_simplify_fraction_l2338_233812

theorem simplify_fraction : (222 : ℚ) / 8888 * 22 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2338_233812


namespace NUMINAMATH_CALUDE_company_j_payroll_company_j_payroll_correct_l2338_233867

/-- Calculates the total monthly payroll for factory workers given the conditions of Company J. -/
theorem company_j_payroll (factory_workers : ℕ) (office_workers : ℕ) 
  (office_payroll : ℕ) (salary_difference : ℕ) : ℕ :=
  let factory_workers := 15
  let office_workers := 30
  let office_payroll := 75000
  let salary_difference := 500
  30000

theorem company_j_payroll_correct : 
  company_j_payroll 15 30 75000 500 = 30000 := by sorry

end NUMINAMATH_CALUDE_company_j_payroll_company_j_payroll_correct_l2338_233867


namespace NUMINAMATH_CALUDE_system_solution_unique_l2338_233888

theorem system_solution_unique (x y : ℝ) : 
  (4 * x + 3 * y = 11 ∧ 4 * x - 3 * y = 5) ↔ (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2338_233888


namespace NUMINAMATH_CALUDE_certain_number_problem_l2338_233852

theorem certain_number_problem (x : ℝ) : ((x + 20) * 2) / 2 - 2 = 88 / 2 ↔ x = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2338_233852


namespace NUMINAMATH_CALUDE_expression_simplification_l2338_233899

theorem expression_simplification :
  (3 * Real.sqrt 12) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 6) = Real.sqrt 3 + 2 * Real.sqrt 2 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2338_233899


namespace NUMINAMATH_CALUDE_valid_colorings_2x9_board_l2338_233838

/-- Represents the number of columns in the board -/
def n : ℕ := 9

/-- Represents the number of colors available -/
def num_colors : ℕ := 3

/-- Represents the number of ways to color the first column -/
def first_column_colorings : ℕ := num_colors * (num_colors - 1)

/-- Represents the number of ways to color each subsequent column -/
def subsequent_column_colorings : ℕ := num_colors - 1

/-- Theorem stating the number of valid colorings for a 2 × 9 board -/
theorem valid_colorings_2x9_board :
  first_column_colorings * subsequent_column_colorings^(n - 1) = 39366 := by
  sorry

end NUMINAMATH_CALUDE_valid_colorings_2x9_board_l2338_233838


namespace NUMINAMATH_CALUDE_log_properties_l2338_233891

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- Main theorem
theorem log_properties (b : ℝ) (x : ℝ) (y : ℝ) 
    (h1 : b > 1) 
    (h2 : y = log b (x^2)) :
  (x = 1 → y = 0) ∧ 
  (x = -b → y = 2) ∧ 
  (-1 < x ∧ x < 1 → y < 0) := by
  sorry

end NUMINAMATH_CALUDE_log_properties_l2338_233891


namespace NUMINAMATH_CALUDE_treewidth_bound_for_grid_free_graphs_l2338_233871

/-- A k-grid of order h in a graph -/
def kGridOfOrderH (G : Graph) (k h : ℕ) : Prop := sorry

/-- The treewidth of a graph -/
def treewidth (G : Graph) : ℕ := sorry

/-- Theorem: If a graph G does not contain a k-grid of order h, then its treewidth is less than h + k - 1 -/
theorem treewidth_bound_for_grid_free_graphs
  (G : Graph) (h k : ℕ) (h_ge_k : h ≥ k) (k_ge_1 : k ≥ 1)
  (no_grid : ¬ kGridOfOrderH G k h) :
  treewidth G < h + k - 1 := by
  sorry

end NUMINAMATH_CALUDE_treewidth_bound_for_grid_free_graphs_l2338_233871


namespace NUMINAMATH_CALUDE_intersection_and_chord_length_l2338_233813

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 2)^2 = 4

-- Define the line l₃
def line_l₃ (x y : ℝ) : Prop :=
  4*x - 3*y - 1 = 0

-- Theorem statement
theorem intersection_and_chord_length :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    line_l₃ A.1 A.2 ∧
    line_l₃ B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_and_chord_length_l2338_233813


namespace NUMINAMATH_CALUDE_inscribed_triangle_sides_l2338_233805

/-- A triangle with an inscribed circle -/
structure InscribedTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first side of the triangle -/
  a : ℝ
  /-- The length of the second side of the triangle -/
  b : ℝ
  /-- The length of the third side of the triangle -/
  c : ℝ
  /-- The length of the first segment of side 'a' -/
  x : ℝ
  /-- The length of the second segment of side 'a' -/
  y : ℝ
  /-- The side 'a' is divided by the point of tangency -/
  side_division : a = x + y
  /-- All sides are positive -/
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  /-- All segments are positive -/
  pos_segments : 0 < x ∧ 0 < y

/-- Theorem about the sides of a triangle with an inscribed circle -/
theorem inscribed_triangle_sides (t : InscribedTriangle) 
  (h1 : t.r = 2)
  (h2 : t.x = 6)
  (h3 : t.y = 14) :
  (t.b = 7 ∧ t.c = 15) ∨ (t.b = 15 ∧ t.c = 7) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_sides_l2338_233805


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2338_233824

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X : Polynomial ℝ)^4 + (X : Polynomial ℝ)^2 - 5 = 
  (X^2 - 3) * q + (4 * X^2 - 5) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2338_233824


namespace NUMINAMATH_CALUDE_polynomial_sum_l2338_233833

theorem polynomial_sum (x : ℝ) (h1 : x^5 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2338_233833


namespace NUMINAMATH_CALUDE_high_school_students_l2338_233801

theorem high_school_students (total : ℕ) (ratio : ℕ) (mia zoe : ℕ) : 
  total = 2500 →
  ratio = 4 →
  mia = ratio * zoe →
  mia + zoe = total →
  mia = 2000 := by
sorry

end NUMINAMATH_CALUDE_high_school_students_l2338_233801


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2338_233831

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 3 * x - 4 * y = -7 ∧ 4 * x - 3 * y = 5 ∧ x = 41 / 7 ∧ y = 43 / 7 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2338_233831


namespace NUMINAMATH_CALUDE_power_division_equality_l2338_233896

theorem power_division_equality : 6^12 / 36^5 = 36 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l2338_233896


namespace NUMINAMATH_CALUDE_fencing_requirement_l2338_233851

/-- Calculates the required fencing for a rectangular field -/
theorem fencing_requirement (area : ℝ) (uncovered_side : ℝ) : 
  area > 0 → uncovered_side > 0 → area = uncovered_side * (area / uncovered_side) →
  uncovered_side + 2 * (area / uncovered_side) = 25 := by
  sorry

#check fencing_requirement

end NUMINAMATH_CALUDE_fencing_requirement_l2338_233851


namespace NUMINAMATH_CALUDE_present_ages_sum_l2338_233868

theorem present_ages_sum (A B S : ℕ) : 
  A + B = S →
  A = 2 * B →
  (A + 3) + (B + 3) = 66 →
  S = 60 := by
sorry

end NUMINAMATH_CALUDE_present_ages_sum_l2338_233868


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2338_233881

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 5) - (x^6 + 2 * x^5 + x^4 - x^3 + 7) =
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2338_233881


namespace NUMINAMATH_CALUDE_certain_number_proof_l2338_233827

theorem certain_number_proof : ∃ x : ℚ, 346 * x = 173 * 240 ∧ x = 120 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2338_233827


namespace NUMINAMATH_CALUDE_minimum_cents_to_win_l2338_233895

/-- Represents the state of the game -/
structure GameState where
  beans : ℕ
  cents : ℕ

/-- Applies the penny rule: multiply beans by 5 and add 1 cent -/
def applyPenny (state : GameState) : GameState :=
  { beans := state.beans * 5, cents := state.cents + 1 }

/-- Applies the nickel rule: add 1 bean and 5 cents -/
def applyNickel (state : GameState) : GameState :=
  { beans := state.beans + 1, cents := state.cents + 5 }

/-- Checks if the game is won -/
def isWinningState (state : GameState) : Prop :=
  state.beans > 2008 ∧ state.beans % 100 = 42

/-- Represents a sequence of moves in the game -/
inductive GameMove
  | penny
  | nickel

def applyMove (state : GameState) (move : GameMove) : GameState :=
  match move with
  | GameMove.penny => applyPenny state
  | GameMove.nickel => applyNickel state

def applyMoves (state : GameState) (moves : List GameMove) : GameState :=
  moves.foldl applyMove state

theorem minimum_cents_to_win :
  ∃ (moves : List GameMove),
    let finalState := applyMoves { beans := 0, cents := 0 } moves
    isWinningState finalState ∧
    finalState.cents = 35 ∧
    (∀ (otherMoves : List GameMove),
      let otherFinalState := applyMoves { beans := 0, cents := 0 } otherMoves
      isWinningState otherFinalState → otherFinalState.cents ≥ 35) :=
by sorry

end NUMINAMATH_CALUDE_minimum_cents_to_win_l2338_233895


namespace NUMINAMATH_CALUDE_sock_ratio_l2338_233853

theorem sock_ratio (total : ℕ) (blue : ℕ) (h1 : total = 180) (h2 : blue = 60) :
  (total - blue) / total = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_sock_ratio_l2338_233853


namespace NUMINAMATH_CALUDE_det_example_and_cube_diff_sum_of_cubes_given_conditions_complex_det_sum_given_conditions_l2338_233828

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Statement 1
theorem det_example_and_cube_diff : 
  det 5 4 8 9 = 13 ∧ ∀ a : ℝ, a^3 - 3*a^2 + 3*a - 1 = (a - 1)^3 := by sorry

-- Statement 2
theorem sum_of_cubes_given_conditions :
  ∀ x y : ℝ, x + y = 3 → x * y = 1 → x^3 + y^3 = 18 := by sorry

-- Statement 3
theorem complex_det_sum_given_conditions :
  ∀ x m n : ℝ, m = x - 1 → n = x + 2 → m * n = 5 →
  det m (3*m^2 + n^2) n (m^2 + 3*n^2) + det (m + n) (-2*n) n (m - n) = -8 := by sorry

end NUMINAMATH_CALUDE_det_example_and_cube_diff_sum_of_cubes_given_conditions_complex_det_sum_given_conditions_l2338_233828


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l2338_233814

-- Problem 1
theorem factorization_1 (a x y : ℝ) : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) := by
  sorry

-- Problem 2
theorem factorization_2 (x : ℝ) : x^3 - 8 * x^2 + 16 * x = x * (x - 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l2338_233814


namespace NUMINAMATH_CALUDE_P_intersect_Q_equals_Q_l2338_233841

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Theorem statement
theorem P_intersect_Q_equals_Q : P ∩ Q = Q := by
  sorry

end NUMINAMATH_CALUDE_P_intersect_Q_equals_Q_l2338_233841


namespace NUMINAMATH_CALUDE_casper_window_problem_l2338_233860

theorem casper_window_problem (total_windows : ℕ) (locked_windows : ℕ) : 
  total_windows = 8 → 
  locked_windows = 1 → 
  (total_windows - locked_windows) * (total_windows - locked_windows - 1) = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_casper_window_problem_l2338_233860


namespace NUMINAMATH_CALUDE_log_sum_equals_three_l2338_233872

theorem log_sum_equals_three : Real.log 4 / Real.log 10 + Real.log 25 / Real.log 10 + (-1/8)^0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_l2338_233872


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2338_233848

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 - a 4 - a 8 - a 12 + a 15 = 2) →
  (a 3 + a 13 = -4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2338_233848


namespace NUMINAMATH_CALUDE_b_net_share_is_1450_l2338_233879

/-- Represents the salary distribution ratio for employees A, B, C, and D. -/
def salary_ratio : Fin 4 → ℕ
  | 0 => 2  -- A
  | 1 => 3  -- B
  | 2 => 4  -- C
  | 3 => 6  -- D

/-- Represents the salary difference between D and C. -/
def salary_difference : ℕ := 700

/-- Represents the minimum wage requirement. -/
def minimum_wage : ℕ := 1000

/-- Represents the tax rates for different salary brackets. -/
def tax_rate (salary : ℕ) : ℚ :=
  if salary ≤ 1000 then 0
  else if salary ≤ 2000 then 1/10
  else if salary ≤ 3000 then 1/5
  else 3/10

/-- Represents the salary caps for each employee. -/
def salary_cap : Fin 4 → ℕ
  | 0 => 4000  -- A
  | 1 => 3500  -- B
  | 2 => 4500  -- C
  | 3 => 6000  -- D

/-- Calculates B's net share after tax deductions. -/
def b_net_share : ℕ := sorry

/-- Theorem stating that B's net share after tax deductions is $1450. -/
theorem b_net_share_is_1450 : b_net_share = 1450 := by sorry

end NUMINAMATH_CALUDE_b_net_share_is_1450_l2338_233879


namespace NUMINAMATH_CALUDE_roots_of_equation_l2338_233870

theorem roots_of_equation : ∀ x : ℝ, (x - 3)^2 = 25 ↔ x = 8 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2338_233870


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l2338_233815

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l2338_233815
