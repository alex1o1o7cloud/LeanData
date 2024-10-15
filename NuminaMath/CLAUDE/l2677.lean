import Mathlib

namespace NUMINAMATH_CALUDE_page_lines_increase_l2677_267771

/-- 
Given an original number of lines L in a page, 
if increasing the number of lines by 80 results in a 50% increase, 
then the new total number of lines is 240.
-/
theorem page_lines_increase (L : ℕ) : 
  (L + 80 = L + L / 2) → (L + 80 = 240) := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l2677_267771


namespace NUMINAMATH_CALUDE_hotel_rooms_booked_l2677_267732

theorem hotel_rooms_booked (single_room_cost double_room_cost total_revenue double_rooms : ℕ)
  (h1 : single_room_cost = 35)
  (h2 : double_room_cost = 60)
  (h3 : total_revenue = 14000)
  (h4 : double_rooms = 196)
  : ∃ single_rooms : ℕ, single_rooms + double_rooms = 260 ∧ 
    single_room_cost * single_rooms + double_room_cost * double_rooms = total_revenue := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_booked_l2677_267732


namespace NUMINAMATH_CALUDE_max_third_side_length_l2677_267766

theorem max_third_side_length (a b : ℝ) (ha : a = 5) (hb : b = 11) :
  ∀ c : ℝ, (c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) → c ≤ 15 :=
by
  sorry

end NUMINAMATH_CALUDE_max_third_side_length_l2677_267766


namespace NUMINAMATH_CALUDE_pencil_difference_l2677_267749

theorem pencil_difference (price : ℚ) (liam_count mia_count : ℕ) : 
  price > 0.01 →
  price * liam_count = 2.10 →
  price * mia_count = 2.82 →
  mia_count - liam_count = 12 := by
sorry

end NUMINAMATH_CALUDE_pencil_difference_l2677_267749


namespace NUMINAMATH_CALUDE_fraction_addition_equivalence_l2677_267760

theorem fraction_addition_equivalence (a b : ℤ) (h : b > 0) :
  ∀ x y : ℤ, y > 0 →
    (a / b + x / y = (a + x) / (b + y)) ↔
    ∃ k : ℤ, x = -a * k^2 ∧ y = b * k :=
by sorry

end NUMINAMATH_CALUDE_fraction_addition_equivalence_l2677_267760


namespace NUMINAMATH_CALUDE_product_of_half_and_two_thirds_l2677_267730

theorem product_of_half_and_two_thirds (x y : ℚ) : 
  x = 1/2 → y = 2/3 → x * y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_half_and_two_thirds_l2677_267730


namespace NUMINAMATH_CALUDE_distribute_4_3_l2677_267729

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    where each box must contain at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  if n < k then 0
  else if n = k then Nat.factorial k
  else sorry  -- Actual implementation would go here

/-- The theorem stating that distributing 4 distinct objects into 3 distinct boxes,
    where each box must contain at least one object, can be done in 60 ways. -/
theorem distribute_4_3 : distribute 4 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_distribute_4_3_l2677_267729


namespace NUMINAMATH_CALUDE_limit_of_function_l2677_267796

theorem limit_of_function : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
  0 < |x| ∧ |x| < δ → 
  |((1 + x * Real.sin x - Real.cos (2 * x)) / (Real.sin x)^2) - 3| < ε := by
sorry

end NUMINAMATH_CALUDE_limit_of_function_l2677_267796


namespace NUMINAMATH_CALUDE_no_real_c_solution_l2677_267761

/-- Given a polynomial x^2 + bx + c with exactly one real root and b = c + 2,
    prove that there are no real values of c that satisfy these conditions. -/
theorem no_real_c_solution (b c : ℝ) 
    (h1 : ∃! x : ℝ, x^2 + b*x + c = 0)  -- exactly one real root
    (h2 : b = c + 2) :                  -- condition b = c + 2
    False :=                            -- no real c satisfies the conditions
  sorry

end NUMINAMATH_CALUDE_no_real_c_solution_l2677_267761


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2677_267743

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x > 2}
def B : Set ℝ := {x : ℝ | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2677_267743


namespace NUMINAMATH_CALUDE_simplify_and_sum_exponents_l2677_267700

def simplify_cube_root (x y z : ℝ) : ℝ := (40 * x^5 * y^9 * z^14) ^ (1/3)

theorem simplify_and_sum_exponents (x y z : ℝ) :
  ∃ (a e : ℝ) (b c d f g h : ℕ),
    simplify_cube_root x y z = a * x^b * y^c * z^d * (e * x^f * y^g * z^h)^(1/3) ∧
    b + c + d = 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_sum_exponents_l2677_267700


namespace NUMINAMATH_CALUDE_average_speed_barney_schwinn_l2677_267706

/-- Proves that the average speed is 31 miles per hour given the problem conditions --/
theorem average_speed_barney_schwinn : 
  let initial_reading : ℕ := 2552
  let final_reading : ℕ := 2992
  let total_time : ℕ := 14
  let distance := final_reading - initial_reading
  let exact_speed := (distance : ℚ) / total_time
  Int.floor (exact_speed + 1/2) = 31 := by sorry

end NUMINAMATH_CALUDE_average_speed_barney_schwinn_l2677_267706


namespace NUMINAMATH_CALUDE_expression_evaluation_l2677_267773

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  (x^3 + 1) / x * (y^3 + 1) / y + (x^3 - 1) / y * (y^3 - 1) / x = 2 * x^2 * y^2 + 2 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2677_267773


namespace NUMINAMATH_CALUDE_largest_fraction_less_than_16_23_l2677_267798

def F : Set ℚ := {q : ℚ | ∃ m n : ℕ+, q = m / n ∧ m + n ≤ 2005}

theorem largest_fraction_less_than_16_23 :
  ∀ q ∈ F, q < 16/23 → q ≤ 816/1189 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_less_than_16_23_l2677_267798


namespace NUMINAMATH_CALUDE_min_value_implications_l2677_267726

theorem min_value_implications (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, |x + a| + |x - b| ≥ 2) : 
  (1 / a + 1 / b + 1 / (a * b) ≥ 3) ∧ 
  (∀ t : ℝ, Real.sin t ^ 4 / a + Real.cos t ^ 4 / b ≥ 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_implications_l2677_267726


namespace NUMINAMATH_CALUDE_four_row_arrangement_has_27_triangles_l2677_267742

/-- Represents a triangular arrangement of smaller triangles -/
structure TriangularArrangement where
  rows : ℕ

/-- Counts the number of small triangles in the arrangement -/
def count_small_triangles (arr : TriangularArrangement) : ℕ :=
  (arr.rows * (arr.rows + 1)) / 2

/-- Counts the number of medium triangles (made of 4 small triangles) -/
def count_medium_triangles (arr : TriangularArrangement) : ℕ :=
  if arr.rows ≥ 3 then
    ((arr.rows - 2) * (arr.rows - 1)) / 2
  else
    0

/-- Counts the number of large triangles (made of 9 small triangles) -/
def count_large_triangles (arr : TriangularArrangement) : ℕ :=
  if arr.rows ≥ 4 then
    (arr.rows - 3)
  else
    0

/-- Counts the total number of triangles in the arrangement -/
def total_triangles (arr : TriangularArrangement) : ℕ :=
  count_small_triangles arr + count_medium_triangles arr + count_large_triangles arr

/-- Theorem: In a triangular arrangement with 4 rows, there are 27 triangles in total -/
theorem four_row_arrangement_has_27_triangles :
  ∀ (arr : TriangularArrangement), arr.rows = 4 → total_triangles arr = 27 := by
  sorry

end NUMINAMATH_CALUDE_four_row_arrangement_has_27_triangles_l2677_267742


namespace NUMINAMATH_CALUDE_circle_area_increase_l2677_267708

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by sorry

end NUMINAMATH_CALUDE_circle_area_increase_l2677_267708


namespace NUMINAMATH_CALUDE_salary_problem_l2677_267711

/-- Proves that A's salary is $3750 given the conditions of the problem -/
theorem salary_problem (a b : ℝ) : 
  a + b = 5000 →
  0.05 * a = 0.15 * b →
  a = 3750 := by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l2677_267711


namespace NUMINAMATH_CALUDE_candy_spending_l2677_267718

/-- The fraction of a dollar spent on candy given initial quarters and remaining cents -/
def fraction_spent (initial_quarters : ℕ) (remaining_cents : ℕ) : ℚ :=
  (initial_quarters * 25 - remaining_cents) / 100

/-- Theorem stating that given 14 quarters initially and 300 cents remaining,
    the fraction of a dollar spent on candy is 1/2 -/
theorem candy_spending :
  fraction_spent 14 300 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_candy_spending_l2677_267718


namespace NUMINAMATH_CALUDE_square_d_perimeter_l2677_267703

def square_perimeter (side_length : ℝ) : ℝ := 4 * side_length

def square_area (side_length : ℝ) : ℝ := side_length ^ 2

theorem square_d_perimeter (perimeter_c : ℝ) (h1 : perimeter_c = 32) :
  let side_c := perimeter_c / 4
  let area_c := square_area side_c
  let area_d := area_c / 3
  let side_d := Real.sqrt area_d
  square_perimeter side_d = (32 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_square_d_perimeter_l2677_267703


namespace NUMINAMATH_CALUDE_great_pyramid_sum_height_width_l2677_267735

/-- The Great Pyramid of Giza's dimensions -/
def great_pyramid (H W : ℕ) : Prop :=
  H = 500 + 20 ∧ W = H + 234

/-- Theorem: The sum of the height and width of the Great Pyramid of Giza is 1274 feet -/
theorem great_pyramid_sum_height_width :
  ∀ H W : ℕ, great_pyramid H W → H + W = 1274 :=
by
  sorry

#check great_pyramid_sum_height_width

end NUMINAMATH_CALUDE_great_pyramid_sum_height_width_l2677_267735


namespace NUMINAMATH_CALUDE_combination_equality_l2677_267723

theorem combination_equality (a : ℕ) : 
  (Nat.choose 17 (2*a - 1) + Nat.choose 17 (2*a) = Nat.choose 18 12) → 
  (a = 3 ∨ a = 6) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l2677_267723


namespace NUMINAMATH_CALUDE_expand_expressions_l2677_267779

theorem expand_expressions (x m n : ℝ) :
  ((-3*x - 5) * (5 - 3*x) = 9*x^2 - 25) ∧
  ((-3*x - 5) * (5 + 3*x) = -9*x^2 - 30*x - 25) ∧
  ((2*m - 3*n + 1) * (2*m + 1 + 3*n) = 4*m^2 + 4*m + 1 - 9*n^2) := by
  sorry

end NUMINAMATH_CALUDE_expand_expressions_l2677_267779


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2677_267793

theorem product_of_three_numbers (x y z : ℚ) : 
  x + y + z = 36 →
  x = 3 * (y + z) →
  y = 6 * z →
  x * y * z = 268 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2677_267793


namespace NUMINAMATH_CALUDE_average_increase_percentage_l2677_267757

def S : Finset Int := {6, 7, 10, 12, 15}
def N : Int := 34

theorem average_increase_percentage (S : Finset Int) (N : Int) :
  S = {6, 7, 10, 12, 15} →
  N = 34 →
  let original_sum := S.sum id
  let original_count := S.card
  let original_avg := original_sum / original_count
  let new_sum := original_sum + N
  let new_count := original_count + 1
  let new_avg := new_sum / new_count
  let increase := new_avg - original_avg
  let percentage_increase := (increase / original_avg) * 100
  percentage_increase = 40 := by
sorry

end NUMINAMATH_CALUDE_average_increase_percentage_l2677_267757


namespace NUMINAMATH_CALUDE_average_of_abc_is_three_l2677_267752

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 2012 * C - 4024 * A = 8048)
  (eq2 : 2012 * B + 6036 * A = 10010) :
  (A + B + C) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_abc_is_three_l2677_267752


namespace NUMINAMATH_CALUDE_distribute_5_3_l2677_267728

/-- The number of ways to distribute n distinct objects into k non-empty groups,
    where the order of the groups matters. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l2677_267728


namespace NUMINAMATH_CALUDE_xyz_value_l2677_267756

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2677_267756


namespace NUMINAMATH_CALUDE_student_arrangements_eq_60_l2677_267725

/-- The number of ways to arrange 6 students among three venues A, B, and C,
    where venue A receives 1 student, venue B receives 2 students,
    and venue C receives 3 students. -/
def student_arrangements : ℕ :=
  Nat.choose 6 1 * Nat.choose 5 2

theorem student_arrangements_eq_60 : student_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangements_eq_60_l2677_267725


namespace NUMINAMATH_CALUDE_taxi_fare_100_miles_l2677_267746

/-- The cost of a taxi trip given the distance traveled. -/
noncomputable def taxi_cost (base_fare : ℝ) (rate : ℝ) (distance : ℝ) : ℝ :=
  base_fare + rate * distance

theorem taxi_fare_100_miles :
  let base_fare : ℝ := 40
  let rate : ℝ := (200 - base_fare) / 80
  taxi_cost base_fare rate 100 = 240 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_100_miles_l2677_267746


namespace NUMINAMATH_CALUDE_spinner_prime_sum_probability_l2677_267714

-- Define the spinners
def spinner1 : List ℕ := [1, 2, 3, 4]
def spinner2 : List ℕ := [3, 4, 5, 6]

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Bool := sorry

-- Define a function to calculate all possible sums
def allSums (s1 s2 : List ℕ) : List ℕ := sorry

-- Define a function to count prime sums
def countPrimeSums (sums : List ℕ) : ℕ := sorry

-- Theorem to prove
theorem spinner_prime_sum_probability :
  let sums := allSums spinner1 spinner2
  let primeCount := countPrimeSums sums
  let totalCount := spinner1.length * spinner2.length
  (primeCount : ℚ) / totalCount = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_spinner_prime_sum_probability_l2677_267714


namespace NUMINAMATH_CALUDE_linear_function_k_value_l2677_267786

theorem linear_function_k_value (k : ℝ) : 
  k ≠ 0 → (1 : ℝ) = k * 3 - 2 → k = 1 := by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l2677_267786


namespace NUMINAMATH_CALUDE_galyas_number_puzzle_l2677_267701

theorem galyas_number_puzzle (N : ℕ) : (∀ k : ℝ, ((k * N + N) / N - N = k - 2021)) ↔ N = 2022 := by sorry

end NUMINAMATH_CALUDE_galyas_number_puzzle_l2677_267701


namespace NUMINAMATH_CALUDE_initial_peaches_l2677_267755

theorem initial_peaches (initial : ℕ) : initial + 52 = 86 → initial = 34 := by
  sorry

end NUMINAMATH_CALUDE_initial_peaches_l2677_267755


namespace NUMINAMATH_CALUDE_turtle_race_time_difference_l2677_267751

theorem turtle_race_time_difference (greta_time gloria_time : ℕ) 
  (h1 : greta_time = 6)
  (h2 : gloria_time = 8)
  (h3 : gloria_time = 2 * (gloria_time / 2)) :
  greta_time - (gloria_time / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_turtle_race_time_difference_l2677_267751


namespace NUMINAMATH_CALUDE_paving_cost_l2677_267704

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 1000) :
  length * width * rate = 20625 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l2677_267704


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2677_267715

theorem absolute_value_equality (x : ℝ) : |x + 2| = |x - 3| → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2677_267715


namespace NUMINAMATH_CALUDE_zoe_yogurt_consumption_l2677_267734

/-- Calculates the number of ounces of yogurt Zoe ate given the following conditions:
  * Zoe ate 12 strawberries and some ounces of yogurt
  * Strawberries have 4 calories each
  * Yogurt has 17 calories per ounce
  * Zoe ate a total of 150 calories
-/
theorem zoe_yogurt_consumption (
  strawberry_count : ℕ)
  (strawberry_calories : ℕ)
  (yogurt_calories_per_ounce : ℕ)
  (total_calories : ℕ)
  (h1 : strawberry_count = 12)
  (h2 : strawberry_calories = 4)
  (h3 : yogurt_calories_per_ounce = 17)
  (h4 : total_calories = 150)
  : (total_calories - strawberry_count * strawberry_calories) / yogurt_calories_per_ounce = 6 := by
  sorry

end NUMINAMATH_CALUDE_zoe_yogurt_consumption_l2677_267734


namespace NUMINAMATH_CALUDE_algebra_books_not_unique_l2677_267747

/-- Represents the number of books on a shelf -/
structure ShelfBooks where
  algebra : ℕ+
  geometry : ℕ+

/-- Represents the two shelves in the library -/
structure Library where
  longer_shelf : ShelfBooks
  shorter_shelf : ShelfBooks
  algebra_only : ℕ+

/-- The conditions of the library problem -/
def LibraryProblem (lib : Library) : Prop :=
  lib.longer_shelf.algebra > lib.shorter_shelf.algebra ∧
  lib.longer_shelf.geometry < lib.shorter_shelf.geometry ∧
  lib.longer_shelf.algebra ≠ lib.longer_shelf.geometry ∧
  lib.longer_shelf.algebra ≠ lib.shorter_shelf.algebra ∧
  lib.longer_shelf.algebra ≠ lib.shorter_shelf.geometry ∧
  lib.longer_shelf.geometry ≠ lib.shorter_shelf.algebra ∧
  lib.longer_shelf.geometry ≠ lib.shorter_shelf.geometry ∧
  lib.shorter_shelf.algebra ≠ lib.shorter_shelf.geometry ∧
  lib.longer_shelf.algebra ≠ lib.algebra_only ∧
  lib.longer_shelf.geometry ≠ lib.algebra_only ∧
  lib.shorter_shelf.algebra ≠ lib.algebra_only ∧
  lib.shorter_shelf.geometry ≠ lib.algebra_only

/-- The theorem stating that the number of algebra books to fill the longer shelf cannot be uniquely determined -/
theorem algebra_books_not_unique (lib : Library) (h : LibraryProblem lib) :
  ∃ (lib' : Library), LibraryProblem lib' ∧ lib'.algebra_only ≠ lib.algebra_only :=
sorry

end NUMINAMATH_CALUDE_algebra_books_not_unique_l2677_267747


namespace NUMINAMATH_CALUDE_count_divisible_sum_l2677_267787

theorem count_divisible_sum : ∃ (S : Finset Nat), 
  (∀ n ∈ S, n > 0 ∧ (10 * n) % ((n * (n + 1)) / 2) = 0) ∧
  (∀ n : Nat, n > 0 ∧ (10 * n) % ((n * (n + 1)) / 2) = 0 → n ∈ S) ∧
  Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_sum_l2677_267787


namespace NUMINAMATH_CALUDE_white_balls_count_l2677_267731

theorem white_balls_count (a b c : ℕ) : 
  a + b + c = 20 → -- Total number of balls
  (a : ℚ) / (20 + b) = a / 20 - 1 / 25 → -- Probability change when doubling blue balls
  b / (20 - a) = b / 20 + 1 / 16 → -- Probability change when removing white balls
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l2677_267731


namespace NUMINAMATH_CALUDE_second_group_size_l2677_267717

/-- Represents the number of man-days required to complete the work -/
def totalManDays : ℕ := 18 * 20

/-- Proves that 12 men can complete the work in 30 days, given that 18 men can complete it in 20 days -/
theorem second_group_size (days : ℕ) (h : days = 30) : 
  (totalManDays / days : ℕ) = 12 := by
  sorry

#check second_group_size

end NUMINAMATH_CALUDE_second_group_size_l2677_267717


namespace NUMINAMATH_CALUDE_average_cost_is_seven_l2677_267797

/-- The average cost per book in cents, rounded to the nearest whole number -/
def average_cost_per_book (num_books : ℕ) (lot_cost : ℚ) (delivery_fee : ℚ) : ℕ :=
  let total_cost_cents := (lot_cost + delivery_fee) * 100
  let average_cost := total_cost_cents / num_books
  (average_cost + 1/2).floor.toNat

/-- Theorem stating that the average cost per book is 7 cents -/
theorem average_cost_is_seven :
  average_cost_per_book 350 (15.30) (9.25) = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_cost_is_seven_l2677_267797


namespace NUMINAMATH_CALUDE_prob_sum_leq_8_is_13_18_l2677_267719

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum ≤ 8) when rolling two dice -/
def favorable_outcomes : ℕ := 26

/-- The probability of the sum being less than or equal to 8 when two dice are tossed -/
def prob_sum_leq_8 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_leq_8_is_13_18 : prob_sum_leq_8 = 13 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_leq_8_is_13_18_l2677_267719


namespace NUMINAMATH_CALUDE_trisected_right_triangle_product_l2677_267792

/-- A right triangle with trisected angle -/
structure TrisectedRightTriangle where
  -- The length of side XY
  xy : ℝ
  -- The length of side YZ
  yz : ℝ
  -- Point P on XZ
  p : ℝ × ℝ
  -- Point Q on XZ
  q : ℝ × ℝ
  -- The angle at Y is trisected
  angle_trisected : Bool
  -- X, P, Q, Z lie on XZ in that order
  point_order : Bool

/-- The main theorem -/
theorem trisected_right_triangle_product (t : TrisectedRightTriangle)
  (h_xy : t.xy = 228)
  (h_yz : t.yz = 2004)
  (h_trisected : t.angle_trisected = true)
  (h_order : t.point_order = true) :
  (Real.sqrt ((t.p.1 - 0)^2 + (t.p.2 - t.yz)^2) + t.yz) *
  (Real.sqrt ((t.q.1 - 0)^2 + (t.q.2 - t.yz)^2) + t.xy) = 1370736 := by
  sorry

end NUMINAMATH_CALUDE_trisected_right_triangle_product_l2677_267792


namespace NUMINAMATH_CALUDE_solution_x_l2677_267762

theorem solution_x (x : Real) 
  (h1 : x ∈ Set.Ioo 0 (π / 2))
  (h2 : 1 / Real.sin x = 1 / Real.sin (2 * x) + 1 / Real.sin (4 * x) + 1 / Real.sin (8 * x)) :
  x = π / 15 ∨ x = π / 5 ∨ x = π / 3 ∨ x = 7 * π / 15 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_l2677_267762


namespace NUMINAMATH_CALUDE_square_root_range_l2677_267712

theorem square_root_range (x : ℝ) : ∃ y : ℝ, y = Real.sqrt (x - 5) ↔ x ≥ 5 := by sorry

end NUMINAMATH_CALUDE_square_root_range_l2677_267712


namespace NUMINAMATH_CALUDE_people_in_room_l2677_267795

theorem people_in_room (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) :
  total_chairs = 25 →
  seated_people = (3 * total_people) / 4 →
  seated_people = (2 * total_chairs) / 3 →
  total_people = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_people_in_room_l2677_267795


namespace NUMINAMATH_CALUDE_sarah_investment_l2677_267763

/-- Proves that given a total investment of $250,000 and the investment in real estate
    being 6 times the investment in mutual funds, the amount invested in real estate
    is $214,285.71. -/
theorem sarah_investment (total : ℝ) (real_estate : ℝ) (mutual_funds : ℝ) 
    (h1 : total = 250000)
    (h2 : real_estate = 6 * mutual_funds)
    (h3 : total = real_estate + mutual_funds) :
  real_estate = 214285.71 := by
  sorry

end NUMINAMATH_CALUDE_sarah_investment_l2677_267763


namespace NUMINAMATH_CALUDE_blocks_added_l2677_267772

theorem blocks_added (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 35)
  (h2 : final_blocks = 65) :
  final_blocks - initial_blocks = 30 := by
sorry

end NUMINAMATH_CALUDE_blocks_added_l2677_267772


namespace NUMINAMATH_CALUDE_box_weight_l2677_267767

/-- Given a pallet with boxes, calculate the weight of each box. -/
theorem box_weight (total_weight : ℕ) (num_boxes : ℕ) (h1 : total_weight = 267) (h2 : num_boxes = 3) :
  total_weight / num_boxes = 89 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_l2677_267767


namespace NUMINAMATH_CALUDE_white_triangle_coincidence_l2677_267724

/-- Represents the number of triangles of each color in each half of the diagram -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the diagram is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_blue : ℕ

/-- Calculates the number of coinciding white triangle pairs given the initial counts and other coinciding pairs -/
def coinciding_white_pairs (counts : TriangleCounts) (pairs : CoincidingPairs) : ℕ :=
  counts.white - (counts.red - 2 * pairs.red_red - pairs.red_blue) - (counts.blue - 2 * pairs.blue_blue - pairs.red_blue)

/-- Theorem stating that under the given conditions, 6 pairs of white triangles exactly coincide -/
theorem white_triangle_coincidence (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.red = 5 ∧ counts.blue = 4 ∧ counts.white = 7 ∧ 
  pairs.red_red = 3 ∧ pairs.blue_blue = 2 ∧ pairs.red_blue = 1 →
  coinciding_white_pairs counts pairs = 6 := by
  sorry

end NUMINAMATH_CALUDE_white_triangle_coincidence_l2677_267724


namespace NUMINAMATH_CALUDE_triangle_angle_bounds_l2677_267758

theorem triangle_angle_bounds (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = π) (h5 : A ≤ B) (h6 : B ≤ C) :
  (0 < A ∧ A ≤ π/3) ∧
  (0 < B ∧ B < π/2) ∧
  (π/3 ≤ C ∧ C < π) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_bounds_l2677_267758


namespace NUMINAMATH_CALUDE_teena_speed_calculation_l2677_267769

/-- Teena's speed in miles per hour -/
def teena_speed : ℝ := 55

/-- Loe's speed in miles per hour -/
def loe_speed : ℝ := 40

/-- Initial distance Teena is behind Loe in miles -/
def initial_distance_behind : ℝ := 7.5

/-- Time after which Teena is ahead of Loe in hours -/
def time_elapsed : ℝ := 1.5

/-- Distance Teena is ahead of Loe after time_elapsed in miles -/
def final_distance_ahead : ℝ := 15

theorem teena_speed_calculation :
  teena_speed * time_elapsed = 
    initial_distance_behind + final_distance_ahead + (loe_speed * time_elapsed) := by
  sorry

end NUMINAMATH_CALUDE_teena_speed_calculation_l2677_267769


namespace NUMINAMATH_CALUDE_max_tournament_size_l2677_267709

/-- Represents a tournament with 2^n students --/
structure Tournament (n : ℕ) where
  students : Fin (2^n)
  day1_pairs : List (Fin (2^n) × Fin (2^n))
  day2_pairs : List (Fin (2^n) × Fin (2^n))

/-- The sets of pairs that played on both days are the same --/
def same_pairs (t : Tournament n) : Prop :=
  t.day1_pairs.toFinset = t.day2_pairs.toFinset

/-- The maximum value of n for which the tournament conditions hold --/
def max_n : ℕ := 3

/-- Theorem stating that 3 is the maximum value of n for which the tournament conditions hold --/
theorem max_tournament_size :
  ∀ n : ℕ, n > max_n → ¬∃ t : Tournament n, same_pairs t :=
sorry

end NUMINAMATH_CALUDE_max_tournament_size_l2677_267709


namespace NUMINAMATH_CALUDE_intersection_union_problem_l2677_267790

theorem intersection_union_problem (m : ℝ) : 
  let A : Set ℝ := {3, 4, m^2 - 3*m - 1}
  let B : Set ℝ := {2*m, -3}
  (A ∩ B = {-3}) → (m = 1 ∧ A ∪ B = {-3, 2, 3, 4}) :=
by sorry

end NUMINAMATH_CALUDE_intersection_union_problem_l2677_267790


namespace NUMINAMATH_CALUDE_two_circles_exist_l2677_267744

/-- The parabola y^2 = 4x with focus F(1,0) and directrix x = -1 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- The directrix of the parabola -/
def Directrix : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -1}

/-- Point M -/
def M : ℝ × ℝ := (4, 4)

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate for a circle passing through two points and tangent to a line -/
def CirclePassesThroughAndTangent (c : Circle) (p1 p2 : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  (c.center.1 - p1.1)^2 + (c.center.2 - p1.2)^2 = c.radius^2 ∧
  (c.center.1 - p2.1)^2 + (c.center.2 - p2.2)^2 = c.radius^2 ∧
  ∃ (q : ℝ × ℝ), q ∈ l ∧ (c.center.1 - q.1)^2 + (c.center.2 - q.2)^2 = c.radius^2

theorem two_circles_exist : ∃ (c1 c2 : Circle),
  CirclePassesThroughAndTangent c1 F M Directrix ∧
  CirclePassesThroughAndTangent c2 F M Directrix ∧
  c1 ≠ c2 ∧
  ∀ (c : Circle), CirclePassesThroughAndTangent c F M Directrix → c = c1 ∨ c = c2 :=
sorry

end NUMINAMATH_CALUDE_two_circles_exist_l2677_267744


namespace NUMINAMATH_CALUDE_simplify_expression_l2677_267770

theorem simplify_expression : 0.4 * 0.5 + 0.3 * 0.2 = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2677_267770


namespace NUMINAMATH_CALUDE_first_number_value_l2677_267750

theorem first_number_value (x y z : ℤ) 
  (sum_xy : x + y = 31)
  (sum_yz : y + z = 47)
  (sum_xz : x + z = 52)
  (condition : y + z = x + 16) :
  x = 31 := by
sorry

end NUMINAMATH_CALUDE_first_number_value_l2677_267750


namespace NUMINAMATH_CALUDE_equal_sum_sequence_18th_term_l2677_267705

/-- An equal sum sequence is a sequence where the sum of each term and its next term is constant. -/
def EqualSumSequence (a : ℕ → ℝ) :=
  ∃ s : ℝ, ∀ n : ℕ, a n + a (n + 1) = s

theorem equal_sum_sequence_18th_term 
  (a : ℕ → ℝ) 
  (h_equal_sum : EqualSumSequence a)
  (h_first_term : a 1 = 2)
  (h_common_sum : ∃ s : ℝ, s = 5 ∧ ∀ n : ℕ, a n + a (n + 1) = s) :
  a 18 = 3 := by
  sorry

#check equal_sum_sequence_18th_term

end NUMINAMATH_CALUDE_equal_sum_sequence_18th_term_l2677_267705


namespace NUMINAMATH_CALUDE_field_dimension_l2677_267716

/-- The value of m for a rectangular field with given dimensions and area -/
theorem field_dimension (m : ℝ) : (3*m + 11) * (m - 3) = 80 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_field_dimension_l2677_267716


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l2677_267748

-- Define the statements p and q
def p (a : ℝ) : Prop := a ≥ 1
def q (a : ℝ) : Prop := a ≤ 2

-- Theorem stating that ¬p is a sufficient but not necessary condition for q
theorem not_p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, ¬(p a) → q a) ∧ ¬(∀ a : ℝ, q a → ¬(p a)) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l2677_267748


namespace NUMINAMATH_CALUDE_three_digit_reverse_divisible_by_11_l2677_267710

theorem three_digit_reverse_divisible_by_11 (a b c : Nat) (ha : a ≠ 0) (hb : b < 10) (hc : c < 10) :
  ∃ k : Nat, 100001 * a + 10010 * b + 1100 * c = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_three_digit_reverse_divisible_by_11_l2677_267710


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_l2677_267768

/-- Triangle ABC with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : (ℝ × ℝ) := (4, 0)
  B : (ℝ × ℝ) := (6, 7)
  C : (ℝ × ℝ) := (0, 3)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.altitudeFromB (t : Triangle) : LineEquation :=
  { a := 3, b := 2, c := -12 }

def Triangle.medianFromB (t : Triangle) : LineEquation :=
  { a := 5, b := 1, c := -20 }

theorem triangle_altitude_and_median (t : Triangle) :
  (t.altitudeFromB = { a := 3, b := 2, c := -12 }) ∧
  (t.medianFromB = { a := 5, b := 1, c := -20 }) := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_l2677_267768


namespace NUMINAMATH_CALUDE_line_transformation_l2677_267707

open Matrix

-- Define the rotation matrix M
def M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

-- Define the scaling matrix N
def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 3]

-- Define the combined transformation matrix NM
def NM : Matrix (Fin 2) (Fin 2) ℝ := N * M

theorem line_transformation (x y : ℝ) :
  (NM.mulVec ![x, y] = ![x, x]) ↔ (3 * x + 2 * y = 0) := by sorry

end NUMINAMATH_CALUDE_line_transformation_l2677_267707


namespace NUMINAMATH_CALUDE_probNotAllSame_l2677_267759

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The probability that all dice show the same number -/
def probAllSame : ℚ := 1 / numSides^(numDice - 1)

/-- The probability that not all dice show the same number -/
theorem probNotAllSame : (1 : ℚ) - probAllSame = 215 / 216 := by sorry

end NUMINAMATH_CALUDE_probNotAllSame_l2677_267759


namespace NUMINAMATH_CALUDE_oliver_vowel_learning_days_l2677_267721

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of days Oliver takes to learn one alphabet -/
def days_per_alphabet : ℕ := 5

/-- The number of days Oliver needs to finish learning all vowels -/
def days_to_learn_vowels : ℕ := num_vowels * days_per_alphabet

/-- Theorem: Oliver needs 25 days to finish learning all vowels -/
theorem oliver_vowel_learning_days : days_to_learn_vowels = 25 := by
  sorry

end NUMINAMATH_CALUDE_oliver_vowel_learning_days_l2677_267721


namespace NUMINAMATH_CALUDE_rudolph_travel_distance_l2677_267738

/-- Represents the number of stop signs Rudolph encountered -/
def total_stop_signs : ℕ := 17 - 3

/-- Represents the number of stop signs per mile -/
def stop_signs_per_mile : ℕ := 2

/-- Calculates the number of miles Rudolph traveled -/
def miles_traveled : ℚ := total_stop_signs / stop_signs_per_mile

theorem rudolph_travel_distance :
  miles_traveled = 7 := by sorry

end NUMINAMATH_CALUDE_rudolph_travel_distance_l2677_267738


namespace NUMINAMATH_CALUDE_distinct_color_selections_eq_62_l2677_267783

/-- The number of ways to select 6 objects from 5 red and 5 blue objects, where order matters only for color. -/
def distinct_color_selections : ℕ :=
  let red := 5
  let blue := 5
  let total_select := 6
  (2 * (Nat.choose total_select 1) +  -- 5 of one color, 1 of the other
   2 * (Nat.choose total_select 2) +  -- 4 of one color, 2 of the other
   Nat.choose total_select 3)         -- 3 of each color

/-- Theorem stating that the number of distinct color selections is 62. -/
theorem distinct_color_selections_eq_62 : distinct_color_selections = 62 := by
  sorry

end NUMINAMATH_CALUDE_distinct_color_selections_eq_62_l2677_267783


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2677_267753

-- Problem 1
theorem problem_1 : (-16) - 25 + (-43) - (-39) = -45 := by sorry

-- Problem 2
theorem problem_2 : (-3/4)^2 * (-8 + 1/3) = -69/16 := by sorry

-- Problem 3
theorem problem_3 : 16 / (-1/2) * 3/8 - |(-45)| / 9 = -17 := by sorry

-- Problem 4
theorem problem_4 : -1^2024 - (2 - 0.75) * 2/7 * (4 - (-5)^2) = 13/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2677_267753


namespace NUMINAMATH_CALUDE_percentage_of_boys_l2677_267791

theorem percentage_of_boys (total_students : ℕ) (boys : ℕ) (percentage : ℚ) : 
  total_students = 220 →
  242 = (220 / 100) * boys →
  percentage = (boys / total_students) * 100 →
  percentage = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_boys_l2677_267791


namespace NUMINAMATH_CALUDE_weight_of_rod_l2677_267789

/-- Represents the weight of a uniform rod -/
structure UniformRod where
  /-- Weight per meter of the rod -/
  weight_per_meter : ℝ
  /-- The rod is uniform (constant weight per meter) -/
  uniform : True

/-- Calculate the weight of a given length of a uniform rod -/
def weight_of_length (rod : UniformRod) (length : ℝ) : ℝ :=
  rod.weight_per_meter * length

/-- Theorem: Given a uniform rod where 8 m weighs 30.4 kg, the weight of 11.25 m is 42.75 kg -/
theorem weight_of_rod (rod : UniformRod) 
  (h : weight_of_length rod 8 = 30.4) : 
  weight_of_length rod 11.25 = 42.75 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_rod_l2677_267789


namespace NUMINAMATH_CALUDE_cosine_sine_equation_solutions_l2677_267775

open Real

theorem cosine_sine_equation_solutions (a α : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   cos (x₁ - a) - sin (x₁ + 2*α) = 0 ∧
   cos (x₂ - a) - sin (x₂ + 2*α) = 0 ∧
   ¬ ∃ k : ℤ, x₁ - x₂ = k * π) ↔ 
  ∃ t : ℤ, a = π * (4*t + 1) / 6 :=
by sorry

end NUMINAMATH_CALUDE_cosine_sine_equation_solutions_l2677_267775


namespace NUMINAMATH_CALUDE_circle_theorem_l2677_267782

/-- Represents the type of person in the circle -/
inductive PersonType
| Knight
| Liar

/-- Checks if a given number is a valid k value -/
def is_valid_k (k : ℕ) : Prop :=
  k < 100 ∧ ∃ (m : ℕ), 100 = m * (k + 1)

/-- The set of all valid k values -/
def valid_k_set : Set ℕ :=
  {1, 3, 4, 9, 19, 24, 49, 99}

/-- A circle of 100 people -/
def Circle := Fin 100 → PersonType

theorem circle_theorem (circle : Circle) :
  ∃ (k : ℕ), is_valid_k k ∧
  (∀ (i : Fin 100),
    (circle i = PersonType.Knight →
      ∀ (j : Fin 100), j < k → circle ((i + j + 1) % 100) = PersonType.Liar) ∧
    (circle i = PersonType.Liar →
      ∃ (j : Fin 100), j < k ∧ circle ((i + j + 1) % 100) = PersonType.Knight)) ↔
  ∃ (k : ℕ), k ∈ valid_k_set :=
sorry

end NUMINAMATH_CALUDE_circle_theorem_l2677_267782


namespace NUMINAMATH_CALUDE_alice_probability_after_three_turns_l2677_267739

/-- Represents the probability of Alice having the ball after three turns in the baseball game. -/
def aliceProbabilityAfterThreeTurns : ℚ :=
  let aliceKeepProb : ℚ := 1/2
  let aliceTossProb : ℚ := 1/2
  let bobTossProb : ℚ := 3/5
  let bobKeepProb : ℚ := 2/5
  
  -- Alice passes to Bob, Bob passes to Alice, Alice keeps
  let seq1 : ℚ := aliceTossProb * bobTossProb * aliceKeepProb
  -- Alice passes to Bob, Bob passes to Alice, Alice passes to Bob
  let seq2 : ℚ := aliceTossProb * bobTossProb * aliceTossProb
  -- Alice keeps, Alice keeps, Alice keeps
  let seq3 : ℚ := aliceKeepProb * aliceKeepProb * aliceKeepProb
  -- Alice keeps, Alice passes to Bob, Bob passes to Alice
  let seq4 : ℚ := aliceKeepProb * aliceTossProb * bobTossProb
  
  seq1 + seq2 + seq3 + seq4

/-- Theorem stating that the probability of Alice having the ball after three turns is 23/40. -/
theorem alice_probability_after_three_turns :
  aliceProbabilityAfterThreeTurns = 23/40 := by
  sorry

end NUMINAMATH_CALUDE_alice_probability_after_three_turns_l2677_267739


namespace NUMINAMATH_CALUDE_class_size_l2677_267737

theorem class_size (n : ℕ) 
  (h1 : 30 * 160 + (n - 30) * 156 = n * 159) : n = 40 := by
  sorry

#check class_size

end NUMINAMATH_CALUDE_class_size_l2677_267737


namespace NUMINAMATH_CALUDE_percent_of_decimal_l2677_267799

theorem percent_of_decimal (part whole : ℝ) (percent : ℝ) : 
  part = 0.01 → whole = 0.1 → percent = 10 → (part / whole) * 100 = percent := by
  sorry

end NUMINAMATH_CALUDE_percent_of_decimal_l2677_267799


namespace NUMINAMATH_CALUDE_expression_evaluation_l2677_267736

theorem expression_evaluation : 
  Real.sqrt 3 + 3 + (1 / (Real.sqrt 3 + 3))^2 + 1 / (3 - Real.sqrt 3) = Real.sqrt 3 + 3 + 5/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2677_267736


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l2677_267788

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (a b c d : Point3D) : ℝ := sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (a b c d : Point3D) : ℝ := sorry

theorem tetrahedron_volume_and_height :
  let a₁ : Point3D := ⟨-2, -1, -1⟩
  let a₂ : Point3D := ⟨0, 3, 2⟩
  let a₃ : Point3D := ⟨3, 1, -4⟩
  let a₄ : Point3D := ⟨-4, 7, 3⟩
  (tetrahedronVolume a₁ a₂ a₃ a₄ = 70/3) ∧
  (tetrahedronHeight a₄ a₁ a₂ a₃ = 140 / Real.sqrt 1021) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l2677_267788


namespace NUMINAMATH_CALUDE_stratified_sample_under35_l2677_267776

/-- Represents the number of employees in each age group -/
structure EmployeeGroups where
  total : ℕ
  under35 : ℕ
  between35and49 : ℕ
  over50 : ℕ

/-- Calculates the number of employees to be drawn from a specific group in stratified sampling -/
def stratifiedSampleSize (groups : EmployeeGroups) (sampleTotal : ℕ) (groupSize : ℕ) : ℕ :=
  (groupSize * sampleTotal) / groups.total

/-- Theorem stating that in the given scenario, 25 employees under 35 should be drawn -/
theorem stratified_sample_under35 (groups : EmployeeGroups) (sampleTotal : ℕ) :
  groups.total = 500 →
  groups.under35 = 125 →
  groups.between35and49 = 280 →
  groups.over50 = 95 →
  sampleTotal = 100 →
  stratifiedSampleSize groups sampleTotal groups.under35 = 25 := by
  sorry

#check stratified_sample_under35

end NUMINAMATH_CALUDE_stratified_sample_under35_l2677_267776


namespace NUMINAMATH_CALUDE_correct_fraction_is_five_thirds_l2677_267780

/-- The percentage error when using an incorrect fraction instead of the correct one. -/
def percentage_error : ℚ := 64.00000000000001

/-- The incorrect fraction used by the student. -/
def incorrect_fraction : ℚ := 3/5

/-- The correct fraction that should have been used. -/
def correct_fraction : ℚ := 5/3

/-- Theorem stating that given the percentage error and incorrect fraction, 
    the correct fraction is 5/3. -/
theorem correct_fraction_is_five_thirds :
  (1 - percentage_error / 100) * correct_fraction = incorrect_fraction :=
sorry

end NUMINAMATH_CALUDE_correct_fraction_is_five_thirds_l2677_267780


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2677_267778

theorem polynomial_division_theorem (x : ℝ) :
  x^4 + 3*x^3 - 17*x^2 + 8*x - 12 = (x - 3) * (x^3 + 6*x^2 + x + 11) + 21 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2677_267778


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2677_267794

theorem chess_tournament_participants :
  let n : ℕ := 28
  let total_games : ℕ := n * (n - 1) / 2
  let uncounted_games : ℕ := 10 * n
  total_games + uncounted_games = 672 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2677_267794


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2677_267785

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2677_267785


namespace NUMINAMATH_CALUDE_max_teams_in_tournament_l2677_267722

/-- Represents a chess tournament with teams of 3 players each --/
structure ChessTournament where
  numTeams : ℕ
  maxGames : ℕ := 250

/-- Calculate the total number of games in the tournament --/
def totalGames (t : ChessTournament) : ℕ :=
  (9 * t.numTeams * (t.numTeams - 1)) / 2

/-- Theorem stating the maximum number of teams in the tournament --/
theorem max_teams_in_tournament (t : ChessTournament) :
  (∀ n : ℕ, n ≤ t.numTeams → totalGames { numTeams := n, maxGames := t.maxGames } ≤ t.maxGames) →
  t.numTeams ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_teams_in_tournament_l2677_267722


namespace NUMINAMATH_CALUDE_bobby_total_consumption_l2677_267727

def bobby_consumption (initial_candy : ℕ) (additional_candy : ℕ) (candy_fraction : ℚ) 
                      (chocolate : ℕ) (chocolate_fraction : ℚ) : ℚ :=
  initial_candy + candy_fraction * additional_candy + chocolate_fraction * chocolate

theorem bobby_total_consumption : 
  bobby_consumption 28 42 (3/4) 63 (1/2) = 91 := by
  sorry

end NUMINAMATH_CALUDE_bobby_total_consumption_l2677_267727


namespace NUMINAMATH_CALUDE_prob_second_red_given_first_red_is_half_l2677_267702

/-- Represents the probability of drawing a red ball as the second draw, given that the first draw was red, from a box containing red and white balls. -/
def probability_second_red_given_first_red (total_red : ℕ) (total_white : ℕ) : ℚ :=
  if total_red > 0 then
    (total_red - 1 : ℚ) / (total_red + total_white - 1 : ℚ)
  else
    0

/-- Theorem stating that in a box with 4 red balls and 3 white balls, 
    if two balls are drawn without replacement and the first ball is red, 
    the probability that the second ball is also red is 1/2. -/
theorem prob_second_red_given_first_red_is_half :
  probability_second_red_given_first_red 4 3 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_red_given_first_red_is_half_l2677_267702


namespace NUMINAMATH_CALUDE_candy_distribution_l2677_267774

theorem candy_distribution (n : ℕ) : 
  (n > 0) → 
  (120 % n = 1) → 
  (n = 7 ∨ n = 17) :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l2677_267774


namespace NUMINAMATH_CALUDE_sum_of_row_and_column_for_2023_l2677_267781

/-- Represents the value in the table at a given row and column -/
def tableValue (row : ℕ) (col : ℕ) : ℕ :=
  if row % 2 = 1 then
    (row - 1) * 20 + (col - 1) * 2 + 1
  else
    row * 20 - (col - 1) * 2 - 1

/-- The row where 2023 is located -/
def m : ℕ := 253

/-- The column where 2023 is located -/
def n : ℕ := 5

theorem sum_of_row_and_column_for_2023 :
  tableValue m n = 2023 → m + n = 258 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_row_and_column_for_2023_l2677_267781


namespace NUMINAMATH_CALUDE_sin_sum_max_in_acute_triangle_l2677_267777

-- Define the convexity property for a function on an interval
def IsConvex (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y t : ℝ, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ 0 ≤ t ∧ t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

-- State the theorem
theorem sin_sum_max_in_acute_triangle :
  IsConvex Real.sin 0 (Real.pi / 2) →
  ∀ A B C : ℝ,
    0 < A ∧ A < Real.pi / 2 →
    0 < B ∧ B < Real.pi / 2 →
    0 < C ∧ C < Real.pi / 2 →
    A + B + C = Real.pi →
    Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_sum_max_in_acute_triangle_l2677_267777


namespace NUMINAMATH_CALUDE_two_digit_number_representation_l2677_267754

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

theorem two_digit_number_representation (n : TwoDigitNumber) :
  n.value = 10 * n.tens + n.units := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_representation_l2677_267754


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_inscribed_circle_theorem_l2677_267733

/-- An isosceles right triangle with an inscribed circle -/
structure IsoscelesRightTriangleWithCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The height from the right angle to the hypotenuse -/
  h : ℝ
  /-- The height is twice the radius -/
  height_radius_relation : h = 2 * r
  /-- The radius is √2/4 -/
  radius_value : r = Real.sqrt 2 / 4

/-- The theorem to be proved -/
theorem isosceles_right_triangle_inscribed_circle_theorem 
  (triangle : IsoscelesRightTriangleWithCircle) : 
  triangle.h - triangle.r = Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_right_triangle_inscribed_circle_theorem_l2677_267733


namespace NUMINAMATH_CALUDE_f_divisibility_l2677_267740

/-- Sequence a defined recursively -/
def a (r s : ℕ) : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => r * a r s (n + 1) + s * a r s n

/-- Product of first n terms of sequence a -/
def f (r s : ℕ) : ℕ → ℕ
  | 0 => 1
  | (n + 1) => f r s n * a r s (n + 1)

/-- Main theorem -/
theorem f_divisibility (r s n k : ℕ) (hr : r > 0) (hs : s > 0) (hk : k > 0) (hnk : n > k) :
  ∃ m : ℕ, f r s n = m * (f r s k * f r s (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_f_divisibility_l2677_267740


namespace NUMINAMATH_CALUDE_oregon_migration_l2677_267720

/-- The number of people moving to Oregon -/
def people_moving : ℕ := 3500

/-- The number of days over which people are moving -/
def days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def average_per_hour : ℚ := people_moving / (days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem oregon_migration :
  round_to_nearest average_per_hour = 29 := by
  sorry

end NUMINAMATH_CALUDE_oregon_migration_l2677_267720


namespace NUMINAMATH_CALUDE_runners_capture_probability_l2677_267765

/-- Represents a runner on a circular track -/
structure Runner where
  direction : Bool -- true for counterclockwise, false for clockwise
  lap_time : ℕ -- time to complete one lap in seconds

/-- Represents the photographer's capture area -/
structure CaptureArea where
  fraction : ℚ -- fraction of the track captured
  center : ℚ -- position of the center of the capture area (0 ≤ center < 1)

/-- Calculates the probability of both runners being in the capture area -/
def probability_both_in_picture (runner1 runner2 : Runner) (capture : CaptureArea) 
  (start_time end_time : ℕ) : ℚ :=
sorry

theorem runners_capture_probability :
  let jenna : Runner := { direction := true, lap_time := 75 }
  let jonathan : Runner := { direction := false, lap_time := 60 }
  let capture : CaptureArea := { fraction := 1/3, center := 0 }
  probability_both_in_picture jenna jonathan capture (15 * 60) (16 * 60) = 2/3 :=
sorry

end NUMINAMATH_CALUDE_runners_capture_probability_l2677_267765


namespace NUMINAMATH_CALUDE_zoo_guides_theorem_l2677_267764

/-- The total number of children addressed by zoo guides --/
def total_children (total_guides : ℕ) 
                   (english_guides : ℕ) 
                   (french_guides : ℕ) 
                   (english_children : ℕ) 
                   (french_children : ℕ) 
                   (spanish_children : ℕ) : ℕ :=
  let spanish_guides := total_guides - english_guides - french_guides
  english_guides * english_children + 
  french_guides * french_children + 
  spanish_guides * spanish_children

/-- Theorem stating the total number of children addressed by zoo guides --/
theorem zoo_guides_theorem : 
  total_children 22 10 6 19 25 30 = 520 := by
  sorry

end NUMINAMATH_CALUDE_zoo_guides_theorem_l2677_267764


namespace NUMINAMATH_CALUDE_square_difference_81_49_l2677_267784

theorem square_difference_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_81_49_l2677_267784


namespace NUMINAMATH_CALUDE_rectangle_division_perimeter_l2677_267713

theorem rectangle_division_perimeter (a b x y : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < x) ∧ (x < a) ∧ (0 < y) ∧ (y < b) →
  (∃ (k₁ k₂ k₃ : ℤ),
    2 * (x + y) = k₁ ∧
    2 * (x + b - y) = k₂ ∧
    2 * (a - x + y) = k₃) →
  ∃ (k₄ : ℤ), 2 * (a - x + b - y) = k₄ :=
by sorry

end NUMINAMATH_CALUDE_rectangle_division_perimeter_l2677_267713


namespace NUMINAMATH_CALUDE_mod_product_equals_one_l2677_267741

theorem mod_product_equals_one (m : ℕ) : 
  187 * 973 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_product_equals_one_l2677_267741


namespace NUMINAMATH_CALUDE_mary_regular_rate_l2677_267745

/-- Represents Mary's work schedule and pay structure --/
structure MaryWork where
  maxHours : ℕ
  regularHours : ℕ
  overtimeRate : ℚ
  weeklyEarnings : ℚ

/-- Calculates Mary's regular hourly rate --/
def regularRate (w : MaryWork) : ℚ :=
  let overtimeHours := w.maxHours - w.regularHours
  w.weeklyEarnings / (w.regularHours + w.overtimeRate * overtimeHours)

/-- Theorem: Mary's regular hourly rate is $8 per hour --/
theorem mary_regular_rate :
  let w : MaryWork := {
    maxHours := 45,
    regularHours := 20,
    overtimeRate := 1.25,
    weeklyEarnings := 410
  }
  regularRate w = 8 := by sorry

end NUMINAMATH_CALUDE_mary_regular_rate_l2677_267745
