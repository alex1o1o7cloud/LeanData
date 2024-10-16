import Mathlib

namespace NUMINAMATH_CALUDE_line_through_points_l2784_278429

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a line passing through distinct vectors a and b, 
    if k*a + (5/6)*b lies on the same line, then k = 5/6 -/
theorem line_through_points (a b : V) (h : a ≠ b) :
  ∃ (t : ℝ), k • a + (5/6) • b = a + t • (b - a) → k = 5/6 :=
sorry

end NUMINAMATH_CALUDE_line_through_points_l2784_278429


namespace NUMINAMATH_CALUDE_cindy_earnings_l2784_278439

/-- Calculates the earnings for teaching one math course in a month --/
def earnings_per_course (total_hours_per_week : ℕ) (num_courses : ℕ) (hourly_rate : ℕ) (weeks_per_month : ℕ) : ℕ :=
  (total_hours_per_week / num_courses) * weeks_per_month * hourly_rate

/-- Theorem: Cindy's earnings for one math course in a month --/
theorem cindy_earnings :
  let total_hours_per_week : ℕ := 48
  let num_courses : ℕ := 4
  let hourly_rate : ℕ := 25
  let weeks_per_month : ℕ := 4
  earnings_per_course total_hours_per_week num_courses hourly_rate weeks_per_month = 1200 := by
  sorry

#eval earnings_per_course 48 4 25 4

end NUMINAMATH_CALUDE_cindy_earnings_l2784_278439


namespace NUMINAMATH_CALUDE_perimeter_plus_area_sum_l2784_278440

/-- A parallelogram with integer coordinates -/
structure IntegerParallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ
  is_parallelogram : v1.1 + v3.1 = v2.1 + v4.1 ∧ v1.2 + v3.2 = v2.2 + v4.2

/-- Calculate the perimeter of a parallelogram -/
def perimeter (p : IntegerParallelogram) : ℝ :=
  2 * (dist p.v1 p.v2 + dist p.v2 p.v3)

/-- Calculate the area of a parallelogram -/
def area (p : IntegerParallelogram) : ℝ :=
  abs ((p.v2.1 - p.v1.1) * (p.v3.2 - p.v1.2) - (p.v3.1 - p.v1.1) * (p.v2.2 - p.v1.2))

/-- The sum of perimeter and area for the specific parallelogram -/
theorem perimeter_plus_area_sum (p : IntegerParallelogram) 
  (h1 : p.v1 = (2, 3)) 
  (h2 : p.v2 = (5, 7)) 
  (h3 : p.v3 = (0, -1)) : 
  perimeter p + area p = 10 + 12 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_plus_area_sum_l2784_278440


namespace NUMINAMATH_CALUDE_homework_completion_l2784_278470

/-- The fraction of homework Sanjay completed on Monday -/
def sanjay_monday : ℚ := 3/5

/-- The fraction of homework Deepak completed on Monday -/
def deepak_monday : ℚ := 2/7

/-- The fraction of remaining homework Sanjay completed on Tuesday -/
def sanjay_tuesday : ℚ := 1/3

/-- The fraction of remaining homework Deepak completed on Tuesday -/
def deepak_tuesday : ℚ := 3/10

/-- The combined fraction of original homework left for Sanjay and Deepak on Wednesday -/
def homework_left : ℚ := 23/30

theorem homework_completion :
  let sanjay_left := (1 - sanjay_monday) * (1 - sanjay_tuesday)
  let deepak_left := (1 - deepak_monday) * (1 - deepak_tuesday)
  sanjay_left + deepak_left = homework_left := by sorry

end NUMINAMATH_CALUDE_homework_completion_l2784_278470


namespace NUMINAMATH_CALUDE_cherry_tree_leaves_l2784_278464

/-- The number of cherry trees originally planned to be planted -/
def original_plan : ℕ := 7

/-- The actual number of cherry trees planted -/
def actual_trees : ℕ := 2 * original_plan

/-- The number of leaves each tree drops -/
def leaves_per_tree : ℕ := 100

/-- The total number of leaves falling from all cherry trees -/
def total_leaves : ℕ := actual_trees * leaves_per_tree

theorem cherry_tree_leaves : total_leaves = 1400 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tree_leaves_l2784_278464


namespace NUMINAMATH_CALUDE_spade_theorem_l2784_278412

-- Define the binary operation ◊
def spade (A B : ℚ) : ℚ := 4 * A + 3 * B - 2

-- Theorem statement
theorem spade_theorem (A : ℚ) : spade A 7 = 40 → A = 21 / 4 := by
  sorry

end NUMINAMATH_CALUDE_spade_theorem_l2784_278412


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2784_278445

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

theorem sufficient_not_necessary_condition
  (a₁ q : ℝ) :
  (a₁ < 0 ∧ 0 < q ∧ q < 1 →
    ∀ n : ℕ, n > 0 → geometric_sequence a₁ q (n + 1) > geometric_sequence a₁ q n) ∧
  (∃ a₁' q' : ℝ, (∀ n : ℕ, n > 0 → geometric_sequence a₁' q' (n + 1) > geometric_sequence a₁' q' n) ∧
    ¬(a₁' < 0 ∧ 0 < q' ∧ q' < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2784_278445


namespace NUMINAMATH_CALUDE_product_remainder_mod_17_l2784_278468

theorem product_remainder_mod_17 :
  (2021 * 2022 * 2023 * 2024 * 2025) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_17_l2784_278468


namespace NUMINAMATH_CALUDE_shaded_fraction_of_square_l2784_278492

theorem shaded_fraction_of_square (total_squares : ℕ) (split_squares : ℕ) (triangle_area_fraction : ℚ) :
  total_squares = 16 →
  split_squares = 4 →
  triangle_area_fraction = 1/2 →
  (split_squares : ℚ) * triangle_area_fraction / total_squares = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_square_l2784_278492


namespace NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l2784_278494

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem fourth_power_of_cube_of_third_smallest_prime :
  (nthSmallestPrime 3) ^ 3 ^ 4 = 244140625 := by sorry

end NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l2784_278494


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2784_278438

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2784_278438


namespace NUMINAMATH_CALUDE_grade_distribution_equals_total_total_students_is_100_l2784_278407

-- Define the total number of students
def total_students : ℕ := 100

-- Define the number of students who received each grade
def a_students : ℕ := total_students / 5
def b_students : ℕ := total_students / 4
def c_students : ℕ := total_students / 2
def d_students : ℕ := 5

-- Theorem stating that the sum of students in each grade category equals the total number of students
theorem grade_distribution_equals_total :
  a_students + b_students + c_students + d_students = total_students := by
  sorry

-- Theorem proving that the total number of students is 100
theorem total_students_is_100 : total_students = 100 := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_equals_total_total_students_is_100_l2784_278407


namespace NUMINAMATH_CALUDE_circumscribed_quadrilateral_sides_l2784_278431

/-- A circumscribed quadrilateral with perimeter 24 and three consecutive sides in ratio 1:2:3 has sides 3, 6, 9, and 6. -/
theorem circumscribed_quadrilateral_sides (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- sides are positive
  a + b + c + d = 24 →  -- perimeter is 24
  a + c = b + d →  -- circumscribed property
  ∃ (x : ℝ), a = x ∧ b = 2*x ∧ c = 3*x →  -- consecutive sides in ratio 1:2:3
  a = 3 ∧ b = 6 ∧ c = 9 ∧ d = 6 := by
  sorry

#check circumscribed_quadrilateral_sides

end NUMINAMATH_CALUDE_circumscribed_quadrilateral_sides_l2784_278431


namespace NUMINAMATH_CALUDE_log_of_expression_l2784_278486

theorem log_of_expression (x : ℝ) : 
  x = 125 * Real.rpow 25 (1/3) * Real.sqrt 25 → 
  Real.log x / Real.log 5 = 14/3 := by
sorry

end NUMINAMATH_CALUDE_log_of_expression_l2784_278486


namespace NUMINAMATH_CALUDE_erasers_per_box_l2784_278418

theorem erasers_per_box (num_boxes : ℕ) (price_per_eraser : ℚ) (total_money : ℚ) :
  num_boxes = 48 →
  price_per_eraser = 3/4 →
  total_money = 864 →
  (total_money / price_per_eraser) / num_boxes = 24 := by
  sorry

end NUMINAMATH_CALUDE_erasers_per_box_l2784_278418


namespace NUMINAMATH_CALUDE_unique_solution_l2784_278454

/-- Represents the ages of two people satisfying the given conditions -/
structure AgesPair where
  first : ℕ
  second : ℕ
  sum_is_35 : first + second = 35
  age_relation : 2 * first - second = second - first

/-- The unique solution to the age problem -/
theorem unique_solution : ∃! (ages : AgesPair), ages.first = 20 ∧ ages.second = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2784_278454


namespace NUMINAMATH_CALUDE_boy_age_problem_l2784_278422

theorem boy_age_problem (current_age : ℕ) (years_ago : ℕ) : 
  current_age = 10 →
  current_age = 2 * (current_age - years_ago) →
  years_ago = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_boy_age_problem_l2784_278422


namespace NUMINAMATH_CALUDE_min_sum_cube_relation_l2784_278426

theorem min_sum_cube_relation (m n : ℕ+) (h : 50 * m = n^3) : 
  (∀ m' n' : ℕ+, 50 * m' = n'^3 → m' + n' ≥ m + n) → m + n = 30 := by
sorry

end NUMINAMATH_CALUDE_min_sum_cube_relation_l2784_278426


namespace NUMINAMATH_CALUDE_triangle_angle_60_degrees_l2784_278423

theorem triangle_angle_60_degrees (A B C : Real) (hABC : A + B + C = Real.pi)
  (h_eq : Real.sin A ^ 2 - Real.sin C ^ 2 + Real.sin B ^ 2 = Real.sin A * Real.sin B) :
  C = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_60_degrees_l2784_278423


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l2784_278446

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (k : ℝ), (2 - 7*I) * (a + b*I) = k*I) : a/b = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l2784_278446


namespace NUMINAMATH_CALUDE_power_function_not_in_fourth_quadrant_l2784_278435

theorem power_function_not_in_fourth_quadrant :
  ∀ (a : ℝ) (x : ℝ), 
    a ∈ ({1, 2, 3, (1/2 : ℝ), -1} : Set ℝ) → 
    x > 0 → 
    x^a > 0 := by
  sorry

end NUMINAMATH_CALUDE_power_function_not_in_fourth_quadrant_l2784_278435


namespace NUMINAMATH_CALUDE_square_side_difference_l2784_278481

theorem square_side_difference (a b : ℝ) 
  (h1 : a + b = 20) 
  (h2 : a^2 - b^2 = 40) : 
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_square_side_difference_l2784_278481


namespace NUMINAMATH_CALUDE_painted_cube_equality_l2784_278459

/-- Represents a cube with edge length n, painted with alternating colors on adjacent faces. -/
structure PaintedCube where
  n : ℕ
  n_gt_two : n > 2

/-- The number of unit cubes with exactly one black face in a painted cube. -/
def black_face_count (cube : PaintedCube) : ℕ :=
  3 * (cube.n - 2)^2

/-- The number of unpainted unit cubes in a painted cube. -/
def unpainted_count (cube : PaintedCube) : ℕ :=
  (cube.n - 2)^3

/-- Theorem stating that the number of unit cubes with exactly one black face
    equals the number of unpainted unit cubes if and only if n = 5. -/
theorem painted_cube_equality (cube : PaintedCube) :
  black_face_count cube = unpainted_count cube ↔ cube.n = 5 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_equality_l2784_278459


namespace NUMINAMATH_CALUDE_cafe_sign_white_area_l2784_278437

/-- Represents a rectangular sign with painted letters -/
structure Sign :=
  (width : ℕ)
  (height : ℕ)
  (c_area : ℕ)
  (a_area : ℕ)
  (f_area : ℕ)
  (e_area : ℕ)

/-- Calculates the white area of the sign -/
def white_area (s : Sign) : ℕ :=
  s.width * s.height - (s.c_area + s.a_area + s.f_area + s.e_area)

/-- Theorem stating that the white area of the given sign is 66 square units -/
theorem cafe_sign_white_area :
  ∃ (s : Sign),
    s.width = 6 ∧
    s.height = 18 ∧
    s.c_area = 11 ∧
    s.a_area = 10 ∧
    s.f_area = 12 ∧
    s.e_area = 9 ∧
    white_area s = 66 :=
sorry

end NUMINAMATH_CALUDE_cafe_sign_white_area_l2784_278437


namespace NUMINAMATH_CALUDE_square_side_length_l2784_278478

theorem square_side_length (d : ℝ) (h : d = 4) : 
  ∃ (s : ℝ), s * s + s * s = d * d ∧ s = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2784_278478


namespace NUMINAMATH_CALUDE_amazing_triangle_exists_l2784_278489

theorem amazing_triangle_exists : ∃ (a b c : ℕ+), 
  (a.val ^ 2 + b.val ^ 2 = c.val ^ 2) ∧ 
  (∃ (d0 d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
    d0 < 10 ∧ d1 < 10 ∧ d2 < 10 ∧ d3 < 10 ∧ d4 < 10 ∧ 
    d5 < 10 ∧ d6 < 10 ∧ d7 < 10 ∧ d8 < 10 ∧
    d0 ≠ d1 ∧ d0 ≠ d2 ∧ d0 ≠ d3 ∧ d0 ≠ d4 ∧ d0 ≠ d5 ∧ d0 ≠ d6 ∧ d0 ≠ d7 ∧ d0 ≠ d8 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧
    d7 ≠ d8 ∧
    a.val = d0 * 100 + d1 * 10 + d2 ∧
    b.val = d3 * 100 + d4 * 10 + d5 ∧
    c.val = d6 * 100 + d7 * 10 + d8) :=
by sorry

end NUMINAMATH_CALUDE_amazing_triangle_exists_l2784_278489


namespace NUMINAMATH_CALUDE_initial_points_count_l2784_278419

theorem initial_points_count (k : ℕ) : 
  k > 0 → 4 * k - 3 = 101 → k = 26 := by
  sorry

end NUMINAMATH_CALUDE_initial_points_count_l2784_278419


namespace NUMINAMATH_CALUDE_inequality_solution_l2784_278402

theorem inequality_solution (x : ℝ) (h1 : x > 0) 
  (h2 : x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2784_278402


namespace NUMINAMATH_CALUDE_solve_inequality_part1_solve_inequality_part2_l2784_278404

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

-- Part 1: Solve f(x) < 0 when a = -4
theorem solve_inequality_part1 : 
  ∀ x : ℝ, f (-4) x < 0 ↔ 1 < x ∧ x < 3 :=
sorry

-- Part 2: Find range of a when f(x) > 0 for all real x
theorem solve_inequality_part2 : 
  (∀ x : ℝ, f a x > 0) ↔ -2 * Real.sqrt 3 < a ∧ a < 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_solve_inequality_part1_solve_inequality_part2_l2784_278404


namespace NUMINAMATH_CALUDE_students_with_all_pets_l2784_278432

theorem students_with_all_pets (total_students : ℕ) 
  (dog_ratio : ℚ) (cat_ratio : ℚ) (other_pets : ℕ) (no_pets : ℕ)
  (only_dogs : ℕ) (dogs_and_other : ℕ) (only_cats : ℕ) :
  total_students = 40 →
  dog_ratio = 1/2 →
  cat_ratio = 2/5 →
  other_pets = 8 →
  no_pets = 7 →
  only_dogs = 12 →
  dogs_and_other = 3 →
  only_cats = 11 →
  ∃ (all_pets : ℕ),
    all_pets = 5 ∧
    total_students * dog_ratio = only_dogs + dogs_and_other + all_pets ∧
    total_students * cat_ratio = only_cats + all_pets ∧
    other_pets = dogs_and_other + all_pets ∧
    total_students - no_pets = only_dogs + dogs_and_other + only_cats + all_pets :=
by sorry

end NUMINAMATH_CALUDE_students_with_all_pets_l2784_278432


namespace NUMINAMATH_CALUDE_tangent_identity_l2784_278472

theorem tangent_identity (β : ℝ) : 
  Real.tan (6 * β) - Real.tan (4 * β) - Real.tan (2 * β) = 
  Real.tan (6 * β) * Real.tan (4 * β) * Real.tan (2 * β) := by
  sorry

end NUMINAMATH_CALUDE_tangent_identity_l2784_278472


namespace NUMINAMATH_CALUDE_correct_num_dogs_l2784_278484

/-- Represents the number of dogs Carly worked on --/
def num_dogs : ℕ := 11

/-- Represents the total number of nails trimmed --/
def total_nails : ℕ := 164

/-- Represents the number of dogs with three legs --/
def three_legged_dogs : ℕ := 3

/-- Represents the number of dogs with three nails on one paw --/
def three_nailed_dogs : ℕ := 2

/-- Represents the number of dogs with an extra nail on one paw --/
def extra_nailed_dogs : ℕ := 1

/-- Represents the number of nails on a regular dog --/
def nails_per_regular_dog : ℕ := 4 * 4

/-- Theorem stating that the number of dogs is correct given the conditions --/
theorem correct_num_dogs :
  num_dogs * nails_per_regular_dog
  - three_legged_dogs * 4
  - three_nailed_dogs
  + extra_nailed_dogs
  = total_nails :=
by sorry

end NUMINAMATH_CALUDE_correct_num_dogs_l2784_278484


namespace NUMINAMATH_CALUDE_carnation_count_flower_vase_problem_l2784_278497

/-- Proves that the number of carnations is 7 given the problem conditions -/
theorem carnation_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (vase_capacity : ℕ) (num_roses : ℕ) (num_vases : ℕ) (num_carnations : ℕ) =>
    (vase_capacity = 6 ∧ num_roses = 47 ∧ num_vases = 9) →
    (num_vases * vase_capacity = num_roses + num_carnations) →
    num_carnations = 7

/-- The main theorem stating the solution to the flower vase problem -/
theorem flower_vase_problem : ∃ (c : ℕ), carnation_count 6 47 9 c := by
  sorry

end NUMINAMATH_CALUDE_carnation_count_flower_vase_problem_l2784_278497


namespace NUMINAMATH_CALUDE_sum_of_digits_seven_to_seventeen_l2784_278461

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_seven_to_seventeen (h : ℕ) :
  h = 7^17 →
  tens_digit h + ones_digit h = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_seven_to_seventeen_l2784_278461


namespace NUMINAMATH_CALUDE_second_rack_dvds_l2784_278467

def dvd_sequence (n : ℕ) : ℕ → ℕ 
  | 0 => 2
  | i + 1 => 2 * dvd_sequence n i

theorem second_rack_dvds (n : ℕ) (h : n ≥ 5) :
  dvd_sequence n 0 = 2 ∧
  dvd_sequence n 2 = 8 ∧
  dvd_sequence n 3 = 16 ∧
  dvd_sequence n 4 = 32 ∧
  dvd_sequence n 5 = 64 →
  dvd_sequence n 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_second_rack_dvds_l2784_278467


namespace NUMINAMATH_CALUDE_pipe_equivalence_l2784_278444

/-- The number of smaller pipes needed to match the water-carrying capacity of a larger pipe -/
theorem pipe_equivalence (r_large r_small : ℝ) (h_large : r_large = 4) (h_small : r_small = 1) :
  (π * r_large ^ 2) / (π * r_small ^ 2) = 16 := by
  sorry

#check pipe_equivalence

end NUMINAMATH_CALUDE_pipe_equivalence_l2784_278444


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_digit_sum_l2784_278466

/-- Define a function that creates a number with n ones -/
def ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Sum of digits of a number with all ones -/
def sum_of_digits (n : ℕ) : ℕ := n

/-- The main theorem -/
theorem infinitely_many_divisible_by_digit_sum :
  ∀ n : ℕ, (ones (3^n)) % (sum_of_digits (3^n)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_digit_sum_l2784_278466


namespace NUMINAMATH_CALUDE_average_b_c_l2784_278449

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : a - c = 80) : 
  (b + c) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_b_c_l2784_278449


namespace NUMINAMATH_CALUDE_curve_tangent_acute_angle_l2784_278457

/-- The curve C: y = x^3 - 2ax^2 + 2ax -/
def C (a : ℤ) (x : ℝ) : ℝ := x^3 - 2*a*x^2 + 2*a*x

/-- The derivative of C with respect to x -/
def C_derivative (a : ℤ) (x : ℝ) : ℝ := 3*x^2 - 4*a*x + 2*a

/-- The condition that the tangent line has an acute angle with the x-axis -/
def acute_angle_condition (a : ℤ) : Prop :=
  ∀ x : ℝ, C_derivative a x > 0

theorem curve_tangent_acute_angle (a : ℤ) (h : acute_angle_condition a) : a = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_tangent_acute_angle_l2784_278457


namespace NUMINAMATH_CALUDE_log_2_15_in_terms_of_a_b_l2784_278463

/-- Given a = log₃6 and b = log₅20, prove that log₂15 = (2a + b - 3) / ((a - 1)(b - 1)) -/
theorem log_2_15_in_terms_of_a_b (a b : ℝ) 
  (ha : a = Real.log 6 / Real.log 3) 
  (hb : b = Real.log 20 / Real.log 5) : 
  Real.log 15 / Real.log 2 = (2 * a + b - 3) / ((a - 1) * (b - 1)) := by
  sorry

end NUMINAMATH_CALUDE_log_2_15_in_terms_of_a_b_l2784_278463


namespace NUMINAMATH_CALUDE_min_dot_product_l2784_278465

open Real

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the fixed point M
def M : ℝ × ℝ := (1, 0)

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the vector from M to a point P
def vector_MP (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 - M.1, P.2 - M.2)

theorem min_dot_product :
  ∃ (min : ℝ),
    (∀ (A B : ℝ × ℝ),
      ellipse A.1 A.2 →
      ellipse B.1 B.2 →
      dot_product (vector_MP A) (vector_MP B) = 0 →
      dot_product (vector_MP A) (A.1 - B.1, A.2 - B.2) ≥ min) ∧
    (∃ (A B : ℝ × ℝ),
      ellipse A.1 A.2 ∧
      ellipse B.1 B.2 ∧
      dot_product (vector_MP A) (vector_MP B) = 0 ∧
      dot_product (vector_MP A) (A.1 - B.1, A.2 - B.2) = min) ∧
    min = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_l2784_278465


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2784_278425

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^6 + b^6) / (a + b)^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2784_278425


namespace NUMINAMATH_CALUDE_tom_tickets_l2784_278408

/-- The number of tickets Tom has left after playing games and spending some tickets -/
def tickets_left (whack_a_mole skee_ball ring_toss hat plush_toy : ℕ) : ℕ :=
  (whack_a_mole + skee_ball + ring_toss) - (hat + plush_toy)

/-- Theorem stating that Tom is left with 100 tickets -/
theorem tom_tickets : 
  tickets_left 45 38 52 12 23 = 100 := by
  sorry

end NUMINAMATH_CALUDE_tom_tickets_l2784_278408


namespace NUMINAMATH_CALUDE_x_power_3a_minus_2b_l2784_278441

theorem x_power_3a_minus_2b (x a b : ℝ) (h1 : x^a = 3) (h2 : x^b = 4) :
  x^(3*a - 2*b) = 27/16 := by
sorry

end NUMINAMATH_CALUDE_x_power_3a_minus_2b_l2784_278441


namespace NUMINAMATH_CALUDE_nancy_small_gardens_l2784_278476

/-- The number of small gardens Nancy had given her seed distribution --/
def small_gardens_count (total_seeds capsicum_seeds cucumber_seeds tomato_seeds big_garden_tomato : ℕ) : ℕ :=
  let remaining_tomato := tomato_seeds - big_garden_tomato
  remaining_tomato / 2

theorem nancy_small_gardens 
  (h1 : total_seeds = 85)
  (h2 : tomato_seeds = 42)
  (h3 : capsicum_seeds = 26)
  (h4 : cucumber_seeds = 17)
  (h5 : big_garden_tomato = 24)
  (h6 : total_seeds = tomato_seeds + capsicum_seeds + cucumber_seeds) :
  small_gardens_count total_seeds capsicum_seeds cucumber_seeds tomato_seeds big_garden_tomato = 9 := by
  sorry

#eval small_gardens_count 85 26 17 42 24

end NUMINAMATH_CALUDE_nancy_small_gardens_l2784_278476


namespace NUMINAMATH_CALUDE_bob_and_alice_heights_l2784_278488

/-- The problem statement about Bob and Alice's heights --/
theorem bob_and_alice_heights :
  ∀ (initial_height : ℝ) (bob_growth_percent : ℝ) (alice_growth_ratio : ℝ) (bob_final_height : ℝ),
  initial_height > 0 →
  bob_growth_percent = 0.25 →
  alice_growth_ratio = 1/3 →
  bob_final_height = 75 →
  bob_final_height = initial_height * (1 + bob_growth_percent) →
  let bob_growth_inches := initial_height * bob_growth_percent
  let alice_growth_inches := bob_growth_inches * alice_growth_ratio
  let alice_final_height := initial_height + alice_growth_inches
  alice_final_height = 65 := by
sorry


end NUMINAMATH_CALUDE_bob_and_alice_heights_l2784_278488


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l2784_278455

theorem root_difference_implies_k_value :
  ∀ (k : ℝ) (r s : ℝ),
  (r^2 + k*r + 12 = 0) ∧ (s^2 + k*s + 12 = 0) →
  ((r+3)^2 - k*(r+3) + 12 = 0) ∧ ((s+3)^2 - k*(s+3) + 12 = 0) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l2784_278455


namespace NUMINAMATH_CALUDE_sqrt_two_squared_l2784_278433

theorem sqrt_two_squared : (Real.sqrt 2) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_l2784_278433


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2784_278471

theorem no_positive_integer_solutions :
  ¬∃ (a b : ℕ+), 4 * (a^2 + a) = b^2 + b := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2784_278471


namespace NUMINAMATH_CALUDE_triangle_inequality_four_points_l2784_278493

-- Define a metric space
variable {X : Type*} [MetricSpace X]

-- Define four points in the metric space
variable (A B C D : X)

-- State the theorem
theorem triangle_inequality_four_points :
  dist A D ≤ dist A B + dist B C + dist C D := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_four_points_l2784_278493


namespace NUMINAMATH_CALUDE_probability_of_black_piece_l2784_278496

def total_pieces : ℕ := 15
def black_pieces : ℕ := 10
def white_pieces : ℕ := 5

theorem probability_of_black_piece :
  (black_pieces : ℚ) / total_pieces = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_of_black_piece_l2784_278496


namespace NUMINAMATH_CALUDE_bianca_candy_problem_l2784_278499

/-- Bianca's Halloween candy problem -/
theorem bianca_candy_problem (initial_candy : ℕ) (piles : ℕ) (pieces_per_pile : ℕ) 
  (h1 : initial_candy = 32)
  (h2 : piles = 4)
  (h3 : pieces_per_pile = 5) :
  initial_candy - (piles * pieces_per_pile) = 12 := by
  sorry

end NUMINAMATH_CALUDE_bianca_candy_problem_l2784_278499


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2784_278469

/-- A regular polygon with perimeter 180 cm and side length 15 cm has 12 sides -/
theorem regular_polygon_sides (P : ℝ) (s : ℝ) (n : ℕ) 
  (h_perimeter : P = 180) 
  (h_side : s = 15) 
  (h_regular : P = n * s) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2784_278469


namespace NUMINAMATH_CALUDE_correct_divisor_l2784_278434

theorem correct_divisor (X : ℕ) (h1 : X / 72 = 24) (h2 : X / 36 = 48) : 36 = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l2784_278434


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l2784_278491

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) :
  area = 289 → price_per_foot = 57 →
  4 * Real.sqrt area * price_per_foot = 3876 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_square_plot_l2784_278491


namespace NUMINAMATH_CALUDE_interior_angle_of_regular_polygon_with_five_diagonals_l2784_278495

/-- Given a regular polygon where at most 5 diagonals can be drawn from a vertex,
    prove that one of its interior angles measures 135°. -/
theorem interior_angle_of_regular_polygon_with_five_diagonals :
  ∀ (n : ℕ), 
    n ≥ 3 →  -- Ensures it's a valid polygon
    n - 3 = 5 →  -- At most 5 diagonals can be drawn from a vertex
    (180 * (n - 2) : ℝ) / n = 135 :=  -- One interior angle measures 135°
by sorry

end NUMINAMATH_CALUDE_interior_angle_of_regular_polygon_with_five_diagonals_l2784_278495


namespace NUMINAMATH_CALUDE_garden_area_l2784_278427

/-- A rectangular garden with specific length-width relationship and perimeter has an area of 12000 square meters. -/
theorem garden_area (w : ℝ) (h1 : w > 0) : 
  let l := 3 * w + 20
  2 * l + 2 * w = 520 →
  w * l = 12000 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l2784_278427


namespace NUMINAMATH_CALUDE_cubic_sum_l2784_278460

theorem cubic_sum (x y : ℝ) (h1 : 1/x + 1/y = 4) (h2 : x + y + x*y = 3) :
  x^3 + y^3 = 1188/125 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_l2784_278460


namespace NUMINAMATH_CALUDE_intercept_triangle_area_zero_l2784_278485

/-- The cubic function f(x) = x³ - x --/
def f (x : ℝ) : ℝ := x^3 - x

/-- The set of x-intercepts of the curve y = x³ - x --/
def x_intercepts : Set ℝ := {x : ℝ | f x = 0}

/-- The y-intercept of the curve y = x³ - x --/
def y_intercept : ℝ × ℝ := (0, f 0)

/-- The area of the triangle formed by the intercepts of the curve y = x³ - x --/
def triangle_area : ℝ := sorry

/-- Theorem: The area of the triangle formed by the intercepts of y = x³ - x is 0 --/
theorem intercept_triangle_area_zero : triangle_area = 0 := by sorry

end NUMINAMATH_CALUDE_intercept_triangle_area_zero_l2784_278485


namespace NUMINAMATH_CALUDE_array_transformation_theorem_l2784_278415

/-- Represents an 8x8 array of +1 and -1 -/
def Array8x8 := Fin 8 → Fin 8 → Int

/-- Represents a move in the array -/
structure Move where
  row : Fin 8
  col : Fin 8

/-- Applies a move to an array -/
def applyMove (arr : Array8x8) (m : Move) : Array8x8 :=
  fun i j => if i = m.row ∨ j = m.col then -arr i j else arr i j

/-- Checks if an array is all +1 -/
def isAllPlusOne (arr : Array8x8) : Prop :=
  ∀ i j, arr i j = 1

theorem array_transformation_theorem :
  ∀ (initial : Array8x8),
  (∀ i j, initial i j = 1 ∨ initial i j = -1) →
  ∃ (moves : List Move),
  isAllPlusOne (moves.foldl applyMove initial) :=
sorry

end NUMINAMATH_CALUDE_array_transformation_theorem_l2784_278415


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2784_278456

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + d * (n - 1)

theorem arithmetic_sequence_formula :
  let a := arithmetic_sequence 2 3
  ∀ n : ℕ, a n = 3 * n - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2784_278456


namespace NUMINAMATH_CALUDE_square_root_problem_l2784_278414

theorem square_root_problem (a b : ℝ) 
  (h1 : Real.sqrt (a + 9) = -5)
  (h2 : (2 * b - a) ^ (1/3 : ℝ) = -2) :
  Real.sqrt (2 * a + b) = 6 := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l2784_278414


namespace NUMINAMATH_CALUDE_optimal_distribution_l2784_278482

/-- Represents the production process for assembling products -/
structure ProductionProcess where
  totalWorkers : ℕ
  productsToAssemble : ℕ
  paintTime : ℕ
  dryTime : ℕ
  assemblyTime : ℕ

/-- Represents the distribution of workers -/
structure WorkerDistribution where
  painters : ℕ
  assemblers : ℕ

/-- Calculates the production time for a given worker distribution -/
def productionTime (process : ProductionProcess) (dist : WorkerDistribution) : ℕ :=
  sorry

/-- Checks if a worker distribution is valid for the given process -/
def isValidDistribution (process : ProductionProcess) (dist : WorkerDistribution) : Prop :=
  dist.painters + dist.assemblers ≤ process.totalWorkers

/-- Theorem stating the optimal worker distribution for the given process -/
theorem optimal_distribution (process : ProductionProcess) 
  (h1 : process.totalWorkers = 10)
  (h2 : process.productsToAssemble = 50)
  (h3 : process.paintTime = 10)
  (h4 : process.dryTime = 5)
  (h5 : process.assemblyTime = 20) :
  ∃ (optDist : WorkerDistribution), 
    optDist.painters = 3 ∧ 
    optDist.assemblers = 6 ∧
    isValidDistribution process optDist ∧
    ∀ (dist : WorkerDistribution), 
      isValidDistribution process dist → 
      productionTime process optDist ≤ productionTime process dist :=
  sorry

end NUMINAMATH_CALUDE_optimal_distribution_l2784_278482


namespace NUMINAMATH_CALUDE_new_year_gift_exchange_l2784_278498

/-- Represents a group of friends exchanging gifts -/
structure GiftExchange where
  num_friends : Nat
  num_exchanges : Nat

/-- Predicate to check if the number of friends receiving 4 gifts is valid -/
def valid_four_gift_recipients (ge : GiftExchange) (n : Nat) : Prop :=
  n = 2 ∨ n = 4

/-- Theorem stating that in the given scenario, the number of friends receiving 4 gifts is either 2 or 4 -/
theorem new_year_gift_exchange (ge : GiftExchange) 
  (h1 : ge.num_friends = 6)
  (h2 : ge.num_exchanges = 13) :
  ∃ n : Nat, valid_four_gift_recipients ge n := by
  sorry

end NUMINAMATH_CALUDE_new_year_gift_exchange_l2784_278498


namespace NUMINAMATH_CALUDE_bottles_from_625_l2784_278417

/-- The number of new bottles that can be made from a given number of initial bottles,
    where 5 bottles are needed to make 1 new bottle. -/
def bottles_made (initial_bottles : ℕ) : ℕ :=
  if initial_bottles < 5 then 0
  else (initial_bottles / 5) + bottles_made (initial_bottles / 5)

/-- Theorem stating that 195 new bottles can be made from 625 initial bottles. -/
theorem bottles_from_625 : bottles_made 625 = 195 := by
  sorry

end NUMINAMATH_CALUDE_bottles_from_625_l2784_278417


namespace NUMINAMATH_CALUDE_max_value_constraint_l2784_278490

theorem max_value_constraint (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) :
  ∃ (max : ℝ), (∀ x' y' : ℝ, 4 * x'^2 + y'^2 + x' * y' = 1 → 2 * x' + y' ≤ max) ∧
                max = 2 * Real.sqrt 10 / 5 :=
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2784_278490


namespace NUMINAMATH_CALUDE_intersection_unique_l2784_278411

/-- The line is defined by (x-2)/1 = (y-3)/1 = (z-4)/2 -/
def line (x y z : ℝ) : Prop :=
  (x - 2) = (y - 3) ∧ (x - 2) = (z - 4) / 2

/-- The plane is defined by 2X + Y + Z = 0 -/
def plane (x y z : ℝ) : Prop :=
  2 * x + y + z = 0

/-- The point of intersection -/
def intersection_point : ℝ × ℝ × ℝ := (-0.2, 0.8, -0.4)

theorem intersection_unique :
  ∃! p : ℝ × ℝ × ℝ, line p.1 p.2.1 p.2.2 ∧ plane p.1 p.2.1 p.2.2 ∧ p = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_unique_l2784_278411


namespace NUMINAMATH_CALUDE_power_seven_mod_twelve_l2784_278401

theorem power_seven_mod_twelve : 7^93 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_twelve_l2784_278401


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l2784_278473

/-- A convex polygon -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ vertices
  finite : Finite vertices

/-- The perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- A triangle formed by three vertices of a polygon -/
def triangle_of_polygon (p : ConvexPolygon) : Set (ConvexPolygon) := sorry

theorem triangle_perimeter_bound (G : ConvexPolygon) :
  ∃ T ∈ triangle_of_polygon G, perimeter T ≥ 0.7 * perimeter G := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l2784_278473


namespace NUMINAMATH_CALUDE_circle_properties_l2784_278458

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 5

-- Define the points A and B
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (3, 0)

-- Define the line x - y = 0
def center_line (x y : ℝ) : Prop := x = y

-- Define the line x + 2y + 4 = 0
def distance_line (x y : ℝ) : Prop := x + 2*y + 4 = 0

-- Main theorem
theorem circle_properties :
  -- The circle passes through A and B
  circle_C point_A.1 point_A.2 ∧ circle_C point_B.1 point_B.2 ∧
  -- The center is on the line x - y = 0
  ∃ x y, circle_C x y ∧ center_line x y ∧
  -- Maximum and minimum distances
  (∀ x y, circle_C x y →
    (∃ d_max, d_max = (12/5)*Real.sqrt 5 ∧
      ∀ d, (∃ x' y', distance_line x' y' ∧ d = Real.sqrt ((x - x')^2 + (y - y')^2)) → d ≤ d_max) ∧
    (∃ d_min, d_min = (2/5)*Real.sqrt 5 ∧
      ∀ d, (∃ x' y', distance_line x' y' ∧ d = Real.sqrt ((x - x')^2 + (y - y')^2)) → d ≥ d_min)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2784_278458


namespace NUMINAMATH_CALUDE_ferris_wheel_rides_correct_l2784_278420

/-- The number of times Billy rode the ferris wheel -/
def ferris_wheel_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_car_rides : ℕ := 3

/-- The cost of each ride in tickets -/
def cost_per_ride : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := 50

/-- Theorem stating that the number of ferris wheel rides is correct -/
theorem ferris_wheel_rides_correct : 
  ferris_wheel_rides * cost_per_ride + bumper_car_rides * cost_per_ride = total_tickets :=
by sorry

end NUMINAMATH_CALUDE_ferris_wheel_rides_correct_l2784_278420


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2784_278424

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_uniqueness
  (f : ℝ → ℝ)
  (h_quad : is_quadratic f)
  (h_solution_set : ∀ x, f x < 0 ↔ 0 < x ∧ x < 5)
  (h_max_value : ∀ x ∈ Set.Icc (-1) 4, f x ≤ 12)
  (h_attains_max : ∃ x ∈ Set.Icc (-1) 4, f x = 12) :
  ∀ x, f x = 2 * x^2 - 10 * x :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2784_278424


namespace NUMINAMATH_CALUDE_sin_negative_945_degrees_l2784_278452

theorem sin_negative_945_degrees : Real.sin ((-945 : ℝ) * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_945_degrees_l2784_278452


namespace NUMINAMATH_CALUDE_two_times_three_plus_two_l2784_278450

theorem two_times_three_plus_two :
  (2 : ℕ) * 3 + 2 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_two_times_three_plus_two_l2784_278450


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2784_278477

-- Define the repeating decimal 7.036̅
def repeating_decimal : ℚ := 7 + 36 / 999

-- Theorem statement
theorem repeating_decimal_as_fraction :
  repeating_decimal = 781 / 111 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2784_278477


namespace NUMINAMATH_CALUDE_amusement_park_admission_l2784_278405

theorem amusement_park_admission (child_fee adult_fee : ℚ) 
  (total_people : ℕ) (total_fees : ℚ) :
  child_fee = 3/2 →
  adult_fee = 4 →
  total_people = 315 →
  total_fees = 810 →
  ∃ (children adults : ℕ),
    children + adults = total_people ∧
    child_fee * children + adult_fee * adults = total_fees ∧
    children = 180 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_admission_l2784_278405


namespace NUMINAMATH_CALUDE_parsley_rows_juvy_parsley_rows_l2784_278448

/-- Calculates the number of rows planted with parsley in Juvy's garden. -/
theorem parsley_rows (total_rows : Nat) (plants_per_row : Nat) (rosemary_rows : Nat) (chives_count : Nat) : Nat :=
  let remaining_rows := total_rows - rosemary_rows
  let chives_rows := chives_count / plants_per_row
  remaining_rows - chives_rows

/-- Proves that Juvy plants parsley in 3 rows given the garden's conditions. -/
theorem juvy_parsley_rows : parsley_rows 20 10 2 150 = 3 := by
  sorry

end NUMINAMATH_CALUDE_parsley_rows_juvy_parsley_rows_l2784_278448


namespace NUMINAMATH_CALUDE_expansion_equality_l2784_278413

theorem expansion_equality (y : ℝ) : 5 * (6 * y^2 - 3 * y + 2) = 30 * y^2 - 15 * y + 10 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l2784_278413


namespace NUMINAMATH_CALUDE_distribute_6_5_l2784_278406

def distribute (n m : ℕ) : ℕ := 
  Nat.choose (m - 1) (n - m)

theorem distribute_6_5 : distribute 6 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_6_5_l2784_278406


namespace NUMINAMATH_CALUDE_all_integers_are_cute_l2784_278428

/-- An integer is cute if it can be written as a^2 + b^3 + c^3 + d^5 for some integers a, b, c, and d. -/
def IsCute (n : ℤ) : Prop :=
  ∃ a b c d : ℤ, n = a^2 + b^3 + c^3 + d^5

/-- All integers are cute. -/
theorem all_integers_are_cute : ∀ n : ℤ, IsCute n := by
  sorry


end NUMINAMATH_CALUDE_all_integers_are_cute_l2784_278428


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2784_278451

/-- Two planar vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- Given two parallel planar vectors (3, 1) and (x, -3), x equals -9 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (3, 1) (x, -3) → x = -9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2784_278451


namespace NUMINAMATH_CALUDE_survey_preferences_l2784_278430

theorem survey_preferences (total : ℕ) (mac_pref : ℕ) (windows_pref : ℕ) : 
  total = 210 →
  mac_pref = 60 →
  windows_pref = 40 →
  ∃ (no_pref : ℕ),
    no_pref = total - (mac_pref + windows_pref + (mac_pref / 3)) ∧
    no_pref = 90 :=
by sorry

end NUMINAMATH_CALUDE_survey_preferences_l2784_278430


namespace NUMINAMATH_CALUDE_stadium_length_conversion_l2784_278416

/-- Conversion factor from feet to yards -/
def feet_per_yard : ℚ := 3

/-- Length of the stadium in feet -/
def stadium_length_feet : ℚ := 183

/-- Length of the stadium in yards -/
def stadium_length_yards : ℚ := stadium_length_feet / feet_per_yard

theorem stadium_length_conversion :
  stadium_length_yards = 61 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_conversion_l2784_278416


namespace NUMINAMATH_CALUDE_chess_tournament_matches_l2784_278421

/-- The number of matches in a round-robin tournament with n players -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_matches :
  num_matches 10 = 45 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_matches_l2784_278421


namespace NUMINAMATH_CALUDE_system_solution_l2784_278447

theorem system_solution (x y a : ℝ) : 
  x - 2*y = a - 6 →
  2*x + 5*y = 2*a →
  x + y = 9 →
  a = 11 := by sorry

end NUMINAMATH_CALUDE_system_solution_l2784_278447


namespace NUMINAMATH_CALUDE_horner_method_proof_l2784_278480

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

-- Define Horner's method for this specific polynomial
def horner (x : ℝ) : ℝ := 
  (((((x - 12) * x + 60) * x - 160) * x + 240) * x - 192) * x + 64

-- Theorem statement
theorem horner_method_proof : f 2 = -80 ∧ f 2 = horner 2 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_proof_l2784_278480


namespace NUMINAMATH_CALUDE_range_of_a_range_of_b_l2784_278474

-- Define propositions
def p (a : ℝ) : Prop := ∀ x, 2^x + 1 ≥ a

def q (a : ℝ) : Prop := ∀ x, a * x^2 - x + a > 0

def m (a b : ℝ) : Prop := ∃ x, x^2 + b*x + a = 0

-- Theorem for part (1)
theorem range_of_a : 
  (∃ a, p a ∧ q a) → (∀ a, p a ∧ q a → a > 1/2 ∧ a ≤ 1) :=
sorry

-- Theorem for part (2)
theorem range_of_b :
  (∀ a b, (¬p a → ¬m a b) ∧ ¬(m a b → ¬p a)) →
  (∀ b, (∃ a, ¬p a ∧ m a b) → b > -2 ∧ b < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_b_l2784_278474


namespace NUMINAMATH_CALUDE_f_6_equals_16_l2784_278483

def f : ℕ → ℕ 
  | x => if x < 5 then 2^x else f (x-1)

theorem f_6_equals_16 : f 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_f_6_equals_16_l2784_278483


namespace NUMINAMATH_CALUDE_siblings_ages_l2784_278442

/-- Represents the ages of three siblings --/
structure SiblingsAges where
  david : ℕ
  yuan : ℕ
  maria : ℕ

/-- Conditions for the siblings' ages --/
def validAges (ages : SiblingsAges) : Prop :=
  ages.yuan = ages.david + 7 ∧
  ages.yuan = 2 * ages.david ∧
  ages.maria = ages.david + 4 ∧
  2 * ages.maria = ages.yuan

theorem siblings_ages :
  ∃ (ages : SiblingsAges), validAges ages ∧ ages.david = 7 ∧ ages.maria = 11 := by
  sorry

end NUMINAMATH_CALUDE_siblings_ages_l2784_278442


namespace NUMINAMATH_CALUDE_find_m_value_l2784_278403

theorem find_m_value (m : ℝ) (h1 : m ≠ 0) :
  (∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 7)) →
  m = 7 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l2784_278403


namespace NUMINAMATH_CALUDE_gcd_of_840_and_1764_l2784_278453

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_840_and_1764_l2784_278453


namespace NUMINAMATH_CALUDE_tan_315_degrees_l2784_278462

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l2784_278462


namespace NUMINAMATH_CALUDE_max_log_sin_l2784_278409

open Real

theorem max_log_sin (x : ℝ) (h : 0 < x ∧ x < π) : 
  ∃ c : ℝ, c = 0 ∧ ∀ y : ℝ, 0 < y ∧ y < π → log (sin y) ≤ c :=
by sorry

end NUMINAMATH_CALUDE_max_log_sin_l2784_278409


namespace NUMINAMATH_CALUDE_cube_surface_area_l2784_278479

/-- Given a cube with side perimeter 24 cm, prove its surface area is 216 cm^2 -/
theorem cube_surface_area (side_perimeter : ℝ) (h : side_perimeter = 24) :
  let edge_length := side_perimeter / 4
  let surface_area := 6 * edge_length^2
  surface_area = 216 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2784_278479


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2784_278443

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 5 = 0 → x₂^2 - 2*x₂ - 5 = 0 → x₁ + x₂ + 3*x₁*x₂ = -13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2784_278443


namespace NUMINAMATH_CALUDE_no_integer_distance_point_l2784_278487

theorem no_integer_distance_point (x y : ℕ) (hx : Odd x) (hy : Odd y) :
  ¬ ∃ (a d : ℝ), 0 < a ∧ a < x ∧ 0 < d ∧ d < y ∧
    (∃ (w x y z : ℕ), 
      a^2 + d^2 = (w : ℝ)^2 ∧
      (x - a)^2 + d^2 = (x : ℝ)^2 ∧
      a^2 + (y - d)^2 = (y : ℝ)^2 ∧
      (x - a)^2 + (y - d)^2 = (z : ℝ)^2) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_distance_point_l2784_278487


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2784_278436

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2784_278436


namespace NUMINAMATH_CALUDE_no_real_some_complex_solutions_l2784_278475

-- Define the system of equations
def equation1 (x y : ℂ) : Prop := y = (x + 1)^2
def equation2 (x y : ℂ) : Prop := x * y^2 + y = 1

-- Theorem statement
theorem no_real_some_complex_solutions :
  (∀ x y : ℝ, ¬(equation1 x y ∧ equation2 x y)) ∧
  (∃ x y : ℂ, equation1 x y ∧ equation2 x y) :=
sorry

end NUMINAMATH_CALUDE_no_real_some_complex_solutions_l2784_278475


namespace NUMINAMATH_CALUDE_inequality_solution_l2784_278400

theorem inequality_solution (p q r : ℝ) 
  (h1 : ∀ x : ℝ, ((x - p) * (x - q)) / (x - r) ≥ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2))
  (h2 : p < q) : 
  p + 2*q + 3*r = 78 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2784_278400


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2784_278410

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_equality : M ∩ (U \ N) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2784_278410
