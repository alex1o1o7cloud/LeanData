import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_f_geq_4_range_of_a_l2985_298563

-- Define the function f
def f (x : ℝ) := |1 - 2*x| - |1 + x|

-- Theorem for the solution set of f(x) ≥ 4
theorem solution_set_f_geq_4 : 
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 6} := by sorry

-- Theorem for the range of a
theorem range_of_a : 
  {a : ℝ | ∃ x, a^2 + 2*a + |1 + x| < f x} = {a : ℝ | -3 < a ∧ a < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_4_range_of_a_l2985_298563


namespace NUMINAMATH_CALUDE_system_solution_unique_l2985_298593

theorem system_solution_unique :
  ∃! (x y : ℚ), 3 * x + 2 * y = 5 ∧ x - 2 * y = 11 ∧ x = 4 ∧ y = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2985_298593


namespace NUMINAMATH_CALUDE_simplify_expression_l2985_298581

theorem simplify_expression (y : ℝ) : 3*y + 5*y + 2*y + 7*y = 17*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2985_298581


namespace NUMINAMATH_CALUDE_function_value_at_negative_two_l2985_298569

theorem function_value_at_negative_two :
  Real.sqrt (4 * (-2) + 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_two_l2985_298569


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2985_298522

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 4024 = 4) :
  a 2013 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2985_298522


namespace NUMINAMATH_CALUDE_smallest_student_count_l2985_298501

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  eighth : ℕ
  seventh : ℕ

/-- Checks if the given counts satisfy the ratio conditions --/
def satisfies_ratios (counts : GradeCount) : Prop :=
  7 * counts.seventh = 4 * counts.ninth ∧
  9 * counts.eighth = 5 * counts.ninth

/-- The total number of students --/
def total_students (counts : GradeCount) : ℕ :=
  counts.ninth + counts.eighth + counts.seventh

/-- Theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (counts : GradeCount),
    satisfies_ratios counts ∧
    total_students counts = 134 ∧
    (∀ (other : GradeCount), satisfies_ratios other → total_students other ≥ 134) := by
  sorry

end NUMINAMATH_CALUDE_smallest_student_count_l2985_298501


namespace NUMINAMATH_CALUDE_room_extension_ratio_l2985_298502

/-- Given a room with original length, width, and an extension to the length,
    prove that the ratio of the new total length to the new perimeter is 35:100. -/
theorem room_extension_ratio (original_length width extension : ℕ) 
  (h1 : original_length = 25)
  (h2 : width = 15)
  (h3 : extension = 10) :
  (original_length + extension) * 100 = 35 * (2 * (original_length + extension + width)) :=
by sorry

end NUMINAMATH_CALUDE_room_extension_ratio_l2985_298502


namespace NUMINAMATH_CALUDE_range_of_a_l2985_298518

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2985_298518


namespace NUMINAMATH_CALUDE_ninth_grade_class_distribution_l2985_298582

theorem ninth_grade_class_distribution (total students_science students_programming : ℕ) 
  (h_total : total = 120)
  (h_science : students_science = 80)
  (h_programming : students_programming = 75) :
  students_science - (total - (students_science + students_programming - total)) = 45 :=
sorry

end NUMINAMATH_CALUDE_ninth_grade_class_distribution_l2985_298582


namespace NUMINAMATH_CALUDE_polygon_sides_l2985_298533

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 4 * 360 - 180) →
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2985_298533


namespace NUMINAMATH_CALUDE_max_value_of_f_l2985_298526

-- Define the function
def f (x : ℝ) : ℝ := x * (1 - 3 * x)

-- State the theorem
theorem max_value_of_f :
  ∃ (max_y : ℝ), max_y = 1/12 ∧
  ∀ (x : ℝ), 0 < x → x < 1/3 → f x ≤ max_y :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2985_298526


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2985_298549

theorem perpendicular_vectors_x_value (x y : ℝ) :
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (3, 2 - x)
  (a.1 * b.1 + a.2 * b.2 = 0) → (x = 3 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2985_298549


namespace NUMINAMATH_CALUDE_oil_containers_per_box_l2985_298527

theorem oil_containers_per_box :
  let trucks_with_20_boxes : ℕ := 7
  let boxes_per_truck_20 : ℕ := 20
  let trucks_with_12_boxes : ℕ := 5
  let boxes_per_truck_12 : ℕ := 12
  let total_trucks_after_redistribution : ℕ := 10
  let containers_per_truck_after_redistribution : ℕ := 160

  let total_boxes : ℕ := trucks_with_20_boxes * boxes_per_truck_20 + trucks_with_12_boxes * boxes_per_truck_12
  let total_containers : ℕ := total_trucks_after_redistribution * containers_per_truck_after_redistribution

  (total_containers / total_boxes : ℚ) = 8 := by sorry

end NUMINAMATH_CALUDE_oil_containers_per_box_l2985_298527


namespace NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l2985_298562

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    is_prime d ∧ d < 20 ∧
    is_prime e ∧ e < 20 ∧
    is_prime (e^2 + 10*d) ∧
    n = d * e * (e^2 + 10*d) ∧
    (∀ (n' d' e' : ℕ),
      is_prime d' ∧ d' < 20 ∧
      is_prime e' ∧ e' < 20 ∧
      is_prime (e'^2 + 10*d') ∧
      n' = d' * e' * (e'^2 + 10*d') →
      n' ≤ n) ∧
    sum_of_digits n = 16 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l2985_298562


namespace NUMINAMATH_CALUDE_equation_solution_l2985_298512

theorem equation_solution (x : ℝ) : 3 - 1 / (2 - x) = 1 / (2 - x) → x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2985_298512


namespace NUMINAMATH_CALUDE_license_plate_count_l2985_298525

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of consonants in the alphabet -/
def consonant_count : ℕ := 21

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 5

/-- The number of odd digits -/
def odd_digit_count : ℕ := 5

/-- The number of even digits -/
def even_digit_count : ℕ := 5

/-- The total number of possible license plates -/
def total_plates : ℕ := alphabet_size * consonant_count * vowel_count * odd_digit_count * odd_digit_count * even_digit_count * even_digit_count

theorem license_plate_count : total_plates = 1706250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2985_298525


namespace NUMINAMATH_CALUDE_triangular_square_iff_pell_solution_l2985_298567

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- A solution to the Pell's equation X^2 - 8Y^2 = 1 -/
def pell_solution (x : ℕ) : Prop := ∃ y : ℕ, x^2 - 8*y^2 = 1

/-- The main theorem: a triangular number is a perfect square iff it has the form (x^2 - 1)/8
    where x is a solution to the Pell's equation X^2 - 8Y^2 = 1 -/
theorem triangular_square_iff_pell_solution :
  ∀ n : ℕ, (∃ k : ℕ, triangular_number n = k^2) ↔ 
  (∃ x : ℕ, pell_solution x ∧ triangular_number n = (x^2 - 1) / 8) :=
sorry

end NUMINAMATH_CALUDE_triangular_square_iff_pell_solution_l2985_298567


namespace NUMINAMATH_CALUDE_brownie_pieces_fit_l2985_298585

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan and brownie piece dimensions -/
def pan : Dimensions := ⟨24, 15⟩
def piece : Dimensions := ⟨3, 2⟩

/-- The number of brownie pieces that fit in the pan -/
def num_pieces : ℕ := area pan / area piece

theorem brownie_pieces_fit :
  num_pieces = 60 ∧
  area pan = num_pieces * area piece :=
sorry

end NUMINAMATH_CALUDE_brownie_pieces_fit_l2985_298585


namespace NUMINAMATH_CALUDE_jonathan_saved_eight_l2985_298504

/-- Calculates the amount of money saved given the costs of three books and the additional amount needed. -/
def money_saved (book1_cost book2_cost book3_cost additional_needed : ℕ) : ℕ :=
  (book1_cost + book2_cost + book3_cost) - additional_needed

/-- Proves that given the specific costs and additional amount needed, the money saved is 8. -/
theorem jonathan_saved_eight :
  money_saved 11 19 7 29 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jonathan_saved_eight_l2985_298504


namespace NUMINAMATH_CALUDE_f_composition_equals_one_third_l2985_298505

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 4
  else 3^x

-- State the theorem
theorem f_composition_equals_one_third :
  f (f (1/4)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_one_third_l2985_298505


namespace NUMINAMATH_CALUDE_x_value_l2985_298590

theorem x_value (x y : ℝ) (h1 : x - y = 18) (h2 : x + y = 10) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2985_298590


namespace NUMINAMATH_CALUDE_rectangle_count_l2985_298542

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem rectangle_count : 
  choose horizontal_lines 2 * choose vertical_lines 2 = 100 := by sorry

end NUMINAMATH_CALUDE_rectangle_count_l2985_298542


namespace NUMINAMATH_CALUDE_tribe_leadership_structure_l2985_298576

theorem tribe_leadership_structure (n : ℕ) (h : n = 12) : 
  n * (n - 1) * (n - 2) * (Nat.choose (n - 3) 3) * (Nat.choose (n - 6) 3) = 2217600 :=
by sorry

end NUMINAMATH_CALUDE_tribe_leadership_structure_l2985_298576


namespace NUMINAMATH_CALUDE_zeros_equality_l2985_298509

/-- 
  f(n) represents the number of 0's in the binary representation of a positive integer n
-/
def f (n : ℕ+) : ℕ := sorry

/-- 
  Theorem: For all positive integers n, 
  the number of 0's in the binary representation of 8n+7 
  is equal to the number of 0's in the binary representation of 4n+3
-/
theorem zeros_equality (n : ℕ+) : f (8*n+7) = f (4*n+3) := by sorry

end NUMINAMATH_CALUDE_zeros_equality_l2985_298509


namespace NUMINAMATH_CALUDE_window_treatment_cost_l2985_298598

def number_of_windows : ℕ := 3
def cost_of_sheers : ℚ := 40
def cost_of_drapes : ℚ := 60

def total_cost : ℚ := number_of_windows * (cost_of_sheers + cost_of_drapes)

theorem window_treatment_cost : total_cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_window_treatment_cost_l2985_298598


namespace NUMINAMATH_CALUDE_sum_of_zeros_greater_than_one_l2985_298597

open Real

theorem sum_of_zeros_greater_than_one (a : ℝ) :
  let f := fun x : ℝ => log x - a * x + 1 / (2 * x)
  let g := fun x : ℝ => f x + a * (x - 1)
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ = 0 → g x₂ = 0 → x₁ + x₂ > 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_greater_than_one_l2985_298597


namespace NUMINAMATH_CALUDE_expansion_equality_l2985_298517

theorem expansion_equality (x y : ℝ) : 25 * (3 * x + 7 - 4 * y) = 75 * x + 175 - 100 * y := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l2985_298517


namespace NUMINAMATH_CALUDE_seven_nanometers_in_meters_l2985_298548

-- Define the conversion factor for nanometers to meters
def nanometer_to_meter : ℝ := 1e-9

-- Theorem statement
theorem seven_nanometers_in_meters :
  7 * nanometer_to_meter = 7e-9 := by
  sorry

end NUMINAMATH_CALUDE_seven_nanometers_in_meters_l2985_298548


namespace NUMINAMATH_CALUDE_triangle_not_right_angle_l2985_298557

theorem triangle_not_right_angle (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0)
  (h4 : A + B + C = 180) (h5 : A / 3 = B / 4) (h6 : A / 3 = C / 5) : 
  ¬(A = 90 ∨ B = 90 ∨ C = 90) := by
sorry

end NUMINAMATH_CALUDE_triangle_not_right_angle_l2985_298557


namespace NUMINAMATH_CALUDE_third_consecutive_odd_integer_l2985_298587

theorem third_consecutive_odd_integer (x : ℤ) : 
  (∀ n : ℤ, (x + 2*n) % 2 ≠ 0) →  -- x is odd
  3*x = 2*(x + 4) + 3 →          -- condition from the problem
  x + 4 = 15 :=                  -- third integer is 15
by sorry

end NUMINAMATH_CALUDE_third_consecutive_odd_integer_l2985_298587


namespace NUMINAMATH_CALUDE_largest_reciprocal_l2985_298524

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/7 → b = 3/4 → c = 2 → d = 8 → e = 100 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l2985_298524


namespace NUMINAMATH_CALUDE_B_equals_D_l2985_298511

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}

-- Define set D
def D : Set ℝ := {y : ℝ | y ≥ 1}

-- Theorem stating that B and D are equal
theorem B_equals_D : B = D := by sorry

end NUMINAMATH_CALUDE_B_equals_D_l2985_298511


namespace NUMINAMATH_CALUDE_patricia_barrels_l2985_298561

/-- Given a scenario where Patricia has some barrels, proves that the number of barrels is 4 -/
theorem patricia_barrels : 
  ∀ (barrel_capacity : ℝ) (flow_rate : ℝ) (fill_time : ℝ) (num_barrels : ℕ),
  barrel_capacity = 7 →
  flow_rate = 3.5 →
  fill_time = 8 →
  (flow_rate * fill_time : ℝ) = (↑num_barrels * barrel_capacity) →
  num_barrels = 4 := by
sorry

end NUMINAMATH_CALUDE_patricia_barrels_l2985_298561


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2985_298514

/-- Given two real numbers a and b that are inversely proportional,
    prove that if a + b = 30 and a - b = 8, then when a = 6, b = 209/6 -/
theorem inverse_proportion_problem (a b : ℝ) (h1 : ∃ k : ℝ, a * b = k) 
    (h2 : a + b = 30) (h3 : a - b = 8) : 
    (a = 6) → (b = 209 / 6) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2985_298514


namespace NUMINAMATH_CALUDE_tan_pi_sixth_minus_alpha_l2985_298532

theorem tan_pi_sixth_minus_alpha (α : ℝ) (h : Real.sin α = 3 * Real.sin (α - π / 3)) :
  Real.tan (π / 6 - α) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_sixth_minus_alpha_l2985_298532


namespace NUMINAMATH_CALUDE_green_packs_count_l2985_298574

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The number of packs of red bouncy balls -/
def red_packs : ℕ := 4

/-- The number of packs of yellow bouncy balls -/
def yellow_packs : ℕ := 8

/-- The total number of bouncy balls bought -/
def total_balls : ℕ := 160

/-- The number of packs of green bouncy balls -/
def green_packs : ℕ := (total_balls - (red_packs + yellow_packs) * balls_per_pack) / balls_per_pack

theorem green_packs_count : green_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_green_packs_count_l2985_298574


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l2985_298540

theorem factorial_fraction_equality : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l2985_298540


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_five_l2985_298573

/-- Given two real-valued functions f and h, where f is linear and h is affine,
    and a condition relating their composition to a linear function,
    prove that the sum of the coefficients of f is 5. -/
theorem sum_of_coefficients_is_five
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h : ℝ → ℝ)
  (h_def : ∀ x, h x = 3 * x - 6)
  (f_def : ∀ x, f x = a * x + b)
  (composition_condition : ∀ x, h (f x) = 4 * x + 5) :
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_five_l2985_298573


namespace NUMINAMATH_CALUDE_triangle_with_arithmetic_sides_is_right_angled_and_inradius_equals_diff_l2985_298541

/-- A triangle with sides in arithmetic progression including its semiperimeter -/
structure TriangleWithArithmeticSides where
  /-- The common difference of the arithmetic progression -/
  d : ℝ
  /-- The middle term of the arithmetic progression -/
  a : ℝ
  /-- Ensures that the sides are positive -/
  d_pos : 0 < d
  a_pos : 0 < a
  /-- Ensures that the triangle inequality holds -/
  triangle_ineq : 2 * d < a

theorem triangle_with_arithmetic_sides_is_right_angled_and_inradius_equals_diff 
  (t : TriangleWithArithmeticSides) : 
  /- The triangle is right-angled -/
  (3 * t.a / 4) ^ 2 + (4 * t.a / 4) ^ 2 = (5 * t.a / 4) ^ 2 ∧ 
  /- The common difference equals the inradius -/
  t.d = (3 * t.a / 4 + 4 * t.a / 4 - 5 * t.a / 4) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_arithmetic_sides_is_right_angled_and_inradius_equals_diff_l2985_298541


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l2985_298538

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l2985_298538


namespace NUMINAMATH_CALUDE_boys_tried_out_l2985_298566

/-- The number of boys who tried out for the basketball team -/
def num_boys : ℕ := sorry

/-- The number of girls who tried out for the basketball team -/
def num_girls : ℕ := 39

/-- The number of students who got called back -/
def called_back : ℕ := 26

/-- The number of students who didn't make the cut -/
def didnt_make_cut : ℕ := 17

theorem boys_tried_out : num_boys = 4 := by
  sorry

end NUMINAMATH_CALUDE_boys_tried_out_l2985_298566


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_below_75_l2985_298560

theorem largest_multiple_of_9_below_75 : ∃ n : ℕ, n * 9 = 72 ∧ 72 < 75 ∧ ∀ m : ℕ, m * 9 < 75 → m * 9 ≤ 72 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_below_75_l2985_298560


namespace NUMINAMATH_CALUDE_roses_age_l2985_298572

theorem roses_age (rose_age mother_age : ℕ) : 
  rose_age = mother_age / 3 →
  rose_age + mother_age = 100 →
  rose_age = 25 := by
sorry

end NUMINAMATH_CALUDE_roses_age_l2985_298572


namespace NUMINAMATH_CALUDE_intersection_count_l2985_298510

/-- The number of distinct intersection points between a circle and a parabola -/
def numIntersectionPoints (r : ℝ) (a b : ℝ) : ℕ :=
  let circle (x y : ℝ) := x^2 + y^2 = r^2
  let parabola (x y : ℝ) := y = a * x^2 + b
  -- Definition of the function to count intersection points
  sorry

/-- Theorem stating that the number of intersection points is 3 for the given equations -/
theorem intersection_count : numIntersectionPoints 4 1 (-4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l2985_298510


namespace NUMINAMATH_CALUDE_sum_equals_140_l2985_298500

theorem sum_equals_140 (p q r s : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (h1 : p^2 + q^2 = 2500)
  (h2 : r^2 + s^2 = 2500)
  (h3 : p * r = 1200)
  (h4 : q * s = 1200) :
  p + q + r + s = 140 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_140_l2985_298500


namespace NUMINAMATH_CALUDE_gold_cube_profit_l2985_298554

-- Define the cube's side length in cm
def cube_side : ℝ := 6

-- Define the density of gold in g/cm³
def gold_density : ℝ := 19

-- Define the buying price per gram in dollars
def buying_price : ℝ := 60

-- Define the selling price multiplier
def selling_multiplier : ℝ := 1.5

-- Theorem statement
theorem gold_cube_profit :
  let volume : ℝ := cube_side ^ 3
  let mass : ℝ := gold_density * volume
  let cost : ℝ := mass * buying_price
  let selling_price : ℝ := cost * selling_multiplier
  selling_price - cost = 123120 := by
  sorry

end NUMINAMATH_CALUDE_gold_cube_profit_l2985_298554


namespace NUMINAMATH_CALUDE_range_of_a_for_max_and_min_l2985_298539

/-- The cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- The discriminant of the quadratic equation f'(x) = 0 -/
def discriminant (a : ℝ) : ℝ := 36*a^2 - 36*(a+2)

/-- The condition for f to have both maximum and minimum -/
def has_max_and_min (a : ℝ) : Prop := discriminant a > 0

theorem range_of_a_for_max_and_min :
  ∀ a : ℝ, has_max_and_min a ↔ (a < -1 ∨ a > 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_max_and_min_l2985_298539


namespace NUMINAMATH_CALUDE_colors_wash_time_l2985_298520

/-- Represents the time in minutes for a laundry load in the washing machine and dryer -/
structure LaundryTime where
  wash : ℕ
  dry : ℕ

/-- The total time for all three loads of laundry -/
def totalTime : ℕ := 344

/-- The laundry time for the whites -/
def whites : LaundryTime := { wash := 72, dry := 50 }

/-- The laundry time for the darks -/
def darks : LaundryTime := { wash := 58, dry := 65 }

/-- The drying time for the colors -/
def colorsDryTime : ℕ := 54

/-- Theorem stating that the washing time for the colors is 45 minutes -/
theorem colors_wash_time :
  ∃ (colorsWashTime : ℕ),
    colorsWashTime = totalTime - (whites.wash + whites.dry + darks.wash + darks.dry + colorsDryTime) ∧
    colorsWashTime = 45 := by
  sorry

end NUMINAMATH_CALUDE_colors_wash_time_l2985_298520


namespace NUMINAMATH_CALUDE_largest_odd_between_1_and_7_l2985_298516

theorem largest_odd_between_1_and_7 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 7 ∧ Odd n → n ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_odd_between_1_and_7_l2985_298516


namespace NUMINAMATH_CALUDE_uncovered_area_of_box_l2985_298552

/-- Given a rectangular box with dimensions 4 inches by 6 inches and a square block with side length 4 inches placed inside, the uncovered area of the box is 8 square inches. -/
theorem uncovered_area_of_box (box_length : ℕ) (box_width : ℕ) (block_side : ℕ) : 
  box_length = 4 → box_width = 6 → block_side = 4 → 
  (box_length * box_width) - (block_side * block_side) = 8 := by
sorry

end NUMINAMATH_CALUDE_uncovered_area_of_box_l2985_298552


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2985_298515

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = 13 - 8*x + Real.sqrt 2 * x^2) →
  (deriv f x₀ = 4) →
  x₀ = 3 * Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) :
  (∀ x, f x = x^2 + 2*x*(deriv f 0)) →
  ¬∃ y, deriv f 0 = y := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2985_298515


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l2985_298551

/-- The area of a right triangle with legs of length 3 and 5 is 7.5 -/
theorem right_triangle_area : Real → Prop :=
  fun a => 
    ∃ (b h : Real),
      b = 3 ∧
      h = 5 ∧
      a = (1 / 2) * b * h ∧
      a = 7.5

/-- Proof of the theorem -/
theorem right_triangle_area_proof : right_triangle_area 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l2985_298551


namespace NUMINAMATH_CALUDE_cubic_roots_sum_series_l2985_298519

def cubic_polynomial (x : ℝ) : ℝ := 30 * x^3 - 50 * x^2 + 22 * x - 1

theorem cubic_roots_sum_series : 
  ∃ (a b c : ℝ),
    (∀ x : ℝ, cubic_polynomial x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
    (∑' n : ℕ, (a^n + b^n + c^n)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_series_l2985_298519


namespace NUMINAMATH_CALUDE_polynomial_root_equivalence_l2985_298568

theorem polynomial_root_equivalence : ∀ r : ℝ, 
  r^2 - 2*r - 1 = 0 → r^5 - 29*r - 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_equivalence_l2985_298568


namespace NUMINAMATH_CALUDE_least_possible_value_l2985_298570

theorem least_possible_value (x y z : ℤ) : 
  Even x → Odd y → Odd z → x < y → y < z → y - x > 5 → (∀ w, w - x ≥ 9 → w ≥ z) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_value_l2985_298570


namespace NUMINAMATH_CALUDE_no_real_roots_composition_l2985_298528

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem no_real_roots_composition (a b c : ℝ) :
  (∀ x : ℝ, QuadraticPolynomial a b c x ≠ x) →
  (∀ x : ℝ, QuadraticPolynomial a b c (QuadraticPolynomial a b c x) ≠ x) :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_composition_l2985_298528


namespace NUMINAMATH_CALUDE_digit_distribution_exists_l2985_298584

theorem digit_distribution_exists : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 0 ∧ 
  n % 5 = 0 ∧ 
  n % 10 = 0 ∧ 
  n / 2 + 2 * (n / 5) + n / 10 = n :=
sorry

end NUMINAMATH_CALUDE_digit_distribution_exists_l2985_298584


namespace NUMINAMATH_CALUDE_f_properties_l2985_298586

def f_property (f : ℝ → ℝ) : Prop :=
  (∃ x, f x ≠ 0) ∧
  (∀ x y, f (x * y) = x * f y + y * f x) ∧
  (∀ x, x > 1 → f x < 0)

theorem f_properties (f : ℝ → ℝ) (h : f_property f) :
  (f 1 = 0 ∧ f (-1) = 0) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x₁ x₂, x₁ > x₂ ∧ x₂ > 1 → f x₁ < f x₂) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2985_298586


namespace NUMINAMATH_CALUDE_quadratic_coefficient_theorem_l2985_298589

theorem quadratic_coefficient_theorem (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 2 ∨ x = -8) → 
  b = 6 ∧ c = -16 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_theorem_l2985_298589


namespace NUMINAMATH_CALUDE_no_maximum_b_plus_c_l2985_298529

/-- A cubic function f(x) = x^3 + bx^2 + cx + d -/
def cubic_function (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- The derivative of the cubic function -/
def cubic_derivative (b c : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

theorem no_maximum_b_plus_c :
  ∀ b c d : ℝ,
  (∀ x ∈ Set.Icc (-1) 2, cubic_derivative b c x ≤ 0) →
  ¬∃ M : ℝ, ∀ b' c' : ℝ, 
    (∀ x ∈ Set.Icc (-1) 2, cubic_derivative b' c' x ≤ 0) →
    b' + c' ≤ M :=
by sorry

end NUMINAMATH_CALUDE_no_maximum_b_plus_c_l2985_298529


namespace NUMINAMATH_CALUDE_g_of_13_l2985_298523

def g (x : ℝ) : ℝ := x^2 + 2*x + 25

theorem g_of_13 : g 13 = 220 := by
  sorry

end NUMINAMATH_CALUDE_g_of_13_l2985_298523


namespace NUMINAMATH_CALUDE_allans_balloons_l2985_298580

theorem allans_balloons (total : ℕ) (jakes_balloons : ℕ) (h1 : total = 3) (h2 : jakes_balloons = 1) :
  total - jakes_balloons = 2 :=
by sorry

end NUMINAMATH_CALUDE_allans_balloons_l2985_298580


namespace NUMINAMATH_CALUDE_clown_balloons_l2985_298553

theorem clown_balloons (initial_balloons : ℕ) (additional_balloons : ℕ) 
  (h1 : initial_balloons = 47) 
  (h2 : additional_balloons = 13) : 
  initial_balloons + additional_balloons = 60 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l2985_298553


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_equation_negation_l2985_298583

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem quadratic_equation_negation : 
  (¬∃ x : ℝ, x^2 + 2*x + 3 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_equation_negation_l2985_298583


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l2985_298547

theorem tax_rate_calculation (total_value tax_free_allowance tax_paid : ℝ) : 
  total_value = 1720 →
  tax_free_allowance = 600 →
  tax_paid = 78.4 →
  (tax_paid / (total_value - tax_free_allowance)) * 100 = 7 := by
sorry

end NUMINAMATH_CALUDE_tax_rate_calculation_l2985_298547


namespace NUMINAMATH_CALUDE_ellipse_foci_l2985_298507

theorem ellipse_foci (x y : ℝ) :
  (x^2 / 25 + y^2 / 169 = 1) →
  (∃ f₁ f₂ : ℝ × ℝ, 
    (f₁ = (0, 12) ∧ f₂ = (0, -12)) ∧
    (∀ p : ℝ × ℝ, p.1^2 / 25 + p.2^2 / 169 = 1 →
      (Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
       Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) =
       2 * Real.sqrt 169))) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_l2985_298507


namespace NUMINAMATH_CALUDE_no_perfect_squares_l2985_298594

theorem no_perfect_squares (a b : ℕ) : ¬(∃k m : ℕ, (a^2 + 2*b^2 = k^2) ∧ (b^2 + 2*a = m^2)) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l2985_298594


namespace NUMINAMATH_CALUDE_cylinder_min_surface_area_l2985_298565

/-- For a cylindrical tank with volume V, the surface area (without a lid) is minimized when the radius and height are both equal to ∛(V/π) -/
theorem cylinder_min_surface_area (V : ℝ) (h : V > 0) :
  let surface_area (r h : ℝ) := π * r^2 + 2 * π * r * h
  let volume (r h : ℝ) := π * r^2 * h
  ∃ (r : ℝ), r > 0 ∧ 
    (∀ (r' h' : ℝ), r' > 0 → h' > 0 → volume r' h' = V → 
      surface_area r' h' ≥ surface_area r r) ∧
    r = (V / π)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_min_surface_area_l2985_298565


namespace NUMINAMATH_CALUDE_existsNonIsoscelesWithFourEqualAreas_l2985_298595

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Function to check if a point is inside a triangle
def isInside (P : Point) (t : Triangle) : Prop := sorry

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop := sorry

-- Function to create smaller triangles by connecting P to vertices and drawing perpendiculars
def createSmallerTriangles (P : Point) (t : Triangle) : List Triangle := sorry

-- Function to check if 4 out of 6 triangles have equal areas
def fourEqualAreas (triangles : List Triangle) : Prop := sorry

-- The main theorem
theorem existsNonIsoscelesWithFourEqualAreas : 
  ∃ (t : Triangle) (P : Point), 
    isInside P t ∧ 
    ¬isIsosceles t ∧ 
    fourEqualAreas (createSmallerTriangles P t) := sorry

end NUMINAMATH_CALUDE_existsNonIsoscelesWithFourEqualAreas_l2985_298595


namespace NUMINAMATH_CALUDE_system_solutions_l2985_298571

theorem system_solutions :
  let eq1 := (fun (x y : ℝ) => x + 3*y + 3*x*y = -1)
  let eq2 := (fun (x y : ℝ) => x^2*y + 3*x*y^2 = -4)
  ∃! (s : Set (ℝ × ℝ)), s = {(-3, -1/3), (-1, -1), (-1, 4/3), (4, -1/3)} ∧
    ∀ (p : ℝ × ℝ), p ∈ s ↔ (eq1 p.1 p.2 ∧ eq2 p.1 p.2) := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l2985_298571


namespace NUMINAMATH_CALUDE_max_regions_correct_max_regions_recurrence_max_regions_is_maximum_l2985_298555

/-- The maximum number of regions a plane can be divided into by n rectangles with parallel sides -/
def max_regions (n : ℕ) : ℕ := 2*n^2 - 2*n + 2

/-- Theorem stating that max_regions gives the correct number of regions for n rectangles -/
theorem max_regions_correct (n : ℕ) : 
  max_regions n = 2*n^2 - 2*n + 2 := by sorry

/-- Theorem stating that max_regions satisfies the recurrence relation -/
theorem max_regions_recurrence (n : ℕ) : 
  max_regions (n + 1) = max_regions n + 4*n := by sorry

/-- Theorem stating that max_regions gives the maximum possible number of regions -/
theorem max_regions_is_maximum (n : ℕ) (k : ℕ) :
  k ≤ max_regions n := by sorry

end NUMINAMATH_CALUDE_max_regions_correct_max_regions_recurrence_max_regions_is_maximum_l2985_298555


namespace NUMINAMATH_CALUDE_range_of_a_l2985_298591

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Define the sufficient condition
def sufficient_condition (a : ℝ) : Prop := ∀ x, x ∈ A → x ∈ B a

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, sufficient_condition a ↔ a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2985_298591


namespace NUMINAMATH_CALUDE_min_value_expression_l2985_298508

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > -1) (hsum : x + y = 1) :
  (x^2 + 3) / x + y^2 / (y + 1) ≥ 2 + Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > -1 ∧ x₀ + y₀ = 1 ∧
    (x₀^2 + 3) / x₀ + y₀^2 / (y₀ + 1) = 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2985_298508


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l2985_298506

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 10) → cows = 5 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l2985_298506


namespace NUMINAMATH_CALUDE_provisions_after_reinforcement_provisions_last_20_days_l2985_298535

theorem provisions_after_reinforcement 
  (initial_garrison : ℕ) 
  (initial_provisions : ℕ) 
  (days_before_reinforcement : ℕ) 
  (reinforcement : ℕ) : ℕ :=
  let remaining_provisions := initial_garrison * (initial_provisions - days_before_reinforcement)
  let total_men := initial_garrison + reinforcement
  remaining_provisions / total_men

theorem provisions_last_20_days 
  (initial_garrison : ℕ) 
  (initial_provisions : ℕ) 
  (days_before_reinforcement : ℕ) 
  (reinforcement : ℕ) :
  initial_garrison = 1000 →
  initial_provisions = 60 →
  days_before_reinforcement = 15 →
  reinforcement = 1250 →
  provisions_after_reinforcement initial_garrison initial_provisions days_before_reinforcement reinforcement = 20 :=
by sorry

end NUMINAMATH_CALUDE_provisions_after_reinforcement_provisions_last_20_days_l2985_298535


namespace NUMINAMATH_CALUDE_bill_donut_order_combinations_l2985_298577

/-- The number of combinations for selecting donuts satisfying the given conditions -/
def donut_combinations (total_donuts : ℕ) (donut_types : ℕ) (types_to_select : ℕ) : ℕ :=
  (donut_types.choose types_to_select) * 
  ((total_donuts - types_to_select + types_to_select - 1).choose (types_to_select - 1))

/-- Theorem stating that the number of combinations for Bill's donut order is 100 -/
theorem bill_donut_order_combinations : 
  donut_combinations 7 5 4 = 100 := by
sorry

end NUMINAMATH_CALUDE_bill_donut_order_combinations_l2985_298577


namespace NUMINAMATH_CALUDE_torn_pages_fine_l2985_298544

/-- Calculates the fine for tearing out pages from a book -/
def calculate_fine (start_page end_page : ℕ) (cost_per_sheet : ℕ) : ℕ :=
  let total_pages := end_page - start_page + 1
  let total_sheets := (total_pages + 1) / 2
  total_sheets * cost_per_sheet

/-- The fine for tearing out pages 15 to 30 is 128 yuan -/
theorem torn_pages_fine :
  calculate_fine 15 30 16 = 128 := by
  sorry

end NUMINAMATH_CALUDE_torn_pages_fine_l2985_298544


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2985_298588

theorem yellow_balls_count (red white : ℕ) (a : ℝ) :
  red = 2 →
  white = 4 →
  (a / (red + white + a) = 1 / 4) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2985_298588


namespace NUMINAMATH_CALUDE_womans_age_multiple_l2985_298558

theorem womans_age_multiple (W S k : ℕ) : 
  W = k * S + 3 →
  W + S = 84 →
  S = 27 →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_womans_age_multiple_l2985_298558


namespace NUMINAMATH_CALUDE_total_cost_is_2495_l2985_298531

/-- Represents the quantity of each fruit in kilograms -/
def apple_qty : ℕ := 8
def mango_qty : ℕ := 9
def banana_qty : ℕ := 6
def grape_qty : ℕ := 4
def cherry_qty : ℕ := 3

/-- Represents the rate of each fruit per kilogram -/
def apple_rate : ℕ := 70
def mango_rate : ℕ := 75
def banana_rate : ℕ := 40
def grape_rate : ℕ := 120
def cherry_rate : ℕ := 180

/-- Calculates the total cost of all fruits -/
def total_cost : ℕ := 
  apple_qty * apple_rate + 
  mango_qty * mango_rate + 
  banana_qty * banana_rate + 
  grape_qty * grape_rate + 
  cherry_qty * cherry_rate

/-- Theorem stating that the total cost of all fruits is 2495 -/
theorem total_cost_is_2495 : total_cost = 2495 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_2495_l2985_298531


namespace NUMINAMATH_CALUDE_smallest_number_properties_l2985_298575

/-- The number of divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The smallest natural number divisible by 35 with exactly 75 divisors -/
def smallestNumber : ℕ := 490000

theorem smallest_number_properties :
  (35 ∣ smallestNumber) ∧
  (numDivisors smallestNumber = 75) ∧
  ∀ n : ℕ, n < smallestNumber → ¬((35 ∣ n) ∧ (numDivisors n = 75)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_properties_l2985_298575


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l2985_298537

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that ax^2 + bx + c = (px + q)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_k (k : ℝ) :
  is_perfect_square_trinomial 1 (-k) 9 → k = 6 ∨ k = -6 :=
by
  sorry

#check perfect_square_trinomial_k

end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l2985_298537


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l2985_298559

theorem walking_rate_ratio (usual_time faster_time : ℝ) (h1 : usual_time = 28) 
  (h2 : faster_time = usual_time - 4) : 
  (usual_time / faster_time) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l2985_298559


namespace NUMINAMATH_CALUDE_first_train_length_l2985_298521

/-- Given two trains with specific speeds, lengths, and crossing time, prove the length of the first train. -/
theorem first_train_length
  (v1 : ℝ) -- Speed of first train
  (v2 : ℝ) -- Speed of second train
  (l2 : ℝ) -- Length of second train
  (d : ℝ)  -- Distance between trains
  (t : ℝ)  -- Time for second train to cross first train
  (h1 : v1 = 10)
  (h2 : v2 = 15)
  (h3 : l2 = 150)
  (h4 : d = 50)
  (h5 : t = 60) :
  ∃ l1 : ℝ, l1 = 100 ∧ l1 + l2 + d = (v2 - v1) * t := by
  sorry

#check first_train_length

end NUMINAMATH_CALUDE_first_train_length_l2985_298521


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_eleven_l2985_298545

theorem four_digit_divisible_by_eleven : 
  ∃ (B : ℕ), B < 10 ∧ (4000 + 100 * B + 10 * B + 2) % 11 = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_eleven_l2985_298545


namespace NUMINAMATH_CALUDE_m_range_l2985_298546

def p (m : ℝ) : Prop := ∀ x : ℝ, 2^x - m + 1 > 0

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (5 - 2*m)^x < (5 - 2*m)^y

theorem m_range (m : ℝ) (h : p m ∧ q m) : m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2985_298546


namespace NUMINAMATH_CALUDE_union_eq_univ_complement_inter_B_a_range_l2985_298503

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 9*x + 18 ≥ 0}
def B : Set ℝ := {x | -2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorems to be proved
theorem union_eq_univ : A ∪ B = Set.univ := by sorry

theorem complement_inter_B : (Aᶜ) ∩ B = {x : ℝ | 3 < x ∧ x < 6} := by sorry

theorem a_range (a : ℝ) : C a ⊆ B → -2 ≤ a ∧ a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_union_eq_univ_complement_inter_B_a_range_l2985_298503


namespace NUMINAMATH_CALUDE_cube_difference_l2985_298564

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l2985_298564


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_l2985_298550

def sum_odd (n : ℕ) : ℕ := n^2

def sum_even (n : ℕ) : ℕ := n * (n + 1)

def odd_terms (max : ℕ) : ℕ := (max - 1) / 2 + 1

def even_terms (max : ℕ) : ℕ := (max - 2) / 2 + 1

theorem odd_even_sum_difference :
  sum_odd (odd_terms 2023) - sum_even (even_terms 2020) = 3034 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_l2985_298550


namespace NUMINAMATH_CALUDE_smallest_integer_square_triple_plus_75_l2985_298530

theorem smallest_integer_square_triple_plus_75 :
  ∃ x : ℤ, (∀ y : ℤ, y^2 = 3*y + 75 → x ≤ y) ∧ x^2 = 3*x + 75 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_square_triple_plus_75_l2985_298530


namespace NUMINAMATH_CALUDE_polygon_triangulation_l2985_298579

/-- Given an n-sided polygon divided into triangles by non-intersecting diagonals,
    this theorem states that the number of triangles with exactly two sides
    as edges of the original polygon is at least 2. -/
theorem polygon_triangulation (n : ℕ) (h : n ≥ 3) :
  ∃ (k₀ k₁ k₂ : ℕ),
    k₀ + k₁ + k₂ = n - 2 ∧
    k₁ + 2 * k₂ = n ∧
    k₂ ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_polygon_triangulation_l2985_298579


namespace NUMINAMATH_CALUDE_square_eq_four_implies_x_values_l2985_298534

theorem square_eq_four_implies_x_values (x : ℝ) :
  (x - 1)^2 = 4 → x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_eq_four_implies_x_values_l2985_298534


namespace NUMINAMATH_CALUDE_gym_guests_ratio_l2985_298592

/-- Represents the number of guests entering the gym each hour -/
structure GymGuests where
  first_hour : ℕ
  second_hour : ℕ
  third_hour : ℕ
  fourth_hour : ℕ

/-- Calculates the total number of guests -/
def total_guests (g : GymGuests) : ℕ :=
  g.first_hour + g.second_hour + g.third_hour + g.fourth_hour

theorem gym_guests_ratio (total_towels : ℕ) (g : GymGuests) : 
  total_towels = 300 →
  g.first_hour = 50 →
  g.second_hour = (120 * g.first_hour) / 100 →
  g.third_hour = (125 * g.second_hour) / 100 →
  g.fourth_hour > g.third_hour →
  total_guests g = 285 →
  (g.fourth_hour - g.third_hour) * 3 = g.third_hour := by
  sorry

#check gym_guests_ratio

end NUMINAMATH_CALUDE_gym_guests_ratio_l2985_298592


namespace NUMINAMATH_CALUDE_mike_pears_l2985_298596

/-- The number of pears Keith picked initially -/
def keith_initial : ℕ := 47

/-- The number of pears Keith gave away -/
def keith_gave_away : ℕ := 46

/-- The total number of pears Keith and Mike have after Keith gave away pears -/
def total_remaining : ℕ := 13

/-- The number of pears Mike picked -/
def mike_picked : ℕ := total_remaining - (keith_initial - keith_gave_away)

theorem mike_pears : mike_picked = 12 := by sorry

end NUMINAMATH_CALUDE_mike_pears_l2985_298596


namespace NUMINAMATH_CALUDE_circle_center_transformation_l2985_298513

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

theorem circle_center_transformation :
  let S : ℝ × ℝ := (-2, 6)
  let reflected := reflect_x S
  let final := translate_up reflected 10
  final = (-2, 4) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l2985_298513


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l2985_298578

-- Define the function f(x) = -x^2
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l2985_298578


namespace NUMINAMATH_CALUDE_percentage_calculation_l2985_298543

theorem percentage_calculation : (1 / 8 / 100 * 160) + 0.5 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2985_298543


namespace NUMINAMATH_CALUDE_locus_C_equation_point_N_coordinates_l2985_298599

-- Define the circle and its properties
def Circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ p : ℝ × ℝ, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Define the locus C
def LocusC (p : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), Circle p r ∧ 
  (p.1 - 1)^2 + p.2^2 = r^2 ∧  -- Tangent to F(1,0)
  (p.1 + 1)^2 = r^2            -- Tangent to x = -1

-- Define point A
def PointA : ℝ × ℝ := (4, 4)

-- Define point B
def PointB : ℝ × ℝ := (0, 4)

-- Define point M
def PointM : ℝ × ℝ := (0, 2)

-- Define point F
def PointF : ℝ × ℝ := (1, 0)

-- Theorem for the equation of locus C
theorem locus_C_equation : 
  ∀ p : ℝ × ℝ, LocusC p ↔ p.2^2 = 4 * p.1 := by sorry

-- Theorem for the coordinates of point N
theorem point_N_coordinates :
  ∃ N : ℝ × ℝ, N.1 = 8/5 ∧ N.2 = 4/5 ∧
  (N.2 - PointM.2) / (N.1 - PointM.1) = -3/4 ∧  -- MN perpendicular to FA
  (PointA.2 - PointF.2) / (PointA.1 - PointF.1) = 4/3 := by sorry

end NUMINAMATH_CALUDE_locus_C_equation_point_N_coordinates_l2985_298599


namespace NUMINAMATH_CALUDE_total_diagonals_two_polygons_l2985_298536

/-- Number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The first polygon has 100 sides -/
def polygon1_sides : ℕ := 100

/-- The second polygon has 150 sides -/
def polygon2_sides : ℕ := 150

/-- Theorem: The total number of diagonals in a 100-sided polygon and a 150-sided polygon is 15875 -/
theorem total_diagonals_two_polygons : 
  diagonals polygon1_sides + diagonals polygon2_sides = 15875 := by
  sorry

end NUMINAMATH_CALUDE_total_diagonals_two_polygons_l2985_298536


namespace NUMINAMATH_CALUDE_second_meeting_at_5_4_minutes_l2985_298556

/-- Represents the race scenario between George and Henry --/
structure RaceScenario where
  pool_length : ℝ
  george_start_time : ℝ
  henry_start_time : ℝ
  first_meeting_time : ℝ
  first_meeting_distance : ℝ

/-- Calculates the time of the second meeting given a race scenario --/
def second_meeting_time (scenario : RaceScenario) : ℝ :=
  sorry

/-- The main theorem stating that the second meeting occurs 5.4 minutes after George's start --/
theorem second_meeting_at_5_4_minutes (scenario : RaceScenario) 
  (h1 : scenario.pool_length = 50)
  (h2 : scenario.george_start_time = 0)
  (h3 : scenario.henry_start_time = 1)
  (h4 : scenario.first_meeting_time = 3)
  (h5 : scenario.first_meeting_distance = 25) : 
  second_meeting_time scenario = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_second_meeting_at_5_4_minutes_l2985_298556
