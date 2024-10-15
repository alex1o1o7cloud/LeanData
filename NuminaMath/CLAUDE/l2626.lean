import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_height_l2626_262638

theorem parallelogram_height 
  (area : ℝ) 
  (base : ℝ) 
  (h1 : area = 308) 
  (h2 : base = 22) : 
  area / base = 14 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2626_262638


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2626_262678

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 1) * x^2 + 6 * x + 3 = 0 ∧ 
   (k - 1) * y^2 + 6 * y + 3 = 0) ↔ 
  (k < 4 ∧ k ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2626_262678


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_above_10_divisible_by_5_l2626_262675

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns true if four consecutive natural numbers are all prime, false otherwise -/
def fourConsecutivePrimes (a b c d : ℕ) : Prop := 
  isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1

theorem smallest_sum_of_four_consecutive_primes_above_10_divisible_by_5 :
  ∃ (a b c d : ℕ),
    fourConsecutivePrimes a b c d ∧
    a > 10 ∧
    (a + b + c + d) % 5 = 0 ∧
    (a + b + c + d = 60) ∧
    ∀ (w x y z : ℕ),
      fourConsecutivePrimes w x y z →
      w > 10 →
      (w + x + y + z) % 5 = 0 →
      (w + x + y + z) ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_above_10_divisible_by_5_l2626_262675


namespace NUMINAMATH_CALUDE_min_base_sum_l2626_262657

theorem min_base_sum : 
  ∃ (a b : ℕ+), 
    (3 * a.val + 5 = 4 * b.val + 2) ∧ 
    (∀ (c d : ℕ+), (3 * c.val + 5 = 4 * d.val + 2) → (a.val + b.val ≤ c.val + d.val)) ∧
    (a.val + b.val = 13) := by
  sorry

end NUMINAMATH_CALUDE_min_base_sum_l2626_262657


namespace NUMINAMATH_CALUDE_vector_properties_l2626_262607

/-- Given vectors a and b, prove the sine of their angle and the value of m for perpendicularity. -/
theorem vector_properties (a b : ℝ × ℝ) (h1 : a = (3, -4)) (h2 : b = (1, 2)) :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  ∃ (m : ℝ),
    Real.sin θ = (2 * Real.sqrt 5) / 5 ∧
    (m * a.1 - b.1) * (a.1 + b.1) + (m * a.2 - b.2) * (a.2 + b.2) = 0 ∧
    m = 0 :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l2626_262607


namespace NUMINAMATH_CALUDE_matrix_product_proof_l2626_262637

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -3; 1, 3, -1; 0, 5, 2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -1, 4; -2, 0, 0; 3, 0, -2]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![-7, -2, 14; -8, -1, 6; -4, 0, -4]

theorem matrix_product_proof : A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_product_proof_l2626_262637


namespace NUMINAMATH_CALUDE_walnut_trees_remaining_l2626_262683

/-- The number of walnut trees remaining after removal -/
def remaining_trees (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

theorem walnut_trees_remaining : remaining_trees 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_remaining_l2626_262683


namespace NUMINAMATH_CALUDE_ab_value_l2626_262639

theorem ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^2 + b^2 = 3) (h2 : a^4 + b^4 = 15/4) : a * b = Real.sqrt 42 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2626_262639


namespace NUMINAMATH_CALUDE_travel_distance_proof_l2626_262667

def speed_limit : ℝ := 60
def speed_above_limit : ℝ := 15
def travel_time : ℝ := 2

theorem travel_distance_proof :
  let actual_speed := speed_limit + speed_above_limit
  actual_speed * travel_time = 150 := by sorry

end NUMINAMATH_CALUDE_travel_distance_proof_l2626_262667


namespace NUMINAMATH_CALUDE_square_area_given_circle_l2626_262604

-- Define the area of the circle
def circle_area : ℝ := 39424

-- Define the relationship between square perimeter and circle radius
def square_perimeter_equals_circle_radius (square_side : ℝ) (circle_radius : ℝ) : Prop :=
  4 * square_side = circle_radius

-- Theorem statement
theorem square_area_given_circle (square_side : ℝ) (circle_radius : ℝ) :
  circle_area = Real.pi * circle_radius^2 →
  square_perimeter_equals_circle_radius square_side circle_radius →
  square_side^2 = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_given_circle_l2626_262604


namespace NUMINAMATH_CALUDE_total_cost_for_index_finger_rings_l2626_262622

def cost_per_ring : ℕ := 24
def index_fingers_per_person : ℕ := 2

theorem total_cost_for_index_finger_rings :
  cost_per_ring * index_fingers_per_person = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_for_index_finger_rings_l2626_262622


namespace NUMINAMATH_CALUDE_cube_of_complex_root_of_unity_l2626_262672

theorem cube_of_complex_root_of_unity (z : ℂ) : 
  z = Complex.cos (2 * Real.pi / 3) - Complex.I * Complex.sin (Real.pi / 3) → 
  z^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_complex_root_of_unity_l2626_262672


namespace NUMINAMATH_CALUDE_square_of_105_l2626_262632

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end NUMINAMATH_CALUDE_square_of_105_l2626_262632


namespace NUMINAMATH_CALUDE_greatest_b_proof_l2626_262635

/-- The greatest integer b for which x^2 + bx + 17 ≠ 0 for all real x -/
def greatest_b : ℤ := 8

theorem greatest_b_proof :
  (∀ x : ℝ, x^2 + (greatest_b : ℝ) * x + 17 ≠ 0) ∧
  (∀ b : ℤ, b > greatest_b → ∃ x : ℝ, x^2 + (b : ℝ) * x + 17 = 0) :=
by sorry

#check greatest_b_proof

end NUMINAMATH_CALUDE_greatest_b_proof_l2626_262635


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2626_262624

theorem repeating_decimal_sum (x : ℚ) : x = 23 / 99 → (x.num + x.den = 122) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2626_262624


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2626_262679

theorem complex_fraction_equality : ∀ (z₁ z₂ : ℂ), 
  z₁ = -1 + 3*I ∧ z₂ = 1 + I → (z₁ + z₂) / (z₁ - z₂) = 1 - I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2626_262679


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l2626_262669

/-- The sum of an arithmetic sequence with n terms, starting from a, with a common difference of d -/
def arithmetic_sum (n : ℕ) (a : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The number of days Murtha collects pebbles -/
def days : ℕ := 15

/-- The number of pebbles Murtha collects on the first day -/
def initial_pebbles : ℕ := 2

/-- The daily increase in pebble collection -/
def daily_increase : ℕ := 1

theorem murtha_pebble_collection :
  arithmetic_sum days initial_pebbles daily_increase = 135 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebble_collection_l2626_262669


namespace NUMINAMATH_CALUDE_missing_files_l2626_262644

/-- Proves that the number of missing files is 15 --/
theorem missing_files (total_files : ℕ) (afternoon_files : ℕ) : 
  total_files = 60 → 
  afternoon_files = 15 → 
  total_files - (total_files / 2 + afternoon_files) = 15 := by
  sorry

end NUMINAMATH_CALUDE_missing_files_l2626_262644


namespace NUMINAMATH_CALUDE_fraction_power_product_specific_fraction_product_l2626_262653

theorem fraction_power_product (a b c d : ℚ) (j : ℕ) :
  (a / b) ^ j * (c / d) ^ j = ((a * c) / (b * d)) ^ j :=
sorry

theorem specific_fraction_product :
  (3 / 4 : ℚ) ^ 3 * (2 / 5 : ℚ) ^ 3 = 27 / 1000 :=
sorry

end NUMINAMATH_CALUDE_fraction_power_product_specific_fraction_product_l2626_262653


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2626_262655

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ 2 ∧ x ≠ -2 →
  (1 - 1 / (x - 1)) / ((x^2 - 4) / (x - 1)) = 1 / (x + 2) ∧
  (1 - 1 / (-1 - 1)) / (((-1)^2 - 4) / (-1 - 1)) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2626_262655


namespace NUMINAMATH_CALUDE_pentagon_angle_C_l2626_262616

/-- Represents the angles of a pentagon in degrees -/
structure PentagonAngles where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- Defines the properties of the pentagon's angles -/
def is_valid_pentagon (p : PentagonAngles) : Prop :=
  p.A > 0 ∧ p.B > 0 ∧ p.C > 0 ∧ p.D > 0 ∧ p.E > 0 ∧
  p.A < p.B ∧ p.B < p.C ∧ p.C < p.D ∧ p.D < p.E ∧
  p.A + p.B + p.C + p.D + p.E = 540 ∧
  ∃ d : ℝ, d > 0 ∧ 
    p.B - p.A = d ∧
    p.C - p.B = d ∧
    p.D - p.C = d ∧
    p.E - p.D = d

theorem pentagon_angle_C (p : PentagonAngles) 
  (h : is_valid_pentagon p) : p.C = 108 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_C_l2626_262616


namespace NUMINAMATH_CALUDE_solution_set_f_gt_g_range_of_a_l2626_262659

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := |2*x - 2|

-- Theorem for the solution set of f(x) > g(x)
theorem solution_set_f_gt_g :
  {x : ℝ | f x > g x} = {x : ℝ | 2/3 < x ∧ x < 2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, 2 * f x + g x > a * x + 1} = {a : ℝ | -4 ≤ a ∧ a < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_g_range_of_a_l2626_262659


namespace NUMINAMATH_CALUDE_no_self_referential_function_l2626_262619

theorem no_self_referential_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), n > 1 → f n = f (f (n - 1)) + f (f (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_self_referential_function_l2626_262619


namespace NUMINAMATH_CALUDE_distance_between_opposite_faces_of_unit_octahedron_l2626_262688

/-- A regular octahedron is a polyhedron with 8 faces, where each face is an equilateral triangle -/
structure RegularOctahedron where
  side_length : ℝ

/-- The distance between two opposite faces of a regular octahedron -/
def distance_between_opposite_faces (o : RegularOctahedron) : ℝ :=
  sorry

/-- Theorem: In a regular octahedron with side length 1, the distance between two opposite faces is √6/3 -/
theorem distance_between_opposite_faces_of_unit_octahedron :
  let o : RegularOctahedron := ⟨1⟩
  distance_between_opposite_faces o = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_opposite_faces_of_unit_octahedron_l2626_262688


namespace NUMINAMATH_CALUDE_solve_turtle_problem_l2626_262620

def turtle_problem (kristen_turtles : ℕ) (kris_ratio : ℚ) (trey_multiplier : ℕ) : Prop :=
  let kris_turtles : ℚ := kris_ratio * kristen_turtles
  let trey_turtles : ℚ := trey_multiplier * kris_turtles
  (trey_turtles - kristen_turtles : ℚ) = 9

theorem solve_turtle_problem :
  turtle_problem 12 (1/4) 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_turtle_problem_l2626_262620


namespace NUMINAMATH_CALUDE_coloring_satisfies_conditions_l2626_262602

-- Define the color type
inductive Color
| White
| Red
| Black

-- Define a lattice point
structure LatticePoint where
  x : Int
  y : Int

-- Define the coloring function
def color (p : LatticePoint) : Color :=
  match p.x, p.y with
  | x, y => if x % 2 = 0 then Color.Red
            else if y % 2 = 0 then Color.Black
            else Color.White

-- Define a line parallel to x-axis
def Line (y : Int) := { p : LatticePoint | p.y = y }

-- Define a parallelogram
def isParallelogram (a b c d : LatticePoint) : Prop :=
  d.x = a.x + c.x - b.x ∧ d.y = a.y + c.y - b.y

-- Main theorem
theorem coloring_satisfies_conditions :
  (∀ c : Color, ∃ (S : Set Int), Infinite S ∧ ∀ y ∈ S, ∃ x : Int, color ⟨x, y⟩ = c) ∧
  (∀ a b c : LatticePoint, 
    color a = Color.White → color b = Color.Red → color c = Color.Black →
    ∃ d : LatticePoint, color d = Color.Red ∧ isParallelogram a b c d) :=
sorry


end NUMINAMATH_CALUDE_coloring_satisfies_conditions_l2626_262602


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l2626_262652

theorem wrapping_paper_fraction (total_fraction : ℚ) (num_presents : ℕ) :
  total_fraction = 5/12 ∧ num_presents = 5 →
  total_fraction / num_presents = 1/12 := by
sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l2626_262652


namespace NUMINAMATH_CALUDE_distance_equals_speed_times_time_l2626_262693

/-- The distance between Emily's house and Timothy's house -/
def distance : ℝ := 10

/-- Emily's speed in miles per hour -/
def speed : ℝ := 5

/-- Time taken for Emily to reach Timothy's house in hours -/
def time : ℝ := 2

/-- Theorem stating that the distance is equal to speed multiplied by time -/
theorem distance_equals_speed_times_time : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_distance_equals_speed_times_time_l2626_262693


namespace NUMINAMATH_CALUDE_base_conversion_and_sum_l2626_262646

-- Define the value of 537 in base 8
def base_8_value : ℕ := 5 * 8^2 + 3 * 8^1 + 7 * 8^0

-- Define the value of 1C2E in base 16, where C = 12 and E = 14
def base_16_value : ℕ := 1 * 16^3 + 12 * 16^2 + 2 * 16^1 + 14 * 16^0

-- Theorem statement
theorem base_conversion_and_sum :
  base_8_value + base_16_value = 7565 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_and_sum_l2626_262646


namespace NUMINAMATH_CALUDE_min_distance_curve_line_l2626_262662

theorem min_distance_curve_line (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) :
  ∃ (min_val : ℝ), min_val = 1 ∧ 
  ∀ (x y : ℝ), Real.log (x + 1) + y - 3 * x = 0 → 
  ∀ (u v : ℝ), 2 * u - v + Real.sqrt 5 = 0 → 
  (y - v)^2 + (x - u)^2 ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_curve_line_l2626_262662


namespace NUMINAMATH_CALUDE_min_value_symmetric_circle_l2626_262660

/-- Given a circle and a line, if the circle is symmetric about the line,
    then the minimum value of 1/a + 2/b is 3 -/
theorem min_value_symmetric_circle (x y a b : ℝ) :
  x^2 + y^2 - 2*x - 4*y + 3 = 0 →
  a > 0 →
  b > 0 →
  a*x + b*y = 3 →
  (∃ (c : ℝ), c > 0 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → a'*x + b'*y = 3 → 1/a' + 2/b' ≥ c) →
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀*x + b₀*y = 3 ∧ 1/a₀ + 2/b₀ = 3) :=
by sorry


end NUMINAMATH_CALUDE_min_value_symmetric_circle_l2626_262660


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2626_262603

theorem pure_imaginary_fraction (b : ℝ) : 
  (∃ (y : ℝ), (b + Complex.I) / (2 + Complex.I) = Complex.I * y) → b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2626_262603


namespace NUMINAMATH_CALUDE_correct_divisor_problem_l2626_262666

theorem correct_divisor_problem (dividend : ℕ) (incorrect_divisor : ℕ) (incorrect_answer : ℕ) (correct_answer : ℕ) :
  dividend = incorrect_divisor * incorrect_answer →
  dividend / correct_answer = 36 →
  incorrect_divisor = 48 →
  incorrect_answer = 24 →
  correct_answer = 32 →
  36 = dividend / correct_answer :=
by sorry

end NUMINAMATH_CALUDE_correct_divisor_problem_l2626_262666


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2626_262641

/-- Calculates the number of whole cubes that fit along a given dimension -/
def cubesAlongDimension (dimension : ℕ) (cubeSize : ℕ) : ℕ :=
  dimension / cubeSize

/-- Calculates the volume of a rectangular box -/
def boxVolume (length width height : ℕ) : ℕ :=
  length * width * height

/-- Calculates the volume of a cube -/
def cubeVolume (size : ℕ) : ℕ :=
  size * size * size

/-- Calculates the total volume occupied by cubes in the box -/
def occupiedVolume (boxLength boxWidth boxHeight cubeSize : ℕ) : ℕ :=
  let numCubesLength := cubesAlongDimension boxLength cubeSize
  let numCubesWidth := cubesAlongDimension boxWidth cubeSize
  let numCubesHeight := cubesAlongDimension boxHeight cubeSize
  let totalCubes := numCubesLength * numCubesWidth * numCubesHeight
  totalCubes * cubeVolume cubeSize

theorem cube_volume_ratio (boxLength boxWidth boxHeight cubeSize : ℕ) 
  (h1 : boxLength = 4)
  (h2 : boxWidth = 7)
  (h3 : boxHeight = 8)
  (h4 : cubeSize = 2) :
  (occupiedVolume boxLength boxWidth boxHeight cubeSize : ℚ) / 
  (boxVolume boxLength boxWidth boxHeight : ℚ) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2626_262641


namespace NUMINAMATH_CALUDE_cube_of_negative_product_l2626_262661

theorem cube_of_negative_product (a b : ℝ) :
  (-2 * a^2 * b^3)^3 = -8 * a^6 * b^9 := by sorry

end NUMINAMATH_CALUDE_cube_of_negative_product_l2626_262661


namespace NUMINAMATH_CALUDE_simplify_fourth_roots_l2626_262664

theorem simplify_fourth_roots : Real.sqrt (Real.sqrt 81) - Real.sqrt (Real.sqrt 256) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fourth_roots_l2626_262664


namespace NUMINAMATH_CALUDE_sum_of_factors_l2626_262690

theorem sum_of_factors (a b c : ℤ) : 
  (∀ x, x^2 + 10*x + 21 = (x + a) * (x + b)) →
  (∀ x, x^2 + 3*x - 88 = (x + b) * (x - c)) →
  a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l2626_262690


namespace NUMINAMATH_CALUDE_cos_315_degrees_l2626_262605

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l2626_262605


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l2626_262625

theorem least_three_digit_multiple_of_13 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 13 ∣ n → 104 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l2626_262625


namespace NUMINAMATH_CALUDE_total_amount_spent_l2626_262636

/-- Calculates the total amount spent on pencils, cucumbers, and notebooks given specific conditions --/
theorem total_amount_spent 
  (initial_cost : ℝ)
  (pencil_discount : ℝ)
  (notebook_discount : ℝ)
  (pencil_tax : ℝ)
  (cucumber_tax : ℝ)
  (cucumber_count : ℕ)
  (notebook_count : ℕ)
  (h1 : initial_cost = 20)
  (h2 : pencil_discount = 0.2)
  (h3 : notebook_discount = 0.3)
  (h4 : pencil_tax = 0.05)
  (h5 : cucumber_tax = 0.1)
  (h6 : cucumber_count = 100)
  (h7 : notebook_count = 25) :
  (cucumber_count / 2 : ℝ) * (initial_cost * (1 - pencil_discount) * (1 + pencil_tax)) +
  (cucumber_count : ℝ) * (initial_cost * (1 + cucumber_tax)) +
  (notebook_count : ℝ) * (initial_cost * (1 - notebook_discount)) = 3390 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_spent_l2626_262636


namespace NUMINAMATH_CALUDE_bisecting_line_slope_intercept_sum_l2626_262699

/-- Triangle ABC with vertices A(0, 8), B(2, 0), C(8, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Check if a line bisects the area of a triangle -/
def bisects_area (l : Line) (t : Triangle) : Prop := sorry

/-- The line through point B that bisects the area of the triangle -/
def bisecting_line (t : Triangle) : Line := sorry

/-- The theorem to be proved -/
theorem bisecting_line_slope_intercept_sum (t : Triangle) :
  t.A = (0, 8) ∧ t.B = (2, 0) ∧ t.C = (8, 0) →
  let l := bisecting_line t
  l.slope + l.y_intercept = -2 := by sorry

end NUMINAMATH_CALUDE_bisecting_line_slope_intercept_sum_l2626_262699


namespace NUMINAMATH_CALUDE_sample_mean_estimates_population_mean_l2626_262692

/-- A type to represent statistical populations -/
structure Population where
  mean : ℝ

/-- A type to represent samples from a population -/
structure Sample where
  mean : ℝ

/-- Predicate to determine if a sample mean is an estimate of a population mean -/
def is_estimate_of (s : Sample) (p : Population) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ |s.mean - p.mean| < ε

/-- Theorem stating that a sample mean is an estimate of the population mean -/
theorem sample_mean_estimates_population_mean (s : Sample) (p : Population) :
  is_estimate_of s p :=
sorry

end NUMINAMATH_CALUDE_sample_mean_estimates_population_mean_l2626_262692


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2626_262677

theorem quadratic_root_value (v : ℝ) : 
  (8 * ((-26 - Real.sqrt 450) / 10)^2 + 26 * ((-26 - Real.sqrt 450) / 10) + v = 0) → 
  v = 113 / 16 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2626_262677


namespace NUMINAMATH_CALUDE_expected_twos_is_half_l2626_262623

/-- The probability of rolling a 2 on a standard die -/
def prob_two : ℚ := 1/6

/-- The probability of not rolling a 2 on a standard die -/
def prob_not_two : ℚ := 1 - prob_two

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 2's when rolling three standard dice -/
def expected_twos : ℚ :=
  0 * (prob_not_two ^ num_dice) +
  1 * (num_dice * prob_two * prob_not_two ^ 2) +
  2 * (num_dice * prob_two ^ 2 * prob_not_two) +
  3 * (prob_two ^ num_dice)

theorem expected_twos_is_half : expected_twos = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_twos_is_half_l2626_262623


namespace NUMINAMATH_CALUDE_turban_count_is_one_l2626_262629

/-- The number of turbans given as part of the annual salary -/
def turban_count : ℕ := sorry

/-- The price of one turban in Rupees -/
def turban_price : ℕ := 30

/-- The base salary in Rupees -/
def base_salary : ℕ := 90

/-- The total annual salary in Rupees -/
def annual_salary : ℕ := base_salary + turban_count * turban_price

/-- The fraction of the year worked by the servant -/
def fraction_worked : ℚ := 3/4

/-- The amount received by the servant after 9 months in Rupees -/
def amount_received : ℕ := 60 + turban_price

theorem turban_count_is_one :
  (fraction_worked * annual_salary = amount_received) → turban_count = 1 := by
  sorry

end NUMINAMATH_CALUDE_turban_count_is_one_l2626_262629


namespace NUMINAMATH_CALUDE_scientific_notation_pm25_express_y_in_terms_of_x_power_evaluation_l2626_262671

-- Problem 1
theorem scientific_notation_pm25 : 
  ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.5 ∧ n = -6 :=
sorry

-- Problem 2
theorem express_y_in_terms_of_x (x y : ℝ) :
  2 * x - 5 * y = 5 → y = 0.4 * x - 1 :=
sorry

-- Problem 3
theorem power_evaluation (x y : ℝ) :
  x + 2 * y - 4 = 0 → (2 : ℝ) ^ (2 * y) * (2 : ℝ) ^ (x - 2) = 4 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_pm25_express_y_in_terms_of_x_power_evaluation_l2626_262671


namespace NUMINAMATH_CALUDE_gasoline_cost_calculation_l2626_262630

/-- Represents the cost of gasoline per liter -/
def gasoline_cost : ℝ := sorry

/-- Represents the trip distance one way in kilometers -/
def one_way_distance : ℝ := 150

/-- Represents the cost of the first car rental option per day, excluding gasoline -/
def first_option_cost : ℝ := 50

/-- Represents the cost of the second car rental option per day, including gasoline -/
def second_option_cost : ℝ := 90

/-- Represents the distance a liter of gasoline can cover in kilometers -/
def km_per_liter : ℝ := 15

/-- Represents the amount saved by choosing the first option over the second option -/
def savings : ℝ := 22

theorem gasoline_cost_calculation : gasoline_cost = 3.4 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_cost_calculation_l2626_262630


namespace NUMINAMATH_CALUDE_car_airplane_energy_consumption_ratio_l2626_262649

theorem car_airplane_energy_consumption_ratio :
  ∀ (maglev airplane car : ℝ),
    maglev > 0 → airplane > 0 → car > 0 →
    maglev = (1/3) * airplane →
    maglev = 0.7 * car →
    car = (10/21) * airplane :=
by sorry

end NUMINAMATH_CALUDE_car_airplane_energy_consumption_ratio_l2626_262649


namespace NUMINAMATH_CALUDE_overall_percentage_favor_l2626_262681

-- Define the given percentages
def starting_favor_percent : ℝ := 0.40
def experienced_favor_percent : ℝ := 0.70

-- Define the number of surveyed entrepreneurs
def num_starting : ℕ := 300
def num_experienced : ℕ := 500

-- Define the total number surveyed
def total_surveyed : ℕ := num_starting + num_experienced

-- Define the number in favor for each group
def num_starting_favor : ℝ := starting_favor_percent * num_starting
def num_experienced_favor : ℝ := experienced_favor_percent * num_experienced

-- Define the total number in favor
def total_favor : ℝ := num_starting_favor + num_experienced_favor

-- Theorem to prove
theorem overall_percentage_favor :
  (total_favor / total_surveyed) * 100 = 58.75 := by
  sorry

end NUMINAMATH_CALUDE_overall_percentage_favor_l2626_262681


namespace NUMINAMATH_CALUDE_cylinder_prism_height_equality_l2626_262684

/-- The height of a cylinder is equal to the height of a rectangular prism 
    when they have the same volume and base area. -/
theorem cylinder_prism_height_equality 
  (V : ℝ) -- Volume of both shapes
  (A : ℝ) -- Base area of both shapes
  (h_cylinder : ℝ) -- Height of the cylinder
  (h_prism : ℝ) -- Height of the rectangular prism
  (h_cylinder_def : h_cylinder = V / A) -- Definition of cylinder height
  (h_prism_def : h_prism = V / A) -- Definition of prism height
  : h_cylinder = h_prism := by
  sorry

end NUMINAMATH_CALUDE_cylinder_prism_height_equality_l2626_262684


namespace NUMINAMATH_CALUDE_base12_addition_l2626_262648

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Digit12 to its corresponding natural number --/
def digit12ToNat (d : Digit12) : ℕ :=
  match d with
  | Digit12.D0 => 0
  | Digit12.D1 => 1
  | Digit12.D2 => 2
  | Digit12.D3 => 3
  | Digit12.D4 => 4
  | Digit12.D5 => 5
  | Digit12.D6 => 6
  | Digit12.D7 => 7
  | Digit12.D8 => 8
  | Digit12.D9 => 9
  | Digit12.A => 10
  | Digit12.B => 11
  | Digit12.C => 12

/-- Represents a number in base 12 --/
def Number12 := List Digit12

/-- Converts a Number12 to its corresponding natural number --/
def number12ToNat (n : Number12) : ℕ :=
  n.foldr (fun d acc => digit12ToNat d + 12 * acc) 0

/-- The theorem to be proved --/
theorem base12_addition :
  let n1 : Number12 := [Digit12.C, Digit12.D9, Digit12.D7]
  let n2 : Number12 := [Digit12.D2, Digit12.D6, Digit12.A]
  let result : Number12 := [Digit12.D3, Digit12.D4, Digit12.D1, Digit12.B]
  number12ToNat n1 + number12ToNat n2 = number12ToNat result := by
  sorry

end NUMINAMATH_CALUDE_base12_addition_l2626_262648


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2626_262670

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem multiplication_puzzle : ∃! (a b : ℕ), 
  (1000 ≤ a * b) ∧ (a * b < 10000) ∧  -- 4-digit product
  (digit_sum a = digit_sum b) ∧       -- same digit sum
  (a * b % 10 = a % 10) ∧             -- ones digit condition
  ((a * b / 10) % 10 = 2) ∧           -- tens digit condition
  a = 2231 ∧ b = 26 := by
sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2626_262670


namespace NUMINAMATH_CALUDE_root_value_l2626_262654

-- Define the polynomials and their roots
def f (x : ℝ) := x^3 + 5*x^2 + 2*x - 8
def g (x p q r : ℝ) := x^3 + p*x^2 + q*x + r

-- Define the roots
variable (a b c : ℝ)

-- State the conditions
axiom root_f : f a = 0 ∧ f b = 0 ∧ f c = 0
axiom root_g : ∃ p q r, g (2*a + b) p q r = 0 ∧ g (2*b + c) p q r = 0 ∧ g (2*c + a) p q r = 0

-- State the theorem to be proved
theorem root_value : ∃ p q, g (2*a + b) p q 18 = 0 ∧ g (2*b + c) p q 18 = 0 ∧ g (2*c + a) p q 18 = 0 :=
sorry

end NUMINAMATH_CALUDE_root_value_l2626_262654


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2626_262665

theorem complex_equation_sum (a b : ℝ) :
  (2 + b * Complex.I) / (1 - Complex.I) = a * Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2626_262665


namespace NUMINAMATH_CALUDE_candy_cost_450_l2626_262686

/-- The cost of buying a specified number of chocolate candies. -/
def candy_cost (total_candies : ℕ) (candies_per_box : ℕ) (cost_per_box : ℕ) : ℕ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem: The cost of buying 450 chocolate candies is $120, given that a box of 30 candies costs $8. -/
theorem candy_cost_450 : candy_cost 450 30 8 = 120 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_450_l2626_262686


namespace NUMINAMATH_CALUDE_range_of_a_l2626_262697

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x + 3*a else a^x

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (∀ x y : ℝ, x < y → f a x > f a y) →
  (1/3 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2626_262697


namespace NUMINAMATH_CALUDE_factorization_equality_l2626_262634

theorem factorization_equality (a b : ℝ) : a^2 * b - b = b * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2626_262634


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l2626_262643

theorem integer_solutions_of_equation :
  ∀ n : ℤ, (1/3 : ℚ) * n^4 - (1/21 : ℚ) * n^3 - n^2 - (11/21 : ℚ) * n + (4/42 : ℚ) = 0 ↔ n = -1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l2626_262643


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l2626_262612

-- Define the number of jelly beans for each color
def red_beans : ℕ := 7
def green_beans : ℕ := 9
def yellow_beans : ℕ := 8
def blue_beans : ℕ := 10
def orange_beans : ℕ := 5

-- Define the total number of jelly beans
def total_beans : ℕ := red_beans + green_beans + yellow_beans + blue_beans + orange_beans

-- Define the number of blue or orange jelly beans
def blue_or_orange_beans : ℕ := blue_beans + orange_beans

-- Theorem statement
theorem jelly_bean_probability : 
  (blue_or_orange_beans : ℚ) / (total_beans : ℚ) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l2626_262612


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l2626_262615

theorem max_students_equal_distribution (pens toys : ℕ) (h_pens : pens = 451) (h_toys : toys = 410) :
  (∃ (students : ℕ), students > 0 ∧ pens % students = 0 ∧ toys % students = 0 ∧
    ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ toys % n ≠ 0)) ↔
  (Nat.gcd pens toys = 41) :=
sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l2626_262615


namespace NUMINAMATH_CALUDE_prime_factors_sum_l2626_262651

theorem prime_factors_sum (w x y z t : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 11^t = 2310 → 2*w + 3*x + 5*y + 7*z + 11*t = 28 := by
sorry

end NUMINAMATH_CALUDE_prime_factors_sum_l2626_262651


namespace NUMINAMATH_CALUDE_gain_percentage_l2626_262691

/-- 
If the cost price of 50 articles equals the selling price of 46 articles, 
then the gain percentage is (1/11.5) * 100.
-/
theorem gain_percentage (C S : ℝ) (h : 50 * C = 46 * S) : 
  (S - C) / C * 100 = (1 / 11.5) * 100 := by
  sorry

end NUMINAMATH_CALUDE_gain_percentage_l2626_262691


namespace NUMINAMATH_CALUDE_art_students_count_l2626_262640

/-- Represents the number of students taking art in a high school -/
def students_taking_art (total students_taking_music students_taking_both students_taking_neither : ℕ) : ℕ :=
  total - students_taking_music - students_taking_neither + students_taking_both

/-- Theorem stating that 10 students are taking art given the conditions -/
theorem art_students_count :
  students_taking_art 500 30 10 470 = 10 := by
  sorry

end NUMINAMATH_CALUDE_art_students_count_l2626_262640


namespace NUMINAMATH_CALUDE_x_values_when_two_in_M_l2626_262633

def M (x : ℝ) : Set ℝ := {-2, 3*x^2 + 3*x - 4}

theorem x_values_when_two_in_M (x : ℝ) : 2 ∈ M x → x = 1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_values_when_two_in_M_l2626_262633


namespace NUMINAMATH_CALUDE_obtuse_triangle_x_range_l2626_262650

/-- Given three line segments with lengths x^2+4, 4x, and x^2+8,
    this theorem states the range of x values that can form an obtuse triangle. -/
theorem obtuse_triangle_x_range (x : ℝ) :
  (∃ (a b c : ℝ), a = x^2 + 4 ∧ b = 4*x ∧ c = x^2 + 8 ∧
   a > 0 ∧ b > 0 ∧ c > 0 ∧
   a + b > c ∧ b + c > a ∧ a + c > b ∧
   c^2 > a^2 + b^2) ↔ 
  (1 < x ∧ x < Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_x_range_l2626_262650


namespace NUMINAMATH_CALUDE_pentagon_perimeter_even_l2626_262601

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a pentagon as a list of 5 points
def Pentagon : Type := List IntPoint

-- Function to calculate the distance between two points
def distance (p1 p2 : IntPoint) : Int :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

-- Function to check if a pentagon has integer side lengths
def hasIntegerSideLengths (p : Pentagon) : Prop :=
  match p with
  | [a, b, c, d, e] => 
    ∃ (l1 l2 l3 l4 l5 : Int),
      distance a b = l1 ^ 2 ∧
      distance b c = l2 ^ 2 ∧
      distance c d = l3 ^ 2 ∧
      distance d e = l4 ^ 2 ∧
      distance e a = l5 ^ 2
  | _ => False

-- Function to calculate the perimeter of a pentagon
def perimeter (p : Pentagon) : Int :=
  match p with
  | [a, b, c, d, e] => 
    Int.sqrt (distance a b) +
    Int.sqrt (distance b c) +
    Int.sqrt (distance c d) +
    Int.sqrt (distance d e) +
    Int.sqrt (distance e a)
  | _ => 0

-- Theorem statement
theorem pentagon_perimeter_even (p : Pentagon) 
  (h1 : p.length = 5)
  (h2 : hasIntegerSideLengths p) :
  Even (perimeter p) := by
  sorry


end NUMINAMATH_CALUDE_pentagon_perimeter_even_l2626_262601


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2626_262606

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) :
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 41 / 20 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2626_262606


namespace NUMINAMATH_CALUDE_product_of_roots_l2626_262647

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 22 → 
  ∃ y : ℝ, (y + 3) * (y - 5) = 22 ∧ x * y = -37 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l2626_262647


namespace NUMINAMATH_CALUDE_binomial_product_l2626_262695

theorem binomial_product (x : ℝ) : (4 * x + 3) * (x - 7) = 4 * x^2 - 25 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l2626_262695


namespace NUMINAMATH_CALUDE_prob_sum_equals_seven_ninths_l2626_262694

def total_balls : ℕ := 9
def black_balls : ℕ := 5
def white_balls : ℕ := 4

def P_A : ℚ := black_balls / total_balls
def P_B_given_A : ℚ := white_balls / (total_balls - 1)

theorem prob_sum_equals_seven_ninths :
  (P_A * P_B_given_A) + P_B_given_A = 7 / 9 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_equals_seven_ninths_l2626_262694


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l2626_262627

/-- Given a complex number Z = 1 + i, prove that the point corresponding to 1/Z + Z 
    lies in the first quadrant. -/
theorem point_in_first_quadrant (Z : ℂ) (h : Z = 1 + Complex.I) : 
  let W := Z⁻¹ + Z
  0 < W.re ∧ 0 < W.im := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l2626_262627


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l2626_262658

theorem solution_satisfies_system :
  let solutions : List (ℝ × ℝ) := [(5, -3), (5, 3), (-Real.sqrt 118 / 2, 3 * Real.sqrt 2 / 2), (-Real.sqrt 118 / 2, -3 * Real.sqrt 2 / 2)]
  ∀ (x y : ℝ), (x, y) ∈ solutions →
    (x^2 + y^2 = 34 ∧ x - y + Real.sqrt ((x - y) / (x + y)) = 20 / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l2626_262658


namespace NUMINAMATH_CALUDE_seven_is_unique_solution_l2626_262682

/-- Product of all prime numbers less than n -/
def n_question_mark (n : ℕ) : ℕ :=
  (Finset.filter Nat.Prime (Finset.range n)).prod id

/-- The theorem stating that 7 is the only solution -/
theorem seven_is_unique_solution :
  ∃! (n : ℕ), n > 3 ∧ n_question_mark n = 2 * n + 16 :=
sorry

end NUMINAMATH_CALUDE_seven_is_unique_solution_l2626_262682


namespace NUMINAMATH_CALUDE_abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three_negation_true_implies_converse_true_l2626_262687

-- Statement 1
theorem abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three :
  (∀ x : ℝ, |x - 1| < 2 → x < 3) ∧
  ¬(∀ x : ℝ, x < 3 → |x - 1| < 2) :=
sorry

-- Statement 2
theorem negation_true_implies_converse_true (P Q : Prop) :
  (¬(P → Q) → (Q → P)) :=
sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three_negation_true_implies_converse_true_l2626_262687


namespace NUMINAMATH_CALUDE_group_size_from_weight_change_l2626_262617

/-- The number of people in a group where replacing a 35 kg person with a 55 kg person
    increases the average weight by 2.5 kg is 8. -/
theorem group_size_from_weight_change (n : ℕ) : 
  (n : ℝ) * 2.5 = 55 - 35 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_group_size_from_weight_change_l2626_262617


namespace NUMINAMATH_CALUDE_triangle_QCA_area_l2626_262674

/-- The area of triangle QCA given the coordinates of Q, A, and C, and that QA is perpendicular to QC -/
theorem triangle_QCA_area (p : ℝ) : 
  let Q : ℝ × ℝ := (0, 15)
  let A : ℝ × ℝ := (3, 15)
  let C : ℝ × ℝ := (0, p)
  -- QA is perpendicular to QC (implicit in the coordinate system)
  (45 - 3 * p) / 2 = (1 / 2) * 3 * (15 - p) := by
  sorry

end NUMINAMATH_CALUDE_triangle_QCA_area_l2626_262674


namespace NUMINAMATH_CALUDE_star_minus_emilio_sum_equals_104_l2626_262631

def star_list := List.range 40 |>.map (· + 1)

def replace_three_with_two (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list := star_list.map replace_three_with_two

theorem star_minus_emilio_sum_equals_104 :
  star_list.sum - emilio_list.sum = 104 := by
  sorry

end NUMINAMATH_CALUDE_star_minus_emilio_sum_equals_104_l2626_262631


namespace NUMINAMATH_CALUDE_sams_new_books_l2626_262600

theorem sams_new_books : 
  ∀ (adventure_books mystery_books used_books : ℕ),
    adventure_books = 24 →
    mystery_books = 37 →
    used_books = 18 →
    adventure_books + mystery_books - used_books = 43 := by
  sorry

end NUMINAMATH_CALUDE_sams_new_books_l2626_262600


namespace NUMINAMATH_CALUDE_keith_seashells_l2626_262610

/-- Proves the number of seashells Keith found given the problem conditions -/
theorem keith_seashells (mary_shells : ℕ) (total_shells : ℕ) (cracked_shells : ℕ) :
  mary_shells = 2 →
  total_shells = 7 →
  cracked_shells = 9 →
  total_shells - mary_shells = 5 :=
by sorry

end NUMINAMATH_CALUDE_keith_seashells_l2626_262610


namespace NUMINAMATH_CALUDE_larger_cuboid_width_l2626_262614

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- The smaller cuboid -/
def small_cuboid : Cuboid := { length := 5, width := 6, height := 3 }

/-- The larger cuboid -/
def large_cuboid (w : ℝ) : Cuboid := { length := 18, width := w, height := 2 }

/-- The number of smaller cuboids that can be formed from the larger cuboid -/
def num_small_cuboids : ℕ := 6

theorem larger_cuboid_width :
  ∃ w : ℝ, volume (large_cuboid w) = num_small_cuboids * volume small_cuboid ∧ w = 15 := by
  sorry

end NUMINAMATH_CALUDE_larger_cuboid_width_l2626_262614


namespace NUMINAMATH_CALUDE_min_employees_for_agency_l2626_262680

/-- Represents the number of employees needed for different pollution monitoring tasks -/
structure EmployeeRequirements where
  water : ℕ
  air : ℕ
  both : ℕ
  soil : ℕ

/-- Calculates the minimum number of employees needed given the requirements -/
def minEmployees (req : EmployeeRequirements) : ℕ :=
  req.water + req.air - req.both

/-- Theorem stating that given the specific requirements, 160 employees are needed -/
theorem min_employees_for_agency (req : EmployeeRequirements) 
  (h_water : req.water = 120)
  (h_air : req.air = 105)
  (h_both : req.both = 65)
  (h_soil : req.soil = 40)
  : minEmployees req = 160 := by
  sorry

#eval minEmployees { water := 120, air := 105, both := 65, soil := 40 }

end NUMINAMATH_CALUDE_min_employees_for_agency_l2626_262680


namespace NUMINAMATH_CALUDE_fraction_simplification_l2626_262626

theorem fraction_simplification :
  1 / (1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4 + 1 / (1/3)^5) = 1 / 360 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2626_262626


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2626_262628

theorem sqrt_product_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (42 * p) * Real.sqrt (7 * p) * Real.sqrt (14 * p) = 42 * p * Real.sqrt (7 * p) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2626_262628


namespace NUMINAMATH_CALUDE_total_fish_l2626_262609

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 9) : 
  lilly_fish + rosy_fish = 19 := by
sorry

end NUMINAMATH_CALUDE_total_fish_l2626_262609


namespace NUMINAMATH_CALUDE_swim_club_percentage_passed_l2626_262656

/-- The percentage of swim club members who have passed the lifesaving test -/
def percentage_passed (total_members : ℕ) (not_passed_with_course : ℕ) (not_passed_without_course : ℕ) : ℚ :=
  1 - (not_passed_with_course + not_passed_without_course : ℚ) / total_members

theorem swim_club_percentage_passed :
  percentage_passed 100 40 30 = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_swim_club_percentage_passed_l2626_262656


namespace NUMINAMATH_CALUDE_bill_amount_is_1550_l2626_262698

/-- Calculates the amount of a bill given its true discount, due date, and interest rate. -/
def bill_amount (true_discount : ℚ) (months : ℚ) (annual_rate : ℚ) : ℚ :=
  let present_value := true_discount / (annual_rate * (months / 12) / (1 + annual_rate * (months / 12)))
  present_value + true_discount

/-- Theorem stating that the bill amount is 1550 given the specified conditions. -/
theorem bill_amount_is_1550 :
  bill_amount 150 9 (16 / 100) = 1550 := by
  sorry

end NUMINAMATH_CALUDE_bill_amount_is_1550_l2626_262698


namespace NUMINAMATH_CALUDE_simplify_expression_l2626_262618

theorem simplify_expression (x y : ℝ) : 20 * (x + y) - 19 * (y + x) = x + y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2626_262618


namespace NUMINAMATH_CALUDE_scaled_tetrahedron_volume_ratio_l2626_262645

-- Define a regular tetrahedron
def RegularTetrahedron : Type := Unit

-- Define a function to scale down coordinates
def scaleDown (t : RegularTetrahedron) : RegularTetrahedron := sorry

-- Define a function to calculate the volume of a tetrahedron
def volume (t : RegularTetrahedron) : ℝ := sorry

-- Theorem statement
theorem scaled_tetrahedron_volume_ratio 
  (t : RegularTetrahedron) : 
  volume (scaleDown t) / volume t = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_scaled_tetrahedron_volume_ratio_l2626_262645


namespace NUMINAMATH_CALUDE_race_theorem_l2626_262642

/-- Represents a runner in the race -/
structure Runner :=
  (speed : ℝ)

/-- The length of the race in meters -/
def race_length : ℝ := 1000

/-- The distance runner A finishes ahead of runner C -/
def a_ahead_of_c : ℝ := 200

/-- The distance runner B finishes ahead of runner C -/
def b_ahead_of_c : ℝ := 157.89473684210532

theorem race_theorem (A B C : Runner) :
  A.speed > B.speed ∧ B.speed > C.speed →
  a_ahead_of_c = A.speed * race_length / C.speed - race_length →
  b_ahead_of_c = B.speed * race_length / C.speed - race_length →
  A.speed * race_length / B.speed - race_length = a_ahead_of_c - b_ahead_of_c :=
by sorry

end NUMINAMATH_CALUDE_race_theorem_l2626_262642


namespace NUMINAMATH_CALUDE_isabel_morning_runs_l2626_262696

/-- Represents the number of times Isabel runs the circuit in the morning -/
def morning_runs : ℕ := 7

/-- Represents the length of the circuit in meters -/
def circuit_length : ℕ := 365

/-- Represents the number of times Isabel runs the circuit in the afternoon -/
def afternoon_runs : ℕ := 3

/-- Represents the total distance Isabel runs in a week in meters -/
def weekly_distance : ℕ := 25550

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

theorem isabel_morning_runs :
  morning_runs * circuit_length * days_in_week +
  afternoon_runs * circuit_length * days_in_week = weekly_distance :=
sorry

end NUMINAMATH_CALUDE_isabel_morning_runs_l2626_262696


namespace NUMINAMATH_CALUDE_quadratic_solution_l2626_262613

theorem quadratic_solution (b : ℤ) : 
  ((-5 : ℤ)^2 + b * (-5) - 35 = 0) → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2626_262613


namespace NUMINAMATH_CALUDE_extended_parallelepiped_volume_calculation_l2626_262608

/-- The volume of the set of points inside or within two units of a rectangular parallelepiped with dimensions 2 by 3 by 4 units -/
def extended_parallelepiped_volume : ℝ := sorry

/-- The dimensions of the rectangular parallelepiped -/
def parallelepiped_dimensions : Fin 3 → ℝ
| 0 => 2
| 1 => 3
| 2 => 4
| _ => 0

/-- The extension distance around the parallelepiped -/
def extension_distance : ℝ := 2

theorem extended_parallelepiped_volume_calculation :
  extended_parallelepiped_volume = (384 + 140 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_extended_parallelepiped_volume_calculation_l2626_262608


namespace NUMINAMATH_CALUDE_school_competition_selections_l2626_262663

theorem school_competition_selections (n m : ℕ) (hn : n = 5) (hm : m = 3) :
  (n.choose m) * m.factorial = 60 := by
  sorry

end NUMINAMATH_CALUDE_school_competition_selections_l2626_262663


namespace NUMINAMATH_CALUDE_common_chord_length_is_2_sqrt_5_l2626_262621

/-- Two circles C₁ and C₂ in a 2D plane -/
structure TwoCircles where
  /-- Center of circle C₁ -/
  center1 : ℝ × ℝ
  /-- Radius of circle C₁ -/
  radius1 : ℝ
  /-- Center of circle C₂ -/
  center2 : ℝ × ℝ
  /-- Radius of circle C₂ -/
  radius2 : ℝ

/-- The length of the common chord between two intersecting circles -/
def commonChordLength (circles : TwoCircles) : ℝ :=
  sorry

/-- Theorem: The length of the common chord between the given intersecting circles is 2√5 -/
theorem common_chord_length_is_2_sqrt_5 :
  let circles : TwoCircles := {
    center1 := (2, 1),
    radius1 := Real.sqrt 10,
    center2 := (-6, -3),
    radius2 := Real.sqrt 50
  }
  commonChordLength circles = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_is_2_sqrt_5_l2626_262621


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_range_l2626_262673

def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem cubic_polynomial_sum_range (a b c d : ℝ) (h_a : a ≠ 0) :
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 2 * f a b c d 2 = t ∧ 3 * f a b c d 3 = t ∧ 4 * f a b c d 4 = t) →
  (∃ y : ℝ, 0 < y ∧ y < 1 ∧ f a b c d 1 + f a b c d 5 = y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_range_l2626_262673


namespace NUMINAMATH_CALUDE_min_additional_marbles_for_lisa_l2626_262689

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed -/
theorem min_additional_marbles_for_lisa : min_additional_marbles 12 34 = 44 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_marbles_for_lisa_l2626_262689


namespace NUMINAMATH_CALUDE_divisibility_property_l2626_262611

theorem divisibility_property (a b c d u : ℤ) 
  (h1 : u ∣ a * c) 
  (h2 : u ∣ b * c + a * d) 
  (h3 : u ∣ b * d) : 
  (u ∣ b * c) ∧ (u ∣ a * d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2626_262611


namespace NUMINAMATH_CALUDE_shorties_eating_today_l2626_262676

/-- Represents the number of shorties who eat donuts every day -/
def daily_eaters : ℕ := 6

/-- Represents the number of shorties who eat donuts every other day -/
def bi_daily_eaters : ℕ := 8

/-- Represents the number of shorties who ate donuts yesterday -/
def yesterday_eaters : ℕ := 11

/-- Theorem stating that the number of shorties who will eat donuts today is 9 -/
theorem shorties_eating_today : 
  ∃ (today_eaters : ℕ), today_eaters = 9 ∧
  today_eaters = daily_eaters + (bi_daily_eaters - (yesterday_eaters - daily_eaters)) :=
by
  sorry


end NUMINAMATH_CALUDE_shorties_eating_today_l2626_262676


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2626_262685

theorem other_root_of_quadratic (m : ℝ) : 
  (3 : ℝ) ^ 2 + m * 3 - 12 = 0 → 
  (-4 : ℝ) ^ 2 + m * (-4) - 12 = 0 := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2626_262685


namespace NUMINAMATH_CALUDE_wax_needed_proof_l2626_262668

/-- Given an amount of wax and a required amount, calculate the additional wax needed -/
def additional_wax_needed (current_amount required_amount : ℕ) : ℕ :=
  required_amount - current_amount

/-- Theorem stating that 17 grams of additional wax are needed -/
theorem wax_needed_proof (current_amount required_amount : ℕ) 
  (h1 : current_amount = 557)
  (h2 : required_amount = 574) :
  additional_wax_needed current_amount required_amount = 17 := by
  sorry

end NUMINAMATH_CALUDE_wax_needed_proof_l2626_262668
