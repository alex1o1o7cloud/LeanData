import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_f_l1028_102811

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y a ≤ f x a) ∧
  (f x a ≤ 11) ∧
  (∃ z ∈ Set.Icc (-2 : ℝ) 2, f z a = 11) →
  (∃ w ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f w a ≤ f y a ∧ f w a = -29) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1028_102811


namespace NUMINAMATH_CALUDE_function_analysis_l1028_102838

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x + a / x^2

theorem function_analysis (a : ℝ) :
  (∀ x, x > 0 → HasDerivAt (f a) ((2 / x) - (2 * a / x^3)) x) →
  (HasDerivAt (f a) 0 1 → a = 1) ∧
  (a > 0 → IsLocalMin (f a) (Real.sqrt a)) ∧
  ((∃ x y, 1 ≤ x ∧ x < y ∧ f a x = 2 ∧ f a y = 2) → 2 ≤ a ∧ a < Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_function_analysis_l1028_102838


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1028_102872

theorem divisibility_theorem (d a n : ℕ) (h1 : 3 ≤ d) (h2 : d ≤ 2^(n+1)) :
  ¬(d ∣ a^(2^n) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1028_102872


namespace NUMINAMATH_CALUDE_two_special_numbers_l1028_102818

/-- A three-digit number divisible by 5 that can be represented as n^3 + n^2 -/
def special_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧  -- three-digit number
  x % 5 = 0 ∧           -- divisible by 5
  ∃ n : ℕ, x = n^3 + n^2  -- can be represented as n^3 + n^2

/-- There are exactly two numbers satisfying the special_number property -/
theorem two_special_numbers : ∃! (a b : ℕ), a ≠ b ∧ special_number a ∧ special_number b ∧
  ∀ x, special_number x → (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_two_special_numbers_l1028_102818


namespace NUMINAMATH_CALUDE_factorization_proof_l1028_102839

theorem factorization_proof (a b c : ℝ) :
  4 * a * b + 2 * a * c = 2 * a * (2 * b + c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1028_102839


namespace NUMINAMATH_CALUDE_quadratic_two_roots_condition_l1028_102899

theorem quadratic_two_roots_condition (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x - 1 + m = 0 ∧ y^2 + 2*y - 1 + m = 0) ↔ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_condition_l1028_102899


namespace NUMINAMATH_CALUDE_proportion_solution_l1028_102877

theorem proportion_solution (y : ℝ) : 
  (0.75 : ℝ) / 1.2 = y / 8 → y = 5 := by
sorry

end NUMINAMATH_CALUDE_proportion_solution_l1028_102877


namespace NUMINAMATH_CALUDE_jellybean_problem_l1028_102822

theorem jellybean_problem (x : ℚ) 
  (caleb_jellybeans : x > 0)
  (sophie_jellybeans : ℚ → ℚ)
  (sophie_half : sophie_jellybeans x = x / 2)
  (total_jellybeans : 12 * x + 12 * (sophie_jellybeans x) = 54) :
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_jellybean_problem_l1028_102822


namespace NUMINAMATH_CALUDE_unique_increasing_matrix_l1028_102823

/-- A 4x4 matrix with entries from 1 to 16 in increasing order -/
def IncreasingMatrix : Type := Matrix (Fin 4) (Fin 4) (Fin 16)

/-- The property that all entries in the matrix are unique and from 1 to 16 -/
def has_all_entries (M : IncreasingMatrix) : Prop :=
  ∀ i : Fin 16, ∃! (row col : Fin 4), M row col = i

/-- The property that each row is in strictly increasing order -/
def rows_increasing (M : IncreasingMatrix) : Prop :=
  ∀ row col₁ col₂, col₁ < col₂ → M row col₁ < M row col₂

/-- The property that each column is in strictly increasing order -/
def cols_increasing (M : IncreasingMatrix) : Prop :=
  ∀ row₁ row₂ col, row₁ < row₂ → M row₁ col < M row₂ col

/-- The main theorem stating there is exactly one matrix satisfying all conditions -/
theorem unique_increasing_matrix :
  ∃! M : IncreasingMatrix,
    has_all_entries M ∧
    rows_increasing M ∧
    cols_increasing M :=
  sorry

end NUMINAMATH_CALUDE_unique_increasing_matrix_l1028_102823


namespace NUMINAMATH_CALUDE_area_of_problem_l_shape_l1028_102836

/-- Represents an L-shaped figure with given dimensions -/
structure LShape where
  short_width : ℝ
  short_length : ℝ
  long_width : ℝ
  long_length : ℝ

/-- Calculates the area of an L-shaped figure -/
def area_of_l_shape (l : LShape) : ℝ :=
  l.short_width * l.short_length + l.long_width * l.long_length

/-- The specific L-shape from the problem -/
def problem_l_shape : LShape :=
  { short_width := 2
    short_length := 3
    long_width := 5
    long_length := 8 }

/-- Theorem stating that the area of the given L-shape is 46 square units -/
theorem area_of_problem_l_shape :
  area_of_l_shape problem_l_shape = 46 := by
  sorry


end NUMINAMATH_CALUDE_area_of_problem_l_shape_l1028_102836


namespace NUMINAMATH_CALUDE_exponent_base_proof_l1028_102858

theorem exponent_base_proof (m : ℤ) (x : ℝ) : 
  ((-2)^(2*m) = x^(12-m)) → (m = 4) → (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_exponent_base_proof_l1028_102858


namespace NUMINAMATH_CALUDE_complex_power_72_l1028_102873

/-- Prove that (cos 215° + i sin 215°)^72 = 1 -/
theorem complex_power_72 : (Complex.exp (215 * Real.pi / 180 * Complex.I))^72 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_72_l1028_102873


namespace NUMINAMATH_CALUDE_pupils_in_program_l1028_102898

/-- Given a program with a total of 238 people and 61 parents, prove that there were 177 pupils present. -/
theorem pupils_in_program (total_people : ℕ) (parents : ℕ) (h1 : total_people = 238) (h2 : parents = 61) :
  total_people - parents = 177 := by
  sorry

end NUMINAMATH_CALUDE_pupils_in_program_l1028_102898


namespace NUMINAMATH_CALUDE_angle_c_in_triangle_l1028_102888

theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_in_triangle_l1028_102888


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1028_102874

def is_geometric_sequence (a : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem geometric_sequence_sum (a : ℕ → ℝ) (h : is_geometric_sequence a 6) 
  (h1 : a 1 = 1) (h2 : a 2 = 2) : 
  (Finset.range 9).sum (λ i => a (i + 1)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1028_102874


namespace NUMINAMATH_CALUDE_equation_solution_l1028_102841

theorem equation_solution (x : ℝ) (h : x ≥ 0) : x + 2 * Real.sqrt x - 8 = 0 ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1028_102841


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l1028_102809

theorem no_perfect_square_in_range : 
  ∀ n : ℕ, 4 ≤ n ∧ n ≤ 12 → ¬∃ m : ℕ, 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l1028_102809


namespace NUMINAMATH_CALUDE_equal_intercept_line_equations_l1028_102819

/-- A line with equal absolute intercepts passing through (3, -2) -/
structure EqualInterceptLine where
  -- The slope of the line
  m : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (3, -2)
  point_condition : -2 = m * 3 + b
  -- The line has equal absolute intercepts
  intercept_condition : ∃ (k : ℝ), k ≠ 0 ∧ (k = -b/m ∨ k = b)

/-- The possible equations of the line -/
def possible_equations (l : EqualInterceptLine) : Prop :=
  (2 * l.m + 3 = 0 ∧ l.b = 0) ∨
  (l.m = -1 ∧ l.b = 1) ∨
  (l.m = 1 ∧ l.b = 5)

theorem equal_intercept_line_equations :
  ∀ (l : EqualInterceptLine), possible_equations l :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equations_l1028_102819


namespace NUMINAMATH_CALUDE_gift_box_volume_l1028_102889

/-- The volume of a rectangular box -/
def box_volume (width length height : ℝ) : ℝ := width * length * height

/-- Theorem: The volume of a box with dimensions 9 cm × 4 cm × 7 cm is 252 cm³ -/
theorem gift_box_volume :
  box_volume 9 4 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_gift_box_volume_l1028_102889


namespace NUMINAMATH_CALUDE_intersection_A_B_l1028_102830

-- Define set A
def A : Set ℝ := {x | |x - 2| ≤ 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1028_102830


namespace NUMINAMATH_CALUDE_max_product_sum_320_l1028_102827

theorem max_product_sum_320 : 
  ∃ (a b : ℤ), a + b = 320 ∧ 
  ∀ (x y : ℤ), x + y = 320 → x * y ≤ a * b ∧
  a * b = 25600 := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_320_l1028_102827


namespace NUMINAMATH_CALUDE_fibonacci_pair_characterization_l1028_102851

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def is_fibonacci_pair (a b : ℝ) : Prop :=
  ∀ n, ∃ m, a * (fibonacci n) + b * (fibonacci (n + 1)) = fibonacci m

theorem fibonacci_pair_characterization :
  ∀ a b : ℝ, is_fibonacci_pair a b ↔ 
    ((a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0) ∨ 
     (∃ k : ℕ, a = fibonacci k ∧ b = fibonacci (k + 1))) :=
sorry

end NUMINAMATH_CALUDE_fibonacci_pair_characterization_l1028_102851


namespace NUMINAMATH_CALUDE_line_point_value_l1028_102803

/-- Given a line passing through points (-1, y) and (4, k), with slope equal to k and k = 1, 
    prove that y = -4 -/
theorem line_point_value (y k : ℝ) (h1 : k = 1) 
    (h2 : (k - y) / (4 - (-1)) = k) : y = -4 := by
  sorry

end NUMINAMATH_CALUDE_line_point_value_l1028_102803


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l1028_102831

theorem muffin_banana_price_ratio :
  ∀ (muffin_price banana_price : ℚ),
  muffin_price > 0 →
  banana_price > 0 →
  4 * muffin_price + 3 * banana_price > 0 →
  2 * (4 * muffin_price + 3 * banana_price) = 2 * muffin_price + 16 * banana_price →
  muffin_price / banana_price = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l1028_102831


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1028_102870

theorem polynomial_simplification (x : ℝ) : 
  5 - 5*x - 10*x^2 + 10 + 15*x - 20*x^2 - 10 + 20*x + 30*x^2 = 5 + 30*x := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1028_102870


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1028_102864

theorem inequality_equivalence (x : ℝ) : 2 * x - 6 < 0 ↔ x < 3 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1028_102864


namespace NUMINAMATH_CALUDE_rams_weight_increase_l1028_102863

theorem rams_weight_increase (ram_weight shyam_weight : ℝ) : 
  ram_weight / shyam_weight = 6 / 5 →
  ∃ (ram_increase : ℝ),
    ram_weight * (1 + ram_increase) + shyam_weight * 1.21 = 82.8 ∧
    (ram_weight * (1 + ram_increase) + shyam_weight * 1.21) / (ram_weight + shyam_weight) = 1.15 →
    ram_increase = 1.48 := by
  sorry

end NUMINAMATH_CALUDE_rams_weight_increase_l1028_102863


namespace NUMINAMATH_CALUDE_total_roses_is_109_l1028_102894

/-- The number of bouquets to be made -/
def num_bouquets : ℕ := 5

/-- The number of table decorations to be made -/
def num_table_decorations : ℕ := 7

/-- The number of white roses used in each bouquet -/
def roses_per_bouquet : ℕ := 5

/-- The number of white roses used in each table decoration -/
def roses_per_table_decoration : ℕ := 12

/-- The total number of white roses needed for all bouquets and table decorations -/
def total_roses_needed : ℕ := num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration

theorem total_roses_is_109 : total_roses_needed = 109 := by
  sorry

end NUMINAMATH_CALUDE_total_roses_is_109_l1028_102894


namespace NUMINAMATH_CALUDE_local_odd_function_part1_local_odd_function_part2_l1028_102821

-- Definition of local odd function
def is_local_odd_function (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∃ x ∈ domain, f (-x) = -f x

-- Part 1
theorem local_odd_function_part1 (m : ℝ) :
  is_local_odd_function (fun x => 2^x + m) (Set.Icc (-2) 2) →
  m ∈ Set.Icc (-17/8) (-1) :=
sorry

-- Part 2
theorem local_odd_function_part2 (m : ℝ) :
  is_local_odd_function (fun x => 4^x + m*2^(x+1) + m^2 - 4) Set.univ →
  m ∈ Set.Icc (-1) (Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_local_odd_function_part1_local_odd_function_part2_l1028_102821


namespace NUMINAMATH_CALUDE_decimal_number_calculation_l1028_102807

theorem decimal_number_calculation (A B : ℝ) 
  (h1 : B - A = 211.5)
  (h2 : B = 10 * A) : 
  A = 23.5 := by
sorry

end NUMINAMATH_CALUDE_decimal_number_calculation_l1028_102807


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1028_102882

theorem inequality_equivalence (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (c - a) * (c - b) * (b - a) < 0 ↔ b*c^2 + c*a^2 + a*b^2 < b^2*c + c^2*a + a^2*b :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1028_102882


namespace NUMINAMATH_CALUDE_greatest_integer_cube_root_three_l1028_102829

theorem greatest_integer_cube_root_three : ⌊(2 + Real.sqrt 3)^3⌋ = 51 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_cube_root_three_l1028_102829


namespace NUMINAMATH_CALUDE_problem_solution_l1028_102820

def f (x : ℝ) : ℝ := x^2 + 10

def g (x : ℝ) : ℝ := x^2 - 6

theorem problem_solution (a : ℝ) (h1 : a > 3) (h2 : f (g a) = 16) : a = Real.sqrt (Real.sqrt 6 + 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1028_102820


namespace NUMINAMATH_CALUDE_sum_distances_constant_l1028_102890

/-- An equilateral triangle -/
structure EquilateralTriangle where
  /-- The side length of the triangle -/
  side_length : ℝ
  /-- Assumption that the side length is positive -/
  side_length_pos : side_length > 0

/-- A point inside an equilateral triangle -/
structure PointInTriangle (t : EquilateralTriangle) where
  /-- The distance from the point to the first side -/
  dist1 : ℝ
  /-- The distance from the point to the second side -/
  dist2 : ℝ
  /-- The distance from the point to the third side -/
  dist3 : ℝ
  /-- Assumption that all distances are non-negative -/
  dist_nonneg : dist1 ≥ 0 ∧ dist2 ≥ 0 ∧ dist3 ≥ 0
  /-- Assumption that the point is inside the triangle -/
  inside : dist1 + dist2 + dist3 < t.side_length * Real.sqrt 3 / 2

/-- The theorem stating that the sum of distances is constant -/
theorem sum_distances_constant (t : EquilateralTriangle) (p : PointInTriangle t) :
  p.dist1 + p.dist2 + p.dist3 = t.side_length * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_distances_constant_l1028_102890


namespace NUMINAMATH_CALUDE_max_sum_reciprocal_constraint_l1028_102896

theorem max_sum_reciprocal_constraint (a b c : ℕ+) : 
  (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c →
  Nat.gcd a.val (Nat.gcd b.val c.val) = 1 →
  a.val + b.val ≤ 2011 →
  (∀ a' b' c' : ℕ+, (1 : ℚ) / a' + (1 : ℚ) / b' = (1 : ℚ) / c' →
    Nat.gcd a'.val (Nat.gcd b'.val c'.val) = 1 →
    a'.val + b'.val ≤ 2011 →
    a'.val + b'.val ≤ a.val + b.val) →
  a.val + b.val = 1936 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_reciprocal_constraint_l1028_102896


namespace NUMINAMATH_CALUDE_twelve_people_round_table_l1028_102852

/-- The number of distinct seating arrangements for n people around a round table,
    where rotations are considered the same. -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem: The number of distinct seating arrangements for 12 people around a round table,
    where rotations are considered the same, is equal to 11!. -/
theorem twelve_people_round_table : roundTableArrangements 12 = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_twelve_people_round_table_l1028_102852


namespace NUMINAMATH_CALUDE_at_least_one_positive_solution_l1028_102849

def f (x : ℝ) : ℝ := x^10 + 4*x^9 + 7*x^8 + 2023*x^7 - 2024*x^6

theorem at_least_one_positive_solution :
  ∃ x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_positive_solution_l1028_102849


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1028_102816

theorem absolute_value_equation_solution (a : ℝ) : 
  50 - |a - 2| = |4 - a| → (a = -22 ∨ a = 28) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1028_102816


namespace NUMINAMATH_CALUDE_line_length_calculation_line_length_proof_l1028_102824

theorem line_length_calculation (initial_length : ℕ) 
  (first_erasure : ℕ) (first_extension : ℕ) 
  (second_erasure : ℕ) (final_addition : ℕ) : ℕ :=
  let step1 := initial_length - first_erasure
  let step2 := step1 + first_extension
  let step3 := step2 - second_erasure
  let final_length := step3 + final_addition
  final_length

theorem line_length_proof :
  line_length_calculation 100 24 35 15 8 = 104 := by
  sorry

end NUMINAMATH_CALUDE_line_length_calculation_line_length_proof_l1028_102824


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l1028_102832

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l1028_102832


namespace NUMINAMATH_CALUDE_max_fraction_sum_l1028_102897

theorem max_fraction_sum (a b c d : ℕ) 
  (h1 : a + c = 1000) 
  (h2 : b + d = 500) : 
  (∀ a' b' c' d' : ℕ, 
    a' + c' = 1000 → 
    b' + d' = 500 → 
    (a' : ℚ) / b' + (c' : ℚ) / d' ≤ 1 / 499 + 999) := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l1028_102897


namespace NUMINAMATH_CALUDE_alan_wings_increase_l1028_102835

/-- Proves that Alan needs to increase his rate by 4 wings per minute to beat Kevin's record -/
theorem alan_wings_increase (kevin_wings : ℕ) (kevin_time : ℕ) (alan_rate : ℕ) : 
  kevin_wings = 64 → 
  kevin_time = 8 → 
  alan_rate = 5 → 
  (kevin_wings / kevin_time : ℚ) - alan_rate = 4 := by
  sorry

#check alan_wings_increase

end NUMINAMATH_CALUDE_alan_wings_increase_l1028_102835


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l1028_102801

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (l m : Line) (α : Plane) :
  subset m α → 
  parallel_lines l m → 
  ¬subset l α → 
  parallel_line_plane l α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l1028_102801


namespace NUMINAMATH_CALUDE_shopping_time_calculation_l1028_102813

/-- Represents the shopping trip details -/
structure ShoppingTrip where
  total_waiting_time : ℕ
  total_active_shopping_time : ℕ
  total_trip_time : ℕ

/-- Calculates the time spent shopping and performing tasks -/
def time_shopping_and_tasks (trip : ShoppingTrip) : ℕ :=
  trip.total_trip_time - trip.total_waiting_time

/-- Theorem stating that the time spent shopping and performing tasks
    is equal to the total trip time minus the total waiting time -/
theorem shopping_time_calculation (trip : ShoppingTrip) 
  (h1 : trip.total_waiting_time = 58)
  (h2 : trip.total_active_shopping_time = 29)
  (h3 : trip.total_trip_time = 135) :
  time_shopping_and_tasks trip = 77 := by
  sorry

#eval time_shopping_and_tasks { total_waiting_time := 58, total_active_shopping_time := 29, total_trip_time := 135 }

end NUMINAMATH_CALUDE_shopping_time_calculation_l1028_102813


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l1028_102847

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 * a 3 - 34 * a 3 + 64 = 0 →
  a 5 * a 5 - 34 * a 5 + 64 = 0 →
  (a 4 = 8 ∨ a 4 = -8) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l1028_102847


namespace NUMINAMATH_CALUDE_division_problem_l1028_102808

theorem division_problem : ∃ (q : ℕ), 
  220070 = (555 + 445) * q + 70 ∧ q = 2 * (555 - 445) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1028_102808


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l1028_102869

theorem triangle_ratio_theorem (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  a = Real.sqrt 3 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (b + c) / (Real.sin B + Real.sin C) = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l1028_102869


namespace NUMINAMATH_CALUDE_mathathon_problem_count_l1028_102867

theorem mathathon_problem_count (rounds : Nat) (problems_per_round : Nat) : 
  rounds = 7 → problems_per_round = 3 → rounds * problems_per_round = 21 := by
  sorry

end NUMINAMATH_CALUDE_mathathon_problem_count_l1028_102867


namespace NUMINAMATH_CALUDE_norma_cards_l1028_102856

theorem norma_cards (initial_cards : ℕ) (lost_fraction : ℚ) (remaining_cards : ℕ) : 
  initial_cards = 88 → 
  lost_fraction = 3/4 → 
  remaining_cards = initial_cards - (initial_cards * lost_fraction).floor → 
  remaining_cards = 22 := by
sorry

end NUMINAMATH_CALUDE_norma_cards_l1028_102856


namespace NUMINAMATH_CALUDE_fraction_power_product_l1028_102843

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by sorry

end NUMINAMATH_CALUDE_fraction_power_product_l1028_102843


namespace NUMINAMATH_CALUDE_hexagon_interior_exterior_angle_sum_l1028_102833

theorem hexagon_interior_exterior_angle_sum : 
  ∃! n : ℕ, n > 2 ∧ (n - 2) * 180 = 2 * 360 := by sorry

#check hexagon_interior_exterior_angle_sum

end NUMINAMATH_CALUDE_hexagon_interior_exterior_angle_sum_l1028_102833


namespace NUMINAMATH_CALUDE_inequality_holds_l1028_102812

theorem inequality_holds (a : ℝ) : (∀ x > 1, x^2 + a*x - 6 > 0) ↔ a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_l1028_102812


namespace NUMINAMATH_CALUDE_max_area_rectangle_l1028_102815

/-- The maximum area of a rectangle with perimeter 40 is 100 -/
theorem max_area_rectangle (p : ℝ) (h : p = 40) : 
  (∃ (a : ℝ), a > 0 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = p / 2 → x * y ≤ a) ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = p / 2 ∧ x * y = 100) :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l1028_102815


namespace NUMINAMATH_CALUDE_prism_on_sphere_surface_area_l1028_102878

/-- A right prism with all vertices on a sphere -/
structure PrismOnSphere where
  /-- The height of the prism -/
  height : ℝ
  /-- The volume of the prism -/
  volume : ℝ
  /-- The surface area of the sphere -/
  sphereSurfaceArea : ℝ

/-- Theorem: If a right prism with all vertices on a sphere has height 4 and volume 64,
    then the surface area of the sphere is 48π -/
theorem prism_on_sphere_surface_area (p : PrismOnSphere) 
    (h_height : p.height = 4)
    (h_volume : p.volume = 64) :
    p.sphereSurfaceArea = 48 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_prism_on_sphere_surface_area_l1028_102878


namespace NUMINAMATH_CALUDE_h_range_proof_range_sum_l1028_102883

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

theorem h_range_proof :
  ∀ x : ℝ, h x > 0 ∧
  (∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, |x| > N → h x < ε) ∧
  (∀ x : ℝ, h x ≤ h 0) →
  Set.range h = Set.Ioo 0 1 := by
sorry

theorem range_sum :
  Set.range h = Set.Ioo 0 1 →
  0 + 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_h_range_proof_range_sum_l1028_102883


namespace NUMINAMATH_CALUDE_amount_after_two_years_l1028_102865

theorem amount_after_two_years 
  (initial_amount : ℝ) 
  (annual_increase_rate : ℝ) 
  (years : ℕ) :
  initial_amount = 32000 →
  annual_increase_rate = 1 / 8 →
  years = 2 →
  initial_amount * (1 + annual_increase_rate) ^ years = 40500 := by
sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l1028_102865


namespace NUMINAMATH_CALUDE_range_of_a_l1028_102850

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x > 0, x + 1/x > a
def q (a : ℝ) : Prop := ∃ x, x^2 - 2*a*x + 1 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(¬(q a))) → 
  (¬(p a ∧ q a)) → 
  ((q a) → (a ≤ -1 ∨ a ≥ 1)) →
  ((¬(p a)) → a ≥ 2) →
  a ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1028_102850


namespace NUMINAMATH_CALUDE_probability_neither_orange_nor_white_l1028_102855

theorem probability_neither_orange_nor_white (orange black white : ℕ) 
  (h_orange : orange = 8) (h_black : black = 7) (h_white : white = 6) :
  (black : ℚ) / (orange + black + white) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_probability_neither_orange_nor_white_l1028_102855


namespace NUMINAMATH_CALUDE_parabola_translation_l1028_102804

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk 2 0 1
  let translated := translate original (-1) (-3)
  translated = Parabola.mk 2 4 (-2) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l1028_102804


namespace NUMINAMATH_CALUDE_reflection_property_l1028_102854

/-- A reflection in ℝ² -/
structure Reflection where
  /-- The function that performs the reflection -/
  apply : ℝ × ℝ → ℝ × ℝ

/-- Theorem stating that a reflection mapping (3, 2) to (1, 6) will map (2, -1) to (-2/5, -11/5) -/
theorem reflection_property (r : Reflection) 
  (h : r.apply (3, 2) = (1, 6)) :
  r.apply (2, -1) = (-2/5, -11/5) := by
  sorry

end NUMINAMATH_CALUDE_reflection_property_l1028_102854


namespace NUMINAMATH_CALUDE_inequality_implies_a_bound_l1028_102826

theorem inequality_implies_a_bound (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → x * y ≤ a * x^2 + 2 * y^2) → 
  a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_bound_l1028_102826


namespace NUMINAMATH_CALUDE_y_coordinate_of_P_l1028_102893

/-- The y-coordinate of point P given specific conditions -/
theorem y_coordinate_of_P (A B C D P : ℝ × ℝ) : 
  A = (-4, 0) →
  B = (-3, 2) →
  C = (3, 2) →
  D = (4, 0) →
  dist P A + dist P D = 10 →
  dist P B + dist P C = 10 →
  P.2 = 6/7 := by
  sorry

#check y_coordinate_of_P

end NUMINAMATH_CALUDE_y_coordinate_of_P_l1028_102893


namespace NUMINAMATH_CALUDE_sum_123_even_descending_l1028_102876

/-- The sum of the first n even natural numbers in descending order -/
def sumEvenDescending (n : ℕ) : ℕ :=
  n * (2 * n + 2) / 2

/-- Theorem: The sum of the first 123 even natural numbers in descending order is 15252 -/
theorem sum_123_even_descending : sumEvenDescending 123 = 15252 := by
  sorry

end NUMINAMATH_CALUDE_sum_123_even_descending_l1028_102876


namespace NUMINAMATH_CALUDE_weight_of_four_parts_l1028_102895

theorem weight_of_four_parts (total_weight : ℚ) (num_parts : ℕ) (parts_of_interest : ℕ) : 
  total_weight = 2 →
  num_parts = 9 →
  parts_of_interest = 4 →
  (parts_of_interest : ℚ) * (total_weight / num_parts) = 8/9 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_four_parts_l1028_102895


namespace NUMINAMATH_CALUDE_book_pages_count_l1028_102884

/-- Calculates the total number of pages in a book given the number of pages read, left to read, and skipped. -/
def totalPages (pagesRead : ℕ) (pagesLeft : ℕ) (pagesSkipped : ℕ) : ℕ :=
  pagesRead + pagesLeft + pagesSkipped

/-- Proves that for the given numbers of pages read, left to read, and skipped, the total number of pages in the book is 372. -/
theorem book_pages_count : totalPages 125 231 16 = 372 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l1028_102884


namespace NUMINAMATH_CALUDE_inequality_proof_l1028_102860

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq : a + d = b + c)
  (abs_ineq : |a - d| < |b - c|) :
  a * d > b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1028_102860


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1028_102840

theorem rational_equation_solution (x : ℝ) : 
  x ≠ 3 → x ≠ -3 → (x / (x + 3) + 6 / (x^2 - 9) = 1 / (x - 3)) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1028_102840


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_l1028_102814

/-- Three lines in a plane -/
structure PlaneLine where
  dir : ℝ × ℝ  -- Direction vector

/-- Perpendicular lines -/
def perpendicular (l1 l2 : PlaneLine) : Prop :=
  l1.dir.1 * l2.dir.1 + l1.dir.2 * l2.dir.2 = 0

/-- Parallel lines -/
def parallel (l1 l2 : PlaneLine) : Prop :=
  l1.dir.1 * l2.dir.2 = l1.dir.2 * l2.dir.1

theorem perpendicular_implies_parallel (a b c : PlaneLine) :
  perpendicular a b → perpendicular b c → parallel a c := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_l1028_102814


namespace NUMINAMATH_CALUDE_largest_common_term_under_1000_l1028_102887

/-- The first arithmetic progression -/
def seq1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression -/
def seq2 (m : ℕ) : ℕ := 7 + 11 * m

/-- A common term of both sequences -/
def commonTerm (n m : ℕ) : Prop := seq1 n = seq2 m

/-- The largest common term less than 1000 -/
theorem largest_common_term_under_1000 :
  ∃ (n m : ℕ), commonTerm n m ∧ seq1 n = 974 ∧ 
  (∀ (k l : ℕ), commonTerm k l → seq1 k < 1000 → seq1 k ≤ 974) :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_under_1000_l1028_102887


namespace NUMINAMATH_CALUDE_theater_ticket_cost_l1028_102846

/-- Proves that the cost of an adult ticket is $7.56 given the conditions of the theater ticket sales. -/
theorem theater_ticket_cost (total_tickets : ℕ) (total_receipts : ℚ) (adult_tickets : ℕ) (child_ticket_cost : ℚ) :
  total_tickets = 130 →
  total_receipts = 840 →
  adult_tickets = 90 →
  child_ticket_cost = 4 →
  ∃ adult_ticket_cost : ℚ,
    adult_ticket_cost * adult_tickets + child_ticket_cost * (total_tickets - adult_tickets) = total_receipts ∧
    adult_ticket_cost = 756 / 100 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_cost_l1028_102846


namespace NUMINAMATH_CALUDE_radius_of_touching_sphere_l1028_102880

/-- A regular quadrilateral pyramid with an inscribed sphere and a touching sphere -/
structure PyramidWithSpheres where
  -- Base side length
  a : ℝ
  -- Lateral edge length
  b : ℝ
  -- Radius of inscribed sphere Q₁
  r₁ : ℝ
  -- Radius of touching sphere Q₂
  r₂ : ℝ
  -- Condition: The pyramid is regular quadrilateral
  regular : a > 0 ∧ b > 0
  -- Condition: Q₁ is inscribed in the pyramid
  q₁_inscribed : r₁ > 0
  -- Condition: Q₂ touches Q₁ and all lateral faces
  q₂_touches : r₂ > 0

/-- Theorem stating the radius of Q₂ in the given pyramid configuration -/
theorem radius_of_touching_sphere (p : PyramidWithSpheres) 
  (h₁ : p.a = 6) 
  (h₂ : p.b = 5) : 
  p.r₂ = 3 * Real.sqrt 7 / 49 := by
  sorry


end NUMINAMATH_CALUDE_radius_of_touching_sphere_l1028_102880


namespace NUMINAMATH_CALUDE_max_third_altitude_l1028_102853

/-- A scalene triangle with specific altitude properties -/
structure ScaleneTriangle where
  /-- The length of the first altitude -/
  altitude1 : ℝ
  /-- The length of the second altitude -/
  altitude2 : ℝ
  /-- The length of the third altitude -/
  altitude3 : ℝ
  /-- Condition that the triangle is scalene -/
  scalene : altitude1 ≠ altitude2 ∧ altitude2 ≠ altitude3 ∧ altitude3 ≠ altitude1
  /-- Condition that two altitudes are 6 and 18 -/
  specific_altitudes : (altitude1 = 6 ∧ altitude2 = 18) ∨ (altitude1 = 18 ∧ altitude2 = 6) ∨
                       (altitude1 = 6 ∧ altitude3 = 18) ∨ (altitude1 = 18 ∧ altitude3 = 6) ∨
                       (altitude2 = 6 ∧ altitude3 = 18) ∨ (altitude2 = 18 ∧ altitude3 = 6)
  /-- Condition that the third altitude is an integer -/
  integer_altitude : ∃ n : ℤ, (altitude3 : ℝ) = n ∨ (altitude2 : ℝ) = n ∨ (altitude1 : ℝ) = n

/-- The maximum possible integer length of the third altitude is 8 -/
theorem max_third_altitude (t : ScaleneTriangle) : 
  ∃ (max_altitude : ℕ), max_altitude = 8 ∧ 
  ∀ (n : ℕ), (n : ℝ) = t.altitude1 ∨ (n : ℝ) = t.altitude2 ∨ (n : ℝ) = t.altitude3 → n ≤ max_altitude :=
sorry

end NUMINAMATH_CALUDE_max_third_altitude_l1028_102853


namespace NUMINAMATH_CALUDE_participation_schemes_count_l1028_102805

/-- The number of students to choose from -/
def totalStudents : Nat := 4

/-- The number of students to be selected -/
def selectedStudents : Nat := 3

/-- The number of subjects -/
def subjects : Nat := 3

/-- Represents that student A must participate -/
def studentAMustParticipate : Prop := True

/-- The total number of different participation schemes -/
def participationSchemes : Nat := 18

theorem participation_schemes_count :
  studentAMustParticipate →
  participationSchemes = (Nat.choose (totalStudents - 1) (selectedStudents - 1)) * (Nat.factorial selectedStudents) :=
by sorry

end NUMINAMATH_CALUDE_participation_schemes_count_l1028_102805


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1028_102866

/-- Given a quadratic function ax² + bx + c that can be expressed as 5(x - 5)² - 3,
    prove that when 4ax² + 4bx + 4c is expressed as n(x - h)² + k, h = 5. -/
theorem quadratic_transformation (a b c : ℝ) : 
  (∃ x, ax^2 + b*x + c = 5*(x - 5)^2 - 3) → 
  (∃ n k, ∀ x, 4*a*x^2 + 4*b*x + 4*c = n*(x - 5)^2 + k) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1028_102866


namespace NUMINAMATH_CALUDE_negation_of_existence_sqrt_leq_x_minus_one_negation_l1028_102879

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) :=
by sorry

theorem sqrt_leq_x_minus_one_negation :
  (¬ ∃ x > 0, Real.sqrt x ≤ x - 1) ↔ (∀ x > 0, Real.sqrt x > x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_sqrt_leq_x_minus_one_negation_l1028_102879


namespace NUMINAMATH_CALUDE_short_walk_probability_l1028_102861

/-- The number of gates in the airport --/
def numGates : ℕ := 20

/-- The distance between adjacent gates in feet --/
def gateDistance : ℕ := 50

/-- The maximum distance Dave can walk in feet --/
def maxWalkDistance : ℕ := 200

/-- The probability of selecting two different gates that are at most maxWalkDistance apart --/
def probabilityShortWalk : ℚ := 67 / 190

theorem short_walk_probability :
  (numGates : ℚ) * (numGates - 1) * probabilityShortWalk =
    (2 * 4) +  -- Gates at extreme ends
    (6 * 5) +  -- Gates 2 to 4 and 17 to 19
    (12 * 8)   -- Gates 5 to 16
  ∧ maxWalkDistance / gateDistance = 4 := by sorry

end NUMINAMATH_CALUDE_short_walk_probability_l1028_102861


namespace NUMINAMATH_CALUDE_jennys_pen_cost_l1028_102802

/-- Proves that the cost of each pen is $1.50 given the conditions of Jenny's purchase --/
theorem jennys_pen_cost 
  (print_cost : ℚ) 
  (copies : ℕ) 
  (pages : ℕ) 
  (num_pens : ℕ) 
  (payment : ℚ) 
  (change : ℚ)
  (h1 : print_cost = 1 / 10)
  (h2 : copies = 7)
  (h3 : pages = 25)
  (h4 : num_pens = 7)
  (h5 : payment = 40)
  (h6 : change = 12) :
  (payment - change - print_cost * copies * pages) / num_pens = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_jennys_pen_cost_l1028_102802


namespace NUMINAMATH_CALUDE_move_point_on_number_line_l1028_102871

theorem move_point_on_number_line (start : ℤ) (movement : ℤ) (result : ℤ) :
  start = -2 →
  movement = 3 →
  result = start + movement →
  result = 1 := by
  sorry

end NUMINAMATH_CALUDE_move_point_on_number_line_l1028_102871


namespace NUMINAMATH_CALUDE_range_of_m_l1028_102834

def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ a^2 = m ∧ b^2 = 2

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem range_of_m :
  ∀ m : ℝ, (¬p m ∧ q m) → (1 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1028_102834


namespace NUMINAMATH_CALUDE_sum_of_largest_odd_factors_l1028_102892

/-- The largest odd factor of a natural number -/
def largest_odd_factor (n : ℕ) : ℕ := sorry

/-- The sum of the first n terms of the sequence of largest odd factors -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem: The sum of the first 2^2016 - 1 terms of the sequence
    of largest odd factors is equal to (4^2016 - 1) / 3 -/
theorem sum_of_largest_odd_factors :
  S (2^2016 - 1) = (4^2016 - 1) / 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_largest_odd_factors_l1028_102892


namespace NUMINAMATH_CALUDE_lattice_triangle_area_bound_l1028_102891

-- Define a lattice point
def LatticePoint := ℤ × ℤ

-- Define a lattice triangle
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

-- Function to count interior lattice points
def countInteriorLatticePoints (t : LatticeTriangle) : ℕ := sorry

-- Function to calculate the area of a triangle
def triangleArea (t : LatticeTriangle) : ℚ := sorry

-- Theorem statement
theorem lattice_triangle_area_bound 
  (t : LatticeTriangle) 
  (h : countInteriorLatticePoints t = 1) : 
  triangleArea t ≤ 9/2 := by sorry

end NUMINAMATH_CALUDE_lattice_triangle_area_bound_l1028_102891


namespace NUMINAMATH_CALUDE_altitude_to_longest_side_l1028_102845

theorem altitude_to_longest_side (a b c h : ℝ) : 
  a = 8 → b = 15 → c = 17 → 
  a^2 + b^2 = c^2 → 
  h * c = 2 * (1/2 * a * b) → 
  h = 120/17 := by
sorry

end NUMINAMATH_CALUDE_altitude_to_longest_side_l1028_102845


namespace NUMINAMATH_CALUDE_oxford_high_school_classes_l1028_102817

/-- Represents the structure of Oxford High School -/
structure OxfordHighSchool where
  teachers : ℕ
  principal : ℕ
  students_per_class : ℕ
  total_people : ℕ

/-- Calculates the number of classes in Oxford High School -/
def number_of_classes (school : OxfordHighSchool) : ℕ :=
  let total_students := school.total_people - school.teachers - school.principal
  total_students / school.students_per_class

/-- Theorem stating that Oxford High School has 15 classes -/
theorem oxford_high_school_classes :
  let school : OxfordHighSchool := {
    teachers := 48,
    principal := 1,
    students_per_class := 20,
    total_people := 349
  }
  number_of_classes school = 15 := by
  sorry


end NUMINAMATH_CALUDE_oxford_high_school_classes_l1028_102817


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l1028_102837

/-- The equation represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c d e f : ℝ), 
    (a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0) ∧
    (∀ x y : ℝ, x^2 - 36*y^2 - 12*x + y + 64 = 0 ↔ 
      a*(x - c)^2 + b*(y - d)^2 + e*(x - c) + f*(y - d) = 1) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l1028_102837


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l1028_102810

theorem simplify_sqrt_product : Real.sqrt 18 * Real.sqrt 72 = 12 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l1028_102810


namespace NUMINAMATH_CALUDE_basketball_team_lineups_l1028_102886

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid lineups for the basketball team -/
def validLineups : ℕ :=
  choose 15 7 - choose 13 5

theorem basketball_team_lineups :
  validLineups = 5148 := by sorry

end NUMINAMATH_CALUDE_basketball_team_lineups_l1028_102886


namespace NUMINAMATH_CALUDE_expression_evaluation_l1028_102885

theorem expression_evaluation : -3^2 + (-12) * |-(1/2)| - 6 / (-1) = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1028_102885


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1028_102857

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b : ℝ, b > a ∧ a > 0 → a * (b + 1) > a^2) ∧ 
  (∃ a b : ℝ, a * (b + 1) > a^2 ∧ ¬(b > a ∧ a > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1028_102857


namespace NUMINAMATH_CALUDE_square_side_length_l1028_102842

/-- Given a square with perimeter 36 cm, prove that the side length is 9 cm -/
theorem square_side_length (perimeter : ℝ) (is_square : Bool) : 
  is_square ∧ perimeter = 36 → (perimeter / 4 : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1028_102842


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l1028_102800

def g (x : ℝ) := (x - 3)^2 + 6

theorem smallest_c_for_inverse :
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l1028_102800


namespace NUMINAMATH_CALUDE_band_members_minimum_l1028_102806

theorem band_members_minimum (n : ℕ) : n = 165 ↔ 
  n > 0 ∧ 
  n % 6 = 3 ∧ 
  n % 8 = 5 ∧ 
  n % 9 = 7 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 3 → m % 8 = 5 → m % 9 = 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_band_members_minimum_l1028_102806


namespace NUMINAMATH_CALUDE_flag_arrangement_remainder_l1028_102875

/-- The number of ways to arrange flags on two poles -/
def M : ℕ := sorry

/-- The total number of flags -/
def total_flags : ℕ := 24

/-- The number of blue flags -/
def blue_flags : ℕ := 14

/-- The number of red flags -/
def red_flags : ℕ := 10

/-- Each flagpole has at least one flag -/
axiom at_least_one_flag : M > 0

/-- Each sequence starts with a blue flag -/
axiom starts_with_blue : True

/-- No two red flags on either pole are adjacent -/
axiom no_adjacent_red : True

theorem flag_arrangement_remainder :
  M % 1000 = 1 := by sorry

end NUMINAMATH_CALUDE_flag_arrangement_remainder_l1028_102875


namespace NUMINAMATH_CALUDE_g_range_and_density_l1028_102828

noncomputable def g (a b c : ℝ) : ℝ := a / (a + b) + b / (b + c) + c / (c + a)

theorem g_range_and_density :
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 1 < g a b c ∧ g a b c < 2) ∧
  (∀ ε : ℝ, ε > 0 → 
    (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ |g a b c - 1| < ε) ∧
    (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ |g a b c - 2| < ε)) :=
by sorry

end NUMINAMATH_CALUDE_g_range_and_density_l1028_102828


namespace NUMINAMATH_CALUDE_problem_solution_l1028_102825

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 7)
  (h_eq2 : y + 1 / x = 20) :
  z + 1 / y = 29 / 139 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1028_102825


namespace NUMINAMATH_CALUDE_proportion_solution_l1028_102848

theorem proportion_solution (x : ℝ) (h : (0.75 : ℝ) / x = 5 / 8) : x = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1028_102848


namespace NUMINAMATH_CALUDE_unknown_number_value_l1028_102881

theorem unknown_number_value : ∃ x : ℝ, 5 + 2 * (8 - x) = 15 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l1028_102881


namespace NUMINAMATH_CALUDE_increasing_function_sum_implication_l1028_102868

theorem increasing_function_sum_implication (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y : ℝ, x < y → f x < f y) :
  f a + f b > f (-a) + f (-b) → a + b > 0 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_sum_implication_l1028_102868


namespace NUMINAMATH_CALUDE_paige_mp3_songs_l1028_102862

theorem paige_mp3_songs (initial : ℕ) (deleted : ℕ) (added : ℕ) : 
  initial = 11 → deleted = 9 → added = 8 → initial - deleted + added = 10 := by
  sorry

end NUMINAMATH_CALUDE_paige_mp3_songs_l1028_102862


namespace NUMINAMATH_CALUDE_hyperbola_trajectory_l1028_102844

/-- The trajectory of point P satisfying |PF₂| - |PF₁| = 4, where F₁(-4,0) and F₂(4,0) -/
theorem hyperbola_trajectory (x y : ℝ) : 
  let f₁ : ℝ × ℝ := (-4, 0)
  let f₂ : ℝ × ℝ := (4, 0)
  let p : ℝ × ℝ := (x, y)
  Real.sqrt ((x - 4)^2 + y^2) - Real.sqrt ((x + 4)^2 + y^2) = 4 →
  x^2 / 4 - y^2 / 12 = 1 ∧ x ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_trajectory_l1028_102844


namespace NUMINAMATH_CALUDE_modular_product_equivalence_l1028_102859

theorem modular_product_equivalence (n : ℕ) : 
  (507 * 873) % 77 = n ∧ 0 ≤ n ∧ n < 77 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_modular_product_equivalence_l1028_102859
