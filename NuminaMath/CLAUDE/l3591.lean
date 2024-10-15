import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_3_l3591_359142

-- Define the equations of the lines
def line1 (a x y : ℝ) : Prop := a * x + 3 * y - 1 = 0
def line2 (a x y : ℝ) : Prop := 2 * x + (a - 1) * y + 1 = 0

-- Define what it means for two lines to be parallel
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- State the theorem
theorem parallel_lines_imply_a_equals_3 (a : ℝ) :
  parallel (line1 a) (line2 a) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_3_l3591_359142


namespace NUMINAMATH_CALUDE_exists_positive_sum_G2_l3591_359193

-- Define the grid as a function from pairs of integers to real numbers
def Grid := ℤ × ℤ → ℝ

-- Define a shape as a finite set of integer pairs (representing cell positions)
def Shape := Finset (ℤ × ℤ)

-- Define the sum of numbers covered by a shape at a given position
def shapeSum (g : Grid) (s : Shape) (pos : ℤ × ℤ) : ℝ :=
  s.sum (λ (x, y) => g (x + pos.1, y + pos.2))

-- State the theorem
theorem exists_positive_sum_G2 (g : Grid) (G1 G2 : Shape) 
  (h : ∀ pos : ℤ × ℤ, shapeSum g G1 pos > 0) :
  ∃ pos : ℤ × ℤ, shapeSum g G2 pos > 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_sum_G2_l3591_359193


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l3591_359109

theorem added_number_after_doubling (original : ℕ) (added : ℕ) : 
  original = 9 → 3 * (2 * original + added) = 72 → added = 6 := by
  sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l3591_359109


namespace NUMINAMATH_CALUDE_tens_digit_of_23_to_1987_l3591_359153

theorem tens_digit_of_23_to_1987 : ∃ k : ℕ, 23^1987 ≡ 40 + k [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_to_1987_l3591_359153


namespace NUMINAMATH_CALUDE_max_daily_revenue_l3591_359147

def P (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 70
  else 0

def Q (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def dailyRevenue (t : ℕ) : ℝ := P t * Q t

theorem max_daily_revenue :
  (∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ dailyRevenue t = 1125) ∧
  (∀ t : ℕ, 0 < t ∧ t ≤ 30 → dailyRevenue t ≤ 1125) ∧
  (∀ t : ℕ, 0 < t ∧ t ≤ 30 ∧ dailyRevenue t = 1125 → t = 25) :=
by sorry

end NUMINAMATH_CALUDE_max_daily_revenue_l3591_359147


namespace NUMINAMATH_CALUDE_hulk_jump_distance_l3591_359137

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem hulk_jump_distance (n : ℕ) : 
  (∀ k < n, geometric_sequence 3 3 k ≤ 500) ∧ 
  geometric_sequence 3 3 n > 500 → n = 6 :=
sorry

end NUMINAMATH_CALUDE_hulk_jump_distance_l3591_359137


namespace NUMINAMATH_CALUDE_square_area_remainder_l3591_359154

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Square in 2D space -/
structure Square where
  center : Point
  side_length : ℝ

/-- Check if a point is on a side of a square -/
def is_on_side (p : Point) (s : Square) : Prop :=
  (abs (p.x - s.center.x) = s.side_length / 2 ∧ abs (p.y - s.center.y) ≤ s.side_length / 2) ∨
  (abs (p.y - s.center.y) = s.side_length / 2 ∧ abs (p.x - s.center.x) ≤ s.side_length / 2)

theorem square_area_remainder (A B C D : Point) (S : Square) :
  A.x = 0 ∧ A.y = 12 ∧
  B.x = 10 ∧ B.y = 9 ∧
  C.x = 8 ∧ C.y = 0 ∧
  D.x = -4 ∧ D.y = 7 ∧
  is_on_side A S ∧ is_on_side B S ∧ is_on_side C S ∧ is_on_side D S ∧
  (∀ S' : Square, is_on_side A S' ∧ is_on_side B S' ∧ is_on_side C S' ∧ is_on_side D S' → S' = S) →
  (10 * S.side_length ^ 2) % 1000 = 936 := by
  sorry

end NUMINAMATH_CALUDE_square_area_remainder_l3591_359154


namespace NUMINAMATH_CALUDE_fraction_sum_equals_three_tenths_l3591_359146

theorem fraction_sum_equals_three_tenths : 
  (1 : ℚ) / 10 + (2 : ℚ) / 20 + (3 : ℚ) / 30 = (3 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_three_tenths_l3591_359146


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l3591_359120

theorem sqrt_sum_fractions : Real.sqrt (1/8 + 1/25) = Real.sqrt 33 / (10 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l3591_359120


namespace NUMINAMATH_CALUDE_exam_class_size_l3591_359104

/-- Represents a class of students with their exam marks -/
structure ExamClass where
  total_students : ℕ
  total_marks : ℕ
  average_mark : ℚ
  excluded_students : ℕ
  excluded_average : ℚ
  remaining_average : ℚ

/-- Theorem stating the conditions and the result to be proven -/
theorem exam_class_size (c : ExamClass) 
  (h1 : c.average_mark = 80)
  (h2 : c.excluded_students = 5)
  (h3 : c.excluded_average = 40)
  (h4 : c.remaining_average = 90)
  (h5 : c.total_marks = c.total_students * c.average_mark)
  (h6 : c.total_marks - c.excluded_students * c.excluded_average = 
        (c.total_students - c.excluded_students) * c.remaining_average) :
  c.total_students = 25 := by
  sorry

end NUMINAMATH_CALUDE_exam_class_size_l3591_359104


namespace NUMINAMATH_CALUDE_circles_are_intersecting_l3591_359180

/-- Two circles are intersecting if the distance between their centers is greater than the absolute 
    difference of their radii and less than the sum of their radii. -/
def are_intersecting (r₁ r₂ d : ℝ) : Prop :=
  d > |r₁ - r₂| ∧ d < r₁ + r₂

/-- Given two circles with radii 5 and 8, whose centers are 8 units apart, 
    prove that they are intersecting. -/
theorem circles_are_intersecting : are_intersecting 5 8 8 := by
  sorry

end NUMINAMATH_CALUDE_circles_are_intersecting_l3591_359180


namespace NUMINAMATH_CALUDE_unique_solution_l3591_359138

theorem unique_solution : 
  ∀ x y : ℤ, (2*x + 5*y + 1)*(2^(Int.natAbs x) + x^2 + x + y) = 105 ↔ x = 0 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3591_359138


namespace NUMINAMATH_CALUDE_flywheel_power_l3591_359178

/-- Calculates the power of a flywheel's driving machine in horsepower -/
theorem flywheel_power (r : ℝ) (m : ℝ) (n : ℝ) (t : ℝ) : 
  r = 3 →
  m = 6000 →
  n = 800 →
  t = 3 →
  ∃ (p : ℝ), abs (p - 1431) < 1 ∧ 
  p = (m * (r * n * 2 * Real.pi / 60)^2) / (2 * t * 60 * 746) := by
  sorry

end NUMINAMATH_CALUDE_flywheel_power_l3591_359178


namespace NUMINAMATH_CALUDE_point_b_value_l3591_359103

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Represents the distance between two points on a number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

/-- Given points A and B on a number line, where A represents -2 and B is 5 units away from A,
    B must represent either -7 or 3. -/
theorem point_b_value (A B : Point) :
  A.value = -2 ∧ distance A B = 5 → B.value = -7 ∨ B.value = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_b_value_l3591_359103


namespace NUMINAMATH_CALUDE_solve_pretzel_problem_l3591_359158

def pretzel_problem (barry_pretzels : ℕ) (angie_ratio : ℕ) (shelly_ratio : ℚ) (dave_percentage : ℚ) : Prop :=
  let shelly_pretzels := barry_pretzels * shelly_ratio
  let angie_pretzels := shelly_pretzels * angie_ratio
  let dave_pretzels := (angie_pretzels + shelly_pretzels) * dave_percentage
  angie_pretzels = 18 ∧ dave_pretzels = 6

theorem solve_pretzel_problem :
  pretzel_problem 12 3 (1/2) (1/4) := by
  sorry

end NUMINAMATH_CALUDE_solve_pretzel_problem_l3591_359158


namespace NUMINAMATH_CALUDE_largest_gold_coins_l3591_359106

theorem largest_gold_coins : ∃ n : ℕ, n < 150 ∧ ∃ k : ℕ, n = 13 * k + 3 ∧ 
  ∀ m : ℕ, m < 150 → (∃ j : ℕ, m = 13 * j + 3) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_gold_coins_l3591_359106


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3591_359101

/-- Given two parallel vectors a and b, prove that the magnitude of b is 2√10 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, 3) → 
  b.1 = -2 → 
  (∃ k : ℝ, b = k • a) → 
  ‖b‖ = 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3591_359101


namespace NUMINAMATH_CALUDE_tamara_height_l3591_359143

/-- Given the heights of Tamara, Kim, and Gavin, prove Tamara's height is 95 inches -/
theorem tamara_height (kim : ℝ) : 
  let tamara := 3 * kim - 4
  let gavin := 2 * kim + 6
  (3 * kim - 4) + kim + (2 * kim + 6) = 200 → tamara = 95 := by
sorry

end NUMINAMATH_CALUDE_tamara_height_l3591_359143


namespace NUMINAMATH_CALUDE_no_right_triangle_perimeter_area_equality_l3591_359165

theorem no_right_triangle_perimeter_area_equality :
  ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (a + b + Real.sqrt (a^2 + b^2))^2 = 2 * a * b := by
  sorry

end NUMINAMATH_CALUDE_no_right_triangle_perimeter_area_equality_l3591_359165


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3591_359191

-- Define the properties of the polygon
def is_valid_polygon (n : ℕ) : Prop :=
  n > 2 ∧
  ∀ k : ℕ, k ≤ n → 100 + (k - 1) * 10 < 180

-- Theorem statement
theorem polygon_sides_count : ∃ (n : ℕ), is_valid_polygon n ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3591_359191


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3591_359149

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 ≤ Real.sin x) ↔ (∃ x : ℝ, x^2 - 2*x + 2 > Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3591_359149


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3591_359192

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) :
  k ≠ 0 →
  p ≠ 1 →
  r ≠ 1 →
  p ≠ r →
  a₂ = k * p →
  a₃ = k * p^2 →
  b₂ = k * r →
  b₃ = k * r^2 →
  a₃ - b₃ = 4 * (a₂ - b₂) →
  p + r = 4 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3591_359192


namespace NUMINAMATH_CALUDE_fencing_requirement_l3591_359119

/-- Given a rectangular field with area 60 sq. feet and one side 20 feet,
    prove that the sum of the other three sides is 26 feet. -/
theorem fencing_requirement (length width : ℝ) : 
  length * width = 60 →
  length = 20 →
  length + 2 * width = 26 := by
  sorry

end NUMINAMATH_CALUDE_fencing_requirement_l3591_359119


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l3591_359195

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  (∀ x : ℝ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l3591_359195


namespace NUMINAMATH_CALUDE_roots_equal_opposite_signs_l3591_359127

theorem roots_equal_opposite_signs (a b c m : ℝ) :
  (∃ x y : ℝ, x ≠ 0 ∧ y = -x ∧
    (x^2 - b*x) / (a*x - c) = (m - 1) / (m + 1) ∧
    (y^2 - b*y) / (a*y - c) = (m - 1) / (m + 1)) →
  m = (a - b) / (a + b) := by
sorry

end NUMINAMATH_CALUDE_roots_equal_opposite_signs_l3591_359127


namespace NUMINAMATH_CALUDE_alex_jamie_pairing_probability_l3591_359157

/-- The number of students participating in the event -/
def total_students : ℕ := 32

/-- The probability of Alex being paired with Jamie in a random pairing -/
def probability_alex_jamie : ℚ := 1 / 31

/-- Theorem stating that the probability of Alex being paired with Jamie
    in a random pairing of 32 students is 1/31 -/
theorem alex_jamie_pairing_probability :
  probability_alex_jamie = 1 / (total_students - 1) :=
sorry

end NUMINAMATH_CALUDE_alex_jamie_pairing_probability_l3591_359157


namespace NUMINAMATH_CALUDE_abc_product_is_one_l3591_359123

theorem abc_product_is_one (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : a^2 + 1/b^2 = b^2 + 1/c^2 ∧ b^2 + 1/c^2 = c^2 + 1/a^2) :
  |a * b * c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_is_one_l3591_359123


namespace NUMINAMATH_CALUDE_tape_shortage_l3591_359164

/-- The amount of tape Joy has -/
def tape_amount : ℕ := 180

/-- The width of the field -/
def field_width : ℕ := 35

/-- The length of the field -/
def field_length : ℕ := 80

/-- The perimeter of a rectangle -/
def rectangle_perimeter (length width : ℕ) : ℕ := 2 * (length + width)

theorem tape_shortage : 
  rectangle_perimeter field_length field_width = tape_amount + 50 := by
  sorry

end NUMINAMATH_CALUDE_tape_shortage_l3591_359164


namespace NUMINAMATH_CALUDE_inequality_range_l3591_359169

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| - |x + 2| ≤ a) ↔ a ∈ Set.Ici 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l3591_359169


namespace NUMINAMATH_CALUDE_geometric_subseq_implies_arithmetic_indices_l3591_359167

/-- Given a geometric sequence with common ratio q ≠ 1, if three terms form a geometric sequence,
    then their indices form an arithmetic sequence. -/
theorem geometric_subseq_implies_arithmetic_indices
  (a : ℕ → ℝ) (q : ℝ) (m n p : ℕ) (hq : q ≠ 1)
  (h_geom : ∀ k, a (k + 1) = q * a k)
  (h_subseq : (a n)^2 = a m * a p) :
  2 * n = m + p :=
sorry

end NUMINAMATH_CALUDE_geometric_subseq_implies_arithmetic_indices_l3591_359167


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3591_359135

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 - a 4 - a 8 - a 12 + a 15 = 2) →
  (a 3 + a 13 = -4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3591_359135


namespace NUMINAMATH_CALUDE_smallest_surface_area_l3591_359184

/-- Represents the dimensions of a cigarette box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  thickness : ℝ
  length_gt_width : length > width
  width_gt_thickness : width > thickness

/-- Calculates the surface area of a rectangular package -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + l * h)

/-- Represents different packaging methods for 10 boxes -/
inductive PackagingMethod
  | method1
  | method2
  | method3
  | method4

/-- Calculates the surface area for a given packaging method -/
def packaging_surface_area (method : PackagingMethod) (box : BoxDimensions) : ℝ :=
  match method with
  | .method1 => surface_area (10 * box.length) box.width box.thickness
  | .method2 => surface_area box.length (10 * box.width) box.thickness
  | .method3 => surface_area box.length box.width (10 * box.thickness)
  | .method4 => surface_area box.length (2 * box.width) (5 * box.thickness)

theorem smallest_surface_area (box : BoxDimensions) 
  (h1 : box.length = 88)
  (h2 : box.width = 58)
  (h3 : box.thickness = 22) :
  (∀ m : PackagingMethod, packaging_surface_area .method4 box ≤ packaging_surface_area m box) ∧ 
  packaging_surface_area .method4 box = 65296 := by
  sorry

#eval surface_area 88 (2 * 58) (5 * 22)

end NUMINAMATH_CALUDE_smallest_surface_area_l3591_359184


namespace NUMINAMATH_CALUDE_edward_pipe_per_bolt_l3591_359156

/-- The number of feet of pipe per bolt in Edward's plumbing job -/
def feet_per_bolt (total_pipe_length : ℕ) (washers_used : ℕ) (washers_per_bolt : ℕ) : ℚ :=
  total_pipe_length / (washers_used / washers_per_bolt)

/-- Theorem stating that Edward uses 5 feet of pipe per bolt -/
theorem edward_pipe_per_bolt :
  feet_per_bolt 40 16 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_edward_pipe_per_bolt_l3591_359156


namespace NUMINAMATH_CALUDE_simplify_fraction_l3591_359139

theorem simplify_fraction (a b : ℝ) (h1 : a = 2) (h2 : b = 3) :
  (15 * a^4 * b) / (75 * a^3 * b^2) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3591_359139


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l3591_359128

theorem smallest_x_satisfying_equation : 
  (∃ x : ℝ, x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8) ∧ 
  (∀ y : ℝ, y > 0 ∧ ⌊y^2⌋ - y * ⌊y⌋ = 8 → y ≥ 89/9) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l3591_359128


namespace NUMINAMATH_CALUDE_stating_max_regions_correct_l3591_359199

/-- 
Given two points A and B on a plane, with m lines passing through A and n lines passing through B,
this function calculates the maximum number of regions these m+n lines can divide the plane into.
-/
def max_regions (m n : ℕ) : ℕ :=
  m * n + 2 * m + 2 * n - 1

/-- 
Theorem stating that for any positive natural numbers m and n, 
the maximum number of regions formed by m+n lines (m through point A, n through point B) 
is given by the function max_regions.
-/
theorem max_regions_correct (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  max_regions m n = m * n + 2 * m + 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_stating_max_regions_correct_l3591_359199


namespace NUMINAMATH_CALUDE_max_red_socks_l3591_359188

theorem max_red_socks (r b : ℕ) : 
  let t := r + b
  (t ≤ 2023) →
  (r * (r - 1) + b * (b - 1)) / (t * (t - 1)) = 2 / 5 →
  r ≤ 990 ∧ ∃ (r' : ℕ), r' = 990 ∧ 
    ∃ (b' : ℕ), (r' + b' ≤ 2023) ∧ 
    (r' * (r' - 1) + b' * (b' - 1)) / ((r' + b') * (r' + b' - 1)) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_red_socks_l3591_359188


namespace NUMINAMATH_CALUDE_max_a_value_l3591_359114

open Real

theorem max_a_value (e : ℝ) (h_e : e = exp 1) :
  let a_max := 1/2 + log 2/2 - e
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (1/e) 2 → (a + e) * x - 1 - log x ≤ 0) →
  a ≤ a_max ∧
  ∃ x : ℝ, x ∈ Set.Icc (1/e) 2 ∧ (a_max + e) * x - 1 - log x = 0 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l3591_359114


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_64_l3591_359133

theorem sqrt_49_times_sqrt_64 : Real.sqrt (49 * Real.sqrt 64) = 14 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_64_l3591_359133


namespace NUMINAMATH_CALUDE_sodium_hydride_requirement_l3591_359113

-- Define the chemical reaction
structure ChemicalReaction where
  naH : ℚ  -- moles of Sodium hydride
  h2o : ℚ  -- moles of Water
  naOH : ℚ -- moles of Sodium hydroxide
  h2 : ℚ   -- moles of Hydrogen

-- Define the balanced equation
def balancedReaction (r : ChemicalReaction) : Prop :=
  r.naH = r.h2o ∧ r.naH = r.naOH ∧ r.naH = r.h2

-- Theorem statement
theorem sodium_hydride_requirement 
  (r : ChemicalReaction) 
  (h1 : r.naOH = 2) 
  (h2 : r.h2 = 2) 
  (h3 : r.h2o = 2) 
  (h4 : balancedReaction r) : 
  r.naH = 2 := by
  sorry

end NUMINAMATH_CALUDE_sodium_hydride_requirement_l3591_359113


namespace NUMINAMATH_CALUDE_a_2007_mod_100_l3591_359151

/-- Sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 7
  | n + 1 => 7^(a n)

/-- Theorem stating that a_2007 ≡ 43 (mod 100) -/
theorem a_2007_mod_100 : a 2006 % 100 = 43 := by
  sorry

end NUMINAMATH_CALUDE_a_2007_mod_100_l3591_359151


namespace NUMINAMATH_CALUDE_parabolas_intersection_l3591_359108

def parabola1 (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 1
def parabola2 (x : ℝ) : ℝ := -x^2 + 4 * x + 6

theorem parabolas_intersection :
  ∃ (y1 y2 : ℝ),
    (∀ x : ℝ, parabola1 x = parabola2 x ↔ x = (-1 + Real.sqrt 61) / 6 ∨ x = (-1 - Real.sqrt 61) / 6) ∧
    parabola1 ((-1 + Real.sqrt 61) / 6) = y1 ∧
    parabola1 ((-1 - Real.sqrt 61) / 6) = y2 :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l3591_359108


namespace NUMINAMATH_CALUDE_ellipse_equal_angles_l3591_359196

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the ellipse -/
def onEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Represents a chord of the ellipse -/
structure Chord where
  A : Point
  B : Point

/-- Checks if a chord passes through a given point -/
def chordThroughPoint (c : Chord) (p : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    p.x = c.A.x + t * (c.B.x - c.A.x) ∧
    p.y = c.A.y + t * (c.B.y - c.A.y)

/-- Calculates the angle between three points -/
def angle (A B C : Point) : ℝ :=
  sorry -- Angle calculation implementation

/-- The main theorem to prove -/
theorem ellipse_equal_angles (e : Ellipse) (F P : Point) :
  e.a = 2 ∧ e.b = 1 ∧
  F.x = Real.sqrt 3 ∧ F.y = 0 ∧
  P.x = 2 ∧ P.y = 0 →
  ∀ (c : Chord), onEllipse e c.A ∧ onEllipse e c.B ∧ chordThroughPoint c F →
    angle c.A P F = angle c.B P F :=
  sorry

end NUMINAMATH_CALUDE_ellipse_equal_angles_l3591_359196


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l3591_359152

/-- A coloring function for the Cartesian plane with integer coordinates. -/
def ColoringFunction := ℤ → ℤ → Fin 3

/-- Proposition that a color appears infinitely many times on infinitely many horizontal lines. -/
def InfiniteAppearance (f : ColoringFunction) (c : Fin 3) : Prop :=
  ∀ N : ℕ, ∃ y > N, ∀ M : ℕ, ∃ x > M, f x y = c

/-- Proposition that three points of different colors are not collinear. -/
def NotCollinear (f : ColoringFunction) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ : ℤ,
    f x₁ y₁ ≠ f x₂ y₂ ∧ f x₂ y₂ ≠ f x₃ y₃ ∧ f x₃ y₃ ≠ f x₁ y₁ →
    (x₁ - x₂) * (y₃ - y₂) ≠ (x₃ - x₂) * (y₁ - y₂)

/-- Theorem stating the existence of a valid coloring function. -/
theorem exists_valid_coloring : ∃ f : ColoringFunction,
  (∀ c : Fin 3, InfiniteAppearance f c) ∧ NotCollinear f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l3591_359152


namespace NUMINAMATH_CALUDE_abc_relationship_l3591_359118

theorem abc_relationship :
  let a : Real := Real.rpow 0.3 0.2
  let b : Real := Real.rpow 0.2 0.3
  let c : Real := Real.rpow 0.3 0.3
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_abc_relationship_l3591_359118


namespace NUMINAMATH_CALUDE_product_of_reals_l3591_359126

theorem product_of_reals (x y : ℝ) (sum_eq : x + y = 10) (sum_cubes_eq : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reals_l3591_359126


namespace NUMINAMATH_CALUDE_solution_count_l3591_359110

open Complex

/-- The number of complex solutions to e^z = (z + i)/(z - i) with |z| < 20 -/
def num_solutions : ℕ := 14

/-- The equation e^z = (z + i)/(z - i) -/
def equation (z : ℂ) : Prop := exp z = (z + I) / (z - I)

/-- The condition |z| < 20 -/
def magnitude_condition (z : ℂ) : Prop := abs z < 20

theorem solution_count :
  (∃ (S : Finset ℂ), S.card = num_solutions ∧
    (∀ z ∈ S, equation z ∧ magnitude_condition z) ∧
    (∀ z : ℂ, equation z ∧ magnitude_condition z → z ∈ S)) := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l3591_359110


namespace NUMINAMATH_CALUDE_thursday_coffee_consumption_l3591_359144

/-- Represents the relationship between coffee consumption, sleep, and preparation time -/
def coffee_relation (k : ℝ) (c h p : ℝ) : Prop :=
  c * (h + p) = k

theorem thursday_coffee_consumption 
  (k : ℝ)
  (c_wed h_wed p_wed : ℝ)
  (h_thu p_thu : ℝ)
  (hw : coffee_relation k c_wed h_wed p_wed)
  (wed_data : c_wed = 3 ∧ h_wed = 8 ∧ p_wed = 2)
  (thu_data : h_thu = 5 ∧ p_thu = 3) :
  ∃ c_thu : ℝ, coffee_relation k c_thu h_thu p_thu ∧ c_thu = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_thursday_coffee_consumption_l3591_359144


namespace NUMINAMATH_CALUDE_max_balls_in_cube_l3591_359124

theorem max_balls_in_cube (cube_side : ℝ) (ball_radius : ℝ) : 
  cube_side = 8 → 
  ball_radius = 1.5 → 
  ⌊(cube_side^3) / ((4/3) * π * ball_radius^3)⌋ = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_balls_in_cube_l3591_359124


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3591_359174

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of x^2 + x - 2 = 0 is 9 -/
theorem quadratic_discriminant :
  discriminant 1 1 (-2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3591_359174


namespace NUMINAMATH_CALUDE_athlete_speed_l3591_359115

/-- Given an athlete running 200 meters in 24 seconds, prove their speed is approximately 30 km/h -/
theorem athlete_speed (distance : Real) (time : Real) (h1 : distance = 200) (h2 : time = 24) :
  ∃ (speed : Real), abs (speed - 30) < 0.1 ∧ speed = (distance / 1000) / (time / 3600) := by
  sorry

end NUMINAMATH_CALUDE_athlete_speed_l3591_359115


namespace NUMINAMATH_CALUDE_max_increasing_subsequences_l3591_359150

theorem max_increasing_subsequences (A : Fin 2001 → ℕ+) :
  (Finset.univ.filter (fun i : Fin 2001 => 
    ∃ j k, i < j ∧ j < k ∧ 
    A j = A i + 1 ∧ A k = A j + 1)).card ≤ 667^3 := by
  sorry

end NUMINAMATH_CALUDE_max_increasing_subsequences_l3591_359150


namespace NUMINAMATH_CALUDE_circle_logarithm_l3591_359117

theorem circle_logarithm (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : 
  (2 * Real.log a = Real.log (a^2)) →
  (4 * Real.log b = Real.log (b^4)) →
  (2 * π * Real.log (a^2) = Real.log (b^4)) →
  Real.log b / Real.log a = π :=
by sorry

end NUMINAMATH_CALUDE_circle_logarithm_l3591_359117


namespace NUMINAMATH_CALUDE_equation_solution_l3591_359134

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (27 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 33 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3591_359134


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3591_359102

/-- Given vectors a, b, and c in ℝ², prove that if a + b is parallel to c, then x = -5 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2*x, x]
  let c : Fin 2 → ℝ := ![3, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • c) →
  x = -5 := by
sorry


end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3591_359102


namespace NUMINAMATH_CALUDE_rohan_salary_l3591_359185

/-- Rohan's monthly expenses and savings -/
structure RohanFinances where
  salary : ℝ
  food_percentage : ℝ
  rent_percentage : ℝ
  entertainment_percentage : ℝ
  conveyance_percentage : ℝ
  savings : ℝ

/-- Theorem stating Rohan's monthly salary given his expenses and savings -/
theorem rohan_salary (r : RohanFinances) 
  (h1 : r.food_percentage = 0.4)
  (h2 : r.rent_percentage = 0.2)
  (h3 : r.entertainment_percentage = 0.1)
  (h4 : r.conveyance_percentage = 0.1)
  (h5 : r.savings = 2000)
  (h6 : r.savings = r.salary * (1 - (r.food_percentage + r.rent_percentage + r.entertainment_percentage + r.conveyance_percentage))) :
  r.salary = 10000 := by
  sorry

#check rohan_salary

end NUMINAMATH_CALUDE_rohan_salary_l3591_359185


namespace NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l3591_359130

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of sides in the polygon -/
def sides : ℕ := 150

/-- Theorem: The number of diagonals in a polygon with 150 sides is 11025 -/
theorem diagonals_150_sided_polygon :
  num_diagonals sides = 11025 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l3591_359130


namespace NUMINAMATH_CALUDE_operation_problem_l3591_359136

-- Define the set of operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the operation
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_problem (diamond circ : Operation) 
  (h : (applyOp diamond 15 3) / (applyOp circ 8 2) = 3) :
  (applyOp diamond 9 4) / (applyOp circ 14 7) = 13/7 := by
  sorry

end NUMINAMATH_CALUDE_operation_problem_l3591_359136


namespace NUMINAMATH_CALUDE_maxwells_speed_l3591_359121

/-- Proves that Maxwell's walking speed is 4 km/h given the problem conditions -/
theorem maxwells_speed (total_distance : ℝ) (brads_speed : ℝ) (maxwell_time : ℝ) 
  (brad_delay : ℝ) (h1 : total_distance = 74) (h2 : brads_speed = 6) 
  (h3 : maxwell_time = 8) (h4 : brad_delay = 1) : 
  ∃ (maxwell_speed : ℝ), maxwell_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_maxwells_speed_l3591_359121


namespace NUMINAMATH_CALUDE_initial_balls_count_l3591_359170

theorem initial_balls_count (initial : ℕ) (current : ℕ) (removed : ℕ) : 
  current = 6 → removed = 2 → initial = current + removed → initial = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_balls_count_l3591_359170


namespace NUMINAMATH_CALUDE_lola_baked_eight_pies_l3591_359175

/-- Represents the number of pastries baked by Lola and Lulu -/
structure Pastries where
  lola_cupcakes : ℕ
  lola_poptarts : ℕ
  lola_pies : ℕ
  lulu_cupcakes : ℕ
  lulu_poptarts : ℕ
  lulu_pies : ℕ

/-- The total number of pastries baked by both Lola and Lulu -/
def total_pastries (p : Pastries) : ℕ :=
  p.lola_cupcakes + p.lola_poptarts + p.lola_pies +
  p.lulu_cupcakes + p.lulu_poptarts + p.lulu_pies

/-- Theorem stating that Lola baked 8 blueberry pies -/
theorem lola_baked_eight_pies (p : Pastries)
  (h1 : p.lola_cupcakes = 13)
  (h2 : p.lola_poptarts = 10)
  (h3 : p.lulu_cupcakes = 16)
  (h4 : p.lulu_poptarts = 12)
  (h5 : p.lulu_pies = 14)
  (h6 : total_pastries p = 73) :
  p.lola_pies = 8 := by
  sorry

end NUMINAMATH_CALUDE_lola_baked_eight_pies_l3591_359175


namespace NUMINAMATH_CALUDE_odd_number_characterization_l3591_359190

theorem odd_number_characterization (n : ℤ) : 
  Odd n ↔ ∃ k : ℤ, n = 2 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_odd_number_characterization_l3591_359190


namespace NUMINAMATH_CALUDE_remainder_5n_mod_3_l3591_359172

theorem remainder_5n_mod_3 (n : ℤ) (h : n % 3 = 2) : (5 * n) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_5n_mod_3_l3591_359172


namespace NUMINAMATH_CALUDE_quadratic_polynomial_identification_l3591_359111

-- Define the polynomials
def p1 (x : ℝ) : ℝ := 3^2 * x + 1
def p2 (x : ℝ) : ℝ := 3 * x^2
def p3 (x y : ℝ) : ℝ := 3 * x * y + 1
def p4 (x : ℝ) : ℝ := 3 * x - 5^2

-- Define what it means for a polynomial to be quadratic
def is_quadratic (p : ℝ → ℝ) : Prop := ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, p x = a * x^2 + b * x + c

-- Define what it means for a two-variable polynomial to be quadratic
def is_quadratic_two_var (p : ℝ → ℝ → ℝ) : Prop := 
  ∃ a b c d e f : ℝ, (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ 
  ∀ x y, p x y = a * x^2 + b * y^2 + c * x * y + d * x + e * y + f

-- State the theorem
theorem quadratic_polynomial_identification :
  ¬ is_quadratic p1 ∧
  is_quadratic p2 ∧
  is_quadratic_two_var p3 ∧
  ¬ is_quadratic p4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_identification_l3591_359111


namespace NUMINAMATH_CALUDE_octal_127_equals_binary_1010111_l3591_359166

def octal_to_decimal (octal : ℕ) : ℕ := 
  (octal % 10) + 8 * ((octal / 10) % 10) + 64 * (octal / 100)

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem octal_127_equals_binary_1010111 : 
  decimal_to_binary (octal_to_decimal 127) = [1, 0, 1, 0, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_octal_127_equals_binary_1010111_l3591_359166


namespace NUMINAMATH_CALUDE_total_dragons_is_eight_l3591_359159

/-- Represents the number of heads on a dragon -/
inductive DragonHeads
  | two
  | seven

/-- Counts the total number of dragons given the conditions of the problem -/
def count_dragons : Nat :=
  let two_headed := 6
  let seven_headed := 2
  two_headed + seven_headed

/-- The main theorem stating that the total number of dragons is 8 -/
theorem total_dragons_is_eight :
  count_dragons = 8 ∧
  ∃ (x y : Nat),
    x * 2 + y * 7 = 25 + 7 ∧  -- Total heads including the counting head
    x + y = count_dragons ∧
    x ≥ 0 ∧ y > 0 :=
by sorry

end NUMINAMATH_CALUDE_total_dragons_is_eight_l3591_359159


namespace NUMINAMATH_CALUDE_hexahedron_octahedron_volume_ratio_l3591_359140

theorem hexahedron_octahedron_volume_ratio :
  ∀ (a b : ℝ),
    a > 0 → b > 0 →
    6 * a^2 = 2 * Real.sqrt 3 * b^2 →
    (a^3) / ((Real.sqrt 2 / 3) * b^3) = 3 / Real.sqrt (6 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hexahedron_octahedron_volume_ratio_l3591_359140


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3591_359160

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 2 + 2 * Complex.I) : 
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3591_359160


namespace NUMINAMATH_CALUDE_correct_sums_l3591_359148

theorem correct_sums (total : ℕ) (h1 : total = 75) : ∃ (right : ℕ), right * 3 = total ∧ right = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_sums_l3591_359148


namespace NUMINAMATH_CALUDE_hope_students_approximation_l3591_359162

/-- Rounds a natural number to the nearest thousand -/
def roundToNearestThousand (n : ℕ) : ℕ :=
  1000 * ((n + 500) / 1000)

/-- The number of students in Hope Primary School -/
def hopeStudents : ℕ := 1996

theorem hope_students_approximation :
  roundToNearestThousand hopeStudents = 2000 := by
  sorry

end NUMINAMATH_CALUDE_hope_students_approximation_l3591_359162


namespace NUMINAMATH_CALUDE_base8_subtraction_result_l3591_359173

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Represents subtraction in base-8 --/
def base8_subtraction (a b : ℕ) : ℤ := sorry

theorem base8_subtraction_result : 
  base8_subtraction (base8_to_base10 46) (base8_to_base10 63) = -13 := by sorry

end NUMINAMATH_CALUDE_base8_subtraction_result_l3591_359173


namespace NUMINAMATH_CALUDE_coin_flip_solution_l3591_359182

def coin_flip_problem (n : ℕ) : Prop :=
  let p_tails : ℚ := 1/2
  let p_sequence : ℚ := 0.0625
  (p_tails ^ 2 * (1 - p_tails) ^ 2 = p_sequence) ∧ (n = 4)

theorem coin_flip_solution :
  ∃ n : ℕ, coin_flip_problem n :=
sorry

end NUMINAMATH_CALUDE_coin_flip_solution_l3591_359182


namespace NUMINAMATH_CALUDE_optimal_inequality_values_l3591_359112

theorem optimal_inequality_values (x : ℝ) (hx : x ∈ Set.Icc 0 1) :
  let a : ℝ := 2
  let b : ℝ := 1/4
  (∀ (a' : ℝ) (b' : ℝ), a' > 0 → b' > 0 →
    (∀ (y : ℝ), y ∈ Set.Icc 0 1 → Real.sqrt (1 - y) + Real.sqrt (1 + y) ≤ 2 - b' * y ^ a') →
    a' ≥ a) ∧
  (∀ (b' : ℝ), b' > b →
    ∃ (y : ℝ), y ∈ Set.Icc 0 1 ∧ Real.sqrt (1 - y) + Real.sqrt (1 + y) > 2 - b' * y ^ a) ∧
  Real.sqrt (1 - x) + Real.sqrt (1 + x) ≤ 2 - b * x ^ a :=
by sorry

end NUMINAMATH_CALUDE_optimal_inequality_values_l3591_359112


namespace NUMINAMATH_CALUDE_range_of_expression_l3591_359194

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (z : ℝ), z = 4*(x - 1/2)^2 + (y - 1)^2 + 4*x*y ∧ 1 ≤ z ∧ z ≤ 22 + 4*Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3591_359194


namespace NUMINAMATH_CALUDE_subtraction_value_l3591_359161

theorem subtraction_value (N x : ℝ) : 
  ((N - x) / 7 = 7) ∧ ((N - 6) / 8 = 6) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_value_l3591_359161


namespace NUMINAMATH_CALUDE_floor_product_equation_l3591_359129

theorem floor_product_equation (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 29 ↔ x ≥ 5.8 ∧ x < 6 :=
sorry

end NUMINAMATH_CALUDE_floor_product_equation_l3591_359129


namespace NUMINAMATH_CALUDE_multiplier_proof_l3591_359189

theorem multiplier_proof (number : ℝ) (difference : ℝ) (subtractor : ℝ) :
  number = 15.0 →
  difference = 40 →
  subtractor = 5 →
  ∃ (multiplier : ℝ), multiplier * number - subtractor = difference ∧ multiplier = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_proof_l3591_359189


namespace NUMINAMATH_CALUDE_payment_bill_value_l3591_359141

/-- Proves the value of a single bill used for payment given the number of items,
    cost per item, number of change bills, and value of each change bill. -/
theorem payment_bill_value
  (num_games : ℕ)
  (cost_per_game : ℕ)
  (num_change_bills : ℕ)
  (change_bill_value : ℕ)
  (h1 : num_games = 6)
  (h2 : cost_per_game = 15)
  (h3 : num_change_bills = 2)
  (h4 : change_bill_value = 5) :
  num_games * cost_per_game + num_change_bills * change_bill_value = 100 := by
  sorry

#check payment_bill_value

end NUMINAMATH_CALUDE_payment_bill_value_l3591_359141


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l3591_359131

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l3591_359131


namespace NUMINAMATH_CALUDE_david_pushups_l3591_359197

theorem david_pushups (zachary_pushups : ℕ) : 
  zachary_pushups + (zachary_pushups + 49) = 53 →
  zachary_pushups + 49 = 51 := by
  sorry

end NUMINAMATH_CALUDE_david_pushups_l3591_359197


namespace NUMINAMATH_CALUDE_other_root_of_complex_polynomial_l3591_359186

theorem other_root_of_complex_polynomial (m n : ℝ) :
  (Complex.I + 3) ^ 2 + m * (Complex.I + 3) + n = 0 →
  (3 - Complex.I) ^ 2 + m * (3 - Complex.I) + n = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_complex_polynomial_l3591_359186


namespace NUMINAMATH_CALUDE_range_of_a_l3591_359168

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (B a ⊆ Aᶜ) ↔ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3591_359168


namespace NUMINAMATH_CALUDE_combined_return_percentage_l3591_359122

theorem combined_return_percentage (investment1 investment2 return1 return2 : ℝ) :
  investment1 = 500 →
  investment2 = 1500 →
  return1 = 0.07 →
  return2 = 0.19 →
  (investment1 * return1 + investment2 * return2) / (investment1 + investment2) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_combined_return_percentage_l3591_359122


namespace NUMINAMATH_CALUDE_shredder_capacity_l3591_359179

/-- Given a paper shredder that can shred 6 pages at a time,
    and 44 shredding operations, prove that the total number
    of pages shredded is 264. -/
theorem shredder_capacity (pages_per_operation : Nat) (num_operations : Nat) :
  pages_per_operation = 6 → num_operations = 44 → pages_per_operation * num_operations = 264 := by
  sorry

end NUMINAMATH_CALUDE_shredder_capacity_l3591_359179


namespace NUMINAMATH_CALUDE_opposite_and_abs_of_sqrt3_minus2_l3591_359100

theorem opposite_and_abs_of_sqrt3_minus2 :
  (-(Real.sqrt 3 - 2) = 2 - Real.sqrt 3) ∧
  (|Real.sqrt 3 - 2| = 2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_and_abs_of_sqrt3_minus2_l3591_359100


namespace NUMINAMATH_CALUDE_sequence_sum_of_squares_l3591_359107

theorem sequence_sum_of_squares (n : ℕ) :
  ∃ y : ℤ, (1 / 4 : ℝ) * ((2 + Real.sqrt 3)^(2*n - 1) + (2 - Real.sqrt 3)^(2*n - 1)) = y^2 + (y + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_of_squares_l3591_359107


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3591_359132

-- Define the curve
def f (x : ℝ) : ℝ := 2*x - x^3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 2 - 3*x^2

-- Define the point of tangency
def x₀ : ℝ := -1
def y₀ : ℝ := f x₀

-- Define the slope of the tangent line at x₀
def m : ℝ := f' x₀

-- Statement: The equation of the tangent line is x + y + 2 = 0
theorem tangent_line_equation :
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ x + y + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3591_359132


namespace NUMINAMATH_CALUDE_meal_cost_is_45_l3591_359116

/-- The cost of a meal consisting of one pizza and three burgers -/
def meal_cost (burger_price : ℝ) : ℝ :=
  let pizza_price := 2 * burger_price
  pizza_price + 3 * burger_price

/-- Theorem: The cost of one pizza and three burgers is $45 -/
theorem meal_cost_is_45 :
  meal_cost 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_is_45_l3591_359116


namespace NUMINAMATH_CALUDE_vertical_shift_theorem_l3591_359155

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define a constant for vertical shift
variable (k : ℝ)

-- Define the shifted function
def shifted_f (x : ℝ) : ℝ := f x + k

-- Theorem: The graph of y = f(x) + k is a vertical shift of y = f(x) by k units
theorem vertical_shift_theorem :
  ∀ (x y : ℝ), y = shifted_f f k x ↔ y - k = f x :=
by sorry

end NUMINAMATH_CALUDE_vertical_shift_theorem_l3591_359155


namespace NUMINAMATH_CALUDE_decagon_ratio_l3591_359105

/-- Represents a decagon with specific properties -/
structure Decagon where
  total_area : ℝ
  squares_below : ℝ
  trapezoid_base1 : ℝ
  trapezoid_base2 : ℝ
  bisector : ℝ → ℝ → Prop

/-- Theorem stating the ratio of XQ to QY in the given decagon -/
theorem decagon_ratio (d : Decagon)
    (h_area : d.total_area = 12)
    (h_squares : d.squares_below = 2)
    (h_base1 : d.trapezoid_base1 = 3)
    (h_base2 : d.trapezoid_base2 = 6)
    (h_bisect : d.bisector (d.squares_below + (d.trapezoid_base1 + d.trapezoid_base2) / 2 * h) (d.total_area / 2))
    (x y : ℝ)
    (h_xy : x + y = 6)
    (h_bisect_xy : d.bisector x y) :
    x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_ratio_l3591_359105


namespace NUMINAMATH_CALUDE_matt_card_trade_profit_l3591_359198

/-- Represents the profit made from trading cards -/
def card_trade_profit (cards_traded : ℕ) (value_per_traded_card : ℕ) 
                      (received_cards_1 : ℕ) (value_per_received_card_1 : ℕ)
                      (received_cards_2 : ℕ) (value_per_received_card_2 : ℕ) : ℤ :=
  (received_cards_1 * value_per_received_card_1 + received_cards_2 * value_per_received_card_2) -
  (cards_traded * value_per_traded_card)

/-- The profit Matt makes from trading two $6 cards for three $2 cards and one $9 card is $3 -/
theorem matt_card_trade_profit : card_trade_profit 2 6 3 2 1 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_matt_card_trade_profit_l3591_359198


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_three_l3591_359177

theorem cube_root_sum_equals_three :
  ∃ (a b : ℝ), 
    a^3 = 9 + 4 * Real.sqrt 5 ∧ 
    b^3 = 9 - 4 * Real.sqrt 5 ∧ 
    (∃ (k : ℤ), a + b = k) → 
    a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_three_l3591_359177


namespace NUMINAMATH_CALUDE_swimmers_passing_theorem_l3591_359181

/-- Represents the number of times two swimmers pass each other in a pool --/
def swimmers_passing_count (pool_length : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of times swimmers pass each other under given conditions --/
theorem swimmers_passing_theorem :
  let pool_length : ℝ := 100
  let speed1 : ℝ := 4
  let speed2 : ℝ := 3
  let total_time : ℝ := 15 * 60  -- 15 minutes in seconds
  swimmers_passing_count pool_length speed1 speed2 total_time = 27 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_passing_theorem_l3591_359181


namespace NUMINAMATH_CALUDE_approximate_and_scientific_notation_l3591_359163

/-- Determines the place value of the last non-zero digit in a number -/
def lastNonZeroDigitPlace (n : ℕ) : ℕ := sorry

/-- Converts a natural number to scientific notation -/
def toScientificNotation (n : ℕ) : ℝ × ℤ := sorry

theorem approximate_and_scientific_notation :
  (lastNonZeroDigitPlace 24000 = 100) ∧
  (toScientificNotation 46400000 = (4.64, 7)) := by sorry

end NUMINAMATH_CALUDE_approximate_and_scientific_notation_l3591_359163


namespace NUMINAMATH_CALUDE_job_candidate_probability_l3591_359187

theorem job_candidate_probability 
  (p_excel : ℝ) 
  (p_day_shift : ℝ) 
  (h_excel : p_excel = 0.2) 
  (h_day_shift : p_day_shift = 0.7) : 
  p_excel * (1 - p_day_shift) = 0.06 := by
sorry

end NUMINAMATH_CALUDE_job_candidate_probability_l3591_359187


namespace NUMINAMATH_CALUDE_lcm_of_32_and_12_l3591_359183

theorem lcm_of_32_and_12 (n m : ℕ+) (h1 : n = 32) (h2 : m = 12) (h3 : Nat.gcd n m = 8) :
  Nat.lcm n m = 48 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_32_and_12_l3591_359183


namespace NUMINAMATH_CALUDE_new_customers_count_l3591_359176

-- Define the given conditions
def initial_customers : ℕ := 13
def final_customers : ℕ := 9
def total_left : ℕ := 8

-- Theorem to prove
theorem new_customers_count :
  ∃ (new_customers : ℕ),
    new_customers = total_left + (final_customers - (initial_customers - total_left)) :=
by
  sorry

end NUMINAMATH_CALUDE_new_customers_count_l3591_359176


namespace NUMINAMATH_CALUDE_football_player_goals_l3591_359171

theorem football_player_goals (average_increase : ℝ) (fifth_match_goals : ℕ) : 
  average_increase = 0.3 →
  fifth_match_goals = 2 →
  ∃ (initial_average : ℝ),
    initial_average * 4 + fifth_match_goals = (initial_average + average_increase) * 5 ∧
    (initial_average + average_increase) * 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_football_player_goals_l3591_359171


namespace NUMINAMATH_CALUDE_computer_price_comparison_l3591_359145

theorem computer_price_comparison (price1 : ℝ) (discount1 : ℝ) (discount2 : ℝ) (price_diff : ℝ) : 
  price1 = 950 ∧ 
  discount1 = 0.06 ∧ 
  discount2 = 0.05 ∧ 
  price_diff = 19 →
  ∃ (price2 : ℝ), 
    price2 * (1 - discount2) = price1 * (1 - discount1) + price_diff ∧ 
    price2 = 960 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_comparison_l3591_359145


namespace NUMINAMATH_CALUDE_quadratic_roots_l3591_359125

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3591_359125
