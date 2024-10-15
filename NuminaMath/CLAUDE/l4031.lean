import Mathlib

namespace NUMINAMATH_CALUDE_fraction_sum_equality_l4031_403101

theorem fraction_sum_equality : (2 / 10 : ℚ) + (7 / 100 : ℚ) + (3 / 1000 : ℚ) + (8 / 10000 : ℚ) = 2738 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l4031_403101


namespace NUMINAMATH_CALUDE_distance_between_three_points_l4031_403105

-- Define a line
structure Line where
  -- Add any necessary properties for a line

-- Define a point on a line
structure Point (l : Line) where
  -- Add any necessary properties for a point on a line

-- Define the distance between two points on a line
def distance (l : Line) (p q : Point l) : ℝ :=
  sorry

-- Theorem statement
theorem distance_between_three_points (l : Line) (A B C : Point l) :
  distance l A B = 5 ∧ distance l B C = 3 →
  distance l A C = 8 ∨ distance l A C = 2 :=
sorry

end NUMINAMATH_CALUDE_distance_between_three_points_l4031_403105


namespace NUMINAMATH_CALUDE_smallest_root_of_quadratic_l4031_403179

theorem smallest_root_of_quadratic (x : ℝ) :
  (12 * x^2 - 50 * x + 48 = 0) → (x ≥ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_quadratic_l4031_403179


namespace NUMINAMATH_CALUDE_min_third_side_right_triangle_l4031_403176

theorem min_third_side_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a = 8 ∧ b = 15) ∨ (a = 8 ∧ c = 15) ∨ (b = 8 ∧ c = 15) →
  a^2 + b^2 = c^2 →
  min a (min b c) ≥ Real.sqrt 161 :=
by sorry

end NUMINAMATH_CALUDE_min_third_side_right_triangle_l4031_403176


namespace NUMINAMATH_CALUDE_f_properties_l4031_403162

noncomputable def f (x : ℝ) : ℝ := (Real.sin x - Real.cos x) * Real.sin (2 * x) / Real.sin x

theorem f_properties :
  (∀ x : ℝ, f x ≠ 0 → x ∉ {y | ∃ k : ℤ, y = k * Real.pi}) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 2 + k * Real.pi → f x ≥ 0) ∧
  (∃ m : ℝ, m > 0 ∧ m = 3 * Real.pi / 8 ∧ ∀ x : ℝ, f (x + m) = f (-x + m)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4031_403162


namespace NUMINAMATH_CALUDE_vowel_word_count_l4031_403132

/-- The number of times vowels A and E appear -/
def vowel_count_ae : ℕ := 6

/-- The number of times vowels I, O, and U appear -/
def vowel_count_iou : ℕ := 5

/-- The length of the words to be formed -/
def word_length : ℕ := 6

/-- The total number of vowel choices for each position -/
def total_choices : ℕ := 2 * vowel_count_ae + 3 * vowel_count_iou

/-- Theorem stating the number of possible six-letter words -/
theorem vowel_word_count : (total_choices ^ word_length : ℕ) = 531441 := by
  sorry

end NUMINAMATH_CALUDE_vowel_word_count_l4031_403132


namespace NUMINAMATH_CALUDE_solution_product_l4031_403152

theorem solution_product (p q : ℝ) : 
  (p - 3) * (3 * p + 8) = p^2 - 5*p + 6 →
  (q - 3) * (3 * q + 8) = q^2 - 5*q + 6 →
  (p + 4) * (q + 4) = 7 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l4031_403152


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4031_403143

theorem complex_equation_solution (z : ℂ) (h : z * (3 - I) = 1 - I) : z = 2/5 - 1/5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4031_403143


namespace NUMINAMATH_CALUDE_largest_x_value_l4031_403135

theorem largest_x_value (x : ℝ) : 
  (((15 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2) → x ≤ 1) ∧
  (∃ x : ℝ, (15 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2 ∧ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l4031_403135


namespace NUMINAMATH_CALUDE_line_segment_ratio_l4031_403197

/-- Given points A, B, C, D, and E on a line in that order, prove that AC:DE = 5:3 -/
theorem line_segment_ratio (A B C D E : ℝ) : 
  (B - A = 3) → 
  (C - B = 7) → 
  (D - C = 4) → 
  (E - A = 20) → 
  (A < B) → (B < C) → (C < D) → (D < E) →
  (C - A) / (E - D) = 5 / 3 := by
  sorry

#check line_segment_ratio

end NUMINAMATH_CALUDE_line_segment_ratio_l4031_403197


namespace NUMINAMATH_CALUDE_calculation_problems_l4031_403168

theorem calculation_problems :
  ((-2 : ℤ) + 5 - abs (-8) + (-5) = -10) ∧
  ((-2 : ℤ)^2 * 5 - (-2)^3 / 4 = 22) := by
  sorry

end NUMINAMATH_CALUDE_calculation_problems_l4031_403168


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l4031_403114

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop :=
  parabola A.1 A.2

-- Define the dot product condition
def dot_product_condition (A : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  let F := focus
  (A.1 - O.1) * (F.1 - A.1) + (A.2 - O.2) * (F.2 - A.2) = -4

-- Theorem statement
theorem parabola_point_coordinates :
  ∀ A : ℝ × ℝ,
  point_on_parabola A →
  dot_product_condition A →
  (A = (1, 2) ∨ A = (1, -2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l4031_403114


namespace NUMINAMATH_CALUDE_area_of_triangle_formed_by_segment_l4031_403147

structure Rectangle where
  width : ℝ
  height : ℝ

structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

def Point := ℝ × ℝ

def Segment (p1 p2 : Point) := {p : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (t * p1.1 + (1 - t) * p2.1, t * p1.2 + (1 - t) * p2.2)}

theorem area_of_triangle_formed_by_segment (rect : Rectangle) (tri : IsoscelesTriangle) 
  (h1 : rect.width = 15 ∧ rect.height = 10)
  (h2 : tri.base = 10 ∧ tri.height = 10)
  (h3 : (15, 0) = (rect.width, 0))
  (h4 : Segment (0, rect.height) (20, 10) ∩ Segment (15, 0) (25, 0) = {(15, 10)}) :
  (1 / 2) * rect.width * rect.height = 75 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_formed_by_segment_l4031_403147


namespace NUMINAMATH_CALUDE_calculate_expression_l4031_403191

theorem calculate_expression : -2⁻¹ * (-8) - Real.sqrt 9 - abs (-4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4031_403191


namespace NUMINAMATH_CALUDE_roots_power_set_difference_l4031_403163

/-- The roots of the polynomial (x^101 - 1) / (x - 1) -/
def roots : Fin 100 → ℂ := sorry

/-- The set S of powers of roots -/
def S : Set ℂ := sorry

/-- The maximum number of unique values in S -/
def M : ℕ := sorry

/-- The minimum number of unique values in S -/
def N : ℕ := sorry

/-- The difference between the maximum and minimum number of unique values in S is 99 -/
theorem roots_power_set_difference : M - N = 99 := by sorry

end NUMINAMATH_CALUDE_roots_power_set_difference_l4031_403163


namespace NUMINAMATH_CALUDE_factorization_x4_minus_5x2_plus_4_l4031_403185

theorem factorization_x4_minus_5x2_plus_4 (x : ℝ) :
  x^4 - 5*x^2 + 4 = (x + 1)*(x - 1)*(x + 2)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_5x2_plus_4_l4031_403185


namespace NUMINAMATH_CALUDE_chicken_chick_difference_l4031_403178

theorem chicken_chick_difference (total : ℕ) (chicks : ℕ) : 
  total = 821 → chicks = 267 → total - chicks - chicks = 287 := by
  sorry

end NUMINAMATH_CALUDE_chicken_chick_difference_l4031_403178


namespace NUMINAMATH_CALUDE_article_cost_price_l4031_403141

theorem article_cost_price (C : ℝ) (S : ℝ) : 
  S = 1.05 * C →
  S - 3 = 1.1 * (0.95 * C) →
  C = 600 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l4031_403141


namespace NUMINAMATH_CALUDE_semester_weeks_calculation_l4031_403104

/-- The number of weeks in a semester before midterms -/
def weeks_in_semester : ℕ := 6

/-- The number of hours Annie spends on extracurriculars per week -/
def hours_per_week : ℕ := 13

/-- The number of weeks Annie takes off sick -/
def sick_weeks : ℕ := 2

/-- The total number of hours Annie spends on extracurriculars before midterms -/
def total_hours : ℕ := 52

theorem semester_weeks_calculation :
  weeks_in_semester * hours_per_week - sick_weeks * hours_per_week = total_hours := by
  sorry

end NUMINAMATH_CALUDE_semester_weeks_calculation_l4031_403104


namespace NUMINAMATH_CALUDE_sum_of_digits_divisible_by_13_l4031_403128

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_divisible_by_13 (n : ℕ) : 
  ∃ k ∈ Finset.range 79, 13 ∣ sum_of_digits (n + k) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_divisible_by_13_l4031_403128


namespace NUMINAMATH_CALUDE_square_difference_l4031_403120

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 64) 
  (h2 : x * y = 15) : 
  (x - y)^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l4031_403120


namespace NUMINAMATH_CALUDE_small_cuboid_height_l4031_403100

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem small_cuboid_height
  (large : CuboidDimensions)
  (small_length : ℝ)
  (small_width : ℝ)
  (num_small_cuboids : ℕ)
  (h_large : large = { length := 16, width := 10, height := 12 })
  (h_small_length : small_length = 5)
  (h_small_width : small_width = 4)
  (h_num_small : num_small_cuboids = 32) :
  ∃ (small_height : ℝ),
    cuboidVolume large = num_small_cuboids * (small_length * small_width * small_height) ∧
    small_height = 3 := by
  sorry


end NUMINAMATH_CALUDE_small_cuboid_height_l4031_403100


namespace NUMINAMATH_CALUDE_linear_system_solution_l4031_403121

theorem linear_system_solution (m : ℚ) :
  let x : ℚ → ℚ := λ m => 6 * m + 1
  let y : ℚ → ℚ := λ m => -10 * m - 1
  (∀ m, x m + y m = -4 * m ∧ 2 * x m + y m = 2 * m + 1) ∧
  (x (1/2) - y (1/2) = 10) :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l4031_403121


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l4031_403110

theorem greatest_q_minus_r : ∃ (q r : ℕ+), 
  (1025 = 23 * q + r) ∧ 
  (∀ (q' r' : ℕ+), 1025 = 23 * q' + r' → q' - r' ≤ q - r) ∧
  (q - r = 27) := by
sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l4031_403110


namespace NUMINAMATH_CALUDE_order_relation_l4031_403122

noncomputable def a (e : ℝ) : ℝ := 5 * Real.log (2^e)
noncomputable def b (e : ℝ) : ℝ := 2 * Real.log (5^e)
def c : ℝ := 10

theorem order_relation (e : ℝ) (h : e > 0) : c > a e ∧ a e > b e := by
  sorry

end NUMINAMATH_CALUDE_order_relation_l4031_403122


namespace NUMINAMATH_CALUDE_simplify_expression_l4031_403119

theorem simplify_expression (x : ℝ) : (5 - 4 * x) - (2 + 5 * x) = 3 - 9 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4031_403119


namespace NUMINAMATH_CALUDE_mashed_potatoes_suggestion_l4031_403123

theorem mashed_potatoes_suggestion (bacon_count : ℕ) (difference : ℕ) : 
  bacon_count = 394 → 
  difference = 63 → 
  bacon_count + difference = 457 :=
by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_suggestion_l4031_403123


namespace NUMINAMATH_CALUDE_solve_for_z_l4031_403117

theorem solve_for_z (x y z : ℚ) : x = 11 → y = -8 → 2 * x - 3 * z = 5 * y → z = 62 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_z_l4031_403117


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l4031_403196

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) + a n

theorem fibonacci_like_sequence
  (a : ℕ → ℕ)
  (h_increasing : ∀ n : ℕ, a n < a (n + 1))
  (h_property : sequence_property a)
  (h_a7 : a 7 = 120) :
  a 8 = 194 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l4031_403196


namespace NUMINAMATH_CALUDE_square_field_area_l4031_403116

theorem square_field_area (wire_length : ℝ) (wire_turns : ℕ) (field_side : ℝ) : 
  wire_length = (4 * field_side * wire_turns) → 
  wire_length = 15840 → 
  wire_turns = 15 → 
  field_side * field_side = 69696 := by
sorry

end NUMINAMATH_CALUDE_square_field_area_l4031_403116


namespace NUMINAMATH_CALUDE_unique_integer_with_properties_l4031_403139

theorem unique_integer_with_properties : ∃! n : ℕ+, 
  (∃ k : ℕ, n = 18 * k) ∧ 
  (30 < Real.sqrt n.val) ∧ 
  (Real.sqrt n.val < 30.5) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_with_properties_l4031_403139


namespace NUMINAMATH_CALUDE_weight_loss_in_april_l4031_403175

/-- Given Michael's weight loss plan:
  * total_weight: Total weight Michael wants to lose
  * march_loss: Weight lost in March
  * may_loss: Weight to lose in May
  * april_loss: Weight lost in April

  This theorem proves that the weight lost in April is equal to
  the total weight minus the weight lost in March and the weight to lose in May. -/
theorem weight_loss_in_april 
  (total_weight march_loss may_loss april_loss : ℕ) : 
  april_loss = total_weight - march_loss - may_loss := by
  sorry

#check weight_loss_in_april

end NUMINAMATH_CALUDE_weight_loss_in_april_l4031_403175


namespace NUMINAMATH_CALUDE_tan_150_degrees_l4031_403165

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l4031_403165


namespace NUMINAMATH_CALUDE_scientific_notation_of_11580000_l4031_403161

theorem scientific_notation_of_11580000 :
  (11580000 : ℝ) = 1.158 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_11580000_l4031_403161


namespace NUMINAMATH_CALUDE_min_value_of_expression_l4031_403142

theorem min_value_of_expression (a : ℝ) (h : 0 ≤ a ∧ a < 4) :
  ∃ m : ℝ, m = 1 ∧ ∀ x : ℝ, 0 ≤ x ∧ x < 4 → m ≤ |x - 2| + |3 - x| :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l4031_403142


namespace NUMINAMATH_CALUDE_robin_gum_count_l4031_403174

theorem robin_gum_count (initial_gum : Real) (additional_gum : Real) : 
  initial_gum = 18.5 → additional_gum = 44.25 → initial_gum + additional_gum = 62.75 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l4031_403174


namespace NUMINAMATH_CALUDE_max_product_sum_2004_l4031_403130

theorem max_product_sum_2004 :
  (∃ (a b : ℤ), a + b = 2004 ∧ a * b = 1004004) ∧
  (∀ (x y : ℤ), x + y = 2004 → x * y ≤ 1004004) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2004_l4031_403130


namespace NUMINAMATH_CALUDE_carls_weight_l4031_403164

theorem carls_weight (al ben carl ed : ℕ) 
  (h1 : al = ben + 25)
  (h2 : ben + 16 = carl)
  (h3 : ed = 146)
  (h4 : al = ed + 38) :
  carl = 175 := by
  sorry

end NUMINAMATH_CALUDE_carls_weight_l4031_403164


namespace NUMINAMATH_CALUDE_exists_unique_box_l4031_403144

/-- Represents a rectangular box with square base -/
structure Box where
  x : ℝ  -- side length of square base
  h : ℝ  -- height of box

/-- Calculates the surface area of the box -/
def surfaceArea (b : Box) : ℝ := 2 * b.x^2 + 4 * b.x * b.h

/-- Calculates the volume of the box -/
def volume (b : Box) : ℝ := b.x^2 * b.h

/-- Theorem stating the existence of a box meeting the given conditions -/
theorem exists_unique_box :
  ∃! b : Box,
    b.h = 2 * b.x + 2 ∧
    surfaceArea b ≥ 150 ∧
    volume b = 100 ∧
    b.x > 0 ∧
    b.h > 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_unique_box_l4031_403144


namespace NUMINAMATH_CALUDE_prob_red_ball_l4031_403194

/-- The probability of drawing a red ball from a bag containing 1 red ball and 2 yellow balls is 1/3. -/
theorem prob_red_ball (num_red : ℕ) (num_yellow : ℕ) (h1 : num_red = 1) (h2 : num_yellow = 2) :
  (num_red : ℚ) / (num_red + num_yellow) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_ball_l4031_403194


namespace NUMINAMATH_CALUDE_division_problem_l4031_403107

theorem division_problem (n : ℕ) : 
  let first_part : ℕ := 19
  let second_part : ℕ := 36 - first_part
  n * first_part + 3 * second_part = 203 → n = 8 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l4031_403107


namespace NUMINAMATH_CALUDE_parabola_vertex_l4031_403140

/-- The vertex of the parabola y = -2(x-2)^2 - 5 is at the point (2, -5) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -2 * (x - 2)^2 - 5 → (2, -5) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l4031_403140


namespace NUMINAMATH_CALUDE_doghouse_area_l4031_403177

/-- The area outside a regular hexagon that can be reached by a tethered point -/
theorem doghouse_area (side_length : ℝ) (rope_length : ℝ) (area : ℝ) :
  side_length = 2 →
  rope_length = 4 →
  area = 12 * Real.pi →
  area = (rope_length^2 * Real.pi * (2/3) + 2 * (side_length^2 * Real.pi * (1/6))) :=
by sorry

end NUMINAMATH_CALUDE_doghouse_area_l4031_403177


namespace NUMINAMATH_CALUDE_smallest_three_digit_square_ends_identical_l4031_403187

/-- A function that returns true if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

/-- A function that returns true if the square of a number ends with three identical non-zero digits -/
def squareEndsWithThreeIdenticalNonZeroDigits (n : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧ n^2 % 1000 = 111 * d

/-- Theorem stating that 462 is the smallest three-digit number whose square ends with three identical non-zero digits -/
theorem smallest_three_digit_square_ends_identical : 
  (isThreeDigit 462 ∧ 
   squareEndsWithThreeIdenticalNonZeroDigits 462 ∧ 
   ∀ n : ℕ, isThreeDigit n → squareEndsWithThreeIdenticalNonZeroDigits n → 462 ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_square_ends_identical_l4031_403187


namespace NUMINAMATH_CALUDE_quadratic_factorization_l4031_403180

theorem quadratic_factorization (k : ℝ) : 
  (∀ x, x^2 - 8*x + 3 = 0 ↔ (x - 4)^2 = k) → k = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l4031_403180


namespace NUMINAMATH_CALUDE_anne_solo_cleaning_time_l4031_403148

/-- Represents the time it takes Anne to clean the house alone -/
def anne_solo_time : ℝ := 12

/-- Represents Bruce's cleaning rate in houses per hour -/
noncomputable def bruce_rate : ℝ := sorry

/-- Represents Anne's cleaning rate in houses per hour -/
noncomputable def anne_rate : ℝ := sorry

/-- Bruce and Anne can clean the house in 4 hours together -/
axiom together_time : bruce_rate + anne_rate = 1 / 4

/-- If Anne's speed were doubled, they could clean the house in 3 hours -/
axiom double_anne_time : bruce_rate + 2 * anne_rate = 1 / 3

theorem anne_solo_cleaning_time : 
  1 / anne_rate = anne_solo_time :=
sorry

end NUMINAMATH_CALUDE_anne_solo_cleaning_time_l4031_403148


namespace NUMINAMATH_CALUDE_sin_239_deg_l4031_403136

theorem sin_239_deg (a : ℝ) (h : Real.cos (31 * π / 180) = a) : 
  Real.sin (239 * π / 180) = -a := by
  sorry

end NUMINAMATH_CALUDE_sin_239_deg_l4031_403136


namespace NUMINAMATH_CALUDE_special_sequence_existence_l4031_403198

theorem special_sequence_existence : ∃ (a : ℕ → ℕ),
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, ∃ y, a n + 1 = y^2) ∧
  (∀ n, ∃ x, 3 * a n + 1 = x^2) ∧
  (∀ n, ∃ z, a n * a (n + 1) = z^2) :=
sorry

end NUMINAMATH_CALUDE_special_sequence_existence_l4031_403198


namespace NUMINAMATH_CALUDE_suit_price_increase_l4031_403103

/-- Proves that the percentage increase in the price of a suit is 30% -/
theorem suit_price_increase (original_price : ℝ) (final_price : ℝ) : 
  original_price = 200 →
  final_price = 182 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 30 ∧
    final_price = original_price * (1 + increase_percentage / 100) * 0.7 :=
by sorry

end NUMINAMATH_CALUDE_suit_price_increase_l4031_403103


namespace NUMINAMATH_CALUDE_smallest_n_for_fraction_inequality_l4031_403156

theorem smallest_n_for_fraction_inequality : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℤ), 0 < m → m < 2001 → 
    ∃ (k : ℤ), (m : ℚ) / 2001 < (k : ℚ) / n ∧ (k : ℚ) / n < ((m + 1) : ℚ) / 2002) ∧
  (∀ (n' : ℕ), 0 < n' → n' < n → 
    ∃ (m : ℤ), 0 < m ∧ m < 2001 ∧
      ∀ (k : ℤ), ¬((m : ℚ) / 2001 < (k : ℚ) / n' ∧ (k : ℚ) / n' < ((m + 1) : ℚ) / 2002)) ∧
  n = 4003 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_fraction_inequality_l4031_403156


namespace NUMINAMATH_CALUDE_quad_pair_sum_l4031_403145

/-- Two distinct quadratic polynomials with specific properties -/
structure QuadraticPair where
  f : ℝ → ℝ
  g : ℝ → ℝ
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  hf : f = fun x ↦ x^2 + p*x + q
  hg : g = fun x ↦ x^2 + r*x + s
  distinct : f ≠ g
  vertex_root : g (-p/2) = 0 ∧ f (-r/2) = 0
  same_min : ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₁, f x₁ = m) ∧ (∀ x, g x ≥ m) ∧ (∃ x₂, g x₂ = m)
  intersection : f 50 = -200 ∧ g 50 = -200

/-- The sum of coefficients p and r is -200 -/
theorem quad_pair_sum (qp : QuadraticPair) : qp.p + qp.r = -200 := by
  sorry

end NUMINAMATH_CALUDE_quad_pair_sum_l4031_403145


namespace NUMINAMATH_CALUDE_min_max_sum_l4031_403131

theorem min_max_sum (p q r s t u : ℕ+) 
  (sum_eq : p + q + r + s + t + u = 2023) : 
  810 ≤ max (p + q) (max (q + r) (max (r + s) (max (s + t) (t + u)))) := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_l4031_403131


namespace NUMINAMATH_CALUDE_min_mobots_correct_l4031_403138

/-- Represents a lawn as a grid with dimensions m and n -/
structure Lawn where
  m : ℕ
  n : ℕ

/-- Represents a mobot that can mow a lawn -/
inductive Mobot
  | east  : Mobot  -- Mobot moving east
  | north : Mobot  -- Mobot moving north

/-- Function to calculate the minimum number of mobots required -/
def minMobotsRequired (lawn : Lawn) : ℕ := min lawn.m lawn.n

/-- Theorem stating that minMobotsRequired gives the correct minimum number of mobots -/
theorem min_mobots_correct (lawn : Lawn) :
  ∀ (mobots : List Mobot), (∀ row col, row < lawn.m ∧ col < lawn.n → 
    ∃ mobot ∈ mobots, (mobot = Mobot.east ∧ ∃ r, r ≤ row) ∨ 
                       (mobot = Mobot.north ∧ ∃ c, c ≤ col)) →
  mobots.length ≥ minMobotsRequired lawn :=
sorry

end NUMINAMATH_CALUDE_min_mobots_correct_l4031_403138


namespace NUMINAMATH_CALUDE_fifth_term_is_81_l4031_403102

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n
  sum_1_3 : a 1 + a 3 = 10
  sum_2_4 : a 2 + a 4 = -30

/-- The fifth term of the geometric sequence is 81 -/
theorem fifth_term_is_81 (seq : GeometricSequence) : seq.a 5 = 81 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_81_l4031_403102


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l4031_403159

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  -- The ellipse equation in the form (x²/a²) + (y²/b²) = 1
  a : ℝ
  b : ℝ
  -- Center at origin
  center_origin : True
  -- Foci on coordinate axis
  foci_on_axis : True
  -- Line y = x + 1 intersects the ellipse
  intersects_line : True
  -- OP ⊥ OQ where P and Q are intersection points
  op_perp_oq : True
  -- |PQ| = √10/2
  pq_length : True

/-- The theorem stating the possible equations of the special ellipse -/
theorem special_ellipse_equation (e : SpecialEllipse) :
  (∀ x y, x^2 + 3*y^2 = 2) ∨ (∀ x y, 3*x^2 + y^2 = 2) :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l4031_403159


namespace NUMINAMATH_CALUDE_rectangle_length_calculation_l4031_403167

theorem rectangle_length_calculation (rectangle_width square_side : ℝ) :
  rectangle_width = 300 ∧ 
  square_side = 700 ∧ 
  (4 * square_side) = 2 * (2 * (rectangle_width + rectangle_length)) →
  rectangle_length = 400 :=
by
  sorry

#check rectangle_length_calculation

end NUMINAMATH_CALUDE_rectangle_length_calculation_l4031_403167


namespace NUMINAMATH_CALUDE_jacks_card_collection_l4031_403173

theorem jacks_card_collection :
  ∀ (football_cards baseball_cards total_cards : ℕ),
    baseball_cards = 3 * football_cards + 5 →
    baseball_cards = 95 →
    total_cards = baseball_cards + football_cards →
    total_cards = 125 := by
  sorry

end NUMINAMATH_CALUDE_jacks_card_collection_l4031_403173


namespace NUMINAMATH_CALUDE_range_of_x_l4031_403133

-- Define the function f
def f (x a : ℝ) := |x - 4| + |x - a|

-- State the theorem
theorem range_of_x (a : ℝ) (h1 : a > 1) 
  (h2 : ∃ (m : ℝ), ∀ x, f x a ≥ m ∧ ∃ y, f y a = m) 
  (h3 : (Classical.choose h2) = 3) :
  ∀ x, f x a ≤ 5 → 3 ≤ x ∧ x ≤ 8 := by
  sorry

#check range_of_x

end NUMINAMATH_CALUDE_range_of_x_l4031_403133


namespace NUMINAMATH_CALUDE_constant_dot_product_l4031_403129

-- Define an equilateral triangle ABC with side length 2
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2

-- Define a point P on side BC
def PointOnBC (B C P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • B + t • C

-- Vector dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

-- Vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

-- Vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2)

theorem constant_dot_product
  (A B C P : ℝ × ℝ)
  (h1 : Triangle A B C)
  (h2 : PointOnBC B C P) :
  dot_product (vec_sub P A) (vec_add (vec_sub B A) (vec_sub C A)) = 6 :=
sorry

end NUMINAMATH_CALUDE_constant_dot_product_l4031_403129


namespace NUMINAMATH_CALUDE_initial_number_proof_l4031_403149

theorem initial_number_proof (N : ℕ) : 
  (∃ k : ℕ, N - 7 = 15 * k) ∧ 
  (∀ m : ℕ, m < 7 → ¬∃ j : ℕ, N - m = 15 * j) → 
  N = 22 := by
sorry

end NUMINAMATH_CALUDE_initial_number_proof_l4031_403149


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l4031_403182

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the line
def Line (k m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + m}

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the vector from P to Q
def vector (P Q : ℝ × ℝ) : ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2)

-- Define the squared magnitude of a vector
def magnitude_squared (v : ℝ × ℝ) : ℝ :=
  v.1^2 + v.2^2

theorem ellipse_line_intersection :
  ∃ (k m : ℝ), 
    let C := Ellipse 2 (Real.sqrt 3)
    let l := Line k m
    let P := (2, 1)
    let M := (1, 3/2)
    (∀ x y, (x, y) ∈ C → (x^2 / 4) + (y^2 / 3) = 1) ∧ 
    (1, 3/2) ∈ C ∧
    (2, 1) ∈ l ∧
    (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧ A ≠ B ∧
      dot_product (vector P A) (vector P B) = magnitude_squared (vector P M)) ∧
    k = 1/2 ∧ m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l4031_403182


namespace NUMINAMATH_CALUDE_min_value_sqrt_and_reciprocal_l4031_403111

theorem min_value_sqrt_and_reciprocal (x : ℝ) (h : x > 0) :
  4 * Real.sqrt x + 4 / x ≥ 8 ∧ ∃ y > 0, 4 * Real.sqrt y + 4 / y = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_and_reciprocal_l4031_403111


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l4031_403124

-- Define the lines and conditions
def l1 (x y m : ℝ) : Prop := x + y - 3*m = 0
def l2 (x y m : ℝ) : Prop := 2*x - y + 2*m - 1 = 0
def y_intercept_l1 : ℝ := 3

-- Define the theorem
theorem intersection_and_perpendicular_line :
  ∃ (x y : ℝ),
    (∀ m : ℝ, l1 x y m ∧ l2 x y m) ∧
    x = 2/3 ∧ y = 7/3 ∧
    ∃ (k : ℝ), 3*x + 6*y + k = 0 ∧
    ∀ (x' y' : ℝ), l2 x' y' 1 → (x' - 2/3) + 2*(y' - 7/3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l4031_403124


namespace NUMINAMATH_CALUDE_voting_difference_l4031_403186

/-- Represents the voting results for a company policy -/
structure VotingResults where
  total_employees : ℕ
  initial_for : ℕ
  initial_against : ℕ
  second_for : ℕ
  second_against : ℕ

/-- Conditions for the voting scenario -/
def voting_conditions (v : VotingResults) : Prop :=
  v.total_employees = 450 ∧
  v.initial_for + v.initial_against = v.total_employees ∧
  v.second_for + v.second_against = v.total_employees ∧
  v.initial_against > v.initial_for ∧
  v.second_for > v.second_against ∧
  (v.second_for - v.second_against) = 3 * (v.initial_against - v.initial_for) ∧
  v.second_for = (10 * v.initial_against) / 9

theorem voting_difference (v : VotingResults) 
  (h : voting_conditions v) : v.second_for - v.initial_for = 52 := by
  sorry

end NUMINAMATH_CALUDE_voting_difference_l4031_403186


namespace NUMINAMATH_CALUDE_tangent_line_at_point_p_l4031_403189

/-- The circle equation -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + m*y = 0

/-- Point P is on the circle -/
def point_on_circle (m : ℝ) : Prop :=
  circle_equation 1 1 m

/-- The tangent line equation -/
def tangent_line_equation (x y : ℝ) : Prop :=
  x - 2*y + 1 = 0

/-- Theorem: The equation of the tangent line at point P(1,1) on the given circle is x - 2y + 1 = 0 -/
theorem tangent_line_at_point_p :
  ∃ m : ℝ, point_on_circle m →
  ∀ x y : ℝ, (x = 1 ∧ y = 1) →
  tangent_line_equation x y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_p_l4031_403189


namespace NUMINAMATH_CALUDE_original_number_is_perfect_square_l4031_403155

theorem original_number_is_perfect_square :
  ∃ (n : ℕ), n^2 = 1296 ∧ ∃ (m : ℕ), (1296 + 148) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_perfect_square_l4031_403155


namespace NUMINAMATH_CALUDE_sum_of_fractions_l4031_403118

theorem sum_of_fractions : 
  (1 / (2 * 3 * 4 : ℚ)) + (1 / (3 * 4 * 5 : ℚ)) + (1 / (4 * 5 * 6 : ℚ)) + 
  (1 / (5 * 6 * 7 : ℚ)) + (1 / (6 * 7 * 8 : ℚ)) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l4031_403118


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l4031_403195

theorem pencil_pen_cost (x y : ℚ) : 
  (7 * x + 6 * y = 46.8) → 
  (3 * x + 5 * y = 32.2) → 
  (x = 2.4 ∧ y = 5) := by
sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l4031_403195


namespace NUMINAMATH_CALUDE_expression_lower_bound_l4031_403188

theorem expression_lower_bound (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b + c) + 1 / (b + c) + 1 / (c + a)) ≥ 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_lower_bound_l4031_403188


namespace NUMINAMATH_CALUDE_product_decreasing_inequality_l4031_403181

theorem product_decreasing_inequality 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h_deriv : ∀ x, (deriv f x) * g x + f x * (deriv g x) < 0) 
  {a b x : ℝ} 
  (h_interval : a < x ∧ x < b) : 
  f x * g x > f b * g b :=
sorry

end NUMINAMATH_CALUDE_product_decreasing_inequality_l4031_403181


namespace NUMINAMATH_CALUDE_scientific_notation_of_248000_l4031_403171

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_248000 :
  toScientificNotation 248000 = ScientificNotation.mk 2.48 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_248000_l4031_403171


namespace NUMINAMATH_CALUDE_cone_surface_area_and_volume_l4031_403137

/-- Represents a cone with given dimensions and properties -/
structure Cone where
  height : ℝ
  lateral_to_total_ratio : ℝ

/-- Calculates the surface area of the cone -/
def surface_area (c : Cone) : ℝ := sorry

/-- Calculates the volume of the cone -/
def volume (c : Cone) : ℝ := sorry

/-- Theorem stating the surface area and volume of a specific cone -/
theorem cone_surface_area_and_volume :
  let c := Cone.mk 96 (25/32)
  (surface_area c = 3584 * Real.pi) ∧ (volume c = 25088 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_and_volume_l4031_403137


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l4031_403150

-- Problem 1
theorem problem_1 (a b : ℤ) (h1 : a = 6) (h2 : b = -1) :
  2*a + 3*b - 2*a*b - a - 4*b - a*b = 25 := by sorry

-- Problem 2
theorem problem_2 (m n : ℤ) (h1 : |m| = 3) (h2 : |n| = 2) (h3 : m < n) :
  m^2 + 2*m*n + n^2 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l4031_403150


namespace NUMINAMATH_CALUDE_regression_line_not_necessarily_through_sample_point_l4031_403160

/-- Given a set of sample data points and a regression line, 
    prove that the line doesn't necessarily pass through any sample point. -/
theorem regression_line_not_necessarily_through_sample_point 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (a b : ℝ) : 
  ¬ (∀ (ε : ℝ), ε > 0 → 
    ∃ (i : Fin n), |y i - (b * x i + a)| < ε) :=
sorry

end NUMINAMATH_CALUDE_regression_line_not_necessarily_through_sample_point_l4031_403160


namespace NUMINAMATH_CALUDE_special_circle_equation_l4031_403170

/-- A circle with center (a, b) and radius r satisfying specific conditions -/
structure SpecialCircle where
  a : ℝ
  b : ℝ
  r : ℝ
  center_on_line : a - 3 * b = 0
  tangent_to_y_axis : r = |a|
  chord_length : ((a - b) ^ 2 / 2) + 7 = r ^ 2

/-- The equation of a circle given its center (a, b) and radius r -/
def circle_equation (c : SpecialCircle) (x y : ℝ) : Prop :=
  (x - c.a) ^ 2 + (y - c.b) ^ 2 = c.r ^ 2

/-- The main theorem stating that a SpecialCircle satisfies one of two specific equations -/
theorem special_circle_equation (c : SpecialCircle) :
  (∀ x y, circle_equation c x y ↔ (x - 3) ^ 2 + (y - 1) ^ 2 = 9) ∨
  (∀ x y, circle_equation c x y ↔ (x + 3) ^ 2 + (y + 1) ^ 2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_special_circle_equation_l4031_403170


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l4031_403184

theorem largest_triangle_perimeter :
  ∀ (x : ℕ),
  (7 + 8 > x) →
  (7 + x > 8) →
  (8 + x > 7) →
  (∀ y : ℕ, (7 + 8 > y) → (7 + y > 8) → (8 + y > 7) → y ≤ x) →
  7 + 8 + x = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l4031_403184


namespace NUMINAMATH_CALUDE_ratio_problem_l4031_403108

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l4031_403108


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l4031_403151

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  (360 : ℝ) / exterior_angle = n →
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l4031_403151


namespace NUMINAMATH_CALUDE_firefighter_net_sag_l4031_403158

/-- The net sag for a person jumping onto a firefighter rescue net -/
def net_sag (m₁ m₂ h₁ h₂ x₁ : ℝ) (x₂ : ℝ) : Prop :=
  m₁ > 0 ∧ m₂ > 0 ∧ h₁ > 0 ∧ h₂ > 0 ∧ x₁ > 0 ∧ x₂ > 0 ∧
  28 * x₂^2 - x₂ - 29 = 0

theorem firefighter_net_sag (m₁ m₂ h₁ h₂ x₁ : ℝ) (hm₁ : m₁ = 78.75) (hm₂ : m₂ = 45)
    (hh₁ : h₁ = 15) (hh₂ : h₂ = 29) (hx₁ : x₁ = 1) :
  ∃ x₂, net_sag m₁ m₂ h₁ h₂ x₁ x₂ :=
by sorry

end NUMINAMATH_CALUDE_firefighter_net_sag_l4031_403158


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l4031_403109

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 4 + 35 / 99 ∧ x = 431 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l4031_403109


namespace NUMINAMATH_CALUDE_solution_values_l4031_403134

def A : Set ℝ := {-1, 1}

def B (a b : ℝ) : Set ℝ := {x | x^2 - 2*a*x + b = 0}

theorem solution_values (a b : ℝ) : 
  B a b ≠ ∅ → A ∪ B a b = A → 
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) :=
by sorry

end NUMINAMATH_CALUDE_solution_values_l4031_403134


namespace NUMINAMATH_CALUDE_units_digit_of_T_is_zero_l4031_403157

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def T : ℕ := (List.range 99).foldl (λ acc i => acc + factorial (i + 3)) 0

theorem units_digit_of_T_is_zero : T % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_T_is_zero_l4031_403157


namespace NUMINAMATH_CALUDE_trilandia_sentinel_sites_l4031_403199

/-- Represents a triangular city with streets and sentinel sites. -/
structure TriangularCity where
  side_length : ℕ
  num_streets : ℕ

/-- Calculates the minimum number of sentinel sites required for a given triangular city. -/
def min_sentinel_sites (city : TriangularCity) : ℕ :=
  3 * (city.side_length / 2) - 1

/-- Theorem stating the minimum number of sentinel sites for Trilandia. -/
theorem trilandia_sentinel_sites :
  let trilandia : TriangularCity := ⟨2012, 6036⟩
  min_sentinel_sites trilandia = 3017 := by
  sorry

#eval min_sentinel_sites ⟨2012, 6036⟩

end NUMINAMATH_CALUDE_trilandia_sentinel_sites_l4031_403199


namespace NUMINAMATH_CALUDE_macaroons_remaining_l4031_403193

def remaining_macaroons (initial_red : ℕ) (initial_green : ℕ) (eaten_green : ℕ) : ℕ :=
  let eaten_red := 2 * eaten_green
  (initial_red - eaten_red) + (initial_green - eaten_green)

theorem macaroons_remaining :
  remaining_macaroons 50 40 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_macaroons_remaining_l4031_403193


namespace NUMINAMATH_CALUDE_system_solution_l4031_403166

-- Define the system of equations
def equation1 (x y a b : ℝ) : Prop := x / (x - a) + y / (y - b) = 2
def equation2 (x y a b : ℝ) : Prop := a * x + b * y = 2 * a * b

-- State the theorem
theorem system_solution (a b : ℝ) (ha : a ≠ b) (hab : a + b ≠ 0) :
  ∃ x y : ℝ, equation1 x y a b ∧ equation2 x y a b ∧ x = 2 * a * b / (a + b) ∧ y = 2 * a * b / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4031_403166


namespace NUMINAMATH_CALUDE_price_theorem_min_bottles_theorem_l4031_403106

-- Define variables
def peanut_oil_price : ℝ := sorry
def corn_oil_price : ℝ := sorry
def peanut_oil_sell_price : ℝ := 60
def peanut_oil_purchased : ℕ := 50

-- Define conditions
axiom condition1 : 20 * peanut_oil_price + 30 * corn_oil_price = 2200
axiom condition2 : 30 * peanut_oil_price + 10 * corn_oil_price = 1900

-- Define theorems to prove
theorem price_theorem :
  peanut_oil_price = 50 ∧ corn_oil_price = 40 :=
sorry

theorem min_bottles_theorem :
  ∀ n : ℕ, n * peanut_oil_sell_price > peanut_oil_purchased * peanut_oil_price →
  n ≥ 42 :=
sorry

end NUMINAMATH_CALUDE_price_theorem_min_bottles_theorem_l4031_403106


namespace NUMINAMATH_CALUDE_green_shirt_percentage_l4031_403115

theorem green_shirt_percentage 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (other_count : ℕ) 
  (h1 : total_students = 900) 
  (h2 : blue_percent = 44/100) 
  (h3 : red_percent = 28/100) 
  (h4 : other_count = 162) :
  (total_students - (blue_percent * total_students + red_percent * total_students + other_count : ℚ)) / total_students = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_green_shirt_percentage_l4031_403115


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4031_403113

/-- The minimum value of 1/m + 1/n given the conditions -/
theorem min_value_reciprocal_sum (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1)
  (h_line : 2*m + 2*n = 1) (h_positive : m*n > 0) :
  1/m + 1/n ≥ 8 ∧ ∃ (m n : ℝ), 2*m + 2*n = 1 ∧ m*n > 0 ∧ 1/m + 1/n = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4031_403113


namespace NUMINAMATH_CALUDE_smallest_winning_number_l4031_403169

def bernardo_wins (N : ℕ) : Prop :=
  2 * N < 1000 ∧
  2 * N + 60 < 1000 ∧
  4 * N + 120 < 1000 ∧
  4 * N + 180 < 1000 ∧
  8 * N + 360 < 1000 ∧
  8 * N + 420 < 1000 ∧
  16 * N + 840 < 1000 ∧
  16 * N + 900 ≥ 1000

theorem smallest_winning_number :
  bernardo_wins 5 ∧ ∀ k : ℕ, k < 5 → ¬bernardo_wins k :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l4031_403169


namespace NUMINAMATH_CALUDE_immigrants_calculation_l4031_403153

/-- The number of people born in the country last year -/
def people_born : ℕ := 90171

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := 106491

/-- The number of people who immigrated to the country last year -/
def immigrants : ℕ := total_new_people - people_born

theorem immigrants_calculation :
  immigrants = 16320 := by
  sorry

end NUMINAMATH_CALUDE_immigrants_calculation_l4031_403153


namespace NUMINAMATH_CALUDE_recycling_points_per_bag_l4031_403190

/-- Calculates the points earned per bag of recycled cans. -/
def points_per_bag (total_bags : ℕ) (unrecycled_bags : ℕ) (total_points : ℕ) : ℚ :=
  total_points / total_bags

theorem recycling_points_per_bag :
  let total_bags : ℕ := 4
  let unrecycled_bags : ℕ := 2
  let total_points : ℕ := 16
  points_per_bag total_bags unrecycled_bags total_points = 4 := by
  sorry

end NUMINAMATH_CALUDE_recycling_points_per_bag_l4031_403190


namespace NUMINAMATH_CALUDE_parabola_b_value_l4031_403125

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- The y-intercept of a parabola -/
def yIntercept (p : Parabola) : ℝ := p.c

theorem parabola_b_value (p : Parabola) (h k : ℝ) :
  h > 0 ∧ k > 0 ∧
  vertex p = (h, k) ∧
  yIntercept p = -k ∧
  k = 2 * h →
  p.b = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_b_value_l4031_403125


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_of_seven_l4031_403146

theorem least_four_digit_multiple_of_seven : ∃ n : ℕ,
  n = 1001 ∧
  7 ∣ n ∧
  1000 ≤ n ∧
  n < 10000 ∧
  ∀ m : ℕ, 7 ∣ m → 1000 ≤ m → m < 10000 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_of_seven_l4031_403146


namespace NUMINAMATH_CALUDE_expand_squared_difference_product_expand_linear_factors_l4031_403172

/-- Theorem for the expansion of (2a-b)^2 * (2a+b)^2 -/
theorem expand_squared_difference_product (a b : ℝ) :
  (2*a - b)^2 * (2*a + b)^2 = 16*a^4 - 8*a^2*b^2 + b^4 := by sorry

/-- Theorem for the expansion of (3a+b-2)(3a-b+2) -/
theorem expand_linear_factors (a b : ℝ) :
  (3*a + b - 2) * (3*a - b + 2) = 9*a^2 - b^2 + 4*b - 4 := by sorry

end NUMINAMATH_CALUDE_expand_squared_difference_product_expand_linear_factors_l4031_403172


namespace NUMINAMATH_CALUDE_base_b_square_l4031_403192

theorem base_b_square (b : ℕ) : 
  (b + 5)^2 = 4*b^2 + 3*b + 6 → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_b_square_l4031_403192


namespace NUMINAMATH_CALUDE_two_circles_common_tangents_l4031_403154

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Number of common tangent lines between two circles -/
def commonTangentLines (c1 c2 : Circle) : ℕ :=
  sorry

/-- The main theorem -/
theorem two_circles_common_tangents :
  let c1 : Circle := { center := (1, 2), radius := 1 }
  let c2 : Circle := { center := (2, 5), radius := 3 }
  commonTangentLines c1 c2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_circles_common_tangents_l4031_403154


namespace NUMINAMATH_CALUDE_select_team_count_l4031_403183

/-- The number of ways to select a team of 7 people from a group of 7 boys and 9 girls, 
    with at least 3 girls in the team -/
def selectTeam (numBoys numGirls teamSize minGirls : ℕ) : ℕ :=
  (Finset.range (teamSize - minGirls + 1)).sum fun i =>
    Nat.choose numGirls (minGirls + i) * Nat.choose numBoys (teamSize - minGirls - i)

/-- Theorem stating that the number of ways to select the team is 10620 -/
theorem select_team_count : selectTeam 7 9 7 3 = 10620 := by
  sorry

end NUMINAMATH_CALUDE_select_team_count_l4031_403183


namespace NUMINAMATH_CALUDE_multiply_add_equality_l4031_403112

theorem multiply_add_equality : (-3) * 2 + 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_equality_l4031_403112


namespace NUMINAMATH_CALUDE_intersection_of_spheres_integer_points_l4031_403127

theorem intersection_of_spheres_integer_points :
  let sphere1 := {(x, y, z) : ℤ × ℤ × ℤ | x^2 + y^2 + (z - 10)^2 ≤ 25}
  let sphere2 := {(x, y, z) : ℤ × ℤ × ℤ | x^2 + y^2 + (z - 4)^2 ≤ 36}
  ∃! p : ℤ × ℤ × ℤ, p ∈ sphere1 ∩ sphere2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_of_spheres_integer_points_l4031_403127


namespace NUMINAMATH_CALUDE_cost_per_friend_is_1650_l4031_403126

/-- The cost per friend when buying erasers and pencils -/
def cost_per_friend (eraser_count : ℕ) (eraser_cost : ℕ) (pencil_count : ℕ) (pencil_cost : ℕ) (friend_count : ℕ) : ℚ :=
  ((eraser_count * eraser_cost + pencil_count * pencil_cost) : ℚ) / friend_count

/-- Theorem: The cost per friend for the given scenario is 1650 won -/
theorem cost_per_friend_is_1650 :
  cost_per_friend 5 200 7 800 4 = 1650 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_friend_is_1650_l4031_403126
