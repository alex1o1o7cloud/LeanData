import Mathlib

namespace NUMINAMATH_CALUDE_divisiblity_by_thirty_l3711_371184

theorem divisiblity_by_thirty (p : ℕ) (h_prime : Nat.Prime p) (h_geq_seven : p ≥ 7) :
  ∃ k : ℕ, p^2 - 1 = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_divisiblity_by_thirty_l3711_371184


namespace NUMINAMATH_CALUDE_water_pouring_time_l3711_371119

/-- Proves that pouring 18 gallons at a rate of 1 gallon every 20 seconds takes 6 minutes -/
theorem water_pouring_time (tank_capacity : ℕ) (pour_rate : ℚ) (remaining : ℕ) (poured : ℕ) :
  tank_capacity = 50 →
  pour_rate = 1 / 20 →
  remaining = 32 →
  poured = 18 →
  (poured : ℚ) / pour_rate / 60 = 6 :=
by sorry

end NUMINAMATH_CALUDE_water_pouring_time_l3711_371119


namespace NUMINAMATH_CALUDE_salem_poem_stanzas_l3711_371103

/-- Represents a poem with a specific structure -/
structure Poem where
  lines_per_stanza : ℕ
  words_per_line : ℕ
  total_words : ℕ

/-- Calculates the number of stanzas in a poem -/
def number_of_stanzas (p : Poem) : ℕ :=
  p.total_words / (p.lines_per_stanza * p.words_per_line)

/-- Theorem: A poem with 10 lines per stanza, 8 words per line, 
    and 1600 total words has 20 stanzas -/
theorem salem_poem_stanzas :
  let p : Poem := ⟨10, 8, 1600⟩
  number_of_stanzas p = 20 := by
  sorry

#check salem_poem_stanzas

end NUMINAMATH_CALUDE_salem_poem_stanzas_l3711_371103


namespace NUMINAMATH_CALUDE_max_sum_squares_sides_l3711_371145

/-- For any acute-angled triangle with side length a and angle α, 
    the sum of squares of the other two side lengths (b² + c²) 
    is less than or equal to a² / (2 sin²(α/2)). -/
theorem max_sum_squares_sides (a : ℝ) (α : ℝ) (h_acute : 0 < α ∧ α < π / 2) :
  ∀ b c : ℝ, 
  (0 < b ∧ 0 < c) → -- Ensure positive side lengths
  (b^2 + c^2 - 2*b*c*Real.cos α = a^2) → -- Cosine rule
  b^2 + c^2 ≤ a^2 / (2 * Real.sin (α/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squares_sides_l3711_371145


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l3711_371120

-- Define the ♡ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l3711_371120


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l3711_371111

/-- Two lines are parallel if their slopes are equal and not equal to their y-intercept ratios -/
def are_parallel (a : ℝ) : Prop :=
  a ≠ 0 ∧ a ≠ -1 ∧ (1 / a = a / 1) ∧ (1 / a ≠ (-2*a - 2) / (-a - 1))

/-- Given two lines l₁: x + ay = 2a + 2 and l₂: ax + y = a + 1 are parallel, prove that a = 1 -/
theorem parallel_lines_imply_a_equals_one :
  ∀ a : ℝ, are_parallel a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l3711_371111


namespace NUMINAMATH_CALUDE_triangle_ratio_proof_l3711_371194

theorem triangle_ratio_proof (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Angle A is 60°
  A = Real.pi / 3 →
  -- Side b is 1
  b = 1 →
  -- Area of triangle is √3/2
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  -- The sum of angles in a triangle is π
  A + B + C = Real.pi →
  -- Triangle inequality
  a < b + c ∧ b < a + c ∧ c < a + b →
  -- Prove that the expression equals 2
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_proof_l3711_371194


namespace NUMINAMATH_CALUDE_second_number_value_l3711_371158

theorem second_number_value (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 3 / 4)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0) :
  b = 40 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l3711_371158


namespace NUMINAMATH_CALUDE_days_of_sending_roses_l3711_371106

def roses_per_day : ℕ := 24  -- 2 dozen roses per day
def total_roses : ℕ := 168   -- total number of roses sent

theorem days_of_sending_roses : 
  total_roses / roses_per_day = 7 :=
sorry

end NUMINAMATH_CALUDE_days_of_sending_roses_l3711_371106


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l3711_371144

theorem sqrt_meaningful_iff_geq_two (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l3711_371144


namespace NUMINAMATH_CALUDE_discount_rate_is_four_percent_l3711_371100

def marked_price : ℝ := 125
def selling_price : ℝ := 120

theorem discount_rate_is_four_percent :
  (marked_price - selling_price) / marked_price * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_discount_rate_is_four_percent_l3711_371100


namespace NUMINAMATH_CALUDE_proportion_fourth_term_l3711_371179

theorem proportion_fourth_term 
  (a b c d : ℚ) 
  (h1 : a + b + c = 58)
  (h2 : c = 2/3 * a)
  (h3 : b = 3/4 * a)
  (h4 : a/b = c/d)
  : d = 12 := by
sorry

end NUMINAMATH_CALUDE_proportion_fourth_term_l3711_371179


namespace NUMINAMATH_CALUDE_percentage_rejected_l3711_371150

theorem percentage_rejected (john_rejection_rate jane_rejection_rate jane_inspection_fraction : ℝ) 
  (h1 : john_rejection_rate = 0.005)
  (h2 : jane_rejection_rate = 0.008)
  (h3 : jane_inspection_fraction = 0.8333333333333333)
  : jane_rejection_rate * jane_inspection_fraction + 
    john_rejection_rate * (1 - jane_inspection_fraction) = 0.0075 := by
  sorry

end NUMINAMATH_CALUDE_percentage_rejected_l3711_371150


namespace NUMINAMATH_CALUDE_least_divisible_by_first_eight_l3711_371198

theorem least_divisible_by_first_eight : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, k > 0 ∧ k ≤ 8 → k ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, k > 0 ∧ k ≤ 8 → k ∣ m) → n ≤ m) ∧
  n = 840 :=
by sorry

end NUMINAMATH_CALUDE_least_divisible_by_first_eight_l3711_371198


namespace NUMINAMATH_CALUDE_marble_distribution_marble_distribution_proof_l3711_371134

theorem marble_distribution (total_marbles : ℕ) (initial_group : ℕ) (joined : ℕ) : Prop :=
  total_marbles = 180 →
  initial_group = 18 →
  (total_marbles / initial_group : ℚ) - (total_marbles / (initial_group + joined) : ℚ) = 1 →
  joined = 2

-- The proof would go here, but we'll use sorry as requested
theorem marble_distribution_proof : marble_distribution 180 18 2 := by sorry

end NUMINAMATH_CALUDE_marble_distribution_marble_distribution_proof_l3711_371134


namespace NUMINAMATH_CALUDE_printer_task_pages_l3711_371172

theorem printer_task_pages : ∀ (P : ℕ),
  (P / 60 + (P / 60 + 3) = P / 24) →
  (P = 360) :=
by
  sorry

#check printer_task_pages

end NUMINAMATH_CALUDE_printer_task_pages_l3711_371172


namespace NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l3711_371156

/-- Pascal's triangle as a function from row and column to the value -/
def pascal (n k : ℕ) : ℕ := sorry

/-- The set of all numbers in Pascal's triangle -/
def pascalNumbers : Set ℕ := sorry

/-- The set of four-digit numbers in Pascal's triangle -/
def fourDigitPascalNumbers : Set ℕ := {n ∈ pascalNumbers | 1000 ≤ n ∧ n ≤ 9999}

/-- The third smallest element in a set of natural numbers -/
def thirdSmallest (S : Set ℕ) : ℕ := sorry

theorem third_smallest_four_digit_pascal : 
  thirdSmallest fourDigitPascalNumbers = 1002 := by sorry

end NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l3711_371156


namespace NUMINAMATH_CALUDE_no_three_reals_satisfying_conditions_l3711_371117

/-- Definition of the set S(a) -/
def S (a : ℝ) : Set ℕ := {n : ℕ | ∃ m : ℕ, n = ⌊m * a⌋}

/-- Theorem stating the impossibility of finding three positive reals satisfying the given conditions -/
theorem no_three_reals_satisfying_conditions :
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (S a ∩ S b = ∅) ∧ (S b ∩ S c = ∅) ∧ (S c ∩ S a = ∅) ∧
    (S a ∪ S b ∪ S c = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_no_three_reals_satisfying_conditions_l3711_371117


namespace NUMINAMATH_CALUDE_tan_ratio_problem_l3711_371136

theorem tan_ratio_problem (x : ℝ) (h : Real.tan (x + π / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_problem_l3711_371136


namespace NUMINAMATH_CALUDE_soccer_camp_afternoon_l3711_371170

/-- Given 2000 kids in total, with half going to soccer camp and 1/4 of those going in the morning,
    the number of kids going to soccer camp in the afternoon is 750. -/
theorem soccer_camp_afternoon (total : ℕ) (soccer : ℕ) (morning : ℕ) : 
  total = 2000 →
  soccer = total / 2 →
  morning = soccer / 4 →
  soccer - morning = 750 := by
  sorry

end NUMINAMATH_CALUDE_soccer_camp_afternoon_l3711_371170


namespace NUMINAMATH_CALUDE_arccos_cos_nine_l3711_371128

/-- The arccosine of the cosine of 9 is equal to 9 modulo 2π. -/
theorem arccos_cos_nine : Real.arccos (Real.cos 9) = 9 % (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_nine_l3711_371128


namespace NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l3711_371118

/-- A positive geometric progression -/
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ q : ℝ), q > 0 ∧ ∀ n, a n = a₁ * q ^ (n - 1)

/-- An arithmetic progression -/
def arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ (b₁ d : ℝ), ∀ n, b n = b₁ + (n - 1) * d

theorem geometric_arithmetic_inequality (a b : ℕ → ℝ) :
  geometric_progression a →
  arithmetic_progression b →
  (∀ n, a n > 0) →
  a 6 = b 7 →
  a 3 + a 9 ≥ b 4 + b 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l3711_371118


namespace NUMINAMATH_CALUDE_puzzle_solution_l3711_371186

theorem puzzle_solution :
  ∃ (g n o u w : Nat),
    g ∈ Finset.range 10 ∧
    n ∈ Finset.range 10 ∧
    o ∈ Finset.range 10 ∧
    u ∈ Finset.range 10 ∧
    w ∈ Finset.range 10 ∧
    (100 * g + 10 * u + n) ^ 2 = 100000 * w + 10000 * o + 1000 * w + 100 * g + 10 * u + n ∧
    o - w = 3 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3711_371186


namespace NUMINAMATH_CALUDE_perpendicular_line_exists_l3711_371163

-- Define a line in 2D space
def Line2D := Set (ℝ × ℝ)

-- Define a point in 2D space
def Point2D := ℝ × ℝ

-- Function to check if a line is perpendicular to another line
def isPerpendicular (l1 l2 : Line2D) : Prop := sorry

-- Function to check if a point is on a line
def isPointOnLine (p : Point2D) (l : Line2D) : Prop := sorry

-- Theorem: For any line and any point, there exists a perpendicular line through that point
theorem perpendicular_line_exists (AB : Line2D) (P : Point2D) : 
  ∃ (l : Line2D), isPerpendicular l AB ∧ isPointOnLine P l := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_exists_l3711_371163


namespace NUMINAMATH_CALUDE_gcd_lcm_inequality_implies_divisibility_l3711_371107

theorem gcd_lcm_inequality_implies_divisibility (a b : ℕ) 
  (h : a * Nat.gcd a b + b * Nat.lcm a b < (5/2) * a * b) : 
  b ∣ a := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_inequality_implies_divisibility_l3711_371107


namespace NUMINAMATH_CALUDE_total_capacity_is_132000_l3711_371125

/-- The capacity of a train's boxcars -/
def train_capacity (num_red num_blue num_black : ℕ) (black_capacity : ℕ) : ℕ :=
  let blue_capacity := 2 * black_capacity
  let red_capacity := 3 * blue_capacity
  num_red * red_capacity + num_blue * blue_capacity + num_black * black_capacity

/-- Theorem: The total capacity of the train's boxcars is 132000 pounds -/
theorem total_capacity_is_132000 :
  train_capacity 3 4 7 4000 = 132000 := by
  sorry

end NUMINAMATH_CALUDE_total_capacity_is_132000_l3711_371125


namespace NUMINAMATH_CALUDE_circle_equation_l3711_371190

/-- Theorem: Given a circle with center (a, 0) where a < 0, radius √5, and tangent to the line x + 2y = 0, 
    the equation of the circle is (x + 5)² + y² = 5. -/
theorem circle_equation (a : ℝ) (h1 : a < 0) :
  let r : ℝ := Real.sqrt 5
  let d : ℝ → ℝ → ℝ := λ x y => |x + 2*y| / Real.sqrt 5
  (d a 0 = r) → 
  (∀ x y, (x - a)^2 + y^2 = 5 ↔ (x + 5)^2 + y^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3711_371190


namespace NUMINAMATH_CALUDE_cos_three_pi_four_plus_two_alpha_l3711_371192

theorem cos_three_pi_four_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 8 - α) = 1 / 6) : 
  Real.cos (3 * π / 4 + 2 * α) = 17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_four_plus_two_alpha_l3711_371192


namespace NUMINAMATH_CALUDE_circular_track_length_l3711_371142

/-- The length of a circular track given specific overtaking conditions -/
theorem circular_track_length : ∃ (x : ℝ), x > 0 ∧ 279 < x ∧ x < 281 ∧
  ∃ (v_fast v_slow : ℝ), v_fast > v_slow ∧ v_fast > 0 ∧ v_slow > 0 ∧
  (150 / (x - 150) = (x + 100) / (x + 50)) := by
  sorry

end NUMINAMATH_CALUDE_circular_track_length_l3711_371142


namespace NUMINAMATH_CALUDE_point_not_in_region_l3711_371110

def plane_region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

theorem point_not_in_region : ¬ plane_region 2 0 := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_region_l3711_371110


namespace NUMINAMATH_CALUDE_one_angle_not_determine_triangle_l3711_371129

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- sides
  (α β γ : ℝ)  -- angles

-- Define what it means for a triangle to be determined
def is_determined (t : Triangle) : Prop :=
  ∀ t' : Triangle, t.α = t'.α ∧ t.β = t'.β ∧ t.γ = t'.γ ∧ 
                   t.a = t'.a ∧ t.b = t'.b ∧ t.c = t'.c

-- Theorem: One angle does not determine a triangle
theorem one_angle_not_determine_triangle :
  ∃ (t t' : Triangle), t.α = t'.α ∧ ¬(is_determined t) :=
sorry

end NUMINAMATH_CALUDE_one_angle_not_determine_triangle_l3711_371129


namespace NUMINAMATH_CALUDE_reading_time_difference_l3711_371168

/-- The difference in reading time between two people with different reading speeds -/
theorem reading_time_difference 
  (tristan_speed : ℝ) 
  (ella_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : tristan_speed = 120)
  (h2 : ella_speed = 40)
  (h3 : book_pages = 360) :
  (book_pages / ella_speed - book_pages / tristan_speed) * 60 = 360 :=
by sorry

end NUMINAMATH_CALUDE_reading_time_difference_l3711_371168


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l3711_371124

theorem arithmetic_simplification :
  (2 * Real.sqrt 12 - (1/2) * Real.sqrt 18) - (Real.sqrt 75 - (1/4) * Real.sqrt 32) = -Real.sqrt 3 - (1/2) * Real.sqrt 2 ∧
  (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + Real.sqrt 48 / (2 * Real.sqrt (1/2)) - Real.sqrt 30 / Real.sqrt 5 = 1 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l3711_371124


namespace NUMINAMATH_CALUDE_orthogonal_vectors_k_values_l3711_371130

theorem orthogonal_vectors_k_values (a b : ℝ × ℝ) (k : ℝ) :
  a = (0, 2) →
  b = (Real.sqrt 3, 1) →
  (a.1 - k * b.1, a.2 - k * b.2) • (k * a.1 + b.1, k * a.2 + b.2) = 0 →
  k = -1 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_k_values_l3711_371130


namespace NUMINAMATH_CALUDE_circular_film_radius_l3711_371199

/-- The radius of a circular film formed by a liquid poured from a rectangular container onto a water surface -/
theorem circular_film_radius 
  (length width height thickness : ℝ) 
  (h_length : length = 10)
  (h_width : width = 4)
  (h_height : height = 8)
  (h_thickness : thickness = 0.15)
  (h_positive : length > 0 ∧ width > 0 ∧ height > 0 ∧ thickness > 0) :
  let volume := length * width * height
  let radius := Real.sqrt (volume / (thickness * Real.pi))
  radius = Real.sqrt (2133.33 / Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_circular_film_radius_l3711_371199


namespace NUMINAMATH_CALUDE_expression_value_l3711_371155

theorem expression_value (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) :
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3711_371155


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l3711_371133

theorem painted_cube_theorem (n : ℕ) (h1 : n > 2) :
  (n - 2)^3 = 6 * (n - 2)^2 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l3711_371133


namespace NUMINAMATH_CALUDE_number_multiplying_a_l3711_371149

theorem number_multiplying_a (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a / 4 = b / 3) :
  ∃ x : ℝ, x * a = 4 * b ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplying_a_l3711_371149


namespace NUMINAMATH_CALUDE_product_theorem_l3711_371175

theorem product_theorem (x : ℝ) : 3.6 * x = 10.08 → 15 * x = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_theorem_l3711_371175


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3711_371164

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0) : 
  (60 / 360 * (2 * Real.pi * C) = 40 / 360 * (2 * Real.pi * D)) → 
  (Real.pi * C^2) / (Real.pi * D^2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3711_371164


namespace NUMINAMATH_CALUDE_zeros_after_decimal_point_of_inverse_40_power_20_l3711_371197

theorem zeros_after_decimal_point_of_inverse_40_power_20 :
  let n : ℕ := 40
  let p : ℕ := 20
  let f : ℚ := 1 / (n^p : ℚ)
  (∃ (x : ℚ) (k : ℕ), f = x * 10^(-k : ℤ) ∧ x ≥ 1/10 ∧ x < 1 ∧ k = 38) :=
by sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_point_of_inverse_40_power_20_l3711_371197


namespace NUMINAMATH_CALUDE_distance_to_origin_l3711_371121

def A : ℝ × ℝ := (-1, -2)

theorem distance_to_origin : Real.sqrt ((A.1 - 0)^2 + (A.2 - 0)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3711_371121


namespace NUMINAMATH_CALUDE_problem_solution_l3711_371102

theorem problem_solution (x : ℝ) (h : x = -3007) :
  |(|Real.sqrt ((|x| - x)) - x| - x) - Real.sqrt (|x - x^2|)| = 3084 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3711_371102


namespace NUMINAMATH_CALUDE_commission_calculation_l3711_371154

/-- The commission percentage for sales exceeding $500 -/
def excess_commission_percentage : ℝ :=
  -- We'll define this value and prove it's equal to 50
  50

theorem commission_calculation (total_sale : ℝ) (total_commission_percentage : ℝ) :
  total_sale = 800 →
  total_commission_percentage = 31.25 →
  (0.2 * 500 + (excess_commission_percentage / 100) * (total_sale - 500)) / total_sale = total_commission_percentage / 100 →
  excess_commission_percentage = 50 := by
  sorry

#check commission_calculation

end NUMINAMATH_CALUDE_commission_calculation_l3711_371154


namespace NUMINAMATH_CALUDE_remainder_of_power_mod_quadratic_l3711_371108

theorem remainder_of_power_mod_quadratic (x : ℤ) : 
  (x + 2)^1004 ≡ -x [ZMOD (x^2 - x + 1)] :=
sorry

end NUMINAMATH_CALUDE_remainder_of_power_mod_quadratic_l3711_371108


namespace NUMINAMATH_CALUDE_halloween_candy_proof_l3711_371157

/-- The number of candy pieces Faye scored on Halloween -/
def initial_candy : ℕ := 47

/-- The number of candy pieces Faye ate on the first night -/
def eaten_candy : ℕ := 25

/-- The number of candy pieces Faye's sister gave her -/
def gifted_candy : ℕ := 40

/-- The number of candy pieces Faye has now -/
def current_candy : ℕ := 62

/-- Theorem stating that the initial number of candy pieces is correct -/
theorem halloween_candy_proof : 
  initial_candy - eaten_candy + gifted_candy = current_candy :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_proof_l3711_371157


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l3711_371178

/-- Given that dried grapes contain 20% water by weight and 40 kg of fresh grapes
    produce 5 kg of dried grapes, prove that the percentage of water in fresh grapes is 90%. -/
theorem water_percentage_in_fresh_grapes :
  ∀ (fresh_weight dried_weight : ℝ) (dried_water_percentage : ℝ),
    fresh_weight = 40 →
    dried_weight = 5 →
    dried_water_percentage = 20 →
    (fresh_weight - dried_weight * (1 - dried_water_percentage / 100)) / fresh_weight * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l3711_371178


namespace NUMINAMATH_CALUDE_min_value_of_function_l3711_371109

theorem min_value_of_function (x : ℝ) (h : x > 4) : 
  ∃ (y_min : ℝ), y_min = 6 ∧ ∀ y, y = x + 1 / (x - 4) → y ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3711_371109


namespace NUMINAMATH_CALUDE_bennys_work_hours_l3711_371138

/-- 
Given that Benny worked 7 hours per day for 14 days, 
prove that his total work hours equals 98.
-/
theorem bennys_work_hours : 
  let hours_per_day : ℕ := 7
  let days_worked : ℕ := 14
  hours_per_day * days_worked = 98 := by sorry

end NUMINAMATH_CALUDE_bennys_work_hours_l3711_371138


namespace NUMINAMATH_CALUDE_y_exceeds_x_l3711_371183

theorem y_exceeds_x (x y : ℝ) (h : x = 0.75 * y) : (y - x) / x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_y_exceeds_x_l3711_371183


namespace NUMINAMATH_CALUDE_problem_solution_l3711_371169

theorem problem_solution (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) : 
  (1/3) * x^8 * y^9 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3711_371169


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3711_371113

theorem rectangle_diagonal (l w : ℝ) (h1 : l + w = 23) (h2 : l * w = 120) :
  Real.sqrt (l^2 + w^2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3711_371113


namespace NUMINAMATH_CALUDE_sally_cost_is_42000_l3711_371180

/-- The cost of Lightning McQueen in dollars -/
def lightning_cost : ℝ := 140000

/-- The cost of Mater as a percentage of Lightning McQueen's cost -/
def mater_percentage : ℝ := 0.10

/-- The factor by which Sally McQueen's cost is greater than Mater's cost -/
def sally_factor : ℝ := 3

/-- The cost of Sally McQueen in dollars -/
def sally_cost : ℝ := lightning_cost * mater_percentage * sally_factor

theorem sally_cost_is_42000 : sally_cost = 42000 := by
  sorry

end NUMINAMATH_CALUDE_sally_cost_is_42000_l3711_371180


namespace NUMINAMATH_CALUDE_river_width_is_500_l3711_371141

/-- Represents the river crossing scenario -/
structure RiverCrossing where
  velocity : ℝ  -- Boatman's velocity in m/sec
  time : ℝ      -- Time taken to cross the river in seconds
  drift : ℝ     -- Drift distance in meters

/-- Calculates the width of the river given the crossing parameters -/
def riverWidth (rc : RiverCrossing) : ℝ :=
  rc.velocity * rc.time

/-- Theorem stating that the width of the river is 500 meters 
    given the specific conditions -/
theorem river_width_is_500 (rc : RiverCrossing) 
  (h1 : rc.velocity = 10)
  (h2 : rc.time = 50)
  (h3 : rc.drift = 300) : 
  riverWidth rc = 500 := by
  sorry

#check river_width_is_500

end NUMINAMATH_CALUDE_river_width_is_500_l3711_371141


namespace NUMINAMATH_CALUDE_greatest_c_value_l3711_371176

theorem greatest_c_value (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 20 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 9*5 - 20 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_c_value_l3711_371176


namespace NUMINAMATH_CALUDE_triangle_problem_l3711_371115

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  A = 30 * π / 180 ∧  -- Convert 30° to radians
  a = 2 ∧
  b = 2 * Real.sqrt 3 ∧
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  c / Real.sin C = a / Real.sin A →
  -- Conclusions
  Real.sin B = Real.sqrt 3 / 2 ∧
  ∃! (B' C' : Real), B' ≠ B ∧ 
    A + B + C = π ∧
    A + B' + C' = π ∧
    a / Real.sin A = b / Real.sin B' ∧
    b / Real.sin B' = c / Real.sin C' ∧
    c / Real.sin C' = a / Real.sin A :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l3711_371115


namespace NUMINAMATH_CALUDE_original_car_price_l3711_371159

/-- Represents the original cost price of the car -/
def original_price : ℝ := sorry

/-- Represents the price after the first sale (11% loss) -/
def first_sale_price : ℝ := original_price * (1 - 0.11)

/-- Represents the price after the second sale (15% gain) -/
def second_sale_price : ℝ := first_sale_price * (1 + 0.15)

/-- Represents the final selling price -/
def final_sale_price : ℝ := 75000

/-- Theorem stating the original price of the car -/
theorem original_car_price : 
  ∃ (price : ℝ), original_price = price ∧ 
  (price ≥ 58744) ∧ (price ≤ 58745) ∧
  (second_sale_price * (1 + 0.25) = final_sale_price) :=
sorry

end NUMINAMATH_CALUDE_original_car_price_l3711_371159


namespace NUMINAMATH_CALUDE_smallest_tangent_circle_radius_l3711_371153

theorem smallest_tangent_circle_radius 
  (square_side : ℝ) 
  (semicircle_radius : ℝ) 
  (quarter_circle_radius : ℝ) 
  (h1 : square_side = 4) 
  (h2 : semicircle_radius = 1) 
  (h3 : quarter_circle_radius = 2) : 
  ∃ r : ℝ, r = Real.sqrt 2 - 3/2 ∧ 
  (∀ x y : ℝ, x^2 + y^2 = r^2 → 
    ((x - 2)^2 + y^2 = 1^2 ∨ 
     (x + 2)^2 + y^2 = 1^2 ∨ 
     x^2 + (y - 2)^2 = 1^2 ∨ 
     x^2 + (y + 2)^2 = 1^2) ∧
    x^2 + y^2 = (2 - r)^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_tangent_circle_radius_l3711_371153


namespace NUMINAMATH_CALUDE_iggy_monday_run_l3711_371181

/-- Represents the number of miles run on each day of the week --/
structure WeeklyRun where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- Calculates the total miles run in a week --/
def totalMiles (run : WeeklyRun) : ℝ :=
  run.monday + run.tuesday + run.wednesday + run.thursday + run.friday

/-- Represents Iggy's running schedule for the week --/
def iggyRun : WeeklyRun where
  monday := 3  -- This is what we want to prove
  tuesday := 4
  wednesday := 6
  thursday := 8
  friday := 3

/-- Iggy's pace in minutes per mile --/
def pace : ℝ := 10

/-- Total time Iggy spent running in minutes --/
def totalTime : ℝ := 4 * 60

theorem iggy_monday_run :
  iggyRun.monday = 3 ∧ 
  totalMiles iggyRun * pace = totalTime := by
  sorry


end NUMINAMATH_CALUDE_iggy_monday_run_l3711_371181


namespace NUMINAMATH_CALUDE_distance_between_centers_l3711_371185

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  right_angle : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0
  xy_length : (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = 80^2
  xz_length : (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 150^2
  yz_length : (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = 170^2

-- Define the inscribed circle C₁
def InscribedCircle (T : Triangle X Y Z) (C : ℝ × ℝ) (r : ℝ) : Prop := sorry

-- Define MN perpendicular to XZ and tangent to C₁
def MN_Perpendicular_Tangent (T : Triangle X Y Z) (C₁ : ℝ × ℝ) (r₁ : ℝ) (M N : ℝ × ℝ) : Prop := sorry

-- Define AB perpendicular to XY and tangent to C₁
def AB_Perpendicular_Tangent (T : Triangle X Y Z) (C₁ : ℝ × ℝ) (r₁ : ℝ) (A B : ℝ × ℝ) : Prop := sorry

-- Define the inscribed circle C₂ of MZN
def InscribedCircle_MZN (T : Triangle X Y Z) (M N : ℝ × ℝ) (C₂ : ℝ × ℝ) (r₂ : ℝ) : Prop := sorry

-- Define the inscribed circle C₃ of YAB
def InscribedCircle_YAB (T : Triangle X Y Z) (A B : ℝ × ℝ) (C₃ : ℝ × ℝ) (r₃ : ℝ) : Prop := sorry

theorem distance_between_centers (X Y Z M N A B C₁ C₂ C₃ : ℝ × ℝ) (r₁ r₂ r₃ : ℝ) 
  (h_triangle : Triangle X Y Z)
  (h_c₁ : InscribedCircle h_triangle C₁ r₁)
  (h_mn : MN_Perpendicular_Tangent h_triangle C₁ r₁ M N)
  (h_ab : AB_Perpendicular_Tangent h_triangle C₁ r₁ A B)
  (h_c₂ : InscribedCircle_MZN h_triangle M N C₂ r₂)
  (h_c₃ : InscribedCircle_YAB h_triangle A B C₃ r₃) :
  (C₂.1 - C₃.1)^2 + (C₂.2 - C₃.2)^2 = 9884.5 := by sorry

end NUMINAMATH_CALUDE_distance_between_centers_l3711_371185


namespace NUMINAMATH_CALUDE_arithmetic_sum_odd_sequence_l3711_371182

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem arithmetic_sum_odd_sequence :
  let seq := arithmetic_sequence 1 2 11
  (∀ x ∈ seq, is_odd x) ∧
  (seq.length = 11) ∧
  (seq.getLast? = some 21) →
  seq.sum = 121 := by
  sorry

#eval arithmetic_sequence 1 2 11

end NUMINAMATH_CALUDE_arithmetic_sum_odd_sequence_l3711_371182


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3711_371177

/-- A geometric sequence with first term a₁ = -1 and a₂ + a₃ = -2 has common ratio q = -2 or q = 1 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : a 1 = -1) 
  (h_sum : a 2 + a 3 = -2) :
  q = -2 ∨ q = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3711_371177


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3711_371166

theorem rectangle_dimension_change 
  (L B : ℝ) -- Original length and breadth
  (L' : ℝ) -- New length
  (h1 : L > 0 ∧ B > 0) -- Positive dimensions
  (h2 : L' * (3 * B) = (3/2) * (L * B)) -- Area increased by 50% and breadth tripled
  : L' = L / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3711_371166


namespace NUMINAMATH_CALUDE_custom_product_equals_interval_l3711_371140

-- Define sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2 * x - x^2)}
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define the custom Cartesian product
def custom_product (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

-- Theorem statement
theorem custom_product_equals_interval :
  custom_product A B = {x : ℝ | 0 ≤ x ∧ x ≤ 1 ∨ x > 2} :=
sorry

end NUMINAMATH_CALUDE_custom_product_equals_interval_l3711_371140


namespace NUMINAMATH_CALUDE_remainder_problem_l3711_371160

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1259 % d = r) (h3 : 1567 % d = r) (h4 : 2257 % d = r) : 
  d - r = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3711_371160


namespace NUMINAMATH_CALUDE_difference_zero_point_eight_and_one_eighth_l3711_371188

theorem difference_zero_point_eight_and_one_eighth : (0.8 : ℝ) - (1 / 8 : ℝ) = 0.675 := by
  sorry

end NUMINAMATH_CALUDE_difference_zero_point_eight_and_one_eighth_l3711_371188


namespace NUMINAMATH_CALUDE_money_distribution_l3711_371162

/-- Represents the share of money for each person -/
structure Shares :=
  (w : ℚ) (x : ℚ) (y : ℚ) (z : ℚ)

/-- The theorem statement -/
theorem money_distribution (s : Shares) :
  s.w + s.x + s.y + s.z > 0 ∧  -- Ensure total sum is positive
  s.x = 6 * s.w ∧              -- Proportion for x
  s.y = 2 * s.w ∧              -- Proportion for y
  s.z = 4 * s.w ∧              -- Proportion for z
  s.x = s.y + 1500             -- x gets $1500 more than y
  →
  s.w = 375 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3711_371162


namespace NUMINAMATH_CALUDE_train_speed_l3711_371193

/-- Proves that a train of given length crossing a bridge of given length in a given time has a specific speed -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  bridge_length = 132 →
  crossing_time = 16.13204276991174 →
  (train_length + bridge_length) / crossing_time * 3.6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3711_371193


namespace NUMINAMATH_CALUDE_least_valid_n_l3711_371116

def is_valid (n : ℕ) : Prop :=
  ∃ k₁ k₂ : ℕ, 1 ≤ k₁ ∧ k₁ ≤ n + 1 ∧ 1 ≤ k₂ ∧ k₂ ≤ n + 1 ∧
  (n^2 - n) % k₁ = 0 ∧ (n^2 - n) % k₂ ≠ 0

theorem least_valid_n :
  is_valid 5 ∧ ∀ m : ℕ, m < 5 → ¬is_valid m :=
sorry

end NUMINAMATH_CALUDE_least_valid_n_l3711_371116


namespace NUMINAMATH_CALUDE_seventh_graders_count_l3711_371196

/-- The number of fifth graders going on the trip -/
def fifth_graders : ℕ := 109

/-- The number of sixth graders going on the trip -/
def sixth_graders : ℕ := 115

/-- The number of teachers chaperoning -/
def teachers : ℕ := 4

/-- The number of parent chaperones per grade -/
def parents_per_grade : ℕ := 2

/-- The number of grades participating -/
def grades : ℕ := 3

/-- The number of buses for the trip -/
def buses : ℕ := 5

/-- The number of seats per bus -/
def seats_per_bus : ℕ := 72

/-- Theorem: The number of seventh graders going on the trip is 118 -/
theorem seventh_graders_count : 
  (buses * seats_per_bus) - 
  (fifth_graders + sixth_graders + (teachers + parents_per_grade * grades)) = 118 := by
  sorry

end NUMINAMATH_CALUDE_seventh_graders_count_l3711_371196


namespace NUMINAMATH_CALUDE_committee_meeting_arrangements_l3711_371173

/-- Represents a school in the club --/
structure School :=
  (members : Nat)

/-- Represents the club with its schools --/
structure Club :=
  (schools : List School)
  (total_members : Nat)

/-- Represents the committee meeting arrangement --/
structure CommitteeMeeting :=
  (host : School)
  (first_non_host : School)
  (second_non_host : School)
  (host_reps : Nat)
  (first_non_host_reps : Nat)
  (second_non_host_reps : Nat)

/-- The number of ways to arrange a committee meeting --/
def arrange_committee_meeting (club : Club) : Nat :=
  sorry

/-- Theorem stating the number of possible committee meeting arrangements --/
theorem committee_meeting_arrangements (club : Club) :
  club.schools.length = 3 ∧
  club.total_members = 18 ∧
  (∀ s ∈ club.schools, s.members = 6) →
  arrange_committee_meeting club = 5400 :=
sorry

end NUMINAMATH_CALUDE_committee_meeting_arrangements_l3711_371173


namespace NUMINAMATH_CALUDE_angle_value_proof_l3711_371126

/-- Given that cos 16° = sin 14° + sin d° and 0 < d < 90, prove that d = 46 -/
theorem angle_value_proof (d : ℝ) 
  (h1 : Real.cos (16 * π / 180) = Real.sin (14 * π / 180) + Real.sin (d * π / 180))
  (h2 : 0 < d)
  (h3 : d < 90) : 
  d = 46 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_proof_l3711_371126


namespace NUMINAMATH_CALUDE_three_X_seven_equals_eight_l3711_371195

/-- The operation X defined for two real numbers -/
def X (a b : ℝ) : ℝ := b + 15 * a - a^2 - 5 * b

/-- Theorem stating that 3X7 equals 8 -/
theorem three_X_seven_equals_eight : X 3 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_X_seven_equals_eight_l3711_371195


namespace NUMINAMATH_CALUDE_spheres_radius_is_correct_l3711_371151

/-- Right circular cone with given dimensions -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Sphere inside the cone -/
structure Sphere where
  radius : ℝ

/-- Configuration of three spheres inside a cone -/
structure SpheresInCone where
  cone : Cone
  sphere : Sphere
  centerPlaneHeight : ℝ

/-- The specific configuration described in the problem -/
def problemConfig : SpheresInCone where
  cone := { baseRadius := 4, height := 15 }
  sphere := { radius := 1.5 }  -- We use the correct answer here
  centerPlaneHeight := 2.5     -- This is r + 1, where r = 1.5

/-- Predicate to check if the spheres are tangent to each other, the base, and the side of the cone -/
def areTangent (config : SpheresInCone) : Prop :=
  -- The actual tangency conditions would be complex to express precisely,
  -- so we use a placeholder predicate
  True

/-- Theorem stating that the given configuration satisfies the problem conditions -/
theorem spheres_radius_is_correct (config : SpheresInCone) :
  config.cone.baseRadius = 4 ∧
  config.cone.height = 15 ∧
  config.centerPlaneHeight = config.sphere.radius + 1 ∧
  areTangent config →
  config.sphere.radius = 1.5 :=
by
  sorry

#check spheres_radius_is_correct

end NUMINAMATH_CALUDE_spheres_radius_is_correct_l3711_371151


namespace NUMINAMATH_CALUDE_quadratic_equation_rewrite_l3711_371135

theorem quadratic_equation_rewrite (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → 
  b + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_rewrite_l3711_371135


namespace NUMINAMATH_CALUDE_pure_imaginary_value_l3711_371114

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_value (x : ℝ) :
  is_pure_imaginary ((x^2 - 1 : ℝ) + (x^2 + 3*x + 2 : ℝ) * I) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_value_l3711_371114


namespace NUMINAMATH_CALUDE_water_heater_theorem_l3711_371187

/-- Calculates the total amount of water in two water heaters -/
def total_water (wallace_capacity : ℚ) : ℚ :=
  let catherine_capacity := wallace_capacity / 2
  let wallace_water := (3 / 4) * wallace_capacity
  let catherine_water := (3 / 4) * catherine_capacity
  wallace_water + catherine_water

/-- Theorem stating that given the conditions, the total water is 45 gallons -/
theorem water_heater_theorem (wallace_capacity : ℚ) 
  (h1 : wallace_capacity = 40) : total_water wallace_capacity = 45 := by
  sorry

#eval total_water 40

end NUMINAMATH_CALUDE_water_heater_theorem_l3711_371187


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l3711_371143

/-- Given an ellipse with equation x²/25 + y²/9 = 1, if a point A on the ellipse has a symmetric 
point B with respect to the origin, and F₂ is its right focus, and AF₂ is perpendicular to BF₂, 
then the area of triangle AF₂B is 9. -/
theorem ellipse_triangle_area (x y : ℝ) (F₂ : ℝ × ℝ) (A B : ℝ × ℝ) : 
  x^2/25 + y^2/9 = 1 →  -- Ellipse equation
  A.1^2/25 + A.2^2/9 = 1 →  -- A is on the ellipse
  B = (-A.1, -A.2) →  -- B is symmetric to A with respect to origin
  F₂.1 = 4 ∧ F₂.2 = 0 →  -- F₂ is the right focus
  (A.1 - F₂.1) * (B.1 - F₂.1) + (A.2 - F₂.2) * (B.2 - F₂.2) = 0 →  -- AF₂ ⊥ BF₂
  (abs (A.1 - F₂.1) * abs (A.2 - B.2)) / 2 = 9 :=  -- Area of triangle AF₂B
by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l3711_371143


namespace NUMINAMATH_CALUDE_checkerboard_coverage_l3711_371101

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- Determines if a checkerboard can be covered by dominoes -/
def can_be_covered (board : Checkerboard) : Prop :=
  Even (board.rows * board.cols)

/-- Theorem stating that a checkerboard can be covered if and only if its area is even -/
theorem checkerboard_coverage (board : Checkerboard) :
  can_be_covered board ↔ Even (board.rows * board.cols) := by sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_l3711_371101


namespace NUMINAMATH_CALUDE_expression_bounds_l3711_371165

theorem expression_bounds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1/2 ≤ |2*a - b| / (|a| + |b|) ∧ |2*a - b| / (|a| + |b|) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l3711_371165


namespace NUMINAMATH_CALUDE_derivative_problems_l3711_371171

open Real

theorem derivative_problems :
  (∀ x > 0, deriv (λ x => (log x) / x) x = (1 - log x) / x^2) ∧
  (∀ x, deriv (λ x => x * exp x) x = (x + 1) * exp x) ∧
  (∀ x, deriv (λ x => cos (2 * x)) x = -2 * sin (2 * x)) :=
by sorry

end NUMINAMATH_CALUDE_derivative_problems_l3711_371171


namespace NUMINAMATH_CALUDE_sentence_is_true_l3711_371167

def sentence := "In this phrase, 1/2 of all digits are '1', fractions of digits '2' and '5' are equal and equal to 1/5, and the proportion of all other digits is 1/10."

def digit_1 := 1
def digit_2 := 2
def digit_3 := 5

def fraction_1 := (1 : ℚ) / 2
def fraction_2 := (1 : ℚ) / 5
def fraction_3 := (1 : ℚ) / 10

theorem sentence_is_true :
  (fraction_1 + 2 * fraction_2 + fraction_3 = 1) ∧
  (digit_1 ≠ digit_2 ∧ digit_1 ≠ digit_3 ∧ digit_2 ≠ digit_3) ∧
  (fraction_1 ≠ fraction_2 ∧ fraction_1 ≠ fraction_3 ∧ fraction_2 ≠ fraction_3) →
  ∃ (total_digits : ℕ),
    (fraction_1 * total_digits).num = (total_digits.digits 10).count digit_1 ∧
    (fraction_2 * total_digits).num = (total_digits.digits 10).count digit_2 ∧
    (fraction_2 * total_digits).num = (total_digits.digits 10).count digit_3 ∧
    (fraction_3 * total_digits).num = total_digits - (fraction_1 * total_digits).num - 2 * (fraction_2 * total_digits).num :=
by sorry

end NUMINAMATH_CALUDE_sentence_is_true_l3711_371167


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_3_l3711_371147

theorem simplify_fraction_with_sqrt_3 :
  (1 / (1 - Real.sqrt 3)) * (1 / (1 + Real.sqrt 3)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_3_l3711_371147


namespace NUMINAMATH_CALUDE_smallest_k_and_digit_sum_l3711_371132

-- Define the function to count digits
def countDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + countDigits (n / 10)

-- Define the function to sum digits
def sumDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumDigits (n / 10)

-- Theorem statement
theorem smallest_k_and_digit_sum :
  ∃ k : ℕ, 
    (k > 0) ∧
    (∀ j : ℕ, j > 0 → j < k → countDigits ((2^j) * (5^300)) < 303) ∧
    (countDigits ((2^k) * (5^300)) = 303) ∧
    (k = 307) ∧
    (sumDigits ((2^k) * (5^300)) = 11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_and_digit_sum_l3711_371132


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l3711_371191

theorem unique_solution_power_equation : 
  ∃! (x y z t : ℕ+), 2^y.val + 2^z.val * 5^t.val - 5^x.val = 1 ∧ 
  x = 2 ∧ y = 4 ∧ z = 1 ∧ t = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l3711_371191


namespace NUMINAMATH_CALUDE_hyperbola_parameter_sum_l3711_371174

/-- Theorem about the sum of parameters for a specific hyperbola -/
theorem hyperbola_parameter_sum :
  let center : ℝ × ℝ := (-3, 1)
  let focus : ℝ × ℝ := (-3 + Real.sqrt 50, 1)
  let vertex : ℝ × ℝ := (-7, 1)
  let h : ℝ := center.1
  let k : ℝ := center.2
  let a : ℝ := |vertex.1 - center.1|
  let c : ℝ := |focus.1 - center.1|
  let b : ℝ := Real.sqrt (c^2 - a^2)
  h + k + a + b = 2 + Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parameter_sum_l3711_371174


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_minus_four_l3711_371137

theorem factorization_of_x_squared_minus_four :
  ∀ x : ℝ, x^2 - 4 = (x + 2) * (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_minus_four_l3711_371137


namespace NUMINAMATH_CALUDE_race_time_proof_l3711_371131

/-- Represents the time taken by runner A to complete the race -/
def time_A : ℝ := 235

/-- Represents the length of the race in meters -/
def race_length : ℝ := 1000

/-- Represents the distance by which A beats B in meters -/
def distance_difference : ℝ := 60

/-- Represents the time difference by which A beats B in seconds -/
def time_difference : ℝ := 15

/-- Theorem stating that given the race conditions, runner A completes the race in 235 seconds -/
theorem race_time_proof :
  (race_length / time_A = (race_length - distance_difference) / time_A) ∧
  (race_length / time_A = race_length / (time_A + time_difference)) →
  time_A = 235 := by
  sorry

end NUMINAMATH_CALUDE_race_time_proof_l3711_371131


namespace NUMINAMATH_CALUDE_workshop_professionals_l3711_371123

theorem workshop_professionals (total : ℕ) (laptops tablets coffee : ℕ)
  (laptops_and_tablets laptops_and_coffee tablets_and_coffee : ℕ)
  (all_three : ℕ) :
  total = 40 →
  laptops = 18 →
  tablets = 14 →
  coffee = 16 →
  laptops_and_tablets = 7 →
  laptops_and_coffee = 5 →
  tablets_and_coffee = 4 →
  all_three = 3 →
  total - (laptops + tablets + coffee -
    laptops_and_tablets - laptops_and_coffee - tablets_and_coffee + all_three) = 5 :=
by sorry

end NUMINAMATH_CALUDE_workshop_professionals_l3711_371123


namespace NUMINAMATH_CALUDE_both_heads_prob_l3711_371127

/-- Represents the outcome of flipping two coins simultaneously -/
inductive CoinFlip
| HH -- Both heads
| HT -- First head, second tail
| TH -- First tail, second head
| TT -- Both tails

/-- The probability of getting a specific outcome when flipping two fair coins -/
def flip_prob : CoinFlip → ℚ
| CoinFlip.HH => 1/4
| CoinFlip.HT => 1/4
| CoinFlip.TH => 1/4
| CoinFlip.TT => 1/4

/-- The process of flipping coins until at least one head appears -/
def flip_until_head : ℕ → ℚ
| 0 => flip_prob CoinFlip.HH
| (n+1) => flip_prob CoinFlip.TT * flip_until_head n

/-- The theorem stating the probability of both coins showing heads when the process stops -/
theorem both_heads_prob : (∑' n, flip_until_head n) = 1/3 :=
sorry


end NUMINAMATH_CALUDE_both_heads_prob_l3711_371127


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l3711_371139

theorem algebraic_expression_simplification (x y : ℝ) :
  3 * (x^2 - 2*x*y + y^2) - 3 * (x^2 - 2*x*y + y^2 - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l3711_371139


namespace NUMINAMATH_CALUDE_spencer_burritos_l3711_371189

/-- Represents the number of ways to make burritos with given constraints -/
def burrito_combinations (total_burritos : ℕ) (max_beef : ℕ) (max_chicken : ℕ) (available_wraps : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 25 ways to make exactly 5 burritos with the given constraints -/
theorem spencer_burritos : burrito_combinations 5 4 3 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_spencer_burritos_l3711_371189


namespace NUMINAMATH_CALUDE_services_total_cost_l3711_371152

def hair_cost : ℝ := 50
def manicure_cost : ℝ := 30
def tip_percentage : ℝ := 0.20

def total_cost : ℝ := hair_cost + manicure_cost + (hair_cost * tip_percentage) + (manicure_cost * tip_percentage)

theorem services_total_cost : total_cost = 96 := by
  sorry

end NUMINAMATH_CALUDE_services_total_cost_l3711_371152


namespace NUMINAMATH_CALUDE_number_problem_l3711_371105

theorem number_problem : 
  ∃ x : ℝ, 2 * x = (10 / 100) * 900 ∧ x = 45 := by sorry

end NUMINAMATH_CALUDE_number_problem_l3711_371105


namespace NUMINAMATH_CALUDE_wall_length_proof_l3711_371104

/-- Proves that the length of a wall is 29 meters given specific brick and wall dimensions and the number of bricks required. -/
theorem wall_length_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_width : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_width = 2 →
  wall_height = 0.75 →
  num_bricks = 29000 →
  ∃ (wall_length : ℝ), wall_length = 29 :=
by
  sorry

#check wall_length_proof

end NUMINAMATH_CALUDE_wall_length_proof_l3711_371104


namespace NUMINAMATH_CALUDE_sandy_fingernail_record_age_l3711_371148

/-- Calculates the age at which Sandy will achieve the world record for longest fingernails -/
theorem sandy_fingernail_record_age 
  (world_record : ℝ)
  (sandy_current_age : ℕ)
  (sandy_current_length : ℝ)
  (growth_rate_per_month : ℝ)
  (h1 : world_record = 26)
  (h2 : sandy_current_age = 12)
  (h3 : sandy_current_length = 2)
  (h4 : growth_rate_per_month = 0.1) :
  sandy_current_age + (world_record - sandy_current_length) / (growth_rate_per_month * 12) = 32 := by
sorry

end NUMINAMATH_CALUDE_sandy_fingernail_record_age_l3711_371148


namespace NUMINAMATH_CALUDE_juan_running_l3711_371112

/-- The distance traveled when moving at a constant speed for a given time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Juan's running problem -/
theorem juan_running :
  let speed : ℝ := 10  -- miles per hour
  let time : ℝ := 8    -- hours
  distance speed time = 80 := by
  sorry

end NUMINAMATH_CALUDE_juan_running_l3711_371112


namespace NUMINAMATH_CALUDE_initial_girls_count_l3711_371146

theorem initial_girls_count (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls = total / 2) →  -- Initially, 50% of the group are girls
  (initial_girls - 3 : ℚ) / total = 2/5 →  -- After changes, 40% are girls
  initial_girls = 15 := by
sorry

end NUMINAMATH_CALUDE_initial_girls_count_l3711_371146


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3711_371161

theorem imaginary_part_of_complex_fraction : 
  Complex.im (2 * Complex.I^3 / (1 - Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3711_371161


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l3711_371122

theorem square_minus_product_plus_square : 6^2 - 5*6 + 4^2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l3711_371122
