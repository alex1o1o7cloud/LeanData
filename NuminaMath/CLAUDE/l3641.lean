import Mathlib

namespace NUMINAMATH_CALUDE_square_tiles_count_l3641_364133

/-- Represents the number of edges for each type of tile -/
def edges_per_tile : Fin 3 → ℕ
| 0 => 3  -- triangular tiles
| 1 => 4  -- square tiles
| 2 => 5  -- pentagonal tiles

theorem square_tiles_count 
  (total_tiles : ℕ) 
  (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 35)
  (h_total_edges : total_edges = 140) :
  ∃ (t s p : ℕ), 
    t + s + p = total_tiles ∧ 
    3*t + 4*s + 5*p = total_edges ∧
    s = 35 :=
sorry

end NUMINAMATH_CALUDE_square_tiles_count_l3641_364133


namespace NUMINAMATH_CALUDE_marble_weight_sum_l3641_364106

theorem marble_weight_sum : 
  let piece1 : ℝ := 0.33
  let piece2 : ℝ := 0.33
  let piece3 : ℝ := 0.08
  piece1 + piece2 + piece3 = 0.74 :=
by sorry

end NUMINAMATH_CALUDE_marble_weight_sum_l3641_364106


namespace NUMINAMATH_CALUDE_larger_box_jellybean_count_l3641_364189

/-- The number of jellybeans in a box with dimensions thrice as large -/
def jellybeans_in_larger_box (original_capacity : ℕ) : ℕ :=
  original_capacity * 27

/-- Theorem: A box with dimensions thrice as large holds 4050 jellybeans -/
theorem larger_box_jellybean_count :
  jellybeans_in_larger_box 150 = 4050 := by
  sorry

end NUMINAMATH_CALUDE_larger_box_jellybean_count_l3641_364189


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_37_l3641_364190

theorem smallest_four_digit_divisible_by_37 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 1036 → ¬(37 ∣ n)) ∧ 
  1000 ≤ 1036 ∧ 
  1036 < 10000 ∧ 
  37 ∣ 1036 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_37_l3641_364190


namespace NUMINAMATH_CALUDE_jack_collection_books_per_author_l3641_364102

/-- Represents Jack's classic book collection -/
structure ClassicCollection where
  authors : Nat
  total_books : Nat

/-- Calculates the number of books per author in a classic collection -/
def books_per_author (c : ClassicCollection) : Nat :=
  c.total_books / c.authors

/-- Theorem: In Jack's collection of 6 authors and 198 books, each author has 33 books -/
theorem jack_collection_books_per_author :
  let jack_collection : ClassicCollection := { authors := 6, total_books := 198 }
  books_per_author jack_collection = 33 := by
  sorry

end NUMINAMATH_CALUDE_jack_collection_books_per_author_l3641_364102


namespace NUMINAMATH_CALUDE_math_reading_homework_difference_l3641_364101

theorem math_reading_homework_difference :
  let math_pages : ℕ := 5
  let reading_pages : ℕ := 2
  math_pages - reading_pages = 3 :=
by sorry

end NUMINAMATH_CALUDE_math_reading_homework_difference_l3641_364101


namespace NUMINAMATH_CALUDE_product_expression_value_l3641_364143

def product_expression : ℚ :=
  (3^3 - 2^3) / (3^3 + 2^3) *
  (4^3 - 3^3) / (4^3 + 3^3) *
  (5^3 - 4^3) / (5^3 + 4^3) *
  (6^3 - 5^3) / (6^3 + 5^3) *
  (7^3 - 6^3) / (7^3 + 6^3)

theorem product_expression_value : product_expression = 17 / 901 := by
  sorry

end NUMINAMATH_CALUDE_product_expression_value_l3641_364143


namespace NUMINAMATH_CALUDE_symmetric_point_l3641_364164

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := 5*x + 4*y + 21 = 0

/-- The original point P --/
def P : ℝ × ℝ := (4, 0)

/-- The symmetric point P' --/
def P' : ℝ × ℝ := (-6, -8)

/-- Theorem stating that P' is symmetric to P with respect to the line of symmetry --/
theorem symmetric_point : 
  let midpoint := ((P.1 + P'.1) / 2, (P.2 + P'.2) / 2)
  line_of_symmetry midpoint.1 midpoint.2 ∧ 
  (P'.2 - P.2) * 5 = -(P'.1 - P.1) * 4 :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_l3641_364164


namespace NUMINAMATH_CALUDE_one_positive_real_solution_l3641_364195

/-- The polynomial function f(x) = x^11 + 5x^10 + 20x^9 + 1000x^8 - 800x^7 -/
def f (x : ℝ) : ℝ := x^11 + 5*x^10 + 20*x^9 + 1000*x^8 - 800*x^7

/-- Theorem stating that f(x) = 0 has exactly one positive real solution -/
theorem one_positive_real_solution : ∃! (x : ℝ), x > 0 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_one_positive_real_solution_l3641_364195


namespace NUMINAMATH_CALUDE_square_property_implies_zero_l3641_364132

theorem square_property_implies_zero (a b : ℤ) : 
  (∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_square_property_implies_zero_l3641_364132


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3641_364126

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed_calculation (bridge_length : ℝ) (train_length : ℝ) (time : ℝ) : 
  bridge_length = 650 →
  train_length = 200 →
  time = 17 →
  (bridge_length + train_length) / time = 50 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l3641_364126


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3641_364152

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3641_364152


namespace NUMINAMATH_CALUDE_rival_awards_count_l3641_364128

/-- The number of awards won by Scott -/
def scott_awards : ℕ := 4

/-- The number of awards won by Jessie relative to Scott -/
def jessie_multiplier : ℕ := 3

/-- The number of awards won by the rival relative to Jessie -/
def rival_multiplier : ℕ := 2

/-- The number of awards won by the rival -/
def rival_awards : ℕ := rival_multiplier * (jessie_multiplier * scott_awards)

theorem rival_awards_count : rival_awards = 24 := by
  sorry

end NUMINAMATH_CALUDE_rival_awards_count_l3641_364128


namespace NUMINAMATH_CALUDE_correct_bonus_distribution_l3641_364175

/-- Represents the bonus distribution problem for a corporation --/
def BonusDistribution (total : ℕ) (a b c d e f : ℕ) : Prop :=
  -- Total bonus is $25,000
  total = 25000 ∧
  -- A receives twice the amount of B
  a = 2 * b ∧
  -- B and C receive the same amount
  b = c ∧
  -- D receives $1,500 less than A
  d = a - 1500 ∧
  -- E receives $2,000 more than C
  e = c + 2000 ∧
  -- F receives half of the total amount received by A and D combined
  f = (a + d) / 2 ∧
  -- The sum of all amounts equals the total bonus
  a + b + c + d + e + f = total

/-- Theorem stating the correct distribution of the bonus --/
theorem correct_bonus_distribution :
  BonusDistribution 25000 4950 2475 2475 3450 4475 4200 := by
  sorry

#check correct_bonus_distribution

end NUMINAMATH_CALUDE_correct_bonus_distribution_l3641_364175


namespace NUMINAMATH_CALUDE_sin_cos_sum_zero_l3641_364122

theorem sin_cos_sum_zero : 
  Real.sin (35 * π / 6) + Real.cos (-11 * π / 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_zero_l3641_364122


namespace NUMINAMATH_CALUDE_max_distance_circle_C_to_line_L_l3641_364171

/-- Circle C with equation x^2 + y^2 - 4x + m = 0 -/
def circle_C (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 + m = 0}

/-- Circle D with equation (x-3)^2 + (y+2√2)^2 = 4 -/
def circle_D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1-3)^2 + (p.2+2*Real.sqrt 2)^2 = 4}

/-- Line L with equation 3x - 4y + 4 = 0 -/
def line_L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3*p.1 - 4*p.2 + 4 = 0}

/-- The distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (L : Set (ℝ × ℝ)) : ℝ := sorry

/-- The maximum distance from any point on a set to a line -/
def max_distance_set_to_line (S : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) : ℝ := sorry

theorem max_distance_circle_C_to_line_L :
  ∃ m : ℝ,
    (circle_C m).Nonempty ∧
    (∃ p : ℝ × ℝ, p ∈ circle_C m ∧ p ∈ circle_D) ∧
    max_distance_set_to_line (circle_C m) line_L = 3 := by sorry

end NUMINAMATH_CALUDE_max_distance_circle_C_to_line_L_l3641_364171


namespace NUMINAMATH_CALUDE_equation_solution_l3641_364168

theorem equation_solution (a b : ℝ) (h : a * 3^2 - b * 3 = 6) : 2023 - 6 * a + 2 * b = 2019 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3641_364168


namespace NUMINAMATH_CALUDE_intersection_A_B_l3641_364129

-- Define set A
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {2, 3, 4, 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3641_364129


namespace NUMINAMATH_CALUDE_expected_value_max_two_dice_rolls_expected_value_max_two_dice_rolls_eq_l3641_364199

/-- The expected value of the maximum of two independent rolls of a fair six-sided die -/
theorem expected_value_max_two_dice_rolls : ℝ :=
  let X : Fin 6 → ℝ := λ i => (i : ℝ) + 1
  let P : Fin 6 → ℝ := λ i =>
    match i with
    | 0 => 1 / 36
    | 1 => 3 / 36
    | 2 => 5 / 36
    | 3 => 7 / 36
    | 4 => 9 / 36
    | 5 => 11 / 36
  161 / 36

/-- The expected value of the maximum of two independent rolls of a fair six-sided die is 161/36 -/
theorem expected_value_max_two_dice_rolls_eq : expected_value_max_two_dice_rolls = 161 / 36 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_max_two_dice_rolls_expected_value_max_two_dice_rolls_eq_l3641_364199


namespace NUMINAMATH_CALUDE_shoe_selection_problem_l3641_364120

theorem shoe_selection_problem (n : ℕ) (h : n = 10) : 
  (n.choose 1) * ((n - 1).choose 2) * (2^2) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_shoe_selection_problem_l3641_364120


namespace NUMINAMATH_CALUDE_student_average_less_than_actual_average_l3641_364165

theorem student_average_less_than_actual_average (a b c : ℝ) (h : a < b ∧ b < c) :
  (a + (b + c) / 2) / 2 < (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_average_less_than_actual_average_l3641_364165


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3641_364187

theorem z_in_first_quadrant :
  ∀ (z : ℂ), (z - Complex.I) * (2 - Complex.I) = 5 →
  ∃ (a b : ℝ), z = Complex.mk a b ∧ a > 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3641_364187


namespace NUMINAMATH_CALUDE_sticker_ratio_l3641_364114

/-- Proves that the ratio of silver stickers to gold stickers is 2:1 --/
theorem sticker_ratio :
  ∀ (gold silver bronze : ℕ),
  gold = 50 →
  bronze = silver - 20 →
  gold + silver + bronze = 5 * 46 →
  silver / gold = 2 := by
  sorry

end NUMINAMATH_CALUDE_sticker_ratio_l3641_364114


namespace NUMINAMATH_CALUDE_arkos_population_2070_l3641_364161

def population_growth (initial_population : ℕ) (growth_factor : ℕ) (years : ℕ) : ℕ :=
  initial_population * growth_factor ^ (years / 10)

theorem arkos_population_2070 :
  let initial_population := 250
  let years := 50
  let growth_factor := 2
  population_growth initial_population growth_factor years = 8000 := by
sorry

end NUMINAMATH_CALUDE_arkos_population_2070_l3641_364161


namespace NUMINAMATH_CALUDE_triangle_problem_l3641_364157

theorem triangle_problem (A B C : Real) (a b c : Real) :
  a + c = 5 →
  a > c →
  b = 3 →
  Real.cos B = 1/3 →
  a = 3 ∧ c = 2 ∧ Real.cos (A + B) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3641_364157


namespace NUMINAMATH_CALUDE_number_wall_m_equals_one_l3641_364188

/-- Represents a simplified version of the number wall structure -/
structure NumberWall where
  m : ℤ
  top : ℤ
  left : ℤ
  right : ℤ

/-- The number wall satisfies the given conditions -/
def valid_wall (w : NumberWall) : Prop :=
  w.top = w.left + w.right ∧ w.left = w.m + 22 ∧ w.right = 35 ∧ w.top = 58

/-- Theorem: In the given number wall structure, m = 1 -/
theorem number_wall_m_equals_one (w : NumberWall) (h : valid_wall w) : w.m = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_m_equals_one_l3641_364188


namespace NUMINAMATH_CALUDE_sqrt_two_squared_inverse_half_l3641_364172

theorem sqrt_two_squared_inverse_half : 
  (((-Real.sqrt 2)^2)^(-1/2 : ℝ)) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_inverse_half_l3641_364172


namespace NUMINAMATH_CALUDE_complex_number_with_conditions_l3641_364180

theorem complex_number_with_conditions (z : ℂ) :
  (((1 : ℂ) + 2 * Complex.I) * z).im = 0 →
  Complex.abs z = Real.sqrt 5 →
  z = 1 - 2 * Complex.I ∨ z = -1 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_with_conditions_l3641_364180


namespace NUMINAMATH_CALUDE_blue_lipstick_count_l3641_364151

theorem blue_lipstick_count (total_students : ℕ) 
  (h_total : total_students = 300)
  (h_lipstick : ∃ lipstick_wearers : ℕ, lipstick_wearers = total_students / 2)
  (h_red : ∃ red_wearers : ℕ, red_wearers = lipstick_wearers / 4)
  (h_pink : ∃ pink_wearers : ℕ, pink_wearers = lipstick_wearers / 3)
  (h_purple : ∃ purple_wearers : ℕ, purple_wearers = lipstick_wearers / 6)
  (h_blue : ∃ blue_wearers : ℕ, blue_wearers = lipstick_wearers - (red_wearers + pink_wearers + purple_wearers)) :
  blue_wearers = 37 := by
sorry

end NUMINAMATH_CALUDE_blue_lipstick_count_l3641_364151


namespace NUMINAMATH_CALUDE_kite_area_l3641_364109

/-- The area of a kite composed of two identical triangles -/
theorem kite_area (base height : ℝ) (h1 : base = 14) (h2 : height = 6) :
  2 * (1/2 * base * height) = 84 := by
  sorry

end NUMINAMATH_CALUDE_kite_area_l3641_364109


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l3641_364112

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem fourth_term_of_sequence (x : ℝ) :
  let a₁ := x
  let a₂ := 3 * x + 6
  let a₃ := 7 * x + 21
  (∃ r : ℝ, r ≠ 0 ∧ 
    geometric_sequence a₁ r 2 = a₂ ∧ 
    geometric_sequence a₁ r 3 = a₃) →
  geometric_sequence a₁ ((3 * x + 6) / x) 4 = 220.5 :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l3641_364112


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3641_364127

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x - 1 ≥ 0}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (univ \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3641_364127


namespace NUMINAMATH_CALUDE_even_sum_probability_l3641_364135

theorem even_sum_probability (p_even_1 p_odd_1 p_even_2 p_odd_2 : ℝ) :
  p_even_1 = 1/2 →
  p_odd_1 = 1/2 →
  p_even_2 = 1/5 →
  p_odd_2 = 4/5 →
  p_even_1 + p_odd_1 = 1 →
  p_even_2 + p_odd_2 = 1 →
  p_even_1 * p_even_2 + p_odd_1 * p_odd_2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_even_sum_probability_l3641_364135


namespace NUMINAMATH_CALUDE_purple_balls_count_l3641_364118

theorem purple_balls_count (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 60 + purple)
  (h2 : white = 22)
  (h3 : green = 18)
  (h4 : yellow = 17)
  (h5 : red = 3)
  (h6 : (white + green + yellow : ℚ) / total = 95/100) : 
  purple = 0 := by
sorry

end NUMINAMATH_CALUDE_purple_balls_count_l3641_364118


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3641_364178

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions : 
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 8 := by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3641_364178


namespace NUMINAMATH_CALUDE_vector_problem_l3641_364192

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem vector_problem (a : ℝ × ℝ) :
  collinear a (1, -2) →
  a.1 * 1 + a.2 * (-2) = -10 →
  a = (-2, 4) ∧ Real.sqrt ((a.1 + 6)^2 + (a.2 - 7)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3641_364192


namespace NUMINAMATH_CALUDE_multiply_three_a_two_ab_l3641_364191

theorem multiply_three_a_two_ab (a b : ℝ) : 3 * a * (2 * a * b) = 6 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_a_two_ab_l3641_364191


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l3641_364134

theorem gcd_digits_bound (a b : ℕ) (ha : a < 100000) (hb : b < 100000)
  (hlcm : 10000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 100000000) :
  Nat.gcd a b < 1000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l3641_364134


namespace NUMINAMATH_CALUDE_sin_five_half_pi_plus_alpha_l3641_364110

theorem sin_five_half_pi_plus_alpha (α : ℝ) : 
  Real.sin ((5 / 2) * Real.pi + α) = Real.cos α := by sorry

end NUMINAMATH_CALUDE_sin_five_half_pi_plus_alpha_l3641_364110


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l3641_364141

theorem smallest_common_multiple_of_6_and_15 (a : ℕ) :
  (a > 0 ∧ 6 ∣ a ∧ 15 ∣ a) → a ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l3641_364141


namespace NUMINAMATH_CALUDE_decagon_diagonals_l3641_364124

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- Formula for the number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3641_364124


namespace NUMINAMATH_CALUDE_sphere_triangle_distance_l3641_364100

/-- The distance between the center of a sphere and the plane of a tangent isosceles triangle -/
theorem sphere_triangle_distance (r : ℝ) (a b : ℝ) (h_sphere : r = 8) 
  (h_triangle : a = 13 ∧ b = 10) (h_isosceles : a ≠ b) (h_tangent : True) :
  ∃ (d : ℝ), d = (20 * Real.sqrt 7) / 3 ∧ 
  d^2 = r^2 - (b/2)^2 / (1 - (b/(2*a))^2) := by
  sorry

end NUMINAMATH_CALUDE_sphere_triangle_distance_l3641_364100


namespace NUMINAMATH_CALUDE_meaningful_expression_l3641_364193

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = x / Real.sqrt (x - 1)) ↔ x > 1 := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3641_364193


namespace NUMINAMATH_CALUDE_albert_earnings_increase_l3641_364104

theorem albert_earnings_increase (E : ℝ) (P : ℝ) 
  (h1 : E * (1 + P) = 693)
  (h2 : E * 1.20 = 660) :
  P = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_albert_earnings_increase_l3641_364104


namespace NUMINAMATH_CALUDE_propositions_truth_values_l3641_364147

theorem propositions_truth_values :
  (∃ a b : ℝ, a + b < 2 * Real.sqrt (a * b)) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (1/x + 9/y = 1) ∧ (x + y < 16)) ∧
  (∀ x : ℝ, x^2 + 4/x^2 ≥ 4) ∧
  (∀ a b : ℝ, (a * b > 0) → (b/a + a/b ≥ 2)) :=
by sorry


end NUMINAMATH_CALUDE_propositions_truth_values_l3641_364147


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l3641_364146

theorem abs_inequality_solution_set (x : ℝ) : 
  (|x - 2| < 1) ↔ (1 < x ∧ x < 3) :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l3641_364146


namespace NUMINAMATH_CALUDE_amelia_win_probability_l3641_364194

/-- The probability of Amelia winning the coin toss game -/
def ameliaWinProbability (ameliaHeadProb blainHeadProb : ℚ) : ℚ :=
  ameliaHeadProb / (1 - (1 - ameliaHeadProb) * (1 - blainHeadProb))

/-- The coin toss game where Amelia goes first -/
theorem amelia_win_probability :
  let ameliaHeadProb : ℚ := 1/3
  let blaineHeadProb : ℚ := 2/5
  ameliaWinProbability ameliaHeadProb blaineHeadProb = 5/9 := by
  sorry

#eval ameliaWinProbability (1/3) (2/5)

end NUMINAMATH_CALUDE_amelia_win_probability_l3641_364194


namespace NUMINAMATH_CALUDE_paper_products_distribution_l3641_364160

theorem paper_products_distribution (total : ℕ) 
  (h1 : total = 20)
  (h2 : total / 2 + total / 4 + total / 5 < total) :
  total - (total / 2 + total / 4 + total / 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_paper_products_distribution_l3641_364160


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3641_364136

/-- Given a quadratic function f(x) = ax^2 - x + c with range [0, +∞),
    the minimum value of 2/a + 2/c is 8 -/
theorem min_value_sum_reciprocals (a c : ℝ) : 
  (∀ x, ∃ y ≥ 0, y = a * x^2 - x + c) →
  (∃ x, a * x^2 - x + c = 0) →
  (∀ x, a * x^2 - x + c ≥ 0) →
  (2 / a + 2 / c ≥ 8) ∧ (∃ a c, 2 / a + 2 / c = 8) := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3641_364136


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l3641_364111

/-- Represents a square tile pattern -/
structure TilePattern where
  blackTiles : ℕ
  whiteTiles : ℕ

/-- Extends a tile pattern by adding two borders of black tiles -/
def extendPattern (pattern : TilePattern) : TilePattern :=
  let side := (pattern.blackTiles + pattern.whiteTiles).sqrt
  let newBlackTiles := pattern.blackTiles + 4 * side + 4
  { blackTiles := newBlackTiles, whiteTiles := pattern.whiteTiles }

/-- The ratio of black to white tiles in a pattern -/
def tileRatio (pattern : TilePattern) : ℚ :=
  pattern.blackTiles / pattern.whiteTiles

theorem extended_pattern_ratio :
  let original := TilePattern.mk 10 26
  let extended := extendPattern original
  tileRatio extended = 37 / 13 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l3641_364111


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3641_364177

/-- 
For an infinite geometric series with common ratio r and sum S, 
the first term a is given by the formula: a = S * (1 - r)
-/
def first_term_infinite_geometric_series (r : ℚ) (S : ℚ) : ℚ := S * (1 - r)

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -1/3) (h2 : S = 18) : 
  first_term_infinite_geometric_series r S = 24 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3641_364177


namespace NUMINAMATH_CALUDE_square_sum_equality_l3641_364169

theorem square_sum_equality (x y : ℝ) 
  (h1 : x + 3 * y = 3) 
  (h2 : x * y = -3) : 
  x^2 + 9 * y^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3641_364169


namespace NUMINAMATH_CALUDE_percentage_problem_l3641_364121

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.15 * 1500 - 15 → x = 840 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3641_364121


namespace NUMINAMATH_CALUDE_intersection_M_N_l3641_364186

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3641_364186


namespace NUMINAMATH_CALUDE_square_root_squared_l3641_364182

theorem square_root_squared (x : ℝ) (h : x ≥ 0) : (Real.sqrt x) ^ 2 = x := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_l3641_364182


namespace NUMINAMATH_CALUDE_polynomial_divisibility_implies_coefficients_l3641_364142

theorem polynomial_divisibility_implies_coefficients
  (p q : ℤ)
  (h : ∀ x : ℝ, (x + 2) * (x - 1) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x + 4)) :
  p = -7 ∧ q = -12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_implies_coefficients_l3641_364142


namespace NUMINAMATH_CALUDE_sequence_equals_primes_l3641_364150

theorem sequence_equals_primes (a p : ℕ → ℕ) :
  (∀ n, 0 < a n) →
  (∀ n k, n < k → a n < a k) →
  (∀ n, Nat.Prime (p n)) →
  (∀ n, p n ∣ a n) →
  (∀ n k, a n - a k = p n - p k) →
  ∀ n, a n = p n :=
by sorry

end NUMINAMATH_CALUDE_sequence_equals_primes_l3641_364150


namespace NUMINAMATH_CALUDE_binomial_and_permutation_7_5_l3641_364139

theorem binomial_and_permutation_7_5 :
  (Nat.choose 7 5 = 21) ∧ (Nat.factorial 7 / Nat.factorial 2 = 2520) := by
  sorry

end NUMINAMATH_CALUDE_binomial_and_permutation_7_5_l3641_364139


namespace NUMINAMATH_CALUDE_isosceles_triangle_dimensions_l3641_364174

/-- An isosceles triangle with equal sides of length x and base of length y,
    where a median on one of the equal sides divides the perimeter into parts of 6 and 12 -/
structure IsoscelesTriangle where
  x : ℝ  -- Length of equal sides
  y : ℝ  -- Length of base
  perimeter_division : x + x/2 = 6 ∧ y/2 + y = 12

theorem isosceles_triangle_dimensions (t : IsoscelesTriangle) : t.x = 8 ∧ t.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_dimensions_l3641_364174


namespace NUMINAMATH_CALUDE_derivative_of_exponential_l3641_364166

variable (a : ℝ) (ha : a > 0)

theorem derivative_of_exponential (x : ℝ) : 
  deriv (fun x => a^x) x = a^x * Real.log a := by sorry

end NUMINAMATH_CALUDE_derivative_of_exponential_l3641_364166


namespace NUMINAMATH_CALUDE_modulus_of_z_is_one_l3641_364105

theorem modulus_of_z_is_one (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) :
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_is_one_l3641_364105


namespace NUMINAMATH_CALUDE_food_to_budget_ratio_l3641_364155

def budget : ℚ := 3000
def supplies_fraction : ℚ := 1/4
def wages : ℚ := 1250

def food_expense : ℚ := budget - supplies_fraction * budget - wages

theorem food_to_budget_ratio :
  food_expense / budget = 1/3 := by sorry

end NUMINAMATH_CALUDE_food_to_budget_ratio_l3641_364155


namespace NUMINAMATH_CALUDE_basketball_tournament_wins_losses_l3641_364170

theorem basketball_tournament_wins_losses 
  (total_games : ℕ) 
  (points_per_win : ℕ) 
  (points_per_loss : ℕ) 
  (total_points : ℕ) 
  (h1 : total_games = 15) 
  (h2 : points_per_win = 3) 
  (h3 : points_per_loss = 1) 
  (h4 : total_points = 41) : 
  ∃ (wins losses : ℕ), 
    wins + losses = total_games ∧ 
    wins * points_per_win + losses * points_per_loss = total_points ∧ 
    wins = 13 ∧ 
    losses = 2 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tournament_wins_losses_l3641_364170


namespace NUMINAMATH_CALUDE_equation_solution_l3641_364183

theorem equation_solution : 
  ∃ x₁ x₂ : ℚ, (x₁ = 5/2 ∧ x₂ = -1/2) ∧ 
  (∀ x : ℚ, 4 * (x - 1)^2 = 9 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3641_364183


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_cube_l3641_364145

theorem cube_sum_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^3 + 1/r^3 = 2 * Real.sqrt 5 ∨ r^3 + 1/r^3 = -2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_cube_l3641_364145


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l3641_364176

theorem quadratic_solution_property (f g : ℝ) : 
  (3 * f^2 - 4 * f + 2 = 0) →
  (3 * g^2 - 4 * g + 2 = 0) →
  (f + 2) * (g + 2) = 22/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l3641_364176


namespace NUMINAMATH_CALUDE_cubic_equation_value_l3641_364148

theorem cubic_equation_value (x : ℝ) (h : x^2 + x - 2 = 0) :
  x^3 + 2*x^2 - x + 2021 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l3641_364148


namespace NUMINAMATH_CALUDE_unique_solution_l3641_364116

theorem unique_solution (x y : ℝ) : 
  x * (x + y)^2 = 9 ∧ x * (y^3 - x^3) = 7 → x = 1 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3641_364116


namespace NUMINAMATH_CALUDE_one_seventh_difference_l3641_364197

theorem one_seventh_difference : ∃ (ε : ℚ), 1/7 - 0.14285714285 = ε ∧ ε > 0 ∧ ε < 1/(7*10^10) := by
  sorry

end NUMINAMATH_CALUDE_one_seventh_difference_l3641_364197


namespace NUMINAMATH_CALUDE_average_difference_implies_unknown_l3641_364198

theorem average_difference_implies_unknown (x : ℝ) : 
  let set1 := [20, 40, 60]
  let set2 := [10, x, 15]
  (set1.sum / set1.length) = (set2.sum / set2.length) + 5 →
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_average_difference_implies_unknown_l3641_364198


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3641_364115

theorem unique_positive_solution (x y z : ℝ) : 
  x > 0 →
  x * y + 3 * x + 2 * y = 12 →
  y * z + 5 * y + 3 * z = 18 →
  x * z + 2 * x + 3 * z = 18 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3641_364115


namespace NUMINAMATH_CALUDE_store_annual_profits_l3641_364156

/-- Calculates the annual profits given the profits for each quarter -/
def annual_profits (q1 q2 q3 q4 : ℕ) : ℕ :=
  q1 + q2 + q3 + q4

/-- Theorem stating that the annual profits are $8,000 given the quarterly profits -/
theorem store_annual_profits :
  let q1 : ℕ := 1500
  let q2 : ℕ := 1500
  let q3 : ℕ := 3000
  let q4 : ℕ := 2000
  annual_profits q1 q2 q3 q4 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_store_annual_profits_l3641_364156


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l3641_364162

theorem binomial_expansion_example : 57^3 + 3*(57^2)*4 + 3*57*(4^2) + 4^3 = 226981 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l3641_364162


namespace NUMINAMATH_CALUDE_common_solution_y_value_l3641_364131

theorem common_solution_y_value : ∃! y : ℝ, ∃ x : ℝ, 
  (x^2 + y^2 - 4 = 0) ∧ (x^2 - 4*y + 8 = 0) :=
by
  -- Proof goes here
  sorry

#check common_solution_y_value

end NUMINAMATH_CALUDE_common_solution_y_value_l3641_364131


namespace NUMINAMATH_CALUDE_memory_card_picture_size_l3641_364173

theorem memory_card_picture_size 
  (total_pictures_a : ℕ) 
  (size_a : ℕ) 
  (total_pictures_b : ℕ) 
  (h1 : total_pictures_a = 3000)
  (h2 : size_a = 8)
  (h3 : total_pictures_b = 4000) :
  (total_pictures_a * size_a) / total_pictures_b = 6 :=
by sorry

end NUMINAMATH_CALUDE_memory_card_picture_size_l3641_364173


namespace NUMINAMATH_CALUDE_currency_notes_problem_l3641_364107

/-- Given a total amount and an amount in 50-rupee notes, calculates the total number of notes -/
def totalNotes (totalAmount : ℕ) (amount50 : ℕ) : ℕ :=
  (amount50 / 50) + ((totalAmount - amount50) / 100)

/-- Theorem stating that given the problem conditions, the total number of notes is 85 -/
theorem currency_notes_problem (totalAmount amount50 : ℕ) 
    (h1 : totalAmount = 5000)
    (h2 : amount50 = 3500) :
  totalNotes totalAmount amount50 = 85 := by
  sorry

end NUMINAMATH_CALUDE_currency_notes_problem_l3641_364107


namespace NUMINAMATH_CALUDE_sequence_sum_l3641_364144

theorem sequence_sum (A B C D E F G H : ℝ) 
  (h1 : C = 7)
  (h2 : ∀ (X Y Z : ℝ), (X = A ∧ Y = B ∧ Z = C) ∨ 
                        (X = B ∧ Y = C ∧ Z = D) ∨ 
                        (X = C ∧ Y = D ∧ Z = E) ∨ 
                        (X = D ∧ Y = E ∧ Z = F) ∨ 
                        (X = E ∧ Y = F ∧ Z = G) ∨ 
                        (X = F ∧ Y = G ∧ Z = H) → X + Y + Z = 36) : 
  A + H = 29 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l3641_364144


namespace NUMINAMATH_CALUDE_base_conversion_l3641_364108

/-- Given that the decimal number 26 converted to base r is 32, prove that r = 8 -/
theorem base_conversion (r : ℕ) (h : r > 1) : 
  (26 : ℕ).digits r = [3, 2] → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l3641_364108


namespace NUMINAMATH_CALUDE_x_plus_y_equals_48_l3641_364103

-- Define the arithmetic sequence
def arithmetic_sequence : List ℝ := [3, 9, 15, 33]

-- Define x and y as the last two terms before 33
def x : ℝ := arithmetic_sequence[arithmetic_sequence.length - 3]
def y : ℝ := arithmetic_sequence[arithmetic_sequence.length - 2]

-- Theorem to prove
theorem x_plus_y_equals_48 : x + y = 48 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_48_l3641_364103


namespace NUMINAMATH_CALUDE_train_length_problem_l3641_364163

/-- Proves that the length of each train is 25 meters given the specified conditions -/
theorem train_length_problem (speed_fast speed_slow : ℝ) (passing_time : ℝ) :
  speed_fast = 46 →
  speed_slow = 36 →
  passing_time = 18 →
  let relative_speed := (speed_fast - speed_slow) * (5 / 18)
  let train_length := (relative_speed * passing_time) / 2
  train_length = 25 := by
sorry


end NUMINAMATH_CALUDE_train_length_problem_l3641_364163


namespace NUMINAMATH_CALUDE_connie_marbles_to_juan_l3641_364158

/-- Represents the number of marbles Connie gave to Juan -/
def marbles_given_to_juan (initial_marbles : ℕ) (remaining_marbles : ℕ) : ℕ :=
  initial_marbles - remaining_marbles

/-- Proves that Connie gave 73 marbles to Juan -/
theorem connie_marbles_to_juan :
  marbles_given_to_juan 143 70 = 73 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_to_juan_l3641_364158


namespace NUMINAMATH_CALUDE_find_p_value_l3641_364154

theorem find_p_value (a b c p : ℝ) 
  (h1 : 9 / (a + b) = p / (a + c)) 
  (h2 : p / (a + c) = 13 / (c - b)) : 
  p = 22 := by
sorry

end NUMINAMATH_CALUDE_find_p_value_l3641_364154


namespace NUMINAMATH_CALUDE_angle_negative_1120_in_fourth_quadrant_l3641_364137

def angle_to_standard_form (angle : ℤ) : ℤ :=
  angle % 360

def quadrant (angle : ℤ) : ℕ :=
  let standard_angle := angle_to_standard_form angle
  if 0 ≤ standard_angle ∧ standard_angle < 90 then 1
  else if 90 ≤ standard_angle ∧ standard_angle < 180 then 2
  else if 180 ≤ standard_angle ∧ standard_angle < 270 then 3
  else 4

theorem angle_negative_1120_in_fourth_quadrant :
  quadrant (-1120) = 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_negative_1120_in_fourth_quadrant_l3641_364137


namespace NUMINAMATH_CALUDE_parabola_equation_and_max_area_l3641_364181

-- Define the parabola
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

-- Define a point on the parabola
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : c.equation x y

-- Define the vector from focus to a point
def vector_from_focus (c : Parabola) (p : PointOnParabola c) : ℝ × ℝ :=
  (p.x - c.focus.1, p.y - c.focus.2)

theorem parabola_equation_and_max_area 
  (c : Parabola)
  (h_focus : c.focus = (0, 1))
  (h_equation : ∀ x y, c.equation x y ↔ x^2 = 2 * c.p * y)
  (A B C : PointOnParabola c)
  (h_vector_sum : vector_from_focus c A + vector_from_focus c B + vector_from_focus c C = (0, 0)) :
  (∀ x y, c.equation x y ↔ x^2 = 4 * y) ∧
  (∃ (max_area : ℝ), max_area = (3 * Real.sqrt 6) / 2 ∧
    ∀ (area : ℝ), area = abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2 →
      area ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_and_max_area_l3641_364181


namespace NUMINAMATH_CALUDE_equal_sum_groups_l3641_364185

/-- A function that checks if a list of natural numbers can be divided into three groups with equal sums -/
def canDivideIntoThreeEqualGroups (list : List Nat) : Prop :=
  ∃ (g1 g2 g3 : List Nat), 
    g1 ++ g2 ++ g3 = list ∧ 
    g1.sum = g2.sum ∧ 
    g2.sum = g3.sum

/-- The list of natural numbers from 1 to n -/
def naturalNumbersUpTo (n : Nat) : List Nat :=
  List.range n |>.map (· + 1)

/-- The main theorem stating the condition for when the natural numbers up to n can be divided into three groups with equal sums -/
theorem equal_sum_groups (n : Nat) : 
  canDivideIntoThreeEqualGroups (naturalNumbersUpTo n) ↔ 
  (∃ k : Nat, (k ≥ 2 ∧ (n = 3 * k ∨ n = 3 * k - 1))) :=
sorry

end NUMINAMATH_CALUDE_equal_sum_groups_l3641_364185


namespace NUMINAMATH_CALUDE_chinese_spanish_difference_l3641_364159

def hours_english : ℕ := 2
def hours_chinese : ℕ := 5
def hours_spanish : ℕ := 4

theorem chinese_spanish_difference : hours_chinese - hours_spanish = 1 := by
  sorry

end NUMINAMATH_CALUDE_chinese_spanish_difference_l3641_364159


namespace NUMINAMATH_CALUDE_border_area_calculation_l3641_364138

/-- Given a rectangular photograph and its frame, calculate the area of the border. -/
theorem border_area_calculation (photo_height photo_width border_width : ℝ) 
  (h1 : photo_height = 12)
  (h2 : photo_width = 15)
  (h3 : border_width = 3) :
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 198 := by
  sorry

#check border_area_calculation

end NUMINAMATH_CALUDE_border_area_calculation_l3641_364138


namespace NUMINAMATH_CALUDE_function_max_min_sum_l3641_364117

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (Real.log x) / (Real.log a)

theorem function_max_min_sum (a : ℝ) :
  a > 0 ∧ a ≠ 1 →
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 2, f a x ≤ max) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = max) ∧
    (∀ x ∈ Set.Icc 1 2, min ≤ f a x) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = min) ∧
    max + min = (Real.log 2) / (Real.log a) + 6) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_function_max_min_sum_l3641_364117


namespace NUMINAMATH_CALUDE_gemma_pizza_change_l3641_364153

def pizza_order (num_pizzas : ℕ) (price_per_pizza : ℕ) (tip : ℕ) (payment : ℕ) : ℕ :=
  payment - (num_pizzas * price_per_pizza + tip)

theorem gemma_pizza_change : pizza_order 4 10 5 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gemma_pizza_change_l3641_364153


namespace NUMINAMATH_CALUDE_function_always_positive_l3641_364184

theorem function_always_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, 2 * f x + x * deriv f x > 0) : 
  ∀ x, f x > 0 := by sorry

end NUMINAMATH_CALUDE_function_always_positive_l3641_364184


namespace NUMINAMATH_CALUDE_symmetric_point_of_A_l3641_364123

def line_equation (x y : ℝ) : Prop := 2*x - 4*y + 9 = 0

def is_symmetric_point (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- The line connecting the points is perpendicular to the given line
  (y₂ - y₁) / (x₂ - x₁) = -1 / (1/2) ∧
  -- The midpoint of the two points lies on the given line
  line_equation ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

theorem symmetric_point_of_A : is_symmetric_point 2 2 1 4 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_of_A_l3641_364123


namespace NUMINAMATH_CALUDE_integer_puzzle_l3641_364113

theorem integer_puzzle (x y : ℕ+) (h1 : x + y = 60) (h2 : x - y = 16) :
  x^2 - y^2 = 960 ∧ x * y = 836 := by
  sorry

end NUMINAMATH_CALUDE_integer_puzzle_l3641_364113


namespace NUMINAMATH_CALUDE_composite_function_solution_l3641_364179

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := x^2 + 5
def g (x : ℝ) : ℝ := x^2 - 3
def h (x : ℝ) : ℝ := 2*x + 1

-- State the theorem
theorem composite_function_solution (a : ℝ) (ha : a > 0) 
  (h_eq : f (g (h a)) = 17) : 
  a = (-1 + Real.sqrt (3 + 2 * Real.sqrt 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_solution_l3641_364179


namespace NUMINAMATH_CALUDE_relationship_equation_l3641_364196

/-- Given a relationship "a number that is 3 more than half of x is equal to twice y",
    prove that the equation (1/2)x + 3 = 2y correctly represents this relationship. -/
theorem relationship_equation (x y : ℝ) :
  (∃ (n : ℝ), n = (1/2) * x + 3 ∧ n = 2 * y) ↔ (1/2) * x + 3 = 2 * y :=
by sorry

end NUMINAMATH_CALUDE_relationship_equation_l3641_364196


namespace NUMINAMATH_CALUDE_semicircle_radius_l3641_364149

theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 162) :
  ∃ (radius : ℝ), perimeter = radius * (Real.pi + 2) ∧ radius = 162 / (Real.pi + 2) := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l3641_364149


namespace NUMINAMATH_CALUDE_existence_implies_upper_bound_l3641_364130

theorem existence_implies_upper_bound (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x - a ≥ 0) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_existence_implies_upper_bound_l3641_364130


namespace NUMINAMATH_CALUDE_second_student_wrong_answers_second_student_wrong_answers_value_l3641_364125

theorem second_student_wrong_answers 
  (total_questions : Nat) 
  (hannah_correct : Nat) 
  (hannah_highest_score : Bool) : Nat :=
  let second_student_correct := hannah_correct - 1
  let second_student_wrong := total_questions - second_student_correct
  second_student_wrong

#check second_student_wrong_answers

theorem second_student_wrong_answers_value :
  second_student_wrong_answers 40 39 true = 2 := by sorry

end NUMINAMATH_CALUDE_second_student_wrong_answers_second_student_wrong_answers_value_l3641_364125


namespace NUMINAMATH_CALUDE_excluded_angle_measure_l3641_364140

/-- Given a polygon where the sum of all but one interior angle is 1680°,
    prove that the measure of the excluded interior angle is 120°. -/
theorem excluded_angle_measure (n : ℕ) (h : n ≥ 3) :
  let sum_interior := (n - 2) * 180
  let sum_except_one := 1680
  sum_interior - sum_except_one = 120 := by
  sorry

end NUMINAMATH_CALUDE_excluded_angle_measure_l3641_364140


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l3641_364167

theorem arithmetic_square_root_of_four : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l3641_364167


namespace NUMINAMATH_CALUDE_set_equivalence_l3641_364119

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem set_equivalence : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equivalence_l3641_364119
