import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_length_l2069_206998

theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 12 →
  rect_width = 6 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 24 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l2069_206998


namespace NUMINAMATH_CALUDE_apple_sellers_average_prices_l2069_206924

/-- Represents the sales data for a fruit seller --/
structure FruitSeller where
  morning_price : ℚ
  afternoon_price : ℚ
  morning_quantity : ℚ
  afternoon_quantity : ℚ

/-- Calculates the average price per apple for a fruit seller --/
def average_price (seller : FruitSeller) : ℚ :=
  (seller.morning_price * seller.morning_quantity + seller.afternoon_price * seller.afternoon_quantity) /
  (seller.morning_quantity + seller.afternoon_quantity)

theorem apple_sellers_average_prices
  (john bill george : FruitSeller)
  (h_morning_price : john.morning_price = bill.morning_price ∧ bill.morning_price = george.morning_price ∧ george.morning_price = 5/2)
  (h_afternoon_price : john.afternoon_price = bill.afternoon_price ∧ bill.afternoon_price = george.afternoon_price ∧ george.afternoon_price = 5/3)
  (h_john_quantities : john.morning_quantity = john.afternoon_quantity)
  (h_bill_revenue : bill.morning_price * bill.morning_quantity = bill.afternoon_price * bill.afternoon_quantity)
  (h_george_ratio : george.morning_quantity / george.afternoon_quantity = (5/3) / (5/2)) :
  average_price john = 25/12 ∧ average_price bill = 2 ∧ average_price george = 2 := by
  sorry


end NUMINAMATH_CALUDE_apple_sellers_average_prices_l2069_206924


namespace NUMINAMATH_CALUDE_fountain_area_l2069_206910

theorem fountain_area (diameter : Real) (radius : Real) :
  diameter = 20 →
  radius * 2 = diameter →
  radius ^ 2 = 244 →
  π * radius ^ 2 = 244 * π :=
by sorry

end NUMINAMATH_CALUDE_fountain_area_l2069_206910


namespace NUMINAMATH_CALUDE_david_min_score_l2069_206940

def david_scores : List Int := [88, 92, 75, 83, 90]

def current_average : Rat :=
  (david_scores.sum : Rat) / david_scores.length

def target_average : Rat := current_average + 4

def min_score : Int :=
  Int.ceil ((target_average * (david_scores.length + 1) : Rat) - david_scores.sum)

theorem david_min_score :
  min_score = 110 := by sorry

end NUMINAMATH_CALUDE_david_min_score_l2069_206940


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2069_206942

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  b = 6 →
  c = a + 2 →
  c = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2069_206942


namespace NUMINAMATH_CALUDE_distance_is_sqrt_51_l2069_206927

def point : ℝ × ℝ × ℝ := (3, 5, -1)
def line_point : ℝ × ℝ × ℝ := (2, 4, 6)
def line_direction : ℝ × ℝ × ℝ := (4, 3, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_sqrt_51 :
  distance_to_line point line_point line_direction = Real.sqrt 51 :=
sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_51_l2069_206927


namespace NUMINAMATH_CALUDE_john_chores_time_l2069_206907

/-- Calculates the number of minutes of chores John has to do based on his cartoon watching time -/
def chores_minutes (cartoon_hours : ℕ) : ℕ :=
  let cartoon_minutes := cartoon_hours * 60
  let chore_blocks := cartoon_minutes / 10
  chore_blocks * 8

/-- Theorem: John has to do 96 minutes of chores when he watches 2 hours of cartoons -/
theorem john_chores_time : chores_minutes 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_john_chores_time_l2069_206907


namespace NUMINAMATH_CALUDE_two_in_A_implies_a_is_one_or_two_l2069_206931

-- Define the set A
def A (a : ℝ) : Set ℝ := {-2, 2*a, a^2 - a}

-- Theorem statement
theorem two_in_A_implies_a_is_one_or_two :
  ∀ a : ℝ, 2 ∈ A a → a = 1 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_two_in_A_implies_a_is_one_or_two_l2069_206931


namespace NUMINAMATH_CALUDE_x_coordinate_of_Q_l2069_206933

/-- A line through the origin equidistant from two points -/
structure EquidistantLine where
  slope : ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  is_equidistant : ∀ (x y : ℝ), y = slope * x → 
    (x - P.1)^2 + (y - P.2)^2 = (x - Q.1)^2 + (y - Q.2)^2

/-- Theorem: Given the conditions, the x-coordinate of Q is 2.5 -/
theorem x_coordinate_of_Q (L : EquidistantLine) 
  (h_slope : L.slope = 0.8)
  (h_Q_y : L.Q.2 = 2) :
  L.Q.1 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_of_Q_l2069_206933


namespace NUMINAMATH_CALUDE_vowel_word_count_l2069_206995

def vowel_count : ℕ := 5
def word_length : ℕ := 5
def max_vowel_occurrence : ℕ := 3

def total_distributions : ℕ := Nat.choose (word_length + vowel_count - 1) (vowel_count - 1)

def invalid_distributions : ℕ := vowel_count * (vowel_count - 1)

theorem vowel_word_count :
  total_distributions - invalid_distributions = 106 :=
sorry

end NUMINAMATH_CALUDE_vowel_word_count_l2069_206995


namespace NUMINAMATH_CALUDE_range_of_a_l2069_206982

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x - 7 else Real.sqrt x

-- State the theorem
theorem range_of_a (a : ℝ) (h : f a < 1) : -3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2069_206982


namespace NUMINAMATH_CALUDE_shekars_mathematics_marks_l2069_206973

/-- Represents the marks scored by Shekar in different subjects -/
structure Marks where
  mathematics : ℕ
  science : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average of marks -/
def average (m : Marks) : ℚ :=
  (m.mathematics + m.science + m.social_studies + m.english + m.biology) / 5

/-- Theorem stating that Shekar's marks in mathematics are 76 -/
theorem shekars_mathematics_marks :
  ∃ m : Marks,
    m.science = 65 ∧
    m.social_studies = 82 ∧
    m.english = 67 ∧
    m.biology = 75 ∧
    average m = 73 ∧
    m.mathematics = 76 := by
  sorry


end NUMINAMATH_CALUDE_shekars_mathematics_marks_l2069_206973


namespace NUMINAMATH_CALUDE_min_break_even_quantity_l2069_206950

/-- The cost function for a product -/
def cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The revenue function for a product -/
def revenue (x : ℕ) : ℝ := 25 * x

/-- The break-even condition -/
def breaks_even (x : ℕ) : Prop := revenue x ≥ cost x

theorem min_break_even_quantity :
  ∃ (x : ℕ), x > 0 ∧ x < 240 ∧ breaks_even x ∧
  ∀ (y : ℕ), y > 0 ∧ y < 240 ∧ breaks_even y → y ≥ 150 :=
sorry

end NUMINAMATH_CALUDE_min_break_even_quantity_l2069_206950


namespace NUMINAMATH_CALUDE_identity_function_divisibility_l2069_206906

theorem identity_function_divisibility (f : ℕ+ → ℕ+) :
  (∀ a b : ℕ+, (a.val ^ 2 + (f a).val * (f b).val) % ((f a).val + b.val) = 0) →
  (∀ n : ℕ+, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_identity_function_divisibility_l2069_206906


namespace NUMINAMATH_CALUDE_complex_modulus_example_l2069_206947

theorem complex_modulus_example : Complex.abs (-3 - (5/4)*Complex.I) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l2069_206947


namespace NUMINAMATH_CALUDE_computer_B_most_popular_l2069_206960

/-- Represents the sales data for a computer over three years -/
structure ComputerSales where
  year2018 : Nat
  year2019 : Nat
  year2020 : Nat

/-- Checks if the sales are consistently increasing -/
def isConsistentlyIncreasing (sales : ComputerSales) : Prop :=
  sales.year2018 < sales.year2019 ∧ sales.year2019 < sales.year2020

/-- Defines the sales data for computers A, B, and C -/
def computerA : ComputerSales := { year2018 := 600, year2019 := 610, year2020 := 590 }
def computerB : ComputerSales := { year2018 := 590, year2019 := 650, year2020 := 700 }
def computerC : ComputerSales := { year2018 := 650, year2019 := 670, year2020 := 660 }

/-- Theorem: Computer B is the most popular choice -/
theorem computer_B_most_popular :
  isConsistentlyIncreasing computerB ∧
  ¬isConsistentlyIncreasing computerA ∧
  ¬isConsistentlyIncreasing computerC :=
sorry

end NUMINAMATH_CALUDE_computer_B_most_popular_l2069_206960


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l2069_206999

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 9 / Real.log 18 + 1) = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l2069_206999


namespace NUMINAMATH_CALUDE_no_positive_subtraction_l2069_206980

theorem no_positive_subtraction (x : ℝ) : x > 0 → 24 - x ≠ 34 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_subtraction_l2069_206980


namespace NUMINAMATH_CALUDE_product_2000_sum_bounds_l2069_206957

theorem product_2000_sum_bounds (a b c d e : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) (hd : d > 1) (he : e > 1)
  (h_product : a * b * c * d * e = 2000) :
  (∃ (x y z w v : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ w > 1 ∧ v > 1 ∧ 
    x * y * z * w * v = 2000 ∧ x + y + z + w + v = 133) ∧
  (∃ (x y z w v : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ w > 1 ∧ v > 1 ∧ 
    x * y * z * w * v = 2000 ∧ x + y + z + w + v = 23) ∧
  (∀ (x y z w v : ℕ), x > 1 → y > 1 → z > 1 → w > 1 → v > 1 → 
    x * y * z * w * v = 2000 → 23 ≤ x + y + z + w + v ∧ x + y + z + w + v ≤ 133) :=
by sorry

end NUMINAMATH_CALUDE_product_2000_sum_bounds_l2069_206957


namespace NUMINAMATH_CALUDE_line_perp_plane_and_line_parallel_plane_implies_lines_perpendicular_l2069_206911

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_and_line_parallel_plane_implies_lines_perpendicular
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel n α → perpendicularLines m n :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_and_line_parallel_plane_implies_lines_perpendicular_l2069_206911


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2069_206936

theorem absolute_value_inequality (x : ℝ) :
  abs (x - 2) + abs (x + 3) < 8 ↔ x ∈ Set.Ioo (-6.5) 3.5 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2069_206936


namespace NUMINAMATH_CALUDE_M_properties_l2069_206901

def M (n : ℕ+) : ℤ := (-2) ^ n.val

theorem M_properties :
  (M 5 + M 6 = 32) ∧
  (2 * M 2015 + M 2016 = 0) ∧
  (∀ n : ℕ+, 2 * M n + M (n + 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_M_properties_l2069_206901


namespace NUMINAMATH_CALUDE_prob_at_most_two_heads_prove_prob_at_most_two_heads_l2069_206943

/-- The probability of getting at most 2 heads when tossing three unbiased coins -/
theorem prob_at_most_two_heads : ℚ :=
  7 / 8

/-- Prove that the probability of getting at most 2 heads when tossing three unbiased coins is 7/8 -/
theorem prove_prob_at_most_two_heads :
  prob_at_most_two_heads = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_two_heads_prove_prob_at_most_two_heads_l2069_206943


namespace NUMINAMATH_CALUDE_symmetric_parabola_l2069_206925

/-- 
Given a parabola with equation y^2 = 2x and a point (-1, 0),
prove that the equation y^2 = -2(x + 2) represents the parabola 
symmetric to the original parabola with respect to the given point.
-/
theorem symmetric_parabola (x y : ℝ) : 
  (∀ x y, y^2 = 2*x → 
   ∃ x' y', x' = -x - 2 ∧ y' = -y ∧ y'^2 = -2*(x' + 2)) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_parabola_l2069_206925


namespace NUMINAMATH_CALUDE_intern_teacher_distribution_l2069_206989

/-- The number of ways to distribute n teachers among k classes with at least one teacher per class -/
def distribution_schemes (n k : ℕ) : ℕ :=
  if n < k then 0
  else (n - k + 1).choose k * (k - 1).choose (n - k)

/-- Theorem: There are 60 ways to distribute 5 intern teachers among 3 freshman classes with at least 1 teacher in each class -/
theorem intern_teacher_distribution : distribution_schemes 5 3 = 60 := by
  sorry


end NUMINAMATH_CALUDE_intern_teacher_distribution_l2069_206989


namespace NUMINAMATH_CALUDE_M_on_angle_bisector_coordinates_M_distance_to_x_axis_coordinates_l2069_206922

def M (m : ℚ) : ℚ × ℚ := (m - 1, 2 * m + 3)

def on_angle_bisector (p : ℚ × ℚ) : Prop :=
  p.1 = p.2 ∨ p.1 = -p.2

def distance_to_x_axis (p : ℚ × ℚ) : ℚ := |p.2|

theorem M_on_angle_bisector_coordinates (m : ℚ) :
  on_angle_bisector (M m) → M m = (-5/3, 5/3) ∨ M m = (-5, -5) := by sorry

theorem M_distance_to_x_axis_coordinates (m : ℚ) :
  distance_to_x_axis (M m) = 1 → M m = (-2, 1) ∨ M m = (-3, -1) := by sorry

end NUMINAMATH_CALUDE_M_on_angle_bisector_coordinates_M_distance_to_x_axis_coordinates_l2069_206922


namespace NUMINAMATH_CALUDE_sin_squared_minus_cos_squared_range_l2069_206981

/-- 
Given an angle θ in standard position with terminal side passing through (x, y),
prove that sin²θ - cos²θ is between -1 and 1, inclusive.
-/
theorem sin_squared_minus_cos_squared_range (θ x y r : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = r^2) →  -- r is the distance from origin to (x, y)
  r > 0 →  -- r is positive (implicitly given in the problem)
  Real.sin θ = y / r → 
  Real.cos θ = x / r → 
  -1 ≤ Real.sin θ^2 - Real.cos θ^2 ∧ Real.sin θ^2 - Real.cos θ^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_minus_cos_squared_range_l2069_206981


namespace NUMINAMATH_CALUDE_lucy_money_ratio_l2069_206990

/-- Proves that the ratio of money lost to initial amount is 1:3 given the conditions of Lucy's spending --/
theorem lucy_money_ratio (initial_amount : ℝ) (lost_amount : ℝ) (remainder : ℝ) (final_amount : ℝ) :
  initial_amount = 30 →
  remainder = initial_amount - lost_amount →
  final_amount = remainder - (1/4) * remainder →
  final_amount = 15 →
  lost_amount / initial_amount = 1/3 := by
sorry

end NUMINAMATH_CALUDE_lucy_money_ratio_l2069_206990


namespace NUMINAMATH_CALUDE_division_remainder_composition_l2069_206966

theorem division_remainder_composition (P D Q R D' Q' R' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = D' * Q' + R') : 
  ∃ k : ℕ, P = (D * D') * Q' + (R + R' * D) + k * (D * D') := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_composition_l2069_206966


namespace NUMINAMATH_CALUDE_coefficient_x_cube_equals_neg_84_equal_coefficients_implies_n_7_l2069_206968

-- Part 1
def binomial_coefficient (n k : ℕ) : ℤ := sorry

def coefficient_x_cube (x : ℝ) : ℤ := 
  binomial_coefficient 9 3 * (-1)^3

theorem coefficient_x_cube_equals_neg_84 : 
  coefficient_x_cube = λ _ ↦ -84 := by sorry

-- Part 2
def nth_term_coefficient (n r : ℕ) : ℤ := 
  binomial_coefficient n r

theorem equal_coefficients_implies_n_7 (n : ℕ) : 
  nth_term_coefficient n 2 = nth_term_coefficient n 5 → n = 7 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_cube_equals_neg_84_equal_coefficients_implies_n_7_l2069_206968


namespace NUMINAMATH_CALUDE_equation_one_solutions_l2069_206983

theorem equation_one_solutions (x : ℝ) : x^2 - 9 = 0 ↔ x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l2069_206983


namespace NUMINAMATH_CALUDE_white_marbles_count_l2069_206964

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 50 →
  blue = 5 →
  red = 9 →
  prob_red_or_white = 9/10 →
  (total - blue - red : ℚ) / total = prob_red_or_white - (red : ℚ) / total →
  total - blue - red = 36 :=
by
  sorry

#check white_marbles_count

end NUMINAMATH_CALUDE_white_marbles_count_l2069_206964


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2069_206928

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2069_206928


namespace NUMINAMATH_CALUDE_minimum_sum_geometric_sequence_l2069_206965

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n

theorem minimum_sum_geometric_sequence (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a)
    (h_positive : ∀ n, a n > 0)
    (h_product : a 3 * a 5 = 64) :
    ∃ (m : ℝ), m = 16 ∧ ∀ x y, x > 0 → y > 0 → x * y = 64 → x + y ≥ m :=
  sorry

end NUMINAMATH_CALUDE_minimum_sum_geometric_sequence_l2069_206965


namespace NUMINAMATH_CALUDE_no_valid_seating_l2069_206992

/-- A seating arrangement of deputies around a circular table. -/
structure Seating :=
  (deputies : Fin 47 → Fin 12)

/-- The property that any 15 consecutive deputies include all 12 regions. -/
def hasAllRegionsIn15 (s : Seating) : Prop :=
  ∀ start : Fin 47, ∃ (f : Fin 12 → Fin 15), ∀ r : Fin 12,
    ∃ i : Fin 15, s.deputies ((start + i) % 47) = r

/-- Theorem stating that no valid seating arrangement exists. -/
theorem no_valid_seating : ¬ ∃ s : Seating, hasAllRegionsIn15 s := by
  sorry

end NUMINAMATH_CALUDE_no_valid_seating_l2069_206992


namespace NUMINAMATH_CALUDE_last_integer_before_100_l2069_206914

def sequence_term (n : ℕ) : ℕ := (16777216 : ℕ) / 2^n

theorem last_integer_before_100 :
  ∃ n : ℕ, sequence_term n = 64 ∧ sequence_term (n + 1) < 100 :=
sorry

end NUMINAMATH_CALUDE_last_integer_before_100_l2069_206914


namespace NUMINAMATH_CALUDE_matrix_property_l2069_206930

/-- A 4x4 complex matrix with the given structure -/
def M (a b c d : ℂ) : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]]

theorem matrix_property (a b c d : ℂ) :
  M a b c d ^ 2 = 1 → a * b * c * d = 1 → a^4 + b^4 + c^4 + d^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_property_l2069_206930


namespace NUMINAMATH_CALUDE_set_inclusion_range_l2069_206905

theorem set_inclusion_range (a : ℝ) : 
  let P : Set ℝ := {x | |x - 1| > 2}
  let S : Set ℝ := {x | x^2 - (a + 1)*x + a > 0}
  (P ⊆ S) → ((-1 ≤ a ∧ a < 1) ∨ (1 < a ∧ a ≤ 3)) := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_range_l2069_206905


namespace NUMINAMATH_CALUDE_digit_puzzle_solution_l2069_206952

theorem digit_puzzle_solution (c o u n t s : ℕ) 
  (h1 : c + o = u)
  (h2 : u + n = t + 1)
  (h3 : t + c = s)
  (h4 : o + n + s = 15)
  (h5 : c ≠ 0 ∧ o ≠ 0 ∧ u ≠ 0 ∧ n ≠ 0 ∧ t ≠ 0 ∧ s ≠ 0)
  (h6 : c < 10 ∧ o < 10 ∧ u < 10 ∧ n < 10 ∧ t < 10 ∧ s < 10) :
  t = 7 := by sorry

end NUMINAMATH_CALUDE_digit_puzzle_solution_l2069_206952


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2069_206921

theorem rationalize_denominator :
  (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2069_206921


namespace NUMINAMATH_CALUDE_concurrent_or_parallel_iff_concyclic_l2069_206959

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the Euclidean plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Definition of a circumcenter -/
def circumcenter (t : Triangle) : Point :=
  sorry

/-- Definition of concurrency for three lines -/
def are_concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Definition of parallel lines -/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Definition of pairwise parallel lines -/
def are_pairwise_parallel (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Definition of concyclic points -/
def are_concyclic (A B C D : Point) : Prop :=
  sorry

/-- The main theorem -/
theorem concurrent_or_parallel_iff_concyclic 
  (A B C D E F : Point) 
  (G : Point := circumcenter ⟨B, C, E⟩) 
  (H : Point := circumcenter ⟨A, D, F⟩) 
  (AB CD GH : Line) :
  (are_concurrent AB CD GH ∨ are_pairwise_parallel AB CD GH) ↔ 
  are_concyclic A B E F :=
sorry

end NUMINAMATH_CALUDE_concurrent_or_parallel_iff_concyclic_l2069_206959


namespace NUMINAMATH_CALUDE_heart_equation_solution_l2069_206904

/-- The heart operation defined on two real numbers -/
def heart (A B : ℝ) : ℝ := 4*A + A*B + 3*B + 6

/-- Theorem stating that 60/7 is the unique solution to A ♥ 3 = 75 -/
theorem heart_equation_solution :
  ∃! A : ℝ, heart A 3 = 75 ∧ A = 60/7 := by sorry

end NUMINAMATH_CALUDE_heart_equation_solution_l2069_206904


namespace NUMINAMATH_CALUDE_farmer_water_capacity_l2069_206954

/-- Calculates the total water capacity of a single truck -/
def truckCapacity (tankCapacities : List ℕ) : ℕ :=
  tankCapacities.sum

/-- Calculates the amount of water in a truck given its capacity and fill percentage -/
def waterInTruck (capacity : ℕ) (fillPercentage : ℕ) : ℕ :=
  capacity * fillPercentage / 100

/-- Represents the problem of calculating total water capacity across multiple trucks -/
def waterCapacityProblem (tankCapacities : List ℕ) (fillPercentages : List ℕ) : Prop :=
  let capacity := truckCapacity tankCapacities
  let waterAmounts := fillPercentages.map (waterInTruck capacity)
  waterAmounts.sum = 2750

/-- The main theorem stating the solution to the water capacity problem -/
theorem farmer_water_capacity :
  waterCapacityProblem [200, 250, 300, 350] [100, 75, 50, 25, 0] := by
  sorry

#check farmer_water_capacity

end NUMINAMATH_CALUDE_farmer_water_capacity_l2069_206954


namespace NUMINAMATH_CALUDE_f_value_at_2_l2069_206969

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : f a b (-2) = 0 → f a b 2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l2069_206969


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2069_206976

def A : Set ℤ := {-1, 1}
def B : Set ℤ := {-1, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2069_206976


namespace NUMINAMATH_CALUDE_autograph_value_change_l2069_206920

theorem autograph_value_change (initial_value : ℝ) : 
  initial_value = 100 → 
  (initial_value * (1 - 0.3) * (1 + 0.4)) = 98 := by
  sorry

end NUMINAMATH_CALUDE_autograph_value_change_l2069_206920


namespace NUMINAMATH_CALUDE_right_triangle_perpendicular_bisector_l2069_206912

theorem right_triangle_perpendicular_bisector 
  (A B C D : ℝ × ℝ) 
  (right_angle_at_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 75)
  (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 100)
  (D_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2))
  (AD_perp_BC : (D.1 - A.1) * (C.1 - B.1) + (D.2 - A.2) * (C.2 - B.2) = 0) :
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 45 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perpendicular_bisector_l2069_206912


namespace NUMINAMATH_CALUDE_abs_equation_solution_l2069_206948

theorem abs_equation_solution :
  ∃! x : ℝ, |x - 3| = 5 - x :=
by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l2069_206948


namespace NUMINAMATH_CALUDE_toy_sale_proof_l2069_206903

/-- Calculates the total selling price of toys given the number of toys sold,
    the cost price per toy, and the gain in terms of number of toys. -/
def totalSellingPrice (numToysSold : ℕ) (costPrice : ℕ) (gainInToys : ℕ) : ℕ :=
  (numToysSold + gainInToys) * costPrice

/-- Proves that the total selling price for 18 toys with a cost price of 1200
    and a gain equal to the cost of 3 toys is 25200. -/
theorem toy_sale_proof :
  totalSellingPrice 18 1200 3 = 25200 := by
  sorry

end NUMINAMATH_CALUDE_toy_sale_proof_l2069_206903


namespace NUMINAMATH_CALUDE_total_flowers_l2069_206979

theorem total_flowers (roses tulips lilies : ℕ) : 
  roses = 58 ∧ 
  tulips = roses - 15 ∧ 
  lilies = roses + 25 → 
  roses + tulips + lilies = 184 := by
sorry

end NUMINAMATH_CALUDE_total_flowers_l2069_206979


namespace NUMINAMATH_CALUDE_monochromatic_right_triangle_exists_l2069_206961

/-- A point on the contour of a square -/
structure ContourPoint where
  x : ℝ
  y : ℝ

/-- Color of a point -/
inductive Color
  | Blue
  | Red

/-- A coloring of the contour of a square -/
def Coloring := ContourPoint → Color

/-- Predicate to check if three points form a right triangle -/
def is_right_triangle (p1 p2 p3 : ContourPoint) : Prop :=
  sorry

/-- Theorem: For any coloring of the contour of a square, there exists a right triangle
    with vertices of the same color -/
theorem monochromatic_right_triangle_exists (coloring : Coloring) :
  ∃ (p1 p2 p3 : ContourPoint),
    is_right_triangle p1 p2 p3 ∧
    coloring p1 = coloring p2 ∧
    coloring p2 = coloring p3 :=
  sorry

end NUMINAMATH_CALUDE_monochromatic_right_triangle_exists_l2069_206961


namespace NUMINAMATH_CALUDE_f_sum_two_three_l2069_206958

/-- An odd function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f satisfies the symmetry condition -/
axiom f_sym (x : ℝ) : f (3/2 + x) = -f (3/2 - x)

/-- f(1) = 2 -/
axiom f_one : f 1 = 2

/-- Theorem: f(2) + f(3) = -2 -/
theorem f_sum_two_three : f 2 + f 3 = -2 := by sorry

end NUMINAMATH_CALUDE_f_sum_two_three_l2069_206958


namespace NUMINAMATH_CALUDE_inscribed_rectangle_perimeter_l2069_206900

theorem inscribed_rectangle_perimeter (circle_area : ℝ) (rect_area : ℝ) :
  circle_area = 32 * Real.pi →
  rect_area = 34 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    a * b = rect_area ∧
    a^2 + b^2 = 2 * circle_area / Real.pi ∧
    2 * (a + b) = 28 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_perimeter_l2069_206900


namespace NUMINAMATH_CALUDE_least_number_remainder_l2069_206977

theorem least_number_remainder (n : ℕ) (h1 : n % 20 = 14) (h2 : n % 2535 = 1929) (h3 : n = 1394) : n % 40 = 34 := by
  sorry

end NUMINAMATH_CALUDE_least_number_remainder_l2069_206977


namespace NUMINAMATH_CALUDE_rabbit_log_cutting_l2069_206997

theorem rabbit_log_cutting (cuts pieces : ℕ) (h1 : cuts = 10) (h2 : pieces = 16) :
  ∃ logs : ℕ, logs + cuts = pieces ∧ logs = 6 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_log_cutting_l2069_206997


namespace NUMINAMATH_CALUDE_election_winner_percentage_l2069_206967

theorem election_winner_percentage 
  (total_votes : ℕ) 
  (winning_margin : ℕ) 
  (h1 : total_votes = 6900)
  (h2 : winning_margin = 1380) :
  (winning_margin : ℚ) / total_votes + 1/2 = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l2069_206967


namespace NUMINAMATH_CALUDE_dish_price_proof_l2069_206996

/-- The original price of a dish satisfying the given conditions -/
def original_price : ℝ := 34

/-- The discount rate applied to the original price -/
def discount_rate : ℝ := 0.1

/-- The tip rate applied to either the original or discounted price -/
def tip_rate : ℝ := 0.15

/-- The difference in total payments between the two people -/
def payment_difference : ℝ := 0.51

theorem dish_price_proof :
  let discounted_price := original_price * (1 - discount_rate)
  let payment1 := discounted_price + original_price * tip_rate
  let payment2 := discounted_price + discounted_price * tip_rate
  payment1 - payment2 = payment_difference := by sorry

end NUMINAMATH_CALUDE_dish_price_proof_l2069_206996


namespace NUMINAMATH_CALUDE_complex_modulus_seven_l2069_206909

theorem complex_modulus_seven (x : ℝ) : 
  x > 0 → (Complex.abs (3 + x * Complex.I) = 7 ↔ x = 2 * Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_seven_l2069_206909


namespace NUMINAMATH_CALUDE_three_number_problem_l2069_206949

theorem three_number_problem (a b c : ℚ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 6 * (b + c))
  (second_eq : b = 9 * c) :
  a - c = 177 / 7 := by
sorry

end NUMINAMATH_CALUDE_three_number_problem_l2069_206949


namespace NUMINAMATH_CALUDE_last_three_average_l2069_206938

theorem last_three_average (numbers : List ℝ) : 
  numbers.length = 5 →
  numbers.sum / numbers.length = 54 →
  (numbers.take 2).sum / 2 = 48 →
  (numbers.drop 2).sum / 3 = 58 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l2069_206938


namespace NUMINAMATH_CALUDE_chessboard_coloring_theorem_l2069_206919

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents a coloring of the chessboard -/
def Coloring := Square → Bool

/-- Checks if three squares form a trimino -/
def isTrimino (s1 s2 s3 : Square) : Prop := sorry

/-- Counts the number of red squares in a coloring -/
def countRedSquares (c : Coloring) : Nat := sorry

/-- Checks if a coloring has no red trimino -/
def hasNoRedTrimino (c : Coloring) : Prop := sorry

/-- Checks if every trimino in a coloring has at least one red square -/
def everyTriminoHasRed (c : Coloring) : Prop := sorry

theorem chessboard_coloring_theorem :
  (∃ c : Coloring, hasNoRedTrimino c ∧ countRedSquares c = 32) ∧
  (∀ c : Coloring, hasNoRedTrimino c → countRedSquares c ≤ 32) ∧
  (∃ c : Coloring, everyTriminoHasRed c ∧ countRedSquares c = 32) ∧
  (∀ c : Coloring, everyTriminoHasRed c → countRedSquares c ≥ 32) := by
  sorry

end NUMINAMATH_CALUDE_chessboard_coloring_theorem_l2069_206919


namespace NUMINAMATH_CALUDE_negation_equivalence_l2069_206994

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2069_206994


namespace NUMINAMATH_CALUDE_drone_production_equations_correct_l2069_206944

/-- Represents the number of drones of type A and B produced by a company -/
structure DroneProduction where
  x : ℝ  -- number of type A drones
  y : ℝ  -- number of type B drones

/-- The system of equations representing the drone production conditions -/
def satisfiesConditions (p : DroneProduction) : Prop :=
  p.x = (1/2) * (p.x + p.y) + 11 ∧ p.y = (1/3) * (p.x + p.y) - 2

/-- Theorem stating that the system of equations correctly represents the given conditions -/
theorem drone_production_equations_correct (p : DroneProduction) :
  satisfiesConditions p ↔
    (p.x = (1/2) * (p.x + p.y) + 11 ∧   -- Type A drones condition
     p.y = (1/3) * (p.x + p.y) - 2) :=  -- Type B drones condition
by sorry

end NUMINAMATH_CALUDE_drone_production_equations_correct_l2069_206944


namespace NUMINAMATH_CALUDE_smallest_cube_for_pyramid_l2069_206946

/-- Represents a pyramid with a square base -/
structure Pyramid where
  height : ℝ
  baseLength : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.sideLength ^ 3

/-- Determines if a cube can contain a pyramid standing upright -/
def canContainPyramid (c : Cube) (p : Pyramid) : Prop :=
  c.sideLength ≥ p.height ∧ c.sideLength ≥ p.baseLength

theorem smallest_cube_for_pyramid (p : Pyramid) (h1 : p.height = 12) (h2 : p.baseLength = 10) :
  ∃ (c : Cube), canContainPyramid c p ∧
    cubeVolume c = 1728 ∧
    ∀ (c' : Cube), canContainPyramid c' p → cubeVolume c' ≥ cubeVolume c :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_for_pyramid_l2069_206946


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l2069_206956

theorem base_10_to_base_7 :
  ∃ (a b c d : ℕ),
    804 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
    a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
    a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 6 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l2069_206956


namespace NUMINAMATH_CALUDE_point_not_on_line_l2069_206917

/-- Given real numbers m and b where mb < 0, the point (1, 2001) does not lie on the line y = m(x^2) + b -/
theorem point_not_on_line (m b : ℝ) (h : m * b < 0) : 
  ¬(2001 = m * (1 : ℝ)^2 + b) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_line_l2069_206917


namespace NUMINAMATH_CALUDE_oliver_unwashed_shirts_l2069_206932

theorem oliver_unwashed_shirts 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 39)
  (h2 : long_sleeve = 47)
  (h3 : washed = 20) :
  short_sleeve + long_sleeve - washed = 66 := by
sorry

end NUMINAMATH_CALUDE_oliver_unwashed_shirts_l2069_206932


namespace NUMINAMATH_CALUDE_complex_magnitude_of_i_times_one_minus_i_l2069_206988

theorem complex_magnitude_of_i_times_one_minus_i : 
  let z : ℂ := Complex.I * (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_of_i_times_one_minus_i_l2069_206988


namespace NUMINAMATH_CALUDE_two_ab_value_l2069_206984

theorem two_ab_value (a b : ℝ) 
  (h1 : a^4 + a^2*b^2 + b^4 = 900) 
  (h2 : a^2 + a*b + b^2 = 45) : 
  2*a*b = 25 := by
sorry

end NUMINAMATH_CALUDE_two_ab_value_l2069_206984


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l2069_206974

/-- Given the cost of pencils and pens, calculate the cost of a different combination -/
theorem pencil_pen_cost (p q : ℝ) 
  (h1 : 3 * p + 2 * q = 4.30)
  (h2 : 2 * p + 3 * q = 4.05) :
  4 * p + 3 * q = 5.97 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l2069_206974


namespace NUMINAMATH_CALUDE_gummy_bear_cost_l2069_206902

theorem gummy_bear_cost
  (total_cost : ℝ)
  (chocolate_chip_cost : ℝ)
  (chocolate_bar_cost : ℝ)
  (num_chocolate_bars : ℕ)
  (num_gummy_bears : ℕ)
  (num_chocolate_chips : ℕ)
  (h1 : total_cost = 150)
  (h2 : chocolate_chip_cost = 5)
  (h3 : chocolate_bar_cost = 3)
  (h4 : num_chocolate_bars = 10)
  (h5 : num_gummy_bears = 10)
  (h6 : num_chocolate_chips = 20)
  : (total_cost - num_chocolate_bars * chocolate_bar_cost - num_chocolate_chips * chocolate_chip_cost) / num_gummy_bears = 2 := by
  sorry

end NUMINAMATH_CALUDE_gummy_bear_cost_l2069_206902


namespace NUMINAMATH_CALUDE_matches_arrangement_count_l2069_206918

/-- The number of ways to arrange matches for n players with some interchangeable players -/
def arrangeMatches (n : ℕ) (interchangeablePairs : ℕ) : ℕ :=
  Nat.factorial n * (2 ^ interchangeablePairs)

/-- Theorem: For 7 players with 3 pairs of interchangeable players, there are 40320 ways to arrange matches -/
theorem matches_arrangement_count :
  arrangeMatches 7 3 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_matches_arrangement_count_l2069_206918


namespace NUMINAMATH_CALUDE_no_valid_coloring_200_points_l2069_206941

/-- Represents a coloring of points and segments -/
structure Coloring (n : ℕ) (k : ℕ) where
  pointColor : Fin n → Fin k
  segmentColor : Fin n → Fin n → Fin k

/-- Predicate for a valid coloring -/
def isValidColoring (n : ℕ) (k : ℕ) (c : Coloring n k) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    c.pointColor i ≠ c.pointColor j ∧
    c.pointColor i ≠ c.segmentColor i j ∧
    c.pointColor j ≠ c.segmentColor i j

/-- Theorem stating the impossibility of valid coloring for 200 points with 7 or 10 colors -/
theorem no_valid_coloring_200_points :
  ¬ (∃ c : Coloring 200 7, isValidColoring 200 7 c) ∧
  ¬ (∃ c : Coloring 200 10, isValidColoring 200 10 c) := by
  sorry


end NUMINAMATH_CALUDE_no_valid_coloring_200_points_l2069_206941


namespace NUMINAMATH_CALUDE_fundraiser_item_price_l2069_206915

theorem fundraiser_item_price 
  (num_brownie_students : ℕ) 
  (num_cookie_students : ℕ) 
  (num_donut_students : ℕ) 
  (brownies_per_student : ℕ) 
  (cookies_per_student : ℕ) 
  (donuts_per_student : ℕ) 
  (total_amount_raised : ℚ) : 
  num_brownie_students = 30 →
  num_cookie_students = 20 →
  num_donut_students = 15 →
  brownies_per_student = 12 →
  cookies_per_student = 24 →
  donuts_per_student = 12 →
  total_amount_raised = 2040 →
  (total_amount_raised / (num_brownie_students * brownies_per_student + 
                          num_cookie_students * cookies_per_student + 
                          num_donut_students * donuts_per_student) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_item_price_l2069_206915


namespace NUMINAMATH_CALUDE_parabola_max_area_l2069_206962

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := y = x + 1

-- Define the condition for p
def p_condition (p : ℝ) : Prop := p > 0

-- Define the points A and B on the parabola
def point_on_parabola (p x y : ℝ) : Prop := parabola p x y

-- Define the condition for x₁ and x₂
def x_condition (x₁ x₂ : ℝ) : Prop := x₁ ≠ x₂ ∧ x₁ + x₂ = 4

-- Define the theorem
theorem parabola_max_area (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  p_condition p →
  parabola p x₁ y₁ →
  parabola p x₂ y₂ →
  x_condition x₁ x₂ →
  (∃ (x y : ℝ), parabola p x y ∧ tangent_line x y) →
  (∃ (area : ℝ), area ≤ 8 ∧ 
    (∀ (other_area : ℝ), other_area ≤ area)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_area_l2069_206962


namespace NUMINAMATH_CALUDE_system_solution_l2069_206963

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x - a * y + a^2 * z = a^3)
  (eq2 : x - b * y + b^2 * z = b^3)
  (eq3 : x - c * y + c^2 * z = c^3)
  (hx : x = a * b * c)
  (hy : y = a * b + a * c + b * c)
  (hz : z = a + b + c)
  (ha : a ≠ b)
  (hb : a ≠ c)
  (hc : b ≠ c) :
  x - a * y + a^2 * z = a^3 ∧
  x - b * y + b^2 * z = b^3 ∧
  x - c * y + c^2 * z = c^3 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2069_206963


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2069_206913

/-- Proves that (8-15i)/(3+4i) = -36/25 - 77/25*i -/
theorem complex_fraction_simplification :
  (8 - 15 * Complex.I) / (3 + 4 * Complex.I) = -36/25 - 77/25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2069_206913


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2069_206929

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 - 2 * a^2 + 5 * a + 15 = 0) →
  (3 * b^3 - 2 * b^2 + 5 * b + 15 = 0) →
  (3 * c^3 - 2 * c^2 + 5 * c + 15 = 0) →
  a^2 + b^2 + c^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2069_206929


namespace NUMINAMATH_CALUDE_fred_marble_count_l2069_206993

/-- The number of blue marbles Tim has -/
def tim_marbles : ℕ := 5

/-- The factor by which Fred has more marbles than Tim -/
def fred_factor : ℕ := 22

/-- The number of blue marbles Fred has -/
def fred_marbles : ℕ := tim_marbles * fred_factor

theorem fred_marble_count : fred_marbles = 110 := by
  sorry

end NUMINAMATH_CALUDE_fred_marble_count_l2069_206993


namespace NUMINAMATH_CALUDE_tom_books_theorem_l2069_206955

def books_problem (initial_books sold_books bought_books : ℕ) : Prop :=
  let remaining_books := initial_books - sold_books
  let final_books := remaining_books + bought_books
  final_books = 39

theorem tom_books_theorem :
  books_problem 5 4 38 := by sorry

end NUMINAMATH_CALUDE_tom_books_theorem_l2069_206955


namespace NUMINAMATH_CALUDE_faster_train_speed_l2069_206945

/-- Proves that given two trains of specified lengths running in opposite directions,
    with a given crossing time and speed of the slower train, the speed of the faster train
    is as calculated. -/
theorem faster_train_speed
  (length_train1 : ℝ)
  (length_train2 : ℝ)
  (crossing_time : ℝ)
  (slower_train_speed : ℝ)
  (h1 : length_train1 = 180)
  (h2 : length_train2 = 360)
  (h3 : crossing_time = 21.598272138228943)
  (h4 : slower_train_speed = 30) :
  ∃ (faster_train_speed : ℝ),
    faster_train_speed = 60 ∧
    (length_train1 + length_train2) / crossing_time * 3.6 = slower_train_speed + faster_train_speed :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l2069_206945


namespace NUMINAMATH_CALUDE_no_parallel_m_l2069_206923

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  (2 : ℝ) / (m + 1) = m / 3

/-- There is no real number m that makes the lines parallel -/
theorem no_parallel_m : ¬ ∃ m : ℝ, parallel_lines m := by
  sorry

end NUMINAMATH_CALUDE_no_parallel_m_l2069_206923


namespace NUMINAMATH_CALUDE_downstream_distance_l2069_206953

/-- The distance swum downstream by a woman given certain conditions -/
theorem downstream_distance (t : ℝ) (d_up : ℝ) (v_still : ℝ) : 
  t > 0 ∧ d_up > 0 ∧ v_still > 0 →
  t = 6 ∧ d_up = 6 ∧ v_still = 5 →
  ∃ d_down : ℝ, d_down = 54 ∧ 
    d_down / (v_still + (d_up / t - v_still)) = t ∧
    d_up / (v_still - (d_up / t - v_still)) = t :=
by sorry


end NUMINAMATH_CALUDE_downstream_distance_l2069_206953


namespace NUMINAMATH_CALUDE_duck_cow_leg_count_l2069_206951

theorem duck_cow_leg_count :
  ∀ (num_ducks : ℕ),
  let num_cows : ℕ := 12
  let total_heads : ℕ := num_ducks + num_cows
  let total_legs : ℕ := 2 * num_ducks + 4 * num_cows
  total_legs - 2 * total_heads = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_cow_leg_count_l2069_206951


namespace NUMINAMATH_CALUDE_five_power_sum_of_squares_l2069_206970

theorem five_power_sum_of_squares (n : ℕ) : ∃ (a b : ℕ), 5^n = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_five_power_sum_of_squares_l2069_206970


namespace NUMINAMATH_CALUDE_infinite_segment_sum_l2069_206987

/-- Given a triangle ABC with sides a, b, c where b > c, and an infinite sequence
    of line segments constructed as follows:
    - BB1 is antiparallel to BC, intersecting AC at B1
    - B1C1 is parallel to BC, intersecting AB at C1
    - This process continues infinitely
    Then the sum of the lengths of these segments (BC + BB1 + B1C1 + ...) is ab / (b - c) -/
theorem infinite_segment_sum (a b c : ℝ) (h : b > c) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (sequence : ℕ → ℝ),
    (sequence 0 = a) ∧
    (∀ n, sequence (n + 1) = sequence n * (c / b)) ∧
    (∑' n, sequence n) = a * b / (b - c) := by
  sorry

end NUMINAMATH_CALUDE_infinite_segment_sum_l2069_206987


namespace NUMINAMATH_CALUDE_cube_root_of_sum_l2069_206986

theorem cube_root_of_sum (a b : ℝ) : 
  (2*a + 1) + (2*a - 5) = 0 → 
  b^(1/3 : ℝ) = 2 → 
  (a + b)^(1/3 : ℝ) = 9^(1/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_sum_l2069_206986


namespace NUMINAMATH_CALUDE_divisibility_problem_l2069_206985

theorem divisibility_problem (a b : Nat) (n : Nat) : 
  a ≤ 9 → b ≤ 9 → a * b ≤ 15 → (110 * a + b) % n = 0 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2069_206985


namespace NUMINAMATH_CALUDE_parabola_equation_l2069_206908

/-- A parabola with vertex at the origin and focus on the positive x-axis -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  h_p_pos : 0 < p
  h_focus : focus = (p / 2, 0)

/-- Two points on the parabola -/
structure ParabolaPoints (C : Parabola) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_parabola : A.2^2 = 2 * C.p * A.1 ∧ B.2^2 = 2 * C.p * B.1
  h_line_through_focus : ∃ k : ℝ, A.2 = k * (A.1 - C.p / 2) ∧ B.2 = k * (B.1 - C.p / 2)

/-- The dot product condition -/
def dot_product_condition (C : Parabola) (P : ParabolaPoints C) : Prop :=
  P.A.1 * P.B.1 + P.A.2 * P.B.2 = -12

/-- The main theorem -/
theorem parabola_equation (C : Parabola) (P : ParabolaPoints C)
  (h_dot : dot_product_condition C P) :
  C.p = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2069_206908


namespace NUMINAMATH_CALUDE_foundation_digging_l2069_206991

/-- Represents the work rate for digging a foundation -/
def work_rate (men : ℕ) (days : ℝ) : ℝ := men * days

theorem foundation_digging 
  (men_first_half : ℕ) (days_first_half : ℝ) 
  (men_second_half : ℕ) :
  men_first_half = 10 →
  days_first_half = 6 →
  men_second_half = 20 →
  work_rate men_first_half days_first_half = work_rate men_second_half 3 :=
by sorry

end NUMINAMATH_CALUDE_foundation_digging_l2069_206991


namespace NUMINAMATH_CALUDE_natural_numbers_with_special_last_digit_l2069_206916

def last_digit (n : ℕ) : ℕ := n % 10

def satisfies_condition (n : ℕ) : Prop :=
  n ≠ 0 ∧ n = 2016 * (last_digit n)

theorem natural_numbers_with_special_last_digit :
  {n : ℕ | satisfies_condition n} = {4032, 8064, 12096, 16128} :=
by sorry

end NUMINAMATH_CALUDE_natural_numbers_with_special_last_digit_l2069_206916


namespace NUMINAMATH_CALUDE_p_on_x_axis_equal_distance_to_axes_l2069_206978

-- Define the point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (8 - 2*m, m - 1)

-- Part 1: P lies on the x-axis implies m = 1
theorem p_on_x_axis (m : ℝ) : (P m).2 = 0 → m = 1 := by sorry

-- Part 2: Equal distance to both axes implies P(2,2) or P(-6,6)
theorem equal_distance_to_axes (m : ℝ) : 
  |8 - 2*m| = |m - 1| → (P m = (2, 2) ∨ P m = (-6, 6)) := by sorry

end NUMINAMATH_CALUDE_p_on_x_axis_equal_distance_to_axes_l2069_206978


namespace NUMINAMATH_CALUDE_same_color_probability_l2069_206937

/-- The probability of drawing three marbles of the same color from a bag containing
    3 red marbles, 7 white marbles, and 5 blue marbles, without replacement. -/
theorem same_color_probability (red : ℕ) (white : ℕ) (blue : ℕ) 
    (h_red : red = 3) (h_white : white = 7) (h_blue : blue = 5) :
    let total := red + white + blue
    let p_all_red := (red / total) * ((red - 1) / (total - 1)) * ((red - 2) / (total - 2))
    let p_all_white := (white / total) * ((white - 1) / (total - 1)) * ((white - 2) / (total - 2))
    let p_all_blue := (blue / total) * ((blue - 1) / (total - 1)) * ((blue - 2) / (total - 2))
    p_all_red + p_all_white + p_all_blue = 23 / 455 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2069_206937


namespace NUMINAMATH_CALUDE_sixth_row_third_number_l2069_206926

/-- Represents the sequence of positive odd numbers -/
def oddSequence (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of elements in the nth row of the table -/
def rowSize (n : ℕ) : ℕ := 2^(n - 1)

/-- Represents the total number of elements up to and including the nth row -/
def totalElements (n : ℕ) : ℕ := 2^n - 1

theorem sixth_row_third_number : 
  let rowNumber := 6
  let positionInRow := 3
  oddSequence (totalElements (rowNumber - 1) + positionInRow) = 67 := by
  sorry

end NUMINAMATH_CALUDE_sixth_row_third_number_l2069_206926


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l2069_206935

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  A * B = !![23, -7; 24, -16] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l2069_206935


namespace NUMINAMATH_CALUDE_min_moves_for_n_triangles_l2069_206971

/-- Represents a robot on a vertex of a polygon -/
structure Robot where
  vertex : ℕ
  target : ℕ

/-- Represents the state of the polygon -/
structure PolygonState where
  n : ℕ
  robots : List Robot

/-- A move rotates a robot to point at a new target -/
def move (state : PolygonState) (robot_index : ℕ) : PolygonState :=
  sorry

/-- Checks if three robots form a triangle -/
def is_triangle (r1 r2 r3 : Robot) : Bool :=
  sorry

/-- Counts the number of triangles in the current state -/
def count_triangles (state : PolygonState) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of moves required -/
theorem min_moves_for_n_triangles (n : ℕ) :
  ∃ (initial_state : PolygonState),
    initial_state.n = n ∧
    initial_state.robots.length = 3 * n ∧
    ∀ (final_state : PolygonState),
      (count_triangles final_state = n) →
      (∃ (move_sequence : List ℕ),
        final_state = (move_sequence.foldl move initial_state) ∧
        move_sequence.length ≥ (9 * n^2 - 7 * n) / 2) :=
sorry

end NUMINAMATH_CALUDE_min_moves_for_n_triangles_l2069_206971


namespace NUMINAMATH_CALUDE_tetrahedron_inequality_l2069_206934

theorem tetrahedron_inequality 
  (h₁ h₂ h₃ h₄ x₁ x₂ x₃ x₄ : ℝ) 
  (h_nonneg : h₁ ≥ 0 ∧ h₂ ≥ 0 ∧ h₃ ≥ 0 ∧ h₄ ≥ 0)
  (x_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0)
  (h_tetrahedron : ∃ (S₁ S₂ S₃ S₄ : ℝ), 
    S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 ∧
    S₁ * h₁ = S₁ * x₁ ∧ 
    S₂ * h₂ = S₂ * x₂ ∧ 
    S₃ * h₃ = S₃ * x₃ ∧ 
    S₄ * h₄ = S₄ * x₄) :
  Real.sqrt (h₁ + h₂ + h₃ + h₄) ≥ Real.sqrt x₁ + Real.sqrt x₂ + Real.sqrt x₃ + Real.sqrt x₄ := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inequality_l2069_206934


namespace NUMINAMATH_CALUDE_expression_evaluation_l2069_206939

theorem expression_evaluation :
  let x : ℝ := Real.sin (30 * π / 180)
  (1 - 2 / (x - 1)) / ((x - 3) / (x^2 - 1)) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2069_206939


namespace NUMINAMATH_CALUDE_range_of_a_l2069_206972

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 ≤ 0 → x^2 + 5*x + 4 < 0) →
  a < 0 →
  -4/3 ≤ a ∧ a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2069_206972


namespace NUMINAMATH_CALUDE_eliminate_x_y_l2069_206975

-- Define the tangent and cotangent functions
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- State the theorem
theorem eliminate_x_y (x y a b c : ℝ) 
  (h1 : tg x + tg y = a)
  (h2 : ctg x + ctg y = b)
  (h3 : x + y = c) :
  ctg c = 1 / a - 1 / b :=
by sorry

end NUMINAMATH_CALUDE_eliminate_x_y_l2069_206975
