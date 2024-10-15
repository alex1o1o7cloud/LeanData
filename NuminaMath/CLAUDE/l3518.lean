import Mathlib

namespace NUMINAMATH_CALUDE_standard_deviation_from_variance_l3518_351849

theorem standard_deviation_from_variance (variance : ℝ) (std_dev : ℝ) :
  variance = 2 → std_dev = Real.sqrt variance → std_dev = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_from_variance_l3518_351849


namespace NUMINAMATH_CALUDE_sticker_distribution_count_l3518_351822

/-- The number of ways to partition n identical objects into k or fewer parts -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 9

/-- The number of sheets -/
def num_sheets : ℕ := 3

theorem sticker_distribution_count : 
  partition_count num_stickers num_sheets = 12 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_count_l3518_351822


namespace NUMINAMATH_CALUDE_midpoint_complex_numbers_l3518_351821

theorem midpoint_complex_numbers : 
  let A : ℂ := 1 / (1 + Complex.I)
  let B : ℂ := 1 / (1 - Complex.I)
  let C : ℂ := (A + B) / 2
  C = (1 : ℂ) / 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_complex_numbers_l3518_351821


namespace NUMINAMATH_CALUDE_concatenated_seven_digit_divisible_by_239_l3518_351841

/-- Represents a sequence of seven-digit numbers -/
def SevenDigitSequence := List Nat

/-- Concatenates a list of natural numbers -/
def concatenate (seq : SevenDigitSequence) : Nat :=
  seq.foldl (fun acc n => acc * 10000000 + n) 0

/-- The sequence of all seven-digit numbers -/
def allSevenDigitNumbers : SevenDigitSequence :=
  List.range 10000000

theorem concatenated_seven_digit_divisible_by_239 :
  ∃ k : ℕ, concatenate allSevenDigitNumbers = 239 * k :=
sorry

end NUMINAMATH_CALUDE_concatenated_seven_digit_divisible_by_239_l3518_351841


namespace NUMINAMATH_CALUDE_common_chord_equation_l3518_351861

/-- Given two circles with equations x^2 + y^2 - 4x = 0 and x^2 + y^2 - 4y = 0,
    the equation of the line where their common chord lies is x - y = 0. -/
theorem common_chord_equation (x y : ℝ) : 
  (x^2 + y^2 - 4*x = 0 ∧ x^2 + y^2 - 4*y = 0) → x - y = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l3518_351861


namespace NUMINAMATH_CALUDE_jackson_courtyard_tile_cost_l3518_351842

/-- Calculates the total cost of tiles for a courtyard -/
def total_tile_cost (length width : ℝ) (tiles_per_sqft : ℝ) (green_tile_percent : ℝ) (green_tile_cost red_tile_cost : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := green_tile_percent * total_tiles
  let red_tiles := total_tiles - green_tiles
  green_tiles * green_tile_cost + red_tiles * red_tile_cost

/-- Theorem stating the total cost of tiles for Jackson's courtyard -/
theorem jackson_courtyard_tile_cost :
  total_tile_cost 10 25 4 0.4 3 1.5 = 2100 :=
by sorry

end NUMINAMATH_CALUDE_jackson_courtyard_tile_cost_l3518_351842


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3518_351866

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 → a * b = 0)) ↔ (a * b ≠ 0 → a ≠ 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3518_351866


namespace NUMINAMATH_CALUDE_writing_speed_ratio_l3518_351869

/-- Jacob and Nathan's writing speeds -/
def writing_problem (jacob_speed nathan_speed : ℚ) : Prop :=
  nathan_speed = 25 ∧ 
  jacob_speed + nathan_speed = 75 ∧
  jacob_speed / nathan_speed = 2

theorem writing_speed_ratio : ∃ (jacob_speed nathan_speed : ℚ), 
  writing_problem jacob_speed nathan_speed :=
sorry

end NUMINAMATH_CALUDE_writing_speed_ratio_l3518_351869


namespace NUMINAMATH_CALUDE_city_population_ratio_l3518_351808

theorem city_population_ratio (pop_x pop_y pop_z : ℝ) (s : ℝ) 
  (h1 : pop_x = 6 * pop_y)
  (h2 : pop_y = s * pop_z)
  (h3 : pop_x / pop_z = 12)
  (h4 : pop_z > 0) : 
  pop_y / pop_z = 2 := by
sorry

end NUMINAMATH_CALUDE_city_population_ratio_l3518_351808


namespace NUMINAMATH_CALUDE_flower_arrangement_daisies_percentage_l3518_351876

theorem flower_arrangement_daisies_percentage
  (total_flowers : ℕ)
  (h1 : total_flowers > 0)
  (yellow_flowers : ℕ)
  (h2 : yellow_flowers = (7 * total_flowers) / 10)
  (white_flowers : ℕ)
  (h3 : white_flowers = total_flowers - yellow_flowers)
  (yellow_tulips : ℕ)
  (h4 : yellow_tulips = yellow_flowers / 2)
  (white_daisies : ℕ)
  (h5 : white_daisies = (2 * white_flowers) / 3)
  (yellow_daisies : ℕ)
  (h6 : yellow_daisies = yellow_flowers - yellow_tulips)
  (total_daisies : ℕ)
  (h7 : total_daisies = yellow_daisies + white_daisies) :
  (total_daisies : ℚ) / total_flowers = 11 / 20 :=
sorry

end NUMINAMATH_CALUDE_flower_arrangement_daisies_percentage_l3518_351876


namespace NUMINAMATH_CALUDE_inequality_empty_solution_set_l3518_351896

theorem inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) ↔ -2 ≤ a ∧ a < 6/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_empty_solution_set_l3518_351896


namespace NUMINAMATH_CALUDE_f_equals_2x_plus_7_l3518_351898

-- Define the functions g and f
def g (x : ℝ) : ℝ := 2 * x + 3
def f (x : ℝ) : ℝ := g (x + 2)

-- State the theorem
theorem f_equals_2x_plus_7 : ∀ x : ℝ, f x = 2 * x + 7 := by
  sorry

end NUMINAMATH_CALUDE_f_equals_2x_plus_7_l3518_351898


namespace NUMINAMATH_CALUDE_expression_equality_l3518_351878

theorem expression_equality : (2^1004 + 5^1005)^2 - (2^1004 - 5^1005)^2 = 20 * 10^1004 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3518_351878


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3518_351826

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 4 * (Real.tan α)^2 + Real.tan α - 3 = 0) : Real.sin α = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3518_351826


namespace NUMINAMATH_CALUDE_reciprocal_sum_property_l3518_351804

theorem reciprocal_sum_property (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : 1 / a + 1 / b + 1 / c = 1 / (a + b + c)) :
  ∀ n : ℤ, (1 / a^n + 1 / b^n + 1 / c^n = 1 / (a^n + b^n + c^n)) ↔ ∃ k : ℕ, n = 2 * k - 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_property_l3518_351804


namespace NUMINAMATH_CALUDE_percentage_not_covering_politics_l3518_351823

-- Define the percentages as real numbers
def country_x : ℝ := 15
def country_y : ℝ := 10
def country_z : ℝ := 8
def x_elections : ℝ := 6
def y_foreign : ℝ := 5
def z_social : ℝ := 3
def not_local : ℝ := 50
def international : ℝ := 5
def economics : ℝ := 2

-- Theorem statement
theorem percentage_not_covering_politics :
  100 - (country_x + country_y + country_z + international + economics + not_local) = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_covering_politics_l3518_351823


namespace NUMINAMATH_CALUDE_binomial_square_condition_l3518_351827

/-- If 9x^2 - 24x + a is the square of a binomial, then a = 16 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ p q : ℝ, ∀ x, 9*x^2 - 24*x + a = (p*x + q)^2) → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l3518_351827


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3518_351813

theorem trigonometric_simplification (α : ℝ) :
  (Real.sin (6 * α) / Real.sin (2 * α)) + (Real.cos (6 * α - π) / Real.cos (2 * α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3518_351813


namespace NUMINAMATH_CALUDE_modular_equation_solution_l3518_351895

theorem modular_equation_solution (x : ℤ) : 
  (10 * x + 3 ≡ 7 [ZMOD 15]) → 
  (∃ (a m : ℕ), m ≥ 2 ∧ a < m ∧ x ≡ a [ZMOD m] ∧ a + m = 27) :=
by sorry

end NUMINAMATH_CALUDE_modular_equation_solution_l3518_351895


namespace NUMINAMATH_CALUDE_worker_pay_calculation_l3518_351884

/-- Calculates the total pay for a worker given their regular pay rate, 
    regular hours, overtime hours, and overtime pay rate multiplier. -/
def totalPay (regularRate : ℝ) (regularHours : ℝ) (overtimeHours : ℝ) (overtimeMultiplier : ℝ) : ℝ :=
  regularRate * regularHours + regularRate * overtimeMultiplier * overtimeHours

theorem worker_pay_calculation :
  let regularRate : ℝ := 3
  let regularHours : ℝ := 40
  let overtimeHours : ℝ := 8
  let overtimeMultiplier : ℝ := 2
  totalPay regularRate regularHours overtimeHours overtimeMultiplier = 168 := by
sorry

end NUMINAMATH_CALUDE_worker_pay_calculation_l3518_351884


namespace NUMINAMATH_CALUDE_max_correct_answers_l3518_351867

/-- Represents a test score --/
structure TestScore where
  total_questions : ℕ
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  score : ℤ

/-- Checks if a TestScore is valid according to the given conditions --/
def is_valid_score (ts : TestScore) : Prop :=
  ts.total_questions = 30 ∧
  ts.correct + ts.incorrect + ts.unanswered = ts.total_questions ∧
  ts.score = 4 * ts.correct - ts.incorrect

/-- Theorem stating the maximum number of correct answers --/
theorem max_correct_answers (ts : TestScore) (h : is_valid_score ts) (score_70 : ts.score = 70) :
  ts.correct ≤ 20 ∧ ∃ (ts' : TestScore), is_valid_score ts' ∧ ts'.score = 70 ∧ ts'.correct = 20 :=
sorry

end NUMINAMATH_CALUDE_max_correct_answers_l3518_351867


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_l3518_351897

theorem simultaneous_inequalities (x : ℝ) :
  x^2 - 12*x + 32 > 0 ∧ x^2 - 13*x + 22 < 0 → 2 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_l3518_351897


namespace NUMINAMATH_CALUDE_thousandth_digit_is_one_l3518_351879

/-- The number of digits in n -/
def num_digits : ℕ := 1998

/-- The number n as a natural number -/
def n : ℕ := (10^num_digits - 1) / 9

/-- The 1000th digit after the decimal point of √n -/
def thousandth_digit_after_decimal (n : ℕ) : ℕ :=
  -- Definition placeholder, actual implementation would be complex
  sorry

/-- Theorem stating that the 1000th digit after the decimal point of √n is 1 -/
theorem thousandth_digit_is_one :
  thousandth_digit_after_decimal n = 1 := by sorry

end NUMINAMATH_CALUDE_thousandth_digit_is_one_l3518_351879


namespace NUMINAMATH_CALUDE_power_two_plus_two_gt_square_l3518_351801

theorem power_two_plus_two_gt_square (n : ℕ) (hn : n > 0) : 2^n + 2 > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_plus_two_gt_square_l3518_351801


namespace NUMINAMATH_CALUDE_medium_box_tape_proof_l3518_351837

/-- The amount of tape (in feet) needed to seal a large box -/
def large_box_tape : ℝ := 4

/-- The amount of tape (in feet) needed to seal a small box -/
def small_box_tape : ℝ := 1

/-- The amount of tape (in feet) needed for the address label on any box -/
def label_tape : ℝ := 1

/-- The number of large boxes packed -/
def num_large_boxes : ℕ := 2

/-- The number of medium boxes packed -/
def num_medium_boxes : ℕ := 8

/-- The number of small boxes packed -/
def num_small_boxes : ℕ := 5

/-- The total amount of tape (in feet) used -/
def total_tape : ℝ := 44

/-- The amount of tape (in feet) needed to seal a medium box -/
def medium_box_tape : ℝ := 2

theorem medium_box_tape_proof :
  medium_box_tape * num_medium_boxes + 
  large_box_tape * num_large_boxes + 
  small_box_tape * num_small_boxes + 
  label_tape * (num_large_boxes + num_medium_boxes + num_small_boxes) = 
  total_tape := by sorry

end NUMINAMATH_CALUDE_medium_box_tape_proof_l3518_351837


namespace NUMINAMATH_CALUDE_amanda_keeps_33_candy_bars_l3518_351814

/-- Calculates the number of candy bars Amanda keeps for herself after a series of events --/
def amanda_candy_bars : ℕ :=
  let initial := 7
  let after_first_give := initial - (initial / 3)
  let after_buying := after_first_give + 30
  let after_second_give := after_buying - (after_buying / 4)
  let after_gift := after_second_give + 15
  let final := after_gift - ((15 * 3) / 5)
  final

/-- Theorem stating that Amanda keeps 33 candy bars for herself --/
theorem amanda_keeps_33_candy_bars : amanda_candy_bars = 33 := by
  sorry

end NUMINAMATH_CALUDE_amanda_keeps_33_candy_bars_l3518_351814


namespace NUMINAMATH_CALUDE_bob_first_six_probability_l3518_351802

/-- The probability of tossing a six on a fair die -/
def probSix : ℚ := 1 / 6

/-- The probability of not tossing a six on a fair die -/
def probNotSix : ℚ := 1 - probSix

/-- The order of players: Alice, Charlie, Bob -/
inductive Player : Type
| Alice : Player
| Charlie : Player
| Bob : Player

/-- The probability that Bob is the first to toss a six in the die-tossing game -/
def probBobFirstSix : ℚ := 25 / 91

theorem bob_first_six_probability :
  probBobFirstSix = (probNotSix * probNotSix * probSix) / (1 - probNotSix * probNotSix * probNotSix) :=
by sorry

end NUMINAMATH_CALUDE_bob_first_six_probability_l3518_351802


namespace NUMINAMATH_CALUDE_shaded_area_semicircle_pattern_shaded_area_semicircle_pattern_correct_l3518_351881

/-- The area of the shaded region in a pattern of semicircles -/
theorem shaded_area_semicircle_pattern (pattern_length : Real) (semicircle_diameter : Real) 
  (h1 : pattern_length = 2 * 12) -- 2 feet converted to inches
  (h2 : semicircle_diameter = 3) -- diameter in inches
  : Real :=
  18 * Real.pi

theorem shaded_area_semicircle_pattern_correct 
  (pattern_length : Real) (semicircle_diameter : Real) 
  (h1 : pattern_length = 2 * 12) -- 2 feet converted to inches
  (h2 : semicircle_diameter = 3) -- diameter in inches
  : shaded_area_semicircle_pattern pattern_length semicircle_diameter h1 h2 = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircle_pattern_shaded_area_semicircle_pattern_correct_l3518_351881


namespace NUMINAMATH_CALUDE_sum_abc_values_l3518_351857

theorem sum_abc_values (a b c : ℤ) 
  (h1 : a - 2*b = 4) 
  (h2 : a*b + c^2 - 1 = 0) : 
  a + b + c = 5 ∨ a + b + c = 3 ∨ a + b + c = -1 ∨ a + b + c = -3 :=
sorry

end NUMINAMATH_CALUDE_sum_abc_values_l3518_351857


namespace NUMINAMATH_CALUDE_exists_rational_less_than_neg_half_l3518_351864

theorem exists_rational_less_than_neg_half : ∃ q : ℚ, q < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_exists_rational_less_than_neg_half_l3518_351864


namespace NUMINAMATH_CALUDE_number_of_green_balls_l3518_351839

/-- Given a bag with blue and green balls, prove the number of green balls -/
theorem number_of_green_balls
  (blue_balls : ℕ)
  (total_balls : ℕ)
  (h_blue_balls : blue_balls = 9)
  (h_prob_blue : blue_balls / total_balls = 3 / 10)
  (h_total : total_balls = blue_balls + green_balls)
  (green_balls : ℕ) :
  green_balls = 21 := by
  sorry

#check number_of_green_balls

end NUMINAMATH_CALUDE_number_of_green_balls_l3518_351839


namespace NUMINAMATH_CALUDE_circle_equation_and_extrema_l3518_351859

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line x + y + 5 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + 5 = 0}

theorem circle_equation_and_extrema 
  (C : ℝ × ℝ) 
  (h1 : C ∈ Line) 
  (h2 : (0, 2) ∈ Circle C ((0 - C.1)^2 + (2 - C.2)^2).sqrt) 
  (h3 : (1, 1) ∈ Circle C ((1 - C.1)^2 + (1 - C.2)^2).sqrt) :
  (∃ (r : ℝ), Circle C r = {p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 + 2)^2 = 25}) ∧ 
  (∀ (P : ℝ × ℝ), P ∈ Circle C ((0 - C.1)^2 + (2 - C.2)^2).sqrt → 
    3 * P.1 - 4 * P.2 ≤ 24) ∧
  (∀ (P : ℝ × ℝ), P ∈ Circle C ((0 - C.1)^2 + (2 - C.2)^2).sqrt → 
    3 * P.1 - 4 * P.2 ≥ -26) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_and_extrema_l3518_351859


namespace NUMINAMATH_CALUDE_product_equals_fraction_l3518_351880

/-- The decimal representation of the repeating decimal 0.456̅ -/
def repeating_decimal : ℚ := 152 / 333

/-- The product of the repeating decimal 0.456̅ and 7 -/
def product : ℚ := repeating_decimal * 7

/-- Theorem stating that the product of 0.456̅ and 7 is equal to 1064/333 -/
theorem product_equals_fraction : product = 1064 / 333 := by sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l3518_351880


namespace NUMINAMATH_CALUDE_two_circles_in_triangle_l3518_351892

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle touches two sides of a triangle --/
def touchesTwoSides (c : Circle) (t : Triangle) : Prop := sorry

/-- Predicate to check if two circles touch each other --/
def circlesAreEqual (c1 c2 : Circle) : Prop := c1.radius = c2.radius

/-- Predicate to check if two circles touch each other --/
def circlesAreInscribed (c1 c2 : Circle) (t : Triangle) : Prop :=
  touchesTwoSides c1 t ∧ touchesTwoSides c2 t ∧ circlesAreEqual c1 c2

/-- Theorem stating that two equal circles can be inscribed in a triangle --/
theorem two_circles_in_triangle (t : Triangle) :
  ∃ c1 c2 : Circle, circlesAreInscribed c1 c2 t := by sorry

end NUMINAMATH_CALUDE_two_circles_in_triangle_l3518_351892


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l3518_351870

theorem max_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hax : a^x = 6) (hby : b^y = 6) 
  (hab : a + b = 2 * Real.sqrt 6) : 
  (∀ x' y' a' b' : ℝ, 
    a' > 1 → b' > 1 → 
    a'^x' = 6 → b'^y' = 6 → 
    a' + b' = 2 * Real.sqrt 6 → 
    1/x + 1/y ≥ 1/x' + 1/y') ∧ 
  (∃ x₀ y₀ a₀ b₀ : ℝ, 
    a₀ > 1 ∧ b₀ > 1 ∧ 
    a₀^x₀ = 6 ∧ b₀^y₀ = 6 ∧ 
    a₀ + b₀ = 2 * Real.sqrt 6 ∧ 
    1/x₀ + 1/y₀ = 1) :=
by sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l3518_351870


namespace NUMINAMATH_CALUDE_no_intersection_absolute_value_graphs_l3518_351809

theorem no_intersection_absolute_value_graphs : 
  ∀ x : ℝ, ¬(|3 * x + 6| = -|4 * x - 1|) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_absolute_value_graphs_l3518_351809


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3518_351848

theorem product_of_three_numbers (a b c : ℕ) : 
  (a * b * c = 224) ∧ 
  (a < b) ∧ (b < c) ∧ 
  (a * 2 = c) ∧
  (∀ x y z : ℕ, x * y * z = 224 ∧ x < y ∧ y < z ∧ x * 2 = z → x = a ∧ y = b ∧ z = c) :=
by sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3518_351848


namespace NUMINAMATH_CALUDE_teacher_age_l3518_351846

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) :
  num_students = 100 →
  student_avg_age = 17 →
  total_avg_age = 18 →
  (num_students : ℝ) * student_avg_age + (num_students + 1 : ℝ) * total_avg_age - (num_students : ℝ) * student_avg_age = 118 :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l3518_351846


namespace NUMINAMATH_CALUDE_win_sector_area_l3518_351835

/-- The area of a WIN sector on a circular spinner --/
theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/3) :
  p * π * r^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l3518_351835


namespace NUMINAMATH_CALUDE_line_point_value_l3518_351819

/-- Given a line containing points (2, 9), (15, m), and (35, 4), prove that m = 232/33 -/
theorem line_point_value (m : ℚ) : 
  (∃ (line : ℝ → ℝ), line 2 = 9 ∧ line 15 = m ∧ line 35 = 4) → m = 232/33 := by
  sorry

end NUMINAMATH_CALUDE_line_point_value_l3518_351819


namespace NUMINAMATH_CALUDE_five_star_seven_l3518_351816

/-- The star operation defined as (a + b + 3)^2 -/
def star (a b : ℕ) : ℕ := (a + b + 3)^2

/-- Theorem stating that 5 ★ 7 = 225 -/
theorem five_star_seven : star 5 7 = 225 := by
  sorry

end NUMINAMATH_CALUDE_five_star_seven_l3518_351816


namespace NUMINAMATH_CALUDE_valid_schedules_count_l3518_351811

/-- The number of employees and days -/
def n : ℕ := 7

/-- Calculate the number of valid schedules -/
def validSchedules : ℕ :=
  n.factorial - 2 * (n - 1).factorial

/-- Theorem stating the number of valid schedules -/
theorem valid_schedules_count :
  validSchedules = 3600 := by sorry

end NUMINAMATH_CALUDE_valid_schedules_count_l3518_351811


namespace NUMINAMATH_CALUDE_priyas_age_l3518_351812

theorem priyas_age (P F : ℕ) : 
  F = P + 31 →
  (P + 8) + (F + 8) = 69 →
  P = 11 :=
by sorry

end NUMINAMATH_CALUDE_priyas_age_l3518_351812


namespace NUMINAMATH_CALUDE_percentage_proof_l3518_351838

/-- Given a number N and a percentage P, proves that P is 50% when N is 456 and P% of N equals 40% of 120 plus 180. -/
theorem percentage_proof (N : ℝ) (P : ℝ) : 
  N = 456 →
  (P / 100) * N = (40 / 100) * 120 + 180 →
  P = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_proof_l3518_351838


namespace NUMINAMATH_CALUDE_power_sum_equality_l3518_351889

theorem power_sum_equality : (-1)^51 + 3^(2^3 + 5^2 - 7^2) = -1 + 1 / 43046721 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3518_351889


namespace NUMINAMATH_CALUDE_carbonic_acid_weight_is_62_024_l3518_351824

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.011

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 15.999

/-- The molecular formula of carbonic acid -/
structure CarbenicAcid where
  hydrogen : ℕ := 2
  carbon : ℕ := 1
  oxygen : ℕ := 3

/-- The molecular weight of carbonic acid in atomic mass units (amu) -/
def carbonic_acid_weight (acid : CarbenicAcid) : ℝ :=
  acid.hydrogen * hydrogen_weight + 
  acid.carbon * carbon_weight + 
  acid.oxygen * oxygen_weight

/-- Theorem stating that the molecular weight of carbonic acid is 62.024 amu -/
theorem carbonic_acid_weight_is_62_024 :
  carbonic_acid_weight { } = 62.024 := by
  sorry

end NUMINAMATH_CALUDE_carbonic_acid_weight_is_62_024_l3518_351824


namespace NUMINAMATH_CALUDE_apple_sales_remaining_fraction_l3518_351806

/-- Proves that the fraction of money remaining after repairs is 1/5 --/
theorem apple_sales_remaining_fraction (apple_price : ℚ) (bike_cost : ℚ) (repair_percentage : ℚ) (apples_sold : ℕ) :
  apple_price = 5/4 →
  bike_cost = 80 →
  repair_percentage = 1/4 →
  apples_sold = 20 →
  let total_earnings := apple_price * apples_sold
  let repair_cost := repair_percentage * bike_cost
  let remaining := total_earnings - repair_cost
  remaining / total_earnings = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_apple_sales_remaining_fraction_l3518_351806


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_l3518_351803

/-- Represents a number formed by repeating a pattern a certain number of times -/
def repeatedPattern (pattern : ℕ) (repetitions : ℕ) : ℕ :=
  sorry

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem sum_of_digits_of_product : 
  let a := repeatedPattern 15 1004 * repeatedPattern 3 52008
  sumOfDigits a = 18072 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_l3518_351803


namespace NUMINAMATH_CALUDE_contrapositive_odd_product_l3518_351805

theorem contrapositive_odd_product (a b : ℤ) :
  (¬(Odd (a * b)) → ¬(Odd a ∧ Odd b)) ↔
  ((Odd a ∧ Odd b) → Odd (a * b)) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_odd_product_l3518_351805


namespace NUMINAMATH_CALUDE_cos_negative_75_degrees_l3518_351820

theorem cos_negative_75_degrees :
  Real.cos (-(75 * π / 180)) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_75_degrees_l3518_351820


namespace NUMINAMATH_CALUDE_kim_pizza_purchase_l3518_351885

/-- Given that Kim buys pizzas where each pizza has 12 slices, 
    the total cost is $72, and 5 slices cost $10, 
    prove that Kim bought 3 pizzas. -/
theorem kim_pizza_purchase : 
  ∀ (slices_per_pizza : ℕ) (total_cost : ℚ) (five_slice_cost : ℚ),
    slices_per_pizza = 12 →
    total_cost = 72 →
    five_slice_cost = 10 →
    (total_cost / (slices_per_pizza * (five_slice_cost / 5))) = 3 := by
  sorry

#check kim_pizza_purchase

end NUMINAMATH_CALUDE_kim_pizza_purchase_l3518_351885


namespace NUMINAMATH_CALUDE_both_selected_probability_l3518_351852

theorem both_selected_probability (p_ram p_ravi : ℚ) 
  (h_ram : p_ram = 5 / 7)
  (h_ravi : p_ravi = 1 / 5) :
  p_ram * p_ravi = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_both_selected_probability_l3518_351852


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_binomial_expansion_l3518_351829

theorem fourth_term_coefficient_binomial_expansion :
  let n : ℕ := 5
  let a : ℤ := 2
  let b : ℤ := -3
  let k : ℕ := 3  -- For the fourth term, we choose 3 from 5
  (n.choose k) * a^(n - k) * b^k = 720 :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_binomial_expansion_l3518_351829


namespace NUMINAMATH_CALUDE_shirt_tie_outfits_l3518_351844

theorem shirt_tie_outfits (shirts : ℕ) (ties : ℕ) (h1 : shirts = 8) (h2 : ties = 6) :
  shirts * ties = 48 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_outfits_l3518_351844


namespace NUMINAMATH_CALUDE_boxer_weight_theorem_l3518_351887

def initial_weight : ℝ := 106

def weight_loss_rate_A1 : ℝ := 2
def weight_loss_rate_A2 : ℝ := 3
def weight_loss_duration_A1 : ℝ := 2
def weight_loss_duration_A2 : ℝ := 2

def weight_loss_rate_B : ℝ := 3
def weight_loss_duration_B : ℝ := 3

def weight_loss_rate_C : ℝ := 4
def weight_loss_duration_C : ℝ := 4

def final_weight_A : ℝ := initial_weight - (weight_loss_rate_A1 * weight_loss_duration_A1 + weight_loss_rate_A2 * weight_loss_duration_A2)
def final_weight_B : ℝ := initial_weight - (weight_loss_rate_B * weight_loss_duration_B)
def final_weight_C : ℝ := initial_weight - (weight_loss_rate_C * weight_loss_duration_C)

theorem boxer_weight_theorem :
  final_weight_A = 96 ∧
  final_weight_B = 97 ∧
  final_weight_C = 90 := by
  sorry

end NUMINAMATH_CALUDE_boxer_weight_theorem_l3518_351887


namespace NUMINAMATH_CALUDE_division_problem_l3518_351800

theorem division_problem : 180 / (12 + 13 * 2) = 45 / 19 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3518_351800


namespace NUMINAMATH_CALUDE_probability_of_red_in_C_l3518_351856

-- Define the initial configuration of balls in each box
def box_A : ℕ × ℕ := (2, 1)  -- (red, yellow)
def box_B : ℕ × ℕ := (1, 2)  -- (red, yellow)
def box_C : ℕ × ℕ := (1, 1)  -- (red, yellow)

-- Define the process of transferring balls
def transfer_process : (ℕ × ℕ) → (ℕ × ℕ) → (ℕ × ℕ) → ℚ := sorry

-- Theorem statement
theorem probability_of_red_in_C :
  transfer_process box_A box_B box_C = 17/36 := by sorry

end NUMINAMATH_CALUDE_probability_of_red_in_C_l3518_351856


namespace NUMINAMATH_CALUDE_expression_simplification_l3518_351836

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 / (x - 1) - 1 / (x + 1)) / (2 / ((x - 1)^2)) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3518_351836


namespace NUMINAMATH_CALUDE_net_increase_is_86400_l3518_351854

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℚ := 8

/-- Represents the death rate in people per two seconds -/
def death_rate : ℚ := 6

/-- Calculates the net population increase in one day -/
def net_population_increase (birth_rate death_rate : ℚ) (seconds_per_day : ℕ) : ℚ :=
  (birth_rate - death_rate) / 2 * seconds_per_day

/-- Theorem stating that the net population increase in one day is 86400 -/
theorem net_increase_is_86400 :
  net_population_increase birth_rate death_rate seconds_per_day = 86400 := by
  sorry

end NUMINAMATH_CALUDE_net_increase_is_86400_l3518_351854


namespace NUMINAMATH_CALUDE_no_same_color_neighbors_probability_l3518_351825

-- Define the number of beads for each color
def num_red : Nat := 5
def num_white : Nat := 3
def num_blue : Nat := 2

-- Define the total number of beads
def total_beads : Nat := num_red + num_white + num_blue

-- Define a function to calculate the number of valid arrangements
def valid_arrangements : Nat := 0

-- Define a function to calculate the total number of possible arrangements
def total_arrangements : Nat := Nat.factorial total_beads / (Nat.factorial num_red * Nat.factorial num_white * Nat.factorial num_blue)

-- Theorem: The probability of no two neighboring beads being the same color is 0
theorem no_same_color_neighbors_probability :
  (valid_arrangements : ℚ) / total_arrangements = 0 := by sorry

end NUMINAMATH_CALUDE_no_same_color_neighbors_probability_l3518_351825


namespace NUMINAMATH_CALUDE_both_brothers_selected_probability_l3518_351872

theorem both_brothers_selected_probability 
  (prob_X : ℚ) 
  (prob_Y : ℚ) 
  (h1 : prob_X = 1 / 3) 
  (h2 : prob_Y = 2 / 7) : 
  prob_X * prob_Y = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_both_brothers_selected_probability_l3518_351872


namespace NUMINAMATH_CALUDE_initial_price_increase_l3518_351828

theorem initial_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.25 = 1.4375 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_price_increase_l3518_351828


namespace NUMINAMATH_CALUDE_base9_addition_l3518_351810

-- Define a function to convert a base 9 number to base 10
def base9ToBase10 (n : List Nat) : Nat :=
  n.foldr (fun digit acc => acc * 9 + digit) 0

-- Define a function to convert a base 10 number to base 9
def base10ToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

-- Define the numbers in base 9
def a : List Nat := [2, 5, 4]
def b : List Nat := [6, 2, 7]
def c : List Nat := [5, 0, 3]

-- Define the expected result in base 9
def result : List Nat := [1, 4, 8, 5]

theorem base9_addition :
  base10ToBase9 (base9ToBase10 a + base9ToBase10 b + base9ToBase10 c) = result :=
sorry

end NUMINAMATH_CALUDE_base9_addition_l3518_351810


namespace NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l3518_351882

theorem max_y_coordinate_polar_curve (θ : Real) :
  let r := Real.sin (2 * θ)
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  |y| ≤ 4 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l3518_351882


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3518_351873

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 - 4*y + m = 0 ∧ y = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3518_351873


namespace NUMINAMATH_CALUDE_lower_bound_sum_squares_roots_l3518_351834

/-- A monic polynomial of degree 4 with real coefficients -/
structure MonicPolynomial4 where
  coeffs : Fin 4 → ℝ
  monic : coeffs 0 = 1

/-- The sum of the squares of the roots of a polynomial -/
def sumSquaresRoots (p : MonicPolynomial4) : ℝ := sorry

/-- The theorem statement -/
theorem lower_bound_sum_squares_roots (p : MonicPolynomial4)
  (h1 : p.coeffs 1 = 0)  -- No cubic term
  (h2 : ∃ a₂ : ℝ, p.coeffs 2 = a₂ ∧ p.coeffs 3 = 2 * a₂) :  -- a₃ = 2a₂
  |sumSquaresRoots p| ≥ (1/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_lower_bound_sum_squares_roots_l3518_351834


namespace NUMINAMATH_CALUDE_floor_equation_equivalence_l3518_351833

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The solution set for the equation -/
def solution_set : Set ℝ :=
  {x | x < 0 ∨ x ≥ 2.5}

/-- Theorem stating the equivalence of the equation and the solution set -/
theorem floor_equation_equivalence (x : ℝ) :
  floor (1 / (1 - x)) = floor (1 / (1.5 - x)) ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_floor_equation_equivalence_l3518_351833


namespace NUMINAMATH_CALUDE_car_speed_proof_l3518_351893

/-- Proves that a car's speed is 112.5 km/h if it takes 2 seconds longer to travel 1 km compared to 120 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v - 1 / 120) * 3600 = 2 ↔ v = 112.5 := by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l3518_351893


namespace NUMINAMATH_CALUDE_unique_valid_number_l3518_351863

/-- Represents a four-digit integer as a tuple of its digits -/
def FourDigitInt := (Fin 10 × Fin 10 × Fin 10 × Fin 10)

/-- Converts a pair of digits to a two-digit integer -/
def twoDigitInt (a b : Fin 10) : Nat := 10 * a.val + b.val

/-- Checks if three numbers form a geometric sequence -/
def isGeometricSequence (x y z : Nat) : Prop := ∃ r : ℚ, r > 1 ∧ y = r * x ∧ z = r * y

/-- Predicate for a valid four-digit integer satisfying the problem conditions -/
def isValidNumber (n : FourDigitInt) : Prop :=
  let (a, b, c, d) := n
  a ≠ 0 ∧
  isGeometricSequence (twoDigitInt a b) (twoDigitInt b c) (twoDigitInt c d)

theorem unique_valid_number :
  ∃! n : FourDigitInt, isValidNumber n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3518_351863


namespace NUMINAMATH_CALUDE_residue_5_2023_mod_11_l3518_351855

theorem residue_5_2023_mod_11 : 5^2023 ≡ 4 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_residue_5_2023_mod_11_l3518_351855


namespace NUMINAMATH_CALUDE_perpendicular_and_equal_intercepts_l3518_351851

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def line3 (x y : ℝ) : Prop := 3 * x + 4 * y - 15 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (3, 2)

-- Define the perpendicular line l1
def l1 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0

-- Define the two possible lines with equal intercepts
def l2_case1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def l2_case2 (x y : ℝ) : Prop := x + y - 5 = 0

theorem perpendicular_and_equal_intercepts :
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = P) →
  (∀ x y : ℝ, l1 x y → (x, y) = P ∨ (3 * x + 4 * y ≠ 15)) ∧
  ((∀ x y : ℝ, l2_case1 x y → (x, y) = P) ∨ (∀ x y : ℝ, l2_case2 x y → (x, y) = P)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_and_equal_intercepts_l3518_351851


namespace NUMINAMATH_CALUDE_min_value_of_complex_expression_l3518_351862

theorem min_value_of_complex_expression (Z : ℂ) (h : Complex.abs Z = 1) :
  ∃ (min_val : ℝ), min_val = 0 ∧ ∀ (W : ℂ), Complex.abs W = 1 → Complex.abs (W^2 - 2*W + 1) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_of_complex_expression_l3518_351862


namespace NUMINAMATH_CALUDE_world_cup_gifts_l3518_351832

/-- Calculates the number of gifts needed for a world cup inauguration event. -/
def gifts_needed (num_teams : ℕ) : ℕ :=
  num_teams * 2

/-- Theorem: The number of gifts needed for the world cup inauguration event with 7 teams is 14. -/
theorem world_cup_gifts : gifts_needed 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_world_cup_gifts_l3518_351832


namespace NUMINAMATH_CALUDE_last_remaining_number_l3518_351883

/-- Represents the state of a number in Melanie's list -/
inductive NumberState
  | Unmarked
  | Marked
  | Eliminated

/-- Represents a round in Melanie's process -/
structure Round where
  skipCount : Nat
  startNumber : Nat

/-- The list of numbers Melanie works with -/
def initialList : List Nat := List.range 50

/-- Applies the marking and skipping pattern for a single round -/
def applyRound (list : List (Nat × NumberState)) (round : Round) : List (Nat × NumberState) :=
  sorry

/-- Applies all rounds until only one number remains unmarked -/
def applyAllRounds (list : List (Nat × NumberState)) : Nat :=
  sorry

/-- The main theorem stating that the last remaining number is 47 -/
theorem last_remaining_number :
  applyAllRounds (initialList.map (λ n => (n + 1, NumberState.Unmarked))) = 47 :=
sorry

end NUMINAMATH_CALUDE_last_remaining_number_l3518_351883


namespace NUMINAMATH_CALUDE_accounting_balance_l3518_351817

/-- Given the equation 3q - x = 15000, where q = 7 and x = 7 + 75i, prove that p = 5005 + 25i -/
theorem accounting_balance (q x p : ℂ) : 
  3 * q - x = 15000 → q = 7 → x = 7 + 75 * Complex.I → p = 5005 + 25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_accounting_balance_l3518_351817


namespace NUMINAMATH_CALUDE_transfer_increases_averages_l3518_351890

/-- Represents a group of students with their average grade and count -/
structure StudentGroup where
  avg_grade : ℝ
  count : ℕ

/-- Checks if transferring students increases average grades in both groups -/
def increases_averages (group_a group_b : StudentGroup) (grade1 grade2 : ℝ) : Prop :=
  let new_a := StudentGroup.mk
    ((group_a.avg_grade * group_a.count - grade1 - grade2) / (group_a.count - 2))
    (group_a.count - 2)
  let new_b := StudentGroup.mk
    ((group_b.avg_grade * group_b.count + grade1 + grade2) / (group_b.count + 2))
    (group_b.count + 2)
  new_a.avg_grade > group_a.avg_grade ∧ new_b.avg_grade > group_b.avg_grade

theorem transfer_increases_averages :
  let group_a := StudentGroup.mk 44.2 10
  let group_b := StudentGroup.mk 38.8 10
  let grade1 := 41
  let grade2 := 44
  increases_averages group_a group_b grade1 grade2 := by
  sorry

end NUMINAMATH_CALUDE_transfer_increases_averages_l3518_351890


namespace NUMINAMATH_CALUDE_inequality_proof_l3518_351886

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (a * c) ≤ a + b + c) ∧
  (a + b + c = 1 → (2 * a * b) / (a + b) + (2 * b * c) / (b + c) + (2 * c * a) / (c + a) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3518_351886


namespace NUMINAMATH_CALUDE_spoke_forms_surface_l3518_351888

/-- Represents a spoke in a bicycle wheel -/
structure Spoke :=
  (length : ℝ)
  (angle : ℝ)

/-- Represents a rotating bicycle wheel -/
structure RotatingWheel :=
  (radius : ℝ)
  (angular_velocity : ℝ)
  (spokes : List Spoke)

/-- Represents the surface formed by rotating spokes -/
def SurfaceFormedBySpokes (wheel : RotatingWheel) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem stating that a rotating spoke forms a surface -/
theorem spoke_forms_surface (wheel : RotatingWheel) (s : Spoke) 
  (h : s ∈ wheel.spokes) : 
  ∃ (surface : Set (ℝ × ℝ × ℝ)), 
    surface = SurfaceFormedBySpokes wheel ∧ 
    (∀ t : ℝ, ∃ p : ℝ × ℝ × ℝ, p ∈ surface) :=
sorry

end NUMINAMATH_CALUDE_spoke_forms_surface_l3518_351888


namespace NUMINAMATH_CALUDE_card_pair_probability_l3518_351845

/-- Represents the number of cards for each value in the deck -/
def cardsPerValue : ℕ := 5

/-- Represents the number of different values in the deck -/
def numValues : ℕ := 10

/-- Represents the total number of cards in the original deck -/
def totalCards : ℕ := cardsPerValue * numValues

/-- Represents the number of pairs removed -/
def pairsRemoved : ℕ := 2

/-- Represents the number of cards remaining after removal -/
def remainingCards : ℕ := totalCards - (2 * pairsRemoved)

/-- Represents the number of values with full sets of cards after removal -/
def fullSets : ℕ := numValues - pairsRemoved

/-- Represents the number of values with reduced sets of cards after removal -/
def reducedSets : ℕ := pairsRemoved

theorem card_pair_probability :
  (fullSets * (cardsPerValue.choose 2) + reducedSets * ((cardsPerValue - 2).choose 2)) /
  (remainingCards.choose 2) = 86 / 1035 := by
  sorry

end NUMINAMATH_CALUDE_card_pair_probability_l3518_351845


namespace NUMINAMATH_CALUDE_milk_carton_volume_l3518_351875

theorem milk_carton_volume (surface_area : ℝ) (h : surface_area = 600) :
  let side_length := Real.sqrt (surface_area / 6)
  side_length ^ 3 = 1000 := by
sorry

end NUMINAMATH_CALUDE_milk_carton_volume_l3518_351875


namespace NUMINAMATH_CALUDE_lisa_process_ends_at_39_l3518_351847

/-- The function that represents one step of Lisa's process -/
def f (x : ℕ) : ℕ :=
  (x / 10) + 4 * (x % 10)

/-- The sequence of numbers generated by Lisa's process -/
def lisa_sequence (x : ℕ) : ℕ → ℕ
  | 0 => x
  | n + 1 => f (lisa_sequence x n)

/-- The theorem stating that Lisa's process always ends at 39 when starting with 53^2022 - 1 -/
theorem lisa_process_ends_at_39 :
  ∃ n : ℕ, ∀ m : ℕ, m ≥ n → lisa_sequence (53^2022 - 1) m = 39 :=
sorry

end NUMINAMATH_CALUDE_lisa_process_ends_at_39_l3518_351847


namespace NUMINAMATH_CALUDE_total_money_found_l3518_351891

-- Define the amount each person receives
def individual_share : ℝ := 32.50

-- Define the number of people sharing the money
def number_of_people : ℕ := 2

-- Theorem to prove
theorem total_money_found (even_split : ℝ → ℕ → ℝ) :
  even_split individual_share number_of_people = 65.00 :=
by sorry

end NUMINAMATH_CALUDE_total_money_found_l3518_351891


namespace NUMINAMATH_CALUDE_mean_of_two_numbers_l3518_351860

theorem mean_of_two_numbers (a b c : ℝ) : 
  (a + b + c + 100) / 4 = 90 →
  a = 70 →
  a ≤ b ∧ b ≤ c ∧ c ≤ 100 →
  (b + c) / 2 = 95 := by
sorry

end NUMINAMATH_CALUDE_mean_of_two_numbers_l3518_351860


namespace NUMINAMATH_CALUDE_unique_solution_system_l3518_351858

theorem unique_solution_system (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + x*y + y^2) * (y^2 + y*z + z^2) * (z^2 + z*x + x^2) = x*y*z ∧
  (x^4 + x^2*y^2 + y^4) * (y^4 + y^2*z^2 + z^4) * (z^4 + z^2*x^2 + x^4) = x^3*y^3*z^3 →
  x = 1/3 ∧ y = 1/3 ∧ z = 1/3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3518_351858


namespace NUMINAMATH_CALUDE_water_percentage_in_container_l3518_351853

/-- Proves that the percentage of a container's capacity filled with 8 liters of water is 20%,
    given that the total capacity of 40 such containers is 1600 liters. -/
theorem water_percentage_in_container (container_capacity : ℝ) : 
  (40 * container_capacity = 1600) → (8 / container_capacity * 100 = 20) := by
  sorry

end NUMINAMATH_CALUDE_water_percentage_in_container_l3518_351853


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_iff_m_less_than_one_l3518_351830

/-- A point P(x, y) is in the third quadrant if both x and y are negative -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The x-coordinate of point P as a function of m -/
def x_coord (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point P as a function of m -/
def y_coord (m : ℝ) : ℝ := 2 * m - 3

/-- Theorem stating that for point P(m-1, 2m-3) to be in the third quadrant, m must be less than 1 -/
theorem point_in_third_quadrant_iff_m_less_than_one (m : ℝ) : 
  in_third_quadrant (x_coord m) (y_coord m) ↔ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_iff_m_less_than_one_l3518_351830


namespace NUMINAMATH_CALUDE_max_points_in_configuration_l3518_351815

/-- A configuration of points in the plane with associated real numbers -/
structure PointConfiguration where
  n : ℕ
  points : Fin n → ℝ × ℝ
  radii : Fin n → ℝ
  distance_property : ∀ (i j : Fin n), i ≠ j →
    Real.sqrt ((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2 = radii i + radii j

/-- The maximum number of points in a valid configuration is 4 -/
theorem max_points_in_configuration :
  (∃ (c : PointConfiguration), c.n = 4) ∧
  (∀ (c : PointConfiguration), c.n ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_points_in_configuration_l3518_351815


namespace NUMINAMATH_CALUDE_pipe_stack_total_l3518_351874

/-- Calculates the total number of pipes in a trapezoidal stack -/
def total_pipes (layers : ℕ) (bottom : ℕ) (top : ℕ) : ℕ :=
  (bottom + top) * layers / 2

/-- Proves that a trapezoidal stack of pipes with given parameters contains 88 pipes -/
theorem pipe_stack_total : total_pipes 11 13 3 = 88 := by
  sorry

end NUMINAMATH_CALUDE_pipe_stack_total_l3518_351874


namespace NUMINAMATH_CALUDE_intersection_parameter_value_l3518_351818

/-- Given two lines that intersect at a specific x-coordinate, 
    prove the value of the parameter m in the first line equation. -/
theorem intersection_parameter_value 
  (x : ℝ) 
  (h1 : x = -7.5) 
  (h2 : ∃ y : ℝ, 3 * x - y = m ∧ -0.4 * x + y = 3) : 
  m = -22.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_parameter_value_l3518_351818


namespace NUMINAMATH_CALUDE_rhombus_area_l3518_351843

/-- The area of a rhombus with side length 3 cm and one angle measuring 45° is (9√2)/2 square cm. -/
theorem rhombus_area (side : ℝ) (angle : ℝ) :
  side = 3 →
  angle = π / 4 →
  let height : ℝ := side / Real.sqrt 2
  let area : ℝ := side * height
  area = (9 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3518_351843


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3518_351865

theorem trig_expression_equality (α : Real) 
  (h : Real.tan α / (Real.tan α - 1) = -1) : 
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3518_351865


namespace NUMINAMATH_CALUDE_sphere_dihedral_angle_segment_fraction_l3518_351894

/-- The fraction of the segment AB that lies outside two equal touching spheres inscribed in a dihedral angle -/
theorem sphere_dihedral_angle_segment_fraction (α : Real) : 
  α > 0 → α < π → 
  let f := (1 - (Real.cos (α / 2))^2) / (1 + (Real.cos (α / 2))^2)
  0 ≤ f ∧ f ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sphere_dihedral_angle_segment_fraction_l3518_351894


namespace NUMINAMATH_CALUDE_equal_charge_at_120_minutes_l3518_351877

/-- United Telephone's base rate in dollars -/
def united_base : ℚ := 6

/-- United Telephone's per-minute rate in dollars -/
def united_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℚ := 12

/-- Atlantic Call's per-minute rate in dollars -/
def atlantic_per_minute : ℚ := 1/5

/-- The number of minutes at which both companies charge the same amount -/
def equal_charge_minutes : ℚ := 120

theorem equal_charge_at_120_minutes :
  united_base + united_per_minute * equal_charge_minutes =
  atlantic_base + atlantic_per_minute * equal_charge_minutes :=
sorry

end NUMINAMATH_CALUDE_equal_charge_at_120_minutes_l3518_351877


namespace NUMINAMATH_CALUDE_percentage_problem_l3518_351807

theorem percentage_problem (n : ℝ) : (0.1 * 0.3 * 0.5 * n = 90) → n = 6000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3518_351807


namespace NUMINAMATH_CALUDE_triangle_side_length_l3518_351868

theorem triangle_side_length (side2 side3 perimeter : ℝ) 
  (h1 : side2 = 10)
  (h2 : side3 = 15)
  (h3 : perimeter = 32) :
  ∃ side1 : ℝ, side1 + side2 + side3 = perimeter ∧ side1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3518_351868


namespace NUMINAMATH_CALUDE_perfect_square_factorization_l3518_351899

theorem perfect_square_factorization (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factorization_l3518_351899


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l3518_351850

/-- The number of zeros between the decimal point and the first non-zero digit
    in the decimal representation of 7/8000 -/
def zeros_before_first_nonzero : ℕ :=
  3

/-- The fraction we're considering -/
def fraction : ℚ :=
  7 / 8000

theorem zeros_before_first_nonzero_of_fraction :
  zeros_before_first_nonzero = 3 :=
sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l3518_351850


namespace NUMINAMATH_CALUDE_coke_calories_is_215_l3518_351840

/-- Represents the calorie content of various food items and meals --/
structure CalorieContent where
  cake : ℕ
  chips : ℕ
  breakfast : ℕ
  lunch : ℕ
  dailyLimit : ℕ
  remainingAfterCoke : ℕ

/-- Calculates the calorie content of the coke --/
def cokeCalories (c : CalorieContent) : ℕ :=
  c.dailyLimit - (c.cake + c.chips + c.breakfast + c.lunch) - c.remainingAfterCoke

/-- Theorem stating that the coke has 215 calories --/
theorem coke_calories_is_215 (c : CalorieContent) 
  (h1 : c.cake = 110)
  (h2 : c.chips = 310)
  (h3 : c.breakfast = 560)
  (h4 : c.lunch = 780)
  (h5 : c.dailyLimit = 2500)
  (h6 : c.remainingAfterCoke = 525) :
  cokeCalories c = 215 := by
  sorry

#eval cokeCalories { cake := 110, chips := 310, breakfast := 560, lunch := 780, dailyLimit := 2500, remainingAfterCoke := 525 }

end NUMINAMATH_CALUDE_coke_calories_is_215_l3518_351840


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3518_351831

/-- Given a quadratic equation x^2 + 2 = 3x, prove that the coefficient of x^2 is 1 and the coefficient of x is -3. -/
theorem quadratic_equation_coefficients :
  let eq : ℝ → Prop := λ x => x^2 + 2 = 3*x
  ∃ a b c : ℝ, (∀ x, eq x ↔ a*x^2 + b*x + c = 0) ∧ a = 1 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3518_351831


namespace NUMINAMATH_CALUDE_two_triangle_range_l3518_351871

theorem two_triangle_range (A B C : ℝ) (a b c : ℝ) :
  A = Real.pi / 3 →  -- 60 degrees in radians
  a = Real.sqrt 3 →
  b = x →
  (∃ (x : ℝ), ∀ B, 
    Real.pi / 3 < B ∧ B < 2 * Real.pi / 3 →  -- 60° < B < 120°
    Real.sin B = x / 2 →
    x > Real.sqrt 3 ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_two_triangle_range_l3518_351871
