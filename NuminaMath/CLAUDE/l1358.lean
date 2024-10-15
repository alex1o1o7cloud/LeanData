import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_numbers_square_l1358_135831

theorem consecutive_numbers_square (a : ℕ) : 
  let b := a + 1
  let c := a * b
  let x := a^2 + b^2 + c^2
  ∃ (k : ℕ), x = (2*k + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_square_l1358_135831


namespace NUMINAMATH_CALUDE_marly_soup_bags_l1358_135843

/-- Calculates the number of bags needed to hold Marly's soup -/
def bags_needed (milk : ℚ) (chicken_stock_ratio : ℚ) (vegetables : ℚ) (bag_capacity : ℚ) : ℚ :=
  let total_volume := milk + (chicken_stock_ratio * milk) + vegetables
  total_volume / bag_capacity

/-- Proves that Marly needs 3 bags for his soup -/
theorem marly_soup_bags :
  bags_needed 2 3 1 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_marly_soup_bags_l1358_135843


namespace NUMINAMATH_CALUDE_income_percentage_l1358_135884

/-- Given that Mart's income is 60% more than Tim's income, and Tim's income is 60% less than Juan's income, 
    prove that Mart's income is 64% of Juan's income. -/
theorem income_percentage (juan tim mart : ℝ) 
  (h1 : tim = 0.4 * juan)  -- Tim's income is 60% less than Juan's
  (h2 : mart = 1.6 * tim)  -- Mart's income is 60% more than Tim's
  : mart = 0.64 * juan := by
  sorry


end NUMINAMATH_CALUDE_income_percentage_l1358_135884


namespace NUMINAMATH_CALUDE_paint_cube_cost_l1358_135858

/-- The cost to paint a cube given paint price, coverage, and cube dimensions -/
theorem paint_cube_cost (paint_price : ℝ) (paint_coverage : ℝ) (cube_side : ℝ) : 
  paint_price = 36.5 →
  paint_coverage = 16 →
  cube_side = 8 →
  6 * cube_side^2 / paint_coverage * paint_price = 876 := by
sorry

end NUMINAMATH_CALUDE_paint_cube_cost_l1358_135858


namespace NUMINAMATH_CALUDE_fraction_addition_l1358_135837

theorem fraction_addition : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1358_135837


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1358_135856

def quadratic_function (x k : ℝ) : ℝ := -2 * x^2 + 4 * x + k

theorem quadratic_inequality (k : ℝ) :
  let x1 : ℝ := -0.99
  let x2 : ℝ := 0.98
  let x3 : ℝ := 0.99
  let y1 : ℝ := quadratic_function x1 k
  let y2 : ℝ := quadratic_function x2 k
  let y3 : ℝ := quadratic_function x3 k
  y1 < y2 ∧ y2 < y3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1358_135856


namespace NUMINAMATH_CALUDE_room002_is_selected_l1358_135839

/-- Represents a room number in the range [1, 60] -/
def RoomNumber := Fin 60

/-- The total number of examination rooms -/
def totalRooms : Nat := 60

/-- The number of rooms to be selected for inspection -/
def selectedRooms : Nat := 12

/-- The sample interval for systematic sampling -/
def sampleInterval : Nat := totalRooms / selectedRooms

/-- Predicate to check if a room is selected in the systematic sampling -/
def isSelected (room : RoomNumber) : Prop :=
  ∃ k : Nat, (room.val + 1) = k * sampleInterval + 2

/-- Theorem stating that room 002 is selected given the conditions -/
theorem room002_is_selected :
  isSelected ⟨1, by norm_num⟩ ∧ isSelected ⟨6, by norm_num⟩ := by
  sorry


end NUMINAMATH_CALUDE_room002_is_selected_l1358_135839


namespace NUMINAMATH_CALUDE_function_always_positive_implies_x_range_l1358_135844

theorem function_always_positive_implies_x_range (f : ℝ → ℝ) :
  (∀ a ∈ Set.Icc (-1) 1, ∀ x, f x = x^2 + (a - 4)*x + 4 - 2*a) →
  (∀ a ∈ Set.Icc (-1) 1, ∀ x, f x > 0) →
  ∀ x, f x > 0 → x ∈ Set.union (Set.Iio 1) (Set.Ioi 3) :=
by sorry

end NUMINAMATH_CALUDE_function_always_positive_implies_x_range_l1358_135844


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1358_135863

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 - 4*x + 4 = 0 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1358_135863


namespace NUMINAMATH_CALUDE_max_score_in_twenty_over_match_l1358_135865

/-- Represents the number of overs in the cricket match -/
def overs : ℕ := 20

/-- Represents the number of balls in an over -/
def balls_per_over : ℕ := 6

/-- Represents the maximum runs that can be scored on a single ball -/
def max_runs_per_ball : ℕ := 6

/-- Calculates the maximum runs a batsman can score in a perfect scenario -/
def max_batsman_score : ℕ := overs * balls_per_over * max_runs_per_ball

theorem max_score_in_twenty_over_match :
  max_batsman_score = 720 :=
by sorry

end NUMINAMATH_CALUDE_max_score_in_twenty_over_match_l1358_135865


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_nineteen_twelfths_l1358_135807

theorem sum_of_roots_eq_nineteen_twelfths :
  let f : ℝ → ℝ := λ x ↦ (4*x + 7) * (3*x - 10)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 19/12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_nineteen_twelfths_l1358_135807


namespace NUMINAMATH_CALUDE_tan_8100_degrees_l1358_135833

theorem tan_8100_degrees : Real.tan (8100 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_8100_degrees_l1358_135833


namespace NUMINAMATH_CALUDE_soap_usage_ratio_l1358_135896

/-- Represents the survey results of household soap usage --/
structure SoapSurvey where
  total : ℕ
  neither : ℕ
  onlyA : ℕ
  both : ℕ
  neitherLtTotal : neither < total
  onlyALtTotal : onlyA < total
  bothLtTotal : both < total

/-- Calculates the number of households using only brand B soap --/
def onlyB (s : SoapSurvey) : ℕ :=
  s.total - s.neither - s.onlyA - s.both

/-- Theorem stating the ratio of households using only brand B to those using both brands --/
theorem soap_usage_ratio (s : SoapSurvey)
  (h1 : s.total = 260)
  (h2 : s.neither = 80)
  (h3 : s.onlyA = 60)
  (h4 : s.both = 30) :
  (onlyB s) / s.both = 3 := by
  sorry

end NUMINAMATH_CALUDE_soap_usage_ratio_l1358_135896


namespace NUMINAMATH_CALUDE_max_squares_on_8x8_board_l1358_135851

/-- Represents a checkerboard --/
structure Checkerboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a straight line on a checkerboard --/
structure Line :=
  (board : Checkerboard)

/-- Returns the maximum number of squares a line can pass through on a checkerboard --/
def max_squares_passed (l : Line) : Nat :=
  l.board.rows + l.board.cols - 1

/-- Theorem: The maximum number of squares a straight line can pass through on an 8x8 checkerboard is 15 --/
theorem max_squares_on_8x8_board :
  ∀ (l : Line), l.board = Checkerboard.mk 8 8 → max_squares_passed l = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_squares_on_8x8_board_l1358_135851


namespace NUMINAMATH_CALUDE_sophie_laundry_loads_l1358_135840

/-- Represents the cost of a box of dryer sheets in dollars -/
def box_cost : ℚ := 5.5

/-- Represents the number of dryer sheets in a box -/
def sheets_per_box : ℕ := 104

/-- Represents the amount saved in a year by not buying dryer sheets, in dollars -/
def yearly_savings : ℚ := 11

/-- Represents the number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Represents the number of dryer sheets used per load of laundry -/
def sheets_per_load : ℕ := 1

/-- Theorem stating that Sophie does 4 loads of laundry per week -/
theorem sophie_laundry_loads : 
  ∃ (loads_per_week : ℕ), 
    loads_per_week = 4 ∧ 
    (yearly_savings / box_cost : ℚ) * sheets_per_box = loads_per_week * weeks_per_year :=
sorry

end NUMINAMATH_CALUDE_sophie_laundry_loads_l1358_135840


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l1358_135899

/-- Given a wire cut into two pieces of lengths x and y, where x forms a square and y forms a regular octagon with equal perimeters, prove that x/y = 1 -/
theorem wire_cut_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_square : 4 * (x / 4) = x) 
  (h_octagon : 8 * (y / 8) = y)
  (h_equal_perimeter : 4 * (x / 4) = 8 * (y / 8)) : 
  x / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l1358_135899


namespace NUMINAMATH_CALUDE_slab_rate_per_sq_meter_l1358_135888

/-- Prove that the rate per square meter for paving a rectangular room is 900 Rs. -/
theorem slab_rate_per_sq_meter (length width total_cost : ℝ) : 
  length = 6 →
  width = 4.75 →
  total_cost = 25650 →
  total_cost / (length * width) = 900 := by
sorry

end NUMINAMATH_CALUDE_slab_rate_per_sq_meter_l1358_135888


namespace NUMINAMATH_CALUDE_algebraic_simplification_and_evaluation_l1358_135862

theorem algebraic_simplification_and_evaluation (a b : ℝ) :
  2 * (a * b^2 + 3 * a^2 * b) - 3 * (a * b^2 + a^2 * b) = -a * b^2 + 3 * a^2 * b ∧
  2 * ((-1) * 2^2 + 3 * (-1)^2 * 2) - 3 * ((-1) * 2^2 + (-1)^2 * 2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_simplification_and_evaluation_l1358_135862


namespace NUMINAMATH_CALUDE_arccos_cos_eleven_l1358_135842

theorem arccos_cos_eleven : 
  ∃! x : ℝ, x ∈ Set.Icc 0 π ∧ (x - 11) ∈ Set.range (λ n : ℤ => 2 * π * n) ∧ x = Real.arccos (Real.cos 11) := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eleven_l1358_135842


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l1358_135872

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∃ k : ℕ, (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13) = 15 * k) ∧
  (∀ m : ℕ, m > 15 → ¬(∀ n : ℕ, Even n → n > 0 →
    ∃ k : ℕ, (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13) = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l1358_135872


namespace NUMINAMATH_CALUDE_prob_10_or_7_prob_below_7_l1358_135892

-- Define the probabilities for each ring
def p10 : ℝ := 0.21
def p9 : ℝ := 0.23
def p8 : ℝ := 0.25
def p7 : ℝ := 0.28

-- Theorem for the probability of hitting either 10 ring or 7 ring
theorem prob_10_or_7 : p10 + p7 = 0.49 := by sorry

-- Theorem for the probability of scoring below 7 ring
theorem prob_below_7 : 1 - (p10 + p9 + p8 + p7) = 0.03 := by sorry

end NUMINAMATH_CALUDE_prob_10_or_7_prob_below_7_l1358_135892


namespace NUMINAMATH_CALUDE_price_difference_l1358_135875

/-- The price difference problem -/
theorem price_difference (discount_price : ℝ) (discount_rate : ℝ) (increase_rate : ℝ) : 
  discount_price = 68 ∧ 
  discount_rate = 0.15 ∧ 
  increase_rate = 0.25 →
  ∃ (original_price final_price : ℝ),
    original_price * (1 - discount_rate) = discount_price ∧
    final_price = discount_price * (1 + increase_rate) ∧
    final_price - original_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_l1358_135875


namespace NUMINAMATH_CALUDE_no_arithmetic_sequence_with_arithmetic_digit_sum_l1358_135802

/-- An arithmetic sequence of positive integers. -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ (a₀ d : ℕ), ∀ n, a n = a₀ + n * d

/-- The sum of digits of a natural number. -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that no infinite arithmetic sequence of distinct positive integers
    exists such that the sum of digits of each term also forms an arithmetic sequence. -/
theorem no_arithmetic_sequence_with_arithmetic_digit_sum :
  ¬ ∃ (a : ℕ → ℕ),
    ArithmeticSequence a ∧
    (∀ n m, n ≠ m → a n ≠ a m) ∧
    ArithmeticSequence (λ n => sumOfDigits (a n)) :=
sorry

end NUMINAMATH_CALUDE_no_arithmetic_sequence_with_arithmetic_digit_sum_l1358_135802


namespace NUMINAMATH_CALUDE_fifth_root_of_1024_l1358_135891

theorem fifth_root_of_1024 : (1024 : ℝ) ^ (1/5) = 4 := by sorry

end NUMINAMATH_CALUDE_fifth_root_of_1024_l1358_135891


namespace NUMINAMATH_CALUDE_prob_2012_higher_than_2011_l1358_135847

/-- Probability of guessing the correct answer to each question -/
def p : ℝ := 0.25

/-- Probability of guessing the incorrect answer to each question -/
def q : ℝ := 1 - p

/-- Number of questions in the 2011 exam -/
def n_2011 : ℕ := 20

/-- Number of correct answers required to pass in 2011 -/
def k_2011 : ℕ := 3

/-- Number of questions in the 2012 exam -/
def n_2012 : ℕ := 40

/-- Number of correct answers required to pass in 2012 -/
def k_2012 : ℕ := 6

/-- Probability of passing the exam in 2011 -/
def prob_2011 : ℝ := 1 - (Finset.sum (Finset.range k_2011) (λ i => Nat.choose n_2011 i * p^i * q^(n_2011 - i)))

/-- Probability of passing the exam in 2012 -/
def prob_2012 : ℝ := 1 - (Finset.sum (Finset.range k_2012) (λ i => Nat.choose n_2012 i * p^i * q^(n_2012 - i)))

/-- Theorem stating that the probability of passing in 2012 is higher than in 2011 -/
theorem prob_2012_higher_than_2011 : prob_2012 > prob_2011 := by
  sorry

end NUMINAMATH_CALUDE_prob_2012_higher_than_2011_l1358_135847


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_value_of_m_for_intersection_l1358_135824

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 ≤ 0}

-- Define set B with parameter m
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1: Intersection of A and complement of B when m = 3
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 3) = {x | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Theorem 2: Value of m when A ∩ B = {x | -1 ≤ x < 4}
theorem value_of_m_for_intersection :
  ∃ m : ℝ, A ∩ B m = {x | -1 ≤ x ∧ x < 4} ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_value_of_m_for_intersection_l1358_135824


namespace NUMINAMATH_CALUDE_tetrahedron_volume_from_triangle_l1358_135895

/-- Given a triangle ABC with sides of length 11, 20, and 21 units,
    the volume of the tetrahedron formed by folding the triangle along
    the lines connecting the midpoints of its sides is 45 cubic units. -/
theorem tetrahedron_volume_from_triangle (a b c : ℝ) (h1 : a = 11) (h2 : b = 20) (h3 : c = 21) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let p := b / 2
  let q := c / 2
  let r := a / 2
  let s_mid := (p + q + r) / 2
  let area_mid := Real.sqrt (s_mid * (s_mid - p) * (s_mid - q) * (s_mid - r))
  let height := Real.sqrt ((q^2) - (area_mid^2 / area^2) * (a^2 / 4))
  (1/3) * area_mid * height = 45 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_from_triangle_l1358_135895


namespace NUMINAMATH_CALUDE_smallest_four_digit_sum_27_l1358_135857

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_sum_27 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 27 → 1899 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_sum_27_l1358_135857


namespace NUMINAMATH_CALUDE_download_speed_scientific_notation_l1358_135866

/-- The network download speed of 5G in KB per second -/
def download_speed : ℕ := 1300000

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Convert a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation := sorry

theorem download_speed_scientific_notation :
  to_scientific_notation download_speed = ScientificNotation.mk 1.3 6 := by sorry

end NUMINAMATH_CALUDE_download_speed_scientific_notation_l1358_135866


namespace NUMINAMATH_CALUDE_simplify_expression_l1358_135800

theorem simplify_expression (a : ℝ) : 5*a^2 - (a^2 - 2*(a^2 - 3*a)) = 6*a^2 - 6*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1358_135800


namespace NUMINAMATH_CALUDE_part_one_part_two_l1358_135846

-- Define the sets A, B, and C
def A : Set ℝ := {1, 4, 7, 10}
def B (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 9}
def C : Set ℝ := {x | 3 ≤ x ∧ x ≤ 6}

-- Part 1
theorem part_one :
  (A ∪ B 1 = {x | 1 ≤ x ∧ x ≤ 10}) ∧
  (A ∩ Cᶜ = {1, 7, 10}) := by sorry

-- Part 2
theorem part_two :
  ∀ m : ℝ, (B m ∩ C = C) → (-3 < m ∧ m < 3) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1358_135846


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l1358_135854

/-- Given a man and his son, where the man is 34 years older than his son
    and the son's current age is 32, proves that the ratio of their ages
    in two years is 2:1. -/
theorem man_son_age_ratio :
  ∀ (son_age man_age : ℕ),
  son_age = 32 →
  man_age = son_age + 34 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l1358_135854


namespace NUMINAMATH_CALUDE_log3_20_approximation_l1358_135873

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_5_approx : ℝ := 0.699

-- Define the target fraction
def target_fraction : ℚ := 33 / 12

-- Theorem statement
theorem log3_20_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |Real.log 20 / Real.log 3 - target_fraction| < ε :=
sorry

end NUMINAMATH_CALUDE_log3_20_approximation_l1358_135873


namespace NUMINAMATH_CALUDE_smallest_a_for_nonprime_polynomial_l1358_135889

theorem smallest_a_for_nonprime_polynomial :
  ∃ (a : ℕ+), (∀ (x : ℤ), ∃ (p q : ℤ), p > 1 ∧ q > 1 ∧ x^4 + (a + 4)^2 = p * q) ∧
  (∀ (b : ℕ+), b < a → ∃ (y : ℤ), ∀ (p q : ℤ), (p > 1 ∧ q > 1 → y^4 + (b + 4)^2 ≠ p * q)) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_nonprime_polynomial_l1358_135889


namespace NUMINAMATH_CALUDE_lyka_initial_money_l1358_135806

/-- Calculates the initial amount of money Lyka has given the cost of a smartphone,
    the saving period in weeks, and the weekly saving rate. -/
def initial_money (smartphone_cost : ℕ) (saving_period : ℕ) (saving_rate : ℕ) : ℕ :=
  smartphone_cost - saving_period * saving_rate

/-- Proves that given a smartphone cost of $160, a saving period of 8 weeks,
    and a saving rate of $15 per week, the initial amount of money Lyka has is $40. -/
theorem lyka_initial_money :
  initial_money 160 8 15 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lyka_initial_money_l1358_135806


namespace NUMINAMATH_CALUDE_radius_is_3_sqrt_13_l1358_135821

/-- Represents a circular sector with an inscribed rectangle -/
structure CircularSectorWithRectangle where
  /-- Radius of the circle -/
  radius : ℝ
  /-- Central angle of the sector in radians -/
  centralAngle : ℝ
  /-- Length of the shorter side of the rectangle -/
  shortSide : ℝ
  /-- Length of the longer side of the rectangle -/
  longSide : ℝ
  /-- The longer side is 3 units longer than the shorter side -/
  sideDifference : longSide = shortSide + 3
  /-- The area of the rectangle is 18 -/
  rectangleArea : shortSide * longSide = 18
  /-- The central angle is 45 degrees (π/4 radians) -/
  angleIs45Degrees : centralAngle = Real.pi / 4

/-- The main theorem stating that the radius is 3√13 -/
theorem radius_is_3_sqrt_13 (sector : CircularSectorWithRectangle) :
  sector.radius = 3 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_radius_is_3_sqrt_13_l1358_135821


namespace NUMINAMATH_CALUDE_division_problem_l1358_135812

theorem division_problem (x : ℝ) 
  (h1 : ∃ k : ℝ, 3*k + 4*k + 5*k + 6*k = x)
  (h2 : ∃ m : ℝ, 3*m + 4*m + 6*m + 7*m = x)
  (h3 : ∃ (k m : ℝ), 6*m + 7*m = 5*k + 6*k + 1400) :
  x = 36000 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1358_135812


namespace NUMINAMATH_CALUDE_maria_water_bottles_l1358_135811

theorem maria_water_bottles (initial bottles_bought bottles_remaining : ℕ) 
  (h1 : initial = 14)
  (h2 : bottles_bought = 45)
  (h3 : bottles_remaining = 51) :
  initial - (bottles_remaining - bottles_bought) = 8 := by
  sorry

end NUMINAMATH_CALUDE_maria_water_bottles_l1358_135811


namespace NUMINAMATH_CALUDE_landscape_length_is_120_l1358_135826

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  playgroundArea : ℝ
  playgroundRatio : ℝ

/-- The length of the landscape is 4 times its breadth -/
def Landscape.length (l : Landscape) : ℝ := 4 * l.breadth

/-- The total area of the landscape -/
def Landscape.totalArea (l : Landscape) : ℝ := l.length * l.breadth

/-- Theorem: Given a landscape with specific properties, its length is 120 meters -/
theorem landscape_length_is_120 (l : Landscape) 
    (h1 : l.playgroundArea = 1200)
    (h2 : l.playgroundRatio = 1/3)
    (h3 : l.playgroundArea = l.playgroundRatio * l.totalArea) : 
  l.length = 120 := by
  sorry

#check landscape_length_is_120

end NUMINAMATH_CALUDE_landscape_length_is_120_l1358_135826


namespace NUMINAMATH_CALUDE_max_value_expression_l1358_135804

theorem max_value_expression : 
  (∀ x : ℝ, (3 * x^2 + 9 * x + 28) / (3 * x^2 + 9 * x + 7) ≤ 85) ∧ 
  (∀ ε > 0, ∃ x : ℝ, (3 * x^2 + 9 * x + 28) / (3 * x^2 + 9 * x + 7) > 85 - ε) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1358_135804


namespace NUMINAMATH_CALUDE_square_root_of_increased_number_l1358_135887

theorem square_root_of_increased_number (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x^2 + 2) = Real.sqrt ((Real.sqrt x^2) + 2) :=
by sorry

end NUMINAMATH_CALUDE_square_root_of_increased_number_l1358_135887


namespace NUMINAMATH_CALUDE_solve_inequality_for_x_find_k_range_l1358_135859

-- Part 1
theorem solve_inequality_for_x (x : ℝ) :
  (|1 - x * 2| > |x - 2|) ↔ (x < -1 ∨ x > 1) := by sorry

-- Part 2
theorem find_k_range (k : ℝ) :
  (∀ x y : ℝ, |x| < 1 → |y| < 1 → |1 - k*x*y| > |k*x - y|) ↔ 
  (k ≥ -1 ∧ k ≤ 1) := by sorry

end NUMINAMATH_CALUDE_solve_inequality_for_x_find_k_range_l1358_135859


namespace NUMINAMATH_CALUDE_heechul_most_books_l1358_135853

/-- The number of books each person has -/
structure BookCollection where
  heejin : ℕ
  heechul : ℕ
  dongkyun : ℕ

/-- Conditions of the book collection -/
def valid_collection (bc : BookCollection) : Prop :=
  bc.heechul = bc.heejin + 2 ∧ bc.dongkyun < bc.heejin

/-- Heechul has the most books -/
def heechul_has_most (bc : BookCollection) : Prop :=
  bc.heechul > bc.heejin ∧ bc.heechul > bc.dongkyun

/-- Theorem: If the collection is valid, then Heechul has the most books -/
theorem heechul_most_books (bc : BookCollection) :
  valid_collection bc → heechul_has_most bc := by
  sorry

end NUMINAMATH_CALUDE_heechul_most_books_l1358_135853


namespace NUMINAMATH_CALUDE_min_redistributions_correct_l1358_135868

/-- Represents the redistribution process for a deck of 8 cards -/
def redistribute (deck : Vector ℕ 8) : Vector ℕ 8 :=
  sorry

/-- Checks if the deck is in its original order -/
def is_original_order (deck original : Vector ℕ 8) : Prop :=
  deck = original

/-- The minimum number of redistributions needed to restore the original order -/
def min_redistributions : ℕ := 3

/-- Theorem stating that the minimum number of redistributions to restore the original order is 3 -/
theorem min_redistributions_correct (original : Vector ℕ 8) :
  ∃ (n : ℕ), n = min_redistributions ∧
  ∀ (m : ℕ), m < n → ¬(is_original_order ((redistribute^[m]) original) original) ∧
  is_original_order ((redistribute^[n]) original) original :=
sorry

end NUMINAMATH_CALUDE_min_redistributions_correct_l1358_135868


namespace NUMINAMATH_CALUDE_bathroom_volume_l1358_135878

theorem bathroom_volume (length width height area volume : ℝ) : 
  length = 4 →
  area = 8 →
  height = 7 →
  area = length * width →
  volume = length * width * height →
  volume = 56 := by
sorry

end NUMINAMATH_CALUDE_bathroom_volume_l1358_135878


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1358_135877

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : 0 < a) (h' : 0 < b) 
  (h_identity : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) :
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1358_135877


namespace NUMINAMATH_CALUDE_consecutive_integers_product_2720_sum_103_l1358_135801

theorem consecutive_integers_product_2720_sum_103 :
  ∀ n : ℕ, n > 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = 103 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_2720_sum_103_l1358_135801


namespace NUMINAMATH_CALUDE_payment_difference_l1358_135836

/-- Represents the pizza and its cost structure -/
structure Pizza :=
  (total_slices : ℕ)
  (plain_cost : ℚ)
  (anchovy_cost : ℚ)
  (mushroom_cost : ℚ)

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : ℚ :=
  p.plain_cost + p.anchovy_cost + p.mushroom_cost

/-- Calculates the number of slices Dave ate -/
def dave_slices (p : Pizza) : ℕ :=
  p.total_slices / 2 + p.total_slices / 4 + 1

/-- Calculates the number of slices Doug ate -/
def doug_slices (p : Pizza) : ℕ :=
  p.total_slices - dave_slices p

/-- Calculates Dave's payment -/
def dave_payment (p : Pizza) : ℚ :=
  total_cost p - (p.plain_cost / p.total_slices) * doug_slices p

/-- Calculates Doug's payment -/
def doug_payment (p : Pizza) : ℚ :=
  (p.plain_cost / p.total_slices) * doug_slices p

/-- The main theorem stating the difference in payments -/
theorem payment_difference (p : Pizza) 
  (h1 : p.total_slices = 8)
  (h2 : p.plain_cost = 8)
  (h3 : p.anchovy_cost = 2)
  (h4 : p.mushroom_cost = 1) :
  dave_payment p - doug_payment p = 9 := by
  sorry

end NUMINAMATH_CALUDE_payment_difference_l1358_135836


namespace NUMINAMATH_CALUDE_five_rows_with_seven_students_l1358_135861

/-- Represents the seating arrangement in a classroom -/
structure Seating :=
  (rows_with_7 : ℕ)
  (rows_with_6 : ℕ)

/-- Checks if a seating arrangement is valid -/
def is_valid_seating (s : Seating) : Prop :=
  s.rows_with_7 * 7 + s.rows_with_6 * 6 = 53

/-- The theorem stating that there are 5 rows with 7 students -/
theorem five_rows_with_seven_students :
  ∃ (s : Seating), is_valid_seating s ∧ s.rows_with_7 = 5 :=
sorry

end NUMINAMATH_CALUDE_five_rows_with_seven_students_l1358_135861


namespace NUMINAMATH_CALUDE_star_diamond_relation_l1358_135874

theorem star_diamond_relation (star diamond : ℤ) 
  (h : 514 - star = 600 - diamond) : 
  star < diamond ∧ diamond - star = 86 := by
  sorry

end NUMINAMATH_CALUDE_star_diamond_relation_l1358_135874


namespace NUMINAMATH_CALUDE_actual_lawn_area_l1358_135893

/-- Actual area of a lawn given its blueprint measurements and scale -/
theorem actual_lawn_area 
  (blueprint_area : ℝ) 
  (blueprint_side : ℝ) 
  (actual_side : ℝ) 
  (h1 : blueprint_area = 300) 
  (h2 : blueprint_side = 5) 
  (h3 : actual_side = 1500) : 
  (actual_side / blueprint_side)^2 * blueprint_area = 2700 * 10000 := by
  sorry

end NUMINAMATH_CALUDE_actual_lawn_area_l1358_135893


namespace NUMINAMATH_CALUDE_average_age_increase_l1358_135819

theorem average_age_increase (initial_count : ℕ) (replaced_age1 replaced_age2 : ℕ) 
  (new_average : ℕ) (h1 : initial_count = 8) (h2 : replaced_age1 = 21) 
  (h3 : replaced_age2 = 23) (h4 : new_average = 30) : 
  let initial_total := initial_count * (initial_count * A - replaced_age1 - replaced_age2) / initial_count
  let new_total := initial_total - replaced_age1 - replaced_age2 + 2 * new_average
  let new_average := new_total / initial_count
  new_average - (initial_total / initial_count) = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l1358_135819


namespace NUMINAMATH_CALUDE_tony_squat_weight_l1358_135832

def curl_weight : ℕ := 90

def military_press_weight (curl : ℕ) : ℕ := 2 * curl

def squat_weight (military_press : ℕ) : ℕ := 5 * military_press

theorem tony_squat_weight : 
  squat_weight (military_press_weight curl_weight) = 900 := by
  sorry

end NUMINAMATH_CALUDE_tony_squat_weight_l1358_135832


namespace NUMINAMATH_CALUDE_younger_brother_silver_fraction_l1358_135848

/-- The fraction of total silver received by the younger brother in a treasure division problem -/
theorem younger_brother_silver_fraction (x y : ℝ) 
  (h1 : x / 5 + y / 7 = 100)  -- Elder brother's share
  (h2 : x / 7 + (700 - x) / 7 = 100)  -- Younger brother's share
  : (700 - x) / (7 * y) = (y - (y - x / 5) / 2) / y := by
  sorry

end NUMINAMATH_CALUDE_younger_brother_silver_fraction_l1358_135848


namespace NUMINAMATH_CALUDE_product_of_cosines_equals_one_eighth_l1358_135870

theorem product_of_cosines_equals_one_eighth :
  (1 + Real.cos (π / 12)) * (1 + Real.cos (5 * π / 12)) *
  (1 + Real.cos (7 * π / 12)) * (1 + Real.cos (11 * π / 12)) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_equals_one_eighth_l1358_135870


namespace NUMINAMATH_CALUDE_michelle_sandwiches_l1358_135828

/-- The number of sandwiches Michelle has left to give to her other co-workers -/
def sandwiches_left (total : ℕ) (first : ℕ) (second : ℕ) : ℕ :=
  total - first - second - (2 * first) - (3 * second)

/-- Proof that Michelle has 26 sandwiches left -/
theorem michelle_sandwiches : sandwiches_left 50 4 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_michelle_sandwiches_l1358_135828


namespace NUMINAMATH_CALUDE_altara_population_2040_l1358_135882

/-- Represents the population of Altara at a given year -/
def population (year : ℕ) : ℕ :=
  sorry

theorem altara_population_2040 :
  (population 2020 = 500) →
  (∀ y : ℕ, y ≥ 2020 → population (y + 10) = 2 * population y) →
  population 2040 = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_altara_population_2040_l1358_135882


namespace NUMINAMATH_CALUDE_floor_div_p_equals_86422_l1358_135897

/-- A function that generates the sequence of 6-digit numbers with digits in non-increasing order -/
def nonIncreasingDigitSequence : ℕ → ℕ := sorry

/-- The 2010th number in the sequence -/
def p : ℕ := nonIncreasingDigitSequence 2010

/-- Theorem stating that the floor division of p by 10 equals 86422 -/
theorem floor_div_p_equals_86422 : p / 10 = 86422 := by sorry

end NUMINAMATH_CALUDE_floor_div_p_equals_86422_l1358_135897


namespace NUMINAMATH_CALUDE_statements_equivalent_l1358_135834

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_angles : angles 0 + angles 1 + angles 2 = π

-- Define an isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.angles 0 = t.angles 1 ∨ t.angles 1 = t.angles 2 ∨ t.angles 0 = t.angles 2

-- Define the three statements
def statement1 (t : Triangle) : Prop :=
  (∃ i j : Fin 3, i ≠ j ∧ t.angles i = t.angles j) → isIsosceles t

def statement2 (t : Triangle) : Prop :=
  ¬isIsosceles t → (∀ i j : Fin 3, i ≠ j → t.angles i ≠ t.angles j)

def statement3 (t : Triangle) : Prop :=
  (∃ i j : Fin 3, i ≠ j ∧ t.angles i = t.angles j) → isIsosceles t

-- Theorem: The three statements are logically equivalent
theorem statements_equivalent : ∀ t : Triangle,
  (statement1 t ↔ statement2 t) ∧ (statement2 t ↔ statement3 t) :=
sorry

end NUMINAMATH_CALUDE_statements_equivalent_l1358_135834


namespace NUMINAMATH_CALUDE_c_5_value_l1358_135841

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

theorem c_5_value (c : ℕ → ℝ) :
  geometric_sequence (λ n => Real.sqrt (c n)) ∧
  Real.sqrt (c 1) = 1 ∧
  Real.sqrt (c 2) = 2 * Real.sqrt (c 1) →
  c 5 = 256 := by
sorry

end NUMINAMATH_CALUDE_c_5_value_l1358_135841


namespace NUMINAMATH_CALUDE_factor_63x_minus_21_l1358_135881

theorem factor_63x_minus_21 : ∀ x : ℝ, 63 * x - 21 = 21 * (3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_63x_minus_21_l1358_135881


namespace NUMINAMATH_CALUDE_area_difference_zero_l1358_135860

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  contains_origin : (center.1)^2 + (center.2)^2 < radius^2
  radius : ℝ

-- Define the areas S+ and S-
def S_plus (c : Circle) : ℝ := sorry
def S_minus (c : Circle) : ℝ := sorry

-- Theorem statement
theorem area_difference_zero (c : Circle) : S_plus c - S_minus c = 0 := by sorry

end NUMINAMATH_CALUDE_area_difference_zero_l1358_135860


namespace NUMINAMATH_CALUDE_expression_simplification_l1358_135880

theorem expression_simplification (x y : ℝ) :
  8 * x + 3 * y - 2 * x + y + 20 + 15 = 6 * x + 4 * y + 35 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1358_135880


namespace NUMINAMATH_CALUDE_pioneer_assignment_l1358_135838

structure Pioneer where
  lastName : String
  firstName : String
  age : Nat

def Burov : Pioneer := sorry
def Gridnev : Pioneer := sorry
def Klimenko : Pioneer := sorry

axiom burov_not_kolya : Burov.firstName ≠ "Kolya"
axiom petya_school_start : ∃ p : Pioneer, p.firstName = "Petya" ∧ p.age = 12
axiom gridnev_grisha_older : Gridnev.age = (Klimenko.age + 1) ∧ Burov.age = (Klimenko.age + 1)

theorem pioneer_assignment :
  (Burov.firstName = "Grisha" ∧ Burov.age = 13) ∧
  (Gridnev.firstName = "Kolya" ∧ Gridnev.age = 13) ∧
  (Klimenko.firstName = "Petya" ∧ Klimenko.age = 12) :=
sorry

end NUMINAMATH_CALUDE_pioneer_assignment_l1358_135838


namespace NUMINAMATH_CALUDE_hexagon_angle_sequences_l1358_135817

/-- Represents a sequence of 6 integers for hexagon interior angles -/
def HexagonAngles := (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ)

/-- Checks if a sequence of angles is valid according to the problem conditions -/
def is_valid_sequence (angles : HexagonAngles) : Prop :=
  let (a₁, a₂, a₃, a₄, a₅, a₆) := angles
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 720) ∧ 
  (30 ≤ a₁) ∧
  (∀ i, i ∈ [a₁, a₂, a₃, a₄, a₅, a₆] → i < 160) ∧
  (a₁ < a₂) ∧ (a₂ < a₃) ∧ (a₃ < a₄) ∧ (a₄ < a₅) ∧ (a₅ < a₆) ∧
  (∃ d : ℕ, d > 0 ∧ a₂ = a₁ + d ∧ a₃ = a₂ + d ∧ a₄ = a₃ + d ∧ a₅ = a₄ + d ∧ a₆ = a₅ + d)

/-- The main theorem stating that there are exactly 4 valid sequences -/
theorem hexagon_angle_sequences :
  ∃! (sequences : Finset HexagonAngles),
    sequences.card = 4 ∧
    (∀ seq ∈ sequences, is_valid_sequence seq) ∧
    (∀ seq, is_valid_sequence seq → seq ∈ sequences) :=
sorry

end NUMINAMATH_CALUDE_hexagon_angle_sequences_l1358_135817


namespace NUMINAMATH_CALUDE_square_difference_l1358_135820

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) :
  (x - y)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1358_135820


namespace NUMINAMATH_CALUDE_amoeba_growth_l1358_135809

/-- The population of amoebas after a given number of 10-minute increments -/
def amoeba_population (initial_population : ℕ) (increments : ℕ) : ℕ :=
  initial_population * (3 ^ increments)

/-- Theorem: The amoeba population after 1 hour (6 increments) is 36450 -/
theorem amoeba_growth : amoeba_population 50 6 = 36450 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_growth_l1358_135809


namespace NUMINAMATH_CALUDE_simplify_product_l1358_135805

theorem simplify_product (b : R) [CommRing R] :
  (2 : R) * b * (3 : R) * b^2 * (4 : R) * b^3 * (5 : R) * b^4 * (6 : R) * b^5 = (720 : R) * b^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_l1358_135805


namespace NUMINAMATH_CALUDE_tan_22_5_deg_sum_l1358_135814

theorem tan_22_5_deg_sum (a b c d : ℕ+) 
  (h1 : Real.tan (22.5 * π / 180) = (a : ℝ).sqrt - b + (c : ℝ).sqrt - (d : ℝ).sqrt)
  (h2 : a ≥ b) (h3 : b ≥ c) (h4 : c ≥ d) : 
  a + b + c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_sum_l1358_135814


namespace NUMINAMATH_CALUDE_faye_pencil_rows_l1358_135898

def total_pencils : ℕ := 35
def pencils_per_row : ℕ := 5

theorem faye_pencil_rows :
  total_pencils / pencils_per_row = 7 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencil_rows_l1358_135898


namespace NUMINAMATH_CALUDE_cupcake_icing_time_l1358_135825

theorem cupcake_icing_time (total_batches : ℕ) (baking_time_per_batch : ℕ) (total_time : ℕ) :
  total_batches = 4 →
  baking_time_per_batch = 20 →
  total_time = 200 →
  (total_time - total_batches * baking_time_per_batch) / total_batches = 30 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_icing_time_l1358_135825


namespace NUMINAMATH_CALUDE_division_equation_proof_l1358_135822

theorem division_equation_proof : (320 : ℝ) / (54 + 26) = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_equation_proof_l1358_135822


namespace NUMINAMATH_CALUDE_number_difference_l1358_135864

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 41402)
  (b_div_100 : 100 ∣ b)
  (a_eq_b_div_100 : a = b / 100) :
  b - a = 40590 :=
sorry

end NUMINAMATH_CALUDE_number_difference_l1358_135864


namespace NUMINAMATH_CALUDE_unique_integer_root_l1358_135867

def polynomial (x : ℤ) : ℤ := x^3 - 4*x^2 - 8*x + 24

theorem unique_integer_root : 
  (∀ x : ℤ, polynomial x = 0 ↔ x = 2) := by sorry

end NUMINAMATH_CALUDE_unique_integer_root_l1358_135867


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1358_135823

-- Define the quadratic function
def f (x : ℝ) := x^2 + 4*x - 5

-- Define the solution set
def solution_set : Set ℝ := {x | x < -5 ∨ x > 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1358_135823


namespace NUMINAMATH_CALUDE_ellipse_parabola_focus_l1358_135810

theorem ellipse_parabola_focus (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (∀ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1) →
  (∃ k : ℝ, ∀ x y : ℝ, x^2 = 8*y → y = k) →
  (n^2 - m^2 = 4) →
  (Real.sqrt (n^2 - m^2) / n = 1/2) →
  m - n = 2 * Real.sqrt 3 - 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_focus_l1358_135810


namespace NUMINAMATH_CALUDE_dividend_calculation_l1358_135845

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 19)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 176 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1358_135845


namespace NUMINAMATH_CALUDE_line_slope_l1358_135818

/-- Given a line with equation y = -5x + 9, its slope is -5 -/
theorem line_slope (x y : ℝ) : y = -5 * x + 9 → (∃ m b : ℝ, y = m * x + b ∧ m = -5) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l1358_135818


namespace NUMINAMATH_CALUDE_incorrect_conclusion_l1358_135827

theorem incorrect_conclusion (a b : ℝ) (h1 : a > b) (h2 : b > a + b) : 
  ¬(a * b > (a + b)^2) := by
sorry

end NUMINAMATH_CALUDE_incorrect_conclusion_l1358_135827


namespace NUMINAMATH_CALUDE_fraction_of_apples_sold_l1358_135808

/-- Proves that the fraction of apples sold is 1/2 given the initial quantities and conditions --/
theorem fraction_of_apples_sold
  (initial_oranges : ℕ)
  (initial_apples : ℕ)
  (orange_fraction_sold : ℚ)
  (total_fruits_left : ℕ)
  (h1 : initial_oranges = 40)
  (h2 : initial_apples = 70)
  (h3 : orange_fraction_sold = 1/4)
  (h4 : total_fruits_left = 65)
  : (initial_apples - (total_fruits_left - (initial_oranges - initial_oranges * orange_fraction_sold))) / initial_apples = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_apples_sold_l1358_135808


namespace NUMINAMATH_CALUDE_product_sum_inequality_l1358_135894

theorem product_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l1358_135894


namespace NUMINAMATH_CALUDE_basketball_score_proof_l1358_135876

theorem basketball_score_proof (two_point_shots three_point_shots free_throws : ℕ) : 
  (3 * three_point_shots = 2 * two_point_shots) →
  (free_throws = 2 * two_point_shots) →
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 72) →
  free_throws = 24 := by
  sorry

#check basketball_score_proof

end NUMINAMATH_CALUDE_basketball_score_proof_l1358_135876


namespace NUMINAMATH_CALUDE_two_box_marble_problem_l1358_135816

/-- Represents a box containing marbles -/
structure MarbleBox where
  total : ℕ
  black : ℕ
  white : ℕ
  h_sum : total = black + white

/-- The probability of drawing a specific color marble from a box -/
def drawProbability (box : MarbleBox) (color : ℕ) : ℚ :=
  color / box.total

theorem two_box_marble_problem (box1 box2 : MarbleBox) : 
  box1.total + box2.total = 25 →
  drawProbability box1 box1.black * drawProbability box2 box2.black = 27/50 →
  drawProbability box1 box1.white * drawProbability box2 box2.white = 1/25 := by
sorry

end NUMINAMATH_CALUDE_two_box_marble_problem_l1358_135816


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_l1358_135830

theorem fraction_inequality_solution (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 4) ≥ 1 ↔ 
  x < -4 ∨ (-2 < x ∧ x < -Real.sqrt 8) ∨ x > Real.sqrt 8 :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_l1358_135830


namespace NUMINAMATH_CALUDE_largest_tile_size_is_correct_courtyard_largest_tile_size_l1358_135886

/-- The largest square tile size that can exactly pave a rectangular courtyard -/
def largest_tile_size (length width : ℕ) : ℕ :=
  Nat.gcd length width

theorem largest_tile_size_is_correct (length width : ℕ) (h1 : length > 0) (h2 : width > 0) :
  let tile_size := largest_tile_size length width
  ∀ n : ℕ, n > tile_size → ¬(n ∣ length ∧ n ∣ width) :=
by sorry

theorem courtyard_largest_tile_size :
  largest_tile_size 378 595 = 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_tile_size_is_correct_courtyard_largest_tile_size_l1358_135886


namespace NUMINAMATH_CALUDE_max_value_F_l1358_135855

theorem max_value_F (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → 
    |(a * x^2 + b * x + c) * (c * x^2 + b * x + a)| ≤ M ∧
    ∃ y : ℝ, y ∈ Set.Icc (-1) 1 ∧ 
      |(a * y^2 + b * y + c) * (c * y^2 + b * y + a)| = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_F_l1358_135855


namespace NUMINAMATH_CALUDE_curve_scaling_transformation_l1358_135803

/-- Given a curve C with equation x² + y² = 1 and a scaling transformation,
    prove that the resulting curve C'' has the equation x² + y²/4 = 1 -/
theorem curve_scaling_transformation (x y x'' y'' : ℝ) :
  (x^2 + y^2 = 1) →    -- Equation of curve C
  (x'' = x) →          -- x-coordinate transformation
  (y'' = 2*y) →        -- y-coordinate transformation
  (x''^2 + y''^2/4 = 1) -- Equation of curve C''
:= by sorry

end NUMINAMATH_CALUDE_curve_scaling_transformation_l1358_135803


namespace NUMINAMATH_CALUDE_max_profit_at_25_yuan_manager_decision_suboptimal_l1358_135885

/-- Profit function based on price reduction -/
def profit (x : ℝ) : ℝ := (2 * x - 20) * (40 - x)

/-- Initial daily sales -/
def initial_sales : ℝ := 20

/-- Initial profit per piece -/
def initial_profit_per_piece : ℝ := 40

/-- Sales increase per yuan of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Theorem stating the maximum profit and corresponding price reduction -/
theorem max_profit_at_25_yuan :
  ∃ (max_reduction : ℝ) (max_profit : ℝ),
    max_reduction = 25 ∧
    max_profit = 1250 ∧
    ∀ x, 0 ≤ x ∧ x ≤ 40 → profit x ≤ max_profit :=
sorry

/-- Theorem comparing manager's decision to optimal decision -/
theorem manager_decision_suboptimal (manager_reduction : ℝ) (h : manager_reduction = 15) :
  ∃ (optimal_reduction : ℝ) (optimal_profit : ℝ),
    optimal_reduction ≠ manager_reduction ∧
    optimal_profit > profit manager_reduction :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_25_yuan_manager_decision_suboptimal_l1358_135885


namespace NUMINAMATH_CALUDE_david_widget_production_difference_l1358_135890

/-- Given David's widget production rates and hours worked, prove the difference between Monday and Tuesday production. -/
theorem david_widget_production_difference 
  (t : ℕ) -- Number of hours worked on Monday
  (w : ℕ) -- Widgets produced per hour on Monday
  (h1 : w = 2 * t) -- Relation between w and t
  : w * t - (w + 5) * (t - 3) = t + 15 := by
  sorry

end NUMINAMATH_CALUDE_david_widget_production_difference_l1358_135890


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1358_135835

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem seventh_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h1 : a 1 + a 2 = 3) 
  (h2 : a 2 + a 3 = 6) : 
  a 7 = 64 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1358_135835


namespace NUMINAMATH_CALUDE_imaginary_part_of_4_plus_3i_l1358_135850

theorem imaginary_part_of_4_plus_3i :
  Complex.im (4 + 3*Complex.I) = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_4_plus_3i_l1358_135850


namespace NUMINAMATH_CALUDE_same_distinct_prime_factors_l1358_135849

-- Define the set of distinct prime factors
def distinct_prime_factors (n : ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ p ∣ n}

-- State the theorem
theorem same_distinct_prime_factors (k : ℕ) (h : k > 1) :
  let A := 2^k - 2
  let B := 2^k * A
  (distinct_prime_factors A = distinct_prime_factors B) ∧
  (distinct_prime_factors (A + 1) = distinct_prime_factors (B + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_same_distinct_prime_factors_l1358_135849


namespace NUMINAMATH_CALUDE_fib_product_divisibility_l1358_135871

/-- Mersenne sequence property: for any two positive integers i and j, gcd(aᵢ, aⱼ) = a_{gcd(i,j)} -/
def is_mersenne_sequence (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i > 0 → j > 0 → Nat.gcd (a i) (a j) = a (Nat.gcd i j)

/-- Fibonacci sequence definition -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n+2 => fib (n+1) + fib n

/-- Product of first n terms of a sequence -/
def seq_product (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * a (i+1)) 1

/-- Main theorem: For Fibonacci sequence, product of k consecutive terms 
    is divisible by product of first k terms -/
theorem fib_product_divisibility (k : ℕ) (n : ℕ) : 
  k > 0 → is_mersenne_sequence fib → 
  (seq_product fib k) ∣ (List.range k).foldl (λ acc i => acc * fib (n+i)) 1 :=
sorry

end NUMINAMATH_CALUDE_fib_product_divisibility_l1358_135871


namespace NUMINAMATH_CALUDE_line_minimum_reciprocal_sum_l1358_135815

theorem line_minimum_reciprocal_sum (m n : ℝ) (h1 : m * n > 0) (h2 : m + n = 2) :
  ∀ (x y : ℝ), x * n + y * m = 2 → x = 1 ∧ y = 1 →
  (1 / m + 1 / n) ≥ 2 ∧ ∃ (m₀ n₀ : ℝ), m₀ * n₀ > 0 ∧ m₀ + n₀ = 2 ∧ 1 / m₀ + 1 / n₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_minimum_reciprocal_sum_l1358_135815


namespace NUMINAMATH_CALUDE_smallest_m_for_distinct_roots_l1358_135879

theorem smallest_m_for_distinct_roots (m : ℤ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 4*x - m = 0 ∧ y^2 + 4*y - m = 0) → 
  (∀ k : ℤ, k < m → ¬∃ x y : ℝ, x ≠ y ∧ x^2 + 4*x - k = 0 ∧ y^2 + 4*y - k = 0) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_distinct_roots_l1358_135879


namespace NUMINAMATH_CALUDE_computer_price_difference_l1358_135813

/-- Proof of the computer price difference problem -/
theorem computer_price_difference 
  (total_price : ℝ)
  (basic_price : ℝ)
  (printer_price : ℝ)
  (enhanced_price : ℝ)
  (h1 : total_price = 2500)
  (h2 : basic_price = 2000)
  (h3 : total_price = basic_price + printer_price)
  (h4 : printer_price = (1/6) * (enhanced_price + printer_price)) :
  enhanced_price - basic_price = 500 := by
  sorry

#check computer_price_difference

end NUMINAMATH_CALUDE_computer_price_difference_l1358_135813


namespace NUMINAMATH_CALUDE_polynomial_roots_l1358_135829

theorem polynomial_roots : 
  let p (x : ℝ) := x^4 - 2*x^3 - 7*x^2 + 14*x - 6
  ∃ (a b c d : ℝ), 
    (a = 1 ∧ b = 2 ∧ c = (-1 + Real.sqrt 13) / 2 ∧ d = (-1 - Real.sqrt 13) / 2) ∧
    (∀ x : ℝ, p x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1358_135829


namespace NUMINAMATH_CALUDE_china_gdp_scientific_notation_l1358_135852

theorem china_gdp_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ |a| ∧ |a| < 10 ∧ 
    n = 5 ∧
    827000 = a * (10 : ℝ)^n ∧
    a = 8.27 := by
  sorry

end NUMINAMATH_CALUDE_china_gdp_scientific_notation_l1358_135852


namespace NUMINAMATH_CALUDE_pythagorean_theorem_3_4_5_l1358_135883

theorem pythagorean_theorem_3_4_5 : 
  ∀ (a b c : ℝ), 
    a = 3 → b = 4 → c^2 = a^2 + b^2 → c = 5 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_3_4_5_l1358_135883


namespace NUMINAMATH_CALUDE_chessboard_diagonal_ratio_l1358_135869

/-- Represents a rectangle with chessboard coloring -/
structure ChessboardRectangle where
  a : ℕ  -- length
  b : ℕ  -- width

/-- Calculates the ratio of white to black segment lengths on the diagonal -/
def diagonalSegmentRatio (rect : ChessboardRectangle) : ℚ :=
  if rect.a = 100 ∧ rect.b = 99 then 1
  else if rect.a = 101 ∧ rect.b = 99 then 5000 / 4999
  else 0  -- undefined for other dimensions

theorem chessboard_diagonal_ratio :
  ∀ (rect : ChessboardRectangle),
    (rect.a = 100 ∧ rect.b = 99 → diagonalSegmentRatio rect = 1) ∧
    (rect.a = 101 ∧ rect.b = 99 → diagonalSegmentRatio rect = 5000 / 4999) :=
by sorry

end NUMINAMATH_CALUDE_chessboard_diagonal_ratio_l1358_135869
