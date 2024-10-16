import Mathlib

namespace NUMINAMATH_CALUDE_triangle_placement_theorem_l501_50119

-- Define the types for points and angles
def Point : Type := ℝ × ℝ
def Angle : Type := ℝ

-- Define a triangle as a triple of points
structure Triangle :=
  (E F G : Point)

-- Define the property that a point lies on an arm of an angle
def lies_on_arm (P : Point) (A : Point) (angle : Angle) : Prop := sorry

-- Define the property that an angle between three points equals a given angle
def angle_equals (A B C : Point) (angle : Angle) : Prop := sorry

theorem triangle_placement_theorem 
  (T : Triangle) (angle_ABC angle_CBD : Angle) : 
  ∃ (B : Point), 
    (lies_on_arm T.E B angle_ABC) ∧ 
    (lies_on_arm T.F B angle_ABC) ∧ 
    (lies_on_arm T.G B angle_CBD) ∧
    (angle_equals T.E B T.F angle_ABC) ∧
    (angle_equals T.F B T.G angle_CBD) := by
  sorry

end NUMINAMATH_CALUDE_triangle_placement_theorem_l501_50119


namespace NUMINAMATH_CALUDE_prob_two_even_out_of_six_l501_50160

/-- The probability of rolling an even number on a fair six-sided die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling an odd number on a fair six-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of dice showing even numbers -/
def num_even : ℕ := 2

/-- The number of ways to choose 2 dice out of 6 -/
def ways_to_choose : ℕ := Nat.choose num_dice num_even

theorem prob_two_even_out_of_six :
  (ways_to_choose : ℚ) * prob_even^num_even * prob_odd^(num_dice - num_even) = 15/64 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_even_out_of_six_l501_50160


namespace NUMINAMATH_CALUDE_bird_ratio_l501_50116

/-- Proves that the ratio of cardinals to bluebirds is 3:1 given the conditions of the bird problem -/
theorem bird_ratio (cardinals bluebirds swallows : ℕ) 
  (swallow_half : swallows = bluebirds / 2)
  (swallow_count : swallows = 2)
  (total_birds : cardinals + bluebirds + swallows = 18) :
  cardinals = 3 * bluebirds := by
  sorry


end NUMINAMATH_CALUDE_bird_ratio_l501_50116


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l501_50109

/-- 
Given a rectangular plot where:
- The area is 21 times its breadth
- The difference between the length and breadth is 10 metres
This theorem proves that the breadth of the plot is 11 metres.
-/
theorem rectangular_plot_breadth (length width : ℝ) 
  (h1 : length * width = 21 * width) 
  (h2 : length - width = 10) : 
  width = 11 := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l501_50109


namespace NUMINAMATH_CALUDE_bacteria_count_scientific_notation_l501_50166

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Checks if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (n : ℝ) : Prop :=
  n = sn.coefficient * (10 : ℝ) ^ sn.exponent

/-- The number of bacteria in a fly's stomach -/
def bacteria_count : ℕ := 28000000

/-- The scientific notation representation of the bacteria count -/
def bacteria_scientific : ScientificNotation where
  coefficient := 2.8
  exponent := 7
  h1 := by sorry

/-- Theorem stating that the scientific notation correctly represents the bacteria count -/
theorem bacteria_count_scientific_notation :
    represents bacteria_scientific (bacteria_count : ℝ) := by sorry

end NUMINAMATH_CALUDE_bacteria_count_scientific_notation_l501_50166


namespace NUMINAMATH_CALUDE_wheels_in_garage_l501_50152

theorem wheels_in_garage : 
  let bicycles : ℕ := 9
  let cars : ℕ := 16
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_car : ℕ := 4
  bicycles * wheels_per_bicycle + cars * wheels_per_car = 82 :=
by sorry

end NUMINAMATH_CALUDE_wheels_in_garage_l501_50152


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l501_50198

/-- Linear function f(x) = 8x - 1 -/
def f (x : ℝ) : ℝ := 8 * x - 1

/-- Point P₁ lies on the graph of f -/
def P₁_on_f (y₁ : ℝ) : Prop := f 3 = y₁

/-- Point P₂ lies on the graph of f -/
def P₂_on_f (y₂ : ℝ) : Prop := f 4 = y₂

theorem y₁_less_than_y₂ (y₁ y₂ : ℝ) (h₁ : P₁_on_f y₁) (h₂ : P₂_on_f y₂) : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l501_50198


namespace NUMINAMATH_CALUDE_infinite_series_sum_l501_50115

theorem infinite_series_sum : 
  let series := fun n : ℕ => (n : ℝ) / 8^n
  ∑' n, series n = 8 / 49 := by
sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l501_50115


namespace NUMINAMATH_CALUDE_largest_b_in_box_l501_50186

theorem largest_b_in_box (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) → (c < b) → (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 12 :=
sorry

end NUMINAMATH_CALUDE_largest_b_in_box_l501_50186


namespace NUMINAMATH_CALUDE_sum_of_digits_product_72_sevens_72_fives_l501_50188

/-- Represents a number consisting of n repetitions of a single digit --/
def repeatedDigit (d : Nat) (n : Nat) : Nat :=
  d * (10^n - 1) / 9

/-- Calculates the sum of digits in a natural number --/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved --/
theorem sum_of_digits_product_72_sevens_72_fives :
  sumOfDigits (repeatedDigit 7 72 * repeatedDigit 5 72) = 576 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_product_72_sevens_72_fives_l501_50188


namespace NUMINAMATH_CALUDE_bubble_gum_cost_l501_50171

theorem bubble_gum_cost (total_pieces : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) : 
  total_pieces = 136 → total_cost = 2448 → cost_per_piece = 18 → 
  total_cost = total_pieces * cost_per_piece :=
by sorry

end NUMINAMATH_CALUDE_bubble_gum_cost_l501_50171


namespace NUMINAMATH_CALUDE_tenth_row_white_squares_l501_50179

/-- Represents the number of squares in the nth row of a stair-step figure -/
def totalSquares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in the nth row of a stair-step figure -/
def whiteSquares (n : ℕ) : ℕ := (totalSquares n) / 2

theorem tenth_row_white_squares :
  whiteSquares 10 = 9 := by sorry

end NUMINAMATH_CALUDE_tenth_row_white_squares_l501_50179


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l501_50153

theorem polygon_angle_sum (n : ℕ) (h : n = 5) :
  (n - 2) * 180 + 360 = 900 :=
sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l501_50153


namespace NUMINAMATH_CALUDE_parabola_symmetric_points_l501_50132

/-- The parabola defined by y^2 = 8x -/
def Parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The line y = (1/4)x - 2 -/
def SymmetryLine (x y : ℝ) : Prop := y = (1/4)*x - 2

/-- Two points are symmetric about a line if their midpoint lies on that line -/
def SymmetricPoints (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  SymmetryLine ((x₁ + x₂)/2) ((y₁ + y₂)/2)

/-- The equation of line AB: 4x + y - 15 = 0 -/
def LineAB (x y : ℝ) : Prop := 4*x + y - 15 = 0

theorem parabola_symmetric_points (x₁ y₁ x₂ y₂ : ℝ) :
  Parabola x₁ y₁ → Parabola x₂ y₂ → SymmetricPoints x₁ y₁ x₂ y₂ →
  ∀ x y, LineAB x y ↔ (y - y₁)/(x - x₁) = (y₂ - y)/(x₂ - x) :=
sorry

end NUMINAMATH_CALUDE_parabola_symmetric_points_l501_50132


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l501_50143

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_30 : a 30 = 100)
  (h_100 : a 100 = 30) :
  ∃ d : ℝ, d = -1 ∧ ∀ n, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l501_50143


namespace NUMINAMATH_CALUDE_max_distance_to_origin_is_three_l501_50144

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- Calculates the maximum distance from any point on a circle to the origin in polar coordinates -/
def maxDistanceToOrigin (c : PolarCircle) : ℝ :=
  c.center.r + c.radius

theorem max_distance_to_origin_is_three :
  let circle := PolarCircle.mk (PolarPoint.mk 2 (π / 6)) 1
  maxDistanceToOrigin circle = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_to_origin_is_three_l501_50144


namespace NUMINAMATH_CALUDE_joan_gained_two_balloons_l501_50161

/-- The number of blue balloons Joan gained -/
def balloons_gained (initial final : ℕ) : ℕ := final - initial

/-- Proof that Joan gained 2 blue balloons -/
theorem joan_gained_two_balloons :
  let initial : ℕ := 9
  let final : ℕ := 11
  balloons_gained initial final = 2 := by sorry

end NUMINAMATH_CALUDE_joan_gained_two_balloons_l501_50161


namespace NUMINAMATH_CALUDE_expression_evaluation_l501_50193

theorem expression_evaluation :
  let x : ℤ := -1
  let y : ℤ := 2
  let expr := -2 * (-x^2*y + x*y^2) - (-3*x^2*y^2 + 3*x^2*y + (3*x^2*y^2 - 3*x*y^2))
  expr = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l501_50193


namespace NUMINAMATH_CALUDE_tank_emptying_l501_50114

theorem tank_emptying (tank_capacity : ℝ) : 
  (3/4 * tank_capacity - 1/3 * tank_capacity = 15) → 
  (1/3 * tank_capacity = 12) :=
by sorry

end NUMINAMATH_CALUDE_tank_emptying_l501_50114


namespace NUMINAMATH_CALUDE_smallest_square_containing_rectangles_l501_50158

/-- The smallest square containing two non-overlapping rectangles -/
theorem smallest_square_containing_rectangles :
  ∀ (w₁ h₁ w₂ h₂ : ℕ),
  w₁ = 3 ∧ h₁ = 5 ∧ w₂ = 4 ∧ h₂ = 6 →
  ∃ (s : ℕ),
    s ≥ w₁ ∧ s ≥ h₁ ∧ s ≥ w₂ ∧ s ≥ h₂ ∧
    s ≥ w₁ + w₂ ∧ s ≥ h₁ ∧ s ≥ h₂ ∧
    (∀ (t : ℕ),
      t ≥ w₁ ∧ t ≥ h₁ ∧ t ≥ w₂ ∧ t ≥ h₂ ∧
      t ≥ w₁ + w₂ ∧ t ≥ h₁ ∧ t ≥ h₂ →
      t ≥ s) ∧
    s^2 = 49 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_containing_rectangles_l501_50158


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l501_50100

def seven_digit_number (n : ℕ) : ℕ := 854 * 10000 + n * 1000 + 526

theorem divisibility_by_eleven (n : ℕ) : 
  (seven_digit_number n) % 11 = 0 ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l501_50100


namespace NUMINAMATH_CALUDE_triangle_area_implies_k_difference_l501_50117

-- Define the lines
def line1 (k₁ b x : ℝ) : ℝ := k₁ * x + 3 * k₁ + b
def line2 (k₂ b x : ℝ) : ℝ := k₂ * x + 3 * k₂ + b

-- Define the theorem
theorem triangle_area_implies_k_difference
  (k₁ k₂ b : ℝ)
  (h1 : k₁ * k₂ < 0)
  (h2 : (1/2) * 3 * |3 * k₁ - 3 * k₂| * 3 = 9) :
  |k₁ - k₂| = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_implies_k_difference_l501_50117


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l501_50169

open Real

theorem triangle_side_sum_range (A B C a b c : Real) : 
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  cos B / b + cos C / c = 2 * sqrt 3 * sin A / (3 * sin C) →
  cos B + sqrt 3 * sin B = 2 →
  3 / 2 < a + c ∧ a + c ≤ sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l501_50169


namespace NUMINAMATH_CALUDE_sin_15_sin_75_equals_half_l501_50170

theorem sin_15_sin_75_equals_half : 2 * Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_sin_75_equals_half_l501_50170


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l501_50105

theorem convex_polygon_sides (n : ℕ) (sum_except_one : ℝ) : 
  sum_except_one = 2190 → 
  (∃ (missing_angle : ℝ), 
    missing_angle > 0 ∧ 
    missing_angle < 180 ∧ 
    sum_except_one + missing_angle = 180 * (n - 2)) → 
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l501_50105


namespace NUMINAMATH_CALUDE_number_of_workers_l501_50194

theorem number_of_workers (total_contribution : ℕ) (extra_contribution : ℕ) (new_total : ℕ) : 
  total_contribution = 300000 →
  extra_contribution = 50 →
  new_total = 325000 →
  (∃ (num_workers : ℕ) (individual_contribution : ℕ), 
    num_workers * individual_contribution = total_contribution ∧
    num_workers * (individual_contribution + extra_contribution) = new_total ∧
    num_workers = 500) :=
by sorry

end NUMINAMATH_CALUDE_number_of_workers_l501_50194


namespace NUMINAMATH_CALUDE_watermelon_pricing_l501_50151

/-- Represents the number of watermelons each brother brought --/
structure Watermelons :=
  (elder : ℕ)
  (second : ℕ)
  (youngest : ℕ)

/-- Represents the number of watermelons sold in the morning --/
structure MorningSales :=
  (elder : ℕ)
  (second : ℕ)
  (youngest : ℕ)

/-- Theorem: Given the conditions, prove that the morning price was 3.75 yuan and the afternoon price was 1.25 yuan --/
theorem watermelon_pricing
  (w : Watermelons)
  (m : MorningSales)
  (h1 : w.elder = 10)
  (h2 : w.second = 16)
  (h3 : w.youngest = 26)
  (h4 : m.elder ≤ w.elder)
  (h5 : m.second ≤ w.second)
  (h6 : m.youngest ≤ w.youngest)
  (h7 : ∃ (morning_price afternoon_price : ℚ),
    morning_price > afternoon_price ∧
    afternoon_price > 0 ∧
    morning_price * m.elder + afternoon_price * (w.elder - m.elder) = 35 ∧
    morning_price * m.second + afternoon_price * (w.second - m.second) = 35 ∧
    morning_price * m.youngest + afternoon_price * (w.youngest - m.youngest) = 35) :
  ∃ (morning_price afternoon_price : ℚ),
    morning_price = 3.75 ∧ afternoon_price = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_pricing_l501_50151


namespace NUMINAMATH_CALUDE_packetB_height_day10_l501_50196

/-- Represents the growth rate of sunflowers --/
structure GrowthRate where
  x : ℝ  -- number of days since planting
  y : ℝ  -- daily average sunlight exposure (hours)
  W : ℝ  -- combined effect of competition and weather (0-10 scale)

/-- Calculates the growth rate for Packet A sunflowers --/
def growthRateA (r : GrowthRate) : ℝ := 2 * r.x + r.y - 0.1 * r.W

/-- Calculates the growth rate for Packet B sunflowers --/
def growthRateB (r : GrowthRate) : ℝ := 3 * r.x - r.y + 0.2 * r.W

/-- Theorem stating the height of Packet B sunflowers on day 10 --/
theorem packetB_height_day10 (r : GrowthRate) 
  (h1 : r.x = 10)
  (h2 : r.y = 6)
  (h3 : r.W = 5)
  (h4 : ∃ (hA hB : ℝ), hA = 192 ∧ hA = 1.2 * hB) :
  ∃ (hB : ℝ), hB = 160 := by
  sorry


end NUMINAMATH_CALUDE_packetB_height_day10_l501_50196


namespace NUMINAMATH_CALUDE_order_of_6_wrt_f_l501_50190

def f (x : ℕ) : ℕ := x^2 % 13

def iterateF (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterateF n x)

theorem order_of_6_wrt_f :
  ∀ k : ℕ, k > 0 → k < 36 → iterateF k 6 ≠ 6 ∧ iterateF 36 6 = 6 := by sorry

end NUMINAMATH_CALUDE_order_of_6_wrt_f_l501_50190


namespace NUMINAMATH_CALUDE_light_distance_scientific_notation_l501_50127

/-- The speed of light in kilometers per second -/
def speed_of_light : ℝ := 300000

/-- The time in seconds -/
def time : ℝ := 10

/-- The distance traveled by light in the given time -/
def distance : ℝ := speed_of_light * time

/-- The exponent in the scientific notation of the distance -/
def n : ℕ := 6

theorem light_distance_scientific_notation :
  ∃ (a : ℝ), a > 0 ∧ a < 10 ∧ distance = a * (10 : ℝ) ^ n :=
sorry

end NUMINAMATH_CALUDE_light_distance_scientific_notation_l501_50127


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_AOB_l501_50148

/-- The minimum perimeter of triangle AOB given the conditions -/
theorem min_perimeter_triangle_AOB :
  let P : ℝ × ℝ := (4, 2)
  let O : ℝ × ℝ := (0, 0)
  ∃ (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)),
    (P ∈ l) ∧
    (A.1 > 0 ∧ A.2 = 0) ∧
    (B.1 = 0 ∧ B.2 > 0) ∧
    (A ∈ l) ∧ (B ∈ l) ∧
    (∀ (A' B' : ℝ × ℝ) (l' : Set (ℝ × ℝ)),
      (P ∈ l') ∧
      (A'.1 > 0 ∧ A'.2 = 0) ∧
      (B'.1 = 0 ∧ B'.2 > 0) ∧
      (A' ∈ l') ∧ (B' ∈ l') →
      dist O A + dist O B + dist A B ≤ dist O A' + dist O B' + dist A' B') ∧
    (dist O A + dist O B + dist A B = 20) :=
by sorry


end NUMINAMATH_CALUDE_min_perimeter_triangle_AOB_l501_50148


namespace NUMINAMATH_CALUDE_pet_food_price_l501_50103

theorem pet_food_price (regular_discount_min : ℝ) (regular_discount_max : ℝ) 
  (additional_discount : ℝ) (lowest_price : ℝ) :
  regular_discount_min = 0.1 →
  regular_discount_max = 0.3 →
  additional_discount = 0.2 →
  lowest_price = 16.8 →
  ∃ (original_price : ℝ), 
    original_price * (1 - regular_discount_max) * (1 - additional_discount) = lowest_price ∧
    original_price = 30 :=
by sorry

end NUMINAMATH_CALUDE_pet_food_price_l501_50103


namespace NUMINAMATH_CALUDE_minimum_value_problem_l501_50180

theorem minimum_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  y / x + 4 / y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ y₀ / x₀ + 4 / y₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l501_50180


namespace NUMINAMATH_CALUDE_expression_simplification_l501_50129

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) :
  (x^2 - y^2) / (x * y) - (x * y - 2 * y^2) / (x * y - x^2) = (x^2 - 2 * y^2) / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l501_50129


namespace NUMINAMATH_CALUDE_product_of_roots_l501_50106

theorem product_of_roots (a b c : ℂ) : 
  (3 * a^3 - 4 * a^2 - 12 * a + 9 = 0) →
  (3 * b^3 - 4 * b^2 - 12 * b + 9 = 0) →
  (3 * c^3 - 4 * c^2 - 12 * c + 9 = 0) →
  a * b * c = -3 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l501_50106


namespace NUMINAMATH_CALUDE_snail_climb_theorem_l501_50189

/-- The number of days it takes for a snail to climb out of a well -/
def snail_climb_days (well_depth : ℝ) (day_climb : ℝ) (night_slide : ℝ) : ℕ :=
  sorry

/-- Theorem: A snail starting 1 meter below the top of a well, 
    climbing 30 cm during the day and sliding down 20 cm each night, 
    will take 8 days to reach the top of the well -/
theorem snail_climb_theorem : 
  snail_climb_days 1 0.3 0.2 = 8 := by sorry

end NUMINAMATH_CALUDE_snail_climb_theorem_l501_50189


namespace NUMINAMATH_CALUDE_abs_value_of_z_l501_50172

/-- The absolute value of the complex number z = (2i)/(1+i) - 2i is √2 -/
theorem abs_value_of_z : Complex.abs ((2 * Complex.I) / (1 + Complex.I) - 2 * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_z_l501_50172


namespace NUMINAMATH_CALUDE_polynomial_simplification_l501_50138

theorem polynomial_simplification (x : ℝ) : 
  x^2 * (4*x^3 - 3*x + 1) - 6*(x^3 - 3*x^2 + 4*x - 5) = 
  4*x^5 - 9*x^3 + 19*x^2 - 24*x + 30 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l501_50138


namespace NUMINAMATH_CALUDE_geometric_series_problem_l501_50175

theorem geometric_series_problem (a r : ℝ) 
  (h1 : |r| < 1) 
  (h2 : a / (1 - r) = 7)
  (h3 : a * r / (1 - r^2) = 3) : 
  a + r = 5/2 := by sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l501_50175


namespace NUMINAMATH_CALUDE_count_equal_pairs_l501_50167

/-- Definition of the sequence a_n -/
def a (n : ℕ) : ℤ := n^2 - 22*n + 10

/-- The number of pairs of distinct positive integers (m,n) satisfying a_m = a_n -/
def num_pairs : ℕ := 10

/-- Theorem stating that there are exactly 10 pairs of distinct positive integers (m,n) 
    satisfying a_m = a_n -/
theorem count_equal_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    pairs.card = num_pairs ∧ 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ 
      m ≠ n ∧ m > 0 ∧ n > 0 ∧ a m = a n) :=
sorry

end NUMINAMATH_CALUDE_count_equal_pairs_l501_50167


namespace NUMINAMATH_CALUDE_odd_function_periodicity_l501_50195

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_periodicity (f : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_sym : ∀ x, f x = f (2 - x)) :
  ∀ x, f (x + 4) = f x := by
  sorry

end NUMINAMATH_CALUDE_odd_function_periodicity_l501_50195


namespace NUMINAMATH_CALUDE_chord_length_l501_50163

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def C₃ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 14/5 = 0

-- Define the common chord of C₁ and C₂
def common_chord (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- Theorem statement
theorem chord_length :
  ∃ (a b : ℝ), 
    (∀ x y, C₁ x y ∧ C₂ x y → common_chord x y) ∧
    (∃ x₁ y₁ x₂ y₂, 
      C₃ x₁ y₁ ∧ C₃ x₂ y₂ ∧ 
      common_chord x₁ y₁ ∧ common_chord x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4^2) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l501_50163


namespace NUMINAMATH_CALUDE_quadratic_root_implies_v_value_l501_50199

theorem quadratic_root_implies_v_value :
  ∀ v : ℝ,
  ((-15 - Real.sqrt 469) / 6 : ℝ) ∈ {x : ℝ | 3 * x^2 + 15 * x + v = 0} →
  v = -122/6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_v_value_l501_50199


namespace NUMINAMATH_CALUDE_quadratic_factorization_l501_50131

theorem quadratic_factorization : ∀ x : ℝ, 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l501_50131


namespace NUMINAMATH_CALUDE_product_OA_OC_constant_C_trajectory_l501_50159

-- Define the rhombus ABCD
structure Rhombus :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_rhombus : side_length = 4)
  (OB_length : ℝ)
  (OD_length : ℝ)
  (OB_OD_equal : OB_length = 6 ∧ OD_length = 6)

-- Define the function for |OA| * |OC|
def product_OA_OC (r : Rhombus) : ℝ := sorry

-- Define the function for the coordinates of C
def C_coordinates (r : Rhombus) (A_x A_y : ℝ) : ℝ × ℝ := sorry

-- Theorem 1: |OA| * |OC| is constant
theorem product_OA_OC_constant (r : Rhombus) : 
  product_OA_OC r = 20 := by sorry

-- Theorem 2: Trajectory of C
theorem C_trajectory (r : Rhombus) (A_x A_y : ℝ) 
  (h1 : (A_x - 2)^2 + A_y^2 = 4) (h2 : 2 ≤ A_x ∧ A_x ≤ 4) :
  ∃ (y : ℝ), C_coordinates r A_x A_y = (5, y) ∧ -5 ≤ y ∧ y ≤ 5 := by sorry

end NUMINAMATH_CALUDE_product_OA_OC_constant_C_trajectory_l501_50159


namespace NUMINAMATH_CALUDE_picture_frame_problem_l501_50184

/-- Represents a rectangular picture frame -/
structure Frame where
  outer_length : ℝ
  outer_width : ℝ
  wood_width : ℝ

/-- Calculates the area of the frame material -/
def frame_area (f : Frame) : ℝ :=
  f.outer_length * f.outer_width - (f.outer_length - 2 * f.wood_width) * (f.outer_width - 2 * f.wood_width)

/-- Calculates the sum of the lengths of the four interior edges -/
def interior_perimeter (f : Frame) : ℝ :=
  2 * (f.outer_length - 2 * f.wood_width) + 2 * (f.outer_width - 2 * f.wood_width)

theorem picture_frame_problem :
  ∀ f : Frame,
    f.wood_width = 2 →
    f.outer_length = 7 →
    frame_area f = 34 →
    interior_perimeter f = 9 := by
  sorry

end NUMINAMATH_CALUDE_picture_frame_problem_l501_50184


namespace NUMINAMATH_CALUDE_star_polygon_n_value_l501_50173

/-- Represents an n-pointed regular star polygon -/
structure RegularStarPolygon where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ

/-- Properties of the regular star polygon -/
def is_valid_star_polygon (star : RegularStarPolygon) : Prop :=
  star.n > 0 ∧
  star.angle_A > 0 ∧
  star.angle_B > 0 ∧
  star.angle_A = star.angle_B - 15 ∧
  star.n * (star.angle_A + star.angle_B) = 360

theorem star_polygon_n_value (star : RegularStarPolygon) 
  (h : is_valid_star_polygon star) : star.n = 24 :=
by sorry

end NUMINAMATH_CALUDE_star_polygon_n_value_l501_50173


namespace NUMINAMATH_CALUDE_x_plus_one_greater_than_x_l501_50112

theorem x_plus_one_greater_than_x : ∀ x : ℝ, x + 1 > x := by
  sorry

end NUMINAMATH_CALUDE_x_plus_one_greater_than_x_l501_50112


namespace NUMINAMATH_CALUDE_unique_solution_xyz_l501_50140

theorem unique_solution_xyz : 
  ∀ x y z : ℕ+, 
    (x : ℤ) + (y : ℤ)^2 + (z : ℤ)^3 = (x : ℤ) * (y : ℤ) * (z : ℤ) → 
    z = Nat.gcd x y → 
    (x = 5 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_xyz_l501_50140


namespace NUMINAMATH_CALUDE_solutions_eq1_solutions_eq2_l501_50123

-- First equation
theorem solutions_eq1 (x : ℝ) : x^2 - 2*x - 3 = 0 ↔ x = 3 ∨ x = -1 := by sorry

-- Second equation
theorem solutions_eq2 (x : ℝ) : x*(x-2) + x - 2 = 0 ↔ x = -1 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_solutions_eq1_solutions_eq2_l501_50123


namespace NUMINAMATH_CALUDE_coinciding_vertices_l501_50139

/-- A point in the plane -/
structure Point :=
  (x y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- An isosceles right triangle defined by two points of the quadrilateral and a third point -/
structure IsoscelesRightTriangle :=
  (P Q R : Point)

/-- Predicate to check if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if two points coincide -/
def coincide (P Q : Point) : Prop := P.x = Q.x ∧ P.y = Q.y

/-- Theorem: If O₁ and O₃ coincide, then O₂ and O₄ coincide -/
theorem coinciding_vertices 
  (q : Quadrilateral) 
  (t1 : IsoscelesRightTriangle) 
  (t2 : IsoscelesRightTriangle) 
  (t3 : IsoscelesRightTriangle) 
  (t4 : IsoscelesRightTriangle) 
  (h1 : is_convex q)
  (h2 : t1.P = q.A ∧ t1.Q = q.B)
  (h3 : t2.P = q.B ∧ t2.Q = q.C)
  (h4 : t3.P = q.C ∧ t3.Q = q.D)
  (h5 : t4.P = q.D ∧ t4.Q = q.A)
  (h6 : coincide t1.R t3.R) :
  coincide t2.R t4.R := by sorry

end NUMINAMATH_CALUDE_coinciding_vertices_l501_50139


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l501_50135

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 400 →
  crossing_time = 45 →
  train_speed = 55.99999999999999 →
  ∃ (bridge_length : ℝ), bridge_length = train_speed * crossing_time - train_length ∧
                         bridge_length = 2120 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l501_50135


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l501_50134

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 + 6*x + 5 = 0 ↔ x = -1 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l501_50134


namespace NUMINAMATH_CALUDE_boys_on_trip_l501_50121

/-- Calculates the number of boys on a family trip given the specified conditions. -/
def number_of_boys (adults : ℕ) (total_eggs : ℕ) (eggs_per_adult : ℕ) (girls : ℕ) (eggs_per_girl : ℕ) : ℕ :=
  let eggs_for_children := total_eggs - adults * eggs_per_adult
  let eggs_for_girls := girls * eggs_per_girl
  let eggs_for_boys := eggs_for_children - eggs_for_girls
  let eggs_per_boy := eggs_per_girl + 1
  eggs_for_boys / eggs_per_boy

/-- Theorem stating that the number of boys on the trip is 10 under the given conditions. -/
theorem boys_on_trip :
  number_of_boys 3 (3 * 12) 3 7 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_on_trip_l501_50121


namespace NUMINAMATH_CALUDE_jim_journey_remaining_distance_l501_50125

/-- Calculates the remaining distance to drive given the total journey distance and the distance already driven. -/
def remaining_distance (total_distance driven_distance : ℕ) : ℕ :=
  total_distance - driven_distance

/-- Theorem stating that for a 1200-mile journey with 923 miles driven, the remaining distance is 277 miles. -/
theorem jim_journey_remaining_distance :
  remaining_distance 1200 923 = 277 := by
  sorry

end NUMINAMATH_CALUDE_jim_journey_remaining_distance_l501_50125


namespace NUMINAMATH_CALUDE_sum_abc_l501_50178

theorem sum_abc (a b c : ℤ) 
  (eq1 : 2 * a + 3 * b = 52)
  (eq2 : 3 * b + c = 41)
  (eq3 : b * c = 60) :
  a + b + c = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_l501_50178


namespace NUMINAMATH_CALUDE_parallel_tangents_and_zero_points_l501_50104

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x - a) / (x^2)

theorem parallel_tangents_and_zero_points (a : ℝ) (h : a > 0) :
  -- Part 1: Parallel tangents imply a = 3.5
  (f_deriv a 3 = f_deriv a (3/2) → a = 3.5) ∧
  -- Part 2: Zero points imply 0 < a ≤ 1
  (∃ x, f a x = 0 → 0 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_parallel_tangents_and_zero_points_l501_50104


namespace NUMINAMATH_CALUDE_find_N_l501_50146

theorem find_N : ∃ N : ℕ, 
  (981 + 983 + 985 + 987 + 989 + 991 + 993 = 7000 - N) ∧ (N = 91) := by
  sorry

end NUMINAMATH_CALUDE_find_N_l501_50146


namespace NUMINAMATH_CALUDE_sum_m_2n_3k_l501_50102

theorem sum_m_2n_3k (m n k : ℕ+) 
  (sum_mn : m + n = 2021)
  (prime_m_3k : Nat.Prime (m - 3*k))
  (prime_n_k : Nat.Prime (n + k)) :
  m + 2*n + 3*k = 2025 ∨ m + 2*n + 3*k = 4040 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_2n_3k_l501_50102


namespace NUMINAMATH_CALUDE_student_earnings_theorem_l501_50164

/-- Calculates the monthly earnings of a student working as a courier after tax deduction -/
def monthly_earnings_after_tax (daily_rate : ℝ) (days_per_week : ℕ) (weeks_per_month : ℕ) (tax_rate : ℝ) : ℝ :=
  let gross_monthly_earnings := daily_rate * (days_per_week : ℝ) * (weeks_per_month : ℝ)
  let tax_amount := gross_monthly_earnings * tax_rate
  gross_monthly_earnings - tax_amount

/-- Theorem stating that the monthly earnings of the student after tax is 17400 rubles -/
theorem student_earnings_theorem :
  monthly_earnings_after_tax 1250 4 4 0.13 = 17400 := by
  sorry

end NUMINAMATH_CALUDE_student_earnings_theorem_l501_50164


namespace NUMINAMATH_CALUDE_hotel_towels_l501_50136

/-- Calculates the total number of towels handed out in a hotel --/
def total_towels (num_rooms : ℕ) (people_per_room : ℕ) (towels_per_person : ℕ) : ℕ :=
  num_rooms * people_per_room * towels_per_person

/-- Proves that a hotel with 10 full rooms, 3 people per room, and 2 towels per person hands out 60 towels --/
theorem hotel_towels : total_towels 10 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_hotel_towels_l501_50136


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l501_50124

theorem unique_prime_with_prime_sums : ∃! p : ℕ, 
  Nat.Prime p ∧ Nat.Prime (p + 28) ∧ Nat.Prime (p + 56) ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l501_50124


namespace NUMINAMATH_CALUDE_y_plus_z_squared_positive_l501_50162

theorem y_plus_z_squared_positive 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 2 < z ∧ z < 3) : 
  0 < y + z^2 := by
  sorry

end NUMINAMATH_CALUDE_y_plus_z_squared_positive_l501_50162


namespace NUMINAMATH_CALUDE_min_value_sqrt_x_squared_plus_two_l501_50147

theorem min_value_sqrt_x_squared_plus_two (x : ℝ) :
  Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) ≥ 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_x_squared_plus_two_l501_50147


namespace NUMINAMATH_CALUDE_equation_solutions_l501_50183

theorem equation_solutions : 
  {x : ℝ | Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6} = {2, -1} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l501_50183


namespace NUMINAMATH_CALUDE_function_difference_l501_50174

theorem function_difference (F : ℝ → ℤ) (h1 : F 3 = 3) (h2 : F 1 = 2) :
  F 3 - F 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_difference_l501_50174


namespace NUMINAMATH_CALUDE_car_average_speed_l501_50133

theorem car_average_speed 
  (total_time : ℝ) 
  (initial_time : ℝ) 
  (initial_speed : ℝ) 
  (remaining_speed : ℝ) 
  (h1 : total_time = 24) 
  (h2 : initial_time = 4) 
  (h3 : initial_speed = 35) 
  (h4 : remaining_speed = 53) : 
  (initial_speed * initial_time + remaining_speed * (total_time - initial_time)) / total_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l501_50133


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l501_50145

theorem smallest_angle_solution (θ : Real) : 
  (θ > 0) → 
  (∀ φ, φ > 0 → φ < θ → Real.sin (10 * Real.pi / 180) ≠ Real.cos (40 * Real.pi / 180) - Real.cos (φ * Real.pi / 180)) →
  Real.sin (10 * Real.pi / 180) = Real.cos (40 * Real.pi / 180) - Real.cos (θ * Real.pi / 180) →
  θ = 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l501_50145


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l501_50187

theorem geometric_progression_ratio (x y z r : ℝ) 
  (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h2 : x * (2 * y - z) ≠ y * (2 * z - x))
  (h3 : y * (2 * z - x) ≠ z * (2 * x - y))
  (h4 : x * (2 * y - z) ≠ z * (2 * x - y))
  (h5 : ∃ (a : ℝ), a ≠ 0 ∧ 
    x * (2 * y - z) = a ∧ 
    y * (2 * z - x) = a * r ∧ 
    z * (2 * x - y) = a * r^2) :
  r^2 + r + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l501_50187


namespace NUMINAMATH_CALUDE_positive_root_m_value_l501_50192

theorem positive_root_m_value (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1 - x) / (x - 2) = m / (2 - x) - 2) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_root_m_value_l501_50192


namespace NUMINAMATH_CALUDE_joshua_friends_count_l501_50126

/-- Given that Joshua gave 40 Skittles to each friend and the total number of Skittles given is 200,
    prove that the number of friends Joshua gave Skittles to is 5. -/
theorem joshua_friends_count (skittles_per_friend : ℕ) (total_skittles : ℕ) 
    (h1 : skittles_per_friend = 40) 
    (h2 : total_skittles = 200) : 
  total_skittles / skittles_per_friend = 5 := by
sorry

end NUMINAMATH_CALUDE_joshua_friends_count_l501_50126


namespace NUMINAMATH_CALUDE_distance_negative_five_to_origin_l501_50137

theorem distance_negative_five_to_origin : 
  abs (-5 : ℝ) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_negative_five_to_origin_l501_50137


namespace NUMINAMATH_CALUDE_ratio_difference_l501_50176

/-- Given three numbers in ratio 3 : 5 : 7 with the largest being 70,
    the difference between the largest and smallest is 40. -/
theorem ratio_difference (a b c : ℝ) : 
  a / b = 3 / 5 → 
  b / c = 5 / 7 → 
  c = 70 → 
  c - a = 40 := by
sorry

end NUMINAMATH_CALUDE_ratio_difference_l501_50176


namespace NUMINAMATH_CALUDE_arrangement_count_l501_50141

def number_of_people : Nat := 6
def number_of_special_people : Nat := 3

theorem arrangement_count : 
  (number_of_people : Nat) = 6 →
  (number_of_special_people : Nat) = 3 →
  (∃ (arrangement_count : Nat), arrangement_count = 144 ∧
    arrangement_count = (number_of_people - number_of_special_people).factorial * 
                        (number_of_people - number_of_special_people + 1).choose number_of_special_people) :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l501_50141


namespace NUMINAMATH_CALUDE_complement_A_union_B_when_a_is_5_A_union_B_equals_A_iff_l501_50130

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | a - 2 ≤ x ∧ x ≤ 2*a - 3}

-- Statement for part 1
theorem complement_A_union_B_when_a_is_5 :
  (Set.univ \ A) ∪ B 5 = {x : ℝ | x ≤ -3 ∨ x ≥ 3} := by sorry

-- Statement for part 2
theorem A_union_B_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ a < 9/2 := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_when_a_is_5_A_union_B_equals_A_iff_l501_50130


namespace NUMINAMATH_CALUDE_win_bonus_area_l501_50191

/-- The combined area of WIN and BONUS sectors in a circular spinner -/
theorem win_bonus_area (r : ℝ) (p_win : ℝ) (p_bonus : ℝ) : 
  r = 8 → p_win = 1/4 → p_bonus = 1/8 → 
  (p_win + p_bonus) * (π * r^2) = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_win_bonus_area_l501_50191


namespace NUMINAMATH_CALUDE_min_ratio_rectangle_l501_50182

theorem min_ratio_rectangle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ k : ℝ, k > 0 → ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = k * a * b ∧ x + y = k * (a + b)) →
  (∃ k₀ : ℝ, k₀ > 0 ∧
    (∀ k : ℝ, k > 0 → (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = k * a * b ∧ x + y = k * (a + b)) → k ≥ k₀) ∧
    k₀ = 4 * a * b / ((a + b) ^ 2)) :=
sorry

end NUMINAMATH_CALUDE_min_ratio_rectangle_l501_50182


namespace NUMINAMATH_CALUDE_append_two_to_three_digit_number_l501_50110

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  is_valid : hundreds < 10 ∧ tens < 10 ∧ units < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNum (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Appends a digit to a number -/
def appendDigit (n : ℕ) (d : ℕ) : ℕ :=
  10 * n + d

theorem append_two_to_three_digit_number (n : ThreeDigitNumber) :
  appendDigit (ThreeDigitNumber.toNum n) 2 =
  1000 * n.hundreds + 100 * n.tens + 10 * n.units + 2 := by
  sorry

end NUMINAMATH_CALUDE_append_two_to_three_digit_number_l501_50110


namespace NUMINAMATH_CALUDE_solution_comparison_l501_50101

theorem solution_comparison (p p' q q' : ℕ+) (hp : p ≠ p') (hq : q ≠ q') :
  (-q : ℚ) / p > (-q' : ℚ) / p' ↔ q * p' < p * q' :=
sorry

end NUMINAMATH_CALUDE_solution_comparison_l501_50101


namespace NUMINAMATH_CALUDE_eddie_earnings_l501_50107

-- Define the work hours for each day
def monday_hours : ℚ := 5/2
def tuesday_hours : ℚ := 7/6
def wednesday_hours : ℚ := 7/4
def saturday_hours : ℚ := 3/4

-- Define the pay rates
def weekday_rate : ℚ := 4
def saturday_rate : ℚ := 6

-- Define the total earnings
def total_earnings : ℚ := 
  monday_hours * weekday_rate + 
  tuesday_hours * weekday_rate + 
  wednesday_hours * weekday_rate + 
  saturday_hours * saturday_rate

-- Theorem to prove
theorem eddie_earnings : total_earnings = 26.17 := by
  sorry

end NUMINAMATH_CALUDE_eddie_earnings_l501_50107


namespace NUMINAMATH_CALUDE_min_value_M_l501_50111

/-- Given a set A with exactly one element defined by a quadratic inequality,
    prove that the minimum value of M is 2√5 + 5 -/
theorem min_value_M (a b c : ℝ) (ha : 0 < a) (hb : a < b) :
  let A := {x : ℝ | a * x^2 + b * x + c ≤ 0}
  (∃! x, x ∈ A) →
  (∃ M₀ : ℝ, M₀ = 2 * Real.sqrt 5 + 5 ∧
   ∀ M : ℝ, M = (a + 3*b + 4*c) / (b - a) → M₀ ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_min_value_M_l501_50111


namespace NUMINAMATH_CALUDE_square_sum_formula_l501_50128

theorem square_sum_formula (x y a b : ℝ) 
  (h1 : x * y = 2 * b) 
  (h2 : 1 / x^2 + 1 / y^2 = a) : 
  (x + y)^2 = 4 * a * b^2 + 4 * b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_formula_l501_50128


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l501_50181

/-- A pyramid with an isosceles triangular base and equal lateral edges -/
structure IsoscelesPyramid where
  base_length : ℝ
  base_height : ℝ
  lateral_edge : ℝ

/-- The volume of an isosceles pyramid -/
def volume (p : IsoscelesPyramid) : ℝ := sorry

/-- Theorem: The volume of a specific isosceles pyramid is 108 -/
theorem specific_pyramid_volume :
  let p : IsoscelesPyramid := {
    base_length := 6,
    base_height := 9,
    lateral_edge := 13
  }
  volume p = 108 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l501_50181


namespace NUMINAMATH_CALUDE_delta_phi_equation_solution_l501_50197

-- Define the functions δ and φ
def δ (x : ℚ) : ℚ := 4 * x + 9
def φ (x : ℚ) : ℚ := 9 * x + 6

-- State the theorem
theorem delta_phi_equation_solution :
  ∃ x : ℚ, δ (φ x) = 10 ∧ x = -23 / 36 := by
  sorry

end NUMINAMATH_CALUDE_delta_phi_equation_solution_l501_50197


namespace NUMINAMATH_CALUDE_probability_two_diamonds_l501_50165

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of suits in a standard deck
def num_suits : ℕ := 4

-- Define the number of ranks in a standard deck
def num_ranks : ℕ := 13

-- Define the number of cards of a single suit (Diamonds in this case)
def cards_per_suit : ℕ := total_cards / num_suits

-- Theorem statement
theorem probability_two_diamonds (total_cards num_suits num_ranks cards_per_suit : ℕ) 
  (h1 : total_cards = 52)
  (h2 : num_suits = 4)
  (h3 : num_ranks = 13)
  (h4 : cards_per_suit = total_cards / num_suits) :
  (cards_per_suit.choose 2 : ℚ) / (total_cards.choose 2) = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_diamonds_l501_50165


namespace NUMINAMATH_CALUDE_unique_function_solution_l501_50149

-- Define the property that f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = 1 - x - y

-- State the theorem
theorem unique_function_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → ∀ x : ℝ, f x = 1/2 - x :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l501_50149


namespace NUMINAMATH_CALUDE_pencil_eraser_combinations_l501_50156

/-- The number of possible combinations when choosing one item from each of two sets -/
def combinations (set1 : ℕ) (set2 : ℕ) : ℕ := set1 * set2

/-- Theorem: The number of combinations when choosing one pencil from 2 types
    and one eraser from 3 types is equal to 6 -/
theorem pencil_eraser_combinations :
  combinations 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_combinations_l501_50156


namespace NUMINAMATH_CALUDE_minAreaLineEquation_l501_50118

/-- A line passing through a point (x₀, y₀) -/
structure Line where
  slope : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- The area of the triangle formed by a line and the coordinate axes -/
def triangleArea (l : Line) : ℝ :=
  sorry

/-- The line passing through (1, 2) that minimizes the triangle area -/
noncomputable def minAreaLine : Line :=
  sorry

theorem minAreaLineEquation :
  let l := minAreaLine
  l.x₀ = 1 ∧ l.y₀ = 2 ∧
  ∀ (m : Line), m.x₀ = 1 ∧ m.y₀ = 2 → triangleArea l ≤ triangleArea m ∧
  2 * l.x₀ + l.y₀ - 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_minAreaLineEquation_l501_50118


namespace NUMINAMATH_CALUDE_valid_numbers_l501_50185

def is_valid_number (n : ℕ) : Prop :=
  Odd n ∧
  ∃ (a b : ℕ),
    10 ≤ a ∧ a ≤ 99 ∧
    (∃ (k : ℕ), n = 10^k * a + b) ∧
    n = 149 * b

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n → n = 745 ∨ n = 3725 :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l501_50185


namespace NUMINAMATH_CALUDE_ellipse_condition_necessary_not_sufficient_l501_50150

/-- The condition for the equation to potentially represent an ellipse -/
def ellipse_condition (m : ℝ) : Prop := 1 < m ∧ m < 3

/-- The equation representing a potential ellipse -/
def ellipse_equation (m x y : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / (3 - m) = 1

/-- Predicate for whether the equation represents an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse_equation m x y ∧ 
  ¬(∃ c : ℝ, ∀ x y : ℝ, ellipse_equation m x y ↔ x^2 + y^2 = c)

theorem ellipse_condition_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → ellipse_condition m) ∧
  ¬(∀ m : ℝ, ellipse_condition m → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_necessary_not_sufficient_l501_50150


namespace NUMINAMATH_CALUDE_granola_bars_eaten_by_parents_l501_50122

theorem granola_bars_eaten_by_parents (total : ℕ) (children : ℕ) (per_child : ℕ) 
  (h1 : total = 200) 
  (h2 : children = 6) 
  (h3 : per_child = 20) : 
  total - (children * per_child) = 80 :=
by sorry

end NUMINAMATH_CALUDE_granola_bars_eaten_by_parents_l501_50122


namespace NUMINAMATH_CALUDE_cubic_system_unique_solution_l501_50155

theorem cubic_system_unique_solution (x y : ℝ) 
  (h1 : x^3 = 2 - y) (h2 : y^3 = 2 - x) : x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_cubic_system_unique_solution_l501_50155


namespace NUMINAMATH_CALUDE_cody_marbles_l501_50108

theorem cody_marbles (initial_marbles : ℕ) : 
  (initial_marbles - initial_marbles / 3 - 5 = 7) → initial_marbles = 18 := by
  sorry

end NUMINAMATH_CALUDE_cody_marbles_l501_50108


namespace NUMINAMATH_CALUDE_trig_identities_l501_50120

/-- Given tan α = 2, prove two trigonometric identities -/
theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 6/13 ∧
  3 * Real.sin α^2 + 3 * Real.sin α * Real.cos α - 2 * Real.cos α^2 = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l501_50120


namespace NUMINAMATH_CALUDE_calories_left_for_dinner_l501_50142

def daily_allowance : ℕ := 2200
def breakfast_calories : ℕ := 353
def lunch_calories : ℕ := 885
def snack_calories : ℕ := 130

theorem calories_left_for_dinner :
  daily_allowance - (breakfast_calories + lunch_calories + snack_calories) = 832 := by
  sorry

end NUMINAMATH_CALUDE_calories_left_for_dinner_l501_50142


namespace NUMINAMATH_CALUDE_chloe_bookcase_problem_l501_50113

theorem chloe_bookcase_problem :
  let average_books_per_shelf : ℚ := 8.5
  let mystery_shelves : ℕ := 7
  let picture_shelves : ℕ := 5
  let scifi_shelves : ℕ := 3
  let history_shelves : ℕ := 2
  let total_shelves : ℕ := mystery_shelves + picture_shelves + scifi_shelves + history_shelves
  let total_books : ℚ := average_books_per_shelf * total_shelves
  ⌈total_books⌉ = 145 := by
  sorry

#check chloe_bookcase_problem

end NUMINAMATH_CALUDE_chloe_bookcase_problem_l501_50113


namespace NUMINAMATH_CALUDE_ball_reaches_top_left_pocket_l501_50157

/-- Represents a point on the billiard table or its reflections -/
structure TablePoint where
  x : Int
  y : Int

/-- Represents the dimensions of the billiard table -/
structure TableDimensions where
  width : Nat
  height : Nat

/-- Checks if a point is a top-left pocket in the reflected grid -/
def isTopLeftPocket (p : TablePoint) (dim : TableDimensions) : Prop :=
  ∃ (m n : Int), p.x = dim.width * m ∧ p.y = dim.height * n ∧ m % 2 = 0 ∧ n % 2 = 1

/-- The theorem stating that the ball will reach the top-left pocket -/
theorem ball_reaches_top_left_pocket (dim : TableDimensions) 
  (h_dim : dim.width = 1965 ∧ dim.height = 26) :
  ∃ (p : TablePoint), p.y = p.x ∧ isTopLeftPocket p dim := by
  sorry

end NUMINAMATH_CALUDE_ball_reaches_top_left_pocket_l501_50157


namespace NUMINAMATH_CALUDE_desmond_toy_purchase_l501_50168

/-- The number of toys Mr. Desmond bought for his elder son -/
def elder_son_toys : ℕ := 60

/-- The number of toys Mr. Desmond bought for his younger son -/
def younger_son_toys : ℕ := 3 * elder_son_toys

/-- The total number of toys Mr. Desmond bought -/
def total_toys : ℕ := elder_son_toys + younger_son_toys

theorem desmond_toy_purchase :
  total_toys = 240 := by sorry

end NUMINAMATH_CALUDE_desmond_toy_purchase_l501_50168


namespace NUMINAMATH_CALUDE_jack_book_sale_l501_50154

/-- Calculates the amount received from selling books after a year --/
def amount_received (books_per_month : ℕ) (cost_per_book : ℕ) (months : ℕ) (loss : ℕ) : ℕ :=
  books_per_month * months * cost_per_book - loss

/-- Proves that Jack received $500 from selling the books --/
theorem jack_book_sale : amount_received 3 20 12 220 = 500 := by
  sorry

end NUMINAMATH_CALUDE_jack_book_sale_l501_50154


namespace NUMINAMATH_CALUDE_decimal_addition_l501_50177

theorem decimal_addition : (5.46 : ℝ) + 4.537 = 9.997 := by sorry

end NUMINAMATH_CALUDE_decimal_addition_l501_50177
