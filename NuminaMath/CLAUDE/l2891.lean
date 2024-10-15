import Mathlib

namespace NUMINAMATH_CALUDE_apple_weeks_theorem_l2891_289149

/-- The number of weeks Henry and his brother can spend eating apples -/
def appleWeeks (applesPerBox : ℕ) (numBoxes : ℕ) (applesPerPersonPerDay : ℕ) (numPeople : ℕ) (daysPerWeek : ℕ) : ℕ :=
  (applesPerBox * numBoxes) / (applesPerPersonPerDay * numPeople * daysPerWeek)

/-- Theorem stating that Henry and his brother can spend 3 weeks eating the apples -/
theorem apple_weeks_theorem : appleWeeks 14 3 1 2 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_weeks_theorem_l2891_289149


namespace NUMINAMATH_CALUDE_certain_percent_problem_l2891_289136

theorem certain_percent_problem (P : ℝ) : 
  (P / 100) * 500 = (50 / 100) * 600 → P = 60 := by
  sorry

end NUMINAMATH_CALUDE_certain_percent_problem_l2891_289136


namespace NUMINAMATH_CALUDE_largest_fraction_l2891_289157

theorem largest_fraction : 
  let a := 4 / (2 - 1/4)
  let b := 4 / (2 + 1/4)
  let c := 4 / (2 - 1/3)
  let d := 4 / (2 + 1/3)
  let e := 4 / (2 - 1/2)
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l2891_289157


namespace NUMINAMATH_CALUDE_min_circle_and_common_chord_for_given_points_l2891_289111

/-- The circle with the smallest circumference passing through two given points -/
structure MinCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- The common chord between two intersecting circles -/
structure CommonChord where
  length : ℝ

/-- Given points A and B, find the circle with smallest circumference passing through them
    and calculate its common chord length with another given circle -/
def find_min_circle_and_common_chord 
  (A B : ℝ × ℝ) 
  (C₂ : ℝ → ℝ → Prop) : MinCircle × CommonChord :=
sorry

theorem min_circle_and_common_chord_for_given_points :
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (2, -2)
  let C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 2*y + 5 = 0
  let (min_circle, common_chord) := find_min_circle_and_common_chord A B C₂
  min_circle.center = (1, 0) ∧
  min_circle.radius = Real.sqrt 5 ∧
  common_chord.length = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_min_circle_and_common_chord_for_given_points_l2891_289111


namespace NUMINAMATH_CALUDE_sum_greater_than_four_necessity_not_sufficiency_l2891_289146

theorem sum_greater_than_four_necessity_not_sufficiency (a b : ℝ) :
  (((a > 2) ∧ (b > 2)) → (a + b > 4)) ∧
  (∃ a b : ℝ, (a + b > 4) ∧ ¬((a > 2) ∧ (b > 2))) :=
by sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_necessity_not_sufficiency_l2891_289146


namespace NUMINAMATH_CALUDE_points_per_round_l2891_289177

theorem points_per_round (total_points : ℕ) (num_rounds : ℕ) (points_per_round : ℕ) 
  (h1 : total_points = 84)
  (h2 : num_rounds = 2)
  (h3 : total_points = num_rounds * points_per_round) :
  points_per_round = 42 := by
sorry

end NUMINAMATH_CALUDE_points_per_round_l2891_289177


namespace NUMINAMATH_CALUDE_no_five_integers_with_prime_triples_l2891_289190

theorem no_five_integers_with_prime_triples : ¬ ∃ (a b c d e : ℕ+),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  (∀ (x y z : ℕ+), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) →
                   (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) →
                   (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) →
                   (x ≠ y ∧ x ≠ z ∧ y ≠ z) →
                   Nat.Prime (x.val + y.val + z.val)) :=
by sorry

end NUMINAMATH_CALUDE_no_five_integers_with_prime_triples_l2891_289190


namespace NUMINAMATH_CALUDE_custom_mul_five_three_l2891_289106

-- Define the custom multiplication operation
def custom_mul (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- Theorem statement
theorem custom_mul_five_three : custom_mul 5 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_five_three_l2891_289106


namespace NUMINAMATH_CALUDE_smallest_non_prime_without_small_factors_l2891_289105

theorem smallest_non_prime_without_small_factors :
  ∃ n : ℕ,
    n > 1 ∧
    ¬ (Nat.Prime n) ∧
    (∀ p : ℕ, Nat.Prime p → p < 10 → ¬ (p ∣ n)) ∧
    (∀ m : ℕ, m > 1 → ¬ (Nat.Prime m) → (∀ q : ℕ, Nat.Prime q → q < 10 → ¬ (q ∣ m)) → m ≥ n) ∧
    120 < n ∧
    n ≤ 130 :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_without_small_factors_l2891_289105


namespace NUMINAMATH_CALUDE_perpendicular_polygon_area_l2891_289181

/-- A polygon with perpendicular adjacent sides -/
structure PerpendicularPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  area : ℝ
  sides_congruent : sides > 0
  perimeter_eq : perimeter = sides * side_length
  area_calc : area = 16 * side_length^2

/-- Theorem: The area of a specific perpendicular polygon -/
theorem perpendicular_polygon_area :
  ∀ (p : PerpendicularPolygon),
    p.sides = 20 ∧ p.perimeter = 60 → p.area = 144 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_polygon_area_l2891_289181


namespace NUMINAMATH_CALUDE_mr_orange_yield_l2891_289137

/-- Calculates the expected orange yield from a triangular garden --/
def expected_orange_yield (base_paces : ℕ) (height_paces : ℕ) (feet_per_pace : ℕ) (yield_per_sqft : ℚ) : ℚ :=
  let base_feet := base_paces * feet_per_pace
  let height_feet := height_paces * feet_per_pace
  let area := (base_feet * height_feet : ℚ) / 2
  area * yield_per_sqft

/-- Theorem stating the expected orange yield for Mr. Orange's garden --/
theorem mr_orange_yield :
  expected_orange_yield 18 24 3 (3/4) = 1458 := by
  sorry

end NUMINAMATH_CALUDE_mr_orange_yield_l2891_289137


namespace NUMINAMATH_CALUDE_opposite_sign_and_integer_part_l2891_289173

theorem opposite_sign_and_integer_part (a b c : ℝ) : 
  (∃ (k : ℝ), k * (Real.sqrt (a - 4)) = -(2 - 2*b)^2 ∧ k ≠ 0) →
  c = ⌊Real.sqrt 10⌋ →
  a = 4 ∧ b = 1 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_and_integer_part_l2891_289173


namespace NUMINAMATH_CALUDE_cos_210_degrees_l2891_289175

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l2891_289175


namespace NUMINAMATH_CALUDE_line_slope_intercept_form_l2891_289129

/-- Definition of the line using vector dot product -/
def line_equation (x y : ℝ) : Prop :=
  (3 * (x - 2)) + (-4 * (y - 8)) = 0

/-- Slope-intercept form of a line -/
def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

/-- Theorem stating that the given line equation is equivalent to y = (3/4)x + 6.5 -/
theorem line_slope_intercept_form :
  ∀ x y : ℝ, line_equation x y ↔ slope_intercept_form (3/4) (13/2) x y :=
by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_form_l2891_289129


namespace NUMINAMATH_CALUDE_derivative_at_one_l2891_289165

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = 2 * x * f' 1 + Real.log x) →
  (∀ x, HasDerivAt f (f' x) x) →
  f' 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2891_289165


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2891_289104

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- Given that point A(a,1) is symmetric to point A'(5,b) with respect to the origin, prove that a + b = -6 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin a 1 5 b) : a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2891_289104


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2891_289160

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 10 ∧ 
  (∀ (y : ℕ), y < x → ¬(21 ∣ (105829 - y))) ∧ 
  (21 ∣ (105829 - x)) := by
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2891_289160


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2891_289126

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, 2^x > 2 → 1/x < 1) ∧ 
  (∃ x, 1/x < 1 ∧ 2^x ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2891_289126


namespace NUMINAMATH_CALUDE_andrew_stickers_l2891_289197

theorem andrew_stickers (daniel_stickers fred_stickers andrew_kept : ℕ) 
  (h1 : daniel_stickers = 250)
  (h2 : fred_stickers = daniel_stickers + 120)
  (h3 : andrew_kept = 130) :
  andrew_kept + daniel_stickers + fred_stickers = 750 :=
by sorry

end NUMINAMATH_CALUDE_andrew_stickers_l2891_289197


namespace NUMINAMATH_CALUDE_range_of_a_l2891_289187

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2^x - 2 > a^2 - 3*a) → a ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2891_289187


namespace NUMINAMATH_CALUDE_players_bought_l2891_289133

/-- Calculates the number of players bought by a football club given their financial transactions -/
theorem players_bought (initial_balance : ℕ) (players_sold : ℕ) (selling_price : ℕ) (buying_price : ℕ) (final_balance : ℕ) : 
  initial_balance + players_sold * selling_price - final_balance = 4 * buying_price :=
by
  sorry

#check players_bought 100000000 2 10000000 15000000 60000000

end NUMINAMATH_CALUDE_players_bought_l2891_289133


namespace NUMINAMATH_CALUDE_impossible_tiling_l2891_289119

/-- Represents a rectangle with shaded cells -/
structure ShadedRectangle where
  rows : Nat
  cols : Nat
  shaded_cells : Nat

/-- Represents a tiling strip -/
structure TilingStrip where
  width : Nat
  height : Nat

/-- Checks if a rectangle can be tiled with given strips -/
def canBeTiled (rect : ShadedRectangle) (strip : TilingStrip) : Prop :=
  rect.rows * rect.cols % (strip.width * strip.height) = 0 ∧
  rect.shaded_cells % strip.width = 0 ∧
  rect.shaded_cells / strip.width = rect.rows * rect.cols / (strip.width * strip.height)

theorem impossible_tiling (rect : ShadedRectangle) (strip : TilingStrip) :
  rect.rows = 4 ∧ rect.cols = 9 ∧ rect.shaded_cells = 15 ∧
  strip.width = 3 ∧ strip.height = 1 →
  ¬ canBeTiled rect strip := by
  sorry

#check impossible_tiling

end NUMINAMATH_CALUDE_impossible_tiling_l2891_289119


namespace NUMINAMATH_CALUDE_f_2009_equals_1_l2891_289166

def is_even_function (f : ℤ → ℤ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_2009_equals_1
  (f : ℤ → ℤ)
  (h_even : is_even_function f)
  (h_f_1 : f 1 = 1)
  (h_f_2008 : f 2008 ≠ 1)
  (h_max : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)) :
  f 2009 = 1 := by
sorry

end NUMINAMATH_CALUDE_f_2009_equals_1_l2891_289166


namespace NUMINAMATH_CALUDE_carter_reads_30_pages_l2891_289147

/-- The number of pages Oliver can read in 1 hour -/
def oliver_pages : ℕ := 40

/-- The number of pages Lucy can read in 1 hour -/
def lucy_pages : ℕ := oliver_pages + 20

/-- The number of pages Carter can read in 1 hour -/
def carter_pages : ℕ := lucy_pages / 2

/-- Theorem: Carter can read 30 pages in 1 hour -/
theorem carter_reads_30_pages : carter_pages = 30 := by
  sorry

end NUMINAMATH_CALUDE_carter_reads_30_pages_l2891_289147


namespace NUMINAMATH_CALUDE_min_boat_speed_l2891_289131

/-- The minimum speed required for a boat to complete a round trip on a river with a given flow speed, distance, and time constraint. -/
theorem min_boat_speed (S v : ℝ) (h_S : S > 0) (h_v : v ≥ 0) :
  let min_speed := (3 * S + Real.sqrt (9 * S^2 + 4 * v^2)) / 2
  ∀ x : ℝ, x ≥ min_speed →
    S / (x - v) + S / (x + v) + 1/12 ≤ 3/4 :=
by sorry

end NUMINAMATH_CALUDE_min_boat_speed_l2891_289131


namespace NUMINAMATH_CALUDE_product_of_powers_l2891_289142

theorem product_of_powers (n : ℕ) (hn : n > 1) :
  (n + 1) * (n^2 + 1) * (n^4 + 1) * (n^8 + 1) * (n^16 + 1) = 
    if n = 2 then
      n^32 - 1
    else
      (n^32 - 1) / (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_product_of_powers_l2891_289142


namespace NUMINAMATH_CALUDE_line_intercepts_equal_l2891_289161

theorem line_intercepts_equal (a : ℝ) :
  (∀ x y : ℝ, (a + 1) * x + y + 2 - a = 0) →
  (∃ k : ℝ, k ≠ 0 ∧ k = a - 2 ∧ k = (a - 2) / (a + 1)) →
  (a = 2 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_equal_l2891_289161


namespace NUMINAMATH_CALUDE_emily_seeds_l2891_289128

/-- Calculates the total number of seeds Emily started with -/
def total_seeds (big_garden_seeds : ℕ) (num_small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  big_garden_seeds + num_small_gardens * seeds_per_small_garden

/-- Proves that Emily started with 41 seeds -/
theorem emily_seeds : 
  total_seeds 29 3 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_emily_seeds_l2891_289128


namespace NUMINAMATH_CALUDE_probability_is_sqrt_two_over_fifteen_l2891_289192

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability of x^2 < y for a point (x,y) randomly picked from the given rectangle --/
def probability_x_squared_less_than_y (rect : Rectangle) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problem_rectangle : Rectangle :=
  { x_min := 0
  , x_max := 5
  , y_min := 0
  , y_max := 2
  , h_x := by norm_num
  , h_y := by norm_num
  }

theorem probability_is_sqrt_two_over_fifteen :
  probability_x_squared_less_than_y problem_rectangle = Real.sqrt 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_sqrt_two_over_fifteen_l2891_289192


namespace NUMINAMATH_CALUDE_four_distinct_roots_condition_l2891_289102

-- Define the equation
def equation (x a : ℝ) : Prop := |x^2 - 4| = a * x + 6

-- Define the condition for four distinct roots
def has_four_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    equation x₁ a ∧ equation x₂ a ∧ equation x₃ a ∧ equation x₄ a

-- Theorem statement
theorem four_distinct_roots_condition (a : ℝ) :
  has_four_distinct_roots a ↔ (-3 < a ∧ a < -2 * Real.sqrt 2) ∨ (2 * Real.sqrt 2 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_four_distinct_roots_condition_l2891_289102


namespace NUMINAMATH_CALUDE_yadav_clothes_transport_expense_l2891_289153

/-- Represents Mr. Yadav's monthly finances -/
structure YadavFinances where
  monthlySalary : ℝ
  consumablePercentage : ℝ
  clothesTransportPercentage : ℝ
  yearlySavings : ℝ

/-- Calculates the monthly amount spent on clothes and transport -/
def clothesTransportExpense (y : YadavFinances) : ℝ :=
  y.monthlySalary * (1 - y.consumablePercentage) * y.clothesTransportPercentage

theorem yadav_clothes_transport_expense :
  ∀ (y : YadavFinances),
    y.consumablePercentage = 0.6 →
    y.clothesTransportPercentage = 0.5 →
    y.yearlySavings = 46800 →
    y.monthlySalary * (1 - y.consumablePercentage) * (1 - y.clothesTransportPercentage) = y.yearlySavings / 12 →
    clothesTransportExpense y = 3900 := by
  sorry

#check yadav_clothes_transport_expense

end NUMINAMATH_CALUDE_yadav_clothes_transport_expense_l2891_289153


namespace NUMINAMATH_CALUDE_rice_distribution_l2891_289130

/-- Given 33/4 pounds of rice divided equally into 4 containers, 
    and 1 pound equals 16 ounces, prove that each container contains 15 ounces of rice. -/
theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 33 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound) / num_containers = 15 :=
by sorry

end NUMINAMATH_CALUDE_rice_distribution_l2891_289130


namespace NUMINAMATH_CALUDE_disjoint_triangles_probability_disjoint_triangles_probability_proof_l2891_289194

/-- The probability that two triangles formed by six points chosen sequentially at random on a circle's circumference are disjoint -/
theorem disjoint_triangles_probability : ℚ :=
  3/10

/-- Total number of distinct arrangements with one point fixed -/
def total_arrangements : ℕ := 120

/-- Number of favorable outcomes where the triangles are disjoint -/
def favorable_outcomes : ℕ := 36

theorem disjoint_triangles_probability_proof :
  disjoint_triangles_probability = (favorable_outcomes : ℚ) / total_arrangements :=
by sorry

end NUMINAMATH_CALUDE_disjoint_triangles_probability_disjoint_triangles_probability_proof_l2891_289194


namespace NUMINAMATH_CALUDE_bridge_length_proof_l2891_289193

/-- Given a train with length 160 meters, traveling at 45 km/hr, that crosses a bridge in 30 seconds, prove that the length of the bridge is 215 meters. -/
theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 160 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 215 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l2891_289193


namespace NUMINAMATH_CALUDE_eighth_term_value_l2891_289132

theorem eighth_term_value (S : ℕ → ℕ) (h : ∀ n : ℕ, S n = n^2) :
  S 8 - S 7 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l2891_289132


namespace NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l2891_289117

/-- The sum of the infinite series Σ(k=1 to ∞) [12^k / ((4^k - 3^k)(4^(k+1) - 3^(k+1)))] is equal to 1 -/
theorem infinite_series_sum_equals_one :
  let series_term (k : ℕ) := (12 : ℝ)^k / ((4 : ℝ)^k - (3 : ℝ)^k) / ((4 : ℝ)^(k+1) - (3 : ℝ)^(k+1))
  ∑' k, series_term k = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l2891_289117


namespace NUMINAMATH_CALUDE_crosswalk_stripe_distance_l2891_289191

theorem crosswalk_stripe_distance 
  (curb_distance : ℝ) 
  (curb_length : ℝ) 
  (stripe_length : ℝ) 
  (h1 : curb_distance = 30) 
  (h2 : curb_length = 10) 
  (h3 : stripe_length = 60) : 
  ∃ (stripe_distance : ℝ), 
    stripe_distance * stripe_length = curb_length * curb_distance ∧ 
    stripe_distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_crosswalk_stripe_distance_l2891_289191


namespace NUMINAMATH_CALUDE_max_value_theorem_l2891_289109

theorem max_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : 0 < c ∧ c < 3) :
  ∃ (M : ℝ), ∀ (x : ℝ), x ≥ 0 →
    3 * (a - x) * (x + Real.sqrt (x^2 + b^2)) + c * x ≤ M ∧
    M = (3 - c) / 2 * b^2 + 9 * a^2 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2891_289109


namespace NUMINAMATH_CALUDE_max_value_of_f_l2891_289152

-- Define the function f(x)
def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = -2) →
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≥ -2) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = 25) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 25) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2891_289152


namespace NUMINAMATH_CALUDE_complex_on_negative_y_axis_l2891_289151

def complex_operation : ℂ := (5 - 6*Complex.I) + (-2 - Complex.I) - (3 + 4*Complex.I)

theorem complex_on_negative_y_axis : 
  complex_operation.re = 0 ∧ complex_operation.im < 0 :=
sorry

end NUMINAMATH_CALUDE_complex_on_negative_y_axis_l2891_289151


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2891_289199

/-- A function that checks if a quadratic equation with coefficients based on A has positive integer solutions -/
def has_positive_integer_solutions (A : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - (2*A)*x + (A+1)*10 = 0 ∧ y^2 - (2*A)*y + (A+1)*10 = 0

/-- The theorem stating that there is exactly one single-digit positive integer A that satisfies the condition -/
theorem unique_quadratic_solution : 
  ∃! A : ℕ, 1 ≤ A ∧ A ≤ 9 ∧ has_positive_integer_solutions A :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2891_289199


namespace NUMINAMATH_CALUDE_cookies_per_person_l2891_289108

theorem cookies_per_person (total_cookies : ℕ) (num_people : ℚ) 
  (h1 : total_cookies = 144) 
  (h2 : num_people = 6.0) : 
  (total_cookies : ℚ) / num_people = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_person_l2891_289108


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l2891_289148

theorem quadratic_equation_value (a : ℝ) (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l2891_289148


namespace NUMINAMATH_CALUDE_min_value_theorem_l2891_289100

theorem min_value_theorem (a b : ℝ) 
  (h : ∀ x : ℝ, Real.log (x + 1) - (a + 2) * x ≤ b - 2) : 
  ∃ m : ℝ, m = 1 - Real.exp 1 ∧ ∀ y : ℝ, y = (b - 3) / (a + 2) → y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2891_289100


namespace NUMINAMATH_CALUDE_find_c_l2891_289107

/-- Given two functions p and q, prove that c = 6 when p(q(3)) = 10 -/
theorem find_c (p q : ℝ → ℝ) (c : ℝ) : 
  (∀ x, p x = 3 * x - 8) →
  (∀ x, q x = 4 * x - c) →
  p (q 3) = 10 →
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_find_c_l2891_289107


namespace NUMINAMATH_CALUDE_equation_satisfaction_l2891_289120

theorem equation_satisfaction (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) : 
  ((10 * a + b) * (10 * b + a) = 100 * a^2 + a * b + 100 * b^2) ↔ (a = b) :=
by sorry

end NUMINAMATH_CALUDE_equation_satisfaction_l2891_289120


namespace NUMINAMATH_CALUDE_line_equation_l2891_289179

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point being on a line
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the midpoint of two points
def is_midpoint (p m q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

-- Theorem statement
theorem line_equation : 
  ∀ (l : Line) (p a b : Point),
    on_line p l →
    p.x = 4 ∧ p.y = 1 →
    hyperbola a.x a.y →
    hyperbola b.x b.y →
    is_midpoint a p b →
    l.a = 1 ∧ l.b = -1 ∧ l.c = -3 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l2891_289179


namespace NUMINAMATH_CALUDE_expected_pine_saplings_l2891_289188

theorem expected_pine_saplings 
  (total_saplings : ℕ) 
  (pine_saplings : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_saplings = 30000) 
  (h2 : pine_saplings = 4000) 
  (h3 : sample_size = 150) : 
  ℕ :=
  20

#check expected_pine_saplings

end NUMINAMATH_CALUDE_expected_pine_saplings_l2891_289188


namespace NUMINAMATH_CALUDE_not_counterexample_58_l2891_289172

theorem not_counterexample_58 (h : ¬ Prime 58) : 
  ¬ (Prime 58 ∧ ¬ Prime 60) := by sorry

end NUMINAMATH_CALUDE_not_counterexample_58_l2891_289172


namespace NUMINAMATH_CALUDE_line_MN_equation_l2891_289159

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := 3 * x^2 + 8 * y^2 = 48

-- Define points A, B, and C
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (2, 0)

-- Define P and Q as points on the ellipse
def P_on_ellipse (P : ℝ × ℝ) : Prop := is_on_ellipse P.1 P.2
def Q_on_ellipse (Q : ℝ × ℝ) : Prop := is_on_ellipse Q.1 Q.2

-- PQ passes through C but not origin
def PQ_through_C (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ C = (t • P.1 + (1 - t) • Q.1, t • P.2 + (1 - t) • Q.2)

-- Define M as intersection of AP and QB
def M_is_intersection (P Q M : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, M = (t₁ • A.1 + (1 - t₁) • P.1, t₁ • A.2 + (1 - t₁) • P.2) ∧
              M = (t₂ • Q.1 + (1 - t₂) • B.1, t₂ • Q.2 + (1 - t₂) • B.2)

-- Define N as intersection of PB and AQ
def N_is_intersection (P Q N : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, N = (t₁ • P.1 + (1 - t₁) • B.1, t₁ • P.2 + (1 - t₁) • B.2) ∧
              N = (t₂ • A.1 + (1 - t₂) • Q.1, t₂ • A.2 + (1 - t₂) • Q.2)

-- The main theorem
theorem line_MN_equation (P Q M N : ℝ × ℝ) :
  P_on_ellipse P → Q_on_ellipse Q → PQ_through_C P Q →
  M_is_intersection P Q M → N_is_intersection P Q N →
  M.1 = 8 ∧ N.1 = 8 :=
sorry

end NUMINAMATH_CALUDE_line_MN_equation_l2891_289159


namespace NUMINAMATH_CALUDE_a_bounds_l2891_289169

theorem a_bounds (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3)
  (sum_squares : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) :
  1 ≤ a ∧ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_a_bounds_l2891_289169


namespace NUMINAMATH_CALUDE_spinner_probability_l2891_289145

/-- A spinner with six sections numbered 1, 3, 5, 7, 8, and 9 -/
def Spinner : Finset ℕ := {1, 3, 5, 7, 8, 9}

/-- The set of numbers on the spinner that are less than 4 -/
def LessThan4 : Finset ℕ := Spinner.filter (· < 4)

/-- The probability of spinning a number less than 4 -/
def probability : ℚ := (LessThan4.card : ℚ) / (Spinner.card : ℚ)

theorem spinner_probability : probability = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2891_289145


namespace NUMINAMATH_CALUDE_tangent_ellipse_major_axis_length_l2891_289138

/-- An ellipse with foci at (3, -4 + 2√3) and (3, -4 - 2√3), tangent to both x and y axes -/
structure TangentEllipse where
  /-- The ellipse is tangent to the x-axis -/
  tangent_x : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_y : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- Ensure the foci are at the specified points -/
  foci_constraint : focus1 = (3, -4 + 2 * Real.sqrt 3) ∧ focus2 = (3, -4 - 2 * Real.sqrt 3)
  /-- Ensure the ellipse is tangent to both axes -/
  tangent_constraint : tangent_x ∧ tangent_y

/-- The length of the major axis of the ellipse -/
def majorAxisLength (e : TangentEllipse) : ℝ := 8

/-- Theorem stating that the major axis length of the specified ellipse is 8 -/
theorem tangent_ellipse_major_axis_length (e : TangentEllipse) : 
  majorAxisLength e = 8 := by sorry

end NUMINAMATH_CALUDE_tangent_ellipse_major_axis_length_l2891_289138


namespace NUMINAMATH_CALUDE_fraction_equivalent_with_difference_l2891_289185

theorem fraction_equivalent_with_difference : ∃ (a b : ℕ), 
  a > 0 ∧ b > 0 ∧ (a : ℚ) / b = 7 / 13 ∧ b - a = 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalent_with_difference_l2891_289185


namespace NUMINAMATH_CALUDE_calculation_proof_l2891_289118

theorem calculation_proof : (1/2)⁻¹ + 4 * Real.cos (60 * π / 180) - (5 - Real.pi)^0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2891_289118


namespace NUMINAMATH_CALUDE_parallel_lines_minimum_value_l2891_289184

-- Define the linear functions f and g
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

-- Define the theorem
theorem parallel_lines_minimum_value 
  (a b c : ℝ) 
  (h1 : a ≠ 0)  -- Ensure lines are not parallel to coordinate axes
  (h2 : ∃ (x : ℝ), (f a b x)^2 + g a c x = 4)  -- Minimum value of (f(x))^2 + g(x) is 4
  : ∃ (x : ℝ), (g a c x)^2 + f a b x = -9/2 :=  -- Minimum value of (g(x))^2 + f(x) is -9/2
by sorry

end NUMINAMATH_CALUDE_parallel_lines_minimum_value_l2891_289184


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2891_289162

/-- Given a line l passing through points (a-2, -1) and (-a-2, 1), perpendicular to a line
    passing through (-2, 1) with slope -2/3, prove that a = -2/3 --/
theorem perpendicular_lines_a_value (a : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (t * (-a-2) + (1-t) * (a-2), t * 1 + (1-t) * (-1))}
  let other_line : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (t + (-2), -2/3 * t + 1)}
  (∀ p ∈ l, ∀ q ∈ other_line, (p.1 - q.1) * (-2/3) + (p.2 - q.2) * 1 = 0) →
  a = -2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2891_289162


namespace NUMINAMATH_CALUDE_number_of_lilies_l2891_289196

theorem number_of_lilies (total_flowers other_flowers : ℕ) 
  (h1 : other_flowers = 120)
  (h2 : total_flowers = 160) :
  total_flowers - other_flowers = 40 := by
sorry

end NUMINAMATH_CALUDE_number_of_lilies_l2891_289196


namespace NUMINAMATH_CALUDE_square_plus_two_times_plus_one_equals_eleven_l2891_289154

theorem square_plus_two_times_plus_one_equals_eleven :
  let a : ℝ := Real.sqrt 11 - 1
  a^2 + 2*a + 1 = 11 := by
sorry

end NUMINAMATH_CALUDE_square_plus_two_times_plus_one_equals_eleven_l2891_289154


namespace NUMINAMATH_CALUDE_third_vertex_coordinates_l2891_289168

/-- Given a triangle with vertices (2, 3), (0, 0), and (x, 0) where x < 0,
    if the area of the triangle is 12 square units, then x = -8. -/
theorem third_vertex_coordinates (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * |3 * x| = 12 → x = -8 := by sorry

end NUMINAMATH_CALUDE_third_vertex_coordinates_l2891_289168


namespace NUMINAMATH_CALUDE_differential_equation_satisfied_l2891_289176

theorem differential_equation_satisfied 
  (x c : ℝ) 
  (y : ℝ → ℝ)
  (h1 : ∀ x, y x = 2 + c * Real.sqrt (1 - x^2))
  (h2 : Differentiable ℝ y) :
  (1 - x^2) * (deriv y x) + x * (y x) = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_differential_equation_satisfied_l2891_289176


namespace NUMINAMATH_CALUDE_pauline_garden_capacity_l2891_289170

/-- Represents Pauline's garden -/
structure Garden where
  rows : ℕ
  spaces_per_row : ℕ
  tomatoes : ℕ
  cucumbers : ℕ
  potatoes : ℕ

/-- Calculates the number of additional vegetables that can be planted in the garden -/
def additional_vegetables (g : Garden) : ℕ :=
  g.rows * g.spaces_per_row - (g.tomatoes + g.cucumbers + g.potatoes)

/-- Theorem stating the number of additional vegetables Pauline can plant -/
theorem pauline_garden_capacity :
  ∀ (g : Garden),
    g.rows = 10 ∧
    g.spaces_per_row = 15 ∧
    g.tomatoes = 15 ∧
    g.cucumbers = 20 ∧
    g.potatoes = 30 →
    additional_vegetables g = 85 := by
  sorry


end NUMINAMATH_CALUDE_pauline_garden_capacity_l2891_289170


namespace NUMINAMATH_CALUDE_track_completion_time_is_80_l2891_289101

/-- Represents a runner on the circular track -/
structure Runner :=
  (id : Nat)

/-- Represents a meeting between two runners -/
structure Meeting :=
  (runner1 : Runner)
  (runner2 : Runner)
  (time : ℕ)

/-- The circular track -/
def Track : Type := Unit

/-- Time for one runner to complete the track -/
def trackCompletionTime (track : Track) : ℕ := sorry

/-- Theorem stating the time to complete the track is 80 minutes -/
theorem track_completion_time_is_80 (track : Track) 
  (r1 r2 r3 : Runner)
  (m1 : Meeting)
  (m2 : Meeting)
  (m3 : Meeting)
  (h1 : m1.runner1 = r1 ∧ m1.runner2 = r2)
  (h2 : m2.runner1 = r2 ∧ m2.runner2 = r3)
  (h3 : m3.runner1 = r3 ∧ m3.runner2 = r1)
  (h4 : m2.time - m1.time = 15)
  (h5 : m3.time - m2.time = 25) :
  trackCompletionTime track = 80 := by sorry

end NUMINAMATH_CALUDE_track_completion_time_is_80_l2891_289101


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l2891_289134

theorem smallest_sum_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  ∀ a b : ℕ+, 
    a ≠ b → 
    (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
    (x : ℤ) + y ≤ (a : ℤ) + b :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l2891_289134


namespace NUMINAMATH_CALUDE_finite_decimal_fractions_l2891_289127

/-- A fraction a/b can be expressed as a finite decimal if and only if
    b in its simplest form is composed of only the prime factors 2 and 5 -/
def is_finite_decimal (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), b = 2^x * 5^y

/-- The set of natural numbers n for which both 1/n and 1/(n+1) are finite decimals -/
def S : Set ℕ := {n : ℕ | is_finite_decimal 1 n ∧ is_finite_decimal 1 (n+1)}

theorem finite_decimal_fractions : S = {1, 4} := by sorry

end NUMINAMATH_CALUDE_finite_decimal_fractions_l2891_289127


namespace NUMINAMATH_CALUDE_b_and_c_earnings_l2891_289114

/-- Given the daily earnings of three individuals a, b, and c, prove that b and c together earn $300 per day. -/
theorem b_and_c_earnings
  (total : ℝ)
  (a_and_c : ℝ)
  (c_earnings : ℝ)
  (h1 : total = 600)
  (h2 : a_and_c = 400)
  (h3 : c_earnings = 100) :
  total - a_and_c + c_earnings = 300 :=
by sorry

end NUMINAMATH_CALUDE_b_and_c_earnings_l2891_289114


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2891_289186

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  Complex.im (5 * i / (3 + 4 * i)) = 3 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2891_289186


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2891_289163

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 250) : x + y = 700 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2891_289163


namespace NUMINAMATH_CALUDE_initial_amount_proof_l2891_289183

/-- Proves that if an amount increases by 1/8th of itself each year for two years
    and becomes 64800, then the initial amount was 51200. -/
theorem initial_amount_proof (initial_amount : ℚ) : 
  (initial_amount * (9/8) * (9/8) = 64800) → initial_amount = 51200 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l2891_289183


namespace NUMINAMATH_CALUDE_john_house_nails_l2891_289121

/-- The number of nails needed for John's house walls -/
def total_nails (large_planks : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  large_planks * nails_per_plank + additional_nails

/-- Theorem: John needs 987 nails for his house walls -/
theorem john_house_nails :
  total_nails 27 36 15 = 987 := by
  sorry

end NUMINAMATH_CALUDE_john_house_nails_l2891_289121


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l2891_289180

/-- Given that 3/4 of 12 bananas are worth as much as 9 oranges,
    prove that 2/5 of 15 bananas are worth as much as 6 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (3/4 : ℚ) * 12 * banana_value = 9 * orange_value →
  (2/5 : ℚ) * 15 * banana_value = 6 * orange_value := by
sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l2891_289180


namespace NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l2891_289115

theorem cos_four_arccos_two_fifths :
  Real.cos (4 * Real.arccos (2 / 5)) = -47 / 625 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l2891_289115


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2891_289150

-- Define a function to calculate the sum of digits
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, 1 < d → d < n → ¬(d ∣ n)

-- Theorem statement
theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 1993 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2891_289150


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l2891_289135

theorem lcm_gcd_product (a b : ℕ) (ha : a = 8) (hb : b = 6) :
  Nat.lcm a b * Nat.gcd a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l2891_289135


namespace NUMINAMATH_CALUDE_a2_range_l2891_289110

def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem a2_range (a : ℕ → ℝ) 
  (h_mono : is_monotonically_increasing a)
  (h_a1 : a 1 = 2)
  (h_ineq : ∀ n : ℕ+, (n + 1 : ℝ) * a n ≥ n * a (2 * n)) :
  2 < a 2 ∧ a 2 ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_a2_range_l2891_289110


namespace NUMINAMATH_CALUDE_smallest_three_digit_ending_l2891_289143

def ends_same_three_digits (x : ℕ) : Prop :=
  x^2 % 1000 = x % 1000

theorem smallest_three_digit_ending : ∀ y > 1, ends_same_three_digits y → y ≥ 376 :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_ending_l2891_289143


namespace NUMINAMATH_CALUDE_parabola_translation_l2891_289171

/-- The translation of a parabola y = x^2 upwards by 3 units and to the left by 1 unit -/
theorem parabola_translation (x y : ℝ) :
  (y = x^2) →  -- Original parabola
  (y = (x + 1)^2 + 3) →  -- Resulting parabola after translation
  (∀ (x' y' : ℝ), y' = x'^2 → y' + 3 = ((x' + 1)^2 + 3)) -- Equivalence of the translation
  := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2891_289171


namespace NUMINAMATH_CALUDE_proposition_false_negation_true_l2891_289182

-- Define the properties of a quadrilateral
structure Quadrilateral :=
  (has_one_pair_parallel_sides : Bool)
  (has_one_pair_equal_sides : Bool)
  (is_parallelogram : Bool)

-- Define the proposition
def proposition (q : Quadrilateral) : Prop :=
  q.has_one_pair_parallel_sides ∧ q.has_one_pair_equal_sides → q.is_parallelogram

-- Define the negation of the proposition
def negation_proposition (q : Quadrilateral) : Prop :=
  q.has_one_pair_parallel_sides ∧ q.has_one_pair_equal_sides ∧ ¬q.is_parallelogram

-- Theorem stating that the proposition is false and its negation is true
theorem proposition_false_negation_true :
  (∃ q : Quadrilateral, ¬(proposition q)) ∧
  (∀ q : Quadrilateral, negation_proposition q → True) :=
sorry

end NUMINAMATH_CALUDE_proposition_false_negation_true_l2891_289182


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2891_289189

theorem inequality_system_solution (x : ℝ) :
  (x + 7) / 3 ≤ x + 3 ∧ 2 * (x + 1) < x + 3 → -1 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2891_289189


namespace NUMINAMATH_CALUDE_original_scissors_count_l2891_289144

theorem original_scissors_count (initial_scissors final_scissors added_scissors : ℕ) :
  final_scissors = initial_scissors + added_scissors →
  added_scissors = 13 →
  final_scissors = 52 →
  initial_scissors = 39 := by
  sorry

end NUMINAMATH_CALUDE_original_scissors_count_l2891_289144


namespace NUMINAMATH_CALUDE_student_survey_l2891_289141

theorem student_survey (french_english : ℕ) (french_not_english : ℕ) 
  (percent_not_french : ℚ) :
  french_english = 20 →
  french_not_english = 60 →
  percent_not_french = 60 / 100 →
  ∃ (total : ℕ), total = 200 ∧ 
    (french_english + french_not_english : ℚ) = (1 - percent_not_french) * total :=
by sorry

end NUMINAMATH_CALUDE_student_survey_l2891_289141


namespace NUMINAMATH_CALUDE_inequality_proof_l2891_289139

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) : 
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2891_289139


namespace NUMINAMATH_CALUDE_vector_b_value_l2891_289156

/-- Given two vectors a and b in ℝ², prove that b = (√2, √2) under the specified conditions. -/
theorem vector_b_value (a b : ℝ × ℝ) : 
  a = (1, 1) →                   -- a is (1,1)
  ‖b‖ = 2 →                      -- magnitude of b is 2
  ∃ (k : ℝ), b = k • a →         -- b is parallel to a
  k > 0 →                        -- a and b have the same direction
  b = (Real.sqrt 2, Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_vector_b_value_l2891_289156


namespace NUMINAMATH_CALUDE_managers_salary_l2891_289123

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 24 →
  avg_salary = 2400 →
  avg_increase = 100 →
  (num_employees * avg_salary + manager_salary) / (num_employees + 1) = avg_salary + avg_increase →
  manager_salary = 4900 :=
by
  sorry

#check managers_salary

end NUMINAMATH_CALUDE_managers_salary_l2891_289123


namespace NUMINAMATH_CALUDE_derivative_ln_2x_squared_minus_4_l2891_289140

open Real

theorem derivative_ln_2x_squared_minus_4 (x : ℝ) (h : x^2 ≠ 2) :
  deriv (λ x => log (2 * x^2 - 4)) x = 2 * x / (x^2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_derivative_ln_2x_squared_minus_4_l2891_289140


namespace NUMINAMATH_CALUDE_probability_divisible_by_45_is_zero_l2891_289178

def digits : List Nat := [1, 3, 3, 4, 5, 9]

def is_divisible_by_45 (n : Nat) : Prop :=
  n % 45 = 0

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length = 6 ∧ arr.toFinset = digits.toFinset

def to_number (arr : List Nat) : Nat :=
  arr.foldl (fun acc d => acc * 10 + d) 0

theorem probability_divisible_by_45_is_zero :
  ∀ arr : List Nat, is_valid_arrangement arr →
    ¬(is_divisible_by_45 (to_number arr)) :=
sorry

end NUMINAMATH_CALUDE_probability_divisible_by_45_is_zero_l2891_289178


namespace NUMINAMATH_CALUDE_monotone_increasing_sequence_condition_l2891_289116

-- Define the sequence a_n
def a (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

-- State the theorem
theorem monotone_increasing_sequence_condition (b : ℝ) :
  (∀ n : ℕ, a (n + 1) b > a n b) → b > -3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_sequence_condition_l2891_289116


namespace NUMINAMATH_CALUDE_exists_partition_count_2007_l2891_289195

/-- Given positive integers N and k, count_partitions N k returns the number of ways
    to write N as a sum of three integers a, b, and c, where 1 ≤ a, b, c ≤ k and the order matters. -/
def count_partitions (N k : ℕ+) : ℕ :=
  (Finset.filter (fun (a, b, c) => a + b + c = N ∧ a ≤ k ∧ b ≤ k ∧ c ≤ k)
    (Finset.product (Finset.range k) (Finset.product (Finset.range k) (Finset.range k)))).card

/-- There exist positive integers N and k such that the number of ways to write N
    as a sum of three integers a, b, and c, where 1 ≤ a, b, c ≤ k and the order matters, is 2007. -/
theorem exists_partition_count_2007 : ∃ (N k : ℕ+), count_partitions N k = 2007 := by
  sorry

end NUMINAMATH_CALUDE_exists_partition_count_2007_l2891_289195


namespace NUMINAMATH_CALUDE_zero_in_interval_l2891_289164

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem zero_in_interval :
  ∃ c : ℝ, c ∈ Set.Ioo 1 2 ∧ f c = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2891_289164


namespace NUMINAMATH_CALUDE_polka_dot_blankets_l2891_289155

theorem polka_dot_blankets (initial_blankets : ℕ) (added_blankets : ℕ) : 
  initial_blankets = 24 →
  added_blankets = 2 →
  (initial_blankets / 3 + added_blankets : ℕ) = 10 := by
sorry

end NUMINAMATH_CALUDE_polka_dot_blankets_l2891_289155


namespace NUMINAMATH_CALUDE_expression_simplification_l2891_289174

theorem expression_simplification (a : ℝ) (h : a = 2023) :
  (a / (a + 1) - 1 / (a + 1)) / ((a - 1) / (a^2 + 2*a + 1)) = 2024 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2891_289174


namespace NUMINAMATH_CALUDE_circle_radius_is_six_l2891_289124

/-- For a circle where the product of three inches and its circumference (in inches) 
    equals its area, the radius of the circle is 6 inches. -/
theorem circle_radius_is_six (r : ℝ) (h : 3 * (2 * Real.pi * r) = Real.pi * r^2) : r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_six_l2891_289124


namespace NUMINAMATH_CALUDE_circle_plus_four_three_l2891_289167

-- Define the operation ⊕
def circle_plus (a b : ℚ) : ℚ := a * (1 + a / b^2)

-- Theorem statement
theorem circle_plus_four_three : circle_plus 4 3 = 52 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_four_three_l2891_289167


namespace NUMINAMATH_CALUDE_quadratic_two_roots_k_range_l2891_289112

theorem quadratic_two_roots_k_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ - k = 0 ∧ x₂^2 + 2*x₂ - k = 0) → k > -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_k_range_l2891_289112


namespace NUMINAMATH_CALUDE_fill_time_both_pipes_l2891_289158

def pipe1_time : ℝ := 8
def pipe2_time : ℝ := 12

theorem fill_time_both_pipes :
  let rate1 := 1 / pipe1_time
  let rate2 := 1 / pipe2_time
  let combined_rate := rate1 + rate2
  (1 / combined_rate) = 4.8 := by sorry

end NUMINAMATH_CALUDE_fill_time_both_pipes_l2891_289158


namespace NUMINAMATH_CALUDE_total_sales_revenue_marie_sales_revenue_l2891_289113

/-- Calculates the total sales revenue from selling magazines and newspapers -/
theorem total_sales_revenue 
  (magazines_sold : ℕ) 
  (newspapers_sold : ℕ) 
  (magazine_price : ℚ) 
  (newspaper_price : ℚ) : ℚ :=
  magazines_sold * magazine_price + newspapers_sold * newspaper_price

/-- Proves that the total sales revenue for the given quantities and prices is correct -/
theorem marie_sales_revenue : 
  total_sales_revenue 425 275 (35/10) (5/4) = 1831.25 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_revenue_marie_sales_revenue_l2891_289113


namespace NUMINAMATH_CALUDE_salt_solution_volume_l2891_289198

/-- Given a mixture of pure water and a salt solution, calculates the volume of the salt solution needed to achieve a specific concentration. -/
theorem salt_solution_volume 
  (pure_water_volume : ℝ)
  (salt_solution_concentration : ℝ)
  (final_concentration : ℝ)
  (h1 : pure_water_volume = 1)
  (h2 : salt_solution_concentration = 0.75)
  (h3 : final_concentration = 0.15) :
  ∃ x : ℝ, x = 0.25 ∧ 
    salt_solution_concentration * x = final_concentration * (pure_water_volume + x) :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l2891_289198


namespace NUMINAMATH_CALUDE_pencil_box_cost_is_280_l2891_289125

/-- Represents the school's purchase of pencils and markers -/
structure SchoolPurchase where
  pencil_cartons : ℕ
  boxes_per_pencil_carton : ℕ
  marker_cartons : ℕ
  boxes_per_marker_carton : ℕ
  marker_carton_cost : ℚ
  total_spent : ℚ

/-- Calculates the cost of each box of pencils -/
def pencil_box_cost (purchase : SchoolPurchase) : ℚ :=
  (purchase.total_spent - purchase.marker_cartons * purchase.marker_carton_cost) /
  (purchase.pencil_cartons * purchase.boxes_per_pencil_carton)

/-- Theorem stating that for the given purchase, each box of pencils costs $2.80 -/
theorem pencil_box_cost_is_280 (purchase : SchoolPurchase) 
  (h1 : purchase.pencil_cartons = 20)
  (h2 : purchase.boxes_per_pencil_carton = 10)
  (h3 : purchase.marker_cartons = 10)
  (h4 : purchase.boxes_per_marker_carton = 5)
  (h5 : purchase.marker_carton_cost = 4)
  (h6 : purchase.total_spent = 600) :
  pencil_box_cost purchase = 280 / 100 := by
  sorry

end NUMINAMATH_CALUDE_pencil_box_cost_is_280_l2891_289125


namespace NUMINAMATH_CALUDE_compare_fractions_l2891_289103

theorem compare_fractions : -2/3 < -3/5 := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l2891_289103


namespace NUMINAMATH_CALUDE_min_value_of_f_l2891_289122

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2 - 5

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ (m = -5) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2891_289122
