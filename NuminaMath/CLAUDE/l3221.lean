import Mathlib

namespace NUMINAMATH_CALUDE_mode_is_tallest_rectangle_midpoint_l3221_322182

/-- Represents a frequency distribution histogram --/
structure FrequencyHistogram where
  -- Add necessary fields here
  -- This is a simplified representation

/-- The mode of a frequency distribution --/
def mode (h : FrequencyHistogram) : ℝ :=
  sorry

/-- The abscissa of the midpoint of the base of the tallest rectangle --/
def tallestRectangleMidpoint (h : FrequencyHistogram) : ℝ :=
  sorry

/-- Theorem stating that the abscissa of the midpoint of the base of the tallest rectangle
    represents the mode in a frequency distribution histogram --/
theorem mode_is_tallest_rectangle_midpoint (h : FrequencyHistogram) :
  mode h = tallestRectangleMidpoint h :=
sorry

end NUMINAMATH_CALUDE_mode_is_tallest_rectangle_midpoint_l3221_322182


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l3221_322195

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to x-axis -/
def symmetricPointXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Theorem: The symmetric point of P(-2, 3) with respect to x-axis is (-2, -3) -/
theorem symmetric_point_x_axis :
  let P : Point2D := { x := -2, y := 3 }
  symmetricPointXAxis P = { x := -2, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l3221_322195


namespace NUMINAMATH_CALUDE_player_field_time_l3221_322126

/-- Proves that each player in a 10-player team plays 36 minutes in a 45-minute match with 8 players always on field --/
theorem player_field_time (team_size : ℕ) (field_players : ℕ) (match_duration : ℕ) 
  (h1 : team_size = 10)
  (h2 : field_players = 8)
  (h3 : match_duration = 45) :
  (field_players * match_duration) / team_size = 36 := by
  sorry

end NUMINAMATH_CALUDE_player_field_time_l3221_322126


namespace NUMINAMATH_CALUDE_one_zero_one_zero_in_interval_l3221_322149

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- Theorem for the case of only one zero
theorem one_zero (a : ℝ) :
  (∃! x, f a x = 0) ↔ (a = 2 ∨ a = -2) :=
sorry

-- Theorem for the case of only one zero in (0, 1)
theorem one_zero_in_interval (a : ℝ) :
  (∃! x, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) ↔ a > 2 :=
sorry

end NUMINAMATH_CALUDE_one_zero_one_zero_in_interval_l3221_322149


namespace NUMINAMATH_CALUDE_convex_polygon_diagonal_triangles_l3221_322156

theorem convex_polygon_diagonal_triangles (n : ℕ) (h : n = 2002) : 
  ¬ ∃ (num_all_diagonal_triangles : ℕ),
    num_all_diagonal_triangles = (n - 2) / 2 ∧
    num_all_diagonal_triangles * 2 = n - 2 :=
by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_diagonal_triangles_l3221_322156


namespace NUMINAMATH_CALUDE_packaging_combinations_l3221_322133

/-- The number of varieties of wrapping paper -/
def wrapping_paper : ℕ := 10

/-- The number of colors of ribbon -/
def ribbon : ℕ := 5

/-- The number of types of gift cards -/
def gift_cards : ℕ := 4

/-- The number of options for decorative stickers -/
def stickers : ℕ := 2

/-- The total number of packaging combinations -/
def total_combinations : ℕ := wrapping_paper * ribbon * gift_cards * stickers

theorem packaging_combinations : total_combinations = 400 := by
  sorry

end NUMINAMATH_CALUDE_packaging_combinations_l3221_322133


namespace NUMINAMATH_CALUDE_no_identical_snakes_swallowing_l3221_322117

/-- A snake is represented as a set of points in a topological space. -/
def Snake (α : Type*) [TopologicalSpace α] := Set α

/-- Two snakes are identical if they are equal as sets. -/
def IdenticalSnakes {α : Type*} [TopologicalSpace α] (s1 s2 : Snake α) : Prop :=
  s1 = s2

/-- The process of one snake swallowing another from the tail. -/
def Swallow {α : Type*} [TopologicalSpace α] (s1 s2 : Snake α) : Prop :=
  ∃ t : Set α, t ⊆ s1 ∧ t = s2

/-- The theorem stating that it's impossible for two identical snakes to swallow each other from the tail. -/
theorem no_identical_snakes_swallowing {α : Type*} [TopologicalSpace α] (s1 s2 : Snake α) :
  IdenticalSnakes s1 s2 → ¬(Swallow s1 s2 ∧ Swallow s2 s1) :=
by
  sorry

end NUMINAMATH_CALUDE_no_identical_snakes_swallowing_l3221_322117


namespace NUMINAMATH_CALUDE_line_symmetry_l3221_322183

-- Define the original line
def original_line (x y : ℝ) : Prop := x * y + 1 = 0

-- Define the axis of symmetry
def symmetry_axis (x : ℝ) : Prop := x = 1

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Theorem statement
theorem line_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_line x₁ y₁ →
    symmetry_axis ((x₁ + x₂) / 2) →
    symmetric_line x₂ y₂ →
    y₁ = y₂ ∧ x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l3221_322183


namespace NUMINAMATH_CALUDE_gcd_108_45_l3221_322122

theorem gcd_108_45 : Nat.gcd 108 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_108_45_l3221_322122


namespace NUMINAMATH_CALUDE_cost_of_shirt_l3221_322148

/-- Given Sandy's shopping trip details, prove the cost of the shirt. -/
theorem cost_of_shirt (pants_cost change : ℚ) (bill : ℚ) : 
  pants_cost = 9.24 →
  change = 2.51 →
  bill = 20 →
  bill - change - pants_cost = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_shirt_l3221_322148


namespace NUMINAMATH_CALUDE_min_value_theorem_l3221_322120

theorem min_value_theorem (x y a b : ℝ) :
  x - y - 1 ≤ 0 →
  2 * x - y - 3 ≥ 0 →
  a > 0 →
  b > 0 →
  (∀ x' y', x' - y' - 1 ≤ 0 → 2 * x' - y' - 3 ≥ 0 → a * x' + b * y' ≥ a * x + b * y) →
  a * x + b * y = 3 →
  (∀ a' b', a' > 0 → b' > 0 → 2 / a' + 1 / b' ≥ 2 / a + 1 / b) →
  2 / a + 1 / b = 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3221_322120


namespace NUMINAMATH_CALUDE_profit_share_difference_is_370_l3221_322178

/-- Calculates the difference between Rose's and Tom's share in the profit --/
def profit_share_difference (john_investment : ℕ) (john_duration : ℕ) 
                            (rose_investment : ℕ) (rose_duration : ℕ) 
                            (tom_investment : ℕ) (tom_duration : ℕ) 
                            (total_profit : ℕ) : ℕ :=
  let john_investment_months := john_investment * john_duration
  let rose_investment_months := rose_investment * rose_duration
  let tom_investment_months := tom_investment * tom_duration
  let total_investment_months := john_investment_months + rose_investment_months + tom_investment_months
  let rose_share := (rose_investment_months * total_profit) / total_investment_months
  let tom_share := (tom_investment_months * total_profit) / total_investment_months
  rose_share - tom_share

/-- Theorem stating that the difference between Rose's and Tom's profit share is 370 --/
theorem profit_share_difference_is_370 :
  profit_share_difference 18000 12 12000 9 9000 8 4070 = 370 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_is_370_l3221_322178


namespace NUMINAMATH_CALUDE_odd_function_value_l3221_322184

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_neg : ∀ x < 0, f x = 1 / (x + 1)) : 
  f (1/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l3221_322184


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l3221_322189

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 15) :
  Nat.lcm a b * Nat.gcd a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l3221_322189


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l3221_322105

/-- Given a tetrahedron with face areas S₁, S₂, S₃, S₄, and inscribed sphere radius R,
    the volume V is (1/3)(S₁ + S₂ + S₃ + S₄)R -/
theorem tetrahedron_volume (S₁ S₂ S₃ S₄ R : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) (h₃ : S₃ > 0) (h₄ : S₄ > 0) (hR : R > 0) :
  ∃ V : ℝ, V = (1/3) * (S₁ + S₂ + S₃ + S₄) * R := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l3221_322105


namespace NUMINAMATH_CALUDE_soda_can_ounces_l3221_322103

/-- Represents the daily soda consumption in cans -/
def daily_soda_cans : ℕ := 5

/-- Represents the daily water consumption in ounces -/
def daily_water_oz : ℕ := 64

/-- Represents the weekly total fluid consumption in ounces -/
def weekly_total_oz : ℕ := 868

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Calculates the number of ounces in each can of soda -/
def ounces_per_soda_can : ℚ :=
  (weekly_total_oz - daily_water_oz * days_in_week) / (daily_soda_cans * days_in_week)

theorem soda_can_ounces :
  ounces_per_soda_can = 12 := by sorry

end NUMINAMATH_CALUDE_soda_can_ounces_l3221_322103


namespace NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l3221_322131

theorem max_value_of_4x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 16*x + 8*y + 20) : 
  4*x + 3*y ≤ 40 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l3221_322131


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3221_322191

/-- A coloring of the infinite grid -/
def GridColoring := ℤ × ℤ → Bool

/-- Existence of monochromatic rectangle in a grid coloring -/
theorem monochromatic_rectangle_exists (c : GridColoring) :
  ∃ (x₁ x₂ y₁ y₂ : ℤ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    c (x₁, y₁) = c (x₁, y₂) ∧
    c (x₁, y₁) = c (x₂, y₁) ∧
    c (x₁, y₁) = c (x₂, y₂) :=
  sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3221_322191


namespace NUMINAMATH_CALUDE_infinite_sum_equality_l3221_322166

/-- Given positive real numbers a and b with a > b, the infinite sum
    1/(b*a^2) + 1/(a^2*(2a^2 - b^2)) + 1/((2a^2 - b^2)*(3a^2 - 2b^2)) + 1/((3a^2 - 2b^2)*(4a^2 - 3b^2)) + ...
    is equal to 1 / ((a^2 - b^2) * b^2) -/
theorem infinite_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series := fun n : ℕ => 1 / ((n * a^2 - (n-1) * b^2) * ((n+1) * a^2 - n * b^2))
  ∑' n, series n = 1 / ((a^2 - b^2) * b^2) := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equality_l3221_322166


namespace NUMINAMATH_CALUDE_m_range_l3221_322101

/-- A function that represents f(x) = -x^2 - tx + 3t --/
def f (t : ℝ) (x : ℝ) : ℝ := -x^2 - t*x + 3*t

/-- Predicate that checks if f has only one zero in the interval (0, 2) --/
def has_one_zero_in_interval (t : ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 2 ∧ f t x = 0

/-- The sufficient but not necessary condition --/
def sufficient_condition (m : ℝ) : Prop :=
  ∀ t, 0 < t ∧ t < m → has_one_zero_in_interval t

/-- The theorem stating the range of m --/
theorem m_range :
  ∀ m, (m > 0 ∧ sufficient_condition m ∧ ¬(∀ t, has_one_zero_in_interval t → (0 < t ∧ t < m))) ↔
       (0 < m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3221_322101


namespace NUMINAMATH_CALUDE_common_divisors_sum_l3221_322152

theorem common_divisors_sum (numbers : List Int) (divisors : List Nat) : 
  numbers = [48, 144, -24, 180, 192] →
  divisors.length = 4 →
  (∀ d ∈ divisors, d > 0) →
  (∀ n ∈ numbers, ∀ d ∈ divisors, n % d = 0) →
  divisors.sum = 12 :=
by sorry

end NUMINAMATH_CALUDE_common_divisors_sum_l3221_322152


namespace NUMINAMATH_CALUDE_strawberry_basket_count_l3221_322108

theorem strawberry_basket_count (baskets : ℕ) (friends : ℕ) (total : ℕ) :
  baskets = 6 →
  friends = 3 →
  total = 1200 →
  ∃ (strawberries_per_basket : ℕ),
    strawberries_per_basket * baskets * (friends + 1) = total ∧
    strawberries_per_basket = 50 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_basket_count_l3221_322108


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_l3221_322170

theorem sufficient_not_necessary_contrapositive (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (¬q → p) ∧ ¬(p → ¬q) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_l3221_322170


namespace NUMINAMATH_CALUDE_quadratic_root_implies_b_value_l3221_322100

theorem quadratic_root_implies_b_value (b : ℝ) : 
  (Complex.I + 3 : ℂ) ^ 2 - 6 * (Complex.I + 3 : ℂ) + b = 0 → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_b_value_l3221_322100


namespace NUMINAMATH_CALUDE_greatest_integer_x_squared_less_than_25_l3221_322157

theorem greatest_integer_x_squared_less_than_25 :
  ∀ x : ℕ+, (x : ℝ) ^ 2 < 25 → x ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_x_squared_less_than_25_l3221_322157


namespace NUMINAMATH_CALUDE_even_function_decreasing_nonpositive_inequality_l3221_322138

/-- A function f is even if f(x) = f(-x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f is decreasing on (-∞, 0] if f(x₂) < f(x₁) for x₁ < x₂ ≤ 0 -/
def DecreasingOnNonPositive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₂ ≤ 0 → f x₂ < f x₁

theorem even_function_decreasing_nonpositive_inequality
  (f : ℝ → ℝ)
  (heven : EvenFunction f)
  (hdecr : DecreasingOnNonPositive f) :
  f 1 < f (-2) ∧ f (-2) < f (-3) :=
sorry

end NUMINAMATH_CALUDE_even_function_decreasing_nonpositive_inequality_l3221_322138


namespace NUMINAMATH_CALUDE_multiples_of_seven_l3221_322190

/-- The number of multiples of 7 between 200 and 500 -/
def c : ℕ := 
  (Nat.div 500 7 - Nat.div 200 7) + 1

theorem multiples_of_seven : c = 43 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_seven_l3221_322190


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3221_322129

/-- Given a geometric sequence {a_n} with positive terms, if 3a_1, (1/2)a_3, and 2a_2 form an arithmetic sequence,
    then (a_2016 + a_2017) / (a_2015 + a_2016) = 3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : 3 * a 1 + 2 * a 2 = a 3) :
  (a 2016 + a 2017) / (a 2015 + a 2016) = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3221_322129


namespace NUMINAMATH_CALUDE_f_2048_equals_121_l3221_322185

/-- A function satisfying the given property for positive integers -/
def special_function (f : ℕ → ℝ) : Prop :=
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a * b = 2^n → f a + f b = n^2

/-- The main theorem to prove -/
theorem f_2048_equals_121 (f : ℕ → ℝ) (h : special_function f) : f 2048 = 121 := by
  sorry

end NUMINAMATH_CALUDE_f_2048_equals_121_l3221_322185


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3221_322146

/-- Calculates the total amount owed after one year with simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + (principal * rate * time)

/-- Theorem: Given $35 principal and 4% simple annual interest, 
    the total amount owed after one year is $36.40 -/
theorem simple_interest_calculation :
  total_amount_owed 35 0.04 1 = 36.40 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3221_322146


namespace NUMINAMATH_CALUDE_sin_2alpha_over_cos_2beta_l3221_322155

theorem sin_2alpha_over_cos_2beta (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2) 
  (h2 : Real.tan (α - β) = 3) : 
  (Real.sin (2 * α)) / (Real.cos (2 * β)) = (Real.sqrt 5 + 3 * Real.sqrt 2) / 20 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_over_cos_2beta_l3221_322155


namespace NUMINAMATH_CALUDE_oranges_eaten_l3221_322168

/-- Given 96 oranges, with half of them ripe, and 1/4 of ripe oranges and 1/8 of unripe oranges eaten, 
    the total number of eaten oranges is 18. -/
theorem oranges_eaten (total : ℕ) (ripe_fraction : ℚ) (ripe_eaten_fraction : ℚ) (unripe_eaten_fraction : ℚ) :
  total = 96 →
  ripe_fraction = 1/2 →
  ripe_eaten_fraction = 1/4 →
  unripe_eaten_fraction = 1/8 →
  (ripe_fraction * total * ripe_eaten_fraction + (1 - ripe_fraction) * total * unripe_eaten_fraction : ℚ) = 18 := by
sorry

end NUMINAMATH_CALUDE_oranges_eaten_l3221_322168


namespace NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l3221_322187

/-- The percentage of chromium in the new alloy formed by melting two alloys -/
theorem chromium_percentage_in_new_alloy
  (chromium_percent1 : ℝ) (chromium_percent2 : ℝ)
  (weight1 : ℝ) (weight2 : ℝ)
  (h1 : chromium_percent1 = 12)
  (h2 : chromium_percent2 = 8)
  (h3 : weight1 = 15)
  (h4 : weight2 = 35) :
  let total_chromium := (chromium_percent1 / 100) * weight1 + (chromium_percent2 / 100) * weight2
  let total_weight := weight1 + weight2
  (total_chromium / total_weight) * 100 = 9.2 := by
sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l3221_322187


namespace NUMINAMATH_CALUDE_min_value_of_square_plus_constant_l3221_322141

theorem min_value_of_square_plus_constant (x : ℝ) :
  (x - 1)^2 + 3 ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_square_plus_constant_l3221_322141


namespace NUMINAMATH_CALUDE_simplify_expression_l3221_322160

theorem simplify_expression (x : ℝ) : (2*x)^5 - (3*x^2)*(x^3) = 29*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3221_322160


namespace NUMINAMATH_CALUDE_specific_sandwich_calories_l3221_322112

/-- Represents a sandwich with bacon strips -/
structure BaconSandwich where
  bacon_strips : ℕ
  calories_per_strip : ℕ
  bacon_percentage : ℚ

/-- Calculates the total calories of a bacon sandwich -/
def total_calories (s : BaconSandwich) : ℚ :=
  (s.bacon_strips * s.calories_per_strip : ℚ) / s.bacon_percentage

/-- Theorem stating the total calories of the specific sandwich -/
theorem specific_sandwich_calories :
  let s : BaconSandwich := {
    bacon_strips := 2,
    calories_per_strip := 125,
    bacon_percentage := 1/5
  }
  total_calories s = 1250 := by sorry

end NUMINAMATH_CALUDE_specific_sandwich_calories_l3221_322112


namespace NUMINAMATH_CALUDE_complex_division_equality_l3221_322192

theorem complex_division_equality : (1 - 2 * Complex.I) / (2 + Complex.I) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equality_l3221_322192


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_product_l3221_322107

theorem consecutive_integers_sum_product (x : ℤ) : 
  (x + (x + 1) + (x + 2) = 27) → (x * (x + 1) * (x + 2) = 720) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_product_l3221_322107


namespace NUMINAMATH_CALUDE_table_cost_l3221_322121

theorem table_cost (chair_cost : ℚ → ℚ) (total_cost : ℚ → ℚ) : 
  (∀ t, chair_cost t = t / 7) →
  (∀ t, total_cost t = t + 4 * (chair_cost t)) →
  (∃ t, total_cost t = 220) →
  (∃ t, t = 140 ∧ total_cost t = 220) :=
by sorry

end NUMINAMATH_CALUDE_table_cost_l3221_322121


namespace NUMINAMATH_CALUDE_shorter_book_pages_l3221_322161

theorem shorter_book_pages (x y : ℕ) (h1 : y = x + 10) (h2 : y / 2 = x) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_shorter_book_pages_l3221_322161


namespace NUMINAMATH_CALUDE_correct_assignment_calculation_symbol_l3221_322137

/-- Enum representing different flowchart symbols -/
inductive FlowchartSymbol
  | StartEnd
  | Decision
  | AssignmentCalculation
  | InputOutput

/-- Function that returns the correct symbol for assignment and calculation -/
def assignmentCalculationSymbol : FlowchartSymbol := FlowchartSymbol.AssignmentCalculation

/-- Theorem stating that the assignment and calculation symbol is correct -/
theorem correct_assignment_calculation_symbol :
  assignmentCalculationSymbol = FlowchartSymbol.AssignmentCalculation := by
  sorry

#check correct_assignment_calculation_symbol

end NUMINAMATH_CALUDE_correct_assignment_calculation_symbol_l3221_322137


namespace NUMINAMATH_CALUDE_tile_ratio_after_modification_l3221_322154

/-- Represents a square tile pattern -/
structure TilePattern where
  side : Nat
  black_tiles : Nat
  white_tiles : Nat

/-- Extends the tile pattern with a double border and replaces middle row and column -/
def extend_pattern (p : TilePattern) : TilePattern :=
  { side := p.side + 4,
    black_tiles := p.black_tiles + (p.side + 1) + (p.side + 2)^2 - p.side^2,
    white_tiles := p.white_tiles + (p.side + 4)^2 - (p.side + 2)^2 }

/-- The main theorem to prove -/
theorem tile_ratio_after_modification (p : TilePattern) 
  (h1 : p.side = 7)
  (h2 : p.black_tiles = 18) 
  (h3 : p.white_tiles = 39) : 
  let extended := extend_pattern p
  (extended.black_tiles : Rat) / extended.white_tiles = 63 / 79 := by
  sorry

end NUMINAMATH_CALUDE_tile_ratio_after_modification_l3221_322154


namespace NUMINAMATH_CALUDE_circle_transformation_l3221_322177

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given distance -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem -/
theorem circle_transformation (C : ℝ × ℝ) (h : C = (3, -4)) :
  (translate_right (reflect_x C) 5) = (8, 4) := by sorry

end NUMINAMATH_CALUDE_circle_transformation_l3221_322177


namespace NUMINAMATH_CALUDE_office_employees_count_l3221_322174

theorem office_employees_count (men women : ℕ) : 
  men = women →
  6 = women / 5 →
  men + women = 60 :=
by sorry

end NUMINAMATH_CALUDE_office_employees_count_l3221_322174


namespace NUMINAMATH_CALUDE_other_communities_count_l3221_322143

theorem other_communities_count (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 850 →
  muslim_percent = 44/100 →
  hindu_percent = 14/100 →
  sikh_percent = 10/100 →
  (total_boys : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 272 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l3221_322143


namespace NUMINAMATH_CALUDE_pigs_count_l3221_322167

/-- The number of pigs initially in the barn -/
def initial_pigs : ℕ := 64

/-- The number of pigs that joined -/
def joined_pigs : ℕ := 22

/-- The total number of pigs after joining -/
def total_pigs : ℕ := 86

/-- Theorem stating that the initial number of pigs plus the joined pigs equals the total pigs -/
theorem pigs_count : initial_pigs + joined_pigs = total_pigs := by
  sorry

end NUMINAMATH_CALUDE_pigs_count_l3221_322167


namespace NUMINAMATH_CALUDE_a5_value_l3221_322123

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem a5_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 + a 7 = 3 →
  a 3 * a 7 = 2 →
  a 5 = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_a5_value_l3221_322123


namespace NUMINAMATH_CALUDE_jovana_shells_total_l3221_322145

/-- The total amount of shells in Jovana's bucket -/
def total_shells (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that given 5 pounds initial and 12 pounds additional shells, the total is 17 pounds -/
theorem jovana_shells_total :
  total_shells 5 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_jovana_shells_total_l3221_322145


namespace NUMINAMATH_CALUDE_sqrt_problem_quadratic_equation_problem_l3221_322116

-- Problem 1
theorem sqrt_problem :
  Real.sqrt 12 * Real.sqrt 75 - Real.sqrt 8 + Real.sqrt 2 = 30 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem quadratic_equation_problem (x : ℝ) :
  (1 / 9 : ℝ) * (3 * x - 2)^2 - 4 = 0 ↔ x = 8/3 ∨ x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_problem_quadratic_equation_problem_l3221_322116


namespace NUMINAMATH_CALUDE_line_point_y_coordinate_l3221_322106

/-- Given a line passing through points (3, -1, 0) and (8, -4, -5),
    the y-coordinate of a point on this line with z-coordinate -3 is -14/5 -/
theorem line_point_y_coordinate :
  let p₁ : ℝ × ℝ × ℝ := (3, -1, 0)
  let p₂ : ℝ × ℝ × ℝ := (8, -4, -5)
  let line := {p : ℝ × ℝ × ℝ | ∃ t : ℝ, p = p₁ + t • (p₂ - p₁)}
  ∃ p : ℝ × ℝ × ℝ, p ∈ line ∧ p.2.2 = -3 ∧ p.2.1 = -14/5 :=
by sorry

end NUMINAMATH_CALUDE_line_point_y_coordinate_l3221_322106


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_f_l3221_322176

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem sum_of_max_and_min_f : 
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
               (∀ x, m ≤ f x) ∧ (∃ x, f x = m) ∧
               (M + m = 2) :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_f_l3221_322176


namespace NUMINAMATH_CALUDE_f_squared_properties_l3221_322164

-- Define a real-valued function f with period T
def f (T : ℝ) (h : T > 0) : ℝ → ℝ := sorry

-- Define the property of f being periodic with period T
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Define the property of f being monotonic on an interval
def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Main theorem
theorem f_squared_properties (T : ℝ) (h : T > 0) :
  (is_periodic (f T h) T) →
  (is_monotonic_on (f T h) 0 T) →
  (¬ ∃ P, is_periodic (fun x ↦ f T h (x^2)) P) ∧
  (is_monotonic_on (fun x ↦ f T h (x^2)) 0 (Real.sqrt T)) := by
  sorry

end NUMINAMATH_CALUDE_f_squared_properties_l3221_322164


namespace NUMINAMATH_CALUDE_lawrence_walking_distance_l3221_322130

/-- Calculates the total distance walked given the daily distance and number of days -/
def totalDistanceWalked (dailyDistance : ℝ) (days : ℝ) : ℝ :=
  dailyDistance * days

/-- Proves that walking 4.0 km a day for 3.0 days results in a total distance of 12.0 km -/
theorem lawrence_walking_distance :
  totalDistanceWalked 4.0 3.0 = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_walking_distance_l3221_322130


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l3221_322169

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x : ℝ, 2 * x^2 + 2 * x - 1 ≤ 0) ↔ (∀ x : ℝ, 2 * x^2 + 2 * x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l3221_322169


namespace NUMINAMATH_CALUDE_min_angle_between_planes_l3221_322119

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Represents a line in 3D space -/
structure Line where
  direction : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Given two planes, compute the angle between them -/
def angle_between_planes (p1 p2 : Plane) : ℝ := sorry

/-- Check if a plane is perpendicular to a line -/
def plane_perpendicular_to_line (p : Plane) (l : Line) : Prop := sorry

/-- Check if a plane is parallel to a line -/
def plane_parallel_to_line (p : Plane) (l : Line) : Prop := sorry

/-- Extract the line A₁C₁ from a cube -/
def line_A₁C₁ (c : Cube) : Line := sorry

/-- Extract the line CD₁ from a cube -/
def line_CD₁ (c : Cube) : Line := sorry

theorem min_angle_between_planes (c : Cube) (α β : Plane) 
  (h1 : plane_perpendicular_to_line α (line_A₁C₁ c))
  (h2 : plane_parallel_to_line β (line_CD₁ c)) :
  ∃ (θ : ℝ), (∀ (α' β' : Plane), 
    plane_perpendicular_to_line α' (line_A₁C₁ c) →
    plane_parallel_to_line β' (line_CD₁ c) →
    angle_between_planes α' β' ≥ θ) ∧
  θ = π / 6 := by sorry

end NUMINAMATH_CALUDE_min_angle_between_planes_l3221_322119


namespace NUMINAMATH_CALUDE_retail_price_l3221_322128

/-- The retail price of a product given its cost price and percentage increase -/
theorem retail_price (a : ℝ) (percent_increase : ℝ) (h : percent_increase = 30) :
  a + (percent_increase / 100) * a = 1.3 * a := by
  sorry

end NUMINAMATH_CALUDE_retail_price_l3221_322128


namespace NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_l3221_322165

/-- The dividend polynomial -/
def dividend (b x : ℚ) : ℚ := 8 * x^3 + 5 * x^2 + b * x - 8

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 3 * x^2 - 2 * x + 4

/-- The remainder when dividend is divided by divisor -/
def remainder (b x : ℚ) : ℚ := dividend b x - divisor x * ((8/3) * x + 2/3)

theorem constant_remainder_iff_b_eq (b : ℚ) : 
  (∃ (c : ℚ), ∀ (x : ℚ), remainder b x = c) ↔ b = -98/9 := by sorry

end NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_l3221_322165


namespace NUMINAMATH_CALUDE_square_equals_cube_of_digit_sum_l3221_322127

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem square_equals_cube_of_digit_sum (n : ℕ) :
  n ∈ Finset.range 1000 →
  (n^2 = (sum_of_digits n)^3) ↔ (n = 1 ∨ n = 27) := by sorry

end NUMINAMATH_CALUDE_square_equals_cube_of_digit_sum_l3221_322127


namespace NUMINAMATH_CALUDE_table_tennis_lineups_l3221_322147

/-- Represents a team in the table tennis competition -/
structure Team :=
  (members : Finset Nat)
  (size : members.card = 5)

/-- Represents a lineup for the competition -/
structure Lineup :=
  (singles1 : Nat)
  (singles2 : Nat)
  (doubles1 : Nat)
  (doubles2 : Nat)
  (all_different : singles1 ≠ singles2 ∧ singles1 ≠ doubles1 ∧ singles1 ≠ doubles2 ∧ 
                   singles2 ≠ doubles1 ∧ singles2 ≠ doubles2 ∧ doubles1 ≠ doubles2)

/-- The theorem to be proved -/
theorem table_tennis_lineups (t : Team) (a : Nat) (h : a ∈ t.members) : 
  (∃ l : Finset Lineup, l.card = 60 ∧ ∀ lineup ∈ l, lineup.singles1 ∈ t.members ∧ 
                                                    lineup.singles2 ∈ t.members ∧ 
                                                    lineup.doubles1 ∈ t.members ∧ 
                                                    lineup.doubles2 ∈ t.members) ∧
  (∃ l : Finset Lineup, l.card = 36 ∧ ∀ lineup ∈ l, lineup.singles1 ∈ t.members ∧ 
                                                    lineup.singles2 ∈ t.members ∧ 
                                                    lineup.doubles1 ∈ t.members ∧ 
                                                    lineup.doubles2 ∈ t.members ∧ 
                                                    (lineup.doubles1 ≠ a ∧ lineup.doubles2 ≠ a)) :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_lineups_l3221_322147


namespace NUMINAMATH_CALUDE_exists_function_and_constant_l3221_322188

theorem exists_function_and_constant : 
  ∃ (f : ℝ → ℝ) (a : ℝ), 
    a ∈ Set.Icc 0 π ∧ 
    (∀ x : ℝ, (1 + Real.sqrt 2 * Real.sin x) * (1 + Real.sqrt 2 * Real.sin (x + π)) = Real.cos (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_exists_function_and_constant_l3221_322188


namespace NUMINAMATH_CALUDE_pot_filling_time_l3221_322104

-- Define the constants
def drops_per_minute : ℕ := 3
def ml_per_drop : ℕ := 20
def pot_capacity_liters : ℕ := 3

-- Define the theorem
theorem pot_filling_time :
  (pot_capacity_liters * 1000) / (drops_per_minute * ml_per_drop) = 50 := by
  sorry

end NUMINAMATH_CALUDE_pot_filling_time_l3221_322104


namespace NUMINAMATH_CALUDE_gcd_10010_15015_l3221_322115

theorem gcd_10010_15015 : Nat.gcd 10010 15015 = 5005 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10010_15015_l3221_322115


namespace NUMINAMATH_CALUDE_distinct_collections_biology_l3221_322118

def Word : Type := List Char

def isVowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U', 'Y']

def isConsonant (c : Char) : Bool :=
  c ∉ ['A', 'E', 'I', 'O', 'U', 'Y']

def biology : Word := ['B', 'I', 'O', 'L', 'O', 'G', 'Y']

def indistinguishable (c : Char) : Bool :=
  c ∈ ['B', 'I', 'G']

def Collection := List Char

def isValidCollection (c : Collection) : Bool :=
  c.length = 6 ∧ 
  (c.filter isVowel).length = 3 ∧
  (c.filter isConsonant).length = 3

def distinctCollections (w : Word) : Finset Collection :=
  sorry

theorem distinct_collections_biology :
  (distinctCollections biology).card = 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_biology_l3221_322118


namespace NUMINAMATH_CALUDE_min_value_theorem_l3221_322109

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation -/
def line_eq (a b x y : ℝ) : Prop := 2*a*x - b*y + 2 = 0

/-- The center of the circle satisfies the circle equation -/
def center_satisfies_circle (x₀ y₀ : ℝ) : Prop := circle_eq x₀ y₀

/-- The line passes through the center of the circle -/
def line_passes_through_center (a b x₀ y₀ : ℝ) : Prop := line_eq a b x₀ y₀

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (x₀ y₀ : ℝ) (h_center : center_satisfies_circle x₀ y₀) 
  (h_line : line_passes_through_center a b x₀ y₀) : 
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), 1 / a₀ + 1 / b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3221_322109


namespace NUMINAMATH_CALUDE_divisors_of_1200_l3221_322102

theorem divisors_of_1200 : Finset.card (Nat.divisors 1200) = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_1200_l3221_322102


namespace NUMINAMATH_CALUDE_correct_change_l3221_322151

-- Define the given conditions
def apples_bought : ℝ := 6
def cost_per_kg : ℝ := 2.2
def money_given : ℝ := 50

-- Define the function to calculate the change
def calculate_change (apples : ℝ) (cost : ℝ) (given : ℝ) : ℝ :=
  given - (apples * cost)

-- Theorem statement
theorem correct_change :
  calculate_change apples_bought cost_per_kg money_given = 36.8 := by
  sorry


end NUMINAMATH_CALUDE_correct_change_l3221_322151


namespace NUMINAMATH_CALUDE_base_ten_to_base_seven_l3221_322171

theorem base_ten_to_base_seven : 
  ∃ (a b c d : ℕ), 
    947 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a = 2 ∧ b = 5 ∧ c = 2 ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_ten_to_base_seven_l3221_322171


namespace NUMINAMATH_CALUDE_sugar_consumption_change_l3221_322193

theorem sugar_consumption_change 
  (original_price : ℝ) 
  (original_consumption : ℝ) 
  (price_increase_percentage : ℝ) 
  (expenditure_increase_percentage : ℝ) 
  (h1 : original_consumption = 30)
  (h2 : price_increase_percentage = 0.32)
  (h3 : expenditure_increase_percentage = 0.10) : 
  ∃ new_consumption : ℝ, 
    new_consumption = 25 ∧ 
    (1 + expenditure_increase_percentage) * (original_consumption * original_price) = 
    new_consumption * ((1 + price_increase_percentage) * original_price) :=
by sorry

end NUMINAMATH_CALUDE_sugar_consumption_change_l3221_322193


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3221_322199

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 5

/-- The axis of symmetry -/
def axis_of_symmetry : ℝ := 1

/-- Theorem: The axis of symmetry of the parabola y = x^2 - 2x + 5 is the line x = 1 -/
theorem parabola_axis_of_symmetry :
  ∀ x : ℝ, parabola (axis_of_symmetry + x) = parabola (axis_of_symmetry - x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3221_322199


namespace NUMINAMATH_CALUDE_order_of_exponentials_l3221_322134

theorem order_of_exponentials :
  let a : ℝ := Real.rpow 0.6 4.2
  let b : ℝ := Real.rpow 0.7 4.2
  let c : ℝ := Real.rpow 0.6 5.1
  b > a ∧ a > c :=
by sorry

end NUMINAMATH_CALUDE_order_of_exponentials_l3221_322134


namespace NUMINAMATH_CALUDE_village_population_l3221_322172

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.1) * (1 - 0.2) = 3168 → P = 4400 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3221_322172


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l3221_322124

theorem circle_diameter_ratio (C D : ℝ → Prop) (rC rD : ℝ) : 
  (∀ x, C x → D x) →  -- C is within D
  rD = 10 →  -- Diameter of D is 20 cm
  (π * (rD^2 - rC^2)) / (π * rC^2) = 2 →  -- Ratio of shaded area to area of C is 2:1
  2 * rC = 20 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l3221_322124


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_l3221_322150

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem stating that x^2 = 0 is a quadratic equation -/
theorem x_squared_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_x_squared_is_quadratic_l3221_322150


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l3221_322142

/-- Proves that arctan(tan 75° - 2 tan 30°) = 75° --/
theorem arctan_tan_difference : 
  Real.arctan (Real.tan (75 * π / 180) - 2 * Real.tan (30 * π / 180)) = 75 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l3221_322142


namespace NUMINAMATH_CALUDE_expected_rainfall_theorem_l3221_322110

-- Define the daily weather probabilities and rainfall amounts
def sunny_prob : ℝ := 0.30
def light_rain_prob : ℝ := 0.20
def heavy_rain_prob : ℝ := 0.25
def cloudy_prob : ℝ := 0.25

def light_rain_amount : ℝ := 5
def heavy_rain_amount : ℝ := 7

def days : ℕ := 7

-- State the theorem
theorem expected_rainfall_theorem :
  let daily_expected_rainfall := 
    light_rain_prob * light_rain_amount + heavy_rain_prob * heavy_rain_amount
  (days : ℝ) * daily_expected_rainfall = 19.25 := by
  sorry


end NUMINAMATH_CALUDE_expected_rainfall_theorem_l3221_322110


namespace NUMINAMATH_CALUDE_car_distance_calculation_l3221_322159

/-- Proves that the distance covered by a car traveling at 99 km/h for 5 hours is 495 km -/
theorem car_distance_calculation (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 99)
  (h2 : time = 5)
  (h3 : distance = speed * time) : 
  distance = 495 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_calculation_l3221_322159


namespace NUMINAMATH_CALUDE_sphere_volume_containing_cube_l3221_322197

theorem sphere_volume_containing_cube (edge_length : ℝ) (h : edge_length = 2) :
  let diagonal := edge_length * Real.sqrt 3
  let radius := diagonal / 2
  let volume := (4 / 3) * Real.pi * radius ^ 3
  volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_containing_cube_l3221_322197


namespace NUMINAMATH_CALUDE_integer_power_sum_l3221_322135

theorem integer_power_sum (x : ℝ) (h : ∃ (k : ℤ), x + 1/x = k) :
  ∀ (n : ℕ), ∃ (m : ℤ), x^n + 1/(x^n) = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l3221_322135


namespace NUMINAMATH_CALUDE_derivative_at_one_l3221_322194

/-- Given a function f(x) = (x-1)^3 + 3(x-1), prove that its derivative at x=1 is 3. -/
theorem derivative_at_one (f : ℝ → ℝ) (h : f = λ x ↦ (x - 1)^3 + 3*(x - 1)) :
  deriv f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3221_322194


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3221_322111

theorem imaginary_part_of_complex_fraction : Complex.im ((3 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3221_322111


namespace NUMINAMATH_CALUDE_tribe_leadership_choices_l3221_322180

/-- The number of ways to choose leadership in a tribe --/
def choose_leadership (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (Nat.choose (n - 3) 2) * (Nat.choose (n - 5) 2)

/-- Theorem: For a tribe of 10 members with the given leadership structure, 
    there are 151200 ways to choose the leadership --/
theorem tribe_leadership_choices :
  choose_leadership 10 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_tribe_leadership_choices_l3221_322180


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l3221_322132

theorem smallest_n_for_candy_purchase : ∃ (n : ℕ+), 
  (∀ (k : ℕ+), 24 * k = Nat.lcm (Nat.lcm 10 15) 18 → n ≤ k) ∧ 
  24 * n = Nat.lcm (Nat.lcm 10 15) 18 ∧ 
  n = 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l3221_322132


namespace NUMINAMATH_CALUDE_set_P_definition_l3221_322173

def U : Set ℕ := {1, 2, 3, 4, 5}
def C_UP : Set ℕ := {4, 5}

theorem set_P_definition : 
  ∃ P : Set ℕ, P = U \ C_UP ∧ P = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_set_P_definition_l3221_322173


namespace NUMINAMATH_CALUDE_valid_paths_count_l3221_322144

/-- Represents a face on the dodecahedron -/
inductive Face
| Top
| Bottom
| TopRing (n : Fin 5)
| BottomRing (n : Fin 5)

/-- Represents a valid path on the dodecahedron -/
def ValidPath : List Face → Prop :=
  sorry

/-- The specific face on the bottom ring that must be passed through -/
def SpecificBottomFace : Face :=
  Face.BottomRing 0

/-- A function that counts the number of valid paths -/
def CountValidPaths : ℕ :=
  sorry

/-- Theorem stating that the number of valid paths is 15 -/
theorem valid_paths_count :
  CountValidPaths = 15 :=
sorry

end NUMINAMATH_CALUDE_valid_paths_count_l3221_322144


namespace NUMINAMATH_CALUDE_cryptic_message_solution_l3221_322175

/-- Represents a digit in the cryptic message --/
structure Digit (d : ℕ) where
  value : ℕ
  is_valid : value < d

/-- Represents the cryptic message as an addition problem --/
def is_valid_solution (d : ℕ) (D E P O N : Digit d) : Prop :=
  let deep := D.value * d^3 + E.value * d^2 + E.value * d + P.value
  let pond := P.value * d^3 + O.value * d^2 + N.value * d + D.value
  let done := D.value * d^3 + O.value * d^2 + N.value * d + E.value
  (deep + pond + deep) % d^4 = done

/-- The main theorem stating the existence of a solution --/
theorem cryptic_message_solution :
  ∃ (d : ℕ) (D E P O N : Digit d),
    d = 10 ∧
    is_valid_solution d D E P O N ∧
    D.value ≠ E.value ∧ D.value ≠ P.value ∧ D.value ≠ O.value ∧ D.value ≠ N.value ∧
    E.value ≠ P.value ∧ E.value ≠ O.value ∧ E.value ≠ N.value ∧
    P.value ≠ O.value ∧ P.value ≠ N.value ∧
    O.value ≠ N.value ∧
    D.value = 3 ∧ E.value = 2 ∧ P.value = 3 ∧ O.value = 6 ∧ N.value = 2 :=
sorry

end NUMINAMATH_CALUDE_cryptic_message_solution_l3221_322175


namespace NUMINAMATH_CALUDE_recycling_efficiency_l3221_322163

/-- The number of pounds Vanessa recycled -/
def vanessa_pounds : ℕ := 20

/-- The number of pounds Vanessa's friends recycled -/
def friends_pounds : ℕ := 16

/-- The total number of points earned -/
def total_points : ℕ := 4

/-- The number of pounds needed to earn one point -/
def pounds_per_point : ℚ := (vanessa_pounds + friends_pounds) / total_points

theorem recycling_efficiency : pounds_per_point = 9 := by sorry

end NUMINAMATH_CALUDE_recycling_efficiency_l3221_322163


namespace NUMINAMATH_CALUDE_quadratic_coefficient_count_l3221_322162

theorem quadratic_coefficient_count : ∀ n : ℤ, 
  (∃ p q : ℤ, p * q = 30 ∧ p + q = n) → 
  (∃ S : Finset ℤ, S.card = 8 ∧ n ∈ S ∧ ∀ m : ℤ, m ∈ S ↔ ∃ p q : ℤ, p * q = 30 ∧ p + q = m) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_count_l3221_322162


namespace NUMINAMATH_CALUDE_grapes_purchased_l3221_322125

/-- Given the cost of grapes, amount of mangoes, cost of mangoes, and total paid,
    calculate the amount of grapes purchased. -/
theorem grapes_purchased 
  (grape_cost : ℕ) 
  (mango_amount : ℕ) 
  (mango_cost : ℕ) 
  (total_paid : ℕ) : 
  grape_cost = 70 →
  mango_amount = 9 →
  mango_cost = 50 →
  total_paid = 1010 →
  (total_paid - mango_amount * mango_cost) / grape_cost = 8 :=
by
  sorry

#check grapes_purchased

end NUMINAMATH_CALUDE_grapes_purchased_l3221_322125


namespace NUMINAMATH_CALUDE_factorization_problem_l3221_322139

theorem factorization_problem (A B : ℤ) : 
  (∀ y : ℝ, 20 * y^2 - 103 * y + 42 = (A * y - 21) * (B * y - 2)) →
  A * B + A = 30 := by
sorry

end NUMINAMATH_CALUDE_factorization_problem_l3221_322139


namespace NUMINAMATH_CALUDE_orchard_tree_difference_l3221_322140

theorem orchard_tree_difference (original_trees dead_trees slightly_damaged_trees : ℕ) 
  (h1 : original_trees = 150)
  (h2 : dead_trees = 92)
  (h3 : slightly_damaged_trees = 15) :
  dead_trees - (original_trees - (dead_trees + slightly_damaged_trees)) = 49 :=
by sorry

end NUMINAMATH_CALUDE_orchard_tree_difference_l3221_322140


namespace NUMINAMATH_CALUDE_product_equals_difference_of_powers_l3221_322158

theorem product_equals_difference_of_powers : 
  (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) * 
  (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64) = 4^128 - 3^128 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_difference_of_powers_l3221_322158


namespace NUMINAMATH_CALUDE_x_range_for_positive_f_l3221_322196

/-- The function f(x) for a given a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

/-- The theorem stating the range of x given the conditions -/
theorem x_range_for_positive_f :
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, ∀ x, f a x > 0) →
  (∀ x, x < 1 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_positive_f_l3221_322196


namespace NUMINAMATH_CALUDE_island_distance_l3221_322186

/-- Given two islands A and B that are 10 nautical miles apart, with an angle of view from A to C and B of 60°, and an angle of view from B to A and C of 75°, the distance between islands B and C is 5√6 nautical miles. -/
theorem island_distance (A B C : ℝ × ℝ) : 
  let distance := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let angle := λ p q r : ℝ × ℝ => Real.arccos ((distance p q)^2 + (distance p r)^2 - (distance q r)^2) / (2 * distance p q * distance p r)
  distance A B = 10 →
  angle A C B = π / 3 →
  angle B A C = 5 * π / 12 →
  distance B C = 5 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_island_distance_l3221_322186


namespace NUMINAMATH_CALUDE_triangle_side_from_median_l3221_322136

theorem triangle_side_from_median (a b k : ℝ) (ha : 0 < a) (hb : 0 < b) (hk : 0 < k) :
  ∃ c : ℝ, c > 0 ∧ c = Real.sqrt ((2 * (a^2 + b^2 - 2 * k^2)) / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_from_median_l3221_322136


namespace NUMINAMATH_CALUDE_f_unique_non_monotonic_range_l3221_322113

/-- A quadratic function f(x) with specific properties -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

/-- The minimum value of f(x) is 1 -/
axiom min_value : ∃ (x : ℝ), f x = 1 ∧ ∀ (y : ℝ), f y ≥ f x

/-- f(0) = f(2) = 3 -/
axiom f_values : f 0 = 3 ∧ f 2 = 3

/-- Theorem: f(x) is the unique quadratic function satisfying the given conditions -/
theorem f_unique : ∀ (g : ℝ → ℝ), (∃ (a b c : ℝ), ∀ (x : ℝ), g x = a * x^2 + b * x + c) →
  (∃ (x : ℝ), g x = 1 ∧ ∀ (y : ℝ), g y ≥ g x) →
  (g 0 = 3 ∧ g 2 = 3) →
  (∀ (x : ℝ), g x = f x) :=
sorry

/-- Theorem: The range of a for which f(x) is not monotonic in [2a, a + 1] is 0 < a < 0.5 -/
theorem non_monotonic_range : ∀ (a : ℝ), 
  (∃ (x y : ℝ), 2 * a ≤ x ∧ x < y ∧ y ≤ a + 1 ∧ f x > f y) ∧
  (∃ (x y : ℝ), 2 * a ≤ x ∧ x < y ∧ y ≤ a + 1 ∧ f x < f y) ↔
  (0 < a ∧ a < 0.5) :=
sorry

end NUMINAMATH_CALUDE_f_unique_non_monotonic_range_l3221_322113


namespace NUMINAMATH_CALUDE_sin_4_arcsin_l3221_322153

theorem sin_4_arcsin (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.sin (4 * Real.arcsin x) = 4 * x * (1 - 2 * x^2) * Real.sqrt (1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_4_arcsin_l3221_322153


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l3221_322198

theorem line_slope_intercept_product (m b : ℚ) : 
  m = 3/4 → b = 2 → m * b > 1 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l3221_322198


namespace NUMINAMATH_CALUDE_school_population_l3221_322179

theorem school_population (total_girls : ℕ) (difference : ℕ) (total_boys : ℕ) : 
  total_girls = 697 → difference = 228 → total_girls - total_boys = difference → total_boys = 469 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l3221_322179


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3221_322181

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (30 * π / 180 + α) = 3/5)
  (h2 : 60 * π / 180 < α)
  (h3 : α < 150 * π / 180) :
  Real.cos α = (3 - 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3221_322181


namespace NUMINAMATH_CALUDE_open_box_volume_is_5208_l3221_322114

/-- Calculates the volume of an open box created from a metallic sheet --/
def open_box_volume (sheet_length : ℝ) (sheet_width : ℝ) (thickness : ℝ) (cut_size : ℝ) : ℝ :=
  let internal_length := sheet_length - 2 * cut_size - 2 * thickness
  let internal_width := sheet_width - 2 * cut_size - 2 * thickness
  let height := cut_size
  internal_length * internal_width * height

/-- Theorem stating that the volume of the open box is 5208 m³ --/
theorem open_box_volume_is_5208 :
  open_box_volume 48 38 0.5 8 = 5208 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_is_5208_l3221_322114
