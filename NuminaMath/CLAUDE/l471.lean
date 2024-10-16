import Mathlib

namespace NUMINAMATH_CALUDE_parabola_point_x_coord_l471_47174

/-- A point on a parabola with a specific distance from the focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 4 * y
  dist_to_focus : Real.sqrt ((x - 0)^2 + (y - 1)^2) = 5

/-- Theorem stating that the x-coordinate of the point is ±4 -/
theorem parabola_point_x_coord (P : ParabolaPoint) : P.x = 4 ∨ P.x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_x_coord_l471_47174


namespace NUMINAMATH_CALUDE_min_value_of_y_l471_47103

theorem min_value_of_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (hrel : b = (1 - a) / 3) :
  ∃ (y_min : ℝ), y_min = 2 * Real.sqrt 3 ∧ ∀ (y : ℝ), y = 3^a + 27^b → y ≥ y_min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_y_l471_47103


namespace NUMINAMATH_CALUDE_f_is_quadratic_l471_47147

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l471_47147


namespace NUMINAMATH_CALUDE_intersection_line_circle_l471_47127

theorem intersection_line_circle (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 - A.2 + 3 = 0 ∧ (A.1 - 1)^2 + (A.2 - 2)^2 = 4) ∧
    (a * B.1 - B.2 + 3 = 0 ∧ (B.1 - 1)^2 + (B.2 - 2)^2 = 4) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l471_47127


namespace NUMINAMATH_CALUDE_clothing_store_inventory_l471_47167

theorem clothing_store_inventory (belts : ℕ) (black_shirts : ℕ) (white_shirts : ℕ) :
  belts = 40 →
  black_shirts = 63 →
  white_shirts = 42 →
  ∃ (ties : ℕ) (scarves : ℕ) (jeans : ℕ),
    jeans = (2 * (black_shirts + white_shirts)) / 3 ∧
    scarves = (ties + belts) / 2 ∧
    jeans = scarves + 33 ∧
    ties = 34 :=
by sorry

end NUMINAMATH_CALUDE_clothing_store_inventory_l471_47167


namespace NUMINAMATH_CALUDE_x_2000_value_l471_47136

def sequence_property (x : ℕ → ℝ) :=
  ∀ n, x (n + 1) + x (n + 2) + x (n + 3) = 20

theorem x_2000_value (x : ℕ → ℝ) 
  (h1 : sequence_property x) 
  (h2 : x 4 = 9) 
  (h3 : x 12 = 7) : 
  x 2000 = 4 := by
sorry

end NUMINAMATH_CALUDE_x_2000_value_l471_47136


namespace NUMINAMATH_CALUDE_min_workers_for_profit_l471_47135

/-- Represents the problem of finding the minimum number of workers needed for profit --/
theorem min_workers_for_profit (
  maintenance_cost : ℝ)
  (worker_hourly_wage : ℝ)
  (widgets_per_hour : ℝ)
  (widget_price : ℝ)
  (work_hours : ℝ)
  (h1 : maintenance_cost = 600)
  (h2 : worker_hourly_wage = 20)
  (h3 : widgets_per_hour = 6)
  (h4 : widget_price = 3.5)
  (h5 : work_hours = 8)
  : ∃ n : ℕ, n = 76 ∧ ∀ m : ℕ, m < n → maintenance_cost + m * worker_hourly_wage * work_hours ≥ m * widgets_per_hour * widget_price * work_hours :=
sorry

end NUMINAMATH_CALUDE_min_workers_for_profit_l471_47135


namespace NUMINAMATH_CALUDE_tamara_garden_walkway_area_l471_47124

/-- Represents the dimensions of a flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Represents the garden layout -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed : FlowerBed
  walkwayWidth : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkwayArea (g : Garden) : ℝ :=
  let totalWidth := g.columns * g.bed.length + (g.columns + 1) * g.walkwayWidth
  let totalHeight := g.rows * g.bed.width + (g.rows + 1) * g.walkwayWidth
  let totalArea := totalWidth * totalHeight
  let bedArea := g.rows * g.columns * g.bed.length * g.bed.width
  totalArea - bedArea

/-- Theorem stating that the walkway area for Tamara's garden is 214 square feet -/
theorem tamara_garden_walkway_area :
  let g : Garden := {
    rows := 3,
    columns := 2,
    bed := { length := 7, width := 3 },
    walkwayWidth := 2
  }
  walkwayArea g = 214 := by
  sorry

end NUMINAMATH_CALUDE_tamara_garden_walkway_area_l471_47124


namespace NUMINAMATH_CALUDE_remaining_oranges_l471_47107

def initial_oranges : ℕ := 60
def oranges_taken : ℕ := 35

theorem remaining_oranges :
  initial_oranges - oranges_taken = 25 := by
  sorry

end NUMINAMATH_CALUDE_remaining_oranges_l471_47107


namespace NUMINAMATH_CALUDE_celebration_attendees_l471_47122

theorem celebration_attendees (men : ℕ) (women : ℕ) : 
  men = 15 →
  men * 4 = women * 3 →
  women = 20 :=
by sorry

end NUMINAMATH_CALUDE_celebration_attendees_l471_47122


namespace NUMINAMATH_CALUDE_ball_count_proof_l471_47180

/-- Theorem: Given a bag of balls with specific color counts and probability,
    prove the total number of balls. -/
theorem ball_count_proof
  (white green yellow red purple : ℕ)
  (prob_not_red_or_purple : ℚ)
  (h1 : white = 50)
  (h2 : green = 20)
  (h3 : yellow = 10)
  (h4 : red = 17)
  (h5 : purple = 3)
  (h6 : prob_not_red_or_purple = 4/5) :
  white + green + yellow + red + purple = 100 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l471_47180


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l471_47117

/-- A geometric sequence with three consecutive terms x, 2x+2, and 3x+3 has x = -4 -/
theorem geometric_sequence_solution (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*x + 2) = x * r ∧ (3*x + 3) = (2*x + 2) * r) → x = -4 :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_solution_l471_47117


namespace NUMINAMATH_CALUDE_marble_count_l471_47172

theorem marble_count (yellow : ℕ) (blue : ℕ) (red : ℕ) : 
  yellow = 5 →
  blue * 4 = red * 3 →
  red = yellow + 3 →
  yellow + blue + red = 19 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l471_47172


namespace NUMINAMATH_CALUDE_intersection_point_solution_l471_47151

/-- Given two lines y = x + b and y = ax + 2 that intersect at point (3, -1),
    prove that the solution to (a - 1)x = b - 2 is x = 3. -/
theorem intersection_point_solution (a b : ℝ) :
  (3 + b = 3 * a + 2) →  -- Intersection point condition
  (-1 = 3 + b) →         -- y-coordinate of intersection point
  ((a - 1) * 3 = b - 2)  -- Solution x = 3 satisfies the equation
  := by sorry

end NUMINAMATH_CALUDE_intersection_point_solution_l471_47151


namespace NUMINAMATH_CALUDE_sin_double_angle_special_l471_47156

/-- Given an angle θ with specific properties, prove that sin(2θ) = -√3/2 -/
theorem sin_double_angle_special (θ : Real) : 
  (∃ (x y : Real), x > 0 ∧ y = -Real.sqrt 3 * x ∧ 
    Real.cos θ = x / Real.sqrt (x^2 + y^2) ∧
    Real.sin θ = y / Real.sqrt (x^2 + y^2)) →
  Real.sin (2 * θ) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_special_l471_47156


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_l471_47178

theorem gcf_lcm_sum_36_56 : Nat.gcd 36 56 + Nat.lcm 36 56 = 508 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_l471_47178


namespace NUMINAMATH_CALUDE_sum_and_product_to_a_plus_b_l471_47194

theorem sum_and_product_to_a_plus_b (a b : ℝ) 
  (sum_eq : (a + Real.sqrt b) + (a - Real.sqrt b) = -8)
  (product_eq : (a + Real.sqrt b) * (a - Real.sqrt b) = 4) :
  a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_to_a_plus_b_l471_47194


namespace NUMINAMATH_CALUDE_smallest_positive_term_l471_47131

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := n * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem smallest_positive_term (d : ℚ) :
  let a := arithmetic_sequence (-12) d
  let S := arithmetic_sum (-12) d
  S 13 = 0 →
  (∀ k < 8, a k ≤ 0) ∧ a 8 > 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_term_l471_47131


namespace NUMINAMATH_CALUDE_hexagon_largest_angle_l471_47152

theorem hexagon_largest_angle :
  ∀ (a b c d e f : ℝ),
    -- The angles are consecutive integers
    (∃ (x : ℝ), a = x - 2 ∧ b = x - 1 ∧ c = x ∧ d = x + 1 ∧ e = x + 2 ∧ f = x + 3) →
    -- Sum of angles in a hexagon is 720°
    a + b + c + d + e + f = 720 →
    -- The largest angle is 122.5°
    max a (max b (max c (max d (max e f)))) = 122.5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_largest_angle_l471_47152


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l471_47195

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value (f : ℝ → ℝ) :
  MonicQuarticPolynomial f →
  f (-2) = -4 →
  f 1 = -1 →
  f 3 = -9 →
  f (-4) = -16 →
  f 2 = -28 := by
    sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l471_47195


namespace NUMINAMATH_CALUDE_triangle_area_solution_l471_47183

theorem triangle_area_solution (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * (2*x) * x = 50 → x = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_solution_l471_47183


namespace NUMINAMATH_CALUDE_combined_tennis_percentage_l471_47168

theorem combined_tennis_percentage
  (north_total : ℕ)
  (south_total : ℕ)
  (north_tennis_percent : ℚ)
  (south_tennis_percent : ℚ)
  (h1 : north_total = 1800)
  (h2 : south_total = 2700)
  (h3 : north_tennis_percent = 25 / 100)
  (h4 : south_tennis_percent = 35 / 100)
  : (north_total * north_tennis_percent + south_total * south_tennis_percent) / (north_total + south_total) = 31 / 100 :=
by sorry

end NUMINAMATH_CALUDE_combined_tennis_percentage_l471_47168


namespace NUMINAMATH_CALUDE_expression_simplification_l471_47153

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((x + 3) / x - 1) / ((x^2 - 1) / (x^2 + x)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l471_47153


namespace NUMINAMATH_CALUDE_cos_sin_difference_l471_47193

theorem cos_sin_difference (α β : Real) 
  (h : Real.cos (α + β) * Real.cos (α - β) = 1/3) : 
  Real.cos α ^ 2 - Real.sin β ^ 2 = 1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_difference_l471_47193


namespace NUMINAMATH_CALUDE_linda_needs_four_more_batches_l471_47141

/-- Represents the number of cookies in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of classmates Linda has -/
def classmates : ℕ := 36

/-- Represents the number of cookies Linda wants to give each classmate -/
def cookies_per_classmate : ℕ := 12

/-- Represents the number of dozens of cookies made by one batch of chocolate chip cookies -/
def choc_chip_dozens_per_batch : ℕ := 3

/-- Represents the number of dozens of cookies made by one batch of oatmeal raisin cookies -/
def oatmeal_dozens_per_batch : ℕ := 4

/-- Represents the number of dozens of cookies made by one batch of peanut butter cookies -/
def pb_dozens_per_batch : ℕ := 5

/-- Represents the number of batches of chocolate chip cookies Linda made -/
def choc_chip_batches : ℕ := 3

/-- Represents the number of batches of oatmeal raisin cookies Linda made -/
def oatmeal_batches : ℕ := 2

/-- Calculates the total number of cookies needed -/
def total_cookies_needed : ℕ := classmates * cookies_per_classmate

/-- Calculates the number of cookies already made -/
def cookies_already_made : ℕ := 
  (choc_chip_batches * choc_chip_dozens_per_batch * dozen) + 
  (oatmeal_batches * oatmeal_dozens_per_batch * dozen)

/-- Calculates the number of cookies still needed -/
def cookies_still_needed : ℕ := total_cookies_needed - cookies_already_made

/-- Represents the number of cookies made by one batch of peanut butter cookies -/
def cookies_per_pb_batch : ℕ := pb_dozens_per_batch * dozen

/-- Theorem stating that Linda needs to bake 4 more batches of peanut butter cookies -/
theorem linda_needs_four_more_batches : 
  (cookies_still_needed + cookies_per_pb_batch - 1) / cookies_per_pb_batch = 4 := by
  sorry

end NUMINAMATH_CALUDE_linda_needs_four_more_batches_l471_47141


namespace NUMINAMATH_CALUDE_dhoni_leftover_earnings_l471_47118

theorem dhoni_leftover_earnings (rent dishwasher bills car groceries leftover : ℚ) : 
  rent = 20/100 →
  dishwasher = 15/100 →
  bills = 10/100 →
  car = 8/100 →
  groceries = 12/100 →
  leftover = 1 - (rent + dishwasher + bills + car + groceries) →
  leftover = 35/100 := by
sorry

end NUMINAMATH_CALUDE_dhoni_leftover_earnings_l471_47118


namespace NUMINAMATH_CALUDE_bag_emptying_probability_l471_47162

/-- Represents the probability of emptying a bag of card pairs -/
def emptyBagProbability (n : ℕ) : ℚ :=
  if n < 2 then 0
  else if n = 2 then 1
  else (3 : ℚ) / (2 * n - 1) * emptyBagProbability (n - 1)

/-- The probability of forming a pair when drawing 3 cards from 12 cards -/
def pairFormationProbability : ℚ := 3 / 11

theorem bag_emptying_probability :
  emptyBagProbability 6 = 9 / 385 :=
sorry

end NUMINAMATH_CALUDE_bag_emptying_probability_l471_47162


namespace NUMINAMATH_CALUDE_maria_bottles_l471_47113

/-- The number of bottles Maria has at the end, given her initial number of bottles,
    the number she drinks, and the number she buys. -/
def final_bottles (initial : ℕ) (drunk : ℕ) (bought : ℕ) : ℕ :=
  initial - drunk + bought

/-- Theorem stating that Maria ends up with 51 bottles given the problem conditions -/
theorem maria_bottles : final_bottles 14 8 45 = 51 := by
  sorry

end NUMINAMATH_CALUDE_maria_bottles_l471_47113


namespace NUMINAMATH_CALUDE_baker_cakes_l471_47133

/-- The number of cakes Baker made initially -/
def initial_cakes : ℕ := sorry

/-- The number of cakes Baker's friend bought -/
def friend_bought : ℕ := 140

/-- The number of cakes Baker still has -/
def remaining_cakes : ℕ := 15

/-- Theorem stating that the initial number of cakes is 155 -/
theorem baker_cakes : initial_cakes = friend_bought + remaining_cakes := by sorry

end NUMINAMATH_CALUDE_baker_cakes_l471_47133


namespace NUMINAMATH_CALUDE_average_temperature_is_42_4_l471_47126

/-- The average daily low temperature in Addington from September 15th to 19th, 2008 -/
def average_temperature : ℚ :=
  let temperatures : List ℤ := [40, 47, 45, 41, 39]
  (temperatures.sum : ℚ) / temperatures.length

/-- Theorem stating that the average temperature is 42.4°F -/
theorem average_temperature_is_42_4 : 
  average_temperature = 424/10 := by sorry

end NUMINAMATH_CALUDE_average_temperature_is_42_4_l471_47126


namespace NUMINAMATH_CALUDE_tile_covers_25_squares_l471_47187

/-- Represents a square tile -/
structure Tile :=
  (sideLength : ℝ)

/-- Represents a checkerboard -/
structure Checkerboard :=
  (size : ℕ)
  (squareWidth : ℝ)

/-- Counts the number of squares completely covered by a tile on a checkerboard -/
def countCoveredSquares (t : Tile) (c : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating that a square tile with side length D placed on a 10x10 checkerboard
    with square width D, such that their centers coincide, covers exactly 25 squares -/
theorem tile_covers_25_squares (D : ℝ) (D_pos : D > 0) :
  let t : Tile := { sideLength := D }
  let c : Checkerboard := { size := 10, squareWidth := D }
  countCoveredSquares t c = 25 :=
sorry

end NUMINAMATH_CALUDE_tile_covers_25_squares_l471_47187


namespace NUMINAMATH_CALUDE_cubic_derivative_problem_l471_47106

/-- Given a cubic function f(x) = ax³ + bx² + 3 where b is a constant,
    if f'(1) = -5, then f'(2) = -4 -/
theorem cubic_derivative_problem (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + 3
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + 2 * b * x
  f' 1 = -5 → f' 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_derivative_problem_l471_47106


namespace NUMINAMATH_CALUDE_f_properties_l471_47109

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (2^x - 1)

theorem f_properties (a : ℝ) :
  (∀ x : ℝ, f a x ≠ 0 ↔ x ≠ 0) ∧
  (∀ x : ℝ, f a (-x) = -(f a x) ↔ a = 1/2) ∧
  (a = 1/2 → ∀ x : ℝ, x ≠ 0 → x^3 * f a x > 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l471_47109


namespace NUMINAMATH_CALUDE_unique_divisor_l471_47148

def is_valid_divisor (d : ℕ) : Prop :=
  ∃ (sequence : Finset ℕ),
    (sequence.card = 8) ∧
    (∀ n ∈ sequence, 29 ≤ n ∧ n ≤ 119) ∧
    (∀ n ∈ sequence, n % d = 0)

theorem unique_divisor :
  ∃! d : ℕ, is_valid_divisor d ∧ d = 13 := by sorry

end NUMINAMATH_CALUDE_unique_divisor_l471_47148


namespace NUMINAMATH_CALUDE_sheila_attend_probability_l471_47159

/-- The probability of rain tomorrow -/
def prob_rain : ℝ := 0.4

/-- The probability Sheila will go if it rains -/
def prob_go_if_rain : ℝ := 0.2

/-- The probability Sheila will go if it's sunny -/
def prob_go_if_sunny : ℝ := 0.8

/-- The probability that Sheila will attend the picnic -/
def prob_sheila_attend : ℝ := prob_rain * prob_go_if_rain + (1 - prob_rain) * prob_go_if_sunny

theorem sheila_attend_probability :
  prob_sheila_attend = 0.56 := by sorry

end NUMINAMATH_CALUDE_sheila_attend_probability_l471_47159


namespace NUMINAMATH_CALUDE_john_water_savings_l471_47157

def water_savings (old_flush_volume : ℝ) (flushes_per_day : ℕ) (water_reduction_percentage : ℝ) (days_in_month : ℕ) : ℝ :=
  let old_daily_usage := old_flush_volume * flushes_per_day
  let old_monthly_usage := old_daily_usage * days_in_month
  let new_flush_volume := old_flush_volume * (1 - water_reduction_percentage)
  let new_daily_usage := new_flush_volume * flushes_per_day
  let new_monthly_usage := new_daily_usage * days_in_month
  old_monthly_usage - new_monthly_usage

theorem john_water_savings :
  water_savings 5 15 0.8 30 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_john_water_savings_l471_47157


namespace NUMINAMATH_CALUDE_system_solution_l471_47166

theorem system_solution : 
  ∃ (x y : ℚ), (3 * x - 4 * y = -7) ∧ (4 * x - 3 * y = 5) ∧ (x = 41/7) ∧ (y = 43/7) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l471_47166


namespace NUMINAMATH_CALUDE_blue_faces_cube_l471_47119

theorem blue_faces_cube (n : ℕ) : n > 0 →
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 := by sorry

end NUMINAMATH_CALUDE_blue_faces_cube_l471_47119


namespace NUMINAMATH_CALUDE_no_multiples_of_2310_in_power_difference_form_l471_47139

theorem no_multiples_of_2310_in_power_difference_form :
  ¬ ∃ (k i j : ℕ), 
    0 ≤ i ∧ i < j ∧ j ≤ 50 ∧ 
    k * 2310 = 2^j - 2^i ∧ 
    k > 0 :=
by sorry

end NUMINAMATH_CALUDE_no_multiples_of_2310_in_power_difference_form_l471_47139


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l471_47102

/-- The set of all real numbers x satisfying |x-5|+|x+1|<8 is equal to the open interval (-2, 6). -/
theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 5| + |x + 1| < 8} = Set.Ioo (-2 : ℝ) 6 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l471_47102


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l471_47125

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l471_47125


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l471_47171

/-- Proves that a train of given length and speed takes 30 seconds to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : Real)
  (train_speed_kmh : Real)
  (bridge_length : Real)
  (h1 : train_length = 140)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 235) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry


end NUMINAMATH_CALUDE_train_bridge_crossing_time_l471_47171


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l471_47111

theorem point_in_fourth_quadrant (a : ℝ) (h : a > 1) :
  let P : ℝ × ℝ := (1 + a, 1 - a)
  P.1 > 0 ∧ P.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l471_47111


namespace NUMINAMATH_CALUDE_real_part_w_cubed_l471_47197

open Complex

/-- Given a complex number w with positive imaginary part, |w| = 5, and the triangle
    with vertices w, w², and w³ has a right angle at w, prove that the real part of w³ is -73. -/
theorem real_part_w_cubed (w : ℂ) 
  (h1 : w.im > 0) 
  (h2 : Complex.abs w = 5) 
  (h3 : (w^2 - w) • (w^3 - w) = 0) : 
  (w^3).re = -73 := by
  sorry

end NUMINAMATH_CALUDE_real_part_w_cubed_l471_47197


namespace NUMINAMATH_CALUDE_correct_road_determination_l471_47134

/-- Represents the two tribes on the island -/
inductive Tribe
| TruthTeller
| Liar

/-- Represents the possible roads -/
inductive Road
| ToVillage
| AwayFromVillage

/-- Represents the possible answers to a question -/
inductive Answer
| Yes
| No

/-- The actual state of the road -/
def actual_road : Road := sorry

/-- The tribe of the islander being asked -/
def islander_tribe : Tribe := sorry

/-- Function that determines how a member of a given tribe would answer a direct question about the road -/
def direct_answer (t : Tribe) (r : Road) : Answer := sorry

/-- Function that determines how an islander would answer the traveler's question -/
def islander_answer (t : Tribe) (r : Road) : Answer := sorry

/-- The traveler's interpretation of the islander's answer -/
def traveler_interpretation (a : Answer) : Road := sorry

theorem correct_road_determination :
  traveler_interpretation (islander_answer islander_tribe actual_road) = actual_road := by sorry

end NUMINAMATH_CALUDE_correct_road_determination_l471_47134


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l471_47143

theorem sum_of_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ 
  (i + 2*i^2 + 3*i^3 + 4*i^4 + 5*i^5 + 6*i^6 + 7*i^7 + 8*i^8 + 9*i^9 = 4 + 5*i) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l471_47143


namespace NUMINAMATH_CALUDE_balloon_min_volume_l471_47100

/-- Represents the relationship between pressure and volume of a gas in a balloon -/
noncomputable def pressure (k : ℝ) (V : ℝ) : ℝ := k / V

theorem balloon_min_volume (k : ℝ) :
  (pressure k 3 = 8000) →
  (∀ V, V ≥ 0.6 → pressure k V ≤ 40000) ∧
  (∀ ε > 0, ∃ V, 0.6 - ε < V ∧ V < 0.6 ∧ pressure k V > 40000) :=
by sorry

end NUMINAMATH_CALUDE_balloon_min_volume_l471_47100


namespace NUMINAMATH_CALUDE_typist_margin_width_l471_47104

/-- Proves that for a 20x30 cm sheet with 3 cm margins on top and bottom,
    if 64% is used for typing, the side margins are 2 cm wide. -/
theorem typist_margin_width (x : ℝ) : 
  x > 0 →                             -- side margin is positive
  x < 10 →                            -- side margin is less than half the sheet width
  (20 - 2*x) * 24 = 0.64 * 600 →      -- 64% of sheet is used for typing
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_typist_margin_width_l471_47104


namespace NUMINAMATH_CALUDE_min_value_theorem_l471_47144

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 4 ∧
  (a^2 + 1 / (b * (a - b)) = 4 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l471_47144


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l471_47138

/-- An ellipse passing through (2,3) with foci at (-2,0) and (2,0) has eccentricity 1/2 -/
theorem ellipse_eccentricity : ∀ (e : ℝ), 
  (∃ (a b : ℝ), 
    (2/a)^2 + (3/b)^2 = 1 ∧  -- ellipse passes through (2,3)
    a > b ∧ b > 0 ∧         -- standard form constraints
    4 = a^2 - b^2) →        -- distance between foci is 4
  e = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l471_47138


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l471_47130

theorem complex_square_one_plus_i : 
  (Complex.I + 1) ^ 2 = 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l471_47130


namespace NUMINAMATH_CALUDE_average_price_approximately_85_85_l471_47137

/-- Represents a bookstore with its purchase details -/
structure Bookstore where
  books : ℕ
  total : ℚ
  discount : ℚ
  tax : ℚ
  specialDeal : ℚ

/-- Calculates the effective price per book for a given bookstore -/
def effectivePrice (store : Bookstore) : ℚ :=
  sorry

/-- The list of bookstores Rahim visited -/
def bookstores : List Bookstore := [
  { books := 25, total := 1600, discount := 15/100, tax := 5/100, specialDeal := 0 },
  { books := 35, total := 3200, discount := 0, tax := 0, specialDeal := 8/35 },
  { books := 40, total := 3800, discount := 1/100, tax := 7/100, specialDeal := 0 },
  { books := 30, total := 2400, discount := 1/60, tax := 6/100, specialDeal := 0 },
  { books := 20, total := 1800, discount := 8/100, tax := 4/100, specialDeal := 0 }
]

/-- Calculates the average price per book across all bookstores -/
def averagePrice (stores : List Bookstore) : ℚ :=
  sorry

theorem average_price_approximately_85_85 :
  abs (averagePrice bookstores - 85.85) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_price_approximately_85_85_l471_47137


namespace NUMINAMATH_CALUDE_inequality_properties_l471_47115

-- Define the inequality and its solution set
def inequality (a x : ℝ) : Prop := a * (x - 1) * (x - 3) + 2 > 0

def solution_set (x₁ x₂ : ℝ) : Set ℝ :=
  {x | x < x₁ ∨ x > x₂}

-- State the theorem
theorem inequality_properties
  (a x₁ x₂ : ℝ)
  (h_solution : ∀ x, inequality a x ↔ x ∈ solution_set x₁ x₂)
  (h_order : x₁ < x₂) :
  (x₁ + x₂ = 4) ∧
  (3 < x₁ * x₂ ∧ x₁ * x₂ < 4) ∧
  (∀ x, (3*a + 2) * x^2 - 4*a*x + a < 0 ↔ 1/x₂ < x ∧ x < 1/x₁) :=
by sorry

end NUMINAMATH_CALUDE_inequality_properties_l471_47115


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l471_47128

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs x > 1
def q (x : ℝ) : Prop := x < -2

-- State the theorem
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(p x) ∧ q x) :=
sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l471_47128


namespace NUMINAMATH_CALUDE_cost_of_bacon_bacon_cost_is_ten_l471_47186

/-- The cost of bacon given Joan's shopping scenario -/
theorem cost_of_bacon (total_budget : ℕ) (hummus_cost : ℕ) (hummus_quantity : ℕ)
  (chicken_cost : ℕ) (vegetable_cost : ℕ) (apple_cost : ℕ) (apple_quantity : ℕ) : ℕ :=
  by
  -- Define the conditions
  have h1 : total_budget = 60 := by sorry
  have h2 : hummus_cost = 5 := by sorry
  have h3 : hummus_quantity = 2 := by sorry
  have h4 : chicken_cost = 20 := by sorry
  have h5 : vegetable_cost = 10 := by sorry
  have h6 : apple_cost = 2 := by sorry
  have h7 : apple_quantity = 5 := by sorry

  -- Prove that the cost of bacon is 10
  sorry

/-- The main theorem stating that the cost of bacon is 10 -/
theorem bacon_cost_is_ten : cost_of_bacon 60 5 2 20 10 2 5 = 10 := by sorry

end NUMINAMATH_CALUDE_cost_of_bacon_bacon_cost_is_ten_l471_47186


namespace NUMINAMATH_CALUDE_sum_after_operations_l471_47192

/-- Given two numbers x and y whose sum is T, prove that if 5 is added to each number
    and then each resulting number is tripled, the sum of the final two numbers is 3T + 30. -/
theorem sum_after_operations (x y T : ℝ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_operations_l471_47192


namespace NUMINAMATH_CALUDE_cistern_leak_time_l471_47129

theorem cistern_leak_time 
  (fill_time_A : ℝ) 
  (fill_time_both : ℝ) 
  (leak_time_B : ℝ) : 
  fill_time_A = 10 →
  fill_time_both = 29.999999999999993 →
  (1 / fill_time_A - 1 / leak_time_B = 1 / fill_time_both) →
  leak_time_B = 15 := by
sorry

end NUMINAMATH_CALUDE_cistern_leak_time_l471_47129


namespace NUMINAMATH_CALUDE_intersection_property_characterization_l471_47176

/-- A function satisfying the property that the line through any two points
    on its graph intersects the y-axis at (0, pq) -/
def IntersectionProperty (f : ℝ → ℝ) : Prop :=
  ∀ p q : ℝ, p ≠ q →
    let m := (f q - f p) / (q - p)
    let b := f p - m * p
    b = p * q

/-- Theorem stating that functions satisfying the intersection property
    are of the form f(x) = x(c + x) for some constant c -/
theorem intersection_property_characterization (f : ℝ → ℝ) :
  IntersectionProperty f ↔ ∃ c : ℝ, ∀ x : ℝ, f x = x * (c + x) :=
sorry

end NUMINAMATH_CALUDE_intersection_property_characterization_l471_47176


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l471_47182

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f := λ x : ℝ => a^(x - 1) + 4
  f 1 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l471_47182


namespace NUMINAMATH_CALUDE_smallest_rational_l471_47146

theorem smallest_rational (a b c d : ℚ) (ha : a = 1) (hb : b = 0) (hc : c = -1/2) (hd : d = -3) :
  d < a ∧ d < b ∧ d < c := by
  sorry

end NUMINAMATH_CALUDE_smallest_rational_l471_47146


namespace NUMINAMATH_CALUDE_elliptical_machine_cost_l471_47198

/-- The cost of an elliptical machine -/
def machine_cost : ℝ := 120

/-- The daily minimum payment -/
def daily_minimum_payment : ℝ := 6

/-- The number of days to pay the remaining balance -/
def payment_days : ℕ := 10

/-- Theorem stating the cost of the elliptical machine -/
theorem elliptical_machine_cost :
  (machine_cost / 2 = daily_minimum_payment * payment_days) ∧
  (machine_cost / 2 = machine_cost - machine_cost / 2) := by
  sorry

#check elliptical_machine_cost

end NUMINAMATH_CALUDE_elliptical_machine_cost_l471_47198


namespace NUMINAMATH_CALUDE_binary_1010101_is_85_l471_47120

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1010101_is_85 :
  binary_to_decimal [true, false, true, false, true, false, true] = 85 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010101_is_85_l471_47120


namespace NUMINAMATH_CALUDE_gcd_of_180_210_588_l471_47164

theorem gcd_of_180_210_588 : Nat.gcd 180 (Nat.gcd 210 588) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_180_210_588_l471_47164


namespace NUMINAMATH_CALUDE_ten_thousand_eight_hundred_seventy_scientific_notation_l471_47105

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * (10 : ℝ) ^ b ∧ 1 ≤ a ∧ a < 10

theorem ten_thousand_eight_hundred_seventy_scientific_notation :
  scientific_notation 10870 1.087 4 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_eight_hundred_seventy_scientific_notation_l471_47105


namespace NUMINAMATH_CALUDE_study_time_difference_l471_47191

/-- Converts hours to minutes -/
def hoursToMinutes (hours : ℚ) : ℚ := hours * 60

/-- Converts days to minutes -/
def daysToMinutes (days : ℚ) : ℚ := days * 24 * 60

theorem study_time_difference :
  let kwame := hoursToMinutes 2.5
  let connor := hoursToMinutes 1.5
  let lexia := 97
  let michael := hoursToMinutes 3 + 45
  let cassandra := 165
  let aria := daysToMinutes 0.5
  (lexia + aria) - (kwame + connor + michael + cassandra) = 187 := by sorry

end NUMINAMATH_CALUDE_study_time_difference_l471_47191


namespace NUMINAMATH_CALUDE_coin_problem_l471_47173

theorem coin_problem (initial_coins : ℚ) : 
  initial_coins > 0 → 
  let lost_coins := (1/3 : ℚ) * initial_coins
  let found_coins := (2/3 : ℚ) * lost_coins
  let remaining_coins := initial_coins - lost_coins + found_coins
  (initial_coins - remaining_coins) / initial_coins = 1/9 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l471_47173


namespace NUMINAMATH_CALUDE_platform_length_l471_47108

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 8 seconds to cross a signal pole, prove that the length of the platform is 1162.5 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 39)
  (h3 : time_pole = 8) :
  let train_speed := train_length / time_pole
  let platform_length := train_speed * time_platform - train_length
  platform_length = 1162.5 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l471_47108


namespace NUMINAMATH_CALUDE_x_plus_y_value_l471_47189

theorem x_plus_y_value (x y : ℝ) (h : (x - 1)^2 + |2*y + 1| = 0) : x + y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l471_47189


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l471_47185

theorem sum_of_roots_squared_equation (x : ℝ) :
  (x - 3)^2 = 16 → ∃ y : ℝ, (y - 3)^2 = 16 ∧ x + y = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l471_47185


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l471_47150

/-- Given a geometric sequence {aₙ}, prove that if a₁ + a₂ + a₃ = 2 and a₃ + a₄ + a₅ = 8, 
    then a₄ + a₅ + a₆ = ±16 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∃ (q : ℝ), ∀ n, a (n + 1) = q * a n) 
  (h_sum1 : a 1 + a 2 + a 3 = 2) (h_sum2 : a 3 + a 4 + a 5 = 8) : 
  a 4 + a 5 + a 6 = 16 ∨ a 4 + a 5 + a 6 = -16 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l471_47150


namespace NUMINAMATH_CALUDE_coca_cola_banknotes_l471_47196

/-- The number of Coca-Cola bottles -/
def num_bottles : ℕ := 40

/-- The price of each Coca-Cola bottle in yuan -/
def price_per_bottle : ℚ := 28/10

/-- The denomination of the banknotes in yuan -/
def banknote_value : ℕ := 20

/-- The minimum number of banknotes needed -/
def min_banknotes : ℕ := 6

theorem coca_cola_banknotes :
  ∃ (n : ℕ), n ≥ min_banknotes ∧ 
  n * banknote_value ≥ (num_bottles : ℚ) * price_per_bottle ∧
  ∀ (m : ℕ), m * banknote_value ≥ (num_bottles : ℚ) * price_per_bottle → m ≥ min_banknotes :=
by sorry

end NUMINAMATH_CALUDE_coca_cola_banknotes_l471_47196


namespace NUMINAMATH_CALUDE_square_minus_one_factorization_l471_47175

theorem square_minus_one_factorization (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_square_minus_one_factorization_l471_47175


namespace NUMINAMATH_CALUDE_max_dot_product_ellipses_l471_47199

/-- The maximum dot product of vectors to points on two specific ellipses -/
theorem max_dot_product_ellipses : 
  let C₁ := {p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1}
  let C₂ := {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 9 = 1}
  ∃ (max : ℝ), max = 15 ∧ 
    ∀ (M N : ℝ × ℝ), M ∈ C₁ → N ∈ C₂ → 
      (M.1 * N.1 + M.2 * N.2 : ℝ) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_ellipses_l471_47199


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l471_47154

/-- Given a line segment with endpoints (4, -7) and (-8, 9), 
    the product of the coordinates of its midpoint is -2. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 4
  let y1 : ℝ := -7
  let x2 : ℝ := -8
  let y2 : ℝ := 9
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = -2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l471_47154


namespace NUMINAMATH_CALUDE_rank_of_Mn_l471_47160

/-- Definition of the matrix Mn -/
def Mn (n : ℕ+) : Matrix (Fin (2*n+1)) (Fin (2*n+1)) ℤ :=
  Matrix.of fun i j =>
    if i = j then 0
    else if i > j then
      if i - j ≤ n then 1 else -1
    else
      if j - i ≤ n then -1 else 1

/-- The rank of Mn is 2n for any positive integer n -/
theorem rank_of_Mn (n : ℕ+) : Matrix.rank (Mn n) = 2*n := by sorry

end NUMINAMATH_CALUDE_rank_of_Mn_l471_47160


namespace NUMINAMATH_CALUDE_certain_number_problem_l471_47170

theorem certain_number_problem (x : ℝ) : (((x + 10) * 7) / 5) - 5 = 88 / 2 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l471_47170


namespace NUMINAMATH_CALUDE_colonization_theorem_l471_47161

/-- Represents the number of different combinations of planets that can be colonized --/
def colonization_combinations (total_planets : ℕ) (earth_like : ℕ) (mars_like : ℕ) 
  (earth_effort : ℕ) (mars_effort : ℕ) (total_effort : ℕ) : ℕ :=
  (Finset.range (earth_like + 1)).sum (fun a =>
    if 2 * a ≤ total_effort ∧ (total_effort - 2 * a) % 2 = 0 ∧ (total_effort - 2 * a) / 2 ≤ mars_like
    then Nat.choose earth_like a * Nat.choose mars_like ((total_effort - 2 * a) / 2)
    else 0)

/-- The main theorem stating the number of colonization combinations --/
theorem colonization_theorem : 
  colonization_combinations 15 7 8 2 1 16 = 1141 := by sorry

end NUMINAMATH_CALUDE_colonization_theorem_l471_47161


namespace NUMINAMATH_CALUDE_product_and_square_calculation_l471_47163

theorem product_and_square_calculation :
  (100.2 * 99.8 = 9999.96) ∧ (103^2 = 10609) := by
  sorry

end NUMINAMATH_CALUDE_product_and_square_calculation_l471_47163


namespace NUMINAMATH_CALUDE_november_rainfall_total_november_rainfall_l471_47140

/-- Calculates the total rainfall in November given specific conditions -/
theorem november_rainfall (days_in_november : ℕ) 
                          (first_period : ℕ) 
                          (daily_rainfall_first_period : ℝ) 
                          (rainfall_ratio_second_period : ℝ) : ℝ :=
  let second_period := days_in_november - first_period
  let daily_rainfall_second_period := daily_rainfall_first_period * rainfall_ratio_second_period
  let total_rainfall_first_period := (first_period : ℝ) * daily_rainfall_first_period
  let total_rainfall_second_period := (second_period : ℝ) * daily_rainfall_second_period
  total_rainfall_first_period + total_rainfall_second_period

/-- The total rainfall in November is 180 inches -/
theorem total_november_rainfall : 
  november_rainfall 30 15 4 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_november_rainfall_total_november_rainfall_l471_47140


namespace NUMINAMATH_CALUDE_min_cooking_time_is_12_l471_47116

/-- Represents the time taken for each step in the cooking process -/
structure CookingSteps where
  step1 : ℕ  -- Wash pot and fill with water
  step2 : ℕ  -- Wash vegetables
  step3 : ℕ  -- Prepare noodles and seasonings
  step4 : ℕ  -- Boil water
  step5 : ℕ  -- Cook noodles and vegetables

/-- Calculates the minimum cooking time given the cooking steps -/
def minCookingTime (steps : CookingSteps) : ℕ :=
  steps.step1 + max steps.step4 (steps.step2 + steps.step3) + steps.step5

/-- Theorem stating that the minimum cooking time is 12 minutes -/
theorem min_cooking_time_is_12 (steps : CookingSteps) 
  (h1 : steps.step1 = 2)
  (h2 : steps.step2 = 3)
  (h3 : steps.step3 = 2)
  (h4 : steps.step4 = 7)
  (h5 : steps.step5 = 3) :
  minCookingTime steps = 12 := by
  sorry


end NUMINAMATH_CALUDE_min_cooking_time_is_12_l471_47116


namespace NUMINAMATH_CALUDE_cost_price_of_article_l471_47149

/-- 
Given an article where the profit obtained by selling it for Rs. 66 
is equal to the loss obtained by selling it for Rs. 52, 
prove that the cost price of the article is Rs. 59.
-/
theorem cost_price_of_article (cost_price : ℤ) : cost_price = 59 :=
  sorry

end NUMINAMATH_CALUDE_cost_price_of_article_l471_47149


namespace NUMINAMATH_CALUDE_toaster_msrp_l471_47145

/-- The MSRP of a toaster given specific conditions -/
theorem toaster_msrp (x : ℝ) : 
  x + 0.2 * x + 0.5 * (x + 0.2 * x) = 54 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_toaster_msrp_l471_47145


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l471_47155

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel m n → perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l471_47155


namespace NUMINAMATH_CALUDE_inverse_sum_product_l471_47132

theorem inverse_sum_product (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : 3*a + b/3 ≠ 0) :
  (3*a + b/3)⁻¹ * ((3*a)⁻¹ + (b/3)⁻¹) = (a*b)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_product_l471_47132


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l471_47123

theorem square_perimeter_ratio (s S : ℝ) (hs : s > 0) (hS : S > 0) : 
  S * Real.sqrt 2 = 7 * (s * Real.sqrt 2) → 
  (4 * S) / (4 * s) = 7 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l471_47123


namespace NUMINAMATH_CALUDE_sundae_price_l471_47165

/-- Proves that given the specified conditions, the price of each sundae is $1.40 -/
theorem sundae_price : 
  ∀ (ice_cream_bars sundaes : ℕ) 
    (total_price ice_cream_price sundae_price : ℚ),
  ice_cream_bars = 125 →
  sundaes = 125 →
  total_price = 250 →
  ice_cream_price = 0.60 →
  total_price = ice_cream_bars * ice_cream_price + sundaes * sundae_price →
  sundae_price = 1.40 := by
sorry

end NUMINAMATH_CALUDE_sundae_price_l471_47165


namespace NUMINAMATH_CALUDE_probability_red_then_white_l471_47169

theorem probability_red_then_white (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
  (h_total : total_balls = 9)
  (h_red : red_balls = 3)
  (h_white : white_balls = 2)
  : (red_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_then_white_l471_47169


namespace NUMINAMATH_CALUDE_find_number_l471_47177

theorem find_number : ∃ x : ℝ, x = 50 ∧ (0.6 * x = 0.5 * 30 + 15) := by
  sorry

end NUMINAMATH_CALUDE_find_number_l471_47177


namespace NUMINAMATH_CALUDE_matrix_addition_theorem_l471_47181

/-- Given matrices A and B, prove that C = 2A + B is equal to the expected result. -/
theorem matrix_addition_theorem (A B : Matrix (Fin 2) (Fin 2) ℤ) : 
  A = !![2, 1; 3, 4] → 
  B = !![0, -5; -1, 6] → 
  2 • A + B = !![4, -3; 5, 14] := by
  sorry

end NUMINAMATH_CALUDE_matrix_addition_theorem_l471_47181


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l471_47158

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, 2^x + x^2 > 0) ↔ (∃ x : ℝ, 2^x + x^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l471_47158


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_number_l471_47101

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total : Nat
  sample_size : Nat
  interval : Nat

/-- Checks if a number is part of the systematic sample -/
def SystematicSampling.isSampled (s : SystematicSampling) (n : Nat) : Prop :=
  ∃ k : Nat, n = s.interval * k + 1 ∧ k < s.sample_size

theorem systematic_sampling_fourth_number 
  (s : SystematicSampling)
  (h_total : s.total = 52)
  (h_sample_size : s.sample_size = 4)
  (h_5 : s.isSampled 5)
  (h_31 : s.isSampled 31)
  (h_44 : s.isSampled 44) :
  s.isSampled 18 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_fourth_number_l471_47101


namespace NUMINAMATH_CALUDE_matchsticks_problem_l471_47188

/-- The number of matchsticks left in the box after Elvis and Ralph make their squares -/
def matchsticks_left (initial_count : ℕ) (elvis_square_size : ℕ) (ralph_square_size : ℕ) 
                     (elvis_squares : ℕ) (ralph_squares : ℕ) : ℕ :=
  initial_count - (elvis_square_size * elvis_squares + ralph_square_size * ralph_squares)

theorem matchsticks_problem : 
  matchsticks_left 50 4 8 5 3 = 6 := by sorry

end NUMINAMATH_CALUDE_matchsticks_problem_l471_47188


namespace NUMINAMATH_CALUDE_union_of_sets_l471_47121

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4}
  A ∪ B = {1, 2, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l471_47121


namespace NUMINAMATH_CALUDE_odd_7x_plus_4_l471_47114

theorem odd_7x_plus_4 (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_odd_7x_plus_4_l471_47114


namespace NUMINAMATH_CALUDE_scientific_notation_of_1680000_l471_47142

theorem scientific_notation_of_1680000 : 
  ∃ (a : ℝ) (n : ℤ), 1680000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.68 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1680000_l471_47142


namespace NUMINAMATH_CALUDE_race_distances_main_theorem_l471_47179

/-- Represents a race between three racers over a certain distance -/
structure Race where
  distance : ℝ
  a_beats_b : ℝ
  b_beats_c : ℝ
  a_beats_c : ℝ

/-- The theorem stating the distances of the two races -/
theorem race_distances (race1 race2 : Race) : 
  race1.distance = 150 ∧ race2.distance = 120 :=
  by
    have h1 : race1 = { distance := 150, a_beats_b := 30, b_beats_c := 15, a_beats_c := 42 } := by sorry
    have h2 : race2 = { distance := 120, a_beats_b := 25, b_beats_c := 20, a_beats_c := 40 } := by sorry
    sorry

/-- The main theorem proving the distances of both races -/
theorem main_theorem : ∃ (race1 race2 : Race), 
  race1.a_beats_b = 30 ∧ 
  race1.b_beats_c = 15 ∧ 
  race1.a_beats_c = 42 ∧
  race2.a_beats_b = 25 ∧ 
  race2.b_beats_c = 20 ∧ 
  race2.a_beats_c = 40 ∧
  race1.distance = 150 ∧ 
  race2.distance = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_race_distances_main_theorem_l471_47179


namespace NUMINAMATH_CALUDE_lemonade_stand_revenue_l471_47110

theorem lemonade_stand_revenue 
  (total_cups : ℝ) 
  (small_cup_price : ℝ) 
  (h1 : small_cup_price > 0) : 
  let small_cups := (3 / 5) * total_cups
  let large_cups := (2 / 5) * total_cups
  let large_cup_price := (1 / 6) * small_cup_price
  let small_revenue := small_cups * small_cup_price
  let large_revenue := large_cups * large_cup_price
  let total_revenue := small_revenue + large_revenue
  (large_revenue / total_revenue) = (1 / 10) := by
sorry

end NUMINAMATH_CALUDE_lemonade_stand_revenue_l471_47110


namespace NUMINAMATH_CALUDE_trailing_zeros_1_to_20_l471_47112

/-- The number of factors of 5 in n! -/
def count_factors_of_5 (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The number of trailing zeros in the product of factorials from 1 to n -/
def trailing_zeros_factorial_product (n : ℕ) : ℕ :=
  count_factors_of_5 n

theorem trailing_zeros_1_to_20 :
  trailing_zeros_factorial_product 20 = 8 ∧
  trailing_zeros_factorial_product 20 % 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_1_to_20_l471_47112


namespace NUMINAMATH_CALUDE_min_diagonal_rectangle_l471_47190

/-- The minimum diagonal length of a rectangle with perimeter 30 -/
theorem min_diagonal_rectangle (l w : ℝ) (h_perimeter : l + w = 15) :
  ∃ (min_diag : ℝ), min_diag = Real.sqrt 112.5 ∧
  ∀ (diag : ℝ), diag = Real.sqrt (l^2 + w^2) → diag ≥ min_diag :=
by sorry

end NUMINAMATH_CALUDE_min_diagonal_rectangle_l471_47190


namespace NUMINAMATH_CALUDE_largest_t_value_for_temperature_l471_47184

theorem largest_t_value_for_temperature (t : ℝ) :
  let f : ℝ → ℝ := λ x => -x^2 + 10*x + 60
  let solutions := {x : ℝ | f x = 80}
  ∃ max_t ∈ solutions, ∀ t ∈ solutions, t ≤ max_t ∧ max_t = 5 + 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_t_value_for_temperature_l471_47184
