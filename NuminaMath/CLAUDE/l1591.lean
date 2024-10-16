import Mathlib

namespace NUMINAMATH_CALUDE_cosine_largest_angle_triangle_l1591_159162

theorem cosine_largest_angle_triangle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  cos_C = -(1/4) := by
  sorry

end NUMINAMATH_CALUDE_cosine_largest_angle_triangle_l1591_159162


namespace NUMINAMATH_CALUDE_tangent_line_sum_l1591_159185

/-- Given a function f: ℝ → ℝ with a tangent line at x = 1 defined by 2x - y + 1 = 0,
    prove that f(1) + f'(1) = 5 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y, y = f x → (x = 1 → 2*x - y + 1 = 0)) : 
    f 1 + (deriv f) 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l1591_159185


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1591_159166

/-- Given a man's speed with the current and the speed of the current, 
    calculates the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  2 * speed_with_current - 3 * speed_of_current

/-- Theorem stating that given the specific speeds in the problem, 
    the man's speed against the current is 11.2 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 18 3.4 = 11.2 := by
  sorry

#eval speed_against_current 18 3.4

end NUMINAMATH_CALUDE_mans_speed_against_current_l1591_159166


namespace NUMINAMATH_CALUDE_problem_solution_l1591_159163

theorem problem_solution (x : ℝ) : (400 * 7000 : ℝ) = 28000 * (100 ^ x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1591_159163


namespace NUMINAMATH_CALUDE_probability_same_heads_l1591_159134

/-- Represents the outcome of tossing two coins -/
inductive CoinToss
| HH -- Two heads
| HT -- Head then tail
| TH -- Tail then head
| TT -- Two tails

/-- The sample space of all possible outcomes when two people each toss two coins -/
def sampleSpace : List (CoinToss × CoinToss) :=
  [(CoinToss.HH, CoinToss.HH), (CoinToss.HH, CoinToss.HT), (CoinToss.HH, CoinToss.TH), (CoinToss.HH, CoinToss.TT),
   (CoinToss.HT, CoinToss.HH), (CoinToss.HT, CoinToss.HT), (CoinToss.HT, CoinToss.TH), (CoinToss.HT, CoinToss.TT),
   (CoinToss.TH, CoinToss.HH), (CoinToss.TH, CoinToss.HT), (CoinToss.TH, CoinToss.TH), (CoinToss.TH, CoinToss.TT),
   (CoinToss.TT, CoinToss.HH), (CoinToss.TT, CoinToss.HT), (CoinToss.TT, CoinToss.TH), (CoinToss.TT, CoinToss.TT)]

/-- Counts the number of heads in a single coin toss -/
def countHeads : CoinToss → Nat
  | CoinToss.HH => 2
  | CoinToss.HT => 1
  | CoinToss.TH => 1
  | CoinToss.TT => 0

/-- Checks if two coin tosses have the same number of heads -/
def sameHeads : CoinToss × CoinToss → Bool
  | (t1, t2) => countHeads t1 = countHeads t2

/-- The probability of getting the same number of heads -/
theorem probability_same_heads :
  (sampleSpace.filter sameHeads).length / sampleSpace.length = 3 / 8 := by
  sorry


end NUMINAMATH_CALUDE_probability_same_heads_l1591_159134


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_5_l1591_159174

theorem units_digit_of_7_pow_3_pow_5 : 7^(3^5) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_5_l1591_159174


namespace NUMINAMATH_CALUDE_min_product_xyz_l1591_159138

theorem min_product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 1) (hz_twice_y : z = 2 * y)
  (hx_le_2y : x ≤ 2 * y) (hy_le_2x : y ≤ 2 * x) (hz_le_2x : z ≤ 2 * x) :
  ∃ (min_val : ℝ), min_val = 8 / 243 ∧ x * y * z ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_product_xyz_l1591_159138


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1591_159111

theorem selling_price_calculation (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) :
  cost_price = 540 →
  markup_percentage = 15 →
  discount_percentage = 26.570048309178745 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let discount_amount := marked_price * (discount_percentage / 100)
  let selling_price := marked_price - discount_amount
  selling_price = 456 := by
sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1591_159111


namespace NUMINAMATH_CALUDE_prime_difference_product_l1591_159149

theorem prime_difference_product (a b : ℕ) : 
  Nat.Prime a → Nat.Prime b → a - b = 35 → a * b = 74 := by
  sorry

end NUMINAMATH_CALUDE_prime_difference_product_l1591_159149


namespace NUMINAMATH_CALUDE_bench_cost_l1591_159126

theorem bench_cost (total_cost bench_cost table_cost : ℝ) : 
  total_cost = 450 →
  table_cost = 2 * bench_cost →
  total_cost = bench_cost + table_cost →
  bench_cost = 150 := by
sorry

end NUMINAMATH_CALUDE_bench_cost_l1591_159126


namespace NUMINAMATH_CALUDE_video_game_price_l1591_159191

theorem video_game_price (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) : 
  total_games = 15 →
  non_working_games = 6 →
  total_earnings = 63 →
  (total_earnings : ℚ) / (total_games - non_working_games : ℚ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_video_game_price_l1591_159191


namespace NUMINAMATH_CALUDE_hilton_marbles_l1591_159178

/-- Calculates the final number of marbles Hilton has -/
def final_marbles (initial : ℝ) (found : ℝ) (lost : ℝ) (compensation_rate : ℝ) : ℝ :=
  initial + found - lost + compensation_rate * lost

/-- Proves that Hilton ends up with 44.5 marbles given the initial conditions -/
theorem hilton_marbles :
  final_marbles 30 8.5 12 1.5 = 44.5 := by
  sorry

end NUMINAMATH_CALUDE_hilton_marbles_l1591_159178


namespace NUMINAMATH_CALUDE_total_amount_spent_l1591_159171

def meal_prices : List Float := [12, 15, 10, 18, 20]
def ice_cream_prices : List Float := [2, 3, 3, 4, 4]
def tip_percentage : Float := 0.15
def tax_percentage : Float := 0.08

theorem total_amount_spent :
  let total_meal_cost := meal_prices.sum
  let total_ice_cream_cost := ice_cream_prices.sum
  let tip := tip_percentage * total_meal_cost
  let tax := tax_percentage * total_meal_cost
  total_meal_cost + total_ice_cream_cost + tip + tax = 108.25 := by
sorry

end NUMINAMATH_CALUDE_total_amount_spent_l1591_159171


namespace NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l1591_159142

/-- Given a paper with a certain number of pages and a number of days to complete it,
    calculate the number of pages that need to be written per day to finish on time. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

/-- Theorem stating that for a 63-page paper due in 3 days,
    21 pages need to be written per day to finish on time. -/
theorem stacy_paper_pages_per_day :
  pages_per_day 63 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l1591_159142


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1591_159106

/-- Given a quadratic equation 3x^2 + 4x + 5 = 0 with roots r and s,
    if we construct a new quadratic equation x^2 + px + q = 0 with roots 2r and 2s,
    then p = 56/9 -/
theorem quadratic_root_relation (r s : ℝ) (p q : ℝ) : 
  (3 * r^2 + 4 * r + 5 = 0) →
  (3 * s^2 + 4 * s + 5 = 0) →
  ((2 * r)^2 + p * (2 * r) + q = 0) →
  ((2 * s)^2 + p * (2 * s) + q = 0) →
  p = 56 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1591_159106


namespace NUMINAMATH_CALUDE_no_integer_geometric_progression_angles_l1591_159196

/-- Represents the angles of a triangle in geometric progression -/
structure TriangleAngles where
  a : ℕ+  -- first angle
  r : ℕ+  -- common ratio
  h1 : a < a * r  -- angles are distinct
  h2 : a * r < a * r * r  -- angles are distinct
  h3 : a + a * r + a * r * r = 180  -- sum of angles is 180 degrees

/-- There are no triangles with angles that are distinct positive integers in a geometric progression -/
theorem no_integer_geometric_progression_angles : ¬∃ (t : TriangleAngles), True :=
sorry

end NUMINAMATH_CALUDE_no_integer_geometric_progression_angles_l1591_159196


namespace NUMINAMATH_CALUDE_puzzle_ratio_is_three_to_one_l1591_159179

/-- Given a total puzzle-solving time, warm-up time, and number of additional puzzles,
    calculates the ratio of time spent on each additional puzzle to the warm-up time. -/
def puzzle_time_ratio (total_time warm_up_time : ℕ) (num_puzzles : ℕ) : ℚ :=
  let remaining_time := total_time - warm_up_time
  let time_per_puzzle := remaining_time / num_puzzles
  (time_per_puzzle : ℚ) / warm_up_time

/-- Proves that for the given conditions, the ratio of time spent on each additional puzzle
    to the warm-up puzzle is 3:1. -/
theorem puzzle_ratio_is_three_to_one :
  puzzle_time_ratio 70 10 2 = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_ratio_is_three_to_one_l1591_159179


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1591_159121

/-- A quadratic equation in one variable x is of the form ax^2 + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation 3x^2 + 2x + 4 = 0 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 4

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1591_159121


namespace NUMINAMATH_CALUDE_angle_negative_2015_in_second_quadrant_l1591_159101

/-- The quadrant of an angle in degrees -/
inductive Quadrant
| first
| second
| third
| fourth

/-- Determine the quadrant of an angle in degrees -/
def angleQuadrant (angle : ℤ) : Quadrant :=
  let normalizedAngle := angle % 360
  if 0 ≤ normalizedAngle && normalizedAngle < 90 then Quadrant.first
  else if 90 ≤ normalizedAngle && normalizedAngle < 180 then Quadrant.second
  else if 180 ≤ normalizedAngle && normalizedAngle < 270 then Quadrant.third
  else Quadrant.fourth

theorem angle_negative_2015_in_second_quadrant :
  angleQuadrant (-2015) = Quadrant.second := by
  sorry

end NUMINAMATH_CALUDE_angle_negative_2015_in_second_quadrant_l1591_159101


namespace NUMINAMATH_CALUDE_complex_number_problem_l1591_159107

theorem complex_number_problem (a : ℝ) (z : ℂ) : 
  z = (Complex.I * (2 + a * Complex.I)) / (1 - Complex.I) →
  (∃ (b : ℝ), z = b * Complex.I) →
  z = 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1591_159107


namespace NUMINAMATH_CALUDE_coefficient_is_40_l1591_159165

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x³y³ in the expansion of (x+y)(2x-y)⁵
def coefficient_x3y3 : ℤ :=
  2^2 * (-1)^3 * binomial 5 3 + 2^3 * binomial 5 2

-- Theorem statement
theorem coefficient_is_40 : coefficient_x3y3 = 40 := by sorry

end NUMINAMATH_CALUDE_coefficient_is_40_l1591_159165


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1591_159135

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals 12 = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1591_159135


namespace NUMINAMATH_CALUDE_abcg_over_defh_value_l1591_159157

theorem abcg_over_defh_value (a b c d e f g h : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6)
  (h6 : f / g = 5 / 2)
  (h7 : g / h = 3 / 4)
  (h8 : b ≠ 0)
  (h9 : c ≠ 0)
  (h10 : d ≠ 0)
  (h11 : e ≠ 0)
  (h12 : f ≠ 0)
  (h13 : g ≠ 0)
  (h14 : h ≠ 0) :
  a * b * c * g / (d * e * f * h) = 5 / 48 := by
  sorry

end NUMINAMATH_CALUDE_abcg_over_defh_value_l1591_159157


namespace NUMINAMATH_CALUDE_not_coplanar_implies_not_collinear_three_collinear_implies_coplanar_l1591_159100

-- Define a type for points in space
variable (Point : Type)

-- Define the property of being coplanar for four points
variable (coplanar : Point → Point → Point → Point → Prop)

-- Define the property of being collinear for three points
variable (collinear : Point → Point → Point → Prop)

-- Theorem 1: If four points are not coplanar, then any three of them are not collinear
theorem not_coplanar_implies_not_collinear 
  (p q r s : Point) : 
  ¬(coplanar p q r s) → 
  (¬(collinear p q r) ∧ ¬(collinear p q s) ∧ ¬(collinear p r s) ∧ ¬(collinear q r s)) :=
sorry

-- Theorem 2: If there exist three collinear points among four points, then these four points are coplanar
theorem three_collinear_implies_coplanar 
  (p q r s : Point) : 
  (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) → 
  coplanar p q r s :=
sorry

end NUMINAMATH_CALUDE_not_coplanar_implies_not_collinear_three_collinear_implies_coplanar_l1591_159100


namespace NUMINAMATH_CALUDE_cyclists_initial_distance_l1591_159115

/-- The initial distance between two cyclists -/
def initial_distance : ℝ := 50

/-- The speed of each cyclist -/
def cyclist_speed : ℝ := 10

/-- The speed of the fly -/
def fly_speed : ℝ := 15

/-- The total distance covered by the fly -/
def fly_distance : ℝ := 37.5

/-- Theorem stating that the initial distance between the cyclists is 50 miles -/
theorem cyclists_initial_distance :
  initial_distance = 
    (2 * cyclist_speed * fly_distance) / fly_speed :=
by sorry

end NUMINAMATH_CALUDE_cyclists_initial_distance_l1591_159115


namespace NUMINAMATH_CALUDE_pascals_triangle_divisibility_l1591_159177

theorem pascals_triangle_divisibility (p : ℕ) (hp : Prime p) (n : ℕ) :
  (∀ k : ℕ, k ≤ n → ¬(p ∣ Nat.choose n k)) ↔
  ∃ (s q : ℕ), s ≥ 0 ∧ 0 < q ∧ q < p ∧ n = p^s * q - 1 :=
by sorry

end NUMINAMATH_CALUDE_pascals_triangle_divisibility_l1591_159177


namespace NUMINAMATH_CALUDE_area_F1AB_when_slope_is_one_line_equation_when_y_intercept_smallest_l1591_159104

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define a point on the ellipse
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : (x^2 / e.a^2) + (y^2 / e.b^2) = 1

-- Define the focal distance
def focalDistance (e : Ellipse) : ℝ := 3

-- Define the ratio of major axis to focal distance
axiom majorAxisFocalRatio (e : Ellipse) : 2 * e.a / (2 * focalDistance e) = Real.sqrt 2

-- Define the right focus
def rightFocus : ℝ × ℝ := (3, 0)

-- Define a line passing through the right focus
structure LineThroughRightFocus where
  slope : ℝ

-- Define the area of triangle F1AB
def areaF1AB (e : Ellipse) (l : LineThroughRightFocus) : ℝ := sorry

-- Define the y-intercept of the perpendicular bisector of AB
def yInterceptPerpBisector (e : Ellipse) (l : LineThroughRightFocus) : ℝ := sorry

-- Theorem 1
theorem area_F1AB_when_slope_is_one (e : Ellipse) :
  areaF1AB e { slope := 1 } = 12 := sorry

-- Theorem 2
theorem line_equation_when_y_intercept_smallest (e : Ellipse) :
  ∃ (l : LineThroughRightFocus),
    (∀ (l' : LineThroughRightFocus), yInterceptPerpBisector e l ≤ yInterceptPerpBisector e l') ∧
    l.slope = -Real.sqrt 2 / 2 := sorry

end NUMINAMATH_CALUDE_area_F1AB_when_slope_is_one_line_equation_when_y_intercept_smallest_l1591_159104


namespace NUMINAMATH_CALUDE_average_salary_proof_l1591_159122

theorem average_salary_proof (n : ℕ) (total_salary : ℕ → ℕ) : 
  (∃ (m : ℕ), m > 0 ∧ total_salary m / m = 8000) →
  total_salary 4 / 4 = 8450 →
  total_salary 1 = 6500 →
  total_salary 5 = 4700 →
  (total_salary 5 + (total_salary 4 - total_salary 1)) / 4 = 8000 :=
by
  sorry

end NUMINAMATH_CALUDE_average_salary_proof_l1591_159122


namespace NUMINAMATH_CALUDE_fraction_sum_approximation_l1591_159169

theorem fraction_sum_approximation : 
  let sum := (2007 : ℚ) / 2999 + 8001 / 5998 + 2001 / 3999 + 4013 / 7997 + 10007 / 15999 + 2803 / 11998
  5.99 < sum ∧ sum < 6.01 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_approximation_l1591_159169


namespace NUMINAMATH_CALUDE_sally_has_more_cards_l1591_159114

theorem sally_has_more_cards (sally_initial : ℕ) (sally_bought : ℕ) (dan_cards : ℕ)
  (h1 : sally_initial = 27)
  (h2 : sally_bought = 20)
  (h3 : dan_cards = 41) :
  sally_initial + sally_bought - dan_cards = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_has_more_cards_l1591_159114


namespace NUMINAMATH_CALUDE_tank_truck_ratio_l1591_159161

theorem tank_truck_ratio (trucks : ℕ) (total : ℕ) : 
  trucks = 20 → total = 140 → (total - trucks) / trucks = 6 := by
  sorry

end NUMINAMATH_CALUDE_tank_truck_ratio_l1591_159161


namespace NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l1591_159167

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem for the first line
theorem line_through_P_and_origin : 
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    (x = P.1 ∧ y = P.2 ∨ x = 0 ∧ y = 0) → 
    y = m*x + b ∧ 
    (∀ (x y : ℝ), y = m*x + b ↔ x + y = 0) :=
sorry

-- Theorem for the second line
theorem line_through_P_perpendicular_to_l₃ : 
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = P.1 ∧ y = P.2) →
    y = m*x + b ∧
    m * (1/2) = -1 ∧
    (∀ (x y : ℝ), y = m*x + b ↔ 2*x + y + 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l1591_159167


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1591_159139

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1/3)
  (h_S : S = 18)
  (h_sum : S = a / (1 - r))
  (h_convergence : abs r < 1) :
  a = 12 :=
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1591_159139


namespace NUMINAMATH_CALUDE_quadratic_radicals_combination_l1591_159154

theorem quadratic_radicals_combination (x : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ x + 1 = k * (2 * x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_combination_l1591_159154


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1591_159164

universe u

def U : Set ℕ := {2, 3, 4}
def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : 
  (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1591_159164


namespace NUMINAMATH_CALUDE_percentage_equality_l1591_159182

theorem percentage_equality (x y : ℝ) (hx : x ≠ 0) :
  (0.4 * 0.5 * x = 0.2 * 0.3 * y) → y = (10/3) * x := by
sorry

end NUMINAMATH_CALUDE_percentage_equality_l1591_159182


namespace NUMINAMATH_CALUDE_regular_polygon_angle_l1591_159144

theorem regular_polygon_angle (n : ℕ) (h1 : n > 3) :
  (n - 3) * 180 / n = 120 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_angle_l1591_159144


namespace NUMINAMATH_CALUDE_sum_of_five_decimals_theorem_l1591_159199

/-- Represents a two-digit number with a decimal point between the digits -/
structure TwoDigitDecimal where
  firstDigit : ℕ
  secondDigit : ℕ
  first_digit_valid : firstDigit < 10
  second_digit_valid : secondDigit < 10

/-- The sum of five TwoDigitDecimal numbers -/
def sumFiveDecimals (a b c d e : TwoDigitDecimal) : ℚ :=
  (a.firstDigit + a.secondDigit / 10 : ℚ) +
  (b.firstDigit + b.secondDigit / 10 : ℚ) +
  (c.firstDigit + c.secondDigit / 10 : ℚ) +
  (d.firstDigit + d.secondDigit / 10 : ℚ) +
  (e.firstDigit + e.secondDigit / 10 : ℚ)

/-- All digits are different -/
def allDifferent (a b c d e : TwoDigitDecimal) : Prop :=
  a.firstDigit ≠ b.firstDigit ∧ a.firstDigit ≠ c.firstDigit ∧ a.firstDigit ≠ d.firstDigit ∧ a.firstDigit ≠ e.firstDigit ∧
  a.firstDigit ≠ a.secondDigit ∧ a.firstDigit ≠ b.secondDigit ∧ a.firstDigit ≠ c.secondDigit ∧ a.firstDigit ≠ d.secondDigit ∧ a.firstDigit ≠ e.secondDigit ∧
  b.firstDigit ≠ c.firstDigit ∧ b.firstDigit ≠ d.firstDigit ∧ b.firstDigit ≠ e.firstDigit ∧
  b.firstDigit ≠ b.secondDigit ∧ b.firstDigit ≠ c.secondDigit ∧ b.firstDigit ≠ d.secondDigit ∧ b.firstDigit ≠ e.secondDigit ∧
  c.firstDigit ≠ d.firstDigit ∧ c.firstDigit ≠ e.firstDigit ∧
  c.firstDigit ≠ c.secondDigit ∧ c.firstDigit ≠ d.secondDigit ∧ c.firstDigit ≠ e.secondDigit ∧
  d.firstDigit ≠ e.firstDigit ∧
  d.firstDigit ≠ d.secondDigit ∧ d.firstDigit ≠ e.secondDigit ∧
  e.firstDigit ≠ e.secondDigit ∧
  a.secondDigit ≠ b.secondDigit ∧ a.secondDigit ≠ c.secondDigit ∧ a.secondDigit ≠ d.secondDigit ∧ a.secondDigit ≠ e.secondDigit ∧
  b.secondDigit ≠ c.secondDigit ∧ b.secondDigit ≠ d.secondDigit ∧ b.secondDigit ≠ e.secondDigit ∧
  c.secondDigit ≠ d.secondDigit ∧ c.secondDigit ≠ e.secondDigit ∧
  d.secondDigit ≠ e.secondDigit

theorem sum_of_five_decimals_theorem (a b c d e : TwoDigitDecimal) 
  (h1 : allDifferent a b c d e)
  (h2 : ∀ x ∈ [a, b, c, d, e], x.secondDigit ≠ 0) :
  sumFiveDecimals a b c d e = 27 ∨ sumFiveDecimals a b c d e = 18 :=
sorry

end NUMINAMATH_CALUDE_sum_of_five_decimals_theorem_l1591_159199


namespace NUMINAMATH_CALUDE_exists_non_negative_sums_l1591_159195

/-- Represents a sign change operation on a matrix -/
inductive SignChange
| Row (i : Nat)
| Col (j : Nat)

/-- Applies a sequence of sign changes to a matrix -/
def applySignChanges (A : Matrix (Fin m) (Fin n) ℝ) (changes : List SignChange) : Matrix (Fin m) (Fin n) ℝ :=
  sorry

/-- Checks if all row sums and column sums are non-negative -/
def allSumsNonNegative (A : Matrix (Fin m) (Fin n) ℝ) : Prop :=
  sorry

/-- Main theorem: For any matrix, there exists a sequence of sign changes that makes all sums non-negative -/
theorem exists_non_negative_sums (m n : Nat) (A : Matrix (Fin m) (Fin n) ℝ) :
  ∃ (changes : List SignChange), allSumsNonNegative (applySignChanges A changes) :=
by
  sorry

end NUMINAMATH_CALUDE_exists_non_negative_sums_l1591_159195


namespace NUMINAMATH_CALUDE_k_range_l1591_159192

theorem k_range (k : ℝ) : (1 - k > -1 ∧ 1 - k ≤ 3) ↔ -2 ≤ k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_k_range_l1591_159192


namespace NUMINAMATH_CALUDE_steven_jill_difference_l1591_159194

/-- The number of peaches each person has -/
structure PeachCounts where
  jake : ℕ
  steven : ℕ
  jill : ℕ

/-- The conditions given in the problem -/
def problem_conditions (p : PeachCounts) : Prop :=
  p.jake + 6 = p.steven ∧
  p.steven > p.jill ∧
  p.jill = 5 ∧
  p.jake = 17

/-- The theorem to be proved -/
theorem steven_jill_difference (p : PeachCounts) 
  (h : problem_conditions p) : p.steven - p.jill = 18 := by
  sorry

end NUMINAMATH_CALUDE_steven_jill_difference_l1591_159194


namespace NUMINAMATH_CALUDE_rectangle_area_18_l1591_159190

def rectangle_area (w l : ℕ+) : ℕ := w.val * l.val

theorem rectangle_area_18 :
  {p : ℕ+ × ℕ+ | rectangle_area p.1 p.2 = 18} =
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_18_l1591_159190


namespace NUMINAMATH_CALUDE_only_earth_revolves_certain_l1591_159152

-- Define the type for events
inductive Event
| earth_revolves : Event
| shooter_hits_bullseye : Event
| three_suns_appear : Event
| red_light_encounter : Event

-- Define the property of being a certain event
def is_certain_event (e : Event) : Prop :=
  match e with
  | Event.earth_revolves => True
  | _ => False

-- Theorem stating that only the Earth revolving is a certain event
theorem only_earth_revolves_certain :
  ∀ e : Event, is_certain_event e ↔ e = Event.earth_revolves :=
sorry

end NUMINAMATH_CALUDE_only_earth_revolves_certain_l1591_159152


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l1591_159113

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l1591_159113


namespace NUMINAMATH_CALUDE_parabola_properties_l1591_159108

/-- The quadratic function f(x) = x^2 - 8x + 12 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 12

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 4

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := -4

theorem parabola_properties :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ 
  f vertex_x = vertex_y ∧
  f 3 = -3 := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1591_159108


namespace NUMINAMATH_CALUDE_distance_between_A_and_C_l1591_159187

-- Define a type for points on a line
structure Point := (x : ℝ)

-- Define a function to calculate distance between two points
def distance (p q : Point) : ℝ := |p.x - q.x|

-- State the theorem
theorem distance_between_A_and_C 
  (A B C : Point) 
  (on_same_line : ∃ (k : ℝ), B.x = k * A.x + (1 - k) * C.x)
  (AB_distance : distance A B = 5)
  (BC_distance : distance B C = 4) :
  distance A C = 1 ∨ distance A C = 9 := by
sorry


end NUMINAMATH_CALUDE_distance_between_A_and_C_l1591_159187


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l1591_159189

theorem midpoint_distance_theorem (t : ℝ) : 
  let P : ℝ × ℝ := (2 * t - 3, 2)
  let Q : ℝ × ℝ := (-2, 2 * t + 1)
  let M : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  (M.1 - P.1) ^ 2 + (M.2 - P.2) ^ 2 = t ^ 2 + 1 →
  t = 1 + Real.sqrt (3 / 2) ∨ t = 1 - Real.sqrt (3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l1591_159189


namespace NUMINAMATH_CALUDE_max_sundays_in_56_days_l1591_159117

theorem max_sundays_in_56_days : ℕ := by
  -- Define the number of days
  let days : ℕ := 56
  
  -- Define the number of days in a week
  let days_per_week : ℕ := 7
  
  -- Define that each week has one Sunday
  let sundays_per_week : ℕ := 1
  
  -- The maximum number of Sundays is the number of complete weeks in 56 days
  have max_sundays : ℕ := days / days_per_week * sundays_per_week
  
  -- Assert that this equals 8
  have : max_sundays = 8 := by sorry
  
  -- Return the result
  exact max_sundays

end NUMINAMATH_CALUDE_max_sundays_in_56_days_l1591_159117


namespace NUMINAMATH_CALUDE_subset_X_l1591_159184

def X : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}

theorem subset_X : {0} ⊆ X := by
  sorry

end NUMINAMATH_CALUDE_subset_X_l1591_159184


namespace NUMINAMATH_CALUDE_sheela_income_proof_l1591_159129

/-- Sheela's monthly income in rupees -/
def monthly_income : ℝ := 17272.73

/-- The amount Sheela deposits in the bank in rupees -/
def deposit : ℝ := 3800

/-- The percentage of Sheela's monthly income that she deposits -/
def deposit_percentage : ℝ := 22

theorem sheela_income_proof :
  deposit = (deposit_percentage / 100) * monthly_income :=
by sorry

end NUMINAMATH_CALUDE_sheela_income_proof_l1591_159129


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1591_159110

def U : Set Nat := {1, 2, 4, 6, 8}
def A : Set Nat := {1, 2, 4}
def B : Set Nat := {2, 4, 6}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1591_159110


namespace NUMINAMATH_CALUDE_min_sum_of_squares_for_sum_16_l1591_159170

theorem min_sum_of_squares_for_sum_16 :
  ∀ a b c : ℕ+,
  a + b + c = 16 →
  a^2 + b^2 + c^2 ≥ 86 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_for_sum_16_l1591_159170


namespace NUMINAMATH_CALUDE_monthly_fee_calculation_l1591_159150

def cost_per_minute : ℚ := 25 / 100

def total_bill : ℚ := 1202 / 100

def minutes_used : ℚ := 2808 / 100

theorem monthly_fee_calculation :
  ∃ (monthly_fee : ℚ),
    monthly_fee + cost_per_minute * minutes_used = total_bill ∧
    monthly_fee = 5 := by
  sorry

end NUMINAMATH_CALUDE_monthly_fee_calculation_l1591_159150


namespace NUMINAMATH_CALUDE_second_row_equals_first_row_l1591_159119

/-- Represents a 3 × n grid with the properties described in the problem -/
structure Grid (n : ℕ) where
  first_row : Fin n → ℝ
  second_row : Fin n → ℝ
  third_row : Fin n → ℝ
  first_row_increasing : ∀ i j, i < j → first_row i < first_row j
  second_row_permutation : ∀ x, ∃ i, second_row i = x ↔ ∃ j, first_row j = x
  third_row_sum : ∀ i, third_row i = first_row i + second_row i
  third_row_increasing : ∀ i j, i < j → third_row i < third_row j

/-- The main theorem stating that the second row must be identical to the first row -/
theorem second_row_equals_first_row {n : ℕ} (grid : Grid n) :
  ∀ i, grid.second_row i = grid.first_row i :=
sorry

end NUMINAMATH_CALUDE_second_row_equals_first_row_l1591_159119


namespace NUMINAMATH_CALUDE_chad_savings_l1591_159120

/-- Chad's savings calculation --/
theorem chad_savings (savings_rate : ℚ) (mowing : ℚ) (birthday : ℚ) (video_games : ℚ) (odd_jobs : ℚ) : 
  savings_rate = 2/5 → 
  mowing = 600 → 
  birthday = 250 → 
  video_games = 150 → 
  odd_jobs = 150 → 
  savings_rate * (mowing + birthday + video_games + odd_jobs) = 460 := by
sorry

end NUMINAMATH_CALUDE_chad_savings_l1591_159120


namespace NUMINAMATH_CALUDE_no_prime_covering_triples_l1591_159183

/-- A polynomial is prime-covering if for every prime p, there exists an integer n for which p divides P(n) -/
def IsPrimeCovering (P : ℤ → ℤ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ∃ n : ℤ, (p : ℤ) ∣ P n

/-- The polynomial P(x) = (x^2 - a)(x^2 - b)(x^2 - c) -/
def P (a b c : ℤ) (x : ℤ) : ℤ :=
  (x^2 - a) * (x^2 - b) * (x^2 - c)

theorem no_prime_covering_triples :
  ¬ ∃ a b c : ℤ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 25 ∧ IsPrimeCovering (P a b c) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_covering_triples_l1591_159183


namespace NUMINAMATH_CALUDE_water_bottles_per_day_l1591_159127

theorem water_bottles_per_day (total_bottles : ℕ) (total_days : ℕ) (bottles_per_day : ℕ) : 
  total_bottles = 153 → 
  total_days = 17 → 
  total_bottles = bottles_per_day * total_days → 
  bottles_per_day = 9 := by
sorry

end NUMINAMATH_CALUDE_water_bottles_per_day_l1591_159127


namespace NUMINAMATH_CALUDE_expression_equals_six_l1591_159158

theorem expression_equals_six :
  (Real.sqrt 27 + Real.sqrt 48) / Real.sqrt 3 - (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 3 + Real.sqrt 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_six_l1591_159158


namespace NUMINAMATH_CALUDE_ratio_equality_l1591_159118

theorem ratio_equality (x y : ℝ) (h1 : 2 * x = 3 * y) (h2 : y ≠ 0) : x / y = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1591_159118


namespace NUMINAMATH_CALUDE_triangle_area_l1591_159175

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  f (A / 2) = -Real.sqrt 3 / 2 →
  a = 3 →
  b + c = 2 * Real.sqrt 3 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1591_159175


namespace NUMINAMATH_CALUDE_factorial_ratio_problem_l1591_159102

theorem factorial_ratio_problem (m n : ℕ) : 
  m > 1 → n > 1 → (Nat.factorial (n + m)) / (Nat.factorial n) = 17297280 → 
  n / m = 1 ∨ n / m = 31 / 2 := by
sorry

end NUMINAMATH_CALUDE_factorial_ratio_problem_l1591_159102


namespace NUMINAMATH_CALUDE_equality_implies_two_equal_l1591_159173

theorem equality_implies_two_equal (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2/y + y^2/z + z^2/x = x^2/z + z^2/y + y^2/x) :
  x = y ∨ x = z ∨ y = z := by
  sorry

end NUMINAMATH_CALUDE_equality_implies_two_equal_l1591_159173


namespace NUMINAMATH_CALUDE_log_seven_eighteen_l1591_159130

theorem log_seven_eighteen (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) :
  Real.log 18 / Real.log 7 = (a + 2 * b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_seven_eighteen_l1591_159130


namespace NUMINAMATH_CALUDE_green_sequins_per_row_l1591_159109

theorem green_sequins_per_row (blue_rows : Nat) (blue_per_row : Nat) 
  (purple_rows : Nat) (purple_per_row : Nat) (green_rows : Nat) (total_sequins : Nat)
  (h1 : blue_rows = 6) (h2 : blue_per_row = 8)
  (h3 : purple_rows = 5) (h4 : purple_per_row = 12)
  (h5 : green_rows = 9) (h6 : total_sequins = 162) :
  (total_sequins - (blue_rows * blue_per_row + purple_rows * purple_per_row)) / green_rows = 6 := by
  sorry

end NUMINAMATH_CALUDE_green_sequins_per_row_l1591_159109


namespace NUMINAMATH_CALUDE_workshop_workers_count_l1591_159181

/-- Proves that the total number of workers in a workshop is 49 given specific salary conditions. -/
theorem workshop_workers_count :
  let average_salary : ℕ := 8000
  let technician_salary : ℕ := 20000
  let other_salary : ℕ := 6000
  let technician_count : ℕ := 7
  ∃ (total_workers : ℕ) (other_workers : ℕ),
    total_workers = technician_count + other_workers ∧
    total_workers * average_salary = technician_count * technician_salary + other_workers * other_salary ∧
    total_workers = 49 := by
  sorry

#check workshop_workers_count

end NUMINAMATH_CALUDE_workshop_workers_count_l1591_159181


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l1591_159186

theorem theater_ticket_difference :
  ∀ (orchestra_price balcony_price : ℕ) 
    (total_tickets total_cost : ℕ) 
    (orchestra_tickets balcony_tickets : ℕ),
  orchestra_price = 12 →
  balcony_price = 8 →
  total_tickets = 360 →
  total_cost = 3320 →
  orchestra_tickets + balcony_tickets = total_tickets →
  orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_cost →
  balcony_tickets - orchestra_tickets = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l1591_159186


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1591_159105

theorem quadratic_factorization (a : ℝ) : a^2 + 4*a + 4 = (a + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1591_159105


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l1591_159151

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let required_coins := friends * (friends + 1) / 2
  if required_coins > initial_coins then
    required_coins - initial_coins
  else
    0

/-- Theorem stating the minimum number of additional coins needed -/
theorem alex_coin_distribution (friends : ℕ) (initial_coins : ℕ)
    (h1 : friends = 15)
    (h2 : initial_coins = 95) :
    min_additional_coins friends initial_coins = 25 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l1591_159151


namespace NUMINAMATH_CALUDE_first_part_to_total_ratio_l1591_159193

theorem first_part_to_total_ratio : 
  ∃ (n : ℕ), (246.95 : ℝ) / 782 = (4939 : ℝ) / (15640 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_first_part_to_total_ratio_l1591_159193


namespace NUMINAMATH_CALUDE_max_distance_origin_to_line_l1591_159172

/-- Given a line l with equation ax + by + c = 0, where a, b, and c form an arithmetic sequence,
    the maximum distance from the origin O(0,0) to the line l is √5. -/
theorem max_distance_origin_to_line (a b c : ℝ) :
  (∃ d : ℝ, a - b = b - c) →  -- a, b, c form an arithmetic sequence
  (∃ x y : ℝ, a * x + b * y + c = 0) →  -- line equation exists
  (∃ d : ℝ, ∀ x y : ℝ, a * x + b * y + c = 0 → d ≥ Real.sqrt (x^2 + y^2)) →  -- distance definition
  (∃ d : ℝ, ∀ x y : ℝ, a * x + b * y + c = 0 → d ≤ Real.sqrt 5) →  -- upper bound
  (∃ x y : ℝ, a * x + b * y + c = 0 ∧ Real.sqrt (x^2 + y^2) = Real.sqrt 5)  -- maximum distance achieved
  := by sorry

end NUMINAMATH_CALUDE_max_distance_origin_to_line_l1591_159172


namespace NUMINAMATH_CALUDE_distribute_5_3_l1591_159159

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 5 distinct objects into 3 distinct groups,
    where each group must contain at least one object, results in 150 different arrangements. -/
theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l1591_159159


namespace NUMINAMATH_CALUDE_pond_soil_extraction_volume_l1591_159116

/-- Calculates the soil extraction volume of a pond with given dimensions and depths -/
theorem pond_soil_extraction_volume 
  (length width depth1 depth2 : ℝ) 
  (h_length : length = 20)
  (h_width : width = 12)
  (h_depth1 : depth1 = 3)
  (h_depth2 : depth2 = 7) :
  length * width * ((depth1 + depth2) / 2) = 1200 :=
by sorry

end NUMINAMATH_CALUDE_pond_soil_extraction_volume_l1591_159116


namespace NUMINAMATH_CALUDE_smallest_four_digit_sum_16_l1591_159148

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_sum_16 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 16 → 1960 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_sum_16_l1591_159148


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1591_159180

/-- Given a line passing through points (1, 3) and (-3, -1), 
    prove that the sum of its slope and y-intercept is 3. -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
  (3 = m * 1 + b) →  -- Point (1, 3) satisfies the line equation
  (-1 = m * (-3) + b) →  -- Point (-3, -1) satisfies the line equation
  m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1591_159180


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1591_159143

def A : Set ℤ := {-1, 3}
def B : Set ℤ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {-1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1591_159143


namespace NUMINAMATH_CALUDE_integers_abs_le_two_l1591_159156

theorem integers_abs_le_two : 
  {x : ℤ | |x| ≤ 2} = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_integers_abs_le_two_l1591_159156


namespace NUMINAMATH_CALUDE_complex_w_values_l1591_159146

theorem complex_w_values (z : ℂ) (w : ℂ) 
  (h1 : ∃ (r : ℝ), (1 + 3*I) * z = r)
  (h2 : w = z / (2 + I))
  (h3 : Complex.abs w = 5 * Real.sqrt 2) :
  w = 1 + 7*I ∨ w = -1 - 7*I := by
  sorry

end NUMINAMATH_CALUDE_complex_w_values_l1591_159146


namespace NUMINAMATH_CALUDE_income_comparison_l1591_159147

theorem income_comparison (a b : ℝ) (h : a = 0.75 * b) : 
  (b - a) / a = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l1591_159147


namespace NUMINAMATH_CALUDE_triangle_with_same_color_and_unit_area_l1591_159136

-- Define a color type
inductive Color
  | Red
  | Green
  | Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point in the plane
def colorFunction : Point → Color := sorry

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem triangle_with_same_color_and_unit_area :
  ∃ (p1 p2 p3 : Point),
    colorFunction p1 = colorFunction p2 ∧
    colorFunction p2 = colorFunction p3 ∧
    triangleArea p1 p2 p3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_same_color_and_unit_area_l1591_159136


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l1591_159133

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, x^3 - Real.sqrt 3 * x^2 + x - (1 + Real.sqrt 3 / 9) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l1591_159133


namespace NUMINAMATH_CALUDE_museum_visitors_l1591_159123

theorem museum_visitors (V : ℕ) : 
  (130 : ℕ) + (3 * V / 4 : ℕ) = V → V = 520 :=
by sorry

end NUMINAMATH_CALUDE_museum_visitors_l1591_159123


namespace NUMINAMATH_CALUDE_combined_work_time_specific_work_time_l1591_159160

/-- The time taken for two workers to complete a task together, given their individual completion times. -/
theorem combined_work_time (x_time y_time : ℝ) (hx : x_time > 0) (hy : y_time > 0) :
  let combined_time := 1 / (1 / x_time + 1 / y_time)
  combined_time = (x_time * y_time) / (x_time + y_time) := by
  sorry

/-- The specific case where one worker takes 20 days and the other takes 40 days. -/
theorem specific_work_time :
  let x_time : ℝ := 20
  let y_time : ℝ := 40
  let combined_time := 1 / (1 / x_time + 1 / y_time)
  combined_time = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_combined_work_time_specific_work_time_l1591_159160


namespace NUMINAMATH_CALUDE_square_inequality_l1591_159176

theorem square_inequality (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l1591_159176


namespace NUMINAMATH_CALUDE_average_physics_chemistry_l1591_159168

/-- Given the scores in three subjects, prove the average of two subjects --/
theorem average_physics_chemistry 
  (total_average : ℝ) 
  (physics_math_average : ℝ) 
  (physics_score : ℝ) 
  (h1 : total_average = 60) 
  (h2 : physics_math_average = 90) 
  (h3 : physics_score = 140) : 
  (physics_score + (3 * total_average - physics_score - (2 * physics_math_average - physics_score))) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_physics_chemistry_l1591_159168


namespace NUMINAMATH_CALUDE_speed_conversion_l1591_159188

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- Given speed in meters per second -/
def speed_mps : ℝ := 12.7788

/-- Theorem stating the conversion of the given speed from m/s to km/h -/
theorem speed_conversion :
  speed_mps * mps_to_kmph = 45.96368 := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l1591_159188


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l1591_159103

theorem complex_modulus_equality (t : ℝ) : 
  t > 0 → Complex.abs (-3 + t * Complex.I) = 5 → t = 4 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l1591_159103


namespace NUMINAMATH_CALUDE_other_root_of_complex_equation_l1591_159153

theorem other_root_of_complex_equation (z : ℂ) :
  z^2 = -100 + 75*I ∧ z = 5 + 10*I → -5 - 10*I = -z :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_complex_equation_l1591_159153


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1591_159112

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₃ * a₉ = 4 * a₄, then a₈ = 4 -/
theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_cond : a 3 * a 9 = 4 * a 4) : 
  a 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1591_159112


namespace NUMINAMATH_CALUDE_inverse_composition_l1591_159125

-- Define the function f and its inverse
def f : ℝ → ℝ := sorry

def f_inv : ℝ → ℝ := sorry

-- Define the conditions
axiom f_4 : f 4 = 6
axiom f_6 : f 6 = 3
axiom f_3 : f 3 = 7
axiom f_7 : f 7 = 2

-- Define the inverse relationship
axiom f_inverse : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Theorem to prove
theorem inverse_composition :
  f_inv (f_inv 7 + f_inv 6) = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_l1591_159125


namespace NUMINAMATH_CALUDE_second_bottle_capacity_l1591_159140

theorem second_bottle_capacity
  (total_milk : ℝ)
  (first_bottle_capacity : ℝ)
  (second_bottle_milk : ℝ)
  (h1 : total_milk = 8)
  (h2 : first_bottle_capacity = 4)
  (h3 : second_bottle_milk = 16 / 3)
  (h4 : ∃ (f : ℝ), f * first_bottle_capacity + second_bottle_milk = total_milk ∧
                   f * first_bottle_capacity ≤ first_bottle_capacity ∧
                   second_bottle_milk ≤ f * (total_milk - first_bottle_capacity * f)) :
  total_milk - first_bottle_capacity * (total_milk - second_bottle_milk) / first_bottle_capacity = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_second_bottle_capacity_l1591_159140


namespace NUMINAMATH_CALUDE_no_base_for_131_perfect_square_l1591_159197

theorem no_base_for_131_perfect_square :
  ¬ ∃ (b : ℕ), b ≥ 2 ∧ ∃ (n : ℕ), b^2 + 3*b + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_131_perfect_square_l1591_159197


namespace NUMINAMATH_CALUDE_equation_proof_l1591_159132

theorem equation_proof : (49 : ℚ) / (7 - 3 / 4) = 196 / 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1591_159132


namespace NUMINAMATH_CALUDE_pictures_on_back_l1591_159145

theorem pictures_on_back (total : ℕ) (front : ℕ) (back : ℕ) 
  (h1 : total = 15) 
  (h2 : front = 6) 
  (h3 : total = front + back) : 
  back = 9 := by
  sorry

end NUMINAMATH_CALUDE_pictures_on_back_l1591_159145


namespace NUMINAMATH_CALUDE_simplify_expression_l1591_159155

theorem simplify_expression :
  (3 * (Real.sqrt 3 + Real.sqrt 5)) / (4 * Real.sqrt (3 + Real.sqrt 4)) =
  (3 * Real.sqrt 15 + 3 * Real.sqrt 5) / 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1591_159155


namespace NUMINAMATH_CALUDE_solve_inequality_find_a_l1591_159198

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + 2 * |x - a|

-- Theorem for part I
theorem solve_inequality (x : ℝ) :
  f x 1 < 5 ↔ -2/3 < x ∧ x < 8/3 :=
sorry

-- Theorem for part II
theorem find_a :
  ∃ (a : ℝ), (∀ x, f x a ≥ 5) ∧ (∃ x, f x a = 5) → a = -4 :=
sorry

end NUMINAMATH_CALUDE_solve_inequality_find_a_l1591_159198


namespace NUMINAMATH_CALUDE_percentage_of_students_with_glasses_l1591_159124

def total_students : ℕ := 325
def students_without_glasses : ℕ := 195

theorem percentage_of_students_with_glasses :
  (((total_students - students_without_glasses) : ℚ) / total_students) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_with_glasses_l1591_159124


namespace NUMINAMATH_CALUDE_lemons_for_new_recipe_l1591_159137

/-- Represents the number of lemons per gallon in the original recipe -/
def original_lemons_per_gallon : ℚ := 36 / 48

/-- Represents the additional lemons per gallon in the new recipe -/
def additional_lemons_per_gallon : ℚ := 2 / 6

/-- Represents the number of gallons we want to make -/
def gallons_to_make : ℚ := 18

/-- Theorem stating that 18 gallons of the new recipe requires 19.5 lemons -/
theorem lemons_for_new_recipe : 
  (original_lemons_per_gallon + additional_lemons_per_gallon) * gallons_to_make = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_lemons_for_new_recipe_l1591_159137


namespace NUMINAMATH_CALUDE_surface_area_of_problem_solid_l1591_159141

/-- Represents a solid formed by unit cubes -/
structure CubeSolid where
  base_length : ℕ
  top_cube_position : ℕ
  total_cubes : ℕ

/-- Calculate the surface area of the cube solid -/
def surface_area (solid : CubeSolid) : ℕ :=
  -- Front and back
  2 * solid.base_length +
  -- Left and right sides
  (solid.base_length - 1) + (solid.top_cube_position + 3) +
  -- Top surface
  solid.base_length + 1

/-- The specific cube solid described in the problem -/
def problem_solid : CubeSolid :=
  { base_length := 7
  , top_cube_position := 2
  , total_cubes := 8 }

theorem surface_area_of_problem_solid :
  surface_area problem_solid = 34 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_problem_solid_l1591_159141


namespace NUMINAMATH_CALUDE_nested_square_root_simplification_l1591_159128

theorem nested_square_root_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (y * Real.sqrt (y * Real.sqrt (y * Real.sqrt y))) = (y ^ 9) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_simplification_l1591_159128


namespace NUMINAMATH_CALUDE_inequality_implies_k_bound_l1591_159131

theorem inequality_implies_k_bound (k : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (2*x + y)) → 
  k ≥ Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_k_bound_l1591_159131
