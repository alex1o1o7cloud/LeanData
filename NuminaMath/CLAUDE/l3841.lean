import Mathlib

namespace product_evaluation_l3841_384153

theorem product_evaluation (a : ℤ) (h : a = 3) :
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * 3 = 0 := by
  sorry

end product_evaluation_l3841_384153


namespace train_journey_distance_l3841_384170

/-- Represents the train's journey with an accident -/
structure TrainJourney where
  initialSpeed : ℝ
  totalDistance : ℝ
  accidentDelay : ℝ
  speedReductionFactor : ℝ
  totalDelay : ℝ
  alternateLaterAccidentDistance : ℝ
  alternateTotalDelay : ℝ

/-- The train journey satisfies the given conditions -/
def satisfiesConditions (j : TrainJourney) : Prop :=
  j.accidentDelay = 0.5 ∧
  j.speedReductionFactor = 3/4 ∧
  j.totalDelay = 3.5 ∧
  j.alternateLaterAccidentDistance = 90 ∧
  j.alternateTotalDelay = 3

/-- The theorem stating that the journey distance is 600 miles -/
theorem train_journey_distance (j : TrainJourney) 
  (h : satisfiesConditions j) : j.totalDistance = 600 :=
sorry

#check train_journey_distance

end train_journey_distance_l3841_384170


namespace power_seven_equals_product_l3841_384174

theorem power_seven_equals_product (a : ℝ) : a^7 = a^3 * a^4 := by
  sorry

end power_seven_equals_product_l3841_384174


namespace at_least_one_fraction_less_than_one_l3841_384186

theorem at_least_one_fraction_less_than_one 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : y - x > 1) : 
  (1 - y) / x < 1 ∨ (1 + 3 * x) / y < 1 := by
  sorry

end at_least_one_fraction_less_than_one_l3841_384186


namespace product_inequality_l3841_384106

theorem product_inequality (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  (x^2 - 2*x + 2) * (y^2 - 2*y + 2) * (z^2 - 2*z + 2) ≤ (x*y*z)^2 - 2*(x*y*z) + 2 :=
by sorry

end product_inequality_l3841_384106


namespace smallest_solution_quadratic_equation_l3841_384191

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ y => 3 * y^2 + 33 * y - 90 - y * (y + 18)
  ∃ y : ℝ, f y = 0 ∧ ∀ z : ℝ, f z = 0 → y ≤ z ∧ y = -18 :=
by sorry

end smallest_solution_quadratic_equation_l3841_384191


namespace max_area_quadrilateral_ellipse_l3841_384152

/-- Given an ellipse with equation x²/a² + y²/b² = 1, where a > 0 and b > 0,
    the maximum area of quadrilateral OAPB is √2/2 * a * b,
    where A is the point on the positive x-axis,
    B is the point on the positive y-axis,
    and P is any point on the ellipse within the first quadrant. -/
theorem max_area_quadrilateral_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let ellipse := {P : ℝ × ℝ | (P.1^2 / a^2) + (P.2^2 / b^2) = 1}
  let A := (a, 0)
  let B := (0, b)
  let valid_P := {P ∈ ellipse | P.1 ≥ 0 ∧ P.2 ≥ 0}
  let area (P : ℝ × ℝ) := (P.1 * P.2 / 2) + ((a - P.1) * b + (0 - P.2) * a) / 2
  (⨆ P ∈ valid_P, area P) = Real.sqrt 2 / 2 * a * b := by
  sorry

end max_area_quadrilateral_ellipse_l3841_384152


namespace problem_solution_l3841_384123

theorem problem_solution (x y : ℝ) (h : -x + 2*y = 5) :
  5*(x - 2*y)^2 - 3*(x - 2*y) - 60 = 80 := by
  sorry

end problem_solution_l3841_384123


namespace range_of_a_range_of_x_l3841_384101

-- Define the conditions p and q
def p (x : ℝ) : Prop := Real.sqrt (x - 1) ≤ 1
def q (x a : ℝ) : Prop := -1 ≤ x ∧ x ≤ a

-- Define the set A based on condition p
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Define the set B based on condition q
def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

-- Theorem 1: Range of a when q is necessary but not sufficient for p
theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, p x → q x a) ∧ (∃ x : ℝ, q x a ∧ ¬p x) ↔ 2 ≤ a :=
sorry

-- Theorem 2: Range of x when a = 1 and at least one of p or q holds true
theorem range_of_x : 
  ∀ x : ℝ, (p x ∨ q x 1) ↔ -1 ≤ x ∧ x ≤ 2 :=
sorry

end range_of_a_range_of_x_l3841_384101


namespace new_water_height_after_cube_submersion_l3841_384121

/-- Calculates the new water height in a fish tank after submerging a cube -/
theorem new_water_height_after_cube_submersion
  (tank_width : ℝ)
  (tank_length : ℝ)
  (initial_height : ℝ)
  (cube_edge : ℝ)
  (h_width : tank_width = 50)
  (h_length : tank_length = 16)
  (h_initial_height : initial_height = 15)
  (h_cube_edge : cube_edge = 10) :
  let tank_area := tank_width * tank_length
  let cube_volume := cube_edge ^ 3
  let height_increase := cube_volume / tank_area
  let new_height := initial_height + height_increase
  new_height = 16.25 := by sorry

end new_water_height_after_cube_submersion_l3841_384121


namespace specific_ellipse_intercept_l3841_384127

/-- Definition of an ellipse with given foci and one x-intercept -/
structure Ellipse where
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  x_intercept1 : ℝ × ℝ
  sum_distances : ℝ

/-- The other x-intercept of the ellipse -/
def other_x_intercept (e : Ellipse) : ℝ × ℝ := sorry

/-- Theorem stating the other x-intercept of the specific ellipse -/
theorem specific_ellipse_intercept :
  let e : Ellipse := {
    foci1 := (0, 3),
    foci2 := (4, 0),
    x_intercept1 := (0, 0),
    sum_distances := 7
  }
  other_x_intercept e = (56/11, 0) := by sorry

end specific_ellipse_intercept_l3841_384127


namespace joe_cars_count_l3841_384155

theorem joe_cars_count (initial_cars additional_cars : ℕ) : 
  initial_cars = 50 → additional_cars = 12 → initial_cars + additional_cars = 62 := by
  sorry

end joe_cars_count_l3841_384155


namespace playground_area_l3841_384160

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playgroundArea : ℝ

/-- Theorem: The area of the playground in a rectangular landscape -/
theorem playground_area (l : Landscape) : 
  l.length = 4 * l.breadth → 
  l.length = 120 → 
  l.playgroundArea = (1/3) * (l.length * l.breadth) → 
  l.playgroundArea = 1200 := by
  sorry

/-- The main result -/
def main_result : ℝ := 1200

#check playground_area
#check main_result

end playground_area_l3841_384160


namespace discretionary_income_ratio_l3841_384156

/-- Jill's financial situation --/
def jill_finances (net_salary : ℚ) (discretionary_income : ℚ) : Prop :=
  net_salary = 3600 ∧
  0.30 * discretionary_income + 0.20 * discretionary_income + 0.35 * discretionary_income + 108 = discretionary_income ∧
  discretionary_income > 0

/-- The ratio of discretionary income to net salary is 1:5 --/
theorem discretionary_income_ratio
  (net_salary discretionary_income : ℚ)
  (h : jill_finances net_salary discretionary_income) :
  discretionary_income / net_salary = 1 / 5 :=
by sorry

end discretionary_income_ratio_l3841_384156


namespace tv_price_change_l3841_384124

theorem tv_price_change (P : ℝ) : P > 0 →
  let price_after_decrease := P * (1 - 0.20)
  let price_after_increase := price_after_decrease * (1 + 0.30)
  price_after_increase = P * 1.04 :=
by
  sorry

end tv_price_change_l3841_384124


namespace line_equation_through_points_l3841_384143

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℚ
  y₁ : ℚ
  x₂ : ℚ
  y₂ : ℚ

/-- The slope of a line -/
def Line.slope (l : Line) : ℚ := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)

/-- The y-intercept of a line -/
def Line.yIntercept (l : Line) : ℚ := l.y₁ - l.slope * l.x₁

/-- The equation of a line in the form y = mx + b -/
def Line.equation (l : Line) (x : ℚ) : ℚ := l.slope * x + l.yIntercept

theorem line_equation_through_points :
  let l : Line := { x₁ := 2, y₁ := 3, x₂ := -1, y₂ := -1 }
  ∀ x, l.equation x = (4/3) * x + (1/3) := by sorry

end line_equation_through_points_l3841_384143


namespace binomial_variance_l3841_384133

/-- A random variable following a binomial distribution with two outcomes -/
structure BinomialRV where
  p : ℝ  -- Probability of success (X = 1)
  q : ℝ  -- Probability of failure (X = 0)
  sum_one : p + q = 1  -- Sum of probabilities is 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.p * X.q

/-- Theorem: The variance of a binomial random variable X is equal to pq -/
theorem binomial_variance (X : BinomialRV) : variance X = X.p * X.q := by
  sorry

end binomial_variance_l3841_384133


namespace max_product_vertical_multiplication_l3841_384113

theorem max_product_vertical_multiplication :
  ∀ a b : ℕ,
  50 ≤ a ∧ a < 100 →
  100 ≤ b ∧ b < 1000 →
  ∃ c d e f g : ℕ,
  a * b = 10000 * c + 1000 * d + 100 * e + 10 * f + g ∧
  c = 2 ∧ d = 0 ∧ e = 1 ∧ f = 5 →
  a * b ≤ 19864 :=
by sorry

end max_product_vertical_multiplication_l3841_384113


namespace total_red_cards_l3841_384100

/-- The number of decks of playing cards --/
def num_decks : ℕ := 8

/-- The number of red cards in one standard deck --/
def red_cards_per_deck : ℕ := 26

/-- Theorem: The total number of red cards in 8 decks is 208 --/
theorem total_red_cards : num_decks * red_cards_per_deck = 208 := by
  sorry

end total_red_cards_l3841_384100


namespace f_max_at_neg_two_l3841_384161

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 18

/-- The statement that f attains its maximum at x = -2 with a value of 26 -/
theorem f_max_at_neg_two :
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≤ f a) ∧
  (∀ (x : ℝ), f x ≤ 26) ∧
  (f (-2) = 26) :=
sorry

end f_max_at_neg_two_l3841_384161


namespace quadratic_equation_roots_l3841_384134

theorem quadratic_equation_roots (k : ℕ) 
  (distinct_roots : ∃ x y : ℕ+, x ≠ y ∧ 
    (k^2 - 1) * x^2 - 6 * (3*k - 1) * x + 72 = 0 ∧
    (k^2 - 1) * y^2 - 6 * (3*k - 1) * y + 72 = 0) :
  k = 2 ∧ ∃ x y : ℕ+, x = 6 ∧ y = 4 ∧
    (k^2 - 1) * x^2 - 6 * (3*k - 1) * x + 72 = 0 ∧
    (k^2 - 1) * y^2 - 6 * (3*k - 1) * y + 72 = 0 :=
by sorry

end quadratic_equation_roots_l3841_384134


namespace required_run_rate_calculation_l3841_384184

/-- Represents a cricket game situation -/
structure CricketGame where
  totalOvers : ℕ
  firstInningOvers : ℕ
  firstInningRunRate : ℚ
  wicketsLost : ℕ
  targetScore : ℕ
  remainingRunsNeeded : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstInningOvers
  let runsScored := game.firstInningRunRate * game.firstInningOvers
  let actualRemainingRuns := game.targetScore - runsScored
  actualRemainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given game situation -/
theorem required_run_rate_calculation (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstInningOvers = 20)
  (h3 : game.firstInningRunRate = 4.2)
  (h4 : game.wicketsLost = 5)
  (h5 : game.targetScore = 250)
  (h6 : game.remainingRunsNeeded = 195) :
  requiredRunRate game = 5.53 := by
  sorry

#eval requiredRunRate {
  totalOvers := 50,
  firstInningOvers := 20,
  firstInningRunRate := 4.2,
  wicketsLost := 5,
  targetScore := 250,
  remainingRunsNeeded := 195
}

end required_run_rate_calculation_l3841_384184


namespace tangent_segment_length_l3841_384144

/-- Given three circles where two touch externally and a common tangent, 
    calculate the length of the tangent segment within the third circle. -/
theorem tangent_segment_length 
  (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 3) 
  (h₂ : r₂ = 4) 
  (h₃ : r₃ = 5) 
  (h_touch : r₁ + r₂ = 7) : 
  ∃ (y : ℝ), y = (40 * Real.sqrt 3) / 7 ∧ 
  y = 2 * Real.sqrt (r₃^2 - ((r₂ - r₁)^2 / (4 * (r₁ + r₂)^2)) * r₃^2) := by
  sorry


end tangent_segment_length_l3841_384144


namespace common_root_and_other_roots_l3841_384142

def f (x : ℝ) : ℝ := x^4 - x^3 - 22*x^2 + 16*x + 96
def g (x : ℝ) : ℝ := x^3 - 2*x^2 - 3*x + 10

theorem common_root_and_other_roots :
  (f (-2) = 0 ∧ g (-2) = 0) ∧
  (f 3 = 0 ∧ f (-4) = 0 ∧ f 4 = 0) :=
sorry

end common_root_and_other_roots_l3841_384142


namespace permutation_count_equals_fibonacci_l3841_384182

/-- The number of permutations satisfying the given condition -/
def P (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else P (n - 1) + P (n - 2)

/-- The nth Fibonacci number -/
def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

/-- Theorem stating the equivalence between P(n) and the (n+1)th Fibonacci number -/
theorem permutation_count_equals_fibonacci (n : ℕ) :
  P n = fib (n + 1) := by
  sorry

end permutation_count_equals_fibonacci_l3841_384182


namespace real_number_line_bijection_sqrt_six_representation_l3841_384119

-- Define the number line as a type isomorphic to ℝ
def NumberLine : Type := ℝ

-- Statement 1: There exists a bijective function between real numbers and points on the number line
theorem real_number_line_bijection : ∃ f : ℝ → NumberLine, Function.Bijective f :=
sorry

-- Statement 2: The arithmetic square root of 6 is represented by √6
theorem sqrt_six_representation : Real.sqrt 6 = (6 : ℝ).sqrt :=
sorry

end real_number_line_bijection_sqrt_six_representation_l3841_384119


namespace geometric_sequence_sum_l3841_384193

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence with common ratio q
  (a 1 + a 2 + a 3 + a 4 = 3) →  -- First condition
  (a 5 + a 6 + a 7 + a 8 = 48) →  -- Second condition
  (a 1 / (1 - q) = -1/5) :=  -- Conclusion to prove
by sorry

end geometric_sequence_sum_l3841_384193


namespace sum_of_digits_of_expression_l3841_384154

-- Define the expression
def expression : ℕ := (2 + 4)^15

-- Function to get the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Function to get the ones digit of a number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem sum_of_digits_of_expression :
  tens_digit expression + ones_digit expression = 13 := by
  sorry

end sum_of_digits_of_expression_l3841_384154


namespace emma_widget_production_difference_l3841_384147

/-- 
Given Emma's widget production rates and working hours on Monday and Tuesday, 
prove the difference in total widgets produced.
-/
theorem emma_widget_production_difference 
  (w t : ℕ) 
  (h1 : w = 3 * t) : 
  w * t - (w + 5) * (t - 3) = 4 * t + 15 := by
  sorry

end emma_widget_production_difference_l3841_384147


namespace nested_sqrt_power_l3841_384159

theorem nested_sqrt_power (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15/16) := by
  sorry

end nested_sqrt_power_l3841_384159


namespace rectangle_side_ratio_l3841_384140

theorem rectangle_side_ratio (a b : ℝ) (h : b = 2 * a) : (b / a) ^ 2 = 4 := by
  sorry

end rectangle_side_ratio_l3841_384140


namespace cyclists_meet_time_l3841_384131

/-- Two cyclists on a circular track meet at the starting point -/
theorem cyclists_meet_time (v1 v2 circumference : ℝ) (h1 : v1 = 7) (h2 : v2 = 8) (h3 : circumference = 300) : 
  (circumference / (v1 + v2) = 20) := by
  sorry

end cyclists_meet_time_l3841_384131


namespace polygon_division_existence_l3841_384169

/-- A polygon represented by a list of points in 2D space -/
def Polygon : Type := List (ℝ × ℝ)

/-- A line segment represented by its two endpoints -/
def LineSegment : Type := (ℝ × ℝ) × (ℝ × ℝ)

/-- Function to check if a line segment divides a polygon into two equal-area parts -/
def divides_equally (p : Polygon) (l : LineSegment) : Prop := sorry

/-- Function to check if a line segment bisects a side of a polygon -/
def bisects_side (p : Polygon) (l : LineSegment) : Prop := sorry

/-- Function to check if a line segment divides a side of a polygon in 1:2 ratio -/
def divides_side_in_ratio (p : Polygon) (l : LineSegment) : Prop := sorry

/-- Function to check if a polygon is convex -/
def is_convex (p : Polygon) : Prop := sorry

theorem polygon_division_existence :
  ∃ (p : Polygon) (l : LineSegment), 
    divides_equally p l ∧ 
    bisects_side p l ∧ 
    divides_side_in_ratio p l ∧
    is_convex p :=
sorry

end polygon_division_existence_l3841_384169


namespace rectangle_length_fraction_of_circle_radius_l3841_384151

theorem rectangle_length_fraction_of_circle_radius 
  (square_area : ℝ) 
  (rectangle_area : ℝ) 
  (rectangle_breadth : ℝ) 
  (h1 : square_area = 3025) 
  (h2 : rectangle_area = 220) 
  (h3 : rectangle_breadth = 10) : 
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
sorry

end rectangle_length_fraction_of_circle_radius_l3841_384151


namespace beijing_winter_olympics_assignment_schemes_l3841_384195

/-- The number of ways to assign volunteers to events -/
def assignment_schemes (n m : ℕ) : ℕ :=
  (n.choose 2) * m.factorial

/-- Theorem stating the number of assignment schemes for 5 volunteers and 4 events -/
theorem beijing_winter_olympics_assignment_schemes :
  assignment_schemes 5 4 = 240 := by
  sorry

end beijing_winter_olympics_assignment_schemes_l3841_384195


namespace not_all_x_heartsuit_zero_eq_x_l3841_384103

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Theorem stating that "x ♡ 0 = x for all x" is false
theorem not_all_x_heartsuit_zero_eq_x : ¬ ∀ x : ℝ, heartsuit x 0 = x := by
  sorry

end not_all_x_heartsuit_zero_eq_x_l3841_384103


namespace square_inscribed_problem_l3841_384163

theorem square_inscribed_problem (inner_perimeter outer_perimeter : ℝ) 
  (h1 : inner_perimeter = 32)
  (h2 : outer_perimeter = 40)
  (h3 : inner_perimeter > 0)
  (h4 : outer_perimeter > 0) :
  let inner_side := inner_perimeter / 4
  let outer_side := outer_perimeter / 4
  let third_side := 2 * inner_side
  (∃ (greatest_distance : ℝ), 
    greatest_distance = Real.sqrt 2 ∧ 
    greatest_distance = (outer_side * Real.sqrt 2 - inner_side * Real.sqrt 2) / 2) ∧
  third_side ^ 2 = 256 := by
sorry

end square_inscribed_problem_l3841_384163


namespace combine_like_terms_l3841_384137

theorem combine_like_terms (a : ℝ) : 2 * a - 5 * a = -3 * a := by
  sorry

end combine_like_terms_l3841_384137


namespace largest_angle_in_345_ratio_triangle_l3841_384162

theorem largest_angle_in_345_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →
    b = (4/3) * a →
    c = (5/3) * a →
    a + b + c = 180 →
    c = 75 := by
sorry

end largest_angle_in_345_ratio_triangle_l3841_384162


namespace k_lower_bound_l3841_384102

/-- Piecewise function f(x) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else k * x

/-- Theorem stating the lower bound of k -/
theorem k_lower_bound (k : ℝ) :
  (∃ x₀ : ℝ, f k (-x₀) = f k x₀) → k ≥ -Real.exp (-1) :=
by sorry

end k_lower_bound_l3841_384102


namespace algebraic_expression_equality_l3841_384171

theorem algebraic_expression_equality (x y : ℝ) :
  2 * x - y + 1 = 3 → 4 * x - 2 * y + 5 = 9 := by
  sorry

end algebraic_expression_equality_l3841_384171


namespace katie_math_problems_l3841_384125

/-- Given that Katie had 9 math problems for homework and 4 problems left to do after the bus ride,
    prove that she finished 5 problems on the bus ride home. -/
theorem katie_math_problems (total : ℕ) (remaining : ℕ) (h1 : total = 9) (h2 : remaining = 4) :
  total - remaining = 5 := by
  sorry

end katie_math_problems_l3841_384125


namespace distribute_basketballs_count_l3841_384196

/-- The number of ways to distribute four labeled basketballs among three kids -/
def distribute_basketballs : ℕ :=
  30

/-- Each kid must get at least one basketball -/
axiom each_kid_gets_one : True

/-- Basketballs are labeled 1, 2, 3, and 4 -/
axiom basketballs_labeled : True

/-- Basketballs 1 and 2 cannot be given to the same kid -/
axiom one_and_two_separate : True

/-- The number of ways to distribute the basketballs satisfying all conditions is 30 -/
theorem distribute_basketballs_count :
  distribute_basketballs = 30 :=
by sorry

end distribute_basketballs_count_l3841_384196


namespace smallest_x_for_equation_l3841_384167

theorem smallest_x_for_equation : 
  ∃ (x : ℕ+), x = 4 ∧ 
  (∀ (y : ℕ+), (3 : ℚ) / 4 = (y : ℚ) / (200 + x)) ∧
  (∀ (x' : ℕ+), x' < x → 
    ¬∃ (y : ℕ+), (3 : ℚ) / 4 = (y : ℚ) / (200 + x')) :=
by sorry

end smallest_x_for_equation_l3841_384167


namespace fertilizer_amounts_l3841_384172

def petunia_flats : ℕ := 4
def petunias_per_flat : ℕ := 8
def rose_flats : ℕ := 3
def roses_per_flat : ℕ := 6
def sunflower_flats : ℕ := 5
def sunflowers_per_flat : ℕ := 10
def orchid_flats : ℕ := 2
def orchids_per_flat : ℕ := 4
def venus_flytraps : ℕ := 2

def petunia_fertilizer_A : ℕ := 8
def rose_fertilizer_B : ℕ := 3
def sunflower_fertilizer_B : ℕ := 6
def orchid_fertilizer_A : ℕ := 4
def orchid_fertilizer_B : ℕ := 4
def venus_flytrap_fertilizer_C : ℕ := 2

theorem fertilizer_amounts :
  let total_fertilizer_A := petunia_flats * petunias_per_flat * petunia_fertilizer_A +
                            orchid_flats * orchids_per_flat * orchid_fertilizer_A
  let total_fertilizer_B := rose_flats * roses_per_flat * rose_fertilizer_B +
                            sunflower_flats * sunflowers_per_flat * sunflower_fertilizer_B +
                            orchid_flats * orchids_per_flat * orchid_fertilizer_B
  let total_fertilizer_C := venus_flytraps * venus_flytrap_fertilizer_C
  total_fertilizer_A = 288 ∧
  total_fertilizer_B = 386 ∧
  total_fertilizer_C = 4 :=
by sorry

end fertilizer_amounts_l3841_384172


namespace third_group_draw_l3841_384199

/-- Represents a systematic sampling sequence -/
def SystematicSampling (first second : ℕ) : ℕ → ℕ := fun n => first + (n - 1) * (second - first)

/-- Theorem: In a systematic sampling where the first group draws 2 and the second group draws 12,
    the third group will draw 22 -/
theorem third_group_draw (first second : ℕ) (h1 : first = 2) (h2 : second = 12) :
  SystematicSampling first second 3 = 22 := by
  sorry

#eval SystematicSampling 2 12 3

end third_group_draw_l3841_384199


namespace scale_division_l3841_384117

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 125

/-- Represents the length of each part in inches -/
def part_length : ℕ := 25

/-- Theorem stating that the scale is divided into 5 equal parts -/
theorem scale_division :
  scale_length / part_length = 5 := by sorry

end scale_division_l3841_384117


namespace john_spent_110_l3841_384197

/-- The amount of money John spent on wigs for his plays -/
def johnSpent (numPlays : ℕ) (numActs : ℕ) (wigsPerAct : ℕ) (wigCost : ℕ) (sellPrice : ℕ) : ℕ :=
  let totalWigs := numPlays * numActs * wigsPerAct
  let totalCost := totalWigs * wigCost
  let soldWigs := numActs * wigsPerAct
  let moneyBack := soldWigs * sellPrice
  totalCost - moneyBack

/-- Theorem stating that John spent $110 on wigs -/
theorem john_spent_110 :
  johnSpent 3 5 2 5 4 = 110 := by
  sorry

end john_spent_110_l3841_384197


namespace triangle_properties_l3841_384192

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

/-- The theorem to be proved -/
theorem triangle_properties (t : AcuteTriangle)
  (h1 : Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A)
  (h2 : t.c = Real.sqrt 7)
  (h3 : t.a = 2) :
  t.C = π/3 ∧ t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3 / 2 := by
  sorry

end triangle_properties_l3841_384192


namespace inequality_range_theorem_l3841_384179

/-- The function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The theorem statement -/
theorem inequality_range_theorem (m : ℝ) :
  (∀ x ∈ Set.Ici (2/3), f (x/m) - 4*m^2*f x ≤ f (x-1) + 4*f m) →
  m ∈ Set.Iic (-Real.sqrt 3 / 2) ∪ Set.Ici (Real.sqrt 3 / 2) :=
by sorry

end inequality_range_theorem_l3841_384179


namespace visibility_time_correct_l3841_384190

/-- Represents a person walking along a straight path -/
structure Walker where
  speed : ℝ
  initial_x : ℝ
  y : ℝ

/-- Represents a circular building -/
structure Building where
  radius : ℝ

/-- Calculates the time when two walkers can see each other again after being blocked by a building -/
def time_to_see_again (jenny : Walker) (kenny : Walker) (building : Building) : ℝ :=
  sorry

theorem visibility_time_correct :
  let jenny : Walker := { speed := 2, initial_x := -75, y := 150 }
  let kenny : Walker := { speed := 4, initial_x := -75, y := -150 }
  let building : Building := { radius := 75 }
  time_to_see_again jenny kenny building = 48 := by sorry

end visibility_time_correct_l3841_384190


namespace problem_solution_g_minimum_l3841_384158

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a

-- State the theorem
theorem problem_solution :
  (∀ x, f 1 x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∀ m : ℝ, (∃ t : ℝ, f 1 (t/2) ≤ m - f 1 (-t)) ↔ 3.5 ≤ m) := by
  sorry

-- Define the function for the second part
def g (t : ℝ) : ℝ := |t - 1| + |2 * t + 1| + 2

-- State the minimum value theorem
theorem g_minimum : ∀ t : ℝ, g t ≥ 3.5 := by
  sorry

end problem_solution_g_minimum_l3841_384158


namespace simplify_expression_l3841_384111

theorem simplify_expression (y : ℝ) : 4 * y^3 + 8 * y + 6 - (3 - 4 * y^3 - 8 * y) = 8 * y^3 + 16 * y + 3 := by
  sorry

end simplify_expression_l3841_384111


namespace stating_line_triangle_intersection_count_l3841_384141

/-- Represents the number of intersection points between a line and a triangle's boundary. -/
inductive IntersectionCount
  | Zero
  | One
  | Two
  | Infinite

/-- A triangle in a 2D plane. -/
structure Triangle where
  -- Add necessary fields (e.g., vertices) here

/-- A line in a 2D plane. -/
structure Line where
  -- Add necessary fields (e.g., points or coefficients) here

/-- 
  Theorem stating that the number of intersection points between a line and 
  a triangle's boundary is either 0, 1, 2, or infinitely many.
-/
theorem line_triangle_intersection_count 
  (t : Triangle) (l : Line) : 
  ∃ (count : IntersectionCount), 
    (count = IntersectionCount.Zero) ∨ 
    (count = IntersectionCount.One) ∨ 
    (count = IntersectionCount.Two) ∨ 
    (count = IntersectionCount.Infinite) :=
by
  sorry


end stating_line_triangle_intersection_count_l3841_384141


namespace steak_meal_cost_l3841_384189

theorem steak_meal_cost (total_initial : ℚ) (num_steaks : ℕ) 
  (burger_cost : ℚ) (ice_cream_cost : ℚ) (remaining : ℚ) :
  total_initial = 99 →
  num_steaks = 2 →
  burger_cost = 2 * 3.5 →
  ice_cream_cost = 3 * 2 →
  remaining = 38 →
  ∃ (steak_cost : ℚ), 
    total_initial - (num_steaks * steak_cost + burger_cost + ice_cream_cost) = remaining ∧
    steak_cost = 24 :=
by sorry

end steak_meal_cost_l3841_384189


namespace greatest_c_for_no_minus_seven_l3841_384135

theorem greatest_c_for_no_minus_seven : ∃ c : ℤ, 
  (∀ x : ℝ, x^2 + c*x + 20 ≠ -7) ∧
  (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 20 = -7) ∧
  c = 10 := by
sorry

end greatest_c_for_no_minus_seven_l3841_384135


namespace hexagon_coin_rotations_l3841_384168

/-- Represents a configuration of coins on a table -/
structure CoinConfiguration where
  num_coins : Nat
  is_closed_chain : Bool

/-- Represents the motion of a rolling coin -/
structure RollingCoin where
  rotations : Nat

/-- Calculates the number of rotations a coin makes when rolling around a hexagon of coins -/
def calculate_rotations (config : CoinConfiguration) : RollingCoin :=
  sorry

/-- Theorem: A coin rolling around a hexagon of coins makes 4 complete rotations -/
theorem hexagon_coin_rotations :
  ∀ (config : CoinConfiguration),
    config.num_coins = 6 ∧ config.is_closed_chain →
    (calculate_rotations config).rotations = 4 :=
  sorry

end hexagon_coin_rotations_l3841_384168


namespace guest_speaker_payment_l3841_384176

theorem guest_speaker_payment (B : Nat) : 
  B < 10 → (100 * 2 + 10 * B + 5) % 13 = 0 → B = 7 :=
by
  sorry

end guest_speaker_payment_l3841_384176


namespace sufficient_not_necessary_l3841_384165

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x > 0 ∧ y > 0 → x * y > 0) ∧
  (∃ x y, x * y > 0 ∧ ¬(x > 0 ∧ y > 0)) := by
  sorry

end sufficient_not_necessary_l3841_384165


namespace prob_equals_two_thirteenths_l3841_384116

-- Define the deck
def total_cards : ℕ := 52
def num_queens : ℕ := 4
def num_jacks : ℕ := 4

-- Define the event
def prob_two_jacks_or_at_least_one_queen : ℚ :=
  (num_jacks * (num_jacks - 1)) / (total_cards * (total_cards - 1)) +
  (num_queens * (total_cards - num_queens)) / (total_cards * (total_cards - 1)) +
  (num_queens * (num_queens - 1)) / (total_cards * (total_cards - 1))

-- State the theorem
theorem prob_equals_two_thirteenths :
  prob_two_jacks_or_at_least_one_queen = 2 / 13 := by
  sorry

end prob_equals_two_thirteenths_l3841_384116


namespace smallest_block_with_231_hidden_cubes_l3841_384129

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in a block given its dimensions. -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of hidden cubes in a block given its dimensions. -/
def hiddenCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Theorem stating that the smallest possible number of cubes in a block
    with 231 hidden cubes is 384. -/
theorem smallest_block_with_231_hidden_cubes :
  ∃ (d : BlockDimensions),
    hiddenCubes d = 231 ∧
    totalCubes d = 384 ∧
    ∀ (d' : BlockDimensions),
      hiddenCubes d' = 231 → totalCubes d' ≥ 384 :=
by
  sorry

end smallest_block_with_231_hidden_cubes_l3841_384129


namespace imaginary_part_of_reciprocal_i_l3841_384173

theorem imaginary_part_of_reciprocal_i : 
  Complex.im (1 / Complex.I) = -1 := by sorry

end imaginary_part_of_reciprocal_i_l3841_384173


namespace solve_lawyer_problem_l3841_384105

def lawyer_problem (upfront_fee : ℝ) (hourly_rate : ℝ) (court_hours : ℝ) (total_payment : ℝ) : Prop :=
  let court_cost := hourly_rate * court_hours
  let total_cost := upfront_fee + court_cost
  let prep_cost := total_payment - total_cost
  let prep_hours := prep_cost / hourly_rate
  let johns_payment := total_payment / 2
  johns_payment = 4000 ∧ prep_hours / court_hours = 2 / 5

theorem solve_lawyer_problem : 
  lawyer_problem 1000 100 50 8000 :=
by
  sorry

end solve_lawyer_problem_l3841_384105


namespace logarithm_sum_simplification_l3841_384110

theorem logarithm_sum_simplification :
  let a := (1 / (Real.log 3 / Real.log 21 + 1))
  let b := (1 / (Real.log 4 / Real.log 14 + 1))
  let c := (1 / (Real.log 7 / Real.log 9 + 1))
  let d := (1 / (Real.log 11 / Real.log 8 + 1))
  a + b + c + d = 1 := by
  sorry

end logarithm_sum_simplification_l3841_384110


namespace largest_multiple_of_9_less_than_120_l3841_384187

theorem largest_multiple_of_9_less_than_120 : 
  ∀ n : ℕ, n % 9 = 0 → n < 120 → n ≤ 117 :=
sorry

end largest_multiple_of_9_less_than_120_l3841_384187


namespace correct_divisor_l3841_384188

theorem correct_divisor (X D : ℕ) (h1 : X % D = 0) (h2 : X % 12 = 0) (h3 : X / 12 = 56) (h4 : X / D = 32) : D = 21 := by
  sorry

end correct_divisor_l3841_384188


namespace quadratic_roots_sum_l3841_384164

theorem quadratic_roots_sum (a b c : ℝ) (x₁ x₂ : ℂ) : 
  (∃ (s t : ℝ), x₁ = s + t * I ∧ t ≠ 0) →  -- x₁ is a complex number
  (a * x₁^2 + b * x₁ + c = 0) →  -- x₁ is a root of the quadratic equation
  (a * x₂^2 + b * x₂ + c = 0) →  -- x₂ is a root of the quadratic equation
  (∃ (r : ℝ), x₁^2 / x₂ = r) →  -- x₁²/x₂ is real
  let S := 1 + x₁/x₂ + (x₁/x₂)^2 + (x₁/x₂)^4 + (x₁/x₂)^8 + (x₁/x₂)^16 + (x₁/x₂)^32
  S = -2 := by
sorry

end quadratic_roots_sum_l3841_384164


namespace random_selection_probability_l3841_384138

theorem random_selection_probability (a : ℝ) : a > 0 → (∃ m : ℝ, 0 ≤ m ∧ m ≤ a) → (2 / a = 1 / 3) → a = 6 := by
  sorry

end random_selection_probability_l3841_384138


namespace fraction_equation_solution_l3841_384178

theorem fraction_equation_solution (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 20 * b) / (b + 20 * a) = 3) : 
  a / b = 0.33 := by sorry

end fraction_equation_solution_l3841_384178


namespace family_income_change_l3841_384175

theorem family_income_change (initial_average : ℚ) (initial_members : ℕ) 
  (deceased_income : ℚ) (new_members : ℕ) : 
  initial_average = 840 →
  initial_members = 4 →
  deceased_income = 1410 →
  new_members = 3 →
  (initial_average * initial_members - deceased_income) / new_members = 650 := by
  sorry

end family_income_change_l3841_384175


namespace A_intersect_B_l3841_384107

def A : Set ℤ := {1, 2, 3, 4}
def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 3}

theorem A_intersect_B : A ∩ B = {2, 3} := by sorry

end A_intersect_B_l3841_384107


namespace quadratic_inequality_l3841_384181

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x + 5 > 0 ↔ x < 1 ∨ x > 5 := by
  sorry

end quadratic_inequality_l3841_384181


namespace roots_of_polynomials_product_DE_l3841_384148

theorem roots_of_polynomials (r : ℝ) : 
  r^2 = r + 1 → r^6 = 8*r + 5 := by sorry

theorem product_DE : ∃ (D E : ℤ), 
  (∀ (r : ℝ), r^2 = r + 1 → r^6 = D*r + E) ∧ D*E = 40 := by sorry

end roots_of_polynomials_product_DE_l3841_384148


namespace german_team_goals_l3841_384157

def journalist1 (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2 (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3 (x : ℕ) : Prop := x % 2 = 1

def exactlyTwoCorrect (x : ℕ) : Prop :=
  (journalist1 x ∧ journalist2 x ∧ ¬journalist3 x) ∨
  (journalist1 x ∧ ¬journalist2 x ∧ journalist3 x) ∨
  (¬journalist1 x ∧ journalist2 x ∧ journalist3 x)

theorem german_team_goals :
  {x : ℕ | exactlyTwoCorrect x} = {11, 12, 14, 16, 17} := by sorry

end german_team_goals_l3841_384157


namespace circle_area_increase_l3841_384126

theorem circle_area_increase (c : ℝ) (r_increase : ℝ) (h1 : c = 16 * Real.pi) (h2 : r_increase = 2) :
  let r := c / (2 * Real.pi)
  let new_r := r + r_increase
  let area_increase := Real.pi * new_r^2 - Real.pi * r^2
  area_increase = 36 * Real.pi :=
by sorry

end circle_area_increase_l3841_384126


namespace wall_breadth_l3841_384139

/-- Proves that the breadth of a wall with given proportions and volume is 0.4 meters -/
theorem wall_breadth (V h l b : ℝ) (hV : V = 12.8) (hh : h = 5 * b) (hl : l = 8 * h) 
  (hvolume : V = l * b * h) : b = 0.4 := by
  sorry

end wall_breadth_l3841_384139


namespace max_a_value_l3841_384114

/-- Given a quadratic trinomial f(x) = x^2 + ax + b, if for any real x there exists a real y 
    such that f(y) = f(x) + y, then the maximum possible value of a is 1/2. -/
theorem max_a_value (a b : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, (y^2 + a*y + b) = (x^2 + a*x + b) + y) → 
  a ≤ (1/2 : ℝ) ∧ ∃ a₀ : ℝ, a₀ ≤ (1/2 : ℝ) ∧ 
    (∀ x : ℝ, ∃ y : ℝ, (y^2 + a₀*y + b) = (x^2 + a₀*x + b) + y) :=
sorry

end max_a_value_l3841_384114


namespace translate_AB_to_origin_l3841_384104

/-- Given two points A and B in a 2D Cartesian coordinate system, 
    this function returns the coordinates of B after translating 
    the line segment AB so that A coincides with the origin. -/
def translate_to_origin (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

/-- Theorem stating that translating the line segment AB 
    from A(-4, 0) to B(0, 2) so that A coincides with the origin 
    results in B having coordinates (4, 2). -/
theorem translate_AB_to_origin : 
  let A : ℝ × ℝ := (-4, 0)
  let B : ℝ × ℝ := (0, 2)
  translate_to_origin A B = (4, 2) := by
  sorry


end translate_AB_to_origin_l3841_384104


namespace sqrt_3_times_sqrt_12_l3841_384194

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l3841_384194


namespace range_of_m_l3841_384145

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}

-- Define the set C parameterized by m
def C (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - 2*m - 1) < 0}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (C m ⊆ B) ↔ m ∈ Set.Icc (-2) 1 :=
sorry

end range_of_m_l3841_384145


namespace tower_lights_problem_l3841_384198

theorem tower_lights_problem (n : ℕ) (r : ℝ) (sum : ℝ) (h1 : n = 7) (h2 : r = 2) (h3 : sum = 381) :
  let first_term := sum * (r - 1) / (r^n - 1)
  first_term = 3 := by sorry

end tower_lights_problem_l3841_384198


namespace compound_molecular_weight_l3841_384130

/-- Atomic weight of Nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- Atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.01

/-- Atomic weight of Bromine in g/mol -/
def bromine_weight : ℝ := 79.90

/-- Number of Nitrogen atoms in the compound -/
def nitrogen_count : ℕ := 1

/-- Number of Hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 4

/-- Number of Bromine atoms in the compound -/
def bromine_count : ℕ := 1

/-- Molecular weight of the compound -/
def molecular_weight : ℝ := 
  nitrogen_count * nitrogen_weight + 
  hydrogen_count * hydrogen_weight + 
  bromine_count * bromine_weight

theorem compound_molecular_weight : 
  molecular_weight = 97.95 := by sorry

end compound_molecular_weight_l3841_384130


namespace sum_of_number_and_predecessor_l3841_384185

theorem sum_of_number_and_predecessor : ∃ n : ℤ, (6 * n - 2 = 100) ∧ (n + (n - 1) = 33) := by
  sorry

end sum_of_number_and_predecessor_l3841_384185


namespace reciprocal_of_mixed_number_l3841_384149

def mixed_number_to_fraction (whole : ℤ) (numerator : ℤ) (denominator : ℤ) : ℚ :=
  (whole * denominator + numerator) / denominator

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem reciprocal_of_mixed_number :
  let original : ℚ := mixed_number_to_fraction (-1) 2 3
  let recip : ℚ := -3 / 5
  (reciprocal original = recip) ∧ (original * recip = 1) := by sorry

end reciprocal_of_mixed_number_l3841_384149


namespace contrapositive_false_proposition_l3841_384109

theorem contrapositive_false_proposition : 
  ¬(∀ x : ℝ, x ≠ 1 → x^2 ≠ 1) := by sorry

end contrapositive_false_proposition_l3841_384109


namespace sqrt_b_minus_a_l3841_384115

theorem sqrt_b_minus_a (a b : ℝ) 
  (h1 : (2 * a - 1).sqrt = 3)
  (h2 : (3 * a + b - 1)^(1/3) = 3) :
  (b - a).sqrt = 2 * Real.sqrt 2 ∨ (b - a).sqrt = -2 * Real.sqrt 2 := by
  sorry

end sqrt_b_minus_a_l3841_384115


namespace barry_dime_value_l3841_384146

def dime_value : ℕ := 10

theorem barry_dime_value (dan_dimes : ℕ) (barry_dimes : ℕ) : 
  dan_dimes = 52 ∧ 
  dan_dimes = barry_dimes / 2 + 2 →
  barry_dimes * dime_value = 1000 := by
sorry

end barry_dime_value_l3841_384146


namespace pascal_triangle_odd_rows_l3841_384180

/-- Represents a row in Pascal's triangle -/
def PascalRow := List Nat

/-- Generates the nth row of Pascal's triangle -/
def generatePascalRow (n : Nat) : PascalRow := sorry

/-- Checks if a row has all odd numbers except for the ends -/
def isAllOddExceptEnds (row : PascalRow) : Bool := sorry

/-- Counts the number of rows up to n that have all odd numbers except for the ends -/
def countAllOddExceptEndsRows (n : Nat) : Nat := sorry

/-- The main theorem to be proved -/
theorem pascal_triangle_odd_rows :
  countAllOddExceptEndsRows 30 = 3 := by sorry

end pascal_triangle_odd_rows_l3841_384180


namespace fourDigitNumbersTheorem_l3841_384166

/-- Represents the multiset of numbers on the cards -/
def cardNumbers : Multiset ℕ := {1, 1, 1, 2, 2, 3, 4}

/-- Number of cards drawn -/
def cardsDrawn : ℕ := 4

/-- Function to calculate the number of different four-digit numbers -/
def fourDigitNumbersCount (cards : Multiset ℕ) (drawn : ℕ) : ℕ := sorry

/-- Theorem stating that the number of different four-digit numbers is 114 -/
theorem fourDigitNumbersTheorem : fourDigitNumbersCount cardNumbers cardsDrawn = 114 := by
  sorry

end fourDigitNumbersTheorem_l3841_384166


namespace spring_center_max_height_l3841_384177

/-- The maximum height reached by the center of a spring connecting two identical masses -/
theorem spring_center_max_height 
  (m : ℝ) -- mass of each object
  (g : ℝ) -- acceleration due to gravity
  (V₁ V₂ : ℝ) -- initial velocities of upper and lower masses
  (α β : ℝ) -- angles of initial velocities with respect to horizontal
  (h : ℝ) -- maximum height reached by the center of the spring
  (h_pos : 0 < h) -- height is positive
  (m_pos : 0 < m) -- mass is positive
  (g_pos : 0 < g) -- gravity is positive
  : h = (1 / (2 * g)) * ((V₁ * Real.sin β + V₂ * Real.sin α) / 2)^2 :=
by sorry

end spring_center_max_height_l3841_384177


namespace store_discount_l3841_384122

theorem store_discount (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) : 
  initial_discount = 0.40 →
  additional_discount = 0.10 →
  claimed_discount = 0.55 →
  let price_after_first_discount := 1 - initial_discount
  let price_after_second_discount := price_after_first_discount * (1 - additional_discount)
  let actual_discount := 1 - price_after_second_discount
  actual_discount = 0.46 ∧ claimed_discount - actual_discount = 0.09 := by
  sorry

end store_discount_l3841_384122


namespace inequality_comparison_l3841_384132

theorem inequality_comparison (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b + c) (hbc : b < c + a) : 
  let K := a^4 + b^4 + c^4 - 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)
  (K < 0 ↔ c < a + b) ∧ 
  (K = 0 ↔ c = a + b) ∧ 
  (K > 0 ↔ c > a + b) := by
sorry

end inequality_comparison_l3841_384132


namespace coefficient_x_squared_in_product_l3841_384128

-- Define the two polynomials
def p (x : ℝ) : ℝ := 5*x^3 - 3*x^2 + 9*x - 2
def q (x : ℝ) : ℝ := 3*x^2 - 4*x + 2

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p x * q x

-- Theorem: The coefficient of x^2 in the product is -48
theorem coefficient_x_squared_in_product : 
  ∃ (a b c d : ℝ), product = fun x ↦ a*x^3 + (-48)*x^2 + b*x + c + d*x^4 := by
  sorry

end coefficient_x_squared_in_product_l3841_384128


namespace simplify_roots_l3841_384112

theorem simplify_roots : (256 : ℝ)^(1/4) * (625 : ℝ)^(1/2) = 100 := by
  sorry

end simplify_roots_l3841_384112


namespace solution_set_inequality_l3841_384150

theorem solution_set_inequality (x : ℝ) : 
  (x - 2) * (3 - x) > 0 ↔ x ∈ Set.Ioo 2 3 := by sorry

end solution_set_inequality_l3841_384150


namespace complex_multiplication_l3841_384108

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 - i) = 1 + i := by
  sorry

end complex_multiplication_l3841_384108


namespace intersection_of_A_and_B_l3841_384120

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l3841_384120


namespace arithmetic_calculation_l3841_384136

theorem arithmetic_calculation : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := by
  sorry

end arithmetic_calculation_l3841_384136


namespace elizabeth_pencil_purchase_l3841_384118

def pencil_cost : ℕ := 600
def elizabeth_money : ℕ := 500
def borrowed_money : ℕ := 53

theorem elizabeth_pencil_purchase : 
  pencil_cost - (elizabeth_money + borrowed_money) = 47 := by
  sorry

end elizabeth_pencil_purchase_l3841_384118


namespace exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l3841_384183

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := 180 * (n - 2)
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- The exterior angle of a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon_is_45 : 
  exterior_angle_regular_octagon = 45 := by
  sorry

end exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l3841_384183
