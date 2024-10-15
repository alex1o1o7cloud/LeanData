import Mathlib

namespace NUMINAMATH_CALUDE_intersection_at_diametrically_opposite_points_l2852_285253

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure ThreeCircles where
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle
  tangent_point : ℝ × ℝ

def are_touching (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  dist c1.center p = c1.radius ∧
  dist c2.center p = c2.radius ∧
  dist c1.center c2.center = c1.radius + c2.radius

def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  dist c.center p = c.radius

def are_diametrically_opposite (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  dist p1 p2 = 2 * c.radius

theorem intersection_at_diametrically_opposite_points
  (tc : ThreeCircles)
  (h1 : tc.circle1.radius = tc.circle2.radius)
  (h2 : tc.circle2.radius = tc.circle3.radius)
  (h3 : are_touching tc.circle1 tc.circle2 tc.tangent_point)
  (h4 : passes_through tc.circle3 tc.tangent_point) :
  ∃ (p1 p2 : ℝ × ℝ),
    p1 ≠ tc.tangent_point ∧
    p2 ≠ tc.tangent_point ∧
    passes_through tc.circle1 p1 ∧
    passes_through tc.circle2 p2 ∧
    passes_through tc.circle3 p1 ∧
    passes_through tc.circle3 p2 ∧
    are_diametrically_opposite tc.circle3 p1 p2 :=
  sorry

end NUMINAMATH_CALUDE_intersection_at_diametrically_opposite_points_l2852_285253


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l2852_285229

/-- Given a person's monthly income and their spending on transport, 
    prove that their income is $2000 if they have $1900 left after transport expenses. -/
theorem monthly_income_calculation (I : ℝ) : 
  I - 0.05 * I = 1900 → I = 2000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l2852_285229


namespace NUMINAMATH_CALUDE_first_woman_work_time_l2852_285283

/-- Represents the wall-building scenario with women joining at intervals -/
structure WallBuilding where
  /-- Total time to build the wall if all women worked together -/
  totalTime : ℝ
  /-- Number of women -/
  numWomen : ℕ
  /-- Time interval between each woman joining -/
  joinInterval : ℝ
  /-- Time all women work together -/
  allWorkTime : ℝ

/-- The first woman works 5 times as long as the last woman -/
def firstLastRatio (w : WallBuilding) : Prop :=
  w.joinInterval * (w.numWomen - 1) + w.allWorkTime = 5 * w.allWorkTime

/-- The total work done is equivalent to all women working for the total time -/
def totalWorkEquivalence (w : WallBuilding) : Prop :=
  (w.joinInterval * (w.numWomen - 1) / 2 + w.allWorkTime) * w.numWomen = w.totalTime * w.numWomen

/-- Main theorem: The first woman works for 75 hours -/
theorem first_woman_work_time (w : WallBuilding) 
    (h1 : w.totalTime = 45)
    (h2 : firstLastRatio w)
    (h3 : totalWorkEquivalence w) : 
  w.joinInterval * (w.numWomen - 1) + w.allWorkTime = 75 := by
  sorry

#check first_woman_work_time

end NUMINAMATH_CALUDE_first_woman_work_time_l2852_285283


namespace NUMINAMATH_CALUDE_intersecting_line_theorem_l2852_285272

/-- A line passing through (a, 0) intersecting y^2 = 4x at P and Q -/
structure IntersectingLine (a : ℝ) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  line_through_a : P.2 / (P.1 - a) = Q.2 / (Q.1 - a)
  P_on_parabola : P.2^2 = 4 * P.1
  Q_on_parabola : Q.2^2 = 4 * Q.1

/-- The reciprocal sum of squared distances is constant -/
def constant_sum (a : ℝ) :=
  ∃ (k : ℝ), ∀ (l : IntersectingLine a),
    1 / ((l.P.1 - a)^2 + l.P.2^2) + 1 / ((l.Q.1 - a)^2 + l.Q.2^2) = k

/-- If the reciprocal sum of squared distances is constant, then a = 2 -/
theorem intersecting_line_theorem :
  ∀ a : ℝ, constant_sum a → a = 2 := by sorry

end NUMINAMATH_CALUDE_intersecting_line_theorem_l2852_285272


namespace NUMINAMATH_CALUDE_evaluate_expression_l2852_285234

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/2) (hy : y = 1/3) (hz : z = 2) :
  (x^3 * y^4 * z)^2 = 1/104976 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2852_285234


namespace NUMINAMATH_CALUDE_seventh_term_is_eleven_l2852_285287

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the first five terms is 35 -/
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 35
  /-- The sixth term is 10 -/
  sixth_term : a + 5*d = 10

/-- The seventh term of the arithmetic sequence is 11 -/
theorem seventh_term_is_eleven (seq : ArithmeticSequence) : seq.a + 6*seq.d = 11 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_eleven_l2852_285287


namespace NUMINAMATH_CALUDE_cone_from_sector_l2852_285246

/-- Given a 270° sector of a circle with radius 8, prove that the cone formed by aligning
    the straight sides of the sector has a base radius of 6 and a slant height of 8. -/
theorem cone_from_sector (r : ℝ) (angle : ℝ) (h1 : r = 8) (h2 : angle = 270) :
  let sector_arc_length := (angle / 360) * (2 * Real.pi * r)
  let cone_base_circumference := sector_arc_length
  let cone_base_radius := cone_base_circumference / (2 * Real.pi)
  let cone_slant_height := r
  (cone_base_radius = 6) ∧ (cone_slant_height = 8) :=
by sorry

end NUMINAMATH_CALUDE_cone_from_sector_l2852_285246


namespace NUMINAMATH_CALUDE_at_most_two_sides_equal_to_longest_diagonal_l2852_285211

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define the concept of a diagonal in a polygon
def diagonal (p : ConvexPolygon) : Type := sorry

-- Define the length of a side or diagonal
def length {T : Type} (x : T) : ℝ := sorry

-- Define the longest diagonal of a polygon
def longest_diagonal (p : ConvexPolygon) : diagonal p := sorry

-- Define a function that counts the number of sides equal to the longest diagonal
def count_sides_equal_to_longest_diagonal (p : ConvexPolygon) : ℕ := sorry

-- Theorem statement
theorem at_most_two_sides_equal_to_longest_diagonal (p : ConvexPolygon) :
  count_sides_equal_to_longest_diagonal p ≤ 2 := by sorry

end NUMINAMATH_CALUDE_at_most_two_sides_equal_to_longest_diagonal_l2852_285211


namespace NUMINAMATH_CALUDE_distance_to_pole_for_given_point_l2852_285216

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The distance from a point to the pole in polar coordinates -/
def distanceToPole (p : PolarPoint) : ℝ := p.r

theorem distance_to_pole_for_given_point :
  let A : PolarPoint := { r := 3, θ := -4 }
  distanceToPole A = 3 := by sorry

end NUMINAMATH_CALUDE_distance_to_pole_for_given_point_l2852_285216


namespace NUMINAMATH_CALUDE_quadratic_factorization_count_l2852_285250

theorem quadratic_factorization_count :
  ∃! (S : Finset Int), 
    (∀ k ∈ S, ∃ a b c d : Int, 2 * X^2 - k * X + 6 = (a * X + b) * (c * X + d)) ∧
    (∀ k : Int, (∃ a b c d : Int, 2 * X^2 - k * X + 6 = (a * X + b) * (c * X + d)) → k ∈ S) ∧
    Finset.card S = 6 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_factorization_count_l2852_285250


namespace NUMINAMATH_CALUDE_joe_new_average_l2852_285223

/-- Calculates the new average score after dropping the lowest score -/
def new_average (num_tests : ℕ) (original_average : ℚ) (lowest_score : ℚ) : ℚ :=
  (num_tests * original_average - lowest_score) / (num_tests - 1)

/-- Theorem: Given Joe's test scores, his new average after dropping the lowest score is 95 -/
theorem joe_new_average :
  let num_tests : ℕ := 4
  let original_average : ℚ := 90
  let lowest_score : ℚ := 75
  new_average num_tests original_average lowest_score = 95 := by
sorry

end NUMINAMATH_CALUDE_joe_new_average_l2852_285223


namespace NUMINAMATH_CALUDE_granger_grocery_bill_l2852_285222

def spam_price : ℕ := 3
def peanut_butter_price : ℕ := 5
def bread_price : ℕ := 2

def spam_quantity : ℕ := 12
def peanut_butter_quantity : ℕ := 3
def bread_quantity : ℕ := 4

def total_cost : ℕ := spam_price * spam_quantity + 
                      peanut_butter_price * peanut_butter_quantity + 
                      bread_price * bread_quantity

theorem granger_grocery_bill : total_cost = 59 := by
  sorry

end NUMINAMATH_CALUDE_granger_grocery_bill_l2852_285222


namespace NUMINAMATH_CALUDE_sum_opposite_angles_inscribed_quadrilateral_l2852_285233

/-- A quadrilateral WXYZ inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The measure of the angle subtended by arc WZ at the circumference -/
  angle_WZ : ℝ
  /-- The measure of the angle subtended by arc XY at the circumference -/
  angle_XY : ℝ

/-- Theorem: Sum of opposite angles in an inscribed quadrilateral -/
theorem sum_opposite_angles_inscribed_quadrilateral 
  (quad : InscribedQuadrilateral) 
  (h1 : quad.angle_WZ = 40)
  (h2 : quad.angle_XY = 20) :
  ∃ (angle_WXY angle_WZY : ℝ), angle_WXY + angle_WZY = 120 :=
sorry

end NUMINAMATH_CALUDE_sum_opposite_angles_inscribed_quadrilateral_l2852_285233


namespace NUMINAMATH_CALUDE_product_odd_even_is_odd_l2852_285274

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem product_odd_even_is_odd (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) :
  IsOdd (fun x ↦ f x * g x) := by
  sorry


end NUMINAMATH_CALUDE_product_odd_even_is_odd_l2852_285274


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l2852_285244

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l2852_285244


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l2852_285260

theorem cos_squared_minus_sin_squared_15_deg (π : Real) :
  let deg15 : Real := π / 12
  (Real.cos deg15)^2 - (Real.sin deg15)^2 = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l2852_285260


namespace NUMINAMATH_CALUDE_staffing_problem_l2852_285268

def number_of_staffing_ways (total_candidates : ℕ) (qualified_for_first : ℕ) (positions : ℕ) : ℕ :=
  qualified_for_first * (List.range (positions - 1)).foldl (fun acc i => acc * (total_candidates - i - 1)) 1

theorem staffing_problem (total_candidates : ℕ) (qualified_for_first : ℕ) (positions : ℕ) 
  (h1 : total_candidates = 15)
  (h2 : qualified_for_first = 8)
  (h3 : positions = 5)
  (h4 : qualified_for_first ≤ total_candidates) :
  number_of_staffing_ways total_candidates qualified_for_first positions = 17472 := by
  sorry

end NUMINAMATH_CALUDE_staffing_problem_l2852_285268


namespace NUMINAMATH_CALUDE_system_solution_l2852_285284

/-- The system of differential equations -/
def system (t x y : ℝ) : Prop :=
  ∃ (dt dx dy : ℝ), dt / (4*y - 5*x) = dx / (5*t - 3*y) ∧ dx / (5*t - 3*y) = dy / (3*x - 4*t)

/-- The general solution of the system -/
def solution (t x y : ℝ) : Prop :=
  ∃ (C₁ C₂ : ℝ), 3*t + 4*x + 5*y = C₁ ∧ t^2 + x^2 + y^2 = C₂

/-- Theorem stating that the solution satisfies the system -/
theorem system_solution :
  ∀ (t x y : ℝ), system t x y → solution t x y :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2852_285284


namespace NUMINAMATH_CALUDE_jellybeans_in_larger_box_l2852_285298

/-- Given a box with jellybeans and another box with tripled dimensions, 
    calculate the number of jellybeans in the larger box. -/
theorem jellybeans_in_larger_box 
  (small_box_jellybeans : ℕ) 
  (scale_factor : ℕ) 
  (h1 : small_box_jellybeans = 150) 
  (h2 : scale_factor = 3) : 
  (scale_factor ^ 3 : ℕ) * small_box_jellybeans = 4050 := by
  sorry

end NUMINAMATH_CALUDE_jellybeans_in_larger_box_l2852_285298


namespace NUMINAMATH_CALUDE_parabola_properties_l2852_285226

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Predicate to check if a point (x, y) is on the parabola -/
def Parabola.contains_point (p : Parabola) (x y : ℝ) : Prop :=
  p.y_at x = y

theorem parabola_properties (p : Parabola) 
  (h1 : p.contains_point (-1) 3)
  (h2 : p.contains_point 0 0)
  (h3 : p.contains_point 1 (-1))
  (h4 : p.contains_point 2 0)
  (h5 : p.contains_point 3 3) :
  (∃ x_sym : ℝ, x_sym = 1 ∧ ∀ x : ℝ, p.y_at (x_sym - x) = p.y_at (x_sym + x)) ∧ 
  (p.a > 0) ∧
  (∀ x y : ℝ, x < 0 ∧ y < 0 → ¬p.contains_point x y) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l2852_285226


namespace NUMINAMATH_CALUDE_smallest_x_value_l2852_285218

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (4 : ℚ) / 5 = y / (205 + x)) : 
  5 ≤ x ∧ ∃ (y' : ℕ+), (4 : ℚ) / 5 = y' / (205 + 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2852_285218


namespace NUMINAMATH_CALUDE_complex_to_exponential_form_l2852_285201

theorem complex_to_exponential_form (z : ℂ) : z = 2 - 2 * Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_to_exponential_form_l2852_285201


namespace NUMINAMATH_CALUDE_sum_of_five_digit_binary_numbers_l2852_285214

/-- The set of all positive integers with five digits in base 2 -/
def T : Set Nat :=
  {n | 16 ≤ n ∧ n ≤ 31}

/-- The sum of all elements in T -/
def sum_T : Nat :=
  (Finset.range 16).sum (fun i => i + 16)

theorem sum_of_five_digit_binary_numbers :
  sum_T = 248 :=
sorry

end NUMINAMATH_CALUDE_sum_of_five_digit_binary_numbers_l2852_285214


namespace NUMINAMATH_CALUDE_rotation_of_A_to_B_l2852_285296

def rotate90CCW (x y : ℝ) : ℝ × ℝ := (-y, x)

theorem rotation_of_A_to_B :
  let A : ℝ × ℝ := (Real.sqrt 3, 1)
  let B : ℝ × ℝ := rotate90CCW A.1 A.2
  B = (-1, Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_rotation_of_A_to_B_l2852_285296


namespace NUMINAMATH_CALUDE_red_light_runners_estimate_l2852_285203

/-- Represents the survey results and conditions -/
structure SurveyData where
  total_students : ℕ
  yes_answers : ℕ
  id_range : Set ℕ

/-- Calculates the estimated number of people who have run a red light -/
def estimate_red_light_runners (data : SurveyData) : ℕ :=
  2 * (data.yes_answers - data.total_students / 4)

/-- Theorem stating the estimated number of red light runners -/
theorem red_light_runners_estimate (data : SurveyData) 
  (h1 : data.total_students = 800)
  (h2 : data.yes_answers = 240)
  (h3 : data.id_range = {n : ℕ | 1 ≤ n ∧ n ≤ 800}) :
  estimate_red_light_runners data = 80 := by
  sorry

end NUMINAMATH_CALUDE_red_light_runners_estimate_l2852_285203


namespace NUMINAMATH_CALUDE_polynomial_from_root_relations_l2852_285277

theorem polynomial_from_root_relations (α β γ : ℝ) : 
  (∀ x, x^3 - 12*x^2 + 44*x - 46 = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (∃ x₁ x₂ x₃ : ℝ, 
    α = x₁ + x₂ ∧ 
    β = x₁ + x₃ ∧ 
    γ = x₂ + x₃ ∧
    (∀ x, x^3 - 6*x^2 + 8*x - 2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_from_root_relations_l2852_285277


namespace NUMINAMATH_CALUDE_max_x_2009_l2852_285264

def sequence_property (x : ℕ → ℝ) :=
  ∀ n, x n - 2 * x (n + 1) + x (n + 2) ≤ 0

theorem max_x_2009 (x : ℕ → ℝ) 
  (h : sequence_property x)
  (h0 : x 0 = 1)
  (h20 : x 20 = 9)
  (h200 : x 200 = 6) :
  x 2009 ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_x_2009_l2852_285264


namespace NUMINAMATH_CALUDE_max_dinner_income_is_136_80_l2852_285262

/-- Represents the chef's restaurant scenario -/
structure RestaurantScenario where
  -- Lunch meals
  pasta_lunch : ℕ
  chicken_lunch : ℕ
  fish_lunch : ℕ
  -- Prices
  pasta_price : ℚ
  chicken_price : ℚ
  fish_price : ℚ
  -- Sold during lunch
  pasta_sold_lunch : ℕ
  chicken_sold_lunch : ℕ
  fish_sold_lunch : ℕ
  -- Dinner meals
  pasta_dinner : ℕ
  chicken_dinner : ℕ
  fish_dinner : ℕ
  -- Discount rate
  discount_rate : ℚ

/-- Calculates the maximum total income during dinner -/
def max_dinner_income (s : RestaurantScenario) : ℚ :=
  let pasta_unsold := s.pasta_lunch - s.pasta_sold_lunch
  let chicken_unsold := s.chicken_lunch - s.chicken_sold_lunch
  let fish_unsold := s.fish_lunch - s.fish_sold_lunch
  let discounted_pasta_price := s.pasta_price * (1 - s.discount_rate)
  let discounted_chicken_price := s.chicken_price * (1 - s.discount_rate)
  let discounted_fish_price := s.fish_price * (1 - s.discount_rate)
  (s.pasta_dinner * s.pasta_price + pasta_unsold * discounted_pasta_price) +
  (s.chicken_dinner * s.chicken_price + chicken_unsold * discounted_chicken_price) +
  (s.fish_dinner * s.fish_price + fish_unsold * discounted_fish_price)

/-- The chef's restaurant scenario -/
def chef_scenario : RestaurantScenario := {
  pasta_lunch := 8
  chicken_lunch := 5
  fish_lunch := 4
  pasta_price := 12
  chicken_price := 15
  fish_price := 18
  pasta_sold_lunch := 6
  chicken_sold_lunch := 3
  fish_sold_lunch := 3
  pasta_dinner := 2
  chicken_dinner := 2
  fish_dinner := 1
  discount_rate := 1/10
}

/-- Theorem stating the maximum total income during dinner -/
theorem max_dinner_income_is_136_80 :
  max_dinner_income chef_scenario = 136.8 := by sorry


end NUMINAMATH_CALUDE_max_dinner_income_is_136_80_l2852_285262


namespace NUMINAMATH_CALUDE_square_symbol_function_l2852_285225

/-- Represents the possible functions of symbols in a program flowchart -/
inductive FlowchartSymbolFunction
  | Output
  | Assignment
  | Decision
  | EndOfAlgorithm
  | Calculation

/-- Represents a symbol in a program flowchart -/
structure FlowchartSymbol where
  shape : String
  function : FlowchartSymbolFunction

/-- The square symbol in a program flowchart -/
def squareSymbol : FlowchartSymbol :=
  { shape := "□", function := FlowchartSymbolFunction.Assignment }

/-- Theorem stating the function of the square symbol in a program flowchart -/
theorem square_symbol_function :
  (squareSymbol.function = FlowchartSymbolFunction.Assignment) ∨
  (squareSymbol.function = FlowchartSymbolFunction.Calculation) :=
by sorry

end NUMINAMATH_CALUDE_square_symbol_function_l2852_285225


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_l2852_285259

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_remainder (n : ℕ) (h : n = 10) : sum_factorials n % 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_l2852_285259


namespace NUMINAMATH_CALUDE_sticker_redistribution_l2852_285237

theorem sticker_redistribution (noah emma liam : ℕ) 
  (h1 : emma = 3 * noah) 
  (h2 : liam = 4 * emma) : 
  (7 : ℚ) / 36 = (liam - (liam + emma + noah) / 3) / liam := by
  sorry

end NUMINAMATH_CALUDE_sticker_redistribution_l2852_285237


namespace NUMINAMATH_CALUDE_closest_multiple_of_18_to_2500_l2852_285293

/-- The multiple of 18 closest to 2500 is 2502 -/
theorem closest_multiple_of_18_to_2500 :
  ∀ n : ℤ, 18 ∣ n → |n - 2500| ≥ |2502 - 2500| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_multiple_of_18_to_2500_l2852_285293


namespace NUMINAMATH_CALUDE_extreme_point_property_l2852_285271

def f (a b x : ℝ) : ℝ := x^3 - a*x - b

theorem extreme_point_property (a b x₀ x₁ : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, x ≠ x₀ ∧ |x - x₀| < ε → f a b x ≠ f a b x₀) →
  x₁ ≠ x₀ →
  f a b x₁ = f a b x₀ →
  x₁ + 2*x₀ = 0 := by
sorry

end NUMINAMATH_CALUDE_extreme_point_property_l2852_285271


namespace NUMINAMATH_CALUDE_closest_ratio_to_one_l2852_285275

/-- Represents the admission fee structure and total collection --/
structure AdmissionData where
  adult_fee : ℕ
  child_fee : ℕ
  total_collection : ℕ

/-- Represents a valid combination of adults and children --/
structure Attendance where
  adults : ℕ
  children : ℕ

/-- Checks if the given attendance satisfies the total collection --/
def is_valid_attendance (data : AdmissionData) (att : Attendance) : Prop :=
  data.adult_fee * att.adults + data.child_fee * att.children = data.total_collection

/-- Calculates the absolute difference between the ratio and 1 --/
def ratio_diff_from_one (att : Attendance) : ℚ :=
  |att.adults / att.children - 1|

theorem closest_ratio_to_one (data : AdmissionData) :
  data.adult_fee = 25 →
  data.child_fee = 12 →
  data.total_collection = 1950 →
  ∃ (best : Attendance),
    is_valid_attendance data best ∧
    best.adults > 0 ∧
    best.children > 0 ∧
    ∀ (att : Attendance),
      is_valid_attendance data att →
      att.adults > 0 →
      att.children > 0 →
      ratio_diff_from_one best ≤ ratio_diff_from_one att ∧
      (best.adults = 54 ∧ best.children = 50) :=
sorry

end NUMINAMATH_CALUDE_closest_ratio_to_one_l2852_285275


namespace NUMINAMATH_CALUDE_peter_pizza_fraction_l2852_285236

theorem peter_pizza_fraction (total_slices : ℕ) (peter_alone : ℕ) (shared_paul : ℚ) (shared_patty : ℚ) :
  total_slices = 16 →
  peter_alone = 3 →
  shared_paul = 1 / 2 →
  shared_patty = 1 / 2 →
  (peter_alone : ℚ) / total_slices + shared_paul / total_slices + shared_patty / total_slices = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_peter_pizza_fraction_l2852_285236


namespace NUMINAMATH_CALUDE_probability_divisible_by_3_l2852_285232

/-- The set of digits to choose from -/
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

/-- A four-digit number formed from the given set of digits -/
structure FourDigitNumber where
  d₁ : ℕ
  d₂ : ℕ
  d₃ : ℕ
  d₄ : ℕ
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  h₅ : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄
  h₆ : d₁ ≠ 0

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : ℕ :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- A number is divisible by 3 if the sum of its digits is divisible by 3 -/
def FourDigitNumber.divisibleBy3 (n : FourDigitNumber) : Prop :=
  (n.d₁ + n.d₂ + n.d₃ + n.d₄) % 3 = 0

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

/-- The set of all four-digit numbers divisible by 3 -/
def divisibleBy3Numbers : Finset FourDigitNumber :=
  sorry

/-- The main theorem -/
theorem probability_divisible_by_3 :
  (Finset.card divisibleBy3Numbers : ℚ) / (Finset.card allFourDigitNumbers) = 8 / 15 :=
sorry

end NUMINAMATH_CALUDE_probability_divisible_by_3_l2852_285232


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2852_285252

/-- Given an ellipse defined by 16(x-2)^2 + 4y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 16 * (x - 2)^2 + 4 * y^2 = 64 ↔ 
      (x - 2)^2 / 4 + y^2 / 16 = 1) →
    (C.1 - 2)^2 / 4 + C.2^2 / 16 = 1 →
    (D.1 - 2)^2 / 4 + D.2^2 / 16 = 1 →
    C.2 = 4 ∨ C.2 = -4 →
    D.1 = 4 ∨ D.1 = 0 →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2852_285252


namespace NUMINAMATH_CALUDE_bed_weight_problem_l2852_285257

theorem bed_weight_problem (single_bed_weight : ℝ) (double_bed_weight : ℝ) : 
  (5 * single_bed_weight = 50) →
  (double_bed_weight = single_bed_weight + 10) →
  (2 * single_bed_weight + 4 * double_bed_weight = 100) :=
by
  sorry

end NUMINAMATH_CALUDE_bed_weight_problem_l2852_285257


namespace NUMINAMATH_CALUDE_fraction_inequality_l2852_285282

theorem fraction_inequality (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hab : a < b) :
  (b + m) / (a + m) < b / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2852_285282


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2852_285204

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (3 - 4*i) / (1 + i)
  Complex.im z = -7/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2852_285204


namespace NUMINAMATH_CALUDE_deceased_member_income_l2852_285210

theorem deceased_member_income
  (initial_members : ℕ)
  (final_members : ℕ)
  (initial_average : ℚ)
  (final_average : ℚ)
  (h1 : initial_members = 4)
  (h2 : final_members = 3)
  (h3 : initial_average = 782)
  (h4 : final_average = 650)
  : (initial_members : ℚ) * initial_average - (final_members : ℚ) * final_average = 1178 := by
  sorry

end NUMINAMATH_CALUDE_deceased_member_income_l2852_285210


namespace NUMINAMATH_CALUDE_tub_volume_ratio_l2852_285248

theorem tub_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (2/3 : ℝ) * V₂ →
  V₁ / V₂ = 8/9 := by
sorry

end NUMINAMATH_CALUDE_tub_volume_ratio_l2852_285248


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l2852_285273

/-- The area of a rectangle inscribed in a triangle -/
theorem inscribed_rectangle_area (b h x : ℝ) (hb : b > 0) (hh : h > 0) (hx : x > 0) (hxh : x < h) :
  let triangle_area := (1/2) * b * h
  let rectangle_base := b * (1 - x/h)
  let rectangle_area := x * rectangle_base
  rectangle_area = (b * x / h) * (h - x) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l2852_285273


namespace NUMINAMATH_CALUDE_simplify_power_of_power_l2852_285254

theorem simplify_power_of_power (x : ℝ) : (2 * x^3)^3 = 8 * x^9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_of_power_l2852_285254


namespace NUMINAMATH_CALUDE_unique_function_solution_l2852_285243

theorem unique_function_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, 2 * f x + f (1 - x) = x^2) ↔ (∀ x : ℝ, f x = (1/3) * (x^2 + 2*x - 1)) :=
sorry

end NUMINAMATH_CALUDE_unique_function_solution_l2852_285243


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2852_285200

/-- A geometric sequence with specific conditions -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 1 + a 3 = 5 ∧
  a 2 + a 4 = 10

/-- The sum of the 6th and 8th terms equals 160 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 6 + a 8 = 160 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2852_285200


namespace NUMINAMATH_CALUDE_fruit_sales_theorem_l2852_285220

/-- Represents the pricing and sales model of a fruit in Huimin Fresh Supermarket -/
structure FruitSalesModel where
  cost_price : ℝ
  initial_selling_price : ℝ
  initial_daily_sales : ℝ
  price_reduction_rate : ℝ
  sales_increase_rate : ℝ

/-- Calculates the daily profit given a price reduction -/
def daily_profit (model : FruitSalesModel) (price_reduction : ℝ) : ℝ :=
  (model.initial_selling_price - price_reduction - model.cost_price) *
  (model.initial_daily_sales + model.sales_increase_rate * price_reduction)

/-- The main theorem about the fruit sales model -/
theorem fruit_sales_theorem (model : FruitSalesModel) 
  (h_cost : model.cost_price = 20)
  (h_initial_price : model.initial_selling_price = 40)
  (h_initial_sales : model.initial_daily_sales = 20)
  (h_price_reduction : model.price_reduction_rate = 1)
  (h_sales_increase : model.sales_increase_rate = 2) :
  (∃ (x : ℝ), x = 10 ∧ daily_profit model x = daily_profit model 0) ∧
  (¬ ∃ (y : ℝ), daily_profit model y = 460) := by
  sorry


end NUMINAMATH_CALUDE_fruit_sales_theorem_l2852_285220


namespace NUMINAMATH_CALUDE_remainder_six_divisor_count_l2852_285213

theorem remainder_six_divisor_count : 
  ∃! (n : ℕ), n > 6 ∧ 67 % n = 6 :=
sorry

end NUMINAMATH_CALUDE_remainder_six_divisor_count_l2852_285213


namespace NUMINAMATH_CALUDE_fewer_cards_l2852_285235

/-- The number of soccer cards Chris has -/
def chris_cards : ℕ := 18

/-- The number of soccer cards Charlie has -/
def charlie_cards : ℕ := 32

/-- The difference in the number of cards between Charlie and Chris -/
def card_difference : ℕ := charlie_cards - chris_cards

theorem fewer_cards : card_difference = 14 := by
  sorry

end NUMINAMATH_CALUDE_fewer_cards_l2852_285235


namespace NUMINAMATH_CALUDE_no_solution_exists_l2852_285217

theorem no_solution_exists : ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (3 / a + 4 / b = 12 / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2852_285217


namespace NUMINAMATH_CALUDE_acid_mixture_water_volume_l2852_285267

/-- Represents the composition of a mixture --/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- Calculates the total volume of a mixture --/
def totalVolume (m : Mixture) : ℝ := m.acid + m.water

/-- Represents the problem setup --/
structure AcidMixtureProblem where
  initialMixture : Mixture
  pureAcidVolume : ℝ
  finalWaterPercentage : ℝ

/-- Calculates the final mixture composition --/
def finalMixture (problem : AcidMixtureProblem) (addedVolume : ℝ) : Mixture :=
  { acid := problem.pureAcidVolume + addedVolume * problem.initialMixture.acid,
    water := addedVolume * problem.initialMixture.water }

/-- The main theorem to prove --/
theorem acid_mixture_water_volume
  (problem : AcidMixtureProblem)
  (h1 : problem.initialMixture.acid = 0.1)
  (h2 : problem.initialMixture.water = 0.9)
  (h3 : problem.pureAcidVolume = 5)
  (h4 : problem.finalWaterPercentage = 0.4) :
  ∃ (addedVolume : ℝ),
    let finalMix := finalMixture problem addedVolume
    finalMix.water / totalVolume finalMix = problem.finalWaterPercentage ∧
    finalMix.water = 3.6 := by
  sorry


end NUMINAMATH_CALUDE_acid_mixture_water_volume_l2852_285267


namespace NUMINAMATH_CALUDE_fixed_order_queue_arrangement_l2852_285288

def queue_arrangements (n : ℕ) (k : ℕ) : Prop :=
  n ≥ k ∧ (n - k).factorial * k.factorial * (n.choose k) = 20

theorem fixed_order_queue_arrangement :
  queue_arrangements 5 3 :=
sorry

end NUMINAMATH_CALUDE_fixed_order_queue_arrangement_l2852_285288


namespace NUMINAMATH_CALUDE_expression_zero_iff_x_neg_two_l2852_285292

theorem expression_zero_iff_x_neg_two (x : ℝ) :
  (x^2 - 4) / (4*x - 8) = 0 ↔ x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_zero_iff_x_neg_two_l2852_285292


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2852_285228

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (carbon_count : ℕ) (hydrogen_count : ℕ) (oxygen_count : ℕ) (nitrogen_count : ℕ) (sulfur_count : ℕ) : ℝ :=
  let carbon_weight : ℝ := 12.01
  let hydrogen_weight : ℝ := 1.008
  let oxygen_weight : ℝ := 16.00
  let nitrogen_weight : ℝ := 14.01
  let sulfur_weight : ℝ := 32.07
  carbon_count * carbon_weight + hydrogen_count * hydrogen_weight + oxygen_count * oxygen_weight + 
  nitrogen_count * nitrogen_weight + sulfur_count * sulfur_weight

/-- Theorem stating that the molecular weight of the given compound is approximately 323.46 g/mol -/
theorem compound_molecular_weight : 
  ∃ ε > 0, |molecular_weight 10 15 4 2 3 - 323.46| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2852_285228


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l2852_285221

theorem complex_subtraction_simplification :
  (-3 - 2*I) - (1 + 4*I) = -4 - 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l2852_285221


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l2852_285261

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  collinear a b → x = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l2852_285261


namespace NUMINAMATH_CALUDE_red_ball_certain_event_l2852_285258

/-- Represents a bag of balls -/
structure Bag where
  balls : Set Color

/-- Represents the color of a ball -/
inductive Color where
  | Red

/-- Represents an event -/
structure Event where
  occurs : Prop

/-- Defines a certain event -/
def CertainEvent (e : Event) : Prop :=
  e.occurs = True

/-- Defines the event of drawing a ball from a bag -/
def DrawBall (b : Bag) (c : Color) : Event where
  occurs := c ∈ b.balls

/-- Theorem: Drawing a red ball from a bag containing only red balls is a certain event -/
theorem red_ball_certain_event (b : Bag) (h : b.balls = {Color.Red}) :
  CertainEvent (DrawBall b Color.Red) := by
  sorry

end NUMINAMATH_CALUDE_red_ball_certain_event_l2852_285258


namespace NUMINAMATH_CALUDE_cube_opposite_face_l2852_285231

structure Cube where
  faces : Finset Char
  adjacent : Char → Char → Prop

def opposite (c : Cube) (f1 f2 : Char) : Prop :=
  f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ ¬c.adjacent f1 f2 ∧
  ∀ f3 ∈ c.faces, f3 ≠ f1 ∧ f3 ≠ f2 → (c.adjacent f1 f3 ↔ ¬c.adjacent f2 f3)

theorem cube_opposite_face (c : Cube) :
  c.faces = {'x', 'A', 'B', 'C', 'D', 'E', 'F'} →
  c.adjacent 'x' 'A' →
  c.adjacent 'x' 'D' →
  c.adjacent 'x' 'F' →
  c.adjacent 'E' 'D' →
  ¬c.adjacent 'x' 'E' →
  opposite c 'x' 'B' := by
  sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l2852_285231


namespace NUMINAMATH_CALUDE_forest_tree_ratio_l2852_285286

/-- Proves the ratio of trees after Monday to initial trees is 3:1 --/
theorem forest_tree_ratio : 
  ∀ (initial_trees monday_trees : ℕ),
    initial_trees = 30 →
    monday_trees + (monday_trees / 3) = 80 →
    (initial_trees + monday_trees) / initial_trees = 3 := by
  sorry

end NUMINAMATH_CALUDE_forest_tree_ratio_l2852_285286


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2852_285219

/-- The eccentricity of a hyperbola with equation y²/4 - x² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  ∃ (e : ℝ), e = (Real.sqrt 5) / 2 ∧ 
  ∀ (x y : ℝ), y^2 / 4 - x^2 = 1 → 
  e = Real.sqrt ((y^2 / 4) + x^2) / (y / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2852_285219


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_product_is_222_l2852_285239

/-- The repeating decimal 0.018018018... as a real number -/
def repeating_decimal : ℚ := 18 / 999

/-- The fraction 2/111 -/
def target_fraction : ℚ := 2 / 111

/-- Theorem stating that the repeating decimal 0.018018018... is equal to 2/111 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

/-- The product of the numerator and denominator of the fraction -/
def numerator_denominator_product : ℕ := 2 * 111

/-- Theorem stating that the product of the numerator and denominator is 222 -/
theorem product_is_222 : numerator_denominator_product = 222 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_product_is_222_l2852_285239


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2852_285238

/-- The function g(x) -/
def g (a b x : ℝ) : ℝ := (a * x - 2) * (x + b)

/-- The theorem stating that if g(x) > 0 has solution set (-1, 2), then a + b = -4 -/
theorem sum_of_a_and_b (a b : ℝ) :
  (∀ x, g a b x > 0 ↔ -1 < x ∧ x < 2) →
  a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2852_285238


namespace NUMINAMATH_CALUDE_martian_age_conversion_l2852_285215

/-- Converts a single digit from base 9 to base 10 -/
def base9ToBase10Digit (d : Nat) : Nat := d

/-- Converts a 3-digit number from base 9 to base 10 -/
def base9ToBase10 (d₂ d₁ d₀ : Nat) : Nat :=
  base9ToBase10Digit d₂ * 9^2 + base9ToBase10Digit d₁ * 9^1 + base9ToBase10Digit d₀ * 9^0

/-- The age of the Martian robot's manufacturing facility in base 9 -/
def martianAge : Nat := 376

theorem martian_age_conversion :
  base9ToBase10 3 7 6 = 312 := by
  sorry

end NUMINAMATH_CALUDE_martian_age_conversion_l2852_285215


namespace NUMINAMATH_CALUDE_fraction_subtraction_division_l2852_285294

theorem fraction_subtraction_division : 
  (10 : ℚ) / 5 - (10 : ℚ) / 2 / ((2 : ℚ) / 5) = -21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_division_l2852_285294


namespace NUMINAMATH_CALUDE_triangle_sin_A_values_l2852_285251

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  AC : Real
  BC : Real

-- Define the theorem
theorem triangle_sin_A_values
  (abc : Triangle)
  (non_obtuse : abc.A ≤ 90 ∧ abc.B ≤ 90 ∧ abc.C ≤ 90)
  (ab_gt_ac : abc.AB > abc.AC)
  (angle_b_45 : abc.B = 45)
  (O : Real) -- Circumcenter
  (I : Real) -- Incenter
  (oi_relation : Real.sqrt 2 * (O - I) = abc.AB - abc.AC) :
  Real.sin abc.A = Real.sqrt 2 / 2 ∨ Real.sin abc.A = Real.sqrt (Real.sqrt 2 - 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sin_A_values_l2852_285251


namespace NUMINAMATH_CALUDE_orange_harvest_l2852_285209

/-- The number of sacks of oranges kept after a given number of harvest days -/
def sacksKept (harvestedPerDay discardedPerDay harvestDays : ℕ) : ℕ :=
  (harvestedPerDay - discardedPerDay) * harvestDays

/-- Theorem: The number of sacks of oranges kept after 51 days of harvest is 153 -/
theorem orange_harvest :
  sacksKept 74 71 51 = 153 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_l2852_285209


namespace NUMINAMATH_CALUDE_lanas_tickets_l2852_285289

/-- The number of tickets Lana bought for herself and friends -/
def tickets_for_friends : ℕ := sorry

/-- The cost of each ticket in dollars -/
def ticket_cost : ℕ := 6

/-- The number of extra tickets Lana bought -/
def extra_tickets : ℕ := 2

/-- The total amount Lana spent in dollars -/
def total_spent : ℕ := 60

theorem lanas_tickets : 
  (tickets_for_friends + extra_tickets) * ticket_cost = total_spent ∧ 
  tickets_for_friends = 8 := by
  sorry

end NUMINAMATH_CALUDE_lanas_tickets_l2852_285289


namespace NUMINAMATH_CALUDE_box_balls_problem_l2852_285241

theorem box_balls_problem (B X : ℕ) (h1 : B = 57) (h2 : B - 44 = X - B) : X = 70 := by
  sorry

end NUMINAMATH_CALUDE_box_balls_problem_l2852_285241


namespace NUMINAMATH_CALUDE_area_BQW_is_48_l2852_285280

/-- Rectangle ABCD with specific measurements and areas -/
structure Rectangle where
  AB : ℝ
  AZ : ℝ
  WC : ℝ
  area_ZWCD : ℝ
  h : AB = 16
  h' : AZ = 8
  h'' : WC = 8
  h''' : area_ZWCD = 160

/-- The area of triangle BQW in the given rectangle -/
def area_BQW (r : Rectangle) : ℝ := 48

/-- Theorem stating that the area of triangle BQW is 48 square units -/
theorem area_BQW_is_48 (r : Rectangle) : area_BQW r = 48 := by
  sorry

end NUMINAMATH_CALUDE_area_BQW_is_48_l2852_285280


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2852_285269

theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 1) ↔ (∃ x y : ℝ, y = m * x + 2 ∧ y = (3 * m - 2) * x + 5) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2852_285269


namespace NUMINAMATH_CALUDE_sqrt_two_subtraction_l2852_285291

theorem sqrt_two_subtraction : 4 * Real.sqrt 2 - Real.sqrt 2 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_subtraction_l2852_285291


namespace NUMINAMATH_CALUDE_yellow_red_difference_after_border_l2852_285208

/-- Represents a hexagonal figure with red and yellow tiles -/
structure HexagonalFigure where
  red_tiles : ℕ
  yellow_tiles : ℕ

/-- Calculates the number of tiles needed for a border around a hexagonal figure -/
def border_tiles : ℕ := 18

/-- Adds a border of yellow tiles to a hexagonal figure -/
def add_border (figure : HexagonalFigure) : HexagonalFigure :=
  { red_tiles := figure.red_tiles,
    yellow_tiles := figure.yellow_tiles + border_tiles }

/-- The initial hexagonal figure -/
def initial_figure : HexagonalFigure :=
  { red_tiles := 15, yellow_tiles := 9 }

/-- Theorem: The difference between yellow and red tiles after adding a border is 12 -/
theorem yellow_red_difference_after_border :
  let new_figure := add_border initial_figure
  new_figure.yellow_tiles - new_figure.red_tiles = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_red_difference_after_border_l2852_285208


namespace NUMINAMATH_CALUDE_oil_depth_calculation_l2852_285263

/-- Represents a right cylindrical tank -/
structure Tank where
  height : ℝ
  base_diameter : ℝ

/-- Calculates the volume of oil in the tank when lying on its side -/
def oil_volume_side (tank : Tank) (depth : ℝ) : ℝ :=
  sorry

/-- Calculates the depth of oil when the tank is standing upright -/
def oil_depth_upright (tank : Tank) (volume : ℝ) : ℝ :=
  sorry

/-- Theorem: For the given tank dimensions and side oil depth, 
    the upright oil depth is approximately 2.2 feet -/
theorem oil_depth_calculation (tank : Tank) (side_depth : ℝ) :
  tank.height = 20 →
  tank.base_diameter = 6 →
  side_depth = 4 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
    |oil_depth_upright tank (oil_volume_side tank side_depth) - 2.2| < ε :=
sorry

end NUMINAMATH_CALUDE_oil_depth_calculation_l2852_285263


namespace NUMINAMATH_CALUDE_sea_turtle_collection_age_difference_l2852_285265

/-- Converts a number from base 8 to base 10 -/
def octalToDecimal (n : ℕ) : ℕ :=
  (n % 10) + 8 * ((n / 10) % 10) + 64 * (n / 100)

theorem sea_turtle_collection_age_difference : 
  octalToDecimal 724 - octalToDecimal 560 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sea_turtle_collection_age_difference_l2852_285265


namespace NUMINAMATH_CALUDE_best_fitting_highest_r_squared_l2852_285270

/-- Represents a regression model with its coefficient of determination -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared : 0 ≤ r_squared ∧ r_squared ≤ 1

/-- Determines if a model is the best-fitting among a list of models -/
def is_best_fitting (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

theorem best_fitting_highest_r_squared 
  (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.r_squared = 0.98)
  (h2 : model2.r_squared = 0.80)
  (h3 : model3.r_squared = 0.50)
  (h4 : model4.r_squared = 0.25) :
  is_best_fitting model1 [model1, model2, model3, model4] :=
by sorry

end NUMINAMATH_CALUDE_best_fitting_highest_r_squared_l2852_285270


namespace NUMINAMATH_CALUDE_cube_difference_negative_l2852_285245

theorem cube_difference_negative {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) : a^3 - b^3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_negative_l2852_285245


namespace NUMINAMATH_CALUDE_triangle_3_5_7_l2852_285255

/-- A set of three line segments can form a triangle if and only if the sum of the lengths of any two sides is greater than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Prove that the set of line segments (3cm, 5cm, 7cm) can form a triangle. -/
theorem triangle_3_5_7 : can_form_triangle 3 5 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_3_5_7_l2852_285255


namespace NUMINAMATH_CALUDE_joyce_apples_l2852_285276

/-- The number of apples Joyce gave to Larry -/
def apples_given : ℕ := 52

/-- The number of apples Joyce had left -/
def apples_left : ℕ := 23

/-- The total number of apples Joyce started with -/
def initial_apples : ℕ := apples_given + apples_left

theorem joyce_apples : initial_apples = 75 := by
  sorry

end NUMINAMATH_CALUDE_joyce_apples_l2852_285276


namespace NUMINAMATH_CALUDE_binomial_25_5_l2852_285242

theorem binomial_25_5 (h1 : (23 : ℕ).choose 3 = 1771)
                      (h2 : (23 : ℕ).choose 4 = 8855)
                      (h3 : (23 : ℕ).choose 5 = 33649) :
  (25 : ℕ).choose 5 = 53130 := by
  sorry

end NUMINAMATH_CALUDE_binomial_25_5_l2852_285242


namespace NUMINAMATH_CALUDE_exponent_addition_l2852_285297

theorem exponent_addition (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l2852_285297


namespace NUMINAMATH_CALUDE_mango_per_tree_l2852_285202

theorem mango_per_tree (papaya_trees : ℕ) (mango_trees : ℕ) (papaya_per_tree : ℕ) (total_fruits : ℕ)
  (h1 : papaya_trees = 2)
  (h2 : mango_trees = 3)
  (h3 : papaya_per_tree = 10)
  (h4 : total_fruits = 80) :
  (total_fruits - papaya_trees * papaya_per_tree) / mango_trees = 20 := by
  sorry

end NUMINAMATH_CALUDE_mango_per_tree_l2852_285202


namespace NUMINAMATH_CALUDE_bruce_calculators_l2852_285230

-- Define the given conditions
def total_money : ℕ := 200
def crayon_cost : ℕ := 5
def book_cost : ℕ := 5
def calculator_cost : ℕ := 5
def bag_cost : ℕ := 10
def crayon_packs : ℕ := 5
def books : ℕ := 10
def bags : ℕ := 11

-- Define the theorem
theorem bruce_calculators :
  let crayon_total := crayon_cost * crayon_packs
  let book_total := book_cost * books
  let remaining_after_books := total_money - (crayon_total + book_total)
  let bag_total := bag_cost * bags
  let remaining_for_calculators := remaining_after_books - bag_total
  remaining_for_calculators / calculator_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_bruce_calculators_l2852_285230


namespace NUMINAMATH_CALUDE_inequality_proof_l2852_285240

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2852_285240


namespace NUMINAMATH_CALUDE_cost_of_seven_cds_cost_of_seven_cds_is_112_l2852_285205

/-- The cost of seven CDs given that two identical CDs cost $32 -/
theorem cost_of_seven_cds : ℝ :=
  let cost_of_two : ℝ := 32
  let cost_of_one : ℝ := cost_of_two / 2
  7 * cost_of_one

/-- Proof that the cost of seven CDs is $112 -/
theorem cost_of_seven_cds_is_112 : cost_of_seven_cds = 112 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_seven_cds_cost_of_seven_cds_is_112_l2852_285205


namespace NUMINAMATH_CALUDE_puppies_sold_l2852_285285

/-- Given a pet store scenario, prove the number of puppies sold. -/
theorem puppies_sold (initial_puppies : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) :
  initial_puppies ≥ puppies_per_cage * cages_used →
  initial_puppies - (puppies_per_cage * cages_used) =
    initial_puppies - puppies_per_cage * cages_used :=
by
  sorry

#check puppies_sold 102 9 9

end NUMINAMATH_CALUDE_puppies_sold_l2852_285285


namespace NUMINAMATH_CALUDE_pebble_collection_sum_l2852_285281

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem pebble_collection_sum : geometric_sum 2 2 10 = 2046 := by
  sorry

end NUMINAMATH_CALUDE_pebble_collection_sum_l2852_285281


namespace NUMINAMATH_CALUDE_permutations_of_five_l2852_285247

theorem permutations_of_five (d : ℕ) : d = Nat.factorial 5 → d = 120 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_five_l2852_285247


namespace NUMINAMATH_CALUDE_red_probability_both_jars_l2852_285206

/-- Represents a jar containing buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents the state of both jars -/
structure JarState :=
  (jarA : Jar)
  (jarB : Jar)

/-- Initial state of the jars -/
def initialState : JarState :=
  { jarA := { red := 6, blue := 10 },
    jarB := { red := 2, blue := 3 } }

/-- Function to transfer buttons between jars -/
def transfer (s : JarState) (n : ℕ) : JarState :=
  { jarA := { red := s.jarA.red - n, blue := s.jarA.blue - n },
    jarB := { red := s.jarB.red + n, blue := s.jarB.blue + n } }

/-- Final state after transfer -/
def finalState : JarState :=
  transfer initialState 3

/-- Probability of selecting a red button from a jar -/
def redProbability (j : Jar) : ℚ :=
  j.red / (j.red + j.blue)

/-- Theorem: The probability of selecting red buttons from both jars is 3/22 -/
theorem red_probability_both_jars :
  redProbability finalState.jarA * redProbability finalState.jarB = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_red_probability_both_jars_l2852_285206


namespace NUMINAMATH_CALUDE_geometry_propositions_l2852_285212

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) : 
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) := by
  sorry

end NUMINAMATH_CALUDE_geometry_propositions_l2852_285212


namespace NUMINAMATH_CALUDE_arithmetic_mean_ge_geometric_mean_l2852_285256

theorem arithmetic_mean_ge_geometric_mean (a b : ℝ) : (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_ge_geometric_mean_l2852_285256


namespace NUMINAMATH_CALUDE_surface_area_specific_cube_l2852_285290

/-- Calculates the surface area of a cube with holes -/
def surface_area_cube_with_holes (cube_edge_length : ℝ) (hole_side_length : ℝ) (num_holes_per_face : ℕ) : ℝ :=
  let original_surface_area := 6 * cube_edge_length^2
  let area_removed_by_holes := 6 * num_holes_per_face * hole_side_length^2
  let area_exposed_by_holes := 6 * num_holes_per_face * 4 * hole_side_length^2
  original_surface_area - area_removed_by_holes + area_exposed_by_holes

/-- Theorem stating the surface area of the specific cube with holes -/
theorem surface_area_specific_cube : surface_area_cube_with_holes 4 1 2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_specific_cube_l2852_285290


namespace NUMINAMATH_CALUDE_exp_monotone_in_interval_l2852_285249

theorem exp_monotone_in_interval (a b : ℝ) (h : -1 < a ∧ a < b ∧ b < 1) : Real.exp a < Real.exp b := by
  sorry

end NUMINAMATH_CALUDE_exp_monotone_in_interval_l2852_285249


namespace NUMINAMATH_CALUDE_sugar_spill_ratio_l2852_285227

/-- Proves that the ratio of sugar that fell to the ground to the sugar in the torn bag before it fell is 1:2 -/
theorem sugar_spill_ratio (initial_sugar : ℕ) (num_bags : ℕ) (remaining_sugar : ℕ) : 
  initial_sugar = 24 →
  num_bags = 4 →
  remaining_sugar = 21 →
  (initial_sugar - remaining_sugar) * 2 = initial_sugar / num_bags :=
by
  sorry

#check sugar_spill_ratio

end NUMINAMATH_CALUDE_sugar_spill_ratio_l2852_285227


namespace NUMINAMATH_CALUDE_tom_bought_three_decks_l2852_285299

/-- The number of decks Tom bought -/
def tom_decks : ℕ := 3

/-- The cost of each deck in dollars -/
def deck_cost : ℕ := 8

/-- The number of decks Tom's friend bought -/
def friend_decks : ℕ := 5

/-- The total amount spent in dollars -/
def total_spent : ℕ := 64

/-- Theorem stating that Tom bought 3 decks given the conditions -/
theorem tom_bought_three_decks : 
  deck_cost * (tom_decks + friend_decks) = total_spent := by
  sorry

end NUMINAMATH_CALUDE_tom_bought_three_decks_l2852_285299


namespace NUMINAMATH_CALUDE_problem_solution_l2852_285295

def f (n : ℤ) : ℤ := 3 * n^6 + 26 * n^4 + 33 * n^2 + 1

def valid_k (k : ℕ) : Prop :=
  k ≤ 100 ∧ ∃ n : ℤ, f n % k = 0

def solution_set : Finset ℕ :=
  {9, 21, 27, 39, 49, 57, 63, 81, 87, 91, 93}

theorem problem_solution :
  ∀ k : ℕ, valid_k k ↔ k ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2852_285295


namespace NUMINAMATH_CALUDE_arcsin_of_neg_one_l2852_285278

theorem arcsin_of_neg_one : Real.arcsin (-1) = -π / 2 := by sorry

end NUMINAMATH_CALUDE_arcsin_of_neg_one_l2852_285278


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_l2852_285266

def line1 (t k : ℝ) : ℝ × ℝ × ℝ := (3 + 2*t, 2 + 3*t, 2 - k*t)
def line2 (u k : ℝ) : ℝ × ℝ × ℝ := (1 + k*u, 5 - u, 6 + 2*u)

def are_coplanar (k : ℝ) : Prop :=
  ∃ (a b c d : ℝ), ∀ (t u : ℝ),
    a * (line1 t k).1 + b * (line1 t k).2.1 + c * (line1 t k).2.2 + d = 0 ∧
    a * (line2 u k).1 + b * (line2 u k).2.1 + c * (line2 u k).2.2 + d = 0

theorem lines_coplanar_iff_k (k : ℝ) :
  are_coplanar k ↔ (k = -5 - 3 * Real.sqrt 3 ∨ k = -5 + 3 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_l2852_285266


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l2852_285224

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Define the universal set U
def U (m : ℝ) : Set ℝ := A ∪ B m

-- Theorem for part (1)
theorem part_one (m : ℝ) (h : m = 3) : 
  A ∩ (U m \ B m) = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem for part (2)
theorem part_two : 
  {m : ℝ | A ∩ B m = ∅} = {m : ℝ | m ≤ -2} := by sorry

-- Theorem for part (3)
theorem part_three : 
  {m : ℝ | A ∩ B m = A} = {m : ℝ | m ≥ 4} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l2852_285224


namespace NUMINAMATH_CALUDE_second_sum_is_1720_l2852_285279

/-- Given a total sum of 2795 rupees divided into two parts, where the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    prove that the second part is equal to 1720 rupees. -/
theorem second_sum_is_1720 (total : ℝ) (first_part second_part : ℝ) : 
  total = 2795 →
  first_part + second_part = total →
  (first_part * 3 * 8) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1720 := by
  sorry

end NUMINAMATH_CALUDE_second_sum_is_1720_l2852_285279


namespace NUMINAMATH_CALUDE_equal_distribution_of_chicken_wings_l2852_285207

def chicken_wings_per_person (num_friends : ℕ) (initial_wings : ℕ) (additional_wings : ℕ) : ℕ :=
  (initial_wings + additional_wings) / num_friends

theorem equal_distribution_of_chicken_wings :
  chicken_wings_per_person 5 20 25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_chicken_wings_l2852_285207
