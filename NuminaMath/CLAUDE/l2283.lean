import Mathlib

namespace NUMINAMATH_CALUDE_valid_routes_l2283_228316

/-- Represents the lengths of route segments between consecutive cities --/
structure RouteLengths where
  ab : ℕ
  bc : ℕ
  cd : ℕ
  de : ℕ
  ef : ℕ

/-- Checks if the given route lengths satisfy all conditions --/
def isValidRoute (r : RouteLengths) : Prop :=
  r.ab > r.bc ∧ r.bc > r.cd ∧ r.cd > r.de ∧ r.de > r.ef ∧
  r.ab = 2 * r.ef ∧
  r.ab + r.bc + r.cd + r.de + r.ef = 53

/-- The theorem stating that only three specific combinations of route lengths are valid --/
theorem valid_routes :
  ∀ r : RouteLengths, isValidRoute r →
    (r = ⟨14, 12, 11, 9, 7⟩ ∨ r = ⟨14, 13, 11, 8, 7⟩ ∨ r = ⟨14, 13, 10, 9, 7⟩) :=
by sorry

end NUMINAMATH_CALUDE_valid_routes_l2283_228316


namespace NUMINAMATH_CALUDE_taylor_painting_time_l2283_228318

/-- The time it takes for Taylor to paint the room alone -/
def taylor_time : ℝ := 12

/-- The time it takes for Jennifer to paint the room alone -/
def jennifer_time : ℝ := 10

/-- The time it takes for Taylor and Jennifer to paint the room together -/
def combined_time : ℝ := 5.45454545455

theorem taylor_painting_time : 
  (1 / taylor_time + 1 / jennifer_time = 1 / combined_time) → taylor_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_taylor_painting_time_l2283_228318


namespace NUMINAMATH_CALUDE_club_officer_selection_l2283_228394

/-- The number of ways to select three distinct positions from a group of n people --/
def selectThreePositions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The number of club members --/
def clubMembers : ℕ := 12

theorem club_officer_selection :
  selectThreePositions clubMembers = 1320 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l2283_228394


namespace NUMINAMATH_CALUDE_students_disliking_both_l2283_228398

theorem students_disliking_both (total : ℕ) (fries : ℕ) (burgers : ℕ) (both : ℕ) :
  total = 25 →
  fries = 15 →
  burgers = 10 →
  both = 6 →
  total - (fries + burgers - both) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_students_disliking_both_l2283_228398


namespace NUMINAMATH_CALUDE_three_planes_division_l2283_228345

/-- A type representing the possible configurations of three non-coincident planes in space -/
inductive PlaneConfiguration
  | AllParallel
  | TwoParallelOneIntersecting
  | IntersectAlongLine
  | IntersectPairwiseParallelLines
  | IntersectAtPoint

/-- The number of parts that space is divided into by three non-coincident planes -/
def numParts (config : PlaneConfiguration) : ℕ :=
  match config with
  | .AllParallel => 4
  | .TwoParallelOneIntersecting => 6
  | .IntersectAlongLine => 6
  | .IntersectPairwiseParallelLines => 7
  | .IntersectAtPoint => 8

/-- Theorem stating that the number of parts is always 4, 6, 7, or 8 -/
theorem three_planes_division (config : PlaneConfiguration) :
  ∃ n : ℕ, (n = 4 ∨ n = 6 ∨ n = 7 ∨ n = 8) ∧ numParts config = n :=
sorry

end NUMINAMATH_CALUDE_three_planes_division_l2283_228345


namespace NUMINAMATH_CALUDE_fraction_is_positive_integer_l2283_228348

theorem fraction_is_positive_integer (q : ℕ+) :
  (∃ k : ℕ+, (5 * q + 40 : ℚ) / (3 * q - 8 : ℚ) = k) ↔ 3 ≤ q ∧ q ≤ 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_positive_integer_l2283_228348


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l2283_228314

theorem quadratic_solution_property : 
  ∀ p q : ℝ, 
  (2 * p^2 + 8 * p - 42 = 0) → 
  (2 * q^2 + 8 * q - 42 = 0) → 
  p ≠ q → 
  (p - q + 2)^2 = 144 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l2283_228314


namespace NUMINAMATH_CALUDE_function_satisfying_lcm_gcd_condition_l2283_228356

theorem function_satisfying_lcm_gcd_condition :
  ∀ (f : ℕ → ℕ),
    (∀ (m n : ℕ), m > 0 ∧ n > 0 → f (m * n) = Nat.lcm m n * Nat.gcd (f m) (f n)) →
    ∃ (k : ℕ), k > 0 ∧ ∀ (x : ℕ), f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_lcm_gcd_condition_l2283_228356


namespace NUMINAMATH_CALUDE_adult_ticket_price_l2283_228315

-- Define the variables and constants
def adult_price : ℝ := sorry
def child_price : ℝ := 3.50
def total_tickets : ℕ := 21
def total_revenue : ℝ := 83.50
def adult_tickets : ℕ := 5

-- Theorem statement
theorem adult_ticket_price :
  adult_price = 5.50 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l2283_228315


namespace NUMINAMATH_CALUDE_total_garden_area_l2283_228371

-- Define the garden dimensions and counts for each person
def mancino_gardens : ℕ := 4
def mancino_length : ℕ := 16
def mancino_width : ℕ := 5

def marquita_gardens : ℕ := 3
def marquita_length : ℕ := 8
def marquita_width : ℕ := 4

def matteo_gardens : ℕ := 2
def matteo_length : ℕ := 12
def matteo_width : ℕ := 6

def martina_gardens : ℕ := 5
def martina_length : ℕ := 10
def martina_width : ℕ := 3

-- Theorem stating the total square footage of all gardens
theorem total_garden_area :
  mancino_gardens * mancino_length * mancino_width +
  marquita_gardens * marquita_length * marquita_width +
  matteo_gardens * matteo_length * matteo_width +
  martina_gardens * martina_length * martina_width = 710 := by
  sorry

end NUMINAMATH_CALUDE_total_garden_area_l2283_228371


namespace NUMINAMATH_CALUDE_base_8_to_10_conversion_l2283_228382

-- Define the base 8 number as a list of digits
def base_8_number : List Nat := [2, 4, 6]

-- Define the conversion function from base 8 to base 10
def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem base_8_to_10_conversion :
  base_8_to_10 base_8_number = 166 := by sorry

end NUMINAMATH_CALUDE_base_8_to_10_conversion_l2283_228382


namespace NUMINAMATH_CALUDE_brand_y_pen_price_l2283_228373

theorem brand_y_pen_price
  (price_x : ℝ)
  (total_pens : ℕ)
  (total_cost : ℝ)
  (num_x_pens : ℕ)
  (h1 : price_x = 4)
  (h2 : total_pens = 12)
  (h3 : total_cost = 40)
  (h4 : num_x_pens = 8) :
  (total_cost - price_x * num_x_pens) / (total_pens - num_x_pens) = 2 := by
  sorry

end NUMINAMATH_CALUDE_brand_y_pen_price_l2283_228373


namespace NUMINAMATH_CALUDE_equality_condition_l2283_228353

theorem equality_condition (p q r : ℝ) : p + q * r = (p + q) * (p + r) ↔ p + q + r = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l2283_228353


namespace NUMINAMATH_CALUDE_problem_statement_l2283_228397

theorem problem_statement (p : ℝ) (h : 126 * 3^8 = p) : 126 * 3^6 = (1/9) * p := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2283_228397


namespace NUMINAMATH_CALUDE_profit_per_meter_l2283_228313

/-- The profit per meter of cloth given the selling price, quantity sold, and cost price per meter -/
theorem profit_per_meter
  (selling_price : ℕ)
  (quantity : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : selling_price = 4950)
  (h2 : quantity = 75)
  (h3 : cost_price_per_meter = 51) :
  (selling_price - quantity * cost_price_per_meter) / quantity = 15 :=
by sorry

end NUMINAMATH_CALUDE_profit_per_meter_l2283_228313


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_negative_300_degrees_to_radians_l2283_228338

theorem degree_to_radian_conversion (angle_in_degrees : ℝ) : 
  angle_in_degrees * (π / 180) = angle_in_degrees * π / 180 := by sorry

theorem negative_300_degrees_to_radians : 
  -300 * (π / 180) = -5 * π / 3 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_negative_300_degrees_to_radians_l2283_228338


namespace NUMINAMATH_CALUDE_tangency_quadrilateral_area_is_1_6_l2283_228335

/-- An isosceles trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- Area of the trapezoid -/
  trapezoidArea : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : Bool
  /-- The circle is inscribed in the trapezoid -/
  isInscribed : Bool

/-- The area of the quadrilateral formed by the points of tangency -/
def tangencyQuadrilateralArea (t : InscribedCircleTrapezoid) : ℝ := sorry

/-- Theorem: The area of the tangency quadrilateral is 1.6 -/
theorem tangency_quadrilateral_area_is_1_6 (t : InscribedCircleTrapezoid) 
  (h1 : t.radius = 1) 
  (h2 : t.trapezoidArea = 5) 
  (h3 : t.isIsosceles = true) 
  (h4 : t.isInscribed = true) : 
  tangencyQuadrilateralArea t = 1.6 := by sorry

end NUMINAMATH_CALUDE_tangency_quadrilateral_area_is_1_6_l2283_228335


namespace NUMINAMATH_CALUDE_range_of_a_l2283_228305

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}

def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}

theorem range_of_a (a : ℝ) : (A ∪ B a = A) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2283_228305


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l2283_228362

theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x + 8) ↔ 
  m ≤ -Real.sqrt 2.4 ∨ m ≥ Real.sqrt 2.4 := by
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l2283_228362


namespace NUMINAMATH_CALUDE_constant_function_if_arithmetic_mean_l2283_228301

def IsArithmeticMean (f : ℤ × ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4

theorem constant_function_if_arithmetic_mean (f : ℤ × ℤ → ℤ) 
  (h1 : ∀ x y : ℤ, f (x, y) > 0)
  (h2 : IsArithmeticMean f) :
  ∃ c : ℤ, ∀ x y : ℤ, f (x, y) = c := by
  sorry

end NUMINAMATH_CALUDE_constant_function_if_arithmetic_mean_l2283_228301


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l2283_228351

theorem max_value_trig_expression (a b : ℝ) :
  (∀ θ : ℝ, a * Real.cos (2 * θ) + b * Real.sin (2 * θ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (2 * θ) + b * Real.sin (2 * θ) = Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l2283_228351


namespace NUMINAMATH_CALUDE_waiter_customers_theorem_l2283_228375

/-- Calculates the final number of customers for a waiter --/
def final_customers (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Proves that the final number of customers is correct --/
theorem waiter_customers_theorem (initial left new : ℕ) 
  (h1 : initial ≥ left) : 
  final_customers initial left new = initial - left + new :=
by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_theorem_l2283_228375


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2283_228310

theorem triangle_abc_properties (A B C : Real) (h : Real) :
  A + B + C = Real.pi →
  A + B = 3 * C →
  2 * Real.sin (A - C) = Real.sin B →
  h * 5 / 2 = Real.sin C * Real.sin A * Real.sin B * 25 →
  Real.sin A = 3 * Real.sqrt 10 / 10 ∧ h = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2283_228310


namespace NUMINAMATH_CALUDE_round_0_0984_to_two_sig_figs_l2283_228370

/-- Rounds a number to a specified number of significant figures -/
def roundToSignificantFigures (x : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- Theorem: Rounding 0.0984 to two significant figures results in 0.098 -/
theorem round_0_0984_to_two_sig_figs :
  roundToSignificantFigures 0.0984 2 = 0.098 := by
  sorry

end NUMINAMATH_CALUDE_round_0_0984_to_two_sig_figs_l2283_228370


namespace NUMINAMATH_CALUDE_hydroton_rainfall_l2283_228326

/-- The total rainfall in Hydroton from 2019 to 2021 -/
def total_rainfall (r2019 r2020 r2021 : ℝ) : ℝ :=
  12 * (r2019 + r2020 + r2021)

/-- Theorem: The total rainfall in Hydroton from 2019 to 2021 is 1884 mm -/
theorem hydroton_rainfall : 
  let r2019 : ℝ := 50
  let r2020 : ℝ := r2019 + 5
  let r2021 : ℝ := r2020 - 3
  total_rainfall r2019 r2020 r2021 = 1884 :=
by
  sorry


end NUMINAMATH_CALUDE_hydroton_rainfall_l2283_228326


namespace NUMINAMATH_CALUDE_no_fraction_satisfies_condition_l2283_228363

theorem no_fraction_satisfies_condition : ¬∃ (x y : ℕ+), 
  (Nat.gcd x.val y.val = 1) ∧ 
  ((x + 2 : ℚ) / (y + 2) = 1.2 * (x : ℚ) / y) := by
  sorry

end NUMINAMATH_CALUDE_no_fraction_satisfies_condition_l2283_228363


namespace NUMINAMATH_CALUDE_jason_toy_count_l2283_228317

/-- The number of toys each person has -/
structure ToyCount where
  rachel : ℝ
  john : ℝ
  jason : ℝ

/-- The conditions of the problem -/
def toy_problem (t : ToyCount) : Prop :=
  t.rachel = 1 ∧
  t.john = t.rachel + 6.5 ∧
  t.jason = 3 * t.john

/-- Theorem stating that Jason has 22.5 toys -/
theorem jason_toy_count (t : ToyCount) (h : toy_problem t) : t.jason = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_jason_toy_count_l2283_228317


namespace NUMINAMATH_CALUDE_function_divides_property_l2283_228306

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem function_divides_property 
  (f : ℤ → ℕ+) 
  (h : ∀ m n : ℤ, divides (f (m - n)) (f m - f n)) :
  ∀ n m : ℤ, f n ≤ f m → divides (f n) (f m) := by
  sorry

end NUMINAMATH_CALUDE_function_divides_property_l2283_228306


namespace NUMINAMATH_CALUDE_no_adjacent_knights_probability_l2283_228329

/-- The number of knights seated in a circle -/
def total_knights : ℕ := 20

/-- The number of knights selected for the quest -/
def selected_knights : ℕ := 4

/-- The probability that no two of the selected knights are sitting next to each other -/
def probability : ℚ := 60 / 7

/-- Theorem stating that the probability of no two selected knights sitting next to each other is 60/7 -/
theorem no_adjacent_knights_probability :
  probability = 60 / 7 := by sorry

end NUMINAMATH_CALUDE_no_adjacent_knights_probability_l2283_228329


namespace NUMINAMATH_CALUDE_g_sum_zero_l2283_228319

def g (x : ℝ) : ℝ := x^2 - 2013*x

theorem g_sum_zero (a b : ℝ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_zero_l2283_228319


namespace NUMINAMATH_CALUDE_delta_zero_implies_c_sqrt_30_l2283_228376

def Δ (a b c : ℝ) : ℝ := c^2 - 3*a*b

theorem delta_zero_implies_c_sqrt_30 (a b c : ℝ) (h1 : Δ a b c = 0) (h2 : a = 2) (h3 : b = 5) :
  c = Real.sqrt 30 ∨ c = -Real.sqrt 30 := by sorry

end NUMINAMATH_CALUDE_delta_zero_implies_c_sqrt_30_l2283_228376


namespace NUMINAMATH_CALUDE_quadratic_root_form_l2283_228341

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 + 6*x + d = 0 ↔ x = (-6 + Real.sqrt d) / 2 ∨ x = (-6 - Real.sqrt d) / 2) →
  d = 36 / 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_form_l2283_228341


namespace NUMINAMATH_CALUDE_not_p_or_q_l2283_228346

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.sin x > 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp (-x) < 0

-- Theorem to prove
theorem not_p_or_q : ¬(p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_q_l2283_228346


namespace NUMINAMATH_CALUDE_cubic_range_l2283_228381

theorem cubic_range (x : ℝ) (h : x^2 - 5*x + 6 < 0) :
  41 < x^3 + 5*x^2 + 6*x + 1 ∧ x^3 + 5*x^2 + 6*x + 1 < 91 := by
  sorry

end NUMINAMATH_CALUDE_cubic_range_l2283_228381


namespace NUMINAMATH_CALUDE_no_three_consecutive_digit_sum_squares_l2283_228378

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Theorem: There do not exist three consecutive integers such that 
    the sum of digits of each is a perfect square -/
theorem no_three_consecutive_digit_sum_squares :
  ¬ ∃ n : ℕ, (is_perfect_square (sum_of_digits n)) ∧ 
             (is_perfect_square (sum_of_digits (n + 1))) ∧ 
             (is_perfect_square (sum_of_digits (n + 2))) :=
sorry

end NUMINAMATH_CALUDE_no_three_consecutive_digit_sum_squares_l2283_228378


namespace NUMINAMATH_CALUDE_square_circumference_l2283_228322

/-- Given a square with an area of 324 square meters, its circumference is 72 meters. -/
theorem square_circumference (s : Real) (area : Real) (h1 : area = 324) (h2 : s^2 = area) :
  4 * s = 72 := by
  sorry

end NUMINAMATH_CALUDE_square_circumference_l2283_228322


namespace NUMINAMATH_CALUDE_freshWaterCostForFamily_l2283_228366

/-- The cost of fresh water for a day for a family, given the cost per gallon, 
    daily water need per person, and number of family members. -/
def freshWaterCost (costPerGallon : ℚ) (dailyNeedPerPerson : ℚ) (familySize : ℕ) : ℚ :=
  costPerGallon * dailyNeedPerPerson * familySize

/-- Theorem stating that the cost of fresh water for a day for a family of 6 is $3, 
    given the specified conditions. -/
theorem freshWaterCostForFamily : 
  freshWaterCost 1 (1/2) 6 = 3 := by
  sorry


end NUMINAMATH_CALUDE_freshWaterCostForFamily_l2283_228366


namespace NUMINAMATH_CALUDE_speed_ratio_is_seven_to_eight_l2283_228323

-- Define the speeds of A and B
def v_A : ℝ := sorry
def v_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := 400

-- Define the time intervals
def time1 : ℝ := 3
def time2 : ℝ := 12

-- Theorem statement
theorem speed_ratio_is_seven_to_eight :
  -- Condition 1: After 3 minutes, A and B are equidistant from O
  (v_A * time1 = |initial_B_position - v_B * time1|) →
  -- Condition 2: After 12 minutes, A and B are again equidistant from O
  (v_A * time2 = |initial_B_position - v_B * time2|) →
  -- Conclusion: The ratio of A's speed to B's speed is 7:8
  (v_A / v_B = 7 / 8) := by
sorry

end NUMINAMATH_CALUDE_speed_ratio_is_seven_to_eight_l2283_228323


namespace NUMINAMATH_CALUDE_percent_fifteen_percent_l2283_228328

-- Define the operations
def percent (y : Int) : Int := 8 - y
def prepercent (y : Int) : Int := y - 8

-- Theorem statement
theorem percent_fifteen_percent : prepercent (percent 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_percent_fifteen_percent_l2283_228328


namespace NUMINAMATH_CALUDE_expression_value_l2283_228324

theorem expression_value : (3^2 - 2 * 3) - (5^2 - 2 * 5) + (7^2 - 2 * 7) = 23 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2283_228324


namespace NUMINAMATH_CALUDE_average_age_proof_l2283_228389

/-- Given the ages of John, Mary, and Tonya with specific relationships, prove their average age --/
theorem average_age_proof (tonya mary john : ℕ) 
  (h1 : john = 2 * mary)
  (h2 : john * 2 = tonya)
  (h3 : tonya = 60) : 
  (tonya + john + mary) / 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_age_proof_l2283_228389


namespace NUMINAMATH_CALUDE_distance_representation_l2283_228383

theorem distance_representation (a : ℝ) : 
  |a + 1| = |a - (-1)| := by sorry

-- The statement proves that |a + 1| is equal to the distance between a and -1,
-- which represents the distance between points A and C on the number line.

end NUMINAMATH_CALUDE_distance_representation_l2283_228383


namespace NUMINAMATH_CALUDE_inequality_proof_l2283_228384

theorem inequality_proof (a b c x y z k : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : a + x = k ∧ b + y = k ∧ c + z = k) : 
  a * x + b * y + c * z < k^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2283_228384


namespace NUMINAMATH_CALUDE_problem_solution_l2283_228352

theorem problem_solution (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x*y + x + y = 5) : 
  x^2*y + x*y^2 + x^2 + y^2 = 18 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2283_228352


namespace NUMINAMATH_CALUDE_target_walmart_tool_difference_l2283_228380

/-- Represents a multitool with its components -/
structure Multitool where
  screwdrivers : Nat
  knives : Nat
  files : Nat
  scissors : Nat
  other_tools : Nat

/-- The Walmart multitool -/
def walmart_multitool : Multitool :=
  { screwdrivers := 1
    knives := 3
    files := 0
    scissors := 0
    other_tools := 2 }

/-- The Target multitool -/
def target_multitool : Multitool :=
  { screwdrivers := 1
    knives := 2 * walmart_multitool.knives
    files := 3
    scissors := 1
    other_tools := 0 }

/-- Total number of tools in a multitool -/
def total_tools (m : Multitool) : Nat :=
  m.screwdrivers + m.knives + m.files + m.scissors + m.other_tools

/-- Theorem stating the difference in the number of tools between Target and Walmart multitools -/
theorem target_walmart_tool_difference :
  total_tools target_multitool - total_tools walmart_multitool = 5 := by
  sorry


end NUMINAMATH_CALUDE_target_walmart_tool_difference_l2283_228380


namespace NUMINAMATH_CALUDE_composition_ratio_l2283_228334

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 1
def g (x : ℝ) : ℝ := 2 * x + 5

-- State the theorem
theorem composition_ratio :
  (g (f (g 3))) / (f (g (f 3))) = 69 / 206 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l2283_228334


namespace NUMINAMATH_CALUDE_divisible_by_three_l2283_228355

theorem divisible_by_three (n : ℕ) : ∃ k : ℤ, 2 * 7^n + 1 = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l2283_228355


namespace NUMINAMATH_CALUDE_instantaneous_rate_of_change_at_zero_l2283_228347

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp (Real.sin x)

theorem instantaneous_rate_of_change_at_zero :
  deriv f 0 = 2 * Real.exp 0 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_rate_of_change_at_zero_l2283_228347


namespace NUMINAMATH_CALUDE_dice_remainder_prob_l2283_228332

/-- The probability of getting a specific remainder when the sum of two dice is divided by 4 -/
def remainder_probability (r : Fin 4) : ℚ := sorry

/-- The sum of all probabilities should be 1 -/
axiom prob_sum_one : remainder_probability 0 + remainder_probability 1 + remainder_probability 2 + remainder_probability 3 = 1

/-- The probabilities are non-negative -/
axiom prob_non_negative (r : Fin 4) : remainder_probability r ≥ 0

theorem dice_remainder_prob :
  2 * remainder_probability 3 - 3 * remainder_probability 2 + remainder_probability 1 - remainder_probability 0 = -2/9 := by
  sorry

end NUMINAMATH_CALUDE_dice_remainder_prob_l2283_228332


namespace NUMINAMATH_CALUDE_range_of_a_l2283_228385

-- Define propositions p and q
def p (x : ℝ) : Prop := (x - 2)^2 ≤ 1

def q (x a : ℝ) : Prop := x^2 + (2*a + 1)*x + a*(a + 1) ≥ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ¬(∀ x, q x a → p x)

-- Theorem statement
theorem range_of_a :
  {a : ℝ | sufficient_not_necessary a} = {a | a ≤ -4 ∨ a ≥ -1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2283_228385


namespace NUMINAMATH_CALUDE_problem_statement_l2283_228312

theorem problem_statement (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : (x + y + z) * (1/x + 1/y + 1/z) = 91/10) :
  ⌊(x^3 + y^3 + z^3) * (1/x^3 + 1/y^3 + 1/z^3)⌋ = 9 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2283_228312


namespace NUMINAMATH_CALUDE_probability_of_specific_sequence_l2283_228336

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of Jacks in a standard deck -/
def NumJacks : ℕ := 4

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Calculates the probability of drawing the specified sequence of cards -/
def probability_of_sequence : ℚ :=
  (NumKings : ℚ) / StandardDeck *
  (NumHearts - 1) / (StandardDeck - 1) *
  NumJacks / (StandardDeck - 2) *
  (NumSpades - 1) / (StandardDeck - 3) *
  NumQueens / (StandardDeck - 4)

theorem probability_of_specific_sequence :
  probability_of_sequence = 3 / 10125 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_sequence_l2283_228336


namespace NUMINAMATH_CALUDE_four_mat_weaves_four_days_l2283_228331

-- Define the rate of weaving (mats per mat-weave per day)
def weaving_rate (mats : ℕ) (mat_weaves : ℕ) (days : ℕ) : ℚ :=
  (mats : ℚ) / ((mat_weaves : ℚ) * (days : ℚ))

theorem four_mat_weaves_four_days (mats : ℕ) :
  -- Condition: 8 mat-weaves weave 16 mats in 8 days
  weaving_rate 16 8 8 = weaving_rate mats 4 4 →
  -- Conclusion: 4 mat-weaves weave 4 mats in 4 days
  mats = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_mat_weaves_four_days_l2283_228331


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l2283_228304

theorem det_2x2_matrix : 
  Matrix.det !![4, 3; 2, 1] = -2 := by
  sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l2283_228304


namespace NUMINAMATH_CALUDE_technician_count_l2283_228302

/-- Proves the number of technicians in a workshop given specific salary and worker information --/
theorem technician_count (total_workers : ℕ) (avg_salary_all : ℚ) (avg_salary_tech : ℚ) (avg_salary_non_tech : ℚ) 
  (h1 : total_workers = 12)
  (h2 : avg_salary_all = 9500)
  (h3 : avg_salary_tech = 12000)
  (h4 : avg_salary_non_tech = 6000) :
  ∃ (tech_count : ℕ), tech_count = 7 ∧ tech_count ≤ total_workers :=
by sorry

end NUMINAMATH_CALUDE_technician_count_l2283_228302


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l2283_228358

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the line l
def line_l (x y : ℝ) : Prop := x = 0 ∨ y = -3/4 * x

-- Define the point A
def point_A : ℝ × ℝ := (2, -1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y = 1

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := y = -2 * x

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

theorem circle_and_line_theorem :
  -- Circle C passes through point A
  circle_C point_A.1 point_A.2 →
  -- Circle C is tangent to the line x+y=1
  ∃ (x y : ℝ), circle_C x y ∧ tangent_line x y →
  -- The center of the circle lies on the line y=-2x
  ∃ (x y : ℝ), circle_C x y ∧ center_line x y →
  -- Line l passes through the origin
  ∃ (x y : ℝ), line_l x y ∧ (x, y) = origin →
  -- The chord intercepted by circle C on line l has a length of 2
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 →
  -- Conclusion: The equations of circle C and line l are correct
  (∀ (x y : ℝ), circle_C x y ↔ (x - 1)^2 + (y + 2)^2 = 2) ∧
  (∀ (x y : ℝ), line_l x y ↔ (x = 0 ∨ y = -3/4 * x)) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_and_line_theorem_l2283_228358


namespace NUMINAMATH_CALUDE_new_students_average_age_l2283_228392

/-- Proves that the average age of new students is 32 years given the problem conditions --/
theorem new_students_average_age
  (original_average : ℝ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℝ)
  (h1 : original_average = 40)
  (h2 : original_strength = 12)
  (h3 : new_students = 12)
  (h4 : average_decrease = 4) :
  let new_average := original_average - average_decrease
  let total_original := original_average * original_strength
  let total_new := new_average * (original_strength + new_students) - total_original
  total_new / new_students = 32 := by
  sorry

#check new_students_average_age

end NUMINAMATH_CALUDE_new_students_average_age_l2283_228392


namespace NUMINAMATH_CALUDE_inequality_and_factorial_l2283_228303

theorem inequality_and_factorial (n : ℕ) : 2 ≤ (1 + 1 / n : ℝ) ^ n ∧ (1 + 1 / n : ℝ) ^ n < 3 ∧ (n / 3 : ℝ) ^ n < n! := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_factorial_l2283_228303


namespace NUMINAMATH_CALUDE_trees_not_replanted_l2283_228393

/-- 
Given a track with trees planted every 4 meters along its 48-meter length,
prove that when replanting trees every 6 meters, 5 trees do not need to be replanted.
-/
theorem trees_not_replanted (track_length : ℕ) (initial_spacing : ℕ) (new_spacing : ℕ) : 
  track_length = 48 → initial_spacing = 4 → new_spacing = 6 → 
  (track_length / Nat.lcm initial_spacing new_spacing) + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_trees_not_replanted_l2283_228393


namespace NUMINAMATH_CALUDE_f_of_4_equals_23_l2283_228343

-- Define the function f
def f : ℝ → ℝ := fun x => 2 * (2 * x + 2) + 3

-- State the theorem
theorem f_of_4_equals_23 : f 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_f_of_4_equals_23_l2283_228343


namespace NUMINAMATH_CALUDE_problem_solution_l2283_228300

theorem problem_solution (A B : ℝ) 
  (h1 : 100 * A = 35^2 - 15^2) 
  (h2 : (A - 1)^6 = 27^B) : 
  A = 10 ∧ B = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2283_228300


namespace NUMINAMATH_CALUDE_money_left_over_l2283_228320

/-- Calculates the money left over after buying a bike given work parameters and bike cost -/
theorem money_left_over 
  (hourly_rate : ℝ) 
  (weekly_hours : ℝ) 
  (weeks_worked : ℝ) 
  (bike_cost : ℝ) 
  (h1 : hourly_rate = 8)
  (h2 : weekly_hours = 35)
  (h3 : weeks_worked = 4)
  (h4 : bike_cost = 400) :
  hourly_rate * weekly_hours * weeks_worked - bike_cost = 720 := by
  sorry


end NUMINAMATH_CALUDE_money_left_over_l2283_228320


namespace NUMINAMATH_CALUDE_battery_difference_proof_l2283_228365

/-- The number of batteries Tom used in flashlights -/
def flashlight_batteries : ℕ := 2

/-- The number of batteries Tom used in toys -/
def toy_batteries : ℕ := 15

/-- The difference between the number of batteries in toys and flashlights -/
def battery_difference : ℕ := toy_batteries - flashlight_batteries

theorem battery_difference_proof : battery_difference = 13 := by
  sorry

end NUMINAMATH_CALUDE_battery_difference_proof_l2283_228365


namespace NUMINAMATH_CALUDE_origami_paper_distribution_l2283_228364

theorem origami_paper_distribution (total_papers : ℝ) (num_cousins : ℝ) 
  (h1 : total_papers = 48.0)
  (h2 : num_cousins = 6.0)
  (h3 : num_cousins ≠ 0) :
  total_papers / num_cousins = 8.0 := by
sorry

end NUMINAMATH_CALUDE_origami_paper_distribution_l2283_228364


namespace NUMINAMATH_CALUDE_speed_train_B_is_25_l2283_228342

/-- Represents the distance between two stations in kilometers -/
def distance_between_stations : ℝ := 155

/-- Represents the speed of the train from station A in km/h -/
def speed_train_A : ℝ := 20

/-- Represents the time difference between the starts of the two trains in hours -/
def time_difference : ℝ := 1

/-- Represents the total time until the trains meet in hours -/
def total_time : ℝ := 4

/-- Represents the time the train from B travels in hours -/
def time_train_B : ℝ := 3

/-- Theorem stating that the speed of the train from station B is 25 km/h -/
theorem speed_train_B_is_25 : 
  ∃ (speed_B : ℝ), 
    speed_B * time_train_B = distance_between_stations - speed_train_A * total_time ∧ 
    speed_B = 25 := by
  sorry

end NUMINAMATH_CALUDE_speed_train_B_is_25_l2283_228342


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2283_228359

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 3| + |x - 1| < a^2 - 3*a) ↔ (a < -1 ∨ a > 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2283_228359


namespace NUMINAMATH_CALUDE_molecular_weight_3_moles_N2O3_l2283_228360

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in a molecule of Dinitrogen trioxide -/
def N_count : ℕ := 2

/-- The number of Oxygen atoms in a molecule of Dinitrogen trioxide -/
def O_count : ℕ := 3

/-- The number of moles of Dinitrogen trioxide -/
def mole_count : ℝ := 3

/-- The molecular weight of Dinitrogen trioxide in g/mol -/
def molecular_weight_N2O3 : ℝ := N_count * atomic_weight_N + O_count * atomic_weight_O

/-- Theorem: The molecular weight of 3 moles of Dinitrogen trioxide is 228.06 grams -/
theorem molecular_weight_3_moles_N2O3 : 
  mole_count * molecular_weight_N2O3 = 228.06 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_3_moles_N2O3_l2283_228360


namespace NUMINAMATH_CALUDE_cube_division_l2283_228340

theorem cube_division (original_size : ℝ) (num_divisions : ℕ) (num_painted : ℕ) :
  original_size = 3 →
  num_divisions ^ 3 = 27 →
  num_painted = 26 →
  ∃ (smaller_size : ℝ),
    smaller_size = 1 ∧
    num_divisions * smaller_size = original_size :=
by sorry

end NUMINAMATH_CALUDE_cube_division_l2283_228340


namespace NUMINAMATH_CALUDE_count_three_digit_numbers_l2283_228367

def digit := Fin 4

def valid_first_digit (d : digit) : Prop := d.val ≠ 0

def three_digit_number := { n : ℕ | 100 ≤ n ∧ n < 1000 }

def count_valid_numbers : ℕ := sorry

theorem count_three_digit_numbers : count_valid_numbers = 48 := by sorry

end NUMINAMATH_CALUDE_count_three_digit_numbers_l2283_228367


namespace NUMINAMATH_CALUDE_commercial_break_duration_l2283_228374

theorem commercial_break_duration :
  let five_minute_commercials : ℕ := 3
  let two_minute_commercials : ℕ := 11
  let five_minute_duration : ℕ := 5
  let two_minute_duration : ℕ := 2
  (five_minute_commercials * five_minute_duration + two_minute_commercials * two_minute_duration) = 37 := by
  sorry

end NUMINAMATH_CALUDE_commercial_break_duration_l2283_228374


namespace NUMINAMATH_CALUDE_coin_not_touching_lines_l2283_228361

/-- The probability that a randomly tossed coin doesn't touch parallel lines -/
theorem coin_not_touching_lines (a r : ℝ) (h : r < a) :
  let p := (a - r) / a
  0 ≤ p ∧ p ≤ 1 ∧ p = (a - r) / a :=
by sorry

end NUMINAMATH_CALUDE_coin_not_touching_lines_l2283_228361


namespace NUMINAMATH_CALUDE_multiplication_preserves_odd_positives_l2283_228395

def P : Set ℕ := {n : ℕ | n % 2 = 1 ∧ n > 0}

def M : Set ℕ := {x : ℕ | ∃ (a b : ℕ), a ∈ P ∧ b ∈ P ∧ x = a * b}

theorem multiplication_preserves_odd_positives (h : M ⊆ P) :
  ∀ (a b : ℕ), a ∈ P → b ∈ P → (a * b) ∈ P := by
  sorry

end NUMINAMATH_CALUDE_multiplication_preserves_odd_positives_l2283_228395


namespace NUMINAMATH_CALUDE_base_conversion_sum_l2283_228390

/-- Converts a number from base 11 to base 10 -/
def base11ToBase10 (n : Nat) : Nat :=
  (n / 100) * 121 + ((n / 10) % 10) * 11 + (n % 10)

/-- Converts a number from base 12 to base 10 -/
def base12ToBase10 (n : Nat) (A B : Nat) : Nat :=
  (n / 100) * 144 + ((n / 10) % 10) * 12 + (n % 10)

theorem base_conversion_sum :
  let n1 : Nat := 249
  let n2 : Nat := 3 * 100 + 10 * 10 + 11
  let A : Nat := 10
  let B : Nat := 11
  base11ToBase10 n1 + base12ToBase10 n2 A B = 858 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l2283_228390


namespace NUMINAMATH_CALUDE_last_nonzero_digit_of_b_d_is_five_l2283_228387

/-- Definition of b_n -/
def b (n : ℕ+) : ℕ := 2 * (Nat.factorial (n + 10) / Nat.factorial (n + 2))

/-- The last nonzero digit of a natural number -/
def lastNonzeroDigit (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- The smallest positive integer d such that the last nonzero digit of b(d) is odd -/
def d : ℕ+ := sorry

theorem last_nonzero_digit_of_b_d_is_five :
  lastNonzeroDigit (b d) = 5 := by sorry

end NUMINAMATH_CALUDE_last_nonzero_digit_of_b_d_is_five_l2283_228387


namespace NUMINAMATH_CALUDE_nearest_fraction_sum_l2283_228337

theorem nearest_fraction_sum : ∃ (x : ℕ), 
  (2007 : ℝ) / 2999 + (8001 : ℝ) / x + (2001 : ℝ) / 3999 = 3.0035428163476343 ∧ 
  x = 4362 := by
sorry

end NUMINAMATH_CALUDE_nearest_fraction_sum_l2283_228337


namespace NUMINAMATH_CALUDE_parabola_directrix_l2283_228368

/-- A parabola with equation y² = -8x that opens to the left has a directrix with equation x = 2 -/
theorem parabola_directrix (y x : ℝ) : 
  (y^2 = -8*x) → 
  (∃ p : ℝ, y^2 = -4*p*x ∧ p > 0) → 
  (∃ a : ℝ, a = 2 ∧ ∀ x₀ y₀ : ℝ, y₀^2 = -8*x₀ → |x₀ - a| = |y₀|/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2283_228368


namespace NUMINAMATH_CALUDE_dog_kennel_problem_l2283_228386

theorem dog_kennel_problem (total : ℕ) (long_fur : ℕ) (brown : ℕ) (neither : ℕ) 
  (h1 : total = 45)
  (h2 : long_fur = 36)
  (h3 : brown = 27)
  (h4 : neither = 8)
  : total - neither - (long_fur + brown - (total - neither)) = 26 := by
  sorry

end NUMINAMATH_CALUDE_dog_kennel_problem_l2283_228386


namespace NUMINAMATH_CALUDE_bethany_age_proof_l2283_228321

/-- Bethany's current age -/
def bethanys_current_age : ℕ := 19

/-- Bethany's younger sister's current age -/
def sisters_current_age : ℕ := 11

/-- Bethany's age three years ago -/
def bethanys_age_three_years_ago : ℕ := bethanys_current_age - 3

/-- Bethany's younger sister's age three years ago -/
def sisters_age_three_years_ago : ℕ := sisters_current_age - 3

theorem bethany_age_proof :
  (bethanys_age_three_years_ago = 2 * sisters_age_three_years_ago) ∧
  (sisters_current_age + 5 = 16) →
  bethanys_current_age = 19 := by
  sorry

end NUMINAMATH_CALUDE_bethany_age_proof_l2283_228321


namespace NUMINAMATH_CALUDE_triangle_interior_center_points_l2283_228330

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle ABC in the Cartesian plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Count of interior center points in a triangle -/
def interiorCenterPoints (t : Triangle) : ℕ :=
  sorry

/-- The main theorem -/
theorem triangle_interior_center_points :
  let t : Triangle := {
    A := { x := 0, y := 0 },
    B := { x := 200, y := 100 },
    C := { x := 30, y := 330 }
  }
  interiorCenterPoints t = 31480 := by sorry

end NUMINAMATH_CALUDE_triangle_interior_center_points_l2283_228330


namespace NUMINAMATH_CALUDE_complement_of_union_equals_five_l2283_228309

def U : Finset ℕ := {1, 3, 5, 9}
def A : Finset ℕ := {1, 3, 9}
def B : Finset ℕ := {1, 9}

theorem complement_of_union_equals_five : (U \ (A ∪ B)) = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_five_l2283_228309


namespace NUMINAMATH_CALUDE_inequality_and_equality_proof_l2283_228377

theorem inequality_and_equality_proof :
  (∀ a b : ℝ, (a^2 + 1) * (b^2 + 1) + 50 ≥ 2 * (2*a + 1) * (3*b + 1)) ∧
  (∀ n p : ℕ+, (n^2 + 1) * (p^2 + 1) + 45 = 2 * (2*n + 1) * (3*p + 1) ↔ n = 2 ∧ p = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_proof_l2283_228377


namespace NUMINAMATH_CALUDE_water_barrel_problem_l2283_228399

theorem water_barrel_problem :
  ∀ (bucket_capacity : ℕ),
    bucket_capacity > 0 →
    bucket_capacity / 2 +
    bucket_capacity / 3 +
    bucket_capacity / 4 +
    bucket_capacity / 5 +
    bucket_capacity / 6 = 29 →
    bucket_capacity ≤ 30 →
    29 ≤ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_water_barrel_problem_l2283_228399


namespace NUMINAMATH_CALUDE_pants_price_satisfies_conditions_l2283_228350

/-- The original price of pants that satisfies the given conditions -/
def original_pants_price : ℝ := 110

/-- The number of pairs of pants purchased -/
def num_pants : ℕ := 4

/-- The number of pairs of socks purchased -/
def num_socks : ℕ := 2

/-- The original price of socks -/
def original_socks_price : ℝ := 60

/-- The discount rate applied to all items -/
def discount_rate : ℝ := 0.3

/-- The total cost after discount -/
def total_cost_after_discount : ℝ := 392

/-- Theorem stating that the original pants price satisfies the given conditions -/
theorem pants_price_satisfies_conditions :
  (num_pants : ℝ) * original_pants_price * (1 - discount_rate) +
  (num_socks : ℝ) * original_socks_price * (1 - discount_rate) =
  total_cost_after_discount := by sorry

end NUMINAMATH_CALUDE_pants_price_satisfies_conditions_l2283_228350


namespace NUMINAMATH_CALUDE_softball_team_size_l2283_228372

/-- Proves that a co-ed softball team with given conditions has 20 total players -/
theorem softball_team_size : ∀ (men women : ℕ),
  women = men + 4 →
  (men : ℚ) / (women : ℚ) = 2/3 →
  men + women = 20 := by
sorry

end NUMINAMATH_CALUDE_softball_team_size_l2283_228372


namespace NUMINAMATH_CALUDE_angle_identity_l2283_228307

theorem angle_identity (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 - 2 * Real.cos A * Real.cos B * Real.cos C = 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_identity_l2283_228307


namespace NUMINAMATH_CALUDE_summer_grain_scientific_notation_l2283_228349

def summer_grain_production : ℝ := 11534000000

/-- Converts a number to scientific notation with a specified number of significant figures -/
def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

theorem summer_grain_scientific_notation :
  to_scientific_notation summer_grain_production 4 = (1.153, 8) :=
sorry

end NUMINAMATH_CALUDE_summer_grain_scientific_notation_l2283_228349


namespace NUMINAMATH_CALUDE_rice_purchase_amount_l2283_228308

/-- The price of rice in cents per pound -/
def rice_price : ℚ := 75

/-- The price of beans in cents per pound -/
def bean_price : ℚ := 35

/-- The total weight of rice and beans in pounds -/
def total_weight : ℚ := 30

/-- The total cost in cents -/
def total_cost : ℚ := 1650

/-- The amount of rice purchased in pounds -/
def rice_amount : ℚ := 15

theorem rice_purchase_amount :
  ∃ (bean_amount : ℚ),
    rice_amount + bean_amount = total_weight ∧
    rice_price * rice_amount + bean_price * bean_amount = total_cost :=
sorry

end NUMINAMATH_CALUDE_rice_purchase_amount_l2283_228308


namespace NUMINAMATH_CALUDE_thirtieth_term_value_l2283_228391

def arithmeticGeometricSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  if n ≤ 4 then
    a₁ + (n - 1) * d
  else
    2 * arithmeticGeometricSequence a₁ d (n - 1)

theorem thirtieth_term_value :
  arithmeticGeometricSequence 4 3 30 = 436207104 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_value_l2283_228391


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l2283_228344

theorem quadratic_integer_roots (p : ℕ) (b : ℕ) (hp : Prime p) (hb : b > 0) :
  (∃ x y : ℤ, x^2 - b*x + b*p = 0 ∧ y^2 - b*y + b*p = 0) ↔ b = (p + 1)^2 ∨ b = 4*p :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l2283_228344


namespace NUMINAMATH_CALUDE_terms_before_five_l2283_228369

/-- An arithmetic sequence with first term 95 and common difference -5 -/
def arithmeticSequence (n : ℕ) : ℤ := 95 - 5 * (n - 1)

theorem terms_before_five : 
  (∃ n : ℕ, arithmeticSequence n = 5) ∧ 
  (∀ k : ℕ, k < 19 → arithmeticSequence k > 5) :=
by sorry

end NUMINAMATH_CALUDE_terms_before_five_l2283_228369


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2283_228379

theorem imaginary_part_of_complex_number :
  (Complex.im ((2 : ℂ) - Complex.I * Complex.I)) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2283_228379


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2283_228339

theorem quadratic_equation_solution :
  let f (x : ℂ) := 2 * (5 * x^2 + 4 * x + 3) - 6
  let g (x : ℂ) := -3 * (2 - 4 * x)
  ∀ x : ℂ, f x = g x ↔ x = (1 + Complex.I * Real.sqrt 14) / 5 ∨ x = (1 - Complex.I * Real.sqrt 14) / 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2283_228339


namespace NUMINAMATH_CALUDE_archie_marbles_problem_l2283_228357

/-- The number of marbles Archie started with. -/
def initial_marbles : ℕ := 100

/-- The fraction of marbles Archie keeps after losing some in the street. -/
def street_loss_fraction : ℚ := 2/5

/-- The fraction of remaining marbles Archie keeps after losing some in the sewer. -/
def sewer_loss_fraction : ℚ := 1/2

/-- The number of marbles Archie has left at the end. -/
def final_marbles : ℕ := 20

theorem archie_marbles_problem :
  (↑final_marbles : ℚ) = ↑initial_marbles * street_loss_fraction * sewer_loss_fraction :=
by sorry

end NUMINAMATH_CALUDE_archie_marbles_problem_l2283_228357


namespace NUMINAMATH_CALUDE_quadratic_properties_l2283_228311

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 4*x + 3

-- Define the domain
def domain : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

-- Theorem statement
theorem quadratic_properties :
  (∀ x ∈ domain, f (-x + 4) = f x) ∧  -- Axis of symmetry at x = 2
  (f 2 = 7) ∧  -- Vertex at (2, 7)
  (∀ x ∈ domain, f x ≤ 7) ∧  -- Maximum value
  (∀ x ∈ domain, f x ≥ 6) ∧  -- Minimum value
  (∃ x ∈ domain, f x = 7) ∧  -- Maximum is attained
  (∃ x ∈ domain, f x = 6) :=  -- Minimum is attained
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2283_228311


namespace NUMINAMATH_CALUDE_progression_existence_l2283_228396

theorem progression_existence (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) :
  (∃ q : ℝ, q > 0 ∧ b = a * q ∧ c = b * q) ∧
  ¬(∃ d : ℝ, b = a + d ∧ c = b + d) := by
  sorry

end NUMINAMATH_CALUDE_progression_existence_l2283_228396


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_l2283_228327

theorem larger_solution_quadratic : ∃ (x y : ℝ), x ≠ y ∧ 
  x^2 - 9*x - 22 = 0 ∧ 
  y^2 - 9*y - 22 = 0 ∧ 
  (∀ z : ℝ, z^2 - 9*z - 22 = 0 → z = x ∨ z = y) ∧
  max x y = 11 := by
sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_l2283_228327


namespace NUMINAMATH_CALUDE_radius_of_special_isosceles_triangle_l2283_228333

/-- Represents an isosceles triangle with a circumscribed circle. -/
structure IsoscelesTriangleWithCircle where
  /-- The length of the base of the isosceles triangle -/
  base : ℝ
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The two equal sides of the triangle are each twice the length of the base -/
  equal_sides_twice_base : base > 0
  /-- The perimeter in inches equals the area of the circumscribed circle in square inches -/
  perimeter_equals_circle_area : 5 * base = π * radius^2

/-- 
The radius of the circumscribed circle of an isosceles triangle is 2√5/π inches,
given that the perimeter in inches equals the area of the circumscribed circle in square inches,
and the two equal sides of the triangle are each twice the length of the base.
-/
theorem radius_of_special_isosceles_triangle (t : IsoscelesTriangleWithCircle) : 
  t.radius = 2 * Real.sqrt 5 / π :=
by sorry

end NUMINAMATH_CALUDE_radius_of_special_isosceles_triangle_l2283_228333


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_statement_l2283_228325

theorem negation_of_absolute_value_statement (x : ℝ) :
  ¬(abs x ≤ 3 ∨ abs x > 5) ↔ (abs x > 3 ∧ abs x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_statement_l2283_228325


namespace NUMINAMATH_CALUDE_root_implies_range_l2283_228354

-- Define the function f(x) = ax^2 - 2ax + a - 9
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + a - 9

-- Define the property that f has at least one root in (-2, 0)
def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -2 < x ∧ x < 0 ∧ f a x = 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -9 ∨ (1 < a ∧ a < 9) ∨ 9 < a

-- State the theorem
theorem root_implies_range :
  ∀ a : ℝ, has_root_in_interval a → a_range a :=
sorry

end NUMINAMATH_CALUDE_root_implies_range_l2283_228354


namespace NUMINAMATH_CALUDE_percentage_of_210_l2283_228388

theorem percentage_of_210 : (33 + 1/3 : ℚ) / 100 * 210 = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_210_l2283_228388
