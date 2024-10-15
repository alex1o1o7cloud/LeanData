import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_perfect_square_and_cube_l3975_397567

theorem smallest_n_perfect_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 4 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 5 * x = y^2) → (∃ (z : ℕ), 4 * x = z^3) → x ≥ n) ∧
  n = 625000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_square_and_cube_l3975_397567


namespace NUMINAMATH_CALUDE_second_to_third_quadrant_l3975_397549

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrants
def isSecondQuadrant (p : Point2D) : Prop := p.x < 0 ∧ p.y > 0
def isThirdQuadrant (p : Point2D) : Prop := p.x < 0 ∧ p.y < 0

-- Define the transformation from P to Q
def transformPtoQ (p : Point2D) : Point2D :=
  { x := -p.y, y := p.x }

-- Theorem statement
theorem second_to_third_quadrant (a b : ℝ) :
  let p := Point2D.mk a b
  let q := transformPtoQ p
  isSecondQuadrant p → isThirdQuadrant q := by
  sorry

end NUMINAMATH_CALUDE_second_to_third_quadrant_l3975_397549


namespace NUMINAMATH_CALUDE_second_tank_volume_l3975_397523

/-- Represents the capacity of each tank in liters -/
def tank_capacity : ℝ := 1000

/-- Represents the volume of water in the first tank in liters -/
def first_tank_volume : ℝ := 300

/-- Represents the fraction of the second tank that is filled -/
def second_tank_fill_ratio : ℝ := 0.45

/-- Represents the additional water needed to fill both tanks in liters -/
def additional_water_needed : ℝ := 1250

/-- Theorem stating that the second tank contains 450 liters of water -/
theorem second_tank_volume :
  let second_tank_volume := second_tank_fill_ratio * tank_capacity
  second_tank_volume = 450 := by sorry

end NUMINAMATH_CALUDE_second_tank_volume_l3975_397523


namespace NUMINAMATH_CALUDE_always_positive_l3975_397558

theorem always_positive (x y : ℝ) : x^2 - 4*x + y^2 + 13 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_l3975_397558


namespace NUMINAMATH_CALUDE_two_intersecting_lines_l3975_397583

/-- A parabola defined by the equation y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- A point (2,4) that lies on the parabola -/
def point : ℝ × ℝ := (2, 4)

/-- A function that returns the number of lines intersecting the parabola at exactly one point -/
def num_intersecting_lines : ℕ := 2

/-- Theorem stating that there are exactly two lines intersecting the parabola at one point -/
theorem two_intersecting_lines :
  parabola point.1 point.2 ∧ num_intersecting_lines = 2 :=
sorry

end NUMINAMATH_CALUDE_two_intersecting_lines_l3975_397583


namespace NUMINAMATH_CALUDE_amys_pencils_l3975_397528

/-- Amy's pencil counting problem -/
theorem amys_pencils (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 3 → bought = 7 → total = initial + bought → total = 10 := by
  sorry

end NUMINAMATH_CALUDE_amys_pencils_l3975_397528


namespace NUMINAMATH_CALUDE_a_6_value_l3975_397577

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem a_6_value (a : ℕ → ℝ) (h1 : arithmetic_sequence a) 
  (h2 : a 2 + a 10 = -12) (h3 : a 2 * a 10 = -8) : a 6 = -6 := by
  sorry

end NUMINAMATH_CALUDE_a_6_value_l3975_397577


namespace NUMINAMATH_CALUDE_pizza_sharing_l3975_397536

theorem pizza_sharing (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ) 
  (h1 : total_slices = 78)
  (h2 : buzz_ratio = 5)
  (h3 : waiter_ratio = 8) :
  waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio)) - 20 = 28 :=
by sorry

end NUMINAMATH_CALUDE_pizza_sharing_l3975_397536


namespace NUMINAMATH_CALUDE_scott_total_oranges_l3975_397563

/-- The number of boxes Scott has for oranges. -/
def num_boxes : ℕ := 8

/-- The number of oranges that must be in each box. -/
def oranges_per_box : ℕ := 7

/-- Theorem stating that Scott has 56 oranges in total. -/
theorem scott_total_oranges : num_boxes * oranges_per_box = 56 := by
  sorry

end NUMINAMATH_CALUDE_scott_total_oranges_l3975_397563


namespace NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_for_prop_b_l3975_397509

theorem prop_a_necessary_not_sufficient_for_prop_b :
  (∀ (a b : ℝ), (1 / b < 1 / a ∧ 1 / a < 0) → a * b > b ^ 2) ∧
  (∃ (a b : ℝ), a * b > b ^ 2 ∧ ¬(1 / b < 1 / a ∧ 1 / a < 0)) := by
  sorry

end NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_for_prop_b_l3975_397509


namespace NUMINAMATH_CALUDE_gift_distribution_count_l3975_397580

/-- The number of bags of gifts -/
def num_bags : ℕ := 5

/-- The number of elderly people -/
def num_people : ℕ := 4

/-- The number of ways to distribute consecutive pairs -/
def consecutive_pairs : ℕ := 4

/-- The number of ways to arrange the remaining bags -/
def remaining_arrangements : ℕ := 24  -- This is A_4^4

/-- The total number of distribution methods -/
def total_distributions : ℕ := consecutive_pairs * remaining_arrangements

theorem gift_distribution_count :
  total_distributions = 96 :=
sorry

end NUMINAMATH_CALUDE_gift_distribution_count_l3975_397580


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_min_value_achieved_l3975_397550

theorem min_value_of_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ (1 / a₀ + 2 / b₀ = 3 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_min_value_achieved_l3975_397550


namespace NUMINAMATH_CALUDE_production_rates_l3975_397594

/-- The production rates of two workers --/
theorem production_rates (total_rate : ℝ) (a_parts b_parts : ℕ) 
  (h1 : total_rate = 35)
  (h2 : (a_parts : ℝ) / x = (b_parts : ℝ) / (total_rate - x))
  (h3 : a_parts = 90)
  (h4 : b_parts = 120) :
  ∃ (x y : ℝ), x + y = total_rate ∧ x = 15 ∧ y = 20 :=
sorry

end NUMINAMATH_CALUDE_production_rates_l3975_397594


namespace NUMINAMATH_CALUDE_plan1_more_cost_effective_when_sessions_gt_8_l3975_397507

/-- Represents the cost of a fitness plan based on the number of sessions -/
structure FitnessPlan where
  fixedFee : ℕ
  perSessionFee : ℕ

/-- Calculates the total cost for a given plan and number of sessions -/
def totalCost (plan : FitnessPlan) (sessions : ℕ) : ℕ :=
  plan.fixedFee + plan.perSessionFee * sessions

/-- Theorem: Plan 1 is more cost-effective than Plan 2 when sessions > 8 -/
theorem plan1_more_cost_effective_when_sessions_gt_8
  (plan1 : FitnessPlan)
  (plan2 : FitnessPlan)
  (h1 : plan1.fixedFee = 80 ∧ plan1.perSessionFee = 10)
  (h2 : plan2.fixedFee = 0 ∧ plan2.perSessionFee = 20)
  : ∀ sessions, sessions > 8 → totalCost plan1 sessions < totalCost plan2 sessions := by
  sorry

#check plan1_more_cost_effective_when_sessions_gt_8

end NUMINAMATH_CALUDE_plan1_more_cost_effective_when_sessions_gt_8_l3975_397507


namespace NUMINAMATH_CALUDE_fraction_equals_44_l3975_397564

theorem fraction_equals_44 : (2450 - 2377)^2 / 121 = 44 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_44_l3975_397564


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_min_perimeter_achieved_l3975_397575

theorem min_perimeter_rectangle (length width : ℝ) : 
  length > 0 → width > 0 → length * width = 64 → 
  2 * (length + width) ≥ 32 := by
  sorry

theorem min_perimeter_achieved (length width : ℝ) :
  length > 0 → width > 0 → length * width = 64 →
  2 * (length + width) = 32 ↔ length = 8 ∧ width = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_min_perimeter_achieved_l3975_397575


namespace NUMINAMATH_CALUDE_paul_pencil_sales_l3975_397500

def pencils_sold (daily_production : ℕ) (work_days : ℕ) (starting_stock : ℕ) (ending_stock : ℕ) : ℕ :=
  daily_production * work_days + starting_stock - ending_stock

theorem paul_pencil_sales : pencils_sold 100 5 80 230 = 350 := by
  sorry

end NUMINAMATH_CALUDE_paul_pencil_sales_l3975_397500


namespace NUMINAMATH_CALUDE_clock_adjustment_theorem_l3975_397569

/-- Represents the gain of the clock in minutes per day -/
def clock_gain : ℚ := 13/4

/-- Represents the number of days between May 1st 10 A.M. and May 10th 2 P.M. -/
def days : ℚ := 9 + 4/24

/-- Calculates the adjustment needed for the clock -/
def adjustment (gain : ℚ) (time : ℚ) : ℚ := gain * time

/-- Theorem stating that the adjustment is approximately 29.8 minutes -/
theorem clock_adjustment_theorem :
  ∃ ε > 0, abs (adjustment clock_gain days - 29.8) < ε :=
sorry

end NUMINAMATH_CALUDE_clock_adjustment_theorem_l3975_397569


namespace NUMINAMATH_CALUDE_choose_two_from_three_l3975_397522

theorem choose_two_from_three : Nat.choose 3 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_three_l3975_397522


namespace NUMINAMATH_CALUDE_principal_amount_l3975_397589

/-- Proves that given the conditions of the problem, the principal amount is 300 --/
theorem principal_amount (P : ℝ) : 
  (P * 4 * 8 / 100 = P - 204) → P = 300 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l3975_397589


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3975_397587

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def FormGeometricSequence (a : ℕ → ℚ) (i j k : ℕ) : Prop :=
  (a j) ^ 2 = a i * a k

theorem arithmetic_sequence_ratio (a : ℕ → ℚ) (d : ℚ) :
  ArithmeticSequence a d →
  FormGeometricSequence a 2 3 9 →
  (a 4 + a 5 + a 6) / (a 2 + a 3 + a 4) = 8 / 3 := by
  sorry

#check arithmetic_sequence_ratio

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3975_397587


namespace NUMINAMATH_CALUDE_f_plus_g_at_one_l3975_397524

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_g_at_one
  (h_even : is_even f)
  (h_odd : is_odd g)
  (h_diff : ∀ x, f x - g x = x^3 + x^2 + 1) :
  f 1 + g 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_plus_g_at_one_l3975_397524


namespace NUMINAMATH_CALUDE_race_end_count_l3975_397503

/-- Represents the total number of people in all cars at the end of a race with given conditions. -/
def total_people_at_end (num_cars : ℕ) (initial_people_per_car : ℕ) 
  (first_quarter_gain : ℕ) (half_way_gain : ℕ) (three_quarter_gain : ℕ) : ℕ :=
  num_cars * (initial_people_per_car + first_quarter_gain + half_way_gain + three_quarter_gain)

/-- Theorem stating that under the given race conditions, the total number of people at the end is 450. -/
theorem race_end_count : 
  total_people_at_end 50 4 2 2 1 = 450 := by
  sorry

#eval total_people_at_end 50 4 2 2 1

end NUMINAMATH_CALUDE_race_end_count_l3975_397503


namespace NUMINAMATH_CALUDE_jolene_babysitting_charge_l3975_397506

theorem jolene_babysitting_charge 
  (num_families : ℕ) 
  (num_cars : ℕ) 
  (car_wash_fee : ℚ) 
  (total_raised : ℚ) :
  num_families = 4 →
  num_cars = 5 →
  car_wash_fee = 12 →
  total_raised = 180 →
  (num_families : ℚ) * (total_raised - num_cars * car_wash_fee) / num_families = 30 := by
  sorry

end NUMINAMATH_CALUDE_jolene_babysitting_charge_l3975_397506


namespace NUMINAMATH_CALUDE_spinner_probability_l3975_397511

theorem spinner_probability (p_D p_E p_FG : ℚ) : 
  p_D = 1/4 → p_E = 1/3 → p_D + p_E + p_FG = 1 → p_FG = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3975_397511


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l3975_397585

theorem sine_cosine_inequality (c : ℝ) :
  (∀ x : ℝ, 3 * Real.sin x - 4 * Real.cos x + c > 0) ↔ c > 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l3975_397585


namespace NUMINAMATH_CALUDE_least_number_with_given_remainders_l3975_397588

theorem least_number_with_given_remainders :
  ∃ (n : ℕ), n > 1 ∧ n % 25 = 1 ∧ n % 7 = 1 ∧
  ∀ (m : ℕ), m > 1 → m % 25 = 1 → m % 7 = 1 → n ≤ m :=
by
  use 176
  sorry

end NUMINAMATH_CALUDE_least_number_with_given_remainders_l3975_397588


namespace NUMINAMATH_CALUDE_test_score_problem_l3975_397586

/-- Prove that given a test with 30 questions, where each correct answer is worth 20 points
    and each incorrect answer deducts 5 points, if all questions are answered and the total
    score is 325, then the number of correct answers is 19. -/
theorem test_score_problem (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) 
    (total_score : ℕ) (h1 : total_questions = 30) (h2 : correct_points = 20) 
    (h3 : incorrect_points = 5) (h4 : total_score = 325) : 
    ∃ (correct_answers : ℕ), 
      correct_answers * correct_points + 
      (total_questions - correct_answers) * (correct_points - incorrect_points) = 
      total_score ∧ correct_answers = 19 := by
  sorry

end NUMINAMATH_CALUDE_test_score_problem_l3975_397586


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_of_squares_l3975_397525

theorem polynomial_coefficient_sum_of_squares 
  (a b c d e f : ℤ) 
  (h : ∀ x : ℝ, 8 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) : 
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 356 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_of_squares_l3975_397525


namespace NUMINAMATH_CALUDE_inequality_solution_l3975_397572

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom g_never_zero : ∀ x, g x ≠ 0
axiom condition_neg : ∀ x, x < 0 → f x * g x - f x * (deriv g x) > 0
axiom f_3_eq_0 : f 3 = 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ (0 < x ∧ x < 3)}

-- State the theorem
theorem inequality_solution :
  {x : ℝ | f x * g x < 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3975_397572


namespace NUMINAMATH_CALUDE_min_distance_exp_ln_curves_l3975_397515

/-- The minimum distance between a point on y = e^x and a point on y = ln x is √2 -/
theorem min_distance_exp_ln_curves : ∃ (d : ℝ),
  d = Real.sqrt 2 ∧
  ∀ (x₁ x₂ : ℝ),
    let P := (x₁, Real.exp x₁)
    let Q := (x₂, Real.log x₂)
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_exp_ln_curves_l3975_397515


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3975_397530

/-- Given two hyperbolas with equations x²/9 - y²/16 = 1 and y²/25 - x²/M = 1,
    prove that if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/25 - x^2/M = 1) →
  (∀ k : ℝ, (∃ x y : ℝ, y = k*x ∧ x^2/9 - y^2/16 = 1) ↔
            (∃ x y : ℝ, y = k*x ∧ y^2/25 - x^2/M = 1)) →
  M = 225/16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3975_397530


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3975_397593

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 2 + a 5 = 18
  product_property : a 3 * a 4 = 32

/-- The theorem stating that for the given arithmetic sequence, a_n = 128 implies n = 8 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  ∃ n : ℕ, seq.a n = 128 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3975_397593


namespace NUMINAMATH_CALUDE_elephant_received_503_pills_l3975_397527

/-- The number of pills given to four animals by Dr. Aibolit -/
def total_pills : ℕ := 2006

/-- The number of pills received by the crocodile -/
def crocodile_pills : ℕ := sorry

/-- The number of pills received by the rhinoceros -/
def rhinoceros_pills : ℕ := crocodile_pills + 1

/-- The number of pills received by the hippopotamus -/
def hippopotamus_pills : ℕ := rhinoceros_pills + 1

/-- The number of pills received by the elephant -/
def elephant_pills : ℕ := hippopotamus_pills + 1

/-- Theorem stating that the elephant received 503 pills -/
theorem elephant_received_503_pills : 
  crocodile_pills + rhinoceros_pills + hippopotamus_pills + elephant_pills = total_pills ∧ 
  elephant_pills = 503 := by
  sorry

end NUMINAMATH_CALUDE_elephant_received_503_pills_l3975_397527


namespace NUMINAMATH_CALUDE_passes_through_fixed_point_not_in_fourth_quadrant_min_area_and_equation_l3975_397526

/-- Definition of the line l with parameter k -/
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

/-- The fixed point that the line passes through -/
def fixed_point : ℝ × ℝ := (-2, 1)

/-- Theorem 1: The line passes through the fixed point for all real k -/
theorem passes_through_fixed_point (k : ℝ) :
  line_l k (fixed_point.1) (fixed_point.2) := by sorry

/-- Theorem 2: The line does not pass through the fourth quadrant iff k ≥ 0 -/
theorem not_in_fourth_quadrant (k : ℝ) :
  (∀ x y, x > 0 → y < 0 → ¬line_l k x y) ↔ k ≥ 0 := by sorry

/-- Function to calculate the area of the triangle formed by the line's intersections -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  if k ≠ 0 then
    (1 + 2 * k) * ((1 + 2 * k) / k) / 2
  else 0

/-- Theorem 3: The minimum area of the triangle is 4, occurring when k = 1/2 -/
theorem min_area_and_equation :
  (∀ k, k > 0 → triangle_area k ≥ 4) ∧
  triangle_area (1/2) = 4 ∧
  line_l (1/2) x y ↔ x - 2 * y + 4 = 0 := by sorry

end NUMINAMATH_CALUDE_passes_through_fixed_point_not_in_fourth_quadrant_min_area_and_equation_l3975_397526


namespace NUMINAMATH_CALUDE_tan_theta_value_l3975_397554

theorem tan_theta_value (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2))
  (h2 : 12 / Real.sin θ + 12 / Real.cos θ = 35) :
  Real.tan θ = 3/4 ∨ Real.tan θ = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l3975_397554


namespace NUMINAMATH_CALUDE_ellipse_foci_l3975_397510

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

/-- The coordinates of a focus -/
def focus_coordinate : ℝ × ℝ := (4, 0)

/-- Theorem stating that the foci of the given ellipse are at (±4, 0) -/
theorem ellipse_foci :
  ∀ (x y : ℝ), ellipse_equation x y → 
    (x = focus_coordinate.1 ∧ y = focus_coordinate.2) ∨
    (x = -focus_coordinate.1 ∧ y = focus_coordinate.2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_l3975_397510


namespace NUMINAMATH_CALUDE_minimum_employees_needed_l3975_397552

theorem minimum_employees_needed (forest_employees : ℕ) (marine_employees : ℕ) (both_employees : ℕ)
  (h1 : forest_employees = 95)
  (h2 : marine_employees = 80)
  (h3 : both_employees = 35)
  (h4 : both_employees ≤ forest_employees ∧ both_employees ≤ marine_employees) :
  forest_employees + marine_employees - both_employees = 140 :=
by sorry

end NUMINAMATH_CALUDE_minimum_employees_needed_l3975_397552


namespace NUMINAMATH_CALUDE_anusha_share_l3975_397508

theorem anusha_share (total : ℕ) (a b e : ℚ) : 
  total = 378 →
  12 * a = 8 * b →
  12 * a = 6 * e →
  a + b + e = total →
  a = 84 := by
sorry

end NUMINAMATH_CALUDE_anusha_share_l3975_397508


namespace NUMINAMATH_CALUDE_apartment_rent_theorem_l3975_397599

/-- Calculates the total rent paid over a period of time with different monthly rates -/
def totalRent (months1 : ℕ) (rate1 : ℕ) (months2 : ℕ) (rate2 : ℕ) : ℕ :=
  months1 * rate1 + months2 * rate2

theorem apartment_rent_theorem :
  totalRent 36 300 24 350 = 19200 := by
  sorry

end NUMINAMATH_CALUDE_apartment_rent_theorem_l3975_397599


namespace NUMINAMATH_CALUDE_marble_pairs_l3975_397597

-- Define the set of marbles
def Marble : Type := 
  Sum (Fin 1) (Sum (Fin 1) (Sum (Fin 1) (Sum (Fin 3) (Fin 2))))

-- Define the function to count distinct pairs
def countDistinctPairs (s : Finset Marble) : ℕ := sorry

-- State the theorem
theorem marble_pairs : 
  let s : Finset Marble := sorry
  countDistinctPairs s = 12 := by sorry

end NUMINAMATH_CALUDE_marble_pairs_l3975_397597


namespace NUMINAMATH_CALUDE_prob_three_consecutive_in_ten_l3975_397576

/-- The number of ways to arrange n items -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n items with a block of k consecutive items -/
def arrangements_with_block (n k : ℕ) : ℕ := (n - k + 1) * k.factorial * (n - k).factorial

/-- The probability of k specific items being consecutive in a random arrangement of n items -/
def prob_consecutive (n k : ℕ) : ℚ :=
  (arrangements_with_block n k : ℚ) / (arrangements n : ℚ)

theorem prob_three_consecutive_in_ten :
  prob_consecutive 10 3 = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_prob_three_consecutive_in_ten_l3975_397576


namespace NUMINAMATH_CALUDE_norm_scalar_multiple_l3975_397561

theorem norm_scalar_multiple {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] 
  (v : V) (h : ‖v‖ = 5) : ‖(4 : ℝ) • v‖ = 20 := by
  sorry

end NUMINAMATH_CALUDE_norm_scalar_multiple_l3975_397561


namespace NUMINAMATH_CALUDE_tangerine_count_l3975_397538

theorem tangerine_count (apples pears tangerines : ℕ) : 
  apples = 45 →
  apples = pears + 21 →
  tangerines = pears + 18 →
  tangerines = 42 := by
sorry

end NUMINAMATH_CALUDE_tangerine_count_l3975_397538


namespace NUMINAMATH_CALUDE_unique_integral_solution_l3975_397573

theorem unique_integral_solution (x y z n : ℤ) 
  (eq1 : x * y + y * z + z * x = 3 * n^2 - 1)
  (eq2 : x + y + z = 3 * n)
  (h1 : x ≥ y)
  (h2 : y ≥ z) :
  x = n + 1 ∧ y = n ∧ z = n - 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integral_solution_l3975_397573


namespace NUMINAMATH_CALUDE_least_integer_with_remainders_l3975_397591

theorem least_integer_with_remainders : ∃! n : ℕ,
  (∀ m : ℕ, m < n →
    (m % 5 ≠ 4 ∨ m % 6 ≠ 5 ∨ m % 7 ≠ 6 ∨ m % 8 ≠ 7 ∨ m % 9 ≠ 8 ∨ m % 10 ≠ 9)) ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  n = 2519 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_remainders_l3975_397591


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l3975_397574

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y - 2 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := 2 * x - 5 * y + b = 0

-- Define perpendicularity of two lines
def perpendicular (a b : ℝ) : Prop := a * 2 + 4 * (-5) = 0

-- Define the foot of the perpendicular
def foot_of_perpendicular (a b c : ℝ) : Prop := l₁ a 1 c ∧ l₂ b 1 c

-- Theorem statement
theorem perpendicular_lines_sum (a b c : ℝ) :
  perpendicular a b →
  foot_of_perpendicular a b c →
  a + b + c = -4 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l3975_397574


namespace NUMINAMATH_CALUDE_certain_number_proof_l3975_397534

theorem certain_number_proof : ∃ (n : ℕ), n + 3327 = 13200 ∧ n = 9873 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3975_397534


namespace NUMINAMATH_CALUDE_profit_margin_calculation_l3975_397540

/-- Profit margin calculation -/
theorem profit_margin_calculation (n : ℝ) (C S M : ℝ) 
  (h1 : M = (1 / n) * (2 * C - S)) 
  (h2 : S - M = C) : 
  M = S / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_profit_margin_calculation_l3975_397540


namespace NUMINAMATH_CALUDE_no_number_with_digit_product_1560_l3975_397529

/-- The product of the decimal digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Theorem stating that no natural number has a digit product of 1560 -/
theorem no_number_with_digit_product_1560 : 
  ¬ ∃ (n : ℕ), digit_product n = 1560 := by sorry

end NUMINAMATH_CALUDE_no_number_with_digit_product_1560_l3975_397529


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3975_397584

/-- The eccentricity of a hyperbola defined by x²/(1+m) - y²/(1-m) = 1 with m > 0 is between 1 and √2 -/
theorem hyperbola_eccentricity_range (m : ℝ) (h1 : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / (1 + m) - y^2 / (1 - m) = 1}
  let e := Real.sqrt 2 / Real.sqrt (1 + m)
  1 < e ∧ e < Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3975_397584


namespace NUMINAMATH_CALUDE_first_place_percentage_l3975_397557

/-- 
Given a pot of money where:
- 8 people each contribute $5
- Third place gets $4
- Second and third place split the remaining money after first place
Prove that first place gets 80% of the total money
-/
theorem first_place_percentage (total_people : Nat) (contribution : ℕ) (third_place_prize : ℕ) :
  total_people = 8 →
  contribution = 5 →
  third_place_prize = 4 →
  (((total_people * contribution - 2 * third_place_prize) : ℚ) / (total_people * contribution)) = 4/5 := by
  sorry

#check first_place_percentage

end NUMINAMATH_CALUDE_first_place_percentage_l3975_397557


namespace NUMINAMATH_CALUDE_fencing_required_l3975_397590

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 400 ∧ uncovered_side = 20 → 
  ∃ (width : ℝ), area = uncovered_side * width ∧ uncovered_side + 2 * width = 60 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l3975_397590


namespace NUMINAMATH_CALUDE_set_properties_l3975_397539

def closed_under_transformation (A : Set ℝ) : Prop :=
  ∀ a ∈ A, (1 + a) / (1 - a) ∈ A

theorem set_properties (A : Set ℝ) (h : closed_under_transformation A) :
  (2 ∈ A → A = {2, -3, -1/2, 1/3}) ∧
  (0 ∉ A ∧ ∃ a ∈ A, A = {a, -a/(a+1), -1/(a+1), 1/(a-1)}) :=
sorry

end NUMINAMATH_CALUDE_set_properties_l3975_397539


namespace NUMINAMATH_CALUDE_remainder_count_l3975_397514

theorem remainder_count : 
  (Finset.filter (fun n => Nat.mod 2017 n = 1 ∨ Nat.mod 2017 n = 2) (Finset.range 2018)).card = 43 := by
  sorry

end NUMINAMATH_CALUDE_remainder_count_l3975_397514


namespace NUMINAMATH_CALUDE_jesse_stamp_ratio_l3975_397553

theorem jesse_stamp_ratio :
  let total_stamps : ℕ := 444
  let european_stamps : ℕ := 333
  let asian_stamps : ℕ := total_stamps - european_stamps
  (european_stamps : ℚ) / (asian_stamps : ℚ) = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_jesse_stamp_ratio_l3975_397553


namespace NUMINAMATH_CALUDE_matildas_chocolate_bars_l3975_397521

/-- Proves that Matilda initially had 4 chocolate bars given the problem conditions -/
theorem matildas_chocolate_bars (total_people : ℕ) (sisters : ℕ) (fathers_remaining : ℕ) 
  (mothers_share : ℕ) (fathers_eaten : ℕ) :
  total_people = sisters + 1 →
  sisters = 4 →
  fathers_remaining = 5 →
  mothers_share = 3 →
  fathers_eaten = 2 →
  ∃ (initial_bars : ℕ),
    initial_bars = (fathers_remaining + mothers_share + fathers_eaten) * 2 / total_people ∧
    initial_bars = 4 :=
by sorry

end NUMINAMATH_CALUDE_matildas_chocolate_bars_l3975_397521


namespace NUMINAMATH_CALUDE_danny_fish_tank_theorem_l3975_397519

/-- Represents the fish tank contents and sales --/
structure FishTank where
  initialGuppies : Nat
  initialAngelfish : Nat
  initialTigerSharks : Nat
  initialOscarFish : Nat
  soldGuppies : Nat
  soldAngelfish : Nat
  soldTigerSharks : Nat
  soldOscarFish : Nat

/-- Calculates the remaining fish in the tank --/
def remainingFish (tank : FishTank) : Nat :=
  (tank.initialGuppies + tank.initialAngelfish + tank.initialTigerSharks + tank.initialOscarFish) -
  (tank.soldGuppies + tank.soldAngelfish + tank.soldTigerSharks + tank.soldOscarFish)

/-- Theorem stating that the remaining fish in Danny's tank is 198 --/
theorem danny_fish_tank_theorem (tank : FishTank) 
  (h1 : tank.initialGuppies = 94)
  (h2 : tank.initialAngelfish = 76)
  (h3 : tank.initialTigerSharks = 89)
  (h4 : tank.initialOscarFish = 58)
  (h5 : tank.soldGuppies = 30)
  (h6 : tank.soldAngelfish = 48)
  (h7 : tank.soldTigerSharks = 17)
  (h8 : tank.soldOscarFish = 24) :
  remainingFish tank = 198 := by
  sorry

end NUMINAMATH_CALUDE_danny_fish_tank_theorem_l3975_397519


namespace NUMINAMATH_CALUDE_initial_value_problem_l3975_397596

theorem initial_value_problem (x : ℤ) : x + 335 = 456 * (x + 335) / 456 → x = 121 :=
by sorry

end NUMINAMATH_CALUDE_initial_value_problem_l3975_397596


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l3975_397560

theorem retailer_profit_percentage
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (discount_percentage : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : discount_percentage = 0.1)
  : (((retail_price * (1 - discount_percentage)) - wholesale_price) / wholesale_price) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l3975_397560


namespace NUMINAMATH_CALUDE_circle_area_and_circumference_l3975_397547

/-- Given a circle with diameter endpoints at (1,1) and (8,6), prove its area and circumference -/
theorem circle_area_and_circumference :
  let C : ℝ × ℝ := (1, 1)
  let D : ℝ × ℝ := (8, 6)
  let diameter := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let radius := diameter / 2
  let area := π * radius^2
  let circumference := 2 * π * radius
  (area = 74 * π / 4) ∧ (circumference = Real.sqrt 74 * π) := by
  sorry


end NUMINAMATH_CALUDE_circle_area_and_circumference_l3975_397547


namespace NUMINAMATH_CALUDE_missy_dog_yells_l3975_397548

/-- The number of times Missy yells at her dogs -/
def total_yells : ℕ := 60

/-- The ratio of yells at the stubborn dog to yells at the obedient dog -/
def stubborn_to_obedient_ratio : ℕ := 4

/-- The number of times Missy yells at the obedient dog -/
def obedient_dog_yells : ℕ := 12

theorem missy_dog_yells :
  obedient_dog_yells * (stubborn_to_obedient_ratio + 1) = total_yells :=
sorry

end NUMINAMATH_CALUDE_missy_dog_yells_l3975_397548


namespace NUMINAMATH_CALUDE_cyclical_sequence_value_of_3_cyclical_sequence_properties_l3975_397502

def cyclical_sequence (n : ℕ) : ℕ :=
  match n % 5 with
  | 1 => 6
  | 2 => 12
  | 3 => 18  -- This is what we want to prove
  | 4 => 24
  | 0 => 30
  | _ => 0   -- This case should never occur

theorem cyclical_sequence_value_of_3 :
  cyclical_sequence 3 = 18 :=
by
  sorry

theorem cyclical_sequence_properties :
  (cyclical_sequence 1 = 6) ∧
  (cyclical_sequence 2 = 12) ∧
  (cyclical_sequence 4 = 24) ∧
  (cyclical_sequence 5 = 30) ∧
  (cyclical_sequence 6 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_cyclical_sequence_value_of_3_cyclical_sequence_properties_l3975_397502


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_P_is_640_l3975_397598

-- Define the points
def Q : ℝ × ℝ := (0, 0)
def R : ℝ × ℝ := (307, 0)
def S : ℝ × ℝ := (450, 280)
def T : ℝ × ℝ := (460, 290)

-- Define the areas of the triangles
def area_PQR : ℝ := 1739
def area_PST : ℝ := 6956

-- Define the function to calculate the sum of possible x-coordinates of P
noncomputable def sum_of_x_coordinates_P : ℝ := sorry

-- Theorem statement
theorem sum_of_x_coordinates_P_is_640 :
  sum_of_x_coordinates_P = 640 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_P_is_640_l3975_397598


namespace NUMINAMATH_CALUDE_indexCardsCostForCarl_l3975_397513

/-- Represents the cost of index cards for Carl's students. -/
def indexCardsCost (
  sixthGradeCards : ℕ
  ) (seventhGradeCards : ℕ
  ) (eighthGradeCards : ℕ
  ) (periodsPerDay : ℕ
  ) (sixthGradersPerPeriod : ℕ
  ) (seventhGradersPerPeriod : ℕ
  ) (eighthGradersPerPeriod : ℕ
  ) (cardsPerPack : ℕ
  ) (costPerPack : ℕ
  ) : ℕ :=
  let totalCards := 
    (sixthGradeCards * sixthGradersPerPeriod + 
     seventhGradeCards * seventhGradersPerPeriod + 
     eighthGradeCards * eighthGradersPerPeriod) * periodsPerDay
  let packsNeeded := (totalCards + cardsPerPack - 1) / cardsPerPack
  packsNeeded * costPerPack

/-- Theorem stating the total cost of index cards for Carl's students. -/
theorem indexCardsCostForCarl : 
  indexCardsCost 8 10 12 6 20 25 30 50 3 = 279 := by
  sorry

end NUMINAMATH_CALUDE_indexCardsCostForCarl_l3975_397513


namespace NUMINAMATH_CALUDE_rabbit_area_l3975_397504

theorem rabbit_area (ear_area : ℝ) (total_area : ℝ) : 
  ear_area = 10 → ear_area = (1/8) * total_area → total_area = 80 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_area_l3975_397504


namespace NUMINAMATH_CALUDE_min_horseshoed_ponies_fraction_l3975_397542

/-- A ranch with horses and ponies -/
structure Ranch where
  horses : ℕ
  ponies : ℕ
  horseshoed_ponies : ℕ
  iceland_horseshoed_ponies : ℕ

/-- The conditions of the ranch problem -/
def ranch_conditions (r : Ranch) : Prop :=
  r.horses = r.ponies + 4 ∧
  r.horses + r.ponies ≥ 40 ∧
  r.iceland_horseshoed_ponies = (2 * r.horseshoed_ponies) / 3

/-- The theorem stating the minimum fraction of ponies with horseshoes -/
theorem min_horseshoed_ponies_fraction (r : Ranch) : 
  ranch_conditions r → r.horseshoed_ponies * 12 ≤ r.ponies := by
  sorry

#check min_horseshoed_ponies_fraction

end NUMINAMATH_CALUDE_min_horseshoed_ponies_fraction_l3975_397542


namespace NUMINAMATH_CALUDE_prob_less_than_two_defective_l3975_397512

/-- The probability of selecting fewer than 2 defective products -/
theorem prob_less_than_two_defective (total : Nat) (defective : Nat) (selected : Nat) 
  (h1 : total = 10) (h2 : defective = 3) (h3 : selected = 2) : 
  (Nat.choose (total - defective) selected + 
   Nat.choose (total - defective) (selected - 1) * Nat.choose defective 1) / 
  Nat.choose total selected = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_two_defective_l3975_397512


namespace NUMINAMATH_CALUDE_fraction_equality_implies_product_l3975_397532

theorem fraction_equality_implies_product (a b : ℝ) : 
  a / 2 = 3 / b → a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_product_l3975_397532


namespace NUMINAMATH_CALUDE_age_difference_l3975_397559

theorem age_difference (a b c d : ℕ) 
  (eq1 : a + b = b + c + 12)
  (eq2 : b + d = c + d + 8)
  (eq3 : d = a + 5) :
  c = a - 12 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3975_397559


namespace NUMINAMATH_CALUDE_opposite_numbers_expression_l3975_397565

theorem opposite_numbers_expression (m n : ℝ) (h : m + n = 0) :
  3 * (m - n) - (1/2) * (2 * m - 10 * n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_expression_l3975_397565


namespace NUMINAMATH_CALUDE_new_students_weight_l3975_397568

theorem new_students_weight (initial_count : ℕ) (replaced_weight1 replaced_weight2 avg_decrease : ℝ) :
  initial_count = 8 →
  replaced_weight1 = 85 →
  replaced_weight2 = 96 →
  avg_decrease = 7.5 →
  (initial_count : ℝ) * avg_decrease = (replaced_weight1 + replaced_weight2) - (new_student_weight1 + new_student_weight2) →
  new_student_weight1 + new_student_weight2 = 121 :=
by
  sorry

#check new_students_weight

end NUMINAMATH_CALUDE_new_students_weight_l3975_397568


namespace NUMINAMATH_CALUDE_dihedral_angle_bounds_l3975_397505

/-- A regular pyramid with an n-sided polygonal base -/
structure RegularPyramid where
  n : ℕ
  base_sides : n > 2

/-- The dihedral angle between two adjacent lateral faces of a regular pyramid -/
def dihedral_angle (p : RegularPyramid) : ℝ :=
  sorry

/-- Theorem: The dihedral angle in a regular pyramid is bounded -/
theorem dihedral_angle_bounds (p : RegularPyramid) :
  (((p.n - 2) / p.n : ℝ) * Real.pi) < dihedral_angle p ∧ dihedral_angle p < Real.pi :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_bounds_l3975_397505


namespace NUMINAMATH_CALUDE_parallelogram_count_is_392_l3975_397501

/-- Represents a parallelogram PQRS with the given properties -/
structure Parallelogram where
  q : ℕ+  -- x-coordinate of Q (also y-coordinate since Q is on y = x)
  s : ℕ+  -- x-coordinate of S
  m : ℕ   -- slope of line y = mx where S lies
  h_m_gt_one : m > 1
  h_area : (m - 1) * q * s = 250000

/-- Counts the number of valid parallelograms -/
def count_parallelograms : ℕ := sorry

/-- The main theorem stating that the count of valid parallelograms is 392 -/
theorem parallelogram_count_is_392 : count_parallelograms = 392 := by sorry

end NUMINAMATH_CALUDE_parallelogram_count_is_392_l3975_397501


namespace NUMINAMATH_CALUDE_compare_power_towers_l3975_397592

def power_tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (power_tower base n)

theorem compare_power_towers (n : ℕ) :
  (n ≥ 3 → power_tower 3 (n - 1) > power_tower 2 n) ∧
  (n ≥ 2 → power_tower 3 n > power_tower 4 (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_compare_power_towers_l3975_397592


namespace NUMINAMATH_CALUDE_blueberry_picking_difference_l3975_397517

theorem blueberry_picking_difference (annie kathryn ben : ℕ) : 
  annie = 8 →
  kathryn = annie + 2 →
  ben < kathryn →
  annie + kathryn + ben = 25 →
  kathryn - ben = 3 :=
by sorry

end NUMINAMATH_CALUDE_blueberry_picking_difference_l3975_397517


namespace NUMINAMATH_CALUDE_martha_guess_probability_l3975_397578

/-- Martha's guessing abilities -/
structure MarthaGuess where
  height_success : Rat
  weight_success : Rat
  child_height_success : Rat
  adult_height_success : Rat
  tight_clothes_weight_success : Rat
  loose_clothes_weight_success : Rat

/-- Represents a person Martha meets -/
inductive Person
  | Child : Bool → Person  -- Bool represents tight (true) or loose (false) clothes
  | Adult : Bool → Person

def martha : MarthaGuess :=
  { height_success := 5/6
  , weight_success := 6/8
  , child_height_success := 4/5
  , adult_height_success := 5/6
  , tight_clothes_weight_success := 3/4
  , loose_clothes_weight_success := 7/10 }

def people : List Person :=
  [Person.Child false, Person.Adult true, Person.Adult false]

/-- Calculates the probability of Martha guessing correctly for a specific person -/
def guessCorrectProb (m : MarthaGuess) (p : Person) : Rat :=
  match p with
  | Person.Child tight =>
      1 - (1 - m.child_height_success) * (1 - (if tight then m.tight_clothes_weight_success else m.loose_clothes_weight_success))
  | Person.Adult tight =>
      1 - (1 - m.adult_height_success) * (1 - (if tight then m.tight_clothes_weight_success else m.loose_clothes_weight_success))

/-- Theorem: The probability of Martha guessing correctly at least once for the given people is 7999/8000 -/
theorem martha_guess_probability :
  1 - (people.map (guessCorrectProb martha)).prod = 7999/8000 := by
  sorry


end NUMINAMATH_CALUDE_martha_guess_probability_l3975_397578


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3975_397546

/-- For an infinite geometric series with first term a and common ratio r,
    if the sum of the series starting from the fourth term is 1/27 times
    the sum of the original series, then r = 1/3. -/
theorem geometric_series_ratio (a r : ℝ) (h : |r| < 1) :
  (a * r^3 / (1 - r)) = (1 / 27) * (a / (1 - r)) →
  r = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3975_397546


namespace NUMINAMATH_CALUDE_set_operations_l3975_397579

-- Define the sets A and B
def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -2 ∨ x > 4}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x | -5 ≤ x ∧ x < -2}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x < -5 ∨ x ≥ -2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3975_397579


namespace NUMINAMATH_CALUDE_wedge_volume_l3975_397516

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (log_diameter : ℝ) (cut_angle : ℝ) : 
  log_diameter = 12 →
  cut_angle = 45 →
  (π * (log_diameter / 2)^2 * log_diameter) / 2 = 216 * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l3975_397516


namespace NUMINAMATH_CALUDE_probability_of_selection_for_student_survey_l3975_397566

/-- Represents a simple random sampling without replacement -/
structure SimpleRandomSampling where
  population : ℕ
  sample_size : ℕ
  h_sample_size_le_population : sample_size ≤ population

/-- The probability of a specific item being selected in a simple random sampling without replacement -/
def probability_of_selection (srs : SimpleRandomSampling) : ℚ :=
  srs.sample_size / srs.population

theorem probability_of_selection_for_student_survey :
  let srs : SimpleRandomSampling := {
    population := 303,
    sample_size := 50,
    h_sample_size_le_population := by sorry
  }
  probability_of_selection srs = 50 / 303 := by sorry

end NUMINAMATH_CALUDE_probability_of_selection_for_student_survey_l3975_397566


namespace NUMINAMATH_CALUDE_eliminated_team_size_is_21_l3975_397531

/-- Represents a team in the competition -/
structure Team where
  size : ℕ
  is_girls : Bool

/-- Represents the state of the competition -/
structure Competition where
  teams : List Team
  eliminated_team_size : ℕ

def Competition.remaining_teams (c : Competition) : List Team :=
  c.teams.filter (λ t => t.size ≠ c.eliminated_team_size)

def Competition.total_players (c : Competition) : ℕ :=
  c.teams.map (λ t => t.size) |>.sum

def Competition.remaining_players (c : Competition) : ℕ :=
  c.total_players - c.eliminated_team_size

def Competition.boys_count (c : Competition) : ℕ :=
  c.remaining_teams.filter (λ t => ¬t.is_girls) |>.map (λ t => t.size) |>.sum

def Competition.girls_count (c : Competition) : ℕ :=
  c.remaining_players - c.boys_count

theorem eliminated_team_size_is_21 (c : Competition) : c.eliminated_team_size = 21 :=
  by
  have team_sizes : c.teams.map (λ t => t.size) = [9, 15, 17, 19, 21] := sorry
  have total_five_teams : c.teams.length = 5 := sorry
  have eliminated_is_girls : c.teams.filter (λ t => t.size = c.eliminated_team_size) |>.all (λ t => t.is_girls) := sorry
  have remaining_girls_triple_boys : c.girls_count = 3 * c.boys_count := sorry
  sorry

#check eliminated_team_size_is_21

end NUMINAMATH_CALUDE_eliminated_team_size_is_21_l3975_397531


namespace NUMINAMATH_CALUDE_midpoint_slope_l3975_397533

/-- The slope of the line containing the midpoints of two specific line segments is 1.5 -/
theorem midpoint_slope : 
  let midpoint1 := ((0 + 8) / 2, (0 + 6) / 2)
  let midpoint2 := ((5 + 5) / 2, (0 + 9) / 2)
  let slope := (midpoint2.2 - midpoint1.2) / (midpoint2.1 - midpoint1.1)
  slope = 1.5 := by sorry

end NUMINAMATH_CALUDE_midpoint_slope_l3975_397533


namespace NUMINAMATH_CALUDE_prob_different_subjects_is_one_sixth_l3975_397570

/-- The number of subjects available for selection -/
def num_subjects : ℕ := 4

/-- The number of subjects each student selects -/
def subjects_per_student : ℕ := 2

/-- The total number of possible subject selection combinations for one student -/
def total_combinations : ℕ := (num_subjects.choose subjects_per_student)

/-- The total number of possible events (combinations for both students) -/
def total_events : ℕ := total_combinations * total_combinations

/-- The number of events where both students select different subjects -/
def different_subjects_events : ℕ := total_combinations * ((num_subjects - subjects_per_student).choose subjects_per_student)

/-- The probability that the two students select different subjects -/
def prob_different_subjects : ℚ := different_subjects_events / total_events

theorem prob_different_subjects_is_one_sixth : 
  prob_different_subjects = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_different_subjects_is_one_sixth_l3975_397570


namespace NUMINAMATH_CALUDE_extremum_point_implies_inequality_non_negative_function_implies_m_range_l3975_397571

noncomputable section

variable (m : ℝ)
def f (x : ℝ) : ℝ := Real.exp (x + m) - Real.log x

def a : ℝ := Real.exp (1 / Real.exp 1)

theorem extremum_point_implies_inequality :
  (∃ (m : ℝ), f m 1 = 0 ∧ (∀ (x : ℝ), x > 0 → f m x ≥ f m 1)) →
  ∀ (x : ℝ), x > 0 → Real.exp x - Real.exp 1 * Real.log x ≥ Real.exp 1 :=
sorry

theorem non_negative_function_implies_m_range :
  (∃ (x₀ : ℝ), x₀ > 0 ∧ (∀ (x : ℝ), x > 0 → f m x ≥ f m x₀)) →
  (∀ (x : ℝ), x > 0 → f m x ≥ 0) →
  m ≥ -a - Real.log a :=
sorry

end

end NUMINAMATH_CALUDE_extremum_point_implies_inequality_non_negative_function_implies_m_range_l3975_397571


namespace NUMINAMATH_CALUDE_inverse_proportion_exists_l3975_397555

theorem inverse_proportion_exists (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁ < 0) (h2 : 0 < x₂) (h3 : y₁ > y₂) : 
  ∃ k : ℝ, k < 0 ∧ y₁ = k / x₁ ∧ y₂ = k / x₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_exists_l3975_397555


namespace NUMINAMATH_CALUDE_function_equality_proof_l3975_397551

theorem function_equality_proof (f : ℝ → ℝ) 
  (h₁ : ∀ x, x > 0 → f x > 0)
  (h₂ : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → 
    |x₁ * f x₂ - x₂ * f x₁| = (f x₁ + f x₂) * (x₂ - x₁)) :
  ∃ c : ℝ, c > 0 ∧ ∀ x, x > 0 → f x = c / x :=
sorry

end NUMINAMATH_CALUDE_function_equality_proof_l3975_397551


namespace NUMINAMATH_CALUDE_min_product_of_three_exists_min_product_l3975_397535

def S : Set Int := {-10, -8, -5, -3, 0, 4, 6}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  a * b * c ≥ -240 :=
sorry

theorem exists_min_product :
  ∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = -240 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_exists_min_product_l3975_397535


namespace NUMINAMATH_CALUDE_divisible_by_64_l3975_397520

theorem divisible_by_64 (n : ℕ+) : ∃ k : ℤ, (5 : ℤ)^n.val - 8*n.val^2 + 4*n.val - 1 = 64*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_64_l3975_397520


namespace NUMINAMATH_CALUDE_random_walk_exits_lawn_l3975_397556

/-- A random walk on a 2D plane -/
def RandomWalk2D := ℕ → ℝ × ℝ

/-- The origin (starting point) of the random walk -/
def origin : ℝ × ℝ := (0, 0)

/-- The radius of the circular lawn -/
def lawn_radius : ℝ := 100

/-- The length of each step in the random walk -/
def step_length : ℝ := 1

/-- The expected distance from the origin after n steps in a 2D random walk -/
noncomputable def expected_distance (n : ℕ) : ℝ := Real.sqrt (n : ℝ)

/-- Theorem: For a sufficiently large number of steps, the expected distance 
    from the origin in a 2D random walk exceeds the lawn radius -/
theorem random_walk_exits_lawn :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → expected_distance n > lawn_radius :=
sorry

end NUMINAMATH_CALUDE_random_walk_exits_lawn_l3975_397556


namespace NUMINAMATH_CALUDE_sequence_length_l3975_397543

theorem sequence_length (a₁ : ℕ) (aₙ : ℕ) (d : ℤ) (n : ℕ) :
  a₁ = 150 ∧ aₙ = 30 ∧ d = -6 →
  n = 21 ∧ aₙ = a₁ + d * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_l3975_397543


namespace NUMINAMATH_CALUDE_subtract_inequality_l3975_397518

theorem subtract_inequality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a - 3 < b - 3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_inequality_l3975_397518


namespace NUMINAMATH_CALUDE_square_area_8m_l3975_397537

theorem square_area_8m (side_length : ℝ) (area : ℝ) : 
  side_length = 8 → area = side_length ^ 2 → area = 64 := by sorry

end NUMINAMATH_CALUDE_square_area_8m_l3975_397537


namespace NUMINAMATH_CALUDE_angle_equivalence_l3975_397545

/-- Proves that 2023° is equivalent to -137° in the context of angle measurements -/
theorem angle_equivalence : ∃ (k : ℤ), 2023 = -137 + 360 * k := by sorry

end NUMINAMATH_CALUDE_angle_equivalence_l3975_397545


namespace NUMINAMATH_CALUDE_stick_length_4_forms_triangle_stick_length_1_cannot_form_triangle_stick_length_2_cannot_form_triangle_stick_length_3_cannot_form_triangle_l3975_397582

/-- Triangle inequality check function -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: A stick of length 4 can form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_4_forms_triangle :
  triangle_inequality 3 6 4 :=
sorry

/-- Theorem: A stick of length 1 cannot form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_1_cannot_form_triangle :
  ¬ triangle_inequality 3 6 1 :=
sorry

/-- Theorem: A stick of length 2 cannot form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_2_cannot_form_triangle :
  ¬ triangle_inequality 3 6 2 :=
sorry

/-- Theorem: A stick of length 3 cannot form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_3_cannot_form_triangle :
  ¬ triangle_inequality 3 6 3 :=
sorry

end NUMINAMATH_CALUDE_stick_length_4_forms_triangle_stick_length_1_cannot_form_triangle_stick_length_2_cannot_form_triangle_stick_length_3_cannot_form_triangle_l3975_397582


namespace NUMINAMATH_CALUDE_angle_conversion_l3975_397541

theorem angle_conversion (angle : Real) : ∃ (α k : Real), 
  angle * (π / 180) = α + 2 * k * π ∧ 
  0 ≤ α ∧ α < 2 * π ∧ 
  α = 7 * π / 4 ∧
  k = -10 := by
  sorry

end NUMINAMATH_CALUDE_angle_conversion_l3975_397541


namespace NUMINAMATH_CALUDE_triangle_sum_in_closed_shape_l3975_397581

theorem triangle_sum_in_closed_shape (n : ℕ) (C : ℝ) : 
  n > 0 → C = 3 * 360 - 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_in_closed_shape_l3975_397581


namespace NUMINAMATH_CALUDE_range_of_x_when_m_is_4_range_of_m_l3975_397595

/-- Definition of proposition p -/
def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

/-- Definition of proposition q -/
def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

/-- Theorem for part (1) -/
theorem range_of_x_when_m_is_4 (x : ℝ) :
  (∃ m : ℝ, m > 0 ∧ m = 4 ∧ p x ∧ q x m) → 4 < x ∧ x < 5 := by sorry

/-- Theorem for part (2) -/
theorem range_of_m (m : ℝ) :
  (m > 0 ∧ (∀ x : ℝ, ¬(q x m) → ¬(p x)) ∧ (∃ x : ℝ, ¬(p x) ∧ q x m)) →
  (5/3 ≤ m ∧ m ≤ 2) := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_m_is_4_range_of_m_l3975_397595


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3975_397544

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^19 + 11^13) ∧ ∀ q, Nat.Prime q → q ∣ (3^19 + 11^13) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3975_397544


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3975_397562

theorem quadratic_inequality_range (m : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + 3*x + m > 0) ↔ m ≤ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3975_397562
