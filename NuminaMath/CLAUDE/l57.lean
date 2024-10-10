import Mathlib

namespace exterior_angle_triangle_l57_5796

theorem exterior_angle_triangle (α β γ : ℝ) : 
  0 < α ∧ 0 < β ∧ 0 < γ →  -- angles are positive
  α + β + γ = 180 →  -- sum of angles in a triangle is 180°
  α + β = 148 →  -- exterior angle
  β = 58 →  -- one interior angle
  γ = 90  -- prove that the other interior angle is 90°
  := by sorry

end exterior_angle_triangle_l57_5796


namespace square_or_double_square_l57_5730

theorem square_or_double_square (p m n : ℕ) : 
  Prime p → 
  m ≠ n → 
  p^2 = (m^2 + n^2) / 2 → 
  ∃ k : ℤ, (2*p - m - n : ℤ) = k^2 ∨ (2*p - m - n : ℤ) = 2*k^2 := by
  sorry

end square_or_double_square_l57_5730


namespace tan_double_angle_l57_5747

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end tan_double_angle_l57_5747


namespace quadrilateral_area_is_15_l57_5713

/-- Represents a triangle divided into four smaller triangles and a quadrilateral -/
structure DividedTriangle where
  total_area : ℝ
  triangle1_area : ℝ
  triangle2_area : ℝ
  triangle3_area : ℝ
  triangle4_area : ℝ
  quadrilateral_area : ℝ
  area_sum : total_area = triangle1_area + triangle2_area + triangle3_area + triangle4_area + quadrilateral_area

/-- Theorem stating that if the areas of the four triangles are 5, 10, 10, and 8, 
    then the area of the quadrilateral is 15 -/
theorem quadrilateral_area_is_15 (t : DividedTriangle) 
    (h1 : t.triangle1_area = 5)
    (h2 : t.triangle2_area = 10)
    (h3 : t.triangle3_area = 10)
    (h4 : t.triangle4_area = 8) :
    t.quadrilateral_area = 15 := by
  sorry


end quadrilateral_area_is_15_l57_5713


namespace max_abs_sum_on_ellipse_l57_5701

theorem max_abs_sum_on_ellipse :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ |x| + |y|
  let S : Set (ℝ × ℝ) := {(x, y) | 4 * x^2 + y^2 = 4}
  ∃ (x y : ℝ), (x, y) ∈ S ∧ f (x, y) = (3 * Real.sqrt 2) / Real.sqrt 5 ∧
  ∀ (a b : ℝ), (a, b) ∈ S → f (a, b) ≤ (3 * Real.sqrt 2) / Real.sqrt 5 :=
by sorry

end max_abs_sum_on_ellipse_l57_5701


namespace johann_oranges_l57_5799

theorem johann_oranges (x : ℕ) : 
  (x - 10) / 2 + 5 = 30 → x = 60 := by sorry

end johann_oranges_l57_5799


namespace circles_intersect_l57_5787

/-- Two circles are intersecting if the distance between their centers is less than the sum of their radii
    and greater than the absolute difference of their radii. -/
def circles_intersecting (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : Prop :=
  let distance := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  distance < radius1 + radius2 ∧ distance > |radius1 - radius2|

/-- Given two circles: (x-a)^2+(y-b)^2=4 and (x-a-1)^2+(y-b-2)^2=1 where a, b ∈ ℝ,
    prove that they are intersecting. -/
theorem circles_intersect (a b : ℝ) : 
  circles_intersecting (a, b) (a+1, b+2) 2 1 := by
  sorry


end circles_intersect_l57_5787


namespace abc_equality_l57_5777

theorem abc_equality (a b c x : ℝ) 
  (h : a * x^2 - b * x - c = b * x^2 - c * x - a ∧ 
       b * x^2 - c * x - a = c * x^2 - a * x - b) : 
  a = b ∧ b = c := by
  sorry

end abc_equality_l57_5777


namespace smallest_valid_circular_arrangement_l57_5735

/-- A function that checks if two natural numbers share at least one digit in their decimal representation -/
def shareDigit (a b : ℕ) : Prop := sorry

/-- A function that checks if a list of natural numbers satisfies the neighboring digit condition -/
def validArrangement (lst : List ℕ) : Prop := sorry

/-- The smallest natural number N ≥ 2 for which a valid circular arrangement exists -/
def smallestValidN : ℕ := 29

theorem smallest_valid_circular_arrangement :
  (smallestValidN ≥ 2) ∧
  (∃ (lst : List ℕ), lst.length = smallestValidN ∧ 
    (∀ n, n ∈ lst ↔ 1 ≤ n ∧ n ≤ smallestValidN) ∧
    validArrangement lst) ∧
  (∀ N < smallestValidN, ¬∃ (lst : List ℕ), lst.length = N ∧
    (∀ n, n ∈ lst ↔ 1 ≤ n ∧ n ≤ N) ∧
    validArrangement lst) := by
  sorry

end smallest_valid_circular_arrangement_l57_5735


namespace representations_equivalence_distinct_representations_equivalence_l57_5759

/-- The number of ways to represent a positive integer as a sum of positive integers -/
def numRepresentations (n m : ℕ+) : ℕ :=
  sorry

/-- The number of ways to represent a positive integer as a sum of distinct positive integers -/
def numDistinctRepresentations (n m : ℕ+) : ℕ :=
  sorry

/-- The number of ways to represent a positive integer as a sum of integers from a given set -/
def numRepresentationsFromSet (n : ℕ) (s : Finset ℕ) : ℕ :=
  sorry

theorem representations_equivalence (n m : ℕ+) :
  numRepresentations n m = numRepresentationsFromSet (n - m) (Finset.range m) :=
sorry

theorem distinct_representations_equivalence (n m : ℕ+) :
  numDistinctRepresentations n m = numRepresentationsFromSet (n - m * (m + 1) / 2) (Finset.range n) :=
sorry

end representations_equivalence_distinct_representations_equivalence_l57_5759


namespace bacterium_diameter_nanometers_l57_5771

/-- Conversion factor from meters to nanometers -/
def meters_to_nanometers : ℝ := 10^9

/-- Diameter of the bacterium in meters -/
def bacterium_diameter_meters : ℝ := 0.00000285

/-- Theorem stating the diameter of the bacterium in nanometers -/
theorem bacterium_diameter_nanometers :
  bacterium_diameter_meters * meters_to_nanometers = 2.85 * 10^3 := by
  sorry

#check bacterium_diameter_nanometers

end bacterium_diameter_nanometers_l57_5771


namespace expression_simplification_l57_5717

theorem expression_simplification (x : ℝ) (h : x = 1) :
  (2 * x) / (x + 2) - x / (x - 2) + (4 * x) / (x^2 - 4) = 1 / 3 :=
by sorry

end expression_simplification_l57_5717


namespace weight_difference_l57_5770

/-- Given the weights of four individuals with specific relationships, prove the weight difference between two of them. -/
theorem weight_difference (total_weight : ℝ) (jack_weight : ℝ) (avg_weight : ℝ) : 
  total_weight = 240 ∧ 
  jack_weight = 52 ∧ 
  avg_weight = 60 →
  ∃ (sam_weight lisa_weight daisy_weight : ℝ),
    sam_weight = jack_weight / 0.8 ∧
    lisa_weight = jack_weight * 1.4 ∧
    daisy_weight = (jack_weight + lisa_weight) / 3 ∧
    total_weight = jack_weight + sam_weight + lisa_weight + daisy_weight ∧
    sam_weight - daisy_weight = 23.4 := by
  sorry

end weight_difference_l57_5770


namespace half_obtuse_angle_in_first_quadrant_l57_5739

theorem half_obtuse_angle_in_first_quadrant (α : Real) (h : π / 2 < α ∧ α < π) :
  π / 4 < α / 2 ∧ α / 2 < π / 2 := by
  sorry

end half_obtuse_angle_in_first_quadrant_l57_5739


namespace room_width_calculation_l57_5792

/-- Given a rectangular room with specified length, paving cost per square meter,
    and total paving cost, prove that the width of the room is as calculated. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : cost_per_sqm = 600)
    (h3 : total_cost = 12375) :
    total_cost / (cost_per_sqm * length) = 3.75 := by
  sorry

end room_width_calculation_l57_5792


namespace complement_A_intersect_B_l57_5775

open Set

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem complement_A_intersect_B :
  (𝕌 \ A) ∩ B = Ioc 2 3 :=
sorry

end complement_A_intersect_B_l57_5775


namespace investment_profit_distribution_l57_5741

/-- Represents the investment and profit distribution problem -/
theorem investment_profit_distribution 
  (total_investment : ℕ) 
  (a_extra : ℕ) 
  (b_extra : ℕ) 
  (profit_ratio_a : ℕ) 
  (profit_ratio_b : ℕ) 
  (profit_ratio_c : ℕ) 
  (total_profit : ℕ) 
  (h1 : total_investment = 120000)
  (h2 : a_extra = 6000)
  (h3 : b_extra = 8000)
  (h4 : profit_ratio_a = 4)
  (h5 : profit_ratio_b = 3)
  (h6 : profit_ratio_c = 2)
  (h7 : total_profit = 50000) :
  (profit_ratio_c : ℚ) / (profit_ratio_a + profit_ratio_b + profit_ratio_c : ℚ) * total_profit = 11111.11 := by
  sorry

end investment_profit_distribution_l57_5741


namespace overtake_time_l57_5779

/-- The time it takes for a faster runner to overtake and finish ahead of a slower runner -/
theorem overtake_time (initial_distance steve_speed john_speed final_distance : ℝ) 
  (h1 : initial_distance = 12)
  (h2 : steve_speed = 3.7)
  (h3 : john_speed = 4.2)
  (h4 : final_distance = 2)
  (h5 : john_speed > steve_speed) :
  (initial_distance + final_distance) / (john_speed - steve_speed) = 28 := by
  sorry

#check overtake_time

end overtake_time_l57_5779


namespace division_theorem_l57_5715

theorem division_theorem (b : ℕ) (hb : b ≠ 0) :
  ∀ n : ℕ, ∃! (q r : ℕ), r < b ∧ n = q * b + r :=
sorry

end division_theorem_l57_5715


namespace balls_in_boxes_count_l57_5784

def num_balls : ℕ := 6
def num_boxes : ℕ := 3

theorem balls_in_boxes_count : 
  (num_boxes : ℕ) ^ (num_balls : ℕ) = 729 := by
  sorry

end balls_in_boxes_count_l57_5784


namespace megan_math_problems_l57_5737

/-- Proves that Megan had 36 math problems given the conditions of the problem -/
theorem megan_math_problems :
  ∀ (total_problems math_problems spelling_problems : ℕ)
    (problems_per_hour hours_taken : ℕ),
  spelling_problems = 28 →
  problems_per_hour = 8 →
  hours_taken = 8 →
  total_problems = math_problems + spelling_problems →
  total_problems = problems_per_hour * hours_taken →
  math_problems = 36 := by
  sorry

end megan_math_problems_l57_5737


namespace moses_esther_difference_l57_5749

theorem moses_esther_difference (total : ℝ) (moses_percentage : ℝ) : 
  total = 50 →
  moses_percentage = 0.4 →
  let moses_share := moses_percentage * total
  let remainder := total - moses_share
  let esther_share := remainder / 2
  moses_share - esther_share = 5 := by
  sorry

end moses_esther_difference_l57_5749


namespace half_abs_diff_squares_15_12_l57_5708

theorem half_abs_diff_squares_15_12 : (1/2 : ℝ) * |15^2 - 12^2| = 40.5 := by
  sorry

end half_abs_diff_squares_15_12_l57_5708


namespace tan_sum_specific_angles_l57_5740

theorem tan_sum_specific_angles (α β : ℝ) 
  (h1 : 2 * Real.tan α = 1) 
  (h2 : Real.tan β = -2) : 
  Real.tan (α + β) = -3/4 := by
  sorry

end tan_sum_specific_angles_l57_5740


namespace complex_arithmetic_equality_l57_5791

theorem complex_arithmetic_equality : 
  -1^10 - (13/14 - 11/12) * (4 - (-2)^2) + 1/2 / 3 = -5/6 := by
  sorry

end complex_arithmetic_equality_l57_5791


namespace allocation_schemes_l57_5744

/-- The number of ways to allocate teachers to buses -/
def allocate_teachers (n : ℕ) (m : ℕ) : ℕ :=
  sorry

/-- There are 3 buses -/
def num_buses : ℕ := 3

/-- There are 5 teachers -/
def num_teachers : ℕ := 5

/-- Each bus must have at least one teacher -/
axiom at_least_one_teacher (b : ℕ) : b ≤ num_buses → b > 0

theorem allocation_schemes :
  allocate_teachers num_teachers num_buses = 150 :=
sorry

end allocation_schemes_l57_5744


namespace hockey_season_length_l57_5786

theorem hockey_season_length 
  (games_per_month : ℕ) 
  (total_games : ℕ) 
  (h1 : games_per_month = 13) 
  (h2 : total_games = 182) : 
  total_games / games_per_month = 14 := by
sorry

end hockey_season_length_l57_5786


namespace ball_difference_l57_5726

/-- Problem: Difference between basketballs and soccer balls --/
theorem ball_difference (total : ℕ) (soccer : ℕ) (tennis : ℕ) (baseball : ℕ) (volleyball : ℕ) (basketball : ℕ) : 
  total = 145 →
  soccer = 20 →
  tennis = 2 * soccer →
  baseball = soccer + 10 →
  volleyball = 30 →
  basketball > soccer →
  total = soccer + tennis + baseball + volleyball + basketball →
  basketball - soccer = 5 := by
  sorry

#check ball_difference

end ball_difference_l57_5726


namespace no_solutions_for_absolute_value_equation_l57_5789

theorem no_solutions_for_absolute_value_equation :
  ¬ ∃ (x : ℝ), |x - 3| = x^2 + 2*x + 4 := by
  sorry

end no_solutions_for_absolute_value_equation_l57_5789


namespace franks_mower_blades_expenditure_l57_5767

theorem franks_mower_blades_expenditure 
  (total_earned : ℕ) 
  (games_affordable : ℕ) 
  (game_price : ℕ) 
  (h1 : total_earned = 19)
  (h2 : games_affordable = 4)
  (h3 : game_price = 2) :
  total_earned - games_affordable * game_price = 11 := by
sorry

end franks_mower_blades_expenditure_l57_5767


namespace min_sum_reciprocals_l57_5757

theorem min_sum_reciprocals (w x y z : ℝ) 
  (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : w + x + y + z = 1) :
  1/w + 1/x + 1/y + 1/z ≥ 16 ∧
  (1/w + 1/x + 1/y + 1/z = 16 ↔ w = 1/4 ∧ x = 1/4 ∧ y = 1/4 ∧ z = 1/4) :=
by sorry

end min_sum_reciprocals_l57_5757


namespace transportation_budget_degrees_l57_5797

theorem transportation_budget_degrees (salaries research_and_development utilities equipment supplies : ℝ)
  (h1 : salaries = 60)
  (h2 : research_and_development = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : supplies = 2)
  (h6 : salaries + research_and_development + utilities + equipment + supplies < 100) :
  let transportation := 100 - (salaries + research_and_development + utilities + equipment + supplies)
  (transportation / 100) * 360 = 72 := by
sorry

end transportation_budget_degrees_l57_5797


namespace two_digit_pairs_count_l57_5745

/-- Given two natural numbers x and y, returns true if they contain only two different digits --/
def hasTwoDigits (x y : ℕ) : Prop := sorry

/-- The number of pairs (x, y) where x and y are three-digit numbers, 
    x + y = 999, and x and y together contain only two different digits --/
def countTwoDigitPairs : ℕ := sorry

theorem two_digit_pairs_count : countTwoDigitPairs = 40 := by sorry

end two_digit_pairs_count_l57_5745


namespace current_speed_l57_5785

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 22)
  (h2 : speed_against_current = 12) :
  ∃ (current_speed : ℝ), current_speed = 5 ∧ 
    speed_with_current = speed_against_current + 2 * current_speed :=
by sorry

end current_speed_l57_5785


namespace line_through_point_l57_5756

/-- Given a line equation 3bx + (2b-1)y = 5b - 3 that passes through the point (3, -7),
    prove that b = 1. -/
theorem line_through_point (b : ℝ) : 
  (3 * b * 3 + (2 * b - 1) * (-7) = 5 * b - 3) → b = 1 := by
  sorry

end line_through_point_l57_5756


namespace f_has_one_zero_max_ab_value_l57_5725

noncomputable def f (a b x : ℝ) : ℝ := Real.log (a * x + b) + Real.exp (x - 1)

theorem f_has_one_zero :
  ∃! x, f (-1) 1 x = 0 :=
sorry

theorem max_ab_value (a b : ℝ) (h : a ≠ 0) :
  (∀ x, f a b x ≤ Real.exp (x - 1) + x + 1) →
  a * b ≤ (1 / 2) * Real.exp 3 :=
sorry

end f_has_one_zero_max_ab_value_l57_5725


namespace shifted_linear_function_equation_l57_5769

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Shifts a linear function vertically by a given amount -/
def shiftVertically (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { slope := f.slope, yIntercept := f.yIntercept + shift }

theorem shifted_linear_function_equation 
  (f : LinearFunction) 
  (h1 : f.slope = 2) 
  (h2 : f.yIntercept = -3) :
  (shiftVertically f 3).yIntercept = 0 := by
  sorry

#check shifted_linear_function_equation

end shifted_linear_function_equation_l57_5769


namespace smallest_average_l57_5763

-- Define the set of digits
def digits : Finset Nat := Finset.range 9

-- Define the property of a valid selection
def valid_selection (single_digits double_digits : Finset Nat) : Prop :=
  single_digits.card = 3 ∧
  double_digits.card = 6 ∧
  (single_digits ∪ double_digits) = digits ∧
  single_digits ∩ double_digits = ∅

-- Define the average of the resulting set of numbers
def average (single_digits double_digits : Finset Nat) : ℚ :=
  let single_sum := single_digits.sum id
  let double_sum := (double_digits.filter (· ≤ 3)).sum (· * 10) +
                    (double_digits.filter (· > 3)).sum id
  (single_sum + double_sum : ℚ) / 6

-- Theorem statement
theorem smallest_average :
  ∀ single_digits double_digits : Finset Nat,
    valid_selection single_digits double_digits →
    average single_digits double_digits ≥ 33/2 :=
sorry

end smallest_average_l57_5763


namespace inequality_problem_l57_5700

theorem inequality_problem (r p q : ℝ) 
  (hr : r < 0) 
  (hpq : p * q ≠ 0) 
  (hineq : p^2 * r > q^2 * r) : 
  ¬((-p > -q) ∧ (-p < q) ∧ (1 < -q/p) ∧ (1 > q/p)) :=
sorry

end inequality_problem_l57_5700


namespace solution_satisfies_equations_l57_5773

theorem solution_satisfies_equations :
  ∃ (x y : ℝ), 3 * x - 7 * y = 2 ∧ 4 * y - x = 6 ∧ x = 10 ∧ y = 4 := by
  sorry

end solution_satisfies_equations_l57_5773


namespace sqrt_inequality_l57_5707

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end sqrt_inequality_l57_5707


namespace vector_addition_l57_5738

/-- Given two 2D vectors a and b, prove that their sum is equal to (4, 6) -/
theorem vector_addition (a b : ℝ × ℝ) (h1 : a = (6, 2)) (h2 : b = (-2, 4)) :
  a + b = (4, 6) := by
  sorry

end vector_addition_l57_5738


namespace m_zero_sufficient_not_necessary_l57_5746

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + m = 0

-- Define what it means for the equation to represent a circle
def is_circle (m : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem stating that m = 0 is sufficient but not necessary
theorem m_zero_sufficient_not_necessary :
  (is_circle 0) ∧ (∃ m : ℝ, m ≠ 0 ∧ is_circle m) :=
sorry

end m_zero_sufficient_not_necessary_l57_5746


namespace fence_poles_for_given_plot_l57_5709

/-- Calculates the number of fence poles needed to enclose a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  (perimeter + pole_distance - 1) / pole_distance

/-- Theorem stating the number of fence poles needed for the given plot -/
theorem fence_poles_for_given_plot :
  fence_poles 250 150 7 = 115 := by
  sorry

end fence_poles_for_given_plot_l57_5709


namespace cloth_coloring_problem_l57_5752

/-- Calculates the length of cloth colored by a group of women in a given number of days -/
def clothLength (women : ℕ) (days : ℕ) (rate : ℝ) : ℝ :=
  women * days * rate

theorem cloth_coloring_problem (rate : ℝ) (h1 : rate > 0) :
  clothLength 5 1 rate = 100 →
  clothLength 6 3 rate = 360 := by
  sorry

end cloth_coloring_problem_l57_5752


namespace unfair_coin_expected_value_l57_5722

/-- The expected value of an unfair coin flip -/
theorem unfair_coin_expected_value :
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := 9
  p_heads * gain_heads + p_tails * (-loss_tails) = 1/3 :=
by sorry

end unfair_coin_expected_value_l57_5722


namespace marathon_remainder_yards_l57_5721

/-- The length of a marathon in miles -/
def marathon_miles : ℕ := 28

/-- The additional yards in a marathon beyond the whole miles -/
def marathon_extra_yards : ℕ := 1500

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The number of marathons run -/
def marathons_run : ℕ := 15

/-- The total number of yards run in all marathons -/
def total_yards : ℕ := marathons_run * (marathon_miles * yards_per_mile + marathon_extra_yards)

/-- The remainder of yards after converting total yards to miles -/
def remainder_yards : ℕ := total_yards % yards_per_mile

theorem marathon_remainder_yards : remainder_yards = 1200 := by
  sorry

end marathon_remainder_yards_l57_5721


namespace sin_300_degrees_l57_5794

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l57_5794


namespace phi_bound_l57_5711

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def satisfies_functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f x + 1

def iterate (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

def phi (f : ℝ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  iterate f n x - x

theorem phi_bound (f : ℝ → ℝ) (n : ℕ) :
  is_non_decreasing f →
  satisfies_functional_equation f →
  ∀ x y, |phi f n x - phi f n y| < 1 := by
  sorry

end phi_bound_l57_5711


namespace fraction_evaluation_l57_5754

theorem fraction_evaluation (a b c : ℚ) (ha : a = 7) (hb : b = 11) (hc : c = 19) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end fraction_evaluation_l57_5754


namespace find_vector_c_l57_5702

/-- Given vectors a and b in ℝ², find vector c satisfying the given conditions -/
theorem find_vector_c (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (-3, 2)) : 
  ∃ c : ℝ × ℝ, 
    (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) ∧ 
    (∃ k : ℝ, (c.1 - a.1, c.2 - a.2) = (k * b.1, k * b.2)) → 
    c = (7/3, 7/9) := by
  sorry

end find_vector_c_l57_5702


namespace initial_blue_pens_l57_5731

theorem initial_blue_pens (initial_black : ℕ) (initial_red : ℕ) 
  (blue_removed : ℕ) (black_removed : ℕ) (remaining : ℕ) :
  initial_black = 21 →
  initial_red = 6 →
  blue_removed = 4 →
  black_removed = 7 →
  remaining = 25 →
  ∃ initial_blue : ℕ, 
    initial_blue + initial_black + initial_red = 
    remaining + blue_removed + black_removed ∧
    initial_blue = 9 :=
by sorry

end initial_blue_pens_l57_5731


namespace solution_set_f_range_g_a_gt_2_range_g_a_lt_2_range_g_a_eq_2_l57_5768

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + x

-- Define the function g
def g (a x : ℝ) : ℝ := f x - |a*x - 1| - x

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f (x : ℝ) : 
  f x ≤ 5 ↔ x ∈ Set.Icc (-6) (4/3) :=
sorry

-- Theorem for the range of g(x) when a > 2
theorem range_g_a_gt_2 (a : ℝ) (h : a > 2) :
  Set.range (g a) = Set.Iic (2/a + 1) :=
sorry

-- Theorem for the range of g(x) when 0 < a < 2
theorem range_g_a_lt_2 (a : ℝ) (h1 : a > 0) (h2 : a < 2) :
  Set.range (g a) = Set.Ici (-a/2 - 1) :=
sorry

-- Theorem for the range of g(x) when a = 2
theorem range_g_a_eq_2 :
  Set.range (g 2) = Set.Icc (-2) 2 :=
sorry

end solution_set_f_range_g_a_gt_2_range_g_a_lt_2_range_g_a_eq_2_l57_5768


namespace recreation_spending_comparison_l57_5780

theorem recreation_spending_comparison (wages_last_week : ℝ) : 
  let recreation_last_week := 0.15 * wages_last_week
  let wages_this_week := 0.90 * wages_last_week
  let recreation_this_week := 0.30 * wages_this_week
  recreation_this_week / recreation_last_week = 1.8 := by
sorry

end recreation_spending_comparison_l57_5780


namespace square_sum_product_l57_5766

theorem square_sum_product (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 24) :
  (x^2 + y^2) * (x + y) = 803 := by
  sorry

end square_sum_product_l57_5766


namespace beach_towel_laundry_loads_l57_5798

theorem beach_towel_laundry_loads 
  (num_families : ℕ) 
  (people_per_family : ℕ) 
  (vacation_days : ℕ) 
  (towels_per_person_per_day : ℕ) 
  (towels_per_load : ℕ) 
  (h1 : num_families = 3) 
  (h2 : people_per_family = 4) 
  (h3 : vacation_days = 7) 
  (h4 : towels_per_person_per_day = 1) 
  (h5 : towels_per_load = 14) : 
  (num_families * people_per_family * vacation_days * towels_per_person_per_day + towels_per_load - 1) / towels_per_load = 6 := by
  sorry

end beach_towel_laundry_loads_l57_5798


namespace intersection_point_of_lines_l57_5788

theorem intersection_point_of_lines (x y : ℚ) :
  (5 * x - 3 * y = 20) ∧ (3 * x + 4 * y = 6) ↔ x = 98/29 ∧ y = 87/58 := by
  sorry

end intersection_point_of_lines_l57_5788


namespace alvin_wood_needed_l57_5743

/-- The number of wood pieces Alvin needs for his house -/
def total_wood_needed (friend_pieces brother_pieces more_pieces : ℕ) : ℕ :=
  friend_pieces + brother_pieces + more_pieces

/-- Theorem: Alvin needs 376 pieces of wood in total -/
theorem alvin_wood_needed :
  total_wood_needed 123 136 117 = 376 := by
  sorry

end alvin_wood_needed_l57_5743


namespace units_digit_of_7_451_l57_5753

theorem units_digit_of_7_451 : (7^451) % 10 = 3 := by
  sorry

end units_digit_of_7_451_l57_5753


namespace inequality_and_minimum_value_l57_5748

theorem inequality_and_minimum_value :
  (∃ m n : ℝ, (∀ x : ℝ, |x + 1| + |2*x - 1| ≤ 3 ↔ m ≤ x ∧ x ≤ n) ∧
   m = -1 ∧ n = 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 →
   (1/a + 1/b + 1/c ≥ 9/2 ∧ 
    ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 2 ∧ 1/a₀ + 1/b₀ + 1/c₀ = 9/2)) :=
by sorry

end inequality_and_minimum_value_l57_5748


namespace boxes_theorem_l57_5783

/-- Represents the operation of adding or removing balls from three consecutive boxes. -/
inductive Operation
  | Add
  | Remove

/-- Represents the state of the boxes after operations. -/
def BoxState (n : ℕ) := Fin n → ℕ

/-- Defines the initial state of the boxes. -/
def initial_state (n : ℕ) : BoxState n :=
  fun i => i.val + 1

/-- Applies an operation to three consecutive boxes. -/
def apply_operation (state : BoxState n) (start : Fin n) (op : Operation) : BoxState n :=
  sorry

/-- Checks if all boxes have exactly k balls. -/
def all_equal (state : BoxState n) (k : ℕ) : Prop :=
  ∀ i : Fin n, state i = k

/-- Main theorem: Characterizes when it's possible to achieve k balls in each box. -/
theorem boxes_theorem (n : ℕ) (h : n ≥ 3) :
  ∀ k : ℕ, k > 0 →
    (∃ (final : BoxState n),
      ∃ (ops : List (Fin n × Operation)),
        all_equal final k ∧
        final = (ops.foldl (fun st (i, op) => apply_operation st i op) (initial_state n))) ↔
    ((n % 3 = 1 ∧ k % 3 = 1) ∨ (n % 3 = 2 ∧ k % 3 = 0)) :=
  sorry

end boxes_theorem_l57_5783


namespace average_rate_of_change_x_squared_plus_x_l57_5720

/-- The average rate of change of f(x) = x^2 + x on [1, 2] is 4 -/
theorem average_rate_of_change_x_squared_plus_x : ∀ (f : ℝ → ℝ),
  (∀ x, f x = x^2 + x) →
  (((f 2) - (f 1)) / (2 - 1) = 4) :=
by sorry

end average_rate_of_change_x_squared_plus_x_l57_5720


namespace angle_between_vectors_l57_5751

/-- Given plane vectors a and b, prove that the angle between a and a+b is π/3 -/
theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (1, 0) →
  b = (-1/2, Real.sqrt 3/2) →
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  Real.arccos ((a.1 * a_plus_b.1 + a.2 * a_plus_b.2) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (a_plus_b.1^2 + a_plus_b.2^2))) = π/3 := by
  sorry

end angle_between_vectors_l57_5751


namespace sum_five_probability_l57_5733

theorem sum_five_probability (n : ℕ) : n ≥ 5 →
  (Nat.choose n 2 : ℚ)⁻¹ * 2 = 1 / 14 ↔ n = 8 := by sorry

end sum_five_probability_l57_5733


namespace trig_sum_problem_l57_5718

theorem trig_sum_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α * Real.cos α = -1/2) :
  1 / (1 + Real.sin α) + 1 / (1 + Real.cos α) = 4 := by
  sorry

end trig_sum_problem_l57_5718


namespace same_terminal_side_angle_l57_5727

/-- The angle with the same terminal side as α = π/12 + 2kπ (k ∈ ℤ) is equivalent to 25π/12 radians. -/
theorem same_terminal_side_angle (k : ℤ) : ∃ (n : ℤ), (π/12 + 2*k*π) = 25*π/12 + 2*n*π := by sorry

end same_terminal_side_angle_l57_5727


namespace no_integer_solution_for_sum_of_cubes_l57_5732

theorem no_integer_solution_for_sum_of_cubes (n : ℤ) : 
  n % 9 = 4 → ¬∃ (x y z : ℤ), x^3 + y^3 + z^3 = n := by
  sorry

end no_integer_solution_for_sum_of_cubes_l57_5732


namespace unique_solution_abc_l57_5782

theorem unique_solution_abc (a b c : ℕ+) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a * b + b * c + c * a = a * b * c) : 
  a = 2 ∧ b = 3 ∧ c = 6 := by
sorry

end unique_solution_abc_l57_5782


namespace square_area_increase_l57_5778

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.05 * s
  let original_area := s ^ 2
  let new_area := new_side ^ 2
  (new_area - original_area) / original_area = 0.1025 := by
sorry

end square_area_increase_l57_5778


namespace fraction_order_l57_5762

theorem fraction_order (a b m n : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : m > 0) (h4 : n > 0) :
  b / a < (b + m) / (a + m) ∧ 
  (b + m) / (a + m) < (a + n) / (b + n) ∧ 
  (a + n) / (b + n) < a / b := by
  sorry

end fraction_order_l57_5762


namespace greatest_common_divisor_of_differences_l57_5755

theorem greatest_common_divisor_of_differences (a b c : ℕ) (h : a < b ∧ b < c) :
  ∃ d : ℕ, d > 0 ∧ 
    (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
    (∀ k : ℕ, k > d → ¬(∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s)) →
  (Nat.gcd (b - a) (c - b) = 10) →
  (a = 20 ∧ b = 40 ∧ c = 90) →
  (∃ d : ℕ, d = 10 ∧ d > 0 ∧ 
    (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
    (∀ k : ℕ, k > d → ¬(∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s))) :=
by sorry

end greatest_common_divisor_of_differences_l57_5755


namespace ellipse_distance_to_y_axis_l57_5706

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the foci
def foci (f : ℝ) : Prop := f^2 = 3

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define the perpendicularity condition
def perpendicular_vectors (x y f : ℝ) : Prop :=
  (x + f) * (x - f) + y * y = 0

-- Theorem statement
theorem ellipse_distance_to_y_axis 
  (x y f : ℝ) 
  (h1 : ellipse x y) 
  (h2 : foci f) 
  (h3 : perpendicular_vectors x y f) : 
  x^2 = 8/3 :=
sorry

end ellipse_distance_to_y_axis_l57_5706


namespace tanner_savings_l57_5781

def savings_september : ℕ := 17
def savings_october : ℕ := 48
def savings_november : ℕ := 25
def video_game_cost : ℕ := 49

theorem tanner_savings : 
  savings_september + savings_october + savings_november - video_game_cost = 41 := by
  sorry

end tanner_savings_l57_5781


namespace mary_ate_seven_slices_l57_5765

/-- The number of slices in a large pizza -/
def slices_per_pizza : ℕ := 8

/-- The number of pizzas Mary ordered -/
def pizzas_ordered : ℕ := 2

/-- The number of slices Mary has remaining -/
def slices_remaining : ℕ := 9

/-- The number of slices Mary ate -/
def slices_eaten : ℕ := pizzas_ordered * slices_per_pizza - slices_remaining

theorem mary_ate_seven_slices : slices_eaten = 7 := by
  sorry

end mary_ate_seven_slices_l57_5765


namespace book_price_increase_l57_5705

theorem book_price_increase (original_price : ℝ) (h : original_price > 0) :
  let price_after_first_increase := original_price * 1.15
  let final_price := price_after_first_increase * 1.15
  (final_price - original_price) / original_price = 0.3225 := by
sorry

end book_price_increase_l57_5705


namespace small_rectangle_perimeter_l57_5742

/-- Given a square with perimeter 256 units divided into 16 equal smaller squares,
    each further divided into two rectangles along a diagonal,
    the perimeter of one of these smaller rectangles is 32 + 16√2 units. -/
theorem small_rectangle_perimeter (large_square_perimeter : ℝ) 
  (h1 : large_square_perimeter = 256) 
  (num_divisions : ℕ) 
  (h2 : num_divisions = 16) : ℝ :=
by
  -- Define the perimeter of one small rectangle
  let small_rectangle_perimeter := 32 + 16 * Real.sqrt 2
  
  -- Prove that this is indeed the perimeter
  sorry

#check small_rectangle_perimeter

end small_rectangle_perimeter_l57_5742


namespace vote_percentages_sum_to_100_l57_5758

theorem vote_percentages_sum_to_100 (candidate1_percent candidate2_percent candidate3_percent : ℝ) 
  (h1 : candidate1_percent = 25)
  (h2 : candidate2_percent = 45)
  (h3 : candidate3_percent = 30) :
  candidate1_percent + candidate2_percent + candidate3_percent = 100 := by
  sorry

end vote_percentages_sum_to_100_l57_5758


namespace max_distinct_dance_counts_l57_5703

/-- Represents the dance count for a person -/
def DanceCount := Nat

/-- Represents a set of distinct dance counts -/
def DistinctCounts := Finset DanceCount

theorem max_distinct_dance_counts 
  (num_boys : Nat) 
  (num_girls : Nat) 
  (h_boys : num_boys = 29) 
  (h_girls : num_girls = 15) :
  ∃ (dc : DistinctCounts), dc.card ≤ 29 ∧ 
  ∀ (dc' : DistinctCounts), dc'.card ≤ dc.card :=
sorry

end max_distinct_dance_counts_l57_5703


namespace john_needs_72_strings_l57_5734

/-- The number of strings John needs to restring all instruments -/
def total_strings (num_basses : ℕ) (strings_per_bass : ℕ) (strings_per_guitar : ℕ) (strings_per_8string_guitar : ℕ) : ℕ :=
  let num_guitars := 2 * num_basses
  let num_8string_guitars := num_guitars - 3
  num_basses * strings_per_bass + num_guitars * strings_per_guitar + num_8string_guitars * strings_per_8string_guitar

/-- Theorem stating the total number of strings John needs -/
theorem john_needs_72_strings :
  total_strings 3 4 6 8 = 72 := by
  sorry

end john_needs_72_strings_l57_5734


namespace trader_gain_percentage_l57_5793

theorem trader_gain_percentage (cost : ℝ) (h : cost > 0) :
  let gain := 30 * cost
  let cost_price := 100 * cost
  let gain_percentage := (gain / cost_price) * 100
  gain_percentage = 30 := by
sorry

end trader_gain_percentage_l57_5793


namespace logarithmic_equation_solution_l57_5795

theorem logarithmic_equation_solution :
  ∃ x : ℝ, (Real.log x / Real.log 4) - 3 * (Real.log 8 / Real.log 2) = 1 - (Real.log 2 / Real.log 2) ∧ x = 262144 := by
  sorry

end logarithmic_equation_solution_l57_5795


namespace a_5_equals_13_l57_5714

/-- A sequence defined by a_n = pn + q -/
def a (p q : ℝ) : ℕ+ → ℝ := fun n ↦ p * n.val + q

/-- Given a sequence a_n where a_1 = 5, a_8 = 19, and a_n = pn + q for all n ∈ ℕ+
    (where p and q are constants), prove that a_5 = 13 -/
theorem a_5_equals_13 (p q : ℝ) (h1 : a p q 1 = 5) (h8 : a p q 8 = 19) : a p q 5 = 13 := by
  sorry

end a_5_equals_13_l57_5714


namespace boat_license_combinations_l57_5704

def possible_letters : Nat := 3
def digits_per_license : Nat := 6
def possible_digits : Nat := 10

theorem boat_license_combinations :
  possible_letters * possible_digits ^ digits_per_license = 3000000 := by
  sorry

end boat_license_combinations_l57_5704


namespace largest_prime_for_integer_sqrt_l57_5710

theorem largest_prime_for_integer_sqrt : ∃ (p : ℕ), 
  Prime p ∧ 
  (∃ (q : ℕ), q^2 = 17*p + 625) ∧
  (∀ (p' : ℕ), Prime p' → (∃ (q' : ℕ), q'^2 = 17*p' + 625) → p' ≤ p) ∧
  p = 67 := by
sorry

end largest_prime_for_integer_sqrt_l57_5710


namespace shopkeeper_loss_theorem_l57_5736

/-- Calculates the loss percent for a shopkeeper given profit margin and theft percentage -/
def shopkeeper_loss_percent (profit_margin : ℝ) (theft_percentage : ℝ) : ℝ :=
  let selling_price := 1 + profit_margin
  let remaining_goods := 1 - theft_percentage
  let actual_revenue := selling_price * remaining_goods
  let actual_profit := actual_revenue - remaining_goods
  let net_loss := theft_percentage - actual_profit
  net_loss * 100

/-- Theorem stating that a shopkeeper with 10% profit margin and 20% theft has a 12% loss -/
theorem shopkeeper_loss_theorem :
  shopkeeper_loss_percent 0.1 0.2 = 12 := by
  sorry

end shopkeeper_loss_theorem_l57_5736


namespace min_value_theorem_l57_5716

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  1/x + 8/(1 - 2*x) ≥ 18 := by
  sorry

end min_value_theorem_l57_5716


namespace binary_110011_equals_51_l57_5760

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end binary_110011_equals_51_l57_5760


namespace field_ratio_l57_5728

theorem field_ratio (field_length field_width pond_side : ℝ) : 
  field_length = 16 →
  field_length = field_width * (field_length / field_width) →
  pond_side = 4 →
  pond_side^2 = (1/8) * (field_length * field_width) →
  field_length / field_width = 2 := by
sorry

end field_ratio_l57_5728


namespace sam_drew_age_multiple_l57_5764

/-- Proves that in five years, Sam's age divided by Drew's age equals 3 -/
theorem sam_drew_age_multiple (drew_current_age sam_current_age : ℕ) : 
  drew_current_age = 12 →
  sam_current_age = 46 →
  (sam_current_age + 5) / (drew_current_age + 5) = 3 := by
sorry

end sam_drew_age_multiple_l57_5764


namespace negation_of_universal_proposition_l57_5761

-- Define a structure for parallelograms
structure Parallelogram where
  -- Add necessary fields (for illustration purposes)
  vertices : Fin 4 → ℝ × ℝ

-- Define properties for diagonals
def diagonals_are_equal (p : Parallelogram) : Prop :=
  -- Add definition here
  sorry

def diagonals_bisect_each_other (p : Parallelogram) : Prop :=
  -- Add definition here
  sorry

-- The theorem to prove
theorem negation_of_universal_proposition :
  (¬ ∀ p : Parallelogram, diagonals_are_equal p ∧ diagonals_bisect_each_other p) ↔
  (∃ p : Parallelogram, ¬(diagonals_are_equal p) ∨ ¬(diagonals_bisect_each_other p)) :=
by sorry

end negation_of_universal_proposition_l57_5761


namespace cheryl_m_and_ms_l57_5750

/-- Cheryl's m&m's problem -/
theorem cheryl_m_and_ms 
  (initial : ℕ) 
  (after_dinner : ℕ) 
  (given_to_sister : ℕ) 
  (h1 : initial = 25) 
  (h2 : after_dinner = 5) 
  (h3 : given_to_sister = 13) :
  initial - (after_dinner + given_to_sister) = 7 :=
by sorry

end cheryl_m_and_ms_l57_5750


namespace windfall_percentage_increase_l57_5723

theorem windfall_percentage_increase 
  (initial_balance : ℝ)
  (weekly_investment : ℝ)
  (weeks_in_year : ℕ)
  (final_balance : ℝ)
  (h1 : initial_balance = 250000)
  (h2 : weekly_investment = 2000)
  (h3 : weeks_in_year = 52)
  (h4 : final_balance = 885000) :
  let balance_before_windfall := initial_balance + weekly_investment * weeks_in_year
  let windfall := final_balance - balance_before_windfall
  (windfall / balance_before_windfall) * 100 = 150 := by
  sorry

end windfall_percentage_increase_l57_5723


namespace somu_age_problem_l57_5729

theorem somu_age_problem (s f : ℕ) : 
  s = f / 4 →
  s - 12 = (f - 12) / 7 →
  s = 24 :=
by sorry

end somu_age_problem_l57_5729


namespace freddy_call_cost_l57_5774

/-- Calculates the total cost of phone calls in dollars -/
def total_call_cost (local_duration : ℕ) (international_duration : ℕ) 
                    (local_rate : ℚ) (international_rate : ℚ) : ℚ :=
  (local_duration : ℚ) * local_rate + (international_duration : ℚ) * international_rate

/-- Proves that Freddy's total call cost is $10.00 -/
theorem freddy_call_cost : 
  total_call_cost 45 31 (5 / 100) (25 / 100) = 10 := by
  sorry

#eval total_call_cost 45 31 (5 / 100) (25 / 100)

end freddy_call_cost_l57_5774


namespace complement_of_M_l57_5719

-- Define the set M
def M : Set ℝ := {x : ℝ | x * (x - 3) > 0}

-- State the theorem
theorem complement_of_M : 
  (Set.univ : Set ℝ) \ M = Set.Icc 0 3 := by sorry

end complement_of_M_l57_5719


namespace equal_discriminants_l57_5790

/-- A monic quadratic polynomial with distinct roots -/
structure MonicQuadratic where
  a : ℝ
  b : ℝ
  distinct_roots : a ≠ b

/-- The value of a monic quadratic polynomial at a given point -/
def evaluate (p : MonicQuadratic) (x : ℝ) : ℝ :=
  (x - p.a) * (x - p.b)

/-- The discriminant of a monic quadratic polynomial -/
def discriminant (p : MonicQuadratic) : ℝ :=
  (p.a - p.b)^2

theorem equal_discriminants (P Q : MonicQuadratic)
  (h : evaluate Q P.a + evaluate Q P.b = evaluate P Q.a + evaluate P Q.b) :
  discriminant P = discriminant Q := by
  sorry

end equal_discriminants_l57_5790


namespace quadratic_factorization_l57_5772

theorem quadratic_factorization (a b : ℕ) : 
  (∀ x, x^2 - 20*x + 96 = (x - a)*(x - b)) →
  a > b →
  4*b - a = 20 := by
sorry

end quadratic_factorization_l57_5772


namespace decryption_theorem_l57_5776

/-- Represents an encrypted text --/
def EncryptedText := String

/-- Represents a decrypted message --/
def DecryptedMessage := String

/-- The encryption method used for the word "МОСКВА" --/
def moscowEncryption (s : String) : EncryptedText :=
  sorry

/-- The decryption method for the given encryption --/
def decrypt (s : EncryptedText) : DecryptedMessage :=
  sorry

/-- Checks if two encrypted texts correspond to the same message --/
def sameMessage (t1 t2 : EncryptedText) : Prop :=
  decrypt t1 = decrypt t2

theorem decryption_theorem 
  (text1 text2 text3 : EncryptedText)
  (h1 : moscowEncryption "МОСКВА" = "ЙМЫВОТСЬЛКЪГВЦАЯЯ")
  (h2 : moscowEncryption "МОСКВА" = "УКМАПОЧСРКЩВЗАХ")
  (h3 : moscowEncryption "МОСКВА" = "ШМФЭОГЧСЙЪКФЬВЫЕАКК")
  (h4 : text1 = "ТПЕОИРВНТМОЛАРГЕИАНВИЛЕДНМТААГТДЬТКУБЧКГЕИШНЕИАЯРЯ")
  (h5 : text2 = "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП")
  (h6 : text3 = "РТПАИОМВСВТИЕОБПРОЕННИГЬКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК")
  (h7 : sameMessage text1 text3 ∨ sameMessage text1 text2 ∨ sameMessage text2 text3)
  : decrypt text1 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧ 
    decrypt text3 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧
    decrypt text2 = "СМОТРИВКОРЕНЬ" :=
  sorry

end decryption_theorem_l57_5776


namespace series_sum_l57_5712

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) equals -7/24 -/
theorem series_sum : ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3)) = -7/24 := by
  sorry

end series_sum_l57_5712


namespace log_8_x_equals_3_5_l57_5724

theorem log_8_x_equals_3_5 (x : ℝ) : 
  Real.log x / Real.log 8 = 3.5 → x = 181.04 := by
  sorry

end log_8_x_equals_3_5_l57_5724
