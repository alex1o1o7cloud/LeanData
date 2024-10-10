import Mathlib

namespace problem_statement_l2741_274119

theorem problem_statement (a b : ℝ) 
  (h1 : a / b + b / a = 5 / 2)
  (h2 : a - b = 3 / 2) :
  (a^2 + 2*a*b + b^2 + 2*a^2*b + 2*a*b^2 + a^2*b^2 = 0) ∨ 
  (a^2 + 2*a*b + b^2 + 2*a^2*b + 2*a*b^2 + a^2*b^2 = 81) := by
sorry

end problem_statement_l2741_274119


namespace intersection_y_intercept_sum_l2741_274129

/-- Given two lines that intersect at (3,6), prove that the sum of their y-intercepts is 6 -/
theorem intersection_y_intercept_sum (c d : ℝ) : 
  (3 = (1/3) * 6 + c) →   -- First line passes through (3,6)
  (6 = (1/3) * 3 + d) →   -- Second line passes through (3,6)
  c + d = 6 := by
sorry

end intersection_y_intercept_sum_l2741_274129


namespace parabolic_archway_height_l2741_274113

/-- Represents a parabolic function of the form f(x) = ax² + 20 -/
def parabolic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 20

theorem parabolic_archway_height :
  ∃ a : ℝ, 
    (parabolic_function a 25 = 0) ∧ 
    (parabolic_function a 0 = 20) ∧
    (parabolic_function a 10 = 16.8) := by
  sorry

end parabolic_archway_height_l2741_274113


namespace turkey_weight_ratio_l2741_274146

/-- The weight of the first turkey in kilograms -/
def first_turkey_weight : ℝ := 6

/-- The weight of the second turkey in kilograms -/
def second_turkey_weight : ℝ := 9

/-- The cost of turkey per kilogram in dollars -/
def cost_per_kg : ℝ := 2

/-- The total amount spent on all turkeys in dollars -/
def total_spent : ℝ := 66

/-- The number of turkeys bought -/
def num_turkeys : ℕ := 3

theorem turkey_weight_ratio :
  let total_weight := total_spent / cost_per_kg
  let third_turkey_weight := total_weight - (first_turkey_weight + second_turkey_weight)
  third_turkey_weight / second_turkey_weight = 2 := by
sorry

end turkey_weight_ratio_l2741_274146


namespace prime_square_minus_one_divisibility_l2741_274103

theorem prime_square_minus_one_divisibility (p : ℕ) :
  Prime p → p ≥ 7 →
  (∃ q : ℕ, Prime q ∧ q ≥ 7 ∧ 40 ∣ (q^2 - 1)) ∧
  (∃ r : ℕ, Prime r ∧ r ≥ 7 ∧ ¬(40 ∣ (r^2 - 1))) := by
  sorry

end prime_square_minus_one_divisibility_l2741_274103


namespace fewer_twos_equals_hundred_l2741_274100

theorem fewer_twos_equals_hundred : (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_equals_hundred_l2741_274100


namespace price_change_percentage_l2741_274127

theorem price_change_percentage (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.84 * P → x = 40 := by
  sorry

end price_change_percentage_l2741_274127


namespace enough_beverages_l2741_274102

/-- Robin's hydration plan and beverage inventory --/
structure HydrationPlan where
  water_per_day : ℕ
  juice_per_day : ℕ
  soda_per_day : ℕ
  plan_duration : ℕ
  water_inventory : ℕ
  juice_inventory : ℕ
  soda_inventory : ℕ

/-- Theorem: Robin has enough beverages for her hydration plan --/
theorem enough_beverages (plan : HydrationPlan)
  (h1 : plan.water_per_day = 9)
  (h2 : plan.juice_per_day = 5)
  (h3 : plan.soda_per_day = 3)
  (h4 : plan.plan_duration = 60)
  (h5 : plan.water_inventory = 617)
  (h6 : plan.juice_inventory = 350)
  (h7 : plan.soda_inventory = 215) :
  plan.water_inventory ≥ plan.water_per_day * plan.plan_duration ∧
  plan.juice_inventory ≥ plan.juice_per_day * plan.plan_duration ∧
  plan.soda_inventory ≥ plan.soda_per_day * plan.plan_duration :=
by
  sorry

#check enough_beverages

end enough_beverages_l2741_274102


namespace mens_wages_proof_l2741_274154

-- Define the number of men, women, and boys
def num_men : ℕ := 5
def num_boys : ℕ := 8

-- Define the total earnings
def total_earnings : ℚ := 90

-- Define the relationship between men, women, and boys
axiom men_women_equality : ∃ w : ℕ, num_men = w
axiom women_boys_equality : ∃ w : ℕ, w = num_boys

-- Define the theorem
theorem mens_wages_proof :
  ∃ (wage_man wage_woman wage_boy : ℚ),
    wage_man > 0 ∧ wage_woman > 0 ∧ wage_boy > 0 ∧
    num_men * wage_man + num_boys * wage_boy + num_boys * wage_woman = total_earnings ∧
    num_men * wage_man = 30 :=
sorry

end mens_wages_proof_l2741_274154


namespace ratio_sum_squares_l2741_274185

theorem ratio_sum_squares (a b c : ℝ) : 
  b = 2 * a ∧ c = 3 * a ∧ a^2 + b^2 + c^2 = 2016 → a + b + c = 72 := by
  sorry

end ratio_sum_squares_l2741_274185


namespace total_oreos_count_l2741_274188

/-- The number of Oreos Jordan has -/
def jordan_oreos : ℕ := 11

/-- The number of Oreos James has -/
def james_oreos : ℕ := 2 * jordan_oreos + 3

/-- The total number of Oreos -/
def total_oreos : ℕ := jordan_oreos + james_oreos

theorem total_oreos_count : total_oreos = 36 := by
  sorry

end total_oreos_count_l2741_274188


namespace solution_set_when_m_eq_5_range_of_m_for_f_geq_7_l2741_274145

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| + |x + 6|

-- Theorem for part I
theorem solution_set_when_m_eq_5 :
  {x : ℝ | f x 5 ≤ 12} = {x : ℝ | -13/2 ≤ x ∧ x ≤ 11/2} := by sorry

-- Theorem for part II
theorem range_of_m_for_f_geq_7 :
  {m : ℝ | ∀ x, f x m ≥ 7} = {m : ℝ | m ≤ -13 ∨ 1 ≤ m} := by sorry

end solution_set_when_m_eq_5_range_of_m_for_f_geq_7_l2741_274145


namespace vector_orthogonality_l2741_274109

def a (k : ℝ) : ℝ × ℝ := (k, 3)
def b : ℝ × ℝ := (1, 4)
def c : ℝ × ℝ := (2, 1)

theorem vector_orthogonality (k : ℝ) :
  (2 • a k - 3 • b) • c = 0 → k = 3 := by
  sorry

end vector_orthogonality_l2741_274109


namespace negative_fraction_comparison_l2741_274125

theorem negative_fraction_comparison : -2/3 < -3/5 := by
  sorry

end negative_fraction_comparison_l2741_274125


namespace odot_inequality_implies_a_range_l2741_274182

-- Define the operation ⊙
def odot (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem odot_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, odot (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end odot_inequality_implies_a_range_l2741_274182


namespace sequence_sum_l2741_274147

theorem sequence_sum (A B C D E F G H I J : ℤ) : 
  D = 7 ∧ 
  A + B + C = 24 ∧ 
  B + C + D = 24 ∧ 
  C + D + E = 24 ∧ 
  D + E + F = 24 ∧ 
  E + F + G = 24 ∧ 
  F + G + H = 24 ∧ 
  G + H + I = 24 ∧ 
  H + I + J = 24 → 
  A + J = 105 := by sorry

end sequence_sum_l2741_274147


namespace mike_score_l2741_274166

theorem mike_score (max_score : ℕ) (passing_percentage : ℚ) (shortfall : ℕ) (actual_score : ℕ) : 
  max_score = 780 → 
  passing_percentage = 30 / 100 → 
  shortfall = 22 → 
  actual_score = (max_score * passing_percentage).floor - shortfall → 
  actual_score = 212 := by
sorry

end mike_score_l2741_274166


namespace parabola_focus_parameter_l2741_274108

/-- Given a parabola with equation x^2 = 2py and focus at (0, 2), prove that p = 4 -/
theorem parabola_focus_parameter : ∀ p : ℝ, 
  (∀ x y : ℝ, x^2 = 2*p*y) →  -- parabola equation
  (0, 2) = (0, p/2) →        -- focus coordinates
  p = 4 := by
sorry

end parabola_focus_parameter_l2741_274108


namespace area_divisibility_l2741_274134

/-- A point with integer coordinates -/
structure IntegerPoint where
  x : ℤ
  y : ℤ

/-- A convex polygon with vertices on a circle -/
structure ConvexPolygonOnCircle where
  vertices : List IntegerPoint
  is_convex : sorry
  on_circle : sorry

/-- The statement of the theorem -/
theorem area_divisibility
  (P : ConvexPolygonOnCircle)
  (n : ℕ)
  (n_odd : Odd n)
  (side_length_squared_div : ∃ (side_length : ℕ), (side_length ^ 2) % n = 0) :
  ∃ (area : ℕ), (2 * area) % n = 0 := by
  sorry


end area_divisibility_l2741_274134


namespace equation_2x_minus_1_is_linear_l2741_274112

/-- A linear equation in one variable is of the form ax + b = 0, where a ≠ 0 and x is the variable. --/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x - 1 = 0 --/
def f (x : ℝ) : ℝ := 2 * x - 1

theorem equation_2x_minus_1_is_linear : is_linear_equation_one_var f := by
  sorry


end equation_2x_minus_1_is_linear_l2741_274112


namespace school_ratio_change_l2741_274197

/-- Represents the ratio of boarders to day students -/
structure Ratio where
  boarders : ℕ
  day_students : ℕ

/-- Represents the school's student population -/
structure School where
  boarders : ℕ
  day_students : ℕ

def initial_ratio : Ratio := { boarders := 5, day_students := 12 }

def initial_school : School := { boarders := 150, day_students := 360 }

def new_boarders : ℕ := 30

def final_school : School := { 
  boarders := initial_school.boarders + new_boarders,
  day_students := initial_school.day_students
}

def final_ratio : Ratio := { 
  boarders := 1,
  day_students := 2
}

theorem school_ratio_change :
  (initial_ratio.boarders * initial_school.day_students = initial_ratio.day_students * initial_school.boarders) ∧
  (final_school.boarders = initial_school.boarders + new_boarders) ∧
  (final_school.day_students = initial_school.day_students) →
  (final_ratio.boarders * final_school.day_students = final_ratio.day_students * final_school.boarders) :=
by sorry

end school_ratio_change_l2741_274197


namespace probability_of_white_ball_l2741_274157

/-- The probability of drawing a white ball from a bag of red and white balls -/
theorem probability_of_white_ball (total : ℕ) (red : ℕ) (white : ℕ) :
  total = red + white →
  white > 0 →
  total > 0 →
  (white : ℚ) / (total : ℚ) = 4 / 9 :=
by
  sorry

end probability_of_white_ball_l2741_274157


namespace time_after_2023_hours_l2741_274136

def hours_later (current_time : Nat) (hours_passed : Nat) : Nat :=
  (current_time + hours_passed) % 12

theorem time_after_2023_hours :
  let current_time := 9
  let hours_passed := 2023
  hours_later current_time hours_passed = 8 := by
sorry

end time_after_2023_hours_l2741_274136


namespace simplify_sqrt_sum_l2741_274194

theorem simplify_sqrt_sum : Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2) = 6 := by
  sorry

end simplify_sqrt_sum_l2741_274194


namespace certain_seconds_proof_l2741_274186

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes given in the problem -/
def given_minutes : ℕ := 6

/-- The first ratio number given in the problem -/
def ratio_1 : ℕ := 18

/-- The second ratio number given in the problem -/
def ratio_2 : ℕ := 9

/-- The certain number of seconds we need to find -/
def certain_seconds : ℕ := 720

theorem certain_seconds_proof : 
  (ratio_1 : ℚ) / certain_seconds = ratio_2 / (given_minutes * seconds_per_minute) :=
sorry

end certain_seconds_proof_l2741_274186


namespace equal_integers_from_divisor_properties_l2741_274151

/-- The product of all divisors of a natural number -/
noncomputable def productOfDivisors (n : ℕ) : ℕ := sorry

/-- The number of divisors of a natural number -/
def numberOfDivisors (n : ℕ) : ℕ := sorry

theorem equal_integers_from_divisor_properties (m n s : ℕ) 
  (h_m_ge_n : m ≥ n) 
  (h_s_pos : s > 0) 
  (h_product : productOfDivisors (s * m) = productOfDivisors (s * n))
  (h_number : numberOfDivisors (s * m) = numberOfDivisors (s * n)) : 
  m = n :=
sorry

end equal_integers_from_divisor_properties_l2741_274151


namespace probability_white_or_red_l2741_274143

def total_balls : ℕ := 8 + 9 + 3
def white_balls : ℕ := 8
def black_balls : ℕ := 9
def red_balls : ℕ := 3

theorem probability_white_or_red :
  (white_balls + red_balls : ℚ) / total_balls = 11 / 20 := by
  sorry

end probability_white_or_red_l2741_274143


namespace parabola_vertex_l2741_274163

/-- The parabola defined by y = (x-1)^2 + 2 has its vertex at (1,2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = (x - 1)^2 + 2 → (1, 2) = (x, y) := by sorry

end parabola_vertex_l2741_274163


namespace valid_paths_count_l2741_274190

/-- Represents a point on the grid -/
structure Point where
  x : Nat
  y : Nat

/-- Represents a vertical line segment -/
structure VerticalSegment where
  x : Nat
  y_start : Nat
  y_end : Nat

/-- Definition of the grid and forbidden segments -/
def grid_height : Nat := 5
def grid_width : Nat := 8
def forbidden_segment1 : VerticalSegment := { x := 3, y_start := 1, y_end := 3 }
def forbidden_segment2 : VerticalSegment := { x := 4, y_start := 2, y_end := 5 }

/-- Function to calculate the number of valid paths -/
def count_valid_paths (height width : Nat) (forbidden1 forbidden2 : VerticalSegment) : Nat :=
  sorry

/-- Theorem stating the number of valid paths -/
theorem valid_paths_count :
  count_valid_paths grid_height grid_width forbidden_segment1 forbidden_segment2 = 838 := by
  sorry

end valid_paths_count_l2741_274190


namespace sum_squares_inequality_sum_squares_equality_l2741_274128

theorem sum_squares_inequality (a b c : ℝ) 
  (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) 
  (h_sum_cubes : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 ≤ 4 := by
  sorry

-- Equality case
theorem sum_squares_equality (a b c : ℝ) 
  (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) 
  (h_sum_cubes : a^3 + b^3 + c^3 = 1) :
  (a + b + c + a^2 + b^2 + c^2 = 4) ↔ 
  ((a = 1 ∧ b = 1 ∧ c = -1) ∨ 
   (a = 1 ∧ b = -1 ∧ c = 1) ∨ 
   (a = -1 ∧ b = 1 ∧ c = 1)) := by
  sorry

end sum_squares_inequality_sum_squares_equality_l2741_274128


namespace fourth_rectangle_area_l2741_274180

/-- A rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  /-- Width of the left rectangles -/
  a : ℝ
  /-- Height of the top rectangles -/
  b : ℝ
  /-- Width of the right rectangles -/
  c : ℝ
  /-- Height of the bottom rectangles -/
  d : ℝ
  /-- Area of the top left rectangle is 6 -/
  top_left_area : a * b = 6
  /-- Area of the top right rectangle is 15 -/
  top_right_area : b * c = 15
  /-- Area of the bottom right rectangle is 25 -/
  bottom_right_area : c * d = 25

/-- The area of the fourth (shaded) rectangle in a DividedRectangle is 10 -/
theorem fourth_rectangle_area (r : DividedRectangle) : r.a * r.d = 10 := by
  sorry


end fourth_rectangle_area_l2741_274180


namespace symmetry_wrt_x_axis_l2741_274133

/-- Given a point P(-3, 5), its symmetrical point P' with respect to the x-axis has coordinates (-3, -5). -/
theorem symmetry_wrt_x_axis :
  let P : ℝ × ℝ := (-3, 5)
  let P' : ℝ × ℝ := (-3, -5)
  (P'.1 = P.1) ∧ (P'.2 = -P.2) := by sorry

end symmetry_wrt_x_axis_l2741_274133


namespace no_common_terms_l2741_274155

-- Define the sequences a_n and b_n
def a_n (a : ℝ) (n : ℕ) : ℝ := a * n + 2
def b_n (b : ℝ) (n : ℕ) : ℝ := b * n + 1

-- Theorem statement
theorem no_common_terms (a b : ℝ) (h : a > b) :
  ∀ n : ℕ, a_n a n ≠ b_n b n := by
  sorry

end no_common_terms_l2741_274155


namespace earth_land_area_scientific_notation_l2741_274189

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a given number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The land area of the Earth in km² -/
def earthLandArea : ℝ := 149000000

/-- The number of significant figures to retain -/
def sigFiguresRequired : ℕ := 3

theorem earth_land_area_scientific_notation :
  toScientificNotation earthLandArea sigFiguresRequired =
    ScientificNotation.mk 1.49 8 (by norm_num) :=
  sorry

end earth_land_area_scientific_notation_l2741_274189


namespace game_C_more_likely_than_D_l2741_274117

/-- Probability of getting heads when tossing the biased coin -/
def p_heads : ℚ := 3/4

/-- Probability of getting tails when tossing the biased coin -/
def p_tails : ℚ := 1/4

/-- Probability of winning Game C -/
def p_win_C : ℚ := p_heads^4

/-- Probability of winning Game D -/
def p_win_D : ℚ := p_heads^5 + p_heads^3 * p_tails^2 + p_tails^3 * p_heads^2 + p_tails^5

theorem game_C_more_likely_than_D : p_win_C - p_win_D = 11/256 := by
  sorry

end game_C_more_likely_than_D_l2741_274117


namespace phoenix_airport_on_time_rate_l2741_274171

theorem phoenix_airport_on_time_rate (late_flights : ℕ) (on_time_flights : ℕ) (additional_on_time_flights : ℕ) :
  late_flights = 1 →
  on_time_flights = 3 →
  additional_on_time_flights = 2 →
  (on_time_flights + additional_on_time_flights : ℚ) / (late_flights + on_time_flights + additional_on_time_flights) > 83.33 / 100 := by
sorry

end phoenix_airport_on_time_rate_l2741_274171


namespace choir_third_group_members_l2741_274121

theorem choir_third_group_members (total_members : ℕ) (group1_members : ℕ) (group2_members : ℕ) 
  (h1 : total_members = 70)
  (h2 : group1_members = 25)
  (h3 : group2_members = 30) :
  total_members - (group1_members + group2_members) = 15 := by
sorry

end choir_third_group_members_l2741_274121


namespace inequality_proof_l2741_274199

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0) 
  (h5 : a + b + c + d = 1) : 
  (a + 2*b + 3*c + 4*d) * (a^a * b^b * c^c * d^d) < 1 := by
  sorry

end inequality_proof_l2741_274199


namespace fuel_cost_per_liter_l2741_274170

-- Define constants
def service_cost : ℝ := 2.10
def mini_vans : ℕ := 3
def trucks : ℕ := 2
def total_cost : ℝ := 299.1
def mini_van_tank : ℝ := 65
def truck_tank_multiplier : ℝ := 2.2  -- 120% bigger means 2.2 times the size

-- Define functions
def total_service_cost : ℝ := service_cost * (mini_vans + trucks)
def truck_tank : ℝ := mini_van_tank * truck_tank_multiplier
def total_fuel_volume : ℝ := mini_vans * mini_van_tank + trucks * truck_tank
def fuel_cost : ℝ := total_cost - total_service_cost

-- Theorem to prove
theorem fuel_cost_per_liter : fuel_cost / total_fuel_volume = 0.60 := by
  sorry

end fuel_cost_per_liter_l2741_274170


namespace jerry_showers_l2741_274165

/-- Represents the water usage scenario for Jerry in July --/
structure WaterUsage where
  totalAllowance : ℕ
  drinkingCooking : ℕ
  showerUsage : ℕ
  poolLength : ℕ
  poolWidth : ℕ
  poolHeight : ℕ
  gallonToCubicFoot : ℕ

/-- Calculates the number of showers Jerry can take in July --/
def calculateShowers (w : WaterUsage) : ℕ :=
  let poolVolume := w.poolLength * w.poolWidth * w.poolHeight
  let remainingWater := w.totalAllowance - w.drinkingCooking - poolVolume
  remainingWater / w.showerUsage

/-- Theorem stating that Jerry can take 15 showers in July --/
theorem jerry_showers (w : WaterUsage) 
  (h1 : w.totalAllowance = 1000)
  (h2 : w.drinkingCooking = 100)
  (h3 : w.showerUsage = 20)
  (h4 : w.poolLength = 10)
  (h5 : w.poolWidth = 10)
  (h6 : w.poolHeight = 6)
  (h7 : w.gallonToCubicFoot = 1) :
  calculateShowers w = 15 := by
  sorry

end jerry_showers_l2741_274165


namespace farm_cows_l2741_274131

theorem farm_cows (milk_per_6_cows : ℝ) (total_milk : ℝ) (weeks : ℕ) :
  milk_per_6_cows = 108 →
  total_milk = 2160 →
  weeks = 5 →
  (total_milk / (milk_per_6_cows / 6) / weeks : ℝ) = 24 :=
by sorry

end farm_cows_l2741_274131


namespace greatest_angle_in_triangle_l2741_274116

theorem greatest_angle_in_triangle (a b c : ℝ) (h : (b / (c - a)) - (a / (b + c)) = 1) :
  ∃ (A B C : ℝ), 
    A + B + C = 180 ∧ 
    0 < A ∧ 0 < B ∧ 0 < C ∧
    a^2 = b^2 + c^2 - 2*b*c*Real.cos A ∧
    b^2 = a^2 + c^2 - 2*a*c*Real.cos B ∧
    c^2 = a^2 + b^2 - 2*a*b*Real.cos C ∧
    max A (max B C) = 120 := by
  sorry

end greatest_angle_in_triangle_l2741_274116


namespace triangle_side_length_l2741_274196

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →  -- Convert 60° to radians
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a = Real.sqrt 13 := by sorry

end triangle_side_length_l2741_274196


namespace total_sugar_calculation_l2741_274142

def chocolate_bars : ℕ := 14
def sugar_per_bar : ℕ := 10
def lollipop_sugar : ℕ := 37

theorem total_sugar_calculation :
  chocolate_bars * sugar_per_bar + lollipop_sugar = 177 := by
  sorry

end total_sugar_calculation_l2741_274142


namespace ratio_sum_problem_l2741_274191

theorem ratio_sum_problem (x y z b : ℚ) : 
  x / y = 4 / 5 →
  y / z = 5 / 6 →
  y = 15 * b - 5 →
  x + y + z = 90 →
  b = 7 / 3 := by
sorry

end ratio_sum_problem_l2741_274191


namespace second_derivative_y_l2741_274181

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.log (1 + Real.sin x)

theorem second_derivative_y (x : ℝ) :
  (deriv (deriv y)) x = 2 * Real.log (1 + Real.sin x) + (4 * x * Real.cos x - x^2) / (1 + Real.sin x) :=
by sorry

end second_derivative_y_l2741_274181


namespace pet_store_profit_l2741_274130

/-- The profit calculation for a pet store reselling geckos --/
theorem pet_store_profit (brandon_price : ℕ) (pet_store_markup : ℕ → ℕ) : 
  brandon_price = 100 → 
  (∀ x, pet_store_markup x = 3 * x + 5) →
  pet_store_markup brandon_price - brandon_price = 205 := by
  sorry

end pet_store_profit_l2741_274130


namespace triangle_determinant_zero_l2741_274178

theorem triangle_determinant_zero (A B C : ℝ) (h : A + B + C = π) :
  let f := fun x => (Real.cos x)^2
  let g := Real.tan
  Matrix.det !![f A, g A, 1; f B, g B, 1; f C, g C, 1] = 0 := by
  sorry

end triangle_determinant_zero_l2741_274178


namespace min_value_expression_min_value_expression_tight_l2741_274169

theorem min_value_expression (a b c : ℝ) (hb : b > 0) (hc : c > 0) (hbc : b > c) (ha : a ≠ 0) :
  ((2*a + b)^2 + (b - c)^2 + (c - 2*a)^2) / b^2 ≥ 4/3 :=
sorry

theorem min_value_expression_tight (a b c : ℝ) (hb : b > 0) (hc : c > 0) (hbc : b > c) (ha : a ≠ 0) :
  ∃ (ε : ℝ), ε > 0 ∧ ((2*a + b)^2 + (b - c)^2 + (c - 2*a)^2) / b^2 < 4/3 + ε :=
sorry

end min_value_expression_min_value_expression_tight_l2741_274169


namespace inequality_proof_l2741_274159

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 0 < a / b ∧ a / b < 1 := by
  sorry

end inequality_proof_l2741_274159


namespace fish_total_weight_l2741_274176

/-- The weight of a fish given specific conditions about its parts -/
def fish_weight (tail_weight head_weight body_weight : ℝ) : Prop :=
  tail_weight = 4 ∧
  head_weight = tail_weight + (body_weight / 2) ∧
  body_weight = head_weight + tail_weight

theorem fish_total_weight :
  ∀ (tail_weight head_weight body_weight : ℝ),
    fish_weight tail_weight head_weight body_weight →
    tail_weight + head_weight + body_weight = 32 :=
by sorry

end fish_total_weight_l2741_274176


namespace beach_probability_l2741_274175

/-- Given a beach scenario where:
  * 75 people are wearing sunglasses
  * 60 people are wearing hats
  * The probability of wearing sunglasses given wearing a hat is 1/3
  This theorem proves that the probability of wearing a hat given wearing sunglasses is 4/15. -/
theorem beach_probability (total_sunglasses : ℕ) (total_hats : ℕ) 
  (prob_sunglasses_given_hat : ℚ) :
  total_sunglasses = 75 →
  total_hats = 60 →
  prob_sunglasses_given_hat = 1/3 →
  (total_hats * prob_sunglasses_given_hat : ℚ) / total_sunglasses = 4/15 :=
by sorry

end beach_probability_l2741_274175


namespace min_circle_property_l2741_274104

/-- Definition of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + y = -1

/-- Definition of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- Definition of the circle with minimal area -/
def minCircle (x y : ℝ) : Prop := x^2 + y^2 + (6/5)*x + (3/5)*y + 1 = 0

/-- Theorem stating that the minCircle passes through the intersection points of circle1 and circle2 and has the minimum area -/
theorem min_circle_property :
  ∀ (x y : ℝ), 
    (circle1 x y ∧ circle2 x y → minCircle x y) ∧
    (∀ (a b c : ℝ), (∀ (u v : ℝ), circle1 u v ∧ circle2 u v → x^2 + y^2 + 2*a*x + 2*b*y + c = 0) →
      (x^2 + y^2 + (6/5)*x + (3/5)*y + 1)^2 ≤ (x^2 + y^2 + 2*a*x + 2*b*y + c)^2) :=
by sorry

end min_circle_property_l2741_274104


namespace virginia_adrienne_teaching_difference_l2741_274120

theorem virginia_adrienne_teaching_difference :
  ∀ (V A D : ℕ),
  V + A + D = 102 →
  D = 43 →
  V = D - 9 →
  V > A →
  V - A = 9 :=
by
  sorry

end virginia_adrienne_teaching_difference_l2741_274120


namespace trig_identity_l2741_274132

theorem trig_identity (x : Real) (h : Real.sin x - 2 * Real.cos x = 0) :
  2 * Real.sin x ^ 2 + Real.cos x ^ 2 + 1 = 14 / 5 := by
  sorry

end trig_identity_l2741_274132


namespace probability_at_least_one_diamond_or_joker_l2741_274160

theorem probability_at_least_one_diamond_or_joker :
  let total_cards : ℕ := 60
  let diamond_cards : ℕ := 15
  let joker_cards : ℕ := 6
  let favorable_cards : ℕ := diamond_cards + joker_cards
  let prob_not_favorable : ℚ := (total_cards - favorable_cards) / total_cards
  let prob_neither_favorable : ℚ := prob_not_favorable * prob_not_favorable
  prob_neither_favorable = 169 / 400 →
  1 - prob_neither_favorable = 231 / 400 :=
by
  sorry

end probability_at_least_one_diamond_or_joker_l2741_274160


namespace girls_in_college_l2741_274139

theorem girls_in_college (total : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) (girls : ℕ) :
  total = 440 →
  ratio_boys = 6 →
  ratio_girls = 5 →
  ratio_boys * girls = ratio_girls * (total - girls) →
  girls = 200 := by
  sorry

end girls_in_college_l2741_274139


namespace gcd_540_180_minus_2_l2741_274107

theorem gcd_540_180_minus_2 : Int.gcd 540 180 - 2 = 178 := by
  sorry

end gcd_540_180_minus_2_l2741_274107


namespace triangles_in_3x4_grid_l2741_274138

/-- Represents a rectangular grid with diagonals --/
structure RectangularGridWithDiagonals where
  rows : Nat
  columns : Nat

/-- Calculates the number of triangles in a rectangular grid with diagonals --/
def count_triangles (grid : RectangularGridWithDiagonals) : Nat :=
  let basic_triangles := 2 * grid.rows * grid.columns
  let row_triangles := grid.rows * (grid.columns - 1) * grid.columns / 2
  let diagonal_triangles := 2
  basic_triangles + row_triangles + diagonal_triangles

/-- Theorem: The number of triangles in a 3x4 grid with diagonals is 44 --/
theorem triangles_in_3x4_grid :
  count_triangles ⟨3, 4⟩ = 44 := by
  sorry

#eval count_triangles ⟨3, 4⟩

end triangles_in_3x4_grid_l2741_274138


namespace parallelogram_centers_coincide_l2741_274105

-- Define a parallelogram
structure Parallelogram (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

-- Define a point on a line segment
def PointOnSegment (V : Type*) [AddCommGroup V] [Module ℝ V] (A B P : V) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

-- Define the center of a parallelogram
def CenterOfParallelogram (V : Type*) [AddCommGroup V] [Module ℝ V] (p : Parallelogram V) : V :=
  (1/2) • (p.A + p.C)

-- State the theorem
theorem parallelogram_centers_coincide
  (V : Type*) [AddCommGroup V] [Module ℝ V]
  (p₁ p₂ : Parallelogram V)
  (h₁ : PointOnSegment V p₁.A p₁.B p₂.A)
  (h₂ : PointOnSegment V p₁.B p₁.C p₂.B)
  (h₃ : PointOnSegment V p₁.C p₁.D p₂.C)
  (h₄ : PointOnSegment V p₁.D p₁.A p₂.D) :
  CenterOfParallelogram V p₁ = CenterOfParallelogram V p₂ :=
sorry

end parallelogram_centers_coincide_l2741_274105


namespace exam_score_difference_l2741_274124

theorem exam_score_difference (score_65 score_75 score_85 score_95 : ℝ)
  (percent_65 percent_75 percent_85 percent_95 : ℝ)
  (h1 : score_65 = 65)
  (h2 : score_75 = 75)
  (h3 : score_85 = 85)
  (h4 : score_95 = 95)
  (h5 : percent_65 = 0.15)
  (h6 : percent_75 = 0.40)
  (h7 : percent_85 = 0.20)
  (h8 : percent_95 = 0.25)
  (h9 : percent_65 + percent_75 + percent_85 + percent_95 = 1) :
  let mean := score_65 * percent_65 + score_75 * percent_75 + score_85 * percent_85 + score_95 * percent_95
  let median := score_75
  mean - median = 5.5 := by
  sorry

end exam_score_difference_l2741_274124


namespace total_players_is_51_l2741_274149

/-- The number of cricket players -/
def cricket_players : ℕ := 10

/-- The number of hockey players -/
def hockey_players : ℕ := 12

/-- The number of football players -/
def football_players : ℕ := 16

/-- The number of softball players -/
def softball_players : ℕ := 13

/-- Theorem stating that the total number of players is 51 -/
theorem total_players_is_51 :
  cricket_players + hockey_players + football_players + softball_players = 51 := by
  sorry

end total_players_is_51_l2741_274149


namespace solve_for_m_l2741_274153

-- Define the equation
def is_quadratic (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, (m + 1) * x^(|m| + 1) + 6 * m * x - 2 = a * x^2 + b * x + c

-- Theorem statement
theorem solve_for_m :
  ∀ m : ℝ, is_quadratic m ∧ m + 1 ≠ 0 → m = 1 := by
  sorry

end solve_for_m_l2741_274153


namespace range_of_3a_plus_2b_l2741_274141

theorem range_of_3a_plus_2b (a b : ℝ) (h : a^2 + b^2 = 4) :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 * Real.sqrt 13) (2 * Real.sqrt 13) ∧ 
  x = 3*a + 2*b ∧ 
  ∀ (y : ℝ), y = 3*a + 2*b → y ∈ Set.Icc (-2 * Real.sqrt 13) (2 * Real.sqrt 13) :=
by sorry

end range_of_3a_plus_2b_l2741_274141


namespace raspberry_pies_l2741_274152

theorem raspberry_pies (total_pies : ℝ) (peach_ratio strawberry_ratio raspberry_ratio : ℝ) :
  total_pies = 36 ∧
  peach_ratio = 2 ∧
  strawberry_ratio = 5 ∧
  raspberry_ratio = 3 →
  (raspberry_ratio / (peach_ratio + strawberry_ratio + raspberry_ratio)) * total_pies = 10.8 :=
by sorry

end raspberry_pies_l2741_274152


namespace candy_problem_solution_l2741_274172

def candy_problem (packets : ℕ) (candies_per_packet : ℕ) (weekdays : ℕ) (weekend_days : ℕ) 
  (weekday_consumption : ℕ) (weekend_consumption : ℕ) : ℕ := 
  (packets * candies_per_packet) / 
  (weekdays * weekday_consumption + weekend_days * weekend_consumption)

theorem candy_problem_solution : 
  candy_problem 2 18 5 2 2 1 = 3 := by
  sorry

end candy_problem_solution_l2741_274172


namespace average_weight_a_b_l2741_274161

-- Define the weights of a, b, and c
variable (a b c : ℝ)

-- Define the conditions
variable (h1 : (a + b + c) / 3 = 45)
variable (h2 : (b + c) / 2 = 45)
variable (h3 : b = 35)

-- Theorem statement
theorem average_weight_a_b : (a + b) / 2 = 40 := by
  sorry

end average_weight_a_b_l2741_274161


namespace cube_sum_inequality_l2741_274137

theorem cube_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x + y)) + (y^3 / (y + z)) + (z^3 / (z + x)) ≥ (x*y + y*z + z*x) / 2 := by
  sorry

end cube_sum_inequality_l2741_274137


namespace probability_of_white_ball_l2741_274118

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.3 → p_black = 0.5 → p_red + p_black + p_white = 1 → p_white = 0.2 := by
  sorry

end probability_of_white_ball_l2741_274118


namespace bookstore_inventory_l2741_274156

/-- The number of books acquired by the bookstore. -/
def total_books : ℕ := 1000

/-- The number of books sold on the first day. -/
def first_day_sales : ℕ := total_books / 2

/-- The number of books sold on the second day. -/
def second_day_sales : ℕ := first_day_sales / 2 + first_day_sales + 50

/-- The number of books remaining after both days of sales. -/
def remaining_books : ℕ := 200

/-- Theorem stating that the total number of books is 1000, given the sales conditions. -/
theorem bookstore_inventory :
  total_books = 1000 ∧
  first_day_sales = total_books / 2 ∧
  second_day_sales = first_day_sales / 2 + first_day_sales + 50 ∧
  remaining_books = 200 ∧
  total_books = first_day_sales + second_day_sales + remaining_books :=
by sorry

end bookstore_inventory_l2741_274156


namespace base3_11111_is_121_l2741_274162

def base3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem base3_11111_is_121 :
  base3_to_decimal [1, 1, 1, 1, 1] = 121 := by
  sorry

end base3_11111_is_121_l2741_274162


namespace cos_x_plus_2y_equals_one_l2741_274198

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (a : ℝ) 
  (hx : x ∈ Set.Icc (-π/4) (π/4)) 
  (hy : y ∈ Set.Icc (-π/4) (π/4)) 
  (eq1 : x^3 + Real.sin x - 2*a = 0)
  (eq2 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
sorry

end cos_x_plus_2y_equals_one_l2741_274198


namespace problem_statement_l2741_274140

theorem problem_statement (x y : ℝ) : 
  y = (3/4) * x →
  x^y = y^x →
  x + y = 448/81 :=
by sorry

end problem_statement_l2741_274140


namespace intersection_of_A_and_B_l2741_274193

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

-- Statement to prove
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_B_l2741_274193


namespace find_divisor_l2741_274150

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 52 →
  quotient = 16 →
  remainder = 4 →
  dividend = divisor * quotient + remainder →
  divisor = 3 := by
sorry

end find_divisor_l2741_274150


namespace volume_ratio_is_correct_l2741_274101

/-- The ratio of the volume of a cube with edge length 9 inches to the volume of a cube with edge length 2 feet -/
def volume_ratio : ℚ :=
  let inch_per_foot : ℚ := 12
  let edge1 : ℚ := 9  -- 9 inches
  let edge2 : ℚ := 2 * inch_per_foot  -- 2 feet in inches
  (edge1 / edge2) ^ 3

theorem volume_ratio_is_correct : volume_ratio = 27 / 512 := by
  sorry

end volume_ratio_is_correct_l2741_274101


namespace simplify_fraction_l2741_274158

theorem simplify_fraction (a : ℝ) (h : a ≠ 1) :
  ((a + 1) / (a - 1) + 1) / (2 * a / (a^2 - 1)) = a + 1 := by
  sorry

end simplify_fraction_l2741_274158


namespace roots_sum_expression_l2741_274192

def quadratic_equation (x : ℝ) : Prop := 5 * x^2 - 3 * x - 4 = 0

theorem roots_sum_expression (x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation x₁) 
  (h₂ : quadratic_equation x₂) 
  (h₃ : x₁ ≠ x₂) : 
  2 * x₁^2 + 3 * x₂^2 = 178 / 25 := by
sorry

end roots_sum_expression_l2741_274192


namespace pencil_length_l2741_274111

theorem pencil_length : ∀ (L : ℝ), 
  (L > 0) →                          -- Ensure positive length
  ((1/8) * L + (1/2) * (7/8) * L + 7/2 = L) →  -- Parts sum to total
  (L = 8) :=                         -- Total length is 8
by
  sorry

end pencil_length_l2741_274111


namespace jakes_comic_books_l2741_274122

/-- Jake's comic book problem -/
theorem jakes_comic_books (jake_books : ℕ) (total_books : ℕ) (brother_books : ℕ) : 
  jake_books = 36 →
  total_books = 87 →
  brother_books > jake_books →
  total_books = jake_books + brother_books →
  brother_books - jake_books = 15 := by
sorry

end jakes_comic_books_l2741_274122


namespace two_questions_determine_number_l2741_274183

theorem two_questions_determine_number : 
  ∃ (q₁ q₂ : ℕ → ℕ → ℕ), 
    (∀ m : ℕ, m ≥ 2 → q₁ m ≥ 2) ∧ 
    (∀ m : ℕ, m ≥ 2 → q₂ m ≥ 2) ∧ 
    (∀ V : ℕ, 1 ≤ V ∧ V ≤ 100 → 
      ∀ V' : ℕ, 1 ≤ V' ∧ V' ≤ 100 → 
        (V / q₁ V = V' / q₁ V' ∧ V / q₂ V = V' / q₂ V') → V = V') :=
sorry

end two_questions_determine_number_l2741_274183


namespace infinite_geometric_sequence_formula_l2741_274173

/-- An infinite geometric sequence satisfying given conditions -/
def InfiniteGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∑' n, a n) = 3 ∧ (∑' n, (a n)^2) = 9/2

/-- The general formula for the sequence -/
def GeneralFormula (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = 2 * (1/3)^(n-1)

/-- Theorem stating that the general formula holds for the given infinite geometric sequence -/
theorem infinite_geometric_sequence_formula 
    (a : ℕ → ℝ) (h : InfiniteGeometricSequence a) : GeneralFormula a := by
  sorry

end infinite_geometric_sequence_formula_l2741_274173


namespace exterior_angles_hexagon_pentagon_l2741_274177

/-- The sum of exterior angles of a polygon -/
def sum_exterior_angles (n : ℕ) : ℝ := 360

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- A pentagon has 5 sides -/
def pentagon_sides : ℕ := 5

theorem exterior_angles_hexagon_pentagon : 
  sum_exterior_angles hexagon_sides = sum_exterior_angles pentagon_sides := by
  sorry

end exterior_angles_hexagon_pentagon_l2741_274177


namespace pencil_case_problem_l2741_274114

theorem pencil_case_problem (total : ℕ) (difference : ℕ) (erasers : ℕ) : 
  total = 240 →
  difference = 2 →
  erasers + (erasers - difference) = total →
  erasers = 121 := by
  sorry

end pencil_case_problem_l2741_274114


namespace stimulus_check_amount_l2741_274135

theorem stimulus_check_amount : ∃ T : ℚ, 
  (27 / 125 : ℚ) * T = 432 ∧ T = 2000 := by
  sorry

end stimulus_check_amount_l2741_274135


namespace probability_of_shaded_triangle_l2741_274126

/-- A triangle in the diagram -/
structure Triangle where
  label : String

/-- The set of all triangles in the diagram -/
def all_triangles : Finset Triangle := sorry

/-- The set of shaded triangles -/
def shaded_triangles : Finset Triangle := sorry

/-- Each triangle has the same probability of being selected -/
axiom equal_probability : ∀ t : Triangle, t ∈ all_triangles → 
  (Finset.card {t} : ℚ) / (Finset.card all_triangles : ℚ) = 1 / (Finset.card all_triangles : ℚ)

theorem probability_of_shaded_triangle :
  (Finset.card shaded_triangles : ℚ) / (Finset.card all_triangles : ℚ) = 1 / 5 := by
  sorry

end probability_of_shaded_triangle_l2741_274126


namespace polynomial_coefficient_bound_l2741_274184

def M : Set (ℝ → ℝ) :=
  {P | ∃ a b c d : ℝ, ∀ x, P x = a * x^3 + b * x^2 + c * x + d ∧ ∀ x ∈ Set.Icc (-1) 1, |P x| ≤ 1}

theorem polynomial_coefficient_bound :
  ∃ k : ℝ, k = 4 ∧ (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| ≤ k) ∧
  ∀ k' : ℝ, (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| ≤ k') → k' ≥ k :=
by sorry

end polynomial_coefficient_bound_l2741_274184


namespace correct_grass_bundle_equations_l2741_274144

/-- Represents the number of roots in grass bundles -/
structure GrassBundles where
  high_quality : ℕ  -- number of roots in one bundle of high-quality grass
  low_quality : ℕ   -- number of roots in one bundle of low-quality grass

/-- Represents the relationships between high-quality and low-quality grass bundles -/
def grass_bundle_relations (g : GrassBundles) : Prop :=
  (5 * g.high_quality - 11 = 7 * g.low_quality) ∧
  (7 * g.high_quality - 25 = 5 * g.low_quality)

/-- Theorem stating that the given system of equations correctly represents the problem -/
theorem correct_grass_bundle_equations (g : GrassBundles) :
  grass_bundle_relations g ↔
  (5 * g.high_quality - 11 = 7 * g.low_quality) ∧
  (7 * g.high_quality - 25 = 5 * g.low_quality) :=
by sorry

end correct_grass_bundle_equations_l2741_274144


namespace triangle_configuration_theorem_l2741_274167

/-- A configuration of wire triangles in space. -/
structure TriangleConfiguration where
  /-- The number of wire triangles. -/
  k : ℕ
  /-- The number of triangles converging at each vertex. -/
  p : ℕ
  /-- Each pair of triangles has exactly one common vertex. -/
  one_common_vertex : True
  /-- At each vertex, the same number p of triangles converge. -/
  p_triangles_at_vertex : True

/-- The theorem stating the possible configurations of wire triangles. -/
theorem triangle_configuration_theorem (config : TriangleConfiguration) :
  (config.k = 1 ∧ config.p = 1) ∨ (config.k = 4 ∧ config.p = 2) ∨ (config.k = 7 ∧ config.p = 3) :=
sorry

end triangle_configuration_theorem_l2741_274167


namespace davids_physics_marks_l2741_274164

/-- Calculates the marks in Physics given marks in other subjects and the average --/
def physics_marks (english : ℕ) (mathematics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + mathematics + chemistry + biology)

/-- Theorem: Given David's marks and average, his Physics marks are 82 --/
theorem davids_physics_marks :
  physics_marks 86 89 87 81 85 = 82 := by
  sorry

end davids_physics_marks_l2741_274164


namespace digital_root_prime_probability_l2741_274187

/-- The digital root of a positive integer -/
def digitalRoot (n : ℕ+) : ℕ :=
  if n.val % 9 = 0 then 9 else n.val % 9

/-- Whether a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The count of numbers with prime digital roots in the first n positive integers -/
def countPrimeDigitalRoots (n : ℕ) : ℕ := sorry

theorem digital_root_prime_probability :
  (countPrimeDigitalRoots 1000 : ℚ) / 1000 = 444 / 1000 := by sorry

end digital_root_prime_probability_l2741_274187


namespace meat_market_sales_l2741_274195

theorem meat_market_sales (thursday_sales : ℝ) : 
  (2 * thursday_sales) + thursday_sales + 130 + (130 / 2) = 500 + 325 → 
  thursday_sales = 210 := by
sorry

end meat_market_sales_l2741_274195


namespace cards_distribution_l2741_274168

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  num_people - (total_cards % num_people) = 3 := by
  sorry

end cards_distribution_l2741_274168


namespace power_comparisons_l2741_274174

theorem power_comparisons :
  (3^40 > 4^30 ∧ 4^30 > 5^20) ∧
  (16^31 > 8^41 ∧ 8^41 > 4^61) ∧
  (∀ a b : ℝ, a > 1 → b > 1 → a^5 = 2 → b^7 = 3 → a < b) := by
  sorry


end power_comparisons_l2741_274174


namespace third_term_coefficient_a_plus_b_10_l2741_274115

def binomial_coefficient (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem third_term_coefficient_a_plus_b_10 :
  binomial_coefficient 10 2 = 45 := by
  sorry

end third_term_coefficient_a_plus_b_10_l2741_274115


namespace reconstruct_quadrilateral_l2741_274106

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A convex quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The intersection point of the diagonals -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- The projection of a point onto a line segment -/
def projectPointOntoSegment (p : Point) (a : Point) (b : Point) : Point :=
  sorry

/-- Theorem: Given four points that are projections of the diagonal intersection
    onto the sides of a convex quadrilateral, we can reconstruct the quadrilateral -/
theorem reconstruct_quadrilateral 
  (M N K L : Point) 
  (h : ∃ (q : Quadrilateral), 
    M = projectPointOntoSegment (diagonalIntersection q) q.A q.B ∧
    N = projectPointOntoSegment (diagonalIntersection q) q.B q.C ∧
    K = projectPointOntoSegment (diagonalIntersection q) q.C q.D ∧
    L = projectPointOntoSegment (diagonalIntersection q) q.D q.A) :
  ∃! (q : Quadrilateral), 
    M = projectPointOntoSegment (diagonalIntersection q) q.A q.B ∧
    N = projectPointOntoSegment (diagonalIntersection q) q.B q.C ∧
    K = projectPointOntoSegment (diagonalIntersection q) q.C q.D ∧
    L = projectPointOntoSegment (diagonalIntersection q) q.D q.A :=
  sorry

end reconstruct_quadrilateral_l2741_274106


namespace smallest_winning_number_l2741_274110

theorem smallest_winning_number : ∃ (N : ℕ), 
  (0 ≤ N ∧ N ≤ 1999) ∧ 
  (∃ (k : ℕ), 1900 ≤ 2 * N + 100 * k ∧ 2 * N + 100 * k ≤ 1999) ∧
  (∀ (M : ℕ), M < N → ¬∃ (j : ℕ), 1900 ≤ 2 * M + 100 * j ∧ 2 * M + 100 * j ≤ 1999) ∧
  N = 800 := by
  sorry

end smallest_winning_number_l2741_274110


namespace dog_cat_food_weight_difference_l2741_274148

theorem dog_cat_food_weight_difference :
  -- Define the constants from the problem
  let cat_food_bags : ℕ := 2
  let cat_food_weight_per_bag : ℕ := 3 -- in pounds
  let dog_food_bags : ℕ := 2
  let ounces_per_pound : ℕ := 16
  let total_pet_food_ounces : ℕ := 256

  -- Calculate total cat food weight in ounces
  let total_cat_food_ounces : ℕ := cat_food_bags * cat_food_weight_per_bag * ounces_per_pound
  
  -- Calculate total dog food weight in ounces
  let total_dog_food_ounces : ℕ := total_pet_food_ounces - total_cat_food_ounces
  
  -- Calculate weight per bag of dog food in ounces
  let dog_food_weight_per_bag_ounces : ℕ := total_dog_food_ounces / dog_food_bags
  
  -- Calculate weight per bag of cat food in ounces
  let cat_food_weight_per_bag_ounces : ℕ := cat_food_weight_per_bag * ounces_per_pound
  
  -- Calculate the difference in weight between dog and cat food bags in ounces
  let weight_difference_ounces : ℕ := dog_food_weight_per_bag_ounces - cat_food_weight_per_bag_ounces
  
  -- Convert the weight difference to pounds
  weight_difference_ounces / ounces_per_pound = 2 := by
  sorry

end dog_cat_food_weight_difference_l2741_274148


namespace min_sum_distances_squared_l2741_274123

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of the center of the ellipse -/
def center : ℝ × ℝ := (0, 0)

/-- Definition of the left focus of the ellipse -/
def left_focus : ℝ × ℝ := (-1, 0)

/-- Square of the distance between two points -/
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The minimum value of |OP|^2 + |PF|^2 is 2 -/
theorem min_sum_distances_squared :
  ∀ (x y : ℝ), is_on_ellipse x y →
  ∃ (min : ℝ), min = 2 ∧
  ∀ (p : ℝ × ℝ), is_on_ellipse p.1 p.2 →
  distance_squared center p + distance_squared p left_focus ≥ min :=
sorry

end min_sum_distances_squared_l2741_274123


namespace geometric_mean_of_4_and_16_l2741_274179

theorem geometric_mean_of_4_and_16 : 
  ∃ x : ℝ, x > 0 ∧ x^2 = 4 * 16 ∧ (x = 8 ∨ x = -8) :=
by sorry

end geometric_mean_of_4_and_16_l2741_274179
