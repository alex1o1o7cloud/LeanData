import Mathlib

namespace midpoint_quadrilateral_perimeter_l3931_393192

/-- A rectangle with a diagonal of 8 units -/
structure Rectangle :=
  (diagonal : ℝ)
  (diagonal_eq : diagonal = 8)

/-- A quadrilateral formed by connecting the midpoints of the sides of a rectangle -/
def MidpointQuadrilateral (rect : Rectangle) : Set (ℝ × ℝ) :=
  sorry

/-- The perimeter of a quadrilateral -/
def perimeter (quad : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem midpoint_quadrilateral_perimeter (rect : Rectangle) :
  perimeter (MidpointQuadrilateral rect) = 16 :=
sorry

end midpoint_quadrilateral_perimeter_l3931_393192


namespace solve_equation_l3931_393128

theorem solve_equation (x : ℝ) (h : Real.sqrt ((2 / x) + 2) = 4 / 3) : x = -9 := by
  sorry

end solve_equation_l3931_393128


namespace children_who_got_off_bus_l3931_393160

theorem children_who_got_off_bus (initial_children : ℕ) (remaining_children : ℕ) 
  (h1 : initial_children = 43) 
  (h2 : remaining_children = 21) : 
  initial_children - remaining_children = 22 := by
  sorry

end children_who_got_off_bus_l3931_393160


namespace percent_palindromes_with_seven_l3931_393175

/-- A palindrome between 1000 and 2000 -/
structure Palindrome :=
  (x y : Fin 10)

/-- Checks if a palindrome contains at least one 7 -/
def containsSeven (p : Palindrome) : Prop :=
  p.x = 7 ∨ p.y = 7

/-- The set of all palindromes between 1000 and 2000 -/
def allPalindromes : Finset Palindrome :=
  sorry

/-- The set of palindromes containing at least one 7 -/
def palindromesWithSeven : Finset Palindrome :=
  sorry

theorem percent_palindromes_with_seven :
  (palindromesWithSeven.card : ℚ) / allPalindromes.card = 19 / 100 := by
  sorry

end percent_palindromes_with_seven_l3931_393175


namespace equation_solutions_l3931_393148

theorem equation_solutions : 
  ∀ x : ℝ, x ≠ 3 ∧ x ≠ 5 →
  ((x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 3) * (x - 2) * (x - 1)) / 
  ((x - 3) * (x - 5) * (x - 3)) = 1 ↔ 
  x = 1 ∨ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2 :=
by sorry

end equation_solutions_l3931_393148


namespace root_conditions_imply_inequalities_l3931_393170

theorem root_conditions_imply_inequalities (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b > 0) (hc : c ≠ 0)
  (h_distinct : ∃ x y : ℝ, x ≠ y ∧ 
    a * x^2 + b * x - c = 0 ∧ 
    a * y^2 + b * y - c = 0)
  (h_cubic : ∀ x : ℝ, a * x^2 + b * x - c = 0 → 
    x^3 + b * x^2 + a * x - c = 0) :
  a * b * c > 16 ∧ a * b * c ≥ 3125 / 108 := by
  sorry

end root_conditions_imply_inequalities_l3931_393170


namespace equilateral_triangle_area_perimeter_ratio_l3931_393181

/-- The ratio of area to perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 10
  let perimeter : ℝ := 3 * s
  let area : ℝ := (Real.sqrt 3 / 4) * s^2
  area / perimeter = 5 * Real.sqrt 3 / 6 := by
  sorry

end equilateral_triangle_area_perimeter_ratio_l3931_393181


namespace tangent_slope_angle_l3931_393106

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the point of interest
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_slope_angle :
  let slope := (deriv f) point.1
  Real.arctan slope = π/4 := by sorry

end tangent_slope_angle_l3931_393106


namespace price_reduction_theorem_l3931_393116

/-- The price of an imported car after 5 years of annual reduction -/
def price_after_five_years (initial_price : ℝ) (annual_reduction_rate : ℝ) : ℝ :=
  initial_price * (1 - annual_reduction_rate)^5

/-- Theorem stating the relationship between the initial price, 
    annual reduction rate, and final price after 5 years -/
theorem price_reduction_theorem (x : ℝ) :
  price_after_five_years 300000 (x / 100) = 30000 * (1 - x / 100)^5 := by
  sorry

#check price_reduction_theorem

end price_reduction_theorem_l3931_393116


namespace student_scores_l3931_393105

theorem student_scores (M P C : ℕ) : 
  M + P = 50 →
  (M + C) / 2 = 35 →
  C > P →
  C - P = 20 := by
sorry

end student_scores_l3931_393105


namespace max_nmmm_value_l3931_393187

/-- Represents a three-digit number where all digits are the same -/
def three_digit_same (d : ℕ) : ℕ := 100 * d + 10 * d + d

/-- Represents a four-digit number NMMM where the last three digits are the same -/
def four_digit_nmmm (n m : ℕ) : ℕ := 1000 * n + 100 * m + 10 * m + m

/-- The maximum value of NMMM given the problem conditions -/
theorem max_nmmm_value :
  ∀ m : ℕ,
  1 ≤ m → m ≤ 9 →
  (∃ n : ℕ, four_digit_nmmm n m = m * three_digit_same m) →
  (∀ k : ℕ, k ≤ 9 → 
    (∃ l : ℕ, four_digit_nmmm l k = k * three_digit_same k) →
    four_digit_nmmm l k ≤ 3996) :=
by sorry

end max_nmmm_value_l3931_393187


namespace evaluate_expression_l3931_393157

theorem evaluate_expression : (-3)^6 / 3^4 + 2^5 - 7^2 = -8 := by
  sorry

end evaluate_expression_l3931_393157


namespace min_sum_of_product_1020_l3931_393169

theorem min_sum_of_product_1020 (a b c : ℕ+) (h : a * b * c = 1020) :
  ∃ (x y z : ℕ+), x * y * z = 1020 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 33 := by
  sorry

end min_sum_of_product_1020_l3931_393169


namespace perfect_square_trinomial_l3931_393110

theorem perfect_square_trinomial (a : ℝ) : a^2 + 2*a + 1 = (a + 1)^2 := by
  sorry

end perfect_square_trinomial_l3931_393110


namespace three_number_problem_l3931_393113

theorem three_number_problem (x y z : ℝ) : 
  x + y + z = 19 → 
  y^2 = x * z → 
  y = (2/3) * z → 
  x = 4 ∧ y = 6 ∧ z = 9 := by
sorry

end three_number_problem_l3931_393113


namespace max_followers_count_l3931_393135

/-- Represents the types of islanders --/
inductive IslanderType
  | Knight
  | Liar
  | Follower

/-- Represents an answer to the question --/
inductive Answer
  | Yes
  | No

/-- Defines the properties of the island and its inhabitants --/
structure Island where
  totalPopulation : Nat
  knightCount : Nat
  liarCount : Nat
  followerCount : Nat
  yesAnswers : Nat
  noAnswers : Nat

/-- Defines the conditions of the problem --/
def isValidIsland (i : Island) : Prop :=
  i.totalPopulation = 2018 ∧
  i.knightCount + i.liarCount + i.followerCount = i.totalPopulation ∧
  i.yesAnswers = 1009 ∧
  i.noAnswers = i.totalPopulation - i.yesAnswers

/-- The main theorem to prove --/
theorem max_followers_count (i : Island) (h : isValidIsland i) :
  i.followerCount ≤ 1009 ∧ ∃ (j : Island), isValidIsland j ∧ j.followerCount = 1009 :=
sorry

end max_followers_count_l3931_393135


namespace parabola_c_value_l3931_393179

/-- Given a parabola y = x^2 + bx + c passing through (2,3) and (4,3), prove c = 11 -/
theorem parabola_c_value (b c : ℝ) 
  (eq1 : 3 = 2^2 + 2*b + c) 
  (eq2 : 3 = 4^2 + 4*b + c) : 
  c = 11 := by sorry

end parabola_c_value_l3931_393179


namespace three_digit_square_ends_with_self_l3931_393163

theorem three_digit_square_ends_with_self (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) ∧ (A^2 % 1000 = A) ↔ (A = 376 ∨ A = 625) := by
sorry

end three_digit_square_ends_with_self_l3931_393163


namespace tank_leak_consistency_l3931_393111

/-- Proves that a leak emptying a tank in 12 hours without an inlet pipe is consistent with the given scenario. -/
theorem tank_leak_consistency 
  (tank_capacity : ℝ) 
  (inlet_rate : ℝ) 
  (emptying_time_with_inlet : ℝ) 
  (emptying_time_without_inlet : ℝ) : 
  tank_capacity = 5760 ∧ 
  inlet_rate = 4 ∧ 
  emptying_time_with_inlet = 8 * 60 ∧ 
  emptying_time_without_inlet = 12 * 60 → 
  ∃ (leak_rate : ℝ), 
    leak_rate > 0 ∧
    tank_capacity / leak_rate = emptying_time_without_inlet ∧
    tank_capacity / (leak_rate - inlet_rate) = emptying_time_with_inlet :=
by sorry

#check tank_leak_consistency

end tank_leak_consistency_l3931_393111


namespace canoe_downstream_speed_l3931_393120

/-- The speed of a canoe rowing downstream, given its upstream speed against a stream -/
theorem canoe_downstream_speed
  (upstream_speed : ℝ)
  (stream_speed : ℝ)
  (h1 : upstream_speed = 4)
  (h2 : stream_speed = 4) :
  upstream_speed + 2 * stream_speed = 12 :=
by sorry

end canoe_downstream_speed_l3931_393120


namespace toms_age_ratio_l3931_393194

/-- Theorem representing Tom's age problem -/
theorem toms_age_ratio (T N : ℚ) : 
  (T > 0) →  -- Tom's age is positive
  (N > 0) →  -- N is positive (years in the past)
  (T - N > 0) →  -- Tom's age N years ago was positive
  (T - 4*N ≥ 0) →  -- Sum of children's ages N years ago was non-negative
  (T - N = 3*(T - 4*N)) →  -- Condition relating Tom's age N years ago to his children's ages
  T / N = 11 / 2 := by
  sorry

end toms_age_ratio_l3931_393194


namespace bagel_cut_theorem_l3931_393145

/-- The number of pieces resulting from cutting a bagel -/
def bagel_pieces (n : ℕ) : ℕ := n + 1

/-- Theorem: Cutting a bagel with 10 cuts results in 11 pieces -/
theorem bagel_cut_theorem :
  bagel_pieces 10 = 11 :=
by sorry

end bagel_cut_theorem_l3931_393145


namespace tonys_normal_temp_l3931_393117

/-- Tony's normal body temperature -/
def normal_temp : ℝ := 95

/-- The fever threshold temperature -/
def fever_threshold : ℝ := 100

/-- Tony's current temperature -/
def current_temp : ℝ := normal_temp + 10

theorem tonys_normal_temp :
  (current_temp = fever_threshold + 5) →
  (fever_threshold = 100) →
  (normal_temp = 95) := by sorry

end tonys_normal_temp_l3931_393117


namespace white_rhino_weight_is_5100_l3931_393124

/-- The weight of one white rhino in pounds -/
def white_rhino_weight : ℝ := 5100

/-- The weight of one black rhino in pounds -/
def black_rhino_weight : ℝ := 2000

/-- The total weight of 7 white rhinos and 8 black rhinos in pounds -/
def total_weight : ℝ := 51700

/-- Theorem: The weight of one white rhino is 5100 pounds -/
theorem white_rhino_weight_is_5100 :
  7 * white_rhino_weight + 8 * black_rhino_weight = total_weight :=
by sorry

end white_rhino_weight_is_5100_l3931_393124


namespace sum_of_coefficients_l3931_393140

theorem sum_of_coefficients (a b c d : ℤ) :
  (∀ x, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + x^3 - 2*x^2 + 17*x + 15) →
  a + b + c + d = 9 := by
  sorry

end sum_of_coefficients_l3931_393140


namespace nancy_games_this_month_l3931_393109

/-- Represents the number of football games Nancy attended or plans to attend -/
structure FootballGames where
  lastMonth : ℕ
  thisMonth : ℕ
  nextMonth : ℕ
  total : ℕ

/-- Theorem stating that Nancy attended 9 games this month -/
theorem nancy_games_this_month (g : FootballGames)
  (h1 : g.lastMonth = 8)
  (h2 : g.nextMonth = 7)
  (h3 : g.total = 24)
  (h4 : g.total = g.lastMonth + g.thisMonth + g.nextMonth) :
  g.thisMonth = 9 := by
  sorry


end nancy_games_this_month_l3931_393109


namespace orthic_triangle_inradius_bound_l3931_393185

/-- Given a triangle ABC with circumradius R = 1 and inradius r, 
    the inradius P of its orthic triangle A'B'C' satisfies P ≤ 1 - (1/3)(1+r)^2 -/
theorem orthic_triangle_inradius_bound (R r P : ℝ) : 
  R = 1 → 0 < r → r ≤ 1/2 → P ≤ 1 - (1/3) * (1 + r)^2 := by
  sorry

end orthic_triangle_inradius_bound_l3931_393185


namespace quadratic_equation_roots_range_l3931_393198

theorem quadratic_equation_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ↔ 
  m < -2 ∨ m > 2 :=
sorry

end quadratic_equation_roots_range_l3931_393198


namespace disjunction_true_l3931_393131

theorem disjunction_true : 
  (∀ x > 0, ∃ y, y = x + 1/(2*x) ∧ y ≥ 1 ∧ ∀ z, z = x + 1/(2*x) → z ≥ y) ∨ 
  (∀ x > 1, x^2 + 2*x - 3 > 0) := by
sorry

end disjunction_true_l3931_393131


namespace hemisphere_volume_calculation_l3931_393154

/-- Given a total volume of water and the number of hemisphere containers,
    calculate the volume of each hemisphere container. -/
def hemisphere_volume (total_volume : ℚ) (num_containers : ℕ) : ℚ :=
  total_volume / num_containers

/-- Theorem stating that the volume of each hemisphere container is 4 L
    when 2735 containers are used to hold 10940 L of water. -/
theorem hemisphere_volume_calculation :
  hemisphere_volume 10940 2735 = 4 := by
  sorry

end hemisphere_volume_calculation_l3931_393154


namespace betty_nuts_purchase_l3931_393168

/-- The number of packs of nuts Betty wants to buy -/
def num_packs : ℕ := 20

/-- Betty's age -/
def betty_age : ℕ := 50

/-- Doug's age -/
def doug_age : ℕ := 40

/-- Cost of one pack of nuts -/
def pack_cost : ℕ := 100

/-- Total cost Betty wants to spend on nuts -/
def total_cost : ℕ := 2000

theorem betty_nuts_purchase :
  (2 * betty_age = pack_cost) ∧
  (betty_age + doug_age = 90) ∧
  (num_packs * pack_cost = total_cost) →
  num_packs = 20 := by sorry

end betty_nuts_purchase_l3931_393168


namespace greening_project_optimization_l3931_393144

/-- The optimization problem for greening project --/
theorem greening_project_optimization (total_area : ℝ) (team_a_rate : ℝ) (team_b_rate : ℝ)
  (team_a_wage : ℝ) (team_b_wage : ℝ) (h1 : total_area = 1200)
  (h2 : team_a_rate = 100) (h3 : team_b_rate = 50) (h4 : team_a_wage = 4000) (h5 : team_b_wage = 3000) :
  ∃ (days_a days_b : ℝ),
    days_a ≥ 3 ∧ days_b ≥ days_a ∧
    team_a_rate * days_a + team_b_rate * days_b = total_area ∧
    ∀ (x y : ℝ),
      x ≥ 3 → y ≥ x →
      team_a_rate * x + team_b_rate * y = total_area →
      team_a_wage * days_a + team_b_wage * days_b ≤ team_a_wage * x + team_b_wage * y ∧
      team_a_wage * days_a + team_b_wage * days_b = 56000 :=
by
  sorry

end greening_project_optimization_l3931_393144


namespace unique_polygon_pair_l3931_393133

/-- The interior angle of a regular polygon with n sides --/
def interior_angle (n : ℕ) : ℚ :=
  180 - 360 / n

/-- The condition for the ratio of interior angles to be 5:3 --/
def angle_ratio_condition (a b : ℕ) : Prop :=
  interior_angle a / interior_angle b = 5 / 3

/-- The main theorem --/
theorem unique_polygon_pair :
  ∃! (pair : ℕ × ℕ), 
    pair.1 > 2 ∧ 
    pair.2 > 2 ∧ 
    angle_ratio_condition pair.1 pair.2 :=
sorry

end unique_polygon_pair_l3931_393133


namespace max_value_of_f_l3931_393174

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a x ≤ f a y) ∧ 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, 3 ≤ f a x) →
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a y ≤ f a x ∧ f a x = 57 :=
by
  sorry


end max_value_of_f_l3931_393174


namespace set_intersection_problem_l3931_393127

/-- Given sets M and N, prove their intersection -/
theorem set_intersection_problem (M N : Set ℝ) 
  (hM : M = {x : ℝ | -2 < x ∧ x < 2})
  (hN : N = {x : ℝ | |x - 1| ≤ 2}) :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end set_intersection_problem_l3931_393127


namespace rectangular_field_dimensions_l3931_393180

theorem rectangular_field_dimensions :
  ∀ (width length : ℝ),
  width > 0 →
  length = 2 * width →
  width * length = 800 →
  width = 20 ∧ length = 40 := by
sorry

end rectangular_field_dimensions_l3931_393180


namespace odd_increasing_function_inequality_l3931_393136

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

theorem odd_increasing_function_inequality 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_incr : is_increasing f) 
  (m : ℝ) 
  (h_ineq : ∀ θ : ℝ, f (Real.cos (2 * θ) - 5) + f (2 * m + 4 * Real.sin θ) > 0) :
  m > 5 := by
sorry

end odd_increasing_function_inequality_l3931_393136


namespace second_worker_time_l3931_393112

/-- The time it takes for two workers to load a truck together -/
def combined_time : ℚ := 30 / 11

/-- The time it takes for the first worker to load a truck alone -/
def worker1_time : ℚ := 6

/-- Theorem stating that the second worker's time to load a truck alone is 5 hours -/
theorem second_worker_time :
  ∃ (worker2_time : ℚ),
    worker2_time = 5 ∧
    1 / worker1_time + 1 / worker2_time = 1 / combined_time :=
by sorry

end second_worker_time_l3931_393112


namespace second_throw_difference_l3931_393158

/-- Represents the number of skips for each throw -/
structure Throws :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)
  (fifth : ℕ)

/-- Conditions for the stone skipping problem -/
def StoneSkippingProblem (t : Throws) : Prop :=
  t.third = 2 * t.second ∧
  t.fourth = t.third - 3 ∧
  t.fifth = t.fourth + 1 ∧
  t.fifth = 8 ∧
  t.first + t.second + t.third + t.fourth + t.fifth = 33

theorem second_throw_difference (t : Throws) 
  (h : StoneSkippingProblem t) : t.second - t.first = 2 := by
  sorry

end second_throw_difference_l3931_393158


namespace cube_edge_increase_l3931_393102

theorem cube_edge_increase (surface_area_increase : Real) 
  (h : surface_area_increase = 69.00000000000001) : 
  ∃ edge_increase : Real, 
    edge_increase = 30 ∧ 
    (1 + edge_increase / 100)^2 = 1 + surface_area_increase / 100 := by
  sorry

end cube_edge_increase_l3931_393102


namespace percentage_time_in_meetings_l3931_393134

/-- Calculates the percentage of time spent in meetings during a work shift -/
theorem percentage_time_in_meetings
  (shift_hours : ℕ) -- Total hours in the shift
  (meeting1_minutes : ℕ) -- Duration of first meeting in minutes
  (meeting2_multiplier : ℕ) -- Multiplier for second meeting duration
  (meeting3_divisor : ℕ) -- Divisor for third meeting duration
  (h1 : shift_hours = 10)
  (h2 : meeting1_minutes = 30)
  (h3 : meeting2_multiplier = 2)
  (h4 : meeting3_divisor = 2)
  : (meeting1_minutes + meeting2_multiplier * meeting1_minutes + 
     (meeting2_multiplier * meeting1_minutes) / meeting3_divisor) * 100 / 
    (shift_hours * 60) = 20 := by
  sorry

#check percentage_time_in_meetings

end percentage_time_in_meetings_l3931_393134


namespace gcd_problem_l3931_393199

theorem gcd_problem : Nat.gcd (122^2 + 234^2 + 345^2 + 10) (123^2 + 233^2 + 347^2 + 10) = 1 := by
  sorry

end gcd_problem_l3931_393199


namespace triangle_properties_l3931_393114

-- Define the triangle
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b)
  (h2 : Real.cos t.B = 1/4)
  (h3 : t.b = 2) :
  Real.sin t.C / Real.sin t.A = 2 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 15 / 4 := by
  sorry

end triangle_properties_l3931_393114


namespace intersection_of_M_and_N_l3931_393123

def M : Set ℝ := {y | ∃ x, y = 3 - x^2}
def N : Set ℝ := {y | ∃ x, y = 2*x^2 - 1}

theorem intersection_of_M_and_N : M ∩ N = Set.Icc (-1) 3 := by
  sorry

end intersection_of_M_and_N_l3931_393123


namespace set_union_problem_l3931_393166

theorem set_union_problem (a b : ℕ) :
  let A : Set ℕ := {5, a + 1}
  let B : Set ℕ := {a, b}
  A ∩ B = {2} → A ∪ B = {1, 2, 5} := by
  sorry

end set_union_problem_l3931_393166


namespace expedition_duration_l3931_393100

theorem expedition_duration (total_time : ℝ) (ratio : ℝ) (h1 : total_time = 10) (h2 : ratio = 3) :
  let first_expedition := total_time / (1 + ratio)
  first_expedition = 2.5 := by
sorry

end expedition_duration_l3931_393100


namespace clock_strike_theorem_l3931_393149

/-- Calculates the time taken for a clock to strike a given number of times,
    given the time it takes to strike 3 times. -/
def strike_time (time_for_three : ℕ) (num_strikes : ℕ) : ℕ :=
  let interval_time := time_for_three / 2
  interval_time * (num_strikes - 1)

/-- Theorem stating that if a clock takes 6 seconds to strike 3 times,
    it will take 33 seconds to strike 12 times. -/
theorem clock_strike_theorem :
  strike_time 6 12 = 33 := by
  sorry

end clock_strike_theorem_l3931_393149


namespace phone_production_ratio_l3931_393125

/-- Proves that the ratio of this year's production to last year's production is 2:1 --/
theorem phone_production_ratio :
  ∀ (this_year last_year : ℕ),
  last_year = 5000 →
  (3 * this_year) / 4 = 7500 →
  (this_year : ℚ) / last_year = 2 := by
sorry

end phone_production_ratio_l3931_393125


namespace main_line_probability_l3931_393143

/-- Represents a train schedule -/
structure TrainSchedule where
  start_time : ℕ
  frequency : ℕ

/-- Calculates the probability of getting the main line train -/
def probability_main_line (main : TrainSchedule) (harbor : TrainSchedule) : ℚ :=
  1 / 2

/-- Theorem stating that the probability of getting the main line train is 1/2 -/
theorem main_line_probability 
  (main : TrainSchedule) 
  (harbor : TrainSchedule) 
  (h1 : main.start_time = 0)
  (h2 : harbor.start_time = 2)
  (h3 : main.frequency = 10)
  (h4 : harbor.frequency = 10) :
  probability_main_line main harbor = 1 / 2 := by
  sorry

end main_line_probability_l3931_393143


namespace missing_number_proof_l3931_393171

theorem missing_number_proof (n : ℕ) (sum_with_missing : ℕ) : 
  (n = 63) → 
  (sum_with_missing = 2012) → 
  (n * (n + 1) / 2 - sum_with_missing = 4) :=
by sorry

end missing_number_proof_l3931_393171


namespace conditional_probability_B_given_A_l3931_393172

-- Define the sample space for a single die roll
def Die : Type := Fin 6

-- Define the probability space for two dice rolls
def TwoDice : Type := Die × Die

-- Define event A: sum of dice is even
def eventA (roll : TwoDice) : Prop :=
  (roll.1.val + 1 + roll.2.val + 1) % 2 = 0

-- Define event B: sum of dice is less than 7
def eventB (roll : TwoDice) : Prop :=
  roll.1.val + 1 + roll.2.val + 1 < 7

-- Define the probability measure
def P : Set TwoDice → ℝ := sorry

-- State the theorem
theorem conditional_probability_B_given_A :
  P {roll : TwoDice | eventB roll ∧ eventA roll} / P {roll : TwoDice | eventA roll} = 1/2 := by
  sorry

end conditional_probability_B_given_A_l3931_393172


namespace wade_hot_dog_truck_l3931_393130

theorem wade_hot_dog_truck (tips_per_customer : ℚ) (friday_customers : ℕ) (total_tips : ℚ) :
  tips_per_customer = 2 →
  friday_customers = 28 →
  total_tips = 296 →
  let saturday_customers := 3 * friday_customers
  let sunday_customers := (total_tips - tips_per_customer * (friday_customers + saturday_customers)) / tips_per_customer
  sunday_customers = 36 := by
sorry


end wade_hot_dog_truck_l3931_393130


namespace alpha_cheaper_at_min_shirts_l3931_393147

/-- Alpha T-Shirt Company's pricing model -/
def alpha_cost (n : ℕ) : ℚ := 80 + 12 * n

/-- Omega T-Shirt Company's pricing model -/
def omega_cost (n : ℕ) : ℚ := 10 + 18 * n

/-- The minimum number of shirts for which Alpha becomes cheaper -/
def min_shirts_for_alpha : ℕ := 12

theorem alpha_cheaper_at_min_shirts :
  alpha_cost min_shirts_for_alpha < omega_cost min_shirts_for_alpha ∧
  ∀ m : ℕ, m < min_shirts_for_alpha → alpha_cost m ≥ omega_cost m :=
by sorry

end alpha_cheaper_at_min_shirts_l3931_393147


namespace bmw_sales_l3931_393177

/-- Proves that the number of BMWs sold is 135 given the specified conditions -/
theorem bmw_sales (total : ℕ) (audi_percent : ℚ) (toyota_percent : ℚ) (acura_percent : ℚ)
  (h_total : total = 300)
  (h_audi : audi_percent = 12 / 100)
  (h_toyota : toyota_percent = 25 / 100)
  (h_acura : acura_percent = 18 / 100)
  (h_sum : audi_percent + toyota_percent + acura_percent < 1) :
  ↑total * (1 - (audi_percent + toyota_percent + acura_percent)) = 135 := by
  sorry


end bmw_sales_l3931_393177


namespace product_xyz_is_negative_one_l3931_393119

theorem product_xyz_is_negative_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 1) 
  (h2 : y + 1/z = 1) : 
  x * y * z = -1 := by
sorry

end product_xyz_is_negative_one_l3931_393119


namespace tiles_required_to_cover_floor_l3931_393118

-- Define the dimensions
def floor_length : ℚ := 10
def floor_width : ℚ := 15
def tile_length : ℚ := 5 / 12  -- 5 inches in feet
def tile_width : ℚ := 2 / 3    -- 8 inches in feet

-- Theorem statement
theorem tiles_required_to_cover_floor :
  (floor_length * floor_width) / (tile_length * tile_width) = 540 := by
  sorry

end tiles_required_to_cover_floor_l3931_393118


namespace complex_sum_powers_l3931_393155

theorem complex_sum_powers (z : ℂ) (h : z = (1 - Complex.I) / (1 + Complex.I)) :
  z^2 + z^4 + z^6 + z^8 + z^10 = -1 := by
  sorry

end complex_sum_powers_l3931_393155


namespace inequality_equivalence_l3931_393161

theorem inequality_equivalence (x : ℝ) : (x - 2) / 2 ≥ (7 - x) / 3 ↔ x ≥ 4 := by
  sorry

end inequality_equivalence_l3931_393161


namespace intersection_of_A_and_B_l3931_393103

def A : Set ℤ := {1, 3, 5}
def B : Set ℤ := {-1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end intersection_of_A_and_B_l3931_393103


namespace ellipse_circle_intersection_l3931_393137

/-- The ellipse C defined by x^2 + 16y^2 = 16 -/
def ellipse_C (x y : ℝ) : Prop := x^2 + 16 * y^2 = 16

/-- The circle Γ with center (0, h) and radius r -/
def circle_Γ (h r : ℝ) (x y : ℝ) : Prop := x^2 + (y - h)^2 = r^2

/-- The foci of ellipse C -/
def foci : Set (ℝ × ℝ) := {(-Real.sqrt 15, 0), (Real.sqrt 15, 0)}

theorem ellipse_circle_intersection (a b : ℝ) :
  (∃ r h, r ∈ Set.Icc a b ∧
    (∀ (f : ℝ × ℝ), f ∈ foci → circle_Γ h r f.1 f.2) ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,
      ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧ ellipse_C x₃ y₃ ∧ ellipse_C x₄ y₄ ∧
      circle_Γ h r x₁ y₁ ∧ circle_Γ h r x₂ y₂ ∧ circle_Γ h r x₃ y₃ ∧ circle_Γ h r x₄ y₄ ∧
      (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
      (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄))) →
  a + b = Real.sqrt 15 + 8 :=
by sorry

end ellipse_circle_intersection_l3931_393137


namespace carrie_farm_earnings_l3931_393197

def total_money (num_tomatoes : ℕ) (num_carrots : ℕ) (price_tomato : ℚ) (price_carrot : ℚ) : ℚ :=
  num_tomatoes * price_tomato + num_carrots * price_carrot

theorem carrie_farm_earnings :
  total_money 200 350 1 (3/2) = 725 := by
  sorry

end carrie_farm_earnings_l3931_393197


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l3931_393173

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x : ℝ => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  (∃ x y : ℝ, x ≠ y ∧ equation x = 0 ∧ equation y = 0) →
  sum_of_roots = x + y :=
sorry

theorem sum_of_roots_specific_equation :
  let equation := fun x : ℝ => x^2 + 2023 * x - 2024
  let sum_of_roots := -2023
  (∃ x y : ℝ, x ≠ y ∧ equation x = 0 ∧ equation y = 0) →
  sum_of_roots = x + y :=
sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l3931_393173


namespace units_digit_of_sum_of_powers_divided_by_11_l3931_393104

theorem units_digit_of_sum_of_powers_divided_by_11 : 
  (3^2018 + 7^2018) % 11 = 3 := by
  sorry

end units_digit_of_sum_of_powers_divided_by_11_l3931_393104


namespace function_always_negative_l3931_393101

theorem function_always_negative (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ m ∈ Set.Ioc (-4) 0 := by
sorry

end function_always_negative_l3931_393101


namespace expression_equality_l3931_393184

theorem expression_equality (x : ℝ) (h : x ≥ 1) :
  let expr := Real.sqrt (x + 2 * Real.sqrt (x - 1)) + Real.sqrt (x - 2 * Real.sqrt (x - 1))
  (x ≤ 2 → expr = 2) ∧ (x > 2 → expr = 2 * Real.sqrt (x - 1)) := by
  sorry

end expression_equality_l3931_393184


namespace max_value_f_l3931_393189

open Real

/-- The maximum value of f(m, n) given the conditions -/
theorem max_value_f (f g : ℝ → ℝ) (m n : ℝ) :
  (∀ x > 0, f x = log x) →
  (∀ x, g x = (2*m + 3)*x + n) →
  (∀ x > 0, f x ≤ g x) →
  let f_mn := (2*m + 3) * n
  ∃ (min_f_mn : ℝ), f_mn ≥ min_f_mn ∧ 
    (∀ m' n', (∀ x > 0, log x ≤ (2*m' + 3)*x + n') → (2*m' + 3) * n' ≥ min_f_mn) →
  (∃ (max_value : ℝ), max_value = 1 / Real.exp 2 ∧
    ∀ m' n', (∀ x > 0, log x ≤ (2*m' + 3)*x + n') →
      let f_m'n' := (2*m' + 3) * n'
      ∃ (min_f_m'n' : ℝ), f_m'n' ≥ min_f_m'n' ∧
        (∀ m'' n'', (∀ x > 0, log x ≤ (2*m'' + 3)*x + n'') → (2*m'' + 3) * n'' ≥ min_f_m'n') →
      min_f_m'n' ≤ max_value) :=
by sorry

end max_value_f_l3931_393189


namespace hcf_problem_l3931_393151

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 1800) (h2 : Nat.lcm a b = 200) :
  Nat.gcd a b = 9 := by
  sorry

end hcf_problem_l3931_393151


namespace joanna_estimate_l3931_393188

theorem joanna_estimate (u v ε₁ ε₂ : ℝ) (h1 : u > v) (h2 : v > 0) (h3 : ε₁ > 0) (h4 : ε₂ > 0) :
  (u + ε₁) - (v - ε₂) > u - v := by
  sorry

end joanna_estimate_l3931_393188


namespace waiter_earnings_l3931_393186

theorem waiter_earnings (total_customers : ℕ) (non_tippers : ℕ) (tip_amount : ℕ) : 
  total_customers = 9 → 
  non_tippers = 5 → 
  tip_amount = 8 → 
  (total_customers - non_tippers) * tip_amount = 32 := by
sorry

end waiter_earnings_l3931_393186


namespace six_students_three_colleges_l3931_393139

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins,
    where each bin must contain at least one object. -/
def distributeWithMinimum (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n

/-- The specific case for 6 students and 3 colleges -/
theorem six_students_three_colleges :
  distributeWithMinimum 6 3 = 540 := by
  sorry

end six_students_three_colleges_l3931_393139


namespace inscribed_sphere_volume_l3931_393162

/-- The volume of a sphere inscribed in a cube with a given diagonal -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 10) :
  let s := d / Real.sqrt 3
  let r := s / 2
  (4 / 3) * Real.pi * r ^ 3 = (500 * Real.sqrt 3 * Real.pi) / 9 := by
  sorry

end inscribed_sphere_volume_l3931_393162


namespace problem_solution_l3931_393129

theorem problem_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 4) (h3 : c^2 / a = 4) :
  a = 64^(1/7) := by
sorry

end problem_solution_l3931_393129


namespace dracula_is_alive_l3931_393176

-- Define the possible states of a person
inductive PersonState
| Sane
| MadVampire
| Other

-- Define the Transylvanian's statement
def transylvanianStatement (personState : PersonState) (draculaAlive : Prop) : Prop :=
  (personState = PersonState.Sane ∨ personState = PersonState.MadVampire) → draculaAlive

-- Theorem to prove
theorem dracula_is_alive : ∃ (personState : PersonState), transylvanianStatement personState (∃ dracula, dracula = "alive") :=
sorry

end dracula_is_alive_l3931_393176


namespace prime_from_divisibility_condition_l3931_393141

-- Define the divisibility condition
def divisibility_condition (n : ℤ) : Prop :=
  ∀ d : ℤ, d ∣ n → (d + 1) ∣ (n + 1)

-- Theorem statement
theorem prime_from_divisibility_condition (n : ℤ) :
  divisibility_condition n → Nat.Prime (Int.natAbs n) :=
by
  sorry

end prime_from_divisibility_condition_l3931_393141


namespace eighteenth_term_of_sequence_l3931_393107

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem eighteenth_term_of_sequence (a₁ a₂ : ℝ) (h : a₂ = a₁ + 6) :
  arithmetic_sequence a₁ (a₂ - a₁) 18 = 105 :=
by
  sorry

#check eighteenth_term_of_sequence 3 9 (by norm_num)

end eighteenth_term_of_sequence_l3931_393107


namespace next_simultaneous_occurrence_l3931_393182

def museum_interval : ℕ := 18
def library_interval : ℕ := 24
def town_hall_interval : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_simultaneous_occurrence :
  ∃ (h : ℕ), h * minutes_in_hour = lcm museum_interval (lcm library_interval town_hall_interval) ∧ h = 6 := by
  sorry

end next_simultaneous_occurrence_l3931_393182


namespace existence_of_subsets_l3931_393190

/-- The set M containing integers from 1 to 10000 -/
def M : Set ℕ := Finset.range 10000

/-- The property that defines the required subsets -/
def has_unique_intersection (A : Finset (Set ℕ)) : Prop :=
  ∀ z ∈ M, ∃ B : Finset (Set ℕ), B ⊆ A ∧ B.card = 8 ∧ (⋂₀ B.toSet : Set ℕ) = {z}

/-- The main theorem stating the existence of 16 subsets with the required property -/
theorem existence_of_subsets : ∃ A : Finset (Set ℕ), A.card = 16 ∧ has_unique_intersection A := by
  sorry

end existence_of_subsets_l3931_393190


namespace school_population_l3931_393196

theorem school_population (num_boys : ℕ) (difference : ℕ) (num_girls : ℕ) : 
  num_boys = 1145 → 
  num_boys = num_girls + difference → 
  difference = 510 → 
  num_girls = 635 := by
sorry

end school_population_l3931_393196


namespace negation_of_universal_statement_l3931_393122

theorem negation_of_universal_statement :
  (¬∀ x : ℝ, x^2 - 3*x + 2 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 2 ≤ 0) := by
  sorry

end negation_of_universal_statement_l3931_393122


namespace pythagorean_fraction_bound_l3931_393167

theorem pythagorean_fraction_bound (m n t : ℝ) (h1 : m^2 + n^2 = t^2) (h2 : t ≠ 0) :
  -Real.sqrt 3 / 3 ≤ n / (m - 2 * t) ∧ n / (m - 2 * t) ≤ Real.sqrt 3 / 3 :=
by sorry

end pythagorean_fraction_bound_l3931_393167


namespace smallest_perfect_cube_multiplier_l3931_393152

def y : ℕ := 2^3^3^4^4^5^5^6^6^7^7^8^8^9

theorem smallest_perfect_cube_multiplier :
  (∃ k : ℕ, k > 0 ∧ ∃ n : ℕ, k * y = n^3) ∧
  (∀ k : ℕ, k > 0 → (∃ n : ℕ, k * y = n^3) → k ≥ 1500) :=
by sorry

end smallest_perfect_cube_multiplier_l3931_393152


namespace vincent_book_purchase_l3931_393164

/-- The number of books about outer space Vincent bought -/
def books_outer_space : ℕ := 1

/-- The number of books about animals Vincent bought -/
def books_animals : ℕ := 10

/-- The number of books about trains Vincent bought -/
def books_trains : ℕ := 3

/-- The cost of each book in dollars -/
def cost_per_book : ℕ := 16

/-- The total amount spent on books in dollars -/
def total_spent : ℕ := 224

theorem vincent_book_purchase :
  books_outer_space = 1 ∧
  books_animals = 10 ∧
  books_trains = 3 ∧
  cost_per_book = 16 ∧
  total_spent = 224 →
  books_outer_space = 1 :=
by sorry

end vincent_book_purchase_l3931_393164


namespace sphere_packing_radius_l3931_393138

/-- A configuration of spheres packed in a cube. -/
structure SpherePacking where
  cube_side : ℝ
  num_spheres : ℕ
  sphere_radius : ℝ

/-- The specific sphere packing configuration described in the problem. -/
def problem_packing : SpherePacking where
  cube_side := 2
  num_spheres := 16
  sphere_radius := 1  -- This is what we want to prove

/-- Predicate to check if a sphere packing configuration is valid according to the problem description. -/
def is_valid_packing (p : SpherePacking) : Prop :=
  p.cube_side = 2 ∧
  p.num_spheres = 16 ∧
  -- One sphere at the center, others tangent to it and three faces
  2 * p.sphere_radius = p.cube_side / 2

theorem sphere_packing_radius : 
  is_valid_packing problem_packing ∧ 
  problem_packing.sphere_radius = 1 :=
by sorry

end sphere_packing_radius_l3931_393138


namespace exponent_simplification_l3931_393191

theorem exponent_simplification (a : ℝ) : (36 * a^9)^4 * (63 * a^9)^4 = a^4 := by
  sorry

end exponent_simplification_l3931_393191


namespace bicycle_speeds_l3931_393165

/-- Represents a bicycle with front and rear gears -/
structure Bicycle where
  front_gears : Nat
  rear_gears : Nat

/-- Calculates the number of unique speeds for a bicycle -/
def unique_speeds (b : Bicycle) : Nat :=
  b.front_gears * b.rear_gears - b.rear_gears

/-- Theorem stating that a bicycle with 3 front gears and 4 rear gears has 8 unique speeds -/
theorem bicycle_speeds :
  ∃ (b : Bicycle), b.front_gears = 3 ∧ b.rear_gears = 4 ∧ unique_speeds b = 8 :=
by
  sorry

#eval unique_speeds ⟨3, 4⟩

end bicycle_speeds_l3931_393165


namespace frank_problems_per_type_l3931_393153

/-- The number of math problems composed by Bill -/
def bill_problems : ℕ := 20

/-- The number of math problems composed by Ryan -/
def ryan_problems : ℕ := 2 * bill_problems

/-- The number of math problems composed by Frank -/
def frank_problems : ℕ := 3 * ryan_problems

/-- The number of different types of math problems -/
def problem_types : ℕ := 4

theorem frank_problems_per_type :
  frank_problems / problem_types = 30 := by sorry

end frank_problems_per_type_l3931_393153


namespace quiz_win_probability_l3931_393108

def num_questions : ℕ := 4
def num_choices : ℕ := 4
def min_correct : ℕ := 3

def prob_correct_one : ℚ := 1 / num_choices

def prob_all_correct : ℚ := prob_correct_one ^ num_questions

def prob_three_correct : ℚ := num_questions * (prob_correct_one ^ 3) * (1 - prob_correct_one)

theorem quiz_win_probability :
  prob_all_correct + prob_three_correct = 13 / 256 := by
  sorry

end quiz_win_probability_l3931_393108


namespace rectangle_perimeter_rectangle_perimeter_400_l3931_393146

/-- A rectangle divided into four identical squares with a given area has a specific perimeter -/
theorem rectangle_perimeter (area : ℝ) (h_area : area > 0) : 
  ∃ (side : ℝ), 
    side > 0 ∧ 
    4 * side^2 = area ∧ 
    8 * side = 80 :=
by
  sorry

/-- The perimeter of a rectangle with area 400 square centimeters, 
    divided into four identical squares, is 80 centimeters -/
theorem rectangle_perimeter_400 : 
  ∃ (side : ℝ), 
    side > 0 ∧ 
    4 * side^2 = 400 ∧ 
    8 * side = 80 :=
by
  sorry

end rectangle_perimeter_rectangle_perimeter_400_l3931_393146


namespace digits_first_1500_even_integers_l3931_393193

/-- The number of digits in a positive integer -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The sum of digits for all positive even integers up to n -/
def sumDigitsEven (n : ℕ) : ℕ := sorry

/-- The 1500th positive even integer -/
def evenInteger1500 : ℕ := 3000

theorem digits_first_1500_even_integers :
  sumDigitsEven evenInteger1500 = 5448 := by sorry

end digits_first_1500_even_integers_l3931_393193


namespace closest_to_sqrt_65_minus_sqrt_63_l3931_393156

theorem closest_to_sqrt_65_minus_sqrt_63 :
  let options : List ℝ := [0.12, 0.13, 0.14, 0.15, 0.16]
  ∀ x ∈ options, x ≠ 0.13 →
    |Real.sqrt 65 - Real.sqrt 63 - 0.13| < |Real.sqrt 65 - Real.sqrt 63 - x| := by
  sorry

end closest_to_sqrt_65_minus_sqrt_63_l3931_393156


namespace arithmetic_sequence_2017th_term_l3931_393142

/-- An arithmetic sequence is monotonically increasing if its common difference is positive -/
def is_monotonically_increasing_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence if the ratio between consecutive terms is constant -/
def is_geometric_sequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

theorem arithmetic_sequence_2017th_term
  (a : ℕ → ℝ)
  (h_incr : is_monotonically_increasing_arithmetic a)
  (h_first : a 1 = 2)
  (h_geom : is_geometric_sequence (a 1 - 1) (a 3) (a 5 + 5)) :
  a 2017 = 1010 := by
  sorry

end arithmetic_sequence_2017th_term_l3931_393142


namespace fixed_distance_to_H_l3931_393195

/-- Given a parabola y^2 = 4x, with O as the origin and moving points A and B on the parabola,
    such that OA ⊥ OB, and OH ⊥ AB where H is the foot of the perpendicular,
    prove that the point (2,0) has a fixed distance to point H. -/
theorem fixed_distance_to_H (O A B H : ℝ × ℝ) :
  O = (0, 0) →
  (∀ (x y : ℝ), A = (x, y) → y^2 = 4*x) →
  (∀ (x y : ℝ), B = (x, y) → y^2 = 4*x) →
  (A.1 * B.1 + A.2 * B.2 = 0) →  -- OA ⊥ OB
  (∃ (m n : ℝ), ∀ (x y : ℝ), (x = m*y + n) ↔ ((x, y) = A ∨ (x, y) = B)) →  -- Line AB: x = my + n
  (O.1 * H.1 + O.2 * H.2 = 0) →  -- OH ⊥ AB
  (∃ (d : ℝ), ∀ (H' : ℝ × ℝ), 
    (O.1 * H'.1 + O.2 * H'.2 = 0) →  -- OH' ⊥ AB
    (∃ (m n : ℝ), ∀ (x y : ℝ), (x = m*y + n) ↔ ((x, y) = A ∨ (x, y) = B)) →
    ((2 - H'.1)^2 + H'.2^2 = d^2)) :=
by sorry

end fixed_distance_to_H_l3931_393195


namespace f_symmetric_f_upper_bound_f_solution_range_l3931_393126

noncomputable section

def f (x : ℝ) : ℝ := Real.log ((1 + x) / (x - 1))

theorem f_symmetric : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem f_upper_bound : ∀ x : ℝ, x > 1 → f x + Real.log (0.5 * (x - 1)) < -1 := by sorry

theorem f_solution_range (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 3 ∧ f x = Real.log (0.5 * (x + k))) ↔ k ∈ Set.Icc (-1) 1 := by sorry

end

end f_symmetric_f_upper_bound_f_solution_range_l3931_393126


namespace cake_sharing_percentage_l3931_393159

theorem cake_sharing_percentage (total : ℝ) (rich_portion : ℝ) (ben_portion : ℝ) : 
  total > 0 →
  rich_portion > 0 →
  ben_portion > 0 →
  rich_portion + ben_portion = total →
  rich_portion / ben_portion = 3 →
  ben_portion / total = 1 / 4 := by
sorry

end cake_sharing_percentage_l3931_393159


namespace television_selection_count_l3931_393178

def type_a_count : ℕ := 4
def type_b_count : ℕ := 5
def selection_size : ℕ := 3

theorem television_selection_count :
  (type_a_count.choose 1) * (type_b_count.choose 1) * ((type_a_count + type_b_count - 2).choose 1) = 140 := by
  sorry

end television_selection_count_l3931_393178


namespace power_of_three_mod_seven_l3931_393183

theorem power_of_three_mod_seven : 3^1503 % 7 = 6 := by
  sorry

end power_of_three_mod_seven_l3931_393183


namespace triangle_area_change_l3931_393132

theorem triangle_area_change (b h : ℝ) (h_pos : 0 < h) (b_pos : 0 < b) :
  let new_height := 0.9 * h
  let new_base := 1.2 * b
  let original_area := (b * h) / 2
  let new_area := (new_base * new_height) / 2
  new_area = 1.08 * original_area := by
sorry

end triangle_area_change_l3931_393132


namespace power_expression_evaluation_l3931_393121

theorem power_expression_evaluation (b : ℕ) (h : b = 4) : b^3 * b^6 / b^2 = 16384 := by
  sorry

end power_expression_evaluation_l3931_393121


namespace tim_children_treats_l3931_393150

/-- Calculates the total number of treats Tim's children get while trick-or-treating --/
def total_treats (num_children : ℕ) 
                 (houses_hour1 houses_hour2 houses_hour3 houses_hour4 : ℕ)
                 (treats_hour1 treats_hour2 treats_hour3 treats_hour4 : ℕ) : ℕ :=
  (houses_hour1 * treats_hour1 * num_children) +
  (houses_hour2 * treats_hour2 * num_children) +
  (houses_hour3 * treats_hour3 * num_children) +
  (houses_hour4 * treats_hour4 * num_children)

/-- Theorem stating that Tim's children get 237 treats in total --/
theorem tim_children_treats : 
  total_treats 3 4 6 5 7 3 4 3 4 = 237 := by
  sorry


end tim_children_treats_l3931_393150


namespace rose_count_l3931_393115

theorem rose_count : ∃ (n : ℕ), 
  300 ≤ n ∧ n ≤ 400 ∧ 
  ∃ (x y : ℕ), n = 21 * x + 13 ∧ n = 15 * y - 8 ∧
  n = 307 := by
  sorry

end rose_count_l3931_393115
