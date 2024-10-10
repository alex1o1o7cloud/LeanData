import Mathlib

namespace remaining_cube_volume_l540_54034

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) : 
  cube_side = 5 → cylinder_radius = 1.5 → 
  cube_side^3 - π * cylinder_radius^2 * cube_side = 125 - 11.25 * π := by
  sorry

#check remaining_cube_volume

end remaining_cube_volume_l540_54034


namespace matrix_determinant_l540_54080

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; -3, 6]

theorem matrix_determinant :
  Matrix.det matrix = 36 := by sorry

end matrix_determinant_l540_54080


namespace geometric_sequence_ratio_sum_l540_54086

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : p ≠ 1) (h2 : r ≠ 1) (h3 : p ≠ r) 
  (h4 : k ≠ 0) (h5 : k * p^2 - k * r^2 = 3 * (k * p - k * r)) : p + r = 3 := by
  sorry

end geometric_sequence_ratio_sum_l540_54086


namespace ratio_problem_l540_54096

theorem ratio_problem (a b c d e f : ℚ) 
  (h1 : a * b * c / (d * e * f) = 1.875)
  (h2 : a / b = 5 / 2)
  (h3 : b / c = 1 / 2)
  (h4 : c / d = 1)
  (h5 : d / e = 3 / 2) :
  e / f = 1 / 3 := by
sorry

end ratio_problem_l540_54096


namespace inequality_proof_l540_54098

theorem inequality_proof (m n : ℝ) (h1 : m > n) (h2 : n > 0) : 
  m * Real.exp n + n < n * Real.exp m + m := by
  sorry

end inequality_proof_l540_54098


namespace sqrt_54_minus_sqrt_6_l540_54047

theorem sqrt_54_minus_sqrt_6 : Real.sqrt 54 - Real.sqrt 6 = 2 * Real.sqrt 6 := by
  sorry

end sqrt_54_minus_sqrt_6_l540_54047


namespace optimal_strategy_with_bicycle_l540_54083

/-- The optimal strategy for two people to reach a destination with one bicycle. -/
theorem optimal_strategy_with_bicycle 
  (total_distance : ℝ) 
  (walking_speed : ℝ) 
  (cycling_speed : ℝ) 
  (ha : total_distance > 0) 
  (hw : walking_speed > 0) 
  (hc : cycling_speed > walking_speed) :
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < total_distance ∧ 
    (x / walking_speed + (total_distance - x) / cycling_speed = 
     x / walking_speed + (total_distance - x) / walking_speed) ∧
    ∀ (y : ℝ), 
      0 < y → 
      y < total_distance → 
      (y / walking_speed + (total_distance - y) / cycling_speed ≥
       x / walking_speed + (total_distance - x) / cycling_speed) :=
by sorry

end optimal_strategy_with_bicycle_l540_54083


namespace collisions_100_balls_l540_54079

/-- The number of collisions between n identical balls moving along a single dimension,
    where each pair of balls can collide exactly once. -/
def numCollisions (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that for 100 balls, the number of collisions is 4950. -/
theorem collisions_100_balls :
  numCollisions 100 = 4950 := by
  sorry

end collisions_100_balls_l540_54079


namespace largest_angle_in_triangle_l540_54069

theorem largest_angle_in_triangle : ∀ x : ℝ,
  x + 35 + 70 = 180 →
  max x (max 35 70) = 75 :=
by
  sorry

end largest_angle_in_triangle_l540_54069


namespace sqrt_equation_solution_l540_54017

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (x + 12) = 10) ∧ (x = 88) := by
sorry

end sqrt_equation_solution_l540_54017


namespace annual_lesson_cost_difference_l540_54032

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The hourly rate for clarinet lessons in dollars -/
def clarinet_rate : ℕ := 40

/-- The number of hours per week of clarinet lessons -/
def clarinet_hours : ℕ := 3

/-- The hourly rate for piano lessons in dollars -/
def piano_rate : ℕ := 28

/-- The number of hours per week of piano lessons -/
def piano_hours : ℕ := 5

/-- The difference in annual spending between piano and clarinet lessons -/
theorem annual_lesson_cost_difference :
  (piano_rate * piano_hours - clarinet_rate * clarinet_hours) * weeks_per_year = 1040 := by
  sorry

end annual_lesson_cost_difference_l540_54032


namespace simple_interest_rate_example_l540_54073

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

theorem simple_interest_rate_example :
  simple_interest_rate 750 1050 5 = 8 := by
  sorry

end simple_interest_rate_example_l540_54073


namespace fraction_zero_implies_x_negative_two_l540_54036

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  (|x| - 2) / (x - 2) = 0 → x = -2 := by
  sorry

end fraction_zero_implies_x_negative_two_l540_54036


namespace circle_cover_theorem_l540_54059

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is inside or on the boundary of a circle -/
def Point.insideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- Check if a set of points can be covered by a circle -/
def coveredBy (points : Set Point) (c : Circle) : Prop :=
  ∀ p ∈ points, p.insideCircle c

/-- Main theorem -/
theorem circle_cover_theorem (n : ℕ) (points : Set Point) 
  (h : ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
    ∃ (c : Circle), c.radius = 1 ∧ coveredBy {p1, p2, p3} c) :
  ∃ (c : Circle), c.radius = 1 ∧ coveredBy points c := by
  sorry

end circle_cover_theorem_l540_54059


namespace min_sum_of_reciprocal_squares_l540_54025

theorem min_sum_of_reciprocal_squares (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 1) : 
  27 ≤ (1/a^2 + 1/b^2 + 1/c^2) := by
sorry

end min_sum_of_reciprocal_squares_l540_54025


namespace am_gm_positive_condition_l540_54075

theorem am_gm_positive_condition (a b : ℝ) (h : a * b ≠ 0) :
  (a > 0 ∧ b > 0) ↔ ((a + b) / 2 ≥ Real.sqrt (a * b)) :=
sorry

end am_gm_positive_condition_l540_54075


namespace sock_selection_theorem_l540_54039

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of socks in the drawer -/
def total_socks : ℕ := 8

/-- The number of socks to be chosen -/
def socks_to_choose : ℕ := 4

/-- The number of non-red socks -/
def non_red_socks : ℕ := 7

theorem sock_selection_theorem :
  choose total_socks socks_to_choose - choose non_red_socks socks_to_choose = 35 := by sorry

end sock_selection_theorem_l540_54039


namespace diophantine_equation_solution_l540_54050

def solution_set : Set (ℕ × ℕ) :=
  {(5, 20), (6, 12), (8, 8), (12, 6), (20, 5)}

theorem diophantine_equation_solution :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
    (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 4 ↔ (x, y) ∈ solution_set :=
by sorry

end diophantine_equation_solution_l540_54050


namespace units_digit_of_7_pow_6_pow_5_l540_54046

-- Define the cycle of units digits for powers of 7
def units_cycle : List Nat := [7, 9, 3, 1]

-- Theorem statement
theorem units_digit_of_7_pow_6_pow_5 : 
  ∃ (n : Nat), 7^(6^5) ≡ 7 [ZMOD 10] ∧ n = 7^(6^5) % 10 := by
  sorry

end units_digit_of_7_pow_6_pow_5_l540_54046


namespace andrews_yearly_donation_l540_54055

/-- Calculates the yearly donation amount given the starting age, current age, and total donation --/
def yearly_donation (start_age : ℕ) (current_age : ℕ) (total_donation : ℕ) : ℚ :=
  (total_donation : ℚ) / ((current_age - start_age) : ℚ)

/-- Theorem stating that Andrew's yearly donation is approximately 7388.89 --/
theorem andrews_yearly_donation :
  let start_age : ℕ := 11
  let current_age : ℕ := 29
  let total_donation : ℕ := 133000
  abs (yearly_donation start_age current_age total_donation - 7388.89) < 0.01 := by
  sorry

end andrews_yearly_donation_l540_54055


namespace email_sending_ways_l540_54097

theorem email_sending_ways (email_addresses : ℕ) (emails_to_send : ℕ) : 
  email_addresses = 3 → emails_to_send = 5 → email_addresses ^ emails_to_send = 243 := by
  sorry

end email_sending_ways_l540_54097


namespace crayon_division_l540_54061

theorem crayon_division (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  total_crayons = num_people * crayons_per_person →
  crayons_per_person = 8 :=
by sorry

end crayon_division_l540_54061


namespace barry_head_stand_theorem_l540_54089

/-- The number of turns Barry can take standing on his head during a 2-hour period -/
def barry_head_stand_turns : ℕ :=
  let head_stand_time : ℕ := 10  -- minutes
  let sit_time : ℕ := 5  -- minutes
  let total_period : ℕ := 2 * 60  -- 2 hours in minutes
  let time_per_turn : ℕ := head_stand_time + sit_time
  total_period / time_per_turn

theorem barry_head_stand_theorem :
  barry_head_stand_turns = 8 := by
  sorry

end barry_head_stand_theorem_l540_54089


namespace no_solution_inequality_l540_54074

theorem no_solution_inequality (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 5| > m) ↔ m < 6 := by sorry

end no_solution_inequality_l540_54074


namespace hyperbola_m_value_l540_54072

/-- A hyperbola with equation mx^2 + y^2 = 1 where the length of its imaginary axis
    is twice the length of its real axis -/
structure Hyperbola (m : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + y^2 = 1
  axis_ratio : (imaginary_axis_length : ℝ) = 2 * (real_axis_length : ℝ)

/-- The value of m for a hyperbola with the given properties is -1/4 -/
theorem hyperbola_m_value (h : Hyperbola m) : m = -1/4 := by
  sorry

end hyperbola_m_value_l540_54072


namespace ruby_apples_l540_54090

/-- The number of apples Ruby has initially -/
def initial_apples : ℕ := 63

/-- The number of apples Emily takes away -/
def apples_taken : ℕ := 55

/-- The number of apples Ruby has after Emily takes some away -/
def remaining_apples : ℕ := initial_apples - apples_taken

theorem ruby_apples : remaining_apples = 8 := by
  sorry

end ruby_apples_l540_54090


namespace shift_graph_l540_54099

-- Define a function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem shift_graph (h : f 0 = 1) : f ((-1) + 1) = 1 := by
  sorry

end shift_graph_l540_54099


namespace smithtown_left_handed_women_percentage_l540_54048

/-- Represents the population of Smithtown -/
structure Population where
  right_handed : ℕ  -- Number of right-handed people
  left_handed : ℕ   -- Number of left-handed people
  men : ℕ           -- Number of men
  women : ℕ         -- Number of women

/-- The conditions of the Smithtown population problem -/
def smithtown_conditions (p : Population) : Prop :=
  -- Ratio of right-handed to left-handed is 3:1
  p.right_handed = 3 * p.left_handed ∧
  -- Ratio of men to women is 3:2
  3 * p.women = 2 * p.men ∧
  -- Number of right-handed men is maximized (all men are right-handed)
  p.men = p.right_handed

/-- The theorem stating that 25% of the population are left-handed women -/
theorem smithtown_left_handed_women_percentage (p : Population)
  (h : smithtown_conditions p) :
  (p.left_handed : ℚ) / (p.right_handed + p.left_handed : ℚ) = 1/4 := by
  sorry

#check smithtown_left_handed_women_percentage

end smithtown_left_handed_women_percentage_l540_54048


namespace hyperbola_eccentricity_range_l540_54063

/-- Given a hyperbola E and a parabola C with specific properties, 
    prove that the eccentricity of E is in the range (1, 3√2/4] -/
theorem hyperbola_eccentricity_range 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (E : Set (ℝ × ℝ)) 
  (C : Set (ℝ × ℝ)) 
  (A : ℝ × ℝ) 
  (F : ℝ × ℝ) 
  (P : ℝ × ℝ) 
  (hE : E = {(x, y) | x^2/a^2 - y^2/b^2 = 1})
  (hC : C = {(x, y) | y^2 = 8*a*x})
  (hA : A = (a, 0))
  (hF : F = (2*a, 0))
  (hP : P ∈ {(x, y) | y = (b/a)*x})  -- P is on the asymptote of E
  (hPerp : (P.1 - A.1) * (P.1 - F.1) + (P.2 - A.2) * (P.2 - F.2) = 0)  -- AP ⊥ FP
  : 1 < Real.sqrt (a^2 + b^2) / a ∧ Real.sqrt (a^2 + b^2) / a ≤ 3 * Real.sqrt 2 / 4 :=
sorry

end hyperbola_eccentricity_range_l540_54063


namespace election_votes_calculation_l540_54002

theorem election_votes_calculation (total_votes : ℕ) : 
  (75 : ℚ) / 100 * ((100 : ℚ) - 15) / 100 * total_votes = 357000 → 
  total_votes = 560000 := by
sorry

end election_votes_calculation_l540_54002


namespace film_festival_theorem_l540_54016

theorem film_festival_theorem (n : ℕ) (m : ℕ) : 
  -- Total number of films
  n > 0 →
  -- Total number of viewers (2m, where m is the number of men/women)
  m > 0 →
  -- Each film is liked by exactly 8 viewers
  -- Each viewer likes the same number of films
  -- The total number of "likes" is 8n
  8 * n = 2 * m * (8 * n / (2 * m)) →
  -- At least 3/7 of the films are liked by at least two men
  ∃ (k : ℕ), k ≥ (3 * n + 6) / 7 ∧ 
    (∀ (i : ℕ), i < k → ∃ (male_viewers : ℕ), male_viewers ≥ 2 ∧ male_viewers ≤ 8) :=
by
  sorry

end film_festival_theorem_l540_54016


namespace quadratic_equation_coefficients_l540_54066

/-- Given a quadratic equation ax² + bx + c = 0, returns the coefficient of x² (a) -/
def quadratic_coefficient (a b c : ℚ) : ℚ := a

/-- Given a quadratic equation ax² + bx + c = 0, returns the constant term (c) -/
def constant_term (a b c : ℚ) : ℚ := c

theorem quadratic_equation_coefficients :
  let a : ℚ := 3
  let b : ℚ := -6
  let c : ℚ := -7
  quadratic_coefficient a b c = 3 ∧ constant_term a b c = -7 := by
  sorry

end quadratic_equation_coefficients_l540_54066


namespace shortest_path_on_cube_is_four_l540_54082

/-- The shortest path on the surface of a regular cube with edge length 2,
    from one corner to the opposite corner. -/
def shortest_path_on_cube : ℝ := 4

/-- Proof that the shortest path on the surface of a regular cube with edge length 2,
    from one corner to the opposite corner, is equal to 4. -/
theorem shortest_path_on_cube_is_four :
  shortest_path_on_cube = 4 := by sorry

end shortest_path_on_cube_is_four_l540_54082


namespace parabola_point_y_coordinate_l540_54029

/-- The y-coordinate of a point on a parabola at a given distance from the focus -/
theorem parabola_point_y_coordinate (x y : ℝ) :
  y = -4 * x^2 →  -- Point M is on the parabola y = -4x²
  (x^2 + (y - 1/4)^2) = 1 →  -- Distance from M to focus (0, 1/4) is 1
  y = -15/16 := by
sorry

end parabola_point_y_coordinate_l540_54029


namespace sqrt_x_minus_one_squared_l540_54040

theorem sqrt_x_minus_one_squared (x : ℝ) (h : |1 - x| = 1 + |x|) : 
  Real.sqrt ((x - 1)^2) = 1 - x :=
by sorry

end sqrt_x_minus_one_squared_l540_54040


namespace rob_baseball_cards_l540_54057

theorem rob_baseball_cards (rob_total : ℕ) (rob_doubles : ℕ) (jess_doubles : ℕ) :
  rob_doubles = rob_total / 3 →
  jess_doubles = 5 * rob_doubles →
  jess_doubles = 40 →
  rob_total = 24 := by
sorry

end rob_baseball_cards_l540_54057


namespace frank_unfilled_boxes_l540_54018

/-- Given a total number of boxes and a number of filled boxes,
    calculate the number of unfilled boxes. -/
def unfilled_boxes (total : ℕ) (filled : ℕ) : ℕ :=
  total - filled

/-- Theorem: Frank has 5 unfilled boxes -/
theorem frank_unfilled_boxes :
  unfilled_boxes 13 8 = 5 := by
  sorry

end frank_unfilled_boxes_l540_54018


namespace complex_equation_solution_l540_54065

theorem complex_equation_solution (a b : ℝ) (h : (Complex.I + a) * (1 + Complex.I) = b * Complex.I) : 
  Complex.mk a b = 1 + 2 * Complex.I := by
  sorry

end complex_equation_solution_l540_54065


namespace gear_speed_ratio_l540_54088

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Proves that for three interconnected gears, if the product of teeth and speed is equal,
    then the ratio of their speeds is proportional to the product of the other two gears' teeth -/
theorem gear_speed_ratio
  (G H I : Gear)
  (h : G.teeth * G.speed = H.teeth * H.speed ∧ H.teeth * H.speed = I.teeth * I.speed) :
  ∃ (k : ℝ), G.speed = k * (H.teeth * I.teeth) ∧
             H.speed = k * (G.teeth * I.teeth) ∧
             I.speed = k * (G.teeth * H.teeth) := by
  sorry

end gear_speed_ratio_l540_54088


namespace triangle_properties_l540_54094

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the triangle with specific properties -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.c - t.b) / t.a = Real.cos t.B / Real.cos t.A) 
  (h2 : t.a = 2 * Real.sqrt 5) : 
  t.A = π / 3 ∧ 
  (∃ (S : ℝ), S = 5 * Real.sqrt 3 ∧ ∀ (area : ℝ), area ≤ S) := by
  sorry


end triangle_properties_l540_54094


namespace simplify_expression_value_under_condition_independence_condition_l540_54041

/-- Given algebraic expressions A and B -/
def A (m y : ℝ) : ℝ := 2 * m^2 + 3 * m * y + 2 * y - 1

def B (m y : ℝ) : ℝ := m^2 - m * y

/-- Theorem 1: Simplification of 3A - 2(A + B) -/
theorem simplify_expression (m y : ℝ) :
  3 * A m y - 2 * (A m y + B m y) = 5 * m * y + 2 * y - 1 := by sorry

/-- Theorem 2: Value of 3A - 2(A + B) under specific condition -/
theorem value_under_condition (m y : ℝ) :
  (m - 1)^2 + |y + 2| = 0 →
  3 * A m y - 2 * (A m y + B m y) = -15 := by sorry

/-- Theorem 3: Condition for 3A - 2(A + B) to be independent of y -/
theorem independence_condition (m : ℝ) :
  (∀ y : ℝ, 3 * A m y - 2 * (A m y + B m y) = 5 * m * y + 2 * y - 1) →
  m = -2/5 := by sorry

end simplify_expression_value_under_condition_independence_condition_l540_54041


namespace parallel_line_slope_l540_54071

/-- Given a line with equation 2x - 4y = 9, prove that any parallel line has slope 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (2 * x - 4 * y = 9) → (slope_of_parallel_line : ℝ) = 1 / 2 := by
  sorry

end parallel_line_slope_l540_54071


namespace tangent_slope_at_point_one_l540_54052

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 4*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 4

-- Theorem statement
theorem tangent_slope_at_point_one :
  f 1 = -3 ∧ f' 1 = -1 :=
by sorry

end tangent_slope_at_point_one_l540_54052


namespace variance_of_data_l540_54085

def data : List ℝ := [3, 2, 1, 0, 0, 0, 1]

theorem variance_of_data : 
  let n : ℝ := data.length
  let mean := (data.sum) / n
  let variance := (data.map (fun x => (x - mean)^2)).sum / n
  variance = 8/7 := by sorry

end variance_of_data_l540_54085


namespace cadence_earnings_increase_l540_54093

/-- Proves that the percentage increase in Cadence's monthly earnings at her new company
    compared to her old company is 20%, given the specified conditions. -/
theorem cadence_earnings_increase (
  old_company_duration : ℕ := 3 * 12
  ) (old_company_monthly_salary : ℕ := 5000)
  (new_company_duration_increase : ℕ := 5)
  (total_earnings : ℕ := 426000) : Real :=
  by
  -- Define the duration of employment at the new company
  let new_company_duration : ℕ := old_company_duration + new_company_duration_increase

  -- Calculate total earnings from the old company
  let old_company_total : ℕ := old_company_duration * old_company_monthly_salary

  -- Calculate total earnings from the new company
  let new_company_total : ℕ := total_earnings - old_company_total

  -- Calculate monthly salary at the new company
  let new_company_monthly_salary : ℕ := new_company_total / new_company_duration

  -- Calculate the percentage increase
  let percentage_increase : Real := 
    (new_company_monthly_salary - old_company_monthly_salary : Real) / old_company_monthly_salary * 100

  -- Prove that the percentage increase is 20%
  sorry


end cadence_earnings_increase_l540_54093


namespace height_difference_pablo_charlene_l540_54077

/-- Given the heights of various people, prove the height difference between Pablo and Charlene. -/
theorem height_difference_pablo_charlene :
  ∀ (height_janet height_ruby height_pablo height_charlene : ℕ),
  height_janet = 62 →
  height_charlene = 2 * height_janet →
  height_ruby = 192 →
  height_pablo = height_ruby + 2 →
  height_pablo - height_charlene = 70 := by
  sorry

end height_difference_pablo_charlene_l540_54077


namespace grid_sum_l540_54037

theorem grid_sum (X Y Z : ℝ) 
  (row1_sum : 1 + X + 3 = 9)
  (row2_sum : 2 + Y + Z = 9) :
  X + Y + Z = 12 := by sorry

end grid_sum_l540_54037


namespace contrapositive_equivalence_l540_54006

/-- The equation x^2 + x - m = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 + x - m = 0

/-- The contrapositive of "If m > 0, then the equation x^2 + x - m = 0 has real roots" 
    is equivalent to "If the equation x^2 + x - m = 0 does not have real roots, then m ≤ 0" -/
theorem contrapositive_equivalence : 
  (¬(has_real_roots m) → m ≤ 0) ↔ (m > 0 → has_real_roots m) :=
sorry

end contrapositive_equivalence_l540_54006


namespace yellow_balls_count_l540_54053

/-- The number of yellow balls in a bag, given the total number of balls,
    the number of balls of each color (except yellow), and the probability
    of choosing a ball that is neither red nor purple. -/
def yellow_balls (total : ℕ) (white green red purple : ℕ) (prob_not_red_purple : ℚ) : ℕ :=
  total - white - green - red - purple

/-- Theorem stating that the number of yellow balls is 5 under the given conditions. -/
theorem yellow_balls_count :
  yellow_balls 60 22 18 6 9 (3/4) = 5 := by
  sorry

end yellow_balls_count_l540_54053


namespace second_month_interest_l540_54003

/-- Calculates the interest charged in the second month for a loan with monthly compound interest. -/
theorem second_month_interest
  (initial_loan : ℝ)
  (monthly_interest_rate : ℝ)
  (h1 : initial_loan = 200)
  (h2 : monthly_interest_rate = 0.1) :
  let first_month_total := initial_loan * (1 + monthly_interest_rate)
  let second_month_interest := first_month_total * monthly_interest_rate
  second_month_interest = 22 := by
sorry

end second_month_interest_l540_54003


namespace f_property_f_upper_bound_minimum_M_l540_54020

/-- The function f(x) = x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The derivative of f(x) -/
def f_derivative (b : ℝ) (x : ℝ) : ℝ := 2*x + b

theorem f_property (b c : ℝ) :
  ∀ x, f_derivative b x ≤ f b c x := sorry

theorem f_upper_bound (b c : ℝ) (h : ∀ x, f_derivative b x ≤ f b c x) :
  ∀ x ≥ 0, f b c x ≤ (x + c)^2 := sorry

theorem minimum_M (b c : ℝ) (h : ∀ x, f_derivative b x ≤ f b c x) :
  ∃ M, (∀ b c, f b c c - f b c b ≤ M * (c^2 - b^2)) ∧
       (∀ M', (∀ b c, f b c c - f b c b ≤ M' * (c^2 - b^2)) → M ≤ M') ∧
       M = 3/2 := sorry

end f_property_f_upper_bound_minimum_M_l540_54020


namespace union_of_sets_l540_54051

theorem union_of_sets : 
  let A : Set ℤ := {1, 3, 5, 6}
  let B : Set ℤ := {-1, 5, 7}
  A ∪ B = {-1, 1, 3, 5, 6, 7} := by
sorry

end union_of_sets_l540_54051


namespace shoes_polished_percentage_l540_54009

def shoes_polished (pairs : ℕ) (left_to_polish : ℕ) : ℚ :=
  let total_shoes := 2 * pairs
  let polished := total_shoes - left_to_polish
  (polished : ℚ) / (total_shoes : ℚ) * 100

theorem shoes_polished_percentage :
  shoes_polished 10 11 = 45 := by
  sorry

end shoes_polished_percentage_l540_54009


namespace shooting_probabilities_shooting_probabilities_alt_l540_54070

-- Define the probabilities given in the problem
def p_9_or_more : ℝ := 0.56
def p_8 : ℝ := 0.22
def p_7 : ℝ := 0.12

-- Theorem statement
theorem shooting_probabilities :
  let p_less_than_8 := 1 - p_9_or_more - p_8
  let p_at_least_7 := p_7 + p_8 + p_9_or_more
  (p_less_than_8 = 0.22) ∧ (p_at_least_7 = 0.9) := by
  sorry

-- Alternative formulation using complement for p_at_least_7
theorem shooting_probabilities_alt :
  let p_less_than_7 := 1 - p_9_or_more - p_8 - p_7
  let p_less_than_8 := p_less_than_7 + p_7
  let p_at_least_7 := 1 - p_less_than_7
  (p_less_than_8 = 0.22) ∧ (p_at_least_7 = 0.9) := by
  sorry

end shooting_probabilities_shooting_probabilities_alt_l540_54070


namespace unique_solution_condition_l540_54000

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = (b - 1) * x + 2) ↔ b ≠ 5 := by
  sorry

end unique_solution_condition_l540_54000


namespace daves_sticks_l540_54015

theorem daves_sticks (sticks_picked : ℕ) (sticks_left : ℕ) : 
  sticks_left = 4 → 
  sticks_picked - sticks_left = 10 → 
  sticks_picked = 14 := by
  sorry

end daves_sticks_l540_54015


namespace min_blue_chips_l540_54056

theorem min_blue_chips (r w b : ℕ) : 
  r ≥ 2 * w →
  r ≤ 2 * b / 3 →
  r + w ≥ 72 →
  ∀ b' : ℕ, (∃ r' w' : ℕ, r' ≥ 2 * w' ∧ r' ≤ 2 * b' / 3 ∧ r' + w' ≥ 72) → b' ≥ 72 :=
by sorry

end min_blue_chips_l540_54056


namespace product_of_distinct_roots_l540_54043

theorem product_of_distinct_roots (x y : ℝ) : 
  x ≠ 0 → y ≠ 0 → x ≠ y → (x + 6 / x = y + 6 / y) → x * y = 6 := by
  sorry

end product_of_distinct_roots_l540_54043


namespace plane_P_satisfies_conditions_l540_54092

def plane1 (x y z : ℝ) : ℝ := 2*x - y + 2*z - 4
def plane2 (x y z : ℝ) : ℝ := 3*x + 4*y - z - 6
def planeP (x y z : ℝ) : ℝ := x + 63*y - 35*z - 34

def point : ℝ × ℝ × ℝ := (4, -2, 2)

theorem plane_P_satisfies_conditions :
  (∀ x y z, plane1 x y z = 0 ∧ plane2 x y z = 0 → planeP x y z = 0) ∧
  (planeP ≠ plane1 ∧ planeP ≠ plane2) ∧
  (abs (planeP point.1 point.2.1 point.2.2) / 
   Real.sqrt ((1:ℝ)^2 + 63^2 + (-35)^2) = 3 / Real.sqrt 6) :=
by sorry

end plane_P_satisfies_conditions_l540_54092


namespace rhombus_diagonal_l540_54068

theorem rhombus_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) : 
  d1 = 30 → area = 180 → area = (d1 * d2) / 2 → d2 = 12 := by
  sorry

end rhombus_diagonal_l540_54068


namespace wheels_per_row_calculation_l540_54027

/-- Calculates the number of wheels per row given the total number of wheels,
    number of trains, carriages per train, and rows of wheels per carriage. -/
def wheels_per_row (total_wheels : ℕ) (num_trains : ℕ) (carriages_per_train : ℕ) (rows_per_carriage : ℕ) : ℕ :=
  total_wheels / (num_trains * carriages_per_train * rows_per_carriage)

/-- Theorem stating that given 4 trains, 4 carriages per train, 3 rows of wheels per carriage,
    and 240 wheels in total, the number of wheels in each row is 5. -/
theorem wheels_per_row_calculation :
  wheels_per_row 240 4 4 3 = 5 := by
  sorry

end wheels_per_row_calculation_l540_54027


namespace empty_can_weight_l540_54005

/-- Given a can that weighs 34 kg when full of milk and 17.5 kg when half-full, 
    prove that the empty can weighs 1 kg. -/
theorem empty_can_weight (full_weight half_weight : ℝ) 
  (h_full : full_weight = 34)
  (h_half : half_weight = 17.5) : 
  ∃ (empty_weight milk_weight : ℝ),
    empty_weight + milk_weight = full_weight ∧
    empty_weight + milk_weight / 2 = half_weight ∧
    empty_weight = 1 := by
  sorry

end empty_can_weight_l540_54005


namespace least_possible_difference_l540_54054

theorem least_possible_difference (x y z : ℤ) 
  (h_x_even : Even x)
  (h_y_odd : Odd y)
  (h_z_odd : Odd z)
  (h_order : x < y ∧ y < z)
  (h_diff : y - x > 5) :
  ∀ w, w = z - x → w ≥ 9 ∧ ∃ (a b c : ℤ), a - b = 9 ∧ Even b ∧ Odd a ∧ Odd c ∧ b < c ∧ c < a ∧ c - b > 5 := by
  sorry

end least_possible_difference_l540_54054


namespace subset_implies_a_values_l540_54026

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem subset_implies_a_values :
  ∀ a : ℝ, (B a ⊆ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) := by
  sorry

end subset_implies_a_values_l540_54026


namespace f_properties_l540_54019

noncomputable def f (x a : ℝ) := 2 * (Real.cos x)^2 + Real.sin (2 * x) + a

theorem f_properties (a : ℝ) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f x a = f (x + T) a ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f x a = f (x + T') a) → T ≤ T') ∧
  (∀ (k : ℤ), ∀ (x : ℝ), x ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) →
    ∀ (y : ℝ), y ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) →
      x ≤ y → f x a ≤ f y a) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 6) → f x a ≤ 2) →
  a = 1 - Real.sqrt 2 ∧
  ∀ (k : ℤ), ∀ (x : ℝ), f x a = f (k * Real.pi + Real.pi / 4 - x) a :=
sorry

end f_properties_l540_54019


namespace painted_cube_equality_l540_54013

/-- Represents a cube with edge length n and two opposite faces painted. -/
structure PaintedCube where
  n : ℕ
  h_n_gt_3 : n > 3

/-- The number of unit cubes with exactly one face painted black. -/
def one_face_painted (cube : PaintedCube) : ℕ :=
  2 * (cube.n - 2)^2

/-- The number of unit cubes with exactly two faces painted black. -/
def two_faces_painted (cube : PaintedCube) : ℕ :=
  4 * (cube.n - 2)

/-- Theorem stating that the number of unit cubes with one face painted
    equals the number of unit cubes with two faces painted iff n = 4. -/
theorem painted_cube_equality (cube : PaintedCube) :
  one_face_painted cube = two_faces_painted cube ↔ cube.n = 4 := by
  sorry

end painted_cube_equality_l540_54013


namespace stating_ladder_of_twos_theorem_l540_54030

/-- 
A function that represents the number of distinct integers obtainable 
from a ladder of n twos by placing nested parentheses.
-/
def ladder_of_twos (n : ℕ) : ℕ :=
  if n ≥ 3 then 2^(n-3) else 0

/-- 
Theorem stating that for n ≥ 3, the number of distinct integers obtainable 
from a ladder of n twos by placing nested parentheses is 2^(n-3).
-/
theorem ladder_of_twos_theorem (n : ℕ) (h : n ≥ 3) : 
  ladder_of_twos n = 2^(n-3) := by
  sorry

end stating_ladder_of_twos_theorem_l540_54030


namespace thirty_blocks_differ_in_two_ways_l540_54058

/-- Represents the number of options for each property of a block -/
structure BlockOptions :=
  (material : Nat)
  (size : Nat)
  (color : Nat)
  (shape : Nat)

/-- Calculates the number of blocks that differ in exactly k ways from a specific block -/
def blocksWithKDifferences (options : BlockOptions) (k : Nat) : Nat :=
  sorry

/-- The specific block options for our problem -/
def ourOptions : BlockOptions :=
  { material := 2, size := 4, color := 4, shape := 4 }

/-- The main theorem stating that 30 blocks differ in exactly 2 ways -/
theorem thirty_blocks_differ_in_two_ways :
  blocksWithKDifferences ourOptions 2 = 30 := by sorry

end thirty_blocks_differ_in_two_ways_l540_54058


namespace smallest_number_proof_l540_54024

theorem smallest_number_proof (a b c d : ℝ) : 
  b = 4 * a →
  c = 2 * a →
  d = a + b + c →
  (a + b + c + d) / 4 = 77 →
  a = 22 := by
sorry

end smallest_number_proof_l540_54024


namespace quadratic_roots_properties_l540_54044

theorem quadratic_roots_properties (p : ℝ) (x₁ x₂ : ℂ) :
  x₁^2 + p * x₁ + 2 = 0 →
  x₂^2 + p * x₂ + 2 = 0 →
  x₁ = 1 + I →
  (x₂ = 1 - I ∧ x₁ / x₂ = I) := by sorry

end quadratic_roots_properties_l540_54044


namespace min_value_expression_min_value_achievable_l540_54001

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((2*x^2 + 5*x + 2)*(2*y^2 + 5*y + 2)*(2*z^2 + 5*z + 2)) / (x*y*z*(1+x)*(1+y)*(1+z)) ≥ 729/8 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  ((2*x^2 + 5*x + 2)*(2*y^2 + 5*y + 2)*(2*z^2 + 5*z + 2)) / (x*y*z*(1+x)*(1+y)*(1+z)) = 729/8 :=
by sorry

end min_value_expression_min_value_achievable_l540_54001


namespace arithmetic_sequence_tenth_term_l540_54042

theorem arithmetic_sequence_tenth_term :
  let a : ℚ := 1/4  -- First term
  let d : ℚ := 1/2  -- Common difference
  let n : ℕ := 10   -- Term number we're looking for
  let a_n : ℚ := a + (n - 1) * d  -- Formula for nth term of arithmetic sequence
  a_n = 19/4 := by sorry

end arithmetic_sequence_tenth_term_l540_54042


namespace coeff_bound_theorem_l540_54035

/-- Represents a real polynomial -/
def RealPolynomial := ℝ → ℝ

/-- The degree of a polynomial -/
def degree (p : RealPolynomial) : ℕ := sorry

/-- The largest absolute value of the coefficients of a polynomial -/
def max_coeff (p : RealPolynomial) : ℝ := sorry

/-- Multiplication of polynomials -/
def poly_mul (p q : RealPolynomial) : RealPolynomial := sorry

/-- Addition of a constant to x -/
def add_const (a : ℝ) : RealPolynomial := sorry

theorem coeff_bound_theorem (p q : RealPolynomial) (a : ℝ) (n : ℕ) (h k : ℝ) :
  p = poly_mul (add_const a) q →
  degree p = n →
  max_coeff p = h →
  max_coeff q = k →
  k ≤ h * n := by sorry

end coeff_bound_theorem_l540_54035


namespace tree_height_difference_l540_54004

/-- The height difference between two trees -/
theorem tree_height_difference (maple_height spruce_height : ℚ) 
  (h_maple : maple_height = 10 + 1/4)
  (h_spruce : spruce_height = 14 + 1/2) :
  spruce_height - maple_height = 19 + 3/4 := by
  sorry

end tree_height_difference_l540_54004


namespace annulus_area_single_element_l540_54087

/-- The area of an annulus can be expressed using only one linear element -/
theorem annulus_area_single_element (R r : ℝ) (h : R > r) :
  ∃ (d : ℝ), (d = R - r ∨ d = R + r) ∧
  (π * (R^2 - r^2) = π * d * (2*R - d) ∨ π * (R^2 - r^2) = π * d * (2*r + d)) := by
  sorry

end annulus_area_single_element_l540_54087


namespace computer_price_l540_54033

theorem computer_price (new_price : ℝ) (price_increase : ℝ) (double_original : ℝ) 
  (h1 : price_increase = 0.3)
  (h2 : new_price = 377)
  (h3 : double_original = 580) : 
  ∃ (original_price : ℝ), 
    original_price * (1 + price_increase) = new_price ∧ 
    2 * original_price = double_original ∧
    original_price = 290 := by
  sorry

end computer_price_l540_54033


namespace range_of_a_l540_54010

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 10 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a + 3}

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (((A ∩ B) ∩ C a) = C a) ↔ 1 ≤ a := by sorry

end range_of_a_l540_54010


namespace cookfire_logs_proof_l540_54049

/-- The number of logs burned per hour -/
def logs_burned_per_hour : ℕ := 3

/-- The number of logs added at the end of each hour -/
def logs_added_per_hour : ℕ := 2

/-- The number of hours the cookfire burns -/
def burn_duration : ℕ := 3

/-- The number of logs left after the burn duration -/
def logs_remaining : ℕ := 3

/-- The initial number of logs in the cookfire -/
def initial_logs : ℕ := 6

theorem cookfire_logs_proof :
  initial_logs - burn_duration * logs_burned_per_hour + (burn_duration - 1) * logs_added_per_hour = logs_remaining :=
by
  sorry

end cookfire_logs_proof_l540_54049


namespace factorization_problems_l540_54014

theorem factorization_problems :
  (∀ x y : ℝ, 2*x^2*y - 8*x*y + 8*y = 2*y*(x-2)^2) ∧
  (∀ a : ℝ, 18*a^2 - 50 = 2*(3*a+5)*(3*a-5)) := by
  sorry

end factorization_problems_l540_54014


namespace power_tower_mod_2000_l540_54095

theorem power_tower_mod_2000 : 7^(7^(7^7)) ≡ 343 [ZMOD 2000] := by sorry

end power_tower_mod_2000_l540_54095


namespace digit_sum_problem_l540_54038

theorem digit_sum_problem (w x y z : ℕ) : 
  w ≤ 9 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧  -- digits are between 0 and 9
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧  -- all digits are different
  y + w = 10 ∧  -- sum in ones place
  x + y + 1 = 10 ∧  -- sum in tens place with carry
  w + z + 1 = 11  -- sum in hundreds place with carry
  →
  w + x + y + z = 23 := by
sorry

end digit_sum_problem_l540_54038


namespace new_earnings_after_raise_l540_54012

-- Define the original weekly earnings
def original_earnings : ℚ := 50

-- Define the percentage increase
def percentage_increase : ℚ := 50 / 100

-- Theorem to prove
theorem new_earnings_after_raise :
  original_earnings * (1 + percentage_increase) = 75 := by
  sorry

end new_earnings_after_raise_l540_54012


namespace min_value_theorem_l540_54021

theorem min_value_theorem (x y k : ℝ) 
  (hx : x > k) (hy : y > k) (hk : k > 1) :
  ∃ (m : ℝ), m = 8 * k ∧ 
  ∀ (a b : ℝ), a > k → b > k → 
  (a^2 / (b - k) + b^2 / (a - k)) ≥ m :=
by sorry

end min_value_theorem_l540_54021


namespace farmer_adds_eight_pigs_l540_54011

/-- Represents the number of animals on a farm -/
structure FarmAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- The initial number of animals on the farm -/
def initial : FarmAnimals := { cows := 2, pigs := 3, goats := 6 }

/-- The number of animals to be added -/
structure AddedAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- The final total number of animals -/
def finalTotal : ℕ := 21

/-- The number of cows and goats to be added -/
def knownAdditions : AddedAnimals := { cows := 3, pigs := 0, goats := 2 }

/-- Calculates the total number of animals -/
def totalAnimals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.pigs + farm.goats

/-- Theorem: The farmer plans to add 8 pigs -/
theorem farmer_adds_eight_pigs :
  ∃ (added : AddedAnimals),
    added.cows = knownAdditions.cows ∧
    added.goats = knownAdditions.goats ∧
    totalAnimals { cows := initial.cows + added.cows,
                   pigs := initial.pigs + added.pigs,
                   goats := initial.goats + added.goats } = finalTotal ∧
    added.pigs = 8 := by
  sorry

end farmer_adds_eight_pigs_l540_54011


namespace sum_x_coordinates_preserved_l540_54091

/-- A polygon in the Cartesian plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- Create a new polygon from the midpoints of the sides of a given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

theorem sum_x_coordinates_preserved (n : ℕ) (Q1 : Polygon) 
  (h1 : Q1.vertices.length = n)
  (Q2 := midpointPolygon Q1)
  (Q3 := midpointPolygon Q2) :
  sumXCoordinates Q3 = sumXCoordinates Q1 :=
sorry

end sum_x_coordinates_preserved_l540_54091


namespace solve_a_given_set_membership_l540_54064

theorem solve_a_given_set_membership (a : ℝ) : 
  -3 ∈ ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ) → a = 0 ∨ a = -1 := by
  sorry

end solve_a_given_set_membership_l540_54064


namespace sum_of_four_consecutive_even_integers_l540_54007

theorem sum_of_four_consecutive_even_integers :
  ¬ (∃ m : ℤ, 56 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 20 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 108 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 88 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 200 = 4*m + 12 ∧ Even m) := by
sorry

end sum_of_four_consecutive_even_integers_l540_54007


namespace researcher_reading_rate_l540_54067

theorem researcher_reading_rate 
  (total_pages : ℕ) 
  (total_hours : ℕ) 
  (h1 : total_pages = 30000) 
  (h2 : total_hours = 150) : 
  (total_pages : ℚ) / total_hours = 200 := by
  sorry

end researcher_reading_rate_l540_54067


namespace right_triangle_properties_l540_54078

-- Define a right triangle with hypotenuse 13 and one side 5
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2
  hypotenuse_length : c = 13
  side_length : a = 5

-- Theorem statement
theorem right_triangle_properties (t : RightTriangle) :
  t.b = 12 ∧
  (1/2 : ℝ) * t.a * t.b = 30 ∧
  t.a + t.b + t.c = 30 ∧
  (∃ θ₁ θ₂ : ℝ, 0 < θ₁ ∧ θ₁ < π/2 ∧ 0 < θ₂ ∧ θ₂ < π/2 ∧ θ₁ + θ₂ = π/2) :=
by sorry

end right_triangle_properties_l540_54078


namespace no_savings_on_joint_purchase_l540_54062

/-- Calculates the cost of windows given the quantity and discount offer -/
def windowCost (quantity : ℕ) (regularPrice : ℕ) (discountRate : ℕ) : ℕ :=
  let discountedQuantity := quantity - (quantity / (discountRate + 2)) * 2
  discountedQuantity * regularPrice

/-- Proves that joint purchase does not lead to savings compared to separate purchases -/
theorem no_savings_on_joint_purchase (regularPrice : ℕ) (discountRate : ℕ) :
  windowCost 22 regularPrice discountRate =
  windowCost 10 regularPrice discountRate + windowCost 12 regularPrice discountRate :=
by sorry

end no_savings_on_joint_purchase_l540_54062


namespace bella_max_number_l540_54060

theorem bella_max_number : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → 3 * (250 - n) ≤ 720 ∧ 
  ∃ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ 3 * (250 - m) = 720 :=
by sorry

end bella_max_number_l540_54060


namespace gain_percent_calculation_l540_54031

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 10)
  (h2 : selling_price = 15) : 
  (selling_price - cost_price) / cost_price * 100 = 50 := by
  sorry

end gain_percent_calculation_l540_54031


namespace original_network_engineers_l540_54081

/-- The number of new network engineers hired from University A -/
def new_hires : ℕ := 8

/-- The fraction of network engineers from University A after hiring -/
def fraction_after : ℚ := 3/4

/-- The fraction of original network engineers from University A -/
def fraction_original : ℚ := 13/20

/-- The original number of network engineers -/
def original_count : ℕ := 20

theorem original_network_engineers :
  ∃ (o : ℕ), 
    (o : ℚ) * fraction_original + new_hires = 
    ((o : ℚ) + new_hires) * fraction_after ∧
    o = original_count :=
by sorry

end original_network_engineers_l540_54081


namespace five_b_value_l540_54008

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 4) (h2 : b - 3 = a) : 5 * b = 65 / 7 := by
  sorry

end five_b_value_l540_54008


namespace women_average_age_l540_54045

theorem women_average_age (n : ℕ) (A : ℝ) :
  n = 12 ∧
  (n * (A + 3) = n * A + 3 * 42 - (25 + 30 + 35)) →
  42 = (3 * 42) / 3 :=
by sorry

end women_average_age_l540_54045


namespace parallelogram_vertices_parabola_parallel_intersection_l540_54022

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points form a parallelogram -/
def isParallelogram (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x = p4.x - p3.x ∧ p2.y - p1.y = p4.y - p3.y) ∨
  (p3.x - p1.x = p4.x - p2.x ∧ p3.y - p1.y = p4.y - p2.y)

/-- The parabola equation x = y^2 -/
def onParabola (p : Point) : Prop :=
  p.x = p.y^2

/-- Theorem about parallelogram vertices -/
theorem parallelogram_vertices :
  ∀ p : Point,
  isParallelogram ⟨0, 0⟩ ⟨1, 1⟩ ⟨1, 0⟩ p →
  (p = ⟨0, 1⟩ ∨ p = ⟨0, -1⟩ ∨ p = ⟨2, 1⟩) :=
sorry

/-- Theorem about parallel lines intersecting parabola -/
theorem parabola_parallel_intersection (a : ℝ) :
  a ≠ 0 → a ≠ 1 → a ≠ -1 →
  ∀ v : Point,
  onParabola ⟨0, 0⟩ ∧ onParabola ⟨1, 1⟩ ∧ onParabola ⟨a^2, a⟩ ∧ onParabola v →
  (∃ l1 l2 : ℝ → ℝ, l1 0 = 0 ∧ l1 1 = 1 ∧ l1 (a^2) = a ∧ l1 v.x = v.y ∧
               l2 0 = 0 ∧ l2 1 = 1 ∧ l2 (a^2) = a ∧ l2 v.x = v.y ∧
               ∀ x, l1 x - l2 x = (l1 1 - l2 1)) →
  (v = ⟨4, a⟩ ∨ v = ⟨4, -a⟩) :=
sorry

end parallelogram_vertices_parabola_parallel_intersection_l540_54022


namespace expression_simplification_l540_54028

theorem expression_simplification (y : ℝ) :
  3 * y - 5 * y^2 + 10 - (8 - 3 * y + 5 * y^2 - y^3) = y^3 - 10 * y^2 + 6 * y + 2 := by
  sorry

end expression_simplification_l540_54028


namespace green_to_yellow_ratio_is_two_to_one_l540_54023

/-- Represents the number of fish of each color in an aquarium -/
structure FishCounts where
  total : ℕ
  yellow : ℕ
  blue : ℕ
  green : ℕ
  other : ℕ

/-- Calculates the ratio of green fish to yellow fish -/
def greenToYellowRatio (fc : FishCounts) : ℚ :=
  fc.green / fc.yellow

/-- Theorem: The ratio of green fish to yellow fish is 2:1 given the conditions -/
theorem green_to_yellow_ratio_is_two_to_one (fc : FishCounts)
  (h1 : fc.total = 42)
  (h2 : fc.yellow = 12)
  (h3 : fc.blue = fc.yellow / 2)
  (h4 : fc.total = fc.yellow + fc.blue + fc.green + fc.other) :
  greenToYellowRatio fc = 2 := by
  sorry

#eval greenToYellowRatio { total := 42, yellow := 12, blue := 6, green := 24, other := 0 }

end green_to_yellow_ratio_is_two_to_one_l540_54023


namespace simplify_fraction_l540_54076

theorem simplify_fraction : (5 : ℚ) * (13 / 3) * (21 / -65) = -7 := by
  sorry

end simplify_fraction_l540_54076


namespace mrs_hilt_hot_dog_cost_l540_54084

/-- The total cost of hot dogs in cents -/
def total_cost (num_hot_dogs : ℕ) (cost_per_hot_dog : ℕ) : ℕ :=
  num_hot_dogs * cost_per_hot_dog

/-- Theorem: Mrs. Hilt's total cost for hot dogs is 300 cents -/
theorem mrs_hilt_hot_dog_cost :
  total_cost 6 50 = 300 := by
  sorry

end mrs_hilt_hot_dog_cost_l540_54084
