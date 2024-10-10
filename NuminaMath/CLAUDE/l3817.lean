import Mathlib

namespace team_average_goals_is_seven_l3817_381796

/-- The average number of goals scored by a soccer team per game -/
def team_average_goals (carter_goals shelby_goals judah_goals : ℝ) : ℝ :=
  carter_goals + shelby_goals + judah_goals

/-- Theorem: Given the conditions, the team's average goals per game is 7 -/
theorem team_average_goals_is_seven :
  ∀ (carter_goals shelby_goals judah_goals : ℝ),
    carter_goals = 4 →
    shelby_goals = carter_goals / 2 →
    judah_goals = 2 * shelby_goals - 3 →
    team_average_goals carter_goals shelby_goals judah_goals = 7 := by
  sorry

end team_average_goals_is_seven_l3817_381796


namespace line_point_sum_l3817_381786

/-- The line equation y = -5/3x + 10 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 10

/-- Point P is on the x-axis -/
def P_on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

/-- Point Q is on the y-axis -/
def Q_on_y_axis (Q : ℝ × ℝ) : Prop := Q.1 = 0

/-- Point T is on line segment PQ -/
def T_on_PQ (P Q T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ T = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2)

/-- Area of triangle POQ is 4 times the area of triangle TOP -/
def area_ratio (P Q T : ℝ × ℝ) : Prop :=
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) =
  4 * abs ((T.1 - 0) * (P.2 - 0) - (P.1 - 0) * (T.2 - 0))

theorem line_point_sum (P Q T : ℝ × ℝ) (r s : ℝ) :
  line_equation P.1 P.2 →
  line_equation Q.1 Q.2 →
  line_equation T.1 T.2 →
  P_on_x_axis P →
  Q_on_y_axis Q →
  T_on_PQ P Q T →
  area_ratio P Q T →
  T = (r, s) →
  r + s = 7 := by sorry

end line_point_sum_l3817_381786


namespace max_area_rectangle_l3817_381768

/-- Represents a rectangle with integer dimensions and perimeter 40 -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_constraint : length + width = 20

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem: The maximum area of a rectangle with perimeter 40 and integer dimensions is 100 -/
theorem max_area_rectangle :
  ∀ r : Rectangle, area r ≤ 100 :=
sorry

end max_area_rectangle_l3817_381768


namespace triangle_angle_sine_inequality_l3817_381744

theorem triangle_angle_sine_inequality (α β γ : Real) 
  (h_triangle : α + β + γ = π) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) :
  2 * (Real.sin α / α + Real.sin β / β + Real.sin γ / γ) ≤ 
    (1 / β + 1 / γ) * Real.sin α + 
    (1 / γ + 1 / α) * Real.sin β + 
    (1 / α + 1 / β) * Real.sin γ := by
  sorry

end triangle_angle_sine_inequality_l3817_381744


namespace solve_linear_equation_l3817_381742

theorem solve_linear_equation :
  ∃ x : ℚ, -3 * x - 12 = 8 * x + 5 ∧ x = -17 / 11 := by
  sorry

end solve_linear_equation_l3817_381742


namespace line_points_product_l3817_381759

/-- Given a line k passing through the origin with slope √7 / 3,
    if points (x, 8) and (20, y) lie on this line, then x * y = 160. -/
theorem line_points_product (x y : ℝ) : 
  (∃ k : ℝ → ℝ, k 0 = 0 ∧ 
   (∀ x₁ x₂, x₁ ≠ x₂ → (k x₂ - k x₁) / (x₂ - x₁) = Real.sqrt 7 / 3) ∧
   k x = 8 ∧ k 20 = y) →
  x * y = 160 := by
sorry


end line_points_product_l3817_381759


namespace negation_of_existence_negation_of_quadratic_inequality_l3817_381750

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_inequality_l3817_381750


namespace product_equals_one_l3817_381784

theorem product_equals_one (a b c : ℝ) 
  (h1 : a^2 + 2 = b^4) 
  (h2 : b^2 + 2 = c^4) 
  (h3 : c^2 + 2 = a^4) : 
  (a^2 - 1) * (b^2 - 1) * (c^2 - 1) = 1 := by
  sorry

end product_equals_one_l3817_381784


namespace gpa_ratio_is_one_third_l3817_381730

/-- Represents a class with two groups of students with different GPAs -/
structure ClassGPA where
  totalStudents : ℕ
  studentsGPA30 : ℕ
  gpa30 : ℝ := 30
  gpa33 : ℝ := 33
  overallGPA : ℝ := 32

/-- The ratio of students with GPA 30 to the total number of students is 1/3 -/
theorem gpa_ratio_is_one_third (c : ClassGPA) 
  (h1 : c.studentsGPA30 ≤ c.totalStudents)
  (h2 : c.totalStudents > 0)
  (h3 : c.gpa30 * c.studentsGPA30 + c.gpa33 * (c.totalStudents - c.studentsGPA30) = c.overallGPA * c.totalStudents) :
  c.studentsGPA30 / c.totalStudents = 1 / 3 := by
  sorry

end gpa_ratio_is_one_third_l3817_381730


namespace circle_radius_from_area_l3817_381785

theorem circle_radius_from_area (A : ℝ) (h : A = 64 * Real.pi) :
  ∃ r : ℝ, r > 0 ∧ A = Real.pi * r^2 ∧ r = 8 := by
  sorry

end circle_radius_from_area_l3817_381785


namespace jills_water_volume_l3817_381791

/-- Represents the number of jars of each size -/
def jars_per_size : ℕ := 48 / 3

/-- Represents the volume of a quart in gallons -/
def quart_volume : ℚ := 1 / 4

/-- Represents the volume of a half-gallon in gallons -/
def half_gallon_volume : ℚ := 1 / 2

/-- Represents the volume of a gallon in gallons -/
def gallon_volume : ℚ := 1

/-- Calculates the total volume of water in gallons -/
def total_water_volume : ℚ :=
  jars_per_size * quart_volume +
  jars_per_size * half_gallon_volume +
  jars_per_size * gallon_volume

theorem jills_water_volume :
  total_water_volume = 28 := by
  sorry

end jills_water_volume_l3817_381791


namespace hyperbola_standard_equation_l3817_381716

/-- Represents a hyperbola with foci on the y-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  focal_distance : ℝ
  focus_to_asymptote : ℝ
  h_positive : a > 0
  b_positive : b > 0
  h_focal_distance : focal_distance = 2 * Real.sqrt 3
  h_focus_to_asymptote : focus_to_asymptote = Real.sqrt 2
  h_c : c = Real.sqrt 3
  h_relation : c^2 = a^2 + b^2
  h_asymptote : b * c / Real.sqrt (a^2 + b^2) = focus_to_asymptote

/-- The standard equation of the hyperbola is y² - x²/2 = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  h.a = 1 ∧ h.b = Real.sqrt 2 := by sorry

end hyperbola_standard_equation_l3817_381716


namespace small_sphere_acceleration_l3817_381700

/-- The acceleration of a small charged sphere after material removal from a larger charged sphere -/
theorem small_sphere_acceleration
  (k : ℝ) -- Coulomb's constant
  (q Q : ℝ) -- Charges of small and large spheres
  (r R : ℝ) -- Radii of small and large spheres
  (m : ℝ) -- Mass of small sphere
  (L S : ℝ) -- Distances
  (g : ℝ) -- Acceleration due to gravity
  (h_r_small : r < R)
  (h_initial_balance : k * q * Q / (L + R)^2 = m * g)
  : ∃ (a : ℝ), a = (k * q * Q * r^3) / (m * R^3 * (L + 2*R - S)^2) :=
sorry

end small_sphere_acceleration_l3817_381700


namespace closest_point_l3817_381799

def w (s : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3 + 8*s
  | 1 => -2 + 6*s
  | 2 => -4 - 2*s

def b : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 5
  | 2 => 6

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 8
  | 1 => 6
  | 2 => -2

theorem closest_point (s : ℝ) : 
  (s = 19/52) ↔ 
  (∀ t : ℝ, ‖w s - b‖ ≤ ‖w t - b‖) :=
sorry

end closest_point_l3817_381799


namespace water_in_tank_after_rain_l3817_381798

/-- Given an initial amount of water, a water flow rate, and a rainstorm duration,
    calculate the final amount of water in the tank. -/
def final_water_amount (initial_amount : ℝ) (flow_rate : ℝ) (duration : ℝ) : ℝ :=
  initial_amount + flow_rate * duration

/-- Theorem stating that given the specific conditions in the problem,
    the final amount of water in the tank is 280 L. -/
theorem water_in_tank_after_rain : final_water_amount 100 2 90 = 280 := by
  sorry

end water_in_tank_after_rain_l3817_381798


namespace modulus_of_3_minus_4i_l3817_381758

theorem modulus_of_3_minus_4i : Complex.abs (3 - 4 * Complex.I) = 5 := by
  sorry

end modulus_of_3_minus_4i_l3817_381758


namespace evenly_geometric_difference_l3817_381739

/-- A 3-digit number is evenly geometric if it comprises 3 distinct even digits
    which form a geometric sequence when read from left to right. -/
def EvenlyGeometric (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧
                 a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                 Even a ∧ Even b ∧ Even c ∧
                 ∃ (r : ℚ), b = a * r ∧ c = a * r^2

theorem evenly_geometric_difference :
  ∃ (max min : ℕ),
    (∀ n, EvenlyGeometric n → n ≤ max) ∧
    (∀ n, EvenlyGeometric n → min ≤ n) ∧
    (EvenlyGeometric max) ∧
    (EvenlyGeometric min) ∧
    max - min = 0 :=
sorry

end evenly_geometric_difference_l3817_381739


namespace ellipse_equation_l3817_381753

/-- An ellipse with center at the origin, right focus at (1,0), and eccentricity 1/2 -/
structure Ellipse where
  /-- The x-coordinate of a point on the ellipse -/
  x : ℝ
  /-- The y-coordinate of a point on the ellipse -/
  y : ℝ
  /-- The distance from the center to the focus -/
  c : ℝ
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The semi-major axis of the ellipse -/
  a : ℝ
  /-- The semi-minor axis of the ellipse -/
  b : ℝ
  /-- The center is at the origin -/
  center_origin : c = 1
  /-- The eccentricity is 1/2 -/
  eccentricity_half : e = 1/2
  /-- The relation between eccentricity, c, and a -/
  eccentricity_def : e = c / a
  /-- The relation between a, b, and c -/
  axis_relation : b^2 = a^2 - c^2

/-- The equation of the ellipse is x^2/4 + y^2/3 = 1 -/
theorem ellipse_equation (C : Ellipse) : C.x^2 / 4 + C.y^2 / 3 = 1 :=
  sorry

end ellipse_equation_l3817_381753


namespace prime_divisibility_problem_l3817_381711

theorem prime_divisibility_problem (p n : ℕ) : 
  p.Prime → 
  n > 0 → 
  n ≤ 2 * p → 
  (n ^ (p - 1) ∣ (p - 1) ^ n + 1) → 
  ((p = 2 ∧ n = 2) ∨ (p = 3 ∧ n = 3) ∨ n = 1) := by
sorry

end prime_divisibility_problem_l3817_381711


namespace cost_price_calculation_l3817_381702

theorem cost_price_calculation (profit_difference : ℝ) 
  (h1 : profit_difference = 72) 
  (h2 : (0.18 - 0.09) * cost_price = profit_difference) : 
  cost_price = 800 :=
by
  sorry

end cost_price_calculation_l3817_381702


namespace d_range_l3817_381733

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- The condition that a₃a₄ + 1 = 0 for an arithmetic sequence -/
def sequence_condition (a₁ d : ℝ) : Prop :=
  (arithmetic_sequence a₁ d 3) * (arithmetic_sequence a₁ d 4) + 1 = 0

/-- The theorem stating the range of possible values for d -/
theorem d_range (a₁ d : ℝ) :
  sequence_condition a₁ d → d ≤ -2 ∨ d ≥ 2 := by
  sorry

end d_range_l3817_381733


namespace cistern_water_depth_l3817_381776

/-- Proves that for a rectangular cistern with given dimensions and wet surface area, the water depth is as calculated. -/
theorem cistern_water_depth
  (length : ℝ)
  (width : ℝ)
  (total_wet_surface_area : ℝ)
  (h : ℝ)
  (h_length : length = 4)
  (h_width : width = 8)
  (h_total_area : total_wet_surface_area = 62)
  (h_depth : h = (total_wet_surface_area - length * width) / (2 * (length + width))) :
  h = 1.25 := by
  sorry

end cistern_water_depth_l3817_381776


namespace squares_to_rectangles_ratio_l3817_381747

/-- The number of ways to choose 2 items from 10 --/
def choose_2_from_10 : ℕ := 45

/-- The number of rectangles on a 10x10 chessboard --/
def num_rectangles : ℕ := choose_2_from_10 * choose_2_from_10

/-- The sum of squares from 1^2 to 10^2 --/
def sum_squares : ℕ := (10 * 11 * 21) / 6

/-- The number of squares on a 10x10 chessboard --/
def num_squares : ℕ := sum_squares

/-- The ratio of squares to rectangles on a 10x10 chessboard is 7/37 --/
theorem squares_to_rectangles_ratio :
  (num_squares : ℚ) / (num_rectangles : ℚ) = 7 / 37 := by sorry

end squares_to_rectangles_ratio_l3817_381747


namespace ninth_grader_wins_l3817_381775

/-- Represents the grade of a student -/
inductive Grade
| Ninth
| Tenth

/-- Represents a chess tournament with ninth and tenth graders -/
structure ChessTournament where
  ninth_graders : ℕ
  tenth_graders : ℕ
  ninth_points : ℕ
  tenth_points : ℕ

/-- Chess tournament satisfying the given conditions -/
def valid_tournament (t : ChessTournament) : Prop :=
  t.tenth_graders = 9 * t.ninth_graders ∧
  t.tenth_points = 4 * t.ninth_points

/-- Maximum points a single player can score -/
def max_player_points (t : ChessTournament) (g : Grade) : ℕ :=
  match g with
  | Grade.Ninth => t.tenth_graders
  | Grade.Tenth => (t.tenth_graders - 1) / 2

/-- Theorem stating that a ninth grader wins the tournament with 9 points -/
theorem ninth_grader_wins (t : ChessTournament) 
  (h : valid_tournament t) (h_ninth : t.ninth_graders > 0) :
  ∃ (n : ℕ), n = 9 ∧ 
    n = max_player_points t Grade.Ninth ∧ 
    n > max_player_points t Grade.Tenth :=
  sorry

end ninth_grader_wins_l3817_381775


namespace kates_hair_length_l3817_381729

/-- Given information about hair lengths of Kate, Emily, and Logan, prove Kate's hair length -/
theorem kates_hair_length (logan_length emily_length kate_length : ℝ) : 
  logan_length = 20 →
  emily_length = logan_length + 6 →
  kate_length = emily_length / 2 →
  kate_length = 13 := by
  sorry

end kates_hair_length_l3817_381729


namespace modular_congruence_in_range_l3817_381754

theorem modular_congruence_in_range : ∃ n : ℤ, 5 ≤ n ∧ n ≤ 12 ∧ n ≡ 10569 [ZMOD 7] ∧ n = 5 := by
  sorry

end modular_congruence_in_range_l3817_381754


namespace equal_quadratic_expressions_l3817_381712

theorem equal_quadratic_expressions (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 6) :
  a * (a - 6) = b * (b - 6) ∧ a * (a - 6) = -9 := by
  sorry

end equal_quadratic_expressions_l3817_381712


namespace equation_solution_l3817_381705

theorem equation_solution : 
  let f : ℝ → ℝ := fun y => y^2 - 3*y - 10 + (y + 2)*(y + 6)
  (f (-1/2) = 0 ∧ f (-2) = 0) ∧ 
  ∀ y : ℝ, f y = 0 → (y = -1/2 ∨ y = -2) := by
sorry

end equation_solution_l3817_381705


namespace theater_revenue_specific_case_l3817_381788

def theater_revenue (orchestra_price balcony_price : ℕ) 
                    (total_tickets balcony_orchestra_diff : ℕ) : ℕ :=
  let orchestra_tickets := (total_tickets - balcony_orchestra_diff) / 2
  let balcony_tickets := total_tickets - orchestra_tickets
  orchestra_price * orchestra_tickets + balcony_price * balcony_tickets

theorem theater_revenue_specific_case :
  theater_revenue 12 8 360 140 = 3320 := by
  sorry

end theater_revenue_specific_case_l3817_381788


namespace rightmost_three_digits_of_3_to_2023_l3817_381745

theorem rightmost_three_digits_of_3_to_2023 : 3^2023 % 1000 = 787 := by
  sorry

end rightmost_three_digits_of_3_to_2023_l3817_381745


namespace exponential_decrease_l3817_381766

theorem exponential_decrease (x y a : Real) 
  (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < 1) : 
  a^x < a^y := by
  sorry

end exponential_decrease_l3817_381766


namespace missing_number_problem_l3817_381726

theorem missing_number_problem (x n : ℕ) (h_pos : x > 0) :
  let numbers := [x, x + 2, x + n, x + 7, x + 17]
  let mean := (x + (x + 2) + (x + n) + (x + 7) + (x + 17)) / 5
  let median := x + n
  (mean = median + 2) → n = 4 := by
sorry

end missing_number_problem_l3817_381726


namespace min_students_above_60_l3817_381782

/-- Represents a score distribution in a math competition. -/
structure ScoreDistribution where
  totalScore : ℕ
  topThreeScores : Fin 3 → ℕ
  lowestScore : ℕ
  maxSameScore : ℕ

/-- The minimum number of students who scored at least 60 points. -/
def minStudentsAbove60 (sd : ScoreDistribution) : ℕ := 61

/-- The given conditions of the math competition. -/
def mathCompetition : ScoreDistribution where
  totalScore := 8250
  topThreeScores := ![88, 85, 80]
  lowestScore := 30
  maxSameScore := 3

/-- Theorem stating that the minimum number of students who scored at least 60 points is 61. -/
theorem min_students_above_60 :
  minStudentsAbove60 mathCompetition = 61 := by
  sorry

#check min_students_above_60

end min_students_above_60_l3817_381782


namespace fish_birth_calculation_l3817_381779

theorem fish_birth_calculation (num_tanks : ℕ) (fish_per_tank : ℕ) (total_young : ℕ) :
  num_tanks = 3 →
  fish_per_tank = 4 →
  total_young = 240 →
  total_young / (num_tanks * fish_per_tank) = 20 :=
by sorry

end fish_birth_calculation_l3817_381779


namespace divisible_by_five_l3817_381793

theorem divisible_by_five (a b : ℕ+) (h : 5 ∣ (a * b)) : 5 ∣ a ∨ 5 ∣ b := by
  sorry

end divisible_by_five_l3817_381793


namespace second_train_speed_l3817_381734

/-- Proves that the speed of the second train is 36 kmph given the conditions of the problem -/
theorem second_train_speed (first_train_speed : ℝ) (time_difference : ℝ) (meeting_distance : ℝ) :
  first_train_speed = 30 →
  time_difference = 5 →
  meeting_distance = 1050 →
  ∃ (second_train_speed : ℝ),
    second_train_speed * (meeting_distance / second_train_speed) =
    meeting_distance - first_train_speed * time_difference +
    first_train_speed * (meeting_distance / second_train_speed) ∧
    second_train_speed = 36 :=
by
  sorry


end second_train_speed_l3817_381734


namespace smallest_value_of_reciprocal_sum_l3817_381771

theorem smallest_value_of_reciprocal_sum (u q a₁ a₂ : ℝ) : 
  (a₁ * a₂ = q) →  -- Vieta's formula for product of roots
  (a₁ + a₂ = u) →  -- Vieta's formula for sum of roots
  (a₁ + a₂ = a₁^2 + a₂^2) →
  (a₁ + a₂ = a₁^3 + a₂^3) →
  (a₁ + a₂ = a₁^4 + a₂^4) →
  (∀ u' q' a₁' a₂' : ℝ, 
    (a₁' * a₂' = q') → 
    (a₁' + a₂' = u') → 
    (a₁' + a₂' = a₁'^2 + a₂'^2) → 
    (a₁' + a₂' = a₁'^3 + a₂'^3) → 
    (a₁' + a₂' = a₁'^4 + a₂'^4) → 
    (a₁' ≠ 0 ∧ a₂' ≠ 0) →
    (1 / a₁^10 + 1 / a₂^10 ≤ 1 / a₁'^10 + 1 / a₂'^10)) →
  1 / a₁^10 + 1 / a₂^10 = 2 :=
by sorry

end smallest_value_of_reciprocal_sum_l3817_381771


namespace nancys_hourly_wage_l3817_381794

/-- Proves that Nancy needs to make $10 per hour to pay the rest of her tuition --/
theorem nancys_hourly_wage (tuition : ℝ) (scholarship : ℝ) (work_hours : ℝ) :
  tuition = 22000 →
  scholarship = 3000 →
  work_hours = 200 →
  (tuition / 2 - scholarship - 2 * scholarship) / work_hours = 10 := by
  sorry

end nancys_hourly_wage_l3817_381794


namespace average_weight_problem_l3817_381701

theorem average_weight_problem (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (B + C) / 2 = 43)
  (h3 : B = 31) :
  (A + B) / 2 = 40 := by
  sorry

end average_weight_problem_l3817_381701


namespace windows_preference_l3817_381770

theorem windows_preference (total : ℕ) (mac : ℕ) (no_pref : ℕ) 
  (h1 : total = 210)
  (h2 : mac = 60)
  (h3 : no_pref = 90) :
  total - mac - (mac / 3) - no_pref = 40 := by
  sorry

end windows_preference_l3817_381770


namespace equal_perimeter_not_necessarily_congruent_l3817_381722

-- Define a triangle type
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define perimeter of a triangle
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Theorem statement
theorem equal_perimeter_not_necessarily_congruent :
  ∃ (t1 t2 : Triangle), perimeter t1 = perimeter t2 ∧ ¬congruent t1 t2 :=
sorry

end equal_perimeter_not_necessarily_congruent_l3817_381722


namespace classification_theorem_l3817_381756

def expressions : List String := [
  "4xy", "m^2n/2", "y^2 + y + 2/y", "2x^3 - 3", "0", "-3/(ab) + a",
  "m", "(m-n)/(m+n)", "(x-1)/2", "3/x"
]

def is_monomial (expr : String) : Bool := sorry

def is_polynomial (expr : String) : Bool := sorry

theorem classification_theorem :
  let monomials := expressions.filter is_monomial
  let polynomials := expressions.filter (λ e => is_polynomial e ∧ ¬is_monomial e)
  let all_polynomials := expressions.filter is_polynomial
  (monomials = ["4xy", "m^2n/2", "0", "m"]) ∧
  (polynomials = ["2x^3 - 3", "(x-1)/2"]) ∧
  (all_polynomials = ["4xy", "m^2n/2", "2x^3 - 3", "0", "m", "(x-1)/2"]) := by
  sorry

end classification_theorem_l3817_381756


namespace range_of_a_l3817_381727

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x - 2|

-- Define the range M of y = 2f(x)
def M : Set ℝ := Set.range (λ x => 2 * f x)

-- Theorem statement
theorem range_of_a (a : ℝ) (h : Set.Icc a (2*a - 1) ⊆ M) : 1 ≤ a ∧ a ≤ 3/2 := by
  sorry


end range_of_a_l3817_381727


namespace boat_speed_ratio_l3817_381790

/-- Calculates the ratio of average speed to still water speed for a boat in a river --/
theorem boat_speed_ratio 
  (v : ℝ) -- Boat speed in still water
  (c : ℝ) -- River current speed
  (d : ℝ) -- Distance traveled each way
  (h1 : v > 0)
  (h2 : c ≥ 0)
  (h3 : c < v)
  (h4 : d > 0)
  : (2 * d) / ((d / (v + c)) + (d / (v - c))) / v = 24 / 25 :=
by sorry

end boat_speed_ratio_l3817_381790


namespace final_S_equals_3_pow_10_l3817_381724

/-- Represents the state of the program at each iteration --/
structure ProgramState where
  S : ℕ
  i : ℕ

/-- The initial state of the program --/
def initial_state : ProgramState := { S := 1, i := 1 }

/-- The transition function for each iteration of the loop --/
def iterate (state : ProgramState) : ProgramState :=
  { S := state.S * 3, i := state.i + 1 }

/-- The final state after the loop completes --/
def final_state : ProgramState :=
  (iterate^[10]) initial_state

/-- The theorem stating that the final value of S is equal to 3^10 --/
theorem final_S_equals_3_pow_10 : final_state.S = 3^10 := by
  sorry

end final_S_equals_3_pow_10_l3817_381724


namespace set_operations_l3817_381708

def A : Set ℝ := {x | x ≤ 5}
def B : Set ℝ := {x | 3 < x ∧ x ≤ 7}

theorem set_operations :
  (A ∩ B = {x | 3 < x ∧ x ≤ 5}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ 5 ∨ x > 7}) := by
  sorry

end set_operations_l3817_381708


namespace multiples_of_15_sequence_two_thousand_sixteen_position_l3817_381765

theorem multiples_of_15_sequence (n : ℕ) : ℕ → ℕ
  | 0 => 0
  | (k + 1) => 15 * (k + 1)

theorem two_thousand_sixteen_position :
  ∃ (n : ℕ), multiples_of_15_sequence n 134 < 2016 ∧ 
             2016 < multiples_of_15_sequence n 135 ∧ 
             multiples_of_15_sequence n 135 - 2016 = 9 := by
  sorry


end multiples_of_15_sequence_two_thousand_sixteen_position_l3817_381765


namespace travel_distance_proof_l3817_381728

theorem travel_distance_proof (total_distance : ℝ) (bus_distance : ℝ) : 
  total_distance = 1800 →
  bus_distance = 720 →
  (1/3 : ℝ) * total_distance + (2/3 : ℝ) * bus_distance + bus_distance = total_distance :=
by
  sorry

end travel_distance_proof_l3817_381728


namespace binary_arithmetic_equality_l3817_381725

/-- Convert a list of bits (0s and 1s) to a natural number -/
def binaryToNat (bits : List Nat) : Nat :=
  bits.foldl (fun acc b => 2 * acc + b) 0

/-- The theorem to prove -/
theorem binary_arithmetic_equality :
  let a := binaryToNat [1, 1, 0, 1]
  let b := binaryToNat [1, 1, 1, 0]
  let c := binaryToNat [1, 0, 1, 1]
  let d := binaryToNat [1, 0, 0, 1]
  let e := binaryToNat [1, 0, 1]
  a + b - c + d - e = binaryToNat [1, 0, 0, 0, 0] := by
  sorry

end binary_arithmetic_equality_l3817_381725


namespace tangerines_count_l3817_381795

/-- The number of tangerines in a fruit basket -/
def num_tangerines (total fruits bananas apples pears : ℕ) : ℕ :=
  total - (bananas + apples + pears)

/-- Theorem: There are 13 tangerines in the fruit basket -/
theorem tangerines_count :
  let total := 60
  let bananas := 32
  let apples := 10
  let pears := 5
  num_tangerines total bananas apples pears = 13 := by
  sorry

end tangerines_count_l3817_381795


namespace polynomial_evaluation_l3817_381709

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 8 where f(-2) = 10, prove that f(2) = 6 -/
theorem polynomial_evaluation (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 8
  f (-2) = 10 → f 2 = 6 := by sorry

end polynomial_evaluation_l3817_381709


namespace triangle_side_a_triangle_angle_B_l3817_381769

-- Part I
theorem triangle_side_a (A B C : ℝ) (a b c : ℝ) : 
  b = Real.sqrt 3 → A = π / 4 → C = 5 * π / 12 → a = Real.sqrt 2 := by sorry

-- Part II
theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) :
  b^2 = a^2 + c^2 + Real.sqrt 2 * a * c → B = 3 * π / 4 := by sorry

end triangle_side_a_triangle_angle_B_l3817_381769


namespace decimal_representation_of_sqrt2_plus_sqrt3_power_1980_l3817_381706

theorem decimal_representation_of_sqrt2_plus_sqrt3_power_1980 :
  let x := (Real.sqrt 2 + Real.sqrt 3) ^ 1980
  ∃ y : ℝ, 0 ≤ y ∧ y < 1 ∧ x = 7 + y ∧ y > 0.9 := by
  sorry

end decimal_representation_of_sqrt2_plus_sqrt3_power_1980_l3817_381706


namespace halloween_candy_count_l3817_381760

/-- The number of candy pieces remaining after Halloween --/
def remaining_candy (debby_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  debby_candy + sister_candy - eaten_candy

/-- Theorem stating the remaining candy count for the given scenario --/
theorem halloween_candy_count : remaining_candy 32 42 35 = 39 := by
  sorry

end halloween_candy_count_l3817_381760


namespace book_sale_profit_l3817_381721

theorem book_sale_profit (cost_price : ℝ) (discount_rate : ℝ) (no_discount_profit_rate : ℝ) :
  discount_rate = 0.05 →
  no_discount_profit_rate = 1.2 →
  let selling_price := cost_price * (1 + no_discount_profit_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let profit_with_discount := discounted_price - cost_price
  let profit_rate_with_discount := profit_with_discount / cost_price
  profit_rate_with_discount = 1.09 := by
sorry

end book_sale_profit_l3817_381721


namespace range_of_a_for_intersection_l3817_381773

theorem range_of_a_for_intersection (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ 
   Real.cos (Real.pi * x₁) = 2^x₂ * a - 1/2) ↔ 
  a ∈ Set.Icc (-1/2) 0 ∪ Set.Ioc 0 (3/2) :=
sorry

end range_of_a_for_intersection_l3817_381773


namespace school_children_count_l3817_381783

/-- The number of children in the school --/
def N : ℕ := sorry

/-- The number of bananas available --/
def B : ℕ := sorry

/-- The number of absent children --/
def absent : ℕ := 330

theorem school_children_count :
  (2 * N = B) ∧                 -- Initial distribution: 2 bananas per child
  (4 * (N - absent) = B) →      -- Actual distribution: 4 bananas per child after absences
  N = 660 := by sorry

end school_children_count_l3817_381783


namespace soccer_games_per_month_l3817_381720

/-- Given a total of 27 soccer games equally divided over 3 months,
    the number of games played per month is 9. -/
theorem soccer_games_per_month :
  ∀ (total_games : ℕ) (num_months : ℕ) (games_per_month : ℕ),
    total_games = 27 →
    num_months = 3 →
    total_games = num_months * games_per_month →
    games_per_month = 9 := by
  sorry

end soccer_games_per_month_l3817_381720


namespace smallest_integer_solution_l3817_381772

theorem smallest_integer_solution (x : ℝ) :
  (x - 3 * (x - 2) ≤ 4 ∧ (1 + 2 * x) / 3 < x - 1) →
  (∀ y : ℤ, y < 5 → ¬(y - 3 * (y - 2) ≤ 4 ∧ (1 + 2 * y) / 3 < y - 1)) ∧
  (5 - 3 * (5 - 2) ≤ 4 ∧ (1 + 2 * 5) / 3 < 5 - 1) :=
by sorry

end smallest_integer_solution_l3817_381772


namespace arccos_cos_nine_l3817_381763

theorem arccos_cos_nine :
  Real.arccos (Real.cos 9) = 9 - 2 * Real.pi := by sorry

end arccos_cos_nine_l3817_381763


namespace trigonometric_equation_solution_l3817_381719

theorem trigonometric_equation_solution (a : ℝ) : 
  (∀ x, Real.cos (3 * a) * Real.sin x + (Real.sin (3 * a) - Real.sin (7 * a)) * Real.cos x = 0) ∧
  Real.cos (3 * a) = 0 ∧
  Real.sin (3 * a) - Real.sin (7 * a) = 0 →
  ∃ t : ℤ, a = π * (2 * ↑t + 1) / 2 :=
by sorry

end trigonometric_equation_solution_l3817_381719


namespace xy_sum_problem_l3817_381707

theorem xy_sum_problem (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + x + y = 7) :
  x^2*y + x*y^2 = 245/36 := by
sorry

end xy_sum_problem_l3817_381707


namespace circle_through_points_l3817_381741

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 2*y + 12 = 0

-- Define the points
def P : ℝ × ℝ := (2, 2)
def M : ℝ × ℝ := (5, 3)
def N : ℝ × ℝ := (3, -1)

-- Theorem statement
theorem circle_through_points :
  circle_equation P.1 P.2 ∧ circle_equation M.1 M.2 ∧ circle_equation N.1 N.2 :=
sorry

end circle_through_points_l3817_381741


namespace universal_set_intersection_l3817_381780

-- Define the universe
variable (U : Type)

-- Define sets A and B
variable (A B : Set U)

-- Define S as the universal set
variable (S : Set U)

-- Theorem statement
theorem universal_set_intersection (h1 : S = Set.univ) (h2 : A ∩ B = S) : A = S ∧ B = S := by
  sorry

end universal_set_intersection_l3817_381780


namespace baker_pastry_cake_difference_l3817_381761

/-- The number of cakes made by the baker -/
def cakes_made : ℕ := 19

/-- The number of pastries made by the baker -/
def pastries_made : ℕ := 131

/-- The difference between pastries and cakes made by the baker -/
def pastry_cake_difference : ℕ := pastries_made - cakes_made

theorem baker_pastry_cake_difference :
  pastry_cake_difference = 112 := by sorry

end baker_pastry_cake_difference_l3817_381761


namespace problem_statement_l3817_381792

theorem problem_statement (C D : ℝ) :
  (∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-2 * x^2 + 8 * x + 28) / (x - 3)) →
  C + D = 20 := by
sorry

end problem_statement_l3817_381792


namespace min_value_theorem_l3817_381743

theorem min_value_theorem (p q r s t u : ℝ) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0) 
  (pos_s : s > 0) (pos_t : t > 0) (pos_u : u > 0)
  (sum_eq_10 : p + q + r + s + t + u = 10) :
  1/p + 9/q + 4/r + 16/s + 25/t + 36/u ≥ 44.1 ∧
  ∃ (p' q' r' s' t' u' : ℝ),
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 10 ∧
    1/p' + 9/q' + 4/r' + 16/s' + 25/t' + 36/u' = 44.1 :=
by sorry

end min_value_theorem_l3817_381743


namespace crayon_selection_proof_l3817_381755

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem crayon_selection_proof :
  choose 12 4 = 495 := by
  sorry

end crayon_selection_proof_l3817_381755


namespace quadratic_equation_roots_l3817_381757

theorem quadratic_equation_roots (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 8 * x + 10 = 0 ↔ x = 2 ∨ x = -5/3) → k = 24 :=
by sorry

end quadratic_equation_roots_l3817_381757


namespace inequality_solution_set_l3817_381764

theorem inequality_solution_set (x : ℝ) : 
  (((2 * x - 1) / 3) > ((3 * x - 2) / 2 - 1)) ↔ (x < 2) := by sorry

end inequality_solution_set_l3817_381764


namespace sqrt_114_plus_44_sqrt_6_l3817_381704

theorem sqrt_114_plus_44_sqrt_6 :
  ∃ (x y z : ℤ), (x + y * Real.sqrt z : ℝ) = Real.sqrt (114 + 44 * Real.sqrt 6) ∧
  z > 0 ∧
  (∀ (w : ℤ), w ^ 2 ∣ z → w = 1 ∨ w = -1) ∧
  x = 5 ∧ y = 2 ∧ z = 6 :=
sorry

end sqrt_114_plus_44_sqrt_6_l3817_381704


namespace equation_solution_l3817_381767

theorem equation_solution :
  ∀ x : ℝ, (Real.sqrt (5 * x^3 - 1) + Real.sqrt (x^3 - 1) = 4) ↔ 
  (x = Real.rpow 10 (1/3) ∨ x = Real.rpow 2 (1/3)) :=
by sorry

end equation_solution_l3817_381767


namespace goldbach_2024_l3817_381740

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem goldbach_2024 : ∃ p q : ℕ, 
  is_prime p ∧ 
  is_prime q ∧ 
  p ≠ q ∧ 
  p + q = 2024 :=
sorry

end goldbach_2024_l3817_381740


namespace floor_sqrt_24_squared_l3817_381781

theorem floor_sqrt_24_squared : ⌊Real.sqrt 24⌋^2 = 16 := by
  sorry

end floor_sqrt_24_squared_l3817_381781


namespace election_votes_proof_l3817_381732

theorem election_votes_proof (total_votes : ℕ) 
  (winner_percent : ℚ) (second_percent : ℚ) (third_percent : ℚ)
  (winner_second_diff : ℕ) (winner_third_diff : ℕ) (winner_fourth_diff : ℕ) :
  winner_percent = 2/5 ∧ 
  second_percent = 7/25 ∧ 
  third_percent = 1/5 ∧
  winner_second_diff = 1536 ∧
  winner_third_diff = 3840 ∧
  winner_fourth_diff = 5632 →
  total_votes = 12800 ∧
  (winner_percent * total_votes).num = 5120 ∧
  (second_percent * total_votes).num = 3584 ∧
  (third_percent * total_votes).num = 2560 ∧
  total_votes - (winner_percent * total_votes).num - 
    (second_percent * total_votes).num - 
    (third_percent * total_votes).num = 1536 := by
  sorry

end election_votes_proof_l3817_381732


namespace algebraic_expression_equality_l3817_381714

theorem algebraic_expression_equality (x : ℝ) (h : x^2 + 3*x + 5 = 7) : 
  3*x^2 + 9*x - 2 = 4 := by
  sorry

end algebraic_expression_equality_l3817_381714


namespace pencil_pen_cost_l3817_381713

/-- Given the cost of pencils and pens, calculate the cost of a specific combination -/
theorem pencil_pen_cost (pencil_cost pen_cost : ℝ) 
  (h1 : 4 * pencil_cost + pen_cost = 2.60)
  (h2 : pencil_cost + 3 * pen_cost = 2.15) :
  3 * pencil_cost + 2 * pen_cost = 2.63 := by
  sorry

end pencil_pen_cost_l3817_381713


namespace speed_ratio_eddy_freddy_l3817_381746

/-- Proves that the ratio of Eddy's average speed to Freddy's average speed is 38:15 -/
theorem speed_ratio_eddy_freddy : 
  ∀ (eddy_distance freddy_distance : ℝ) 
    (eddy_time freddy_time : ℝ),
  eddy_distance = 570 →
  freddy_distance = 300 →
  eddy_time = 3 →
  freddy_time = 4 →
  (eddy_distance / eddy_time) / (freddy_distance / freddy_time) = 38 / 15 := by
  sorry


end speed_ratio_eddy_freddy_l3817_381746


namespace add_and_round_to_hundredth_l3817_381731

-- Define the two numbers to be added
def a : Float := 123.456
def b : Float := 78.9102

-- Define the sum of the two numbers
def sum : Float := a + b

-- Define a function to round to the nearest hundredth
def roundToHundredth (x : Float) : Float :=
  (x * 100).round / 100

-- Theorem statement
theorem add_and_round_to_hundredth :
  roundToHundredth sum = 202.37 := by
  sorry

end add_and_round_to_hundredth_l3817_381731


namespace smallest_top_block_exists_l3817_381748

/-- Represents a block in the pyramid --/
structure Block where
  layer : Nat
  value : Nat

/-- Represents the pyramid structure --/
structure Pyramid where
  blocks : List Block
  layer1 : List Nat
  layer2 : List Nat
  layer3 : List Nat
  layer4 : Nat

/-- Check if a pyramid configuration is valid --/
def isValidPyramid (p : Pyramid) : Prop :=
  p.blocks.length = 54 ∧
  p.layer1.length = 30 ∧
  p.layer2.length = 15 ∧
  p.layer3.length = 8 ∧
  ∀ n ∈ p.layer1, 1 ≤ n ∧ n ≤ 30

/-- Calculate the value of a block in an upper layer --/
def calculateBlockValue (below : List Nat) : Nat :=
  below.sum

/-- The main theorem --/
theorem smallest_top_block_exists (p : Pyramid) :
  isValidPyramid p →
  ∃ (minTop : Nat), 
    p.layer4 = minTop ∧
    ∀ (p' : Pyramid), isValidPyramid p' → p'.layer4 ≥ minTop := by
  sorry


end smallest_top_block_exists_l3817_381748


namespace fertilizer_calculation_l3817_381736

theorem fertilizer_calculation (total_area : ℝ) (partial_area : ℝ) (partial_fertilizer : ℝ) 
  (h1 : total_area = 7200)
  (h2 : partial_area = 3600)
  (h3 : partial_fertilizer = 600) :
  (total_area / partial_area) * partial_fertilizer = 1200 := by
sorry

end fertilizer_calculation_l3817_381736


namespace max_income_at_11_l3817_381749

def bicycle_rental (x : ℕ) : ℝ :=
  if x ≤ 6 then 50 * x - 115
  else -3 * x^2 + 68 * x - 115

theorem max_income_at_11 :
  ∀ x : ℕ, 3 ≤ x → x ≤ 20 →
    bicycle_rental x ≤ bicycle_rental 11 := by
  sorry

end max_income_at_11_l3817_381749


namespace equation_solution_l3817_381751

theorem equation_solution : 
  ∀ x : ℝ, (Real.sqrt (4 * x + 10) / Real.sqrt (8 * x + 2) = 2 / Real.sqrt 5) → x = 7 / 2 := by
  sorry

end equation_solution_l3817_381751


namespace yard_area_l3817_381737

theorem yard_area (fence_length : ℝ) (unfenced_side : ℝ) (h1 : fence_length = 64) (h2 : unfenced_side = 40) :
  ∃ (width : ℝ), 
    unfenced_side + 2 * width = fence_length ∧ 
    unfenced_side * width = 480 :=
by sorry

end yard_area_l3817_381737


namespace ice_cream_arrangement_l3817_381717

theorem ice_cream_arrangement (n : ℕ) (h : n = 6) : Nat.factorial n = 720 := by
  sorry

end ice_cream_arrangement_l3817_381717


namespace correct_height_l3817_381738

theorem correct_height (n : ℕ) (initial_avg actual_avg wrong_height : ℝ) :
  n = 35 →
  initial_avg = 180 →
  actual_avg = 178 →
  wrong_height = 156 →
  ∃ (correct_height : ℝ),
    correct_height = n * actual_avg - (n * initial_avg - wrong_height) := by
  sorry

end correct_height_l3817_381738


namespace base9_sum_and_subtract_l3817_381715

/-- Converts a base 9 number represented as a list of digits to a natural number. -/
def base9ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 9 * acc + d) 0

/-- Converts a natural number to its base 9 representation as a list of digits. -/
def natToBase9 (n : Nat) : List Nat :=
  if n < 9 then [n]
  else (n % 9) :: natToBase9 (n / 9)

theorem base9_sum_and_subtract :
  let a := base9ToNat [1, 5, 3]  -- 351₉
  let b := base9ToNat [5, 6, 4]  -- 465₉
  let c := base9ToNat [2, 3, 1]  -- 132₉
  let d := base9ToNat [7, 4, 1]  -- 147₉
  natToBase9 (a + b + c - d) = [7, 4, 8] := by
  sorry

end base9_sum_and_subtract_l3817_381715


namespace proposition_relationship_l3817_381752

theorem proposition_relationship :
  (∀ x : ℝ, (0 < x ∧ x < 5) → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end proposition_relationship_l3817_381752


namespace smallest_cube_multiplier_l3817_381787

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_cube_multiplier (k : ℕ) : 
  (∀ m : ℕ, m < 4500 → ¬ ∃ n : ℕ, m * y = n^3) ∧ 
  (∃ n : ℕ, 4500 * y = n^3) := by
sorry

end smallest_cube_multiplier_l3817_381787


namespace exterior_angle_regular_pentagon_l3817_381703

theorem exterior_angle_regular_pentagon :
  ∀ (exterior_angle : ℝ),
  (exterior_angle = 180 - (540 / 5)) →
  exterior_angle = 72 := by
sorry

end exterior_angle_regular_pentagon_l3817_381703


namespace geometric_arithmetic_sequence_l3817_381735

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  (2 * a 3 = a 1 + a 2) →           -- arithmetic sequence condition
  (q = 1 ∨ q = -1/2) :=             -- conclusion
by sorry

end geometric_arithmetic_sequence_l3817_381735


namespace max_digit_sum_for_reciprocal_decimal_l3817_381774

/-- Given digits a, b, c forming a decimal 0.abc that equals 1/y for some integer y between 1 and 12,
    the sum a + b + c is at most 8. -/
theorem max_digit_sum_for_reciprocal_decimal (a b c y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (0 < y ∧ y ≤ 12) →            -- 0 < y ≤ 12
  (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y →  -- 0.abc = 1/y
  a + b + c ≤ 8 := by
sorry

end max_digit_sum_for_reciprocal_decimal_l3817_381774


namespace function_machine_output_l3817_381723

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 15 then
    step1 + 10
  else
    step1 - 3

theorem function_machine_output : function_machine 12 = 33 := by
  sorry

end function_machine_output_l3817_381723


namespace books_sold_total_l3817_381762

/-- The total number of books sold by three salespeople over three days -/
def total_books_sold (matias_monday olivia_monday luke_monday : ℕ) : ℕ :=
  let matias_tuesday := 2 * matias_monday
  let olivia_tuesday := 3 * olivia_monday
  let luke_tuesday := luke_monday / 2
  let matias_wednesday := 3 * matias_tuesday
  let olivia_wednesday := 4 * olivia_tuesday
  let luke_wednesday := luke_tuesday
  matias_monday + matias_tuesday + matias_wednesday +
  olivia_monday + olivia_tuesday + olivia_wednesday +
  luke_monday + luke_tuesday + luke_wednesday

/-- Theorem stating the total number of books sold by Matias, Olivia, and Luke over three days -/
theorem books_sold_total : total_books_sold 7 5 12 = 167 := by
  sorry

end books_sold_total_l3817_381762


namespace geometric_sequence_formula_l3817_381718

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_formula
  (a : ℕ → ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_geometric : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_relation : ∀ n : ℕ, (a (n + 2))^2 + 4*(a n)^2 = 4*(a (n + 1))^2) :
  ∀ n : ℕ, a n = 2^((n + 1) / 2) :=
by sorry

end geometric_sequence_formula_l3817_381718


namespace divisible_by_11_iff_valid_pair_l3817_381789

def is_valid_pair (a b : Nat) : Prop :=
  (a, b) ∈ [(8, 0), (9, 1), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9)]

def number_from_digits (a b : Nat) : Nat :=
  380000 + a * 1000 + 750 + b

theorem divisible_by_11_iff_valid_pair (a b : Nat) :
  a < 10 ∧ b < 10 →
  (number_from_digits a b) % 11 = 0 ↔ is_valid_pair a b := by
  sorry

end divisible_by_11_iff_valid_pair_l3817_381789


namespace rational_equation_equality_l3817_381778

theorem rational_equation_equality (x : ℝ) (h : x ≠ -1) : 
  (1 / (x + 1)) + (1 / (x + 1)^2) + ((-x - 1) / (x^2 + x + 1)) = 
  1 / ((x + 1)^2 * (x^2 + x + 1)) := by sorry

end rational_equation_equality_l3817_381778


namespace sticker_distribution_l3817_381710

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def stars_and_bars (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute stickers onto sheets --/
def distribute_stickers (total_stickers sheets min_per_sheet : ℕ) : ℕ :=
  stars_and_bars (total_stickers - sheets * min_per_sheet) sheets

theorem sticker_distribution :
  distribute_stickers 10 5 2 = 1 := by sorry

end sticker_distribution_l3817_381710


namespace max_m_value_l3817_381777

theorem max_m_value (m : ℝ) : 
  (∀ x ∈ Set.Icc (-π/4 : ℝ) (π/4 : ℝ), m ≤ Real.tan x + 1) → 
  (∃ M : ℝ, (∀ x ∈ Set.Icc (-π/4 : ℝ) (π/4 : ℝ), M ≤ Real.tan x + 1) ∧ M = 2) :=
by sorry

end max_m_value_l3817_381777


namespace f_range_on_interval_l3817_381797

/-- The function f(x) = 1 - 4x - 2x^2 -/
def f (x : ℝ) : ℝ := 1 - 4*x - 2*x^2

/-- The range of f(x) on the interval (1, +∞) is (-∞, -5) -/
theorem f_range_on_interval :
  Set.range (fun x => f x) ∩ Set.Ioi 1 = Set.Iio (-5) := by sorry

end f_range_on_interval_l3817_381797
