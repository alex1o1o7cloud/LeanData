import Mathlib

namespace airline_seats_per_row_l1154_115471

/-- Proves that the number of seats in each row is 7 for an airline company with given conditions. -/
theorem airline_seats_per_row :
  let num_airplanes : ℕ := 5
  let rows_per_airplane : ℕ := 20
  let flights_per_airplane_per_day : ℕ := 2
  let total_passengers_per_day : ℕ := 1400
  let seats_per_row : ℕ := total_passengers_per_day / (num_airplanes * flights_per_airplane_per_day * rows_per_airplane)
  seats_per_row = 7 := by sorry

end airline_seats_per_row_l1154_115471


namespace mom_shirt_purchase_l1154_115401

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom needs to buy -/
def packages_to_buy : ℚ := 11.83333333

/-- The total number of t-shirts Mom wants to buy -/
def total_shirts : ℕ := 71

theorem mom_shirt_purchase :
  ⌊(packages_to_buy * shirts_per_package : ℚ)⌋ = total_shirts := by
  sorry

end mom_shirt_purchase_l1154_115401


namespace gdp_scientific_notation_correct_l1154_115476

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The GDP value in ten thousand yuan -/
def gdp : ℝ := 84300000

/-- The scientific notation representation of the GDP -/
def gdp_scientific : ScientificNotation := {
  coefficient := 8.43,
  exponent := 7,
  h1 := by sorry
}

/-- Theorem stating that the GDP in scientific notation is correct -/
theorem gdp_scientific_notation_correct : 
  gdp = gdp_scientific.coefficient * (10 : ℝ) ^ gdp_scientific.exponent := by sorry

end gdp_scientific_notation_correct_l1154_115476


namespace scientific_notation_3930_billion_l1154_115418

-- Define billion as 10^9
def billion : ℕ := 10^9

-- Theorem to prove the equality
theorem scientific_notation_3930_billion :
  (3930 : ℝ) * billion = 3.93 * (10 : ℝ)^12 := by
  sorry

end scientific_notation_3930_billion_l1154_115418


namespace half_distance_time_l1154_115428

/-- Represents the total distance of Tony's errands in miles -/
def total_distance : ℝ := 10 + 15 + 5 + 20 + 25

/-- Represents Tony's constant speed in miles per hour -/
def speed : ℝ := 50

/-- Theorem stating that the time taken to drive half the total distance at the given speed is 0.75 hours -/
theorem half_distance_time : (total_distance / 2) / speed = 0.75 := by
  sorry

end half_distance_time_l1154_115428


namespace jenny_money_l1154_115420

theorem jenny_money (original : ℚ) : 
  (4/7 : ℚ) * original = 24 → (1/2 : ℚ) * original = 21 := by
sorry

end jenny_money_l1154_115420


namespace proportion_problem_l1154_115462

theorem proportion_problem (x y : ℚ) 
  (h1 : (3/4 : ℚ) / x = 5 / 7)
  (h2 : y / 19 = 11 / 3) :
  x = 21/20 ∧ y = 209/3 := by
  sorry

end proportion_problem_l1154_115462


namespace robert_interest_l1154_115445

/-- Calculates the total interest earned in a year given an inheritance amount,
    two interest rates, and the amount invested at the higher rate. -/
def total_interest (inheritance : ℝ) (rate1 : ℝ) (rate2 : ℝ) (amount_at_rate2 : ℝ) : ℝ :=
  let amount_at_rate1 := inheritance - amount_at_rate2
  let interest1 := amount_at_rate1 * rate1
  let interest2 := amount_at_rate2 * rate2
  interest1 + interest2

/-- Theorem stating that given Robert's inheritance and investment conditions,
    the total interest earned in a year is $227. -/
theorem robert_interest :
  total_interest 4000 0.05 0.065 1800 = 227 := by
  sorry

end robert_interest_l1154_115445


namespace fraction_inequality_l1154_115473

theorem fraction_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by sorry

end fraction_inequality_l1154_115473


namespace vivian_daily_songs_l1154_115448

/-- The number of songs Vivian plays each day -/
def vivian_songs : ℕ := sorry

/-- The number of songs Clara plays each day -/
def clara_songs : ℕ := sorry

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The number of weekend days in June -/
def weekend_days : ℕ := 8

/-- The total number of songs both listened to in June -/
def total_songs : ℕ := 396

theorem vivian_daily_songs :
  (vivian_songs = 10) ∧
  (clara_songs = vivian_songs - 2) ∧
  (total_songs = (june_days - weekend_days) * (vivian_songs + clara_songs)) := by
  sorry

end vivian_daily_songs_l1154_115448


namespace range_of_a_theorem_l1154_115431

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (a - 2) * x + 1 ≠ 0

def prop_q (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + a * x + 1)

def range_of_a : Set ℝ := Set.Iic (-2) ∪ Set.Ioo 1 2 ∪ Set.Ici 3

theorem range_of_a_theorem :
  (∀ a : ℝ, (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a)) →
  {a : ℝ | prop_p a ∨ prop_q a} = range_of_a := by sorry

end range_of_a_theorem_l1154_115431


namespace right_triangle_area_l1154_115486

theorem right_triangle_area (a b c : ℝ) (h1 : a = 12) (h2 : c = 15) (h3 : a^2 + b^2 = c^2) : 
  (a * b) / 2 = 54 := by
  sorry

end right_triangle_area_l1154_115486


namespace fifth_term_of_sequence_l1154_115421

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 20 = 12 →
  arithmetic_sequence a₁ d 21 = 15 →
  arithmetic_sequence a₁ d 5 = -33 :=
by sorry

end fifth_term_of_sequence_l1154_115421


namespace village_population_l1154_115485

theorem village_population (initial_population : ℕ) 
  (death_rate : ℚ) (leaving_rate : ℚ) : 
  initial_population = 4500 →
  death_rate = 1/10 →
  leaving_rate = 1/5 →
  (initial_population - initial_population * death_rate) * (1 - leaving_rate) = 3240 :=
by sorry

end village_population_l1154_115485


namespace crayon_boxes_l1154_115463

theorem crayon_boxes (total_crayons : ℕ) (crayons_per_box : ℕ) (boxes_needed : ℕ) : 
  total_crayons = 80 → 
  crayons_per_box = 8 → 
  boxes_needed = total_crayons / crayons_per_box →
  boxes_needed = 10 := by
sorry

end crayon_boxes_l1154_115463


namespace person_b_hit_six_shots_l1154_115496

/-- A shooting competition between two people -/
structure ShootingCompetition where
  hits_points : ℕ     -- Points gained for each hit
  miss_points : ℕ     -- Points deducted for each miss
  total_shots : ℕ     -- Total number of shots per person
  total_score : ℕ     -- Combined score of both persons
  score_diff  : ℕ     -- Score difference between person A and B

/-- The number of shots hit by person B in the competition -/
def person_b_hits (comp : ShootingCompetition) : ℕ := 
  sorry

/-- Theorem stating that person B hit 6 shots in the given competition -/
theorem person_b_hit_six_shots 
  (comp : ShootingCompetition) 
  (h1 : comp.hits_points = 20)
  (h2 : comp.miss_points = 12)
  (h3 : comp.total_shots = 10)
  (h4 : comp.total_score = 208)
  (h5 : comp.score_diff = 64) : 
  person_b_hits comp = 6 := by
  sorry

end person_b_hit_six_shots_l1154_115496


namespace room_width_is_seven_l1154_115432

/-- Represents the dimensions and features of a room -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  doorCount : ℕ
  doorArea : ℝ
  largeWindowCount : ℕ
  largeWindowArea : ℝ
  smallWindowCount : ℕ
  smallWindowArea : ℝ
  paintCostPerSqM : ℝ
  totalPaintCost : ℝ

/-- Calculates the paintable area of the room -/
def paintableArea (r : Room) : ℝ :=
  2 * (r.height * r.length + r.height * r.width) -
  (r.doorCount * r.doorArea + r.largeWindowCount * r.largeWindowArea + r.smallWindowCount * r.smallWindowArea)

/-- Theorem stating that the width of the room is 7 meters -/
theorem room_width_is_seven (r : Room) 
  (h1 : r.length = 10)
  (h2 : r.height = 5)
  (h3 : r.doorCount = 2)
  (h4 : r.doorArea = 3)
  (h5 : r.largeWindowCount = 1)
  (h6 : r.largeWindowArea = 3)
  (h7 : r.smallWindowCount = 2)
  (h8 : r.smallWindowArea = 1.5)
  (h9 : r.paintCostPerSqM = 3)
  (h10 : r.totalPaintCost = 474)
  (h11 : paintableArea r * r.paintCostPerSqM = r.totalPaintCost) :
  r.width = 7 := by
  sorry

end room_width_is_seven_l1154_115432


namespace pressure_functions_exist_l1154_115468

/-- Represents the gas pressure in a vessel as a function of time. -/
def PressureFunction := ℝ → ℝ

/-- Represents the parameters of the gas system. -/
structure GasSystem where
  V₁ : ℝ  -- Volume of vessel 1
  V₂ : ℝ  -- Volume of vessel 2
  P₁ : ℝ  -- Initial pressure in vessel 1
  P₂ : ℝ  -- Initial pressure in vessel 2
  a  : ℝ  -- Flow rate coefficient
  b  : ℝ  -- Pressure change coefficient

/-- Defines the conditions for valid pressure functions in the gas system. -/
def ValidPressureFunctions (sys : GasSystem) (p₁ p₂ : PressureFunction) : Prop :=
  -- Initial conditions
  p₁ 0 = sys.P₁ ∧ p₂ 0 = sys.P₂ ∧
  -- Conservation of mass
  ∀ t, sys.V₁ * p₁ t + sys.V₂ * p₂ t = sys.V₁ * sys.P₁ + sys.V₂ * sys.P₂ ∧
  -- Differential equations
  ∀ t, sys.a * (p₁ t ^ 2 - p₂ t ^ 2) = -sys.b * sys.V₁ * (deriv p₁ t) ∧
  ∀ t, sys.a * (p₁ t ^ 2 - p₂ t ^ 2) = sys.b * sys.V₂ * (deriv p₂ t)

/-- Theorem stating the existence of valid pressure functions for a given gas system. -/
theorem pressure_functions_exist (sys : GasSystem) :
  ∃ (p₁ p₂ : PressureFunction), ValidPressureFunctions sys p₁ p₂ := by
  sorry


end pressure_functions_exist_l1154_115468


namespace company_ratio_is_9_47_l1154_115491

/-- Represents the ratio of managers to non-managers in a company -/
structure ManagerRatio where
  managers : ℕ
  non_managers : ℕ

/-- The company's policy for manager to non-manager ratio -/
axiom company_ratio : ManagerRatio

/-- The ratio is constant across all departments -/
axiom ratio_constant (dept1 dept2 : ManagerRatio) : 
  dept1.managers * dept2.non_managers = dept1.non_managers * dept2.managers

/-- In a department with 9 managers, the maximum number of non-managers is 47 -/
axiom max_non_managers : ∃ (dept : ManagerRatio), dept.managers = 9 ∧ dept.non_managers = 47

/-- The company ratio is equal to 9:47 -/
theorem company_ratio_is_9_47 : company_ratio.managers = 9 ∧ company_ratio.non_managers = 47 := by
  sorry

end company_ratio_is_9_47_l1154_115491


namespace more_white_boxes_than_red_l1154_115490

theorem more_white_boxes_than_red (balls_per_box : ℕ) (white_balls : ℕ) (red_balls : ℕ)
  (h1 : balls_per_box = 6)
  (h2 : white_balls = 30)
  (h3 : red_balls = 18) :
  white_balls / balls_per_box - red_balls / balls_per_box = 2 :=
by
  sorry

end more_white_boxes_than_red_l1154_115490


namespace fraction_simplification_l1154_115440

theorem fraction_simplification : 
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 := by sorry

end fraction_simplification_l1154_115440


namespace complex_abs_sum_l1154_115479

theorem complex_abs_sum : Complex.abs (2 - 4 * Complex.I) + Complex.abs (2 + 4 * Complex.I) = 4 * Real.sqrt 5 := by
  sorry

end complex_abs_sum_l1154_115479


namespace triangle_with_squares_sum_l1154_115417

/-- A right-angled triangle with two inscribed squares -/
structure TriangleWithSquares where
  -- The side lengths of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The areas of the inscribed squares
  area_s1 : ℝ
  area_s2 : ℝ
  -- Conditions
  right_angle : c^2 = a^2 + b^2
  inscribed_square1 : area_s1 = 40 * b + 1
  inscribed_square2 : area_s2 = 40 * b
  sum_sides : c = a + b

theorem triangle_with_squares_sum (t : TriangleWithSquares) : t.c = 462 := by
  sorry

end triangle_with_squares_sum_l1154_115417


namespace eric_quarters_count_l1154_115483

/-- The number of dimes Cindy tosses -/
def cindy_dimes : ℕ := 5

/-- The number of nickels Garrick throws -/
def garrick_nickels : ℕ := 8

/-- The number of pennies Ivy drops -/
def ivy_pennies : ℕ := 60

/-- The total amount in the pond in cents -/
def total_cents : ℕ := 200

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of quarters Eric flipped into the pond -/
def eric_quarters : ℕ := 2

theorem eric_quarters_count :
  eric_quarters * quarter_value = 
    total_cents - (cindy_dimes * dime_value + garrick_nickels * nickel_value + ivy_pennies * penny_value) :=
by sorry

end eric_quarters_count_l1154_115483


namespace rhombus_in_quadrilateral_l1154_115434

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Represents a rhombus -/
structure Rhombus :=
  (X Y Z V : Point)

/-- Checks if two line segments are parallel -/
def are_parallel (P1 P2 Q1 Q2 : Point) : Prop :=
  (P2.x - P1.x) * (Q2.y - Q1.y) = (P2.y - P1.y) * (Q2.x - Q1.x)

/-- Checks if a point is inside a quadrilateral -/
def is_inside (P : Point) (Q : Quadrilateral) : Prop :=
  sorry -- Definition of a point being inside a quadrilateral

/-- Main theorem: There exists a rhombus within a given quadrilateral
    such that its sides are parallel to the quadrilateral's diagonals -/
theorem rhombus_in_quadrilateral (ABCD : Quadrilateral) :
  ∃ (XYZV : Rhombus),
    (is_inside XYZV.X ABCD) ∧ (is_inside XYZV.Y ABCD) ∧
    (is_inside XYZV.Z ABCD) ∧ (is_inside XYZV.V ABCD) ∧
    (are_parallel XYZV.X XYZV.Y ABCD.A ABCD.C) ∧
    (are_parallel XYZV.X XYZV.Z ABCD.B ABCD.D) ∧
    (are_parallel XYZV.Y XYZV.Z ABCD.A ABCD.C) ∧
    (are_parallel XYZV.V XYZV.Y ABCD.B ABCD.D) :=
  sorry -- Proof goes here

end rhombus_in_quadrilateral_l1154_115434


namespace real_equal_roots_iff_k_values_l1154_115477

/-- The quadratic equation in question -/
def equation (k x : ℝ) : ℝ := 3 * x^2 - 2 * k * x + 3 * x + 12

/-- Condition for real and equal roots -/
def has_real_equal_roots (k : ℝ) : Prop :=
  ∃ x : ℝ, equation k x = 0 ∧ 
  ∀ y : ℝ, equation k y = 0 → y = x

/-- Theorem stating the values of k for which the equation has real and equal roots -/
theorem real_equal_roots_iff_k_values :
  ∀ k : ℝ, has_real_equal_roots k ↔ (k = -9/2 ∨ k = 15/2) :=
sorry

end real_equal_roots_iff_k_values_l1154_115477


namespace complex_modulus_equality_l1154_115433

theorem complex_modulus_equality : Complex.abs (1/3 - 3*I) = Real.sqrt 82 / 3 := by
  sorry

end complex_modulus_equality_l1154_115433


namespace slope_height_calculation_l1154_115465

theorem slope_height_calculation (slope_ratio : Real) (distance : Real) (height : Real) : 
  slope_ratio = 1 / 2.4 →
  distance = 130 →
  height ^ 2 + (height * 2.4) ^ 2 = distance ^ 2 →
  height = 50 := by
sorry

end slope_height_calculation_l1154_115465


namespace square_side_length_l1154_115435

/-- Given a ribbon of length 78 cm used to make a triangle and a square,
    with the triangle having a perimeter of 46 cm,
    prove that the length of one side of the square is 8 cm. -/
theorem square_side_length (total_ribbon : ℝ) (triangle_perimeter : ℝ) (square_side : ℝ) :
  total_ribbon = 78 ∧ 
  triangle_perimeter = 46 ∧ 
  square_side * 4 = total_ribbon - triangle_perimeter → 
  square_side = 8 := by
  sorry

end square_side_length_l1154_115435


namespace sum_of_roots_quadratic_l1154_115410

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) :
  x₁^2 - 5*x₁ + 4 = 0 →
  x₂^2 - 5*x₂ + 4 = 0 →
  x₁ + x₂ = 5 := by
sorry

end sum_of_roots_quadratic_l1154_115410


namespace dog_catches_fox_dog_catches_fox_at_120m_l1154_115414

/-- The distance at which a dog catches a fox given their jump lengths and frequencies -/
theorem dog_catches_fox (initial_distance : ℝ) (dog_jump : ℝ) (fox_jump : ℝ) 
  (dog_jumps_per_unit : ℕ) (fox_jumps_per_unit : ℕ) : ℝ :=
  let dog_distance_per_unit := dog_jump * dog_jumps_per_unit
  let fox_distance_per_unit := fox_jump * fox_jumps_per_unit
  let net_gain_per_unit := dog_distance_per_unit - fox_distance_per_unit
  let time_units := initial_distance / net_gain_per_unit
  dog_distance_per_unit * time_units

/-- Proof that the dog catches the fox at 120 meters from the starting point -/
theorem dog_catches_fox_at_120m : 
  dog_catches_fox 30 2 1 2 3 = 120 := by
  sorry

end dog_catches_fox_dog_catches_fox_at_120m_l1154_115414


namespace nate_optimal_speed_l1154_115419

/-- The speed at which Nate should drive to arrive just in time -/
def optimal_speed : ℝ := 48

/-- The time it takes for Nate to arrive on time -/
def on_time : ℝ := 5

/-- The distance Nate needs to travel -/
def distance : ℝ := 240

theorem nate_optimal_speed :
  (distance = 40 * (on_time + 1)) ∧
  (distance = 60 * (on_time - 1)) →
  optimal_speed = distance / on_time :=
by sorry

end nate_optimal_speed_l1154_115419


namespace circle_radius_l1154_115424

theorem circle_radius (x y : ℝ) :
  x > 0 ∧ y > 0 ∧ 
  (∃ r : ℝ, r > 0 ∧ x = π * r^2 ∧ y = 2 * π * r) ∧
  x + y = 72 * π →
  ∃ r : ℝ, r = 6 ∧ x = π * r^2 ∧ y = 2 * π * r :=
by sorry

end circle_radius_l1154_115424


namespace volleyball_team_lineup_count_l1154_115466

def volleyball_team_size : ℕ := 14
def starting_lineup_size : ℕ := 6
def triplet_size : ℕ := 3

-- Define a function to calculate combinations
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem volleyball_team_lineup_count :
  choose volleyball_team_size starting_lineup_size -
  choose (volleyball_team_size - triplet_size) (starting_lineup_size - triplet_size) = 2838 :=
sorry

end volleyball_team_lineup_count_l1154_115466


namespace fifth_girl_siblings_l1154_115426

def number_set : List ℕ := [1, 6, 10, 4, 3, 11, 3, 10]

theorem fifth_girl_siblings (mean : ℚ) (h1 : mean = 57/10) 
  (h2 : (number_set.sum + x) / 9 = mean) : x = 3 :=
sorry

end fifth_girl_siblings_l1154_115426


namespace function_composition_equality_l1154_115412

theorem function_composition_equality 
  (m n p q : ℝ) 
  (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = m * x + n) 
  (hg : ∀ x, g x = p * x + q) : 
  (∀ x, f (g x) = g (f x)) ↔ m + q = n + p := by
sorry

end function_composition_equality_l1154_115412


namespace am_gm_difference_bound_l1154_115481

theorem am_gm_difference_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  (x + y) / 2 - Real.sqrt (x * y) ≥ (x - y)^2 / (8 * x) := by
  sorry

end am_gm_difference_bound_l1154_115481


namespace jacksons_grade_l1154_115458

/-- Calculates a student's grade based on study time and grade increase rate -/
def calculate_grade (video_game_hours : ℝ) (study_time_ratio : ℝ) (grade_increase_rate : ℝ) : ℝ :=
  video_game_hours * study_time_ratio * grade_increase_rate

/-- Proves that Jackson's grade is 45 points given the problem conditions -/
theorem jacksons_grade :
  let video_game_hours : ℝ := 9
  let study_time_ratio : ℝ := 1/3
  let grade_increase_rate : ℝ := 15
  calculate_grade video_game_hours study_time_ratio grade_increase_rate = 45 := by
  sorry


end jacksons_grade_l1154_115458


namespace product_of_solutions_l1154_115400

theorem product_of_solutions (x : ℝ) : (|x| = 3 * (|x| - 2)) → ∃ y : ℝ, (|y| = 3 * (|y| - 2)) ∧ x * y = -9 := by
  sorry

end product_of_solutions_l1154_115400


namespace arithmetic_calculations_l1154_115474

theorem arithmetic_calculations :
  ((-5 + 8 - 2 : ℚ) = 1) ∧
  ((-3 * (5/6) / (-1/4) : ℚ) = 10) ∧
  ((-3/17 + (-3.75) + (-14/17) + 3 * (3/4) : ℚ) = -1) ∧
  ((-1^10 - (13/14 - 11/12) * (4 - (-2)^2) + 1/2 / 3 : ℚ) = -5/6) :=
by sorry

end arithmetic_calculations_l1154_115474


namespace crayons_per_day_l1154_115442

def boxes_per_day : ℕ := 45
def crayons_per_box : ℕ := 7

theorem crayons_per_day : boxes_per_day * crayons_per_box = 315 := by
  sorry

end crayons_per_day_l1154_115442


namespace bacteria_population_after_nine_days_l1154_115492

/-- Represents the population of bacteria after a given number of 3-day periods -/
def bacteriaPopulation (initialCount : ℕ) (periods : ℕ) : ℕ :=
  initialCount * (3 ^ periods)

/-- Theorem stating that the bacteria population after 9 days (3 periods) is 36 -/
theorem bacteria_population_after_nine_days :
  bacteriaPopulation 4 3 = 36 := by
  sorry

end bacteria_population_after_nine_days_l1154_115492


namespace picture_area_l1154_115413

/-- The area of a picture on a sheet of paper with given dimensions and margins. -/
theorem picture_area (paper_width paper_length margin : ℝ) 
  (hw : paper_width = 8.5)
  (hl : paper_length = 10)
  (hm : margin = 1.5) : 
  (paper_width - 2 * margin) * (paper_length - 2 * margin) = 38.5 := by
  sorry

end picture_area_l1154_115413


namespace panteleimon_twos_count_l1154_115425

/-- Represents the grades of a student -/
structure Grades :=
  (fives : ℕ)
  (fours : ℕ)
  (threes : ℕ)
  (twos : ℕ)

/-- The total number of grades for each student -/
def total_grades : ℕ := 20

/-- Calculates the average grade -/
def average_grade (g : Grades) : ℚ :=
  (5 * g.fives + 4 * g.fours + 3 * g.threes + 2 * g.twos : ℚ) / total_grades

theorem panteleimon_twos_count 
  (p g : Grades) -- Panteleimon's and Gerasim's grades
  (h1 : p.fives + p.fours + p.threes + p.twos = total_grades)
  (h2 : g.fives + g.fours + g.threes + g.twos = total_grades)
  (h3 : p.fives = g.fours)
  (h4 : p.fours = g.threes)
  (h5 : p.threes = g.twos)
  (h6 : p.twos = g.fives)
  (h7 : average_grade p = average_grade g) :
  p.twos = 5 := by
  sorry

end panteleimon_twos_count_l1154_115425


namespace chocolate_candies_cost_l1154_115461

theorem chocolate_candies_cost (box_size : ℕ) (box_cost : ℚ) (total_candies : ℕ) : 
  box_size = 30 → box_cost = 7.5 → total_candies = 450 → 
  (total_candies / box_size : ℚ) * box_cost = 112.5 := by
sorry

end chocolate_candies_cost_l1154_115461


namespace induction_base_case_not_always_one_l1154_115405

/-- In mathematical induction, the base case is not always n = 1. -/
theorem induction_base_case_not_always_one : ∃ (P : ℕ → Prop) (n₀ : ℕ), 
  n₀ ≠ 1 ∧ (∀ n ≥ n₀, P n → P (n + 1)) → (∀ n ≥ n₀, P n) :=
sorry

end induction_base_case_not_always_one_l1154_115405


namespace book_arrangement_proof_l1154_115427

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem book_arrangement_proof :
  let total_books : ℕ := 9
  let arabic_books : ℕ := 2
  let french_books : ℕ := 3
  let english_books : ℕ := 4
  let arabic_group : ℕ := 1
  let english_group : ℕ := 1
  let total_groups : ℕ := arabic_group + english_group + french_books

  (factorial total_groups) * (factorial arabic_books) * (factorial english_books) = 5760 :=
by sorry

end book_arrangement_proof_l1154_115427


namespace quadratic_sum_l1154_115409

/-- A quadratic function with vertex (h, k) and passing through point (x₀, y₀) -/
def quadratic_function (a b c h k x₀ y₀ : ℝ) : Prop :=
  ∀ x, a * (x - h)^2 + k = a * x^2 + b * x + c ∧
  a * (x₀ - h)^2 + k = y₀

theorem quadratic_sum (a b c : ℝ) :
  quadratic_function a b c 2 3 3 2 →
  a + b + 2 * c = 2 := by
  sorry

#check quadratic_sum

end quadratic_sum_l1154_115409


namespace fraction_zero_implies_x_equals_one_l1154_115472

theorem fraction_zero_implies_x_equals_one (x : ℝ) : (x - 1) / (x + 2) = 0 → x = 1 := by
  sorry

end fraction_zero_implies_x_equals_one_l1154_115472


namespace inscribed_sphere_volume_l1154_115484

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (θ : ℝ) (h_d : d = 12 * Real.sqrt 2) (h_θ : θ = π / 4) :
  let r := d / 4
  (4 / 3) * π * r^3 = 288 * π := by sorry

end inscribed_sphere_volume_l1154_115484


namespace fish_dog_lifespan_difference_l1154_115498

/-- The difference between a fish's lifespan and a dog's lifespan is 2 years -/
theorem fish_dog_lifespan_difference :
  let hamster_lifespan : ℝ := 2.5
  let dog_lifespan : ℝ := 4 * hamster_lifespan
  let fish_lifespan : ℝ := 12
  fish_lifespan - dog_lifespan = 2 := by
  sorry

end fish_dog_lifespan_difference_l1154_115498


namespace problem_solution_l1154_115467

theorem problem_solution (x y z M : ℚ) 
  (sum_eq : x + y + z = 120)
  (x_eq : x - 10 = M)
  (y_eq : y + 10 = M)
  (z_eq : 10 * z = M) :
  M = 400 / 7 := by
  sorry

end problem_solution_l1154_115467


namespace arithmetic_sequence_cos_relation_l1154_115422

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_cos_relation (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 + a 5 + a 9 = 8 * Real.pi → Real.cos (a 3 + a 7) = -1/2 := by
  sorry

end arithmetic_sequence_cos_relation_l1154_115422


namespace workshop_workers_l1154_115482

theorem workshop_workers (total_average : ℕ) (tech_count : ℕ) (tech_average : ℕ) (nontech_average : ℕ) :
  total_average = 8000 →
  tech_count = 10 →
  tech_average = 12000 →
  nontech_average = 6000 →
  ∃ (total_workers : ℕ), 
    total_workers * total_average = tech_count * tech_average + (total_workers - tech_count) * nontech_average ∧
    total_workers = 30 :=
by sorry

end workshop_workers_l1154_115482


namespace min_value_expression_min_value_achievable_l1154_115457

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 4 / (x + y)^2 ≥ 2 * Real.sqrt 2 :=
by sorry

theorem min_value_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^2 + y^2 + 4 / (x + y)^2 = 2 * Real.sqrt 2 :=
by sorry

end min_value_expression_min_value_achievable_l1154_115457


namespace inequality_proof_l1154_115446

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) : x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end inequality_proof_l1154_115446


namespace tens_digit_of_19_power_1987_l1154_115411

theorem tens_digit_of_19_power_1987 : ∃ n : ℕ, 19^1987 ≡ 30 + n [ZMOD 100] :=
sorry

end tens_digit_of_19_power_1987_l1154_115411


namespace right_triangle_altitude_reciprocal_square_sum_l1154_115436

/-- Given a right triangle with legs a and b, hypotenuse c, and altitude h drawn to the hypotenuse,
    prove that 1/h^2 = 1/a^2 + 1/b^2. -/
theorem right_triangle_altitude_reciprocal_square_sum 
  (a b c h : ℝ) 
  (h_positive : h > 0)
  (a_positive : a > 0)
  (b_positive : b > 0)
  (right_triangle : a^2 + b^2 = c^2)
  (altitude_formula : h * c = a * b) : 
  1 / h^2 = 1 / a^2 + 1 / b^2 := by
sorry

end right_triangle_altitude_reciprocal_square_sum_l1154_115436


namespace difference_quotient_of_f_l1154_115455

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 1

-- State the theorem
theorem difference_quotient_of_f (Δx : ℝ) :
  let y1 := f 1
  let y2 := f (1 + Δx)
  (y2 - y1) / Δx = 4 + 2 * Δx := by
  sorry

end difference_quotient_of_f_l1154_115455


namespace maximum_mark_calculation_maximum_mark_is_500_l1154_115475

theorem maximum_mark_calculation (passing_threshold : ℝ) (student_score : ℕ) (failure_margin : ℕ) : ℝ :=
  let passing_mark : ℕ := student_score + failure_margin
  let maximum_mark : ℝ := passing_mark / passing_threshold
  maximum_mark

theorem maximum_mark_is_500 :
  maximum_mark_calculation 0.33 125 40 = 500 := by
  sorry

end maximum_mark_calculation_maximum_mark_is_500_l1154_115475


namespace cubic_inequality_l1154_115460

theorem cubic_inequality (x : ℝ) : x^3 - 9*x^2 + 36*x > -16*x ↔ x > 0 := by
  sorry

end cubic_inequality_l1154_115460


namespace zhuoma_combinations_l1154_115454

/-- The number of different styles of backpacks -/
def num_backpack_styles : ℕ := 2

/-- The number of different styles of pencil cases -/
def num_pencil_case_styles : ℕ := 2

/-- The number of different combinations of backpack and pencil case styles -/
def num_combinations : ℕ := num_backpack_styles * num_pencil_case_styles

theorem zhuoma_combinations :
  num_combinations = 4 :=
by sorry

end zhuoma_combinations_l1154_115454


namespace no_non_zero_integer_solution_l1154_115439

theorem no_non_zero_integer_solution :
  ∀ (a b c n : ℤ), 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end no_non_zero_integer_solution_l1154_115439


namespace wall_length_proof_l1154_115450

/-- Given a square mirror and a rectangular wall, prove the wall's length. -/
theorem wall_length_proof (mirror_side : ℕ) (wall_width : ℕ) : 
  mirror_side = 54 →
  wall_width = 68 →
  (mirror_side * mirror_side : ℕ) * 2 = wall_width * (wall_width * 2 - wall_width % 2) →
  wall_width * 2 - wall_width % 2 = 86 :=
by
  sorry

end wall_length_proof_l1154_115450


namespace investment_growth_l1154_115452

/-- The initial investment amount that grows to $563.35 after 5 years at 12% annual interest rate compounded yearly. -/
def initial_investment : ℝ := 319.77

/-- The final amount after 5 years of investment. -/
def final_amount : ℝ := 563.35

/-- The annual interest rate as a decimal. -/
def interest_rate : ℝ := 0.12

/-- The number of years the money is invested. -/
def years : ℕ := 5

/-- Theorem stating that the initial investment grows to the final amount after the specified time and interest rate. -/
theorem investment_growth :
  final_amount = initial_investment * (1 + interest_rate) ^ years := by
  sorry

#eval initial_investment

end investment_growth_l1154_115452


namespace toy_store_shelves_l1154_115402

/-- Calculates the number of shelves needed to display stuffed bears in a toy store. -/
def shelves_needed (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Proves that given the initial conditions, the number of shelves needed is 4. -/
theorem toy_store_shelves :
  shelves_needed 6 18 6 = 4 := by
  sorry

end toy_store_shelves_l1154_115402


namespace africa_passenger_fraction_l1154_115459

theorem africa_passenger_fraction :
  let total_passengers : ℕ := 108
  let north_america_fraction : ℚ := 1 / 12
  let europe_fraction : ℚ := 1 / 4
  let asia_fraction : ℚ := 1 / 6
  let other_continents : ℕ := 42
  let africa_fraction : ℚ := 1 - north_america_fraction - europe_fraction - asia_fraction - (other_continents : ℚ) / total_passengers
  africa_fraction = 1 / 9 := by
  sorry

end africa_passenger_fraction_l1154_115459


namespace min_reciprocal_sum_l1154_115438

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 1 / y) ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 1 / y = 4 :=
sorry

end min_reciprocal_sum_l1154_115438


namespace lizard_to_gecko_ratio_l1154_115416

/-- Represents the number of bugs eaten by each animal -/
structure BugsEaten where
  gecko : ℕ
  lizard : ℕ
  frog : ℕ
  toad : ℕ

/-- Conditions of the bug-eating scenario -/
def bugEatingScenario (b : BugsEaten) : Prop :=
  b.gecko = 12 ∧
  b.frog = 3 * b.lizard ∧
  b.toad = (3 * b.lizard) + (3 * b.lizard) / 2 ∧
  b.gecko + b.lizard + b.frog + b.toad = 63

/-- The ratio of bugs eaten by the lizard to bugs eaten by the gecko is 1:2 -/
theorem lizard_to_gecko_ratio (b : BugsEaten) 
  (h : bugEatingScenario b) : b.lizard * 2 = b.gecko := by
  sorry

#check lizard_to_gecko_ratio

end lizard_to_gecko_ratio_l1154_115416


namespace intersection_A_complement_B_l1154_115453

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x < -1}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {x : ℝ | -1 ≤ x ∧ x < 0} := by sorry

end intersection_A_complement_B_l1154_115453


namespace log_sum_equals_two_l1154_115447

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end log_sum_equals_two_l1154_115447


namespace negative_inequality_l1154_115470

theorem negative_inequality (a b : ℝ) (h : a > b) : -2 - a < -2 - b := by
  sorry

end negative_inequality_l1154_115470


namespace exists_counterexample_l1154_115464

/-- A binary operation on a set S satisfying a * (b * a) = b for all a, b in S -/
class SpecialOperation (S : Type) where
  op : S → S → S
  property : ∀ (a b : S), op a (op b a) = b

/-- Theorem stating that there exist elements a and b in S such that (a*b)*a ≠ a -/
theorem exists_counterexample {S : Type} [SpecialOperation S] [Inhabited S] [Nontrivial S] :
  ∃ (a b : S), (SpecialOperation.op (SpecialOperation.op a b) a) ≠ a := by sorry

end exists_counterexample_l1154_115464


namespace cats_after_sale_l1154_115404

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 15 → house = 49 → sold = 19 → siamese + house - sold = 45 := by
  sorry

end cats_after_sale_l1154_115404


namespace largest_three_digit_multiple_of_12_with_digit_sum_24_l1154_115403

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_12_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n.mod 12 = 0 → digit_sum n = 24 → n ≤ 996 :=
by sorry

end largest_three_digit_multiple_of_12_with_digit_sum_24_l1154_115403


namespace regular_polygon_45_degree_exterior_angle_is_octagon_l1154_115415

/-- A regular polygon with exterior angles of 45° is a regular octagon -/
theorem regular_polygon_45_degree_exterior_angle_is_octagon :
  ∀ (n : ℕ), n > 2 →
  (360 / n : ℚ) = 45 →
  n = 8 :=
by sorry

end regular_polygon_45_degree_exterior_angle_is_octagon_l1154_115415


namespace line_symmetry_l1154_115408

-- Define the lines
def l (x y : ℝ) : Prop := x - y - 1 = 0
def l₁ (x y : ℝ) : Prop := 2*x - y - 2 = 0
def l₂ (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (f g h : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y → ∃ x' y', g x' y' ∧ h x y ∧
    ((x + x') / 2, (y + y') / 2) ∈ {(a, b) | f a b}

-- Theorem statement
theorem line_symmetry :
  symmetric_wrt l l₁ l₂ :=
sorry

end line_symmetry_l1154_115408


namespace difference_of_cubes_factorization_l1154_115499

theorem difference_of_cubes_factorization (t : ℝ) : t^3 - 125 = (t - 5) * (t^2 + 5*t + 25) := by
  sorry

end difference_of_cubes_factorization_l1154_115499


namespace square_root_squared_sqrt_2023_squared_l1154_115497

theorem square_root_squared (x : ℝ) (h : x > 0) : (Real.sqrt x)^2 = x := by sorry

theorem sqrt_2023_squared : (Real.sqrt 2023)^2 = 2023 := by
  apply square_root_squared
  norm_num

end square_root_squared_sqrt_2023_squared_l1154_115497


namespace triangle_existence_l1154_115493

/-- Represents a triangle with side lengths a, b, c and angles α, β, γ. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem stating the existence of a triangle satisfying given conditions. -/
theorem triangle_existence (d β γ : ℝ) : 
  ∃ (t : Triangle), 
    t.b + t.c - t.a = d ∧ 
    t.β = β ∧ 
    t.γ = γ ∧
    t.α + t.β + t.γ = π :=
sorry

end triangle_existence_l1154_115493


namespace largest_n_divisible_by_seven_n_99996_satisfies_condition_n_99996_is_largest_l1154_115480

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 100000 ∧ 
  (9 * (n - 3)^6 - 3 * n^3 + 21 * n - 42) % 7 = 0 →
  n ≤ 99996 :=
by sorry

theorem n_99996_satisfies_condition : 
  (9 * (99996 - 3)^6 - 3 * 99996^3 + 21 * 99996 - 42) % 7 = 0 :=
by sorry

theorem n_99996_is_largest :
  ∀ m : ℕ, m < 100000 ∧ 
  (9 * (m - 3)^6 - 3 * m^3 + 21 * m - 42) % 7 = 0 →
  m ≤ 99996 :=
by sorry

end largest_n_divisible_by_seven_n_99996_satisfies_condition_n_99996_is_largest_l1154_115480


namespace bullseye_mean_hits_l1154_115495

/-- The mean number of hits in a series of independent Bernoulli trials -/
def meanHits (p : ℝ) (n : ℕ) : ℝ := n * p

/-- The probability of hitting the bullseye -/
def bullseyeProbability : ℝ := 0.9

/-- The number of consecutive shots -/
def numShots : ℕ := 10

theorem bullseye_mean_hits :
  meanHits bullseyeProbability numShots = 9 := by
  sorry

end bullseye_mean_hits_l1154_115495


namespace total_profit_is_35000_l1154_115443

/-- Represents the subscription amounts and profit for a business venture -/
structure BusinessVenture where
  total_subscription : ℕ
  a_extra : ℕ
  b_extra : ℕ
  a_profit : ℕ

/-- Calculates the total profit given a BusinessVenture -/
def calculate_total_profit (bv : BusinessVenture) : ℕ :=
  sorry

/-- Theorem stating that for the given business venture, the total profit is 35000 -/
theorem total_profit_is_35000 : 
  let bv : BusinessVenture := {
    total_subscription := 50000,
    a_extra := 4000,
    b_extra := 5000,
    a_profit := 14700
  }
  calculate_total_profit bv = 35000 := by
  sorry

end total_profit_is_35000_l1154_115443


namespace gcd_of_44_33_55_l1154_115456

/-- The greatest common divisor of 44, 33, and 55 is 11. -/
theorem gcd_of_44_33_55 : Nat.gcd 44 (Nat.gcd 33 55) = 11 := by
  sorry

end gcd_of_44_33_55_l1154_115456


namespace right_triangle_ratio_square_l1154_115441

theorem right_triangle_ratio_square (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) 
  (h4 : c^2 = a^2 + b^2) (h5 : a / b = b / c) : (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end right_triangle_ratio_square_l1154_115441


namespace keith_bought_four_digimon_packs_l1154_115487

/-- The number of Digimon card packs Keith bought -/
def num_digimon_packs : ℕ := 4

/-- The cost of each Digimon card pack in dollars -/
def digimon_pack_cost : ℝ := 4.45

/-- The cost of a deck of baseball cards in dollars -/
def baseball_deck_cost : ℝ := 6.06

/-- The total amount spent in dollars -/
def total_spent : ℝ := 23.86

/-- Theorem stating that Keith bought 4 packs of Digimon cards -/
theorem keith_bought_four_digimon_packs :
  (num_digimon_packs : ℝ) * digimon_pack_cost + baseball_deck_cost = total_spent :=
by sorry

end keith_bought_four_digimon_packs_l1154_115487


namespace cleaning_payment_l1154_115488

theorem cleaning_payment (rate : ℚ) (rooms : ℚ) : 
  rate = 12 / 3 → rooms = 9 / 4 → rate * rooms = 9 := by
  sorry

end cleaning_payment_l1154_115488


namespace infinitely_many_primes_composite_l1154_115406

theorem infinitely_many_primes_composite (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ¬Nat.Prime (a * p + b)} :=
sorry

end infinitely_many_primes_composite_l1154_115406


namespace find_a_interest_rate_l1154_115407

-- Define constants
def total_amount : ℝ := 10000
def years : ℝ := 2
def b_interest_rate : ℝ := 18
def interest_difference : ℝ := 360
def b_amount : ℝ := 4000

-- Define variables
variable (a_amount : ℝ) (a_interest_rate : ℝ)

-- Theorem statement
theorem find_a_interest_rate :
  a_amount + b_amount = total_amount →
  (a_amount * a_interest_rate * years) / 100 = (b_amount * b_interest_rate * years) / 100 + interest_difference →
  a_interest_rate = 15 := by
  sorry

end find_a_interest_rate_l1154_115407


namespace intersection_when_m_zero_necessary_not_sufficient_condition_l1154_115478

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | (x - m + 1)*(x - m - 1) > 0}

theorem intersection_when_m_zero :
  A ∩ B 0 = {x : ℝ | 1 < x ∧ x ≤ 3} := by sorry

theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) ↔ m < -2 ∨ m > 4 := by sorry

end intersection_when_m_zero_necessary_not_sufficient_condition_l1154_115478


namespace geometric_sequence_property_l1154_115489

/-- A geometric sequence with a_m = 3 and a_{m+6} = 24 -/
def GeometricSequence (a : ℕ → ℝ) (m : ℕ) : Prop :=
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) ∧ 
  a m = 3 ∧ 
  a (m + 6) = 24

theorem geometric_sequence_property (a : ℕ → ℝ) (m : ℕ) 
  (h : GeometricSequence a m) : 
  a (m + 18) = 1536 := by
  sorry

end geometric_sequence_property_l1154_115489


namespace min_faces_two_dice_l1154_115430

structure Dice where
  faces : ℕ
  min_faces : ℕ
  distinct_numbering : faces ≥ min_faces

def probability_sum (a b : Dice) (sum : ℕ) : ℚ :=
  (Finset.filter (fun (x, y) => x + y = sum) (Finset.product (Finset.range a.faces) (Finset.range b.faces))).card /
  (a.faces * b.faces : ℚ)

theorem min_faces_two_dice (a b : Dice) : 
  a.min_faces = 7 → 
  b.min_faces = 5 → 
  probability_sum a b 13 = 2 * probability_sum a b 8 →
  probability_sum a b 16 = 1/20 →
  a.faces + b.faces ≥ 24 ∧ 
  ∀ (a' b' : Dice), a'.faces + b'.faces < 24 → 
    (a'.min_faces = 7 ∧ b'.min_faces = 5 ∧ 
     probability_sum a' b' 13 = 2 * probability_sum a' b' 8 ∧
     probability_sum a' b' 16 = 1/20) → False :=
by sorry

end min_faces_two_dice_l1154_115430


namespace common_roots_imply_a_b_values_l1154_115494

-- Define the two cubic polynomials
def p (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 11*x + 6
def q (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + 14*x + 8

-- Define a predicate for having two distinct common roots
def has_two_distinct_common_roots (a b : ℝ) : Prop :=
  ∃ (r s : ℝ), r ≠ s ∧ p a r = 0 ∧ p a s = 0 ∧ q b r = 0 ∧ q b s = 0

-- State the theorem
theorem common_roots_imply_a_b_values :
  ∀ a b : ℝ, has_two_distinct_common_roots a b → (a = 6 ∧ b = 7) :=
by sorry

end common_roots_imply_a_b_values_l1154_115494


namespace logical_judgment_structures_l1154_115449

-- Define the basic structures of algorithms
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

-- Define a property for structures that require logical judgment
def RequiresLogicalJudgment (s : AlgorithmStructure) : Prop :=
  match s with
  | AlgorithmStructure.Conditional => True
  | AlgorithmStructure.Loop => True
  | _ => False

-- Theorem statement
theorem logical_judgment_structures :
  ∀ s : AlgorithmStructure,
    RequiresLogicalJudgment s ↔ (s = AlgorithmStructure.Conditional ∨ s = AlgorithmStructure.Loop) :=
by sorry

end logical_judgment_structures_l1154_115449


namespace functional_equation_problem_l1154_115469

theorem functional_equation_problem (f : ℝ → ℝ) 
  (h : ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3)
  (h1 : f 1 = 1)
  (h4 : f 4 = 7) :
  f 2022 = 4043 := by
  sorry

end functional_equation_problem_l1154_115469


namespace ones_digit_largest_power_of_three_dividing_factorial_ones_digit_largest_power_of_three_dividing_27_factorial_l1154_115437

theorem ones_digit_largest_power_of_three_dividing_factorial : ℕ → Prop :=
  fun n => 
    let factorial := Nat.factorial n
    let largest_power := Nat.log 3 factorial
    (3^largest_power % 10 = 3 ∧ n = 27)

-- The proof
theorem ones_digit_largest_power_of_three_dividing_27_factorial :
  ones_digit_largest_power_of_three_dividing_factorial 27 := by
  sorry

end ones_digit_largest_power_of_three_dividing_factorial_ones_digit_largest_power_of_three_dividing_27_factorial_l1154_115437


namespace equation_equivalence_and_domain_x_domain_l1154_115423

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  x = (2 * y + 1) / (y - 2)

-- Define the inverted equation
def inverted_equation (x y : ℝ) : Prop :=
  y = (2 * x + 1) / (x - 2)

-- Theorem stating the equivalence of the equations and the domain of x
theorem equation_equivalence_and_domain :
  ∀ x y : ℝ, original_equation x y ↔ (inverted_equation x y ∧ x ≠ 2) :=
by
  sorry

-- Theorem stating the domain of x
theorem x_domain : ∀ x : ℝ, (∃ y : ℝ, original_equation x y) → x ≠ 2 :=
by
  sorry

end equation_equivalence_and_domain_x_domain_l1154_115423


namespace total_sheets_l1154_115451

-- Define the number of brown and yellow sheets
def brown_sheets : ℕ := 28
def yellow_sheets : ℕ := 27

-- Theorem to prove
theorem total_sheets : brown_sheets + yellow_sheets = 55 := by
  sorry

end total_sheets_l1154_115451


namespace sqrt_27_div_sqrt_3_eq_3_l1154_115444

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end sqrt_27_div_sqrt_3_eq_3_l1154_115444


namespace average_salary_feb_to_may_l1154_115429

theorem average_salary_feb_to_may (
  avg_jan_to_apr : ℝ) 
  (salary_may : ℝ)
  (salary_jan : ℝ)
  (h1 : avg_jan_to_apr = 8000)
  (h2 : salary_may = 6500)
  (h3 : salary_jan = 4700) :
  (4 * avg_jan_to_apr - salary_jan + salary_may) / 4 = 8450 := by
  sorry

end average_salary_feb_to_may_l1154_115429
