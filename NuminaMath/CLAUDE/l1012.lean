import Mathlib

namespace samuel_money_left_l1012_101242

/-- Calculates the amount Samuel has left after receiving a share of the total amount and spending on drinks -/
def samuel_remaining_money (total : ℝ) (share_fraction : ℝ) (spend_fraction : ℝ) : ℝ :=
  total * share_fraction - total * spend_fraction

/-- Theorem stating that given the conditions in the problem, Samuel has $132 left -/
theorem samuel_money_left :
  let total : ℝ := 240
  let share_fraction : ℝ := 3/4
  let spend_fraction : ℝ := 1/5
  samuel_remaining_money total share_fraction spend_fraction = 132 := by
  sorry


end samuel_money_left_l1012_101242


namespace birth_year_problem_l1012_101278

theorem birth_year_problem : ∃! x : ℕ, x ∈ Finset.range 50 ∧ x^2 - x = 1892 := by
  sorry

end birth_year_problem_l1012_101278


namespace area_bisector_l1012_101299

/-- A polygon in the xy-plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- The polygon described in the problem -/
def problemPolygon : Polygon :=
  { vertices := [(0, 0), (0, 4), (4, 4), (4, 2), (6, 2), (6, 0)] }

/-- Calculate the area of a polygon -/
def area (p : Polygon) : ℝ := sorry

/-- Calculate the area of a polygon on one side of a line y = mx passing through the origin -/
def areaOneSide (p : Polygon) (m : ℝ) : ℝ := sorry

/-- The main theorem -/
theorem area_bisector (p : Polygon) :
  p = problemPolygon →
  areaOneSide p (5/3) = (area p) / 2 := by
  sorry

end area_bisector_l1012_101299


namespace smallest_x_absolute_value_equation_l1012_101276

theorem smallest_x_absolute_value_equation :
  ∀ x : ℝ, |x - 3| = 8 → x ≥ -5 ∧ |-5 - 3| = 8 :=
by sorry

end smallest_x_absolute_value_equation_l1012_101276


namespace bauble_painting_friends_l1012_101236

/-- The number of friends needed to complete the bauble painting task -/
def friends_needed (total_baubles : ℕ) (total_colors : ℕ) (first_group_colors : ℕ) 
  (second_group_colors : ℕ) (baubles_per_hour : ℕ) (available_hours : ℕ) : ℕ :=
  let first_group_baubles_per_color := total_baubles / (first_group_colors + 2 * second_group_colors)
  let second_group_baubles_per_color := 2 * first_group_baubles_per_color
  let baubles_per_hour_needed := total_baubles / available_hours
  baubles_per_hour_needed / baubles_per_hour

theorem bauble_painting_friends (total_baubles : ℕ) (total_colors : ℕ) (first_group_colors : ℕ) 
  (second_group_colors : ℕ) (baubles_per_hour : ℕ) (available_hours : ℕ) 
  (h1 : total_baubles = 1000)
  (h2 : total_colors = 20)
  (h3 : first_group_colors = 15)
  (h4 : second_group_colors = 5)
  (h5 : baubles_per_hour = 10)
  (h6 : available_hours = 50)
  (h7 : first_group_colors + second_group_colors = total_colors) :
  friends_needed total_baubles total_colors first_group_colors second_group_colors baubles_per_hour available_hours = 2 := by
  sorry

end bauble_painting_friends_l1012_101236


namespace cereal_eating_time_l1012_101272

theorem cereal_eating_time (fat_rate mr_thin_rate : ℚ) (total_cereal : ℚ) : 
  fat_rate = 1 / 25 →
  mr_thin_rate = 1 / 40 →
  total_cereal = 5 →
  (total_cereal / (fat_rate + mr_thin_rate) : ℚ) = 1000 / 13 := by
  sorry

end cereal_eating_time_l1012_101272


namespace inverse_variation_result_l1012_101252

/-- Given that c² varies inversely with d⁴, this function represents their relationship -/
def inverse_relation (k : ℝ) (c d : ℝ) : Prop :=
  c^2 * d^4 = k

theorem inverse_variation_result (k : ℝ) :
  inverse_relation k 8 2 →
  inverse_relation k c 4 →
  c^2 = 4 := by
  sorry

#check inverse_variation_result

end inverse_variation_result_l1012_101252


namespace fixed_point_of_exponential_function_l1012_101259

/-- The function f(x) = 2a^(x+1) - 3 has a fixed point at (-1, -1) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x + 1) - 3
  f (-1) = -1 := by
sorry

end fixed_point_of_exponential_function_l1012_101259


namespace f_one_root_f_odd_when_c_zero_f_symmetric_f_more_than_two_roots_l1012_101289

-- Define the function f
def f (x b c : ℝ) : ℝ := |x| * x + b * x + c

-- Statement 1
theorem f_one_root (c : ℝ) (h : c > 0) : 
  ∃! x, f x 0 c = 0 := by sorry

-- Statement 2
theorem f_odd_when_c_zero (b : ℝ) :
  ∀ x, f (-x) b 0 = -(f x b 0) := by sorry

-- Statement 3
theorem f_symmetric (b c : ℝ) :
  ∀ x, f x b c = f (-x) b c := by sorry

-- Statement 4
theorem f_more_than_two_roots :
  ∃ b c, ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0 := by sorry

end f_one_root_f_odd_when_c_zero_f_symmetric_f_more_than_two_roots_l1012_101289


namespace union_and_complement_of_sets_l1012_101233

-- Define the sets A and B
def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem union_and_complement_of_sets :
  (A ∪ B = {x | x ≤ 3 ∨ x > 4}) ∧
  ((Set.univ \ A) ∪ (Set.univ \ B) = {x | x < -4 ∨ x ≥ -1}) := by
  sorry

end union_and_complement_of_sets_l1012_101233


namespace square_condition_l1012_101239

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k^2

theorem square_condition (n : ℕ) :
  n > 0 → (is_perfect_square ((n^2 + 11*n - 4) * n.factorial + 33 * 13^n + 4) ↔ n = 1 ∨ n = 2) :=
by sorry

end square_condition_l1012_101239


namespace inequality_proof_l1012_101269

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
by sorry

end inequality_proof_l1012_101269


namespace union_of_M_and_N_l1012_101287

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by sorry

end union_of_M_and_N_l1012_101287


namespace power_of_two_equation_l1012_101255

theorem power_of_two_equation (r : ℤ) : 
  2^2001 - 2^2000 - 2^1999 + 2^1998 = r * 2^1998 → r = 3 := by
  sorry

end power_of_two_equation_l1012_101255


namespace cycle_original_price_l1012_101286

/-- Given a cycle sold at a 20% loss for Rs. 1120, prove that the original price was Rs. 1400 -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1120)
  (h2 : loss_percentage = 20) : 
  ∃ (original_price : ℝ), 
    original_price = 1400 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end cycle_original_price_l1012_101286


namespace rotated_angle_measure_l1012_101225

/-- Given an angle of 60 degrees rotated 600 degrees clockwise, 
    the resulting acute angle measure is 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 → 
  rotation = 600 → 
  (initial_angle - (rotation % 360)) % 360 = 60 := by
  sorry

end rotated_angle_measure_l1012_101225


namespace lawnmower_value_drop_l1012_101273

/-- Proves the percentage drop in lawnmower value after 6 months -/
theorem lawnmower_value_drop (initial_value : ℝ) (final_value : ℝ) (yearly_drop_percent : ℝ) :
  initial_value = 100 →
  final_value = 60 →
  yearly_drop_percent = 20 →
  final_value = initial_value * (1 - yearly_drop_percent / 100) →
  (initial_value - (final_value / (1 - yearly_drop_percent / 100))) / initial_value * 100 = 25 := by
  sorry

#check lawnmower_value_drop

end lawnmower_value_drop_l1012_101273


namespace largest_angle_in_ratio_triangle_l1012_101274

/-- A triangle with interior angles in the ratio 1:2:3 has its largest angle equal to 90 degrees -/
theorem largest_angle_in_ratio_triangle : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  b = 2 * a →
  c = 3 * a →
  a + b + c = 180 →
  c = 90 := by
sorry

end largest_angle_in_ratio_triangle_l1012_101274


namespace coordinate_proof_l1012_101281

/-- 
Given two points A(x₁, y₁) and B(x₂, y₂) in the first quadrant of a Cartesian coordinate system,
prove that under certain conditions, their coordinates are (1, 5) and (8, 9) respectively.
-/
theorem coordinate_proof (x₁ y₁ x₂ y₂ : ℕ) : 
  -- Both coordinates are positive integers
  0 < x₁ ∧ 0 < y₁ ∧ 0 < x₂ ∧ 0 < y₂ →
  -- Angle OA > 45°
  y₁ > x₁ →
  -- Angle OB < 45°
  x₂ > y₂ →
  -- Area difference condition
  x₂ * y₂ = x₁ * y₁ + 67 →
  -- Conclusion: coordinates are (1, 5) and (8, 9)
  x₁ = 1 ∧ y₁ = 5 ∧ x₂ = 8 ∧ y₂ = 9 := by
sorry

end coordinate_proof_l1012_101281


namespace tennis_ball_difference_l1012_101205

/-- Given the number of tennis balls for Brian, Frodo, and Lily, prove that Frodo has 8 more tennis balls than Lily. -/
theorem tennis_ball_difference (brian frodo lily : ℕ) : 
  brian = 2 * frodo → 
  lily = 3 → 
  brian = 22 → 
  frodo - lily = 8 := by
  sorry

end tennis_ball_difference_l1012_101205


namespace high_school_students_l1012_101277

theorem high_school_students (total_students : ℕ) : 
  (total_students * 40 / 100 : ℕ) * 70 / 100 = 140 → 
  total_students = 500 := by
sorry

end high_school_students_l1012_101277


namespace statement_holds_for_given_numbers_l1012_101283

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def given_numbers : List ℕ := [45, 54, 63, 81]

theorem statement_holds_for_given_numbers :
  ∀ n ∈ given_numbers, (sum_of_digits n) % 9 = 0 → n % 9 = 0 := by
  sorry

end statement_holds_for_given_numbers_l1012_101283


namespace ordered_pairs_1806_l1012_101243

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def count_ordered_pairs (n : ℕ) : ℕ := sorry

theorem ordered_pairs_1806 :
  prime_factorization 1806 = [(2, 1), (3, 2), (101, 1)] →
  count_ordered_pairs 1806 = 12 := by
  sorry

end ordered_pairs_1806_l1012_101243


namespace student_ranking_l1012_101208

theorem student_ranking (total : Nat) (rank_right : Nat) (rank_left : Nat) : 
  total = 31 → rank_right = 21 → rank_left = total - rank_right + 1 → rank_left = 11 := by
  sorry

end student_ranking_l1012_101208


namespace satellite_upgraded_fraction_l1012_101285

/-- Represents a satellite with modular units and sensors. -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on a satellite. -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

theorem satellite_upgraded_fraction :
  ∀ s : Satellite,
    s.units = 24 →
    s.non_upgraded_per_unit * 6 = s.total_upgraded →
    upgraded_fraction s = 1 / 5 := by
  sorry

end satellite_upgraded_fraction_l1012_101285


namespace circle_proof_l1012_101240

-- Define the points
def A : ℝ × ℝ := (5, 2)
def B : ℝ × ℝ := (3, 2)
def O : ℝ × ℝ := (0, 0)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Define the circle equation for the first part
def circle_eq1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 10

-- Define the circle equation for the second part
def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

theorem circle_proof :
  -- Part 1
  (∀ x y : ℝ, circle_eq1 x y ↔ 
    ((x, y) = A ∨ (x, y) = B) ∧ 
    (∃ cx cy : ℝ, line_eq cx cy ∧ (x - cx)^2 + (y - cy)^2 = (5 - cx)^2 + (2 - cy)^2)) ∧
  -- Part 2
  (∀ x y : ℝ, circle_eq2 x y ↔ 
    ((x, y) = O ∨ (x, y) = (2, 0) ∨ (x, y) = (0, 4)) ∧ 
    (∃ cx cy r : ℝ, (x - cx)^2 + (y - cy)^2 = r^2 ∧ 
                    (0 - cx)^2 + (0 - cy)^2 = r^2 ∧ 
                    (2 - cx)^2 + (0 - cy)^2 = r^2 ∧ 
                    (0 - cx)^2 + (4 - cy)^2 = r^2)) := by
  sorry

end circle_proof_l1012_101240


namespace sunset_colors_proof_l1012_101271

/-- The number of colors the sky turns during a sunset --/
def sunset_colors (sunset_duration : ℕ) (color_change_interval : ℕ) : ℕ :=
  sunset_duration / color_change_interval

theorem sunset_colors_proof (hours : ℕ) (minutes_per_hour : ℕ) (color_change_interval : ℕ) :
  hours = 2 →
  minutes_per_hour = 60 →
  color_change_interval = 10 →
  sunset_colors (hours * minutes_per_hour) color_change_interval = 12 := by
  sorry

end sunset_colors_proof_l1012_101271


namespace negation_equivalence_l1012_101228

/-- A triangle is a geometric shape with three sides and three angles. -/
structure Triangle where
  -- We don't need to define the specifics of a triangle for this problem

/-- An angle is obtuse if it is greater than 90 degrees. -/
def is_obtuse_angle (angle : ℝ) : Prop := angle > 90

/-- The original statement: Every triangle has at least two obtuse angles. -/
def original_statement : Prop :=
  ∀ t : Triangle, ∃ a b : ℝ, is_obtuse_angle a ∧ is_obtuse_angle b ∧ a ≠ b

/-- The negation: There exists a triangle that has at most one obtuse angle. -/
def negation : Prop :=
  ∃ t : Triangle, ∀ a b : ℝ, is_obtuse_angle a ∧ is_obtuse_angle b → a = b

/-- The negation of the original statement is equivalent to the given negation. -/
theorem negation_equivalence : ¬original_statement ↔ negation := by sorry

end negation_equivalence_l1012_101228


namespace bruce_son_age_l1012_101229

/-- Bruce's current age -/
def bruce_age : ℕ := 36

/-- Number of years in the future -/
def years_future : ℕ := 6

/-- Bruce's son's current age -/
def son_age : ℕ := 8

theorem bruce_son_age :
  (bruce_age + years_future) = 3 * (son_age + years_future) :=
sorry

end bruce_son_age_l1012_101229


namespace complex_inequality_complex_inequality_equality_complex_inequality_equality_at_one_l1012_101203

theorem complex_inequality (z : ℂ) : Complex.abs z ^ 2 + 2 * Complex.abs (z - 1) ≥ 1 :=
by sorry

theorem complex_inequality_equality : ∃ z : ℂ, Complex.abs z ^ 2 + 2 * Complex.abs (z - 1) = 1 :=
by sorry

theorem complex_inequality_equality_at_one : Complex.abs (1 : ℂ) ^ 2 + 2 * Complex.abs (1 - 1) = 1 :=
by sorry

end complex_inequality_complex_inequality_equality_complex_inequality_equality_at_one_l1012_101203


namespace eddy_rate_is_correct_l1012_101294

/-- Represents the climbing scenario of Hillary and Eddy on Mt. Everest -/
structure ClimbingScenario where
  summit_distance : ℝ  -- Distance from base camp to summit in feet
  hillary_rate : ℝ     -- Hillary's climbing rate in ft/hr
  hillary_stop : ℝ     -- Distance from summit where Hillary stops
  hillary_descent : ℝ  -- Hillary's descent rate in ft/hr
  start_time : ℝ       -- Start time in hours (0 represents 06:00)
  meet_time : ℝ        -- Time when Hillary and Eddy meet in hours

/-- Calculates Eddy's climbing rate given a climbing scenario -/
def eddy_rate (scenario : ClimbingScenario) : ℝ :=
  -- The actual calculation of Eddy's rate
  sorry

/-- Theorem stating that Eddy's climbing rate is 5000/6 ft/hr given the specific scenario -/
theorem eddy_rate_is_correct (scenario : ClimbingScenario) 
  (h1 : scenario.summit_distance = 5000)
  (h2 : scenario.hillary_rate = 800)
  (h3 : scenario.hillary_stop = 1000)
  (h4 : scenario.hillary_descent = 1000)
  (h5 : scenario.start_time = 0)
  (h6 : scenario.meet_time = 6) : 
  eddy_rate scenario = 5000 / 6 := by
  sorry

end eddy_rate_is_correct_l1012_101294


namespace movie_of_the_year_criterion_l1012_101200

/-- The number of members in the Cinematic Academy -/
def academy_members : ℕ := 1500

/-- The fraction of top-10 lists a film must appear on to be considered for "movie of the year" -/
def required_fraction : ℚ := 1/2

/-- The smallest number of top-10 lists a film must appear on to be considered for "movie of the year" -/
def min_lists : ℕ := 750

theorem movie_of_the_year_criterion :
  min_lists = (academy_members : ℚ) * required_fraction :=
by sorry

end movie_of_the_year_criterion_l1012_101200


namespace building_area_scientific_notation_l1012_101279

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem building_area_scientific_notation :
  toScientificNotation 258000 = ScientificNotation.mk 2.58 5 sorry := by sorry

end building_area_scientific_notation_l1012_101279


namespace rancher_cattle_movement_l1012_101268

/-- A problem about a rancher moving cattle to higher ground. -/
theorem rancher_cattle_movement
  (total_cattle : ℕ)
  (truck_capacity : ℕ)
  (truck_speed : ℝ)
  (total_time : ℝ)
  (h1 : total_cattle = 400)
  (h2 : truck_capacity = 20)
  (h3 : truck_speed = 60)
  (h4 : total_time = 40)
  : (total_time * truck_speed) / (2 * (total_cattle / truck_capacity)) = 60 :=
by sorry

end rancher_cattle_movement_l1012_101268


namespace specificGrid_toothpicks_l1012_101216

/-- Represents a rectangular grid with diagonal supports -/
structure ToothpickGrid where
  length : ℕ
  width : ℕ
  diagonalInterval : ℕ

/-- Calculates the total number of toothpicks used in the grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  let verticalToothpicks := (grid.length + 1) * grid.width
  let horizontalToothpicks := (grid.width + 1) * grid.length
  let diagonalLines := (grid.length + 1) / grid.diagonalInterval + (grid.width + 1) / grid.diagonalInterval
  let diagonalToothpicks := diagonalLines * 7  -- Approximation of √50
  verticalToothpicks + horizontalToothpicks + diagonalToothpicks

/-- The specific grid described in the problem -/
def specificGrid : ToothpickGrid :=
  { length := 45
    width := 25
    diagonalInterval := 5 }

theorem specificGrid_toothpicks :
  totalToothpicks specificGrid = 2446 := by
  sorry

end specificGrid_toothpicks_l1012_101216


namespace equation_solution_l1012_101298

theorem equation_solution : ∃ x : ℝ, (3034 - (1002 / x) = 2984) ∧ x = 20.04 := by
  sorry

end equation_solution_l1012_101298


namespace geometric_series_sum_l1012_101254

def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 7
  geometricSum a r n = 4/3 := by
  sorry

end geometric_series_sum_l1012_101254


namespace right_triangle_k_values_l1012_101275

/-- A right-angled triangle in a 2D Cartesian coordinate system. -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angled : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∨
                    (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 0 ∨
                    (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

/-- The theorem stating the possible values of k in the given right-angled triangle. -/
theorem right_triangle_k_values (triangle : RightTriangle)
  (h1 : triangle.B.1 - triangle.A.1 = 2 ∧ triangle.B.2 - triangle.A.2 = 1)
  (h2 : triangle.C.1 - triangle.A.1 = 3)
  (h3 : ∃ k, triangle.C.2 - triangle.A.2 = k) :
  ∃ k, (k = -6 ∨ k = -1) ∧ triangle.C.2 - triangle.A.2 = k :=
sorry


end right_triangle_k_values_l1012_101275


namespace only_B_is_random_event_l1012_101207

-- Define the events
inductive Event
| A : Event  -- Water boils at 100°C under standard atmospheric pressure
| B : Event  -- Buying a lottery ticket and winning a prize
| C : Event  -- A runner's speed is 30 meters per second
| D : Event  -- Drawing a red ball from a bag containing only white and black balls

-- Define the property of being a random event
def isRandomEvent (e : Event) : Prop :=
  match e with
  | Event.A => false
  | Event.B => true
  | Event.C => false
  | Event.D => false

-- Theorem: Only Event B is a random event
theorem only_B_is_random_event :
  ∀ e : Event, isRandomEvent e ↔ e = Event.B :=
by sorry

end only_B_is_random_event_l1012_101207


namespace opposite_signs_absolute_difference_l1012_101265

theorem opposite_signs_absolute_difference (a b : ℝ) :
  (abs a = 4) → (abs b = 2) → (a * b < 0) → abs (a - b) = 6 := by
  sorry

end opposite_signs_absolute_difference_l1012_101265


namespace fraction_power_approximation_l1012_101288

theorem fraction_power_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000000000000000001 ∧ 
  |((1 : ℝ) / 9)^2 - 0.012345679012345678| < ε :=
sorry

end fraction_power_approximation_l1012_101288


namespace smallest_x_absolute_value_exists_smallest_x_smallest_x_value_l1012_101217

theorem smallest_x_absolute_value (x : ℝ) : 
  (|5 * x - 3| = 32) → x ≥ -29/5 :=
by sorry

theorem exists_smallest_x : 
  ∃ x : ℝ, |5 * x - 3| = 32 ∧ ∀ y : ℝ, |5 * y - 3| = 32 → y ≥ x :=
by sorry

theorem smallest_x_value : 
  ∃ x : ℝ, x = -29/5 ∧ |5 * x - 3| = 32 ∧ ∀ y : ℝ, |5 * y - 3| = 32 → y ≥ x :=
by sorry

end smallest_x_absolute_value_exists_smallest_x_smallest_x_value_l1012_101217


namespace inequality_and_equality_condition_l1012_101224

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b) ↔ a = b) :=
by sorry

end inequality_and_equality_condition_l1012_101224


namespace calculation_proof_l1012_101202

theorem calculation_proof : (-2)^0 - Real.sqrt 8 - abs (-5) + 4 * Real.sin (π/4) = -4 := by
  sorry

end calculation_proof_l1012_101202


namespace train_carriages_l1012_101211

theorem train_carriages (initial_seats : ℕ) (additional_capacity : ℕ) (total_passengers : ℕ) (num_trains : ℕ) :
  initial_seats = 25 →
  additional_capacity = 10 →
  total_passengers = 420 →
  num_trains = 3 →
  (total_passengers / (num_trains * (initial_seats + additional_capacity))) = 4 :=
by sorry

end train_carriages_l1012_101211


namespace imaginary_part_of_z_l1012_101292

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (1 + Complex.I) → z.im = 1 := by
  sorry

end imaginary_part_of_z_l1012_101292


namespace zoo_trip_remaining_money_l1012_101214

/-- Calculates the amount of money left for lunch and snacks after a zoo trip -/
theorem zoo_trip_remaining_money 
  (ticket_price : ℚ)
  (bus_fare : ℚ)
  (total_money : ℚ)
  (num_people : ℕ)
  (h1 : ticket_price = 5)
  (h2 : bus_fare = 3/2)
  (h3 : total_money = 40)
  (h4 : num_people = 2)
  : total_money - (num_people * ticket_price + num_people * bus_fare * 2) = 24 := by
  sorry

end zoo_trip_remaining_money_l1012_101214


namespace sequence_divisibility_l1012_101213

def a (n : ℕ) : ℕ := 2 * 10^(n + 1) + 19

def divides_sequence (p : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 1 ∧ p ∣ a n

theorem sequence_divisibility :
  {p : ℕ | p.Prime ∧ p ≤ 19 ∧ divides_sequence p} = {3, 7, 13, 17} := by sorry

end sequence_divisibility_l1012_101213


namespace problem_solution_l1012_101267

theorem problem_solution :
  (∃ a b c : ℝ, a * c = b * c ∧ a ≠ b) ∧
  (∀ a : ℝ, (¬ ∃ q : ℚ, a + 5 = q) ↔ (¬ ∃ q : ℚ, a = q)) ∧
  ((∀ a b : ℝ, a = b → a^2 = b^2) ∧ (∃ a b : ℝ, a^2 = b^2 ∧ a ≠ b)) ∧
  (∃ x : ℝ, x^2 < 1) :=
by sorry


end problem_solution_l1012_101267


namespace student_scores_l1012_101248

theorem student_scores (M P C : ℕ) : 
  M + P = 60 →
  C = P + 10 →
  (M + C) / 2 = 35 := by
sorry

end student_scores_l1012_101248


namespace no_extreme_points_l1012_101209

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x

def f_derivative (x : ℝ) : ℝ := 3*(x - 1)^2

theorem no_extreme_points (h : ∀ x, f_derivative x ≥ 0) :
  ∀ x, ¬ (∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x) :=
sorry

end no_extreme_points_l1012_101209


namespace linear_function_slope_condition_l1012_101262

/-- Given a linear function y = (m-2)x + 2 + m with two points on its graph,
    prove that if x₁ < x₂ and y₁ > y₂, then m < 2 -/
theorem linear_function_slope_condition (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ)
  (h1 : y₁ = (m - 2) * x₁ + 2 + m)
  (h2 : y₂ = (m - 2) * x₂ + 2 + m)
  (h3 : x₁ < x₂)
  (h4 : y₁ > y₂) :
  m < 2 :=
sorry

end linear_function_slope_condition_l1012_101262


namespace orange_banana_relationship_l1012_101219

/-- The cost of fruits at Frank's Fruit Market -/
structure FruitCost where
  banana_to_apple : ℚ  -- ratio of bananas to apples
  apple_to_orange : ℚ  -- ratio of apples to oranges

/-- Given the cost ratios, calculate how many oranges cost as much as 24 bananas -/
def oranges_for_24_bananas (cost : FruitCost) : ℚ :=
  24 * (cost.banana_to_apple * cost.apple_to_orange)

/-- Theorem stating the relationship between banana and orange costs -/
theorem orange_banana_relationship (cost : FruitCost)
  (h1 : cost.banana_to_apple = 4 / 3)
  (h2 : cost.apple_to_orange = 5 / 2) :
  oranges_for_24_bananas cost = 36 / 5 := by
  sorry

#eval oranges_for_24_bananas ⟨4/3, 5/2⟩

end orange_banana_relationship_l1012_101219


namespace torn_pages_sum_not_1990_l1012_101293

/-- Represents a sheet in the notebook -/
structure Sheet :=
  (number : ℕ)
  (h_range : number ≥ 1 ∧ number ≤ 96)

/-- The sum of page numbers on a sheet -/
def sheet_sum (s : Sheet) : ℕ := 4 * s.number - 1

/-- A selection of 25 sheets -/
def SheetSelection := { sel : Finset Sheet // sel.card = 25 }

theorem torn_pages_sum_not_1990 (sel : SheetSelection) :
  (sel.val.sum sheet_sum) ≠ 1990 := by
  sorry


end torn_pages_sum_not_1990_l1012_101293


namespace rectangle_area_l1012_101223

theorem rectangle_area (b : ℝ) : 
  let square_area : ℝ := 2500
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_breadth : ℝ := b
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area = 20 * b := by sorry

end rectangle_area_l1012_101223


namespace draw_three_from_fifteen_l1012_101201

def box_numbers : List Nat := [1, 2, 3, 4, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

def total_combinations (n k : Nat) : Nat :=
  Nat.choose n k

theorem draw_three_from_fifteen :
  total_combinations (List.length box_numbers) 3 = 455 := by
  sorry

end draw_three_from_fifteen_l1012_101201


namespace smallest_circle_radius_l1012_101264

/-- Three circles are pairwise tangent if the distance between their centers
    is equal to the sum of their radii -/
def pairwise_tangent (r₁ r₂ r₃ : ℝ) (d₁₂ d₁₃ d₂₃ : ℝ) : Prop :=
  d₁₂ = r₁ + r₂ ∧ d₁₃ = r₁ + r₃ ∧ d₂₃ = r₂ + r₃

/-- The segments connecting the centers of three circles form a right triangle
    if the square of the longest side equals the sum of squares of the other two sides -/
def right_triangle (d₁₂ d₁₃ d₂₃ : ℝ) : Prop :=
  d₂₃^2 = d₁₂^2 + d₁₃^2

theorem smallest_circle_radius
  (r : ℝ)
  (h₁ : r > 0)
  (h₂ : r < 4)
  (h₃ : pairwise_tangent r 4 6 (r + 4) (r + 6) 10)
  (h₄ : right_triangle (r + 4) (r + 6) 10) :
  r = 2 := by sorry

end smallest_circle_radius_l1012_101264


namespace angle_in_third_quadrant_l1012_101284

theorem angle_in_third_quadrant (α : Real) 
  (h1 : Real.sin (2 * α) > 0) 
  (h2 : Real.sin α + Real.cos α < 0) : 
  α ∈ Set.Icc (Real.pi) (3 * Real.pi / 2) := by
  sorry

end angle_in_third_quadrant_l1012_101284


namespace yellow_candles_count_l1012_101222

/-- The number of yellow candles on a birthday cake --/
def yellow_candles (total_candles red_candles blue_candles : ℕ) : ℕ :=
  total_candles - (red_candles + blue_candles)

/-- Theorem: The number of yellow candles is 27 --/
theorem yellow_candles_count :
  yellow_candles 79 14 38 = 27 := by
  sorry

end yellow_candles_count_l1012_101222


namespace imaginary_part_of_complex_product_l1012_101235

theorem imaginary_part_of_complex_product : 
  let z : ℂ := (2 + Complex.I) * (1 - Complex.I)
  Complex.im z = -1 := by sorry

end imaginary_part_of_complex_product_l1012_101235


namespace product_second_fourth_is_seven_l1012_101251

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ
  /-- The tenth term of the sequence is 25 -/
  tenth_term : a₁ + 9 * d = 25
  /-- The common difference is 3 -/
  diff_is_3 : d = 3

/-- The product of the second and fourth terms is 7 -/
theorem product_second_fourth_is_seven (seq : ArithmeticSequence) :
  (seq.a₁ + seq.d) * (seq.a₁ + 3 * seq.d) = 7 := by
  sorry

end product_second_fourth_is_seven_l1012_101251


namespace x_plus_y_values_l1012_101234

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) :
  (x + y = -3) ∨ (x + y = -9) :=
sorry

end x_plus_y_values_l1012_101234


namespace recipe_cups_needed_l1012_101258

theorem recipe_cups_needed (servings : ℝ) (cups_per_serving : ℝ) 
  (h1 : servings = 18.0) 
  (h2 : cups_per_serving = 2.0) : 
  servings * cups_per_serving = 36.0 := by
  sorry

end recipe_cups_needed_l1012_101258


namespace complex_square_plus_self_l1012_101291

theorem complex_square_plus_self (z : ℂ) (h : z = 1 + I) : z^2 + z = 1 + 3*I := by
  sorry

end complex_square_plus_self_l1012_101291


namespace sqrt_sum_equals_sqrt_72_l1012_101261

theorem sqrt_sum_equals_sqrt_72 (k : ℕ+) :
  Real.sqrt 2 + Real.sqrt 8 + Real.sqrt 18 = Real.sqrt k → k = 72 := by
  sorry

end sqrt_sum_equals_sqrt_72_l1012_101261


namespace solve_for_b_l1012_101210

/-- The piecewise function f(x) -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - b else 3^x

/-- Theorem stating that if f(f(1/2)) = 9, then b = -1/2 -/
theorem solve_for_b :
  ∀ b : ℝ, f b (f b (1/2)) = 9 → b = -1/2 := by
  sorry

end solve_for_b_l1012_101210


namespace isosceles_triangle_construction_impossibility_l1012_101247

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  -- Side length of the two equal sides
  side : ℝ
  -- Base angle (half of the apex angle)
  base_angle : ℝ
  -- Height from the apex to the base
  height : ℝ
  -- Length of the angle bisector from the apex
  bisector : ℝ
  -- Constraint that the base angle is positive and less than π/2
  angle_constraint : 0 < base_angle ∧ base_angle < π/2

/-- Represents the ability to construct a geometric figure -/
def Constructible (α : Type) : Prop := sorry

/-- Represents the ability to trisect an angle -/
def AngleTrisectable (angle : ℝ) : Prop := sorry

/-- The main theorem stating the impossibility of general isosceles triangle construction -/
theorem isosceles_triangle_construction_impossibility 
  (h : ℝ) (l : ℝ) (h_pos : h > 0) (l_pos : l > 0) :
  ¬∀ (t : IsoscelesTriangle), 
    t.height = h ∧ t.bisector = l → 
    Constructible IsoscelesTriangle ∧ 
    ¬∀ (angle : ℝ), AngleTrisectable angle :=
sorry

end isosceles_triangle_construction_impossibility_l1012_101247


namespace base_seven_sum_l1012_101226

/-- Given A, B, C are distinct digits in base 7 and ABC_7 + BCA_7 + CAB_7 = AAA1_7,
    prove that B + C = 6 (in base 7) if A = 1, or B + C = 12 (in base 7) if A = 2. -/
theorem base_seven_sum (A B C : ℕ) : 
  A < 7 → B < 7 → C < 7 → 
  A ≠ B → B ≠ C → A ≠ C →
  (7^2 * A + 7 * B + C) + (7^2 * B + 7 * C + A) + (7^2 * C + 7 * A + B) = 
    7^3 * A + 7^2 * A + 7 * A + 1 →
  (A = 1 ∧ B + C = 6) ∨ (A = 2 ∧ B + C = 12) :=
by sorry

end base_seven_sum_l1012_101226


namespace min_k_value_l1012_101295

theorem min_k_value (k : ℕ) : 
  (∃ x₀ : ℝ, x₀ > 2 ∧ k * (x₀ - 2) > x₀ * (Real.log x₀ + 1)) →
  k ≥ 5 :=
sorry

end min_k_value_l1012_101295


namespace fraction_equality_l1012_101238

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a + c = 2*b) 
  (h2 : 2*b*d = c*(b + d)) 
  (h3 : b ≠ 0) 
  (h4 : d ≠ 0) : 
  a / b = c / d := by sorry

end fraction_equality_l1012_101238


namespace quadratic_polynomial_property_l1012_101296

/-- Given a quadratic polynomial P(x) = x^2 + ax + b, 
    if P(10) + P(30) = 40, then P(20) = -80 -/
theorem quadratic_polynomial_property (a b : ℝ) : 
  let P : ℝ → ℝ := λ x => x^2 + a*x + b
  (P 10 + P 30 = 40) → P 20 = -80 := by sorry

end quadratic_polynomial_property_l1012_101296


namespace sqrt_expression_equals_three_l1012_101257

theorem sqrt_expression_equals_three :
  (Real.sqrt 2 + 1)^2 - Real.sqrt 18 + 2 * Real.sqrt (1/2) = 3 := by
sorry

end sqrt_expression_equals_three_l1012_101257


namespace girl_multiplication_mistake_l1012_101237

theorem girl_multiplication_mistake (x : ℤ) : 43 * x - 34 * x = 1242 → x = 138 := by
  sorry

end girl_multiplication_mistake_l1012_101237


namespace reading_ratio_two_to_three_nights_l1012_101231

/-- Represents the number of pages read on each night -/
structure ReadingPattern where
  threeNightsAgo : ℕ
  twoNightsAgo : ℕ
  lastNight : ℕ
  tonight : ℕ

/-- Theorem stating the ratio of pages read two nights ago to three nights ago -/
theorem reading_ratio_two_to_three_nights (r : ReadingPattern) : 
  r.threeNightsAgo = 15 →
  r.lastNight = r.twoNightsAgo + 5 →
  r.tonight = 20 →
  r.threeNightsAgo + r.twoNightsAgo + r.lastNight + r.tonight = 100 →
  r.twoNightsAgo / r.threeNightsAgo = 2 := by
  sorry

#check reading_ratio_two_to_three_nights

end reading_ratio_two_to_three_nights_l1012_101231


namespace quadratic_roots_difference_l1012_101227

theorem quadratic_roots_difference (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + 42*x + 384
  let roots := {x : ℝ | f x = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 8 := by
  sorry

end quadratic_roots_difference_l1012_101227


namespace card_drawing_probability_l1012_101266

theorem card_drawing_probability : 
  let cards : Finset ℕ := {1, 2, 3, 4, 5}
  let odd_cards : Finset ℕ := {1, 3, 5}
  let even_cards : Finset ℕ := {2, 4}
  let total_cards := cards.card
  let odd_count := odd_cards.card
  let even_count := even_cards.card

  let prob_first_odd := odd_count / total_cards
  let prob_second_even_given_first_odd := even_count / (total_cards - 1)

  prob_second_even_given_first_odd = 1 / 2 :=
by
  sorry

end card_drawing_probability_l1012_101266


namespace no_consecutive_heads_probability_l1012_101204

/-- The number of ways to toss n coins such that no two heads appear consecutively -/
def f : ℕ → ℕ
| 0 => 1  -- Convention for empty sequence
| 1 => 2  -- Base case
| 2 => 3  -- Base case
| (n + 3) => f (n + 2) + f (n + 1)

/-- The probability of no two heads appearing consecutively in 10 coin tosses -/
theorem no_consecutive_heads_probability :
  (f 10 : ℚ) / (2^10 : ℚ) = 9/64 := by sorry

end no_consecutive_heads_probability_l1012_101204


namespace village_population_equality_l1012_101215

theorem village_population_equality (x_initial : ℕ) (x_rate : ℕ) (y_initial : ℕ) (y_rate : ℕ) :
  x_initial = 72000 →
  x_rate = 1200 →
  y_initial = 42000 →
  y_rate = 800 →
  ∃ n : ℕ, (x_initial - n * x_rate = y_initial + n * y_rate) ∧ n = 15 :=
by
  sorry

end village_population_equality_l1012_101215


namespace probability_second_math_given_first_math_l1012_101218

def total_questions : ℕ := 5
def math_questions : ℕ := 3
def physics_questions : ℕ := 2

theorem probability_second_math_given_first_math :
  let P : ℝ := (math_questions * (math_questions - 1)) / (total_questions * (total_questions - 1))
  let Q : ℝ := math_questions / total_questions
  P / Q = 1 / 2 := by sorry

end probability_second_math_given_first_math_l1012_101218


namespace toyota_not_less_than_honda_skoda_combined_l1012_101220

/-- Proves that the number of Toyotas is not less than the number of Hondas and Skodas combined in a parking lot with specific conditions. -/
theorem toyota_not_less_than_honda_skoda_combined 
  (C T H S X Y : ℕ) 
  (h1 : C - H = (3 * (C - X)) / 2)
  (h2 : C - S = (3 * (C - Y)) / 2)
  (h3 : C - T = (X + Y) / 2) :
  T ≥ H + S := by sorry

end toyota_not_less_than_honda_skoda_combined_l1012_101220


namespace arithmetic_sequence_problem_l1012_101246

/-- Given an arithmetic sequence {a_n} with common difference d, 
    if a_1 + a_8 + a_15 = 72, then a_5 + 3d = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 8 + a 15 = 72 →           -- given sum condition
  a 5 + 3 * d = 24 := by            -- conclusion to prove
sorry


end arithmetic_sequence_problem_l1012_101246


namespace sum_of_roots_l1012_101263

theorem sum_of_roots (a b : ℝ) (ha : a * (a - 6) = 7) (hb : b * (b - 6) = 7) (hab : a ≠ b) :
  a + b = 6 := by sorry

end sum_of_roots_l1012_101263


namespace class_survey_l1012_101249

theorem class_survey (total_students : ℕ) (green_students : ℕ) (yellow_students : ℕ) (girls : ℕ) : 
  total_students = 30 →
  green_students = total_students / 2 →
  yellow_students = 9 →
  girls * 3 = (total_students - green_students - yellow_students) * 3 + girls →
  girls = 18 := by
sorry

end class_survey_l1012_101249


namespace laundry_calculation_correct_l1012_101232

/-- Represents the laundry problem setup -/
structure LaundrySetup where
  tub_capacity : Real
  clothes_weight : Real
  required_concentration : Real
  initial_detergent : Real

/-- Calculates the additional detergent and water needed for the laundry -/
def calculate_additions (setup : LaundrySetup) : Real × Real :=
  let additional_detergent := setup.tub_capacity * setup.required_concentration - setup.initial_detergent - setup.clothes_weight
  let additional_water := setup.tub_capacity - setup.clothes_weight - setup.initial_detergent - additional_detergent
  (additional_detergent, additional_water)

/-- The main theorem stating the correct additional amounts -/
theorem laundry_calculation_correct (setup : LaundrySetup) 
  (h1 : setup.tub_capacity = 15)
  (h2 : setup.clothes_weight = 4)
  (h3 : setup.required_concentration = 0.004)
  (h4 : setup.initial_detergent = 0.04) :
  calculate_additions setup = (0.004, 10.956) := by
  sorry

#eval calculate_additions { 
  tub_capacity := 15, 
  clothes_weight := 4, 
  required_concentration := 0.004, 
  initial_detergent := 0.04 
}

end laundry_calculation_correct_l1012_101232


namespace jacks_healthcare_contribution_l1012_101250

/-- Calculates the healthcare contribution in cents per hour given an hourly wage in dollars and a contribution rate as a percentage. -/
def healthcare_contribution (hourly_wage : ℚ) (contribution_rate : ℚ) : ℚ :=
  hourly_wage * 100 * (contribution_rate / 100)

/-- Proves that Jack's healthcare contribution is 57.5 cents per hour. -/
theorem jacks_healthcare_contribution :
  healthcare_contribution 25 2.3 = 57.5 := by
  sorry

end jacks_healthcare_contribution_l1012_101250


namespace circle_area_difference_l1012_101256

/-- The difference in area between two circles -/
theorem circle_area_difference : 
  let r1 : ℝ := 30  -- radius of the first circle
  let d2 : ℝ := 15  -- diameter of the second circle
  π * r1^2 - π * (d2/2)^2 = 843.75 * π := by sorry

end circle_area_difference_l1012_101256


namespace sharon_wants_254_supplies_l1012_101244

/-- The number of kitchen supplies Sharon wants to buy -/
def sharons_supplies (angela_pots : ℕ) : ℕ :=
  let angela_plates := 3 * angela_pots + 6
  let angela_cutlery := angela_plates / 2
  let sharon_pots := angela_pots / 2
  let sharon_plates := 3 * angela_plates - 20
  let sharon_cutlery := 2 * angela_cutlery
  sharon_pots + sharon_plates + sharon_cutlery

/-- Theorem stating that Sharon wants to buy 254 kitchen supplies -/
theorem sharon_wants_254_supplies : sharons_supplies 20 = 254 := by
  sorry

end sharon_wants_254_supplies_l1012_101244


namespace x_lt_2_necessary_not_sufficient_l1012_101280

theorem x_lt_2_necessary_not_sufficient :
  ∃ (x : ℝ), x^2 - x - 2 < 0 → x < 2 ∧
  ∃ (y : ℝ), y < 2 ∧ ¬(y^2 - y - 2 < 0) := by
  sorry

end x_lt_2_necessary_not_sufficient_l1012_101280


namespace average_increase_l1012_101241

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  avgRuns : ℚ

/-- Calculate the new average after scoring additional runs -/
def newAverage (player : CricketPlayer) (additionalRuns : ℕ) : ℚ :=
  (player.totalRuns + additionalRuns) / (player.innings + 1)

/-- The main theorem about the increase in average -/
theorem average_increase (player : CricketPlayer) (additionalRuns : ℕ) :
  player.innings = 10 ∧ 
  player.avgRuns = 35 ∧ 
  additionalRuns = 79 →
  newAverage player additionalRuns - player.avgRuns = 4 := by
sorry


end average_increase_l1012_101241


namespace shoe_shirt_earnings_l1012_101230

theorem shoe_shirt_earnings : 
  let shoe_pairs : ℕ := 6
  let shoe_price : ℕ := 3
  let shirt_count : ℕ := 18
  let shirt_price : ℕ := 2
  let total_earnings := shoe_pairs * shoe_price + shirt_count * shirt_price
  let people_count : ℕ := 2
  (total_earnings / people_count : ℕ) = 27 := by sorry

end shoe_shirt_earnings_l1012_101230


namespace melanie_trout_count_l1012_101270

/-- Prove that Melanie caught 8 trouts given the conditions -/
theorem melanie_trout_count (tom_count : ℕ) (melanie_count : ℕ) 
  (h1 : tom_count = 16) 
  (h2 : tom_count = 2 * melanie_count) : 
  melanie_count = 8 := by
  sorry

end melanie_trout_count_l1012_101270


namespace parallel_vectors_x_value_l1012_101297

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 4)
  parallel a b → x = 2 := by sorry

end parallel_vectors_x_value_l1012_101297


namespace morning_run_distance_l1012_101245

/-- Represents the distances of various activities in miles -/
structure DailyActivities where
  morningRun : ℝ
  afternoonWalk : ℝ
  eveningBikeRide : ℝ

/-- Calculates the total distance covered in a day -/
def totalDistance (activities : DailyActivities) : ℝ :=
  activities.morningRun + activities.afternoonWalk + activities.eveningBikeRide

/-- Theorem stating that given the conditions, the morning run distance is 2 miles -/
theorem morning_run_distance 
  (activities : DailyActivities)
  (h1 : totalDistance activities = 18)
  (h2 : activities.afternoonWalk = 2 * activities.morningRun)
  (h3 : activities.eveningBikeRide = 12) :
  activities.morningRun = 2 := by
  sorry

end morning_run_distance_l1012_101245


namespace modulo_equivalence_exists_unique_l1012_101282

theorem modulo_equivalence_exists_unique : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end modulo_equivalence_exists_unique_l1012_101282


namespace mass_of_man_is_80kg_l1012_101260

/-- The mass of a man who causes a boat to sink by a certain depth -/
def mass_of_man (boat_length boat_breadth sinking_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sinking_depth * water_density

/-- Theorem stating that the mass of the man is 80 kg -/
theorem mass_of_man_is_80kg :
  mass_of_man 4 2 0.01 1000 = 80 := by sorry

end mass_of_man_is_80kg_l1012_101260


namespace tea_mixture_price_l1012_101253

/-- Given two teas mixed in equal proportions, proves that if one tea costs 74 rupees per kg
    and the mixture costs 69 rupees per kg, then the other tea costs 64 rupees per kg. -/
theorem tea_mixture_price (price_tea2 mixture_price : ℝ) 
  (h1 : price_tea2 = 74)
  (h2 : mixture_price = 69) :
  ∃ (price_tea1 : ℝ), 
    price_tea1 = 64 ∧ 
    (price_tea1 + price_tea2) / 2 = mixture_price :=
by
  sorry

end tea_mixture_price_l1012_101253


namespace smallest_BD_is_five_l1012_101221

/-- Represents a quadrilateral with side lengths and an angle -/
structure Quadrilateral :=
  (AB BC CD DA : ℝ)
  (angleBDA : ℝ)

/-- The smallest possible integer value of BD in the given quadrilateral -/
def smallest_integer_BD (q : Quadrilateral) : ℕ :=
  sorry

/-- Theorem stating the smallest possible integer value of BD -/
theorem smallest_BD_is_five (q : Quadrilateral) 
  (h1 : q.AB = 7)
  (h2 : q.BC = 15)
  (h3 : q.CD = 7)
  (h4 : q.DA = 11)
  (h5 : q.angleBDA = 90) :
  smallest_integer_BD q = 5 :=
sorry

end smallest_BD_is_five_l1012_101221


namespace total_baseball_cards_l1012_101212

def mary_cards : ℕ := 15
def sam_cards : ℕ := 15
def keith_cards : ℕ := 15
def alyssa_cards : ℕ := 15
def john_cards : ℕ := 12
def sarah_cards : ℕ := 18
def emma_cards : ℕ := 10

theorem total_baseball_cards :
  mary_cards + sam_cards + keith_cards + alyssa_cards +
  john_cards + sarah_cards + emma_cards = 100 := by
  sorry

end total_baseball_cards_l1012_101212


namespace prime_arithmetic_mean_median_l1012_101290

theorem prime_arithmetic_mean_median (a b c : ℕ) : 
  a = 2 → 
  Nat.Prime a → 
  Nat.Prime b → 
  Nat.Prime c → 
  a < b → 
  b < c → 
  b ≠ a + 1 → 
  (a + b + c) / 3 = 6 * b → 
  c / b = 83 / 5 := by
sorry

end prime_arithmetic_mean_median_l1012_101290


namespace first_term_to_diff_ratio_l1012_101206

/-- An arithmetic sequence with a given property -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_property : (9 * a + 36 * d) = 3 * (6 * a + 15 * d)

/-- The ratio of the first term to the common difference is 1:-1 -/
theorem first_term_to_diff_ratio (seq : ArithmeticSequence) : seq.a / seq.d = -1 := by
  sorry

#check first_term_to_diff_ratio

end first_term_to_diff_ratio_l1012_101206
