import Mathlib

namespace angle_three_times_complement_l3730_373052

theorem angle_three_times_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end angle_three_times_complement_l3730_373052


namespace student_hamster_difference_l3730_373047

/-- The number of third-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 20

/-- The number of hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 1

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * students_per_classroom

/-- The total number of hamsters in all classrooms -/
def total_hamsters : ℕ := num_classrooms * hamsters_per_classroom

theorem student_hamster_difference :
  total_students - total_hamsters = 95 := by
  sorry

end student_hamster_difference_l3730_373047


namespace train_crossing_time_l3730_373009

/-- Time for a train to cross a man moving in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 270 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 36 := by
  sorry

end train_crossing_time_l3730_373009


namespace unique_solution_exponential_equation_l3730_373049

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x + 2) * (4 : ℝ)^(3*x + 7) = (8 : ℝ)^(5*x + 6) := by
  sorry

end unique_solution_exponential_equation_l3730_373049


namespace condition_sufficient_not_necessary_l3730_373050

theorem condition_sufficient_not_necessary (a b c : ℝ) :
  (∀ a b c : ℝ, a > b ∧ c > 0 → a * c > b * c) ∧
  ¬(∀ a b c : ℝ, a * c > b * c → a > b ∧ c > 0) :=
by sorry

end condition_sufficient_not_necessary_l3730_373050


namespace five_digit_sum_l3730_373087

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 9

theorem five_digit_sum (x : ℕ) (h1 : is_valid_digit x) 
  (h2 : x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 6) 
  (h3 : 120 * (1 + 3 + 4 + 6 + x) = 2640) : x = 8 := by
  sorry

end five_digit_sum_l3730_373087


namespace Ann_age_is_6_l3730_373088

/-- Ann's current age -/
def Ann_age : ℕ := sorry

/-- Tom's current age -/
def Tom_age : ℕ := 2 * Ann_age

/-- The sum of their ages 10 years later -/
def sum_ages_later : ℕ := Ann_age + 10 + Tom_age + 10

theorem Ann_age_is_6 : Ann_age = 6 := by
  have h1 : sum_ages_later = 38 := sorry
  sorry

end Ann_age_is_6_l3730_373088


namespace hawkeye_battery_charges_l3730_373034

def battery_problem (cost_per_charge : ℚ) (initial_budget : ℚ) (remaining_money : ℚ) : Prop :=
  let total_spent : ℚ := initial_budget - remaining_money
  let number_of_charges : ℚ := total_spent / cost_per_charge
  number_of_charges = 4

theorem hawkeye_battery_charges : 
  battery_problem (35/10) 20 6 := by
  sorry

end hawkeye_battery_charges_l3730_373034


namespace rounded_number_accuracy_l3730_373026

/-- Given a number 5.60 × 10^5 rounded to the nearest whole number,
    prove that it is accurate to the thousandth place. -/
theorem rounded_number_accuracy (n : ℝ) (h : n = 5.60 * 10^5) :
  ∃ (m : ℕ), |n - m| ≤ 5 * 10^2 :=
sorry

end rounded_number_accuracy_l3730_373026


namespace fence_cost_l3730_373093

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 289 →
  price_per_foot = 60 →
  cost = 4 * Real.sqrt area * price_per_foot →
  cost = 4080 := by
  sorry


end fence_cost_l3730_373093


namespace f_minimum_at_three_halves_l3730_373006

def f (x : ℝ) := 3 * x^2 - 9 * x + 2

theorem f_minimum_at_three_halves :
  ∃ (y : ℝ), ∀ (x : ℝ), f (3/2) ≤ f x :=
sorry

end f_minimum_at_three_halves_l3730_373006


namespace benny_total_spend_l3730_373067

def total_cost (soft_drink_cost : ℕ) (candy_bar_cost : ℕ) (num_candy_bars : ℕ) : ℕ :=
  soft_drink_cost + candy_bar_cost * num_candy_bars

theorem benny_total_spend :
  let soft_drink_cost : ℕ := 2
  let candy_bar_cost : ℕ := 5
  let num_candy_bars : ℕ := 5
  total_cost soft_drink_cost candy_bar_cost num_candy_bars = 27 := by
sorry

end benny_total_spend_l3730_373067


namespace robin_extra_drinks_l3730_373016

/-- Calculates the number of extra drinks given the quantities bought and consumed --/
def extra_drinks (sodas_bought energy_bought smoothies_bought
                  sodas_drunk energy_drunk smoothies_drunk : ℕ) : ℕ :=
  (sodas_bought + energy_bought + smoothies_bought) -
  (sodas_drunk + energy_drunk + smoothies_drunk)

/-- Theorem stating that Robin has 32 extra drinks --/
theorem robin_extra_drinks :
  extra_drinks 22 15 12 6 9 2 = 32 := by
  sorry

end robin_extra_drinks_l3730_373016


namespace minimum_score_for_average_increase_miguel_minimum_score_l3730_373005

def current_scores : List ℕ := [92, 88, 76, 84, 90]
def desired_increase : ℕ := 4

def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem minimum_score_for_average_increase 
  (scores : List ℕ) 
  (increase : ℕ) 
  (min_score : ℕ) : Prop :=
  let current_avg := average scores
  let new_scores := scores ++ [min_score]
  let new_avg := average new_scores
  new_avg ≥ current_avg + increase ∧
  ∀ (score : ℕ), score < min_score → 
    average (scores ++ [score]) < current_avg + increase

theorem miguel_minimum_score : 
  minimum_score_for_average_increase current_scores desired_increase 110 := by
  sorry

end minimum_score_for_average_increase_miguel_minimum_score_l3730_373005


namespace rahul_work_time_l3730_373021

theorem rahul_work_time (meena_time : ℝ) (combined_time : ℝ) (rahul_time : ℝ) : 
  meena_time = 10 →
  combined_time = 10 / 3 →
  1 / rahul_time + 1 / meena_time = 1 / combined_time →
  rahul_time = 5 := by
sorry

end rahul_work_time_l3730_373021


namespace remy_water_usage_l3730_373084

theorem remy_water_usage (roman_usage : ℕ) 
  (h1 : roman_usage + (3 * roman_usage + 1) = 33) : 
  3 * roman_usage + 1 = 25 := by
  sorry

end remy_water_usage_l3730_373084


namespace tickets_left_l3730_373085

def tickets_bought : ℕ := 11
def tickets_spent : ℕ := 3

theorem tickets_left : tickets_bought - tickets_spent = 8 := by
  sorry

end tickets_left_l3730_373085


namespace hexagon_y_coordinate_l3730_373094

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon with six vertices -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Calculates the area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- Checks if a hexagon has a vertical line of symmetry -/
def hasVerticalSymmetry (h : Hexagon) : Prop := sorry

theorem hexagon_y_coordinate 
  (h : Hexagon)
  (symm : hasVerticalSymmetry h)
  (aCoord : h.A = ⟨0, 0⟩)
  (bCoord : h.B = ⟨0, 6⟩)
  (eCoord : h.E = ⟨4, 0⟩)
  (dCoord : h.D.x = 4)
  (area : hexagonArea h = 58) :
  h.D.y = 14.5 := by sorry

end hexagon_y_coordinate_l3730_373094


namespace quadratic_roots_to_coefficients_l3730_373028

theorem quadratic_roots_to_coefficients :
  ∀ (p q : ℝ),
    (∀ x : ℝ, x^2 + p*x + q = 0 ↔ x = -2 ∨ x = 3) →
    p = -1 ∧ q = -6 := by
  sorry

end quadratic_roots_to_coefficients_l3730_373028


namespace product_of_repeating_decimals_l3730_373057

/-- The product of two repeating decimals 0.151515... and 0.353535... is equal to 175/3267 -/
theorem product_of_repeating_decimals : 
  (15 : ℚ) / 99 * (35 : ℚ) / 99 = (175 : ℚ) / 3267 := by
  sorry

end product_of_repeating_decimals_l3730_373057


namespace great_wall_scientific_notation_l3730_373092

theorem great_wall_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 21200000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.12 ∧ n = 7 := by
  sorry

end great_wall_scientific_notation_l3730_373092


namespace soccer_goals_proof_l3730_373053

theorem soccer_goals_proof (total_goals : ℕ) : 
  (total_goals / 3 : ℚ) + (total_goals / 5 : ℚ) + 8 + 20 = total_goals →
  20 ≤ 27 →
  ∃ (individual_goals : List ℕ), 
    individual_goals.length = 9 ∧ 
    individual_goals.sum = 20 ∧
    ∀ g ∈ individual_goals, g ≤ 3 :=
by sorry

end soccer_goals_proof_l3730_373053


namespace students_painting_l3730_373082

theorem students_painting (green red both : ℕ) 
  (h1 : green = 52)
  (h2 : red = 56)
  (h3 : both = 38) :
  green + red - both = 70 := by
  sorry

end students_painting_l3730_373082


namespace unique_root_in_interval_l3730_373075

/-- Theorem: Given a cubic function f(x) = -2x^3 - x + 1 defined on the interval [m, n],
    where f(m)f(n) < 0, the equation f(x) = 0 has exactly one real root in the interval [m, n]. -/
theorem unique_root_in_interval (m n : ℝ) (h : m ≤ n) :
  let f : ℝ → ℝ := λ x ↦ -2 * x^3 - x + 1
  (f m) * (f n) < 0 →
  ∃! x, m ≤ x ∧ x ≤ n ∧ f x = 0 :=
by sorry

end unique_root_in_interval_l3730_373075


namespace cherry_weekly_earnings_l3730_373015

/-- Represents Cherry's delivery service earnings --/
def cherry_earnings : ℝ → ℝ → ℝ → ℝ → ℝ := λ price_small price_large num_small num_large =>
  (price_small * num_small + price_large * num_large) * 7

/-- Theorem stating Cherry's weekly earnings --/
theorem cherry_weekly_earnings :
  let price_small := 2.5
  let price_large := 4
  let num_small := 4
  let num_large := 2
  cherry_earnings price_small price_large num_small num_large = 126 :=
by sorry

end cherry_weekly_earnings_l3730_373015


namespace hiking_trail_length_l3730_373099

/-- Represents the hiking trail problem -/
def HikingTrail :=
  {length : ℝ // length > 0}

/-- The total time for the round trip in hours -/
def totalTime : ℝ := 3

/-- The uphill speed in km/h -/
def uphillSpeed : ℝ := 2

/-- The downhill speed in km/h -/
def downhillSpeed : ℝ := 4

/-- Theorem stating that the length of the hiking trail is 4 km -/
theorem hiking_trail_length :
  ∃ (trail : HikingTrail),
    (trail.val / uphillSpeed + trail.val / downhillSpeed = totalTime) ∧
    trail.val = 4 := by sorry

end hiking_trail_length_l3730_373099


namespace monthly_compound_interest_greater_than_yearly_l3730_373013

theorem monthly_compound_interest_greater_than_yearly :
  1 + 5 / 100 < (1 + 5 / (12 * 100)) ^ 12 := by
  sorry

end monthly_compound_interest_greater_than_yearly_l3730_373013


namespace unique_five_digit_sum_l3730_373048

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def remove_digit (n : ℕ) (pos : Fin 5) : ℕ :=
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  let removed := digits.removeNth pos
  removed.foldl (λ acc d => acc * 10 + d) 0

theorem unique_five_digit_sum (n : ℕ) : 
  is_five_digit n ∧ 
  (∃ (pos : Fin 5), n + remove_digit n pos = 54321) ↔ 
  n = 49383 :=
sorry

end unique_five_digit_sum_l3730_373048


namespace line_parallel_to_x_axis_line_through_1_2_parallel_to_x_axis_l3730_373019

/-- A line parallel to the x-axis passing through a point (x₀, y₀) has the equation y = y₀ -/
theorem line_parallel_to_x_axis (x₀ y₀ : ℝ) :
  let line := {(x, y) : ℝ × ℝ | y = y₀}
  (∀ (x : ℝ), (x, y₀) ∈ line) ∧ ((x₀, y₀) ∈ line) → 
  ∀ (x y : ℝ), (x, y) ∈ line ↔ y = y₀ :=
by sorry

/-- The equation of the line passing through (1,2) and parallel to the x-axis is y = 2 -/
theorem line_through_1_2_parallel_to_x_axis :
  let line := {(x, y) : ℝ × ℝ | y = 2}
  (∀ (x : ℝ), (x, 2) ∈ line) ∧ ((1, 2) ∈ line) → 
  ∀ (x y : ℝ), (x, y) ∈ line ↔ y = 2 :=
by sorry

end line_parallel_to_x_axis_line_through_1_2_parallel_to_x_axis_l3730_373019


namespace smallest_purple_balls_l3730_373054

theorem smallest_purple_balls (x : ℕ) (y : ℕ) : 
  x > 0 ∧ 
  x % 10 = 0 ∧ 
  x % 8 = 0 ∧ 
  x % 3 = 0 ∧
  x / 8 + 10 = 8 ∧
  y = x / 10 + x / 8 + x / 3 + (x / 10 + 9) + 8 + x / 8 - x ∧
  y > 0 →
  y ≥ 25 :=
sorry

end smallest_purple_balls_l3730_373054


namespace complete_gear_exists_l3730_373012

/-- Represents a gear with a certain number of teeth and missing teeth positions -/
structure Gear where
  num_teeth : Nat
  missing_teeth : Finset Nat

/-- The problem statement -/
theorem complete_gear_exists (gear1 gear2 : Gear)
  (h1 : gear1.num_teeth = 14)
  (h2 : gear2.num_teeth = 14)
  (h3 : gear1.missing_teeth.card = 4)
  (h4 : gear2.missing_teeth.card = 4) :
  ∃ (rotation : Nat), ∀ (pos : Nat),
    pos ∈ gear1.missing_teeth →
    (pos + rotation) % gear1.num_teeth ∉ gear2.missing_teeth :=
sorry

end complete_gear_exists_l3730_373012


namespace tangent_circle_rectangle_existence_l3730_373074

/-- Given a circle of radius r tangent to the legs of a right angle,
    there exists a point M on the circumference forming a rectangle MPOQ
    with perimeter 2p if and only if r(2 - √2) ≤ p ≤ r(2 + √2) -/
theorem tangent_circle_rectangle_existence (r p : ℝ) (hr : r > 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = p ∧ x^2 + y^2 = 2*r*p) ↔
  r*(2 - Real.sqrt 2) ≤ p ∧ p ≤ r*(2 + Real.sqrt 2) :=
sorry

end tangent_circle_rectangle_existence_l3730_373074


namespace find_n_l3730_373065

theorem find_n : ∀ n : ℤ, (∀ x : ℝ, (x - 2) * (x + 1) = x^2 + n*x - 2) → n = -1 := by
  sorry

end find_n_l3730_373065


namespace point_satisfies_conditions_l3730_373066

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of the line y = -2x + 3 -/
def on_line (p : Point) : Prop :=
  p.y = -2 * p.x + 3

/-- The condition for a point to be in the first quadrant -/
def in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The point (1, 1) -/
def point : Point :=
  { x := 1, y := 1 }

theorem point_satisfies_conditions :
  in_first_quadrant point ∧ on_line point :=
by sorry

end point_satisfies_conditions_l3730_373066


namespace expression_simplification_l3730_373018

theorem expression_simplification (x : ℤ) (h : x = 2018) :
  x^2 + 2*x - x*(x + 1) = 2018 := by
sorry

end expression_simplification_l3730_373018


namespace inequality_proof_l3730_373056

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_condition : x * y + y * z + z * x = 1) : 
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5/2 :=
by sorry

end inequality_proof_l3730_373056


namespace milk_fraction_in_second_cup_l3730_373031

theorem milk_fraction_in_second_cup 
  (V : ℝ) -- Volume of each cup
  (x : ℝ) -- Fraction of milk in the second cup
  (h1 : V > 0) -- Volume is positive
  (h2 : 0 ≤ x ∧ x ≤ 1) -- x is a valid fraction
  : ((2/5 * V + (1 - x) * V) / ((3/5 * V + x * V))) = 3/7 → x = 4/5 := by
  sorry

end milk_fraction_in_second_cup_l3730_373031


namespace single_elimination_tournament_games_l3730_373090

/-- 
Calculates the number of games in a single-elimination tournament.
num_teams: The number of teams in the tournament.
-/
def num_games (num_teams : ℕ) : ℕ :=
  num_teams - 1

theorem single_elimination_tournament_games :
  num_games 16 = 15 := by
  sorry

end single_elimination_tournament_games_l3730_373090


namespace min_value_of_f_l3730_373025

/-- The function f(x) = -x^3 + 3x^2 + 9x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

/-- Theorem: Given f(x) = -x^3 + 3x^2 + 9x + a, where a is a constant,
    and the maximum value of f(x) in the interval [-2, 2] is 20,
    the minimum value of f(x) in the interval [-2, 2] is -7. -/
theorem min_value_of_f (a : ℝ) (h : ∃ x ∈ Set.Icc (-2) 2, f a x = 20 ∧ ∀ y ∈ Set.Icc (-2) 2, f a y ≤ 20) :
  ∃ x ∈ Set.Icc (-2) 2, f a x = -7 ∧ ∀ y ∈ Set.Icc (-2) 2, f a y ≥ -7 :=
sorry

end min_value_of_f_l3730_373025


namespace division_with_remainder_l3730_373000

theorem division_with_remainder (n : ℕ) (h1 : n % 17 ≠ 0) (h2 : n / 17 = 25) :
  n ≤ 441 ∧ n ≥ 426 := by
  sorry

end division_with_remainder_l3730_373000


namespace certain_number_proof_l3730_373039

/-- Given that 213 * 16 = 3408, prove that the number x satisfying x * 2.13 = 0.03408 is equal to 0.016 -/
theorem certain_number_proof (h : 213 * 16 = 3408) : 
  ∃ x : ℝ, x * 2.13 = 0.03408 ∧ x = 0.016 := by
  sorry

end certain_number_proof_l3730_373039


namespace function_transformation_l3730_373058

/-- Given a function f such that f(x-1) = 2x^2 + 3x for all x,
    prove that f(x) = 2x^2 + 7x + 5 for all x. -/
theorem function_transformation (f : ℝ → ℝ) 
    (h : ∀ x, f (x - 1) = 2 * x^2 + 3 * x) : 
    ∀ x, f x = 2 * x^2 + 7 * x + 5 := by
  sorry

end function_transformation_l3730_373058


namespace linear_function_not_in_third_quadrant_l3730_373097

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

-- Define the roots of the quadratic equation
def roots (a b : ℝ) : Prop := quadratic_eq a ∧ quadratic_eq b ∧ a ≠ b

-- Define the linear function
def linear_function (x : ℝ) (a b : ℝ) : ℝ := (a*b - 1)*x + a + b

-- Theorem: The linear function does not pass through the third quadrant
theorem linear_function_not_in_third_quadrant (a b : ℝ) (h : roots a b) :
  ∀ x y : ℝ, y = linear_function x a b → ¬(x < 0 ∧ y < 0) :=
sorry

end linear_function_not_in_third_quadrant_l3730_373097


namespace min_value_g_l3730_373080

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points -/
def distance (p q : Point3D) : ℝ := sorry

/-- Definition of the tetrahedron EFGH -/
def Tetrahedron (E F G H : Point3D) : Prop :=
  distance E H = 30 ∧
  distance F G = 30 ∧
  distance E G = 40 ∧
  distance F H = 40 ∧
  distance E F = 48 ∧
  distance G H = 48

/-- Function g(Y) as defined in the problem -/
def g (E F G H Y : Point3D) : ℝ :=
  distance E Y + distance F Y + distance G Y + distance H Y

/-- Theorem stating the minimum value of g(Y) -/
theorem min_value_g (E F G H : Point3D) :
  Tetrahedron E F G H →
  ∃ (min : ℝ), min = 4 * Real.sqrt 578 ∧
    ∀ (Y : Point3D), g E F G H Y ≥ min :=
by sorry

end min_value_g_l3730_373080


namespace sufficient_not_necessary_l3730_373045

theorem sufficient_not_necessary (x : ℝ) : 
  (x ≥ (1/2) → 2*x^2 + x - 1 ≥ 0) ∧ 
  ¬(2*x^2 + x - 1 ≥ 0 → x ≥ (1/2)) :=
by sorry

end sufficient_not_necessary_l3730_373045


namespace blue_pill_cost_proof_l3730_373036

def treatment_duration : ℕ := 3 * 7 -- 3 weeks in days

def daily_blue_pills : ℕ := 1
def daily_yellow_pills : ℕ := 1

def total_cost : ℚ := 735

def blue_pill_cost : ℚ := 18.5
def yellow_pill_cost : ℚ := blue_pill_cost - 2

theorem blue_pill_cost_proof :
  blue_pill_cost * (treatment_duration * daily_blue_pills) +
  yellow_pill_cost * (treatment_duration * daily_yellow_pills) = total_cost :=
by sorry

end blue_pill_cost_proof_l3730_373036


namespace rationalize_sqrt_sum_l3730_373072

def rationalize_and_simplify (x y z : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  sorry

theorem rationalize_sqrt_sum : 
  let (A, B, C, D, E, F) := rationalize_and_simplify 5 2 7
  A + B + C + D + E + F = 84 := by sorry

end rationalize_sqrt_sum_l3730_373072


namespace arithmetic_sequence_sum_l3730_373040

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 8) :
  a 2 + a 4 + a 5 + a 9 = 32 :=
by
  sorry

end arithmetic_sequence_sum_l3730_373040


namespace simplify_expression_l3730_373095

theorem simplify_expression : (((3 + 4 + 6 + 7) / 3) + ((3 * 6 + 9) / 4)) = 161 / 12 := by
  sorry

end simplify_expression_l3730_373095


namespace parabola_equation_l3730_373062

/-- A parabola with a vertical axis of symmetry -/
structure VerticalParabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a vertical parabola in vertex form -/
def VerticalParabola.equation (p : VerticalParabola) (x y : ℝ) : Prop :=
  y = p.a * (x - p.h)^2 + p.k

/-- The equation of a vertical parabola in standard form -/
def VerticalParabola.standardForm (p : VerticalParabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + (-2 * p.a * p.h) * x + (p.a * p.h^2 + p.k)

theorem parabola_equation (p : VerticalParabola) (h_vertex : p.h = 3 ∧ p.k = -2)
    (h_point : p.equation 5 6) :
  p.standardForm x y ↔ y = 2 * x^2 - 12 * x + 16 := by sorry

end parabola_equation_l3730_373062


namespace negation_of_proposition_negation_of_specific_proposition_l3730_373024

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, x > 0 → p x) ↔ (∃ x : ℝ, x > 0 ∧ ¬(p x)) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by sorry

end negation_of_proposition_negation_of_specific_proposition_l3730_373024


namespace root_set_equivalence_l3730_373002

/-- The function f(x) = x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

/-- The set of real roots of f(x) = 0 -/
def rootSet (a : ℝ) : Set ℝ := {x : ℝ | f a x = 0}

/-- The set of real roots of f(f(x)) = 0 -/
def composedRootSet (a : ℝ) : Set ℝ := {x : ℝ | f a (f a x) = 0}

/-- The theorem stating the equivalence between the condition and the range of a -/
theorem root_set_equivalence :
  ∀ a : ℝ, (rootSet a = composedRootSet a ∧ rootSet a ≠ ∅) ↔ 0 ≤ a ∧ a < 4 :=
sorry

end root_set_equivalence_l3730_373002


namespace inequality_proof_l3730_373020

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 := by
  sorry

end inequality_proof_l3730_373020


namespace heptagon_internal_angles_sum_heptagon_internal_angles_sum_is_540_l3730_373010

/-- The sum of internal angles of a heptagon, excluding the central point when divided into triangles -/
theorem heptagon_internal_angles_sum : ℝ :=
  let n : ℕ := 7  -- number of vertices in the heptagon
  let polygon_angle_sum : ℝ := (n - 2) * 180
  let central_angle_sum : ℝ := 360
  polygon_angle_sum - central_angle_sum

/-- Proof that the sum of internal angles of a heptagon, excluding the central point, is 540 degrees -/
theorem heptagon_internal_angles_sum_is_540 :
  heptagon_internal_angles_sum = 540 := by
  sorry

end heptagon_internal_angles_sum_heptagon_internal_angles_sum_is_540_l3730_373010


namespace fractional_equation_solution_l3730_373004

theorem fractional_equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  (4 / (x - 1) = 3 / x) ↔ x = -3 := by
sorry

end fractional_equation_solution_l3730_373004


namespace perfect_squares_solution_l3730_373061

theorem perfect_squares_solution (x y : ℤ) :
  (∃ a : ℤ, x + y = a^2) →
  (∃ b : ℤ, 2*x + 3*y = b^2) →
  (∃ c : ℤ, 3*x + y = c^2) →
  x = 0 ∧ y = 0 := by
sorry

end perfect_squares_solution_l3730_373061


namespace snow_on_monday_l3730_373042

theorem snow_on_monday (total_snow : ℝ) (tuesday_snow : ℝ) 
  (h1 : total_snow = 0.53)
  (h2 : tuesday_snow = 0.21) :
  total_snow - tuesday_snow = 0.53 - 0.21 := by
sorry

end snow_on_monday_l3730_373042


namespace line_properties_l3730_373046

-- Define a type for lines in 2D Cartesian plane
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a function to check if two lines are perpendicular
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

-- Define a function to represent a line passing through (1,1) with slope 1
def line_through_1_1_slope_1 (x : ℝ) : Prop :=
  x ≠ 1 → ∃ y : ℝ, y - 1 = x - 1

-- Theorem stating the properties we want to prove
theorem line_properties :
  ∀ (l1 l2 : Line),
    (are_perpendicular l1 l2 → l1.slope * l2.slope = -1) ∧
    line_through_1_1_slope_1 2 := by
  sorry

end line_properties_l3730_373046


namespace problem_statement_l3730_373044

theorem problem_statement (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27*x^3) 
  (h3 : a - b = 3*x) : 
  a = 3*x := by
sorry

end problem_statement_l3730_373044


namespace isosceles_triangle_altitude_l3730_373033

theorem isosceles_triangle_altitude (a : ℝ) : 
  let r : ℝ := 7
  let circle_x_circumference : ℝ := 14 * Real.pi
  let circle_y_radius : ℝ := 2 * a
  (circle_x_circumference = 2 * Real.pi * r) →
  (circle_y_radius = r) →
  let h : ℝ := Real.sqrt 3 * a
  (h^2 + a^2 = r^2) :=
by sorry

end isosceles_triangle_altitude_l3730_373033


namespace overlapping_rectangles_perimeter_l3730_373027

/-- The perimeter of a shape formed by two overlapping rectangles -/
theorem overlapping_rectangles_perimeter :
  ∀ (length width : ℝ),
  length = 7 →
  width = 3 →
  (2 * (length + width)) * 2 - 2 * width = 28 :=
by
  sorry

end overlapping_rectangles_perimeter_l3730_373027


namespace square_plus_reciprocal_square_l3730_373068

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 4) : x^2 + 1/x^2 = 14 := by
  sorry

end square_plus_reciprocal_square_l3730_373068


namespace parallel_lines_solution_l3730_373051

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a b c d e f : ℝ) : Prop :=
  a * e = b * d

/-- The first line: ax + 2y + 6 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 6 = 0

/-- The second line: x + (a - 1)y + (a^2 - 1) = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  x + (a - 1) * y + (a^2 - 1) = 0

theorem parallel_lines_solution :
  ∀ a : ℝ, parallel_lines a 2 1 (a - 1) 1 1 → a = -1 := by
  sorry

end parallel_lines_solution_l3730_373051


namespace janet_song_time_l3730_373043

theorem janet_song_time (original_time : ℝ) (speed_increase : ℝ) (new_time : ℝ) : 
  original_time = 200 →
  speed_increase = 0.25 →
  new_time = original_time / (1 + speed_increase) →
  new_time = 160 := by
sorry

end janet_song_time_l3730_373043


namespace problem_solution_l3730_373077

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = -6) : 
  y = 19.5 := by
  sorry

end problem_solution_l3730_373077


namespace problem_statement_l3730_373014

theorem problem_statement : 2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1600 := by
  sorry

end problem_statement_l3730_373014


namespace number_plus_273_l3730_373038

theorem number_plus_273 (x : ℤ) : x - 477 = 273 → x + 273 = 1023 := by
  sorry

end number_plus_273_l3730_373038


namespace parallel_vectors_imply_m_equals_one_l3730_373029

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

/-- Given vectors a and b, prove that if they are parallel, then m = 1 -/
theorem parallel_vectors_imply_m_equals_one (m : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (m, m + 1)
  are_parallel a b → m = 1 := by
  sorry

end parallel_vectors_imply_m_equals_one_l3730_373029


namespace can_display_total_l3730_373032

def triangle_display (n : ℕ) (first_row : ℕ) (increment : ℕ) : ℕ :=
  (n * (2 * first_row + (n - 1) * increment)) / 2

theorem can_display_total :
  let n := 9  -- number of rows
  let seventh_row := 19  -- number of cans in the seventh row
  let increment := 3  -- difference in cans between adjacent rows
  let first_row := seventh_row - 6 * increment  -- number of cans in the first row
  triangle_display n first_row increment = 117 :=
by
  sorry

end can_display_total_l3730_373032


namespace exist_non_adjacent_colors_l3730_373017

/-- Represents a coloring of a 50x50 square --/
def Coloring := Fin 50 → Fin 50 → Fin 100

/-- No single-color domino exists in the coloring --/
def NoSingleColorDomino (c : Coloring) : Prop :=
  ∀ i j, (i < 49 → c i j ≠ c (i+1) j) ∧ (j < 49 → c i j ≠ c i (j+1))

/-- All colors are present in the coloring --/
def AllColorsPresent (c : Coloring) : Prop :=
  ∀ color, ∃ i j, c i j = color

/-- Two colors are not adjacent if they don't appear next to each other anywhere --/
def ColorsNotAdjacent (c : Coloring) (color1 color2 : Fin 100) : Prop :=
  ∀ i j, (i < 49 → (c i j ≠ color1 ∨ c (i+1) j ≠ color2) ∧ (c i j ≠ color2 ∨ c (i+1) j ≠ color1)) ∧
         (j < 49 → (c i j ≠ color1 ∨ c i (j+1) ≠ color2) ∧ (c i j ≠ color2 ∨ c i (j+1) ≠ color1))

/-- Main theorem: There exist two non-adjacent colors in any valid coloring --/
theorem exist_non_adjacent_colors (c : Coloring) 
  (h1 : NoSingleColorDomino c) (h2 : AllColorsPresent c) : 
  ∃ color1 color2, ColorsNotAdjacent c color1 color2 := by
  sorry

end exist_non_adjacent_colors_l3730_373017


namespace complex_number_in_second_quadrant_l3730_373069

/-- The complex number i(1+i) corresponds to a point in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := Complex.I * (1 + Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_number_in_second_quadrant_l3730_373069


namespace max_value_x_y3_z4_l3730_373064

theorem max_value_x_y3_z4 (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x + y + z = 1) :
  ∃ (max : ℝ), max = 1 ∧ ∀ (a b c : ℝ), 
    a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 1 → 
    a + b^3 + c^4 ≤ max :=
sorry

end max_value_x_y3_z4_l3730_373064


namespace product_of_xy_l3730_373098

theorem product_of_xy (x y z w : ℕ+) 
  (h1 : x = w)
  (h2 : y = z)
  (h3 : w + w = w * w)
  (h4 : y = w)
  (h5 : z = 3) :
  x * y = 4 := by
  sorry

end product_of_xy_l3730_373098


namespace journey_duration_l3730_373081

/-- Represents the journey of a spaceship --/
structure SpaceshipJourney where
  initial_travel : ℕ
  initial_break : ℕ
  second_travel : ℕ
  second_break : ℕ
  subsequent_travel : ℕ
  subsequent_break : ℕ
  total_non_moving : ℕ

/-- Calculates the total journey time for a spaceship --/
def total_journey_time (j : SpaceshipJourney) : ℕ :=
  let remaining_break := j.total_non_moving - j.initial_break - j.second_break
  let subsequent_segments := remaining_break / j.subsequent_break
  j.initial_travel + j.initial_break + j.second_travel + j.second_break +
  subsequent_segments * (j.subsequent_travel + j.subsequent_break)

/-- Theorem stating that the journey takes 72 hours --/
theorem journey_duration (j : SpaceshipJourney)
  (h1 : j.initial_travel = 10)
  (h2 : j.initial_break = 3)
  (h3 : j.second_travel = 10)
  (h4 : j.second_break = 1)
  (h5 : j.subsequent_travel = 11)
  (h6 : j.subsequent_break = 1)
  (h7 : j.total_non_moving = 8) :
  total_journey_time j = 72 := by
  sorry


end journey_duration_l3730_373081


namespace line_intersection_canonical_equations_l3730_373022

/-- The canonical equations of the line of intersection of two planes -/
theorem line_intersection_canonical_equations 
  (x y z : ℝ) : 
  (6*x - 5*y + 3*z + 8 = 0) ∧ (6*x + 5*y - 4*z + 4 = 0) →
  ∃ (t : ℝ), x = 5*t - 1 ∧ y = 42*t + 2/5 ∧ z = 60*t :=
by sorry

end line_intersection_canonical_equations_l3730_373022


namespace natural_number_solution_xy_l3730_373083

theorem natural_number_solution_xy : 
  ∀ (x y : ℕ), x + y = x * y ↔ x = 2 ∧ y = 2 := by sorry

end natural_number_solution_xy_l3730_373083


namespace final_height_in_feet_l3730_373073

def initial_height : ℕ := 66
def growth_rate : ℕ := 2
def growth_duration : ℕ := 3
def inches_per_foot : ℕ := 12

theorem final_height_in_feet :
  (initial_height + growth_rate * growth_duration) / inches_per_foot = 6 :=
by sorry

end final_height_in_feet_l3730_373073


namespace exists_valid_numbering_9_not_exists_valid_numbering_10_l3730_373037

/-- A convex n-gon with a point inside -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  inner_point : ℝ × ℝ
  is_convex : sorry -- Add convexity condition

/-- Numbering of sides and segments -/
def Numbering (n : ℕ) := Fin n → Fin n

/-- Sum of numbers in a triangle -/
def triangle_sum (n : ℕ) (polygon : ConvexNGon n) (numbering : Numbering n) (i : Fin n) : ℕ := sorry

/-- Existence of a valid numbering for n = 9 -/
theorem exists_valid_numbering_9 :
  ∃ (polygon : ConvexNGon 9) (numbering : Numbering 9),
    ∀ (i j : Fin 9), triangle_sum 9 polygon numbering i = triangle_sum 9 polygon numbering j :=
sorry

/-- Non-existence of a valid numbering for n = 10 -/
theorem not_exists_valid_numbering_10 :
  ¬ ∃ (polygon : ConvexNGon 10) (numbering : Numbering 10),
    ∀ (i j : Fin 10), triangle_sum 10 polygon numbering i = triangle_sum 10 polygon numbering j :=
sorry

end exists_valid_numbering_9_not_exists_valid_numbering_10_l3730_373037


namespace is_fractional_expression_example_l3730_373041

/-- A fractional expression is an expression where the denominator contains a variable. -/
def IsFractionalExpression (n d : ℝ → ℝ) : Prop :=
  ∃ x, d x ≠ 0 ∧ (∀ y, d y ≠ d x)

/-- The expression (x + 3) / x is a fractional expression. -/
theorem is_fractional_expression_example :
  IsFractionalExpression (λ x => x + 3) (λ x => x) := by
  sorry

end is_fractional_expression_example_l3730_373041


namespace bicycle_exchange_point_exists_l3730_373011

/-- Represents the problem of finding the optimal bicycle exchange point --/
theorem bicycle_exchange_point_exists :
  ∃ x : ℝ, 0 < x ∧ x < 20 ∧
  (x / 10 + (20 - x) / 4 = (20 - x) / 8 + x / 5) := by
  sorry

#check bicycle_exchange_point_exists

end bicycle_exchange_point_exists_l3730_373011


namespace nancy_quarters_l3730_373001

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The total amount Nancy saved in dollars -/
def total_saved : ℚ := 3

/-- The number of quarters Nancy saved -/
def num_quarters : ℕ := 12

theorem nancy_quarters :
  (quarter_value * num_quarters : ℚ) = total_saved := by sorry

end nancy_quarters_l3730_373001


namespace equal_division_of_money_l3730_373076

/-- Proves that when $3.75 is equally divided among 3 people, each person receives $1.25. -/
theorem equal_division_of_money (total_amount : ℚ) (num_people : ℕ) :
  total_amount = 3.75 ∧ num_people = 3 →
  total_amount / (num_people : ℚ) = 1.25 := by
  sorry

end equal_division_of_money_l3730_373076


namespace count_integer_pairs_l3730_373023

theorem count_integer_pairs : ∃ (count : ℕ), 
  count = (Finset.filter (fun p : ℕ × ℕ => 
    let m := p.1
    let n := p.2
    1 ≤ m ∧ m ≤ 2887 ∧ 
    (7 : ℝ)^n < 3^m ∧ 3^m < 3^(m+3) ∧ 3^(m+3) < 7^(n+1))
  (Finset.product (Finset.range 2888) (Finset.range (3^2889 / 7^1233 + 1)))).card ∧
  3^2888 < 7^1233 ∧ 7^1233 < 3^2889 ∧
  count = 2466 :=
by sorry

end count_integer_pairs_l3730_373023


namespace rafting_and_tubing_count_l3730_373091

theorem rafting_and_tubing_count (total_kids : ℕ) 
  (h1 : total_kids = 40) 
  (tubing_fraction : ℚ) 
  (h2 : tubing_fraction = 1/4) 
  (rafting_fraction : ℚ) 
  (h3 : rafting_fraction = 1/2) : ℕ :=
  let tubing_kids := (total_kids : ℚ) * tubing_fraction
  let rafting_and_tubing_kids := tubing_kids * rafting_fraction
  5

#check rafting_and_tubing_count

end rafting_and_tubing_count_l3730_373091


namespace magnitude_of_complex_power_l3730_373079

theorem magnitude_of_complex_power : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end magnitude_of_complex_power_l3730_373079


namespace job_fair_theorem_l3730_373035

/-- Represents a candidate in the job fair --/
structure Candidate where
  correct_answers : ℕ
  prob_correct : ℚ

/-- The job fair scenario --/
structure JobFair where
  total_questions : ℕ
  selected_questions : ℕ
  candidate_a : Candidate
  candidate_b : Candidate

/-- Calculates the probability of a specific sequence of answers for candidate A --/
def prob_sequence (jf : JobFair) : ℚ :=
  (1 - jf.candidate_a.correct_answers / jf.total_questions) *
  (jf.candidate_a.correct_answers / (jf.total_questions - 1)) *
  ((jf.candidate_a.correct_answers - 1) / (jf.total_questions - 2))

/-- Calculates the variance of correct answers for candidate A --/
def variance_a (jf : JobFair) : ℚ := sorry

/-- Calculates the variance of correct answers for candidate B --/
def variance_b (jf : JobFair) : ℚ := sorry

/-- The main theorem to be proved --/
theorem job_fair_theorem (jf : JobFair)
    (h1 : jf.total_questions = 8)
    (h2 : jf.selected_questions = 3)
    (h3 : jf.candidate_a.correct_answers = 6)
    (h4 : jf.candidate_b.prob_correct = 3/4) :
    prob_sequence jf = 5/7 ∧ variance_a jf < variance_b jf := by
  sorry

end job_fair_theorem_l3730_373035


namespace triangular_floor_area_l3730_373096

theorem triangular_floor_area (length_feet : ℝ) (width_feet : ℝ) (feet_per_yard : ℝ) : 
  length_feet = 15 → width_feet = 12 → feet_per_yard = 3 →
  (1 / 2) * (length_feet / feet_per_yard) * (width_feet / feet_per_yard) = 10 := by
sorry

end triangular_floor_area_l3730_373096


namespace cheddar_package_size_l3730_373059

/-- The number of slices in a package of Swiss cheese -/
def swiss_slices_per_package : ℕ := 28

/-- The total number of slices bought for each type of cheese -/
def total_slices_per_type : ℕ := 84

/-- The number of slices in a package of cheddar cheese -/
def cheddar_slices_per_package : ℕ := sorry

theorem cheddar_package_size :
  cheddar_slices_per_package = swiss_slices_per_package :=
by
  sorry

#check cheddar_package_size

end cheddar_package_size_l3730_373059


namespace unique_modular_congruence_l3730_373008

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ -300 ≡ n [ZMOD 25] ∧ n = 0 := by
  sorry

end unique_modular_congruence_l3730_373008


namespace not_p_or_not_q_must_be_true_l3730_373070

theorem not_p_or_not_q_must_be_true (h1 : ¬(p ∧ q)) (h2 : p ∨ q) : ¬p ∨ ¬q :=
by
  sorry

end not_p_or_not_q_must_be_true_l3730_373070


namespace average_salary_feb_to_may_l3730_373089

def average_salary_jan_to_apr : ℝ := 8000
def average_salary_some_months : ℝ := 8800
def salary_may : ℝ := 6500
def salary_jan : ℝ := 3300

theorem average_salary_feb_to_may :
  let total_salary_jan_to_apr := average_salary_jan_to_apr * 4
  let total_salary_feb_to_apr := total_salary_jan_to_apr - salary_jan
  let total_salary_feb_to_may := total_salary_feb_to_apr + salary_may
  total_salary_feb_to_may / 4 = average_salary_some_months :=
by sorry

end average_salary_feb_to_may_l3730_373089


namespace geometric_sequence_values_l3730_373003

theorem geometric_sequence_values (a b c : ℝ) : 
  (∃ q : ℝ, q ≠ 0 ∧ 2 * q = a ∧ a * q = b ∧ b * q = c ∧ c * q = 32) → 
  ((a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = -4 ∧ b = 8 ∧ c = -16)) := by
  sorry

end geometric_sequence_values_l3730_373003


namespace dealer_articles_purchased_l3730_373071

theorem dealer_articles_purchased
  (total_purchase_price : ℝ)
  (num_articles_sold : ℕ)
  (total_selling_price : ℝ)
  (profit_percentage : ℝ)
  (h1 : total_purchase_price = 25)
  (h2 : num_articles_sold = 12)
  (h3 : total_selling_price = 33)
  (h4 : profit_percentage = 0.65)
  : ∃ (num_articles_purchased : ℕ),
    (num_articles_purchased : ℝ) * (total_selling_price / num_articles_sold) =
    total_purchase_price * (1 + profit_percentage) ∧
    num_articles_purchased = 15 :=
by sorry

end dealer_articles_purchased_l3730_373071


namespace three_lines_exist_l3730_373030

-- Define the line segment AB
def AB : ℝ := 10

-- Define the distances from points A and B to line l
def distance_A_to_l : ℝ := 6
def distance_B_to_l : ℝ := 4

-- Define a function that counts the number of lines satisfying the conditions
def count_lines : ℕ := sorry

-- Theorem statement
theorem three_lines_exist : count_lines = 3 := by sorry

end three_lines_exist_l3730_373030


namespace rice_distribution_l3730_373055

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 25 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound) / num_containers = 25 := by
  sorry

end rice_distribution_l3730_373055


namespace bottles_left_after_purchase_l3730_373086

/-- Given a store shelf with bottles of milk, prove the number of bottles left after purchases. -/
theorem bottles_left_after_purchase (initial : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) 
  (h1 : initial = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6) :
  initial - (jason_buys + harry_buys) = 24 := by
  sorry

#check bottles_left_after_purchase

end bottles_left_after_purchase_l3730_373086


namespace parabola_one_x_intercept_l3730_373060

-- Define the parabola function
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- Theorem: The parabola has exactly one x-intercept
theorem parabola_one_x_intercept : 
  ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
sorry

end parabola_one_x_intercept_l3730_373060


namespace product_of_roots_l3730_373063

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 24 → ∃ y : ℝ, (x + 3) * (x - 5) = 24 ∧ (x * y = -39) := by
  sorry

end product_of_roots_l3730_373063


namespace system_one_solution_system_two_solution_l3730_373078

-- System 1
theorem system_one_solution (x : ℝ) :
  (2 * x + 10 ≤ 5 * x + 1 ∧ 3 * (x - 1) > 9) ↔ x > 4 := by sorry

-- System 2
theorem system_two_solution (x : ℝ) :
  (3 * (x + 2) ≥ 2 * x + 5 ∧ 2 * x - (3 * x + 1) / 2 < 1) ↔ -1 ≤ x ∧ x < 3 := by sorry

end system_one_solution_system_two_solution_l3730_373078


namespace quadratic_factorization_l3730_373007

theorem quadratic_factorization (C E : ℤ) :
  (∀ x, 20 * x^2 - 87 * x + 91 = (C * x - 13) * (E * x - 7)) →
  C * E + C = 25 := by
  sorry

end quadratic_factorization_l3730_373007
