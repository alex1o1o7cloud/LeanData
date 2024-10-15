import Mathlib

namespace NUMINAMATH_CALUDE_empire_state_building_total_height_l1033_103365

/-- The height of the Empire State Building -/
def empire_state_building_height (top_floor_height antenna_height : ℕ) : ℕ :=
  top_floor_height + antenna_height

/-- Theorem: The Empire State Building is 1454 feet tall -/
theorem empire_state_building_total_height :
  empire_state_building_height 1250 204 = 1454 := by
  sorry

end NUMINAMATH_CALUDE_empire_state_building_total_height_l1033_103365


namespace NUMINAMATH_CALUDE_gear_system_teeth_count_l1033_103393

theorem gear_system_teeth_count (teeth1 teeth2 rotations3 : ℕ) 
  (h1 : teeth1 = 32)
  (h2 : teeth2 = 24)
  (h3 : rotations3 = 8)
  (h4 : ∃ total_teeth : ℕ, 
    total_teeth % 8 = 0 ∧ 
    total_teeth > teeth1 * 4 ∧ 
    total_teeth < teeth2 * 6 ∧
    total_teeth % teeth1 = 0 ∧
    total_teeth % teeth2 = 0 ∧
    total_teeth % rotations3 = 0) :
  total_teeth / rotations3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gear_system_teeth_count_l1033_103393


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1033_103367

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 10)^2 - 11*(a 10) + 16 = 0 →
  (a 30)^2 - 11*(a 30) + 16 = 0 →
  a 20 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1033_103367


namespace NUMINAMATH_CALUDE_production_performance_l1033_103349

/-- Represents the production schedule and actual performance of a team of workers. -/
structure ProductionSchedule where
  total_parts : ℕ
  days_ahead : ℕ
  extra_parts_per_day : ℕ

/-- Calculates the intended time frame and daily overachievement percentage. -/
def calculate_performance (schedule : ProductionSchedule) : ℕ × ℚ :=
  sorry

/-- Theorem stating that for the given production schedule, 
    the intended time frame was 40 days and the daily overachievement was 25%. -/
theorem production_performance :
  let schedule := ProductionSchedule.mk 8000 8 50
  calculate_performance schedule = (40, 25/100) := by
  sorry

end NUMINAMATH_CALUDE_production_performance_l1033_103349


namespace NUMINAMATH_CALUDE_function_attains_minimum_l1033_103324

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsAdditive (f : ℝ → ℝ) : Prop := ∀ x y, f (x + y) = f x + f y

theorem function_attains_minimum (f : ℝ → ℝ) (a b : ℝ) 
  (h_odd : IsOdd f) 
  (h_additive : IsAdditive f)
  (h_neg : ∀ x > 0, f x < 0)
  (h_ab : a < b) :
  ∀ x ∈ Set.Icc a b, f b ≤ f x :=
sorry

end NUMINAMATH_CALUDE_function_attains_minimum_l1033_103324


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1033_103380

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 15 → b = 21 → h^2 = a^2 + b^2 → h = Real.sqrt 666 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1033_103380


namespace NUMINAMATH_CALUDE_beach_problem_l1033_103323

/-- The number of people in the third row at the beach -/
def third_row_count (total_rows : Nat) (initial_first_row : Nat) (left_first_row : Nat) 
  (initial_second_row : Nat) (left_second_row : Nat) (total_left : Nat) : Nat :=
  total_left - ((initial_first_row - left_first_row) + (initial_second_row - left_second_row))

/-- Theorem: The number of people in the third row is 18 -/
theorem beach_problem : 
  third_row_count 3 24 3 20 5 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_beach_problem_l1033_103323


namespace NUMINAMATH_CALUDE_megacorp_fine_l1033_103326

/-- Represents the days of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Calculate the profit percentage for a given day -/
def profitPercentage (d : Day) : ℝ :=
  match d with
  | Day.Monday => 0.1
  | Day.Tuesday => 0.2
  | Day.Wednesday => 0.15
  | Day.Thursday => 0.25
  | Day.Friday => 0.3
  | Day.Saturday => 0
  | Day.Sunday => 0

/-- Daily earnings from mining -/
def miningEarnings : ℝ := 3000000

/-- Daily earnings from oil refining -/
def oilRefiningEarnings : ℝ := 5000000

/-- Monthly expenses -/
def monthlyExpenses : ℝ := 30000000

/-- Tax rate on profits -/
def taxRate : ℝ := 0.35

/-- Fine rate on annual profits -/
def fineRate : ℝ := 0.01

/-- Number of days in a month (assumed average) -/
def daysInMonth : ℕ := 30

/-- Number of months in a year -/
def monthsInYear : ℕ := 12

/-- Calculate MegaCorp's fine -/
def calculateFine : ℝ :=
  let dailyEarnings := miningEarnings + oilRefiningEarnings
  let weeklyProfits := (List.sum (List.map (fun d => dailyEarnings * profitPercentage d) [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday]))
  let monthlyRevenue := dailyEarnings * daysInMonth
  let monthlyProfits := monthlyRevenue - monthlyExpenses - (taxRate * (weeklyProfits * 4))
  let annualProfits := monthlyProfits * monthsInYear
  fineRate * annualProfits

theorem megacorp_fine : calculateFine = 23856000 := by sorry


end NUMINAMATH_CALUDE_megacorp_fine_l1033_103326


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l1033_103371

/-- The minimum distance between a point on the circle (x-2)² + y² = 4
    and a point on the line x - y + 3 = 0 is (5√2)/2 - 2 -/
theorem min_distance_circle_line :
  let circle := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}
  let line := {q : ℝ × ℝ | q.1 - q.2 + 3 = 0}
  ∃ (d : ℝ), d = (5 * Real.sqrt 2) / 2 - 2 ∧
    ∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ circle → q ∈ line →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry


end NUMINAMATH_CALUDE_min_distance_circle_line_l1033_103371


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_range_l1033_103386

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 + (a^3 - a) * x + 1

-- State the theorem
theorem increasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → x ≤ -1 → f a x < f a y) →
  -Real.sqrt 3 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_range_l1033_103386


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l1033_103355

theorem fixed_point_parabola (m : ℝ) : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + m * x + 3 * m
  f (-3) = 45 := by sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l1033_103355


namespace NUMINAMATH_CALUDE_a_can_be_any_real_l1033_103309

theorem a_can_be_any_real (a b c d e : ℝ) (h1 : b * e ≠ 0) (h2 : a / b < c / b - d / e) :
  ∃ (x y z : ℝ), x > 0 ∧ y < 0 ∧ z = 0 ∧
  (a = x ∨ a = y ∨ a = z) :=
sorry

end NUMINAMATH_CALUDE_a_can_be_any_real_l1033_103309


namespace NUMINAMATH_CALUDE_function_inequality_equivalence_l1033_103370

theorem function_inequality_equivalence 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = 3 * (x + 2)^2 - 1) 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) : 
  (∀ x, |x + 2| < b → |f x - 7| < a) ↔ b^2 = (8 + a) / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_equivalence_l1033_103370


namespace NUMINAMATH_CALUDE_biology_marks_proof_l1033_103344

def english_marks : ℕ := 86
def math_marks : ℕ := 85
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 87
def average_marks : ℕ := 89
def total_subjects : ℕ := 5

def calculate_biology_marks (eng : ℕ) (math : ℕ) (phys : ℕ) (chem : ℕ) (avg : ℕ) (total : ℕ) : ℕ :=
  avg * total - (eng + math + phys + chem)

theorem biology_marks_proof :
  calculate_biology_marks english_marks math_marks physics_marks chemistry_marks average_marks total_subjects = 95 := by
  sorry

end NUMINAMATH_CALUDE_biology_marks_proof_l1033_103344


namespace NUMINAMATH_CALUDE_exists_wonderful_with_many_primes_l1033_103398

/-- A number is wonderful if it's divisible by the sum of its prime factors -/
def IsWonderful (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (factors : List ℕ), (factors.all Nat.Prime) ∧ 
  (factors.prod = n) ∧ (n % factors.sum = 0)

/-- There exists a wonderful number with at least 10^2002 distinct prime factors -/
theorem exists_wonderful_with_many_primes : 
  ∃ (n : ℕ), IsWonderful n ∧ (∃ (factors : List ℕ), 
    (factors.all Nat.Prime) ∧ (factors.prod = n) ∧ 
    (factors.length ≥ 10^2002) ∧ (factors.Nodup)) := by
  sorry

end NUMINAMATH_CALUDE_exists_wonderful_with_many_primes_l1033_103398


namespace NUMINAMATH_CALUDE_special_quadratic_a_range_l1033_103351

/-- A quadratic function satisfying the given conditions -/
structure SpecialQuadratic where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  max_at_midpoint : ∀ a : ℝ, ∀ x : ℝ, f x ≤ f ((1 - 2*a) / 2)
  decreasing_away_from_zero : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ + x₂ ≠ 0 → f x₁ > f x₂

/-- The range of a for a SpecialQuadratic function -/
theorem special_quadratic_a_range (sq : SpecialQuadratic) : 
  ∀ a : ℝ, (∀ x : ℝ, sq.f x ≤ sq.f ((1 - 2*a) / 2)) → a > 1/2 :=
sorry

end NUMINAMATH_CALUDE_special_quadratic_a_range_l1033_103351


namespace NUMINAMATH_CALUDE_shaded_shape_area_l1033_103315

/-- The area of a shape composed of a central square and four right triangles -/
theorem shaded_shape_area (grid_size : ℕ) (square_side : ℕ) (triangle_side : ℕ) : 
  grid_size = 10 → 
  square_side = 2 → 
  triangle_side = 5 → 
  (square_side * square_side + 4 * (triangle_side * triangle_side / 2 : ℚ)) = 54 := by
  sorry

#check shaded_shape_area

end NUMINAMATH_CALUDE_shaded_shape_area_l1033_103315


namespace NUMINAMATH_CALUDE_square_of_98_l1033_103361

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by sorry

end NUMINAMATH_CALUDE_square_of_98_l1033_103361


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1033_103384

/-- Given a right circular cylinder of radius 2 intersected by a plane forming an ellipse,
    if the major axis is 20% longer than the minor axis, then the length of the major axis is 4.8. -/
theorem ellipse_major_axis_length (cylinder_radius : ℝ) (minor_axis : ℝ) (major_axis : ℝ) : 
  cylinder_radius = 2 →
  minor_axis = 2 * cylinder_radius →
  major_axis = minor_axis * 1.2 →
  major_axis = 4.8 := by
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1033_103384


namespace NUMINAMATH_CALUDE_tournament_committee_count_l1033_103369

/-- The number of teams in the league -/
def num_teams : ℕ := 4

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members selected from the host team -/
def host_selection : ℕ := 3

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 2

/-- The total number of members in the tournament committee -/
def committee_size : ℕ := 9

/-- Theorem stating the total number of possible tournament committees -/
theorem tournament_committee_count :
  (num_teams : ℕ) * (Nat.choose team_size host_selection) * 
  (Nat.choose team_size non_host_selection)^(num_teams - 1) = 1296540 := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l1033_103369


namespace NUMINAMATH_CALUDE_circle_M_properties_l1033_103359

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 1 = 0

-- Define the line L
def line_L (x y : ℝ) : Prop := x + 3*y - 2 = 0

-- Theorem statement
theorem circle_M_properties :
  -- The radius of M is √5
  (∃ (h k r : ℝ), r = Real.sqrt 5 ∧ ∀ (x y : ℝ), circle_M x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧
  -- M is symmetric with respect to the line L
  (∃ (h k : ℝ), circle_M h k ∧ line_L h k ∧
    ∀ (x y : ℝ), circle_M x y → 
      ∃ (x' y' : ℝ), circle_M x' y' ∧ line_L ((x + x')/2) ((y + y')/2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_M_properties_l1033_103359


namespace NUMINAMATH_CALUDE_apple_distribution_l1033_103313

theorem apple_distribution (total_apples : ℕ) (num_people : ℕ) (apples_per_person : ℕ) :
  total_apples = 15 →
  num_people = 3 →
  apples_per_person * num_people ≤ total_apples →
  apples_per_person = total_apples / num_people →
  apples_per_person = 5 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l1033_103313


namespace NUMINAMATH_CALUDE_acute_angle_equality_l1033_103311

theorem acute_angle_equality (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : 1 + (Real.sqrt 3 / Real.tan (80 * Real.pi / 180)) = 1 / Real.sin α) : 
  α = 50 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_equality_l1033_103311


namespace NUMINAMATH_CALUDE_mans_upstream_speed_l1033_103301

/-- Given a man's downstream speed and still water speed, calculate his upstream speed -/
theorem mans_upstream_speed (downstream_speed still_water_speed : ℝ) 
  (h1 : downstream_speed = 20)
  (h2 : still_water_speed = 15) :
  still_water_speed - (downstream_speed - still_water_speed) = 10 := by
  sorry

#check mans_upstream_speed

end NUMINAMATH_CALUDE_mans_upstream_speed_l1033_103301


namespace NUMINAMATH_CALUDE_not_all_exponential_increasing_l1033_103312

theorem not_all_exponential_increasing :
  ¬ (∀ a : ℝ, a > 0 ∧ a ≠ 1 → (∀ x y : ℝ, x < y → a^x < a^y)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_exponential_increasing_l1033_103312


namespace NUMINAMATH_CALUDE_work_completion_indeterminate_l1033_103310

structure WorkScenario where
  men : ℕ
  days : ℕ
  hours_per_day : ℝ

def total_work (scenario : WorkScenario) : ℝ :=
  scenario.men * scenario.days * scenario.hours_per_day

theorem work_completion_indeterminate 
  (scenario1 scenario2 : WorkScenario)
  (h1 : scenario1.men = 8)
  (h2 : scenario1.days = 24)
  (h3 : scenario2.men = 12)
  (h4 : scenario2.days = 16)
  (h5 : scenario1.hours_per_day = scenario2.hours_per_day)
  (h6 : total_work scenario1 = total_work scenario2) :
  ∀ (h : ℝ), ∃ (scenario1' scenario2' : WorkScenario),
    scenario1'.men = scenario1.men ∧
    scenario1'.days = scenario1.days ∧
    scenario2'.men = scenario2.men ∧
    scenario2'.days = scenario2.days ∧
    scenario1'.hours_per_day = h ∧
    scenario2'.hours_per_day = h ∧
    total_work scenario1' = total_work scenario2' :=
sorry

end NUMINAMATH_CALUDE_work_completion_indeterminate_l1033_103310


namespace NUMINAMATH_CALUDE_age_difference_l1033_103302

/-- Represents the ages of Linda and Jane -/
structure Ages where
  linda : ℕ
  jane : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.linda = 13 ∧
  ages.linda + ages.jane + 10 = 28 ∧
  ages.linda > 2 * ages.jane

/-- The theorem to prove -/
theorem age_difference (ages : Ages) :
  problem_conditions ages →
  ages.linda - 2 * ages.jane = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1033_103302


namespace NUMINAMATH_CALUDE_least_expensive_route_cost_l1033_103332

/-- Represents the cost of travel between two cities -/
structure TravelCost where
  car : ℝ
  train : ℝ

/-- Calculates the travel cost between two cities given the distance -/
def calculateTravelCost (distance : ℝ) : TravelCost :=
  { car := 0.20 * distance,
    train := 150 + 0.15 * distance }

/-- Theorem: The least expensive route for Dereven's trip costs $37106.25 -/
theorem least_expensive_route_cost :
  let xz : ℝ := 5000
  let xy : ℝ := 5500
  let yz : ℝ := Real.sqrt (xy^2 - xz^2)
  let costXY := calculateTravelCost xy
  let costYZ := calculateTravelCost yz
  let costZX := calculateTravelCost xz
  min costXY.car costXY.train + min costYZ.car costYZ.train + min costZX.car costZX.train = 37106.25 := by
  sorry


end NUMINAMATH_CALUDE_least_expensive_route_cost_l1033_103332


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_l1033_103337

/-- Given parametric equations representing a curve, prove that it forms a hyperbola -/
theorem curve_is_hyperbola (θ : ℝ) (h_θ : ∀ n : ℤ, θ ≠ n * π / 2) :
  ∃ (x y : ℝ → ℝ),
    (∀ t, x t = ((Real.exp t + Real.exp (-t)) / 2) * Real.cos θ) ∧
    (∀ t, y t = ((Real.exp t - Real.exp (-t)) / 2) * Real.sin θ) →
    ∀ t, (x t)^2 / (Real.cos θ)^2 - (y t)^2 / (Real.sin θ)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_l1033_103337


namespace NUMINAMATH_CALUDE_probability_all_white_drawn_l1033_103346

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 6

def probability_all_white : ℚ := 4 / 715

theorem probability_all_white_drawn (total : ℕ) (white : ℕ) (black : ℕ) (drawn : ℕ) :
  total = white + black →
  white ≥ drawn →
  probability_all_white = (Nat.choose white drawn : ℚ) / (Nat.choose total drawn : ℚ) :=
sorry

end NUMINAMATH_CALUDE_probability_all_white_drawn_l1033_103346


namespace NUMINAMATH_CALUDE_solution_a_l1033_103330

-- Define the function F
def F (a b c : ℝ) : ℝ := a * b^3 + c

-- State the theorem
theorem solution_a : ∃ (a : ℝ), F a 2 3 = F a 3 10 ∧ a = -7/19 := by
  sorry

end NUMINAMATH_CALUDE_solution_a_l1033_103330


namespace NUMINAMATH_CALUDE_equivalent_functions_l1033_103356

theorem equivalent_functions (x : ℝ) : x^2 = (x^6)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_functions_l1033_103356


namespace NUMINAMATH_CALUDE_decline_type_composition_l1033_103379

/-- Represents the age composition of a population --/
inductive AgeComposition
  | Growth
  | Stable
  | Decline

/-- Represents the relative distribution of age groups in a population --/
structure PopulationDistribution where
  young : ℕ
  adult : ℕ
  elderly : ℕ

/-- Determines the age composition based on the population distribution --/
def determineAgeComposition (pop : PopulationDistribution) : AgeComposition :=
  sorry

/-- Theorem stating that a population with fewer young individuals and more adults and elderly individuals has a decline type age composition --/
theorem decline_type_composition (pop : PopulationDistribution) 
  (h1 : pop.young < pop.adult)
  (h2 : pop.young < pop.elderly) :
  determineAgeComposition pop = AgeComposition.Decline :=
sorry

end NUMINAMATH_CALUDE_decline_type_composition_l1033_103379


namespace NUMINAMATH_CALUDE_battery_problem_l1033_103347

theorem battery_problem :
  ∀ (x y z : ℚ),
  (x > 0) → (y > 0) → (z > 0) →
  (4*x + 18*y + 16*z = 4*x + 15*y + 24*z) →
  (4*x + 18*y + 16*z = 6*x + 12*y + 20*z) →
  (∃ (W : ℚ), W * z = 4*x + 18*y + 16*z ∧ W = 48) :=
by
  sorry

end NUMINAMATH_CALUDE_battery_problem_l1033_103347


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1033_103399

theorem necessary_not_sufficient_condition (a b c : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a / b = b / c ∧ b ^ 2 = a * c) ∧ 
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b ^ 2 = a * c ∧ a / b ≠ b / c) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1033_103399


namespace NUMINAMATH_CALUDE_triangle_inequality_with_area_l1033_103320

/-- Triangle inequality theorem for sides and area -/
theorem triangle_inequality_with_area (a b c S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : S > 0)
  (h5 : S = Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S + (a - b)^2 + (b - c)^2 + (c - a)^2 ∧ 
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * S + (a - b)^2 + (b - c)^2 + (c - a)^2 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_area_l1033_103320


namespace NUMINAMATH_CALUDE_min_cosine_sqrt3_sine_l1033_103395

theorem min_cosine_sqrt3_sine (A : Real) :
  let f := λ A : Real => Real.cos (A / 2) + Real.sqrt 3 * Real.sin (A / 2)
  ∃ (min : Real), f min ≤ f A ∧ min = 840 * Real.pi / 180 :=
sorry

end NUMINAMATH_CALUDE_min_cosine_sqrt3_sine_l1033_103395


namespace NUMINAMATH_CALUDE_lock_min_moves_l1033_103376

/-- Represents a combination lock with n discs, each having d digits -/
structure CombinationLock (n : ℕ) (d : ℕ) where
  discs : Fin n → Fin d

/-- Represents a move on the lock -/
def move (lock : CombinationLock n d) (disc : Fin n) (direction : Bool) : CombinationLock n d :=
  sorry

/-- Checks if a combination is valid (for part b) -/
def is_valid_combination (lock : CombinationLock n d) : Bool :=
  sorry

/-- The number of moves required to ensure finding the correct combination -/
def min_moves (n : ℕ) (d : ℕ) (initial : CombinationLock n d) (valid : CombinationLock n d → Bool) : ℕ :=
  sorry

theorem lock_min_moves :
  let n : ℕ := 6
  let d : ℕ := 10
  let initial : CombinationLock n d := sorry
  let valid_a : CombinationLock n d → Bool := λ _ => true
  let valid_b : CombinationLock n d → Bool := is_valid_combination
  (∀ (i : Fin n), initial.discs i = 0) →
  min_moves n d initial valid_a = 999998 ∧
  min_moves n d initial valid_b = 999998 :=
sorry

end NUMINAMATH_CALUDE_lock_min_moves_l1033_103376


namespace NUMINAMATH_CALUDE_value_equivalence_l1033_103382

theorem value_equivalence : 3000 * (3000^3000 + 3000^2999) = 3001 * 3000^3000 := by
  sorry

end NUMINAMATH_CALUDE_value_equivalence_l1033_103382


namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l1033_103391

theorem smallest_x_for_perfect_cube : ∃ (x : ℕ+), 
  (∀ (y : ℕ+), ∃ (M : ℤ), 1800 * y = M^3 → x ≤ y) ∧
  (∃ (M : ℤ), 1800 * x = M^3) ∧
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l1033_103391


namespace NUMINAMATH_CALUDE_triangle_pq_distance_l1033_103342

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  AB = 4 ∧ AC = 3 ∧ BC = Real.sqrt 37

-- Define point P as the midpoint of AB
def Midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define point Q on AC at distance 1 from C
def PointOnLine (Q A C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, Q.1 = C.1 + t * (A.1 - C.1) ∧ Q.2 = C.2 + t * (A.2 - C.2)

def DistanceFromC (Q C : ℝ × ℝ) : Prop :=
  Real.sqrt ((Q.1 - C.1)^2 + (Q.2 - C.2)^2) = 1

-- Theorem statement
theorem triangle_pq_distance (A B C P Q : ℝ × ℝ) :
  Triangle A B C → Midpoint P A B → PointOnLine Q A C → DistanceFromC Q C →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_pq_distance_l1033_103342


namespace NUMINAMATH_CALUDE_solid_price_is_four_l1033_103354

/-- The price of solid color gift wrap per roll -/
def solid_price : ℝ := 4

/-- The total number of rolls sold -/
def total_rolls : ℕ := 480

/-- The total amount of money collected in dollars -/
def total_money : ℝ := 2340

/-- The number of print rolls sold -/
def print_rolls : ℕ := 210

/-- The price of print gift wrap per roll in dollars -/
def print_price : ℝ := 6

/-- Theorem stating that the price of solid color gift wrap is $4.00 per roll -/
theorem solid_price_is_four :
  solid_price = (total_money - print_rolls * print_price) / (total_rolls - print_rolls) :=
by sorry

end NUMINAMATH_CALUDE_solid_price_is_four_l1033_103354


namespace NUMINAMATH_CALUDE_molecular_weight_of_Y_l1033_103333

/-- Represents a chemical compound with its molecular weight -/
structure Compound where
  molecularWeight : ℝ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactant1 : Compound
  reactant2 : Compound
  product : Compound
  reactant2Coefficient : ℕ

/-- The law of conservation of mass in a chemical reaction -/
def conservationOfMass (r : Reaction) : Prop :=
  r.reactant1.molecularWeight + r.reactant2Coefficient * r.reactant2.molecularWeight = r.product.molecularWeight

/-- Theorem: The molecular weight of Y in the given reaction -/
theorem molecular_weight_of_Y : 
  let X : Compound := ⟨136⟩
  let C6H8O7 : Compound := ⟨192⟩
  let Y : Compound := ⟨1096⟩
  let reaction : Reaction := ⟨X, C6H8O7, Y, 5⟩
  conservationOfMass reaction := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_of_Y_l1033_103333


namespace NUMINAMATH_CALUDE_strongest_signal_l1033_103366

def signal_strength (x : ℤ) : ℝ := |x|

def is_stronger (x y : ℤ) : Prop := signal_strength x < signal_strength y

theorem strongest_signal :
  let signals : List ℤ := [-50, -60, -70, -80]
  ∀ s ∈ signals, s ≠ -50 → is_stronger (-50) s :=
by sorry

end NUMINAMATH_CALUDE_strongest_signal_l1033_103366


namespace NUMINAMATH_CALUDE_f_properties_l1033_103303

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2*x - 1|

-- State the theorem
theorem f_properties :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, m = 1 → (f x m ≥ 3 ↔ x ≤ -1 ∨ x ≥ 1)) ∧
  (∀ x : ℝ, x ∈ Set.Icc m (2*m^2) → (1/2 * f x m ≤ |x + 1|)) ↔
  (1/2 < m ∧ m ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1033_103303


namespace NUMINAMATH_CALUDE_right_triangle_sin_R_l1033_103319

theorem right_triangle_sin_R (P Q R : ℝ) (h_right_triangle : P + Q + R = π) 
  (h_sin_P : Real.sin P = 3/5) (h_sin_Q : Real.sin Q = 1) : Real.sin R = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_R_l1033_103319


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l1033_103390

/-- Calculates the measured weight loss percentage at the final weigh-in -/
def measuredWeightLoss (initialLoss : ℝ) (clothesWeight : ℝ) (waterRetention : ℝ) : ℝ :=
  (1 - (1 - initialLoss) * (1 + clothesWeight) * (1 + waterRetention)) * 100

/-- Theorem stating the measured weight loss percentage for given conditions -/
theorem weight_loss_challenge (initialLoss clothesWeight waterRetention : ℝ) 
  (h1 : initialLoss = 0.11)
  (h2 : clothesWeight = 0.02)
  (h3 : waterRetention = 0.015) :
  abs (measuredWeightLoss initialLoss clothesWeight waterRetention - 7.64) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l1033_103390


namespace NUMINAMATH_CALUDE_specific_sequence_terms_l1033_103350

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℕ
  last : ℕ
  diff : ℕ

/-- Calculates the number of terms in an arithmetic sequence -/
def numTerms (seq : ArithmeticSequence) : ℕ :=
  (seq.last - seq.first) / seq.diff + 1

theorem specific_sequence_terms : 
  let seq := ArithmeticSequence.mk 2 3007 5
  numTerms seq = 602 := by
  sorry

end NUMINAMATH_CALUDE_specific_sequence_terms_l1033_103350


namespace NUMINAMATH_CALUDE_initial_typists_count_l1033_103364

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 20

/-- The number of letters typed by the initial group in 20 minutes -/
def letters_20min : ℕ := 44

/-- The number of typists in the second group -/
def second_group_typists : ℕ := 30

/-- The number of letters typed by the second group in 1 hour -/
def letters_1hour : ℕ := 198

/-- The ratio of 1 hour to 20 minutes -/
def time_ratio : ℕ := 3

theorem initial_typists_count :
  initial_typists * second_group_typists * letters_20min * time_ratio = letters_1hour * initial_typists * initial_typists :=
sorry

end NUMINAMATH_CALUDE_initial_typists_count_l1033_103364


namespace NUMINAMATH_CALUDE_tangent_line_slope_l1033_103341

/-- The curve y = x³ + x + 16 -/
def f (x : ℝ) : ℝ := x^3 + x + 16

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

/-- The line ℓ passing through (0,0) and tangent to f -/
structure TangentLine where
  t : ℝ
  slope : ℝ
  tangent_point : (ℝ × ℝ) := (t, f t)
  passes_origin : slope * t = f t
  is_tangent : slope = f' t

theorem tangent_line_slope : 
  ∃ (ℓ : TangentLine), ℓ.slope = 13 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l1033_103341


namespace NUMINAMATH_CALUDE_family_bought_three_soft_tacos_l1033_103374

/-- Represents the taco truck's sales during lunch rush -/
structure TacoSales where
  soft_taco_price : ℕ
  hard_taco_price : ℕ
  family_hard_tacos : ℕ
  other_customers : ℕ
  soft_tacos_per_customer : ℕ
  total_revenue : ℕ

/-- Calculates the number of soft tacos bought by the family -/
def family_soft_tacos (sales : TacoSales) : ℕ :=
  (sales.total_revenue -
   sales.family_hard_tacos * sales.hard_taco_price -
   sales.other_customers * sales.soft_tacos_per_customer * sales.soft_taco_price) /
  sales.soft_taco_price

/-- Theorem stating that the family bought 3 soft tacos -/
theorem family_bought_three_soft_tacos (sales : TacoSales)
  (h1 : sales.soft_taco_price = 2)
  (h2 : sales.hard_taco_price = 5)
  (h3 : sales.family_hard_tacos = 4)
  (h4 : sales.other_customers = 10)
  (h5 : sales.soft_tacos_per_customer = 2)
  (h6 : sales.total_revenue = 66) :
  family_soft_tacos sales = 3 := by
  sorry

end NUMINAMATH_CALUDE_family_bought_three_soft_tacos_l1033_103374


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1033_103377

theorem smallest_x_absolute_value_equation :
  ∃ x : ℝ, (∀ y : ℝ, y * |y| = 3 * y + 2 → x ≤ y) ∧ x * |x| = 3 * x + 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1033_103377


namespace NUMINAMATH_CALUDE_sum_of_powers_mod_17_l1033_103381

theorem sum_of_powers_mod_17 : ∃ (a b c d : ℕ), 
  (3 * a) % 17 = 1 ∧ 
  (3 * b) % 17 = 3 ∧ 
  (3 * c) % 17 = 9 ∧ 
  (3 * d) % 17 = 10 ∧ 
  (a + b + c + d) % 17 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_mod_17_l1033_103381


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1033_103357

theorem inequality_solution_range (a : ℝ) : 
  ((3 - a) * (3 + 2*a - 1)^2 * (3 - 3*a) ≤ 0) →
  (a = -1 ∨ (1 ≤ a ∧ a ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1033_103357


namespace NUMINAMATH_CALUDE_f_one_equals_four_l1033_103396

/-- A function f(x) that is always non-negative for real x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 3*a - 9

/-- The theorem stating that f(1) = 4 given the conditions -/
theorem f_one_equals_four (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : f a 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_one_equals_four_l1033_103396


namespace NUMINAMATH_CALUDE_ratio_equality_l1033_103378

theorem ratio_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  (a / 4) / (b / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1033_103378


namespace NUMINAMATH_CALUDE_x_cube_minus_six_x_squared_l1033_103389

theorem x_cube_minus_six_x_squared (x : ℝ) : x = 3 → x^6 - 6*x^2 = 675 := by
  sorry

end NUMINAMATH_CALUDE_x_cube_minus_six_x_squared_l1033_103389


namespace NUMINAMATH_CALUDE_product_remainder_l1033_103368

theorem product_remainder (a b c : ℕ) (h : a * b * c = 1225 * 1227 * 1229) : 
  (a * b * c) % 12 = 7 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_l1033_103368


namespace NUMINAMATH_CALUDE_honey_shop_problem_l1033_103338

/-- The honey shop problem -/
theorem honey_shop_problem (bulk_price tax min_spend penny_paid : ℚ)
  (h1 : bulk_price = 5)
  (h2 : tax = 1)
  (h3 : min_spend = 40)
  (h4 : penny_paid = 240) :
  (penny_paid / (bulk_price + tax) - min_spend / bulk_price) = 32 := by
  sorry

end NUMINAMATH_CALUDE_honey_shop_problem_l1033_103338


namespace NUMINAMATH_CALUDE_sum_of_P_and_R_is_eight_l1033_103325

theorem sum_of_P_and_R_is_eight :
  ∀ (P Q R S : ℕ),
    P ∈ ({1, 2, 3, 5} : Set ℕ) →
    Q ∈ ({1, 2, 3, 5} : Set ℕ) →
    R ∈ ({1, 2, 3, 5} : Set ℕ) →
    S ∈ ({1, 2, 3, 5} : Set ℕ) →
    P ≠ Q → P ≠ R → P ≠ S → Q ≠ R → Q ≠ S → R ≠ S →
    (P : ℚ) / Q - (R : ℚ) / S = 2 →
    P + R = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_P_and_R_is_eight_l1033_103325


namespace NUMINAMATH_CALUDE_triangle_side_length_l1033_103335

/-- Given a triangle ABC where ∠B = 45°, AB = 100, and AC = 100√2, prove that BC = 100√(5 + √2(√6 - √2)). -/
theorem triangle_side_length (A B C : ℝ×ℝ) : 
  let angleB := Real.arccos ((B.1 - A.1) / Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))
  let sideAB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let sideAC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  angleB = π/4 ∧ sideAB = 100 ∧ sideAC = 100 * Real.sqrt 2 →
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 100 * Real.sqrt (5 + Real.sqrt 2 * (Real.sqrt 6 - Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1033_103335


namespace NUMINAMATH_CALUDE_ninth_grade_students_count_l1033_103352

def total_payment : ℝ := 1936
def additional_sets : ℕ := 88
def discount_rate : ℝ := 0.2

theorem ninth_grade_students_count :
  ∃ x : ℕ, 
    (total_payment / x) * (1 - discount_rate) = total_payment / (x + additional_sets) ∧
    x = 352 := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_students_count_l1033_103352


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l1033_103388

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 - 30 * x + c = 0) →
  a + c = 41 →
  a < c →
  (a = (41 + Real.sqrt 781) / 2 ∧ c = (41 - Real.sqrt 781) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l1033_103388


namespace NUMINAMATH_CALUDE_root_sum_squares_l1033_103385

theorem root_sum_squares (p q r : ℝ) : 
  (p^3 - 15*p^2 + 22*p - 8 = 0) → 
  (q^3 - 15*q^2 + 22*q - 8 = 0) → 
  (r^3 - 15*r^2 + 22*r - 8 = 0) → 
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 406 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l1033_103385


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1033_103305

theorem necessary_not_sufficient (a : ℝ) :
  (∀ a, 1 / a > 1 → a < 1) ∧ (∃ a, a < 1 ∧ 1 / a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1033_103305


namespace NUMINAMATH_CALUDE_exists_non_isosceles_equidistant_inscribed_center_l1033_103336

/-- A triangle with side lengths a, b, and c. -/
structure Triangle :=
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

/-- The center of the inscribed circle of a triangle. -/
def InscribedCenter (t : Triangle) : ℝ × ℝ := sorry

/-- The midpoint of a line segment. -/
def Midpoint (a b : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The distance between two points. -/
def Distance (a b : ℝ × ℝ) : ℝ := sorry

/-- Predicate to check if a triangle is isosceles. -/
def IsIsosceles (t : Triangle) : Prop := 
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Theorem: There exists a non-isosceles triangle where the center of its inscribed circle
    is equidistant from the midpoints of two sides. -/
theorem exists_non_isosceles_equidistant_inscribed_center :
  ∃ (t : Triangle), 
    ¬IsIsosceles t ∧
    ∃ (s₁ s₂ : ℝ × ℝ), 
      Distance (InscribedCenter t) (Midpoint s₁ s₂) = 
      Distance (InscribedCenter t) (Midpoint s₂ (s₁.1 + t.a - s₁.1, s₁.2 + t.b - s₁.2)) :=
sorry

end NUMINAMATH_CALUDE_exists_non_isosceles_equidistant_inscribed_center_l1033_103336


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1033_103387

theorem fraction_sum_equals_decimal : (3 / 15) + (5 / 125) + (7 / 1000) = 0.247 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1033_103387


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_roots_l1033_103372

theorem cubic_polynomial_integer_roots :
  ∀ (a c : ℤ), ∃ (x y z : ℤ),
    ∀ (X : ℤ), X^3 + a*X^2 - X + c = 0 ↔ (X = x ∨ X = y ∨ X = z) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_roots_l1033_103372


namespace NUMINAMATH_CALUDE_inequality_solution_empty_l1033_103340

theorem inequality_solution_empty (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_empty_l1033_103340


namespace NUMINAMATH_CALUDE_least_multiple_divisible_l1033_103362

theorem least_multiple_divisible (x : ℕ) : 
  (∀ y : ℕ, y > 0 ∧ y < 57 → ¬(57 ∣ 23 * y)) ∧ (57 ∣ 23 * 57) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_divisible_l1033_103362


namespace NUMINAMATH_CALUDE_parabola_vertex_l1033_103375

/-- The vertex of the parabola y = x^2 - 2 is at the point (0, -2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = x^2 - 2 → (0, -2) = (x, y) ↔ x = 0 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1033_103375


namespace NUMINAMATH_CALUDE_stable_number_theorem_l1033_103327

/-- Definition of a stable number -/
def is_stable (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 ≠ 0) ∧ ((n / 10) % 10 ≠ 0) ∧ (n % 10 ≠ 0) ∧
  (n / 100 + (n / 10) % 10 > n % 10) ∧
  (n / 100 + n % 10 > (n / 10) % 10) ∧
  ((n / 10) % 10 + n % 10 > n / 100)

/-- Definition of F(n) -/
def F (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10) % 10

/-- Definition of Q(n) -/
def Q (n : ℕ) : ℕ := ((n / 10) % 10) * 10 + n / 100

/-- Main theorem -/
theorem stable_number_theorem (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 5) (hb : 1 ≤ b ∧ b ≤ 4) :
  let s := 100 * a + 101 * b + 30
  is_stable s ∧ (5 * F s + 2 * Q s) % 11 = 0 → s = 432 ∨ s = 534 := by
  sorry

end NUMINAMATH_CALUDE_stable_number_theorem_l1033_103327


namespace NUMINAMATH_CALUDE_irrational_count_l1033_103300

-- Define the set of numbers
def S : Set ℝ := {4 * Real.pi, 0, Real.sqrt 7, Real.sqrt 16 / 2, 0.1, 0.212212221}

-- Define a function to count irrational numbers in a set
def count_irrational (T : Set ℝ) : ℕ := sorry

-- Theorem statement
theorem irrational_count : count_irrational S = 3 := by sorry

end NUMINAMATH_CALUDE_irrational_count_l1033_103300


namespace NUMINAMATH_CALUDE_employee_pay_l1033_103307

theorem employee_pay (total_pay x y : ℝ) : 
  total_pay = 880 ∧ x = 1.2 * y → y = 400 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l1033_103307


namespace NUMINAMATH_CALUDE_complex_real_part_l1033_103383

theorem complex_real_part (z : ℂ) (h : (z^2 + z).im = 0) : z.re = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_part_l1033_103383


namespace NUMINAMATH_CALUDE_hypotenuse_square_l1033_103304

-- Define a right triangle with integer legs
def RightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧ b = a + 1

-- Theorem statement
theorem hypotenuse_square (a : ℕ) :
  ∀ b c : ℕ, RightTriangle a b c → c^2 = 2*a^2 + 2*a + 1 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_square_l1033_103304


namespace NUMINAMATH_CALUDE_linear_function_increases_iff_positive_slope_increasing_linear_function_k_equals_four_l1033_103343

/-- A linear function y = mx + b increases if and only if its slope m is positive -/
theorem linear_function_increases_iff_positive_slope {m b : ℝ} :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → m * x₁ + b < m * x₂ + b) ↔ m > 0 := by sorry

/-- For the function y = (k - 3)x + 2, if y increases as x increases, then k = 4 -/
theorem increasing_linear_function_k_equals_four (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (k - 3) * x₁ + 2 < (k - 3) * x₂ + 2) → k = 4 := by sorry

end NUMINAMATH_CALUDE_linear_function_increases_iff_positive_slope_increasing_linear_function_k_equals_four_l1033_103343


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l1033_103331

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of boys in the arrangement. -/
def num_boys : ℕ := 4

/-- The number of girls in the arrangement. -/
def num_girls : ℕ := 2

/-- The total number of people in the arrangement. -/
def total_people : ℕ := num_boys + num_girls

theorem photo_arrangement_count :
  arrangements total_people -
  arrangements (total_people - 1) -
  arrangements (total_people - 1) +
  arrangements (total_people - 2) = 504 := by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l1033_103331


namespace NUMINAMATH_CALUDE_horsemen_speeds_exist_l1033_103334

/-- Represents a set of speeds for horsemen on a circular track -/
def SpeedSet (n : ℕ) := Fin n → ℝ

/-- Predicate that checks if all speeds in a set are distinct and positive -/
def distinct_positive (s : SpeedSet n) : Prop :=
  ∀ i j, i ≠ j → s i ≠ s j ∧ s i > 0 ∧ s j > 0

/-- Predicate that checks if all overtakings occur at a single point -/
def single_overtaking_point (s : SpeedSet n) : Prop :=
  ∀ i j, i ≠ j → ∃ k : ℤ, (s i) / (s i - s j) = k

/-- Theorem stating that for any number of horsemen (≥ 3), 
    there exists a set of speeds satisfying the required conditions -/
theorem horsemen_speeds_exist (n : ℕ) (h : n ≥ 3) :
  ∃ (s : SpeedSet n), distinct_positive s ∧ single_overtaking_point s :=
sorry

end NUMINAMATH_CALUDE_horsemen_speeds_exist_l1033_103334


namespace NUMINAMATH_CALUDE_marble_draw_theorem_l1033_103373

/-- Represents the number of marbles of each color in the bucket -/
structure MarbleCounts where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat
  orange : Nat
  purple : Nat

/-- The actual counts of marbles in the bucket -/
def initialCounts : MarbleCounts :=
  { red := 35, green := 25, blue := 24, yellow := 18, orange := 15, purple := 12 }

/-- The minimum number of marbles to guarantee at least 20 of a single color -/
def minMarblesToDraw : Nat := 103

theorem marble_draw_theorem (counts : MarbleCounts := initialCounts) :
  (∀ n : Nat, n < minMarblesToDraw →
    ∃ c : MarbleCounts, c.red < 20 ∧ c.green < 20 ∧ c.blue < 20 ∧
      c.yellow < 20 ∧ c.orange < 20 ∧ c.purple < 20 ∧
      c.red + c.green + c.blue + c.yellow + c.orange + c.purple = n) ∧
  (∀ c : MarbleCounts,
    c.red + c.green + c.blue + c.yellow + c.orange + c.purple = minMarblesToDraw →
    c.red ≥ 20 ∨ c.green ≥ 20 ∨ c.blue ≥ 20 ∨ c.yellow ≥ 20 ∨ c.orange ≥ 20 ∨ c.purple ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_marble_draw_theorem_l1033_103373


namespace NUMINAMATH_CALUDE_other_factor_of_prime_multiple_l1033_103318

theorem other_factor_of_prime_multiple (p n : ℕ) : 
  Nat.Prime p → 
  (∃ k, n = k * p) → 
  (∀ d : ℕ, d ∣ n ↔ d = 1 ∨ d = n) → 
  ∃ k : ℕ, n = k * p ∧ k = 1 :=
by sorry

end NUMINAMATH_CALUDE_other_factor_of_prime_multiple_l1033_103318


namespace NUMINAMATH_CALUDE_log_simplification_l1033_103314

theorem log_simplification (x y z w t v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (ht : t > 0) (hv : v > 0) :
  Real.log (x / z) + Real.log (z / y) + Real.log (y / w) - Real.log (x * v / (w * t)) = Real.log (t / v) :=
by sorry

end NUMINAMATH_CALUDE_log_simplification_l1033_103314


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1033_103306

theorem fraction_equivalence : 
  ∀ (n : ℚ), (3 + n) / (4 + n) = 4 / 5 → n = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1033_103306


namespace NUMINAMATH_CALUDE_problem_solution_l1033_103328

theorem problem_solution (x z : ℝ) (hx : x ≠ 0) (h1 : x/3 = z^2 + 1) (h2 : x/5 = 5*z + 2) :
  x = (685 + 25 * Real.sqrt 541) / 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1033_103328


namespace NUMINAMATH_CALUDE_third_number_problem_l1033_103339

theorem third_number_problem (first second third : ℕ) : 
  (3 * first + 3 * second + 3 * third + 11 = 170) →
  (first = 16) →
  (second = 17) →
  third = 20 := by
sorry

end NUMINAMATH_CALUDE_third_number_problem_l1033_103339


namespace NUMINAMATH_CALUDE_tv_price_change_l1033_103394

theorem tv_price_change (initial_price : ℝ) (h : initial_price > 0) : 
  let price_after_decrease : ℝ := 0.8 * initial_price
  let final_price : ℝ := 1.24 * initial_price
  ∃ x : ℝ, price_after_decrease * (1 + x / 100) = final_price ∧ x = 55 := by
sorry

end NUMINAMATH_CALUDE_tv_price_change_l1033_103394


namespace NUMINAMATH_CALUDE_calculate_biology_marks_l1033_103322

theorem calculate_biology_marks (english math physics chemistry : ℕ) (average : ℚ) :
  english = 96 →
  math = 95 →
  physics = 82 →
  chemistry = 87 →
  average = 90.4 →
  (english + math + physics + chemistry + (5 * average - (english + math + physics + chemistry))) / 5 = average :=
by sorry

end NUMINAMATH_CALUDE_calculate_biology_marks_l1033_103322


namespace NUMINAMATH_CALUDE_integer_solution_l1033_103363

theorem integer_solution (n : ℤ) : n + 5 > 7 ∧ -3*n > -15 → n = 3 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_l1033_103363


namespace NUMINAMATH_CALUDE_gcd_14568_78452_l1033_103358

theorem gcd_14568_78452 : Int.gcd 14568 78452 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_14568_78452_l1033_103358


namespace NUMINAMATH_CALUDE_part_one_part_two_l1033_103317

-- Define polynomials A, B, and C
def A (x y : ℝ) : ℝ := x^2 + x*y + 3*y
def B (x y : ℝ) : ℝ := x^2 - x*y

-- Theorem for part 1
theorem part_one (x y : ℝ) : 3 * A x y - B x y = 2*x^2 + 4*x*y + 9*y := by sorry

-- Theorem for part 2
theorem part_two (x y : ℝ) :
  (∃ C : ℝ → ℝ → ℝ, A x y + (1/3) * C x y = 2*x*y + 5*y) →
  (∃ C : ℝ → ℝ → ℝ, C x y = -3*x^2 + 3*x*y + 6*y) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1033_103317


namespace NUMINAMATH_CALUDE_height_increase_calculation_l1033_103345

/-- Represents the increase in height per decade for a specific plant species -/
def height_increase_per_decade : ℝ := sorry

/-- The number of decades in 4 centuries -/
def decades_in_four_centuries : ℕ := 40

/-- The total increase in height over 4 centuries in meters -/
def total_height_increase : ℝ := 3000

theorem height_increase_calculation :
  height_increase_per_decade * (decades_in_four_centuries : ℝ) = total_height_increase ∧
  height_increase_per_decade = 75 := by sorry

end NUMINAMATH_CALUDE_height_increase_calculation_l1033_103345


namespace NUMINAMATH_CALUDE_expression_evaluation_l1033_103348

theorem expression_evaluation (a b c : ℝ) (ha : a = 3) (hb : b = 2) (hc : c = 1) :
  (c^2 + a^2 + b)^2 - (c^2 + a^2 - b)^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1033_103348


namespace NUMINAMATH_CALUDE_sum_mod_nine_l1033_103360

theorem sum_mod_nine : (7155 + 7156 + 7157 + 7158 + 7159) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l1033_103360


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1033_103353

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 7 → 
  (a + b + c) / 3 = a + 12 → 
  (a + b + c) / 3 = c - 18 → 
  a + b + c = 39 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1033_103353


namespace NUMINAMATH_CALUDE_product_inequality_l1033_103308

theorem product_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq_one : a + b + c = 1) :
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1033_103308


namespace NUMINAMATH_CALUDE_parallel_iff_equal_slope_l1033_103397

/-- Two lines in the plane -/
structure Line where
  k : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, k * x + y + c = 0

/-- Parallel lines have the same slope -/
def parallel (l₁ l₂ : Line) : Prop := l₁.k = l₂.k

theorem parallel_iff_equal_slope (l₁ l₂ : Line) : 
  parallel l₁ l₂ ↔ l₁.k = l₂.k :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_equal_slope_l1033_103397


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_coefficient_l1033_103329

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y - 1 = 0
def l₅ (a x y : ℝ) : Prop := a * x - 2 * y + 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem 1: The equation of line l₄
theorem parallel_line_equation : 
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ (P.1 = x ∧ P.2 = y ∨ l₃ x y)) ∧ m = 1/2 ∧ b = 3 := by
  sorry

-- Theorem 2: The value of a for perpendicular lines
theorem perpendicular_line_coefficient :
  ∃! a : ℝ, ∀ x y : ℝ, (l₅ a x y ∧ l₂ x y) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_coefficient_l1033_103329


namespace NUMINAMATH_CALUDE_simplify_expression_l1033_103316

theorem simplify_expression : (256 : ℝ)^(1/4) * (125 : ℝ)^(1/3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1033_103316


namespace NUMINAMATH_CALUDE_average_writing_rate_l1033_103392

/-- Given a writer who completed 50,000 words in 100 hours, 
    prove that the average writing rate is 500 words per hour. -/
theorem average_writing_rate 
  (total_words : ℕ) 
  (total_hours : ℕ) 
  (h1 : total_words = 50000) 
  (h2 : total_hours = 100) : 
  (total_words : ℚ) / total_hours = 500 := by
  sorry

end NUMINAMATH_CALUDE_average_writing_rate_l1033_103392


namespace NUMINAMATH_CALUDE_point_on_graph_l1033_103321

/-- A point (x, y) lies on the graph of y = -6/x if and only if xy = -6 -/
def lies_on_graph (x y : ℝ) : Prop := x * y = -6

/-- The function f(x) = -6/x -/
noncomputable def f (x : ℝ) : ℝ := -6 / x

theorem point_on_graph : lies_on_graph 2 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l1033_103321
