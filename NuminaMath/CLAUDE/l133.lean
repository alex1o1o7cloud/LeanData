import Mathlib

namespace NUMINAMATH_CALUDE_jake_work_hours_l133_13394

/-- Calculates the number of hours needed to work off a debt -/
def hoursToWorkOff (initialDebt : ℚ) (amountPaid : ℚ) (hourlyRate : ℚ) : ℚ :=
  (initialDebt - amountPaid) / hourlyRate

/-- Proves that Jake needs to work 4 hours to pay off his debt -/
theorem jake_work_hours :
  let initialDebt : ℚ := 100
  let amountPaid : ℚ := 40
  let hourlyRate : ℚ := 15
  hoursToWorkOff initialDebt amountPaid hourlyRate = 4 := by
  sorry

end NUMINAMATH_CALUDE_jake_work_hours_l133_13394


namespace NUMINAMATH_CALUDE_triangle_properties_l133_13397

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let ab := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let bc := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let ac := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (ab + bc + ac) / 2
  ab = 6 ∧ bc = 5 ∧ Real.sqrt (s * (s - ab) * (s - bc) * (s - ac)) = 9

-- Define an acute triangle
def AcuteTriangle (A B C : ℝ × ℝ) : Prop :=
  Triangle A B C ∧
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) > 0 ∧
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) > 0 ∧
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) > 0

theorem triangle_properties (A B C : ℝ × ℝ) :
  Triangle A B C →
  (∃ ac : ℝ, (ac = Real.sqrt 13 ∨ ac = Real.sqrt 109) ∧
   ac = Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)) ∧
  (AcuteTriangle A B C →
   ∃ angle_A : ℝ,
   Real.cos (2 * angle_A + π / 6) = (-5 * Real.sqrt 3 - 12) / 26) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l133_13397


namespace NUMINAMATH_CALUDE_max_salary_in_semipro_league_l133_13387

/-- Represents a baseball team -/
structure Team where
  players : Nat
  minSalary : Nat
  maxTotalSalary : Nat

/-- Calculates the maximum possible salary for a single player in a team -/
def maxSinglePlayerSalary (team : Team) : Nat :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player in the given conditions -/
theorem max_salary_in_semipro_league :
  let team : Team := {
    players := 21,
    minSalary := 15000,
    maxTotalSalary := 700000
  }
  maxSinglePlayerSalary team = 400000 := by sorry

end NUMINAMATH_CALUDE_max_salary_in_semipro_league_l133_13387


namespace NUMINAMATH_CALUDE_initial_marbles_relationship_l133_13350

/-- Represents the marble collection problem --/
structure MarbleCollection where
  initial : ℕ  -- Initial number of marbles
  lost : ℕ     -- Number of marbles lost
  found : ℕ    -- Number of marbles found
  current : ℕ  -- Current number of marbles after losses and finds

/-- The marble collection satisfies the problem conditions --/
def validCollection (m : MarbleCollection) : Prop :=
  m.lost = 16 ∧ m.found = 8 ∧ m.lost - m.found = 8 ∧ m.current = m.initial - m.lost + m.found

/-- Theorem stating the relationship between initial and current marbles --/
theorem initial_marbles_relationship (m : MarbleCollection) 
  (h : validCollection m) : m.initial = m.current + 8 := by
  sorry

#check initial_marbles_relationship

end NUMINAMATH_CALUDE_initial_marbles_relationship_l133_13350


namespace NUMINAMATH_CALUDE_seashells_given_to_sam_l133_13388

theorem seashells_given_to_sam (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 70) 
  (h2 : remaining_seashells = 27) : 
  initial_seashells - remaining_seashells = 43 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_sam_l133_13388


namespace NUMINAMATH_CALUDE_new_boarders_correct_l133_13304

-- Define the initial conditions
def initial_boarders : ℕ := 120
def initial_ratio_boarders : ℕ := 2
def initial_ratio_day : ℕ := 5
def new_ratio_boarders : ℕ := 1
def new_ratio_day : ℕ := 2

-- Define the function to calculate the number of new boarders
def new_boarders : ℕ := 30

-- Theorem statement
theorem new_boarders_correct :
  let initial_day_students := (initial_boarders * initial_ratio_day) / initial_ratio_boarders
  (new_ratio_boarders * (initial_boarders + new_boarders)) = (new_ratio_day * initial_day_students) :=
by sorry

end NUMINAMATH_CALUDE_new_boarders_correct_l133_13304


namespace NUMINAMATH_CALUDE_car_distance_ratio_l133_13370

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

/-- Theorem stating the ratio of distances covered by Car A and Car B -/
theorem car_distance_ratio (carA carB : Car)
    (hA : carA = { speed := 80, time := 5 })
    (hB : carB = { speed := 100, time := 2 }) :
    distance carA / distance carB = 2 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_ratio_l133_13370


namespace NUMINAMATH_CALUDE_conveyor_belts_combined_time_l133_13382

/-- The time taken for two conveyor belts to move one day's coal output together -/
theorem conveyor_belts_combined_time (old_rate new_rate : ℝ) 
  (h1 : old_rate = 1 / 21)
  (h2 : new_rate = 1 / 15) : 
  1 / (old_rate + new_rate) = 35 / 4 := by
  sorry

end NUMINAMATH_CALUDE_conveyor_belts_combined_time_l133_13382


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l133_13332

def total_balls : ℕ := 11
def white_balls : ℕ := 4
def black_balls : ℕ := 7
def drawn_balls : ℕ := 3

def probability_all_white : ℚ :=
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ)

theorem probability_three_white_balls :
  probability_all_white = 4 / 165 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l133_13332


namespace NUMINAMATH_CALUDE_cellar_water_pumping_time_l133_13379

/-- Calculates the time needed to pump out water from a flooded cellar. -/
theorem cellar_water_pumping_time
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (num_pumps : ℕ)
  (pump_rate : ℝ)
  (water_density : ℝ)
  (h_length : length = 30)
  (h_width : width = 40)
  (h_depth : depth = 2)
  (h_num_pumps : num_pumps = 4)
  (h_pump_rate : pump_rate = 10)
  (h_water_density : water_density = 7.5) :
  (length * width * depth * water_density) / (num_pumps * pump_rate) = 450 :=
sorry

end NUMINAMATH_CALUDE_cellar_water_pumping_time_l133_13379


namespace NUMINAMATH_CALUDE_minor_arc_intercept_l133_13360

/-- Given a circle x^2 + y^2 = 4 and a line y = -√3x + b, if the minor arc intercepted
    by the line on the circle corresponds to a central angle of 120°, then b = ±2 -/
theorem minor_arc_intercept (b : ℝ) : 
  (∀ x y, x^2 + y^2 = 4 → y = -Real.sqrt 3 * x + b) →
  (∃ θ, θ = 2 * Real.pi / 3) →
  (b = 2 ∨ b = -2) := by
  sorry

end NUMINAMATH_CALUDE_minor_arc_intercept_l133_13360


namespace NUMINAMATH_CALUDE_infinitely_many_primes_of_form_l133_13307

theorem infinitely_many_primes_of_form (p : Nat) (hp : Nat.Prime p) (hp_odd : Odd p) :
  ∃ (S : Set Nat), (∀ n ∈ S, Nat.Prime n ∧ ∃ x, n = 2 * p * x + 1) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_of_form_l133_13307


namespace NUMINAMATH_CALUDE_sum_of_integers_l133_13336

theorem sum_of_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4 →
  m + n + p + q = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l133_13336


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l133_13361

theorem ratio_sum_theorem (a b c : ℝ) (h : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  (a + b + c) / c = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l133_13361


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_parallel_lines_implies_perpendicular_planes_l133_13306

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the parallel relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the relation of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_parallel_implies_perpendicular
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : parallel_planes α β) :
  perpendicular_lines l m :=
sorry

-- Theorem 2
theorem parallel_lines_implies_perpendicular_planes
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : parallel_lines l m) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_parallel_lines_implies_perpendicular_planes_l133_13306


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l133_13395

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x^2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

-- Theorem statement
theorem tangent_slope_at_one :
  (f' 1) = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l133_13395


namespace NUMINAMATH_CALUDE_prob_club_then_heart_l133_13312

/-- The number of cards in a standard deck --/
def standard_deck_size : ℕ := 52

/-- The number of clubs in a standard deck --/
def num_clubs : ℕ := 13

/-- The number of hearts in a standard deck --/
def num_hearts : ℕ := 13

/-- Probability of drawing a club first and then a heart from a standard 52-card deck --/
theorem prob_club_then_heart : 
  (num_clubs : ℚ) / standard_deck_size * num_hearts / (standard_deck_size - 1) = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_prob_club_then_heart_l133_13312


namespace NUMINAMATH_CALUDE_ten_factorial_divided_by_nine_factorial_l133_13302

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem ten_factorial_divided_by_nine_factorial : 
  factorial 10 / factorial 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_factorial_divided_by_nine_factorial_l133_13302


namespace NUMINAMATH_CALUDE_inequality_solution_set_l133_13398

theorem inequality_solution_set (x : ℝ) : 3 * x > 2 * x + 4 ↔ x > 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l133_13398


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l133_13308

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_4 + a_6 + a_8 = 12,
    prove that a_8 - (1/2)a_10 = 2 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 4 + a 6 + a 8 = 12) :
  a 8 - (1/2) * a 10 = 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l133_13308


namespace NUMINAMATH_CALUDE_complex_modulus_l133_13338

theorem complex_modulus (z : ℂ) (h : z * (1 - Complex.I) = 2 - 3 * Complex.I) :
  Complex.abs z = Real.sqrt 26 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l133_13338


namespace NUMINAMATH_CALUDE_max_red_points_is_13_l133_13384

/-- Represents a point on the circle -/
inductive Point
| Red : ℕ → Point  -- Red point with number of connections
| Blue : Point

/-- The configuration of points on the circle -/
structure CircleConfig where
  points : Finset Point
  red_count : ℕ
  blue_count : ℕ
  total_count : ℕ
  total_is_25 : total_count = 25
  total_is_sum : total_count = red_count + blue_count
  unique_connections : ∀ p q : Point, p ∈ points → q ∈ points → 
    p ≠ q → (∃ n m : ℕ, p = Point.Red n ∧ q = Point.Red m) → n ≠ m

/-- The maximum number of red points possible -/
def max_red_points : ℕ := 13

/-- Theorem stating that the maximum number of red points is 13 -/
theorem max_red_points_is_13 (config : CircleConfig) : 
  config.red_count ≤ max_red_points :=
sorry

end NUMINAMATH_CALUDE_max_red_points_is_13_l133_13384


namespace NUMINAMATH_CALUDE_min_n_value_l133_13309

theorem min_n_value (m : ℝ) :
  (∀ x : ℝ, |x - m| ≤ 2 → -1 ≤ x ∧ x ≤ 3) ∧
  ¬(∀ x : ℝ, |x - m| ≤ 2 → -1 ≤ x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_min_n_value_l133_13309


namespace NUMINAMATH_CALUDE_lcm_four_eight_l133_13381

theorem lcm_four_eight : ∀ n : ℕ,
  (∃ m : ℕ, 4 ∣ m ∧ 8 ∣ m ∧ n ∣ m) →
  n ≥ 8 →
  Nat.lcm 4 8 = 8 :=
by sorry

end NUMINAMATH_CALUDE_lcm_four_eight_l133_13381


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l133_13393

theorem at_least_one_non_negative (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  max (a₁*a₃ + a₂*a₄) (max (a₁*a₅ + a₂*a₆) (max (a₁*a₇ + a₂*a₈) 
    (max (a₃*a₅ + a₄*a₆) (max (a₃*a₇ + a₄*a₈) (a₅*a₇ + a₆*a₈))))) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l133_13393


namespace NUMINAMATH_CALUDE_power_division_rule_l133_13300

theorem power_division_rule (x : ℝ) : x^6 / x^3 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l133_13300


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l133_13351

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l133_13351


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l133_13341

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l133_13341


namespace NUMINAMATH_CALUDE_linda_earnings_l133_13356

/-- Calculates the total money earned from selling jeans and tees -/
def total_money_earned (jeans_price : ℕ) (tees_price : ℕ) (jeans_sold : ℕ) (tees_sold : ℕ) : ℕ :=
  jeans_price * jeans_sold + tees_price * tees_sold

/-- Proves that Linda earned $100 from selling jeans and tees -/
theorem linda_earnings : total_money_earned 11 8 4 7 = 100 := by
  sorry

end NUMINAMATH_CALUDE_linda_earnings_l133_13356


namespace NUMINAMATH_CALUDE_lcm_18_27_l133_13334

theorem lcm_18_27 : Nat.lcm 18 27 = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_27_l133_13334


namespace NUMINAMATH_CALUDE_expression_simplification_l133_13365

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 3) / x / ((x^2 - 6*x + 9) / (x^2 - 9)) - (x + 1) / x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l133_13365


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l133_13399

/-- Given a sphere with surface area 256π cm², its volume is 2048π/3 cm³. -/
theorem sphere_volume_from_surface_area :
  ∀ r : ℝ,
  (4 : ℝ) * π * r^2 = 256 * π →
  (4 : ℝ) / 3 * π * r^3 = 2048 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l133_13399


namespace NUMINAMATH_CALUDE_veg_eaters_count_l133_13315

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  only_veg : ℕ
  only_nonveg : ℕ
  both_veg_and_nonveg : ℕ

/-- Calculates the total number of people who eat veg in the family -/
def total_veg_eaters (fd : FamilyDiet) : ℕ :=
  fd.only_veg + fd.both_veg_and_nonveg

/-- Theorem: The number of people who eat veg in the given family is 26 -/
theorem veg_eaters_count (fd : FamilyDiet) 
  (h1 : fd.only_veg = 15)
  (h2 : fd.only_nonveg = 8)
  (h3 : fd.both_veg_and_nonveg = 11) : 
  total_veg_eaters fd = 26 := by
  sorry

end NUMINAMATH_CALUDE_veg_eaters_count_l133_13315


namespace NUMINAMATH_CALUDE_isosceles_triangle_arctan_sum_l133_13345

/-- In an isosceles triangle ABC where AB = AC, 
    arctan(c/(a+b)) + arctan(a/(b+c)) = π/4 -/
theorem isosceles_triangle_arctan_sum (a b c : ℝ) (α : ℝ) :
  b = c →  -- AB = AC implies b = c
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  α > 0 ∧ α < π →  -- Valid angle measure
  Real.arctan (c / (a + b)) + Real.arctan (a / (b + c)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_arctan_sum_l133_13345


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l133_13348

/-- Given a cube with surface area 294 square centimeters, prove its volume is 343 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 294 → s^3 = 343 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l133_13348


namespace NUMINAMATH_CALUDE_investment_problem_l133_13339

/-- Investment problem -/
theorem investment_problem (x y : ℝ) 
  (h1 : 0.06 * x = 0.05 * y + 160)  -- Income difference condition
  (h2 : 0.05 * y = 6000)            -- Income from 5% part
  : x + y = 222666.67 := by          -- Total investment
sorry

end NUMINAMATH_CALUDE_investment_problem_l133_13339


namespace NUMINAMATH_CALUDE_pencil_case_notebook_prices_l133_13321

theorem pencil_case_notebook_prices :
  ∀ (notebook_price pencil_case_price : ℚ),
    pencil_case_price = notebook_price + 3 →
    (200 : ℚ) / notebook_price = (350 : ℚ) / pencil_case_price →
    notebook_price = 4 ∧ pencil_case_price = 7 := by
  sorry

end NUMINAMATH_CALUDE_pencil_case_notebook_prices_l133_13321


namespace NUMINAMATH_CALUDE_symmetric_line_theorem_l133_13331

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two points are symmetric with respect to a vertical line -/
def symmetric_points (x₁ y₁ x₂ y₂ x_sym : ℝ) : Prop :=
  x₁ + x₂ = 2 * x_sym ∧ y₁ = y₂

/-- Check if a point (x, y) lies on a given line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are symmetric with respect to a vertical line -/
def symmetric_lines (l₁ l₂ : Line) (x_sym : ℝ) : Prop :=
  ∀ x₁ y₁, point_on_line x₁ y₁ l₁ →
    ∃ x₂ y₂, point_on_line x₂ y₂ l₂ ∧ symmetric_points x₁ y₁ x₂ y₂ x_sym

theorem symmetric_line_theorem :
  let l₁ : Line := ⟨2, -1, 1⟩
  let l₂ : Line := ⟨2, 1, -5⟩
  let x_sym : ℝ := 1
  symmetric_lines l₁ l₂ x_sym := by sorry

end NUMINAMATH_CALUDE_symmetric_line_theorem_l133_13331


namespace NUMINAMATH_CALUDE_select_specific_boy_and_girl_probability_l133_13313

/-- The probability of selecting both boy A and girl B when randomly choosing 1 boy and 2 girls from a group of 8 boys and 3 girls -/
theorem select_specific_boy_and_girl_probability :
  let total_boys : ℕ := 8
  let total_girls : ℕ := 3
  let boys_to_select : ℕ := 1
  let girls_to_select : ℕ := 2
  let total_events : ℕ := (total_boys.choose boys_to_select) * (total_girls.choose girls_to_select)
  let favorable_events : ℕ := 2  -- Only 2 ways to select the other girl
  (favorable_events : ℚ) / total_events = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_select_specific_boy_and_girl_probability_l133_13313


namespace NUMINAMATH_CALUDE_unique_a_for_equal_roots_l133_13329

theorem unique_a_for_equal_roots :
  ∃! a : ℝ, ∀ x : ℝ, x^2 - (a + 1) * x + a = 0 → (∃! y : ℝ, y^2 - (a + 1) * y + a = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_for_equal_roots_l133_13329


namespace NUMINAMATH_CALUDE_larger_number_theorem_l133_13363

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem larger_number_theorem (a b : ℕ) 
  (h1 : Nat.gcd a b = 37)
  (h2 : is_prime 37)
  (h3 : ∃ (k : ℕ), Nat.lcm a b = k * 37 * 17 * 23 * 29 * 31) :
  max a b = 13007833 := by
sorry

end NUMINAMATH_CALUDE_larger_number_theorem_l133_13363


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l133_13357

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  8 * a^4 + 16 * b^4 + 27 * c^4 + 1 / (6 * a * b * c) ≥ 12 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  8 * a^4 + 16 * b^4 + 27 * c^4 + 1 / (6 * a * b * c) < 12 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l133_13357


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l133_13375

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation that z satisfies
def equation (z : ℂ) : Prop := (1 + i) * z = 1 - 2 * i^3

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l133_13375


namespace NUMINAMATH_CALUDE_base6_division_theorem_l133_13324

/-- Convert a number from base 6 to base 10 -/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Convert a number from base 10 to base 6 -/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Perform division in base 6 -/
def divBase6 (a b : List Nat) : List Nat × Nat :=
  let a10 := base6ToBase10 a
  let b10 := base6ToBase10 b
  let q := a10 / b10
  let r := a10 % b10
  (base10ToBase6 q, r)

theorem base6_division_theorem :
  let a := [3, 2, 1, 2]  -- 2123 in base 6
  let b := [3, 2]        -- 23 in base 6
  let (q, r) := divBase6 a b
  q = [2, 5] ∧ r = 3 := by
  sorry

end NUMINAMATH_CALUDE_base6_division_theorem_l133_13324


namespace NUMINAMATH_CALUDE_slanted_line_angle_l133_13380

/-- The angle between a slanted line segment and a plane, given that the slanted line segment
    is twice the length of its projection on the plane. -/
theorem slanted_line_angle (L l : ℝ) (h : L = 2 * l) :
  Real.arccos (l / L) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_slanted_line_angle_l133_13380


namespace NUMINAMATH_CALUDE_find_number_l133_13317

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 7) = 99 :=
  sorry

end NUMINAMATH_CALUDE_find_number_l133_13317


namespace NUMINAMATH_CALUDE_tickets_left_l133_13325

/-- The number of tickets Tom won from 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- The number of tickets Tom won from 'skee ball' -/
def skee_ball_tickets : ℕ := 25

/-- The number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- Theorem: Given the conditions, Tom has 50 tickets left -/
theorem tickets_left : 
  whack_a_mole_tickets + skee_ball_tickets - spent_tickets = 50 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_l133_13325


namespace NUMINAMATH_CALUDE_cube_color_probability_l133_13378

-- Define the colors
inductive Color
| Black
| White
| Gray

-- Define a cube as a list of 6 colors
def Cube := List Color

-- Function to check if a cube meets the conditions
def meetsConditions (cube : Cube) : Bool :=
  sorry

-- Probability of a specific color
def colorProb : ℚ := 1 / 3

-- Total number of possible cube colorings
def totalColorings : ℕ := 729

-- Number of colorings that meet the conditions
def validColorings : ℕ := 39

theorem cube_color_probability :
  (validColorings : ℚ) / totalColorings = 13 / 243 := by
  sorry

end NUMINAMATH_CALUDE_cube_color_probability_l133_13378


namespace NUMINAMATH_CALUDE_diamond_properties_l133_13326

def diamond (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem diamond_properties :
  (∀ x y : ℝ, diamond x y = diamond y x) ∧
  (∃ x y : ℝ, 2 * (diamond x y) ≠ diamond (2*x) (2*y)) ∧
  (∀ x : ℝ, diamond x 0 = x^2) ∧
  (∀ x : ℝ, diamond x x = 0) ∧
  (∀ x y : ℝ, x = y → diamond x y = 0) :=
by sorry

end NUMINAMATH_CALUDE_diamond_properties_l133_13326


namespace NUMINAMATH_CALUDE_remainder_sum_mod15_l133_13310

theorem remainder_sum_mod15 (p q : ℤ) 
  (hp : p % 60 = 53) 
  (hq : q % 75 = 24) : 
  (p + q) % 15 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_mod15_l133_13310


namespace NUMINAMATH_CALUDE_all_dihedral_angles_equal_all_polyhedral_angles_equal_l133_13322

/-- A nearly regular polyhedron -/
structure NearlyRegularPolyhedron where
  /-- The polyhedron has a high degree of symmetry -/
  high_symmetry : Prop
  /-- Each face is a regular polygon -/
  regular_faces : Prop
  /-- Faces are arranged symmetrically around each vertex -/
  symmetric_face_arrangement : Prop
  /-- The polyhedron has vertex-transitivity property -/
  vertex_transitivity : Prop

/-- Dihedral angle of a polyhedron -/
def dihedral_angle (P : NearlyRegularPolyhedron) : Type := sorry

/-- Polyhedral angle of a polyhedron -/
def polyhedral_angle (P : NearlyRegularPolyhedron) : Type := sorry

/-- Theorem stating that all dihedral angles of a nearly regular polyhedron are equal -/
theorem all_dihedral_angles_equal (P : NearlyRegularPolyhedron) :
  ∀ a b : dihedral_angle P, a = b :=
sorry

/-- Theorem stating that all polyhedral angles of a nearly regular polyhedron are equal -/
theorem all_polyhedral_angles_equal (P : NearlyRegularPolyhedron) :
  ∀ a b : polyhedral_angle P, a = b :=
sorry

end NUMINAMATH_CALUDE_all_dihedral_angles_equal_all_polyhedral_angles_equal_l133_13322


namespace NUMINAMATH_CALUDE_batch_size_proof_l133_13335

theorem batch_size_proof :
  ∃! n : ℕ, 500 ≤ n ∧ n ≤ 600 ∧ n % 20 = 13 ∧ n % 27 = 20 ∧ n = 533 := by
  sorry

end NUMINAMATH_CALUDE_batch_size_proof_l133_13335


namespace NUMINAMATH_CALUDE_sin_20_cos_10_minus_cos_200_sin_10_l133_13305

open Real

theorem sin_20_cos_10_minus_cos_200_sin_10 :
  sin (20 * π / 180) * cos (10 * π / 180) - cos (200 * π / 180) * sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_20_cos_10_minus_cos_200_sin_10_l133_13305


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l133_13369

-- Define the surface area of the cube
def surface_area : ℝ := 864

-- Theorem stating the relationship between surface area and volume
theorem cube_volume_from_surface_area :
  ∃ (side_length : ℝ), 
    side_length > 0 ∧ 
    6 * side_length^2 = surface_area ∧ 
    side_length^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l133_13369


namespace NUMINAMATH_CALUDE_cannot_form_triangle_5_6_11_l133_13320

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem cannot_form_triangle_5_6_11 :
  ¬ can_form_triangle 5 6 11 := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_triangle_5_6_11_l133_13320


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l133_13374

theorem sqrt_product_simplification : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l133_13374


namespace NUMINAMATH_CALUDE_robin_cupcakes_sold_l133_13327

/-- Represents the number of cupcakes Robin initially made -/
def initial_cupcakes : ℕ := 42

/-- Represents the number of additional cupcakes Robin made -/
def additional_cupcakes : ℕ := 39

/-- Represents the final number of cupcakes Robin had -/
def final_cupcakes : ℕ := 59

/-- Represents the number of cupcakes Robin sold -/
def sold_cupcakes : ℕ := 22

theorem robin_cupcakes_sold :
  initial_cupcakes - sold_cupcakes + additional_cupcakes = final_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_robin_cupcakes_sold_l133_13327


namespace NUMINAMATH_CALUDE_cube_root_equation_sum_l133_13330

theorem cube_root_equation_sum (x y z : ℕ+) :
  (4 : ℝ) * ((7 : ℝ)^(1/3) - (6 : ℝ)^(1/3))^(1/2) = x.val^(1/3) + y.val^(1/3) - z.val^(1/3) →
  x.val + y.val + z.val = 51 := by
sorry

end NUMINAMATH_CALUDE_cube_root_equation_sum_l133_13330


namespace NUMINAMATH_CALUDE_circle_point_distance_sum_l133_13392

/-- Given a circle with diameter AB and radius R, and a tangent AT at point A,
    prove that a point M on the circle satisfying the condition that the sum of
    its distances to lines AB and AT is l exists if and only if l ≤ R(√2 + 1). -/
theorem circle_point_distance_sum (R l : ℝ) : 
  ∃ (M : ℝ × ℝ), 
    (M.1^2 + M.2^2 = R^2) ∧ 
    (M.1 + M.2 = l) ↔ 
    l ≤ R * (Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_point_distance_sum_l133_13392


namespace NUMINAMATH_CALUDE_school_population_l133_13385

/-- The number of students that each classroom holds -/
def students_per_classroom : ℕ := 30

/-- The number of classrooms needed -/
def number_of_classrooms : ℕ := 13

/-- The total number of students in the school -/
def total_students : ℕ := students_per_classroom * number_of_classrooms

theorem school_population : total_students = 390 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l133_13385


namespace NUMINAMATH_CALUDE_cannot_finish_fourth_l133_13319

-- Define the set of runners
inductive Runner : Type
  | A | B | C | D | E | F | G

-- Define the race result as a function from Runner to Nat (position)
def RaceResult := Runner → Nat

-- Define the conditions of the race
def ValidRaceResult (result : RaceResult) : Prop :=
  (result Runner.A < result Runner.B) ∧
  (result Runner.A < result Runner.C) ∧
  (result Runner.B < result Runner.D) ∧
  (result Runner.C < result Runner.E) ∧
  (result Runner.A < result Runner.F) ∧ (result Runner.F < result Runner.B) ∧
  (result Runner.B < result Runner.G) ∧ (result Runner.G < result Runner.C)

-- Theorem to prove
theorem cannot_finish_fourth (result : RaceResult) 
  (h : ValidRaceResult result) : 
  result Runner.A ≠ 4 ∧ result Runner.F ≠ 4 ∧ result Runner.G ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_cannot_finish_fourth_l133_13319


namespace NUMINAMATH_CALUDE_exists_right_triangle_with_perpendicular_medians_l133_13377

/-- A right-angled triangle with one given leg and perpendicular medians to the other two sides -/
structure RightTriangleWithPerpendicularMedians where
  /-- The length of the given leg -/
  a : ℝ
  /-- The length of the second leg -/
  b : ℝ
  /-- The length of the hypotenuse -/
  c : ℝ
  /-- The given leg is positive -/
  a_pos : 0 < a
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagoras : a^2 + b^2 = c^2
  /-- The medians to the other two sides are perpendicular -/
  medians_perpendicular : (2*c^2 + 2*b^2 - a^2) * (2*c^2 + 2*a^2 - b^2) = 9*a^2*b^2

/-- There exists a right-angled triangle with one given leg and perpendicular medians to the other two sides -/
theorem exists_right_triangle_with_perpendicular_medians (a : ℝ) (ha : 0 < a) : 
  ∃ t : RightTriangleWithPerpendicularMedians, t.a = a :=
sorry

end NUMINAMATH_CALUDE_exists_right_triangle_with_perpendicular_medians_l133_13377


namespace NUMINAMATH_CALUDE_unique_solution_system_l133_13318

theorem unique_solution_system (x y : ℝ) : 
  (x^3 + y^3 + 3*x*y = 1 ∧ x^2 - y^2 = 1) →
  ((x ≥ 0 ∧ y ≥ 0) ∨ (x + y > 0)) →
  x = 1 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l133_13318


namespace NUMINAMATH_CALUDE_large_cube_edge_is_one_meter_l133_13386

/-- The edge length of a cubical box that can contain a given number of smaller cubes -/
def large_cube_edge_length (small_cube_edge : ℝ) (num_small_cubes : ℝ) : ℝ :=
  (small_cube_edge^3 * num_small_cubes)^(1/3)

/-- Theorem: The edge length of a cubical box that can contain 999.9999999999998 cubes 
    with 10 cm edge length is 1 meter -/
theorem large_cube_edge_is_one_meter :
  large_cube_edge_length 0.1 999.9999999999998 = 1 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_edge_is_one_meter_l133_13386


namespace NUMINAMATH_CALUDE_no_geometric_sequence_satisfies_conditions_l133_13354

theorem no_geometric_sequence_satisfies_conditions :
  ¬ ∃ (a : ℕ → ℝ) (q : ℝ),
    (∀ n : ℕ, a (n + 1) = q * a n) ∧  -- geometric sequence
    (a 1 + a 6 = 11) ∧  -- condition 1
    (a 3 * a 4 = 32 / 9) ∧  -- condition 1
    (∀ n : ℕ, a (n + 1) > a n) ∧  -- condition 2
    (∃ m : ℕ, m > 4 ∧ 
      2 * (a m)^2 = 2/3 * a (m - 1) + (a (m + 1) + 4/9)) :=  -- condition 3
by sorry

end NUMINAMATH_CALUDE_no_geometric_sequence_satisfies_conditions_l133_13354


namespace NUMINAMATH_CALUDE_blue_highlighters_count_l133_13396

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 9

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 8

/-- The total number of highlighters in the teacher's desk -/
def total_highlighters : ℕ := 22

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := total_highlighters - (pink_highlighters + yellow_highlighters)

theorem blue_highlighters_count : blue_highlighters = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_highlighters_count_l133_13396


namespace NUMINAMATH_CALUDE_find_number_to_add_l133_13390

theorem find_number_to_add : ∃ x : ℝ, (5 * 12) / (180 / 3) + x = 71 := by
  sorry

end NUMINAMATH_CALUDE_find_number_to_add_l133_13390


namespace NUMINAMATH_CALUDE_range_of_expression_l133_13362

theorem range_of_expression (α β : ℝ) 
  (h_α : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h_β : β ∈ Set.Icc 0 (Real.pi / 2)) :
  ∃ (x : ℝ), x ∈ Set.Ioo (-Real.pi / 6) Real.pi ∧
  ∃ (α' β' : ℝ), α' ∈ Set.Ioo 0 (Real.pi / 2) ∧
                 β' ∈ Set.Icc 0 (Real.pi / 2) ∧
                 x = 2 * α' - β' / 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l133_13362


namespace NUMINAMATH_CALUDE_no_opposite_midpoints_l133_13344

/-- Represents a rectangular billiard table -/
structure BilliardTable where
  length : ℝ
  width : ℝ
  corner_pockets : Bool

/-- Represents the trajectory of a ball on the billiard table -/
structure BallTrajectory where
  table : BilliardTable
  start_corner : Fin 4
  angle : ℝ

/-- Predicate to check if a point is on the midpoint of a side -/
def is_side_midpoint (table : BilliardTable) (x y : ℝ) : Prop :=
  (x = 0 ∧ y = table.width / 2) ∨
  (x = table.length ∧ y = table.width / 2) ∨
  (y = 0 ∧ x = table.length / 2) ∨
  (y = table.width ∧ x = table.length / 2)

/-- Theorem stating that a ball cannot visit midpoints of opposite sides -/
theorem no_opposite_midpoints (trajectory : BallTrajectory) 
  (h1 : trajectory.angle = π/4)
  (h2 : ∃ (x1 y1 : ℝ), is_side_midpoint trajectory.table x1 y1) :
  ¬ ∃ (x2 y2 : ℝ), 
    is_side_midpoint trajectory.table x2 y2 ∧ 
    ((x1 = 0 ∧ x2 = trajectory.table.length) ∨ 
     (x1 = trajectory.table.length ∧ x2 = 0) ∨
     (y1 = 0 ∧ y2 = trajectory.table.width) ∨
     (y1 = trajectory.table.width ∧ y2 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_opposite_midpoints_l133_13344


namespace NUMINAMATH_CALUDE_ruler_cost_l133_13323

theorem ruler_cost (total_students : ℕ) (buyers : ℕ) (rulers_per_student : ℕ) (ruler_cost : ℕ) :
  total_students = 36 →
  buyers > total_students / 2 →
  rulers_per_student > 1 →
  ruler_cost > rulers_per_student →
  buyers * rulers_per_student * ruler_cost = 1729 →
  ruler_cost = 13 :=
by sorry

end NUMINAMATH_CALUDE_ruler_cost_l133_13323


namespace NUMINAMATH_CALUDE_mike_payment_l133_13389

def medical_costs (x_ray : ℝ) (blood_tests : ℝ) (deductible : ℝ) : ℝ :=
  let mri := 3 * x_ray
  let ct_scan := 2 * mri
  let ultrasound := 0.5 * mri
  let total_cost := x_ray + mri + ct_scan + blood_tests + ultrasound
  let remaining_amount := total_cost - deductible
  let insurance_coverage := 0.8 * x_ray + 0.8 * mri + 0.7 * ct_scan + 0.5 * blood_tests + 0.6 * ultrasound
  remaining_amount - insurance_coverage

theorem mike_payment (x_ray : ℝ) (blood_tests : ℝ) (deductible : ℝ) 
  (h1 : x_ray = 250)
  (h2 : blood_tests = 200)
  (h3 : deductible = 500) :
  medical_costs x_ray blood_tests deductible = 400 := by
  sorry

end NUMINAMATH_CALUDE_mike_payment_l133_13389


namespace NUMINAMATH_CALUDE_eleven_students_in_line_l133_13340

/-- The number of students in a line, given Yoonjung's position -/
def total_students (students_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  students_in_front + 1 + (position_from_back - 1)

/-- Theorem: There are 11 students in the line -/
theorem eleven_students_in_line : 
  total_students 6 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_students_in_line_l133_13340


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l133_13366

open Real

theorem triangle_abc_properties (a b c A B C : ℝ) (k : ℤ) :
  -- Conditions
  (2 * Real.sqrt 3 * a * Real.sin C * Real.sin B = a * Real.sin A + b * Real.sin B - c * Real.sin C) →
  (a * Real.cos (π / 2 - B) = b * Real.cos (2 * ↑k * π + A)) →
  (a = 2) →
  -- Conclusions
  (C = π / 6) ∧
  (1 / 2 * a * c * Real.sin B = (1 + Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l133_13366


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_sum_l133_13352

/-- A quadrilateral with vertices at (1,2), (4,6), (5,4), and (2,0) has a perimeter that can be
    expressed as a√2 + b√5 + c√10 where a, b, and c are integers, and their sum is 2. -/
theorem quadrilateral_perimeter_sum (a b c : ℤ) : 
  let v1 : ℝ × ℝ := (1, 2)
  let v2 : ℝ × ℝ := (4, 6)
  let v3 : ℝ × ℝ := (5, 4)
  let v4 : ℝ × ℝ := (2, 0)
  let perimeter := dist v1 v2 + dist v2 v3 + dist v3 v4 + dist v4 v1
  perimeter = a * Real.sqrt 2 + b * Real.sqrt 5 + c * Real.sqrt 10 →
  a + b + c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_sum_l133_13352


namespace NUMINAMATH_CALUDE_alcohol_percentage_x_is_correct_l133_13364

/-- The percentage of alcohol by volume in solution x -/
def alcohol_percentage_x : ℝ := 0.10

/-- The percentage of alcohol by volume in solution y -/
def alcohol_percentage_y : ℝ := 0.30

/-- The volume of solution y in milliliters -/
def volume_y : ℝ := 600

/-- The volume of solution x in milliliters -/
def volume_x : ℝ := 200

/-- The percentage of alcohol by volume in the final mixture -/
def alcohol_percentage_final : ℝ := 0.25

theorem alcohol_percentage_x_is_correct :
  alcohol_percentage_x * volume_x + alcohol_percentage_y * volume_y =
  alcohol_percentage_final * (volume_x + volume_y) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_x_is_correct_l133_13364


namespace NUMINAMATH_CALUDE_carla_restock_theorem_l133_13346

/-- Represents the food bank inventory and distribution problem -/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day2_people : ℕ
  day2_cans_per_person : ℕ
  final_restock : ℕ
  total_given_away : ℕ

/-- Calculates the number of cans restocked after the first day -/
def cans_restocked_after_day1 (fb : FoodBank) : ℕ :=
  fb.total_given_away - (fb.initial_stock - fb.day1_people * fb.day1_cans_per_person) +
  (fb.final_restock - fb.day2_people * fb.day2_cans_per_person)

/-- Theorem stating that Carla restocked 2000 cans after the first day -/
theorem carla_restock_theorem (fb : FoodBank)
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day2_people = 1000)
  (h5 : fb.day2_cans_per_person = 2)
  (h6 : fb.final_restock = 3000)
  (h7 : fb.total_given_away = 2500) :
  cans_restocked_after_day1 fb = 2000 := by
  sorry

end NUMINAMATH_CALUDE_carla_restock_theorem_l133_13346


namespace NUMINAMATH_CALUDE_tom_distance_covered_l133_13337

theorem tom_distance_covered (swim_time : ℝ) (swim_speed : ℝ) : 
  swim_time = 2 →
  swim_speed = 2 →
  let run_time := swim_time / 2
  let run_speed := 4 * swim_speed
  let swim_distance := swim_time * swim_speed
  let run_distance := run_time * run_speed
  swim_distance + run_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_tom_distance_covered_l133_13337


namespace NUMINAMATH_CALUDE_regular_tetrahedron_inradius_l133_13391

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  /-- The altitude of the regular tetrahedron -/
  altitude : ℝ
  /-- The inradius of the regular tetrahedron -/
  inradius : ℝ

/-- The inradius of a regular tetrahedron is one fourth of its altitude -/
theorem regular_tetrahedron_inradius (t : RegularTetrahedron) :
  t.inradius = (1 / 4) * t.altitude := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_inradius_l133_13391


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l133_13383

def base_8_to_decimal (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∀ n : Nat,
  n < 1000 →
  n > 0 →
  base_8_to_decimal n % 7 = 0 →
  n ≤ 777 :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l133_13383


namespace NUMINAMATH_CALUDE_consecutive_integers_product_210_l133_13349

theorem consecutive_integers_product_210 (n : ℤ) :
  n * (n + 1) * (n + 2) = 210 → n + (n + 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_210_l133_13349


namespace NUMINAMATH_CALUDE_tony_running_distance_tony_running_distance_proof_l133_13301

/-- Proves that Tony runs 10 miles without the backpack each morning given his exercise routine. -/
theorem tony_running_distance : ℝ → Prop :=
  fun x =>
    let walk_distance : ℝ := 3
    let walk_speed : ℝ := 3
    let run_speed : ℝ := 5
    let total_exercise_time : ℝ := 21
    let days_per_week : ℝ := 7
    
    let daily_walk_time : ℝ := walk_distance / walk_speed
    let daily_run_time : ℝ := x / run_speed
    let weekly_exercise_time : ℝ := days_per_week * (daily_walk_time + daily_run_time)
    
    weekly_exercise_time = total_exercise_time → x = 10

/-- The proof of the theorem. -/
theorem tony_running_distance_proof : tony_running_distance 10 := by
  sorry

end NUMINAMATH_CALUDE_tony_running_distance_tony_running_distance_proof_l133_13301


namespace NUMINAMATH_CALUDE_sum_of_radii_tangent_circles_l133_13314

/-- The sum of all possible radii of a circle tangent to both axes and externally tangent to another circle -/
theorem sum_of_radii_tangent_circles : ∃ (r₁ r₂ : ℝ),
  let c₁ : ℝ × ℝ := (r₁, r₁)  -- Center of the first circle
  let c₂ : ℝ × ℝ := (5, 0)    -- Center of the second circle
  let r₃ : ℝ := 3             -- Radius of the second circle
  (0 < r₁ ∧ 0 < r₂) ∧         -- Radii are positive
  (c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2 = (r₁ + r₃)^2 ∧  -- Circles are externally tangent
  r₁ + r₂ = 16 :=             -- Sum of radii is 16
by sorry

end NUMINAMATH_CALUDE_sum_of_radii_tangent_circles_l133_13314


namespace NUMINAMATH_CALUDE_solution_x_proportion_l133_13311

/-- Represents a solution with a certain percentage of material a -/
structure Solution :=
  (a_percent : ℝ)

/-- Represents the mixture of solutions -/
structure Mixture :=
  (x y z : ℝ)
  (x_sol y_sol z_sol : Solution)

/-- The conditions of the problem -/
def problem_conditions (m : Mixture) : Prop :=
  m.x_sol.a_percent = 0.2 ∧
  m.y_sol.a_percent = 0.3 ∧
  m.z_sol.a_percent = 0.4 ∧
  m.x_sol.a_percent * m.x + m.y_sol.a_percent * m.y + m.z_sol.a_percent * m.z = 0.25 * (m.x + m.y + m.z) ∧
  m.y = 1.5 * m.z ∧
  m.x > 0 ∧ m.y > 0 ∧ m.z > 0

/-- The theorem to be proved -/
theorem solution_x_proportion (m : Mixture) : 
  problem_conditions m → m.x / (m.x + m.y + m.z) = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_proportion_l133_13311


namespace NUMINAMATH_CALUDE_inequality_theorem_l133_13367

theorem inequality_theorem (x y a : ℝ) (h1 : x < y) (h2 : a < 1) : x + a < y + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l133_13367


namespace NUMINAMATH_CALUDE_boat_travel_l133_13328

theorem boat_travel (boat_speed : ℝ) (time_against : ℝ) (time_with : ℝ)
  (h1 : boat_speed = 12)
  (h2 : time_against = 10)
  (h3 : time_with = 6) :
  ∃ (current_speed : ℝ) (distance : ℝ),
    current_speed = 3 ∧
    distance = 90 ∧
    (boat_speed - current_speed) * time_against = (boat_speed + current_speed) * time_with :=
by sorry

end NUMINAMATH_CALUDE_boat_travel_l133_13328


namespace NUMINAMATH_CALUDE_tree_spacing_l133_13358

theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 800 →
  num_trees = 26 →
  num_trees ≥ 2 →
  (yard_length / (num_trees - 1 : ℝ)) = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l133_13358


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l133_13376

theorem quadratic_equation_integer_roots (a : ℝ) : 
  (a > 0 ∧ ∃ x y : ℤ, x ≠ y ∧ 
    a^2 * (x : ℝ)^2 + a * (x : ℝ) + 1 - 7 * a^2 = 0 ∧
    a^2 * (y : ℝ)^2 + a * (y : ℝ) + 1 - 7 * a^2 = 0) ↔ 
  (a = 1 ∨ a = 1/2 ∨ a = 1/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l133_13376


namespace NUMINAMATH_CALUDE_solution_range_l133_13372

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1/4)^x + (1/2)^(x-1) + a = 0) → 
  -3 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l133_13372


namespace NUMINAMATH_CALUDE_volleyball_tournament_wins_l133_13343

theorem volleyball_tournament_wins (n : ℕ) (h_n : n = 73) :
  ∀ (p m : ℕ) (x : ℕ) (h_x : 0 < x ∧ x < n),
  x * p + (n - x) * m = n * (n - 1) / 2 →
  p = m :=
by sorry

end NUMINAMATH_CALUDE_volleyball_tournament_wins_l133_13343


namespace NUMINAMATH_CALUDE_quadratic_equation_with_roots_as_coefficients_l133_13333

/-- A quadratic equation with coefficients a, b, and c, represented as ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a quadratic equation -/
structure Roots where
  x₁ : ℝ
  x₂ : ℝ

/-- Checks if the given roots satisfy the quadratic equation -/
def satisfiesEquation (eq : QuadraticEquation) (roots : Roots) : Prop :=
  eq.a * roots.x₁^2 + eq.b * roots.x₁ + eq.c = 0 ∧
  eq.a * roots.x₂^2 + eq.b * roots.x₂ + eq.c = 0

/-- The theorem stating that given a quadratic equation with its roots as coefficients,
    only two specific equations are valid -/
theorem quadratic_equation_with_roots_as_coefficients
  (eq : QuadraticEquation)
  (roots : Roots)
  (h : satisfiesEquation eq roots)
  (h_coeff : eq.a = 1 ∧ eq.b = roots.x₁ ∧ eq.c = roots.x₂) :
  (eq.a = 1 ∧ eq.b = 0 ∧ eq.c = 0) ∨
  (eq.a = 1 ∧ eq.b = 1 ∧ eq.c = -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_roots_as_coefficients_l133_13333


namespace NUMINAMATH_CALUDE_pedestrian_cyclist_speeds_l133_13373

theorem pedestrian_cyclist_speeds
  (distance : ℝ)
  (pedestrian_start : ℝ)
  (cyclist1_start : ℝ)
  (cyclist2_start : ℝ)
  (pedestrian_speed : ℝ)
  (cyclist_speed : ℝ)
  (h1 : distance = 40)
  (h2 : cyclist1_start - pedestrian_start = 10/3)
  (h3 : cyclist2_start - pedestrian_start = 4.5)
  (h4 : pedestrian_speed * ((cyclist1_start - pedestrian_start) + (distance/2 - pedestrian_speed * (cyclist1_start - pedestrian_start)) / (cyclist_speed - pedestrian_speed)) = distance/2)
  (h5 : pedestrian_speed * ((cyclist1_start - pedestrian_start) + (distance/2 - pedestrian_speed * (cyclist1_start - pedestrian_start)) / (cyclist_speed - pedestrian_speed) + 1) + cyclist_speed * ((cyclist2_start - pedestrian_start) - ((cyclist1_start - pedestrian_start) + (distance/2 - pedestrian_speed * (cyclist1_start - pedestrian_start)) / (cyclist_speed - pedestrian_speed) + 1)) = distance)
  : pedestrian_speed = 5 ∧ cyclist_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_pedestrian_cyclist_speeds_l133_13373


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l133_13353

theorem triangle_sine_inequality (A B C : Real) :
  A + B + C = 180 →
  0 < A ∧ A ≤ 180 →
  0 < B ∧ B ≤ 180 →
  0 < C ∧ C ≤ 180 →
  Real.sin ((A - 30) * π / 180) + Real.sin ((B - 30) * π / 180) + Real.sin ((C - 30) * π / 180) ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l133_13353


namespace NUMINAMATH_CALUDE_intersection_above_axis_implies_no_roots_l133_13359

/-- 
Given that the graphs of y = ax², y = bx, and y = c intersect at a point above the x-axis,
prove that the equation ax² + bx + c = 0 has no real roots.
-/
theorem intersection_above_axis_implies_no_roots 
  (a b c : ℝ) 
  (ha : a > 0)
  (hc : c > 0)
  (h_intersect : ∃ (m : ℝ), a * m^2 = b * m ∧ b * m = c) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_above_axis_implies_no_roots_l133_13359


namespace NUMINAMATH_CALUDE_largest_minus_smallest_l133_13355

def problem (A B C : ℤ) : Prop :=
  A = 10 * 2 + 9 ∧
  A = B + 16 ∧
  C = B * 3

theorem largest_minus_smallest (A B C : ℤ) 
  (h : problem A B C) : 
  max A (max B C) - min A (min B C) = 26 := by
  sorry

end NUMINAMATH_CALUDE_largest_minus_smallest_l133_13355


namespace NUMINAMATH_CALUDE_isosceles_triangle_l133_13303

/-- A triangle is isosceles if it has at least two equal sides -/
def IsIsosceles (A B C : ℝ × ℝ) : Prop :=
  (dist A B = dist B C) ∨ (dist B C = dist A C) ∨ (dist A C = dist A B)

/-- The perimeter of a triangle is the sum of the lengths of its sides -/
def Perimeter (A B C : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist A C

theorem isosceles_triangle (A B C M N : ℝ × ℝ) :
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = (1 - t) • A + t • B) →  -- M is on AB
  (∃ s : ℝ, 0 < s ∧ s < 1 ∧ N = (1 - s) • B + s • C) →  -- N is on BC
  Perimeter A M C = Perimeter C N A →                  -- Perimeter condition 1
  Perimeter A N B = Perimeter C M B →                  -- Perimeter condition 2
  IsIsosceles A B C :=                                 -- Conclusion
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l133_13303


namespace NUMINAMATH_CALUDE_stage_8_area_l133_13371

/-- The area of the rectangle at a given stage in the square-adding process -/
def rectangleArea (stage : ℕ) : ℕ :=
  stage * (4 * 4)

/-- Theorem: The area of the rectangle at Stage 8 is 128 square inches -/
theorem stage_8_area : rectangleArea 8 = 128 := by
  sorry

end NUMINAMATH_CALUDE_stage_8_area_l133_13371


namespace NUMINAMATH_CALUDE_dot_path_length_on_rolled_cube_l133_13368

/-- The path length of a dot on a cube when rolled twice --/
theorem dot_path_length_on_rolled_cube : 
  let cube_edge_length : ℝ := 2
  let dot_distance_from_center : ℝ := cube_edge_length / 4
  let roll_count : ℕ := 2
  let radius : ℝ := Real.sqrt (1^2 + dot_distance_from_center^2)
  let path_length : ℝ := roll_count * 2 * π * radius
  path_length = 2.236 * π := by sorry

end NUMINAMATH_CALUDE_dot_path_length_on_rolled_cube_l133_13368


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_external_tangents_l133_13347

/-- Given two externally tangent circles, this theorem proves the radius of the circle
    tangent to their common external tangents and the line segment connecting the
    external points of tangency on the larger circle. -/
theorem inscribed_circle_radius_external_tangents
  (R : ℝ) (r : ℝ) (h_R : R = 4) (h_r : r = 3) (h_touch : R > r) :
  let d := R + r  -- Distance between circle centers
  let inscribed_radius := (R * r) / d
  inscribed_radius = 12 / 7 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_external_tangents_l133_13347


namespace NUMINAMATH_CALUDE_no_integer_solution_a_squared_minus_3b_squared_equals_8_l133_13342

theorem no_integer_solution_a_squared_minus_3b_squared_equals_8 :
  ¬ ∃ (a b : ℤ), a^2 - 3*b^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_a_squared_minus_3b_squared_equals_8_l133_13342


namespace NUMINAMATH_CALUDE_smallest_n_for_negative_sum_l133_13316

-- Define the arithmetic sequence and its sum
def a (n : ℕ) : ℤ := 7 - 2 * (n - 1)
def S (n : ℕ) : ℤ := n * (2 * 7 + (n - 1) * (-2)) / 2

-- State the theorem
theorem smallest_n_for_negative_sum :
  (∀ k < 9, S k ≥ 0) ∧ (S 9 < 0) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_negative_sum_l133_13316
