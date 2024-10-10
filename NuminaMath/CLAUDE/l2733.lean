import Mathlib

namespace annulus_area_l2733_273354

/-- The area of an annulus with specific properties -/
theorem annulus_area (r s t : ℝ) (h1 : r > s) (h2 : t = 2 * s) (h3 : r^2 = s^2 + (t/2)^2) :
  π * (r^2 - s^2) = π * s^2 := by
  sorry

end annulus_area_l2733_273354


namespace wipes_count_l2733_273334

/-- The number of wipes initially in the container -/
def initial_wipes : ℕ := 70

/-- The number of wipes used during the day -/
def wipes_used : ℕ := 20

/-- The number of wipes added after using some -/
def wipes_added : ℕ := 10

/-- The number of wipes left at night -/
def wipes_at_night : ℕ := 60

theorem wipes_count : initial_wipes - wipes_used + wipes_added = wipes_at_night := by
  sorry

end wipes_count_l2733_273334


namespace algae_free_day_l2733_273369

/-- The number of days it takes for the pond to be completely covered in algae -/
def total_days : ℕ := 20

/-- The fraction of the pond covered by algae on a given day -/
def algae_coverage (day : ℕ) : ℚ :=
  if day ≥ total_days then 1
  else (1 / 2) ^ (total_days - day)

/-- The day on which the pond is 87.5% algae-free -/
def target_day : ℕ :=
  total_days - 3

theorem algae_free_day :
  algae_coverage target_day = 1 - (7 / 8) :=
sorry

end algae_free_day_l2733_273369


namespace field_area_l2733_273312

/-- A rectangular field with specific properties -/
structure RectangularField where
  breadth : ℝ
  length : ℝ
  length_relation : length = breadth + 30
  perimeter : ℝ
  perimeter_formula : perimeter = 2 * (length + breadth)
  perimeter_value : perimeter = 540

/-- The area of the rectangular field is 18000 square metres -/
theorem field_area (field : RectangularField) : field.length * field.breadth = 18000 := by
  sorry

end field_area_l2733_273312


namespace deepthi_material_usage_l2733_273380

theorem deepthi_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 4 / 17)
  (h2 : material2 = 3 / 10)
  (h3 : leftover = 9 / 30) :
  material1 + material2 - leftover = 4 / 17 := by
sorry

end deepthi_material_usage_l2733_273380


namespace complex_equation_solution_l2733_273326

theorem complex_equation_solution (z : ℂ) :
  (Complex.I - 1) * z = 2 → z = -1 - Complex.I := by
  sorry

end complex_equation_solution_l2733_273326


namespace divisor_problem_l2733_273322

theorem divisor_problem (D : ℚ) : D ≠ 0 → (72 / D + 5 = 17) → D = 6 := by sorry

end divisor_problem_l2733_273322


namespace min_value_expression_l2733_273350

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  2 * x^2 + 8 * x * y + 6 * y^2 + 16 * y * z + 3 * z^2 ≥ 24 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧
    2 * a^2 + 8 * a * b + 6 * b^2 + 16 * b * c + 3 * c^2 = 24 :=
by sorry

end min_value_expression_l2733_273350


namespace correct_assignment_count_l2733_273375

/-- Represents an assignment statement --/
inductive AssignmentStatement
  | Constant : ℕ → String → AssignmentStatement
  | Variable : String → String → AssignmentStatement
  | Expression : String → String → AssignmentStatement
  | SelfAssignment : String → AssignmentStatement

/-- Checks if an assignment statement is valid --/
def isValidAssignment (stmt : AssignmentStatement) : Bool :=
  match stmt with
  | AssignmentStatement.Constant _ _ => false
  | AssignmentStatement.Variable _ _ => true
  | AssignmentStatement.Expression _ _ => false
  | AssignmentStatement.SelfAssignment _ => true

/-- The list of given assignment statements --/
def givenStatements : List AssignmentStatement :=
  [AssignmentStatement.Constant 2 "A",
   AssignmentStatement.Expression "x_+_y" "2",
   AssignmentStatement.Expression "A_-_B" "-2",
   AssignmentStatement.SelfAssignment "A"]

/-- Counts the number of valid assignment statements in a list --/
def countValidAssignments (stmts : List AssignmentStatement) : ℕ :=
  (stmts.filter isValidAssignment).length

theorem correct_assignment_count :
  countValidAssignments givenStatements = 1 := by sorry

end correct_assignment_count_l2733_273375


namespace ten_thousand_squared_l2733_273344

theorem ten_thousand_squared (x : ℕ) (h : x = 10^4) : x * x = 10^8 := by
  sorry

end ten_thousand_squared_l2733_273344


namespace circle_symmetry_axis_l2733_273308

/-- Given a circle and a line that is its axis of symmetry, prove that the parameter a in the line equation equals 1 -/
theorem circle_symmetry_axis (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 2*y - 3 = 0 → 
    (∃ c : ℝ, ∀ x' y' : ℝ, (x' - 2*a*y' - 3 = 0 ∧ 
      x'^2 + y'^2 - 2*x' + 2*y' - 3 = 0) ↔ 
      (2*c - x' - 2*a*y' - 3 = 0 ∧ 
       (2*c - x')^2 + y'^2 - 2*(2*c - x') + 2*y' - 3 = 0))) → 
  a = 1 := by
sorry

end circle_symmetry_axis_l2733_273308


namespace min_value_smallest_at_a_l2733_273338

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

/-- Theorem stating that the minimum value of f(x) is smallest when a = 82/43 -/
theorem min_value_smallest_at_a (a : ℝ) :
  (∀ x : ℝ, f (82/43) x ≤ f a x) :=
sorry

end min_value_smallest_at_a_l2733_273338


namespace digit_sum_10_2017_position_l2733_273393

/-- A sequence of positive integers whose digits sum to 10, arranged in ascending order -/
def digit_sum_10_sequence : ℕ → ℕ := sorry

/-- Predicate to check if a natural number's digits sum to 10 -/
def digits_sum_to_10 (n : ℕ) : Prop := sorry

/-- The sequence digit_sum_10_sequence contains all and only the numbers whose digits sum to 10 -/
axiom digit_sum_10_sequence_property :
  ∀ n : ℕ, digits_sum_to_10 (digit_sum_10_sequence n) ∧
  (∀ m : ℕ, digits_sum_to_10 m → ∃ k : ℕ, digit_sum_10_sequence k = m)

/-- The sequence digit_sum_10_sequence is strictly increasing -/
axiom digit_sum_10_sequence_increasing :
  ∀ n m : ℕ, n < m → digit_sum_10_sequence n < digit_sum_10_sequence m

theorem digit_sum_10_2017_position :
  ∃ n : ℕ, digit_sum_10_sequence n = 2017 ∧ n = 110 := by sorry

end digit_sum_10_2017_position_l2733_273393


namespace towel_area_decrease_l2733_273382

theorem towel_area_decrease :
  ∀ (L B : ℝ), L > 0 → B > 0 →
  let new_length := 0.7 * L
  let new_breadth := 0.75 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.475 :=
by sorry

end towel_area_decrease_l2733_273382


namespace als_investment_l2733_273301

theorem als_investment (al betty clare : ℝ) 
  (total_initial : al + betty + clare = 1200)
  (total_final : (al - 200) + (2 * betty) + (1.5 * clare) = 1800)
  : al = 600 := by
  sorry

end als_investment_l2733_273301


namespace power_of_three_mod_eight_l2733_273387

theorem power_of_three_mod_eight : 3^1234 % 8 = 1 := by
  sorry

end power_of_three_mod_eight_l2733_273387


namespace henrikh_walk_time_per_block_l2733_273373

/-- The time it takes Henrikh to walk one block to work -/
def walkTimePerBlock : ℝ := 60

/-- The number of blocks from Henrikh's home to his office -/
def distanceInBlocks : ℕ := 12

/-- The time it takes Henrikh to ride his bicycle for one block -/
def bikeTimePerBlock : ℝ := 20

/-- The additional time it takes to walk compared to riding a bicycle for the entire distance -/
def additionalWalkTime : ℝ := 8 * 60  -- 8 minutes in seconds

theorem henrikh_walk_time_per_block :
  walkTimePerBlock * distanceInBlocks = 
  bikeTimePerBlock * distanceInBlocks + additionalWalkTime :=
by sorry

end henrikh_walk_time_per_block_l2733_273373


namespace gertrude_has_ten_fleas_l2733_273359

/-- The number of fleas on Gertrude's chicken -/
def gertrude_fleas : ℕ := sorry

/-- The number of fleas on Maud's chicken -/
def maud_fleas : ℕ := sorry

/-- The number of fleas on Olive's chicken -/
def olive_fleas : ℕ := sorry

/-- Maud has 5 times the amount of fleas as Olive -/
axiom maud_olive_relation : maud_fleas = 5 * olive_fleas

/-- Olive has half the amount of fleas as Gertrude -/
axiom olive_gertrude_relation : olive_fleas * 2 = gertrude_fleas

/-- The total number of fleas is 40 -/
axiom total_fleas : gertrude_fleas + maud_fleas + olive_fleas = 40

/-- Theorem: Gertrude has 10 fleas -/
theorem gertrude_has_ten_fleas : gertrude_fleas = 10 := by sorry

end gertrude_has_ten_fleas_l2733_273359


namespace linear_equation_solution_l2733_273337

theorem linear_equation_solution (a : ℝ) : 
  (∃ x y : ℝ, x = 2 ∧ y = 3 ∧ a * x - 3 * y = 3) → a = 6 := by
  sorry

end linear_equation_solution_l2733_273337


namespace combined_value_l2733_273304

def sum_even (a b : ℕ) : ℕ := 
  (b - a + 2) / 2 * (a + b) / 2

def sum_odd (a b : ℕ) : ℕ := 
  ((b - a) / 2 + 1) * (a + b) / 2

def i : ℕ := sum_even 2 500
def k : ℕ := sum_even 8 200
def j : ℕ := sum_odd 5 133

theorem combined_value : 2 * i - k + 3 * j = 128867 := by sorry

end combined_value_l2733_273304


namespace basketball_lineup_combinations_l2733_273305

theorem basketball_lineup_combinations (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 6) :
  (n.factorial / (n - k).factorial) = 360360 := by
  sorry

end basketball_lineup_combinations_l2733_273305


namespace extremum_at_one_l2733_273362

def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

theorem extremum_at_one (a b : ℝ) :
  f a b 1 = 10 ∧ (deriv (f a b)) 1 = 0 → a = -4 ∧ b = 11 :=
by sorry

end extremum_at_one_l2733_273362


namespace sum_of_angles_convex_polygon_l2733_273349

/-- The sum of interior angles of a convex polygon with n sides, where n ≥ 3 -/
def sumOfAngles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- Theorem: For any convex polygon with n sides, where n ≥ 3,
    the sum of its interior angles is equal to (n-2) * 180° -/
theorem sum_of_angles_convex_polygon (n : ℕ) (h : n ≥ 3) :
  sumOfAngles n = (n - 2) * 180 := by
  sorry

end sum_of_angles_convex_polygon_l2733_273349


namespace base3_10212_equals_104_l2733_273367

/-- Converts a base 3 number to base 10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- Theorem: The base 10 representation of 10212 in base 3 is 104 --/
theorem base3_10212_equals_104 : base3ToBase10 [1, 0, 2, 1, 2] = 104 := by
  sorry

end base3_10212_equals_104_l2733_273367


namespace olivia_friday_hours_l2733_273361

/-- Calculates the number of hours Olivia worked on Friday given her hourly rate, work hours on Monday and Wednesday, and total earnings for the week. -/
def fridayHours (hourlyRate : ℚ) (mondayHours wednesdayHours : ℚ) (totalEarnings : ℚ) : ℚ :=
  (totalEarnings - hourlyRate * (mondayHours + wednesdayHours)) / hourlyRate

/-- Proves that Olivia worked 6 hours on Friday given the specified conditions. -/
theorem olivia_friday_hours :
  fridayHours 9 4 3 117 = 6 := by
  sorry

end olivia_friday_hours_l2733_273361


namespace equation_solution_l2733_273313

theorem equation_solution :
  ∀ x y : ℝ, y = 3 * x →
  (5 * y^2 + 3 * y + 2 = 3 * (8 * x^2 + y + 1)) ↔ 
  (x = 1 / Real.sqrt 21 ∨ x = -(1 / Real.sqrt 21)) :=
by sorry

end equation_solution_l2733_273313


namespace polynomial_division_remainder_l2733_273335

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^3 + 3•X^2 - 4) = (X^2 + X - 2) * q + 0 := by
  sorry

end polynomial_division_remainder_l2733_273335


namespace investment_growth_l2733_273311

/-- Represents the investment growth over a two-year period -/
theorem investment_growth 
  (initial_investment : ℝ) 
  (final_investment : ℝ) 
  (growth_rate : ℝ) 
  (h1 : initial_investment = 800) 
  (h2 : final_investment = 960) 
  (h3 : initial_investment * (1 + growth_rate)^2 = final_investment) : 
  800 * (1 + growth_rate)^2 = 960 :=
by sorry

end investment_growth_l2733_273311


namespace wages_payment_duration_l2733_273356

/-- Given a sum of money that can pay two workers' wages separately for different periods,
    this theorem proves how long it can pay both workers together. -/
theorem wages_payment_duration (S : ℝ) (p q : ℝ) (hp : S = 24 * p) (hq : S = 40 * q) :
  ∃ D : ℝ, D = 15 ∧ S = D * (p + q) := by
  sorry

end wages_payment_duration_l2733_273356


namespace certain_number_proof_l2733_273314

theorem certain_number_proof : ∃ x : ℤ, (9823 + x = 13200) ∧ (x = 3377) := by
  sorry

end certain_number_proof_l2733_273314


namespace min_reciprocal_sum_l2733_273398

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
by sorry

end min_reciprocal_sum_l2733_273398


namespace munchausen_polygon_theorem_l2733_273327

/-- A polygon in 2D space -/
structure Polygon :=
  (vertices : Set (ℝ × ℝ))

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- A line in 2D space -/
structure Line :=
  (a b c : ℝ)

/-- Predicate to check if a point is inside a polygon -/
def is_inside (p : Point) (poly : Polygon) : Prop :=
  sorry

/-- Predicate to check if a line intersects a polygon at exactly two points -/
def intersects_at_two_points (l : Line) (poly : Polygon) : Prop :=
  sorry

/-- Predicate to check if a line passes through a point -/
def passes_through (l : Line) (p : Point) : Prop :=
  sorry

/-- Predicate to check if a line divides a polygon into three smaller polygons -/
def divides_into_three (l : Line) (poly : Polygon) : Prop :=
  sorry

/-- Theorem stating that there exists a polygon and a point inside it
    such that any line passing through this point divides the polygon into three smaller polygons -/
theorem munchausen_polygon_theorem :
  ∃ (poly : Polygon) (p : Point),
    is_inside p poly ∧
    ∀ (l : Line),
      passes_through l p →
      intersects_at_two_points l poly ∧
      divides_into_three l poly :=
sorry

end munchausen_polygon_theorem_l2733_273327


namespace no_tetrahedron_with_heights_1_2_3_6_l2733_273317

/-- Represents a tetrahedron with face heights -/
structure Tetrahedron where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ

/-- The theorem stating that a tetrahedron with heights 1, 2, 3, and 6 cannot exist -/
theorem no_tetrahedron_with_heights_1_2_3_6 :
  ¬ ∃ (t : Tetrahedron), t.h₁ = 1 ∧ t.h₂ = 2 ∧ t.h₃ = 3 ∧ t.h₄ = 6 := by
  sorry

end no_tetrahedron_with_heights_1_2_3_6_l2733_273317


namespace emissions_2019_safe_m_range_l2733_273365

/-- Represents the carbon emissions of City A over years -/
def CarbonEmissions (m : ℝ) : ℕ → ℝ
  | 0 => 400  -- 2017 emissions
  | n + 1 => 0.9 * CarbonEmissions m n + m

/-- The maximum allowed annual carbon emissions -/
def MaxEmissions : ℝ := 550

/-- Theorem stating the carbon emissions of City A in 2019 -/
theorem emissions_2019 (m : ℝ) (h : m > 0) : 
  CarbonEmissions m 2 = 324 + 1.9 * m := by sorry

/-- Theorem stating the range of m for which emergency measures are never needed -/
theorem safe_m_range : 
  ∀ m : ℝ, (m > 0 ∧ m ≤ 55) ↔ 
    (∀ n : ℕ, CarbonEmissions m n ≤ MaxEmissions) := by sorry

end emissions_2019_safe_m_range_l2733_273365


namespace base_number_proof_l2733_273321

theorem base_number_proof (n : ℕ) (x : ℕ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = x^22)
  (h2 : n = 21) : x = 4 := by
  sorry

end base_number_proof_l2733_273321


namespace unpainted_cubes_in_4x4x4_cube_l2733_273324

/-- Represents a 4x4x4 cube composed of unit cubes -/
structure Cube :=
  (size : Nat)
  (total_units : Nat)
  (painted_corners : Nat)

/-- The number of unpainted unit cubes in a cube with painted corners -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - c.painted_corners

/-- Theorem stating the number of unpainted cubes in the specific 4x4x4 cube -/
theorem unpainted_cubes_in_4x4x4_cube :
  ∃ (c : Cube), c.size = 4 ∧ c.total_units = 64 ∧ c.painted_corners = 8 ∧ unpainted_cubes c = 56 := by
  sorry

end unpainted_cubes_in_4x4x4_cube_l2733_273324


namespace sum_between_13_and_14_l2733_273345

theorem sum_between_13_and_14 : ∃ x : ℚ, 
  13 < x ∧ x < 14 ∧ 
  x = (3 + 3/8) + (4 + 2/5) + (6 + 1/11) := by
  sorry

end sum_between_13_and_14_l2733_273345


namespace no_multiple_of_four_l2733_273385

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_form_1C34 (n : ℕ) : Prop :=
  ∃ C : ℕ, C < 10 ∧ n = 1000 + 100 * C + 34

theorem no_multiple_of_four :
  ¬∃ n : ℕ, is_four_digit n ∧ has_form_1C34 n ∧ 4 ∣ n :=
sorry

end no_multiple_of_four_l2733_273385


namespace x_intercepts_count_l2733_273318

/-- The number of x-intercepts of y = sin(1/x) in the interval (0.00005, 0.0005) -/
theorem x_intercepts_count : 
  (⌊(20000 : ℝ) / Real.pi⌋ - ⌊(2000 : ℝ) / Real.pi⌋ : ℤ) = 5729 := by
  sorry

end x_intercepts_count_l2733_273318


namespace tv_watching_time_l2733_273355

/-- The number of episodes of Jeopardy watched -/
def jeopardy_episodes : ℕ := 2

/-- The number of episodes of Wheel of Fortune watched -/
def wheel_episodes : ℕ := 2

/-- The duration of one episode of Jeopardy in minutes -/
def jeopardy_duration : ℕ := 20

/-- The duration of one episode of Wheel of Fortune in minutes -/
def wheel_duration : ℕ := 2 * jeopardy_duration

/-- The total time spent watching TV in minutes -/
def total_time : ℕ := jeopardy_episodes * jeopardy_duration + wheel_episodes * wheel_duration

/-- Conversion factor from minutes to hours -/
def minutes_per_hour : ℕ := 60

/-- Theorem: James watched TV for 2 hours -/
theorem tv_watching_time : total_time / minutes_per_hour = 2 := by
  sorry

end tv_watching_time_l2733_273355


namespace friday_temperature_l2733_273346

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem friday_temperature 
  (M T W Th F : ℤ) 
  (avg_mon_to_thu : (M + T + W + Th) / 4 = 48)
  (avg_tue_to_fri : (T + W + Th + F) / 4 = 46)
  (monday_temp : M = 43)
  (all_odd : is_odd M ∧ is_odd T ∧ is_odd W ∧ is_odd Th ∧ is_odd F) :
  F = 35 := by
  sorry

end friday_temperature_l2733_273346


namespace equal_share_of_sweets_l2733_273399

/-- Represents the number of sweets Jennifer has of each color -/
structure Sweets where
  green : Nat
  blue : Nat
  yellow : Nat

/-- The total number of people sharing the sweets -/
def totalPeople : Nat := 4

/-- Jennifer's sweets -/
def jenniferSweets : Sweets := { green := 212, blue := 310, yellow := 502 }

/-- Theorem stating that each person gets 256 sweets when Jennifer shares equally -/
theorem equal_share_of_sweets (s : Sweets) (h : s = jenniferSweets) :
  (s.green + s.blue + s.yellow) / totalPeople = 256 := by
  sorry

end equal_share_of_sweets_l2733_273399


namespace erikas_savings_l2733_273353

theorem erikas_savings (gift_cost cake_cost leftover : ℕ) 
  (h1 : gift_cost = 250)
  (h2 : cake_cost = 25)
  (h3 : leftover = 5)
  (ricks_savings : ℕ) (h4 : ricks_savings = gift_cost / 2)
  (total_savings : ℕ) (h5 : total_savings = gift_cost + cake_cost + leftover) :
  total_savings - ricks_savings = 155 := by
sorry

end erikas_savings_l2733_273353


namespace complex_point_location_l2733_273383

theorem complex_point_location (x y : ℝ) (h : x / (1 + Complex.I) = 1 - y * Complex.I) :
  x > 0 ∧ y > 0 := by
  sorry

end complex_point_location_l2733_273383


namespace common_chord_of_circles_l2733_273351

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord x y :=
by sorry

end common_chord_of_circles_l2733_273351


namespace kyle_gas_and_maintenance_amount_l2733_273390

/-- Calculates the amount left for gas and maintenance given Kyle's income and expenses --/
def amount_for_gas_and_maintenance (monthly_income : ℝ) (rent : ℝ) (utilities : ℝ) 
  (retirement_savings : ℝ) (groceries : ℝ) (insurance : ℝ) (miscellaneous : ℝ) 
  (car_payment : ℝ) : ℝ :=
  monthly_income - (rent + utilities + retirement_savings + groceries + insurance + miscellaneous + car_payment)

/-- Theorem stating that Kyle's amount left for gas and maintenance is $350 --/
theorem kyle_gas_and_maintenance_amount :
  amount_for_gas_and_maintenance 3200 1250 150 400 300 200 200 350 = 350 := by
  sorry

end kyle_gas_and_maintenance_amount_l2733_273390


namespace inscribed_angle_chord_length_l2733_273396

/-- Given a circle with radius R and an inscribed angle α that subtends a chord of length a,
    prove that a = 2R sin α. -/
theorem inscribed_angle_chord_length (R : ℝ) (α : ℝ) (a : ℝ) 
    (h_circle : R > 0) 
    (h_angle : 0 < α ∧ α < π) 
    (h_chord : a > 0) : 
  a = 2 * R * Real.sin α := by
  sorry

end inscribed_angle_chord_length_l2733_273396


namespace solve_exponential_equation_l2733_273307

theorem solve_exponential_equation : ∃ x : ℝ, (100 : ℝ) ^ 4 = 5 ^ x ∧ x = 8 := by
  sorry

end solve_exponential_equation_l2733_273307


namespace number_ordering_l2733_273320

theorem number_ordering (a b c : ℝ) (ha : a = (-0.3)^0) (hb : b = 0.32) (hc : c = 20.3) :
  b < a ∧ a < c := by sorry

end number_ordering_l2733_273320


namespace ellipse_eccentricity_l2733_273330

/-- The eccentricity of the ellipse x^2 + 4y^2 = 4 is √3/2 -/
theorem ellipse_eccentricity : 
  let equation := fun (x y : ℝ) => x^2 + 4*y^2 = 4
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  equation 0 1 ∧ e = Real.sqrt 3 / 2 := by sorry

end ellipse_eccentricity_l2733_273330


namespace f_is_odd_l2733_273325

noncomputable def f (x : ℝ) : ℝ := Real.log ((2 / (1 - x)) - 1) / Real.log 10

theorem f_is_odd : ∀ x : ℝ, x ≠ 1 → f (-x) = -f x := by
  sorry

end f_is_odd_l2733_273325


namespace a_minus_c_equals_three_l2733_273352

theorem a_minus_c_equals_three (a b c d : ℤ) 
  (h1 : a - b = c + d + 9) 
  (h2 : a + b = c - d - 3) : 
  a - c = 3 := by
sorry

end a_minus_c_equals_three_l2733_273352


namespace book_sale_result_l2733_273341

/-- Represents the book sale scenario -/
structure BookSale where
  initial_fiction : ℕ
  initial_nonfiction : ℕ
  fiction_sold : ℕ
  fiction_remaining : ℕ
  total_earnings : ℕ
  fiction_price : ℕ
  nonfiction_price : ℕ

/-- Theorem stating the results of the book sale -/
theorem book_sale_result (sale : BookSale)
  (h1 : sale.fiction_sold = 137)
  (h2 : sale.fiction_remaining = 105)
  (h3 : sale.total_earnings = 685)
  (h4 : sale.fiction_price = 3)
  (h5 : sale.nonfiction_price = 5)
  (h6 : sale.initial_fiction = sale.fiction_sold + sale.fiction_remaining) :
  sale.initial_fiction = 242 ∧
  (sale.total_earnings - sale.fiction_sold * sale.fiction_price) / sale.nonfiction_price = 54 := by
  sorry


end book_sale_result_l2733_273341


namespace parking_lot_levels_l2733_273336

/-- Represents a multi-story parking lot -/
structure ParkingLot where
  totalCapacity : ℕ
  levelCapacity : ℕ
  additionalCars : ℕ
  initialCars : ℕ

/-- Calculates the number of levels in the parking lot -/
def ParkingLot.levels (p : ParkingLot) : ℕ :=
  p.totalCapacity / p.levelCapacity

/-- Theorem: The specific parking lot has 5 levels -/
theorem parking_lot_levels :
  ∀ (p : ParkingLot),
    p.totalCapacity = 425 →
    p.additionalCars = 62 →
    p.initialCars = 23 →
    p.levelCapacity = p.additionalCars + p.initialCars →
    p.levels = 5 := by
  sorry

end parking_lot_levels_l2733_273336


namespace triangle_area_l2733_273306

def a : Fin 2 → ℝ := ![4, -1]
def b : Fin 2 → ℝ := ![3, 5]

theorem triangle_area : 
  (1/2 : ℝ) * |a 0 * b 1 - a 1 * b 0| = 23/2 := by sorry

end triangle_area_l2733_273306


namespace max_intersection_points_ellipse_three_lines_l2733_273371

/-- Represents a line in a 2D plane -/
structure Line :=
  (a b c : ℝ)

/-- Represents an ellipse in a 2D plane -/
structure Ellipse :=
  (a b c d e f : ℝ)

/-- Counts the maximum number of intersection points between an ellipse and a line -/
def maxIntersectionPointsEllipseLine : ℕ := 2

/-- Counts the maximum number of intersection points between two distinct lines -/
def maxIntersectionPointsTwoLines : ℕ := 1

/-- The number of distinct pairs of lines given 3 lines -/
def numLinePairs : ℕ := 3

/-- The number of lines -/
def numLines : ℕ := 3

theorem max_intersection_points_ellipse_three_lines :
  ∀ (e : Ellipse) (l₁ l₂ l₃ : Line),
    l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₂ ≠ l₃ →
    (maxIntersectionPointsEllipseLine * numLines) + 
    (maxIntersectionPointsTwoLines * numLinePairs) = 9 :=
by sorry

end max_intersection_points_ellipse_three_lines_l2733_273371


namespace bucket_capacity_l2733_273391

theorem bucket_capacity (tank_capacity : ℝ) (first_scenario_buckets : ℕ) (second_scenario_buckets : ℕ) (second_scenario_capacity : ℝ) :
  first_scenario_buckets = 30 →
  second_scenario_buckets = 45 →
  second_scenario_capacity = 9 →
  tank_capacity = first_scenario_buckets * (tank_capacity / first_scenario_buckets) →
  tank_capacity = second_scenario_buckets * second_scenario_capacity →
  tank_capacity / first_scenario_buckets = 13.5 := by
sorry

end bucket_capacity_l2733_273391


namespace largest_common_term_l2733_273303

theorem largest_common_term (n m : ℕ) : 
  (∃ n m : ℕ, 187 = 3 + 8 * n ∧ 187 = 5 + 9 * m) ∧ 
  (∀ k : ℕ, k > 187 → k ≤ 200 → ¬(∃ p q : ℕ, k = 3 + 8 * p ∧ k = 5 + 9 * q)) := by
  sorry

end largest_common_term_l2733_273303


namespace gift_bags_total_l2733_273315

theorem gift_bags_total (daily_rate : ℕ) (days_needed : ℕ) (h1 : daily_rate = 42) (h2 : days_needed = 13) :
  daily_rate * days_needed = 546 := by
  sorry

end gift_bags_total_l2733_273315


namespace income_comparison_l2733_273388

theorem income_comparison (juan tim mart : ℝ) 
  (h1 : mart = 1.6 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mart = 0.96 * juan := by
  sorry

end income_comparison_l2733_273388


namespace second_number_calculation_l2733_273343

theorem second_number_calculation (x y z : ℚ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 5 / 8) :
  y = 2400 / 67 := by
sorry

end second_number_calculation_l2733_273343


namespace remainder_problem_l2733_273323

theorem remainder_problem (N : ℕ) : 
  (∃ R, N = 7 * 5 + R ∧ R < 7) → 
  (∃ Q, N = 11 * Q + 2) → 
  N % 7 = 4 := by sorry

end remainder_problem_l2733_273323


namespace election_votes_l2733_273348

theorem election_votes (total_votes : ℕ) 
  (h1 : (70 : ℚ) / 100 * total_votes - (30 : ℚ) / 100 * total_votes = 182) : 
  total_votes = 455 := by
  sorry

end election_votes_l2733_273348


namespace village_population_l2733_273386

/-- The number of residents who speak Bashkir -/
def bashkir_speakers : ℕ := 912

/-- The number of residents who speak Russian -/
def russian_speakers : ℕ := 653

/-- The number of residents who speak both Bashkir and Russian -/
def bilingual_speakers : ℕ := 435

/-- The total number of residents in the village -/
def total_residents : ℕ := bashkir_speakers + russian_speakers - bilingual_speakers

theorem village_population :
  total_residents = 1130 :=
by sorry

end village_population_l2733_273386


namespace line_equations_correct_l2733_273331

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define a line by its slope and a point it passes through
structure Line1 where
  slope : ℝ
  point : Point

-- Define a line by two points it passes through
structure Line2 where
  point1 : Point
  point2 : Point

-- Function to get the equation of a line given slope and point
def lineEquation1 (l : Line1) : ℝ → ℝ → Prop :=
  fun x y => y - l.point.2 = l.slope * (x - l.point.1)

-- Function to get the equation of a line given two points
def lineEquation2 (l : Line2) : ℝ → ℝ → Prop :=
  fun x y => (y - l.point1.2) * (l.point2.1 - l.point1.1) = 
             (l.point2.2 - l.point1.2) * (x - l.point1.1)

theorem line_equations_correct :
  let line1 := Line1.mk (-1/2) (8, -2)
  let line2 := Line2.mk (3, -2) (5, -4)
  (∀ x y, lineEquation1 line1 x y ↔ x + 2*y - 4 = 0) ∧
  (∀ x y, lineEquation2 line2 x y ↔ x + y - 1 = 0) := by
  sorry

end line_equations_correct_l2733_273331


namespace doll_count_sum_l2733_273372

/-- The number of dolls each person has -/
structure DollCounts where
  vera : ℕ
  lisa : ℕ
  sophie : ℕ
  aida : ℕ

/-- The conditions of the doll counting problem -/
def doll_problem (d : DollCounts) : Prop :=
  d.aida = 3 * d.sophie ∧
  d.sophie = 2 * d.vera ∧
  d.vera = d.lisa / 3 ∧
  d.lisa = d.vera + 10 ∧
  d.vera = 15

theorem doll_count_sum (d : DollCounts) : 
  doll_problem d → d.aida + d.sophie + d.vera + d.lisa = 160 := by
  sorry

end doll_count_sum_l2733_273372


namespace fraction_to_decimal_l2733_273395

theorem fraction_to_decimal (numerator denominator : ℕ) (decimal : ℚ) : 
  numerator = 16 → denominator = 50 → decimal = 0.32 → 
  (numerator : ℚ) / (denominator : ℚ) = decimal :=
by
  sorry

end fraction_to_decimal_l2733_273395


namespace unique_root_quadratic_l2733_273397

theorem unique_root_quadratic (X Y Z : ℝ) (hX : X ≠ 0) (hY : Y ≠ 0) (hZ : Z ≠ 0) :
  (∀ t : ℝ, X * t^2 - Y * t + Z = 0 ↔ t = Y) → X = 1/2 := by
  sorry

end unique_root_quadratic_l2733_273397


namespace regular_polygon_sides_l2733_273357

/-- A regular polygon with an exterior angle of 12 degrees has 30 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 12 → (n : ℝ) * exterior_angle = 360 → n = 30 := by
sorry

end regular_polygon_sides_l2733_273357


namespace ln_ln_pi_lt_ln_pi_lt_exp_ln_pi_l2733_273368

theorem ln_ln_pi_lt_ln_pi_lt_exp_ln_pi : 
  Real.log (Real.log Real.pi) < Real.log Real.pi ∧ Real.log Real.pi < 2 ^ Real.log Real.pi := by
  sorry

end ln_ln_pi_lt_ln_pi_lt_exp_ln_pi_l2733_273368


namespace equidistant_points_on_line_in_quadrants_I_II_l2733_273384

/-- A point (x, y) is in the first quadrant if both x and y are positive -/
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- A point (x, y) is in the second quadrant if x is negative and y is positive -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- A point (x, y) is equidistant from the coordinate axes if |x| = |y| -/
def equidistant_from_axes (x y : ℝ) : Prop := abs x = abs y

/-- A point (x, y) is on the line 4x + 6y = 24 -/
def on_line (x y : ℝ) : Prop := 4*x + 6*y = 24

theorem equidistant_points_on_line_in_quadrants_I_II :
  ∃ x y : ℝ, on_line x y ∧ equidistant_from_axes x y ∧ (in_first_quadrant x y ∨ in_second_quadrant x y) ∧
  ∀ x' y' : ℝ, on_line x' y' ∧ equidistant_from_axes x' y' → (in_first_quadrant x' y' ∨ in_second_quadrant x' y') :=
sorry

end equidistant_points_on_line_in_quadrants_I_II_l2733_273384


namespace cosine_function_properties_l2733_273394

/-- 
Given a cosine function y = a * cos(b * x + c) where:
1. The minimum occurs at x = 0
2. The peak-to-peak amplitude is 6
Prove that c = π
-/
theorem cosine_function_properties (a b c : ℝ) : 
  (∀ x, a * Real.cos (b * x + c) ≥ a * Real.cos c) →  -- minimum at x = 0
  (2 * |a| = 6) →                                     -- peak-to-peak amplitude is 6
  c = π :=
by sorry

end cosine_function_properties_l2733_273394


namespace track_length_is_360_l2733_273389

/-- Represents a circular running track -/
structure Track where
  length : ℝ
  start_points_opposite : Bool
  runners_opposite_directions : Bool

/-- Represents a runner on the track -/
structure Runner where
  speed : ℝ
  distance_to_first_meeting : ℝ
  distance_between_meetings : ℝ

/-- The main theorem statement -/
theorem track_length_is_360 (track : Track) (brenda sally : Runner) : 
  track.start_points_opposite ∧ 
  track.runners_opposite_directions ∧
  brenda.distance_to_first_meeting = 120 ∧
  sally.speed = 2 * brenda.speed ∧
  sally.distance_between_meetings = 180 →
  track.length = 360 := by sorry

end track_length_is_360_l2733_273389


namespace cousin_distribution_count_l2733_273358

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- There are 5 rooms -/
def num_rooms : ℕ := 5

/-- The number of ways to distribute the cousins among the rooms -/
def num_distributions : ℕ := distribute num_cousins num_rooms

theorem cousin_distribution_count : num_distributions = 137 := by sorry

end cousin_distribution_count_l2733_273358


namespace board_numbers_l2733_273339

theorem board_numbers (a b : ℕ) (h1 : a > b) (h2 : a = 1580) :
  (((a - b) : ℚ) / (2^10 : ℚ)).isInt → b = 556 := by
  sorry

end board_numbers_l2733_273339


namespace cyclic_sum_root_l2733_273310

theorem cyclic_sum_root {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := (a * b * c) / (a * b + b * c + c * a + 2 * Real.sqrt (a * b * c * (a + b + c)))
  (Real.sqrt (a * b * x * (a + b + x)) + 
   Real.sqrt (b * c * x * (b + c + x)) + 
   Real.sqrt (c * a * x * (c + a + x))) = 
  Real.sqrt (a * b * c * (a + b + c)) :=
by sorry

end cyclic_sum_root_l2733_273310


namespace average_and_difference_l2733_273392

theorem average_and_difference (y : ℝ) : 
  (45 + y) / 2 = 37 → |45 - y| = 16 := by
  sorry

end average_and_difference_l2733_273392


namespace boys_average_weight_l2733_273378

/-- Proves that given a group of 10 students with 5 girls and 5 boys, where the average weight of
    the girls is 45 kg and the average weight of all students is 50 kg, then the average weight
    of the boys is 55 kg. -/
theorem boys_average_weight 
  (num_students : Nat) 
  (num_girls : Nat) 
  (num_boys : Nat) 
  (girls_avg_weight : ℝ) 
  (total_avg_weight : ℝ) : ℝ :=
by
  have h1 : num_students = 10 := by sorry
  have h2 : num_girls = 5 := by sorry
  have h3 : num_boys = 5 := by sorry
  have h4 : girls_avg_weight = 45 := by sorry
  have h5 : total_avg_weight = 50 := by sorry

  -- The average weight of the boys
  let boys_avg_weight : ℝ := 55

  -- Proof that boys_avg_weight = 55
  sorry

end boys_average_weight_l2733_273378


namespace initial_amount_proof_l2733_273329

theorem initial_amount_proof (P : ℚ) : 
  (P * (1 + 1/8) * (1 + 1/8) = 105300) → P = 83200 := by
  sorry

end initial_amount_proof_l2733_273329


namespace cupcake_distribution_exists_l2733_273333

theorem cupcake_distribution_exists (total_cupcakes : ℕ) 
  (cupcakes_per_cousin : ℕ) (cupcakes_per_friend : ℕ) : 
  total_cupcakes = 42 → cupcakes_per_cousin = 3 → cupcakes_per_friend = 2 →
  ∃ (n : ℕ), ∃ (cousins : ℕ), ∃ (friends : ℕ),
    n = cousins + friends ∧ 
    cousins * cupcakes_per_cousin + friends * cupcakes_per_friend = total_cupcakes :=
by sorry

end cupcake_distribution_exists_l2733_273333


namespace hillary_activities_lcm_l2733_273374

theorem hillary_activities_lcm : Nat.lcm (Nat.lcm 6 4) 16 = 48 := by
  sorry

end hillary_activities_lcm_l2733_273374


namespace overall_gain_calculation_l2733_273319

def flat1_purchase : ℝ := 675958
def flat1_gain_percent : ℝ := 0.14

def flat2_purchase : ℝ := 848592
def flat2_loss_percent : ℝ := 0.10

def flat3_purchase : ℝ := 940600
def flat3_gain_percent : ℝ := 0.07

def calculate_selling_price (purchase : ℝ) (gain_percent : ℝ) : ℝ :=
  purchase * (1 + gain_percent)

theorem overall_gain_calculation :
  let flat1_selling := calculate_selling_price flat1_purchase flat1_gain_percent
  let flat2_selling := calculate_selling_price flat2_purchase (-flat2_loss_percent)
  let flat3_selling := calculate_selling_price flat3_purchase flat3_gain_percent
  let total_purchase := flat1_purchase + flat2_purchase + flat3_purchase
  let total_selling := flat1_selling + flat2_selling + flat3_selling
  total_selling - total_purchase = 75617.92 := by
  sorry

end overall_gain_calculation_l2733_273319


namespace exists_team_rating_l2733_273364

variable {Team : Type}
variable (d : Team → Team → ℝ)

axiom goal_difference_symmetry :
  ∀ (A B : Team), d A B + d B A = 0

axiom goal_difference_transitivity :
  ∀ (A B C : Team), d A B + d B C + d C A = 0

theorem exists_team_rating :
  ∃ (f : Team → ℝ), ∀ (A B : Team), d A B = f A - f B :=
sorry

end exists_team_rating_l2733_273364


namespace star_3_5_l2733_273381

/-- The star operation defined for real numbers -/
def star (x y : ℝ) : ℝ := x^2 + x*y + y^2

/-- Theorem stating that 3 ⋆ 5 = 49 -/
theorem star_3_5 : star 3 5 = 49 := by
  sorry

end star_3_5_l2733_273381


namespace special_triangle_angles_l2733_273377

/-- A triangle with excircle radii and circumradius satisfying certain conditions -/
structure SpecialTriangle where
  /-- Excircle radius opposite to side a -/
  r_a : ℝ
  /-- Excircle radius opposite to side b -/
  r_b : ℝ
  /-- Excircle radius opposite to side c -/
  r_c : ℝ
  /-- Circumradius of the triangle -/
  R : ℝ
  /-- First condition: r_a + r_b = 3R -/
  cond1 : r_a + r_b = 3 * R
  /-- Second condition: r_b + r_c = 2R -/
  cond2 : r_b + r_c = 2 * R

/-- The angles of a SpecialTriangle are 30°, 60°, and 90° -/
theorem special_triangle_angles (t : SpecialTriangle) :
  ∃ (A B C : Real),
    A = 30 * π / 180 ∧
    B = 60 * π / 180 ∧
    C = 90 * π / 180 ∧
    A + B + C = π :=
by sorry

end special_triangle_angles_l2733_273377


namespace total_cost_is_49_l2733_273360

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def discount_threshold : ℕ := 35
def discount_amount : ℕ := 5
def num_sandwiches : ℕ := 6
def num_sodas : ℕ := 10

def total_cost : ℕ :=
  let pre_discount := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  if pre_discount > discount_threshold then
    pre_discount - discount_amount
  else
    pre_discount

theorem total_cost_is_49 : total_cost = 49 := by
  sorry

end total_cost_is_49_l2733_273360


namespace hyperbola_eccentricity_from_focus_distance_l2733_273300

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The semi-focal length of a hyperbola -/
def semi_focal_length (h : Hyperbola) : ℝ := sorry

/-- The distance from a focus to an asymptote of a hyperbola -/
def focus_to_asymptote_distance (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity_from_focus_distance (h : Hyperbola) 
  (h_dist : focus_to_asymptote_distance h = (Real.sqrt 5 / 3) * semi_focal_length h) : 
  eccentricity h = 3 / 2 := by sorry

end hyperbola_eccentricity_from_focus_distance_l2733_273300


namespace range_of_a_l2733_273340

-- Define the conditions
def condition_p (x : ℝ) : Prop := x^2 + x - 2 > 0
def condition_q (x a : ℝ) : Prop := x > a

-- Define the sufficient but not necessary relationship
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬(q x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, 
    (sufficient_not_necessary (condition_p) (condition_q a)) → 
    a ≥ 1 :=
by
  sorry

end range_of_a_l2733_273340


namespace davids_trip_money_l2733_273302

theorem davids_trip_money (initial_amount spent_amount remaining_amount : ℕ) :
  remaining_amount = 500 →
  remaining_amount = spent_amount - 500 →
  initial_amount = spent_amount + remaining_amount →
  initial_amount = 1500 := by
sorry

end davids_trip_money_l2733_273302


namespace sufficient_condition_for_inequality_l2733_273332

theorem sufficient_condition_for_inequality (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a < b) (h3 : b < 0) :
  1 / a^2 > 1 / b^2 := by
  sorry

end sufficient_condition_for_inequality_l2733_273332


namespace garrison_size_l2733_273328

/-- The number of men initially in the garrison -/
def initial_men : ℕ := 2000

/-- The number of days the initial provisions would last -/
def initial_days : ℕ := 54

/-- The number of days after which reinforcements arrive -/
def days_before_reinforcement : ℕ := 21

/-- The number of men that arrive as reinforcement -/
def reinforcement : ℕ := 1300

/-- The number of days the provisions last after reinforcement -/
def remaining_days : ℕ := 20

theorem garrison_size :
  initial_men * initial_days = 
  (initial_men + reinforcement) * remaining_days + 
  initial_men * days_before_reinforcement := by
  sorry

end garrison_size_l2733_273328


namespace arithmetic_sequence_difference_l2733_273370

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem arithmetic_sequence_difference :
  let C := arithmetic_sequence 20 15
  let D := arithmetic_sequence 20 (-15)
  |C 31 - D 31| = 900 := by
sorry

end arithmetic_sequence_difference_l2733_273370


namespace early_winner_emerges_l2733_273347

/-- The number of participants in the tournament -/
def n : ℕ := 10

/-- The number of matches each participant plays -/
def matches_per_participant : ℕ := n - 1

/-- The total number of matches in the tournament -/
def total_matches : ℕ := n * matches_per_participant / 2

/-- The number of matches per round -/
def matches_per_round : ℕ := n / 2

/-- The maximum points a participant can score in one round -/
def max_points_per_round : ℚ := 1

/-- The minimum number of rounds required for an early winner to emerge -/
def min_rounds_for_winner : ℕ := 7

theorem early_winner_emerges (
  winner_points : ℚ → ℚ → Prop) 
  (other_max_points : ℚ → ℚ → Prop) : 
  (∀ r : ℕ, r < min_rounds_for_winner → 
    ¬(winner_points r > other_max_points r)) ∧
  (winner_points min_rounds_for_winner > 
    other_max_points min_rounds_for_winner) := by
  sorry

end early_winner_emerges_l2733_273347


namespace problem_solution_l2733_273363

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - 4*x^3 + a*x^2 - 1

-- Define the function g
def g (b : ℝ) (x : ℝ) : ℝ := b*x^2 - 1

theorem problem_solution :
  -- Part 1
  (∀ x y, (0 ≤ x ∧ x < y ∧ y ≤ 1) → f 4 x < f 4 y) ∧
  (∀ x y, (1 ≤ x ∧ x < y ∧ y ≤ 2) → f 4 x > f 4 y) →
  -- Part 2
  (∃ b₁ b₂, b₁ ≠ b₂ ∧
    (∀ b, (∃! x₁ x₂, x₁ ≠ x₂ ∧ f 4 x₁ = g b x₁ ∧ f 4 x₂ = g b x₂) ↔ (b = b₁ ∨ b = b₂))) ∧
  -- Part 3
  (∀ m n, m ∈ Set.Icc (-6 : ℝ) (-2) →
    ((∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f 4 x ≤ m*x^3 + 2*x^2 - n) →
      n ∈ Set.Iic (-4 : ℝ))) := by
  sorry

end problem_solution_l2733_273363


namespace quadratic_polynomial_discriminant_l2733_273379

-- Define a quadratic polynomial
def QuadraticPolynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic polynomial
def Discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_polynomial_discriminant 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∃! x, QuadraticPolynomial a b c x = x - 2) ∧ 
  (∃! x, QuadraticPolynomial a b c x = 1 - x/2) →
  Discriminant a b c = -1/2 := by
sorry

end quadratic_polynomial_discriminant_l2733_273379


namespace matrix_power_4_l2733_273309

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_power_4 : A^4 = !![(-8), 8; 0, 3] := by sorry

end matrix_power_4_l2733_273309


namespace pyramid_volume_approx_l2733_273342

-- Define the pyramid
structure Pyramid where
  baseArea : ℝ
  face1Area : ℝ
  face2Area : ℝ

-- Define the volume function
def pyramidVolume (p : Pyramid) : ℝ :=
  sorry

-- Theorem statement
theorem pyramid_volume_approx (p : Pyramid) 
  (h1 : p.baseArea = 256)
  (h2 : p.face1Area = 120)
  (h3 : p.face2Area = 104) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |pyramidVolume p - 1163| < ε :=
sorry

end pyramid_volume_approx_l2733_273342


namespace phone_rep_hourly_wage_l2733_273316

/-- Calculates the hourly wage for phone reps given the number of reps, hours worked per day, days worked, and total payment -/
def hourly_wage (num_reps : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (total_payment : ℕ) : ℚ :=
  total_payment / (num_reps * hours_per_day * days_worked)

/-- Proves that the hourly wage for phone reps is $14 given the specified conditions -/
theorem phone_rep_hourly_wage :
  hourly_wage 50 8 5 28000 = 14 := by
  sorry

#eval hourly_wage 50 8 5 28000

end phone_rep_hourly_wage_l2733_273316


namespace susan_ate_six_candies_l2733_273366

/-- The number of candies Susan ate during the week -/
def candies_eaten (bought_tuesday bought_thursday bought_friday remaining : ℕ) : ℕ :=
  bought_tuesday + bought_thursday + bought_friday - remaining

/-- Theorem stating that Susan ate 6 candies during the week -/
theorem susan_ate_six_candies :
  candies_eaten 3 5 2 4 = 6 := by
  sorry

end susan_ate_six_candies_l2733_273366


namespace fifty_seventh_pair_l2733_273376

def pair_sequence : ℕ → ℕ × ℕ
| n => sorry

theorem fifty_seventh_pair :
  pair_sequence 57 = (2, 10) := by sorry

end fifty_seventh_pair_l2733_273376
