import Mathlib

namespace ellipse_right_triangle_l314_31488

-- Define the ellipse
def Γ (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the vertices
def A (a b : ℝ) : ℝ × ℝ := (-a, 0)
def B (a b : ℝ) : ℝ × ℝ := (a, 0)
def C (a b : ℝ) : ℝ × ℝ := (0, b)
def D (a b : ℝ) : ℝ × ℝ := (0, -b)

-- Define the theorem
theorem ellipse_right_triangle (a b : ℝ) (P Q R : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  Γ a b P.1 P.2 ∧
  Γ a b Q.1 Q.2 ∧
  P.1 ≥ 0 ∧ P.2 ≥ 0 ∧
  Q.1 ≥ 0 ∧ Q.2 ≥ 0 ∧
  (∃ k : ℝ, Q = k • (A a b - P)) ∧
  (∃ t : ℝ, R = t • ((P.1 / 2, P.2 / 2) : ℝ × ℝ)) ∧
  Γ a b R.1 R.2 →
  ‖Q‖^2 + ‖R‖^2 = ‖B a b - C a b‖^2 :=
sorry

end ellipse_right_triangle_l314_31488


namespace square_product_inequality_l314_31483

theorem square_product_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end square_product_inequality_l314_31483


namespace system_of_equations_solution_l314_31465

theorem system_of_equations_solution :
  ∃! (a b : ℝ), 3*a + 2*b = -26 ∧ 2*a - b = -22 :=
by
  sorry

end system_of_equations_solution_l314_31465


namespace quadratic_factorization_l314_31454

theorem quadratic_factorization (A B : ℤ) :
  (∀ y : ℝ, 10 * y^2 - 51 * y + 21 = (A * y - 7) * (B * y - 3)) →
  A * B + B = 12 := by
  sorry

end quadratic_factorization_l314_31454


namespace cell_phone_customers_l314_31497

theorem cell_phone_customers (us_customers other_customers : ℕ) 
  (h1 : us_customers = 723)
  (h2 : other_customers = 6699) :
  us_customers + other_customers = 7422 := by
sorry

end cell_phone_customers_l314_31497


namespace equilateral_triangle_on_parallel_lines_l314_31477

/-- A line in a plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Three parallel lines --/
def parallel_lines : Vector Line 3 :=
  sorry

/-- Definition of an equilateral triangle --/
def is_equilateral_triangle (a b c : Point) : Prop :=
  sorry

/-- Theorem: There exists an equilateral triangle with vertices on three parallel lines --/
theorem equilateral_triangle_on_parallel_lines :
  ∃ (a b c : Point),
    (∀ i : Fin 3, ∃ j : Fin 3, a.y = parallel_lines[i].slope * a.x + parallel_lines[i].intercept ∨
                               b.y = parallel_lines[i].slope * b.x + parallel_lines[i].intercept ∨
                               c.y = parallel_lines[i].slope * c.x + parallel_lines[i].intercept) ∧
    is_equilateral_triangle a b c :=
  sorry

end equilateral_triangle_on_parallel_lines_l314_31477


namespace cross_product_result_l314_31448

def a : ℝ × ℝ × ℝ := (4, 3, -7)
def b : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

theorem cross_product_result : cross_product a b = (5, -30, -10) := by
  sorry

end cross_product_result_l314_31448


namespace average_weight_problem_l314_31462

theorem average_weight_problem (a b c : ℝ) :
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 37 →
  (b + c) / 2 = 46 :=
by
  sorry

end average_weight_problem_l314_31462


namespace min_reciprocal_sum_l314_31469

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ 1/a + 1/b) →
  1/a + 1/b = 4 :=
by sorry

end min_reciprocal_sum_l314_31469


namespace roberts_soccer_kicks_l314_31478

theorem roberts_soccer_kicks (kicks_before_break kicks_after_break kicks_remaining : ℕ) :
  kicks_before_break = 43 →
  kicks_after_break = 36 →
  kicks_remaining = 19 →
  kicks_before_break + kicks_after_break + kicks_remaining = 98 := by
  sorry

end roberts_soccer_kicks_l314_31478


namespace tenth_number_with_digit_sum_12_l314_31429

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits add up to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 10th number with digit sum 12 is 147 -/
theorem tenth_number_with_digit_sum_12 : nth_number_with_digit_sum_12 10 = 147 := by
  sorry

end tenth_number_with_digit_sum_12_l314_31429


namespace square_and_add_l314_31400

theorem square_and_add (x : ℝ) (h : x = 5) : 2 * x^2 + 5 = 55 := by
  sorry

end square_and_add_l314_31400


namespace find_a_l314_31438

theorem find_a : ∃ a : ℕ, 
  (∀ K : ℤ, K ≠ 27 → ∃ m : ℤ, a - K^3 = m * (27 - K)) → 
  a = 3^9 := by
sorry

end find_a_l314_31438


namespace remainder_53_pow_10_mod_8_l314_31479

theorem remainder_53_pow_10_mod_8 : 53^10 % 8 = 1 := by
  sorry

end remainder_53_pow_10_mod_8_l314_31479


namespace intersection_complement_A_and_B_l314_31424

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- Define the complement of A in ℝ
def complementA : Set ℝ := {x : ℝ | x ≤ 3}

-- State the theorem
theorem intersection_complement_A_and_B :
  (complementA ∩ B) = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end intersection_complement_A_and_B_l314_31424


namespace min_reciprocal_sum_l314_31418

def equation (a b : ℕ) : Prop := 4 * a + b = 6

theorem min_reciprocal_sum :
  ∀ a b : ℕ, equation a b →
  (a ≠ 0 ∧ b ≠ 0) →
  (1 : ℚ) / a + (1 : ℚ) / b ≥ (1 : ℚ) / 1 + (1 : ℚ) / 2 :=
by sorry

end min_reciprocal_sum_l314_31418


namespace sum_of_a_and_b_l314_31441

theorem sum_of_a_and_b (a b : ℝ) (h1 : |a| = 1) (h2 : |b| = 4) (h3 : a * b < 0) :
  a + b = 3 ∨ a + b = -3 := by
  sorry

end sum_of_a_and_b_l314_31441


namespace inequality_implication_l314_31412

theorem inequality_implication (a b c : ℝ) (hc : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by
  sorry

end inequality_implication_l314_31412


namespace minjeong_marbles_l314_31489

/-- Given that the total number of marbles is 43 and Yunjae has 5 more marbles than Minjeong,
    prove that Minjeong has 19 marbles. -/
theorem minjeong_marbles : 
  ∀ (y m : ℕ), y + m = 43 → y = m + 5 → m = 19 := by
  sorry

end minjeong_marbles_l314_31489


namespace sphere_wedge_properties_l314_31466

/-- Represents a sphere cut into eight congruent wedges -/
structure SphereWedge where
  circumference : ℝ
  num_wedges : ℕ

/-- Calculates the volume of one wedge of the sphere -/
def wedge_volume (s : SphereWedge) : ℝ := sorry

/-- Calculates the surface area of one wedge of the sphere -/
def wedge_surface_area (s : SphereWedge) : ℝ := sorry

theorem sphere_wedge_properties (s : SphereWedge) 
  (h1 : s.circumference = 16 * Real.pi)
  (h2 : s.num_wedges = 8) : 
  wedge_volume s = (256 / 3) * Real.pi ∧ 
  wedge_surface_area s = 32 * Real.pi := by sorry

end sphere_wedge_properties_l314_31466


namespace sqrt3_expressions_l314_31476

theorem sqrt3_expressions (x y : ℝ) 
  (hx : x = Real.sqrt 3 + 1) 
  (hy : y = Real.sqrt 3 - 1) : 
  (x^2 + 2*x*y + y^2 = 12) ∧ (x^2 - y^2 = 4 * Real.sqrt 3) := by
  sorry

end sqrt3_expressions_l314_31476


namespace remainder_seven_n_mod_three_l314_31491

theorem remainder_seven_n_mod_three (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := by
  sorry

end remainder_seven_n_mod_three_l314_31491


namespace largest_m_satisfying_inequality_l314_31444

theorem largest_m_satisfying_inequality :
  ∀ m : ℕ, (1 : ℚ) / 4 + (m : ℚ) / 6 < 3 / 2 ↔ m ≤ 7 :=
sorry

end largest_m_satisfying_inequality_l314_31444


namespace painting_cost_is_147_l314_31426

/-- Represents a side of the street with houses --/
structure StreetSide where
  start : ℕ
  diff : ℕ
  count : ℕ

/-- Calculates the cost of painting house numbers for a given street side --/
def calculate_side_cost (side : StreetSide) : ℕ := sorry

/-- Calculates the additional cost for numbers that are multiples of 10 --/
def calculate_multiples_of_10_cost (south : StreetSide) (north : StreetSide) : ℕ := sorry

/-- Main theorem: The total cost of painting all house numbers is $147 --/
theorem painting_cost_is_147 
  (south : StreetSide)
  (north : StreetSide)
  (h_south : south = { start := 5, diff := 7, count := 25 })
  (h_north : north = { start := 6, diff := 8, count := 25 }) :
  calculate_side_cost south + calculate_side_cost north + 
  calculate_multiples_of_10_cost south north = 147 := by sorry

end painting_cost_is_147_l314_31426


namespace student_turtle_difference_is_85_l314_31461

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 20

/-- The number of pet turtles in each fourth-grade classroom -/
def turtles_per_classroom : ℕ := 3

/-- The difference between the total number of students and the total number of turtles -/
def student_turtle_difference : ℕ :=
  num_classrooms * students_per_classroom - num_classrooms * turtles_per_classroom

theorem student_turtle_difference_is_85 : student_turtle_difference = 85 := by
  sorry

end student_turtle_difference_is_85_l314_31461


namespace petya_friends_count_l314_31433

/-- The number of Petya's friends -/
def num_friends : ℕ := 19

/-- The number of stickers Petya has -/
def total_stickers : ℕ := num_friends * 5 + 8

/-- Theorem: Petya has 19 friends -/
theorem petya_friends_count : 
  (total_stickers = num_friends * 5 + 8) ∧ 
  (total_stickers = num_friends * 6 - 11) → 
  num_friends = 19 :=
by
  sorry


end petya_friends_count_l314_31433


namespace regular_polygon_exterior_angle_l314_31453

/-- A regular polygon with interior angle sum of 540° has an exterior angle of 72° --/
theorem regular_polygon_exterior_angle (n : ℕ) : 
  (n - 2) * 180 = 540 → 360 / n = 72 := by sorry

end regular_polygon_exterior_angle_l314_31453


namespace percent_difference_l314_31409

theorem percent_difference (p q : ℝ) (h : p = 1.5 * q) : 
  (q / p) = 2/3 ∧ ((p - q) / q) = 1/2 := by sorry

end percent_difference_l314_31409


namespace helios_population_2060_l314_31446

/-- The population growth function for Helios -/
def helios_population (initial_population : ℕ) (years_passed : ℕ) : ℕ :=
  initial_population * (2 ^ (years_passed / 20))

/-- Theorem stating the population of Helios in 2060 -/
theorem helios_population_2060 :
  helios_population 250 60 = 2000 := by
  sorry

#eval helios_population 250 60

end helios_population_2060_l314_31446


namespace building_height_l314_31436

/-- Given a flagpole and a building casting shadows under similar conditions,
    prove that the height of the building is 22 meters. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 55)
  : (flagpole_height / flagpole_shadow) * building_shadow = 22 :=
by sorry

end building_height_l314_31436


namespace sqrt_negative_eight_a_cubed_l314_31416

theorem sqrt_negative_eight_a_cubed (a : ℝ) (h : a ≤ 0) :
  Real.sqrt (-8 * a^3) = -2 * a * Real.sqrt (-2 * a) :=
by sorry

end sqrt_negative_eight_a_cubed_l314_31416


namespace solution_set_part1_range_of_a_part2_l314_31493

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + x

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 1 x ≤ 7} = Set.Iic 4 := by sorry

-- Part 2
theorem range_of_a_part2 : 
  {a : ℝ | ∀ x, f a x ≥ 2*a + 1} = Set.Iic (-1) := by sorry

end solution_set_part1_range_of_a_part2_l314_31493


namespace duck_buying_problem_l314_31452

theorem duck_buying_problem (adelaide ephraim kolton : ℕ) : 
  adelaide = 30 →
  adelaide = 2 * ephraim →
  kolton = ephraim + 45 →
  (adelaide + ephraim + kolton) % 9 = 0 →
  ephraim ≥ 1 →
  kolton ≥ 1 →
  (adelaide + ephraim + kolton) / 3 = 36 :=
by sorry

end duck_buying_problem_l314_31452


namespace x_intercepts_count_l314_31430

theorem x_intercepts_count : 
  (⌊(100000 : ℝ) / Real.pi⌋ - ⌊(10000 : ℝ) / Real.pi⌋ : ℤ) = 28647 := by
  sorry

end x_intercepts_count_l314_31430


namespace shift_by_two_equiv_l314_31460

/-- A function that represents a vertical shift of another function -/
def verticalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := fun x ↦ f x + shift

/-- Theorem stating that f(x) + 2 is equivalent to shifting f(x) upward by 2 units -/
theorem shift_by_two_equiv (f : ℝ → ℝ) (x : ℝ) : 
  f x + 2 = verticalShift f 2 x := by sorry

end shift_by_two_equiv_l314_31460


namespace mount_everest_temperature_difference_l314_31468

/-- Temperature difference between two points -/
def temperature_difference (t1 : ℝ) (t2 : ℝ) : ℝ := t1 - t2

/-- Temperature at the foot of Mount Everest in °C -/
def foot_temperature : ℝ := 24

/-- Temperature at the summit of Mount Everest in °C -/
def summit_temperature : ℝ := -50

/-- Theorem stating the temperature difference between the foot and summit of Mount Everest -/
theorem mount_everest_temperature_difference :
  temperature_difference foot_temperature summit_temperature = 74 := by
  sorry

end mount_everest_temperature_difference_l314_31468


namespace shaded_rectangle_perimeter_l314_31492

theorem shaded_rectangle_perimeter
  (total_perimeter : ℝ)
  (square_area : ℝ)
  (h_total_perimeter : total_perimeter = 30)
  (h_square_area : square_area = 9) :
  let square_side := Real.sqrt square_area
  let remaining_sum := (total_perimeter / 2) - 2 * square_side
  2 * remaining_sum = 18 :=
by sorry

end shaded_rectangle_perimeter_l314_31492


namespace johnsons_share_l314_31499

/-- 
Given a profit-sharing ratio and Mike's total share, calculate Johnson's share.
-/
theorem johnsons_share 
  (mike_ratio : ℕ) 
  (johnson_ratio : ℕ) 
  (mike_total_share : ℕ) : 
  mike_ratio = 2 → 
  johnson_ratio = 5 → 
  mike_total_share = 1000 → 
  (mike_total_share * johnson_ratio) / mike_ratio = 2500 := by
  sorry

#check johnsons_share

end johnsons_share_l314_31499


namespace sqrt_two_division_l314_31404

theorem sqrt_two_division : 3 * Real.sqrt 2 / Real.sqrt 2 = 3 := by
  sorry

end sqrt_two_division_l314_31404


namespace trivia_team_score_l314_31431

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_points : ℕ) :
  total_members = 12 →
  absent_members = 4 →
  total_points = 64 →
  (total_points / (total_members - absent_members) = 8) :=
by sorry

end trivia_team_score_l314_31431


namespace g_of_13_l314_31464

def g (x : ℝ) : ℝ := x^2 + 2*x + 25

theorem g_of_13 : g 13 = 220 := by
  sorry

end g_of_13_l314_31464


namespace rectangular_to_polar_conversion_l314_31437

theorem rectangular_to_polar_conversion :
  ∃ (r : ℝ) (θ : ℝ), 
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r * Real.cos θ = 2 ∧
    r * Real.sin θ = -2 ∧
    r = 2 * Real.sqrt 2 ∧
    θ = 7 * Real.pi / 4 := by
  sorry

end rectangular_to_polar_conversion_l314_31437


namespace bears_in_stock_is_four_l314_31406

/-- The number of bears in the new shipment -/
def new_shipment : ℕ := 10

/-- The number of bears on each shelf -/
def bears_per_shelf : ℕ := 7

/-- The number of shelves used -/
def shelves_used : ℕ := 2

/-- The number of bears in stock before the shipment -/
def bears_in_stock : ℕ := shelves_used * bears_per_shelf - new_shipment

theorem bears_in_stock_is_four : bears_in_stock = 4 := by
  sorry

end bears_in_stock_is_four_l314_31406


namespace water_tank_capacity_l314_31435

/-- Represents a cylindrical water tank with a given capacity and initial water level. -/
structure WaterTank where
  capacity : ℝ
  initialWater : ℝ

/-- Proves that a water tank with the given properties has a capacity of 30 liters. -/
theorem water_tank_capacity (tank : WaterTank)
  (h1 : tank.initialWater / tank.capacity = 1 / 6)
  (h2 : (tank.initialWater + 5) / tank.capacity = 1 / 3) :
  tank.capacity = 30 := by
  sorry

end water_tank_capacity_l314_31435


namespace contractor_daily_wage_l314_31473

/-- Calculates the daily wage for a contractor given the contract terms and outcomes. -/
def calculate_daily_wage (total_days : ℕ) (fine_per_day : ℚ) (total_received : ℚ) (absent_days : ℕ) : ℚ :=
  let worked_days := total_days - absent_days
  let total_fine := fine_per_day * absent_days
  (total_received + total_fine) / worked_days

/-- Proves that the daily wage is 25 given the specific contract terms. -/
theorem contractor_daily_wage :
  calculate_daily_wage 30 (7.5 : ℚ) 360 12 = 25 := by
  sorry

end contractor_daily_wage_l314_31473


namespace arithmetic_sequence_property_l314_31463

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 4024 = 4) :
  a 2013 = 2 :=
sorry

end arithmetic_sequence_property_l314_31463


namespace carol_and_alex_peanuts_l314_31472

def peanut_distribution (initial_peanuts : ℕ) (multiplier : ℕ) (num_people : ℕ) : ℕ :=
  (initial_peanuts + initial_peanuts * multiplier) / num_people

theorem carol_and_alex_peanuts :
  peanut_distribution 2 5 2 = 6 := by
  sorry

end carol_and_alex_peanuts_l314_31472


namespace banana_arrangements_l314_31456

theorem banana_arrangements : 
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  (total_letters.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end banana_arrangements_l314_31456


namespace x_value_l314_31445

def x : ℚ := (320 / 2) / 3

theorem x_value : x = 160 / 3 := by
  sorry

end x_value_l314_31445


namespace stratified_sampling_middle_managers_l314_31415

theorem stratified_sampling_middle_managers :
  ∀ (total_employees : ℕ) 
    (middle_managers : ℕ) 
    (sample_size : ℕ),
  total_employees = 160 →
  middle_managers = 30 →
  sample_size = 32 →
  (middle_managers * sample_size) / total_employees = 6 :=
by
  sorry

end stratified_sampling_middle_managers_l314_31415


namespace pizza_toppings_combinations_l314_31485

/-- The number of combinations of k items chosen from a set of n items. -/
def combinations (n k : ℕ) : ℕ :=
  if k ≤ n then
    Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  else
    0

/-- Theorem: The number of combinations of 3 toppings chosen from 7 available toppings is 35. -/
theorem pizza_toppings_combinations :
  combinations 7 3 = 35 := by
  sorry

end pizza_toppings_combinations_l314_31485


namespace complex_equation_sum_l314_31427

theorem complex_equation_sum (a b : ℝ) : 
  (a / (1 - Complex.I)) + (b / (1 - 2 * Complex.I)) = (1 + 3 * Complex.I) / 4 → 
  a + b = 2 := by
sorry

end complex_equation_sum_l314_31427


namespace find_a_l314_31451

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}
def B : Set ℝ := Set.Ioo (-3) 2

-- Define the property that A ∩ B is the solution set of x^2 + ax + b < 0
def is_solution_set (a b : ℝ) : Prop :=
  ∀ x, x ∈ A ∩ B ↔ x^2 + a*x + b < 0

-- State the theorem
theorem find_a :
  ∃ b, is_solution_set (-1) b :=
sorry

end find_a_l314_31451


namespace expected_pollen_allergy_l314_31432

theorem expected_pollen_allergy (total_sample : ℕ) (allergy_ratio : ℚ) 
  (h1 : total_sample = 400) 
  (h2 : allergy_ratio = 1 / 4) : 
  ↑total_sample * allergy_ratio = 100 := by
  sorry

end expected_pollen_allergy_l314_31432


namespace inscribed_prism_volume_l314_31470

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- A regular triangular prism inscribed in a regular tetrahedron -/
structure InscribedPrism (a : ℝ) extends RegularTetrahedron a where
  /-- One base of the prism has vertices on the lateral edges of the tetrahedron -/
  base_on_edges : Bool
  /-- The other base of the prism lies in the plane of the tetrahedron's base -/
  base_in_plane : Bool
  /-- All edges of the prism are equal -/
  equal_edges : Bool

/-- The volume of the inscribed prism -/
noncomputable def prism_volume (p : InscribedPrism a) : ℝ :=
  (a^3 * (27 * Real.sqrt 2 - 22 * Real.sqrt 3)) / 2

/-- Theorem: The volume of the inscribed prism is (a³(27√2 - 22√3))/2 -/
theorem inscribed_prism_volume (a : ℝ) (p : InscribedPrism a) :
  prism_volume p = (a^3 * (27 * Real.sqrt 2 - 22 * Real.sqrt 3)) / 2 := by
  sorry

end inscribed_prism_volume_l314_31470


namespace rearrangement_divisible_by_seven_l314_31457

/-- A function that checks if a natural number contains the digits 1, 3, 7, and 9 -/
def containsRequiredDigits (n : ℕ) : Prop := sorry

/-- A function that represents all possible rearrangements of digits in a natural number -/
def rearrangeDigits (n : ℕ) : Set ℕ := sorry

/-- Theorem: For any natural number containing the digits 1, 3, 7, and 9,
    there exists a rearrangement of its digits that is divisible by 7 -/
theorem rearrangement_divisible_by_seven (n : ℕ) :
  containsRequiredDigits n →
  ∃ m ∈ rearrangeDigits n, m % 7 = 0 :=
sorry

end rearrangement_divisible_by_seven_l314_31457


namespace fraction_1790_1799_l314_31414

/-- The number of states that joined the union from 1790 to 1799 -/
def states_1790_1799 : ℕ := 10

/-- The total number of states in Sophie's collection -/
def total_states : ℕ := 25

/-- The fraction of states that joined from 1790 to 1799 among the first 25 states -/
theorem fraction_1790_1799 : 
  (states_1790_1799 : ℚ) / total_states = 2 / 5 := by
  sorry

end fraction_1790_1799_l314_31414


namespace square_equation_solution_l314_31484

theorem square_equation_solution : ∃ x : ℝ, (72 - x)^2 = x^2 ∧ x = 36 := by
  sorry

end square_equation_solution_l314_31484


namespace intersection_M_N_l314_31422

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def N : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x ≤ 4} := by sorry

end intersection_M_N_l314_31422


namespace point_symmetry_l314_31420

def f (x : ℝ) : ℝ := x^3

theorem point_symmetry (a b : ℝ) : 
  (f a = b) → (f (-a) = -b) := by
  sorry

end point_symmetry_l314_31420


namespace share_distribution_l314_31439

theorem share_distribution (total : ℕ) (ratio_a ratio_b ratio_c : ℕ) (share_c : ℕ) :
  total = 945 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  share_c = (total * ratio_c) / (ratio_a + ratio_b + ratio_c) →
  share_c = 420 :=
by sorry

end share_distribution_l314_31439


namespace age_ratio_proof_l314_31482

/-- Prove that given the conditions, the ratio of B's age to C's age is 2:1 -/
theorem age_ratio_proof (A B C : ℕ) : 
  A = B + 2 →  -- A is two years older than B
  A + B + C = 37 →  -- The total of the ages of A, B, and C is 37
  B = 14 →  -- B is 14 years old
  B / C = 2  -- The ratio of B's age to C's age is 2:1
  :=
by sorry

end age_ratio_proof_l314_31482


namespace wire_length_proof_l314_31459

/-- The length of wire used to make an equilateral triangle plus the leftover wire -/
def total_wire_length (side_length : ℝ) (leftover : ℝ) : ℝ :=
  3 * side_length + leftover

/-- Theorem: Given an equilateral triangle with side length 19 cm and 15 cm of leftover wire,
    the total length of wire is 72 cm. -/
theorem wire_length_proof :
  total_wire_length 19 15 = 72 := by
  sorry

end wire_length_proof_l314_31459


namespace figurine_cost_is_17_l314_31474

/-- The cost of each figurine in Annie's purchase --/
def figurine_cost : ℚ :=
  let brand_a_cost : ℚ := 65
  let brand_b_cost : ℚ := 75
  let brand_c_cost : ℚ := 85
  let brand_a_count : ℕ := 3
  let brand_b_count : ℕ := 2
  let brand_c_count : ℕ := 4
  let figurine_count : ℕ := 10
  let figurine_total_cost : ℚ := 2 * brand_c_cost
  figurine_total_cost / figurine_count

theorem figurine_cost_is_17 : figurine_cost = 17 := by
  sorry

end figurine_cost_is_17_l314_31474


namespace ramesh_refrigerator_cost_l314_31408

theorem ramesh_refrigerator_cost 
  (P : ℝ)  -- Labeled price
  (discount_rate : ℝ)  -- Discount rate
  (transport_cost : ℝ)  -- Transport cost
  (installation_cost : ℝ)  -- Installation cost
  (profit_rate : ℝ)  -- Profit rate
  (selling_price : ℝ)  -- Selling price for profit
  (h1 : discount_rate = 0.2)
  (h2 : transport_cost = 125)
  (h3 : installation_cost = 250)
  (h4 : profit_rate = 0.18)
  (h5 : selling_price = 18880)
  (h6 : selling_price = P * (1 + profit_rate)) :
  P * (1 - discount_rate) + transport_cost + installation_cost = 13175 :=
by sorry

end ramesh_refrigerator_cost_l314_31408


namespace james_new_hourly_wage_l314_31495

/-- Jame's hourly wage calculation --/
theorem james_new_hourly_wage :
  ∀ (new_hours_per_week old_hours_per_week old_hourly_wage : ℕ)
    (weeks_per_year : ℕ) (yearly_increase : ℕ),
  new_hours_per_week = 40 →
  old_hours_per_week = 25 →
  old_hourly_wage = 16 →
  weeks_per_year = 52 →
  yearly_increase = 20800 →
  ∃ (new_hourly_wage : ℕ),
    new_hourly_wage = 530 ∧
    new_hourly_wage * new_hours_per_week * weeks_per_year =
      old_hourly_wage * old_hours_per_week * weeks_per_year + yearly_increase :=
by
  sorry

end james_new_hourly_wage_l314_31495


namespace total_stripes_is_34_l314_31419

/-- The total number of stripes on Vaishali's hats -/
def total_stripes : ℕ :=
  let hats_with_3_stripes := 4
  let hats_with_4_stripes := 3
  let hats_with_0_stripes := 6
  let hats_with_5_stripes := 2
  hats_with_3_stripes * 3 +
  hats_with_4_stripes * 4 +
  hats_with_0_stripes * 0 +
  hats_with_5_stripes * 5

/-- Theorem stating that the total number of stripes on Vaishali's hats is 34 -/
theorem total_stripes_is_34 : total_stripes = 34 := by
  sorry

end total_stripes_is_34_l314_31419


namespace conic_eccentricity_l314_31443

/-- A conic section with foci F₁ and F₂ -/
structure ConicSection where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a conic section -/
def eccentricity (c : ConicSection) : ℝ := sorry

/-- A point on a conic section -/
def Point (c : ConicSection) := ℝ × ℝ

theorem conic_eccentricity (c : ConicSection) :
  ∃ (P : Point c), 
    distance P c.F₁ / distance c.F₁ c.F₂ = 4/3 ∧
    distance c.F₁ c.F₂ / distance P c.F₂ = 3/2 →
    eccentricity c = 1/2 ∨ eccentricity c = 3/2 := by
  sorry

end conic_eccentricity_l314_31443


namespace bear_food_per_day_l314_31498

/-- The weight of Victor in pounds -/
def victor_weight : ℝ := 126

/-- The number of "Victors" worth of food a bear eats in 3 weeks -/
def victors_eaten : ℝ := 15

/-- The number of weeks -/
def weeks : ℝ := 3

/-- The number of days in a week -/
def days_per_week : ℝ := 7

/-- Theorem: A bear eats 90 pounds of food per day -/
theorem bear_food_per_day :
  (victor_weight * victors_eaten) / (weeks * days_per_week) = 90 := by
  sorry

end bear_food_per_day_l314_31498


namespace arithmetic_mean_of_reciprocals_first_four_primes_l314_31496

def first_four_primes : List ℕ := [2, 3, 5, 7]

theorem arithmetic_mean_of_reciprocals_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length : ℚ) = 247 / 840 := by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l314_31496


namespace car_service_month_l314_31450

/-- Represents the months of the year -/
inductive Month : Type
| jan | feb | mar | apr | may | jun | jul | aug | sep | oct | nov | dec

/-- Convert a number to a month -/
def num_to_month (n : Nat) : Month :=
  match n % 12 with
  | 1 => Month.jan
  | 2 => Month.feb
  | 3 => Month.mar
  | 4 => Month.apr
  | 5 => Month.may
  | 6 => Month.jun
  | 7 => Month.jul
  | 8 => Month.aug
  | 9 => Month.sep
  | 10 => Month.oct
  | 11 => Month.nov
  | _ => Month.dec

theorem car_service_month (service_interval : Nat) (first_service : Month) (n : Nat) :
  service_interval = 7 →
  first_service = Month.jan →
  n = 30 →
  num_to_month ((n - 1) * service_interval % 12 + 1) = Month.dec :=
by
  sorry

end car_service_month_l314_31450


namespace ellipse_major_axis_length_l314_31407

/-- Represents a cylinder with two spheres inside it -/
structure CylinderWithSpheres where
  cylinderRadius : ℝ
  sphereRadius : ℝ
  sphereCenterDistance : ℝ

/-- Represents the ellipse formed by the intersection of a plane with the cylinder -/
structure IntersectionEllipse where
  majorAxis : ℝ

/-- The length of the major axis of the ellipse formed by a plane tangent to both spheres 
    and intersecting the cylindrical surface is equal to the distance between sphere centers -/
theorem ellipse_major_axis_length 
  (c : CylinderWithSpheres) 
  (h1 : c.cylinderRadius = 6) 
  (h2 : c.sphereRadius = 6) 
  (h3 : c.sphereCenterDistance = 13) : 
  ∃ e : IntersectionEllipse, e.majorAxis = c.sphereCenterDistance :=
sorry

end ellipse_major_axis_length_l314_31407


namespace line_contains_point_l314_31401

/-- 
Given a line represented by the equation 1-kx = -3y that contains the point (4, -3),
prove that the value of k is -2.
-/
theorem line_contains_point (k : ℝ) : 
  (1 - k * 4 = -3 * (-3)) → k = -2 := by sorry

end line_contains_point_l314_31401


namespace min_m_value_l314_31413

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := Real.exp x - Real.exp (-x)
def g (m : ℝ) (x : ℝ) := Real.log (m * x^2 - x + 1/4)

-- State the theorem
theorem min_m_value :
  (∀ x1 ∈ Set.Iic (0 : ℝ), ∃ x2 : ℝ, f x1 = g m x2) →
  (∀ m' : ℝ, m' < -1/3 → ¬(∀ x1 ∈ Set.Iic (0 : ℝ), ∃ x2 : ℝ, f x1 = g m' x2)) →
  (∃ x1 ∈ Set.Iic (0 : ℝ), ∃ x2 : ℝ, f x1 = g (-1/3) x2) →
  m = -1/3 :=
sorry

end

end min_m_value_l314_31413


namespace smallest_surface_areas_100_cubes_l314_31411

/-- Represents a polyhedron formed by unit cubes -/
structure Polyhedron :=
  (length width height : ℕ)
  (total_cubes : ℕ)
  (surface_area : ℕ)

/-- Calculates the surface area of a rectangular prism -/
def calculate_surface_area (l w h : ℕ) : ℕ :=
  2 * (l * w + l * h + w * h)

/-- Generates all possible polyhedra from 100 unit cubes -/
def generate_polyhedra (n : ℕ) : List Polyhedron :=
  sorry

/-- Finds the first 6 smallest surface areas -/
def first_6_surface_areas (polyhedra : List Polyhedron) : List ℕ :=
  sorry

theorem smallest_surface_areas_100_cubes :
  let polyhedra := generate_polyhedra 100
  let areas := first_6_surface_areas polyhedra
  areas = [130, 134, 136, 138, 140, 142] :=
sorry

end smallest_surface_areas_100_cubes_l314_31411


namespace power_of_power_l314_31440

theorem power_of_power (a : ℝ) : (a^5)^2 = a^10 := by
  sorry

end power_of_power_l314_31440


namespace x_gt_3_sufficient_not_necessary_l314_31417

theorem x_gt_3_sufficient_not_necessary :
  (∃ x : ℝ, x ≤ 3 ∧ 1 / x < 1 / 3) ∧
  (∀ x : ℝ, x > 3 → 1 / x < 1 / 3) :=
by sorry

end x_gt_3_sufficient_not_necessary_l314_31417


namespace reynald_soccer_balls_l314_31442

/-- The number of soccer balls Reynald bought -/
def soccer_balls : ℕ := 20

/-- The total number of balls Reynald bought -/
def total_balls : ℕ := 145

/-- The number of volleyballs Reynald bought -/
def volleyballs : ℕ := 30

theorem reynald_soccer_balls :
  soccer_balls = 20 ∧
  soccer_balls + (soccer_balls + 5) + (2 * soccer_balls) + (soccer_balls + 10) + volleyballs = total_balls :=
by sorry

end reynald_soccer_balls_l314_31442


namespace buses_needed_l314_31402

theorem buses_needed (classrooms : ℕ) (students_per_classroom : ℕ) (seats_per_bus : ℕ) : 
  classrooms = 67 → students_per_classroom = 66 → seats_per_bus = 6 →
  (classrooms * students_per_classroom + seats_per_bus - 1) / seats_per_bus = 738 := by
  sorry

end buses_needed_l314_31402


namespace mrsHiltFramePerimeter_l314_31490

/-- Represents an irregular pentagonal picture frame with given side lengths -/
structure IrregularPentagon where
  base : ℝ
  leftSide : ℝ
  rightSide : ℝ
  topLeftDiagonal : ℝ
  topRightDiagonal : ℝ

/-- Calculates the perimeter of an irregular pentagonal picture frame -/
def perimeter (p : IrregularPentagon) : ℝ :=
  p.base + p.leftSide + p.rightSide + p.topLeftDiagonal + p.topRightDiagonal

/-- Mrs. Hilt's irregular pentagonal picture frame -/
def mrsHiltFrame : IrregularPentagon :=
  { base := 10
    leftSide := 12
    rightSide := 11
    topLeftDiagonal := 6
    topRightDiagonal := 7 }

/-- Theorem: The perimeter of Mrs. Hilt's irregular pentagonal picture frame is 46 inches -/
theorem mrsHiltFramePerimeter : perimeter mrsHiltFrame = 46 := by
  sorry

end mrsHiltFramePerimeter_l314_31490


namespace zeros_before_first_nonzero_l314_31455

theorem zeros_before_first_nonzero (n : ℕ) (m : ℕ) : 
  let fraction := 1 / (2^n * 5^m)
  let zeros := m - n
  zeros > 0 → zeros = 5 :=
by
  sorry

end zeros_before_first_nonzero_l314_31455


namespace parallelogram_circle_theorem_l314_31434

/-- Represents a parallelogram KLMN with a circle tangent to NK and NM, passing through L, 
    and intersecting KL at C and ML at D. -/
structure ParallelogramWithCircle where
  -- The length of side KL
  kl : ℝ
  -- The ratio KC : LC
  kc_lc_ratio : ℝ × ℝ
  -- The ratio LD : MD
  ld_md_ratio : ℝ × ℝ

/-- Theorem stating that under the given conditions, KN = 10 -/
theorem parallelogram_circle_theorem (p : ParallelogramWithCircle) 
  (h1 : p.kl = 8)
  (h2 : p.kc_lc_ratio = (4, 5))
  (h3 : p.ld_md_ratio = (8, 1)) :
  ∃ (kn : ℝ), kn = 10 := by
  sorry

end parallelogram_circle_theorem_l314_31434


namespace problem_solution_l314_31494

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x/3 = y^2) 
  (h3 : x/5 = 5*y + 2) : 
  x = (685 + 25 * Real.sqrt 745) / 6 := by
sorry

end problem_solution_l314_31494


namespace valid_plans_count_l314_31487

/-- Represents the three universities --/
inductive University : Type
| Peking : University
| Tsinghua : University
| Renmin : University

/-- Represents the five students --/
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

/-- A recommendation plan is a function from Student to University --/
def RecommendationPlan := Student → University

/-- Checks if a recommendation plan is valid --/
def isValidPlan (plan : RecommendationPlan) : Prop :=
  (∃ s, plan s = University.Peking) ∧
  (∃ s, plan s = University.Tsinghua) ∧
  (∃ s, plan s = University.Renmin) ∧
  (plan Student.A ≠ University.Peking)

/-- The number of valid recommendation plans --/
def numberOfValidPlans : ℕ := sorry

theorem valid_plans_count : numberOfValidPlans = 100 := by sorry

end valid_plans_count_l314_31487


namespace rectangle_area_l314_31405

/-- Given a rectangle with perimeter 100 cm and diagonal x cm, its area is 1250 - (x^2 / 2) square cm -/
theorem rectangle_area (x : ℝ) :
  let perimeter : ℝ := 100
  let diagonal : ℝ := x
  let area : ℝ := 1250 - (x^2 / 2)
  (∃ (length width : ℝ),
    length > 0 ∧ 
    width > 0 ∧
    2 * (length + width) = perimeter ∧
    length^2 + width^2 = diagonal^2 ∧
    length * width = area) :=
by sorry

end rectangle_area_l314_31405


namespace find_fifth_month_sale_l314_31481

def sales_problem (sales : Fin 6 → ℕ) (average : ℕ) : Prop :=
  sales 0 = 800 ∧
  sales 1 = 900 ∧
  sales 2 = 1000 ∧
  sales 3 = 700 ∧
  sales 5 = 900 ∧
  (sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = average

theorem find_fifth_month_sale (sales : Fin 6 → ℕ) (average : ℕ) 
  (h : sales_problem sales average) : sales 4 = 800 := by
  sorry

end find_fifth_month_sale_l314_31481


namespace gcd_192_144_320_l314_31486

theorem gcd_192_144_320 : Nat.gcd 192 (Nat.gcd 144 320) = 16 := by sorry

end gcd_192_144_320_l314_31486


namespace equipment_cannot_fit_l314_31423

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The L-shaped corridor width -/
def corridorWidth : ℝ := 3

/-- The center of the unit circle representing the corner of the L-shaped corridor -/
def circleCenter : Point := ⟨corridorWidth, corridorWidth⟩

/-- The radius of the circle representing the corner of the L-shaped corridor -/
def circleRadius : ℝ := 1

/-- The origin point -/
def origin : Point := ⟨0, 0⟩

/-- The maximum length of the equipment -/
def maxEquipmentLength : ℝ := 7

/-- A line passing through the origin -/
structure Line where
  a : ℝ
  b : ℝ

/-- The length of the line segment from the origin to its intersection with the circle -/
def lineSegmentLength (l : Line) : ℝ := sorry

/-- The minimum length of a line segment intersecting the circle and passing through the origin -/
def minLineSegmentLength : ℝ := sorry

/-- Theorem stating that the equipment cannot fit through the L-shaped corridor -/
theorem equipment_cannot_fit : minLineSegmentLength > maxEquipmentLength := by sorry

end equipment_cannot_fit_l314_31423


namespace min_value_of_expression_l314_31480

theorem min_value_of_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) / (c + d) + (a + c) / (b + d) + (a + d) / (b + c) +
  (b + c) / (a + d) + (b + d) / (a + c) + (c + d) / (a + b) ≥ 6 ∧
  ((a + b) / (c + d) + (a + c) / (b + d) + (a + d) / (b + c) +
   (b + c) / (a + d) + (b + d) / (a + c) + (c + d) / (a + b) = 6 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end min_value_of_expression_l314_31480


namespace corporation_full_time_employees_l314_31447

/-- Given a corporation with part-time and full-time employees, 
    we calculate the number of full-time employees. -/
theorem corporation_full_time_employees 
  (total_employees : ℕ) 
  (part_time_employees : ℕ) 
  (h1 : total_employees = 65134) 
  (h2 : part_time_employees = 2041) : 
  total_employees - part_time_employees = 63093 := by
  sorry

end corporation_full_time_employees_l314_31447


namespace all_stars_arrangement_l314_31458

/-- The number of ways to arrange All-Stars in a row -/
def arrange_all_stars (total : ℕ) (cubs : ℕ) (red_sox : ℕ) (yankees : ℕ) (dodgers : ℕ) : ℕ :=
  Nat.factorial 4 * Nat.factorial cubs * Nat.factorial red_sox * Nat.factorial yankees * Nat.factorial dodgers

/-- Theorem stating the number of arrangements for the given problem -/
theorem all_stars_arrangement :
  arrange_all_stars 10 4 3 2 1 = 6912 := by
  sorry

end all_stars_arrangement_l314_31458


namespace probability_second_quality_l314_31425

theorem probability_second_quality (p : ℝ) : 
  (1 - p^2 = 0.91) → p = 0.3 := by
  sorry

end probability_second_quality_l314_31425


namespace mean_squares_sum_l314_31428

theorem mean_squares_sum (x y z : ℝ) : 
  (x + y + z) / 3 = 10 →
  (x * y * z) ^ (1/3 : ℝ) = 6 →
  3 / (1/x + 1/y + 1/z) = 4 →
  x^2 + y^2 + z^2 = 576 := by
sorry

end mean_squares_sum_l314_31428


namespace ticket_circle_circumference_l314_31410

/-- The circumference of a circle formed by overlapping tickets -/
theorem ticket_circle_circumference
  (ticket_length : ℝ)
  (overlap : ℝ)
  (num_tickets : ℕ)
  (h1 : ticket_length = 10.4)
  (h2 : overlap = 3.5)
  (h3 : num_tickets = 16) :
  (ticket_length - overlap) * num_tickets = 110.4 :=
by sorry

end ticket_circle_circumference_l314_31410


namespace smallest_solution_quartic_l314_31471

theorem smallest_solution_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 →
  x ≥ -Real.sqrt 26 ∧
  ∃ y, y^4 - 50*y^2 + 576 = 0 ∧ y = -Real.sqrt 26 :=
by sorry

end smallest_solution_quartic_l314_31471


namespace max_value_when_t_2_t_value_when_max_2_l314_31449

-- Define the function f(x, t)
def f (x t : ℝ) : ℝ := |2 * x - 1| - |t * x + 3|

-- Theorem 1: Maximum value of f(x) when t = 2 is 4
theorem max_value_when_t_2 :
  ∃ M : ℝ, M = 4 ∧ ∀ x : ℝ, f x 2 ≤ M :=
sorry

-- Theorem 2: When maximum value of f(x) is 2, t = 6
theorem t_value_when_max_2 :
  ∃ t : ℝ, t > 0 ∧ (∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, f x t ≤ M) → t = 6 :=
sorry

end max_value_when_t_2_t_value_when_max_2_l314_31449


namespace fraction_of_b_equal_to_third_of_a_prove_fraction_of_b_equal_to_third_of_a_l314_31403

theorem fraction_of_b_equal_to_third_of_a : ℝ → ℝ → ℝ → Prop :=
  fun a b x =>
    a + b = 1210 →
    b = 484 →
    (1/3) * a = x * b →
    x = 1/2

-- Proof
theorem prove_fraction_of_b_equal_to_third_of_a :
  ∃ (a b x : ℝ), fraction_of_b_equal_to_third_of_a a b x :=
by
  sorry

end fraction_of_b_equal_to_third_of_a_prove_fraction_of_b_equal_to_third_of_a_l314_31403


namespace intersection_A_complement_B_l314_31421

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 3}
def B : Set Nat := {3, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {2} := by sorry

end intersection_A_complement_B_l314_31421


namespace friends_picnic_only_l314_31467

/-- Given information about friends meeting for different activities, 
    prove that the number of friends meeting for picnic only is 20. -/
theorem friends_picnic_only (total : ℕ) (movie : ℕ) (games : ℕ) 
  (movie_picnic : ℕ) (movie_games : ℕ) (picnic_games : ℕ) (all_three : ℕ) :
  total = 31 ∧ 
  movie = 10 ∧ 
  games = 5 ∧ 
  movie_picnic = 4 ∧ 
  movie_games = 2 ∧ 
  picnic_games = 0 ∧ 
  all_three = 2 → 
  ∃ (movie_only picnic_only games_only : ℕ),
    total = movie_only + picnic_only + games_only + movie_picnic + movie_games + picnic_games + all_three ∧
    movie = movie_only + movie_picnic + movie_games + all_three ∧
    games = games_only + movie_games + all_three ∧
    picnic_only = 20 := by
  sorry

end friends_picnic_only_l314_31467


namespace parallel_vectors_m_value_l314_31475

/-- Given vectors a, b, and c in ℝ², prove that if a is parallel to m*b - c, then m = -3. -/
theorem parallel_vectors_m_value (a b c : ℝ × ℝ) (m : ℝ) 
    (ha : a = (2, -1))
    (hb : b = (1, 0))
    (hc : c = (1, -2))
    (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • (m • b - c)) :
  m = -3 := by
  sorry

end parallel_vectors_m_value_l314_31475
