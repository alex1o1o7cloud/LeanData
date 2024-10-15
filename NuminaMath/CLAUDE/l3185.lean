import Mathlib

namespace NUMINAMATH_CALUDE_cylinder_ellipse_major_axis_length_l3185_318500

/-- Represents the properties of an ellipse formed by intersecting a right circular cylinder with a plane -/
structure CylinderEllipse where
  cylinder_radius : ℝ
  major_axis_ratio : ℝ

/-- Calculates the length of the major axis of the ellipse -/
def major_axis_length (e : CylinderEllipse) : ℝ :=
  2 * e.cylinder_radius * (1 + e.major_axis_ratio)

/-- Theorem stating the length of the major axis for the given conditions -/
theorem cylinder_ellipse_major_axis_length :
  let e : CylinderEllipse := { cylinder_radius := 3, major_axis_ratio := 0.4 }
  major_axis_length e = 8.4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_ellipse_major_axis_length_l3185_318500


namespace NUMINAMATH_CALUDE_total_balloons_count_l3185_318577

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 41

/-- The total number of blue balloons Joan and Melanie have -/
def total_balloons : ℕ := joan_balloons + melanie_balloons

theorem total_balloons_count : total_balloons = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_count_l3185_318577


namespace NUMINAMATH_CALUDE_last_three_digits_of_2_power_10000_l3185_318568

theorem last_three_digits_of_2_power_10000 (h : 2^500 ≡ 1 [ZMOD 1250]) :
  2^10000 ≡ 1 [ZMOD 1000] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_2_power_10000_l3185_318568


namespace NUMINAMATH_CALUDE_cube_product_equals_728_39_l3185_318517

theorem cube_product_equals_728_39 : 
  (((4^3 - 1) / (4^3 + 1)) * 
   ((5^3 - 1) / (5^3 + 1)) * 
   ((6^3 - 1) / (6^3 + 1)) * 
   ((7^3 - 1) / (7^3 + 1)) * 
   ((8^3 - 1) / (8^3 + 1)) * 
   ((9^3 - 1) / (9^3 + 1))) = 728 / 39 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_equals_728_39_l3185_318517


namespace NUMINAMATH_CALUDE_final_x_value_l3185_318502

/-- Represents the state of the program at each iteration -/
structure State where
  x : ℕ
  s : ℕ

/-- Updates the state for one iteration -/
def update_state (st : State) : State :=
  { x := st.x + 3, s := st.s + st.x^2 }

/-- Checks if the termination condition is met -/
def terminate? (st : State) : Bool :=
  st.s ≥ 1000

/-- Runs the program until termination -/
def run_program : ℕ → State → State
  | 0, st => st
  | n + 1, st => if terminate? st then st else run_program n (update_state st)

/-- The initial state of the program -/
def initial_state : State :=
  { x := 4, s := 0 }

theorem final_x_value :
  (run_program 1000 initial_state).x = 22 :=
sorry

end NUMINAMATH_CALUDE_final_x_value_l3185_318502


namespace NUMINAMATH_CALUDE_circle_packing_theorem_l3185_318520

theorem circle_packing_theorem :
  ∃ (n : ℕ+), (n : ℝ) / 2 > 2008 ∧
  ∀ (i j : Fin (n^2)), i ≠ j →
  ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧
  (x - (i.val % n : ℕ) / n)^2 + (y - (i.val / n : ℕ) / n)^2 ≤ (1 / (2*n))^2 ∧
  (x - (j.val % n : ℕ) / n)^2 + (y - (j.val / n : ℕ) / n)^2 ≤ (1 / (2*n))^2 →
  (x - (i.val % n : ℕ) / n)^2 + (y - (i.val / n : ℕ) / n)^2 = (1 / (2*n))^2 ∨
  (x - (j.val % n : ℕ) / n)^2 + (y - (j.val / n : ℕ) / n)^2 = (1 / (2*n))^2 ∨
  ((x - (i.val % n : ℕ) / n) - (x - (j.val % n : ℕ) / n))^2 +
  ((y - (i.val / n : ℕ) / n) - (y - (j.val / n : ℕ) / n))^2 ≥ (1 / n)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_packing_theorem_l3185_318520


namespace NUMINAMATH_CALUDE_skee_ball_tickets_value_l3185_318576

/-- The number of tickets Kaleb won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 8

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 5

/-- The number of candies Kaleb can buy -/
def candies_bought : ℕ := 3

/-- The number of tickets Kaleb won playing 'skee ball' -/
def skee_ball_tickets : ℕ := candy_cost * candies_bought - whack_a_mole_tickets

theorem skee_ball_tickets_value : skee_ball_tickets = 7 := by
  sorry

end NUMINAMATH_CALUDE_skee_ball_tickets_value_l3185_318576


namespace NUMINAMATH_CALUDE_cricket_players_l3185_318525

/-- The number of students who like to play basketball -/
def B : ℕ := 7

/-- The number of students who like to play both basketball and cricket -/
def B_and_C : ℕ := 5

/-- The number of students who like to play basketball or cricket or both -/
def B_or_C : ℕ := 10

/-- The number of students who like to play cricket -/
def C : ℕ := B_or_C - B + B_and_C

theorem cricket_players : C = 8 := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_l3185_318525


namespace NUMINAMATH_CALUDE_larger_integer_value_l3185_318580

theorem larger_integer_value (a b : ℕ+) (h1 : (a : ℚ) / (b : ℚ) = 3 / 2) (h2 : (a : ℕ) * b = 108) : 
  a = ⌊9 * Real.sqrt 2⌋ := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l3185_318580


namespace NUMINAMATH_CALUDE_point_on_circle_l3185_318595

/-- Given a circle C with maximum radius 2 containing points (2,y) and (-2,0),
    prove that the y-coordinate of (2,y) is 0 -/
theorem point_on_circle (y : ℝ) : 
  (∃ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ),
    radius ≤ 2 ∧
    (2, y) ∈ C ∧
    (-2, 0) ∈ C ∧
    C = {p : ℝ × ℝ | dist p center = radius}) →
  y = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_circle_l3185_318595


namespace NUMINAMATH_CALUDE_systematic_sample_largest_l3185_318557

/-- Represents a systematic sample from a range of numbered products -/
structure SystematicSample where
  total_products : Nat
  smallest : Nat
  second_smallest : Nat
  largest : Nat

/-- Theorem stating the properties of the systematic sample in the problem -/
theorem systematic_sample_largest (sample : SystematicSample) : 
  sample.total_products = 300 ∧ 
  sample.smallest = 2 ∧ 
  sample.second_smallest = 17 →
  sample.largest = 287 := by
  sorry

#check systematic_sample_largest

end NUMINAMATH_CALUDE_systematic_sample_largest_l3185_318557


namespace NUMINAMATH_CALUDE_sum_of_powers_zero_l3185_318527

theorem sum_of_powers_zero : 
  (-(1 : ℤ)^2010) + (-1)^2013 + 1^2014 + (-1)^2016 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_zero_l3185_318527


namespace NUMINAMATH_CALUDE_rooster_ratio_l3185_318585

theorem rooster_ratio (total : ℕ) (roosters : ℕ) (hens : ℕ) :
  total = 80 →
  total = roosters + hens →
  roosters + (1/4 : ℚ) * hens = 35 →
  (roosters : ℚ) / total = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_rooster_ratio_l3185_318585


namespace NUMINAMATH_CALUDE_combined_capacity_after_transfer_l3185_318597

/-- Represents the capacity and fill level of a drum --/
structure Drum where
  capacity : ℝ
  fillLevel : ℝ

/-- Theorem stating the combined capacity of three drums --/
theorem combined_capacity_after_transfer
  (drumX : Drum)
  (drumY : Drum)
  (drumZ : Drum)
  (hX : drumX.capacity = A ∧ drumX.fillLevel = 1/2)
  (hY : drumY.capacity = 2*A ∧ drumY.fillLevel = 1/5)
  (hZ : drumZ.capacity = B ∧ drumZ.fillLevel = 1/4)
  : drumX.capacity + drumY.capacity + drumZ.capacity = 3*A + B :=
by
  sorry

#check combined_capacity_after_transfer

end NUMINAMATH_CALUDE_combined_capacity_after_transfer_l3185_318597


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l3185_318590

/-- A function f(x) = ax^2 - x - 1 has exactly one root if and only if a = 0 or a = -1/4 -/
theorem unique_root_quadratic (a : ℝ) : 
  (∃! x, a * x^2 - x - 1 = 0) ↔ (a = 0 ∨ a = -1/4) := by
sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l3185_318590


namespace NUMINAMATH_CALUDE_average_age_calculation_l3185_318545

theorem average_age_calculation (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 36 →
  let total_age := num_students * avg_age_students + num_parents * avg_age_parents
  let total_people := num_students + num_parents
  (total_age / total_people : ℚ) = 26.4 := by
sorry

end NUMINAMATH_CALUDE_average_age_calculation_l3185_318545


namespace NUMINAMATH_CALUDE_lcm_gcd_product_30_75_l3185_318589

theorem lcm_gcd_product_30_75 : Nat.lcm 30 75 * Nat.gcd 30 75 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_30_75_l3185_318589


namespace NUMINAMATH_CALUDE_stream_speed_prove_stream_speed_l3185_318579

/-- The speed of a stream given a swimmer's still water speed and relative upstream/downstream times -/
theorem stream_speed (still_speed : ℝ) (time_ratio : ℝ) : ℝ :=
  let stream_speed := (still_speed * (time_ratio - 1)) / (time_ratio + 1)
  stream_speed

/-- Proves that the speed of the stream is 3 km/h given the conditions -/
theorem prove_stream_speed :
  stream_speed 9 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_prove_stream_speed_l3185_318579


namespace NUMINAMATH_CALUDE_solution_set_k_zero_k_range_two_zeros_l3185_318556

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - k * x else k * x^2 - x + 1

-- Theorem for the solution set when k = 0
theorem solution_set_k_zero :
  {x : ℝ | f 0 x < 2} = {x : ℝ | -1 < x ∧ x < Real.log 2} := by sorry

-- Theorem for the range of k when f has exactly two zeros
theorem k_range_two_zeros :
  ∀ k : ℝ, (∃! x y : ℝ, x ≠ y ∧ f k x = 0 ∧ f k y = 0) ↔ k > Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_k_zero_k_range_two_zeros_l3185_318556


namespace NUMINAMATH_CALUDE_greatest_x_value_l3185_318537

theorem greatest_x_value (x : ℤ) (h : 3.134 * (10 : ℝ) ^ (x : ℝ) < 31000) :
  x ≤ 3 ∧ ∃ y : ℤ, y > 3 → 3.134 * (10 : ℝ) ^ (y : ℝ) ≥ 31000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3185_318537


namespace NUMINAMATH_CALUDE_symmetric_log_value_of_a_l3185_318592

/-- Given a function f and a real number a, we say f is symmetric to log₂(x+a) with respect to y = x -/
def symmetric_to_log (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x = 2^x - a

theorem symmetric_log_value_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_sym : symmetric_to_log f a) (h_sum : f 2 + f 4 = 6) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_log_value_of_a_l3185_318592


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_five_squared_l3185_318518

theorem reciprocal_of_negative_five_squared :
  ((-5 : ℝ)^2)⁻¹ = (1 / 25 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_five_squared_l3185_318518


namespace NUMINAMATH_CALUDE_car_rental_daily_rate_l3185_318561

theorem car_rental_daily_rate (weekly_rate : ℕ) (total_days : ℕ) (total_cost : ℕ) : 
  weekly_rate = 190 → total_days = 11 → total_cost = 310 →
  ∃ (daily_rate : ℕ), daily_rate = 30 ∧ total_cost = weekly_rate + daily_rate * (total_days - 7) :=
by sorry

end NUMINAMATH_CALUDE_car_rental_daily_rate_l3185_318561


namespace NUMINAMATH_CALUDE_ellipse_equation_l3185_318511

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  major_axis : ℝ
  focal_distance : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / (e.major_axis/2)^2 + y^2 / ((e.major_axis/2)^2 - (e.focal_distance/2)^2) = 1

/-- Theorem stating the standard equation for a specific ellipse -/
theorem ellipse_equation (e : Ellipse) 
    (h1 : e.center = (0, 0))
    (h2 : e.major_axis = 18)
    (h3 : e.focal_distance = 12) :
    ∀ x y : ℝ, standard_equation e x y ↔ x^2/81 + y^2/45 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3185_318511


namespace NUMINAMATH_CALUDE_shanna_garden_harvest_l3185_318555

/-- Calculates the total number of vegetables harvested given the initial plant counts and deaths --/
def total_vegetables_harvested (tomato_plants eggplant_plants pepper_plants : ℕ) 
  (tomato_deaths pepper_deaths : ℕ) (vegetables_per_plant : ℕ) : ℕ :=
  let surviving_tomatoes := tomato_plants - tomato_deaths
  let surviving_peppers := pepper_plants - pepper_deaths
  let total_surviving_plants := surviving_tomatoes + surviving_peppers + eggplant_plants
  total_surviving_plants * vegetables_per_plant

/-- Proves that Shanna harvested 56 vegetables given the initial conditions --/
theorem shanna_garden_harvest : 
  total_vegetables_harvested 6 2 4 3 1 7 = 56 := by
  sorry

end NUMINAMATH_CALUDE_shanna_garden_harvest_l3185_318555


namespace NUMINAMATH_CALUDE_inequality_proof_l3185_318593

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3185_318593


namespace NUMINAMATH_CALUDE_division_result_l3185_318599

theorem division_result : (0.05 : ℚ) / (0.002 : ℚ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l3185_318599


namespace NUMINAMATH_CALUDE_function_range_l3185_318594

theorem function_range (f : ℝ → ℝ) : 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x = Real.sin x - Real.sqrt 3 * Real.cos x) →
  Set.range f = Set.Icc (-Real.sqrt 3) 1 := by
sorry

end NUMINAMATH_CALUDE_function_range_l3185_318594


namespace NUMINAMATH_CALUDE_clothing_production_l3185_318505

theorem clothing_production (fabric_A B : ℝ) (sets_B : ℕ) : 
  (fabric_A + 2 * B = 5) →
  (3 * fabric_A + B = 7) →
  (∀ m : ℕ, m + sets_B = 100 → fabric_A * m + B * sets_B ≤ 168) →
  (fabric_A = 1.8 ∧ B = 1.6 ∧ sets_B ≥ 60) :=
by sorry

end NUMINAMATH_CALUDE_clothing_production_l3185_318505


namespace NUMINAMATH_CALUDE_remaining_average_l3185_318550

theorem remaining_average (total : ℝ) (avg1 avg2 : ℝ) :
  total = 6 * 5.40 ∧
  avg1 = 5.2 ∧
  avg2 = 5.80 →
  (total - 2 * avg1 - 2 * avg2) / 2 = 5.20 :=
by sorry

end NUMINAMATH_CALUDE_remaining_average_l3185_318550


namespace NUMINAMATH_CALUDE_hex_1F4B_equals_8011_l3185_318578

-- Define the hexadecimal digits and their decimal equivalents
def hex_to_dec (c : Char) : Nat :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3 | '4' => 4 | '5' => 5 | '6' => 6 | '7' => 7
  | '8' => 8 | '9' => 9 | 'A' => 10 | 'B' => 11 | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

-- Define the conversion function from hexadecimal to decimal
def hex_to_decimal (s : String) : Nat :=
  s.foldl (fun acc c => 16 * acc + hex_to_dec c) 0

-- Theorem statement
theorem hex_1F4B_equals_8011 :
  hex_to_decimal "1F4B" = 8011 := by
  sorry

end NUMINAMATH_CALUDE_hex_1F4B_equals_8011_l3185_318578


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l3185_318524

-- Define the geometric sequences and their properties
def geometric_sequence (k a₂ a₃ : ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 1 ∧ a₂ = k * p ∧ a₃ = k * p^2

-- Define the theorem
theorem sum_of_common_ratios_is_three
  (k a₂ a₃ b₂ b₃ : ℝ)
  (h₁ : geometric_sequence k a₂ a₃)
  (h₂ : geometric_sequence k b₂ b₃)
  (h₃ : ∃ p r : ℝ, p ≠ r ∧ a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2)
  (h₄ : a₃ - b₃ = 3 * (a₂ - b₂))
  (h₅ : k ≠ 0) :
  ∃ p r : ℝ, p + r = 3 ∧ 
    geometric_sequence k a₂ a₃ ∧
    geometric_sequence k b₂ b₃ ∧
    a₂ = k * p ∧ a₃ = k * p^2 ∧
    b₂ = k * r ∧ b₃ = k * r^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l3185_318524


namespace NUMINAMATH_CALUDE_total_fruits_l3185_318541

theorem total_fruits (cucumbers watermelons : ℕ) : 
  cucumbers = 18 → 
  watermelons = cucumbers + 8 → 
  cucumbers + watermelons = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fruits_l3185_318541


namespace NUMINAMATH_CALUDE_nick_running_speed_l3185_318521

/-- Represents the speed required for the fourth lap to achieve a target average speed -/
def fourth_lap_speed (first_three_speed : ℝ) (target_avg_speed : ℝ) : ℝ :=
  4 * target_avg_speed - 3 * first_three_speed

/-- Proves that if a runner completes three laps at 9 mph and needs to achieve an average 
    speed of 10 mph for four laps, then the speed required for the fourth lap is 15 mph -/
theorem nick_running_speed : fourth_lap_speed 9 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_nick_running_speed_l3185_318521


namespace NUMINAMATH_CALUDE_abes_age_l3185_318534

theorem abes_age (present_age : ℕ) 
  (h : present_age + (present_age - 7) = 27) : 
  present_age = 17 := by
sorry

end NUMINAMATH_CALUDE_abes_age_l3185_318534


namespace NUMINAMATH_CALUDE_hiking_duration_is_six_hours_l3185_318598

/-- Represents the hiking scenario with given initial weights and consumption rates. --/
structure HikingScenario where
  initialWater : ℝ
  initialFood : ℝ
  initialGear : ℝ
  waterConsumptionRate : ℝ
  foodConsumptionRate : ℝ

/-- Calculates the remaining weight after a given number of hours. --/
def remainingWeight (scenario : HikingScenario) (hours : ℝ) : ℝ :=
  scenario.initialWater + scenario.initialFood + scenario.initialGear -
  scenario.waterConsumptionRate * hours -
  scenario.foodConsumptionRate * hours

/-- Theorem stating that under the given conditions, the hiking duration is 6 hours. --/
theorem hiking_duration_is_six_hours (scenario : HikingScenario)
  (h1 : scenario.initialWater = 20)
  (h2 : scenario.initialFood = 10)
  (h3 : scenario.initialGear = 20)
  (h4 : scenario.waterConsumptionRate = 2)
  (h5 : scenario.foodConsumptionRate = 2/3)
  (h6 : remainingWeight scenario 6 = 34) :
  ∃ (h : ℝ), h = 6 ∧ remainingWeight scenario h = 34 := by
  sorry


end NUMINAMATH_CALUDE_hiking_duration_is_six_hours_l3185_318598


namespace NUMINAMATH_CALUDE_unique_products_count_l3185_318543

def bag_A : Finset ℕ := {1, 3, 5, 7}
def bag_B : Finset ℕ := {2, 4, 6, 8}

theorem unique_products_count : 
  Finset.card ((bag_A.product bag_B).image (λ (p : ℕ × ℕ) => p.1 * p.2)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_products_count_l3185_318543


namespace NUMINAMATH_CALUDE_smallest_sum_with_gcd_conditions_l3185_318573

theorem smallest_sum_with_gcd_conditions (a b c : ℕ+) : 
  (Nat.gcd a.val (Nat.gcd b.val c.val) = 1) →
  (Nat.gcd a.val (b.val + c.val) > 1) →
  (Nat.gcd b.val (c.val + a.val) > 1) →
  (Nat.gcd c.val (a.val + b.val) > 1) →
  (∃ (x y z : ℕ+), x.val + y.val + z.val < a.val + b.val + c.val) →
  a.val + b.val + c.val ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_with_gcd_conditions_l3185_318573


namespace NUMINAMATH_CALUDE_bug_visits_29_tiles_l3185_318542

/-- Represents a rectangular floor --/
structure RectangularFloor where
  width : ℕ
  length : ℕ

/-- Calculates the number of tiles a bug visits when walking diagonally across a rectangular floor --/
def tilesVisited (floor : RectangularFloor) : ℕ :=
  floor.width + floor.length - Nat.gcd floor.width floor.length

/-- The specific floor in the problem --/
def problemFloor : RectangularFloor :=
  { width := 11, length := 19 }

/-- Theorem stating that a bug walking diagonally across the problem floor visits 29 tiles --/
theorem bug_visits_29_tiles : tilesVisited problemFloor = 29 := by
  sorry

end NUMINAMATH_CALUDE_bug_visits_29_tiles_l3185_318542


namespace NUMINAMATH_CALUDE_radhika_video_games_l3185_318515

/-- The number of video games Radhika received on Christmas. -/
def christmas_games : ℕ := 12

/-- The number of video games Radhika received on her birthday. -/
def birthday_games : ℕ := 8

/-- The number of video games Radhika already owned. -/
def owned_games : ℕ := (christmas_games + birthday_games) / 2

/-- The total number of video games Radhika owns now. -/
def total_games : ℕ := christmas_games + birthday_games + owned_games

theorem radhika_video_games :
  total_games = 30 :=
by sorry

end NUMINAMATH_CALUDE_radhika_video_games_l3185_318515


namespace NUMINAMATH_CALUDE_license_plate_count_l3185_318540

def license_plate_options : ℕ :=
  let first_char_options := 5  -- 3, 5, 6, 8, 9
  let second_char_options := 3 -- B, C, D
  let other_char_options := 4  -- 1, 3, 6, 9
  first_char_options * second_char_options * other_char_options * other_char_options * other_char_options

theorem license_plate_count : license_plate_options = 960 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3185_318540


namespace NUMINAMATH_CALUDE_expression_evaluation_l3185_318570

theorem expression_evaluation (a b : ℝ) (h1 : a = -1) (h2 : b = -4) :
  ((a - 2*b)^2 + (a - 2*b)*(a + 2*b) + 2*a*(2*a - b)) / (2*a) = 9 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3185_318570


namespace NUMINAMATH_CALUDE_expression_value_l3185_318533

theorem expression_value : ((2525 - 2424)^2 + 100) / 225 = 46 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3185_318533


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l3185_318536

theorem subtraction_of_fractions :
  (3 : ℚ) / 2 - (3 : ℚ) / 5 = (9 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l3185_318536


namespace NUMINAMATH_CALUDE_inequality_range_l3185_318574

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x > 2*a*x + a) ↔ -4 < a ∧ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3185_318574


namespace NUMINAMATH_CALUDE_smaller_number_proof_l3185_318558

theorem smaller_number_proof (x y : ℝ) : 
  y = 3 * x + 11 → x + y = 55 → x = 11 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l3185_318558


namespace NUMINAMATH_CALUDE_prop_p_necessary_not_sufficient_for_q_l3185_318503

theorem prop_p_necessary_not_sufficient_for_q :
  (∀ x y : ℝ, x + y ≠ 4 → (x ≠ 1 ∨ y ≠ 3)) ∧
  (∃ x y : ℝ, (x ≠ 1 ∨ y ≠ 3) ∧ x + y = 4) :=
by sorry

end NUMINAMATH_CALUDE_prop_p_necessary_not_sufficient_for_q_l3185_318503


namespace NUMINAMATH_CALUDE_rectangle_ratio_l3185_318501

/-- Given a rectangle with width w, length 10, and perimeter 30, 
    prove that the ratio of width to length is 1:2 -/
theorem rectangle_ratio (w : ℝ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 30 → w / 10 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l3185_318501


namespace NUMINAMATH_CALUDE_weeks_to_save_for_coat_l3185_318559

/-- Calculates the number of weeks needed to save for a coat given specific conditions -/
theorem weeks_to_save_for_coat (weekly_savings : ℚ) (bill_fraction : ℚ) (gift : ℚ) (coat_cost : ℚ) :
  weekly_savings = 25 ∧ 
  bill_fraction = 1/3 ∧ 
  gift = 70 ∧ 
  coat_cost = 170 →
  ∃ w : ℕ, w * weekly_savings - (bill_fraction * 7 * weekly_savings) + gift = coat_cost ∧ w = 19 :=
by sorry

end NUMINAMATH_CALUDE_weeks_to_save_for_coat_l3185_318559


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3185_318562

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (hypotenuse : ℝ) : 
  leg = 15 → 
  angle = 60 * π / 180 → 
  hypotenuse = 10 * Real.sqrt 3 → 
  leg * Real.sin angle = hypotenuse * Real.sin (π / 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3185_318562


namespace NUMINAMATH_CALUDE_diaz_future_age_l3185_318539

/-- Given that 40 less than 10 times Diaz's age is 20 more than 10 times Sierra's age,
    and Sierra is currently 30 years old, prove that Diaz will be 56 years old 20 years from now. -/
theorem diaz_future_age :
  ∀ (diaz_age sierra_age : ℕ),
  sierra_age = 30 →
  10 * diaz_age - 40 = 10 * sierra_age + 20 →
  diaz_age + 20 = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_diaz_future_age_l3185_318539


namespace NUMINAMATH_CALUDE_unique_trig_value_l3185_318575

open Real

theorem unique_trig_value (x : ℝ) (h1 : 0 < x) (h2 : x < π / 3) 
  (h3 : cos x = tan x) : sin x = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_trig_value_l3185_318575


namespace NUMINAMATH_CALUDE_nines_count_to_500_l3185_318516

/-- Count of digit 9 appearances in a number -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Sum of digit 9 appearances in a range of numbers -/
def sum_nines (start finish : ℕ) : ℕ := sorry

/-- The count of digit 9 appearances in all integers from 1 to 500 is 100 -/
theorem nines_count_to_500 : sum_nines 1 500 = 100 := by sorry

end NUMINAMATH_CALUDE_nines_count_to_500_l3185_318516


namespace NUMINAMATH_CALUDE_no_solution_iff_m_leq_neg_one_l3185_318506

theorem no_solution_iff_m_leq_neg_one (m : ℝ) :
  (∀ x : ℝ, ¬(x - m < 0 ∧ 3*x - 1 > 2*(x - 1))) ↔ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_leq_neg_one_l3185_318506


namespace NUMINAMATH_CALUDE_benny_work_hours_l3185_318551

theorem benny_work_hours (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) : 
  hours_per_day = 5 → days_worked = 12 → total_hours = hours_per_day * days_worked → total_hours = 60 := by
  sorry

end NUMINAMATH_CALUDE_benny_work_hours_l3185_318551


namespace NUMINAMATH_CALUDE_negative_reciprocal_inequality_l3185_318546

theorem negative_reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  -1/a < -1/b := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_inequality_l3185_318546


namespace NUMINAMATH_CALUDE_first_term_of_constant_ratio_l3185_318563

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℕ) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (h : d = 5) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → arithmetic_sum a d (4 * n) / arithmetic_sum a d n = c) →
  a = 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_first_term_of_constant_ratio_l3185_318563


namespace NUMINAMATH_CALUDE_real_number_operations_closure_l3185_318552

theorem real_number_operations_closure :
  (∀ (a b : ℝ), ∃ (c : ℝ), a + b = c) ∧
  (∀ (a b : ℝ), ∃ (c : ℝ), a - b = c) ∧
  (∀ (a b : ℝ), ∃ (c : ℝ), a * b = c) ∧
  (∀ (a b : ℝ), b ≠ 0 → ∃ (c : ℝ), a / b = c) :=
by sorry

end NUMINAMATH_CALUDE_real_number_operations_closure_l3185_318552


namespace NUMINAMATH_CALUDE_happy_valley_farm_arrangement_l3185_318512

/-- The number of ways to arrange animals in cages -/
def arrange_animals (chickens dogs cats : Nat) : Nat :=
  Nat.factorial 3 * Nat.factorial chickens * Nat.factorial dogs * Nat.factorial cats

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem happy_valley_farm_arrangement :
  arrange_animals 5 3 4 = 103680 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_farm_arrangement_l3185_318512


namespace NUMINAMATH_CALUDE_translated_cosine_monotonicity_l3185_318530

open Real

theorem translated_cosine_monotonicity (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = 2 * cos (2 * x)) →
  (∀ x, g x = f (x - π / 6)) →
  (∀ x ∈ Set.Icc (2 * a) (7 * π / 6), StrictMono g) →
  a ∈ Set.Icc (π / 3) (7 * π / 12) :=
sorry

end NUMINAMATH_CALUDE_translated_cosine_monotonicity_l3185_318530


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_49_l3185_318522

theorem factor_t_squared_minus_49 : ∀ t : ℝ, t^2 - 49 = (t - 7) * (t + 7) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_49_l3185_318522


namespace NUMINAMATH_CALUDE_base_prime_repr_360_l3185_318572

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- The number 360 -/
def n : ℕ := 360

theorem base_prime_repr_360 :
  base_prime_repr n = [3, 2, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_base_prime_repr_360_l3185_318572


namespace NUMINAMATH_CALUDE_trig_identity_l3185_318569

theorem trig_identity (α : Real) 
  (h : Real.sin α + Real.cos α = 1/5) : 
  ((Real.sin α - Real.cos α)^2 = 49/25) ∧ 
  (Real.sin α^3 + Real.cos α^3 = 37/125) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3185_318569


namespace NUMINAMATH_CALUDE_concatenated_digits_theorem_l3185_318554

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Concatenation of two natural numbers -/
def concatenate (a b : ℕ) : ℕ := sorry

theorem concatenated_digits_theorem :
  num_digits (concatenate (5^1971) (2^1971)) = 1972 := by sorry

end NUMINAMATH_CALUDE_concatenated_digits_theorem_l3185_318554


namespace NUMINAMATH_CALUDE_ceiling_sqrt_196_l3185_318526

theorem ceiling_sqrt_196 : ⌈Real.sqrt 196⌉ = 14 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_196_l3185_318526


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l3185_318510

/-- A quadratic function f(x) = 4x² - mx + 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f m x < f m y

/-- If f(x) = 4x² - mx + 5 is increasing on [-2, +∞), then m ≤ -16 -/
theorem quadratic_increasing_condition (m : ℝ) :
  is_increasing_on_interval m → m ≤ -16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_condition_l3185_318510


namespace NUMINAMATH_CALUDE_dollar_op_six_three_l3185_318549

def dollar_op (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem dollar_op_six_three : dollar_op 6 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_dollar_op_six_three_l3185_318549


namespace NUMINAMATH_CALUDE_percentage_relation_l3185_318567

/-- Given that j is 25% less than p, j is 20% less than t, and t is q% less than p, prove that q = 6.25% -/
theorem percentage_relation (p t j : ℝ) (q : ℝ) 
  (h1 : j = p * (1 - 0.25))
  (h2 : j = t * (1 - 0.20))
  (h3 : t = p * (1 - q / 100)) :
  q = 6.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l3185_318567


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3185_318565

theorem largest_constant_inequality (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^3 + b^3 + c^3 + d^3)) ∧
  ∀ k > 2, ∃ a' b' c' d' : ℝ, 
    (0 ≤ a' ∧ a' ≤ 1) ∧ (0 ≤ b' ∧ b' ≤ 1) ∧ (0 ≤ c' ∧ c' ≤ 1) ∧ (0 ≤ d' ∧ d' ≤ 1) ∧
    (a'^2 * b' + b'^2 * c' + c'^2 * d' + d'^2 * a' + 4 < k * (a'^3 + b'^3 + c'^3 + d'^3)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3185_318565


namespace NUMINAMATH_CALUDE_chicken_nuggets_cost_l3185_318548

/-- Calculates the total cost of chicken nuggets including discount and tax -/
def total_cost (nuggets : ℕ) (box_size : ℕ) (box_price : ℚ) (discount_threshold : ℕ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let boxes := nuggets / box_size
  let initial_cost := boxes * box_price
  let discounted_cost := if nuggets ≥ discount_threshold then initial_cost * (1 - discount_rate) else initial_cost
  let total := discounted_cost * (1 + tax_rate)
  total

/-- The problem statement -/
theorem chicken_nuggets_cost :
  total_cost 100 20 4 80 (75/1000) (8/100) = 1998/100 :=
sorry

end NUMINAMATH_CALUDE_chicken_nuggets_cost_l3185_318548


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l3185_318591

/-- 
Given a cone with base radius √2 whose lateral surface can be unfolded into a semicircle,
prove that the length of its generatrix is 2√2.
-/
theorem cone_generatrix_length 
  (base_radius : ℝ) 
  (h_base_radius : base_radius = Real.sqrt 2) 
  (lateral_surface_is_semicircle : Bool) 
  (h_lateral_surface : lateral_surface_is_semicircle = true) : 
  ∃ (generatrix_length : ℝ), generatrix_length = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l3185_318591


namespace NUMINAMATH_CALUDE_resultant_of_quadratics_l3185_318528

/-- The resultant of two quadratic polynomials -/
def resultant (a b p q : ℝ) : ℝ :=
  (p - a) * (p * b - a * q) + (q - b)^2

/-- Roots of a quadratic polynomial -/
structure QuadraticRoots (a b : ℝ) where
  x₁ : ℝ
  x₂ : ℝ
  sum : x₁ + x₂ = -a
  product : x₁ * x₂ = b

theorem resultant_of_quadratics (a b p q : ℝ) 
  (f_roots : QuadraticRoots a b) (g_roots : QuadraticRoots p q) :
  (f_roots.x₁ - g_roots.x₁) * (f_roots.x₁ - g_roots.x₂) * 
  (f_roots.x₂ - g_roots.x₁) * (f_roots.x₂ - g_roots.x₂) = 
  resultant a b p q := by
  sorry

end NUMINAMATH_CALUDE_resultant_of_quadratics_l3185_318528


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l3185_318523

theorem quadratic_root_k_value (k : ℝ) : 
  (2 : ℝ)^2 - k = 5 → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l3185_318523


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_of_polynomial_l3185_318582

theorem pure_imaginary_solutions_of_polynomial (x : ℂ) :
  (x^4 - 4*x^3 + 6*x^2 - 40*x - 48 = 0) ∧ (∃ k : ℝ, x = k * Complex.I) ↔
  x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_of_polynomial_l3185_318582


namespace NUMINAMATH_CALUDE_jane_max_tickets_l3185_318553

/-- Calculates the maximum number of concert tickets that can be purchased given a budget and pricing structure. -/
def max_tickets (budget : ℕ) (regular_price : ℕ) (discounted_price : ℕ) (discount_threshold : ℕ) : ℕ :=
  let regular_tickets := min discount_threshold (budget / regular_price)
  let remaining_budget := budget - regular_tickets * regular_price
  let discounted_tickets := remaining_budget / discounted_price
  regular_tickets + discounted_tickets

/-- Theorem stating that given the specific conditions, the maximum number of tickets Jane can buy is 8. -/
theorem jane_max_tickets :
  max_tickets 120 15 12 5 = 8 := by
  sorry

#eval max_tickets 120 15 12 5

end NUMINAMATH_CALUDE_jane_max_tickets_l3185_318553


namespace NUMINAMATH_CALUDE_billy_homework_problem_l3185_318547

theorem billy_homework_problem (first_hour second_hour third_hour total : ℕ) : 
  first_hour > 0 →
  second_hour = 2 * first_hour →
  third_hour = 3 * first_hour →
  third_hour = 132 →
  total = first_hour + second_hour + third_hour →
  total = 264 := by
  sorry

end NUMINAMATH_CALUDE_billy_homework_problem_l3185_318547


namespace NUMINAMATH_CALUDE_history_score_l3185_318509

theorem history_score (math_score : ℚ) (third_subject_score : ℚ) (average_score : ℚ) :
  math_score = 74 ∧ third_subject_score = 70 ∧ average_score = 75 →
  (math_score + third_subject_score + (3 * average_score - math_score - third_subject_score)) / 3 = average_score :=
by
  sorry

#eval (74 + 70 + (3 * 75 - 74 - 70)) / 3  -- Should evaluate to 75

end NUMINAMATH_CALUDE_history_score_l3185_318509


namespace NUMINAMATH_CALUDE_points_to_win_match_jeff_tennis_points_l3185_318566

/-- Calculates the number of points needed to win a tennis match given the total playing time,
    point scoring interval, and number of games won. -/
theorem points_to_win_match 
  (total_time : ℕ) 
  (point_interval : ℕ) 
  (games_won : ℕ) : ℕ :=
  let total_minutes := total_time * 60
  let total_points := total_minutes / point_interval
  total_points / games_won

/-- Proves that 8 points are needed to win a match given the specific conditions. -/
theorem jeff_tennis_points : points_to_win_match 2 5 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_points_to_win_match_jeff_tennis_points_l3185_318566


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l3185_318560

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def numerator_sum : ℚ := arithmetic_sum 3 3 33
def denominator_sum : ℚ := arithmetic_sum 4 4 24

theorem arithmetic_sequences_ratio :
  numerator_sum / denominator_sum = 1683 / 1200 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l3185_318560


namespace NUMINAMATH_CALUDE_face_mask_cost_per_box_l3185_318507

/-- Represents the problem of calculating the cost per box of face masks --/
theorem face_mask_cost_per_box :
  ∀ (total_boxes : ℕ) 
    (masks_per_box : ℕ) 
    (repacked_boxes : ℕ) 
    (repacked_price : ℚ) 
    (repacked_quantity : ℕ) 
    (remaining_masks : ℕ) 
    (baggie_price : ℚ) 
    (baggie_quantity : ℕ) 
    (profit : ℚ),
  total_boxes = 12 →
  masks_per_box = 50 →
  repacked_boxes = 6 →
  repacked_price = 5 →
  repacked_quantity = 25 →
  remaining_masks = 300 →
  baggie_price = 3 →
  baggie_quantity = 10 →
  profit = 42 →
  ∃ (cost_per_box : ℚ),
    cost_per_box = 9 ∧
    (repacked_boxes * masks_per_box / repacked_quantity * repacked_price +
     remaining_masks / baggie_quantity * baggie_price) - 
    (total_boxes * cost_per_box) = profit :=
by sorry


end NUMINAMATH_CALUDE_face_mask_cost_per_box_l3185_318507


namespace NUMINAMATH_CALUDE_expression_equals_two_l3185_318514

theorem expression_equals_two (α : Real) (h : Real.tan α = 3) :
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π/2 - α) + Real.cos (π/2 + α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l3185_318514


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3185_318519

theorem algebraic_expression_value 
  (a b m n x : ℝ) 
  (h1 : a = -b) 
  (h2 : m * n = 1) 
  (h3 : |x - 2| = 3) : 
  (a + b - m * n) * x + (a + b) ^ 2022 + (-m * n) ^ 2023 = -6 ∨ 
  (a + b - m * n) * x + (a + b) ^ 2022 + (-m * n) ^ 2023 = 0 :=
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3185_318519


namespace NUMINAMATH_CALUDE_distance_from_point_to_y_axis_l3185_318531

-- Define a point in 2D Cartesian coordinate system
def point : ℝ × ℝ := (3, -5)

-- Define the distance from a point to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ := |p.1|

-- Theorem statement
theorem distance_from_point_to_y_axis :
  distance_to_y_axis point = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_y_axis_l3185_318531


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l3185_318581

/-- The area of a rectangular yard with a square cut out -/
def fenced_area (length width cut_size : ℝ) : ℝ :=
  length * width - cut_size * cut_size

/-- Theorem: The area of a 20-foot by 18-foot rectangular region with a 4-foot by 4-foot square cut out is 344 square feet -/
theorem fenced_area_calculation :
  fenced_area 20 18 4 = 344 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l3185_318581


namespace NUMINAMATH_CALUDE_cosine_sine_difference_equals_sine_double_angle_l3185_318513

theorem cosine_sine_difference_equals_sine_double_angle (α : ℝ) :
  (Real.cos (π / 4 - α))^2 - (Real.sin (π / 4 - α))^2 = Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_difference_equals_sine_double_angle_l3185_318513


namespace NUMINAMATH_CALUDE_percent_relation_l3185_318587

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : z = 1.2 * x) : 
  y = 0.75 * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l3185_318587


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l3185_318596

theorem quadratic_equation_result (y : ℝ) (h : 7 * y^2 + 6 = 5 * y + 14) : 
  (14 * y - 2)^2 = 258 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l3185_318596


namespace NUMINAMATH_CALUDE_exactly_one_true_l3185_318584

-- Define the polynomials
def A (x : ℝ) : ℝ := 2 * x^2
def B (x : ℝ) : ℝ := x + 1
def C (x : ℝ) : ℝ := -2 * x
def D (y : ℝ) : ℝ := y^2
def E (x y : ℝ) : ℝ := 2 * x - y

-- Define the three statements
def statement1 : Prop :=
  ∀ y : ℕ+, ∀ x : ℝ, B x * C x + A x + D y + E x y > 0

def statement2 : Prop :=
  ∃ x y : ℝ, A x + D y + 2 * E x y = -2

def statement3 : Prop :=
  ∀ x : ℝ, ∀ m : ℝ,
    (∃ k : ℝ, 3 * (A x - B x) + m * B x * C x = k * x^2 + (3 : ℝ)) →
    3 * (A x - B x) + m * B x * C x > -3

theorem exactly_one_true : (statement1 ∧ ¬statement2 ∧ ¬statement3) ∨
                           (¬statement1 ∧ statement2 ∧ ¬statement3) ∨
                           (¬statement1 ∧ ¬statement2 ∧ statement3) :=
  sorry

end NUMINAMATH_CALUDE_exactly_one_true_l3185_318584


namespace NUMINAMATH_CALUDE_circle_T_six_three_l3185_318564

-- Define the operation ⊤
def circle_T (a b : ℤ) : ℤ := 4 * a - 7 * b

-- Theorem statement
theorem circle_T_six_three : circle_T 6 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_T_six_three_l3185_318564


namespace NUMINAMATH_CALUDE_prob_not_overcome_is_half_l3185_318532

-- Define the set of elements
inductive Element : Type
| Metal : Element
| Wood : Element
| Water : Element
| Fire : Element
| Earth : Element

-- Define the overcoming relation
def overcomes : Element → Element → Prop
| Element.Metal, Element.Wood => True
| Element.Wood, Element.Earth => True
| Element.Earth, Element.Water => True
| Element.Water, Element.Fire => True
| Element.Fire, Element.Metal => True
| _, _ => False

-- Define the probability of selecting two elements that do not overcome each other
def prob_not_overcome : ℚ :=
  let total_pairs := (5 * 4) / 2  -- C(5,2)
  let overcoming_pairs := 5       -- Number of overcoming relationships
  1 - (overcoming_pairs : ℚ) / total_pairs

-- State the theorem
theorem prob_not_overcome_is_half : prob_not_overcome = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_overcome_is_half_l3185_318532


namespace NUMINAMATH_CALUDE_parallelogram_area_l3185_318535

/-- Proves that the area of a parallelogram with base 7 and altitude twice the base is 98 square units --/
theorem parallelogram_area : ∀ (base altitude area : ℝ),
  base = 7 →
  altitude = 2 * base →
  area = base * altitude →
  area = 98 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3185_318535


namespace NUMINAMATH_CALUDE_expression_value_l3185_318544

theorem expression_value : 
  Real.sqrt 3 / 2 - Real.sqrt 3 * (Real.sin (15 * π / 180))^2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3185_318544


namespace NUMINAMATH_CALUDE_percentage_problem_l3185_318586

theorem percentage_problem (X : ℝ) : 
  (28 / 100) * 400 + (45 / 100) * X = 224.5 → X = 250 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3185_318586


namespace NUMINAMATH_CALUDE_no_real_roots_l3185_318588

theorem no_real_roots : ¬ ∃ (x : ℝ), x^2 + 3*x + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3185_318588


namespace NUMINAMATH_CALUDE_number_of_observations_proof_l3185_318583

theorem number_of_observations_proof (initial_mean : ℝ) (incorrect_obs : ℝ) (correct_obs : ℝ) (new_mean : ℝ) :
  initial_mean = 32 →
  incorrect_obs = 23 →
  correct_obs = 48 →
  new_mean = 32.5 →
  ∃ n : ℕ, n > 0 ∧ n * initial_mean + (correct_obs - incorrect_obs) = n * new_mean ∧ n = 50 :=
by
  sorry

#check number_of_observations_proof

end NUMINAMATH_CALUDE_number_of_observations_proof_l3185_318583


namespace NUMINAMATH_CALUDE_cost_price_change_l3185_318504

theorem cost_price_change (x : ℝ) : 
  (50 * (1 + x / 100) * (1 - x / 100) = 48) → x = 20 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_change_l3185_318504


namespace NUMINAMATH_CALUDE_three_does_not_divide_31_l3185_318538

theorem three_does_not_divide_31 : ¬ ∃ q : ℤ, 31 = 3 * q := by
  sorry

end NUMINAMATH_CALUDE_three_does_not_divide_31_l3185_318538


namespace NUMINAMATH_CALUDE_money_distribution_l3185_318508

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 350)
  (ac_sum : A + C = 200)
  (bc_sum : B + C = 350) :
  C = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3185_318508


namespace NUMINAMATH_CALUDE_midpoint_locus_l3185_318571

-- Define the line l
def line_l (x y : ℝ) : Prop := x / 12 + y / 8 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 24 + y^2 / 16 = 1

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define tangent points A and B on ellipse C
def tangent_point (x y : ℝ) : Prop := ellipse_C x y

-- Define the midpoint M of AB
def midpoint_M (x y x1 y1 x2 y2 : ℝ) : Prop :=
  x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

-- Theorem statement
theorem midpoint_locus
  (P_x P_y A_x A_y B_x B_y M_x M_y : ℝ)
  (h_P : point_P P_x P_y)
  (h_A : tangent_point A_x A_y)
  (h_B : tangent_point B_x B_y)
  (h_M : midpoint_M M_x M_y A_x A_y B_x B_y) :
  (M_x - 1)^2 / (5/2) + (M_y - 1)^2 / (5/3) = 1 :=
sorry

end NUMINAMATH_CALUDE_midpoint_locus_l3185_318571


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3185_318529

/-- Two points are symmetric about the x-axis if their x-coordinates are equal and 
    their y-coordinates are equal in magnitude but opposite in sign -/
def symmetric_about_x_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = x₂ ∧ y₁ = -y₂

/-- Given two points A(2,m) and B(n,-3) that are symmetric about the x-axis,
    prove that m + n = 5 -/
theorem symmetric_points_sum (m n : ℝ) 
  (h : symmetric_about_x_axis 2 m n (-3)) : m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3185_318529
