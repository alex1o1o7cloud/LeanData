import Mathlib

namespace NUMINAMATH_CALUDE_butterfat_mixture_l3035_303568

/-- Proves that mixing 8 gallons of 50% butterfat milk with 24 gallons of 10% butterfat milk 
    results in a mixture that is 20% butterfat -/
theorem butterfat_mixture : 
  let milk_50_percent : ℝ := 8
  let milk_10_percent : ℝ := 24
  let butterfat_50_percent : ℝ := 0.5
  let butterfat_10_percent : ℝ := 0.1
  let target_butterfat_percent : ℝ := 0.2
  
  (milk_50_percent * butterfat_50_percent + milk_10_percent * butterfat_10_percent) / 
  (milk_50_percent + milk_10_percent) = target_butterfat_percent :=
by
  sorry

#check butterfat_mixture

end NUMINAMATH_CALUDE_butterfat_mixture_l3035_303568


namespace NUMINAMATH_CALUDE_dan_has_five_limes_l3035_303564

/-- The number of limes Dan has after giving some to Sara -/
def dans_remaining_limes (initial_limes : ℕ) (limes_given : ℕ) : ℕ :=
  initial_limes - limes_given

/-- Theorem stating that Dan has 5 limes after giving 4 to Sara -/
theorem dan_has_five_limes :
  dans_remaining_limes 9 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dan_has_five_limes_l3035_303564


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l3035_303594

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l3035_303594


namespace NUMINAMATH_CALUDE_odd_function_sum_l3035_303549

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f 3 = -2) :
  f (-3) + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l3035_303549


namespace NUMINAMATH_CALUDE_train_speed_problem_l3035_303515

/-- Given two trains traveling in opposite directions, this theorem proves
    the speed of the second train given the conditions of the problem. -/
theorem train_speed_problem (v : ℝ) : v = 50 := by
  -- Define the speed of the first train
  let speed1 : ℝ := 64
  -- Define the time of travel
  let time : ℝ := 2.5
  -- Define the total distance between trains after the given time
  let total_distance : ℝ := 285
  
  -- The equation representing the problem:
  -- speed1 * time + v * time = total_distance
  have h : speed1 * time + v * time = total_distance := by sorry
  
  -- Prove that v = 50 given the above equation
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3035_303515


namespace NUMINAMATH_CALUDE_heating_plant_consumption_l3035_303578

/-- Represents the fuel consumption of a heating plant -/
structure HeatingPlant where
  consumption_rate : ℝ  -- Liters per hour

/-- Given a heating plant that consumes 7 liters of fuel in 21 hours,
    prove that it will consume 30 liters of fuel in 90 hours -/
theorem heating_plant_consumption 
  (plant : HeatingPlant) 
  (h1 : plant.consumption_rate * 21 = 7) :
  plant.consumption_rate * 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_heating_plant_consumption_l3035_303578


namespace NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l3035_303579

theorem x_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, x^2 + x - 2 = 0 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x^2 + x - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l3035_303579


namespace NUMINAMATH_CALUDE_hannah_grocery_cost_l3035_303532

theorem hannah_grocery_cost 
  (total_cost : ℝ)
  (cookie_price : ℝ)
  (carrot_price : ℝ)
  (cabbage_price : ℝ)
  (orange_price : ℝ)
  (h1 : cookie_price + carrot_price + cabbage_price + orange_price = total_cost)
  (h2 : orange_price = 3 * cookie_price)
  (h3 : cabbage_price = cookie_price - carrot_price)
  (h4 : total_cost = 24) :
  carrot_price + cabbage_price = 24 / 5 := by
sorry

end NUMINAMATH_CALUDE_hannah_grocery_cost_l3035_303532


namespace NUMINAMATH_CALUDE_unique_digit_divisibility_l3035_303501

theorem unique_digit_divisibility : 
  ∃! (A : ℕ), A < 10 ∧ 70 % A = 0 ∧ (546200 + 10 * A + 4) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_divisibility_l3035_303501


namespace NUMINAMATH_CALUDE_max_profit_children_clothing_l3035_303573

/-- Profit function for children's clothing sales -/
def profit (x : ℝ) : ℝ :=
  (x - 30) * (-2 * x + 200) - 450

/-- Theorem: Maximum profit for children's clothing sales -/
theorem max_profit_children_clothing :
  let x_min : ℝ := 30
  let x_max : ℝ := 60
  ∀ x ∈ Set.Icc x_min x_max,
    profit x ≤ profit x_max ∧
    profit x_max = 1950 := by
  sorry

#check max_profit_children_clothing

end NUMINAMATH_CALUDE_max_profit_children_clothing_l3035_303573


namespace NUMINAMATH_CALUDE_xenia_earnings_l3035_303590

/-- Xenia's work and earnings over two weeks -/
theorem xenia_earnings 
  (hours_week1 : ℕ) 
  (hours_week2 : ℕ) 
  (wage : ℚ) 
  (extra_earnings : ℚ) 
  (h1 : hours_week1 = 12)
  (h2 : hours_week2 = 20)
  (h3 : extra_earnings = 36)
  (h4 : wage * (hours_week2 - hours_week1) = extra_earnings) :
  wage * (hours_week1 + hours_week2) = 144 :=
sorry

end NUMINAMATH_CALUDE_xenia_earnings_l3035_303590


namespace NUMINAMATH_CALUDE_solve_equation_and_calculate_l3035_303566

theorem solve_equation_and_calculate (x : ℝ) :
  Real.sqrt ((3 / x) + 1) = 4 / 3 →
  x = 27 / 7 ∧ x + 6 = 69 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_calculate_l3035_303566


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l3035_303586

/-- The surface area of a cube inscribed in a sphere of radius 2 -/
theorem inscribed_cube_surface_area (r : ℝ) (a : ℝ) : 
  r = 2 →  -- The radius of the sphere is 2
  3 * a^2 = (2*r)^2 →  -- The cube's diagonal equals the sphere's diameter
  6 * a^2 = 32 :=  -- The surface area of the cube is 32
by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l3035_303586


namespace NUMINAMATH_CALUDE_floor_sqrt_245_l3035_303577

theorem floor_sqrt_245 : ⌊Real.sqrt 245⌋ = 15 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_245_l3035_303577


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3035_303543

theorem polynomial_evaluation : 7^4 - 4 * 7^3 + 6 * 7^2 - 5 * 7 + 3 = 1553 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3035_303543


namespace NUMINAMATH_CALUDE_valentine_cards_theorem_l3035_303570

theorem valentine_cards_theorem (x y : ℕ) : 
  x * y = x + y + 30 → x * y = 64 := by
  sorry

end NUMINAMATH_CALUDE_valentine_cards_theorem_l3035_303570


namespace NUMINAMATH_CALUDE_circumcenter_outside_l3035_303595

/-- An isosceles trapezoid with specific angle measurements -/
structure IsoscelesTrapezoid where
  /-- The angle at the base of the trapezoid -/
  base_angle : ℝ
  /-- The angle between the diagonals adjacent to the lateral side -/
  diagonal_angle : ℝ
  /-- Condition that the base angle is 50 degrees -/
  base_angle_eq : base_angle = 50
  /-- Condition that the diagonal angle is 40 degrees -/
  diagonal_angle_eq : diagonal_angle = 40

/-- The location of the circumcenter relative to the trapezoid -/
inductive CircumcenterLocation
  | Inside
  | Outside

/-- Theorem stating that the circumcenter is outside the trapezoid -/
theorem circumcenter_outside (t : IsoscelesTrapezoid) : 
  CircumcenterLocation.Outside = 
    CircumcenterLocation.Outside := by sorry

end NUMINAMATH_CALUDE_circumcenter_outside_l3035_303595


namespace NUMINAMATH_CALUDE_chess_tournament_impossibility_l3035_303581

theorem chess_tournament_impossibility (n : ℕ) (g : ℕ) (x : ℕ) : 
  n = 50 →  -- Total number of players
  g = 61 →  -- Total number of games played
  x ≤ n →   -- Number of players who played 3 games
  (3 * x + 2 * (n - x)) / 2 = g →  -- Total games calculation
  x * 3 > g →  -- Contradiction: games played by 3-game players exceed total games
  False :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_impossibility_l3035_303581


namespace NUMINAMATH_CALUDE_points_for_level_completion_l3035_303565

/-- Given a game scenario, prove the points earned for completing the level -/
theorem points_for_level_completion 
  (enemies_defeated : ℕ) 
  (points_per_enemy : ℕ) 
  (total_points : ℕ) 
  (h1 : enemies_defeated = 6)
  (h2 : points_per_enemy = 9)
  (h3 : total_points = 62) :
  total_points - (enemies_defeated * points_per_enemy) = 8 :=
by sorry

end NUMINAMATH_CALUDE_points_for_level_completion_l3035_303565


namespace NUMINAMATH_CALUDE_quadratic_b_value_l3035_303517

/-- Given a quadratic function y = ax² + bx + c, prove that b = 3 when
    (2, y₁) and (-2, y₂) are points on the graph and y₁ - y₂ = 12 -/
theorem quadratic_b_value (a c y₁ y₂ : ℝ) :
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = 12 →
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_b_value_l3035_303517


namespace NUMINAMATH_CALUDE_unique_solution_l3035_303559

/-- Represents the number of correct answers for each friend -/
structure ExamResults :=
  (A B C D : Nat)

/-- Checks if the given results satisfy the conditions of the problem -/
def satisfiesConditions (results : ExamResults) : Prop :=
  -- Total correct answers is 6
  results.A + results.B + results.C + results.D = 6 ∧
  -- Each result is between 0 and 3
  results.A ≤ 3 ∧ results.B ≤ 3 ∧ results.C ≤ 3 ∧ results.D ≤ 3 ∧
  -- Number of true statements matches correct answers
  (results.A = 1 ∨ results.A = 2) ∧
  (results.B = 0 ∨ results.B = 2) ∧
  (results.C = 0 ∨ results.C = 1) ∧
  (results.D = 0 ∨ results.D = 3) ∧
  -- Relative performance statements
  (results.A > results.B → results.A = 2) ∧
  (results.C < results.D → results.A = 2) ∧
  (results.C = 0 → results.B = 3) ∧
  (results.A < results.D → results.B = 3) ∧
  (results.D = 2 → results.C = 1) ∧
  (results.B < results.A → results.C = 1) ∧
  (results.C < results.D → results.D = 3) ∧
  (results.A < results.B → results.D = 3)

theorem unique_solution :
  ∃! results : ExamResults, satisfiesConditions results ∧ 
    results.A = 1 ∧ results.B = 2 ∧ results.C = 0 ∧ results.D = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3035_303559


namespace NUMINAMATH_CALUDE_log_length_l3035_303554

/-- Represents the properties of a log that has been cut in half -/
structure LogCut where
  weight_per_foot : ℝ
  weight_of_piece : ℝ
  original_length : ℝ

/-- Theorem stating that given the conditions, the original log length is 20 feet -/
theorem log_length (log : LogCut) 
  (h1 : log.weight_per_foot = 150)
  (h2 : log.weight_of_piece = 1500) :
  log.original_length = 20 := by
  sorry

#check log_length

end NUMINAMATH_CALUDE_log_length_l3035_303554


namespace NUMINAMATH_CALUDE_average_xyz_is_five_sixths_l3035_303589

theorem average_xyz_is_five_sixths (x y z : ℚ) 
  (eq1 : 2003 * z - 4006 * x = 1002)
  (eq2 : 2003 * y + 6009 * x = 4004) :
  (x + y + z) / 3 = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_average_xyz_is_five_sixths_l3035_303589


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_l3035_303563

theorem rectangular_box_dimensions (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A * B = 50 →
  A * C = 90 →
  B * C = 100 →
  A + B + C = 24 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_l3035_303563


namespace NUMINAMATH_CALUDE_prob_two_target_rolls_l3035_303542

/-- The number of sides on each die -/
def num_sides : ℕ := 7

/-- The sum we're aiming for -/
def target_sum : ℕ := 8

/-- The set of all possible outcomes when rolling two dice -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_sides) (Finset.range num_sides)

/-- The set of outcomes that sum to the target -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun (a, b) => a + b + 2 = target_sum)

/-- The probability of rolling the target sum once -/
def prob_target : ℚ :=
  (favorable_outcomes.card : ℚ) / (all_outcomes.card : ℚ)

theorem prob_two_target_rolls : prob_target * prob_target = 1 / 49 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_target_rolls_l3035_303542


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l3035_303518

theorem sqrt_difference_equality : Real.sqrt 27 - Real.sqrt (1/3) = (8/3) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l3035_303518


namespace NUMINAMATH_CALUDE_no_solution_exists_l3035_303533

theorem no_solution_exists : ¬∃ (x : ℤ), x^2 = 3*x + 75 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3035_303533


namespace NUMINAMATH_CALUDE_gcd_lcm_inequality_implies_divisibility_l3035_303510

theorem gcd_lcm_inequality_implies_divisibility (a b : ℕ) 
  (h : a * Nat.gcd a b + b * Nat.lcm a b < (5/2) * a * b) : 
  b ∣ a := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_inequality_implies_divisibility_l3035_303510


namespace NUMINAMATH_CALUDE_johns_allowance_is_150_cents_l3035_303513

def johns_allowance (A : ℚ) : Prop :=
  let arcade_spent : ℚ := 3 / 5 * A
  let remaining_after_arcade : ℚ := A - arcade_spent
  let toy_store_spent : ℚ := 1 / 3 * remaining_after_arcade
  let remaining_after_toy_store : ℚ := remaining_after_arcade - toy_store_spent
  remaining_after_toy_store = 40 / 100

theorem johns_allowance_is_150_cents :
  ∃ A : ℚ, johns_allowance A ∧ A = 150 / 100 :=
sorry

end NUMINAMATH_CALUDE_johns_allowance_is_150_cents_l3035_303513


namespace NUMINAMATH_CALUDE_piper_gym_schedule_theorem_l3035_303537

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns true if Piper goes to gym on the given day -/
def goes_to_gym (day : DayOfWeek) : Bool :=
  match day with
  | DayOfWeek.Sunday => false
  | DayOfWeek.Monday => true
  | DayOfWeek.Tuesday => false
  | DayOfWeek.Wednesday => true
  | DayOfWeek.Thursday => false
  | DayOfWeek.Friday => true
  | DayOfWeek.Saturday => true

/-- Returns the next day of the week -/
def next_day (day : DayOfWeek) : DayOfWeek :=
  match day with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Returns the day after n days from the given start day -/
def day_after (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n + 1 => day_after (next_day start) n

/-- Counts the number of gym sessions from the start day up to n days -/
def count_sessions (start : DayOfWeek) (n : Nat) : Nat :=
  match n with
  | 0 => if goes_to_gym start then 1 else 0
  | n + 1 => count_sessions start n + if goes_to_gym (day_after start n) then 1 else 0

theorem piper_gym_schedule_theorem :
  ∃ (n : Nat), count_sessions DayOfWeek.Monday n = 35 ∧ 
               day_after DayOfWeek.Monday n = DayOfWeek.Monday :=
  sorry


end NUMINAMATH_CALUDE_piper_gym_schedule_theorem_l3035_303537


namespace NUMINAMATH_CALUDE_fraction_division_problem_solution_l3035_303504

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem problem_solution : (5 / 3) / (8 / 15) = 25 / 8 :=
by
  -- Apply the fraction division theorem
  have h1 : (5 / 3) / (8 / 15) = (5 * 15) / (3 * 8) := by sorry
  
  -- Simplify the numerator and denominator
  have h2 : (5 * 15) / (3 * 8) = 75 / 24 := by sorry
  
  -- Further simplify the fraction
  have h3 : 75 / 24 = 25 / 8 := by sorry
  
  -- Combine the steps
  sorry

end NUMINAMATH_CALUDE_fraction_division_problem_solution_l3035_303504


namespace NUMINAMATH_CALUDE_harmonic_sum_equality_l3035_303575

/-- The nth harmonic number -/
def h (n : ℕ+) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

/-- The sum of harmonic numbers up to n-1 -/
def sum_h (n : ℕ+) : ℚ :=
  (Finset.range (n - 1)).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem: n + sum of h(1) to h(n-1) equals n * h(n) for n ≥ 2 -/
theorem harmonic_sum_equality (n : ℕ+) (hn : n ≥ 2) :
  (n : ℚ) + sum_h n = n * h n := by sorry

end NUMINAMATH_CALUDE_harmonic_sum_equality_l3035_303575


namespace NUMINAMATH_CALUDE_field_trip_van_occupancy_l3035_303505

theorem field_trip_van_occupancy :
  let num_vans : ℕ := 2
  let num_buses : ℕ := 3
  let people_per_bus : ℕ := 20
  let total_people : ℕ := 76
  let people_in_vans : ℕ := total_people - (num_buses * people_per_bus)
  people_in_vans / num_vans = 8 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_van_occupancy_l3035_303505


namespace NUMINAMATH_CALUDE_solution_set_of_trigonometric_system_l3035_303550

theorem solution_set_of_trigonometric_system :
  let S := {(x, y) | 
    2 * (Real.cos x)^2 + 2 * Real.sqrt 2 * Real.cos x * (Real.cos (4*x))^2 + (Real.cos (4*x))^2 = 0 ∧
    Real.sin x = Real.cos y}
  S = {(x, y) | 
    (∃ k n : ℤ, x = 3 * Real.pi / 4 + 2 * Real.pi * ↑k ∧ (y = Real.pi / 4 + 2 * Real.pi * ↑n ∨ y = -Real.pi / 4 + 2 * Real.pi * ↑n)) ∨
    (∃ k n : ℤ, x = -3 * Real.pi / 4 + 2 * Real.pi * ↑k ∧ (y = 3 * Real.pi / 4 + 2 * Real.pi * ↑n ∨ y = -3 * Real.pi / 4 + 2 * Real.pi * ↑n))} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_trigonometric_system_l3035_303550


namespace NUMINAMATH_CALUDE_composite_cubes_surface_area_l3035_303514

/-- Represents a composite shape formed by two cubes -/
structure CompositeCubes where
  large_cube_edge : ℝ
  small_cube_edge : ℝ

/-- Calculate the surface area of the composite shape -/
def surface_area (shape : CompositeCubes) : ℝ :=
  let large_cube_area := 6 * shape.large_cube_edge ^ 2
  let small_cube_area := 6 * shape.small_cube_edge ^ 2
  let covered_area := shape.small_cube_edge ^ 2
  let exposed_small_cube_area := 4 * shape.small_cube_edge ^ 2
  large_cube_area - covered_area + exposed_small_cube_area

/-- Theorem stating that the surface area of the specific composite shape is 49 -/
theorem composite_cubes_surface_area : 
  let shape := CompositeCubes.mk 3 1
  surface_area shape = 49 := by
  sorry

end NUMINAMATH_CALUDE_composite_cubes_surface_area_l3035_303514


namespace NUMINAMATH_CALUDE_no_infinite_sequence_exists_l3035_303576

theorem no_infinite_sequence_exists : ¬ ∃ (k : ℕ → ℝ), 
  (∀ n : ℕ, k (n + 1) = k n - 1 / k n) ∧ 
  (∀ n : ℕ, k n * k (n + 1) ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_exists_l3035_303576


namespace NUMINAMATH_CALUDE_smallest_n_for_monochromatic_isosceles_trapezoid_l3035_303535

/-- A coloring of vertices with three colors -/
def Coloring (n : ℕ) := Fin n → Fin 3

/-- Check if four vertices form an isosceles trapezoid in an n-gon -/
def IsIsoscelesTrapezoid (n : ℕ) (v1 v2 v3 v4 : Fin n) : Prop := sorry

/-- Check if a coloring contains four vertices of the same color forming an isosceles trapezoid -/
def HasMonochromaticIsoscelesTrapezoid (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (v1 v2 v3 v4 : Fin n), 
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4 ∧
    c v1 = c v2 ∧ c v1 = c v3 ∧ c v1 = c v4 ∧
    IsIsoscelesTrapezoid n v1 v2 v3 v4

/-- The main theorem -/
theorem smallest_n_for_monochromatic_isosceles_trapezoid :
  (∀ (c : Coloring 17), HasMonochromaticIsoscelesTrapezoid 17 c) ∧
  (∀ (n : ℕ), n < 17 → ∃ (c : Coloring n), ¬HasMonochromaticIsoscelesTrapezoid n c) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_monochromatic_isosceles_trapezoid_l3035_303535


namespace NUMINAMATH_CALUDE_special_function_ratio_bounds_l3035_303536

open Real

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioi 0
  pos : ∀ x ∈ domain, f x > 0
  deriv_bound : ∀ x ∈ domain, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x

/-- The main theorem -/
theorem special_function_ratio_bounds (sf : SpecialFunction) :
    1/8 < sf.f 1 / sf.f 2 ∧ sf.f 1 / sf.f 2 < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_special_function_ratio_bounds_l3035_303536


namespace NUMINAMATH_CALUDE_range_of_m_l3035_303588

def p (m : ℝ) : Prop := ∃ (x y : ℝ), x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ < 0 ∧ x₁^2 - x₁ + m - 4 = 0 ∧ x₂^2 - x₂ + m - 4 = 0

theorem range_of_m (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬p m) :
  m ≤ 1 - Real.sqrt 2 ∨ (1 + Real.sqrt 2 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3035_303588


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3035_303522

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  S seq 3 = seq.a 2 + 10 * seq.a 1 →
  seq.a 5 = 9 →
  seq.a 1 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3035_303522


namespace NUMINAMATH_CALUDE_project_wage_difference_l3035_303516

theorem project_wage_difference (total_pay : ℝ) (p_hours q_hours : ℝ) 
  (hp : total_pay = 420)
  (hpq : q_hours = p_hours + 10)
  (hw : p_hours * (1.5 * (total_pay / q_hours)) = total_pay) :
  1.5 * (total_pay / q_hours) - (total_pay / q_hours) = 7 := by
  sorry

end NUMINAMATH_CALUDE_project_wage_difference_l3035_303516


namespace NUMINAMATH_CALUDE_largest_non_expressible_l3035_303584

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ a b, a > 0 ∧ is_composite b ∧ n = 36 * a + b

theorem largest_non_expressible : 
  (∀ n > 188, is_expressible n) ∧ ¬is_expressible 188 :=
sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l3035_303584


namespace NUMINAMATH_CALUDE_triangle_properties_l3035_303526

/-- Proves the properties of an acute triangle ABC with given conditions -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π / 2 →  -- A is acute
  0 < B ∧ B < π / 2 →  -- B is acute
  0 < C ∧ C < π / 2 →  -- C is acute
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  a = Real.sqrt 21 →
  b = 5 →
  A + B + C = π →  -- Sum of angles in a triangle
  a * Real.sin B = b * Real.sin A →  -- Sine law
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →  -- Cosine law
  A = π / 3 ∧ c = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3035_303526


namespace NUMINAMATH_CALUDE_equation_a_is_quadratic_l3035_303531

-- Define what a quadratic equation is
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific equation we want to prove is quadratic
def f (x : ℝ) : ℝ := x^2 + 2

-- Theorem statement
theorem equation_a_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_a_is_quadratic_l3035_303531


namespace NUMINAMATH_CALUDE_maintenance_time_is_three_minutes_l3035_303521

/-- Represents the passage scenario with order maintenance --/
structure PassageScenario where
  normal_rate : ℕ
  congested_rate : ℕ
  people_waiting : ℕ
  time_saved : ℕ

/-- Calculates the time spent maintaining order --/
def maintenance_time (scenario : PassageScenario) : ℕ :=
  let total_wait_time := scenario.people_waiting / scenario.congested_rate
  let actual_wait_time := total_wait_time - scenario.time_saved
  actual_wait_time

/-- Theorem stating that the maintenance time is 3 minutes for the given scenario --/
theorem maintenance_time_is_three_minutes 
  (scenario : PassageScenario)
  (h1 : scenario.normal_rate = 9)
  (h2 : scenario.congested_rate = 3)
  (h3 : scenario.people_waiting = 36)
  (h4 : scenario.time_saved = 6) :
  maintenance_time scenario = 3 := by
  sorry

#eval maintenance_time { normal_rate := 9, congested_rate := 3, people_waiting := 36, time_saved := 6 }

end NUMINAMATH_CALUDE_maintenance_time_is_three_minutes_l3035_303521


namespace NUMINAMATH_CALUDE_distribute_five_four_l3035_303502

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of partitions of n into at most k parts -/
def partitions (n k : ℕ) : ℕ := sorry

theorem distribute_five_four : distribute 5 4 = 6 := by sorry

end NUMINAMATH_CALUDE_distribute_five_four_l3035_303502


namespace NUMINAMATH_CALUDE_extrema_sum_implies_a_range_l3035_303574

/-- Given a function f(x) = ax - x^2 - ln x, if f(x) has extrema and the sum of these extrema
    is not less than 4 + ln 2, then a ∈ [2√3, +∞). -/
theorem extrema_sum_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (∀ x > 0, a * x - x^2 - Real.log x ≤ max (a * x₁ - x₁^2 - Real.log x₁) (a * x₂ - x₂^2 - Real.log x₂)) ∧
    (a * x₁ - x₁^2 - Real.log x₁) + (a * x₂ - x₂^2 - Real.log x₂) ≥ 4 + Real.log 2) →
  a ≥ 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_extrema_sum_implies_a_range_l3035_303574


namespace NUMINAMATH_CALUDE_total_three_digit_numbers_l3035_303552

/-- Represents a card with two numbers -/
structure Card where
  side1 : Nat
  side2 : Nat
  different : side1 ≠ side2

/-- The set of cards given in the problem -/
def problemCards : Finset Card := sorry

/-- The number of ways to arrange 3 cards -/
def cardArrangements : Nat := sorry

/-- The number of ways to choose sides for 3 cards -/
def sideChoices : Nat := sorry

/-- Theorem stating the total number of different three-digit numbers -/
theorem total_three_digit_numbers : 
  cardArrangements * sideChoices = 48 := by sorry

end NUMINAMATH_CALUDE_total_three_digit_numbers_l3035_303552


namespace NUMINAMATH_CALUDE_images_per_card_l3035_303555

/-- The number of pictures John takes per day -/
def pictures_per_day : ℕ := 10

/-- The number of years John has been taking pictures -/
def years : ℕ := 3

/-- The cost of each memory card in dollars -/
def cost_per_card : ℕ := 60

/-- The total amount John spent on memory cards in dollars -/
def total_spent : ℕ := 13140

/-- The number of days in a year (assuming no leap years) -/
def days_per_year : ℕ := 365

theorem images_per_card : 
  (years * days_per_year * pictures_per_day) / (total_spent / cost_per_card) = 50 := by
  sorry

end NUMINAMATH_CALUDE_images_per_card_l3035_303555


namespace NUMINAMATH_CALUDE_area_inside_rectangle_outside_circles_l3035_303553

/-- The area of the region inside a rectangle but outside three quarter circles --/
theorem area_inside_rectangle_outside_circles (π : ℝ) :
  let rectangle_area : ℝ := 4 * 6
  let circle_e_area : ℝ := π * 2^2
  let circle_f_area : ℝ := π * 3^2
  let circle_g_area : ℝ := π * 4^2
  let quarter_circles_area : ℝ := (circle_e_area + circle_f_area + circle_g_area) / 4
  rectangle_area - quarter_circles_area = 24 - (29 * π) / 4 :=
by sorry

end NUMINAMATH_CALUDE_area_inside_rectangle_outside_circles_l3035_303553


namespace NUMINAMATH_CALUDE_problem_solution_l3035_303545

-- Define the expression as a function of a, b, and x
def expression (a b x : ℝ) : ℝ := (a * x^2 + b * x + 2) - (5 * x^2 + 3 * x)

theorem problem_solution :
  (∀ x, expression 7 (-1) x = 2 * x^2 - 4 * x + 2) ∧
  (∀ x, expression 5 (-3) x = -6 * x + 2) ∧
  (∃ a b, ∀ x, expression a b x = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3035_303545


namespace NUMINAMATH_CALUDE_dummies_remainder_l3035_303530

theorem dummies_remainder (n : ℕ) (h : n % 9 = 7) : (3 * n) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dummies_remainder_l3035_303530


namespace NUMINAMATH_CALUDE_largest_integer_for_binary_op_l3035_303512

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem largest_integer_for_binary_op :
  ∃ m : ℤ, m = -19 ∧
  (∀ n : ℤ, n > 0 → binary_op n < m → n ≤ 5) ∧
  (∀ m' : ℤ, m' > m → ∃ n : ℤ, n > 0 ∧ n > 5 ∧ binary_op n < m') :=
sorry

end NUMINAMATH_CALUDE_largest_integer_for_binary_op_l3035_303512


namespace NUMINAMATH_CALUDE_scientific_notation_of_40_9_billion_l3035_303538

theorem scientific_notation_of_40_9_billion :
  (40.9 : ℝ) * 1000000000 = 4.09 * (10 : ℝ)^9 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_40_9_billion_l3035_303538


namespace NUMINAMATH_CALUDE_max_servings_emily_l3035_303534

/-- Represents the recipe requirements for 4 servings --/
structure Recipe :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (berries : ℕ)
  (almond_milk : ℕ)

/-- Represents Emily's available ingredients --/
structure Available :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (berries : ℕ)
  (almond_milk : ℕ)

/-- Calculates the maximum number of servings possible --/
def max_servings (recipe : Recipe) (available : Available) : ℕ :=
  min
    (available.bananas * 4 / recipe.bananas)
    (min
      (available.yogurt * 4 / recipe.yogurt)
      (min
        (available.berries * 4 / recipe.berries)
        (available.almond_milk * 4 / recipe.almond_milk)))

/-- The theorem to be proved --/
theorem max_servings_emily :
  let recipe := Recipe.mk 3 2 1 1
  let available := Available.mk 9 5 3 4
  max_servings recipe available = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_emily_l3035_303534


namespace NUMINAMATH_CALUDE_male_students_in_school_l3035_303529

/-- Represents the number of students in a school population --/
structure SchoolPopulation where
  total : Nat
  sample : Nat
  females_in_sample : Nat

/-- Calculates the number of male students in the school --/
def male_students (pop : SchoolPopulation) : Nat :=
  pop.total - (pop.total * pop.females_in_sample / pop.sample)

/-- Theorem stating the number of male students in the given scenario --/
theorem male_students_in_school (pop : SchoolPopulation) 
  (h1 : pop.total = 1600)
  (h2 : pop.sample = 200)
  (h3 : pop.females_in_sample = 95) :
  male_students pop = 840 := by
  sorry

#eval male_students { total := 1600, sample := 200, females_in_sample := 95 }

end NUMINAMATH_CALUDE_male_students_in_school_l3035_303529


namespace NUMINAMATH_CALUDE_tiles_crossed_specific_floor_l3035_303546

/-- Represents a rectangular floor -/
structure Floor :=
  (width : ℕ) (length : ℕ)

/-- Represents a rectangular tile -/
structure Tile :=
  (width : ℕ) (length : ℕ)

/-- Counts the number of tiles crossed by a diagonal line on a floor -/
def tilesCrossedByDiagonal (f : Floor) (t : Tile) : ℕ :=
  f.width + f.length - Nat.gcd f.width f.length

theorem tiles_crossed_specific_floor :
  let floor := Floor.mk 12 19
  let tile := Tile.mk 1 2
  tilesCrossedByDiagonal floor tile = 30 := by
  sorry

#eval tilesCrossedByDiagonal (Floor.mk 12 19) (Tile.mk 1 2)

end NUMINAMATH_CALUDE_tiles_crossed_specific_floor_l3035_303546


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3035_303585

/-- Given a hyperbola and a circle, prove the equations of the hyperbola's asymptotes -/
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ (x y : ℝ), x^2 / 9 - y^2 / m = 1 ∧ x^2 + y^2 - 4*x - 5 = 0) →
  (∃ (k : ℝ), k = 4/3 ∧ 
    (∀ (x y : ℝ), (x^2 / 9 - y^2 / m = 1) → (y = k*x ∨ y = -k*x))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3035_303585


namespace NUMINAMATH_CALUDE_inverse_f_123_l3035_303597

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 + 6

theorem inverse_f_123 : f⁻¹ 123 = (39 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_inverse_f_123_l3035_303597


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_square_l3035_303557

theorem sqrt_equality_implies_square (x : ℝ) : 
  Real.sqrt (3 * x + 5) = 5 → (3 * x + 5)^2 = 625 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_square_l3035_303557


namespace NUMINAMATH_CALUDE_x_value_proof_l3035_303519

theorem x_value_proof (x : ℝ) (h : (1/2 : ℝ) - (1/3 : ℝ) = 3/x) : x = 18 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3035_303519


namespace NUMINAMATH_CALUDE_max_tuesday_13ths_l3035_303524

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents the months of the year -/
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August
| September
| October
| November
| December

/-- Returns the number of days in a given month -/
def daysInMonth (m : Month) : Nat :=
  match m with
  | .February => 28
  | .April | .June | .September | .November => 30
  | _ => 31

/-- Returns the day of the week for the 13th of a given month, 
    given the day of the week for January 13th -/
def dayOf13th (m : Month) (jan13 : DayOfWeek) : DayOfWeek :=
  sorry

/-- Counts the number of times the 13th falls on a Tuesday in a year -/
def countTuesday13ths (jan13 : DayOfWeek) : Nat :=
  sorry

theorem max_tuesday_13ths :
  ∃ (jan13 : DayOfWeek), countTuesday13ths jan13 = 3 ∧
  ∀ (d : DayOfWeek), countTuesday13ths d ≤ 3 :=
  sorry

end NUMINAMATH_CALUDE_max_tuesday_13ths_l3035_303524


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3035_303599

theorem floor_plus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ + s = 20.5 :=
by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3035_303599


namespace NUMINAMATH_CALUDE_max_volume_container_l3035_303541

/-- Represents the dimensions of a rectangular container --/
structure ContainerDimensions where
  length : Real
  width : Real
  height : Real

/-- Calculates the volume of a rectangular container --/
def volume (d : ContainerDimensions) : Real :=
  d.length * d.width * d.height

/-- Represents the constraint of the total length of the steel bar --/
def totalLength (d : ContainerDimensions) : Real :=
  2 * (d.length + d.width) + 4 * d.height

/-- Theorem stating the maximum volume and corresponding height --/
theorem max_volume_container :
  ∃ (d : ContainerDimensions),
    totalLength d = 14.8 ∧
    d.length = d.width + 0.5 ∧
    d.height = 1.2 ∧
    volume d = 2.2 ∧
    ∀ (d' : ContainerDimensions),
      totalLength d' = 14.8 ∧ d'.length = d'.width + 0.5 →
      volume d' ≤ volume d :=
by sorry

end NUMINAMATH_CALUDE_max_volume_container_l3035_303541


namespace NUMINAMATH_CALUDE_subset_implies_a_value_l3035_303500

theorem subset_implies_a_value (A B : Set ℤ) (a : ℤ) 
  (h1 : A = {0, 1}) 
  (h2 : B = {-1, 0, a+3}) 
  (h3 : A ⊆ B) : 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_value_l3035_303500


namespace NUMINAMATH_CALUDE_green_ball_probability_l3035_303598

-- Define the containers and their contents
def container_A : Nat × Nat := (3, 7)  -- (red, green)
def container_B : Nat × Nat := (5, 5)
def container_C : Nat × Nat := (5, 5)

-- Define the probability of selecting each container
def container_prob : Rat := 1/3

-- Define the probability of selecting a green ball from each container
def green_prob_A : Rat := container_prob * (container_A.2 / (container_A.1 + container_A.2))
def green_prob_B : Rat := container_prob * (container_B.2 / (container_B.1 + container_B.2))
def green_prob_C : Rat := container_prob * (container_C.2 / (container_C.1 + container_C.2))

-- Theorem: The probability of selecting a green ball is 17/30
theorem green_ball_probability : 
  green_prob_A + green_prob_B + green_prob_C = 17/30 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l3035_303598


namespace NUMINAMATH_CALUDE_identity_function_l3035_303525

theorem identity_function (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) : 
  ∀ n : ℕ, f n = n := by
sorry

end NUMINAMATH_CALUDE_identity_function_l3035_303525


namespace NUMINAMATH_CALUDE_john_star_wars_toys_cost_l3035_303561

/-- The total cost of John's Star Wars toys, including the lightsaber -/
def total_cost (other_toys_cost lightsaber_cost : ℕ) : ℕ :=
  other_toys_cost + lightsaber_cost

/-- The cost of the lightsaber -/
def lightsaber_cost (other_toys_cost : ℕ) : ℕ :=
  2 * other_toys_cost

theorem john_star_wars_toys_cost (other_toys_cost : ℕ) 
  (h : other_toys_cost = 1000) : 
  total_cost other_toys_cost (lightsaber_cost other_toys_cost) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_john_star_wars_toys_cost_l3035_303561


namespace NUMINAMATH_CALUDE_distance_scientific_notation_l3035_303539

/-- The distance from the Chinese space station to the apogee of the Earth in meters -/
def distance : ℝ := 347000

/-- The coefficient in the scientific notation representation -/
def coefficient : ℝ := 3.47

/-- The exponent in the scientific notation representation -/
def exponent : ℕ := 5

/-- Theorem stating that the distance is equal to its scientific notation representation -/
theorem distance_scientific_notation : distance = coefficient * (10 ^ exponent) := by
  sorry

end NUMINAMATH_CALUDE_distance_scientific_notation_l3035_303539


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3035_303548

/-- A polynomial of degree 5 with coefficients in ℝ -/
def polynomial (p q : ℝ) (x : ℝ) : ℝ :=
  x^5 - x^4 + x^3 - p*x^2 + q*x + 9

/-- The condition that the polynomial is divisible by (x + 3)(x - 2) -/
def is_divisible (p q : ℝ) : Prop :=
  ∀ x : ℝ, (x + 3 = 0 ∨ x - 2 = 0) → polynomial p q x = 0

/-- The main theorem stating that if the polynomial is divisible by (x + 3)(x - 2),
    then p = -130.5 and q = -277.5 -/
theorem polynomial_divisibility (p q : ℝ) :
  is_divisible p q → p = -130.5 ∧ q = -277.5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3035_303548


namespace NUMINAMATH_CALUDE_jogger_distance_ahead_l3035_303572

def jogger_speed : ℝ := 9 -- km/hr
def train_speed : ℝ := 45 -- km/hr
def train_length : ℝ := 210 -- meters
def passing_time : ℝ := 41 -- seconds

theorem jogger_distance_ahead (jogger_speed train_speed train_length passing_time : ℝ) :
  jogger_speed = 9 ∧ 
  train_speed = 45 ∧ 
  train_length = 210 ∧ 
  passing_time = 41 →
  (train_speed - jogger_speed) * passing_time / 3600 * 1000 - train_length = 200 := by
  sorry

end NUMINAMATH_CALUDE_jogger_distance_ahead_l3035_303572


namespace NUMINAMATH_CALUDE_average_temperature_of_three_cities_l3035_303587

/-- Proves that the average temperature of three cities is 95 degrees given specific temperature relationships --/
theorem average_temperature_of_three_cities
  (temp_new_york : ℝ)
  (h1 : temp_new_york = 80)
  (temp_miami : ℝ)
  (h2 : temp_miami = temp_new_york + 10)
  (temp_san_diego : ℝ)
  (h3 : temp_san_diego = temp_miami + 25) :
  (temp_new_york + temp_miami + temp_san_diego) / 3 = 95 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_of_three_cities_l3035_303587


namespace NUMINAMATH_CALUDE_dress_design_combinations_l3035_303558

theorem dress_design_combinations (num_colors num_patterns : ℕ) 
  (h_colors : num_colors = 5)
  (h_patterns : num_patterns = 6) :
  num_colors * num_patterns = 30 := by
sorry

end NUMINAMATH_CALUDE_dress_design_combinations_l3035_303558


namespace NUMINAMATH_CALUDE_expression_result_l3035_303571

theorem expression_result : (7.5 * 7.5 + 37.5 + 2.5 * 2.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l3035_303571


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l3035_303506

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 →
  a * b + b * c + c * d + d * a ≤ 10000 ∧ 
  ∃ (a' b' c' d' : ℝ), a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ d' ≥ 0 ∧ 
    a' + b' + c' + d' = 200 ∧
    a' * b' + b' * c' + c' * d' + d' * a' = 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l3035_303506


namespace NUMINAMATH_CALUDE_aquarium_water_after_45_days_l3035_303591

/-- Calculates the remaining water in an aquarium after a given time period. -/
def remainingWater (initialVolume : ℝ) (lossRate : ℝ) (days : ℝ) : ℝ :=
  initialVolume - lossRate * days

/-- Theorem stating the remaining water volume in the aquarium after 45 days. -/
theorem aquarium_water_after_45_days :
  remainingWater 500 1.2 45 = 446 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_water_after_45_days_l3035_303591


namespace NUMINAMATH_CALUDE_equation_of_line_l_equations_of_line_m_l3035_303583

-- Define the slope of line l
def slope_l : ℚ := -3/4

-- Define the equation of the line that point P is on
def line_p (k : ℚ) (x y : ℝ) : Prop := k * x - y + 2 * k + 5 = 0

-- Define point P
def point_p : ℝ × ℝ := (-2, 5)

-- Define the distance from point P to line m
def distance_p_to_m : ℝ := 3

-- Theorem for the equation of line l
theorem equation_of_line_l :
  ∃ (A B C : ℝ), A * point_p.1 + B * point_p.2 + C = 0 ∧
  B ≠ 0 ∧ -A/B = slope_l ∧
  ∀ (x y : ℝ), A * x + B * y + C = 0 ↔ y = slope_l * x + (point_p.2 - slope_l * point_p.1) :=
sorry

-- Theorem for the equations of line m
theorem equations_of_line_m :
  ∃ (b₁ b₂ : ℝ), 
    (∀ (x y : ℝ), y = slope_l * x + b₁ ↔ 
      distance_p_to_m = |slope_l * point_p.1 - point_p.2 + b₁| / Real.sqrt (slope_l^2 + 1)) ∧
    (∀ (x y : ℝ), y = slope_l * x + b₂ ↔ 
      distance_p_to_m = |slope_l * point_p.1 - point_p.2 + b₂| / Real.sqrt (slope_l^2 + 1)) ∧
    b₁ ≠ b₂ :=
sorry

end NUMINAMATH_CALUDE_equation_of_line_l_equations_of_line_m_l3035_303583


namespace NUMINAMATH_CALUDE_correct_statements_l3035_303593

/-- Represents a mathematical statement about proofs and principles -/
inductive MathStatement
  | InductionInfinite
  | ProofStructure
  | TheoremProof
  | AxiomPostulate
  | NoUnprovenConjectures

/-- Determines if a given mathematical statement is correct -/
def is_correct (statement : MathStatement) : Prop :=
  match statement with
  | MathStatement.InductionInfinite => False
  | MathStatement.ProofStructure => True
  | MathStatement.TheoremProof => True
  | MathStatement.AxiomPostulate => True
  | MathStatement.NoUnprovenConjectures => True

/-- Theorem stating that statement A is incorrect while B, C, D, and E are correct -/
theorem correct_statements :
  ¬(is_correct MathStatement.InductionInfinite) ∧
  (is_correct MathStatement.ProofStructure) ∧
  (is_correct MathStatement.TheoremProof) ∧
  (is_correct MathStatement.AxiomPostulate) ∧
  (is_correct MathStatement.NoUnprovenConjectures) :=
sorry

end NUMINAMATH_CALUDE_correct_statements_l3035_303593


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l3035_303569

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 2 = 0

-- Define the point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem circle_and_tangent_line :
  -- The radius of circle C is √2
  (∃ center : ℝ × ℝ, ∀ x y : ℝ, circle_C x y ↔ (x - center.1)^2 + (y - center.2)^2 = 2) ∧
  -- The line l passes through point P and is tangent to circle C
  (∀ x y : ℝ, line_l x y → (x = point_P.1 ∧ y = point_P.2 ∨ 
    (∃! p : ℝ × ℝ, circle_C p.1 p.2 ∧ line_l p.1 p.2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l3035_303569


namespace NUMINAMATH_CALUDE_rectangle_circle_overlap_area_l3035_303528

/-- The area of overlap between a rectangle and a circle with shared center -/
theorem rectangle_circle_overlap_area 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (circle_radius : ℝ) 
  (h1 : rectangle_width = 8) 
  (h2 : rectangle_height = 2 * Real.sqrt 2) 
  (h3 : circle_radius = 2) : 
  ∃ (overlap_area : ℝ), overlap_area = 2 * Real.pi + 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_overlap_area_l3035_303528


namespace NUMINAMATH_CALUDE_mass_of_man_in_boat_l3035_303540

/-- The mass of a man who causes a boat to sink by a certain amount in water. -/
def mass_of_man (boat_length boat_breadth boat_sinkage water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinkage * water_density

/-- Theorem stating that the mass of the man is 60 kg given the specified conditions. -/
theorem mass_of_man_in_boat : 
  mass_of_man 3 2 0.01 1000 = 60 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_man_in_boat_l3035_303540


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3035_303560

theorem solve_linear_equation (x y : ℝ) :
  4 * x - y = 3 → y = 4 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3035_303560


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l3035_303556

theorem slope_angle_of_line (x y : ℝ) :
  x + Real.sqrt 3 * y - 3 = 0 →
  let m := -1 / Real.sqrt 3
  let α := Real.arctan m
  α = 150 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l3035_303556


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l3035_303509

/-- The area of a square with adjacent vertices at (1,3) and (5,6) is 25 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (5, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l3035_303509


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3035_303507

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (m, 1)
  are_parallel a b → m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3035_303507


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l3035_303503

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 9) :
  a / c = 135 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l3035_303503


namespace NUMINAMATH_CALUDE_intersection_point_properties_l3035_303544

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := ((-1 : ℝ), (2 : ℝ))

-- Define the given line for perpendicularity
def perp_line (x y : ℝ) : Prop := 2 * x + y + 5 = 0

-- Theorem statement
theorem intersection_point_properties :
  l₁ M.1 M.2 ∧ l₂ M.1 M.2 →
  (∀ x y : ℝ, y = -2 * x ↔ ∃ t : ℝ, x = t * M.1 ∧ y = t * M.2) ∧
  (∀ x y : ℝ, x - 2 * y + 5 = 0 ↔ (y - M.2 = (1/2) * (x - M.1) ∧ 
    ∃ a b : ℝ, perp_line a b ∧ (b - M.2) = (-2) * (a - M.1))) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_properties_l3035_303544


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l3035_303520

-- Define the two curves
def curve1 (x y : ℝ) : Prop := y = 8 / (x^2 + 4)
def curve2 (x y : ℝ) : Prop := x + y = 2

-- Theorem stating that the x-coordinate of the intersection point is 0
theorem intersection_x_coordinate :
  ∃ y : ℝ, curve1 0 y ∧ curve2 0 y :=
sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l3035_303520


namespace NUMINAMATH_CALUDE_remainder_of_power_mod_quadratic_l3035_303511

theorem remainder_of_power_mod_quadratic (x : ℤ) : 
  (x + 2)^1004 ≡ -x [ZMOD (x^2 - x + 1)] :=
sorry

end NUMINAMATH_CALUDE_remainder_of_power_mod_quadratic_l3035_303511


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_26_l3035_303562

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def first_digit (n : ℕ) : ℕ := n / 100

def second_digit (n : ℕ) : ℕ := (n / 10) % 10

def third_digit (n : ℕ) : ℕ := n % 10

def sum_of_squared_digits (n : ℕ) : ℕ :=
  (first_digit n)^2 + (second_digit n)^2 + (third_digit n)^2

def valid_number (n : ℕ) : Prop :=
  is_three_digit n ∧ 
  first_digit n ≠ 0 ∧
  26 % (sum_of_squared_digits n) = 0

theorem three_digit_divisible_by_26 :
  {n : ℕ | valid_number n} = 
  {100, 110, 101, 302, 320, 230, 203, 431, 413, 314, 341, 134, 143, 510, 501, 150, 105} :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_26_l3035_303562


namespace NUMINAMATH_CALUDE_percentage_relationship_l3035_303523

theorem percentage_relationship (p t j w : ℝ) : 
  j = 0.75 * p → 
  j = 0.80 * t → 
  t = p * (1 - w / 100) → 
  w = 6.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l3035_303523


namespace NUMINAMATH_CALUDE_hanna_roses_to_friends_l3035_303547

/-- Calculates the number of roses Hanna gives to her friends --/
def roses_given_to_friends (total_money : ℚ) (rose_price : ℚ) 
  (jenna_fraction : ℚ) (imma_fraction : ℚ) : ℚ :=
  let total_roses := total_money / rose_price
  let jenna_roses := jenna_fraction * total_roses
  let imma_roses := imma_fraction * total_roses
  jenna_roses + imma_roses

/-- Theorem stating the number of roses Hanna gives to her friends --/
theorem hanna_roses_to_friends : 
  roses_given_to_friends 300 2 (1/3) (1/2) = 125 := by
  sorry

end NUMINAMATH_CALUDE_hanna_roses_to_friends_l3035_303547


namespace NUMINAMATH_CALUDE_serena_mother_triple_age_l3035_303527

/-- The number of years it will take for the mother to be three times as old as the daughter. -/
def years_until_triple_age (daughter_age : ℕ) (mother_age : ℕ) : ℕ :=
  (mother_age - 3 * daughter_age) / 2

/-- Theorem stating that it will take 6 years for Serena's mother to be three times as old as Serena. -/
theorem serena_mother_triple_age :
  years_until_triple_age 9 39 = 6 := by
  sorry

end NUMINAMATH_CALUDE_serena_mother_triple_age_l3035_303527


namespace NUMINAMATH_CALUDE_catches_ratio_l3035_303580

theorem catches_ratio (joe_catches tammy_catches derek_catches : ℕ) : 
  joe_catches = 23 →
  tammy_catches = 30 →
  tammy_catches = derek_catches / 3 + 16 →
  derek_catches / joe_catches = 42 / 23 := by
  sorry

end NUMINAMATH_CALUDE_catches_ratio_l3035_303580


namespace NUMINAMATH_CALUDE_distance_covered_l3035_303551

theorem distance_covered (time_minutes : ℝ) (speed_km_per_hour : ℝ) :
  time_minutes = 24 →
  speed_km_per_hour = 10 →
  (time_minutes / 60) * speed_km_per_hour = 4 :=
by sorry

end NUMINAMATH_CALUDE_distance_covered_l3035_303551


namespace NUMINAMATH_CALUDE_sasha_kolya_distance_l3035_303582

/-- Represents the race scenario with three runners -/
structure RaceScenario where
  race_length : ℝ
  sasha_speed : ℝ
  lesha_speed : ℝ
  kolya_speed : ℝ
  sasha_lesha_gap : ℝ
  lesha_kolya_gap : ℝ
  (sasha_speed_pos : sasha_speed > 0)
  (lesha_speed_pos : lesha_speed > 0)
  (kolya_speed_pos : kolya_speed > 0)
  (race_length_pos : race_length > 0)
  (sasha_lesha_gap_pos : sasha_lesha_gap > 0)
  (lesha_kolya_gap_pos : lesha_kolya_gap > 0)
  (sasha_fastest : sasha_speed > lesha_speed ∧ sasha_speed > kolya_speed)
  (lesha_second : lesha_speed > kolya_speed)
  (sasha_lesha_relation : lesha_speed * race_length = sasha_speed * (race_length - sasha_lesha_gap))
  (lesha_kolya_relation : kolya_speed * race_length = lesha_speed * (race_length - lesha_kolya_gap))

/-- Theorem stating the distance between Sasha and Kolya when Sasha finishes -/
theorem sasha_kolya_distance (scenario : RaceScenario) :
  let sasha_finish_time := scenario.race_length / scenario.sasha_speed
  let kolya_distance := scenario.kolya_speed * sasha_finish_time
  scenario.race_length - kolya_distance = 19 := by sorry

end NUMINAMATH_CALUDE_sasha_kolya_distance_l3035_303582


namespace NUMINAMATH_CALUDE_min_bailing_rate_l3035_303592

/-- Minimum bailing rate problem -/
theorem min_bailing_rate (distance : ℝ) (leak_rate : ℝ) (capacity : ℝ) (speed : ℝ) 
  (h1 : distance = 2)
  (h2 : leak_rate = 15)
  (h3 : capacity = 50)
  (h4 : speed = 3) : 
  ∃ (bailing_rate : ℝ), bailing_rate ≥ 14 ∧ 
  (distance / speed * 60 * (leak_rate - bailing_rate) ≤ capacity) := by
  sorry

end NUMINAMATH_CALUDE_min_bailing_rate_l3035_303592


namespace NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l3035_303508

theorem linear_function_not_in_fourth_quadrant (b : ℝ) (h : b ≥ 0) :
  ∀ x y : ℝ, y = 2 * x + b → ¬(x > 0 ∧ y < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l3035_303508


namespace NUMINAMATH_CALUDE_smallest_multiple_45_div_3_l3035_303596

theorem smallest_multiple_45_div_3 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 3 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_45_div_3_l3035_303596


namespace NUMINAMATH_CALUDE_number_of_hens_l3035_303567

theorem number_of_hens (total_animals : ℕ) (total_feet : ℕ) (hen_feet cow_feet : ℕ) :
  total_animals = 48 →
  total_feet = 140 →
  hen_feet = 2 →
  cow_feet = 4 →
  ∃ (num_hens num_cows : ℕ),
    num_hens + num_cows = total_animals ∧
    num_hens * hen_feet + num_cows * cow_feet = total_feet ∧
    num_hens = 26 :=
by sorry

end NUMINAMATH_CALUDE_number_of_hens_l3035_303567
