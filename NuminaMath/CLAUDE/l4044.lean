import Mathlib

namespace NUMINAMATH_CALUDE_tommy_initial_candy_l4044_404493

/-- The amount of candy each person has after sharing equally -/
def shared_amount : ℕ := 7

/-- The number of people sharing the candy -/
def num_people : ℕ := 3

/-- Hugh's initial amount of candy -/
def hugh_initial : ℕ := 8

/-- Melany's initial amount of candy -/
def melany_initial : ℕ := 7

/-- Tommy's initial amount of candy -/
def tommy_initial : ℕ := shared_amount * num_people - hugh_initial - melany_initial

theorem tommy_initial_candy : tommy_initial = 6 := by
  sorry

end NUMINAMATH_CALUDE_tommy_initial_candy_l4044_404493


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l4044_404492

theorem adult_ticket_cost 
  (total_tickets : ℕ) 
  (total_receipts : ℕ) 
  (child_ticket_cost : ℕ) 
  (child_tickets_sold : ℕ) 
  (h1 : total_tickets = 130) 
  (h2 : total_receipts = 840) 
  (h3 : child_ticket_cost = 4) 
  (h4 : child_tickets_sold = 90) : 
  (total_receipts - child_tickets_sold * child_ticket_cost) / (total_tickets - child_tickets_sold) = 12 := by
sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l4044_404492


namespace NUMINAMATH_CALUDE_old_supervisor_salary_is_870_l4044_404401

/-- Calculates the old supervisor's salary given the conditions of the problem -/
def old_supervisor_salary (num_workers : ℕ) (initial_avg : ℚ) (new_avg : ℚ) (new_supervisor_salary : ℚ) : ℚ :=
  (num_workers + 1) * initial_avg - (num_workers * new_avg + new_supervisor_salary)

/-- Theorem stating that the old supervisor's salary is $870 given the problem conditions -/
theorem old_supervisor_salary_is_870 :
  old_supervisor_salary 8 430 410 690 = 870 := by
  sorry

#eval old_supervisor_salary 8 430 410 690

end NUMINAMATH_CALUDE_old_supervisor_salary_is_870_l4044_404401


namespace NUMINAMATH_CALUDE_temp_increase_pressure_decrease_sea_water_heat_engine_possible_l4044_404480

-- Define an ideal gas
structure IdealGas where
  temperature : ℝ
  pressure : ℝ
  volume : ℝ
  particle_count : ℕ

-- Define the ideal gas law
axiom ideal_gas_law (gas : IdealGas) : gas.pressure * gas.volume = gas.particle_count * gas.temperature

-- Define average kinetic energy of molecules
def avg_kinetic_energy (gas : IdealGas) : ℝ := gas.temperature

-- Define a heat engine
structure HeatEngine where
  hot_reservoir : ℝ
  cold_reservoir : ℝ

-- Theorem 1: Temperature increase can lead to increased kinetic energy but decreased pressure
theorem temp_increase_pressure_decrease (gas1 gas2 : IdealGas) 
  (h_temp : gas2.temperature > gas1.temperature)
  (h_volume : gas2.volume = gas1.volume)
  (h_particles : gas2.particle_count = gas1.particle_count) :
  avg_kinetic_energy gas2 > avg_kinetic_energy gas1 ∧ 
  ∃ (p : ℝ), gas2.pressure = p ∧ p < gas1.pressure :=
sorry

-- Theorem 2: Heat engine using sea water temperature difference is theoretically possible
theorem sea_water_heat_engine_possible (shallow_temp deep_temp : ℝ) 
  (h_temp_diff : shallow_temp > deep_temp) :
  ∃ (engine : HeatEngine), engine.hot_reservoir = shallow_temp ∧ 
    engine.cold_reservoir = deep_temp ∧
    (∃ (work : ℝ), work > 0) :=
sorry

end NUMINAMATH_CALUDE_temp_increase_pressure_decrease_sea_water_heat_engine_possible_l4044_404480


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l4044_404442

/-- Given a set of consecutive integers from a to a+n-1,
    where the sum of the nth number from the beginning and the nth number from the end is 50,
    prove that the median of the set is 25. -/
theorem consecutive_integers_median (a n : ℕ) (h : ∀ k, 0 ≤ k ∧ k < n → a + k + (a + n - 1 - k) = 50) :
  (if n % 2 = 0
   then (a + n / 2 - 1 + (a + n / 2)) / 2
   else a + n / 2) = 25 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l4044_404442


namespace NUMINAMATH_CALUDE_right_triangle_area_l4044_404426

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l4044_404426


namespace NUMINAMATH_CALUDE_meeting_probability_4x3_grid_l4044_404485

/-- Represents a grid network --/
structure GridNetwork where
  rows : ℕ
  cols : ℕ

/-- Represents a person moving on the grid --/
structure Person where
  start_row : ℕ
  start_col : ℕ
  end_row : ℕ
  end_col : ℕ

/-- The probability of two persons meeting on a grid network --/
def meeting_probability (grid : GridNetwork) (p1 p2 : Person) : ℚ :=
  sorry

/-- Theorem stating the probability of meeting in a 4x3 grid --/
theorem meeting_probability_4x3_grid :
  let grid : GridNetwork := ⟨4, 3⟩
  let person1 : Person := ⟨0, 0, 3, 4⟩  -- A to B
  let person2 : Person := ⟨3, 4, 0, 0⟩  -- B to A
  meeting_probability grid person1 person2 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_meeting_probability_4x3_grid_l4044_404485


namespace NUMINAMATH_CALUDE_leo_weight_l4044_404462

/-- Given that Leo's weight plus 10 pounds is 1.5 times Kendra's weight,
    and that their combined weight is 180 pounds,
    prove that Leo's current weight is 104 pounds. -/
theorem leo_weight (leo kendra : ℝ) 
  (h1 : leo + 10 = 1.5 * kendra) 
  (h2 : leo + kendra = 180) : 
  leo = 104 := by sorry

end NUMINAMATH_CALUDE_leo_weight_l4044_404462


namespace NUMINAMATH_CALUDE_min_sum_of_squares_of_roots_l4044_404414

theorem min_sum_of_squares_of_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*m*x₁ + (m^2 + 2*m + 3) = 0 →
  x₂^2 - 2*m*x₂ + (m^2 + 2*m + 3) = 0 →
  x₁ ≠ x₂ →
  ∃ (k : ℝ), k = x₁^2 + x₂^2 ∧ k ≥ 9/2 ∧ 
  (∀ (m' : ℝ) (y₁ y₂ : ℝ), 
    y₁^2 - 2*m'*y₁ + (m'^2 + 2*m' + 3) = 0 →
    y₂^2 - 2*m'*y₂ + (m'^2 + 2*m' + 3) = 0 →
    y₁ ≠ y₂ →
    y₁^2 + y₂^2 ≥ 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_of_roots_l4044_404414


namespace NUMINAMATH_CALUDE_number_divided_by_005_equals_1500_l4044_404499

theorem number_divided_by_005_equals_1500 (x : ℝ) : x / 0.05 = 1500 → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_005_equals_1500_l4044_404499


namespace NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l4044_404496

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary
  (a b : Line) (α β : Plane)
  (h1 : contains α a)
  (h2 : perpendicular b β) :
  (∀ a b α β, parallel α β → perpendicularLines a b) ∧
  (∃ a b α β, perpendicularLines a b ∧ ¬ parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l4044_404496


namespace NUMINAMATH_CALUDE_scout_weekend_earnings_280_l4044_404466

/-- Calculates Scout's earnings for the weekend given the specified conditions --/
def scout_weekend_earnings (base_pay : ℕ) (sat_hours : ℕ) (sat_deliveries : ℕ) (sat_tip : ℕ)
  (sun_hours : ℕ) (sun_deliveries : ℕ) (sun_low_tip : ℕ) (sun_high_tip : ℕ)
  (transport_cost : ℕ) : ℕ :=
  let sat_earnings := base_pay * sat_hours + sat_deliveries * sat_tip - sat_deliveries * transport_cost
  let sun_earnings := 2 * base_pay * sun_hours + (sun_deliveries / 2) * (sun_low_tip + sun_high_tip) - sun_deliveries * transport_cost
  sat_earnings + sun_earnings

/-- Theorem stating that Scout's weekend earnings are $280.00 --/
theorem scout_weekend_earnings_280 :
  scout_weekend_earnings 10 6 5 5 8 10 3 7 1 = 280 := by
  sorry

#eval scout_weekend_earnings 10 6 5 5 8 10 3 7 1

end NUMINAMATH_CALUDE_scout_weekend_earnings_280_l4044_404466


namespace NUMINAMATH_CALUDE_touchdown_points_l4044_404494

theorem touchdown_points : ℕ → Prop :=
  fun p =>
    let team_a_touchdowns : ℕ := 7
    let team_b_touchdowns : ℕ := 9
    let point_difference : ℕ := 14
    (team_b_touchdowns * p = team_a_touchdowns * p + point_difference) →
    p = 7

-- Proof
example : touchdown_points 7 := by
  sorry

end NUMINAMATH_CALUDE_touchdown_points_l4044_404494


namespace NUMINAMATH_CALUDE_probability_one_black_one_white_l4044_404418

def total_balls : ℕ := 6 + 2
def black_balls : ℕ := 6
def white_balls : ℕ := 2

theorem probability_one_black_one_white :
  let total_ways := Nat.choose total_balls 2
  let favorable_ways := black_balls * white_balls
  (favorable_ways : ℚ) / total_ways = 3 / 7 := by
    sorry

end NUMINAMATH_CALUDE_probability_one_black_one_white_l4044_404418


namespace NUMINAMATH_CALUDE_average_home_runs_l4044_404459

theorem average_home_runs (players_5 players_7 players_9 players_11 players_13 : ℕ) 
  (h1 : players_5 = 3)
  (h2 : players_7 = 2)
  (h3 : players_9 = 1)
  (h4 : players_11 = 2)
  (h5 : players_13 = 1) :
  (5 * players_5 + 7 * players_7 + 9 * players_9 + 11 * players_11 + 13 * players_13) / 
  (players_5 + players_7 + players_9 + players_11 + players_13) = 73 / 9 :=
by sorry

end NUMINAMATH_CALUDE_average_home_runs_l4044_404459


namespace NUMINAMATH_CALUDE_smallest_number_l4044_404495

theorem smallest_number (S : Set ℕ) (h : S = {5, 8, 3, 2, 6}) : 
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l4044_404495


namespace NUMINAMATH_CALUDE_students_not_in_biology_l4044_404429

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) :
  total_students = 840 →
  biology_percentage = 35 / 100 →
  (total_students : ℚ) * (1 - biology_percentage) = 546 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l4044_404429


namespace NUMINAMATH_CALUDE_train_distance_l4044_404472

/-- Given a train that travels 1 mile every 2 minutes, prove it will travel 45 miles in 90 minutes -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 1 / 2 → time = 90 → distance = speed * time → distance = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l4044_404472


namespace NUMINAMATH_CALUDE_ones_digit_of_33_power_power_of_3_cycle_power_mod_4_main_theorem_l4044_404489

theorem ones_digit_of_33_power (n : ℕ) : n > 0 → (33^n) % 10 = (3^n) % 10 := by sorry

theorem power_of_3_cycle (n : ℕ) : (3^n) % 10 = (3^(n % 4)) % 10 := by sorry

theorem power_mod_4 (a b : ℕ) : a > 0 → b > 0 → (a^b) % 4 = (a % 4)^(b % 4) % 4 := by sorry

theorem main_theorem : (33^(33 * 7^7)) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_33_power_power_of_3_cycle_power_mod_4_main_theorem_l4044_404489


namespace NUMINAMATH_CALUDE_tan_alpha_value_l4044_404476

theorem tan_alpha_value (α : Real) :
  2 * Real.cos (π / 2 - α) - Real.sin (3 * π / 2 + α) = -Real.sqrt 5 →
  Real.tan α = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l4044_404476


namespace NUMINAMATH_CALUDE_vector_problem_l4044_404498

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (-1, 7)

theorem vector_problem (a b : ℝ × ℝ) (ha : a = (3, 4)) (hb : b = (-1, 7)) :
  (a.1 * b.1 + a.2 * b.2 = 25) ∧ 
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l4044_404498


namespace NUMINAMATH_CALUDE_line_direction_vector_l4044_404403

/-- Prove that for a line passing through (1, -3) and (5, 3), 
    if its direction vector is of the form (3, c), then c = 9/2 -/
theorem line_direction_vector (c : ℚ) : 
  (∃ (t : ℚ), (1 + 3*t = 5) ∧ (-3 + c*t = 3)) → c = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l4044_404403


namespace NUMINAMATH_CALUDE_product_equals_900_l4044_404449

theorem product_equals_900 (a : ℝ) (h : (a + 25)^2 = 1000) : (a + 15) * (a + 35) = 900 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_900_l4044_404449


namespace NUMINAMATH_CALUDE_ada_original_seat_l4044_404439

/-- Represents the seats in the theater --/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends --/
inductive Friend
| ada
| bea
| ceci
| dee
| edie
| fred

/-- Represents the movement of a friend --/
structure Movement where
  friend : Friend
  displacement : Int

/-- The seating arrangement before Ada left --/
def initial_arrangement : Friend → Seat := sorry

/-- The seating arrangement after all movements --/
def final_arrangement : Friend → Seat := sorry

/-- The list of all movements --/
def movements : List Movement := sorry

/-- Calculates the net displacement of all movements --/
def net_displacement (mvs : List Movement) : Int := sorry

/-- Checks if a seat is an end seat --/
def is_end_seat (s : Seat) : Prop := s = Seat.one ∨ s = Seat.six

theorem ada_original_seat (h1 : net_displacement movements = 0)
                          (h2 : is_end_seat (final_arrangement Friend.ada)) :
  is_end_seat (initial_arrangement Friend.ada) := by sorry

end NUMINAMATH_CALUDE_ada_original_seat_l4044_404439


namespace NUMINAMATH_CALUDE_complex_modulus_constraint_l4044_404460

theorem complex_modulus_constraint (a : ℝ) :
  (∀ θ : ℝ, Complex.abs ((a - Real.cos θ) + (a - 1 - Real.sin θ) * Complex.I) ≤ 2) →
  0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_constraint_l4044_404460


namespace NUMINAMATH_CALUDE_jiwon_distance_to_school_l4044_404444

/-- The distance from Taehong's house to school in kilometers -/
def taehong_distance : ℝ := 1.05

/-- The difference between Taehong's and Jiwon's distances in kilometers -/
def distance_difference : ℝ := 0.46

/-- The distance from Jiwon's house to school in kilometers -/
def jiwon_distance : ℝ := taehong_distance - distance_difference

theorem jiwon_distance_to_school :
  jiwon_distance = 0.59 := by sorry

end NUMINAMATH_CALUDE_jiwon_distance_to_school_l4044_404444


namespace NUMINAMATH_CALUDE_probability_theorem_l4044_404473

def total_cups : ℕ := 8
def white_cups : ℕ := 3
def red_cups : ℕ := 3
def black_cups : ℕ := 2
def selected_cups : ℕ := 5

def probability_specific_order : ℚ := (white_cups * (white_cups - 1) * red_cups * (red_cups - 1) * black_cups) / 
  (total_cups * (total_cups - 1) * (total_cups - 2) * (total_cups - 3) * (total_cups - 4))

def number_of_arrangements : ℕ := Nat.factorial selected_cups / 
  (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

theorem probability_theorem : 
  (↑number_of_arrangements * probability_specific_order : ℚ) = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l4044_404473


namespace NUMINAMATH_CALUDE_remainder_s_15_plus_1_l4044_404458

theorem remainder_s_15_plus_1 (s : ℤ) : (s^15 + 1) % (s - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_s_15_plus_1_l4044_404458


namespace NUMINAMATH_CALUDE_tan_sum_product_22_23_degrees_l4044_404413

theorem tan_sum_product_22_23_degrees :
  Real.tan (22 * π / 180) + Real.tan (23 * π / 180) + Real.tan (22 * π / 180) * Real.tan (23 * π / 180) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_product_22_23_degrees_l4044_404413


namespace NUMINAMATH_CALUDE_function_properties_l4044_404430

/-- Given functions f and g on ℝ satisfying certain properties, 
    prove specific characteristics of their derivatives. -/
theorem function_properties
  (f g : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f (-x + 2))
  (h2 : ∀ x, g (-x + 1) - 2 = -(g (x + 1) - 2))
  (h3 : ∀ x, f (3 - x) + g (x - 1) = 2) :
  (deriv f 2022 = 0) ∧
  (∀ x, deriv g (-x) = -(deriv g x)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4044_404430


namespace NUMINAMATH_CALUDE_problem_solution_l4044_404463

theorem problem_solution : ∃! x : ℝ, x * 13.26 + x * 9.43 + x * 77.31 = 470 ∧ x = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4044_404463


namespace NUMINAMATH_CALUDE_symmetric_cubic_homogeneous_decomposition_non_negative_equivalence_l4044_404484

-- Define the symmetric polynomials g₁, g₂, g₃
def g₁ (x y z : ℝ) : ℝ := x * (x - y) * (x - z) + y * (y - x) * (y - z) + z * (z - x) * (z - y)
def g₂ (x y z : ℝ) : ℝ := (y + z) * (x - y) * (x - z) + (x + z) * (y - x) * (y - z) + (x + y) * (z - x) * (z - y)
def g₃ (x y z : ℝ) : ℝ := x * y * z

-- Define a ternary symmetric cubic homogeneous polynomial
def SymmetricCubicHomogeneous (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f x y z = f y z x ∧ f x y z = f y x z ∧ ∀ t : ℝ, f (t*x) (t*y) (t*z) = t^3 * f x y z

theorem symmetric_cubic_homogeneous_decomposition
  (f : ℝ → ℝ → ℝ → ℝ) (h : SymmetricCubicHomogeneous f) :
  ∃! (a b c : ℝ), ∀ x y z : ℝ, f x y z = a * g₁ x y z + b * g₂ x y z + c * g₃ x y z :=
sorry

theorem non_negative_equivalence
  (f : ℝ → ℝ → ℝ → ℝ) (h : SymmetricCubicHomogeneous f)
  (a b c : ℝ) (h_decomp : ∀ x y z : ℝ, f x y z = a * g₁ x y z + b * g₂ x y z + c * g₃ x y z) :
  (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → f x y z ≥ 0) ↔ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_cubic_homogeneous_decomposition_non_negative_equivalence_l4044_404484


namespace NUMINAMATH_CALUDE_other_amount_theorem_l4044_404447

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem other_amount_theorem :
  let initial_principal : ℝ := 200
  let initial_rate : ℝ := 0.1
  let initial_time : ℝ := 12
  let other_rate : ℝ := 0.12
  let other_time : ℝ := 2
  let other_principal : ℝ := 1000
  simple_interest initial_principal initial_rate initial_time =
    simple_interest other_principal other_rate other_time := by
  sorry

end NUMINAMATH_CALUDE_other_amount_theorem_l4044_404447


namespace NUMINAMATH_CALUDE_jimmy_change_l4044_404461

def pen_cost : ℕ := 1
def notebook_cost : ℕ := 3
def folder_cost : ℕ := 5

def num_pens : ℕ := 3
def num_notebooks : ℕ := 4
def num_folders : ℕ := 2

def bill_amount : ℕ := 50

theorem jimmy_change :
  bill_amount - (num_pens * pen_cost + num_notebooks * notebook_cost + num_folders * folder_cost) = 25 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_change_l4044_404461


namespace NUMINAMATH_CALUDE_area_condition_implies_parallel_to_KL_l4044_404422

/-- A quadrilateral with non-parallel sides AB and CD -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (not_parallel : ¬ (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1))

/-- The area of a triangle given by three points -/
noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The area of a quadrilateral -/
noncomputable def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- The intersection point of lines AB and CD -/
noncomputable def intersectionPoint (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Point K on the extension of AB such that OK = AB -/
noncomputable def pointK (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Point L on the extension of CD such that OL = CD -/
noncomputable def pointL (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Check if three points are collinear -/
def collinear (P Q R : ℝ × ℝ) : Prop := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (X : ℝ × ℝ) (q : Quadrilateral) : Prop := sorry

/-- Check if two lines are parallel -/
def parallel (P Q R S : ℝ × ℝ) : Prop := sorry

theorem area_condition_implies_parallel_to_KL (q : Quadrilateral) (X : ℝ × ℝ) :
  isInside X q →
  triangleArea q.A q.B X + triangleArea q.C q.D X = (quadrilateralArea q) / 2 →
  ∃ P Q : ℝ × ℝ, collinear P Q X ∧ parallel P Q (pointK q) (pointL q) :=
sorry

end NUMINAMATH_CALUDE_area_condition_implies_parallel_to_KL_l4044_404422


namespace NUMINAMATH_CALUDE_external_roads_different_colors_l4044_404455

/-- Represents a city with colored streets and intersections -/
structure ColoredCity where
  /-- Number of intersections in the city -/
  n : ℕ
  /-- Number of colors used for streets (assumed to be 3) -/
  num_colors : ℕ
  /-- Number of streets meeting at each intersection (assumed to be 3) -/
  streets_per_intersection : ℕ
  /-- Number of roads leading out of the city (assumed to be 3) -/
  num_external_roads : ℕ
  /-- Condition: Streets are colored using three colors -/
  h_num_colors : num_colors = 3
  /-- Condition: Exactly three streets meet at each intersection -/
  h_streets_per_intersection : streets_per_intersection = 3
  /-- Condition: Three roads lead out of the city -/
  h_num_external_roads : num_external_roads = 3

/-- Theorem: In a ColoredCity, the three roads leading out of the city have different colors -/
theorem external_roads_different_colors (city : ColoredCity) :
  ∃ (c₁ c₂ c₃ : ℕ), c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ ∧
  c₁ ≤ city.num_colors ∧ c₂ ≤ city.num_colors ∧ c₃ ≤ city.num_colors :=
sorry

end NUMINAMATH_CALUDE_external_roads_different_colors_l4044_404455


namespace NUMINAMATH_CALUDE_m_value_l4044_404437

theorem m_value (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : a + b = 2) : 
  m = 100 := by
sorry

end NUMINAMATH_CALUDE_m_value_l4044_404437


namespace NUMINAMATH_CALUDE_grinder_price_is_15000_l4044_404419

/-- The price of a grinder and a mobile phone transaction --/
def GrinderMobileTransaction (grinder_price : ℝ) : Prop :=
  let mobile_price : ℝ := 8000
  let grinder_sell_price : ℝ := grinder_price * 0.96
  let mobile_sell_price : ℝ := mobile_price * 1.10
  let total_buy_price : ℝ := grinder_price + mobile_price
  let total_sell_price : ℝ := grinder_sell_price + mobile_sell_price
  total_sell_price - total_buy_price = 200

/-- The grinder price is 15000 given the transaction conditions --/
theorem grinder_price_is_15000 : 
  ∃ (price : ℝ), GrinderMobileTransaction price ∧ price = 15000 := by
  sorry

end NUMINAMATH_CALUDE_grinder_price_is_15000_l4044_404419


namespace NUMINAMATH_CALUDE_center_line_perpendicular_iff_arithmetic_progression_l4044_404456

/-- A triangle with sides a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The incenter of a triangle. -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle. -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- The line passing through two points. -/
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The angle bisectors of a triangle. -/
def angle_bisectors (t : Triangle) : List (Set (ℝ × ℝ)) := sorry

/-- Two lines are perpendicular. -/
def perpendicular (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

/-- The sides of a triangle form an arithmetic progression. -/
def arithmetic_progression (t : Triangle) : Prop :=
  t.a - t.b = t.b - t.c

theorem center_line_perpendicular_iff_arithmetic_progression (t : Triangle) :
  ∃ (bisector : Set (ℝ × ℝ)), bisector ∈ angle_bisectors t ∧
    perpendicular (line_through (incenter t) (circumcenter t)) bisector
  ↔ arithmetic_progression t := by sorry

end NUMINAMATH_CALUDE_center_line_perpendicular_iff_arithmetic_progression_l4044_404456


namespace NUMINAMATH_CALUDE_parabola_directrix_l4044_404400

/-- Given a parabola with equation x^2 = (1/2)y, its directrix is y = -1/8 -/
theorem parabola_directrix (x y : ℝ) : 
  (x^2 = (1/2) * y) → (∃ p : ℝ, p = 1/4 ∧ y = -p/2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4044_404400


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l4044_404434

-- Define the ellipse and parabola equations
def ellipse (x y a : ℝ) : Prop := x^2 + 4*(y-a)^2 = 4
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the theorem
theorem ellipse_parabola_intersection_range (a : ℝ) :
  (∃ x y : ℝ, ellipse x y a ∧ parabola x y) →
  -1 ≤ a ∧ a ≤ 17/8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l4044_404434


namespace NUMINAMATH_CALUDE_smallest_angle_trig_equation_l4044_404436

theorem smallest_angle_trig_equation : 
  (∃ (x : ℝ), x > 0 ∧ x < 10 * π / 180 ∧ Real.sin (4*x) * Real.sin (5*x) = Real.cos (4*x) * Real.cos (5*x)) ∨
  (∀ (x : ℝ), x > 0 ∧ Real.sin (4*x) * Real.sin (5*x) = Real.cos (4*x) * Real.cos (5*x) → x ≥ 10 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_trig_equation_l4044_404436


namespace NUMINAMATH_CALUDE_intersection_chord_length_squared_l4044_404450

/-- Given two circles with radii 9 and 7, whose centers are 15 units apart,
    and a line drawn through one of their intersection points P such that
    the chords QP and PR have equal length, prove that |QP|^2 = 144 -/
theorem intersection_chord_length_squared
  (O₁ O₂ P Q R : ℝ × ℝ)  -- Points in 2D plane
  (h_dist : dist O₁ O₂ = 15)  -- Distance between centers
  (h_rad₁ : dist O₁ P = 9)   -- Radius of first circle
  (h_rad₂ : dist O₂ P = 7)   -- Radius of second circle
  (h_equal_chords : dist Q P = dist P R)  -- Equal chords
  : (dist Q P) ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_intersection_chord_length_squared_l4044_404450


namespace NUMINAMATH_CALUDE_absolute_value_sum_range_l4044_404471

theorem absolute_value_sum_range (m : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| + |x - 3| ≥ m) ↔ m ∈ Set.Iic 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_range_l4044_404471


namespace NUMINAMATH_CALUDE_algebraic_simplification_l4044_404478

variable (a b m n : ℝ)

theorem algebraic_simplification :
  (5 * a * b^2 - 2 * a^2 * b + 3 * a * b^2 - a^2 * b - 4 * a * b^2 = 4 * a * b^2 - 3 * a^2 * b) ∧
  (-5 * m * n^2 - (2 * m^2 * n - 2 * (m^2 * n - 2 * m * n^2)) = -9 * m * n^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l4044_404478


namespace NUMINAMATH_CALUDE_nonagon_ribbon_theorem_l4044_404468

def nonagon_ribbon_length (a b c d e f g h i : ℝ) : Prop :=
  a + b + c + d + e + f + g + h + i = 62 →
  1.5 * (a + b + c + d + e + f + g + h + i) = 93

theorem nonagon_ribbon_theorem :
  ∀ a b c d e f g h i : ℝ, nonagon_ribbon_length a b c d e f g h i :=
by
  sorry

end NUMINAMATH_CALUDE_nonagon_ribbon_theorem_l4044_404468


namespace NUMINAMATH_CALUDE_perpendicular_condition_l4044_404428

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line y = ax + 1 -/
def slope1 (a : ℝ) : ℝ := a

/-- The slope of the second line y = (a-2)x - 1 -/
def slope2 (a : ℝ) : ℝ := a - 2

/-- Theorem: a = 1 is the necessary and sufficient condition for the lines to be perpendicular -/
theorem perpendicular_condition : 
  ∀ a : ℝ, perpendicular (slope1 a) (slope2 a) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l4044_404428


namespace NUMINAMATH_CALUDE_apartment_cost_ratio_l4044_404448

/-- Proves that the ratio of room costs on the third floor to the first floor is 4/3 --/
theorem apartment_cost_ratio :
  ∀ (cost_floor1 cost_floor2 rooms_per_floor total_earnings : ℕ),
    cost_floor1 = 15 →
    cost_floor2 = 20 →
    rooms_per_floor = 3 →
    total_earnings = 165 →
    (total_earnings - (cost_floor1 * rooms_per_floor + cost_floor2 * rooms_per_floor)) / rooms_per_floor / cost_floor1 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_apartment_cost_ratio_l4044_404448


namespace NUMINAMATH_CALUDE_multiply_44_22_l4044_404479

theorem multiply_44_22 : 44 * 22 = 88 * 11 := by
  sorry

end NUMINAMATH_CALUDE_multiply_44_22_l4044_404479


namespace NUMINAMATH_CALUDE_secret_spreading_day_l4044_404405

/-- The number of new students who learn the secret on day n -/
def new_students (n : ℕ) : ℕ := 3^n

/-- The total number of students who know the secret after n days -/
def total_students (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when 3280 students know the secret -/
theorem secret_spreading_day : 
  ∃ n : ℕ, total_students n = 3280 ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_secret_spreading_day_l4044_404405


namespace NUMINAMATH_CALUDE_total_candle_weight_l4044_404407

/-- Represents the composition of a candle in ounces -/
structure CandleComposition where
  beeswax : ℝ
  coconut_oil : ℝ
  essential_oils : ℝ

/-- Calculates the total weight of a candle given its composition -/
def candle_weight (c : CandleComposition) : ℝ :=
  c.beeswax + c.coconut_oil + c.essential_oils

/-- Defines the composition of a small candle -/
def small_candle : CandleComposition :=
  { beeswax := 4, coconut_oil := 2, essential_oils := 0.5 }

/-- Defines the composition of a medium candle -/
def medium_candle : CandleComposition :=
  { beeswax := 8, coconut_oil := 1, essential_oils := 1 }

/-- Defines the composition of a large candle -/
def large_candle : CandleComposition :=
  { beeswax := 16, coconut_oil := 3, essential_oils := 2 }

/-- The number of small candles made -/
def num_small_candles : ℕ := 4

/-- The number of medium candles made -/
def num_medium_candles : ℕ := 3

/-- The number of large candles made -/
def num_large_candles : ℕ := 2

/-- Theorem stating that the total weight of all candles is 98 ounces -/
theorem total_candle_weight :
  (num_small_candles : ℝ) * candle_weight small_candle +
  (num_medium_candles : ℝ) * candle_weight medium_candle +
  (num_large_candles : ℝ) * candle_weight large_candle = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_candle_weight_l4044_404407


namespace NUMINAMATH_CALUDE_inequality_proof_l4044_404411

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) ≥ 
  2 / (1 + a) + 2 / (1 + b) + 2 / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4044_404411


namespace NUMINAMATH_CALUDE_fathers_sons_age_sum_l4044_404469

theorem fathers_sons_age_sum (father_age son_age : ℕ) : 
  father_age = 40 → 
  son_age = 15 → 
  2 * son_age + father_age = 70 → 
  2 * father_age + son_age = 95 :=
by sorry

end NUMINAMATH_CALUDE_fathers_sons_age_sum_l4044_404469


namespace NUMINAMATH_CALUDE_base6_addition_proof_l4044_404409

/-- Convert a base 6 number to base 10 -/
def base6to10 (x y z : Nat) : Nat :=
  x * 36 + y * 6 + z

/-- Addition in base 6 -/
def addBase6 (x₁ y₁ z₁ x₂ y₂ z₂ : Nat) : Nat × Nat × Nat :=
  let sum := base6to10 x₁ y₁ z₁ + base6to10 x₂ y₂ z₂
  (sum / 36, (sum % 36) / 6, sum % 6)

theorem base6_addition_proof (C D : Nat) :
  C < 6 ∧ D < 6 ∧
  addBase6 5 C D 0 5 2 = (1, 2, C) →
  C + D = 5 := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_proof_l4044_404409


namespace NUMINAMATH_CALUDE_cubic_difference_l4044_404445

theorem cubic_difference (x y : ℝ) (h1 : x + y = 8) (h2 : 3 * x + y = 14) :
  x^3 - y^3 = -98 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l4044_404445


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l4044_404475

/-- The number of distinct diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l4044_404475


namespace NUMINAMATH_CALUDE_nine_by_nine_corner_sum_l4044_404433

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- The value at a given position in the grid -/
def Grid.value (g : Grid) (row col : ℕ) : ℕ :=
  (row - 1) * g.size + col

/-- The sum of the corner values in the grid -/
def Grid.cornerSum (g : Grid) : ℕ :=
  g.value 1 1 + g.value 1 g.size + g.value g.size 1 + g.value g.size g.size

/-- Theorem: The sum of corner values in a 9x9 grid is 164 -/
theorem nine_by_nine_corner_sum :
  ∀ g : Grid, g.size = 9 → g.cornerSum = 164 :=
by
  sorry

end NUMINAMATH_CALUDE_nine_by_nine_corner_sum_l4044_404433


namespace NUMINAMATH_CALUDE_complement_of_union_l4044_404457

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {3, 4}

theorem complement_of_union (h1 : U = {1, 2, 3, 4, 5}) 
                            (h2 : A = {1, 2, 3}) 
                            (h3 : B = {3, 4}) : 
  U \ (A ∪ B) = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l4044_404457


namespace NUMINAMATH_CALUDE_min_marking_for_range_l4044_404443

def covers (marked : Finset ℕ) (n : ℕ) : Prop :=
  n ∈ marked ∨ (∃ m ∈ marked, n ∣ m ∨ m ∣ n)

def covers_range (marked : Finset ℕ) (start finish : ℕ) : Prop :=
  ∀ n, start ≤ n → n ≤ finish → covers marked n

theorem min_marking_for_range :
  ∃ (marked : Finset ℕ), covers_range marked 2 30 ∧ marked.card = 5 ∧
    ∀ (other : Finset ℕ), covers_range other 2 30 → other.card ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_marking_for_range_l4044_404443


namespace NUMINAMATH_CALUDE_num_tables_made_l4044_404412

-- Define the total number of furniture legs
def total_legs : Nat := 40

-- Define the number of chairs
def num_chairs : Nat := 6

-- Define the number of legs per furniture piece
def legs_per_piece : Nat := 4

-- Theorem to prove
theorem num_tables_made : 
  (total_legs - num_chairs * legs_per_piece) / legs_per_piece = 4 := by
  sorry


end NUMINAMATH_CALUDE_num_tables_made_l4044_404412


namespace NUMINAMATH_CALUDE_textbook_order_cost_l4044_404474

/-- Calculate the total cost of textbooks --/
def total_cost (english_count : ℕ) (english_price : ℚ)
                (geography_count : ℕ) (geography_price : ℚ)
                (math_count : ℕ) (math_price : ℚ)
                (science_count : ℕ) (science_price : ℚ) : ℚ :=
  english_count * english_price +
  geography_count * geography_price +
  math_count * math_price +
  science_count * science_price

/-- The total cost of the textbook order is $1155.00 --/
theorem textbook_order_cost :
  total_cost 35 (7.5) 35 (10.5) 20 12 30 (9.5) = 1155 := by
  sorry

end NUMINAMATH_CALUDE_textbook_order_cost_l4044_404474


namespace NUMINAMATH_CALUDE_T_simplification_l4044_404440

theorem T_simplification (x : ℝ) : 
  (x - 2)^4 + 8*(x - 2)^3 + 24*(x - 2)^2 + 32*(x - 2) + 16 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_T_simplification_l4044_404440


namespace NUMINAMATH_CALUDE_small_bottles_sold_percentage_l4044_404465

/-- Given the initial number of small and big bottles, the percentage of big bottles sold,
    and the total number of bottles remaining, prove that 15% of small bottles were sold. -/
theorem small_bottles_sold_percentage
  (initial_small : ℕ)
  (initial_big : ℕ)
  (big_bottles_sold_percent : ℚ)
  (total_remaining : ℕ)
  (h1 : initial_small = 5000)
  (h2 : initial_big = 12000)
  (h3 : big_bottles_sold_percent = 18/100)
  (h4 : total_remaining = 14090)
  (h5 : total_remaining = initial_small + initial_big -
        (initial_small * small_bottles_sold_percent / 100 +
         initial_big * big_bottles_sold_percent).floor) :
  small_bottles_sold_percent = 15/100 :=
sorry

end NUMINAMATH_CALUDE_small_bottles_sold_percentage_l4044_404465


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l4044_404488

/-- Given points A(0,6), B(0,0), C(8,0), and D the midpoint of AB, 
    prove that the sum of the slope and y-intercept of the line passing through C and D is 21/8 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 6) → B = (0, 0) → C = (8, 0) → D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let slope := (D.2 - C.2) / (D.1 - C.1)
  let y_intercept := D.2
  slope + y_intercept = 21 / 8 := by
sorry


end NUMINAMATH_CALUDE_slope_intercept_sum_l4044_404488


namespace NUMINAMATH_CALUDE_max_value_theorem_l4044_404432

theorem max_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (x : ℝ), x = Real.rpow (a * b * c) (1/3) ∧
  (∀ (y : ℝ), (∃ (p q r : ℝ), 0 < p ∧ 0 < q ∧ 0 < r ∧ p + q + r = 1 ∧
    y ≤ a * p / q ∧ y ≤ b * q / r ∧ y ≤ c * r / p) → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4044_404432


namespace NUMINAMATH_CALUDE_triangle_radii_inequality_l4044_404441

theorem triangle_radii_inequality (r R α β γ : Real) : 
  r > 0 → R > 0 → 
  0 < α ∧ α < π → 0 < β ∧ β < π → 0 < γ ∧ γ < π →
  α + β + γ = π →
  r / R ≤ 2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_radii_inequality_l4044_404441


namespace NUMINAMATH_CALUDE_completed_square_q_value_l4044_404408

theorem completed_square_q_value (a b c : ℝ) (h : a = 1 ∧ b = -6 ∧ c = 5) :
  ∃ (p q : ℝ), ∀ x, (x^2 + b*x + c = 0 ↔ (x + p)^2 = q) ∧ q = 4 := by
  sorry

end NUMINAMATH_CALUDE_completed_square_q_value_l4044_404408


namespace NUMINAMATH_CALUDE_power_of_power_l4044_404410

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l4044_404410


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_correct_l4044_404491

/-- A natural number is a perfect square if it's equal to some natural number squared. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

/-- A natural number is a perfect cube if it's equal to some natural number cubed. -/
def IsPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

/-- The smallest natural number that satisfies the given conditions. -/
def SmallestSatisfyingNumber : ℕ := 216

/-- Theorem stating that SmallestSatisfyingNumber is the smallest natural number
    that when multiplied by 2 becomes a perfect square and
    when multiplied by 3 becomes a perfect cube. -/
theorem smallest_satisfying_number_correct :
  (IsPerfectSquare (2 * SmallestSatisfyingNumber)) ∧
  (IsPerfectCube (3 * SmallestSatisfyingNumber)) ∧
  (∀ n : ℕ, n < SmallestSatisfyingNumber →
    ¬(IsPerfectSquare (2 * n) ∧ IsPerfectCube (3 * n))) := by
  sorry

#eval SmallestSatisfyingNumber -- Should output 216

end NUMINAMATH_CALUDE_smallest_satisfying_number_correct_l4044_404491


namespace NUMINAMATH_CALUDE_smallest_number_proof_smallest_number_l4044_404406

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_proof (N : ℕ) (x : ℕ) : Prop :=
  N - x = 746 ∧
  is_divisible_by (N - x) 8 ∧
  is_divisible_by (N - x) 14 ∧
  is_divisible_by (N - x) 26 ∧
  is_divisible_by (N - x) 28 ∧
  ∀ M : ℕ, M < N → ¬(∃ y : ℕ, smallest_number_proof M y)

theorem smallest_number : smallest_number_proof 1474 728 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_proof_smallest_number_l4044_404406


namespace NUMINAMATH_CALUDE_equal_angles_implies_rectangle_l4044_404464

-- Define a quadrilateral
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define the concept of equal angles in a quadrilateral
def has_four_equal_angles (q : Quadrilateral) : Prop := sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem equal_angles_implies_rectangle (q : Quadrilateral) :
  has_four_equal_angles q → is_rectangle q := by sorry

end NUMINAMATH_CALUDE_equal_angles_implies_rectangle_l4044_404464


namespace NUMINAMATH_CALUDE_nth_term_formula_l4044_404427

/-- Represents the coefficient of the nth term in the sequence -/
def coeff (n : ℕ) : ℕ := n + 1

/-- Represents the exponent of 'a' in the nth term of the sequence -/
def exponent (n : ℕ) : ℕ := n

/-- Represents the nth term in the sequence as a function of 'a' -/
def nthTerm (n : ℕ) (a : ℝ) : ℝ := (coeff n : ℝ) * (a ^ exponent n)

/-- The theorem stating that the nth term of the sequence is (n+1)aⁿ -/
theorem nth_term_formula (n : ℕ) (a : ℝ) : nthTerm n a = (n + 1 : ℝ) * a ^ n := by sorry

end NUMINAMATH_CALUDE_nth_term_formula_l4044_404427


namespace NUMINAMATH_CALUDE_compute_expression_l4044_404417

theorem compute_expression : 12 + 4 * (5 - 10 / 2)^3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l4044_404417


namespace NUMINAMATH_CALUDE_fraction_difference_squared_l4044_404497

theorem fraction_difference_squared (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) 
  (h1 : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 1 / x - 1 / y = 1 / (x + y)) : 
  1 / a^2 - 1 / b^2 = 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_squared_l4044_404497


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4044_404402

-- Define the conditions
def p (x : ℝ) : Prop := (x - 2)^2 ≤ 1
def q (x : ℝ) : Prop := 2 / (x - 1) ≥ 1

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4044_404402


namespace NUMINAMATH_CALUDE_scientific_notation_of_113800_l4044_404490

theorem scientific_notation_of_113800 :
  ∃ (a : ℝ) (n : ℤ), 
    113800 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ a ∧ a < 10 ∧
    a = 1.138 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_113800_l4044_404490


namespace NUMINAMATH_CALUDE_foreign_language_score_l4044_404420

theorem foreign_language_score (chinese_score math_score foreign_score : ℕ) : 
  (chinese_score + math_score + foreign_score) / 3 = 95 →
  (chinese_score + math_score) / 2 = 93 →
  foreign_score = 99 := by
sorry

end NUMINAMATH_CALUDE_foreign_language_score_l4044_404420


namespace NUMINAMATH_CALUDE_complex_cube_root_product_l4044_404435

theorem complex_cube_root_product (w : ℂ) (hw : w^3 = 1) :
  (1 - w + w^2) * (1 + w - w^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_product_l4044_404435


namespace NUMINAMATH_CALUDE_circle_center_l4044_404421

/-- Given a circle with equation x^2 + y^2 - 2mx - 3 = 0, where m < 0 and radius 2, 
    prove that its center is (-1, 0) -/
theorem circle_center (m : ℝ) (h1 : m < 0) :
  let eq := fun (x y : ℝ) ↦ x^2 + y^2 - 2*m*x - 3 = 0
  let r : ℝ := 2
  ∃ (C : ℝ × ℝ), C = (-1, 0) ∧ 
    (∀ (x y : ℝ), eq x y ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l4044_404421


namespace NUMINAMATH_CALUDE_A_power_150_is_identity_l4044_404453

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_150_is_identity :
  A ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end NUMINAMATH_CALUDE_A_power_150_is_identity_l4044_404453


namespace NUMINAMATH_CALUDE_dice_throw_pigeonhole_l4044_404404

/-- Represents a throw of four fair six-sided dice -/
def DiceThrow := Fin 4 → Fin 6

/-- The sum of a dice throw -/
def throwSum (t : DiceThrow) : ℕ := (t 0).val + 1 + (t 1).val + 1 + (t 2).val + 1 + (t 3).val + 1

/-- A sequence of dice throws -/
def ThrowSequence (n : ℕ) := Fin n → DiceThrow

theorem dice_throw_pigeonhole :
  ∀ (s : ThrowSequence 22), ∃ (i j : Fin 22), i ≠ j ∧ throwSum (s i) = throwSum (s j) :=
sorry

end NUMINAMATH_CALUDE_dice_throw_pigeonhole_l4044_404404


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4044_404431

theorem sufficient_not_necessary (a b x : ℝ) :
  (∀ a b x : ℝ, x > a^2 + b^2 → x > 2*a*b) ∧
  (∃ a b x : ℝ, x > 2*a*b ∧ x ≤ a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4044_404431


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4044_404482

theorem imaginary_part_of_complex_fraction : Complex.im (5 * Complex.I / (1 + 2 * Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4044_404482


namespace NUMINAMATH_CALUDE_smallest_solution_l4044_404454

def equation (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6 ∧
  1 / (x - 3) + 1 / (x - 5) + 1 / (x - 6) = 4 / (x - 4)

theorem smallest_solution :
  ∀ x : ℝ, equation x → x ≥ 16 ∧ equation 16 := by sorry

end NUMINAMATH_CALUDE_smallest_solution_l4044_404454


namespace NUMINAMATH_CALUDE_ellipse_focus_directrix_distance_l4044_404424

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 64 + y^2 / 28 = 1

/-- Distance from P to the left focus -/
def distance_to_left_focus (P : ℝ × ℝ) : ℝ := 4

/-- Distance from P to the right directrix -/
def distance_to_right_directrix (P : ℝ × ℝ) : ℝ := 16

/-- Theorem: If P is on the ellipse and 4 units from the left focus,
    then it is 16 units from the right directrix -/
theorem ellipse_focus_directrix_distance (P : ℝ × ℝ) :
  is_on_ellipse P.1 P.2 →
  distance_to_left_focus P = 4 →
  distance_to_right_directrix P = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_directrix_distance_l4044_404424


namespace NUMINAMATH_CALUDE_michelle_initial_crayons_l4044_404477

theorem michelle_initial_crayons :
  ∀ (michelle_initial janet : ℕ),
    janet = 2 →
    michelle_initial + janet = 4 →
    michelle_initial = 2 := by
  sorry

end NUMINAMATH_CALUDE_michelle_initial_crayons_l4044_404477


namespace NUMINAMATH_CALUDE_correct_statements_l4044_404481

theorem correct_statements :
  (∀ x : ℝ, x < 0 → x^3 < x) ∧
  (∀ x : ℝ, x^3 > 0 → x > 0) ∧
  (∀ x : ℝ, x > 1 → x^3 > x) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l4044_404481


namespace NUMINAMATH_CALUDE_probability_neither_cake_nor_muffin_l4044_404446

def total_buyers : ℕ := 100
def cake_buyers : ℕ := 50
def muffin_buyers : ℕ := 40
def both_buyers : ℕ := 16

theorem probability_neither_cake_nor_muffin :
  let buyers_of_at_least_one := cake_buyers + muffin_buyers - both_buyers
  let buyers_of_neither := total_buyers - buyers_of_at_least_one
  (buyers_of_neither : ℚ) / total_buyers = 26 / 100 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_cake_nor_muffin_l4044_404446


namespace NUMINAMATH_CALUDE_percentage_problem_l4044_404415

theorem percentage_problem (p : ℝ) (h1 : 0.25 * 660 = (p/100) * 1500 - 15) : p = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l4044_404415


namespace NUMINAMATH_CALUDE_journey_duration_first_part_l4044_404416

/-- Proves the duration of the first part of a journey given specific conditions -/
theorem journey_duration_first_part 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_first_part : ℝ) 
  (speed_second_part : ℝ) 
  (h1 : total_distance = 240)
  (h2 : total_time = 5)
  (h3 : speed_first_part = 40)
  (h4 : speed_second_part = 60) :
  ∃ (t1 : ℝ), t1 = 3 ∧ 
    speed_first_part * t1 + speed_second_part * (total_time - t1) = total_distance :=
by sorry


end NUMINAMATH_CALUDE_journey_duration_first_part_l4044_404416


namespace NUMINAMATH_CALUDE_car_distance_l4044_404487

/-- Given a total distance of 40 kilometers, if 1/4 of the distance is traveled by foot
    and 1/2 of the distance is traveled by bus, then the remaining distance traveled
    by car is 10 kilometers. -/
theorem car_distance (total_distance : ℝ) (foot_fraction : ℝ) (bus_fraction : ℝ) 
    (h1 : total_distance = 40)
    (h2 : foot_fraction = 1/4)
    (h3 : bus_fraction = 1/2) :
    total_distance - (foot_fraction * total_distance) - (bus_fraction * total_distance) = 10 := by
  sorry


end NUMINAMATH_CALUDE_car_distance_l4044_404487


namespace NUMINAMATH_CALUDE_parallel_line_family_l4044_404425

/-- The line equation as a function of x, y, and a -/
def line_equation (x y a : ℝ) : ℝ := (a - 1) * x - y + 2 * a + 1

/-- Theorem stating that the lines form a parallel family -/
theorem parallel_line_family :
  ∀ a₁ a₂ : ℝ, ∃ k : ℝ, ∀ x y : ℝ,
    line_equation x y a₁ = 0 ↔ line_equation x y a₂ = k := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_family_l4044_404425


namespace NUMINAMATH_CALUDE_equation_substitution_l4044_404470

theorem equation_substitution :
  let eq1 : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y - 2
  let eq2 : ℝ → ℝ := λ y => 2 * y - 1
  ∀ y : ℝ, eq1 (eq2 y) y = 3 * (2 * y - 1) - 4 * y - 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_substitution_l4044_404470


namespace NUMINAMATH_CALUDE_set_intersection_empty_implies_a_range_l4044_404423

def A (a : ℝ) := {x : ℝ | a - 1 < x ∧ x < 2*a + 1}
def B := {x : ℝ | 0 < x ∧ x < 1}

theorem set_intersection_empty_implies_a_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_set_intersection_empty_implies_a_range_l4044_404423


namespace NUMINAMATH_CALUDE_notebook_cost_l4044_404486

theorem notebook_cost (notebook_cost pen_cost : ℝ) 
  (total_cost : notebook_cost + pen_cost = 3.50)
  (cost_difference : notebook_cost = pen_cost + 3) : 
  notebook_cost = 3.25 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l4044_404486


namespace NUMINAMATH_CALUDE_triangle_formation_l4044_404483

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  ¬(can_form_triangle 2 3 5) ∧
  can_form_triangle 5 6 10 ∧
  ¬(can_form_triangle 1 1 3) ∧
  ¬(can_form_triangle 3 4 9) :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l4044_404483


namespace NUMINAMATH_CALUDE_project_hours_theorem_l4044_404467

theorem project_hours_theorem (kate_hours mark_hours pat_hours : ℕ) : 
  pat_hours = 2 * kate_hours →
  pat_hours = mark_hours / 3 →
  mark_hours = kate_hours + 100 →
  kate_hours + pat_hours + mark_hours = 180 := by
sorry

end NUMINAMATH_CALUDE_project_hours_theorem_l4044_404467


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_3n_l4044_404452

/-- An arithmetic progression with partial sums S_n -/
structure ArithmeticProgression where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- Given S_n = a and S_{2n} = b, prove S_{3n} = 3b - 2a -/
theorem arithmetic_progression_sum_3n 
  (ap : ArithmeticProgression) (n : ℕ) (a b : ℝ) 
  (h1 : ap.S n = a) 
  (h2 : ap.S (2 * n) = b) : 
  ap.S (3 * n) = 3 * b - 2 * a := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_3n_l4044_404452


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l4044_404451

theorem smallest_x_for_equation : 
  ∀ x : ℝ, x ≠ 6 → 
  ((x^2 - x - 30) / (x - 6) = 2 / (x + 4)) → 
  x ≥ -6 ∧ 
  ∃ y : ℝ, y = -6 ∧ ((y^2 - y - 30) / (y - 6) = 2 / (y + 4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l4044_404451


namespace NUMINAMATH_CALUDE_cost_of_apple_and_watermelon_l4044_404438

/-- Represents the price of fruits in yuan per kilogram -/
structure FruitPrices where
  apple : ℝ
  watermelon : ℝ
  orange : ℝ

/-- Represents a purchase of fruits -/
structure Purchase where
  apple : ℝ
  watermelon : ℝ
  orange : ℝ
  total : ℝ

def xiaoming_purchase : Purchase :=
  { apple := 1, watermelon := 4, orange := 2, total := 27.2 }

def xiaohui_purchase : Purchase :=
  { apple := 2, watermelon := 6, orange := 2, total := 32.4 }

theorem cost_of_apple_and_watermelon (prices : FruitPrices) :
  xiaoming_purchase.apple * prices.apple +
  xiaoming_purchase.watermelon * prices.watermelon +
  xiaoming_purchase.orange * prices.orange = xiaoming_purchase.total ∧
  xiaohui_purchase.apple * prices.apple +
  xiaohui_purchase.watermelon * prices.watermelon +
  xiaohui_purchase.orange * prices.orange = xiaohui_purchase.total →
  prices.apple + 2 * prices.watermelon = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_apple_and_watermelon_l4044_404438
