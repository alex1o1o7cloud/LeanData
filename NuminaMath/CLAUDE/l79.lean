import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_by_three_l79_7960

theorem divisibility_by_three (a b : ℕ) : 
  (3 ∣ (a * b)) → (3 ∣ a) ∨ (3 ∣ b) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l79_7960


namespace NUMINAMATH_CALUDE_cubes_fill_box_l79_7908

def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_size : ℕ := 2

theorem cubes_fill_box : 
  (box_length / cube_size) * (box_width / cube_size) * (box_height / cube_size) * (cube_size^3) = 
  box_length * box_width * box_height := by
  sorry

end NUMINAMATH_CALUDE_cubes_fill_box_l79_7908


namespace NUMINAMATH_CALUDE_expression_simplification_l79_7997

theorem expression_simplification (x y : ℝ) (hx : x = 1) (hy : y = -2) :
  2 * (x + y) * (x - y) + (x + y)^2 - (6 * x^3 - 4 * x^2 * y - 2 * x * y^2) / (2 * x) = -8 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l79_7997


namespace NUMINAMATH_CALUDE_cow_herd_division_l79_7977

theorem cow_herd_division (n : ℕ) : 
  (n / 2 : ℚ) + (n / 4 : ℚ) + (n / 5 : ℚ) + 7 = n → n = 140 := by
  sorry

end NUMINAMATH_CALUDE_cow_herd_division_l79_7977


namespace NUMINAMATH_CALUDE_sheila_work_hours_l79_7946

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ  -- Hours worked on Monday, Wednesday, and Friday combined
  tue_thu_hours : ℕ      -- Hours worked on Tuesday and Thursday combined
  hourly_rate : ℕ        -- Hourly rate in dollars
  weekly_earnings : ℕ    -- Total weekly earnings in dollars

/-- Theorem: Given Sheila's work schedule and earnings, prove she works 24 hours on Mon, Wed, Fri -/
theorem sheila_work_hours (s : WorkSchedule) 
  (h1 : s.tue_thu_hours = 12)     -- 6 hours each on Tuesday and Thursday
  (h2 : s.hourly_rate = 12)       -- $12 per hour
  (h3 : s.weekly_earnings = 432)  -- $432 per week
  : s.mon_wed_fri_hours = 24 := by
  sorry


end NUMINAMATH_CALUDE_sheila_work_hours_l79_7946


namespace NUMINAMATH_CALUDE_estimation_theorem_l79_7937

-- Define a function to estimate multiplication
def estimate_mult (a b : ℕ) : ℕ :=
  let a' := (a + 5) / 10 * 10  -- Round to nearest ten
  a' * b

-- Define a function to estimate division
def estimate_div (a b : ℕ) : ℕ :=
  let a' := (a + 50) / 100 * 100  -- Round to nearest hundred
  a' / b

-- State the theorem
theorem estimation_theorem :
  estimate_mult 47 20 = 1000 ∧ estimate_div 744 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_estimation_theorem_l79_7937


namespace NUMINAMATH_CALUDE_least_number_divisibility_l79_7914

theorem least_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 234 → ¬((3072 + m) % 57 = 0 ∧ (3072 + m) % 29 = 0)) ∧ 
  ((3072 + 234) % 57 = 0 ∧ (3072 + 234) % 29 = 0) := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l79_7914


namespace NUMINAMATH_CALUDE_inequality_solution_range_l79_7949

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, -x^2 + k*x - 4 < 0) → -4 < k ∧ k < 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l79_7949


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l79_7972

theorem complex_subtraction_simplification :
  (-3 - 2*I) - (1 + 4*I) = -4 - 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l79_7972


namespace NUMINAMATH_CALUDE_fixed_point_linear_function_l79_7956

theorem fixed_point_linear_function (k : ℝ) : 
  (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_linear_function_l79_7956


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l79_7973

theorem sqrt_expression_equality : 
  Real.sqrt 12 - Real.sqrt 2 * (Real.sqrt 8 - 3 * Real.sqrt (1/2)) = 2 * Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l79_7973


namespace NUMINAMATH_CALUDE_max_distance_point_to_circle_l79_7907

/-- The maximum distance between a point and a circle -/
theorem max_distance_point_to_circle :
  let P : ℝ × ℝ := (-1, -1)
  let center : ℝ × ℝ := (3, 0)
  let radius : ℝ := 2
  let circle := {(x, y) : ℝ × ℝ | (x - 3)^2 + y^2 = 4}
  (∀ Q ∈ circle, Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt 17 + 2) ∧
  (∃ Q ∈ circle, Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 17 + 2) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_point_to_circle_l79_7907


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l79_7994

theorem cos_squared_minus_sin_squared_15_deg (π : Real) :
  let deg15 : Real := π / 12
  (Real.cos deg15)^2 - (Real.sin deg15)^2 = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l79_7994


namespace NUMINAMATH_CALUDE_exists_winning_strategy_l79_7919

/-- Represents the state of the candy game -/
structure GameState where
  pile1 : Nat
  pile2 : Nat

/-- Defines a valid move in the game -/
def ValidMove (state : GameState) (newState : GameState) : Prop :=
  (newState.pile1 = state.pile1 ∧ newState.pile2 < state.pile2 ∧ (state.pile2 - newState.pile2) % state.pile1 = 0) ∨
  (newState.pile2 = state.pile2 ∧ newState.pile1 < state.pile1 ∧ (state.pile1 - newState.pile1) % state.pile2 = 0)

/-- Defines a winning state -/
def WinningState (state : GameState) : Prop :=
  state.pile1 = 0 ∨ state.pile2 = 0

/-- Theorem stating that there exists a winning strategy -/
theorem exists_winning_strategy :
  ∃ (strategy : GameState → GameState),
    let initialState := GameState.mk 1000 2357
    ∀ (state : GameState),
      state = initialState ∨ (∃ (prevState : GameState), ValidMove prevState state) →
      WinningState state ∨ (ValidMove state (strategy state) ∧ 
        ¬∃ (nextState : GameState), ValidMove (strategy state) nextState ∧ ¬WinningState nextState) :=
sorry


end NUMINAMATH_CALUDE_exists_winning_strategy_l79_7919


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l79_7979

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_small_number :
  toScientificNotation 0.00000002 = ScientificNotation.mk 2 (-8) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l79_7979


namespace NUMINAMATH_CALUDE_equal_area_and_perimeter_l79_7971

-- Define the quadrilaterals
def quadrilateralA : List (ℝ × ℝ) := [(0,0), (3,0), (3,2), (0,3)]
def quadrilateralB : List (ℝ × ℝ) := [(0,0), (3,0), (3,3), (0,2)]

-- Function to calculate area of a quadrilateral
def area (quad : List (ℝ × ℝ)) : ℝ := sorry

-- Function to calculate perimeter of a quadrilateral
def perimeter (quad : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the areas and perimeters are equal
theorem equal_area_and_perimeter :
  area quadrilateralA = area quadrilateralB ∧
  perimeter quadrilateralA = perimeter quadrilateralB := by
  sorry

end NUMINAMATH_CALUDE_equal_area_and_perimeter_l79_7971


namespace NUMINAMATH_CALUDE_ice_cream_group_size_l79_7970

/-- The number of days it takes one person to eat a gallon of ice cream -/
def days_per_person : ℕ := 5 * 16

/-- The number of days it takes the group to eat a gallon of ice cream -/
def days_for_group : ℕ := 10

/-- The number of people in the group -/
def people_in_group : ℕ := days_per_person / days_for_group

theorem ice_cream_group_size :
  people_in_group = 8 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_group_size_l79_7970


namespace NUMINAMATH_CALUDE_prove_average_speed_l79_7955

-- Define the distances traveled on each day
def distance_day1 : ℝ := 160
def distance_day2 : ℝ := 280

-- Define the time difference between the two trips
def time_difference : ℝ := 3

-- Define the average speed
def average_speed : ℝ := 40

-- Theorem statement
theorem prove_average_speed :
  (distance_day2 / average_speed) - (distance_day1 / average_speed) = time_difference :=
by
  sorry

end NUMINAMATH_CALUDE_prove_average_speed_l79_7955


namespace NUMINAMATH_CALUDE_time_to_change_tires_l79_7951

def minutes_to_wash_car : ℕ := 10
def minutes_to_change_oil : ℕ := 15
def cars_washed : ℕ := 9
def cars_oil_changed : ℕ := 6
def sets_of_tires_changed : ℕ := 2
def hours_worked : ℕ := 4

theorem time_to_change_tires :
  let total_minutes : ℕ := hours_worked * 60
  let washing_time : ℕ := cars_washed * minutes_to_wash_car
  let oil_change_time : ℕ := cars_oil_changed * minutes_to_change_oil
  let remaining_time : ℕ := total_minutes - (washing_time + oil_change_time)
  remaining_time / sets_of_tires_changed = 30 := by sorry

end NUMINAMATH_CALUDE_time_to_change_tires_l79_7951


namespace NUMINAMATH_CALUDE_number_of_larger_planes_l79_7932

/-- Represents the number of airplanes --/
def total_planes : ℕ := 4

/-- Represents the capacity of smaller tanks in liters --/
def smaller_tank_capacity : ℕ := 60

/-- Represents the fuel cost per liter in cents --/
def fuel_cost_per_liter : ℕ := 50

/-- Represents the service charge per plane in cents --/
def service_charge : ℕ := 10000

/-- Represents the total cost to fill all planes in cents --/
def total_cost : ℕ := 55000

/-- Calculates the capacity of larger tanks --/
def larger_tank_capacity : ℕ := smaller_tank_capacity + smaller_tank_capacity / 2

/-- Calculates the fuel cost for a smaller plane in cents --/
def smaller_plane_fuel_cost : ℕ := smaller_tank_capacity * fuel_cost_per_liter

/-- Calculates the fuel cost for a larger plane in cents --/
def larger_plane_fuel_cost : ℕ := larger_tank_capacity * fuel_cost_per_liter

/-- Calculates the total cost for a smaller plane in cents --/
def smaller_plane_total_cost : ℕ := smaller_plane_fuel_cost + service_charge

/-- Calculates the total cost for a larger plane in cents --/
def larger_plane_total_cost : ℕ := larger_plane_fuel_cost + service_charge

/-- Proves that the number of larger planes is 2 --/
theorem number_of_larger_planes : 
  ∃ (n : ℕ), n + (total_planes - n) = total_planes ∧ 
             n * larger_plane_total_cost + (total_planes - n) * smaller_plane_total_cost = total_cost ∧
             n = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_larger_planes_l79_7932


namespace NUMINAMATH_CALUDE_total_pumpkins_l79_7900

theorem total_pumpkins (sandy_pumpkins mike_pumpkins : ℕ) 
  (h1 : sandy_pumpkins = 51) 
  (h2 : mike_pumpkins = 23) : 
  sandy_pumpkins + mike_pumpkins = 74 := by
  sorry

end NUMINAMATH_CALUDE_total_pumpkins_l79_7900


namespace NUMINAMATH_CALUDE_min_vertical_distance_l79_7923

/-- The absolute value function -/
def abs_func (x : ℝ) : ℝ := |x - 1|

/-- The quadratic function -/
def quad_func (x : ℝ) : ℝ := -x^2 - 4*x - 3

/-- The vertical distance between the two functions -/
def vertical_distance (x : ℝ) : ℝ := abs_func x - quad_func x

theorem min_vertical_distance :
  ∃ (min_dist : ℝ), min_dist = 7/4 ∧
  ∀ (x : ℝ), vertical_distance x ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l79_7923


namespace NUMINAMATH_CALUDE_increase_by_percentage_l79_7910

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 70 ∧ percentage = 50 → final = initial * (1 + percentage / 100) → final = 105 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l79_7910


namespace NUMINAMATH_CALUDE_not_divisible_by_2006_l79_7929

theorem not_divisible_by_2006 (k : ℤ) : ¬(2006 ∣ (k^2 + k + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_2006_l79_7929


namespace NUMINAMATH_CALUDE_intersection_count_l79_7909

/-- Two circles in a plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- The number of intersection points between two circles -/
def intersectionPoints (circles : TwoCircles) : ℕ :=
  sorry

/-- Theorem: The number of intersection points between the given circles is 4 -/
theorem intersection_count : 
  let circles : TwoCircles := {
    center1 := (0, 3),
    radius1 := 3,
    center2 := (3/2, 0),
    radius2 := 3/2
  }
  intersectionPoints circles = 4 := by sorry

end NUMINAMATH_CALUDE_intersection_count_l79_7909


namespace NUMINAMATH_CALUDE_increasing_sequence_condition_l79_7953

theorem increasing_sequence_condition (a : ℝ) :
  (∀ n : ℕ+, (n : ℝ) - a < ((n + 1) : ℝ) - a) ↔ a < (3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_increasing_sequence_condition_l79_7953


namespace NUMINAMATH_CALUDE_correct_propositions_l79_7948

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| SpecificToGeneral
| GeneralToSpecific
| SpecificToSpecific
| GeneralToGeneral

-- Define a function to check if a statement about reasoning is correct
def isCorrectStatement (rt : ReasoningType) (rd : ReasoningDirection) : Prop :=
  match rt, rd with
  | ReasoningType.Inductive, ReasoningDirection.SpecificToGeneral => True
  | ReasoningType.Deductive, ReasoningDirection.GeneralToSpecific => True
  | ReasoningType.Analogical, ReasoningDirection.SpecificToSpecific => True
  | _, _ => False

-- Define the five propositions
def proposition1 := isCorrectStatement ReasoningType.Inductive ReasoningDirection.SpecificToGeneral
def proposition2 := isCorrectStatement ReasoningType.Inductive ReasoningDirection.GeneralToGeneral
def proposition3 := isCorrectStatement ReasoningType.Deductive ReasoningDirection.GeneralToSpecific
def proposition4 := isCorrectStatement ReasoningType.Analogical ReasoningDirection.SpecificToGeneral
def proposition5 := isCorrectStatement ReasoningType.Analogical ReasoningDirection.SpecificToSpecific

-- Theorem to prove
theorem correct_propositions :
  {n : Nat | n ∈ [1, 3, 5]} = {n : Nat | n ∈ [1, 2, 3, 4, 5] ∧ 
    match n with
    | 1 => proposition1
    | 2 => proposition2
    | 3 => proposition3
    | 4 => proposition4
    | 5 => proposition5
    | _ => False} :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l79_7948


namespace NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l79_7969

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l79_7969


namespace NUMINAMATH_CALUDE_remainder_divisibility_l79_7903

theorem remainder_divisibility (x : ℤ) (h : x % 66 = 14) : x % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l79_7903


namespace NUMINAMATH_CALUDE_max_value_ratio_l79_7941

/-- An arithmetic sequence with properties S_4 = 10 and S_8 = 36 -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  S : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_def : ∀ n : ℕ+, S n = (n : ℚ) * (a 1 + a n) / 2
  S_4 : S 4 = 10
  S_8 : S 8 = 36

/-- The maximum value of a_n / S_(n+3) for the given arithmetic sequence is 1/7 -/
theorem max_value_ratio (seq : ArithmeticSequence) :
  (∃ n : ℕ+, seq.a n / seq.S (n + 3) = 1 / 7) ∧
  (∀ n : ℕ+, seq.a n / seq.S (n + 3) ≤ 1 / 7) := by
  sorry

end NUMINAMATH_CALUDE_max_value_ratio_l79_7941


namespace NUMINAMATH_CALUDE_same_terminal_side_as_pi_sixth_l79_7901

def coterminal (θ₁ θ₂ : Real) : Prop :=
  ∃ k : Int, θ₁ = θ₂ + 2 * k * Real.pi

theorem same_terminal_side_as_pi_sixth (θ : Real) : 
  coterminal θ (π/6) ↔ ∃ k : Int, θ = π/6 + 2 * k * π :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_as_pi_sixth_l79_7901


namespace NUMINAMATH_CALUDE_expression_simplification_l79_7936

theorem expression_simplification (x : ℝ) (h : x = 1 + Real.sqrt 3) :
  (x + 3) / (x^2 - 2*x + 1) * (x - 1) / (x^2 + 3*x) + 1 / x = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l79_7936


namespace NUMINAMATH_CALUDE_tan_2theta_value_l79_7999

theorem tan_2theta_value (θ : Real) (h1 : θ ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin θ - Real.cos θ = Real.sqrt 5 / 5) : 
  Real.tan (2 * θ) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_2theta_value_l79_7999


namespace NUMINAMATH_CALUDE_quadratic_root_range_l79_7939

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - 1

theorem quadratic_root_range (a b : ℝ) :
  a > 0 →
  (∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) →
  (∃ z : ℝ, 1 < z ∧ z < 2 ∧ f a b z = 0) →
  ∀ k : ℝ, -1 < k ∧ k < 1 ↔ ∃ a b : ℝ, a - b = k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l79_7939


namespace NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l79_7965

theorem arcsin_one_half_equals_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l79_7965


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l79_7967

theorem seven_digit_divisible_by_11 : ∃ (a g : ℕ), ∃ (b c d e : ℕ),
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ g ∧ g ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9 ∧
  b + c + d + e = 18 ∧
  (a * 1000000 + b * 100000 + c * 10000 + d * 1000 + e * 100 + 7 * 10 + g) % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l79_7967


namespace NUMINAMATH_CALUDE_min_phi_value_l79_7963

/-- Given a function f and a constant φ, this theorem proves that under certain conditions,
    the minimum value of φ is 5π/12. -/
theorem min_phi_value (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x) * Real.cos (2 * φ) + Real.cos (2 * x) * Real.sin (2 * φ)) →
  φ > 0 →
  (∀ x, f x = f (2 * π / 3 - x)) →
  ∃ k : ℤ, φ = k * π / 2 - π / 12 ∧ 
  (∀ m : ℤ, m * π / 2 - π / 12 > 0 → φ ≤ m * π / 2 - π / 12) :=
sorry

end NUMINAMATH_CALUDE_min_phi_value_l79_7963


namespace NUMINAMATH_CALUDE_max_min_quadratic_function_l79_7987

theorem max_min_quadratic_function :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - 2
  let interval : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
  (∀ x ∈ interval, f x ≤ -2) ∧
  (∃ x ∈ interval, f x = -2) ∧
  (∀ x ∈ interval, f x ≥ -6) ∧
  (∃ x ∈ interval, f x = -6) :=
by sorry

end NUMINAMATH_CALUDE_max_min_quadratic_function_l79_7987


namespace NUMINAMATH_CALUDE_triangle_side_sum_l79_7915

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (3 * b - a) * Real.cos C = c * Real.cos A →
  c^2 = a * b →
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 2 →
  a + b = Real.sqrt 33 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l79_7915


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l79_7990

-- Define sets A and B
def A : Set ℝ := {x | -5 < x ∧ x < 2}
def B : Set ℝ := {x | |x| < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l79_7990


namespace NUMINAMATH_CALUDE_sum_160_45_base4_l79_7935

/-- Convert a decimal number to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Convert a list of base 4 digits to decimal -/
def fromBase4 (l : List ℕ) : ℕ :=
  sorry

/-- Add two numbers in base 4 -/
def addBase4 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_160_45_base4 :
  addBase4 (toBase4 160) (toBase4 45) = [2, 4, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_sum_160_45_base4_l79_7935


namespace NUMINAMATH_CALUDE_cookies_sold_l79_7931

theorem cookies_sold (total : ℕ) (ratio_brownies : ℕ) (ratio_cookies : ℕ) (cookies : ℕ) : 
  total = 104 →
  ratio_brownies = 7 →
  ratio_cookies = 6 →
  ratio_brownies * cookies = ratio_cookies * (total - cookies) →
  cookies = 48 := by
sorry

end NUMINAMATH_CALUDE_cookies_sold_l79_7931


namespace NUMINAMATH_CALUDE_book_selling_price_l79_7980

theorem book_selling_price (CP : ℝ) : 
  (0.9 * CP = CP - 0.1 * CP) →  -- 10% loss condition
  (1.1 * CP = 990) →            -- 10% gain condition
  (0.9 * CP = 810) :=           -- Original selling price
by sorry

end NUMINAMATH_CALUDE_book_selling_price_l79_7980


namespace NUMINAMATH_CALUDE_derivative_exp_cos_l79_7902

open Real

theorem derivative_exp_cos (x : ℝ) : 
  deriv (λ x => exp x * cos x) x = exp x * (cos x - sin x) := by
sorry

end NUMINAMATH_CALUDE_derivative_exp_cos_l79_7902


namespace NUMINAMATH_CALUDE_find_divisor_l79_7985

theorem find_divisor (dividend quotient remainder : ℕ) : 
  dividend = 12401 → quotient = 76 → remainder = 13 →
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 163 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l79_7985


namespace NUMINAMATH_CALUDE_opposite_of_negative_negative_five_l79_7986

theorem opposite_of_negative_negative_five :
  -(-(5 : ℤ)) = -5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_negative_five_l79_7986


namespace NUMINAMATH_CALUDE_tan_equality_proof_l79_7983

theorem tan_equality_proof (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (345 * π / 180) → n = -15 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_proof_l79_7983


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l79_7968

/-- Calculates the expected potato yield from a rectangular garden --/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) 
  (usable_percentage : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let length_feet := length_steps * step_length
  let width_feet := width_steps * step_length
  let total_area := length_feet * width_feet
  let usable_area := total_area * usable_percentage
  usable_area * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden --/
theorem mr_green_potato_yield :
  expected_potato_yield 18 25 3 0.9 0.5 = 1822.5 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l79_7968


namespace NUMINAMATH_CALUDE_A_work_days_l79_7926

/-- The number of days B takes to finish the work alone -/
def B_days : ℕ := 15

/-- The total wages when A and B work together -/
def total_wages : ℕ := 3100

/-- A's share of the wages when working together with B -/
def A_wages : ℕ := 1860

/-- The number of days A takes to finish the work alone -/
def A_days : ℕ := 10

theorem A_work_days :
  B_days = 15 ∧
  total_wages = 3100 ∧
  A_wages = 1860 →
  A_days = 10 :=
by sorry

end NUMINAMATH_CALUDE_A_work_days_l79_7926


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l79_7995

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  collinear a b → x = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l79_7995


namespace NUMINAMATH_CALUDE_fraction_greater_than_one_implication_false_l79_7904

theorem fraction_greater_than_one_implication_false : 
  ¬(∀ a b : ℝ, a / b > 1 → a > b) := by sorry

end NUMINAMATH_CALUDE_fraction_greater_than_one_implication_false_l79_7904


namespace NUMINAMATH_CALUDE_quadratic_inequality_l79_7950

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- Define the solution set condition
def solution_set (b c : ℝ) : Set ℝ := {x | x > 2 ∨ x < 1}

-- Theorem statement
theorem quadratic_inequality (b c : ℝ) :
  (∀ x, x ∈ solution_set b c ↔ f b c x > 0) →
  (b = -3 ∧ c = 2) ∧
  (∀ x, x ∈ {x | 1/2 ≤ x ∧ x ≤ 1} ↔ 2*x^2 - 3*x + 1 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l79_7950


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l79_7996

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) → x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l79_7996


namespace NUMINAMATH_CALUDE_smallest_n_same_factors_l79_7991

/-- Count the number of factors of a natural number -/
def countFactors (n : ℕ) : ℕ := sorry

/-- Check if three consecutive numbers have the same number of factors -/
def sameFactorCount (n : ℕ) : Prop :=
  countFactors n = countFactors (n + 1) ∧ countFactors n = countFactors (n + 2)

/-- 33 is the smallest natural number n such that n, n+1, and n+2 have the same number of factors -/
theorem smallest_n_same_factors : 
  (∀ m : ℕ, m < 33 → ¬(sameFactorCount m)) ∧ sameFactorCount 33 := by sorry

end NUMINAMATH_CALUDE_smallest_n_same_factors_l79_7991


namespace NUMINAMATH_CALUDE_crayon_ratio_l79_7974

def initial_crayons : ℕ := 18
def new_crayons : ℕ := 20
def total_crayons : ℕ := 29

theorem crayon_ratio :
  (initial_crayons - (total_crayons - new_crayons)) * 2 = initial_crayons :=
sorry

end NUMINAMATH_CALUDE_crayon_ratio_l79_7974


namespace NUMINAMATH_CALUDE_post_office_mail_handling_l79_7962

/-- Represents the number of months required for a post office to handle a given amount of mail --/
def months_to_handle_mail (letters_per_day : ℕ) (packages_per_day : ℕ) (days_per_month : ℕ) (total_mail : ℕ) : ℕ :=
  total_mail / ((letters_per_day + packages_per_day) * days_per_month)

/-- Theorem stating that it takes 6 months to handle 14400 pieces of mail given the specified conditions --/
theorem post_office_mail_handling :
  months_to_handle_mail 60 20 30 14400 = 6 := by
  sorry

end NUMINAMATH_CALUDE_post_office_mail_handling_l79_7962


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l79_7975

def cost_price : ℝ := 900
def selling_price : ℝ := 1170

theorem gain_percent_calculation : 
  (selling_price - cost_price) / cost_price * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l79_7975


namespace NUMINAMATH_CALUDE_sales_tax_difference_example_l79_7981

/-- The difference between two sales tax amounts on a given price -/
def salesTaxDifference (price : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  price * rate2 - price * rate1

/-- Theorem stating that the difference between 8% and 7.5% sales tax on $50 is $0.25 -/
theorem sales_tax_difference_example : salesTaxDifference 50 0.075 0.08 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_example_l79_7981


namespace NUMINAMATH_CALUDE_divisibility_by_1947_l79_7928

theorem divisibility_by_1947 (n : ℕ) (h : Odd n) :
  (46^n + 296 * 13^n) % 1947 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1947_l79_7928


namespace NUMINAMATH_CALUDE_triangle_formation_check_l79_7964

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 9), (50, 60, 12), (11, 11, 31), (20, 30, 50)]

theorem triangle_formation_check :
  ∃! set : ℝ × ℝ × ℝ, set ∈ segment_sets ∧ 
    let (a, b, c) := set
    can_form_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_check_l79_7964


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l79_7906

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l79_7906


namespace NUMINAMATH_CALUDE_difference_of_half_and_third_l79_7916

theorem difference_of_half_and_third : 1/2 - 1/3 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_half_and_third_l79_7916


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l79_7961

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

-- State the theorem
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧ (∃ a : ℝ, a ∈ M ∧ a ∉ N) :=
by sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l79_7961


namespace NUMINAMATH_CALUDE_number_problem_l79_7988

theorem number_problem : ∃ x : ℝ, x * 0.007 = 0.0063 ∧ x = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l79_7988


namespace NUMINAMATH_CALUDE_ribbon_division_theorem_l79_7978

theorem ribbon_division_theorem (p q r s : ℝ) :
  p + q + r + s = 36 →
  (p + q) / 2 + (r + s) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_division_theorem_l79_7978


namespace NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l79_7918

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem triangle_at_most_one_obtuse (T : Triangle) :
  ¬ (∃ i j : Fin 3, i ≠ j ∧ is_obtuse (T.angles i) ∧ is_obtuse (T.angles j)) :=
sorry

end NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l79_7918


namespace NUMINAMATH_CALUDE_book_sale_problem_l79_7952

theorem book_sale_problem (cost_loss book_loss_price book_gain_price : ℝ) :
  cost_loss = 175 →
  book_loss_price = book_gain_price →
  book_loss_price = 0.85 * cost_loss →
  ∃ cost_gain : ℝ,
    book_gain_price = 1.19 * cost_gain ∧
    cost_loss + cost_gain = 300 :=
by sorry

end NUMINAMATH_CALUDE_book_sale_problem_l79_7952


namespace NUMINAMATH_CALUDE_tangent_line_determines_function_l79_7998

/-- Given a function f(x) = (mx-6)/(x^2+n) with a tangent line at P(-1, f(-1))
    with equation x + 2y + 5 = 0, prove that f(x) = (2x-6)/(x^2+3) -/
theorem tangent_line_determines_function (m n : ℝ) :
  let f : ℝ → ℝ := λ x => (m * x - 6) / (x^2 + n)
  let f' : ℝ → ℝ := λ x => ((m * (x^2 + n) - (2 * x * (m * x - 6))) / (x^2 + n)^2)
  (f' (-1) = -1/2) →
  (f (-1) = -2) →
  (∀ x, f x = (2 * x - 6) / (x^2 + 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_line_determines_function_l79_7998


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_1419_l79_7945

def consecutiveEvenProduct (n : ℕ) : ℕ :=
  (List.range ((n / 2) - 1)).foldl (λ acc i => acc * (2 * (i + 2))) 2

theorem smallest_n_divisible_by_1419 : 
  (∀ m : ℕ, m < 106 → m % 2 = 0 → ¬(consecutiveEvenProduct m % 1419 = 0)) ∧ 
  (consecutiveEvenProduct 106 % 1419 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_1419_l79_7945


namespace NUMINAMATH_CALUDE_expected_democrat_votes_l79_7934

/-- Represents the percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.60

/-- Represents the percentage of Republican voters expected to vote for candidate A -/
def republican_vote_percentage : ℝ := 0.20

/-- Represents the total percentage of votes candidate A is expected to receive -/
def total_vote_percentage : ℝ := 0.53

/-- Represents the percentage of Democrat voters expected to vote for candidate A -/
def democrat_vote_percentage : ℝ := 0.75

theorem expected_democrat_votes :
  democrat_vote_percentage * democrat_percentage + 
  republican_vote_percentage * (1 - democrat_percentage) = 
  total_vote_percentage :=
sorry

end NUMINAMATH_CALUDE_expected_democrat_votes_l79_7934


namespace NUMINAMATH_CALUDE_all_gp_lines_through_origin_l79_7957

/-- A line in the 2D plane represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three real numbers form a geometric progression -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

/-- The point (0, 0) in the 2D plane -/
def origin : ℝ × ℝ := (0, 0)

/-- Checks if a point lies on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 = l.c

theorem all_gp_lines_through_origin :
  ∀ l : Line, isGeometricProgression l.a l.b l.c → pointOnLine origin l :=
sorry

end NUMINAMATH_CALUDE_all_gp_lines_through_origin_l79_7957


namespace NUMINAMATH_CALUDE_log_expression_equality_l79_7982

theorem log_expression_equality (a b c d e x y : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hx : x > 0) (hy : y > 0) :
  Real.log (a^2 / b) + Real.log (b^3 / c^2) + Real.log (c / d) + Real.log (d^2 / e) - Real.log (a^3 * y / (e^2 * x)) = Real.log ((b^2 * e * x) / (c * a * y)) :=
by sorry

end NUMINAMATH_CALUDE_log_expression_equality_l79_7982


namespace NUMINAMATH_CALUDE_board_numbers_l79_7921

theorem board_numbers (a b : ℕ) (h1 : a > b) (h2 : a = 1580) :
  (((a - b) : ℚ) / (2^10 : ℚ)).isInt → b = 556 := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_l79_7921


namespace NUMINAMATH_CALUDE_waynes_age_l79_7944

theorem waynes_age (birth_year_julia : ℕ) (current_year : ℕ) : 
  birth_year_julia = 1979 → current_year = 2021 →
  ∃ (age_wayne age_peter age_julia : ℕ),
    age_julia = current_year - birth_year_julia ∧
    age_peter = age_julia - 2 ∧
    age_wayne = age_peter - 3 ∧
    age_wayne = 37 :=
by sorry

end NUMINAMATH_CALUDE_waynes_age_l79_7944


namespace NUMINAMATH_CALUDE_shifted_sine_equals_cosine_l79_7913

open Real

theorem shifted_sine_equals_cosine (ω φ : ℝ) (h_ω : ω < 0) :
  (∀ x, sin (ω * (x - π / 12) + φ) = cos (2 * x)) →
  ∃ k : ℤ, φ = π / 3 + 2 * π * ↑k := by sorry

end NUMINAMATH_CALUDE_shifted_sine_equals_cosine_l79_7913


namespace NUMINAMATH_CALUDE_prime_fraction_equation_l79_7924

theorem prime_fraction_equation (p q : ℕ) (hp : Prime p) (hq : Prime q) (n : ℕ+) 
  (h : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / (p * q) = (1 : ℚ) / n) :
  (p = 2 ∧ q = 3 ∧ n = 1) ∨ (p = 3 ∧ q = 2 ∧ n = 1) := by
sorry

end NUMINAMATH_CALUDE_prime_fraction_equation_l79_7924


namespace NUMINAMATH_CALUDE_truck_distance_l79_7911

/-- Proves that a truck traveling at a rate of 2 miles per 4 minutes will cover 90 miles in 3 hours -/
theorem truck_distance (rate : ℚ) (time : ℚ) : 
  rate = 2 / 4 → time = 3 * 60 → rate * time = 90 :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_l79_7911


namespace NUMINAMATH_CALUDE_equivalent_statements_l79_7942

variable (P Q : Prop)

theorem equivalent_statements : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by sorry

end NUMINAMATH_CALUDE_equivalent_statements_l79_7942


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l79_7930

theorem right_triangle_acute_angles (α β : ℝ) : 
  α = 60 → β = 90 → α + β + (180 - α - β) = 180 → 180 - α - β = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l79_7930


namespace NUMINAMATH_CALUDE_find_k_l79_7940

theorem find_k (a b c k : ℚ) : 
  (∀ x : ℚ, (a*x^2 + b*x + c + b*x^2 + a*x - 7 + k*x^2 + c*x + 3) / (x^2 - 2*x - 5) = 1) → 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_find_k_l79_7940


namespace NUMINAMATH_CALUDE_rational_root_of_polynomial_l79_7989

def f (x : ℚ) : ℚ := 3 * x^3 - 7 * x^2 - 8 * x + 4

theorem rational_root_of_polynomial :
  ∀ x : ℚ, f x = 0 ↔ x = 1/3 := by sorry

end NUMINAMATH_CALUDE_rational_root_of_polynomial_l79_7989


namespace NUMINAMATH_CALUDE_expression_evaluation_l79_7954

theorem expression_evaluation : -20 + 12 * ((5 + 15) / 4) = 40 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l79_7954


namespace NUMINAMATH_CALUDE_min_value_smallest_at_a_l79_7920

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

/-- Theorem stating that the minimum value of f(x) is smallest when a = 82/43 -/
theorem min_value_smallest_at_a (a : ℝ) :
  (∀ x : ℝ, f (82/43) x ≤ f a x) :=
sorry

end NUMINAMATH_CALUDE_min_value_smallest_at_a_l79_7920


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l79_7959

theorem quadratic_inequality_solution_set (x : ℝ) : 
  (8 * x^2 + 10 * x - 16 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 3/4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l79_7959


namespace NUMINAMATH_CALUDE_ball_count_in_bag_l79_7984

/-- Given a bag with red, black, and white balls, prove that the total number of balls is 7
    when the probability of drawing a red ball equals the probability of drawing a white ball. -/
theorem ball_count_in_bag (x : ℕ) : 
  (3 : ℚ) / (4 + x) = (x : ℚ) / (4 + x) → 3 + 1 + x = 7 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_in_bag_l79_7984


namespace NUMINAMATH_CALUDE_initial_men_count_l79_7917

/-- Given a piece of work that can be completed by some number of men in 25 hours,
    or by 12 men in 75 hours, prove that the initial number of men is 36. -/
theorem initial_men_count : ℕ :=
  let initial_time : ℕ := 25
  let new_men_count : ℕ := 12
  let new_time : ℕ := 75
  36

#check initial_men_count

end NUMINAMATH_CALUDE_initial_men_count_l79_7917


namespace NUMINAMATH_CALUDE_domain_of_f_given_range_l79_7947

-- Define the function f
def f (x : ℝ) : ℝ := x + 1

-- Define the theorem
theorem domain_of_f_given_range :
  (∀ y ∈ Set.Ioo 2 3, ∃ x, f x = y) ∧ f 2 = 3 →
  {x : ℝ | ∃ y ∈ Set.Ioo 2 3, f x = y} ∪ {2} = Set.Ioo 1 2 ∪ {2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_given_range_l79_7947


namespace NUMINAMATH_CALUDE_blake_lollipops_l79_7976

def problem (num_lollipops : ℕ) : Prop :=
  let num_chocolate_packs : ℕ := 6
  let lollipop_price : ℕ := 2
  let chocolate_pack_price : ℕ := 4 * lollipop_price
  let total_paid : ℕ := 6 * 10
  let change : ℕ := 4
  let total_spent : ℕ := total_paid - change
  let chocolate_cost : ℕ := num_chocolate_packs * chocolate_pack_price
  let lollipop_cost : ℕ := total_spent - chocolate_cost
  num_lollipops * lollipop_price = lollipop_cost

theorem blake_lollipops : ∃ (n : ℕ), problem n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_blake_lollipops_l79_7976


namespace NUMINAMATH_CALUDE_square_grid_15_toothpicks_l79_7933

/-- Calculates the total number of toothpicks in a square grid -/
def toothpicks_in_square_grid (side_length : ℕ) : ℕ :=
  2 * side_length * (side_length + 1)

/-- Theorem: A square grid with sides of 15 toothpicks uses 480 toothpicks in total -/
theorem square_grid_15_toothpicks :
  toothpicks_in_square_grid 15 = 480 := by
  sorry

end NUMINAMATH_CALUDE_square_grid_15_toothpicks_l79_7933


namespace NUMINAMATH_CALUDE_equation_solution_l79_7912

theorem equation_solution : 
  ∃ (x : ℤ), (1 + 1 / x : ℚ) ^ (x + 1) = (1 + 1 / 2003 : ℚ) ^ 2003 :=
by
  use -2004
  sorry

end NUMINAMATH_CALUDE_equation_solution_l79_7912


namespace NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l79_7905

theorem square_root_of_factorial_fraction : 
  Real.sqrt (Nat.factorial 9 / 84) = 24 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l79_7905


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l79_7925

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l79_7925


namespace NUMINAMATH_CALUDE_lcm_36_100_l79_7943

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_100_l79_7943


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l79_7922

theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (1, 2) →
  b = (2, x) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l79_7922


namespace NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l79_7958

theorem cauchy_schwarz_inequality (a b a₁ b₁ : ℝ) :
  (a * a₁ + b * b₁)^2 ≤ (a^2 + b^2) * (a₁^2 + b₁^2) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l79_7958


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_l79_7993

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_remainder (n : ℕ) (h : n = 10) : sum_factorials n % 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_l79_7993


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l79_7992

-- Define the original number
def original_number : ℕ := 141260

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.4126 * (10 ^ 5)

-- Theorem to prove
theorem scientific_notation_equivalence :
  (original_number : ℝ) = scientific_notation :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l79_7992


namespace NUMINAMATH_CALUDE_tangent_line_problem_l79_7927

theorem tangent_line_problem (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.log x + a / x
  let f' : ℝ → ℝ := λ x => 1 / x - a / (x^2)
  (∀ y, 4 * y - 1 - b = 0 ↔ y = f 1 + f' 1 * (1 - 1)) →
  a * b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l79_7927


namespace NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l79_7938

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2 / a + 1 / b = 1 → 2 * x + y ≤ 2 * a + b ∧ 2 * x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l79_7938


namespace NUMINAMATH_CALUDE_unique_quartic_polynomial_l79_7966

/-- A monic quartic polynomial with real coefficients -/
def QuarticPolynomial (a b c d : ℝ) : ℝ → ℂ :=
  fun x ↦ (x^4 : ℂ) + a*(x^3 : ℂ) + b*(x^2 : ℂ) + c*(x : ℂ) + d

theorem unique_quartic_polynomial 
  (q : ℝ → ℂ) 
  (monic : q = QuarticPolynomial (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)) 
  (root_complex : q (1 - 3*I) = 0) 
  (root_zero : q 0 = -48) 
  (root_one : q 1 = 0) : 
  q = QuarticPolynomial (-7.8) 25.4 (-23.8) 48 := by
  sorry

end NUMINAMATH_CALUDE_unique_quartic_polynomial_l79_7966
