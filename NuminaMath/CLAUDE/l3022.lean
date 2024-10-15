import Mathlib

namespace NUMINAMATH_CALUDE_acidic_solution_concentration_l3022_302287

/-- Represents the properties of an acidic solution -/
structure AcidicSolution where
  initialVolume : ℝ
  removedVolume : ℝ
  finalConcentration : ℝ
  initialConcentration : ℝ

/-- Theorem stating the relationship between initial and final concentrations -/
theorem acidic_solution_concentration 
  (solution : AcidicSolution)
  (h1 : solution.initialVolume = 27)
  (h2 : solution.removedVolume = 9)
  (h3 : solution.finalConcentration = 60)
  (h4 : solution.initialConcentration * solution.initialVolume = 
        solution.finalConcentration * (solution.initialVolume - solution.removedVolume)) :
  solution.initialConcentration = 40 := by
  sorry

#check acidic_solution_concentration

end NUMINAMATH_CALUDE_acidic_solution_concentration_l3022_302287


namespace NUMINAMATH_CALUDE_final_s_is_negative_one_l3022_302282

/-- Represents the state of the algorithm at each iteration -/
structure AlgorithmState where
  s : Int
  iterations : Nat

/-- The algorithm's step function -/
def step (state : AlgorithmState) : AlgorithmState :=
  if state.iterations % 2 = 0 then
    { s := state.s + 1, iterations := state.iterations + 1 }
  else
    { s := state.s - 1, iterations := state.iterations + 1 }

/-- The initial state of the algorithm -/
def initialState : AlgorithmState := { s := 0, iterations := 0 }

/-- Applies the step function n times -/
def applyNTimes (n : Nat) (state : AlgorithmState) : AlgorithmState :=
  match n with
  | 0 => state
  | n + 1 => step (applyNTimes n state)

/-- The final state after 5 iterations -/
def finalState : AlgorithmState := applyNTimes 5 initialState

/-- The theorem stating that the final value of s is -1 -/
theorem final_s_is_negative_one : finalState.s = -1 := by
  sorry


end NUMINAMATH_CALUDE_final_s_is_negative_one_l3022_302282


namespace NUMINAMATH_CALUDE_division_problem_l3022_302259

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3022_302259


namespace NUMINAMATH_CALUDE_stock_transaction_profit_l3022_302251

theorem stock_transaction_profit
  (initial_price : ℝ)
  (profit_percentage : ℝ)
  (loss_percentage : ℝ)
  (final_sale_percentage : ℝ)
  (h1 : initial_price = 1000)
  (h2 : profit_percentage = 0.1)
  (h3 : loss_percentage = 0.1)
  (h4 : final_sale_percentage = 0.9) :
  let first_sale_price := initial_price * (1 + profit_percentage)
  let second_sale_price := first_sale_price * (1 - loss_percentage)
  let final_sale_price := second_sale_price * final_sale_percentage
  final_sale_price - initial_price = 1 :=
by sorry

end NUMINAMATH_CALUDE_stock_transaction_profit_l3022_302251


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l3022_302294

/-- The line passing through points (2, 6) and (4, 10) intersects the x-axis at (-1, 0) -/
theorem line_intersection_x_axis :
  let p1 : ℝ × ℝ := (2, 6)
  let p2 : ℝ × ℝ := (4, 10)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let line (x : ℝ) : ℝ := m * x + b
  ∃ x : ℝ, line x = 0 ∧ x = -1 :=
sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l3022_302294


namespace NUMINAMATH_CALUDE_cookie_bags_problem_l3022_302239

/-- Given a total number of cookies and the number of cookies per bag,
    calculate the number of bags. -/
def number_of_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  total_cookies / cookies_per_bag

theorem cookie_bags_problem :
  let total_cookies : ℕ := 14
  let cookies_per_bag : ℕ := 2
  number_of_bags total_cookies cookies_per_bag = 7 := by
  sorry


end NUMINAMATH_CALUDE_cookie_bags_problem_l3022_302239


namespace NUMINAMATH_CALUDE_total_amount_is_234_l3022_302241

/-- Represents the share of each person in rupees -/
structure Share where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total amount distributed -/
def total_amount (s : Share) : ℝ := s.x + s.y + s.z

/-- The condition that y gets 45 paisa for each rupee x gets -/
def y_ratio (s : Share) : Prop := s.y = 0.45 * s.x

/-- The condition that z gets 50 paisa for each rupee x gets -/
def z_ratio (s : Share) : Prop := s.z = 0.50 * s.x

/-- The condition that y's share is 54 rupees -/
def y_share (s : Share) : Prop := s.y = 54

theorem total_amount_is_234 (s : Share) 
  (hy : y_ratio s) (hz : z_ratio s) (hy_share : y_share s) : 
  total_amount s = 234 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_234_l3022_302241


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l3022_302276

-- Define a 7-arithmetic fractional-linear function
def is_7_arithmetic_fractional_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x : ℝ, f x = (a * x + b) / (c * x + d)

-- State the theorem
theorem unique_function_satisfying_conditions :
  ∃! f : ℝ → ℝ, 
    is_7_arithmetic_fractional_linear f ∧ 
    f 0 = 0 ∧ 
    f 1 = 4 ∧ 
    f 4 = 2 ∧
    ∀ x : ℝ, f x = x / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l3022_302276


namespace NUMINAMATH_CALUDE_spread_diluted_ecoli_correct_l3022_302211

/-- Represents different biological experimental procedures -/
inductive ExperimentalProcedure
  | SpreadDilutedEColi
  | IntroduceSterileAir
  | InoculateSoilLeachate
  | UseOpenRoseFlowers

/-- Represents the outcome of an experimental procedure -/
inductive ExperimentOutcome
  | Success
  | Failure

/-- Function that determines the outcome of a given experimental procedure -/
def experimentResult (procedure : ExperimentalProcedure) : ExperimentOutcome :=
  match procedure with
  | ExperimentalProcedure.SpreadDilutedEColi => ExperimentOutcome.Success
  | _ => ExperimentOutcome.Failure

/-- Theorem stating that spreading diluted E. coli culture is the correct method -/
theorem spread_diluted_ecoli_correct :
  ∀ (procedure : ExperimentalProcedure),
    experimentResult procedure = ExperimentOutcome.Success ↔
    procedure = ExperimentalProcedure.SpreadDilutedEColi :=
by
  sorry

#check spread_diluted_ecoli_correct

end NUMINAMATH_CALUDE_spread_diluted_ecoli_correct_l3022_302211


namespace NUMINAMATH_CALUDE_cost_price_calculation_cost_price_proof_l3022_302254

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  let discounted_price := selling_price * (1 - discount_rate)
  let cost_price := discounted_price / (1 + profit_rate)
  cost_price

theorem cost_price_proof :
  cost_price_calculation 12000 0.1 0.08 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_cost_price_proof_l3022_302254


namespace NUMINAMATH_CALUDE_system_solution_l3022_302233

theorem system_solution (x y : ℝ) : 
  (2 * x^2 - 7 * x * y - 4 * y^2 + 9 * x - 18 * y + 10 = 0 ∧ x^2 + 2 * y^2 = 6) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) ∨ (x = -22/9 ∧ y = -1/9)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3022_302233


namespace NUMINAMATH_CALUDE_nested_custom_op_equals_two_l3022_302299

/-- Custom operation [a, b, c] defined as (a + b) / c where c ≠ 0 -/
def customOp (a b c : ℚ) : ℚ := (a + b) / c

/-- Theorem stating that [[50,25,75],[6,3,9],[8,4,12]] = 2 -/
theorem nested_custom_op_equals_two :
  customOp (customOp 50 25 75) (customOp 6 3 9) (customOp 8 4 12) = 2 := by
  sorry


end NUMINAMATH_CALUDE_nested_custom_op_equals_two_l3022_302299


namespace NUMINAMATH_CALUDE_no_square_divisible_by_six_between_55_and_120_l3022_302208

theorem no_square_divisible_by_six_between_55_and_120 : ¬ ∃ x : ℕ, 
  (∃ n : ℕ, x = n ^ 2) ∧ 
  (x % 6 = 0) ∧ 
  (55 < x) ∧ 
  (x < 120) := by
sorry

end NUMINAMATH_CALUDE_no_square_divisible_by_six_between_55_and_120_l3022_302208


namespace NUMINAMATH_CALUDE_hot_sauce_serving_size_l3022_302218

/-- Calculates the number of ounces per serving of hot sauce -/
theorem hot_sauce_serving_size (servings_per_day : ℕ) (quart_size : ℕ) (container_reduction : ℕ) (days_lasting : ℕ) :
  servings_per_day = 3 →
  quart_size = 32 →
  container_reduction = 2 →
  days_lasting = 20 →
  (quart_size - container_reduction : ℚ) / (servings_per_day * days_lasting) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hot_sauce_serving_size_l3022_302218


namespace NUMINAMATH_CALUDE_final_state_is_green_l3022_302253

/-- Represents the colors of chameleons -/
inductive Color
  | Yellow
  | Red
  | Green

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons -/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 34

/-- Function to model the color change when two chameleons of different colors meet -/
def colorChange (state : ChameleonState) : ChameleonState :=
  sorry

/-- Predicate to check if all chameleons have the same color -/
def allSameColor (state : ChameleonState) : Prop :=
  (state.yellow = totalChameleons) ∨ (state.red = totalChameleons) ∨ (state.green = totalChameleons)

/-- The main theorem to prove -/
theorem final_state_is_green :
  ∃ (finalState : ChameleonState),
    (allSameColor finalState) ∧ (finalState.green = totalChameleons) :=
  sorry

end NUMINAMATH_CALUDE_final_state_is_green_l3022_302253


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_negation_of_proposition_l3022_302228

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x < 2, P x) ↔ (∀ x < 2, ¬ P x) := by sorry

theorem negation_of_inequality (x : ℝ) :
  ¬(x^2 - 2*x < 0) ↔ (x^2 - 2*x ≥ 0) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x < 2, x^2 - 2*x < 0) ↔ (∀ x < 2, x^2 - 2*x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_negation_of_proposition_l3022_302228


namespace NUMINAMATH_CALUDE_louisa_travel_l3022_302223

/-- Louisa's travel problem -/
theorem louisa_travel (average_speed : ℝ) (second_day_distance : ℝ) (time_difference : ℝ) :
  average_speed = 40 →
  second_day_distance = 280 →
  time_difference = 3 →
  let second_day_time := second_day_distance / average_speed
  let first_day_time := second_day_time - time_difference
  let first_day_distance := average_speed * first_day_time
  first_day_distance = 160 := by
  sorry

end NUMINAMATH_CALUDE_louisa_travel_l3022_302223


namespace NUMINAMATH_CALUDE_sin_6phi_value_l3022_302280

theorem sin_6phi_value (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5) :
  Real.sin (6 * φ) = -396 * Real.sqrt 2 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_sin_6phi_value_l3022_302280


namespace NUMINAMATH_CALUDE_a_range_l3022_302265

/-- The line passing through points (x, y) with parameter a -/
def line (x y a : ℝ) : ℝ := x + y - a

/-- Predicate for points being on opposite sides of the line -/
def opposite_sides (a : ℝ) : Prop :=
  (line 0 0 a) * (line 1 1 a) < 0

/-- Theorem stating the range of a given the conditions -/
theorem a_range : 
  (∀ a : ℝ, opposite_sides a ↔ 0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l3022_302265


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l3022_302225

theorem sine_cosine_inequality (x : ℝ) (n : ℕ) :
  (Real.sin (2 * x))^n + ((Real.sin x)^n - (Real.cos x)^n)^2 ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l3022_302225


namespace NUMINAMATH_CALUDE_ourSystem_is_valid_l3022_302274

-- Define a structure for a linear equation
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  -- ax + by = c

-- Define a system of two linear equations
structure SystemOfTwoLinearEquations where
  eq1 : LinearEquation
  eq2 : LinearEquation

-- Define the specific system we want to prove is valid
def ourSystem : SystemOfTwoLinearEquations := {
  eq1 := { a := 1, b := 1, c := 5 },  -- x + y = 5
  eq2 := { a := 0, b := 1, c := 2 }   -- y = 2
}

-- Theorem stating that our system is a valid system of two linear equations
theorem ourSystem_is_valid : 
  (ourSystem.eq1.a ≠ 0 ∨ ourSystem.eq1.b ≠ 0) ∧ 
  (ourSystem.eq2.a ≠ 0 ∨ ourSystem.eq2.b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ourSystem_is_valid_l3022_302274


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3022_302268

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 29) →
  (a 3 + a 6 + a 9 = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3022_302268


namespace NUMINAMATH_CALUDE_unique_complementary_digit_l3022_302238

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem unique_complementary_digit (N : ℕ) : 
  ∃! d : ℕ, 0 < d ∧ d < 9 ∧ (sum_of_digits N + d) % 9 = 0 := by sorry

end NUMINAMATH_CALUDE_unique_complementary_digit_l3022_302238


namespace NUMINAMATH_CALUDE_student_arrangement_l3022_302277

/-- The number of ways to arrange students with specific conditions -/
def arrangement_count : ℕ := 120

/-- The number of male students -/
def male_students : ℕ := 3

/-- The number of female students -/
def female_students : ℕ := 4

/-- The number of students that must stand at the ends -/
def end_students : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := male_students + female_students

theorem student_arrangement :
  arrangement_count = 
    (end_students * (total_students - end_students).factorial) :=
sorry

end NUMINAMATH_CALUDE_student_arrangement_l3022_302277


namespace NUMINAMATH_CALUDE_odometer_reading_l3022_302235

theorem odometer_reading (initial_reading lunch_reading total_distance : ℝ) :
  lunch_reading - initial_reading = 372.0 →
  total_distance = 584.3 →
  initial_reading = 212.3 :=
by sorry

end NUMINAMATH_CALUDE_odometer_reading_l3022_302235


namespace NUMINAMATH_CALUDE_greatest_n_value_l3022_302290

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 6400) : n ≤ 7 ∧ ∃ m : ℤ, m = 7 ∧ 101 * m^2 ≤ 6400 :=
sorry

end NUMINAMATH_CALUDE_greatest_n_value_l3022_302290


namespace NUMINAMATH_CALUDE_davids_trip_money_l3022_302261

theorem davids_trip_money (initial_amount spent_amount remaining_amount : ℕ) :
  remaining_amount = 500 →
  spent_amount = remaining_amount + 800 →
  initial_amount = spent_amount + remaining_amount →
  initial_amount = 1800 :=
by sorry

end NUMINAMATH_CALUDE_davids_trip_money_l3022_302261


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3022_302250

theorem inequality_solution_set : 
  {x : ℝ | 3 ≤ |5 - 2*x| ∧ |5 - 2*x| < 9} = 
  Set.union (Set.Ioc (-2) 1) (Set.Icc 4 7) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3022_302250


namespace NUMINAMATH_CALUDE_pythagorean_preservation_l3022_302242

theorem pythagorean_preservation (a b c α β γ : ℝ) 
  (h1 : a^2 + b^2 = c^2)
  (h2 : α^2 + β^2 - γ^2 = 2)
  (s := a * α + b * β - c * γ)
  (p := a - α * s)
  (q := b - β * s)
  (r := c - γ * s) :
  p^2 + q^2 = r^2 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_preservation_l3022_302242


namespace NUMINAMATH_CALUDE_xray_cost_correct_l3022_302200

/-- The cost of an x-ray, given the conditions of the problem -/
def xray_cost : ℝ := 250

/-- The cost of an MRI, given that it's triple the x-ray cost -/
def mri_cost : ℝ := 3 * xray_cost

/-- The total cost of both procedures -/
def total_cost : ℝ := xray_cost + mri_cost

/-- The insurance coverage percentage -/
def insurance_coverage : ℝ := 0.8

/-- The amount paid by the patient -/
def patient_payment : ℝ := 200

/-- Theorem stating that the x-ray cost is correct given the problem conditions -/
theorem xray_cost_correct : 
  mri_cost = 3 * xray_cost ∧ 
  (1 - insurance_coverage) * total_cost = patient_payment ∧
  xray_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_xray_cost_correct_l3022_302200


namespace NUMINAMATH_CALUDE_min_distance_parabola_circle_l3022_302236

/-- The minimum distance between a point on the parabola y^2 = x and a point on the circle (x-3)^2 + y^2 = 1 is (√11)/2 - 1 -/
theorem min_distance_parabola_circle :
  let parabola := {p : ℝ × ℝ | p.2^2 = p.1}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 1}
  ∃ (d : ℝ), d = Real.sqrt 11 / 2 - 1 ∧
    ∀ (m : ℝ × ℝ) (n : ℝ × ℝ), m ∈ parabola → n ∈ circle →
      Real.sqrt ((m.1 - n.1)^2 + (m.2 - n.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parabola_circle_l3022_302236


namespace NUMINAMATH_CALUDE_rectangle_length_l3022_302260

theorem rectangle_length (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth →
  area = length * breadth →
  area = 6075 →
  length = 135 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l3022_302260


namespace NUMINAMATH_CALUDE_compute_expression_l3022_302292

theorem compute_expression : 7 + 4 * (5 - 9)^3 = -249 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3022_302292


namespace NUMINAMATH_CALUDE_circle_packing_line_division_l3022_302224

/-- A circle in the coordinate plane --/
structure Circle where
  center : ℝ × ℝ
  diameter : ℝ

/-- The region formed by the union of circular regions --/
def Region (circles : List Circle) : Set (ℝ × ℝ) := sorry

/-- A line in the coordinate plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a line divides a region into two equal areas --/
def dividesEquallyArea (l : Line) (r : Set (ℝ × ℝ)) : Prop := sorry

/-- Express a line in the form ax = by + c --/
def lineToStandardForm (l : Line) : ℕ × ℕ × ℕ := sorry

/-- The greatest common divisor of three natural numbers --/
def gcd3 (a b c : ℕ) : ℕ := sorry

theorem circle_packing_line_division :
  ∀ (circles : List Circle) (l : Line),
    circles.length = 6 ∧
    (∀ c ∈ circles, c.diameter = 2 ∧ c.center.1 > 0 ∧ c.center.2 > 0) ∧
    l.slope = 2 ∧
    dividesEquallyArea l (Region circles) →
    let (a, b, c) := lineToStandardForm l
    gcd3 a b c = 1 →
    a^2 + b^2 + c^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_packing_line_division_l3022_302224


namespace NUMINAMATH_CALUDE_set_problem_l3022_302227

theorem set_problem (U A B C : Finset ℕ) 
  (h_U : U.card = 300)
  (h_A : A.card = 80)
  (h_B : B.card = 70)
  (h_C : C.card = 60)
  (h_AB : (A ∩ B).card = 30)
  (h_AC : (A ∩ C).card = 25)
  (h_BC : (B ∩ C).card = 20)
  (h_ABC : (A ∩ B ∩ C).card = 15)
  (h_outside : (U \ (A ∪ B ∪ C)).card = 65)
  (h_subset : A ∪ B ∪ C ⊆ U) :
  (A \ (B ∪ C)).card = 40 := by
sorry

end NUMINAMATH_CALUDE_set_problem_l3022_302227


namespace NUMINAMATH_CALUDE_min_surface_area_five_cubes_l3022_302240

/-- Represents a shape made of unit cubes -/
structure Shape :=
  (num_cubes : ℕ)
  (num_joins : ℕ)

/-- Calculates the surface area of a shape -/
def surface_area (s : Shape) : ℕ :=
  s.num_cubes * 6 - s.num_joins * 2

/-- Theorem: Among shapes with 5 unit cubes, the one with 5 joins has the smallest surface area -/
theorem min_surface_area_five_cubes (s : Shape) (h1 : s.num_cubes = 5) (h2 : s.num_joins ≤ 5) :
  surface_area s ≥ surface_area { num_cubes := 5, num_joins := 5 } :=
sorry

end NUMINAMATH_CALUDE_min_surface_area_five_cubes_l3022_302240


namespace NUMINAMATH_CALUDE_last_problem_number_l3022_302281

theorem last_problem_number (start : ℕ) (solved : ℕ) (last : ℕ) : 
  start = 78 → solved = 48 → last = start + solved - 1 → last = 125 := by
  sorry

end NUMINAMATH_CALUDE_last_problem_number_l3022_302281


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l3022_302203

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 15 * x + c = 0) → 
  (a + c = 24) → 
  (a < c) → 
  (a = (24 - Real.sqrt 351) / 2 ∧ c = (24 + Real.sqrt 351) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l3022_302203


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l3022_302269

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_equals_zero :
  i^14560 + i^14561 + i^14562 + i^14563 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l3022_302269


namespace NUMINAMATH_CALUDE_range_of_m_for_sqrt_function_l3022_302295

/-- Given a function f(x) = √(x² - 2x + 2m - 1) with domain ℝ, 
    prove that the range of m is [1, ∞) -/
theorem range_of_m_for_sqrt_function (m : ℝ) : 
  (∀ x, ∃ y, y = Real.sqrt (x^2 - 2*x + 2*m - 1)) → m ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_sqrt_function_l3022_302295


namespace NUMINAMATH_CALUDE_fifteenth_triangular_less_than_square_l3022_302252

-- Define the triangular number function
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem statement
theorem fifteenth_triangular_less_than_square :
  triangular_number 15 < 15^2 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_triangular_less_than_square_l3022_302252


namespace NUMINAMATH_CALUDE_joan_remaining_books_l3022_302222

/-- The number of books Joan has after selling some -/
def books_remaining (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

/-- Theorem: Joan has 7 books remaining -/
theorem joan_remaining_books : books_remaining 33 26 = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_books_l3022_302222


namespace NUMINAMATH_CALUDE_batsman_average_increase_l3022_302283

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (b : Batsman) (additionalRuns : ℕ) : ℚ :=
  (b.totalRuns + additionalRuns) / (b.innings + 1)

/-- Theorem: If a batsman's average increases by 5 after scoring 110 runs in the 11th inning, 
    then his new average is 60 runs -/
theorem batsman_average_increase 
  (b : Batsman) 
  (h1 : b.innings = 10) 
  (h2 : newAverage b 110 = b.average + 5) : 
  newAverage b 110 = 60 := by
  sorry

#check batsman_average_increase

end NUMINAMATH_CALUDE_batsman_average_increase_l3022_302283


namespace NUMINAMATH_CALUDE_max_xyz_value_l3022_302262

theorem max_xyz_value (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (eq_cond : (2*x * 2*y) + 3*z = (x + 2*z) * (y + 2*z))
  (sum_cond : x + y + z = 2) :
  x * y * z ≤ 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_max_xyz_value_l3022_302262


namespace NUMINAMATH_CALUDE_dress_cost_theorem_l3022_302286

/-- The total cost of dresses for Patty, Ida, Jean, and Pauline -/
def total_cost (patty ida jean pauline : ℕ) : ℕ := patty + ida + jean + pauline

/-- Theorem stating the total cost of dresses given the conditions -/
theorem dress_cost_theorem :
  ∀ (patty ida jean pauline : ℕ),
    patty = ida + 10 →
    ida = jean + 30 →
    jean = pauline - 10 →
    pauline = 30 →
    total_cost patty ida jean pauline = 160 := by
  sorry

end NUMINAMATH_CALUDE_dress_cost_theorem_l3022_302286


namespace NUMINAMATH_CALUDE_vector_sum_equality_l3022_302285

theorem vector_sum_equality : 
  4 • ![-3, 6] + 3 • ![-2, 5] = ![-18, 39] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l3022_302285


namespace NUMINAMATH_CALUDE_polygon_exterior_angle_l3022_302258

theorem polygon_exterior_angle (n : ℕ) (h : n > 2) : 
  (360 : ℝ) / (n : ℝ) = 24 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angle_l3022_302258


namespace NUMINAMATH_CALUDE_course_choice_related_probability_three_males_l3022_302296

-- Define the total number of students
def total_students : ℕ := 200

-- Define the number of female students
def female_students : ℕ := 80

-- Define the number of female students majoring in the field
def female_major : ℕ := 70

-- Define the number of male students not majoring in the field
def male_non_major : ℕ := 40

-- Define the chi-square statistic threshold for 99.9% certainty
def chi_square_threshold : ℚ := 10828 / 1000

-- Define the function to calculate the chi-square statistic
def chi_square (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Theorem for the relationship between course choice, major, and gender
theorem course_choice_related :
  let male_students : ℕ := total_students - female_students
  let male_major : ℕ := male_students - male_non_major
  let female_non_major : ℕ := female_students - female_major
  chi_square female_major male_major female_non_major male_non_major > chi_square_threshold := by sorry

-- Theorem for the probability of selecting 3 males out of 5 students
theorem probability_three_males :
  (Nat.choose 4 3 : ℚ) / (Nat.choose 5 3) = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_course_choice_related_probability_three_males_l3022_302296


namespace NUMINAMATH_CALUDE_johns_remaining_money_l3022_302213

/-- Calculates the remaining money after John's purchases -/
def remaining_money (initial_amount : ℚ) : ℚ :=
  let after_snacks := initial_amount * (1 - 1/5)
  let after_necessities := after_snacks * (1 - 3/4)
  after_necessities * (1 - 1/4)

/-- Theorem stating that John's remaining money is $3 -/
theorem johns_remaining_money :
  remaining_money 20 = 3 := by sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l3022_302213


namespace NUMINAMATH_CALUDE_nurses_count_l3022_302248

theorem nurses_count (total_staff : ℕ) (doctor_ratio nurse_ratio : ℕ) : 
  total_staff = 200 → 
  doctor_ratio = 4 → 
  nurse_ratio = 6 → 
  (nurse_ratio : ℚ) / (doctor_ratio + nurse_ratio : ℚ) * total_staff = 120 := by
sorry

end NUMINAMATH_CALUDE_nurses_count_l3022_302248


namespace NUMINAMATH_CALUDE_morning_and_evening_emails_sum_l3022_302266

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 3

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- Theorem: The sum of emails Jack received in the morning and evening is 11 -/
theorem morning_and_evening_emails_sum :
  morning_emails + evening_emails = 11 := by sorry

end NUMINAMATH_CALUDE_morning_and_evening_emails_sum_l3022_302266


namespace NUMINAMATH_CALUDE_parrot_seed_consumption_l3022_302243

/-- Calculates the weekly seed consumption of a parrot given the total birdseed supply,
    number of weeks, and the cockatiel's weekly consumption. --/
theorem parrot_seed_consumption
  (total_boxes : ℕ)
  (seeds_per_box : ℕ)
  (weeks : ℕ)
  (cockatiel_weekly : ℕ)
  (h1 : total_boxes = 8)
  (h2 : seeds_per_box = 225)
  (h3 : weeks = 12)
  (h4 : cockatiel_weekly = 50) :
  (total_boxes * seeds_per_box - weeks * cockatiel_weekly) / weeks = 100 := by
  sorry

#check parrot_seed_consumption

end NUMINAMATH_CALUDE_parrot_seed_consumption_l3022_302243


namespace NUMINAMATH_CALUDE_revenue_change_after_price_and_sales_change_l3022_302204

theorem revenue_change_after_price_and_sales_change 
  (original_price original_quantity : ℝ) 
  (price_increase_percent : ℝ) 
  (sales_decrease_percent : ℝ) : 
  price_increase_percent = 60 → 
  sales_decrease_percent = 35 → 
  let new_price := original_price * (1 + price_increase_percent / 100)
  let new_quantity := original_quantity * (1 - sales_decrease_percent / 100)
  let original_revenue := original_price * original_quantity
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue * 100 = 4 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_after_price_and_sales_change_l3022_302204


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3022_302289

/-- The speed of a boat in still water, given downstream travel information and stream speed. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 6)
  (h2 : downstream_distance = 72)
  (h3 : downstream_time = 3.6) :
  downstream_distance / downstream_time - stream_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3022_302289


namespace NUMINAMATH_CALUDE_division_problem_l3022_302207

theorem division_problem (dividend quotient remainder : ℕ) 
  (h1 : dividend = 3086)
  (h2 : quotient = 36)
  (h3 : remainder = 26)
  : ∃ divisor : ℕ, 
    dividend = divisor * quotient + remainder ∧ 
    divisor = 85 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3022_302207


namespace NUMINAMATH_CALUDE_max_triangle_area_l3022_302264

def parabola (x : ℝ) : ℝ := -x^2 + 6*x - 5

theorem max_triangle_area :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (4, 3)
  let C (p : ℝ) : ℝ × ℝ := (p, parabola p)
  let triangle_area (p : ℝ) : ℝ := 
    (1/2) * abs ((A.1 * B.2 + B.1 * (C p).2 + (C p).1 * A.2) - 
                 (A.2 * B.1 + B.2 * (C p).1 + (C p).2 * A.1))
  ∀ p : ℝ, 1 ≤ p ∧ p ≤ 4 → triangle_area p ≤ 27/8 :=
by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_max_triangle_area_l3022_302264


namespace NUMINAMATH_CALUDE_exists_decreasing_arithmetic_with_non_decreasing_sums_l3022_302298

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sequence of partial sums of a given sequence -/
def partial_sums (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => partial_sums a n + a (n + 1)

/-- A sequence is decreasing -/
def is_decreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ a n

theorem exists_decreasing_arithmetic_with_non_decreasing_sums :
  ∃ a : ℕ → ℝ,
    arithmetic_sequence a ∧
    is_decreasing a ∧
    (∀ n : ℕ, a n = -2 * n + 7) ∧
    ¬(is_decreasing (partial_sums a)) := by
  sorry

end NUMINAMATH_CALUDE_exists_decreasing_arithmetic_with_non_decreasing_sums_l3022_302298


namespace NUMINAMATH_CALUDE_distribution_four_to_three_l3022_302232

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distributionCount (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribution_four_to_three :
  distributionCount 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_distribution_four_to_three_l3022_302232


namespace NUMINAMATH_CALUDE_constant_reciprocal_sum_parabola_l3022_302226

/-- Theorem: Constant Reciprocal Sum of Squared Distances on Parabola
  Given a point P(a,0) on the x-axis and a line through P intersecting
  the parabola y^2 = 8x at points A and B, if the sum of reciprocals of
  squared distances 1/|AP^2| + 1/|BP^2| is constant for all such lines,
  then a = 4. -/
theorem constant_reciprocal_sum_parabola (a : ℝ) : 
  (∀ m : ℝ, ∃ A B : ℝ × ℝ, 
    (A.2)^2 = 8 * A.1 ∧ 
    (B.2)^2 = 8 * B.1 ∧ 
    A.1 = m * A.2 + a ∧ 
    B.1 = m * B.2 + a ∧
    (∃ k : ℝ, ∀ m : ℝ, 
      1 / ((A.1 - a)^2 + (A.2)^2) + 1 / ((B.1 - a)^2 + (B.2)^2) = k)) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_reciprocal_sum_parabola_l3022_302226


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3022_302256

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ, 7^x + 1 = 3^y + 5^z →
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3022_302256


namespace NUMINAMATH_CALUDE_birthday_cards_l3022_302220

theorem birthday_cards (initial_cards total_cards : ℕ) 
  (h1 : initial_cards = 64)
  (h2 : total_cards = 82) :
  total_cards - initial_cards = 18 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cards_l3022_302220


namespace NUMINAMATH_CALUDE_leak_emptying_time_l3022_302234

theorem leak_emptying_time (tank_capacity : ℝ) (inlet_rate : ℝ) (emptying_time_with_inlet : ℝ) :
  tank_capacity = 6048 →
  inlet_rate = 6 →
  emptying_time_with_inlet = 12 →
  (tank_capacity / (tank_capacity / emptying_time_with_inlet + inlet_rate * 60)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_leak_emptying_time_l3022_302234


namespace NUMINAMATH_CALUDE_valid_assignment_l3022_302275

-- Define the squares
inductive Square
| A | B | C | D | E | F | G

-- Define the arrow directions
def nextSquare : Square → Square
| Square.B => Square.E
| Square.E => Square.C
| Square.C => Square.D
| Square.D => Square.A
| Square.A => Square.G
| Square.G => Square.F
| Square.F => Square.A  -- This should point to the square with 9, which is not in our Square type

-- Define the assignment of numbers to squares
def assignment : Square → Fin 8
| Square.A => 6
| Square.B => 2
| Square.C => 4
| Square.D => 5
| Square.E => 3
| Square.F => 8
| Square.G => 7

-- Theorem statement
theorem valid_assignment : 
  (∀ s : Square, assignment (nextSquare s) = assignment s + 1) ∧
  (∀ i : Fin 8, ∃ s : Square, assignment s = i) :=
by sorry

end NUMINAMATH_CALUDE_valid_assignment_l3022_302275


namespace NUMINAMATH_CALUDE_tangent_line_slope_at_zero_l3022_302271

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 1)

theorem tangent_line_slope_at_zero :
  let f' := deriv f
  f' 0 = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_at_zero_l3022_302271


namespace NUMINAMATH_CALUDE_line_parallel_to_y_axis_l3022_302237

/-- A line passing through the point (-1, 3) and parallel to the y-axis has the equation x = -1 -/
theorem line_parallel_to_y_axis (line : Set (ℝ × ℝ)) : 
  ((-1, 3) ∈ line) → 
  (∀ (x y₁ y₂ : ℝ), ((x, y₁) ∈ line ∧ (x, y₂) ∈ line) → y₁ = y₂) →
  (line = {p : ℝ × ℝ | p.1 = -1}) :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_y_axis_l3022_302237


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3022_302257

theorem imaginary_part_of_z (z : ℂ) (h : (3 + 4*I)*z = 5) : 
  z.im = -4/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3022_302257


namespace NUMINAMATH_CALUDE_forester_tree_planting_l3022_302284

/-- A forester's tree planting problem --/
theorem forester_tree_planting (initial_trees : ℕ) (total_goal : ℕ) : 
  initial_trees = 30 →
  total_goal = 300 →
  let monday_planted := 2 * initial_trees
  let tuesday_planted := monday_planted / 3
  let wednesday_planted := 2 * tuesday_planted
  let total_planted := monday_planted + tuesday_planted + wednesday_planted
  total_planted = 120 ∧ initial_trees + total_planted = total_goal := by
  sorry

end NUMINAMATH_CALUDE_forester_tree_planting_l3022_302284


namespace NUMINAMATH_CALUDE_matrix_sum_equality_l3022_302288

def A : Matrix (Fin 3) (Fin 3) ℤ := !![4, 1, -3; 0, -2, 5; 7, 0, 1]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![-6, 9, 2; 3, -4, -8; 0, 5, -3]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![-2, 10, -1; 3, -6, -3; 7, 5, -2]

theorem matrix_sum_equality : A + B = C := by sorry

end NUMINAMATH_CALUDE_matrix_sum_equality_l3022_302288


namespace NUMINAMATH_CALUDE_function_inequality_implies_non_negative_l3022_302209

theorem function_inequality_implies_non_negative 
  (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), f (x * y) + f (y - x) ≥ f (y + x)) : 
  ∀ (x : ℝ), f x ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_non_negative_l3022_302209


namespace NUMINAMATH_CALUDE_easter_egg_count_l3022_302219

/-- The number of Easter eggs found in the club house -/
def club_house_eggs : ℕ := 40

/-- The number of Easter eggs found in the park -/
def park_eggs : ℕ := 25

/-- The number of Easter eggs found in the town hall -/
def town_hall_eggs : ℕ := 15

/-- The total number of Easter eggs found -/
def total_eggs : ℕ := club_house_eggs + park_eggs + town_hall_eggs

theorem easter_egg_count : total_eggs = 80 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_count_l3022_302219


namespace NUMINAMATH_CALUDE_count_prime_pairs_sum_50_l3022_302212

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def primePairSum50 (p q : ℕ) : Prop := isPrime p ∧ isPrime q ∧ p + q = 50

theorem count_prime_pairs_sum_50 : 
  ∃! (count : ℕ), ∃ (pairs : Finset (ℕ × ℕ)), 
    pairs.card = count ∧ 
    (∀ (p q : ℕ), (p, q) ∈ pairs ↔ primePairSum50 p q ∧ p ≤ q) ∧
    count = 4 :=
sorry

end NUMINAMATH_CALUDE_count_prime_pairs_sum_50_l3022_302212


namespace NUMINAMATH_CALUDE_correct_equation_l3022_302273

theorem correct_equation : (-3)^2 * |-(1/3)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l3022_302273


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3022_302205

theorem right_triangle_hypotenuse (x : ℝ) :
  x > 0 ∧
  (1/2 * x * (2*x - 1) = 72) →
  Real.sqrt (x^2 + (2*x - 1)^2) = Real.sqrt 370 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3022_302205


namespace NUMINAMATH_CALUDE_race_time_proof_l3022_302278

-- Define the race parameters
def race_distance : ℝ := 120
def distance_difference : ℝ := 72
def time_difference : ℝ := 10

-- Define the theorem
theorem race_time_proof :
  ∀ (v_a v_b t_a : ℝ),
  v_a > 0 → v_b > 0 → t_a > 0 →
  v_a = race_distance / t_a →
  v_b = (race_distance - distance_difference) / t_a →
  v_b = distance_difference / (t_a + time_difference) →
  t_a = 20 := by
sorry


end NUMINAMATH_CALUDE_race_time_proof_l3022_302278


namespace NUMINAMATH_CALUDE_jessys_reading_plan_l3022_302217

/-- Jessy's reading plan problem -/
theorem jessys_reading_plan (total_pages : ℕ) (days : ℕ) (pages_per_session : ℕ) (additional_pages : ℕ)
  (h1 : total_pages = 140)
  (h2 : days = 7)
  (h3 : pages_per_session = 6)
  (h4 : additional_pages = 2) :
  ∃ (sessions : ℕ), sessions * pages_per_session * days + additional_pages * days = total_pages ∧ sessions = 3 :=
by sorry

end NUMINAMATH_CALUDE_jessys_reading_plan_l3022_302217


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3022_302202

theorem sqrt_equation_solution (t : ℝ) : 
  (Real.sqrt (49 - (t - 3)^2) - 7 = 0) ↔ (t = 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3022_302202


namespace NUMINAMATH_CALUDE_our_ellipse_equation_l3022_302231

-- Define the ellipse
structure Ellipse where
  f1 : ℝ × ℝ  -- Focus 1
  f2 : ℝ × ℝ  -- Focus 2
  min_dist : ℝ -- Shortest distance from a point on the ellipse to F₁

-- Define our specific ellipse
def our_ellipse : Ellipse :=
  { f1 := (0, -4)
  , f2 := (0, 4)
  , min_dist := 2
  }

-- Define the equation of an ellipse
def is_ellipse_equation (e : Ellipse) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x, y) ∈ {p : ℝ × ℝ | dist p e.f1 + dist p e.f2 = 2 * (e.f2.1 - e.f1.1)}

-- Theorem statement
theorem our_ellipse_equation :
  is_ellipse_equation our_ellipse (fun x y => x^2/20 + y^2/36 = 1) :=
sorry

end NUMINAMATH_CALUDE_our_ellipse_equation_l3022_302231


namespace NUMINAMATH_CALUDE_john_shirts_total_l3022_302291

theorem john_shirts_total (initial_shirts : ℕ) (bought_shirts : ℕ) : 
  initial_shirts = 12 → bought_shirts = 4 → initial_shirts + bought_shirts = 16 :=
by sorry

end NUMINAMATH_CALUDE_john_shirts_total_l3022_302291


namespace NUMINAMATH_CALUDE_parabola_vertex_l3022_302293

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = -5 * (x + 2)^2 - 6

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-2, -6)

/-- Theorem: The vertex of the parabola y = -5(x+2)^2 - 6 is at the point (-2, -6) -/
theorem parabola_vertex :
  ∀ (x y : ℝ), parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3022_302293


namespace NUMINAMATH_CALUDE_walking_path_area_l3022_302245

/-- The area of a circular walking path -/
theorem walking_path_area (outer_radius inner_radius : ℝ) 
  (h_outer : outer_radius = 26)
  (h_inner : inner_radius = 16) : 
  π * (outer_radius^2 - inner_radius^2) = 420 * π := by
  sorry

#check walking_path_area

end NUMINAMATH_CALUDE_walking_path_area_l3022_302245


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3022_302221

/-- Represents a repeating decimal with a single repeating digit -/
def repeating_decimal_single (d : ℕ) : ℚ :=
  d / 9

/-- Represents a repeating decimal with two repeating digits -/
def repeating_decimal_double (d : ℕ) : ℚ :=
  d / 99

theorem sum_of_repeating_decimals : 
  repeating_decimal_single 2 + repeating_decimal_double 2 = 8 / 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3022_302221


namespace NUMINAMATH_CALUDE_power_three_mod_eleven_l3022_302210

theorem power_three_mod_eleven : 3^2040 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_eleven_l3022_302210


namespace NUMINAMATH_CALUDE_infinite_sum_of_square_and_prime_infinite_not_sum_of_square_and_prime_l3022_302270

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a perfect square can be expressed as the sum of a perfect square and a prime
def is_sum_of_square_and_prime (n : ℕ) : Prop :=
  is_perfect_square n ∧ ∃ a b : ℕ, is_perfect_square a ∧ is_prime b ∧ n = a + b

-- Statement 1: The set of perfect squares that can be expressed as the sum of a perfect square and a prime number is infinite
theorem infinite_sum_of_square_and_prime :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_sum_of_square_and_prime n :=
sorry

-- Statement 2: The set of perfect squares that cannot be expressed as the sum of a perfect square and a prime number is infinite
theorem infinite_not_sum_of_square_and_prime :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_perfect_square n ∧ ¬is_sum_of_square_and_prime n :=
sorry

end NUMINAMATH_CALUDE_infinite_sum_of_square_and_prime_infinite_not_sum_of_square_and_prime_l3022_302270


namespace NUMINAMATH_CALUDE_factorization_equality_l3022_302216

theorem factorization_equality (x : ℝ) : x * (x + 2) + (x + 2)^2 = 2 * (x + 2) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3022_302216


namespace NUMINAMATH_CALUDE_keith_cd_player_cost_l3022_302255

/-- The amount Keith spent on speakers -/
def speakers_cost : ℚ := 136.01

/-- The amount Keith spent on new tires -/
def tires_cost : ℚ := 112.46

/-- The total amount Keith spent -/
def total_cost : ℚ := 387.85

/-- The amount Keith spent on the CD player -/
def cd_player_cost : ℚ := total_cost - (speakers_cost + tires_cost)

theorem keith_cd_player_cost :
  cd_player_cost = 139.38 := by sorry

end NUMINAMATH_CALUDE_keith_cd_player_cost_l3022_302255


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3022_302244

theorem possible_values_of_a (A B : Set ℝ) (a : ℝ) : 
  A = {x : ℝ | a * x + 2 = 0} → 
  B = {-1, 2} → 
  A ⊆ B → 
  {a | ∃ (A : Set ℝ), A = {x : ℝ | a * x + 2 = 0} ∧ A ⊆ B} = {-1, 0, 2} := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3022_302244


namespace NUMINAMATH_CALUDE_olivias_initial_amount_l3022_302215

/-- The amount of money Olivia had in her wallet initially -/
def initial_amount : ℕ := sorry

/-- The amount of money Olivia spent at the supermarket -/
def amount_spent : ℕ := 25

/-- The amount of money Olivia had left after visiting the supermarket -/
def amount_left : ℕ := 29

/-- Theorem stating that Olivia's initial amount of money was $54 -/
theorem olivias_initial_amount : initial_amount = 54 := by sorry

end NUMINAMATH_CALUDE_olivias_initial_amount_l3022_302215


namespace NUMINAMATH_CALUDE_jen_age_theorem_l3022_302247

def jen_age_when_son_born (jen_present_age : ℕ) (son_present_age : ℕ) : ℕ :=
  jen_present_age - son_present_age

theorem jen_age_theorem (jen_present_age : ℕ) (son_present_age : ℕ) :
  son_present_age = 16 →
  jen_present_age = 3 * son_present_age - 7 →
  jen_age_when_son_born jen_present_age son_present_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_jen_age_theorem_l3022_302247


namespace NUMINAMATH_CALUDE_least_valid_number_l3022_302267

def is_valid (n : ℕ) : Prop :=
  n > 1 ∧
  n % 3 = 2 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 6 = 2 ∧
  n % 7 = 2 ∧
  n % 8 = 2 ∧
  n % 9 = 2 ∧
  n % 11 = 2

theorem least_valid_number : 
  is_valid 27722 ∧ ∀ m : ℕ, m < 27722 → ¬is_valid m :=
by sorry

end NUMINAMATH_CALUDE_least_valid_number_l3022_302267


namespace NUMINAMATH_CALUDE_tank_A_height_approx_5_l3022_302249

/-- The circumference of Tank A in meters -/
def circumference_A : ℝ := 4

/-- The circumference of Tank B in meters -/
def circumference_B : ℝ := 10

/-- The height of Tank B in meters -/
def height_B : ℝ := 8

/-- The ratio of Tank A's capacity to Tank B's capacity -/
def capacity_ratio : ℝ := 0.10000000000000002

/-- The height of Tank A in meters -/
noncomputable def height_A : ℝ := 
  capacity_ratio * (circumference_B / circumference_A)^2 * height_B

theorem tank_A_height_approx_5 : 
  ∃ ε > 0, abs (height_A - 5) < ε := by sorry

end NUMINAMATH_CALUDE_tank_A_height_approx_5_l3022_302249


namespace NUMINAMATH_CALUDE_polynomial_roots_theorem_l3022_302297

theorem polynomial_roots_theorem (a b c : ℝ) : 
  (∃ (r s t : ℝ), 
    (∀ x : ℝ, x^4 - a*x^3 + b*x^2 - c*x + a = 0 ↔ x = 0 ∨ x = r ∨ x = s ∨ x = t) ∧
    (a > 0) ∧
    (∀ a' : ℝ, a' > 0 → a' ≥ a)) →
  a = 3 * Real.sqrt 3 ∧ b = 9 ∧ c = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_theorem_l3022_302297


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l3022_302246

def scores : List ℝ := [93, 87, 90, 94, 88, 92]

theorem arithmetic_mean_of_scores :
  (scores.sum / scores.length : ℝ) = 90.6667 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l3022_302246


namespace NUMINAMATH_CALUDE_tan_sum_given_sin_cos_sum_l3022_302272

theorem tan_sum_given_sin_cos_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 5/13)
  (h2 : Real.cos x + Real.cos y = 12/13) : 
  Real.tan x + Real.tan y = 240/119 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_given_sin_cos_sum_l3022_302272


namespace NUMINAMATH_CALUDE_magic_trick_always_succeeds_l3022_302206

/-- Represents a box in the magic trick setup -/
structure Box :=
  (index : Fin 13)

/-- Represents the state of the magic trick setup -/
structure MagicTrickSetup :=
  (boxes : Fin 13 → Box)
  (coin_boxes : Fin 2 → Box)
  (opened_box : Box)

/-- Represents the magician's strategy -/
structure MagicianStrategy :=
  (choose_boxes : MagicTrickSetup → Fin 4 → Box)

/-- Predicate to check if a strategy is successful -/
def is_successful_strategy (strategy : MagicianStrategy) : Prop :=
  ∀ (setup : MagicTrickSetup),
    ∃ (i j : Fin 4),
      strategy.choose_boxes setup i = setup.coin_boxes 0 ∧
      strategy.choose_boxes setup j = setup.coin_boxes 1

theorem magic_trick_always_succeeds :
  ∃ (strategy : MagicianStrategy), is_successful_strategy strategy := by
  sorry

end NUMINAMATH_CALUDE_magic_trick_always_succeeds_l3022_302206


namespace NUMINAMATH_CALUDE_quadratic_root_shift_l3022_302279

theorem quadratic_root_shift (a b c t : ℤ) (ha : a ≠ 0) :
  (a * t^2 + b * t + c = 0) →
  ∃ (p q r : ℤ), p ≠ 0 ∧ p * (t + 2)^2 + q * (t + 2) + r = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_shift_l3022_302279


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l3022_302201

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (10*x - 4) * (10*x) * (5*x + 15) = 1200 * k) ∧
  (∀ (m : ℤ), m > 1200 → ∃ (y : ℤ), Odd y ∧ ¬(∃ (l : ℤ), (10*y - 4) * (10*y) * (5*y + 15) = m * l)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l3022_302201


namespace NUMINAMATH_CALUDE_inverse_mod_53_l3022_302214

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 31) : (36⁻¹ : ZMod 53) = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l3022_302214


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3022_302230

theorem unique_quadratic_solution (a c : ℤ) : 
  (∃! x : ℝ, a * x^2 - 6 * x + c = 0) →  -- exactly one solution
  (a + c = 12) →                        -- sum condition
  (a < c) →                             -- order condition
  (a = 3 ∧ c = 9) :=                    -- unique solution
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3022_302230


namespace NUMINAMATH_CALUDE_tournament_games_l3022_302229

/-- The number of games played in a single-elimination tournament -/
def games_played (n : ℕ) : ℕ :=
  n - 1

/-- Theorem: In a single-elimination tournament with 32 teams, 31 games are played -/
theorem tournament_games : games_played 32 = 31 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_l3022_302229


namespace NUMINAMATH_CALUDE_electric_water_ratio_l3022_302263

def monthly_earnings : ℚ := 6000
def house_rental : ℚ := 640
def food_expense : ℚ := 380
def insurance_ratio : ℚ := 1 / 5
def remaining_money : ℚ := 2280

theorem electric_water_ratio :
  let insurance_cost := insurance_ratio * monthly_earnings
  let total_expenses := house_rental + food_expense + insurance_cost
  let electric_water_bill := monthly_earnings - total_expenses - remaining_money
  electric_water_bill / monthly_earnings = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_electric_water_ratio_l3022_302263
