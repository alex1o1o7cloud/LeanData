import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l2642_264209

theorem simplify_expression (a b : ℝ) : (2*a^2 - 3*a*b + 8) - (-a*b - a^2 + 8) = 3*a^2 - 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2642_264209


namespace NUMINAMATH_CALUDE_lego_castle_ratio_l2642_264269

/-- Proves that the ratio of Legos used for the castle to the total number of Legos is 1:2 --/
theorem lego_castle_ratio :
  let total_legos : ℕ := 500
  let legos_put_back : ℕ := 245
  let missing_legos : ℕ := 5
  let castle_legos : ℕ := total_legos - legos_put_back - missing_legos
  (castle_legos : ℚ) / total_legos = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_lego_castle_ratio_l2642_264269


namespace NUMINAMATH_CALUDE_sector_area_l2642_264296

/-- Given a sector with arc length 20 meters and diameter 24 meters,
    prove that its area is 120 square meters using the formula:
    area = (diameter * circumference) / 4 -/
theorem sector_area (arc_length diameter : ℝ) (h1 : arc_length = 20) (h2 : diameter = 24) :
  (diameter * arc_length) / 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2642_264296


namespace NUMINAMATH_CALUDE_rhombus_to_square_l2642_264260

/-- Represents a rhombus with diagonals d1 and d2 -/
structure Rhombus where
  d1 : ℝ
  d2 : ℝ
  d1_positive : d1 > 0
  d2_positive : d2 > 0
  d1_twice_d2 : d1 = 2 * d2

/-- Represents a square -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Represents the three parts that the rhombus is cut into -/
structure RhombusParts where
  part1 : Set ℝ
  part2 : Set ℝ
  part3 : Set ℝ

/-- Function to cut a rhombus into three parts -/
def cutRhombus (r : Rhombus) : RhombusParts :=
  sorry

/-- Function to form a square from three parts -/
def formSquare (parts : RhombusParts) : Square :=
  sorry

/-- Theorem stating that a rhombus with one diagonal twice the other 
    can be cut into 3 parts that form a square -/
theorem rhombus_to_square (r : Rhombus) : 
  ∃ (s : Square), formSquare (cutRhombus r) = s :=
  sorry

end NUMINAMATH_CALUDE_rhombus_to_square_l2642_264260


namespace NUMINAMATH_CALUDE_stratified_sampling_l2642_264246

theorem stratified_sampling (total_employees : ℕ) (total_sample : ℕ) (dept_employees : ℕ) :
  total_employees = 240 →
  total_sample = 20 →
  dept_employees = 60 →
  (dept_employees * total_sample) / total_employees = 5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2642_264246


namespace NUMINAMATH_CALUDE_part_one_part_two_l2642_264205

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| - |x + m|

-- Part 1
theorem part_one :
  ∀ x : ℝ, (f x 2 + 2 < 0) ↔ (x > 1/2) :=
sorry

-- Part 2
theorem part_two :
  (∀ x ∈ Set.Icc 0 2, f x m + |x - 4| > 0) ↔ m ∈ Set.Ioo (-4) 1 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2642_264205


namespace NUMINAMATH_CALUDE_reciprocal_and_absolute_value_l2642_264207

theorem reciprocal_and_absolute_value :
  (1 / (- (-2))) = 1/2 ∧ 
  {x : ℝ | |x| = 5} = {-5, 5} := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_absolute_value_l2642_264207


namespace NUMINAMATH_CALUDE_factory_production_l2642_264291

/-- Calculates the number of toys produced per day in a factory -/
def toys_per_day (total_toys : ℕ) (work_days : ℕ) : ℕ :=
  total_toys / work_days

theorem factory_production :
  let total_weekly_production := 6000
  let work_days_per_week := 4
  toys_per_day total_weekly_production work_days_per_week = 1500 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l2642_264291


namespace NUMINAMATH_CALUDE_third_angle_measure_l2642_264232

-- Define the triangle and its angles
def triangle_angles (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Theorem statement
theorem third_angle_measure :
  ∀ x : ℝ, triangle_angles 40 (3 * x) x → x = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_third_angle_measure_l2642_264232


namespace NUMINAMATH_CALUDE_cube_root_of_three_times_five_to_seven_l2642_264220

theorem cube_root_of_three_times_five_to_seven (x : ℝ) :
  x = (5^7 + 5^7 + 5^7)^(1/3) → x = 3^(1/3) * 5^(7/3) := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_three_times_five_to_seven_l2642_264220


namespace NUMINAMATH_CALUDE_f_symmetry_l2642_264268

/-- Given a function f(x) = x^3 + 2x, prove that f(a) + f(-a) = 0 for any real number a -/
theorem f_symmetry (a : ℝ) : (fun x : ℝ ↦ x^3 + 2*x) a + (fun x : ℝ ↦ x^3 + 2*x) (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l2642_264268


namespace NUMINAMATH_CALUDE_other_ticket_cost_l2642_264224

/-- Given a total of 29 tickets, with 11 tickets costing $9 each,
    and a total cost of $225 for all tickets,
    prove that the remaining tickets cost $7 each. -/
theorem other_ticket_cost (total_tickets : ℕ) (nine_dollar_tickets : ℕ) 
  (total_cost : ℕ) (h1 : total_tickets = 29) (h2 : nine_dollar_tickets = 11) 
  (h3 : total_cost = 225) : 
  (total_cost - nine_dollar_tickets * 9) / (total_tickets - nine_dollar_tickets) = 7 :=
by sorry

end NUMINAMATH_CALUDE_other_ticket_cost_l2642_264224


namespace NUMINAMATH_CALUDE_max_value_of_f_l2642_264286

noncomputable def f (t : ℝ) : ℝ := (3^t - 4*t)*t / 9^t

theorem max_value_of_f :
  ∃ (max : ℝ), max = 1/16 ∧ ∀ (t : ℝ), f t ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2642_264286


namespace NUMINAMATH_CALUDE_base_eight_53_equals_43_l2642_264255

/-- Converts a two-digit base-eight number to base-ten. -/
def baseEightToBaseTen (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  tens * 8 + ones

/-- The base-eight number 53 is equal to 43 in base-ten. -/
theorem base_eight_53_equals_43 : baseEightToBaseTen 53 = 43 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_53_equals_43_l2642_264255


namespace NUMINAMATH_CALUDE_tree_break_height_l2642_264289

theorem tree_break_height (tree_height road_width break_height : ℝ) 
  (h_tree : tree_height = 36)
  (h_road : road_width = 12)
  (h_pythagoras : (tree_height - break_height)^2 = break_height^2 + road_width^2) :
  break_height = 16 := by
sorry

end NUMINAMATH_CALUDE_tree_break_height_l2642_264289


namespace NUMINAMATH_CALUDE_power_calculation_l2642_264263

theorem power_calculation : 16^4 * 8^2 / 4^10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2642_264263


namespace NUMINAMATH_CALUDE_ellipse_sum_a_k_l2642_264245

def ellipse (h k a b : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

def focus1 : ℝ × ℝ := (2, 2)
def focus2 : ℝ × ℝ := (2, 6)
def point_on_ellipse : ℝ × ℝ := (-3, 4)

theorem ellipse_sum_a_k (h k a b : ℝ) :
  a > 0 → b > 0 →
  ellipse h k a b (point_on_ellipse.1) (point_on_ellipse.2) →
  (∀ x y, ellipse h k a b x y →
    Real.sqrt ((x - focus1.1)^2 + (y - focus1.2)^2) +
    Real.sqrt ((x - focus2.1)^2 + (y - focus2.2)^2) =
    Real.sqrt ((point_on_ellipse.1 - focus1.1)^2 + (point_on_ellipse.2 - focus1.2)^2) +
    Real.sqrt ((point_on_ellipse.1 - focus2.1)^2 + (point_on_ellipse.2 - focus2.2)^2)) →
  a + k = (Real.sqrt 29 + 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_a_k_l2642_264245


namespace NUMINAMATH_CALUDE_mike_gave_ten_books_l2642_264236

/-- The number of books Mike gave to Lily -/
def books_from_mike : ℕ := sorry

/-- The number of books Corey gave to Lily -/
def books_from_corey : ℕ := sorry

/-- The total number of books Lily received -/
def total_books : ℕ := 35

theorem mike_gave_ten_books :
  (books_from_mike = 10) ∧
  (books_from_corey = books_from_mike + 15) ∧
  (books_from_mike + books_from_corey = total_books) :=
sorry

end NUMINAMATH_CALUDE_mike_gave_ten_books_l2642_264236


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l2642_264259

/-- Represents a recurring decimal with a single digit repeating -/
def RecurringDecimal (d : ℕ) : ℚ :=
  d / 9

theorem recurring_decimal_sum :
  let a := RecurringDecimal 5
  let b := RecurringDecimal 1
  let c := RecurringDecimal 3
  let d := RecurringDecimal 6
  a + b - c + d = 1 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l2642_264259


namespace NUMINAMATH_CALUDE_solve_x_l2642_264228

def symbol_value (a b c d : ℤ) : ℤ := a * d - b * c

theorem solve_x : ∃ x : ℤ, symbol_value (x - 1) 2 3 (-5) = 9 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_solve_x_l2642_264228


namespace NUMINAMATH_CALUDE_total_tagged_numbers_l2642_264267

def card_sum (w x y z : ℕ) : ℕ := w + x + y + z

theorem total_tagged_numbers : ∃ (w x y z : ℕ),
  w = 200 ∧
  x = w / 2 ∧
  y = x + w ∧
  z = 400 ∧
  card_sum w x y z = 1000 := by
sorry

end NUMINAMATH_CALUDE_total_tagged_numbers_l2642_264267


namespace NUMINAMATH_CALUDE_remainder_of_binary_number_div_8_l2642_264254

def binary_number : ℕ := 0b100101110011

theorem remainder_of_binary_number_div_8 :
  binary_number % 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_binary_number_div_8_l2642_264254


namespace NUMINAMATH_CALUDE_lower_limit_of_range_l2642_264273

theorem lower_limit_of_range (x : ℕ) : 
  x ≤ 100 ∧ 
  (∃ (S : Finset ℕ), S.card = 13 ∧ 
    (∀ n ∈ S, x ≤ n ∧ n ≤ 100 ∧ n % 6 = 0) ∧
    (∀ n, x ≤ n ∧ n ≤ 100 ∧ n % 6 = 0 → n ∈ S)) →
  x = 24 :=
by sorry

end NUMINAMATH_CALUDE_lower_limit_of_range_l2642_264273


namespace NUMINAMATH_CALUDE_calculate_b_amount_l2642_264235

/-- Given a total amount and the ratio between two parts, calculate the second part -/
theorem calculate_b_amount (total : ℚ) (a b : ℚ) (h1 : a + b = total) (h2 : 2/3 * a = 1/2 * b) : 
  b = 691.43 := by
  sorry

end NUMINAMATH_CALUDE_calculate_b_amount_l2642_264235


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2642_264240

theorem smaller_number_problem (x y : ℤ) 
  (h1 : x = 2 * y - 3) 
  (h2 : x + y = 51) : 
  min x y = 18 := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2642_264240


namespace NUMINAMATH_CALUDE_no_return_after_2020_rounds_l2642_264234

/-- Represents the state of the ball game -/
structure BallState :=
  (N : ℕ)           -- Total number of boxes
  (position : ℕ)    -- Current position of the ball (1 ≤ position ≤ N)
  (number : ℕ)      -- Current number on the ball (1 ≤ number ≤ N)

/-- Moves the ball according to the game rules -/
def move (state : BallState) : BallState :=
  { N := state.N,
    position := (state.position + state.number - 1) % state.N + 1,
    number := if state.number < state.N then state.number + 1 else 1 }

/-- Applies the move function k times -/
def moveKTimes (k : ℕ) (state : BallState) : BallState :=
  match k with
  | 0 => state
  | k + 1 => move (moveKTimes k state)

/-- Theorem stating that it's impossible to return to the initial state after 2020 rounds -/
theorem no_return_after_2020_rounds (N : ℕ) (initial_position : ℕ) (initial_number : ℕ)
    (hN : N > 0)
    (hpos : initial_position > 0 ∧ initial_position ≤ N)
    (hnum : initial_number > 0 ∧ initial_number ≤ N) :
    let initial_state : BallState := ⟨N, initial_position, initial_number⟩
    moveKTimes 2020 initial_state ≠ initial_state :=
  sorry

end NUMINAMATH_CALUDE_no_return_after_2020_rounds_l2642_264234


namespace NUMINAMATH_CALUDE_mitchs_family_milk_consumption_l2642_264217

/-- The total milk consumption of Mitch's family in one week -/
def total_milk_consumption (regular_milk soy_milk almond_milk oat_milk : ℝ) : ℝ :=
  regular_milk + soy_milk + almond_milk + oat_milk

/-- Theorem stating the total milk consumption of Mitch's family -/
theorem mitchs_family_milk_consumption :
  total_milk_consumption 1.75 0.85 1.25 0.65 = 4.50 := by
  sorry

end NUMINAMATH_CALUDE_mitchs_family_milk_consumption_l2642_264217


namespace NUMINAMATH_CALUDE_max_ratio_three_digit_number_l2642_264262

theorem max_ratio_three_digit_number :
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 →
    0 ≤ b ∧ b ≤ 9 →
    0 ≤ c ∧ c ≤ 9 →
    let N := 100 * a + 10 * b + c
    let S := a + b + c
    (N : ℚ) / S ≤ 100 ∧ 
    (∃ a' b' c', 
      1 ≤ a' ∧ a' ≤ 9 ∧ 
      0 ≤ b' ∧ b' ≤ 9 ∧ 
      0 ≤ c' ∧ c' ≤ 9 ∧ 
      let N' := 100 * a' + 10 * b' + c'
      let S' := a' + b' + c'
      (N' : ℚ) / S' = 100) :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_three_digit_number_l2642_264262


namespace NUMINAMATH_CALUDE_two_machines_total_copies_l2642_264214

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  rate : ℕ  -- copies per minute

/-- Calculates the number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Represents the problem setup with two copy machines -/
structure TwoMachinesProblem where
  machine1 : CopyMachine
  machine2 : CopyMachine
  time : ℕ  -- in minutes

/-- The main theorem to be proved -/
theorem two_machines_total_copies 
  (problem : TwoMachinesProblem) 
  (h1 : problem.machine1.rate = 25)
  (h2 : problem.machine2.rate = 55)
  (h3 : problem.time = 30) : 
  copies_made problem.machine1 problem.time + copies_made problem.machine2 problem.time = 2400 :=
by sorry

end NUMINAMATH_CALUDE_two_machines_total_copies_l2642_264214


namespace NUMINAMATH_CALUDE_root_triple_relation_l2642_264201

theorem root_triple_relation (p q r : ℝ) (h : ∃ x y : ℝ, p * x^2 + q * x + r = 0 ∧ p * y^2 + q * y + r = 0 ∧ y = 3 * x) :
  3 * q^2 = 8 * p * r := by
sorry

end NUMINAMATH_CALUDE_root_triple_relation_l2642_264201


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2642_264294

/-- A quadratic function that takes specific values for consecutive natural numbers. -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = -9 ∧ f (n + 1) = -9 ∧ f (n + 2) = -15

/-- The maximum value of a quadratic function with the given properties. -/
theorem quadratic_max_value (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = -33/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2642_264294


namespace NUMINAMATH_CALUDE_race_finish_times_l2642_264264

/-- Race parameters and runner speeds -/
def race_distance : ℝ := 15
def malcolm_speed : ℝ := 5
def joshua_speed : ℝ := 7
def emily_speed : ℝ := 6

/-- Calculate finish time for a runner given their speed -/
def finish_time (speed : ℝ) : ℝ := race_distance * speed

/-- Calculate time difference between two runners -/
def time_difference (speed1 speed2 : ℝ) : ℝ := finish_time speed1 - finish_time speed2

/-- Theorem stating the time differences for Joshua and Emily relative to Malcolm -/
theorem race_finish_times :
  (time_difference joshua_speed malcolm_speed = 30) ∧
  (time_difference emily_speed malcolm_speed = 15) := by
  sorry

end NUMINAMATH_CALUDE_race_finish_times_l2642_264264


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l2642_264226

theorem quadratic_equation_general_form :
  ∃ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 3 ∧
  ∀ x, (x - 1)^2 = 3*x - 2 ↔ a*x^2 + b*x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l2642_264226


namespace NUMINAMATH_CALUDE_triangle_angle_A_l2642_264272

theorem triangle_angle_A (b c S_ABC : ℝ) (h1 : b = 8) (h2 : c = 8 * Real.sqrt 3)
  (h3 : S_ABC = 16 * Real.sqrt 3) :
  let A := Real.arcsin (1 / 2)
  A = π / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l2642_264272


namespace NUMINAMATH_CALUDE_negative_square_times_cube_l2642_264257

theorem negative_square_times_cube (x : ℝ) : (-x)^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_times_cube_l2642_264257


namespace NUMINAMATH_CALUDE_fayes_carrots_l2642_264283

/-- Proof of the number of carrots Faye picked -/
theorem fayes_carrots (good_carrots bad_carrots moms_carrots : ℕ) 
  (h1 : good_carrots = 12)
  (h2 : bad_carrots = 16)
  (h3 : moms_carrots = 5) :
  good_carrots + bad_carrots - moms_carrots = 23 := by
  sorry

end NUMINAMATH_CALUDE_fayes_carrots_l2642_264283


namespace NUMINAMATH_CALUDE_inequalities_for_negative_numbers_l2642_264237

theorem inequalities_for_negative_numbers (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧ (Real.sqrt (-a) > Real.sqrt (-b)) ∧ (abs a > -b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_negative_numbers_l2642_264237


namespace NUMINAMATH_CALUDE_xyz_inequality_l2642_264275

theorem xyz_inequality (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) :
  x * y > x * z :=
by sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2642_264275


namespace NUMINAMATH_CALUDE_tangent_perpendicular_condition_l2642_264204

/-- The function f(x) = x³ - x² + ax + b -/
def f (a b x : ℝ) : ℝ := x^3 - x^2 + a*x + b

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_perpendicular_condition (a b : ℝ) : 
  (f_derivative a 1) * 2 = -1 ↔ a = -3/2 := by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_condition_l2642_264204


namespace NUMINAMATH_CALUDE_swimming_practice_months_l2642_264218

theorem swimming_practice_months (total_required : ℕ) (completed : ℕ) (monthly_practice : ℕ) : 
  total_required = 1500 →
  completed = 180 →
  monthly_practice = 220 →
  (total_required - completed) / monthly_practice = 6 := by
sorry

end NUMINAMATH_CALUDE_swimming_practice_months_l2642_264218


namespace NUMINAMATH_CALUDE_divisibility_property_l2642_264256

theorem divisibility_property (p : ℕ) (hp : p > 3) (hodd : Odd p) :
  ∃ k : ℤ, (p - 3) ^ ((p - 1) / 2) - 1 = k * (p - 4) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2642_264256


namespace NUMINAMATH_CALUDE_binomial_square_constant_l2642_264225

theorem binomial_square_constant (a : ℝ) : 
  (∃ b c : ℝ, ∀ x, 9*x^2 - 27*x + a = (b*x + c)^2) → a = 20.25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l2642_264225


namespace NUMINAMATH_CALUDE_arcsin_one_half_l2642_264206

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_l2642_264206


namespace NUMINAMATH_CALUDE_system_solution_l2642_264284

theorem system_solution :
  ∃ (x y : ℚ),
    (16 * x^2 + 8 * x * y + 4 * y^2 + 20 * x + 2 * y = -7) ∧
    (8 * x^2 - 16 * x * y + 2 * y^2 + 20 * x - 14 * y = -11) ∧
    (x = -3/4) ∧ (y = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2642_264284


namespace NUMINAMATH_CALUDE_desiree_age_is_six_l2642_264261

/-- Desiree's current age -/
def desiree_age : ℕ := sorry

/-- Desiree's cousin's current age -/
def cousin_age : ℕ := sorry

/-- Desiree's age is twice her cousin's age -/
axiom desiree_twice_cousin : desiree_age = 2 * cousin_age

/-- In 30 years, Desiree's age will be 14 years more than 2/3 of her cousin's age -/
axiom future_age_relation : desiree_age + 30 = (2/3 : ℚ) * (cousin_age + 30 : ℚ) + 14

/-- Theorem stating Desiree's current age is 6 -/
theorem desiree_age_is_six : desiree_age = 6 := by sorry

end NUMINAMATH_CALUDE_desiree_age_is_six_l2642_264261


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2642_264297

/-- Represents the total number of students in a school -/
def total_students : ℕ := 3500 + 1500

/-- Represents the number of middle school students -/
def middle_school_students : ℕ := 1500

/-- Represents the number of students sampled from the middle school stratum -/
def middle_school_sample : ℕ := 30

/-- Calculates the total sample size in a stratified sampling -/
def total_sample_size : ℕ := (middle_school_sample * total_students) / middle_school_students

theorem stratified_sampling_theorem :
  total_sample_size = 100 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2642_264297


namespace NUMINAMATH_CALUDE_remainder_of_3_500_mod_17_l2642_264202

theorem remainder_of_3_500_mod_17 (h1 : Nat.Prime 17) (h2 : ¬(17 ∣ 3)) : 
  3^500 % 17 = 13 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_3_500_mod_17_l2642_264202


namespace NUMINAMATH_CALUDE_volleyball_preference_percentage_l2642_264252

theorem volleyball_preference_percentage
  (north_students : ℕ)
  (south_students : ℕ)
  (north_volleyball_percentage : ℚ)
  (south_volleyball_percentage : ℚ)
  (h1 : north_students = 1800)
  (h2 : south_students = 2700)
  (h3 : north_volleyball_percentage = 25 / 100)
  (h4 : south_volleyball_percentage = 35 / 100)
  : (north_students * north_volleyball_percentage + south_students * south_volleyball_percentage) /
    (north_students + south_students) = 31 / 100 := by
  sorry


end NUMINAMATH_CALUDE_volleyball_preference_percentage_l2642_264252


namespace NUMINAMATH_CALUDE_train_length_l2642_264227

/-- The length of a train given its speed, platform length, and time to cross the platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5/18) →
  platform_length = 240 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 280 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2642_264227


namespace NUMINAMATH_CALUDE_fred_money_left_l2642_264270

/-- Calculates the amount of money Fred has left after spending half his allowance on movies and earning money from washing a car. -/
def money_left (allowance : ℕ) (car_wash_earnings : ℕ) : ℕ :=
  allowance / 2 + car_wash_earnings

/-- Proves that Fred has 14 dollars left given his allowance and car wash earnings. -/
theorem fred_money_left :
  money_left 16 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fred_money_left_l2642_264270


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_with_5_hundreds_l2642_264247

theorem largest_multiple_of_12_with_5_hundreds : ∃ (n : ℕ), 
  (n = 588) ∧ 
  (12 ∣ n) ∧ 
  (100 ≤ n) ∧ (n < 1000) ∧
  (n / 100 = 5) ∧
  (∀ m : ℕ, (12 ∣ m) ∧ (100 ≤ m) ∧ (m < 1000) ∧ (m / 100 = 5) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_with_5_hundreds_l2642_264247


namespace NUMINAMATH_CALUDE_computer_distribution_l2642_264244

def distribute_computers (n : ℕ) (k : ℕ) (min : ℕ) : ℕ :=
  -- The number of ways to distribute n identical items among k recipients,
  -- with each recipient receiving at least min items
  sorry

theorem computer_distribution :
  distribute_computers 9 3 2 = 10 := by sorry

end NUMINAMATH_CALUDE_computer_distribution_l2642_264244


namespace NUMINAMATH_CALUDE_number_puzzle_l2642_264288

theorem number_puzzle : ∃ x : ℝ, x^2 + 50 = (x - 10)^2 ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2642_264288


namespace NUMINAMATH_CALUDE_no_integer_roots_for_odd_coefficients_l2642_264250

theorem no_integer_roots_for_odd_coefficients (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) : 
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_for_odd_coefficients_l2642_264250


namespace NUMINAMATH_CALUDE_axis_of_symmetry_translated_trig_l2642_264210

/-- The axis of symmetry of a translated trigonometric function -/
theorem axis_of_symmetry_translated_trig (k : ℤ) :
  let f (x : ℝ) := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)
  let g (x : ℝ) := f (x + π / 6)
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), 
    g x = A * Real.sin (B * x + C) ∧
    (x = k * π / 2 - π / 12) → (B * x + C = n * π + π / 2) :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_translated_trig_l2642_264210


namespace NUMINAMATH_CALUDE_negative_abs_of_negative_one_l2642_264299

theorem negative_abs_of_negative_one : -|-1| = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_of_negative_one_l2642_264299


namespace NUMINAMATH_CALUDE_razorback_shop_profit_l2642_264266

/-- Calculates the total profit from selling various items in the Razorback shop -/
def total_profit (jersey_profit t_shirt_profit hoodie_profit hat_profit : ℕ)
                 (jerseys_sold t_shirts_sold hoodies_sold hats_sold : ℕ) : ℕ :=
  jersey_profit * jerseys_sold +
  t_shirt_profit * t_shirts_sold +
  hoodie_profit * hoodies_sold +
  hat_profit * hats_sold

/-- The total profit from the Razorback shop during the Arkansas and Texas Tech game -/
theorem razorback_shop_profit :
  total_profit 76 204 132 48 2 158 75 120 = 48044 := by
  sorry

end NUMINAMATH_CALUDE_razorback_shop_profit_l2642_264266


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2642_264230

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2642_264230


namespace NUMINAMATH_CALUDE_sqrt_144_times_3_squared_l2642_264281

theorem sqrt_144_times_3_squared : (Real.sqrt 144 * 3) ^ 2 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_144_times_3_squared_l2642_264281


namespace NUMINAMATH_CALUDE_license_plate_count_l2642_264233

/-- The number of possible digits (0 to 9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A to Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def num_plate_digits : ℕ := 6

/-- The number of letters in a license plate -/
def num_plate_letters : ℕ := 3

/-- The number of possible positions for the letter block -/
def num_block_positions : ℕ := num_plate_digits + 1

/-- The total number of possible distinct license plates -/
def total_license_plates : ℕ := 
  num_block_positions * (num_digits ^ num_plate_digits) * (num_letters ^ num_plate_letters)

theorem license_plate_count : total_license_plates = 122504000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2642_264233


namespace NUMINAMATH_CALUDE_smallest_survey_size_l2642_264239

theorem smallest_survey_size : ∀ n : ℕ, 
  n > 0 → 
  (∃ y n_yes n_no : ℕ, 
    n_yes = (76 * n) / 100 ∧ 
    n_no = (24 * n) / 100 ∧ 
    n_yes + n_no = n) → 
  n ≥ 25 := by sorry

end NUMINAMATH_CALUDE_smallest_survey_size_l2642_264239


namespace NUMINAMATH_CALUDE_age_difference_l2642_264258

/-- Given that B is currently 42 years old, and in 10 years A will be twice as old as B was 10 years ago, prove that A is 12 years older than B. -/
theorem age_difference (A B : ℕ) : B = 42 → A + 10 = 2 * (B - 10) → A - B = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2642_264258


namespace NUMINAMATH_CALUDE_not_necessarily_linear_l2642_264216

open Set MeasureTheory

-- Define the type of real-valued functions
def RealFunction := ℝ → ℝ

-- Define the Minkowski sum of graphs
def minkowskiSumGraphs (f g : RealFunction) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x + y, f x + g y)}

-- State the theorem
theorem not_necessarily_linear :
  ∃ (f g : RealFunction),
    Continuous f ∧
    (volume (minkowskiSumGraphs f g) = 0) ∧
    ¬∃ (a b : ℝ), ∀ x, f x = a * x + b :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_linear_l2642_264216


namespace NUMINAMATH_CALUDE_tangent_line_hyperbola_l2642_264200

/-- The equation of the tangent line to the hyperbola x^2 - y^2/2 = 1 at the point (√2, √2) is 2x - y - √2 = 0 -/
theorem tangent_line_hyperbola (x y : ℝ) :
  (x^2 - y^2/2 = 1) →
  let P : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
  let tangent_line := fun (x y : ℝ) ↦ 2*x - y - Real.sqrt 2 = 0
  (x = P.1 ∧ y = P.2) →
  tangent_line x y :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_hyperbola_l2642_264200


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l2642_264243

def blocks_west : ℕ := 9
def blocks_south : ℕ := 15
def mile_per_block : ℚ := 1/4

theorem arthur_walk_distance :
  (blocks_west + blocks_south : ℚ) * mile_per_block = 6 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l2642_264243


namespace NUMINAMATH_CALUDE_ruths_track_length_l2642_264211

theorem ruths_track_length (sean_piece_length ruth_piece_length total_length : ℝ) :
  sean_piece_length = 8 →
  total_length = 72 →
  (total_length / sean_piece_length) * sean_piece_length = (total_length / ruth_piece_length) * ruth_piece_length →
  ruth_piece_length = 8 :=
by sorry

end NUMINAMATH_CALUDE_ruths_track_length_l2642_264211


namespace NUMINAMATH_CALUDE_hexadecimal_to_decimal_l2642_264293

/-- Given that the hexadecimal number (10k5)₆ (where k is a positive integer) 
    is equivalent to the decimal number 239, prove that k = 3. -/
theorem hexadecimal_to_decimal (k : ℕ+) : 
  (1 * 6^3 + k * 6 + 5) = 239 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexadecimal_to_decimal_l2642_264293


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l2642_264222

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ

/-- Calculates the number of sampled elements within the given interval -/
def elements_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) * s.sample_size + s.population - 1) / s.population

/-- The main theorem stating that for the given scenario, 12 people fall within the interval -/
theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.population = 840)
  (h2 : s.sample_size = 42)
  (h3 : s.interval_start = 481)
  (h4 : s.interval_end = 720) :
  elements_in_interval s = 12 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l2642_264222


namespace NUMINAMATH_CALUDE_not_both_perfect_squares_l2642_264290

theorem not_both_perfect_squares (x y z t : ℕ+) 
  (h1 : x.val * y.val - z.val * t.val = x.val + y.val)
  (h2 : x.val + y.val = z.val + t.val) :
  ¬(∃ (a c : ℕ), x.val * y.val = a^2 ∧ z.val * t.val = c^2) :=
by sorry

end NUMINAMATH_CALUDE_not_both_perfect_squares_l2642_264290


namespace NUMINAMATH_CALUDE_xy_addition_identity_l2642_264241

theorem xy_addition_identity (x y : ℝ) : -x*y - x*y = -2*(x*y) := by
  sorry

end NUMINAMATH_CALUDE_xy_addition_identity_l2642_264241


namespace NUMINAMATH_CALUDE_guitar_savings_l2642_264285

/-- The suggested retail price of the guitar -/
def suggested_price : ℝ := 1000

/-- The discount percentage offered by Guitar Center -/
def gc_discount : ℝ := 0.15

/-- The shipping fee charged by Guitar Center -/
def gc_shipping : ℝ := 100

/-- The discount percentage offered by Sweetwater -/
def sw_discount : ℝ := 0.10

/-- The cost of the guitar at Guitar Center -/
def gc_cost : ℝ := suggested_price * (1 - gc_discount) + gc_shipping

/-- The cost of the guitar at Sweetwater -/
def sw_cost : ℝ := suggested_price * (1 - sw_discount)

/-- The savings when buying from the cheaper store (Sweetwater) -/
theorem guitar_savings : gc_cost - sw_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_guitar_savings_l2642_264285


namespace NUMINAMATH_CALUDE_circle_parabola_intersection_l2642_264212

/-- The number of intersection points between a circle and a parabola -/
def intersection_count (b : ℝ) : ℕ :=
  sorry

/-- The curves x^2 + y^2 = b^2 and y = x^2 - b + 1 intersect at exactly 4 points
    if and only if b > 2 -/
theorem circle_parabola_intersection (b : ℝ) :
  intersection_count b = 4 ↔ b > 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_parabola_intersection_l2642_264212


namespace NUMINAMATH_CALUDE_root_equation_implies_b_equals_four_l2642_264271

theorem root_equation_implies_b_equals_four
  (a b c : ℕ)
  (ha : a > 1)
  (hb : b > 1)
  (hc : c > 1)
  (h : ∀ (N : ℝ), N ≠ 1 → (N^3 * (N^2 * N^(1/c))^(1/b))^(1/a) = N^(39/48)) :
  b = 4 :=
sorry

end NUMINAMATH_CALUDE_root_equation_implies_b_equals_four_l2642_264271


namespace NUMINAMATH_CALUDE_product_remainder_l2642_264231

def sequence_product : ℕ → ℕ
  | 0 => 3
  | n + 1 => sequence_product n * (3 + 10 * (n + 1))

def sequence_length : ℕ := (93 - 3) / 10 + 1

theorem product_remainder (n : ℕ) : 
  n = sequence_length - 1 → sequence_product n ≡ 4 [MOD 7] :=
by sorry

end NUMINAMATH_CALUDE_product_remainder_l2642_264231


namespace NUMINAMATH_CALUDE_profit_percentage_l2642_264253

theorem profit_percentage (selling_price cost_price : ℝ) (h : cost_price = 0.95 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 95 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l2642_264253


namespace NUMINAMATH_CALUDE_cos_sum_fifth_circle_l2642_264238

theorem cos_sum_fifth_circle : Real.cos (2 * Real.pi / 5) + Real.cos (4 * Real.pi / 5) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_fifth_circle_l2642_264238


namespace NUMINAMATH_CALUDE_exists_p_with_conditions_l2642_264276

theorem exists_p_with_conditions : ∃ p : ℕ+, 
  ∃ q r s : ℕ+,
  (Nat.gcd p q = 40) ∧
  (Nat.gcd q r = 45) ∧
  (Nat.gcd r s = 60) ∧
  (∃ k : ℕ+, Nat.gcd s p = 10 * k ∧ k ≥ 10 ∧ k < 100) ∧
  (∃ m : ℕ+, p = 7 * m) := by
sorry

end NUMINAMATH_CALUDE_exists_p_with_conditions_l2642_264276


namespace NUMINAMATH_CALUDE_existence_of_non_divisible_k_l2642_264229

theorem existence_of_non_divisible_k (a b c n : ℤ) (h : n ≥ 3) :
  ∃ k : ℤ, ¬(n ∣ (k + a)) ∧ ¬(n ∣ (k + b)) ∧ ¬(n ∣ (k + c)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_divisible_k_l2642_264229


namespace NUMINAMATH_CALUDE_lighter_ball_problem_l2642_264215

/-- Represents the maximum number of balls that can be checked in a given number of weighings -/
def max_balls (weighings : ℕ) : ℕ := 3^weighings

/-- The problem statement -/
theorem lighter_ball_problem (n : ℕ) :
  (∀ m : ℕ, m > n → max_balls 5 < m) →
  (∃ strategy : Unit, true) →  -- placeholder for the existence of a strategy
  n ≤ max_balls 5 :=
sorry

end NUMINAMATH_CALUDE_lighter_ball_problem_l2642_264215


namespace NUMINAMATH_CALUDE_inequality_proof_l2642_264265

-- Define the logarithm function with base 1/8
noncomputable def log_base_1_8 (x : ℝ) : ℝ := Real.log x / Real.log (1/8)

-- State the theorem
theorem inequality_proof (x : ℝ) (h1 : x ≥ 1/2) (h2 : x < 1) :
  9.244 * Real.sqrt (1 - 9 * (log_base_1_8 x)^2) > 1 - 4 * log_base_1_8 x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2642_264265


namespace NUMINAMATH_CALUDE_notebook_cost_l2642_264277

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (s n c : Nat),
  -- Total number of students
  total_students = 42 ∧
  -- Majority of students bought notebooks
  s > total_students / 2 ∧
  -- Number of notebooks per student is greater than 2
  n > 2 ∧
  -- Cost in cents is greater than number of notebooks
  c > n ∧
  -- Total cost equation
  s * n * c = total_cost ∧
  -- Given total cost
  total_cost = 2773 →
  -- Conclusion: cost of a notebook is 103 cents
  c = 103 :=
sorry

end NUMINAMATH_CALUDE_notebook_cost_l2642_264277


namespace NUMINAMATH_CALUDE_tower_divisibility_l2642_264248

-- Define the tower of 2's function
def f : ℕ → ℕ
| 0 => 1
| (n + 1) => 2^(f n)

-- State the theorem
theorem tower_divisibility (n : ℕ) (h : n ≥ 2) : 
  n ∣ (f n - f (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_tower_divisibility_l2642_264248


namespace NUMINAMATH_CALUDE_integer_solutions_equation_l2642_264251

theorem integer_solutions_equation (n m : ℤ) : 
  n^6 + 3*n^5 + 3*n^4 + 2*n^3 + 3*n^2 + 3*n + 1 = m^3 ↔ (n = 0 ∧ m = 1) ∨ (n = -1 ∧ m = 0) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_equation_l2642_264251


namespace NUMINAMATH_CALUDE_nine_triangles_perimeter_l2642_264280

theorem nine_triangles_perimeter (large_perimeter : ℝ) (num_small_triangles : ℕ) 
  (h1 : large_perimeter = 120)
  (h2 : num_small_triangles = 9) :
  ∃ (small_perimeter : ℝ), 
    small_perimeter * num_small_triangles = large_perimeter ∧ 
    small_perimeter = 40 := by
  sorry

end NUMINAMATH_CALUDE_nine_triangles_perimeter_l2642_264280


namespace NUMINAMATH_CALUDE_four_painters_work_days_l2642_264208

/-- The number of work-days required for a given number of painters to complete a job -/
def work_days (num_painters : ℕ) (total_work : ℚ) : ℚ :=
  total_work / num_painters

theorem four_painters_work_days :
  let total_work : ℚ := 6 * (3/2)  -- 6 painters * 1.5 days
  (work_days 4 total_work) = 2 + (1/4) := by sorry

end NUMINAMATH_CALUDE_four_painters_work_days_l2642_264208


namespace NUMINAMATH_CALUDE_parabola_point_comparison_l2642_264274

/-- Theorem: For a parabola y = ax^2 - 4ax + 2 where a > 0, 
    and points (-1, y₁) and (1, y₂) on the parabola, y₁ > y₂ -/
theorem parabola_point_comparison 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (y₁ y₂ : ℝ) 
  (h_y₁ : y₁ = a * (-1)^2 - 4 * a * (-1) + 2) 
  (h_y₂ : y₂ = a * 1^2 - 4 * a * 1 + 2) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_comparison_l2642_264274


namespace NUMINAMATH_CALUDE_james_carrot_sticks_l2642_264295

/-- The number of carrot sticks James ate before dinner -/
def before_dinner : ℕ := 22

/-- The number of carrot sticks James ate after dinner -/
def after_dinner : ℕ := 15

/-- The total number of carrot sticks James ate -/
def total_carrot_sticks : ℕ := before_dinner + after_dinner

theorem james_carrot_sticks : total_carrot_sticks = 37 := by
  sorry

end NUMINAMATH_CALUDE_james_carrot_sticks_l2642_264295


namespace NUMINAMATH_CALUDE_prob_even_sum_is_31_66_l2642_264279

/-- A set of twelve prime numbers including two even primes -/
def prime_set : Finset ℕ := sorry

/-- The number of prime numbers in the set -/
def n : ℕ := 12

/-- The number of even prime numbers in the set -/
def even_primes : ℕ := 2

/-- The number of primes to be selected -/
def k : ℕ := 5

/-- Predicate to check if a set of natural numbers has an even sum -/
def has_even_sum (s : Finset ℕ) : Prop := Even (s.sum id)

/-- The probability of selecting k primes from prime_set with two even primes such that their sum is even -/
def prob_even_sum : ℚ := sorry

theorem prob_even_sum_is_31_66 : prob_even_sum = 31 / 66 := by sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_31_66_l2642_264279


namespace NUMINAMATH_CALUDE_rectangle_count_l2642_264219

theorem rectangle_count (a : ℝ) (ha : a > 0) : 
  ∃! (x y : ℝ), x < 2*a ∧ y < 2*a ∧ 
  2*(x + y) = 2*((2*a + 3*a) * (2/3)) ∧ 
  x*y = (2*a * 3*a) * (2/9) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_count_l2642_264219


namespace NUMINAMATH_CALUDE_solve_plane_problem_l2642_264223

def plane_problem (distance : ℝ) (time_with_wind : ℝ) (time_against_wind : ℝ) : Prop :=
  ∃ (plane_speed : ℝ) (wind_speed : ℝ),
    (plane_speed + wind_speed) * time_with_wind = distance ∧
    (plane_speed - wind_speed) * time_against_wind = distance ∧
    plane_speed = 262.5

theorem solve_plane_problem :
  plane_problem 900 3 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_plane_problem_l2642_264223


namespace NUMINAMATH_CALUDE_original_workers_count_l2642_264242

/-- Given a work that can be completed by an unknown number of workers in 45 days,
    and that adding 10 workers allows the work to be completed in 35 days,
    prove that the original number of workers is 35. -/
theorem original_workers_count (work : ℝ) (h1 : work > 0) : ∃ (workers : ℕ),
  (workers : ℝ) * 45 = work ∧
  (workers + 10 : ℝ) * 35 = work ∧
  workers = 35 := by
sorry

end NUMINAMATH_CALUDE_original_workers_count_l2642_264242


namespace NUMINAMATH_CALUDE_sqrt_one_half_equals_sqrt_two_over_two_l2642_264282

theorem sqrt_one_half_equals_sqrt_two_over_two : 
  Real.sqrt (1/2) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_one_half_equals_sqrt_two_over_two_l2642_264282


namespace NUMINAMATH_CALUDE_slope_of_line_l2642_264298

/-- The slope of a line given by the equation √3x - y + 1 = 0 is √3. -/
theorem slope_of_line (x y : ℝ) : 
  (Real.sqrt 3) * x - y + 1 = 0 → 
  ∃ m : ℝ, m = Real.sqrt 3 ∧ y = m * x + 1 := by
sorry

end NUMINAMATH_CALUDE_slope_of_line_l2642_264298


namespace NUMINAMATH_CALUDE_annual_cost_difference_l2642_264221

/-- Calculates the annual cost difference between combined piano, violin, and singing lessons
    and clarinet lessons, given the hourly rates and weekly hours for each lesson type. -/
theorem annual_cost_difference
  (clarinet_rate : ℕ) (clarinet_hours : ℕ)
  (piano_rate : ℕ) (piano_hours : ℕ)
  (violin_rate : ℕ) (violin_hours : ℕ)
  (singing_rate : ℕ) (singing_hours : ℕ)
  (h1 : clarinet_rate = 40)
  (h2 : clarinet_hours = 3)
  (h3 : piano_rate = 28)
  (h4 : piano_hours = 5)
  (h5 : violin_rate = 35)
  (h6 : violin_hours = 2)
  (h7 : singing_rate = 45)
  (h8 : singing_hours = 1)
  : (piano_rate * piano_hours + violin_rate * violin_hours + singing_rate * singing_hours) * 52 -
    (clarinet_rate * clarinet_hours) * 52 = 7020 := by
  sorry

#eval (28 * 5 + 35 * 2 + 45 * 1) * 52 - (40 * 3) * 52

end NUMINAMATH_CALUDE_annual_cost_difference_l2642_264221


namespace NUMINAMATH_CALUDE_theater_company_max_members_l2642_264287

/-- The number of columns in the rectangular formation -/
def n : ℕ := 15

/-- The total number of members in the theater company -/
def total_members : ℕ := n * (n + 9)

/-- Theorem stating that the maximum number of members satisfying the given conditions is 360 -/
theorem theater_company_max_members :
  (∃ k : ℕ, total_members = k^2 + 3) ∧
  (total_members = n * (n + 9)) ∧
  (∀ m > total_members, ¬(∃ j : ℕ, m = j^2 + 3) ∨ ¬(∃ p : ℕ, m = p * (p + 9))) ∧
  total_members = 360 := by
  sorry

end NUMINAMATH_CALUDE_theater_company_max_members_l2642_264287


namespace NUMINAMATH_CALUDE_system_solution_range_l2642_264213

theorem system_solution_range (a x y : ℝ) : 
  (5 * x + 2 * y = 11 * a + 18) →
  (2 * x - 3 * y = 12 * a - 8) →
  (x > 0) →
  (y > 0) →
  (-2/3 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_range_l2642_264213


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_squared_l2642_264203

-- Define the polynomial Q(z)
def Q (s t : ℂ) (z : ℂ) : ℂ := z^3 + s*z + t

-- Define the theorem
theorem right_triangle_hypotenuse_squared (p q r s t : ℂ) :
  Q s t p = 0 →
  Q s t q = 0 →
  Q s t r = 0 →
  Complex.abs p ^ 2 + Complex.abs q ^ 2 + Complex.abs r ^ 2 = 300 →
  ∃ (a b : ℂ), (a = q - r ∧ b = p - q) ∧ (a • b = 0) →
  Complex.abs (p - r) ^ 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_squared_l2642_264203


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l2642_264278

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an additional innings -/
def newAverage (bp : BatsmanPerformance) (newScore : ℕ) : ℚ :=
  (bp.totalScore + newScore) / (bp.innings + 1)

theorem batsman_average_after_17th_innings 
  (bp : BatsmanPerformance) 
  (h1 : bp.innings = 16) 
  (h2 : newAverage bp 85 = bp.average + 3) :
  newAverage bp 85 = 37 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l2642_264278


namespace NUMINAMATH_CALUDE_secretaries_working_hours_l2642_264292

theorem secretaries_working_hours (t₁ t₂ t₃ : ℝ) : 
  t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0 →  -- Ensuring positive working times
  t₂ = 2 * t₁ →               -- Ratio condition for t₂
  t₃ = 5 * t₁ →               -- Ratio condition for t₃
  t₃ = 75 →                   -- Longest working time
  t₁ + t₂ + t₃ = 120 :=       -- Combined total
by sorry


end NUMINAMATH_CALUDE_secretaries_working_hours_l2642_264292


namespace NUMINAMATH_CALUDE_relic_age_conversion_l2642_264249

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

/-- The octal representation of the relic's age --/
def relic_age_octal : List Nat := [4, 6, 5, 7]

theorem relic_age_conversion :
  octal_to_decimal relic_age_octal = 3956 := by
  sorry

end NUMINAMATH_CALUDE_relic_age_conversion_l2642_264249
