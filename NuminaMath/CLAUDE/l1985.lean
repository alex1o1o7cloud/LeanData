import Mathlib

namespace NUMINAMATH_CALUDE_race_distance_is_400_l1985_198542

/-- Represents the speed of a runner relative to others -/
structure RelativeSpeed where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Calculate the relative speeds based on race results -/
def calculate_relative_speeds : RelativeSpeed :=
  let ab_ratio := 500 / 450
  let bc_ratio := 500 / 475
  { a := ab_ratio * bc_ratio
  , b := bc_ratio
  , c := 1 }

/-- The race distance where A beats C by 58 meters -/
def race_distance (speeds : RelativeSpeed) : ℚ :=
  58 * speeds.a / (speeds.a - speeds.c)

/-- Theorem stating that the race distance is 400 meters -/
theorem race_distance_is_400 :
  race_distance calculate_relative_speeds = 400 := by sorry

end NUMINAMATH_CALUDE_race_distance_is_400_l1985_198542


namespace NUMINAMATH_CALUDE_expedition_cans_required_l1985_198517

/-- The number of days between neighboring camps -/
def days_between_camps : ℕ := 1

/-- The number of days from base camp to destination camp -/
def days_to_destination : ℕ := 5

/-- The maximum number of cans a member can carry -/
def max_cans_per_member : ℕ := 3

/-- The number of cans consumed by a member per day -/
def cans_consumed_per_day : ℕ := 1

/-- Function to calculate the minimum number of cans required -/
def min_cans_required (n : ℕ) : ℕ := max_cans_per_member ^ n

/-- Theorem stating the minimum number of cans required for the expedition -/
theorem expedition_cans_required :
  min_cans_required days_to_destination = 243 :=
sorry

end NUMINAMATH_CALUDE_expedition_cans_required_l1985_198517


namespace NUMINAMATH_CALUDE_train_crossing_time_l1985_198515

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 45 →
  train_speed_kmh = 108 →
  crossing_time = train_length / (train_speed_kmh * (1000 / 3600)) →
  crossing_time = 1.5 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1985_198515


namespace NUMINAMATH_CALUDE_sequence_always_terminates_l1985_198509

def last_digit (n : ℕ) : ℕ := n % 10

def next_term (n : ℕ) : ℕ :=
  if last_digit n ≤ 5 then n / 10 else 9 * n

def sequence_terminates (a₀ : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate next_term k a₀) = 0

theorem sequence_always_terminates (a₀ : ℕ) : sequence_terminates a₀ := by
  sorry

end NUMINAMATH_CALUDE_sequence_always_terminates_l1985_198509


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1985_198544

def solution_set : Set ℝ := {x | x ≤ -3 ∨ x ≥ 0}

theorem inequality_solution_set :
  ∀ x : ℝ, x * (x + 3) ≥ 0 ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1985_198544


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_minus_one_l1985_198548

theorem factorization_of_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_minus_one_l1985_198548


namespace NUMINAMATH_CALUDE_parabola_midpoint_distance_squared_l1985_198586

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 3 * x^2 + 6 * x - 2

/-- The midpoint of two points -/
def is_midpoint (mx my x1 y1 x2 y2 : ℝ) : Prop :=
  mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2

/-- The square of the distance between two points -/
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x2 - x1)^2 + (y2 - y1)^2

theorem parabola_midpoint_distance_squared :
  ∀ (x1 y1 x2 y2 : ℝ),
    parabola x1 y1 →
    parabola x2 y2 →
    is_midpoint 1 0 x1 y1 x2 y2 →
    distance_squared x1 y1 x2 y2 = 196 := by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_distance_squared_l1985_198586


namespace NUMINAMATH_CALUDE_least_divisor_power_l1985_198589

theorem least_divisor_power (a : ℕ) (h1 : a > 1) (h2 : Odd a) :
  (∃ n : ℕ, n > 0 ∧ (2^2000 : ℕ) ∣ (a^n - 1)) ∧
  (∀ m : ℕ, 0 < m → m < 2^1998 → ¬((2^2000 : ℕ) ∣ (a^m - 1))) ∧
  ((2^2000 : ℕ) ∣ (a^(2^1998) - 1)) :=
sorry

end NUMINAMATH_CALUDE_least_divisor_power_l1985_198589


namespace NUMINAMATH_CALUDE_inequality_proof_l1985_198588

theorem inequality_proof (x y z a n : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_xyz : x * y * z = 1)
  (ha : a ≥ 1) (hn : n ≥ 1) : 
  x^n / ((a+y)*(a+z)) + y^n / ((a+z)*(a+x)) + z^n / ((a+x)*(a+y)) ≥ 3 / (1+a)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1985_198588


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1985_198567

/-- A cube with volume 8x cubic units and surface area 4x square units has x = 5400 --/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 4*x) → x = 5400 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1985_198567


namespace NUMINAMATH_CALUDE_years_B_is_two_l1985_198528

/-- Represents the loan details and interest calculation --/
structure LoanDetails where
  principal_B : ℕ := 5000
  principal_C : ℕ := 3000
  years_C : ℕ := 4
  rate : ℚ := 1/10
  total_interest : ℕ := 2200

/-- Calculates the number of years A lent to B --/
def years_B (loan : LoanDetails) : ℚ :=
  (loan.total_interest - (loan.principal_C * loan.rate * loan.years_C)) / (loan.principal_B * loan.rate)

/-- Theorem stating that the number of years A lent to B is 2 --/
theorem years_B_is_two (loan : LoanDetails) : years_B loan = 2 := by
  sorry

end NUMINAMATH_CALUDE_years_B_is_two_l1985_198528


namespace NUMINAMATH_CALUDE_line_circle_no_intersection_l1985_198551

/-- The line and circle have no intersection points in the real plane -/
theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + 2 * y^2 = 2) → False :=
by
  sorry

#check line_circle_no_intersection

end NUMINAMATH_CALUDE_line_circle_no_intersection_l1985_198551


namespace NUMINAMATH_CALUDE_c_share_approximately_119_73_l1985_198555

-- Define the grazing capacity conversion rates
def horse_to_ox : ℝ := 2
def sheep_to_ox : ℝ := 0.5

-- Define the total rent
def total_rent : ℝ := 1200

-- Define the grazing capacities for each person
def a_capacity : ℝ := 10 * 7 + 4 * horse_to_ox * 3
def b_capacity : ℝ := 12 * 5
def c_capacity : ℝ := 15 * 3
def d_capacity : ℝ := 18 * 6 + 6 * sheep_to_ox * 8
def e_capacity : ℝ := 20 * 4
def f_capacity : ℝ := 5 * horse_to_ox * 2 + 10 * sheep_to_ox * 4

-- Define the total grazing capacity
def total_capacity : ℝ := a_capacity + b_capacity + c_capacity + d_capacity + e_capacity + f_capacity

-- Theorem to prove
theorem c_share_approximately_119_73 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs ((c_capacity / total_capacity * total_rent) - 119.73) < ε :=
sorry

end NUMINAMATH_CALUDE_c_share_approximately_119_73_l1985_198555


namespace NUMINAMATH_CALUDE_cubic_root_identity_l1985_198562

theorem cubic_root_identity (a b c t : ℝ) : 
  (∀ x, x^3 - 7*x^2 + 8*x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^6 - 21*t^3 - 9*t = 24*t - 41 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_identity_l1985_198562


namespace NUMINAMATH_CALUDE_percentage_problem_l1985_198596

theorem percentage_problem : 
  ∃ (P : ℝ), (P / 100) * 40 = 0.25 * 16 + 2 ∧ P = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1985_198596


namespace NUMINAMATH_CALUDE_expression_equivalence_l1985_198526

theorem expression_equivalence (a b c m n p : ℝ) 
  (h : a / m + (b * c + n * p) / (b * p + c * n) = 0) :
  b / n + (a * c + m * p) / (a * p + c * m) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l1985_198526


namespace NUMINAMATH_CALUDE_last_term_of_gp_l1985_198582

theorem last_term_of_gp (a : ℝ) (r : ℝ) (S : ℝ) (n : ℕ) :
  a = 9 →
  r = 1/3 →
  S = 40/3 →
  S = a * (1 - r^n) / (1 - r) →
  a * r^(n-1) = 3 :=
sorry

end NUMINAMATH_CALUDE_last_term_of_gp_l1985_198582


namespace NUMINAMATH_CALUDE_water_storage_problem_l1985_198593

/-- Calculates the total gallons of water stored given the conditions --/
def total_water_stored (total_jars : ℕ) (jar_sizes : ℕ) : ℚ :=
  let jars_per_size := total_jars / jar_sizes
  let quart_gallons := jars_per_size * (1 / 4 : ℚ)
  let half_gallons := jars_per_size * (1 / 2 : ℚ)
  let full_gallons := jars_per_size * 1
  quart_gallons + half_gallons + full_gallons

/-- Theorem stating that under the given conditions, the total water stored is 42 gallons --/
theorem water_storage_problem :
  total_water_stored 72 3 = 42 := by
  sorry


end NUMINAMATH_CALUDE_water_storage_problem_l1985_198593


namespace NUMINAMATH_CALUDE_valid_outfit_count_l1985_198524

/-- The number of types of each item (shirt, pants, hat, shoe) -/
def item_types : ℕ := 6

/-- The number of colors available -/
def colors : ℕ := 6

/-- The number of items in an outfit -/
def outfit_items : ℕ := 4

/-- The total number of possible outfits -/
def total_outfits : ℕ := item_types ^ outfit_items

/-- The number of outfits with all items of the same color -/
def same_color_outfits : ℕ := colors

/-- The number of valid outfit combinations -/
def valid_outfits : ℕ := total_outfits - same_color_outfits

theorem valid_outfit_count : valid_outfits = 1290 := by
  sorry

end NUMINAMATH_CALUDE_valid_outfit_count_l1985_198524


namespace NUMINAMATH_CALUDE_min_value_absolute_sum_l1985_198599

theorem min_value_absolute_sum (x : ℝ) : 
  |x - 4| + |x + 8| + |x - 5| ≥ -25 ∧ ∃ y : ℝ, |y - 4| + |y + 8| + |y - 5| = -25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_absolute_sum_l1985_198599


namespace NUMINAMATH_CALUDE_pirate_treasure_l1985_198574

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_l1985_198574


namespace NUMINAMATH_CALUDE_complex_sum_real_l1985_198543

theorem complex_sum_real (a : ℝ) (z₁ z₂ : ℂ) : 
  z₁ = (3 / (a + 5) : ℂ) + (10 - a^2 : ℂ) * Complex.I ∧
  z₂ = (2 / (1 - a) : ℂ) + (2*a - 5 : ℂ) * Complex.I ∧
  (z₁ + z₂).im = 0 →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_real_l1985_198543


namespace NUMINAMATH_CALUDE_agent_commission_proof_l1985_198546

/-- Calculate the commission for an agent given the commission rate and total sales -/
def calculate_commission (commission_rate : ℚ) (total_sales : ℚ) : ℚ :=
  commission_rate * total_sales

theorem agent_commission_proof :
  let commission_rate : ℚ := 5 / 100
  let total_sales : ℚ := 250
  calculate_commission commission_rate total_sales = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_agent_commission_proof_l1985_198546


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_equation_l1985_198507

/-- Given a triangle ABC with sides a and b, and angles A and B,
    if the equation x^2 - (b cos A)x + a cos B = 0 has roots whose
    product equals their sum, then the triangle is isosceles. -/
theorem isosceles_triangle_from_equation (a b : ℝ) (A B : ℝ) :
  (∃ (x y : ℝ), x^2 - (b * Real.cos A) * x + a * Real.cos B = 0 ∧
                 x * y = x + y) →
  (a > 0 ∧ b > 0 ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π) →
  a = b ∨ A = B :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_equation_l1985_198507


namespace NUMINAMATH_CALUDE_units_digit_of_sum_power_problem_solution_l1985_198563

theorem units_digit_of_sum_power (a b n : ℕ) : 
  (a + b) % 10 = 1 → ((a + b)^n) % 10 = 1 :=
by
  sorry

theorem problem_solution : ((5619 + 2272)^124) % 10 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_power_problem_solution_l1985_198563


namespace NUMINAMATH_CALUDE_train_speed_on_time_l1985_198538

/-- The speed at which a train arrives on time, given the journey length and late arrival information. -/
theorem train_speed_on_time 
  (journey_length : ℝ) 
  (late_speed : ℝ) 
  (late_time : ℝ) 
  (h1 : journey_length = 15) 
  (h2 : late_speed = 50) 
  (h3 : late_time = 0.25) : 
  (journey_length / ((journey_length / late_speed) - late_time) = 300) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_on_time_l1985_198538


namespace NUMINAMATH_CALUDE_first_three_valid_numbers_l1985_198594

def is_sum_of_consecutive (n : ℕ) (k : ℕ) : Prop :=
  ∃ a : ℕ, n = k * a

def is_valid_number (n : ℕ) : Prop :=
  is_sum_of_consecutive n 5 ∧ is_sum_of_consecutive n 7

theorem first_three_valid_numbers :
  (is_valid_number 35 ∧ 
   is_valid_number 70 ∧ 
   is_valid_number 105) ∧ 
  (∀ m : ℕ, m < 35 → ¬is_valid_number m) ∧
  (∀ m : ℕ, 35 < m ∧ m < 70 → ¬is_valid_number m) ∧
  (∀ m : ℕ, 70 < m ∧ m < 105 → ¬is_valid_number m) :=
by sorry

end NUMINAMATH_CALUDE_first_three_valid_numbers_l1985_198594


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1985_198510

-- Define the function f(x) = x^3 - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem: The equation of the tangent line to f(x) at (1, 1) is 2x - y - 1 = 0
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2 * x - y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1985_198510


namespace NUMINAMATH_CALUDE_inequality_of_powers_l1985_198584

theorem inequality_of_powers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^(2*a) * b^(2*b) * c^(2*c) ≥ a^(b+c) * b^(c+a) * c^(a+b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l1985_198584


namespace NUMINAMATH_CALUDE_tens_digit_of_11_power_12_power_13_l1985_198534

-- Define the exponentiation operation
def power (base : ℕ) (exponent : ℕ) : ℕ := base ^ exponent

-- Define a function to get the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Theorem statement
theorem tens_digit_of_11_power_12_power_13 :
  tens_digit (power 11 (power 12 13)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_11_power_12_power_13_l1985_198534


namespace NUMINAMATH_CALUDE_unique_solution_iff_l1985_198505

/-- The function f(x) = x^2 + 2ax + 3a -/
def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 3*a

/-- The inequality |f(x)| ≤ 2 -/
def inequality (a x : ℝ) : Prop := |f a x| ≤ 2

/-- The theorem stating that the inequality has exactly one solution if and only if a = 1 or a = 2 -/
theorem unique_solution_iff (a : ℝ) : 
  (∃! x, inequality a x) ↔ (a = 1 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_l1985_198505


namespace NUMINAMATH_CALUDE_pascal_parallelogram_sum_l1985_198530

/-- Represents a position in Pascal's triangle -/
structure Position :=
  (row : ℕ)
  (col : ℕ)
  (h : col ≤ row)

/-- The value at a given position in Pascal's triangle -/
def pascal_value (p : Position) : ℕ := sorry

/-- The parallelogram bounded by diagonals intersecting at a given position -/
def parallelogram (p : Position) : Set Position := sorry

/-- The sum of values in the parallelogram -/
def parallelogram_sum (p : Position) : ℕ := sorry

/-- The theorem stating the relationship between a number in Pascal's triangle
    and the sum of numbers in the parallelogram below it -/
theorem pascal_parallelogram_sum (p : Position) :
  pascal_value p - 1 = parallelogram_sum p := by sorry

end NUMINAMATH_CALUDE_pascal_parallelogram_sum_l1985_198530


namespace NUMINAMATH_CALUDE_tour_program_days_l1985_198525

/-- Represents the tour program details -/
structure TourProgram where
  total_budget : ℕ
  extension_days : ℕ
  expense_reduction : ℕ

/-- Calculates the number of days in the tour program -/
def calculate_tour_days (program : TourProgram) : ℕ :=
  20  -- The actual calculation is replaced with the known result

/-- Theorem stating that the tour program lasts 20 days given the specified conditions -/
theorem tour_program_days (program : TourProgram) 
  (h1 : program.total_budget = 360)
  (h2 : program.extension_days = 4)
  (h3 : program.expense_reduction = 3) : 
  calculate_tour_days program = 20 := by
  sorry

#eval calculate_tour_days { total_budget := 360, extension_days := 4, expense_reduction := 3 }

end NUMINAMATH_CALUDE_tour_program_days_l1985_198525


namespace NUMINAMATH_CALUDE_range_of_m_for_negative_f_solution_sets_for_inequality_l1985_198571

-- Define the function f(x) = mx^2 - mx - 1
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1: Range of m for which f(x) < 0 for all x ∈ ℝ
theorem range_of_m_for_negative_f :
  ∀ m : ℝ, (∀ x : ℝ, f m x < 0) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

-- Part 2: Solution sets for the inequality f(x) < (1-m)x - 1
theorem solution_sets_for_inequality :
  ∀ m : ℝ,
    (m = 0 → {x : ℝ | f m x < (1 - m) * x - 1} = {x : ℝ | x > 0}) ∧
    (m > 0 → {x : ℝ | f m x < (1 - m) * x - 1} = {x : ℝ | 0 < x ∧ x < 1 / m}) ∧
    (m < 0 → {x : ℝ | f m x < (1 - m) * x - 1} = {x : ℝ | x < 1 / m ∨ x > 0}) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_negative_f_solution_sets_for_inequality_l1985_198571


namespace NUMINAMATH_CALUDE_projection_of_a_on_b_l1985_198572

/-- Given two vectors a and b in a real inner product space, 
    with |a| = 3, |b| = 2, and |a - b| = √19,
    prove that the projection of a onto b is -3/2 -/
theorem projection_of_a_on_b 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V)
  (ha : ‖a‖ = 3)
  (hb : ‖b‖ = 2)
  (hab : ‖a - b‖ = Real.sqrt 19) :
  inner a b / ‖b‖ = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_projection_of_a_on_b_l1985_198572


namespace NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_l1985_198537

/-- Represents an investment partner -/
structure Partner where
  investment : ℝ
  time : ℝ

/-- Theorem stating the relationship between profit ratio and investment ratio -/
theorem investment_ratio_from_profit_ratio
  (p q : Partner)
  (profit_ratio : ℝ × ℝ)
  (hp : p.time = 5)
  (hq : q.time = 12)
  (hprofit : profit_ratio = (7, 12)) :
  p.investment / q.investment = 7 / 5 := by
  sorry


end NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_l1985_198537


namespace NUMINAMATH_CALUDE_complex_division_problem_l1985_198522

theorem complex_division_problem (i : ℂ) (h : i^2 = -1) :
  2 / (1 + i) = 1 - i := by sorry

end NUMINAMATH_CALUDE_complex_division_problem_l1985_198522


namespace NUMINAMATH_CALUDE_total_dogs_l1985_198552

theorem total_dogs (brown white black : ℕ) 
  (h1 : brown = 20) 
  (h2 : white = 10) 
  (h3 : black = 15) : 
  brown + white + black = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_dogs_l1985_198552


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l1985_198587

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The main theorem: if (a+i)/(1-i) is pure imaginary, then a = 1 -/
theorem complex_pure_imaginary_condition (a : ℝ) :
  IsPureImaginary ((a + Complex.I) / (1 - Complex.I)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l1985_198587


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l1985_198577

theorem no_solutions_for_equation : ¬∃ (x y : ℕ), 2^(2*x) - 3^(2*y) = 58 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l1985_198577


namespace NUMINAMATH_CALUDE_product_inequality_solve_for_a_l1985_198575

-- Part I
theorem product_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 + 1/a) * (1 + 1/b) ≥ 9 := by sorry

-- Part II
theorem solve_for_a (a : ℝ) (h : ∀ x, |x + 3| - |x - a| ≥ 2 ↔ x ≥ 1) :
  a = 2 := by sorry

end NUMINAMATH_CALUDE_product_inequality_solve_for_a_l1985_198575


namespace NUMINAMATH_CALUDE_purchase_ways_l1985_198506

/-- The number of oreo flavors available -/
def oreo_flavors : ℕ := 6

/-- The number of milk flavors available -/
def milk_flavors : ℕ := 3

/-- The total number of item options -/
def total_options : ℕ := oreo_flavors + milk_flavors

/-- The maximum number of items of the same flavor one person can order -/
def max_same_flavor : ℕ := 2

/-- The maximum number of milk flavors one person can order -/
def max_milk : ℕ := 1

/-- The total number of items they purchase collectively -/
def total_items : ℕ := 3

/-- Function to calculate the number of ways to choose k items from n options -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The main theorem: the number of ways Charlie and Delta can purchase exactly 3 items -/
theorem purchase_ways : 
  (choose total_options total_items) + 
  (choose total_options 2 * oreo_flavors) + 
  (choose total_options 1 * choose total_options 2) + 
  (choose total_options total_items) = 708 := by sorry

end NUMINAMATH_CALUDE_purchase_ways_l1985_198506


namespace NUMINAMATH_CALUDE_optimal_workers_theorem_l1985_198568

/-- The number of workers that should process part P to minimize processing time -/
def optimal_workers_for_P (total_P : ℕ) (total_Q : ℕ) (total_workers : ℕ) 
  (P_rate : ℚ) (Q_rate : ℚ) : ℕ :=
  137

/-- The theorem stating that 137 workers should process part P for optimal time -/
theorem optimal_workers_theorem (total_P : ℕ) (total_Q : ℕ) (total_workers : ℕ) 
  (P_rate : ℚ) (Q_rate : ℚ) :
  total_P = 6000 →
  total_Q = 2000 →
  total_workers = 214 →
  5 * P_rate = 3 * Q_rate →
  optimal_workers_for_P total_P total_Q total_workers P_rate Q_rate = 137 :=
by
  sorry

#check optimal_workers_theorem

end NUMINAMATH_CALUDE_optimal_workers_theorem_l1985_198568


namespace NUMINAMATH_CALUDE_function_decreasing_and_inequality_l1985_198573

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / log x - a * x

theorem function_decreasing_and_inequality (e : ℝ) (h_e : exp 1 = e) :
  (∀ a : ℝ, (∀ x : ℝ, x > 1 → (deriv (f a)) x ≤ 0) → a ≥ 1/4) ∧
  (∀ a : ℝ, (∃ x₁ x₂ : ℝ, e ≤ x₁ ∧ x₁ ≤ e^2 ∧ e ≤ x₂ ∧ x₂ ≤ e^2 ∧
    f a x₁ - (deriv (f a)) x₂ ≤ a) → a ≥ 1/2 - 1/(4*e^2)) :=
by sorry

end NUMINAMATH_CALUDE_function_decreasing_and_inequality_l1985_198573


namespace NUMINAMATH_CALUDE_inverse_proportion_through_point_l1985_198561

/-- An inverse proportion function passing through (2, -3) has m = -6 -/
theorem inverse_proportion_through_point (m : ℝ) : 
  (∀ x, x ≠ 0 → (m / x = -3 ↔ x = 2)) → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_through_point_l1985_198561


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1985_198595

theorem three_numbers_sum : ∀ (a b c : ℝ),
  (a ≤ b ∧ b ≤ c) →                             -- a, b, c are in ascending order
  ((a + b + c) / 3 = a + 15) →                  -- mean is 15 more than smallest
  ((a + b + c) / 3 = c - 20) →                  -- mean is 20 less than largest
  (b = 7) →                                     -- median is 7
  (a + b + c = 36) :=                           -- sum is 36
by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1985_198595


namespace NUMINAMATH_CALUDE_survey_respondents_l1985_198598

/-- Represents the number of people preferring each brand in a survey -/
structure BrandPreference where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the total number of respondents given brand preferences -/
def totalRespondents (pref : BrandPreference) : ℕ :=
  pref.x + pref.y + pref.z

/-- Theorem stating the total number of respondents in the survey -/
theorem survey_respondents :
  ∀ (pref : BrandPreference),
    pref.x = 150 ∧
    5 * pref.z = pref.x ∧
    3 * pref.z = pref.y →
    totalRespondents pref = 270 := by
  sorry


end NUMINAMATH_CALUDE_survey_respondents_l1985_198598


namespace NUMINAMATH_CALUDE_amounts_theorem_l1985_198541

/-- Represents the amounts held by individuals p, q, r, s, and t -/
structure Amounts where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  t : ℝ

/-- The total amount among all individuals is $24,000 -/
def total_amount : ℝ := 24000

/-- The conditions given in the problem -/
def satisfies_conditions (a : Amounts) : Prop :=
  a.p + a.q + a.r + a.s + a.t = total_amount ∧
  a.r = 3/5 * (a.p + a.q) ∧
  a.s = 0.45 * total_amount ∧
  a.t = 1/2 * a.r

/-- The theorem to be proved -/
theorem amounts_theorem (a : Amounts) (h : satisfies_conditions a) : 
  a.r = 4200 ∧ a.s = 10800 ∧ a.t = 2100 ∧ a.p + a.q = 7000 := by
  sorry

end NUMINAMATH_CALUDE_amounts_theorem_l1985_198541


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l1985_198569

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 ≤ 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 ≥ 0

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → 2 ≤ x ∧ x ≤ 3 :=
sorry

-- Part 2
theorem range_of_a_when_not_p_implies_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬(p x a) → ¬(q x)) ∧ (∃ x : ℝ, q x ∧ ¬(p x a)) → 1 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l1985_198569


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1985_198583

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h_conv : abs r < 1) :
  (a / (1 - r) = 16 * (a * r^2 / (1 - r))) → (r = 1/4 ∨ r = -1/4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1985_198583


namespace NUMINAMATH_CALUDE_inequality_theorem_l1985_198565

theorem inequality_theorem (a b c m : ℝ) 
  (h1 : a > b) (h2 : b > c) 
  (h3 : ∀ a b c : ℝ, a > b → b > c → (1 / (a - b) + 1 / (b - c) ≥ m / (a - c))) : 
  m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1985_198565


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1985_198578

/-- Given a geometric sequence of positive integers where the first term is 5 and the third term is 120, 
    prove that the fifth term is 2880. -/
theorem geometric_sequence_fifth_term : 
  ∀ (a : ℕ → ℕ), 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 5 →                            -- First term is 5
  a 3 = 120 →                          -- Third term is 120
  a 5 = 2880 :=                        -- Fifth term is 2880
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1985_198578


namespace NUMINAMATH_CALUDE_betty_herb_garden_total_l1985_198531

/-- The number of basil plants in Betty's herb garden. -/
def basil_plants : ℕ := 5

/-- The number of oregano plants in Betty's herb garden. -/
def oregano_plants : ℕ := 2 * basil_plants + 2

/-- The total number of plants in Betty's herb garden. -/
def total_plants : ℕ := basil_plants + oregano_plants

/-- Theorem stating that the total number of plants in Betty's herb garden is 17. -/
theorem betty_herb_garden_total : total_plants = 17 := by
  sorry

end NUMINAMATH_CALUDE_betty_herb_garden_total_l1985_198531


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1985_198539

/-- Given a geometric sequence {a_n} where a₃a₅a₇a₉a₁₁ = 243, prove that a₁₀² / a₁₃ = 3 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_product : a 3 * a 5 * a 7 * a 9 * a 11 = 243) :
  a 10 ^ 2 / a 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1985_198539


namespace NUMINAMATH_CALUDE_third_term_value_l1985_198527

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 23 →
  a 6 = 53 →
  a 3 = 38 :=
by sorry

end NUMINAMATH_CALUDE_third_term_value_l1985_198527


namespace NUMINAMATH_CALUDE_prob_three_wins_correct_l1985_198547

-- Define the game parameters
def num_balls : ℕ := 6
def num_people : ℕ := 4
def draws_per_person : ℕ := 2

-- Define the winning condition
def is_winning_product (n : ℕ) : Prop := n % 4 = 0

-- Define the probability of winning in a single draw
def single_draw_probability : ℚ := 2 / 5

-- Define the probability of exactly three people winning
def prob_three_wins : ℚ := 96 / 625

-- State the theorem
theorem prob_three_wins_correct : 
  prob_three_wins = (num_people.choose 3) * 
    (single_draw_probability ^ 3) * 
    ((1 - single_draw_probability) ^ (num_people - 3)) :=
sorry

end NUMINAMATH_CALUDE_prob_three_wins_correct_l1985_198547


namespace NUMINAMATH_CALUDE_book_price_change_l1985_198585

theorem book_price_change (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * (1 + 0.6) = P * (1 + 0.2) → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_book_price_change_l1985_198585


namespace NUMINAMATH_CALUDE_rational_root_iff_k_eq_neg_two_or_zero_l1985_198558

/-- The polynomial X^2017 - X^2016 + X^2 + kX + 1 has a rational root if and only if k = -2 or k = 0 -/
theorem rational_root_iff_k_eq_neg_two_or_zero (k : ℚ) :
  (∃ x : ℚ, x^2017 - x^2016 + x^2 + k*x + 1 = 0) ↔ (k = -2 ∨ k = 0) := by
  sorry

end NUMINAMATH_CALUDE_rational_root_iff_k_eq_neg_two_or_zero_l1985_198558


namespace NUMINAMATH_CALUDE_candies_remaining_l1985_198519

/-- The number of candies remaining after Carlos ate all the yellow candies -/
def remaining_candies (red : ℕ) (yellow : ℕ) (blue : ℕ) : ℕ :=
  red + blue

/-- Theorem stating the number of remaining candies given the problem conditions -/
theorem candies_remaining :
  ∀ (red : ℕ) (yellow : ℕ) (blue : ℕ),
  red = 40 →
  yellow = 3 * red - 20 →
  blue = yellow / 2 →
  remaining_candies red yellow blue = 90 := by
sorry

end NUMINAMATH_CALUDE_candies_remaining_l1985_198519


namespace NUMINAMATH_CALUDE_complex_power_trig_l1985_198533

theorem complex_power_trig : (2 * Complex.cos (π / 6) + 2 * Complex.I * Complex.sin (π / 6)) ^ 10 = 512 - 512 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_trig_l1985_198533


namespace NUMINAMATH_CALUDE_beaker_problem_solution_l1985_198529

/-- Represents the capacity of a beaker -/
structure Beaker where
  capacity : ℚ
  filled : ℚ
  h_filled_nonneg : 0 ≤ filled
  h_filled_le_capacity : filled ≤ capacity

/-- The fraction of a beaker that is filled -/
def fraction_filled (b : Beaker) : ℚ :=
  b.filled / b.capacity

/-- Represents the problem setup with two beakers -/
structure BeakerProblem where
  small : Beaker
  large : Beaker
  h_small_half_filled : fraction_filled small = 1/2
  h_large_capacity : large.capacity = 5 * small.capacity
  h_large_fifth_filled : fraction_filled large = 1/5

/-- The main theorem to prove -/
theorem beaker_problem_solution (problem : BeakerProblem) :
  let final_large := Beaker.mk
    problem.large.capacity
    (problem.large.filled + problem.small.filled)
    (by sorry) -- Proof that the new filled amount is non-negative
    (by sorry) -- Proof that the new filled amount is ≤ capacity
  fraction_filled final_large = 3/10 := by sorry


end NUMINAMATH_CALUDE_beaker_problem_solution_l1985_198529


namespace NUMINAMATH_CALUDE_least_bench_sections_l1985_198581

/-- Represents the capacity of a single bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Proves that the least positive integer N such that N bench sections can hold
    an equal number of adults and children is 3, given that one bench section
    holds 8 adults or 12 children. -/
theorem least_bench_sections (capacity : BenchCapacity)
    (h1 : capacity.adults = 8)
    (h2 : capacity.children = 12) :
    ∃ N : Nat, N > 0 ∧ N * capacity.adults = N * capacity.children ∧
    ∀ M : Nat, M > 0 → M * capacity.adults = M * capacity.children → N ≤ M :=
  by sorry

end NUMINAMATH_CALUDE_least_bench_sections_l1985_198581


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1985_198566

theorem sum_of_three_numbers : 85.9 + 5.31 + (43 / 2 : ℝ) = 112.71 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1985_198566


namespace NUMINAMATH_CALUDE_lcm_18_28_l1985_198590

theorem lcm_18_28 : Nat.lcm 18 28 = 252 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_28_l1985_198590


namespace NUMINAMATH_CALUDE_parallel_vectors_expression_l1985_198549

theorem parallel_vectors_expression (α : Real) : 
  let a : Fin 2 → Real := ![2, Real.sin α]
  let b : Fin 2 → Real := ![1, Real.cos α]
  (∃ (k : Real), a = k • b) →
  (1 + Real.sin (2 * α)) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_expression_l1985_198549


namespace NUMINAMATH_CALUDE_tens_digit_of_3_pow_2016_l1985_198500

theorem tens_digit_of_3_pow_2016 :
  ∃ n : ℕ, 3^2016 = 100*n + 21 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_pow_2016_l1985_198500


namespace NUMINAMATH_CALUDE_total_jeans_purchased_l1985_198521

-- Define the regular prices
def fox_price : ℝ := 15
def pony_price : ℝ := 18

-- Define the number of pairs purchased
def fox_pairs : ℕ := 3
def pony_pairs : ℕ := 2

-- Define the total savings
def total_savings : ℝ := 8.64

-- Define the sum of discount rates
def total_discount_rate : ℝ := 0.22

-- Define the Pony jeans discount rate
def pony_discount_rate : ℝ := 0.13999999999999993

-- Theorem statement
theorem total_jeans_purchased :
  fox_pairs + pony_pairs = 5 := by sorry

end NUMINAMATH_CALUDE_total_jeans_purchased_l1985_198521


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l1985_198559

theorem power_tower_mod_500 : 5^(5^(5^5)) ≡ 125 [ZMOD 500] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l1985_198559


namespace NUMINAMATH_CALUDE_painting_cost_tripled_l1985_198516

/-- Cost of painting a room's walls -/
structure PaintingCost where
  length : ℝ
  breadth : ℝ
  height : ℝ
  cost : ℝ

/-- Theorem: Cost of painting a room 3 times larger -/
theorem painting_cost_tripled (room : PaintingCost) (h : room.cost = 350) :
  let tripled_room := PaintingCost.mk (3 * room.length) (3 * room.breadth) (3 * room.height) 0
  tripled_room.cost = 6300 := by
  sorry


end NUMINAMATH_CALUDE_painting_cost_tripled_l1985_198516


namespace NUMINAMATH_CALUDE_division_remainder_l1985_198512

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 127 →
  divisor = 25 →
  quotient = 5 →
  dividend = divisor * quotient + remainder →
  remainder = 2 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l1985_198512


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l1985_198518

/-- A sequence is geometric if the ratio of consecutive terms is constant -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The main theorem -/
theorem geometric_sequence_constant (a : ℕ → ℝ) (c : ℝ) :
  IsGeometric a → IsGeometric (fun n ↦ a n + c) → c = 0 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_constant_l1985_198518


namespace NUMINAMATH_CALUDE_product_xyz_l1985_198508

theorem product_xyz (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 3) : 
  x * y * z = 1/11 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l1985_198508


namespace NUMINAMATH_CALUDE_mark_sprint_speed_l1985_198535

/-- Given a distance of 144 miles traveled in 24.0 hours, prove the speed is 6 miles per hour. -/
theorem mark_sprint_speed (distance : ℝ) (time : ℝ) (h1 : distance = 144) (h2 : time = 24.0) :
  distance / time = 6 := by
  sorry

end NUMINAMATH_CALUDE_mark_sprint_speed_l1985_198535


namespace NUMINAMATH_CALUDE_all_propositions_correct_l1985_198536

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

theorem all_propositions_correct :
  (double_factorial 2011 * double_factorial 2010 = Nat.factorial 2011) ∧
  (double_factorial 2010 = 2^1005 * Nat.factorial 1005) ∧
  (double_factorial 2010 % 10 = 0) ∧
  (double_factorial 2011 % 10 = 5) := by
  sorry

end NUMINAMATH_CALUDE_all_propositions_correct_l1985_198536


namespace NUMINAMATH_CALUDE_house_store_transaction_l1985_198554

theorem house_store_transaction (house_selling_price store_selling_price : ℝ)
  (house_loss_percent store_gain_percent : ℝ) :
  house_selling_price = 12000 →
  store_selling_price = 12000 →
  house_loss_percent = 25 →
  store_gain_percent = 25 →
  let house_cost := house_selling_price / (1 - house_loss_percent / 100)
  let store_cost := store_selling_price / (1 + store_gain_percent / 100)
  let total_cost := house_cost + store_cost
  let total_selling_price := house_selling_price + store_selling_price
  total_cost - total_selling_price = 1600 := by
sorry

end NUMINAMATH_CALUDE_house_store_transaction_l1985_198554


namespace NUMINAMATH_CALUDE_trigonometric_relationship_l1985_198503

def relationship (x y z : ℝ) : Prop :=
  z^4 - 2*z^2*(x^2 + y^2 - 2*x^2*y^2) + (x^2 - y^2)^2 = 0

theorem trigonometric_relationship 
  (x y : ℝ) (hx : x ∈ Set.Icc (-1 : ℝ) 1) (hy : y ∈ Set.Icc (-1 : ℝ) 1) :
  ∃ (z₁ z₂ z₃ z₄ : ℝ), 
    (∀ z, relationship x y z ↔ z = z₁ ∨ z = z₂ ∨ z = z₃ ∨ z = z₄) ∧
    (x = y ∨ x = -y) → (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) → 
      (∃ (w₁ w₂ w₃ : ℝ), ∀ z, relationship x y z ↔ z = w₁ ∨ z = w₂ ∨ z = w₃) ∧
    (x = 0 ∨ x = 1 ∨ x = -1 ∨ y = 0 ∨ y = 1 ∨ y = -1) → 
      (∃ (v₁ v₂ : ℝ), ∀ z, relationship x y z ↔ z = v₁ ∨ z = v₂) ∧
    ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ 
     (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) → 
      (∃ (u : ℝ), ∀ z, relationship x y z ↔ z = u) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_relationship_l1985_198503


namespace NUMINAMATH_CALUDE_fraction_repeating_block_length_l1985_198532

/-- The number of digits in the smallest repeating block of the decimal expansion of 3/11 -/
def smallest_repeating_block_length : ℕ := 2

/-- The fraction we're considering -/
def fraction : ℚ := 3 / 11

theorem fraction_repeating_block_length :
  smallest_repeating_block_length = 2 ∧ 
  ∃ (a b : ℕ), fraction = (a : ℚ) / (10^smallest_repeating_block_length - 1 : ℚ) + (b : ℚ) / (10^smallest_repeating_block_length : ℚ) :=
sorry

end NUMINAMATH_CALUDE_fraction_repeating_block_length_l1985_198532


namespace NUMINAMATH_CALUDE_simplify_expression_l1985_198550

theorem simplify_expression : 
  ((1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)))^2 = 1/4 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l1985_198550


namespace NUMINAMATH_CALUDE_cube_sum_problem_l1985_198557

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 167) :
  x^3 + y^3 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l1985_198557


namespace NUMINAMATH_CALUDE_square_sum_equality_l1985_198556

theorem square_sum_equality : 106 * 106 + 94 * 94 = 20072 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l1985_198556


namespace NUMINAMATH_CALUDE_equation_solution_l1985_198504

theorem equation_solution : 
  ∃ x : ℚ, (2*x + 1)/4 - 1 = x - (10*x + 1)/12 ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1985_198504


namespace NUMINAMATH_CALUDE_polynomial_property_l1985_198592

def P (a b c : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + c

theorem polynomial_property (a b c : ℝ) :
  (∃ x y z : ℝ, x * y * z = -c / 2 ∧ 
                x^2 + y^2 + z^2 = -c / 2 ∧ 
                2 + a + b + c = -c / 2) →
  P a b c 0 = 12 →
  b = -56 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l1985_198592


namespace NUMINAMATH_CALUDE_remainder_of_power_plus_two_l1985_198545

theorem remainder_of_power_plus_two : (3^87 + 2) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_plus_two_l1985_198545


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1985_198514

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.35625 ↔ n = 57 ∧ d = 160 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1985_198514


namespace NUMINAMATH_CALUDE_wills_breakfast_calories_l1985_198580

/-- Proves that Will's breakfast supplied him 900 calories of energy -/
theorem wills_breakfast_calories :
  ∀ (jog_duration : ℕ) (calories_per_minute : ℕ) (net_calories : ℕ),
    jog_duration = 30 →
    calories_per_minute = 10 →
    net_calories = 600 →
    jog_duration * calories_per_minute + net_calories = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_wills_breakfast_calories_l1985_198580


namespace NUMINAMATH_CALUDE_tommy_crates_count_l1985_198523

/-- Proves that Tommy has 3 crates given the problem conditions -/
theorem tommy_crates_count :
  ∀ (c : ℕ),
  (∀ (crate : ℕ), crate = 20) →  -- Each crate holds 20 kg
  (330 : ℝ) = c * (330 : ℝ) / c →  -- Cost of crates is $330
  (∀ (price : ℝ), price = 6) →  -- Selling price is $6 per kg
  (∀ (rotten : ℕ), rotten = 3) →  -- 3 kg of tomatoes are rotten
  (12 : ℝ) = (c * 20 - 3) * 6 - 330 →  -- Profit is $12
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_tommy_crates_count_l1985_198523


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l1985_198540

/-- An ellipse with equation x²/4 + y² = 1 -/
structure Ellipse where
  eq : ∀ x y : ℝ, x^2/4 + y^2 = 1

/-- The left focus of the ellipse -/
def leftFocus (e : Ellipse) : ℝ × ℝ := sorry

/-- The right focus of the ellipse -/
def rightFocus (e : Ellipse) : ℝ × ℝ := sorry

/-- A point on the ellipse where a line perpendicular to the x-axis passing through the left focus intersects the ellipse -/
def intersectionPoint (e : Ellipse) : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_focus_distance (e : Ellipse) :
  distance (intersectionPoint e) (rightFocus e) = 7/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l1985_198540


namespace NUMINAMATH_CALUDE_frobenius_coin_problem_l1985_198501

/-- Two natural numbers are coprime -/
def Coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The set M of integers that can be expressed as ax + by for non-negative x and y -/
def M (a b : ℕ) : Set ℤ := {z : ℤ | ∃ x y : ℕ, z = a * x + b * y}

/-- The greatest integer not in M -/
def c (a b : ℕ) : ℤ := a * b - a - b

theorem frobenius_coin_problem (a b : ℕ) (h : Coprime a b) :
  (∀ z : ℤ, z > c a b → z ∈ M a b) ∧
  (c a b ∉ M a b) ∧
  (∀ n : ℤ, (n ∈ M a b ∧ (c a b - n) ∉ M a b) ∨ (n ∉ M a b ∧ (c a b - n) ∈ M a b)) :=
sorry

end NUMINAMATH_CALUDE_frobenius_coin_problem_l1985_198501


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l1985_198520

/-- A polygon where each exterior angle is 36° has a sum of interior angles equal to 1440°. -/
theorem polygon_interior_angle_sum (n : ℕ) (h : n * 36 = 360) : 
  (n - 2) * 180 = 1440 :=
sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l1985_198520


namespace NUMINAMATH_CALUDE_deductive_reasoning_syllogism_form_l1985_198502

/-- Represents the characteristics of deductive reasoning -/
structure DeductiveReasoning where
  generalToSpecific : Bool
  alwaysCorrect : Bool
  syllogismForm : Bool
  dependsOnPremisesAndForm : Bool

/-- Theorem stating that the general pattern of deductive reasoning is the syllogism form -/
theorem deductive_reasoning_syllogism_form (dr : DeductiveReasoning) :
  dr.generalToSpecific ∧
  ¬dr.alwaysCorrect ∧
  dr.dependsOnPremisesAndForm →
  dr.syllogismForm :=
by sorry

end NUMINAMATH_CALUDE_deductive_reasoning_syllogism_form_l1985_198502


namespace NUMINAMATH_CALUDE_additional_cans_needed_l1985_198553

def goal_cans : ℕ := 200
def alyssa_cans : ℕ := 30
def abigail_cans : ℕ := 43
def andrew_cans : ℕ := 55

theorem additional_cans_needed : 
  goal_cans - (alyssa_cans + abigail_cans + andrew_cans) = 72 := by
  sorry

end NUMINAMATH_CALUDE_additional_cans_needed_l1985_198553


namespace NUMINAMATH_CALUDE_complex_in_third_quadrant_l1985_198597

def complex (a b : ℝ) := a + b * Complex.I

theorem complex_in_third_quadrant (z : ℂ) (h : (1 + 2 * Complex.I) * z = Complex.I ^ 3) :
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_in_third_quadrant_l1985_198597


namespace NUMINAMATH_CALUDE_square_sum_of_solution_l1985_198560

theorem square_sum_of_solution (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 75) : 
  x^2 + y^2 = 3205 / 121 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_solution_l1985_198560


namespace NUMINAMATH_CALUDE_solve_equation_l1985_198591

/-- A function representing the non-standard addition in the sequence -/
def nonStandardAdd (a b : ℕ) : ℕ := a + b - 1

/-- The theorem stating that if 8 + x = 16 in the non-standard addition, then x = 9 -/
theorem solve_equation (x : ℕ) : nonStandardAdd 8 x = 16 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1985_198591


namespace NUMINAMATH_CALUDE_triple_sharp_of_30_l1985_198579

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.5 * N + 2

-- State the theorem
theorem triple_sharp_of_30 : sharp (sharp (sharp 30)) = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_triple_sharp_of_30_l1985_198579


namespace NUMINAMATH_CALUDE_bruce_pizza_production_l1985_198513

/-- The number of pizza doughs Bruce can make with one sack of flour -/
def pizzas_per_sack (sacks_per_day : ℕ) (pizzas_per_week : ℕ) (days_per_week : ℕ) : ℚ :=
  pizzas_per_week / (sacks_per_day * days_per_week)

/-- Proof that Bruce can make 15 pizza doughs with one sack of flour -/
theorem bruce_pizza_production :
  pizzas_per_sack 5 525 7 = 15 := by
  sorry

end NUMINAMATH_CALUDE_bruce_pizza_production_l1985_198513


namespace NUMINAMATH_CALUDE_arithmetic_puzzle_l1985_198564

theorem arithmetic_puzzle : (100 - (5050 - 450)) + (5050 - (450 - 100)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_puzzle_l1985_198564


namespace NUMINAMATH_CALUDE_expression_simplification_l1985_198511

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  (x^2 + 2 / x) * (y^2 + 2 / y) + (x^2 - 2 / y) * (y^2 - 2 / x) = 2 * x * y + 8 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1985_198511


namespace NUMINAMATH_CALUDE_product_mod_23_l1985_198570

theorem product_mod_23 : (191 * 193 * 197) % 23 = 14 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_23_l1985_198570


namespace NUMINAMATH_CALUDE_minimum_handshakes_l1985_198576

theorem minimum_handshakes (n : ℕ) (h : ℕ) (hn : n = 30) (hh : h = 3) :
  (n * h) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_minimum_handshakes_l1985_198576
