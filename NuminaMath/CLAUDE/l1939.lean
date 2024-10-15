import Mathlib

namespace NUMINAMATH_CALUDE_point_not_on_line_l1939_193945

theorem point_not_on_line (m k : ℝ) (h1 : m * k > 0) :
  ¬(∃ (x y : ℝ), x = 2000 ∧ y = 0 ∧ y = m * x + k) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l1939_193945


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_point_seven_l1939_193978

theorem at_least_one_greater_than_point_seven (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x > 0.7 ∨ y > 0.7 ∨ (1 / (x + y)) > 0.7 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_point_seven_l1939_193978


namespace NUMINAMATH_CALUDE_position_of_2007_l1939_193942

/-- Represents the position of a number in the table -/
structure Position where
  row : ℕ
  column : ℕ

/-- The arrangement of positive odd numbers in 5 columns -/
def arrangement (n : ℕ) : Position :=
  let cycle := (n - 1) / 8
  let position := (n - 1) % 8
  match position with
  | 0 => ⟨cycle * 2 + 1, 2⟩
  | 1 => ⟨cycle * 2 + 1, 3⟩
  | 2 => ⟨cycle * 2 + 1, 4⟩
  | 3 => ⟨cycle * 2 + 1, 5⟩
  | 4 => ⟨cycle * 2 + 2, 1⟩
  | 5 => ⟨cycle * 2 + 2, 2⟩
  | 6 => ⟨cycle * 2 + 2, 3⟩
  | 7 => ⟨cycle * 2 + 2, 4⟩
  | _ => ⟨0, 0⟩  -- This case should never occur

theorem position_of_2007 : arrangement 2007 = ⟨251, 5⟩ := by
  sorry

end NUMINAMATH_CALUDE_position_of_2007_l1939_193942


namespace NUMINAMATH_CALUDE_oil_price_reduction_l1939_193987

/-- Proves that given a 35% reduction in oil price allowing 5 kg more for Rs. 800, the reduced price is Rs. 36.4 per kg -/
theorem oil_price_reduction (original_price : ℝ) : 
  (800 / (0.65 * original_price) - 800 / original_price = 5) →
  (0.65 * original_price = 36.4) := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l1939_193987


namespace NUMINAMATH_CALUDE_power_multiplication_l1939_193902

theorem power_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1939_193902


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1939_193940

theorem unique_integer_solution :
  ∃! (x : ℤ), x - 8 / (x - 2) = 5 - 8 / (x - 2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1939_193940


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1939_193930

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = 16) →
  (b / a = Real.sqrt 55 / 11) →
  (∀ x y : ℝ, x^2 / 11 - y^2 / 5 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1939_193930


namespace NUMINAMATH_CALUDE_work_completion_larger_group_size_l1939_193903

theorem work_completion (work_days : ℕ) (small_group : ℕ) (large_group_days : ℕ) : ℕ :=
  let total_man_days := work_days * small_group
  total_man_days / large_group_days

theorem larger_group_size : work_completion 25 12 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_larger_group_size_l1939_193903


namespace NUMINAMATH_CALUDE_max_value_expression_l1939_193943

theorem max_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * (x + y + z)^2) / ((x + y)^2 * (y + z)^2) ≤ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1939_193943


namespace NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l1939_193933

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence 
  (a : ℕ → ℤ) (h : arithmetic_sequence a) (h5 : a 5 = 3) (h6 : a 6 = -2) : 
  ∃ d : ℤ, d = -5 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l1939_193933


namespace NUMINAMATH_CALUDE_floor_sqrt_17_squared_l1939_193955

theorem floor_sqrt_17_squared : ⌊Real.sqrt 17⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_17_squared_l1939_193955


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l1939_193969

theorem largest_four_digit_divisible_by_35 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 35 = 0 → n ≤ 9975 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l1939_193969


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1939_193916

theorem inequality_solution_set (x : ℝ) : x + 2 > 3 ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1939_193916


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l1939_193929

theorem theater_ticket_sales (orchestra_price balcony_price premium_price : ℕ)
                             (total_tickets : ℕ) (total_revenue : ℕ)
                             (orchestra balcony premium : ℕ) :
  orchestra_price = 15 →
  balcony_price = 10 →
  premium_price = 25 →
  total_tickets = 550 →
  total_revenue = 9750 →
  orchestra + balcony + premium = total_tickets →
  orchestra_price * orchestra + balcony_price * balcony + premium_price * premium = total_revenue →
  premium = 5 * orchestra →
  orchestra ≥ 50 →
  balcony - orchestra = 179 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l1939_193929


namespace NUMINAMATH_CALUDE_day_of_week_proof_l1939_193973

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Calculate the number of days between two dates -/
def daysBetween (d1 d2 : Date) : Int :=
  sorry

/-- Get the day of the week for a given date -/
def getDayOfWeek (d : Date) (knownDate : Date) (knownDay : DayOfWeek) : DayOfWeek :=
  sorry

theorem day_of_week_proof :
  let knownDate := Date.mk 1998 4 10
  let knownDay := DayOfWeek.Friday
  let date1 := Date.mk 1918 7 6
  let date2 := Date.mk 2018 6 6
  (getDayOfWeek date1 knownDate knownDay = DayOfWeek.Saturday) ∧
  (getDayOfWeek date2 knownDate knownDay = DayOfWeek.Tuesday) := by
  sorry

end NUMINAMATH_CALUDE_day_of_week_proof_l1939_193973


namespace NUMINAMATH_CALUDE_expression_evaluation_l1939_193913

theorem expression_evaluation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  ((x^2 - 1)^2 * (x^3 - x^2 + 1)^2 / (x^5 - 1)^2)^2 * 
  ((x^2 + 1)^2 * (x^3 + x^2 + 1)^2 / (x^5 + 1)^2)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1939_193913


namespace NUMINAMATH_CALUDE_max_m_over_n_l1939_193951

open Real

noncomputable def f (m n x : ℝ) : ℝ := Real.exp (-x) + (n * x) / (m * x + n)

theorem max_m_over_n (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, x ≥ 0 → f m n x ≥ 1) ∧ f m n 0 = 1 →
  m / n ≤ (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_over_n_l1939_193951


namespace NUMINAMATH_CALUDE_gp_common_ratio_l1939_193937

/-- Geometric progression properties -/
structure GeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  n : ℕ  -- number of terms
  last : ℝ  -- last term
  sum : ℝ  -- sum of terms

/-- Theorem: Common ratio of a specific geometric progression -/
theorem gp_common_ratio 
  (gp : GeometricProgression) 
  (h1 : gp.a = 9)
  (h2 : gp.last = 1/3)
  (h3 : gp.sum = 40/3) :
  gp.r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_gp_common_ratio_l1939_193937


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l1939_193995

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 6)
  (h2 : new_price = 7.5)
  (h3 : initial_price > 0 ∧ new_price > 0) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l1939_193995


namespace NUMINAMATH_CALUDE_escalator_ride_time_main_escalator_theorem_l1939_193931

/-- Represents the time it takes Leo to ride an escalator in different scenarios -/
structure EscalatorRide where
  stationary_walk : ℝ  -- Time to walk down stationary escalator
  moving_walk : ℝ      -- Time to walk down moving escalator
  no_walk : ℝ          -- Time to ride without walking (to be proven)

/-- Theorem stating that given the conditions, the time to ride without walking is 48 seconds -/
theorem escalator_ride_time (ride : EscalatorRide) 
  (h1 : ride.stationary_walk = 80)
  (h2 : ride.moving_walk = 30) : 
  ride.no_walk = 48 := by
  sorry

/-- Main theorem combining all conditions and the result -/
theorem main_escalator_theorem : 
  ∃ (ride : EscalatorRide), ride.stationary_walk = 80 ∧ ride.moving_walk = 30 ∧ ride.no_walk = 48 := by
  sorry

end NUMINAMATH_CALUDE_escalator_ride_time_main_escalator_theorem_l1939_193931


namespace NUMINAMATH_CALUDE_magnitude_of_b_l1939_193990

/-- Given vectors a and b in ℝ², prove that |b| = √2 under the given conditions -/
theorem magnitude_of_b (a b : ℝ × ℝ) : 
  a = (-Real.sqrt 3, 1) →
  (a.1 + 2 * b.1) * a.1 + (a.2 + 2 * b.2) * a.2 = 0 →
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2 = 0 →
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_magnitude_of_b_l1939_193990


namespace NUMINAMATH_CALUDE_difference_from_averages_l1939_193971

theorem difference_from_averages (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50) 
  (h2 : (b + c) / 2 = 70) : 
  c - a = 40 := by
sorry

end NUMINAMATH_CALUDE_difference_from_averages_l1939_193971


namespace NUMINAMATH_CALUDE_cell_population_after_9_days_l1939_193941

/-- Represents the growth and mortality of a cell population over time -/
def cell_population (initial_cells : ℕ) (growth_rate : ℚ) (mortality_rate : ℚ) (cycles : ℕ) : ℕ :=
  sorry

/-- Theorem stating the cell population after 9 days -/
theorem cell_population_after_9_days :
  cell_population 5 2 (9/10) 3 = 28 :=
sorry

end NUMINAMATH_CALUDE_cell_population_after_9_days_l1939_193941


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_specific_case_l1939_193906

theorem least_subtrahend_for_divisibility (n : Nat) (d : Nat) (h : Prime d) :
  let r := n % d
  r = (n - (n - r)) % d ∧ 
  ∀ m : Nat, m < r → (n - m) % d ≠ 0 :=
by
  sorry

#eval 2376819 % 139  -- This should evaluate to 135

theorem specific_case : 
  let n := 2376819
  let d := 139
  Prime d ∧ 
  (n - 135) % d = 0 ∧
  ∀ m : Nat, m < 135 → (n - m) % d ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_specific_case_l1939_193906


namespace NUMINAMATH_CALUDE_largest_angle_in_3_4_5_ratio_triangle_l1939_193997

theorem largest_angle_in_3_4_5_ratio_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  (a + b + c = 180) →
  (b = (4/3) * a) →
  (c = (5/3) * a) →
  c = 75 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_3_4_5_ratio_triangle_l1939_193997


namespace NUMINAMATH_CALUDE_innocent_statement_l1939_193972

/-- Represents the type of person making a statement --/
inductive PersonType
| Knight
| Liar
| Normal

/-- Represents a statement that can be made --/
inductive Statement
| IAmALiar

/-- Defines whether a statement is true or false --/
def isTrue : PersonType → Statement → Prop
| PersonType.Knight, Statement.IAmALiar => False
| PersonType.Liar, Statement.IAmALiar => False
| PersonType.Normal, Statement.IAmALiar => True

theorem innocent_statement :
  ∀ (p : PersonType), p ≠ PersonType.Normal → ¬(isTrue p Statement.IAmALiar) := by
  sorry

end NUMINAMATH_CALUDE_innocent_statement_l1939_193972


namespace NUMINAMATH_CALUDE_pirate_treasure_sum_l1939_193935

-- Define a function to convert from base 8 to base 10
def base8ToBase10 (n : Nat) : Nat :=
  let digits := n.digits 8
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Define the theorem
theorem pirate_treasure_sum :
  let silk := 5267
  let stones := 6712
  let spices := 327
  base8ToBase10 silk + base8ToBase10 stones + base8ToBase10 spices = 6488 := by
  sorry


end NUMINAMATH_CALUDE_pirate_treasure_sum_l1939_193935


namespace NUMINAMATH_CALUDE_only_parallelogram_coincides_l1939_193915

-- Define the shapes
inductive Shape
  | Parallelogram
  | EquilateralTriangle
  | IsoscelesRightTriangle
  | RegularPentagon

-- Define a function to check if a shape coincides with itself after 180° rotation
def coincides_after_180_rotation (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | _ => False

-- Theorem statement
theorem only_parallelogram_coincides :
  ∀ (s : Shape), coincides_after_180_rotation s ↔ s = Shape.Parallelogram :=
by sorry

end NUMINAMATH_CALUDE_only_parallelogram_coincides_l1939_193915


namespace NUMINAMATH_CALUDE_tangent_curves_a_value_l1939_193950

theorem tangent_curves_a_value (a : ℝ) : 
  let f (x : ℝ) := x + Real.log x
  let g (x : ℝ) := a * x^2 + (a + 2) * x + 1
  let f' (x : ℝ) := 1 + 1 / x
  let g' (x : ℝ) := 2 * a * x + (a + 2)
  (f 1 = g 1) ∧ 
  (f' 1 = g' 1) ∧ 
  (∀ x ≠ 1, f x ≠ g x) →
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_tangent_curves_a_value_l1939_193950


namespace NUMINAMATH_CALUDE_lewis_weekly_earnings_l1939_193968

/-- Lewis's earnings during harvest -/
def harvest_earnings : ℕ := 178

/-- Duration of harvest in weeks -/
def harvest_duration : ℕ := 89

/-- Lewis's weekly earnings during harvest -/
def weekly_earnings : ℚ := harvest_earnings / harvest_duration

theorem lewis_weekly_earnings :
  weekly_earnings = 2 :=
sorry

end NUMINAMATH_CALUDE_lewis_weekly_earnings_l1939_193968


namespace NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l1939_193963

theorem square_sum_from_sum_and_product (x y : ℝ) :
  x + y = 5 → x * y = 6 → x^2 + y^2 = 13 := by sorry

end NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l1939_193963


namespace NUMINAMATH_CALUDE_range_of_m_l1939_193985

/-- Given the conditions:
    1. p: |4-x| ≤ 6
    2. q: x^2 - 2x + 1 ≤ 0 (m > 0)
    3. p is not a necessary but not sufficient condition for q
    
    Prove that the range of values for the real number m is m ≥ 9. -/
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |4 - x| ≤ 6 → (x^2 - 2*x + 1 ≤ 0 ∧ m > 0)) →
  (∃ x : ℝ, |4 - x| ≤ 6 ∧ (x^2 - 2*x + 1 > 0 ∨ m ≤ 0)) →
  (∀ x : ℝ, (x^2 - 2*x + 1 ≤ 0 ∧ m > 0) → |4 - x| ≤ 6) →
  m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1939_193985


namespace NUMINAMATH_CALUDE_perpendicular_lines_and_circle_l1939_193953

-- Define the lines and circle
def l₁ (a x y : ℝ) : Prop := a * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def C (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 8*y + 21 = 0

-- State the theorem
theorem perpendicular_lines_and_circle 
  (a : ℝ) -- Coefficient of x in l₁
  (h_perp : a * 2 + 4 = 0) -- Perpendicularity condition
  : 
  -- Part 1: Intersection point
  (∃ x y : ℝ, l₁ a x y ∧ l₂ x y ∧ x = -1 ∧ y = 0) ∧ 
  -- Part 2: No common points between l₁ and C
  (∀ x y : ℝ, ¬(l₁ a x y ∧ C x y)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_and_circle_l1939_193953


namespace NUMINAMATH_CALUDE_min_sum_of_product_1800_l1939_193989

theorem min_sum_of_product_1800 (a b c : ℕ+) (h : a * b * c = 1800) :
  (∀ x y z : ℕ+, x * y * z = 1800 → a + b + c ≤ x + y + z) ∧ a + b + c = 64 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1800_l1939_193989


namespace NUMINAMATH_CALUDE_four_balls_four_boxes_l1939_193979

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls_boxes (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 5 ways to distribute 4 indistinguishable balls into 4 indistinguishable boxes -/
theorem four_balls_four_boxes : distribute_balls_boxes 4 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_balls_four_boxes_l1939_193979


namespace NUMINAMATH_CALUDE_tax_discount_order_invariance_l1939_193911

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_order_invariance 
  (original_price tax_rate discount_rate : ℝ) 
  (hp : 0 < original_price) 
  (ht : 0 ≤ tax_rate) 
  (hd : 0 ≤ discount_rate) 
  (hd1 : discount_rate ≤ 1) :
  original_price * (1 + tax_rate) * (1 - discount_rate) = 
  original_price * (1 - discount_rate) * (1 + tax_rate) :=
sorry

end NUMINAMATH_CALUDE_tax_discount_order_invariance_l1939_193911


namespace NUMINAMATH_CALUDE_salary_calculation_l1939_193949

theorem salary_calculation (salary : ℚ) 
  (food : ℚ) (rent : ℚ) (clothes : ℚ) (transport : ℚ) (personal_care : ℚ) 
  (remaining : ℚ) :
  food = 1/4 * salary →
  rent = 1/6 * salary →
  clothes = 3/8 * salary →
  transport = 1/12 * salary →
  personal_care = 1/24 * salary →
  remaining = 45000 →
  salary - (food + rent + clothes + transport + personal_care) = remaining →
  salary = 540000 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l1939_193949


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1939_193954

theorem fraction_subtraction : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1939_193954


namespace NUMINAMATH_CALUDE_point_distance_constraint_l1939_193905

/-- Given points A(1, 0) and B(4, 0), and a point P on the line x + my - 1 = 0
    such that |PA| = 2|PB|, prove that the range of values for m is m ≥ √3 or m ≤ -√3. -/
theorem point_distance_constraint (m : ℝ) : 
  (∃ (x y : ℝ), x + m * y - 1 = 0 ∧ 
   (x - 1)^2 + y^2 = 4 * ((x - 4)^2 + y^2)) ↔ 
  (m ≥ Real.sqrt 3 ∨ m ≤ -Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_point_distance_constraint_l1939_193905


namespace NUMINAMATH_CALUDE_number_factorization_l1939_193999

theorem number_factorization (n : ℤ) : 
  (∃ x y : ℤ, n = x * y ∧ y - x = 6 ∧ x^4 + y^4 = 272) → n = -8 := by
  sorry

end NUMINAMATH_CALUDE_number_factorization_l1939_193999


namespace NUMINAMATH_CALUDE_intersection_point_congruences_l1939_193936

/-- Proves that (15, 8) is the unique intersection point of two congruences modulo 20 -/
theorem intersection_point_congruences : ∃! (x y : ℕ), 
  x < 20 ∧ 
  y < 20 ∧ 
  (7 * x + 3) % 20 = y ∧ 
  (13 * x + 18) % 20 = y :=
sorry

end NUMINAMATH_CALUDE_intersection_point_congruences_l1939_193936


namespace NUMINAMATH_CALUDE_program_output_correct_verify_output_l1939_193907

/-- Represents the result of the program execution -/
structure ProgramResult where
  x : Int
  y : Int

/-- Executes the program logic based on initial values -/
def executeProgram (initialX initialY : Int) : ProgramResult :=
  if initialX < 0 then
    { x := initialY - 4, y := initialY }
  else
    { x := initialX, y := initialY + 4 }

/-- Theorem stating the program output for given initial values -/
theorem program_output_correct :
  let result := executeProgram 2 (-30)
  result.x - result.y = 28 ∧ result.y - result.x = -28 := by
  sorry

/-- Verifies that the program output matches the expected result -/
theorem verify_output :
  let result := executeProgram 2 (-30)
  (result.x - result.y, result.y - result.x) = (28, -28) := by
  sorry

end NUMINAMATH_CALUDE_program_output_correct_verify_output_l1939_193907


namespace NUMINAMATH_CALUDE_pens_after_sale_l1939_193910

theorem pens_after_sale (initial_pens : ℕ) (sold_pens : ℕ) (h1 : initial_pens = 106) (h2 : sold_pens = 92) :
  initial_pens - sold_pens = 14 := by
  sorry

end NUMINAMATH_CALUDE_pens_after_sale_l1939_193910


namespace NUMINAMATH_CALUDE_jeremy_earnings_l1939_193920

theorem jeremy_earnings (steven_rate mark_rate steven_rooms mark_rooms : ℚ) 
  (h1 : steven_rate = 12 / 3)
  (h2 : mark_rate = 10 / 4)
  (h3 : steven_rooms = 8 / 3)
  (h4 : mark_rooms = 9 / 4) :
  steven_rate * steven_rooms + mark_rate * mark_rooms = 391 / 24 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_earnings_l1939_193920


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1939_193932

theorem simplify_trig_expression :
  7 * 8 * (Real.sin (10 * π / 180) + Real.sin (20 * π / 180)) /
  (Real.cos (10 * π / 180) + Real.cos (20 * π / 180)) =
  Real.tan (15 * π / 180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1939_193932


namespace NUMINAMATH_CALUDE_complex_triangle_problem_l1939_193976

theorem complex_triangle_problem (x y z : ℂ) 
  (eq1 : x^2 + y^2 + z^2 = x*y + y*z + z*x)
  (eq2 : Complex.abs (x + y + z) = 21)
  (eq3 : Complex.abs (x - y) = 2 * Real.sqrt 3)
  (eq4 : Complex.abs x = 3 * Real.sqrt 3) :
  Complex.abs y^2 + Complex.abs z^2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_complex_triangle_problem_l1939_193976


namespace NUMINAMATH_CALUDE_white_marble_probability_l1939_193966

theorem white_marble_probability (total_marbles : ℕ) 
  (p_green p_red_or_blue : ℝ) : 
  total_marbles = 84 →
  p_green = 2 / 7 →
  p_red_or_blue = 0.4642857142857143 →
  1 - (p_green + p_red_or_blue) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_white_marble_probability_l1939_193966


namespace NUMINAMATH_CALUDE_probability_is_34_39_l1939_193938

-- Define the total number of students and enrollments
def total_students : ℕ := 40
def french_enrollment : ℕ := 28
def spanish_enrollment : ℕ := 26
def german_enrollment : ℕ := 15
def french_spanish : ℕ := 10
def french_german : ℕ := 6
def spanish_german : ℕ := 8
def all_three : ℕ := 3

-- Define the function to calculate the probability
def probability_different_classes : ℚ := by sorry

-- Theorem statement
theorem probability_is_34_39 : 
  probability_different_classes = 34 / 39 := by sorry

end NUMINAMATH_CALUDE_probability_is_34_39_l1939_193938


namespace NUMINAMATH_CALUDE_geometric_sequence_m_range_l1939_193965

theorem geometric_sequence_m_range 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (m : ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = q * a n)
  (h_q_range : q > Real.rpow 5 (1/3) ∧ q < 2)
  (h_equation : m * a 6 * a 7 = a 8 ^ 2 - 2 * a 4 * a 9) :
  m > 3 ∧ m < 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_m_range_l1939_193965


namespace NUMINAMATH_CALUDE_henrys_deductions_l1939_193952

/-- Henry's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- State tax rate as a decimal -/
def tax_rate : ℚ := 21 / 1000

/-- Fixed community fee in dollars per hour -/
def community_fee : ℚ := 1 / 2

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℚ := 100

/-- Calculate the total deductions in cents -/
def total_deductions : ℚ :=
  hourly_wage * tax_rate * dollars_to_cents + community_fee * dollars_to_cents

theorem henrys_deductions :
  total_deductions = 102.5 := by sorry

end NUMINAMATH_CALUDE_henrys_deductions_l1939_193952


namespace NUMINAMATH_CALUDE_emily_flower_spending_l1939_193934

def flower_price : ℝ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2
def discount_threshold : ℕ := 3
def discount_rate : ℝ := 0.2

def total_flowers : ℕ := roses_bought + daisies_bought

def apply_discount (price : ℝ) : ℝ :=
  if total_flowers > discount_threshold then
    price * (1 - discount_rate)
  else
    price

theorem emily_flower_spending :
  apply_discount (flower_price * (roses_bought + daisies_bought : ℝ)) = 9.60 := by
  sorry

end NUMINAMATH_CALUDE_emily_flower_spending_l1939_193934


namespace NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l1939_193998

theorem sum_of_quotient_dividend_divisor (N : ℕ) (h : N = 40) : 
  (N / 2) + N + 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l1939_193998


namespace NUMINAMATH_CALUDE_arithmetic_mean_implies_arithmetic_progression_geometric_mean_implies_geometric_progression_l1939_193908

/-- A sequence is an arithmetic progression if the difference between consecutive terms is constant. -/
def IsArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- A sequence is a geometric progression if the ratio between consecutive terms is constant. -/
def IsGeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) / a n = r

/-- Theorem: If each term (from the second to the second-to-last) in a sequence is the arithmetic mean
    of its neighboring terms, then the sequence is an arithmetic progression. -/
theorem arithmetic_mean_implies_arithmetic_progression (a : ℕ → ℝ) (n : ℕ) (h : n ≥ 3)
    (h_arithmetic_mean : ∀ k, 2 ≤ k ∧ k < n → a k = (a (k - 1) + a (k + 1)) / 2) :
    IsArithmeticProgression a := by sorry

/-- Theorem: If each term (from the second to the second-to-last) in a sequence is the geometric mean
    of its neighboring terms, then the sequence is a geometric progression. -/
theorem geometric_mean_implies_geometric_progression (a : ℕ → ℝ) (n : ℕ) (h : n ≥ 3)
    (h_geometric_mean : ∀ k, 2 ≤ k ∧ k < n → a k = Real.sqrt (a (k - 1) * a (k + 1))) :
    IsGeometricProgression a := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_implies_arithmetic_progression_geometric_mean_implies_geometric_progression_l1939_193908


namespace NUMINAMATH_CALUDE_solution_when_k_gt_neg_one_no_solution_when_k_eq_neg_one_solution_when_k_lt_neg_one_k_upper_bound_l1939_193946

/-- The function f(x) defined in the problem -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (1 - k)*x + 2 - k

/-- Theorem stating the solution for f(x) < 2 when k > -1 -/
theorem solution_when_k_gt_neg_one (k : ℝ) (x : ℝ) (h : k > -1) :
  f k x < 2 ↔ -1 < x ∧ x < k :=
sorry

/-- Theorem stating there's no solution for f(x) < 2 when k = -1 -/
theorem no_solution_when_k_eq_neg_one (x : ℝ) :
  ¬(f (-1) x < 2) :=
sorry

/-- Theorem stating the solution for f(x) < 2 when k < -1 -/
theorem solution_when_k_lt_neg_one (k : ℝ) (x : ℝ) (h : k < -1) :
  f k x < 2 ↔ k < x ∧ x < -1 :=
sorry

/-- Theorem stating the upper bound of k when f(n) + 11 ≥ 0 for all natural numbers n -/
theorem k_upper_bound (k : ℝ) (h : ∀ (n : ℕ), f k n + 11 ≥ 0) :
  k ≤ 25/4 :=
sorry

end NUMINAMATH_CALUDE_solution_when_k_gt_neg_one_no_solution_when_k_eq_neg_one_solution_when_k_lt_neg_one_k_upper_bound_l1939_193946


namespace NUMINAMATH_CALUDE_allowance_increase_l1939_193958

/-- The base amount of Kathleen's middle school allowance -/
def base_amount : ℝ := 8

/-- Kathleen's middle school allowance -/
def middle_school_allowance (x : ℝ) : ℝ := x + 2

/-- Kathleen's senior year allowance -/
def senior_year_allowance (x : ℝ) : ℝ := 5 + 2 * (x + 2)

/-- The percentage increase in Kathleen's weekly allowance -/
def percentage_increase : ℝ := 150

theorem allowance_increase (x : ℝ) :
  x = base_amount ↔
  (1 + percentage_increase / 100) * middle_school_allowance x = senior_year_allowance x :=
sorry

end NUMINAMATH_CALUDE_allowance_increase_l1939_193958


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l1939_193947

-- Define a function to represent the power tower of 2's
def powerTower (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 ^ (powerTower n)

-- Define the right-hand side of the inequality
def rightHandSide : ℕ := 3^(3^(3^3))

-- Theorem statement
theorem smallest_n_for_inequality :
  (∀ k < 6, powerTower k ≤ rightHandSide) ∧
  (powerTower 6 > rightHandSide) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l1939_193947


namespace NUMINAMATH_CALUDE_union_equals_reals_subset_condition_l1939_193994

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a < x ∧ x < 3 + a}
def B : Set ℝ := {x | x ≤ -1 ∨ x ≥ 1}

-- Theorem 1: A ∪ B = ℝ iff -2 ≤ a ≤ -1
theorem union_equals_reals (a : ℝ) : A a ∪ B = Set.univ ↔ -2 ≤ a ∧ a ≤ -1 := by
  sorry

-- Theorem 2: A ⊆ B iff a ≤ -4 or a ≥ 1
theorem subset_condition (a : ℝ) : A a ⊆ B ↔ a ≤ -4 ∨ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_reals_subset_condition_l1939_193994


namespace NUMINAMATH_CALUDE_special_line_equation_l1939_193991

/-- A line passing through point (2, 3) with intercepts on the coordinate axes that are opposite numbers -/
structure SpecialLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2, 3)
  passes_through : m * 2 + b = 3
  -- The intercepts are opposite numbers
  opposite_intercepts : b = m * b

theorem special_line_equation (L : SpecialLine) :
  (L.m = 3/2 ∧ L.b = 0) ∨ (L.m = 1 ∧ L.b = -1) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l1939_193991


namespace NUMINAMATH_CALUDE_percentage_difference_l1939_193988

theorem percentage_difference : (70 / 100 * 100) - (60 / 100 * 80) = 22 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1939_193988


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1939_193986

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  S : ℕ → ℝ
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Theorem: For an arithmetic sequence where S_17 = 17/2, a_3 + a_15 = 1 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
    (h : seq.S 17 = 17 / 2) : seq.a 3 + seq.a 15 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1939_193986


namespace NUMINAMATH_CALUDE_symmetric_polynomial_square_factor_l1939_193993

/-- A polynomial in two variables that is symmetric in its arguments -/
def SymmetricPolynomial (R : Type) [CommRing R] :=
  {p : R → R → R // ∀ x y, p x y = p y x}

theorem symmetric_polynomial_square_factor
  {R : Type} [CommRing R] (p : SymmetricPolynomial R)
  (h : ∃ q : R → R → R, ∀ x y, p.val x y = (x - y) * q x y) :
  ∃ r : R → R → R, ∀ x y, p.val x y = (x - y)^2 * r x y := by
  sorry

end NUMINAMATH_CALUDE_symmetric_polynomial_square_factor_l1939_193993


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1939_193923

theorem trigonometric_identity (x y : Real) 
  (h : Real.cos (x + y) = 2 / 3) : 
  Real.sin (x - 3 * Real.pi / 10) * Real.cos (y - Real.pi / 5) - 
  Real.sin (x + Real.pi / 5) * Real.cos (y + 3 * Real.pi / 10) = 
  -2 / 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1939_193923


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1939_193983

/-- Proves that the annual interest rate is 5% given the specified conditions -/
theorem interest_rate_proof (principal : ℝ) (time : ℕ) (amount : ℝ) :
  principal = 973.913043478261 →
  time = 3 →
  amount = 1120 →
  (amount - principal) / (principal * time) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l1939_193983


namespace NUMINAMATH_CALUDE_min_queries_needed_l1939_193957

/-- Represents a quadratic polynomial ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates a quadratic polynomial at a given point -/
def evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Represents Petia's strategy of choosing which polynomial value to return -/
def PetiaStrategy := ℕ → Bool

/-- Represents Vasya's strategy of choosing query points -/
def VasyaStrategy := ℕ → ℝ

/-- Determines if Vasya can identify one of Petia's polynomials after n queries -/
def canIdentifyPolynomial (f g : QuadraticPolynomial) (petiaStrat : PetiaStrategy) (vasyaStrat : VasyaStrategy) (n : ℕ) : Prop :=
  ∃ (i : Fin n), 
    let x := vasyaStrat i
    let y := if petiaStrat i then evaluate f x else evaluate g x
    ∀ (f' g' : QuadraticPolynomial), 
      (∀ (j : Fin n), 
        let x' := vasyaStrat j
        let y' := if petiaStrat j then evaluate f' x' else evaluate g' x'
        y' = if petiaStrat j then evaluate f x' else evaluate g x') →
      f' = f ∨ g' = g

/-- The main theorem: 8 is the smallest number of queries needed -/
theorem min_queries_needed : 
  (∃ (vasyaStrat : VasyaStrategy), ∀ (f g : QuadraticPolynomial) (petiaStrat : PetiaStrategy), 
    canIdentifyPolynomial f g petiaStrat vasyaStrat 8) ∧ 
  (∀ (n : ℕ), n < 8 → 
    ∀ (vasyaStrat : VasyaStrategy), ∃ (f g : QuadraticPolynomial) (petiaStrat : PetiaStrategy), 
      ¬canIdentifyPolynomial f g petiaStrat vasyaStrat n) := by
  sorry

end NUMINAMATH_CALUDE_min_queries_needed_l1939_193957


namespace NUMINAMATH_CALUDE_coefficient_x_fifth_power_l1939_193981

theorem coefficient_x_fifth_power (x : ℝ) : 
  (Finset.range 10).sum (λ k => (Nat.choose 9 k : ℝ) * x^(9 - k) * (3 * Real.sqrt 2)^k) = 
  40824 * x^5 + (Finset.range 10).sum (λ k => if k ≠ 4 then (Nat.choose 9 k : ℝ) * x^(9 - k) * (3 * Real.sqrt 2)^k else 0) := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_fifth_power_l1939_193981


namespace NUMINAMATH_CALUDE_group_formation_count_l1939_193914

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of groups that can be formed with one boy and two girls -/
def oneBoytwoGirls (boys girls : ℕ) : ℕ := 
  binomial boys 1 * binomial girls 2

/-- The number of groups that can be formed with two boys and one girl -/
def twoBoyoneGirl (boys girls : ℕ) : ℕ := 
  binomial boys 2 * binomial girls 1

/-- The total number of valid groups that can be formed -/
def totalGroups (boys girls : ℕ) : ℕ := 
  oneBoytwoGirls boys girls + twoBoyoneGirl boys girls

theorem group_formation_count :
  totalGroups 9 12 = 1026 := by sorry

end NUMINAMATH_CALUDE_group_formation_count_l1939_193914


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1939_193922

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -1/2x - 2 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = -1/2 * x - 2
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y ↔ y = 1/2 * x - 3/2) →  -- Slope of L1 is 1/2
  (L2 P.1 P.2) →  -- L2 passes through P
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (x₂ - x₁) * (-1/2) = -(y₂ - y₁) / (x₂ - x₁)) →  -- L1 and L2 are perpendicular
  L2 x y
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1939_193922


namespace NUMINAMATH_CALUDE_average_shift_l1939_193909

theorem average_shift (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 40) :
  ((x₁ + 40) + (x₂ + 40) + (x₃ + 40)) / 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_shift_l1939_193909


namespace NUMINAMATH_CALUDE_abs_ratio_greater_than_one_l1939_193961

theorem abs_ratio_greater_than_one {a b : ℝ} (h1 : a < b) (h2 : b < 0) : |a| / |b| > 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_greater_than_one_l1939_193961


namespace NUMINAMATH_CALUDE_luke_birthday_stickers_l1939_193927

/-- Represents the number of stickers Luke has at different stages --/
structure StickerCount where
  initial : ℕ
  bought : ℕ
  birthday : ℕ
  given_away : ℕ
  used : ℕ
  final : ℕ

/-- Calculates the number of stickers Luke got for his birthday --/
def birthday_stickers (s : StickerCount) : ℕ :=
  s.final + s.given_away + s.used - s.initial - s.bought

/-- Theorem stating that Luke got 20 stickers for his birthday --/
theorem luke_birthday_stickers :
  ∀ s : StickerCount,
    s.initial = 20 ∧
    s.bought = 12 ∧
    s.given_away = 5 ∧
    s.used = 8 ∧
    s.final = 39 →
    birthday_stickers s = 20 := by
  sorry


end NUMINAMATH_CALUDE_luke_birthday_stickers_l1939_193927


namespace NUMINAMATH_CALUDE_derivative_problems_l1939_193956

open Real

theorem derivative_problems :
  (∀ x : ℝ, x ≠ 0 → deriv (λ x => x * (1 + 2/x + 2/x^2)) x = 1 - 2/x^2) ∧
  (∀ x : ℝ, deriv (λ x => x^4 - 3*x^2 - 5*x + 6) x = 4*x^3 - 6*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_derivative_problems_l1939_193956


namespace NUMINAMATH_CALUDE_lcm_fraction_even_l1939_193960

theorem lcm_fraction_even (n : ℕ) : 
  (n > 0) → (∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    n = (Nat.lcm x y + Nat.lcm y z) / Nat.lcm x z) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_lcm_fraction_even_l1939_193960


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1939_193900

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ x y : ℤ, f (x + y) = f x + f y - 2023) : 
  ∃ c : ℤ, ∀ x : ℤ, f x = c * x + 2023 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1939_193900


namespace NUMINAMATH_CALUDE_xyz_congruence_l1939_193912

theorem xyz_congruence (x y z : ℕ) : 
  x < 8 → y < 8 → z < 8 →
  (x + 3*y + 2*z) % 8 = 1 →
  (2*x + y + 3*z) % 8 = 5 →
  (3*x + 2*y + z) % 8 = 3 →
  (x*y*z) % 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_xyz_congruence_l1939_193912


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l1939_193926

theorem chinese_remainder_theorem_example :
  ∃ x : ℤ, (x ≡ 1 [ZMOD 3] ∧
             x ≡ -1 [ZMOD 5] ∧
             x ≡ 2 [ZMOD 7] ∧
             x ≡ -2 [ZMOD 11]) ↔
            x ≡ 394 [ZMOD 1155] := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l1939_193926


namespace NUMINAMATH_CALUDE_production_scale_l1939_193901

/-- Production function that calculates the number of items produced given the number of workers, hours per day, number of days, and production rate. -/
def production (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) (rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours_per_day : ℚ) * (days : ℚ) * rate

/-- Theorem stating that if 8 workers produce 512 items in 8 hours a day for 8 days, 
    then 10 workers working 10 hours a day for 10 days will produce 1000 items, 
    assuming a constant production rate. -/
theorem production_scale (rate : ℚ) : 
  production 8 8 8 rate = 512 → production 10 10 10 rate = 1000 := by
  sorry

#check production_scale

end NUMINAMATH_CALUDE_production_scale_l1939_193901


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1939_193975

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_ones : ones < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Checks if all digits in a three-digit number are the same -/
def allDigitsSame (n : Nat) : Prop :=
  (n / 100 = (n / 10) % 10) ∧ ((n / 10) % 10 = n % 10)

theorem unique_three_digit_number :
  ∃! (n : ThreeDigitNumber),
    (n.hundreds + n.ones = 5) ∧
    (n.tens = 3) ∧
    (n.hundreds ≠ n.tens) ∧
    (n.tens ≠ n.ones) ∧
    (n.hundreds ≠ n.ones) ∧
    allDigitsSame (n.toNat + 124) ∧
    n.toNat = 431 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1939_193975


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l1939_193917

theorem abc_sum_sqrt (a b c : ℝ) 
  (eq1 : b + c = 20) 
  (eq2 : c + a = 22) 
  (eq3 : a + b = 24) : 
  Real.sqrt (a * b * c * (a + b + c)) = 357 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l1939_193917


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l1939_193919

theorem min_four_dollar_frisbees 
  (total_frisbees : ℕ) 
  (total_receipts : ℕ) 
  (h_total : total_frisbees = 60) 
  (h_receipts : total_receipts = 204) : 
  ∃ (three_dollar : ℕ) (four_dollar : ℕ), 
    three_dollar + four_dollar = total_frisbees ∧ 
    3 * three_dollar + 4 * four_dollar = total_receipts ∧ 
    four_dollar ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l1939_193919


namespace NUMINAMATH_CALUDE_range_of_a_l1939_193974

theorem range_of_a (a : ℝ) : 
  (∀ x θ : ℝ, θ ∈ Set.Icc 0 (Real.pi / 2) → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1939_193974


namespace NUMINAMATH_CALUDE_third_side_length_l1939_193918

theorem third_side_length (a b c : ℝ) : 
  a = 4 → b = 10 → c = 12 →
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end NUMINAMATH_CALUDE_third_side_length_l1939_193918


namespace NUMINAMATH_CALUDE_west_is_negative_of_east_l1939_193939

/-- Represents distance and direction, where positive values indicate east and negative values indicate west. -/
def Distance := ℤ

/-- Converts a distance in kilometers to the corresponding Distance representation. -/
def km_to_distance (x : ℤ) : Distance := x

/-- The distance representation for 2km east. -/
def two_km_east : Distance := km_to_distance 2

/-- The distance representation for 1km west. -/
def one_km_west : Distance := km_to_distance (-1)

theorem west_is_negative_of_east (h : two_km_east = km_to_distance 2) :
  one_km_west = km_to_distance (-1) := by sorry

end NUMINAMATH_CALUDE_west_is_negative_of_east_l1939_193939


namespace NUMINAMATH_CALUDE_broadcast_end_date_prove_broadcast_end_date_l1939_193944

/-- Represents a date with year, month, and day. -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Represents a day of the week. -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the broadcasting schedule. -/
structure BroadcastSchedule where
  wednesday : Nat
  friday : Nat
  saturday : Nat
  sunday : Nat

/-- Calculates the end date of the broadcast. -/
def calculateEndDate (startDate : Date) (totalEpisodes : Nat) (schedule : BroadcastSchedule) : Date :=
  sorry

/-- Determines the day of the week for a given date. -/
def getDayOfWeek (date : Date) : DayOfWeek :=
  sorry

/-- Main theorem to prove -/
theorem broadcast_end_date (startDate : Date) (totalEpisodes : Nat) (schedule : BroadcastSchedule) :
  let endDate := calculateEndDate startDate totalEpisodes schedule
  endDate.year = 2016 ∧ endDate.month = 5 ∧ endDate.day = 29 ∧
  getDayOfWeek endDate = DayOfWeek.Sunday :=
by
  sorry

/-- Initial conditions -/
def initialDate : Date := { year := 2015, month := 12, day := 26 }
def episodeCount : Nat := 135
def broadcastSchedule : BroadcastSchedule := { wednesday := 1, friday := 1, saturday := 2, sunday := 2 }

/-- Proof of the main theorem with initial conditions -/
theorem prove_broadcast_end_date :
  let endDate := calculateEndDate initialDate episodeCount broadcastSchedule
  endDate.year = 2016 ∧ endDate.month = 5 ∧ endDate.day = 29 ∧
  getDayOfWeek endDate = DayOfWeek.Sunday :=
by
  sorry

end NUMINAMATH_CALUDE_broadcast_end_date_prove_broadcast_end_date_l1939_193944


namespace NUMINAMATH_CALUDE_equivalent_angle_proof_l1939_193928

/-- The angle (in degrees) that has the same terminal side as -60° within [0°, 360°) -/
def equivalent_angle : ℝ := 300

theorem equivalent_angle_proof :
  ∃ (k : ℤ), equivalent_angle = k * 360 - 60 ∧ 
  0 ≤ equivalent_angle ∧ equivalent_angle < 360 :=
by sorry

end NUMINAMATH_CALUDE_equivalent_angle_proof_l1939_193928


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1939_193921

theorem angle_between_vectors (a b : ℝ × ℝ) : 
  a = (1, 2) → b = (-1, 3) → 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  θ = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1939_193921


namespace NUMINAMATH_CALUDE_find_A_l1939_193948

theorem find_A : ∃ A B : ℚ, A - 3 * B = 303.1 ∧ A = 10 * B → A = 433 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l1939_193948


namespace NUMINAMATH_CALUDE_existence_and_uniqueness_of_t_l1939_193980

def f (x : ℝ) := -x - 4

theorem existence_and_uniqueness_of_t :
  ∃! t : ℝ,
    (∀ x : ℝ, f x = -x - 4) ∧
    (f (-6) = 2 ∧ f 2 = -6) ∧
    (∀ k : ℝ, k > 0 → ∀ x : ℝ, f (x + k) < f x) ∧
    ({x : ℝ | |f (x - t) + 2| < 4} = Set.Ioo (-4 : ℝ) 4) :=
by sorry

#check existence_and_uniqueness_of_t

end NUMINAMATH_CALUDE_existence_and_uniqueness_of_t_l1939_193980


namespace NUMINAMATH_CALUDE_fred_initial_cards_l1939_193964

/-- Given that Fred gave away 18 cards, found 40 new cards, and ended up with 48 cards,
    prove that he must have started with 26 cards. -/
theorem fred_initial_cards :
  ∀ (initial_cards given_away new_cards final_cards : ℕ),
    given_away = 18 →
    new_cards = 40 →
    final_cards = 48 →
    initial_cards - given_away + new_cards = final_cards →
    initial_cards = 26 := by
  sorry

end NUMINAMATH_CALUDE_fred_initial_cards_l1939_193964


namespace NUMINAMATH_CALUDE_max_value_implies_a_l1939_193970

def f (x : ℝ) := x^2 - 2*x + 1

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 2), f x ≤ 4) ∧
  (∃ x ∈ Set.Icc a (a + 2), f x = 4) →
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l1939_193970


namespace NUMINAMATH_CALUDE_expand_expression_l1939_193904

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 * y - 2) = 36 * x + 48 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1939_193904


namespace NUMINAMATH_CALUDE_egg_problem_l1939_193992

theorem egg_problem (x : ℕ) : x > 0 ∧ 
  x % 2 = 1 ∧ 
  x % 3 = 1 ∧ 
  x % 4 = 1 ∧ 
  x % 5 = 1 ∧ 
  x % 6 = 1 ∧ 
  x % 7 = 0 → 
  x ≥ 301 :=
by sorry

end NUMINAMATH_CALUDE_egg_problem_l1939_193992


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l1939_193925

theorem average_of_a_and_b (a b c : ℝ) : 
  (b + c) / 2 = 50 → 
  c - a = 10 → 
  (a + b) / 2 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l1939_193925


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1939_193962

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ (∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0)) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1939_193962


namespace NUMINAMATH_CALUDE_solution_to_equation_l1939_193982

theorem solution_to_equation : ∃! x : ℤ, (2008 + x)^2 = x^2 ∧ x = -1004 := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1939_193982


namespace NUMINAMATH_CALUDE_hyperbola_distance_property_l1939_193977

/-- A point on a hyperbola with specific distance properties -/
structure HyperbolaPoint where
  P : ℝ × ℝ
  on_hyperbola : (P.1^2 / 4) - P.2^2 = 1
  distance_to_right_focus : Real.sqrt ((P.1 - Real.sqrt 5)^2 + P.2^2) = 5

/-- The theorem stating the distance property of the hyperbola point -/
theorem hyperbola_distance_property (hp : HyperbolaPoint) :
  let d := Real.sqrt ((hp.P.1 + Real.sqrt 5)^2 + hp.P.2^2)
  d = 1 ∨ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_distance_property_l1939_193977


namespace NUMINAMATH_CALUDE_units_digit_of_power_difference_l1939_193996

theorem units_digit_of_power_difference : ∃ n : ℕ, (5^2019 - 3^2019) % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_power_difference_l1939_193996


namespace NUMINAMATH_CALUDE_evaluate_expression_l1939_193959

theorem evaluate_expression : 2 - (-3) - 4 * (-5) - 6 - (-7) - 8 * (-9) + 10 = 108 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1939_193959


namespace NUMINAMATH_CALUDE_circular_road_width_l1939_193967

theorem circular_road_width 
  (inner_radius outer_radius : ℝ) 
  (h1 : 2 * Real.pi * inner_radius + 2 * Real.pi * outer_radius = 88) 
  (h2 : inner_radius = (1/3) * outer_radius) : 
  outer_radius - inner_radius = 22 / Real.pi := by
sorry

end NUMINAMATH_CALUDE_circular_road_width_l1939_193967


namespace NUMINAMATH_CALUDE_power_2017_mod_11_l1939_193984

theorem power_2017_mod_11 : 2^2017 % 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_2017_mod_11_l1939_193984


namespace NUMINAMATH_CALUDE_addition_puzzle_l1939_193924

theorem addition_puzzle (x y : ℕ) : 
  x ≠ y →
  x < 10 →
  y < 10 →
  307 + 700 + x = 1010 →
  y - x = 7 :=
by sorry

end NUMINAMATH_CALUDE_addition_puzzle_l1939_193924
