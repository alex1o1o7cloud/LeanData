import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_property_l4125_412523

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum_property
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_property_l4125_412523


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4125_412552

theorem triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  b / a = 4 / 3 →          -- ratio of second to first side is 4:3
  c / a = 5 / 3 →          -- ratio of third to first side is 5:3
  c - a = 6 →              -- difference between longest and shortest side is 6
  a + b + c = 36 :=        -- perimeter is 36
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4125_412552


namespace NUMINAMATH_CALUDE_inverse_composition_l4125_412577

-- Define the function f and its inverse
def f : ℝ → ℝ := sorry

def f_inv : ℝ → ℝ := sorry

-- Define the conditions
axiom f_4 : f 4 = 6
axiom f_6 : f 6 = 3
axiom f_3 : f 3 = 7
axiom f_7 : f 7 = 2

-- Define the inverse relationship
axiom f_inverse : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Theorem to prove
theorem inverse_composition :
  f_inv (f_inv 7 + f_inv 6) = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_l4125_412577


namespace NUMINAMATH_CALUDE_fencing_required_l4125_412548

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : 
  area > 0 → uncovered_side > 0 → area = uncovered_side * (area / uncovered_side) →
  2 * (area / uncovered_side) + uncovered_side = 32 := by
  sorry

#check fencing_required 120 20

end NUMINAMATH_CALUDE_fencing_required_l4125_412548


namespace NUMINAMATH_CALUDE_original_price_calculation_l4125_412566

/-- The original price of a meal given the total amount paid and various fees and discounts -/
theorem original_price_calculation (total_paid : ℝ) (discount_rate : ℝ) (sales_tax_rate : ℝ) 
  (service_fee_rate : ℝ) (tip_rate : ℝ) (h_total : total_paid = 165) 
  (h_discount : discount_rate = 0.15) (h_sales_tax : sales_tax_rate = 0.10) 
  (h_service_fee : service_fee_rate = 0.05) (h_tip : tip_rate = 0.20) :
  ∃ (P : ℝ), P = total_paid / ((1 - discount_rate) * (1 + sales_tax_rate + service_fee_rate) * (1 + tip_rate)) := by
  sorry

#eval (165 : Float) / (0.85 * 1.15 * 1.20)

end NUMINAMATH_CALUDE_original_price_calculation_l4125_412566


namespace NUMINAMATH_CALUDE_bananas_arrangements_eq_240_l4125_412578

/-- The number of letters in the word BANANAS -/
def total_letters : ℕ := 7

/-- The number of 'B's in BANANAS -/
def count_B : ℕ := 1

/-- The number of 'A's in BANANAS -/
def count_A : ℕ := 3

/-- The number of 'N's in BANANAS -/
def count_N : ℕ := 1

/-- The number of 'S's in BANANAS -/
def count_S : ℕ := 2

/-- The function to calculate the number of arrangements of BANANAS with no 'A' at the first position -/
def bananas_arrangements : ℕ := sorry

/-- Theorem stating that the number of arrangements of BANANAS with no 'A' at the first position is 240 -/
theorem bananas_arrangements_eq_240 : bananas_arrangements = 240 := by sorry

end NUMINAMATH_CALUDE_bananas_arrangements_eq_240_l4125_412578


namespace NUMINAMATH_CALUDE_amusement_park_revenue_l4125_412525

def ticket_price : ℕ := 3
def weekday_visitors : ℕ := 100
def saturday_visitors : ℕ := 200
def sunday_visitors : ℕ := 300
def days_in_week : ℕ := 7
def weekdays : ℕ := 5

def total_revenue : ℕ := ticket_price * (weekday_visitors * weekdays + saturday_visitors + sunday_visitors)

theorem amusement_park_revenue : total_revenue = 3000 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_revenue_l4125_412525


namespace NUMINAMATH_CALUDE_train_speed_problem_l4125_412573

/-- Given a train that covers a distance in 3 hours at its initial speed,
    and covers the same distance in 1 hour at 450 kmph,
    prove that its initial speed is 150 kmph. -/
theorem train_speed_problem (distance : ℝ) (initial_speed : ℝ) : 
  distance = initial_speed * 3 → distance = 450 * 1 → initial_speed = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l4125_412573


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l4125_412597

theorem repeating_decimal_division :
  let x : ℚ := 63 / 99
  let y : ℚ := 84 / 99
  x / y = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l4125_412597


namespace NUMINAMATH_CALUDE_concrete_volume_l4125_412555

-- Define constants
def sidewalk_width : ℚ := 4/3  -- in yards
def sidewalk_length : ℚ := 80/3  -- in yards
def sidewalk_thickness : ℚ := 1/9  -- in yards
def border_width : ℚ := 2/3  -- in yards (1 foot on each side)
def border_thickness : ℚ := 1/18  -- in yards

-- Define the theorem
theorem concrete_volume : 
  let sidewalk_volume := sidewalk_width * sidewalk_length * sidewalk_thickness
  let border_volume := border_width * sidewalk_length * border_thickness
  let total_volume := sidewalk_volume + border_volume
  ⌈total_volume⌉ = 6 := by
sorry


end NUMINAMATH_CALUDE_concrete_volume_l4125_412555


namespace NUMINAMATH_CALUDE_implication_equivalence_l4125_412530

theorem implication_equivalence (R S : Prop) :
  (R → S) ↔ (¬S → ¬R) := by sorry

end NUMINAMATH_CALUDE_implication_equivalence_l4125_412530


namespace NUMINAMATH_CALUDE_total_selections_is_57_l4125_412582

/-- Represents the arrangement of circles in the figure -/
structure CircleArrangement where
  total_circles : Nat
  horizontal_rows : List Nat
  diagonal_length : Nat

/-- Calculates the number of ways to select three consecutive circles in a row -/
def consecutive_selections (row_length : Nat) : Nat :=
  max (row_length - 2) 0

/-- Calculates the total number of ways to select three consecutive circles in the figure -/
def total_selections (arrangement : CircleArrangement) : Nat :=
  let horizontal_selections := arrangement.horizontal_rows.map consecutive_selections |>.sum
  let diagonal_selections := List.range arrangement.diagonal_length |>.map consecutive_selections |>.sum
  horizontal_selections + 2 * diagonal_selections

/-- The main theorem stating that the total number of selections is 57 -/
theorem total_selections_is_57 (arrangement : CircleArrangement) :
  arrangement.total_circles = 33 →
  arrangement.horizontal_rows = [6, 5, 4, 3, 2, 1] →
  arrangement.diagonal_length = 6 →
  total_selections arrangement = 57 := by
  sorry

#eval total_selections { total_circles := 33, horizontal_rows := [6, 5, 4, 3, 2, 1], diagonal_length := 6 }

end NUMINAMATH_CALUDE_total_selections_is_57_l4125_412582


namespace NUMINAMATH_CALUDE_x_plus_y_values_l4125_412508

theorem x_plus_y_values (x y : ℝ) 
  (eq1 : x^2 + x*y + 2*y = 10) 
  (eq2 : y^2 + x*y + 2*x = 14) : 
  x + y = 4 ∨ x + y = -6 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l4125_412508


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l4125_412514

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / (x - 2)) ↔ (x ≥ -1 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l4125_412514


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_10_l4125_412510

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_10 :
  unitsDigit (sumFactorials 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_10_l4125_412510


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l4125_412595

theorem quadratic_equation_properties (m : ℝ) :
  (∀ x, x^2 - (2*m - 3)*x + m^2 + 1 = 0 → x = m) →
    m = -1/3 ∧
  m < 0 →
    (2*m - 3)^2 - 4*(m^2 + 1) > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l4125_412595


namespace NUMINAMATH_CALUDE_square_difference_equality_l4125_412579

theorem square_difference_equality : (25 + 15)^2 - (25 - 15)^2 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l4125_412579


namespace NUMINAMATH_CALUDE_maryville_population_increase_l4125_412587

/-- The average annual population increase in Maryville between 2000 and 2005 -/
def average_annual_increase (pop_2000 pop_2005 : ℕ) : ℚ :=
  (pop_2005 - pop_2000 : ℚ) / 5

/-- Theorem stating the average annual population increase in Maryville between 2000 and 2005 -/
theorem maryville_population_increase :
  average_annual_increase 450000 467000 = 3400 := by
  sorry

end NUMINAMATH_CALUDE_maryville_population_increase_l4125_412587


namespace NUMINAMATH_CALUDE_contrapositive_absolute_value_l4125_412545

theorem contrapositive_absolute_value (a b : ℝ) :
  (¬(|a| > |b|) → ¬(a > b)) ↔ (|a| ≤ |b| → a ≤ b) := by sorry

end NUMINAMATH_CALUDE_contrapositive_absolute_value_l4125_412545


namespace NUMINAMATH_CALUDE_age_difference_ratio_l4125_412516

/-- Represents the current ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.roy = ages.julia + 8 ∧
  ages.roy + 2 = 2 * (ages.julia + 2) ∧
  (ages.roy + 2) * (ages.kelly + 2) = 192

/-- The theorem to be proved -/
theorem age_difference_ratio (ages : Ages) :
  satisfiesConditions ages →
  (ages.roy - ages.julia) / (ages.roy - ages.kelly) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_ratio_l4125_412516


namespace NUMINAMATH_CALUDE_triangle_point_trajectory_l4125_412568

theorem triangle_point_trajectory (A B C D : ℝ × ℝ) : 
  B = (-2, 0) →
  C = (2, 0) →
  D = (0, 0) →
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = 3^2 →
  A.2 ≠ 0 →
  A.1^2 + A.2^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_point_trajectory_l4125_412568


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l4125_412591

theorem ceiling_floor_difference : 
  ⌈(20 : ℝ) / 9 * ⌈(-53 : ℝ) / 4⌉⌉ - ⌊(20 : ℝ) / 9 * ⌊(-53 : ℝ) / 4⌋⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l4125_412591


namespace NUMINAMATH_CALUDE_solve_for_m_l4125_412536

theorem solve_for_m (x y m : ℝ) 
  (h1 : x = 3 * m + 1)
  (h2 : y = 2 * m - 2)
  (h3 : 4 * x - 3 * y = 10) : 
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_solve_for_m_l4125_412536


namespace NUMINAMATH_CALUDE_fraction_before_simplification_l4125_412550

theorem fraction_before_simplification
  (n d : ℕ)  -- n and d are the numerator and denominator before simplification
  (h1 : n + d = 80)  -- sum of numerator and denominator is 80
  (h2 : n / d = 3 / 7)  -- fraction simplifies to 3/7
  : n = 24 ∧ d = 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_before_simplification_l4125_412550


namespace NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l4125_412556

theorem businessmen_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 12)
  (h4 : both = 7) :
  total - (coffee + tea - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l4125_412556


namespace NUMINAMATH_CALUDE_waiter_customers_l4125_412559

/-- Given a number of tables and the number of women and men at each table,
    calculate the total number of customers. -/
def total_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) : ℕ :=
  num_tables * (women_per_table + men_per_table)

/-- Theorem: The waiter has 90 customers in total. -/
theorem waiter_customers :
  total_customers 9 7 3 = 90 := by
  sorry

#eval total_customers 9 7 3

end NUMINAMATH_CALUDE_waiter_customers_l4125_412559


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l4125_412592

theorem imaginary_part_of_complex_expression (z : ℂ) (h : z = 3 + 4*I) : 
  Complex.im (z + Complex.abs z / z) = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l4125_412592


namespace NUMINAMATH_CALUDE_triangle_problem_l4125_412598

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →   -- A is acute
  0 < B ∧ B < π/2 →   -- B is acute
  0 < C ∧ C < π/2 →   -- C is acute
  Real.sqrt 3 * c = 2 * a * Real.sin C →  -- √3c = 2a sin C
  a = Real.sqrt 7 →  -- a = √7
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 →  -- Area of triangle ABC
  (A = π/3) ∧ (a + b + c = Real.sqrt 7 + 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l4125_412598


namespace NUMINAMATH_CALUDE_intersecting_lines_y_intercept_sum_l4125_412583

/-- Given two lines that intersect at a specific point, prove their y-intercepts sum to zero -/
theorem intersecting_lines_y_intercept_sum (a b : ℝ) : 
  (3 = (1/3) * (-3) + a) ∧ (-3 = (1/3) * 3 + b) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_y_intercept_sum_l4125_412583


namespace NUMINAMATH_CALUDE_absent_students_sum_l4125_412519

/-- Proves that the sum of absent students over three days equals 200 --/
theorem absent_students_sum (T : ℕ) (A1 A2 A3 : ℕ) : 
  T = 280 →
  A3 = T / 7 →
  A2 = 2 * A3 →
  T - A2 + 40 = T - A1 →
  A1 + A2 + A3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_absent_students_sum_l4125_412519


namespace NUMINAMATH_CALUDE_three_tangent_lines_l4125_412503

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in the 2D plane represented by its equation y^2 = ax -/
structure Parabola where
  a : ℝ

/-- Predicate to check if a line passes through a given point -/
def Line.passesThrough (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Predicate to check if a line has only one common point with a parabola -/
def Line.hasOnlyOneCommonPoint (l : Line) (p : Parabola) : Prop :=
  ∃! x y, l.passesThrough x y ∧ y^2 = p.a * x

/-- The main theorem stating that there are exactly 3 lines passing through (0,6)
    and having only one common point with the parabola y^2 = -12x -/
theorem three_tangent_lines :
  ∃! (lines : Finset Line),
    (∀ l ∈ lines, l.passesThrough 0 6 ∧ l.hasOnlyOneCommonPoint (Parabola.mk (-12))) ∧
    lines.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_tangent_lines_l4125_412503


namespace NUMINAMATH_CALUDE_alloy_mix_solvable_l4125_412599

/-- Represents an alloy of copper and tin -/
structure Alloy where
  mass : ℝ
  copper_percentage : ℝ

/-- Represents the problem of mixing two alloys -/
def AlloyMixProblem (alloy1 alloy2 : Alloy) (target_mass : ℝ) (target_percentage : ℝ) :=
  alloy1.mass ≥ 0 ∧
  alloy2.mass ≥ 0 ∧
  alloy1.copper_percentage ≥ 0 ∧ alloy1.copper_percentage ≤ 100 ∧
  alloy2.copper_percentage ≥ 0 ∧ alloy2.copper_percentage ≤ 100 ∧
  target_mass > 0 ∧
  target_percentage ≥ 0 ∧ target_percentage ≤ 100

theorem alloy_mix_solvable (alloy1 alloy2 : Alloy) (target_mass : ℝ) (p : ℝ) :
  AlloyMixProblem alloy1 alloy2 target_mass p →
  (alloy1.mass = 3 ∧ 
   alloy2.mass = 7 ∧ 
   alloy1.copper_percentage = 40 ∧ 
   alloy2.copper_percentage = 30 ∧
   target_mass = 8) →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ alloy1.mass ∧ 
            0 ≤ target_mass - x ∧ target_mass - x ≤ alloy2.mass ∧
            alloy1.copper_percentage * x / 100 + alloy2.copper_percentage * (target_mass - x) / 100 = target_mass * p / 100) ↔
  (31.25 ≤ p ∧ p ≤ 33.75) :=
by sorry

end NUMINAMATH_CALUDE_alloy_mix_solvable_l4125_412599


namespace NUMINAMATH_CALUDE_three_zeros_condition_l4125_412575

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - a * x
  else -x^2 - (a + 2) * x + 1

/-- The number of zeros of f(x) -/
def number_of_zeros (a : ℝ) : ℕ := sorry

/-- Theorem stating the condition for f(x) to have exactly 3 zeros -/
theorem three_zeros_condition (a : ℝ) :
  number_of_zeros a = 3 ↔ a > Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_condition_l4125_412575


namespace NUMINAMATH_CALUDE_parabola_intersection_ratio_l4125_412534

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def focus : Point := ⟨1, 0⟩

/-- The fixed point A -/
def A : Point := ⟨0, -2⟩

/-- Point M on the parabola -/
def M : Point := sorry

/-- Point N on the directrix -/
def N : Point := sorry

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Theorem: The ratio |MN| : |FN| = √5 : (1 + √5) -/
theorem parabola_intersection_ratio :
  (distance M N) / (distance focus N) = Real.sqrt 5 / (1 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_ratio_l4125_412534


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l4125_412521

-- Define the ellipse
structure Ellipse where
  isTangentToXAxis : Bool
  isTangentToYAxis : Bool
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

-- Define the theorem
theorem ellipse_major_axis_length 
  (e : Ellipse) 
  (h1 : e.isTangentToXAxis = true) 
  (h2 : e.isTangentToYAxis = true)
  (h3 : e.focus1 = (2, -3 + Real.sqrt 13))
  (h4 : e.focus2 = (2, -3 - Real.sqrt 13)) :
  ∃ (majorAxisLength : ℝ), majorAxisLength = 6 :=
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l4125_412521


namespace NUMINAMATH_CALUDE_safe_round_trip_exists_l4125_412564

/-- Represents the cycle of a dragon's fire-breathing pattern -/
structure DragonCycle where
  active : ℕ
  sleep : ℕ

/-- Represents the travel times for the journey -/
structure TravelTimes where
  road : ℕ
  path : ℕ

/-- Checks if a given hour is safe from both dragons -/
def is_safe (h : ℕ) (d1 d2 : DragonCycle) : Prop :=
  h % (d1.active + d1.sleep) > d1.active ∧ 
  h % (d2.active + d2.sleep) > d2.active

/-- Checks if a round trip is possible within a given time frame -/
def round_trip_possible (start : ℕ) (t : TravelTimes) (d1 d2 : DragonCycle) : Prop :=
  ∀ h : ℕ, start ≤ h ∧ h < start + 2 * (t.road + t.path) → is_safe h d1 d2

/-- Main theorem: There exists a safe starting time for the round trip -/
theorem safe_round_trip_exists (t : TravelTimes) (d1 d2 : DragonCycle) : 
  ∃ start : ℕ, round_trip_possible start t d1 d2 :=
sorry

end NUMINAMATH_CALUDE_safe_round_trip_exists_l4125_412564


namespace NUMINAMATH_CALUDE_range_of_a_range_of_a_for_local_minimum_l4125_412572

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2*a) * (x^2 + a^2*x + 2*a^3)

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : 
  (∀ x, x < 0 → (3*x^2 + 2*(a^2 - 2*a)*x < 0)) ↔ (a < 0 ∨ a > 2) :=
sorry

/-- Main theorem proving the range of a -/
theorem range_of_a_for_local_minimum :
  {a : ℝ | IsLocalMin (f a) 0} = {a : ℝ | a < 0 ∨ a > 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_a_for_local_minimum_l4125_412572


namespace NUMINAMATH_CALUDE_minimum_shots_for_high_probability_l4125_412553

theorem minimum_shots_for_high_probability (p : ℝ) (n : ℕ) : 
  p = 1/2 → 
  (1 - (1 - p)^n > 0.9 ↔ n ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_minimum_shots_for_high_probability_l4125_412553


namespace NUMINAMATH_CALUDE_intersected_prisms_count_l4125_412541

def small_prism_dimensions : Fin 3 → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 5

def cube_edge_length : ℕ := 90

def count_intersected_prisms (dimensions : Fin 3 → ℕ) (edge_length : ℕ) : ℕ :=
  sorry

theorem intersected_prisms_count :
  count_intersected_prisms small_prism_dimensions cube_edge_length = 66 := by sorry

end NUMINAMATH_CALUDE_intersected_prisms_count_l4125_412541


namespace NUMINAMATH_CALUDE_inequality_solution_l4125_412574

/-- Given constants p, q, and r satisfying the conditions, prove that p + 2q + 3r = 32 -/
theorem inequality_solution (p q r : ℝ) (h1 : p < q)
  (h2 : ∀ x : ℝ, (x - p) * (x - q) / (x - r) ≥ 0 ↔ x > 5 ∨ (3 ≤ x ∧ x ≤ 7)) :
  p + 2*q + 3*r = 32 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l4125_412574


namespace NUMINAMATH_CALUDE_green_sequins_per_row_l4125_412580

theorem green_sequins_per_row (blue_rows : Nat) (blue_per_row : Nat) 
  (purple_rows : Nat) (purple_per_row : Nat) (green_rows : Nat) (total_sequins : Nat)
  (h1 : blue_rows = 6) (h2 : blue_per_row = 8)
  (h3 : purple_rows = 5) (h4 : purple_per_row = 12)
  (h5 : green_rows = 9) (h6 : total_sequins = 162) :
  (total_sequins - (blue_rows * blue_per_row + purple_rows * purple_per_row)) / green_rows = 6 := by
  sorry

end NUMINAMATH_CALUDE_green_sequins_per_row_l4125_412580


namespace NUMINAMATH_CALUDE_new_clock_conversion_l4125_412513

/-- Represents a time on the new clock -/
structure NewClockTime where
  hours : ℕ
  minutes : ℕ

/-- Represents a time in Beijing -/
structure BeijingTime where
  hours : ℕ
  minutes : ℕ

/-- Converts NewClockTime to total minutes -/
def newClockToMinutes (t : NewClockTime) : ℕ :=
  t.hours * 100 + t.minutes

/-- Converts BeijingTime to total minutes -/
def beijingToMinutes (t : BeijingTime) : ℕ :=
  t.hours * 60 + t.minutes

/-- The theorem to be proved -/
theorem new_clock_conversion (newClock : NewClockTime) (beijing : BeijingTime) :
  (newClockToMinutes ⟨5, 0⟩ = beijingToMinutes ⟨12, 0⟩) →
  (newClockToMinutes ⟨6, 75⟩ = beijingToMinutes ⟨16, 12⟩) := by
  sorry


end NUMINAMATH_CALUDE_new_clock_conversion_l4125_412513


namespace NUMINAMATH_CALUDE_quadratic_vertex_l4125_412547

/-- The quadratic function f(x) = 2(x-3)^2 + 1 has its vertex at (3, 1). -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * (x - 3)^2 + 1
  (∀ x, f x ≥ f 3) ∧ f 3 = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l4125_412547


namespace NUMINAMATH_CALUDE_range_of_x_l4125_412590

theorem range_of_x (x : ℝ) : 
  (∀ m : ℝ, m ≠ 0 → |5*m - 3| + |3 - 4*m| ≥ |m| * (x - 2/x)) →
  x ∈ Set.Ici (-1) ∪ Set.Ioc 0 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l4125_412590


namespace NUMINAMATH_CALUDE_subset_M_l4125_412570

def M : Set ℝ := {x : ℝ | x > -1}

theorem subset_M : {0} ⊆ M := by sorry

end NUMINAMATH_CALUDE_subset_M_l4125_412570


namespace NUMINAMATH_CALUDE_painted_cube_problem_l4125_412515

theorem painted_cube_problem (n : ℕ) : 
  n > 0 → 
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → 
  n = 2 ∧ n^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l4125_412515


namespace NUMINAMATH_CALUDE_factor_expression_l4125_412596

theorem factor_expression (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4125_412596


namespace NUMINAMATH_CALUDE_engine_capacity_proof_l4125_412537

/-- Represents the relationship between diesel volume, distance, and engine capacity -/
structure DieselEngineRelation where
  volume : ℝ  -- Volume of diesel in litres
  distance : ℝ  -- Distance in km
  capacity : ℝ  -- Engine capacity in cc

/-- The relation between diesel volume and engine capacity is directly proportional -/
axiom diesel_capacity_proportion (r1 r2 : DieselEngineRelation) :
  r1.volume / r1.capacity = r2.volume / r2.capacity

/-- Given data for the first scenario -/
def scenario1 : DieselEngineRelation :=
  { volume := 60, distance := 600, capacity := 800 }

/-- Given data for the second scenario -/
def scenario2 : DieselEngineRelation :=
  { volume := 120, distance := 800, capacity := 1600 }

/-- Theorem stating that the engine capacity for the second scenario is 1600 cc -/
theorem engine_capacity_proof :
  scenario2.capacity = 1600 :=
by sorry

end NUMINAMATH_CALUDE_engine_capacity_proof_l4125_412537


namespace NUMINAMATH_CALUDE_triangle_area_is_six_l4125_412501

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ := 
  (1/2) * b * c * Real.sin A

theorem triangle_area_is_six (a b c : ℝ) (A B C : ℝ) 
  (h1 : c = 4)
  (h2 : Real.tan A = 3)
  (h3 : Real.cos C = Real.sqrt 5 / 5) : 
  triangle_area a b c A B C = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_six_l4125_412501


namespace NUMINAMATH_CALUDE_smallest_solution_equation_smallest_solution_is_four_minus_sqrt_two_l4125_412507

theorem smallest_solution_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ↔ (x = 4 - Real.sqrt 2 ∨ x = 4 + Real.sqrt 2) :=
sorry

theorem smallest_solution_is_four_minus_sqrt_two :
  ∃ (x : ℝ), (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  (∀ (y : ℝ), (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x) ∧
  x = 4 - Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_equation_smallest_solution_is_four_minus_sqrt_two_l4125_412507


namespace NUMINAMATH_CALUDE_problem_solution_l4125_412511

theorem problem_solution : 
  (0.064 ^ (-(1/3 : ℝ)) - (-(1/8 : ℝ))^0 + 16^(3/4 : ℝ) + 0.25^(1/2 : ℝ) = 10) ∧
  ((2 * Real.log 2 + Real.log 3) / (1 + (1/2 : ℝ) * Real.log 0.36 + (1/3 : ℝ) * Real.log 8) = 1) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l4125_412511


namespace NUMINAMATH_CALUDE_polynomial_integrality_l4125_412522

theorem polynomial_integrality (x : ℤ) : ∃ k : ℤ, (1/5 : ℚ) * x^5 + (1/3 : ℚ) * x^3 + (7/15 : ℚ) * x = k := by
  sorry

end NUMINAMATH_CALUDE_polynomial_integrality_l4125_412522


namespace NUMINAMATH_CALUDE_k_range_for_specific_inequalities_l4125_412594

/-- Given a real number k, this theorem states that if the system of inequalities
    x^2 - x - 2 > 0 and 2x^2 + (2k+5)x + 5k < 0 has {-2} as its only integer solution,
    then k must be in the range [-3, 2). -/
theorem k_range_for_specific_inequalities (k : ℝ) :
  (∀ x : ℤ, (x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+5)*x + 5*k < 0) ↔ x = -2) →
  -3 ≤ k ∧ k < 2 :=
sorry

end NUMINAMATH_CALUDE_k_range_for_specific_inequalities_l4125_412594


namespace NUMINAMATH_CALUDE_divisible_by_six_l4125_412544

theorem divisible_by_six (n : ℕ) : ∃ k : ℤ, (2 * n^3 + 9 * n^2 + 13 * n : ℤ) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l4125_412544


namespace NUMINAMATH_CALUDE_cube_inequality_l4125_412562

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l4125_412562


namespace NUMINAMATH_CALUDE_additional_bottles_needed_l4125_412518

/-- Represents the number of bottles in a case of water -/
def bottles_per_case : ℕ := 24

/-- Represents the number of cases purchased -/
def cases_purchased : ℕ := 13

/-- Represents the duration of the camp in days -/
def camp_duration : ℕ := 3

/-- Represents the number of children in the first group -/
def group1_children : ℕ := 14

/-- Represents the number of children in the second group -/
def group2_children : ℕ := 16

/-- Represents the number of children in the third group -/
def group3_children : ℕ := 12

/-- Represents the number of bottles consumed by each child per day -/
def bottles_per_child_per_day : ℕ := 3

/-- Calculates the total number of children in the camp -/
def total_children : ℕ :=
  let first_three := group1_children + group2_children + group3_children
  first_three + first_three / 2

/-- Calculates the total number of bottles needed for the entire camp -/
def total_bottles_needed : ℕ :=
  total_children * bottles_per_child_per_day * camp_duration

/-- Calculates the number of bottles already purchased -/
def bottles_purchased : ℕ :=
  cases_purchased * bottles_per_case

/-- Theorem stating that 255 additional bottles are needed -/
theorem additional_bottles_needed : 
  total_bottles_needed - bottles_purchased = 255 := by
  sorry

end NUMINAMATH_CALUDE_additional_bottles_needed_l4125_412518


namespace NUMINAMATH_CALUDE_section_4_eight_times_section_1_l4125_412571

/-- Represents a circular target divided into sections -/
structure CircularTarget where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  α : ℝ
  β : ℝ
  (r₁_pos : 0 < r₁)
  (r₂_pos : 0 < r₂)
  (r₃_pos : 0 < r₃)
  (r₁_lt_r₂ : r₁ < r₂)
  (r₂_lt_r₃ : r₂ < r₃)
  (α_pos : 0 < α)
  (β_pos : 0 < β)
  (section_equality : r₁^2 * β = α * (r₂^2 - r₁^2))
  (section_2_half_3 : β * (r₂^2 - r₁^2) = 2 * r₁^2 * β)

/-- The theorem stating that the area of section 4 is 8 times the area of section 1 -/
theorem section_4_eight_times_section_1 (t : CircularTarget) : 
  (t.β * (t.r₃^2 - t.r₂^2)) / (t.α * t.r₁^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_section_4_eight_times_section_1_l4125_412571


namespace NUMINAMATH_CALUDE_min_rooms_sufficient_l4125_412576

/-- The minimum number of hotel rooms required for 100 tourists given k rooms under renovation -/
def min_rooms (k : ℕ) : ℕ :=
  let m := k / 2
  if k % 2 = 0 then 100 * (m + 1) else 100 * (m + 1) + 1

/-- Theorem stating that min_rooms provides sufficient rooms for 100 tourists -/
theorem min_rooms_sufficient (k : ℕ) :
  ∀ (arrangement : Fin k → Fin (min_rooms k)),
  ∃ (allocation : Fin 100 → Fin (min_rooms k)),
  (∀ i j, i ≠ j → allocation i ≠ allocation j) ∧
  (∀ i, allocation i ∉ Set.range arrangement) :=
sorry

end NUMINAMATH_CALUDE_min_rooms_sufficient_l4125_412576


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l4125_412517

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x - y - 2 * z = 6) :
  x^2 + y^2 + z^2 ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l4125_412517


namespace NUMINAMATH_CALUDE_least_three_digit_11_heavy_l4125_412549

def is_11_heavy (n : ℕ) : Prop := n % 11 > 7

theorem least_three_digit_11_heavy : ∀ n : ℕ, 100 ≤ n ∧ n < 108 → ¬(is_11_heavy n) ∧ is_11_heavy 108 := by
  sorry

#check least_three_digit_11_heavy

end NUMINAMATH_CALUDE_least_three_digit_11_heavy_l4125_412549


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l4125_412586

theorem at_least_one_non_negative (a b c d e f g h : ℝ) :
  (max (a*c + b*d) (max (a*e + b*f) (max (a*g + b*h) (max (c*e + d*f) (max (c*g + d*h) (e*g + f*h)))))) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l4125_412586


namespace NUMINAMATH_CALUDE_k_range_l4125_412502

/-- Piecewise function f(x) -/
noncomputable def f (k a x : ℝ) : ℝ :=
  if x ≥ 0 then k^2 * x + a^2 - k
  else x^2 + (a^2 + 4*a) * x + (2-a)^2

/-- Condition for the existence of a unique nonzero x₂ for any nonzero x₁ -/
def unique_nonzero_solution (k a : ℝ) : Prop :=
  ∀ x₁ : ℝ, x₁ ≠ 0 → ∃! x₂ : ℝ, x₂ ≠ 0 ∧ x₂ ≠ x₁ ∧ f k a x₂ = f k a x₁

theorem k_range (k a : ℝ) :
  unique_nonzero_solution k a → k ∈ Set.Icc (-20) (-4) :=
by sorry

end NUMINAMATH_CALUDE_k_range_l4125_412502


namespace NUMINAMATH_CALUDE_annas_meal_cost_difference_l4125_412588

/-- Represents the cost of Anna's meals -/
def annas_meals (bagel_price cream_cheese_price orange_juice_price orange_juice_discount
                 sandwich_price avocado_price milk_price milk_discount : ℚ) : ℚ :=
  let breakfast_cost := bagel_price + cream_cheese_price + orange_juice_price * (1 - orange_juice_discount)
  let lunch_cost := sandwich_price + avocado_price + milk_price * (1 - milk_discount)
  lunch_cost - breakfast_cost

/-- The difference between Anna's lunch and breakfast costs is $4.14 -/
theorem annas_meal_cost_difference :
  annas_meals 0.95 0.50 1.25 0.32 4.65 0.75 1.15 0.10 = 4.14 := by
  sorry

end NUMINAMATH_CALUDE_annas_meal_cost_difference_l4125_412588


namespace NUMINAMATH_CALUDE_apple_cost_price_l4125_412512

theorem apple_cost_price (selling_price : ℝ) (loss_fraction : ℝ) (cost_price : ℝ) : 
  selling_price = 18 →
  loss_fraction = 1/6 →
  selling_price = cost_price - (loss_fraction * cost_price) →
  cost_price = 21.6 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_price_l4125_412512


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l4125_412584

theorem sum_of_roots_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2004*x + 2021
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 2004) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l4125_412584


namespace NUMINAMATH_CALUDE_x_plus_y_value_l4125_412506

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 2023)
  (h2 : x + 2023 * Real.sin y = 2022)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2022 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l4125_412506


namespace NUMINAMATH_CALUDE_eight_by_ten_grid_theorem_l4125_412546

/-- Represents a rectangular grid -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the number of squares not intersected by diagonals in a grid -/
def squares_not_intersected (g : Grid) : ℕ :=
  sorry

/-- Theorem: In an 8 × 10 grid, 48 squares are not intersected by either diagonal -/
theorem eight_by_ten_grid_theorem : 
  let g : Grid := { rows := 8, cols := 10 }
  squares_not_intersected g = 48 := by
  sorry

end NUMINAMATH_CALUDE_eight_by_ten_grid_theorem_l4125_412546


namespace NUMINAMATH_CALUDE_simplify_expression_l4125_412535

theorem simplify_expression (x : ℝ) : (3*x - 10) + (7*x + 20) - (2*x - 5) = 8*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4125_412535


namespace NUMINAMATH_CALUDE_quadratic_function_coefficients_l4125_412567

/-- Given a quadratic function f(x) = ax^2 + bx + 7, 
    if f(x+1) - f(x) = 8x - 2 for all x, then a = 4 and b = -6 -/
theorem quadratic_function_coefficients 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + 7)
  (h2 : ∀ x, f (x + 1) - f x = 8 * x - 2) : 
  a = 4 ∧ b = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_coefficients_l4125_412567


namespace NUMINAMATH_CALUDE_fundraiser_total_l4125_412520

/-- Calculates the total amount raised from cake sales and donations --/
def total_raised (num_cakes : ℕ) (slices_per_cake : ℕ) (price_per_slice : ℚ) 
                 (donation1_per_slice : ℚ) (donation2_per_slice : ℚ) : ℚ :=
  let total_slices := num_cakes * slices_per_cake
  let sales := total_slices * price_per_slice
  let donation1 := total_slices * donation1_per_slice
  let donation2 := total_slices * donation2_per_slice
  sales + donation1 + donation2

/-- Theorem stating that under given conditions, the total amount raised is $140 --/
theorem fundraiser_total : 
  total_raised 10 8 1 (1/2) (1/4) = 140 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_total_l4125_412520


namespace NUMINAMATH_CALUDE_min_rooms_for_departments_l4125_412554

/-- Given two departments with student counts and room constraints, 
    calculate the minimum number of rooms required. -/
theorem min_rooms_for_departments (dept1_count dept2_count : ℕ) : 
  dept1_count = 72 →
  dept2_count = 5824 →
  ∃ (room_size : ℕ), 
    room_size > 0 ∧
    dept1_count % room_size = 0 ∧
    dept2_count % room_size = 0 ∧
    (dept1_count / room_size + dept2_count / room_size) = 737 := by
  sorry

end NUMINAMATH_CALUDE_min_rooms_for_departments_l4125_412554


namespace NUMINAMATH_CALUDE_four_circles_in_larger_circle_l4125_412543

-- Define a circle with a center point and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being externally tangent
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

-- Define the property of a circle being internally tangent to another circle
def internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

theorem four_circles_in_larger_circle (c1 c2 c3 c4 large : Circle) :
  c1.radius = 2 ∧ c2.radius = 2 ∧ c3.radius = 2 ∧ c4.radius = 2 →
  externally_tangent c1 c2 ∧ externally_tangent c1 c3 ∧ externally_tangent c1 c4 ∧
  externally_tangent c2 c3 ∧ externally_tangent c2 c4 ∧ externally_tangent c3 c4 →
  internally_tangent c1 large ∧ internally_tangent c2 large ∧
  internally_tangent c3 large ∧ internally_tangent c4 large →
  large.radius = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_circles_in_larger_circle_l4125_412543


namespace NUMINAMATH_CALUDE_davids_travel_expenses_l4125_412504

/-- Represents currency amounts in their respective denominations -/
structure Expenses where
  usd : ℝ
  eur : ℝ
  gbp : ℝ
  jpy : ℝ

/-- Represents exchange rates to USD -/
structure ExchangeRates where
  eur_to_usd : ℝ
  gbp_to_usd : ℝ
  jpy_to_usd : ℝ

/-- Calculates the total expenses in USD -/
def total_expenses (e : Expenses) (r : ExchangeRates) : ℝ :=
  e.usd + e.eur * r.eur_to_usd + e.gbp * r.gbp_to_usd + e.jpy * r.jpy_to_usd

/-- Theorem representing David's travel expenses problem -/
theorem davids_travel_expenses 
  (initial_amount : ℝ)
  (expenses : Expenses)
  (initial_rates : ExchangeRates)
  (final_rates : ExchangeRates)
  (loan : ℝ)
  (h1 : initial_amount = 1500)
  (h2 : expenses = { usd := 400, eur := 300, gbp := 150, jpy := 5000 })
  (h3 : initial_rates = { eur_to_usd := 1.10, gbp_to_usd := 1.35, jpy_to_usd := 0.009 })
  (h4 : final_rates = { eur_to_usd := 1.08, gbp_to_usd := 1.32, jpy_to_usd := 0.009 })
  (h5 : loan = 200)
  (h6 : initial_amount - total_expenses expenses initial_rates - loan = 
        total_expenses expenses initial_rates - 500) :
  initial_amount - total_expenses expenses initial_rates + loan = 677.5 := by
  sorry

end NUMINAMATH_CALUDE_davids_travel_expenses_l4125_412504


namespace NUMINAMATH_CALUDE_atop_difference_l4125_412542

-- Define the @ operation
def atop (x y : ℤ) : ℤ := x * y - 3 * x

-- State the theorem
theorem atop_difference : atop 8 5 - atop 5 8 = -9 := by
  sorry

end NUMINAMATH_CALUDE_atop_difference_l4125_412542


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_base6_l4125_412557

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10ToBase6 (n : ℕ) : ℕ := sorry

/-- The sum of an arithmetic series -/
def arithmeticSeriesSum (a : ℕ) (l : ℕ) (n : ℕ) : ℕ :=
  n * (a + l) / 2

theorem arithmetic_series_sum_base6 :
  let first := 1
  let last := base6ToBase10 55
  let terms := base6ToBase10 55
  let sum := arithmeticSeriesSum first last terms
  (sum = 630) ∧ (base10ToBase6 sum = 2530) := by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_base6_l4125_412557


namespace NUMINAMATH_CALUDE_expression_evaluation_l4125_412528

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  (x + 2*y)^2 - (x + y)*(3*x - y) - 5*y^2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4125_412528


namespace NUMINAMATH_CALUDE_compute_expression_l4125_412538

theorem compute_expression : (12 : ℚ) * (1/3 + 1/4 + 1/6)⁻¹ = 16 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l4125_412538


namespace NUMINAMATH_CALUDE_odd_function_property_l4125_412539

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h : OddFunction f) :
  ∀ x : ℝ, f x + f (-x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l4125_412539


namespace NUMINAMATH_CALUDE_perpendicular_vector_l4125_412581

/-- Given three points A, B, C in ℝ³ and a vector a, if a is perpendicular to both AB and AC,
    then a = (1, 1, 1) -/
theorem perpendicular_vector (A B C a : ℝ × ℝ × ℝ) :
  A = (0, 2, 3) →
  B = (-2, 1, 6) →
  C = (1, -1, 5) →
  a.2.2 = 1 →
  (a.1 * (B.1 - A.1) + a.2.1 * (B.2.1 - A.2.1) + a.2.2 * (B.2.2 - A.2.2) = 0) →
  (a.1 * (C.1 - A.1) + a.2.1 * (C.2.1 - A.2.1) + a.2.2 * (C.2.2 - A.2.2) = 0) →
  a = (1, 1, 1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_l4125_412581


namespace NUMINAMATH_CALUDE_problem_2017_l4125_412531

theorem problem_2017 : (2017^2 - 2017 + 1) / 2017 = 2016 + 1 / 2017 := by
  sorry

end NUMINAMATH_CALUDE_problem_2017_l4125_412531


namespace NUMINAMATH_CALUDE_euler_minus_i_pi_l4125_412529

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State Euler's formula
axiom euler_formula (x : ℝ) : cexp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x

-- Theorem to prove
theorem euler_minus_i_pi : cexp (-Complex.I * Real.pi) = -1 := by sorry

end NUMINAMATH_CALUDE_euler_minus_i_pi_l4125_412529


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4125_412505

theorem polynomial_simplification (x : ℝ) : 
  (x - 2)^4 - 4*(x - 2)^3 + 6*(x - 2)^2 - 4*(x - 2) + 1 = (x - 3)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4125_412505


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l4125_412569

theorem parabola_point_coordinates :
  ∀ (x y : ℝ),
  y^2 = 12*x →                           -- Point (x, y) is on the parabola y^2 = 12x
  (x - 3)^2 + y^2 = 9^2 →                -- Point is 9 units away from the focus (3, 0)
  (x = 6 ∧ (y = 6*Real.sqrt 2 ∨ y = -6*Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l4125_412569


namespace NUMINAMATH_CALUDE_angles_with_same_terminal_side_eq_l4125_412585

/-- Given an angle α whose terminal side is the same as 8π/5, 
    this function returns the set of angles in [0, 2π] 
    whose terminal sides are the same as α/4 -/
def anglesWithSameTerminalSide (α : ℝ) : Set ℝ :=
  {x | x ∈ Set.Icc 0 (2 * Real.pi) ∧ 
       ∃ k : ℤ, α = 2 * k * Real.pi + 8 * Real.pi / 5 ∧ 
               x = (k * Real.pi / 2 + 2 * Real.pi / 5) % (2 * Real.pi)}

/-- Theorem stating that the set of angles with the same terminal side as α/4 
    is equal to the specific set of four angles -/
theorem angles_with_same_terminal_side_eq (α : ℝ) 
    (h : ∃ k : ℤ, α = 2 * k * Real.pi + 8 * Real.pi / 5) : 
  anglesWithSameTerminalSide α = {2 * Real.pi / 5, 9 * Real.pi / 10, 7 * Real.pi / 5, 19 * Real.pi / 10} := by
  sorry

end NUMINAMATH_CALUDE_angles_with_same_terminal_side_eq_l4125_412585


namespace NUMINAMATH_CALUDE_rihanna_remaining_money_l4125_412540

/-- Calculates the remaining money after a purchase --/
def remaining_money (initial_amount mango_price juice_price mango_count juice_count : ℕ) : ℕ :=
  initial_amount - (mango_price * mango_count + juice_price * juice_count)

/-- Theorem: Rihanna's remaining money after shopping --/
theorem rihanna_remaining_money :
  remaining_money 50 3 3 6 6 = 14 := by
  sorry

#eval remaining_money 50 3 3 6 6

end NUMINAMATH_CALUDE_rihanna_remaining_money_l4125_412540


namespace NUMINAMATH_CALUDE_fourth_quarter_total_points_l4125_412524

/-- Represents the points scored by a team in each quarter -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℕ)

/-- The game between Raiders and Wildcats -/
structure BasketballGame :=
  (raiders : TeamScores)
  (wildcats : TeamScores)

/-- Conditions of the game -/
def game_conditions (g : BasketballGame) : Prop :=
  let r := g.raiders
  let w := g.wildcats
  -- Game tied at halftime
  r.q1 + r.q2 = w.q1 + w.q2 ∧
  -- Raiders' points form an increasing arithmetic sequence
  ∃ (d : ℕ), r.q2 = r.q1 + d ∧ r.q3 = r.q2 + d ∧ r.q4 = r.q3 + d ∧
  -- Wildcats' points are equal in first two quarters, then decrease by same difference
  ∃ (j : ℕ), w.q1 = w.q2 ∧ w.q3 = w.q2 - j ∧ w.q4 = w.q3 - j ∧
  -- Wildcats won by exactly four points
  (w.q1 + w.q2 + w.q3 + w.q4) = (r.q1 + r.q2 + r.q3 + r.q4) + 4

theorem fourth_quarter_total_points (g : BasketballGame) :
  game_conditions g → g.raiders.q4 + g.wildcats.q4 = 28 :=
by sorry

end NUMINAMATH_CALUDE_fourth_quarter_total_points_l4125_412524


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l4125_412551

theorem min_value_quadratic (x : ℝ) : 
  7 * x^2 - 28 * x + 1702 ≥ 1674 := by
sorry

theorem min_value_quadratic_achieved : 
  ∃ x : ℝ, 7 * x^2 - 28 * x + 1702 = 1674 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l4125_412551


namespace NUMINAMATH_CALUDE_width_to_perimeter_ratio_l4125_412533

/-- The ratio of width to perimeter for a rectangular room -/
theorem width_to_perimeter_ratio (length width : ℝ) (h1 : length = 15) (h2 : width = 13) :
  width / (2 * (length + width)) = 13 / 56 := by
  sorry

end NUMINAMATH_CALUDE_width_to_perimeter_ratio_l4125_412533


namespace NUMINAMATH_CALUDE_number_difference_l4125_412565

theorem number_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : |x - y| = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l4125_412565


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l4125_412589

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 9 = 7 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 7 → m ≤ n ↔ n = 97 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l4125_412589


namespace NUMINAMATH_CALUDE_no_three_numbers_exist_l4125_412500

theorem no_three_numbers_exist : ¬∃ (a b c : ℕ), 
  (a > 1 ∧ b > 1 ∧ c > 1) ∧ 
  ((∃ k : ℕ, a^2 - 1 = b * k ∨ a^2 - 1 = c * k) ∧
   (∃ l : ℕ, b^2 - 1 = a * l ∨ b^2 - 1 = c * l) ∧
   (∃ m : ℕ, c^2 - 1 = a * m ∨ c^2 - 1 = b * m)) :=
by sorry

end NUMINAMATH_CALUDE_no_three_numbers_exist_l4125_412500


namespace NUMINAMATH_CALUDE_sound_speed_in_new_rod_l4125_412561

/-- The speed of sound in a new rod given experimental data -/
theorem sound_speed_in_new_rod (a b l : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : l > 0) (h4 : b > a) : ∃ v : ℝ,
  v = 3 * l / (2 * (b - a)) ∧
  (∃ (t1 t2 t3 t4 : ℝ),
    t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧ t4 > 0 ∧
    t1 + t2 + t3 = a ∧
    t1 = 2 * (t2 + t3) ∧
    t1 + t4 + t3 = b ∧
    t1 + t4 = 2 * t3 ∧
    v = l / t4) :=
by sorry

end NUMINAMATH_CALUDE_sound_speed_in_new_rod_l4125_412561


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l4125_412593

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := -1 + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l4125_412593


namespace NUMINAMATH_CALUDE_equation_solution_l4125_412527

theorem equation_solution (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (h : x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 
       2 * (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))) : 
  x = (3 + Real.sqrt 13) / 2 ∧ y = (3 + Real.sqrt 13) / 2 ∧ z = (3 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4125_412527


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l4125_412509

theorem smallest_n_congruence (k : ℕ) (h : k > 0) :
  (7 ^ k) % 3 = (k ^ 7) % 3 → k ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l4125_412509


namespace NUMINAMATH_CALUDE_tina_brownies_l4125_412558

theorem tina_brownies (total_brownies : ℕ) (days : ℕ) (husband_daily : ℕ) (shared_guests : ℕ) (leftover : ℕ) :
  total_brownies = 24 →
  days = 5 →
  husband_daily = 1 →
  shared_guests = 4 →
  leftover = 5 →
  (total_brownies - (days * husband_daily + shared_guests + leftover)) / (days * 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tina_brownies_l4125_412558


namespace NUMINAMATH_CALUDE_remaining_trip_time_l4125_412563

/-- Proves that the time to complete the second half of a 510 km journey at 85 km/h is 3 hours -/
theorem remaining_trip_time (total_distance : ℝ) (speed : ℝ) (h1 : total_distance = 510) (h2 : speed = 85) :
  (total_distance / 2) / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_trip_time_l4125_412563


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l4125_412526

/-- Given a nonzero constant a for which the equation ax^2 + 16x + 9 = 0 has only one solution,
    prove that this solution is -9/8. -/
theorem unique_quadratic_solution (a : ℝ) (ha : a ≠ 0) 
    (h_unique : ∃! x : ℝ, a * x^2 + 16 * x + 9 = 0) :
  ∃ x : ℝ, a * x^2 + 16 * x + 9 = 0 ∧ x = -9/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l4125_412526


namespace NUMINAMATH_CALUDE_purchase_total_l4125_412560

/-- The total amount spent on a vacuum cleaner and dishwasher after applying a coupon -/
theorem purchase_total (vacuum_cost dishwasher_cost coupon_value : ℕ) : 
  vacuum_cost = 250 → 
  dishwasher_cost = 450 → 
  coupon_value = 75 → 
  vacuum_cost + dishwasher_cost - coupon_value = 625 := by
sorry

end NUMINAMATH_CALUDE_purchase_total_l4125_412560


namespace NUMINAMATH_CALUDE_time_difference_to_halfway_l4125_412532

/-- Time difference for Steve and Danny to reach halfway point -/
theorem time_difference_to_halfway (danny_time : ℝ) (steve_time : ℝ) : 
  danny_time = 31 →
  steve_time = 2 * danny_time →
  steve_time / 2 - danny_time / 2 = 15.5 := by
sorry

end NUMINAMATH_CALUDE_time_difference_to_halfway_l4125_412532
