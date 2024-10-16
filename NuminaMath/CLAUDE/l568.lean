import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l568_56817

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {-1, -3, a}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (Set.compl A ∩ B a).Nonempty → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l568_56817


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l568_56864

/-- Given a nonzero constant a for which the equation ax^2 + 16x + 9 = 0 has only one solution,
    prove that this solution is -9/8. -/
theorem unique_quadratic_solution (a : ℝ) (ha : a ≠ 0) 
    (h_unique : ∃! x : ℝ, a * x^2 + 16 * x + 9 = 0) :
  ∃ x : ℝ, a * x^2 + 16 * x + 9 = 0 ∧ x = -9/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l568_56864


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l568_56818

/-- A circle with radius 5, center on the x-axis, and tangent to the line x=3 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_x_axis : (center.2 = 0)
  radius_is_5 : radius = 5
  tangent_to_x3 : |center.1 - 3| = 5

/-- The equation of the circle is (x-8)^2 + y^2 = 25 or (x+2)^2 + y^2 = 25 -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y : ℝ, (x - 8)^2 + y^2 = 25 ∨ (x + 2)^2 + y^2 = 25 ↔ 
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l568_56818


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_l568_56832

theorem smallest_four_digit_multiple : ∃ (n : ℕ), 
  (n = 1119) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < n → ¬(((m + 1) % 5 = 0) ∧ ((m + 1) % 7 = 0) ∧ ((m + 1) % 8 = 0))) ∧
  ((n + 1) % 5 = 0) ∧ ((n + 1) % 7 = 0) ∧ ((n + 1) % 8 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_l568_56832


namespace NUMINAMATH_CALUDE_dhoni_leftover_percentage_l568_56812

/-- Represents Dhoni's spending and savings as percentages of his monthly earnings -/
structure DhoniFinances where
  rent_percent : ℝ
  dishwasher_percent : ℝ
  leftover_percent : ℝ

/-- Calculates Dhoni's finances based on given conditions -/
def calculate_finances (rent_percent : ℝ) : DhoniFinances :=
  let dishwasher_percent := rent_percent - (0.1 * rent_percent)
  let spent_percent := rent_percent + dishwasher_percent
  let leftover_percent := 100 - spent_percent
  { rent_percent := rent_percent,
    dishwasher_percent := dishwasher_percent,
    leftover_percent := leftover_percent }

/-- Theorem stating that Dhoni has 52.5% of his earnings left over -/
theorem dhoni_leftover_percentage :
  (calculate_finances 25).leftover_percent = 52.5 := by sorry

end NUMINAMATH_CALUDE_dhoni_leftover_percentage_l568_56812


namespace NUMINAMATH_CALUDE_sum_of_two_with_prime_bound_l568_56853

theorem sum_of_two_with_prime_bound (n : ℕ) (h : n ≥ 50) :
  ∃ x y : ℕ, n = x + y ∧
    ∀ p : ℕ, p.Prime → (p ∣ x ∨ p ∣ y) → (n : ℝ).sqrt ≥ p :=
  sorry

end NUMINAMATH_CALUDE_sum_of_two_with_prime_bound_l568_56853


namespace NUMINAMATH_CALUDE_past_five_weeks_income_sum_l568_56822

/-- Represents the weekly income of a salesman -/
structure WeeklyIncome where
  base : ℕ
  commission : ℕ

/-- Calculates the total income for a given number of weeks -/
def totalIncome (income : WeeklyIncome) (weeks : ℕ) : ℕ :=
  (income.base + income.commission) * weeks

/-- Represents the salesman's income data -/
structure SalesmanIncome where
  baseSalary : ℕ
  pastWeeks : ℕ
  futureWeeks : ℕ
  avgCommissionFuture : ℕ
  avgTotalIncome : ℕ

/-- Theorem: The sum of weekly incomes for the past 5 weeks is $2070 -/
theorem past_five_weeks_income_sum 
  (s : SalesmanIncome) 
  (h1 : s.baseSalary = 400)
  (h2 : s.pastWeeks = 5)
  (h3 : s.futureWeeks = 2)
  (h4 : s.avgCommissionFuture = 315)
  (h5 : s.avgTotalIncome = 500)
  (h6 : s.pastWeeks + s.futureWeeks = 7) :
  totalIncome ⟨s.baseSalary, 0⟩ s.pastWeeks + 
  (s.avgTotalIncome * (s.pastWeeks + s.futureWeeks) - 
   totalIncome ⟨s.baseSalary, s.avgCommissionFuture⟩ s.futureWeeks) = 2070 :=
by
  sorry


end NUMINAMATH_CALUDE_past_five_weeks_income_sum_l568_56822


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l568_56898

/-- A geometric sequence with common ratio q < 0 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q < 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (a 2 = 1 - a 1) →
  (a 4 = 4 - a 3) →
  a 5 + a 6 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l568_56898


namespace NUMINAMATH_CALUDE_investment_income_is_575_l568_56872

/-- Calculates the simple interest for a given principal, rate, and time. -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Represents the total annual income from two investments with simple interest. -/
def totalAnnualIncome (investment1 : ℝ) (rate1 : ℝ) (investment2 : ℝ) (rate2 : ℝ) : ℝ :=
  simpleInterest investment1 rate1 1 + simpleInterest investment2 rate2 1

/-- Theorem stating that the total annual income from the given investments is $575. -/
theorem investment_income_is_575 :
  totalAnnualIncome 3000 0.085 5000 0.064 = 575 := by
  sorry

end NUMINAMATH_CALUDE_investment_income_is_575_l568_56872


namespace NUMINAMATH_CALUDE_at_op_four_neg_one_l568_56826

/-- Definition of the @ operation -/
def at_op (x y : ℤ) : ℤ := x * (y + 2) + 2 * x * y

/-- Theorem stating that 4 @ (-1) = -4 -/
theorem at_op_four_neg_one : at_op 4 (-1) = -4 := by sorry

end NUMINAMATH_CALUDE_at_op_four_neg_one_l568_56826


namespace NUMINAMATH_CALUDE_twoPointThreeFive_equals_fraction_l568_56899

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + (d.repeatingPart : ℚ) / (99 : ℚ)

/-- The repeating decimal 2.35̄ -/
def twoPointThreeFive : RepeatingDecimal :=
  { integerPart := 2, repeatingPart := 35 }

theorem twoPointThreeFive_equals_fraction :
  toRational twoPointThreeFive = 233 / 99 := by
  sorry

end NUMINAMATH_CALUDE_twoPointThreeFive_equals_fraction_l568_56899


namespace NUMINAMATH_CALUDE_two_mono_triangles_probability_l568_56820

/-- A complete graph K6 with edges colored either green or yellow -/
structure ColoredK6 where
  edges : Fin 15 → Bool  -- True for green, False for yellow

/-- The probability of an edge being green -/
def p_green : ℚ := 2/3

/-- The probability of an edge being yellow -/
def p_yellow : ℚ := 1/3

/-- The probability of a specific triangle being monochromatic -/
def p_mono_triangle : ℚ := 1/3

/-- The total number of triangles in K6 -/
def total_triangles : ℕ := 20

/-- The probability of exactly two monochromatic triangles in a ColoredK6 -/
def prob_two_mono_triangles : ℚ := 49807360/3486784401

theorem two_mono_triangles_probability (g : ColoredK6) : 
  prob_two_mono_triangles = (total_triangles.choose 2 : ℚ) * p_mono_triangle^2 * (1 - p_mono_triangle)^(total_triangles - 2) :=
sorry

end NUMINAMATH_CALUDE_two_mono_triangles_probability_l568_56820


namespace NUMINAMATH_CALUDE_parallel_angles_theorem_l568_56879

/-- Two angles in space with parallel sides --/
structure ParallelAngles where
  α : Real
  β : Real
  sides_parallel : Bool

/-- The theorem stating that if two angles have parallel sides and one is 30°, the other is either 30° or 150° --/
theorem parallel_angles_theorem (angles : ParallelAngles) 
  (h1 : angles.sides_parallel = true) 
  (h2 : angles.α = 30) : 
  angles.β = 30 ∨ angles.β = 150 := by
  sorry

end NUMINAMATH_CALUDE_parallel_angles_theorem_l568_56879


namespace NUMINAMATH_CALUDE_large_rectangle_area_l568_56857

def small_rectangle_perimeter : ℕ := 20

def large_rectangle_side_difference : ℕ := 2

def valid_areas : Set ℕ := {3300, 4000, 4500, 4800, 4900}

theorem large_rectangle_area (l w : ℕ) :
  (l + w = small_rectangle_perimeter / 2) →
  (l > 0 ∧ w > 0) →
  ((l + large_rectangle_side_difference) * (w + large_rectangle_side_difference) * 100) ∈ valid_areas :=
by sorry

end NUMINAMATH_CALUDE_large_rectangle_area_l568_56857


namespace NUMINAMATH_CALUDE_freshman_percentage_l568_56892

-- Define the total number of students
variable (T : ℝ)
-- Define the fraction of freshmen (to be proven)
variable (F : ℝ)

-- Conditions from the problem
axiom liberal_arts : F * T * 0.5 = T * 0.1 / 0.5

-- Theorem to prove
theorem freshman_percentage : F = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_freshman_percentage_l568_56892


namespace NUMINAMATH_CALUDE_inequality_proof_l568_56874

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y)) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l568_56874


namespace NUMINAMATH_CALUDE_polynomial_independence_l568_56897

theorem polynomial_independence (x m : ℝ) : 
  (∀ m, 6 * x^2 + (1 - 2*m) * x + 7*m = 6 * x^2 + x) → x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_independence_l568_56897


namespace NUMINAMATH_CALUDE_log_z_w_value_l568_56886

theorem log_z_w_value (x y z w : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1) (hw : w > 0)
  (hlogx : Real.log w / Real.log x = 24)
  (hlogy : Real.log w / Real.log y = 40)
  (hlogxyz : Real.log w / Real.log (x * y * z) = 12) :
  Real.log w / Real.log z = 60 := by
  sorry

end NUMINAMATH_CALUDE_log_z_w_value_l568_56886


namespace NUMINAMATH_CALUDE_bug_meeting_point_l568_56801

/-- Represents a triangle with given side lengths -/
structure Triangle where
  pq : ℝ
  qr : ℝ
  pr : ℝ

/-- Represents a bug moving along the perimeter of a triangle -/
structure Bug where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Calculates the meeting point of two bugs on a triangle's perimeter -/
def meetingPoint (t : Triangle) (b1 b2 : Bug) : ℝ :=
  sorry

theorem bug_meeting_point (t : Triangle) (b1 b2 : Bug) :
  t.pq = 8 ∧ t.qr = 10 ∧ t.pr = 12 ∧
  b1.speed = 2 ∧ b2.speed = 3 ∧
  b1.direction ≠ b2.direction →
  meetingPoint t b1 b2 = 3 :=
sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l568_56801


namespace NUMINAMATH_CALUDE_sunny_candles_proof_l568_56871

/-- Calculates the total number of candles used by Sunny --/
def total_candles (initial_cakes : ℕ) (given_away : ℕ) (candles_per_cake : ℕ) : ℕ :=
  (initial_cakes - given_away) * candles_per_cake

/-- Proves that Sunny will use 36 candles in total --/
theorem sunny_candles_proof :
  total_candles 8 2 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sunny_candles_proof_l568_56871


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l568_56843

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The tens digit of a natural number. -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- The units digit of a natural number. -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of digits of a two-digit number. -/
def sumOfDigits (n : ℕ) : ℕ := tensDigit n + unitsDigit n

/-- The main theorem stating that 24 is the unique two-digit number satisfying the given conditions. -/
theorem unique_two_digit_number : 
  ∃! n : ℕ, TwoDigitNumber n ∧ 
            tensDigit n = unitsDigit n / 2 ∧ 
            n - sumOfDigits n = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l568_56843


namespace NUMINAMATH_CALUDE_train_passing_time_train_passing_man_time_l568_56815

/-- The time it takes for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_ms := relative_speed * (1000 / 3600)
  train_length / relative_speed_ms

/-- Proof that the time for a 110m train moving at 40 km/h to pass a man moving at 4 km/h in the opposite direction is approximately 8.99 seconds -/
theorem train_passing_man_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_passing_time 110 40 4 - 8.99| < ε :=
sorry

end NUMINAMATH_CALUDE_train_passing_time_train_passing_man_time_l568_56815


namespace NUMINAMATH_CALUDE_max_value_function_l568_56827

theorem max_value_function (x : ℝ) (h : x < 5/4) :
  4*x - 2 + 1/(4*x - 5) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_function_l568_56827


namespace NUMINAMATH_CALUDE_parcel_cost_formula_l568_56825

/-- The cost function for sending a parcel post package -/
def parcel_cost (P : ℕ) : ℕ :=
  20 + 5 * (P - 1)

theorem parcel_cost_formula (P : ℕ) (h : P ≥ 2) :
  parcel_cost P = 20 + 5 * (P - 1) :=
by sorry

end NUMINAMATH_CALUDE_parcel_cost_formula_l568_56825


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_493_l568_56861

theorem smallest_next_divisor_after_493 (n : ℕ) : 
  1000 ≤ n ∧ n < 10000 ∧  -- n is a 4-digit number
  Even n ∧                -- n is even
  n % 493 = 0 →           -- 493 is a divisor of n
  (∃ (d : ℕ), d > 493 ∧ n % d = 0 ∧ d ≤ 510 ∧ 
    ∀ (k : ℕ), 493 < k ∧ k < d → n % k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_493_l568_56861


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l568_56834

/-- Given a two-digit number where the difference between its digits is 9,
    prove that the difference between the original number and the number
    with interchanged digits is always 81. -/
theorem two_digit_number_difference (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 9 →
  (10 * x + y) - (10 * y + x) = 81 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l568_56834


namespace NUMINAMATH_CALUDE_like_terms_exponents_l568_56809

theorem like_terms_exponents (m n : ℤ) : 
  (∀ x y : ℝ, ∃ k : ℝ, -3 * x^(m-1) * y^3 = k * (5/2 * x^n * y^(m+n))) → 
  m = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l568_56809


namespace NUMINAMATH_CALUDE_triangle_side_length_l568_56877

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = π/3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l568_56877


namespace NUMINAMATH_CALUDE_ellipse_condition_l568_56842

-- Define the equation
def equation (x y z m : ℝ) : Prop :=
  3 * x^2 + 9 * y^2 - 12 * x + 18 * y + 6 * z = m

-- Define what it means for the equation to represent a non-degenerate ellipse when projected onto the xy-plane
def is_nondegenerate_ellipse_projection (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), 
    ∃ (z : ℝ), equation x y z m ↔ (x - c)^2 / a + (y - c)^2 / b = 1

-- State the theorem
theorem ellipse_condition (m : ℝ) : 
  is_nondegenerate_ellipse_projection m ↔ m > -21 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l568_56842


namespace NUMINAMATH_CALUDE_marcus_rachel_percentage_l568_56866

def marcus_score : ℕ := 5 * 3 + 10 * 2 + 8 * 1 + 2 * 4
def brian_score : ℕ := 6 * 3 + 8 * 2 + 9 * 1 + 1 * 4
def rachel_score : ℕ := 4 * 3 + 12 * 2 + 7 * 1 + 0 * 4
def team_total_score : ℕ := 150

theorem marcus_rachel_percentage :
  (marcus_score + rachel_score : ℚ) / team_total_score * 100 = 62.67 := by
  sorry

end NUMINAMATH_CALUDE_marcus_rachel_percentage_l568_56866


namespace NUMINAMATH_CALUDE_count_integer_lengths_specific_triangle_l568_56878

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ
  df : ℕ
  is_right : de^2 + ef^2 = df^2

/-- Counts the number of distinct integer lengths of line segments from E to DF -/
def count_integer_lengths (t : RightTriangle) : ℕ :=
  let max_length := max t.de t.ef
  let min_length := min t.de t.ef
  max_length - min_length + 1

/-- The main theorem -/
theorem count_integer_lengths_specific_triangle :
  ∃ (t : RightTriangle), t.de = 12 ∧ t.ef = 16 ∧ count_integer_lengths t = 5 :=
sorry

end NUMINAMATH_CALUDE_count_integer_lengths_specific_triangle_l568_56878


namespace NUMINAMATH_CALUDE_f_composition_of_three_l568_56851

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem f_composition_of_three : f (f (f (f 3))) = 24 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l568_56851


namespace NUMINAMATH_CALUDE_negation_of_p_l568_56880

-- Define the proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

-- State the theorem
theorem negation_of_p (f : ℝ → ℝ) : ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
  sorry

end NUMINAMATH_CALUDE_negation_of_p_l568_56880


namespace NUMINAMATH_CALUDE_marble_remainder_l568_56869

theorem marble_remainder (r p : ℕ) 
  (h_ringo : r % 6 = 4) 
  (h_paul : p % 6 = 3) : 
  (r + p) % 6 = 1 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l568_56869


namespace NUMINAMATH_CALUDE_coordinates_of_P_wrt_origin_l568_56884

-- Define a point in 2D Cartesian coordinate system
def Point := ℝ × ℝ

-- Define point P
def P : Point := (-5, 3)

-- Theorem stating that the coordinates of P with respect to the origin are (-5, 3)
theorem coordinates_of_P_wrt_origin :
  P = (-5, 3) := by sorry

end NUMINAMATH_CALUDE_coordinates_of_P_wrt_origin_l568_56884


namespace NUMINAMATH_CALUDE_car_trip_speed_proof_l568_56800

/-- Proves that given a trip of 8 hours with an average speed of 34 miles per hour,
    where the first 6 hours are traveled at 30 miles per hour,
    the average speed for the remaining 2 hours is 46 miles per hour. -/
theorem car_trip_speed_proof :
  let total_time : ℝ := 8
  let first_part_time : ℝ := 6
  let first_part_speed : ℝ := 30
  let total_average_speed : ℝ := 34
  let remaining_time : ℝ := total_time - first_part_time
  let total_distance : ℝ := total_time * total_average_speed
  let first_part_distance : ℝ := first_part_time * first_part_speed
  let remaining_distance : ℝ := total_distance - first_part_distance
  let remaining_speed : ℝ := remaining_distance / remaining_time
  remaining_speed = 46 := by sorry

end NUMINAMATH_CALUDE_car_trip_speed_proof_l568_56800


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l568_56882

/-- Given a function f: ℝ → ℝ satisfying the conditions:
    1) f(x+5) = 4x^3 + 5x^2 + 9x + 6
    2) f(x) = ax^3 + bx^2 + cx + d
    Prove that a + b + c + d = -206 -/
theorem sum_of_coefficients (f : ℝ → ℝ) (a b c d : ℝ) :
  (∀ x, f (x + 5) = 4 * x^3 + 5 * x^2 + 9 * x + 6) →
  (∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  a + b + c + d = -206 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l568_56882


namespace NUMINAMATH_CALUDE_seahorse_penguin_ratio_l568_56889

theorem seahorse_penguin_ratio :
  let seahorses : ℕ := 70
  let penguins : ℕ := seahorses + 85
  (seahorses : ℚ) / penguins = 14 / 31 := by
  sorry

end NUMINAMATH_CALUDE_seahorse_penguin_ratio_l568_56889


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l568_56876

variable (k : ℕ)

def first_term : ℕ → ℕ := λ k => 3 * k^2 + 2
def common_difference : ℕ := 2
def num_terms : ℕ → ℕ := λ k => 4 * k + 3

theorem arithmetic_series_sum :
  (λ k : ℕ => (num_terms k) * (2 * first_term k + (num_terms k - 1) * common_difference) / 2) =
  (λ k : ℕ => 12 * k^3 + 28 * k^2 + 28 * k + 12) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l568_56876


namespace NUMINAMATH_CALUDE_cucumbers_for_24_apples_l568_56821

/-- The cost of a single apple -/
def apple_cost : ℝ := 1

/-- The cost of a single banana -/
def banana_cost : ℝ := 2

/-- The cost of a single cucumber -/
def cucumber_cost : ℝ := 1.5

/-- 12 apples cost the same as 6 bananas -/
axiom apple_banana_relation : 12 * apple_cost = 6 * banana_cost

/-- 3 bananas cost the same as 4 cucumbers -/
axiom banana_cucumber_relation : 3 * banana_cost = 4 * cucumber_cost

/-- The number of cucumbers that can be bought for the price of 24 apples is 16 -/
theorem cucumbers_for_24_apples : 
  (24 * apple_cost) / cucumber_cost = 16 := by sorry

end NUMINAMATH_CALUDE_cucumbers_for_24_apples_l568_56821


namespace NUMINAMATH_CALUDE_expression_equals_53_l568_56839

theorem expression_equals_53 : (-6)^4 / 6^2 + 2^5 - 6^1 - 3^2 = 53 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_53_l568_56839


namespace NUMINAMATH_CALUDE_annulus_area_l568_56855

theorem annulus_area (r₁ r₂ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 2) :
  π * r₂^2 - π * r₁^2 = 3 * π := by sorry

end NUMINAMATH_CALUDE_annulus_area_l568_56855


namespace NUMINAMATH_CALUDE_max_distance_complex_l568_56836

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (max_dist : ℝ), max_dist = 8 * (Real.sqrt 29 + 2) ∧
  ∀ (w : ℂ), Complex.abs w = 2 →
    Complex.abs ((5 + 2*I)*w^3 - w^4) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l568_56836


namespace NUMINAMATH_CALUDE_combination_sum_l568_56860

theorem combination_sum (n : ℕ) : 
  (5 : ℚ) / 2 ≤ n ∧ n ≤ 3 → Nat.choose (2*n) (10 - 2*n) + Nat.choose (3 + n) (2*n) = 16 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_l568_56860


namespace NUMINAMATH_CALUDE_time_difference_to_halfway_l568_56850

/-- Time difference for Steve and Danny to reach halfway point -/
theorem time_difference_to_halfway (danny_time : ℝ) (steve_time : ℝ) : 
  danny_time = 31 →
  steve_time = 2 * danny_time →
  steve_time / 2 - danny_time / 2 = 15.5 := by
sorry

end NUMINAMATH_CALUDE_time_difference_to_halfway_l568_56850


namespace NUMINAMATH_CALUDE_point_condition_l568_56862

-- Define the unit circle ω in the xy-plane
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the condition for right angles
def right_angle (A B C P : Point3D) : Prop :=
  (A.x - P.x) * (B.x - P.x) + (A.y - P.y) * (B.y - P.y) + (A.z - P.z) * (B.z - P.z) = 0

-- Main theorem
theorem point_condition (P : Point3D) (h_not_xy : P.z ≠ 0) :
  (∃ A B C : Point3D, 
    unit_circle A.x A.y ∧ 
    unit_circle B.x B.y ∧ 
    unit_circle C.x C.y ∧ 
    right_angle A B P C ∧ 
    right_angle A C P B ∧ 
    right_angle B C P A) →
  P.x^2 + P.y^2 + 2*P.z^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_condition_l568_56862


namespace NUMINAMATH_CALUDE_fraction_division_l568_56895

theorem fraction_division (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y) / (1 / x) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l568_56895


namespace NUMINAMATH_CALUDE_davids_travel_expenses_l568_56891

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

end NUMINAMATH_CALUDE_davids_travel_expenses_l568_56891


namespace NUMINAMATH_CALUDE_harry_galleons_l568_56808

theorem harry_galleons (H He R : ℕ) : 
  (H + He = 12) →
  (H + R = 120) →
  (∃ k : ℕ, H + He + R = 7 * k) →
  (H + He + R ≥ H) →
  (H + He + R ≥ He) →
  (H + He + R ≥ R) →
  (H > 0) →
  (H = 6) := by
sorry

end NUMINAMATH_CALUDE_harry_galleons_l568_56808


namespace NUMINAMATH_CALUDE_correct_quotient_l568_56856

theorem correct_quotient (N : ℕ) : 
  (N / 7 = 12 ∧ N % 7 = 5) → N / 8 = 11 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_l568_56856


namespace NUMINAMATH_CALUDE_distance_AB_l568_56804

/-- The distance between points A(2,1) and B(5,-1) is √13. -/
theorem distance_AB : Real.sqrt 13 = Real.sqrt ((5 - 2)^2 + (-1 - 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_l568_56804


namespace NUMINAMATH_CALUDE_favorite_numbers_exist_l568_56844

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem favorite_numbers_exist : ∃ (a b c : ℕ), 
  a * b * c = 71668 ∧ 
  a * sum_of_digits a = 10 * a ∧ 
  b * sum_of_digits b = 10 * b ∧ 
  c * sum_of_digits c = 10 * c ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end NUMINAMATH_CALUDE_favorite_numbers_exist_l568_56844


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l568_56854

/-- Given two quadratic equations with coefficients a, b, c, d where the roots of
    a²x² + bx + c = 0 are 2011 times the roots of cx² + dx + a = 0,
    prove that b² = d² -/
theorem quadratic_root_relation (a b c d : ℝ) 
  (h : ∀ (x₁ x₂ : ℝ), c * x₁^2 + d * x₁ + a = 0 ∧ c * x₂^2 + d * x₂ + a = 0 → 
       a^2 * (2011 * x₁)^2 + b * (2011 * x₁) + c = 0 ∧ 
       a^2 * (2011 * x₂)^2 + b * (2011 * x₂) + c = 0) : 
  b^2 = d^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l568_56854


namespace NUMINAMATH_CALUDE_minimal_kamber_group_common_meal_l568_56847

/-- The number of citizens in the city -/
def num_citizens : ℕ := 2017

/-- The number of meal types available -/
def num_meals : ℕ := 25

/-- A citizen is represented by a natural number -/
def Citizen := Fin num_citizens

/-- A meal is represented by a natural number -/
def Meal := Fin num_meals

/-- Predicate indicating whether a citizen likes a meal -/
def likes (c : Citizen) (m : Meal) : Prop := sorry

/-- A set of citizens is a suitable list if each meal is liked by at least one person in the set -/
def is_suitable_list (s : Set Citizen) : Prop :=
  ∀ m : Meal, ∃ c ∈ s, likes c m

/-- A set of citizens is a kamber group if it contains at least one person from each suitable list -/
def is_kamber_group (k : Set Citizen) : Prop :=
  ∀ s : Set Citizen, is_suitable_list s → (∃ c ∈ k, c ∈ s)

/-- A kamber group is minimal if no proper subset is also a kamber group -/
def is_minimal_kamber_group (k : Set Citizen) : Prop :=
  is_kamber_group k ∧ ∀ k' ⊂ k, ¬is_kamber_group k'

theorem minimal_kamber_group_common_meal (k : Set Citizen) 
  (h : is_minimal_kamber_group k) : 
  ∃ m : Meal, ∀ c ∈ k, likes c m := by sorry


end NUMINAMATH_CALUDE_minimal_kamber_group_common_meal_l568_56847


namespace NUMINAMATH_CALUDE_inverse_g_inverse_g_14_l568_56810

def g (x : ℝ) : ℝ := 5 * x - 3

theorem inverse_g_inverse_g_14 : 
  (Function.invFun g) ((Function.invFun g) 14) = 32 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_inverse_g_14_l568_56810


namespace NUMINAMATH_CALUDE_arithmetic_sequence_298_l568_56890

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem arithmetic_sequence_298 :
  ∃ n : ℕ, arithmetic_sequence 1 3 n = 298 ∧ n = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_298_l568_56890


namespace NUMINAMATH_CALUDE_x_plus_y_value_l568_56806

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 2023)
  (h2 : x + 2023 * Real.sin y = 2022)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2022 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l568_56806


namespace NUMINAMATH_CALUDE_book_ratio_is_one_fifth_l568_56837

/-- The ratio of Queen's extra books to Alannah's books -/
def book_ratio (beatrix alannah queen total : ℕ) : ℚ :=
  let queen_extra := queen - alannah
  ↑queen_extra / ↑alannah

theorem book_ratio_is_one_fifth 
  (beatrix alannah queen total : ℕ) 
  (h1 : beatrix = 30)
  (h2 : alannah = beatrix + 20)
  (h3 : total = beatrix + alannah + queen)
  (h4 : total = 140) :
  book_ratio beatrix alannah queen total = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_book_ratio_is_one_fifth_l568_56837


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l568_56803

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 1) (h2 : x * y = 2 * x + y + 2) :
  x + y ≥ 7 ∧ ∃ x0 y0, x0 > 1 ∧ x0 * y0 = 2 * x0 + y0 + 2 ∧ x0 + y0 = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l568_56803


namespace NUMINAMATH_CALUDE_train_route_length_l568_56868

/-- Given two trains traveling towards each other on a route, where Train X takes 4 hours
    to complete the trip, Train Y takes 3 hours to complete the trip, and Train X has
    traveled 60 km when they meet, prove that the total length of the route is 140 km. -/
theorem train_route_length (x_time y_time x_distance : ℝ) 
    (hx : x_time = 4)
    (hy : y_time = 3)
    (hd : x_distance = 60) : 
  x_distance * (1 / x_time + 1 / y_time) = 140 := by
  sorry

#check train_route_length

end NUMINAMATH_CALUDE_train_route_length_l568_56868


namespace NUMINAMATH_CALUDE_sin_120_degrees_l568_56883

theorem sin_120_degrees : Real.sin (120 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l568_56883


namespace NUMINAMATH_CALUDE_value_of_expression_l568_56814

theorem value_of_expression (x y : ℝ) (hx : x = 12) (hy : y = 7) :
  (x - y) * (x + y) = 95 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l568_56814


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l568_56887

/-- The trajectory of the midpoint M of a line segment PP', where P is on a circle
    with center (0,0) and radius 2, and P' is the projection of P on the x-axis. -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x₀ y₀ : ℝ, 
    x₀^2 + y₀^2 = 4 ∧   -- P is on the circle
    x = x₀ ∧            -- M's x-coordinate is same as P's
    2 * y = y₀) →       -- M's y-coordinate is half of P's
  x^2 / 4 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l568_56887


namespace NUMINAMATH_CALUDE_distance_between_points_with_given_distances_from_origin_l568_56811

def distance_between_points (a b : ℝ) : ℝ := |a - b|

theorem distance_between_points_with_given_distances_from_origin :
  ∀ (a b : ℝ),
  distance_between_points 0 a = 2 →
  distance_between_points 0 b = 7 →
  distance_between_points a b = 5 ∨ distance_between_points a b = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_with_given_distances_from_origin_l568_56811


namespace NUMINAMATH_CALUDE_fair_tickets_sold_l568_56846

theorem fair_tickets_sold (total : ℕ) (second_week : ℕ) (left_to_sell : ℕ) 
  (h1 : total = 90)
  (h2 : second_week = 17)
  (h3 : left_to_sell = 35) :
  total - second_week - left_to_sell = 38 := by
sorry

end NUMINAMATH_CALUDE_fair_tickets_sold_l568_56846


namespace NUMINAMATH_CALUDE_sin_cos_identity_l568_56867

theorem sin_cos_identity : Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
                           Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l568_56867


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l568_56819

theorem polynomial_division_quotient : ∀ x : ℝ,
  (9 * x^3 - 5 * x^2 + 8 * x - 12) = (x - 3) * (9 * x^2 + 22 * x + 74) + 210 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l568_56819


namespace NUMINAMATH_CALUDE_expression_value_l568_56859

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 1)    -- absolute value of m is 1
  : m + (2024 * (a + b)) / 2023 - (c * d)^2 = 0 ∨ 
    m + (2024 * (a + b)) / 2023 - (c * d)^2 = -2 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l568_56859


namespace NUMINAMATH_CALUDE_min_gumballs_for_five_same_color_l568_56813

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat

/-- The minimum number of gumballs needed to guarantee at least 5 of the same color -/
def minGumballsForFiveSameColor (machine : GumballMachine) : Nat :=
  13

/-- Theorem stating that for the given gumball machine configuration, 
    the minimum number of gumballs needed is 13 -/
theorem min_gumballs_for_five_same_color 
  (machine : GumballMachine) 
  (h1 : machine.red = 12) 
  (h2 : machine.white = 10) 
  (h3 : machine.blue = 11) : 
  minGumballsForFiveSameColor machine = 13 := by
  sorry

#check min_gumballs_for_five_same_color

end NUMINAMATH_CALUDE_min_gumballs_for_five_same_color_l568_56813


namespace NUMINAMATH_CALUDE_amusement_park_revenue_l568_56863

def ticket_price : ℕ := 3
def weekday_visitors : ℕ := 100
def saturday_visitors : ℕ := 200
def sunday_visitors : ℕ := 300
def days_in_week : ℕ := 7
def weekdays : ℕ := 5

def total_revenue : ℕ := ticket_price * (weekday_visitors * weekdays + saturday_visitors + sunday_visitors)

theorem amusement_park_revenue : total_revenue = 3000 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_revenue_l568_56863


namespace NUMINAMATH_CALUDE_implication_equivalence_l568_56848

theorem implication_equivalence (R S : Prop) :
  (R → S) ↔ (¬S → ¬R) := by sorry

end NUMINAMATH_CALUDE_implication_equivalence_l568_56848


namespace NUMINAMATH_CALUDE_simplify_fraction_l568_56828

theorem simplify_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(3/2) * b^(5/2)) / (a*b)^(1/2) = a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l568_56828


namespace NUMINAMATH_CALUDE_equation_solution_l568_56865

theorem equation_solution (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (h : x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 
       2 * (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))) : 
  x = (3 + Real.sqrt 13) / 2 ∧ y = (3 + Real.sqrt 13) / 2 ∧ z = (3 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l568_56865


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l568_56823

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = c ∨ x = d) : 
  c = 1 ∧ d = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l568_56823


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l568_56894

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 2*x > 0} = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l568_56894


namespace NUMINAMATH_CALUDE_pet_supply_store_dog_food_l568_56852

/-- Given a pet supply store with cat food and dog food, prove the number of bags of dog food. -/
theorem pet_supply_store_dog_food (cat_food : ℕ) (difference : ℕ) : 
  cat_food = 327 → difference = 273 → cat_food + difference = 600 := by
  sorry

end NUMINAMATH_CALUDE_pet_supply_store_dog_food_l568_56852


namespace NUMINAMATH_CALUDE_cylinder_volume_l568_56858

/-- The volume of a cylinder with base radius 1 cm and generatrix length 2 cm is 2π cm³ -/
theorem cylinder_volume (π : ℝ) : ℝ := by
  sorry

#check cylinder_volume

end NUMINAMATH_CALUDE_cylinder_volume_l568_56858


namespace NUMINAMATH_CALUDE_problem_2017_l568_56849

theorem problem_2017 : (2017^2 - 2017 + 1) / 2017 = 2016 + 1 / 2017 := by
  sorry

end NUMINAMATH_CALUDE_problem_2017_l568_56849


namespace NUMINAMATH_CALUDE_age_sum_in_two_years_l568_56829

theorem age_sum_in_two_years :
  let fem_current_age : ℕ := 11
  let matt_current_age : ℕ := 4 * fem_current_age
  let jake_current_age : ℕ := matt_current_age + 5
  let fem_future_age : ℕ := fem_current_age + 2
  let matt_future_age : ℕ := matt_current_age + 2
  let jake_future_age : ℕ := jake_current_age + 2
  fem_future_age + matt_future_age + jake_future_age = 110
  := by sorry

end NUMINAMATH_CALUDE_age_sum_in_two_years_l568_56829


namespace NUMINAMATH_CALUDE_absolute_value_equality_l568_56833

theorem absolute_value_equality (y : ℝ) : |y| = |y - 3| → y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l568_56833


namespace NUMINAMATH_CALUDE_three_digit_primes_exist_l568_56802

theorem three_digit_primes_exist : 
  ∃ (S : Finset Nat), 
    (1 ≤ S.card ∧ S.card ≤ 10) ∧ 
    (∀ p ∈ S, 100 ≤ p ∧ p ≤ 999 ∧ Nat.Prime p) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_primes_exist_l568_56802


namespace NUMINAMATH_CALUDE_intersection_equals_nonnegative_reals_l568_56893

-- Define set A
def A : Set ℝ := {x : ℝ | |x| = x}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 + x ≥ 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_nonnegative_reals :
  A_intersect_B = {x : ℝ | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_nonnegative_reals_l568_56893


namespace NUMINAMATH_CALUDE_customer_difference_l568_56830

theorem customer_difference (initial : ℕ) (remained : ℕ) : 
  initial = 11 → remained = 3 → (initial - remained) - remained = 5 := by
sorry

end NUMINAMATH_CALUDE_customer_difference_l568_56830


namespace NUMINAMATH_CALUDE_focus_of_our_parabola_l568_56896

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  /-- The equation of the parabola in the form y = ax² + bx + c -/
  equation : ℝ → ℝ → Prop

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
def focus (p : Parabola) : Point :=
  sorry

/-- The parabola defined by x² = y -/
def our_parabola : Parabola :=
  { equation := fun x y ↦ x^2 = y }

/-- Theorem stating that the focus of our parabola is at (0, 1) -/
theorem focus_of_our_parabola :
  focus our_parabola = Point.mk 0 1 := by
  sorry

end NUMINAMATH_CALUDE_focus_of_our_parabola_l568_56896


namespace NUMINAMATH_CALUDE_vasya_mistake_l568_56841

-- Define the function to calculate the number of digits used for page numbering
def digits_used (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + 2 * (n - 9)
  else 189 + 3 * (n - 99)

-- Theorem statement
theorem vasya_mistake :
  ¬ ∃ (n : ℕ), digits_used n = 301 :=
sorry

end NUMINAMATH_CALUDE_vasya_mistake_l568_56841


namespace NUMINAMATH_CALUDE_floor_of_4_7_l568_56807

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l568_56807


namespace NUMINAMATH_CALUDE_choose_captains_l568_56835

theorem choose_captains (n k : ℕ) (hn : n = 15) (hk : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_captains_l568_56835


namespace NUMINAMATH_CALUDE_intersection_A_B_l568_56805

def A : Set ℝ := {x | x * (x - 3) < 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l568_56805


namespace NUMINAMATH_CALUDE_min_value_expression_l568_56831

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + y) / x + 1 / y ≥ 2 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l568_56831


namespace NUMINAMATH_CALUDE_equation_system_solution_l568_56873

theorem equation_system_solution :
  ∀ (x y a : ℝ),
  (2 * x + y = a) →
  (x + y = 3) →
  (x = 2) →
  (a = 5 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l568_56873


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l568_56875

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- The side length of the hexagon
  side : ℝ
  -- Assertion that four nonadjacent interior angles are 45°
  has_four_45_angles : Bool
  -- The area of the hexagon
  area : ℝ
  -- The area is 12√2
  area_is_12_root_2 : area = 12 * Real.sqrt 2

/-- The perimeter of a hexagon is 6 times its side length -/
def perimeter (h : SpecialHexagon) : ℝ := 6 * h.side

/-- Theorem stating the perimeter of the special hexagon is 6√6 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : 
  perimeter h = 6 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l568_56875


namespace NUMINAMATH_CALUDE_line_circle_intersection_l568_56840

/-- A line passing through (-2,0) with slope k intersects the circle x^2 + y^2 = 2x at two points
    if and only if -√2/4 < k < √2/4 -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    (k * x₁ - y₁ + 2*k = 0) ∧ 
    (k * x₂ - y₂ + 2*k = 0) ∧ 
    (x₁^2 + y₁^2 = 2*x₁) ∧ 
    (x₂^2 + y₂^2 = 2*x₂)) ↔ 
  (-Real.sqrt 2 / 4 < k ∧ k < Real.sqrt 2 / 4) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l568_56840


namespace NUMINAMATH_CALUDE_circle_a_l568_56845

theorem circle_a (x y : ℝ) : 
  (x - 3)^2 + (y + 2)^2 = 16 → (∃ (center : ℝ × ℝ) (radius : ℝ), center = (3, -2) ∧ radius = 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_a_l568_56845


namespace NUMINAMATH_CALUDE_unique_rectangle_from_rods_l568_56838

theorem unique_rectangle_from_rods (n : ℕ) (h : n = 22) : 
  (∃! (l w : ℕ), l + w = n / 2 ∧ l * 2 + w * 2 = n ∧ l > 0 ∧ w > 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_rectangle_from_rods_l568_56838


namespace NUMINAMATH_CALUDE_seventh_fibonacci_is_eight_l568_56870

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem seventh_fibonacci_is_eight :
  fibonacci 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_fibonacci_is_eight_l568_56870


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l568_56816

theorem smallest_x_absolute_value_equation :
  ∃ (x : ℝ), x = -4 ∧ |2*x - 6| = 14 ∧ ∀ (y : ℝ), |2*y - 6| = 14 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l568_56816


namespace NUMINAMATH_CALUDE_track_circumference_l568_56888

/-- Represents the circumference of the circular track -/
def circumference : ℝ := 720

/-- Represents the distance B has traveled at the first meeting -/
def first_meeting_distance : ℝ := 150

/-- Represents the distance A has left to complete one lap at the second meeting -/
def second_meeting_remaining : ℝ := 90

/-- Represents the number of laps A has completed at the third meeting -/
def third_meeting_laps : ℝ := 1.5

theorem track_circumference :
  (first_meeting_distance + (circumference - first_meeting_distance) = circumference) ∧
  (circumference - second_meeting_remaining + (circumference / 2 + second_meeting_remaining) = circumference) ∧
  (third_meeting_laps * circumference + (circumference + first_meeting_distance) = 2 * circumference) :=
by sorry

end NUMINAMATH_CALUDE_track_circumference_l568_56888


namespace NUMINAMATH_CALUDE_bouquet_calculation_l568_56824

theorem bouquet_calculation (total_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : 
  total_flowers = 53 → 
  flowers_per_bouquet = 7 → 
  wilted_flowers = 18 → 
  (total_flowers - wilted_flowers) / flowers_per_bouquet = 5 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_calculation_l568_56824


namespace NUMINAMATH_CALUDE_candy_distribution_convergence_l568_56885

/-- Represents the state of candy distribution among students -/
structure CandyState where
  numStudents : Nat
  candies : Fin numStudents → Nat

/-- Represents one round of candy distribution -/
def distributeCandy (state : CandyState) : CandyState :=
  sorry

/-- The teacher gives one candy to students with an odd number of candies -/
def teacherIntervention (state : CandyState) : CandyState :=
  sorry

/-- Checks if all students have the same number of candies -/
def allEqual (state : CandyState) : Bool :=
  sorry

/-- Main theorem: After a finite number of rounds, all students will have the same number of candies -/
theorem candy_distribution_convergence
  (initialState : CandyState)
  (h_even_initial : ∀ i, Even (initialState.candies i)) :
  ∃ n : Nat, allEqual (((teacherIntervention ∘ distributeCandy)^[n]) initialState) = true :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_convergence_l568_56885


namespace NUMINAMATH_CALUDE_max_product_sum_constant_l568_56881

theorem max_product_sum_constant (a b M : ℝ) : 
  a > 0 → b > 0 → a + b = M → (∀ x y : ℝ, x > 0 → y > 0 → x + y = M → x * y ≤ 2) → M = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_constant_l568_56881
