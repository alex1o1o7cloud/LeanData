import Mathlib

namespace calculation_proof_l2232_223241

theorem calculation_proof : (2 - Real.pi) ^ 0 - 2⁻¹ + Real.cos (60 * π / 180) = 1 := by
  sorry

end calculation_proof_l2232_223241


namespace simple_interest_rate_change_l2232_223244

/-- Given the conditions of a simple interest problem, prove that the new interest rate is 8% -/
theorem simple_interest_rate_change
  (P : ℝ) (R1 T1 SI T2 : ℝ)
  (h1 : R1 = 5)
  (h2 : T1 = 8)
  (h3 : SI = 840)
  (h4 : T2 = 5)
  (h5 : P = (SI * 100) / (R1 * T1))
  (h6 : SI = (P * R1 * T1) / 100)
  (h7 : SI = (P * R2 * T2) / 100)
  : R2 = 8 := by
  sorry

end simple_interest_rate_change_l2232_223244


namespace inverse_square_theorem_l2232_223275

def inverse_square_relation (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ y : ℝ, y ≠ 0 → f y = k / (y * y)

theorem inverse_square_theorem (f : ℝ → ℝ) 
  (h1 : inverse_square_relation f)
  (h2 : ∃ y : ℝ, f y = 1)
  (h3 : f 6 = 0.25) :
  f 3 = 1 := by
sorry

end inverse_square_theorem_l2232_223275


namespace balloon_arrangements_l2232_223277

theorem balloon_arrangements : 
  let total_letters : ℕ := 7
  let repeated_letters : ℕ := 2
  let repetitions_per_letter : ℕ := 2
  (total_letters.factorial) / (repetitions_per_letter.factorial ^ repeated_letters) = 1260 := by
  sorry

end balloon_arrangements_l2232_223277


namespace doctor_lindsay_daily_income_is_2200_l2232_223262

/-- Calculates the total money Doctor Lindsay receives in a typical 8-hour day -/
def doctor_lindsay_daily_income : ℕ := by
  -- Define the number of adult patients per hour
  let adult_patients_per_hour : ℕ := 4
  -- Define the number of child patients per hour
  let child_patients_per_hour : ℕ := 3
  -- Define the cost for an adult's office visit
  let adult_visit_cost : ℕ := 50
  -- Define the cost for a child's office visit
  let child_visit_cost : ℕ := 25
  -- Define the number of working hours per day
  let working_hours_per_day : ℕ := 8
  
  -- Calculate the total income
  exact adult_patients_per_hour * adult_visit_cost * working_hours_per_day + 
        child_patients_per_hour * child_visit_cost * working_hours_per_day

/-- Theorem stating that Doctor Lindsay's daily income is $2200 -/
theorem doctor_lindsay_daily_income_is_2200 : 
  doctor_lindsay_daily_income = 2200 := by
  sorry

end doctor_lindsay_daily_income_is_2200_l2232_223262


namespace triangle_weights_equal_l2232_223293

/-- Given a triangle ABC with side weights x, y, and z, if the sum of weights on any two sides
    equals the weight on the third side multiplied by a constant k, then all weights are equal. -/
theorem triangle_weights_equal (x y z k : ℝ) 
  (h1 : x + y = k * z) 
  (h2 : y + z = k * x) 
  (h3 : z + x = k * y) : 
  x = y ∧ y = z := by
  sorry

end triangle_weights_equal_l2232_223293


namespace problem_solution_l2232_223215

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 3 = y ^ 2) 
  (h3 : x / 5 = 5 * y) : 
  x = 625 / 3 := by
sorry

end problem_solution_l2232_223215


namespace complex_power_sum_l2232_223231

theorem complex_power_sum (i : ℂ) : i^2 = -1 → i^50 + 3 * i^303 - 2 * i^101 = -1 - 5*i := by
  sorry

end complex_power_sum_l2232_223231


namespace books_together_l2232_223234

/-- The number of books Tim and Mike have together -/
def total_books (tim_books mike_books : ℕ) : ℕ := tim_books + mike_books

/-- Theorem stating that Tim and Mike have 42 books together -/
theorem books_together : total_books 22 20 = 42 := by
  sorry

end books_together_l2232_223234


namespace equal_intercept_line_equations_equidistant_point_locus_l2232_223281

/-- A line passing through a point with equal intercepts on both axes -/
structure EqualInterceptLine where
  a : ℝ
  b : ℝ
  passes_through : a + b = 4
  equal_intercepts : a = b

/-- A point equidistant from two parallel lines -/
structure EquidistantPoint where
  x : ℝ
  y : ℝ
  equidistant : |4*x + 6*y - 10| = |4*x + 6*y + 8|

/-- Theorem for the equal intercept line -/
theorem equal_intercept_line_equations (l : EqualInterceptLine) :
  (∀ x y, y = 3*x) ∨ (∀ x y, y = -x + 4) :=
sorry

/-- Theorem for the locus of equidistant points -/
theorem equidistant_point_locus (p : EquidistantPoint) :
  4*p.x + 6*p.y - 9 = 0 :=
sorry

end equal_intercept_line_equations_equidistant_point_locus_l2232_223281


namespace complex_pure_imaginary_condition_l2232_223282

/-- For z = 1 + i and a ∈ ℝ, if (1 - ai) / z is a pure imaginary number, then a = 1 -/
theorem complex_pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := 1 + I
  (((1 : ℂ) - a * I) / z).re = 0 → (((1 : ℂ) - a * I) / z).im ≠ 0 → a = 1 := by
  sorry

end complex_pure_imaginary_condition_l2232_223282


namespace train_length_l2232_223207

/-- The length of a train given its speed and time to cross a post -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 27 ∧ time = 20 → speed * time * (5 / 18) = 150 := by
  sorry

end train_length_l2232_223207


namespace x_intercept_of_perpendicular_line_l2232_223253

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- The x-intercept of a line. -/
def x_intercept (l : Line) : ℚ := -l.y_intercept / l.slope

/-- The slope of a line perpendicular to a given line. -/
def perpendicular_slope (m : ℚ) : ℚ := -1 / m

/-- The slope of the line 4x - 3y = 12. -/
def given_line_slope : ℚ := 4 / 3

theorem x_intercept_of_perpendicular_line :
  let perpendicular_line := Line.mk (perpendicular_slope given_line_slope) 4
  x_intercept perpendicular_line = 16 / 3 := by sorry

end x_intercept_of_perpendicular_line_l2232_223253


namespace sum_of_coefficients_is_one_l2232_223219

/-- A polynomial in two variables that represents a^2005 + b^2005 -/
def P : (ℝ → ℝ → ℝ) → Prop :=
  λ p => ∀ a b : ℝ, p (a + b) (a * b) = a^2005 + b^2005

/-- The sum of coefficients of a polynomial in two variables -/
def sum_of_coefficients (p : ℝ → ℝ → ℝ) : ℝ := p 1 1

theorem sum_of_coefficients_is_one (p : ℝ → ℝ → ℝ) (h : P p) : 
  sum_of_coefficients p = 1 := by
  sorry

#check sum_of_coefficients_is_one

end sum_of_coefficients_is_one_l2232_223219


namespace max_value_k_l2232_223206

theorem max_value_k (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) : 
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧ 
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k) := by
sorry

end max_value_k_l2232_223206


namespace square_equation_solution_l2232_223274

theorem square_equation_solution (x : ℝ) : (x + 3)^2 = 121 ↔ x = 8 ∨ x = -14 := by
  sorry

end square_equation_solution_l2232_223274


namespace pigeon_chicks_count_l2232_223246

/-- Proves that each pigeon has 6 chicks given the problem conditions -/
theorem pigeon_chicks_count :
  ∀ (total_pigeons : ℕ) (adult_pigeons : ℕ) (remaining_pigeons : ℕ),
    adult_pigeons = 40 →
    remaining_pigeons = 196 →
    (remaining_pigeons : ℚ) = 0.7 * total_pigeons →
    (total_pigeons - adult_pigeons) / adult_pigeons = 6 := by
  sorry


end pigeon_chicks_count_l2232_223246


namespace ellipse_m_range_l2232_223257

/-- Given an ellipse represented by the equation x²/(m-1) + y²/(2-m) = 1 with foci on the y-axis,
    the range of values for m is (1, 3/2). -/
theorem ellipse_m_range (x y m : ℝ) :
  (∀ x y, x^2 / (m - 1) + y^2 / (2 - m) = 1) →  -- Ellipse equation
  (∃ c : ℝ, ∀ x, x^2 / (m - 1) + 0^2 / (2 - m) = 1 → x^2 ≤ c^2) →  -- Foci on y-axis
  1 < m ∧ m < 3/2 :=
by sorry

end ellipse_m_range_l2232_223257


namespace sum_of_x_and_y_l2232_223296

theorem sum_of_x_and_y (x y S : ℝ) 
  (h1 : x + y = S) 
  (h2 : y - 3 * x = 7) 
  (h3 : y - x = 7.5) : 
  S = 8 := by
sorry

end sum_of_x_and_y_l2232_223296


namespace simplify_polynomial_l2232_223268

theorem simplify_polynomial (x : ℝ) : 3 * (3 * x^2 + 9 * x - 4) - 2 * (x^2 + 7 * x - 14) = 7 * x^2 + 13 * x + 16 := by
  sorry

end simplify_polynomial_l2232_223268


namespace line_intersection_l2232_223292

theorem line_intersection :
  ∃! p : ℚ × ℚ, 8 * p.1 - 3 * p.2 = 20 ∧ 9 * p.1 + 2 * p.2 = 17 :=
by
  use (91/43, 61/43)
  sorry

end line_intersection_l2232_223292


namespace first_month_sale_correct_l2232_223245

/-- Represents the sales data for a grocery shop -/
structure SalesData where
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ
  average : ℕ

/-- Calculates the sale in the first month given the sales data -/
def calculate_first_month_sale (data : SalesData) : ℕ :=
  data.average * 6 - (data.month2 + data.month3 + data.month4 + data.month5 + data.month6)

/-- Theorem stating that the calculated first month sale is correct -/
theorem first_month_sale_correct (data : SalesData) 
  (h : data = { month2 := 6927, month3 := 6855, month4 := 7230, month5 := 6562, 
                month6 := 5091, average := 6500 }) : 
  calculate_first_month_sale data = 6335 := by
  sorry

#eval calculate_first_month_sale { month2 := 6927, month3 := 6855, month4 := 7230, 
                                   month5 := 6562, month6 := 5091, average := 6500 }

end first_month_sale_correct_l2232_223245


namespace land_increase_percentage_l2232_223237

theorem land_increase_percentage (A B C D E : ℝ) 
  (h1 : B = 1.5 * A)
  (h2 : C = 2 * A)
  (h3 : D = 2.5 * A)
  (h4 : E = 3 * A)
  (h5 : A > 0) :
  let initial_area := A + B + C + D + E
  let increase := 0.1 * A + (1 / 15) * B + 0.05 * C + 0.04 * D + (1 / 30) * E
  increase / initial_area = 0.05 := by
sorry

end land_increase_percentage_l2232_223237


namespace inequality_theorem_l2232_223280

theorem inequality_theorem (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + b = c + d) (prod_gt : a * b > c * d) : 
  (Real.sqrt a + Real.sqrt b > Real.sqrt c + Real.sqrt d) ∧ 
  (|a - b| < |c - d|) := by
sorry

end inequality_theorem_l2232_223280


namespace symmetry_across_x_axis_l2232_223200

def point_symmetrical_to_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetry_across_x_axis :
  let M : ℝ × ℝ := (1, 3)
  let N : ℝ × ℝ := (1, -3)
  point_symmetrical_to_x_axis M N :=
by
  sorry

end symmetry_across_x_axis_l2232_223200


namespace min_value_sum_squared_fractions_l2232_223299

theorem min_value_sum_squared_fractions (x y z : ℕ+) (h : x + y + z = 9) :
  (x^2 + y^2) / (x + y : ℝ) + (x^2 + z^2) / (x + z : ℝ) + (y^2 + z^2) / (y + z : ℝ) ≥ 9 := by
  sorry

end min_value_sum_squared_fractions_l2232_223299


namespace meeting_time_percentage_l2232_223242

def total_work_day : ℕ := 10 -- in hours
def lunch_break : ℕ := 1 -- in hours
def first_meeting : ℕ := 30 -- in minutes
def second_meeting : ℕ := 3 * first_meeting -- in minutes

def actual_work_minutes : ℕ := (total_work_day - lunch_break) * 60
def total_meeting_minutes : ℕ := first_meeting + second_meeting

def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (actual_work_minutes : ℚ) * 100

theorem meeting_time_percentage : 
  ∃ (ε : ℚ), abs (meeting_percentage - 22) < ε ∧ ε > 0 ∧ ε < 1 :=
sorry

end meeting_time_percentage_l2232_223242


namespace fourth_power_sum_l2232_223261

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 3) 
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a^4 + b^4 + c^4 = 37/6 := by
sorry

end fourth_power_sum_l2232_223261


namespace find_divisor_l2232_223247

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 18 + remainder) :
  ∃ (divisor : ℕ), dividend = quotient * divisor + remainder ∧ divisor = 18 := by
  sorry

end find_divisor_l2232_223247


namespace trevors_age_when_brother_is_three_times_older_l2232_223228

theorem trevors_age_when_brother_is_three_times_older (trevor_current_age : ℕ) (brother_current_age : ℕ) :
  trevor_current_age = 11 →
  brother_current_age = 20 →
  ∃ (future_age : ℕ), future_age = 24 ∧ brother_current_age + future_age - trevor_current_age = 3 * trevor_current_age :=
by
  sorry

end trevors_age_when_brother_is_three_times_older_l2232_223228


namespace prime_fraction_solutions_l2232_223238

theorem prime_fraction_solutions (x y : ℕ) :
  (x > 0 ∧ y > 0) →
  (∃ p : ℕ, Nat.Prime p ∧ x * y^2 = p * (x + y)) ↔ 
  ((x = 2 ∧ y = 2) ∨ (x = 6 ∧ y = 2)) := by
sorry

end prime_fraction_solutions_l2232_223238


namespace quadratic_equation_roots_l2232_223213

theorem quadratic_equation_roots (k : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 14 * x + k = 0 ↔ x = (-14 + Real.sqrt 10) / 4 ∨ x = (-14 - Real.sqrt 10) / 4) →
  k = 93 / 4 := by
sorry

end quadratic_equation_roots_l2232_223213


namespace expected_total_rolls_leap_year_l2232_223243

/-- Represents the outcome of rolling an eight-sided die -/
inductive DieRoll
| one | two | three | four | five | six | seven | eight

/-- Defines if a roll is a perfect square (1 or 4) -/
def isPerfectSquare (roll : DieRoll) : Prop :=
  roll = DieRoll.one ∨ roll = DieRoll.four

/-- Calculates the probability of rolling a perfect square -/
def probPerfectSquare : ℚ := 1/4

/-- Calculates the probability of not rolling a perfect square -/
def probNotPerfectSquare : ℚ := 3/4

/-- The number of days in a leap year -/
def daysInLeapYear : ℕ := 366

/-- The expected number of rolls per day -/
noncomputable def expectedRollsPerDay : ℚ := 4/3

/-- Theorem: The expected total number of rolls in a leap year is 488 -/
theorem expected_total_rolls_leap_year :
  (expectedRollsPerDay * daysInLeapYear : ℚ) = 488 := by
  sorry

end expected_total_rolls_leap_year_l2232_223243


namespace cookie_distribution_l2232_223212

theorem cookie_distribution (total : ℚ) (blue green orange red : ℚ) : 
  blue + green + orange + red = total →
  blue + green + orange = 11 / 12 * total →
  red = 1 / 12 * total →
  blue = 1 / 6 * total →
  green = 5 / 12 * total →
  orange = 1 / 3 * total :=
by sorry

end cookie_distribution_l2232_223212


namespace two_white_balls_possible_l2232_223278

/-- Represents the four types of ball replacements --/
inductive Replacement
  | ThreeBlackToOneBlack
  | TwoBlackOneWhiteToOneBlackOneWhite
  | OneBlackTwoWhiteToTwoWhite
  | ThreeWhiteToOneBlackOneWhite

/-- Represents the state of the box --/
structure BoxState :=
  (black : ℕ)
  (white : ℕ)

/-- Applies a single replacement to the box state --/
def applyReplacement (state : BoxState) (r : Replacement) : BoxState :=
  match r with
  | Replacement.ThreeBlackToOneBlack => 
      { black := state.black - 2, white := state.white }
  | Replacement.TwoBlackOneWhiteToOneBlackOneWhite => 
      { black := state.black - 1, white := state.white }
  | Replacement.OneBlackTwoWhiteToTwoWhite => 
      { black := state.black - 1, white := state.white - 1 }
  | Replacement.ThreeWhiteToOneBlackOneWhite => 
      { black := state.black + 1, white := state.white - 2 }

/-- Represents a sequence of replacements --/
def ReplacementSequence := List Replacement

/-- Applies a sequence of replacements to the initial box state --/
def applyReplacements (initial : BoxState) (seq : ReplacementSequence) : BoxState :=
  seq.foldl applyReplacement initial

/-- The theorem to be proved --/
theorem two_white_balls_possible : 
  ∃ (seq : ReplacementSequence), 
    (applyReplacements { black := 100, white := 100 } seq).white = 2 := by
  sorry


end two_white_balls_possible_l2232_223278


namespace resident_price_proof_l2232_223272

/-- Calculates the ticket price for residents given the total attendees, number of residents,
    price for non-residents, and total revenue. -/
def resident_price (total_attendees : ℕ) (num_residents : ℕ) (non_resident_price : ℚ) (total_revenue : ℚ) : ℚ :=
  (total_revenue - (total_attendees - num_residents : ℚ) * non_resident_price) / num_residents

/-- Proves that the resident price is approximately $12.95 given the problem conditions. -/
theorem resident_price_proof :
  let total_attendees : ℕ := 586
  let num_residents : ℕ := 219
  let non_resident_price : ℚ := 17.95
  let total_revenue : ℚ := 9423.70
  abs (resident_price total_attendees num_residents non_resident_price total_revenue - 12.95) < 0.01 := by
  sorry

#eval resident_price 586 219 (17.95 : ℚ) (9423.70 : ℚ)

end resident_price_proof_l2232_223272


namespace cistern_filling_time_l2232_223230

theorem cistern_filling_time (partial_fill_time : ℝ) (partial_fill_fraction : ℝ) 
  (h1 : partial_fill_time = 5)
  (h2 : partial_fill_fraction = 1 / 11) :
  partial_fill_time / partial_fill_fraction = 55 := by
  sorry

end cistern_filling_time_l2232_223230


namespace sum_of_coordinates_after_reflection_l2232_223225

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the y-axis reflection function
def reflect_y (p : Point) : Point :=
  (-p.1, p.2)

-- Define the problem statement
theorem sum_of_coordinates_after_reflection :
  let C : Point := (3, 8)
  let D : Point := reflect_y C
  C.1 + C.2 + D.1 + D.2 = 16 := by
  sorry

end sum_of_coordinates_after_reflection_l2232_223225


namespace intersection_range_l2232_223285

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt ((x + 1)^2 + y^2) = 2

-- Define the line l
def l (k x y : ℝ) : Prop :=
  y = k * x + 1 - 2 * k

-- Theorem statement
theorem intersection_range (k : ℝ) :
  (∃ x y, C x y ∧ l k x y) → k ∈ Set.Icc (1/3 : ℝ) 1 :=
sorry

end intersection_range_l2232_223285


namespace second_expression_proof_l2232_223260

theorem second_expression_proof (a : ℝ) (x : ℝ) :
  a = 34 →
  ((2 * a + 16) + x) / 2 = 89 →
  x = 94 := by
sorry

end second_expression_proof_l2232_223260


namespace quadratic_root_transformation_l2232_223221

/-- Given a quadratic equation px^2 + qx + r = 0 with roots u and v,
    prove that qu + r and qv + r are roots of px^2 - (2pr-q)x + (pr-q^2+qr) = 0 -/
theorem quadratic_root_transformation (p q r u v : ℝ) 
  (hu : p * u^2 + q * u + r = 0)
  (hv : p * v^2 + q * v + r = 0) :
  p * (q * u + r)^2 - (2 * p * r - q) * (q * u + r) + (p * r - q^2 + q * r) = 0 ∧
  p * (q * v + r)^2 - (2 * p * r - q) * (q * v + r) + (p * r - q^2 + q * r) = 0 :=
by sorry

end quadratic_root_transformation_l2232_223221


namespace three_planes_max_parts_l2232_223270

/-- The maximum number of parts that can be created by n planes in 3D space -/
def maxParts (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => maxParts k + k + 1

/-- Theorem: Three planes can divide 3D space into at most 8 parts -/
theorem three_planes_max_parts :
  maxParts 3 = 8 := by
  sorry

end three_planes_max_parts_l2232_223270


namespace min_draws_for_18_l2232_223283

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee at least n of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The actual box contents -/
def boxContents : BallCounts :=
  { red := 30, green := 25, yellow := 22, blue := 15, white := 12, black := 6 }

/-- The main theorem -/
theorem min_draws_for_18 :
  minDraws boxContents 18 = 85 := by
  sorry

end min_draws_for_18_l2232_223283


namespace jake_weight_proof_l2232_223218

/-- Jake's weight in pounds -/
def jake_weight : ℝ := 230

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 111

/-- Jake's brother's weight in pounds -/
def brother_weight : ℝ := 139

theorem jake_weight_proof :
  -- Condition 1: If Jake loses 8 pounds, he will weigh twice as much as his sister
  jake_weight - 8 = 2 * sister_weight ∧
  -- Condition 2: Jake's brother is currently 6 pounds heavier than twice Jake's weight
  brother_weight = 2 * jake_weight + 6 ∧
  -- Condition 3: Together, all three of them now weigh 480 pounds
  jake_weight + sister_weight + brother_weight = 480 ∧
  -- Condition 4: The brother's weight is 125% of the sister's weight
  brother_weight = 1.25 * sister_weight →
  -- Conclusion: Jake's weight is 230 pounds
  jake_weight = 230 := by
  sorry

end jake_weight_proof_l2232_223218


namespace external_tangent_circle_distance_l2232_223279

theorem external_tangent_circle_distance 
  (O P : ℝ × ℝ) 
  (r₁ r₂ : ℝ) 
  (Q : ℝ × ℝ) 
  (T : ℝ × ℝ) 
  (Z : ℝ × ℝ) :
  r₁ = 10 →
  r₂ = 3 →
  dist O P = r₁ + r₂ →
  dist O T = r₁ →
  dist P Z = r₂ →
  (T.1 - O.1) * (Z.1 - T.1) + (T.2 - O.2) * (Z.2 - T.2) = 0 →
  (Z.1 - P.1) * (Z.1 - T.1) + (Z.2 - P.2) * (Z.2 - T.2) = 0 →
  dist O Z = 2 * Real.sqrt 145 :=
by sorry

end external_tangent_circle_distance_l2232_223279


namespace round_23_36_to_nearest_tenth_l2232_223227

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest tenth. -/
def roundToNearestTenth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The given repeating decimal 23.363636... -/
def givenNumber : RepeatingDecimal :=
  { integerPart := 23, repeatingPart := 36 }

theorem round_23_36_to_nearest_tenth :
  roundToNearestTenth givenNumber = 23.4 := by
  sorry

end round_23_36_to_nearest_tenth_l2232_223227


namespace second_box_capacity_l2232_223210

/-- Represents the amount of clay a box can hold based on its dimensions -/
def clay_capacity (height width length : ℝ) : ℝ := sorry

theorem second_box_capacity :
  let first_box_height : ℝ := 2
  let first_box_width : ℝ := 3
  let first_box_length : ℝ := 5
  let first_box_capacity : ℝ := 40
  let second_box_height : ℝ := 2 * first_box_height
  let second_box_width : ℝ := 3 * first_box_width
  let second_box_length : ℝ := first_box_length
  clay_capacity first_box_height first_box_width first_box_length = first_box_capacity →
  clay_capacity second_box_height second_box_width second_box_length = 240 :=
by sorry

end second_box_capacity_l2232_223210


namespace equation_solutions_l2232_223254

def solution_set : Set (ℤ × ℤ) :=
  {(3, 2), (2, 3), (1, -1), (-1, 1), (0, -1), (-1, 0)}

def satisfies_equation (p : ℤ × ℤ) : Prop :=
  (p.1)^3 + (p.2)^3 + 1 = (p.1)^2 * (p.2)^2

theorem equation_solutions :
  ∀ (x y : ℤ), satisfies_equation (x, y) ↔ (x, y) ∈ solution_set := by
sorry

end equation_solutions_l2232_223254


namespace work_completion_time_l2232_223233

theorem work_completion_time (b_time : ℝ) (joint_work_time : ℝ) (work_completed : ℝ) (a_time : ℝ) : 
  b_time = 20 →
  joint_work_time = 2 →
  work_completed = 0.2333333333333334 →
  joint_work_time * ((1 / a_time) + (1 / b_time)) = work_completed →
  a_time = 15 := by
sorry

end work_completion_time_l2232_223233


namespace franks_change_is_four_l2232_223204

/-- The amount of change Frank receives from his purchase. -/
def franks_change (chocolate_bars : ℕ) (chips : ℕ) (chocolate_price : ℚ) (chips_price : ℚ) (money_given : ℚ) : ℚ :=
  money_given - (chocolate_bars * chocolate_price + chips * chips_price)

/-- Theorem stating that Frank's change is $4 given the problem conditions. -/
theorem franks_change_is_four :
  franks_change 5 2 2 3 20 = 4 := by
  sorry

end franks_change_is_four_l2232_223204


namespace geometric_arithmetic_geometric_progression_l2232_223256

theorem geometric_arithmetic_geometric_progression
  (a b c : ℝ) :
  (∃ q : ℝ, b = a * q ∧ c = a * q^2) →  -- Initial geometric progression
  (2 * (b + 2) = a + c) →               -- Arithmetic progression after increasing b by 2
  ((b + 2)^2 = a * (c + 9)) →           -- Geometric progression after increasing c by 9
  ((a = 4/25 ∧ b = -16/25 ∧ c = 64/25) ∨ (a = 4 ∧ b = 8 ∧ c = 16)) :=
by sorry

end geometric_arithmetic_geometric_progression_l2232_223256


namespace range_of_a_l2232_223267

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + (a - 1) * x + 1 = 0 ∧ 
             y^2 + (a - 1) * y + 1 = 0 ∧ 
             0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2

theorem range_of_a :
  ∀ a : ℝ, (proposition_p a ∨ proposition_q a) ∧ 
           ¬(proposition_p a ∧ proposition_q a) →
           (a ∈ Set.Ioc (-2) (-3/2) ∪ Set.Icc (-1) 2) :=
sorry

end range_of_a_l2232_223267


namespace functions_with_inverses_l2232_223271

-- Define the four functions
def function_A : ℝ → ℝ := sorry
def function_B : ℝ → ℝ := sorry
def function_C : ℝ → ℝ := sorry
def function_D : ℝ → ℝ := sorry

-- Define the property of being a straight line through the origin
def is_straight_line_through_origin (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a downward-opening parabola with vertex at (0, 1)
def is_downward_parabola_vertex_0_1 (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being an upper semicircle with radius 3 centered at origin
def is_upper_semicircle_radius_3 (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a piecewise linear function as described
def is_piecewise_linear_as_described (f : ℝ → ℝ) : Prop := sorry

-- Define the property of having an inverse
def has_inverse (f : ℝ → ℝ) : Prop := sorry

theorem functions_with_inverses :
  is_straight_line_through_origin function_A ∧
  is_downward_parabola_vertex_0_1 function_B ∧
  is_upper_semicircle_radius_3 function_C ∧
  is_piecewise_linear_as_described function_D →
  has_inverse function_A ∧
  ¬ has_inverse function_B ∧
  ¬ has_inverse function_C ∧
  has_inverse function_D := by sorry

end functions_with_inverses_l2232_223271


namespace rectangle_measurement_error_l2232_223201

/-- Given a rectangle with sides L and W, prove that if one side is measured 5% in excess
    and the calculated area has an error of 0.8%, the other side must be measured 4% in deficit. -/
theorem rectangle_measurement_error (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let L' := 1.05 * L
  let W' := W * (1 - p)
  let A := L * W
  let A' := L' * W'
  A' = 1.008 * A →
  p = 0.04
  := by sorry

end rectangle_measurement_error_l2232_223201


namespace m_range_proof_l2232_223224

-- Define the conditions
def condition_p (x : ℝ) : Prop := x^2 - 3*x - 4 ≤ 0
def condition_q (x m : ℝ) : Prop := x^2 - 6*x + 9 - m^2 ≤ 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ -4 ∨ m ≥ 4

-- Theorem statement
theorem m_range_proof :
  (∀ x, condition_p x → condition_q x m) ∧
  (∃ x, condition_q x m ∧ ¬condition_p x) →
  m_range m :=
sorry

end m_range_proof_l2232_223224


namespace complex_radical_equation_l2232_223248

theorem complex_radical_equation : 
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 6 + 1) - Real.sqrt (4 - 2 * Real.sqrt 3)
  M = Real.sqrt 3 / 2 + 1 := by
  sorry

end complex_radical_equation_l2232_223248


namespace range_of_a_theorem_l2232_223226

/-- Proposition P: For any real number x, ax^2 + ax + 1 > 0 always holds -/
def P (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + a*x + 1 > 0

/-- Proposition Q: The equation x^2 - x + a = 0 has real roots -/
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

/-- The range of a satisfying the given conditions -/
def range_of_a : Set ℝ := {a : ℝ | a < 0 ∨ (0 < a ∧ a < 4)}

theorem range_of_a_theorem :
  ∀ a : ℝ, (¬(P a ∧ Q a) ∧ (P a ∨ Q a)) ↔ a ∈ range_of_a :=
sorry

end range_of_a_theorem_l2232_223226


namespace total_pens_bought_l2232_223205

theorem total_pens_bought (pen_cost : ℕ) (masha_spent : ℕ) (olya_spent : ℕ) 
  (h1 : pen_cost > 10)
  (h2 : masha_spent = 357)
  (h3 : olya_spent = 441)
  (h4 : masha_spent % pen_cost = 0)
  (h5 : olya_spent % pen_cost = 0) :
  masha_spent / pen_cost + olya_spent / pen_cost = 38 := by
sorry

end total_pens_bought_l2232_223205


namespace f_satisfies_conditions_l2232_223251

/-- A function that represents the relationship between x and y -/
def f (x : ℝ) : ℝ := -2 * x + 6

/-- The proposition that f satisfies the given conditions -/
theorem f_satisfies_conditions :
  (∃ k : ℝ, ∀ x : ℝ, f x = k * (x - 3)) ∧  -- y is directly proportional to x-3
  (f 5 = -4)                               -- When x = 5, y = -4
  := by sorry

end f_satisfies_conditions_l2232_223251


namespace min_value_of_a_l2232_223222

theorem min_value_of_a (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 25) →
  a ≥ 16 :=
by sorry

end min_value_of_a_l2232_223222


namespace prop_p_and_q_implies_range_of_a_l2232_223297

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -2 ∨ a = 1

-- Theorem statement
theorem prop_p_and_q_implies_range_of_a :
  ∀ a : ℝ, p a ∧ q a → range_of_a a :=
by sorry

end prop_p_and_q_implies_range_of_a_l2232_223297


namespace coordinates_of_C_l2232_223208

-- Define the points
def A : ℝ × ℝ := (7, 2)
def B : ℝ × ℝ := (-1, 9)
def D : ℝ × ℝ := (2, 7)

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  -- AB = AC (isosceles triangle)
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
  -- D is on line BC
  (D.1 - B.1) * (C.2 - B.2) = (D.2 - B.2) * (C.1 - B.1) ∧
  -- AD is perpendicular to BC (altitude condition)
  (A.1 - D.1) * (C.1 - B.1) + (A.2 - D.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem coordinates_of_C :
  ∃ (C : ℝ × ℝ), triangle_ABC C ∧ C = (5, 5) := by
  sorry

end coordinates_of_C_l2232_223208


namespace savings_calculation_l2232_223240

/-- Represents the financial situation of a person in a particular month --/
structure FinancialSituation where
  k : ℝ  -- Constant factor
  x : ℝ  -- Variable for income
  y : ℝ  -- Variable for expenditure
  I : ℝ  -- Total income
  E : ℝ  -- Regular expenditure
  U : ℝ  -- Unplanned expense
  S : ℝ  -- Savings

/-- The conditions of the financial situation --/
def financial_conditions (fs : FinancialSituation) : Prop :=
  fs.I = fs.k * fs.x ∧
  fs.E = fs.k * fs.y ∧
  fs.x / fs.y = 5 / 4 ∧
  fs.U = 0.2 * fs.E ∧
  fs.I = 16000 ∧
  fs.S = fs.I - (fs.E + fs.U)

/-- The theorem stating that under the given conditions, the savings is 640 --/
theorem savings_calculation (fs : FinancialSituation) :
  financial_conditions fs → fs.S = 640 := by
  sorry


end savings_calculation_l2232_223240


namespace vector_b_magnitude_l2232_223295

def a : ℝ × ℝ × ℝ := (-1, 2, -3)
def b : ℝ × ℝ × ℝ := (-4, -1, 2)

theorem vector_b_magnitude : ‖b‖ = Real.sqrt 21 := by
  sorry

end vector_b_magnitude_l2232_223295


namespace negation_of_existence_quadratic_inequality_negation_l2232_223290

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem quadratic_inequality_negation : 
  (¬ ∃ x₀ : ℝ, x₀^2 + 1 > 3*x₀) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x) := by sorry

end negation_of_existence_quadratic_inequality_negation_l2232_223290


namespace square_side_length_average_l2232_223284

theorem square_side_length_average (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 144) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end square_side_length_average_l2232_223284


namespace z_in_first_quadrant_l2232_223250

def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def in_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem z_in_first_quadrant (z₁ z₂ : ℂ) 
  (h₁ : complex_to_point z₁ = (2, 3))
  (h₂ : z₂ = -1 + 2*Complex.I) :
  in_first_quadrant (complex_to_point (z₁ - z₂)) := by
  sorry

end z_in_first_quadrant_l2232_223250


namespace same_remainder_divisor_l2232_223286

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧
  N = 23 ∧
  (∀ (k : ℕ), k > N → 
    (1743 % k = 2019 % k ∧ 2019 % k = 3008 % k) → false) ∧
  1743 % N = 2019 % N ∧ 2019 % N = 3008 % N :=
by sorry

end same_remainder_divisor_l2232_223286


namespace train_speed_l2232_223229

/-- 
Given a train with length 150 meters that crosses an electric pole in 3 seconds,
prove that its speed is 50 meters per second.
-/
theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) : 
  length = 150 ∧ 
  time = 3 ∧ 
  speed = length / time → 
  speed = 50 := by sorry

end train_speed_l2232_223229


namespace pauls_initial_pens_l2232_223249

/-- Represents the number of items Paul has --/
structure PaulsItems where
  initialBooks : ℕ
  initialPens : ℕ
  finalBooks : ℕ
  finalPens : ℕ
  soldPens : ℕ

/-- Theorem stating that Paul's initial number of pens is 42 --/
theorem pauls_initial_pens (items : PaulsItems)
    (h1 : items.initialBooks = 143)
    (h2 : items.finalBooks = 113)
    (h3 : items.finalPens = 19)
    (h4 : items.soldPens = 23) :
    items.initialPens = 42 := by
  sorry

#check pauls_initial_pens

end pauls_initial_pens_l2232_223249


namespace complex_modulus_range_l2232_223255

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z = a + Complex.I) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end complex_modulus_range_l2232_223255


namespace no_primes_satisfying_conditions_l2232_223203

theorem no_primes_satisfying_conditions : ¬∃ (p q : ℕ), 
  Prime p ∧ Prime q ∧ p > 3 ∧ q > 3 ∧ 
  (q ∣ (p^2 - 1)) ∧ (p ∣ (q^2 - 1)) := by
sorry

end no_primes_satisfying_conditions_l2232_223203


namespace number_equation_solution_l2232_223276

theorem number_equation_solution :
  ∃ x : ℝ, (3/4 * x + 3^2 = 1/5 * (x - 8 * x^(1/3))) ∧ x = -27 := by
  sorry

end number_equation_solution_l2232_223276


namespace largest_number_divisible_by_sum_of_digits_eight_eight_eight_satisfies_property_l2232_223263

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property we're looking for
def hasSumOfDigitsDivisibility (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

-- State the theorem
theorem largest_number_divisible_by_sum_of_digits :
  ∀ n : ℕ, n < 900 → hasSumOfDigitsDivisibility n → n ≤ 888 :=
by
  sorry

-- Prove that 888 satisfies the property
theorem eight_eight_eight_satisfies_property :
  hasSumOfDigitsDivisibility 888 :=
by
  sorry

end largest_number_divisible_by_sum_of_digits_eight_eight_eight_satisfies_property_l2232_223263


namespace carol_rectangle_length_l2232_223287

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem carol_rectangle_length 
  (jordan : Rectangle) 
  (carol : Rectangle) 
  (h1 : jordan.length = 3) 
  (h2 : jordan.width = 40) 
  (h3 : carol.width = 24) 
  (h4 : area jordan = area carol) : 
  carol.length = 5 := by
sorry

end carol_rectangle_length_l2232_223287


namespace canyon_trail_length_l2232_223291

/-- Represents the hike on Canyon Trail -/
structure CanyonTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hike -/
def validHike (hike : CanyonTrail) : Prop :=
  hike.day1 + hike.day2 + hike.day3 = 36 ∧
  (hike.day2 + hike.day3 + hike.day4) / 3 = 14 ∧
  hike.day3 + hike.day4 + hike.day5 = 45 ∧
  hike.day1 + hike.day4 = 29

/-- The theorem stating the total length of the Canyon Trail -/
theorem canyon_trail_length (hike : CanyonTrail) (h : validHike hike) :
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 71 := by
  sorry

end canyon_trail_length_l2232_223291


namespace final_sum_theorem_l2232_223265

theorem final_sum_theorem (a b S : ℝ) (h : a + b = S) :
  2 * (a + 5) + 2 * (b - 5) = 2 * S := by
  sorry

end final_sum_theorem_l2232_223265


namespace partner_capital_l2232_223294

/-- Given the profit distribution and profit rate change, calculate A's capital -/
theorem partner_capital (total_profit : ℝ) (a_profit_share : ℝ) (a_income_increase : ℝ) 
  (initial_rate : ℝ) (final_rate : ℝ) :
  (a_profit_share = 2/3) →
  (a_income_increase = 300) →
  (initial_rate = 0.05) →
  (final_rate = 0.07) →
  (a_income_increase = a_profit_share * total_profit * (final_rate - initial_rate)) →
  (∃ (a_capital : ℝ), a_capital = 300000 ∧ a_profit_share * total_profit = initial_rate * a_capital) :=
by sorry

end partner_capital_l2232_223294


namespace greatest_common_factor_of_palindromes_l2232_223259

def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ a b : ℕ, n = 100*a + 10*b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3 * k

def set_of_valid_palindromes : Set ℕ :=
  {n : ℕ | is_three_digit_palindrome n ∧ is_multiple_of_three n}

theorem greatest_common_factor_of_palindromes :
  ∃ g : ℕ, g > 0 ∧ 
    (∀ n ∈ set_of_valid_palindromes, g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n ∈ set_of_valid_palindromes, d ∣ n) → d ≤ g) ∧
    g = 3 :=
  sorry

end greatest_common_factor_of_palindromes_l2232_223259


namespace circle_symmetry_line_l2232_223239

/-- Given a circle C: x^2 + y^2 + mx - 4 = 0 and two points on C symmetric 
    with respect to the line x - y + 3 = 0, prove that m = 6 -/
theorem circle_symmetry_line (m : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 + A.2^2 + m*A.1 - 4 = 0) ∧ 
    (B.1^2 + B.2^2 + m*B.1 - 4 = 0) ∧ 
    (A.1 - A.2 + 3 = B.1 - B.2 + 3)) → 
  m = 6 := by
sorry

end circle_symmetry_line_l2232_223239


namespace apples_per_pie_l2232_223202

theorem apples_per_pie 
  (initial_apples : ℕ) 
  (handed_out : ℕ) 
  (num_pies : ℕ) 
  (h1 : initial_apples = 62) 
  (h2 : handed_out = 8) 
  (h3 : num_pies = 6) 
  (h4 : num_pies ≠ 0) : 
  (initial_apples - handed_out) / num_pies = 9 := by
sorry

end apples_per_pie_l2232_223202


namespace sqrt_2x_plus_4_real_l2232_223223

theorem sqrt_2x_plus_4_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x + 4) ↔ x ≥ -2 := by
  sorry

end sqrt_2x_plus_4_real_l2232_223223


namespace intersection_range_l2232_223220

theorem intersection_range (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ - 1 ∧ 
    y₂ = k * x₂ - 1 ∧ 
    x₁^2 - y₁^2 = 4 ∧ 
    x₂^2 - y₂^2 = 4 ∧ 
    x₁ > 0 ∧ 
    x₂ > 0) ↔ 
  (1 < k ∧ k < Real.sqrt 5 / 2) :=
by sorry

end intersection_range_l2232_223220


namespace faster_train_speed_l2232_223289

/-- The speed of the faster train given two trains crossing each other -/
theorem faster_train_speed 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : train_length = 150)
  (h2 : crossing_time = 18)
  (h3 : speed_ratio = 3) : 
  ∃ (v : ℝ), v = 12.5 ∧ v = (2 * train_length) / (crossing_time * (1 + 1 / speed_ratio)) :=
by
  sorry

#check faster_train_speed

end faster_train_speed_l2232_223289


namespace garage_sale_pants_price_l2232_223288

/-- Proves that the price of each pair of pants is $3 in Kekai's garage sale scenario --/
theorem garage_sale_pants_price (shirt_price : ℚ) (num_shirts num_pants : ℕ) (remaining_money : ℚ) :
  shirt_price = 1 →
  num_shirts = 5 →
  num_pants = 5 →
  remaining_money = 10 →
  ∃ (pants_price : ℚ),
    pants_price = 3 ∧
    remaining_money = (shirt_price * num_shirts + pants_price * num_pants) / 2 := by
  sorry


end garage_sale_pants_price_l2232_223288


namespace partition_exists_five_equal_parts_exist_l2232_223266

/-- Represents a geometric shape composed of squares and triangles -/
structure GeometricFigure where
  squares : ℕ
  triangles : ℕ

/-- Represents a partition of a geometric figure -/
structure Partition where
  parts : ℕ
  part_composition : GeometricFigure

/-- Predicate to check if a partition is valid for a given figure -/
def is_valid_partition (figure : GeometricFigure) (partition : Partition) : Prop :=
  figure.squares = partition.parts * partition.part_composition.squares ∧
  figure.triangles = partition.parts * partition.part_composition.triangles

/-- The specific figure from the problem -/
def problem_figure : GeometricFigure :=
  { squares := 10, triangles := 5 }

/-- The desired partition -/
def desired_partition : Partition :=
  { parts := 5, part_composition := { squares := 2, triangles := 1 } }

/-- Theorem stating that the desired partition is valid for the problem figure -/
theorem partition_exists : is_valid_partition problem_figure desired_partition := by
  sorry

/-- Main theorem proving the existence of the required partition -/
theorem five_equal_parts_exist : ∃ (p : Partition), 
  p.parts = 5 ∧ 
  p.part_composition.squares = 2 ∧ 
  p.part_composition.triangles = 1 ∧
  is_valid_partition problem_figure p := by
  sorry

end partition_exists_five_equal_parts_exist_l2232_223266


namespace squared_difference_of_quadratic_roots_l2232_223298

theorem squared_difference_of_quadratic_roots : ∀ Φ φ : ℝ, 
  Φ ≠ φ →
  Φ^2 = 2*Φ + 1 →
  φ^2 = 2*φ + 1 →
  (Φ - φ)^2 = 8 := by
sorry

end squared_difference_of_quadratic_roots_l2232_223298


namespace union_of_A_and_B_l2232_223236

def A : Set Nat := {1, 2, 3, 5}
def B : Set Nat := {2, 3, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5, 6} := by
  sorry

end union_of_A_and_B_l2232_223236


namespace rogers_money_l2232_223269

theorem rogers_money (initial amount_spent final : ℤ) 
  (h1 : initial = 45)
  (h2 : amount_spent = 20)
  (h3 : final = 71) :
  final - (initial - amount_spent) = 46 := by
  sorry

end rogers_money_l2232_223269


namespace sqrt_seven_minus_one_half_less_than_one_l2232_223209

theorem sqrt_seven_minus_one_half_less_than_one :
  (Real.sqrt 7 - 1) / 2 < 1 := by sorry

end sqrt_seven_minus_one_half_less_than_one_l2232_223209


namespace cake_ingredient_difference_l2232_223214

/-- Given a cake recipe and partially added ingredients, calculate the difference
    between remaining flour and required sugar. -/
theorem cake_ingredient_difference
  (total_sugar : ℕ)
  (total_flour : ℕ)
  (added_flour : ℕ)
  (h1 : total_sugar = 6)
  (h2 : total_flour = 9)
  (h3 : added_flour = 2)
  : total_flour - added_flour - total_sugar = 1 := by
  sorry

end cake_ingredient_difference_l2232_223214


namespace train_speed_calculation_l2232_223232

/-- Proves that the speed of the first train is approximately 120.016 kmph given the conditions -/
theorem train_speed_calculation (length1 length2 speed2 time : ℝ) 
  (h1 : length1 = 290) 
  (h2 : length2 = 210.04)
  (h3 : speed2 = 80)
  (h4 : time = 9)
  : ∃ speed1 : ℝ, abs (speed1 - 120.016) < 0.001 := by
  sorry

end train_speed_calculation_l2232_223232


namespace joe_watching_schedule_l2232_223217

/-- The number of episodes Joe needs to watch per day to catch up with the season premiere. -/
def episodes_per_day (days_until_premiere : ℕ) (num_seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  (num_seasons * episodes_per_season) / days_until_premiere

/-- Theorem stating that Joe needs to watch 6 episodes per day. -/
theorem joe_watching_schedule :
  episodes_per_day 10 4 15 = 6 := by
  sorry

end joe_watching_schedule_l2232_223217


namespace largest_divisor_n4_minus_n_l2232_223258

/-- A positive integer n is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The largest integer that always divides n^4 - n for all composite n is 6 -/
theorem largest_divisor_n4_minus_n (n : ℕ) (h : IsComposite n) :
  (∀ k : ℕ, k > 6 → ∃ m : ℕ, IsComposite m ∧ (m^4 - m) % k ≠ 0) ∧
  (∀ n : ℕ, IsComposite n → (n^4 - n) % 6 = 0) :=
sorry

end largest_divisor_n4_minus_n_l2232_223258


namespace geometric_progression_unique_p_l2232_223264

theorem geometric_progression_unique_p : 
  ∃! (p : ℝ), p > 0 ∧ (2 * Real.sqrt p) ^ 2 = (p - 2) * (-3 - p) := by
  sorry

end geometric_progression_unique_p_l2232_223264


namespace monotonic_quadratic_function_l2232_223252

/-- The function f is monotonic on the interval [1, 2] if and only if a is in the specified range -/
theorem monotonic_quadratic_function (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => x^2 + (2*a + 1)*x + 1)) ↔ 
  a ∈ Set.Iic (-3/2) ∪ Set.Ioi (-5/2) :=
sorry

end monotonic_quadratic_function_l2232_223252


namespace absolute_difference_mn_l2232_223235

theorem absolute_difference_mn (m n : ℝ) 
  (h1 : m * n = 6)
  (h2 : m + n = 7)
  (h3 : m^2 - n^2 = 13) : 
  |m - n| = 13/7 := by sorry

end absolute_difference_mn_l2232_223235


namespace solve_linear_equation_l2232_223216

theorem solve_linear_equation (x : ℝ) : 5 * x + 3 = 10 * x - 17 → x = 4 := by
  sorry

end solve_linear_equation_l2232_223216


namespace parallel_condition_necessary_not_sufficient_l2232_223273

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define parallelism for two lines -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1 ≠ l2

/-- The first line: 2x + ay - 1 = 0 -/
def line1 (a : ℝ) : Line2D :=
  { a := 2, b := a, c := -1 }

/-- The second line: bx + 2y - 2 = 0 -/
def line2 (b : ℝ) : Line2D :=
  { a := b, b := 2, c := -2 }

/-- The main theorem -/
theorem parallel_condition_necessary_not_sufficient :
  (∀ a b : ℝ, parallel (line1 a) (line2 b) → a * b = 4) ∧
  ¬(∀ a b : ℝ, a * b = 4 → parallel (line1 a) (line2 b)) := by
  sorry

end parallel_condition_necessary_not_sufficient_l2232_223273


namespace xy_product_l2232_223211

theorem xy_product (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 21 := by
  sorry

end xy_product_l2232_223211
