import Mathlib

namespace NUMINAMATH_CALUDE_f_has_one_zero_in_interval_l1773_177310

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 7

-- State the theorem
theorem f_has_one_zero_in_interval :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_one_zero_in_interval_l1773_177310


namespace NUMINAMATH_CALUDE_smallest_angle_of_dividable_isosceles_triangle_l1773_177303

-- Define an isosceles triangle
structure IsoscelesTriangle where
  α : ℝ
  -- The base angles are equal (α) and the sum of all angles is 180°
  angleSum : α + α + (180 - 2*α) = 180

-- Define a function that checks if a triangle can be divided into two isosceles triangles
def canDivideIntoTwoIsosceles (t : IsoscelesTriangle) : Prop :=
  -- This is a placeholder for the actual condition
  -- In reality, this would involve a more complex geometric condition
  true

-- Theorem statement
theorem smallest_angle_of_dividable_isosceles_triangle :
  ∀ t : IsoscelesTriangle, 
    canDivideIntoTwoIsosceles t → 
    (min t.α (180 - 2*t.α) ≥ 180 / 7) ∧ 
    (∃ t' : IsoscelesTriangle, canDivideIntoTwoIsosceles t' ∧ min t'.α (180 - 2*t'.α) = 180 / 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_of_dividable_isosceles_triangle_l1773_177303


namespace NUMINAMATH_CALUDE_probability_between_R_and_S_l1773_177325

/-- Given a line segment PQ with points R and S, where PQ = 4PR and PQ = 8QR,
    the probability that a randomly selected point on PQ lies between R and S is 5/8. -/
theorem probability_between_R_and_S (P Q R S : Real) (h1 : Q - P = 4 * (R - P)) (h2 : Q - P = 8 * (Q - R)) :
  (S - R) / (Q - P) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_between_R_and_S_l1773_177325


namespace NUMINAMATH_CALUDE_f_one_equals_four_l1773_177385

/-- The function f(x) = x^2 + ax - 3a - 9 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 3*a - 9

/-- The theorem stating that f(1) = 4 given the conditions -/
theorem f_one_equals_four (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : f a 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_one_equals_four_l1773_177385


namespace NUMINAMATH_CALUDE_line_no_dot_count_l1773_177365

/-- Represents the number of letters in the alphabet -/
def total_letters : ℕ := 40

/-- Represents the number of letters containing both a dot and a straight line -/
def dot_and_line : ℕ := 13

/-- Represents the number of letters containing a dot but not a straight line -/
def dot_no_line : ℕ := 3

/-- Theorem stating that the number of letters containing a straight line but not a dot is 24 -/
theorem line_no_dot_count : 
  total_letters - (dot_and_line + dot_no_line) = 24 := by sorry

end NUMINAMATH_CALUDE_line_no_dot_count_l1773_177365


namespace NUMINAMATH_CALUDE_number_satisfying_equation_l1773_177359

theorem number_satisfying_equation : ∃! x : ℚ, x + 72 = 2 * x / (2/3) := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_equation_l1773_177359


namespace NUMINAMATH_CALUDE_linda_travel_distance_l1773_177328

/-- Represents the travel data for one day --/
structure DayTravel where
  totalTime : ℕ
  timePerMile : ℕ

/-- Calculates the distance traveled in a day --/
def distanceTraveled (day : DayTravel) : ℚ :=
  day.totalTime / day.timePerMile

/-- Represents Linda's travel data over three days --/
structure ThreeDayTravel where
  day1 : DayTravel
  day2 : DayTravel
  day3 : DayTravel

/-- The main theorem to prove --/
theorem linda_travel_distance 
  (travel : ThreeDayTravel)
  (time_condition : travel.day1.totalTime = 60 ∧ 
                    travel.day2.totalTime = 75 ∧ 
                    travel.day3.totalTime = 90)
  (time_increase : travel.day2.timePerMile = travel.day1.timePerMile + 3 ∧
                   travel.day3.timePerMile = travel.day2.timePerMile + 3)
  (integer_distance : ∀ d : DayTravel, d ∈ [travel.day1, travel.day2, travel.day3] → 
                      (distanceTraveled d).den = 1)
  (integer_time : ∀ d : DayTravel, d ∈ [travel.day1, travel.day2, travel.day3] → 
                  d.timePerMile > 0) :
  (distanceTraveled travel.day1 + distanceTraveled travel.day2 + distanceTraveled travel.day3 : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_linda_travel_distance_l1773_177328


namespace NUMINAMATH_CALUDE_triangle_area_equation_l1773_177308

theorem triangle_area_equation : ∃! (x : ℝ), x > 3 ∧ (1/2 : ℝ) * (x - 3) * (3*x + 7) = 12*x - 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_equation_l1773_177308


namespace NUMINAMATH_CALUDE_probability_theorem_l1773_177376

def red_marbles : ℕ := 15
def blue_marbles : ℕ := 9
def green_marbles : ℕ := 6
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles

def probability_two_blue_one_red_one_green : ℚ :=
  (Nat.choose blue_marbles 2 * Nat.choose red_marbles 1 * Nat.choose green_marbles 1) /
  Nat.choose total_marbles 4

theorem probability_theorem :
  probability_two_blue_one_red_one_green = 5 / 812 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1773_177376


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1773_177355

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x / 3 < 1 - (x - 3) / 6 ∧ x < m) ↔ x < 3) → m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1773_177355


namespace NUMINAMATH_CALUDE_calculate_premium_rate_l1773_177345

/-- Calculates the premium rate for shares given the investment details --/
theorem calculate_premium_rate (investment total_dividend face_value dividend_rate : ℚ)
  (h1 : investment = 14400)
  (h2 : total_dividend = 600)
  (h3 : face_value = 100)
  (h4 : dividend_rate = 5 / 100) :
  ∃ premium_rate : ℚ,
    premium_rate = 20 ∧
    (investment / (face_value + premium_rate)) * (face_value * dividend_rate) = total_dividend :=
by sorry

end NUMINAMATH_CALUDE_calculate_premium_rate_l1773_177345


namespace NUMINAMATH_CALUDE_common_tangent_count_possibilities_l1773_177399

/-- The number of possible values for the count of common tangents between two circles -/
def possible_tangent_counts : ℕ := 5

/-- The radii of the two circles -/
def circle_radii : Fin 2 → ℝ
  | 0 => 2
  | 1 => 3

/-- The set of possible numbers of common tangents -/
def tangent_counts : Finset ℕ := {0, 1, 2, 3, 4}

/-- Theorem stating that the number of possible values for the count of common tangents
    between two circles with radii 2 and 3 is equal to the cardinality of the set of
    possible numbers of common tangents -/
theorem common_tangent_count_possibilities :
  possible_tangent_counts = Finset.card tangent_counts :=
sorry

end NUMINAMATH_CALUDE_common_tangent_count_possibilities_l1773_177399


namespace NUMINAMATH_CALUDE_g_properties_l1773_177380

noncomputable def f (a x : ℝ) : ℝ := a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)

noncomputable def g (a : ℝ) : ℝ := ⨆ (x : ℝ), f a x

theorem g_properties (a : ℝ) :
  (a > -1/2 → g a = a + 2) ∧
  (-Real.sqrt 2 / 2 < a ∧ a ≤ -1/2 → g a = -a - 1/(2*a)) ∧
  (a ≤ -Real.sqrt 2 / 2 → g a = Real.sqrt 2) ∧
  (g a = g (1/a) ↔ a = 1 ∨ (-Real.sqrt 2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2)) :=
sorry

end NUMINAMATH_CALUDE_g_properties_l1773_177380


namespace NUMINAMATH_CALUDE_five_digit_reverse_multiply_nine_l1773_177395

theorem five_digit_reverse_multiply_nine :
  ∃! n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧
    (∃ a b c d e : ℕ,
      n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
      9 * n = 10000 * e + 1000 * d + 100 * c + 10 * b + a ∧
      a ≠ 0) ∧
    n = 10989 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_reverse_multiply_nine_l1773_177395


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l1773_177366

/-- A hyperbola with focus and asymptotes -/
structure Hyperbola where
  /-- The right focus of the hyperbola -/
  focus : ℝ × ℝ
  /-- The asymptotes of the hyperbola, represented as slopes -/
  asymptotes : ℝ × ℝ

/-- The symmetric point of a point with respect to a line -/
def symmetricPoint (p : ℝ × ℝ) (slope : ℝ) : ℝ × ℝ := sorry

/-- Check if a point lies on a line given by its slope -/
def liesOn (p : ℝ × ℝ) (slope : ℝ) : Prop := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: If the symmetric point of the focus with respect to one asymptote
    lies on the other asymptote, then the eccentricity is 2 -/
theorem hyperbola_eccentricity_is_two (h : Hyperbola) :
  let (slope1, slope2) := h.asymptotes
  let symPoint := symmetricPoint h.focus slope1
  liesOn symPoint slope2 → eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l1773_177366


namespace NUMINAMATH_CALUDE_identity_function_characterization_l1773_177309

theorem identity_function_characterization (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y)
  (h_one : f 1 = 1)
  (h_additive : ∀ x y, f (x + y) = f x + f y) :
  ∀ x, f x = x :=
sorry

end NUMINAMATH_CALUDE_identity_function_characterization_l1773_177309


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l1773_177371

/-- Given a real number a and a function f(x) = x^3 + ax^2 + (a - 2)x whose derivative
    is an even function, the tangent line to f(x) at the origin has equation y = -2x. -/
theorem tangent_line_at_origin (a : ℝ) :
  let f := fun x : ℝ => x^3 + a*x^2 + (a - 2)*x
  let f' := fun x : ℝ => 3*x^2 + 2*a*x + (a - 2)
  (∀ x, f' x = f' (-x)) →
  (fun x => -2*x) = fun x => (f' 0) * x :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l1773_177371


namespace NUMINAMATH_CALUDE_derivative_at_one_l1773_177314

-- Define the function
def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1773_177314


namespace NUMINAMATH_CALUDE_max_chocolates_bob_l1773_177384

/-- Given that Bob and Carol share 36 chocolates, and Carol eats a positive multiple
    of Bob's chocolates, prove that the maximum number of chocolates Bob could have eaten is 18. -/
theorem max_chocolates_bob (total : ℕ) (bob carol : ℕ) (k : ℕ) : 
  total = 36 →
  bob + carol = total →
  carol = k * bob →
  k > 0 →
  bob ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_max_chocolates_bob_l1773_177384


namespace NUMINAMATH_CALUDE_second_grade_sample_count_l1773_177367

/-- Represents a high school with three grades forming an arithmetic sequence -/
structure HighSchool where
  total_students : ℕ
  sampled_students : ℕ
  grade_sequence : Fin 3 → ℕ
  is_arithmetic_sequence : ∃ (d : ℤ), 
    (grade_sequence 1 : ℤ) = (grade_sequence 0 : ℤ) + d ∧
    (grade_sequence 2 : ℤ) = (grade_sequence 1 : ℤ) + d
  sum_equals_total : (grade_sequence 0) + (grade_sequence 1) + (grade_sequence 2) = total_students

/-- The number of students sampled from the second grade in a stratified sampling -/
def sampled_from_second_grade (school : HighSchool) : ℕ :=
  (school.grade_sequence 1 * school.sampled_students) / school.total_students

/-- Theorem stating the number of students sampled from the second grade -/
theorem second_grade_sample_count 
  (school : HighSchool)
  (h1 : school.total_students = 1200)
  (h2 : school.sampled_students = 48) :
  sampled_from_second_grade school = 16 := by
  sorry


end NUMINAMATH_CALUDE_second_grade_sample_count_l1773_177367


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1773_177301

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1773_177301


namespace NUMINAMATH_CALUDE_union_equality_condition_l1773_177390

open Set

theorem union_equality_condition (a : ℝ) :
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
  let B : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
  (A ∪ B = A) ↔ a ∈ Set.Icc (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_condition_l1773_177390


namespace NUMINAMATH_CALUDE_stoichiometric_ratio_l1773_177305

-- Define the reaction rates
variable (vA vB vC : ℝ)

-- Define the relationships between reaction rates
axiom rate_relation1 : vB = 3 * vA
axiom rate_relation2 : 3 * vC = 2 * vB

-- Define the stoichiometric coefficients
variable (a b c : ℕ)

-- Theorem: Given the rate relationships, prove the stoichiometric coefficient ratio
theorem stoichiometric_ratio : 
  vB = 3 * vA → 3 * vC = 2 * vB → a = 1 ∧ b = 3 ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_stoichiometric_ratio_l1773_177305


namespace NUMINAMATH_CALUDE_bitcoin_transfer_theorem_l1773_177393

/-- Represents the number of bitcoins each person has at a given step -/
structure BitcoinState where
  sasha : ℤ
  pasha : ℤ
  arkasha : ℤ

/-- Performs the described series of transfers -/
def perform_transfers (initial : BitcoinState) : BitcoinState :=
  let step1 := BitcoinState.mk
    (initial.sasha - initial.pasha)
    (2 * initial.pasha)
    initial.arkasha
  let step2 := BitcoinState.mk
    (step1.sasha - initial.arkasha)
    step1.pasha
    (2 * initial.arkasha)
  let step3 := BitcoinState.mk
    (2 * step2.sasha)
    (step2.pasha - step2.sasha - step2.arkasha)
    (2 * step2.arkasha)
  BitcoinState.mk
    (step3.sasha + step3.sasha)
    (step3.pasha + step3.sasha)
    (step3.arkasha - step3.sasha - step3.pasha)

theorem bitcoin_transfer_theorem (initial : BitcoinState) :
  let final := perform_transfers initial
  final.sasha = 8 ∧ final.pasha = 8 ∧ final.arkasha = 8 →
  initial.sasha = 13 ∧ initial.pasha = 7 ∧ initial.arkasha = 4 := by
  sorry

#check bitcoin_transfer_theorem

end NUMINAMATH_CALUDE_bitcoin_transfer_theorem_l1773_177393


namespace NUMINAMATH_CALUDE_fruit_arrangement_theorem_l1773_177388

def number_of_arrangements (total : ℕ) (group1 : ℕ) (group2 : ℕ) (unique : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial group1 * Nat.factorial group2 * Nat.factorial unique)

theorem fruit_arrangement_theorem :
  number_of_arrangements 7 4 2 1 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_theorem_l1773_177388


namespace NUMINAMATH_CALUDE_tangent_slope_circle_l1773_177348

/-- Slope of the line tangent to a circle -/
theorem tangent_slope_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (3, 2) → point = (5, 5) → 
  (let radius_slope := (point.2 - center.2) / (point.1 - center.1);
   -1 / radius_slope) = -2/3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_circle_l1773_177348


namespace NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l1773_177354

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℝ) :
  a 1 = 1 →
  a 4 = ∫ x in (1 : ℝ)..2, 3 * x^2 →
  ∃ d, arithmetic_sequence a d ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l1773_177354


namespace NUMINAMATH_CALUDE_green_balls_count_l1773_177358

theorem green_balls_count (red : ℕ) (blue : ℕ) (prob : ℚ) (green : ℕ) : 
  red = 3 → 
  blue = 2 → 
  prob = 1/12 → 
  (red : ℚ)/(red + blue + green : ℚ) * ((red - 1 : ℚ)/(red + blue + green - 1 : ℚ)) = prob → 
  green = 4 :=
by sorry

end NUMINAMATH_CALUDE_green_balls_count_l1773_177358


namespace NUMINAMATH_CALUDE_pq_length_l1773_177394

/-- The exact length of PQ given the specified conditions -/
theorem pq_length : ∃ (P Q : ℝ × ℝ),
  let R : ℝ × ℝ := (10, 8)
  let line1 (x y : ℝ) := 7 * y = 9 * x
  let line2 (x y : ℝ) := 12 * y = 5 * x
  (P.1 + Q.1) / 2 = R.1 ∧
  (P.2 + Q.2) / 2 = R.2 ∧
  line1 P.1 P.2 ∧
  line2 Q.1 Q.2 →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 4 * Real.sqrt 134481 / 73 :=
by sorry

end NUMINAMATH_CALUDE_pq_length_l1773_177394


namespace NUMINAMATH_CALUDE_original_number_proof_l1773_177397

theorem original_number_proof (x : ℝ) : x * 1.2 = 1800 → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1773_177397


namespace NUMINAMATH_CALUDE_expo_stamps_theorem_l1773_177389

theorem expo_stamps_theorem (total_cost : ℕ) (cost_4 cost_8 : ℕ) (difference : ℕ) :
  total_cost = 660 →
  cost_4 = 4 →
  cost_8 = 8 →
  difference = 30 →
  ∃ (stamps_4 stamps_8 : ℕ),
    stamps_8 = stamps_4 + difference ∧
    total_cost = cost_4 * stamps_4 + cost_8 * stamps_8 →
    stamps_4 + stamps_8 = 100 :=
by sorry

end NUMINAMATH_CALUDE_expo_stamps_theorem_l1773_177389


namespace NUMINAMATH_CALUDE_champagne_discount_percentage_l1773_177319

-- Define the problem parameters
def hot_tub_capacity : ℝ := 40
def bottle_capacity : ℝ := 1
def quarts_per_gallon : ℝ := 4
def original_price_per_bottle : ℝ := 50
def total_spent_after_discount : ℝ := 6400

-- Define the theorem
theorem champagne_discount_percentage :
  let total_quarts : ℝ := hot_tub_capacity * quarts_per_gallon
  let total_bottles : ℝ := total_quarts / bottle_capacity
  let full_price : ℝ := total_bottles * original_price_per_bottle
  let discount_amount : ℝ := full_price - total_spent_after_discount
  let discount_percentage : ℝ := (discount_amount / full_price) * 100
  discount_percentage = 20 := by
  sorry


end NUMINAMATH_CALUDE_champagne_discount_percentage_l1773_177319


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1773_177311

theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : marked_price = 200)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.2) :
  ∃ (cost_price : ℝ),
    cost_price = 150 ∧ 
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1773_177311


namespace NUMINAMATH_CALUDE_q_share_is_7200_l1773_177339

/-- Calculates the share of profit for a partner in a business partnership. -/
def calculateShareOfProfit (investment1 : ℕ) (investment2 : ℕ) (totalProfit : ℕ) : ℕ :=
  let totalInvestment := investment1 + investment2
  (investment2 * totalProfit) / totalInvestment

/-- Theorem stating that Q's share of the profit is 7200 given the specified investments and total profit. -/
theorem q_share_is_7200 :
  calculateShareOfProfit 54000 36000 18000 = 7200 := by
  sorry

#eval calculateShareOfProfit 54000 36000 18000

end NUMINAMATH_CALUDE_q_share_is_7200_l1773_177339


namespace NUMINAMATH_CALUDE_triangle_base_value_l1773_177322

theorem triangle_base_value (L R B : ℝ) : 
  L + R + B = 50 →
  R = L + 2 →
  L = 12 →
  B = 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_value_l1773_177322


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_primes_l1773_177315

def first_17_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

def is_divisible_by_all_except_two_consecutive (n : Nat) (primes : List Nat) (i : Nat) : Prop :=
  ∀ (p : Nat), p ∈ primes → (p ≠ primes[i]! ∧ p ≠ primes[i+1]!) → n % p = 0

theorem smallest_number_divisible_by_primes : ∃ (n : Nat),
  is_divisible_by_all_except_two_consecutive n first_17_primes 15 ∧
  ∀ (m : Nat), m < n → ¬is_divisible_by_all_except_two_consecutive m first_17_primes 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_primes_l1773_177315


namespace NUMINAMATH_CALUDE_increasing_quadratic_condition_l1773_177387

/-- If f(x) = -x^2 + 2ax - 3 is increasing on (-∞, 4), then a < 4 -/
theorem increasing_quadratic_condition (a : ℝ) : 
  (∀ x < 4, Monotone (fun x => -x^2 + 2*a*x - 3)) → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_increasing_quadratic_condition_l1773_177387


namespace NUMINAMATH_CALUDE_library_book_distribution_l1773_177396

/-- The number of ways to distribute n identical objects between two locations,
    with at least one object in each location. -/
def distributionWays (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

/-- The problem statement as a theorem -/
theorem library_book_distribution :
  distributionWays 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_library_book_distribution_l1773_177396


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l1773_177300

theorem chocolate_box_problem (C : ℝ) : 
  C > 0 →  -- Ensure the number of chocolates is positive
  (C / 2 - 0.8 * (C / 2)) + (C / 2 - 0.5 * (C / 2)) = 28 →
  C = 80 := by
sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l1773_177300


namespace NUMINAMATH_CALUDE_prob_three_even_out_of_five_l1773_177332

-- Define a fair 20-sided die
def fair_20_sided_die : Finset ℕ := Finset.range 20

-- Define the probability of rolling an even number on a fair 20-sided die
def prob_even (d : Finset ℕ) : ℚ :=
  (d.filter (λ x => x % 2 = 0)).card / d.card

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the number of dice we want to show even
def num_even : ℕ := 3

-- Theorem statement
theorem prob_three_even_out_of_five :
  prob_even fair_20_sided_die = 1/2 →
  (num_dice.choose num_even : ℚ) * (1/2)^num_dice = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_out_of_five_l1773_177332


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_seven_l1773_177317

theorem reciprocal_of_negative_seven :
  (1 : ℚ) / (-7 : ℚ) = -1/7 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_seven_l1773_177317


namespace NUMINAMATH_CALUDE_pen_difference_after_four_weeks_l1773_177333

/-- The difference in pens between Alex and Jane after 4 weeks -/
def pen_difference (A B : ℕ) (X Y : ℝ) (M N : ℕ) : ℕ :=
  M - N

/-- Theorem stating the difference in pens after 4 weeks -/
theorem pen_difference_after_four_weeks 
  (A B : ℕ) (X Y : ℝ) (M N : ℕ) 
  (hM : M = A * X^4) 
  (hN : N = B * Y^4) :
  pen_difference A B X Y M N = M - N :=
by
  sorry

end NUMINAMATH_CALUDE_pen_difference_after_four_weeks_l1773_177333


namespace NUMINAMATH_CALUDE_max_positive_numbers_with_zero_average_l1773_177336

theorem max_positive_numbers_with_zero_average (numbers : List ℝ) : 
  numbers.length = 20 → numbers.sum / numbers.length = 0 → 
  (numbers.filter (λ x => x > 0)).length ≤ 19 := by
sorry

end NUMINAMATH_CALUDE_max_positive_numbers_with_zero_average_l1773_177336


namespace NUMINAMATH_CALUDE_distinct_colorings_count_l1773_177362

/-- Represents the symmetries of a square -/
inductive SquareSymmetry
| Rotation0 | Rotation90 | Rotation180 | Rotation270
| ReflectionSide1 | ReflectionSide2
| ReflectionDiag1 | ReflectionDiag2

/-- Represents a coloring of the square's disks -/
structure SquareColoring :=
(blue1 : Fin 4)
(blue2 : Fin 4)
(red : Fin 4)
(green : Fin 4)

/-- The group of symmetries of a square -/
def squareSymmetryGroup : List SquareSymmetry :=
[SquareSymmetry.Rotation0, SquareSymmetry.Rotation90, SquareSymmetry.Rotation180, SquareSymmetry.Rotation270,
 SquareSymmetry.ReflectionSide1, SquareSymmetry.ReflectionSide2,
 SquareSymmetry.ReflectionDiag1, SquareSymmetry.ReflectionDiag2]

/-- Checks if a coloring is valid (2 blue, 1 red, 1 green) -/
def isValidColoring (c : SquareColoring) : Bool :=
  c.blue1 ≠ c.blue2 ∧ c.blue1 ≠ c.red ∧ c.blue1 ≠ c.green ∧
  c.blue2 ≠ c.red ∧ c.blue2 ≠ c.green ∧ c.red ≠ c.green

/-- Checks if a coloring is fixed by a given symmetry -/
def isFixedBy (c : SquareColoring) (s : SquareSymmetry) : Bool := sorry

/-- Counts the number of colorings fixed by each symmetry -/
def countFixedColorings (s : SquareSymmetry) : Nat := sorry

/-- The main theorem: there are 3 distinct colorings under symmetry -/
theorem distinct_colorings_count :
  (List.sum (List.map countFixedColorings squareSymmetryGroup)) / squareSymmetryGroup.length = 3 := sorry

end NUMINAMATH_CALUDE_distinct_colorings_count_l1773_177362


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l1773_177346

/-- Represents the profit function for a product -/
def profit_function (initial_price initial_volume : ℝ) (price_increase : ℝ) : ℝ → ℝ :=
  λ x => (initial_price + x - 80) * (initial_volume - 20 * x)

/-- Theorem stating that the profit-maximizing price is 95 yuan -/
theorem profit_maximizing_price :
  let initial_price : ℝ := 90
  let initial_volume : ℝ := 400
  let price_increase : ℝ := 1
  let profit := profit_function initial_price initial_volume price_increase
  ∃ (max_price : ℝ), max_price = 95 ∧
    ∀ (x : ℝ), profit x ≤ profit (max_price - initial_price) :=
by sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l1773_177346


namespace NUMINAMATH_CALUDE_expected_girls_left_of_boys_l1773_177375

/-- The number of boys in the lineup -/
def num_boys : ℕ := 10

/-- The number of girls in the lineup -/
def num_girls : ℕ := 7

/-- The total number of students in the lineup -/
def total_students : ℕ := num_boys + num_girls

/-- The expected number of girls standing to the left of all boys -/
def expected_girls_left : ℚ := 7 / 11

theorem expected_girls_left_of_boys :
  let random_arrangement := (Finset.range total_students).powerset
  expected_girls_left = (num_girls : ℚ) / (total_students + 1 : ℚ) := by sorry

end NUMINAMATH_CALUDE_expected_girls_left_of_boys_l1773_177375


namespace NUMINAMATH_CALUDE_fraction_problem_l1773_177318

theorem fraction_problem (N : ℚ) : (5 / 6 : ℚ) * N = (5 / 16 : ℚ) * N + 250 → N = 480 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1773_177318


namespace NUMINAMATH_CALUDE_train_crossing_time_train_crossing_time_specific_l1773_177323

/-- The time taken for a train to cross a post, given its speed and length -/
theorem train_crossing_time (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  let speed_ms : ℝ := speed_kmh * 1000 / 3600
  length_m / speed_ms

/-- Proof that a train with speed 40 km/h and length 220.0176 m takes approximately 19.80176 seconds to cross a post -/
theorem train_crossing_time_specific :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |train_crossing_time 40 220.0176 - 19.80176| < ε :=
sorry

end NUMINAMATH_CALUDE_train_crossing_time_train_crossing_time_specific_l1773_177323


namespace NUMINAMATH_CALUDE_alice_unanswered_questions_l1773_177369

/-- Represents the scoring systems and Alice's results in a math competition. -/
structure MathCompetition where
  total_questions : ℕ
  new_correct_points : ℕ
  new_incorrect_points : ℕ
  new_unanswered_points : ℕ
  old_start_points : ℕ
  old_correct_points : ℕ
  old_incorrect_points : Int
  old_unanswered_points : ℕ
  new_score : ℕ
  old_score : ℕ

/-- Calculates the number of unanswered questions in the math competition. -/
def calculate_unanswered_questions (comp : MathCompetition) : ℕ :=
  sorry

/-- Theorem stating that Alice left 2 questions unanswered. -/
theorem alice_unanswered_questions (comp : MathCompetition)
  (h1 : comp.total_questions = 30)
  (h2 : comp.new_correct_points = 4)
  (h3 : comp.new_incorrect_points = 0)
  (h4 : comp.new_unanswered_points = 1)
  (h5 : comp.old_start_points = 20)
  (h6 : comp.old_correct_points = 3)
  (h7 : comp.old_incorrect_points = -1)
  (h8 : comp.old_unanswered_points = 0)
  (h9 : comp.new_score = 87)
  (h10 : comp.old_score = 75) :
  calculate_unanswered_questions comp = 2 := by
  sorry

end NUMINAMATH_CALUDE_alice_unanswered_questions_l1773_177369


namespace NUMINAMATH_CALUDE_hypotenuse_length_is_double_short_leg_l1773_177377

/-- A right triangle with a 30-60-90 degree angle configuration -/
structure RightTriangle30_60_90 where
  -- The length of the side opposite to the 30° angle
  short_leg : ℝ
  -- Assertion that the short leg is positive
  short_leg_pos : short_leg > 0

/-- The length of the hypotenuse in a 30-60-90 right triangle -/
def hypotenuse_length (t : RightTriangle30_60_90) : ℝ :=
  2 * t.short_leg

/-- Theorem: In a 30-60-90 right triangle with short leg of length 5,
    the hypotenuse has length 10 -/
theorem hypotenuse_length_is_double_short_leg :
  let t : RightTriangle30_60_90 := ⟨5, by norm_num⟩
  hypotenuse_length t = 10 := by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_is_double_short_leg_l1773_177377


namespace NUMINAMATH_CALUDE_ice_cream_arrangement_count_l1773_177338

theorem ice_cream_arrangement_count : Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangement_count_l1773_177338


namespace NUMINAMATH_CALUDE_balloon_ratio_is_seven_l1773_177347

-- Define the number of balloons for Dan and Tim
def dans_balloons : ℕ := 29
def tims_balloons : ℕ := 203

-- Define the ratio of Tim's balloons to Dan's balloons
def balloon_ratio : ℚ := tims_balloons / dans_balloons

-- Theorem stating that the ratio is 7
theorem balloon_ratio_is_seven : balloon_ratio = 7 := by
  sorry

end NUMINAMATH_CALUDE_balloon_ratio_is_seven_l1773_177347


namespace NUMINAMATH_CALUDE_katie_marbles_count_l1773_177304

def pink_marbles : ℕ := 13

def orange_marbles (pink : ℕ) : ℕ := pink - 9

def purple_marbles (orange : ℕ) : ℕ := 4 * orange

def total_marbles (pink orange purple : ℕ) : ℕ := pink + orange + purple

theorem katie_marbles_count :
  total_marbles pink_marbles (orange_marbles pink_marbles) (purple_marbles (orange_marbles pink_marbles)) = 33 :=
by
  sorry


end NUMINAMATH_CALUDE_katie_marbles_count_l1773_177304


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l1773_177372

theorem largest_prime_factors_difference (n : Nat) (h : n = 178469) : 
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧ 
  p ∣ n ∧ q ∣ n ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ n → r ≤ p) ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
  p - q = 2 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l1773_177372


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l1773_177342

theorem division_multiplication_problem : 377 / 13 / 29 * (1 / 4) / 2 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l1773_177342


namespace NUMINAMATH_CALUDE_number_problem_l1773_177392

theorem number_problem (x : ℝ) : 0.1 * x = 0.2 * 650 + 190 → x = 3200 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1773_177392


namespace NUMINAMATH_CALUDE_length_of_24_l1773_177351

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer. -/
def length (n : ℕ) : ℕ := sorry

/-- 24 can be expressed as a product of 4 prime factors. -/
theorem length_of_24 : length 24 = 4 := by sorry

end NUMINAMATH_CALUDE_length_of_24_l1773_177351


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l1773_177398

theorem inscribed_quadrilateral_fourth_side 
  (r : ℝ) 
  (a b c : ℝ) 
  (h1 : r = 150 * Real.sqrt 3)
  (h2 : a = 150)
  (h3 : b = 300)
  (h4 : c = 150)
  (h5 : ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ Real.cos θ = 0) :
  ∃ d : ℝ, d = 300 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l1773_177398


namespace NUMINAMATH_CALUDE_rectangle_formation_ways_l1773_177331

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 5

theorem rectangle_formation_ways : 
  (Nat.choose horizontal_lines 2) * (Nat.choose vertical_lines 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_ways_l1773_177331


namespace NUMINAMATH_CALUDE_triangle_area_l1773_177334

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define the height of the triangle
def Height (A B C H : ℝ × ℝ) (h : ℝ) : Prop := sorry

-- Define the angles of the triangle
def Angle (A B C : ℝ × ℝ) (α : ℝ) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area (A B C H : ℝ × ℝ) (h α γ : ℝ) :
  Triangle A B C →
  Height A B C H h →
  Angle B A C α →
  Angle B C A γ →
  TriangleArea A B C = (h^2 * Real.sin α) / (2 * Real.sin γ * Real.sin (α + γ)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1773_177334


namespace NUMINAMATH_CALUDE_remainder_13_pow_2048_mod_11_l1773_177324

theorem remainder_13_pow_2048_mod_11 : 13^2048 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13_pow_2048_mod_11_l1773_177324


namespace NUMINAMATH_CALUDE_alternating_series_sum_l1773_177330

def alternating_series (n : ℕ) : ℤ := 
  if n % 2 = 0 then (n + 1) else -(n + 1)

def series_sum (n : ℕ) : ℤ := 
  (Finset.range n).sum (λ i => alternating_series i)

theorem alternating_series_sum : series_sum 10001 = -5001 := by
  sorry

end NUMINAMATH_CALUDE_alternating_series_sum_l1773_177330


namespace NUMINAMATH_CALUDE_plane_equation_proof_l1773_177357

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane equation in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point satisfies a plane equation -/
def satisfiesPlaneEquation (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- Check if two plane equations are parallel -/
def areParallelPlanes (eq1 eq2 : PlaneEquation) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ eq1.A = k * eq2.A ∧ eq1.B = k * eq2.B ∧ eq1.C = k * eq2.C

theorem plane_equation_proof : 
  let givenPoint : Point3D := ⟨2, -3, 1⟩
  let givenPlane : PlaneEquation := ⟨3, -2, 1, -5⟩
  let resultPlane : PlaneEquation := ⟨3, -2, 1, -13⟩
  satisfiesPlaneEquation givenPoint resultPlane ∧ 
  areParallelPlanes resultPlane givenPlane ∧
  resultPlane.A > 0 ∧
  Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs resultPlane.A) (Int.natAbs resultPlane.B)) 
                   (Int.natAbs resultPlane.C)) 
          (Int.natAbs resultPlane.D) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l1773_177357


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1773_177378

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 - 9 * x ≤ 15) ↔ ((3 - Real.sqrt 29) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 29) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1773_177378


namespace NUMINAMATH_CALUDE_system_solutions_l1773_177370

theorem system_solutions (a : ℝ) (x y : ℝ) 
  (h1 : x - 2*y = 3 - a) 
  (h2 : x + y = 2*a) 
  (h3 : -2 ≤ a ∧ a ≤ 0) : 
  (a = 0 → x = -y) ∧ 
  (a = -1 → 2*x - y = 1 - a) := by
sorry

end NUMINAMATH_CALUDE_system_solutions_l1773_177370


namespace NUMINAMATH_CALUDE_farm_animal_ratio_l1773_177313

/-- Proves that the initial ratio of horses to cows is 4:1 given the problem conditions --/
theorem farm_animal_ratio :
  ∀ (h c : ℕ),  -- Initial number of horses and cows
  (h - 15 : ℚ) / (c + 15 : ℚ) = 13 / 7 →  -- Ratio after transaction
  h - 15 = c + 15 + 30 →  -- Difference after transaction
  h / c = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_farm_animal_ratio_l1773_177313


namespace NUMINAMATH_CALUDE_exists_Q_on_x_axis_l1773_177379

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the fixed line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 4}

-- Define the locus E
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4 + p.2^2 / 3 = 1)}

-- Define point A as the intersection of E with negative x-axis
def A : ℝ × ℝ := (-2, 0)

-- Define a function to represent a line through F not coinciding with x-axis
def lineThruF (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = m * p.2 + 1}

-- Define B and C as intersections of E with lineThruF
def B (m : ℝ) : ℝ × ℝ := sorry
def C (m : ℝ) : ℝ × ℝ := sorry

-- Define M and N as intersections of AB and AC with l
def M (m : ℝ) : ℝ × ℝ := sorry
def N (m : ℝ) : ℝ × ℝ := sorry

-- Define the theorem
theorem exists_Q_on_x_axis :
  ∃ x₀ : ℝ, let Q := (x₀, 0)
  ∀ m : ℝ, m ≠ 0 →
    (Q.1 - (M m).1) * (Q.1 - (N m).1) +
    (Q.2 - (M m).2) * (Q.2 - (N m).2) = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_Q_on_x_axis_l1773_177379


namespace NUMINAMATH_CALUDE_maynards_dog_holes_l1773_177352

theorem maynards_dog_holes : 
  ∀ (total : ℕ) (filled : ℕ) (unfilled : ℕ),
    filled = (75 * total) / 100 →
    unfilled = 2 →
    total = filled + unfilled →
    total = 8 := by
  sorry

end NUMINAMATH_CALUDE_maynards_dog_holes_l1773_177352


namespace NUMINAMATH_CALUDE_gcd_of_390_455_546_l1773_177383

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_390_455_546_l1773_177383


namespace NUMINAMATH_CALUDE_toy_bridge_weight_l1773_177356

theorem toy_bridge_weight (total_weight : ℕ) (num_full_cans : ℕ) (soda_weight : ℕ) (empty_can_weight : ℕ) :
  total_weight = 88 →
  num_full_cans = 6 →
  soda_weight = 12 →
  empty_can_weight = 2 →
  (num_full_cans * (soda_weight + empty_can_weight) + (total_weight - num_full_cans * (soda_weight + empty_can_weight))) / empty_can_weight = 2 :=
by sorry

end NUMINAMATH_CALUDE_toy_bridge_weight_l1773_177356


namespace NUMINAMATH_CALUDE_range_of_a_l1773_177363

theorem range_of_a (p q : Prop) (a : ℝ) 
  (hp : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (hq : ∃ x : ℝ, x^2 - 4*x + a ≤ 0) :
  a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1773_177363


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1773_177321

theorem polygon_interior_angles_sum (n : ℕ) (sum : ℝ) : 
  sum = 900 → (n - 2) * 180 = sum → n = 7 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1773_177321


namespace NUMINAMATH_CALUDE_jeffrey_steps_l1773_177382

/-- Represents Jeffrey's walking pattern --/
structure WalkingPattern where
  forward : Nat
  backward : Nat

/-- Calculates the effective steps for a given pattern and number of repetitions --/
def effectiveSteps (pattern : WalkingPattern) (repetitions : Nat) : Nat :=
  (pattern.forward - pattern.backward) * repetitions

/-- Calculates the total steps taken for a given pattern and number of repetitions --/
def totalSteps (pattern : WalkingPattern) (repetitions : Nat) : Nat :=
  (pattern.forward + pattern.backward) * repetitions

/-- Theorem stating the total number of steps Jeffrey takes --/
theorem jeffrey_steps :
  let initialPattern : WalkingPattern := ⟨3, 2⟩
  let changedPattern : WalkingPattern := ⟨4, 1⟩
  let totalDistance : Nat := 66
  let initialEffectiveSteps : Nat := 30
  let initialRepetitions : Nat := initialEffectiveSteps / (initialPattern.forward - initialPattern.backward)
  let remainingDistance : Nat := totalDistance - initialEffectiveSteps
  let changedRepetitions : Nat := remainingDistance / (changedPattern.forward - changedPattern.backward)
  totalSteps initialPattern initialRepetitions + totalSteps changedPattern changedRepetitions = 210 := by
  sorry

end NUMINAMATH_CALUDE_jeffrey_steps_l1773_177382


namespace NUMINAMATH_CALUDE_article_gain_percentage_l1773_177349

/-- Calculates the cost price given the selling price and loss percentage -/
def costPrice (sellingPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  sellingPrice / (1 - lossPercentage / 100)

/-- Calculates the gain percentage given the cost price and selling price -/
def gainPercentage (costPrice : ℚ) (sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

theorem article_gain_percentage :
  let cp := costPrice 170 15
  gainPercentage cp 240 = 20 := by
  sorry

end NUMINAMATH_CALUDE_article_gain_percentage_l1773_177349


namespace NUMINAMATH_CALUDE_min_translation_for_even_sine_l1773_177344

theorem min_translation_for_even_sine (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.sin (3 * x + π / 4)) →
  m > 0 →
  (∀ x, f (x + m) = f (-x - m)) →
  m ≥ π / 12 ∧ ∃ m₀ > 0, m₀ < m → ¬(∀ x, f (x + m₀) = f (-x - m₀)) :=
by sorry

end NUMINAMATH_CALUDE_min_translation_for_even_sine_l1773_177344


namespace NUMINAMATH_CALUDE_fraction_equality_with_different_numerator_denominator_relations_l1773_177341

theorem fraction_equality_with_different_numerator_denominator_relations : 
  ∃ (a b c d : ℤ), a < b ∧ c > d ∧ (a : ℚ) / b = (c : ℚ) / d := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_with_different_numerator_denominator_relations_l1773_177341


namespace NUMINAMATH_CALUDE_area_at_stage_8_l1773_177327

/-- The side length of each square in inches -/
def square_side : ℝ := 4

/-- The number of squares at a given stage -/
def num_squares (stage : ℕ) : ℕ := stage

/-- The area of the rectangle at a given stage in square inches -/
def rectangle_area (stage : ℕ) : ℝ :=
  (num_squares stage) * (square_side ^ 2)

/-- Theorem: The area of the rectangle at Stage 8 is 128 square inches -/
theorem area_at_stage_8 : rectangle_area 8 = 128 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l1773_177327


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_7_divisible_by_5_l1773_177364

def ends_in_7 (n : ℕ) : Prop := n % 10 = 7

theorem smallest_positive_integer_ending_in_7_divisible_by_5 :
  ∃ (n : ℕ), n > 0 ∧ ends_in_7 n ∧ n % 5 = 0 ∧
  ∀ (m : ℕ), m > 0 ∧ ends_in_7 m ∧ m % 5 = 0 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_7_divisible_by_5_l1773_177364


namespace NUMINAMATH_CALUDE_standard_deviation_measures_stability_l1773_177306

-- Define a type for yield per acre
def YieldPerAcre := ℝ

-- Define a function to calculate the standard deviation
def standardDeviation (yields : List YieldPerAcre) : ℝ :=
  sorry  -- Implementation details omitted

-- Define a predicate for stability measure
def isStabilityMeasure (f : List YieldPerAcre → ℝ) : Prop :=
  sorry  -- Implementation details omitted

-- Theorem statement
theorem standard_deviation_measures_stability :
  ∀ (n : ℕ) (yields : List YieldPerAcre),
    n > 0 →
    yields.length = n →
    isStabilityMeasure standardDeviation :=
by sorry

end NUMINAMATH_CALUDE_standard_deviation_measures_stability_l1773_177306


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1773_177326

theorem count_integers_satisfying_inequality : 
  ∃ (S : Finset Int), (∀ n : Int, n ∈ S ↔ (n - 3) * (n + 5) < 0) ∧ Finset.card S = 7 :=
sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1773_177326


namespace NUMINAMATH_CALUDE_last_car_probability_2012_l1773_177335

/-- Represents the parking procedure for a given number of spots. -/
def ParkingProcedure (n : ℕ) : Type :=
  Unit

/-- Calculates the probability of the last car parking in spot 1 given the parking procedure. -/
noncomputable def lastCarProbability (n : ℕ) (proc : ParkingProcedure n) : ℚ :=
  sorry

/-- The theorem stating the probability of the last car parking in spot 1 for 2012 spots. -/
theorem last_car_probability_2012 :
  ∃ (proc : ParkingProcedure 2012), lastCarProbability 2012 proc = 1 / 2062300 :=
by
  sorry

end NUMINAMATH_CALUDE_last_car_probability_2012_l1773_177335


namespace NUMINAMATH_CALUDE_problem_statement_l1773_177337

theorem problem_statement (a b : ℤ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 5 * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1773_177337


namespace NUMINAMATH_CALUDE_probability_13_or_more_points_l1773_177361

/-- Represents the face cards in a deck --/
inductive FaceCard
  | A
  | K
  | Q
  | J

/-- Assigns points to a face card --/
def point_value (card : FaceCard) : ℕ :=
  match card with
  | FaceCard.A => 4
  | FaceCard.K => 3
  | FaceCard.Q => 2
  | FaceCard.J => 1

/-- Calculates the total points for a hand of face cards --/
def hand_points (hand : List FaceCard) : ℕ :=
  hand.map point_value |>.sum

/-- Represents all possible 4-card hands of face cards --/
def all_hands : List (List FaceCard) :=
  sorry

/-- Checks if a hand has 13 or more points --/
def has_13_or_more_points (hand : List FaceCard) : Bool :=
  hand_points hand ≥ 13

/-- Counts the number of hands with 13 or more points --/
def count_13_or_more : ℕ :=
  all_hands.filter has_13_or_more_points |>.length

theorem probability_13_or_more_points :
  count_13_or_more / all_hands.length = 197 / 1820 := by
  sorry

end NUMINAMATH_CALUDE_probability_13_or_more_points_l1773_177361


namespace NUMINAMATH_CALUDE_only_whole_number_between_l1773_177350

theorem only_whole_number_between (N : ℤ) : 
  (9.25 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 9.75) ↔ N = 38 := by
  sorry

end NUMINAMATH_CALUDE_only_whole_number_between_l1773_177350


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l1773_177329

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

-- Define the problem statement
theorem geometric_sequence_minimum (a₁ : ℝ) (q : ℝ) :
  (a₁ > 0) →
  (q > 0) →
  (geometric_sequence a₁ q 2017 = geometric_sequence a₁ q 2016 + 2 * geometric_sequence a₁ q 2015) →
  (∃ m n : ℕ, (geometric_sequence a₁ q m) * (geometric_sequence a₁ q n) = 16 * a₁^2) →
  (∃ m n : ℕ, ∀ k l : ℕ, 4/k + 1/l ≥ 4/m + 1/n ∧ 4/m + 1/n = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l1773_177329


namespace NUMINAMATH_CALUDE_peaches_in_basket_l1773_177373

/-- The number of peaches in a basket after adding more peaches is equal to
    the sum of the initial number of peaches and the number of peaches added. -/
theorem peaches_in_basket (initial : ℕ) (added : ℕ) :
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_peaches_in_basket_l1773_177373


namespace NUMINAMATH_CALUDE_smallest_translation_l1773_177374

open Real

theorem smallest_translation (φ : ℝ) : φ > 0 ∧ 
  (∀ x : ℝ, sin (2 * (x + φ)) = cos (2 * x - π / 3)) →
  φ = π / 12 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_translation_l1773_177374


namespace NUMINAMATH_CALUDE_x_is_perfect_square_l1773_177312

/-- The sequence x_n as defined in the problem -/
def x : ℕ → ℚ
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 1
  | (n + 4) => ((n^2 + n + 1) * (n + 1) / n) * x (n + 3) + 
               (n^2 + n + 1) * x (n + 2) - 
               ((n + 1) / n) * x (n + 1)

/-- The theorem stating that all members of x_n are perfect squares -/
theorem x_is_perfect_square : ∀ n : ℕ, ∃ y : ℤ, x n = (y : ℚ)^2 := by
  sorry

end NUMINAMATH_CALUDE_x_is_perfect_square_l1773_177312


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l1773_177340

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℝ := 15

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℝ := green_pill_cost - 2

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 588

theorem green_pill_cost_proof :
  green_pill_cost = 15 ∧
  pink_pill_cost = green_pill_cost - 2 ∧
  treatment_days * (green_pill_cost + pink_pill_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_green_pill_cost_proof_l1773_177340


namespace NUMINAMATH_CALUDE_arithmetic_progression_square_sum_l1773_177353

def is_four_identical_digits (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 9 ∧ n = k * 1111

theorem arithmetic_progression_square_sum (n : ℕ) : 
  is_four_identical_digits ((n - 2)^2 + n^2 + (n + 2)^2) ↔ n = 43 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_square_sum_l1773_177353


namespace NUMINAMATH_CALUDE_number_of_subjects_proof_l1773_177381

/-- Given the average scores and individual subject scores, prove the number of subjects. -/
theorem number_of_subjects_proof (physics chemistry mathematics : ℝ) : 
  (physics + chemistry + mathematics) / 3 = 75 →
  (physics + mathematics) / 2 = 90 →
  (physics + chemistry) / 2 = 70 →
  physics = 95 →
  ∃ (n : ℕ), n = 3 ∧ n > 0 := by
  sorry

#check number_of_subjects_proof

end NUMINAMATH_CALUDE_number_of_subjects_proof_l1773_177381


namespace NUMINAMATH_CALUDE_min_ear_sightings_l1773_177320

/-- Represents the direction a child is facing -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the grid of children -/
def Grid (n : Nat) := Position → Direction

/-- Counts the number of children seeing an ear in the given grid -/
def countEarSightings (n : Nat) (grid : Grid n) : Nat :=
  sorry

/-- Theorem stating the minimal number of children seeing an ear -/
theorem min_ear_sightings (n : Nat) :
  (∃ (grid : Grid n), countEarSightings n grid = n + 2) ∧
  (∀ (grid : Grid n), countEarSightings n grid ≥ n + 2) :=
sorry

end NUMINAMATH_CALUDE_min_ear_sightings_l1773_177320


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1773_177386

theorem absolute_value_inequality (x : ℝ) : |2*x + 1| < 3 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1773_177386


namespace NUMINAMATH_CALUDE_units_digit_of_F_F_10_l1773_177368

def modifiedFibonacci : ℕ → ℕ
  | 0 => 4
  | 1 => 3
  | (n + 2) => modifiedFibonacci (n + 1) + modifiedFibonacci n

theorem units_digit_of_F_F_10 : 
  (modifiedFibonacci (modifiedFibonacci 10)) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_F_F_10_l1773_177368


namespace NUMINAMATH_CALUDE_total_cost_is_56_15_l1773_177302

-- Define the prices and quantities
def spam_price : ℚ := 3
def peanut_butter_price : ℚ := 5
def bread_price : ℚ := 2
def spam_quantity : ℕ := 12
def peanut_butter_quantity : ℕ := 3
def bread_quantity : ℕ := 4

-- Define the discount and tax rates
def spam_discount : ℚ := 0.1
def peanut_butter_tax : ℚ := 0.05

-- Define the total cost function
def total_cost : ℚ :=
  (spam_price * spam_quantity * (1 - spam_discount)) +
  (peanut_butter_price * peanut_butter_quantity * (1 + peanut_butter_tax)) +
  (bread_price * bread_quantity)

-- Theorem statement
theorem total_cost_is_56_15 : total_cost = 56.15 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_56_15_l1773_177302


namespace NUMINAMATH_CALUDE_two_propositions_true_l1773_177391

theorem two_propositions_true : 
  (¬(∀ x : ℝ, x^2 > 0)) ∧ 
  (∃ x : ℝ, x^2 ≤ x) ∧ 
  (∀ M N : Set α, ∀ x : α, x ∈ M ∩ N → x ∈ M ∧ x ∈ N) := by
  sorry

end NUMINAMATH_CALUDE_two_propositions_true_l1773_177391


namespace NUMINAMATH_CALUDE_two_by_one_prism_net_squares_valid_nine_square_net_two_by_one_prism_net_property_l1773_177307

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of a rectangular prism --/
structure PrismNet where
  squares : ℕ

/-- Function to calculate the number of squares in a prism net --/
def netSquares (prism : RectangularPrism) : ℕ :=
  2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height)

/-- Theorem stating that a 2x1x1 prism net has 10 squares --/
theorem two_by_one_prism_net_squares :
  let prism : RectangularPrism := ⟨2, 1, 1⟩
  netSquares prism = 10 := by sorry

/-- Theorem stating that removing one square from a 10-square net results in a 9-square net --/
theorem valid_nine_square_net (net : PrismNet) (h : net.squares = 10) :
  ∃ (reduced_net : PrismNet), reduced_net.squares = 9 := by sorry

/-- Main theorem combining the above results --/
theorem two_by_one_prism_net_property :
  let prism : RectangularPrism := ⟨2, 1, 1⟩
  let net : PrismNet := ⟨netSquares prism⟩
  ∃ (reduced_net : PrismNet), reduced_net.squares = 9 := by sorry

end NUMINAMATH_CALUDE_two_by_one_prism_net_squares_valid_nine_square_net_two_by_one_prism_net_property_l1773_177307


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l1773_177316

/-- Represents the composition of a cricket team -/
structure CricketTeam where
  total_players : ℕ
  throwers : ℕ
  hitters : ℕ
  runners : ℕ
  left_handed_hitters : ℕ
  left_handed_runners : ℕ

/-- Calculates the total number of right-handed players in a cricket team -/
def right_handed_players (team : CricketTeam) : ℕ :=
  team.throwers + (team.hitters - team.left_handed_hitters) + (team.runners - team.left_handed_runners)

/-- Theorem stating the total number of right-handed players in the given cricket team -/
theorem cricket_team_right_handed_players :
  ∃ (team : CricketTeam),
    team.total_players = 300 ∧
    team.throwers = 165 ∧
    team.hitters = team.runners ∧
    team.hitters + team.runners = team.total_players - team.throwers ∧
    team.left_handed_hitters * 5 = team.hitters * 2 ∧
    team.left_handed_runners * 7 = team.runners * 3 ∧
    right_handed_players team = 243 :=
  sorry


end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l1773_177316


namespace NUMINAMATH_CALUDE_expression_simplification_l1773_177343

theorem expression_simplification (m : ℝ) (h1 : m ≠ 2) (h2 : m ≠ -3) :
  (m - (4*m - 9) / (m - 2)) / ((m^2 - 9) / (m - 2)) = (m - 3) / (m + 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1773_177343


namespace NUMINAMATH_CALUDE_circle_center_l1773_177360

/-- The equation of a circle in the x-y plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 4*y = -4

/-- The center of a circle -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 16

/-- Theorem: The center of the circle with equation x^2 - 8x + y^2 + 4y = -4 is (4, -2) -/
theorem circle_center : is_center 4 (-2) := by sorry

end NUMINAMATH_CALUDE_circle_center_l1773_177360
