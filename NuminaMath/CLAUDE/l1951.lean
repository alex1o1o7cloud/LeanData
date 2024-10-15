import Mathlib

namespace NUMINAMATH_CALUDE_piggy_bank_dime_difference_l1951_195150

theorem piggy_bank_dime_difference :
  ∀ (nickels dimes half_dollars : ℕ),
    nickels + dimes + half_dollars = 100 →
    5 * nickels + 10 * dimes + 50 * half_dollars = 1350 →
    ∃ (max_dimes min_dimes : ℕ),
      (∀ d : ℕ, (∃ n h : ℕ, n + d + h = 100 ∧ 5 * n + 10 * d + 50 * h = 1350) → d ≤ max_dimes) ∧
      (∀ d : ℕ, (∃ n h : ℕ, n + d + h = 100 ∧ 5 * n + 10 * d + 50 * h = 1350) → d ≥ min_dimes) ∧
      max_dimes - min_dimes = 162 :=
by sorry

end NUMINAMATH_CALUDE_piggy_bank_dime_difference_l1951_195150


namespace NUMINAMATH_CALUDE_circumcircle_equation_l1951_195171

-- Define the points O, A, and B
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (6, 2)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 2*y = 0

-- State the theorem
theorem circumcircle_equation :
  circle_equation O.1 O.2 ∧
  circle_equation A.1 A.2 ∧
  circle_equation B.1 B.2 ∧
  (∀ (x y : ℝ), circle_equation x y → (x - 3)^2 + (y - 1)^2 = 10) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l1951_195171


namespace NUMINAMATH_CALUDE_bianca_cupcake_sale_l1951_195105

/-- Bianca's cupcake sale problem --/
theorem bianca_cupcake_sale (initial : ℕ) (made_later : ℕ) (left_at_end : ℕ) :
  initial + made_later - left_at_end = (initial + made_later) - left_at_end :=
by
  sorry

/-- Solving Bianca's cupcake sale problem --/
def solve_bianca_cupcake_sale (initial : ℕ) (made_later : ℕ) (left_at_end : ℕ) : ℕ :=
  initial + made_later - left_at_end

#eval solve_bianca_cupcake_sale 14 17 25

end NUMINAMATH_CALUDE_bianca_cupcake_sale_l1951_195105


namespace NUMINAMATH_CALUDE_election_total_votes_l1951_195143

/-- Represents the number of votes for each candidate in the election. -/
structure ElectionResult where
  winner : ℕ
  opponent1 : ℕ
  opponent2 : ℕ
  opponent3 : ℕ
  fourth_place : ℕ

/-- Conditions of the election result. -/
def valid_election_result (e : ElectionResult) : Prop :=
  e.winner = e.opponent1 + 53 ∧
  e.winner = e.opponent2 + 79 ∧
  e.winner = e.opponent3 + 105 ∧
  e.fourth_place = 199

/-- Calculates the total votes in the election. -/
def total_votes (e : ElectionResult) : ℕ :=
  e.winner + e.opponent1 + e.opponent2 + e.opponent3 + e.fourth_place

/-- Theorem stating that the total votes in the election is 1598. -/
theorem election_total_votes :
  ∀ e : ElectionResult, valid_election_result e → total_votes e = 1598 :=
by sorry

end NUMINAMATH_CALUDE_election_total_votes_l1951_195143


namespace NUMINAMATH_CALUDE_fraction_equality_l1951_195169

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : a/b + (a+6*b)/(b+6*a) = 3) :
  a/b = (8 + Real.sqrt 46)/6 ∨ a/b = (8 - Real.sqrt 46)/6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1951_195169


namespace NUMINAMATH_CALUDE_rosie_pies_calculation_l1951_195138

-- Define the function that calculates the number of pies
def pies_from_apples (apples_per_3_pies : ℕ) (available_apples : ℕ) : ℕ :=
  (available_apples * 3) / apples_per_3_pies

-- Theorem statement
theorem rosie_pies_calculation :
  pies_from_apples 12 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_calculation_l1951_195138


namespace NUMINAMATH_CALUDE_fixed_points_bound_l1951_195197

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial n) : ℕ := n - 1

/-- Evaluation of a polynomial at a point -/
def eval (p : IntPolynomial n) (x : ℤ) : ℤ := sorry

/-- Composition of a polynomial with itself k times -/
def composeK (p : IntPolynomial n) (k : ℕ) : IntPolynomial n := sorry

/-- The number of integer solutions to the equation Q_k(t) = t -/
def numFixedPoints (p : IntPolynomial n) (k : ℕ) : ℕ := sorry

theorem fixed_points_bound (n : ℕ) (p : IntPolynomial n) (k : ℕ) :
  degree p > 1 → numFixedPoints p k ≤ degree p := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_bound_l1951_195197


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1951_195161

theorem train_speed_calculation (train_length bridge_length : Real) (crossing_time : Real) : 
  train_length = 145 ∧ bridge_length = 230 ∧ crossing_time = 30 →
  ((train_length + bridge_length) / crossing_time) * 3.6 = 45 := by
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1951_195161


namespace NUMINAMATH_CALUDE_game_probability_l1951_195170

/-- Represents the probability of winning for each player -/
structure PlayerProbabilities where
  alex : ℚ
  mel : ℚ
  chelsea : ℚ

/-- Calculates the probability of a specific outcome in the game -/
def outcome_probability (probs : PlayerProbabilities) (alex_wins mel_wins chelsea_wins : ℕ) : ℚ :=
  (probs.alex ^ alex_wins) * (probs.mel ^ mel_wins) * (probs.chelsea ^ chelsea_wins)

/-- Calculates the number of ways to arrange wins in a given number of rounds -/
def arrangements (total_rounds alex_wins mel_wins chelsea_wins : ℕ) : ℕ :=
  Nat.choose total_rounds alex_wins * Nat.choose (total_rounds - alex_wins) mel_wins

/-- The main theorem stating the probability of the specific outcome -/
theorem game_probability : ∃ (probs : PlayerProbabilities),
  probs.alex = 1/4 ∧
  probs.mel = 2 * probs.chelsea ∧
  probs.alex + probs.mel + probs.chelsea = 1 ∧
  (outcome_probability probs 2 3 3 * arrangements 8 2 3 3 : ℚ) = 35/512 := by
  sorry

end NUMINAMATH_CALUDE_game_probability_l1951_195170


namespace NUMINAMATH_CALUDE_business_profit_l1951_195147

/-- Represents a business with spending and income -/
structure Business where
  spending : ℕ
  income : ℕ

/-- Calculates the profit of a business -/
def profit (b : Business) : ℕ := b.income - b.spending

/-- Theorem stating the profit for a business with given conditions -/
theorem business_profit :
  ∀ (b : Business),
  (b.spending : ℚ) / b.income = 5 / 9 →
  b.income = 108000 →
  profit b = 48000 := by
  sorry

end NUMINAMATH_CALUDE_business_profit_l1951_195147


namespace NUMINAMATH_CALUDE_minimum_coins_for_purchase_l1951_195117

def quarter : ℕ := 25
def dime : ℕ := 10
def nickel : ℕ := 5

def candy_bar : ℕ := 45
def chewing_gum : ℕ := 35
def chocolate_bar : ℕ := 65
def juice_pack : ℕ := 70
def cookies : ℕ := 80

def total_cost : ℕ := 2 * candy_bar + 3 * chewing_gum + chocolate_bar + 2 * juice_pack + cookies

theorem minimum_coins_for_purchase :
  ∃ (q d n : ℕ), 
    q * quarter + d * dime + n * nickel = total_cost ∧ 
    q + d + n = 20 ∧ 
    q = 19 ∧ 
    d = 0 ∧ 
    n = 1 ∧
    ∀ (q' d' n' : ℕ), 
      q' * quarter + d' * dime + n' * nickel = total_cost → 
      q' + d' + n' ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_minimum_coins_for_purchase_l1951_195117


namespace NUMINAMATH_CALUDE_inverse_equals_scaled_sum_l1951_195189

/-- Given a 2x2 matrix M, prove that its inverse is equal to (1/6)*M + (1/6)*I -/
theorem inverse_equals_scaled_sum (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M = !![2, 0; 1, -3]) : 
  M⁻¹ = (1/6 : ℝ) • M + (1/6 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inverse_equals_scaled_sum_l1951_195189


namespace NUMINAMATH_CALUDE_rental_crossover_point_l1951_195153

/-- Represents the rental rates for a car agency -/
structure AgencyRates where
  dailyRate : ℝ
  mileRate : ℝ

/-- Theorem stating the crossover point for car rental agencies -/
theorem rental_crossover_point (days : ℝ) (agency1 agency2 : AgencyRates) 
  (h1 : agency1.dailyRate = 20.25)
  (h2 : agency1.mileRate = 0.14)
  (h3 : agency2.dailyRate = 18.25)
  (h4 : agency2.mileRate = 0.22)
  : ∃ m : ℝ, m = 25 * days ∧ 
    agency1.dailyRate * days + agency1.mileRate * m = agency2.dailyRate * days + agency2.mileRate * m :=
by sorry

end NUMINAMATH_CALUDE_rental_crossover_point_l1951_195153


namespace NUMINAMATH_CALUDE_circle_ratio_l1951_195131

theorem circle_ratio (r R : ℝ) (hr : r > 0) (hR : R > 0) 
  (h : π * R^2 - π * r^2 = 4 * (π * r^2)) : r / R = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l1951_195131


namespace NUMINAMATH_CALUDE_S_value_l1951_195113

/-- The sum Sₙ for n points on a line and a point off the line -/
def S (n : ℕ) (l : Set (ℝ × ℝ)) (Q : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem stating the value of Sₙ based on n -/
theorem S_value (n : ℕ) (l : Set (ℝ × ℝ)) (Q : ℝ × ℝ) 
  (h1 : n ≥ 3)
  (h2 : ∃ (p : Fin n → ℝ × ℝ), (∀ i, p i ∈ l) ∧ (∀ i j, i ≠ j → p i ≠ p j))
  (h3 : Q ∉ l) :
  S n l Q = if n = 3 then 1 else 0 :=
sorry

end NUMINAMATH_CALUDE_S_value_l1951_195113


namespace NUMINAMATH_CALUDE_no_such_function_l1951_195126

theorem no_such_function : ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, x > 0 → y > 0 → f (x + y) > f x * (1 + y * f x) := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l1951_195126


namespace NUMINAMATH_CALUDE_line_equation_sum_l1951_195104

/-- Proves that for a line with slope 8 passing through the point (-2, 4),
    if its equation is of the form y = mx + b, then m + b = 28. -/
theorem line_equation_sum (m b : ℝ) : 
  m = 8 ∧ 4 = m * (-2) + b → m + b = 28 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_sum_l1951_195104


namespace NUMINAMATH_CALUDE_equation_solution_l1951_195129

theorem equation_solution :
  ∀ x : ℚ, x ≠ 4 → ((7 * x + 2) / (x - 4) = -6 / (x - 4) ↔ x = -8 / 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1951_195129


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1951_195177

/-- A continuous function satisfying f(x) = a^x * f(x/2) for all x -/
def FunctionalEquation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  Continuous f ∧ a > 0 ∧ ∀ x, f x = a^x * f (x/2)

theorem functional_equation_solution {f : ℝ → ℝ} {a : ℝ} 
  (h : FunctionalEquation f a) : 
  ∃ C : ℝ, ∀ x, f x = C * a^(2*x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1951_195177


namespace NUMINAMATH_CALUDE_arc_length_radius_l1951_195155

/-- Given an arc length of 2.5π cm and a central angle of 75°, the radius of the circle is 6 cm. -/
theorem arc_length_radius (L : ℝ) (θ : ℝ) (R : ℝ) : 
  L = 2.5 * π ∧ θ = 75 → R = 6 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_radius_l1951_195155


namespace NUMINAMATH_CALUDE_inverse_function_solution_l1951_195173

/-- Given a function g(x) = 1 / (cx + d) where c and d are nonzero constants,
    prove that the solution to g^(-1)(x) = 2 is x = (1 - 2d) / (2c) -/
theorem inverse_function_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  let g : ℝ → ℝ := λ x => 1 / (c * x + d)
  ∃! x, g x = 2⁻¹ ∧ x = (1 - 2 * d) / (2 * c) :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_solution_l1951_195173


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1951_195176

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  firstTerm : ℚ
  commonDiff : ℚ

/-- Sum of the first n terms of an arithmetic sequence. -/
def sumFirstNTerms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.firstTerm + (n - 1 : ℚ) * seq.commonDiff)

/-- Sum of terms from index m to n (inclusive) of an arithmetic sequence. -/
def sumTermsMtoN (seq : ArithmeticSequence) (m n : ℕ) : ℚ :=
  sumFirstNTerms seq n - sumFirstNTerms seq (m - 1)

theorem arithmetic_sequence_first_term 
  (seq : ArithmeticSequence) 
  (h1 : sumFirstNTerms seq 30 = 450)
  (h2 : sumTermsMtoN seq 31 60 = 1650) : 
  seq.firstTerm = -13/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1951_195176


namespace NUMINAMATH_CALUDE_triangle_properties_l1951_195112

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_properties (t : Triangle) :
  (t.a^2 + t.b^2 < t.c^2 → π/2 < t.C) ∧
  (Real.sin t.A > Real.sin t.B → t.a > t.b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1951_195112


namespace NUMINAMATH_CALUDE_original_number_form_l1951_195124

theorem original_number_form (N : ℤ) : 
  (∃ m : ℤ, (N + 3) = 9 * m) → ∃ k : ℤ, N = 9 * k + 3 :=
by sorry

end NUMINAMATH_CALUDE_original_number_form_l1951_195124


namespace NUMINAMATH_CALUDE_expand_binomials_l1951_195106

theorem expand_binomials (x y : ℝ) : (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l1951_195106


namespace NUMINAMATH_CALUDE_passing_percentage_is_25_percent_l1951_195191

/-- The percentage of total marks needed to pass a test -/
def passing_percentage (pradeep_score : ℕ) (failed_by : ℕ) (max_marks : ℕ) : ℚ :=
  (pradeep_score + failed_by : ℚ) / max_marks * 100

/-- Theorem stating that the passing percentage is 25% given the problem conditions -/
theorem passing_percentage_is_25_percent :
  passing_percentage 185 25 840 = 25 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_is_25_percent_l1951_195191


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_specific_quadratic_equation_roots_l1951_195167

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  discriminant = 0 → ∃! x : ℝ, a*x^2 + b*x + c = 0 :=
by sorry

theorem specific_quadratic_equation_roots :
  ∃! x : ℝ, x^2 + 6*x + 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_specific_quadratic_equation_roots_l1951_195167


namespace NUMINAMATH_CALUDE_number_of_students_l1951_195198

-- Define the lottery winnings
def lottery_winnings : ℚ := 155250

-- Define the fraction given to each student
def fraction_per_student : ℚ := 1 / 1000

-- Define the total amount received by students
def total_received : ℚ := 15525

-- Theorem to prove
theorem number_of_students : 
  (total_received / (lottery_winnings * fraction_per_student) : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l1951_195198


namespace NUMINAMATH_CALUDE_missing_exponent_proof_l1951_195136

theorem missing_exponent_proof :
  (9 ^ 5.6 * 9 ^ 10.3) / 9 ^ 2.56256 = 9 ^ 13.33744 := by
  sorry

end NUMINAMATH_CALUDE_missing_exponent_proof_l1951_195136


namespace NUMINAMATH_CALUDE_hyperbola_center_l1951_195159

/-- The center of the hyperbola given by the equation (4x+8)^2/16 - (5y-5)^2/25 = 1 is (-2, 1) -/
theorem hyperbola_center : ∃ (h k : ℝ), 
  (∀ x y : ℝ, (4*x + 8)^2 / 16 - (5*y - 5)^2 / 25 = 1 ↔ 
    (x - h)^2 - (y - k)^2 = 1) ∧ 
  h = -2 ∧ k = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l1951_195159


namespace NUMINAMATH_CALUDE_no_real_solutions_for_ratio_equation_l1951_195163

theorem no_real_solutions_for_ratio_equation :
  ¬∃ (x : ℝ), (x + 3) / (2*x + 5) = (5*x + 4) / (8*x + 5) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_ratio_equation_l1951_195163


namespace NUMINAMATH_CALUDE_product_and_gcd_conditions_l1951_195101

theorem product_and_gcd_conditions (a b : ℕ+) : 
  a * b = 864 ∧ Nat.gcd a b = 6 ↔ (a = 6 ∧ b = 144) ∨ (a = 144 ∧ b = 6) ∨ (a = 18 ∧ b = 48) ∨ (a = 48 ∧ b = 18) := by
  sorry

end NUMINAMATH_CALUDE_product_and_gcd_conditions_l1951_195101


namespace NUMINAMATH_CALUDE_max_sum_of_products_l1951_195196

theorem max_sum_of_products (f g h j : ℕ) : 
  f ∈ ({6, 7, 8, 9} : Set ℕ) → 
  g ∈ ({6, 7, 8, 9} : Set ℕ) → 
  h ∈ ({6, 7, 8, 9} : Set ℕ) → 
  j ∈ ({6, 7, 8, 9} : Set ℕ) → 
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j → 
  f * g + g * h + h * j + f * j ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_products_l1951_195196


namespace NUMINAMATH_CALUDE_two_digit_sum_theorem_l1951_195146

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : tens ≤ 9 ∧ ones ≤ 9

/-- The sum of five identical two-digit numbers equals another two-digit number -/
def sum_property (ab mb : TwoDigitNumber) : Prop :=
  5 * (10 * ab.tens + ab.ones) = 10 * mb.tens + mb.ones

/-- Different letters represent different digits -/
def different_digits (ab mb : TwoDigitNumber) : Prop :=
  ab.tens ≠ ab.ones ∧ 
  (ab.tens ≠ mb.tens ∨ ab.ones ≠ mb.ones)

theorem two_digit_sum_theorem (ab mb : TwoDigitNumber) 
  (h_sum : sum_property ab mb) 
  (h_diff : different_digits ab mb) : 
  (ab.tens = 1 ∧ ab.ones = 0) ∨ (ab.tens = 1 ∧ ab.ones = 5) :=
sorry

end NUMINAMATH_CALUDE_two_digit_sum_theorem_l1951_195146


namespace NUMINAMATH_CALUDE_measure_11_grams_l1951_195160

/-- Represents the number of ways to measure a weight using given weights -/
def measure_ways (one_gram : ℕ) (two_gram : ℕ) (four_gram : ℕ) (target : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 4 ways to measure 11 grams
    given three 1-gram weights, four 2-gram weights, and two 4-gram weights -/
theorem measure_11_grams :
  measure_ways 3 4 2 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_measure_11_grams_l1951_195160


namespace NUMINAMATH_CALUDE_bus_capacity_problem_l1951_195145

theorem bus_capacity_problem (capacity : ℕ) (first_trip_fraction : ℚ) (total_people : ℕ) 
  (h1 : capacity = 200)
  (h2 : first_trip_fraction = 3 / 4)
  (h3 : total_people = 310) :
  ∃ (return_trip_fraction : ℚ), 
    (first_trip_fraction * capacity + return_trip_fraction * capacity = total_people) ∧
    return_trip_fraction = 4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_bus_capacity_problem_l1951_195145


namespace NUMINAMATH_CALUDE_log_equation_solution_l1951_195168

theorem log_equation_solution (y : ℝ) (h : y > 0) : 
  Real.log y^3 / Real.log 3 + Real.log y / Real.log (1/3) = 6 → y = 27 := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1951_195168


namespace NUMINAMATH_CALUDE_number_difference_l1951_195195

theorem number_difference (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : |x - y| = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1951_195195


namespace NUMINAMATH_CALUDE_number_problem_l1951_195156

theorem number_problem (x : ℝ) : 
  (1.5 * x) / 7 = 271.07142857142856 → x = 1265 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1951_195156


namespace NUMINAMATH_CALUDE_exact_arrival_speed_l1951_195123

theorem exact_arrival_speed (d : ℝ) (t : ℝ) (h1 : d = 30 * (t + 1/30)) (h2 : d = 50 * (t - 1/30)) :
  d / t = 37.5 := by sorry

end NUMINAMATH_CALUDE_exact_arrival_speed_l1951_195123


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1951_195184

/-- Given two positive integers with specific LCM and HCF, prove that if one number is 385, the other is 180 -/
theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 385 →
  B = 180 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1951_195184


namespace NUMINAMATH_CALUDE_absolute_value_fraction_inequality_l1951_195199

theorem absolute_value_fraction_inequality (x : ℝ) :
  x ≠ 0 → (|(x + 2) / x| < 1 ↔ x < -1) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_inequality_l1951_195199


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_l1951_195115

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the non-coincidence relation for lines
variable (non_coincident_lines : Line → Line → Prop)

-- Define the non-coincidence relation for planes
variable (non_coincident_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_implies_parallel 
  (m n : Line) (α β : Plane)
  (h1 : non_coincident_lines m n)
  (h2 : non_coincident_planes α β)
  (h3 : perpendicular m α)
  (h4 : perpendicular m β) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_l1951_195115


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l1951_195130

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 6) ((Nat.factorial 9) / (Nat.factorial 4)) = 720 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l1951_195130


namespace NUMINAMATH_CALUDE_ab_value_l1951_195154

theorem ab_value (a b : ℤ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1951_195154


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l1951_195144

theorem smallest_constant_inequality (C : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ↔ C ≤ Real.sqrt (4/3) :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l1951_195144


namespace NUMINAMATH_CALUDE_linear_regression_center_point_l1951_195142

/-- Given a linear regression equation y = 0.2x - m with the center of sample points at (m, 1.6), prove that m = -2 -/
theorem linear_regression_center_point (m : ℝ) : 
  (∀ x y : ℝ, y = 0.2 * x - m) → -- Linear regression equation
  (m, 1.6) = (m, 0.2 * m - m) → -- Center of sample points
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_linear_regression_center_point_l1951_195142


namespace NUMINAMATH_CALUDE_nine_n_sum_of_squares_nine_n_sum_of_squares_not_div_by_three_l1951_195125

theorem nine_n_sum_of_squares (n a b c : ℕ) (h : n = a^2 + b^2 + c^2) :
  ∃ (p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ : ℤ), 
    (p₁ ≠ 0 ∧ q₁ ≠ 0 ∧ r₁ ≠ 0 ∧ p₂ ≠ 0 ∧ q₂ ≠ 0 ∧ r₂ ≠ 0 ∧ p₃ ≠ 0 ∧ q₃ ≠ 0 ∧ r₃ ≠ 0) ∧
    (9 * n = (p₁ * a + q₁ * b + r₁ * c)^2 + (p₂ * a + q₂ * b + r₂ * c)^2 + (p₃ * a + q₃ * b + r₃ * c)^2) :=
sorry

theorem nine_n_sum_of_squares_not_div_by_three (n a b c : ℕ) (h₁ : n = a^2 + b^2 + c^2) 
  (h₂ : ¬(3 ∣ a) ∨ ¬(3 ∣ b) ∨ ¬(3 ∣ c)) :
  ∃ (x y z : ℕ), 
    (¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z)) ∧
    (9 * n = x^2 + y^2 + z^2) :=
sorry

end NUMINAMATH_CALUDE_nine_n_sum_of_squares_nine_n_sum_of_squares_not_div_by_three_l1951_195125


namespace NUMINAMATH_CALUDE_part_one_part_two_l1951_195164

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 3}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- Theorem for part 1
theorem part_one : (Set.univ \ A 1) ∩ B = {x | -1 ≤ x ∧ x < 0} := by sorry

-- Theorem for part 2
theorem part_two (a : ℝ) : A a ⊆ B ↔ a < -4 ∨ (0 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1951_195164


namespace NUMINAMATH_CALUDE_edward_final_earnings_l1951_195107

/-- Edward's lawn mowing business earnings and expenses --/
def edward_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

/-- Theorem: Edward's final earnings --/
theorem edward_final_earnings :
  edward_business 2 27 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_edward_final_earnings_l1951_195107


namespace NUMINAMATH_CALUDE_stationary_train_length_is_1296_l1951_195152

/-- The length of a stationary train given the time it takes for another train to pass it. -/
def stationary_train_length (time_to_pass_pole : ℝ) (time_to_cross_stationary : ℝ) (train_speed : ℝ) : ℝ :=
  train_speed * time_to_cross_stationary - train_speed * time_to_pass_pole

/-- Theorem stating that the length of the stationary train is 1296 meters under the given conditions. -/
theorem stationary_train_length_is_1296 :
  stationary_train_length 5 25 64.8 = 1296 := by
  sorry

#eval stationary_train_length 5 25 64.8

end NUMINAMATH_CALUDE_stationary_train_length_is_1296_l1951_195152


namespace NUMINAMATH_CALUDE_negation_forall_geq_zero_equivalent_exists_lt_zero_l1951_195103

theorem negation_forall_geq_zero_equivalent_exists_lt_zero :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_forall_geq_zero_equivalent_exists_lt_zero_l1951_195103


namespace NUMINAMATH_CALUDE_molecular_weight_CuCO3_l1951_195186

/-- The atomic weight of Copper in g/mol -/
def Cu_weight : ℝ := 63.55

/-- The atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The molecular weight of CuCO3 in g/mol -/
def CuCO3_weight : ℝ := Cu_weight + C_weight + 3 * O_weight

/-- The number of moles of CuCO3 -/
def moles : ℝ := 8

/-- Theorem: The molecular weight of 8 moles of CuCO3 is 988.48 grams -/
theorem molecular_weight_CuCO3 : moles * CuCO3_weight = 988.48 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_CuCO3_l1951_195186


namespace NUMINAMATH_CALUDE_cakes_served_during_lunch_l1951_195137

theorem cakes_served_during_lunch :
  ∀ (lunch_cakes dinner_cakes : ℕ),
    dinner_cakes = 9 →
    dinner_cakes = lunch_cakes + 3 →
    lunch_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_during_lunch_l1951_195137


namespace NUMINAMATH_CALUDE_set_intersection_and_subset_l1951_195116

def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}

def B (a : ℝ) : Set ℝ := {x | (x - a) / (x - (a^2 + 1)) < 0}

theorem set_intersection_and_subset (a : ℝ) :
  (a = 2 → A a ∩ B a = {x | 2 < x ∧ x < 5}) ∧
  (B a ⊆ A a ↔ a ∈ Set.Icc (-1) (-1/2) ∪ Set.Icc 2 3) :=
sorry

end NUMINAMATH_CALUDE_set_intersection_and_subset_l1951_195116


namespace NUMINAMATH_CALUDE_max_distance_is_217_12_l1951_195162

-- Define the constants
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def total_gallons : ℝ := 23

-- Define the percentages for regular and peak traffic
def regular_highway_percent : ℝ := 0.4
def regular_city_percent : ℝ := 0.6
def peak_highway_percent : ℝ := 0.25
def peak_city_percent : ℝ := 0.75

-- Calculate distances for regular and peak traffic
def regular_distance : ℝ := 
  (regular_highway_percent * total_gallons * highway_mpg) + 
  (regular_city_percent * total_gallons * city_mpg)

def peak_distance : ℝ := 
  (peak_highway_percent * total_gallons * highway_mpg) + 
  (peak_city_percent * total_gallons * city_mpg)

-- Theorem to prove
theorem max_distance_is_217_12 : 
  max regular_distance peak_distance = 217.12 := by sorry

end NUMINAMATH_CALUDE_max_distance_is_217_12_l1951_195162


namespace NUMINAMATH_CALUDE_f_prime_zero_l1951_195165

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x) * Real.exp x

theorem f_prime_zero : (deriv f) 0 = -2 := by sorry

end NUMINAMATH_CALUDE_f_prime_zero_l1951_195165


namespace NUMINAMATH_CALUDE_women_in_luxury_class_l1951_195149

theorem women_in_luxury_class 
  (total_passengers : ℕ) 
  (women_percentage : ℚ) 
  (luxury_class_percentage : ℚ) 
  (h1 : total_passengers = 300) 
  (h2 : women_percentage = 80 / 100) 
  (h3 : luxury_class_percentage = 15 / 100) : 
  ℕ := by
  sorry

#check women_in_luxury_class

end NUMINAMATH_CALUDE_women_in_luxury_class_l1951_195149


namespace NUMINAMATH_CALUDE_max_value_expression_l1951_195179

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + 2 * b^2)) ≤ a^2 + 3 * b^2) ∧
  (∃ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + 2 * b^2)) = a^2 + 3 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1951_195179


namespace NUMINAMATH_CALUDE_multiple_of_reciprocal_l1951_195187

theorem multiple_of_reciprocal (x : ℝ) (m : ℝ) (h1 : x > 0) (h2 : x + 17 = m * (1 / x)) (h3 : x = 3) : m = 60 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_reciprocal_l1951_195187


namespace NUMINAMATH_CALUDE_hexadecagon_diagonals_l1951_195119

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexadecagon is a 16-sided polygon -/
def hexadecagon_sides : ℕ := 16

theorem hexadecagon_diagonals :
  num_diagonals hexadecagon_sides = 104 := by
  sorry

end NUMINAMATH_CALUDE_hexadecagon_diagonals_l1951_195119


namespace NUMINAMATH_CALUDE_remainder_b39_mod_125_l1951_195158

def reverse_concatenate (n : ℕ) : ℕ :=
  -- Definition of b_n
  sorry

theorem remainder_b39_mod_125 : reverse_concatenate 39 % 125 = 21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_b39_mod_125_l1951_195158


namespace NUMINAMATH_CALUDE_greatest_solution_is_negative_two_l1951_195190

def equation (x : ℝ) : Prop :=
  x ≠ 9 ∧ (x^2 - 5*x - 84) / (x - 9) = 4 / (x + 6)

theorem greatest_solution_is_negative_two :
  ∃ x_max : ℝ, x_max = -2 ∧ equation x_max ∧ ∀ y : ℝ, equation y → y ≤ x_max :=
sorry

end NUMINAMATH_CALUDE_greatest_solution_is_negative_two_l1951_195190


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1951_195100

/-- Given a line segment with one endpoint at (3, 4) and midpoint at (5, -8),
    the sum of the coordinates of the other endpoint is -13. -/
theorem endpoint_coordinate_sum :
  let a : ℝ × ℝ := (3, 4)  -- First endpoint
  let m : ℝ × ℝ := (5, -8) -- Midpoint
  let b : ℝ × ℝ := (2 * m.1 - a.1, 2 * m.2 - a.2) -- Other endpoint
  b.1 + b.2 = -13 := by sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1951_195100


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l1951_195132

theorem semicircle_perimeter (r : ℝ) (h : r = 2.1) : 
  let perimeter := π * r + 2 * r
  perimeter = π * 2.1 + 4.2 := by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l1951_195132


namespace NUMINAMATH_CALUDE_oscillating_cosine_shift_l1951_195182

theorem oscillating_cosine_shift (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →
  (∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) →
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_oscillating_cosine_shift_l1951_195182


namespace NUMINAMATH_CALUDE_sequence_property_l1951_195127

/-- Given a sequence a and its partial sum S, prove that a_n = 2^n + n for all n ∈ ℕ⁺ -/
theorem sequence_property (a : ℕ+ → ℕ) (S : ℕ+ → ℕ) 
  (h : ∀ n : ℕ+, 2 * S n = 4 * a n + (n - 4) * (n + 1)) :
  ∀ n : ℕ+, a n = 2^(n : ℕ) + n := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l1951_195127


namespace NUMINAMATH_CALUDE_binomial_square_constant_l1951_195122

theorem binomial_square_constant (a : ℚ) : 
  (∃ b c : ℚ, ∀ x, 9*x^2 + 27*x + a = (b*x + c)^2) → a = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l1951_195122


namespace NUMINAMATH_CALUDE_park_trees_after_planting_l1951_195181

/-- The number of walnut trees in the park after planting -/
def total_trees (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem stating that the total number of walnut trees after planting is 211 -/
theorem park_trees_after_planting :
  total_trees 107 104 = 211 := by
  sorry

end NUMINAMATH_CALUDE_park_trees_after_planting_l1951_195181


namespace NUMINAMATH_CALUDE_sqrt_30_simplest_l1951_195141

/-- Predicate to check if a number is a perfect square --/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Predicate to check if a square root is in its simplest form --/
def IsSimplestSquareRoot (n : ℝ) : Prop :=
  ∃ m : ℕ, n = Real.sqrt m ∧ m > 0 ∧ ¬∃ k : ℕ, k > 1 ∧ IsPerfectSquare k ∧ k ∣ m

/-- Theorem stating that √30 is the simplest square root among the given options --/
theorem sqrt_30_simplest :
  IsSimplestSquareRoot (Real.sqrt 30) ∧
  ¬IsSimplestSquareRoot (Real.sqrt 0.1) ∧
  ¬IsSimplestSquareRoot (1/2 : ℝ) ∧
  ¬IsSimplestSquareRoot (Real.sqrt 18) :=
by sorry


end NUMINAMATH_CALUDE_sqrt_30_simplest_l1951_195141


namespace NUMINAMATH_CALUDE_cubic_system_solution_method_l1951_195188

/-- A cubic polynomial -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

/-- The statement of the theorem -/
theorem cubic_system_solution_method
  (a b c d : ℝ) (p : ℝ → ℝ) (hp : p = CubicPolynomial a b c d) :
  ∃ (cubic : ℝ → ℝ) (quadratic : ℝ → ℝ),
    (∀ x y : ℝ, x = p y ∧ y = p x ↔ 
      (cubic x = 0 ∧ quadratic y = 0) ∨ 
      (cubic y = 0 ∧ quadratic x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_system_solution_method_l1951_195188


namespace NUMINAMATH_CALUDE_sum_of_roots_symmetric_function_l1951_195157

/-- A function g: ℝ → ℝ that satisfies g(3+x) = g(3-x) for all real x -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The theorem stating that if g is symmetric about 3 and has exactly four distinct real roots,
    then the sum of these roots is 12 -/
theorem sum_of_roots_symmetric_function
  (g : ℝ → ℝ) 
  (h_sym : SymmetricAboutThree g)
  (h_roots : ∃! (s₁ s₂ s₃ s₄ : ℝ), s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₁ ≠ s₄ ∧ s₂ ≠ s₃ ∧ s₂ ≠ s₄ ∧ s₃ ≠ s₄ ∧ 
              g s₁ = 0 ∧ g s₂ = 0 ∧ g s₃ = 0 ∧ g s₄ = 0) :
  ∃ (s₁ s₂ s₃ s₄ : ℝ), g s₁ = 0 ∧ g s₂ = 0 ∧ g s₃ = 0 ∧ g s₄ = 0 ∧ s₁ + s₂ + s₃ + s₄ = 12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_symmetric_function_l1951_195157


namespace NUMINAMATH_CALUDE_grade_distribution_l1951_195180

theorem grade_distribution (total_students : ℕ) 
  (prob_A prob_B prob_C prob_D : ℝ) 
  (h1 : prob_A = 0.6 * prob_B)
  (h2 : prob_C = 1.3 * prob_B)
  (h3 : prob_D = 0.8 * prob_B)
  (h4 : prob_A + prob_B + prob_C + prob_D = 1)
  (h5 : total_students = 50) :
  ∃ (num_B : ℕ), num_B = 14 ∧ 
    (↑num_B : ℝ) / total_students = prob_B := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l1951_195180


namespace NUMINAMATH_CALUDE_total_carvings_eq_56_l1951_195175

/-- The number of wood carvings that can be contained in each shelf -/
def carvings_per_shelf : ℕ := 8

/-- The number of shelves filled with carvings -/
def filled_shelves : ℕ := 7

/-- The total number of wood carvings displayed -/
def total_carvings : ℕ := carvings_per_shelf * filled_shelves

theorem total_carvings_eq_56 : total_carvings = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_carvings_eq_56_l1951_195175


namespace NUMINAMATH_CALUDE_problem_statement_l1951_195183

theorem problem_statement (A B C D : ℤ) 
  (h1 : A - B = 30) 
  (h2 : C + D = 20) : 
  (B + C) - (A - D) = -10 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1951_195183


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1951_195185

-- Part 1
theorem simplify_expression_1 (x : ℝ) (hx : x ≠ 0) :
  5 * x^2 * x^4 + x^8 / (-x)^2 = 6 * x^6 :=
sorry

-- Part 2
theorem simplify_expression_2 (x y : ℝ) (hx : x ≠ 0) :
  ((x + y)^2 - (x - y)^2) / (2 * x) = 2 * y :=
sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1951_195185


namespace NUMINAMATH_CALUDE_passengers_in_buses_l1951_195135

/-- Given that 456 passengers fit into 12 buses, 
    prove that 266 passengers fit into 7 buses. -/
theorem passengers_in_buses 
  (total_passengers : ℕ) 
  (total_buses : ℕ) 
  (target_buses : ℕ) 
  (h1 : total_passengers = 456) 
  (h2 : total_buses = 12) 
  (h3 : target_buses = 7) :
  (total_passengers / total_buses) * target_buses = 266 := by
  sorry

end NUMINAMATH_CALUDE_passengers_in_buses_l1951_195135


namespace NUMINAMATH_CALUDE_white_lights_replacement_l1951_195121

/-- The number of white lights Malcolm had initially --/
def total_white_lights : ℕ := by sorry

/-- The number of red lights initially purchased --/
def initial_red_lights : ℕ := 16

/-- The number of yellow lights purchased --/
def yellow_lights : ℕ := 4

/-- The number of blue lights initially purchased --/
def initial_blue_lights : ℕ := 2 * yellow_lights

/-- The number of green lights purchased --/
def green_lights : ℕ := 8

/-- The number of purple lights purchased --/
def purple_lights : ℕ := 3

/-- The additional number of red lights needed --/
def additional_red_lights : ℕ := 10

/-- The additional number of blue lights needed --/
def additional_blue_lights : ℕ := initial_blue_lights / 4

theorem white_lights_replacement :
  total_white_lights = 
    initial_red_lights + additional_red_lights +
    yellow_lights +
    initial_blue_lights + additional_blue_lights +
    green_lights +
    purple_lights := by sorry

end NUMINAMATH_CALUDE_white_lights_replacement_l1951_195121


namespace NUMINAMATH_CALUDE_parallelepiped_edge_length_l1951_195114

/-- A rectangular parallelepiped constructed from unit cubes -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ
  volume : ℕ
  edge_min : ℕ
  total_cubes : ℕ

/-- The total length of all edges of a rectangular parallelepiped -/
def total_edge_length (p : Parallelepiped) : ℕ :=
  4 * (p.length + p.width + p.height)

/-- Theorem: The total edge length of the specific parallelepiped is 96 cm -/
theorem parallelepiped_edge_length :
  ∀ (p : Parallelepiped),
    p.volume = p.length * p.width * p.height →
    p.total_cubes = 440 →
    p.edge_min = 5 →
    p.length ≥ p.edge_min →
    p.width ≥ p.edge_min →
    p.height ≥ p.edge_min →
    total_edge_length p = 96 := by
  sorry

#check parallelepiped_edge_length

end NUMINAMATH_CALUDE_parallelepiped_edge_length_l1951_195114


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1951_195140

/-- Two real numbers are inversely proportional if their product is constant. -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_problem (x y : ℝ → ℝ) 
  (h1 : InverselyProportional x y)
  (h2 : x 1 = 36 ∧ y 1 = 4)
  (h3 : y 2 = 12) :
  x 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1951_195140


namespace NUMINAMATH_CALUDE_polo_shirt_price_l1951_195151

/-- The regular price of a polo shirt -/
def regular_price : ℝ := 50

/-- The number of polo shirts purchased -/
def num_shirts : ℕ := 2

/-- The discount percentage on the shirts -/
def discount_percent : ℝ := 40

/-- The total amount paid for the shirts -/
def total_paid : ℝ := 60

/-- Theorem stating that the regular price of each polo shirt is $50 -/
theorem polo_shirt_price :
  regular_price = 50 ∧
  num_shirts * regular_price * (1 - discount_percent / 100) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_polo_shirt_price_l1951_195151


namespace NUMINAMATH_CALUDE_sugar_profit_theorem_l1951_195110

/-- Represents the profit calculation for a sugar trader --/
def sugar_profit (total_quantity : ℝ) (quantity_at_unknown_profit : ℝ) (known_profit : ℝ) (overall_profit : ℝ) : Prop :=
  let quantity_at_known_profit := total_quantity - quantity_at_unknown_profit
  let unknown_profit := (overall_profit * total_quantity - known_profit * quantity_at_known_profit) / quantity_at_unknown_profit
  unknown_profit = 12

/-- Theorem stating the profit percentage on the rest of the sugar --/
theorem sugar_profit_theorem :
  sugar_profit 1600 1200 8 11 := by
  sorry

end NUMINAMATH_CALUDE_sugar_profit_theorem_l1951_195110


namespace NUMINAMATH_CALUDE_parabola_vertex_l1951_195193

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -(x - 1)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 3)

/-- Theorem: The vertex of the parabola y = -(x-1)^2 + 3 is (1, 3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≤ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1951_195193


namespace NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_when_product_negative_l1951_195133

theorem abs_sum_lt_sum_abs_when_product_negative (a b : ℝ) :
  a * b < 0 → |a + b| < |a| + |b| := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_when_product_negative_l1951_195133


namespace NUMINAMATH_CALUDE_deluxe_premium_time_fraction_l1951_195174

/-- Represents the production details of stereos by Company S -/
structure StereoProduction where
  basicFraction : ℚ
  deluxeFraction : ℚ
  premiumFraction : ℚ
  deluxeTimeFactor : ℚ
  premiumTimeFactor : ℚ

/-- Calculates the fraction of total production time spent on deluxe and premium stereos -/
def deluxePremiumTimeFraction (prod : StereoProduction) : ℚ :=
  let totalTime := prod.basicFraction + prod.deluxeFraction * prod.deluxeTimeFactor + 
                   prod.premiumFraction * prod.premiumTimeFactor
  let deluxePremiumTime := prod.deluxeFraction * prod.deluxeTimeFactor + 
                           prod.premiumFraction * prod.premiumTimeFactor
  deluxePremiumTime / totalTime

/-- Theorem stating that the fraction of time spent on deluxe and premium stereos is 123/163 -/
theorem deluxe_premium_time_fraction :
  let prod : StereoProduction := {
    basicFraction := 2/5,
    deluxeFraction := 3/10,
    premiumFraction := 1 - 2/5 - 3/10,
    deluxeTimeFactor := 8/5,
    premiumTimeFactor := 5/2
  }
  deluxePremiumTimeFraction prod = 123/163 := by sorry

end NUMINAMATH_CALUDE_deluxe_premium_time_fraction_l1951_195174


namespace NUMINAMATH_CALUDE_z_max_min_difference_l1951_195111

theorem z_max_min_difference (x y z : ℝ) 
  (sum_eq : x + y + z = 5)
  (sum_squares_eq : x^2 + y^2 + z^2 = 20)
  (xy_eq : x * y = 2) : 
  ∃ (z_max z_min : ℝ), 
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 20 ∧ x' * y' = 2) → z' ≤ z_max) ∧
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 20 ∧ x' * y' = 2) → z' ≥ z_min) ∧
    z_max - z_min = 6 :=
by sorry

end NUMINAMATH_CALUDE_z_max_min_difference_l1951_195111


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l1951_195192

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x ≠ 2 ∧ (x^3 - 3*x^2)/(x^2 - 4*x + 4) + x = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l1951_195192


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1951_195172

theorem polynomial_expansion (x : ℝ) : 
  (1 + x^4) * (1 - x^5) * (1 + x^7) = 
  1 + x^4 - x^5 + x^7 + x^11 - x^9 - x^12 - x^16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1951_195172


namespace NUMINAMATH_CALUDE_lights_after_2011_toggles_l1951_195120

/-- Represents the state of a light (on or off) -/
inductive LightState
| On : LightState
| Off : LightState

/-- Represents a row of 7 lights -/
def LightRow := Fin 7 → LightState

def initialState : LightRow := fun i =>
  if i = 0 ∨ i = 2 ∨ i = 4 ∨ i = 6 then LightState.On else LightState.Off

def toggleLights : LightRow → LightRow := sorry

theorem lights_after_2011_toggles (initialState : LightRow) 
  (h1 : ∀ state, (toggleLights^[14]) state = state)
  (h2 : (toggleLights^[9]) initialState 0 = LightState.On ∧ 
        (toggleLights^[9]) initialState 3 = LightState.On ∧ 
        (toggleLights^[9]) initialState 5 = LightState.On) :
  (toggleLights^[2011]) initialState 0 = LightState.On ∧
  (toggleLights^[2011]) initialState 3 = LightState.On ∧
  (toggleLights^[2011]) initialState 5 = LightState.On :=
sorry

end NUMINAMATH_CALUDE_lights_after_2011_toggles_l1951_195120


namespace NUMINAMATH_CALUDE_vacation_cost_difference_l1951_195108

theorem vacation_cost_difference (total_cost : ℕ) (initial_people : ℕ) (new_people : ℕ) : 
  total_cost = 375 → initial_people = 3 → new_people = 5 → 
  (total_cost / initial_people) - (total_cost / new_people) = 50 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_difference_l1951_195108


namespace NUMINAMATH_CALUDE_birthday_cake_candles_l1951_195134

theorem birthday_cake_candles (total : ℕ) (yellow : ℕ) (red : ℕ) (blue : ℕ) : 
  total = 79 →
  yellow = 27 →
  red = 14 →
  blue = total - yellow - red →
  blue = 38 := by
sorry

end NUMINAMATH_CALUDE_birthday_cake_candles_l1951_195134


namespace NUMINAMATH_CALUDE_divisibility_problem_l1951_195102

theorem divisibility_problem (n : ℤ) : 
  n > 101 →
  n % 101 = 0 →
  (∀ d : ℤ, d ∣ n → 1 < d → d < n → ∃ k m : ℤ, k ∣ n ∧ m ∣ n ∧ d = k - m) →
  n % 100 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1951_195102


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1951_195109

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1951_195109


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1951_195194

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    and a point P on its right branch,
    a line through P intersects the asymptotes at A and B,
    where A is in the first quadrant and B is in the fourth quadrant,
    O is the origin, AP = (1/2)PB, and the area of triangle AOB is 2b,
    prove that the length of the real axis of C is 32/9. -/
theorem hyperbola_real_axis_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0)
  (P A B : ℝ × ℝ)
  (hC : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (hP : P.1 > 0)
  (hA : A.1 > 0 ∧ A.2 > 0)
  (hB : B.1 > 0 ∧ B.2 < 0)
  (hAP : A - P = (1/2) • (P - B))
  (hAOB : abs ((A.1 * B.2 - A.2 * B.1) / 2) = 2 * b) :
  2 * a = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1951_195194


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1951_195166

theorem perpendicular_lines_a_value (a : ℝ) :
  let l1 : ℝ → ℝ → Prop := λ x y => a * x - y - 2 = 0
  let l2 : ℝ → ℝ → Prop := λ x y => (a + 2) * x - y + 1 = 0
  (∀ x1 y1 x2 y2, l1 x1 y1 → l2 x2 y2 → (x2 - x1) * (y2 - y1) = 0) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1951_195166


namespace NUMINAMATH_CALUDE_f_has_maximum_l1951_195128

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

theorem f_has_maximum : ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 28/3 := by
  sorry

end NUMINAMATH_CALUDE_f_has_maximum_l1951_195128


namespace NUMINAMATH_CALUDE_touchdown_points_l1951_195148

theorem touchdown_points (total_points : ℕ) (num_touchdowns : ℕ) (points_per_touchdown : ℕ) :
  total_points = 21 →
  num_touchdowns = 3 →
  total_points = num_touchdowns * points_per_touchdown →
  points_per_touchdown = 7 := by
  sorry

end NUMINAMATH_CALUDE_touchdown_points_l1951_195148


namespace NUMINAMATH_CALUDE_largest_angle_of_pentagon_l1951_195178

/-- Represents the measures of interior angles of a convex pentagon --/
structure PentagonAngles where
  x : ℝ
  angle1 : ℝ := x - 3
  angle2 : ℝ := x - 2
  angle3 : ℝ := x - 1
  angle4 : ℝ := x
  angle5 : ℝ := x + 1

/-- The sum of interior angles of a pentagon is 540° --/
def sumOfPentagonAngles : ℝ := 540

theorem largest_angle_of_pentagon (p : PentagonAngles) :
  p.angle1 + p.angle2 + p.angle3 + p.angle4 + p.angle5 = sumOfPentagonAngles →
  p.angle5 = 110 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_of_pentagon_l1951_195178


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l1951_195139

theorem multiply_mixed_number : 9 * (7 + 2/5) = 66 + 3/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l1951_195139


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l1951_195118

theorem roots_sum_of_squares (a b c : ℝ) (r s : ℝ) : 
  r^2 - (a+b)*r + ab + c = 0 → 
  s^2 - (a+b)*s + ab + c = 0 → 
  r^2 + s^2 = a^2 + b^2 - 2*c := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l1951_195118
