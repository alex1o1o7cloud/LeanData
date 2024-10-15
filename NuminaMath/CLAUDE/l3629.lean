import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l3629_362905

theorem equation_solution : ∃ x : ℝ, (10 - 2 * x = 16) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3629_362905


namespace NUMINAMATH_CALUDE_shoes_lost_example_l3629_362994

/-- Given an initial number of shoe pairs and a maximum number of remaining pairs,
    calculate the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (max_remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * max_remaining_pairs

/-- Theorem: Given 27 initial pairs of shoes and 22 maximum remaining pairs,
    the number of individual shoes lost is 10. -/
theorem shoes_lost_example : shoes_lost 27 22 = 10 := by
  sorry

end NUMINAMATH_CALUDE_shoes_lost_example_l3629_362994


namespace NUMINAMATH_CALUDE_mary_paper_problem_l3629_362984

/-- Represents the initial state of Mary's paper pieces -/
structure InitialState where
  squares : ℕ
  triangles : ℕ
  total_pieces : ℕ
  total_pieces_eq : squares + triangles = total_pieces

/-- Represents the final state after cutting some squares -/
structure FinalState where
  initial : InitialState
  squares_cut : ℕ
  total_vertices : ℕ
  squares_cut_constraint : squares_cut ≤ initial.squares

theorem mary_paper_problem (state : InitialState) (final : FinalState)
  (h_initial_pieces : state.total_pieces = 10)
  (h_squares_cut : final.squares_cut = 3)
  (h_final_pieces : state.total_pieces + final.squares_cut = 13)
  (h_final_vertices : final.total_vertices = 42)
  : state.triangles = 4 := by
  sorry

end NUMINAMATH_CALUDE_mary_paper_problem_l3629_362984


namespace NUMINAMATH_CALUDE_quadratic_equation_property_l3629_362978

/-- 
A quadratic equation with coefficients a, b, and c, where a ≠ 0,
satisfying a + b + c = 0 and having two equal real roots.
-/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  sum_zero : a + b + c = 0
  equal_roots : ∃ x : ℝ, ∀ y : ℝ, a * y^2 + b * y + c = 0 ↔ y = x

theorem quadratic_equation_property (eq : QuadraticEquation) : eq.a = eq.c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_property_l3629_362978


namespace NUMINAMATH_CALUDE_find_fifth_month_sale_l3629_362915

def sales_problem (m1 m2 m3 m4 m6 avg : ℕ) : Prop :=
  ∃ m5 : ℕ, 
    (m1 + m2 + m3 + m4 + m5 + m6) / 6 = avg ∧
    m5 = 7560

theorem find_fifth_month_sale (m1 m2 m3 m4 m6 avg : ℕ) 
  (h1 : m1 = 7435)
  (h2 : m2 = 7920)
  (h3 : m3 = 7855)
  (h4 : m4 = 8230)
  (h5 : m6 = 6000)
  (h6 : avg = 7500) :
  sales_problem m1 m2 m3 m4 m6 avg :=
by
  sorry

end NUMINAMATH_CALUDE_find_fifth_month_sale_l3629_362915


namespace NUMINAMATH_CALUDE_problem_solution_l3629_362922

theorem problem_solution (n x y : ℝ) 
  (h1 : x = 4 * n)
  (h2 : y = x / 2)
  (h3 : 2 * n + 3 = 0.20 * 25)
  (h4 : y^3 - 4 = (1/3) * x) :
  y = (16/3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3629_362922


namespace NUMINAMATH_CALUDE_sum_of_fraction_is_correct_l3629_362970

/-- The repeating decimal 0.̅14 as a real number -/
def repeating_decimal : ℚ := 14 / 99

/-- The sum of numerator and denominator of the fraction representation of 0.̅14 -/
def sum_of_fraction : ℕ := 113

/-- Theorem stating that the sum of numerator and denominator of 0.̅14 in lowest terms is 113 -/
theorem sum_of_fraction_is_correct : 
  ∃ (n d : ℕ), d ≠ 0 ∧ repeating_decimal = n / d ∧ Nat.gcd n d = 1 ∧ n + d = sum_of_fraction := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fraction_is_correct_l3629_362970


namespace NUMINAMATH_CALUDE_operation_sequence_l3629_362956

theorem operation_sequence (x : ℝ) : ((x / 4 + 3) * 7 - 5 = 30) ↔ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_operation_sequence_l3629_362956


namespace NUMINAMATH_CALUDE_gcd_9013_4357_l3629_362902

theorem gcd_9013_4357 : Nat.gcd 9013 4357 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9013_4357_l3629_362902


namespace NUMINAMATH_CALUDE_x_value_when_z_is_64_l3629_362940

/-- Given that x is directly proportional to y², y is inversely proportional to √z,
    and x = 4 when z = 16, prove that x = 1 when z = 64. -/
theorem x_value_when_z_is_64 
  (k : ℝ) (n : ℝ) -- Constants of proportionality
  (h1 : ∀ (y z : ℝ), x = k * y^2) -- x is directly proportional to y²
  (h2 : ∀ (y z : ℝ), y = n / Real.sqrt z) -- y is inversely proportional to √z
  (h3 : k * (n / Real.sqrt 16)^2 = 4) -- x = 4 when z = 16
  : k * (n / Real.sqrt 64)^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_x_value_when_z_is_64_l3629_362940


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_6_equals_3_sqrt_2_l3629_362964

theorem sqrt_3_times_sqrt_6_equals_3_sqrt_2 : Real.sqrt 3 * Real.sqrt 6 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_6_equals_3_sqrt_2_l3629_362964


namespace NUMINAMATH_CALUDE_parallelogram_area_l3629_362971

/-- The area of a parallelogram with given side lengths and angle between them -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (ha : a = 15) (hb : b = 20) (hθ : θ = 35 * π / 180) :
  abs (a * b * Real.sin θ - 172.08) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3629_362971


namespace NUMINAMATH_CALUDE_counterexample_exists_l3629_362931

theorem counterexample_exists : ∃ (a b c : ℝ), a > b ∧ ¬(a * c > b * c) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3629_362931


namespace NUMINAMATH_CALUDE_emily_trivia_score_l3629_362924

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round second_round last_round : ℤ) 
  (h1 : first_round = 16)
  (h2 : second_round = 33)
  (h3 : last_round = -48) :
  first_round + second_round + last_round = 1 := by
  sorry


end NUMINAMATH_CALUDE_emily_trivia_score_l3629_362924


namespace NUMINAMATH_CALUDE_least_possible_QGK_l3629_362966

theorem least_possible_QGK : ∃ (G Q K : ℕ),
  (G ≥ 1 ∧ G ≤ 9) ∧
  (Q ≥ 0 ∧ Q ≤ 9) ∧
  (K ≥ 0 ∧ K ≤ 9) ∧
  (G ≠ K) ∧
  (10 * G + G) * G = 100 * Q + 10 * G + K ∧
  ∀ (G' Q' K' : ℕ),
    (G' ≥ 1 ∧ G' ≤ 9) →
    (Q' ≥ 0 ∧ Q' ≤ 9) →
    (K' ≥ 0 ∧ K' ≤ 9) →
    (G' ≠ K') →
    (10 * G' + G') * G' = 100 * Q' + 10 * G' + K' →
    100 * Q + 10 * G + K ≤ 100 * Q' + 10 * G' + K' ∧
  100 * Q + 10 * G + K = 044 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_QGK_l3629_362966


namespace NUMINAMATH_CALUDE_inequality_properties_l3629_362938

theorem inequality_properties (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a * b > b ^ 2) ∧ (1 / a < 1 / b) ∧ (a ^ 2 > a * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l3629_362938


namespace NUMINAMATH_CALUDE_percentage_of_women_employees_l3629_362981

theorem percentage_of_women_employees (men_with_degree : ℝ) (men_without_degree : ℕ) (total_women : ℕ) : 
  men_with_degree = 0.75 * (men_with_degree + men_without_degree) →
  men_without_degree = 8 →
  total_women = 48 →
  (total_women : ℝ) / ((men_with_degree + men_without_degree : ℝ) + total_women) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_women_employees_l3629_362981


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3629_362942

/-- An arithmetic sequence with a positive common difference -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  (a 1 + a 2 + a 3 = 15) →
  (a 1 * a 2 * a 3 = 80) →
  (a 11 + a 12 + a 13 = 105) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3629_362942


namespace NUMINAMATH_CALUDE_max_ships_on_board_l3629_362977

/-- Represents a ship placement on a board -/
structure ShipPlacement where
  board_size : Nat × Nat
  ship_size : Nat × Nat
  ship_count : Nat

/-- Checks if a ship placement is valid -/
def is_valid_placement (p : ShipPlacement) : Prop :=
  p.board_size.1 = 10 ∧
  p.board_size.2 = 10 ∧
  p.ship_size.1 = 1 ∧
  p.ship_size.2 = 4 ∧
  p.ship_count ≤ 25

/-- Theorem stating the maximum number of ships -/
theorem max_ships_on_board :
  ∃ (p : ShipPlacement), is_valid_placement p ∧
    ∀ (q : ShipPlacement), is_valid_placement q → q.ship_count ≤ p.ship_count :=
sorry

end NUMINAMATH_CALUDE_max_ships_on_board_l3629_362977


namespace NUMINAMATH_CALUDE_student_meeting_probability_l3629_362935

def library_open_time : ℝ := 120

theorem student_meeting_probability (n : ℝ) : 
  (0 < n) → 
  (n < library_open_time) → 
  ((library_open_time - n)^2 / library_open_time^2 = 1/2) → 
  (n = 120 - 60 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_student_meeting_probability_l3629_362935


namespace NUMINAMATH_CALUDE_functional_equation_implies_even_l3629_362937

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + k * f b

theorem functional_equation_implies_even (f : ℝ → ℝ) (k : ℝ) 
    (h_eq : FunctionalEquation f k) (h_nonzero : ∃ x, f x ≠ 0) : 
    ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_even_l3629_362937


namespace NUMINAMATH_CALUDE_meeting_probability_l3629_362903

-- Define the time range in minutes
def timeRange : ℝ := 60

-- Define the waiting time in minutes
def waitTime : ℝ := 10

-- Define the probability of meeting function
def probabilityOfMeeting (arrivalRange1 : ℝ) (arrivalRange2 : ℝ) : ℚ :=
  sorry

theorem meeting_probability :
  (probabilityOfMeeting timeRange timeRange = 11/36) ∧
  (probabilityOfMeeting (timeRange/2) timeRange = 11/36) ∧
  (probabilityOfMeeting (5*timeRange/6) timeRange = 19/60) :=
sorry

end NUMINAMATH_CALUDE_meeting_probability_l3629_362903


namespace NUMINAMATH_CALUDE_sales_tax_rate_zero_l3629_362992

theorem sales_tax_rate_zero (sale_price_with_tax : ℝ) (profit_percentage : ℝ) (cost_price : ℝ)
  (h1 : sale_price_with_tax = 616)
  (h2 : profit_percentage = 16)
  (h3 : cost_price = 531.03) :
  let profit := (profit_percentage / 100) * cost_price
  let sale_price_before_tax := cost_price + profit
  let sales_tax_rate := ((sale_price_with_tax - sale_price_before_tax) / sale_price_before_tax) * 100
  sales_tax_rate = 0 := by sorry

end NUMINAMATH_CALUDE_sales_tax_rate_zero_l3629_362992


namespace NUMINAMATH_CALUDE_range_of_m_l3629_362993

open Set Real

theorem range_of_m (m : ℝ) : 
  let A : Set ℝ := {x | -1 < x ∧ x < 7}
  let B : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 3 * m + 1}
  let p := A ∩ B = B
  let q := ∃! x, x^2 + 2*m*x + 2*m ≤ 0
  ¬(p ∨ q) → m ∈ Ici 2 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_m_l3629_362993


namespace NUMINAMATH_CALUDE_fifth_term_is_seven_l3629_362911

/-- An arithmetic sequence with first term a, common difference d, and n-th term given by a + (n-1)d -/
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1 : ℝ) * d

/-- The second term of the sequence -/
def x : ℝ := 1

/-- Given an arithmetic sequence where the first three terms are -1, x, and 3, 
    the fifth term of this sequence is 7 -/
theorem fifth_term_is_seven :
  let a := -1
  let d := x - a
  arithmetic_sequence a d 5 = 7 := by sorry

end NUMINAMATH_CALUDE_fifth_term_is_seven_l3629_362911


namespace NUMINAMATH_CALUDE_prob_one_male_one_female_proof_l3629_362910

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of students to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting one male and one female student -/
def prob_one_male_one_female : ℚ := 3 / 5

theorem prob_one_male_one_female_proof :
  (num_male.choose 1 * num_female.choose 1) / total_students.choose num_selected = prob_one_male_one_female :=
sorry

end NUMINAMATH_CALUDE_prob_one_male_one_female_proof_l3629_362910


namespace NUMINAMATH_CALUDE_gcd_1237_1849_l3629_362927

theorem gcd_1237_1849 : Nat.gcd 1237 1849 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1237_1849_l3629_362927


namespace NUMINAMATH_CALUDE_set_B_when_a_is_2_A_equals_B_when_a_is_negative_one_l3629_362943

-- Define set A
def setA (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 3*a - 1) < 0}

-- Define set B (domain of log(x))
def setB : Set ℝ := {x | x > 0}

-- Theorem 1: When a=2, B = {x | 2 < x < 7}
theorem set_B_when_a_is_2 :
  setB = {x : ℝ | 2 < x ∧ x < 7} ∧ ∀ x ∈ setB, (x - 2) * (x - 7) < 0 :=
sorry

-- Theorem 2: A = B only when a = -1
theorem A_equals_B_when_a_is_negative_one :
  ∃! a : ℝ, setA a = setB ∧ a = -1 :=
sorry

end NUMINAMATH_CALUDE_set_B_when_a_is_2_A_equals_B_when_a_is_negative_one_l3629_362943


namespace NUMINAMATH_CALUDE_three_fifths_square_specific_number_l3629_362906

theorem three_fifths_square_specific_number : 
  (3 / 5 : ℝ) * (14.500000000000002 ^ 2) = 126.15000000000002 := by
  sorry

end NUMINAMATH_CALUDE_three_fifths_square_specific_number_l3629_362906


namespace NUMINAMATH_CALUDE_nested_sqrt_calculation_l3629_362907

theorem nested_sqrt_calculation : Real.sqrt (32 * Real.sqrt (16 * Real.sqrt (8 * Real.sqrt 4))) = 16 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_calculation_l3629_362907


namespace NUMINAMATH_CALUDE_impossibleTransformation_l3629_362947

-- Define the colors
inductive Color
| Green
| Blue
| Red

-- Define the circle as a list of colors
def Circle := List Color

-- Define the initial and target states
def initialState : Circle := [Color.Green, Color.Blue, Color.Red]
def targetState : Circle := [Color.Blue, Color.Green, Color.Red]

-- Define the operations
def addBetweenDifferent (c : Circle) (i : Nat) (newColor : Color) : Circle := sorry
def addBetweenSame (c : Circle) (i : Nat) (newColor : Color) : Circle := sorry
def deleteMiddle (c : Circle) (i : Nat) : Circle := sorry

-- Define a single step transformation
def step (c : Circle) : Circle := sorry

-- Define the transformation process
def transform (c : Circle) (n : Nat) : Circle :=
  match n with
  | 0 => c
  | n + 1 => step (transform c n)

-- Theorem statement
theorem impossibleTransformation : 
  ∀ n : Nat, transform initialState n ≠ targetState := sorry

end NUMINAMATH_CALUDE_impossibleTransformation_l3629_362947


namespace NUMINAMATH_CALUDE_arithmetic_sequence_specific_terms_l3629_362921

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + seq.diff * (n - 1)

/-- Theorem: In an arithmetic sequence where the 4th term is 23 and the 6th term is 47, the 8th term is 71 -/
theorem arithmetic_sequence_specific_terms
  (seq : ArithmeticSequence)
  (h4 : seq.nthTerm 4 = 23)
  (h6 : seq.nthTerm 6 = 47) :
  seq.nthTerm 8 = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_specific_terms_l3629_362921


namespace NUMINAMATH_CALUDE_golden_ratio_logarithm_l3629_362919

theorem golden_ratio_logarithm (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 15 ∧ 
       Real.log p / Real.log 8 = Real.log (p + q) / Real.log 18) : 
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_logarithm_l3629_362919


namespace NUMINAMATH_CALUDE_min_time_proof_l3629_362908

-- Define the quantities of honey and milk
def honey_pots : ℕ := 10
def milk_cans : ℕ := 22

-- Define the time taken by Pooh and Piglet for honey and milk
def pooh_honey_time : ℕ := 2
def pooh_milk_time : ℕ := 1
def piglet_honey_time : ℕ := 5
def piglet_milk_time : ℕ := 3

-- Define the function to calculate the minimum time
def min_consumption_time : ℕ :=
  -- The actual calculation is not implemented here
  30

-- State the theorem
theorem min_time_proof :
  min_consumption_time = 30 :=
sorry

end NUMINAMATH_CALUDE_min_time_proof_l3629_362908


namespace NUMINAMATH_CALUDE_no_solution_exists_l3629_362934

/-- Set A defined as {(x, y) | x = n, y = na + b, n ∈ ℤ} -/
def A (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ n : ℤ, p.1 = n ∧ p.2 = n * a + b}

/-- Set B defined as {(x, y) | x = m, y = 3m^2 + 15, m ∈ ℤ} -/
def B : Set (ℝ × ℝ) :=
  {p | ∃ m : ℤ, p.1 = m ∧ p.2 = 3 * m^2 + 15}

/-- Set C defined as {(x, y) | x^2 + y^2 ≤ 144, x, y ∈ ℝ} -/
def C : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 ≤ 144}

/-- Theorem stating that there do not exist real numbers a and b satisfying both conditions -/
theorem no_solution_exists : ¬∃ a b : ℝ, (A a b ∩ B).Nonempty ∧ (a, b) ∈ C := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3629_362934


namespace NUMINAMATH_CALUDE_mary_nickels_l3629_362967

def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 5
def mom_multiplier : ℕ := 2

def final_nickels : ℕ := initial_nickels + dad_nickels + mom_multiplier * (initial_nickels + dad_nickels)

theorem mary_nickels : final_nickels = 36 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_l3629_362967


namespace NUMINAMATH_CALUDE_units_digit_problem_l3629_362912

theorem units_digit_problem : ∃ n : ℕ, (8 * 18 * 1978 - 8^3) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l3629_362912


namespace NUMINAMATH_CALUDE_frog_arrangements_eq_25200_l3629_362953

/-- The number of ways to arrange 8 frogs (3 green, 4 red, 1 blue) in a row,
    where green frogs cannot sit next to the blue frog. -/
def frog_arrangements : ℕ :=
  let total_frogs : ℕ := 8
  let green_frogs : ℕ := 3
  let red_frogs : ℕ := 4
  let blue_frogs : ℕ := 1
  let red_arrangements : ℕ := Nat.factorial red_frogs
  let blue_positions : ℕ := red_frogs + 1
  let green_positions : ℕ := total_frogs - 1
  let green_arrangements : ℕ := Nat.choose green_positions green_frogs * Nat.factorial green_frogs
  red_arrangements * blue_positions * green_arrangements

theorem frog_arrangements_eq_25200 : frog_arrangements = 25200 := by
  sorry

end NUMINAMATH_CALUDE_frog_arrangements_eq_25200_l3629_362953


namespace NUMINAMATH_CALUDE_proposition_analysis_l3629_362939

-- Define proposition P
def P (x y : ℝ) : Prop := x ≠ y → abs x ≠ abs y

-- Define the converse of P
def P_converse (x y : ℝ) : Prop := abs x ≠ abs y → x ≠ y

-- Define the negation of P
def P_negation (x y : ℝ) : Prop := ¬(x ≠ y → abs x ≠ abs y)

-- Define the contrapositive of P
def P_contrapositive (x y : ℝ) : Prop := abs x = abs y → x = y

theorem proposition_analysis :
  (∃ x y : ℝ, ¬(P x y)) ∧
  (∀ x y : ℝ, P_converse x y) ∧
  (∀ x y : ℝ, P_negation x y) ∧
  (∃ x y : ℝ, ¬(P_contrapositive x y)) :=
sorry

end NUMINAMATH_CALUDE_proposition_analysis_l3629_362939


namespace NUMINAMATH_CALUDE_inequality_proof_l3629_362995

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b / a > b - a / b) ∧ (1 / a + c < 1 / b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3629_362995


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_for_modified_quadratic_l3629_362913

theorem no_positive_integer_solutions_for_modified_quadratic :
  ∀ A : ℕ, 1 ≤ A → A ≤ 9 →
    ¬∃ x : ℕ, x > 0 ∧ x^2 - (10 * A + 1) * x + (10 * A + A) = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_for_modified_quadratic_l3629_362913


namespace NUMINAMATH_CALUDE_traffic_class_drunk_drivers_l3629_362951

theorem traffic_class_drunk_drivers :
  ∀ (drunk_drivers speeders : ℕ),
    drunk_drivers + speeders = 45 →
    speeders = 7 * drunk_drivers - 3 →
    drunk_drivers = 6 := by
sorry

end NUMINAMATH_CALUDE_traffic_class_drunk_drivers_l3629_362951


namespace NUMINAMATH_CALUDE_f_derivative_l3629_362973

noncomputable def f (x : ℝ) : ℝ := -2 * Real.exp x * Real.sin x

theorem f_derivative (x : ℝ) : 
  deriv f x = -2 * Real.exp x * (Real.sin x + Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l3629_362973


namespace NUMINAMATH_CALUDE_hash_prehash_eighteen_l3629_362914

-- Define the # operator
def hash (x : ℝ) : ℝ := x + 5

-- Define the # prefix operator
def prehash (x : ℝ) : ℝ := x - 5

-- Theorem statement
theorem hash_prehash_eighteen : prehash (hash 18) = 18 := by
  sorry

end NUMINAMATH_CALUDE_hash_prehash_eighteen_l3629_362914


namespace NUMINAMATH_CALUDE_store_b_cheaper_for_40_l3629_362936

-- Define the rental fee functions
def y₁ (x : ℕ) : ℝ := 96 * x

def y₂ (x : ℕ) : ℝ :=
  if x ≤ 6 then 160 * x else 80 * x + 480

-- Theorem statement
theorem store_b_cheaper_for_40 :
  y₂ 40 < y₁ 40 := by
  sorry

end NUMINAMATH_CALUDE_store_b_cheaper_for_40_l3629_362936


namespace NUMINAMATH_CALUDE_positive_even_product_sum_zero_l3629_362904

theorem positive_even_product_sum_zero (n : ℕ) (h_pos : n > 0) (h_even : Even n) :
  ∃ (a b : ℤ), (n : ℤ) = a * b ∧ a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_even_product_sum_zero_l3629_362904


namespace NUMINAMATH_CALUDE_otimes_properties_l3629_362991

-- Define the operation
def otimes (a b : ℝ) : ℝ := a * (1 - b)

-- Statement of the theorem
theorem otimes_properties :
  (otimes 2 (-2) = 6) ∧ 
  (∃ a b : ℝ, otimes a b ≠ otimes b a) ∧
  (∀ a b : ℝ, a + b = 0 → otimes a a + otimes b b = 2 * a * b) ∧
  (∃ a b : ℝ, otimes a b = 0 ∧ a ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_otimes_properties_l3629_362991


namespace NUMINAMATH_CALUDE_subset_iff_range_l3629_362983

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - a + 1) * (x - a - 1) ≤ 0}
def B : Set ℝ := {x | |x - 1/2| ≤ 3/2}

-- State the theorem
theorem subset_iff_range (a : ℝ) : A a ⊆ B ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_subset_iff_range_l3629_362983


namespace NUMINAMATH_CALUDE_shadow_length_of_shorter_cycle_l3629_362960

/-- Given two similar right-angled triangles formed by cycles and their shadows,
    this theorem proves the length of the shadow for the shorter cycle. -/
theorem shadow_length_of_shorter_cycle
  (H1 : ℝ) (S1 : ℝ) (H2 : ℝ)
  (height1 : H1 = 2.5)
  (shadow1 : S1 = 5)
  (height2 : H2 = 2)
  (similar_triangles : H1 / S1 = H2 / S2)
  : S2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_shadow_length_of_shorter_cycle_l3629_362960


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3629_362965

theorem fixed_point_on_line (k : ℝ) : 
  (k + 1) * 4 + (-6) + 2 - 4 * k = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3629_362965


namespace NUMINAMATH_CALUDE_f_value_at_2017_5_l3629_362990

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def f_on_unit_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

theorem f_value_at_2017_5 (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period_2 f) 
  (h_unit : f_on_unit_interval f) : 
  f 2017.5 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_2017_5_l3629_362990


namespace NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l3629_362985

theorem evaluate_sqrt_fraction (x : ℝ) (h : x < -1) :
  Real.sqrt (x / (1 - (x - 2) / (x + 1))) = -x * (x + 1) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l3629_362985


namespace NUMINAMATH_CALUDE_abs_x_less_than_2_sufficient_not_necessary_l3629_362969

theorem abs_x_less_than_2_sufficient_not_necessary :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  ¬(∀ x : ℝ, x^2 - x - 6 < 0 → |x| < 2) := by
sorry

end NUMINAMATH_CALUDE_abs_x_less_than_2_sufficient_not_necessary_l3629_362969


namespace NUMINAMATH_CALUDE_board_cut_theorem_l3629_362949

/-- Given a board of length 120 cm cut into two pieces, where the longer piece is 15 cm longer
    than twice the length of the shorter piece, prove that the shorter piece is 35 cm long. -/
theorem board_cut_theorem (shorter_piece longer_piece : ℝ) : 
  shorter_piece + longer_piece = 120 →
  longer_piece = 2 * shorter_piece + 15 →
  shorter_piece = 35 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l3629_362949


namespace NUMINAMATH_CALUDE_equation_solution_l3629_362989

theorem equation_solution :
  ∀ x : ℚ,
  (x^2 - 4*x + 3) / (x^2 - 7*x + 6) = (x^2 - 3*x - 10) / (x^2 - 2*x - 15) →
  x = -3/4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3629_362989


namespace NUMINAMATH_CALUDE_kens_height_l3629_362925

/-- Given the heights of Ivan and Jackie, and the relationship between the averages,
    prove Ken's height. -/
theorem kens_height (h_ivan : ℝ) (h_jackie : ℝ) (h_ken : ℝ) :
  h_ivan = 175 →
  h_jackie = 175 →
  (h_ivan + h_jackie + h_ken) / 3 = 1.04 * (h_ivan + h_jackie) / 2 →
  h_ken = 196 := by
  sorry

end NUMINAMATH_CALUDE_kens_height_l3629_362925


namespace NUMINAMATH_CALUDE_g_composition_3_l3629_362968

def g : ℕ → ℕ
| x => if x % 2 = 0 then x / 2
       else if x < 10 then 3 * x + 2
       else x - 1

theorem g_composition_3 : g (g (g (g (g 3)))) = 16 := by sorry

end NUMINAMATH_CALUDE_g_composition_3_l3629_362968


namespace NUMINAMATH_CALUDE_protective_clothing_production_l3629_362961

/-- Represents the situation of a factory producing protective clothing --/
theorem protective_clothing_production 
  (total_production : ℕ) 
  (overtime_increase : ℚ) 
  (days_ahead : ℕ) 
  (x : ℚ) 
  (h1 : total_production = 1000) 
  (h2 : overtime_increase = 1/5) 
  (h3 : days_ahead = 2) 
  (h4 : x > 0) :
  (total_production / x) - (total_production / ((1 + overtime_increase) * x)) = days_ahead :=
sorry

end NUMINAMATH_CALUDE_protective_clothing_production_l3629_362961


namespace NUMINAMATH_CALUDE_M_is_solution_set_inequality_holds_l3629_362980

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Define the set M
def M : Set ℝ := {x | x < -1 ∨ x > 1}

-- Statement 1: M is the solution set for f(x) < |2x+1| - 1
theorem M_is_solution_set : ∀ x : ℝ, x ∈ M ↔ f x < |2*x + 1| - 1 :=
sorry

-- Statement 2: For any a, b ∈ M, f(ab) > f(a) - f(-b)
theorem inequality_holds : ∀ a b : ℝ, a ∈ M → b ∈ M → f (a*b) > f a - f (-b) :=
sorry

end NUMINAMATH_CALUDE_M_is_solution_set_inequality_holds_l3629_362980


namespace NUMINAMATH_CALUDE_min_value_expression_l3629_362901

theorem min_value_expression (c : ℝ) (a b : ℝ) (hc : c > 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (h_eq : 4 * a^2 - 2 * a * b + b^2 - c = 0)
  (h_max : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → 4 * x^2 - 2 * x * y + y^2 - c = 0 → |2 * x + y| ≤ |2 * a + b|) :
  ∃ (k : ℝ), k = 1/a + 2/b + 4/c ∧ k ≥ -1 ∧ (∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z > 0 →
    4 * x^2 - 2 * x * y + y^2 - z = 0 → 1/x + 2/y + 4/z ≥ k) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3629_362901


namespace NUMINAMATH_CALUDE_number_exceeds_fraction_l3629_362959

theorem number_exceeds_fraction : 
  ∀ x : ℚ, x = (3/8) * x + 25 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeds_fraction_l3629_362959


namespace NUMINAMATH_CALUDE_igloo_bottom_row_bricks_l3629_362986

/-- Represents the structure of an igloo --/
structure Igloo where
  total_rows : ℕ
  top_row_bricks : ℕ
  total_bricks : ℕ

/-- Calculates the number of bricks in each row of the bottom half of the igloo --/
def bottom_row_bricks (igloo : Igloo) : ℕ :=
  let bottom_rows := igloo.total_rows / 2
  let top_bricks := bottom_rows * igloo.top_row_bricks
  (igloo.total_bricks - top_bricks) / bottom_rows

/-- Theorem stating that for the given igloo specifications, 
    the number of bricks in each row of the bottom half is 12 --/
theorem igloo_bottom_row_bricks :
  let igloo : Igloo := { total_rows := 10, top_row_bricks := 8, total_bricks := 100 }
  bottom_row_bricks igloo = 12 := by
  sorry


end NUMINAMATH_CALUDE_igloo_bottom_row_bricks_l3629_362986


namespace NUMINAMATH_CALUDE_symmetric_axis_of_sine_function_l3629_362948

/-- Given a function y = 2sin(2x + φ) where |φ| < π/2, and the graph passes through (0, √3),
    prove that one symmetric axis of the graph is x = π/12 -/
theorem symmetric_axis_of_sine_function (φ : ℝ) (h1 : |φ| < π/2) 
    (h2 : 2 * Real.sin φ = Real.sqrt 3) : 
  ∃ (k : ℤ), π/12 = k * π/2 + π/4 - φ/2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_axis_of_sine_function_l3629_362948


namespace NUMINAMATH_CALUDE_shopkeeper_total_cards_l3629_362972

/-- The number of complete decks of standard playing cards -/
def standard_decks : ℕ := 3

/-- The number of cards in a complete deck of standard playing cards -/
def cards_per_standard_deck : ℕ := 52

/-- The number of incomplete decks of tarot cards -/
def tarot_decks : ℕ := 2

/-- The number of cards in each incomplete tarot deck -/
def cards_per_tarot_deck : ℕ := 72

/-- The number of sets of trading cards -/
def trading_card_sets : ℕ := 5

/-- The number of cards in each trading card set -/
def cards_per_trading_set : ℕ := 100

/-- The number of additional random cards -/
def random_cards : ℕ := 27

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 
  standard_decks * cards_per_standard_deck + 
  tarot_decks * cards_per_tarot_deck + 
  trading_card_sets * cards_per_trading_set + 
  random_cards

theorem shopkeeper_total_cards : total_cards = 827 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_total_cards_l3629_362972


namespace NUMINAMATH_CALUDE_sin_150_degrees_l3629_362944

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l3629_362944


namespace NUMINAMATH_CALUDE_right_triangle_min_leg_sum_l3629_362974

theorem right_triangle_min_leg_sum (a b : ℝ) (h_right : a > 0 ∧ b > 0) (h_area : (1/2) * a * b = 50) :
  a + b ≥ 20 ∧ (a + b = 20 ↔ a = 10 ∧ b = 10) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_min_leg_sum_l3629_362974


namespace NUMINAMATH_CALUDE_line_equation_l3629_362955

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 4 * x + y + 6 = 0
def l₂ (x y : ℝ) : Prop := 3 * x - 5 * y - 6 = 0

-- Define the midpoint P
def P : ℝ × ℝ := (0, 0)

-- Define the line l (to be proven)
def l (x y : ℝ) : Prop := y = -1/6 * x

-- Theorem statement
theorem line_equation (A B : ℝ × ℝ) :
  l₁ A.1 A.2 →
  l₂ B.1 B.2 →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∀ x y, l x y ↔ ∃ t, x = t * A.1 ∧ y = t * A.2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3629_362955


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_parallel_l3629_362958

/-- Two lines are parallel if their slopes are equal and they are not identical -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  (a * f = b * d) ∧ (a * e ≠ b * c)

theorem sufficient_but_not_necessary_parallel :
  (are_parallel 3 2 1 3 2 (-2)) ∧
  (∃ a : ℝ, a ≠ 3 ∧ are_parallel a 2 1 3 (a - 1) (-2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_parallel_l3629_362958


namespace NUMINAMATH_CALUDE_sqrt_of_neg_two_squared_l3629_362950

theorem sqrt_of_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_neg_two_squared_l3629_362950


namespace NUMINAMATH_CALUDE_square_field_area_l3629_362917

/-- The area of a square field with a diagonal of 26 meters is 338 square meters. -/
theorem square_field_area (diagonal : ℝ) (h : diagonal = 26) :
  (diagonal ^ 2) / 2 = 338 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l3629_362917


namespace NUMINAMATH_CALUDE_max_rectangles_after_removal_l3629_362982

/-- Represents a grid with some squares removed -/
structure Grid :=
  (size : Nat)
  (removedSquares : List (Nat × Nat × Nat))

/-- Represents a rectangle -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- The maximum number of rectangles that can be cut from a grid -/
def maxRectangles (g : Grid) (r : Rectangle) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem max_rectangles_after_removal :
  let initialGrid : Grid := { size := 8, removedSquares := [(2, 2, 3)] }
  let targetRectangle : Rectangle := { width := 1, height := 3 }
  maxRectangles initialGrid targetRectangle = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_after_removal_l3629_362982


namespace NUMINAMATH_CALUDE_correct_donations_l3629_362941

/-- Represents the donation amounts to each charity -/
structure CharityDonations where
  homeless : ℝ
  foodBank : ℝ
  parkRestoration : ℝ
  animalRescue : ℝ

/-- Calculates the total donations to charities given the bake sale earnings and conditions -/
def calculateDonations (totalEarnings personalDonation costOfIngredients : ℝ) : CharityDonations :=
  let remainingForCharity := totalEarnings - costOfIngredients
  let homelessShare := 0.30 * remainingForCharity + personalDonation
  let foodBankShare := 0.25 * remainingForCharity + personalDonation
  let parkRestorationShare := 0.20 * remainingForCharity + personalDonation
  let animalRescueShare := 0.25 * remainingForCharity + personalDonation
  { homeless := homelessShare
  , foodBank := foodBankShare
  , parkRestoration := parkRestorationShare
  , animalRescue := animalRescueShare }

theorem correct_donations :
  let donations := calculateDonations 500 15 110
  donations.homeless = 132 ∧
  donations.foodBank = 112.5 ∧
  donations.parkRestoration = 93 ∧
  donations.animalRescue = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_correct_donations_l3629_362941


namespace NUMINAMATH_CALUDE_vacation_cost_l3629_362998

theorem vacation_cost (num_people : ℕ) (plane_ticket_cost : ℕ) (hotel_cost_per_day : ℕ) (num_days : ℕ) : 
  num_people = 2 → 
  plane_ticket_cost = 24 → 
  hotel_cost_per_day = 12 → 
  num_days = 3 → 
  num_people * plane_ticket_cost + num_people * hotel_cost_per_day * num_days = 120 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_l3629_362998


namespace NUMINAMATH_CALUDE_multiply_and_add_equality_l3629_362988

theorem multiply_and_add_equality : 52 * 46 + 104 * 52 = 7800 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_equality_l3629_362988


namespace NUMINAMATH_CALUDE_cos_three_pi_four_plus_two_alpha_l3629_362945

theorem cos_three_pi_four_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 8 - α) = 1 / 6) : 
  Real.cos (3 * π / 4 + 2 * α) = 17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_four_plus_two_alpha_l3629_362945


namespace NUMINAMATH_CALUDE_mixed_fraction_calculation_l3629_362909

theorem mixed_fraction_calculation : 
  (((5:ℚ)/2 - 10/3)^2) / ((17:ℚ)/4 + 7/6) = 5/39 := by sorry

end NUMINAMATH_CALUDE_mixed_fraction_calculation_l3629_362909


namespace NUMINAMATH_CALUDE_remainder_problem_l3629_362957

theorem remainder_problem (x : ℤ) : x % 84 = 25 → x % 14 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3629_362957


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3629_362954

/-- An isosceles triangle with side lengths 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Given side lengths
    (b = c) →                  -- Isosceles condition
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (a + b + c = 22)           -- Perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : 
  ∃ (a b c : ℝ), isosceles_triangle_perimeter a b c :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3629_362954


namespace NUMINAMATH_CALUDE_football_purchase_theorem_l3629_362930

/-- Represents the cost and quantity of footballs purchased by a school --/
structure FootballPurchase where
  type_a_cost : ℕ
  type_b_cost : ℕ
  type_a_quantity : ℕ
  type_b_quantity : ℕ
  total_cost : ℕ
  cost_difference : ℕ

/-- Represents the second purchase with budget constraints --/
structure SecondPurchase where
  budget : ℕ
  total_quantity : ℕ

/-- Theorem stating the costs of footballs and minimum quantity of type A footballs in second purchase --/
theorem football_purchase_theorem (fp : FootballPurchase) (sp : SecondPurchase) :
  fp.type_a_quantity = 50 ∧ 
  fp.type_b_quantity = 25 ∧ 
  fp.total_cost = 7500 ∧ 
  fp.cost_difference = 30 ∧ 
  fp.type_b_cost = fp.type_a_cost + fp.cost_difference ∧
  sp.budget = 4800 ∧
  sp.total_quantity = 50 →
  fp.type_a_cost = 90 ∧ 
  fp.type_b_cost = 120 ∧ 
  (∃ m : ℕ, m ≥ 40 ∧ m * fp.type_a_cost + (sp.total_quantity - m) * fp.type_b_cost ≤ sp.budget) :=
by sorry

end NUMINAMATH_CALUDE_football_purchase_theorem_l3629_362930


namespace NUMINAMATH_CALUDE_remaining_grain_l3629_362928

theorem remaining_grain (original : ℕ) (spilled : ℕ) (remaining : ℕ) : 
  original = 50870 → spilled = 49952 → remaining = original - spilled → remaining = 918 := by
  sorry

end NUMINAMATH_CALUDE_remaining_grain_l3629_362928


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l3629_362926

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_hours : ℝ)
  (overtime_rate_increase : ℝ) :
  regular_rate = 16 →
  regular_hours = 40 →
  overtime_hours = 8 →
  overtime_rate_increase = 0.75 →
  regular_rate * regular_hours +
  (regular_rate * (1 + overtime_rate_increase)) * overtime_hours = 864 :=
by sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l3629_362926


namespace NUMINAMATH_CALUDE_percentage_not_sold_l3629_362918

def initial_stock : ℕ := 1200
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales
def books_not_sold : ℕ := initial_stock - total_sales

theorem percentage_not_sold : 
  (books_not_sold : ℚ) / initial_stock * 100 = 66.5 := by sorry

end NUMINAMATH_CALUDE_percentage_not_sold_l3629_362918


namespace NUMINAMATH_CALUDE_remainder_seven_eight_mod_hundred_l3629_362952

theorem remainder_seven_eight_mod_hundred : 7^8 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_eight_mod_hundred_l3629_362952


namespace NUMINAMATH_CALUDE_xiao_ming_current_age_l3629_362975

/-- Xiao Ming's age this year -/
def xiao_ming_age : ℕ := sorry

/-- Xiao Ming's mother's age this year -/
def mother_age : ℕ := sorry

/-- Xiao Ming's age three years from now -/
def xiao_ming_age_future : ℕ := sorry

/-- Xiao Ming's mother's age three years from now -/
def mother_age_future : ℕ := sorry

/-- The theorem stating Xiao Ming's age this year -/
theorem xiao_ming_current_age :
  (mother_age = 3 * xiao_ming_age) ∧
  (mother_age_future = 2 * xiao_ming_age_future + 10) ∧
  (xiao_ming_age_future = xiao_ming_age + 3) ∧
  (mother_age_future = mother_age + 3) →
  xiao_ming_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_current_age_l3629_362975


namespace NUMINAMATH_CALUDE_refrigerator_payment_proof_l3629_362963

def refrigerator_problem (first_payment second_payment third_payment : ℝ)
  (first_percent second_percent third_percent sales_tax_rate : ℝ)
  (delivery_fee : ℝ) : Prop :=
  let total_cost := first_payment / first_percent
  let sales_tax := sales_tax_rate * total_cost
  let total_with_tax_and_fee := total_cost + sales_tax + delivery_fee
  let total_payments := first_payment + second_payment + third_payment
  let remaining_payment := total_with_tax_and_fee - total_payments
  remaining_payment = 1137.50

theorem refrigerator_payment_proof :
  refrigerator_problem 875 650 1200 0.25 0.15 0.35 0.075 100 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_payment_proof_l3629_362963


namespace NUMINAMATH_CALUDE_cable_section_length_l3629_362900

theorem cable_section_length :
  ∀ (total_length : ℝ) (sections : ℝ) (kept_sections : ℝ),
    total_length = 1000 →
    sections > 0 →
    kept_sections = 15 →
    kept_sections = (1/2) * (3/4) * (total_length / sections) →
    sections = total_length / 25 :=
by
  sorry

end NUMINAMATH_CALUDE_cable_section_length_l3629_362900


namespace NUMINAMATH_CALUDE_series_sum_equals_one_l3629_362933

/-- The sum of the series ∑(n=1 to ∞) (4n-3)/3^n is equal to 1 -/
theorem series_sum_equals_one :
  (∑' n : ℕ, (4 * n - 3 : ℝ) / 3^n) = 1 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_l3629_362933


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3629_362996

/-- Given that (z - 2i)(2 - i) = 5, prove that z = 2 + 3i -/
theorem complex_equation_solution (z : ℂ) (h : (z - 2*Complex.I)*(2 - Complex.I) = 5) :
  z = 2 + 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3629_362996


namespace NUMINAMATH_CALUDE_opposite_of_neg_2023_l3629_362923

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- Theorem: The opposite of -2023 is 2023. -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_2023_l3629_362923


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_l3629_362976

theorem greatest_integer_fraction (x : ℤ) : (5 : ℚ) / 8 > (x : ℚ) / 15 ↔ x ≤ 9 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_l3629_362976


namespace NUMINAMATH_CALUDE_quadratic_a_value_l3629_362916

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The theorem stating the value of 'a' for a quadratic function with given properties -/
theorem quadratic_a_value (f : QuadraticFunction) 
  (vertex : f.f (-2) = 3)
  (point : f.f 1 = 6) :
  f.a = 1/3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_a_value_l3629_362916


namespace NUMINAMATH_CALUDE_first_box_not_empty_count_l3629_362946

/-- The number of ways to distribute three distinct balls into four boxes. -/
def total_distributions : ℕ := 4^3

/-- The number of ways to distribute three distinct balls into four boxes
    such that the first box is empty. -/
def distributions_with_empty_first_box : ℕ := 3^3

theorem first_box_not_empty_count :
  total_distributions - distributions_with_empty_first_box = 37 := by
  sorry

end NUMINAMATH_CALUDE_first_box_not_empty_count_l3629_362946


namespace NUMINAMATH_CALUDE_triangle_property_l3629_362929

theorem triangle_property (A B C : Real) (a b c : Real) :
  2 * Real.sin (2 * A) * Real.cos A - Real.sin (3 * A) + Real.sqrt 3 * Real.cos A = Real.sqrt 3 →
  a = 1 →
  Real.sin A + Real.sin (B - C) = 2 * Real.sin (2 * C) →
  A = π / 3 ∧ (1 / 2) * a * b * Real.sin C = Real.sqrt 3 / 6 :=
by sorry


end NUMINAMATH_CALUDE_triangle_property_l3629_362929


namespace NUMINAMATH_CALUDE_people_disliking_tv_and_books_l3629_362999

def total_surveyed : ℕ := 1500
def tv_dislike_percentage : ℚ := 25 / 100
def book_and_tv_dislike_percentage : ℚ := 15 / 100

theorem people_disliking_tv_and_books :
  ⌊(tv_dislike_percentage * total_surveyed : ℚ) * book_and_tv_dislike_percentage⌋ = 56 := by
  sorry

end NUMINAMATH_CALUDE_people_disliking_tv_and_books_l3629_362999


namespace NUMINAMATH_CALUDE_pizza_feeding_capacity_l3629_362932

theorem pizza_feeding_capacity 
  (total_people : ℕ) 
  (pizza_cost : ℕ) 
  (earnings_per_night : ℕ) 
  (babysitting_nights : ℕ) : 
  total_people / (babysitting_nights * earnings_per_night / pizza_cost) = 3 :=
by
  -- Assuming:
  -- total_people = 15
  -- pizza_cost = 12
  -- earnings_per_night = 4
  -- babysitting_nights = 15
  sorry

#check pizza_feeding_capacity

end NUMINAMATH_CALUDE_pizza_feeding_capacity_l3629_362932


namespace NUMINAMATH_CALUDE_orange_bucket_difference_l3629_362987

/-- Proves that the difference between the number of oranges in the second and first buckets is 17 -/
theorem orange_bucket_difference :
  ∀ (second_bucket : ℕ),
  22 + second_bucket + (second_bucket - 11) = 89 →
  second_bucket - 22 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_bucket_difference_l3629_362987


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3629_362920

theorem functional_equation_solution (f : ℚ → ℝ) 
  (h : ∀ x y : ℚ, f (x + y) = f x + f y + 2 * x * y) : 
  ∃ k : ℝ, ∀ x : ℚ, f x = x^2 + k * x := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3629_362920


namespace NUMINAMATH_CALUDE_part_one_part_two_l3629_362997

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

-- Part 1
theorem part_one (m : ℝ) : 
  (∀ x, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  (∀ x, |x - a| ≥ f 3 x) → (a ≥ 6 ∨ a ≤ 0) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3629_362997


namespace NUMINAMATH_CALUDE_total_diagonals_50_75_l3629_362962

def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem total_diagonals_50_75 : diagonals 50 + diagonals 75 = 3875 := by
  sorry

end NUMINAMATH_CALUDE_total_diagonals_50_75_l3629_362962


namespace NUMINAMATH_CALUDE_cube_difference_l3629_362979

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l3629_362979
