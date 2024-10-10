import Mathlib

namespace fraction_equality_l2589_258931

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 20)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 15) :
  a / d = 4 / 15 := by
  sorry

end fraction_equality_l2589_258931


namespace domain_sum_l2589_258908

theorem domain_sum (y : ℝ → ℝ) (A B : ℝ) : 
  (∀ x, y x = 5 * x / (3 * x^2 - 9 * x + 6)) →
  (3 * A^2 - 9 * A + 6 = 0) →
  (3 * B^2 - 9 * B + 6 = 0) →
  A + B = 3 := by
sorry

end domain_sum_l2589_258908


namespace holly_401k_contribution_l2589_258935

/-- Calculates the total contribution to Holly's 401k after 1 year -/
def total_contribution (paychecks_per_year : ℕ) (contribution_per_paycheck : ℚ) (company_match_percentage : ℚ) : ℚ :=
  let employee_contribution := paychecks_per_year * contribution_per_paycheck
  let company_contribution := employee_contribution * company_match_percentage
  employee_contribution + company_contribution

/-- Theorem stating that Holly's total 401k contribution after 1 year is $2,756.00 -/
theorem holly_401k_contribution :
  total_contribution 26 100 (6 / 100) = 2756 :=
by sorry

end holly_401k_contribution_l2589_258935


namespace total_birds_in_marsh_l2589_258963

def geese : ℕ := 58
def ducks : ℕ := 37
def herons : ℕ := 23
def kingfishers : ℕ := 46
def swans : ℕ := 15

theorem total_birds_in_marsh : geese + ducks + herons + kingfishers + swans = 179 := by
  sorry

end total_birds_in_marsh_l2589_258963


namespace negation_of_existence_negation_of_proposition_l2589_258905

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end negation_of_existence_negation_of_proposition_l2589_258905


namespace no_inscribed_circle_pentagon_l2589_258939

/-- A pentagon with side lengths a, b, c, d, e has an inscribed circle if and only if
    there exists a positive real number r such that
    2(a + b + c + d + e) = (a + b - c - d + e)(a - b + c - d + e)(-a + b + c - d + e)(-a - b + c + d + e)/r^2 -/
def has_inscribed_circle (a b c d e : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 2*(a + b + c + d + e) = (a + b - c - d + e)*(a - b + c - d + e)*(-a + b + c - d + e)*(-a - b + c + d + e)/(r^2)

/-- Theorem: There does not exist a pentagon with side lengths 3, 4, 9, 11, and 13 cm
    that has an inscribed circle -/
theorem no_inscribed_circle_pentagon : ¬ has_inscribed_circle 3 4 9 11 13 := by
  sorry

end no_inscribed_circle_pentagon_l2589_258939


namespace cookies_eaten_l2589_258985

theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) 
  (h1 : initial = 93)
  (h2 : remaining = 78)
  (h3 : initial = remaining + eaten) :
  eaten = 15 := by
  sorry

end cookies_eaten_l2589_258985


namespace all_terms_are_integers_l2589_258921

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (1 + (sequence_a (n + 1))^2) / (sequence_a n)

theorem all_terms_are_integers :
  ∀ n : ℕ, ∃ k : ℤ, sequence_a n = k :=
by sorry

end all_terms_are_integers_l2589_258921


namespace trigonometric_identity_l2589_258991

theorem trigonometric_identity : 
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (225 * π / 180) * Real.sin (165 * π / 180) = 1/2 := by
  sorry

end trigonometric_identity_l2589_258991


namespace largest_fraction_l2589_258994

theorem largest_fraction : 
  let f1 := 5 / 11
  let f2 := 6 / 13
  let f3 := 18 / 37
  let f4 := 101 / 202
  let f5 := 200 / 399
  f5 > f1 ∧ f5 > f2 ∧ f5 > f3 ∧ f5 > f4 := by
  sorry

end largest_fraction_l2589_258994


namespace pitcher_problem_l2589_258900

theorem pitcher_problem (C : ℝ) (h : C > 0) :
  let juice_in_pitcher := (5 / 6) * C
  let juice_per_cup := juice_in_pitcher / 3
  juice_per_cup / C = 5 / 18 := by
sorry

end pitcher_problem_l2589_258900


namespace z3_magnitude_range_l2589_258945

open Complex

theorem z3_magnitude_range (z₁ z₂ z₃ : ℂ) 
  (h1 : abs z₁ = Real.sqrt 2)
  (h2 : abs z₂ = Real.sqrt 2)
  (h3 : (z₁.re * z₂.re + z₁.im * z₂.im) = 0)
  (h4 : abs (z₁ + z₂ - z₃) = 2) :
  ∃ (r : ℝ), r ∈ Set.Icc 0 4 ∧ abs z₃ = r :=
by sorry

end z3_magnitude_range_l2589_258945


namespace intersection_chord_length_l2589_258984

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → ℝ → Prop

/-- Calculates the chord length of intersection between a circle and a line -/
def chordLength (c : PolarCircle) (l : PolarLine) : ℝ := sorry

/-- Main theorem: If a circle ρ = 4cosθ is intersected by a line ρsin(θ - φ) = a 
    with a chord length of 2, then a = 0 or a = -2 -/
theorem intersection_chord_length 
  (c : PolarCircle) 
  (l : PolarLine) 
  (h1 : c.equation = λ ρ θ => ρ = 4 * Real.cos θ)
  (h2 : l.equation = λ ρ θ φ => ρ * Real.sin (θ - φ) = a)
  (h3 : chordLength c l = 2) :
  a = 0 ∨ a = -2 := by sorry

end intersection_chord_length_l2589_258984


namespace range_of_a_l2589_258913

/-- An odd function with period 3 -/
def OddPeriodic3 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 3) = f x)

/-- The main theorem -/
theorem range_of_a (a : ℝ) :
  (∃ f : ℝ → ℝ, OddPeriodic3 f ∧ f 2 > 1 ∧ f 2014 = (2 * a - 3) / (a + 1)) →
  -1 < a ∧ a < 2/3 := by
sorry

end range_of_a_l2589_258913


namespace average_stickers_per_pack_l2589_258902

def sticker_counts : List ℕ := [5, 7, 7, 10, 11]

def num_packs : ℕ := 5

theorem average_stickers_per_pack :
  (sticker_counts.sum / num_packs : ℚ) = 8 := by
  sorry

end average_stickers_per_pack_l2589_258902


namespace tangent_line_implies_a_equals_two_l2589_258914

-- Define the curve and tangent line
def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1
def tangent_line (x : ℝ) : ℝ := 2*x + 1

-- Theorem statement
theorem tangent_line_implies_a_equals_two (a : ℝ) :
  (∃ x₀ : ℝ, curve a x₀ = tangent_line x₀ ∧ 
    ∀ x : ℝ, x ≠ x₀ → curve a x ≠ tangent_line x) →
  a = 2 :=
sorry

end tangent_line_implies_a_equals_two_l2589_258914


namespace lisa_marbles_problem_l2589_258910

/-- The minimum number of additional marbles needed for Lisa to distribute to her friends -/
def minimum_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

theorem lisa_marbles_problem (num_friends : ℕ) (initial_marbles : ℕ) 
  (h1 : num_friends = 12) (h2 : initial_marbles = 50) :
  minimum_additional_marbles num_friends initial_marbles = 28 := by
  sorry

end lisa_marbles_problem_l2589_258910


namespace largest_k_for_distinct_roots_l2589_258957

theorem largest_k_for_distinct_roots : 
  ∃ (k : ℤ), k = 8 ∧ 
  (∀ (x : ℝ), x^2 - 6*x + k = 0 → (∃ (y : ℝ), x ≠ y ∧ y^2 - 6*y + k = 0)) ∧
  (∀ (m : ℤ), m > k → ¬(∀ (x : ℝ), x^2 - 6*x + m = 0 → (∃ (y : ℝ), x ≠ y ∧ y^2 - 6*y + m = 0))) :=
by sorry

end largest_k_for_distinct_roots_l2589_258957


namespace complex_number_equality_l2589_258977

theorem complex_number_equality : ((-1 + Complex.I * Real.sqrt 3) ^ 5) / (1 + Complex.I * Real.sqrt 3) = -16 := by
  sorry

end complex_number_equality_l2589_258977


namespace snow_probability_l2589_258968

theorem snow_probability (p : ℝ) (h : p = 2/3) :
  3 * p^2 * (1 - p) = 4/9 := by
  sorry

end snow_probability_l2589_258968


namespace min_diff_is_one_l2589_258925

-- Define the functions
def f (x : ℤ) : ℝ := 2 * (abs x)
def g (x : ℤ) : ℝ := -(x^2) - 4*x - 1

-- Define the difference function
def diff (x : ℤ) : ℝ := f x - g x

-- Theorem statement
theorem min_diff_is_one :
  ∃ (x : ℤ), diff x = 1 ∧ ∀ (y : ℤ), diff y ≥ 1 :=
sorry

end min_diff_is_one_l2589_258925


namespace order_of_numbers_l2589_258907

/-- Converts a number from base b to base 10 --/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

theorem order_of_numbers :
  let a := toBase10 0x12 16
  let b := toBase10 25 7
  let c := toBase10 33 4
  c < a ∧ a < b := by sorry

end order_of_numbers_l2589_258907


namespace greatest_base_eight_digit_sum_l2589_258983

/-- Represents a positive integer in base 8 --/
def BaseEightRepresentation (n : ℕ+) : List ℕ := sorry

/-- Calculates the sum of digits in a base 8 representation --/
def sumOfDigits (digits : List ℕ) : ℕ := sorry

/-- Theorem stating that the greatest possible sum of digits in base 8 for numbers less than 1728 is 23 --/
theorem greatest_base_eight_digit_sum :
  ∃ (n : ℕ+), n < 1728 ∧
  sumOfDigits (BaseEightRepresentation n) = 23 ∧
  ∀ (m : ℕ+), m < 1728 →
    sumOfDigits (BaseEightRepresentation m) ≤ 23 := by
  sorry

end greatest_base_eight_digit_sum_l2589_258983


namespace round_trip_ticket_percentage_l2589_258952

/-- Given a ship's passenger statistics, calculate the percentage of round-trip ticket holders. -/
theorem round_trip_ticket_percentage
  (total_passengers : ℝ)
  (h1 : total_passengers > 0)
  (h2 : (20 : ℝ) / 100 * total_passengers = (60 : ℝ) / 100 * (round_trip_tickets : ℝ)) :
  (round_trip_tickets : ℝ) / total_passengers = (100 : ℝ) / 3 :=
by sorry

#check round_trip_ticket_percentage

end round_trip_ticket_percentage_l2589_258952


namespace probability_two_heads_in_four_flips_l2589_258962

def coin_flip_probability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

theorem probability_two_heads_in_four_flips :
  coin_flip_probability 4 2 = 3/8 := by
  sorry

end probability_two_heads_in_four_flips_l2589_258962


namespace roots_of_polynomial_l2589_258933

def p (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) := by sorry

end roots_of_polynomial_l2589_258933


namespace expression_simplification_l2589_258943

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4*x + 4) / (x - 1)) = -5 := by
  sorry

end expression_simplification_l2589_258943


namespace negation_of_proposition_l2589_258903

theorem negation_of_proposition :
  (¬ ∀ (a b : ℤ), a = 0 → a * b = 0) ↔ (∃ (a b : ℤ), a = 0 ∧ a * b ≠ 0) := by
  sorry

end negation_of_proposition_l2589_258903


namespace sum_geq_three_over_product_l2589_258944

theorem sum_geq_three_over_product {a b c : ℝ} 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : a + b + c > 1/a + 1/b + 1/c) : 
  a + b + c ≥ 3/(a*b*c) := by
  sorry

end sum_geq_three_over_product_l2589_258944


namespace end_on_multiple_of_four_probability_l2589_258981

def num_cards : ℕ := 12
def move_right_prob : ℚ := 1/2
def move_left_prob : ℚ := 1/4
def move_two_right_prob : ℚ := 1/4

def is_multiple_of_four (n : ℕ) : Prop := ∃ k, n = 4 * k

theorem end_on_multiple_of_four_probability :
  let total_outcomes := num_cards * 4 * 4  -- 12 cards * 4 spinner outcomes * 4 spinner outcomes
  let favorable_outcomes := 21  -- This is derived from the problem constraints
  (favorable_outcomes : ℚ) / total_outcomes = 21 / 192 := by sorry

end end_on_multiple_of_four_probability_l2589_258981


namespace clock_equal_angles_l2589_258917

/-- The time in minutes when the hour and minute hands form equal angles with their positions at 12 o'clock -/
def equal_angle_time : ℚ := 55 + 5/13

/-- The angular speed of the minute hand in degrees per minute -/
def minute_hand_speed : ℚ := 6

/-- The angular speed of the hour hand in degrees per hour -/
def hour_hand_speed : ℚ := 30

theorem clock_equal_angles :
  let t : ℚ := equal_angle_time / 60  -- Convert minutes to hours
  minute_hand_speed * 60 * t = 360 - hour_hand_speed * t := by sorry

#eval equal_angle_time

end clock_equal_angles_l2589_258917


namespace max_m_value_inequality_proof_l2589_258946

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x + 2|

-- Theorem for part (1)
theorem max_m_value (M : ℝ) : (∀ x, f x ≥ |M + 1|) → M ≤ 4 :=
sorry

-- Theorem for part (2)
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + 2*b + c = 4) : 1 / (a + b) + 1 / (b + c) ≥ 1 :=
sorry

end max_m_value_inequality_proof_l2589_258946


namespace fraction_subtraction_simplification_l2589_258990

theorem fraction_subtraction_simplification :
  8 / 21 - 10 / 63 = 2 / 9 := by
  sorry

end fraction_subtraction_simplification_l2589_258990


namespace projection_problem_l2589_258927

def v (z : ℝ) : Fin 3 → ℝ := ![4, -1, z]
def u : Fin 3 → ℝ := ![6, -2, 3]

theorem projection_problem (z : ℝ) : 
  (v z • u) / (u • u) = 20 / 49 → z = -2 := by sorry

end projection_problem_l2589_258927


namespace range_of_a_l2589_258936

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- Define the property that the solution set is not empty
def has_solution (a : ℝ) : Prop := ∃ x : ℝ, f a x < 0

-- Theorem statement
theorem range_of_a (a : ℝ) : has_solution a ↔ a < -4 ∨ a > 4 := by sorry

end range_of_a_l2589_258936


namespace quadratic_equation_solution_l2589_258909

theorem quadratic_equation_solution :
  let x₁ : ℝ := -1 + Real.sqrt 6 / 2
  let x₂ : ℝ := -1 - Real.sqrt 6 / 2
  2 * x₁^2 + 4 * x₁ - 1 = 0 ∧ 2 * x₂^2 + 4 * x₂ - 1 = 0 :=
by sorry

end quadratic_equation_solution_l2589_258909


namespace arithmetic_calculation_l2589_258998

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 = 15 := by sorry

end arithmetic_calculation_l2589_258998


namespace dina_dolls_count_l2589_258992

/-- The number of dolls Ivy has -/
def ivy_dolls : ℕ := 30

/-- The number of collector's edition dolls Ivy has -/
def ivy_collector_dolls : ℕ := 20

/-- The ratio of collector's edition dolls to total dolls for Ivy -/
def collector_ratio : ℚ := 2/3

/-- The number of dolls Dina has -/
def dina_dolls : ℕ := 2 * ivy_dolls

theorem dina_dolls_count : dina_dolls = 60 := by
  sorry

end dina_dolls_count_l2589_258992


namespace at_least_one_first_class_l2589_258971

theorem at_least_one_first_class (n m k : ℕ) (h1 : n = 20) (h2 : m = 16) (h3 : k = 3) :
  (Nat.choose m 1 * Nat.choose (n - m) 2) +
  (Nat.choose m 2 * Nat.choose (n - m) 1) +
  (Nat.choose m 3) = 1136 :=
by sorry

end at_least_one_first_class_l2589_258971


namespace max_display_sum_l2589_258982

def is_valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23

def is_valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def display_sum (h m : ℕ) : ℕ :=
  digit_sum h + digit_sum m

theorem max_display_sum :
  ∃ (h m : ℕ), is_valid_hour h ∧ is_valid_minute m ∧
  ∀ (h' m' : ℕ), is_valid_hour h' → is_valid_minute m' →
  display_sum h' m' ≤ display_sum h m ∧
  display_sum h m = 24 :=
sorry

end max_display_sum_l2589_258982


namespace equivalent_statements_l2589_258993

theorem equivalent_statements (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by
  sorry

end equivalent_statements_l2589_258993


namespace binary_representation_of_89_l2589_258951

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem binary_representation_of_89 :
  decimal_to_binary 89 = [1, 0, 1, 1, 0, 0, 1] := by
  sorry

end binary_representation_of_89_l2589_258951


namespace exists_a_b_counterexample_l2589_258922

theorem exists_a_b_counterexample : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by sorry

end exists_a_b_counterexample_l2589_258922


namespace apple_problem_l2589_258988

theorem apple_problem (older younger : ℕ) 
  (h1 : older - 1 = younger + 1)
  (h2 : older + 1 = 2 * (younger - 1)) :
  older + younger = 12 := by
  sorry

end apple_problem_l2589_258988


namespace equilateral_triangle_cd_l2589_258924

/-- An equilateral triangle with vertices at (0,0), (c,14), and (d,41) has cd = -2208 -/
theorem equilateral_triangle_cd (c d : ℝ) : 
  (Complex.abs (Complex.I * 14 - c - Complex.I * 14) = Complex.abs (Complex.I * 41 - c - Complex.I * 14)) ∧
  (Complex.abs (Complex.I * 41 - 0) = Complex.abs (c + Complex.I * 14 - 0)) ∧
  (Complex.abs (c + Complex.I * 14 - 0) = Complex.abs (Complex.I * 14 - 0)) →
  c * d = -2208 := by
sorry

end equilateral_triangle_cd_l2589_258924


namespace correct_calculation_l2589_258953

theorem correct_calculation (x : ℝ) : (4 * x + 16 = 32) → (x / 4 + 16 = 17) := by
  sorry

end correct_calculation_l2589_258953


namespace max_tuesdays_in_80_days_l2589_258999

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Counts the number of Tuesdays in the first n days of a year -/
def countTuesdays (startDay : DayOfWeek) (n : ℕ) : ℕ :=
  sorry

/-- The maximum number of Tuesdays in the first 80 days of a year is 12 -/
theorem max_tuesdays_in_80_days :
  ∃ (startDay : DayOfWeek), countTuesdays startDay 80 = 12 ∧
  ∀ (d : DayOfWeek), countTuesdays d 80 ≤ 12 :=
sorry

end max_tuesdays_in_80_days_l2589_258999


namespace highest_score_is_174_l2589_258947

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  total_innings : ℕ
  total_runs : ℕ
  highest_score : ℕ
  lowest_score : ℕ

/-- Calculates the average score for a batsman -/
def average_score (stats : BatsmanStats) : ℚ :=
  stats.total_runs / stats.total_innings

/-- Calculates the average score excluding highest and lowest scores -/
def average_score_excluding_extremes (stats : BatsmanStats) : ℚ :=
  (stats.total_runs - stats.highest_score - stats.lowest_score) / (stats.total_innings - 2)

/-- Theorem: Given the conditions, the batsman's highest score is 174 runs -/
theorem highest_score_is_174 (stats : BatsmanStats) :
  stats.total_innings = 40 ∧
  average_score stats = 50 ∧
  stats.highest_score = stats.lowest_score + 172 ∧
  average_score_excluding_extremes stats = 48 →
  stats.highest_score = 174 := by
  sorry

#check highest_score_is_174

end highest_score_is_174_l2589_258947


namespace quiz_sum_l2589_258926

theorem quiz_sum (x y : ℕ+) (h1 : x - y = 4) (h2 : x * y = 104) : x + y = 20 := by
  sorry

end quiz_sum_l2589_258926


namespace sequence_sum_l2589_258980

theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (h : ∀ n, S n = n^3) :
  a 6 + a 7 + a 8 + a 9 = 604 :=
by
  sorry

end sequence_sum_l2589_258980


namespace horner_method_polynomial_evaluation_l2589_258928

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 6*x + 7

theorem horner_method_polynomial_evaluation :
  f 5 = 2677 := by
  sorry

end horner_method_polynomial_evaluation_l2589_258928


namespace jessie_score_is_30_l2589_258979

-- Define the scoring system
def correct_points : ℚ := 2
def incorrect_points : ℚ := -0.5
def unanswered_points : ℚ := 0

-- Define Jessie's answers
def correct_answers : ℕ := 16
def incorrect_answers : ℕ := 4
def unanswered_questions : ℕ := 10

-- Define Jessie's score calculation
def jessie_score : ℚ :=
  (correct_answers : ℚ) * correct_points +
  (incorrect_answers : ℚ) * incorrect_points +
  (unanswered_questions : ℚ) * unanswered_points

-- Theorem to prove
theorem jessie_score_is_30 : jessie_score = 30 := by
  sorry

end jessie_score_is_30_l2589_258979


namespace integer_root_values_l2589_258901

def polynomial (x b : ℤ) : ℤ := x^3 + 2*x^2 + b*x + 8

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-81, -26, -12, -6, 4, 9, 47} := by sorry

end integer_root_values_l2589_258901


namespace slope_angle_of_parametric_line_l2589_258964

/-- The slope angle of a line given by parametric equations -/
theorem slope_angle_of_parametric_line :
  let x : ℝ → ℝ := λ t ↦ 5 - 3 * t
  let y : ℝ → ℝ := λ t ↦ 3 + Real.sqrt 3 * t
  (∃ α : ℝ, α = 150 * π / 180 ∧
    ∀ t : ℝ, (y t - y 0) / (x t - x 0) = Real.tan α) :=
by sorry

end slope_angle_of_parametric_line_l2589_258964


namespace expected_ones_is_half_l2589_258941

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1/6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expected_ones : ℚ := 
  0 * (prob_not_one^num_dice) +
  1 * (num_dice.choose 1 * prob_one * prob_not_one^2) +
  2 * (num_dice.choose 2 * prob_one^2 * prob_not_one) +
  3 * prob_one^num_dice

theorem expected_ones_is_half : expected_ones = 1/2 := by
  sorry

end expected_ones_is_half_l2589_258941


namespace chocolate_chip_difference_l2589_258956

/-- The number of chocolate chips Viviana has exceeds the number Susana has -/
def viviana_more_chocolate (viviana_chocolate susana_chocolate : ℕ) : Prop :=
  viviana_chocolate > susana_chocolate

/-- The problem statement -/
theorem chocolate_chip_difference 
  (viviana_vanilla susana_chocolate : ℕ) 
  (h1 : viviana_vanilla = 20)
  (h2 : susana_chocolate = 25)
  (h3 : ∃ (viviana_chocolate susana_vanilla : ℕ), 
    viviana_more_chocolate viviana_chocolate susana_chocolate ∧
    susana_vanilla = 3 * viviana_vanilla / 4 ∧
    viviana_chocolate + viviana_vanilla + susana_chocolate + susana_vanilla = 90) :
  ∃ (viviana_chocolate : ℕ), viviana_chocolate - susana_chocolate = 5 := by
sorry

end chocolate_chip_difference_l2589_258956


namespace quadratic_completing_square_sum_l2589_258930

theorem quadratic_completing_square_sum (x q t : ℝ) : 
  (9 * x^2 - 54 * x - 36 = 0) →
  ((x + q)^2 = t) →
  (9 * (x + q)^2 = 9 * t) →
  (q + t = 10) := by
sorry

end quadratic_completing_square_sum_l2589_258930


namespace representatives_count_l2589_258937

/-- The number of ways to select representatives from male and female students -/
def selectRepresentatives (numMale numFemale numReps : ℕ) (minMale minFemale : ℕ) : ℕ :=
  (numMale.choose (numReps - minFemale) * numFemale.choose minFemale) +
  (numMale.choose minMale * numFemale.choose (numReps - minMale))

/-- Theorem stating the number of ways to select representatives -/
theorem representatives_count :
  selectRepresentatives 5 4 4 2 1 = 100 := by
  sorry

#eval selectRepresentatives 5 4 4 2 1

end representatives_count_l2589_258937


namespace bad_carrots_count_l2589_258959

theorem bad_carrots_count (olivia_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) 
  (h1 : olivia_carrots = 20)
  (h2 : mom_carrots = 14)
  (h3 : good_carrots = 19) :
  olivia_carrots + mom_carrots - good_carrots = 15 :=
by sorry

end bad_carrots_count_l2589_258959


namespace smallest_n_square_and_cube_l2589_258911

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 5 * x = y^2) → (∃ (z : ℕ), 3 * x = z^3) → x ≥ n) ∧
  n = 45 := by
  sorry

end smallest_n_square_and_cube_l2589_258911


namespace geoffrey_games_l2589_258975

/-- The number of games Geoffrey bought -/
def num_games : ℕ := sorry

/-- The amount of money Geoffrey had before his birthday -/
def initial_money : ℕ := sorry

/-- The cost of each game -/
def game_cost : ℕ := 35

/-- The amount of money Geoffrey received from his grandmother -/
def grandmother_gift : ℕ := 20

/-- The amount of money Geoffrey received from his aunt -/
def aunt_gift : ℕ := 25

/-- The amount of money Geoffrey received from his uncle -/
def uncle_gift : ℕ := 30

/-- The total amount of money Geoffrey has after receiving gifts -/
def total_money : ℕ := 125

/-- The amount of money Geoffrey has left after buying games -/
def money_left : ℕ := 20

theorem geoffrey_games :
  num_games = 3 ∧
  initial_money + grandmother_gift + aunt_gift + uncle_gift = total_money ∧
  total_money - money_left = num_games * game_cost :=
sorry

end geoffrey_games_l2589_258975


namespace number_puzzle_l2589_258906

theorem number_puzzle : ∃ x : ℚ, (x / 6) * 12 = 8 ∧ x = 4 := by
  sorry

end number_puzzle_l2589_258906


namespace sin_sum_of_complex_exponentials_l2589_258920

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = 4/5 + 3/5 * Complex.I →
  Complex.exp (φ * Complex.I) = -5/13 + 12/13 * Complex.I →
  Real.sin (θ + φ) = 84/65 := by
  sorry

end sin_sum_of_complex_exponentials_l2589_258920


namespace cherry_pies_count_l2589_258934

/-- Represents the types of pies --/
inductive PieType
  | Apple
  | Blueberry
  | Cherry

/-- Calculates the number of cherry pies given the total number of pies and the ratio --/
def cherry_pies (total : ℕ) (apple_ratio : ℕ) (blueberry_ratio : ℕ) (cherry_ratio : ℕ) : ℕ :=
  let ratio_sum := apple_ratio + blueberry_ratio + cherry_ratio
  let pies_per_ratio := total / ratio_sum
  cherry_ratio * pies_per_ratio

/-- Theorem stating that given 30 total pies and a 1:5:4 ratio, there are 12 cherry pies --/
theorem cherry_pies_count : cherry_pies 30 1 5 4 = 12 := by
  sorry

end cherry_pies_count_l2589_258934


namespace shekar_average_proof_l2589_258949

def shekar_average_marks (math science social_studies english biology : ℕ) : ℚ :=
  (math + science + social_studies + english + biology : ℚ) / 5

theorem shekar_average_proof :
  shekar_average_marks 76 65 82 67 75 = 73 := by
  sorry

end shekar_average_proof_l2589_258949


namespace greatest_multiple_of_5_and_7_less_than_1000_l2589_258942

theorem greatest_multiple_of_5_and_7_less_than_1000 :
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 7 = 0 → n ≤ 980 :=
by
  sorry

end greatest_multiple_of_5_and_7_less_than_1000_l2589_258942


namespace factorization_proof_l2589_258915

theorem factorization_proof (a b x y : ℝ) : 
  (4 * a^2 * b - 6 * a * b^2 = 2 * a * b * (2 * a - 3 * b)) ∧ 
  (25 * x^2 - 9 * y^2 = (5 * x + 3 * y) * (5 * x - 3 * y)) ∧ 
  (2 * a^2 * b - 8 * a * b^2 + 8 * b^3 = 2 * b * (a - 2 * b)^2) ∧ 
  ((x + 2) * (x - 8) + 25 = (x - 3)^2) :=
by sorry


end factorization_proof_l2589_258915


namespace delores_initial_money_l2589_258916

def computer_cost : ℕ := 400
def printer_cost : ℕ := 40
def money_left : ℕ := 10

theorem delores_initial_money : 
  computer_cost + printer_cost + money_left = 450 := by sorry

end delores_initial_money_l2589_258916


namespace deepak_age_l2589_258965

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 5 / 7 →
  arun_age + 6 = 36 →
  deepak_age = 42 := by
sorry

end deepak_age_l2589_258965


namespace eighteen_player_tournament_l2589_258967

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A round-robin tournament with 18 players has 153 games -/
theorem eighteen_player_tournament : num_games 18 = 153 := by
  sorry

end eighteen_player_tournament_l2589_258967


namespace simultaneous_integer_fractions_l2589_258950

theorem simultaneous_integer_fractions (x : ℤ) :
  (∃ y z : ℤ, (x - 3) / 7 = y ∧ (x - 2) / 5 = z) ↔ 
  (∃ t : ℤ, x = 35 * t + 17) :=
by sorry

end simultaneous_integer_fractions_l2589_258950


namespace second_mixture_percentage_l2589_258955

/-- Represents the composition of an alcohol mixture -/
structure AlcoholMixture where
  volume : ℝ
  percentage : ℝ

/-- Proves that the second mixture has 50% alcohol content -/
theorem second_mixture_percentage
  (total_mixture : AlcoholMixture)
  (first_mixture : AlcoholMixture)
  (h_total_volume : total_mixture.volume = 10)
  (h_total_percentage : total_mixture.percentage = 45)
  (h_first_volume : first_mixture.volume = 2.5)
  (h_first_percentage : first_mixture.percentage = 30)
  : ∃ (second_mixture : AlcoholMixture),
    second_mixture.volume = total_mixture.volume - first_mixture.volume ∧
    second_mixture.percentage = 50 := by
  sorry

end second_mixture_percentage_l2589_258955


namespace sum_of_squares_divisibility_l2589_258961

theorem sum_of_squares_divisibility (n : ℕ) : 
  (∃ k : ℕ, n = 6 * k - 1 ∨ n = 6 * k + 1) ↔ 
  (∃ m : ℕ, n * (n + 1) * (2 * n + 1) = 6 * m) := by
  sorry

end sum_of_squares_divisibility_l2589_258961


namespace intersection_A_B_l2589_258969

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {y | ∃ x, y = Real.exp x}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end intersection_A_B_l2589_258969


namespace distribute_five_balls_four_boxes_l2589_258987

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- There are 5 balls -/
def num_balls : ℕ := 5

/-- There are 4 boxes -/
def num_boxes : ℕ := 4

/-- The theorem stating that there are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes : 
  distribute_balls num_balls num_boxes = 56 := by sorry

end distribute_five_balls_four_boxes_l2589_258987


namespace stone_counting_l2589_258995

theorem stone_counting (n : Nat) (h : n = 99) : n % 16 = 3 := by
  sorry

end stone_counting_l2589_258995


namespace otimes_calculation_l2589_258966

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a^3 - b

-- Theorem statement
theorem otimes_calculation (a : ℝ) : otimes a (otimes a (otimes a a)) = a^3 - a := by
  sorry

end otimes_calculation_l2589_258966


namespace linear_function_proof_l2589_258996

def f (x : ℝ) : ℝ := -x + 1

theorem linear_function_proof :
  (∀ x y t : ℝ, f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y) ∧
  f 0 = 1 ∧
  ∀ x1 x2 : ℝ, x2 > x1 → f x2 < f x1 :=
by sorry

end linear_function_proof_l2589_258996


namespace flower_calculation_l2589_258938

/- Define the initial quantities -/
def initial_roses : ℕ := 36
def initial_chocolates : ℕ := 5
def initial_cupcakes : ℕ := 10
def initial_sunflowers : ℕ := 24

/- Define the trading events -/
def trade_day5 : ℕ × ℕ := (12, 6)  -- (roses, sunflowers)
def trade_day6 : ℕ × ℕ := (12, 20)  -- (roses, cupcakes)
def trade_day7 : ℕ := 15  -- daffodils

/- Define the wilting rates -/
def wilt_rate_day5 : ℚ := 1/10
def wilt_rate_day6_roses : ℚ := 1/5
def wilt_rate_day6_sunflowers : ℚ := 3/10
def wilt_rate_day7_roses : ℚ := 1/4
def wilt_rate_day7_sunflowers : ℚ := 3/20
def wilt_rate_day7_daffodils : ℚ := 1/5

/- Define the function to calculate the number of unwilted flowers -/
def calculate_unwilted_flowers (initial_roses initial_sunflowers : ℕ) 
  (trade_day5 trade_day6 : ℕ × ℕ) (trade_day7 : ℕ)
  (wilt_rate_day5 wilt_rate_day6_roses wilt_rate_day6_sunflowers 
   wilt_rate_day7_roses wilt_rate_day7_sunflowers wilt_rate_day7_daffodils : ℚ) :
  ℕ × ℕ × ℕ := sorry

/- Theorem statement -/
theorem flower_calculation :
  calculate_unwilted_flowers initial_roses initial_sunflowers
    trade_day5 trade_day6 trade_day7
    wilt_rate_day5 wilt_rate_day6_roses wilt_rate_day6_sunflowers
    wilt_rate_day7_roses wilt_rate_day7_sunflowers wilt_rate_day7_daffodils
  = (34, 18, 12) := by sorry

end flower_calculation_l2589_258938


namespace quadratic_properties_l2589_258989

/-- Quadratic function f(x) = 2x^2 + 4x - 6 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 6

/-- Vertex form of f(x) -/
def vertex_form (x : ℝ) : ℝ := 2 * (x + 1)^2 - 8

/-- Axis of symmetry -/
def axis_of_symmetry : ℝ := -1

/-- Vertex coordinates -/
def vertex : ℝ × ℝ := (-1, -8)

theorem quadratic_properties :
  (∀ x, f x = vertex_form x) ∧
  (axis_of_symmetry = -1) ∧
  (vertex = (-1, -8)) := by
  sorry

end quadratic_properties_l2589_258989


namespace triangle_area_squared_l2589_258960

theorem triangle_area_squared (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + CA) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - CA))
  AB = 7 ∧ BC = 9 ∧ CA = 4 → area^2 = 180 := by
  sorry

end triangle_area_squared_l2589_258960


namespace prob_same_group_l2589_258929

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The probability of a student joining any specific group -/
def prob_join_group : ℚ := 1 / num_groups

/-- The total number of possible outcomes for two students joining groups -/
def total_outcomes : ℕ := num_groups * num_groups

/-- The number of outcomes where both students join the same group -/
def same_group_outcomes : ℕ := num_groups

theorem prob_same_group :
  (same_group_outcomes : ℚ) / total_outcomes = 1 / 3 :=
sorry

end prob_same_group_l2589_258929


namespace trumpington_band_max_size_l2589_258954

theorem trumpington_band_max_size :
  ∃ m : ℕ,
    (∀ k : ℕ, 24 * k < 1000 → 24 * k ≤ 24 * m) ∧
    (24 * m < 1000) ∧
    (24 * m % 30 = 6) ∧
    (24 * m = 936) := by
  sorry

end trumpington_band_max_size_l2589_258954


namespace range_of_a_l2589_258986

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 4}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem range_of_a (a : ℝ) : C a ⊆ (A ∩ B) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l2589_258986


namespace tens_digit_of_expression_l2589_258972

-- Define the expression
def expression : ℤ := 2027^2028 - 2029

-- Theorem statement
theorem tens_digit_of_expression :
  (expression / 10) % 10 = 1 :=
by sorry

end tens_digit_of_expression_l2589_258972


namespace print_shop_charge_difference_l2589_258919

/-- The charge difference between two print shops for a given number of copies -/
def charge_difference (price_x price_y : ℚ) (num_copies : ℕ) : ℚ :=
  num_copies * (price_y - price_x)

/-- The price per copy at print shop X -/
def price_x : ℚ := 1.25

/-- The price per copy at print shop Y -/
def price_y : ℚ := 2.75

/-- The number of copies to be printed -/
def num_copies : ℕ := 60

theorem print_shop_charge_difference :
  charge_difference price_x price_y num_copies = 90 := by
  sorry

end print_shop_charge_difference_l2589_258919


namespace polynomial_factorization_l2589_258997

theorem polynomial_factorization (x : ℝ) :
  let P : ℝ → ℝ := λ x => x^8 + x^4 + 1
  (P x = (x^4 + x^2 + 1) * (x^4 - x^2 + 1)) ∧
  (P x = (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + Real.sqrt 3 * x + 1) * (x^2 - Real.sqrt 3 * x + 1)) :=
by sorry

end polynomial_factorization_l2589_258997


namespace erased_number_proof_l2589_258978

theorem erased_number_proof (b : ℕ) (x : ℕ) : 
  3 ≤ b →
  (b - 2) * (b + 3) / 2 - x = 1015 * (b - 3) / 19 →
  x = 805 :=
sorry

end erased_number_proof_l2589_258978


namespace problem_statement_l2589_258932

theorem problem_statement (a x y : ℝ) (h1 : a ≠ x) (h2 : a ≠ y) (h3 : x ≠ y)
  (h4 : Real.sqrt (a * (x - a)) + Real.sqrt (a * (y - a)) = Real.sqrt (x - a) - Real.sqrt (a - y)) :
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1 / 3 := by
  sorry

end problem_statement_l2589_258932


namespace safari_count_l2589_258958

theorem safari_count (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 300) 
  (h2 : total_legs = 710) : ∃ (birds mammals tripeds : ℕ),
  birds + mammals + tripeds = total_heads ∧
  2 * birds + 4 * mammals + 3 * tripeds = total_legs ∧
  birds = 139 := by
  sorry

end safari_count_l2589_258958


namespace max_a_value_l2589_258923

/-- The quadratic function f(x) = ax^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

/-- The maximum possible value of a for the quadratic function f(x) = ax^2 - ax + 1
    such that |f(x)| ≤ 1 for all x in [0, 1] is 8 -/
theorem max_a_value :
  ∃ (a_max : ℝ), a_max = 8 ∧
  (∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |f a x| ≤ 1) →
               a ≤ a_max) ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |f a_max x| ≤ 1) :=
sorry

end max_a_value_l2589_258923


namespace fraction_of_three_fourths_that_is_one_fifth_l2589_258940

theorem fraction_of_three_fourths_that_is_one_fifth (x : ℚ) : x * (3/4 : ℚ) = (1/5 : ℚ) ↔ x = (4/15 : ℚ) := by
  sorry

end fraction_of_three_fourths_that_is_one_fifth_l2589_258940


namespace highest_probability_l2589_258918

-- Define the sample space
variable (Ω : Type)

-- Define the probability measure
variable (P : Set Ω → ℝ)

-- Define events A, B, and C
variable (A B C : Set Ω)

-- State the theorem
theorem highest_probability :
  C ⊆ B → B ⊆ A → P A ≥ P B ∧ P A ≥ P C := by
  sorry

end highest_probability_l2589_258918


namespace alice_bob_meet_l2589_258976

/-- Represents the number of points on the circle -/
def numPoints : ℕ := 18

/-- Represents Alice's movement per turn (clockwise) -/
def aliceMove : ℕ := 7

/-- Represents Bob's movement per turn (counterclockwise) -/
def bobMove : ℕ := 13

/-- Calculates the effective clockwise movement of a player given their movement -/
def effectiveMove (move : ℕ) : ℕ :=
  move % numPoints

/-- Calculates the relative movement between Alice and Bob in one turn -/
def relativeMove : ℤ :=
  (effectiveMove aliceMove : ℤ) - (effectiveMove (numPoints - bobMove) : ℤ)

/-- The number of turns it takes for Alice and Bob to meet -/
def numTurns : ℕ := 9

theorem alice_bob_meet :
  (numTurns : ℤ) * relativeMove % (numPoints : ℤ) = 0 :=
sorry

end alice_bob_meet_l2589_258976


namespace right_triangle_area_l2589_258904

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 10 →
  angle = 45 * π / 180 →
  (1 / 2) * hypotenuse * hypotenuse * Real.sin angle = 25 :=
by sorry

end right_triangle_area_l2589_258904


namespace three_times_more_plus_constant_problem_solution_l2589_258973

theorem three_times_more_plus_constant (base : ℝ) (more : ℕ) (constant : ℝ) :
  (base * (1 + more : ℝ) + constant = base * (more + 1 : ℝ) + constant) := by sorry

theorem problem_solution : 
  (608 : ℝ) * (1 + 3 : ℝ) + 12.8 = 2444.8 := by sorry

end three_times_more_plus_constant_problem_solution_l2589_258973


namespace solve_for_x_l2589_258974

/-- The operation defined for real numbers a, b, c, d -/
def operation (a b c d : ℝ) : ℝ := a * d - b * c

/-- The theorem stating that if the operation on the given matrix equals 2023, then x = 2018 -/
theorem solve_for_x (x : ℝ) : operation (x + 1) (x + 2) (x - 3) (x - 1) = 2023 → x = 2018 := by
  sorry

end solve_for_x_l2589_258974


namespace angle_sum_in_circle_l2589_258970

theorem angle_sum_in_circle (y : ℚ) : 
  (6 * y + 3 * y + y + 4 * y = 360) → y = 180 / 7 := by
  sorry

end angle_sum_in_circle_l2589_258970


namespace parabola_x_intercepts_l2589_258912

/-- The number of x-intercepts of the parabola y = 3x^2 - 4x + 1 -/
theorem parabola_x_intercepts :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 1
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end parabola_x_intercepts_l2589_258912


namespace square_value_when_product_zero_l2589_258948

theorem square_value_when_product_zero (a : ℝ) :
  (a^2 - 3) * (a^2 + 1) = 0 → a^2 = 3 :=
by
  sorry

end square_value_when_product_zero_l2589_258948
