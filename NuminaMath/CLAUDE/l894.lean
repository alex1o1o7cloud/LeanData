import Mathlib

namespace NUMINAMATH_CALUDE_yield_difference_l894_89428

-- Define the initial yields and growth rates
def tomatoes_initial : ℝ := 2073
def corn_initial : ℝ := 4112
def onions_initial : ℝ := 985
def carrots_initial : ℝ := 6250

def tomatoes_growth_rate : ℝ := 0.12
def corn_growth_rate : ℝ := 0.15
def onions_growth_rate : ℝ := 0.08
def carrots_growth_rate : ℝ := 0.10

-- Calculate the yields after growth
def tomatoes_yield : ℝ := tomatoes_initial * (1 + tomatoes_growth_rate)
def corn_yield : ℝ := corn_initial * (1 + corn_growth_rate)
def onions_yield : ℝ := onions_initial * (1 + onions_growth_rate)
def carrots_yield : ℝ := carrots_initial * (1 + carrots_growth_rate)

-- Define the theorem
theorem yield_difference : 
  (max tomatoes_yield (max corn_yield (max onions_yield carrots_yield))) - 
  (min tomatoes_yield (min corn_yield (min onions_yield carrots_yield))) = 5811.2 := by
  sorry

end NUMINAMATH_CALUDE_yield_difference_l894_89428


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l894_89458

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (x : ℕ) : ℝ := 75 + 12 * x

/-- Gamma T-Shirt Company's pricing function -/
def gamma_cost (x : ℕ) : ℝ := 18 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme_cheaper : ℕ := 13

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme_cheaper < gamma_cost min_shirts_for_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_for_acme_cheaper → 
    acme_cost n ≥ gamma_cost n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l894_89458


namespace NUMINAMATH_CALUDE_complex_equidistant_points_l894_89487

theorem complex_equidistant_points : ∃ (z : ℂ), 
  Complex.abs (z - 2) = 3 ∧ 
  Complex.abs (z + 1 + 2*I) = 3 ∧ 
  Complex.abs (z - 3*I) = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equidistant_points_l894_89487


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l894_89498

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l894_89498


namespace NUMINAMATH_CALUDE_weaving_problem_l894_89448

/-- Represents the daily weaving length in an arithmetic sequence -/
def weaving_sequence (initial_length : ℚ) (daily_increase : ℚ) (day : ℕ) : ℚ :=
  initial_length + (day - 1) * daily_increase

/-- Represents the total weaving length over a period of days -/
def total_weaving (initial_length : ℚ) (daily_increase : ℚ) (days : ℕ) : ℚ :=
  (days : ℚ) * initial_length + (days * (days - 1) / 2) * daily_increase

theorem weaving_problem (initial_length daily_increase : ℚ) :
  initial_length = 5 →
  total_weaving initial_length daily_increase 30 = 390 →
  weaving_sequence initial_length daily_increase 5 = 209 / 29 := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l894_89448


namespace NUMINAMATH_CALUDE_counterexample_acute_angles_sum_l894_89449

theorem counterexample_acute_angles_sum : 
  ∃ (A B : ℝ), 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ A + B ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_acute_angles_sum_l894_89449


namespace NUMINAMATH_CALUDE_average_marks_abc_l894_89403

theorem average_marks_abc (M : ℝ) (D : ℝ) :
  -- The average marks of a, b, c is M
  -- When d joins, the average becomes 47
  3 * M + D = 4 * 47 →
  -- The average marks of b, c, d, e is 48
  -- E has 3 more marks than d
  -- The marks of a is 43
  (3 * M - 43) + D + (D + 3) = 4 * 48 →
  -- The average marks of a, b, c is 48
  M = 48 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_abc_l894_89403


namespace NUMINAMATH_CALUDE_complex_expression_equals_9980_l894_89462

theorem complex_expression_equals_9980 : 
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_9980_l894_89462


namespace NUMINAMATH_CALUDE_circumscribed_circle_properties_l894_89454

/-- Triangle with vertices A(1,4), B(-2,3), and C(4,-5) -/
structure Triangle where
  A : ℝ × ℝ := (1, 4)
  B : ℝ × ℝ := (-2, 3)
  C : ℝ × ℝ := (4, -5)

/-- Circumscribed circle of a triangle -/
structure CircumscribedCircle (t : Triangle) where
  /-- Equation of the circle in the form ax^2 + ay^2 + bx + cy + d = 0 -/
  equation : ℝ → ℝ → ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem about the circumscribed circle of the specific triangle -/
theorem circumscribed_circle_properties (t : Triangle) :
  ∃ (c : CircumscribedCircle t),
    (∀ x y, c.equation x y = x^2 + y^2 - 2*x + 2*y - 23) ∧
    c.center = (1, -1) ∧
    c.radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_properties_l894_89454


namespace NUMINAMATH_CALUDE_max_b_value_l894_89434

/-- The function f(x) = ax^3 + bx^2 - a^2x -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 - a^2 * x

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x - a^2

theorem max_b_value (a b : ℝ) (x₁ x₂ : ℝ) (ha : a > 0) (hx : x₁ ≠ x₂)
  (hextreme : f' a b x₁ = 0 ∧ f' a b x₂ = 0)
  (hsum : abs x₁ + abs x₂ = 2 * Real.sqrt 2) :
  b ≤ 4 * Real.sqrt 6 ∧ ∃ b₀, b₀ = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_max_b_value_l894_89434


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l894_89490

theorem right_triangle_leg_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a = 8 → c = 17 →
  b = 15 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l894_89490


namespace NUMINAMATH_CALUDE_set_difference_equals_singleton_l894_89424

def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2002}

def N : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2003}

theorem set_difference_equals_singleton : N \ M = {2003} := by
  sorry

end NUMINAMATH_CALUDE_set_difference_equals_singleton_l894_89424


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_l894_89457

theorem not_p_and_q_implies_at_most_one (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_l894_89457


namespace NUMINAMATH_CALUDE_bounded_recurrence_sequence_is_constant_two_l894_89451

/-- A sequence of natural numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))

/-- A sequence is bounded if there exists an upper bound for all its terms -/
def BoundedSequence (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n : ℕ, a n ≤ M

theorem bounded_recurrence_sequence_is_constant_two (a : ℕ → ℕ) 
  (h_recurrence : RecurrenceSequence a) (h_bounded : BoundedSequence a) :
  ∀ n : ℕ, a n = 2 := by
  sorry

end NUMINAMATH_CALUDE_bounded_recurrence_sequence_is_constant_two_l894_89451


namespace NUMINAMATH_CALUDE_price_reduction_achieves_desired_profit_l894_89435

/-- Represents the profit and sales scenario for black pork zongzi --/
structure ZongziSales where
  initialProfit : ℝ  -- Initial profit per box
  initialQuantity : ℝ  -- Initial quantity sold
  priceElasticity : ℝ  -- Additional boxes sold per dollar of price reduction
  priceReduction : ℝ  -- Amount of price reduction per box
  desiredTotalProfit : ℝ  -- Desired total profit

/-- Calculates the new profit per box after price reduction --/
def newProfitPerBox (s : ZongziSales) : ℝ :=
  s.initialProfit - s.priceReduction

/-- Calculates the new quantity sold after price reduction --/
def newQuantitySold (s : ZongziSales) : ℝ :=
  s.initialQuantity + s.priceElasticity * s.priceReduction

/-- Calculates the total profit after price reduction --/
def totalProfit (s : ZongziSales) : ℝ :=
  newProfitPerBox s * newQuantitySold s

/-- Theorem stating that a price reduction of 15 achieves the desired total profit --/
theorem price_reduction_achieves_desired_profit (s : ZongziSales)
  (h1 : s.initialProfit = 50)
  (h2 : s.initialQuantity = 50)
  (h3 : s.priceElasticity = 2)
  (h4 : s.priceReduction = 15)
  (h5 : s.desiredTotalProfit = 2800) :
  totalProfit s = s.desiredTotalProfit := by
  sorry

#eval totalProfit { initialProfit := 50, initialQuantity := 50, priceElasticity := 2, priceReduction := 15, desiredTotalProfit := 2800 }

end NUMINAMATH_CALUDE_price_reduction_achieves_desired_profit_l894_89435


namespace NUMINAMATH_CALUDE_expression_evaluation_l894_89415

theorem expression_evaluation :
  let x : ℤ := -1
  let y : ℤ := 2
  let A : ℤ := 2*x + y
  let B : ℤ := 2*x - y
  (A^2 - B^2) * (x - 2*y) = 80 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l894_89415


namespace NUMINAMATH_CALUDE_phone_number_proof_l894_89411

def is_harmonic_mean (a b c : ℕ) : Prop :=
  2 * b * a * c = b * (a + c)

def is_six_digit (a b c d : ℕ) : Prop :=
  100000 ≤ a * 100000 + b * 10000 + c * 100 + d ∧
  a * 100000 + b * 10000 + c * 100 + d < 1000000

theorem phone_number_proof (a b c d : ℕ) : 
  a = 6 ∧ b = 8 ∧ c = 12 ∧ d = 24 →
  a < b ∧ b < c ∧ c < d ∧
  is_harmonic_mean a b c ∧
  is_harmonic_mean b c d ∧
  is_six_digit a b c d := by
  sorry

#eval [6, 8, 12, 24].map (λ x => x.toDigits 10)

end NUMINAMATH_CALUDE_phone_number_proof_l894_89411


namespace NUMINAMATH_CALUDE_absolute_value_equation_range_l894_89471

theorem absolute_value_equation_range :
  ∀ x : ℝ, (|3*x - 2| + |3*x + 1| = 3) ↔ (-1/3 ≤ x ∧ x ≤ 2/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_range_l894_89471


namespace NUMINAMATH_CALUDE_randy_money_problem_l894_89491

theorem randy_money_problem (M : ℝ) : 
  M > 0 →
  (1/4 : ℝ) * (M - 10) = 5 →
  M = 30 := by
sorry

end NUMINAMATH_CALUDE_randy_money_problem_l894_89491


namespace NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l894_89442

-- Problem 1
theorem problem_one : 6 + (-8) - (-5) = 3 := by sorry

-- Problem 2
theorem problem_two : 5 + 3/5 + (-5 - 2/3) + 4 + 2/5 + (-1/3) = 4 := by sorry

-- Problem 3
theorem problem_three : (-1/2 + 1/6 - 1/4) * 12 = -7 := by sorry

-- Problem 4
theorem problem_four : -1^2022 + 27 * (-1/3)^2 - |(-5)| = -3 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l894_89442


namespace NUMINAMATH_CALUDE_air_quality_probability_l894_89474

theorem air_quality_probability (p_single : ℝ) (p_consecutive : ℝ) (p_next : ℝ) : 
  p_single = 0.75 → p_consecutive = 0.6 → p_next = p_consecutive / p_single → p_next = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probability_l894_89474


namespace NUMINAMATH_CALUDE_problem_statement_l894_89466

theorem problem_statement : (-5)^5 / 5^3 + 3^4 - 6^1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l894_89466


namespace NUMINAMATH_CALUDE_simplify_exponents_l894_89431

theorem simplify_exponents (t s : ℝ) : (t^2 * t^5) * s^3 = t^7 * s^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponents_l894_89431


namespace NUMINAMATH_CALUDE_black_white_pieces_difference_l894_89439

theorem black_white_pieces_difference (B W : ℕ) : 
  (B - 1) / W = 9 / 7 →
  B / (W - 1) = 7 / 5 →
  B - W = 7 :=
by sorry

end NUMINAMATH_CALUDE_black_white_pieces_difference_l894_89439


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l894_89409

theorem quadratic_solution_difference_squared : 
  ∀ α β : ℝ, 
  (α^2 = 2*α + 1) → 
  (β^2 = 2*β + 1) → 
  (α ≠ β) → 
  (α - β)^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l894_89409


namespace NUMINAMATH_CALUDE_quadratic_inequality_requires_conditional_branch_l894_89441

/-- Represents an algorithm --/
inductive Algorithm
  | ProductOfTwoNumbers
  | DistancePointToLine
  | QuadraticInequality
  | TrapezoidArea

/-- Determines if an algorithm requires a conditional branch structure --/
def requires_conditional_branch (a : Algorithm) : Prop :=
  match a with
  | Algorithm.QuadraticInequality => True
  | _ => False

/-- Theorem stating that only solving a quadratic inequality requires a conditional branch structure --/
theorem quadratic_inequality_requires_conditional_branch :
  ∀ (a : Algorithm), requires_conditional_branch a ↔ a = Algorithm.QuadraticInequality :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_requires_conditional_branch_l894_89441


namespace NUMINAMATH_CALUDE_no_both_squares_l894_89470

theorem no_both_squares : ¬∃ (x y : ℕ+), 
  ∃ (a b : ℕ+), (x^2 + 2*y : ℕ) = a^2 ∧ (y^2 + 2*x : ℕ) = b^2 := by
  sorry

end NUMINAMATH_CALUDE_no_both_squares_l894_89470


namespace NUMINAMATH_CALUDE_divisor_count_relation_l894_89438

-- Define a function to count divisors
def count_divisors (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem divisor_count_relation (n : ℕ) :
  n > 0 → count_divisors (210 * n^3) = 210 → count_divisors (64 * n^5) = 22627 :=
by sorry

end NUMINAMATH_CALUDE_divisor_count_relation_l894_89438


namespace NUMINAMATH_CALUDE_reading_time_difference_problem_l894_89492

/-- The difference in reading time between two people reading the same book -/
def reading_time_difference (ken_speed lisa_speed book_pages : ℕ) : ℕ :=
  let ken_time := book_pages / ken_speed
  let lisa_time := book_pages / lisa_speed
  (lisa_time - ken_time) * 60

/-- Theorem stating the difference in reading time for the given problem -/
theorem reading_time_difference_problem :
  reading_time_difference 75 60 360 = 72 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_problem_l894_89492


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l894_89430

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2
  arith_def : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Given conditions on the arithmetic sequence imply S₁₀ = 65 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.a 3 = 4)
    (h2 : seq.S 9 - seq.S 6 = 27) :
  seq.S 10 = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l894_89430


namespace NUMINAMATH_CALUDE_multiply_sum_problem_l894_89460

theorem multiply_sum_problem (x : ℝ) (h : x = 62.5) :
  ∃! y : ℝ, ((x + 5) * y / 5) - 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_multiply_sum_problem_l894_89460


namespace NUMINAMATH_CALUDE_triangle_ad_length_l894_89401

/-- Triangle ABC with perpendicular from A to BC at point D -/
structure Triangle :=
  (A B C D : ℝ × ℝ)
  (AB : ℝ)
  (AC : ℝ)
  (BD : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (is_right_angle : (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0)
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = AB)
  (AC_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = AC)
  (BD_CD_ratio : BD / CD = 2 / 5)

/-- Theorem: In triangle ABC, if AB = 10, AC = 17, D is the foot of the perpendicular from A to BC,
    and BD:CD = 2:5, then AD = 8 -/
theorem triangle_ad_length (t : Triangle) (h1 : t.AB = 10) (h2 : t.AC = 17) : t.AD = 8 := by
  sorry


end NUMINAMATH_CALUDE_triangle_ad_length_l894_89401


namespace NUMINAMATH_CALUDE_lowest_degree_polynomial_l894_89450

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := ℕ → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- The set of coefficients of a polynomial -/
def coefficients (p : IntPolynomial) : Set ℤ := sorry

/-- Predicate for a polynomial satisfying the given conditions -/
def satisfies_conditions (p : IntPolynomial) : Prop :=
  ∃ b : ℤ, (∃ x ∈ coefficients p, x < b) ∧
            (∃ y ∈ coefficients p, y > b) ∧
            b ∉ coefficients p

/-- The main theorem -/
theorem lowest_degree_polynomial :
  ∃ p : IntPolynomial, satisfies_conditions p ∧
    degree p = 4 ∧
    ∀ q : IntPolynomial, satisfies_conditions q → degree q ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_lowest_degree_polynomial_l894_89450


namespace NUMINAMATH_CALUDE_least_band_members_l894_89478

/-- Represents the target ratio for each instrument -/
def target_ratio : Vector ℕ 5 := ⟨[5, 3, 6, 2, 4], by rfl⟩

/-- Represents the minimum number of successful candidates for each instrument -/
def min_candidates : Vector ℕ 5 := ⟨[16, 15, 20, 2, 12], by rfl⟩

/-- Checks if a given number of band members satisfies the target ratio and minimum requirements -/
def satisfies_requirements (total_members : ℕ) : Prop :=
  ∃ (x : ℕ), x > 0 ∧
    (∀ i : Fin 5, 
      (target_ratio.get i) * x ≥ min_candidates.get i) ∧
    (target_ratio.get 0) * x + 
    (target_ratio.get 1) * x + 
    (target_ratio.get 2) * x + 
    (target_ratio.get 3) * x + 
    (target_ratio.get 4) * x = total_members

/-- The main theorem stating that 100 is the least number of total band members satisfying the requirements -/
theorem least_band_members : 
  satisfies_requirements 100 ∧ 
  (∀ n : ℕ, n < 100 → ¬satisfies_requirements n) :=
sorry

end NUMINAMATH_CALUDE_least_band_members_l894_89478


namespace NUMINAMATH_CALUDE_consecutive_products_sum_l894_89404

theorem consecutive_products_sum : ∃ (a b c d e : ℕ), 
  (b = a + 1) ∧ 
  (d = c + 1) ∧ 
  (e = d + 1) ∧ 
  (a * b = 210) ∧ 
  (c * d * e = 210) ∧ 
  (a + b + c + d + e = 47) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_products_sum_l894_89404


namespace NUMINAMATH_CALUDE_transistors_in_2000_l894_89463

/-- Moore's law states that the number of transistors doubles every two years -/
def moores_law (t : ℕ) : ℕ := 2^(t/2)

/-- The number of transistors in a typical CPU in 1990 -/
def transistors_1990 : ℕ := 1000000

/-- The year we're calculating for -/
def target_year : ℕ := 2000

/-- The starting year -/
def start_year : ℕ := 1990

theorem transistors_in_2000 : 
  transistors_1990 * moores_law (target_year - start_year) = 32000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_2000_l894_89463


namespace NUMINAMATH_CALUDE_log_sum_theorem_l894_89479

theorem log_sum_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 4) 
  (h2 : x * y = 64) : 
  (x + y) / 2 = (64^(3/(5+Real.sqrt 3)) + 64^(1/(5+Real.sqrt 3))) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_sum_theorem_l894_89479


namespace NUMINAMATH_CALUDE_prob_even_card_l894_89412

/-- The probability of drawing a card with an even number from a set of cards -/
theorem prob_even_card (total_cards : ℕ) (even_cards : ℕ) 
  (h1 : total_cards = 6) 
  (h2 : even_cards = 3) : 
  (even_cards : ℚ) / total_cards = 1 / 2 := by
  sorry

#check prob_even_card

end NUMINAMATH_CALUDE_prob_even_card_l894_89412


namespace NUMINAMATH_CALUDE_binary_101110_to_octal_56_l894_89444

def binary_to_octal (b : List Bool) : Nat :=
  let binary_to_decimal := b.foldl (λ acc x => 2 * acc + if x then 1 else 0) 0
  let decimal_to_octal := binary_to_decimal.digits 8
  decimal_to_octal.foldl (λ acc x => 10 * acc + x) 0

theorem binary_101110_to_octal_56 :
  binary_to_octal [true, false, true, true, true, false] = 56 := by
  sorry

end NUMINAMATH_CALUDE_binary_101110_to_octal_56_l894_89444


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l894_89419

def M : Set ℤ := {0, 1, 2, 3}
def N : Set ℤ := {-1, 1}

theorem intersection_of_M_and_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l894_89419


namespace NUMINAMATH_CALUDE_smallest_base_for_100_in_three_digits_l894_89410

theorem smallest_base_for_100_in_three_digits :
  ∃ (b : ℕ), b = 5 ∧ b^2 ≤ 100 ∧ 100 < b^3 ∧ ∀ (x : ℕ), x < b → (x^2 ≤ 100 → 100 ≥ x^3) :=
sorry

end NUMINAMATH_CALUDE_smallest_base_for_100_in_three_digits_l894_89410


namespace NUMINAMATH_CALUDE_doubling_points_properties_l894_89486

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be a "doubling point" of another
def isDoublingPoint (p q : Point) : Prop :=
  2 * (p.x + q.x) = p.y + q.y

-- Given point P₁
def P₁ : Point := ⟨2, 0⟩

-- Theorem statement
theorem doubling_points_properties :
  -- 1. Q₁ and Q₂ are doubling points of P₁
  (isDoublingPoint P₁ ⟨2, 8⟩ ∧ isDoublingPoint P₁ ⟨-3, -2⟩) ∧
  -- 2. A(-2, 0) on y = x + 2 is a doubling point of P₁
  (∃ A : Point, A.y = A.x + 2 ∧ A.x = -2 ∧ A.y = 0 ∧ isDoublingPoint P₁ A) ∧
  -- 3. Two points on y = x² - 2x - 3 are doubling points of P₁
  (∃ B C : Point, B ≠ C ∧
    B.y = B.x^2 - 2*B.x - 3 ∧ C.y = C.x^2 - 2*C.x - 3 ∧
    isDoublingPoint P₁ B ∧ isDoublingPoint P₁ C) ∧
  -- 4. Minimum distance to any doubling point is 8√5/5
  (∃ minDist : ℝ, minDist = 8 * Real.sqrt 5 / 5 ∧
    ∀ Q : Point, isDoublingPoint P₁ Q →
      Real.sqrt ((Q.x - P₁.x)^2 + (Q.y - P₁.y)^2) ≥ minDist) :=
by sorry

end NUMINAMATH_CALUDE_doubling_points_properties_l894_89486


namespace NUMINAMATH_CALUDE_mary_biking_time_l894_89472

def total_away_time : ℕ := 570 -- 9.5 hours in minutes
def class_time : ℕ := 45
def num_classes : ℕ := 7
def lunch_time : ℕ := 40
def additional_time : ℕ := 105 -- 1 hour 45 minutes in minutes

def total_school_time : ℕ := class_time * num_classes + lunch_time + additional_time

theorem mary_biking_time :
  total_away_time - total_school_time = 110 :=
sorry

end NUMINAMATH_CALUDE_mary_biking_time_l894_89472


namespace NUMINAMATH_CALUDE_marshmallow_roasting_l894_89473

/-- The number of marshmallows Joe's dad has -/
def dads_marshmallows : ℕ := 21

/-- The number of marshmallows Joe has -/
def joes_marshmallows : ℕ := 4 * dads_marshmallows

/-- The number of marshmallows Joe's dad roasts -/
def dads_roasted : ℕ := dads_marshmallows / 3

/-- The number of marshmallows Joe roasts -/
def joes_roasted : ℕ := joes_marshmallows / 2

/-- The total number of marshmallows roasted -/
def total_roasted : ℕ := dads_roasted + joes_roasted

theorem marshmallow_roasting :
  total_roasted = 49 := by sorry

end NUMINAMATH_CALUDE_marshmallow_roasting_l894_89473


namespace NUMINAMATH_CALUDE_circle_passes_through_origin_l894_89453

/-- A circle in the 2D plane -/
structure Circle where
  a : ℝ  -- x-coordinate of the center
  b : ℝ  -- y-coordinate of the center
  r : ℝ  -- radius

/-- Predicate to check if a point (x, y) is on the circle -/
def onCircle (c : Circle) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y - c.b)^2 = c.r^2

/-- Theorem: A circle passes through the origin iff a^2 + b^2 = r^2 -/
theorem circle_passes_through_origin (c : Circle) :
  onCircle c 0 0 ↔ c.a^2 + c.b^2 = c.r^2 := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_origin_l894_89453


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l894_89485

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l894_89485


namespace NUMINAMATH_CALUDE_interest_rate_is_five_percent_l894_89427

/-- Calculates the interest rate given the principal, time, and simple interest -/
def calculate_interest_rate (principal time simple_interest : ℚ) : ℚ :=
  (simple_interest * 100) / (principal * time)

/-- Proof that the interest rate is 5% given the specified conditions -/
theorem interest_rate_is_five_percent :
  let principal : ℚ := 16065
  let time : ℚ := 5
  let simple_interest : ℚ := 4016.25
  calculate_interest_rate principal time simple_interest = 5 := by
  sorry

#eval calculate_interest_rate 16065 5 4016.25

end NUMINAMATH_CALUDE_interest_rate_is_five_percent_l894_89427


namespace NUMINAMATH_CALUDE_simplify_expression_l894_89467

theorem simplify_expression (x : ℝ) : 2*x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -4*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l894_89467


namespace NUMINAMATH_CALUDE_most_suitable_sampling_method_l894_89432

/-- Represents the age groups in the population --/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | ExcludeOneElderlyThenStratified

/-- Represents the population composition --/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Determines if a sampling method is suitable for a given population and sample size --/
def isSuitableMethod (pop : Population) (sampleSize : Nat) (method : SamplingMethod) : Prop :=
  sorry

/-- The theorem stating that excluding one elderly person and then using stratified sampling
    is the most suitable method for the given population and sample size --/
theorem most_suitable_sampling_method
  (pop : Population)
  (h1 : pop.elderly = 28)
  (h2 : pop.middleAged = 54)
  (h3 : pop.young = 81)
  (sampleSize : Nat)
  (h4 : sampleSize = 36) :
  isSuitableMethod pop sampleSize SamplingMethod.ExcludeOneElderlyThenStratified ∧
  ∀ m : SamplingMethod,
    isSuitableMethod pop sampleSize m →
    m = SamplingMethod.ExcludeOneElderlyThenStratified :=
  sorry


end NUMINAMATH_CALUDE_most_suitable_sampling_method_l894_89432


namespace NUMINAMATH_CALUDE_gymnastics_performance_participants_l894_89416

/-- The number of grades participating in the gymnastics performance -/
def num_grades : ℕ := 3

/-- The number of classes in each grade -/
def classes_per_grade : ℕ := 4

/-- The number of participants selected from each class -/
def participants_per_class : ℕ := 15

/-- The total number of participants in the gymnastics performance -/
def total_participants : ℕ := num_grades * classes_per_grade * participants_per_class

theorem gymnastics_performance_participants : total_participants = 180 := by
  sorry

end NUMINAMATH_CALUDE_gymnastics_performance_participants_l894_89416


namespace NUMINAMATH_CALUDE_median_of_special_list_l894_89456

/-- The sum of integers from 1 to n -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list length -/
def list_length : ℕ := triangular_number 150

/-- The median position -/
def median_position : ℕ := (list_length + 1) / 2

/-- The cumulative count up to n -/
def cumulative_count (n : ℕ) : ℕ := triangular_number n

theorem median_of_special_list : ∃ (n : ℕ), n = 106 ∧ 
  cumulative_count (n - 1) < median_position ∧ 
  cumulative_count n ≥ median_position := by sorry

end NUMINAMATH_CALUDE_median_of_special_list_l894_89456


namespace NUMINAMATH_CALUDE_competition_results_l894_89481

def team_a_scores : List ℝ := [7, 8, 9, 7, 10, 10, 9, 10, 10, 10]
def team_b_scores : List ℝ := [10, 8, 7, 9, 8, 10, 10, 9, 10, 9]

def median (scores : List ℝ) : ℝ := sorry
def mode (scores : List ℝ) : ℝ := sorry
def average (scores : List ℝ) : ℝ := sorry
def variance (scores : List ℝ) : ℝ := sorry

theorem competition_results :
  median team_a_scores = 9.5 ∧
  mode team_b_scores = 10 ∧
  average team_b_scores = 9 ∧
  variance team_b_scores = 1 ∧
  variance team_a_scores = 1.4 ∧
  variance team_b_scores < variance team_a_scores :=
by sorry

end NUMINAMATH_CALUDE_competition_results_l894_89481


namespace NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l894_89407

theorem odd_sum_of_squares_implies_odd_sum (n m : ℤ) :
  Odd (n^2 + m^2) → Odd (n + m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l894_89407


namespace NUMINAMATH_CALUDE_a_minus_c_equals_296_l894_89495

theorem a_minus_c_equals_296 (A B C : ℤ) 
  (h1 : A = B - 397)
  (h2 : A = 742)
  (h3 : B = C + 693) : 
  A - C = 296 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_c_equals_296_l894_89495


namespace NUMINAMATH_CALUDE_complex_modulus_l894_89425

theorem complex_modulus (z : ℂ) : i * z = Real.sqrt 2 - i → Complex.abs z = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l894_89425


namespace NUMINAMATH_CALUDE_rectangle_area_is_twelve_l894_89414

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle defined by its four vertices -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The area of a rectangle -/
def rectangleArea (rect : Rectangle) : ℝ :=
  (rect.B.x - rect.A.x) * (rect.C.y - rect.B.y)

theorem rectangle_area_is_twelve :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨3, 0⟩
  let C : Point := ⟨3, 4⟩
  let D : Point := ⟨0, 4⟩
  let rect : Rectangle := ⟨A, B, C, D⟩
  rectangleArea rect = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_twelve_l894_89414


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l894_89480

theorem sum_of_two_numbers (s l : ℝ) : 
  s = 10.0 → 
  7 * s = 5 * l → 
  s + l = 24.0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l894_89480


namespace NUMINAMATH_CALUDE_amalie_coin_spending_l894_89476

/-- Proof that Amalie spends 3/4 of her coins on toys -/
theorem amalie_coin_spending :
  ∀ (elsa_coins amalie_coins : ℕ),
    -- The ratio of Elsa's coins to Amalie's coins is 10:45
    elsa_coins * 45 = amalie_coins * 10 →
    -- The total number of coins they have is 440
    elsa_coins + amalie_coins = 440 →
    -- Amalie remains with 90 coins after spending
    ∃ (spent_coins : ℕ),
      spent_coins ≤ amalie_coins ∧
      amalie_coins - spent_coins = 90 →
    -- The fraction of coins Amalie spends on toys is 3/4
    (spent_coins : ℚ) / amalie_coins = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_amalie_coin_spending_l894_89476


namespace NUMINAMATH_CALUDE_envelope_equals_cycloid_l894_89484

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle rolling along the x-axis -/
structure RollingCircle where
  radius : ℝ
  center : Point2D

/-- Represents a cycloid curve -/
def Cycloid := ℝ → Point2D

/-- Generates the cycloid traced by a point on the circumference of a circle -/
def circumferenceCycloid (radius : ℝ) : Cycloid := sorry

/-- Generates the envelope of a diameter of a rolling circle -/
def diameterEnvelope (radius : ℝ) : Cycloid := sorry

/-- Theorem stating that the envelope of a diameter is identical to the cycloid traced by a point on the circumference -/
theorem envelope_equals_cycloid (a : ℝ) :
  diameterEnvelope a = circumferenceCycloid (a / 2) := by sorry

end NUMINAMATH_CALUDE_envelope_equals_cycloid_l894_89484


namespace NUMINAMATH_CALUDE_inverse_of_M_l894_89493

/-- The line 2x - y = 3 -/
def line (x y : ℝ) : Prop := 2 * x - y = 3

/-- The matrix M -/
def M (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![-1, a],
    ![b, 3]]

/-- M maps the line onto itself -/
def M_maps_line (a b : ℝ) : Prop :=
  ∀ x y : ℝ, line x y → line (-x + a*y) (b*x + 3*y)

/-- The inverse of M -/
def M_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, -1],
    ![4, -1]]

theorem inverse_of_M (a b : ℝ) (h : M_maps_line a b) :
  M a b * M_inv = 1 ∧ M_inv * M a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_M_l894_89493


namespace NUMINAMATH_CALUDE_incorrect_operation_l894_89452

theorem incorrect_operation (a b c d e f : ℝ) 
  (h1 : a = Real.sqrt 2)
  (h2 : b = Real.sqrt 3)
  (h3 : c = Real.sqrt 5)
  (h4 : d = Real.sqrt 6)
  (h5 : e = Real.sqrt (1/2))
  (h6 : f = Real.sqrt 8)
  (prop1 : a * b = d)
  (prop2 : a / e = 2)
  (prop3 : a + f = 3 * a) :
  a + b ≠ c := by
  sorry

end NUMINAMATH_CALUDE_incorrect_operation_l894_89452


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_eleven_twelfths_l894_89437

theorem sum_of_solutions_eq_eleven_twelfths :
  let f : ℝ → ℝ := λ x ↦ (4*x + 7)*(3*x - 8) + 12
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x + y = 11/12) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_eleven_twelfths_l894_89437


namespace NUMINAMATH_CALUDE_second_bucket_contents_l894_89496

def bucket_contents : List ℕ := [11, 13, 12, 16, 10]

theorem second_bucket_contents (h : ∃ x ∈ bucket_contents, x + 10 = 23) :
  (List.sum bucket_contents) - 23 = 39 := by
  sorry

end NUMINAMATH_CALUDE_second_bucket_contents_l894_89496


namespace NUMINAMATH_CALUDE_max_expression_value_l894_89426

def expression (x y z w : ℕ) : ℕ := x * y^z - w

theorem max_expression_value :
  ∃ (x y z w : ℕ),
    x ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    y ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    z ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    w ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
    expression x y z w = 24 ∧
    ∀ (a b c d : ℕ),
      a ∈ ({0, 1, 2, 3} : Set ℕ) →
      b ∈ ({0, 1, 2, 3} : Set ℕ) →
      c ∈ ({0, 1, 2, 3} : Set ℕ) →
      d ∈ ({0, 1, 2, 3} : Set ℕ) →
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
      expression a b c d ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_max_expression_value_l894_89426


namespace NUMINAMATH_CALUDE_percentage_increase_l894_89489

theorem percentage_increase (w : ℝ) (P : ℝ) : 
  w = 80 →
  (w + P / 100 * w) - (w - 25 / 100 * w) = 30 →
  P = 12.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l894_89489


namespace NUMINAMATH_CALUDE_gcd_8008_12012_l894_89420

theorem gcd_8008_12012 : Nat.gcd 8008 12012 = 4004 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8008_12012_l894_89420


namespace NUMINAMATH_CALUDE_indeterminate_roots_l894_89433

theorem indeterminate_roots (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_equal_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ 
    ∀ y : ℝ, a * y^2 + b * y + c = 0 → y = x) :
  ¬∃ (root_nature : Prop), 
    (∀ x : ℝ, (a + 1) * x^2 + (b + 2) * x + (c + 1) = 0 ↔ root_nature) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_roots_l894_89433


namespace NUMINAMATH_CALUDE_problem_statement_l894_89464

theorem problem_statement : 
  let a := ((7 + 4 * Real.sqrt 3)^(1/2) - (7 - 4 * Real.sqrt 3)^(1/2)) / Real.sqrt 3
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l894_89464


namespace NUMINAMATH_CALUDE_sqrt_one_implies_one_l894_89447

theorem sqrt_one_implies_one (a : ℝ) : Real.sqrt a = 1 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_implies_one_l894_89447


namespace NUMINAMATH_CALUDE_max_value_x_plus_inverse_l894_89469

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_inverse_l894_89469


namespace NUMINAMATH_CALUDE_square_area_26_l894_89461

/-- The area of a square with vertices at (0, 0), (-5, -1), (-4, -6), and (1, -5) is 26 square units. -/
theorem square_area_26 : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (-5, -1)
  let C : ℝ × ℝ := (-4, -6)
  let D : ℝ × ℝ := (1, -5)
  let square_area := (B.1 - A.1)^2 + (B.2 - A.2)^2
  square_area = 26 := by
  sorry


end NUMINAMATH_CALUDE_square_area_26_l894_89461


namespace NUMINAMATH_CALUDE_probability_ace_of_hearts_fifth_l894_89475

-- Define the number of cards in a standard deck
def standard_deck_size : ℕ := 52

-- Define a function to calculate the probability of a specific card in a specific position
def probability_specific_card_in_position (deck_size : ℕ) : ℚ :=
  1 / deck_size

-- Theorem statement
theorem probability_ace_of_hearts_fifth : 
  probability_specific_card_in_position standard_deck_size = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_probability_ace_of_hearts_fifth_l894_89475


namespace NUMINAMATH_CALUDE_johns_allowance_l894_89497

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : A = 4.80 :=
  let arcade_spent := (3 : ℚ) / 5
  let arcade_remaining := 1 - arcade_spent
  let toy_store_spent := (1 : ℚ) / 3 * arcade_remaining
  let candy_store_remaining := arcade_remaining - toy_store_spent
  have h1 : arcade_remaining = (2 : ℚ) / 5 := by sorry
  have h2 : candy_store_remaining = (4 : ℚ) / 15 := by sorry
  have h3 : candy_store_remaining * A = 1.28 := by sorry
  sorry

#eval (4.80 : ℚ)

end NUMINAMATH_CALUDE_johns_allowance_l894_89497


namespace NUMINAMATH_CALUDE_largest_intersection_x_coordinate_l894_89418

/-- The polynomial function -/
def P (d : ℝ) (x : ℝ) : ℝ := x^6 - 5*x^5 + 5*x^4 + 5*x^3 + d*x^2

/-- The parabola function -/
def Q (e f g : ℝ) (x : ℝ) : ℝ := e*x^2 + f*x + g

/-- The difference between the polynomial and the parabola -/
def R (d e f g : ℝ) (x : ℝ) : ℝ := P d x - Q e f g x

theorem largest_intersection_x_coordinate
  (d e f g : ℝ)
  (h1 : ∃ a b c : ℝ, ∀ x : ℝ, R d e f g x = (x - a)^2 * (x - b)^2 * (x - c)^2)
  (h2 : ∃! a b c : ℝ, ∀ x : ℝ, R d e f g x = (x - a)^2 * (x - b)^2 * (x - c)^2) :
  ∃ x : ℝ, (∀ y : ℝ, R d e f g y = 0 → y ≤ x) ∧ R d e f g x = 0 ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_largest_intersection_x_coordinate_l894_89418


namespace NUMINAMATH_CALUDE_part_one_part_two_l894_89477

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := (k - 1) * x^2 - 4 * x + 3

-- Part 1
theorem part_one (k : ℝ) :
  (quadratic_equation k 1 = 0) → 
  (k = 2 ∧ ∃ x, x ≠ 1 ∧ quadratic_equation k x = 0 ∧ x = 3) :=
by sorry

-- Part 2
theorem part_two (k x₁ x₂ : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0) →
  (x₁^2 * x₂ + x₁ * x₂^2 = 3) →
  (k = -1) :=
by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l894_89477


namespace NUMINAMATH_CALUDE_hyperbola_equation_l894_89405

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_equation :
  (∀ x y : ℝ, asymptotes x y → (∃ a b : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ b^2 = 3 * a^2)) →
  hyperbola 2 3 →
  ∀ x y : ℝ, hyperbola x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l894_89405


namespace NUMINAMATH_CALUDE_second_white_given_first_white_l894_89417

/-- Represents the number of white balls initially in the bag -/
def white_balls : ℕ := 5

/-- Represents the number of red balls initially in the bag -/
def red_balls : ℕ := 3

/-- Represents the total number of balls initially in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- Represents the probability of drawing a white ball on the second draw
    given that the first draw was white -/
def prob_second_white_given_first_white : ℚ := 4 / 7

/-- Theorem stating that the probability of drawing a white ball on the second draw
    given that the first draw was white is 4/7 -/
theorem second_white_given_first_white :
  prob_second_white_given_first_white = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_second_white_given_first_white_l894_89417


namespace NUMINAMATH_CALUDE_lee_cookies_l894_89406

/-- Given that Lee can make 24 cookies with 3 cups of flour, 
    this theorem proves he can make 36 cookies with 4.5 cups of flour. -/
theorem lee_cookies (cookies_per_3_cups : ℕ) (cookies_per_4_5_cups : ℕ) 
  (h1 : cookies_per_3_cups = 24) :
  cookies_per_4_5_cups = 36 :=
by
  sorry

#check lee_cookies

end NUMINAMATH_CALUDE_lee_cookies_l894_89406


namespace NUMINAMATH_CALUDE_no_nonzero_solution_for_diophantine_equation_l894_89421

theorem no_nonzero_solution_for_diophantine_equation :
  ∀ (x y z : ℤ), 2 * x^4 + y^4 = 7 * z^4 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_solution_for_diophantine_equation_l894_89421


namespace NUMINAMATH_CALUDE_equation_solution_l894_89429

theorem equation_solution :
  ∃! x : ℝ, (3 : ℝ) ^ (2 * x + 2) = (1 : ℝ) / 81 :=
by
  use -3
  sorry

end NUMINAMATH_CALUDE_equation_solution_l894_89429


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l894_89422

theorem rectangle_side_lengths (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) 
  (h4 : a * b = 2 * (a + b)) : a < 4 ∧ b > 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l894_89422


namespace NUMINAMATH_CALUDE_ellen_orange_juice_amount_l894_89402

/-- The amount of orange juice in Ellen's smoothie --/
def orange_juice_amount (strawberries yogurt total : ℚ) : ℚ :=
  total - (strawberries + yogurt)

/-- Theorem: Ellen used 0.2 cups of orange juice in her smoothie --/
theorem ellen_orange_juice_amount :
  orange_juice_amount (2/10) (1/10) (5/10) = 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ellen_orange_juice_amount_l894_89402


namespace NUMINAMATH_CALUDE_distance_AF_l894_89440

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : Parabola x y

-- Define the property of the midpoint
def MidpointProperty (A : PointOnParabola) : Prop :=
  (A.x + Focus.1) / 2 = 2

-- Theorem statement
theorem distance_AF (A : PointOnParabola) 
  (h : MidpointProperty A) : 
  Real.sqrt ((A.x - Focus.1)^2 + (A.y - Focus.2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_AF_l894_89440


namespace NUMINAMATH_CALUDE_apple_calculation_l894_89499

/-- The number of apples Pinky, Danny, and Benny collectively have after accounting for Lucy's sales -/
def total_apples (pinky_apples danny_apples lucy_sales benny_apples : ℝ) : ℝ :=
  pinky_apples + danny_apples + benny_apples - lucy_sales

/-- Theorem stating the total number of apples after Lucy's sales -/
theorem apple_calculation :
  total_apples 36.5 73.2 15.7 48.8 = 142.8 := by
  sorry

end NUMINAMATH_CALUDE_apple_calculation_l894_89499


namespace NUMINAMATH_CALUDE_apartment_keys_theorem_l894_89445

/-- The number of keys needed for apartment complexes -/
def keys_needed (num_complexes : ℕ) (apartments_per_complex : ℕ) (keys_per_apartment : ℕ) : ℕ :=
  num_complexes * apartments_per_complex * keys_per_apartment

/-- Theorem: Given two apartment complexes with 12 apartments each, 
    and requiring 3 keys per apartment, the total number of keys needed is 72 -/
theorem apartment_keys_theorem :
  keys_needed 2 12 3 = 72 := by
  sorry


end NUMINAMATH_CALUDE_apartment_keys_theorem_l894_89445


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l894_89494

theorem mixed_number_calculation : 
  (4 + 2/7 : ℚ) * (5 + 1/2) - ((3 + 1/3) + (2 + 1/6)) = 18 + 1/14 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l894_89494


namespace NUMINAMATH_CALUDE_f_inequality_l894_89483

open Real

-- Define the function f on (0, +∞)
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State that f' is indeed the derivative of f
variable (hf' : ∀ x, x > 0 → HasDerivAt f (f' x) x)

-- State the condition xf'(x) > 2f(x)
variable (h_cond : ∀ x, x > 0 → x * f' x > 2 * f x)

-- Define a and b
variable (a b : ℝ)

-- State that a > b > 0
variable (hab : a > b ∧ b > 0)

-- Theorem statement
theorem f_inequality : b^2 * f a > a^2 * f b := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l894_89483


namespace NUMINAMATH_CALUDE_log_expression_equals_one_l894_89482

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_one :
  (log10 5)^2 + log10 50 * log10 2 = 1 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_one_l894_89482


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l894_89408

theorem number_of_divisors_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l894_89408


namespace NUMINAMATH_CALUDE_smallest_c_value_l894_89400

theorem smallest_c_value (a b c : ℤ) : 
  (b - a = c - b) →  -- arithmetic progression
  (c * c = a * b) →  -- geometric progression
  (∃ (a' b' c' : ℤ), b' - a' = c' - b' ∧ c' * c' = a' * b' ∧ c' < c) →
  c ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_value_l894_89400


namespace NUMINAMATH_CALUDE_martin_failed_by_200_l894_89443

/-- Calculates the number of marks by which a student failed an exam -/
def marksFailedBy (maxMarks passingPercentage studentScore : ℕ) : ℕ :=
  let passingMark := (passingPercentage * maxMarks) / 100
  passingMark - studentScore

/-- Proves that Martin failed the exam by 200 marks -/
theorem martin_failed_by_200 :
  let maxMarks : ℕ := 500
  let passingPercentage : ℕ := 80
  let martinScore : ℕ := 200
  let passingMark := (passingPercentage * maxMarks) / 100
  martinScore < passingMark →
  marksFailedBy maxMarks passingPercentage martinScore = 200 := by
  sorry

end NUMINAMATH_CALUDE_martin_failed_by_200_l894_89443


namespace NUMINAMATH_CALUDE_comic_books_average_l894_89488

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

theorem comic_books_average (a₁ : ℕ) (d : ℕ) (n : ℕ) 
  (h₁ : a₁ = 10) (h₂ : d = 6) (h₃ : n = 8) : 
  (arithmetic_sequence a₁ d n).sum / n = 31 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_average_l894_89488


namespace NUMINAMATH_CALUDE_puzzle_min_cost_l894_89446

/-- Represents the cost structure and purchase requirement for puzzles -/
structure PuzzlePurchase where
  single_cost : ℕ  -- Cost of a single puzzle
  box_cost : ℕ    -- Cost of a box of puzzles
  box_size : ℕ    -- Number of puzzles in a box
  required : ℕ    -- Number of puzzles required

/-- Calculates the minimum cost for purchasing the required number of puzzles -/
def minCost (p : PuzzlePurchase) : ℕ :=
  let boxes := p.required / p.box_size
  let singles := p.required % p.box_size
  boxes * p.box_cost + singles * p.single_cost

/-- Theorem stating that the minimum cost for 25 puzzles is $210 -/
theorem puzzle_min_cost :
  let p : PuzzlePurchase := {
    single_cost := 10,
    box_cost := 50,
    box_size := 6,
    required := 25
  }
  minCost p = 210 := by
  sorry


end NUMINAMATH_CALUDE_puzzle_min_cost_l894_89446


namespace NUMINAMATH_CALUDE_sqrt_three_difference_of_squares_l894_89423

theorem sqrt_three_difference_of_squares : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_difference_of_squares_l894_89423


namespace NUMINAMATH_CALUDE_least_six_digit_divisible_by_198_l894_89465

theorem least_six_digit_divisible_by_198 : ∃ n : ℕ, 
  (n ≥ 100000 ∧ n < 1000000) ∧  -- 6-digit number condition
  n % 198 = 0 ∧                 -- divisibility condition
  ∀ m : ℕ, (m ≥ 100000 ∧ m < 1000000) ∧ m % 198 = 0 → n ≤ m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_six_digit_divisible_by_198_l894_89465


namespace NUMINAMATH_CALUDE_student_incorrect_answer_l894_89455

theorem student_incorrect_answer 
  (D : ℕ) -- Dividend
  (h1 : D / 36 = 42) -- Correct division
  (h2 : 63 ≠ 36) -- Student used wrong divisor
  : D / 63 = 24 := by
  sorry

end NUMINAMATH_CALUDE_student_incorrect_answer_l894_89455


namespace NUMINAMATH_CALUDE_min_value_expression_l894_89436

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) ≥ -2040200 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l894_89436


namespace NUMINAMATH_CALUDE_betty_initial_marbles_l894_89468

/-- Proves that Betty initially had 60 marbles given the conditions of the problem -/
theorem betty_initial_marbles :
  ∀ (betty_initial : ℕ) (stuart_initial : ℕ) (stuart_final : ℕ),
    stuart_initial = 56 →
    stuart_final = 80 →
    stuart_final = stuart_initial + (betty_initial * 40 / 100) →
    betty_initial = 60 :=
by sorry

end NUMINAMATH_CALUDE_betty_initial_marbles_l894_89468


namespace NUMINAMATH_CALUDE_tram_length_l894_89459

/-- The length of a tram given its passing time and tunnel transit time -/
theorem tram_length (passing_time tunnel_time tunnel_length : ℝ) 
  (h1 : passing_time = 4)
  (h2 : tunnel_time = 12)
  (h3 : tunnel_length = 64)
  (h4 : passing_time > 0)
  (h5 : tunnel_time > 0)
  (h6 : tunnel_length > 0) :
  (tunnel_length * passing_time) / (tunnel_time - passing_time) = 32 := by
  sorry

end NUMINAMATH_CALUDE_tram_length_l894_89459


namespace NUMINAMATH_CALUDE_circle_radius_largest_radius_l894_89413

/-- A circle tangent to both x and y axes with center (r,r) passing through (9,2) has radius 17 or 5 -/
theorem circle_radius (r : ℝ) : 
  (r > 0) → 
  ((9 - r)^2 + (2 - r)^2 = r^2) → 
  (r = 17 ∨ r = 5) :=
by sorry

/-- The largest possible radius of a circle tangent to both x and y axes and passing through (9,2) is 17 -/
theorem largest_radius : 
  ∃ (r : ℝ), (r > 0) ∧ 
  ((9 - r)^2 + (2 - r)^2 = r^2) ∧ 
  (∀ (s : ℝ), (s > 0) ∧ ((9 - s)^2 + (2 - s)^2 = s^2) → s ≤ r) ∧
  r = 17 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_largest_radius_l894_89413
