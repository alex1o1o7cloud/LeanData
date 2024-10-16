import Mathlib

namespace NUMINAMATH_CALUDE_someone_answers_no_l1484_148412

/-- Represents a person who can be either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- Represents the circular arrangement of people -/
def CircularArrangement := Vector Person 100

/-- A function that determines if a person sees more knights than liars -/
def seesMoreKnights (arrangement : CircularArrangement) (position : Fin 100) : Bool :=
  sorry

/-- A function that determines how a person would answer the question -/
def personAnswer (arrangement : CircularArrangement) (position : Fin 100) : Bool :=
  match arrangement.get position with
  | Person.Knight => seesMoreKnights arrangement position
  | Person.Liar => ¬(seesMoreKnights arrangement position)

theorem someone_answers_no
  (arrangement : CircularArrangement)
  (h1 : ∃ i, arrangement.get i = Person.Knight)
  (h2 : ∃ i, arrangement.get i = Person.Liar) :
  ∃ i, personAnswer arrangement i = false :=
sorry

end NUMINAMATH_CALUDE_someone_answers_no_l1484_148412


namespace NUMINAMATH_CALUDE_library_books_end_of_month_l1484_148457

theorem library_books_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) : 
  initial_books = 75 →
  loaned_books = 20 →
  return_rate = 65 / 100 →
  initial_books - loaned_books + (↑loaned_books * return_rate).floor = 68 :=
by sorry

end NUMINAMATH_CALUDE_library_books_end_of_month_l1484_148457


namespace NUMINAMATH_CALUDE_megacorp_fine_l1484_148414

def daily_mining_revenue : ℝ := 3000000
def daily_oil_revenue : ℝ := 5000000
def monthly_expenses : ℝ := 30000000
def fine_percentage : ℝ := 0.01
def days_in_year : ℕ := 365
def months_in_year : ℕ := 12

theorem megacorp_fine :
  let daily_revenue := daily_mining_revenue + daily_oil_revenue
  let annual_revenue := daily_revenue * days_in_year
  let annual_expenses := monthly_expenses * months_in_year
  let annual_profit := annual_revenue - annual_expenses
  let fine := annual_profit * fine_percentage
  fine = 25600000 := by sorry

end NUMINAMATH_CALUDE_megacorp_fine_l1484_148414


namespace NUMINAMATH_CALUDE_faye_pencils_l1484_148421

/-- The number of rows of pencils -/
def num_rows : ℕ := 14

/-- The number of pencils in each row -/
def pencils_per_row : ℕ := 11

/-- The total number of pencils -/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem faye_pencils : total_pencils = 154 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencils_l1484_148421


namespace NUMINAMATH_CALUDE_card_pair_probability_l1484_148471

/-- Represents the number of cards for each value in the deck -/
def cardsPerValue : ℕ := 5

/-- Represents the number of different values in the deck -/
def numValues : ℕ := 10

/-- Represents the total number of cards in the original deck -/
def totalCards : ℕ := cardsPerValue * numValues

/-- Represents the number of pairs removed -/
def pairsRemoved : ℕ := 2

/-- Represents the number of cards remaining after removal -/
def remainingCards : ℕ := totalCards - (2 * pairsRemoved)

/-- Represents the number of values with full sets of cards after removal -/
def fullSets : ℕ := numValues - pairsRemoved

/-- Represents the number of values with reduced sets of cards after removal -/
def reducedSets : ℕ := pairsRemoved

theorem card_pair_probability :
  (fullSets * (cardsPerValue.choose 2) + reducedSets * ((cardsPerValue - 2).choose 2)) /
  (remainingCards.choose 2) = 86 / 1035 := by
  sorry

end NUMINAMATH_CALUDE_card_pair_probability_l1484_148471


namespace NUMINAMATH_CALUDE_quiz_winning_probability_l1484_148432

-- Define the quiz parameters
def num_questions : ℕ := 4
def num_choices : ℕ := 3
def min_correct : ℕ := 3

-- Define the probability of guessing one question correctly
def prob_correct : ℚ := 1 / num_choices

-- Define the probability of guessing one question incorrectly
def prob_incorrect : ℚ := 1 - prob_correct

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the probability of winning
def prob_winning : ℚ :=
  (binomial num_questions num_questions) * (prob_correct ^ num_questions) +
  (binomial num_questions min_correct) * (prob_correct ^ min_correct) * (prob_incorrect ^ (num_questions - min_correct))

-- Theorem statement
theorem quiz_winning_probability :
  prob_winning = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_quiz_winning_probability_l1484_148432


namespace NUMINAMATH_CALUDE_slope_of_line_l1484_148403

theorem slope_of_line (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : (4 / x₁) + (5 / y₁) = 0) (h₃ : (4 / x₂) + (5 / y₂) = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l1484_148403


namespace NUMINAMATH_CALUDE_base10_729_equals_base7_261_l1484_148489

-- Define a function to convert a base-7 number to base-10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

-- Define the base-7 representation of 261₇
def base7_261 : List Nat := [1, 6, 2]

-- Theorem statement
theorem base10_729_equals_base7_261 :
  base7ToBase10 base7_261 = 729 := by
  sorry

end NUMINAMATH_CALUDE_base10_729_equals_base7_261_l1484_148489


namespace NUMINAMATH_CALUDE_experience_ratio_l1484_148477

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.roger + 8 = 50 ∧
  e.peter = 12 ∧
  e.robert = e.peter - 4 ∧
  e.robert = e.mike + 2

theorem experience_ratio (e : Experience) 
  (h : satisfiesConditions e) : e.tom = e.robert :=
sorry

end NUMINAMATH_CALUDE_experience_ratio_l1484_148477


namespace NUMINAMATH_CALUDE_vector_problem_l1484_148453

/-- Given two vectors a and b in ℝ², prove that:
    1) a = (-3, 4) and b = (5, -12)
    2) The dot product of a and b is -63
    3) The cosine of the angle between a and b is -63/65
-/
theorem vector_problem (a b : ℝ × ℝ) :
  (a.1 + b.1 = 2 ∧ a.2 + b.2 = -8) ∧  -- a + b = (2, -8)
  (a.1 - b.1 = -8 ∧ a.2 - b.2 = 16) → -- a - b = (-8, 16)
  (a = (-3, 4) ∧ b = (5, -12)) ∧      -- Part 1
  (a.1 * b.1 + a.2 * b.2 = -63) ∧     -- Part 2
  ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -63/65) -- Part 3
  := by sorry

end NUMINAMATH_CALUDE_vector_problem_l1484_148453


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l1484_148494

theorem max_stamps_purchasable (stamp_price : ℕ) (budget : ℕ) : 
  stamp_price = 25 → budget = 5000 → 
  ∃ n : ℕ, n * stamp_price ≤ budget ∧ 
  ∀ m : ℕ, m * stamp_price ≤ budget → m ≤ n ∧ 
  n = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l1484_148494


namespace NUMINAMATH_CALUDE_equation_solution_l1484_148413

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1484_148413


namespace NUMINAMATH_CALUDE_simplify_expression_l1484_148466

theorem simplify_expression (x : ℝ) : 4 * x^2 - (2 * x^2 + x - 1) + (2 - x^2 + 3 * x) = x^2 + 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1484_148466


namespace NUMINAMATH_CALUDE_ten_cent_coin_count_l1484_148415

theorem ten_cent_coin_count :
  ∀ (x y : ℕ),
  x + y = 20 →
  10 * x + 50 * y = 800 →
  x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ten_cent_coin_count_l1484_148415


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l1484_148483

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l1484_148483


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l1484_148423

def ellipse_eccentricity_range (a b c : ℝ) (F₁ F₂ P : ℝ × ℝ) : Prop :=
  let e := c / a
  0 < b ∧ b < a ∧
  c^2 + b^2 = a^2 ∧
  F₁ = (-c, 0) ∧
  F₂ = (c, 0) ∧
  P.1 = a^2 / c ∧
  (∃ m : ℝ, P = (a^2 / c, m) ∧
    let K := ((a^2 - c^2) / (2 * c), m / 2)
    (P.2 - F₁.2) * (K.2 - F₂.2) = -(P.1 - F₁.1) * (K.1 - F₂.1)) →
  Real.sqrt 3 / 3 ≤ e ∧ e < 1

theorem ellipse_eccentricity_theorem (a b c : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  ellipse_eccentricity_range a b c F₁ F₂ P := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l1484_148423


namespace NUMINAMATH_CALUDE_circle_passes_through_origin_l1484_148438

/-- Definition of the circle C with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m-1)*x + 2*(m-1)*y + 2*m^2 - 6*m + 4 = 0

/-- Theorem stating that the circle passes through the origin when m = 2 -/
theorem circle_passes_through_origin :
  ∃ m : ℝ, circle_equation 0 0 m ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_origin_l1484_148438


namespace NUMINAMATH_CALUDE_p_investment_l1484_148405

/-- Given that Q invested 15000 and the profit is divided in the ratio 5:1, prove that P's investment is 75000 --/
theorem p_investment (q_investment : ℕ) (profit_ratio_p profit_ratio_q : ℕ) :
  q_investment = 15000 →
  profit_ratio_p = 5 →
  profit_ratio_q = 1 →
  profit_ratio_p * q_investment = profit_ratio_q * 75000 :=
by sorry

end NUMINAMATH_CALUDE_p_investment_l1484_148405


namespace NUMINAMATH_CALUDE_negation_of_forall_inequality_l1484_148454

theorem negation_of_forall_inequality :
  (¬ ∀ x : ℝ, x^2 - x > x + 1) ↔ (∃ x : ℝ, x^2 - x ≤ x + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_inequality_l1484_148454


namespace NUMINAMATH_CALUDE_set_equality_l1484_148455

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {1, 2}

theorem set_equality : M = N := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l1484_148455


namespace NUMINAMATH_CALUDE_min_value_3x_plus_4y_l1484_148402

theorem min_value_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_4y_l1484_148402


namespace NUMINAMATH_CALUDE_sevenPeopleRoundTable_l1484_148400

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def seatingArrangements (totalPeople : ℕ) (adjacentPair : ℕ) : ℕ :=
  if totalPeople ≤ 1 then 0
  else
    let effectiveUnits := totalPeople - adjacentPair + 1
    (factorial effectiveUnits * adjacentPair) / totalPeople

theorem sevenPeopleRoundTable :
  seatingArrangements 7 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_sevenPeopleRoundTable_l1484_148400


namespace NUMINAMATH_CALUDE_family_trip_eggs_l1484_148488

theorem family_trip_eggs (adults girls : ℕ) (total_eggs : ℕ) : 
  adults = 3 →
  girls = 7 →
  total_eggs = 36 →
  ∃ (boys : ℕ), 
    adults * 3 + girls * 1 + boys * 2 = total_eggs ∧
    boys = 10 :=
by sorry

end NUMINAMATH_CALUDE_family_trip_eggs_l1484_148488


namespace NUMINAMATH_CALUDE_probability_below_8_l1484_148465

theorem probability_below_8 (p_10 p_9 p_8 : ℝ) 
  (h1 : p_10 = 0.24)
  (h2 : p_9 = 0.28)
  (h3 : p_8 = 0.19) :
  1 - (p_10 + p_9 + p_8) = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_probability_below_8_l1484_148465


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1484_148461

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {-1, a^2 + 1, a^2 - 3}
  let B : Set ℝ := {-4, a - 1, a + 1}
  A ∩ B = {-2} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1484_148461


namespace NUMINAMATH_CALUDE_exists_unreachable_number_l1484_148410

/-- A function that returns true if a number is a 4-digit integer -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that counts the number of differing digits between two numbers -/
def digit_difference (n m : ℕ) : ℕ := sorry

/-- The theorem stating that there exists a 4-digit number that cannot be changed 
    into a multiple of 1992 by changing 3 of its digits -/
theorem exists_unreachable_number : 
  ∃ n : ℕ, is_four_digit n ∧ 
    ∀ m : ℕ, is_four_digit m → m % 1992 = 0 → digit_difference n m > 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_unreachable_number_l1484_148410


namespace NUMINAMATH_CALUDE_last_episode_length_correct_l1484_148487

/-- Represents the duration of a TV series viewing session -/
structure SeriesViewing where
  episodeLengths : List Nat
  breakLength : Nat
  totalTime : Nat

/-- Calculates the length of the last episode given the viewing details -/
def lastEpisodeLength (s : SeriesViewing) : Nat :=
  s.totalTime
    - (s.episodeLengths.sum + s.breakLength * s.episodeLengths.length)

theorem last_episode_length_correct (s : SeriesViewing) :
  s.episodeLengths = [58, 62, 65, 71, 79] ∧
  s.breakLength = 12 ∧
  s.totalTime = 9 * 60 →
  lastEpisodeLength s = 145 := by
  sorry

#eval lastEpisodeLength {
  episodeLengths := [58, 62, 65, 71, 79],
  breakLength := 12,
  totalTime := 9 * 60
}

end NUMINAMATH_CALUDE_last_episode_length_correct_l1484_148487


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1484_148418

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) = f (-x + 2)) ∧ 
  f 0 = 3 ∧ 
  f 2 = 1

/-- The range of m for which the function has max 3 and min 1 on [0,m] -/
def ValidRange (f : ℝ → ℝ) (m : ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 m, f x ≤ 3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 3) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 1)

/-- The main theorem -/
theorem quadratic_function_range (f : ℝ → ℝ) (h : QuadraticFunction f) :
  {m | ValidRange f m} = Set.Icc 2 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1484_148418


namespace NUMINAMATH_CALUDE_dollar_op_neg_three_neg_four_l1484_148407

def dollar_op (x y : Int) : Int := x * (y + 1) + x * y

theorem dollar_op_neg_three_neg_four : dollar_op (-3) (-4) = 21 := by
  sorry

end NUMINAMATH_CALUDE_dollar_op_neg_three_neg_four_l1484_148407


namespace NUMINAMATH_CALUDE_complex_magnitude_l1484_148440

theorem complex_magnitude (a b : ℝ) (z : ℂ) :
  (a + Complex.I)^2 = b * Complex.I →
  z = a + b * Complex.I →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1484_148440


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l1484_148495

/-- A function y of x is quadratic if it can be written in the form y = ax² + bx + c, where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The main theorem stating that m = -1 is the only value satisfying the given conditions -/
theorem quadratic_function_m_value :
  ∃! m : ℝ, IsQuadratic (fun x ↦ (m - 1) * x^(m^2 + 1) + 3 * x) ∧ m - 1 ≠ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_m_value_l1484_148495


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_formula_l1484_148467

/-- The sum of the first k+1 terms of an arithmetic series with first term k^2 + 2 and common difference 2 -/
def arithmetic_series_sum (k : ℕ) : ℕ := sorry

/-- The first term of the arithmetic series -/
def first_term (k : ℕ) : ℕ := k^2 + 2

/-- The common difference of the arithmetic series -/
def common_difference : ℕ := 2

/-- The number of terms in the series -/
def num_terms (k : ℕ) : ℕ := k + 1

theorem arithmetic_series_sum_formula (k : ℕ) :
  arithmetic_series_sum k = k^3 + 2*k^2 + 3*k + 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_formula_l1484_148467


namespace NUMINAMATH_CALUDE_puppies_adoption_time_l1484_148481

/-- The number of days required to adopt all puppies -/
def adoption_days (initial_puppies : ℕ) (additional_puppies : ℕ) (adoption_rate : ℕ) : ℕ :=
  (initial_puppies + additional_puppies) / adoption_rate

/-- Theorem: Given the initial conditions, it takes 9 days to adopt all puppies -/
theorem puppies_adoption_time :
  adoption_days 2 34 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_puppies_adoption_time_l1484_148481


namespace NUMINAMATH_CALUDE_min_x_for_sqrt_2x_minus_1_l1484_148409

theorem min_x_for_sqrt_2x_minus_1 :
  ∀ x : ℝ, (∃ y : ℝ, y^2 = 2*x - 1) → x ≥ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_min_x_for_sqrt_2x_minus_1_l1484_148409


namespace NUMINAMATH_CALUDE_binomial_sum_distinct_values_l1484_148430

theorem binomial_sum_distinct_values :
  ∃ (S : Finset ℕ), (∀ r : ℤ, 7 ≤ r ∧ r ≤ 9 →
    (Nat.choose 10 (r.toNat + 1) + Nat.choose 10 (17 - r.toNat)) ∈ S) ∧ 
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_distinct_values_l1484_148430


namespace NUMINAMATH_CALUDE_orange_eaters_difference_l1484_148497

def family_gathering (total : ℕ) (orange_eaters : ℕ) (banana_eaters : ℕ) (apple_eaters : ℕ) : Prop :=
  total = 20 ∧
  orange_eaters = total / 2 ∧
  banana_eaters = (total - orange_eaters) / 2 ∧
  apple_eaters = total - orange_eaters - banana_eaters ∧
  orange_eaters < total

theorem orange_eaters_difference (total orange_eaters banana_eaters apple_eaters : ℕ) :
  family_gathering total orange_eaters banana_eaters apple_eaters →
  total - orange_eaters = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_eaters_difference_l1484_148497


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l1484_148468

theorem smallest_undefined_inverse (a : ℕ) : 
  (a > 0) → 
  (¬ ∃ (x : ℕ), x * a ≡ 1 [MOD 77]) → 
  (¬ ∃ (y : ℕ), y * a ≡ 1 [MOD 91]) → 
  (∀ (b : ℕ), b > 0 ∧ b < a → 
    (∃ (x : ℕ), x * b ≡ 1 [MOD 77]) ∨ 
    (∃ (y : ℕ), y * b ≡ 1 [MOD 91])) → 
  a = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l1484_148468


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1484_148428

/-- Given a point M that is the midpoint of line segment AB, 
    and the coordinates of points M and A, 
    prove that the sum of coordinates of point B is 2. -/
theorem midpoint_coordinate_sum (xM yM xA yA : ℝ) : 
  xM = 3 → 
  yM = 5 → 
  xA = 6 → 
  yA = 8 → 
  (∃ xB yB : ℝ, 
    xM = (xA + xB) / 2 ∧ 
    yM = (yA + yB) / 2 ∧ 
    xB + yB = 2) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1484_148428


namespace NUMINAMATH_CALUDE_largest_d_for_range_l1484_148479

theorem largest_d_for_range (g : ℝ → ℝ) (d : ℝ) : 
  (∃ x, g x = -5) ∧ 
  (∀ x, g x = x^2 + 5*x + d) ∧ 
  (∀ d', d' > d → ¬∃ x, x^2 + 5*x + d' = -5) → 
  d = 5/4 := by sorry

end NUMINAMATH_CALUDE_largest_d_for_range_l1484_148479


namespace NUMINAMATH_CALUDE_potato_bag_weight_l1484_148426

theorem potato_bag_weight (original_weight : ℝ) : 
  (original_weight / (original_weight / 2) = 36) → original_weight = 648 :=
by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l1484_148426


namespace NUMINAMATH_CALUDE_power_of_fraction_l1484_148449

theorem power_of_fraction : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_l1484_148449


namespace NUMINAMATH_CALUDE_determine_bal_meaning_l1484_148463

/-- Represents the possible responses from a native --/
inductive Response
| Bal
| Da

/-- Represents the possible meanings of a word --/
inductive Meaning
| Yes
| No

/-- A native person who can respond to questions --/
structure Native where
  response : String → Response

/-- The meaning of the word "bal" --/
def balMeaning (n : Native) : Meaning :=
  match n.response "Are you a human?" with
  | Response.Bal => Meaning.Yes
  | Response.Da => Meaning.No

/-- Theorem stating that it's possible to determine the meaning of "bal" with a single question --/
theorem determine_bal_meaning (n : Native) :
  (∀ q : String, n.response q = Response.Bal ∨ n.response q = Response.Da) →
  (n.response "Are you a human?" = Response.Da → Meaning.Yes = Meaning.Yes) →
  (∀ q : String, n.response q = Response.Bal → Meaning.Yes = balMeaning n) :=
by
  sorry


end NUMINAMATH_CALUDE_determine_bal_meaning_l1484_148463


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1484_148460

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_coefficient (q : QuadraticFunction) 
  (vertex_x : q.f 2 = 5) 
  (point : q.f 1 = 6) : 
  q.a = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1484_148460


namespace NUMINAMATH_CALUDE_oil_ratio_in_first_bottle_l1484_148447

theorem oil_ratio_in_first_bottle 
  (C : ℝ) 
  (h1 : C > 0)
  (oil_in_second : ℝ)
  (h2 : oil_in_second = C / 2)
  (total_content : ℝ)
  (h3 : total_content = 3 * C)
  (total_oil : ℝ)
  (h4 : total_oil = total_content / 3)
  (oil_in_first : ℝ)
  (h5 : oil_in_first + oil_in_second = total_oil) :
  oil_in_first / C = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_oil_ratio_in_first_bottle_l1484_148447


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l1484_148473

theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_A : ℝ) (ethanol_B : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 218 →
  ethanol_A = 0.12 →
  ethanol_B = 0.16 →
  total_ethanol = 30 →
  ∃ (V_A : ℝ), V_A = 122 ∧
    ∃ (V_B : ℝ), V_A + V_B = tank_capacity ∧
    ethanol_A * V_A + ethanol_B * V_B = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l1484_148473


namespace NUMINAMATH_CALUDE_fraction_not_simplifiable_l1484_148433

theorem fraction_not_simplifiable (n : ℕ) : ¬ ∃ (d : ℤ), d > 1 ∧ d ∣ (21 * n + 4) ∧ d ∣ (14 * n + 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_simplifiable_l1484_148433


namespace NUMINAMATH_CALUDE_production_days_l1484_148491

theorem production_days (n : ℕ) 
  (h1 : (n * 50 + 115) / (n + 1) = 55) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l1484_148491


namespace NUMINAMATH_CALUDE_shirt_tie_outfits_l1484_148470

theorem shirt_tie_outfits (shirts : ℕ) (ties : ℕ) (h1 : shirts = 8) (h2 : ties = 6) :
  shirts * ties = 48 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_outfits_l1484_148470


namespace NUMINAMATH_CALUDE_triangle_area_formula_right_angle_l1484_148462

theorem triangle_area_formula_right_angle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (1/2) * (a * b) / Real.sin (π/2) = (1/2) * a * b := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_formula_right_angle_l1484_148462


namespace NUMINAMATH_CALUDE_sugar_recipe_reduction_l1484_148408

theorem sugar_recipe_reduction : 
  let full_recipe : ℚ := 5 + 3/4
  let reduced_recipe : ℚ := full_recipe / 3
  reduced_recipe = 1 + 11/12 := by sorry

end NUMINAMATH_CALUDE_sugar_recipe_reduction_l1484_148408


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tan_l1484_148482

/-- Given an arithmetic sequence {a_n} where a₁ + a₇ + a₁₃ = π, 
    prove that tan(a₂ + a₁₂) = -√3 -/
theorem arithmetic_sequence_tan (a : ℕ → ℝ) :
  (∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence
  a 1 + a 7 + a 13 = Real.pi →                      -- given condition
  Real.tan (a 2 + a 12) = -Real.sqrt 3 :=           -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tan_l1484_148482


namespace NUMINAMATH_CALUDE_nth_equation_l1484_148485

theorem nth_equation (n : ℕ+) : (10 * n + 5)^2 = n * (n + 1) * 100 + 5^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l1484_148485


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_l1484_148456

/-- The first term of an infinite geometric series with common ratio -1/3 and sum 24 is 32 -/
theorem first_term_of_geometric_series (a : ℝ) : 
  (∑' n, a * (-1/3)^n : ℝ) = 24 → a = 32 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_l1484_148456


namespace NUMINAMATH_CALUDE_gerald_price_is_264_60_verify_hendricks_price_l1484_148448

-- Define the original price of the guitar
def original_price : ℝ := 280

-- Define Hendricks' discount rate
def hendricks_discount_rate : ℝ := 0.15

-- Define Gerald's discount rate
def gerald_discount_rate : ℝ := 0.10

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.05

-- Define Hendricks' final price
def hendricks_price : ℝ := 250

-- Function to calculate the final price after discount and tax
def calculate_final_price (price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

-- Theorem stating that Gerald's price is $264.60
theorem gerald_price_is_264_60 :
  calculate_final_price original_price gerald_discount_rate sales_tax_rate = 264.60 := by
  sorry

-- Theorem verifying Hendricks' price
theorem verify_hendricks_price :
  calculate_final_price original_price hendricks_discount_rate sales_tax_rate = hendricks_price := by
  sorry

end NUMINAMATH_CALUDE_gerald_price_is_264_60_verify_hendricks_price_l1484_148448


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1484_148499

theorem rectangular_field_area (width length perimeter area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  area = width * length →
  area = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1484_148499


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1484_148441

theorem imaginary_part_of_complex_division :
  let i : ℂ := Complex.I
  let z₁ : ℂ := 1 + i
  let z₂ : ℂ := 1 - i
  (z₁ / z₂).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1484_148441


namespace NUMINAMATH_CALUDE_only_3_4_5_is_right_triangle_l1484_148478

/-- Checks if three numbers form a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given groups of numbers -/
def groups : List (ℕ × ℕ × ℕ) :=
  [(1, 2, 3), (3, 4, 5), (4, 5, 6), (7, 8, 9)]

theorem only_3_4_5_is_right_triangle :
  ∃! g : ℕ × ℕ × ℕ, g ∈ groups ∧ isPythagoreanTriple g.1 g.2.1 g.2.2 :=
by
  sorry

end NUMINAMATH_CALUDE_only_3_4_5_is_right_triangle_l1484_148478


namespace NUMINAMATH_CALUDE_power_two_equals_four_l1484_148445

theorem power_two_equals_four : 2^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_two_equals_four_l1484_148445


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l1484_148458

-- Define the necessary types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (belongs_to : Point → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (α β : Plane) (l n : Line) :
  parallel α β →
  perpendicular l α →
  contained_in n β →
  perpendicular_lines l n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l1484_148458


namespace NUMINAMATH_CALUDE_polynomial_division_proof_l1484_148404

theorem polynomial_division_proof (z : ℝ) : 
  ((4/3 : ℝ) * z^4 - (17/9 : ℝ) * z^3 + (56/27 : ℝ) * z^2 - (167/81 : ℝ) * z + 500/243) * (3 * z + 1) = 
  4 * z^5 - 5 * z^4 + 7 * z^3 - 15 * z^2 + 9 * z - 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_proof_l1484_148404


namespace NUMINAMATH_CALUDE_simplify_fraction_l1484_148431

theorem simplify_fraction : 5 * (18 / 7) * (21 / -54) = -5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1484_148431


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l1484_148452

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 1| = |x - 2| + |x + 3| + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l1484_148452


namespace NUMINAMATH_CALUDE_holiday_duration_l1484_148427

theorem holiday_duration (total_rain_days : ℕ) (sunny_mornings : ℕ) (sunny_afternoons : ℕ)
  (h1 : total_rain_days = 7)
  (h2 : sunny_mornings = 5)
  (h3 : sunny_afternoons = 6) :
  ∃ (total_days : ℕ), total_days = 9 ∧ total_days ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_holiday_duration_l1484_148427


namespace NUMINAMATH_CALUDE_cyclists_problem_l1484_148422

/-- Two cyclists problem -/
theorem cyclists_problem (v₁ v₂ t : ℝ) :
  v₁ > 0 ∧ v₂ > 0 ∧ t > 0 ∧
  v₁ * t = v₂ * (1.5 : ℝ) ∧
  v₂ * t = v₁ * (2/3 : ℝ) →
  t = 1 ∧ v₁ / v₂ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_problem_l1484_148422


namespace NUMINAMATH_CALUDE_train_length_l1484_148475

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 40 → time = 27 → speed * time * (1000 / 3600) = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1484_148475


namespace NUMINAMATH_CALUDE_ariella_savings_after_two_years_l1484_148436

/-- Calculates the final amount in a savings account after simple interest is applied. -/
def final_amount (initial_amount : ℝ) (interest_rate : ℝ) (years : ℝ) : ℝ :=
  initial_amount * (1 + interest_rate * years)

/-- Proves that Ariella will have $720 after two years given the problem conditions. -/
theorem ariella_savings_after_two_years 
  (daniella_amount : ℝ)
  (ariella_excess : ℝ)
  (interest_rate : ℝ)
  (years : ℝ)
  (h1 : daniella_amount = 400)
  (h2 : ariella_excess = 200)
  (h3 : interest_rate = 0.1)
  (h4 : years = 2) :
  final_amount (daniella_amount + ariella_excess) interest_rate years = 720 :=
by
  sorry

#check ariella_savings_after_two_years

end NUMINAMATH_CALUDE_ariella_savings_after_two_years_l1484_148436


namespace NUMINAMATH_CALUDE_childrens_tickets_l1484_148429

theorem childrens_tickets (adult_price child_price total_tickets total_cost : ℚ)
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_tickets child_tickets : ℚ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_cost ∧
    child_tickets = 16 := by
  sorry

end NUMINAMATH_CALUDE_childrens_tickets_l1484_148429


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l1484_148476

theorem pulley_centers_distance (r1 r2 contact_distance : ℝ) 
  (h1 : r1 = 12)
  (h2 : r2 = 6)
  (h3 : contact_distance = 30) :
  ∃ (center_distance : ℝ), center_distance = 2 * Real.sqrt 234 := by
sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l1484_148476


namespace NUMINAMATH_CALUDE_count_five_digit_numbers_without_five_is_52488_l1484_148451

/-- The count of five-digit numbers not containing the digit 5 -/
def count_five_digit_numbers_without_five : ℕ :=
  8 * 9^4

/-- Theorem stating that the count of five-digit numbers not containing the digit 5 is 52488 -/
theorem count_five_digit_numbers_without_five_is_52488 :
  count_five_digit_numbers_without_five = 52488 := by
  sorry

end NUMINAMATH_CALUDE_count_five_digit_numbers_without_five_is_52488_l1484_148451


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odds_l1484_148480

theorem largest_common_divisor_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  ∃ (k : ℕ), k = 315 ∧ 
  (∀ (d : ℕ), d ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → d ≤ k) ∧
  k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odds_l1484_148480


namespace NUMINAMATH_CALUDE_smallest_side_of_triangle_l1484_148469

theorem smallest_side_of_triangle : ∃ (s : ℕ),
  (s : ℝ) > 0 ∧ 
  7.5 + (s : ℝ) > 11 ∧ 
  7.5 + 11 > (s : ℝ) ∧ 
  11 + (s : ℝ) > 7.5 ∧
  ∀ (t : ℕ), t > 0 → 
    (7.5 + (t : ℝ) > 11 ∧ 
     7.5 + 11 > (t : ℝ) ∧ 
     11 + (t : ℝ) > 7.5) → 
    s ≤ t ∧
  s = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_side_of_triangle_l1484_148469


namespace NUMINAMATH_CALUDE_geometric_sum_remainder_l1484_148442

theorem geometric_sum_remainder (n : ℕ) :
  (7^(n+1) - 1) / 6 % 500 = 1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_remainder_l1484_148442


namespace NUMINAMATH_CALUDE_range_of_2x_minus_3_l1484_148472

theorem range_of_2x_minus_3 (x : ℝ) (h : -1 < 2*x + 3 ∧ 2*x + 3 < 1) :
  ∃! (n : ℤ), ∃ (y : ℝ), 2*y - 3 = ↑n ∧ -1 < 2*y + 3 ∧ 2*y + 3 < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_2x_minus_3_l1484_148472


namespace NUMINAMATH_CALUDE_total_stickers_l1484_148425

/-- Given the following conditions:
    - There are 10 stickers on a page originally
    - There are 22 pages of stickers
    - 3 stickers are missing from each page
    Prove that the total number of stickers is 154 -/
theorem total_stickers (original_stickers : ℕ) (pages : ℕ) (missing_stickers : ℕ)
  (h1 : original_stickers = 10)
  (h2 : pages = 22)
  (h3 : missing_stickers = 3) :
  (original_stickers - missing_stickers) * pages = 154 :=
by sorry

end NUMINAMATH_CALUDE_total_stickers_l1484_148425


namespace NUMINAMATH_CALUDE_special_function_properties_l1484_148417

/-- A function satisfying f(ab) = af(b) + bf(a) for all a, b ∈ ℝ -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a * b) = a * f b + b * f a

theorem special_function_properties (f : ℝ → ℝ) 
  (h : special_function f) (h_not_zero : ∃ x, f x ≠ 0) :
  (f 0 = 0 ∧ f 1 = 0) ∧ (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l1484_148417


namespace NUMINAMATH_CALUDE_find_original_number_l1484_148490

theorem find_original_number : ∃ x : ℕ, 
  (x : ℚ) / 25 * 85 = x * 67 / 25 + 3390 ∧ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_find_original_number_l1484_148490


namespace NUMINAMATH_CALUDE_initial_price_is_four_l1484_148439

/-- Represents the sales data for a day --/
structure DaySales where
  price : ℝ
  quantity : ℝ

/-- Represents the sales data for three days --/
structure ThreeDaySales where
  day1 : DaySales
  day2 : DaySales
  day3 : DaySales

/-- Calculates the revenue for a given day --/
def revenue (day : DaySales) : ℝ :=
  day.price * day.quantity

/-- Checks if the sales data satisfies the problem conditions --/
def satisfiesConditions (sales : ThreeDaySales) : Prop :=
  sales.day2.price = sales.day1.price - 1 ∧
  sales.day2.quantity = sales.day1.quantity + 100 ∧
  sales.day3.price = sales.day2.price + 3 ∧
  sales.day3.quantity = sales.day2.quantity - 200 ∧
  revenue sales.day1 = revenue sales.day2 ∧
  revenue sales.day2 = revenue sales.day3

/-- The main theorem: if the sales data satisfies the conditions, the initial price was 4 yuan --/
theorem initial_price_is_four (sales : ThreeDaySales) :
  satisfiesConditions sales → sales.day1.price = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_price_is_four_l1484_148439


namespace NUMINAMATH_CALUDE_grid_last_row_digits_l1484_148493

/-- Represents a 3x4 grid of integers -/
def Grid := Matrix (Fin 3) (Fin 4) ℕ

/-- Check if a grid satisfies the given conditions -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 7 \ {0}) ∧
  (∀ i j₁ j₂, j₁ ≠ j₂ → g i j₁ ≠ g i j₂) ∧
  (∀ i₁ i₂ j, i₁ ≠ i₂ → g i₁ j ≠ g i₂ j) ∧
  g 1 1 = 5 ∧
  g 2 3 = 6

theorem grid_last_row_digits (g : Grid) (h : is_valid_grid g) :
  g 2 0 * 10000 + g 2 1 * 1000 + g 2 2 * 100 + g 2 3 * 10 + g 1 3 = 46123 :=
by sorry

end NUMINAMATH_CALUDE_grid_last_row_digits_l1484_148493


namespace NUMINAMATH_CALUDE_expression_equality_l1484_148420

theorem expression_equality : (2^1004 + 5^1005)^2 - (2^1004 - 5^1005)^2 = 20 * 10^1004 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1484_148420


namespace NUMINAMATH_CALUDE_opposite_numbers_theorem_l1484_148450

theorem opposite_numbers_theorem (a b c d : ℤ) : 
  (a + b = 0) → 
  (c = -1) → 
  (d = 1 ∨ d = -1) → 
  (2*a + 2*b - c*d = 1 ∨ 2*a + 2*b - c*d = -1) :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_theorem_l1484_148450


namespace NUMINAMATH_CALUDE_jim_sara_equal_savings_l1484_148464

/-- The number of weeks in the saving period -/
def weeks : ℕ := 820

/-- Sara's initial savings in dollars -/
def sara_initial : ℕ := 4100

/-- Sara's weekly savings in dollars -/
def sara_weekly : ℕ := 10

/-- Jim's weekly savings in dollars -/
def jim_weekly : ℕ := 15

/-- Total savings after the given period -/
def total_savings (initial weekly : ℕ) : ℕ :=
  initial + weekly * weeks

theorem jim_sara_equal_savings :
  total_savings 0 jim_weekly = total_savings sara_initial sara_weekly := by
  sorry

end NUMINAMATH_CALUDE_jim_sara_equal_savings_l1484_148464


namespace NUMINAMATH_CALUDE_range_of_m_l1484_148486

-- Define the conditions
def p (x : ℝ) : Prop := (x + 2) / (10 - x) ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, q x m → p x) →  -- p is a necessary condition for q
  (m < 0) →             -- Given condition
  m ≥ -3 ∧ m < 0        -- Conclusion: range of m is [-3, 0)
  := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1484_148486


namespace NUMINAMATH_CALUDE_commodity_consumption_increase_l1484_148401

theorem commodity_consumption_increase
  (original_tax : ℝ)
  (original_consumption : ℝ)
  (h_tax_positive : original_tax > 0)
  (h_consumption_positive : original_consumption > 0)
  (h_tax_reduction : ℝ)
  (h_revenue_decrease : ℝ)
  (h_consumption_increase : ℝ)
  (h_tax_reduction_eq : h_tax_reduction = 0.20)
  (h_revenue_decrease_eq : h_revenue_decrease = 0.16)
  (h_new_tax : ℝ := original_tax * (1 - h_tax_reduction))
  (h_new_consumption : ℝ := original_consumption * (1 + h_consumption_increase))
  (h_new_revenue : ℝ := h_new_tax * h_new_consumption)
  (h_original_revenue : ℝ := original_tax * original_consumption)
  (h_revenue_equation : h_new_revenue = h_original_revenue * (1 - h_revenue_decrease)) :
  h_consumption_increase = 0.05 := by sorry

end NUMINAMATH_CALUDE_commodity_consumption_increase_l1484_148401


namespace NUMINAMATH_CALUDE_equilateral_triangles_are_similar_l1484_148435

/-- An equilateral triangle is a triangle with all sides equal -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- Similarity of two equilateral triangles -/
def are_similar (t1 t2 : EquilateralTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.side = k * t1.side

/-- Theorem: Any two equilateral triangles are similar -/
theorem equilateral_triangles_are_similar (t1 t2 : EquilateralTriangle) :
  are_similar t1 t2 := by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangles_are_similar_l1484_148435


namespace NUMINAMATH_CALUDE_chickens_and_rabbits_l1484_148496

theorem chickens_and_rabbits (total_heads total_feet : ℕ) 
  (h1 : total_heads = 35) 
  (h2 : total_feet = 94) : 
  ∃ (chickens rabbits : ℕ), 
    chickens + rabbits = total_heads ∧ 
    2 * chickens + 4 * rabbits = total_feet ∧ 
    chickens = 23 ∧ 
    rabbits = 12 := by
  sorry

#check chickens_and_rabbits

end NUMINAMATH_CALUDE_chickens_and_rabbits_l1484_148496


namespace NUMINAMATH_CALUDE_alpine_school_math_players_l1484_148411

/-- The number of players taking mathematics in Alpine School -/
def mathematics_players (total_players physics_players both_players : ℕ) : ℕ :=
  total_players - (physics_players - both_players)

/-- Theorem: Given the conditions, prove that 10 players are taking mathematics -/
theorem alpine_school_math_players :
  mathematics_players 15 9 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_alpine_school_math_players_l1484_148411


namespace NUMINAMATH_CALUDE_standard_deviation_from_variance_l1484_148498

theorem standard_deviation_from_variance (variance : ℝ) (std_dev : ℝ) :
  variance = 2 → std_dev = Real.sqrt variance → std_dev = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_from_variance_l1484_148498


namespace NUMINAMATH_CALUDE_dan_placed_13_scissors_l1484_148406

/-- The number of scissors Dan placed in the drawer -/
def scissors_placed (initial_count final_count : ℕ) : ℕ :=
  final_count - initial_count

/-- Proof that Dan placed 13 scissors in the drawer -/
theorem dan_placed_13_scissors (initial_count final_count : ℕ) 
  (h1 : initial_count = 39)
  (h2 : final_count = 52) : 
  scissors_placed initial_count final_count = 13 := by
  sorry

#eval scissors_placed 39 52  -- Should output 13

end NUMINAMATH_CALUDE_dan_placed_13_scissors_l1484_148406


namespace NUMINAMATH_CALUDE_coin_packing_l1484_148484

theorem coin_packing (n : ℕ) (r R : ℝ) (hn : n > 0) (hr : r > 0) (hR : R > r) :
  (1 / 2 : ℝ) * (R / r - 1) ≤ Real.sqrt n ∧ Real.sqrt n ≤ R / r :=
by sorry

end NUMINAMATH_CALUDE_coin_packing_l1484_148484


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l1484_148474

theorem arctan_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.arctan (2 / x) + Real.arctan (1 / x^2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l1484_148474


namespace NUMINAMATH_CALUDE_second_number_value_l1484_148492

theorem second_number_value (A B : ℝ) : 
  A = 15 → 
  0.4 * A = 0.8 * B + 2 → 
  B = 5 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1484_148492


namespace NUMINAMATH_CALUDE_curve_ellipse_equivalence_l1484_148424

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+4)^2) + Real.sqrt (x^2 + (y-4)^2) = 10

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  y^2/25 + x^2/9 = 1

-- Theorem stating the equivalence of the two equations
theorem curve_ellipse_equivalence :
  ∀ x y : ℝ, curve_equation x y ↔ ellipse_equation x y :=
by sorry

end NUMINAMATH_CALUDE_curve_ellipse_equivalence_l1484_148424


namespace NUMINAMATH_CALUDE_equal_charge_at_120_minutes_l1484_148419

/-- United Telephone's base rate in dollars -/
def united_base : ℚ := 6

/-- United Telephone's per-minute rate in dollars -/
def united_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℚ := 12

/-- Atlantic Call's per-minute rate in dollars -/
def atlantic_per_minute : ℚ := 1/5

/-- The number of minutes at which both companies charge the same amount -/
def equal_charge_minutes : ℚ := 120

theorem equal_charge_at_120_minutes :
  united_base + united_per_minute * equal_charge_minutes =
  atlantic_base + atlantic_per_minute * equal_charge_minutes :=
sorry

end NUMINAMATH_CALUDE_equal_charge_at_120_minutes_l1484_148419


namespace NUMINAMATH_CALUDE_brick_breadth_is_10cm_l1484_148459

/-- Prove that the breadth of a brick is 10 cm given the specified conditions -/
theorem brick_breadth_is_10cm 
  (courtyard_length : ℝ) 
  (courtyard_width : ℝ) 
  (brick_length : ℝ) 
  (total_bricks : ℕ) 
  (h1 : courtyard_length = 20) 
  (h2 : courtyard_width = 16) 
  (h3 : brick_length = 0.2) 
  (h4 : total_bricks = 16000) : 
  ∃ (brick_width : ℝ), brick_width = 0.1 ∧ 
    courtyard_length * courtyard_width = (brick_length * brick_width) * total_bricks :=
by sorry

end NUMINAMATH_CALUDE_brick_breadth_is_10cm_l1484_148459


namespace NUMINAMATH_CALUDE_fourth_term_is_negative_24_l1484_148446

-- Define a geometric sequence
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

-- Define the conditions of our specific sequence
def our_sequence (x : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 1 => x
  | 2 => 3*x + 3
  | 3 => 6*x + 6
  | _ => geometric_sequence x 2 n

-- Theorem statement
theorem fourth_term_is_negative_24 : 
  ∀ x : ℝ, our_sequence x 4 = -24 := by sorry

end NUMINAMATH_CALUDE_fourth_term_is_negative_24_l1484_148446


namespace NUMINAMATH_CALUDE_remainder_theorem_l1484_148444

theorem remainder_theorem (x y u v : ℤ) 
  (x_pos : 0 < x) (y_pos : 0 < y) 
  (division : x = u * y + v) (rem_bound : 0 ≤ v ∧ v < y) : 
  (x + y * u^2 + 3 * v) % y = (4 * v) % y := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1484_148444


namespace NUMINAMATH_CALUDE_sum_abc_values_l1484_148434

theorem sum_abc_values (a b c : ℤ) 
  (h1 : a - 2*b = 4) 
  (h2 : a*b + c^2 - 1 = 0) : 
  a + b + c = 5 ∨ a + b + c = 3 ∨ a + b + c = -1 ∨ a + b + c = -3 :=
sorry

end NUMINAMATH_CALUDE_sum_abc_values_l1484_148434


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l1484_148416

theorem geometric_sequence_common_ratio_sum 
  (k p q : ℝ) 
  (h1 : p ≠ q) 
  (h2 : k ≠ 0) 
  (h3 : k * p^2 - k * q^2 = 5 * (k * p - k * q)) : 
  p + q = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l1484_148416


namespace NUMINAMATH_CALUDE_original_number_l1484_148443

theorem original_number (x : ℝ) : 3 * (2 * x + 5) = 135 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1484_148443


namespace NUMINAMATH_CALUDE_no_prime_sum_53_l1484_148437

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Theorem statement
theorem no_prime_sum_53 : ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 53 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_53_l1484_148437
