import Mathlib

namespace NUMINAMATH_CALUDE_max_sales_revenue_l1359_135942

/-- Sales volume function -/
def f (t : ℕ) : ℝ := -2 * t + 200

/-- Price function -/
def g (t : ℕ) : ℝ :=
  if t ≤ 30 then 0.5 * t + 30 else 40

/-- Daily sales revenue function -/
def S (t : ℕ) : ℝ := f t * g t

/-- The maximum daily sales revenue occurs at t = 20 and is equal to 6400 -/
theorem max_sales_revenue :
  ∃ (t : ℕ), t ∈ Finset.range 50 ∧
  S t = 6400 ∧
  ∀ (t' : ℕ), t' ∈ Finset.range 50 → S t' ≤ S t :=
by sorry

end NUMINAMATH_CALUDE_max_sales_revenue_l1359_135942


namespace NUMINAMATH_CALUDE_total_cats_l1359_135956

theorem total_cats (asleep : ℕ) (awake : ℕ) (h1 : asleep = 92) (h2 : awake = 6) :
  asleep + awake = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_l1359_135956


namespace NUMINAMATH_CALUDE_maria_savings_l1359_135976

/-- The amount left in Maria's savings after buying sweaters and scarves -/
def savings_left (sweater_price scarf_price sweater_count scarf_count initial_savings : ℕ) : ℕ :=
  initial_savings - (sweater_price * sweater_count + scarf_price * scarf_count)

/-- Theorem stating that Maria will have $200 left in her savings -/
theorem maria_savings : savings_left 30 20 6 6 500 = 200 := by
  sorry

end NUMINAMATH_CALUDE_maria_savings_l1359_135976


namespace NUMINAMATH_CALUDE_total_legs_bees_and_spiders_l1359_135950

theorem total_legs_bees_and_spiders :
  let bee_legs : ℕ := 6
  let spider_legs : ℕ := 8
  let num_bees : ℕ := 5
  let num_spiders : ℕ := 2
  (num_bees * bee_legs + num_spiders * spider_legs) = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_total_legs_bees_and_spiders_l1359_135950


namespace NUMINAMATH_CALUDE_vectors_parallel_opposite_l1359_135995

/-- Given vectors a = (-1, 2) and b = (2, -4), prove that they are parallel and in opposite directions. -/
theorem vectors_parallel_opposite (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (2, -4) → ∃ k : ℝ, k < 0 ∧ b = k • a := by sorry

end NUMINAMATH_CALUDE_vectors_parallel_opposite_l1359_135995


namespace NUMINAMATH_CALUDE_new_students_average_age_l1359_135962

/-- Proves that the average age of new students is 32 years given the problem conditions --/
theorem new_students_average_age
  (original_average : ℝ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℝ)
  (h1 : original_average = 40)
  (h2 : original_strength = 12)
  (h3 : new_students = 12)
  (h4 : average_decrease = 4) :
  let new_average := original_average - average_decrease
  let total_original := original_average * original_strength
  let total_new := new_average * (original_strength + new_students) - total_original
  total_new / new_students = 32 := by
  sorry

#check new_students_average_age

end NUMINAMATH_CALUDE_new_students_average_age_l1359_135962


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1359_135913

theorem algebraic_expression_value (x y : ℝ) 
  (eq1 : x + y = 0.2) 
  (eq2 : x + 3*y = 1) : 
  x^2 + 4*x*y + 4*y^2 = 0.36 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1359_135913


namespace NUMINAMATH_CALUDE_max_payment_is_31_l1359_135929

def is_valid_number (n : ℕ) : Prop :=
  2000 ≤ n ∧ n ≤ 2099

def divisibility_payment (n : ℕ) : ℕ :=
  (if n % 1 = 0 then 1 else 0) +
  (if n % 3 = 0 then 3 else 0) +
  (if n % 5 = 0 then 5 else 0) +
  (if n % 7 = 0 then 7 else 0) +
  (if n % 9 = 0 then 9 else 0) +
  (if n % 11 = 0 then 11 else 0)

theorem max_payment_is_31 :
  ∃ n : ℕ, is_valid_number n ∧
    divisibility_payment n = 31 ∧
    ∀ m : ℕ, is_valid_number m → divisibility_payment m ≤ 31 :=
by sorry

end NUMINAMATH_CALUDE_max_payment_is_31_l1359_135929


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1359_135903

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the point of tangency
def P : ℝ × ℝ := (1, 3)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), (
    -- The line y = mx + b passes through P
    m * P.1 + b = P.2 ∧
    -- The slope m is equal to f'(1)
    m = (6 : ℝ) * P.1 - 1 ∧
    -- The resulting equation is 2x - y + 1 = 0
    m = 2 ∧ b = 1 ∧
    ∀ x y, y = m * x + b ↔ 2 * x - y + 1 = 0
  ) := by sorry


end NUMINAMATH_CALUDE_tangent_line_equation_l1359_135903


namespace NUMINAMATH_CALUDE_no_valid_pop_of_223_l1359_135978

/-- Represents the population of Minerva -/
structure MinervaPop where
  people : ℕ
  horses : ℕ
  sheep : ℕ
  cows : ℕ
  ducks : ℕ

/-- Checks if a given population satisfies the Minerva conditions -/
def isValidMinervaPop (pop : MinervaPop) : Prop :=
  pop.people = 4 * pop.horses ∧
  pop.sheep = 3 * pop.cows ∧
  pop.ducks = 2 * pop.people - 2

/-- The total population of Minerva -/
def totalPop (pop : MinervaPop) : ℕ :=
  pop.people + pop.horses + pop.sheep + pop.cows + pop.ducks

/-- Theorem stating that 223 cannot be the total population of Minerva -/
theorem no_valid_pop_of_223 :
  ¬ ∃ (pop : MinervaPop), isValidMinervaPop pop ∧ totalPop pop = 223 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_pop_of_223_l1359_135978


namespace NUMINAMATH_CALUDE_fraction_undefined_at_two_l1359_135981

theorem fraction_undefined_at_two (x : ℝ) : 
  x / (2 - x) = x / (2 - x) → x ≠ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_undefined_at_two_l1359_135981


namespace NUMINAMATH_CALUDE_grade_distribution_l1359_135918

theorem grade_distribution (total_students : ℝ) (prob_A prob_B prob_C : ℝ) :
  total_students = 40 →
  prob_A = 0.6 * prob_B →
  prob_C = 1.5 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  prob_B * total_students = 40 / 3.1 :=
by sorry

end NUMINAMATH_CALUDE_grade_distribution_l1359_135918


namespace NUMINAMATH_CALUDE_malcolm_green_lights_l1359_135939

/-- The number of green lights Malcolm bought -/
def green_lights (red blue green total_needed : ℕ) : Prop :=
  green = total_needed - (red + blue)

/-- Theorem stating the number of green lights Malcolm bought -/
theorem malcolm_green_lights :
  ∃ (green : ℕ), 
    let red := 12
    let blue := 3 * red
    let total_needed := 59 - 5
    green_lights red blue green total_needed ∧ green = 6 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_green_lights_l1359_135939


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l1359_135938

/-- The radius of a circle tangent to eight semicircles lining the inside of a square --/
theorem tangent_circle_radius (square_side : ℝ) (h : square_side = 4) :
  let semicircle_radius : ℝ := square_side / 4
  let diagonal : ℝ := Real.sqrt (square_side ^ 2 / 4 + (square_side / 4) ^ 2)
  diagonal - semicircle_radius = Real.sqrt 5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l1359_135938


namespace NUMINAMATH_CALUDE_tennis_tournament_has_three_cycle_l1359_135975

/-- Represents a tennis tournament as a directed graph -/
structure TennisTournament where
  -- The set of participants
  V : Type
  -- The "wins against" relation
  E : V → V → Prop
  -- There are at least three participants
  atleastThree : ∃ (a b c : V), a ≠ b ∧ b ≠ c ∧ a ≠ c
  -- Every participant plays against every other participant exactly once
  complete : ∀ (a b : V), a ≠ b → (E a b ∨ E b a) ∧ ¬(E a b ∧ E b a)
  -- Every participant wins at least one match
  hasWin : ∀ (a : V), ∃ (b : V), E a b

/-- A 3-cycle in the tournament -/
def HasThreeCycle (T : TennisTournament) : Prop :=
  ∃ (a b c : T.V), T.E a b ∧ T.E b c ∧ T.E c a

/-- The main theorem: every tennis tournament has a 3-cycle -/
theorem tennis_tournament_has_three_cycle (T : TennisTournament) : HasThreeCycle T := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_has_three_cycle_l1359_135975


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1359_135997

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ∃ (x y : ℝ),
  x ≥ 0 ∧ 
  y = Real.sqrt x ∧ 
  x^2 / a^2 - y^2 / b^2 = 1 ∧
  (∃ (m : ℝ), m * (x + 1) = y ∧ m = 1 / (2 * Real.sqrt x)) →
  (Real.sqrt (a^2 + b^2)) / a = (Real.sqrt 5 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1359_135997


namespace NUMINAMATH_CALUDE_round_0_0984_to_two_sig_figs_l1359_135986

/-- Rounds a number to a specified number of significant figures -/
def roundToSignificantFigures (x : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- Theorem: Rounding 0.0984 to two significant figures results in 0.098 -/
theorem round_0_0984_to_two_sig_figs :
  roundToSignificantFigures 0.0984 2 = 0.098 := by
  sorry

end NUMINAMATH_CALUDE_round_0_0984_to_two_sig_figs_l1359_135986


namespace NUMINAMATH_CALUDE_terms_before_five_l1359_135985

/-- An arithmetic sequence with first term 95 and common difference -5 -/
def arithmeticSequence (n : ℕ) : ℤ := 95 - 5 * (n - 1)

theorem terms_before_five : 
  (∃ n : ℕ, arithmeticSequence n = 5) ∧ 
  (∀ k : ℕ, k < 19 → arithmeticSequence k > 5) :=
by sorry

end NUMINAMATH_CALUDE_terms_before_five_l1359_135985


namespace NUMINAMATH_CALUDE_max_rooks_on_100x100_board_l1359_135949

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a nearsighted rook --/
structure NearsightedRook :=
  (range : ℕ)

/-- Calculates the maximum number of non-attacking nearsighted rooks on a chessboard --/
def max_non_attacking_rooks (board : Chessboard) (rook : NearsightedRook) : ℕ :=
  sorry

/-- Theorem stating the maximum number of non-attacking nearsighted rooks on a 100x100 board --/
theorem max_rooks_on_100x100_board :
  let board : Chessboard := ⟨100⟩
  let rook : NearsightedRook := ⟨60⟩
  max_non_attacking_rooks board rook = 178 :=
sorry

end NUMINAMATH_CALUDE_max_rooks_on_100x100_board_l1359_135949


namespace NUMINAMATH_CALUDE_divisibility_by_fifteen_l1359_135904

theorem divisibility_by_fifteen (a : ℤ) :
  15 ∣ ((5 * a + 1) * (3 * a + 2)) ↔ a % 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_fifteen_l1359_135904


namespace NUMINAMATH_CALUDE_longest_collection_pages_l1359_135934

/-- Represents a book collection --/
structure Collection where
  inches_per_page : ℚ
  total_inches : ℚ

/-- Calculates the total number of pages in a collection --/
def total_pages (c : Collection) : ℚ :=
  c.total_inches / c.inches_per_page

theorem longest_collection_pages (miles daphne : Collection)
  (h1 : miles.inches_per_page = 1/5)
  (h2 : daphne.inches_per_page = 1/50)
  (h3 : miles.total_inches = 240)
  (h4 : daphne.total_inches = 25) :
  max (total_pages miles) (total_pages daphne) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_longest_collection_pages_l1359_135934


namespace NUMINAMATH_CALUDE_cycling_jogging_swimming_rates_l1359_135954

theorem cycling_jogging_swimming_rates : ∃ (b j s : ℕ), 
  (3 * b + 2 * j + 4 * s = 66) ∧ 
  (3 * j + 2 * s + 4 * b = 96) ∧ 
  (b^2 + j^2 + s^2 = 612) := by
  sorry

end NUMINAMATH_CALUDE_cycling_jogging_swimming_rates_l1359_135954


namespace NUMINAMATH_CALUDE_thirtieth_term_value_l1359_135961

def arithmeticGeometricSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  if n ≤ 4 then
    a₁ + (n - 1) * d
  else
    2 * arithmeticGeometricSequence a₁ d (n - 1)

theorem thirtieth_term_value :
  arithmeticGeometricSequence 4 3 30 = 436207104 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_value_l1359_135961


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l1359_135946

-- Define the number of DVDs and prices for each store
def store_a_dvds : ℕ := 8
def store_a_price : ℚ := 15
def store_b_dvds : ℕ := 12
def store_b_price : ℚ := 12
def online_dvds : ℕ := 5
def online_price : ℚ := 16.99

-- Define the discount percentage
def discount_percent : ℚ := 15

-- Define the total cost function
def total_cost (store_a_dvds store_b_dvds online_dvds : ℕ) 
               (store_a_price store_b_price online_price : ℚ) 
               (discount_percent : ℚ) : ℚ :=
  let physical_store_cost := store_a_dvds * store_a_price + store_b_dvds * store_b_price
  let online_store_cost := online_dvds * online_price
  let discount := physical_store_cost * (discount_percent / 100)
  (physical_store_cost - discount) + online_store_cost

-- Theorem statement
theorem total_cost_is_correct : 
  total_cost store_a_dvds store_b_dvds online_dvds 
             store_a_price store_b_price online_price 
             discount_percent = 309.35 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l1359_135946


namespace NUMINAMATH_CALUDE_sum_between_15_and_16_l1359_135974

theorem sum_between_15_and_16 : 
  let a : ℚ := 10/3
  let b : ℚ := 19/4
  let c : ℚ := 123/20
  15 < a + b + c ∧ a + b + c < 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_between_15_and_16_l1359_135974


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l1359_135988

/-- A parabola with directrix y = -4 has the standard equation x² = 16y -/
theorem parabola_standard_equation (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, y = -4 → (x^2 = 2*p*y ↔ x^2 = 16*y)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l1359_135988


namespace NUMINAMATH_CALUDE_logarithm_simplification_l1359_135999

open Real

theorem logarithm_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) (hy_neq_1 : y ≠ 1) :
  (log x^2 / log y^8) * (log y^5 / log x^4) * (log x^3 / log y^5) * (log y^8 / log x^3) * (log x^4 / log y^3) = 
  (1/3) * (log x / log y) := by
sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l1359_135999


namespace NUMINAMATH_CALUDE_rectangle_area_l1359_135914

theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let perimeter := 2 * (l + b)
  perimeter = 64 → l * b = 192 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1359_135914


namespace NUMINAMATH_CALUDE_variable_value_l1359_135931

theorem variable_value (w x v : ℝ) 
  (h1 : 5 / w + 5 / x = 5 / v) 
  (h2 : w * x = v)
  (h3 : (w + x) / 2 = 0.5) : 
  v = 0.25 := by
sorry

end NUMINAMATH_CALUDE_variable_value_l1359_135931


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1359_135908

theorem absolute_value_inequality_solution_set (x : ℝ) :
  (|x - 1| < 1) ↔ (x ∈ Set.Ioo 0 2) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1359_135908


namespace NUMINAMATH_CALUDE_max_area_rectangle_perimeter_24_l1359_135968

/-- The maximum area of a rectangle with perimeter 24 is 36 -/
theorem max_area_rectangle_perimeter_24 :
  ∀ (length width : ℝ), length > 0 → width > 0 →
  2 * (length + width) = 24 →
  length * width ≤ 36 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_perimeter_24_l1359_135968


namespace NUMINAMATH_CALUDE_expansion_properties_l1359_135947

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the kth term in the expansion of (x + 1/(2√x))^n -/
def coefficient (n k : ℕ) : ℚ := (1 / 2^k : ℚ) * binomial n k

/-- The expansion of (x + 1/(2√x))^n has its first three coefficients in arithmetic sequence -/
def first_three_in_arithmetic_sequence (n : ℕ) : Prop :=
  coefficient n 0 + coefficient n 2 = 2 * coefficient n 1

/-- The kth term has the maximum coefficient in the expansion -/
def max_coefficient (n k : ℕ) : Prop :=
  ∀ i, i ≠ k → coefficient n k ≥ coefficient n i

theorem expansion_properties :
  ∃ n : ℕ,
    first_three_in_arithmetic_sequence n ∧
    max_coefficient n 2 ∧
    max_coefficient n 3 ∧
    ∀ k, k ≠ 2 ∧ k ≠ 3 → ¬(max_coefficient n k) :=
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l1359_135947


namespace NUMINAMATH_CALUDE_dividend_calculation_l1359_135958

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 158 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1359_135958


namespace NUMINAMATH_CALUDE_tiles_count_theorem_l1359_135926

/-- Represents a square floor tiled with congruent square tiles -/
structure TiledSquare where
  side_length : ℕ

/-- The number of tiles along the diagonals and central line of a tiled square -/
def diagonal_and_central_count (s : TiledSquare) : ℕ :=
  3 * s.side_length - 2

/-- The total number of tiles covering the floor -/
def total_tiles (s : TiledSquare) : ℕ :=
  s.side_length ^ 2

/-- Theorem stating that if the diagonal and central count is 55, 
    then the total number of tiles is 361 -/
theorem tiles_count_theorem (s : TiledSquare) :
  diagonal_and_central_count s = 55 → total_tiles s = 361 := by
  sorry

end NUMINAMATH_CALUDE_tiles_count_theorem_l1359_135926


namespace NUMINAMATH_CALUDE_not_both_bidirectional_l1359_135906

-- Define the proof methods
inductive ProofMethod
| Synthetic
| Analytic

-- Define the reasoning directions
inductive ReasoningDirection
| CauseToEffect
| EffectToCause

-- Define the properties of the proof methods
def methodProperties (m : ProofMethod) : ReasoningDirection :=
  match m with
  | ProofMethod.Synthetic => ReasoningDirection.CauseToEffect
  | ProofMethod.Analytic => ReasoningDirection.EffectToCause

-- Theorem statement
theorem not_both_bidirectional : 
  ¬ (∀ (m : ProofMethod), 
      methodProperties m = ReasoningDirection.CauseToEffect ∧ 
      methodProperties m = ReasoningDirection.EffectToCause) :=
by sorry

end NUMINAMATH_CALUDE_not_both_bidirectional_l1359_135906


namespace NUMINAMATH_CALUDE_no_integer_roots_l1359_135922

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Evaluates a polynomial at a given integer -/
def eval (p : IntPolynomial) (x : ℤ) : ℤ := p x

/-- Predicate for odd integers -/
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem no_integer_roots (p : IntPolynomial) 
  (h0 : is_odd (eval p 0)) 
  (h1 : is_odd (eval p 1)) : 
  ∀ k : ℤ, eval p k ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_integer_roots_l1359_135922


namespace NUMINAMATH_CALUDE_unique_bases_sum_l1359_135994

theorem unique_bases_sum : ∃! (R₃ R₄ : ℕ), 
  (R₃ > 0 ∧ R₄ > 0) ∧
  ((4 * R₃ + 6) * (R₄^2 - 1) = (4 * R₄ + 9) * (R₃^2 - 1)) ∧
  ((6 * R₃ + 4) * (R₄^2 - 1) = (9 * R₄ + 4) * (R₃^2 - 1)) ∧
  (R₃ + R₄ = 23) := by
  sorry

end NUMINAMATH_CALUDE_unique_bases_sum_l1359_135994


namespace NUMINAMATH_CALUDE_train_crossing_time_l1359_135971

/-- Proves the time it takes for a train to cross a stationary man on a platform --/
theorem train_crossing_time (train_speed_kmph : ℝ) (train_speed_mps : ℝ) 
  (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed_kmph = 72 →
  train_speed_mps = 20 →
  platform_length = 300 →
  platform_crossing_time = 33 →
  ∃ (train_length : ℝ),
    train_length = train_speed_mps * platform_crossing_time - platform_length ∧
    train_length / train_speed_mps = 18 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1359_135971


namespace NUMINAMATH_CALUDE_unique_perfect_square_solution_l1359_135901

theorem unique_perfect_square_solution : 
  ∃! (n : ℕ), n > 0 ∧ ∃ (m : ℕ), n^4 - n^3 + 3*n^2 + 5 = m^2 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_solution_l1359_135901


namespace NUMINAMATH_CALUDE_average_shift_l1359_135932

theorem average_shift (x₁ x₂ x₃ x₄ : ℝ) :
  (x₁ + x₂ + x₃ + x₄) / 4 = 2 →
  ((x₁ + 3) + (x₂ + 3) + (x₃ + 3) + (x₄ + 3)) / 4 = 5 := by
sorry

end NUMINAMATH_CALUDE_average_shift_l1359_135932


namespace NUMINAMATH_CALUDE_alligator_growth_in_year_l1359_135969

def initial_population : ℝ := 4
def growth_factor : ℝ := 1.5
def months : ℕ := 12

def alligator_population (t : ℕ) : ℝ :=
  initial_population * growth_factor ^ t

theorem alligator_growth_in_year :
  alligator_population months = 518.9853515625 :=
sorry

end NUMINAMATH_CALUDE_alligator_growth_in_year_l1359_135969


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1359_135960

/-- The number of schools -/
def num_schools : ℕ := 4

/-- The number of members from each school -/
def members_per_school : ℕ := 6

/-- The number of representatives from the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives from each non-host school -/
def other_representatives : ℕ := 1

/-- The total number of members in the club -/
def total_members : ℕ := num_schools * members_per_school

/-- The number of ways to arrange the presidency meeting -/
def meeting_arrangements : ℕ := 
  num_schools * (members_per_school.choose host_representatives) * 
  (members_per_school.choose other_representatives)^(num_schools - 1)

theorem presidency_meeting_arrangements : 
  meeting_arrangements = 17280 :=
sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1359_135960


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l1359_135953

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallelLines : Line → Line → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (perpendicularLineToPlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_condition 
  (m n : Line) (α β : Plane) :
  perpendicularLineToPlane m α →
  perpendicularLineToPlane n β →
  parallelLines m n →
  parallelPlanes α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l1359_135953


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l1359_135998

/-- Given two points on a line and another line equation, prove the value of k -/
theorem parallel_lines_k_value (k : ℝ) : 
  (∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b → 
    (x = 3 ∧ y = -12) ∨ (x = k ∧ y = 22)) ∧
   (∀ x y : ℝ, 4 * x + 6 * y = 36 → y = m * x + (36 / 6 - 4 * x / 6))) →
  k = -48 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l1359_135998


namespace NUMINAMATH_CALUDE_expression_evaluation_l1359_135919

theorem expression_evaluation : 
  20 * ((150 / 3) + (40 / 5) + (16 / 25) + 2) = 1212.8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1359_135919


namespace NUMINAMATH_CALUDE_circle_and_tangent_properties_l1359_135977

/-- Given a circle with center C on the line x-y+1=0 and passing through points A(1,1) and B(2,-2) -/
structure CircleData where
  C : ℝ × ℝ
  center_on_line : C.1 - C.2 + 1 = 0
  passes_through_A : (C.1 - 1)^2 + (C.2 - 1)^2 = (C.1 - 2)^2 + (C.2 + 2)^2

/-- The standard equation of the circle is (x+3)^2 + (y+2)^2 = 25 -/
def circle_equation (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 2)^2 = 25

/-- The equation of the tangent line passing through point (1,1) is 4x + 3y - 7 = 0 -/
def tangent_line_equation (x y : ℝ) : Prop :=
  4*x + 3*y - 7 = 0

theorem circle_and_tangent_properties (data : CircleData) :
  (∀ x y, circle_equation x y ↔ ((x - data.C.1)^2 + (y - data.C.2)^2 = (1 - data.C.1)^2 + (1 - data.C.2)^2)) ∧
  tangent_line_equation 1 1 ∧
  (∀ x y, tangent_line_equation x y → (x - 1) * (1 - data.C.1) + (y - 1) * (1 - data.C.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_properties_l1359_135977


namespace NUMINAMATH_CALUDE_P_not_in_second_quadrant_l1359_135941

/-- A point is in the second quadrant if its x-coordinate is negative and its y-coordinate is positive -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The coordinates of point P as a function of m -/
def P (m : ℝ) : ℝ × ℝ := (m^2 + m, m - 1)

/-- Theorem stating that P(m) cannot be in the second quadrant for any real m -/
theorem P_not_in_second_quadrant (m : ℝ) : ¬ second_quadrant (P m).1 (P m).2 := by
  sorry

end NUMINAMATH_CALUDE_P_not_in_second_quadrant_l1359_135941


namespace NUMINAMATH_CALUDE_max_value_problem_l1359_135916

theorem max_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 2 * Real.sqrt (x * y) - 4 * x^2 - y^2 ≤ 2 * Real.sqrt (a * b) - 4 * a^2 - b^2) →
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 = (Real.sqrt 2 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l1359_135916


namespace NUMINAMATH_CALUDE_regression_unit_change_l1359_135973

/-- Represents a linear regression equation of the form y = mx + b -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- The change in y for a unit change in x in a linear regression -/
def unitChange (reg : LinearRegression) : ℝ := reg.slope

theorem regression_unit_change 
  (reg : LinearRegression) 
  (h : reg = { slope := -1.5, intercept := 2 }) : 
  unitChange reg = -1.5 := by sorry

end NUMINAMATH_CALUDE_regression_unit_change_l1359_135973


namespace NUMINAMATH_CALUDE_great_eighteen_hockey_league_games_l1359_135957

/-- Represents a sports league with the given structure -/
structure League where
  total_teams : ℕ
  divisions : ℕ
  teams_per_division : ℕ
  intra_division_games : ℕ
  inter_division_games : ℕ

/-- Calculates the total number of games in the league -/
def total_games (l : League) : ℕ :=
  (l.total_teams * (l.teams_per_division - 1) * l.intra_division_games +
   l.total_teams * (l.total_teams - l.teams_per_division) * l.inter_division_games) / 2

/-- Theorem stating that the given league structure results in 243 total games -/
theorem great_eighteen_hockey_league_games :
  ∃ (l : League),
    l.total_teams = 18 ∧
    l.divisions = 3 ∧
    l.teams_per_division = 6 ∧
    l.intra_division_games = 3 ∧
    l.inter_division_games = 1 ∧
    total_games l = 243 := by
  sorry


end NUMINAMATH_CALUDE_great_eighteen_hockey_league_games_l1359_135957


namespace NUMINAMATH_CALUDE_supermarket_theorem_l1359_135911

/-- Represents the supermarket's agricultural product distribution problem -/
structure SupermarketProblem where
  total_boxes : ℕ
  brand_a_cost : ℝ
  brand_a_price : ℝ
  brand_b_cost : ℝ
  brand_b_price : ℝ
  total_expenditure : ℝ
  min_total_profit : ℝ

/-- Theorem for the supermarket problem -/
theorem supermarket_theorem (p : SupermarketProblem)
  (h_total : p.total_boxes = 100)
  (h_a_cost : p.brand_a_cost = 80)
  (h_a_price : p.brand_a_price = 120)
  (h_b_cost : p.brand_b_cost = 130)
  (h_b_price : p.brand_b_price = 200)
  (h_expenditure : p.total_expenditure = 10000)
  (h_min_profit : p.min_total_profit = 5600) :
  (∃ (x y : ℕ), x + y = p.total_boxes ∧ 
    p.brand_a_cost * x + p.brand_b_cost * y = p.total_expenditure ∧
    x = 60 ∧ y = 40) ∧
  (∃ (z : ℕ), z ≥ 54 ∧
    (p.brand_a_price - p.brand_a_cost) * (p.total_boxes - z) +
    (p.brand_b_price - p.brand_b_cost) * z ≥ p.min_total_profit) :=
by sorry


end NUMINAMATH_CALUDE_supermarket_theorem_l1359_135911


namespace NUMINAMATH_CALUDE_palm_meadows_beds_l1359_135980

theorem palm_meadows_beds (total_rooms : ℕ) (two_bed_rooms : ℕ) (total_beds : ℕ) 
  (h1 : total_rooms = 13)
  (h2 : two_bed_rooms = 8)
  (h3 : total_beds = 31)
  (h4 : two_bed_rooms ≤ total_rooms) :
  (total_beds - 2 * two_bed_rooms) / (total_rooms - two_bed_rooms) = 3 := by
  sorry

end NUMINAMATH_CALUDE_palm_meadows_beds_l1359_135980


namespace NUMINAMATH_CALUDE_worker_count_l1359_135992

theorem worker_count (total : ℕ) (increased_total : ℕ) (extra_contribution : ℕ) : 
  (total = 300000) → 
  (increased_total = 325000) → 
  (extra_contribution = 50) → 
  (∃ (n : ℕ), (n * (total / n) = total) ∧ 
              (n * (total / n + extra_contribution) = increased_total) ∧ 
              (n = 500)) := by
  sorry

end NUMINAMATH_CALUDE_worker_count_l1359_135992


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1359_135923

def polynomial (z : ℂ) : ℂ := z^9 + z^7 - z^5 + z^3 - z

def is_root (z : ℂ) : Prop := polynomial z = 0

def imaginary_part (z : ℂ) : ℝ := z.im

theorem max_imaginary_part_of_roots :
  ∃ (θ : ℝ), 
    -π/2 ≤ θ ∧ θ ≤ π/2 ∧
    (∀ (z : ℂ), is_root z → imaginary_part z ≤ Real.sin θ) ∧
    θ = π/2 :=
sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1359_135923


namespace NUMINAMATH_CALUDE_units_digit_of_subtraction_is_seven_l1359_135917

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Converts a ThreeDigitNumber to its integer value -/
def to_int (n : ThreeDigitNumber) : Int :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Reverses a ThreeDigitNumber -/
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.units
  tens := n.tens
  units := n.hundreds
  is_valid := by sorry

/-- The main theorem -/
theorem units_digit_of_subtraction_is_seven (n : ThreeDigitNumber) 
  (h : n.hundreds = n.units + 3) : 
  (to_int n - to_int (reverse n)) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_subtraction_is_seven_l1359_135917


namespace NUMINAMATH_CALUDE_manager_average_salary_l1359_135972

/-- Proves that the average salary of managers is $90,000 given the conditions of the company. -/
theorem manager_average_salary 
  (num_managers : ℕ) 
  (num_associates : ℕ) 
  (associate_avg_salary : ℚ) 
  (company_avg_salary : ℚ) : 
  num_managers = 15 → 
  num_associates = 75 → 
  associate_avg_salary = 30000 → 
  company_avg_salary = 40000 → 
  (num_managers * (num_managers * company_avg_salary - num_associates * associate_avg_salary)) / 
   (num_managers * (num_managers + num_associates)) = 90000 := by
  sorry

end NUMINAMATH_CALUDE_manager_average_salary_l1359_135972


namespace NUMINAMATH_CALUDE_point_P_satisfies_conditions_l1359_135940

def P₁ : ℝ × ℝ := (1, 3)
def P₂ : ℝ × ℝ := (4, -6)
def P : ℝ × ℝ := (3, -3)

def vector (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, vector A C = (t • (vector A B).1, t • (vector A B).2)

theorem point_P_satisfies_conditions :
  collinear P₁ P₂ P ∧ vector P₁ P = (2 • (vector P P₂).1, 2 • (vector P P₂).2) := by
  sorry

end NUMINAMATH_CALUDE_point_P_satisfies_conditions_l1359_135940


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_zero_l1359_135987

theorem fraction_zero_implies_x_zero (x : ℚ) : 
  x / (2 * x - 1) = 0 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_zero_l1359_135987


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1359_135951

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (|2*x - 3| < 1 → x*(x - 3) < 0)) ∧
  (∃ x : ℝ, x*(x - 3) < 0 ∧ ¬(|2*x - 3| < 1)) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1359_135951


namespace NUMINAMATH_CALUDE_ratio_problem_l1359_135902

theorem ratio_problem (w x y z : ℚ) 
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) :
  w / y = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1359_135902


namespace NUMINAMATH_CALUDE_recipe_reduction_l1359_135907

/-- Represents a mixed number as a pair of integers (whole, fractional) -/
def MixedNumber := ℤ × ℚ

/-- Converts a mixed number to a rational number -/
def mixedToRational (m : MixedNumber) : ℚ :=
  m.1 + m.2

/-- The amount of flour in the original recipe -/
def originalFlour : MixedNumber := (5, 3/4)

/-- The amount of sugar in the original recipe -/
def originalSugar : MixedNumber := (2, 1/2)

/-- The fraction of the recipe we want to make -/
def recipeFraction : ℚ := 1/3

theorem recipe_reduction :
  (mixedToRational originalFlour * recipeFraction = 23/12) ∧
  (mixedToRational originalSugar * recipeFraction = 5/6) :=
sorry

end NUMINAMATH_CALUDE_recipe_reduction_l1359_135907


namespace NUMINAMATH_CALUDE_roundness_of_eight_million_l1359_135964

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 8,000,000 is 15 -/
theorem roundness_of_eight_million : roundness 8000000 = 15 := by sorry

end NUMINAMATH_CALUDE_roundness_of_eight_million_l1359_135964


namespace NUMINAMATH_CALUDE_probability_independent_events_l1359_135970

theorem probability_independent_events (a b : Set α) (p : Set α → ℝ) 
  (h1 : p a = 4/7)
  (h2 : p (a ∩ b) = 0.22857142857142856)
  (h3 : p (a ∩ b) = p a * p b) : 
  p b = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_probability_independent_events_l1359_135970


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l1359_135983

-- Define the function f'(x)
def f' (x : ℝ) : ℝ := x^2 - 2*x - 3

-- State the theorem
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3, f' x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l1359_135983


namespace NUMINAMATH_CALUDE_rain_probability_theorem_l1359_135982

/-- Given probabilities for rain events in counties -/
theorem rain_probability_theorem 
  (p_monday : ℝ) 
  (p_neither : ℝ) 
  (p_both : ℝ) 
  (h1 : p_monday = 0.7) 
  (h2 : p_neither = 0.35) 
  (h3 : p_both = 0.6) :
  ∃ (p_tuesday : ℝ), p_tuesday = 0.55 := by
sorry


end NUMINAMATH_CALUDE_rain_probability_theorem_l1359_135982


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1359_135909

/-- The distance between the foci of a hyperbola defined by xy = 4 is 8 -/
theorem hyperbola_foci_distance : 
  ∀ (x y : ℝ), x * y = 4 → 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 * f₁.2 = 4 ∧ f₂.1 * f₂.2 = 4) ∧ 
    ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2)^(1/2 : ℝ) = 8 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1359_135909


namespace NUMINAMATH_CALUDE_sum_of_four_sequential_terms_l1359_135933

theorem sum_of_four_sequential_terms (n : ℝ) : 
  n + (n + 1) + (n + 2) + (n + 3) = 20 → n = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_sequential_terms_l1359_135933


namespace NUMINAMATH_CALUDE_nearly_regular_polyhedra_theorem_l1359_135928

/-- A structure representing a polyhedron -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Definition of a nearly regular polyhedron -/
def NearlyRegularPolyhedron (p : Polyhedron) : Prop := sorry

/-- Intersection of two polyhedra -/
def intersect (p1 p2 : Polyhedron) : Polyhedron := sorry

/-- Tetrahedron -/
def Tetrahedron : Polyhedron := ⟨4, 6, 4⟩

/-- Octahedron -/
def Octahedron : Polyhedron := ⟨8, 12, 6⟩

/-- Cube -/
def Cube : Polyhedron := ⟨6, 12, 8⟩

/-- Dodecahedron -/
def Dodecahedron : Polyhedron := ⟨12, 30, 20⟩

/-- Icosahedron -/
def Icosahedron : Polyhedron := ⟨20, 30, 12⟩

/-- The set of nearly regular polyhedra -/
def NearlyRegularPolyhedra : Set Polyhedron := sorry

theorem nearly_regular_polyhedra_theorem :
  ∃ (p1 p2 p3 p4 p5 : Polyhedron),
    p1 ∈ NearlyRegularPolyhedra ∧
    p2 ∈ NearlyRegularPolyhedra ∧
    p3 ∈ NearlyRegularPolyhedra ∧
    p4 ∈ NearlyRegularPolyhedra ∧
    p5 ∈ NearlyRegularPolyhedra ∧
    p1 = intersect Tetrahedron Octahedron ∧
    p2 = intersect Cube Octahedron ∧
    p3 = intersect Dodecahedron Icosahedron ∧
    NearlyRegularPolyhedron p4 ∧
    NearlyRegularPolyhedron p5 :=
  sorry

end NUMINAMATH_CALUDE_nearly_regular_polyhedra_theorem_l1359_135928


namespace NUMINAMATH_CALUDE_hotpot_revenue_problem_l1359_135937

/-- Represents the revenue from different sources in a hotpot restaurant -/
structure HotpotRevenue where
  diningIn : ℝ
  takeout : ℝ
  stall : ℝ

/-- The revenue increase from different sources in July -/
structure JulyIncrease where
  diningIn : ℝ
  takeout : ℝ
  stall : ℝ

/-- Theorem representing the hotpot restaurant revenue problem -/
theorem hotpot_revenue_problem 
  (june : HotpotRevenue) 
  (july_increase : JulyIncrease) 
  (july : HotpotRevenue) :
  -- June revenue ratio condition
  june.diningIn / june.takeout = 3 / 5 ∧ 
  june.takeout / june.stall = 5 / 2 ∧
  -- July stall revenue increase condition
  july_increase.stall = 2 / 5 * (july_increase.diningIn + july_increase.takeout + july_increase.stall) ∧
  -- July stall revenue proportion condition
  july.stall / (july.diningIn + july.takeout + july.stall) = 7 / 20 ∧
  -- July dining in to takeout ratio condition
  july.diningIn / july.takeout = 8 / 5 ∧
  -- July revenue calculation
  july.diningIn = june.diningIn + july_increase.diningIn ∧
  july.takeout = june.takeout + july_increase.takeout ∧
  july.stall = june.stall + july_increase.stall
  →
  -- Conclusion: Additional takeout revenue in July compared to total July revenue
  july_increase.takeout / (july.diningIn + july.takeout + july.stall) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_hotpot_revenue_problem_l1359_135937


namespace NUMINAMATH_CALUDE_outfit_count_l1359_135912

/-- The number of different outfits with different colored shirt and hat -/
def number_of_outfits (blue_shirts green_shirts pants blue_hats green_hats : ℕ) : ℕ :=
  (blue_shirts * green_hats * pants) + (green_shirts * blue_hats * pants)

/-- Theorem stating the number of outfits given the specific quantities -/
theorem outfit_count :
  number_of_outfits 7 6 7 10 9 = 861 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l1359_135912


namespace NUMINAMATH_CALUDE_total_nuts_is_half_cup_l1359_135900

/-- The amount of walnuts Karen added to the trail mix in cups -/
def walnuts : ℚ := 0.25

/-- The amount of almonds Karen added to the trail mix in cups -/
def almonds : ℚ := 0.25

/-- The total amount of nuts Karen added to the trail mix in cups -/
def total_nuts : ℚ := walnuts + almonds

/-- Theorem stating that the total amount of nuts Karen added is 0.50 cups -/
theorem total_nuts_is_half_cup : total_nuts = 0.50 := by sorry

end NUMINAMATH_CALUDE_total_nuts_is_half_cup_l1359_135900


namespace NUMINAMATH_CALUDE_total_sales_is_28_l1359_135924

/-- The number of crates of eggs Gabrielle sells on Monday -/
def monday_sales : ℕ := 5

/-- The number of crates of eggs Gabrielle sells on Tuesday -/
def tuesday_sales : ℕ := 2 * monday_sales

/-- The number of crates of eggs Gabrielle sells on Wednesday -/
def wednesday_sales : ℕ := tuesday_sales - 2

/-- The number of crates of eggs Gabrielle sells on Thursday -/
def thursday_sales : ℕ := tuesday_sales / 2

/-- The total number of crates of eggs Gabrielle sells over 4 days -/
def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales

theorem total_sales_is_28 : total_sales = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_is_28_l1359_135924


namespace NUMINAMATH_CALUDE_shopkeeper_red_cards_l1359_135944

/-- Calculates the total number of red cards in all decks --/
def total_red_cards (total_decks : ℕ) (standard_decks : ℕ) (special_decks : ℕ) 
  (red_cards_standard : ℕ) (additional_red_cards_special : ℕ) : ℕ :=
  (standard_decks * red_cards_standard) + 
  (special_decks * (red_cards_standard + additional_red_cards_special))

theorem shopkeeper_red_cards : 
  total_red_cards 15 5 10 26 4 = 430 := by
  sorry

#eval total_red_cards 15 5 10 26 4

end NUMINAMATH_CALUDE_shopkeeper_red_cards_l1359_135944


namespace NUMINAMATH_CALUDE_third_artist_set_duration_l1359_135925

/-- The duration of the music festival in minutes -/
def festival_duration : ℕ := 6 * 60

/-- The duration of the first artist's set in minutes -/
def first_artist_set : ℕ := 70 + 5

/-- The duration of the second artist's set in minutes -/
def second_artist_set : ℕ := 15 * 4 + 6 * 7 + 15 + 2 * 10

/-- The duration of the third artist's set in minutes -/
def third_artist_set : ℕ := festival_duration - first_artist_set - second_artist_set

theorem third_artist_set_duration : third_artist_set = 148 := by
  sorry

end NUMINAMATH_CALUDE_third_artist_set_duration_l1359_135925


namespace NUMINAMATH_CALUDE_book_words_per_page_l1359_135952

theorem book_words_per_page 
  (total_pages : ℕ) 
  (words_per_page : ℕ) 
  (max_words_per_page : ℕ) 
  (total_words_mod : ℕ) :
  total_pages = 150 →
  words_per_page ≤ max_words_per_page →
  max_words_per_page = 120 →
  (total_pages * words_per_page) % 221 = total_words_mod →
  total_words_mod = 200 →
  words_per_page = 118 := by
sorry

end NUMINAMATH_CALUDE_book_words_per_page_l1359_135952


namespace NUMINAMATH_CALUDE_sum_four_consecutive_odd_divisible_by_two_l1359_135967

theorem sum_four_consecutive_odd_divisible_by_two (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1) + (2*n + 3) + (2*n + 5) + (2*n + 7) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_four_consecutive_odd_divisible_by_two_l1359_135967


namespace NUMINAMATH_CALUDE_students_liking_food_l1359_135990

theorem students_liking_food (total : ℕ) (dislike : ℕ) (like : ℕ) : 
  total = 814 → dislike = 431 → like = total - dislike → like = 383 := by sorry

end NUMINAMATH_CALUDE_students_liking_food_l1359_135990


namespace NUMINAMATH_CALUDE_last_digit_power_of_two_divisibility_l1359_135948

theorem last_digit_power_of_two_divisibility (k : ℕ) (N a A : ℕ) :
  k ≥ 3 →
  N = 2^k →
  a = N % 10 →
  A * 10 + a = N →
  6 ∣ a * A :=
by sorry

end NUMINAMATH_CALUDE_last_digit_power_of_two_divisibility_l1359_135948


namespace NUMINAMATH_CALUDE_sin_pi_fourth_plus_alpha_l1359_135927

theorem sin_pi_fourth_plus_alpha (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.tan (α - π/4) = 1/3) : Real.sin (π/4 + α) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_fourth_plus_alpha_l1359_135927


namespace NUMINAMATH_CALUDE_rationalize_and_product_l1359_135920

theorem rationalize_and_product : ∃ (A B C : ℚ),
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C ∧
  A = 11/4 ∧ B = 5/4 ∧ C = 5 ∧ A * B * C = 275/16 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_product_l1359_135920


namespace NUMINAMATH_CALUDE_vector_sum_zero_l1359_135965

variable {V : Type*} [AddCommGroup V]

/-- Given four points A, B, C, and D in a vector space, 
    prove that AB + BD - AC - CD equals the zero vector -/
theorem vector_sum_zero (A B C D : V) : 
  (B - A) + (D - B) - (C - A) - (D - C) = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l1359_135965


namespace NUMINAMATH_CALUDE_equation_solution_l1359_135991

theorem equation_solution : ∃ x : ℝ, x ≠ -2 ∧ (4*x^2 - 3*x + 2) / (x + 2) = 4*x - 3 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1359_135991


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l1359_135993

theorem sum_of_roots_cubic_equation : 
  let f (x : ℝ) := (x^3 - 3*x^2 - 12*x) / (x + 3)
  ∃ (x₁ x₂ : ℝ), (f x₁ = 7 ∧ f x₂ = 7 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l1359_135993


namespace NUMINAMATH_CALUDE_probability_same_number_l1359_135955

def emily_options : ℕ := 250 / 20
def eli_options : ℕ := 250 / 30
def common_options : ℕ := 250 / 60

theorem probability_same_number : 
  (emily_options : ℚ) * eli_options ≠ 0 →
  (common_options : ℚ) / (emily_options * eli_options) = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_number_l1359_135955


namespace NUMINAMATH_CALUDE_max_storage_period_is_56_days_l1359_135921

/-- Represents the financial parameters for a wholesale product --/
structure WholesaleProduct where
  wholesalePrice : ℝ
  grossProfitMargin : ℝ
  borrowedCapitalRatio : ℝ
  monthlyInterestRate : ℝ
  dailyStorageCost : ℝ

/-- Calculates the maximum storage period without incurring a loss --/
def maxStoragePeriod (p : WholesaleProduct) : ℕ :=
  sorry

/-- Theorem stating the maximum storage period for the given product --/
theorem max_storage_period_is_56_days (p : WholesaleProduct)
  (h1 : p.wholesalePrice = 500)
  (h2 : p.grossProfitMargin = 0.04)
  (h3 : p.borrowedCapitalRatio = 0.8)
  (h4 : p.monthlyInterestRate = 0.0042)
  (h5 : p.dailyStorageCost = 0.30) :
  maxStoragePeriod p = 56 :=
sorry

end NUMINAMATH_CALUDE_max_storage_period_is_56_days_l1359_135921


namespace NUMINAMATH_CALUDE_trays_from_second_table_l1359_135915

theorem trays_from_second_table
  (trays_per_trip : ℕ)
  (num_trips : ℕ)
  (trays_from_first_table : ℕ)
  (h1 : trays_per_trip = 4)
  (h2 : num_trips = 3)
  (h3 : trays_from_first_table = 10) :
  trays_per_trip * num_trips - trays_from_first_table = 2 :=
by sorry

end NUMINAMATH_CALUDE_trays_from_second_table_l1359_135915


namespace NUMINAMATH_CALUDE_square_side_length_l1359_135945

/-- Proves that a square with perimeter 52 cm and area 169 square cm has sides of length 13 cm -/
theorem square_side_length (s : ℝ) 
  (perimeter : s * 4 = 52) 
  (area : s * s = 169) : 
  s = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1359_135945


namespace NUMINAMATH_CALUDE_union_M_N_when_a_9_M_superset_N_iff_a_range_l1359_135959

-- Define the sets M and N
def M : Set ℝ := {x | (x + 5) / (x - 8) ≥ 0}
def N (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}

-- Theorem for part 1
theorem union_M_N_when_a_9 :
  M ∪ N 9 = {x : ℝ | x ≤ -5 ∨ x ≥ 8} := by sorry

-- Theorem for part 2
theorem M_superset_N_iff_a_range (a : ℝ) :
  M ⊇ N a ↔ a ≤ -6 ∨ a > 9 := by sorry

end NUMINAMATH_CALUDE_union_M_N_when_a_9_M_superset_N_iff_a_range_l1359_135959


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_60_l1359_135963

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the general term of the expansion
def generalTerm (r : ℕ) : ℚ :=
  (-1)^r * binomial 6 r * 2^r

-- Theorem statement
theorem coefficient_x_squared_is_60 :
  generalTerm 2 = 60 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_60_l1359_135963


namespace NUMINAMATH_CALUDE_book_purchase_problem_l1359_135943

/-- Given information about book purchases, prove the number of people who purchased both books --/
theorem book_purchase_problem (A B AB : ℕ) 
  (h1 : A = 2 * B)
  (h2 : AB = 2 * (B - AB))
  (h3 : A - AB = 1000) :
  AB = 500 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l1359_135943


namespace NUMINAMATH_CALUDE_M_equals_N_l1359_135989

-- Define set M
def M : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi) / 2 + Real.pi / 4}

-- Define set N
def N : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 4 ∨ x = k * Real.pi - Real.pi / 4}

-- Theorem stating M = N
theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l1359_135989


namespace NUMINAMATH_CALUDE_draw_probability_value_l1359_135979

/-- The number of green chips in the bag -/
def green_chips : ℕ := 4

/-- The number of blue chips in the bag -/
def blue_chips : ℕ := 3

/-- The number of yellow chips in the bag -/
def yellow_chips : ℕ := 5

/-- The total number of chips in the bag -/
def total_chips : ℕ := green_chips + blue_chips + yellow_chips

/-- The number of ways to arrange the color groups (green-blue-yellow or yellow-green-blue) -/
def color_group_arrangements : ℕ := 2

/-- The probability of drawing the chips in the specified order -/
def draw_probability : ℚ :=
  (Nat.factorial green_chips * Nat.factorial blue_chips * Nat.factorial yellow_chips * color_group_arrangements : ℚ) /
  Nat.factorial total_chips

theorem draw_probability_value : draw_probability = 1 / 13860 := by
  sorry

end NUMINAMATH_CALUDE_draw_probability_value_l1359_135979


namespace NUMINAMATH_CALUDE_anna_remaining_money_l1359_135935

-- Define the given values
def initial_money : ℝ := 50
def gum_price : ℝ := 1.50
def gum_quantity : ℕ := 4
def chocolate_price : ℝ := 2.25
def chocolate_quantity : ℕ := 7
def candy_cane_price : ℝ := 0.75
def candy_cane_quantity : ℕ := 3
def jelly_beans_original_price : ℝ := 3.00
def jelly_beans_discount : ℝ := 0.20
def sales_tax_rate : ℝ := 0.075

-- Calculate the total cost and remaining money
def calculate_remaining_money : ℝ :=
  let gum_cost := gum_price * gum_quantity
  let chocolate_cost := chocolate_price * chocolate_quantity
  let candy_cane_cost := candy_cane_price * candy_cane_quantity
  let jelly_beans_cost := jelly_beans_original_price * (1 - jelly_beans_discount)
  let total_before_tax := gum_cost + chocolate_cost + candy_cane_cost + jelly_beans_cost
  let total_after_tax := total_before_tax * (1 + sales_tax_rate)
  initial_money - total_after_tax

-- Theorem to prove
theorem anna_remaining_money :
  calculate_remaining_money = 21.62 := by sorry

end NUMINAMATH_CALUDE_anna_remaining_money_l1359_135935


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l1359_135905

-- Define the polynomial Q(x)
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 8

-- Theorem statement
theorem factor_implies_d_value :
  ∀ d : ℝ, (∀ x : ℝ, (x - 3) ∣ Q d x) → d = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l1359_135905


namespace NUMINAMATH_CALUDE_absolute_difference_31st_terms_l1359_135996

/-- Defines an arithmetic sequence with a given first term and common difference -/
def arithmeticSequence (a₁ : ℤ) (d : ℤ) : ℕ → ℤ :=
  λ n => a₁ + d * (n - 1)

/-- The 31st term of sequence C -/
def C : ℤ := arithmeticSequence (-20) 12 31

/-- The 31st term of sequence D -/
def D : ℤ := arithmeticSequence 50 (-8) 31

/-- Theorem stating the absolute difference between the 31st terms of C and D -/
theorem absolute_difference_31st_terms : |C - D| = 492 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_31st_terms_l1359_135996


namespace NUMINAMATH_CALUDE_f_geq_g_for_all_real_l1359_135984

theorem f_geq_g_for_all_real : ∀ x : ℝ, x^2 * Real.exp x ≥ 2 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_f_geq_g_for_all_real_l1359_135984


namespace NUMINAMATH_CALUDE_five_people_arrangement_with_restriction_l1359_135930

/-- The number of ways to arrange n people in a line -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line where one specific person cannot be first or last -/
def arrangementsWithRestriction (n : ℕ) : ℕ :=
  (n - 2) * Nat.factorial (n - 1)

/-- Theorem: There are 72 ways to arrange 5 people in a line where one specific person cannot be first or last -/
theorem five_people_arrangement_with_restriction :
  arrangementsWithRestriction 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_five_people_arrangement_with_restriction_l1359_135930


namespace NUMINAMATH_CALUDE_pizza_distribution_l1359_135910

/-- Given 6 people sharing 3 pizzas with 8 slices each, if they all eat the same amount and finish all the pizzas, each person will eat 4 slices. -/
theorem pizza_distribution (num_people : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_people = 6)
  (h2 : num_pizzas = 3)
  (h3 : slices_per_pizza = 8) :
  (num_pizzas * slices_per_pizza) / num_people = 4 :=
by
  sorry

#check pizza_distribution

end NUMINAMATH_CALUDE_pizza_distribution_l1359_135910


namespace NUMINAMATH_CALUDE_power_product_evaluation_l1359_135966

theorem power_product_evaluation (a : ℕ) (h : a = 2) : a^3 * a^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_product_evaluation_l1359_135966


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1359_135936

theorem sum_of_reciprocals (x y : ℝ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = 2) :
  x + y = 4/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1359_135936
