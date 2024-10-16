import Mathlib

namespace NUMINAMATH_CALUDE_third_batch_average_l1072_107297

theorem third_batch_average (n₁ n₂ n₃ : ℕ) (a₁ a₂ a_total : ℚ) :
  n₁ = 40 →
  n₂ = 50 →
  n₃ = 60 →
  a₁ = 45 →
  a₂ = 55 →
  a_total = 56333333333333336 / 1000000000000000 →
  (n₁ * a₁ + n₂ * a₂ + n₃ * (3900 / 60)) / (n₁ + n₂ + n₃) = a_total :=
by sorry

end NUMINAMATH_CALUDE_third_batch_average_l1072_107297


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l1072_107220

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l1072_107220


namespace NUMINAMATH_CALUDE_bet_is_unfair_a_has_advantage_l1072_107242

/-- Represents the outcome of rolling two dice -/
def DiceRoll := Fin 6 × Fin 6

/-- The probability of A winning (sum < 8) -/
def probAWins : ℚ := 7/12

/-- The probability of B winning (sum ≥ 8) -/
def probBWins : ℚ := 5/12

/-- A's bet amount in forints -/
def aBet : ℚ := 10

/-- B's bet amount in forints -/
def bBet : ℚ := 8

/-- Expected gain for A in forints -/
def expectedGainA : ℚ := bBet * probAWins - aBet * probBWins

theorem bet_is_unfair : expectedGainA = 1/2 := by sorry

theorem a_has_advantage : expectedGainA > 0 := by sorry

end NUMINAMATH_CALUDE_bet_is_unfair_a_has_advantage_l1072_107242


namespace NUMINAMATH_CALUDE_books_from_second_shop_is_35_l1072_107215

/-- The number of books Rahim bought from the second shop -/
def books_from_second_shop : ℕ := sorry

/-- The total amount spent on books -/
def total_spent : ℕ := 6500 + 2000

/-- The total number of books bought -/
def total_books : ℕ := 65 + books_from_second_shop

/-- The average price per book -/
def average_price : ℚ := 85

theorem books_from_second_shop_is_35 :
  books_from_second_shop = 35 ∧
  65 * 100 = 6500 ∧
  books_from_second_shop * average_price = 2000 ∧
  average_price * total_books = total_spent := by sorry

end NUMINAMATH_CALUDE_books_from_second_shop_is_35_l1072_107215


namespace NUMINAMATH_CALUDE_intersection_point_l1072_107259

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The equation of the first line: 2x + y + 2 = 0 -/
def line1 (x y : ℝ) : Prop := 2 * x + y + 2 = 0

/-- The equation of the second line: ax + 4y - 2 = 0 -/
def line2 (a x y : ℝ) : Prop := a * x + 4 * y - 2 = 0

/-- The theorem stating that the intersection point of the two perpendicular lines is (-1, 0) -/
theorem intersection_point :
  ∃ (a : ℝ),
    (∀ x y : ℝ, perpendicular (-2) (-a/4)) →
    (∀ x y : ℝ, line1 x y ∧ line2 a x y → x = -1 ∧ y = 0) :=
by sorry


end NUMINAMATH_CALUDE_intersection_point_l1072_107259


namespace NUMINAMATH_CALUDE_complex_subtraction_l1072_107237

theorem complex_subtraction (i : ℂ) (h : i * i = -1) :
  let z₁ : ℂ := 3 + 4 * i
  let z₂ : ℂ := 1 + 2 * i
  z₁ - z₂ = 2 + 2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1072_107237


namespace NUMINAMATH_CALUDE_random_walk_properties_l1072_107276

/-- Represents a random walk on a line -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of the random walk -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences that achieve the maximum range -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Theorem stating the properties of the random walk -/
theorem random_walk_properties (w : RandomWalk) : 
  (max_range w = w.a) ∧ 
  (min_range w = w.a - w.b) ∧ 
  (max_range_sequences w = w.b + 1) := by
  sorry

end NUMINAMATH_CALUDE_random_walk_properties_l1072_107276


namespace NUMINAMATH_CALUDE_dogs_and_movies_percentage_l1072_107256

theorem dogs_and_movies_percentage
  (total_students : ℕ)
  (dogs_and_games_percentage : ℚ)
  (dogs_preference : ℕ)
  (h1 : total_students = 30)
  (h2 : dogs_and_games_percentage = 1/2)
  (h3 : dogs_preference = 18) :
  (dogs_preference - (dogs_and_games_percentage * total_students)) / total_students = 1/10 :=
sorry

end NUMINAMATH_CALUDE_dogs_and_movies_percentage_l1072_107256


namespace NUMINAMATH_CALUDE_find_A_value_l1072_107233

theorem find_A_value (A B : Nat) (h1 : A < 10) (h2 : B < 10) 
  (h3 : 500 + 10 * A + 8 - (100 * B + 14) = 364) : A = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_A_value_l1072_107233


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1072_107291

theorem hyperbola_eccentricity (a b c : ℝ) (h1 : b^2 / a^2 = 1) (h2 : c^2 = a^2 + b^2) :
  c / a = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1072_107291


namespace NUMINAMATH_CALUDE_parallelogram_area_l1072_107210

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area :
  let base : ℝ := 20
  let height : ℝ := 4
  let area : ℝ := base * height
  area = 80 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1072_107210


namespace NUMINAMATH_CALUDE_six_balls_two_boxes_l1072_107287

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2^(n-1) - 2

/-- The problem statement -/
theorem six_balls_two_boxes : distribute_balls 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_two_boxes_l1072_107287


namespace NUMINAMATH_CALUDE_keith_cards_l1072_107254

theorem keith_cards (x : ℕ) : 
  (x + 8) / 2 = 46 → x = 84 := by
  sorry

end NUMINAMATH_CALUDE_keith_cards_l1072_107254


namespace NUMINAMATH_CALUDE_double_money_l1072_107245

/-- Given an initial amount of money and an increase equal to that amount,
    prove that the final amount is twice the initial amount. -/
theorem double_money (initial : ℕ) : initial + initial = 2 * initial := by
  sorry

end NUMINAMATH_CALUDE_double_money_l1072_107245


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l1072_107296

/-- The area of a rhombus formed by the intersection of two equilateral triangles in a square -/
theorem rhombus_area_in_square (s : ℝ) (h : s = 4) : 
  let triangle_height := s * Real.sqrt 3 / 2
  let rhombus_diagonal1 := 2 * triangle_height - s
  let rhombus_diagonal2 := s
  rhombus_diagonal1 * rhombus_diagonal2 / 2 = 8 * Real.sqrt 3 - 8 := by
  sorry

#check rhombus_area_in_square

end NUMINAMATH_CALUDE_rhombus_area_in_square_l1072_107296


namespace NUMINAMATH_CALUDE_divisibility_by_30_l1072_107225

theorem divisibility_by_30 (a m n : ℕ) (k : ℤ) 
  (h1 : m > n) (h2 : n ≥ 2) (h3 : m - n = 4 * k.natAbs) : 
  ∃ (q : ℤ), a^m - a^n = 30 * q :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_30_l1072_107225


namespace NUMINAMATH_CALUDE_inequality_solution_l1072_107255

theorem inequality_solution :
  {x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4} =
  {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3)} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1072_107255


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1072_107209

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (j : ℕ), 5 * n = j^3) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (k : ℕ), 4 * m = k^2) → 
    (∃ (j : ℕ), 5 * m = j^3) → 
    m ≥ n) ∧
  n = 125 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1072_107209


namespace NUMINAMATH_CALUDE_fraction_simplification_l1072_107298

theorem fraction_simplification (x : ℝ) (h : x ≠ 3) :
  (3 * x) / (x - 3) + (x + 6) / (3 - x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1072_107298


namespace NUMINAMATH_CALUDE_complement_of_A_l1072_107229

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ 0}

theorem complement_of_A (x : ℝ) : x ∈ Aᶜ ↔ 0 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1072_107229


namespace NUMINAMATH_CALUDE_gold_asymmetric_probability_l1072_107285

/-- Represents a coin --/
inductive Coin
| Gold
| Silver

/-- Represents the symmetry of a coin --/
inductive Symmetry
| Symmetric
| Asymmetric

/-- The probability of getting heads for the asymmetric coin --/
def asymmetricHeadsProbability : ℝ := 0.6

/-- The result of a coin flip --/
inductive FlipResult
| Heads
| Tails

/-- Represents the sequence of coin flips --/
structure FlipSequence where
  goldResult : FlipResult
  silverResult1 : FlipResult
  silverResult2 : FlipResult

/-- The observed flip sequence --/
def observedFlips : FlipSequence := {
  goldResult := FlipResult.Heads,
  silverResult1 := FlipResult.Tails,
  silverResult2 := FlipResult.Heads
}

/-- The probability that the gold coin is asymmetric given the observed flip sequence --/
def probGoldAsymmetric (flips : FlipSequence) : ℝ := sorry

theorem gold_asymmetric_probability :
  probGoldAsymmetric observedFlips = 6/10 := by sorry

end NUMINAMATH_CALUDE_gold_asymmetric_probability_l1072_107285


namespace NUMINAMATH_CALUDE_cards_per_pack_l1072_107290

theorem cards_per_pack (num_packs : ℕ) (num_pages : ℕ) (cards_per_page : ℕ) : 
  num_packs = 60 → num_pages = 42 → cards_per_page = 10 →
  (num_pages * cards_per_page) / num_packs = 7 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_pack_l1072_107290


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l1072_107257

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from total_chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def favorable_outcomes : ℕ := n.choose k

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := favorable_outcomes / total_selections

theorem convex_quadrilateral_probability : probability = 2 / 585 := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l1072_107257


namespace NUMINAMATH_CALUDE_park_circle_diameter_l1072_107228

/-- Represents the circular arrangement in the park -/
structure ParkCircle where
  fountain_diameter : ℝ
  garden_width : ℝ
  inner_path_width : ℝ
  outer_path_width : ℝ

/-- Calculates the diameter of the outermost boundary of the park circle -/
def outer_diameter (park : ParkCircle) : ℝ :=
  park.fountain_diameter + 2 * (park.garden_width + park.inner_path_width + park.outer_path_width)

/-- Theorem stating that for the given dimensions, the outer diameter is 50 feet -/
theorem park_circle_diameter :
  let park : ParkCircle := {
    fountain_diameter := 12,
    garden_width := 9,
    inner_path_width := 3,
    outer_path_width := 7
  }
  outer_diameter park = 50 := by
  sorry

end NUMINAMATH_CALUDE_park_circle_diameter_l1072_107228


namespace NUMINAMATH_CALUDE_paul_strawberries_l1072_107239

/-- The number of strawberries Paul has after picking more -/
def total_strawberries (initial : ℕ) (picked : ℕ) : ℕ := initial + picked

/-- Theorem: Paul has 63 strawberries after picking more -/
theorem paul_strawberries : total_strawberries 28 35 = 63 := by
  sorry

end NUMINAMATH_CALUDE_paul_strawberries_l1072_107239


namespace NUMINAMATH_CALUDE_find_A_l1072_107234

theorem find_A : ∀ A B : ℕ,
  (A ≥ 1 ∧ A ≤ 9) →
  (B ≥ 0 ∧ B ≤ 9) →
  (10 * A + 3 ≥ 10 ∧ 10 * A + 3 ≤ 99) →
  (610 + B ≥ 100 ∧ 610 + B ≤ 999) →
  (10 * A + 3) + (610 + B) = 695 →
  A = 8 := by
sorry

end NUMINAMATH_CALUDE_find_A_l1072_107234


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_13_greatest_two_digit_multiple_of_13_is_91_l1072_107208

theorem greatest_two_digit_multiple_of_13 : ℕ → Prop :=
  fun n =>
    (n ≤ 99) ∧ 
    (n ≥ 10) ∧ 
    (∃ k : ℕ, n = 13 * k) ∧ 
    (∀ m : ℕ, m ≤ 99 ∧ m ≥ 10 ∧ (∃ j : ℕ, m = 13 * j) → m ≤ n) →
    n = 91

-- The proof would go here, but we'll skip it as requested
theorem greatest_two_digit_multiple_of_13_is_91 : greatest_two_digit_multiple_of_13 91 := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_13_greatest_two_digit_multiple_of_13_is_91_l1072_107208


namespace NUMINAMATH_CALUDE_ac_less_than_bc_l1072_107270

theorem ac_less_than_bc (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * c < b * c := by
  sorry

end NUMINAMATH_CALUDE_ac_less_than_bc_l1072_107270


namespace NUMINAMATH_CALUDE_equation_solution_l1072_107221

theorem equation_solution : ∃ x : ℝ, 3 * x - 6 = |(-20 + 5)| ∧ x = 7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1072_107221


namespace NUMINAMATH_CALUDE_linear_system_solution_l1072_107283

theorem linear_system_solution (x y z : ℚ) : 
  x + 2 * y = 12 ∧ 
  y + 3 * z = 15 ∧ 
  3 * x - z = 6 → 
  x = 54 / 17 ∧ y = 75 / 17 ∧ z = 60 / 17 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1072_107283


namespace NUMINAMATH_CALUDE_tommys_balloons_l1072_107248

theorem tommys_balloons (initial_balloons : ℝ) (mom_gave : ℝ) (total_balloons : ℝ)
  (h1 : mom_gave = 78.5)
  (h2 : total_balloons = 132.25)
  (h3 : total_balloons = initial_balloons + mom_gave) :
  initial_balloons = 53.75 := by
  sorry

end NUMINAMATH_CALUDE_tommys_balloons_l1072_107248


namespace NUMINAMATH_CALUDE_number_difference_l1072_107232

theorem number_difference (S L : ℕ) (h1 : S = 476) (h2 : L = 6 * S + 15) :
  L - S = 2395 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1072_107232


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1072_107213

theorem fraction_equals_zero (x : ℝ) : x / (x^2 - 1) = 0 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1072_107213


namespace NUMINAMATH_CALUDE_ab_value_l1072_107268

theorem ab_value (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) 
  (h3 : a + c = 2*b) : 
  a * b = 10 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l1072_107268


namespace NUMINAMATH_CALUDE_parabola_translation_correct_l1072_107216

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 2x² -/
def original_parabola : Parabola := { a := 2, b := 0, c := 0 }

/-- Translates a parabola horizontally by h units and vertically by k units -/
def translate (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + k }

/-- The resulting parabola after translation -/
def translated_parabola : Parabola :=
  translate (translate original_parabola 3 0) 0 4

theorem parabola_translation_correct :
  translated_parabola.a = 2 ∧
  translated_parabola.b = -12 ∧
  translated_parabola.c = 22 :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_correct_l1072_107216


namespace NUMINAMATH_CALUDE_poster_height_l1072_107243

/-- Given a rectangular poster with width 4 inches and area 28 square inches, its height is 7 inches. -/
theorem poster_height (width : ℝ) (area : ℝ) (height : ℝ) 
    (h_width : width = 4)
    (h_area : area = 28)
    (h_rect_area : area = width * height) : height = 7 := by
  sorry

end NUMINAMATH_CALUDE_poster_height_l1072_107243


namespace NUMINAMATH_CALUDE_log_sum_power_twenty_l1072_107253

theorem log_sum_power_twenty (log_2 log_5 : ℝ) (h : log_2 + log_5 = 1) :
  (log_2 + log_5)^20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_power_twenty_l1072_107253


namespace NUMINAMATH_CALUDE_rockham_soccer_league_members_l1072_107262

theorem rockham_soccer_league_members : 
  let sock_cost : ℕ := 6
  let tshirt_cost : ℕ := sock_cost + 7
  let member_cost : ℕ := 2 * (sock_cost + tshirt_cost)
  let custom_fee : ℕ := 200
  let total_cost : ℕ := 2892
  ∃ (n : ℕ), n * member_cost + custom_fee = total_cost ∧ n = 70 :=
by sorry

end NUMINAMATH_CALUDE_rockham_soccer_league_members_l1072_107262


namespace NUMINAMATH_CALUDE_committee_selection_with_president_l1072_107218

/-- The number of ways to choose a committee with a required member -/
def choose_committee_with_required (n m k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: Choosing a 5-person committee from 12 people with at least one being the president -/
theorem committee_selection_with_president :
  choose_committee_with_required 12 1 5 = 330 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_with_president_l1072_107218


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_8_l1072_107273

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (1 : ℕ)^k * (1 : ℕ)^(8 - k)) = 256 ∧
  Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_8_l1072_107273


namespace NUMINAMATH_CALUDE_six_hour_rental_cost_l1072_107265

/-- Represents the cost structure for kayak and paddle rental --/
structure RentalCost where
  paddleFee : ℕ
  kayakHourlyRate : ℕ

/-- Calculates the total cost for a given number of hours --/
def totalCost (rc : RentalCost) (hours : ℕ) : ℕ :=
  rc.paddleFee + rc.kayakHourlyRate * hours

theorem six_hour_rental_cost 
  (rc : RentalCost)
  (three_hour_cost : totalCost rc 3 = 30)
  (kayak_rate : rc.kayakHourlyRate = 5) :
  totalCost rc 6 = 45 := by
  sorry

#check six_hour_rental_cost

end NUMINAMATH_CALUDE_six_hour_rental_cost_l1072_107265


namespace NUMINAMATH_CALUDE_least_distinct_values_is_184_l1072_107261

/-- Represents a list of positive integers with a unique mode -/
structure IntegerList where
  elements : List Nat
  size : elements.length = 2023
  mode_frequency : Nat
  mode_unique : mode_frequency = 12
  is_mode : elements.count mode_frequency = mode_frequency
  other_frequencies : ∀ n, n ≠ mode_frequency → elements.count n < mode_frequency

/-- The least number of distinct values in the list -/
def leastDistinctValues (list : IntegerList) : Nat :=
  list.elements.toFinset.card

/-- Theorem: The least number of distinct values in the list is 184 -/
theorem least_distinct_values_is_184 (list : IntegerList) :
  leastDistinctValues list = 184 := by
  sorry


end NUMINAMATH_CALUDE_least_distinct_values_is_184_l1072_107261


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1072_107219

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 2 = 4 →                     -- given condition
  ∃ r : ℝ,                      -- existence of common ratio for geometric sequence
    (1 + a 3) * r = a 6 ∧       -- geometric sequence conditions
    a 6 * r = 4 + a 10 →
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1072_107219


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l1072_107207

/-- A perfect square trinomial in the form ax^2 + bx + c -/
structure PerfectSquareTrinomial (a b c : ℝ) : Prop where
  is_perfect_square : ∃ (p q : ℝ), a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, PerfectSquareTrinomial 1 (-m) 25 → m = 10 ∨ m = -10 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l1072_107207


namespace NUMINAMATH_CALUDE_pens_bought_proof_l1072_107227

/-- The cost of one pen in rubles -/
def pen_cost : ℕ := 21

/-- The amount Masha spent on pens in rubles -/
def masha_spent : ℕ := 357

/-- The amount Olya spent on pens in rubles -/
def olya_spent : ℕ := 441

/-- The total number of pens bought by Masha and Olya -/
def total_pens : ℕ := 38

theorem pens_bought_proof :
  pen_cost > 10 ∧
  masha_spent % pen_cost = 0 ∧
  olya_spent % pen_cost = 0 ∧
  masha_spent / pen_cost + olya_spent / pen_cost = total_pens :=
by sorry

end NUMINAMATH_CALUDE_pens_bought_proof_l1072_107227


namespace NUMINAMATH_CALUDE_carrie_fourth_day_miles_l1072_107241

/-- Represents Carrie's four-day trip --/
structure CarrieTrip where
  day1_miles : ℕ
  day2_miles : ℕ
  day3_miles : ℕ
  day4_miles : ℕ
  charge_interval : ℕ
  total_charges : ℕ

/-- Theorem: Given the conditions of Carrie's trip, she drove 189 miles on the fourth day --/
theorem carrie_fourth_day_miles (trip : CarrieTrip)
  (h1 : trip.day1_miles = 135)
  (h2 : trip.day2_miles = trip.day1_miles + 124)
  (h3 : trip.day3_miles = 159)
  (h4 : trip.charge_interval = 106)
  (h5 : trip.total_charges = 7)
  : trip.day4_miles = 189 := by
  sorry

#check carrie_fourth_day_miles

end NUMINAMATH_CALUDE_carrie_fourth_day_miles_l1072_107241


namespace NUMINAMATH_CALUDE_euclidean_algorithm_bound_l1072_107203

/-- The number of divisions performed by the Euclidean algorithm -/
def euclidean_divisions (a b : ℕ) : ℕ := sorry

/-- The number of digits of a natural number in decimal -/
def num_digits (n : ℕ) : ℕ := sorry

theorem euclidean_algorithm_bound (a b : ℕ) (h1 : a > b) (h2 : b > 0) :
  euclidean_divisions a b ≤ 5 * (num_digits b) := by sorry

end NUMINAMATH_CALUDE_euclidean_algorithm_bound_l1072_107203


namespace NUMINAMATH_CALUDE_max_z3_value_max_z3_value_tight_l1072_107267

theorem max_z3_value (z₁ z₂ z₃ : ℂ) 
  (h₁ : Complex.abs z₁ ≤ 1)
  (h₂ : Complex.abs z₂ ≤ 2)
  (h₃ : Complex.abs (2 * z₃ - z₁ - z₂) ≤ Complex.abs (z₁ - z₂)) :
  Complex.abs z₃ ≤ Real.sqrt 5 :=
by
  sorry

theorem max_z3_value_tight : ∃ (z₁ z₂ z₃ : ℂ),
  Complex.abs z₁ ≤ 1 ∧
  Complex.abs z₂ ≤ 2 ∧
  Complex.abs (2 * z₃ - z₁ - z₂) ≤ Complex.abs (z₁ - z₂) ∧
  Complex.abs z₃ = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_z3_value_max_z3_value_tight_l1072_107267


namespace NUMINAMATH_CALUDE_ball_drawing_properties_l1072_107286

/-- Probability of drawing a red ball on the nth draw -/
def P (n : ℕ) : ℚ :=
  1/2 + 1/(2^(2*n + 1))

/-- Sum of the first n terms of the sequence P_n -/
def S (n : ℕ) : ℚ :=
  (1/6) * (3*n + 1 - (1/4)^n)

theorem ball_drawing_properties :
  (P 2 = 17/32) ∧
  (∀ n : ℕ, 4 * P (n+2) + P n = 5 * P (n+1)) ∧
  (∀ n : ℕ, S n = (1/6) * (3*n + 1 - (1/4)^n)) :=
by sorry

end NUMINAMATH_CALUDE_ball_drawing_properties_l1072_107286


namespace NUMINAMATH_CALUDE_hat_promotion_savings_l1072_107266

/-- Calculates the percentage saved when buying three hats under a promotional offer --/
theorem hat_promotion_savings : 
  let regular_price : ℝ := 60
  let discount_second : ℝ := 0.25
  let discount_third : ℝ := 0.35
  let total_regular : ℝ := 3 * regular_price
  let price_first : ℝ := regular_price
  let price_second : ℝ := regular_price * (1 - discount_second)
  let price_third : ℝ := regular_price * (1 - discount_third)
  let total_discounted : ℝ := price_first + price_second + price_third
  let savings : ℝ := total_regular - total_discounted
  let percentage_saved : ℝ := (savings / total_regular) * 100
  percentage_saved = 20 := by
  sorry


end NUMINAMATH_CALUDE_hat_promotion_savings_l1072_107266


namespace NUMINAMATH_CALUDE_min_distance_to_hyperbola_l1072_107258

/-- The minimum distance between A(4,4) and P(x, 1/x) where x > 0 is √14 -/
theorem min_distance_to_hyperbola :
  let A : ℝ × ℝ := (4, 4)
  let P : ℝ → ℝ × ℝ := fun x ↦ (x, 1/x)
  let distance (x : ℝ) : ℝ := Real.sqrt ((P x).1 - A.1)^2 + ((P x).2 - A.2)^2
  ∀ x > 0, distance x ≥ Real.sqrt 14 ∧ ∃ x₀ > 0, distance x₀ = Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_hyperbola_l1072_107258


namespace NUMINAMATH_CALUDE_sally_boxes_proof_l1072_107235

/-- The number of boxes Sally sold on Saturday -/
def saturday_boxes : ℕ := 60

/-- The number of boxes Sally sold on Sunday -/
def sunday_boxes : ℕ := (3 * saturday_boxes) / 2

/-- The total number of boxes Sally sold over two days -/
def total_boxes : ℕ := 150

theorem sally_boxes_proof :
  saturday_boxes + sunday_boxes = total_boxes ∧
  sunday_boxes = (3 * saturday_boxes) / 2 :=
sorry

end NUMINAMATH_CALUDE_sally_boxes_proof_l1072_107235


namespace NUMINAMATH_CALUDE_total_results_l1072_107212

theorem total_results (average : ℝ) (first_five_avg : ℝ) (last_seven_avg : ℝ) (fifth_result : ℝ)
  (h1 : average = 42)
  (h2 : first_five_avg = 49)
  (h3 : last_seven_avg = 52)
  (h4 : fifth_result = 147) :
  ∃ n : ℕ, n = 11 ∧ n * average = 5 * first_five_avg + 7 * last_seven_avg - fifth_result := by
  sorry

end NUMINAMATH_CALUDE_total_results_l1072_107212


namespace NUMINAMATH_CALUDE_circle_equation_through_ABC_circle_equation_center_y_2_l1072_107217

-- Define the circle P
def CircleP : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (c : ℝ × ℝ) (r : ℝ), (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (6, -2)

-- Theorem 1
theorem circle_equation_through_ABC :
  A ∈ CircleP ∧ B ∈ CircleP ∧ C ∈ CircleP →
  ∃ (D E F : ℝ), ∀ (x y : ℝ), (x, y) ∈ CircleP ↔ x^2 + y^2 + D*x + E*y + F = 0 :=
sorry

-- Theorem 2
theorem circle_equation_center_y_2 :
  A ∈ CircleP ∧ B ∈ CircleP ∧ (∃ (c : ℝ × ℝ), c ∈ CircleP ∧ c.2 = 2) →
  ∃ (c : ℝ × ℝ) (r : ℝ), c = (5/2, 2) ∧ r = 5/2 ∧
    ∀ (x y : ℝ), (x, y) ∈ CircleP ↔ (x - c.1)^2 + (y - c.2)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_through_ABC_circle_equation_center_y_2_l1072_107217


namespace NUMINAMATH_CALUDE_shooting_training_results_l1072_107299

/-- Represents the shooting scores and their frequencies -/
structure ShootingData :=
  (scores : List Nat)
  (frequencies : List Nat)
  (excellent_threshold : Nat)
  (total_freshmen : Nat)

/-- Calculates the mode of the shooting data -/
def mode (data : ShootingData) : Nat :=
  sorry

/-- Calculates the average score of the shooting data -/
def average_score (data : ShootingData) : Rat :=
  sorry

/-- Estimates the number of excellent shooters -/
def estimate_excellent_shooters (data : ShootingData) : Nat :=
  sorry

/-- The main theorem proving the results of the shooting training -/
theorem shooting_training_results (data : ShootingData) 
  (h1 : data.scores = [6, 7, 8, 9])
  (h2 : data.frequencies = [1, 6, 3, 2])
  (h3 : data.excellent_threshold = 8)
  (h4 : data.total_freshmen = 1500) :
  mode data = 7 ∧ 
  average_score data = 15/2 ∧ 
  estimate_excellent_shooters data = 625 :=
sorry

end NUMINAMATH_CALUDE_shooting_training_results_l1072_107299


namespace NUMINAMATH_CALUDE_lisa_caffeine_consumption_l1072_107272

/-- Represents the number of each beverage Lisa consumed --/
structure BeverageConsumption where
  coffee : ℕ
  soda : ℕ
  tea : ℕ
  energyDrink : ℕ

/-- Represents the caffeine content of each beverage in milligrams --/
structure CaffeineContent where
  coffee : ℕ
  soda : ℕ
  tea : ℕ
  energyDrink : ℕ

def totalCaffeine (consumption : BeverageConsumption) (content : CaffeineContent) : ℕ :=
  consumption.coffee * content.coffee +
  consumption.soda * content.soda +
  consumption.tea * content.tea +
  consumption.energyDrink * content.energyDrink

theorem lisa_caffeine_consumption
  (consumption : BeverageConsumption)
  (content : CaffeineContent)
  (daily_goal : ℕ)
  (h_consumption : consumption = { coffee := 3, soda := 1, tea := 2, energyDrink := 1 })
  (h_content : content = { coffee := 95, soda := 45, tea := 55, energyDrink := 120 })
  (h_goal : daily_goal = 200) :
  totalCaffeine consumption content = 560 ∧ totalCaffeine consumption content - daily_goal = 360 := by
  sorry


end NUMINAMATH_CALUDE_lisa_caffeine_consumption_l1072_107272


namespace NUMINAMATH_CALUDE_is_circle_center_l1072_107205

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 12*y + 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -6)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center : 
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l1072_107205


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l1072_107277

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation 1673000000 = ScientificNotation.mk 1.673 9 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l1072_107277


namespace NUMINAMATH_CALUDE_miquel_point_existence_l1072_107275

-- Define the basic geometric objects
variable (A B C D H M N S T : Point)

-- Define the quadrilateral ABCD
def is_convex_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define that ABCD is not a kite
def is_not_kite (A B C D : Point) : Prop := sorry

-- Define perpendicular diagonals
def perpendicular_diagonals (A B C D H : Point) : Prop := sorry

-- Define midpoints
def is_midpoint (M : Point) (B C : Point) : Prop := sorry

-- Define ray intersection
def ray_intersects (M H S A D : Point) : Prop := sorry

-- Define point outside quadrilateral
def point_outside_quadrilateral (E A B C D : Point) : Prop := sorry

-- Define angle bisector
def is_angle_bisector (E H B S : Point) : Prop := sorry

-- Define equal angles
def equal_angles (B E N M D : Point) : Prop := sorry

-- Main theorem
theorem miquel_point_existence 
  (h1 : is_convex_cyclic_quadrilateral A B C D)
  (h2 : is_not_kite A B C D)
  (h3 : perpendicular_diagonals A B C D H)
  (h4 : is_midpoint M B C)
  (h5 : is_midpoint N C D)
  (h6 : ray_intersects M H S A D)
  (h7 : ray_intersects N H T A B) :
  ∃ E : Point,
    point_outside_quadrilateral E A B C D ∧
    is_angle_bisector E H B S ∧
    is_angle_bisector E H T D ∧
    equal_angles B E N M D :=
sorry

end NUMINAMATH_CALUDE_miquel_point_existence_l1072_107275


namespace NUMINAMATH_CALUDE_jogger_distance_ahead_l1072_107263

/-- Calculates the distance a jogger is ahead of a train given their speeds, the train's length, and the time it takes for the train to pass the jogger. -/
theorem jogger_distance_ahead (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 210 →
  passing_time = 45 →
  (train_speed - jogger_speed) * passing_time - train_length = 240 :=
by sorry

end NUMINAMATH_CALUDE_jogger_distance_ahead_l1072_107263


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1072_107202

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_abc_properties (t : Triangle) 
  (ha : t.a = 3)
  (hb : t.b = 2)
  (hcosA : Real.cos t.A = 1/3) :
  Real.sin t.B = 4 * Real.sqrt 2 / 9 ∧ t.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l1072_107202


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1072_107200

theorem polynomial_divisibility (a b c : ℕ) :
  ∃ q : Polynomial ℚ, X^(3*a) + X^(3*b+1) + X^(3*c+2) = (X^2 + X + 1) * q :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1072_107200


namespace NUMINAMATH_CALUDE_complex_points_on_circle_l1072_107260

theorem complex_points_on_circle 
  (a₁ a₂ a₃ a₄ a₅ : ℂ) 
  (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0)
  (h_ratio : a₂ / a₁ = a₃ / a₂ ∧ a₃ / a₂ = a₄ / a₃ ∧ a₄ / a₃ = a₅ / a₄)
  (S : ℝ)
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅))
  (h_S_real : a₁ + a₂ + a₃ + a₄ + a₅ = S)
  (h_S_bound : abs S ≤ 2) :
  ∃ (r : ℝ), r > 0 ∧ Complex.abs a₁ = r ∧ Complex.abs a₂ = r ∧ 
             Complex.abs a₃ = r ∧ Complex.abs a₄ = r ∧ Complex.abs a₅ = r := by
  sorry

end NUMINAMATH_CALUDE_complex_points_on_circle_l1072_107260


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1072_107214

theorem partial_fraction_decomposition (M₁ M₂ : ℚ) :
  (∀ x : ℚ, x ≠ 2 → x ≠ 3 → (45 * x - 82) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = -424 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1072_107214


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1072_107204

theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n : ℕ, a n > 0) →  -- Positive sequence
  (∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n) →  -- Geometric sequence
  a 3 = 3 →  -- Given condition
  a 5 = 8 * a 7 →  -- Given condition
  a 10 = 3 * Real.sqrt 2 / 128 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1072_107204


namespace NUMINAMATH_CALUDE_nougat_caramel_ratio_l1072_107282

def chocolate_problem (total caramels truffles peanut_clusters nougats : ℕ) : Prop :=
  total = 50 ∧
  caramels = 3 ∧
  truffles = caramels + 6 ∧
  peanut_clusters = (64 * total) / 100 ∧
  nougats = total - caramels - truffles - peanut_clusters ∧
  nougats = 2 * caramels

theorem nougat_caramel_ratio :
  ∀ total caramels truffles peanut_clusters nougats : ℕ,
  chocolate_problem total caramels truffles peanut_clusters nougats →
  nougats = 2 * caramels :=
by
  sorry

#check nougat_caramel_ratio

end NUMINAMATH_CALUDE_nougat_caramel_ratio_l1072_107282


namespace NUMINAMATH_CALUDE_inequality_solution_l1072_107206

theorem inequality_solution (m : ℝ) (hm : 0 < m ∧ m < 1) :
  {x : ℝ | m * x / (x - 3) > 1} = {x : ℝ | 3 < x ∧ x < 3 / (1 - m)} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1072_107206


namespace NUMINAMATH_CALUDE_isabellas_total_items_l1072_107293

/-- Given that Alexis bought 3 times more pants and dresses than Isabella,
    21 pairs of pants, and 18 dresses, prove that Isabella bought a total
    of 13 pairs of pants and dresses. -/
theorem isabellas_total_items
  (alexis_multiplier : ℕ)
  (alexis_pants : ℕ)
  (alexis_dresses : ℕ)
  (h1 : alexis_multiplier = 3)
  (h2 : alexis_pants = 21)
  (h3 : alexis_dresses = 18) :
  alexis_pants / alexis_multiplier + alexis_dresses / alexis_multiplier = 13 :=
by
  sorry

#check isabellas_total_items

end NUMINAMATH_CALUDE_isabellas_total_items_l1072_107293


namespace NUMINAMATH_CALUDE_chessboard_pawn_placement_l1072_107271

/-- The number of columns on the chessboard -/
def n : ℕ := 8

/-- The number of rows on the chessboard -/
def m : ℕ := 8

/-- The number of ways to place a pawn in a single row -/
def ways_per_row : ℕ := n + 1

/-- The total number of ways to place pawns on the chessboard -/
def total_ways : ℕ := ways_per_row ^ m

theorem chessboard_pawn_placement :
  total_ways = 3^16 :=
sorry

end NUMINAMATH_CALUDE_chessboard_pawn_placement_l1072_107271


namespace NUMINAMATH_CALUDE_dog_grouping_ways_l1072_107201

def total_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 5
def group3_size : ℕ := 3

theorem dog_grouping_ways :
  let remaining_dogs := total_dogs - 2  -- Sparky and Rex are already placed
  let remaining_group1 := group1_size - 1  -- Sparky is already in group 1
  let remaining_group2 := group2_size - 1  -- Rex is already in group 2
  (Nat.choose remaining_dogs remaining_group1) *
  (Nat.choose (remaining_dogs - remaining_group1) remaining_group2) *
  (Nat.choose (remaining_dogs - remaining_group1 - remaining_group2) group3_size) = 4200 := by
sorry

end NUMINAMATH_CALUDE_dog_grouping_ways_l1072_107201


namespace NUMINAMATH_CALUDE_min_value_theorem_l1072_107269

theorem min_value_theorem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 3 * x + y = 5 * x * y) :
  4 * x + 3 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 * x₀ + y₀ = 5 * x₀ * y₀ ∧ 4 * x₀ + 3 * y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1072_107269


namespace NUMINAMATH_CALUDE_tan_3_expression_zero_l1072_107284

theorem tan_3_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_3_expression_zero_l1072_107284


namespace NUMINAMATH_CALUDE_athena_spent_14_l1072_107240

/-- The total amount Athena spent on snacks for her friends -/
def total_spent (sandwich_price : ℚ) (sandwich_quantity : ℕ) (drink_price : ℚ) (drink_quantity : ℕ) : ℚ :=
  sandwich_price * sandwich_quantity + drink_price * drink_quantity

/-- Theorem: Athena spent $14 in total -/
theorem athena_spent_14 :
  total_spent 3 3 (5/2) 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_athena_spent_14_l1072_107240


namespace NUMINAMATH_CALUDE_fraction_simplification_l1072_107247

theorem fraction_simplification :
  (5 : ℝ) / (3 * Real.sqrt 50 + Real.sqrt 18 + 4 * Real.sqrt 8) = (5 * Real.sqrt 2) / 52 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1072_107247


namespace NUMINAMATH_CALUDE_combined_rent_C_and_D_l1072_107211

-- Define the parameters for C and D
def oxen_C : ℕ := 15
def months_C : ℕ := 3
def rent_Z : ℕ := 100

def oxen_D : ℕ := 20
def months_D : ℕ := 6
def rent_W : ℕ := 120

-- Define the function to calculate rent
def calculate_rent (months : ℕ) (monthly_rent : ℕ) : ℕ :=
  months * monthly_rent

-- Theorem statement
theorem combined_rent_C_and_D :
  calculate_rent months_C rent_Z + calculate_rent months_D rent_W = 1020 := by
  sorry


end NUMINAMATH_CALUDE_combined_rent_C_and_D_l1072_107211


namespace NUMINAMATH_CALUDE_special_sequence_properties_l1072_107222

/-- A sequence and its partial sums satisfying certain conditions -/
structure SpecialSequence where
  q : ℝ
  a : ℕ → ℝ
  S : ℕ → ℝ
  h1 : q * (q - 1) ≠ 0
  h2 : ∀ n : ℕ, (1 - q) * S n + q^n = 1
  h3 : S 3 - S 9 = S 9 - S 6

/-- The main theorem about the special sequence -/
theorem special_sequence_properties (seq : SpecialSequence) :
  (∀ n : ℕ, seq.a n = seq.q^(n - 1)) ∧
  (seq.a 2 - seq.a 8 = seq.a 8 - seq.a 5) := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_properties_l1072_107222


namespace NUMINAMATH_CALUDE_honest_person_different_answers_possible_l1072_107295

-- Define a person who always tells the truth
structure HonestPerson where
  name : String
  always_truthful : Bool

-- Define a question with its context
structure Question where
  text : String
  context : String

-- Define an answer
structure Answer where
  text : String

-- Define a function to represent a person answering a question
def answer (person : HonestPerson) (q : Question) : Answer :=
  sorry

-- Theorem: It's possible for an honest person to give different answers to the same question asked twice
theorem honest_person_different_answers_possible 
  (person : HonestPerson) 
  (q : Question) 
  (q_repeated : Question) 
  (different_context : q.context ≠ q_repeated.context) :
  ∃ (a1 a2 : Answer), 
    person.always_truthful = true ∧ 
    q.text = q_repeated.text ∧ 
    answer person q = a1 ∧ 
    answer person q_repeated = a2 ∧ 
    a1 ≠ a2 :=
  sorry

end NUMINAMATH_CALUDE_honest_person_different_answers_possible_l1072_107295


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1072_107252

theorem simplify_fraction_product : (210 : ℚ) / 18 * 6 / 150 * 9 / 4 = 21 / 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1072_107252


namespace NUMINAMATH_CALUDE_decrease_xyz_squared_l1072_107224

theorem decrease_xyz_squared (x y z : ℝ) :
  let x' := 0.6 * x
  let y' := 0.6 * y
  let z' := 0.6 * z
  x' * y' * z' ^ 2 = 0.1296 * x * y * z ^ 2 := by
sorry

end NUMINAMATH_CALUDE_decrease_xyz_squared_l1072_107224


namespace NUMINAMATH_CALUDE_pears_in_D_l1072_107251

/-- The number of baskets --/
def num_baskets : ℕ := 5

/-- The average number of fruits per basket --/
def avg_fruits_per_basket : ℕ := 25

/-- The number of apples in basket A --/
def apples_in_A : ℕ := 15

/-- The number of mangoes in basket B --/
def mangoes_in_B : ℕ := 30

/-- The number of peaches in basket C --/
def peaches_in_C : ℕ := 20

/-- The number of bananas in basket E --/
def bananas_in_E : ℕ := 35

/-- The theorem stating the number of pears in basket D --/
theorem pears_in_D : 
  (num_baskets * avg_fruits_per_basket) - (apples_in_A + mangoes_in_B + peaches_in_C + bananas_in_E) = 25 := by
  sorry

end NUMINAMATH_CALUDE_pears_in_D_l1072_107251


namespace NUMINAMATH_CALUDE_circle_tangent_slope_l1072_107238

/-- The circle with center (2,0) and radius √3 -/
def Circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 3

/-- The vector from origin to point M -/
def OM (x y : ℝ) : ℝ × ℝ := (x, y)

/-- The vector from center C to point M -/
def CM (x y : ℝ) : ℝ × ℝ := (x - 2, y)

/-- The dot product of OM and CM -/
def dotProduct (x y : ℝ) : ℝ := x * (x - 2) + y * y

theorem circle_tangent_slope (x y : ℝ) :
  Circle x y →
  dotProduct x y = 0 →
  (y / x = Real.sqrt 3 ∨ y / x = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_slope_l1072_107238


namespace NUMINAMATH_CALUDE_outdoor_dining_area_expansion_l1072_107289

/-- The total area of three sections of an outdoor dining area -/
theorem outdoor_dining_area_expansion (rectangle_area rectangle_width : ℝ)
                                      (semicircle_radius : ℝ)
                                      (triangle_base triangle_height : ℝ) :
  rectangle_area = 35 →
  rectangle_width = 7 →
  semicircle_radius = 4 →
  triangle_base = 5 →
  triangle_height = 6 →
  rectangle_area + (π * semicircle_radius ^ 2) / 2 + (triangle_base * triangle_height) / 2 = 35 + 8 * π + 15 := by
  sorry

end NUMINAMATH_CALUDE_outdoor_dining_area_expansion_l1072_107289


namespace NUMINAMATH_CALUDE_y_value_when_x_is_one_l1072_107246

-- Define the inverse square relationship between x and y
def inverse_square_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, x = k / (y * y) ∧ k ≠ 0

-- State the theorem
theorem y_value_when_x_is_one
  (h1 : inverse_square_relation 0.1111111111111111 6)
  (h2 : ∀ x y : ℝ, inverse_square_relation x y → inverse_square_relation 1 y) :
  ∃ y : ℝ, inverse_square_relation 1 y ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_y_value_when_x_is_one_l1072_107246


namespace NUMINAMATH_CALUDE_infinite_log_3_64_equals_4_l1072_107292

noncomputable def log_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem infinite_log_3_64_equals_4 :
  ∃! x : ℝ, x > 0 ∧ x = log_3 (64 + x) ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_infinite_log_3_64_equals_4_l1072_107292


namespace NUMINAMATH_CALUDE_triangle_sine_sides_l1072_107280

theorem triangle_sine_sides (a b c : Real) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h3 : a + b + c ≤ 2 * Real.pi) :
  Real.sin a + Real.sin b > Real.sin c ∧ 
  Real.sin b + Real.sin c > Real.sin a ∧ 
  Real.sin c + Real.sin a > Real.sin b := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_sides_l1072_107280


namespace NUMINAMATH_CALUDE_lemonade_ratio_l1072_107264

/-- Given that 30 lemons make 25 gallons of lemonade, prove that 12 lemons make 10 gallons -/
theorem lemonade_ratio (lemons : ℕ) (gallons : ℕ) 
  (h : (30 : ℚ) / 25 = lemons / gallons) (h10 : gallons = 10) : lemons = 12 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_ratio_l1072_107264


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1072_107278

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1072_107278


namespace NUMINAMATH_CALUDE_inequalities_proof_l1072_107279

theorem inequalities_proof (n : ℕ) (a : ℝ) (h1 : n ≥ 1) (h2 : a > 0) :
  2^(n-1) ≤ n! ∧ 
  n! ≤ n^n ∧ 
  (n+3)^2 ≤ 2^(n+3) ∧ 
  1 + n * a ≤ (1+a)^n := by
  sorry


end NUMINAMATH_CALUDE_inequalities_proof_l1072_107279


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l1072_107294

theorem quadratic_root_theorem (a b : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, x^2 + b*x + a = 0 ∧ x = -a) → b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l1072_107294


namespace NUMINAMATH_CALUDE_matrix_linear_combination_l1072_107236

theorem matrix_linear_combination : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, 1; 3, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-3, -2; 2, -4]
  2 • A + 3 • B = !![-1, -4; 12, -2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_linear_combination_l1072_107236


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1072_107231

/-- A regular polygon with side length 8 units and exterior angle 45 degrees has a perimeter of 64 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 8 ∧ 
  exterior_angle = 45 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 64 := by
sorry


end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1072_107231


namespace NUMINAMATH_CALUDE_largest_x_floor_ratio_l1072_107281

theorem largest_x_floor_ratio : ∃ (x : ℝ), x = 63/8 ∧ 
  (∀ (y : ℝ), y > x → (⌊y⌋ : ℝ) / y ≠ 8/9) ∧ 
  (⌊x⌋ : ℝ) / x = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_ratio_l1072_107281


namespace NUMINAMATH_CALUDE_bus_tour_tickets_l1072_107249

/-- Proves that the number of regular tickets sold is 41 -/
theorem bus_tour_tickets (total_tickets : ℕ) (senior_price regular_price : ℚ) (total_sales : ℚ) 
  (h1 : total_tickets = 65)
  (h2 : senior_price = 10)
  (h3 : regular_price = 15)
  (h4 : total_sales = 855) :
  ∃ (senior_tickets regular_tickets : ℕ),
    senior_tickets + regular_tickets = total_tickets ∧
    senior_price * senior_tickets + regular_price * regular_tickets = total_sales ∧
    regular_tickets = 41 := by
  sorry

end NUMINAMATH_CALUDE_bus_tour_tickets_l1072_107249


namespace NUMINAMATH_CALUDE_stone_145_is_5_l1072_107250

def stone_number (n : ℕ) : ℕ := 
  let cycle := 28
  n % cycle

theorem stone_145_is_5 : stone_number 145 = stone_number 5 := by
  sorry

end NUMINAMATH_CALUDE_stone_145_is_5_l1072_107250


namespace NUMINAMATH_CALUDE_cricket_theorem_l1072_107274

def cricket_problem (team_scores : List Nat) : Prop :=
  let n := team_scores.length
  let lost_matches := 6
  let won_matches := n - lost_matches
  let opponent_scores_lost := List.map (λ x => x + 2) (team_scores.take lost_matches)
  let opponent_scores_won := List.map (λ x => (x + 2) / 3) (team_scores.drop lost_matches)
  let total_opponent_score := opponent_scores_lost.sum + opponent_scores_won.sum
  
  n = 12 ∧
  team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] ∧
  total_opponent_score = 54

theorem cricket_theorem : 
  ∃ (team_scores : List Nat), cricket_problem team_scores :=
sorry

end NUMINAMATH_CALUDE_cricket_theorem_l1072_107274


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l1072_107244

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem symmetric_point_y_axis :
  let P : Point := { x := 2, y := 1 }
  let P' : Point := reflect_y_axis P
  P'.x = -2 ∧ P'.y = 1 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l1072_107244


namespace NUMINAMATH_CALUDE_equation_system_solutions_l1072_107223

/-- The system of equations has two types of solutions:
    1. (3, 5, 7, 9)
    2. (t, -t, t, -t) for any real t -/
theorem equation_system_solutions :
  ∀ (a b c d : ℝ),
    (a * b + a * c = 3 * b + 3 * c) ∧
    (b * c + b * d = 5 * c + 5 * d) ∧
    (a * c + c * d = 7 * a + 7 * d) ∧
    (a * d + b * d = 9 * a + 9 * b) →
    ((a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9) ∨
     (∃ t : ℝ, a = t ∧ b = -t ∧ c = t ∧ d = -t)) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solutions_l1072_107223


namespace NUMINAMATH_CALUDE_parallel_alternate_interior_false_l1072_107226

-- Define the concept of lines
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define the concept of angles
structure Angle :=
  (measure : ℝ)

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define alternate interior angles
def alternate_interior (a1 a2 : Angle) (l1 l2 : Line) : Prop :=
  -- This definition is simplified for the purpose of this statement
  true

-- The theorem to be proved
theorem parallel_alternate_interior_false :
  ∃ (l1 l2 : Line) (a1 a2 : Angle),
    parallel l1 l2 ∧ ¬(alternate_interior a1 a2 l1 l2 → a1.measure = a2.measure) :=
sorry

end NUMINAMATH_CALUDE_parallel_alternate_interior_false_l1072_107226


namespace NUMINAMATH_CALUDE_cafeteria_apple_count_l1072_107230

def initial_apples : ℕ := 65
def used_percentage : ℚ := 20 / 100
def bought_apples : ℕ := 15

theorem cafeteria_apple_count : 
  initial_apples - (initial_apples * used_percentage).floor + bought_apples = 67 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_apple_count_l1072_107230


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1072_107288

/-- Given that the solution set of ax^2 - bx - 1 ≥ 0 is [-1/2, -1/3], 
    prove that the solution set of ax^2 - bx - 1 < 0 is (2, 3) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ x ∈ Set.Icc (-1/2) (-1/3)) →
  (∀ x, ax^2 - b*x - 1 < 0 ↔ x ∈ Set.Ioo 2 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1072_107288
