import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l2964_296446

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a ≥ 0) → a ≥ -8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2964_296446


namespace NUMINAMATH_CALUDE_john_reaches_floor_pushups_in_12_weeks_l2964_296481

/-- Represents the number of days John trains per week -/
def training_days_per_week : ℕ := 5

/-- Represents the number of push-up variations John needs to progress through -/
def num_variations : ℕ := 4

/-- Represents the number of reps John needs to reach before progressing to the next variation -/
def reps_to_progress : ℕ := 20

/-- Calculates the total number of days it takes John to progress through all variations -/
def total_training_days : ℕ := (num_variations - 1) * reps_to_progress

/-- Calculates the number of weeks it takes John to reach floor push-ups -/
def weeks_to_floor_pushups : ℕ := total_training_days / training_days_per_week

/-- Theorem stating that it takes John 12 weeks to reach floor push-ups -/
theorem john_reaches_floor_pushups_in_12_weeks : weeks_to_floor_pushups = 12 := by
  sorry

end NUMINAMATH_CALUDE_john_reaches_floor_pushups_in_12_weeks_l2964_296481


namespace NUMINAMATH_CALUDE_jenny_basket_eggs_l2964_296404

def is_valid_basket_size (n : ℕ) : Prop :=
  n ≥ 5 ∧ 30 % n = 0 ∧ 42 % n = 0

theorem jenny_basket_eggs : ∃! n : ℕ, is_valid_basket_size n ∧ ∀ m : ℕ, is_valid_basket_size m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_jenny_basket_eggs_l2964_296404


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2964_296451

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and eccentricity √3,
    its asymptotes are given by y = ±√2 x -/
theorem hyperbola_asymptotes (a b : ℝ) (h : a > 0) (k : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  ((a^2 + b^2) / a^2 = 3) →
  (∃ c : ℝ, ∀ x : ℝ, (y = c * x ∨ y = -c * x) ↔ (x / a = y / b ∨ x / a = -y / b)) ∧
  c = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2964_296451


namespace NUMINAMATH_CALUDE_chess_team_photo_arrangements_l2964_296499

def chess_team_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  2 * (Nat.factorial num_boys) * (Nat.factorial num_girls)

theorem chess_team_photo_arrangements :
  chess_team_arrangements 3 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_photo_arrangements_l2964_296499


namespace NUMINAMATH_CALUDE_problem_solution_l2964_296472

/-- The function f(x) defined in the problem -/
def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

/-- The theorem statement -/
theorem problem_solution (m a : ℝ) (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∀ x : ℝ, f x a m ≥ 2) (h4 : a ≤ -5 ∨ a ≥ 5) : m = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2964_296472


namespace NUMINAMATH_CALUDE_original_price_l2964_296455

/-- Given an article with price changes and final price, calculate the original price -/
theorem original_price (q r : ℚ) : 
  (∃ (x : ℚ), x * (1 + q / 100) * (1 - r / 100) = 2) →
  (∃ (x : ℚ), x = 200 / (100 + q - r - q * r / 100)) :=
by sorry

end NUMINAMATH_CALUDE_original_price_l2964_296455


namespace NUMINAMATH_CALUDE_discount_calculation_l2964_296445

theorem discount_calculation (marked_price : ℝ) (discount_rate : ℝ) : 
  marked_price = 17.5 →
  discount_rate = 0.3 →
  2 * marked_price * (1 - discount_rate) = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l2964_296445


namespace NUMINAMATH_CALUDE_eight_round_game_probability_l2964_296407

/-- Represents the probability of a specific outcome in an 8-round game -/
def game_probability (p1 p2 p3 : ℝ) (n1 n2 n3 : ℕ) : ℝ :=
  (p1^n1 * p2^n2 * p3^n3) * (Nat.choose 8 n1 * Nat.choose (8 - n1) n2)

theorem eight_round_game_probability :
  let p1 := (1 : ℝ) / 2
  let p2 := (1 : ℝ) / 3
  let p3 := (1 : ℝ) / 6
  game_probability p1 p2 p3 4 3 1 = 35 / 324 := by
  sorry

end NUMINAMATH_CALUDE_eight_round_game_probability_l2964_296407


namespace NUMINAMATH_CALUDE_page_lines_increase_l2964_296458

theorem page_lines_increase (L : ℕ) (h : L + 60 = 240) : 
  (60 : ℝ) / L = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l2964_296458


namespace NUMINAMATH_CALUDE_smallest_divisor_of_Q_l2964_296411

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def Q (visible : Finset ℕ) : ℕ := 
  visible.prod id

theorem smallest_divisor_of_Q : 
  ∀ visible : Finset ℕ, visible ⊆ die_numbers → visible.card = 7 → 
    (∃ k : ℕ, Q visible = 192 * k) ∧ 
    ∀ m : ℕ, m < 192 → (∃ v : Finset ℕ, v ⊆ die_numbers ∧ v.card = 7 ∧ ¬(∃ k : ℕ, Q v = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_Q_l2964_296411


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2964_296422

-- Define a right-angled triangle with one side of length 11 and the other two sides being natural numbers
def RightTriangle (a b c : ℕ) : Prop :=
  a = 11 ∧ a^2 + b^2 = c^2

-- Define the perimeter of the triangle
def Perimeter (a b c : ℕ) : ℕ := a + b + c

-- Theorem statement
theorem right_triangle_perimeter :
  ∃ (a b c : ℕ), RightTriangle a b c ∧ Perimeter a b c = 132 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2964_296422


namespace NUMINAMATH_CALUDE_inheritance_calculation_l2964_296453

/-- The inheritance amount in dollars -/
def inheritance : ℝ := 38621

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The state tax rate as a decimal -/
def state_tax_rate : ℝ := 0.15

/-- The processing fee in dollars -/
def processing_fee : ℝ := 2500

/-- The total amount paid in taxes and fees in dollars -/
def total_paid : ℝ := 16500

theorem inheritance_calculation (x : ℝ) (h : x = inheritance) :
  federal_tax_rate * x + state_tax_rate * (1 - federal_tax_rate) * x + processing_fee = total_paid :=
by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l2964_296453


namespace NUMINAMATH_CALUDE_prob_three_same_color_l2964_296440

/-- A deck of cards with red and black colors -/
structure Deck :=
  (total : ℕ)
  (red : ℕ)
  (black : ℕ)
  (h1 : red + black = total)

/-- The probability of drawing three cards of the same color -/
def prob_same_color (d : Deck) : ℚ :=
  2 * (d.red.choose 3 / d.total.choose 3)

/-- The specific deck described in the problem -/
def modified_deck : Deck :=
  { total := 60
  , red := 30
  , black := 30
  , h1 := by simp }

/-- The main theorem stating the probability for the given deck -/
theorem prob_three_same_color :
  prob_same_color modified_deck = 406 / 1711 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_same_color_l2964_296440


namespace NUMINAMATH_CALUDE_remaining_cakes_l2964_296467

/-- The number of cakes Baker initially had -/
def initial_cakes : ℕ := 167

/-- The number of cakes Baker sold -/
def sold_cakes : ℕ := 108

/-- Theorem: The number of remaining cakes is 59 -/
theorem remaining_cakes : initial_cakes - sold_cakes = 59 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cakes_l2964_296467


namespace NUMINAMATH_CALUDE_cameron_paper_count_l2964_296401

theorem cameron_paper_count (initial_papers : ℕ) : 
  (initial_papers : ℚ) * (60 : ℚ) / 100 = 240 → initial_papers = 400 := by
  sorry

end NUMINAMATH_CALUDE_cameron_paper_count_l2964_296401


namespace NUMINAMATH_CALUDE_floor_sqrt_23_squared_l2964_296480

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_23_squared_l2964_296480


namespace NUMINAMATH_CALUDE_local_minimum_at_two_l2964_296487

/-- The function f(x) = x^3 - 12x --/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 12

theorem local_minimum_at_two :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_at_two_l2964_296487


namespace NUMINAMATH_CALUDE_three_number_difference_l2964_296432

theorem three_number_difference (x y z : ℝ) : 
  x = 2 * y ∧ x = 3 * z ∧ (x + y + z) / 3 = 88 → x - z = 96 := by
  sorry

end NUMINAMATH_CALUDE_three_number_difference_l2964_296432


namespace NUMINAMATH_CALUDE_value_preserving_interval_iff_m_in_M_l2964_296439

/-- A function f has a value-preserving interval [a,b] if it is monotonic on [a,b]
    and its range on [a,b] is [a,b] -/
def has_value_preserving_interval (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ f y < f x)) ∧
    (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

/-- The set of m values for which f(x) = x^2 - (1/2)x + m has a value-preserving interval -/
def M : Set ℝ :=
  {m | m ∈ Set.Icc (5/16) (9/16) ∪ Set.Icc (-11/16) (-7/16)}

/-- The main theorem stating the equivalence between the existence of a value-preserving interval
    and m being in the set M -/
theorem value_preserving_interval_iff_m_in_M :
  ∀ m : ℝ, has_value_preserving_interval (fun x => x^2 - (1/2)*x + m) ↔ m ∈ M :=
sorry

end NUMINAMATH_CALUDE_value_preserving_interval_iff_m_in_M_l2964_296439


namespace NUMINAMATH_CALUDE_angle_B_magnitude_triangle_area_l2964_296469

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition1 (t : Triangle) : Prop :=
  t.a + 2 * t.c = 2 * t.b * Real.cos t.A

def satisfiesCondition2 (t : Triangle) : Prop :=
  t.b = 2 * Real.sqrt 3

def satisfiesCondition3 (t : Triangle) : Prop :=
  t.a + t.c = 4

-- Theorem 1
theorem angle_B_magnitude (t : Triangle) (h : satisfiesCondition1 t) : t.B = 2 * Real.pi / 3 := by
  sorry

-- Theorem 2
theorem triangle_area (t : Triangle) 
  (h1 : satisfiesCondition1 t) 
  (h2 : satisfiesCondition2 t) 
  (h3 : satisfiesCondition3 t) : 
  (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_magnitude_triangle_area_l2964_296469


namespace NUMINAMATH_CALUDE_old_man_coins_l2964_296447

theorem old_man_coins (x y : ℕ) (h1 : x ≠ y) (h2 : x^2 - y^2 = 49 * (x - y)) : x + y = 49 := by
  sorry

end NUMINAMATH_CALUDE_old_man_coins_l2964_296447


namespace NUMINAMATH_CALUDE_system_solution_l2964_296443

theorem system_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2 * x - Real.sqrt (x * y) - 4 * Real.sqrt (x / y) + 2 = 0 ∧
   2 * x^2 + x^2 * y^4 = 18 * y^2) ↔
  ((x = 2 ∧ y = 2) ∨ (x = Real.rpow 286 (1/4) / 4 ∧ y = Real.rpow 286 (1/4))) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2964_296443


namespace NUMINAMATH_CALUDE_segment_movement_area_reduction_l2964_296406

theorem segment_movement_area_reduction (AB d : ℝ) (hAB : AB > 0) (hd : d > 0) :
  ∃ (swept_area : ℝ), swept_area < (AB * d) / 10000 ∧ swept_area ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_segment_movement_area_reduction_l2964_296406


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2964_296470

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = 2/3 ∧ Q = 8/9 ∧ R = -5/9) ∧
    ∀ (x : ℚ), x ≠ 1 → x ≠ 4 → x ≠ -2 →
      (x^2 + x - 8) / ((x - 1) * (x - 4) * (x + 2)) =
      P / (x - 1) + Q / (x - 4) + R / (x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2964_296470


namespace NUMINAMATH_CALUDE_base6_45_equals_decimal_29_l2964_296457

-- Define a function to convert a base-6 number to decimal
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

-- Theorem statement
theorem base6_45_equals_decimal_29 :
  base6ToDecimal [5, 4] = 29 := by
  sorry

end NUMINAMATH_CALUDE_base6_45_equals_decimal_29_l2964_296457


namespace NUMINAMATH_CALUDE_garment_costs_l2964_296430

/-- The cost of garment A -/
def cost_A : ℝ := 300

/-- The cost of garment B -/
def cost_B : ℝ := 200

/-- The total cost of garments A and B -/
def total_cost : ℝ := 500

/-- The profit margin for garment A -/
def profit_margin_A : ℝ := 0.3

/-- The profit margin for garment B -/
def profit_margin_B : ℝ := 0.2

/-- The total profit -/
def total_profit : ℝ := 130

/-- Theorem: Given the conditions, the costs of garments A and B are 300 yuan and 200 yuan respectively -/
theorem garment_costs : 
  cost_A + cost_B = total_cost ∧ 
  profit_margin_A * cost_A + profit_margin_B * cost_B = total_profit := by
  sorry

end NUMINAMATH_CALUDE_garment_costs_l2964_296430


namespace NUMINAMATH_CALUDE_max_value_condition_l2964_296494

theorem max_value_condition (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 4, |x^2 - 4*x + 9 - 2*m| + 2*m ≤ 9) ∧ 
  (∃ x ∈ Set.Icc 0 4, |x^2 - 4*x + 9 - 2*m| + 2*m = 9) ↔ 
  m ≤ 7/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_condition_l2964_296494


namespace NUMINAMATH_CALUDE_rug_area_theorem_l2964_296493

theorem rug_area_theorem (total_floor_area : ℝ) (two_layer_area : ℝ) (three_layer_area : ℝ) 
  (h1 : total_floor_area = 140)
  (h2 : two_layer_area = 22)
  (h3 : three_layer_area = 19) :
  total_floor_area + two_layer_area + 2 * three_layer_area = 200 :=
by sorry

end NUMINAMATH_CALUDE_rug_area_theorem_l2964_296493


namespace NUMINAMATH_CALUDE_runner_ends_at_start_l2964_296409

/-- A runner on a circular track --/
structure Runner where
  start : ℝ  -- Starting position on the track (in feet)
  distance : ℝ  -- Total distance run (in feet)

/-- The circular track --/
def track_circumference : ℝ := 60

/-- Theorem: A runner who starts at any point and runs exactly 5400 feet will end at the same point --/
theorem runner_ends_at_start (runner : Runner) (h : runner.distance = 5400) :
  runner.start = (runner.start + runner.distance) % track_circumference := by
  sorry

end NUMINAMATH_CALUDE_runner_ends_at_start_l2964_296409


namespace NUMINAMATH_CALUDE_central_square_area_central_square_area_proof_l2964_296427

/-- Given a square with side length 6 composed of smaller squares with side length 2,
    the area of the central square formed by removing one small square from each corner is 20. -/
theorem central_square_area : ℕ → ℕ → ℕ → Prop :=
  fun large_side small_side central_area =>
    large_side = 6 ∧ 
    small_side = 2 ∧ 
    large_side % small_side = 0 ∧
    (large_side / small_side) ^ 2 - 4 = 5 ∧ 
    central_area = 5 * small_side ^ 2 ∧
    central_area = 20

/-- Proof of the theorem -/
theorem central_square_area_proof : central_square_area 6 2 20 := by
  sorry

end NUMINAMATH_CALUDE_central_square_area_central_square_area_proof_l2964_296427


namespace NUMINAMATH_CALUDE_wickets_before_last_match_l2964_296464

/-- Represents a bowler's statistics -/
structure BowlerStats where
  wickets : ℕ
  runs : ℕ
  average : ℚ

/-- Calculates the new average after a match -/
def newAverage (stats : BowlerStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns) / (stats.wickets + newWickets)

/-- Theorem: Given the conditions, prove the bowler had taken 175 wickets before the last match -/
theorem wickets_before_last_match 
  (initialStats : BowlerStats)
  (h1 : initialStats.average = 12.4)
  (h2 : newAverage initialStats 8 26 = 12)
  : initialStats.wickets = 175 := by
  sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_l2964_296464


namespace NUMINAMATH_CALUDE_sqrt_difference_squared_l2964_296431

theorem sqrt_difference_squared : (Real.sqrt 25 - Real.sqrt 9)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_squared_l2964_296431


namespace NUMINAMATH_CALUDE_negation_and_to_or_l2964_296479

theorem negation_and_to_or (x y : ℝ) :
  ¬(x > 1 ∧ y > 2) ↔ (x ≤ 1 ∨ y ≤ 2) := by sorry

end NUMINAMATH_CALUDE_negation_and_to_or_l2964_296479


namespace NUMINAMATH_CALUDE_rahul_savings_fraction_l2964_296442

def total_savings : ℕ := 180000
def ppf_savings : ℕ := 72000
def nsc_savings : ℕ := total_savings - ppf_savings

def fraction_equality (x : ℚ) : Prop :=
  x * nsc_savings = (1/2 : ℚ) * ppf_savings

theorem rahul_savings_fraction : 
  ∃ (x : ℚ), fraction_equality x ∧ x = (1/3 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_rahul_savings_fraction_l2964_296442


namespace NUMINAMATH_CALUDE_wire_length_l2964_296488

/-- The length of a wire stretched between two poles -/
theorem wire_length (d h₁ h₂ : ℝ) (hd : d = 20) (hh₁ : h₁ = 8) (hh₂ : h₂ = 9) :
  Real.sqrt ((d ^ 2) + ((h₂ - h₁) ^ 2)) = Real.sqrt 401 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_l2964_296488


namespace NUMINAMATH_CALUDE_cost_price_correct_l2964_296474

/-- The cost price of a piece of clothing -/
def cost_price : ℝ := 108

/-- The marked price of the clothing -/
def marked_price : ℝ := 132

/-- The discount rate applied to the clothing -/
def discount_rate : ℝ := 0.1

/-- The profit rate after applying the discount -/
def profit_rate : ℝ := 0.1

/-- Theorem stating that the cost price is correct given the conditions -/
theorem cost_price_correct :
  marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_correct_l2964_296474


namespace NUMINAMATH_CALUDE_parabola_focus_to_line_distance_l2964_296417

/-- The distance from the focus of the parabola y² = 2x to the line x - √3y = 0 is 1/4 -/
theorem parabola_focus_to_line_distance : 
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 2*x}
  let line := {(x, y) : ℝ × ℝ | x - Real.sqrt 3 * y = 0}
  let focus : ℝ × ℝ := (1/2, 0)
  ∃ d : ℝ, d = 1/4 ∧ ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt ((p.1 - focus.1)^2 + (p.2 - focus.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_to_line_distance_l2964_296417


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l2964_296478

theorem log_sum_equals_two :
  2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l2964_296478


namespace NUMINAMATH_CALUDE_trig_equality_l2964_296452

theorem trig_equality (a b : ℝ) (θ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (Real.sin θ ^ 6 / a ^ 2 + Real.cos θ ^ 6 / b ^ 2 = 1 / (a ^ 2 + b ^ 2)) →
  (Real.sin θ ^ 12 / a ^ 5 + Real.cos θ ^ 12 / b ^ 5 = 1 / a ^ 5 + 1 / b ^ 5) :=
by sorry

end NUMINAMATH_CALUDE_trig_equality_l2964_296452


namespace NUMINAMATH_CALUDE_correct_calculation_l2964_296415

theorem correct_calculation (x : ℤ) (h : x - 59 = 43) : x - 46 = 56 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2964_296415


namespace NUMINAMATH_CALUDE_triangle_inequality_l2964_296400

theorem triangle_inequality (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2964_296400


namespace NUMINAMATH_CALUDE_min_value_in_region_l2964_296489

-- Define the region
def enclosed_region (x y : ℝ) : Prop :=
  abs x ≤ y ∧ y ≤ 2

-- Define the function to minimize
def f (x y : ℝ) : ℝ := 2 * x - y

-- Theorem statement
theorem min_value_in_region :
  ∃ (min : ℝ), min = -6 ∧
  ∀ (x y : ℝ), enclosed_region x y → f x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_in_region_l2964_296489


namespace NUMINAMATH_CALUDE_supplemental_tanks_needed_l2964_296437

def total_diving_time : ℕ := 8
def primary_tank_duration : ℕ := 2
def supplemental_tank_duration : ℕ := 1

theorem supplemental_tanks_needed :
  (total_diving_time - primary_tank_duration) / supplemental_tank_duration = 6 :=
by sorry

end NUMINAMATH_CALUDE_supplemental_tanks_needed_l2964_296437


namespace NUMINAMATH_CALUDE_ernie_makes_four_circles_l2964_296444

/-- The number of circles Ernie can make -/
def ernies_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ali_circles : ℕ) : ℕ :=
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle

/-- Theorem: Given the conditions from the problem, Ernie can make 4 circles -/
theorem ernie_makes_four_circles :
  ernies_circles 80 8 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ernie_makes_four_circles_l2964_296444


namespace NUMINAMATH_CALUDE_circle_area_with_constraints_fountain_base_area_l2964_296433

/-- The area of a circle with specific constraints -/
theorem circle_area_with_constraints (d : ℝ) (r : ℝ) :
  d = 20 →  -- diameter is 20 feet
  r ^ 2 = 10 ^ 2 + 15 ^ 2 →  -- radius squared equals 10^2 + 15^2 (from Pythagorean theorem)
  π * r ^ 2 = 325 * π := by
  sorry

/-- The main theorem proving the area of the circular base -/
theorem fountain_base_area : ∃ (A : ℝ), A = 325 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_constraints_fountain_base_area_l2964_296433


namespace NUMINAMATH_CALUDE_smallest_n_for_geometric_sums_l2964_296468

def is_geometric_sum (x : ℕ) : Prop :=
  ∃ (a r : ℕ), r > 1 ∧ x = a + a*r + a*r^2

theorem smallest_n_for_geometric_sums : 
  (∀ n : ℕ, n < 6 → ¬(is_geometric_sum (7*n + 1) ∧ is_geometric_sum (8*n + 1))) ∧
  (is_geometric_sum (7*6 + 1) ∧ is_geometric_sum (8*6 + 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_geometric_sums_l2964_296468


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2964_296435

/-- Given a quadratic inequality ax² + bx + c < 0 with solution set {x | x < -2 ∨ x > -1/2},
    prove that the solution set for ax² - bx + c > 0 is {x | 1/2 < x ∧ x < 2} -/
theorem quadratic_inequality_solution_set
  (a b c : ℝ)
  (h : ∀ x : ℝ, (a * x^2 + b * x + c < 0) ↔ (x < -2 ∨ x > -(1/2))) :
  ∀ x : ℝ, (a * x^2 - b * x + c > 0) ↔ (1/2 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2964_296435


namespace NUMINAMATH_CALUDE_josh_book_cost_l2964_296418

/-- Represents the cost of items and quantities purchased by Josh --/
structure ShoppingTrip where
  numFilms : ℕ
  filmCost : ℕ
  numBooks : ℕ
  numCDs : ℕ
  cdCost : ℕ
  totalSpent : ℕ

/-- Calculates the cost of each book given the shopping trip details --/
def bookCost (trip : ShoppingTrip) : ℕ :=
  (trip.totalSpent - trip.numFilms * trip.filmCost - trip.numCDs * trip.cdCost) / trip.numBooks

/-- Theorem stating that the cost of each book in Josh's shopping trip is 4 --/
theorem josh_book_cost :
  let trip : ShoppingTrip := {
    numFilms := 9,
    filmCost := 5,
    numBooks := 4,
    numCDs := 6,
    cdCost := 3,
    totalSpent := 79
  }
  bookCost trip = 4 := by sorry

end NUMINAMATH_CALUDE_josh_book_cost_l2964_296418


namespace NUMINAMATH_CALUDE_n_squared_divisible_by_144_l2964_296491

theorem n_squared_divisible_by_144 (n : ℕ+) (h : ∃ t : ℕ+, t = 12 ∧ ∀ k : ℕ+, k ∣ n → k ≤ t) :
  144 ∣ n^2 := by
  sorry

end NUMINAMATH_CALUDE_n_squared_divisible_by_144_l2964_296491


namespace NUMINAMATH_CALUDE_apples_on_tree_l2964_296441

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := 7

/-- The number of apples Rachel picked -/
def picked_apples : ℕ := 4

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := initial_apples - picked_apples

theorem apples_on_tree : remaining_apples = 3 := by
  sorry

end NUMINAMATH_CALUDE_apples_on_tree_l2964_296441


namespace NUMINAMATH_CALUDE_tank_difference_l2964_296419

theorem tank_difference (total : ℕ) (german allied sanchalian : ℕ) 
  (h1 : total = 115)
  (h2 : german = 2 * allied + 2)
  (h3 : allied = 3 * sanchalian + 1)
  (h4 : total = german + allied + sanchalian) :
  german - sanchalian = 59 := by
  sorry

end NUMINAMATH_CALUDE_tank_difference_l2964_296419


namespace NUMINAMATH_CALUDE_jacks_healthcare_contribution_l2964_296484

/-- Calculates the healthcare contribution in cents per hour given an hourly wage in dollars and a contribution rate as a percentage. -/
def healthcare_contribution (hourly_wage : ℚ) (contribution_rate : ℚ) : ℚ :=
  hourly_wage * 100 * (contribution_rate / 100)

/-- Proves that Jack's healthcare contribution is 57.5 cents per hour. -/
theorem jacks_healthcare_contribution :
  healthcare_contribution 25 2.3 = 57.5 := by
  sorry

end NUMINAMATH_CALUDE_jacks_healthcare_contribution_l2964_296484


namespace NUMINAMATH_CALUDE_expression_evaluation_l2964_296423

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2964_296423


namespace NUMINAMATH_CALUDE_function_properties_l2964_296465

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h1 : ∀ x, f (10 + x) = f (10 - x))
    (h2 : ∀ x, f (20 - x) = -f (20 + x)) :
    is_odd f ∧ has_period f 40 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2964_296465


namespace NUMINAMATH_CALUDE_regular_nonagon_diagonal_sum_l2964_296421

/-- A regular nonagon is a 9-sided polygon with all sides equal and all angles equal. -/
structure RegularNonagon where
  side_length : ℝ
  shortest_diagonal : ℝ
  longest_diagonal : ℝ

/-- 
In a regular nonagon, the longest diagonal is equal to the sum of 
the side length and the shortest diagonal.
-/
theorem regular_nonagon_diagonal_sum (n : RegularNonagon) : 
  n.longest_diagonal = n.side_length + n.shortest_diagonal := by
  sorry

end NUMINAMATH_CALUDE_regular_nonagon_diagonal_sum_l2964_296421


namespace NUMINAMATH_CALUDE_negation_of_existence_circle_negation_l2964_296424

theorem negation_of_existence (P : ℝ × ℝ → Prop) :
  (¬ ∃ p, P p) ↔ (∀ p, ¬ P p) := by sorry

theorem circle_negation :
  (¬ ∃ (x y : ℝ), x^2 + y^2 - 1 ≤ 0) ↔ (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_circle_negation_l2964_296424


namespace NUMINAMATH_CALUDE_min_unheard_lines_l2964_296434

/-- Represents the number of lines in a sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of sonnets Horatio read -/
def sonnets_read : ℕ := 7

/-- Represents the minimum number of additional sonnets Horatio prepared -/
def min_additional_sonnets : ℕ := 1

/-- Theorem stating the minimum number of unheard lines -/
theorem min_unheard_lines :
  min_additional_sonnets * lines_per_sonnet = 14 :=
by sorry

end NUMINAMATH_CALUDE_min_unheard_lines_l2964_296434


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2964_296403

/-- Given an arithmetic sequence with the first four terms as specified,
    prove that the fifth term has the given form. -/
theorem arithmetic_sequence_fifth_term
  (x y : ℝ)
  (seq : ℕ → ℝ)
  (h1 : seq 0 = x + y^2)
  (h2 : seq 1 = x + 2*y)
  (h3 : seq 2 = x*y^2)
  (h4 : seq 3 = x/y^2)
  (h_arithmetic : ∀ n : ℕ, seq (n + 1) - seq n = seq 1 - seq 0) :
  seq 4 = (y^6 - 2*y^5 + 4*y) / (y^4 + y^2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2964_296403


namespace NUMINAMATH_CALUDE_solution_difference_l2964_296459

/-- The equation we're working with -/
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3

/-- Definition of p and q as solutions to the equation -/
def p_and_q_are_solutions (p q : ℝ) : Prop :=
  equation p ∧ equation q ∧ p ≠ q

theorem solution_difference (p q : ℝ) :
  p_and_q_are_solutions p q → p > q → p - q = 10 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l2964_296459


namespace NUMINAMATH_CALUDE_emily_quiz_score_l2964_296410

def emily_scores : List ℝ := [96, 88, 90, 85, 94]

theorem emily_quiz_score (target_mean : ℝ) (sixth_score : ℝ) :
  target_mean = 92 ∧ sixth_score = 99 →
  (emily_scores.sum + sixth_score) / 6 = target_mean := by
  sorry

end NUMINAMATH_CALUDE_emily_quiz_score_l2964_296410


namespace NUMINAMATH_CALUDE_second_pipe_fill_time_l2964_296490

/-- The time it takes for the first pipe to fill the cistern -/
def t1 : ℝ := 10

/-- The time it takes for the third pipe to empty the cistern -/
def t3 : ℝ := 25

/-- The time it takes to fill the cistern when all pipes are opened simultaneously -/
def t_all : ℝ := 6.976744186046512

/-- The time it takes for the second pipe to fill the cistern -/
def t2 : ℝ := 11.994

theorem second_pipe_fill_time :
  ∃ (t2 : ℝ), t2 > 0 ∧ (1 / t1 + 1 / t2 - 1 / t3 = 1 / t_all) :=
sorry

end NUMINAMATH_CALUDE_second_pipe_fill_time_l2964_296490


namespace NUMINAMATH_CALUDE_product_second_fourth_is_seven_l2964_296485

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ
  /-- The tenth term of the sequence is 25 -/
  tenth_term : a₁ + 9 * d = 25
  /-- The common difference is 3 -/
  diff_is_3 : d = 3

/-- The product of the second and fourth terms is 7 -/
theorem product_second_fourth_is_seven (seq : ArithmeticSequence) :
  (seq.a₁ + seq.d) * (seq.a₁ + 3 * seq.d) = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_second_fourth_is_seven_l2964_296485


namespace NUMINAMATH_CALUDE_square_difference_of_product_and_sum_l2964_296448

theorem square_difference_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 16) (h2 : p + q = 10) : (p - q)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_product_and_sum_l2964_296448


namespace NUMINAMATH_CALUDE_hall_paving_l2964_296426

/-- The number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℚ :=
  (hall_length * hall_width) / (stone_length * stone_width)

/-- Theorem: 1800 stones are required to pave a 36m x 15m hall with 6dm x 5dm stones -/
theorem hall_paving :
  stones_required 36 15 0.6 0.5 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_hall_paving_l2964_296426


namespace NUMINAMATH_CALUDE_min_trains_for_800_passengers_l2964_296463

/-- Given a maximum capacity of passengers per train and a total number of passengers to transport,
    calculate the minimum number of trains required. -/
def min_trains (capacity : ℕ) (total_passengers : ℕ) : ℕ :=
  (total_passengers + capacity - 1) / capacity

theorem min_trains_for_800_passengers :
  min_trains 50 800 = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_trains_for_800_passengers_l2964_296463


namespace NUMINAMATH_CALUDE_train_length_l2964_296461

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 42 / 3600 → speed * time * 1000 = 700 := by sorry

end NUMINAMATH_CALUDE_train_length_l2964_296461


namespace NUMINAMATH_CALUDE_angle_identities_l2964_296497

/-- Given that α is an angle in the second quadrant and cos(α + π) = 3/13,
    prove that tan α = -4√10/3 and sin(α - π/2) * sin(-α - π) = -12√10/169 -/
theorem angle_identities (α : Real) 
    (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
    (h2 : Real.cos (α + π) = 3/13) :
    Real.tan α = -4 * Real.sqrt 10 / 3 ∧ 
    Real.sin (α - π/2) * Real.sin (-α - π) = -12 * Real.sqrt 10 / 169 := by
  sorry

end NUMINAMATH_CALUDE_angle_identities_l2964_296497


namespace NUMINAMATH_CALUDE_hundredth_ring_squares_l2964_296492

/-- The number of unit squares in the nth ring around a 2x2 central square -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 8

/-- The 100th ring contains 808 unit squares -/
theorem hundredth_ring_squares : ring_squares 100 = 808 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_ring_squares_l2964_296492


namespace NUMINAMATH_CALUDE_complex_coordinates_l2964_296454

theorem complex_coordinates (z : ℂ) (h : z = Complex.I * (2 + 4 * Complex.I)) : 
  z.re = -4 ∧ z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinates_l2964_296454


namespace NUMINAMATH_CALUDE_garage_wheels_l2964_296462

/-- The number of wheels in a garage with bicycles and cars -/
def total_wheels (num_bicycles : ℕ) (num_cars : ℕ) : ℕ :=
  num_bicycles * 2 + num_cars * 4

/-- Theorem: The total number of wheels in the garage is 82 -/
theorem garage_wheels :
  total_wheels 9 16 = 82 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_l2964_296462


namespace NUMINAMATH_CALUDE_tetrahedron_acute_angles_l2964_296425

/-- A tetrahedron with vertices S, A, B, and C -/
structure Tetrahedron where
  S : Point
  A : Point
  B : Point
  C : Point

/-- The dihedral angle between two faces of a tetrahedron -/
def dihedralAngle (t : Tetrahedron) (face1 face2 : Fin 4) : ℝ := sorry

/-- The planar angle at a vertex of a face in a tetrahedron -/
def planarAngle (t : Tetrahedron) (face : Fin 4) (vertex : Fin 3) : ℝ := sorry

/-- A predicate stating that an angle is acute -/
def isAcute (angle : ℝ) : Prop := angle > 0 ∧ angle < Real.pi / 2

theorem tetrahedron_acute_angles (t : Tetrahedron) :
  (∀ face1 face2, isAcute (dihedralAngle t face1 face2)) →
  (∀ face vertex, isAcute (planarAngle t face vertex)) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_acute_angles_l2964_296425


namespace NUMINAMATH_CALUDE_parallel_vectors_angle_l2964_296482

theorem parallel_vectors_angle (x : Real) : 
  let a : ℝ × ℝ := (Real.sin x, 3/4)
  let b : ℝ × ℝ := (1/3, (1/2) * Real.cos x)
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → 
  0 < x ∧ x < π/2 → 
  x = π/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_angle_l2964_296482


namespace NUMINAMATH_CALUDE_lateral_surface_area_l2964_296412

-- Define the frustum
structure Frustum where
  r₁ : ℝ  -- upper base radius
  r₂ : ℝ  -- lower base radius
  h : ℝ   -- height
  l : ℝ   -- slant height

-- Define the conditions
def frustum_conditions (f : Frustum) : Prop :=
  f.r₂ = 4 * f.r₁ ∧ f.h = 4 * f.r₁ ∧ f.l = 10

-- Theorem to prove
theorem lateral_surface_area (f : Frustum) 
  (hf : frustum_conditions f) : 
  π * (f.r₁ + f.r₂) * f.l = 100 * π := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_l2964_296412


namespace NUMINAMATH_CALUDE_minute_hand_rotation_1h50m_l2964_296414

/-- Represents the rotation of a clock's minute hand in degrees -/
def minute_hand_rotation (hours : ℕ) (minutes : ℕ) : ℤ :=
  -(hours * 360 + (minutes * 360) / 60)

/-- Theorem stating that for 1 hour and 50 minutes, the minute hand rotates -660 degrees -/
theorem minute_hand_rotation_1h50m : 
  minute_hand_rotation 1 50 = -660 := by
  sorry

end NUMINAMATH_CALUDE_minute_hand_rotation_1h50m_l2964_296414


namespace NUMINAMATH_CALUDE_always_quadratic_l2964_296498

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation (k^2+1)x^2-2x+1=0 is always a quadratic equation -/
theorem always_quadratic (k : ℝ) : 
  is_quadratic_equation (λ x => (k^2 + 1) * x^2 - 2 * x + 1) :=
sorry

end NUMINAMATH_CALUDE_always_quadratic_l2964_296498


namespace NUMINAMATH_CALUDE_recurrence_relations_hold_l2964_296466

def circle_radius : ℝ := 1

def perimeter_circumscribed (n : ℕ) : ℝ := sorry

def perimeter_inscribed (n : ℕ) : ℝ := sorry

theorem recurrence_relations_hold (n : ℕ) (h : n ≥ 3) :
  perimeter_circumscribed (2 * n) = (2 * perimeter_circumscribed n * perimeter_inscribed n) / (perimeter_circumscribed n + perimeter_inscribed n) ∧
  perimeter_inscribed (2 * n) = Real.sqrt (perimeter_inscribed n * perimeter_circumscribed (2 * n)) :=
sorry

end NUMINAMATH_CALUDE_recurrence_relations_hold_l2964_296466


namespace NUMINAMATH_CALUDE_gcf_of_2000_and_7700_l2964_296495

theorem gcf_of_2000_and_7700 : Nat.gcd 2000 7700 = 100 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_2000_and_7700_l2964_296495


namespace NUMINAMATH_CALUDE_platform_length_l2964_296460

/-- The length of a platform given train parameters -/
theorem platform_length (train_length : Real) (train_speed_kmh : Real) (time_to_pass : Real) :
  train_length = 360 ∧ 
  train_speed_kmh = 45 ∧ 
  time_to_pass = 43.2 →
  (train_speed_kmh * (1000 / 3600) * time_to_pass) - train_length = 180 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2964_296460


namespace NUMINAMATH_CALUDE_solution_range_l2964_296473

def P (a : ℝ) : Set ℝ := {x : ℝ | (x + 1) / (x + a) < 2}

theorem solution_range (a : ℝ) : (1 ∉ P a) ↔ a ∈ Set.Icc (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2964_296473


namespace NUMINAMATH_CALUDE_angle_trig_values_l2964_296456

def l₁ (x y : ℝ) : Prop := x - y = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y - 3 = 0

def intersection_point (P : ℝ × ℝ) : Prop :=
  l₁ P.1 P.2 ∧ l₂ P.1 P.2

theorem angle_trig_values (α : ℝ) (P : ℝ × ℝ) :
  intersection_point P →
  Real.sin α = Real.sqrt 2 / 2 ∧
  Real.cos α = Real.sqrt 2 / 2 ∧
  Real.tan α = 1 :=
by sorry

end NUMINAMATH_CALUDE_angle_trig_values_l2964_296456


namespace NUMINAMATH_CALUDE_inequality_range_l2964_296438

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l2964_296438


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l2964_296450

theorem circle_diameter_ratio (R S : ℝ) (hR : R > 0) (hS : S > 0)
  (h_area : π * R^2 = 0.25 * (π * S^2)) :
  2 * R = 0.5 * (2 * S) := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l2964_296450


namespace NUMINAMATH_CALUDE_not_prime_5n_plus_3_l2964_296436

theorem not_prime_5n_plus_3 (n : ℕ) (k m : ℤ) 
  (h1 : 2 * n + 1 = k^2) 
  (h2 : 3 * n + 1 = m^2) : 
  ¬ Nat.Prime (5 * n + 3) := by
sorry

end NUMINAMATH_CALUDE_not_prime_5n_plus_3_l2964_296436


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l2964_296413

theorem quadratic_two_real_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (m - 1) * x^2 - 4*x + 1 = 0 ∧ (m - 1) * y^2 - 4*y + 1 = 0) ↔ 
  (m ≤ 5 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l2964_296413


namespace NUMINAMATH_CALUDE_inequality_system_product_l2964_296416

theorem inequality_system_product (x y : ℤ) : 
  (x^3 + y^2 - 3*y + 1 < 0 ∧ 3*x^3 - y^2 + 3*y > 0) → 
  (∃ (y1 y2 : ℤ), y1 ≠ y2 ∧ 
    (x^3 + y1^2 - 3*y1 + 1 < 0 ∧ 3*x^3 - y1^2 + 3*y1 > 0) ∧
    (x^3 + y2^2 - 3*y2 + 1 < 0 ∧ 3*x^3 - y2^2 + 3*y2 > 0) ∧
    y1 * y2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_product_l2964_296416


namespace NUMINAMATH_CALUDE_zoo_giraffe_difference_l2964_296471

theorem zoo_giraffe_difference (total_giraffes : ℕ) (other_animals : ℕ) : 
  total_giraffes = 300 → 
  total_giraffes = 3 * other_animals → 
  total_giraffes - other_animals = 200 := by
sorry

end NUMINAMATH_CALUDE_zoo_giraffe_difference_l2964_296471


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2964_296475

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}

theorem complement_of_A_in_U :
  U \ A = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2964_296475


namespace NUMINAMATH_CALUDE_egg_pack_size_l2964_296402

/-- The number of rotten eggs in the pack -/
def rotten_eggs : ℕ := 3

/-- The probability of choosing 2 rotten eggs -/
def prob_two_rotten : ℚ := 47619047619047615 / 10000000000000000

/-- The total number of eggs in the pack -/
def total_eggs : ℕ := 36

/-- Theorem stating that given the number of rotten eggs and the probability of choosing 2 rotten eggs, 
    the total number of eggs in the pack is 36 -/
theorem egg_pack_size :
  (rotten_eggs : ℚ) / total_eggs * (rotten_eggs - 1 : ℚ) / (total_eggs - 1) = prob_two_rotten :=
sorry

end NUMINAMATH_CALUDE_egg_pack_size_l2964_296402


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l2964_296449

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: When a = 2, prove the solution set of f(x) ≥ 4
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Prove the range of a for which f(x) ≥ 4
theorem range_of_a :
  {a : ℝ | ∃ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l2964_296449


namespace NUMINAMATH_CALUDE_cubic_expression_value_l2964_296486

theorem cubic_expression_value (a b : ℝ) :
  (a * 1^3 + b * 1 + 1 = 5) → (a * (-1)^3 + b * (-1) + 1 = -3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l2964_296486


namespace NUMINAMATH_CALUDE_alpha_values_l2964_296429

/-- Given a function f where f(α) = 4, prove that α is either -4 or 2 -/
theorem alpha_values (f : ℝ → ℝ) (α : ℝ) (h : f α = 4) : α = -4 ∨ α = 2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_values_l2964_296429


namespace NUMINAMATH_CALUDE_kellys_games_l2964_296496

/-- Kelly's nintendo games problem -/
theorem kellys_games (initial_games : ℕ) (given_away : ℕ) (remaining_games : ℕ) : 
  initial_games = 106 → given_away = 64 → remaining_games = initial_games - given_away → remaining_games = 42 := by
  sorry

end NUMINAMATH_CALUDE_kellys_games_l2964_296496


namespace NUMINAMATH_CALUDE_incorrect_statement_l2964_296477

theorem incorrect_statement : ¬ (∀ a b c : ℝ, a > b → a * c > b * c) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l2964_296477


namespace NUMINAMATH_CALUDE_min_value_expression_l2964_296476

theorem min_value_expression (x y z k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k > 0) :
  (k * 4 * z) / (2 * x + y) + (k * 4 * x) / (y + 2 * z) + (k * y) / (x + z) ≥ 3 * k :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2964_296476


namespace NUMINAMATH_CALUDE_ball_bearing_savings_l2964_296428

/-- Calculates the savings when buying ball bearings during a sale with bulk discount --/
theorem ball_bearing_savings
  (num_machines : ℕ)
  (bearings_per_machine : ℕ)
  (regular_price : ℚ)
  (sale_price : ℚ)
  (bulk_discount : ℚ)
  (h1 : num_machines = 10)
  (h2 : bearings_per_machine = 30)
  (h3 : regular_price = 1)
  (h4 : sale_price = 3/4)
  (h5 : bulk_discount = 1/5)
  : (num_machines * bearings_per_machine * regular_price) -
    (num_machines * bearings_per_machine * sale_price * (1 - bulk_discount)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ball_bearing_savings_l2964_296428


namespace NUMINAMATH_CALUDE_statement_true_for_lines_statement_true_for_planes_statement_true_cases_l2964_296420

-- Define a type for geometric objects (lines or planes)
inductive GeometricObject
| Line
| Plane

-- Define a parallel relation
def parallel (a b : GeometricObject) : Prop := sorry

-- Define the statement we want to prove
def statement (x y z : GeometricObject) : Prop :=
  (parallel x z ∧ parallel y z) ∧ ¬(parallel x y)

-- Theorem for the case when all objects are lines
theorem statement_true_for_lines :
  ∃ (x y z : GeometricObject), 
    x = GeometricObject.Line ∧ 
    y = GeometricObject.Line ∧ 
    z = GeometricObject.Line ∧ 
    statement x y z := by sorry

-- Theorem for the case when all objects are planes
theorem statement_true_for_planes :
  ∃ (x y z : GeometricObject), 
    x = GeometricObject.Plane ∧ 
    y = GeometricObject.Plane ∧ 
    z = GeometricObject.Plane ∧ 
    statement x y z := by sorry

-- Main theorem combining both cases
theorem statement_true_cases :
  (∃ (x y z : GeometricObject), 
    x = GeometricObject.Line ∧ 
    y = GeometricObject.Line ∧ 
    z = GeometricObject.Line ∧ 
    statement x y z) ∧
  (∃ (x y z : GeometricObject), 
    x = GeometricObject.Plane ∧ 
    y = GeometricObject.Plane ∧ 
    z = GeometricObject.Plane ∧ 
    statement x y z) := by sorry

end NUMINAMATH_CALUDE_statement_true_for_lines_statement_true_for_planes_statement_true_cases_l2964_296420


namespace NUMINAMATH_CALUDE_no_four_digit_n_over_5_and_5n_l2964_296408

theorem no_four_digit_n_over_5_and_5n : 
  ¬ ∃ (n : ℕ), n > 0 ∧ 
    (1000 ≤ n / 5 ∧ n / 5 ≤ 9999) ∧ 
    (1000 ≤ 5 * n ∧ 5 * n ≤ 9999) :=
by sorry

end NUMINAMATH_CALUDE_no_four_digit_n_over_5_and_5n_l2964_296408


namespace NUMINAMATH_CALUDE_village_language_probability_l2964_296483

/-- Given a village with the following properties:
  - Total population is 1500
  - 800 people speak Tamil
  - 650 people speak English
  - 250 people speak both Tamil and English
  Prove that the probability of a randomly chosen person speaking neither English nor Tamil is 1/5 -/
theorem village_language_probability (total : ℕ) (tamil : ℕ) (english : ℕ) (both : ℕ)
  (h_total : total = 1500)
  (h_tamil : tamil = 800)
  (h_english : english = 650)
  (h_both : both = 250) :
  (total - (tamil + english - both)) / total = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_village_language_probability_l2964_296483


namespace NUMINAMATH_CALUDE_princess_puff_whisker_count_l2964_296405

/-- The number of whiskers Princess Puff has -/
def princess_puff_whiskers : ℕ := 14

/-- The number of whiskers Catman Do has -/
def catman_do_whiskers : ℕ := 22

theorem princess_puff_whisker_count :
  princess_puff_whiskers = 14 ∧
  catman_do_whiskers = 22 ∧
  catman_do_whiskers = 2 * princess_puff_whiskers - 6 :=
by sorry

end NUMINAMATH_CALUDE_princess_puff_whisker_count_l2964_296405
