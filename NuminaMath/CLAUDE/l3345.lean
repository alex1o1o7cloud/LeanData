import Mathlib

namespace NUMINAMATH_CALUDE_modular_inverse_of_7_mod_31_l3345_334505

theorem modular_inverse_of_7_mod_31 :
  ∃ x : ℕ, x ≤ 30 ∧ (7 * x) % 31 = 1 :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_7_mod_31_l3345_334505


namespace NUMINAMATH_CALUDE_coffee_mix_ratio_l3345_334598

theorem coffee_mix_ratio (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (50 * x + 40 * y) / (x + y) = (55 * x + 34 * y) / (x + y) ↔ x / y = 6 / 5 :=
by sorry

end NUMINAMATH_CALUDE_coffee_mix_ratio_l3345_334598


namespace NUMINAMATH_CALUDE_equation_solutions_l3345_334578

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (x₁ - 1) * (x₁ + 3) = 12 ∧ 
  (x₂ - 1) * (x₂ + 3) = 12 ∧ 
  x₁ = -5 ∧ 
  x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3345_334578


namespace NUMINAMATH_CALUDE_roots_sum_equals_four_l3345_334526

/-- Given that x₁ and x₂ are the roots of ln|x-2| = m for some real m, prove that x₁ + x₂ = 4 -/
theorem roots_sum_equals_four (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : Real.log (|x₁ - 2|) = m) 
  (h₂ : Real.log (|x₂ - 2|) = m) : 
  x₁ + x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_equals_four_l3345_334526


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l3345_334556

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -5 →
    1 / (x^3 + 2*x^2 - 25*x - 50) = A / (x - 2) + B / (x + 5) + C / ((x + 5)^2)) →
  B = -11/490 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l3345_334556


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l3345_334510

def has_exactly_six_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 6

def all_divisors_accommodate (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → n % d = 0

theorem smallest_number_with_conditions : 
  ∃ n : ℕ, 
    n % 18 = 0 ∧ 
    has_exactly_six_divisors n ∧
    all_divisors_accommodate n ∧
    (∀ m : ℕ, m < n → 
      ¬(m % 18 = 0 ∧ 
        has_exactly_six_divisors m ∧ 
        all_divisors_accommodate m)) ∧
    n = 72 :=
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l3345_334510


namespace NUMINAMATH_CALUDE_funfair_unsold_tickets_l3345_334508

/-- Calculates the number of unsold tickets at a school funfair --/
theorem funfair_unsold_tickets (total_rolls : ℕ) (tickets_per_roll : ℕ)
  (fourth_grade_percent : ℚ) (fifth_grade_percent : ℚ) (sixth_grade_percent : ℚ)
  (seventh_grade_percent : ℚ) (eighth_grade_percent : ℚ) (ninth_grade_tickets : ℕ) :
  total_rolls = 50 →
  tickets_per_roll = 250 →
  fourth_grade_percent = 30 / 100 →
  fifth_grade_percent = 40 / 100 →
  sixth_grade_percent = 25 / 100 →
  seventh_grade_percent = 35 / 100 →
  eighth_grade_percent = 20 / 100 →
  ninth_grade_tickets = 150 →
  ∃ (unsold : ℕ), unsold = 1898 := by
  sorry

#check funfair_unsold_tickets

end NUMINAMATH_CALUDE_funfair_unsold_tickets_l3345_334508


namespace NUMINAMATH_CALUDE_missing_number_proof_l3345_334504

theorem missing_number_proof : ∃ x : ℝ, x + Real.sqrt (-4 + 6 * 4 / 3) = 13 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3345_334504


namespace NUMINAMATH_CALUDE_sequence_a_10_l3345_334561

/-- A sequence satisfying the given properties -/
def Sequence (a : ℕ+ → ℤ) : Prop :=
  (∀ p q : ℕ+, a (p + q) = a p + a q) ∧ (a 2 = -6)

/-- The theorem to be proved -/
theorem sequence_a_10 (a : ℕ+ → ℤ) (h : Sequence a) : a 10 = -30 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_10_l3345_334561


namespace NUMINAMATH_CALUDE_unique_representation_l3345_334546

theorem unique_representation (A : ℕ) : 
  ∃! (x y : ℕ), A = ((x + y)^2 + 3*x + y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_representation_l3345_334546


namespace NUMINAMATH_CALUDE_range_of_x_l3345_334554

/-- Given a set M containing two elements, x^2 - 5x + 7 and 1, 
    prove that the range of real numbers x is all real numbers except 2 and 3. -/
theorem range_of_x (M : Set ℝ) (h : M = {x^2 - 5*x + 7 | x : ℝ} ∪ {1}) :
  {x : ℝ | x^2 - 5*x + 7 ≠ 1} = {x : ℝ | x ≠ 2 ∧ x ≠ 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l3345_334554


namespace NUMINAMATH_CALUDE_bea_earned_more_than_dawn_l3345_334532

/-- Calculates the difference in earnings between two lemonade sellers -/
def lemonade_earnings_difference (bea_price bea_sold dawn_price dawn_sold : ℕ) : ℕ :=
  bea_price * bea_sold - dawn_price * dawn_sold

/-- Proves that Bea earned 26 cents more than Dawn given the conditions -/
theorem bea_earned_more_than_dawn :
  lemonade_earnings_difference 25 10 28 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_bea_earned_more_than_dawn_l3345_334532


namespace NUMINAMATH_CALUDE_ellipse_and_circle_properties_l3345_334581

-- Define the points and shapes
structure Point where
  x : ℝ
  y : ℝ

def F : Point := ⟨0, -1⟩
def A : Point := ⟨0, 2⟩
def O : Point := ⟨0, 0⟩

structure Circle where
  center : Point
  radius : ℝ

structure Ellipse where
  center : Point
  a : ℝ
  b : ℝ

def Line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Define the problem conditions
def is_on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

def is_tangent_to_line (c : Circle) (l : ℝ → ℝ) : Prop :=
  ∃ p : Point, is_on_circle p c ∧ p.y = l p.x ∧
  ∀ q : Point, q ≠ p → is_on_circle q c → q.y ≠ l q.x

def is_on_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.b^2 + (p.y - e.center.y)^2 / e.a^2 = 1

def is_focus_of_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 + (p.y - e.center.y)^2 = e.a^2 - e.b^2

-- Define the theorem
theorem ellipse_and_circle_properties :
  ∀ (Q : Circle) (N : Ellipse) (M : ℝ → ℝ) (m : ℝ → ℝ → ℝ) (Z : ℝ → ℝ),
  (∀ x : ℝ, is_on_circle F Q) →
  (is_tangent_to_line Q (Line 0 1)) →
  (N.center = O) →
  (is_focus_of_ellipse F N) →
  (is_on_ellipse A N) →
  (∀ k : ℝ, ∃ B C D E : Point,
    is_on_ellipse B N ∧ is_on_ellipse C N ∧
    B.y = m k B.x ∧ C.y = m k C.x ∧
    D.x^2 = -4 * D.y ∧ E.x^2 = -4 * E.y ∧
    D.y = m k D.x ∧ E.y = m k E.x) →
  (∀ x : ℝ, M x = -x^2 / 4) →
  (N.a = 2 ∧ N.b = Real.sqrt 3) →
  (∀ k : ℝ, 9 ≤ Z k ∧ Z k < 12) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_properties_l3345_334581


namespace NUMINAMATH_CALUDE_expression_simplification_l3345_334522

theorem expression_simplification (x : ℝ) (h : x = Real.pi ^ 0 + 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (2 * x + 2)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3345_334522


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3345_334574

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3345_334574


namespace NUMINAMATH_CALUDE_square_ratio_problem_l3345_334537

theorem square_ratio_problem :
  let area_ratio : ℚ := 18 / 50
  let side_ratio : ℝ := Real.sqrt (area_ratio)
  ∃ (a b c : ℕ), 
    (a : ℝ) * Real.sqrt b / c = side_ratio ∧
    a = 3 ∧ b = 2 ∧ c = 5 ∧
    a + b + c = 10 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_problem_l3345_334537


namespace NUMINAMATH_CALUDE_minimum_value_complex_l3345_334552

theorem minimum_value_complex (z : ℂ) (h : Complex.abs (z - 3 + Complex.I) = 3) :
  (Complex.abs (z + 2 - 3 * Complex.I))^2 + (Complex.abs (z - 6 + 2 * Complex.I))^2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_complex_l3345_334552


namespace NUMINAMATH_CALUDE_first_player_wins_l3345_334519

/-- Represents the game state -/
structure GameState where
  stones : ℕ
  last_move : ℕ

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (move : ℕ) : Prop :=
  move > 0 ∧ move ≤ state.stones ∧
  (state.last_move = 0 ∨ state.last_move % move = 0)

/-- Defines the winning condition -/
def is_winning_state (state : GameState) : Prop :=
  state.stones = 0

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  ∃ (first_move : ℕ),
    valid_move { stones := 1992, last_move := 0 } first_move ∧
    ∀ (second_move : ℕ),
      valid_move { stones := 1992 - first_move, last_move := first_move } second_move →
      ∃ (strategy : GameState → ℕ),
        (∀ (state : GameState),
          valid_move state (strategy state)) ∧
        (∀ (state : GameState),
          ¬is_winning_state state →
          is_winning_state { stones := state.stones - strategy state, last_move := strategy state }) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l3345_334519


namespace NUMINAMATH_CALUDE_apple_distribution_l3345_334506

theorem apple_distribution (t x : ℕ) (h1 : t = 4) (h2 : (9 * t * x) / 10 - 6 = 48) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3345_334506


namespace NUMINAMATH_CALUDE_ages_sum_l3345_334549

theorem ages_sum (a b s : ℕ+) : 
  (3 * a + 5 + b = s) →
  (6 * s^2 = 2 * a^2 + 10 * b^2) →
  (Nat.gcd (Nat.gcd a.val b.val) s.val = 1) →
  (a + b + s = 19) := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l3345_334549


namespace NUMINAMATH_CALUDE_subtraction_problem_l3345_334586

theorem subtraction_problem : 
  (7000 / 10) - (7000 * (1 / 10) / 100) = 693 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3345_334586


namespace NUMINAMATH_CALUDE_solution_for_n_equals_S_plus_U_squared_l3345_334540

def S (n : ℕ) : ℕ := sorry  -- Sum of digits of n

def U (n : ℕ) : ℕ := sorry  -- Unit digit of n

theorem solution_for_n_equals_S_plus_U_squared :
  ∀ n : ℕ, n > 0 → (n = S n + (U n)^2) ↔ (n = 13 ∨ n = 46 ∨ n = 99) := by
  sorry

end NUMINAMATH_CALUDE_solution_for_n_equals_S_plus_U_squared_l3345_334540


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l3345_334553

theorem pure_imaginary_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ k : ℝ, (3 - 5*Complex.I) * (a + b*Complex.I) * (1 + 2*Complex.I) = k * Complex.I) →
  a / b = -1 / 7 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l3345_334553


namespace NUMINAMATH_CALUDE_average_headcount_rounded_l3345_334527

def fall_03_04_headcount : ℕ := 11500
def fall_04_05_headcount : ℕ := 11600
def fall_05_06_headcount : ℕ := 11300

def average_headcount : ℚ := (fall_03_04_headcount + fall_04_05_headcount + fall_05_06_headcount) / 3

def round_to_nearest (x : ℚ) : ℕ := 
  (x + 1/2).floor.toNat

theorem average_headcount_rounded : round_to_nearest average_headcount = 11467 := by
  sorry

end NUMINAMATH_CALUDE_average_headcount_rounded_l3345_334527


namespace NUMINAMATH_CALUDE_van_distance_van_distance_proof_l3345_334591

/-- The distance covered by a van under specific conditions -/
theorem van_distance : ℝ :=
  let initial_time : ℝ := 6
  let new_time_factor : ℝ := 3/2
  let new_speed : ℝ := 28
  let distance := new_speed * (new_time_factor * initial_time)
  252

/-- Proof that the van's distance is 252 km -/
theorem van_distance_proof : van_distance = 252 := by
  sorry

end NUMINAMATH_CALUDE_van_distance_van_distance_proof_l3345_334591


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_not_q_l3345_334559

-- Define the conditions p and q
def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 4
def q (x : ℝ) : Prop := |x - 2| > 1

-- Define the negation of q
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that p is a necessary but not sufficient condition for ¬q
theorem p_necessary_not_sufficient_for_not_q :
  (∀ x, not_q x → p x) ∧ 
  (∃ x, p x ∧ ¬(not_q x)) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_not_q_l3345_334559


namespace NUMINAMATH_CALUDE_complete_graph_10_coloring_l3345_334583

/-- A complete graph with 10 vertices -/
def CompleteGraph10 := Fin 10

/-- The type of edge colorings for CompleteGraph10 -/
def EdgeColoring (k : ℕ) := CompleteGraph10 → CompleteGraph10 → Fin k

/-- Predicate to check if k vertices form a k-colored subgraph -/
def is_k_colored_subgraph (k : ℕ) (coloring : EdgeColoring k) (vertices : Finset CompleteGraph10) : Prop :=
  vertices.card = k ∧
  ∀ (v w : CompleteGraph10), v ∈ vertices → w ∈ vertices → v ≠ w →
    ∃ (c : Fin k), ∀ (x y : CompleteGraph10), x ∈ vertices → y ∈ vertices → x ≠ y →
      coloring x y = c → x = v ∧ y = w

/-- Main theorem: k-coloring of CompleteGraph10 is possible iff k ≥ 5 -/
theorem complete_graph_10_coloring (k : ℕ) :
  (∃ (coloring : EdgeColoring k),
    ∀ (vertices : Finset CompleteGraph10),
      vertices.card = k → is_k_colored_subgraph k coloring vertices) ↔
  k ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_complete_graph_10_coloring_l3345_334583


namespace NUMINAMATH_CALUDE_investment_percentage_problem_l3345_334533

theorem investment_percentage_problem (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ) 
  (second_rate : ℝ) (third_rate : ℝ) (desired_income : ℝ) (x : ℝ) :
  total_investment = 10000 ∧ 
  first_investment = 4000 ∧ 
  second_investment = 3500 ∧ 
  second_rate = 0.04 ∧ 
  third_rate = 0.064 ∧ 
  desired_income = 500 ∧
  first_investment * (x / 100) + second_investment * second_rate + 
    (total_investment - first_investment - second_investment) * third_rate = desired_income →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_problem_l3345_334533


namespace NUMINAMATH_CALUDE_exists_k_sum_of_digits_equal_l3345_334543

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number contains the digit 9 -/
def hasNoNine (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem exists_k_sum_of_digits_equal : 
  ∃ k : ℕ, k > 0 ∧ hasNoNine k ∧ sumOfDigits k = sumOfDigits (2^(24^2017) * k) := by sorry

end NUMINAMATH_CALUDE_exists_k_sum_of_digits_equal_l3345_334543


namespace NUMINAMATH_CALUDE_books_to_buy_l3345_334597

/-- Given that 3 books cost $18.72 and you have $37.44, prove that you can buy 6 books. -/
theorem books_to_buy (cost_of_three : ℝ) (total_money : ℝ) : 
  cost_of_three = 18.72 → total_money = 37.44 → 
  (total_money / (cost_of_three / 3)) = 6 := by
sorry

end NUMINAMATH_CALUDE_books_to_buy_l3345_334597


namespace NUMINAMATH_CALUDE_line_parabola_intersection_condition_l3345_334588

/-- Parabola C with equation x² = 1/2 * y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 1/2 * y

/-- Line passing through points (0, -4) and (t, 0) -/
def line_AB (t x y : ℝ) : Prop := 4*x - t*y - 4*t = 0

/-- The line does not intersect the parabola -/
def no_intersection (t : ℝ) : Prop :=
  ∀ x y : ℝ, parabola_C x y ∧ line_AB t x y → False

/-- The range of t for which the line does not intersect the parabola -/
theorem line_parabola_intersection_condition (t : ℝ) :
  no_intersection t ↔ t < -Real.sqrt 2 ∨ t > Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_condition_l3345_334588


namespace NUMINAMATH_CALUDE_rays_dog_walking_problem_l3345_334502

/-- Ray's dog walking problem -/
theorem rays_dog_walking_problem (x : ℕ) : 
  (∀ (total_blocks : ℕ), total_blocks = 3 * (x + 7 + 11) → total_blocks = 66) → 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_rays_dog_walking_problem_l3345_334502


namespace NUMINAMATH_CALUDE_profit_ratio_theorem_l3345_334518

/-- Represents the investment and profit information for a partner -/
structure Partner where
  investment : ℕ
  months : ℕ

/-- Calculates the profit factor for a partner -/
def profitFactor (p : Partner) : ℕ := p.investment * p.months

/-- Theorem stating the profit ratio of two partners given their investments and time periods -/
theorem profit_ratio_theorem (p q : Partner) 
  (h_investment_ratio : p.investment * 5 = q.investment * 7)
  (h_p_months : p.months = 5)
  (h_q_months : q.months = 9) :
  profitFactor p * 9 = profitFactor q * 7 := by
  sorry

#check profit_ratio_theorem

end NUMINAMATH_CALUDE_profit_ratio_theorem_l3345_334518


namespace NUMINAMATH_CALUDE_house_transaction_net_change_l3345_334517

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  ownsHouse : Bool

/-- Represents a house transaction -/
structure Transaction where
  seller : String
  buyer : String
  price : Int

/-- Calculate the net change in wealth after transactions -/
def netChangeInWealth (initial : FinancialState) (final : FinancialState) (initialHouseValue : Int) : Int :=
  final.cash - initial.cash + (if final.ownsHouse then initialHouseValue else 0) - (if initial.ownsHouse then initialHouseValue else 0)

theorem house_transaction_net_change :
  let initialHouseValue := 15000
  let initialA := FinancialState.mk 15000 true
  let initialB := FinancialState.mk 20000 false
  let transaction1 := Transaction.mk "A" "B" 18000
  let transaction2 := Transaction.mk "B" "A" 12000
  let finalA := FinancialState.mk 21000 true
  let finalB := FinancialState.mk 14000 false
  (netChangeInWealth initialA finalA initialHouseValue = 6000) ∧
  (netChangeInWealth initialB finalB initialHouseValue = -6000) := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_net_change_l3345_334517


namespace NUMINAMATH_CALUDE_calculate_expression_l3345_334582

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 398 = 5186 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3345_334582


namespace NUMINAMATH_CALUDE_f_sum_reciprocal_l3345_334536

theorem f_sum_reciprocal (x : ℝ) (hx : x > 0) : 
  let f := fun (y : ℝ) => y / (y + 1)
  f x + f (1/x) = 1 := by
sorry

end NUMINAMATH_CALUDE_f_sum_reciprocal_l3345_334536


namespace NUMINAMATH_CALUDE_characterization_of_M_inequality_for_product_one_l3345_334542

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Define the set M
def M : Set ℝ := {x | f x ≤ 2}

-- Theorem 1: Characterization of set M
theorem characterization_of_M : M = {x | -5 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: Inequality for positive numbers with product 1
theorem inequality_for_product_one (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 1/a + 1/b + 1/c := by sorry

end NUMINAMATH_CALUDE_characterization_of_M_inequality_for_product_one_l3345_334542


namespace NUMINAMATH_CALUDE_otimes_identity_l3345_334515

-- Define the new operation ⊗
def otimes (x y : ℝ) : ℝ := x^2 + y^3

-- Theorem statement
theorem otimes_identity (k : ℝ) : otimes k (otimes k k) = k^2 + k^6 + 6*k^7 + k^9 := by
  sorry

end NUMINAMATH_CALUDE_otimes_identity_l3345_334515


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3345_334512

theorem modulus_of_complex_fraction : 
  Complex.abs ((2 - Complex.I) / (1 + Complex.I)) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3345_334512


namespace NUMINAMATH_CALUDE_only_solution_is_two_l3345_334531

/-- Represents the number constructed in the problem -/
def constructNumber (k : ℕ) : ℕ :=
  (10^2000 - 1) - (10^k - 1) * 10^(2000 - k) - (10^1001 - 1)

/-- The main theorem stating that k = 2 is the only solution -/
theorem only_solution_is_two :
  ∃! k : ℕ, k > 0 ∧ ∃ m : ℕ, constructNumber k = m^2 :=
sorry

end NUMINAMATH_CALUDE_only_solution_is_two_l3345_334531


namespace NUMINAMATH_CALUDE_augmented_matrix_solution_l3345_334541

/-- Given an augmented matrix and its solution, prove that c₁ - c₂ = -1 -/
theorem augmented_matrix_solution (c₁ c₂ : ℝ) : 
  (2 * 2 + 3 * 1 = c₁) → 
  (3 * 2 + 2 * 1 = c₂) → 
  c₁ - c₂ = -1 := by
sorry

end NUMINAMATH_CALUDE_augmented_matrix_solution_l3345_334541


namespace NUMINAMATH_CALUDE_circus_receipts_l3345_334593

theorem circus_receipts (total_tickets : ℕ) (adult_ticket_cost : ℕ) (child_ticket_cost : ℕ) (adult_tickets_sold : ℕ) :
  total_tickets = 522 →
  adult_ticket_cost = 15 →
  child_ticket_cost = 8 →
  adult_tickets_sold = 130 →
  (adult_tickets_sold * adult_ticket_cost + (total_tickets - adult_tickets_sold) * child_ticket_cost) = 5086 :=
by sorry

end NUMINAMATH_CALUDE_circus_receipts_l3345_334593


namespace NUMINAMATH_CALUDE_same_terminal_side_l3345_334585

-- Define a function to normalize angles to the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Theorem stating that -390° and 330° have the same terminal side
theorem same_terminal_side : normalizeAngle (-390) = normalizeAngle 330 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l3345_334585


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a9_l3345_334523

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a9 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 7 = 16)
  (h_a3 : a 3 = 1) :
  a 9 = 15 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a9_l3345_334523


namespace NUMINAMATH_CALUDE_chloe_winter_clothing_l3345_334539

/-- Calculates the total number of winter clothing items given the number of boxes and items per box. -/
def total_winter_clothing (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) : ℕ :=
  num_boxes * (scarves_per_box + mittens_per_box)

/-- Proves that Chloe has 32 pieces of winter clothing given the problem conditions. -/
theorem chloe_winter_clothing : 
  total_winter_clothing 4 2 6 = 32 := by
  sorry

#eval total_winter_clothing 4 2 6

end NUMINAMATH_CALUDE_chloe_winter_clothing_l3345_334539


namespace NUMINAMATH_CALUDE_new_energy_vehicle_analysis_l3345_334562

def daily_distances : List Int := [-8, -12, -16, 0, 22, 31, 33]
def standard_distance : Int := 50
def gasoline_consumption : Rat := 5.5
def gasoline_price : Rat := 8.4
def electric_consumption : Rat := 15
def electricity_price : Rat := 0.5

theorem new_energy_vehicle_analysis :
  let max_distance := daily_distances.foldl max (daily_distances.head!)
  let min_distance := daily_distances.foldl min (daily_distances.head!)
  let total_distance := daily_distances.sum
  let gasoline_cost := (total_distance : Rat) / 100 * gasoline_consumption * gasoline_price
  let electric_cost := (total_distance : Rat) / 100 * electric_consumption * electricity_price
  (max_distance - min_distance = 49) ∧
  (total_distance = 50) ∧
  (gasoline_cost - electric_cost = 154.8) := by
  sorry


end NUMINAMATH_CALUDE_new_energy_vehicle_analysis_l3345_334562


namespace NUMINAMATH_CALUDE_minimum_value_at_one_l3345_334513

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (a + 1) * x^2 - (a^2 + 3*a - 3) * x

theorem minimum_value_at_one (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 1) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_at_one_l3345_334513


namespace NUMINAMATH_CALUDE_lines_cannot_form_triangle_l3345_334525

-- Define the lines
def l₁ (x y : ℝ) : Prop := 4 * x + y = 4
def l₂ (m x y : ℝ) : Prop := m * x + y = 0
def l₃ (m x y : ℝ) : Prop := 2 * x - 3 * m * y = 4

-- Define when lines are parallel
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

-- Define when three lines intersect at a single point
def intersect_at_point (m : ℝ) : Prop :=
  ∃ x y : ℝ, l₁ x y ∧ l₂ m x y ∧ l₃ m x y

-- Theorem statement
theorem lines_cannot_form_triangle (m : ℝ) : 
  (¬∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ, 
    l₁ x₁ y₁ ∧ l₂ m x₂ y₂ ∧ l₃ m x₃ y₃ ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧ (x₃ ≠ x₁ ∨ y₃ ≠ y₁)) ↔ 
  (m = 4 ∨ m = -1/6 ∨ m = -1 ∨ m = 2/3) :=
sorry

end NUMINAMATH_CALUDE_lines_cannot_form_triangle_l3345_334525


namespace NUMINAMATH_CALUDE_chucks_team_leads_l3345_334501

/-- Represents a team's scoring in a single quarter -/
structure QuarterScore where
  fieldGoals : ℕ
  threePointers : ℕ
  freeThrows : ℕ

/-- Calculates the total points for a quarter -/
def quarterPoints (qs : QuarterScore) : ℕ :=
  2 * qs.fieldGoals + 3 * qs.threePointers + qs.freeThrows

/-- Represents a team's scoring for the entire game -/
structure GameScore where
  q1 : QuarterScore
  q2 : QuarterScore
  q3 : QuarterScore
  q4 : QuarterScore
  technicalFouls : ℕ

/-- Calculates the total points for a team in the game -/
def totalPoints (gs : GameScore) : ℕ :=
  quarterPoints gs.q1 + quarterPoints gs.q2 + quarterPoints gs.q3 + quarterPoints gs.q4 + gs.technicalFouls

theorem chucks_team_leads :
  let chucksTeam : GameScore := {
    q1 := { fieldGoals := 9, threePointers := 0, freeThrows := 5 },
    q2 := { fieldGoals := 6, threePointers := 3, freeThrows := 0 },
    q3 := { fieldGoals := 4, threePointers := 2, freeThrows := 6 },
    q4 := { fieldGoals := 8, threePointers := 1, freeThrows := 0 },
    technicalFouls := 3
  }
  let yellowTeam : GameScore := {
    q1 := { fieldGoals := 7, threePointers := 4, freeThrows := 0 },
    q2 := { fieldGoals := 5, threePointers := 2, freeThrows := 3 },
    q3 := { fieldGoals := 6, threePointers := 2, freeThrows := 0 },
    q4 := { fieldGoals := 4, threePointers := 3, freeThrows := 2 },
    technicalFouls := 2
  }
  totalPoints chucksTeam - totalPoints yellowTeam = 2 := by
  sorry

end NUMINAMATH_CALUDE_chucks_team_leads_l3345_334501


namespace NUMINAMATH_CALUDE_prove_M_value_l3345_334507

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : Int
  diff : Int

/-- The row sequence -/
def rowSeq : ArithmeticSequence := { first := 12, diff := -7 }

/-- The first column sequence -/
def col1Seq : ArithmeticSequence := { first := -11, diff := 9 }

/-- The second column sequence -/
def col2Seq : ArithmeticSequence := { first := -35, diff := 5 }

/-- Get the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : Nat) : Int :=
  seq.first + seq.diff * (n - 1)

theorem prove_M_value : 
  nthTerm rowSeq 1 = 12 ∧ 
  nthTerm col1Seq 4 = 7 ∧ 
  nthTerm col1Seq 5 = 16 ∧
  nthTerm col2Seq 5 = -10 ∧
  col2Seq.first = -35 := by sorry

end NUMINAMATH_CALUDE_prove_M_value_l3345_334507


namespace NUMINAMATH_CALUDE_min_value_a2b_l3345_334530

-- Define the function f
def f (x : ℝ) : ℝ := |x^2 - 6|

-- State the theorem
theorem min_value_a2b (a b : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : f a = f b) :
  ∃ (m : ℝ), m = -4 ∧ ∀ (x y : ℝ), x < y ∧ y < 0 ∧ f x = f y → m ≤ x^2 * y :=
sorry

end NUMINAMATH_CALUDE_min_value_a2b_l3345_334530


namespace NUMINAMATH_CALUDE_ellipse_equation_l3345_334516

theorem ellipse_equation (x y : ℝ) :
  let a : ℝ := 4
  let b : ℝ := Real.sqrt 7
  let ε : ℝ := 0.75
  let passes_through : Prop := (-3)^2 / a^2 + 1.75^2 / b^2 = 1
  let eccentricity : Prop := ε = Real.sqrt (a^2 - b^2) / a
  passes_through ∧ eccentricity →
  x^2 / 16 + y^2 / 7 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3345_334516


namespace NUMINAMATH_CALUDE_cone_properties_l3345_334569

/-- Properties of a right circular cone -/
theorem cone_properties (V h r l : ℝ) (hV : V = 16 * Real.pi) (hh : h = 6) 
  (hVol : (1/3) * Real.pi * r^2 * h = V) 
  (hSlant : l^2 = r^2 + h^2) : 
  2 * Real.pi * r = 4 * Real.sqrt 2 * Real.pi ∧ 
  Real.pi * r * l = 4 * Real.sqrt 22 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_cone_properties_l3345_334569


namespace NUMINAMATH_CALUDE_hundred_decomposition_l3345_334577

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def isPerfectCube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def isValidDecomposition (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  isPerfectSquare a ∧ isPerfectSquare b ∧ isPerfectCube c

theorem hundred_decomposition :
  ∃! (a b c : ℕ), a + b + c = 100 ∧ isValidDecomposition a b c :=
sorry

end NUMINAMATH_CALUDE_hundred_decomposition_l3345_334577


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3345_334528

/-- The surface area of a sphere circumscribing a cube with edge length 1 is 3π. -/
theorem circumscribed_sphere_surface_area (cube_edge : ℝ) (h : cube_edge = 1) :
  let sphere_radius := (Real.sqrt 3 / 2) * cube_edge
  4 * Real.pi * sphere_radius^2 = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3345_334528


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3345_334599

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
    (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = -1 ∧
    B = -3 ∧
    C = 1 ∧
    D = 2/3 ∧
    E = 33 ∧
    F = 17 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3345_334599


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_for_given_solution_set_l3345_334558

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Theorem for part (I)
theorem solution_set_when_a_is_one (x : ℝ) :
  (f 1 x ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) :=
sorry

-- Theorem for part (II)
theorem a_value_for_given_solution_set (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_for_given_solution_set_l3345_334558


namespace NUMINAMATH_CALUDE_cubic_factorization_l3345_334563

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3345_334563


namespace NUMINAMATH_CALUDE_investment_difference_l3345_334595

theorem investment_difference (x y : ℝ) : 
  x = 1000 →
  x + y = 1000 →
  0.02 * x + 0.04 * (x + y) = 92 →
  y = 800 := by
sorry

end NUMINAMATH_CALUDE_investment_difference_l3345_334595


namespace NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l3345_334520

theorem sqrt_plus_reciprocal_inequality (x : ℝ) (h : x > 0) :
  Real.sqrt x + 1 / Real.sqrt x ≥ 2 ∧
  (Real.sqrt x + 1 / Real.sqrt x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l3345_334520


namespace NUMINAMATH_CALUDE_right_triangle_area_l3345_334570

/-- The area of a right-angled triangle with perpendicular sides of lengths √12 cm and √6 cm is 3√2 square centimeters. -/
theorem right_triangle_area : 
  let side1 : ℝ := Real.sqrt 12
  let side2 : ℝ := Real.sqrt 6
  (1 / 2 : ℝ) * side1 * side2 = 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3345_334570


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3345_334571

-- Problem 1
theorem problem_1 (x : ℝ) : x^2 * x^3 - x^5 = 0 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (a + 1)^2 + 2*a*(a - 1) = 3*a^2 + 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3345_334571


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l3345_334503

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  ∃ (a : ℝ), (x - a)^2 + (y - a)^2 = 36 ∧ a = 4

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  3*x - 4*y - 16 = 0 ∨ x = 0

-- Theorem statement
theorem circle_and_line_properties :
  -- Circle M passes through A(√2, -√2) and B(10, 4)
  circle_M (Real.sqrt 2) (-Real.sqrt 2) ∧ circle_M 10 4 ∧
  -- The center of circle M lies on the line y = x
  ∃ (a : ℝ), circle_M a a ∧
  -- A line m passing through (0, -4) intersects circle M to form a chord of length 4√5
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_m x₁ y₁ ∧ line_m x₂ y₂ ∧
    circle_M x₁ y₁ ∧ circle_M x₂ y₂ ∧
    line_m 0 (-4) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 80 →
  -- The standard equation of circle M is (x-4)² + (y-4)² = 36
  ∀ (x y : ℝ), circle_M x y ↔ (x - 4)^2 + (y - 4)^2 = 36 ∧
  -- The equation of line m is either 3x - 4y - 16 = 0 or x = 0
  ∀ (x y : ℝ), line_m x y ↔ (3*x - 4*y - 16 = 0 ∨ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l3345_334503


namespace NUMINAMATH_CALUDE_soccer_school_admission_probability_l3345_334547

/-- Represents the probability of being admitted to the soccer school -/
def admission_probability (p_assistant : ℝ) (p_head : ℝ) : ℝ :=
  p_assistant * p_assistant + 2 * p_assistant * (1 - p_assistant) * p_head

/-- The probability of the young soccer enthusiast being admitted to the well-known soccer school is 0.4 -/
theorem soccer_school_admission_probability : 
  admission_probability 0.5 0.3 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_soccer_school_admission_probability_l3345_334547


namespace NUMINAMATH_CALUDE_total_rectangles_in_diagram_l3345_334524

/-- Represents a rectangle in the diagram -/
structure Rectangle where
  id : Nat

/-- Represents the diagram with rectangles -/
structure Diagram where
  rectangles : List Rectangle

/-- Counts the number of unique rectangles in the diagram -/
def count_unique_rectangles (d : Diagram) : Nat :=
  d.rectangles.length

/-- Theorem stating the total number of unique rectangles in the specific diagram -/
theorem total_rectangles_in_diagram :
  ∃ (d : Diagram),
    (∃ (r1 r2 r3 : Rectangle), r1 ∈ d.rectangles ∧ r2 ∈ d.rectangles ∧ r3 ∈ d.rectangles) ∧  -- 3 large rectangles
    (∃ (r4 r5 r6 r7 : Rectangle), r4 ∈ d.rectangles ∧ r5 ∈ d.rectangles ∧ r6 ∈ d.rectangles ∧ r7 ∈ d.rectangles) ∧  -- 4 small rectangles
    (∀ (r s : Rectangle), r ∈ d.rectangles → s ∈ d.rectangles → ∃ (t : Rectangle), t ∈ d.rectangles) →  -- Combination of rectangles
    count_unique_rectangles d = 11 :=
by
  sorry


end NUMINAMATH_CALUDE_total_rectangles_in_diagram_l3345_334524


namespace NUMINAMATH_CALUDE_milk_dilution_l3345_334565

/-- Proves that adding 15 liters of pure milk to 10 liters of milk with 5% water content
    results in a final water content of 2% -/
theorem milk_dilution (initial_milk : ℝ) (pure_milk : ℝ) (initial_water_percent : ℝ) :
  initial_milk = 10 →
  pure_milk = 15 →
  initial_water_percent = 5 →
  let total_milk := initial_milk + pure_milk
  let water_volume := initial_milk * (initial_water_percent / 100)
  let final_water_percent := (water_volume / total_milk) * 100
  final_water_percent = 2 := by
sorry

end NUMINAMATH_CALUDE_milk_dilution_l3345_334565


namespace NUMINAMATH_CALUDE_cos_five_pi_thirds_l3345_334551

theorem cos_five_pi_thirds : Real.cos (5 * π / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_five_pi_thirds_l3345_334551


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l3345_334579

/-- Given two employees with a total pay of 550 rupees, where one employee is paid 120% of the other,
    prove that the employee with lower pay receives 250 rupees. -/
theorem employee_pay_calculation (total_pay : ℝ) (x y : ℝ) : 
  total_pay = 550 →
  x = 1.2 * y →
  x + y = total_pay →
  y = 250 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l3345_334579


namespace NUMINAMATH_CALUDE_subset_iff_a_eq_one_l3345_334590

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_iff_a_eq_one (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_a_eq_one_l3345_334590


namespace NUMINAMATH_CALUDE_square_of_sum_17_5_l3345_334567

theorem square_of_sum_17_5 : 17^2 + 2*(17*5) + 5^2 = 484 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_17_5_l3345_334567


namespace NUMINAMATH_CALUDE_first_condition_second_condition_l3345_334580

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem for the first condition
theorem first_condition (a : ℝ) : 
  (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) → a = -2 := by sorry

-- Theorem for the second condition
theorem second_condition (a : ℝ) :
  (A a ∩ B = A a ∩ C) ∧ (A a ∩ B ≠ ∅) → a = -3 := by sorry

end NUMINAMATH_CALUDE_first_condition_second_condition_l3345_334580


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_m_part2_l3345_334560

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - 3| + |x + m|

-- Part 1: Solution set of f(x) ≥ 6 when m = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -2 ∨ x ≥ 4} :=
sorry

-- Part 2: Range of m when solution set of f(x) ≤ 5 is not empty
theorem range_of_m_part2 :
  (∃ x : ℝ, f m x ≤ 5) → m ∈ Set.Icc (-8) (-2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_m_part2_l3345_334560


namespace NUMINAMATH_CALUDE_train_speed_theorem_l3345_334529

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_theorem (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 120)
  (h2 : bridge_length = 255)
  (h3 : crossing_time = 30)
  : (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_theorem

end NUMINAMATH_CALUDE_train_speed_theorem_l3345_334529


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l3345_334589

def f (x : ℝ) := 6 - 12 * x + x^3

theorem f_max_min_on_interval :
  let a := -1/3
  let b := 1
  ∃ (x_max x_min : ℝ),
    x_max ∈ Set.Icc a b ∧
    x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    x_max = a ∧
    x_min = b ∧
    f x_max = 269/27 ∧
    f x_min = -5 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l3345_334589


namespace NUMINAMATH_CALUDE_frozen_food_storage_temp_l3345_334521

def standard_temp : ℝ := -18
def temp_range : ℝ := 2

def is_within_range (temp : ℝ) : Prop :=
  (standard_temp - temp_range) ≤ temp ∧ temp ≤ (standard_temp + temp_range)

theorem frozen_food_storage_temp :
  ¬(is_within_range (-21)) ∧
  is_within_range (-19) ∧
  is_within_range (-18) ∧
  is_within_range (-17) := by
sorry

end NUMINAMATH_CALUDE_frozen_food_storage_temp_l3345_334521


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3345_334535

/-- The time required for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) 
  (h1 : train_length = 100) 
  (h2 : train_speed_kmh = 45) 
  (h3 : bridge_length = 275) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3345_334535


namespace NUMINAMATH_CALUDE_ellipse_distance_sum_constant_l3345_334594

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with slope m passing through point P -/
structure Line where
  m : ℝ
  P : Point

theorem ellipse_distance_sum_constant
  (C : Ellipse)
  (h_ecc : C.a^2 - C.b^2 = (C.a / 2)^2) -- eccentricity is 1/2
  (h_chord : 2 * C.b^2 / C.a = 3) -- chord length condition
  (P : Point)
  (h_P_on_axis : P.y = 0 ∧ P.x^2 ≤ C.a^2) -- P is on the major axis
  (l : Line)
  (h_l_slope : l.m = C.b / C.a) -- line l has slope b/a
  (h_l_through_P : l.P = P) -- line l passes through P
  (A B : Point)
  (h_A_on_C : A.x^2 / C.a^2 + A.y^2 / C.b^2 = 1) -- A is on ellipse C
  (h_B_on_C : B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1) -- B is on ellipse C
  (h_A_on_l : A.y = l.m * (A.x - P.x)) -- A is on line l
  (h_B_on_l : B.y = l.m * (B.x - P.x)) -- B is on line l
  : (A.x - P.x)^2 + (A.y - P.y)^2 + (B.x - P.x)^2 + (B.y - P.y)^2 = C.a^2 + C.b^2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_distance_sum_constant_l3345_334594


namespace NUMINAMATH_CALUDE_product_terminal_zeros_l3345_334592

/-- The number of terminal zeros in a natural number -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The product of 75 and 480 -/
def product : ℕ := 75 * 480

/-- Theorem: The number of terminal zeros in the product of 75 and 480 is 3 -/
theorem product_terminal_zeros : terminalZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_terminal_zeros_l3345_334592


namespace NUMINAMATH_CALUDE_second_bus_ride_duration_l3345_334550

def first_bus_wait : ℕ := 12
def first_bus_ride : ℕ := 30

def total_first_bus_time : ℕ := first_bus_wait + first_bus_ride

def second_bus_time : ℕ := total_first_bus_time / 2

theorem second_bus_ride_duration : second_bus_time = 21 := by
  sorry

end NUMINAMATH_CALUDE_second_bus_ride_duration_l3345_334550


namespace NUMINAMATH_CALUDE_complex_number_problem_l3345_334584

theorem complex_number_problem (a : ℝ) (z : ℂ) (i : ℂ) : 
  a < 0 → 
  i^2 = -1 → 
  z = a * i / (1 - 2 * i) → 
  Complex.abs z = Real.sqrt 5 → 
  a = -5 := by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3345_334584


namespace NUMINAMATH_CALUDE_common_tangent_parabola_log_l3345_334568

theorem common_tangent_parabola_log (a s t : ℝ) : 
  a > 0 → 
  t = a * s^2 → 
  t = Real.log s → 
  (2 * a * s) = (1 / s) → 
  a = 1 / (2 * Real.exp 1) := by
sorry

end NUMINAMATH_CALUDE_common_tangent_parabola_log_l3345_334568


namespace NUMINAMATH_CALUDE_pizza_slices_l3345_334544

theorem pizza_slices (total_pizzas : ℕ) (total_slices : ℕ) (h1 : total_pizzas = 21) (h2 : total_slices = 168) :
  total_slices / total_pizzas = 8 := by
sorry

end NUMINAMATH_CALUDE_pizza_slices_l3345_334544


namespace NUMINAMATH_CALUDE_average_difference_l3345_334545

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 115)
  (h2 : (b + c) / 2 = 160) :
  a - c = -90 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l3345_334545


namespace NUMINAMATH_CALUDE_solve_system_l3345_334573

theorem solve_system (x y : ℝ) (h1 : 3 * x - 482 = 2 * y) (h2 : 7 * x + 517 = 5 * y) :
  x = 3444 ∧ y = 4925 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3345_334573


namespace NUMINAMATH_CALUDE_arithmetic_operations_l3345_334555

theorem arithmetic_operations : 
  (6 + (-8) - (-5) = 3) ∧ (18 / (-3) + (-2) * (-4) = 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l3345_334555


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3345_334511

/-- The quadratic function f(x) = (x-1)^2 - 2 -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

/-- The vertex of f(x) -/
def vertex : ℝ × ℝ := (1, -2)

theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3345_334511


namespace NUMINAMATH_CALUDE_factor_in_range_l3345_334500

theorem factor_in_range : ∃ (n : ℕ), 
  1210000 < n ∧ 
  n < 1220000 ∧ 
  1464101210001 % n = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_factor_in_range_l3345_334500


namespace NUMINAMATH_CALUDE_square_root_divided_by_15_l3345_334566

theorem square_root_divided_by_15 : Real.sqrt 3600 / 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_divided_by_15_l3345_334566


namespace NUMINAMATH_CALUDE_equal_positive_numbers_l3345_334575

theorem equal_positive_numbers (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : a^4 + b^4 + c^4 + d^4 = 4*a*b*c*d) : 
  a = b ∧ b = c ∧ c = d := by
sorry

end NUMINAMATH_CALUDE_equal_positive_numbers_l3345_334575


namespace NUMINAMATH_CALUDE_unique_positive_integer_l3345_334596

theorem unique_positive_integer : ∃! (n : ℕ), n > 0 ∧ 15 * n = n^2 + 56 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_l3345_334596


namespace NUMINAMATH_CALUDE_power_of_power_l3345_334534

theorem power_of_power (a : ℝ) : (a^3)^3 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3345_334534


namespace NUMINAMATH_CALUDE_circle_center_correct_l3345_334564

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, return its center -/
def findCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 1 4 1 (-6) (-20)
  findCenter eq = CircleCenter.mk (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_correct_l3345_334564


namespace NUMINAMATH_CALUDE_journey_possible_l3345_334572

/-- Represents a location along the route -/
structure Location :=
  (distance : ℝ)
  (from_quixajuba : Bool)

/-- Represents a person's state during the journey -/
structure PersonState :=
  (location : Location)
  (has_bicycle : Bool)

/-- Represents the state of the entire system at a given time -/
structure SystemState :=
  (time : ℝ)
  (person_a : PersonState)
  (person_b : PersonState)
  (person_c : PersonState)

/-- Defines the problem parameters -/
def problem_params : (ℝ × ℝ × ℝ) :=
  (24, 6, 18)  -- total_distance, walking_speed, biking_speed

/-- Defines a valid initial state -/
def initial_state : SystemState :=
  { time := 0,
    person_a := { location := { distance := 0, from_quixajuba := true }, has_bicycle := true },
    person_b := { location := { distance := 0, from_quixajuba := true }, has_bicycle := false },
    person_c := { location := { distance := 24, from_quixajuba := false }, has_bicycle := false } }

/-- Defines what it means for a system state to be valid -/
def is_valid_state (params : ℝ × ℝ × ℝ) (state : SystemState) : Prop :=
  let (total_distance, _, _) := params
  0 ≤ state.time ∧
  0 ≤ state.person_a.location.distance ∧ state.person_a.location.distance ≤ total_distance ∧
  0 ≤ state.person_b.location.distance ∧ state.person_b.location.distance ≤ total_distance ∧
  0 ≤ state.person_c.location.distance ∧ state.person_c.location.distance ≤ total_distance ∧
  (state.person_a.has_bicycle ∨ state.person_b.has_bicycle ∨ state.person_c.has_bicycle)

/-- Defines what it means for a system state to be a goal state -/
def is_goal_state (params : ℝ × ℝ × ℝ) (state : SystemState) : Prop :=
  let (total_distance, _, _) := params
  state.person_a.location.distance = total_distance ∧
  state.person_b.location.distance = total_distance ∧
  state.person_c.location.distance = 0 ∧
  state.time ≤ 160/60  -- 2 hours and 40 minutes in decimal hours

/-- The main theorem to be proved -/
theorem journey_possible (params : ℝ × ℝ × ℝ) (init : SystemState) :
  is_valid_state params init →
  ∃ (final : SystemState), is_valid_state params final ∧ is_goal_state params final :=
sorry

end NUMINAMATH_CALUDE_journey_possible_l3345_334572


namespace NUMINAMATH_CALUDE_fraction_zero_value_l3345_334576

theorem fraction_zero_value (x : ℝ) : 
  (x^2 - 4) / (x - 2) = 0 ∧ x - 2 ≠ 0 → x = -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_value_l3345_334576


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3345_334587

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, 7) and (4, -3) is 9 -/
theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (10, 7)
  let p₂ : ℝ × ℝ := (4, -3)
  let midpoint : ℝ × ℝ := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 9 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3345_334587


namespace NUMINAMATH_CALUDE_adult_meal_cost_l3345_334557

theorem adult_meal_cost 
  (total_people : ℕ) 
  (kids : ℕ) 
  (total_cost : ℚ) 
  (h1 : total_people = 11) 
  (h2 : kids = 2) 
  (h3 : total_cost = 72) : 
  (total_cost / (total_people - kids) : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_adult_meal_cost_l3345_334557


namespace NUMINAMATH_CALUDE_set_union_problem_l3345_334548

theorem set_union_problem (S T : Set ℕ) (h1 : S = {0, 1}) (h2 : T = {0}) :
  S ∪ T = {0, 1} := by sorry

end NUMINAMATH_CALUDE_set_union_problem_l3345_334548


namespace NUMINAMATH_CALUDE_five_people_arrangement_l3345_334538

/-- The number of ways to arrange n people in a row. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange two specific people next to each other in a row of n people. -/
def adjacentPairArrangements (n : ℕ) : ℕ := 2 * (n - 1)

/-- The number of ways to arrange 5 people in a row with two specific people next to each other. -/
theorem five_people_arrangement : 
  adjacentPairArrangements 5 * arrangements 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_five_people_arrangement_l3345_334538


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3345_334509

theorem no_integer_solutions (p₁ p₂ α n : ℕ) : 
  Prime p₁ → Prime p₂ → Odd p₁ → Odd p₂ → α > 1 → n > 1 →
  ¬ ∃ (α n : ℕ), ((p₂ - 1) / 2)^p₁ + ((p₂ + 1) / 2)^p₁ = α^n :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3345_334509


namespace NUMINAMATH_CALUDE_lines_parallel_iff_l3345_334514

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes are equal -/
def parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  A1 / B1 = A2 / B2

/-- The first line: ax + 2y + 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 1 = 0

/-- The second line: x + y + 4 = 0 -/
def line2 (x y : ℝ) : Prop :=
  x + y + 4 = 0

theorem lines_parallel_iff (a : ℝ) :
  parallel a 2 1 1 1 4 ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_l3345_334514
