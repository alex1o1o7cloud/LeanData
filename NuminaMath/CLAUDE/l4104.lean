import Mathlib

namespace complement_A_intersect_B_l4104_410447

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 ≤ 0}
def B : Set ℝ := {x | x > 5/2}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | x > 6} :=
sorry

end complement_A_intersect_B_l4104_410447


namespace complex_modulus_problem_l4104_410472

theorem complex_modulus_problem (z : ℂ) : 
  ((1 + Complex.I) / (1 - Complex.I)) * z = 3 + 4 * Complex.I → Complex.abs z = 5 := by
sorry

end complex_modulus_problem_l4104_410472


namespace game_winnable_iff_k_leq_n_minus_one_l4104_410456

/-- Represents the game state -/
structure GameState :=
  (k : ℕ)
  (n : ℕ)
  (h1 : 2 ≤ k)
  (h2 : k ≤ n)

/-- Predicate to determine if the game is winnable -/
def is_winnable (g : GameState) : Prop :=
  g.k ≤ g.n - 1

/-- Theorem stating the condition for the game to be winnable -/
theorem game_winnable_iff_k_leq_n_minus_one (g : GameState) :
  is_winnable g ↔ g.k ≤ g.n - 1 :=
sorry

end game_winnable_iff_k_leq_n_minus_one_l4104_410456


namespace fraction_sum_equality_l4104_410462

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (75 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 15 / (75 - b) + 11 / (55 - c) = 187 / 30 :=
by sorry

end fraction_sum_equality_l4104_410462


namespace slope_problem_l4104_410419

theorem slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = m) : m = (1 + Real.sqrt 17) / 2 := by
  sorry

end slope_problem_l4104_410419


namespace sara_pumpkins_l4104_410487

def pumpkins_grown : ℕ := 43
def pumpkins_eaten : ℕ := 23

theorem sara_pumpkins : pumpkins_grown - pumpkins_eaten = 20 := by
  sorry

end sara_pumpkins_l4104_410487


namespace room_width_is_four_meters_l4104_410480

/-- Proves that the width of a rectangular room is 4 meters given the specified conditions -/
theorem room_width_is_four_meters 
  (length : ℝ) 
  (cost_per_sqm : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : cost_per_sqm = 700)
  (h3 : total_cost = 15400) :
  ∃ (width : ℝ), width = 4 ∧ length * width * cost_per_sqm = total_cost :=
by sorry

end room_width_is_four_meters_l4104_410480


namespace wage_cut_and_raise_l4104_410494

theorem wage_cut_and_raise (original_wage : ℝ) (h : original_wage > 0) :
  let reduced_wage := 0.75 * original_wage
  let raise_percentage := 1 / 3
  reduced_wage * (1 + raise_percentage) = original_wage := by sorry

end wage_cut_and_raise_l4104_410494


namespace min_value_and_location_l4104_410402

theorem min_value_and_location (f : ℝ → ℝ) :
  (∀ x, f x = 2 * Real.sin x ^ 4 + 2 * Real.cos x ^ 4 + Real.cos (2 * x) ^ 2 - 3) →
  (∃ x_min ∈ Set.Icc (π / 16) (3 * π / 16), 
    (∀ x ∈ Set.Icc (π / 16) (3 * π / 16), f x_min ≤ f x) ∧
    x_min = 3 * π / 16 ∧
    f x_min = -(Real.sqrt 2 + 2) / 2) := by
  sorry

end min_value_and_location_l4104_410402


namespace tangent_cubic_to_line_l4104_410477

/-- Given that the graph of y = ax³ + 1 is tangent to the line y = x, prove that a = 4/27 -/
theorem tangent_cubic_to_line (a : ℝ) : 
  (∃ x : ℝ, x = a * x^3 + 1 ∧ 3 * a * x^2 = 1) → a = 4/27 := by
  sorry

end tangent_cubic_to_line_l4104_410477


namespace arithmetic_geometric_progression_l4104_410453

/-- 
Given an arithmetic progression where a₁₂, a₁₃, a₁₅ are the 12th, 13th, and 15th terms respectively,
and their squares form a geometric progression with common ratio q,
prove that q must be one of: 4, 4 - 2√3, 4 + 2√3, or 9/25.
-/
theorem arithmetic_geometric_progression (a₁₂ a₁₃ a₁₅ d q : ℝ) : 
  (a₁₃ = a₁₂ + d ∧ a₁₅ = a₁₃ + 2*d) →  -- arithmetic progression condition
  (a₁₃^2)^2 = a₁₂^2 * a₁₅^2 →  -- geometric progression condition
  (q = (a₁₃^2 / a₁₂^2)) →  -- definition of q
  (q = 4 ∨ q = 4 - 2*Real.sqrt 3 ∨ q = 4 + 2*Real.sqrt 3 ∨ q = 9/25) :=
by sorry

end arithmetic_geometric_progression_l4104_410453


namespace complex_square_one_minus_i_l4104_410496

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem complex_square_one_minus_i : (1 - i)^2 = -2*i := by sorry

end complex_square_one_minus_i_l4104_410496


namespace anthony_painting_time_l4104_410490

/-- The time it takes Kathleen and Anthony to paint two rooms together -/
def joint_time : ℝ := 3.428571428571429

/-- The time it takes Kathleen to paint one room -/
def kathleen_time : ℝ := 3

/-- Anthony's painting time for one room -/
def anthony_time : ℝ := 4

/-- Theorem stating that given Kathleen's painting time and their joint time for two rooms, 
    Anthony's individual painting time for one room is 4 hours -/
theorem anthony_painting_time : 
  (1 / kathleen_time + 1 / anthony_time) * joint_time = 2 :=
sorry

end anthony_painting_time_l4104_410490


namespace no_rational_roots_for_odd_m_n_l4104_410424

theorem no_rational_roots_for_odd_m_n (m n : ℤ) (hm : Odd m) (hn : Odd n) :
  ∀ x : ℚ, x^2 + 2*m*x + 2*n ≠ 0 := by
  sorry

end no_rational_roots_for_odd_m_n_l4104_410424


namespace prime_power_sum_product_l4104_410430

theorem prime_power_sum_product (p : ℕ) : 
  Prime p → 
  (∃ x y z : ℕ, ∃ q r s : ℕ, 
    Prime q ∧ Prime r ∧ Prime s ∧ 
    q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    x^p + y^p + z^p - x - y - z = q * r * s) ↔ 
  p = 2 ∨ p = 3 ∨ p = 5 := by
sorry

end prime_power_sum_product_l4104_410430


namespace percentage_calculation_l4104_410425

theorem percentage_calculation (x y z : ℝ) 
  (hx : 0.2 * x = 200)
  (hy : 0.3 * y = 150)
  (hz : 0.4 * z = 80) :
  (0.9 * x - 0.6 * y) + 0.5 * (x + y + z) = 1450 := by
  sorry

end percentage_calculation_l4104_410425


namespace man_swimming_speed_l4104_410446

/-- The speed of a man in still water, given his downstream and upstream swimming distances and times. -/
theorem man_swimming_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (h1 : downstream_distance = 51) 
  (h2 : upstream_distance = 18) 
  (h3 : time = 3) :
  ∃ (man_speed stream_speed : ℝ), 
    downstream_distance = (man_speed + stream_speed) * time ∧ 
    upstream_distance = (man_speed - stream_speed) * time ∧ 
    man_speed = 11.5 := by
sorry

end man_swimming_speed_l4104_410446


namespace team_formation_with_girls_l4104_410438

theorem team_formation_with_girls (total : Nat) (boys : Nat) (girls : Nat) (team_size : Nat) :
  total = boys + girls → boys = 5 → girls = 5 → team_size = 3 →
  (Nat.choose total team_size) - (Nat.choose boys team_size) = 110 := by
  sorry

end team_formation_with_girls_l4104_410438


namespace no_prime_solution_l4104_410454

theorem no_prime_solution : ¬ ∃ (p : ℕ), 
  Nat.Prime p ∧ 
  p > 8 ∧ 
  2 * p^3 + 7 * p^2 + 6 * p + 20 = 6 * p^2 + 19 * p + 10 := by
sorry

end no_prime_solution_l4104_410454


namespace fraction_equality_l4104_410489

theorem fraction_equality (p q : ℚ) : 
  11 / 7 + (2 * q - p) / (2 * q + p) = 2 → p / q = 4 / 5 := by
  sorry

end fraction_equality_l4104_410489


namespace a_investment_l4104_410415

theorem a_investment (b_investment c_investment total_profit a_profit_share : ℕ) 
  (hb : b_investment = 7200)
  (hc : c_investment = 9600)
  (hp : total_profit = 9000)
  (ha : a_profit_share = 1125) : 
  ∃ a_investment : ℕ, 
    a_investment = 2400 ∧ 
    a_profit_share * (a_investment + b_investment + c_investment) = a_investment * total_profit :=
by sorry

end a_investment_l4104_410415


namespace songs_ratio_l4104_410417

def initial_songs : ℕ := 54
def deleted_songs : ℕ := 9

theorem songs_ratio :
  let kept_songs := initial_songs - deleted_songs
  let ratio := kept_songs / deleted_songs
  ratio = 5 := by sorry

end songs_ratio_l4104_410417


namespace alpha_value_l4104_410451

theorem alpha_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1)
  (h_min : ∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 9/y ≥ 1/m + 9/n)
  (α : ℝ) (h_curve : m^α = 2/3 * n) : α = 1/2 := by
  sorry

end alpha_value_l4104_410451


namespace line_equation_correct_l4104_410404

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The equation 3x - y + 4 = 0 represents a line with slope 3 and y-intercept 4 -/
theorem line_equation_correct (l : Line) (eq : LineEquation) :
  l.slope = 3 ∧ l.intercept = 4 ∧ 
  eq.a = 3 ∧ eq.b = -1 ∧ eq.c = 4 →
  eq.a * x + eq.b * y + eq.c = 0 ↔ y = l.slope * x + l.intercept :=
by sorry

end line_equation_correct_l4104_410404


namespace sqrt_equation_solution_l4104_410439

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (y + 10) = 12 → y = 134 := by
  sorry

end sqrt_equation_solution_l4104_410439


namespace average_equation_holds_for_all_reals_solution_is_all_reals_l4104_410434

theorem average_equation_holds_for_all_reals (y : ℝ) : 
  ((2*y + 5) + (3*y + 4) + (7*y - 2)) / 3 = 4*y + 7/3 := by
  sorry

theorem solution_is_all_reals : 
  ∀ y : ℝ, ((2*y + 5) + (3*y + 4) + (7*y - 2)) / 3 = 4*y + 7/3 := by
  sorry

end average_equation_holds_for_all_reals_solution_is_all_reals_l4104_410434


namespace square_congruent_neg_one_mod_prime_l4104_410469

theorem square_congruent_neg_one_mod_prime (p : ℕ) (hp : Nat.Prime p) :
  (∃ k : ℤ, k^2 ≡ -1 [ZMOD p]) ↔ p = 2 ∨ p ≡ 1 [ZMOD 4] :=
sorry

end square_congruent_neg_one_mod_prime_l4104_410469


namespace exists_larger_area_figure_l4104_410411

/-- A convex figure in a 2D plane -/
structure ConvexFigure where
  -- We don't need to define the internal structure for this problem
  area : ℝ
  perimeter : ℝ

/-- A chord of a convex figure -/
structure Chord (F : ConvexFigure) where
  -- We don't need to define the internal structure for this problem
  dividesPerimeterInHalf : Bool
  dividesAreaUnequally : Bool

/-- Theorem: If a convex figure has a chord that divides its perimeter in half
    and its area unequally, then there exists another figure with the same
    perimeter but larger area -/
theorem exists_larger_area_figure (F : ConvexFigure) 
  (h : ∃ c : Chord F, c.dividesPerimeterInHalf ∧ c.dividesAreaUnequally) :
  ∃ G : ConvexFigure, G.perimeter = F.perimeter ∧ G.area > F.area :=
by sorry

end exists_larger_area_figure_l4104_410411


namespace scientific_notation_correct_l4104_410449

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 682000000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 6.82
    exponent := 8
    is_valid := by sorry }

/-- Theorem stating that the proposed scientific notation correctly represents the original number -/
theorem scientific_notation_correct :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = original_number := by sorry

end scientific_notation_correct_l4104_410449


namespace matrix_sum_theorem_l4104_410470

theorem matrix_sum_theorem (a b c : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![a^2, b^2, c^2; b^2, c^2, a^2; c^2, a^2, b^2]
  ¬(IsUnit (M.det)) →
  (a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) + c^2 / (a^2 + b^2) = 3/2) ∨
  (a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) + c^2 / (a^2 + b^2) = -3) :=
by sorry


end matrix_sum_theorem_l4104_410470


namespace geometric_harmonic_mean_inequality_l4104_410468

theorem geometric_harmonic_mean_inequality {a b : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt (a * b) ≥ 2 / (1 / a + 1 / b) ∧
  (Real.sqrt (a * b) = 2 / (1 / a + 1 / b) ↔ a = b) :=
by sorry

end geometric_harmonic_mean_inequality_l4104_410468


namespace sufficient_not_necessary_l4104_410445

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Predicate for a sequence being arithmetic -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Predicate for a point being on the line y = 2x + 1 -/
def on_line (n : ℕ) (a : Sequence) : Prop :=
  a n = 2 * n + 1

theorem sufficient_not_necessary :
  (∀ a : Sequence, (∀ n : ℕ, n > 0 → on_line n a) → is_arithmetic a) ∧
  (∃ a : Sequence, is_arithmetic a ∧ ∃ n : ℕ, n > 0 ∧ ¬on_line n a) :=
sorry

end sufficient_not_necessary_l4104_410445


namespace set_equality_l4104_410465

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {3, 4, 5}
def P : Set ℕ := {1, 3, 6}

theorem set_equality : (U \ M) ∩ (U \ P) = {2, 7, 8} := by sorry

end set_equality_l4104_410465


namespace tangent_slope_circle_l4104_410460

/-- Given a circle with center (2,3) and a point (8,7) on the circle,
    the slope of the line tangent to the circle at (8,7) is -3/2. -/
theorem tangent_slope_circle (center : ℝ × ℝ) (point : ℝ × ℝ) :
  center = (2, 3) →
  point = (8, 7) →
  (((point.2 - center.2) / (point.1 - center.1)) * (-1 / ((point.2 - center.2) / (point.1 - center.1)))) = -3/2 :=
by sorry

end tangent_slope_circle_l4104_410460


namespace sqrt_6_irrational_l4104_410416

theorem sqrt_6_irrational : Irrational (Real.sqrt 6) := by
  sorry

end sqrt_6_irrational_l4104_410416


namespace sqrt_sum_comparison_l4104_410431

theorem sqrt_sum_comparison : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

end sqrt_sum_comparison_l4104_410431


namespace sisyphus_earning_zero_l4104_410442

/-- Represents the state of the stone boxes and Sisyphus's earnings -/
structure BoxState where
  a : ℕ  -- number of stones in box A
  b : ℕ  -- number of stones in box B
  c : ℕ  -- number of stones in box C
  x : ℤ  -- Sisyphus's earnings (can be negative)

/-- Represents a move of a stone from one box to another -/
inductive Move
  | AB : Move  -- from A to B
  | AC : Move  -- from A to C
  | BA : Move  -- from B to A
  | BC : Move  -- from B to C
  | CA : Move  -- from C to A
  | CB : Move  -- from C to B

/-- Applies a move to a BoxState -/
def applyMove (state : BoxState) (move : Move) : BoxState :=
  match move with
  | Move.AB => { state with 
      a := state.a - 1, 
      b := state.b + 1, 
      x := state.x + (state.a - state.b - 1) }
  | Move.AC => { state with 
      a := state.a - 1, 
      c := state.c + 1, 
      x := state.x + (state.a - state.c - 1) }
  | Move.BA => { state with 
      b := state.b - 1, 
      a := state.a + 1, 
      x := state.x + (state.b - state.a - 1) }
  | Move.BC => { state with 
      b := state.b - 1, 
      c := state.c + 1, 
      x := state.x + (state.b - state.c - 1) }
  | Move.CA => { state with 
      c := state.c - 1, 
      a := state.a + 1, 
      x := state.x + (state.c - state.a - 1) }
  | Move.CB => { state with 
      c := state.c - 1, 
      b := state.b + 1, 
      x := state.x + (state.c - state.b - 1) }

/-- Theorem: The greatest possible earning of Sisyphus is 0 -/
theorem sisyphus_earning_zero 
  (initial : BoxState) 
  (moves : List Move) 
  (h1 : moves.length = 24 * 365 * 1000) -- 1000 years of hourly moves
  (h2 : (moves.foldl applyMove initial).a = initial.a) -- stones return to initial state
  (h3 : (moves.foldl applyMove initial).b = initial.b)
  (h4 : (moves.foldl applyMove initial).c = initial.c) :
  (moves.foldl applyMove initial).x ≤ 0 :=
sorry

#check sisyphus_earning_zero

end sisyphus_earning_zero_l4104_410442


namespace asha_win_probability_l4104_410433

theorem asha_win_probability (lose_prob : ℚ) (h1 : lose_prob = 4/9) :
  1 - lose_prob = 5/9 := by
  sorry

end asha_win_probability_l4104_410433


namespace premium_rate_calculation_l4104_410400

theorem premium_rate_calculation (total_investment dividend_rate share_face_value total_dividend : ℚ)
  (h1 : total_investment = 14400)
  (h2 : dividend_rate = 5 / 100)
  (h3 : share_face_value = 100)
  (h4 : total_dividend = 576) :
  ∃ premium_rate : ℚ,
    premium_rate = 25 ∧
    total_dividend = dividend_rate * share_face_value * (total_investment / (share_face_value + premium_rate)) :=
by sorry


end premium_rate_calculation_l4104_410400


namespace system_one_solution_system_two_solution_l4104_410428

-- System 1
theorem system_one_solution (x y : ℝ) : 
  2 * x - y = 5 ∧ 7 * x - 3 * y = 20 → x = 5 ∧ y = 5 := by
  sorry

-- System 2
theorem system_two_solution (x y : ℝ) :
  3 * (x + y) - 4 * (x - y) = 16 ∧ (x + y) / 2 + (x - y) / 6 = 1 → 
  x = 1/3 ∧ y = 7/3 := by
  sorry

end system_one_solution_system_two_solution_l4104_410428


namespace find_divisor_l4104_410413

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) : 
  dividend = quotient * divisor + remainder → 
  dividend = 22 → 
  quotient = 7 → 
  remainder = 1 → 
  divisor = 3 := by
sorry

end find_divisor_l4104_410413


namespace arithmetic_sequence_common_ratio_l4104_410429

/-- An arithmetic sequence {a_n} with a common ratio q -/
def ArithmeticSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem arithmetic_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_increasing : ∀ n, a (n + 1) > a n)
  (h_positive : a 1 > 0)
  (h_condition : ∀ n, 2 * (a (n + 2) - a n) = 3 * a (n + 1))
  (h_arithmetic : ArithmeticSequence a q) :
  q = 2 :=
sorry

end arithmetic_sequence_common_ratio_l4104_410429


namespace probability_all_genuine_proof_l4104_410471

/-- The total number of coins -/
def total_coins : ℕ := 15

/-- The number of genuine coins -/
def genuine_coins : ℕ := 12

/-- The number of counterfeit coins -/
def counterfeit_coins : ℕ := 3

/-- The number of pairs selected -/
def pairs_selected : ℕ := 3

/-- The number of coins in each pair -/
def coins_per_pair : ℕ := 2

/-- Predicate that the weight of counterfeit coins is different from genuine coins -/
axiom counterfeit_weight_different : True

/-- Predicate that the combined weight of all three pairs is the same -/
axiom combined_weight_same : True

/-- The probability of selecting all genuine coins given the conditions -/
def probability_all_genuine : ℚ := 264 / 443

/-- Theorem stating that the probability of selecting all genuine coins
    given the conditions is equal to 264/443 -/
theorem probability_all_genuine_proof :
  probability_all_genuine = 264 / 443 :=
by sorry

end probability_all_genuine_proof_l4104_410471


namespace intersection_equality_implies_range_l4104_410493

theorem intersection_equality_implies_range (a : ℝ) : 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) ↔ (1 ≤ x ∧ x ≤ 2 ∧ 2 - a ≤ x ∧ x ≤ 1 + a)) →
  a ≥ 1 := by
sorry

end intersection_equality_implies_range_l4104_410493


namespace special_function_is_even_l4104_410495

/-- A function satisfying the given functional equation -/
structure SpecialFunction (f : ℝ → ℝ) : Prop where
  functional_eq : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y
  nonzero_at_zero : f 0 ≠ 0

/-- The main theorem: if f is a SpecialFunction, then it is even -/
theorem special_function_is_even (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∀ x : ℝ, f (-x) = f x := by
  sorry

end special_function_is_even_l4104_410495


namespace triangle_ratio_theorem_l4104_410461

-- Define the triangle PQR
def Triangle (P Q R : ℝ) : Prop := 
  0 < P ∧ 0 < Q ∧ 0 < R ∧ P + Q > R ∧ P + R > Q ∧ Q + R > P

-- Define the sides of the triangle
def PQ : ℝ := 8
def PR : ℝ := 7
def QR : ℝ := 5

-- State the theorem
theorem triangle_ratio_theorem (P Q R : ℝ) 
  (h : Triangle P Q R) 
  (h_pq : PQ = 8) 
  (h_pr : PR = 7) 
  (h_qr : QR = 5) : 
  (Real.cos ((P - Q) / 2) / Real.sin (R / 2)) - 
  (Real.sin ((P - Q) / 2) / Real.cos (R / 2)) = 5 / 7 := by
  sorry

end triangle_ratio_theorem_l4104_410461


namespace unique_real_solution_l4104_410405

theorem unique_real_solution :
  ∃! x : ℝ, x + Real.sqrt (x - 2) = 4 := by sorry

end unique_real_solution_l4104_410405


namespace sum_of_digits_n_l4104_410473

/-- The least 7-digit number that leaves a remainder of 4 when divided by 5, 850, 35, 27, and 90 -/
def n : ℕ := sorry

/-- Condition: n is a 7-digit number -/
axiom n_seven_digits : 1000000 ≤ n ∧ n < 10000000

/-- Condition: n leaves a remainder of 4 when divided by 5 -/
axiom n_mod_5 : n % 5 = 4

/-- Condition: n leaves a remainder of 4 when divided by 850 -/
axiom n_mod_850 : n % 850 = 4

/-- Condition: n leaves a remainder of 4 when divided by 35 -/
axiom n_mod_35 : n % 35 = 4

/-- Condition: n leaves a remainder of 4 when divided by 27 -/
axiom n_mod_27 : n % 27 = 4

/-- Condition: n leaves a remainder of 4 when divided by 90 -/
axiom n_mod_90 : n % 90 = 4

/-- Condition: n is the least number satisfying all the above conditions -/
axiom n_least : ∀ m : ℕ, (1000000 ≤ m ∧ m < 10000000 ∧ 
                          m % 5 = 4 ∧ m % 850 = 4 ∧ m % 35 = 4 ∧ m % 27 = 4 ∧ m % 90 = 4) → n ≤ m

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (k : ℕ) : ℕ := sorry

/-- Theorem: The sum of the digits of n is 22 -/
theorem sum_of_digits_n : sum_of_digits n = 22 := by sorry

end sum_of_digits_n_l4104_410473


namespace garden_perimeter_l4104_410414

theorem garden_perimeter : ∀ w l : ℝ,
  w > 0 →
  l > 0 →
  w * l = 200 →
  w^2 + l^2 = 30^2 →
  l = w + 4 →
  2 * (w + l) = 84 :=
by
  sorry

end garden_perimeter_l4104_410414


namespace find_certain_number_l4104_410488

theorem find_certain_number : ∃ x : ℝ,
  (20 + 40 + 60) / 3 = ((10 + x + 16) / 3 + 8) ∧ x = 70 :=
by
  sorry

end find_certain_number_l4104_410488


namespace can_meet_in_three_jumps_l4104_410484

-- Define the grid
def Grid := ℤ × ℤ

-- Define the color of a square
inductive Color
| Red
| White

-- Define the coloring function
def coloring : Grid → Color := sorry

-- Define the grasshopper's position
def grasshopper : Grid := sorry

-- Define the flea's position
def flea : Grid := sorry

-- Define a valid jump
def valid_jump (start finish : Grid) (c : Color) : Prop :=
  (coloring start = c ∧ coloring finish = c) ∧
  (start.1 = finish.1 ∨ start.2 = finish.2)

-- Define adjacent squares
def adjacent (a b : Grid) : Prop :=
  (abs (a.1 - b.1) + abs (a.2 - b.2) = 1)

-- Main theorem
theorem can_meet_in_three_jumps :
  ∃ (g1 g2 g3 f1 f2 f3 : Grid),
    (valid_jump grasshopper g1 Color.Red ∨ g1 = grasshopper) ∧
    (valid_jump g1 g2 Color.Red ∨ g2 = g1) ∧
    (valid_jump g2 g3 Color.Red ∨ g3 = g2) ∧
    (valid_jump flea f1 Color.White ∨ f1 = flea) ∧
    (valid_jump f1 f2 Color.White ∨ f2 = f1) ∧
    (valid_jump f2 f3 Color.White ∨ f3 = f2) ∧
    adjacent g3 f3 :=
  sorry


end can_meet_in_three_jumps_l4104_410484


namespace linear_function_decreasing_values_l4104_410418

theorem linear_function_decreasing_values (x₁ : ℝ) : 
  let f := fun (x : ℝ) => -3 * x + 1
  let y₁ := f x₁
  let y₂ := f (x₁ + 1)
  let y₃ := f (x₁ + 2)
  y₃ < y₂ ∧ y₂ < y₁ := by
sorry

end linear_function_decreasing_values_l4104_410418


namespace amount_equals_scientific_notation_l4104_410437

/-- Represents the amount in yuan -/
def amount : ℝ := 2.51e6

/-- Represents the scientific notation of the amount -/
def scientific_notation : ℝ := 2.51 * (10 ^ 6)

/-- Theorem stating that the amount is equal to its scientific notation representation -/
theorem amount_equals_scientific_notation : amount = scientific_notation := by
  sorry

end amount_equals_scientific_notation_l4104_410437


namespace fraction_simplification_l4104_410407

theorem fraction_simplification :
  (2 - 4 + 8 - 16 + 32 - 64 + 128 - 256) / (4 - 8 + 16 - 32 + 64 - 128 + 256 - 512) = 1 / 2 := by
  sorry

end fraction_simplification_l4104_410407


namespace field_length_width_difference_l4104_410406

/-- Proves that for a rectangular field with length 24 meters and width 13.5 meters,
    the difference between twice the width and the length is 3 meters. -/
theorem field_length_width_difference :
  let length : ℝ := 24
  let width : ℝ := 13.5
  2 * width - length = 3 := by sorry

end field_length_width_difference_l4104_410406


namespace dog_tricks_conversion_l4104_410499

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

theorem dog_tricks_conversion :
  base9ToBase10 [1, 2, 5] = 424 := by
  sorry

end dog_tricks_conversion_l4104_410499


namespace inequality_proof_l4104_410422

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a * b / (c - 1)) + (b * c / (a - 1)) + (c * a / (b - 1)) ≥ 12 ∧
  ((a * b / (c - 1)) + (b * c / (a - 1)) + (c * a / (b - 1)) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

#check inequality_proof

end inequality_proof_l4104_410422


namespace tangent_product_equals_two_l4104_410483

theorem tangent_product_equals_two (x y : Real) 
  (h1 : x = 21 * π / 180) 
  (h2 : y = 24 * π / 180) 
  (h3 : Real.tan (π / 4) = 1) 
  (h4 : π / 4 = x + y) : 
  (1 + Real.tan x) * (1 + Real.tan y) = 2 := by
  sorry

end tangent_product_equals_two_l4104_410483


namespace cats_sold_proof_l4104_410467

/-- Calculates the number of cats sold during a sale at a pet store. -/
def cats_sold (siamese : ℕ) (house : ℕ) (left : ℕ) : ℕ :=
  siamese + house - left

/-- Proves that the number of cats sold during the sale is 45. -/
theorem cats_sold_proof :
  cats_sold 38 25 18 = 45 := by
  sorry

end cats_sold_proof_l4104_410467


namespace cerulean_somewhat_green_l4104_410435

/-- The number of people surveyed -/
def total_surveyed : ℕ := 120

/-- The number of people who think cerulean is "kind of blue" -/
def kind_of_blue : ℕ := 80

/-- The number of people who think cerulean is both "kind of blue" and "somewhat green" -/
def both : ℕ := 35

/-- The number of people who think cerulean is neither "kind of blue" nor "somewhat green" -/
def neither : ℕ := 20

/-- The theorem states that the number of people who believe cerulean is "somewhat green" is 55 -/
theorem cerulean_somewhat_green : 
  total_surveyed - kind_of_blue + both = 55 :=
by sorry

end cerulean_somewhat_green_l4104_410435


namespace problem_solution_l4104_410452

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
  sorry

end problem_solution_l4104_410452


namespace frank_can_buy_seven_candies_l4104_410475

-- Define the given conditions
def whack_a_mole_tickets : ℕ := 33
def skee_ball_tickets : ℕ := 9
def candy_cost : ℕ := 6

-- Define the total number of tickets
def total_tickets : ℕ := whack_a_mole_tickets + skee_ball_tickets

-- Define the number of candies Frank can buy
def candies_bought : ℕ := total_tickets / candy_cost

-- Theorem statement
theorem frank_can_buy_seven_candies : candies_bought = 7 := by
  sorry

end frank_can_buy_seven_candies_l4104_410475


namespace sanchez_sum_problem_l4104_410463

theorem sanchez_sum_problem (x y : ℕ+) : x - y = 5 → x * y = 84 → x + y = 19 := by
  sorry

end sanchez_sum_problem_l4104_410463


namespace square_completion_l4104_410497

theorem square_completion (a h k : ℝ) : 
  (∀ x, x^2 - 6*x = a*(x - h)^2 + k) → k = -9 := by
  sorry

end square_completion_l4104_410497


namespace total_production_proof_l4104_410478

def week1_production : ℕ := 320
def week2_production : ℕ := 400
def week3_production : ℕ := 300
def increase_percentage : ℚ := 20 / 100

theorem total_production_proof :
  let average := (week1_production + week2_production + week3_production) / 3
  let week4_production := average + (average * increase_percentage).floor
  week1_production + week2_production + week3_production + week4_production = 1428 := by
  sorry

end total_production_proof_l4104_410478


namespace triangle_property_l4104_410421

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : (3 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C)
  (h2 : 1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 2) : 
  Real.cos t.A = 1/3 ∧ 
  ∃ (p : ℝ), p = t.a + t.b + t.c ∧ 
  p ≥ 2 * Real.sqrt 6 + 2 * Real.sqrt 2 ∧
  (p = 2 * Real.sqrt 6 + 2 * Real.sqrt 2 ↔ t.a = t.b) := by
  sorry

end triangle_property_l4104_410421


namespace smallest_divisor_with_remainder_l4104_410476

theorem smallest_divisor_with_remainder (d : ℕ) : d = 6 ↔ 
  d > 1 ∧ 
  (∀ n : ℤ, n % d = 1 → (5 * n) % d = 5) ∧
  (∀ d' : ℕ, d' < d → d' > 1 → ∃ n : ℤ, n % d' = 1 ∧ (5 * n) % d' ≠ 5) :=
by sorry

#check smallest_divisor_with_remainder

end smallest_divisor_with_remainder_l4104_410476


namespace solution_set_nonempty_l4104_410459

theorem solution_set_nonempty (a : ℝ) : 
  ∃ x : ℝ, a * x^2 - (a - 2) * x - 2 ≤ 0 := by
  sorry

end solution_set_nonempty_l4104_410459


namespace gamma_value_l4104_410482

theorem gamma_value (γ δ : ℂ) 
  (h1 : (γ + δ).re > 0)
  (h2 : (Complex.I * (γ - δ)).re > 0)
  (h3 : δ = 2 + 3 * Complex.I) : 
  γ = 2 - 3 * Complex.I := by
sorry

end gamma_value_l4104_410482


namespace inequality_proof_l4104_410455

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 ∧
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry


end inequality_proof_l4104_410455


namespace ascending_order_abc_l4104_410466

theorem ascending_order_abc :
  let a := Real.sin (17 * π / 180) * Real.cos (45 * π / 180) + Real.cos (17 * π / 180) * Real.sin (45 * π / 180)
  let b := 2 * (Real.cos (13 * π / 180))^2 - 1
  let c := Real.sqrt 3 / 2
  c < a ∧ a < b := by
  sorry

end ascending_order_abc_l4104_410466


namespace set_membership_equivalence_l4104_410420

theorem set_membership_equivalence (C M N : Set α) (x : α) :
  x ∈ C ∪ (M ∩ N) ↔ (x ∈ C ∪ M ∨ x ∈ C ∪ N) := by sorry

end set_membership_equivalence_l4104_410420


namespace floor_ceil_sqrt_50_sum_squares_l4104_410481

theorem floor_ceil_sqrt_50_sum_squares : ⌊Real.sqrt 50⌋^2 + ⌈Real.sqrt 50⌉^2 = 113 := by
  sorry

end floor_ceil_sqrt_50_sum_squares_l4104_410481


namespace polynomial_division_result_l4104_410448

theorem polynomial_division_result (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (25 * x^2 * y - 5 * x * y^2) / (5 * x * y) = 5 * x - y := by
  sorry

end polynomial_division_result_l4104_410448


namespace georges_car_cylinders_l4104_410426

def oil_per_cylinder : ℕ := 8
def oil_already_added : ℕ := 16
def additional_oil_needed : ℕ := 32

theorem georges_car_cylinders :
  (oil_already_added + additional_oil_needed) / oil_per_cylinder = 6 :=
by sorry

end georges_car_cylinders_l4104_410426


namespace factor_expression_value_l4104_410401

theorem factor_expression_value (k m n : ℕ) : 
  k > 1 → m > 1 → n > 1 →
  (∃ (Z : ℕ), Z = 2^k * 3^m * 5^n ∧ (2^60 * 3^35 * 5^20 * 7^7) % Z = 0) →
  ∃ (k' m' n' : ℕ), k' > 1 ∧ m' > 1 ∧ n' > 1 ∧
    2^k' + 3^m' + k'^3 * m'^n' - n' = 43 :=
by sorry

end factor_expression_value_l4104_410401


namespace sqrt_14_plus_2_bounds_l4104_410436

theorem sqrt_14_plus_2_bounds : 5 < Real.sqrt 14 + 2 ∧ Real.sqrt 14 + 2 < 6 := by
  sorry

end sqrt_14_plus_2_bounds_l4104_410436


namespace grid_rectangle_division_l4104_410486

/-- A grid rectangle with cell side length 1 cm and area 2021 cm² -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  area_eq : width * height = 2021

/-- A cut configuration for the grid rectangle -/
structure CutConfig where
  hor_cut : ℕ
  ver_cut : ℕ

/-- The four parts resulting from a cut configuration -/
def parts (rect : GridRectangle) (cut : CutConfig) : Fin 4 → ℕ
| ⟨0, _⟩ => cut.hor_cut * cut.ver_cut
| ⟨1, _⟩ => cut.hor_cut * (rect.width - cut.ver_cut)
| ⟨2, _⟩ => (rect.height - cut.hor_cut) * cut.ver_cut
| ⟨3, _⟩ => (rect.height - cut.hor_cut) * (rect.width - cut.ver_cut)
| _ => 0

/-- The theorem to be proved -/
theorem grid_rectangle_division (rect : GridRectangle) :
  ∀ (cut : CutConfig), cut.hor_cut < rect.height → cut.ver_cut < rect.width →
  ∃ (i : Fin 4), parts rect cut i ≥ 528 := by
  sorry

end grid_rectangle_division_l4104_410486


namespace square_perimeter_ratio_l4104_410432

theorem square_perimeter_ratio (a b : ℝ) (h : a^2 / b^2 = 49 / 64) :
  (4 * a) / (4 * b) = 7 / 8 := by
  sorry

end square_perimeter_ratio_l4104_410432


namespace area_of_triangle_DEF_l4104_410408

-- Define the square PQRS
def square_PQRS : Real := 36

-- Define the side length of the smaller squares
def small_square_side : Real := 2

-- Define the triangle DEF
structure Triangle_DEF where
  DE : Real
  EF : Real
  isIsosceles : DE = DF

-- Define the folding property
def folding_property (t : Triangle_DEF) : Prop :=
  ∃ (center : Real), 
    center = (square_PQRS.sqrt / 2) ∧
    t.DE = center + 2 * small_square_side

-- Theorem statement
theorem area_of_triangle_DEF : 
  ∀ (t : Triangle_DEF), 
    folding_property t → 
    (1/2 : Real) * t.EF * t.DE = 7 := by
  sorry

end area_of_triangle_DEF_l4104_410408


namespace hyperbola_eccentricity_l4104_410479

theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b)
  (h4 : (a + b) / 2 = 7 / 2) (h5 : Real.sqrt (a * b) = 2 * Real.sqrt 3) :
  let c := Real.sqrt (a^2 + b^2)
  Real.sqrt (c^2 - b^2) / b = 5 / 3 :=
by sorry

end hyperbola_eccentricity_l4104_410479


namespace luke_sticker_count_l4104_410423

/-- Represents the number of stickers Luke has at different stages -/
structure StickerCount where
  initial : ℕ
  afterBuying : ℕ
  afterGift : ℕ
  afterGiving : ℕ
  afterUsing : ℕ
  final : ℕ

/-- Theorem stating the relationship between Luke's initial and final sticker counts -/
theorem luke_sticker_count (s : StickerCount) 
  (hbuy : s.afterBuying = s.initial + 12)
  (hgift : s.afterGift = s.afterBuying + 20)
  (hgive : s.afterGiving = s.afterGift - 5)
  (huse : s.afterUsing = s.afterGiving - 8)
  (hfinal : s.final = s.afterUsing)
  (h_final_count : s.final = 39) :
  s.initial = 20 := by
  sorry

end luke_sticker_count_l4104_410423


namespace kenya_has_133_peanuts_l4104_410458

-- Define the number of peanuts Jose has
def jose_peanuts : ℕ := 85

-- Define the difference in peanuts between Kenya and Jose
def peanut_difference : ℕ := 48

-- Define Kenya's peanuts in terms of Jose's peanuts and the difference
def kenya_peanuts : ℕ := jose_peanuts + peanut_difference

-- Theorem stating that Kenya has 133 peanuts
theorem kenya_has_133_peanuts : kenya_peanuts = 133 := by
  sorry

end kenya_has_133_peanuts_l4104_410458


namespace geometric_sequence_b_value_l4104_410444

theorem geometric_sequence_b_value (b : ℝ) (h1 : b > 0) 
  (h2 : ∃ r : ℝ, r > 0 ∧ b = 30 * r ∧ 9/4 = b * r) : 
  b = 3 * Real.sqrt 30 / 2 := by
  sorry

end geometric_sequence_b_value_l4104_410444


namespace road_length_calculation_l4104_410491

/-- Given a map scale and a road length on the map, calculate the actual road length in kilometers. -/
def actual_road_length (map_scale : ℚ) (map_length : ℚ) : ℚ :=
  map_length * map_scale / 100000

/-- Theorem stating that for a map scale of 1:2500000 and a road length of 6 cm on the map,
    the actual length of the road is 150 km. -/
theorem road_length_calculation :
  let map_scale : ℚ := 2500000
  let map_length : ℚ := 6
  actual_road_length map_scale map_length = 150 := by
  sorry

#eval actual_road_length 2500000 6

end road_length_calculation_l4104_410491


namespace additional_marbles_for_lisa_l4104_410498

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed -/
theorem additional_marbles_for_lisa : 
  min_additional_marbles 12 40 = 38 := by
  sorry

end additional_marbles_for_lisa_l4104_410498


namespace dany_farm_cows_l4104_410409

/-- Represents the number of cows on Dany's farm -/
def num_cows : ℕ := 4

/-- Represents the number of sheep on Dany's farm -/
def num_sheep : ℕ := 3

/-- Represents the number of chickens on Dany's farm -/
def num_chickens : ℕ := 7

/-- Represents the number of bushels a sheep eats per day -/
def sheep_bushels : ℕ := 2

/-- Represents the number of bushels a chicken eats per day -/
def chicken_bushels : ℕ := 3

/-- Represents the number of bushels a cow eats per day -/
def cow_bushels : ℕ := 2

/-- Represents the total number of bushels needed for all animals per day -/
def total_bushels : ℕ := 35

theorem dany_farm_cows :
  num_cows * cow_bushels + num_sheep * sheep_bushels + num_chickens * chicken_bushels = total_bushels :=
by sorry

end dany_farm_cows_l4104_410409


namespace mountain_trail_length_l4104_410403

/-- Represents the hike of Phoenix on the Mountain Trail --/
structure MountainTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of Phoenix's hike --/
def HikeConditions (hike : MountainTrail) : Prop :=
  hike.day1 + hike.day2 = 28 ∧
  (hike.day2 + hike.day3) / 2 = 15 ∧
  hike.day4 + hike.day5 = 34 ∧
  hike.day1 + hike.day3 = 32

/-- Theorem: The total length of the Mountain Trail is 94 miles --/
theorem mountain_trail_length (hike : MountainTrail) 
  (h : HikeConditions hike) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 94 := by
  sorry


end mountain_trail_length_l4104_410403


namespace inscribed_square_pyramid_dimensions_l4104_410427

/-- Regular pentagonal pyramid with square pyramid inscribed -/
structure PentagonalPyramidWithInscribedSquare where
  a : ℝ  -- side length of pentagonal base
  e : ℝ  -- height of pentagonal pyramid
  x : ℝ  -- side length of inscribed square base

/-- Theorem about the dimensions of the inscribed square pyramid -/
theorem inscribed_square_pyramid_dimensions
  (P : PentagonalPyramidWithInscribedSquare)
  (h_a_pos : P.a > 0)
  (h_e_pos : P.e > 0) :
  P.x = P.a / (2 * Real.sin (18 * π / 180) + Real.tan (18 * π / 180)) ∧
  ∃ (SR₁ SR₃ : ℝ),
    SR₁^2 = (P.a * Real.cos (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)))^2 +
            P.e^2 - P.a^2 * Real.cos (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)) ∧
    SR₃^2 = (P.a * Real.sin (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)))^2 +
            P.e^2 - P.a^2 * Real.sin (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)) :=
by sorry

end inscribed_square_pyramid_dimensions_l4104_410427


namespace P_roots_l4104_410450

def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | n + 1, x => x^(5 * (n + 1)) - P n x

theorem P_roots (n : ℕ) :
  (n % 2 = 1 → P n 1 = 0 ∧ ∀ x : ℝ, x ≠ 1 → P n x ≠ 0) ∧
  (n % 2 = 0 → ∀ x : ℝ, P n x ≠ 0) :=
by sorry

end P_roots_l4104_410450


namespace min_hours_is_eight_l4104_410457

/-- Represents Biff's expenses and earnings during the bus trip -/
structure BusTrip where
  ticket : ℕ
  snacks : ℕ
  headphones : ℕ
  lunch : ℕ
  dinner : ℕ
  accommodation : ℕ
  hourly_rate : ℕ
  day_wifi_rate : ℕ
  night_wifi_rate : ℕ

/-- Calculates the total fixed expenses for the trip -/
def total_fixed_expenses (trip : BusTrip) : ℕ :=
  trip.ticket + trip.snacks + trip.headphones + trip.lunch + trip.dinner + trip.accommodation

/-- Calculates the minimum number of hours needed to break even -/
def min_hours_to_break_even (trip : BusTrip) : ℕ :=
  (total_fixed_expenses trip + trip.night_wifi_rate - 1) / (trip.hourly_rate - trip.night_wifi_rate) + 1

/-- Theorem stating that the minimum number of hours to break even is 8 -/
theorem min_hours_is_eight (trip : BusTrip)
  (h1 : trip.ticket = 11)
  (h2 : trip.snacks = 3)
  (h3 : trip.headphones = 16)
  (h4 : trip.lunch = 8)
  (h5 : trip.dinner = 10)
  (h6 : trip.accommodation = 35)
  (h7 : trip.hourly_rate = 12)
  (h8 : trip.day_wifi_rate = 2)
  (h9 : trip.night_wifi_rate = 1) :
  min_hours_to_break_even trip = 8 := by
  sorry

#eval min_hours_to_break_even {
  ticket := 11,
  snacks := 3,
  headphones := 16,
  lunch := 8,
  dinner := 10,
  accommodation := 35,
  hourly_rate := 12,
  day_wifi_rate := 2,
  night_wifi_rate := 1
}

end min_hours_is_eight_l4104_410457


namespace initial_men_correct_l4104_410464

/-- The number of days it takes to dig the entire tunnel with the initial workforce -/
def initial_days : ℝ := 30

/-- The number of days worked before adding more men -/
def days_before_addition : ℝ := 10

/-- The number of additional men added to the workforce -/
def additional_men : ℕ := 20

/-- The number of days it takes to complete the tunnel after adding more men -/
def remaining_days : ℝ := 10.000000000000002

/-- The initial number of men digging the tunnel -/
def initial_men : ℕ := 6

/-- Theorem stating that the initial number of men is correct given the conditions -/
theorem initial_men_correct :
  (initial_men : ℝ) * initial_days =
    (initial_men + additional_men) * remaining_days * (2/3) :=
by sorry

end initial_men_correct_l4104_410464


namespace function_types_l4104_410441

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2
def p (x : ℝ) (a : ℝ) : ℝ := a * x^2

-- State the theorem
theorem function_types (a x : ℝ) (ha : a ≠ 0) (hx : x ≠ 0) :
  (∃ b c : ℝ, ∀ x, f a x = x^2 + b*x + c) ∧
  (∃ m b : ℝ, ∀ a, p x a = m*a + b) :=
sorry

end function_types_l4104_410441


namespace regular_polygon_exterior_angle_l4104_410492

theorem regular_polygon_exterior_angle (n : ℕ) (h : n > 2) :
  (360 : ℝ) / n = 60 → n = 6 :=
by
  sorry

end regular_polygon_exterior_angle_l4104_410492


namespace survey_result_l4104_410412

/-- Represents the survey results and conditions -/
structure SurveyData where
  total_students : Nat
  yes_responses : Nat
  id_range : Nat × Nat

/-- Calculates the expected number of students who have cheated -/
def expected_cheaters (data : SurveyData) : Nat :=
  let odd_ids := (data.id_range.2 - data.id_range.1 + 1) / 2
  let expected_yes_to_odd := odd_ids / 2
  let expected_cheaters_half := data.yes_responses - expected_yes_to_odd
  2 * expected_cheaters_half

/-- Theorem stating the expected number of cheaters based on the survey data -/
theorem survey_result (data : SurveyData) 
  (h1 : data.total_students = 2000)
  (h2 : data.yes_responses = 510)
  (h3 : data.id_range = (1, 2000)) :
  expected_cheaters data = 20 := by
  sorry


end survey_result_l4104_410412


namespace sin_three_pi_halves_l4104_410440

theorem sin_three_pi_halves : Real.sin (3 * π / 2) = -1 := by
  sorry

end sin_three_pi_halves_l4104_410440


namespace find_number_l4104_410474

theorem find_number : ∃ x : ℝ, 0.45 * x - 85 = 10 := by
  sorry

end find_number_l4104_410474


namespace opposite_of_four_l4104_410410

-- Define the concept of opposite number
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of 4 is -4
theorem opposite_of_four : opposite 4 = -4 := by
  sorry

end opposite_of_four_l4104_410410


namespace john_bought_two_shirts_l4104_410443

/-- The number of shirts John bought -/
def num_shirts : ℕ := 2

/-- The cost of the first shirt in dollars -/
def cost_first_shirt : ℕ := 15

/-- The cost of the second shirt in dollars -/
def cost_second_shirt : ℕ := cost_first_shirt - 6

/-- The total cost of the shirts in dollars -/
def total_cost : ℕ := 24

theorem john_bought_two_shirts :
  num_shirts = 2 ∧
  cost_first_shirt = cost_second_shirt + 6 ∧
  cost_first_shirt = 15 ∧
  cost_first_shirt + cost_second_shirt = total_cost :=
by sorry

end john_bought_two_shirts_l4104_410443


namespace triangle_angle_C_l4104_410485

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_angle_C (t : Triangle) 
  (h : t.b * (2 * sin t.B + sin t.A) + (2 * t.a + t.b) * sin t.A = 2 * t.c * sin t.C) :
  t.C = 2 * π / 3 := by
  sorry

end triangle_angle_C_l4104_410485
