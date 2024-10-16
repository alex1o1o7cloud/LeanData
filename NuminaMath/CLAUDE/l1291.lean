import Mathlib

namespace NUMINAMATH_CALUDE_penny_halfDollar_same_probability_l1291_129178

/-- Represents the outcome of a single coin flip -/
inductive CoinSide
| Heads
| Tails

/-- Represents the outcome of flipping six different coins -/
structure SixCoinFlip :=
  (penny : CoinSide)
  (nickel : CoinSide)
  (dime : CoinSide)
  (quarter : CoinSide)
  (halfDollar : CoinSide)
  (dollar : CoinSide)

/-- The set of all possible outcomes when flipping six coins -/
def allOutcomes : Finset SixCoinFlip := sorry

/-- The set of outcomes where the penny and half-dollar show the same side -/
def sameOutcomes : Finset SixCoinFlip := sorry

/-- The probability of an event occurring is the number of favorable outcomes
    divided by the total number of possible outcomes -/
def probability (event : Finset SixCoinFlip) : Rat :=
  (event.card : Rat) / (allOutcomes.card : Rat)

theorem penny_halfDollar_same_probability :
  probability sameOutcomes = 1/2 := by sorry

end NUMINAMATH_CALUDE_penny_halfDollar_same_probability_l1291_129178


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1291_129143

/-- Given a real number a, prove that the function f(x) = a^(x-1) + 3 passes through the point (1, 4) -/
theorem fixed_point_of_exponential_function (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 3
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1291_129143


namespace NUMINAMATH_CALUDE_base_conversion_sum_l1291_129111

-- Define the base conversion function
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def n1 : Nat := to_base_10 [2, 1, 4] 8
def n2 : Nat := to_base_10 [3, 2] 5
def n3 : Nat := to_base_10 [3, 4, 3] 9
def n4 : Nat := to_base_10 [1, 3, 3] 4

-- State the theorem
theorem base_conversion_sum :
  (n1 : ℚ) / n2 + (n3 : ℚ) / n4 = 9134 / 527 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l1291_129111


namespace NUMINAMATH_CALUDE_first_player_wins_l1291_129112

/-- Represents the state of the candy game -/
structure GameState :=
  (box1 : Nat) (box2 : Nat)

/-- Checks if a move is valid according to the game rules -/
def isValidMove (s : GameState) (newBox1 : Nat) (newBox2 : Nat) : Prop :=
  (newBox1 < s.box1 ∨ newBox2 < s.box2) ∧
  (newBox1 ≠ 0 ∧ newBox2 ≠ 0) ∧
  ¬(newBox1 % newBox2 = 0 ∨ newBox2 % newBox1 = 0)

/-- Defines a winning strategy for the first player -/
def hasWinningStrategy (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → GameState),
    (∀ s : GameState, isValidMove s (strategy s).box1 (strategy s).box2) ∧
    (∀ s : GameState, ∃ n : Nat, (strategy s).box1 = 2*n ∧ (strategy s).box2 = 2*n + 1) ∧
    (∀ s : GameState, ∀ move : GameState, 
      isValidMove s move.box1 move.box2 → 
      isValidMove (strategy move) (strategy (strategy move)).box1 (strategy (strategy move)).box2)

/-- The main theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  hasWinningStrategy ⟨2017, 2018⟩ :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l1291_129112


namespace NUMINAMATH_CALUDE_first_five_terms_of_series_l1291_129104

def a (n : ℕ+) : ℚ := 1 / (n * (n + 1))

theorem first_five_terms_of_series :
  (List.range 5).map (fun i => a ⟨i + 1, Nat.succ_pos i⟩) = [1/2, 1/6, 1/12, 1/20, 1/30] := by
  sorry

end NUMINAMATH_CALUDE_first_five_terms_of_series_l1291_129104


namespace NUMINAMATH_CALUDE_males_in_band_only_l1291_129154

/-- Represents the number of students in various musical groups and their intersections --/
structure MusicGroups where
  band_male : ℕ
  band_female : ℕ
  orchestra_male : ℕ
  orchestra_female : ℕ
  choir_male : ℕ
  choir_female : ℕ
  band_orchestra_male : ℕ
  band_orchestra_female : ℕ
  band_choir_male : ℕ
  band_choir_female : ℕ
  orchestra_choir_male : ℕ
  orchestra_choir_female : ℕ
  total_students : ℕ

/-- Theorem stating the number of males in the band who are not in the orchestra or choir --/
theorem males_in_band_only (g : MusicGroups)
  (h1 : g.band_male = 120)
  (h2 : g.band_female = 100)
  (h3 : g.orchestra_male = 90)
  (h4 : g.orchestra_female = 130)
  (h5 : g.choir_male = 40)
  (h6 : g.choir_female = 60)
  (h7 : g.band_orchestra_male = 50)
  (h8 : g.band_orchestra_female = 70)
  (h9 : g.band_choir_male = 30)
  (h10 : g.band_choir_female = 40)
  (h11 : g.orchestra_choir_male = 20)
  (h12 : g.orchestra_choir_female = 30)
  (h13 : g.total_students = 260) :
  g.band_male - (g.band_orchestra_male + g.band_choir_male - 20) = 60 := by
  sorry

end NUMINAMATH_CALUDE_males_in_band_only_l1291_129154


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1291_129105

theorem polynomial_simplification (x : ℝ) :
  (2 * x^2 + 5 * x - 3) - (2 * x^2 + 9 * x - 7) = -4 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1291_129105


namespace NUMINAMATH_CALUDE_rectangle_side_length_l1291_129144

theorem rectangle_side_length (area : ℚ) (side1 : ℚ) (side2 : ℚ) : 
  area = 9/16 → side1 = 3/4 → side1 * side2 = area → side2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l1291_129144


namespace NUMINAMATH_CALUDE_runner_catch_up_count_l1291_129131

def num_flags : ℕ := 2015
def laps_A : ℕ := 23
def laps_B : ℕ := 13

theorem runner_catch_up_count :
  let relative_speed := laps_A - laps_B
  let catch_up_count := (relative_speed * num_flags) / (2 * num_flags)
  catch_up_count = 5 := by
  sorry

end NUMINAMATH_CALUDE_runner_catch_up_count_l1291_129131


namespace NUMINAMATH_CALUDE_rational_equation_system_l1291_129168

theorem rational_equation_system (x y z : ℚ) 
  (eq1 : x - y + 2 * z = 1)
  (eq2 : x + y + 4 * z = 3) : 
  x + 2 * y + 5 * z = 4 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_system_l1291_129168


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1291_129128

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + x₁ - 1 = 0) ∧ (x₂^2 + x₂ - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1291_129128


namespace NUMINAMATH_CALUDE_equation_solution_l1291_129102

theorem equation_solution :
  ∃ x : ℚ, ((15 - 2 + (4/x))/2 * 8 = 77) ∧ (x = 16/25) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1291_129102


namespace NUMINAMATH_CALUDE_dividend_proof_l1291_129188

theorem dividend_proof : (10918788 : ℕ) / 12 = 909899 := by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l1291_129188


namespace NUMINAMATH_CALUDE_notebook_purchase_solution_l1291_129145

/-- Represents the number of notebooks bought at each price point -/
structure NotebookPurchase where
  two_dollar : ℕ
  five_dollar : ℕ
  six_dollar : ℕ

/-- Checks if the purchase satisfies the given conditions -/
def is_valid_purchase (p : NotebookPurchase) : Prop :=
  p.two_dollar ≥ 1 ∧ 
  p.five_dollar ≥ 1 ∧ 
  p.six_dollar ≥ 1 ∧
  p.two_dollar + p.five_dollar + p.six_dollar = 20 ∧
  2 * p.two_dollar + 5 * p.five_dollar + 6 * p.six_dollar = 62

theorem notebook_purchase_solution :
  ∃ (p : NotebookPurchase), is_valid_purchase p ∧ p.two_dollar = 14 :=
by sorry

end NUMINAMATH_CALUDE_notebook_purchase_solution_l1291_129145


namespace NUMINAMATH_CALUDE_impossible_to_empty_heap_l1291_129159

/-- Represents the state of the three heaps of stones -/
structure HeapState :=
  (heap1 : Nat) (heap2 : Nat) (heap3 : Nat)

/-- Defines the allowed operations on the heaps -/
inductive Operation
  | Add (target : Nat) (source1 : Nat) (source2 : Nat)
  | Remove (target : Nat) (source1 : Nat) (source2 : Nat)

/-- Applies an operation to a heap state -/
def applyOperation (state : HeapState) (op : Operation) : HeapState :=
  match op with
  | Operation.Add 0 1 2 => HeapState.mk (state.heap1 + state.heap2 + state.heap3) state.heap2 state.heap3
  | Operation.Add 1 0 2 => HeapState.mk state.heap1 (state.heap2 + state.heap1 + state.heap3) state.heap3
  | Operation.Add 2 0 1 => HeapState.mk state.heap1 state.heap2 (state.heap3 + state.heap1 + state.heap2)
  | Operation.Remove 0 1 2 => HeapState.mk (state.heap1 - state.heap2 - state.heap3) state.heap2 state.heap3
  | Operation.Remove 1 0 2 => HeapState.mk state.heap1 (state.heap2 - state.heap1 - state.heap3) state.heap3
  | Operation.Remove 2 0 1 => HeapState.mk state.heap1 state.heap2 (state.heap3 - state.heap1 - state.heap2)
  | _ => state  -- Invalid operations return the original state

/-- Defines the initial state of the heaps -/
def initialState : HeapState := HeapState.mk 1993 199 19

/-- Theorem stating that it's impossible to make a heap empty -/
theorem impossible_to_empty_heap :
  ∀ (operations : List Operation),
    let finalState := operations.foldl applyOperation initialState
    ¬(finalState.heap1 = 0 ∨ finalState.heap2 = 0 ∨ finalState.heap3 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_to_empty_heap_l1291_129159


namespace NUMINAMATH_CALUDE_cake_eating_problem_l1291_129171

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem cake_eating_problem : 
  geometric_series_sum (1/3) (1/3) 7 = 1093/2187 := by sorry

end NUMINAMATH_CALUDE_cake_eating_problem_l1291_129171


namespace NUMINAMATH_CALUDE_fraction_simplification_and_result_l1291_129167

theorem fraction_simplification_and_result (a : ℤ) (h : a = 2018) : 
  (a + 1 : ℚ) / a - a / (a + 1) = (2 * a + 1 : ℚ) / (a * (a + 1)) ∧ 
  2 * a + 1 = 4037 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_and_result_l1291_129167


namespace NUMINAMATH_CALUDE_additional_boys_on_slide_l1291_129152

theorem additional_boys_on_slide (initial_boys total_boys : ℕ) 
  (h1 : initial_boys = 22)
  (h2 : total_boys = 35) :
  total_boys - initial_boys = 13 := by
  sorry

end NUMINAMATH_CALUDE_additional_boys_on_slide_l1291_129152


namespace NUMINAMATH_CALUDE_carls_open_house_l1291_129135

/-- Carl's open house problem -/
theorem carls_open_house 
  (definite_attendees : ℕ) 
  (potential_attendees : ℕ)
  (extravagant_bags : ℕ)
  (average_bags : ℕ)
  (h1 : definite_attendees = 50)
  (h2 : potential_attendees = 40)
  (h3 : extravagant_bags = 10)
  (h4 : average_bags = 20) :
  definite_attendees + potential_attendees - (extravagant_bags + average_bags) = 60 :=
by sorry

end NUMINAMATH_CALUDE_carls_open_house_l1291_129135


namespace NUMINAMATH_CALUDE_investment_proof_l1291_129179

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof of the investment problem -/
theorem investment_proof :
  let initial_investment : ℝ := 400
  let interest_rate : ℝ := 0.12
  let time_period : ℕ := 5
  let final_balance : ℝ := 705.03
  compound_interest initial_investment interest_rate time_period = final_balance :=
by
  sorry


end NUMINAMATH_CALUDE_investment_proof_l1291_129179


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1291_129139

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), (8 * x₁) / 40 = 7 / x₁ ∧ 
                 (8 * x₂) / 40 = 7 / x₂ ∧ 
                 x₁ + x₂ = 0 ∧
                 ∀ (y : ℝ), (8 * y) / 40 = 7 / y → y = x₁ ∨ y = x₂ := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1291_129139


namespace NUMINAMATH_CALUDE_nine_candies_four_bags_l1291_129132

/-- The number of ways to distribute distinct candies among bags --/
def distribute_candies (num_candies : ℕ) (num_bags : ℕ) : ℕ :=
  num_bags ^ (num_candies - num_bags)

/-- Theorem stating the number of ways to distribute 9 distinct candies among 4 bags --/
theorem nine_candies_four_bags : 
  distribute_candies 9 4 = 1024 :=
sorry

end NUMINAMATH_CALUDE_nine_candies_four_bags_l1291_129132


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1291_129186

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1291_129186


namespace NUMINAMATH_CALUDE_volume_ratio_of_cubes_l1291_129118

/-- The perimeter of a square face of a cube -/
def face_perimeter (s : ℝ) : ℝ := 4 * s

/-- The volume of a cube -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- Theorem: Given two cubes A and B with face perimeters 40 cm and 64 cm respectively, 
    the ratio of their volumes is 125:512 -/
theorem volume_ratio_of_cubes (s_A s_B : ℝ) 
  (h_A : face_perimeter s_A = 40)
  (h_B : face_perimeter s_B = 64) : 
  (cube_volume s_A) / (cube_volume s_B) = 125 / 512 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_of_cubes_l1291_129118


namespace NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l1291_129185

theorem triangle_pentagon_side_ratio : ∀ (t p : ℝ),
  (3 * t = 24) →  -- Perimeter of equilateral triangle
  (5 * p = 24) →  -- Perimeter of regular pentagon
  (t / p = 5 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l1291_129185


namespace NUMINAMATH_CALUDE_bobs_hair_length_at_last_cut_l1291_129129

/-- The length of Bob's hair at his last haircut, given his current hair length,
    hair growth rate, and time since last haircut. -/
def hair_length_at_last_cut (current_length : ℝ) (growth_rate : ℝ) (years_since_cut : ℝ) : ℝ :=
  current_length - growth_rate * 12 * years_since_cut

/-- Theorem stating that Bob's hair length at his last haircut was 6 inches,
    given the provided conditions. -/
theorem bobs_hair_length_at_last_cut :
  hair_length_at_last_cut 36 0.5 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bobs_hair_length_at_last_cut_l1291_129129


namespace NUMINAMATH_CALUDE_system_of_equations_l1291_129190

theorem system_of_equations (x y c d : ℝ) : 
  x ≠ 0 →
  y ≠ 0 →
  d ≠ 0 →
  8 * x - 5 * y = c →
  10 * y - 12 * x = d →
  c / d = -2 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l1291_129190


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l1291_129109

-- Define the polynomials
def p (x : ℝ) := x^2 + 2
def q (x : ℝ) := 3*x^3 + 5*x^2 + 2
def r (x : ℝ) := x^4 - 3*x^3 + 2*x^2

-- Define the expression
def expression (x : ℝ) := p x * q x - 2 * r x

-- Theorem statement
theorem nonzero_terms_count : 
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
  ∀ x, expression x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e :=
sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l1291_129109


namespace NUMINAMATH_CALUDE_wind_pressure_theorem_l1291_129103

/-- Represents the joint variation of pressure with area and velocity squared -/
noncomputable def pressure (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^2

/-- Theorem stating the relationship between pressure, area, and velocity -/
theorem wind_pressure_theorem (k : ℝ) :
  (pressure k 2 20 = 4) →
  (pressure k 4 (40 * Real.sqrt 2) = 64) :=
by sorry

end NUMINAMATH_CALUDE_wind_pressure_theorem_l1291_129103


namespace NUMINAMATH_CALUDE_refrigerator_price_l1291_129148

/-- The price paid for a refrigerator given specific conditions --/
theorem refrigerator_price (discount_rate : ℝ) (transport_cost : ℝ) (installation_cost : ℝ)
  (profit_rate : ℝ) (selling_price : ℝ) :
  discount_rate = 0.20 →
  transport_cost = 125 →
  installation_cost = 250 →
  profit_rate = 0.16 →
  selling_price = 18560 →
  ∃ (labelled_price : ℝ),
    selling_price = labelled_price * (1 + profit_rate) ∧
    labelled_price * (1 - discount_rate) + transport_cost + installation_cost = 13175 :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_price_l1291_129148


namespace NUMINAMATH_CALUDE_cost_of_apples_and_oranges_l1291_129160

/-- The cost of apples and oranges given the initial amount and remaining amount -/
def cost_of_fruits (initial_amount remaining_amount : ℚ) : ℚ :=
  initial_amount - remaining_amount

/-- Theorem: The cost of apples and oranges is $15.00 -/
theorem cost_of_apples_and_oranges :
  cost_of_fruits 100 85 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_apples_and_oranges_l1291_129160


namespace NUMINAMATH_CALUDE_max_consecutive_irreducible_l1291_129142

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_irreducible (n : ℕ) : Prop :=
  is_five_digit n ∧ ∀ a b : ℕ, is_three_digit a → is_three_digit b → n ≠ a * b

def consecutive_irreducible (start : ℕ) (count : ℕ) : Prop :=
  ∀ i : ℕ, i < count → is_irreducible (start + i)

theorem max_consecutive_irreducible :
  ∃ start : ℕ, consecutive_irreducible start 99 ∧
  ∀ start' count' : ℕ, count' > 99 → ¬(consecutive_irreducible start' count') :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_irreducible_l1291_129142


namespace NUMINAMATH_CALUDE_complex_solution_l1291_129194

/-- Given two complex numbers a and b satisfying the equations
    2a^2 + ab + 2b^2 = 0 and a + 2b = 5, prove that both a and b are non-real. -/
theorem complex_solution (a b : ℂ) 
  (eq1 : 2 * a^2 + a * b + 2 * b^2 = 0)
  (eq2 : a + 2 * b = 5) :
  ¬(a.im = 0 ∧ b.im = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_solution_l1291_129194


namespace NUMINAMATH_CALUDE_students_failed_l1291_129156

def Q : ℕ := 14

theorem students_failed (x : ℕ) (h1 : x < 4 * Q) 
  (h2 : x % 3 = 0) (h3 : x % 7 = 0) (h4 : x % 2 = 0) 
  (h5 : x = 42) : x - (x / 3 + x / 7 + x / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_students_failed_l1291_129156


namespace NUMINAMATH_CALUDE_wood_length_after_sawing_l1291_129155

/-- The new length of a piece of wood after sawing off a portion. -/
def new_wood_length (original_length saw_off_length : ℝ) : ℝ :=
  original_length - saw_off_length

/-- Theorem stating that the new length of the wood is 6.6 cm. -/
theorem wood_length_after_sawing :
  new_wood_length 8.9 2.3 = 6.6 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_after_sawing_l1291_129155


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l1291_129161

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_nonneg : ∀ x ≥ 0, f x = x * (x + 1)) :
  ∀ x < 0, f x = x * (-x + 1) := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l1291_129161


namespace NUMINAMATH_CALUDE_leanna_leftover_money_l1291_129191

/-- Represents the amount of money Leanna has left after purchasing one CD and two cassettes --/
def money_left_over (total_money : ℕ) (cd_price : ℕ) : ℕ :=
  let cassette_price := total_money - 2 * cd_price
  total_money - (cd_price + 2 * cassette_price)

/-- Theorem stating that Leanna will have $5 left over if she chooses to buy one CD and two cassettes --/
theorem leanna_leftover_money : 
  money_left_over 37 14 = 5 := by
  sorry

end NUMINAMATH_CALUDE_leanna_leftover_money_l1291_129191


namespace NUMINAMATH_CALUDE_park_entrance_cost_is_5_l1291_129177

def park_entrance_cost : ℝ → Prop :=
  λ cost =>
    let num_children := 4
    let num_parents := 2
    let num_grandmother := 1
    let attraction_cost_kid := 2
    let attraction_cost_adult := 4
    let total_paid := 55
    let total_family_members := num_children + num_parents + num_grandmother
    let total_attraction_cost := num_children * attraction_cost_kid + 
                                 (num_parents + num_grandmother) * attraction_cost_adult
    total_paid = total_family_members * cost + total_attraction_cost

theorem park_entrance_cost_is_5 : park_entrance_cost 5 := by
  sorry

end NUMINAMATH_CALUDE_park_entrance_cost_is_5_l1291_129177


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_4_l1291_129134

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2
theorem range_of_a_when_f_geq_4 :
  ∀ x a : ℝ, f x a ≥ 4 → a ≤ -1 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_4_l1291_129134


namespace NUMINAMATH_CALUDE_f_composition_value_l1291_129106

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_value : f (f (1/3)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1291_129106


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1291_129107

/-- Given a point (a,b) outside a circle with radius r centered at the origin,
    prove that the line ax + by = r^2 intersects the circle. -/
theorem line_intersects_circle
  (a b r : ℝ)
  (r_nonzero : r ≠ 0)
  (point_outside : a^2 + b^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ a*x + b*y = r^2 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1291_129107


namespace NUMINAMATH_CALUDE_all_divisors_of_30240_l1291_129150

theorem all_divisors_of_30240 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 9 → 30240 % n = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_divisors_of_30240_l1291_129150


namespace NUMINAMATH_CALUDE_quadratic_equation_with_opposite_roots_l1291_129108

theorem quadratic_equation_with_opposite_roots (x y : ℝ) :
  x^2 - 6*x + 9 = -|y - 1| →
  ∃ (a b c : ℝ), a ≠ 0 ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 ∧
  a = 1 ∧ b = -4 ∧ c = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_opposite_roots_l1291_129108


namespace NUMINAMATH_CALUDE_m_mod_1000_l1291_129162

/-- The set of integers from 1 to 12 -/
def T : Finset ℕ := Finset.range 12

/-- The number of sets of two non-empty disjoint subsets of T -/
def m : ℕ := (3^12 - 2 * 2^12 + 1) / 2

/-- Theorem stating that the remainder of m divided by 1000 is 625 -/
theorem m_mod_1000 : m % 1000 = 625 := by
  sorry

end NUMINAMATH_CALUDE_m_mod_1000_l1291_129162


namespace NUMINAMATH_CALUDE_six_digit_square_reverse_square_exists_l1291_129124

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem six_digit_square_reverse_square_exists : ∃ n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧
  is_perfect_square n ∧
  is_perfect_square (reverse_digits n) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_square_reverse_square_exists_l1291_129124


namespace NUMINAMATH_CALUDE_gcd_372_684_l1291_129199

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_372_684_l1291_129199


namespace NUMINAMATH_CALUDE_tangent_line_and_minimum_value_and_a_range_l1291_129158

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + (1 - x) * Real.exp x

noncomputable def g (a x : ℝ) : ℝ := x - (1 + a) * Real.log x - a / x

theorem tangent_line_and_minimum_value_and_a_range 
  (a : ℝ) 
  (h_a : a < 1) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 0, ∃ x₂ ∈ Set.Icc (Real.exp 1) 3, f x₁ > g a x₂) →
  (Real.exp 2 - 2 * Real.exp 1) / (Real.exp 1 + 1) < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_minimum_value_and_a_range_l1291_129158


namespace NUMINAMATH_CALUDE_tank_capacity_l1291_129147

theorem tank_capacity (initial_fill : ℚ) (added_gallons : ℚ) (final_fill : ℚ) :
  initial_fill = 3 / 4 →
  added_gallons = 9 →
  final_fill = 9 / 10 →
  ∃ (capacity : ℚ), capacity = 60 ∧ 
    final_fill * capacity = initial_fill * capacity + added_gallons :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l1291_129147


namespace NUMINAMATH_CALUDE_nine_point_centers_property_l1291_129126

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The intersection point of the diagonals -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- Checks if four points are collinear -/
def areCollinear (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Checks if four points form a parallelogram -/
def formParallelogram (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Computes the nine-point center of a triangle -/
def ninePointCenter (a b c : Point) : Point :=
  sorry

/-- The main theorem -/
theorem nine_point_centers_property (q : Quadrilateral) :
  let X := diagonalIntersection q
  let center1 := ninePointCenter X q.A q.B
  let center2 := ninePointCenter X q.B q.C
  let center3 := ninePointCenter X q.C q.D
  let center4 := ninePointCenter X q.D q.A
  areCollinear center1 center2 center3 center4 ∨ 
  formParallelogram center1 center2 center3 center4 := by
  sorry

end NUMINAMATH_CALUDE_nine_point_centers_property_l1291_129126


namespace NUMINAMATH_CALUDE_smallest_tree_height_l1291_129123

/-- Given three trees with specific height relationships, prove the height of the smallest tree -/
theorem smallest_tree_height (tallest middle smallest : ℝ) : 
  tallest = 108 →
  middle = tallest / 2 - 6 →
  smallest = middle / 4 →
  smallest = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_tree_height_l1291_129123


namespace NUMINAMATH_CALUDE_matrix_equality_implies_ratio_l1291_129198

theorem matrix_equality_implies_ratio (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  A * B = B * A ∧ 4 * b ≠ c →
  (a - d) / (c - 4 * b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_implies_ratio_l1291_129198


namespace NUMINAMATH_CALUDE_right_triangle_minimum_side_l1291_129117

theorem right_triangle_minimum_side : ∃ (s : ℕ), 
  (s ≥ 25) ∧ 
  (∀ (t : ℕ), t < 25 → ¬(7^2 + 24^2 = t^2)) ∧
  (7^2 + 24^2 = s^2) ∧
  (7 + 24 > s) ∧ (24 + s > 7) ∧ (7 + s > 24) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_minimum_side_l1291_129117


namespace NUMINAMATH_CALUDE_haleys_initial_marbles_l1291_129181

/-- The number of marbles Haley gives to each boy -/
def marbles_per_boy : ℕ := 2

/-- The number of boys in Haley's class -/
def number_of_boys : ℕ := 14

/-- The theorem stating the number of marbles Haley had initially -/
theorem haleys_initial_marbles : 
  marbles_per_boy * number_of_boys = 28 := by
  sorry

end NUMINAMATH_CALUDE_haleys_initial_marbles_l1291_129181


namespace NUMINAMATH_CALUDE_problem_statement_l1291_129151

theorem problem_statement (a : ℝ) (h : a^2 + a = 0) : 4*a^2 + 4*a + 2011 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1291_129151


namespace NUMINAMATH_CALUDE_triangle_side_length_l1291_129101

theorem triangle_side_length (A B C : ℝ × ℝ) : 
  let angle_BAC := Real.pi / 3  -- 60 degrees in radians
  let AB := 2
  let AC := 4
  let BC := ‖B - C‖  -- Euclidean distance between B and C
  (angle_BAC = Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / (AB * AC)) →  -- angle condition
  (AB = ‖B - A‖) →  -- AB length condition
  (AC = ‖C - A‖) →  -- AC length condition
  BC = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1291_129101


namespace NUMINAMATH_CALUDE_one_weighing_sufficient_l1291_129141

/-- Represents a coin, which can be either real or counterfeit -/
inductive Coin
  | Real
  | Counterfeit

/-- Represents the result of weighing two coins -/
inductive WeighResult
  | Left  -- Left side is lighter
  | Right -- Right side is lighter
  | Equal -- Both sides are equal

/-- A function that simulates weighing two coins -/
def weigh (a b : Coin) : WeighResult :=
  match a, b with
  | Coin.Counterfeit, Coin.Real    => WeighResult.Left
  | Coin.Real, Coin.Counterfeit    => WeighResult.Right
  | Coin.Real, Coin.Real           => WeighResult.Equal
  | Coin.Counterfeit, Coin.Counterfeit => WeighResult.Equal

/-- A function that determines the counterfeit coin given three coins -/
def findCounterfeit (a b c : Coin) : Coin :=
  match weigh a b with
  | WeighResult.Left  => a
  | WeighResult.Right => b
  | WeighResult.Equal => c

theorem one_weighing_sufficient :
  ∀ (a b c : Coin),
  (∃! x, x = Coin.Counterfeit) →
  (a = Coin.Counterfeit ∨ b = Coin.Counterfeit ∨ c = Coin.Counterfeit) →
  findCounterfeit a b c = Coin.Counterfeit :=
by sorry

end NUMINAMATH_CALUDE_one_weighing_sufficient_l1291_129141


namespace NUMINAMATH_CALUDE_triangle_exists_but_not_isosceles_l1291_129138

def stick_lengths : List ℝ := [1, 1.9, 1.9^2, 1.9^3, 1.9^4, 1.9^5, 1.9^6, 1.9^7, 1.9^8, 1.9^9]

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∧ a + b > c) ∨ (b = c ∧ b + c > a) ∨ (c = a ∧ c + a > b)

theorem triangle_exists_but_not_isosceles :
  (∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_triangle a b c) ∧
  (¬ ∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_isosceles_triangle a b c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_exists_but_not_isosceles_l1291_129138


namespace NUMINAMATH_CALUDE_circles_intersect_l1291_129157

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 9

-- Define the centers and radii
def center1 : ℝ × ℝ := (-2, 0)
def center2 : ℝ × ℝ := (2, 1)
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles intersect
theorem circles_intersect :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l1291_129157


namespace NUMINAMATH_CALUDE_range_of_fraction_l1291_129125

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  (1/6 : ℝ) ≤ x/y ∧ x/y ≤ (4/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1291_129125


namespace NUMINAMATH_CALUDE_min_value_m_plus_n_l1291_129114

theorem min_value_m_plus_n (m n : ℝ) : 
  m * 1 + n * 1 - 3 * m * n = 0 → 
  m * n > 0 → 
  m + n ≥ 4/3 ∧ ∃ (m₀ n₀ : ℝ), m₀ * 1 + n₀ * 1 - 3 * m₀ * n₀ = 0 ∧ m₀ * n₀ > 0 ∧ m₀ + n₀ = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_m_plus_n_l1291_129114


namespace NUMINAMATH_CALUDE_decimal_equals_fraction_fraction_is_lowest_terms_sum_of_numerator_and_denominator_l1291_129195

/-- The decimal representation of the number we're considering -/
def repeating_decimal : ℚ := 0.47

/-- The fraction representation of the repeating decimal -/
def fraction : ℚ := 47 / 99

/-- Theorem stating that the repeating decimal equals the fraction -/
theorem decimal_equals_fraction : repeating_decimal = fraction := by sorry

/-- Theorem stating that the fraction is in lowest terms -/
theorem fraction_is_lowest_terms : 
  ∀ (a b : ℕ), a / b = fraction → b ≠ 0 → a.gcd b = 1 := by sorry

/-- The main theorem to prove -/
theorem sum_of_numerator_and_denominator : 
  ∃ (n d : ℕ), n / d = fraction ∧ n.gcd d = 1 ∧ n + d = 146 := by sorry

end NUMINAMATH_CALUDE_decimal_equals_fraction_fraction_is_lowest_terms_sum_of_numerator_and_denominator_l1291_129195


namespace NUMINAMATH_CALUDE_philip_initial_paintings_l1291_129164

/-- Represents the number of paintings Philip makes per day -/
def paintings_per_day : ℕ := 2

/-- Represents the number of days Philip will paint -/
def days : ℕ := 30

/-- Represents the total number of paintings Philip will have after 30 days -/
def total_paintings : ℕ := 80

/-- Calculates the initial number of paintings Philip had -/
def initial_paintings : ℕ := total_paintings - (paintings_per_day * days)

theorem philip_initial_paintings : initial_paintings = 20 := by
  sorry

end NUMINAMATH_CALUDE_philip_initial_paintings_l1291_129164


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l1291_129165

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ,
    Odd n ∧
    contains_digit n 5 ∧
    3 ∣ n ∧
    12^2 < n ∧
    n < 13^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l1291_129165


namespace NUMINAMATH_CALUDE_cos_225_degrees_l1291_129121

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l1291_129121


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l1291_129133

theorem power_mod_seventeen : 5^2023 ≡ 11 [ZMOD 17] := by sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l1291_129133


namespace NUMINAMATH_CALUDE_remainder_less_than_divisor_l1291_129176

theorem remainder_less_than_divisor (a d : ℤ) (h : d ≠ 0) :
  ∃ (q r : ℤ), a = q * d + r ∧ 0 ≤ r ∧ r < |d| := by
  sorry

end NUMINAMATH_CALUDE_remainder_less_than_divisor_l1291_129176


namespace NUMINAMATH_CALUDE_circle_equation_l1291_129100

theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -1)
  let point : ℝ × ℝ := (-1, 3)
  let radius : ℝ := Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2)
  (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1291_129100


namespace NUMINAMATH_CALUDE_ellipse_condition_l1291_129166

/-- An ellipse equation with parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (9 - k) + y^2 / (k - 4) = 1

/-- The condition 4 < k < 9 -/
def condition (k : ℝ) : Prop := 4 < k ∧ k < 9

/-- The statement to be proven -/
theorem ellipse_condition :
  (∀ k, is_ellipse k → condition k) ∧
  ¬(∀ k, condition k → is_ellipse k) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1291_129166


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l1291_129183

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℤ) + (⌈x⌉ : ℤ) = 7 ↔ 3 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l1291_129183


namespace NUMINAMATH_CALUDE_cards_playing_with_l1291_129120

/-- The number of cards in a standard deck --/
def standard_deck : Nat := 52

/-- The number of cards kept away --/
def cards_kept_away : Nat := 7

/-- Theorem: The number of cards they were playing with is 45 --/
theorem cards_playing_with : 
  standard_deck - cards_kept_away = 45 := by
  sorry

end NUMINAMATH_CALUDE_cards_playing_with_l1291_129120


namespace NUMINAMATH_CALUDE_cat_finishes_food_on_day_l1291_129116

/-- Represents the days of the week -/
inductive Day : Type
| monday : Day
| tuesday : Day
| wednesday : Day
| thursday : Day
| friday : Day
| saturday : Day
| sunday : Day

/-- Calculates the number of days since Monday -/
def daysSinceMonday (d : Day) : ℕ :=
  match d with
  | Day.monday => 0
  | Day.tuesday => 1
  | Day.wednesday => 2
  | Day.thursday => 3
  | Day.friday => 4
  | Day.saturday => 5
  | Day.sunday => 6

/-- The amount of food the cat eats in the morning (in cans) -/
def morningMeal : ℚ := 2/5

/-- The amount of food the cat eats in the evening (in cans) -/
def eveningMeal : ℚ := 1/6

/-- The total number of cans in the box -/
def totalCans : ℕ := 10

/-- The day on which the cat finishes all the food -/
def finishDay : Day := Day.saturday

/-- Theorem stating that the cat finishes all the food on the specified day -/
theorem cat_finishes_food_on_day :
  (morningMeal + eveningMeal) * (daysSinceMonday finishDay + 1 : ℚ) > totalCans ∧
  (morningMeal + eveningMeal) * (daysSinceMonday finishDay : ℚ) ≤ totalCans :=
by sorry


end NUMINAMATH_CALUDE_cat_finishes_food_on_day_l1291_129116


namespace NUMINAMATH_CALUDE_gcd_of_integer_differences_l1291_129193

theorem gcd_of_integer_differences (a b c d : ℤ) : 
  ∃ k : ℤ, (a - b) * (b - c) * (c - d) * (d - a) * (a - c) * (b - d) = 12 * k :=
sorry

end NUMINAMATH_CALUDE_gcd_of_integer_differences_l1291_129193


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1291_129169

theorem sum_of_coefficients (a c : ℚ) : 
  (3 : ℚ) ∈ {x | a * x^2 - 6 * x + c = 0} →
  (1/3 : ℚ) ∈ {x | a * x^2 - 6 * x + c = 0} →
  a + c = 18/5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1291_129169


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l1291_129170

theorem smallest_right_triangle_area :
  let a : ℝ := 6
  let b : ℝ := 8
  let area1 : ℝ := (1/2) * a * b
  let area2 : ℝ := (1/2) * a * Real.sqrt (b^2 - a^2)
  min area1 area2 = (3 : ℝ) * Real.sqrt 28 := by
sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l1291_129170


namespace NUMINAMATH_CALUDE_circle_radius_l1291_129187

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 8 = 2*x + 4*y) → 
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1291_129187


namespace NUMINAMATH_CALUDE_average_position_l1291_129119

def fractions : List ℚ := [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]

theorem average_position (average : ℚ := (fractions.sum) / 6) :
  average = 223/840 ∧ 1/4 < average ∧ average < 1/3 := by sorry

end NUMINAMATH_CALUDE_average_position_l1291_129119


namespace NUMINAMATH_CALUDE_percentage_difference_l1291_129122

theorem percentage_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.25 * y) : 
  x - y = 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1291_129122


namespace NUMINAMATH_CALUDE_x_squared_coefficient_zero_l1291_129175

/-- The coefficient of x^2 in the expansion of (x^2+ax+1)(x^2-3a+2) is zero when a = 1 -/
theorem x_squared_coefficient_zero (a : ℝ) : 
  (a = 1) ↔ ((-3 * a + 2 + 1) = 0) := by sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_zero_l1291_129175


namespace NUMINAMATH_CALUDE_karen_graded_eight_tests_l1291_129189

/-- Represents the bonus calculation for a teacher based on test scores. -/
def bonus_calculation (n : ℕ) : Prop :=
  let base_bonus := 500
  let extra_bonus_per_point := 10
  let base_threshold := 75
  let max_score := 150
  let current_average := 70
  let last_two_tests_score := 290
  let target_bonus := 600
  let total_current_points := n * current_average
  let total_points_after := total_current_points + last_two_tests_score
  let final_average := total_points_after / (n + 2)
  (final_average > base_threshold) ∧
  (target_bonus = base_bonus + (final_average - base_threshold) * extra_bonus_per_point) ∧
  (∀ m : ℕ, m ≤ n + 2 → m * max_score ≥ total_points_after)

/-- Theorem stating that Karen has graded 8 tests. -/
theorem karen_graded_eight_tests : ∃ (n : ℕ), bonus_calculation n ∧ n = 8 :=
  sorry

end NUMINAMATH_CALUDE_karen_graded_eight_tests_l1291_129189


namespace NUMINAMATH_CALUDE_aaron_sweaters_count_l1291_129115

/-- The number of sweaters Aaron made -/
def aaron_sweaters : ℕ := 5

/-- The number of scarves Aaron made -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Enid made -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used -/
def total_wool : ℕ := 82

theorem aaron_sweaters_count : 
  aaron_sweaters * wool_per_sweater + 
  aaron_scarves * wool_per_scarf + 
  enid_sweaters * wool_per_sweater = total_wool :=
sorry

end NUMINAMATH_CALUDE_aaron_sweaters_count_l1291_129115


namespace NUMINAMATH_CALUDE_line_symmetry_l1291_129182

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y + 3 = 0
def line2 (x y : ℝ) : Prop := y = x + 2
def symmetric_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (f g h : ℝ → ℝ → Prop) : Prop :=
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    f x₁ y₁ → h x₂ y₂ → 
    ∃ (x_mid y_mid : ℝ), g x_mid y_mid ∧
    (x₂ - x_mid = x_mid - x₁) ∧ (y₂ - y_mid = y_mid - y₁)

-- Theorem statement
theorem line_symmetry : symmetric_wrt line1 line2 symmetric_line :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l1291_129182


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1291_129192

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 24 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y + 2 * y + 24 = 0 → y = x) ↔ 
  (k = 2 + 12 * Real.sqrt 2 ∨ k = 2 - 12 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1291_129192


namespace NUMINAMATH_CALUDE_inverse_function_point_l1291_129127

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the condition that f(x-1) passes through (1, 2)
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f (1 - 1) = 2

-- Define the inverse function of f
noncomputable def f_inverse (f : ℝ → ℝ) : ℝ → ℝ :=
  Function.invFun f

-- Theorem statement
theorem inverse_function_point (f : ℝ → ℝ) :
  passes_through_point f → f_inverse f 2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_function_point_l1291_129127


namespace NUMINAMATH_CALUDE_steves_oranges_l1291_129197

/-- Steve's orange sharing problem -/
theorem steves_oranges (initial_oranges shared_oranges : ℕ) :
  initial_oranges = 46 →
  shared_oranges = 4 →
  initial_oranges - shared_oranges = 42 := by
  sorry

end NUMINAMATH_CALUDE_steves_oranges_l1291_129197


namespace NUMINAMATH_CALUDE_even_number_of_solutions_l1291_129153

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  (y^2 + 6) * (x - 1) = y * (x^2 + 1) ∧
  (x^2 + 6) * (y - 1) = x * (y^2 + 1)

/-- The set of solutions to the system -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | system p.1 p.2}

/-- The number of solutions is finite -/
axiom finite_solutions : Set.Finite solution_set

/-- Theorem: The system has an even number of real solutions -/
theorem even_number_of_solutions : ∃ n : ℕ, n % 2 = 0 ∧ Set.ncard solution_set = n := by
  sorry

end NUMINAMATH_CALUDE_even_number_of_solutions_l1291_129153


namespace NUMINAMATH_CALUDE_nova_annual_donation_l1291_129110

/-- Nova's monthly donation in dollars -/
def monthly_donation : ℕ := 1707

/-- Number of months in a year -/
def months_in_year : ℕ := 12

/-- Nova's total annual donation in dollars -/
def annual_donation : ℕ := monthly_donation * months_in_year

theorem nova_annual_donation :
  annual_donation = 20484 := by
  sorry

end NUMINAMATH_CALUDE_nova_annual_donation_l1291_129110


namespace NUMINAMATH_CALUDE_fishing_line_length_l1291_129196

/-- The original length of a fishing line can be calculated from its current length. -/
theorem fishing_line_length (current_length : ℝ) (h : current_length = 8.9) :
  (current_length + 3.1) * 3.1 * 2.1 = 78.12 := by
  sorry

#check fishing_line_length

end NUMINAMATH_CALUDE_fishing_line_length_l1291_129196


namespace NUMINAMATH_CALUDE_grapes_distribution_l1291_129163

theorem grapes_distribution (total_grapes : ℕ) (num_kids : ℕ) 
  (h1 : total_grapes = 50)
  (h2 : num_kids = 7) :
  total_grapes % num_kids = 1 := by
  sorry

end NUMINAMATH_CALUDE_grapes_distribution_l1291_129163


namespace NUMINAMATH_CALUDE_inequality_conversions_l1291_129130

theorem inequality_conversions (x : ℝ) : 
  ((5 * x > 4 * x - 1) ↔ (x > -1)) ∧ 
  ((-x - 2 < 7) ↔ (x > -9)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_conversions_l1291_129130


namespace NUMINAMATH_CALUDE_fenced_perimeter_is_50_l1291_129136

/-- Represents a rectangular yard with given dimensions -/
structure RectangularYard where
  width : ℝ
  length : ℝ
  area : ℝ
  width_value : width = 40
  area_value : area = 200
  area_eq : area = width * length

/-- Calculates the fenced perimeter of the yard -/
def fenced_perimeter (yard : RectangularYard) : ℝ :=
  2 * yard.length + yard.width

theorem fenced_perimeter_is_50 (yard : RectangularYard) :
  fenced_perimeter yard = 50 := by
  sorry

end NUMINAMATH_CALUDE_fenced_perimeter_is_50_l1291_129136


namespace NUMINAMATH_CALUDE_second_smallest_odd_number_l1291_129180

theorem second_smallest_odd_number (a b c d : ℕ) : 
  (∃ x : ℕ, a = 2*x + 1 ∧ b = 2*x + 3 ∧ c = 2*x + 5 ∧ d = 2*x + 7) →  -- consecutive odd numbers
  a + b + c + d = 112 →                                             -- sum is 112
  b = 27                                                            -- 2nd smallest is 27
  := by sorry

end NUMINAMATH_CALUDE_second_smallest_odd_number_l1291_129180


namespace NUMINAMATH_CALUDE_alcohol_solution_concentration_l1291_129174

theorem alcohol_solution_concentration 
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_alcohol : ℝ)
  (h1 : initial_volume = 6)
  (h2 : initial_percentage = 0.4)
  (h3 : added_alcohol = 1.2) :
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  let final_percentage := final_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_concentration_l1291_129174


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_power_10000_l1291_129137

theorem last_three_digits_of_7_power_10000 (h : 7^250 ≡ 1 [ZMOD 1250]) :
  7^10000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_power_10000_l1291_129137


namespace NUMINAMATH_CALUDE_trapezoid_area_l1291_129149

/-- Given two equilateral triangles and four congruent trapezoids between them,
    this theorem proves that the area of one trapezoid is 8 square units. -/
theorem trapezoid_area
  (outer_triangle_area : ℝ)
  (inner_triangle_area : ℝ)
  (num_trapezoids : ℕ)
  (h_outer_area : outer_triangle_area = 36)
  (h_inner_area : inner_triangle_area = 4)
  (h_num_trapezoids : num_trapezoids = 4) :
  (outer_triangle_area - inner_triangle_area) / num_trapezoids = 8 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1291_129149


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l1291_129113

/-- A bag containing red and white balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- The number of balls drawn from the bag -/
def drawn : ℕ := 3

/-- The initial bag configuration -/
def initial_bag : Bag := ⟨5, 3⟩

/-- Event: Exactly one red ball is drawn -/
def exactly_one_red (b : Bag) : Prop := sorry

/-- Event: Exactly two red balls are drawn -/
def exactly_two_red (b : Bag) : Prop := sorry

/-- Two events are mutually exclusive -/
def mutually_exclusive (e1 e2 : Bag → Prop) : Prop := sorry

/-- Two events are contradictory -/
def contradictory (e1 e2 : Bag → Prop) : Prop := sorry

theorem events_mutually_exclusive_not_contradictory :
  mutually_exclusive exactly_one_red exactly_two_red ∧
  ¬contradictory exactly_one_red exactly_two_red :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l1291_129113


namespace NUMINAMATH_CALUDE_min_S_value_l1291_129184

noncomputable def S (x y z : ℝ) : ℝ := (z + 1)^2 / (2 * x * y * z)

theorem min_S_value (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_constraint : x^2 + y^2 + z^2 = 1) :
  (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x'^2 + y'^2 + z'^2 = 1 → 
    S x y z ≤ S x' y' z') →
  x = Real.sqrt (Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_min_S_value_l1291_129184


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1291_129146

theorem complex_equation_solution (x y : ℝ) (i : ℂ) (h : i * i = -1) :
  (2 * x - 1 : ℂ) + i = y - (3 - y) * i →
  x = 5 / 2 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1291_129146


namespace NUMINAMATH_CALUDE_tangent_point_bisects_second_side_l1291_129172

/-- A pentagon inscribed around a circle -/
structure InscribedPentagon where
  /-- The lengths of the sides of the pentagon -/
  sides : Fin 5 → ℕ
  /-- The first and third sides have length 1 -/
  first_third_sides_one : sides 0 = 1 ∧ sides 2 = 1
  /-- The point where the circle touches the second side of the pentagon -/
  tangent_point : ℝ
  /-- The tangent point is between 0 and the length of the second side -/
  tangent_point_valid : 0 < tangent_point ∧ tangent_point < sides 1

/-- The theorem stating that the tangent point divides the second side into two equal segments -/
theorem tangent_point_bisects_second_side (p : InscribedPentagon) :
  p.tangent_point = (p.sides 1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_bisects_second_side_l1291_129172


namespace NUMINAMATH_CALUDE_area_triangle_STU_l1291_129173

/-- Square pyramid with given dimensions and points -/
structure SquarePyramid where
  -- Base side length
  base : ℝ
  -- Altitude
  height : ℝ
  -- Point S position ratio
  s_ratio : ℝ
  -- Point T position ratio
  t_ratio : ℝ
  -- Point U position ratio
  u_ratio : ℝ

/-- Theorem stating the area of triangle STU in the square pyramid -/
theorem area_triangle_STU (p : SquarePyramid) 
  (h_base : p.base = 4)
  (h_height : p.height = 8)
  (h_s : p.s_ratio = 1/4)
  (h_t : p.t_ratio = 1/2)
  (h_u : p.u_ratio = 3/4) :
  ∃ (area : ℝ), area = 7.5 ∧ 
  area = (1/2) * Real.sqrt ((p.s_ratio * p.height)^2 + (p.base/2)^2) * 
         (p.u_ratio * Real.sqrt (p.height^2 + (p.base/2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_STU_l1291_129173


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1291_129140

theorem quadratic_discriminant : 
  let a : ℝ := 1
  let b : ℝ := -4
  let c : ℝ := 3
  b^2 - 4*a*c = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1291_129140
