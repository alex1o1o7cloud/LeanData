import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2300_230010

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : is_geometric_sequence a)
  (h_mean : Real.sqrt (a 4 * a 14) = 2 * Real.sqrt 2) :
  (2 * a 7 + a 11) ≥ 8 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 8 ∧ x * y = 8 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2300_230010


namespace NUMINAMATH_CALUDE_real_complex_condition_l2300_230095

theorem real_complex_condition (a : ℝ) : 
  (Complex.I * (a - 1)^2 + 4*a).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_complex_condition_l2300_230095


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2300_230029

theorem solve_linear_equation (x : ℝ) : 3*x - 5*x + 7*x = 210 → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2300_230029


namespace NUMINAMATH_CALUDE_picture_position_l2300_230087

theorem picture_position (wall_width picture_width shift : ℝ) 
  (hw : wall_width = 25)
  (hp : picture_width = 4)
  (hs : shift = 1) :
  let center := wall_width / 2
  let picture_center := center + shift
  let left_edge := picture_center - picture_width / 2
  left_edge = 11.5 := by sorry

end NUMINAMATH_CALUDE_picture_position_l2300_230087


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l2300_230008

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + 2*y + z = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ ∀ (a b c : ℝ), a + 2*b + c = 1 → a^2 + 4*b^2 + c^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l2300_230008


namespace NUMINAMATH_CALUDE_train_passengers_l2300_230003

theorem train_passengers (initial_passengers : ℕ) : 
  (initial_passengers - 263 + 419 = 725) → initial_passengers = 569 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l2300_230003


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2300_230081

theorem arithmetic_calculations :
  (((1 : ℤ) - 32 - 11 + (-9) - (-16)) = -36) ∧
  (-(1 : ℚ)^4 - |0 - 1| * 2 - (-3)^2 / (-3/2) = 3) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2300_230081


namespace NUMINAMATH_CALUDE_max_xy_value_l2300_230015

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) :
  x * y ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    4 * x₀^2 + 9 * y₀^2 + 3 * x₀ * y₀ = 30 ∧ x₀ * y₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l2300_230015


namespace NUMINAMATH_CALUDE_six_x_value_l2300_230077

theorem six_x_value (x : ℝ) (h : 3 * x - 9 = 12) : 6 * x = 42 := by
  sorry

end NUMINAMATH_CALUDE_six_x_value_l2300_230077


namespace NUMINAMATH_CALUDE_percentage_relation_l2300_230097

theorem percentage_relation (A B x : ℝ) (hA : A > 0) (hB : B > 0) (h_relation : A = (x / 100) * B) : 
  x = 100 * (A / B) := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l2300_230097


namespace NUMINAMATH_CALUDE_average_age_of_new_men_l2300_230042

theorem average_age_of_new_men (n : ℕ) (old_avg : ℝ) (age1 age2 : ℕ) (increase : ℝ) :
  n = 15 →
  age1 = 21 →
  age2 = 23 →
  increase = 2 →
  (n * (old_avg + increase) - n * old_avg) = ((n * increase + age1 + age2) / 2) →
  ((n * increase + age1 + age2) / 2) = 37 :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_new_men_l2300_230042


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2300_230046

theorem quadratic_inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + a^2 > 0) ↔ (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2300_230046


namespace NUMINAMATH_CALUDE_alexis_has_60_mangoes_l2300_230092

/-- Represents the number of mangoes each person has -/
structure MangoDistribution where
  alexis : ℕ
  dilan : ℕ
  ashley : ℕ

/-- Defines the conditions of the mango distribution problem -/
def validDistribution (d : MangoDistribution) : Prop :=
  (d.alexis = 4 * (d.dilan + d.ashley)) ∧
  (d.alexis + d.dilan + d.ashley = 75)

/-- Theorem stating that Alexis has 60 mangoes in a valid distribution -/
theorem alexis_has_60_mangoes (d : MangoDistribution) 
  (h : validDistribution d) : d.alexis = 60 := by
  sorry

end NUMINAMATH_CALUDE_alexis_has_60_mangoes_l2300_230092


namespace NUMINAMATH_CALUDE_sum_division_l2300_230078

/-- The problem of dividing a sum among x, y, and z -/
theorem sum_division (x y z : ℝ) : 
  (∀ (r : ℝ), y = 0.45 * r → z = 0.5 * r → x = r) →  -- For each rupee x gets, y gets 0.45 and z gets 0.5
  y = 63 →  -- y's share is 63 rupees
  x + y + z = 273 := by  -- The total amount is 273 rupees
sorry


end NUMINAMATH_CALUDE_sum_division_l2300_230078


namespace NUMINAMATH_CALUDE_equation_solution_l2300_230050

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => (x^2 - x + 1) * (3*x^2 - 10*x + 3) - 20*x^2
  ∀ x : ℝ, f x = 0 ↔ x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2300_230050


namespace NUMINAMATH_CALUDE_cistern_length_l2300_230069

theorem cistern_length (width depth area : ℝ) (h1 : width = 4)
    (h2 : depth = 1.25) (h3 : area = 55.5) :
  ∃ length : ℝ, length = 7 ∧ 
    area = (length * width) + 2 * (length * depth) + 2 * (width * depth) :=
by sorry

end NUMINAMATH_CALUDE_cistern_length_l2300_230069


namespace NUMINAMATH_CALUDE_maple_trees_after_planting_l2300_230028

theorem maple_trees_after_planting (initial_trees : ℕ) (new_trees : ℕ) : 
  initial_trees = 2 → new_trees = 9 → initial_trees + new_trees = 11 := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_after_planting_l2300_230028


namespace NUMINAMATH_CALUDE_problem_statement_l2300_230013

theorem problem_statement (x y : ℝ) 
  (h1 : x * y + x + y = 17)
  (h2 : x^2 * y + x * y^2 = 66) :
  x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4 = 12499 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2300_230013


namespace NUMINAMATH_CALUDE_second_year_interest_rate_l2300_230059

/-- Calculates the interest rate for the second year given the initial amount,
    first-year interest rate, and final amount after two years. -/
theorem second_year_interest_rate
  (initial_amount : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_amount = 9000)
  (h2 : first_year_rate = 0.04)
  (h3 : final_amount = 9828) :
  ∃ (second_year_rate : ℝ),
    second_year_rate = 0.05 ∧
    final_amount = initial_amount * (1 + first_year_rate) * (1 + second_year_rate) :=
by sorry


end NUMINAMATH_CALUDE_second_year_interest_rate_l2300_230059


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l2300_230030

theorem ratio_of_numbers (sum : ℚ) (bigger : ℚ) (h1 : sum = 143) (h2 : bigger = 104) :
  (sum - bigger) / bigger = 39 / 104 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l2300_230030


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2300_230082

theorem circle_area_ratio (s r : ℝ) (hs : s > 0) (hr : r > 0) (h : r = 0.6 * s) : 
  (π * (r / 2)^2) / (π * (s / 2)^2) = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2300_230082


namespace NUMINAMATH_CALUDE_negative_six_greater_than_negative_seven_l2300_230079

theorem negative_six_greater_than_negative_seven : -6 > -7 := by
  sorry

end NUMINAMATH_CALUDE_negative_six_greater_than_negative_seven_l2300_230079


namespace NUMINAMATH_CALUDE_evaluate_expression_l2300_230065

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 2) :
  4 * x^(y + 1) + 5 * y^(x + 1) = 188 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2300_230065


namespace NUMINAMATH_CALUDE_expression_evaluation_l2300_230071

theorem expression_evaluation :
  let x : ℚ := -1/4
  let y : ℚ := -1/2
  4*x*y - ((x^2 + 5*x*y - y^2) - (x^2 + 3*x*y - 2*y^2)) = 0 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2300_230071


namespace NUMINAMATH_CALUDE_simplify_expression_l2300_230017

theorem simplify_expression (x y : ℝ) (n : ℤ) :
  (4 * x^(n+1) * y^n)^2 / ((-x*y)^2)^n = 16 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2300_230017


namespace NUMINAMATH_CALUDE_dinner_bill_proof_l2300_230076

/-- The number of friends who went to dinner -/
def total_friends : ℕ := 10

/-- The number of friends who paid -/
def paying_friends : ℕ := 9

/-- The extra amount each paying friend contributed -/
def extra_payment : ℚ := 3

/-- The discount rate applied to the bill -/
def discount_rate : ℚ := 1/10

/-- The original bill before discount -/
def original_bill : ℚ := 300

theorem dinner_bill_proof :
  let discounted_bill := original_bill * (1 - discount_rate)
  let individual_share := discounted_bill / total_friends
  paying_friends * (individual_share + extra_payment) = discounted_bill :=
by sorry

end NUMINAMATH_CALUDE_dinner_bill_proof_l2300_230076


namespace NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_l2300_230009

theorem solution_set_x_squared_minus_one (x : ℝ) : x^2 - 1 ≥ 0 ↔ x ≥ 1 ∨ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_l2300_230009


namespace NUMINAMATH_CALUDE_cannot_empty_both_piles_l2300_230027

/-- Represents the state of the two piles of coins -/
structure CoinPiles :=
  (pile1 : ℕ)
  (pile2 : ℕ)

/-- Represents the allowed operations on the piles -/
inductive Operation
  | transferAndAdd : Operation
  | removeFour : Operation

/-- Applies an operation to the current state of the piles -/
def applyOperation (state : CoinPiles) (op : Operation) : CoinPiles :=
  match op with
  | Operation.transferAndAdd => 
      if state.pile1 > 0 then 
        CoinPiles.mk (state.pile1 - 1) (state.pile2 + 3)
      else 
        CoinPiles.mk (state.pile1 + 3) (state.pile2 - 1)
  | Operation.removeFour => 
      if state.pile1 ≥ 4 then 
        CoinPiles.mk (state.pile1 - 4) state.pile2
      else 
        CoinPiles.mk state.pile1 (state.pile2 - 4)

/-- The initial state of the piles -/
def initialState : CoinPiles := CoinPiles.mk 1 0

/-- Theorem stating that it's impossible to empty both piles -/
theorem cannot_empty_both_piles :
  ¬∃ (ops : List Operation), 
    let finalState := ops.foldl applyOperation initialState
    finalState.pile1 = 0 ∧ finalState.pile2 = 0 :=
sorry

end NUMINAMATH_CALUDE_cannot_empty_both_piles_l2300_230027


namespace NUMINAMATH_CALUDE_hyperbola_mn_value_l2300_230038

/-- Given a hyperbola with equation x²/m - y²/n = 1, eccentricity 2, and one focus at (1,0), prove that mn = 3/16 -/
theorem hyperbola_mn_value (m n : ℝ) (h1 : m * n ≠ 0) :
  (∀ x y : ℝ, x^2 / m - y^2 / n = 1) →  -- Hyperbola equation
  (∃ a b : ℝ, (x - a)^2 / m - (y - b)^2 / n = 1 ∧ ((a + 1)^2 + b^2)^(1/2) = 2) →  -- Eccentricity is 2
  (∃ x y : ℝ, x^2 / m - y^2 / n = 1 ∧ x = 1 ∧ y = 0) →  -- One focus at (1,0)
  m * n = 3/16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_mn_value_l2300_230038


namespace NUMINAMATH_CALUDE_zero_subset_M_l2300_230060

def M : Set ℝ := {x | x > -2}

theorem zero_subset_M : {0} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_zero_subset_M_l2300_230060


namespace NUMINAMATH_CALUDE_function_value_problem_l2300_230068

theorem function_value_problem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (2 * x + 1) = 3 * x - 2) →
  f a = 4 →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_function_value_problem_l2300_230068


namespace NUMINAMATH_CALUDE_sum_quadratic_residues_divisible_l2300_230091

theorem sum_quadratic_residues_divisible (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ s : ℕ, (s > 0) ∧ (s < p) ∧ (∀ x : ℕ, x < p → (∃ y : ℕ, y < p ∧ y^2 ≡ x [ZMOD p]) → s ≡ s + x [ZMOD p]) :=
sorry

end NUMINAMATH_CALUDE_sum_quadratic_residues_divisible_l2300_230091


namespace NUMINAMATH_CALUDE_simplify_fraction_l2300_230062

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2300_230062


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_three_sqrt_one_third_l2300_230054

theorem sqrt_twelve_minus_three_sqrt_one_third (x : ℝ) : 
  Real.sqrt 12 - 3 * Real.sqrt (1/3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_three_sqrt_one_third_l2300_230054


namespace NUMINAMATH_CALUDE_center_number_is_five_l2300_230063

-- Define the 2x3 array type
def Array2x3 := Fin 2 → Fin 3 → Nat

-- Define a predicate for consecutive numbers
def Consecutive (a b : Nat) : Prop := a + 1 = b ∨ b + 1 = a

-- Define diagonal adjacency
def DiagonallyAdjacent (i1 j1 i2 j2 : Nat) : Prop :=
  (i1 + 1 = i2 ∧ j1 + 1 = j2) ∨ (i1 + 1 = i2 ∧ j1 = j2 + 1) ∨
  (i1 = i2 + 1 ∧ j1 + 1 = j2) ∨ (i1 = i2 + 1 ∧ j1 = j2 + 1)

-- Define the property of consecutive numbers being diagonally adjacent
def ConsecutiveAreDiagonallyAdjacent (arr : Array2x3) : Prop :=
  ∀ i1 j1 i2 j2, Consecutive (arr i1 j1) (arr i2 j2) → DiagonallyAdjacent i1 j1 i2 j2

-- Define the property that all numbers from 1 to 5 are present
def ContainsAllNumbers (arr : Array2x3) : Prop :=
  ∀ n, n ≥ 1 ∧ n ≤ 5 → ∃ i j, arr i j = n

-- Define the property that corner numbers on one long side sum to 6
def CornersSum6 (arr : Array2x3) : Prop :=
  (arr 0 0 + arr 0 2 = 6) ∨ (arr 1 0 + arr 1 2 = 6)

-- The main theorem
theorem center_number_is_five (arr : Array2x3) 
  (h1 : ConsecutiveAreDiagonallyAdjacent arr)
  (h2 : ContainsAllNumbers arr)
  (h3 : CornersSum6 arr) :
  (arr 0 1 = 5) ∨ (arr 1 1 = 5) :=
sorry

end NUMINAMATH_CALUDE_center_number_is_five_l2300_230063


namespace NUMINAMATH_CALUDE_remaining_water_bottles_l2300_230099

/-- Calculates the number of remaining water bottles after a soccer match --/
theorem remaining_water_bottles (initial_bottles : ℕ) 
  (first_break_players : ℕ) (first_break_bottles_per_player : ℕ)
  (second_break_players : ℕ) (second_break_extra_bottles : ℕ)
  (third_break_players : ℕ) : 
  initial_bottles = 5 * 12 →
  first_break_players = 11 →
  first_break_bottles_per_player = 2 →
  second_break_players = 14 →
  second_break_extra_bottles = 4 →
  third_break_players = 12 →
  initial_bottles - 
  (first_break_players * first_break_bottles_per_player +
   second_break_players + second_break_extra_bottles +
   third_break_players) = 8 := by
sorry

end NUMINAMATH_CALUDE_remaining_water_bottles_l2300_230099


namespace NUMINAMATH_CALUDE_tan_expression_value_l2300_230057

theorem tan_expression_value (x : ℝ) (h : Real.tan (3 * Real.pi - x) = 2) :
  (2 * (Real.cos (x / 2))^2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_expression_value_l2300_230057


namespace NUMINAMATH_CALUDE_edward_spent_five_on_supplies_l2300_230014

/-- Edward's lawn mowing business finances -/
def lawn_mowing_problem (spring_earnings summer_earnings final_amount : ℤ) : Prop :=
  let total_earnings := spring_earnings + summer_earnings
  let supplies_cost := total_earnings - final_amount
  supplies_cost = 5

/-- Theorem: Edward spent $5 on supplies -/
theorem edward_spent_five_on_supplies :
  lawn_mowing_problem 2 27 24 := by sorry

end NUMINAMATH_CALUDE_edward_spent_five_on_supplies_l2300_230014


namespace NUMINAMATH_CALUDE_ryan_age_problem_l2300_230084

/-- Ryan's age problem -/
theorem ryan_age_problem : ∃ x : ℕ, 
  (∃ n : ℕ, x - 2 = n^3) ∧ 
  (∃ m : ℕ, x + 3 = m^2) ∧ 
  x = 2195 :=
sorry

end NUMINAMATH_CALUDE_ryan_age_problem_l2300_230084


namespace NUMINAMATH_CALUDE_equal_prob_when_four_prob_when_six_l2300_230002

-- Define the set of paper slips
def slips : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the probability of winning for Xiao Ming and Xiao Ying given Xiao Ming's draw
def win_prob (xiao_ming_draw : ℕ) : ℚ × ℚ :=
  let remaining_slips := slips.erase xiao_ming_draw
  let xiao_ming_wins := (remaining_slips.filter (· < xiao_ming_draw)).card
  let xiao_ying_wins := (remaining_slips.filter (· > xiao_ming_draw)).card
  (xiao_ming_wins / remaining_slips.card, xiao_ying_wins / remaining_slips.card)

-- Theorem 1: When Xiao Ming draws 4, both have equal probability of winning
theorem equal_prob_when_four : win_prob 4 = (1/2, 1/2) := by sorry

-- Theorem 2: When Xiao Ming draws 6, probabilities are 5/6 and 1/6
theorem prob_when_six : win_prob 6 = (5/6, 1/6) := by sorry

end NUMINAMATH_CALUDE_equal_prob_when_four_prob_when_six_l2300_230002


namespace NUMINAMATH_CALUDE_all_terms_irrational_l2300_230023

theorem all_terms_irrational (a : ℕ → ℝ) 
  (h_pos : ∀ k, a k > 0)
  (h_rel : ∀ k, (a (k + 1) + k) * a k = 1) :
  ∀ k, Irrational (a k) := by
sorry

end NUMINAMATH_CALUDE_all_terms_irrational_l2300_230023


namespace NUMINAMATH_CALUDE_choir_members_count_l2300_230085

theorem choir_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 1 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l2300_230085


namespace NUMINAMATH_CALUDE_unique_three_digit_perfect_square_product_l2300_230086

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that converts a three-digit number to its cyclic permutations -/
def cyclic_permutations (n : ℕ) : Fin 3 → ℕ
| 0 => n
| 1 => (n % 100) * 10 + n / 100
| 2 => (n % 10) * 100 + n / 10

/-- The main theorem stating that 243 is the only three-digit number satisfying the given conditions -/
theorem unique_three_digit_perfect_square_product :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  (n / 100 ≤ n / 10 % 10 ∧ n / 100 ≤ n % 10) ∧
  is_perfect_square (cyclic_permutations n 0 * cyclic_permutations n 1 * cyclic_permutations n 2) ∧
  n = 243 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_perfect_square_product_l2300_230086


namespace NUMINAMATH_CALUDE_movie_ticket_distribution_l2300_230075

theorem movie_ticket_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (n.descFactorial k) = 720 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_distribution_l2300_230075


namespace NUMINAMATH_CALUDE_maria_carrots_l2300_230035

def carrot_problem (initial_carrots thrown_out_carrots picked_next_day : ℕ) : Prop :=
  initial_carrots - thrown_out_carrots + picked_next_day = 52

theorem maria_carrots : carrot_problem 48 11 15 := by
  sorry

end NUMINAMATH_CALUDE_maria_carrots_l2300_230035


namespace NUMINAMATH_CALUDE_similar_triangle_leg_length_l2300_230093

theorem similar_triangle_leg_length
  (a b c : ℝ)  -- sides of the first triangle
  (d e f : ℝ)  -- sides of the second triangle
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hd : d > 0) (he : e > 0) (hf : f > 0)
  (right_triangle1 : a^2 + b^2 = c^2)  -- first triangle is right triangle
  (right_triangle2 : d^2 + e^2 = f^2)  -- second triangle is right triangle
  (similar : a / d = b / e ∧ b / e = c / f)  -- triangles are similar
  (leg1 : a = 15)  -- one leg of first triangle
  (hyp1 : c = 17)  -- hypotenuse of first triangle
  (hyp2 : f = 51)  -- hypotenuse of second triangle
  : e = 24 :=  -- corresponding leg in second triangle
by sorry

end NUMINAMATH_CALUDE_similar_triangle_leg_length_l2300_230093


namespace NUMINAMATH_CALUDE_sally_grew_113_turnips_l2300_230058

/-- The number of turnips Sally grew -/
def sallys_turnips : ℕ := 113

/-- The number of pumpkins Sally grew -/
def sallys_pumpkins : ℕ := 118

/-- The number of turnips Mary grew -/
def marys_turnips : ℕ := 129

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := 242

/-- Theorem stating that Sally grew 113 turnips -/
theorem sally_grew_113_turnips :
  sallys_turnips = total_turnips - marys_turnips :=
by sorry

end NUMINAMATH_CALUDE_sally_grew_113_turnips_l2300_230058


namespace NUMINAMATH_CALUDE_sin_390_l2300_230031

-- Define the period of the sine function
def sine_period : ℝ := 360

-- Define the periodicity property of sine
axiom sine_periodic (x : ℝ) : Real.sin (x + sine_period) = Real.sin x

-- Define the known value of sin 30°
axiom sin_30 : Real.sin 30 = 1 / 2

-- Theorem to prove
theorem sin_390 : Real.sin 390 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_390_l2300_230031


namespace NUMINAMATH_CALUDE_monthly_income_of_P_l2300_230064

/-- Given the average monthly incomes of three individuals P, Q, and R,
    prove that the monthly income of P is 4000. -/
theorem monthly_income_of_P (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  (P + R) / 2 = 5200 →
  P = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_of_P_l2300_230064


namespace NUMINAMATH_CALUDE_stability_comparison_l2300_230024

/-- Represents an athlete's shooting performance -/
structure Athlete where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines stability of performance based on variance -/
def more_stable (a b : Athlete) : Prop :=
  a.variance < b.variance

theorem stability_comparison 
  (a b : Athlete) 
  (h_same_avg : a.average_score = b.average_score) 
  (h_var_a : a.variance = 0.4) 
  (h_var_b : b.variance = 2) : 
  more_stable a b :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l2300_230024


namespace NUMINAMATH_CALUDE_range_of_a_l2300_230026

/-- A function f(x) that depends on a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 3*a*x + 1

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + 3*a

/-- The discriminant of f'(x) = 0 -/
def discriminant (a : ℝ) : ℝ := 4*a^2 - 36*a

theorem range_of_a (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ min ≤ f a x) →
  (a < 0 ∨ a > 9) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2300_230026


namespace NUMINAMATH_CALUDE_work_completion_time_l2300_230036

theorem work_completion_time (x : ℝ) : 
  x > 0 ∧ 
  5 * (1 / x + 1 / 20) = 1 - 0.41666666666666663 →
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2300_230036


namespace NUMINAMATH_CALUDE_problem_solution_l2300_230073

theorem problem_solution (x : ℝ) (h : 2 * x + 6 = 16) : x + 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2300_230073


namespace NUMINAMATH_CALUDE_percentage_equality_l2300_230016

theorem percentage_equality (x : ℝ) : 
  0.65 * x = 0.20 * 422.50 → x = 130 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l2300_230016


namespace NUMINAMATH_CALUDE_probability_divisible_by_three_l2300_230055

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 15}

def is_divisible_by_three (x y z : ℕ) : Prop :=
  (x * y * z - x * y - y * z - z * x + x + y + z) % 3 = 0

def favorable_outcomes : ℕ := 60

def total_outcomes : ℕ := Nat.choose 15 3

theorem probability_divisible_by_three :
  (favorable_outcomes : ℚ) / total_outcomes = 12 / 91 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_three_l2300_230055


namespace NUMINAMATH_CALUDE_seven_balance_removal_l2300_230012

/-- A function that counts the number of sevens in even positions of a natural number -/
def countSevenEven (n : ℕ) : ℕ := sorry

/-- A function that counts the number of sevens in odd positions of a natural number -/
def countSevenOdd (n : ℕ) : ℕ := sorry

/-- A function that removes the i-th digit from a natural number -/
def removeDigit (n : ℕ) (i : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in a natural number -/
def digitCount (n : ℕ) : ℕ := sorry

theorem seven_balance_removal (n : ℕ) (h : Odd (digitCount n)) :
  ∃ i : ℕ, i < digitCount n ∧ 
    countSevenEven (removeDigit n i) = countSevenOdd (removeDigit n i) := by
  sorry

end NUMINAMATH_CALUDE_seven_balance_removal_l2300_230012


namespace NUMINAMATH_CALUDE_smallest_student_count_l2300_230049

/-- Represents the number of students in each grade --/
structure GradeCount where
  sixth : ℕ
  eighth : ℕ
  ninth : ℕ

/-- Checks if the given counts satisfy the required ratios --/
def satisfiesRatios (counts : GradeCount) : Prop :=
  5 * counts.sixth = 3 * counts.eighth ∧
  7 * counts.ninth = 4 * counts.eighth

/-- The total number of students --/
def totalStudents (counts : GradeCount) : ℕ :=
  counts.sixth + counts.eighth + counts.ninth

/-- Theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (counts : GradeCount),
    satisfiesRatios counts ∧
    totalStudents counts = 76 ∧
    ∀ (other : GradeCount),
      satisfiesRatios other →
      totalStudents other ≥ 76 :=
by sorry

end NUMINAMATH_CALUDE_smallest_student_count_l2300_230049


namespace NUMINAMATH_CALUDE_identity_unique_solution_l2300_230021

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The identity function is the unique solution to the functional equation -/
theorem identity_unique_solution :
  ∃! f : ℝ → ℝ, SatisfiesEquation f ∧ ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_CALUDE_identity_unique_solution_l2300_230021


namespace NUMINAMATH_CALUDE_definite_integral_x_plus_one_squared_ln_squared_l2300_230005

theorem definite_integral_x_plus_one_squared_ln_squared :
  ∫ x in (0:ℝ)..2, (x + 1)^2 * (Real.log (x + 1))^2 = 9 * (Real.log 3)^2 - 6 * Real.log 3 + 79 / 27 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_x_plus_one_squared_ln_squared_l2300_230005


namespace NUMINAMATH_CALUDE_identical_roots_quadratic_l2300_230041

/-- If the quadratic equation 3x^2 - 6x + k = 0 has two identical real roots, then k = 3 -/
theorem identical_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, (3 * x^2 - 6 * x + k = 0) ∧ 
   (∀ y : ℝ, 3 * y^2 - 6 * y + k = 0 → y = x)) → 
  k = 3 := by sorry

end NUMINAMATH_CALUDE_identical_roots_quadratic_l2300_230041


namespace NUMINAMATH_CALUDE_decimal_division_l2300_230000

theorem decimal_division : (0.1 : ℝ) / 0.004 = 25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l2300_230000


namespace NUMINAMATH_CALUDE_team_games_total_l2300_230047

theorem team_games_total (first_games : ℕ) (first_win_rate : ℚ) (remaining_win_rate : ℚ) (total_win_rate : ℚ) : 
  first_games = 30 →
  first_win_rate = 2/5 →
  remaining_win_rate = 4/5 →
  total_win_rate = 3/5 →
  ∃ (total_games : ℕ), total_games = 60 ∧ 
    (first_win_rate * first_games + remaining_win_rate * (total_games - first_games) = total_win_rate * total_games) :=
by
  sorry

#check team_games_total

end NUMINAMATH_CALUDE_team_games_total_l2300_230047


namespace NUMINAMATH_CALUDE_star_associativity_l2300_230032

universe u

variable {U : Type u}

def star (X Y : Set U) : Set U := X ∩ Y

theorem star_associativity (X Y Z : Set U) : star (star X Y) Z = (X ∩ Y) ∩ Z := by
  sorry

end NUMINAMATH_CALUDE_star_associativity_l2300_230032


namespace NUMINAMATH_CALUDE_mobile_phone_purchase_price_l2300_230025

theorem mobile_phone_purchase_price 
  (grinder_price : ℕ) 
  (grinder_loss_percent : ℚ) 
  (mobile_profit_percent : ℚ) 
  (total_profit : ℕ) :
  let mobile_price : ℕ := 8000
  let grinder_sold_price : ℚ := grinder_price * (1 - grinder_loss_percent)
  let mobile_sold_price : ℚ := mobile_price * (1 + mobile_profit_percent)
  grinder_price = 15000 ∧ 
  grinder_loss_percent = 2 / 100 ∧ 
  mobile_profit_percent = 10 / 100 ∧ 
  total_profit = 500 →
  grinder_sold_price + mobile_sold_price = grinder_price + mobile_price + total_profit :=
by sorry

end NUMINAMATH_CALUDE_mobile_phone_purchase_price_l2300_230025


namespace NUMINAMATH_CALUDE_pig_teeth_count_l2300_230061

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of dogs Vann will clean -/
def num_dogs : ℕ := 5

/-- The number of cats Vann will clean -/
def num_cats : ℕ := 10

/-- The number of pigs Vann will clean -/
def num_pigs : ℕ := 7

/-- The total number of teeth Vann will clean -/
def total_teeth : ℕ := 706

/-- Theorem stating that pigs have 28 teeth each -/
theorem pig_teeth_count : 
  (total_teeth - (num_dogs * dog_teeth + num_cats * cat_teeth)) / num_pigs = 28 := by
  sorry

end NUMINAMATH_CALUDE_pig_teeth_count_l2300_230061


namespace NUMINAMATH_CALUDE_area_of_region_l2300_230022

-- Define the circle and chord properties
def circle_radius : ℝ := 50
def chord_length : ℝ := 84
def intersection_distance : ℝ := 24

-- Define the area calculation function
def area_calculation (r : ℝ) (c : ℝ) (d : ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_region :
  area_calculation circle_radius chord_length intersection_distance = 1250 * Real.sqrt 3 + (1250 / 3) * Real.pi :=
sorry

end NUMINAMATH_CALUDE_area_of_region_l2300_230022


namespace NUMINAMATH_CALUDE_max_students_for_given_supplies_l2300_230083

/-- The maximum number of students among whom pens and pencils can be distributed equally -/
def max_students (pens : ℕ) (pencils : ℕ) : ℕ :=
  Nat.gcd pens pencils

/-- Theorem stating that the GCD of 1048 and 828 is the maximum number of students -/
theorem max_students_for_given_supplies : 
  max_students 1048 828 = 4 := by sorry

end NUMINAMATH_CALUDE_max_students_for_given_supplies_l2300_230083


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2300_230089

/-- The polynomial function we're considering -/
def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 5*x - 10

/-- The roots of the polynomial -/
def roots : Set ℝ := {-2, Real.sqrt 5, -Real.sqrt 5}

/-- Theorem stating that the given set contains all roots of the polynomial -/
theorem roots_of_polynomial :
  ∀ x : ℝ, f x = 0 ↔ x ∈ roots := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2300_230089


namespace NUMINAMATH_CALUDE_paint_usage_l2300_230098

/-- Calculates the total amount of paint used by an artist for large and small canvases -/
theorem paint_usage (large_paint : ℕ) (small_paint : ℕ) (large_count : ℕ) (small_count : ℕ) 
  (h1 : large_paint = 3) 
  (h2 : small_paint = 2) 
  (h3 : large_count = 3) 
  (h4 : small_count = 4) : 
  large_paint * large_count + small_paint * small_count = 17 := by
  sorry

#check paint_usage

end NUMINAMATH_CALUDE_paint_usage_l2300_230098


namespace NUMINAMATH_CALUDE_impossible_to_equalize_l2300_230074

/-- Represents the numbers in the six sectors of the circle -/
structure CircleNumbers where
  sectors : Fin 6 → ℤ

/-- Represents an operation of increasing two adjacent numbers by 1 -/
inductive Operation
  | increase_adjacent : Fin 6 → Operation

/-- Applies an operation to the circle numbers -/
def apply_operation (nums : CircleNumbers) (op : Operation) : CircleNumbers :=
  match op with
  | Operation.increase_adjacent i =>
      let j := (i + 1) % 6
      { sectors := fun k =>
          if k = i || k = j then nums.sectors k + 1 else nums.sectors k }

/-- Checks if all numbers in the circle are equal -/
def all_equal (nums : CircleNumbers) : Prop :=
  ∀ i j : Fin 6, nums.sectors i = nums.sectors j

/-- The main theorem stating that it's impossible to make all numbers equal -/
theorem impossible_to_equalize (initial : CircleNumbers) :
  ¬∃ (ops : List Operation), all_equal (ops.foldl apply_operation initial) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_equalize_l2300_230074


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2300_230056

/-- An arithmetic sequence with the given properties has its 7th term equal to 19 -/
theorem arithmetic_sequence_seventh_term (n : ℕ) (a d : ℚ) 
  (h1 : n > 7)
  (h2 : 5 * a + 10 * d = 34)
  (h3 : 5 * a + 5 * (n - 1) * d = 146)
  (h4 : n * (2 * a + (n - 1) * d) / 2 = 234) :
  a + 6 * d = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2300_230056


namespace NUMINAMATH_CALUDE_bus_passengers_l2300_230080

theorem bus_passengers (x : ℕ) (h1 : x ≥ 2) : ∃ n : ℕ,
  n + 5 * (x - 1) = x * n ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_bus_passengers_l2300_230080


namespace NUMINAMATH_CALUDE_floor_tiles_l2300_230001

theorem floor_tiles (black_tiles : ℕ) (total_tiles : ℕ) : 
  black_tiles = 441 → 
  ∃ (side_length : ℕ), 
    side_length * side_length = total_tiles ∧
    side_length = (black_tiles.sqrt : ℕ) + 2 * 3 →
    total_tiles = 729 :=
by sorry

end NUMINAMATH_CALUDE_floor_tiles_l2300_230001


namespace NUMINAMATH_CALUDE_angle_conversion_l2300_230004

theorem angle_conversion (θ : Real) : 
  θ * (π / 180) = -10 * π + 7 * π / 4 → 
  ∃ (k : ℤ) (α : Real), 
    θ * (π / 180) = 2 * k * π + α ∧ 
    0 < α ∧ 
    α < 2 * π :=
by sorry

end NUMINAMATH_CALUDE_angle_conversion_l2300_230004


namespace NUMINAMATH_CALUDE_undecagon_diagonals_l2300_230070

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular undecagon has 11 sides -/
def undecagon_sides : ℕ := 11

/-- Theorem: A regular undecagon (11-sided polygon) has 44 diagonals -/
theorem undecagon_diagonals :
  num_diagonals undecagon_sides = 44 := by sorry

end NUMINAMATH_CALUDE_undecagon_diagonals_l2300_230070


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2300_230088

theorem quadratic_factorization (y : ℝ) : 9*y^2 - 30*y + 25 = (3*y - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2300_230088


namespace NUMINAMATH_CALUDE_lauryns_company_men_count_l2300_230051

/-- The number of men employed by Lauryn's computer company. -/
def num_men : ℕ := 80

/-- The number of women employed by Lauryn's computer company. -/
def num_women : ℕ := num_men + 20

/-- The total number of employees in Lauryn's computer company. -/
def total_employees : ℕ := 180

/-- Theorem stating that the number of men employed by Lauryn is 80,
    given the conditions of the problem. -/
theorem lauryns_company_men_count :
  (num_men + num_women = total_employees) ∧ 
  (num_women = num_men + 20) →
  num_men = 80 := by
  sorry

end NUMINAMATH_CALUDE_lauryns_company_men_count_l2300_230051


namespace NUMINAMATH_CALUDE_jeff_trucks_count_l2300_230044

theorem jeff_trucks_count :
  ∀ (trucks cars : ℕ),
    cars = 2 * trucks →
    trucks + cars = 60 →
    trucks = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_jeff_trucks_count_l2300_230044


namespace NUMINAMATH_CALUDE_letter_distribution_l2300_230039

theorem letter_distribution (n : ℕ) (k : ℕ) : 
  n = 4 ∧ k = 3 → k^n = 81 := by
  sorry

end NUMINAMATH_CALUDE_letter_distribution_l2300_230039


namespace NUMINAMATH_CALUDE_five_by_five_perimeter_l2300_230045

/-- The number of points on the perimeter of a square grid -/
def perimeterPoints (n : ℕ) : ℕ := 4 * n - 4

/-- Theorem: The number of points on the perimeter of a 5x5 grid is 16 -/
theorem five_by_five_perimeter : perimeterPoints 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_by_five_perimeter_l2300_230045


namespace NUMINAMATH_CALUDE_recurring_decimal_equals_fraction_l2300_230067

/-- Represents the decimal expansion 7.836836836... -/
def recurring_decimal : ℚ := 7 + 836 / 999

/-- The fraction representation of the recurring decimal -/
def fraction : ℚ := 7829 / 999

theorem recurring_decimal_equals_fraction :
  recurring_decimal = fraction := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_equals_fraction_l2300_230067


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_max_l2300_230094

theorem triangle_cosine_sum_max (a b c : ℝ) (x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hxyz : x + y + z = π) : 
  (∃ (x y z : ℝ), x + y + z = π ∧ 
    a * Real.cos x + b * Real.cos y + c * Real.cos z ≤ (1/2) * (a*b/c + a*c/b + b*c/a)) :=
sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_max_l2300_230094


namespace NUMINAMATH_CALUDE_largest_n_is_max_l2300_230043

/-- The largest value of n for which 3x^2 + nx + 108 can be factored as the product of two linear factors with integer coefficients -/
def largest_n : ℕ := 325

/-- A polynomial of the form 3x^2 + nx + 108 -/
def polynomial (n : ℕ) (x : ℝ) : ℝ := 3 * x^2 + n * x + 108

/-- Predicate to check if a polynomial can be factored as the product of two linear factors with integer coefficients -/
def can_be_factored (n : ℕ) : Prop :=
  ∃ (a b : ℤ), ∀ (x : ℝ), polynomial n x = (3 * x + a) * (x + b)

/-- Theorem stating that largest_n is the largest value of n for which the polynomial can be factored -/
theorem largest_n_is_max :
  can_be_factored largest_n ∧
  ∀ (m : ℕ), m > largest_n → ¬(can_be_factored m) :=
sorry

end NUMINAMATH_CALUDE_largest_n_is_max_l2300_230043


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l2300_230040

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet, rounding down -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem sculpture_and_base_height :
  let sculpture_height := feet_inches_to_inches 2 10
  let base_height := 2
  let total_height := sculpture_height + base_height
  inches_to_feet total_height = 3 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l2300_230040


namespace NUMINAMATH_CALUDE_problem_solution_l2300_230072

theorem problem_solution (k : ℚ) (h : 3 * k = 10) : (6 / 5) * k - 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2300_230072


namespace NUMINAMATH_CALUDE_complex_sum_value_l2300_230033

theorem complex_sum_value : 
  ∀ (c d : ℂ), c = 3 + 2*I ∧ d = 1 - 2*I → 3*c + 4*d = 13 - 2*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_value_l2300_230033


namespace NUMINAMATH_CALUDE_amoeba_count_after_week_l2300_230019

/-- The number of amoebas in the puddle on a given day -/
def amoeba_count (day : ℕ) : ℕ :=
  if day = 0 then 0
  else if day = 1 then 1
  else 2 * amoeba_count (day - 1)

/-- The theorem stating that after 7 days, there are 64 amoebas in the puddle -/
theorem amoeba_count_after_week : amoeba_count 7 = 64 := by
  sorry

#eval amoeba_count 7  -- This should output 64

end NUMINAMATH_CALUDE_amoeba_count_after_week_l2300_230019


namespace NUMINAMATH_CALUDE_special_number_theorem_l2300_230053

def is_nine_digit_number (n : ℕ) : Prop :=
  100000000 ≤ n ∧ n ≤ 999999999

def has_special_form (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    a < 1000 ∧ b < 1000 ∧
    n = a * 1000000 + b * 1000 + a

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    a < 1000 ∧ b < 1000 ∧
    n = a * 1000000 + b * 1000 + a ∧
    b = 2 * a

def is_product_of_five_primes_squared (n : ℕ) : Prop :=
  ∃ (p q r s t : ℕ),
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ Nat.Prime t ∧
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
    q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
    r ≠ s ∧ r ≠ t ∧
    s ≠ t ∧
    n = (p * q * r * s * t) ^ 2

theorem special_number_theorem (n : ℕ) :
  is_nine_digit_number n ∧
  has_special_form n ∧
  satisfies_condition n ∧
  is_product_of_five_primes_squared n →
  n = 100200100 ∨ n = 225450225 := by
sorry

end NUMINAMATH_CALUDE_special_number_theorem_l2300_230053


namespace NUMINAMATH_CALUDE_total_cost_usd_l2300_230037

/-- The cost of items in British pounds and US dollars -/
def cost_in_usd (tea_gbp : ℝ) (scone_gbp : ℝ) (exchange_rate : ℝ) : ℝ :=
  (tea_gbp + scone_gbp) * exchange_rate

/-- Theorem: The total cost in USD for a tea and a scone is $10.80 -/
theorem total_cost_usd :
  cost_in_usd 5 3 1.35 = 10.80 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_usd_l2300_230037


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l2300_230090

/-- Proves that under given conditions, the cost increase percentage is 25% -/
theorem cost_increase_percentage (C : ℝ) (P : ℝ) : 
  C > 0 → -- Ensure cost is positive
  let S := 4.2 * C -- Original selling price
  let new_profit := 0.7023809523809523 * S -- New profit after cost increase
  3.2 * C - (P / 100) * C = new_profit → -- Equation relating new profit to cost increase
  P = 25 := by
  sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l2300_230090


namespace NUMINAMATH_CALUDE_roots_equation_value_l2300_230048

theorem roots_equation_value (α β : ℝ) : 
  (α^2 + 2*α - 1 = 0) → 
  (β^2 + 2*β - 1 = 0) → 
  α^2 + 3*α + β = -1 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_value_l2300_230048


namespace NUMINAMATH_CALUDE_salary_calculation_l2300_230034

theorem salary_calculation (salary : ℝ) 
  (food_expense : salary * (1 / 5) = salary / 5)
  (rent_expense : salary * (1 / 10) = salary / 10)
  (clothes_expense : salary * (3 / 5) = 3 * salary / 5)
  (remaining : salary - (salary / 5 + salary / 10 + 3 * salary / 5) = 14000) :
  salary = 140000 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l2300_230034


namespace NUMINAMATH_CALUDE_sum_of_recorded_products_25_coins_l2300_230020

/-- Represents the process of dividing coins into groups and recording products. -/
def divide_coins (n : ℕ) : ℕ := 
  (n * (n - 1)) / 2

/-- The theorem stating that the sum of recorded products when dividing 25 coins is 300. -/
theorem sum_of_recorded_products_25_coins : 
  divide_coins 25 = 300 := by sorry

end NUMINAMATH_CALUDE_sum_of_recorded_products_25_coins_l2300_230020


namespace NUMINAMATH_CALUDE_hen_count_l2300_230018

theorem hen_count (total_heads : ℕ) (total_feet : ℕ) (hen_heads : ℕ) (hen_feet : ℕ) (cow_heads : ℕ) (cow_feet : ℕ) 
  (h1 : total_heads = 44)
  (h2 : total_feet = 128)
  (h3 : hen_heads = 1)
  (h4 : hen_feet = 2)
  (h5 : cow_heads = 1)
  (h6 : cow_feet = 4) :
  ∃ (num_hens : ℕ), num_hens = 24 ∧ 
    num_hens * hen_heads + (total_heads - num_hens) * cow_heads = total_heads ∧
    num_hens * hen_feet + (total_heads - num_hens) * cow_feet = total_feet :=
by sorry

end NUMINAMATH_CALUDE_hen_count_l2300_230018


namespace NUMINAMATH_CALUDE_bucket_weight_l2300_230096

theorem bucket_weight (c d : ℝ) : ℝ :=
  let weight_three_quarters : ℝ := c
  let weight_one_third : ℝ := d
  let weight_full : ℝ := (8 * c / 5) - (3 * d / 5)
  weight_full

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l2300_230096


namespace NUMINAMATH_CALUDE_apple_profit_calculation_l2300_230011

/-- Profit percentage for the first half of apples -/
def P : ℝ := sorry

/-- Cost price of 1 kg of apples -/
def C : ℝ := sorry

theorem apple_profit_calculation :
  (50 * C + 50 * C * (P / 100) + 50 * C + 50 * C * (30 / 100) = 100 * C + 100 * C * (27.5 / 100)) →
  P = 25 := by
  sorry

end NUMINAMATH_CALUDE_apple_profit_calculation_l2300_230011


namespace NUMINAMATH_CALUDE_huron_michigan_fishes_l2300_230007

def total_fishes : ℕ := 97
def ontario_erie_fishes : ℕ := 23
def superior_fishes : ℕ := 44

theorem huron_michigan_fishes :
  total_fishes - (ontario_erie_fishes + superior_fishes) = 30 := by
  sorry

end NUMINAMATH_CALUDE_huron_michigan_fishes_l2300_230007


namespace NUMINAMATH_CALUDE_fourth_part_diminished_l2300_230006

theorem fourth_part_diminished (x : ℝ) (y : ℝ) (h : x = 160) (h2 : (x / 5) + 4 = (x / 4) - y) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_part_diminished_l2300_230006


namespace NUMINAMATH_CALUDE_stratified_sample_distribution_l2300_230066

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the total number of students -/
def totalStudents (d : GradeDistribution) : ℕ :=
  d.grade10 + d.grade11 + d.grade12

/-- Represents the sample size for each grade -/
structure SampleDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the total sample size -/
def totalSample (s : SampleDistribution) : ℕ :=
  s.grade10 + s.grade11 + s.grade12

theorem stratified_sample_distribution 
  (population : GradeDistribution)
  (sample : SampleDistribution) :
  totalStudents population = 4000 →
  population.grade10 = 32 * k →
  population.grade11 = 33 * k →
  population.grade12 = 35 * k →
  totalSample sample = 200 →
  sample.grade10 = 64 ∧ sample.grade11 = 66 ∧ sample.grade12 = 70 :=
by sorry


end NUMINAMATH_CALUDE_stratified_sample_distribution_l2300_230066


namespace NUMINAMATH_CALUDE_percentage_calculation_l2300_230052

theorem percentage_calculation (x : ℝ) :
  (30 / 100) * ((60 / 100) * ((70 / 100) * x)) = (126 / 1000) * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2300_230052
