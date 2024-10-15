import Mathlib

namespace NUMINAMATH_CALUDE_difference_of_squares_and_sum_l2262_226279

theorem difference_of_squares_and_sum (m n : ℤ) 
  (h1 : m^2 - n^2 = 6) 
  (h2 : m + n = 3) : 
  n - m = -2 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_and_sum_l2262_226279


namespace NUMINAMATH_CALUDE_quadratic_equation_nonnegative_solutions_l2262_226263

theorem quadratic_equation_nonnegative_solutions :
  ∃! x : ℝ, x ≥ 0 ∧ x^2 = -6*x :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_nonnegative_solutions_l2262_226263


namespace NUMINAMATH_CALUDE_grid_with_sequence_exists_l2262_226273

-- Define the grid type
def Grid := Matrix (Fin 6) (Fin 6) (Fin 4)

-- Define a predicate for valid subgrids
def valid_subgrid (g : Grid) (i j : Fin 2) : Prop :=
  ∀ n : Fin 4, ∃! x y : Fin 2, g (2 * i + x) (2 * j + y) = n

-- Define a predicate for adjacent cells being different
def adjacent_different (g : Grid) : Prop :=
  ∀ i j i' j' : Fin 6, 
    (i = i' ∧ |j - j'| = 1) ∨ 
    (j = j' ∧ |i - i'| = 1) ∨ 
    (|i - i'| = 1 ∧ |j - j'| = 1) → 
    g i j ≠ g i' j'

-- Define the existence of the sequence 3521 in the grid
def sequence_exists (g : Grid) : Prop :=
  ∃ i₁ j₁ i₂ j₂ i₃ j₃ i₄ j₄ : Fin 6,
    g i₁ j₁ = 3 ∧ g i₂ j₂ = 5 ∧ g i₃ j₃ = 2 ∧ g i₄ j₄ = 1

-- The main theorem
theorem grid_with_sequence_exists : 
  ∃ g : Grid, 
    (∀ i j : Fin 2, valid_subgrid g i j) ∧ 
    adjacent_different g ∧
    sequence_exists g :=
sorry

end NUMINAMATH_CALUDE_grid_with_sequence_exists_l2262_226273


namespace NUMINAMATH_CALUDE_calc_difference_l2262_226283

-- Define the correct calculation (Mark's method)
def correct_calc : ℤ := 12 - (3 + 6)

-- Define the incorrect calculation (Jane's method)
def incorrect_calc : ℤ := 12 - 3 + 6

-- Theorem statement
theorem calc_difference : correct_calc - incorrect_calc = -12 := by
  sorry

end NUMINAMATH_CALUDE_calc_difference_l2262_226283


namespace NUMINAMATH_CALUDE_total_net_amount_is_218_l2262_226232

/-- Represents a lottery ticket with its cost and number of winning numbers -/
structure LotteryTicket where
  cost : ℕ
  winningNumbers : ℕ

/-- Calculates the payout for a single ticket based on its winning numbers -/
def calculatePayout (ticket : LotteryTicket) : ℕ :=
  if ticket.winningNumbers ≤ 2 then
    ticket.winningNumbers * 15
  else
    30 + (ticket.winningNumbers - 2) * 20

/-- Calculates the net amount won for a single ticket -/
def calculateNetAmount (ticket : LotteryTicket) : ℤ :=
  (calculatePayout ticket : ℤ) - ticket.cost

/-- The set of lottery tickets Tony bought -/
def tonyTickets : List LotteryTicket := [
  ⟨5, 3⟩,
  ⟨7, 5⟩,
  ⟨4, 2⟩,
  ⟨6, 4⟩
]

/-- Theorem stating that the total net amount Tony won is $218 -/
theorem total_net_amount_is_218 :
  (tonyTickets.map calculateNetAmount).sum = 218 := by
  sorry

end NUMINAMATH_CALUDE_total_net_amount_is_218_l2262_226232


namespace NUMINAMATH_CALUDE_number_division_proof_l2262_226275

theorem number_division_proof (x : ℝ) : 4 * x = 166.08 → x / 4 = 10.38 := by
  sorry

end NUMINAMATH_CALUDE_number_division_proof_l2262_226275


namespace NUMINAMATH_CALUDE_prime_divisor_congruent_to_one_l2262_226239

theorem prime_divisor_congruent_to_one (p : ℕ) (hp : Prime p) :
  ∃ q : ℕ, Prime q ∧ q ∣ (p^p - 1) ∧ q % p = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_congruent_to_one_l2262_226239


namespace NUMINAMATH_CALUDE_cone_surface_area_l2262_226212

theorem cone_surface_area (θ : Real) (S_lateral : Real) (S_total : Real) : 
  θ = 2 * Real.pi / 3 →  -- 120° in radians
  S_lateral = 3 * Real.pi →
  S_total = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l2262_226212


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2262_226248

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b + 1 → a > b) ∧
  (∃ a b, a > b ∧ ¬(a > b + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2262_226248


namespace NUMINAMATH_CALUDE_cube_sum_geq_mixed_terms_l2262_226228

theorem cube_sum_geq_mixed_terms (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^3 + y^3 ≥ x^2*y + x*y^2 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_geq_mixed_terms_l2262_226228


namespace NUMINAMATH_CALUDE_solution_value_l2262_226251

theorem solution_value (x m : ℤ) : 
  x = -2 → 3 * x + 5 = x - m → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2262_226251


namespace NUMINAMATH_CALUDE_curve_C_equation_m_equilateral_triangle_m_range_dot_product_negative_l2262_226208

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > 0 ∧ p.1^2 = 4 * p.2}

-- Define point F
def F : ℝ × ℝ := (0, 1)

-- Define the line y = -2
def line_y_neg_2 (x : ℝ) : ℝ := -2

-- Define the distance condition
def distance_condition (p : ℝ × ℝ) : Prop :=
  Real.sqrt ((p.1 - F.1)^2 + (p.2 - F.2)^2) + 1 = p.2 - line_y_neg_2 p.1

-- Define the theorem for the equation of curve C
theorem curve_C_equation :
  ∀ p : ℝ × ℝ, p ∈ C ↔ p.2 > 0 ∧ p.1^2 = 4 * p.2 :=
sorry

-- Define the theorem for the value of m when triangle AFB is equilateral
theorem m_equilateral_triangle :
  ∀ m : ℝ, (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧
    (A.2 = B.2) ∧ (A.1 = -B.1) ∧
    Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) ∧
    Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  (m = 7 + 4 * Real.sqrt 3 ∨ m = 7 - 4 * Real.sqrt 3) :=
sorry

-- Define the theorem for the range of m when FA · FB < 0
theorem m_range_dot_product_negative :
  ∀ m : ℝ, (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧
    ((A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) < 0)) →
  (3 - 2 * Real.sqrt 2 < m ∧ m < 3 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_curve_C_equation_m_equilateral_triangle_m_range_dot_product_negative_l2262_226208


namespace NUMINAMATH_CALUDE_base5_division_l2262_226216

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Theorem: The quotient of 2314₅ divided by 21₅ is equal to 110₅ -/
theorem base5_division :
  base10ToBase5 (base5ToBase10 [4, 1, 3, 2] / base5ToBase10 [1, 2]) = [0, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_base5_division_l2262_226216


namespace NUMINAMATH_CALUDE_henry_games_count_henry_games_count_proof_l2262_226265

theorem henry_games_count : ℕ → ℕ → ℕ → Prop :=
  fun initial_neil initial_henry games_given =>
    -- Neil's initial games count
    let initial_neil_games := 7

    -- Henry initially had 3 times more games than Neil
    (initial_henry = 3 * initial_neil_games + initial_neil_games) →
    
    -- After giving Neil 6 games, Henry has 4 times more games than Neil
    (initial_henry - games_given = 4 * (initial_neil_games + games_given)) →
    
    -- The number of games given to Neil
    (games_given = 6) →
    
    -- Conclusion: Henry's initial game count
    initial_henry = 58

-- The proof of the theorem
theorem henry_games_count_proof : henry_games_count 7 58 6 := by
  sorry

end NUMINAMATH_CALUDE_henry_games_count_henry_games_count_proof_l2262_226265


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2262_226257

theorem inequality_equivalence (x : ℝ) : (x + 1) * (2 - x) > 0 ↔ x ∈ Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2262_226257


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2262_226284

def I : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 4, 5}
def B : Set Nat := {1, 4}

theorem intersection_A_complement_B :
  A ∩ (I \ B) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2262_226284


namespace NUMINAMATH_CALUDE_smallest_circle_equation_l2262_226237

/-- A parabola with equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola y^2 = 4x -/
def Focus : ℝ × ℝ := (1, 0)

/-- A circle with center on the parabola and passing through the focus -/
def CircleOnParabola (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1 - Focus.1)^2 + (center.2 - Focus.2)^2}

/-- The theorem stating that the circle with smallest radius has equation x^2 + y^2 = 1 -/
theorem smallest_circle_equation :
  ∃ (center : ℝ × ℝ),
    center ∈ Parabola ∧
    Focus ∈ CircleOnParabola center ∧
    (∀ (other_center : ℝ × ℝ),
      other_center ∈ Parabola →
      Focus ∈ CircleOnParabola other_center →
      (center.1 - Focus.1)^2 + (center.2 - Focus.2)^2 ≤ (other_center.1 - Focus.1)^2 + (other_center.2 - Focus.2)^2) →
    CircleOnParabola center = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_equation_l2262_226237


namespace NUMINAMATH_CALUDE_derivative_f_at_neg_two_l2262_226274

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem derivative_f_at_neg_two :
  (deriv f) (-2) = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_neg_two_l2262_226274


namespace NUMINAMATH_CALUDE_short_trees_after_planting_l2262_226241

/-- The number of short trees in the park after planting -/
def total_short_trees (
  initial_short_oak : ℕ)
  (initial_short_pine : ℕ)
  (initial_short_maple : ℕ)
  (new_short_oak : ℕ)
  (new_short_pine : ℕ)
  (new_short_maple : ℕ) : ℕ :=
  initial_short_oak + initial_short_pine + initial_short_maple +
  new_short_oak + new_short_pine + new_short_maple

/-- Theorem stating the total number of short trees after planting -/
theorem short_trees_after_planting :
  total_short_trees 3 4 5 9 6 4 = 31 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_after_planting_l2262_226241


namespace NUMINAMATH_CALUDE_complex_division_l2262_226266

theorem complex_division (z : ℂ) (h1 : z.re = 1) (h2 : z.im = -2) : 
  (5 * Complex.I) / z = -2 + Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_division_l2262_226266


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l2262_226226

theorem fruit_basket_problem (total_fruits : ℕ) 
  (basket_A basket_B : ℕ) 
  (apples_A pears_A apples_B pears_B : ℕ) :
  total_fruits = 82 →
  (basket_A + basket_B = total_fruits) →
  (basket_A ≥ basket_B → basket_A - basket_B < 10) →
  (basket_B > basket_A → basket_B - basket_A < 10) →
  (5 * apples_A = 2 * basket_A) →
  (7 * pears_B = 4 * basket_B) →
  (basket_A = apples_A + pears_A) →
  (basket_B = apples_B + pears_B) →
  (pears_A = 24 ∧ apples_B = 18) :=
by sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l2262_226226


namespace NUMINAMATH_CALUDE_largest_integer_in_inequality_l2262_226238

theorem largest_integer_in_inequality : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 7/11 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/11) → y ≤ x :=
by
  use 4
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_inequality_l2262_226238


namespace NUMINAMATH_CALUDE_a_value_l2262_226294

def P (a : ℝ) : Set ℝ := {1, 2, a}
def Q : Set ℝ := {x | x^2 - 9 = 0}

theorem a_value (a : ℝ) : P a ∩ Q = {3} → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l2262_226294


namespace NUMINAMATH_CALUDE_diophantine_equation_prime_divisor_l2262_226218

theorem diophantine_equation_prime_divisor (b : ℕ+) (h : Nat.gcd b.val 6 = 1) :
  (∃ (x y : ℕ+), (1 : ℚ) / x.val + (1 : ℚ) / y.val = (3 : ℚ) / b.val) ↔
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ b.val ∧ ∃ (k : ℕ), p = 6 * k - 1 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_prime_divisor_l2262_226218


namespace NUMINAMATH_CALUDE_necessary_condition_for_greater_than_not_sufficient_condition_for_greater_than_l2262_226243

theorem necessary_condition_for_greater_than (a b : ℝ) :
  (a > b) → (a + 1 > b) :=
sorry

theorem not_sufficient_condition_for_greater_than :
  ∃ (a b : ℝ), (a + 1 > b) ∧ ¬(a > b) :=
sorry

end NUMINAMATH_CALUDE_necessary_condition_for_greater_than_not_sufficient_condition_for_greater_than_l2262_226243


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_problem_solution_l2262_226280

theorem odd_even_sum_difference : ℕ → Prop :=
  fun n =>
    let odd_sum := (n^2 + n) / 2
    let even_sum := n * (n + 1)
    odd_sum - even_sum = n + 1

theorem problem_solution :
  let n : ℕ := 1009
  let odd_sum := ((2*n + 1)^2 + (2*n + 1)) / 2
  let even_sum := n * (n + 1)
  odd_sum - even_sum = 1010 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_problem_solution_l2262_226280


namespace NUMINAMATH_CALUDE_total_toys_l2262_226215

theorem total_toys (num_dolls : ℕ) (h1 : num_dolls = 18) : ℕ :=
  let total := 4 * num_dolls / 3
  have h2 : total = 24 := by sorry
  total

#check total_toys

end NUMINAMATH_CALUDE_total_toys_l2262_226215


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l2262_226210

theorem least_five_digit_square_cube : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  (∃ a : ℕ, n = a^2) ∧ 
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, m < n → ¬(m ≥ 10000 ∧ m < 100000 ∧ (∃ x : ℕ, m = x^2) ∧ (∃ y : ℕ, m = y^3))) ∧
  n = 15625 := by
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l2262_226210


namespace NUMINAMATH_CALUDE_min_a_correct_l2262_226233

/-- The number of cards in the deck -/
def n : ℕ := 51

/-- The probability that Alex and Dylan are on the same team given that Alex picks one of the cards a and a+7, and Dylan picks the other -/
def p (a : ℕ) : ℚ :=
  (Nat.choose (42 - a) 2 + Nat.choose (a - 1) 2) / Nat.choose (n - 2) 2

/-- The minimum value of a for which p(a) ≥ 1/2 -/
def min_a : ℕ := 22

theorem min_a_correct :
  (∀ a : ℕ, 1 ≤ a ∧ a + 7 ≤ n → p a ≥ 1/2 → a ≥ min_a) ∧
  p min_a ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_min_a_correct_l2262_226233


namespace NUMINAMATH_CALUDE_good_pair_exists_l2262_226278

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ 
  ∃ a b : ℕ, m * n = a ^ 2 ∧ (m + 1) * (n + 1) = b ^ 2 := by
  let n := m * (4 * m + 3) ^ 2
  have h1 : n > m := sorry
  have h2 : ∃ a : ℕ, m * n = a ^ 2 := sorry
  have h3 : ∃ b : ℕ, (m + 1) * (n + 1) = b ^ 2 := sorry
  exact ⟨n, h1, h2.choose, h3.choose, h2.choose_spec, h3.choose_spec⟩

end NUMINAMATH_CALUDE_good_pair_exists_l2262_226278


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l2262_226236

/-- A geometric sequence with positive terms and common ratio greater than 1 -/
structure GeometricSequence where
  b : ℕ → ℝ
  positive : ∀ n, b n > 0
  q : ℝ
  q_gt_one : q > 1
  geometric : ∀ n, b (n + 1) = q * b n

/-- The inequality holds for the 4th, 5th, 7th, and 8th terms of the geometric sequence -/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  seq.b 4 * seq.b 8 > seq.b 5 * seq.b 7 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l2262_226236


namespace NUMINAMATH_CALUDE_new_person_weight_is_106_l2262_226234

/-- The number of persons in the initial group -/
def initial_group_size : ℕ := 12

/-- The increase in average weight when the new person joins (in kg) -/
def average_weight_increase : ℝ := 4

/-- The weight of the person being replaced (in kg) -/
def replaced_person_weight : ℝ := 58

/-- The weight of the new person (in kg) -/
def new_person_weight : ℝ := 106

/-- Theorem stating that the weight of the new person is 106 kg -/
theorem new_person_weight_is_106 :
  new_person_weight = replaced_person_weight + initial_group_size * average_weight_increase :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_is_106_l2262_226234


namespace NUMINAMATH_CALUDE_shopping_trip_theorem_l2262_226254

/-- Shopping Trip Theorem -/
theorem shopping_trip_theorem (initial_amount : ℕ) (shoe_cost : ℕ) :
  initial_amount = 158 →
  shoe_cost = 45 →
  let bag_cost := shoe_cost - 17
  let lunch_cost := bag_cost / 4
  let total_spent := shoe_cost + bag_cost + lunch_cost
  initial_amount - total_spent = 78 := by
sorry

end NUMINAMATH_CALUDE_shopping_trip_theorem_l2262_226254


namespace NUMINAMATH_CALUDE_theater_performance_duration_l2262_226220

/-- The duration of a theater performance in hours -/
def performance_duration : ℝ := 3

/-- The number of weeks Mark visits the theater -/
def weeks : ℕ := 6

/-- The price per hour for a theater ticket in dollars -/
def price_per_hour : ℝ := 5

/-- The total amount spent on theater visits in dollars -/
def total_spent : ℝ := 90

theorem theater_performance_duration :
  performance_duration * price_per_hour * weeks = total_spent :=
by sorry

end NUMINAMATH_CALUDE_theater_performance_duration_l2262_226220


namespace NUMINAMATH_CALUDE_polygon_diagonals_sides_l2262_226203

theorem polygon_diagonals_sides (n : ℕ) : n > 2 →
  (n * (n - 3) / 2 = 2 * n) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_sides_l2262_226203


namespace NUMINAMATH_CALUDE_union_of_A_and_B_when_m_is_3_necessary_but_not_sufficient_condition_l2262_226296

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 2*m + 1}

-- Part 1
theorem union_of_A_and_B_when_m_is_3 :
  A ∪ B 3 = {x : ℝ | -1 ≤ x ∧ x < 7} := by sorry

-- Part 2
theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m) ↔ m ≤ -2 ∨ (0 ≤ m ∧ m ≤ 1) := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_when_m_is_3_necessary_but_not_sufficient_condition_l2262_226296


namespace NUMINAMATH_CALUDE_min_sum_of_indices_l2262_226250

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem min_sum_of_indices (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n : ℕ, a m + a n = 4 * a 1) →
  ∃ m n : ℕ, a m + a n = 4 * a 1 ∧ m + n = 4 ∧ ∀ k l : ℕ, a k + a l = 4 * a 1 → k + l ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_indices_l2262_226250


namespace NUMINAMATH_CALUDE_sum_of_squares_l2262_226292

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 28) : x^2 + y^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2262_226292


namespace NUMINAMATH_CALUDE_average_first_10_even_numbers_l2262_226272

theorem average_first_10_even_numbers : 
  let first_10_even : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
  (first_10_even.sum / first_10_even.length : ℚ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_first_10_even_numbers_l2262_226272


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2262_226201

theorem complex_equation_sum (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b - Complex.I →
  a + b = 1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2262_226201


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2262_226206

/-- Convert a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Convert a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : List (Fin 4) := sorry

/-- The binary representation of 110111001₂ -/
def binary_number : List Bool := [true, true, false, true, true, true, false, false, true]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_number) = [1, 3, 2, 2, 1] := by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2262_226206


namespace NUMINAMATH_CALUDE_inscribe_two_equal_circles_l2262_226259

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle is tangent to a line segment --/
def isTangentToSide (c : Circle) (p1 p2 : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if two circles are tangent to each other --/
def areTangent (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a circle is inside a triangle --/
def isInside (c : Circle) (t : Triangle) : Prop := sorry

/-- Theorem stating that two equal circles can be inscribed in any triangle,
    each touching two sides of the triangle and the other circle --/
theorem inscribe_two_equal_circles (t : Triangle) : 
  ∃ (c1 c2 : Circle), 
    c1.radius = c2.radius ∧ 
    isInside c1 t ∧ 
    isInside c2 t ∧ 
    (isTangentToSide c1 t.A t.B ∨ isTangentToSide c1 t.B t.C ∨ isTangentToSide c1 t.C t.A) ∧
    (isTangentToSide c1 t.A t.B ∨ isTangentToSide c1 t.B t.C ∨ isTangentToSide c1 t.C t.A) ∧
    (isTangentToSide c2 t.A t.B ∨ isTangentToSide c2 t.B t.C ∨ isTangentToSide c2 t.C t.A) ∧
    (isTangentToSide c2 t.A t.B ∨ isTangentToSide c2 t.B t.C ∨ isTangentToSide c2 t.C t.A) ∧
    areTangent c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_inscribe_two_equal_circles_l2262_226259


namespace NUMINAMATH_CALUDE_range_f_minus_g_theorem_l2262_226253

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define the range of f + g
def range_f_plus_g (f g : ℝ → ℝ) : Set ℝ := Set.range (λ x ↦ f x + g x)

-- Define the range of f - g
def range_f_minus_g (f g : ℝ → ℝ) : Set ℝ := Set.range (λ x ↦ f x - g x)

-- State the theorem
theorem range_f_minus_g_theorem (hf : is_odd f) (hg : is_even g) 
  (h_range : range_f_plus_g f g = Set.Icc 1 3) : 
  range_f_minus_g f g = Set.Ioc (-3) (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_f_minus_g_theorem_l2262_226253


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l2262_226224

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l2262_226224


namespace NUMINAMATH_CALUDE_sequence_length_is_751_l2262_226290

/-- Given a sequence of real numbers satisfying certain conditions, prove that the length of the sequence is 751. -/
theorem sequence_length_is_751 (n : ℕ) (b : ℕ → ℝ) 
  (h_pos : n > 0)
  (h_b0 : b 0 = 40)
  (h_b1 : b 1 = 75)
  (h_bn : b n = 0)
  (h_rec : ∀ k, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 4 / b k) :
  n = 751 := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_is_751_l2262_226290


namespace NUMINAMATH_CALUDE_air_quality_probabilities_l2262_226211

def prob_grade_A : ℝ := 0.8
def prob_grade_B : ℝ := 0.1
def prob_grade_C : ℝ := 0.1

def prob_satisfactory (p_A p_B p_C : ℝ) : ℝ :=
  p_A * p_A + 2 * p_A * (1 - p_A)

def prob_two_out_of_three (p : ℝ) : ℝ :=
  3 * p * p * (1 - p)

theorem air_quality_probabilities :
  prob_satisfactory prob_grade_A prob_grade_B prob_grade_C = 0.96 ∧
  prob_two_out_of_three (prob_satisfactory prob_grade_A prob_grade_B prob_grade_C) = 0.110592 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probabilities_l2262_226211


namespace NUMINAMATH_CALUDE_exists_positive_c_less_than_sum_l2262_226287

theorem exists_positive_c_less_than_sum (a b : ℝ) (h : a < b) :
  ∃ c : ℝ, c > 0 ∧ a < b + c := by
sorry

end NUMINAMATH_CALUDE_exists_positive_c_less_than_sum_l2262_226287


namespace NUMINAMATH_CALUDE_pqr_value_l2262_226242

theorem pqr_value (a b c p q r : ℂ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h_a : a = (b + c) / (p - 3))
  (h_b : b = (a + c) / (q - 3))
  (h_c : c = (a + b) / (r - 3))
  (h_sum_prod : p * q + p * r + q * r = 10)
  (h_sum : p + q + r = 6) :
  p * q * r = 14 := by
sorry

end NUMINAMATH_CALUDE_pqr_value_l2262_226242


namespace NUMINAMATH_CALUDE_quadratic_radicals_sum_product_l2262_226207

theorem quadratic_radicals_sum_product (a b c d e : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (he : 0 ≤ e)
  (h1 : a = 3) (h2 : b = 5) (h3 : c = 7) (h4 : d = 9) (h5 : e = 11) :
  (Real.sqrt a - 1 + Real.sqrt b - Real.sqrt a + Real.sqrt c - Real.sqrt b + 
   Real.sqrt d - Real.sqrt c + Real.sqrt e - Real.sqrt d) * 
  (Real.sqrt e + 1) = 10 := by
  sorry

-- Additional lemmas to represent the given conditions
lemma quadratic_radical_diff (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  2 / (Real.sqrt a + Real.sqrt b) = Real.sqrt a - Real.sqrt b := by
  sorry

lemma quadratic_radical_sum (a : ℝ) (ha : 0 ≤ a) :
  Real.sqrt (a + 2 * Real.sqrt (a - 1)) = Real.sqrt a + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_sum_product_l2262_226207


namespace NUMINAMATH_CALUDE_point_move_left_l2262_226264

def number_line_move (initial_position : ℝ) (move_distance : ℝ) : ℝ :=
  initial_position - move_distance

theorem point_move_left :
  let initial_position : ℝ := -4
  let move_distance : ℝ := 2
  number_line_move initial_position move_distance = -6 := by
sorry

end NUMINAMATH_CALUDE_point_move_left_l2262_226264


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2262_226227

theorem floor_ceil_sum : ⌊(1.002 : ℝ)⌋ + ⌈(3.998 : ℝ)⌉ + ⌈(-0.999 : ℝ)⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2262_226227


namespace NUMINAMATH_CALUDE_equation_solution_l2262_226213

theorem equation_solution : ∃! x : ℝ, 5 * x - 3 * x = 210 + 6 * (x + 4) ∧ x = -58.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2262_226213


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l2262_226245

/-- Represents a quadratic equation of the form ax^2 - 3x√3 + b = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ

/-- The discriminant of the quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ := 27 - 4 * eq.a * eq.b

/-- Predicate for real and distinct roots -/
def has_real_distinct_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq ≠ 0 ∧ discriminant eq > 0

theorem quadratic_roots_nature (eq : QuadraticEquation) 
  (h : discriminant eq ≠ 0) : 
  has_real_distinct_roots eq :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l2262_226245


namespace NUMINAMATH_CALUDE_paired_with_32_l2262_226205

def numbers : List ℕ := [36, 27, 42, 32, 28, 31, 23, 17]

theorem paired_with_32 (pair_sum : ℕ) 
  (h1 : pair_sum = (numbers.sum / 4))
  (h2 : ∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → a + b = pair_sum → a ≠ b)
  (h3 : ∀ (n : ℕ), n ∈ numbers → ∃ (m : ℕ), m ∈ numbers ∧ m ≠ n ∧ n + m = pair_sum) :
  ∃ (pairs : List (ℕ × ℕ)), 
    pairs.length = 4 ∧ 
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 = pair_sum) ∧
    (32, 27) ∈ pairs :=
sorry

end NUMINAMATH_CALUDE_paired_with_32_l2262_226205


namespace NUMINAMATH_CALUDE_minimum_laptops_l2262_226281

theorem minimum_laptops (n p : ℕ) (h1 : n > 3) (h2 : p > 0) : 
  (p / n + (n - 3) * (p / n + 15) - p = 105) → n ≥ 10 :=
by
  sorry

#check minimum_laptops

end NUMINAMATH_CALUDE_minimum_laptops_l2262_226281


namespace NUMINAMATH_CALUDE_total_insects_l2262_226256

def insect_collection (R S C P B E : ℕ) : Prop :=
  R = 15 ∧
  S = 2 * R - 8 ∧
  C = R / 2 + 3 ∧
  P = 3 * S + 7 ∧
  B = 4 * C - 2 ∧
  E = 3 * (R + S + C + P + B)

theorem total_insects (R S C P B E : ℕ) :
  insect_collection R S C P B E →
  R + S + C + P + B + E = 652 :=
by sorry

end NUMINAMATH_CALUDE_total_insects_l2262_226256


namespace NUMINAMATH_CALUDE_computer_additions_l2262_226229

/-- The number of additions a computer can perform in 12 hours with pauses -/
def computeAdditions (additionsPerSecond : ℕ) (totalHours : ℕ) (pauseMinutes : ℕ) : ℕ :=
  let workingMinutesPerHour := 60 - pauseMinutes
  let workingSecondsPerHour := workingMinutesPerHour * 60
  let additionsPerHour := additionsPerSecond * workingSecondsPerHour
  additionsPerHour * totalHours

/-- Theorem stating that a computer with given specifications performs 540,000,000 additions in 12 hours -/
theorem computer_additions :
  computeAdditions 15000 12 10 = 540000000 := by
  sorry

end NUMINAMATH_CALUDE_computer_additions_l2262_226229


namespace NUMINAMATH_CALUDE_points_are_concyclic_l2262_226297

-- Define the points
variable (A B C D E F G H : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- State the given conditions
axiom condition1 : angle A E H = angle F E B
axiom condition2 : angle E F B = angle C F G
axiom condition3 : angle C G F = angle D G H
axiom condition4 : angle D H G = angle A H E

-- Define concyclicity
def concyclic (A B C D : Point) : Prop := sorry

-- State the theorem
theorem points_are_concyclic : concyclic A B C D := sorry

end NUMINAMATH_CALUDE_points_are_concyclic_l2262_226297


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l2262_226223

theorem largest_angle_in_special_triangle : 
  ∀ (y : ℝ), 
    60 + 70 + y = 180 →  -- Sum of angles in a triangle
    y = 70 + 15 →        -- y is 15° more than the second smallest angle (70°)
    max 60 (max 70 y) = 85 :=  -- The largest angle is 85°
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l2262_226223


namespace NUMINAMATH_CALUDE_angle_measure_l2262_226260

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l2262_226260


namespace NUMINAMATH_CALUDE_linda_coin_ratio_l2262_226262

/-- Represents the coin types in Linda's bag -/
inductive Coin
  | Dime
  | Quarter
  | Nickel

/-- Represents Linda's initial coin counts -/
structure InitialCoins where
  dimes : Nat
  quarters : Nat
  nickels : Nat

/-- Represents the additional coins given by Linda's mother -/
structure AdditionalCoins where
  dimes : Nat
  quarters : Nat

def total_coins : Nat := 35

theorem linda_coin_ratio 
  (initial : InitialCoins)
  (additional : AdditionalCoins)
  (h_initial_dimes : initial.dimes = 2)
  (h_initial_quarters : initial.quarters = 6)
  (h_initial_nickels : initial.nickels = 5)
  (h_additional_dimes : additional.dimes = 2)
  (h_additional_quarters : additional.quarters = 10)
  (h_total_coins : total_coins = 35) :
  (total_coins - (initial.dimes + additional.dimes + initial.quarters + additional.quarters) - initial.nickels) / initial.nickels = 2 := by
  sorry


end NUMINAMATH_CALUDE_linda_coin_ratio_l2262_226262


namespace NUMINAMATH_CALUDE_smallest_difference_sides_l2262_226252

theorem smallest_difference_sides (DE EF FD : ℕ) : 
  (DE < EF ∧ EF ≤ FD) →
  (DE + EF + FD = 1801) →
  (DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF) →
  (∀ DE' EF' FD' : ℕ, 
    (DE' < EF' ∧ EF' ≤ FD') →
    (DE' + EF' + FD' = 1801) →
    (DE' + EF' > FD' ∧ EF' + FD' > DE' ∧ FD' + DE' > EF') →
    EF' - DE' ≥ EF - DE) →
  EF - DE = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_difference_sides_l2262_226252


namespace NUMINAMATH_CALUDE_sum_of_min_max_x_l2262_226276

theorem sum_of_min_max_x (x y z : ℝ) (sum_eq : x + y + z = 4) (sum_sq_eq : x^2 + y^2 + z^2 = 6) :
  ∃ (m M : ℝ), (∀ x', (∃ y' z', x' + y' + z' = 4 ∧ x'^2 + y'^2 + z'^2 = 6) → m ≤ x' ∧ x' ≤ M) ∧
  m + M = 8/3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_max_x_l2262_226276


namespace NUMINAMATH_CALUDE_subset_implies_m_eq_neg_two_l2262_226277

def set_A (m : ℝ) : Set ℝ := {3, 4, 4*m - 4}
def set_B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_eq_neg_two (m : ℝ) :
  set_B m ⊆ set_A m → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_eq_neg_two_l2262_226277


namespace NUMINAMATH_CALUDE_triangle_property_l2262_226247

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) →
  -- Given condition
  (a - 2*c) * (Real.cos B) + b * (Real.cos A) = 0 →
  -- Given value for sin A
  Real.sin A = 3 * (Real.sqrt 10) / 10 →
  -- Prove these
  Real.cos B = 1/3 ∧ b/c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l2262_226247


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2262_226286

theorem binomial_expansion_coefficient (x : ℝ) :
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
    (2*x - 3)^6 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 ∧
    a₄ = 240 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2262_226286


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l2262_226221

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  b = 4 →
  (1/2) * b * a * Real.sin C = 6 * Real.sqrt 3 →
  Real.sqrt 3 * a * Real.cos C - c * Real.sin A = 0 →
  C = π / 3 ∧ c = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l2262_226221


namespace NUMINAMATH_CALUDE_solution_range_l2262_226289

theorem solution_range (x m : ℝ) : 
  (x + m) / 3 - (2 * x - 1) / 2 = m ∧ x ≤ 0 → m ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2262_226289


namespace NUMINAMATH_CALUDE_monogram_count_is_66_l2262_226214

/-- The number of letters available for the first two initials -/
def n : ℕ := 12

/-- The number of initials to choose (first and middle) -/
def k : ℕ := 2

/-- The number of ways to choose k distinct letters from n letters in alphabetical order -/
def monogram_count : ℕ := Nat.choose n k

/-- Theorem stating that the number of possible monograms is 66 -/
theorem monogram_count_is_66 : monogram_count = 66 := by
  sorry

end NUMINAMATH_CALUDE_monogram_count_is_66_l2262_226214


namespace NUMINAMATH_CALUDE_new_person_weight_l2262_226246

/-- Proves that the weight of a new person is 85 kg given the conditions of the problem -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 65 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 85 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2262_226246


namespace NUMINAMATH_CALUDE_sqrt3_plus_minus_2_power_2023_l2262_226222

theorem sqrt3_plus_minus_2_power_2023 :
  (Real.sqrt 3 + 2) ^ 2023 * (Real.sqrt 3 - 2) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_plus_minus_2_power_2023_l2262_226222


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2262_226288

/-- The slope of the given line y = 2x -/
def slope_given : ℚ := 2

/-- The point through which the perpendicular line passes -/
def point : ℚ × ℚ := (1, 1)

/-- The equation of the line to be proved -/
def line_equation (x y : ℚ) : Prop := x + 2 * y - 3 = 0

/-- Theorem stating that the line equation represents the perpendicular line -/
theorem perpendicular_line_equation :
  (∀ x y, line_equation x y ↔ y - point.2 = (-1 / slope_given) * (x - point.1)) ∧
  line_equation point.1 point.2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2262_226288


namespace NUMINAMATH_CALUDE_page_added_thrice_l2262_226282

/-- Given a book with pages numbered from 2 to n, if one page number p
    is added three times instead of once, resulting in a total sum of 4090,
    then p = 43. -/
theorem page_added_thrice (n : ℕ) (p : ℕ) (h1 : n ≥ 2) 
    (h2 : n * (n + 1) / 2 - 1 + 2 * p = 4090) : p = 43 := by
  sorry

end NUMINAMATH_CALUDE_page_added_thrice_l2262_226282


namespace NUMINAMATH_CALUDE_probability_in_B_l2262_226202

-- Define set A
def A : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 ≤ 2}

-- Define set B
def B : Set (ℝ × ℝ) := {p ∈ A | p.2 ≤ p.1^2}

-- Define the area function
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem probability_in_B : (area B) / (area A) = 17 / 24 := by sorry

end NUMINAMATH_CALUDE_probability_in_B_l2262_226202


namespace NUMINAMATH_CALUDE_parabola_vertex_l2262_226225

/-- Given a quadratic function f(x) = -x^2 + cx + d where the solution to f(x) ≤ 0 
    is [-7,1] ∪ [3,∞), prove that the vertex of the parabola is (-3, 16) -/
theorem parabola_vertex (c d : ℝ) 
  (h : Set.Icc (-7 : ℝ) 1 ∪ Set.Ici 3 = {x : ℝ | -x^2 + c*x + d ≤ 0}) : 
  let f := fun (x : ℝ) ↦ -x^2 + c*x + d
  ∃ (v : ℝ × ℝ), v = (-3, 16) ∧ ∀ (x : ℝ), f x ≤ f v.1 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2262_226225


namespace NUMINAMATH_CALUDE_existence_of_unique_distance_point_l2262_226200

-- Define a lattice point as a pair of integers
def LatticePoint := ℤ × ℤ

-- Define a function to calculate the squared distance between two points
def squaredDistance (x y : ℝ × ℝ) : ℝ :=
  (x.1 - y.1)^2 + (x.2 - y.2)^2

theorem existence_of_unique_distance_point :
  ∃ (P : ℝ × ℝ), 
    (∃ (a b : ℝ), P = (a, b) ∧ Irrational a ∧ Irrational b) ∧
    (∀ (L₁ L₂ : LatticePoint), 
      L₁ ≠ L₂ → squaredDistance (P.1, P.2) (↑L₁.1, ↑L₁.2) ≠ 
                 squaredDistance (P.1, P.2) (↑L₂.1, ↑L₂.2)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_unique_distance_point_l2262_226200


namespace NUMINAMATH_CALUDE_min_c_plus_d_is_15_l2262_226255

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

theorem min_c_plus_d_is_15 :
  ∀ (A B C D : Digit),
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A.val + B.val : ℕ) ≠ 0 →
    (C.val + D.val : ℕ) ≠ 0 →
    (A.val + B.val : ℕ) < (C.val + D.val) →
    (C.val + D.val) % (A.val + B.val) = 0 →
    ∀ (E F G H : Digit),
      E ≠ F → E ≠ G → E ≠ H → F ≠ G → F ≠ H → G ≠ H →
      (E.val + F.val : ℕ) ≠ 0 →
      (G.val + H.val : ℕ) ≠ 0 →
      (E.val + F.val : ℕ) < (G.val + H.val) →
      (G.val + H.val) % (E.val + F.val) = 0 →
      (A.val + B.val : ℕ) / (C.val + D.val : ℕ) ≤ (E.val + F.val : ℕ) / (G.val + H.val : ℕ) →
      (C.val + D.val : ℕ) ≤ 15 :=
by sorry

#check min_c_plus_d_is_15

end NUMINAMATH_CALUDE_min_c_plus_d_is_15_l2262_226255


namespace NUMINAMATH_CALUDE_jacques_initial_gumballs_l2262_226230

theorem jacques_initial_gumballs : ℕ :=
  let joanna_initial : ℕ := 40
  let purchase_multiplier : ℕ := 4
  let final_each : ℕ := 250
  let jacques_initial : ℕ := 60

  have h1 : joanna_initial + jacques_initial + purchase_multiplier * (joanna_initial + jacques_initial) = 2 * final_each :=
    by sorry

  jacques_initial

end NUMINAMATH_CALUDE_jacques_initial_gumballs_l2262_226230


namespace NUMINAMATH_CALUDE_work_completion_time_l2262_226231

/-- 
Given that:
- A does 20% less work than B per unit time
- A completes the work in 7.5 hours
Prove that B completes the same work in 6 hours
-/
theorem work_completion_time (work_rate_A work_rate_B : ℝ) 
  (h1 : work_rate_A = 0.8 * work_rate_B) 
  (h2 : work_rate_A * 7.5 = work_rate_B * 6) : 
  work_rate_B * 6 = work_rate_A * 7.5 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l2262_226231


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_sides_l2262_226219

-- Define the quadrilateral
structure InscribedQuadrilateral where
  radius : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  perpendicular : Bool

-- Define the theorem
theorem inscribed_quadrilateral_sides
  (q : InscribedQuadrilateral)
  (h1 : q.radius = 10)
  (h2 : q.diagonal1 = 12)
  (h3 : q.diagonal2 = 10 * Real.sqrt 3)
  (h4 : q.perpendicular = true) :
  ∃ (s1 s2 s3 s4 : ℝ),
    (s1 = 4 * Real.sqrt 15 + 2 * Real.sqrt 5 ∧
     s2 = 4 * Real.sqrt 15 - 2 * Real.sqrt 5 ∧
     s3 = 4 * Real.sqrt 5 + 2 * Real.sqrt 15 ∧
     s4 = 4 * Real.sqrt 5 - 2 * Real.sqrt 15) ∨
    (s1 = 4 * Real.sqrt 15 + 2 * Real.sqrt 5 ∧
     s2 = 4 * Real.sqrt 15 - 2 * Real.sqrt 5 ∧
     s3 = 4 * Real.sqrt 5 - 2 * Real.sqrt 15 ∧
     s4 = 4 * Real.sqrt 5 + 2 * Real.sqrt 15) :=
by sorry


end NUMINAMATH_CALUDE_inscribed_quadrilateral_sides_l2262_226219


namespace NUMINAMATH_CALUDE_quadrilateral_with_equal_sine_sums_l2262_226240

/-- A convex quadrilateral with angles α, β, γ, δ -/
structure ConvexQuadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 360
  all_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < δ

/-- Definition of a parallelogram -/
def is_parallelogram (q : ConvexQuadrilateral) : Prop :=
  q.α = q.γ ∧ q.β = q.δ

/-- Definition of a trapezoid -/
def is_trapezoid (q : ConvexQuadrilateral) : Prop :=
  q.α + q.β = 180 ∨ q.β + q.γ = 180

theorem quadrilateral_with_equal_sine_sums (q : ConvexQuadrilateral)
  (h : Real.sin q.α + Real.sin q.γ = Real.sin q.β + Real.sin q.δ) :
  is_parallelogram q ∨ is_trapezoid q :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_with_equal_sine_sums_l2262_226240


namespace NUMINAMATH_CALUDE_cube_difference_not_divisible_l2262_226293

theorem cube_difference_not_divisible (a b : ℤ) 
  (ha : Odd a) (hb : Odd b) (hab : a ≠ b) : 
  ¬ (2 * (a - b) ∣ (a^3 - b^3)) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_not_divisible_l2262_226293


namespace NUMINAMATH_CALUDE_point_on_y_axis_l2262_226295

/-- A point P on the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point P with coordinates (a^2 - 1, a + 1) -/
def P (a : ℝ) : Point := ⟨a^2 - 1, a + 1⟩

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (p : Point) : Prop := p.x = 0

/-- Theorem: If P(a^2 - 1, a + 1) is on the y-axis, then its coordinates are (0, 2) or (0, 0) -/
theorem point_on_y_axis (a : ℝ) : 
  on_y_axis (P a) → (P a = ⟨0, 2⟩ ∨ P a = ⟨0, 0⟩) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l2262_226295


namespace NUMINAMATH_CALUDE_average_speed_comparison_l2262_226298

theorem average_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  3 / (1/u + 1/v + 1/w) ≤ (u + v + w) / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_comparison_l2262_226298


namespace NUMINAMATH_CALUDE_jerry_initial_money_l2262_226258

-- Define Jerry's financial situation
def jerry_spent : ℕ := 6
def jerry_left : ℕ := 12

-- Theorem to prove
theorem jerry_initial_money : 
  jerry_spent + jerry_left = 18 := by sorry

end NUMINAMATH_CALUDE_jerry_initial_money_l2262_226258


namespace NUMINAMATH_CALUDE_movie_profit_l2262_226285

def movie_production (main_actor_fee supporting_actor_fee extra_fee : ℕ)
                     (main_actor_food supporting_actor_food crew_food : ℕ)
                     (post_production_cost revenue : ℕ) : Prop :=
  let main_actors := 2
  let supporting_actors := 3
  let extras := 1
  let total_people := 50
  let actor_fees := main_actors * main_actor_fee + 
                    supporting_actors * supporting_actor_fee + 
                    extras * extra_fee
  let food_cost := main_actors * main_actor_food + 
                   (supporting_actors + extras) * supporting_actor_food + 
                   (total_people - main_actors - supporting_actors - extras) * crew_food
  let equipment_rental := 2 * (actor_fees + food_cost)
  let total_cost := actor_fees + food_cost + equipment_rental + post_production_cost
  let profit := revenue - total_cost
  profit = 4584

theorem movie_profit :
  movie_production 500 100 50 10 5 3 850 10000 :=
by sorry

end NUMINAMATH_CALUDE_movie_profit_l2262_226285


namespace NUMINAMATH_CALUDE_fourth_term_is_one_l2262_226270

-- Define the geometric progression
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the specific sequence
def our_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = (3 : ℝ) ^ (1/3) ∧ 
  a 2 = (3 : ℝ) ^ (1/4) ∧ 
  a 3 = (3 : ℝ) ^ (1/12)

-- State the theorem
theorem fourth_term_is_one 
  (a : ℕ → ℝ) 
  (h1 : geometric_progression a) 
  (h2 : our_sequence a) : 
  a 4 = 1 := by
  sorry


end NUMINAMATH_CALUDE_fourth_term_is_one_l2262_226270


namespace NUMINAMATH_CALUDE_casper_initial_candies_l2262_226209

def candy_problem (initial : ℕ) : Prop :=
  let day1_after_eating : ℚ := (3/4) * initial
  let day1_remaining : ℚ := day1_after_eating - 3
  let day2_after_eating : ℚ := (1/2) * day1_remaining
  let day2_remaining : ℚ := day2_after_eating - 2
  day2_remaining = 10

theorem casper_initial_candies :
  candy_problem 36 := by sorry

end NUMINAMATH_CALUDE_casper_initial_candies_l2262_226209


namespace NUMINAMATH_CALUDE_divisibility_by_11_l2262_226267

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def seven_digit_number (m : ℕ) : ℕ :=
  856 * 10000 + m * 1000 + 248

theorem divisibility_by_11 (m : ℕ) : 
  is_divisible_by_11 (seven_digit_number m) ↔ m = 4 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_11_l2262_226267


namespace NUMINAMATH_CALUDE_unique_score_is_correct_l2262_226217

/-- Represents the score on the Mini-AHSME exam -/
structure MiniAHSMEScore where
  total : ℕ
  correct : ℕ
  wrong : ℕ
  h_total : total = 20 + 3 * correct - wrong
  h_questions : correct + wrong ≤ 20

/-- The unique score that satisfies all conditions of the problem -/
def unique_score : MiniAHSMEScore := ⟨53, 11, 0, by simp, by simp⟩

theorem unique_score_is_correct :
  ∀ s : MiniAHSMEScore,
    s.total > 50 →
    (∀ t : MiniAHSMEScore, t.total > 50 ∧ t.total < s.total → 
      ∃ u : MiniAHSMEScore, u.total = t.total ∧ u.correct ≠ t.correct) →
    s = unique_score := by sorry

end NUMINAMATH_CALUDE_unique_score_is_correct_l2262_226217


namespace NUMINAMATH_CALUDE_count_solutions_l2262_226261

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem count_solutions : 
  (Finset.filter (fun n => n + S n + S (S n) = 2023) (Finset.range 2024)).card = 4 := by sorry

end NUMINAMATH_CALUDE_count_solutions_l2262_226261


namespace NUMINAMATH_CALUDE_points_collinear_and_m_values_l2262_226291

noncomputable section

-- Define the points and vectors
def O : ℝ × ℝ := (0, 0)
def A (x : ℝ) : ℝ × ℝ := (1, Real.cos x)
def B (x : ℝ) : ℝ × ℝ := (1 + Real.sin x, Real.cos x)
def OA (x : ℝ) : ℝ × ℝ := A x
def OB (x : ℝ) : ℝ × ℝ := B x
def OC (x : ℝ) : ℝ × ℝ := (1/3 : ℝ) • (OA x) + (2/3 : ℝ) • (OB x)

-- Define the function f
def f (x m : ℝ) : ℝ :=
  (OA x).1 * (OC x).1 + (OA x).2 * (OC x).2 +
  (2*m + 1/3) * Real.sqrt ((B x).1 - (A x).1)^2 + ((B x).2 - (A x).2)^2 +
  m^2

-- Theorem statement
theorem points_collinear_and_m_values (x : ℝ) (h : x ∈ Set.Icc 0 (Real.pi / 2)) :
  (∃ t : ℝ, OC x = t • OA x + (1 - t) • OB x) ∧
  (∃ m : ℝ, (∀ y ∈ Set.Icc 0 (Real.pi / 2), f y m ≥ 5) ∧ f x m = 5 ∧ (m = -3 ∨ m = Real.sqrt 3)) :=
sorry

end NUMINAMATH_CALUDE_points_collinear_and_m_values_l2262_226291


namespace NUMINAMATH_CALUDE_expression_evaluation_l2262_226269

theorem expression_evaluation :
  let a : ℝ := 1
  let b : ℝ := -2
  a * (a - 2*b) + (a + b)^2 - (a + b)*(a - b) = 9 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2262_226269


namespace NUMINAMATH_CALUDE_equation_sum_squares_l2262_226244

theorem equation_sum_squares (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_sum_squares_l2262_226244


namespace NUMINAMATH_CALUDE_characterize_satisfying_functions_l2262_226235

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

/-- The theorem stating the characterization of functions satisfying the equation -/
theorem characterize_satisfying_functions :
  ∀ f : ℤ → ℤ, SatisfiesEquation f →
    (∀ n : ℤ, f n = 0) ∨ (∃ K : ℤ, ∀ n : ℤ, f n = 2 * n + K) := by
  sorry

end NUMINAMATH_CALUDE_characterize_satisfying_functions_l2262_226235


namespace NUMINAMATH_CALUDE_chess_tournament_green_teams_l2262_226271

theorem chess_tournament_green_teams (red_players green_players total_players total_teams red_red_teams : ℕ)
  (h1 : red_players = 64)
  (h2 : green_players = 68)
  (h3 : total_players = red_players + green_players)
  (h4 : total_teams = 66)
  (h5 : total_players = 2 * total_teams)
  (h6 : red_red_teams = 20) :
  ∃ green_green_teams : ℕ, green_green_teams = 22 ∧ 
  green_green_teams = total_teams - red_red_teams - (red_players - 2 * red_red_teams) := by
  sorry

#check chess_tournament_green_teams

end NUMINAMATH_CALUDE_chess_tournament_green_teams_l2262_226271


namespace NUMINAMATH_CALUDE_runner_speed_ratio_l2262_226268

/-- The runner's problem -/
theorem runner_speed_ratio (total_distance v₁ v₂ : ℝ) : 
  total_distance > 0 ∧
  v₁ > 0 ∧
  v₂ > 0 ∧
  total_distance / 2 / v₁ + 11 = total_distance / 2 / v₂ ∧
  total_distance / 2 / v₂ = 22 →
  v₁ / v₂ = 2 := by
  sorry

#check runner_speed_ratio

end NUMINAMATH_CALUDE_runner_speed_ratio_l2262_226268


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2262_226204

theorem cylinder_surface_area (h : ℝ) (d : ℝ) (cylinder_height : h = 2) (sphere_diameter : d = 2 * Real.sqrt 6) :
  let r := Real.sqrt (((d / 2) ^ 2 - (h / 2) ^ 2))
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = (10 + 4 * Real.sqrt 5) * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2262_226204


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2262_226299

theorem polynomial_simplification (s : ℝ) :
  (2 * s^2 + 5 * s - 3) - (s^2 + 9 * s - 4) = s^2 - 4 * s + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2262_226299


namespace NUMINAMATH_CALUDE_cube_sum_divided_by_quadratic_difference_l2262_226249

theorem cube_sum_divided_by_quadratic_difference (a c : ℝ) (h1 : a = 6) (h2 : c = 3) :
  (a^3 + c^3) / (a^2 - a*c + c^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divided_by_quadratic_difference_l2262_226249
