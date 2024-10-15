import Mathlib

namespace NUMINAMATH_CALUDE_total_sightings_is_280_l1267_126727

/-- Represents the data for a single month in the national park -/
structure MonthData where
  families : ℕ
  sightings : ℕ

/-- Calculates the total number of animal sightings over six months -/
def totalSightings (jan feb mar apr may jun : MonthData) : ℕ :=
  jan.sightings + feb.sightings + mar.sightings + apr.sightings + may.sightings + jun.sightings

/-- Theorem stating that the total number of animal sightings is 280 -/
theorem total_sightings_is_280 
  (jan : MonthData)
  (feb : MonthData)
  (mar : MonthData)
  (apr : MonthData)
  (may : MonthData)
  (jun : MonthData)
  (h1 : jan.families = 100 ∧ jan.sightings = 26)
  (h2 : feb.families = 150 ∧ feb.sightings = 78)
  (h3 : mar.families = 120 ∧ mar.sightings = 39)
  (h4 : apr.families = 204 ∧ apr.sightings = 55)
  (h5 : may.families = 204 ∧ may.sightings = 41)
  (h6 : jun.families = 265 ∧ jun.sightings = 41) :
  totalSightings jan feb mar apr may jun = 280 := by
  sorry

#check total_sightings_is_280

end NUMINAMATH_CALUDE_total_sightings_is_280_l1267_126727


namespace NUMINAMATH_CALUDE_prob_reach_edge_in_six_hops_l1267_126756

/-- Represents the 4x4 grid --/
inductive Grid
| Center : Grid
| Edge : Grid

/-- Represents the possible directions of movement --/
inductive Direction
| Up | Down | Left | Right

/-- Defines the movement rules on the grid --/
def move (g : Grid) (d : Direction) : Grid :=
  match g with
  | Grid.Center => Grid.Edge  -- Simplified for this problem
  | Grid.Edge => Grid.Edge

/-- Calculates the probability of reaching an edge square within n hops --/
def prob_reach_edge (n : ℕ) : ℚ :=
  sorry  -- Proof to be implemented

/-- Main theorem: The probability of reaching an edge square within 6 hops is 211/256 --/
theorem prob_reach_edge_in_six_hops :
  prob_reach_edge 6 = 211 / 256 :=
sorry

end NUMINAMATH_CALUDE_prob_reach_edge_in_six_hops_l1267_126756


namespace NUMINAMATH_CALUDE_curve_crosses_at_point_one_eight_l1267_126785

-- Define the curve
def x (t : ℝ) : ℝ := 2 * t^2 + 1
def y (t : ℝ) : ℝ := 2 * t^3 - 6 * t^2 + 8

-- Theorem statement
theorem curve_crosses_at_point_one_eight :
  ∃ (a b : ℝ), a ≠ b ∧ x a = x b ∧ y a = y b ∧ x a = 1 ∧ y a = 8 := by
  sorry

end NUMINAMATH_CALUDE_curve_crosses_at_point_one_eight_l1267_126785


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1267_126737

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31/13 - 1/13 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1267_126737


namespace NUMINAMATH_CALUDE_multiple_value_l1267_126740

theorem multiple_value (a b m : ℤ) : 
  a * b = m * (a + b) + 1 → 
  b = 7 → 
  b - a = 4 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_multiple_value_l1267_126740


namespace NUMINAMATH_CALUDE_complex_power_2018_l1267_126762

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_2018 : ((1 + i) / (1 - i)) ^ 2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_2018_l1267_126762


namespace NUMINAMATH_CALUDE_condition_sufficiency_not_necessity_l1267_126705

theorem condition_sufficiency_not_necessity :
  (∀ x : ℝ, x^2 - 4*x < 0 → 0 < x ∧ x < 5) ∧
  (∃ x : ℝ, 0 < x ∧ x < 5 ∧ x^2 - 4*x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficiency_not_necessity_l1267_126705


namespace NUMINAMATH_CALUDE_distinct_cube_paintings_eq_30_l1267_126771

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def num_rotations : ℕ := 24

/-- The number of distinct ways to paint a cube -/
def distinct_cube_paintings : ℕ := (num_colors.factorial) / num_rotations

theorem distinct_cube_paintings_eq_30 : distinct_cube_paintings = 30 := by
  sorry

end NUMINAMATH_CALUDE_distinct_cube_paintings_eq_30_l1267_126771


namespace NUMINAMATH_CALUDE_unique_a_sqrt_2_l1267_126732

-- Define the set of options
def options : Set ℝ := {Real.sqrt (2/3), Real.sqrt 3, Real.sqrt 8, Real.sqrt 12}

-- Define the property of being expressible as a * √2
def is_a_sqrt_2 (x : ℝ) : Prop := ∃ (a : ℚ), x = a * Real.sqrt 2

-- Theorem statement
theorem unique_a_sqrt_2 : ∃! (x : ℝ), x ∈ options ∧ is_a_sqrt_2 x :=
sorry

end NUMINAMATH_CALUDE_unique_a_sqrt_2_l1267_126732


namespace NUMINAMATH_CALUDE_equal_squares_in_5x8_grid_l1267_126726

/-- A rectangular grid with alternating light and dark squares -/
structure AlternatingGrid where
  rows : ℕ
  cols : ℕ

/-- Count of dark squares in an AlternatingGrid -/
def dark_squares (grid : AlternatingGrid) : ℕ :=
  sorry

/-- Count of light squares in an AlternatingGrid -/
def light_squares (grid : AlternatingGrid) : ℕ :=
  sorry

/-- Theorem: In a 5 × 8 grid with alternating squares, the number of dark squares equals the number of light squares -/
theorem equal_squares_in_5x8_grid :
  let grid : AlternatingGrid := ⟨5, 8⟩
  dark_squares grid = light_squares grid :=
by sorry

end NUMINAMATH_CALUDE_equal_squares_in_5x8_grid_l1267_126726


namespace NUMINAMATH_CALUDE_max_value_x4y2z_l1267_126796

theorem max_value_x4y2z (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x^2 + y^2 + z^2 = 1) :
  x^4 * y^2 * z ≤ 32 / (16807 * Real.sqrt 7) ∧ 
  ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = 1 ∧ x^4 * y^2 * z = 32 / (16807 * Real.sqrt 7) := by
  sorry

#check max_value_x4y2z

end NUMINAMATH_CALUDE_max_value_x4y2z_l1267_126796


namespace NUMINAMATH_CALUDE_total_dress_designs_l1267_126765

/-- The number of fabric colors available -/
def num_colors : ℕ := 5

/-- The number of patterns available -/
def num_patterns : ℕ := 4

/-- The number of sleeve length options -/
def num_sleeve_lengths : ℕ := 2

/-- Each dress design requires exactly one color, one pattern, and one sleeve length -/
axiom dress_design_requirement : True

/-- The total number of different dress designs possible -/
def total_designs : ℕ := num_colors * num_patterns * num_sleeve_lengths

/-- Theorem stating that the total number of different dress designs is 40 -/
theorem total_dress_designs : total_designs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l1267_126765


namespace NUMINAMATH_CALUDE_distribute_four_books_to_three_people_l1267_126733

/-- Represents the number of ways to distribute books to people. -/
def distribute_books (num_books : ℕ) (num_people : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 4 different books to 3 people,
    with each person getting at least one book, can be done in 36 ways. -/
theorem distribute_four_books_to_three_people :
  distribute_books 4 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_distribute_four_books_to_three_people_l1267_126733


namespace NUMINAMATH_CALUDE_max_value_expression_l1267_126758

theorem max_value_expression (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  (∀ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) ≤ 3/2 * (c^2 + d^2)) ∧
  (∃ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) = 3/2 * (c^2 + d^2)) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l1267_126758


namespace NUMINAMATH_CALUDE_linear_function_kb_positive_l1267_126755

/-- A linear function passing through the second, third, and fourth quadrants -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = k * x + b
  third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = k * x + b
  fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = k * x + b

/-- Theorem: For a linear function passing through the second, third, and fourth quadrants, kb > 0 -/
theorem linear_function_kb_positive (f : LinearFunction) : f.k * f.b > 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_kb_positive_l1267_126755


namespace NUMINAMATH_CALUDE_smaller_part_is_eleven_l1267_126793

theorem smaller_part_is_eleven (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : 
  min x y = 11 := by
  sorry

end NUMINAMATH_CALUDE_smaller_part_is_eleven_l1267_126793


namespace NUMINAMATH_CALUDE_sum_of_digits_inequality_l1267_126786

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_inequality (N : ℕ) :
  sum_of_digits N ≤ 5 * sum_of_digits (5^5 * N) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_inequality_l1267_126786


namespace NUMINAMATH_CALUDE_custom_op_example_l1267_126774

-- Define the custom operation
def custom_op (m n p q : ℚ) : ℚ := m * p * ((q + n) / n)

-- State the theorem
theorem custom_op_example :
  custom_op 5 9 7 4 = 455 / 9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l1267_126774


namespace NUMINAMATH_CALUDE_emma_bank_balance_emma_final_balance_l1267_126751

theorem emma_bank_balance (initial_balance : ℝ) (shoe_percentage : ℝ) 
  (tuesday_deposit_percentage : ℝ) (wednesday_deposit_percentage : ℝ) 
  (final_withdrawal_percentage : ℝ) : ℝ :=
  let shoe_cost := initial_balance * shoe_percentage
  let monday_balance := initial_balance - shoe_cost
  let tuesday_deposit := shoe_cost * tuesday_deposit_percentage
  let tuesday_balance := monday_balance + tuesday_deposit
  let wednesday_deposit := shoe_cost * wednesday_deposit_percentage
  let wednesday_balance := tuesday_balance + wednesday_deposit
  let final_withdrawal := wednesday_balance * final_withdrawal_percentage
  let final_balance := wednesday_balance - final_withdrawal
  final_balance
  
theorem emma_final_balance : 
  emma_bank_balance 1200 0.08 0.25 1.5 0.05 = 1208.40 := by
  sorry

end NUMINAMATH_CALUDE_emma_bank_balance_emma_final_balance_l1267_126751


namespace NUMINAMATH_CALUDE_better_hay_cost_is_18_l1267_126724

/-- The cost of better quality hay per bale -/
def better_hay_cost (initial_bales : ℕ) (price_increase : ℕ) (previous_cost : ℕ) : ℕ :=
  (initial_bales * previous_cost + price_increase) / (2 * initial_bales)

/-- Proof that the cost of better quality hay is $18 per bale -/
theorem better_hay_cost_is_18 :
  better_hay_cost 10 210 15 = 18 := by
  sorry

end NUMINAMATH_CALUDE_better_hay_cost_is_18_l1267_126724


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1267_126734

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4 + Real.sqrt 4 / Real.sqrt 5) * (Real.sqrt 5 / Real.sqrt 6) =
  (Real.sqrt 10 + 2 * Real.sqrt 2) / 4 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1267_126734


namespace NUMINAMATH_CALUDE_percentage_increase_l1267_126743

theorem percentage_increase (initial final : ℝ) (h : initial > 0) :
  let increase := (final - initial) / initial * 100
  initial = 150 ∧ final = 210 → increase = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1267_126743


namespace NUMINAMATH_CALUDE_probability_of_two_boys_l1267_126721

theorem probability_of_two_boys (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total = 15)
  (h2 : boys = 9)
  (h3 : girls = 6)
  (h4 : total = boys + girls) :
  (Nat.choose boys 2 : ℚ) / (Nat.choose total 2 : ℚ) = 12 / 35 := by
sorry

end NUMINAMATH_CALUDE_probability_of_two_boys_l1267_126721


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1267_126736

-- Define the propositions p and q
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q : 
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1267_126736


namespace NUMINAMATH_CALUDE_alloy_mixing_solution_exists_l1267_126787

/-- Represents an alloy of copper and tin -/
structure Alloy where
  mass : ℝ
  copper_percentage : ℝ

/-- Proves that a solution exists for the alloy mixing problem if and only if p is within the specified range -/
theorem alloy_mixing_solution_exists (alloy1 alloy2 : Alloy) (target_mass : ℝ) (p : ℝ) :
  alloy1.mass = 3 ∧ 
  alloy1.copper_percentage = 40 ∧
  alloy2.mass = 7 ∧
  alloy2.copper_percentage = 30 ∧
  target_mass = 8 →
  (∃ x : ℝ, 
    0 ≤ x ∧ 
    x ≤ alloy1.mass ∧ 
    0 ≤ target_mass - x ∧ 
    target_mass - x ≤ alloy2.mass ∧
    (alloy1.copper_percentage / 100 * x + alloy2.copper_percentage / 100 * (target_mass - x)) / target_mass = p / 100) ↔
  31.25 ≤ p ∧ p ≤ 33.75 := by
  sorry

#check alloy_mixing_solution_exists

end NUMINAMATH_CALUDE_alloy_mixing_solution_exists_l1267_126787


namespace NUMINAMATH_CALUDE_planted_area_fraction_l1267_126746

theorem planted_area_fraction (a b c x : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c^2 = a^2 + b^2) 
  (h4 : x^2 - 7*x + 9 = 0) (h5 : x > 0) (h6 : x < a) (h7 : x < b) :
  (a*b/2 - x^2) / (a*b/2) = 30/30 - ((7 - Real.sqrt 13)/2)^2 / 30 := by
  sorry

end NUMINAMATH_CALUDE_planted_area_fraction_l1267_126746


namespace NUMINAMATH_CALUDE_probability_at_least_three_hits_l1267_126798

def probability_hit_single_shot : ℝ := 0.8
def number_of_shots : ℕ := 4
def minimum_hits : ℕ := 3

theorem probability_at_least_three_hits :
  let p := probability_hit_single_shot
  let n := number_of_shots
  let k := minimum_hits
  (Finset.sum (Finset.range (n - k + 1))
    (λ i => (n.choose (k + i)) * p^(k + i) * (1 - p)^(n - k - i))) = 0.8192 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_three_hits_l1267_126798


namespace NUMINAMATH_CALUDE_expression_value_l1267_126720

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  x^2 + y^2 - z^2 + 3*x*y = -10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1267_126720


namespace NUMINAMATH_CALUDE_groom_age_l1267_126728

theorem groom_age (bride_age groom_age : ℕ) : 
  bride_age = groom_age + 19 →
  bride_age + groom_age = 185 →
  groom_age = 83 := by
sorry

end NUMINAMATH_CALUDE_groom_age_l1267_126728


namespace NUMINAMATH_CALUDE_unique_number_with_equal_sums_l1267_126731

def ends_with_9876 (n : ℕ) : Prop :=
  n % 10000 = 9876

def masha_sum (n : ℕ) : ℕ :=
  (n / 1000) * 10 + n % 1000

def misha_sum (n : ℕ) : ℕ :=
  (n / 10000) + n % 10000

theorem unique_number_with_equal_sums :
  ∃! n : ℕ, n > 9999 ∧ ends_with_9876 n ∧ masha_sum n = misha_sum n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_equal_sums_l1267_126731


namespace NUMINAMATH_CALUDE_call_center_efficiency_l1267_126706

-- Define the number of agents in each team
variable (A B : ℕ)

-- Define the fraction of calls processed by each team
variable (calls_A calls_B : ℚ)

-- Define the theorem
theorem call_center_efficiency
  (h1 : A = (5 : ℚ) / 8 * B)  -- Team A has 5/8 as many agents as team B
  (h2 : calls_B = 8 / 11)     -- Team B processed 8/11 of the total calls
  (h3 : calls_A + calls_B = 1) -- Total calls processed by both teams is 1
  : (calls_A / A) / (calls_B / B) = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_call_center_efficiency_l1267_126706


namespace NUMINAMATH_CALUDE_common_chord_length_l1267_126791

/-- Curve C1 defined by (x-1)^2 + y^2 = 4 -/
def C1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

/-- Curve C2 defined by x^2 + y^2 = 4y -/
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 4*y

/-- The length of the common chord between C1 and C2 is √11 -/
theorem common_chord_length :
  ∃ (a b c d : ℝ), C1 a b ∧ C1 c d ∧ C2 a b ∧ C2 c d ∧
  ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l1267_126791


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1267_126784

-- Define what a quadratic equation is
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function we're checking
def f (x : ℝ) : ℝ := x^2 - 2

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1267_126784


namespace NUMINAMATH_CALUDE_flowers_in_basket_is_four_l1267_126716

/-- The number of flowers in each basket after planting, growth, and distribution -/
def flowers_per_basket (daughters : ℕ) (flowers_per_daughter : ℕ) (new_flowers : ℕ) (dead_flowers : ℕ) (num_baskets : ℕ) : ℕ :=
  let initial_flowers := daughters * flowers_per_daughter
  let total_flowers := initial_flowers + new_flowers
  let remaining_flowers := total_flowers - dead_flowers
  remaining_flowers / num_baskets

/-- Theorem stating that under the given conditions, each basket will contain 4 flowers -/
theorem flowers_in_basket_is_four :
  flowers_per_basket 2 5 20 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_flowers_in_basket_is_four_l1267_126716


namespace NUMINAMATH_CALUDE_polynomial_equality_l1267_126753

theorem polynomial_equality (x : ℝ) : let p : ℝ → ℝ := λ x => -7*x^4 - 5*x^3 - 8*x^2 + 8*x - 9
  4*x^4 + 7*x^3 - 2*x + 5 + p x = -3*x^4 + 2*x^3 - 8*x^2 + 6*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1267_126753


namespace NUMINAMATH_CALUDE_total_apples_to_pack_l1267_126797

/-- The number of apples in one dozen -/
def apples_per_dozen : ℕ := 12

/-- The number of boxes needed -/
def boxes_needed : ℕ := 90

/-- Theorem stating the total number of apples to be packed -/
theorem total_apples_to_pack : apples_per_dozen * boxes_needed = 1080 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_to_pack_l1267_126797


namespace NUMINAMATH_CALUDE_exists_specific_number_l1267_126745

theorem exists_specific_number : ∃ y : ℕ+, 
  (y.val % 4 = 0) ∧ 
  (y.val % 5 = 0) ∧ 
  (y.val % 7 = 0) ∧ 
  (y.val % 13 = 0) ∧ 
  (y.val % 8 ≠ 0) ∧ 
  (y.val % 15 ≠ 0) ∧ 
  (y.val % 50 ≠ 0) ∧ 
  (y.val % 10 = 0) ∧ 
  (y.val = 1820) :=
by sorry

end NUMINAMATH_CALUDE_exists_specific_number_l1267_126745


namespace NUMINAMATH_CALUDE_net_profit_calculation_l1267_126719

/-- Given the purchase price, overhead percentage, and markup, calculate the net profit --/
def calculate_net_profit (purchase_price overhead_percentage markup : ℝ) : ℝ :=
  let overhead := purchase_price * overhead_percentage
  markup - overhead

/-- Theorem stating that given the specified conditions, the net profit is $27.60 --/
theorem net_profit_calculation :
  let purchase_price : ℝ := 48
  let overhead_percentage : ℝ := 0.05
  let markup : ℝ := 30
  calculate_net_profit purchase_price overhead_percentage markup = 27.60 := by
  sorry

#eval calculate_net_profit 48 0.05 30

end NUMINAMATH_CALUDE_net_profit_calculation_l1267_126719


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1267_126789

/-- Given a > 0 and a ≠ 1, prove that (2, 3) is the fixed point of f(x) = a^(x-2) + 2 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 2
  f 2 = 3 ∧ ∀ x : ℝ, f x = x → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1267_126789


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_two_l1267_126708

theorem factorial_ratio_equals_two : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_two_l1267_126708


namespace NUMINAMATH_CALUDE_root_product_plus_one_l1267_126767

theorem root_product_plus_one (r s t : ℝ) : 
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  s^3 - 15*s^2 + 25*s - 10 = 0 →
  t^3 - 15*t^2 + 25*t - 10 = 0 →
  (1+r)*(1+s)*(1+t) = 51 := by
sorry

end NUMINAMATH_CALUDE_root_product_plus_one_l1267_126767


namespace NUMINAMATH_CALUDE_sqrt_24_minus_3sqrt_2_3_l1267_126722

theorem sqrt_24_minus_3sqrt_2_3 : Real.sqrt 24 - 3 * Real.sqrt (2/3) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_24_minus_3sqrt_2_3_l1267_126722


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1267_126768

theorem inequality_solution_set (x : ℝ) : 
  (((1 - 2*x) / ((x - 3) * (2*x + 1))) ≥ 0) ↔ 
  (x ∈ Set.Iio (-1/2) ∪ Set.Icc (1/2) 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1267_126768


namespace NUMINAMATH_CALUDE_nantucket_meeting_attendance_l1267_126757

theorem nantucket_meeting_attendance :
  let total_population : ℕ := 300
  let females_attending : ℕ := 50
  let males_attending : ℕ := 2 * females_attending
  let total_attending : ℕ := males_attending + females_attending
  (total_attending : ℚ) / total_population = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_nantucket_meeting_attendance_l1267_126757


namespace NUMINAMATH_CALUDE_m_value_l1267_126750

def A (m : ℝ) : Set ℝ := {-1, 3, m^2}
def B : Set ℝ := {3, 4}

theorem m_value (m : ℝ) (h : B ⊆ A m) : m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l1267_126750


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1267_126779

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x ≤ 0}

-- State the theorem
theorem quadratic_inequality_theorem 
  (a b c : ℝ) 
  (h : solution_set a b c = {x : ℝ | x ≤ -1 ∨ x ≥ 3}) :
  (a + b + c > 0) ∧ 
  (4*a - 2*b + c < 0) ∧ 
  ({x : ℝ | c*x^2 - b*x + a < 0} = {x : ℝ | -1/3 < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1267_126779


namespace NUMINAMATH_CALUDE_train_speed_equation_l1267_126715

theorem train_speed_equation (x : ℝ) (h1 : x > 80) : 
  (353 / (x - 80) - 353 / x = 5 / 3) ↔ 
  (353 / (x - 80) - 353 / x = 100 / 60) := by sorry

end NUMINAMATH_CALUDE_train_speed_equation_l1267_126715


namespace NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l1267_126723

/-- A structure representing angles in a geometric context -/
structure Angle where
  measure : ℝ

/-- A predicate to determine if two angles are corresponding -/
def are_corresponding (a b : Angle) : Prop := sorry

/-- The theorem stating that the general claim "Corresponding angles are equal" is false -/
theorem corresponding_angles_not_always_equal :
  ¬ (∀ (a b : Angle), are_corresponding a b → a = b) := by sorry

end NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l1267_126723


namespace NUMINAMATH_CALUDE_brand_z_fraction_l1267_126766

/-- Represents the state of the fuel tank -/
structure TankState where
  z : ℚ  -- Amount of brand Z gasoline
  y : ℚ  -- Amount of brand Y gasoline

/-- Fills the tank with brand Y gasoline when it's partially empty -/
def fillWithY (s : TankState) (emptyFraction : ℚ) : TankState :=
  { z := s.z, y := s.y + emptyFraction }

/-- Fills the tank with brand Z gasoline when it's partially empty -/
def fillWithZ (s : TankState) (emptyFraction : ℚ) : TankState :=
  { z := s.z + emptyFraction, y := s.y }

/-- Empties the tank by a given fraction -/
def emptyTank (s : TankState) (emptyFraction : ℚ) : TankState :=
  { z := s.z * (1 - emptyFraction), y := s.y * (1 - emptyFraction) }

/-- The final state of the tank after the described filling process -/
def finalState : TankState :=
  let s1 := { z := 1, y := 0 }
  let s2 := fillWithY (emptyTank s1 (3/4)) (3/4)
  let s3 := fillWithZ (emptyTank s2 (1/2)) (1/2)
  fillWithY (emptyTank s3 (1/2)) (1/2)

/-- The fraction of brand Z gasoline in the final state is 5/16 -/
theorem brand_z_fraction :
  finalState.z / (finalState.z + finalState.y) = 5/16 := by sorry

end NUMINAMATH_CALUDE_brand_z_fraction_l1267_126766


namespace NUMINAMATH_CALUDE_vitamin_d_scientific_notation_l1267_126714

theorem vitamin_d_scientific_notation : 0.0000046 = 4.6 * 10^(-6) := by
  sorry

end NUMINAMATH_CALUDE_vitamin_d_scientific_notation_l1267_126714


namespace NUMINAMATH_CALUDE_test_score_probability_and_expectation_l1267_126792

-- Define the scoring system
def score_correct : ℕ := 5
def score_incorrect : ℕ := 0

-- Define the total number of questions and correct answers
def total_questions : ℕ := 10
def correct_answers : ℕ := 6

-- Define the probabilities for the remaining questions
def prob_two_eliminated : ℚ := 1/2
def prob_one_eliminated : ℚ := 1/3
def prob_guessed : ℚ := 1/4

-- Define the score distribution
def score_distribution : List (ℕ × ℚ) := [
  (30, 1/8),
  (35, 17/48),
  (40, 17/48),
  (45, 7/48),
  (50, 1/48)
]

-- Theorem statement
theorem test_score_probability_and_expectation :
  (List.lookup 45 score_distribution = some (7/48)) ∧
  (List.foldl (λ acc (score, prob) => acc + score * prob) 0 score_distribution = 455/12) := by
  sorry

end NUMINAMATH_CALUDE_test_score_probability_and_expectation_l1267_126792


namespace NUMINAMATH_CALUDE_min_socks_for_ten_pairs_l1267_126777

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (black : ℕ)

/-- Calculates the minimum number of socks to draw to guarantee a certain number of pairs -/
def minSocksForPairs (drawer : SockDrawer) (numPairs : ℕ) : ℕ :=
  4 + 1 + 2 * (numPairs - 1)

/-- Theorem stating the minimum number of socks to draw for 10 pairs -/
theorem min_socks_for_ten_pairs (drawer : SockDrawer) 
  (h_red : drawer.red = 100)
  (h_green : drawer.green = 80)
  (h_blue : drawer.blue = 60)
  (h_black : drawer.black = 40) :
  minSocksForPairs drawer 10 = 23 := by
  sorry

#eval minSocksForPairs ⟨100, 80, 60, 40⟩ 10

end NUMINAMATH_CALUDE_min_socks_for_ten_pairs_l1267_126777


namespace NUMINAMATH_CALUDE_x_equals_160_l1267_126725

/-- Given a relationship between x, y, and z, prove that x equals 160 when y is 16 and z is 7. -/
theorem x_equals_160 (k : ℝ) (x y z : ℝ → ℝ) :
  (∀ t, x t = k * y t / (z t)^2) →  -- Relationship between x, y, and z
  (x 0 = 10 ∧ y 0 = 4 ∧ z 0 = 14) →  -- Initial condition
  (y 1 = 16 ∧ z 1 = 7) →  -- New condition
  x 1 = 160 := by
sorry

end NUMINAMATH_CALUDE_x_equals_160_l1267_126725


namespace NUMINAMATH_CALUDE_total_crayons_l1267_126760

/-- Theorem: The total number of crayons after adding more is the sum of the initial number and the added number. -/
theorem total_crayons (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_crayons_l1267_126760


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1267_126717

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_mono : monotonically_increasing a)
  (h_sum : a 1 + a 2 + a 3 = 21)
  (h_prod : a 1 * a 2 * a 3 = 231) :
  (a 2 = 7) ∧ (∀ n : ℕ, a n = 4 * n - 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1267_126717


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l1267_126730

theorem sphere_radius_ratio (V_L V_S r_L r_S : ℝ) : 
  V_L = 675 * Real.pi → 
  V_S = 0.2 * V_L → 
  V_L = (4/3) * Real.pi * r_L^3 → 
  V_S = (4/3) * Real.pi * r_S^3 → 
  r_S / r_L = 1 / Real.rpow 5 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l1267_126730


namespace NUMINAMATH_CALUDE_team_total_score_l1267_126718

def team_score (connor_initial : ℕ) (amy_initial : ℕ) (jason_initial : ℕ) 
  (connor_bonus : ℕ) (amy_bonus : ℕ) (jason_bonus : ℕ) (emily : ℕ) : ℕ :=
  connor_initial + connor_bonus + amy_initial + amy_bonus + jason_initial + jason_bonus + emily

theorem team_total_score :
  let connor_initial := 2
  let amy_initial := connor_initial + 4
  let jason_initial := amy_initial * 2
  let connor_bonus := 3
  let amy_bonus := 5
  let jason_bonus := 1
  let emily := 3 * (connor_initial + amy_initial + jason_initial)
  team_score connor_initial amy_initial jason_initial connor_bonus amy_bonus jason_bonus emily = 89 := by
  sorry

end NUMINAMATH_CALUDE_team_total_score_l1267_126718


namespace NUMINAMATH_CALUDE_truck_wheels_count_l1267_126764

/-- Calculates the toll for a truck given the number of axles -/
def toll (x : ℕ) : ℚ := 3.50 + 0.50 * (x - 2)

/-- Calculates the total number of wheels on a truck given the number of axles -/
def totalWheels (x : ℕ) : ℕ := 2 + 4 * (x - 1)

theorem truck_wheels_count :
  ∃ (x : ℕ), 
    x > 0 ∧
    toll x = 5 ∧
    totalWheels x = 18 :=
by sorry

end NUMINAMATH_CALUDE_truck_wheels_count_l1267_126764


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l1267_126794

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  geometric_sequence a → a 4 = 2 → a 8 = 32 → a 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l1267_126794


namespace NUMINAMATH_CALUDE_hotdogs_sold_l1267_126752

theorem hotdogs_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 99 → remaining = 97 → sold = initial - remaining → sold = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotdogs_sold_l1267_126752


namespace NUMINAMATH_CALUDE_inequality_a_l1267_126790

theorem inequality_a (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 * y^2 * z + y^4 * x^2 * z + y^4 * z^2 * x + z^4 * y^2 * x + x^4 * z^2 * y + z^4 * x^2 * y ≥
  2 * (x^3 * y^2 * z^2 + x^2 * y^3 * z^2 + x^2 * y^2 * z^3) := by
sorry


end NUMINAMATH_CALUDE_inequality_a_l1267_126790


namespace NUMINAMATH_CALUDE_original_rate_l1267_126700

/-- Given a reduction of 'a' yuan followed by a 20% reduction resulting in a final rate of 'b' yuan per minute, 
    the original rate was a + 1.25b yuan per minute. -/
theorem original_rate (a b : ℝ) : 
  (∃ x : ℝ, 0.8 * (x - a) = b) → 
  (∃ x : ℝ, x = a + 1.25 * b ∧ 0.8 * (x - a) = b) :=
by sorry

end NUMINAMATH_CALUDE_original_rate_l1267_126700


namespace NUMINAMATH_CALUDE_rice_price_fall_l1267_126738

theorem rice_price_fall (old_price new_price : ℝ) 
  (h : 40 * old_price = 50 * new_price) : 
  (old_price - new_price) / old_price = 1/5 := by
sorry

end NUMINAMATH_CALUDE_rice_price_fall_l1267_126738


namespace NUMINAMATH_CALUDE_chess_team_arrangement_l1267_126712

/-- The number of boys on the chess team -/
def num_boys : ℕ := 3

/-- The number of girls on the chess team -/
def num_girls : ℕ := 2

/-- The total number of students on the chess team -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to arrange the chess team in a row with a girl at each end and boys in the middle -/
def num_arrangements : ℕ := num_girls.factorial * num_boys.factorial

theorem chess_team_arrangement :
  num_arrangements = 12 :=
sorry

end NUMINAMATH_CALUDE_chess_team_arrangement_l1267_126712


namespace NUMINAMATH_CALUDE_P_n_roots_P_2018_roots_l1267_126783

-- Define the sequence of polynomials
def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | 1, x => x
  | (n + 2), x => x * P (n + 1) x - P n x

-- Define a function to count distinct real roots
noncomputable def count_distinct_real_roots (f : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem P_n_roots (n : ℕ) : count_distinct_real_roots (P n) = n := by
  sorry

-- Specific case for P_2018
theorem P_2018_roots : count_distinct_real_roots (P 2018) = 2018 := by
  sorry

end NUMINAMATH_CALUDE_P_n_roots_P_2018_roots_l1267_126783


namespace NUMINAMATH_CALUDE_simplify_expression_l1267_126748

theorem simplify_expression (a b : ℝ) : 120*a - 55*a + 33*b - 7*b = 65*a + 26*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1267_126748


namespace NUMINAMATH_CALUDE_gcd_459_357_l1267_126707

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1267_126707


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l1267_126770

theorem rectangular_plot_length
  (width : ℝ)
  (num_poles : ℕ)
  (pole_spacing : ℝ)
  (h1 : width = 50)
  (h2 : num_poles = 14)
  (h3 : pole_spacing = 20)
  : ∃ (length : ℝ), length = 80 ∧ 2 * (length + width) = (num_poles - 1) * pole_spacing :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l1267_126770


namespace NUMINAMATH_CALUDE_circuit_malfunction_probability_l1267_126713

/-- Represents an electronic component with a given failure rate -/
structure Component where
  failureRate : ℝ
  hFailureRate : 0 ≤ failureRate ∧ failureRate ≤ 1

/-- Represents a circuit with two components connected in series -/
structure Circuit where
  componentA : Component
  componentB : Component

/-- The probability of a circuit malfunctioning -/
def malfunctionProbability (c : Circuit) : ℝ :=
  1 - (1 - c.componentA.failureRate) * (1 - c.componentB.failureRate)

theorem circuit_malfunction_probability (c : Circuit) 
    (hA : c.componentA.failureRate = 0.2)
    (hB : c.componentB.failureRate = 0.5) :
    malfunctionProbability c = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_circuit_malfunction_probability_l1267_126713


namespace NUMINAMATH_CALUDE_sine_squares_sum_l1267_126772

theorem sine_squares_sum (α : Real) : 
  (Real.sin (α - π/3))^2 + (Real.sin α)^2 + (Real.sin (α + π/3))^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sine_squares_sum_l1267_126772


namespace NUMINAMATH_CALUDE_mary_bike_rental_hours_l1267_126782

/-- Calculates the number of hours a bike was rented given the total payment, fixed fee, and hourly rate. -/
def rent_hours (total_payment fixed_fee hourly_rate : ℚ) : ℚ :=
  (total_payment - fixed_fee) / hourly_rate

/-- Proves that Mary rented the bike for 9 hours given the specified conditions. -/
theorem mary_bike_rental_hours :
  let fixed_fee : ℚ := 17
  let hourly_rate : ℚ := 7
  let total_payment : ℚ := 80
  rent_hours total_payment fixed_fee hourly_rate = 9 := by
  sorry


end NUMINAMATH_CALUDE_mary_bike_rental_hours_l1267_126782


namespace NUMINAMATH_CALUDE_share_multiple_l1267_126702

theorem share_multiple (total : ℝ) (c_share : ℝ) (k : ℝ) :
  total = 427 →
  c_share = 84 →
  (∃ (a_share b_share : ℝ), 
    total = a_share + b_share + c_share ∧
    3 * a_share = 4 * b_share ∧
    3 * a_share = k * c_share) →
  k = 7 := by
sorry

end NUMINAMATH_CALUDE_share_multiple_l1267_126702


namespace NUMINAMATH_CALUDE_equation_solution_existence_l1267_126769

theorem equation_solution_existence (a : ℝ) :
  (∃ x : ℝ, 3 * 4^(x - 2) + 27 = a + a * 4^(x - 2)) ↔ 3 < a ∧ a < 27 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_existence_l1267_126769


namespace NUMINAMATH_CALUDE_smallest_non_representable_as_cube_sum_l1267_126795

theorem smallest_non_representable_as_cube_sum : ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), m < n → ∃ (x y : ℤ), m = x^3 + 3*y^3) ∧
  ¬∃ (x y : ℤ), n = x^3 + 3*y^3 ∧ 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_representable_as_cube_sum_l1267_126795


namespace NUMINAMATH_CALUDE_custom_op_value_l1267_126711

-- Define the custom operation *
def custom_op (a b : ℚ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem custom_op_value (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 12) (prod_eq : a * b = 32) : 
  custom_op a b = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_value_l1267_126711


namespace NUMINAMATH_CALUDE_initial_population_l1267_126754

theorem initial_population (P : ℝ) : 
  (P * (1 - 0.1)^2 = 8100) → P = 10000 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_l1267_126754


namespace NUMINAMATH_CALUDE_quadratic_sum_and_square_sum_l1267_126761

theorem quadratic_sum_and_square_sum (a b c d m n : ℕ+) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 1989)
  (h2 : a + b + c + d = m^2)
  (h3 : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_and_square_sum_l1267_126761


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1267_126742

theorem intersection_implies_a_value (A B : Set ℤ) (a : ℤ) : 
  A = {-1, 0, 1} →
  B = {0, a, 2} →
  A ∩ B = {-1, 0} →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1267_126742


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l1267_126749

theorem quadratic_root_ratio (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ / x₁ = -4 ∧ x₁^2 + p*x₁ - 16 = 0 ∧ x₂^2 + p*x₂ - 16 = 0) → 
  (p = 6 ∨ p = -6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l1267_126749


namespace NUMINAMATH_CALUDE_expression_approximately_equal_to_0_2436_l1267_126776

-- Define the expression
def expression : ℚ := (108 * 3 - (108 + 92)) / (92 * 7 - (45 * 3))

-- State the theorem
theorem expression_approximately_equal_to_0_2436 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.00005 ∧ |expression - 0.2436| < ε := by
  sorry

end NUMINAMATH_CALUDE_expression_approximately_equal_to_0_2436_l1267_126776


namespace NUMINAMATH_CALUDE_investment_rate_problem_l1267_126741

theorem investment_rate_problem (total_interest amount_invested_low rate_high : ℚ) 
  (h1 : total_interest = 520)
  (h2 : amount_invested_low = 2000)
  (h3 : rate_high = 5 / 100) : 
  ∃ (rate_low : ℚ), 
    amount_invested_low * rate_low + 4 * amount_invested_low * rate_high = total_interest ∧ 
    rate_low = 6 / 100 := by
sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l1267_126741


namespace NUMINAMATH_CALUDE_triangle_inequality_squared_l1267_126747

theorem triangle_inequality_squared (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 + b^2 + c^2 < 2 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_squared_l1267_126747


namespace NUMINAMATH_CALUDE_distinct_points_on_curve_l1267_126704

theorem distinct_points_on_curve (a b : ℝ) : 
  a ≠ b →
  (a^2 + Real.sqrt π^4 = 2 * (Real.sqrt π)^2 * a + 1) →
  (b^2 + Real.sqrt π^4 = 2 * (Real.sqrt π)^2 * b + 1) →
  |a - b| = 2 := by sorry

end NUMINAMATH_CALUDE_distinct_points_on_curve_l1267_126704


namespace NUMINAMATH_CALUDE_probability_ace_second_draw_l1267_126759

/-- The probability of drawing an Ace in the second draw without replacement from a deck of 52 cards, given that an Ace was drawn in the first draw. -/
theorem probability_ace_second_draw (initial_deck_size : ℕ) (initial_aces : ℕ) 
  (h1 : initial_deck_size = 52)
  (h2 : initial_aces = 4)
  (h3 : initial_aces > 0) :
  (initial_aces - 1 : ℚ) / (initial_deck_size - 1 : ℚ) = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_ace_second_draw_l1267_126759


namespace NUMINAMATH_CALUDE_max_sum_AB_l1267_126709

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem max_sum_AB :
  ∃ (A B C D : ℕ),
    is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (C + D > 1) ∧
    (∃ k : ℕ, k * (C + D) = A + B) ∧
    (∀ A' B' C' D' : ℕ,
      is_digit A' ∧ is_digit B' ∧ is_digit C' ∧ is_digit D' →
      A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ B' ≠ C' ∧ B' ≠ D' ∧ C' ≠ D' →
      (C' + D' > 1) →
      (∃ k' : ℕ, k' * (C' + D') = A' + B') →
      A' + B' ≤ A + B) →
    A + B = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_AB_l1267_126709


namespace NUMINAMATH_CALUDE_simplify_tan_cot_expression_l1267_126763

theorem simplify_tan_cot_expression :
  let tan_45 : Real := 1
  let cot_45 : Real := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_tan_cot_expression_l1267_126763


namespace NUMINAMATH_CALUDE_minimum_race_distance_l1267_126744

/-- The minimum distance a runner must travel in a race with given constraints -/
theorem minimum_race_distance (wall_length : ℝ) (distance_A : ℝ) (distance_B : ℝ) :
  wall_length = 1500 →
  distance_A = 400 →
  distance_B = 600 →
  let min_distance := Real.sqrt (wall_length ^ 2 + (distance_A + distance_B) ^ 2)
  ⌊min_distance + 0.5⌋ = 1803 := by
  sorry

end NUMINAMATH_CALUDE_minimum_race_distance_l1267_126744


namespace NUMINAMATH_CALUDE_regular_polygon_assembly_l1267_126703

theorem regular_polygon_assembly (interior_angle : ℝ) (h1 : interior_angle = 150) :
  ∃ (n : ℕ) (m : ℕ), n * interior_angle + m * 60 = 360 :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_assembly_l1267_126703


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l1267_126739

/-- If the lines x + 2y = 3 and nx + my = 4 are parallel, then m = 2n -/
theorem parallel_lines_slope (n m : ℝ) : 
  (∀ x y : ℝ, x + 2*y = 3 → nx + m*y = 4) →  -- Lines exist
  (∃ k : ℝ, ∀ x : ℝ, 
    (3 - x) / 2 = (4 - n*x) / m) →           -- Lines are parallel
  m = 2*n :=                                 -- Conclusion
by sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l1267_126739


namespace NUMINAMATH_CALUDE_expression_evaluation_l1267_126788

theorem expression_evaluation : (3^2 - 3) - 2 * (4^2 - 4) + (5^2 - 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1267_126788


namespace NUMINAMATH_CALUDE_volunteer_selection_l1267_126780

/-- The number of ways to select exactly one person to serve both days
    given 5 volunteers and 2 days of service where 2 people are selected each day. -/
theorem volunteer_selection (n : ℕ) (d : ℕ) (s : ℕ) (p : ℕ) : 
  n = 5 → d = 2 → s = 2 → p = 1 →
  (n.choose p) * ((n - p).choose (s - p)) * ((n - s).choose (s - p)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_selection_l1267_126780


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1267_126710

theorem quadratic_inequality_solution (a : ℝ) : 
  (a > 0 ∧ ∃ x : ℝ, x^2 - 8*x + a < 0) ↔ (0 < a ∧ a < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1267_126710


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1267_126773

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 0 < x ∧ x < 2}

def B : Set ℝ := {x | abs x ≤ 1}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1267_126773


namespace NUMINAMATH_CALUDE_probability_age_21_to_30_l1267_126735

theorem probability_age_21_to_30 (total_people : ℕ) (people_21_to_30 : ℕ) 
  (h1 : total_people = 160) (h2 : people_21_to_30 = 70) : 
  (people_21_to_30 : ℚ) / total_people = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_age_21_to_30_l1267_126735


namespace NUMINAMATH_CALUDE_direct_proportion_unique_k_l1267_126729

/-- A function f: ℝ → ℝ is a direct proportion if there exists a non-zero constant m such that f(x) = m * x for all x -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ (m : ℝ), m ≠ 0 ∧ ∀ x, f x = m * x

/-- The function defined by k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x + k^2 - 1

/-- Theorem stating that k = -1 is the only value that makes f a direct proportion function -/
theorem direct_proportion_unique_k :
  ∃! k, is_direct_proportion (f k) ∧ k = -1 :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_unique_k_l1267_126729


namespace NUMINAMATH_CALUDE_num_tangent_lines_specific_case_l1267_126781

/-- Two circles are internally tangent if the distance between their centers
    equals the absolute difference of their radii. -/
def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d = |r₁ - r₂|

/-- The number of common tangent lines for two internally tangent circles is 1. -/
def num_common_tangents_internal : ℕ := 1

/-- Theorem: For two circles with radii 4 and 5, and distance between centers 3,
    the number of lines simultaneously tangent to both circles is 1. -/
theorem num_tangent_lines_specific_case :
  let r₁ : ℝ := 4
  let r₂ : ℝ := 5
  let d : ℝ := 3
  internally_tangent r₁ r₂ d →
  (num_common_tangents_internal : ℕ) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_num_tangent_lines_specific_case_l1267_126781


namespace NUMINAMATH_CALUDE_mass_of_man_is_90kg_l1267_126799

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_width boat_sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_width * boat_sink_depth * water_density

/-- Theorem stating that the mass of the man is 90 kg under given conditions. -/
theorem mass_of_man_is_90kg :
  let boat_length : ℝ := 3
  let boat_width : ℝ := 2
  let boat_sink_depth : ℝ := 0.015  -- 1.5 cm in meters
  let water_density : ℝ := 1000     -- kg/m³
  mass_of_man boat_length boat_width boat_sink_depth water_density = 90 := by
sorry

#eval mass_of_man 3 2 0.015 1000  -- Should evaluate to 90

end NUMINAMATH_CALUDE_mass_of_man_is_90kg_l1267_126799


namespace NUMINAMATH_CALUDE_perpendicular_construction_l1267_126778

/-- A two-sided ruler with parallel edges -/
structure TwoSidedRuler :=
  (width : ℝ)
  (width_pos : width > 0)

/-- A line in a plane -/
structure Line :=
  (a b c : ℝ)
  (not_all_zero : a ≠ 0 ∨ b ≠ 0)

/-- A point in a plane -/
structure Point :=
  (x y : ℝ)

/-- Checks if a point is on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem -/
theorem perpendicular_construction 
  (l : Line) (M : Point) (h : M.on_line l) :
  ∃ (P : Point), ∃ (n : Line), 
    M.on_line n ∧ P.on_line n ∧ n.perpendicular l :=
sorry

end NUMINAMATH_CALUDE_perpendicular_construction_l1267_126778


namespace NUMINAMATH_CALUDE_total_arms_collected_l1267_126775

theorem total_arms_collected (starfish_count : ℕ) (starfish_arms : ℕ) (seastar_count : ℕ) (seastar_arms : ℕ) :
  starfish_count = 7 →
  starfish_arms = 5 →
  seastar_count = 1 →
  seastar_arms = 14 →
  starfish_count * starfish_arms + seastar_count * seastar_arms = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_total_arms_collected_l1267_126775


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1267_126701

/-- Returns true if n is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- Returns true if n starts with 2 -/
def startsWith2 (n : ℕ) : Prop :=
  20 ≤ n ∧ n ≤ 29

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The smallest two-digit prime number starting with 2 such that 
    reversing its digits produces a composite number is 23 -/
theorem smallest_two_digit_prime_with_composite_reverse : 
  ∃ (n : ℕ), 
    isTwoDigit n ∧ 
    startsWith2 n ∧ 
    Nat.Prime n ∧ 
    ¬(Nat.Prime (reverseDigits n)) ∧
    (∀ m, m < n → ¬(isTwoDigit m ∧ startsWith2 m ∧ Nat.Prime m ∧ ¬(Nat.Prime (reverseDigits m)))) ∧
    n = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1267_126701
