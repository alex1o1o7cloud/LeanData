import Mathlib

namespace NUMINAMATH_CALUDE_coin_array_problem_l2041_204129

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem coin_array_problem :
  ∃ (n : ℕ), triangular_sum n = 3003 ∧ n = 77 ∧ sum_of_digits n = 14 :=
sorry

end NUMINAMATH_CALUDE_coin_array_problem_l2041_204129


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_conditions_l2041_204155

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

def consecutive (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_conditions : 
  let N : ℕ := 2329089562800
  ∀ k : ℕ, k ≤ 30 → k ≠ 28 → k ≠ 29 → is_divisible N k ∧ 
  ¬is_divisible N 28 ∧ 
  ¬is_divisible N 29 ∧
  consecutive 28 29 ∧
  28 > 15 ∧ 29 > 15 ∧
  (∀ m : ℕ, m < N → 
    ¬(∀ j : ℕ, j ≤ 30 → j ≠ 28 → j ≠ 29 → is_divisible m j) ∨ 
    is_divisible m 28 ∨ 
    is_divisible m 29 ∨
    ¬(∃ p q : ℕ, p > 15 ∧ q > 15 ∧ consecutive p q ∧ ¬is_divisible m p ∧ ¬is_divisible m q)
  ) :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_with_divisibility_conditions_l2041_204155


namespace NUMINAMATH_CALUDE_increasing_absolute_value_function_l2041_204151

-- Define the function f(x) = |x - a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem increasing_absolute_value_function (a : ℝ) :
  (∀ x y, 1 ≤ x → x < y → f a x < f a y) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_absolute_value_function_l2041_204151


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2041_204125

theorem proposition_equivalence (m : ℝ) :
  (∃ x : ℝ, -x^2 - 2*m*x + 2*m - 3 ≥ 0) ↔ (m ≤ -3 ∨ m ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2041_204125


namespace NUMINAMATH_CALUDE_cell_count_after_12_days_first_six_days_growth_next_six_days_growth_l2041_204154

/-- Represents the cell growth model over 12 days -/
def CellGrowth : Nat → Nat
| 0 => 5  -- Initial cell count
| n + 1 =>
  if n < 6 then
    CellGrowth n * 3  -- Growth rate for first 6 days
  else if n < 12 then
    CellGrowth n * 2  -- Growth rate for next 6 days
  else
    CellGrowth n      -- No growth after 12 days

/-- Theorem stating the number of cells after 12 days -/
theorem cell_count_after_12_days :
  CellGrowth 12 = 180 := by
  sorry

/-- Verifies the growth pattern for the first 6 days -/
theorem first_six_days_growth (n : Nat) (h : n < 6) :
  CellGrowth (n + 1) = CellGrowth n * 3 := by
  sorry

/-- Verifies the growth pattern for days 7 to 12 -/
theorem next_six_days_growth (n : Nat) (h1 : 6 ≤ n) (h2 : n < 12) :
  CellGrowth (n + 1) = CellGrowth n * 2 := by
  sorry

end NUMINAMATH_CALUDE_cell_count_after_12_days_first_six_days_growth_next_six_days_growth_l2041_204154


namespace NUMINAMATH_CALUDE_clips_ratio_april_to_may_l2041_204121

def clips_sold_april : ℕ := 48
def total_clips_sold : ℕ := 72

theorem clips_ratio_april_to_may :
  (clips_sold_april : ℚ) / (total_clips_sold - clips_sold_april : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_clips_ratio_april_to_may_l2041_204121


namespace NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l2041_204165

theorem neither_necessary_nor_sufficient (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) :=
sorry

end NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l2041_204165


namespace NUMINAMATH_CALUDE_integral_shift_reciprocal_l2041_204195

/-- For a continuous function f: ℝ → ℝ, if the integral of f over the real line exists,
    then the integral of f(x - 1/x) over the real line equals the integral of f. -/
theorem integral_shift_reciprocal (f : ℝ → ℝ) (hf : Continuous f) 
  (L : ℝ) (hL : ∫ (x : ℝ), f x = L) :
  ∫ (x : ℝ), f (x - 1/x) = L := by
  sorry

end NUMINAMATH_CALUDE_integral_shift_reciprocal_l2041_204195


namespace NUMINAMATH_CALUDE_perimeter_ABCDE_l2041_204116

-- Define the points as 2D vectors
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom AB_eq_BC : dist A B = dist B C
axiom AB_eq_4 : dist A B = 4
axiom AE_eq_5 : dist A E = 5
axiom ED_eq_8 : dist E D = 8
axiom right_angle_AEB : (B.1 - E.1) * (A.2 - E.2) = (A.1 - E.1) * (B.2 - E.2)
axiom right_angle_BAE : (E.1 - A.1) * (B.2 - A.2) = (B.1 - A.1) * (E.2 - A.2)
axiom right_angle_ABC : (C.1 - B.1) * (A.2 - B.2) = (A.1 - B.1) * (C.2 - B.2)

-- Define the perimeter function
def perimeter (A B C D E : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C D + dist D E + dist E A

-- State the theorem
theorem perimeter_ABCDE :
  perimeter A B C D E = 21 + Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_perimeter_ABCDE_l2041_204116


namespace NUMINAMATH_CALUDE_gas_cans_volume_l2041_204139

/-- The volume of gas needed to fill a given number of gas cans with a specified capacity. -/
def total_gas_volume (num_cans : ℕ) (can_capacity : ℝ) : ℝ :=
  num_cans * can_capacity

/-- Theorem: The total volume of gas needed to fill 4 gas cans, each with a capacity of 5.0 gallons, is equal to 20.0 gallons. -/
theorem gas_cans_volume :
  total_gas_volume 4 5.0 = 20.0 := by
  sorry

end NUMINAMATH_CALUDE_gas_cans_volume_l2041_204139


namespace NUMINAMATH_CALUDE_diagonal_sequence_theorem_l2041_204192

/-- A convex polygon with 1994 sides and 997 diagonals -/
structure ConvexPolygon :=
  (sides : ℕ)
  (diagonals : ℕ)
  (is_convex : Bool)
  (sides_eq : sides = 1994)
  (diagonals_eq : diagonals = 997)
  (convex : is_convex = true)

/-- The length of a diagonal is the number of sides in the smaller part of the perimeter it divides -/
def diagonal_length (p : ConvexPolygon) (d : ℕ) : ℕ := sorry

/-- Each vertex has exactly one diagonal emanating from it -/
def one_diagonal_per_vertex (p : ConvexPolygon) : Prop := sorry

/-- The sequence of diagonal lengths in decreasing order -/
def diagonal_sequence (p : ConvexPolygon) : List ℕ := sorry

theorem diagonal_sequence_theorem (p : ConvexPolygon) 
  (h : one_diagonal_per_vertex p) :
  (∃ (seq : List ℕ), diagonal_sequence p = seq ∧ 
    seq.length = 997 ∧
    seq.count 3 = 991 ∧ 
    seq.count 2 = 6) ∧
  ¬(∃ (seq : List ℕ), diagonal_sequence p = seq ∧ 
    seq.length = 997 ∧
    seq.count 8 = 4 ∧ 
    seq.count 6 = 985 ∧ 
    seq.count 3 = 8) :=
sorry

end NUMINAMATH_CALUDE_diagonal_sequence_theorem_l2041_204192


namespace NUMINAMATH_CALUDE_cube_cross_sections_l2041_204152

/-- A regular polygon obtained by cutting a cube with a plane. -/
inductive CubeCrossSection
  | Triangle
  | Square
  | Hexagon

/-- The set of all possible regular polygons obtained by cutting a cube with a plane. -/
def ValidCubeCrossSections : Set CubeCrossSection :=
  {CubeCrossSection.Triangle, CubeCrossSection.Square, CubeCrossSection.Hexagon}

/-- Theorem: The only regular polygons that can be obtained by cutting a cube with a plane
    are triangles, squares, and hexagons. -/
theorem cube_cross_sections (cs : CubeCrossSection) :
  cs ∈ ValidCubeCrossSections := by sorry

end NUMINAMATH_CALUDE_cube_cross_sections_l2041_204152


namespace NUMINAMATH_CALUDE_room_length_proof_l2041_204133

/-- Proves that a room with given width, paving cost per area, and total paving cost has a specific length -/
theorem room_length_proof (width : ℝ) (cost_per_area : ℝ) (total_cost : ℝ) :
  width = 3.75 →
  cost_per_area = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_area) / width = 5.5 := by
  sorry

#check room_length_proof

end NUMINAMATH_CALUDE_room_length_proof_l2041_204133


namespace NUMINAMATH_CALUDE_sqrt_comparison_quadratic_inequality_solution_l2041_204104

-- Part 1
theorem sqrt_comparison : Real.sqrt 7 + Real.sqrt 10 > Real.sqrt 3 + Real.sqrt 14 := by sorry

-- Part 2
theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x, -1/2 * x^2 + 2*x > m*x ↔ 0 < x ∧ x < 2) → m = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_comparison_quadratic_inequality_solution_l2041_204104


namespace NUMINAMATH_CALUDE_complex_modulus_l2041_204123

theorem complex_modulus (z : ℂ) (h : z + 2*Complex.I - 3 = 3 - 3*Complex.I) : 
  Complex.abs z = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2041_204123


namespace NUMINAMATH_CALUDE_probability_at_least_four_out_of_five_l2041_204112

theorem probability_at_least_four_out_of_five (p : ℝ) (h : p = 4/5) :
  let binomial (n k : ℕ) := Nat.choose n k
  let prob_exactly (k : ℕ) := (binomial 5 k : ℝ) * p^k * (1 - p)^(5 - k)
  prob_exactly 4 + prob_exactly 5 = 2304/3125 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_four_out_of_five_l2041_204112


namespace NUMINAMATH_CALUDE_problem_statement_l2041_204166

theorem problem_statement (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  let expr1 := (1 + a*b)/(a - b) * (1 + b*c)/(b - c) + 
               (1 + b*c)/(b - c) * (1 + c*a)/(c - a) + 
               (1 + c*a)/(c - a) * (1 + a*b)/(a - b)
  let expr2 := (1 - a*b)/(a - b) * (1 - b*c)/(b - c) + 
               (1 - b*c)/(b - c) * (1 - c*a)/(c - a) + 
               (1 - c*a)/(c - a) * (1 - a*b)/(a - b)
  let expr3 := (1 + a^2*b^2)/(a - b)^2 + (1 + b^2*c^2)/(b - c)^2 + (1 + c^2*a^2)/(c - a)^2
  (expr1 = 1) ∧ 
  (expr2 = -1) ∧ 
  (expr3 ≥ (3/2)) ∧ 
  (expr3 = (3/2) ↔ a = b ∨ b = c ∨ c = a) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2041_204166


namespace NUMINAMATH_CALUDE_sisters_portions_l2041_204170

/-- Represents the types of granola bars --/
inductive BarType
  | ChocolateChip
  | OatAndHoney
  | PeanutButter

/-- Represents the number of bars of each type --/
structure BarCounts where
  chocolateChip : ℕ
  oatAndHoney : ℕ
  peanutButter : ℕ

/-- Represents the initial distribution of bars --/
def initialDistribution : BarCounts :=
  { chocolateChip := 8, oatAndHoney := 6, peanutButter := 6 }

/-- Represents the bars set aside daily --/
def dailySetAside : BarCounts :=
  { chocolateChip := 3, oatAndHoney := 2, peanutButter := 2 }

/-- Represents the bars left after setting aside --/
def afterSetAside : BarCounts :=
  { chocolateChip := 5, oatAndHoney := 4, peanutButter := 4 }

/-- Represents the bars traded --/
def traded : BarCounts :=
  { chocolateChip := 2, oatAndHoney := 4, peanutButter := 0 }

/-- Represents the bars left after trading --/
def afterTrading : BarCounts :=
  { chocolateChip := 3, oatAndHoney := 0, peanutButter := 3 }

/-- Represents the whole bars given to each sister --/
def wholeBarsGiven (type : BarType) : ℕ := 2

/-- Theorem: Each sister receives 2.5 portions of their preferred granola bar type --/
theorem sisters_portions (type : BarType) : 
  (wholeBarsGiven type : ℚ) + (1 : ℚ) / 2 = (5 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sisters_portions_l2041_204170


namespace NUMINAMATH_CALUDE_simple_interest_years_l2041_204187

/-- Given a principal amount and the additional interest earned from a 1% rate increase,
    calculate the number of years the sum was put at simple interest. -/
theorem simple_interest_years (principal : ℝ) (additional_interest : ℝ) : 
  principal = 2400 →
  additional_interest = 72 →
  (principal * 0.01 * (3 : ℝ)) = additional_interest :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_years_l2041_204187


namespace NUMINAMATH_CALUDE_det_special_matrix_l2041_204108

theorem det_special_matrix (k a b : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![1, k * Real.sin (a - b), Real.sin a],
                                        ![k * Real.sin (a - b), 1, k * Real.sin b],
                                        ![Real.sin a, k * Real.sin b, 1]]
  Matrix.det M = 1 - Real.sin a ^ 2 - k ^ 2 * Real.sin b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l2041_204108


namespace NUMINAMATH_CALUDE_inequality_solution_l2041_204196

-- Define the inequality function
def inequality (x : ℝ) : Prop :=
  9.216 * (Real.log x / Real.log 5) + (Real.log x - Real.log 3) / (Real.log x)
  < ((Real.log x / Real.log 5) * (2 - Real.log x / Real.log 3)) / (Real.log x / Real.log 3)

-- State the theorem
theorem inequality_solution :
  ∀ x : ℝ, 
  x > 0 → 
  inequality x ↔ (0 < x ∧ x < 1 / Real.sqrt 5) ∨ (1 < x ∧ x < 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2041_204196


namespace NUMINAMATH_CALUDE_parallel_vectors_solution_l2041_204199

open Real

theorem parallel_vectors_solution (x : ℝ) :
  let a : ℝ × ℝ := (sin x, 3/4)
  let b : ℝ × ℝ := (1/3, (1/2) * cos x)
  let c : ℝ × ℝ := (1/6, cos x)
  0 < x ∧ x < (5 * π) / 12 ∧ 
  (∃ (k : ℝ), k * a.1 = (b.1 + c.1) ∧ k * a.2 = (b.2 + c.2)) →
  x = π / 12 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_solution_l2041_204199


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2041_204103

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^3 + 2*x^2 = (x^2 + 3*x + 2) * q + (x + 2) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2041_204103


namespace NUMINAMATH_CALUDE_fruit_price_adjustment_l2041_204131

/-- Represents the problem of adjusting fruit quantities to achieve a desired average price --/
theorem fruit_price_adjustment
  (apple_price : ℚ)
  (orange_price : ℚ)
  (total_fruits : ℕ)
  (initial_avg_price : ℚ)
  (desired_avg_price : ℚ)
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruits = 10)
  (h4 : initial_avg_price = 52/100)
  (h5 : desired_avg_price = 44/100)
  : ∃ (oranges_to_remove : ℕ),
    oranges_to_remove = 5 ∧
    ∃ (apples : ℕ) (oranges : ℕ),
      apples + oranges = total_fruits ∧
      (apple_price * apples + orange_price * oranges) / total_fruits = initial_avg_price ∧
      (apple_price * apples + orange_price * (oranges - oranges_to_remove)) / (total_fruits - oranges_to_remove) = desired_avg_price :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_price_adjustment_l2041_204131


namespace NUMINAMATH_CALUDE_cards_rick_keeps_rick_keeps_fifteen_cards_l2041_204184

/-- The number of cards Rick keeps for himself given the initial number of cards and the distribution to others. -/
theorem cards_rick_keeps (initial_cards : ℕ) (cards_to_miguel : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) (num_sisters : ℕ) (cards_per_sister : ℕ) : ℕ :=
  initial_cards - cards_to_miguel - (num_friends * cards_per_friend) - (num_sisters * cards_per_sister)

/-- Proof that Rick keeps 15 cards for himself -/
theorem rick_keeps_fifteen_cards :
  cards_rick_keeps 130 13 8 12 2 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cards_rick_keeps_rick_keeps_fifteen_cards_l2041_204184


namespace NUMINAMATH_CALUDE_frequency_distribution_purpose_l2041_204169

/-- A frequency distribution table showing sample data sizes in groups -/
structure FrequencyDistributionTable where
  groups : Set (ℕ → ℕ)  -- Each function represents a group mapping sample size to frequency

/-- The proportion of data in each group -/
def proportion (t : FrequencyDistributionTable) : Set (ℕ → ℝ) :=
  sorry

/-- The overall corresponding situation being estimated -/
def overallSituation (t : FrequencyDistributionTable) : Type :=
  sorry

/-- Theorem stating the equivalence between the frequency distribution table
    and understanding proportions and estimating the overall situation -/
theorem frequency_distribution_purpose (t : FrequencyDistributionTable) :
  (∃ p : Set (ℕ → ℝ), p = proportion t) ∧
  (∃ s : Type, s = overallSituation t) :=
sorry

end NUMINAMATH_CALUDE_frequency_distribution_purpose_l2041_204169


namespace NUMINAMATH_CALUDE_largest_attendance_difference_largest_attendance_difference_holds_l2041_204171

/-- The largest possible difference between attendances in Chicago and Detroit --/
theorem largest_attendance_difference : ℝ → Prop :=
  fun max_diff =>
  ∀ (chicago_actual detroit_actual : ℝ),
  (chicago_actual ≥ 80000 * 0.95 ∧ chicago_actual ≤ 80000 * 1.05) →
  (detroit_actual ≥ 95000 / 1.15 ∧ detroit_actual ≤ 95000 / 0.85) →
  max_diff = 36000 ∧
  ∀ (diff : ℝ),
  diff ≤ detroit_actual - chicago_actual →
  ⌊diff / 1000⌋ * 1000 ≤ max_diff

/-- The theorem holds --/
theorem largest_attendance_difference_holds :
  largest_attendance_difference 36000 := by sorry

end NUMINAMATH_CALUDE_largest_attendance_difference_largest_attendance_difference_holds_l2041_204171


namespace NUMINAMATH_CALUDE_total_marbles_l2041_204110

theorem total_marbles (num_boxes : ℕ) (marbles_per_box : ℕ) 
  (h1 : num_boxes = 10) (h2 : marbles_per_box = 100) : 
  num_boxes * marbles_per_box = 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l2041_204110


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2041_204157

theorem sum_of_fractions : (2 : ℚ) / 5 + (3 : ℚ) / 10 = (7 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2041_204157


namespace NUMINAMATH_CALUDE_remaining_sugar_is_one_l2041_204135

/-- Represents the recipe and Mary's baking process -/
structure Recipe where
  sugar_total : ℕ
  salt_total : ℕ
  flour_added : ℕ
  sugar_salt_diff : ℕ

/-- Calculates the remaining sugar to be added based on the recipe and current state -/
def remaining_sugar (r : Recipe) : ℕ :=
  r.sugar_total - (r.salt_total + r.sugar_salt_diff)

/-- Theorem stating that the remaining sugar to be added is 1 cup -/
theorem remaining_sugar_is_one (r : Recipe) 
  (h1 : r.sugar_total = 8)
  (h2 : r.salt_total = 7)
  (h3 : r.flour_added = 5)
  (h4 : r.sugar_salt_diff = 1) : 
  remaining_sugar r = 1 := by
  sorry

#check remaining_sugar_is_one

end NUMINAMATH_CALUDE_remaining_sugar_is_one_l2041_204135


namespace NUMINAMATH_CALUDE_largest_quantity_l2041_204180

theorem largest_quantity (a b c d : ℝ) (h : a + 1 = b - 3 ∧ a + 1 = c + 4 ∧ a + 1 = d - 2) :
  b ≥ a ∧ b ≥ c ∧ b ≥ d :=
by sorry

end NUMINAMATH_CALUDE_largest_quantity_l2041_204180


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2041_204188

theorem arithmetic_sequence_sum (n : ℕ) (sum : ℕ) (d : ℕ) : 
  n = 4020 →
  d = 2 →
  sum = 10614 →
  ∃ (a : ℕ), 
    (a + (n - 1) * d / 2) * n = sum ∧
    a + (n / 2 - 1) * (2 * d) = 3297 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2041_204188


namespace NUMINAMATH_CALUDE_football_throw_distance_l2041_204120

theorem football_throw_distance (parker_distance : ℝ) :
  let grant_distance := parker_distance * 1.25
  let kyle_distance := grant_distance * 2
  kyle_distance - parker_distance = 24 →
  parker_distance = 16 := by
sorry

end NUMINAMATH_CALUDE_football_throw_distance_l2041_204120


namespace NUMINAMATH_CALUDE_probability_at_least_six_fives_value_l2041_204141

/-- The probability of rolling at least a five on a fair die -/
def p_at_least_five : ℚ := 1/3

/-- The probability of not rolling at least a five on a fair die -/
def p_not_at_least_five : ℚ := 2/3

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The minimum number of successful rolls (at least a five) -/
def min_success : ℕ := 6

/-- Calculates the probability of exactly k successes in n trials -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1-p)^(n-k)

/-- The probability of rolling at least a five at least six times in eight rolls -/
def probability_at_least_six_fives : ℚ :=
  binomial_probability num_rolls min_success p_at_least_five +
  binomial_probability num_rolls (min_success + 1) p_at_least_five +
  binomial_probability num_rolls (min_success + 2) p_at_least_five

theorem probability_at_least_six_fives_value :
  probability_at_least_six_fives = 129/6561 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_six_fives_value_l2041_204141


namespace NUMINAMATH_CALUDE_G_is_leftmost_l2041_204176

/-- Represents a square with four integer labels -/
structure Square where
  name : Char
  w : Int
  x : Int
  y : Int
  z : Int

/-- The set of all squares -/
def squares : Finset Square := sorry

/-- Predicate to check if a square is leftmost -/
def is_leftmost (s : Square) : Prop := sorry

/-- The squares are arranged in a row without rotating or reflecting -/
axiom squares_in_row : sorry

/-- All squares are distinct -/
axiom squares_distinct : sorry

/-- The specific squares given in the problem -/
def F : Square := ⟨'F', 5, 1, 7, 9⟩
def G : Square := ⟨'G', 1, 0, 4, 6⟩
def H : Square := ⟨'H', 4, 8, 6, 2⟩
def I : Square := ⟨'I', 8, 5, 3, 7⟩
def J : Square := ⟨'J', 9, 2, 8, 0⟩

/-- All given squares are in the set of squares -/
axiom all_squares_in_set : F ∈ squares ∧ G ∈ squares ∧ H ∈ squares ∧ I ∈ squares ∧ J ∈ squares

/-- Theorem: Square G is the leftmost square -/
theorem G_is_leftmost : is_leftmost G := by sorry

end NUMINAMATH_CALUDE_G_is_leftmost_l2041_204176


namespace NUMINAMATH_CALUDE_swim_meet_capacity_theorem_l2041_204156

/-- Represents the swimming club's transportation scenario -/
structure SwimMeetTransport where
  num_cars : ℕ
  num_vans : ℕ
  people_per_car : ℕ
  people_per_van : ℕ
  max_car_capacity : ℕ
  max_van_capacity : ℕ

/-- Calculates the number of additional people that could have ridden with the swim team -/
def additional_capacity (t : SwimMeetTransport) : ℕ :=
  (t.num_cars * t.max_car_capacity + t.num_vans * t.max_van_capacity) -
  (t.num_cars * t.people_per_car + t.num_vans * t.people_per_van)

/-- Theorem stating that 17 more people could have ridden with the swim team -/
theorem swim_meet_capacity_theorem (t : SwimMeetTransport)
  (h1 : t.num_cars = 2)
  (h2 : t.num_vans = 3)
  (h3 : t.people_per_car = 5)
  (h4 : t.people_per_van = 3)
  (h5 : t.max_car_capacity = 6)
  (h6 : t.max_van_capacity = 8) :
  additional_capacity t = 17 := by
  sorry

#eval additional_capacity {
  num_cars := 2,
  num_vans := 3,
  people_per_car := 5,
  people_per_van := 3,
  max_car_capacity := 6,
  max_van_capacity := 8
}

end NUMINAMATH_CALUDE_swim_meet_capacity_theorem_l2041_204156


namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l2041_204193

theorem complex_arithmetic_equation : 
  ((4501 * 2350) - (7125 / 9)) + (3250 ^ 2) * 4167 = 44045164058.33 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l2041_204193


namespace NUMINAMATH_CALUDE_total_registration_methods_l2041_204140

-- Define the number of students and clubs
def num_students : Nat := 5
def num_clubs : Nat := 3

-- Define the students with restrictions
structure RestrictedStudent where
  name : String
  restricted_club : Nat

-- Define the list of restricted students
def restricted_students : List RestrictedStudent := [
  { name := "Xiao Bin", restricted_club := 1 },  -- 1 represents chess club
  { name := "Xiao Cong", restricted_club := 0 }, -- 0 represents basketball club
  { name := "Xiao Hao", restricted_club := 2 }   -- 2 represents environmental club
]

-- Define the theorem
theorem total_registration_methods :
  (restricted_students.length * 2 + (num_students - restricted_students.length) * num_clubs) ^ num_students = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_registration_methods_l2041_204140


namespace NUMINAMATH_CALUDE_unique_valid_n_l2041_204173

def is_valid_n (n : ℕ+) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ : ℕ+),
    d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧
    (∀ (d : ℕ+), d ∣ n → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d₄ < d) ∧
    n = d₁^2 + d₂^2 + d₃^2 + d₄^2

theorem unique_valid_n :
  ∃! (n : ℕ+), is_valid_n n ∧ n = 130 := by sorry

end NUMINAMATH_CALUDE_unique_valid_n_l2041_204173


namespace NUMINAMATH_CALUDE_min_folders_required_l2041_204194

/-- Represents the types of files --/
inductive FileType
  | PDF
  | Word
  | PPT

/-- Represents the initial file counts --/
structure InitialFiles where
  pdf : Nat
  word : Nat
  ppt : Nat

/-- Represents the deleted file counts --/
structure DeletedFiles where
  pdf : Nat
  ppt : Nat

/-- Calculates the remaining files after deletion --/
def remainingFiles (initial : InitialFiles) (deleted : DeletedFiles) : Nat :=
  initial.pdf + initial.word + initial.ppt - deleted.pdf - deleted.ppt

/-- Represents the folder allocation problem --/
structure FolderAllocationProblem where
  initial : InitialFiles
  deleted : DeletedFiles
  folderCapacity : Nat
  wordImportance : Nat

/-- Theorem: The minimum number of folders required is 6 --/
theorem min_folders_required (problem : FolderAllocationProblem)
  (h1 : problem.initial = ⟨43, 30, 30⟩)
  (h2 : problem.deleted = ⟨33, 30⟩)
  (h3 : problem.folderCapacity = 7)
  (h4 : problem.wordImportance = 2) :
  let remainingWordFiles := problem.initial.word
  let remainingPDFFiles := problem.initial.pdf - problem.deleted.pdf
  let totalRemainingFiles := remainingFiles problem.initial problem.deleted
  let minFolders := 
    (remainingWordFiles / problem.folderCapacity) +
    ((remainingWordFiles % problem.folderCapacity + remainingPDFFiles + problem.folderCapacity - 1) / problem.folderCapacity)
  minFolders = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_folders_required_l2041_204194


namespace NUMINAMATH_CALUDE_production_exceeds_target_in_seventh_year_l2041_204161

-- Define the initial production and growth rate
def initial_production : ℝ := 40000
def growth_rate : ℝ := 1.2

-- Define the target production
def target_production : ℝ := 120000

-- Define the function to calculate production after n years
def production (n : ℕ) : ℝ := initial_production * growth_rate ^ n

-- Theorem statement
theorem production_exceeds_target_in_seventh_year :
  ∀ n : ℕ, n < 7 → production n ≤ target_production ∧
  production 7 > target_production :=
sorry

end NUMINAMATH_CALUDE_production_exceeds_target_in_seventh_year_l2041_204161


namespace NUMINAMATH_CALUDE_prize_selection_ways_l2041_204109

/-- The number of ways to select prize winners from finalists -/
def select_winners (n : ℕ) : ℕ :=
  n * (n - 1).choose 2

/-- Theorem stating that selecting 1 first prize, 2 second prizes, and 3 third prizes
    from 6 finalists can be done in 60 ways -/
theorem prize_selection_ways : select_winners 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_prize_selection_ways_l2041_204109


namespace NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l2041_204111

/-- Given two lines in a plane, where one is y = 3x + 4 and the other is perpendicular to it
    passing through the point (3, 2), their intersection point is (3/10, 49/10). -/
theorem intersection_of_perpendicular_lines 
  (line1 : ℝ → ℝ → Prop) 
  (line2 : ℝ → ℝ → Prop) 
  (h1 : ∀ x y, line1 x y ↔ y = 3 * x + 4)
  (h2 : ∀ x y, line2 x y → (y - 2) = -(1/3) * (x - 3))
  (h3 : line2 3 2)
  : ∃ x y, line1 x y ∧ line2 x y ∧ x = 3/10 ∧ y = 49/10 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l2041_204111


namespace NUMINAMATH_CALUDE_equation_solution_range_l2041_204159

theorem equation_solution_range (a : ℝ) : 
  ∃ x : ℝ, 
    ((x - 3) / (x - 2) + 1 = 3 / (2 - x)) ∧ 
    ((2 - a) * x - 3 > 0) → 
    a < -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2041_204159


namespace NUMINAMATH_CALUDE_fish_count_approximation_l2041_204101

/-- Approximates the total number of fish in a pond based on a tagging and recapture experiment. -/
def approximate_fish_count (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) : ℕ :=
  (initial_tagged * second_catch) / tagged_in_second

/-- Theorem stating that under the given conditions, the approximate number of fish in the pond is 313. -/
theorem fish_count_approximation :
  let initial_tagged := 50
  let second_catch := 50
  let tagged_in_second := 8
  approximate_fish_count initial_tagged second_catch tagged_in_second = 313 :=
by
  sorry

#eval approximate_fish_count 50 50 8

end NUMINAMATH_CALUDE_fish_count_approximation_l2041_204101


namespace NUMINAMATH_CALUDE_flagpole_height_l2041_204143

/-- The height of a flagpole given shadow lengths -/
theorem flagpole_height (flagpole_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ)
  (h1 : flagpole_shadow = 90)
  (h2 : tree_height = 15)
  (h3 : tree_shadow = 30)
  : ∃ (flagpole_height : ℝ), flagpole_height = 45 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_l2041_204143


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2041_204190

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℤ, (x - 2) * (x + 1) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x - 8)) →
  p = -1 ∧ q = -10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2041_204190


namespace NUMINAMATH_CALUDE_amanda_summer_work_hours_l2041_204181

/-- Calculates the required weekly work hours for Amanda during summer -/
theorem amanda_summer_work_hours 
  (winter_weekly_hours : ℝ) 
  (winter_weeks : ℝ) 
  (winter_earnings : ℝ) 
  (summer_weeks : ℝ) 
  (summer_earnings : ℝ) 
  (h1 : winter_weekly_hours = 45) 
  (h2 : winter_weeks = 8) 
  (h3 : winter_earnings = 3600) 
  (h4 : summer_weeks = 20) 
  (h5 : summer_earnings = 4500) :
  (summer_earnings / (winter_earnings / (winter_weekly_hours * winter_weeks))) / summer_weeks = 22.5 := by
  sorry

#check amanda_summer_work_hours

end NUMINAMATH_CALUDE_amanda_summer_work_hours_l2041_204181


namespace NUMINAMATH_CALUDE_equation_solution_l2041_204162

theorem equation_solution : ∃! x : ℚ, (3*x^2 + 2*x + 1) / (x - 1) = 3*x + 1 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2041_204162


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2041_204144

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2) ↔ (a = 1 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2041_204144


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l2041_204153

/-- Given a square divided into four congruent rectangles, 
    if the perimeter of each rectangle is 30 inches, 
    then the perimeter of the square is 48 inches. -/
theorem square_perimeter_from_rectangle_perimeter :
  ∀ s : ℝ,
  s > 0 →
  (2 * s + 2 * (s / 4) = 30) →
  (4 * s = 48) :=
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l2041_204153


namespace NUMINAMATH_CALUDE_danny_bottle_caps_indeterminate_l2041_204179

/-- Represents Danny's collection of bottle caps and wrappers --/
structure Collection where
  bottle_caps : ℕ
  wrappers : ℕ

/-- The problem statement --/
theorem danny_bottle_caps_indeterminate 
  (initial : Collection) 
  (park_found : Collection)
  (final_wrappers : ℕ) :
  park_found.bottle_caps = 22 →
  park_found.wrappers = 30 →
  final_wrappers = initial.wrappers + park_found.wrappers →
  final_wrappers = 57 →
  ¬∃ (n : ℕ), ∀ (x : ℕ), initial.bottle_caps = n ∨ initial.bottle_caps ≠ x :=
by sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_indeterminate_l2041_204179


namespace NUMINAMATH_CALUDE_rational_expression_equality_algebraic_expression_equality_l2041_204160

/-- Prove the equality of the given rational expression -/
theorem rational_expression_equality (m : ℝ) (hm1 : m ≠ -4) (hm2 : m ≠ -2) : 
  (m^2 - 16) / (m^2 + 8*m + 16) / ((m - 4) / (2*m + 8)) * ((m - 2) / (m + 2)) = 2*(m - 2) / (m + 2) := by
  sorry

/-- Prove the equality of the given algebraic expression -/
theorem algebraic_expression_equality (x : ℝ) (hx1 : x ≠ -2) (hx2 : x ≠ 2) : 
  3 / (x + 2) + 1 / (2 - x) - (2*x) / (4 - x^2) = 4 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_equality_algebraic_expression_equality_l2041_204160


namespace NUMINAMATH_CALUDE_independence_test_most_appropriate_l2041_204150

/-- Represents the survey data --/
structure SurveyData where
  male_total : Nat
  male_opposing : Nat
  female_total : Nat
  female_opposing : Nat

/-- Represents different statistical methods --/
inductive StatisticalMethod
  | MeanAndVariance
  | RegressionLine
  | IndependenceTest
  | Probability

/-- Determines the most appropriate method for analyzing the relationship between gender and judgment --/
def most_appropriate_method (data : SurveyData) : StatisticalMethod :=
  StatisticalMethod.IndependenceTest

/-- Theorem stating that the Independence test is the most appropriate method for the given survey data --/
theorem independence_test_most_appropriate (data : SurveyData) :
  most_appropriate_method data = StatisticalMethod.IndependenceTest :=
sorry

end NUMINAMATH_CALUDE_independence_test_most_appropriate_l2041_204150


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2041_204106

theorem polynomial_factorization (x : ℝ) :
  9 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (3 * x^2 + 52 * x + 231) * (3 * x^2 + 56 * x + 231) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2041_204106


namespace NUMINAMATH_CALUDE_all_methods_applicable_l2041_204138

structure Population where
  total : Nat
  farmers : Nat
  workers : Nat
  sample_size : Nat

inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

def is_applicable (pop : Population) (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.SimpleRandom => pop.workers > 0
  | SamplingMethod.Systematic => pop.farmers > 0
  | SamplingMethod.Stratified => pop.farmers ≠ pop.workers

theorem all_methods_applicable (pop : Population) 
  (h1 : pop.total = 2004)
  (h2 : pop.farmers = 1600)
  (h3 : pop.workers = 303)
  (h4 : pop.sample_size = 40) :
  (∀ m : SamplingMethod, is_applicable pop m) :=
by sorry

end NUMINAMATH_CALUDE_all_methods_applicable_l2041_204138


namespace NUMINAMATH_CALUDE_average_running_time_l2041_204174

theorem average_running_time (f : ℕ) : 
  let third_graders := 9 * f
  let fourth_graders := 3 * f
  let fifth_graders := f
  let total_students := third_graders + fourth_graders + fifth_graders
  let total_minutes := 10 * third_graders + 18 * fourth_graders + 12 * fifth_graders
  (total_minutes : ℚ) / total_students = 12 := by
sorry

end NUMINAMATH_CALUDE_average_running_time_l2041_204174


namespace NUMINAMATH_CALUDE_seventeen_integer_chords_l2041_204124

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceToP : ℝ

/-- Counts the number of integer-length chords containing P in the given circle -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem seventeen_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 13) 
  (h2 : c.distanceToP = 12) : 
  countIntegerChords c = 17 :=
sorry

end NUMINAMATH_CALUDE_seventeen_integer_chords_l2041_204124


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l2041_204167

/-- Given a right triangle OAB with O at the origin, this structure represents the hyperbola
    y = k/x passing through the midpoint of OB and intersecting AB at C. -/
structure Hyperbola_Triangle :=
  (a b : ℝ)  -- Coordinates of point B (a, b)
  (k : ℝ)    -- Parameter of the hyperbola y = k/x
  (h_k_pos : k > 0)  -- k is positive
  (h_right_triangle : a * b = 2 * 3)  -- Area of OAB is 3, so a * b / 2 = 3
  (h_midpoint : k / (a/2) = b/2)  -- Hyperbola passes through midpoint of OB
  (c : ℝ)    -- x-coordinate of point C
  (h_c_on_ab : 0 < c ∧ c < a)  -- C is between O and B on AB
  (h_c_on_hyperbola : k / c = b * (1 - c/a))  -- C is on the hyperbola

/-- The main theorem: if the area of OBC is 3, then k = 2 -/
theorem hyperbola_triangle_area (ht : Hyperbola_Triangle) 
  (h_area_obc : ht.a * ht.b * (1 - ht.c/ht.a) / 2 = 3) : ht.k = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l2041_204167


namespace NUMINAMATH_CALUDE_prime_has_property_P_infinitely_many_composite_with_property_P_l2041_204183

-- Define property P
def has_property_P (n : ℕ) : Prop :=
  ∀ a : ℕ, a > 0 → (n ∣ a^n - 1) → (n^2 ∣ a^n - 1)

-- Theorem 1: Every prime number has property P
theorem prime_has_property_P (p : ℕ) (hp : Prime p) : has_property_P p := by
  sorry

-- Theorem 2: There are infinitely many composite numbers with property P
theorem infinitely_many_composite_with_property_P :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ¬Prime m ∧ has_property_P m := by
  sorry

end NUMINAMATH_CALUDE_prime_has_property_P_infinitely_many_composite_with_property_P_l2041_204183


namespace NUMINAMATH_CALUDE_beaker_volume_difference_l2041_204137

theorem beaker_volume_difference (total_volume : ℝ) (beaker_one_volume : ℝ) 
  (h1 : total_volume = 9.28)
  (h2 : beaker_one_volume = 2.95) : 
  abs (beaker_one_volume - (total_volume - beaker_one_volume)) = 3.38 := by
  sorry

end NUMINAMATH_CALUDE_beaker_volume_difference_l2041_204137


namespace NUMINAMATH_CALUDE_variable_value_proof_l2041_204132

theorem variable_value_proof : ∃ x : ℝ, 3 * x + 36 = 48 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_variable_value_proof_l2041_204132


namespace NUMINAMATH_CALUDE_set_operations_l2041_204128

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 + 2*x - 3 > 0}

-- Define the theorem
theorem set_operations :
  (Set.compl (A ∪ B) = {x | -3 ≤ x ∧ x ≤ 0}) ∧
  ((Set.compl A) ∩ B = {x | x > 1 ∨ x < -3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2041_204128


namespace NUMINAMATH_CALUDE_v_2010_equals_0_l2041_204146

-- Define the function g
def g : ℕ → ℕ
| 0 => 2
| 1 => 5
| 2 => 3
| 3 => 0
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, although not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 0
| (n + 1) => g (v n)

-- Theorem to prove
theorem v_2010_equals_0 : v 2010 = 0 := by
  sorry

end NUMINAMATH_CALUDE_v_2010_equals_0_l2041_204146


namespace NUMINAMATH_CALUDE_chalkboard_area_l2041_204148

/-- The area of a rectangle with width 3.5 feet and length 2.3 times its width is 28.175 square feet. -/
theorem chalkboard_area : 
  let width : ℝ := 3.5
  let length : ℝ := 2.3 * width
  width * length = 28.175 := by sorry

end NUMINAMATH_CALUDE_chalkboard_area_l2041_204148


namespace NUMINAMATH_CALUDE_apollonian_circle_m_range_l2041_204113

theorem apollonian_circle_m_range :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (2, 0)
  let C (m : ℝ) := {P : ℝ × ℝ | (P.1 - 2)^2 + (P.2 - m)^2 = 1/4}
  ∀ m > 0, (∃ P ∈ C m, dist P A = 2 * dist P B) →
    m ∈ Set.Icc (Real.sqrt 5 / 2) (Real.sqrt 21 / 2) :=
by sorry


end NUMINAMATH_CALUDE_apollonian_circle_m_range_l2041_204113


namespace NUMINAMATH_CALUDE_tenths_place_of_five_twelfths_l2041_204175

theorem tenths_place_of_five_twelfths (ε : ℚ) : 
  ε = 5 / 12 → 
  ∃ (n : ℕ) (r : ℚ), ε = (4 : ℚ) / 10 + n / 100 + r ∧ 0 ≤ r ∧ r < 1 / 100 :=
sorry

end NUMINAMATH_CALUDE_tenths_place_of_five_twelfths_l2041_204175


namespace NUMINAMATH_CALUDE_baseball_team_selection_l2041_204117

theorem baseball_team_selection (total_players : Nat) (selected_players : Nat) (twins : Nat) :
  total_players = 16 →
  selected_players = 9 →
  twins = 2 →
  Nat.choose (total_players - twins) (selected_players - twins) = 3432 := by
sorry

end NUMINAMATH_CALUDE_baseball_team_selection_l2041_204117


namespace NUMINAMATH_CALUDE_western_olympiad_2004_l2041_204119

theorem western_olympiad_2004 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_western_olympiad_2004_l2041_204119


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l2041_204189

/-- Given a rectangle ABCD and a square EFGH, if 30% of the rectangle's area
    overlaps with the square, and 25% of the square's area overlaps with the rectangle,
    then the ratio of the rectangle's length to its width is 7.5. -/
theorem rectangle_square_overlap_ratio :
  ∀ (l w s : ℝ),
    l > 0 → w > 0 → s > 0 →
    (0.3 * l * w = 0.25 * s^2) →
    (l / w = 7.5) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l2041_204189


namespace NUMINAMATH_CALUDE_compound_ratio_example_l2041_204130

-- Define the compound ratio function
def compound_ratio (a b c d e f g h : ℚ) : ℚ := (a * c * e * g) / (b * d * f * h)

-- State the theorem
theorem compound_ratio_example : compound_ratio 2 3 6 7 1 3 3 8 = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_compound_ratio_example_l2041_204130


namespace NUMINAMATH_CALUDE_tablet_cash_savings_l2041_204185

/-- Represents the cost and payment structure for a tablet purchase -/
structure TabletPurchase where
  cash_price : ℕ
  down_payment : ℕ
  first_four_months_payment : ℕ
  middle_four_months_payment : ℕ
  last_four_months_payment : ℕ

/-- Calculates the total amount paid through installments -/
def total_installment_cost (tp : TabletPurchase) : ℕ :=
  tp.down_payment + 4 * tp.first_four_months_payment + 4 * tp.middle_four_months_payment + 4 * tp.last_four_months_payment

/-- Calculates the savings when buying in cash versus installments -/
def cash_savings (tp : TabletPurchase) : ℕ :=
  total_installment_cost tp - tp.cash_price

/-- Theorem stating the savings when buying the tablet in cash -/
theorem tablet_cash_savings :
  ∃ (tp : TabletPurchase),
    tp.cash_price = 450 ∧
    tp.down_payment = 100 ∧
    tp.first_four_months_payment = 40 ∧
    tp.middle_four_months_payment = 35 ∧
    tp.last_four_months_payment = 30 ∧
    cash_savings tp = 70 := by
  sorry

end NUMINAMATH_CALUDE_tablet_cash_savings_l2041_204185


namespace NUMINAMATH_CALUDE_ellipse_circle_intersection_l2041_204126

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  25 * x^2 + 36 * y^2 = 900

noncomputable def point_M : ℝ × ℝ := (4.8, Real.sqrt (900 / 36 - 25 * 4.8^2 / 36))

noncomputable def tangent_line (x y : ℝ) : Prop :=
  25 * 4.8 * x + 36 * point_M.2 * y = 900

noncomputable def point_N : ℝ × ℝ := (0, 900 / (36 * point_M.2))

noncomputable def circle_equation (x y : ℝ) : Prop :=
  x^2 + (y - (263 / 75))^2 = (362 / 75)^2

theorem ellipse_circle_intersection :
  ∀ x y : ℝ,
  ellipse_equation x y ∧ circle_equation x y ∧ y = 0 →
  x = Real.sqrt 11 ∨ x = -Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_intersection_l2041_204126


namespace NUMINAMATH_CALUDE_pairings_equal_twenty_l2041_204163

/-- The number of items in the first set -/
def set1_size : ℕ := 5

/-- The number of items in the second set -/
def set2_size : ℕ := 4

/-- The total number of possible pairings -/
def total_pairings : ℕ := set1_size * set2_size

/-- Theorem: The total number of possible pairings is 20 -/
theorem pairings_equal_twenty : total_pairings = 20 := by
  sorry

end NUMINAMATH_CALUDE_pairings_equal_twenty_l2041_204163


namespace NUMINAMATH_CALUDE_work_completion_time_l2041_204198

theorem work_completion_time (ajay_time vijay_time combined_time : ℝ) : 
  ajay_time = 8 →
  combined_time = 6 →
  1 / ajay_time + 1 / vijay_time = 1 / combined_time →
  vijay_time = 24 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2041_204198


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2041_204145

/-- Given a train of length 100 meters traveling at 45 km/hr that crosses a bridge in 30 seconds, 
    the length of the bridge is 275 meters. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 275 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2041_204145


namespace NUMINAMATH_CALUDE_three_digit_square_proof_l2041_204172

theorem three_digit_square_proof : 
  ∃! (S : Finset Nat), 
    (∀ n ∈ S, 100 ≤ n ∧ n < 1000) ∧ 
    (∀ n ∈ S, ∃ k : Nat, 1000 * n = n^2 + k ∧ k < 1000) ∧
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_square_proof_l2041_204172


namespace NUMINAMATH_CALUDE_two_part_journey_average_speed_l2041_204114

/-- Calculates the average speed of a two-part journey -/
theorem two_part_journey_average_speed 
  (t1 : ℝ) (v1 : ℝ) (t2 : ℝ) (v2 : ℝ) 
  (h1 : t1 = 5) 
  (h2 : v1 = 40) 
  (h3 : t2 = 3) 
  (h4 : v2 = 80) : 
  (t1 * v1 + t2 * v2) / (t1 + t2) = 55 := by
  sorry

#check two_part_journey_average_speed

end NUMINAMATH_CALUDE_two_part_journey_average_speed_l2041_204114


namespace NUMINAMATH_CALUDE_least_possible_xy_l2041_204100

theorem least_possible_xy (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) : 
  ∃ (min_xy : ℕ), (x * y : ℕ) ≥ min_xy ∧ 
  (∃ (x' y' : ℕ+), (1 : ℚ) / x' + (1 : ℚ) / (3 * y') = (1 : ℚ) / 8 ∧ (x' * y' : ℕ) = min_xy) :=
sorry

end NUMINAMATH_CALUDE_least_possible_xy_l2041_204100


namespace NUMINAMATH_CALUDE_k_greater_than_one_over_e_l2041_204127

/-- Given that k(e^(kx)+1)-(1+1/x)ln(x) > 0 for all x > 0, prove that k > 1/e -/
theorem k_greater_than_one_over_e (k : ℝ) 
  (h : ∀ x : ℝ, x > 0 → k * (Real.exp (k * x) + 1) - (1 + 1 / x) * Real.log x > 0) : 
  k > 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_k_greater_than_one_over_e_l2041_204127


namespace NUMINAMATH_CALUDE_sum_of_divisors_360_l2041_204197

/-- The sum of the positive whole number divisors of 360 is 1170. -/
theorem sum_of_divisors_360 : (Finset.filter (· ∣ 360) (Finset.range 361)).sum id = 1170 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_360_l2041_204197


namespace NUMINAMATH_CALUDE_intersection_point_y_axis_l2041_204158

/-- The intersection point of a line with the y-axis -/
def y_axis_intersection (m a : ℝ) : ℝ × ℝ := (0, a)

/-- The line equation y = mx + b -/
def line_equation (m b : ℝ) (x : ℝ) : ℝ := m * x + b

theorem intersection_point_y_axis :
  let m : ℝ := 2
  let b : ℝ := 2
  y_axis_intersection m b = (0, line_equation m b 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_y_axis_l2041_204158


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l2041_204134

theorem orthogonal_vectors (y : ℝ) : 
  (2 * -3 + -4 * y + 5 * 2 = 0) ↔ (y = 1) :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l2041_204134


namespace NUMINAMATH_CALUDE_y_derivative_l2041_204178

noncomputable def y (x : ℝ) : ℝ :=
  (Real.cos x) / (3 * (2 + Real.sin x)) + (4 / (3 * Real.sqrt 3)) * Real.arctan ((2 * Real.tan (x / 2) + 1) / Real.sqrt 3)

theorem y_derivative (x : ℝ) :
  deriv y x = (2 * Real.sin x + 7) / (3 * (2 + Real.sin x)^2) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l2041_204178


namespace NUMINAMATH_CALUDE_remaining_time_is_three_and_half_l2041_204136

/-- The time taken for Cameron and Sandra to complete the remaining task -/
def remaining_time (cameron_rate : ℚ) (combined_rate : ℚ) (cameron_solo_days : ℚ) : ℚ :=
  (1 - cameron_rate * cameron_solo_days) / combined_rate

/-- Theorem stating the remaining time is 3.5 days -/
theorem remaining_time_is_three_and_half :
  remaining_time (1/18) (1/7) 9 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_remaining_time_is_three_and_half_l2041_204136


namespace NUMINAMATH_CALUDE_quadratic_one_root_from_geometric_sequence_l2041_204107

/-- If a, b, c form a geometric sequence of real numbers, then ax^2 + bx + c has exactly one real root -/
theorem quadratic_one_root_from_geometric_sequence (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) → 
  ∃! x : ℝ, a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_from_geometric_sequence_l2041_204107


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l2041_204115

/-- Rectangle represented by its side lengths -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Square represented by its side length -/
structure Square where
  side : ℝ

/-- The overlapping area between a rectangle and a square -/
def overlap_area (r : Rectangle) (s : Square) : ℝ := sorry

theorem rectangle_square_overlap_ratio :
  ∀ (r : Rectangle) (s : Square),
    overlap_area r s = 0.4 * r.length * r.width →
    overlap_area r s = 0.25 * s.side * s.side →
    r.length / r.width = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l2041_204115


namespace NUMINAMATH_CALUDE_quadratic_always_positive_a_squared_plus_a_zero_not_equivalent_a_zero_a_plus_b_greater_than_two_ab_greater_than_one_not_equivalent_a_greater_than_four_iff_positive_roots_l2041_204149

-- Proposition A
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - x + 1 > 0 := by sorry

-- Proposition B
theorem a_squared_plus_a_zero_not_equivalent_a_zero : ∃ a : ℝ, a^2 + a = 0 ∧ a ≠ 0 := by sorry

-- Proposition C
theorem a_plus_b_greater_than_two_ab_greater_than_one_not_equivalent :
  ∃ a b : ℝ, a + b > 2 ∧ a * b > 1 ∧ ¬(a > 1 ∧ b > 1) := by sorry

-- Proposition D
theorem a_greater_than_four_iff_positive_roots (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a = 0 → x > 0) ↔ a > 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_a_squared_plus_a_zero_not_equivalent_a_zero_a_plus_b_greater_than_two_ab_greater_than_one_not_equivalent_a_greater_than_four_iff_positive_roots_l2041_204149


namespace NUMINAMATH_CALUDE_max_value_expression_l2041_204164

theorem max_value_expression (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  (∃ x y z w, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1 ∧ 0 ≤ w ∧ w ≤ 1 ∧ 
    x + y + z + w - x*y - y*z - z*w - w*x = 2) ∧ 
  (∀ a b c d, 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 → 
    a + b + c + d - a*b - b*c - c*d - d*a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2041_204164


namespace NUMINAMATH_CALUDE_modular_congruence_l2041_204142

theorem modular_congruence (n : ℕ) : 
  0 ≤ n ∧ n < 37 ∧ (5 * n) % 37 = 1 → 
  (((2^n)^3 - 2) % 37 = 1) := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_l2041_204142


namespace NUMINAMATH_CALUDE_calculate_expression_solve_inequalities_l2041_204105

-- Part 1
theorem calculate_expression : 
  |3 - Real.pi| - (-2)⁻¹ + 4 * Real.cos (60 * π / 180) = Real.pi - 1/2 := by sorry

-- Part 2
theorem solve_inequalities (x : ℝ) : 
  (5*x - 1 > 3*(x + 1) ∧ 1 + 2*x ≥ x - 1) ↔ x > 2 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_inequalities_l2041_204105


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_43_l2041_204102

theorem least_positive_integer_multiple_of_43 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬((3*y)^2 + 3*29*3*y + 29^2) % 43 = 0) ∧
  ((3*x)^2 + 3*29*3*x + 29^2) % 43 = 0 ∧
  x = 19 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_43_l2041_204102


namespace NUMINAMATH_CALUDE_three_brothers_selection_probability_l2041_204122

theorem three_brothers_selection_probability 
  (p_ram : ℚ) (p_ravi : ℚ) (p_raj : ℚ) 
  (h_ram : p_ram = 2/7)
  (h_ravi : p_ravi = 1/5)
  (h_raj : p_raj = 3/8) :
  p_ram * p_ravi * p_raj = 3/140 := by
sorry

end NUMINAMATH_CALUDE_three_brothers_selection_probability_l2041_204122


namespace NUMINAMATH_CALUDE_parallel_tangents_imply_m_value_monotonicity_intervals_l2041_204147

noncomputable section

variable (m : ℝ)

def f (x : ℝ) : ℝ := (1/2) * m * x^2 + 1

def g (x : ℝ) : ℝ := 2 * Real.log x - (2*m + 1) * x - 1

def h (x : ℝ) : ℝ := f m x + g m x

def h_derivative (x : ℝ) : ℝ := m * x - (2*m + 1) + 2 / x

theorem parallel_tangents_imply_m_value :
  (h_derivative m 1 = h_derivative m 3) → m = 2/3 := by sorry

theorem monotonicity_intervals (x : ℝ) (hx : x > 0) :
  (m ≤ 0 → 
    (x < 2 → h_derivative m x > 0) ∧ 
    (x > 2 → h_derivative m x < 0)) ∧
  (0 < m ∧ m < 1/2 → 
    ((x < 2 ∨ x > 1/m) → h_derivative m x > 0) ∧ 
    (2 < x ∧ x < 1/m → h_derivative m x < 0)) ∧
  (m = 1/2 → h_derivative m x > 0) ∧
  (m > 1/2 → 
    ((x < 1/m ∨ x > 2) → h_derivative m x > 0) ∧ 
    (1/m < x ∧ x < 2 → h_derivative m x < 0)) := by sorry

end

end NUMINAMATH_CALUDE_parallel_tangents_imply_m_value_monotonicity_intervals_l2041_204147


namespace NUMINAMATH_CALUDE_shadow_problem_l2041_204118

/-- Given a cube with edge length 2 cm and a light source y cm above an upper vertex
    casting a shadow with area 175 sq cm (excluding the area beneath the cube),
    prove that the greatest integer less than or equal to 100y is 333. -/
theorem shadow_problem (y : ℝ) : 
  (2 : ℝ) > 0 ∧ 
  y > 0 ∧ 
  175 = (Real.sqrt 179 - 2)^2 →
  ⌊100 * y⌋ = 333 := by
  sorry

end NUMINAMATH_CALUDE_shadow_problem_l2041_204118


namespace NUMINAMATH_CALUDE_largest_hexagon_angle_l2041_204182

/-- Represents the angles of a hexagon -/
structure HexagonAngles where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  a₄ : ℝ
  a₅ : ℝ
  a₆ : ℝ

/-- The sum of angles in a hexagon is 720 degrees -/
axiom hexagon_angle_sum (h : HexagonAngles) : h.a₁ + h.a₂ + h.a₃ + h.a₄ + h.a₅ + h.a₆ = 720

/-- The angles of the hexagon are in the ratio 3:3:3:4:5:6 -/
def hexagon_angle_ratio (h : HexagonAngles) : Prop :=
  ∃ x : ℝ, h.a₁ = 3*x ∧ h.a₂ = 3*x ∧ h.a₃ = 3*x ∧ h.a₄ = 4*x ∧ h.a₅ = 5*x ∧ h.a₆ = 6*x

/-- The largest angle in the hexagon is 180 degrees -/
theorem largest_hexagon_angle (h : HexagonAngles) 
  (ratio : hexagon_angle_ratio h) : h.a₆ = 180 := by
  sorry

end NUMINAMATH_CALUDE_largest_hexagon_angle_l2041_204182


namespace NUMINAMATH_CALUDE_total_snow_volume_l2041_204168

/-- Calculates the total volume of snow on two sidewalk sections -/
theorem total_snow_volume 
  (length1 width1 depth1 : ℝ)
  (length2 width2 depth2 : ℝ)
  (h1 : length1 = 30)
  (h2 : width1 = 3)
  (h3 : depth1 = 1)
  (h4 : length2 = 15)
  (h5 : width2 = 2)
  (h6 : depth2 = 1/2) :
  length1 * width1 * depth1 + length2 * width2 * depth2 = 105 := by
sorry

end NUMINAMATH_CALUDE_total_snow_volume_l2041_204168


namespace NUMINAMATH_CALUDE_not_prime_special_polynomial_l2041_204177

theorem not_prime_special_polynomial (n : ℕ+) : 
  ¬ Nat.Prime (n.val^2 - 2^2014 * 2014 * n.val + 4^2013 * (2014^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_special_polynomial_l2041_204177


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_condition_l2041_204186

/-- Theorem: For an infinite geometric sequence with first term a₁ and common ratio q,
    if the sum of the sequence is 1/2, then 2a₁ + q = 1. -/
theorem geometric_sequence_sum_condition (a₁ q : ℝ) (h : |q| < 1) :
  (∑' n, a₁ * q ^ (n - 1) = 1/2) → 2 * a₁ + q = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_condition_l2041_204186


namespace NUMINAMATH_CALUDE_difference_of_squares_l2041_204191

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2041_204191
