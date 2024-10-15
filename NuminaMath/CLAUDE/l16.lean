import Mathlib

namespace NUMINAMATH_CALUDE_effective_price_for_8kg_l16_1615

/-- Represents the shopkeeper's pricing scheme -/
structure PricingScheme where
  false_weight : Real
  discount_rate : Real
  tax_rate : Real

/-- Calculates the effective price for a given purchase -/
def effective_price (scheme : PricingScheme) (purchase_weight : Real) (cost_price : Real) : Real :=
  let actual_weight := purchase_weight * (scheme.false_weight / 1000)
  let discounted_price := purchase_weight * cost_price * (1 - scheme.discount_rate)
  discounted_price * (1 + scheme.tax_rate)

/-- Theorem stating the effective price for the given scenario -/
theorem effective_price_for_8kg (scheme : PricingScheme) (cost_price : Real) :
  scheme.false_weight = 980 →
  scheme.discount_rate = 0.1 →
  scheme.tax_rate = 0.03 →
  effective_price scheme 8 cost_price = 7.416 * cost_price :=
by sorry

end NUMINAMATH_CALUDE_effective_price_for_8kg_l16_1615


namespace NUMINAMATH_CALUDE_only_two_works_l16_1637

/-- A move that can be applied to a table --/
inductive Move
  | MultiplyRow (row : Nat) : Move
  | SubtractColumn (col : Nat) : Move

/-- Definition of a rectangular table with positive integer entries --/
def Table := List (List Nat)

/-- Apply a move to a table --/
def applyMove (n : Nat) (t : Table) (m : Move) : Table :=
  sorry

/-- Check if all entries in a table are zero --/
def allZero (t : Table) : Prop :=
  sorry

/-- The main theorem --/
theorem only_two_works (n : Nat) : 
  (n > 0) → 
  (∀ t : Table, ∃ moves : List Move, allZero (moves.foldl (applyMove n) t)) ↔ 
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_only_two_works_l16_1637


namespace NUMINAMATH_CALUDE_fraction_division_equals_three_l16_1653

theorem fraction_division_equals_three : 
  (-1/6 + 3/8 - 1/12) / (1/24) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equals_three_l16_1653


namespace NUMINAMATH_CALUDE_min_colors_theorem_l16_1664

/-- The size of the board --/
def boardSize : Nat := 2016

/-- A color assignment for the board --/
def ColorAssignment := Fin boardSize → Fin boardSize → Nat

/-- Checks if a color assignment satisfies the diagonal condition --/
def satisfiesDiagonalCondition (c : ColorAssignment) : Prop :=
  ∀ i, c i i = 1

/-- Checks if a color assignment satisfies the symmetry condition --/
def satisfiesSymmetryCondition (c : ColorAssignment) : Prop :=
  ∀ i j, c i j = c j i

/-- Checks if a color assignment satisfies the row condition --/
def satisfiesRowCondition (c : ColorAssignment) : Prop :=
  ∀ i j k, i ≠ j ∧ (i < j ∧ j < k ∨ k < j ∧ j < i) → c i k ≠ c j k

/-- Checks if a color assignment is valid --/
def isValidColorAssignment (c : ColorAssignment) : Prop :=
  satisfiesDiagonalCondition c ∧ satisfiesSymmetryCondition c ∧ satisfiesRowCondition c

/-- The minimum number of colors required --/
def minColors : Nat := 11

/-- Theorem stating the minimum number of colors required --/
theorem min_colors_theorem :
  (∃ (c : ColorAssignment), isValidColorAssignment c ∧ (∀ i j, c i j < minColors)) ∧
  (∀ k < minColors, ¬∃ (c : ColorAssignment), isValidColorAssignment c ∧ (∀ i j, c i j < k)) :=
sorry

end NUMINAMATH_CALUDE_min_colors_theorem_l16_1664


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l16_1672

theorem arithmetic_calculations :
  (8 / (8 / 17) = 17) ∧
  ((6 / 11) / 3 = 2 / 11) ∧
  ((5 / 4) * (1 / 5) = 1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l16_1672


namespace NUMINAMATH_CALUDE_circle_equation_through_origin_l16_1669

/-- The equation of a circle with center (1, 1) passing through the origin (0, 0) is (x-1)^2 + (y-1)^2 = 2 -/
theorem circle_equation_through_origin (x y : ℝ) :
  let center : ℝ × ℝ := (1, 1)
  let origin : ℝ × ℝ := (0, 0)
  let on_circle (p : ℝ × ℝ) := (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1 - origin.1)^2 + (center.2 - origin.2)^2
  on_circle (x, y) ↔ (x - 1)^2 + (y - 1)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_through_origin_l16_1669


namespace NUMINAMATH_CALUDE_deposit_calculation_l16_1678

theorem deposit_calculation (remaining_amount : ℝ) (deposit_percentage : ℝ) : 
  remaining_amount = 1260 ∧ deposit_percentage = 0.1 → 
  (remaining_amount / (1 - deposit_percentage)) * deposit_percentage = 140 := by
sorry

end NUMINAMATH_CALUDE_deposit_calculation_l16_1678


namespace NUMINAMATH_CALUDE_no_quadruple_sum_2013_divisors_l16_1618

theorem no_quadruple_sum_2013_divisors :
  ¬ (∃ (a b c d : ℕ+), 
      (a.val + b.val + c.val + d.val = 2013) ∧ 
      (2013 % a.val = 0) ∧ 
      (2013 % b.val = 0) ∧ 
      (2013 % c.val = 0) ∧ 
      (2013 % d.val = 0)) := by
  sorry

end NUMINAMATH_CALUDE_no_quadruple_sum_2013_divisors_l16_1618


namespace NUMINAMATH_CALUDE_number_of_parents_l16_1613

theorem number_of_parents (girls : ℕ) (boys : ℕ) (playgroups : ℕ) (group_size : ℕ) : 
  girls = 14 → 
  boys = 11 → 
  playgroups = 3 → 
  group_size = 25 → 
  playgroups * group_size - (girls + boys) = 50 := by
sorry

end NUMINAMATH_CALUDE_number_of_parents_l16_1613


namespace NUMINAMATH_CALUDE_players_count_l16_1668

/-- Represents the number of socks in each washing machine -/
structure SockCounts where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the number of players based on sock counts -/
def calculate_players (socks : SockCounts) : ℕ :=
  min socks.red (socks.blue + socks.green)

/-- Theorem stating that the number of players is 12 given the specific sock counts -/
theorem players_count (socks : SockCounts)
  (h_red : socks.red = 12)
  (h_blue : socks.blue = 10)
  (h_green : socks.green = 16) :
  calculate_players socks = 12 := by
  sorry

#eval calculate_players ⟨12, 10, 16⟩

end NUMINAMATH_CALUDE_players_count_l16_1668


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l16_1683

theorem square_plus_inverse_square (x : ℝ) (h : x + (1/x) = 2) : x^2 + (1/x^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l16_1683


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l16_1629

theorem min_four_dollar_frisbees (total_frisbees : ℕ) (total_receipts : ℕ) : 
  total_frisbees = 64 →
  total_receipts = 200 →
  ∃ (three_dollar : ℕ) (four_dollar : ℕ),
    three_dollar + four_dollar = total_frisbees ∧
    3 * three_dollar + 4 * four_dollar = total_receipts ∧
    ∀ (other_four_dollar : ℕ),
      other_four_dollar + (total_frisbees - other_four_dollar) = total_frisbees ∧
      3 * (total_frisbees - other_four_dollar) + 4 * other_four_dollar = total_receipts →
      four_dollar ≤ other_four_dollar ∧
      four_dollar = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l16_1629


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_perpendicular_l16_1605

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_perpendicular 
  (m n : Line) (α : Plane) :
  parallel m n → perpendicular n α → perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_perpendicular_l16_1605


namespace NUMINAMATH_CALUDE_no_solution_in_naturals_l16_1604

theorem no_solution_in_naturals :
  ¬ ∃ (x y z : ℕ), (2 * x) ^ (2 * x) - 1 = y ^ (z + 1) := by
sorry

end NUMINAMATH_CALUDE_no_solution_in_naturals_l16_1604


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l16_1697

theorem complex_purely_imaginary (m : ℝ) : 
  let z : ℂ := m + 2*I
  (∃ (y : ℝ), (2 + I) * z = y * I) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l16_1697


namespace NUMINAMATH_CALUDE_systematic_sampling_characterization_l16_1609

/-- Represents a population in a sampling context -/
structure Population where
  size : ℕ
  is_large : Prop

/-- Represents a sampling method -/
structure SamplingMethod where
  divides_population : Prop
  uses_predetermined_rule : Prop
  selects_one_per_part : Prop

/-- Definition of systematic sampling -/
def systematic_sampling (pop : Population) (method : SamplingMethod) : Prop :=
  pop.is_large ∧ 
  method.divides_population ∧ 
  method.uses_predetermined_rule ∧ 
  method.selects_one_per_part

/-- Theorem stating the characterization of systematic sampling -/
theorem systematic_sampling_characterization 
  (pop : Population) 
  (method : SamplingMethod) : 
  systematic_sampling pop method ↔ 
    (method.divides_population ∧ 
     method.uses_predetermined_rule ∧ 
     method.selects_one_per_part) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_characterization_l16_1609


namespace NUMINAMATH_CALUDE_equation_solution_l16_1667

theorem equation_solution (n : ℚ) :
  (2 / (n + 2) + 4 / (n + 2) + n / (n + 2) = 4) → n = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l16_1667


namespace NUMINAMATH_CALUDE_divisibility_property_l16_1600

theorem divisibility_property :
  ∀ k : ℤ, k ≠ 2013 → (2013 - k) ∣ (2013^2014 - k^2014) := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l16_1600


namespace NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l16_1602

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l16_1602


namespace NUMINAMATH_CALUDE_horner_method_V_4_l16_1675

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial coefficients in descending order of degree -/
def f_coeffs : List ℤ := [3, 5, 6, 79, -8, 35, 12]

/-- The x-value at which to evaluate the polynomial -/
def x_val : ℤ := -4

/-- V_4 in Horner's method is the 5th intermediate value (0-indexed) -/
def V_4 : ℤ := (horner_eval (f_coeffs.take 5) x_val) * x_val + f_coeffs[5]

theorem horner_method_V_4 :
  V_4 = 220 :=
sorry

end NUMINAMATH_CALUDE_horner_method_V_4_l16_1675


namespace NUMINAMATH_CALUDE_orange_juice_percentage_l16_1627

/-- Represents the composition and pricing of a drink made from milk and orange juice -/
structure DrinkComposition where
  milk_mass : ℝ
  juice_mass : ℝ
  initial_milk_price : ℝ
  initial_juice_price : ℝ
  milk_price_change : ℝ
  juice_price_change : ℝ

/-- The theorem stating the mass percentage of orange juice in the drink -/
theorem orange_juice_percentage (drink : DrinkComposition) 
  (h_price_ratio : drink.initial_juice_price = 6 * drink.initial_milk_price)
  (h_milk_change : drink.milk_price_change = -0.15)
  (h_juice_change : drink.juice_price_change = 0.1)
  (h_cost_unchanged : 
    drink.milk_mass * drink.initial_milk_price * (1 + drink.milk_price_change) + 
    drink.juice_mass * drink.initial_juice_price * (1 + drink.juice_price_change) = 
    drink.milk_mass * drink.initial_milk_price + 
    drink.juice_mass * drink.initial_juice_price) :
  drink.juice_mass / (drink.milk_mass + drink.juice_mass) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_percentage_l16_1627


namespace NUMINAMATH_CALUDE_fraction_simplification_l16_1699

theorem fraction_simplification : (27 : ℚ) / 25 * 20 / 33 * 55 / 54 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l16_1699


namespace NUMINAMATH_CALUDE_juhyes_money_l16_1692

theorem juhyes_money (initial_money : ℝ) : 
  (1/3 : ℝ) * (3/4 : ℝ) * initial_money = 2500 → initial_money = 10000 := by
  sorry

end NUMINAMATH_CALUDE_juhyes_money_l16_1692


namespace NUMINAMATH_CALUDE_equal_roots_count_l16_1677

/-- The number of real values of p for which the quadratic equation
    x^2 - (p+1)x + (p+1)^2 = 0 has equal roots is exactly one. -/
theorem equal_roots_count : ∃! p : ℝ,
  let a : ℝ := 1
  let b : ℝ := -(p + 1)
  let c : ℝ := (p + 1)^2
  b^2 - 4*a*c = 0 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_count_l16_1677


namespace NUMINAMATH_CALUDE_unique_prime_pair_squares_l16_1662

theorem unique_prime_pair_squares : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ 
  ∃ (a b : ℕ), (p - q = a^2) ∧ (p*q - q = b^2) := by
sorry

end NUMINAMATH_CALUDE_unique_prime_pair_squares_l16_1662


namespace NUMINAMATH_CALUDE_room_width_calculation_l16_1670

/-- Given a rectangular room with length 5.5 m, if the cost of paving its floor
    at a rate of 1000 per sq. meter is 20625, then the width of the room is 3.75 m. -/
theorem room_width_calculation (length cost rate : ℝ) (h1 : length = 5.5)
    (h2 : cost = 20625) (h3 : rate = 1000) : 
    cost / rate / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l16_1670


namespace NUMINAMATH_CALUDE_marston_county_population_l16_1686

theorem marston_county_population (num_cities : ℕ) (lower_bound upper_bound : ℝ) :
  num_cities = 25 →
  lower_bound = 4800 →
  upper_bound = 5200 →
  (num_cities : ℝ) * ((lower_bound + upper_bound) / 2) = 125000 := by
  sorry

end NUMINAMATH_CALUDE_marston_county_population_l16_1686


namespace NUMINAMATH_CALUDE_inverse_odd_implies_a_eq_one_l16_1634

/-- A function f: ℝ → ℝ -/
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x ↦ 2^x - a * 2^(-x)

/-- The inverse function of f -/
noncomputable def f_inv (a : ℝ) : ℝ → ℝ := Function.invFun (f a)

/-- Theorem stating that if f_inv is odd and a is positive, then a = 1 -/
theorem inverse_odd_implies_a_eq_one (a : ℝ) (h_pos : a > 0) 
  (h_odd : ∀ x, f_inv a (-x) = -(f_inv a x)) : a = 1 := by
  sorry

#check inverse_odd_implies_a_eq_one

end NUMINAMATH_CALUDE_inverse_odd_implies_a_eq_one_l16_1634


namespace NUMINAMATH_CALUDE_collinear_vectors_l16_1640

/-- Given vectors a, b, and c in ℝ², prove that if k*a + b is collinear with c,
    then k = -26/15 -/
theorem collinear_vectors (a b c : ℝ × ℝ) (k : ℝ) 
    (ha : a = (1, 2))
    (hb : b = (2, 3))
    (hc : c = (4, -7))
    (hcollinear : ∃ t : ℝ, t ≠ 0 ∧ k • a + b = t • c) :
    k = -26/15 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l16_1640


namespace NUMINAMATH_CALUDE_yellow_shirt_pairs_l16_1603

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 60 →
  yellow_students = 72 →
  total_students = 132 →
  total_pairs = 66 →
  blue_blue_pairs = 25 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 31 ∧ 
    yellow_yellow_pairs = total_pairs - blue_blue_pairs - (blue_students - 2 * blue_blue_pairs) :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_shirt_pairs_l16_1603


namespace NUMINAMATH_CALUDE_exactly_two_pairs_exist_l16_1625

/-- Two lines in the xy-plane -/
structure TwoLines where
  a : ℝ
  d : ℝ

/-- The condition for two lines to be identical -/
def are_identical (l : TwoLines) : Prop :=
  ∀ x y : ℝ, (4 * x + l.a * y + l.d = 0) ↔ (l.d * x - 3 * y + 15 = 0)

/-- The theorem stating that there are exactly two pairs (a, d) satisfying the condition -/
theorem exactly_two_pairs_exist :
  ∃! (s : Finset TwoLines), s.card = 2 ∧ (∀ l ∈ s, are_identical l) ∧
    (∀ l : TwoLines, are_identical l → l ∈ s) :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_pairs_exist_l16_1625


namespace NUMINAMATH_CALUDE_x_equals_n_l16_1695

def x : ℕ → ℚ
  | 0 => 0
  | n + 1 => ((n^2 + n + 1) * x n + 1) / (n^2 + n + 1 - x n)

theorem x_equals_n (n : ℕ) : x n = n := by
  sorry

end NUMINAMATH_CALUDE_x_equals_n_l16_1695


namespace NUMINAMATH_CALUDE_dianas_biking_speed_l16_1649

/-- Given Diana's biking scenario, prove her speed after getting tired -/
theorem dianas_biking_speed 
  (total_distance : ℝ) 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (total_time : ℝ) 
  (h1 : total_distance = 10)
  (h2 : initial_speed = 3)
  (h3 : initial_time = 2)
  (h4 : total_time = 6) :
  let initial_distance := initial_speed * initial_time
  let remaining_distance := total_distance - initial_distance
  let remaining_time := total_time - initial_time
  remaining_distance / remaining_time = 1 := by
sorry


end NUMINAMATH_CALUDE_dianas_biking_speed_l16_1649


namespace NUMINAMATH_CALUDE_correct_average_l16_1606

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℕ) :
  n = 10 →
  initial_avg = 16 →
  incorrect_num = 25 →
  correct_num = 45 →
  (n : ℚ) * initial_avg + (correct_num - incorrect_num : ℚ) = n * 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l16_1606


namespace NUMINAMATH_CALUDE_max_sum_is_24_l16_1660

/-- Represents the grid configuration -/
structure Grid :=
  (a b c d e : ℕ)

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {5, 8, 11, 14}

/-- Checks if the grid contains only the available numbers -/
def Grid.isValid (g : Grid) : Prop :=
  {g.a, g.b, g.c, g.d, g.e} ⊆ availableNumbers

/-- Calculates the horizontal sum -/
def Grid.horizontalSum (g : Grid) : ℕ := g.a + g.b + g.e

/-- Calculates the vertical sum -/
def Grid.verticalSum (g : Grid) : ℕ := g.a + g.c + 2 * g.e

/-- Checks if the grid satisfies the sum condition -/
def Grid.satisfiesSumCondition (g : Grid) : Prop :=
  g.horizontalSum = g.verticalSum

theorem max_sum_is_24 :
  ∃ (g : Grid), g.isValid ∧ g.satisfiesSumCondition ∧
  (∀ (h : Grid), h.isValid → h.satisfiesSumCondition →
    g.horizontalSum ≥ h.horizontalSum ∧ g.verticalSum ≥ h.verticalSum) ∧
  g.horizontalSum = 24 ∧ g.verticalSum = 24 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_24_l16_1660


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l16_1641

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x - 4 * y + 4 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (2, -3)

-- Define the equation of the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 4 * x + 3 * y + 1 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
    (∃ (m : ℝ), perpendicular_line x y ∧
      (∀ (x' y' : ℝ), given_line x' y' → (y - point.2 = m * (x - point.1))) ∧
      (m * (4 / 3) = -1)) →
    perpendicular_line x y :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l16_1641


namespace NUMINAMATH_CALUDE_not_perfect_square_with_mostly_fives_l16_1619

/-- A function that checks if a list of digits represents a number with all but at most one digit being 5 -/
def allButOneAre5 (digits : List Nat) : Prop :=
  digits.length = 1000 ∧ (digits.filter (· ≠ 5)).length ≤ 1

/-- The theorem stating that a number with 1000 digits, all but at most one being 5, is not a perfect square -/
theorem not_perfect_square_with_mostly_fives (digits : List Nat) (h : allButOneAre5 digits) :
    ¬∃ (n : Nat), n * n = digits.foldl (fun acc d => acc * 10 + d) 0 := by
  sorry


end NUMINAMATH_CALUDE_not_perfect_square_with_mostly_fives_l16_1619


namespace NUMINAMATH_CALUDE_certain_number_proof_l16_1643

theorem certain_number_proof (h : 213 * 16 = 3408) : 
  ∃ x : ℝ, 213 * x = 340.8 ∧ x = 1.6 := by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l16_1643


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l16_1617

theorem smallest_part_of_proportional_division (total : ℝ) (prop1 prop2 prop3 : ℝ) (additional : ℝ) :
  total = 120 ∧ prop1 = 3 ∧ prop2 = 5 ∧ prop3 = 7 ∧ additional = 4 →
  let x := (total - 3 * additional) / (prop1 + prop2 + prop3)
  let part1 := prop1 * x + additional
  let part2 := prop2 * x + additional
  let part3 := prop3 * x + additional
  min part1 (min part2 part3) = 25.6 := by
sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l16_1617


namespace NUMINAMATH_CALUDE_correct_mark_is_ten_l16_1682

/-- Proves that the correct mark of a student is 10, given the conditions of the problem. -/
theorem correct_mark_is_ten (n : ℕ) (initial_avg final_avg wrong_mark : ℚ) :
  n = 30 →
  initial_avg = 100 →
  wrong_mark = 70 →
  final_avg = 98 →
  (n : ℚ) * initial_avg - wrong_mark + (n : ℚ) * final_avg = (n : ℚ) * initial_avg →
  (n : ℚ) * initial_avg - wrong_mark + 10 = (n : ℚ) * final_avg :=
by sorry

end NUMINAMATH_CALUDE_correct_mark_is_ten_l16_1682


namespace NUMINAMATH_CALUDE_quadratic_sequence_l16_1654

/-- Given a quadratic equation with real roots and a specific condition, 
    prove the relation between consecutive terms and the geometric nature of a derived sequence. -/
theorem quadratic_sequence (n : ℕ+) (a : ℕ+ → ℝ) (α β : ℝ) 
  (h1 : a n * α^2 - a (n + 1) * α + 1 = 0)
  (h2 : a n * β^2 - a (n + 1) * β + 1 = 0)
  (h3 : 6 * α - 2 * α * β + 6 * β = 3) :
  (∀ m : ℕ+, a (m + 1) = 1/2 * a m + 1/3) ∧ 
  (∃ r : ℝ, ∀ m : ℕ+, a (m + 1) - 2/3 = r * (a m - 2/3)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_l16_1654


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l16_1680

/-- The area of the region between a circle circumscribing two externally tangent circles and those two circles -/
theorem shaded_area_between_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5) : 
  let R := r2 + (r1 + r2) / 2
  π * R^2 - π * r1^2 - π * r2^2 = 49.25 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l16_1680


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l16_1689

theorem min_value_of_fraction (x : ℝ) (h : x > 9) :
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l16_1689


namespace NUMINAMATH_CALUDE_miles_driven_proof_l16_1630

def miles_per_gallon : ℝ := 40
def cost_per_gallon : ℝ := 5
def budget : ℝ := 25

theorem miles_driven_proof : 
  (budget / cost_per_gallon) * miles_per_gallon = 200 := by sorry

end NUMINAMATH_CALUDE_miles_driven_proof_l16_1630


namespace NUMINAMATH_CALUDE_range_of_k_l16_1624

def is_sufficient_condition (k : ℝ) : Prop :=
  ∀ x, x > k → 3 / (x + 1) < 1

def is_not_necessary_condition (k : ℝ) : Prop :=
  ∃ x, 3 / (x + 1) < 1 ∧ x ≤ k

theorem range_of_k : 
  ∀ k, (is_sufficient_condition k ∧ is_not_necessary_condition k) ↔ k ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l16_1624


namespace NUMINAMATH_CALUDE_vector_collinearity_implies_ratio_l16_1644

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 5)

-- Define collinearity for 2D vectors
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- State the theorem
theorem vector_collinearity_implies_ratio (m n : ℝ) (h_n : n ≠ 0) :
  collinear ((m * a.1 - n * b.1, m * a.2 - n * b.2) : ℝ × ℝ) (a.1 + 2 * b.1, a.2 + 2 * b.2) →
  m / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_implies_ratio_l16_1644


namespace NUMINAMATH_CALUDE_binomial_arithmetic_sequence_l16_1622

theorem binomial_arithmetic_sequence (n k : ℕ) :
  (∃ (a : ℕ), Nat.choose n (k-1) + a = Nat.choose n k ∧ Nat.choose n k + a = Nat.choose n (k+1)) ↔
  (∃ (u : ℕ), u ≥ 3 ∧ n = u^2 - 2 ∧ (k = Nat.choose u 2 - 1 ∨ k = Nat.choose (u+1) 2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_arithmetic_sequence_l16_1622


namespace NUMINAMATH_CALUDE_remainder_mod_11_l16_1659

theorem remainder_mod_11 : (8735+100) + (8736+100) + (8737+100) + (8738+100) * 2 ≡ 10 [MOD 11] := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_11_l16_1659


namespace NUMINAMATH_CALUDE_gingerbreads_in_unknown_tray_is_20_l16_1688

/-- The number of gingerbreads in each of the first four trays -/
def gingerbreads_per_tray : ℕ := 25

/-- The number of trays with a known number of gingerbreads -/
def known_trays : ℕ := 4

/-- The number of trays with an unknown number of gingerbreads -/
def unknown_trays : ℕ := 3

/-- The total number of gingerbreads baked -/
def total_gingerbreads : ℕ := 160

/-- The number of gingerbreads in each of the unknown trays -/
def gingerbreads_in_unknown_tray : ℕ := (total_gingerbreads - known_trays * gingerbreads_per_tray) / unknown_trays

theorem gingerbreads_in_unknown_tray_is_20 :
  gingerbreads_in_unknown_tray = 20 := by
  sorry

end NUMINAMATH_CALUDE_gingerbreads_in_unknown_tray_is_20_l16_1688


namespace NUMINAMATH_CALUDE_sum_lent_is_500_l16_1657

/-- The sum of money lent -/
def P : ℝ := 500

/-- The annual interest rate as a decimal -/
def R : ℝ := 0.04

/-- The time period in years -/
def T : ℝ := 8

/-- The simple interest formula -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

theorem sum_lent_is_500 : 
  simple_interest P R T = P - 340 → P = 500 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_is_500_l16_1657


namespace NUMINAMATH_CALUDE_regular_tetrahedron_inscribed_sphere_ratio_l16_1656

/-- For a regular tetrahedron with height H and inscribed sphere radius R, 
    the ratio R:H is 1:4 -/
theorem regular_tetrahedron_inscribed_sphere_ratio 
  (H : ℝ) (R : ℝ) (h : H > 0) (r : R > 0) : R / H = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_inscribed_sphere_ratio_l16_1656


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_29_l16_1690

theorem modular_inverse_of_5_mod_29 :
  ∃ a : ℕ, a ≤ 28 ∧ (5 * a) % 29 = 1 ∧ a = 6 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_29_l16_1690


namespace NUMINAMATH_CALUDE_kayla_bought_fifteen_items_l16_1691

/-- Represents the number of chocolate bars bought by Theresa -/
def theresa_chocolate_bars : ℕ := 12

/-- Represents the number of soda cans bought by Theresa -/
def theresa_soda_cans : ℕ := 18

/-- Represents the ratio of items bought by Theresa compared to Kayla -/
def theresa_to_kayla_ratio : ℕ := 2

/-- Calculates the total number of items bought by Kayla -/
def kayla_total_items : ℕ := 
  (theresa_chocolate_bars / theresa_to_kayla_ratio) + 
  (theresa_soda_cans / theresa_to_kayla_ratio)

/-- Theorem stating that Kayla bought 15 items in total -/
theorem kayla_bought_fifteen_items : kayla_total_items = 15 := by
  sorry

end NUMINAMATH_CALUDE_kayla_bought_fifteen_items_l16_1691


namespace NUMINAMATH_CALUDE_bus_speed_with_stoppages_l16_1648

/-- Calculates the speed of a bus including stoppages -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages = 54 →
  stoppage_time = 10 →
  total_time = 60 →
  (speed_without_stoppages * (total_time - stoppage_time) / total_time) = 45 :=
by
  sorry

#check bus_speed_with_stoppages

end NUMINAMATH_CALUDE_bus_speed_with_stoppages_l16_1648


namespace NUMINAMATH_CALUDE_original_number_proof_l16_1623

theorem original_number_proof (x y : ℝ) : 
  x = 13.0 →
  7 * x + 5 * y = 146 →
  x + y = 24.0 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l16_1623


namespace NUMINAMATH_CALUDE_ellipse_problem_l16_1621

noncomputable section

def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def FocalLength (c : ℝ) := 2 * Real.sqrt 3

def Eccentricity (e : ℝ) := Real.sqrt 2 / 2

def RightFocus (F : ℝ × ℝ) := F.1 > 0 ∧ F.2 = 0

def VectorDot (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2

def LineIntersection (k : ℝ) (N : Set (ℝ × ℝ)) := 
  {p : ℝ × ℝ | p.2 = k * (p.1 - 2) ∧ p ∈ N}

def VectorLength (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)

theorem ellipse_problem (a b c : ℝ) (C : Set (ℝ × ℝ)) (F B : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  C = Ellipse a b ∧
  FocalLength c = 2 * Real.sqrt 3 ∧
  Eccentricity (c / a) = Real.sqrt 2 / 2 ∧
  RightFocus F ∧
  B = (0, b) →
  (∃ A ∈ C, VectorDot (A.1 - B.1, A.2 - B.2) (F.1 - B.1, F.2 - B.2) = -6 →
    (∃ O r, (∀ p, p ∈ {q | (q.1 - O.1)^2 + (q.2 - O.2)^2 = r^2} ↔ 
      (p = A ∨ p = B ∨ p = F)) ∧
      (O = (0, 0) ∧ r = Real.sqrt 3 ∨
       O = (2 * Real.sqrt 3 / 3, 2 * Real.sqrt 3 / 3) ∧ r = Real.sqrt 15 / 3))) ∧
  (∀ k G H, G ∈ LineIntersection k (Ellipse a b) ∧ 
            H ∈ LineIntersection k (Ellipse a b) ∧ 
            G ≠ H ∧
            VectorLength (H.1 - G.1, H.2 - G.2) < 2 * Real.sqrt 5 / 3 →
    (-Real.sqrt 2 / 2 < k ∧ k < -1/2) ∨ (1/2 < k ∧ k < Real.sqrt 2 / 2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_problem_l16_1621


namespace NUMINAMATH_CALUDE_erased_number_proof_l16_1628

theorem erased_number_proof (n : ℕ) (x : ℕ) :
  n > 0 →
  x > 0 →
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1) = 182 / 5 →
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l16_1628


namespace NUMINAMATH_CALUDE_gcd_102_238_l16_1635

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by sorry

end NUMINAMATH_CALUDE_gcd_102_238_l16_1635


namespace NUMINAMATH_CALUDE_grocery_store_soda_l16_1601

theorem grocery_store_soda (total : ℕ) (diet : ℕ) (regular : ℕ) : 
  total = 30 → diet = 2 → regular = total - diet → regular = 28 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_soda_l16_1601


namespace NUMINAMATH_CALUDE_tangent_line_at_one_f_greater_than_one_l16_1679

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.exp x * (Real.log x + 1 / x)

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (m : ℝ), ∀ (x y : ℝ), y = m * (x - 1) + f 1 ↔ Real.exp x - y = 0 :=
sorry

-- Theorem for the magnitude comparison
theorem f_greater_than_one :
  ∀ (x : ℝ), x > 0 → f x > 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_f_greater_than_one_l16_1679


namespace NUMINAMATH_CALUDE_charity_raffle_winnings_l16_1631

theorem charity_raffle_winnings (X : ℝ) : 
  let remaining_after_donation := 0.75 * X
  let remaining_after_lunch := remaining_after_donation * 0.9
  let remaining_after_gift := remaining_after_lunch * 0.85
  let amount_for_investment := remaining_after_gift * 0.3
  let investment_return := amount_for_investment * 0.5
  let final_amount := remaining_after_gift - amount_for_investment + investment_return
  final_amount = 320 → X = 485 :=
by sorry

end NUMINAMATH_CALUDE_charity_raffle_winnings_l16_1631


namespace NUMINAMATH_CALUDE_lilliputian_matchboxes_theorem_l16_1676

/-- The scale factor between Gulliver's homeland and Lilliput -/
def scale_factor : ℕ := 12

/-- The number of Lilliputian matchboxes that can fit into a matchbox from Gulliver's homeland -/
def lilliputian_matchboxes_count : ℕ := scale_factor ^ 3

/-- Theorem stating that the number of Lilliputian matchboxes that can fit into a matchbox from Gulliver's homeland is 1728 -/
theorem lilliputian_matchboxes_theorem : lilliputian_matchboxes_count = 1728 := by
  sorry

end NUMINAMATH_CALUDE_lilliputian_matchboxes_theorem_l16_1676


namespace NUMINAMATH_CALUDE_sherman_weekly_driving_time_l16_1694

/-- Calculates the total weekly driving time for Sherman given his commute and weekend driving schedules. -/
theorem sherman_weekly_driving_time 
  (weekday_commute_minutes : ℕ) -- Daily commute time (round trip) in minutes
  (weekdays : ℕ) -- Number of weekdays
  (weekend_driving_hours : ℕ) -- Daily weekend driving time in hours
  (weekend_days : ℕ) -- Number of weekend days
  (h1 : weekday_commute_minutes = 60) -- 30 minutes to office + 30 minutes back home
  (h2 : weekdays = 5) -- 5 weekdays in a week
  (h3 : weekend_driving_hours = 2) -- 2 hours of driving each weekend day
  (h4 : weekend_days = 2) -- 2 days in a weekend
  : (weekday_commute_minutes * weekdays) / 60 + weekend_driving_hours * weekend_days = 9 :=
by sorry

end NUMINAMATH_CALUDE_sherman_weekly_driving_time_l16_1694


namespace NUMINAMATH_CALUDE_metal_sheet_width_l16_1626

/-- Represents the dimensions and volume of a box created from a metal sheet -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ

/-- Calculates the original width of the metal sheet given the box dimensions -/
def calculate_original_width (box : BoxDimensions) : ℝ :=
  box.width + 2 * box.height

/-- Theorem stating that given the specified conditions, the original width of the sheet must be 36 m -/
theorem metal_sheet_width
  (box : BoxDimensions)
  (h1 : box.length = 48 - 2 * 4)
  (h2 : box.height = 4)
  (h3 : box.volume = 4480)
  (h4 : box.volume = box.length * box.width * box.height) :
  calculate_original_width box = 36 := by
  sorry

#check metal_sheet_width

end NUMINAMATH_CALUDE_metal_sheet_width_l16_1626


namespace NUMINAMATH_CALUDE_min_sum_squares_l16_1610

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  ∃ (m : ℝ), m = t^2 / 3 ∧ ∀ (x y z : ℝ), x + y + z = t → x^2 + y^2 + z^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l16_1610


namespace NUMINAMATH_CALUDE_geometry_problem_l16_1663

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the operations and relations
variable (intersection : Plane → Plane → Line)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem geometry_problem 
  (l m : Line) (α β γ : Plane)
  (h1 : intersection β γ = l)
  (h2 : parallel l α)
  (h3 : subset m α)
  (h4 : perpendicular m γ) :
  perpendicularPlanes α γ ∧ perpendicularLines l m := by
  sorry

end NUMINAMATH_CALUDE_geometry_problem_l16_1663


namespace NUMINAMATH_CALUDE_daniel_initial_noodles_l16_1646

/-- The number of noodles Daniel gave to William -/
def noodles_given : ℝ := 12.0

/-- The number of noodles Daniel had left -/
def noodles_left : ℕ := 42

/-- The initial number of noodles Daniel had -/
def initial_noodles : ℝ := noodles_given + noodles_left

theorem daniel_initial_noodles : initial_noodles = 54.0 := by sorry

end NUMINAMATH_CALUDE_daniel_initial_noodles_l16_1646


namespace NUMINAMATH_CALUDE_max_value_of_function_l16_1687

theorem max_value_of_function (x : ℝ) : 
  (∀ x, -1 ≤ Real.cos x ∧ Real.cos x ≤ 1) →
  (∃ M : ℝ, M = 3 ∧ ∀ x, (2 + Real.cos x) / (2 - Real.cos x) ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l16_1687


namespace NUMINAMATH_CALUDE_pause_point_correct_l16_1661

/-- Represents the duration of a movie in minutes -/
def MovieLength : ℕ := 60

/-- Represents the remaining time to watch in minutes -/
def RemainingTime : ℕ := 30

/-- Calculates the point at which the movie was paused -/
def PausePoint : ℕ := MovieLength - RemainingTime

theorem pause_point_correct : PausePoint = 30 := by
  sorry

end NUMINAMATH_CALUDE_pause_point_correct_l16_1661


namespace NUMINAMATH_CALUDE_unique_fraction_condition_l16_1684

def is_simplest_proper_fraction (n d : ℤ) : Prop :=
  0 < n ∧ n < d ∧ Nat.gcd n.natAbs d.natAbs = 1

def is_improper_fraction (n d : ℤ) : Prop :=
  n ≥ d

theorem unique_fraction_condition (x : ℤ) : 
  (is_simplest_proper_fraction x 8 ∧ is_improper_fraction x 6) ↔ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_fraction_condition_l16_1684


namespace NUMINAMATH_CALUDE_length_a_prime_b_prime_l16_1696

/-- Given points A, B, C, and the line y = x, prove that the length of A'B' is 4√2 -/
theorem length_a_prime_b_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 6) →
  B = (0, 10) →
  C = (3, 7) →
  (A'.1 = A'.2 ∧ B'.1 = B'.2) →  -- A' and B' are on the line y = x
  (∃ t : ℝ, A + t • (A' - A) = C) →  -- AA' passes through C
  (∃ s : ℝ, B + s • (B' - B) = C) →  -- BB' passes through C
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_length_a_prime_b_prime_l16_1696


namespace NUMINAMATH_CALUDE_no_linear_factor_l16_1655

/-- The polynomial p(x, y, z) = x^2 - y^2 + z^2 - 2yz + 2x - 3y + z -/
def p (x y z : ℤ) : ℤ := x^2 - y^2 + z^2 - 2*y*z + 2*x - 3*y + z

/-- Theorem stating that p(x, y, z) cannot be factored with a linear integer factor -/
theorem no_linear_factor :
  ¬ ∃ (a b c d : ℤ) (q : ℤ → ℤ → ℤ → ℤ),
    ∀ x y z, p x y z = (a*x + b*y + c*z + d) * q x y z :=
by sorry

end NUMINAMATH_CALUDE_no_linear_factor_l16_1655


namespace NUMINAMATH_CALUDE_fraction_equality_l16_1658

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3/4) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l16_1658


namespace NUMINAMATH_CALUDE_expression_evaluation_l16_1685

theorem expression_evaluation :
  let x : ℝ := -5
  let y : ℝ := 8
  let z : ℝ := 3
  let w : ℝ := 2
  Real.sqrt (2 * z * (w - y)^2 - x^3 * y) + Real.sin (Real.pi * z) * x * w^2 - Real.tan (Real.pi * x^2) * z^3 = Real.sqrt 1216 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l16_1685


namespace NUMINAMATH_CALUDE_equal_sets_implies_b_minus_a_equals_one_l16_1652

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, 0, a}
def B (a b : ℝ) : Set ℝ := {1/a, |a|, b/a}

-- State the theorem
theorem equal_sets_implies_b_minus_a_equals_one (a b : ℝ) :
  A a = B a b → b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_sets_implies_b_minus_a_equals_one_l16_1652


namespace NUMINAMATH_CALUDE_power_division_equals_512_l16_1650

theorem power_division_equals_512 : 8^15 / 64^6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_512_l16_1650


namespace NUMINAMATH_CALUDE_second_pipe_fill_time_l16_1642

theorem second_pipe_fill_time (pipe1_rate : ℝ) (pipe2_rate : ℝ) (pipe3_rate : ℝ) 
  (combined_fill_time : ℝ) :
  pipe1_rate = 1 / 10 →
  pipe3_rate = -1 / 20 →
  combined_fill_time = 7.5 →
  pipe1_rate + pipe2_rate + pipe3_rate = 1 / combined_fill_time →
  1 / pipe2_rate = 60 := by
  sorry

end NUMINAMATH_CALUDE_second_pipe_fill_time_l16_1642


namespace NUMINAMATH_CALUDE_square_perimeter_no_conditional_l16_1633

-- Define the problem types
inductive Problem
| OppositeNumber
| SquarePerimeter
| MaximumOfThree
| BinaryToDecimal

-- Define a predicate for problems that don't require conditional statements
def NoConditionalRequired (p : Problem) : Prop :=
  match p with
  | Problem.SquarePerimeter => True
  | _ => False

-- Theorem statement
theorem square_perimeter_no_conditional :
  NoConditionalRequired Problem.SquarePerimeter ∧
  ¬NoConditionalRequired Problem.OppositeNumber ∧
  ¬NoConditionalRequired Problem.MaximumOfThree ∧
  ¬NoConditionalRequired Problem.BinaryToDecimal :=
sorry

end NUMINAMATH_CALUDE_square_perimeter_no_conditional_l16_1633


namespace NUMINAMATH_CALUDE_max_ratio_OB_OA_l16_1607

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y = 1
def C₂ (x y φ : ℝ) : Prop := x = 2 + 2 * Real.cos φ ∧ y = 2 * Real.sin φ ∧ 0 ≤ φ ∧ φ < 2 * Real.pi

-- Define the ray l
def l (ρ θ α : ℝ) : Prop := θ = α ∧ ρ ≥ 0

-- Define points A and B
def A (ρ θ : ℝ) : Prop := C₁ (ρ * Real.cos θ) (ρ * Real.sin θ) ∧ l ρ θ θ
def B (ρ θ : ℝ) : Prop := ∃ φ, C₂ (ρ * Real.cos θ) (ρ * Real.sin θ) φ ∧ l ρ θ θ

-- State the theorem
theorem max_ratio_OB_OA :
  ∃ (max : ℝ), max = 2 + 2 * Real.sqrt 2 ∧
  ∀ α : ℝ, 0 ≤ α ∧ α ≤ Real.pi / 2 →
    ∀ ρA ρB θA θB : ℝ,
      A ρA θA → B ρB θB →
      ρB / ρA ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_ratio_OB_OA_l16_1607


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_13_l16_1636

theorem smallest_n_multiple_of_13 (x y : ℤ) 
  (h1 : (2 * x - 3) % 13 = 0) 
  (h2 : (3 * y + 4) % 13 = 0) : 
  ∃ n : ℕ+, (x^2 - x*y + y^2 + n) % 13 = 0 ∧ 
  ∀ m : ℕ+, m < n → (x^2 - x*y + y^2 + m) % 13 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_13_l16_1636


namespace NUMINAMATH_CALUDE_smallest_n_dividing_2016_l16_1616

theorem smallest_n_dividing_2016 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (2016 ∣ (20^m - 16^m)) → m ≥ n) ∧ 
  (2016 ∣ (20^n - 16^n)) ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_dividing_2016_l16_1616


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l16_1645

theorem at_least_one_equation_has_two_roots (p q₁ q₂ : ℝ) (h : p = q₁ + q₂ + 1) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + x + q₁ = 0 ∧ y^2 + y + q₁ = 0) ∨
  (∃ x y : ℝ, x ≠ y ∧ x^2 + p*x + q₂ = 0 ∧ y^2 + p*y + q₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l16_1645


namespace NUMINAMATH_CALUDE_student_count_l16_1611

theorem student_count (band : ℕ) (sports : ℕ) (both : ℕ) (total : ℕ) : 
  band = 85 → 
  sports = 200 → 
  both = 60 → 
  total = 225 → 
  band + sports - both = total :=
by sorry

end NUMINAMATH_CALUDE_student_count_l16_1611


namespace NUMINAMATH_CALUDE_equation_properties_l16_1693

def p (x : ℝ) := x^4 - x^3 - 1

theorem equation_properties :
  (∃ (r₁ r₂ : ℝ), p r₁ = 0 ∧ p r₂ = 0 ∧ r₁ ≠ r₂ ∧
    (∀ (r : ℝ), p r = 0 → r = r₁ ∨ r = r₂)) ∧
  (∀ (r₁ r₂ : ℝ), p r₁ = 0 → p r₂ = 0 → r₁ + r₂ > 6/11) ∧
  (∀ (r₁ r₂ : ℝ), p r₁ = 0 → p r₂ = 0 → r₁ * r₂ < -11/10) :=
by sorry

end NUMINAMATH_CALUDE_equation_properties_l16_1693


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_35_degrees_l16_1673

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125 degrees. -/
theorem supplement_of_complement_of_35_degrees : 
  let original_angle : ℝ := 35
  let complement : ℝ := 90 - original_angle
  let supplement : ℝ := 180 - complement
  supplement = 125 := by
sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_35_degrees_l16_1673


namespace NUMINAMATH_CALUDE_square_of_larger_number_l16_1620

theorem square_of_larger_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 8) : x^2 = 1156 := by
  sorry

end NUMINAMATH_CALUDE_square_of_larger_number_l16_1620


namespace NUMINAMATH_CALUDE_polynomial_expansion_l16_1647

theorem polynomial_expansion :
  ∀ x : ℝ, (3 * x^2 + 4 * x - 5) * (4 * x^3 - 3 * x + 2) = 
    12 * x^5 + 16 * x^4 - 24 * x^3 - 6 * x^2 + 17 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l16_1647


namespace NUMINAMATH_CALUDE_max_value_reciprocal_sum_l16_1632

/-- Given a quadratic polynomial x^2 - tx + q with roots α and β,
    where α + β = α^2 + β^2 = α^3 + β^3 = ... = α^2010 + β^2010,
    the maximum possible value of 1/α^2011 + 1/β^2011 is 2. -/
theorem max_value_reciprocal_sum (t q α β : ℝ) : 
  (∀ k : ℕ, k ≥ 1 ∧ k ≤ 2010 → α^k + β^k = α + β) →
  α^2 - t*α + q = 0 →
  β^2 - t*β + q = 0 →
  α ≠ 0 →
  β ≠ 0 →
  (1/α^2011 + 1/β^2011 : ℝ) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_reciprocal_sum_l16_1632


namespace NUMINAMATH_CALUDE_smallest_positive_solution_of_quartic_l16_1608

theorem smallest_positive_solution_of_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 ∧ x > 0 → x = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_of_quartic_l16_1608


namespace NUMINAMATH_CALUDE_area_of_triangle_AOC_l16_1674

/-- Given three collinear points A, B, and C in a Cartesian coordinate system with origin O,
    where OA = (-2, m), OB = (n, 1), OC = (5, -1), OA ⊥ OB,
    G is the centroid of triangle OAC, and OB = (3/2) * OG,
    prove that the area of triangle AOC is 13/2. -/
theorem area_of_triangle_AOC (m n : ℝ) (A B C G : ℝ × ℝ) :
  A.1 = -2 ∧ A.2 = m →
  B.1 = n ∧ B.2 = 1 →
  C = (5, -1) →
  A.1 * B.1 + A.2 * B.2 = 0 →  -- OA ⊥ OB
  G = ((0 + A.1 + C.1) / 3, (0 + A.2 + C.2) / 3) →  -- G is centroid of OAC
  B = (3/2 : ℝ) • G →  -- OB = (3/2) * OG
  (A.1 - C.1) * (B.2 - A.2) = (B.1 - A.1) * (A.2 - C.2) →  -- A, B, C are collinear
  abs ((A.1 * C.2 - C.1 * A.2) / 2) = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AOC_l16_1674


namespace NUMINAMATH_CALUDE_abc_product_l16_1671

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 30 * Real.rpow 3 (1/3))
  (hac : a * c = 42 * Real.rpow 3 (1/3))
  (hbc : b * c = 21 * Real.rpow 3 (1/3)) :
  a * b * c = 210 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l16_1671


namespace NUMINAMATH_CALUDE_range_of_a_l16_1638

-- Define the set of real numbers x in [1,2]
def X : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }

-- Define the set of real numbers y in [2,3]
def Y : Set ℝ := { y | 2 ≤ y ∧ y ≤ 3 }

-- State the theorem
theorem range_of_a (x : ℝ) (y : ℝ) (h1 : x ∈ X) (h2 : y ∈ Y) :
  ∃ a : ℝ, (∀ (x' : ℝ) (y' : ℝ), x' ∈ X → y' ∈ Y → x'*y' ≤ a*x'^2 + 2*y'^2) ∧
            (a ≥ -1) ∧
            (∀ b : ℝ, b > a → ∃ (x' : ℝ) (y' : ℝ), x' ∈ X ∧ y' ∈ Y ∧ x'*y' > b*x'^2 + 2*y'^2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l16_1638


namespace NUMINAMATH_CALUDE_monster_hunt_sum_l16_1666

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem monster_hunt_sum :
  geometric_sum 2 2 5 = 62 :=
sorry

end NUMINAMATH_CALUDE_monster_hunt_sum_l16_1666


namespace NUMINAMATH_CALUDE_fraction_simplification_l16_1665

theorem fraction_simplification (x : ℝ) (h : x = 3) :
  (x^8 + 20*x^4 + 100) / (x^4 + 10) = 91 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l16_1665


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l16_1681

/-- Given three lines that intersect at the same point, find the value of p -/
theorem intersection_of_three_lines (p : ℝ) : 
  (∃ x y : ℝ, y = 3*x - 6 ∧ y = -4*x + 8 ∧ y = 7*x + p) → p = -14 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l16_1681


namespace NUMINAMATH_CALUDE_maia_daily_requests_l16_1698

/-- The number of client requests Maia works on each day -/
def requests_per_day : ℕ := 4

/-- The number of days Maia works -/
def days_worked : ℕ := 5

/-- The number of client requests remaining after the working period -/
def remaining_requests : ℕ := 10

/-- The number of client requests Maia gets every day -/
def daily_requests : ℕ := 6

theorem maia_daily_requests : 
  days_worked * daily_requests = days_worked * requests_per_day + remaining_requests :=
by sorry

end NUMINAMATH_CALUDE_maia_daily_requests_l16_1698


namespace NUMINAMATH_CALUDE_smallest_gamma_for_integer_solution_l16_1612

theorem smallest_gamma_for_integer_solution :
  ∃ (γ : ℕ), γ > 0 ∧
  (∃ (x : ℕ), (Real.sqrt x - Real.sqrt (24 * γ) = 4 * Real.sqrt 2)) ∧
  (∀ (γ' : ℕ), 0 < γ' ∧ γ' < γ →
    ¬∃ (x : ℕ), (Real.sqrt x - Real.sqrt (24 * γ') = 4 * Real.sqrt 2)) ∧
  γ = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gamma_for_integer_solution_l16_1612


namespace NUMINAMATH_CALUDE_prime_power_sum_l16_1614

theorem prime_power_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 3250 → 2*w + 3*x + 4*y + 5*z = 19 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l16_1614


namespace NUMINAMATH_CALUDE_fractional_simplification_l16_1651

theorem fractional_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  2 / (x + 1) - (x + 5) / (x^2 - 1) = (x - 7) / ((x + 1) * (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fractional_simplification_l16_1651


namespace NUMINAMATH_CALUDE_satellite_orbits_in_week_l16_1639

/-- The number of orbits a satellite completes in one week -/
def orbits_in_week (hours_per_orbit : ℕ) (days_per_week : ℕ) (hours_per_day : ℕ) : ℕ :=
  (days_per_week * hours_per_day) / hours_per_orbit

/-- Theorem: A satellite orbiting Earth once every 7 hours completes 24 orbits in one week -/
theorem satellite_orbits_in_week :
  orbits_in_week 7 7 24 = 24 := by
  sorry

#eval orbits_in_week 7 7 24

end NUMINAMATH_CALUDE_satellite_orbits_in_week_l16_1639
