import Mathlib

namespace NUMINAMATH_CALUDE_Q_iff_a_in_open_interval_P_or_Q_and_not_P_and_Q_iff_a_in_union_l2212_221271

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def Q (a : ℝ) : Prop := ∃ x y : ℝ, x^2 / (a + 1) + y^2 / (a - 2) = 1 ∧ (a + 1) * (a - 2) < 0

-- Theorem 1
theorem Q_iff_a_in_open_interval (a : ℝ) : Q a ↔ a ∈ Set.Ioo (-1) 2 := by sorry

-- Theorem 2
theorem P_or_Q_and_not_P_and_Q_iff_a_in_union (a : ℝ) : 
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) ↔ a ∈ Set.Ioo 1 2 ∪ Set.Iic (-1) := by sorry

end NUMINAMATH_CALUDE_Q_iff_a_in_open_interval_P_or_Q_and_not_P_and_Q_iff_a_in_union_l2212_221271


namespace NUMINAMATH_CALUDE_trig_identity_l2212_221272

theorem trig_identity (α β : ℝ) : 
  (Real.cos α - Real.cos β)^2 - (Real.sin α - Real.sin β)^2 = 
  -4 * (Real.sin ((α - β)/2))^2 * Real.cos (α + β) := by sorry

end NUMINAMATH_CALUDE_trig_identity_l2212_221272


namespace NUMINAMATH_CALUDE_last_interval_correct_l2212_221246

/-- Represents a clock with specific ringing behavior -/
structure Clock where
  n : ℕ  -- number of rings per day
  x : ℝ  -- time between first two rings (in hours)
  y : ℝ  -- increase in time between subsequent rings (in hours)

/-- The time between the last two rings of the clock -/
def lastInterval (c : Clock) : ℝ :=
  c.x + (c.n - 3 : ℝ) * c.y

theorem last_interval_correct (c : Clock) (h : c.n ≥ 2) :
  lastInterval c = c.x + (c.n - 3 : ℝ) * c.y :=
sorry

end NUMINAMATH_CALUDE_last_interval_correct_l2212_221246


namespace NUMINAMATH_CALUDE_range_of_a_l2212_221260

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x : ℝ | 2*a - 1 < x ∧ x < 4*a}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}

-- State the theorem
theorem range_of_a (h : N ⊆ M a) : a ∈ Set.Icc (1/2 : ℝ) 2 := by
  sorry

-- Note: Set.Icc represents a closed interval [1/2, 2]

end NUMINAMATH_CALUDE_range_of_a_l2212_221260


namespace NUMINAMATH_CALUDE_diborane_combustion_heat_correct_l2212_221267

/-- Represents the heat of vaporization of water in kJ/mol -/
def water_vaporization_heat : ℝ := 44

/-- Represents the amount of diborane in moles -/
def diborane_amount : ℝ := 0.3

/-- Represents the heat released during combustion in kJ -/
def heat_released : ℝ := 609.9

/-- Represents the heat of combustion of diborane in kJ/mol -/
def diborane_combustion_heat : ℝ := -2165

/-- Theorem stating that the given heat of combustion of diborane is correct -/
theorem diborane_combustion_heat_correct : 
  diborane_combustion_heat = -heat_released / diborane_amount - 3 * water_vaporization_heat :=
sorry

end NUMINAMATH_CALUDE_diborane_combustion_heat_correct_l2212_221267


namespace NUMINAMATH_CALUDE_largest_n_implies_x_l2212_221281

/-- Binary operation @ defined as n - (n * x) -/
def binary_op (n : ℤ) (x : ℝ) : ℝ := n - (n * x)

/-- Theorem stating that if 5 is the largest positive integer n such that n @ x < 21, then x = -3 -/
theorem largest_n_implies_x (x : ℝ) :
  (∀ n : ℤ, n > 0 → binary_op n x < 21 → n ≤ 5) ∧
  (binary_op 5 x < 21) →
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_implies_x_l2212_221281


namespace NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l2212_221296

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

def decimal_division (a b : Nat) : Nat :=
  a / b

theorem base_conversion_and_arithmetic :
  let n1 := base_to_decimal [3, 6, 4, 1] 7
  let n2 := base_to_decimal [1, 2, 1] 5
  let n3 := base_to_decimal [4, 5, 7, 1] 6
  let n4 := base_to_decimal [6, 5, 4, 3] 7
  decimal_division n1 n2 - n3 * 2 + n4 = 278 := by sorry

end NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l2212_221296


namespace NUMINAMATH_CALUDE_no_abc_divisible_by_9_l2212_221287

theorem no_abc_divisible_by_9 :
  ∀ (a b c : ℤ), ∃ (x : ℤ), ¬ (9 ∣ ((x + a) * (x + b) * (x + c) - x^3 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_abc_divisible_by_9_l2212_221287


namespace NUMINAMATH_CALUDE_same_color_probability_is_correct_l2212_221266

def white_balls : ℕ := 7
def black_balls : ℕ := 6
def red_balls : ℕ := 2

def total_balls : ℕ := white_balls + black_balls + red_balls

def same_color_probability : ℚ :=
  (Nat.choose white_balls 2 + Nat.choose black_balls 2 + Nat.choose red_balls 2) /
  Nat.choose total_balls 2

theorem same_color_probability_is_correct :
  same_color_probability = 37 / 105 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_correct_l2212_221266


namespace NUMINAMATH_CALUDE_f_properties_l2212_221236

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - 2*a| + |x - a|

theorem f_properties (a : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (a = 1 ∧ f 1 x > 3) ↔ (x < 0 ∨ x > 3)) ∧
  (∀ b : ℝ, b ≠ 0 → f a b ≥ f a a) ∧
  (∀ b : ℝ, b ≠ 0 → (f a b = f a a ↔ (2*a - b) * (b - a) ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2212_221236


namespace NUMINAMATH_CALUDE_book_profit_rate_l2212_221216

/-- Given a cost price and a selling price, calculate the rate of profit -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at 50 Rs and sold at 70 Rs is 40% -/
theorem book_profit_rate : rate_of_profit 50 70 = 40 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_rate_l2212_221216


namespace NUMINAMATH_CALUDE_complete_square_sum_l2212_221285

theorem complete_square_sum (a b c : ℤ) : 
  (∀ x : ℝ, 25 * x^2 + 30 * x - 75 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = -58 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_sum_l2212_221285


namespace NUMINAMATH_CALUDE_other_number_proof_l2212_221257

/-- Given two positive integers with specific HCF and LCM, prove that if one number is 36, the other is 154 -/
theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 14) (h2 : Nat.lcm a b = 396) (h3 : a = 36) : b = 154 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l2212_221257


namespace NUMINAMATH_CALUDE_imaginary_town_population_l2212_221232

theorem imaginary_town_population (n m p : ℕ) 
  (h1 : n^2 + 150 = m^2 + 1) 
  (h2 : n^2 + 300 = p^2) : 
  4 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_imaginary_town_population_l2212_221232


namespace NUMINAMATH_CALUDE_interest_rate_difference_l2212_221278

theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 750) 
  (h2 : time = 2) 
  (h3 : interest_difference = 60) : 
  ∃ (original_rate higher_rate : ℝ),
    principal * higher_rate * time / 100 - principal * original_rate * time / 100 = interest_difference ∧ 
    higher_rate - original_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l2212_221278


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2212_221262

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 3 * x / (x - 2) + (3 * x^2 - 36) / x
  ∃ (y : ℝ), y = (2 - Real.sqrt 58) / 3 ∧ 
    f y = 13 ∧ 
    ∀ (z : ℝ), f z = 13 → y ≤ z := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2212_221262


namespace NUMINAMATH_CALUDE_root_equation_result_l2212_221274

/-- Given two constants c and d, if the equation ((x+c)(x+d)(x-15))/((x-4)^2) = 0 has exactly 3 distinct roots,
    and the equation ((x+2c)(x-4)(x-9))/((x+d)(x-15)) = 0 has exactly 1 distinct root,
    then 100c + d = -391 -/
theorem root_equation_result (c d : ℝ) 
  (h1 : ∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
    ∀ x, (x + c) * (x + d) * (x - 15) = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3)
  (h2 : ∃! (r : ℝ), ∀ x, (x + 2*c) * (x - 4) * (x - 9) = 0 ↔ x = r) :
  100 * c + d = -391 :=
sorry

end NUMINAMATH_CALUDE_root_equation_result_l2212_221274


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2212_221244

theorem arithmetic_calculation : 8 / 2 - 3 - 12 + 3 * (5^2 - 4) = 52 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2212_221244


namespace NUMINAMATH_CALUDE_tickets_won_later_l2212_221250

/-- Given Cody's initial tickets, tickets spent on a beanie, and final ticket count,
    prove the number of tickets he won later. -/
theorem tickets_won_later
  (initial_tickets : ℕ)
  (tickets_spent : ℕ)
  (final_tickets : ℕ)
  (h1 : initial_tickets = 49)
  (h2 : tickets_spent = 25)
  (h3 : final_tickets = 30) :
  final_tickets - (initial_tickets - tickets_spent) = 6 := by
  sorry

end NUMINAMATH_CALUDE_tickets_won_later_l2212_221250


namespace NUMINAMATH_CALUDE_square_sum_equation_l2212_221226

theorem square_sum_equation (x y : ℝ) (h1 : x + 2*y = 8) (h2 : x*y = 1) : x^2 + 4*y^2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equation_l2212_221226


namespace NUMINAMATH_CALUDE_change_calculation_l2212_221210

/-- Given the cost of milk and water, and the amount paid, calculate the change received. -/
theorem change_calculation (milk_cost water_cost paid : ℕ) 
  (h_milk : milk_cost = 350)
  (h_water : water_cost = 500)
  (h_paid : paid = 1000) :
  paid - (milk_cost + water_cost) = 150 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l2212_221210


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_ratio_l2212_221273

theorem two_numbers_sum_and_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 900) (h4 : y = 19 * x) : x = 45 ∧ y = 855 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_ratio_l2212_221273


namespace NUMINAMATH_CALUDE_polynomial_integer_values_l2212_221270

/-- A polynomial of degree 3 with real coefficients -/
def Polynomial3 := ℝ → ℝ

/-- Predicate to check if a number is an integer -/
def IsInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Main theorem: If a polynomial of degree 3 takes integer values at four consecutive integers,
    then it takes integer values at all integers -/
theorem polynomial_integer_values (P : Polynomial3) (i : ℤ) 
  (h1 : IsInteger (P i))
  (h2 : IsInteger (P (i + 1)))
  (h3 : IsInteger (P (i + 2)))
  (h4 : IsInteger (P (i + 3))) :
  ∀ n : ℤ, IsInteger (P n) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_integer_values_l2212_221270


namespace NUMINAMATH_CALUDE_f_neg_one_equals_five_l2212_221220

-- Define the function f
def f (x : ℝ) : ℝ := (1 - x)^2 + 1

-- State the theorem
theorem f_neg_one_equals_five : f (-1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_equals_five_l2212_221220


namespace NUMINAMATH_CALUDE_book_selection_theorem_l2212_221292

def number_of_ways_to_select_books (total_books : ℕ) (identical_books : ℕ) (different_books : ℕ) (books_to_select : ℕ) : ℕ :=
  -- Select all identical books
  (if books_to_select ≤ identical_books then 1 else 0) +
  -- Select some identical books and some different books
  (Finset.sum (Finset.range (min identical_books books_to_select + 1)) (fun i =>
    Nat.choose identical_books i * Nat.choose different_books (books_to_select - i)))

theorem book_selection_theorem :
  number_of_ways_to_select_books 9 3 6 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l2212_221292


namespace NUMINAMATH_CALUDE_johns_remaining_money_l2212_221201

def remaining_money (initial_amount : ℚ) (snack_fraction : ℚ) (necessity_fraction : ℚ) : ℚ :=
  let after_snacks := initial_amount * (1 - snack_fraction)
  after_snacks * (1 - necessity_fraction)

theorem johns_remaining_money :
  remaining_money 20 (1/5) (3/4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l2212_221201


namespace NUMINAMATH_CALUDE_cost_of_20_pencils_15_notebooks_l2212_221215

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook -/
def notebook_cost : ℝ := sorry

/-- The first condition: 9 pencils and 10 notebooks cost $5.45 -/
axiom condition1 : 9 * pencil_cost + 10 * notebook_cost = 5.45

/-- The second condition: 6 pencils and 4 notebooks cost $2.50 -/
axiom condition2 : 6 * pencil_cost + 4 * notebook_cost = 2.50

/-- Theorem: The cost of 20 pencils and 15 notebooks is $9.04 -/
theorem cost_of_20_pencils_15_notebooks :
  20 * pencil_cost + 15 * notebook_cost = 9.04 := by sorry

end NUMINAMATH_CALUDE_cost_of_20_pencils_15_notebooks_l2212_221215


namespace NUMINAMATH_CALUDE_evaluate_g_l2212_221230

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + x + 1

-- State the theorem
theorem evaluate_g : 3 * g 2 + 2 * g (-2) = -9 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l2212_221230


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2212_221298

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = (1/x^2 - 2*x)^6) ∧ 
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) ∧
  (∃! c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) ∧
  c = 240 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2212_221298


namespace NUMINAMATH_CALUDE_polynomial_division_l2212_221233

theorem polynomial_division (x : ℝ) :
  8 * x^3 - 2 * x^2 + 4 * x - 9 = (x - 3) * (8 * x^2 + 22 * x + 70) + 201 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l2212_221233


namespace NUMINAMATH_CALUDE_unique_equidistant_point_l2212_221291

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- The distance between a point and a line -/
def distancePointToLine (p : Point) (l : Line) : ℝ :=
  sorry

/-- The distance between a point and a circle -/
def distancePointToCircle (p : Point) (c : Circle) : ℝ :=
  sorry

/-- Checks if two lines are parallel -/
def areParallel (l1 l2 : Line) : Prop :=
  sorry

/-- Theorem stating that there is exactly one point equidistant from a circle
    and two parallel tangents under specific conditions -/
theorem unique_equidistant_point
  (c : Circle)
  (l1 l2 : Line)
  (h1 : c.radius = 4)
  (h2 : areParallel l1 l2)
  (h3 : distancePointToLine c.center l1 = 6)
  (h4 : distancePointToLine c.center l2 = 6) :
  ∃! p : Point,
    distancePointToCircle p c = distancePointToLine p l1 ∧
    distancePointToCircle p c = distancePointToLine p l2 :=
  sorry

end NUMINAMATH_CALUDE_unique_equidistant_point_l2212_221291


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l2212_221222

/-- The perimeter of a quadrilateral with sides x, x + 1, 6, and 10, where x = 3, is 23. -/
theorem quadrilateral_perimeter (x : ℝ) (h : x = 3) : x + (x + 1) + 6 + 10 = 23 := by
  sorry

#check quadrilateral_perimeter

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l2212_221222


namespace NUMINAMATH_CALUDE_yellow_yarns_count_l2212_221221

/-- The number of scarves that can be made from one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

/-- The total number of scarves May can make -/
def total_scarves : ℕ := 36

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := (total_scarves - (red_yarns + blue_yarns) * scarves_per_yarn) / scarves_per_yarn

theorem yellow_yarns_count : yellow_yarns = 4 := by
  sorry

end NUMINAMATH_CALUDE_yellow_yarns_count_l2212_221221


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l2212_221289

/-- The number of positive perfect square factors of (2^14)(3^18)(7^21) -/
def perfect_square_factors : ℕ := sorry

/-- The given number -/
def given_number : ℕ := 2^14 * 3^18 * 7^21

theorem count_perfect_square_factors :
  perfect_square_factors = 880 :=
sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l2212_221289


namespace NUMINAMATH_CALUDE_franks_earnings_l2212_221265

/-- Represents Frank's work schedule and pay rates -/
structure WorkSchedule where
  totalHours : ℕ
  days : ℕ
  regularRate : ℚ
  overtimeRate : ℚ
  day1Hours : ℕ
  day2Hours : ℕ
  day3Hours : ℕ
  day4Hours : ℕ

/-- Calculates the total earnings based on the work schedule -/
def calculateEarnings (schedule : WorkSchedule) : ℚ :=
  let regularHours := min schedule.totalHours (schedule.days * 8)
  let overtimeHours := schedule.totalHours - regularHours
  regularHours * schedule.regularRate + overtimeHours * schedule.overtimeRate

/-- Frank's work schedule for the week -/
def franksSchedule : WorkSchedule :=
  { totalHours := 32
  , days := 4
  , regularRate := 15
  , overtimeRate := 22.5
  , day1Hours := 12
  , day2Hours := 8
  , day3Hours := 8
  , day4Hours := 12
  }

/-- Theorem stating that Frank's total earnings for the week are $660 -/
theorem franks_earnings : calculateEarnings franksSchedule = 660 := by
  sorry

end NUMINAMATH_CALUDE_franks_earnings_l2212_221265


namespace NUMINAMATH_CALUDE_prob_at_least_one_2_or_3_is_7_16_l2212_221293

/-- The probability of at least one of two fair 8-sided dice showing a 2 or a 3 -/
def prob_at_least_one_2_or_3 : ℚ := 7/16

/-- Two fair 8-sided dice are rolled -/
axiom fair_8_sided_dice : True

theorem prob_at_least_one_2_or_3_is_7_16 :
  prob_at_least_one_2_or_3 = 7/16 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_2_or_3_is_7_16_l2212_221293


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2212_221256

def U : Set ℕ := {x | x ≤ 8}
def A : Set ℕ := {1, 3, 7}
def B : Set ℕ := {2, 3, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {0, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2212_221256


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l2212_221280

/-- Represents a geometric sequence with sum S_n = 3^n + a -/
structure GeometricSequence where
  a : ℝ  -- The constant term in the sum formula
  -- Sequence definition: a_n = S_n - S_{n-1}
  seq : ℕ → ℝ := λ n => 3^n + a - (3^(n-1) + a)

/-- The first term of the sequence is 2 -/
axiom first_term (s : GeometricSequence) : s.seq 1 = 2

/-- The common ratio of the sequence is 3 -/
axiom common_ratio (s : GeometricSequence) : s.seq 2 = 3 * s.seq 1

/-- Theorem: The value of 'a' in the sum formula S_n = 3^n + a is -1 -/
theorem geometric_sequence_constant (s : GeometricSequence) : s.a = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l2212_221280


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2212_221253

theorem complex_sum_problem (u v w x y z : ℂ) : 
  v = 2 →
  y = -u - w →
  u + v * I + w + x * I + y + z * I = 2 * I →
  x + z = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2212_221253


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l2212_221258

/-- The number of dogwood trees in the park after planting -/
def total_trees (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem stating that the total number of dogwood trees after planting is 83 -/
theorem dogwood_tree_count : total_trees 34 49 = 83 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l2212_221258


namespace NUMINAMATH_CALUDE_ashton_sheets_l2212_221209

theorem ashton_sheets (jimmy_sheets : ℕ) (tommy_sheets : ℕ) (ashton_sheets : ℕ) : 
  jimmy_sheets = 32 →
  tommy_sheets = jimmy_sheets + 10 →
  jimmy_sheets + ashton_sheets = tommy_sheets + 30 →
  ashton_sheets = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_ashton_sheets_l2212_221209


namespace NUMINAMATH_CALUDE_seven_lines_intersections_l2212_221228

/-- The maximum number of intersection points for n lines in a plane -/
def max_intersections (n : ℕ) : ℕ := n.choose 2

/-- The set of possible numbers of intersection points for 7 lines in a plane -/
def possible_intersections : Set ℕ :=
  {0, 1} ∪ Set.Icc 6 21

theorem seven_lines_intersections :
  (max_intersections 7 = 21) ∧
  (possible_intersections = {0, 1} ∪ Set.Icc 6 21) :=
sorry

end NUMINAMATH_CALUDE_seven_lines_intersections_l2212_221228


namespace NUMINAMATH_CALUDE_average_shift_l2212_221224

theorem average_shift (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 40) :
  ((x₁ + 40) + (x₂ + 40) + (x₃ + 40)) / 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_shift_l2212_221224


namespace NUMINAMATH_CALUDE_gcd_power_minus_one_l2212_221235

theorem gcd_power_minus_one (m n : ℕ+) :
  Nat.gcd ((2 : ℕ) ^ m.val - 1) ((2 : ℕ) ^ n.val - 1) = (2 : ℕ) ^ (Nat.gcd m.val n.val) - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_minus_one_l2212_221235


namespace NUMINAMATH_CALUDE_sum_reciprocal_squares_cubic_l2212_221261

theorem sum_reciprocal_squares_cubic (a b c : ℝ) : 
  a^3 - 12*a^2 + 20*a - 3 = 0 →
  b^3 - 12*b^2 + 20*b - 3 = 0 →
  c^3 - 12*c^2 + 20*c - 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 328/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squares_cubic_l2212_221261


namespace NUMINAMATH_CALUDE_problem_statement_l2212_221264

theorem problem_statement (a b : ℝ) (h : a * b + b^2 = 12) :
  (a + b)^2 - (a + b) * (a - b) = 24 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2212_221264


namespace NUMINAMATH_CALUDE_equation_roots_existence_l2212_221279

-- Define the equation
def equation (x a : ℝ) : Prop := |x^2 - x| - a = 0

-- Define the number of different real roots for a given 'a'
def num_roots (a : ℝ) : ℕ := sorry

-- Theorem statement
theorem equation_roots_existence :
  (∃ a : ℝ, num_roots a = 2) ∧
  (∃ a : ℝ, num_roots a = 3) ∧
  (∃ a : ℝ, num_roots a = 4) ∧
  (¬ ∃ a : ℝ, num_roots a = 6) :=
sorry

end NUMINAMATH_CALUDE_equation_roots_existence_l2212_221279


namespace NUMINAMATH_CALUDE_no_common_points_l2212_221242

theorem no_common_points : ¬∃ (x y : ℝ), 
  (x^2 + 4*y^2 = 4) ∧ (4*x^2 + y^2 = 4) ∧ (x^2 + y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_common_points_l2212_221242


namespace NUMINAMATH_CALUDE_min_value_fraction_l2212_221247

theorem min_value_fraction (x : ℝ) (h : x > 10) :
  x^2 / (x - 10) ≥ 40 ∧ ∃ y > 10, y^2 / (y - 10) = 40 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2212_221247


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2212_221229

theorem sqrt_sum_inequality (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) 
  (sum_eq : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2212_221229


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2212_221286

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m > 24 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ k : ℤ, k ≤ 24 → k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2212_221286


namespace NUMINAMATH_CALUDE_toluene_formation_l2212_221218

-- Define the chemical species involved in the reaction
structure ChemicalSpecies where
  formula : String
  moles : ℝ

-- Define the chemical reaction
def reaction (reactant1 reactant2 product1 product2 : ChemicalSpecies) : Prop :=
  reactant1.formula = "C6H6" ∧ 
  reactant2.formula = "CH4" ∧ 
  product1.formula = "C6H5CH3" ∧ 
  product2.formula = "H2" ∧
  reactant1.moles = reactant2.moles ∧
  product1.moles = product2.moles ∧
  reactant1.moles = product1.moles

-- Theorem statement
theorem toluene_formation 
  (benzene : ChemicalSpecies)
  (methane : ChemicalSpecies)
  (toluene : ChemicalSpecies)
  (hydrogen : ChemicalSpecies)
  (h1 : reaction benzene methane toluene hydrogen)
  (h2 : methane.moles = 3)
  (h3 : hydrogen.moles = 3) :
  toluene.moles = 3 :=
sorry

end NUMINAMATH_CALUDE_toluene_formation_l2212_221218


namespace NUMINAMATH_CALUDE_lamp_arrangement_probability_l2212_221282

/-- The total number of lamps -/
def total_lamps : ℕ := 8

/-- The number of red lamps -/
def red_lamps : ℕ := 4

/-- The number of blue lamps -/
def blue_lamps : ℕ := 4

/-- The number of lamps to be turned on -/
def lamps_on : ℕ := 4

/-- The probability of the specific arrangement -/
def target_probability : ℚ := 4 / 49

/-- Theorem stating the probability of the specific arrangement -/
theorem lamp_arrangement_probability :
  let total_arrangements := Nat.choose total_lamps red_lamps * Nat.choose total_lamps lamps_on
  let favorable_arrangements := Nat.choose (total_lamps - 2) (red_lamps - 1) * Nat.choose (total_lamps - 2) (lamps_on - 1)
  (favorable_arrangements : ℚ) / total_arrangements = target_probability := by
  sorry


end NUMINAMATH_CALUDE_lamp_arrangement_probability_l2212_221282


namespace NUMINAMATH_CALUDE_zoo_animals_l2212_221251

theorem zoo_animals (sea_lions : ℕ) (penguins : ℕ) : 
  sea_lions = 48 →
  sea_lions * 11 = penguins * 4 →
  penguins > sea_lions →
  penguins - sea_lions = 84 := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_l2212_221251


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2212_221200

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^5 + 1 = (x^2 - 3*x + 5) * q + (11*x - 14) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2212_221200


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_f_always_nonpositive_l2212_221248

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - 2 * |x - a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | 2/3 < x ∧ x < 2} :=
sorry

-- Part II
theorem a_value_when_f_always_nonpositive :
  (∀ x : ℝ, f a x ≤ 0) → a = -1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_f_always_nonpositive_l2212_221248


namespace NUMINAMATH_CALUDE_first_month_sale_l2212_221263

def last_four_months_sales : List Int := [5660, 6200, 6350, 6500]
def sixth_month_sale : Int := 8270
def average_sale : Int := 6400
def num_months : Int := 6

theorem first_month_sale :
  (num_months * average_sale) - (sixth_month_sale + last_four_months_sales.sum) = 5420 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_l2212_221263


namespace NUMINAMATH_CALUDE_divisibility_criterion_1207_l2212_221217

-- Define a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define the sum of cubes of digits
def sum_of_cubes_of_digits (n : ℕ) : ℕ :=
  (n / 10) ^ 3 + (n % 10) ^ 3

-- Theorem statement
theorem divisibility_criterion_1207 (x : ℕ) :
  is_two_digit x →
  sum_of_cubes_of_digits x = 344 →
  (1207 % x = 0 ↔ (x = 17 ∨ x = 71)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_criterion_1207_l2212_221217


namespace NUMINAMATH_CALUDE_smallest_upper_bound_for_triangle_ratio_l2212_221241

/-- Triangle sides -/
structure Triangle :=
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (triangle_ineq_ab : c < a + b)
  (triangle_ineq_bc : a < b + c)
  (triangle_ineq_ca : b < c + a)
  (a_neq_b : a ≠ b)

/-- The smallest upper bound for (a² + b²) / c² in any triangle with unequal sides -/
theorem smallest_upper_bound_for_triangle_ratio :
  ∃ N : ℝ, (∀ t : Triangle, (t.a^2 + t.b^2) / t.c^2 < N) ∧
           (∀ ε > 0, ∃ t : Triangle, N - ε < (t.a^2 + t.b^2) / t.c^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_for_triangle_ratio_l2212_221241


namespace NUMINAMATH_CALUDE_valid_pairs_l2212_221276

def is_valid_pair (a b : ℕ+) : Prop :=
  (∃ k : ℤ, (a.val ^ 3 * b.val - 1) = k * (a.val + 1)) ∧
  (∃ m : ℤ, (b.val ^ 3 * a.val + 1) = m * (b.val - 1))

theorem valid_pairs :
  ∀ a b : ℕ+, is_valid_pair a b →
    ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l2212_221276


namespace NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l2212_221223

/-- Given a square cut into four equal rectangles arranged to form a 'P' shape with a perimeter of 56,
    the perimeter of the original square is 74 2/3. -/
theorem square_perimeter_from_p_shape (x : ℚ) : 
  (2 * (4 * x) + 4 * x = 56) →  -- Perimeter of 'P' shape
  (4 * (4 * x) = 74 + 2/3) -- Perimeter of original square
  := by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l2212_221223


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_in_third_sector_l2212_221211

/-- The radius of an inscribed circle in a sector that is one-third of a circle -/
theorem inscribed_circle_radius_in_third_sector (R : ℝ) (h : R = 5) :
  let r := (R * Real.sqrt 3 - R) / 2
  r * (1 + Real.sqrt 3) = R :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_in_third_sector_l2212_221211


namespace NUMINAMATH_CALUDE_max_consecutive_odd_exponents_is_seven_l2212_221212

/-- A natural number has odd prime factor exponents if all exponents in its prime factorization are odd. -/
def has_odd_prime_factor_exponents (n : ℕ) : Prop :=
  ∀ p k, p.Prime → p ^ k ∣ n → k % 2 = 1

/-- The maximum number of consecutive natural numbers with odd prime factor exponents. -/
def max_consecutive_odd_exponents : ℕ := 7

/-- Theorem stating that the maximum number of consecutive natural numbers 
    with odd prime factor exponents is 7. -/
theorem max_consecutive_odd_exponents_is_seven :
  ∀ n : ℕ, ∃ m ∈ Finset.range 7, ¬(has_odd_prime_factor_exponents (n + m)) ∧
  ∃ k : ℕ, ∀ i ∈ Finset.range 7, has_odd_prime_factor_exponents (k + i) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_odd_exponents_is_seven_l2212_221212


namespace NUMINAMATH_CALUDE_arrangement_count_l2212_221208

/-- The number of ways to arrange 15 letters (5 A's, 5 B's, 5 C's) with restrictions -/
def restricted_arrangements : ℕ :=
  Finset.sum (Finset.range 6) (fun k => (Nat.choose 5 k) ^ 3)

/-- The conditions of the problem -/
theorem arrangement_count :
  restricted_arrangements =
    (Finset.sum (Finset.range 6) (fun k =>
      /- Number of ways to arrange k A's and (5-k) C's in the first 5 positions -/
      (Nat.choose 5 k) *
      /- Number of ways to arrange k B's and (5-k) A's in the middle 5 positions -/
      (Nat.choose 5 k) *
      /- Number of ways to arrange k C's and (5-k) B's in the last 5 positions -/
      (Nat.choose 5 k))) :=
by
  sorry

#eval restricted_arrangements

end NUMINAMATH_CALUDE_arrangement_count_l2212_221208


namespace NUMINAMATH_CALUDE_yellow_score_mixture_l2212_221297

theorem yellow_score_mixture (white_ratio : ℕ) (black_ratio : ℕ) (total_yellow : ℕ) 
  (h1 : white_ratio = 7)
  (h2 : black_ratio = 6)
  (h3 : total_yellow = 78) :
  (white_ratio * (total_yellow / (white_ratio + black_ratio)) - 
   black_ratio * (total_yellow / (white_ratio + black_ratio))) / total_yellow = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_yellow_score_mixture_l2212_221297


namespace NUMINAMATH_CALUDE_triangle_max_area_l2212_221259

open Real

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + c = 6 →
  (3 - cos A) * tan (B / 2) = sin A →
  ∃ (S : ℝ), S ≤ 2 * sqrt 2 ∧
    ∀ (S' : ℝ), S' = (1 / 2) * a * c * sin B → S' ≤ S :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2212_221259


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l2212_221225

theorem trigonometric_expression_equality : 
  (2 * Real.sin (25 * π / 180) ^ 2 - 1) / (Real.sin (20 * π / 180) * Real.cos (20 * π / 180)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l2212_221225


namespace NUMINAMATH_CALUDE_inner_square_area_ratio_is_one_fourth_l2212_221227

/-- The ratio of the area of a square formed by connecting the center of a larger square
    to the midpoints of its sides, to the area of the larger square. -/
def inner_square_area_ratio : ℚ := 1 / 4

/-- Theorem stating that the ratio of the area of the inner square to the outer square is 1/4. -/
theorem inner_square_area_ratio_is_one_fourth :
  inner_square_area_ratio = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inner_square_area_ratio_is_one_fourth_l2212_221227


namespace NUMINAMATH_CALUDE_scientific_notation_of_population_l2212_221243

theorem scientific_notation_of_population (population : ℝ) : 
  population = 2184.3 * 1000000 → 
  ∃ (a : ℝ) (n : ℤ), population = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.1843 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_population_l2212_221243


namespace NUMINAMATH_CALUDE_polynomial_value_l2212_221255

def star (x y : ℤ) : ℤ := (x + 1) * (y + 1)

def star_square (x : ℤ) : ℤ := star x x

theorem polynomial_value : 
  let x := 2
  3 * (star_square x) - 2 * x + 1 = 32 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_l2212_221255


namespace NUMINAMATH_CALUDE_weighted_arithmetic_geometric_mean_inequality_l2212_221288

theorem weighted_arithmetic_geometric_mean_inequality 
  {n : ℕ} (a w : Fin n → ℝ) (h_pos_a : ∀ i, a i > 0) (h_pos_w : ∀ i, w i > 0) :
  let W := (Finset.univ.sum w)
  (W⁻¹ * Finset.univ.sum (λ i => w i * a i)) ≥ 
    (Finset.univ.prod (λ i => (a i) ^ (w i))) ^ (W⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_weighted_arithmetic_geometric_mean_inequality_l2212_221288


namespace NUMINAMATH_CALUDE_probability_three_primes_and_at_least_one_eight_l2212_221290

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Bool := sorry

/-- The set of prime numbers on an 8-sided die -/
def primesOnDie : Finset ℕ := {2, 3, 5, 7}

/-- The probability of rolling a prime number on a single 8-sided die -/
def probPrime : ℚ := (Finset.card primesOnDie : ℚ) / 8

/-- The probability of rolling an 8 on a single 8-sided die -/
def probEight : ℚ := 1 / 8

/-- The number of ways to choose 3 dice out of 6 -/
def chooseThreeOutOfSix : ℕ := Nat.choose 6 3

theorem probability_three_primes_and_at_least_one_eight :
  let probExactlyThreePrimes := chooseThreeOutOfSix * probPrime^3 * (1 - probPrime)^3
  let probAtLeastOneEight := 1 - (1 - probEight)^6
  probExactlyThreePrimes * probAtLeastOneEight = 2899900 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_primes_and_at_least_one_eight_l2212_221290


namespace NUMINAMATH_CALUDE_intersection_sum_l2212_221294

def M : Set ℝ := {x | x^2 - 4*x < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < 5}

theorem intersection_sum (m n : ℝ) : 
  M ∩ N m = {x | 3 < x ∧ x < n} → m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2212_221294


namespace NUMINAMATH_CALUDE_total_turnips_l2212_221275

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139) 
  (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l2212_221275


namespace NUMINAMATH_CALUDE_linear_inequalities_solution_sets_l2212_221284

theorem linear_inequalities_solution_sets :
  (∀ x : ℝ, (4 * (x + 1) ≤ 7 * x + 10 ∧ x - 5 < (x - 8) / 3) ↔ (-2 ≤ x ∧ x < 7 / 2)) ∧
  (∀ x : ℝ, (x - 3 * (x - 2) ≥ 4 ∧ (2 * x - 1) / 5 ≥ (x + 1) / 2) ↔ x ≤ -7) :=
by sorry

end NUMINAMATH_CALUDE_linear_inequalities_solution_sets_l2212_221284


namespace NUMINAMATH_CALUDE_min_abs_sum_l2212_221252

open Complex

variable (α γ : ℂ)

def f (z : ℂ) : ℂ := (2 + 3*I)*z^2 + α*z + γ

theorem min_abs_sum (h1 : (f α γ 1).im = 0) (h2 : (f α γ I).im = 0) :
  ∃ (α₀ γ₀ : ℂ), (abs α₀ + abs γ₀ = 3) ∧ 
    ∀ (α' γ' : ℂ), (f α' γ' 1).im = 0 → (f α' γ' I).im = 0 → abs α' + abs γ' ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_abs_sum_l2212_221252


namespace NUMINAMATH_CALUDE_women_in_luxury_suite_l2212_221234

def total_passengers : ℕ := 300
def women_percentage : ℚ := 1/2
def luxury_suite_percentage : ℚ := 3/20

theorem women_in_luxury_suite :
  ⌊(total_passengers : ℚ) * women_percentage * luxury_suite_percentage⌋ = 23 := by
  sorry

end NUMINAMATH_CALUDE_women_in_luxury_suite_l2212_221234


namespace NUMINAMATH_CALUDE_tim_younger_than_jenny_l2212_221231

/-- Given the ages of Tim, Rommel, and Jenny, prove that Tim is 12 years younger than Jenny. -/
theorem tim_younger_than_jenny (tim_age rommel_age jenny_age : ℕ) : 
  tim_age = 5 →
  rommel_age = 3 * tim_age →
  jenny_age = rommel_age + 2 →
  jenny_age - tim_age = 12 := by
sorry

end NUMINAMATH_CALUDE_tim_younger_than_jenny_l2212_221231


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l2212_221202

theorem geometric_sequence_value (a : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*a + 2) = r * a ∧ (3*a + 3) = r * (2*a + 2)) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_value_l2212_221202


namespace NUMINAMATH_CALUDE_wind_velocity_calculation_l2212_221203

/-- The relationship between pressure, area, and velocity -/
def pressure_relationship (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^3

/-- The theorem to prove -/
theorem wind_velocity_calculation (k : ℝ) :
  pressure_relationship k 2 8 = 4 →
  pressure_relationship k 4 12.8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_wind_velocity_calculation_l2212_221203


namespace NUMINAMATH_CALUDE_original_price_after_changes_l2212_221269

/-- Given an item with original price x, increased by q% and then reduced by r%,
    resulting in a final price of 2 dollars, prove that the original price x
    is equal to 20000 / (10000 + 100 * (q - r) - q * r) -/
theorem original_price_after_changes (q r : ℝ) (x : ℝ) 
    (h1 : x * (1 + q / 100) * (1 - r / 100) = 2) :
  x = 20000 / (10000 + 100 * (q - r) - q * r) := by
  sorry


end NUMINAMATH_CALUDE_original_price_after_changes_l2212_221269


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l2212_221237

/-- The amount of material Cheryl used for her project -/
def material_used (material1 material2 leftover : ℚ) : ℚ :=
  material1 + material2 - leftover

/-- Theorem stating the total amount of material Cheryl used -/
theorem cheryl_material_usage :
  let material1 : ℚ := 5 / 11
  let material2 : ℚ := 2 / 3
  let leftover : ℚ := 25 / 55
  material_used material1 material2 leftover = 22 / 33 := by
sorry

#eval material_used (5/11) (2/3) (25/55)

end NUMINAMATH_CALUDE_cheryl_material_usage_l2212_221237


namespace NUMINAMATH_CALUDE_f_geq_6_iff_l2212_221213

def f (x : ℝ) := |x + 1| + |2*x - 4|

theorem f_geq_6_iff (x : ℝ) : f x ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_f_geq_6_iff_l2212_221213


namespace NUMINAMATH_CALUDE_cost_of_tea_cake_eclair_l2212_221249

/-- Given the costs of tea and a cake, tea and an éclair, and a cake and an éclair,
    prove that the sum of the costs of tea, a cake, and an éclair
    is equal to half the sum of all three given costs. -/
theorem cost_of_tea_cake_eclair
  (t c e : ℝ)  -- t: cost of tea, c: cost of cake, e: cost of éclair
  (h1 : t + c = 4.5)  -- cost of tea and cake
  (h2 : t + e = 4)    -- cost of tea and éclair
  (h3 : c + e = 6.5)  -- cost of cake and éclair
  : t + c + e = (4.5 + 4 + 6.5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_tea_cake_eclair_l2212_221249


namespace NUMINAMATH_CALUDE_justin_bought_two_striped_jerseys_l2212_221299

-- Define the cost of each type of jersey
def long_sleeved_cost : ℕ := 15
def striped_cost : ℕ := 10

-- Define the number of long-sleeved jerseys bought
def long_sleeved_count : ℕ := 4

-- Define the total amount spent
def total_spent : ℕ := 80

-- Define the number of striped jerseys as a function
def striped_jerseys : ℕ := (total_spent - long_sleeved_cost * long_sleeved_count) / striped_cost

-- Theorem to prove
theorem justin_bought_two_striped_jerseys : striped_jerseys = 2 := by
  sorry

end NUMINAMATH_CALUDE_justin_bought_two_striped_jerseys_l2212_221299


namespace NUMINAMATH_CALUDE_equation_solution_exists_l2212_221283

theorem equation_solution_exists : ∃ x : ℝ, 
  (x^3 - (0.1)^3) / (x^2 + 0.066 + (0.1)^2) = 0.5599999999999999 ∧ 
  abs (x - 0.8) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l2212_221283


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l2212_221295

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0 → x + y + z = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l2212_221295


namespace NUMINAMATH_CALUDE_smallest_k_for_tangent_circle_l2212_221219

theorem smallest_k_for_tangent_circle : ∃ (h : ℕ+), 
  (1 - h.val)^2 + (1000 + 58 - h.val)^2 = h.val^2 ∧
  ∀ (k : ℕ), k < 58 → ¬∃ (h : ℕ+), (1 - h.val)^2 + (1000 + k - h.val)^2 = h.val^2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_tangent_circle_l2212_221219


namespace NUMINAMATH_CALUDE_initial_kittens_count_l2212_221214

/-- The number of kittens Alyssa's cat initially had -/
def initial_kittens : ℕ := sorry

/-- The number of kittens Alyssa gave to her friends -/
def given_away : ℕ := 4

/-- The number of kittens Alyssa has left -/
def kittens_left : ℕ := 4

/-- Theorem stating that the initial number of kittens is 8 -/
theorem initial_kittens_count : initial_kittens = 8 :=
by sorry

end NUMINAMATH_CALUDE_initial_kittens_count_l2212_221214


namespace NUMINAMATH_CALUDE_expand_polynomial_l2212_221268

theorem expand_polynomial (x y : ℝ) : 
  (1 + x^2 + y^3) * (1 - x^3 - y^3) = 1 + x^2 - x^3 - y^3 - x^5 - x^2 * y^3 - x^3 * y^3 - y^6 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l2212_221268


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2212_221205

def coin_flips : ℕ := 12
def desired_heads : ℕ := 9

theorem probability_nine_heads_in_twelve_flips :
  (Nat.choose coin_flips desired_heads : ℚ) / (2 ^ coin_flips) = 220 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2212_221205


namespace NUMINAMATH_CALUDE_x_zero_sufficient_not_necessary_l2212_221254

theorem x_zero_sufficient_not_necessary :
  (∃ x : ℝ, x = 0 → x^2 - 2*x = 0) ∧
  (∃ x : ℝ, x^2 - 2*x = 0 ∧ x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_zero_sufficient_not_necessary_l2212_221254


namespace NUMINAMATH_CALUDE_swiss_cheese_probability_l2212_221239

theorem swiss_cheese_probability :
  let cheddar : ℕ := 22
  let mozzarella : ℕ := 34
  let pepperjack : ℕ := 29
  let swiss : ℕ := 45
  let gouda : ℕ := 20
  let total : ℕ := cheddar + mozzarella + pepperjack + swiss + gouda
  (swiss : ℚ) / (total : ℚ) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_swiss_cheese_probability_l2212_221239


namespace NUMINAMATH_CALUDE_optimal_layoffs_maximizes_benefit_l2212_221206

/-- Represents the number of employees to lay off for maximum economic benefit -/
def optimal_layoffs (a : ℕ) : ℚ :=
  if 70 < a ∧ a ≤ 140 then a - 70
  else if 140 < a ∧ a < 210 then a / 2
  else 0

theorem optimal_layoffs_maximizes_benefit (a b : ℕ) :
  140 < 2 * a ∧ 2 * a < 420 ∧ 
  ∃ k, a = 2 * k ∧
  (∀ x : ℚ, 0 < x ∧ x ≤ a / 2 →
    ((2 * a - x) * (b + 0.01 * b * x) - 0.4 * b * x) ≤
    ((2 * a - optimal_layoffs a) * (b + 0.01 * b * optimal_layoffs a) - 0.4 * b * optimal_layoffs a)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_layoffs_maximizes_benefit_l2212_221206


namespace NUMINAMATH_CALUDE_correct_statements_l2212_221238

-- Define the data sets
def dataSetA : List ℝ := sorry
def dataSetB : List ℝ := sorry
def dataSetC : List ℝ := [1, 2, 5, 5, 5, 3, 3]

-- Define variance function
def variance (data : List ℝ) : ℝ := sorry

-- Define median function
def median (data : List ℝ) : ℝ := sorry

-- Define mode function
def mode (data : List ℝ) : ℝ := sorry

-- Theorem to prove
theorem correct_statements :
  -- Statement A is incorrect
  ¬ (∀ (n : ℕ), n = 1000 → ∃ (h : ℕ), h = 500 ∧ h = n / 2) ∧
  -- Statement B is correct
  (variance dataSetA = 0.03 ∧ variance dataSetB = 0.1 → variance dataSetA < variance dataSetB) ∧
  -- Statement C is correct
  (median dataSetC = 3 ∧ mode dataSetC = 5) ∧
  -- Statement D is incorrect
  ¬ (∀ (population : Type) (property : population → Prop),
     (∀ x : population, property x) ↔ (∃ survey : population → Prop, ∀ x, survey x → property x))
  := by sorry

end NUMINAMATH_CALUDE_correct_statements_l2212_221238


namespace NUMINAMATH_CALUDE_roll_one_probability_l2212_221277

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Fin 6)

/-- The probability of rolling a specific number on a fair six-sided die -/
def roll_probability (d : FairDie) (n : Fin 6) : ℚ := 1 / 6

/-- The independence of die rolls -/
axiom roll_independence (d : FairDie) (n m : Fin 6) : 
  roll_probability d n = roll_probability d n

/-- Theorem: The probability of rolling a 1 on a fair six-sided die is 1/6 -/
theorem roll_one_probability (d : FairDie) : 
  roll_probability d 0 = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_roll_one_probability_l2212_221277


namespace NUMINAMATH_CALUDE_number_equals_1038_l2212_221207

theorem number_equals_1038 : ∃ n : ℝ, n * 40 = 173 * 240 ∧ n = 1038 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_1038_l2212_221207


namespace NUMINAMATH_CALUDE_prime_divides_sum_of_powers_l2212_221204

theorem prime_divides_sum_of_powers (p : ℕ) (hp : Prime p) :
  ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_sum_of_powers_l2212_221204


namespace NUMINAMATH_CALUDE_sine_graph_transformation_l2212_221240

theorem sine_graph_transformation (x : ℝ) : 
  4 * Real.sin (2 * x + π / 5) = 4 * Real.sin ((2 * x / 2) + π / 5) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_transformation_l2212_221240


namespace NUMINAMATH_CALUDE_bible_length_l2212_221245

/-- The number of pages in John's bible --/
def bible_pages : ℕ := sorry

/-- The number of hours John reads per day --/
def hours_per_day : ℕ := 2

/-- The number of pages John reads per hour --/
def pages_per_hour : ℕ := 50

/-- The number of weeks it takes John to read the entire bible --/
def weeks_to_read : ℕ := 4

/-- The number of days in a week --/
def days_per_week : ℕ := 7

theorem bible_length : bible_pages = 2800 := by sorry

end NUMINAMATH_CALUDE_bible_length_l2212_221245
