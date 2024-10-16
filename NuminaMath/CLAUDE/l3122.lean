import Mathlib

namespace NUMINAMATH_CALUDE_regular_polygon_150_degrees_l3122_312266

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides. -/
theorem regular_polygon_150_degrees : 
  ∀ n : ℕ, 
  n > 2 → 
  (180 * (n - 2) : ℝ) = 150 * n → 
  n = 12 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degrees_l3122_312266


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_arithmetic_progression_l3122_312248

/-- Given a triangle with sides a, b, c forming an arithmetic progression,
    prove that the radius of the inscribed circle is one-third of the altitude to side b. -/
theorem inscribed_circle_radius_arithmetic_progression
  (a b c : ℝ) (h_order : a ≤ b ∧ b ≤ c) (h_arithmetic : 2 * b = a + c)
  (r : ℝ) (h_b : ℝ) (S : ℝ) 
  (h_area_inradius : 2 * S = r * (a + b + c))
  (h_area_altitude : 2 * S = h_b * b) :
  r = h_b / 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_arithmetic_progression_l3122_312248


namespace NUMINAMATH_CALUDE_largest_multiple_of_nine_less_than_hundred_l3122_312247

theorem largest_multiple_of_nine_less_than_hundred : 
  ∃ (n : ℕ), n * 9 = 99 ∧ 
  ∀ (m : ℕ), m * 9 < 100 → m * 9 ≤ 99 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_nine_less_than_hundred_l3122_312247


namespace NUMINAMATH_CALUDE_smartphone_savings_smartphone_savings_proof_l3122_312213

/-- Calculates the weekly savings required to purchase a smartphone --/
theorem smartphone_savings (smartphone_cost current_savings : ℚ) : ℚ :=
  let remaining_amount := smartphone_cost - current_savings
  let weeks_in_two_months := 2 * (52 / 12 : ℚ)
  remaining_amount / weeks_in_two_months

/-- Proves that the weekly savings for the given scenario is approximately $13.86 --/
theorem smartphone_savings_proof :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |smartphone_savings 160 40 - (13.86 : ℚ)| < ε :=
sorry

end NUMINAMATH_CALUDE_smartphone_savings_smartphone_savings_proof_l3122_312213


namespace NUMINAMATH_CALUDE_exists_divisible_pair_l3122_312278

/-- A function that checks if a number is three-digit -/
def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

/-- A function that checks if a number is two-digit -/
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

/-- A function that checks if a number uses only the digits 1, 2, 3, 4, 5 -/
def usesOnlyGivenDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [1, 2, 3, 4, 5]

/-- The main theorem -/
theorem exists_divisible_pair :
  ∃ (a b : ℕ),
    isThreeDigit a ∧
    isTwoDigit b ∧
    usesOnlyGivenDigits a ∧
    usesOnlyGivenDigits b ∧
    a % b = 0 :=
  sorry

end NUMINAMATH_CALUDE_exists_divisible_pair_l3122_312278


namespace NUMINAMATH_CALUDE_mersenne_coprime_iff_exponents_coprime_l3122_312212

theorem mersenne_coprime_iff_exponents_coprime (p q : ℕ+) :
  Nat.gcd ((2 : ℕ) ^ p.val - 1) ((2 : ℕ) ^ q.val - 1) = 1 ↔ Nat.gcd p.val q.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_mersenne_coprime_iff_exponents_coprime_l3122_312212


namespace NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l3122_312243

theorem proposition_p_sufficient_not_necessary_for_q :
  (∀ a b : ℝ, a * b ≠ 0 → a^2 + b^2 ≠ 0) ∧
  (∃ a b : ℝ, a^2 + b^2 ≠ 0 ∧ a * b = 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l3122_312243


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l3122_312286

theorem quadratic_form_ratio (j : ℝ) : ∃ (c p q : ℝ),
  (6 * j^2 - 4 * j + 12 = c * (j + p)^2 + q) ∧ (q / p = -34) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l3122_312286


namespace NUMINAMATH_CALUDE_eBook_readers_count_l3122_312258

/-- The number of eBook readers Anna bought -/
def anna_readers : ℕ := 50

/-- The number of eBook readers John bought initially -/
def john_initial_readers : ℕ := anna_readers - 15

/-- The number of eBook readers John lost -/
def john_lost_readers : ℕ := 3

/-- The number of eBook readers John has after losing some -/
def john_final_readers : ℕ := john_initial_readers - john_lost_readers

/-- The total number of eBook readers John and Anna have together -/
def total_readers : ℕ := anna_readers + john_final_readers

theorem eBook_readers_count : total_readers = 82 := by
  sorry

end NUMINAMATH_CALUDE_eBook_readers_count_l3122_312258


namespace NUMINAMATH_CALUDE_cube_inscribed_in_sphere_surface_area_l3122_312293

/-- The surface area of a cube inscribed in a sphere with radius 5 units is 200 square units. -/
theorem cube_inscribed_in_sphere_surface_area :
  let r : ℝ := 5  -- radius of the sphere
  let s : ℝ := 10 * Real.sqrt 3 / 3  -- edge length of the cube
  6 * s^2 = 200 := by sorry

end NUMINAMATH_CALUDE_cube_inscribed_in_sphere_surface_area_l3122_312293


namespace NUMINAMATH_CALUDE_total_marbles_count_l3122_312269

/-- Represents the number of marbles of each color in the container -/
structure MarbleCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- The ratio of marbles in the container -/
def marbleRatio : MarbleCount := ⟨2, 3, 4⟩

/-- The actual number of yellow marbles in the container -/
def yellowMarbleCount : ℕ := 40

/-- Theorem stating that the total number of marbles is 90 -/
theorem total_marbles_count :
  let total := marbleRatio.red + marbleRatio.blue + marbleRatio.yellow
  (yellowMarbleCount * total) / marbleRatio.yellow = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_count_l3122_312269


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3122_312252

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

-- Define the theorem
theorem possible_values_of_a :
  ∀ a : ℝ, (M ∩ N a = N a) → (a = -1 ∨ a = 0 ∨ a = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3122_312252


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3122_312292

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + 2 * Complex.I) / (1 + 2 * Complex.I) = b * Complex.I) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3122_312292


namespace NUMINAMATH_CALUDE_magic_card_price_l3122_312208

/-- The initial price of a Magic card that triples in value and results in a $200 profit when sold -/
def initial_price : ℝ := 100

/-- The value of the card after tripling -/
def tripled_value (price : ℝ) : ℝ := 3 * price

/-- The profit made from selling the card -/
def profit (initial_price : ℝ) : ℝ := tripled_value initial_price - initial_price

theorem magic_card_price :
  profit initial_price = 200 ∧ tripled_value initial_price = 3 * initial_price := by
  sorry

end NUMINAMATH_CALUDE_magic_card_price_l3122_312208


namespace NUMINAMATH_CALUDE_student_number_problem_l3122_312246

theorem student_number_problem (x : ℝ) : 2 * x - 148 = 110 → x = 129 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3122_312246


namespace NUMINAMATH_CALUDE_projection_of_v_onto_Q_l3122_312217

/-- The plane Q is defined by the equation x + 2y - z = 0 -/
def Q : Set (Fin 3 → ℝ) :=
  {v | v 0 + 2 * v 1 - v 2 = 0}

/-- The vector v -/
def v : Fin 3 → ℝ := ![2, 3, 1]

/-- The projection of v onto Q -/
def projection (v : Fin 3 → ℝ) (Q : Set (Fin 3 → ℝ)) : Fin 3 → ℝ :=
  sorry

theorem projection_of_v_onto_Q :
  projection v Q = ![5/6, 4/6, 13/6] := by sorry

end NUMINAMATH_CALUDE_projection_of_v_onto_Q_l3122_312217


namespace NUMINAMATH_CALUDE_complex_modulus_inequality_l3122_312250

theorem complex_modulus_inequality (x y : ℝ) : 
  let z : ℂ := Complex.mk x y
  ‖z‖ ≤ |x| + |y| := by sorry

end NUMINAMATH_CALUDE_complex_modulus_inequality_l3122_312250


namespace NUMINAMATH_CALUDE_no_integer_solution_l3122_312263

theorem no_integer_solution : ∀ m n : ℤ, m^2 ≠ n^5 - 4 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3122_312263


namespace NUMINAMATH_CALUDE_soft_drink_cost_l3122_312220

/-- The cost of a soft drink given the total spent and the cost of candy bars. -/
theorem soft_drink_cost (total_spent : ℕ) (candy_bars : ℕ) (candy_bar_cost : ℕ) (soft_drink_cost : ℕ) : 
  total_spent = 27 ∧ candy_bars = 5 ∧ candy_bar_cost = 5 → soft_drink_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_soft_drink_cost_l3122_312220


namespace NUMINAMATH_CALUDE_opposite_of_three_l3122_312264

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

theorem opposite_of_three : opposite 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3122_312264


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3122_312289

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    if the length of the minor axis is equal to the focal length,
    then the eccentricity of the ellipse is √2/2 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let c := Real.sqrt (a^2 - b^2)
  2 * b = 2 * c → Real.sqrt ((a^2 - b^2) / a^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3122_312289


namespace NUMINAMATH_CALUDE_expression_evaluation_l3122_312206

theorem expression_evaluation :
  let x : ℝ := -1
  (x - 2)^2 - (2*x + 3)*(2*x - 3) - 4*x*(x - 1) = 6 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3122_312206


namespace NUMINAMATH_CALUDE_product_sum_relation_l3122_312238

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 10 → b = 9 → b - a = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l3122_312238


namespace NUMINAMATH_CALUDE_min_value_theorem_l3122_312226

theorem min_value_theorem (m n p x y z : ℝ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0)
  (hmnp : m * n * p = 8)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxyz : x * y * z = 8) :
  ∃ (min : ℝ), min = 12 + 4 * (m + n + p) ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a * b * c = 8 →
    a^2 + b^2 + c^2 + m*a*b + n*a*c + p*b*c ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3122_312226


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3122_312273

/-- An arithmetic sequence with the given properties has a common difference of 2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- The arithmetic sequence
  (h1 : a 1 + a 5 = 10) -- First condition: a_1 + a_5 = 10
  (h2 : a 4 = 7) -- Second condition: a_4 = 7
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) -- Definition of arithmetic sequence
  : a 2 - a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3122_312273


namespace NUMINAMATH_CALUDE_investment_return_calculation_l3122_312272

theorem investment_return_calculation (total_investment small_investment large_investment : ℝ)
  (combined_return_rate small_return_rate : ℝ) :
  total_investment = small_investment + large_investment →
  small_investment = 500 →
  large_investment = 1500 →
  combined_return_rate = 0.085 →
  small_return_rate = 0.07 →
  (combined_return_rate * total_investment = small_return_rate * small_investment + 
    (large_investment * 0.09)) :=
by sorry

end NUMINAMATH_CALUDE_investment_return_calculation_l3122_312272


namespace NUMINAMATH_CALUDE_max_brownies_is_294_l3122_312201

/-- Represents the dimensions of a rectangular pan of brownies -/
structure BrowniePan where
  m : ℕ  -- length
  n : ℕ  -- width

/-- Calculates the number of interior pieces in a brownie pan -/
def interiorPieces (pan : BrowniePan) : ℕ :=
  (pan.m - 2) * (pan.n - 2)

/-- Calculates the number of perimeter pieces in a brownie pan -/
def perimeterPieces (pan : BrowniePan) : ℕ :=
  2 * pan.m + 2 * pan.n - 4

/-- Checks if the interior pieces are twice the perimeter pieces -/
def validCutting (pan : BrowniePan) : Prop :=
  interiorPieces pan = 2 * perimeterPieces pan

/-- Calculates the total number of brownies in a pan -/
def totalBrownies (pan : BrowniePan) : ℕ :=
  pan.m * pan.n

/-- Theorem: The maximum number of brownies is 294 given the conditions -/
theorem max_brownies_is_294 :
  ∃ (pan : BrowniePan), validCutting pan ∧
    (∀ (other : BrowniePan), validCutting other → totalBrownies other ≤ totalBrownies pan) ∧
    totalBrownies pan = 294 :=
  sorry

end NUMINAMATH_CALUDE_max_brownies_is_294_l3122_312201


namespace NUMINAMATH_CALUDE_complex_magnitude_l3122_312268

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3122_312268


namespace NUMINAMATH_CALUDE_johns_order_cost_l3122_312219

/-- Calculates the discounted price of an order given the store's discount policy and purchase details. -/
def discountedPrice (itemPrice : ℕ) (itemCount : ℕ) (discountThreshold : ℕ) (discountRate : ℚ) : ℚ :=
  let totalPrice := itemPrice * itemCount
  let discountableAmount := max (totalPrice - discountThreshold) 0
  let discount := (discountableAmount : ℚ) * discountRate
  (totalPrice : ℚ) - discount

/-- Theorem stating that John's order costs $1360 after the discount. -/
theorem johns_order_cost :
  discountedPrice 200 7 1000 (1 / 10) = 1360 := by
  sorry

end NUMINAMATH_CALUDE_johns_order_cost_l3122_312219


namespace NUMINAMATH_CALUDE_expected_total_cost_is_350_l3122_312251

/-- Represents the outcome of a single test -/
inductive TestResult
| Defective
| NonDefective

/-- Represents the possible total costs of the testing process -/
inductive TotalCost
| Cost200
| Cost300
| Cost400

/-- The probability of getting a specific test result -/
def testProbability (result : TestResult) : ℚ :=
  match result with
  | TestResult.Defective => 2/5
  | TestResult.NonDefective => 3/5

/-- The probability of getting a specific total cost -/
def costProbability (cost : TotalCost) : ℚ :=
  match cost with
  | TotalCost.Cost200 => 1/10
  | TotalCost.Cost300 => 3/10
  | TotalCost.Cost400 => 3/5

/-- The cost in yuan for a specific total cost outcome -/
def costValue (cost : TotalCost) : ℚ :=
  match cost with
  | TotalCost.Cost200 => 200
  | TotalCost.Cost300 => 300
  | TotalCost.Cost400 => 400

/-- The expected value of the total cost -/
def expectedTotalCost : ℚ :=
  (costValue TotalCost.Cost200 * costProbability TotalCost.Cost200) +
  (costValue TotalCost.Cost300 * costProbability TotalCost.Cost300) +
  (costValue TotalCost.Cost400 * costProbability TotalCost.Cost400)

theorem expected_total_cost_is_350 :
  expectedTotalCost = 350 := by sorry

end NUMINAMATH_CALUDE_expected_total_cost_is_350_l3122_312251


namespace NUMINAMATH_CALUDE_find_number_l3122_312257

theorem find_number : ∃ x : ℝ, (0.38 * 80) - (0.12 * x) = 11.2 ∧ x = 160 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3122_312257


namespace NUMINAMATH_CALUDE_prime_sum_of_powers_l3122_312245

theorem prime_sum_of_powers (a b p : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime p → p = a^b + b^a → p = 17 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_of_powers_l3122_312245


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3122_312296

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

-- Theorem statement
theorem min_value_of_sum (a : ℝ) :
  (∃ x, x = 2 ∧ f_derivative a x = 0) →
  (∀ m n, m ∈ Set.Icc (-1 : ℝ) 1 → n ∈ Set.Icc (-1 : ℝ) 1 →
    f a m + f_derivative a n ≥ -13) ∧
  (∃ m n, m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f_derivative a n = -13) :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_sum_l3122_312296


namespace NUMINAMATH_CALUDE_john_incentive_calculation_l3122_312221

/-- The incentive calculation for John's agency fees --/
theorem john_incentive_calculation 
  (commission : ℕ) 
  (advance_fees : ℕ) 
  (amount_given : ℕ) 
  (h1 : commission = 25000)
  (h2 : advance_fees = 8280)
  (h3 : amount_given = 18500) :
  amount_given - (commission - advance_fees) = 1780 :=
by sorry

end NUMINAMATH_CALUDE_john_incentive_calculation_l3122_312221


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3122_312281

-- Define the polynomial g(x)
def g (c d x : ℝ) : ℝ := c * x^3 - 7 * x^2 + d * x - 4

-- State the theorem
theorem polynomial_remainder_theorem (c d : ℝ) :
  (g c d 2 = -4) ∧ (g c d (-1) = -22) → c = 19/3 ∧ d = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3122_312281


namespace NUMINAMATH_CALUDE_function_max_value_l3122_312285

open Real

theorem function_max_value (a : ℝ) (h1 : a > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.log x - a * x
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≤ -4) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f x = -4) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_max_value_l3122_312285


namespace NUMINAMATH_CALUDE_sum_calculation_l3122_312261

theorem sum_calculation : 3 * 501 + 2 * 501 + 4 * 501 + 500 = 5009 := by
  sorry

end NUMINAMATH_CALUDE_sum_calculation_l3122_312261


namespace NUMINAMATH_CALUDE_work_left_fraction_l3122_312234

theorem work_left_fraction (days_A days_B days_together : ℕ) (h1 : days_A = 20) (h2 : days_B = 30) (h3 : days_together = 4) :
  1 - (days_together : ℚ) * (1 / days_A + 1 / days_B) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_work_left_fraction_l3122_312234


namespace NUMINAMATH_CALUDE_distance_center_to_origin_l3122_312236

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the center of the circle
def center_C : ℝ × ℝ := (1, 0)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem distance_center_to_origin :
  Real.sqrt ((center_C.1 - origin.1)^2 + (center_C.2 - origin.2)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_center_to_origin_l3122_312236


namespace NUMINAMATH_CALUDE_smallest_k_for_800_digit_sum_l3122_312262

/-- Represents a number consisting of k digits of 7 -/
def seventySevenNumber (k : ℕ) : ℕ :=
  (7 * (10^k - 1)) / 9

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved -/
theorem smallest_k_for_800_digit_sum :
  ∃ k : ℕ, k = 88 ∧
  (∀ m : ℕ, m < k → sumOfDigits (5 * seventySevenNumber m) < 800) ∧
  sumOfDigits (5 * seventySevenNumber k) = 800 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_800_digit_sum_l3122_312262


namespace NUMINAMATH_CALUDE_area_relationship_l3122_312239

/-- The area of an isosceles triangle with sides 17, 17, and 16. -/
def P : ℝ := sorry

/-- The area of a right triangle with legs 15 and 20. -/
def Q : ℝ := sorry

/-- Theorem stating the relationship between P and Q. -/
theorem area_relationship : P = (4/5) * Q := by sorry

end NUMINAMATH_CALUDE_area_relationship_l3122_312239


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3122_312209

theorem rectangle_dimension_change (b h : ℝ) (h_pos : 0 < h) (b_pos : 0 < b) :
  let new_base := 1.1 * b
  let new_height := h * (new_base * h) / (b * h) / new_base
  (h - new_height) / h = 1 / 11 := by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3122_312209


namespace NUMINAMATH_CALUDE_root_difference_of_equation_l3122_312260

theorem root_difference_of_equation : ∃ a b : ℝ,
  (∀ x : ℝ, (6 * x - 18) / (x^2 + 3 * x - 28) = x + 3 ↔ x = a ∨ x = b) ∧
  a > b ∧
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_root_difference_of_equation_l3122_312260


namespace NUMINAMATH_CALUDE_max_value_constraint_l3122_312215

theorem max_value_constraint (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 5 + 9 * y * z ≤ Real.sqrt 86 := by
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3122_312215


namespace NUMINAMATH_CALUDE_negative_three_below_zero_l3122_312244

/-- Represents temperature in Celsius -/
structure Temperature where
  value : ℝ
  unit : String

/-- Defines the concept of opposite meanings for temperatures -/
def oppositeMeaning (t1 t2 : Temperature) : Prop :=
  t1.value = -t2.value ∧ t1.unit = t2.unit

/-- Axiom: If two numbers have opposite meanings, they are respectively called positive and negative -/
axiom positive_negative_opposite (t1 t2 : Temperature) :
  oppositeMeaning t1 t2 → (t1.value > 0 ↔ t2.value < 0)

/-- Given: +10°C represents a temperature of 10°C above zero -/
axiom positive_ten_above_zero :
  ∃ (t : Temperature), t.value = 10 ∧ t.unit = "°C"

/-- Theorem: -3°C represents a temperature of 3°C below zero -/
theorem negative_three_below_zero :
  ∃ (t : Temperature), t.value = -3 ∧ t.unit = "°C" ∧
  ∃ (t_pos : Temperature), oppositeMeaning t t_pos ∧ t_pos.value = 3 :=
sorry

end NUMINAMATH_CALUDE_negative_three_below_zero_l3122_312244


namespace NUMINAMATH_CALUDE_equidistant_points_l3122_312204

def equidistant (p q : ℝ × ℝ) : Prop :=
  max (|p.1|) (|p.2|) = max (|q.1|) (|q.2|)

theorem equidistant_points :
  (equidistant (-3, 7) (3, -7) ∧ equidistant (-3, 7) (7, 4)) ∧
  (equidistant (-4, 2) (-4, -3) ∧ equidistant (-4, 2) (3, 4)) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_points_l3122_312204


namespace NUMINAMATH_CALUDE_sum_three_numbers_l3122_312271

/-- Given three numbers a, b, and c, and a value T, satisfying the following conditions:
    1. a + b + c = 84
    2. a - 5 = T
    3. b + 9 = T
    4. 5 * c = T
    Prove that T = 40 -/
theorem sum_three_numbers (a b c T : ℝ) 
  (sum_eq : a + b + c = 84)
  (a_minus : a - 5 = T)
  (b_plus : b + 9 = T)
  (c_times : 5 * c = T) : 
  T = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l3122_312271


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l3122_312287

theorem complex_square_one_plus_i : (1 + Complex.I) ^ 2 = 2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l3122_312287


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3122_312297

theorem complex_fraction_simplification :
  (3 + 5*Complex.I) / (-2 + 3*Complex.I) = 9/13 - 19/13 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3122_312297


namespace NUMINAMATH_CALUDE_a_sufficient_not_necessary_l3122_312232

/-- The function f(x) = x³ + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a

/-- f is strictly increasing on ℝ -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem a_sufficient_not_necessary :
  (∃ a : ℝ, a > 1 ∧ strictly_increasing (f a)) ∧
  (∃ a : ℝ, strictly_increasing (f a) ∧ ¬(a > 1)) :=
sorry

end NUMINAMATH_CALUDE_a_sufficient_not_necessary_l3122_312232


namespace NUMINAMATH_CALUDE_remainder_sum_factorials_25_l3122_312256

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem remainder_sum_factorials_25 :
  (sum_factorials 50) % 25 = (sum_factorials 4) % 25 :=
by sorry

end NUMINAMATH_CALUDE_remainder_sum_factorials_25_l3122_312256


namespace NUMINAMATH_CALUDE_jezebel_roses_l3122_312280

/-- The number of red roses Jezebel needs to buy -/
def num_red_roses : ℕ := sorry

/-- The cost of one red rose in dollars -/
def cost_red_rose : ℚ := 3/2

/-- The number of sunflowers Jezebel needs to buy -/
def num_sunflowers : ℕ := 3

/-- The cost of one sunflower in dollars -/
def cost_sunflower : ℚ := 3

/-- The total cost of all flowers in dollars -/
def total_cost : ℚ := 45

theorem jezebel_roses :
  num_red_roses * cost_red_rose + num_sunflowers * cost_sunflower = total_cost ∧
  num_red_roses = 24 := by sorry

end NUMINAMATH_CALUDE_jezebel_roses_l3122_312280


namespace NUMINAMATH_CALUDE_complex_equation_proof_l3122_312284

theorem complex_equation_proof (z : ℂ) (h : z = 1 - Complex.I) : z^2 - 2*z + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l3122_312284


namespace NUMINAMATH_CALUDE_lowest_divisible_by_primes_10_to_50_l3122_312237

def primes_10_to_50 : List Nat := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def is_divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  ∀ p ∈ list, n % p = 0

theorem lowest_divisible_by_primes_10_to_50 :
  ∃ (n : Nat), n > 0 ∧
  is_divisible_by_all n primes_10_to_50 ∧
  ∀ (m : Nat), m > 0 ∧ is_divisible_by_all m primes_10_to_50 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_lowest_divisible_by_primes_10_to_50_l3122_312237


namespace NUMINAMATH_CALUDE_mikes_net_spending_l3122_312205

/-- The net amount Mike spent at the music store -/
def net_spent (trumpet_cost discount_percent stand_cost sheet_music_cost songbook_sale : ℚ) : ℚ :=
  (trumpet_cost * (1 - discount_percent / 100) + stand_cost + sheet_music_cost) - songbook_sale

/-- Theorem stating that Mike's net spending at the music store is $209.16 -/
theorem mikes_net_spending :
  net_spent 250 30 25 15 5.84 = 209.16 := by
  sorry

end NUMINAMATH_CALUDE_mikes_net_spending_l3122_312205


namespace NUMINAMATH_CALUDE_intersection_range_l3122_312230

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt ((x + 1)^2 + y^2) = 2

-- Define the line l
def l (k x y : ℝ) : Prop :=
  y = k * x + 1 - 2 * k

-- Theorem statement
theorem intersection_range (k : ℝ) :
  (∃ x y, C x y ∧ l k x y) → k ∈ Set.Icc (1/3 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l3122_312230


namespace NUMINAMATH_CALUDE_binomial_coefficient_1450_2_l3122_312295

theorem binomial_coefficient_1450_2 : Nat.choose 1450 2 = 1050205 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1450_2_l3122_312295


namespace NUMINAMATH_CALUDE_total_problems_l3122_312291

def marvin_yesterday : ℕ := 40

def marvin_today (x : ℕ) : ℕ := 3 * x

def arvin_daily (x : ℕ) : ℕ := 2 * x

theorem total_problems :
  marvin_yesterday + marvin_today marvin_yesterday +
  arvin_daily marvin_yesterday + arvin_daily (marvin_today marvin_yesterday) = 480 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_l3122_312291


namespace NUMINAMATH_CALUDE_cubic_factorization_l3122_312259

theorem cubic_factorization (a : ℝ) : a^3 - 25*a = a*(a+5)*(a-5) := by sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3122_312259


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3122_312233

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h_positive : L > 0 ∧ B > 0) :
  B' = 1.15 * B →
  L' * B' = 1.2075 * (L * B) →
  L' = 1.05 * L := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3122_312233


namespace NUMINAMATH_CALUDE_divisibility_of_quadratic_form_l3122_312249

theorem divisibility_of_quadratic_form (p a b k : ℤ) : 
  Prime p → 
  p = 3*k + 2 → 
  p ∣ (a^2 + a*b + b^2) → 
  p ∣ a ∧ p ∣ b :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_quadratic_form_l3122_312249


namespace NUMINAMATH_CALUDE_point_on_line_l3122_312298

theorem point_on_line (m n p : ℝ) : 
  (m = n / 5 - 2 / 5) ∧ (m + p = (n + 15) / 5 - 2 / 5) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l3122_312298


namespace NUMINAMATH_CALUDE_smallest_class_size_l3122_312231

theorem smallest_class_size :
  ∀ n : ℕ,
  (∃ x : ℕ, n = 5 * x + 2 ∧ x > 0) →
  n > 30 →
  n ≥ 32 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l3122_312231


namespace NUMINAMATH_CALUDE_lg_sum_equals_zero_l3122_312200

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_zero : lg 2 + lg 0.5 = 0 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_zero_l3122_312200


namespace NUMINAMATH_CALUDE_function_inequality_condition_l3122_312203

/-- A function f(x) = ax^2 + b satisfies f(xy) + f(x + y) ≥ f(x)f(y) for all real x and y
    if and only if 0 < a < 1, 0 < b ≤ 1, and 2a + b - 2 ≤ 0 -/
theorem function_inequality_condition (a b : ℝ) :
  (∀ x y : ℝ, a * (x * y)^2 + b + a * (x + y)^2 + b ≥ (a * x^2 + b) * (a * y^2 + b)) ↔
  (0 < a ∧ a < 1 ∧ 0 < b ∧ b ≤ 1 ∧ 2 * a + b - 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l3122_312203


namespace NUMINAMATH_CALUDE_equal_derivatives_of_quadratic_functions_l3122_312228

theorem equal_derivatives_of_quadratic_functions :
  let f (x : ℝ) := 1 - 2 * x^2
  let g (x : ℝ) := -2 * x^2 + 3
  ∀ x, deriv f x = deriv g x := by
sorry

end NUMINAMATH_CALUDE_equal_derivatives_of_quadratic_functions_l3122_312228


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l3122_312282

open Real

theorem angle_sum_theorem (x : ℝ) :
  (0 ≤ x ∧ x ≤ 2 * π) →
  (sin x ^ 5 - cos x ^ 5 = 1 / cos x - 1 / sin x) →
  ∃ (y : ℝ), (0 ≤ y ∧ y ≤ 2 * π) ∧
             (sin y ^ 5 - cos y ^ 5 = 1 / cos y - 1 / sin y) ∧
             x + y = 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l3122_312282


namespace NUMINAMATH_CALUDE_ratio_problem_l3122_312288

theorem ratio_problem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_ratio : ∃ (k : ℝ), a = 6*k ∧ b = 3*k ∧ c = k) :
  3 * b^2 / (2 * a^2 + b * c) = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3122_312288


namespace NUMINAMATH_CALUDE_animal_shelter_cats_l3122_312227

theorem animal_shelter_cats (dogs : ℕ) (initial_ratio_dogs initial_ratio_cats : ℕ) 
  (final_ratio_dogs final_ratio_cats : ℕ) (additional_cats : ℕ) : 
  dogs = 75 →
  initial_ratio_dogs = 15 →
  initial_ratio_cats = 7 →
  final_ratio_dogs = 15 →
  final_ratio_cats = 11 →
  (dogs : ℚ) / (dogs * initial_ratio_cats / initial_ratio_dogs : ℚ) = 
    initial_ratio_dogs / initial_ratio_cats →
  (dogs : ℚ) / (dogs * initial_ratio_cats / initial_ratio_dogs + additional_cats : ℚ) = 
    final_ratio_dogs / final_ratio_cats →
  additional_cats = 20 := by
  sorry

end NUMINAMATH_CALUDE_animal_shelter_cats_l3122_312227


namespace NUMINAMATH_CALUDE_height_difference_is_9_l3122_312254

/-- The height of the Empire State Building in meters -/
def empire_state_height : ℝ := 443

/-- The height of the Petronas Towers in meters -/
def petronas_height : ℝ := 452

/-- The height difference between the Petronas Towers and the Empire State Building -/
def height_difference : ℝ := petronas_height - empire_state_height

theorem height_difference_is_9 : height_difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_is_9_l3122_312254


namespace NUMINAMATH_CALUDE_square_nonnegative_l3122_312210

theorem square_nonnegative (a : ℝ) : a^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_nonnegative_l3122_312210


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3122_312225

/-- Given a circle and a point of symmetry, find the equation of the symmetric circle -/
theorem symmetric_circle_equation (x y : ℝ) :
  let given_circle := (x - 2)^2 + (y - 1)^2 = 1
  let point_of_symmetry := (1, 2)
  let symmetric_circle := x^2 + (y - 3)^2 = 1
  symmetric_circle = true := by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3122_312225


namespace NUMINAMATH_CALUDE_chord_length_circle_line_intersection_chord_length_proof_l3122_312270

/-- The chord length cut by the line y = x from the circle x^2 + (y+2)^2 = 4 is 2√2 -/
theorem chord_length_circle_line_intersection : Real → Prop :=
  λ chord_length =>
    let circle := λ x y => x^2 + (y+2)^2 = 4
    let line := λ x y => y = x
    ∃ x₁ y₁ x₂ y₂,
      circle x₁ y₁ ∧ circle x₂ y₂ ∧
      line x₁ y₁ ∧ line x₂ y₂ ∧
      chord_length = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ∧
      chord_length = 2 * Real.sqrt 2

/-- Proof of the chord length theorem -/
theorem chord_length_proof : chord_length_circle_line_intersection (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_circle_line_intersection_chord_length_proof_l3122_312270


namespace NUMINAMATH_CALUDE_overall_loss_is_450_l3122_312214

/-- Calculates the overall loss amount for a stock sale scenario --/
def calculate_loss (stock_value : ℝ) : ℝ :=
  let profit_portion := 0.2
  let loss_portion := 0.8
  let profit_rate := 0.1
  let loss_rate := 0.05
  let selling_price := (profit_portion * stock_value * (1 + profit_rate)) +
                       (loss_portion * stock_value * (1 - loss_rate))
  stock_value - selling_price

/-- Theorem stating that the overall loss amount is 450 for the given scenario --/
theorem overall_loss_is_450 :
  calculate_loss 22500 = 450 := by sorry

end NUMINAMATH_CALUDE_overall_loss_is_450_l3122_312214


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l3122_312240

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧  -- f is an odd function
  (∀ x y : ℝ, 0 < x → x < y → f x < f y)  -- f is monotonically increasing on (0, +∞)
  := by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l3122_312240


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3122_312229

/-- For an arithmetic sequence, if the first term is greater than or equal to the second term,
    then the square of the second term is greater than or equal to the product of the first and third terms. -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) :
  a 1 ≥ a 2 → a 2 ^ 2 ≥ a 1 * a 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3122_312229


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3122_312216

theorem complex_modulus_problem (z : ℂ) (i : ℂ) (h : i^2 = -1) (hz : z = (3 - i) / (1 + i)) :
  Complex.abs (z + i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3122_312216


namespace NUMINAMATH_CALUDE_value_of_M_l3122_312275

theorem value_of_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1500) ∧ (M = 4500) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l3122_312275


namespace NUMINAMATH_CALUDE_last_integer_in_sequence_l3122_312241

def sequence_term (n : ℕ) : ℚ :=
  524288 / 2^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem last_integer_in_sequence :
  ∃ (k : ℕ), (∀ (n : ℕ), n ≤ k → is_integer (sequence_term n)) ∧
             (∀ (m : ℕ), m > k → ¬ is_integer (sequence_term m)) ∧
             sequence_term k = 1 :=
sorry

end NUMINAMATH_CALUDE_last_integer_in_sequence_l3122_312241


namespace NUMINAMATH_CALUDE_mean_temperature_l3122_312235

def temperatures : List ℝ := [-6.5, -2, -3.5, -1, 0.5, 4, 1.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = -1 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l3122_312235


namespace NUMINAMATH_CALUDE_triangle_hyperbola_ratio_l3122_312223

-- Define the right triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  let (xC, yC) := C
  (xB - xA)^2 + (yB - yA)^2 = 3^2 ∧
  (xC - xA)^2 + (yC - yA)^2 = 1^2 ∧
  (xB - xA) * (xC - xA) + (yB - yA) * (yC - yA) = 0

-- Define the hyperbola passing through A and intersecting AB at D
def Hyperbola (A B C D : ℝ × ℝ) (a b : ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  let (xC, yC) := C
  let (xD, yD) := D
  a > 0 ∧ b > 0 ∧
  xA^2 / a^2 - yA^2 / b^2 = 1 ∧
  xD^2 / a^2 - yD^2 / b^2 = 1 ∧
  (xD - xA) * (yB - yA) = (yD - yA) * (xB - xA)

-- Theorem statement
theorem triangle_hyperbola_ratio 
  (A B C D : ℝ × ℝ) (a b : ℝ) :
  Triangle A B C → Hyperbola A B C D a b →
  let (xA, yA) := A
  let (xB, yB) := B
  let (xD, yD) := D
  Real.sqrt ((xD - xA)^2 + (yD - yA)^2) / Real.sqrt ((xB - xD)^2 + (yB - yD)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_hyperbola_ratio_l3122_312223


namespace NUMINAMATH_CALUDE_birds_storks_difference_l3122_312277

/-- Given the initial conditions of birds and storks on a fence, prove that there are 3 more birds than storks. -/
theorem birds_storks_difference :
  let initial_birds : ℕ := 2
  let additional_birds : ℕ := 5
  let storks : ℕ := 4
  let total_birds : ℕ := initial_birds + additional_birds
  total_birds - storks = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_birds_storks_difference_l3122_312277


namespace NUMINAMATH_CALUDE_point_coordinates_l3122_312253

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := |y|

def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem point_coordinates :
  ∀ (x y : ℝ),
    fourth_quadrant x y →
    distance_to_x_axis y = 2 →
    distance_to_y_axis x = 4 →
    x = 4 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3122_312253


namespace NUMINAMATH_CALUDE_no_real_solution_l3122_312267

theorem no_real_solution (a : ℝ) (ha : a > 0) (h : a^3 = 6*(a + 1)) :
  ∀ x : ℝ, x^2 + a*x + a^2 - 6 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_l3122_312267


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3122_312276

theorem rectangle_dimension_change (l b : ℝ) (h1 : l > 0) (h2 : b > 0) : 
  let new_l := l / 2
  let new_area := (l * b) / 2
  ∃ new_b : ℝ, new_area = new_l * new_b ∧ new_b = b / 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3122_312276


namespace NUMINAMATH_CALUDE_max_product_sum_300_l3122_312255

theorem max_product_sum_300 :
  (∃ (a b : ℤ), a + b = 300 ∧ ∀ (x y : ℤ), x + y = 300 → x * y ≤ a * b) ∧
  (∀ (a b : ℤ), a + b = 300 → a * b ≤ 22500) :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l3122_312255


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l3122_312202

/-- The trajectory of the center of a moving circle -/
def trajectory_equation (x y : ℝ) : Prop := y^2 = 8*x

/-- A point on the circle -/
def fixed_point : ℝ × ℝ := (4, 0)

/-- Length of the chord on y-axis -/
def chord_length : ℝ := 8

theorem trajectory_of_moving_circle :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), (x - 4)^2 + y^2 = r^2) ∧ 
  (∃ (a : ℝ), a^2 + x^2 = (chord_length/2)^2) →
  trajectory_equation x y := by sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l3122_312202


namespace NUMINAMATH_CALUDE_real_part_of_z_is_negative_four_l3122_312207

theorem real_part_of_z_is_negative_four :
  let i : ℂ := Complex.I
  let z : ℂ := (3 + 4 * i) * i
  (z.re : ℝ) = -4 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_is_negative_four_l3122_312207


namespace NUMINAMATH_CALUDE_system_solution_l3122_312224

theorem system_solution (a b : ℝ) : 
  a^2 + b^2 = 25 ∧ 3*(a + b) - a*b = 15 ↔ 
  ((a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0) ∨ (a = 4 ∧ b = -3) ∨ (a = -3 ∧ b = 4)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3122_312224


namespace NUMINAMATH_CALUDE_blue_balloon_count_l3122_312294

/-- The number of blue balloons owned by Joan, Melanie, Alex, and Gary, respectively --/
def joan_balloons : ℕ := 60
def melanie_balloons : ℕ := 85
def alex_balloons : ℕ := 37
def gary_balloons : ℕ := 48

/-- The total number of blue balloons --/
def total_blue_balloons : ℕ := joan_balloons + melanie_balloons + alex_balloons + gary_balloons

theorem blue_balloon_count : total_blue_balloons = 230 := by
  sorry

end NUMINAMATH_CALUDE_blue_balloon_count_l3122_312294


namespace NUMINAMATH_CALUDE_part_one_part_two_l3122_312290

/-- Definition of proposition p -/
def p (x a : ℝ) : Prop := x^2 - 2*a*x - 3*a^2 < 0

/-- Definition of proposition q -/
def q (x : ℝ) : Prop := 2 ≤ x ∧ x < 4

/-- Theorem for part (1) -/
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 ≤ x ∧ x < 3) :=
sorry

/-- Theorem for part (2) -/
theorem part_two :
  ∀ a : ℝ, (a > 0 ∧ (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x)) ↔ (4/3 ≤ a) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3122_312290


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3122_312265

theorem polynomial_evaluation : 7^4 - 4 * 7^3 + 6 * 7^2 - 4 * 7 + 1 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3122_312265


namespace NUMINAMATH_CALUDE_altitudes_5_12_13_impossible_l3122_312274

-- Define a function to check if three numbers can be altitudes of a triangle
def canBeAltitudes (a b c : ℝ) : Prop :=
  ∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
    x * a = y * b ∧ y * b = z * c →
    x + y > z ∧ y + z > x ∧ z + x > y

-- Theorem statement
theorem altitudes_5_12_13_impossible :
  ¬ canBeAltitudes 5 12 13 :=
sorry

end NUMINAMATH_CALUDE_altitudes_5_12_13_impossible_l3122_312274


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3122_312218

theorem complex_magnitude_problem (x y : ℝ) (h : (x + y * Complex.I) * Complex.I = 1 + Complex.I) :
  Complex.abs (x + 2 * y * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3122_312218


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3122_312279

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a^2 + b^2 = c^2) : 
  (a^3 + b^3 + c^3) / (a * b * (a + b + c)) ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3122_312279


namespace NUMINAMATH_CALUDE_area_of_region_R_approx_l3122_312242

/-- Represents a rhombus ABCD -/
structure Rhombus :=
  (side_length : ℝ)
  (angle_B : ℝ)

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem statement -/
theorem area_of_region_R_approx (r : Rhombus) :
  r.side_length = 4 ∧ r.angle_B = π/3 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |area (region_R r) - 3| < ε :=
sorry

end NUMINAMATH_CALUDE_area_of_region_R_approx_l3122_312242


namespace NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l3122_312222

/-- A linear function f(x) = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The four quadrants of the Cartesian plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Determine if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I  => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- A linear function passes through a quadrant if there exists a point (x, y) in that quadrant satisfying the function equation -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  ∃ x y : ℝ, y = f.m * x + f.b ∧ inQuadrant x y q

/-- The main theorem: the graph of y = 2x - 3 does not pass through the second quadrant -/
theorem linear_function_not_in_second_quadrant :
  ¬ passesThrough ⟨2, -3⟩ Quadrant.II := by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l3122_312222


namespace NUMINAMATH_CALUDE_min_value_expression_l3122_312211

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  y / x + 16 * x / (2 * x + y) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3122_312211


namespace NUMINAMATH_CALUDE_total_material_is_correct_l3122_312299

/-- The amount of sand required for the renovation project in truck-loads -/
def sand : ℚ := 0.16666666666666666

/-- The amount of dirt required for the renovation project in truck-loads -/
def dirt : ℚ := 0.3333333333333333

/-- The amount of cement required for the renovation project in truck-loads -/
def cement : ℚ := 0.16666666666666666

/-- The total amount of material required for the renovation project in truck-loads -/
def total_material : ℚ := sand + dirt + cement

/-- Theorem stating that the total amount of material required is 0.6666666666666666 truck-loads -/
theorem total_material_is_correct : total_material = 0.6666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_total_material_is_correct_l3122_312299


namespace NUMINAMATH_CALUDE_additional_apples_needed_l3122_312283

def apples_needed (current_apples : ℕ) (people : ℕ) (apples_per_person : ℕ) : ℕ :=
  if people * apples_per_person ≤ current_apples then
    0
  else
    people * apples_per_person - current_apples

theorem additional_apples_needed :
  apples_needed 68 14 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_additional_apples_needed_l3122_312283
