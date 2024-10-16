import Mathlib

namespace NUMINAMATH_CALUDE_marian_needs_31_trays_l3192_319258

/-- The number of trays Marian needs to prepare cookies for classmates and teachers -/
def trays_needed (cookies_for_classmates : ℕ) (cookies_for_teachers : ℕ) (cookies_per_tray : ℕ) : ℕ :=
  (cookies_for_classmates + cookies_for_teachers + cookies_per_tray - 1) / cookies_per_tray

/-- Proof that Marian needs 31 trays to prepare cookies for classmates and teachers -/
theorem marian_needs_31_trays :
  trays_needed 276 92 12 = 31 := by
  sorry

end NUMINAMATH_CALUDE_marian_needs_31_trays_l3192_319258


namespace NUMINAMATH_CALUDE_problem_solution_l3192_319232

theorem problem_solution : 
  let tan60 := Real.sqrt 3
  |Real.sqrt 2 - Real.sqrt 3| - tan60 + 1 / Real.sqrt 2 = -(Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3192_319232


namespace NUMINAMATH_CALUDE_total_spots_granger_and_cisco_l3192_319200

/-- The number of spots Rover has -/
def rover_spots : ℕ := 46

/-- The number of spots Cisco has -/
def cisco_spots : ℕ := rover_spots / 2 - 5

/-- The number of spots Granger has -/
def granger_spots : ℕ := 5 * cisco_spots

/-- Theorem stating the total number of spots Granger and Cisco have combined -/
theorem total_spots_granger_and_cisco : 
  granger_spots + cisco_spots = 108 := by sorry

end NUMINAMATH_CALUDE_total_spots_granger_and_cisco_l3192_319200


namespace NUMINAMATH_CALUDE_ratio_of_terms_l3192_319244

/-- Two arithmetic sequences and their properties -/
structure ArithmeticSequences where
  a : ℕ → ℚ  -- First sequence
  b : ℕ → ℚ  -- Second sequence
  S : ℕ → ℚ  -- Sum of first n terms of a
  T : ℕ → ℚ  -- Sum of first n terms of b
  h_arithmetic_a : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  h_arithmetic_b : ∀ n : ℕ, b (n + 2) - b (n + 1) = b (n + 1) - b n
  h_sum_a : ∀ n : ℕ, S n = (n : ℚ) * (a 1 + a n) / 2
  h_sum_b : ∀ n : ℕ, T n = (n : ℚ) * (b 1 + b n) / 2
  h_ratio : ∀ n : ℕ, S n / T n = (2 * n : ℚ) / (3 * n + 1)

/-- Main theorem: If the ratio of sums is given, then a_5 / b_6 = 9 / 17 -/
theorem ratio_of_terms (seq : ArithmeticSequences) : seq.a 5 / seq.b 6 = 9 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_terms_l3192_319244


namespace NUMINAMATH_CALUDE_johnson_family_seating_l3192_319212

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem johnson_family_seating (num_boys num_girls : ℕ) 
  (h1 : num_boys = 5) 
  (h2 : num_girls = 4) 
  (h3 : num_boys + num_girls = 9) : 
  factorial (num_boys + num_girls) - factorial num_boys * factorial num_girls = 359760 :=
sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l3192_319212


namespace NUMINAMATH_CALUDE_remaining_pie_portion_l3192_319269

theorem remaining_pie_portion (carlos_share maria_fraction : ℝ) : 
  carlos_share = 0.6 →
  maria_fraction = 0.25 →
  (1 - carlos_share) * (1 - maria_fraction) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pie_portion_l3192_319269


namespace NUMINAMATH_CALUDE_garden_area_calculation_l3192_319230

def garden_length : ℝ := 18
def garden_width : ℝ := 15
def cutout1_side : ℝ := 4
def cutout2_side : ℝ := 2

theorem garden_area_calculation :
  garden_length * garden_width - (cutout1_side * cutout1_side + cutout2_side * cutout2_side) = 250 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_calculation_l3192_319230


namespace NUMINAMATH_CALUDE_abc_equation_solutions_l3192_319223

theorem abc_equation_solutions (a b c : ℕ+) :
  a * b * c + a * b + c = a ^ 3 →
  ((b = a - 1 ∧ c = a) ∨ (b = 1 ∧ c = a * (a - 1))) :=
sorry

end NUMINAMATH_CALUDE_abc_equation_solutions_l3192_319223


namespace NUMINAMATH_CALUDE_distinct_primes_not_dividing_l3192_319263

/-- A function that pairs the positive divisors of a number -/
def divisor_pairing (n : ℕ+) : Set (ℕ × ℕ) := sorry

/-- Predicate to check if a number is prime -/
def is_prime (p : ℕ) : Prop := Nat.Prime p

/-- The main theorem -/
theorem distinct_primes_not_dividing (n : ℕ+) 
  (h : ∀ (pair : ℕ × ℕ), pair ∈ divisor_pairing n → is_prime (pair.1 + pair.2)) :
  (∀ (p q : ℕ), 
    (∃ (pair1 pair2 : ℕ × ℕ), pair1 ∈ divisor_pairing n ∧ pair2 ∈ divisor_pairing n ∧ 
      p = pair1.1 + pair1.2 ∧ q = pair2.1 + pair2.2 ∧ p ≠ q) →
    (∀ (r : ℕ), (∃ (pair : ℕ × ℕ), pair ∈ divisor_pairing n ∧ r = pair.1 + pair.2) → 
      ¬(r ∣ n))) :=
sorry

end NUMINAMATH_CALUDE_distinct_primes_not_dividing_l3192_319263


namespace NUMINAMATH_CALUDE_g_value_l3192_319284

/-- Definition of g(n) as the smallest possible number of integers left on the blackboard --/
def g (n : ℕ) : ℕ := sorry

/-- Theorem stating the value of g(n) for all n ≥ 2 --/
theorem g_value (n : ℕ) (h : n ≥ 2) :
  (∃ k : ℕ, n = 2^k ∧ g n = 2) ∨ (¬∃ k : ℕ, n = 2^k) ∧ g n = 3 := by sorry

end NUMINAMATH_CALUDE_g_value_l3192_319284


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l3192_319214

theorem symmetric_complex_product :
  ∀ (z₁ z₂ : ℂ),
  z₁ = 1 + Complex.I →
  Complex.re z₂ = -Complex.re z₁ →
  Complex.im z₂ = Complex.im z₁ →
  z₁ * z₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l3192_319214


namespace NUMINAMATH_CALUDE_oxygen_atoms_in_compound_l3192_319281

def atomic_weight_carbon : ℕ := 12
def atomic_weight_hydrogen : ℕ := 1
def atomic_weight_oxygen : ℕ := 16

def num_carbon_atoms : ℕ := 3
def num_hydrogen_atoms : ℕ := 6
def total_molecular_weight : ℕ := 58

theorem oxygen_atoms_in_compound :
  let weight_carbon_hydrogen := num_carbon_atoms * atomic_weight_carbon + num_hydrogen_atoms * atomic_weight_hydrogen
  let weight_oxygen := total_molecular_weight - weight_carbon_hydrogen
  weight_oxygen / atomic_weight_oxygen = 1 := by sorry

end NUMINAMATH_CALUDE_oxygen_atoms_in_compound_l3192_319281


namespace NUMINAMATH_CALUDE_find_number_l3192_319290

theorem find_number : ∃ x : ℝ, 0.20 * x + 0.25 * 60 = 23 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3192_319290


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l3192_319266

/-- The number of walnut trees planted in a park --/
theorem walnut_trees_planted (initial final planted : ℕ) :
  initial = 22 →
  final = 55 →
  planted = final - initial →
  planted = 33 := by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l3192_319266


namespace NUMINAMATH_CALUDE_billy_has_24_balloons_l3192_319260

/-- The number of water balloons Billy is left with after the water balloon fight -/
def billys_balloons (total_packs : ℕ) (balloons_per_pack : ℕ) (num_people : ℕ) 
  (extra_milly : ℕ) (extra_tamara : ℕ) (extra_floretta : ℕ) : ℕ :=
  (total_packs * balloons_per_pack) / num_people

/-- Theorem stating that Billy is left with 24 water balloons -/
theorem billy_has_24_balloons : 
  billys_balloons 12 8 4 11 9 4 = 24 := by
  sorry

#eval billys_balloons 12 8 4 11 9 4

end NUMINAMATH_CALUDE_billy_has_24_balloons_l3192_319260


namespace NUMINAMATH_CALUDE_f_negative_one_value_l3192_319204

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The definition of f for positive x -/
def f_pos (x : ℝ) : ℝ :=
  2 * x^2 - 1

theorem f_negative_one_value
    (f : ℝ → ℝ)
    (h_odd : IsOdd f)
    (h_pos : ∀ x > 0, f x = f_pos x) :
    f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_one_value_l3192_319204


namespace NUMINAMATH_CALUDE_at_least_two_equations_have_real_solutions_l3192_319243

theorem at_least_two_equations_have_real_solutions (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  let eq1 := fun x => (x - a) * (x - b) = x - c
  let eq2 := fun x => (x - c) * (x - b) = x - a
  let eq3 := fun x => (x - a) * (x - c) = x - b
  let has_real_solution := fun f => ∃ x : ℝ, f x
  (has_real_solution eq1 ∧ has_real_solution eq2) ∨
  (has_real_solution eq1 ∧ has_real_solution eq3) ∨
  (has_real_solution eq2 ∧ has_real_solution eq3) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_equations_have_real_solutions_l3192_319243


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l3192_319242

def a : ℝ × ℝ := (-1, -3)
def b (t : ℝ) : ℝ × ℝ := (2, t)

theorem parallel_vectors_subtraction :
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b t →
  a - b t = (-3, -9) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l3192_319242


namespace NUMINAMATH_CALUDE_star_three_neg_two_thirds_l3192_319247

-- Define the ☆ operation
def star (x y : ℚ) : ℚ := x^2 + x*y

-- State the theorem
theorem star_three_neg_two_thirds : star 3 (-2/3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_star_three_neg_two_thirds_l3192_319247


namespace NUMINAMATH_CALUDE_corn_purchase_proof_l3192_319276

/-- The cost of corn in cents per pound -/
def corn_cost : ℚ := 99

/-- The cost of beans in cents per pound -/
def bean_cost : ℚ := 45

/-- The total weight of corn and beans in pounds -/
def total_weight : ℚ := 24

/-- The total cost in cents -/
def total_cost : ℚ := 1809

/-- The number of pounds of corn bought -/
def corn_pounds : ℚ := 13.5

theorem corn_purchase_proof :
  ∃ (bean_pounds : ℚ),
    bean_pounds + corn_pounds = total_weight ∧
    bean_cost * bean_pounds + corn_cost * corn_pounds = total_cost :=
by sorry

end NUMINAMATH_CALUDE_corn_purchase_proof_l3192_319276


namespace NUMINAMATH_CALUDE_banana_cost_18lbs_l3192_319240

/-- The cost of bananas given a rate, weight, and discount condition -/
def banana_cost (rate : ℚ) (rate_weight : ℚ) (weight : ℚ) (discount_threshold : ℚ) (discount_rate : ℚ) : ℚ :=
  let base_cost := (weight / rate_weight) * rate
  if weight ≥ discount_threshold then
    base_cost * (1 - discount_rate)
  else
    base_cost

/-- Theorem stating the cost of 18 pounds of bananas given the specified conditions -/
theorem banana_cost_18lbs : 
  banana_cost 3 3 18 15 (1/10) = 162/10 := by
  sorry

end NUMINAMATH_CALUDE_banana_cost_18lbs_l3192_319240


namespace NUMINAMATH_CALUDE_percentage_problem_l3192_319222

theorem percentage_problem (x : ℝ) (h : 75 = 0.6 * x) : x = 125 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3192_319222


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3192_319280

theorem inequality_system_solution_set :
  ∀ x : ℝ, (2 - x > 0 ∧ 2*x + 3 > 1) ↔ (-1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3192_319280


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3192_319251

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3192_319251


namespace NUMINAMATH_CALUDE_investment_problem_l3192_319297

theorem investment_problem (P : ℝ) : 
  let A1 := 1.02 * P - 100
  let A2 := 1.03 * A1 + 200
  let A3 := 1.04 * A2
  let A4 := 1.05 * A3
  let A5 := 1.06 * A4
  A5 = 750 →
  1.19304696 * P + 112.27824 = 750 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l3192_319297


namespace NUMINAMATH_CALUDE_potato_planting_l3192_319265

theorem potato_planting (rows : ℕ) (additional_plants : ℕ) (total_plants : ℕ) 
  (h1 : rows = 7)
  (h2 : additional_plants = 15)
  (h3 : total_plants = 141)
  : (total_plants - additional_plants) / rows = 18 := by
  sorry

end NUMINAMATH_CALUDE_potato_planting_l3192_319265


namespace NUMINAMATH_CALUDE_m_is_even_l3192_319285

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem m_is_even (M : ℕ) (h1 : sum_of_digits M = 100) (h2 : sum_of_digits (5 * M) = 50) : 
  Even M := by sorry

end NUMINAMATH_CALUDE_m_is_even_l3192_319285


namespace NUMINAMATH_CALUDE_min_value_xy_l3192_319259

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) :
  x * y ≥ 64 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2/x₀ + 8/y₀ = 1 ∧ x₀ * y₀ = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xy_l3192_319259


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_18_mod_25_l3192_319291

theorem largest_five_digit_congruent_to_18_mod_25 : ∃ n : ℕ,
  n = 99993 ∧
  n ≥ 10000 ∧ n < 100000 ∧
  n % 25 = 18 ∧
  ∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 25 = 18 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_18_mod_25_l3192_319291


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_40_60_1_7_l3192_319241

/-- Sum of an arithmetic series -/
def arithmetic_series_sum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_40_60_1_7 :
  arithmetic_series_sum 40 60 (1/7) = 7050 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_40_60_1_7_l3192_319241


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3192_319271

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r ≠ 0, ∀ n, a (n + 1) = r * a n

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0)
  (h_sum : a 1 + a 2 + a 5 = 13)
  (h_geometric : geometric_sequence (λ n ↦ a (2 * n - 1)))
  (h_arithmetic : arithmetic_sequence a d) :
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3192_319271


namespace NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l3192_319293

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem parallel_transitivity_counterexample 
  (a : Line) (α β : Plane) :
  ¬(∀ a α β, parallel_line_plane a α → parallel_line_plane a β → 
    parallel_plane_plane α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l3192_319293


namespace NUMINAMATH_CALUDE_remainder_problem_l3192_319221

theorem remainder_problem (x y z : ℤ) 
  (hx : x % 186 = 19)
  (hy : y % 248 = 23)
  (hz : z % 372 = 29) :
  ((x * y * z) + 47) % 93 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3192_319221


namespace NUMINAMATH_CALUDE_train_crossing_time_l3192_319264

/-- The time taken for a train to cross a man walking in the same direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 700 →
  train_speed = 63 * 1000 / 3600 →
  man_speed = 3 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 42 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3192_319264


namespace NUMINAMATH_CALUDE_smallest_class_size_l3192_319253

theorem smallest_class_size : ∃ (n : ℕ), n > 0 ∧ (∃ (k : ℕ), k > 0 ∧ 29/10 < (100 * k : ℚ)/n ∧ (100 * k : ℚ)/n < 31/10) ∧
  ∀ (m : ℕ), m > 0 → m < n → ¬(∃ (j : ℕ), j > 0 ∧ 29/10 < (100 * j : ℚ)/m ∧ (100 * j : ℚ)/m < 31/10) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l3192_319253


namespace NUMINAMATH_CALUDE_five_Y_three_equals_two_l3192_319225

-- Define the Y operation
def Y (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2 - x + y

-- Theorem statement
theorem five_Y_three_equals_two : Y 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_five_Y_three_equals_two_l3192_319225


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3192_319255

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h1 : geometric_sequence a q)
  (h2 : a 1 + a 2 + a 3 = 1)
  (h3 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 9) : 
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3192_319255


namespace NUMINAMATH_CALUDE_least_positive_four_digit_solution_l3192_319215

theorem least_positive_four_digit_solution (x : ℕ) : 
  (10 * x ≡ 30 [ZMOD 20]) ∧ 
  (2 * x + 10 ≡ 19 [ZMOD 9]) ∧ 
  (-3 * x + 1 ≡ x [ZMOD 19]) ∧ 
  (x ≥ 1000) ∧ (x < 10000) →
  x ≥ 1296 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_four_digit_solution_l3192_319215


namespace NUMINAMATH_CALUDE_existence_of_squares_with_difference_2023_l3192_319292

theorem existence_of_squares_with_difference_2023 :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 = y^2 + 2023 ∧
  ((x = 1012 ∧ y = 1011) ∨ (x = 148 ∧ y = 141) ∨ (x = 68 ∧ y = 51)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_squares_with_difference_2023_l3192_319292


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l3192_319202

theorem quadratic_root_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 14) →
  p + q = 69 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l3192_319202


namespace NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l3192_319224

/-- A function that returns true if a number is a single-digit prime -/
def isSingleDigitPrime (p : ℕ) : Prop :=
  p < 10 ∧ Nat.Prime p

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem -/
theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    isSingleDigitPrime d ∧
    isSingleDigitPrime e ∧
    Nat.Prime (d^2 + e^2) ∧
    n = d * e * (d^2 + e^2) ∧
    (∀ (m : ℕ), m > n →
      ¬(∃ (p q r : ℕ), isSingleDigitPrime p ∧
                        isSingleDigitPrime q ∧
                        Nat.Prime r ∧
                        r = p^2 + q^2 ∧
                        m = p * q * r)) ∧
    sumOfDigits n = 11 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l3192_319224


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3192_319248

theorem constant_term_expansion (x : ℝ) (x_ne_zero : x ≠ 0) : 
  ∃ (c : ℕ), c = 17920 ∧ 
  ∃ (f : ℝ → ℝ), (λ x => (2*x + 2/x)^8) = (λ x => c + f x) ∧ 
  (∀ x ≠ 0, f x ≠ 0 → ∃ (n : ℤ), n ≠ 0 ∧ f x = x^n * (f x / x^n)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3192_319248


namespace NUMINAMATH_CALUDE_marble_count_l3192_319233

theorem marble_count (r b g : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.7 * r) :
  r + b + g = 3.469 * r := by sorry

end NUMINAMATH_CALUDE_marble_count_l3192_319233


namespace NUMINAMATH_CALUDE_regression_line_correct_l3192_319231

def points : List (ℝ × ℝ) := [(1, 2), (2, 3), (3, 4), (4, 5)]

def regression_line (points : List (ℝ × ℝ)) : ℝ → ℝ := 
  fun x => x + 1

theorem regression_line_correct : 
  regression_line points = fun x => x + 1 := by sorry

end NUMINAMATH_CALUDE_regression_line_correct_l3192_319231


namespace NUMINAMATH_CALUDE_job_completion_time_l3192_319268

theorem job_completion_time 
  (initial_men : ℕ) 
  (initial_days : ℕ) 
  (new_men : ℕ) 
  (prep_days : ℕ) 
  (h1 : initial_men = 10) 
  (h2 : initial_days = 15) 
  (h3 : new_men = 15) 
  (h4 : prep_days = 2) : 
  (initial_men * initial_days) / new_men + prep_days = 12 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l3192_319268


namespace NUMINAMATH_CALUDE_inequality_equivalence_fraction_comparison_l3192_319283

-- Problem 1
theorem inequality_equivalence (m x : ℝ) (h : m > 2) :
  m * x + 4 < m^2 + 2 * x ↔ x < m + 2 := by sorry

-- Problem 2
theorem fraction_comparison (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  x / (1 + x) > y / (1 + y) := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_fraction_comparison_l3192_319283


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3192_319208

/-- Represents a geometric sequence -/
structure GeometricSequence (α : Type*) [Ring α] where
  a : ℕ → α
  r : α
  h : ∀ n, a (n + 1) = r * a n

/-- Sum of the first n terms of a geometric sequence -/
def sum_n {α : Type*} [Ring α] (seq : GeometricSequence α) (n : ℕ) : α :=
  sorry

/-- The main theorem stating that for any geometric sequence, 
    a_{2016}(S_{2016}-S_{2015}) ≠ 0 -/
theorem geometric_sequence_property {α : Type*} [Field α] (seq : GeometricSequence α) :
  seq.a 2016 * (sum_n seq 2016 - sum_n seq 2015) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3192_319208


namespace NUMINAMATH_CALUDE_triangle_area_triple_altitude_l3192_319206

theorem triangle_area_triple_altitude (b h : ℝ) (h_pos : 0 < h) :
  let A := (1/2) * b * h
  let A' := (1/2) * b * (3*h)
  A' = 3 * A := by sorry

end NUMINAMATH_CALUDE_triangle_area_triple_altitude_l3192_319206


namespace NUMINAMATH_CALUDE_change_difference_is_twenty_percent_l3192_319295

-- Define the percentages as rationals between 0 and 1
def initial_yes : ℚ := 60 / 100
def initial_no : ℚ := 40 / 100
def final_yes : ℚ := 85 / 100
def final_no : ℚ := 15 / 100

-- Define the function to calculate the difference between max and min change
def change_difference : ℚ := by
  sorry

-- Theorem statement
theorem change_difference_is_twenty_percent :
  change_difference = 20 / 100 := by
  sorry

end NUMINAMATH_CALUDE_change_difference_is_twenty_percent_l3192_319295


namespace NUMINAMATH_CALUDE_integral_f_minus_x_equals_five_sixths_l3192_319218

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x

-- State the theorem
theorem integral_f_minus_x_equals_five_sixths :
  (∀ x, deriv f x = 2 * x + 1) →
  ∫ x in (1)..(2), f (-x) = 5/6 := by sorry

end NUMINAMATH_CALUDE_integral_f_minus_x_equals_five_sixths_l3192_319218


namespace NUMINAMATH_CALUDE_root_implies_q_value_l3192_319279

theorem root_implies_q_value (p q : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 : ℂ) * (3 + 2 * Complex.I)^2 + p * (3 + 2 * Complex.I) + q = 0 →
  q = 26 := by
sorry

end NUMINAMATH_CALUDE_root_implies_q_value_l3192_319279


namespace NUMINAMATH_CALUDE_amusement_park_initial_cost_l3192_319245

/-- The initial cost to open an amusement park, given the conditions described in the problem. -/
def initial_cost : ℝ → Prop := λ C =>
  let daily_running_cost := 0.01 * C
  let daily_revenue := 1500
  let days_to_breakeven := 200
  C = days_to_breakeven * (daily_revenue - daily_running_cost)

/-- Theorem stating that the initial cost to open the amusement park is $100,000. -/
theorem amusement_park_initial_cost :
  ∃ C : ℝ, initial_cost C ∧ C = 100000 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_initial_cost_l3192_319245


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l3192_319299

theorem max_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (x + 34) + Real.sqrt (17 - x) + Real.sqrt (2 * x) ≤ Real.sqrt 51 + Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l3192_319299


namespace NUMINAMATH_CALUDE_first_digit_change_largest_l3192_319274

def original : ℚ := 0.1234567

def change_digit (n : ℚ) (pos : ℕ) : ℚ :=
  n + (8 - (n * 10^pos % 10)) / 10^pos

theorem first_digit_change_largest :
  ∀ pos : ℕ, pos > 0 → change_digit original 0 ≥ change_digit original pos :=
by
  sorry

end NUMINAMATH_CALUDE_first_digit_change_largest_l3192_319274


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l3192_319287

theorem units_digit_of_7_power_2023 : ∃ n : ℕ, 7^2023 ≡ 3 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l3192_319287


namespace NUMINAMATH_CALUDE_circle_C_equation_l3192_319267

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 2)^2 = 13

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  2*x - 7*y + 8 = 0

-- Define points A and B
def point_A : ℝ × ℝ := (6, 0)
def point_B : ℝ × ℝ := (1, 5)

-- Theorem statement
theorem circle_C_equation :
  ∀ (x y : ℝ),
    (∃ (cx cy : ℝ), line_l cx cy ∧ 
      (x - cx)^2 + (y - cy)^2 = (point_A.1 - cx)^2 + (point_A.2 - cy)^2 ∧
      (x - cx)^2 + (y - cy)^2 = (point_B.1 - cx)^2 + (point_B.2 - cy)^2) →
    circle_C x y :=
by
  sorry

end NUMINAMATH_CALUDE_circle_C_equation_l3192_319267


namespace NUMINAMATH_CALUDE_perfect_square_values_l3192_319237

theorem perfect_square_values (x : ℕ) : 
  (x = 0 ∨ x = 9 ∨ x = 12) → 
  ∃ y : ℕ, 2^6 + 2^10 + 2^x = y^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_values_l3192_319237


namespace NUMINAMATH_CALUDE_pages_left_to_write_l3192_319216

-- Define the total number of pages for the book
def total_pages : ℕ := 500

-- Define the number of pages written on each day
def day1_pages : ℕ := 25
def day2_pages : ℕ := 2 * day1_pages
def day3_pages : ℕ := 2 * day2_pages
def day4_pages : ℕ := 10

-- Define the total number of pages written so far
def pages_written : ℕ := day1_pages + day2_pages + day3_pages + day4_pages

-- Define the number of pages left to write
def pages_left : ℕ := total_pages - pages_written

-- Theorem stating that the number of pages left to write is 315
theorem pages_left_to_write : pages_left = 315 := by sorry

end NUMINAMATH_CALUDE_pages_left_to_write_l3192_319216


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l3192_319250

-- Define the line
def line (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0

-- Define the third quadrant
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem: The line does not pass through the third quadrant
theorem line_not_in_third_quadrant : 
  ¬ ∃ (x y : ℝ), line x y ∧ third_quadrant x y := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l3192_319250


namespace NUMINAMATH_CALUDE_probability_x_plus_y_le_5_l3192_319210

/-- The probability that x + y ≤ 5 for a point (x,y) randomly chosen from [0,4] × [0,6] -/
theorem probability_x_plus_y_le_5 :
  let total_area : ℝ := 4 * 6
  let favorable_area : ℝ := (1 / 2) * 4 * 5
  favorable_area / total_area = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_le_5_l3192_319210


namespace NUMINAMATH_CALUDE_calculate_expression_l3192_319227

theorem calculate_expression : -2^3 / (-2) + (-2)^2 * (-5) = -16 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3192_319227


namespace NUMINAMATH_CALUDE_wine_consumption_equations_l3192_319277

/-- Represents the wine consumption and intoxication scenario from the Ming Dynasty poem --/
theorem wine_consumption_equations :
  ∃ (x y : ℚ),
    (x + y = 19) ∧
    (3 * x + (1/3) * y = 33) ∧
    (x ≥ 0) ∧ (y ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_wine_consumption_equations_l3192_319277


namespace NUMINAMATH_CALUDE_sparkling_water_cost_l3192_319252

/-- The cost of sparkling water bottles for Mary Anne -/
theorem sparkling_water_cost (bottles_per_night : ℚ) (yearly_cost : ℕ) : 
  bottles_per_night = 1/5 → yearly_cost = 146 → 
  (365 : ℚ) / 5 * (yearly_cost : ℚ) / ((365 : ℚ) / 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sparkling_water_cost_l3192_319252


namespace NUMINAMATH_CALUDE_quadratic_roots_and_m_value_l3192_319270

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-3)*x - m

-- Theorem statement
theorem quadratic_roots_and_m_value (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0) ∧
  (∀ x₁ x₂ : ℝ, quadratic m x₁ = 0 → quadratic m x₂ = 0 → x₁^2 + x₂^2 - x₁*x₂ = 7 → m = 1 ∨ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_m_value_l3192_319270


namespace NUMINAMATH_CALUDE_smallest_piece_length_l3192_319239

/-- Given a rod of length 120 cm cut into three pieces proportional to 3, 5, and 7,
    the length of the smallest piece is 24 cm. -/
theorem smallest_piece_length :
  let total_length : ℝ := 120
  let ratio_sum : ℝ := 3 + 5 + 7
  let smallest_ratio : ℝ := 3
  smallest_ratio * (total_length / ratio_sum) = 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_piece_length_l3192_319239


namespace NUMINAMATH_CALUDE_one_can_per_person_day1_l3192_319220

/-- Represents the food bank scenario --/
structure FoodBank where
  initialStock : ℕ
  day1People : ℕ
  day1Restock : ℕ
  day2People : ℕ
  day2CansPerPerson : ℕ
  day2Restock : ℕ
  totalGivenAway : ℕ

/-- Calculates the number of cans each person took on the first day --/
def cansPerPersonDay1 (fb : FoodBank) : ℕ :=
  (fb.totalGivenAway - fb.day2People * fb.day2CansPerPerson) / fb.day1People

/-- Theorem stating that each person took 1 can on the first day --/
theorem one_can_per_person_day1 (fb : FoodBank)
    (h1 : fb.initialStock = 2000)
    (h2 : fb.day1People = 500)
    (h3 : fb.day1Restock = 1500)
    (h4 : fb.day2People = 1000)
    (h5 : fb.day2CansPerPerson = 2)
    (h6 : fb.day2Restock = 3000)
    (h7 : fb.totalGivenAway = 2500) :
    cansPerPersonDay1 fb = 1 := by
  sorry

#eval cansPerPersonDay1 {
  initialStock := 2000,
  day1People := 500,
  day1Restock := 1500,
  day2People := 1000,
  day2CansPerPerson := 2,
  day2Restock := 3000,
  totalGivenAway := 2500
}

end NUMINAMATH_CALUDE_one_can_per_person_day1_l3192_319220


namespace NUMINAMATH_CALUDE_special_trapezoid_area_ratios_l3192_319235

/-- A trapezoid with a diagonal forming a 45° angle with the base, 
    and both inscribed and circumscribed circles -/
structure SpecialTrapezoid where
  -- Base lengths
  a : ℝ
  b : ℝ
  -- Height
  h : ℝ
  -- Diagonal forms 45° angle with base
  diagonal_angle : Real.cos (45 * π / 180) = h / (a - b)
  -- Inscribed circle exists
  inscribed_circle_exists : ∃ r : ℝ, r > 0 ∧ r = h / 2
  -- Circumscribed circle exists
  circumscribed_circle_exists : ∃ R : ℝ, R > 0 ∧ R = h / Real.sqrt 2

/-- The main theorem about the area ratios -/
theorem special_trapezoid_area_ratios (t : SpecialTrapezoid) : 
  (t.a + t.b) * t.h / (π * (t.h / 2)^2) = 4 / π ∧
  (t.a + t.b) * t.h / (π * (t.h / Real.sqrt 2)^2) = 2 / π := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_area_ratios_l3192_319235


namespace NUMINAMATH_CALUDE_min_value_xyz_product_min_value_achieved_l3192_319296

theorem min_value_xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x + 2 * y) * (y + 2 * z) * (x * z + 1) ≥ 16 :=
sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 2 * y₀) * (y₀ + 2 * z₀) * (x₀ * z₀ + 1) = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_xyz_product_min_value_achieved_l3192_319296


namespace NUMINAMATH_CALUDE_tommy_balloons_l3192_319288

/-- The number of balloons Tommy initially had -/
def initial_balloons : ℕ := 71

/-- The number of balloons Tommy's mom gave him -/
def mom_balloons : ℕ := 34

/-- The number of balloons Tommy gave to his friends -/
def friend_balloons : ℕ := 15

/-- The number of teddy bears Tommy got after exchanging balloons -/
def teddy_bears : ℕ := 30

/-- The exchange rate of balloons to teddy bears -/
def exchange_rate : ℕ := 3

theorem tommy_balloons : 
  initial_balloons + mom_balloons - friend_balloons = teddy_bears * exchange_rate := by
  sorry

end NUMINAMATH_CALUDE_tommy_balloons_l3192_319288


namespace NUMINAMATH_CALUDE_expression_equality_l3192_319219

theorem expression_equality (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) :
  a*(a^2 - b^2) + b*(b^2 - c^2) + c*(c^2 - a^2) = 0 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l3192_319219


namespace NUMINAMATH_CALUDE_alcohol_concentration_l3192_319298

theorem alcohol_concentration (original_volume : ℝ) (added_water : ℝ) (final_concentration : ℝ) :
  original_volume = 9 →
  added_water = 3 →
  final_concentration = 42.75 →
  (original_volume * (57 / 100)) = ((original_volume + added_water) * (final_concentration / 100)) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_concentration_l3192_319298


namespace NUMINAMATH_CALUDE_expression_evaluation_l3192_319246

theorem expression_evaluation (c : ℕ) (h : c = 4) : 
  (c^c - c * (c - 1)^(c - 1))^c = 148^4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3192_319246


namespace NUMINAMATH_CALUDE_sum_of_squares_l3192_319256

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 51)
  (h2 : x * x * y + x * y * y = 560) :
  x * x + y * y = 186 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3192_319256


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3192_319278

theorem simplify_trig_expression :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3192_319278


namespace NUMINAMATH_CALUDE_equation_solutions_l3192_319203

theorem equation_solutions :
  (∀ x : ℝ, 3 * (x - 1)^2 = 27 ↔ x = 4 ∨ x = -2) ∧
  (∀ x : ℝ, x^3 / 8 + 2 = 3 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3192_319203


namespace NUMINAMATH_CALUDE_bakers_sales_comparison_l3192_319272

/-- Baker's sales comparison -/
theorem bakers_sales_comparison 
  (usual_pastries : ℕ) (usual_bread : ℕ) 
  (today_pastries : ℕ) (today_bread : ℕ) 
  (pastry_price : ℕ) (bread_price : ℕ) : 
  usual_pastries = 20 → 
  usual_bread = 10 → 
  today_pastries = 14 → 
  today_bread = 25 → 
  pastry_price = 2 → 
  bread_price = 4 → 
  (today_pastries * pastry_price + today_bread * bread_price) - 
  (usual_pastries * pastry_price + usual_bread * bread_price) = 48 := by
  sorry

end NUMINAMATH_CALUDE_bakers_sales_comparison_l3192_319272


namespace NUMINAMATH_CALUDE_circle_coloring_exists_l3192_319234

/-- A point on a circle --/
structure CirclePoint where
  angle : Real

/-- A color (red or blue) --/
inductive Color
  | Red
  | Blue

/-- A coloring function for points on a circle --/
def ColoringFunction := CirclePoint → Color

/-- Predicate to check if three points form a right-angled triangle inscribed in the circle --/
def IsRightTriangle (p1 p2 p3 : CirclePoint) : Prop :=
  -- We assume this predicate exists and is correctly defined
  sorry

theorem circle_coloring_exists :
  ∃ (f : ColoringFunction),
    ∀ (p1 p2 p3 : CirclePoint),
      IsRightTriangle p1 p2 p3 →
        (f p1 ≠ f p2) ∨ (f p1 ≠ f p3) ∨ (f p2 ≠ f p3) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_coloring_exists_l3192_319234


namespace NUMINAMATH_CALUDE_monomial_exponent_product_l3192_319211

/-- Given that 4x^(2m)y^(m-n) and -5x^6y^2 form a monomial when added together, prove that mn = 3 -/
theorem monomial_exponent_product (m n : ℤ) : 
  (∃ (x y : ℝ), 4 * x^(2*m) * y^(m-n) + (-5) * x^6 * y^2 = c * x^a * y^b) → m * n = 3 :=
by sorry

end NUMINAMATH_CALUDE_monomial_exponent_product_l3192_319211


namespace NUMINAMATH_CALUDE_sector_area_proof_l3192_319207

-- Define the given conditions
def circle_arc_length : ℝ := 2
def central_angle : ℝ := 2

-- Define the theorem
theorem sector_area_proof :
  let radius := circle_arc_length / central_angle
  let sector_area := (1 / 2) * radius^2 * central_angle
  sector_area = 1 := by sorry

end NUMINAMATH_CALUDE_sector_area_proof_l3192_319207


namespace NUMINAMATH_CALUDE_b_min_at_3_l3192_319257

def a (n : ℕ+) : ℕ := n

def S (n : ℕ+) : ℕ := n * (n + 1) / 2

def b (n : ℕ+) : ℚ := (2 * S n + 7) / n

theorem b_min_at_3 :
  ∀ n : ℕ+, n ≠ 3 → b n > b 3 :=
sorry

end NUMINAMATH_CALUDE_b_min_at_3_l3192_319257


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3192_319238

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 = 24 / 77 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3192_319238


namespace NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l3192_319282

/-- If a, b, and c form a geometric sequence, then ax^2 + bx + c = 0 has no real roots -/
theorem no_real_roots_for_geometric_sequence_quadratic 
  (a b c : ℝ) (h_geom : b^2 = a*c ∧ a*c > 0) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l3192_319282


namespace NUMINAMATH_CALUDE_spotted_fluffy_cats_l3192_319236

def village_cats : ℕ := 120

def spotted_fraction : ℚ := 1/3

def fluffy_spotted_fraction : ℚ := 1/4

theorem spotted_fluffy_cats :
  (village_cats : ℚ) * spotted_fraction * fluffy_spotted_fraction = 10 := by
  sorry

end NUMINAMATH_CALUDE_spotted_fluffy_cats_l3192_319236


namespace NUMINAMATH_CALUDE_special_square_area_special_square_area_is_64_l3192_319217

/-- A square in the coordinate plane with specific properties -/
structure SpecialSquare where
  verticesOnY2 : ℝ × ℝ → Prop
  verticesOnY10 : ℝ × ℝ → Prop
  sidesParallelOrPerpendicular : Prop

/-- The area of the special square is 64 -/
theorem special_square_area (s : SpecialSquare) : ℝ :=
  64

/-- The main theorem stating that the area of the special square is 64 -/
theorem special_square_area_is_64 (s : SpecialSquare) : special_square_area s = 64 := by
  sorry

end NUMINAMATH_CALUDE_special_square_area_special_square_area_is_64_l3192_319217


namespace NUMINAMATH_CALUDE_parallelepiped_base_sides_l3192_319229

/-- Given a rectangular parallelepiped with a cross-section having diagonals of 20 and 8 units
    intersecting at a 60° angle, the lengths of the sides of its base are 2√5 and √30. -/
theorem parallelepiped_base_sides (d₁ d₂ : ℝ) (θ : ℝ) 
  (h₁ : d₁ = 20) (h₂ : d₂ = 8) (h₃ : θ = Real.pi / 3) :
  ∃ (a b : ℝ), a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 30 ∧ 
  (a * a + b * b = d₁ * d₁) ∧ 
  (d₂ * d₂ = 2 * a * b * Real.cos θ) := by
sorry


end NUMINAMATH_CALUDE_parallelepiped_base_sides_l3192_319229


namespace NUMINAMATH_CALUDE_different_smallest_angles_l3192_319226

/-- A type representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A type representing a set of 6 points in a plane -/
structure SixPoints :=
  (points : Fin 6 → Point)

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Predicate to check if no three points in a set of six points are collinear -/
def no_three_collinear (s : SixPoints) : Prop :=
  ∀ (i j k : Fin 6), i ≠ j → j ≠ k → i ≠ k →
    ¬collinear (s.points i) (s.points j) (s.points k)

/-- Function to calculate the angle between three points -/
noncomputable def angle (p q r : Point) : ℝ := sorry

/-- Function to find the smallest angle in a triangle -/
noncomputable def smallest_angle (p q r : Point) : ℝ :=
  min (angle p q r) (min (angle q r p) (angle r p q))

/-- The main theorem -/
theorem different_smallest_angles (s : SixPoints) (h : no_three_collinear s) :
  ∃ (i₁ j₁ k₁ i₂ j₂ k₂ : Fin 6),
    smallest_angle (s.points i₁) (s.points j₁) (s.points k₁) ≠
    smallest_angle (s.points i₂) (s.points j₂) (s.points k₂) :=
  sorry

end NUMINAMATH_CALUDE_different_smallest_angles_l3192_319226


namespace NUMINAMATH_CALUDE_large_bus_most_cost_effective_l3192_319205

/-- Represents the transportation options for the field trip --/
inductive TransportOption
  | Van
  | Minibus
  | LargeBus

/-- Calculates the number of vehicles needed for a given option --/
def vehiclesNeeded (option : TransportOption) : ℕ :=
  match option with
  | .Van => 6
  | .Minibus => 3
  | .LargeBus => 1

/-- Calculates the total cost for a given option --/
def totalCost (option : TransportOption) : ℕ :=
  match option with
  | .Van => 50 * vehiclesNeeded .Van
  | .Minibus => 100 * vehiclesNeeded .Minibus
  | .LargeBus => 250

/-- States that the large bus is the most cost-effective option --/
theorem large_bus_most_cost_effective :
  ∀ option : TransportOption, totalCost .LargeBus ≤ totalCost option :=
by sorry

end NUMINAMATH_CALUDE_large_bus_most_cost_effective_l3192_319205


namespace NUMINAMATH_CALUDE_root_product_equals_negative_183_l3192_319275

-- Define the polynomial h
def h (y : ℝ) : ℝ := y^5 - y^3 + 2*y + 3

-- Define the polynomial p
def p (y : ℝ) : ℝ := y^2 - 3

-- State the theorem
theorem root_product_equals_negative_183 
  (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (h_roots : h y₁ = 0 ∧ h y₂ = 0 ∧ h y₃ = 0 ∧ h y₄ = 0 ∧ h y₅ = 0) :
  p y₁ * p y₂ * p y₃ * p y₄ * p y₅ = -183 :=
sorry

end NUMINAMATH_CALUDE_root_product_equals_negative_183_l3192_319275


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3192_319289

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 1 = 2 ∧
  a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3192_319289


namespace NUMINAMATH_CALUDE_multiplier_can_be_greater_than_one_l3192_319254

theorem multiplier_can_be_greater_than_one (a b : ℚ) (h : a * b ≤ b) : 
  ∃ (a : ℚ), a * b ≤ b ∧ a > 1 :=
sorry

end NUMINAMATH_CALUDE_multiplier_can_be_greater_than_one_l3192_319254


namespace NUMINAMATH_CALUDE_prob_three_students_same_group_l3192_319213

/-- The probability that three specific students are assigned to the same group
    when 800 students are randomly assigned to 4 equal-sized groups -/
theorem prob_three_students_same_group :
  let total_students : ℕ := 800
  let num_groups : ℕ := 4
  let group_size : ℕ := total_students / num_groups
  -- Assuming each group has equal size
  (∀ g : Fin num_groups, (group_size : ℚ) = total_students / num_groups)
  →
  (probability_same_group : ℚ) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_students_same_group_l3192_319213


namespace NUMINAMATH_CALUDE_value_of_a_l3192_319201

theorem value_of_a (a b c : ℝ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 6) 
  (eq3 : c = 4) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3192_319201


namespace NUMINAMATH_CALUDE_right_angled_projection_l3192_319228

structure Plane where
  α : Type

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def Triangle (A B C : Point) := True

def RightAngledTriangle (A B C : Point) := Triangle A B C

def IsInPlane (p : Point) (α : Plane) : Prop := sorry

def IsOutsidePlane (p : Point) (α : Plane) : Prop := sorry

def Projection (p : Point) (α : Plane) : Point := sorry

def IsOn (p : Point) (A B : Point) : Prop := sorry

theorem right_angled_projection 
  (α : Plane) (A B C C1 : Point) : 
  RightAngledTriangle A B C →
  IsInPlane A α →
  IsInPlane B α →
  IsOutsidePlane C α →
  C1 = Projection C α →
  ¬IsOn C1 A B →
  RightAngledTriangle A B C1 := by sorry

end NUMINAMATH_CALUDE_right_angled_projection_l3192_319228


namespace NUMINAMATH_CALUDE_constant_polynomial_l3192_319209

-- Define the type for polynomials from ℝ × ℝ to ℝ × ℝ
def RealPoly := (ℝ × ℝ) → (ℝ × ℝ)

-- State the theorem
theorem constant_polynomial (P : RealPoly) 
  (h : ∀ (x y : ℝ), P (x, y) = P (x + y, x - y)) :
  ∃ (a b : ℝ), ∀ (x y : ℝ), P (x, y) = (a, b) := by
  sorry

end NUMINAMATH_CALUDE_constant_polynomial_l3192_319209


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l3192_319286

theorem unique_function_satisfying_equation :
  ∃! f : ℤ → ℝ, (∀ x y z : ℤ, f (x * y) + f (x * z) - f x * f (y * z) = 1) ∧
                 (∀ x : ℤ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l3192_319286


namespace NUMINAMATH_CALUDE_part1_part2_l3192_319262

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + |x - b|

-- Part 1
theorem part1 (a b c : ℝ) (h : |a - b| > c) : ∀ x : ℝ, f x a b > c := by sorry

-- Part 2
theorem part2 (a : ℝ) :
  (∃ x : ℝ, f x a 1 < 2 - |a - 2|) ↔ (1/2 < a ∧ a < 5/2) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3192_319262


namespace NUMINAMATH_CALUDE_problem_grid_triangles_l3192_319273

/-- Represents a triangular grid with a given number of rows -/
structure TriangularGrid where
  rows : ℕ

/-- Calculates the total number of triangles in a triangular grid -/
def totalTriangles (grid : TriangularGrid) : ℕ :=
  sorry

/-- The specific triangular grid described in the problem -/
def problemGrid : TriangularGrid :=
  { rows := 4 }

theorem problem_grid_triangles :
  totalTriangles problemGrid = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_grid_triangles_l3192_319273


namespace NUMINAMATH_CALUDE_jenny_lasagna_sales_l3192_319294

/-- The number of pans of lasagna Jenny makes and sells -/
def num_pans : ℕ := 20

/-- The cost to make each pan of lasagna -/
def cost_per_pan : ℚ := 10

/-- The selling price of each pan of lasagna -/
def price_per_pan : ℚ := 25

/-- The profit after expenses -/
def profit : ℚ := 300

/-- Theorem stating that the number of pans sold is correct given the conditions -/
theorem jenny_lasagna_sales : 
  (price_per_pan - cost_per_pan) * num_pans = profit := by sorry

end NUMINAMATH_CALUDE_jenny_lasagna_sales_l3192_319294


namespace NUMINAMATH_CALUDE_cos_negative_330_degrees_l3192_319261

theorem cos_negative_330_degrees : Real.cos (-(330 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_330_degrees_l3192_319261


namespace NUMINAMATH_CALUDE_ellipse_properties_l3192_319249

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := 16 * x^2 + 25 * y^2 = 400

/-- Theorem about the properties of the ellipse -/
theorem ellipse_properties :
  ∃ (a b c : ℝ),
    a = 5 ∧ b = 4 ∧ c = 3 ∧
    (∀ x y, is_ellipse x y →
      (2 * a = 10 ∧ b = 4) ∧
      (is_ellipse (-c) 0 ∧ is_ellipse c 0) ∧
      (is_ellipse (-a) 0 ∧ is_ellipse a 0 ∧ is_ellipse 0 b ∧ is_ellipse 0 (-b))) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3192_319249
