import Mathlib

namespace NUMINAMATH_CALUDE_budget_allocation_l1885_188582

def budget : ℝ := 1000

def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.20
def transportation_percentage : ℝ := 0.10
def clothes_percentage : ℝ := 0.05

def coursework_percentage : ℝ :=
  1 - (food_percentage + accommodation_percentage + entertainment_percentage + transportation_percentage + clothes_percentage)

def combined_percentage : ℝ :=
  entertainment_percentage + transportation_percentage + coursework_percentage

def combined_amount : ℝ :=
  combined_percentage * budget

theorem budget_allocation :
  combined_percentage = 0.50 ∧ combined_amount = 500 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_l1885_188582


namespace NUMINAMATH_CALUDE_largest_negative_integer_negation_l1885_188518

theorem largest_negative_integer_negation (x : ℤ) : 
  (∀ y : ℤ, y < 0 → y ≤ x) ∧ x < 0 → -(-(-x)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_negation_l1885_188518


namespace NUMINAMATH_CALUDE_fraction_simplification_l1885_188537

theorem fraction_simplification (a b m : ℝ) (h : a + b ≠ 0) :
  (m * a) / (a + b) + (m * b) / (a + b) = m := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1885_188537


namespace NUMINAMATH_CALUDE_complex_square_equality_l1885_188523

theorem complex_square_equality (c d : ℕ+) :
  (c + d * Complex.I) ^ 2 = 7 + 24 * Complex.I →
  c + d * Complex.I = 4 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equality_l1885_188523


namespace NUMINAMATH_CALUDE_boat_production_l1885_188568

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem boat_production : geometric_sum 5 3 4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_boat_production_l1885_188568


namespace NUMINAMATH_CALUDE_abc_fraction_l1885_188545

theorem abc_fraction (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a * b / (a + b) = 3)
  (hbc : b * c / (b + c) = 4)
  (hca : c * a / (c + a) = 5) :
  a * b * c / (a * b + b * c + c * a) = 120 / 47 := by
sorry

end NUMINAMATH_CALUDE_abc_fraction_l1885_188545


namespace NUMINAMATH_CALUDE_water_formed_is_zero_l1885_188506

-- Define the chemical compounds
inductive Compound
| NH4Cl
| NaOH
| BaNO3_2
| NH4OH
| NaCl
| HNO3
| NH4NO3
| H2O
| NaNO3
| BaCl2

-- Define a reaction
structure Reaction :=
(reactants : List (Compound × ℕ))
(products : List (Compound × ℕ))

-- Define the given reactions
def reaction1 : Reaction :=
{ reactants := [(Compound.NH4Cl, 1), (Compound.NaOH, 1)]
, products := [(Compound.NH4OH, 1), (Compound.NaCl, 1)] }

def reaction2 : Reaction :=
{ reactants := [(Compound.NH4OH, 1), (Compound.HNO3, 1)]
, products := [(Compound.NH4NO3, 1), (Compound.H2O, 1)] }

def reaction3 : Reaction :=
{ reactants := [(Compound.BaNO3_2, 1), (Compound.NaCl, 2)]
, products := [(Compound.NaNO3, 2), (Compound.BaCl2, 1)] }

-- Define the initial reactants
def initialReactants : List (Compound × ℕ) :=
[(Compound.NH4Cl, 3), (Compound.NaOH, 3), (Compound.BaNO3_2, 2)]

-- Define a function to calculate the moles of water formed
def molesOfWaterFormed (initialReactants : List (Compound × ℕ)) 
                       (reactions : List Reaction) : ℕ :=
  sorry

-- Theorem statement
theorem water_formed_is_zero :
  molesOfWaterFormed initialReactants [reaction1, reaction2, reaction3] = 0 :=
sorry

end NUMINAMATH_CALUDE_water_formed_is_zero_l1885_188506


namespace NUMINAMATH_CALUDE_exists_fourth_power_product_l1885_188513

def is_not_divisible_by_primes_greater_than_28 (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p > 28 → ¬(p ∣ n)

theorem exists_fourth_power_product 
  (M : Finset ℕ) 
  (h_card : M.card = 2008) 
  (h_distinct : M.card = Finset.card (M.image id))
  (h_positive : ∀ n ∈ M, n > 0)
  (h_not_div : ∀ n ∈ M, is_not_divisible_by_primes_greater_than_28 n) :
  ∃ a b c d : ℕ, a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ k : ℕ, a * b * c * d = k^4 :=
sorry

end NUMINAMATH_CALUDE_exists_fourth_power_product_l1885_188513


namespace NUMINAMATH_CALUDE_place_value_ratio_in_53687_4921_l1885_188567

/-- The place value of a digit in a decimal number -/
def place_value (digit_position : Int) : ℚ :=
  10 ^ digit_position

/-- The position of a digit in a decimal number, counting from right to left,
    with the decimal point at position 0 -/
def digit_position (n : ℚ) (d : ℕ) : Int :=
  sorry

theorem place_value_ratio_in_53687_4921 :
  let n : ℚ := 53687.4921
  let pos_8 := digit_position n 8
  let pos_2 := digit_position n 2
  place_value pos_8 / place_value pos_2 = 1000 := by sorry

end NUMINAMATH_CALUDE_place_value_ratio_in_53687_4921_l1885_188567


namespace NUMINAMATH_CALUDE_simplification_problems_l1885_188590

theorem simplification_problems :
  ((-1/2 + 2/3 - 1/4) / (-1/24) = 2) ∧
  (7/2 * (-5/7) - (-5/7) * 5/2 - 5/7 * (-1/2) = -5/14) := by
  sorry

end NUMINAMATH_CALUDE_simplification_problems_l1885_188590


namespace NUMINAMATH_CALUDE_james_old_wage_l1885_188574

/-- Jame's old hourly wage -/
def old_wage : ℝ := 16

/-- Jame's new hourly wage -/
def new_wage : ℝ := 20

/-- Jame's old weekly work hours -/
def old_hours : ℝ := 25

/-- Jame's new weekly work hours -/
def new_hours : ℝ := 40

/-- Number of weeks worked per year -/
def weeks_per_year : ℝ := 52

/-- Difference in annual earnings between new and old job -/
def annual_difference : ℝ := 20800

theorem james_old_wage :
  old_wage * old_hours * weeks_per_year + annual_difference = new_wage * new_hours * weeks_per_year :=
by sorry

end NUMINAMATH_CALUDE_james_old_wage_l1885_188574


namespace NUMINAMATH_CALUDE_wall_length_proof_l1885_188557

/-- Given a wall with specified dimensions and number of bricks, prove its length --/
theorem wall_length_proof (wall_height : ℝ) (wall_thickness : ℝ) 
  (brick_count : ℝ) (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) :
  wall_height = 100 →
  wall_thickness = 5 →
  brick_count = 242.42424242424244 →
  brick_length = 25 →
  brick_width = 11 →
  brick_height = 6 →
  (brick_length * brick_width * brick_height * brick_count) / (wall_height * wall_thickness) = 800 := by
  sorry

#check wall_length_proof

end NUMINAMATH_CALUDE_wall_length_proof_l1885_188557


namespace NUMINAMATH_CALUDE_max_product_constraint_l1885_188565

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 * b = 1) :
  a * b ≤ 1 / 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 4 * b₀ = 1 ∧ a₀ * b₀ = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_max_product_constraint_l1885_188565


namespace NUMINAMATH_CALUDE_three_lines_equidistant_l1885_188542

/-- A line in a plane --/
structure Line where
  -- Add necessary fields for a line

/-- Distance between a point and a line --/
def distance_point_line (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

theorem three_lines_equidistant (A B : ℝ × ℝ) (h : dist A B = 5) :
  ∃! (s : Finset Line), s.card = 3 ∧ 
    (∀ l ∈ s, distance_point_line A l = 2 ∧ distance_point_line B l = 3) :=
sorry

end NUMINAMATH_CALUDE_three_lines_equidistant_l1885_188542


namespace NUMINAMATH_CALUDE_retirement_percentage_l1885_188519

def gross_pay : ℝ := 1120
def tax_deduction : ℝ := 100
def net_pay : ℝ := 740

theorem retirement_percentage :
  (gross_pay - net_pay - tax_deduction) / gross_pay * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_retirement_percentage_l1885_188519


namespace NUMINAMATH_CALUDE_f_properties_l1885_188559

noncomputable def f (x : ℝ) := 6 * (Real.cos x)^2 - Real.sqrt 3 * Real.sin (2 * x)

theorem f_properties :
  (∃ (max_value : ℝ), ∀ (x : ℝ), f x ≤ max_value ∧ max_value = 2 * Real.sqrt 3 + 3) ∧
  (∃ (period : ℝ), period > 0 ∧ ∀ (x : ℝ), f (x + period) = f x ∧ 
    ∀ (p : ℝ), p > 0 → (∀ (x : ℝ), f (x + p) = f x) → period ≤ p) ∧
  (∀ (α : ℝ), 0 < α ∧ α < Real.pi / 2 → 
    f α = 3 - 2 * Real.sqrt 3 → Real.tan (4 * α / 5) = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1885_188559


namespace NUMINAMATH_CALUDE_compare_roots_l1885_188529

theorem compare_roots : (2 * Real.sqrt 6 < 5) ∧ (-Real.sqrt 5 < -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_compare_roots_l1885_188529


namespace NUMINAMATH_CALUDE_equation_roots_l1885_188536

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => (18 / (x^2 - 9)) - (3 / (x - 3)) - 2
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -4.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l1885_188536


namespace NUMINAMATH_CALUDE_f_minimum_value_l1885_188589

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x^2 + 1/(x^2 + 1/x^2)

theorem f_minimum_value :
  (∀ x > 0, f x ≥ 5/2) ∧ (∃ x > 0, f x = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1885_188589


namespace NUMINAMATH_CALUDE_cats_left_after_sale_l1885_188539

/-- Calculates the number of cats left after a sale --/
theorem cats_left_after_sale (siamese house persian maine_coon : ℕ)
  (siamese_sold house_sold persian_sold maine_coon_sold : ℚ)
  (h_siamese : siamese = 38)
  (h_house : house = 25)
  (h_persian : persian = 15)
  (h_maine_coon : maine_coon = 12)
  (h_siamese_sold : siamese_sold = 60 / 100)
  (h_house_sold : house_sold = 40 / 100)
  (h_persian_sold : persian_sold = 75 / 100)
  (h_maine_coon_sold : maine_coon_sold = 50 / 100) :
  ⌊siamese - siamese * siamese_sold⌋ +
  ⌊house - house * house_sold⌋ +
  ⌊persian - persian * persian_sold⌋ +
  ⌊maine_coon - maine_coon * maine_coon_sold⌋ = 41 := by
  sorry


end NUMINAMATH_CALUDE_cats_left_after_sale_l1885_188539


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1885_188581

theorem unique_solution_for_equation : ∃! (n : ℕ), 
  ∃ (x : ℕ), x > 0 ∧ 
  n = 2^(2*x - 1) - 5*x - 3 ∧
  n = (2^(x-1) - 1) * (2^x + 1) ∧
  n = 2015 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1885_188581


namespace NUMINAMATH_CALUDE_interior_angles_theorem_l1885_188504

/-- The sum of interior angles of a convex polygon with n sides, in degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: Given a convex polygon with n sides where the sum of interior angles is 3600 degrees,
    the sum of interior angles for a polygon with n+2 sides is 3960 degrees,
    and for a polygon with n-2 sides is 3240 degrees. -/
theorem interior_angles_theorem (n : ℕ) (h : sum_interior_angles n = 3600) :
  sum_interior_angles (n + 2) = 3960 ∧ sum_interior_angles (n - 2) = 3240 := by
  sorry

#check interior_angles_theorem

end NUMINAMATH_CALUDE_interior_angles_theorem_l1885_188504


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l1885_188576

theorem chocolate_bars_count (bar_price : ℕ) (remaining_bars : ℕ) (total_sales : ℕ) : 
  bar_price = 6 →
  remaining_bars = 6 →
  total_sales = 42 →
  ∃ (total_bars : ℕ), total_bars = 13 ∧ bar_price * (total_bars - remaining_bars) = total_sales :=
by sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l1885_188576


namespace NUMINAMATH_CALUDE_units_digit_17_pow_2023_l1885_188561

theorem units_digit_17_pow_2023 : ∃ k : ℕ, 17^2023 ≡ 3 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_2023_l1885_188561


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1885_188515

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  num_terms : ℕ
  common_diff : ℝ

/-- Theorem stating the properties of the specific arithmetic sequence -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h1 : seq.first_term = 3)
  (h2 : seq.last_term = 50)
  (h3 : seq.sum = 318) :
  seq.common_diff = 47 / 11 := by
  sorry

#check arithmetic_sequence_property

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1885_188515


namespace NUMINAMATH_CALUDE_legs_fraction_of_height_l1885_188579

/-- Represents the height measurements of a person --/
structure PersonHeight where
  total : ℝ
  head : ℝ
  restOfBody : ℝ

/-- Theorem stating the fraction of total height occupied by legs --/
theorem legs_fraction_of_height (p : PersonHeight) 
  (h_total : p.total = 60)
  (h_head : p.head = 1/4 * p.total)
  (h_rest : p.restOfBody = 25) :
  (p.total - p.head - p.restOfBody) / p.total = 1/3 := by
  sorry

#check legs_fraction_of_height

end NUMINAMATH_CALUDE_legs_fraction_of_height_l1885_188579


namespace NUMINAMATH_CALUDE_unique_k_value_l1885_188580

/-- A predicate to check if a number is a non-zero digit -/
def is_nonzero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- The expression as a function of k and t -/
def expression (k t : ℕ) : ℤ := 8 * k * 100 + 8 + k * 100 + 88 - 16 * t * 10 - 6

theorem unique_k_value :
  ∀ k t : ℕ,
  is_nonzero_digit k →
  is_nonzero_digit t →
  t = 6 →
  (∃ m : ℤ, expression k t = m) →
  k = 9 := by sorry

end NUMINAMATH_CALUDE_unique_k_value_l1885_188580


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1885_188566

/-- The cubic polynomial f(x) = x^3 + 9x^2 + 26x + 24 -/
def f (x : ℝ) : ℝ := x^3 + 9*x^2 + 26*x + 24

/-- The set of roots of f -/
def roots : Set ℝ := {x | f x = 0}

theorem cubic_equation_roots :
  ∃ (r₁ r₂ r₃ : ℝ), r₁ > 0 ∧ r₂ < 0 ∧ r₃ < 0 ∧
  roots = {r₁, r₂, r₃} :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1885_188566


namespace NUMINAMATH_CALUDE_remainder_eleven_power_2023_mod_13_l1885_188572

theorem remainder_eleven_power_2023_mod_13 : 11^2023 % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eleven_power_2023_mod_13_l1885_188572


namespace NUMINAMATH_CALUDE_num_divisors_10_factorial_l1885_188548

/-- The number of positive divisors of n! -/
def numDivisorsFactorial (n : ℕ) : ℕ := sorry

/-- Theorem: The number of positive divisors of 10! is 192 -/
theorem num_divisors_10_factorial :
  numDivisorsFactorial 10 = 192 := by sorry

end NUMINAMATH_CALUDE_num_divisors_10_factorial_l1885_188548


namespace NUMINAMATH_CALUDE_lcm_problem_l1885_188564

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 30 m = 90) 
  (h2 : Nat.lcm m 50 = 200) : 
  m = 10 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1885_188564


namespace NUMINAMATH_CALUDE_sum_ways_2002_l1885_188591

/-- The number of ways to express 2002 as the sum of 3 positive integers, without considering order -/
def ways_to_sum_2002 : ℕ := 334000

/-- A function that counts the number of ways to express a given natural number as the sum of 3 positive integers, without considering order -/
def count_sum_ways (n : ℕ) : ℕ :=
  sorry

theorem sum_ways_2002 : count_sum_ways 2002 = ways_to_sum_2002 := by
  sorry

end NUMINAMATH_CALUDE_sum_ways_2002_l1885_188591


namespace NUMINAMATH_CALUDE_complex_function_property_l1885_188547

theorem complex_function_property (a b : ℝ) :
  (∀ z : ℂ, Complex.abs ((a + b * Complex.I) * z^2 - z) = Complex.abs ((a + b * Complex.I) * z^2)) →
  Complex.abs (a + b * Complex.I) = 5 →
  b^2 = 99/4 := by sorry

end NUMINAMATH_CALUDE_complex_function_property_l1885_188547


namespace NUMINAMATH_CALUDE_function_count_l1885_188571

theorem function_count (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x * y) + f x + f y - f x * f y ≥ 2) ↔ 
  (∀ x : ℝ, f x = 1 ∨ f x = 2) :=
sorry

end NUMINAMATH_CALUDE_function_count_l1885_188571


namespace NUMINAMATH_CALUDE_factors_of_48_l1885_188587

/-- The number of distinct positive factors of 48 -/
def num_factors_48 : ℕ := (Nat.factors 48).card

/-- Theorem stating that the number of distinct positive factors of 48 is 10 -/
theorem factors_of_48 : num_factors_48 = 10 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_48_l1885_188587


namespace NUMINAMATH_CALUDE_log_one_over_twenty_five_base_five_l1885_188592

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_over_twenty_five_base_five : log 5 (1/25) = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_one_over_twenty_five_base_five_l1885_188592


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1885_188509

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b + t.c = 180

-- Define the specific conditions of our triangle
def our_triangle (t : Triangle) : Prop :=
  is_valid_triangle t ∧
  t.a = 70 ∧
  t.b = 40 ∧
  t.c = 70

-- Theorem statement
theorem triangle_angle_sum (t : Triangle) :
  our_triangle t → t.c = 40 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1885_188509


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l1885_188511

theorem complex_number_coordinates : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l1885_188511


namespace NUMINAMATH_CALUDE_function_value_order_l1885_188577

-- Define the function f
def f (a x : ℝ) : ℝ := -x^2 + 6*x + a^2 - 1

-- State the theorem
theorem function_value_order (a : ℝ) : 
  f a (Real.sqrt 2) < f a 4 ∧ f a 4 < f a 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_order_l1885_188577


namespace NUMINAMATH_CALUDE_map_width_l1885_188532

/-- The width of a rectangular map given its length and area -/
theorem map_width (length : ℝ) (area : ℝ) (h1 : length = 2) (h2 : area = 20) :
  area / length = 10 := by
  sorry

end NUMINAMATH_CALUDE_map_width_l1885_188532


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l1885_188593

theorem smallest_prime_dividing_sum : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^11 + 5^13) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (3^11 + 5^13) → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l1885_188593


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1885_188503

theorem inequality_equivalence (m : ℝ) : (3 * m - 4 < 6) ↔ (m < 6) := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1885_188503


namespace NUMINAMATH_CALUDE_production_line_uses_systematic_sampling_l1885_188556

/-- Represents different sampling methods --/
inductive SamplingMethod
  | Systematic
  | Random
  | Stratified
  | Cluster

/-- Represents a production line with its characteristics --/
structure ProductionLine where
  daily_production : ℕ
  sampling_frequency : ℕ  -- days per week
  samples_per_day : ℕ
  sampling_start_time : ℕ  -- in minutes past midnight
  sampling_end_time : ℕ    -- in minutes past midnight

/-- Determines the sampling method based on production line characteristics --/
def determine_sampling_method (pl : ProductionLine) : SamplingMethod :=
  sorry  -- Proof to be implemented

/-- Theorem stating that the given production line uses systematic sampling --/
theorem production_line_uses_systematic_sampling (pl : ProductionLine) 
  (h1 : pl.daily_production = 128)
  (h2 : pl.sampling_frequency = 7)  -- weekly
  (h3 : pl.samples_per_day = 8)
  (h4 : pl.sampling_start_time = 14 * 60)  -- 2:00 PM
  (h5 : pl.sampling_end_time = 14 * 60 + 30)  -- 2:30 PM
  : determine_sampling_method pl = SamplingMethod.Systematic :=
by
  sorry  -- Proof to be implemented

end NUMINAMATH_CALUDE_production_line_uses_systematic_sampling_l1885_188556


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l1885_188526

/-- Two lines are parallel if their slopes are equal or if they are both vertical -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 = 0 ∧ a2 = 0) ∨ (b1 = 0 ∧ b2 = 0) ∨ (a1 * b2 = a2 * b1 ∧ a1 ≠ 0 ∧ a2 ≠ 0)

/-- The statement to be proved -/
theorem parallel_lines_m_values (m : ℝ) :
  are_parallel (m - 2) (-1) 5 (m - 2) (3 - m) 2 → m = 2 ∨ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l1885_188526


namespace NUMINAMATH_CALUDE_shortest_to_longest_diagonal_ratio_l1885_188550

/-- A regular octagon -/
structure RegularOctagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The shortest diagonal of a regular octagon -/
def shortest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- The longest diagonal of a regular octagon -/
def longest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- The ratio of the shortest diagonal to the longest diagonal in a regular octagon is 1/2 -/
theorem shortest_to_longest_diagonal_ratio (o : RegularOctagon) :
  shortest_diagonal o / longest_diagonal o = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_shortest_to_longest_diagonal_ratio_l1885_188550


namespace NUMINAMATH_CALUDE_percentage_increase_l1885_188599

theorem percentage_increase (original_earnings new_earnings : ℝ) 
  (h1 : original_earnings = 60)
  (h2 : new_earnings = 78) : 
  (new_earnings - original_earnings) / original_earnings * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1885_188599


namespace NUMINAMATH_CALUDE_apple_tree_production_decrease_l1885_188594

theorem apple_tree_production_decrease (season1 season2 season3 total : ℕ) : 
  season1 = 200 →
  season3 = 2 * season2 →
  total = season1 + season2 + season3 →
  total = 680 →
  (season1 - season2 : ℚ) / season1 = 1/5 := by sorry

end NUMINAMATH_CALUDE_apple_tree_production_decrease_l1885_188594


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l1885_188541

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for part I
theorem solution_set_f (x : ℝ) : f x < 8 ↔ -5/2 < x ∧ x < 3/2 :=
sorry

-- Theorem for part II
theorem range_of_m (m : ℝ) : (∃ x, f x ≤ |3*m + 1|) → (m ≤ -5/3 ∨ m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l1885_188541


namespace NUMINAMATH_CALUDE_infinite_occurrences_l1885_188527

-- Define the sequence
def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 
    let prev := a n
    let god := (n + 1).factorization.prod (λ p k => p)  -- greatest odd divisor
    if god % 4 = 1 then prev + 1 else prev - 1

-- State the theorem
theorem infinite_occurrences :
  (∀ k : ℕ+, Set.Infinite {n : ℕ | a n = k}) ∧
  Set.Infinite {n : ℕ | a n = 1} := by
  sorry

end NUMINAMATH_CALUDE_infinite_occurrences_l1885_188527


namespace NUMINAMATH_CALUDE_distinct_d_values_l1885_188555

theorem distinct_d_values (a b c : ℂ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃! (s : Finset ℂ), s.card = 6 ∧ 
  (∀ d : ℂ, d ∈ s ↔ 
    (∀ z : ℂ, (z - a) * (z - b) * (z - c) = (z - d^2 * a) * (z - d^2 * b) * (z - d^2 * c))) :=
by sorry

end NUMINAMATH_CALUDE_distinct_d_values_l1885_188555


namespace NUMINAMATH_CALUDE_fraction_equality_l1885_188533

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 2) 
  (h2 : s / u = 7 / 11) : 
  (5 * p * s - 3 * q * u) / (7 * q * u - 4 * p * s) = 109 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1885_188533


namespace NUMINAMATH_CALUDE_remainder_problem_l1885_188540

theorem remainder_problem (k : ℕ+) (h : 120 % (k^2 : ℕ) = 8) : 150 % (k : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1885_188540


namespace NUMINAMATH_CALUDE_double_series_convergence_l1885_188535

/-- The double series ∑_{m=1}^∞ ∑_{n=1}^∞ 1/(mn(m+n+2)) converges to 3/2. -/
theorem double_series_convergence :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * n * (m + n + 2))) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_double_series_convergence_l1885_188535


namespace NUMINAMATH_CALUDE_inscribed_circle_circumference_l1885_188530

/-- Given a circle with radius R and an arc subtending 120°, 
    the radius r of the circle inscribed between this arc and its tangents 
    satisfies 2πr = (2πR)/3 -/
theorem inscribed_circle_circumference (R r : ℝ) : r = R / 3 → 2 * π * r = 2 * π * R / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_circumference_l1885_188530


namespace NUMINAMATH_CALUDE_batsman_average_l1885_188508

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat

/-- Calculates the average score of a batsman -/
def average (b : Batsman) : Nat :=
  b.totalRuns / b.innings

/-- Theorem: Given the conditions, prove that the batsman's average after 10 innings is 33 -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 10)
  (h2 : b.totalRuns = (average b * 9) + 60)
  (h3 : average { innings := b.innings, totalRuns := b.totalRuns, averageIncrease := b.averageIncrease } = 
        average { innings := b.innings - 1, totalRuns := b.totalRuns - 60, averageIncrease := b.averageIncrease } + 3) :
  average b = 33 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_l1885_188508


namespace NUMINAMATH_CALUDE_quadratic_sum_equals_27_l1885_188510

theorem quadratic_sum_equals_27 (m n : ℝ) (h : m + n = 4) : 
  2 * m^2 + 4 * m * n + 2 * n^2 - 5 = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_equals_27_l1885_188510


namespace NUMINAMATH_CALUDE_equation_solutions_l1885_188588

theorem equation_solutions : 
  let solutions := {x : ℝ | (x - 1)^2 = 4}
  solutions = {3, -1} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1885_188588


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1885_188517

theorem fraction_product_simplification :
  (21 : ℚ) / 28 * 14 / 33 * 99 / 42 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1885_188517


namespace NUMINAMATH_CALUDE_min_distance_hyperbola_circle_l1885_188578

theorem min_distance_hyperbola_circle (a b c d : ℝ) 
  (h1 : a * b = 1) (h2 : c^2 + d^2 = 1) : 
  ∃ (min : ℝ), min = 3 - 2 * Real.sqrt 2 ∧ 
  ∀ (x y z w : ℝ), x * y = 1 → z^2 + w^2 = 1 → 
  (x - z)^2 + (y - w)^2 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_distance_hyperbola_circle_l1885_188578


namespace NUMINAMATH_CALUDE_letters_theorem_l1885_188544

def total_letters (brother_letters : ℕ) : ℕ :=
  let greta_letters := brother_letters + 10
  let mother_letters := 2 * (brother_letters + greta_letters)
  brother_letters + greta_letters + mother_letters

theorem letters_theorem : total_letters 40 = 270 := by
  sorry

end NUMINAMATH_CALUDE_letters_theorem_l1885_188544


namespace NUMINAMATH_CALUDE_triangle_inequality_l1885_188521

/-- Theorem: For any triangle with side lengths a, b, c, and area S,
    the inequality a² + b² + c² ≥ 4√3 S holds, with equality if and only if
    the triangle is equilateral. -/
theorem triangle_inequality (a b c S : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0)
    (h_area_def : S = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
    (h_s_def : s = (a + b + c) / 2) :
    a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S ∧
    (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * S ↔ a = b ∧ b = c) :=
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1885_188521


namespace NUMINAMATH_CALUDE_gcf_78_104_l1885_188546

theorem gcf_78_104 : Nat.gcd 78 104 = 26 := by
  sorry

end NUMINAMATH_CALUDE_gcf_78_104_l1885_188546


namespace NUMINAMATH_CALUDE_salary_calculation_l1885_188584

theorem salary_calculation (salary : ℝ) 
  (h1 : salary / 5 + salary / 10 + 3 * salary / 5 + 14000 = salary) : 
  salary = 140000 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l1885_188584


namespace NUMINAMATH_CALUDE_euclids_lemma_l1885_188596

theorem euclids_lemma (p a b : ℕ) (hp : Prime p) (hab : p ∣ a * b) : p ∣ a ∨ p ∣ b := by
  sorry

-- Gauss's lemma (given)
axiom gauss_lemma (p a b : ℕ) (hp : Prime p) (hab : p ∣ a * b) (hna : ¬(p ∣ a)) : p ∣ b

end NUMINAMATH_CALUDE_euclids_lemma_l1885_188596


namespace NUMINAMATH_CALUDE_smallest_prime_in_sum_l1885_188524

theorem smallest_prime_in_sum (p q r s : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → Nat.Prime s →
  p + q + r = 2 * s →
  1 < p → p < q → q < r →
  p = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_in_sum_l1885_188524


namespace NUMINAMATH_CALUDE_max_rectangular_pen_area_l1885_188516

/-- Given 60 feet of fencing, the maximum area of a rectangular pen is 225 square feet. -/
theorem max_rectangular_pen_area (perimeter : ℝ) (h : perimeter = 60) : 
  ∃ (width height : ℝ), 
    width > 0 ∧ 
    height > 0 ∧ 
    2 * (width + height) = perimeter ∧ 
    ∀ (w h : ℝ), w > 0 → h > 0 → 2 * (w + h) = perimeter → w * h ≤ width * height ∧ 
    width * height = 225 :=
sorry

end NUMINAMATH_CALUDE_max_rectangular_pen_area_l1885_188516


namespace NUMINAMATH_CALUDE_expected_winnings_l1885_188597

/-- Represents the outcome of rolling the die -/
inductive DieOutcome
  | Six
  | Odd
  | Even

/-- The probability of rolling a 6 -/
def prob_six : ℚ := 1/4

/-- The probability of rolling an odd number (1, 3, or 5) -/
def prob_odd : ℚ := (1 - prob_six) * (3/5)

/-- The probability of rolling an even number (2 or 4) -/
def prob_even : ℚ := (1 - prob_six) * (2/5)

/-- The payoff for each outcome -/
def payoff (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Six => -2
  | DieOutcome.Odd => 2
  | DieOutcome.Even => 4

/-- The expected value of rolling the die -/
def expected_value : ℚ :=
  prob_six * payoff DieOutcome.Six +
  prob_odd * payoff DieOutcome.Odd +
  prob_even * payoff DieOutcome.Even

theorem expected_winnings :
  expected_value = 8/5 := by sorry

end NUMINAMATH_CALUDE_expected_winnings_l1885_188597


namespace NUMINAMATH_CALUDE_four_fives_equal_100_l1885_188598

/-- An arithmetic expression using fives -/
inductive FiveExpr
  | Const : FiveExpr
  | Add : FiveExpr → FiveExpr → FiveExpr
  | Sub : FiveExpr → FiveExpr → FiveExpr
  | Mul : FiveExpr → FiveExpr → FiveExpr

/-- Evaluate a FiveExpr to an integer -/
def eval : FiveExpr → Int
  | FiveExpr.Const => 5
  | FiveExpr.Add a b => eval a + eval b
  | FiveExpr.Sub a b => eval a - eval b
  | FiveExpr.Mul a b => eval a * eval b

/-- Count the number of fives in a FiveExpr -/
def countFives : FiveExpr → Nat
  | FiveExpr.Const => 1
  | FiveExpr.Add a b => countFives a + countFives b
  | FiveExpr.Sub a b => countFives a + countFives b
  | FiveExpr.Mul a b => countFives a + countFives b

/-- Theorem: There exists an arithmetic expression using exactly four fives that equals 100 -/
theorem four_fives_equal_100 : ∃ e : FiveExpr, countFives e = 4 ∧ eval e = 100 := by
  sorry


end NUMINAMATH_CALUDE_four_fives_equal_100_l1885_188598


namespace NUMINAMATH_CALUDE_slope_from_angle_l1885_188558

theorem slope_from_angle (θ : Real) (h : θ = 5 * Real.pi / 6) :
  Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_from_angle_l1885_188558


namespace NUMINAMATH_CALUDE_area_equality_l1885_188507

-- Define a square
structure Square :=
  (A B C D : Point)

-- Define the property of being inside a square
def InsideSquare (P : Point) (s : Square) : Prop := sorry

-- Define the angle between three points
def Angle (P Q R : Point) : ℝ := sorry

-- Define the area of a triangle
def TriangleArea (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem area_equality (s : Square) (P Q : Point) 
  (h_inside_P : InsideSquare P s)
  (h_inside_Q : InsideSquare Q s)
  (h_angle_PAQ : Angle s.A P Q = 45)
  (h_angle_PCQ : Angle s.C P Q = 45) :
  TriangleArea P s.A s.B + TriangleArea P s.C Q + TriangleArea Q s.A s.D =
  TriangleArea Q s.C s.D + TriangleArea P s.A Q + TriangleArea P s.B s.C :=
sorry

end NUMINAMATH_CALUDE_area_equality_l1885_188507


namespace NUMINAMATH_CALUDE_equidistant_function_property_l1885_188522

def f (a b : ℝ) (z : ℂ) : ℂ := (Complex.mk a b) * z

theorem equidistant_function_property (a b : ℝ) :
  (∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z)) →
  Complex.abs (Complex.mk a b) = 5 →
  b^2 = 99/4 := by sorry

end NUMINAMATH_CALUDE_equidistant_function_property_l1885_188522


namespace NUMINAMATH_CALUDE_predictor_variable_is_fertilizer_l1885_188534

/-- Represents a variable in the study -/
inductive StudyVariable
  | YieldOfCrops
  | AmountOfFertilizer
  | Experimenter
  | OtherVariables

/-- Defines the characteristics of the study -/
structure CropStudy where
  predictedVariable : StudyVariable
  predictorVariable : StudyVariable
  aim : String

/-- Theorem stating that the predictor variable in the crop yield study is the amount of fertilizer -/
theorem predictor_variable_is_fertilizer (study : CropStudy) :
  study.aim = "determine whether the yield of crops can be predicted based on the amount of fertilizer applied" →
  study.predictedVariable = StudyVariable.YieldOfCrops →
  study.predictorVariable = StudyVariable.AmountOfFertilizer :=
by sorry

end NUMINAMATH_CALUDE_predictor_variable_is_fertilizer_l1885_188534


namespace NUMINAMATH_CALUDE_unique_g_30_equals_48_l1885_188560

def sumOfDivisors (n : ℕ) : ℕ := sorry

def g₁ (n : ℕ) : ℕ := 4 * sumOfDivisors n

def g (j n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | j+1 => g₁ (g j n)

theorem unique_g_30_equals_48 :
  ∃! n : ℕ, n ≤ 30 ∧ g 30 n = 48 := by sorry

end NUMINAMATH_CALUDE_unique_g_30_equals_48_l1885_188560


namespace NUMINAMATH_CALUDE_net_effect_on_sale_value_l1885_188505

/-- Theorem: Net effect on sale value after price reduction and sales increase -/
theorem net_effect_on_sale_value 
  (price_reduction : Real) 
  (sales_increase : Real) 
  (h1 : price_reduction = 0.25)
  (h2 : sales_increase = 0.75) : 
  (1 - price_reduction) * (1 + sales_increase) - 1 = 0.3125 := by
  sorry

#eval (1 - 0.25) * (1 + 0.75) - 1

end NUMINAMATH_CALUDE_net_effect_on_sale_value_l1885_188505


namespace NUMINAMATH_CALUDE_calculate_hourly_wage_l1885_188549

/-- Calculates the hourly wage of a worker given their work conditions and pay --/
theorem calculate_hourly_wage (hours_per_week : ℕ) (deduction_per_lateness : ℕ) 
  (lateness_count : ℕ) (pay_after_deductions : ℕ) : 
  hours_per_week = 18 → 
  deduction_per_lateness = 5 → 
  lateness_count = 3 → 
  pay_after_deductions = 525 → 
  (pay_after_deductions + lateness_count * deduction_per_lateness) / hours_per_week = 30 := by
  sorry

end NUMINAMATH_CALUDE_calculate_hourly_wage_l1885_188549


namespace NUMINAMATH_CALUDE_books_left_over_l1885_188586

/-- Given a repacking scenario, proves the number of books left over -/
theorem books_left_over 
  (initial_boxes : ℕ) 
  (books_per_initial_box : ℕ) 
  (book_weight : ℕ) 
  (books_per_new_box : ℕ) 
  (max_new_box_weight : ℕ) 
  (h1 : initial_boxes = 1430)
  (h2 : books_per_initial_box = 42)
  (h3 : book_weight = 200)
  (h4 : books_per_new_box = 45)
  (h5 : max_new_box_weight = 9000)
  (h6 : books_per_new_box * book_weight ≤ max_new_box_weight) :
  (initial_boxes * books_per_initial_box) % books_per_new_box = 30 := by
  sorry

#check books_left_over

end NUMINAMATH_CALUDE_books_left_over_l1885_188586


namespace NUMINAMATH_CALUDE_tylers_puppies_l1885_188585

theorem tylers_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) (h1 : num_dogs = 25) (h2 : puppies_per_dog = 8) :
  num_dogs * puppies_per_dog = 200 := by
  sorry

end NUMINAMATH_CALUDE_tylers_puppies_l1885_188585


namespace NUMINAMATH_CALUDE_triangle_area_inequalities_l1885_188552

/-- Given two triangles and a third triangle constructed from their sides, 
    the area of the third triangle is greater than or equal to 
    both the geometric and arithmetic means of the areas of the original triangles. -/
theorem triangle_area_inequalities 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁)
  (h₂ : 0 < a₂ ∧ 0 < b₂ ∧ 0 < c₂)
  (h₃ : a₁ + b₁ > c₁ ∧ a₁ + c₁ > b₁ ∧ b₁ + c₁ > a₁)
  (h₄ : a₂ + b₂ > c₂ ∧ a₂ + c₂ > b₂ ∧ b₂ + c₂ > a₂)
  (a : ℝ) (ha : a = Real.sqrt ((a₁^2 + a₂^2) / 2))
  (b : ℝ) (hb : b = Real.sqrt ((b₁^2 + b₂^2) / 2))
  (c : ℝ) (hc : c = Real.sqrt ((c₁^2 + c₂^2) / 2))
  (h₅ : a + b > c ∧ a + c > b ∧ b + c > a)
  (S₁ : ℝ) (hS₁ : S₁ = Real.sqrt (s₁ * (s₁ - a₁) * (s₁ - b₁) * (s₁ - c₁)))
  (S₂ : ℝ) (hS₂ : S₂ = Real.sqrt (s₂ * (s₂ - a₂) * (s₂ - b₂) * (s₂ - c₂)))
  (S : ℝ)  (hS : S = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
  (s₁ : ℝ) (hs₁ : s₁ = (a₁ + b₁ + c₁) / 2)
  (s₂ : ℝ) (hs₂ : s₂ = (a₂ + b₂ + c₂) / 2)
  (s : ℝ)  (hs : s = (a + b + c) / 2) :
  S ≥ Real.sqrt (S₁ * S₂) ∧ S ≥ (S₁ + S₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_inequalities_l1885_188552


namespace NUMINAMATH_CALUDE_tan_80_plus_tan_40_minus_sqrt3_tan_80_tan_40_l1885_188502

theorem tan_80_plus_tan_40_minus_sqrt3_tan_80_tan_40 :
  let t80 := Real.tan (80 * π / 180)
  let t40 := Real.tan (40 * π / 180)
  t80 + t40 - Real.sqrt 3 * t80 * t40 = -Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_80_plus_tan_40_minus_sqrt3_tan_80_tan_40_l1885_188502


namespace NUMINAMATH_CALUDE_points_in_segment_l1885_188583

theorem points_in_segment (n : ℕ) : 
  1 < (n^4 + n^2 + 2) / (n^4 + n^2 + 1) ∧ (n^4 + n^2 + 2) / (n^4 + n^2 + 1) ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_points_in_segment_l1885_188583


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1885_188562

theorem complex_fraction_equality : (10 * Complex.I) / (2 - Complex.I) = -2 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1885_188562


namespace NUMINAMATH_CALUDE_consecutive_sets_summing_to_150_l1885_188551

/-- A structure representing a set of consecutive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum_is_150 : start * length + (length * (length - 1)) / 2 = 150
  at_least_two : length ≥ 2

/-- The theorem stating that there are exactly 3 sets of consecutive positive integers summing to 150 -/
theorem consecutive_sets_summing_to_150 : 
  ∃! (sets : Finset ConsecutiveSet), sets.card = 3 ∧ 
    (∀ s ∈ sets, s.start > 0 ∧ s.length ≥ 2 ∧ 
      s.start * s.length + (s.length * (s.length - 1)) / 2 = 150) ∧
    (∀ a b : ℕ, a > 0 → b ≥ 2 → 
      (a * b + (b * (b - 1)) / 2 = 150 → ∃ s ∈ sets, s.start = a ∧ s.length = b)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_sets_summing_to_150_l1885_188551


namespace NUMINAMATH_CALUDE_average_playing_time_is_ten_l1885_188512

/-- Represents the number of players --/
def num_players : ℕ := 8

/-- Represents the start time in hours since midnight --/
def start_time : ℕ := 8

/-- Represents the end time in hours since midnight --/
def end_time : ℕ := 18

/-- Represents the number of chess games being played simultaneously --/
def num_games : ℕ := 2

/-- Calculates the average playing time per person --/
def average_playing_time : ℚ :=
  (end_time - start_time : ℚ) * num_games / num_players

theorem average_playing_time_is_ten :
  average_playing_time = 10 := by sorry

end NUMINAMATH_CALUDE_average_playing_time_is_ten_l1885_188512


namespace NUMINAMATH_CALUDE_number_of_pupils_l1885_188514

/-- Given a program with parents and pupils, calculate the number of pupils -/
theorem number_of_pupils (total_people parents : ℕ) (h1 : parents = 105) (h2 : total_people = 803) :
  total_people - parents = 698 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_l1885_188514


namespace NUMINAMATH_CALUDE_short_show_episodes_count_l1885_188595

/-- The number of episodes of the short show -/
def short_show_episodes : ℕ := 24

/-- The duration of one episode of the short show in hours -/
def short_show_duration : ℚ := 1/2

/-- The duration of one episode of the long show in hours -/
def long_show_duration : ℚ := 1

/-- The number of episodes of the long show -/
def long_show_episodes : ℕ := 12

/-- The total time Tim watched TV in hours -/
def total_watch_time : ℕ := 24

theorem short_show_episodes_count :
  short_show_episodes * short_show_duration + long_show_episodes * long_show_duration = total_watch_time := by
  sorry

end NUMINAMATH_CALUDE_short_show_episodes_count_l1885_188595


namespace NUMINAMATH_CALUDE_unique_solution_complex_magnitude_and_inequality_l1885_188554

theorem unique_solution_complex_magnitude_and_inequality :
  ∃! (n : ℝ), n > 0 ∧ Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 ∧ n^2 + 5*n > 50 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_complex_magnitude_and_inequality_l1885_188554


namespace NUMINAMATH_CALUDE_third_vertex_coordinates_l1885_188538

/-- Given a triangle with vertices at (7, 3), (0, 0), and (x, 0) where x < 0,
    if the area of the triangle is 24 square units, then x = -48/√58. -/
theorem third_vertex_coordinates (x : ℝ) :
  x < 0 →
  (1/2 : ℝ) * |x| * 3 * Real.sqrt 58 = 24 →
  x = -48 / Real.sqrt 58 := by
sorry

end NUMINAMATH_CALUDE_third_vertex_coordinates_l1885_188538


namespace NUMINAMATH_CALUDE_double_reflection_of_H_l1885_188543

-- Define the point type
def Point := ℝ × ℝ

-- Define the parallelogram
def E : Point := (3, 6)
def F : Point := (5, 10)
def G : Point := (7, 6)
def H : Point := (5, 2)

-- Define reflection across x-axis
def reflect_x_axis (p : Point) : Point :=
  (p.1, -p.2)

-- Define reflection across y = x + 2
def reflect_y_eq_x_plus_2 (p : Point) : Point :=
  (p.2 - 2, p.1 + 2)

-- Define the composition of the two reflections
def double_reflection (p : Point) : Point :=
  reflect_y_eq_x_plus_2 (reflect_x_axis p)

-- Theorem statement
theorem double_reflection_of_H :
  double_reflection H = (-4, 7) := by sorry

end NUMINAMATH_CALUDE_double_reflection_of_H_l1885_188543


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1885_188501

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x < 0}

def B : Set ℝ := {x | x ≥ 1}

theorem intersection_A_complement_B : A ∩ Bᶜ = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1885_188501


namespace NUMINAMATH_CALUDE_slope_range_for_intersection_l1885_188525

/-- A line with slope k intersects a hyperbola at two distinct points -/
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ ∧ y₂ = k * x₂ ∧
    x₁^2 - y₁^2 = 2 ∧ x₂^2 - y₂^2 = 2

/-- The theorem stating the range of k for which the line intersects the hyperbola at two points -/
theorem slope_range_for_intersection :
  ∀ k : ℝ, intersects_at_two_points k ↔ -1 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_slope_range_for_intersection_l1885_188525


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1885_188500

theorem chess_tournament_games (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 7) : 
  (n * k) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1885_188500


namespace NUMINAMATH_CALUDE_inequality_not_hold_l1885_188520

theorem inequality_not_hold (m n a : Real) 
  (h1 : m > n) (h2 : n > 1) (h3 : 0 < a) (h4 : a < 1) : 
  ¬(a^m > a^n) := by
sorry

end NUMINAMATH_CALUDE_inequality_not_hold_l1885_188520


namespace NUMINAMATH_CALUDE_average_study_time_difference_l1885_188553

def asha_times : List ℝ := [40, 60, 50, 70, 30, 55, 45]
def sasha_times : List ℝ := [50, 70, 40, 100, 10, 55, 0]

theorem average_study_time_difference :
  (List.sum (List.zipWith (·-·) sasha_times asha_times)) / asha_times.length = -25 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l1885_188553


namespace NUMINAMATH_CALUDE_second_discount_percentage_second_discount_is_25_percent_l1885_188573

theorem second_discount_percentage 
  (original_price : ℝ) 
  (first_discount_percent : ℝ) 
  (final_price : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let second_discount_amount := price_after_first_discount - final_price
  let second_discount_percent := (second_discount_amount / price_after_first_discount) * 100
  second_discount_percent

theorem second_discount_is_25_percent :
  second_discount_percentage 33.78 25 19 = 25 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_second_discount_is_25_percent_l1885_188573


namespace NUMINAMATH_CALUDE_discovery_uses_visualization_vr_l1885_188563

/-- Represents different digital Earth technologies -/
inductive DigitalEarthTechnology
  | InformationSuperhighway
  | HighResolutionSatellite
  | SpatialInformation
  | VisualizationAndVirtualReality

/-- Represents a TV program -/
structure TVProgram where
  name : String
  episode : String
  content : String

/-- Determines the digital Earth technology used in a TV program -/
def technology_used (program : TVProgram) : DigitalEarthTechnology :=
  if program.content = "vividly recreated various dinosaurs and their living environments"
  then DigitalEarthTechnology.VisualizationAndVirtualReality
  else DigitalEarthTechnology.InformationSuperhighway

/-- The CCTV Discovery program -/
def discovery_program : TVProgram :=
  { name := "Discovery"
  , episode := "Back to the Dinosaur Era"
  , content := "vividly recreated various dinosaurs and their living environments" }

theorem discovery_uses_visualization_vr :
  technology_used discovery_program = DigitalEarthTechnology.VisualizationAndVirtualReality := by
  sorry


end NUMINAMATH_CALUDE_discovery_uses_visualization_vr_l1885_188563


namespace NUMINAMATH_CALUDE_four_team_win_structure_exists_l1885_188528

/-- Represents the result of a match between two teams -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a volleyball tournament -/
structure Tournament where
  teams : Finset Nat
  results : Nat → Nat → MatchResult
  round_robin : ∀ i j, i ≠ j → (results i j = MatchResult.Win ↔ results j i = MatchResult.Loss)

/-- The main theorem to be proved -/
theorem four_team_win_structure_exists (t : Tournament) 
  (h_eight_teams : t.teams.card = 8) :
  ∃ A B C D, A ∈ t.teams ∧ B ∈ t.teams ∧ C ∈ t.teams ∧ D ∈ t.teams ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    t.results A B = MatchResult.Win ∧
    t.results A C = MatchResult.Win ∧
    t.results A D = MatchResult.Win ∧
    t.results B C = MatchResult.Win ∧
    t.results B D = MatchResult.Win ∧
    t.results C D = MatchResult.Win :=
  sorry

end NUMINAMATH_CALUDE_four_team_win_structure_exists_l1885_188528


namespace NUMINAMATH_CALUDE_no_root_greater_than_four_l1885_188569

-- Define the three equations
def equation1 (x : ℝ) : Prop := 5 * x^2 - 15 = 35
def equation2 (x : ℝ) : Prop := (3*x - 2)^2 = (2*x - 3)^2
def equation3 (x : ℝ) : Prop := (x^2 - 16 : ℝ) = 2*x - 4

-- Theorem stating that no root is greater than 4 for all equations
theorem no_root_greater_than_four :
  (∀ x > 4, ¬ equation1 x) ∧
  (∀ x > 4, ¬ equation2 x) ∧
  (∀ x > 4, ¬ equation3 x) :=
by sorry

end NUMINAMATH_CALUDE_no_root_greater_than_four_l1885_188569


namespace NUMINAMATH_CALUDE_triangle_theorem_l1885_188570

noncomputable def triangle_problem (a b c A B C : ℝ) (m n : ℝ × ℝ) : Prop :=
  let m_x := m.1
  let m_y := m.2
  let n_x := n.1
  let n_y := n.2
  (m_x = Real.sin A ∧ m_y = 1) ∧ 
  (n_x = Real.cos A ∧ n_y = Real.sqrt 3) ∧
  (m_x / n_x = m_y / n_y) ∧  -- parallel vectors condition
  (a = 2) ∧ 
  (b = 2 * Real.sqrt 2) ∧
  (A = Real.pi / 6) ∧
  ((1 / 2 * a * b * Real.sin C = 1 + Real.sqrt 3) ∨ 
   (1 / 2 * a * b * Real.sin C = Real.sqrt 3 - 1))

theorem triangle_theorem (a b c A B C : ℝ) (m n : ℝ × ℝ) :
  triangle_problem a b c A B C m n := by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1885_188570


namespace NUMINAMATH_CALUDE_range_of_a_l1885_188531

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ∈ Set.Iic (-2) ∪ {1} := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1885_188531


namespace NUMINAMATH_CALUDE_odd_function_properties_l1885_188575

-- Define the function h
noncomputable def h : ℝ → ℝ := fun x ↦ 2^x

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ (1 - h x) / (1 + h x)

-- State the theorem
theorem odd_function_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (h 2 = 4) ∧             -- h(2) = 4
  (∀ x, f x = (1 - 2^x) / (1 + 2^x)) ∧  -- Analytical form of f
  (∀ x, f (2*x - 1) > f (x + 1) ↔ x < 2/3) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_l1885_188575
