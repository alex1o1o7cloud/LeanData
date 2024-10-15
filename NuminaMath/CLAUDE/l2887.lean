import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l2887_288709

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l2887_288709


namespace NUMINAMATH_CALUDE_no_real_solutions_for_log_equation_l2887_288772

theorem no_real_solutions_for_log_equation :
  ∀ (p q : ℝ), Real.log (p * q) = Real.log (p^2 + q^2 + 1) → False :=
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_log_equation_l2887_288772


namespace NUMINAMATH_CALUDE_equation_graph_l2887_288712

/-- The set of points (x, y) satisfying (x+y)³ = x³ + y³ is equivalent to the union of three lines -/
theorem equation_graph (x y : ℝ) :
  (x + y)^3 = x^3 + y^3 ↔ (x = 0 ∨ y = 0 ∨ x + y = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_graph_l2887_288712


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2887_288787

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_with_current = 21) 
  (h2 : current_speed = 2.5) : 
  speed_with_current - 2 * current_speed = 16 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l2887_288787


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2887_288708

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2887_288708


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l2887_288775

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 40 →
  triangle_height = 60 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * x →
  x = 10/3 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l2887_288775


namespace NUMINAMATH_CALUDE_room_length_l2887_288748

/-- The length of a rectangular room with given width and area -/
theorem room_length (width : ℝ) (area : ℝ) (h1 : width = 20) (h2 : area = 80) :
  area / width = 4 := by sorry

end NUMINAMATH_CALUDE_room_length_l2887_288748


namespace NUMINAMATH_CALUDE_equation_solution_l2887_288703

theorem equation_solution (x : ℝ) : 
  (1 / (Real.sqrt x + Real.sqrt (x - 2)) + 1 / (Real.sqrt (x + 2) + Real.sqrt x) = 1 / 4) → 
  x = 257 / 16 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2887_288703


namespace NUMINAMATH_CALUDE_unique_n_mod_10_l2887_288700

theorem unique_n_mod_10 : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -4000 [ZMOD 10] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_mod_10_l2887_288700


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l2887_288776

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 15 - 6 * a) : 
  a = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l2887_288776


namespace NUMINAMATH_CALUDE_ratio_as_percent_l2887_288741

theorem ratio_as_percent (first_part second_part : ℕ) (h1 : first_part = 4) (h2 : second_part = 20) :
  (first_part : ℚ) / second_part * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_as_percent_l2887_288741


namespace NUMINAMATH_CALUDE_triple_composition_even_l2887_288740

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem triple_composition_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l2887_288740


namespace NUMINAMATH_CALUDE_expression_evaluation_l2887_288789

theorem expression_evaluation : (18 * 3 + 6) / (6 - 3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2887_288789


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l2887_288765

theorem roots_quadratic_equation (α β : ℝ) : 
  (α^2 - 2*α - 1 = 0) → 
  (β^2 - 2*β - 1 = 0) → 
  (4 * α^3 + 5 * β^4 = -40*α + 153) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l2887_288765


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2887_288773

theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eccentricity : Real.sqrt (1 + b^2 / a^2) = Real.sqrt 6 / 2) :
  let asymptote (x : ℝ) := Real.sqrt 2 / 2 * x
  ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → 
    (y = asymptote x ∨ y = -asymptote x) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2887_288773


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_462_l2887_288795

theorem sum_of_distinct_prime_factors_462 : 
  (Finset.sum (Nat.factors 462).toFinset id) = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_462_l2887_288795


namespace NUMINAMATH_CALUDE_stockholm_malmo_distance_l2887_288770

/-- The scale factor of the map, representing kilometers per centimeter. -/
def scale : ℝ := 10

/-- The distance between Stockholm and Malmo on the map, in centimeters. -/
def map_distance : ℝ := 112

/-- The actual distance between Stockholm and Malmo, in kilometers. -/
def actual_distance : ℝ := map_distance * scale

theorem stockholm_malmo_distance : actual_distance = 1120 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_malmo_distance_l2887_288770


namespace NUMINAMATH_CALUDE_mass_o2_for_combustion_l2887_288713

/-- The mass of O2 gas required for complete combustion of C8H18 -/
theorem mass_o2_for_combustion (moles_c8h18 : ℝ) (molar_mass_o2 : ℝ) : 
  moles_c8h18 = 7 → molar_mass_o2 = 32 → 
  (25 / 2 * moles_c8h18 * molar_mass_o2 : ℝ) = 2800 := by
  sorry

#check mass_o2_for_combustion

end NUMINAMATH_CALUDE_mass_o2_for_combustion_l2887_288713


namespace NUMINAMATH_CALUDE_impossible_arrangement_l2887_288750

/-- A table is a function from pairs of indices to natural numbers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Predicate to check if two cells are adjacent in the table -/
def adjacent (i j k l : Fin 10) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨ (i = k ∧ l.val + 1 = j.val) ∨
  (j = l ∧ i.val + 1 = k.val) ∧ (j = l ∧ k.val + 1 = i.val)

/-- Predicate to check if a quadratic equation has two integer roots -/
def has_two_int_roots (a b : ℕ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ x^2 - a*x + b = 0 ∧ y^2 - a*y + b = 0

theorem impossible_arrangement : ¬∃ (t : Table),
  (∀ i j : Fin 10, 51 ≤ t i j ∧ t i j ≤ 150) ∧
  (∀ i j k l : Fin 10, adjacent i j k l →
    has_two_int_roots (t i j) (t k l) ∨ has_two_int_roots (t k l) (t i j)) :=
sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l2887_288750


namespace NUMINAMATH_CALUDE_distance_between_points_on_lines_l2887_288785

/-- The distance between two points on different lines with a given midpoint. -/
theorem distance_between_points_on_lines (xP yP xQ yQ : ℝ) :
  -- P is on the line 6y = 17x
  6 * yP = 17 * xP →
  -- Q is on the line 8y = 5x
  8 * yQ = 5 * xQ →
  -- (10, 5) is the midpoint of PQ
  (xP + xQ) / 2 = 10 →
  (yP + yQ) / 2 = 5 →
  -- The distance formula
  let distance := Real.sqrt ((xP - xQ)^2 + (yP - yQ)^2)
  -- The distance is equal to some real value (which we don't specify)
  ∃ (d : ℝ), distance = d :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_on_lines_l2887_288785


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2887_288744

-- Define the line equation
def line_equation (a x y : ℝ) : Prop :=
  (a + 2) * x + (1 - a) * y - 3 = 0

-- Theorem statement
theorem fixed_point_on_line (a : ℝ) (h : a ≠ 0) : 
  line_equation a 1 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2887_288744


namespace NUMINAMATH_CALUDE_least_number_of_candles_l2887_288797

theorem least_number_of_candles (b : ℕ) : 
  b > 0 ∧ 
  b % 6 = 5 ∧ 
  b % 8 = 7 ∧ 
  b % 9 = 3 ∧ 
  (∀ c : ℕ, c > 0 ∧ c % 6 = 5 ∧ c % 8 = 7 ∧ c % 9 = 3 → c ≥ b) → 
  b = 119 :=
sorry

end NUMINAMATH_CALUDE_least_number_of_candles_l2887_288797


namespace NUMINAMATH_CALUDE_sixth_term_is_three_l2887_288725

/-- An arithmetic sequence with 10 terms -/
def ArithmeticSequence := Fin 10 → ℝ

/-- The property that the sequence is arithmetic -/
def is_arithmetic (a : ArithmeticSequence) : Prop :=
  ∃ d : ℝ, ∀ i j : Fin 10, a j - a i = d * (j - i)

/-- The sum of even-numbered terms is 15 -/
def sum_even_terms_is_15 (a : ArithmeticSequence) : Prop :=
  a 1 + a 3 + a 5 + a 7 + a 9 = 15

theorem sixth_term_is_three
  (a : ArithmeticSequence)
  (h_arith : is_arithmetic a)
  (h_sum : sum_even_terms_is_15 a) :
  a 5 = 3 :=
sorry

end NUMINAMATH_CALUDE_sixth_term_is_three_l2887_288725


namespace NUMINAMATH_CALUDE_correct_amount_to_return_l2887_288771

/-- Calculates the amount to be returned in rubles given an initial deposit in USD and an exchange rate. -/
def amount_to_return (initial_deposit : ℝ) (exchange_rate : ℝ) : ℝ :=
  initial_deposit * exchange_rate

/-- Theorem stating that given the specific initial deposit and exchange rate, the amount to be returned is 581,500 rubles. -/
theorem correct_amount_to_return :
  amount_to_return 10000 58.15 = 581500 := by
  sorry

end NUMINAMATH_CALUDE_correct_amount_to_return_l2887_288771


namespace NUMINAMATH_CALUDE_exactly_three_primes_probability_l2887_288783

-- Define a die as a type with 6 possible outcomes
def Die := Fin 6

-- Define a function to check if a number is prime (for a 6-sided die)
def isPrime (n : Die) : Bool :=
  n.val + 1 = 2 || n.val + 1 = 3 || n.val + 1 = 5

-- Define the probability of rolling a prime number on a single die
def probPrime : ℚ := 1/2

-- Define the number of dice
def numDice : ℕ := 6

-- Define the number of dice we want to show prime numbers
def targetPrimes : ℕ := 3

-- State the theorem
theorem exactly_three_primes_probability :
  (numDice.choose targetPrimes : ℚ) * probPrime^targetPrimes * (1 - probPrime)^(numDice - targetPrimes) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_primes_probability_l2887_288783


namespace NUMINAMATH_CALUDE_equidistant_points_in_quadrants_I_II_l2887_288756

/-- A point on the line 3x + 5y = 15 that is equidistant from the coordinate axes -/
def equidistant_point (x y : ℝ) : Prop :=
  3 * x + 5 * y = 15 ∧ (x = y ∨ x = -y)

/-- The point is in quadrant I -/
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The point is in quadrant II -/
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The point is in quadrant III -/
def in_quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The point is in quadrant IV -/
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem equidistant_points_in_quadrants_I_II :
  ∀ x y : ℝ, equidistant_point x y → (in_quadrant_I x y ∨ in_quadrant_II x y) ∧
  ¬(in_quadrant_III x y ∨ in_quadrant_IV x y) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_points_in_quadrants_I_II_l2887_288756


namespace NUMINAMATH_CALUDE_password_from_polynomial_factorization_password_for_given_values_l2887_288755

/-- Generates a password from the factors of x^3 - xy^2 --/
def generate_password (x y : ℕ) : ℕ :=
  x * 10000 + (x + y) * 100 + (x - y)

/-- The polynomial x^3 - xy^2 factors as x(x-y)(x+y) --/
theorem password_from_polynomial_factorization (x y : ℕ) :
  x^3 - x*y^2 = x * (x - y) * (x + y) :=
sorry

/-- The password generated from x^3 - xy^2 with x=18 and y=5 is 181323 --/
theorem password_for_given_values :
  generate_password 18 5 = 181323 :=
sorry

end NUMINAMATH_CALUDE_password_from_polynomial_factorization_password_for_given_values_l2887_288755


namespace NUMINAMATH_CALUDE_nancy_pots_proof_l2887_288746

/-- Represents the number of clay pots Nancy created on Monday -/
def monday_pots : ℕ := sorry

/-- The total number of pots Nancy created over three days -/
def total_pots : ℕ := 50

/-- The number of pots Nancy created on Wednesday -/
def wednesday_pots : ℕ := 14

theorem nancy_pots_proof :
  monday_pots = 12 ∧
  monday_pots + 2 * monday_pots + wednesday_pots = total_pots :=
sorry

end NUMINAMATH_CALUDE_nancy_pots_proof_l2887_288746


namespace NUMINAMATH_CALUDE_probability_of_sum_26_l2887_288743

-- Define the faces of the dice
def die1_faces : Finset ℕ := Finset.range 20 \ {0, 19}
def die2_faces : Finset ℕ := (Finset.range 22 \ {0, 8, 21}) ∪ {0}

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 20 * 20

-- Define the favorable outcomes
def favorable_outcomes : ℕ := 13

-- Theorem statement
theorem probability_of_sum_26 :
  (favorable_outcomes : ℚ) / total_outcomes = 13 / 400 :=
sorry

end NUMINAMATH_CALUDE_probability_of_sum_26_l2887_288743


namespace NUMINAMATH_CALUDE_function_and_range_theorem_l2887_288793

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Define the function g
def g (m : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - m * x

-- State the theorem
theorem function_and_range_theorem (a b m : ℝ) :
  a ≠ 0 ∧
  (∀ x, f a b (x + 1) - f a b x = 2 * x - 1) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
    |g m (f a b) x₁ - g m (f a b) x₂| ≤ 2) →
  (∀ x, f a b x = x^2 - 2*x + 3) ∧
  m ∈ Set.Icc (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_function_and_range_theorem_l2887_288793


namespace NUMINAMATH_CALUDE_equation_solutions_l2887_288706

theorem equation_solutions (x y n : ℕ+) : 
  (((x : ℝ)^2 + (y : ℝ)^2)^(n : ℝ) = ((x * y : ℝ)^2016)) ↔ 
  n ∈ ({1344, 1728, 1792, 1920, 1984} : Set ℕ+) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2887_288706


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2887_288778

def opposite (x : ℝ) : ℝ := -x

theorem opposite_of_negative_two :
  opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2887_288778


namespace NUMINAMATH_CALUDE_negative_eighth_power_2009_times_eight_power_2009_l2887_288721

theorem negative_eighth_power_2009_times_eight_power_2009 :
  (-0.125)^2009 * 8^2009 = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_eighth_power_2009_times_eight_power_2009_l2887_288721


namespace NUMINAMATH_CALUDE_range_of_a_l2887_288705

-- Define the inequality function
def f (x a : ℝ) : ℝ := x^2 + (2-a)*x + 4-2*a

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ≥ 2, f x a > 0) ↔ a < 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2887_288705


namespace NUMINAMATH_CALUDE_heloise_gave_ten_dogs_l2887_288767

/-- The number of dogs Heloise gave to Janet -/
def dogs_given_to_janet (total_pets : ℕ) (remaining_dogs : ℕ) : ℕ :=
  let dog_ratio := 10
  let cat_ratio := 17
  let total_ratio := dog_ratio + cat_ratio
  let pets_per_ratio := total_pets / total_ratio
  let original_dogs := dog_ratio * pets_per_ratio
  original_dogs - remaining_dogs

/-- Proof that Heloise gave 10 dogs to Janet -/
theorem heloise_gave_ten_dogs :
  dogs_given_to_janet 189 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_heloise_gave_ten_dogs_l2887_288767


namespace NUMINAMATH_CALUDE_coin_landing_probability_l2887_288717

/-- Represents the specially colored square -/
structure ColoredSquare where
  side_length : ℝ
  triangle_leg : ℝ
  diamond_side : ℝ

/-- Represents the circular coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of the coin landing on a black region -/
def black_region_probability (square : ColoredSquare) (coin : Coin) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem coin_landing_probability 
  (square : ColoredSquare)
  (coin : Coin)
  (h_square_side : square.side_length = 8)
  (h_triangle_leg : square.triangle_leg = 2)
  (h_diamond_side : square.diamond_side = 2 * Real.sqrt 2)
  (h_coin_diameter : coin.diameter = 1) :
  ∃ (a b : ℕ), 
    black_region_probability square coin = 1 / 196 * (a + b * Real.sqrt 2 + Real.pi) ∧
    a + b = 68 :=
  sorry

end NUMINAMATH_CALUDE_coin_landing_probability_l2887_288717


namespace NUMINAMATH_CALUDE_eggs_equal_to_rice_l2887_288732

/-- The cost of a pound of rice in dollars -/
def rice_cost : ℚ := 33/100

/-- The cost of a liter of kerosene in dollars -/
def kerosene_cost : ℚ := 22/100

/-- The number of eggs that cost as much as a half-liter of kerosene -/
def eggs_per_half_liter : ℕ := 4

/-- Theorem stating that 12 eggs cost as much as a pound of rice -/
theorem eggs_equal_to_rice : ℕ := by
  sorry

end NUMINAMATH_CALUDE_eggs_equal_to_rice_l2887_288732


namespace NUMINAMATH_CALUDE_computer_price_proof_l2887_288726

/-- The original price of the computer in yuan -/
def original_price : ℝ := 5000

/-- The installment price of the computer -/
def installment_price (price : ℝ) : ℝ := 1.04 * price

/-- The cash price of the computer -/
def cash_price (price : ℝ) : ℝ := 0.9 * price

/-- Theorem stating that the original price satisfies the given conditions -/
theorem computer_price_proof : 
  installment_price original_price - cash_price original_price = 700 := by
  sorry


end NUMINAMATH_CALUDE_computer_price_proof_l2887_288726


namespace NUMINAMATH_CALUDE_remainder_properties_l2887_288707

theorem remainder_properties (a b n : ℤ) (hn : n ≠ 0) :
  (((a + b) % n = ((a % n + b % n) % n)) ∧
   ((a - b) % n = ((a % n - b % n) % n)) ∧
   ((a * b) % n = ((a % n * b % n) % n))) := by
  sorry

end NUMINAMATH_CALUDE_remainder_properties_l2887_288707


namespace NUMINAMATH_CALUDE_total_students_is_3700_l2887_288730

/-- Represents a high school with three grades -/
structure HighSchool where
  total_students : ℕ
  senior_students : ℕ
  sample_size : ℕ
  freshman_sample : ℕ
  sophomore_sample : ℕ

/-- The conditions of the problem -/
def problem_conditions (school : HighSchool) : Prop :=
  school.senior_students = 1000 ∧
  school.sample_size = 185 ∧
  school.freshman_sample = 75 ∧
  school.sophomore_sample = 60 ∧
  (school.senior_students : ℚ) / school.total_students = 
    (school.sample_size - school.freshman_sample - school.sophomore_sample : ℚ) / school.sample_size

/-- The theorem stating that under the given conditions, the total number of students is 3700 -/
theorem total_students_is_3700 (school : HighSchool) 
  (h : problem_conditions school) : school.total_students = 3700 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_3700_l2887_288730


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l2887_288752

def vector_a : Fin 2 → ℝ := ![(-1), 3]
def vector_b (t : ℝ) : Fin 2 → ℝ := ![1, t]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ i, u i = k * v i

theorem parallel_vectors_t_value :
  ∀ t : ℝ, parallel vector_a (vector_b t) → t = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l2887_288752


namespace NUMINAMATH_CALUDE_sector_area_l2887_288739

/-- Given a sector with radius R and perimeter 4R, its area is R^2 -/
theorem sector_area (R : ℝ) (R_pos : R > 0) : 
  let perimeter := 4 * R
  let arc_length := perimeter - 2 * R
  let area := (1 / 2) * R * arc_length
  area = R^2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l2887_288739


namespace NUMINAMATH_CALUDE_coefficient_a3_value_l2887_288737

theorem coefficient_a3_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 + 3*x^3 + 1 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₃ = 13 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a3_value_l2887_288737


namespace NUMINAMATH_CALUDE_expression_value_at_x_2_l2887_288764

theorem expression_value_at_x_2 :
  let x : ℝ := 2
  (3 * x + 4)^2 - 10 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_x_2_l2887_288764


namespace NUMINAMATH_CALUDE_mice_breeding_experiment_l2887_288761

/-- Calculates the number of mice after two generations of breeding and some pups being eaten --/
def final_mice_count (initial_mice : ℕ) (pups_per_mouse : ℕ) (pups_eaten_per_adult : ℕ) : ℕ :=
  let first_gen_total := initial_mice + initial_mice * pups_per_mouse
  let second_gen_total := first_gen_total + first_gen_total * pups_per_mouse
  second_gen_total - (first_gen_total * pups_eaten_per_adult)

/-- Theorem stating that under the given conditions, the final number of mice is 280 --/
theorem mice_breeding_experiment :
  final_mice_count 8 6 2 = 280 := by
  sorry

end NUMINAMATH_CALUDE_mice_breeding_experiment_l2887_288761


namespace NUMINAMATH_CALUDE_cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_division_cube_set_not_closed_under_squaring_l2887_288738

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def cube_set : Set ℕ := {n : ℕ | is_cube n ∧ n > 0}

theorem cube_set_closed_under_multiplication (a b : ℕ) (ha : a ∈ cube_set) (hb : b ∈ cube_set) :
  (a * b) ∈ cube_set :=
sorry

theorem cube_set_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ (a + b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_division :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ b ≠ 0 ∧ (a / b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_squaring :
  ∃ a : ℕ, a ∈ cube_set ∧ (a^2) ∉ cube_set :=
sorry

end NUMINAMATH_CALUDE_cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_division_cube_set_not_closed_under_squaring_l2887_288738


namespace NUMINAMATH_CALUDE_parabola_one_intersection_l2887_288716

/-- A parabola that intersects the x-axis at exactly one point -/
def one_intersection_parabola (c : ℝ) : Prop :=
  ∃! x, x^2 + x + c = 0

/-- The theorem stating that the parabola y = x^2 + x + c intersects 
    the x-axis at exactly one point when c = 1/4 -/
theorem parabola_one_intersection :
  one_intersection_parabola (1/4 : ℝ) ∧ 
  ∀ c : ℝ, one_intersection_parabola c → c = 1/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_one_intersection_l2887_288716


namespace NUMINAMATH_CALUDE_parameter_a_condition_l2887_288742

theorem parameter_a_condition (a : ℝ) : 
  (∀ x y : ℝ, 2 * a * x^2 + 2 * a * y^2 + 4 * a * x * y - 2 * x * y - y^2 - 2 * x + 1 ≥ 0) → 
  a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parameter_a_condition_l2887_288742


namespace NUMINAMATH_CALUDE_kaeli_problems_per_day_l2887_288757

def marie_pascale_problems_per_day : ℕ := 4
def marie_pascale_total_problems : ℕ := 72
def kaeli_extra_problems : ℕ := 54

def days : ℕ := marie_pascale_total_problems / marie_pascale_problems_per_day

def kaeli_total_problems : ℕ := marie_pascale_total_problems + kaeli_extra_problems

theorem kaeli_problems_per_day : 
  kaeli_total_problems / days = 7 :=
sorry

end NUMINAMATH_CALUDE_kaeli_problems_per_day_l2887_288757


namespace NUMINAMATH_CALUDE_factor_x4_minus_81_l2887_288799

theorem factor_x4_minus_81 (x : ℝ) : 
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_81_l2887_288799


namespace NUMINAMATH_CALUDE_intersection_points_count_l2887_288715

/-- The number of distinct intersection points for the given equations -/
def num_intersection_points : ℕ :=
  let eq1 := fun (x y : ℝ) => (x - y + 2) * (2 * x + 3 * y - 6) = 0
  let eq2 := fun (x y : ℝ) => (3 * x - 2 * y - 1) * (x + 2 * y - 4) = 0
  2

/-- Theorem stating that the number of distinct intersection points is 2 -/
theorem intersection_points_count :
  num_intersection_points = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_points_count_l2887_288715


namespace NUMINAMATH_CALUDE_sperner_theorem_l2887_288722

/-- The largest number of subsets of an n-element set such that no subset is contained in any other -/
def largestSperner (n : ℕ) : ℕ :=
  Nat.choose n (n / 2)

/-- Sperner's theorem -/
theorem sperner_theorem (n : ℕ) :
  largestSperner n = Nat.choose n (n / 2) :=
sorry

end NUMINAMATH_CALUDE_sperner_theorem_l2887_288722


namespace NUMINAMATH_CALUDE_candy_theorem_l2887_288728

def candy_problem (bars_per_friend : ℕ) (num_friends : ℕ) (spare_bars : ℕ) : ℕ :=
  bars_per_friend * num_friends + spare_bars

theorem candy_theorem (bars_per_friend : ℕ) (num_friends : ℕ) (spare_bars : ℕ) :
  candy_problem bars_per_friend num_friends spare_bars =
  bars_per_friend * num_friends + spare_bars :=
by
  sorry

#eval candy_problem 2 7 10

end NUMINAMATH_CALUDE_candy_theorem_l2887_288728


namespace NUMINAMATH_CALUDE_choose_service_providers_and_accessories_l2887_288774

def total_individuals : ℕ := 4
def total_service_providers : ℕ := 25
def total_accessories : ℕ := 5

def ways_to_choose : ℕ := (total_service_providers - 0) *
                           (total_service_providers - 1) *
                           (total_service_providers - 2) *
                           (total_service_providers - 3) *
                           (total_accessories - 0) *
                           (total_accessories - 1) *
                           (total_accessories - 2) *
                           (total_accessories - 3)

theorem choose_service_providers_and_accessories :
  ways_to_choose = 36432000 :=
sorry

end NUMINAMATH_CALUDE_choose_service_providers_and_accessories_l2887_288774


namespace NUMINAMATH_CALUDE_trajectory_equation_l2887_288751

theorem trajectory_equation (a b x y : ℝ) : 
  a^2 + b^2 = 100 →  -- Line segment length is 10
  x = a / 5 →        -- AM = 4MB implies x = a/(1+4)
  y = 4*b / 5 →      -- AM = 4MB implies y = 4b/(1+4)
  16*x^2 + y^2 = 64  -- Trajectory equation
:= by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2887_288751


namespace NUMINAMATH_CALUDE_prime_sum_gcd_ratio_composite_sum_gcd_ratio_l2887_288786

-- Part 1
theorem prime_sum_gcd_ratio (n : ℕ) (hn : Nat.Prime (2 * n - 1)) :
  ∀ (a : Fin n → ℕ), Function.Injective a →
  ∃ i j : Fin n, (a i + a j : ℚ) / Nat.gcd (a i) (a j) ≥ 2 * n - 1 := by sorry

-- Part 2
theorem composite_sum_gcd_ratio (n : ℕ) (hn : ¬Nat.Prime (2 * n - 1)) (hn2 : 2 * n - 1 > 1) :
  ∃ (a : Fin n → ℕ), Function.Injective a ∧
  ∀ i j : Fin n, (a i + a j : ℚ) / Nat.gcd (a i) (a j) < 2 * n - 1 := by sorry

end NUMINAMATH_CALUDE_prime_sum_gcd_ratio_composite_sum_gcd_ratio_l2887_288786


namespace NUMINAMATH_CALUDE_min_faces_two_dice_l2887_288731

theorem min_faces_two_dice (a b : ℕ) : 
  a ≥ 8 → b ≥ 8 →  -- Both dice have at least 8 faces
  (∀ i j, 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b) →  -- Each face has a distinct integer from 1 to the number of faces
  (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b ∧ i + j = 9} : ℚ) / (a * b : ℚ) = 
    (2/3) * ((Nat.card {(i, j) | 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b ∧ i + j = 11} : ℚ) / (a * b : ℚ)) →  -- Probability condition for sum of 9 and 11
  (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b ∧ i + j = 14} : ℚ) / (a * b : ℚ) = 1/9 →  -- Probability condition for sum of 14
  a + b ≥ 22 ∧ ∀ c d, c ≥ 8 → d ≥ 8 → 
    (∀ i j, 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d) →
    (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d ∧ i + j = 9} : ℚ) / (c * d : ℚ) = 
      (2/3) * ((Nat.card {(i, j) | 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d ∧ i + j = 11} : ℚ) / (c * d : ℚ)) →
    (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d ∧ i + j = 14} : ℚ) / (c * d : ℚ) = 1/9 →
    c + d ≥ 22 :=
by sorry

end NUMINAMATH_CALUDE_min_faces_two_dice_l2887_288731


namespace NUMINAMATH_CALUDE_thirty_switch_network_connections_l2887_288769

/-- Represents a network of switches with their connections. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ
  no_multiple_connections : Bool

/-- Calculates the total number of connections in the network. -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- Theorem stating that a network of 30 switches, each connected to 4 others,
    has 60 total connections. -/
theorem thirty_switch_network_connections :
  let network := SwitchNetwork.mk 30 4 true
  total_connections network = 60 := by
  sorry

end NUMINAMATH_CALUDE_thirty_switch_network_connections_l2887_288769


namespace NUMINAMATH_CALUDE_find_divisor_l2887_288782

theorem find_divisor : 
  ∃ d : ℕ, d > 0 ∧ 136 = 9 * d + 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2887_288782


namespace NUMINAMATH_CALUDE_cost_graph_two_segments_l2887_288759

/-- The cost function for pencils -/
def cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 10 * n else 8 * n - 40

/-- The graph of the cost function consists of two connected linear segments -/
theorem cost_graph_two_segments :
  ∃ (a b : ℕ) (m₁ m₂ c₁ c₂ : ℚ),
    a < b ∧
    (∀ n, 1 ≤ n ∧ n ≤ a → cost n = m₁ * n + c₁) ∧
    (∀ n, b ≤ n ∧ n ≤ 20 → cost n = m₂ * n + c₂) ∧
    (m₁ * a + c₁ = m₂ * b + c₂) ∧
    m₁ ≠ m₂ :=
sorry

end NUMINAMATH_CALUDE_cost_graph_two_segments_l2887_288759


namespace NUMINAMATH_CALUDE_shaded_area_is_ten_l2887_288724

/-- A rectangle composed of twelve 1x1 squares -/
structure Rectangle where
  width : ℕ
  height : ℕ
  area : ℕ
  h1 : width = 3
  h2 : height = 4
  h3 : area = width * height
  h4 : area = 12

/-- The unshaded triangular region in the rectangle -/
structure UnshadedTriangle where
  base : ℕ
  height : ℕ
  area : ℝ
  h1 : base = 1
  h2 : height = 4
  h3 : area = (base * height : ℝ) / 2

/-- The total shaded area in the rectangle -/
def shadedArea (r : Rectangle) (ut : UnshadedTriangle) : ℝ :=
  (r.area : ℝ) - ut.area

theorem shaded_area_is_ten (r : Rectangle) (ut : UnshadedTriangle) :
  shadedArea r ut = 10 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_ten_l2887_288724


namespace NUMINAMATH_CALUDE_price_difference_proof_l2887_288734

def shop_x_price : ℚ := 1.25
def shop_y_price : ℚ := 2.75
def num_copies : ℕ := 40

theorem price_difference_proof :
  (shop_y_price * num_copies) - (shop_x_price * num_copies) = 60 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_proof_l2887_288734


namespace NUMINAMATH_CALUDE_savings_fraction_is_5_17_l2887_288753

/-- Represents the worker's savings scenario -/
structure WorkerSavings where
  monthly_pay : ℝ
  savings_fraction : ℝ
  savings_fraction_constant : Prop
  monthly_pay_constant : Prop
  all_savings_from_pay : Prop
  total_savings_eq_5times_unsaved : Prop

/-- Theorem stating that the savings fraction is 5/17 -/
theorem savings_fraction_is_5_17 (w : WorkerSavings) : w.savings_fraction = 5 / 17 :=
by sorry

end NUMINAMATH_CALUDE_savings_fraction_is_5_17_l2887_288753


namespace NUMINAMATH_CALUDE_sean_whistle_count_l2887_288791

/-- Given that Charles has 128 whistles and Sean has 95 more whistles than Charles,
    prove that Sean has 223 whistles. -/
theorem sean_whistle_count :
  let charles_whistles : ℕ := 128
  let sean_extra_whistles : ℕ := 95
  let sean_whistles : ℕ := charles_whistles + sean_extra_whistles
  sean_whistles = 223 := by
  sorry

end NUMINAMATH_CALUDE_sean_whistle_count_l2887_288791


namespace NUMINAMATH_CALUDE_jump_data_mode_l2887_288749

def jump_data : List Nat := [160, 163, 160, 157, 160]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem jump_data_mode :
  mode jump_data = 160 := by
  sorry

end NUMINAMATH_CALUDE_jump_data_mode_l2887_288749


namespace NUMINAMATH_CALUDE_older_brother_stamps_l2887_288777

theorem older_brother_stamps (total : ℕ) (younger : ℕ) (older : ℕ) : 
  total = 25 →
  older = 2 * younger + 1 →
  total = older + younger →
  older = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_older_brother_stamps_l2887_288777


namespace NUMINAMATH_CALUDE_lcm_of_primes_l2887_288796

theorem lcm_of_primes (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) 
  (hxy : x > y) (heq : 2 * x + y = 12) : 
  Nat.lcm x y = 10 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_primes_l2887_288796


namespace NUMINAMATH_CALUDE_quadratic_root_values_l2887_288720

/-- Given that 1 - i is a root of a real-coefficient quadratic equation x² + ax + b = 0,
    prove that a = -2 and b = 2 -/
theorem quadratic_root_values (a b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I)^2 + a*(1 - Complex.I) + b = 0 →
  a = -2 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_values_l2887_288720


namespace NUMINAMATH_CALUDE_candidate_a_democratic_votes_l2887_288798

theorem candidate_a_democratic_votes 
  (total_voters : ℝ) 
  (dem_percent : ℝ) 
  (rep_percent : ℝ) 
  (rep_for_a_percent : ℝ) 
  (total_for_a_percent : ℝ) :
  dem_percent = 0.6 →
  rep_percent = 0.4 →
  rep_for_a_percent = 0.2 →
  total_for_a_percent = 0.59 →
  ∃ (dem_for_a_percent : ℝ),
    dem_for_a_percent * dem_percent * total_voters + 
    rep_for_a_percent * rep_percent * total_voters = 
    total_for_a_percent * total_voters ∧
    dem_for_a_percent = 0.85 :=
by sorry

end NUMINAMATH_CALUDE_candidate_a_democratic_votes_l2887_288798


namespace NUMINAMATH_CALUDE_count_divisible_numbers_l2887_288768

theorem count_divisible_numbers : 
  (Finset.filter (fun n : ℕ => 
    n ≤ 10^10 ∧ 
    (∀ k : ℕ, k ∈ Finset.range 10 → k.succ ∣ n)
  ) (Finset.range (10^10 + 1))).card = 3968253 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_l2887_288768


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l2887_288727

/-- 
Theorem: In an isosceles, obtuse triangle where one angle is 60% larger than a right angle, 
each of the two smallest angles measures 18°.
-/
theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (a b c : ℝ), 
  -- The triangle is isosceles
  a = b →
  -- The triangle is obtuse (one angle > 90°)
  c > 90 →
  -- One angle (c) is 60% larger than a right angle
  c = 90 * 1.6 →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- Each of the two smallest angles (a and b) measures 18°
  a = 18 ∧ b = 18 := by
sorry

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l2887_288727


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l2887_288736

open Set Real

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

-- Theorem statement
theorem subset_implies_a_range (a : ℝ) :
  A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l2887_288736


namespace NUMINAMATH_CALUDE_distance_to_reflection_over_y_axis_l2887_288790

/-- Given a point F with coordinates (-4, 3), prove that the distance between F
    and its reflection over the y-axis is 8. -/
theorem distance_to_reflection_over_y_axis :
  let F : ℝ × ℝ := (-4, 3)
  let F' : ℝ × ℝ := (4, 3)  -- Reflection of F over y-axis
  dist F F' = 8 := by
  sorry

#check distance_to_reflection_over_y_axis

end NUMINAMATH_CALUDE_distance_to_reflection_over_y_axis_l2887_288790


namespace NUMINAMATH_CALUDE_product_and_sum_of_roots_l2887_288762

theorem product_and_sum_of_roots : 
  (16 : ℝ) ^ (1/4 : ℝ) * (32 : ℝ) ^ (1/5 : ℝ) + (64 : ℝ) ^ (1/6 : ℝ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_and_sum_of_roots_l2887_288762


namespace NUMINAMATH_CALUDE_divisibility_condition_l2887_288723

theorem divisibility_condition (m : ℕ+) :
  (∀ k : ℕ, k ≥ 3 → Odd k → (k^(m : ℕ) - 1) % 2^(m : ℕ) = 0) ↔ m = 1 ∨ m = 2 ∨ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2887_288723


namespace NUMINAMATH_CALUDE_seventh_root_product_l2887_288745

theorem seventh_root_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_product_l2887_288745


namespace NUMINAMATH_CALUDE_absolute_value_equation_l2887_288711

theorem absolute_value_equation (x : ℝ) :
  |x - 25| + |x - 15| = |2*x - 40| → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l2887_288711


namespace NUMINAMATH_CALUDE_expansion_equality_l2887_288702

theorem expansion_equality (m n : ℝ) : (m + n) * (m - 2*n) = m^2 - m*n - 2*n^2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l2887_288702


namespace NUMINAMATH_CALUDE_exponential_fraction_simplification_l2887_288729

theorem exponential_fraction_simplification :
  (3^1011 + 3^1009) / (3^1011 - 3^1009) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_fraction_simplification_l2887_288729


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2887_288701

theorem inequality_solution_set (x : ℝ) :
  (x ∈ {x : ℝ | -6 * x^2 + 2 < x}) ↔ (x < -2/3 ∨ x > 1/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2887_288701


namespace NUMINAMATH_CALUDE_total_distance_swam_l2887_288733

/-- Represents the swimming styles -/
inductive SwimmingStyle
| Freestyle
| Butterfly

/-- Calculates the distance swam for a given style -/
def distance_swam (style : SwimmingStyle) (total_time : ℕ) : ℕ :=
  match style with
  | SwimmingStyle.Freestyle =>
    let cycle_time := 26  -- 20 minutes swimming + 6 minutes rest
    let cycles := total_time / cycle_time
    let distance_per_cycle := 500  -- 100 meters in 4 minutes, so 500 meters in 20 minutes
    cycles * distance_per_cycle
  | SwimmingStyle.Butterfly =>
    let cycle_time := 35  -- 30 minutes swimming + 5 minutes rest
    let cycles := total_time / cycle_time
    let distance_per_cycle := 429  -- 100 meters in 7 minutes, so approximately 429 meters in 30 minutes
    cycles * distance_per_cycle

theorem total_distance_swam :
  let freestyle_time := 90  -- 1 hour and 30 minutes in minutes
  let butterfly_time := 90  -- 1 hour and 30 minutes in minutes
  let freestyle_distance := distance_swam SwimmingStyle.Freestyle freestyle_time
  let butterfly_distance := distance_swam SwimmingStyle.Butterfly butterfly_time
  freestyle_distance + butterfly_distance = 2358 := by
  sorry


end NUMINAMATH_CALUDE_total_distance_swam_l2887_288733


namespace NUMINAMATH_CALUDE_square_perimeter_l2887_288763

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 900) (h2 : side * side = area) : 
  4 * side = 120 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2887_288763


namespace NUMINAMATH_CALUDE_energy_drink_cost_l2887_288784

/-- The cost of an energy drink bottle given the sales and purchases of a basketball team. -/
theorem energy_drink_cost (cupcakes : ℕ) (cupcake_price : ℚ) 
  (cookies : ℕ) (cookie_price : ℚ)
  (basketballs : ℕ) (basketball_price : ℚ)
  (energy_drinks : ℕ) :
  cupcakes = 50 →
  cupcake_price = 2 →
  cookies = 40 →
  cookie_price = 1/2 →
  basketballs = 2 →
  basketball_price = 40 →
  energy_drinks = 20 →
  (cupcakes : ℚ) * cupcake_price + (cookies : ℚ) * cookie_price 
    - (basketballs : ℚ) * basketball_price = (energy_drinks : ℚ) * 2 :=
by sorry

end NUMINAMATH_CALUDE_energy_drink_cost_l2887_288784


namespace NUMINAMATH_CALUDE_ratio_of_60_to_12_l2887_288766

theorem ratio_of_60_to_12 : 
  let a := 60
  let b := 12
  (a : ℚ) / b = 5 / 1 := by sorry

end NUMINAMATH_CALUDE_ratio_of_60_to_12_l2887_288766


namespace NUMINAMATH_CALUDE_expression_value_l2887_288714

theorem expression_value : 105^3 - 3 * 105^2 + 3 * 105 - 1 = 1124864 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2887_288714


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2887_288792

/-- The line √3x - y + m = 0 is tangent to the circle x^2 + y^2 - 2y = 0 if and only if m = -1 or m = 3 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, (Real.sqrt 3 * x - y + m = 0) → (x^2 + y^2 - 2*y = 0) → 
   (∀ ε > 0, ∃ x' y' : ℝ, x' ≠ x ∨ y' ≠ y ∧ 
    (Real.sqrt 3 * x' - y' + m = 0) ∧ 
    (x'^2 + y'^2 - 2*y' ≠ 0) ∧
    ((x' - x)^2 + (y' - y)^2 < ε^2))) ↔ 
  (m = -1 ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2887_288792


namespace NUMINAMATH_CALUDE_big_bottles_count_l2887_288719

/-- The number of big bottles initially in storage -/
def big_bottles : ℕ := 14000

/-- The number of small bottles initially in storage -/
def small_bottles : ℕ := 6000

/-- The percentage of small bottles sold -/
def small_bottles_sold_percent : ℚ := 20 / 100

/-- The percentage of big bottles sold -/
def big_bottles_sold_percent : ℚ := 23 / 100

/-- The total number of bottles remaining in storage -/
def total_remaining : ℕ := 15580

theorem big_bottles_count :
  (small_bottles * (1 - small_bottles_sold_percent) : ℚ).floor +
  (big_bottles * (1 - big_bottles_sold_percent) : ℚ).floor = total_remaining := by
  sorry

end NUMINAMATH_CALUDE_big_bottles_count_l2887_288719


namespace NUMINAMATH_CALUDE_gold_cube_side_length_l2887_288747

/-- Proves that a gold cube with given parameters has a side length of 6 cm -/
theorem gold_cube_side_length (L : ℝ) 
  (density : ℝ) (buy_price : ℝ) (sell_factor : ℝ) (profit : ℝ) :
  density = 19 →
  buy_price = 60 →
  sell_factor = 1.5 →
  profit = 123120 →
  profit = (sell_factor * buy_price * density * L^3) - (buy_price * density * L^3) →
  L = 6 :=
by sorry

end NUMINAMATH_CALUDE_gold_cube_side_length_l2887_288747


namespace NUMINAMATH_CALUDE_kelly_apples_l2887_288718

/-- The number of apples Kelly initially has -/
def initial_apples : ℕ := 56

/-- The number of additional apples Kelly needs to pick -/
def apples_to_pick : ℕ := 49

/-- The total number of apples Kelly wants to have -/
def total_apples : ℕ := initial_apples + apples_to_pick

theorem kelly_apples : total_apples = 105 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l2887_288718


namespace NUMINAMATH_CALUDE_part_one_part_two_l2887_288710

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |a * x + 1|
def g (x : ℝ) : ℝ := |x + 1| + 2

-- Part I
theorem part_one :
  {x : ℝ | f (1/2) x < 2} = {x : ℝ | 0 < x ∧ x < 4/3} := by sorry

-- Part II
theorem part_two :
  (∀ x ∈ Set.Ioo 0 1, f a x ≤ g x) → -5 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2887_288710


namespace NUMINAMATH_CALUDE_clique_of_nine_l2887_288758

/-- Represents the relationship of knowing each other in a group of people -/
def Knows (n : ℕ) := Fin n → Fin n → Prop

/-- States that the 'Knows' relation is symmetric -/
def SymmetricKnows {n : ℕ} (knows : Knows n) :=
  ∀ i j : Fin n, knows i j → knows j i

/-- States that among any 3 people, at least two know each other -/
def AtLeastTwoKnowEachOther {n : ℕ} (knows : Knows n) :=
  ∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    knows i j ∨ knows j k ∨ knows i k

/-- Defines a clique of size 4 where everyone knows each other -/
def HasCliqueFour {n : ℕ} (knows : Knows n) :=
  ∃ i j k l : Fin n, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
    knows i j ∧ knows i k ∧ knows i l ∧
    knows j k ∧ knows j l ∧
    knows k l

theorem clique_of_nine (knows : Knows 9) 
  (symm : SymmetricKnows knows) 
  (atleast_two : AtLeastTwoKnowEachOther knows) : 
  HasCliqueFour knows := by
  sorry

end NUMINAMATH_CALUDE_clique_of_nine_l2887_288758


namespace NUMINAMATH_CALUDE_system_solution_l2887_288704

theorem system_solution (x y a b : ℝ) : 
  x = 1 ∧ 
  y = -2 ∧ 
  3 * x + 2 * y = a ∧ 
  b * x - y = 5 → 
  b - a = 4 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2887_288704


namespace NUMINAMATH_CALUDE_room_pave_cost_l2887_288794

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the cost to pave a rectangle given the cost per square meter -/
def pave_cost (r : Rectangle) (cost_per_sqm : ℝ) : ℝ := area r * cost_per_sqm

/-- The total cost to pave two rectangles -/
def total_pave_cost (r1 r2 : Rectangle) (cost1 cost2 : ℝ) : ℝ :=
  pave_cost r1 cost1 + pave_cost r2 cost2

theorem room_pave_cost :
  let rect1 : Rectangle := { length := 6, width := 4.75 }
  let rect2 : Rectangle := { length := 3, width := 2 }
  let cost1 : ℝ := 900
  let cost2 : ℝ := 750
  total_pave_cost rect1 rect2 cost1 cost2 = 30150 := by
  sorry

end NUMINAMATH_CALUDE_room_pave_cost_l2887_288794


namespace NUMINAMATH_CALUDE_single_point_condition_l2887_288735

/-- The equation represents a single point if and only if d equals 125/4 -/
theorem single_point_condition (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + 2 * p.2^2 + 9 * p.1 - 14 * p.2 + d = 0) ↔ 
  d = 125 / 4 := by
  sorry

end NUMINAMATH_CALUDE_single_point_condition_l2887_288735


namespace NUMINAMATH_CALUDE_factoring_expression_l2887_288760

theorem factoring_expression (y : ℝ) : 3*y*(2*y+5) + 4*(2*y+5) = (3*y+4)*(2*y+5) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l2887_288760


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2887_288788

theorem quadratic_solution_sum (c d : ℝ) : 
  (∀ x, x^2 - 6*x + 11 = 23 ↔ x = c ∨ x = d) →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2887_288788


namespace NUMINAMATH_CALUDE_big_dig_mining_theorem_l2887_288780

/-- Represents a mine with its daily production and ore percentages -/
structure Mine where
  dailyProduction : ℝ
  copperPercentage : ℝ
  ironPercentage : ℝ
  nickelPercentage : ℝ
  zincPercentage : ℝ

/-- Calculates the daily copper production for a given mine -/
def dailyCopperProduction (m : Mine) : ℝ :=
  m.dailyProduction * m.copperPercentage

/-- The Big Dig Mining Company problem -/
theorem big_dig_mining_theorem (mineA mineB mineC : Mine)
  (hA : mineA = { dailyProduction := 3000
                , copperPercentage := 0.05
                , ironPercentage := 0.60
                , nickelPercentage := 0.10
                , zincPercentage := 0.25 })
  (hB : mineB = { dailyProduction := 4000
                , copperPercentage := 0.10
                , ironPercentage := 0.50
                , nickelPercentage := 0.30
                , zincPercentage := 0.10 })
  (hC : mineC = { dailyProduction := 3500
                , copperPercentage := 0.15
                , ironPercentage := 0.45
                , nickelPercentage := 0.20
                , zincPercentage := 0.20 }) :
  dailyCopperProduction mineA + dailyCopperProduction mineB + dailyCopperProduction mineC = 1075 := by
  sorry

end NUMINAMATH_CALUDE_big_dig_mining_theorem_l2887_288780


namespace NUMINAMATH_CALUDE_min_cubes_in_prism_l2887_288781

/-- Given a rectangular prism built with N identical 1-cm cubes,
    where 420 cubes are hidden from a viewpoint showing three faces,
    the minimum possible value of N is 630. -/
theorem min_cubes_in_prism (N : ℕ) (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 420 →
  N = l * m * n →
  (∀ l' m' n' : ℕ, (l' - 1) * (m' - 1) * (n' - 1) = 420 → l' * m' * n' ≥ N) →
  N = 630 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_in_prism_l2887_288781


namespace NUMINAMATH_CALUDE_problem_statement_l2887_288779

theorem problem_statement (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 5)
  (h2 : a/x + b/y + c/z = 6) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 13 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2887_288779


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l2887_288754

theorem ceiling_floor_difference : 
  ⌈(12 : ℚ) / 7 * (-29 : ℚ) / 3⌉ - ⌊(12 : ℚ) / 7 * ⌊(-29 : ℚ) / 3⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l2887_288754
