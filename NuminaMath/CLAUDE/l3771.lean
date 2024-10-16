import Mathlib

namespace NUMINAMATH_CALUDE_unique_denomination_l3771_377159

/-- Given unlimited supply of stamps of denominations 4, n, and n+1 cents,
    57 cents is the greatest postage that cannot be formed -/
def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 57 → ∃ a b c : ℕ, k = 4*a + n*b + (n+1)*c

/-- 21 is the only positive integer satisfying the condition -/
theorem unique_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n :=
sorry

end NUMINAMATH_CALUDE_unique_denomination_l3771_377159


namespace NUMINAMATH_CALUDE_quadratic_no_roots_l3771_377126

theorem quadratic_no_roots (b c : ℝ) 
  (h : ∀ x : ℝ, x^2 + b*x + c > 0) : 
  ¬ ∃ x : ℝ, x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_l3771_377126


namespace NUMINAMATH_CALUDE_clothing_store_profit_l3771_377192

/-- Represents the daily profit function for a clothing store -/
def daily_profit (cost : ℝ) (initial_price : ℝ) (initial_sales : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_price - price_reduction - cost) * (initial_sales + 2 * price_reduction)

theorem clothing_store_profit 
  (cost : ℝ) (initial_price : ℝ) (initial_sales : ℝ)
  (h_cost : cost = 50)
  (h_initial_price : initial_price = 90)
  (h_initial_sales : initial_sales = 20) :
  (∃ (x : ℝ), daily_profit cost initial_price initial_sales x = 1200) ∧
  (¬ ∃ (y : ℝ), daily_profit cost initial_price initial_sales y = 2000) := by
  sorry

#check clothing_store_profit

end NUMINAMATH_CALUDE_clothing_store_profit_l3771_377192


namespace NUMINAMATH_CALUDE_max_value_of_f_l3771_377176

/-- A cubic function with a constant term -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

/-- The minimum value of f on the interval [1,3] -/
def min_value : ℝ := 2

/-- The interval on which we're considering the function -/
def interval : Set ℝ := Set.Icc 1 3

theorem max_value_of_f (m : ℝ) (h : ∃ x ∈ interval, ∀ y ∈ interval, f m y ≥ f m x ∧ f m x = min_value) :
  ∃ x ∈ interval, ∀ y ∈ interval, f m y ≤ f m x ∧ f m x = 10 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3771_377176


namespace NUMINAMATH_CALUDE_probability_three_different_suits_l3771_377167

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := 13

/-- The probability of picking three cards of different suits from a standard deck without replacement -/
def probabilityDifferentSuits : ℚ :=
  (CardsPerSuit * (StandardDeck - NumberOfSuits)) / 
  (StandardDeck * (StandardDeck - 1))

theorem probability_three_different_suits :
  probabilityDifferentSuits = 169 / 425 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_different_suits_l3771_377167


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3771_377168

theorem polynomial_factorization (a b : ℝ) : a^2 - b^2 + 2*a + 1 = (a - b + 1) * (a + b + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3771_377168


namespace NUMINAMATH_CALUDE_stating_discount_calculation_l3771_377124

/-- Represents the profit percentage after discount -/
def profit_after_discount : ℝ := 25

/-- Represents the profit percentage without discount -/
def profit_without_discount : ℝ := 38.89

/-- Represents the discount percentage -/
def discount_percentage : ℝ := 10

/-- 
Theorem stating that given the profit percentages with and without discount, 
the discount percentage is 10%
-/
theorem discount_calculation (cost : ℝ) (cost_positive : cost > 0) :
  let selling_price := cost * (1 + profit_after_discount / 100)
  let marked_price := cost * (1 + profit_without_discount / 100)
  selling_price = marked_price * (1 - discount_percentage / 100) :=
by
  sorry


end NUMINAMATH_CALUDE_stating_discount_calculation_l3771_377124


namespace NUMINAMATH_CALUDE_max_abs_z_l3771_377125

theorem max_abs_z (z : ℂ) : 
  Complex.abs (z + 3 + 4*I) ≤ 2 → 
  ∃ (w : ℂ), Complex.abs (w + 3 + 4*I) ≤ 2 ∧ 
             ∀ (u : ℂ), Complex.abs (u + 3 + 4*I) ≤ 2 → Complex.abs u ≤ Complex.abs w ∧
             Complex.abs w = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z_l3771_377125


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3771_377134

theorem simplify_and_evaluate (x : ℝ) (h : x ≠ 2) :
  (x + 2 + 3 / (x - 2)) / ((1 + 2*x + x^2) / (x - 2)) = (x - 1) / (x + 1) ∧
  (4 + 2 + 3 / (4 - 2)) / ((1 + 2*4 + 4^2) / (4 - 2)) = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3771_377134


namespace NUMINAMATH_CALUDE_probability_three_odd_in_six_rolls_l3771_377197

/-- The probability of getting an odd number on a single roll of a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of rolls -/
def num_rolls : ℕ := 6

/-- The number of desired odd outcomes -/
def desired_odd : ℕ := 3

/-- The probability of getting exactly k successes in n trials 
    with probability p for each trial -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem probability_three_odd_in_six_rolls :
  binomial_probability num_rolls desired_odd prob_odd = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_odd_in_six_rolls_l3771_377197


namespace NUMINAMATH_CALUDE_roof_ratio_l3771_377137

/-- Proves that a rectangular roof with given area and length-width difference has a specific length-to-width ratio -/
theorem roof_ratio (length width : ℝ) 
  (area_eq : length * width = 675)
  (diff_eq : length - width = 30) :
  length / width = 3 := by
sorry

end NUMINAMATH_CALUDE_roof_ratio_l3771_377137


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3771_377170

theorem sufficient_not_necessary (a : ℝ) :
  (a > 10 → (1 / a < 1 / 10)) ∧ ¬((1 / a < 1 / 10) → a > 10) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3771_377170


namespace NUMINAMATH_CALUDE_inverse_difference_l3771_377160

-- Define a real-valued function f and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the condition that f(x+2) is the inverse of f⁻¹(x-1)
axiom inverse_condition : ∀ x, f (x + 2) = f_inv (x - 1)

-- Define the theorem
theorem inverse_difference :
  f_inv 2010 - f_inv 1 = 4018 :=
sorry

end NUMINAMATH_CALUDE_inverse_difference_l3771_377160


namespace NUMINAMATH_CALUDE_max_xy_value_l3771_377147

theorem max_xy_value (x y : ℕ+) (h1 : 7 * x + 2 * y = 140) (h2 : x ≤ 15) : 
  x * y ≤ 350 ∧ ∃ (x₀ y₀ : ℕ+), 7 * x₀ + 2 * y₀ = 140 ∧ x₀ ≤ 15 ∧ x₀ * y₀ = 350 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l3771_377147


namespace NUMINAMATH_CALUDE_bingo_paths_l3771_377161

/-- Represents the number of paths to spell BINGO on a grid --/
def num_bingo_paths (b_to_i : Nat) (i_to_n : Nat) (n_to_g : Nat) (g_to_o : Nat) : Nat :=
  b_to_i * i_to_n * n_to_g * g_to_o

/-- Theorem stating the number of paths to spell BINGO --/
theorem bingo_paths :
  ∀ (b_to_i i_to_n n_to_g g_to_o : Nat),
    b_to_i = 3 →
    i_to_n = 3 →
    n_to_g = 2 →
    g_to_o = 2 →
    num_bingo_paths b_to_i i_to_n n_to_g g_to_o = 36 :=
by
  sorry

#eval num_bingo_paths 3 3 2 2

end NUMINAMATH_CALUDE_bingo_paths_l3771_377161


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3771_377115

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  (∀ n, a (n + 1) = a n * q) →
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3771_377115


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3771_377158

theorem quadratic_inequality_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 3 > 0) → (k > 2 ∨ k < -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3771_377158


namespace NUMINAMATH_CALUDE_fraction_simplification_l3771_377136

theorem fraction_simplification (y : ℝ) (h : y = 5) : 
  (y^4 - 8*y^2 + 16) / (y^2 - 4) = 21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3771_377136


namespace NUMINAMATH_CALUDE_product_difference_l3771_377196

theorem product_difference (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : ∃ k, b = 10 * k) :
  let correct_product := a * b
  let incorrect_product := (a * b) / 10
  correct_product = 10 * incorrect_product :=
by
  sorry

end NUMINAMATH_CALUDE_product_difference_l3771_377196


namespace NUMINAMATH_CALUDE_intersection_M_N_l3771_377169

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3771_377169


namespace NUMINAMATH_CALUDE_initial_deposit_calculation_l3771_377186

/-- Proves that the initial deposit is 8000 given the conditions of the problem -/
theorem initial_deposit_calculation (P R : ℝ) 
  (h1 : P * (1 + 3 * R / 100) = 9200)
  (h2 : P * (1 + 3 * (R + 1) / 100) = 9440) : 
  P = 8000 := by
  sorry

end NUMINAMATH_CALUDE_initial_deposit_calculation_l3771_377186


namespace NUMINAMATH_CALUDE_circle_radius_zero_l3771_377143

theorem circle_radius_zero (x y : ℝ) : 
  4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0 → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l3771_377143


namespace NUMINAMATH_CALUDE_exponential_decreasing_for_base_less_than_one_l3771_377183

theorem exponential_decreasing_for_base_less_than_one 
  (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
  a^((-0.1) : ℝ) > a^(0.1 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_exponential_decreasing_for_base_less_than_one_l3771_377183


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l3771_377145

/-- 
Given a two-digit number where the difference between the original number 
and the number with interchanged digits is 27, prove that the difference 
between its two digits is 3.
-/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 27 → x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l3771_377145


namespace NUMINAMATH_CALUDE_complex_magnitude_l3771_377144

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z + w) = 2) :
  Complex.abs z = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3771_377144


namespace NUMINAMATH_CALUDE_binary_ternary_equality_l3771_377138

theorem binary_ternary_equality (a b : ℕ) : 
  a ∈ ({0, 1, 2} : Set ℕ) → 
  b ∈ ({0, 1} : Set ℕ) → 
  (8 + 2 * b + 1 = 9 * a + 2) → 
  (a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_binary_ternary_equality_l3771_377138


namespace NUMINAMATH_CALUDE_ball_probabilities_l3771_377130

/-- Represents the color of a ball -/
inductive BallColor
  | Yellow
  | Green
  | Red

/-- Represents the box of balls -/
structure BallBox where
  total : Nat
  yellow : Nat
  green : Nat
  red : Nat

/-- Calculates the probability of drawing a ball of a specific color -/
def probability (box : BallBox) (color : BallColor) : Rat :=
  match color with
  | BallColor.Yellow => box.yellow / box.total
  | BallColor.Green => box.green / box.total
  | BallColor.Red => box.red / box.total

/-- The main theorem to prove -/
theorem ball_probabilities (box : BallBox) : 
  box.total = 10 ∧ 
  box.yellow = 1 ∧ 
  box.green = 3 ∧ 
  box.red = box.total - box.yellow - box.green →
  probability box BallColor.Green > probability box BallColor.Yellow ∧
  probability box BallColor.Red = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ball_probabilities_l3771_377130


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_323_l3771_377140

theorem smallest_next_divisor_after_323 (n : ℕ) (h1 : 1000 ≤ n ∧ n ≤ 9999) 
  (h2 : Even n) (h3 : n % 323 = 0) : 
  ∃ (d : ℕ), d > 323 ∧ n % d = 0 ∧ d ≥ 340 ∧ 
  ∀ (d' : ℕ), d' > 323 ∧ n % d' = 0 → d' ≥ 340 := by
  sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_323_l3771_377140


namespace NUMINAMATH_CALUDE_original_price_correct_l3771_377198

/-- The original price of a shirt before discount -/
def original_price : ℝ := 975

/-- The discount percentage applied to the shirt -/
def discount_percentage : ℝ := 0.20

/-- The discounted price of the shirt -/
def discounted_price : ℝ := 780

/-- Theorem stating that the original price is correct given the discount and discounted price -/
theorem original_price_correct : 
  original_price * (1 - discount_percentage) = discounted_price :=
by sorry

end NUMINAMATH_CALUDE_original_price_correct_l3771_377198


namespace NUMINAMATH_CALUDE_stating_danny_bottle_caps_l3771_377133

/-- Represents the number of bottle caps Danny had initially. -/
def initial_caps : ℕ := 69

/-- Represents the number of bottle caps Danny threw away. -/
def thrown_away_caps : ℕ := 60

/-- Represents the number of new bottle caps Danny found. -/
def new_caps : ℕ := 58

/-- Represents the number of bottle caps Danny has now. -/
def current_caps : ℕ := 67

/-- 
Theorem stating that the initial number of bottle caps minus the thrown away caps,
plus the new caps found, equals the current number of caps.
-/
theorem danny_bottle_caps : 
  initial_caps - thrown_away_caps + new_caps = current_caps := by
  sorry

#check danny_bottle_caps

end NUMINAMATH_CALUDE_stating_danny_bottle_caps_l3771_377133


namespace NUMINAMATH_CALUDE_stating_two_thousandth_hit_on_second_string_l3771_377141

/-- Represents the number of strings on the guitar. -/
def num_strings : ℕ := 6

/-- Represents the total number of hits we're interested in. -/
def total_hits : ℕ := 2000

/-- 
Represents the string number for a given hit in the sequence.
n: The hit number
-/
def string_number (n : ℕ) : ℕ :=
  let cycle_length := 2 * num_strings - 2
  let position_in_cycle := n % cycle_length
  if position_in_cycle ≤ num_strings
  then position_in_cycle
  else 2 * num_strings - position_in_cycle

/-- 
Theorem stating that the 2000th hit lands on string number 2.
-/
theorem two_thousandth_hit_on_second_string : 
  string_number total_hits = 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_two_thousandth_hit_on_second_string_l3771_377141


namespace NUMINAMATH_CALUDE_cube_section_not_pentagon_cube_section_can_be_hexagon_l3771_377179

/-- A cube in 3D space --/
structure Cube where
  side : ℝ
  center : ℝ × ℝ × ℝ

/-- A plane in 3D space --/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- The intersection of a plane and a cube --/
def PlaneSection (c : Cube) (p : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Predicate to check if a set of points forms a regular polygon --/
def IsRegularPolygon (s : Set (ℝ × ℝ × ℝ)) (n : ℕ) : Prop :=
  sorry

theorem cube_section_not_pentagon (c : Cube) :
  ¬ ∃ p : Plane, IsRegularPolygon (PlaneSection c p) 5 :=
sorry

theorem cube_section_can_be_hexagon :
  ∃ c : Cube, ∃ p : Plane, IsRegularPolygon (PlaneSection c p) 6 :=
sorry

end NUMINAMATH_CALUDE_cube_section_not_pentagon_cube_section_can_be_hexagon_l3771_377179


namespace NUMINAMATH_CALUDE_largest_negative_congruent_to_two_mod_twentynine_l3771_377128

theorem largest_negative_congruent_to_two_mod_twentynine : 
  ∃ (n : ℤ), 
    n = -1011 ∧ 
    n ≡ 2 [ZMOD 29] ∧ 
    n < 0 ∧ 
    -9999 ≤ n ∧ 
    n ≥ -999 ∧ 
    ∀ (m : ℤ), 
      m ≡ 2 [ZMOD 29] → 
      m < 0 → 
      -9999 ≤ m → 
      m ≥ -999 → 
      m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_congruent_to_two_mod_twentynine_l3771_377128


namespace NUMINAMATH_CALUDE_car_speed_problem_l3771_377154

theorem car_speed_problem (speed_A : ℝ) (time_A : ℝ) (time_B : ℝ) (distance_ratio : ℝ) :
  speed_A = 70 →
  time_A = 10 →
  time_B = 10 →
  distance_ratio = 2 →
  ∃ speed_B : ℝ, speed_B = 35 ∧ speed_A * time_A = distance_ratio * (speed_B * time_B) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3771_377154


namespace NUMINAMATH_CALUDE_monotonic_increasing_range_l3771_377177

def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x

theorem monotonic_increasing_range (a : ℝ) :
  Monotone (f a) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_range_l3771_377177


namespace NUMINAMATH_CALUDE_noodle_portions_l3771_377106

-- Define the variables
def total_spent : ℕ := 3000
def total_portions : ℕ := 170
def price_mixed : ℕ := 15
def price_beef : ℕ := 20

-- Define the theorem
theorem noodle_portions :
  ∃ (mixed beef : ℕ),
    mixed + beef = total_portions ∧
    price_mixed * mixed + price_beef * beef = total_spent ∧
    mixed = 80 ∧
    beef = 90 := by
  sorry

end NUMINAMATH_CALUDE_noodle_portions_l3771_377106


namespace NUMINAMATH_CALUDE_farm_ratio_change_l3771_377101

/-- Represents the farm's livestock inventory --/
structure Farm where
  horses : ℕ
  cows : ℕ

/-- Calculates the ratio of horses to cows as a pair of natural numbers --/
def ratio (f : Farm) : ℕ × ℕ :=
  let gcd := Nat.gcd f.horses f.cows
  (f.horses / gcd, f.cows / gcd)

theorem farm_ratio_change (initial : Farm) (final : Farm) : 
  (ratio initial = (3, 1)) →
  (final.horses = initial.horses - 15) →
  (final.cows = initial.cows + 15) →
  (final.horses = final.cows + 30) →
  (ratio final = (5, 3)) := by
  sorry


end NUMINAMATH_CALUDE_farm_ratio_change_l3771_377101


namespace NUMINAMATH_CALUDE_three_digit_addition_theorem_l3771_377103

/-- Represents a three-digit number in the form xyz --/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to a natural number --/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem three_digit_addition_theorem (a b : Nat) :
  let n1 : ThreeDigitNumber := ⟨4, a, 5, by sorry, by sorry, by sorry⟩
  let n2 : ThreeDigitNumber := ⟨4, 3, 8, by sorry, by sorry, by sorry⟩
  let result : ThreeDigitNumber := ⟨8, b, 3, by sorry, by sorry, by sorry⟩
  (n1.toNat + n2.toNat = result.toNat) →
  (result.toNat % 3 = 0) →
  a + b = 1 := by
  sorry

#check three_digit_addition_theorem

end NUMINAMATH_CALUDE_three_digit_addition_theorem_l3771_377103


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3771_377113

theorem circle_area_ratio : 
  ∀ (r₁ r₂ : ℝ), 
  r₁ > 0 → r₂ > 0 → r₂ = 2 * r₁ →
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3771_377113


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_minus_i_l3771_377157

theorem imaginary_part_of_i_times_one_minus_i (i : ℂ) : 
  i * i = -1 → Complex.im (i * (1 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_minus_i_l3771_377157


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l3771_377188

/-- The line on which the center of the circle lies -/
def center_line (x y : ℝ) : Prop := x - y - 4 = 0

/-- The first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0

/-- The second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

/-- The equation of the circle we want to prove -/
def target_circle (x y : ℝ) : Prop := x^2 + y^2 - x + 7*y - 32 = 0

/-- Theorem stating that the target_circle satisfies the given conditions -/
theorem circle_satisfies_conditions :
  ∃ (h k : ℝ), 
    (center_line h k) ∧ 
    (∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → target_circle x y) ∧
    ((h - 1/2)^2 + (k - 7/2)^2 = (33/2)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_satisfies_conditions_l3771_377188


namespace NUMINAMATH_CALUDE_square_area_unchanged_l3771_377112

theorem square_area_unchanged (k : ℝ) : k > 0 → k^2 = 1 → k = 1 := by sorry

end NUMINAMATH_CALUDE_square_area_unchanged_l3771_377112


namespace NUMINAMATH_CALUDE_unique_intersection_point_l3771_377108

/-- The function g(x) = x^3 - 9x^2 + 27x - 29 -/
def g (x : ℝ) : ℝ := x^3 - 9*x^2 + 27*x - 29

/-- The point (1, 1) is the unique intersection of y = g(x) and y = g^(-1)(x) -/
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (1, 1) := by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l3771_377108


namespace NUMINAMATH_CALUDE_f_neq_for_prime_sum_l3771_377171

/-- Sum of positive integers not relatively prime to n -/
def f (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => if Nat.gcd k n ≠ 1 then k else 0)

/-- Theorem stating that f(n+p) ≠ f(n) for n ≥ 2 and prime p -/
theorem f_neq_for_prime_sum (n : ℕ) (p : ℕ) (h1 : n ≥ 2) (h2 : Nat.Prime p) :
  f (n + p) ≠ f n :=
by
  sorry

end NUMINAMATH_CALUDE_f_neq_for_prime_sum_l3771_377171


namespace NUMINAMATH_CALUDE_binomial_12_9_l3771_377123

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l3771_377123


namespace NUMINAMATH_CALUDE_other_endpoint_coordinates_l3771_377178

/-- Given a line segment with midpoint (3, 7) and one endpoint at (0, 11),
    prove that the other endpoint is at (6, 3). -/
theorem other_endpoint_coordinates :
  ∀ (x y : ℝ),
  (3 = (0 + x) / 2) →
  (7 = (11 + y) / 2) →
  (x = 6 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_other_endpoint_coordinates_l3771_377178


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l3771_377117

theorem greatest_three_digit_multiple_of_23 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 23 ∣ n → n ≤ 989 := by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l3771_377117


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3771_377191

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 + Complex.I)) :
  z.im = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3771_377191


namespace NUMINAMATH_CALUDE_david_solo_completion_time_l3771_377152

/-- The number of days it takes David to complete the job alone -/
def david_solo_days : ℝ := 12

/-- The number of days David works alone before Moore joins -/
def david_solo_work : ℝ := 6

/-- The number of days it takes David and Moore to complete the job together -/
def david_moore_total : ℝ := 6

/-- The number of days it takes David and Moore to complete the remaining job after David works alone -/
def david_moore_remaining : ℝ := 3

theorem david_solo_completion_time :
  (david_solo_work / david_solo_days) + 
  (david_moore_remaining / david_moore_total) = 1 :=
sorry

end NUMINAMATH_CALUDE_david_solo_completion_time_l3771_377152


namespace NUMINAMATH_CALUDE_hyperbola_axes_length_l3771_377195

theorem hyperbola_axes_length (x y : ℝ) :
  x^2 - 8*y^2 = 32 →
  ∃ (real_axis imaginary_axis : ℝ),
    real_axis = 8 * Real.sqrt 2 ∧
    imaginary_axis = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_axes_length_l3771_377195


namespace NUMINAMATH_CALUDE_pasta_bins_l3771_377151

theorem pasta_bins (total_bins soup_bins vegetable_bins : ℝ) 
  (h_total : total_bins = 0.75)
  (h_soup : soup_bins = 0.12)
  (h_vegetable : vegetable_bins = 0.12) :
  total_bins - soup_bins - vegetable_bins = 0.51 := by
sorry

end NUMINAMATH_CALUDE_pasta_bins_l3771_377151


namespace NUMINAMATH_CALUDE_total_homework_pages_l3771_377185

def math_homework_pages : ℕ := 10
def reading_homework_difference : ℕ := 3

def total_pages : ℕ := math_homework_pages + (math_homework_pages + reading_homework_difference)

theorem total_homework_pages : total_pages = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_homework_pages_l3771_377185


namespace NUMINAMATH_CALUDE_alcohol_water_mixture_ratio_l3771_377166

/-- Given two jars of alcohol-water mixtures with volumes V and 2V, and ratios p:1 and q:1 respectively,
    the ratio of alcohol to water in the resulting mixture is (p(q+1) + 2p + 2q) : (q+1 + 2p + 2) -/
theorem alcohol_water_mixture_ratio (V p q : ℝ) (hV : V > 0) (hp : p > 0) (hq : q > 0) :
  let first_jar_alcohol := (p / (p + 1)) * V
  let first_jar_water := (1 / (p + 1)) * V
  let second_jar_alcohol := (2 * q / (q + 1)) * V
  let second_jar_water := (2 / (q + 1)) * V
  let total_alcohol := first_jar_alcohol + second_jar_alcohol
  let total_water := first_jar_water + second_jar_water
  total_alcohol / total_water = (p * (q + 1) + 2 * p + 2 * q) / (q + 1 + 2 * p + 2) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_water_mixture_ratio_l3771_377166


namespace NUMINAMATH_CALUDE_scalene_triangle_gp_ratio_bounds_l3771_377111

/-- A scalene triangle with sides in geometric progression -/
structure ScaleneTriangleGP where
  -- The first side of the triangle
  a : ℝ
  -- The common ratio of the geometric progression
  q : ℝ
  -- Ensure the triangle is scalene and sides are positive
  h_scalene : a ≠ a * q ∧ a * q ≠ a * q^2 ∧ a ≠ a * q^2 ∧ a > 0 ∧ q > 0

/-- The common ratio of a scalene triangle with sides in geometric progression
    must be between (1 - √5)/2 and (1 + √5)/2 -/
theorem scalene_triangle_gp_ratio_bounds (t : ScaleneTriangleGP) :
  (1 - Real.sqrt 5) / 2 < t.q ∧ t.q < (1 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_scalene_triangle_gp_ratio_bounds_l3771_377111


namespace NUMINAMATH_CALUDE_smallest_side_difference_l3771_377109

def is_valid_triangle (pq qr pr : ℕ) : Prop :=
  pq + qr > pr ∧ pq + pr > qr ∧ qr + pr > pq

theorem smallest_side_difference (pq qr pr : ℕ) :
  pq + qr + pr = 3030 →
  pq < qr →
  qr ≤ pr →
  is_valid_triangle pq qr pr →
  (∀ pq' qr' pr' : ℕ, 
    pq' + qr' + pr' = 3030 →
    pq' < qr' →
    qr' ≤ pr' →
    is_valid_triangle pq' qr' pr' →
    qr - pq ≤ qr' - pq') →
  qr - pq = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_side_difference_l3771_377109


namespace NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l3771_377163

theorem tan_neg_seven_pi_sixths : Real.tan (-7 * Real.pi / 6) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l3771_377163


namespace NUMINAMATH_CALUDE_mothers_age_is_50_point_5_l3771_377187

def allen_age (mother_age : ℝ) : ℝ := mother_age - 30

theorem mothers_age_is_50_point_5 (mother_age : ℝ) :
  allen_age mother_age = mother_age - 30 →
  allen_age mother_age + 7 + (mother_age + 7) = 85 →
  mother_age = 50.5 := by
  sorry

end NUMINAMATH_CALUDE_mothers_age_is_50_point_5_l3771_377187


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3771_377132

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * α) = Real.sin (π / 4 - α)) : 
  Real.sin (2 * α) = -17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3771_377132


namespace NUMINAMATH_CALUDE_lucy_cookie_packs_l3771_377182

/-- The number of cookie packs Lucy bought at the grocery store. -/
def cookie_packs : ℕ := 28 - 16

/-- The total number of grocery packs Lucy bought. -/
def total_packs : ℕ := 28

/-- The number of noodle packs Lucy bought. -/
def noodle_packs : ℕ := 16

theorem lucy_cookie_packs : 
  cookie_packs = 12 ∧ 
  total_packs = cookie_packs + noodle_packs :=
by sorry

end NUMINAMATH_CALUDE_lucy_cookie_packs_l3771_377182


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_twice_l3771_377105

theorem parabola_intersects_x_axis_twice (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ m * x₁^2 + (m - 3) * x₁ - 1 = 0 ∧ m * x₂^2 + (m - 3) * x₂ - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_twice_l3771_377105


namespace NUMINAMATH_CALUDE_debby_water_consumption_l3771_377150

/-- Given that Debby bought 355 bottles of water that would last her 71 days,
    prove that she drinks 5 bottles per day. -/
theorem debby_water_consumption (total_bottles : ℕ) (total_days : ℕ) 
  (h1 : total_bottles = 355) (h2 : total_days = 71) :
  total_bottles / total_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_consumption_l3771_377150


namespace NUMINAMATH_CALUDE_min_gumballs_for_five_correct_l3771_377173

/-- Represents the number of gumballs of each color -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- The minimum number of gumballs needed to guarantee 5 of the same color -/
def minGumballsForFive (m : GumballMachine) : ℕ := 17

/-- Theorem stating that for the given gumball machine, 
    17 is the minimum number of gumballs needed to guarantee 5 of the same color -/
theorem min_gumballs_for_five_correct (m : GumballMachine) 
  (h_red : m.red = 12) 
  (h_white : m.white = 10) 
  (h_blue : m.blue = 9) 
  (h_green : m.green = 8) : 
  minGumballsForFive m = 17 := by
  sorry


end NUMINAMATH_CALUDE_min_gumballs_for_five_correct_l3771_377173


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l3771_377139

/-- The total cost of apples given weight, price per kg, and packaging fee -/
def total_cost (weight : ℝ) (price_per_kg : ℝ) (packaging_fee : ℝ) : ℝ :=
  weight * (price_per_kg + packaging_fee)

/-- Theorem stating that the total cost of 2.5 kg of apples is 38.875 -/
theorem apple_cost_calculation :
  total_cost 2.5 15.3 0.25 = 38.875 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l3771_377139


namespace NUMINAMATH_CALUDE_tan_cube_identity_l3771_377190

theorem tan_cube_identity (x y : ℝ) (φ : ℝ) (h : Real.tan φ ^ 3 = x / y) :
  x / Real.sin φ + y / Real.cos φ = (x ^ (2/3) + y ^ (2/3)) ^ (3/2) := by
  sorry

end NUMINAMATH_CALUDE_tan_cube_identity_l3771_377190


namespace NUMINAMATH_CALUDE_opposite_numbers_iff_differ_in_sign_l3771_377116

/-- Two real numbers are opposite if and only if they differ only in their sign -/
theorem opposite_numbers_iff_differ_in_sign (a b : ℝ) : 
  (a = -b) ↔ (abs a = abs b) := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_iff_differ_in_sign_l3771_377116


namespace NUMINAMATH_CALUDE_find_other_number_l3771_377193

theorem find_other_number (x y : ℕ+) 
  (h_lcm : Nat.lcm x y = 5040)
  (h_gcd : Nat.gcd x y = 24)
  (h_x : x = 240) :
  y = 504 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l3771_377193


namespace NUMINAMATH_CALUDE_octal_26_is_decimal_22_l3771_377153

def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones_digit := octal % 10
  let eights_digit := octal / 10
  eights_digit * 8 + ones_digit

theorem octal_26_is_decimal_22 : octal_to_decimal 26 = 22 := by
  sorry

end NUMINAMATH_CALUDE_octal_26_is_decimal_22_l3771_377153


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3771_377100

theorem initial_money_calculation (M : ℚ) : 
  (((M * (3/5) * (2/3) * (3/4) * (4/7)) : ℚ) = 700) → M = 24500/6 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l3771_377100


namespace NUMINAMATH_CALUDE_bridge_length_l3771_377102

/-- The length of a bridge given specific train conditions -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  train_speed = 45 * 1000 / 3600 →
  crossing_time = 30 →
  train_speed * crossing_time - train_length = 205 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3771_377102


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3771_377165

/-- A circle intersected by four equally spaced parallel lines -/
structure ParallelLinesCircle where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The lengths of the four chords created by the parallel lines -/
  chord_lengths : Fin 4 → ℝ
  /-- The chords have the specified lengths -/
  chord_length_values : chord_lengths = ![42, 36, 36, 30]
  /-- The parallel lines are equally spaced -/
  equally_spaced : ∀ i j : Fin 3, d = d

/-- The theorem stating that the distance between adjacent parallel lines is √2 -/
theorem parallel_lines_distance (c : ParallelLinesCircle) : c.d = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3771_377165


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3771_377162

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3771_377162


namespace NUMINAMATH_CALUDE_charles_pictures_l3771_377175

theorem charles_pictures (initial_papers : ℕ) (today_pictures : ℕ) (yesterday_before_work : ℕ) (papers_left : ℕ) :
  initial_papers = 20 →
  today_pictures = 6 →
  yesterday_before_work = 6 →
  papers_left = 2 →
  initial_papers - today_pictures - yesterday_before_work - papers_left = 6 := by
  sorry

end NUMINAMATH_CALUDE_charles_pictures_l3771_377175


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_equation_l3771_377119

theorem unique_integer_satisfying_equation : 
  ∃! (n : ℕ), n > 0 ∧ (n + 1500) / 90 = ⌊Real.sqrt n⌋ ∧ n = 4530 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_equation_l3771_377119


namespace NUMINAMATH_CALUDE_sticker_distribution_l3771_377142

/-- The number of ways to partition n identical objects into at most k parts -/
def partitions (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 30 ways to partition 10 identical objects into at most 5 parts -/
theorem sticker_distribution : partitions 10 5 = 30 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3771_377142


namespace NUMINAMATH_CALUDE_inequality_solution_expression_value_l3771_377155

-- Problem 1
theorem inequality_solution (x : ℝ) : 2*x - 3 > x + 1 ↔ x > 4 := by sorry

-- Problem 2
theorem expression_value (a b : ℝ) (h : a^2 + 3*a*b = 5) : 
  (a + b) * (a + 2*b) - 2*b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_expression_value_l3771_377155


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3771_377131

theorem largest_n_satisfying_inequality :
  ∀ n : ℤ, (1/2 : ℚ) + n/9 < 1 ↔ n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3771_377131


namespace NUMINAMATH_CALUDE_total_students_is_1480_l3771_377194

/-- Represents a campus in the school district -/
structure Campus where
  grades : ℕ  -- number of grades
  students_per_grade : ℕ  -- number of students per grade
  extra_students : ℕ  -- number of extra students in special programs

/-- Calculates the total number of students in a campus -/
def campus_total (c : Campus) : ℕ :=
  c.grades * c.students_per_grade + c.extra_students

/-- The school district with its three campuses -/
structure SchoolDistrict where
  campus_a : Campus
  campus_b : Campus
  campus_c : Campus

/-- Represents the specific school district described in the problem -/
def our_district : SchoolDistrict :=
  { campus_a := { grades := 5, students_per_grade := 100, extra_students := 30 }
  , campus_b := { grades := 5, students_per_grade := 120, extra_students := 0 }
  , campus_c := { grades := 2, students_per_grade := 150, extra_students := 50 }
  }

/-- Theorem stating that the total number of students in our school district is 1480 -/
theorem total_students_is_1480 : 
  campus_total our_district.campus_a + 
  campus_total our_district.campus_b + 
  campus_total our_district.campus_c = 1480 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_1480_l3771_377194


namespace NUMINAMATH_CALUDE_shooting_competition_sequences_l3771_377118

/-- The number of ways to arrange a multiset with 4 A's, 3 B's, 2 C's, and 1 D -/
def shooting_sequences : ℕ := 12600

/-- The total number of targets -/
def total_targets : ℕ := 10

/-- The number of targets in column A -/
def targets_A : ℕ := 4

/-- The number of targets in column B -/
def targets_B : ℕ := 3

/-- The number of targets in column C -/
def targets_C : ℕ := 2

/-- The number of targets in column D -/
def targets_D : ℕ := 1

theorem shooting_competition_sequences :
  shooting_sequences = (total_targets.factorial) / 
    (targets_A.factorial * targets_B.factorial * 
     targets_C.factorial * targets_D.factorial) :=
by sorry

end NUMINAMATH_CALUDE_shooting_competition_sequences_l3771_377118


namespace NUMINAMATH_CALUDE_correct_cobs_per_row_l3771_377184

/-- Represents the number of corn cobs in each row -/
def cobs_per_row : ℕ := 4

/-- Represents the number of rows in the first field -/
def rows_field1 : ℕ := 13

/-- Represents the number of rows in the second field -/
def rows_field2 : ℕ := 16

/-- Represents the total number of corn cobs -/
def total_cobs : ℕ := 116

/-- Theorem stating that the number of corn cobs per row is correct -/
theorem correct_cobs_per_row : 
  cobs_per_row * rows_field1 + cobs_per_row * rows_field2 = total_cobs := by
  sorry

end NUMINAMATH_CALUDE_correct_cobs_per_row_l3771_377184


namespace NUMINAMATH_CALUDE_definite_integral_arctg_x_l3771_377148

theorem definite_integral_arctg_x : 
  ∫ x in (0 : ℝ)..1, (4 * Real.arctan x - x) / (1 + x^2) = (π^2 - 4 * Real.log 2) / 8 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_arctg_x_l3771_377148


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3771_377156

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem fifteenth_term_of_sequence :
  let a₁ : ℤ := -3
  let d : ℤ := 4
  arithmetic_sequence a₁ d 15 = 53 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3771_377156


namespace NUMINAMATH_CALUDE_jessica_seashells_l3771_377129

theorem jessica_seashells (joan_shells jessica_shells total_shells : ℕ) 
  (h1 : joan_shells = 6)
  (h2 : total_shells = 14)
  (h3 : joan_shells + jessica_shells = total_shells) :
  jessica_shells = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessica_seashells_l3771_377129


namespace NUMINAMATH_CALUDE_sue_shoe_probability_l3771_377180

/-- Represents the distribution of shoes by color --/
structure ShoeDistribution where
  black : Nat
  brown : Nat
  gray : Nat
  red : Nat

/-- Calculates the probability of selecting two shoes of the same color
    with one left and one right, given a shoe distribution --/
def samePairProbability (d : ShoeDistribution) : Rat :=
  let totalShoes := 2 * (d.black + d.brown + d.gray + d.red)
  let blackProb := (d.black : Rat) * (d.black - 1) / (totalShoes * (totalShoes - 1))
  let brownProb := (d.brown : Rat) * (d.brown - 1) / (totalShoes * (totalShoes - 1))
  let grayProb := (d.gray : Rat) * (d.gray - 1) / (totalShoes * (totalShoes - 1))
  let redProb := (d.red : Rat) * (d.red - 1) / (totalShoes * (totalShoes - 1))
  blackProb + brownProb + grayProb + redProb

theorem sue_shoe_probability :
  let sueShoes := ShoeDistribution.mk 7 4 2 1
  samePairProbability sueShoes = 20 / 63 := by
  sorry

end NUMINAMATH_CALUDE_sue_shoe_probability_l3771_377180


namespace NUMINAMATH_CALUDE_negative_expression_l3771_377135

theorem negative_expression : 
  (-(-3) > 0) ∧ (-3^2 < 0) ∧ ((-3)^2 > 0) ∧ (|(-3)| > 0) :=
by sorry


end NUMINAMATH_CALUDE_negative_expression_l3771_377135


namespace NUMINAMATH_CALUDE_sqrt2_irrational_sqrt2_approximation_no_exact_rational_sqrt2_l3771_377181

-- Define √2 as an irrational number
noncomputable def sqrt2 : ℝ := Real.sqrt 2

-- Statement that √2 is irrational
theorem sqrt2_irrational : Irrational sqrt2 := sorry

-- Statement that √2 can be approximated by rationals
theorem sqrt2_approximation :
  ∀ ε > 0, ∃ p q : ℤ, q ≠ 0 ∧ |((p : ℝ) / q)^2 - 2| < ε := sorry

-- Statement that no rational number exactly equals √2
theorem no_exact_rational_sqrt2 :
  ¬∃ p q : ℤ, q ≠ 0 ∧ ((p : ℝ) / q)^2 = 2 := sorry

end NUMINAMATH_CALUDE_sqrt2_irrational_sqrt2_approximation_no_exact_rational_sqrt2_l3771_377181


namespace NUMINAMATH_CALUDE_polynomial_factor_l3771_377164

theorem polynomial_factor (a : ℚ) : 
  (∀ x : ℚ, (x + 5) ∣ (a * x^4 + 12 * x^2 - 5 * a * x + 42)) → 
  a = -57/100 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_l3771_377164


namespace NUMINAMATH_CALUDE_log_equation_holds_l3771_377107

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 4 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_holds_l3771_377107


namespace NUMINAMATH_CALUDE_max_value_sin_cos_function_l3771_377127

theorem max_value_sin_cos_function :
  ∃ (M : ℝ), M = 1/2 - Real.sqrt 3/4 ∧
  ∀ (x : ℝ), Real.sin (3*Real.pi/2 + x) * Real.cos (Real.pi/6 - x) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_function_l3771_377127


namespace NUMINAMATH_CALUDE_subtracted_number_l3771_377110

theorem subtracted_number (x y : ℤ) : x = 125 ∧ 2 * x - y = 112 → y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l3771_377110


namespace NUMINAMATH_CALUDE_two_integers_problem_l3771_377199

theorem two_integers_problem (x y : ℕ+) 
  (h1 : x * y = 18)
  (h2 : x - y = 4) : 
  x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_problem_l3771_377199


namespace NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l3771_377189

-- Define the universe of goods
variable (Goods : Type)

-- Define predicates for "cheap" and "good quality"
variable (cheap : Goods → Prop)
variable (good_quality : Goods → Prop)

-- State the given condition
variable (h : ∀ g : Goods, cheap g → ¬(good_quality g))

-- Theorem statement
theorem not_cheap_necessary_for_good_quality :
  ∀ g : Goods, good_quality g → ¬(cheap g) :=
by
  sorry

end NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l3771_377189


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l3771_377122

theorem cubic_equation_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p = 12 →
  q^3 - 6*q^2 + 11*q = 12 →
  r^3 - 6*r^2 + 11*r = 12 →
  p * q / r + q * r / p + r * p / q = -23/12 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l3771_377122


namespace NUMINAMATH_CALUDE_a_45_value_l3771_377174

def a : ℕ → ℤ
  | 0 => 11
  | 1 => 11
  | n + 2 => sorry  -- This will be defined using the recurrence relation

-- Define the recurrence relation
axiom a_rec : ∀ (m n : ℕ), a (m + n) = (1/2) * (a (2*m) + a (2*n)) - (m - n)^2

theorem a_45_value : a 45 = 1991 := by
  sorry

end NUMINAMATH_CALUDE_a_45_value_l3771_377174


namespace NUMINAMATH_CALUDE_final_balance_calculation_l3771_377172

def calculate_final_balance (initial_investment : ℝ) (interest_rates : List ℝ) 
  (deposits : List (Nat × ℝ)) (withdrawals : List (Nat × ℝ)) : ℝ :=
  sorry

theorem final_balance_calculation :
  let initial_investment : ℝ := 10000
  let interest_rates : List ℝ := [0.02, 0.03, 0.04, 0.025, 0.035, 0.04, 0.03, 0.035, 0.04]
  let deposits : List (Nat × ℝ) := [(3, 1000), (6, 1000)]
  let withdrawals : List (Nat × ℝ) := [(9, 2000)]
  calculate_final_balance initial_investment interest_rates deposits withdrawals = 13696.95 := by
  sorry

end NUMINAMATH_CALUDE_final_balance_calculation_l3771_377172


namespace NUMINAMATH_CALUDE_complex_abs_calculation_l3771_377114

def z : ℂ := 7 + 3 * Complex.I

theorem complex_abs_calculation : Complex.abs (z^2 + 4*z + 40) = 54 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_calculation_l3771_377114


namespace NUMINAMATH_CALUDE_winter_clothing_boxes_l3771_377120

/-- Given that each box contains 10 pieces of clothing and the total number of pieces is 60,
    prove that the number of boxes is 6. -/
theorem winter_clothing_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (num_boxes : ℕ) :
  pieces_per_box = 10 →
  total_pieces = 60 →
  num_boxes * pieces_per_box = total_pieces →
  num_boxes = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_winter_clothing_boxes_l3771_377120


namespace NUMINAMATH_CALUDE_decimal_arithmetic_l3771_377121

theorem decimal_arithmetic : 3.456 - 1.78 + 0.032 = 1.678 := by
  sorry

end NUMINAMATH_CALUDE_decimal_arithmetic_l3771_377121


namespace NUMINAMATH_CALUDE_code_decryption_probability_l3771_377104

theorem code_decryption_probability :
  let p := 1 / 5  -- probability of success for each person
  let n := 3      -- number of people
  let prob_at_least_two := 
    Finset.sum (Finset.range (n - 1 + 1)) (fun k => 
      if k ≥ 2 then Nat.choose n k * p^k * (1 - p)^(n - k) else 0)
  prob_at_least_two = 13 / 125 := by
sorry

end NUMINAMATH_CALUDE_code_decryption_probability_l3771_377104


namespace NUMINAMATH_CALUDE_power_composition_l3771_377146

theorem power_composition (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3*a + 2*b) = 72 := by
  sorry

end NUMINAMATH_CALUDE_power_composition_l3771_377146


namespace NUMINAMATH_CALUDE_least_number_with_remainder_seven_l3771_377149

theorem least_number_with_remainder_seven (n : ℕ) : n = 1547 ↔ 
  (∀ d ∈ ({11, 17, 21, 29, 35} : Set ℕ), n % d = 7) ∧
  (∀ m < n, ∃ d ∈ ({11, 17, 21, 29, 35} : Set ℕ), m % d ≠ 7) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_seven_l3771_377149
