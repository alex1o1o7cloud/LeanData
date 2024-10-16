import Mathlib

namespace NUMINAMATH_CALUDE_sequence_inequality_l2640_264015

theorem sequence_inequality (a : ℕ → ℕ) 
  (h0 : ∀ n, a n > 0)
  (h1 : a 1 > a 0)
  (h2 : ∀ n ∈ Finset.range 99, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2640_264015


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2640_264076

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 2x² + (2 + 1/2)x + 1/2 -/
def a : ℚ := 2
def b : ℚ := 5/2
def c : ℚ := 1/2

theorem quadratic_discriminant :
  discriminant a b c = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2640_264076


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_twelve_l2640_264003

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := (n : ℚ) / 2 * (2 * a₁ + (n - 1) * d)

theorem arithmetic_sequence_sum_twelve (a₁ d : ℚ) :
  arithmetic_sequence a₁ d 5 = 1 →
  arithmetic_sequence a₁ d 17 = 18 →
  arithmetic_sum a₁ d 12 = 37.5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_twelve_l2640_264003


namespace NUMINAMATH_CALUDE_vector_problem_l2640_264054

/-- Given two planar vectors a and b, where a is orthogonal to b and their sum with a third vector c is zero, 
    prove that the first component of a is 2 and the magnitude of c is 5. -/
theorem vector_problem (m : ℝ) :
  let a : ℝ × ℝ := (m, -1)
  let b : ℝ × ℝ := (2, 4)
  let c : ℝ × ℝ := (-a.1 - b.1, -a.2 - b.2)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⊥ b
  (m = 2 ∧ Real.sqrt ((c.1 ^ 2) + (c.2 ^ 2)) = 5) := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l2640_264054


namespace NUMINAMATH_CALUDE_four_digit_number_count_special_four_digit_number_count_l2640_264004

/-- The set of available digits --/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A four-digit number with no repeating digits --/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  h₅ : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄
  h₆ : d₁ ≠ 0  -- Ensures it's a four-digit number

/-- The set of all valid four-digit numbers --/
def allFourDigitNumbers : Finset FourDigitNumber := sorry

/-- Four-digit numbers with tens digit larger than both units and hundreds digits --/
def specialFourDigitNumbers : Finset FourDigitNumber :=
  allFourDigitNumbers.filter (fun n => n.d₃ > n.d₂ ∧ n.d₃ > n.d₄)

theorem four_digit_number_count :
  Finset.card allFourDigitNumbers = 300 := by sorry

theorem special_four_digit_number_count :
  Finset.card specialFourDigitNumbers = 100 := by sorry

end NUMINAMATH_CALUDE_four_digit_number_count_special_four_digit_number_count_l2640_264004


namespace NUMINAMATH_CALUDE_expression_value_l2640_264049

theorem expression_value :
  let x : ℤ := 3
  let y : ℤ := 2
  let z : ℤ := 4
  let w : ℤ := -1
  x^2 * y - 2 * x * y + 3 * x * z - (x + y) * (y + z) * (z + w) = -48 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2640_264049


namespace NUMINAMATH_CALUDE_line_constant_value_l2640_264032

/-- Given a line passing through points (m, n) and (m + 2, n + 0.5) with equation x = k * y + 5, prove that k = 4 -/
theorem line_constant_value (m n k : ℝ) : 
  (m = k * n + 5) ∧ (m + 2 = k * (n + 0.5) + 5) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_constant_value_l2640_264032


namespace NUMINAMATH_CALUDE_fib_divisibility_l2640_264050

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- State the theorem
theorem fib_divisibility (m n : ℕ) (h : m > 0) (h' : n > 0) : 
  m ∣ n → (fib (m - 1)) ∣ (fib (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fib_divisibility_l2640_264050


namespace NUMINAMATH_CALUDE_largest_invertible_interval_for_g_l2640_264093

-- Define the function g(x)
def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

-- Define the theorem
theorem largest_invertible_interval_for_g :
  ∃ (a : ℝ), 
    (∀ (I : Set ℝ), (2 ∈ I) → (∀ (x y : ℝ), x ∈ I → y ∈ I → x ≠ y → g x ≠ g y) → 
      I ⊆ {x : ℝ | a ≤ x}) ∧
    ({x : ℝ | a ≤ x} ⊆ {x : ℝ | ∀ (y : ℝ), y ∈ {x : ℝ | a ≤ x} → y ≠ x → g y ≠ g x}) ∧
    a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_largest_invertible_interval_for_g_l2640_264093


namespace NUMINAMATH_CALUDE_circle_center_proof_l2640_264017

/-- The equation of a circle in polar coordinates -/
def polar_circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The center of a circle in polar coordinates -/
def circle_center : ℝ × ℝ := (1, 0)

/-- Theorem stating that the center of the circle ρ = 2cosθ is at (1, 0) in polar coordinates -/
theorem circle_center_proof :
  ∀ ρ θ : ℝ, polar_circle_equation ρ θ → circle_center = (1, 0) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_center_proof_l2640_264017


namespace NUMINAMATH_CALUDE_plywood_area_conservation_l2640_264086

theorem plywood_area_conservation (A W : ℝ) (h : A > 0 ∧ W > 0) :
  let L : ℝ := A / W
  let L' : ℝ := A / (2 * W)
  A = W * L ∧ A = (2 * W) * L' := by sorry

end NUMINAMATH_CALUDE_plywood_area_conservation_l2640_264086


namespace NUMINAMATH_CALUDE_solution_range_l2640_264016

theorem solution_range (x : ℝ) : 
  x > 9 → 
  Real.sqrt (x - 5 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 5 * Real.sqrt (x - 9)) - 3 → 
  x ≥ 20.80 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2640_264016


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l2640_264007

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℝ) 
  (set2_count : ℕ) (set2_mean : ℝ) : 
  set1_count = 5 → 
  set1_mean = 16 → 
  set2_count = 8 → 
  set2_mean = 21 → 
  let total_count := set1_count + set2_count
  let combined_mean := (set1_count * set1_mean + set2_count * set2_mean) / total_count
  combined_mean = 19.08 := by
sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l2640_264007


namespace NUMINAMATH_CALUDE_q_is_false_l2640_264067

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬(¬p)) : ¬q :=
sorry

end NUMINAMATH_CALUDE_q_is_false_l2640_264067


namespace NUMINAMATH_CALUDE_complex_square_quadrant_l2640_264071

theorem complex_square_quadrant (z : ℂ) : 
  z = Complex.exp (Complex.I * Real.pi * (5/12)) → 
  (z^2).re < 0 ∧ (z^2).im > 0 :=
sorry

end NUMINAMATH_CALUDE_complex_square_quadrant_l2640_264071


namespace NUMINAMATH_CALUDE_product_of_base_8_digits_l2640_264074

-- Define the base 10 number
def base_10_number : ℕ := 7354

-- Function to convert a number from base 10 to base 8
def to_base_8 (n : ℕ) : List ℕ :=
  sorry

-- Function to calculate the product of a list of numbers
def product_of_list (l : List ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem product_of_base_8_digits :
  product_of_list (to_base_8 base_10_number) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_base_8_digits_l2640_264074


namespace NUMINAMATH_CALUDE_toys_per_week_l2640_264095

/-- The number of days worked per week -/
def days_per_week : ℕ := 3

/-- The number of toys produced per day -/
def toys_per_day : ℝ := 2133.3333333333335

/-- Theorem: The number of toys produced per week is 6400 -/
theorem toys_per_week : ℕ := by
  sorry

end NUMINAMATH_CALUDE_toys_per_week_l2640_264095


namespace NUMINAMATH_CALUDE_marks_increase_ratio_class_marks_double_l2640_264031

/-- Given a class of students, prove that if their marks are increased by a certain ratio,
    the ratio of new marks to original marks can be determined by the new and old averages. -/
theorem marks_increase_ratio (n : ℕ) (old_avg new_avg : ℚ) :
  n > 0 →
  old_avg > 0 →
  new_avg > old_avg →
  (n * new_avg) / (n * old_avg) = new_avg / old_avg := by sorry

/-- In a class of 11 students, if the average marks increase from 36 to 72,
    prove that the ratio of new marks to original marks is 2. -/
theorem class_marks_double :
  let n : ℕ := 11
  let old_avg : ℚ := 36
  let new_avg : ℚ := 72
  (n * new_avg) / (n * old_avg) = 2 := by sorry

end NUMINAMATH_CALUDE_marks_increase_ratio_class_marks_double_l2640_264031


namespace NUMINAMATH_CALUDE_binary_multiplication_addition_l2640_264018

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits (least significant bit first) -/
def Binary := List Bool

def binary_11011 : Binary := [true, true, false, true, true]
def binary_111 : Binary := [true, true, true]
def binary_1010 : Binary := [false, true, false, true]
def binary_11000111 : Binary := [true, true, true, false, false, false, true, true]

theorem binary_multiplication_addition :
  (binaryToDecimal binary_11011 * binaryToDecimal binary_111 + binaryToDecimal binary_1010) =
  binaryToDecimal binary_11000111 := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_addition_l2640_264018


namespace NUMINAMATH_CALUDE_length_AB_given_P_Q_positions_AB_length_is_189_l2640_264077

/-- Represents a point on a line segment -/
structure PointOnSegment (A B : ℝ) where
  x : ℝ
  h1 : A ≤ x
  h2 : x ≤ B

/-- Theorem: Length of AB given P and Q positions -/
theorem length_AB_given_P_Q_positions
  (A B : ℝ)
  (P : PointOnSegment A B)
  (Q : PointOnSegment A B)
  (h_same_side : (P.x - (A + B) / 2) * (Q.x - (A + B) / 2) > 0)
  (h_P_ratio : P.x - A = 3 / 7 * (B - A))
  (h_Q_ratio : Q.x - A = 4 / 9 * (B - A))
  (h_PQ_distance : |Q.x - P.x| = 3)
  : B - A = 189 := by
  sorry

/-- Corollary: AB length is 189 -/
theorem AB_length_is_189 : ∃ A B : ℝ, B - A = 189 ∧ 
  ∃ (P Q : PointOnSegment A B), 
    (P.x - (A + B) / 2) * (Q.x - (A + B) / 2) > 0 ∧
    P.x - A = 3 / 7 * (B - A) ∧
    Q.x - A = 4 / 9 * (B - A) ∧
    |Q.x - P.x| = 3 := by
  sorry

end NUMINAMATH_CALUDE_length_AB_given_P_Q_positions_AB_length_is_189_l2640_264077


namespace NUMINAMATH_CALUDE_oranges_picked_total_l2640_264066

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 122

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 105

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_total :
  total_oranges = 227 :=
by sorry

end NUMINAMATH_CALUDE_oranges_picked_total_l2640_264066


namespace NUMINAMATH_CALUDE_max_product_digits_l2640_264099

theorem max_product_digits : ∀ a b : ℕ, 
  10000 ≤ a ∧ a < 100000 → 1000 ≤ b ∧ b < 10000 → 
  a * b < 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_digits_l2640_264099


namespace NUMINAMATH_CALUDE_complex_simplification_l2640_264029

theorem complex_simplification :
  (-5 + 3*I : ℂ) - (2 - 7*I) + (1 + 2*I) * (4 - 3*I) = 3 + 15*I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2640_264029


namespace NUMINAMATH_CALUDE_sqrt_3_binary_representation_l2640_264058

open Real

theorem sqrt_3_binary_representation (n : ℕ+) :
  ¬ (2^(n.val + 1) ∣ ⌊2^(2 * n.val) * Real.sqrt 3⌋) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_binary_representation_l2640_264058


namespace NUMINAMATH_CALUDE_constant_term_product_l2640_264084

theorem constant_term_product (x : ℝ) : 
  (x^4 + x^2 + 7) * (2*x^5 + 3*x^3 + 10) = 70 + x * (2*x^8 + 3*x^6 + 20*x^4 + 3*x^7 + 10*x^5 + 7*2*x^5 + 7*3*x^3) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_product_l2640_264084


namespace NUMINAMATH_CALUDE_remainder_sum_l2640_264030

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 53)
  (hd : d % 45 = 28) : 
  (c + d) % 15 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l2640_264030


namespace NUMINAMATH_CALUDE_abs_of_nonnegative_l2640_264069

theorem abs_of_nonnegative (x : ℝ) : x ≥ 0 → |x| = x := by
  sorry

end NUMINAMATH_CALUDE_abs_of_nonnegative_l2640_264069


namespace NUMINAMATH_CALUDE_players_in_both_games_l2640_264020

theorem players_in_both_games (total : ℕ) (outdoor : ℕ) (indoor : ℕ) 
  (h1 : total = 400) 
  (h2 : outdoor = 350) 
  (h3 : indoor = 110) : 
  outdoor + indoor - total = 60 := by
  sorry

end NUMINAMATH_CALUDE_players_in_both_games_l2640_264020


namespace NUMINAMATH_CALUDE_gcd_linear_combination_l2640_264082

theorem gcd_linear_combination (a b d : ℕ) :
  d = Nat.gcd a b →
  d = Nat.gcd (5 * a + 3 * b) (13 * a + 8 * b) := by
  sorry

end NUMINAMATH_CALUDE_gcd_linear_combination_l2640_264082


namespace NUMINAMATH_CALUDE_greenfield_high_lockers_l2640_264052

/-- The cost in cents for each plastic digit used in labeling lockers -/
def digit_cost : ℚ := 3

/-- The total cost in dollars for labeling all lockers -/
def total_cost : ℚ := 273.39

/-- Calculates the cost of labeling lockers from 1 to n -/
def labeling_cost (n : ℕ) : ℚ :=
  let ones := min n 9
  let tens := min (n - 9) 90
  let hundreds := min (n - 99) 900
  let thousands := max (n - 999) 0
  (ones * digit_cost + 
   tens * 2 * digit_cost + 
   hundreds * 3 * digit_cost + 
   thousands * 4 * digit_cost) / 100

/-- The number of lockers at Greenfield High -/
def num_lockers : ℕ := 2555

theorem greenfield_high_lockers : 
  labeling_cost num_lockers = total_cost :=
sorry

end NUMINAMATH_CALUDE_greenfield_high_lockers_l2640_264052


namespace NUMINAMATH_CALUDE_sum_of_roots_l2640_264023

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2640_264023


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2640_264039

theorem min_value_expression (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  a^2 + c^2 + 1/a^2 + c/a + 1/c^2 ≥ Real.sqrt 15 :=
sorry

theorem equality_condition (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  ∃ a c, a^2 + c^2 + 1/a^2 + c/a + 1/c^2 = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2640_264039


namespace NUMINAMATH_CALUDE_birch_count_l2640_264078

/-- Represents the number of trees of each species in the forest --/
structure ForestComposition where
  oak : ℕ
  pine : ℕ
  spruce : ℕ
  birch : ℕ

/-- Theorem stating the number of birch trees in the forest --/
theorem birch_count (forest : ForestComposition) : forest.birch = 2160 :=
  by
  have total_trees : forest.oak + forest.pine + forest.spruce + forest.birch = 4000 := by sorry
  have spruce_percentage : forest.spruce = 4000 * 10 / 100 := by sorry
  have pine_percentage : forest.pine = 4000 * 13 / 100 := by sorry
  have oak_count : forest.oak = forest.spruce + forest.pine := by sorry
  sorry


end NUMINAMATH_CALUDE_birch_count_l2640_264078


namespace NUMINAMATH_CALUDE_correct_average_weight_l2640_264060

theorem correct_average_weight (n : ℕ) (initial_avg : ℚ) (misread_weight : ℚ) (correct_weight : ℚ) :
  n = 20 →
  initial_avg = 58.4 →
  misread_weight = 56 →
  correct_weight = 66 →
  (n : ℚ) * initial_avg + (correct_weight - misread_weight) = n * 58.9 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l2640_264060


namespace NUMINAMATH_CALUDE_prime_divides_power_plus_one_l2640_264012

theorem prime_divides_power_plus_one (n b p : ℕ) :
  n ≠ 0 →
  b ≠ 0 →
  Nat.Prime p →
  Odd p →
  p ∣ b^(2^n) + 1 →
  ∃ m : ℕ, p = 2^(n+1) * m + 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_power_plus_one_l2640_264012


namespace NUMINAMATH_CALUDE_farm_animal_difference_l2640_264089

theorem farm_animal_difference (goats chickens ducks pigs : ℕ) : 
  goats = 66 →
  chickens = 2 * goats →
  ducks = (goats + chickens) / 2 →
  pigs = ducks / 3 →
  goats - pigs = 33 := by
sorry

end NUMINAMATH_CALUDE_farm_animal_difference_l2640_264089


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l2640_264008

/-- A geometric sequence with the given first four terms -/
def geometric_sequence (y : ℝ) : ℕ → ℝ
  | 0 => 3
  | 1 => 9 * y
  | 2 => 27 * y^2
  | 3 => 81 * y^3
  | n + 4 => geometric_sequence y 3 * (3 * y)^(n + 1)

/-- The fifth term of the geometric sequence is 243y^4 -/
theorem fifth_term_of_sequence (y : ℝ) :
  geometric_sequence y 4 = 243 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l2640_264008


namespace NUMINAMATH_CALUDE_prob_same_tails_value_l2640_264035

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The probability of getting tails on a single penny toss -/
def prob_tails : ℚ := 1/2

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := 2^keiko_pennies * 2^ephraim_pennies

/-- The number of favorable outcomes (same number of tails) -/
def favorable_outcomes : ℕ := 3

/-- The probability of Ephraim getting the same number of tails as Keiko -/
def prob_same_tails : ℚ := favorable_outcomes / total_outcomes

theorem prob_same_tails_value : prob_same_tails = 3/32 := by sorry

end NUMINAMATH_CALUDE_prob_same_tails_value_l2640_264035


namespace NUMINAMATH_CALUDE_constant_difference_of_equal_second_derivatives_l2640_264011

theorem constant_difference_of_equal_second_derivatives 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h : ∀ x, (deriv^[2] f) x = (deriv^[2] g) x) : 
  ∃ c : ℝ, ∀ x, f x - g x = c :=
sorry

end NUMINAMATH_CALUDE_constant_difference_of_equal_second_derivatives_l2640_264011


namespace NUMINAMATH_CALUDE_power_inequality_l2640_264027

def S : Set ℤ := {-2, -1, 0, 1, 2, 3}

theorem power_inequality (n : ℤ) : 
  n ∈ S → ((-1/2 : ℚ)^n > (-1/5 : ℚ)^n ↔ n = -1 ∨ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2640_264027


namespace NUMINAMATH_CALUDE_apple_orange_price_l2640_264056

theorem apple_orange_price (x y z : ℝ) 
  (eq1 : 24 * x = 28 * y)
  (eq2 : 45 * x + 60 * y = 1350 * z) :
  30 * x + 40 * y = 118.2857 * z :=
by sorry

end NUMINAMATH_CALUDE_apple_orange_price_l2640_264056


namespace NUMINAMATH_CALUDE_recommendation_plans_count_l2640_264070

/-- Represents the number of recommendation spots for each language -/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the gender distribution of selected candidates -/
structure SelectedCandidates :=
  (males : Nat)
  (females : Nat)

/-- Calculates the number of different recommendation plans -/
def countRecommendationPlans (spots : RecommendationSpots) (candidates : SelectedCandidates) : Nat :=
  sorry

theorem recommendation_plans_count :
  let spots : RecommendationSpots := ⟨2, 2, 1⟩
  let candidates : SelectedCandidates := ⟨3, 2⟩
  countRecommendationPlans spots candidates = 24 := by sorry

end NUMINAMATH_CALUDE_recommendation_plans_count_l2640_264070


namespace NUMINAMATH_CALUDE_circle_ratio_l2640_264097

theorem circle_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (h_area : π * r₂^2 - π * r₁^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l2640_264097


namespace NUMINAMATH_CALUDE_coin_fraction_missing_l2640_264062

theorem coin_fraction_missing (x : ℚ) : x > 0 →
  let lost := (1 / 3 : ℚ) * x
  let found := (3 / 4 : ℚ) * lost
  let remaining := x - lost + found
  x - remaining = (1 / 12 : ℚ) * x := by
  sorry

end NUMINAMATH_CALUDE_coin_fraction_missing_l2640_264062


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_non_obtuse_triangle_l2640_264044

/-- For any non-obtuse triangle with angles α, β, and γ, the sum of the sines of these angles 
is greater than the sum of the cosines of these angles. -/
theorem sine_cosine_inequality_non_obtuse_triangle (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi)
  (h_non_obtuse : α ≤ Real.pi/2 ∧ β ≤ Real.pi/2 ∧ γ ≤ Real.pi/2) :
  Real.sin α + Real.sin β + Real.sin γ > Real.cos α + Real.cos β + Real.cos γ :=
sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_non_obtuse_triangle_l2640_264044


namespace NUMINAMATH_CALUDE_cylinder_radius_proof_l2640_264033

/-- The radius of a cylinder with specific properties -/
def cylinder_radius : ℝ := 12

/-- The original height of the cylinder -/
def original_height : ℝ := 4

/-- The increase in radius or height -/
def increase : ℝ := 8

theorem cylinder_radius_proof :
  (cylinder_radius + increase)^2 * original_height = 
  cylinder_radius^2 * (original_height + increase) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_proof_l2640_264033


namespace NUMINAMATH_CALUDE_fifth_subject_score_l2640_264068

/-- Given a student with 5 subject scores, prove that if 4 scores are known
    and the average of all 5 scores is 73, then the fifth score must be 85. -/
theorem fifth_subject_score
  (scores : Fin 5 → ℕ)
  (known_scores : scores 0 = 55 ∧ scores 1 = 67 ∧ scores 2 = 76 ∧ scores 3 = 82)
  (average : (scores 0 + scores 1 + scores 2 + scores 3 + scores 4) / 5 = 73) :
  scores 4 = 85 := by
sorry

end NUMINAMATH_CALUDE_fifth_subject_score_l2640_264068


namespace NUMINAMATH_CALUDE_correct_households_using_both_l2640_264014

/-- Represents the survey results of household soap usage -/
structure SoapSurvey where
  total : Nat
  neither : Nat
  onlyA : Nat
  bothRatio : Nat
  /-- Proves that the number of households using both brands is 30 -/
  householdsUsingBoth : Nat

/-- The actual survey data -/
def actualSurvey : SoapSurvey := {
  total := 260
  neither := 80
  onlyA := 60
  bothRatio := 3
  householdsUsingBoth := 30
}

/-- Theorem stating that the number of households using both brands is correct -/
theorem correct_households_using_both (s : SoapSurvey) : 
  s.householdsUsingBoth = 30 ∧ 
  s.total = s.neither + s.onlyA + s.householdsUsingBoth + s.bothRatio * s.householdsUsingBoth :=
by sorry

end NUMINAMATH_CALUDE_correct_households_using_both_l2640_264014


namespace NUMINAMATH_CALUDE_unique_amazing_rectangle_l2640_264038

/-- An amazing rectangle is a rectangle where the area is equal to three times its perimeter,
    one side is double the other, and both sides are positive integers. -/
structure AmazingRectangle where
  width : ℕ+
  length : ℕ+
  is_double : length = 2 * width
  is_amazing : width * length = 3 * (2 * (width + length))

/-- Theorem stating that there exists only one amazing rectangle and its area is 162. -/
theorem unique_amazing_rectangle :
  (∃! r : AmazingRectangle, True) ∧
  (∀ r : AmazingRectangle, r.width * r.length = 162) := by
  sorry


end NUMINAMATH_CALUDE_unique_amazing_rectangle_l2640_264038


namespace NUMINAMATH_CALUDE_total_disks_l2640_264092

/-- Represents the number of disks of each color in the bag -/
structure DiskCount where
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- The properties of the disk distribution in the bag -/
def validDiskCount (d : DiskCount) : Prop :=
  ∃ (x : ℕ),
    d.blue = 3 * x ∧
    d.yellow = 7 * x ∧
    d.green = 8 * x ∧
    d.green = d.blue + 15

/-- The theorem stating the total number of disks in the bag -/
theorem total_disks (d : DiskCount) (h : validDiskCount d) : 
  d.blue + d.yellow + d.green = 54 := by
  sorry


end NUMINAMATH_CALUDE_total_disks_l2640_264092


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l2640_264079

/-- The area of a triangle inscribed in a circle, where the triangle's vertices
    divide the circle into three arcs of lengths 4, 5, and 7. -/
theorem inscribed_triangle_area : ∃ (A : ℝ), 
  (∀ (r : ℝ), r > 0 → r = 8 / Real.pi → 
    ∃ (θ₁ θ₂ θ₃ : ℝ), 
      θ₁ > 0 ∧ θ₂ > 0 ∧ θ₃ > 0 ∧
      4 * r = 4 * θ₁ ∧
      5 * r = 5 * θ₂ ∧
      7 * r = 7 * θ₃ ∧
      θ₁ + θ₂ + θ₃ = 2 * Real.pi ∧
      A = (1/2) * r^2 * (Real.sin (2*θ₁) + Real.sin (2*(θ₁+θ₂)) + Real.sin (2*(θ₁+θ₂+θ₃)))) ∧
  A = 147.6144 / Real.pi^2 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l2640_264079


namespace NUMINAMATH_CALUDE_smallest_k_is_two_l2640_264073

/-- A five-digit number -/
def FiveDigitNumber := Fin 100000

/-- Predicate to check if a number has digits in non-decreasing order -/
def hasNonDecreasingDigits (n : FiveDigitNumber) : Prop := sorry

/-- Predicate to check if two numbers have at least one digit in common -/
def hasCommonDigit (n m : FiveDigitNumber) : Prop := sorry

/-- The set of five-digit numbers satisfying the problem conditions -/
def SpecialSet (k : ℕ) : Set FiveDigitNumber := sorry

theorem smallest_k_is_two :
  ∀ k : ℕ,
    (∀ n : FiveDigitNumber, hasNonDecreasingDigits n →
      ∃ m ∈ SpecialSet k, hasCommonDigit n m) →
    k ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_is_two_l2640_264073


namespace NUMINAMATH_CALUDE_problem_solution_l2640_264022

theorem problem_solution (n : ℝ) (h : n + 1/n = 10) : n^2 + 1/n^2 + 5 = 103 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2640_264022


namespace NUMINAMATH_CALUDE_spaceship_total_distance_l2640_264098

/-- The total distance traveled by a spaceship between Earth and various planets -/
theorem spaceship_total_distance (d_earth_x d_x_y d_y_z d_z_w d_w_earth : ℝ) 
  (h1 : d_earth_x = 3.37)
  (h2 : d_x_y = 1.57)
  (h3 : d_y_z = 2.19)
  (h4 : d_z_w = 4.27)
  (h5 : d_w_earth = 1.89) :
  d_earth_x + d_x_y + d_y_z + d_z_w + d_w_earth = 13.29 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_total_distance_l2640_264098


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l2640_264083

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Main theorem -/
theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 5 = 16)
  (h_fourth : a 4 = 8) :
  a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l2640_264083


namespace NUMINAMATH_CALUDE_new_person_weight_l2640_264051

/-- Given a group of 9 people where one person weighing 86 kg is replaced by a new person,
    and the average weight of the group increases by 5.5 kg,
    prove that the weight of the new person is 135.5 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 9 →
  weight_increase = 5.5 →
  replaced_weight = 86 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 135.5 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2640_264051


namespace NUMINAMATH_CALUDE_max_squared_ratio_is_four_thirds_l2640_264053

/-- The maximum squared ratio of a to b satisfying the given conditions -/
def max_squared_ratio (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≥ b ∧
  ∃ ρ : ℝ, ρ > 0 ∧
    ∀ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧
      a^2 + y^2 = b^2 + x^2 ∧
      b^2 + x^2 = (a - x)^2 + (b - y)^2 →
      (a / b)^2 ≤ ρ^2 ∧
    ρ^2 = 4/3

theorem max_squared_ratio_is_four_thirds (a b : ℝ) :
  max_squared_ratio a b :=
sorry

end NUMINAMATH_CALUDE_max_squared_ratio_is_four_thirds_l2640_264053


namespace NUMINAMATH_CALUDE_min_A_over_C_is_zero_l2640_264063

theorem min_A_over_C_is_zero (x : ℝ) (A C : ℝ) (h1 : x ≠ 0) (h2 : A > 0) (h3 : C > 0)
  (h4 : x^2 + 1/x^2 = A) (h5 : x + 1/x = C) :
  ∃ ε > 0, ∀ δ > 0, ∃ A' C', A' > 0 ∧ C' > 0 ∧ A'/C' < δ ∧
  ∃ x' : ℝ, x' ≠ 0 ∧ x'^2 + 1/x'^2 = A' ∧ x' + 1/x' = C' :=
sorry

end NUMINAMATH_CALUDE_min_A_over_C_is_zero_l2640_264063


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_negative_four_condition_for_b_subset_complement_a_l2640_264000

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 5}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

-- Theorem for the first part of the problem
theorem intersection_and_union_when_a_is_negative_four :
  (A ∩ B (-4)) = {x | 1/2 ≤ x ∧ x < 2} ∧
  (A ∪ B (-4)) = {x | -2 < x ∧ x ≤ 3} := by sorry

-- Theorem for the second part of the problem
theorem condition_for_b_subset_complement_a (a : ℝ) :
  (B a ∩ Aᶜ = B a) ↔ a ≥ -1/4 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_negative_four_condition_for_b_subset_complement_a_l2640_264000


namespace NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l2640_264010

theorem power_of_product_equals_product_of_powers (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l2640_264010


namespace NUMINAMATH_CALUDE_bahs_equal_to_yahs_l2640_264001

/-- Conversion rate from bahs to rahs -/
def bah_to_rah : ℚ := 16 / 10

/-- Conversion rate from rahs to yahs -/
def rah_to_yah : ℚ := 15 / 9

/-- The number of yahs we want to convert -/
def target_yahs : ℚ := 1500

theorem bahs_equal_to_yahs : 
  (target_yahs / (bah_to_rah * rah_to_yah) : ℚ) = 562.5 := by sorry

end NUMINAMATH_CALUDE_bahs_equal_to_yahs_l2640_264001


namespace NUMINAMATH_CALUDE_circle_tangency_l2640_264087

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the given line
def givenLine (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x

-- Define the y-axis
def yAxis (x : ℝ) : Prop := x = 0

-- Define the possible circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - Real.sqrt 3)^2 = 1
def circle2 (x y : ℝ) : Prop := (x + 1)^2 + (y + Real.sqrt 3)^2 = 1
def circle3 (x y : ℝ) : Prop := (x - 2*Real.sqrt 3 - 3)^2 + (y + 2 + Real.sqrt 3)^2 = 21 + 12*Real.sqrt 3
def circle4 (x y : ℝ) : Prop := (x + 2*Real.sqrt 3 + 3)^2 + (y - 2 - Real.sqrt 3)^2 = 21 + 12*Real.sqrt 3

-- Define external tangency
def externallyTangent (c1 c2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define tangency to a line
def tangentToLine (c : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop := sorry

-- Define tangency to y-axis
def tangentToYAxis (c : ℝ → ℝ → Prop) : Prop := sorry

theorem circle_tangency :
  (externallyTangent circle1 givenCircle ∧ tangentToLine circle1 givenLine ∧ tangentToYAxis circle1) ∨
  (externallyTangent circle2 givenCircle ∧ tangentToLine circle2 givenLine ∧ tangentToYAxis circle2) ∨
  (externallyTangent circle3 givenCircle ∧ tangentToLine circle3 givenLine ∧ tangentToYAxis circle3) ∨
  (externallyTangent circle4 givenCircle ∧ tangentToLine circle4 givenLine ∧ tangentToYAxis circle4) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangency_l2640_264087


namespace NUMINAMATH_CALUDE_inequality_holds_l2640_264091

/-- An equilateral triangle with height 1 -/
structure EquilateralTriangle :=
  (height : ℝ)
  (height_eq_one : height = 1)

/-- A point inside the equilateral triangle -/
structure PointInTriangle (t : EquilateralTriangle) :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (sum_eq_height : x + y + z = t.height)
  (all_positive : x > 0 ∧ y > 0 ∧ z > 0)

/-- The inequality holds for any point inside the equilateral triangle -/
theorem inequality_holds (t : EquilateralTriangle) (p : PointInTriangle t) :
  p.x^2 + p.y^2 + p.z^2 ≥ p.x^3 + p.y^3 + p.z^3 + 6*p.x*p.y*p.z :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_l2640_264091


namespace NUMINAMATH_CALUDE_trumpet_cost_l2640_264046

/-- The cost of Mike's trumpet, given the net amount spent and the amount received for selling a song book. -/
theorem trumpet_cost (net_spent : ℝ) (song_book_sold : ℝ) (h1 : net_spent = 139.32) (h2 : song_book_sold = 5.84) :
  net_spent + song_book_sold = 145.16 := by
  sorry

end NUMINAMATH_CALUDE_trumpet_cost_l2640_264046


namespace NUMINAMATH_CALUDE_rectangle_y_value_l2640_264064

/-- A rectangle with vertices (-2, y), (6, y), (-2, 2), and (6, 2) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := 8 * (r.y - 2)

/-- The perimeter of the rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (8 + (r.y - 2))

/-- Theorem stating that if the area is 64 and the perimeter is 32, then y = 10 -/
theorem rectangle_y_value (r : Rectangle) 
  (h_area : area r = 64) 
  (h_perimeter : perimeter r = 32) : 
  r.y = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l2640_264064


namespace NUMINAMATH_CALUDE_T_coprime_and_sum_reciprocals_l2640_264075

def T : ℕ → ℕ
  | 0 => 2
  | n + 1 => T n^2 - T n + 1

theorem T_coprime_and_sum_reciprocals :
  (∀ m n, m ≠ n → Nat.gcd (T m) (T n) = 1) ∧
  (∑' i, (T i)⁻¹ : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_T_coprime_and_sum_reciprocals_l2640_264075


namespace NUMINAMATH_CALUDE_symmetry_of_graphs_l2640_264009

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define a real number a
variable (a : ℝ)

-- Define symmetry about a vertical line
def symmetricAboutVerticalLine (g h : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y, g x = y ↔ h (2*c - x) = y

-- State the theorem
theorem symmetry_of_graphs :
  symmetricAboutVerticalLine (fun x ↦ f (a - x)) (fun x ↦ f (x - a)) a :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_graphs_l2640_264009


namespace NUMINAMATH_CALUDE_one_third_1206_percent_of_134_l2640_264037

theorem one_third_1206_percent_of_134 : 
  (1206 / 3) / 134 * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_one_third_1206_percent_of_134_l2640_264037


namespace NUMINAMATH_CALUDE_weight_increase_percentage_l2640_264081

/-- The percentage increase in total weight of two people given their initial weight ratio and individual weight increases -/
theorem weight_increase_percentage 
  (ram_ratio : ℝ) 
  (shyam_ratio : ℝ) 
  (ram_increase : ℝ) 
  (shyam_increase : ℝ) 
  (new_total_weight : ℝ) 
  (h1 : ram_ratio = 2) 
  (h2 : shyam_ratio = 5) 
  (h3 : ram_increase = 0.1) 
  (h4 : shyam_increase = 0.17) 
  (h5 : new_total_weight = 82.8) : 
  ∃ (percentage_increase : ℝ), 
    abs (percentage_increase - 15.06) < 0.01 ∧ 
    percentage_increase = 
      (new_total_weight - (ram_ratio + shyam_ratio) * 
        (new_total_weight / (ram_ratio * (1 + ram_increase) + shyam_ratio * (1 + shyam_increase)))) / 
      ((ram_ratio + shyam_ratio) * 
        (new_total_weight / (ram_ratio * (1 + ram_increase) + shyam_ratio * (1 + shyam_increase)))) 
      * 100 := by
  sorry


end NUMINAMATH_CALUDE_weight_increase_percentage_l2640_264081


namespace NUMINAMATH_CALUDE_no_triangle_condition_l2640_264025

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) : Prop := 4 * x + 3 * y + 5 = 0
def line3 (m x y : ℝ) : Prop := m * x - y - 1 = 0

-- Define when three lines form a triangle
def form_triangle (l1 l2 l3 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    l1 x1 y1 ∧ l2 x2 y2 ∧ l3 x3 y3 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧ (x2 ≠ x3 ∨ y2 ≠ y3) ∧ (x3 ≠ x1 ∨ y3 ≠ y1)

-- Theorem statement
theorem no_triangle_condition (m : ℝ) :
  ¬(form_triangle line1 line2 (line3 m)) ↔ m ∈ ({-4/3, 2/3, 4/3} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_no_triangle_condition_l2640_264025


namespace NUMINAMATH_CALUDE_mathematics_players_count_l2640_264036

-- Define the set of all players
def TotalPlayers : ℕ := 30

-- Define the set of players taking physics
def PhysicsPlayers : ℕ := 15

-- Define the set of players taking both physics and mathematics
def BothSubjectsPlayers : ℕ := 7

-- Define the set of players taking mathematics
def MathematicsPlayers : ℕ := TotalPlayers - (PhysicsPlayers - BothSubjectsPlayers)

-- Theorem statement
theorem mathematics_players_count : MathematicsPlayers = 22 := by
  sorry

end NUMINAMATH_CALUDE_mathematics_players_count_l2640_264036


namespace NUMINAMATH_CALUDE_real_estate_transaction_gain_l2640_264026

theorem real_estate_transaction_gain 
  (flat1_purchase : ℝ) (flat1_gain_percent : ℝ)
  (flat2_purchase : ℝ) (flat2_loss_percent : ℝ)
  (flat3_purchase : ℝ) (flat3_gain_percent : ℝ)
  (h1 : flat1_purchase = 675958)
  (h2 : flat1_gain_percent = 14)
  (h3 : flat2_purchase = 848592)
  (h4 : flat2_loss_percent = 10)
  (h5 : flat3_purchase = 940600)
  (h6 : flat3_gain_percent = 7) :
  let flat1_sell := flat1_purchase * (1 + flat1_gain_percent / 100)
  let flat2_sell := flat2_purchase * (1 - flat2_loss_percent / 100)
  let flat3_sell := flat3_purchase * (1 + flat3_gain_percent / 100)
  let total_purchase := flat1_purchase + flat2_purchase + flat3_purchase
  let total_sell := flat1_sell + flat2_sell + flat3_sell
  total_sell - total_purchase = 75617.92 := by sorry

end NUMINAMATH_CALUDE_real_estate_transaction_gain_l2640_264026


namespace NUMINAMATH_CALUDE_fraction_comparison_l2640_264041

theorem fraction_comparison : (1 / (Real.sqrt 5 - 2)) < (1 / (Real.sqrt 6 - Real.sqrt 5)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2640_264041


namespace NUMINAMATH_CALUDE_sqrt_x_minus_two_defined_l2640_264006

theorem sqrt_x_minus_two_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_two_defined_l2640_264006


namespace NUMINAMATH_CALUDE_product_5832_sum_62_l2640_264085

theorem product_5832_sum_62 : ∃ (a b c : ℕ+),
  (a.val > 1) ∧ (b.val > 1) ∧ (c.val > 1) ∧
  (a * b * c = 5832) ∧
  (Nat.gcd a.val b.val = 1) ∧ (Nat.gcd b.val c.val = 1) ∧ (Nat.gcd c.val a.val = 1) ∧
  (a + b + c = 62) := by
sorry

end NUMINAMATH_CALUDE_product_5832_sum_62_l2640_264085


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l2640_264043

theorem sum_remainder_mod_seven : (5283 + 5284 + 5285 + 5286 + 5287) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l2640_264043


namespace NUMINAMATH_CALUDE_lucy_fish_purchase_l2640_264028

theorem lucy_fish_purchase (current : ℕ) (desired : ℕ) (h1 : current = 212) (h2 : desired = 280) :
  desired - current = 68 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_purchase_l2640_264028


namespace NUMINAMATH_CALUDE_min_red_to_blue_l2640_264040

/-- Represents the colors of chameleons -/
inductive Color
| Red
| Blue
| Other1
| Other2
| Other3

/-- Represents the color-changing rule when one chameleon bites another -/
def bite_rule : Color → Color → Color := sorry

/-- Represents a sequence of bites -/
def BiteSequence := List (Nat × Nat)

/-- Represents the state of chameleons after a sequence of bites -/
def apply_bites (initial : List Color) (sequence : BiteSequence) : List Color := sorry

/-- Checks if all chameleons in a list are blue -/
def all_blue (chameleons : List Color) : Prop :=
  ∀ c ∈ chameleons, c = Color.Blue

/-- The main theorem stating that 5 is the minimum number of red chameleons
    required to guarantee they can all become blue -/
theorem min_red_to_blue :
  (∃ n : Nat, ∃ seq : BiteSequence,
    all_blue (apply_bites (List.replicate n Color.Red) seq)) ∧
  (∀ k < 5, ∃ seq : BiteSequence,
    ¬all_blue (apply_bites (List.replicate k Color.Red) seq)) ∧
  (∃ seq : BiteSequence,
    all_blue (apply_bites (List.replicate 5 Color.Red) seq)) :=
sorry

end NUMINAMATH_CALUDE_min_red_to_blue_l2640_264040


namespace NUMINAMATH_CALUDE_multiples_6_10_not_5_8_empty_l2640_264088

theorem multiples_6_10_not_5_8_empty : 
  {n : ℤ | 1 ≤ n ∧ n ≤ 300 ∧ 6 ∣ n ∧ 10 ∣ n ∧ ¬(5 ∣ n) ∧ ¬(8 ∣ n)} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_multiples_6_10_not_5_8_empty_l2640_264088


namespace NUMINAMATH_CALUDE_tangent_to_ln_curve_l2640_264021

theorem tangent_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ (∀ y : ℝ, y > 0 → k * y ≤ Real.log y)) → 
  k = 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_to_ln_curve_l2640_264021


namespace NUMINAMATH_CALUDE_interest_rate_proof_l2640_264045

/-- Proves that for given conditions, the annual interest rate is 10% -/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (diff : ℝ) : 
  principal = 1700 → 
  time = 1 → 
  diff = 4.25 → 
  ∃ (rate : ℝ), 
    rate = 10 ∧ 
    principal * ((1 + rate / 200)^2 - 1) - principal * rate * time / 100 = diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l2640_264045


namespace NUMINAMATH_CALUDE_solve_equation_l2640_264057

theorem solve_equation (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2640_264057


namespace NUMINAMATH_CALUDE_school_students_count_l2640_264005

theorem school_students_count :
  ∀ (total_students boys girls : ℕ),
  total_students = boys + girls →
  boys = 80 →
  girls = (80 * total_students) / 100 →
  total_students = 400 := by
sorry

end NUMINAMATH_CALUDE_school_students_count_l2640_264005


namespace NUMINAMATH_CALUDE_rectangle_area_l2640_264072

/-- The area of a rectangle with sides 5.9 cm and 3 cm is 17.7 square centimeters. -/
theorem rectangle_area : 
  let side1 : ℝ := 5.9
  let side2 : ℝ := 3
  side1 * side2 = 17.7 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2640_264072


namespace NUMINAMATH_CALUDE_cassidy_grounding_l2640_264013

/-- The number of days Cassidy is grounded for lying about her report card -/
def base_grounding : ℕ := 14

/-- The number of extra days Cassidy is grounded for each grade below a B -/
def extra_days_per_grade : ℕ := 3

/-- The number of grades Cassidy got below a B -/
def grades_below_b : ℕ := 4

/-- The total number of days Cassidy is grounded -/
def total_grounding : ℕ := base_grounding + extra_days_per_grade * grades_below_b

theorem cassidy_grounding :
  total_grounding = 26 := by
  sorry

end NUMINAMATH_CALUDE_cassidy_grounding_l2640_264013


namespace NUMINAMATH_CALUDE_binomial_10_2_l2640_264048

theorem binomial_10_2 : (Nat.choose 10 2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_2_l2640_264048


namespace NUMINAMATH_CALUDE_erica_money_proof_l2640_264055

def total_money : ℕ := 91
def sam_money : ℕ := 38

theorem erica_money_proof :
  total_money - sam_money = 53 := by
  sorry

end NUMINAMATH_CALUDE_erica_money_proof_l2640_264055


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2640_264061

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2640_264061


namespace NUMINAMATH_CALUDE_ellipse_foci_l2640_264094

/-- The equation of an ellipse in standard form -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 25 = 1

/-- The coordinates of a focus of the ellipse -/
def is_focus (x y : ℝ) : Prop := x = 0 ∧ (y = 3 ∨ y = -3)

/-- Theorem: The foci of the given ellipse are at (0, ±3) -/
theorem ellipse_foci :
  ∀ x y : ℝ, is_ellipse x y → ∃ fx fy : ℝ, is_focus fx fy :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l2640_264094


namespace NUMINAMATH_CALUDE_binomial_8_4_l2640_264047

theorem binomial_8_4 : (8 : ℕ).choose 4 = 70 := by sorry

end NUMINAMATH_CALUDE_binomial_8_4_l2640_264047


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2640_264090

theorem complex_magnitude_problem (s : ℝ) (w : ℂ) 
  (h1 : |s| < 3) 
  (h2 : w + 2 / w = s) : 
  Complex.abs w = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2640_264090


namespace NUMINAMATH_CALUDE_axis_of_symmetry_parabola_l2640_264019

/-- The axis of symmetry for the parabola y² = -8x is the line x = 2 -/
theorem axis_of_symmetry_parabola (x y : ℝ) : 
  y^2 = -8*x → (x = 2 ↔ ∀ y', y'^2 = -8*x → y'^2 = y^2) :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_parabola_l2640_264019


namespace NUMINAMATH_CALUDE_bus_passengers_l2640_264024

theorem bus_passengers (initial_students : ℕ) (num_stops : ℕ) : 
  initial_students = 60 → num_stops = 4 → 
  ⌊(initial_students : ℚ) * (2/3)^num_stops⌋ = 11 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l2640_264024


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l2640_264065

theorem sum_of_three_squares (K : ℕ) (L : ℤ) (h : L % 8 = 7) :
  ¬ ∃ (a b c : ℤ), 4^K * L = a^2 + b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l2640_264065


namespace NUMINAMATH_CALUDE_parallelepiped_sphere_properties_l2640_264059

/-- Represents a parallelepiped ABCDA₁B₁C₁D₁ with a sphere Ω touching its edges -/
structure Parallelepiped where
  -- Edge length of A₁A
  edge_length : ℝ
  -- Volume of the parallelepiped
  volume : ℝ
  -- Radius of the sphere Ω
  sphere_radius : ℝ
  -- A₁A is perpendicular to ABCD
  edge_perpendicular : edge_length > 0
  -- Sphere Ω touches BB₁, B₁C₁, C₁C, CB, C₁D₁, and AD
  sphere_touches_edges : True
  -- Ω touches C₁D₁ at K where C₁K = 9 and KD₁ = 4
  sphere_touch_point : edge_length > 13

/-- The theorem stating the properties of the parallelepiped and sphere -/
theorem parallelepiped_sphere_properties : 
  ∃ (p : Parallelepiped), 
    p.edge_length = 18 ∧ 
    p.volume = 3888 ∧ 
    p.sphere_radius = 3 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_sphere_properties_l2640_264059


namespace NUMINAMATH_CALUDE_reverse_square_digits_l2640_264002

theorem reverse_square_digits : ∃! n : ℕ, n > 0 ∧
  (n^2 % 100 = 10 * ((n+1)^2 % 10) + ((n+1)^2 / 10 % 10)) ∧
  ((n+1)^2 % 100 = 10 * (n^2 % 10) + (n^2 / 10 % 10)) :=
sorry

end NUMINAMATH_CALUDE_reverse_square_digits_l2640_264002


namespace NUMINAMATH_CALUDE_race_probability_l2640_264042

structure Race where
  total_cars : ℕ
  prob_x : ℚ
  prob_y : ℚ
  prob_z : ℚ
  no_dead_heat : Bool

def Race.prob_one_wins (r : Race) : ℚ :=
  r.prob_x + r.prob_y + r.prob_z

theorem race_probability (r : Race) 
  (h1 : r.total_cars = 10)
  (h2 : r.prob_x = 1 / 7)
  (h3 : r.prob_y = 1 / 3)
  (h4 : r.prob_z = 1 / 5)
  (h5 : r.no_dead_heat = true) :
  r.prob_one_wins = 71 / 105 := by
  sorry

end NUMINAMATH_CALUDE_race_probability_l2640_264042


namespace NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l2640_264080

-- Define the set M
def M : Set ℝ := {x : ℝ | |2*x - 1| < 1}

-- Theorem 1: Characterization of set M
theorem characterization_of_M : M = {x : ℝ | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_for_M_elements (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  a * b + 1 > a + b := by sorry

end NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l2640_264080


namespace NUMINAMATH_CALUDE_shelter_adoption_percentage_l2640_264034

def initial_dogs : ℕ := 80
def returned_dogs : ℕ := 5
def final_dogs : ℕ := 53

def adoption_percentage : ℚ := 40

theorem shelter_adoption_percentage :
  (initial_dogs - (initial_dogs * adoption_percentage / 100) + returned_dogs : ℚ) = final_dogs :=
sorry

end NUMINAMATH_CALUDE_shelter_adoption_percentage_l2640_264034


namespace NUMINAMATH_CALUDE_coffee_stock_solution_l2640_264096

/-- Represents the coffee stock problem -/
def coffee_stock_problem (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
  (second_batch_decaf_percent : ℝ) (final_decaf_percent : ℝ) : Prop :=
  ∃ (second_batch : ℝ),
    second_batch > 0 ∧
    (initial_stock * initial_decaf_percent + second_batch * second_batch_decaf_percent) / 
    (initial_stock + second_batch) = final_decaf_percent

/-- The solution to the coffee stock problem -/
theorem coffee_stock_solution :
  coffee_stock_problem 400 0.30 0.60 0.36 → 
  ∃ (second_batch : ℝ), second_batch = 100 := by
  sorry

end NUMINAMATH_CALUDE_coffee_stock_solution_l2640_264096
