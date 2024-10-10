import Mathlib

namespace exists_special_subset_l2200_220098

/-- Given a set of 40 elements and a function that maps each 19-element subset to a unique element (common friend), 
    there exists a 20-element subset M₀ such that for all a ∈ M₀, the common friend of M₀ \ {a} is not a. -/
theorem exists_special_subset (I : Finset Nat) (f : Finset Nat → Nat) : 
  I.card = 40 → 
  (∀ A : Finset Nat, A ⊆ I → A.card = 19 → f A ∈ I) →
  (∀ A : Finset Nat, A ⊆ I → A.card = 19 → f A ∉ A) →
  ∃ M₀ : Finset Nat, M₀ ⊆ I ∧ M₀.card = 20 ∧ 
    ∀ a ∈ M₀, f (M₀ \ {a}) ≠ a := by
  sorry

end exists_special_subset_l2200_220098


namespace roundness_of_1728_l2200_220055

/-- Roundness of a number is defined as the sum of the exponents in its prime factorization -/
def roundness (n : Nat) : Nat :=
  sorry

/-- 1728 can be expressed as 2^6 * 3^3 -/
axiom factorization_1728 : 1728 = 2^6 * 3^3

theorem roundness_of_1728 : roundness 1728 = 9 := by
  sorry

end roundness_of_1728_l2200_220055


namespace cone_volume_l2200_220006

/-- Given a cone with lateral area 20π and angle between slant height and base arccos(4/5),
    prove that its volume is 16π. -/
theorem cone_volume (r l h : ℝ) (lateral_area : ℝ) (angle : ℝ) : 
  lateral_area = 20 * Real.pi →
  angle = Real.arccos (4/5) →
  r / l = 4 / 5 →
  lateral_area = Real.pi * r * l →
  h = Real.sqrt (l^2 - r^2) →
  (1/3) * Real.pi * r^2 * h = 16 * Real.pi :=
by sorry

end cone_volume_l2200_220006


namespace stratified_sample_second_year_l2200_220072

theorem stratified_sample_second_year (total_students : ℕ) (second_year_students : ℕ) (sample_size : ℕ) : 
  total_students = 1000 →
  second_year_students = 320 →
  sample_size = 200 →
  (second_year_students * sample_size) / total_students = 64 :=
by sorry

end stratified_sample_second_year_l2200_220072


namespace total_cost_calculation_l2200_220002

theorem total_cost_calculation : 
  let sandwich_price : ℚ := 349/100
  let soda_price : ℚ := 87/100
  let sandwich_quantity : ℕ := 2
  let soda_quantity : ℕ := 4
  let total_cost : ℚ := sandwich_price * sandwich_quantity + soda_price * soda_quantity
  total_cost = 1046/100 := by
sorry

end total_cost_calculation_l2200_220002


namespace complex_modulus_problem_l2200_220050

theorem complex_modulus_problem (a : ℝ) (z : ℂ) : 
  z = (1 - a * Complex.I) * (1 + Complex.I) →
  z.im = -3 →
  Complex.abs z = Real.sqrt 34 := by
sorry

end complex_modulus_problem_l2200_220050


namespace triangle_probability_theorem_l2200_220017

/-- The number of points in the plane -/
def num_points : ℕ := 10

/-- The total number of possible segments -/
def total_segments : ℕ := (num_points * (num_points - 1)) / 2

/-- The number of segments chosen -/
def chosen_segments : ℕ := 4

/-- The probability of choosing 4 segments that form a triangle -/
def triangle_probability : ℚ := 1680 / 49665

theorem triangle_probability_theorem :
  triangle_probability = (num_points.choose 3 * (total_segments - 3)) / total_segments.choose chosen_segments :=
by sorry

end triangle_probability_theorem_l2200_220017


namespace f_range_l2200_220064

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -2 ≤ y ∧ y ≤ 12 := by
  sorry

end f_range_l2200_220064


namespace number_circle_exists_l2200_220003

/-- A type representing a three-digit number with no zero digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_nonzero : hundreds ≠ 0
  tens_nonzero : tens ≠ 0
  ones_nonzero : ones ≠ 0
  hundreds_lt_ten : hundreds < 10
  tens_lt_ten : tens < 10
  ones_lt_ten : ones < 10

/-- A type representing a circle of six three-digit numbers -/
structure NumberCircle where
  numbers : Fin 6 → ThreeDigitNumber
  all_different : ∀ i j, i ≠ j → numbers i ≠ numbers j
  circular_property : ∀ i, 
    (numbers i).tens = (numbers ((i + 1) % 6)).hundreds ∧
    (numbers i).ones = (numbers ((i + 1) % 6)).tens

/-- Function to check if a number is divisible by n -/
def isDivisibleBy (num : ThreeDigitNumber) (n : Nat) : Prop :=
  (100 * num.hundreds + 10 * num.tens + num.ones) % n = 0

/-- The main theorem -/
theorem number_circle_exists (n : Nat) : 
  (∃ circle : NumberCircle, ∀ i, isDivisibleBy (circle.numbers i) n) ↔ n = 3 ∨ n = 7 :=
sorry

end number_circle_exists_l2200_220003


namespace min_value_of_f_l2200_220075

-- Define the expression as a function of x
def f (x : ℝ) : ℝ := (15 - x) * (8 - x) * (15 + x) * (8 + x) + 200

-- State the theorem
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ -6290.25 ∧ ∃ y : ℝ, f y = -6290.25 := by sorry

end min_value_of_f_l2200_220075


namespace sum_of_coefficients_abs_l2200_220058

theorem sum_of_coefficients_abs (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) :
  (∀ x, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
  sorry

end sum_of_coefficients_abs_l2200_220058


namespace probability_specific_case_l2200_220039

/-- The probability of drawing one green, one white, and one blue ball simultaneously -/
def probability_three_colors (green white blue : ℕ) : ℚ :=
  let total := green + white + blue
  let favorable := green * white * blue
  let total_combinations := (total * (total - 1) * (total - 2)) / 6
  (favorable : ℚ) / total_combinations

/-- Theorem stating the probability of drawing one green, one white, and one blue ball -/
theorem probability_specific_case : 
  probability_three_colors 12 10 8 = 24 / 101 := by
  sorry


end probability_specific_case_l2200_220039


namespace min_value_reciprocal_sum_l2200_220054

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  1 / a + 1 / b ≥ 4 := by
  sorry

end min_value_reciprocal_sum_l2200_220054


namespace chord_equation_l2200_220027

def Circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9}

def P : ℝ × ℝ := (1, 1)

def is_midpoint (m : ℝ × ℝ) (p : ℝ × ℝ) (n : ℝ × ℝ) : Prop :=
  p.1 = (m.1 + n.1) / 2 ∧ p.2 = (m.2 + n.2) / 2

theorem chord_equation (M N : ℝ × ℝ) (h1 : M ∈ Circle) (h2 : N ∈ Circle)
  (h3 : is_midpoint M P N) :
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧
                 ∀ (x y : ℝ), (x, y) ∈ Circle → 
                 ((x, y) = M ∨ (x, y) = N) → 
                 a * x + b * y + c = 0 ∧
                 (a, b, c) = (2, -1, -1) := by
  sorry

end chord_equation_l2200_220027


namespace ellipse_line_intersection_range_l2200_220077

-- Define the ellipse equation
def ellipse (x y m : ℝ) : Prop := x^2/3 + y^2/m = 1

-- Define the line equation
def line (x y : ℝ) : Prop := x + 2*y - 2 = 0

-- Define the intersection condition
def intersect_at_two_points (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ ellipse x₁ y₁ m ∧ ellipse x₂ y₂ m ∧ 
                  line x₁ y₁ ∧ line x₂ y₂

-- Theorem statement
theorem ellipse_line_intersection_range :
  ∀ m : ℝ, intersect_at_two_points m ↔ (1/4 < m ∧ m < 3) ∨ m > 3 :=
sorry

end ellipse_line_intersection_range_l2200_220077


namespace product_equals_two_thirds_l2200_220016

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => 1 + (a n - 1)^2

-- Define the infinite product of a_n
def infiniteProduct : ℚ := sorry

-- Theorem statement
theorem product_equals_two_thirds : infiniteProduct = 2/3 := by sorry

end product_equals_two_thirds_l2200_220016


namespace prob_more_ones_than_sixes_l2200_220026

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The total number of possible outcomes when rolling numDice dice -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The number of outcomes where the same number of 1's and 6's are rolled -/
def sameOnesSixes : ℕ := 2424

/-- The probability of rolling more 1's than 6's when rolling numDice fair numSides-sided dice -/
def probMoreOnesThanSixes : ℚ := 2676 / 7776

theorem prob_more_ones_than_sixes :
  probMoreOnesThanSixes = 1 / 2 * (1 - sameOnesSixes / totalOutcomes) :=
sorry

end prob_more_ones_than_sixes_l2200_220026


namespace bakery_storage_theorem_l2200_220036

def bakery_storage_problem (sugar : ℕ) (flour : ℕ) (baking_soda : ℕ) (added_baking_soda : ℕ) : Prop :=
  sugar = 2400 ∧
  sugar = flour ∧
  10 * baking_soda = flour ∧
  added_baking_soda = 60 ∧
  8 * (baking_soda + added_baking_soda) = flour

theorem bakery_storage_theorem :
  ∃ (sugar flour baking_soda added_baking_soda : ℕ),
    bakery_storage_problem sugar flour baking_soda added_baking_soda :=
by
  sorry

end bakery_storage_theorem_l2200_220036


namespace boys_girls_relation_l2200_220009

/-- Represents the number of girls a boy dances with based on his position -/
def girls_danced_with (n : ℕ) : ℕ := 2 * n + 1

/-- 
Theorem: In a class where boys dance with girls following a specific pattern,
the number of boys is related to the number of girls by b = (g - 1) / 2.
-/
theorem boys_girls_relation (b g : ℕ) (h1 : b > 0) (h2 : g > 0) 
  (h3 : ∀ n, n ∈ Finset.range b → girls_danced_with n ≤ g) 
  (h4 : girls_danced_with b = g) : 
  b = (g - 1) / 2 := by
  sorry


end boys_girls_relation_l2200_220009


namespace tangent_line_to_circle_l2200_220049

theorem tangent_line_to_circle (x y : ℝ) : 
  -- The line is perpendicular to y = x + 1
  (∀ x₁ y₁ x₂ y₂ : ℝ, y₁ = x₁ + 1 → y₂ = x₂ + 1 → (y₂ - y₁) * (x + y - Real.sqrt 2 - y₁) = -(x₂ - x₁)) →
  -- The line is tangent to the circle x^2 + y^2 = 1
  ((x^2 + y^2 = 1 ∧ x + y - Real.sqrt 2 = 0) → 
    ∀ a b : ℝ, a^2 + b^2 = 1 → (a + b - Real.sqrt 2) * (a + b - Real.sqrt 2) ≥ 0) →
  -- The tangent point is in the first quadrant
  (x > 0 ∧ y > 0) →
  -- The equation of the line is x + y - √2 = 0
  x + y - Real.sqrt 2 = 0 := by
  sorry

end tangent_line_to_circle_l2200_220049


namespace chess_players_lost_to_ai_castor_island_ai_losses_l2200_220078

/-- The number of chess players who have lost to a computer at least once on Castor island -/
theorem chess_players_lost_to_ai (total_players : ℝ) (never_lost_fraction : ℚ) : ℝ :=
  let never_lost := total_players * (never_lost_fraction : ℝ)
  let lost_to_ai := total_players - never_lost
  ⌊lost_to_ai + 0.5⌋

/-- Given the conditions on Castor island, prove that approximately 48 players have lost to a computer -/
theorem castor_island_ai_losses : 
  ⌊chess_players_lost_to_ai 157.83 (37/53) + 0.5⌋ = 48 := by
sorry

end chess_players_lost_to_ai_castor_island_ai_losses_l2200_220078


namespace salt_solution_volume_l2200_220022

/-- Given a salt solution where 25 cubic centimeters contain 0.375 grams of salt,
    the volume of solution containing 15 grams of salt is 1000 cubic centimeters. -/
theorem salt_solution_volume (volume : ℝ) (salt_mass : ℝ) 
    (h1 : volume > 0)
    (h2 : salt_mass > 0)
    (h3 : 25 / volume = 0.375 / salt_mass) : 
  volume * (15 / salt_mass) = 1000 := by
sorry

end salt_solution_volume_l2200_220022


namespace unique_number_with_digit_product_l2200_220083

/-- Given a natural number n, multiply_digits n returns the product of n and all its digits. -/
def multiply_digits (n : ℕ) : ℕ := sorry

/-- Given a natural number n, digits n returns the list of digits of n. -/
def digits (n : ℕ) : List ℕ := sorry

theorem unique_number_with_digit_product : ∃! n : ℕ, multiply_digits n = 1995 ∧ digits n = [5, 7] := by sorry

end unique_number_with_digit_product_l2200_220083


namespace matrix_N_satisfies_conditions_l2200_220033

open Matrix

theorem matrix_N_satisfies_conditions :
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![1, -2, 0; 4, 6, 1; -3, 5, 2]
  let i : Matrix (Fin 3) (Fin 1) ℝ := !![1; 0; 0]
  let j : Matrix (Fin 3) (Fin 1) ℝ := !![0; 1; 0]
  let k : Matrix (Fin 3) (Fin 1) ℝ := !![0; 0; 1]
  N * i = !![1; 4; -3] ∧
  N * j = !![-2; 6; 5] ∧
  N * k = !![0; 1; 2] ∧
  det N ≠ 0 :=
by sorry

end matrix_N_satisfies_conditions_l2200_220033


namespace john_toy_store_spending_l2200_220047

def weekly_allowance : ℚ := 9/4  -- $2.25 as a rational number

def arcade_fraction : ℚ := 3/5

def candy_store_spending : ℚ := 3/5  -- $0.60 as a rational number

theorem john_toy_store_spending :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction)
  let toy_store_spending := remaining_after_arcade - candy_store_spending
  toy_store_spending / remaining_after_arcade = 1/3 := by sorry

end john_toy_store_spending_l2200_220047


namespace unique_solution_l2200_220053

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem unique_solution :
  ∀ x y : ℕ+, (3 ^ x.val + x.val ^ 4 = factorial y.val + 2019) ↔ (x = 6 ∧ y = 3) := by
  sorry

end unique_solution_l2200_220053


namespace common_factor_of_polynomial_l2200_220069

/-- The common factor of a polynomial 4x(m-n) + 2y(m-n)^2 is 2(m-n) -/
theorem common_factor_of_polynomial (x y m n : ℤ) :
  ∃ (k : ℤ), (4*x*(m-n) + 2*y*(m-n)^2) = 2*(m-n) * k :=
by sorry

end common_factor_of_polynomial_l2200_220069


namespace max_parts_formula_max_parts_special_cases_l2200_220059

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ := (n^2 + n + 2) / 2

/-- Theorem: The maximum number of parts a plane can be divided into by n lines is (n^2 + n + 2) / 2 -/
theorem max_parts_formula (n : ℕ) : max_parts n = (n^2 + n + 2) / 2 := by
  sorry

/-- Corollary: Special cases for n = 1, 2, 3, and 4 -/
theorem max_parts_special_cases :
  (max_parts 1 = 2) ∧
  (max_parts 2 = 4) ∧
  (max_parts 3 = 7) ∧
  (max_parts 4 = 11) := by
  sorry

end max_parts_formula_max_parts_special_cases_l2200_220059


namespace expected_value_is_negative_one_fifth_l2200_220063

/-- A die with two faces: Star and Moon -/
inductive DieFace
| star
| moon

/-- The probability of getting a Star face -/
def probStar : ℚ := 2/5

/-- The probability of getting a Moon face -/
def probMoon : ℚ := 3/5

/-- The winnings for Star face -/
def winStar : ℚ := 4

/-- The losses for Moon face -/
def lossMoon : ℚ := -3

/-- The expected value of one roll of the die -/
def expectedValue : ℚ := probStar * winStar + probMoon * lossMoon

theorem expected_value_is_negative_one_fifth :
  expectedValue = -1/5 := by sorry

end expected_value_is_negative_one_fifth_l2200_220063


namespace reduced_rate_fraction_l2200_220070

/-- Represents the fraction of a day with reduced rates -/
def weekdayReducedRateFraction : ℚ := 12 / 24

/-- Represents the fraction of a day with reduced rates on weekends -/
def weekendReducedRateFraction : ℚ := 1

/-- Represents the number of weekdays in a week -/
def weekdaysPerWeek : ℕ := 5

/-- Represents the number of weekend days in a week -/
def weekendDaysPerWeek : ℕ := 2

/-- Represents the total number of days in a week -/
def daysPerWeek : ℕ := 7

/-- Theorem stating that the fraction of a week with reduced rates is 9/14 -/
theorem reduced_rate_fraction :
  (weekdayReducedRateFraction * weekdaysPerWeek + weekendReducedRateFraction * weekendDaysPerWeek) / daysPerWeek = 9 / 14 := by
  sorry


end reduced_rate_fraction_l2200_220070


namespace solve_exponential_equation_l2200_220025

theorem solve_exponential_equation :
  ∃ x : ℝ, (1000 : ℝ)^2 = 10^x ∧ x = 6 := by sorry

end solve_exponential_equation_l2200_220025


namespace mean_weight_of_participants_l2200_220015

/-- Represents a stem and leaf plot entry -/
structure StemLeafEntry :=
  (stem : ℕ)
  (leaves : List ℕ)

/-- Calculates the sum of weights from a stem and leaf entry -/
def sumWeights (entry : StemLeafEntry) : ℕ :=
  entry.leaves.sum + entry.stem * 100 * entry.leaves.length

/-- Calculates the number of participants from a stem and leaf entry -/
def countParticipants (entry : StemLeafEntry) : ℕ :=
  entry.leaves.length

theorem mean_weight_of_participants (data : List StemLeafEntry) 
  (h1 : data = [
    ⟨12, [3, 5]⟩, 
    ⟨13, [0, 2, 3, 5, 7, 8]⟩, 
    ⟨14, [1, 5, 5, 9, 9]⟩, 
    ⟨15, [0, 2, 3, 5, 8]⟩, 
    ⟨16, [4, 7, 7, 9]⟩
  ]) : 
  (data.map sumWeights).sum / (data.map countParticipants).sum = 3217 / 22 := by
  sorry

end mean_weight_of_participants_l2200_220015


namespace divisors_of_n_squared_less_than_n_not_dividing_n_l2200_220086

def n : ℕ := 2^31 * 3^19 * 5^7

-- Function to count divisors of a number given its prime factorization
def count_divisors (factorization : List (ℕ × ℕ)) : ℕ :=
  factorization.foldl (λ acc (_, exp) => acc * (exp + 1)) 1

-- Function to count divisors less than n
def count_divisors_less_than_n (total_divisors : ℕ) : ℕ :=
  (total_divisors - 1) / 2

theorem divisors_of_n_squared_less_than_n_not_dividing_n :
  let n_squared_factorization : List (ℕ × ℕ) := [(2, 62), (3, 38), (5, 14)]
  let n_factorization : List (ℕ × ℕ) := [(2, 31), (3, 19), (5, 7)]
  let total_divisors_n_squared := count_divisors n_squared_factorization
  let divisors_less_than_n := count_divisors_less_than_n total_divisors_n_squared
  let divisors_of_n := count_divisors n_factorization
  divisors_less_than_n - divisors_of_n = 13307 :=
by sorry

end divisors_of_n_squared_less_than_n_not_dividing_n_l2200_220086


namespace units_digit_difference_l2200_220088

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_difference (p : ℕ) 
  (h1 : p % 2 = 0) 
  (h2 : units_digit p > 0) 
  (h3 : units_digit (p + 2) = 8) : 
  units_digit (p^3) - units_digit (p^2) = 0 := by
sorry

end units_digit_difference_l2200_220088


namespace max_value_of_expression_l2200_220089

theorem max_value_of_expression (x : ℝ) :
  (x^6) / (x^10 + 3*x^8 - 6*x^6 + 12*x^4 + 32) ≤ 1/18 ∧
  (2^6) / (2^10 + 3*2^8 - 6*2^6 + 12*2^4 + 32) = 1/18 := by
  sorry

end max_value_of_expression_l2200_220089


namespace original_price_correct_l2200_220000

/-- The original price of an article before discounts -/
def original_price : ℝ := 81.30

/-- The final sale price after all discounts -/
def final_price : ℝ := 36

/-- The list of discount rates -/
def discount_rates : List ℝ := [0.15, 0.25, 0.20, 0.18]

/-- Calculate the price after applying all discounts -/
def price_after_discounts (price : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (fun acc rate => acc * (1 - rate)) price

theorem original_price_correct : 
  ∃ ε > 0, abs (price_after_discounts original_price discount_rates - final_price) < ε :=
by
  sorry

#eval price_after_discounts original_price discount_rates

end original_price_correct_l2200_220000


namespace tax_discount_commute_petes_equals_pollys_l2200_220042

/-- Proves that the order of applying tax and discount doesn't affect the final price -/
theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h1 : 0 ≤ tax_rate) (h2 : 0 ≤ discount_rate) (h3 : discount_rate ≤ 1) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

/-- Calculates Pete's method: tax then discount -/
def petes_method (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate)

/-- Calculates Polly's method: discount then tax -/
def pollys_method (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

/-- Proves that Pete's and Polly's methods yield the same result -/
theorem petes_equals_pollys (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h1 : 0 ≤ tax_rate) (h2 : 0 ≤ discount_rate) (h3 : discount_rate ≤ 1) :
  petes_method price tax_rate discount_rate = pollys_method price tax_rate discount_rate :=
by sorry

end tax_discount_commute_petes_equals_pollys_l2200_220042


namespace matrix_inverse_proof_l2200_220043

def A : Matrix (Fin 2) (Fin 2) ℚ := !![5, 4; -2, 8]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1/6, -1/12; 1/24, 5/48]

theorem matrix_inverse_proof :
  A * A_inv = 1 ∧ A_inv * A = 1 :=
by sorry

end matrix_inverse_proof_l2200_220043


namespace complex_multiplication_complex_division_l2200_220012

-- Define complex numbers
def i : ℂ := Complex.I

-- Part 1
theorem complex_multiplication :
  (1 - 2*i) * (3 + 4*i) * (-2 + i) = -20 + 15*i := by sorry

-- Part 2
theorem complex_division (x a : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : x > a) :
  (Complex.ofReal x) / (Complex.ofReal (x - a)) = -1/5 + 2/5*i := by sorry

end complex_multiplication_complex_division_l2200_220012


namespace sequence_formula_l2200_220068

theorem sequence_formula (a : ℕ → ℚ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → 3 * a (n + 1) + 2 * a (n + 1) * a n - a n = 0) →
  ∀ n : ℕ, n ≥ 1 → a n = 1 / (2 * 3^(n - 1) - 1) :=
by sorry

end sequence_formula_l2200_220068


namespace tamikas_speed_l2200_220065

/-- Tamika's driving problem -/
theorem tamikas_speed (tamika_time logan_time logan_speed extra_distance : ℝ) 
  (h1 : tamika_time = 8)
  (h2 : logan_time = 5)
  (h3 : logan_speed = 55)
  (h4 : extra_distance = 85)
  : (logan_time * logan_speed + extra_distance) / tamika_time = 45 := by
  sorry

#check tamikas_speed

end tamikas_speed_l2200_220065


namespace mary_has_fifty_cards_l2200_220044

/-- The number of Pokemon cards Mary has after receiving new cards from Sam -/
def marys_final_cards (initial_cards torn_cards new_cards : ℕ) : ℕ :=
  initial_cards - torn_cards + new_cards

/-- Theorem stating that Mary has 50 Pokemon cards after the given scenario -/
theorem mary_has_fifty_cards :
  marys_final_cards 33 6 23 = 50 := by
  sorry

end mary_has_fifty_cards_l2200_220044


namespace expansion_and_factorization_l2200_220030

theorem expansion_and_factorization :
  (∀ y : ℝ, (y - 1) * (y + 5) = y^2 + 4*y - 5) ∧
  (∀ x y : ℝ, -x^2 + 4*x*y - 4*y^2 = -(x - 2*y)^2) := by
  sorry

end expansion_and_factorization_l2200_220030


namespace different_color_probability_l2200_220092

def total_balls : ℕ := 5
def blue_balls : ℕ := 3
def yellow_balls : ℕ := 2

theorem different_color_probability :
  let total_outcomes := Nat.choose total_balls 2
  let favorable_outcomes := blue_balls * yellow_balls
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 5 := by
sorry

end different_color_probability_l2200_220092


namespace john_profit_l2200_220085

/-- Calculates the profit from buying and selling ducks -/
def duck_profit (num_ducks : ℕ) (cost_per_duck : ℚ) (weight_per_duck : ℚ) (selling_price_per_pound : ℚ) : ℚ :=
  let total_cost := num_ducks * cost_per_duck
  let total_weight := num_ducks * weight_per_duck
  let total_revenue := total_weight * selling_price_per_pound
  total_revenue - total_cost

/-- Proves that John's profit is $300 -/
theorem john_profit :
  duck_profit 30 10 4 5 = 300 := by
  sorry

end john_profit_l2200_220085


namespace cube_root_of_negative_eight_l2200_220061

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end cube_root_of_negative_eight_l2200_220061


namespace hot_chocolate_consumption_l2200_220097

/-- The number of cups of hot chocolate John drinks in 5 hours -/
def cups_in_five_hours : ℕ := 15

/-- The time interval between each cup of hot chocolate in minutes -/
def interval : ℕ := 20

/-- The total time in minutes -/
def total_time : ℕ := 5 * 60

theorem hot_chocolate_consumption :
  cups_in_five_hours = total_time / interval :=
by sorry

end hot_chocolate_consumption_l2200_220097


namespace evaluate_expression_l2200_220082

theorem evaluate_expression (a : ℝ) (h : a = 2) : (7 * a ^ 2 - 10 * a + 3) * (3 * a - 4) = 22 := by
  sorry

end evaluate_expression_l2200_220082


namespace angle_range_given_monotonic_function_l2200_220038

/-- Given two vectors a and b in a real inner product space, 
    with |a| = 2√2|b| ≠ 0 and f(x) = 2x³ + 3|a|x² + 6(a · b)x + 7 
    monotonically increasing on ℝ, prove that the angle θ between 
    a and b satisfies 0 ≤ θ ≤ π/4 -/
theorem angle_range_given_monotonic_function 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (h1 : ‖a‖ = 2 * Real.sqrt 2 * ‖b‖) 
  (h2 : ‖b‖ ≠ 0) 
  (h3 : Monotone (fun x : ℝ => 2 * x^3 + 3 * ‖a‖ * x^2 + 6 * inner a b * x + 7)) :
  let θ := Real.arccos (inner a b / (‖a‖ * ‖b‖))
  0 ≤ θ ∧ θ ≤ π/4 := by
  sorry

end angle_range_given_monotonic_function_l2200_220038


namespace continued_fraction_value_l2200_220028

theorem continued_fraction_value : 
  ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) → x = (3 + Real.sqrt 39) / 2 := by
  sorry

end continued_fraction_value_l2200_220028


namespace min_value_expression_l2200_220048

theorem min_value_expression :
  ∃ (s₀ t₀ : ℝ), ∀ (s t : ℝ), (s + 5 - 3 * |Real.cos t|)^2 + (s - 2 * |Real.sin t|)^2 ≥ 2 ∧
  (s₀ + 5 - 3 * |Real.cos t₀|)^2 + (s₀ - 2 * |Real.sin t₀|)^2 = 2 := by
  sorry

end min_value_expression_l2200_220048


namespace soccer_league_female_fraction_l2200_220021

theorem soccer_league_female_fraction :
  -- Last year's male participants
  ∀ (last_year_males : ℕ),
  last_year_males = 30 →
  -- Total participation increase
  ∀ (total_increase_rate : ℚ),
  total_increase_rate = 108/100 →
  -- Male participation increase
  ∀ (male_increase_rate : ℚ),
  male_increase_rate = 110/100 →
  -- Female participation increase
  ∀ (female_increase_rate : ℚ),
  female_increase_rate = 115/100 →
  -- The fraction of female participants this year
  ∃ (female_fraction : ℚ),
  female_fraction = 10/43 ∧
  (∃ (last_year_females : ℕ),
    -- Total participants this year
    total_increase_rate * (last_year_males + last_year_females : ℚ) =
    -- Males this year + Females this year
    male_increase_rate * last_year_males + female_increase_rate * last_year_females ∧
    -- Female fraction calculation
    female_fraction = (female_increase_rate * last_year_females) /
      (male_increase_rate * last_year_males + female_increase_rate * last_year_females)) :=
by
  sorry

end soccer_league_female_fraction_l2200_220021


namespace minimize_distance_l2200_220011

/-- Given points P and Q in the xy-plane, and R on the line y = 2x - 4,
    prove that the value of n that minimizes PR + RQ is 0 -/
theorem minimize_distance (P Q R : ℝ × ℝ) : 
  P = (-1, -3) →
  Q = (5, 3) →
  R.1 = 2 →
  R.2 = 2 * R.1 - 4 →
  (∀ S : ℝ × ℝ, S.1 = 2 ∧ S.2 = 2 * S.1 - 4 → 
    Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) ≤
    Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) + Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2)) →
  R.2 = 0 := by
sorry

end minimize_distance_l2200_220011


namespace trapezoid_cd_length_l2200_220024

/-- Represents a trapezoid ABCD with given side lengths and perimeter -/
structure Trapezoid where
  ab : ℝ
  ad : ℝ
  bc : ℝ
  perimeter : ℝ

/-- Calculates the length of CD in the trapezoid -/
def calculate_cd (t : Trapezoid) : ℝ :=
  t.perimeter - (t.ab + t.ad + t.bc)

/-- Theorem stating that for a trapezoid with given measurements, CD = 16 -/
theorem trapezoid_cd_length (t : Trapezoid) 
  (h1 : t.ab = 12)
  (h2 : t.ad = 5)
  (h3 : t.bc = 7)
  (h4 : t.perimeter = 40) : 
  calculate_cd t = 16 := by
  sorry

#eval calculate_cd { ab := 12, ad := 5, bc := 7, perimeter := 40 }

end trapezoid_cd_length_l2200_220024


namespace max_value_of_f_l2200_220076

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^2 + 6 * x

-- Define the interval
def I : Set ℝ := {x | -2 < x ∧ x ≤ 2}

-- Theorem statement
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ I ∧ ∀ (x : ℝ), x ∈ I → f x ≤ f c ∧ f c = 9/2 :=
sorry

end max_value_of_f_l2200_220076


namespace johns_cloth_cost_l2200_220073

/-- The total cost of cloth for John, given the length and price per metre -/
def total_cost (length : ℝ) (price_per_metre : ℝ) : ℝ :=
  length * price_per_metre

/-- Theorem stating that John's total cost for cloth is $444 -/
theorem johns_cloth_cost : 
  total_cost 9.25 48 = 444 := by
  sorry

end johns_cloth_cost_l2200_220073


namespace fifth_month_sale_l2200_220001

/-- Given sales data for 6 months, prove the sale amount for the fifth month --/
theorem fifth_month_sale 
  (sale1 sale2 sale3 sale4 sale6 : ℕ)
  (average : ℚ)
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h4 : sale4 = 7230)
  (h6 : sale6 = 7391)
  (h_avg : average = 6900)
  (h_avg_def : average = (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6) :
  sale5 = 6562 := by
  sorry

end fifth_month_sale_l2200_220001


namespace abs_x_minus_one_leq_two_solution_set_l2200_220007

theorem abs_x_minus_one_leq_two_solution_set :
  {x : ℝ | |x - 1| ≤ 2} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end abs_x_minus_one_leq_two_solution_set_l2200_220007


namespace complex_equation_solution_l2200_220034

/-- Given that (1-√3i)z = √3+i, prove that z = i -/
theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I * Real.sqrt 3) * z = Real.sqrt 3 + Complex.I) : z = Complex.I := by
  sorry

end complex_equation_solution_l2200_220034


namespace multiplication_value_l2200_220040

theorem multiplication_value : 
  let original_number : ℝ := 6.5
  let divisor : ℝ := 6
  let result : ℝ := 13
  let multiplication_factor : ℝ := 12
  (original_number / divisor) * multiplication_factor = result := by
sorry

end multiplication_value_l2200_220040


namespace angle_C_measure_l2200_220066

def triangle_ABC (A B C : ℝ) := 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

theorem angle_C_measure (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : triangle_ABC A B C)
  (h_AC : c = Real.sqrt 6)
  (h_BC : b = 2)
  (h_angle_B : B = Real.pi / 3) :
  C = 5 * Real.pi / 12 := by
  sorry

end angle_C_measure_l2200_220066


namespace f_of_2_equals_3_l2200_220079

/-- Given a function f(x) = x^2 - 2x + 3, prove that f(2) = 3 -/
theorem f_of_2_equals_3 : let f : ℝ → ℝ := fun x ↦ x^2 - 2*x + 3
  f 2 = 3 := by
  sorry

end f_of_2_equals_3_l2200_220079


namespace souvenir_sales_problem_l2200_220087

/-- Souvenir sales problem -/
theorem souvenir_sales_problem
  (cost_price : ℕ)
  (initial_price : ℕ)
  (initial_sales : ℕ)
  (price_change : ℕ → ℤ)
  (sales_change : ℕ → ℤ)
  (h1 : cost_price = 40)
  (h2 : initial_price = 44)
  (h3 : initial_sales = 300)
  (h4 : ∀ x : ℕ, price_change x = x)
  (h5 : ∀ x : ℕ, sales_change x = -10 * x)
  (h6 : ∀ x : ℕ, initial_price + price_change x ≥ 44)
  (h7 : ∀ x : ℕ, initial_price + price_change x ≤ 60) :
  (∃ x : ℕ, (initial_price + price_change x - cost_price) * (initial_sales + sales_change x) = 2640 ∧
             initial_price + price_change x = 52) ∧
  (∃ x : ℕ, ∀ y : ℕ, 
    (initial_price + price_change x - cost_price) * (initial_sales + sales_change x) ≥
    (initial_price + price_change y - cost_price) * (initial_sales + sales_change y) ∧
    initial_price + price_change x = 57) ∧
  (∃ max_profit : ℕ, 
    (∃ x : ℕ, (initial_price + price_change x - cost_price) * (initial_sales + sales_change x) = max_profit) ∧
    (∀ y : ℕ, (initial_price + price_change y - cost_price) * (initial_sales + sales_change y) ≤ max_profit) ∧
    max_profit = 2890) :=
by sorry

end souvenir_sales_problem_l2200_220087


namespace triangle_center_distance_l2200_220062

/-- Given a triangle with circumradius R, inradius r, and distance d between
    the circumcenter and incenter, prove that d^2 = R^2 - 2Rr. -/
theorem triangle_center_distance (R r d : ℝ) (hR : R > 0) (hr : r > 0) (hd : d > 0) :
  d^2 = R^2 - 2*R*r := by
  sorry

end triangle_center_distance_l2200_220062


namespace tunnel_construction_equation_l2200_220057

/-- Represents the tunnel construction scenario -/
def tunnel_construction (x : ℝ) : Prop :=
  let total_length : ℝ := 1280
  let increased_speed : ℝ := 1.4 * x
  let weeks_saved : ℝ := 2
  (total_length - x) / x = (total_length - x) / increased_speed + weeks_saved

theorem tunnel_construction_equation :
  ∀ x : ℝ, x > 0 → tunnel_construction x :=
by
  sorry

end tunnel_construction_equation_l2200_220057


namespace reflection_composition_l2200_220014

/-- Two lines in the xy-plane that intersect at the origin -/
structure IntersectingLines where
  ℓ₁ : Set (ℝ × ℝ)
  ℓ₂ : Set (ℝ × ℝ)
  intersect_origin : (0, 0) ∈ ℓ₁ ∩ ℓ₂

/-- A point in the xy-plane -/
def Point := ℝ × ℝ

/-- Reflection of a point over a line -/
def reflect (p : Point) (ℓ : Set Point) : Point := sorry

theorem reflection_composition 
  (lines : IntersectingLines)
  (Q : Point)
  (h₁ : Q = (-2, 3))
  (h₂ : lines.ℓ₁ = {(x, y) | 3 * x - y = 0})
  (h₃ : reflect (reflect Q lines.ℓ₁) lines.ℓ₂ = (5, -2)) :
  lines.ℓ₂ = {(x, y) | x + 4 * y = 0} := by
  sorry

end reflection_composition_l2200_220014


namespace nikolai_wins_l2200_220035

/-- Represents a mountain goat with its jump distance -/
structure Goat where
  name : String
  jump_distance : ℕ

/-- Calculates the number of jumps needed to cover a given distance -/
def jumps_needed (goat : Goat) (distance : ℕ) : ℕ :=
  (distance + goat.jump_distance - 1) / goat.jump_distance

/-- Represents the race between two goats -/
structure Race where
  goat1 : Goat
  goat2 : Goat
  distance : ℕ

/-- Determines if the first goat is faster than the second goat -/
def is_faster (race : Race) : Prop :=
  jumps_needed race.goat1 race.distance < jumps_needed race.goat2 race.distance

theorem nikolai_wins (gennady nikolai : Goat) (h1 : gennady.jump_distance = 6)
    (h2 : nikolai.jump_distance = 4) : is_faster { goat1 := nikolai, goat2 := gennady, distance := 2000 } := by
  sorry

#check nikolai_wins

end nikolai_wins_l2200_220035


namespace angle_between_vectors_l2200_220081

/-- Given vectors a, b, and c in a real inner product space,
    if their norms are equal and nonzero, and a + b = √3 * c,
    then the angle between a and c is π/6. -/
theorem angle_between_vectors (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b c : V) (h1 : ‖a‖ = ‖b‖) (h2 : ‖b‖ = ‖c‖) (h3 : ‖a‖ ≠ 0)
  (h4 : a + b = Real.sqrt 3 • c) :
  Real.arccos (inner a c / (‖a‖ * ‖c‖)) = π / 6 := by
  sorry

end angle_between_vectors_l2200_220081


namespace log_base_condition_l2200_220052

theorem log_base_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ici 2 → |Real.log x / Real.log a| > 1) → 
  (a < 2 ∧ a ≠ 1) := by
  sorry

end log_base_condition_l2200_220052


namespace original_numbers_proof_l2200_220046

theorem original_numbers_proof : ∃ (a b c : ℕ), 
  a + b = 39 ∧ 
  b + c = 96 ∧ 
  a = 21 ∧ 
  b = 18 := by
sorry

end original_numbers_proof_l2200_220046


namespace yellow_tint_percentage_l2200_220018

/-- Calculates the percentage of yellow tint in an updated mixture -/
theorem yellow_tint_percentage 
  (original_volume : ℝ) 
  (original_yellow_percentage : ℝ) 
  (added_yellow : ℝ) : 
  original_volume = 20 →
  original_yellow_percentage = 0.5 →
  added_yellow = 6 →
  let original_yellow := original_volume * original_yellow_percentage
  let total_yellow := original_yellow + added_yellow
  let new_volume := original_volume + added_yellow
  (total_yellow / new_volume) * 100 = 61.5 := by
sorry

end yellow_tint_percentage_l2200_220018


namespace solve_star_equation_l2200_220045

-- Define the * operation
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- State the theorem
theorem solve_star_equation : 
  ∃! x : ℝ, star (x - 4) 1 = 0 ∧ x = 5 := by sorry

end solve_star_equation_l2200_220045


namespace phi_11_0_decomposition_l2200_220041

/-- The Φ₁₁⁰ series -/
def phi_11_0 : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 1
| 3 => 2
| 4 => 3
| 5 => 5
| 6 => 8
| 7 => 2
| 8 => 10
| 9 => 1
| n + 10 => phi_11_0 n

/-- The decomposed series -/
def c (n : ℕ) : ℚ := 3 * 8^n + 8 * 4^n

/-- Predicate to check if a sequence is an 11-arithmetic Fibonacci series -/
def is_11_arithmetic_fibonacci (f : ℕ → ℚ) : Prop :=
  ∀ n, f (n + 11) = f (n + 10) + f (n + 9)

/-- Predicate to check if a sequence is a geometric progression -/
def is_geometric_progression (f : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n, f (n + 1) = r * f n

theorem phi_11_0_decomposition :
  (∃ f g : ℕ → ℚ, 
    (∀ n, c n = f n + g n) ∧
    is_11_arithmetic_fibonacci f ∧
    is_11_arithmetic_fibonacci g ∧
    is_geometric_progression f ∧
    is_geometric_progression g) ∧
  (∀ n, (phi_11_0 n : ℚ) = c n) :=
sorry

end phi_11_0_decomposition_l2200_220041


namespace min_value_theorem_l2200_220032

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (x + 2*y) = Real.log x + Real.log y) : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log (a + 2*b) = Real.log a + Real.log b → 2*a + b ≥ 2*x + y) ∧ 
  (2*x + y = 9) ∧ (x = 3) := by
sorry

end min_value_theorem_l2200_220032


namespace infinitely_many_sqrt_eight_eight_eight_l2200_220094

theorem infinitely_many_sqrt_eight_eight_eight (k : ℕ) : 
  (9 * k - 1 + 0.888 : ℝ) < Real.sqrt (81 * k^2 - 2 * k) ∧ 
  Real.sqrt (81 * k^2 - 2 * k) < (9 * k - 1 + 0.889 : ℝ) := by
  sorry

end infinitely_many_sqrt_eight_eight_eight_l2200_220094


namespace line_perp_para_implies_planes_perp_l2200_220051

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (plane_perp : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_para_implies_planes_perp 
  (a : Line) (α β : Plane) :
  perp a α → para a β → plane_perp α β :=
sorry

end line_perp_para_implies_planes_perp_l2200_220051


namespace largest_number_in_ratio_l2200_220090

theorem largest_number_in_ratio (a b c d : ℕ) : 
  a + b + c + d = 1344 →
  2 * b = 3 * a →
  4 * a = 3 * b →
  5 * a = 3 * c →
  d = 5 * a / 2 →
  d = 480 := by
sorry

end largest_number_in_ratio_l2200_220090


namespace rectangle_area_measurement_error_l2200_220037

theorem rectangle_area_measurement_error (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let true_area := L * W
  let measured_length := 1.20 * L
  let measured_width := 0.90 * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - true_area
  let error_percent := (error / true_area) * 100
  error_percent = 8 := by sorry

end rectangle_area_measurement_error_l2200_220037


namespace fourth_member_income_l2200_220020

/-- Given a family of 4 members with an average income of 10000,
    where 3 members earn 8000, 15000, and 6000 respectively,
    prove that the income of the fourth member is 11000. -/
theorem fourth_member_income
  (num_members : Nat)
  (avg_income : Nat)
  (income1 income2 income3 : Nat)
  (h1 : num_members = 4)
  (h2 : avg_income = 10000)
  (h3 : income1 = 8000)
  (h4 : income2 = 15000)
  (h5 : income3 = 6000) :
  num_members * avg_income - (income1 + income2 + income3) = 11000 :=
by sorry

end fourth_member_income_l2200_220020


namespace problem_statement_l2200_220031

theorem problem_statement : |1 - Real.sqrt 3| - Real.sqrt 3 * (Real.sqrt 3 + 1) = -4 := by
  sorry

end problem_statement_l2200_220031


namespace sum_of_roots_zero_l2200_220074

theorem sum_of_roots_zero (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) → 
  p = -q → 
  p + q = 0 := by sorry

end sum_of_roots_zero_l2200_220074


namespace matt_current_age_l2200_220091

def james_age_3_years_ago : ℕ := 27
def years_since_james_age : ℕ := 3
def years_until_matt_double : ℕ := 5

def james_current_age : ℕ := james_age_3_years_ago + years_since_james_age

def james_future_age : ℕ := james_current_age + years_until_matt_double

def matt_future_age : ℕ := 2 * james_future_age

theorem matt_current_age : matt_future_age - years_until_matt_double = 65 := by
  sorry

end matt_current_age_l2200_220091


namespace prime_divides_binomial_coefficient_l2200_220004

theorem prime_divides_binomial_coefficient (p k : ℕ) (hp : Nat.Prime p) (hk : 1 ≤ k ∧ k ≤ p - 1) :
  p ∣ Nat.choose p k := by
  sorry

end prime_divides_binomial_coefficient_l2200_220004


namespace equation_equivalent_to_circles_l2200_220095

def equation (x y : ℝ) : Prop :=
  x^4 - 16*x^2 + 2*x^2*y^2 - 16*y^2 + y^4 = 4*x^3 + 4*x*y^2 - 64*x

def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 16

def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

theorem equation_equivalent_to_circles :
  ∀ x y : ℝ, equation x y ↔ (circle1 x y ∨ circle2 x y) :=
sorry

end equation_equivalent_to_circles_l2200_220095


namespace mary_fruit_purchase_cost_l2200_220019

/-- Represents the cost of each fruit type -/
structure FruitCosts where
  apple : ℕ
  orange : ℕ
  banana : ℕ
  peach : ℕ
  grape : ℕ

/-- Represents the quantity of each fruit type bought -/
structure FruitQuantities where
  apple : ℕ
  orange : ℕ
  banana : ℕ
  peach : ℕ
  grape : ℕ

/-- Calculates the total cost before discounts -/
def totalCostBeforeDiscounts (costs : FruitCosts) (quantities : FruitQuantities) : ℕ :=
  costs.apple * quantities.apple +
  costs.orange * quantities.orange +
  costs.banana * quantities.banana +
  costs.peach * quantities.peach +
  costs.grape * quantities.grape

/-- Calculates the discount for every 5 fruits bought -/
def fiveForOneDiscount (totalFruits : ℕ) : ℕ :=
  totalFruits / 5

/-- Calculates the discount for peaches and grapes bought together -/
def peachGrapeDiscount (peaches : ℕ) (grapes : ℕ) : ℕ :=
  (min (peaches / 3) (grapes / 2)) * 3

/-- Calculates the final cost after applying discounts -/
def finalCost (costs : FruitCosts) (quantities : FruitQuantities) : ℕ :=
  let totalFruits := quantities.apple + quantities.orange + quantities.banana + quantities.peach + quantities.grape
  let costBeforeDiscounts := totalCostBeforeDiscounts costs quantities
  let fiveForOneDiscountAmount := fiveForOneDiscount totalFruits
  let peachGrapeDiscountAmount := peachGrapeDiscount quantities.peach quantities.grape
  costBeforeDiscounts - fiveForOneDiscountAmount - peachGrapeDiscountAmount

/-- Theorem: Mary will pay $51 for her fruit purchase -/
theorem mary_fruit_purchase_cost :
  let costs : FruitCosts := { apple := 1, orange := 2, banana := 3, peach := 4, grape := 5 }
  let quantities : FruitQuantities := { apple := 5, orange := 3, banana := 2, peach := 6, grape := 4 }
  finalCost costs quantities = 51 := by
  sorry

end mary_fruit_purchase_cost_l2200_220019


namespace exactly_one_from_each_class_passing_at_least_one_student_passing_l2200_220071

-- Define the probability of a student passing
def p_pass : ℝ := 0.6

-- Define the number of students from each class
def n_students : ℕ := 2

-- Define the probability of exactly one student from a class passing
def p_one_pass : ℝ := n_students * p_pass * (1 - p_pass)

-- Theorem for the first question
theorem exactly_one_from_each_class_passing : 
  p_one_pass * p_one_pass = 0.2304 := by sorry

-- Theorem for the second question
theorem at_least_one_student_passing : 
  1 - (1 - p_pass)^(2 * n_students) = 0.9744 := by sorry

end exactly_one_from_each_class_passing_at_least_one_student_passing_l2200_220071


namespace number_of_divisors_720_l2200_220056

theorem number_of_divisors_720 : Finset.card (Nat.divisors 720) = 30 := by
  sorry

end number_of_divisors_720_l2200_220056


namespace root_bound_average_l2200_220084

theorem root_bound_average (A B C D : ℝ) 
  (h1 : ∀ x : ℂ, x^2 + A*x + B = 0 → Complex.abs x < 1)
  (h2 : ∀ x : ℂ, x^2 + C*x + D = 0 → Complex.abs x < 1) :
  ∀ x : ℂ, x^2 + ((A+C)/2)*x + ((B+D)/2) = 0 → Complex.abs x < 1 :=
by
  sorry

end root_bound_average_l2200_220084


namespace garage_cars_count_l2200_220029

theorem garage_cars_count (total_wheels : ℕ) (total_bicycles : ℕ) 
  (bicycle_wheels : ℕ) (car_wheels : ℕ) :
  total_wheels = 82 →
  total_bicycles = 9 →
  bicycle_wheels = 2 →
  car_wheels = 4 →
  ∃ (total_cars : ℕ), 
    total_wheels = (total_bicycles * bicycle_wheels) + (total_cars * car_wheels) ∧
    total_cars = 16 := by
  sorry

end garage_cars_count_l2200_220029


namespace largest_perfect_square_factor_3402_l2200_220093

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_3402 : 
  largest_perfect_square_factor 3402 = 81 := by sorry

end largest_perfect_square_factor_3402_l2200_220093


namespace operations_are_finite_l2200_220060

/-- Represents a (2n+1)-gon with integers assigned to its vertices -/
structure Polygon (n : ℕ) where
  vertices : Fin (2*n+1) → ℤ
  sum_positive : 0 < (Finset.univ.sum vertices)

/-- Represents an operation on three consecutive vertices -/
def operation (p : Polygon n) (i : Fin (2*n+1)) : Polygon n :=
  sorry

/-- Predicate to check if an operation is valid (i.e., y < 0) -/
def is_valid_operation (p : Polygon n) (i : Fin (2*n+1)) : Prop :=
  sorry

/-- A sequence of operations -/
def operation_sequence (p : Polygon n) : List (Fin (2*n+1)) → Polygon n
  | [] => p
  | (i :: is) => operation_sequence (operation p i) is

/-- Theorem stating that any sequence of valid operations is finite -/
theorem operations_are_finite (n : ℕ) (p : Polygon n) :
  ∃ (N : ℕ), ∀ (seq : List (Fin (2*n+1))),
    (∀ i ∈ seq, is_valid_operation p i) →
    seq.length ≤ N :=
  sorry

end operations_are_finite_l2200_220060


namespace parabola_range_l2200_220096

theorem parabola_range (a b m : ℝ) : 
  (∃ x y : ℝ, y = -x^2 + 2*a*x + b ∧ y = x^2) →
  (m*a - (a^2 + b) - 2*m + 1 = 0) →
  (m ≥ 5/2 ∨ m ≤ 3/2) :=
by sorry

end parabola_range_l2200_220096


namespace inequality_of_three_nonnegative_reals_l2200_220080

theorem inequality_of_three_nonnegative_reals (a b c : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) : 
  |c * a - a * b| + |a * b - b * c| + |b * c - c * a| ≤ 
  |b^2 - c^2| + |c^2 - a^2| + |a^2 - b^2| := by
  sorry

end inequality_of_three_nonnegative_reals_l2200_220080


namespace smallest_fraction_greater_than_four_ninths_l2200_220005

theorem smallest_fraction_greater_than_four_ninths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (4 : ℚ) / 9 < (a : ℚ) / b →
    (41 : ℚ) / 92 ≤ (a : ℚ) / b :=
by sorry

end smallest_fraction_greater_than_four_ninths_l2200_220005


namespace log_2_bounds_l2200_220008

theorem log_2_bounds (h1 : 10^3 = 1000) (h2 : 10^4 = 10000) 
                     (h3 : 2^11 = 2048) (h4 : 2^14 = 16384) :
  3/11 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 2/7 := by
  sorry

end log_2_bounds_l2200_220008


namespace mortgage_duration_l2200_220067

theorem mortgage_duration (house_price deposit monthly_payment : ℕ) :
  house_price = 280000 →
  deposit = 40000 →
  monthly_payment = 2000 →
  (house_price - deposit) / monthly_payment / 12 = 10 := by
  sorry

end mortgage_duration_l2200_220067


namespace four_genuine_probability_l2200_220023

/-- The number of genuine coins -/
def genuine_coins : ℕ := 12

/-- The number of counterfeit coins -/
def counterfeit_coins : ℕ := 3

/-- The total number of coins -/
def total_coins : ℕ := genuine_coins + counterfeit_coins

/-- The probability of selecting 4 genuine coins when drawing two pairs randomly without replacement -/
def prob_four_genuine : ℚ := 33 / 91

theorem four_genuine_probability :
  (genuine_coins.choose 2 * (genuine_coins - 2).choose 2) / (total_coins.choose 2 * (total_coins - 2).choose 2) = prob_four_genuine := by
  sorry

end four_genuine_probability_l2200_220023


namespace perpendicular_vectors_x_equals_one_l2200_220099

/-- Two vectors a and b in R² -/
def a : ℝ × ℝ := (3, 1)
def b : ℝ → ℝ × ℝ := λ x ↦ (x, -3)

/-- The dot product of two vectors in R² -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: If a and b are perpendicular, then x = 1 -/
theorem perpendicular_vectors_x_equals_one :
  (∃ x : ℝ, dot_product a (b x) = 0) → 
  (∃ x : ℝ, b x = (1, -3)) :=
by sorry

end perpendicular_vectors_x_equals_one_l2200_220099


namespace triangle_area_zero_l2200_220013

theorem triangle_area_zero (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a + b + c = 6 →
  a*b + b*c + a*c = 11 →
  a*b*c = 6 →
  ∃ (s : ℝ), s*(s - a)*(s - b)*(s - c) = 0 := by
  sorry

end triangle_area_zero_l2200_220013


namespace ellipse_and_quadratic_conditions_l2200_220010

/-- Represents an ellipse equation with parameter a -/
def is_ellipse (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2*a) + y^2 / (3*a - 6) = 1

/-- Checks if the ellipse has foci on the x-axis -/
def has_foci_on_x_axis (a : ℝ) : Prop :=
  2*a < 3*a - 6

/-- Represents the quadratic inequality with parameter a -/
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  x^2 + (a + 4)*x + 16 > 0

/-- Checks if the solution set of the quadratic inequality is ℝ -/
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x : ℝ, quadratic_inequality a x

/-- The main theorem stating the conditions for a -/
theorem ellipse_and_quadratic_conditions (a : ℝ) :
  (is_ellipse a ∧ has_foci_on_x_axis a ∧ solution_set_is_reals a) ↔ (2 < a ∧ a < 4) :=
sorry

end ellipse_and_quadratic_conditions_l2200_220010
