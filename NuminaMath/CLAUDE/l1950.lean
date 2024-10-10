import Mathlib

namespace age_ratio_proof_l1950_195055

/-- 
Given three people a, b, and c, where:
- a is two years older than b
- The total age of a, b, and c is 27
- b is 10 years old

This theorem proves that the ratio of b's age to c's age is 2:1
-/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 27 →
  b = 10 →
  b = 2 * c :=
by
  sorry

end age_ratio_proof_l1950_195055


namespace club_officer_selection_l1950_195072

/-- Represents the number of ways to choose officers in a club -/
def chooseOfficers (totalMembers boyCount girlCount : ℕ) : ℕ :=
  totalMembers * (if boyCount = girlCount then boyCount else 0) * (boyCount - 1)

/-- Theorem stating the number of ways to choose officers in the given conditions -/
theorem club_officer_selection :
  let totalMembers := 24
  let boyCount := 12
  let girlCount := 12
  chooseOfficers totalMembers boyCount girlCount = 3168 := by
  sorry

end club_officer_selection_l1950_195072


namespace f_max_values_l1950_195083

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1)^2 / (x * (x^2 - 1))

theorem f_max_values (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = a ∧ f y = a ∧ f z = a) ↔ a < -4 ∨ a > 4 :=
sorry

end f_max_values_l1950_195083


namespace min_value_of_inverse_squares_l1950_195022

theorem min_value_of_inverse_squares (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*a*x + 4*a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y + b^2 - 1 = 0) →
  (∃! (l : ℝ → ℝ), ∀ (x y : ℝ), (x^2 + y^2 + 4*a*x + 4*a^2 - 4 = 0 ∨ x^2 + y^2 - 2*b*y + b^2 - 1 = 0) → 
    y = l x ∧ (∀ (x' y' : ℝ), y' = l x' → (x'^2 + y'^2 + 4*a*x' + 4*a^2 - 4 > 0 ∧ x'^2 + y'^2 - 2*b*y' + b^2 - 1 > 0) ∨
    (x'^2 + y'^2 + 4*a*x' + 4*a^2 - 4 < 0 ∧ x'^2 + y'^2 - 2*b*y' + b^2 - 1 < 0))) →
  (1 / a^2 + 1 / b^2 ≥ 9) ∧ (∃ (a' b' : ℝ), a' ≠ 0 ∧ b' ≠ 0 ∧ 1 / a'^2 + 1 / b'^2 = 9) :=
by sorry

end min_value_of_inverse_squares_l1950_195022


namespace autumn_grain_purchase_l1950_195097

/-- 
Theorem: If the total purchase of autumn grain nationwide exceeded 180 million tons, 
and x represents the amount of autumn grain purchased in China this year in billion tons, 
then x > 1.8.
-/
theorem autumn_grain_purchase (x : ℝ) 
  (h : x * 1000 > 180) : x > 1.8 := by
  sorry

end autumn_grain_purchase_l1950_195097


namespace f_value_at_one_l1950_195064

def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 20

def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 50*x + c

theorem f_value_at_one (a b c : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c 1 = -217 := by
  sorry

end f_value_at_one_l1950_195064


namespace sphere_to_cone_radius_l1950_195050

/-- The radius of a sphere that transforms into a cone with equal volume --/
theorem sphere_to_cone_radius (r : ℝ) (h : r = 3 * Real.rpow 2 (1/3)) :
  ∃ R : ℝ, 
    (4/3) * Real.pi * R^3 = 2 * Real.pi * r^3 ∧ 
    R = 3 * Real.rpow 3 (1/3) :=
sorry

end sphere_to_cone_radius_l1950_195050


namespace sum_of_digits_999_base_7_l1950_195030

def base_7_representation (n : ℕ) : List ℕ :=
  sorry

def sum_of_digits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_999_base_7 :
  sum_of_digits (base_7_representation 999) = 15 := by
  sorry

end sum_of_digits_999_base_7_l1950_195030


namespace three_eighths_percent_of_160_l1950_195074

theorem three_eighths_percent_of_160 : (3 / 8 / 100) * 160 = 0.6 := by
  sorry

end three_eighths_percent_of_160_l1950_195074


namespace new_average_production_l1950_195096

theorem new_average_production (n : ℕ) (past_average : ℝ) (today_production : ℝ) 
  (h1 : n = 11)
  (h2 : past_average = 50)
  (h3 : today_production = 110) : 
  (n * past_average + today_production) / (n + 1) = 55 :=
by sorry

end new_average_production_l1950_195096


namespace seashells_count_l1950_195042

theorem seashells_count (mary_shells jessica_shells : ℕ) 
  (h1 : mary_shells = 18) 
  (h2 : jessica_shells = 41) : 
  mary_shells + jessica_shells = 59 := by
  sorry

end seashells_count_l1950_195042


namespace base_4_9_digit_difference_l1950_195040

theorem base_4_9_digit_difference :
  let n : ℕ := 1234
  let base_4_digits := (Nat.log n 4).succ
  let base_9_digits := (Nat.log n 9).succ
  base_4_digits = base_9_digits + 2 :=
by sorry

end base_4_9_digit_difference_l1950_195040


namespace muffin_banana_cost_ratio_l1950_195036

theorem muffin_banana_cost_ratio :
  ∀ (muffin_cost banana_cost : ℝ),
  (6 * muffin_cost + 5 * banana_cost = (3 * muffin_cost + 20 * banana_cost) / 2) →
  (muffin_cost / banana_cost = 10 / 9) :=
by
  sorry

end muffin_banana_cost_ratio_l1950_195036


namespace constant_fifth_term_binomial_expansion_l1950_195084

theorem constant_fifth_term_binomial_expansion (a x : ℝ) (n : ℕ) :
  (∃ k : ℝ, k ≠ 0 ∧ (Nat.choose n 4) * a^(n-4) * (-1)^4 * x^(n-8) = k) →
  n = 8 := by
sorry

end constant_fifth_term_binomial_expansion_l1950_195084


namespace joan_gave_43_seashells_l1950_195033

/-- The number of seashells Joan initially found on the beach. -/
def initial_seashells : ℕ := 70

/-- The number of seashells Joan has left after giving some to Sam. -/
def remaining_seashells : ℕ := 27

/-- The number of seashells Joan gave to Sam. -/
def seashells_given_to_sam : ℕ := initial_seashells - remaining_seashells

/-- Theorem stating that Joan gave 43 seashells to Sam. -/
theorem joan_gave_43_seashells : seashells_given_to_sam = 43 := by
  sorry

end joan_gave_43_seashells_l1950_195033


namespace exists_card_with_1024_l1950_195065

/-- The number of cards for each natural number up to 1968 -/
def num_cards (n : ℕ) : ℕ := n

/-- The condition that each card has divisors of its number written on it -/
def has_divisors (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d ≤ 1968 → (num_cards d) ≥ 1

/-- The main theorem to prove -/
theorem exists_card_with_1024 (h : ∀ n ≤ 1968, has_divisors n) :
  (num_cards 1024) > 0 :=
sorry

end exists_card_with_1024_l1950_195065


namespace parabola_axis_of_symmetry_axis_of_symmetry_example_l1950_195017

/-- The axis of symmetry of a parabola y = ax^2 + bx + c is x = -b / (2a) -/
theorem parabola_axis_of_symmetry (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  ∃ x₀ : ℝ, x₀ = -b / (2 * a) ∧ ∀ x : ℝ, f (x₀ + x) = f (x₀ - x) :=
sorry

/-- The axis of symmetry of the parabola y = -x^2 + 4x + 1 is the line x = 2 -/
theorem axis_of_symmetry_example :
  let f : ℝ → ℝ := λ x => -x^2 + 4*x + 1
  ∃ x₀ : ℝ, x₀ = 2 ∧ ∀ x : ℝ, f (x₀ + x) = f (x₀ - x) :=
sorry

end parabola_axis_of_symmetry_axis_of_symmetry_example_l1950_195017


namespace f_difference_512_256_l1950_195013

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Define the f function as described in the problem
def f (n : ℕ) : ℚ := (sum_of_divisors n : ℚ) / n

-- Theorem statement
theorem f_difference_512_256 : f 512 - f 256 = 1 / 512 := by sorry

end f_difference_512_256_l1950_195013


namespace absolute_value_inequality_solution_set_l1950_195023

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 1} = Set.Ioo 0 2 := by sorry

end absolute_value_inequality_solution_set_l1950_195023


namespace house_locations_contradiction_l1950_195057

-- Define the directions
inductive Direction
  | North
  | South
  | East
  | West
  | Northeast
  | Northwest
  | Southeast
  | Southwest

-- Define a function to get the opposite direction
def oppositeDirection (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.South
  | Direction.South => Direction.North
  | Direction.East => Direction.West
  | Direction.West => Direction.East
  | Direction.Northeast => Direction.Southwest
  | Direction.Northwest => Direction.Southeast
  | Direction.Southeast => Direction.Northwest
  | Direction.Southwest => Direction.Northeast

-- Define the theorem
theorem house_locations_contradiction :
  ∀ (house1 house2 : Type) (direction1 direction2 : Direction),
    (direction1 = Direction.Southeast ∧ direction2 = Direction.Southwest) →
    (oppositeDirection direction1 ≠ direction2) :=
by sorry

end house_locations_contradiction_l1950_195057


namespace fraction_equality_l1950_195063

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 8)
  (h2 : c / b = 4)
  (h3 : c / d = 1 / 3) :
  d / a = 3 / 2 := by
  sorry

end fraction_equality_l1950_195063


namespace nearest_integer_to_3_plus_sqrt2_pow6_l1950_195092

theorem nearest_integer_to_3_plus_sqrt2_pow6 :
  ∃ n : ℤ, n = 7414 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2) ^ 6 - n| ≤ |((3 : ℝ) + Real.sqrt 2) ^ 6 - m| :=
sorry

end nearest_integer_to_3_plus_sqrt2_pow6_l1950_195092


namespace log2_derivative_l1950_195035

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end log2_derivative_l1950_195035


namespace ages_ratio_three_to_one_l1950_195021

/-- Represents a person's age --/
structure Age where
  years : ℕ

/-- Represents the ages of Claire and Pete --/
structure AgesPair where
  claire : Age
  pete : Age

/-- The conditions of the problem --/
def problem_conditions (ages : AgesPair) : Prop :=
  (ages.claire.years - 3 = 2 * (ages.pete.years - 3)) ∧
  (ages.pete.years - 7 = (ages.claire.years - 7) / 4)

/-- The theorem to prove --/
theorem ages_ratio_three_to_one (ages : AgesPair) :
  problem_conditions ages →
  ∃ (claire_age pete_age : ℕ),
    claire_age = ages.claire.years - 6 ∧
    pete_age = ages.pete.years - 6 ∧
    3 * pete_age = claire_age :=
by
  sorry


end ages_ratio_three_to_one_l1950_195021


namespace arithmetic_sequence_product_l1950_195051

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

/-- The sequence is increasing -/
def IncreasingSequence (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b →
  IncreasingSequence b →
  b 5 * b 6 = 21 →
  b 4 * b 7 = -779 := by
  sorry

end arithmetic_sequence_product_l1950_195051


namespace extreme_value_derivative_l1950_195078

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for f to have an extreme value at x₀
def has_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≤ f x ∨ f x₀ ≥ f x

-- State the theorem
theorem extreme_value_derivative (x₀ : ℝ) :
  (has_extreme_value f x₀ → deriv f x₀ = 0) ∧
  ¬(deriv f x₀ = 0 → has_extreme_value f x₀) :=
sorry

end extreme_value_derivative_l1950_195078


namespace sum_of_a_and_b_l1950_195070

theorem sum_of_a_and_b (a b : ℝ) (h1 : a > b) (h2 : |a| = 9) (h3 : b^2 = 4) :
  a + b = 11 ∨ a + b = 7 := by
  sorry

end sum_of_a_and_b_l1950_195070


namespace age_height_not_function_l1950_195089

-- Define the relationships
def angle_sine_relation : Set (ℝ × ℝ) := sorry
def square_side_area_relation : Set (ℝ × ℝ) := sorry
def polygon_sides_angles_relation : Set (ℕ × ℝ) := sorry
def age_height_relation : Set (ℕ × ℝ) := sorry

-- Define the property of being a function
def is_function (r : Set (α × β)) : Prop := 
  ∀ x y z, (x, y) ∈ r → (x, z) ∈ r → y = z

-- State the theorem
theorem age_height_not_function :
  is_function angle_sine_relation ∧ 
  is_function square_side_area_relation ∧ 
  is_function polygon_sides_angles_relation → 
  ¬ is_function age_height_relation := by
sorry

end age_height_not_function_l1950_195089


namespace three_chapters_eight_pages_l1950_195075

/-- Calculates the total number of pages read given the number of chapters and pages per chapter -/
def pages_read (chapters : ℕ) (pages_per_chapter : ℕ) : ℕ :=
  chapters * pages_per_chapter

/-- Proves that reading 3 chapters of 8 pages each results in 24 pages read -/
theorem three_chapters_eight_pages :
  pages_read 3 8 = 24 := by
  sorry

end three_chapters_eight_pages_l1950_195075


namespace multiples_of_4_between_80_and_300_l1950_195093

theorem multiples_of_4_between_80_and_300 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n > 80 ∧ n < 300) (Finset.range 300)).card = 54 := by
  sorry

end multiples_of_4_between_80_and_300_l1950_195093


namespace function_properties_l1950_195019

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- State the theorem
theorem function_properties (a b : ℝ) :
  a ≠ 0 →
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (a = -1 ∧ b = 4) ∧
  (f a b 1 = 2 → a > 0 → b > 0 → 
    (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1/a' + 4/b' ≥ 9) ∧
    (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1/a' + 4/b' = 9)) :=
by sorry

end function_properties_l1950_195019


namespace system_solution_l1950_195077

/-- Given a system of equations in x, y, and m, prove the relationship between x and y,
    and find the value of m when x + y = -10 -/
theorem system_solution (x y m : ℝ) 
  (eq1 : 3 * x + 5 * y = m + 2)
  (eq2 : 2 * x + 3 * y = m) :
  (y = 1 - x / 2) ∧ 
  (x + y = -10 → m = -8) := by
  sorry

end system_solution_l1950_195077


namespace modulus_of_complex_number_l1950_195056

theorem modulus_of_complex_number (z : ℂ) : z = 1 - 2*I → Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_complex_number_l1950_195056


namespace triangle_side_sum_range_l1950_195043

open Real

theorem triangle_side_sum_range (A B C a b c : Real) : 
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  cos B / b + cos C / c = 2 * sqrt 3 * sin A / (3 * sin C) →
  cos B + sqrt 3 * sin B = 2 →
  3 / 2 < a + c ∧ a + c ≤ sqrt 3 := by
sorry

end triangle_side_sum_range_l1950_195043


namespace prob_no_roots_l1950_195037

/-- A random variable following a normal distribution with mean 1 and variance s² -/
def normal_dist (s : ℝ) : Type := ℝ

/-- The probability density function of a normal distribution -/
noncomputable def pdf (s : ℝ) (x : ℝ) : ℝ := sorry

/-- The cumulative distribution function of a normal distribution -/
noncomputable def cdf (s : ℝ) (x : ℝ) : ℝ := sorry

/-- The quadratic function f(x) = x² + 2x + ξ -/
def f (ξ : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + ξ

/-- The statement that f(x) has no roots -/
def no_roots (ξ : ℝ) : Prop := ∀ x, f ξ x ≠ 0

/-- The main theorem -/
theorem prob_no_roots (s : ℝ) (h : s > 0) : 
  (1 - cdf s 1) = 1/2 := by sorry

end prob_no_roots_l1950_195037


namespace infinitely_many_not_sum_of_seven_sixth_powers_l1950_195025

theorem infinitely_many_not_sum_of_seven_sixth_powers :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ 
  (∀ a ∈ S, ∀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ, 
   a ≠ a₁^6 + a₂^6 + a₃^6 + a₄^6 + a₅^6 + a₆^6 + a₇^6) := by
  sorry

end infinitely_many_not_sum_of_seven_sixth_powers_l1950_195025


namespace divisor_problem_l1950_195059

theorem divisor_problem (N : ℕ) (D : ℕ) : 
  N % 5 = 0 ∧ N / 5 = 5 ∧ N % D = 3 → D = 4 := by
  sorry

end divisor_problem_l1950_195059


namespace factorization_of_polynomial_l1950_195039

theorem factorization_of_polynomial (x : ℝ) :
  x^2 + 6*x + 9 - 100*x^4 = (-10*x^2 + x + 3) * (10*x^2 + x + 3) :=
by sorry

end factorization_of_polynomial_l1950_195039


namespace intersection_complement_P_and_Q_l1950_195060

-- Define the set P
def P : Set ℝ := {y | ∃ x > 0, y = (1/2)^x}

-- Define the set Q
def Q : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}

-- Define the complement of P in ℝ
def complement_P : Set ℝ := {y | y ≤ 0 ∨ y ≥ 1}

-- Theorem statement
theorem intersection_complement_P_and_Q :
  (complement_P ∩ Q) = Set.Icc 1 2 := by sorry

end intersection_complement_P_and_Q_l1950_195060


namespace initial_investment_l1950_195061

/-- Given an initial investment A at a simple annual interest rate r,
    prove that A = 5000 when the interest on A is $250 and
    the interest on $20,000 at the same rate is $1000. -/
theorem initial_investment (A r : ℝ) : 
  A > 0 →
  r > 0 →
  A * r / 100 = 250 →
  20000 * r / 100 = 1000 →
  A = 5000 := by
sorry

end initial_investment_l1950_195061


namespace constant_term_expansion_l1950_195087

theorem constant_term_expansion (a : ℝ) : 
  a > 0 → (∃ k : ℕ, k = (a^2 * 2^2 * 6 : ℝ) ∧ k = 96) → a = 2 := by
  sorry

end constant_term_expansion_l1950_195087


namespace greatest_divisor_with_remainders_l1950_195031

theorem greatest_divisor_with_remainders (n : ℕ) : 
  (∃ k1 : ℕ, 1657 = n * k1 + 6) ∧ 
  (∃ k2 : ℕ, 2037 = n * k2 + 5) ∧ 
  (∀ m : ℕ, (∃ j1 : ℕ, 1657 = m * j1 + 6) ∧ (∃ j2 : ℕ, 2037 = m * j2 + 5) → m ≤ n) →
  n = 127 := by
sorry

end greatest_divisor_with_remainders_l1950_195031


namespace johns_age_ratio_l1950_195053

/-- The ratio of John's age 5 years ago to his age in 8 years -/
def age_ratio (current_age : ℕ) : ℚ :=
  (current_age - 5 : ℚ) / (current_age + 8)

/-- Theorem stating that the ratio of John's age 5 years ago to his age in 8 years is 1:2 -/
theorem johns_age_ratio :
  age_ratio 18 = 1 / 2 := by
  sorry

end johns_age_ratio_l1950_195053


namespace max_k_for_sqrt_inequality_l1950_195038

theorem max_k_for_sqrt_inequality : 
  (∃ (k : ℝ), ∀ (l : ℝ), 
    (∃ (x : ℝ), 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ l) → 
    k ≥ l) ∧ 
  (∃ (x : ℝ), 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ Real.sqrt 6) :=
by sorry

end max_k_for_sqrt_inequality_l1950_195038


namespace cube_volume_problem_l1950_195003

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  a^3 - ((a - 2) * a * (a + 2)) = 16 → 
  a^3 = 8 := by
sorry

end cube_volume_problem_l1950_195003


namespace initial_queue_size_l1950_195026

theorem initial_queue_size (n : ℕ) : 
  (∀ A : ℕ, A = 41 * n) →  -- Current total age
  (A + 69 = 45 * (n + 1)) → -- New total age after 7th person joins
  n = 6 := by
sorry

end initial_queue_size_l1950_195026


namespace circle_max_sum_l1950_195024

theorem circle_max_sum :
  ∀ x y : ℤ, x^2 + y^2 = 16 → x + y ≤ 4 :=
by sorry

end circle_max_sum_l1950_195024


namespace max_dot_product_unit_vector_l1950_195086

theorem max_dot_product_unit_vector (a b : ℝ × ℝ) :
  (∀ (x y : ℝ), a = (x, y) → x^2 + y^2 = 1) →
  b = (Real.sqrt 3, -1) →
  (∃ (m : ℝ), m = (a.1 * b.1 + a.2 * b.2) ∧ 
    ∀ (x y : ℝ), x^2 + y^2 = 1 → (x * b.1 + y * b.2) ≤ m) →
  (∃ (max : ℝ), max = 2 ∧ 
    ∀ (x y : ℝ), x^2 + y^2 = 1 → (x * b.1 + y * b.2) ≤ max) :=
by
  sorry

end max_dot_product_unit_vector_l1950_195086


namespace absolute_value_equation_unique_solution_l1950_195014

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 3| = |x + 5| := by
sorry

end absolute_value_equation_unique_solution_l1950_195014


namespace circle_C_equation_l1950_195090

/-- The standard equation of a circle with center (h, k) and radius r -/
def standard_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Circle C with center (1, 2) and radius 3 -/
def circle_C (x y : ℝ) : Prop :=
  standard_circle_equation x y 1 2 3

theorem circle_C_equation :
  ∀ x y : ℝ, circle_C x y ↔ (x - 1)^2 + (y - 2)^2 = 9 :=
by
  sorry

end circle_C_equation_l1950_195090


namespace intersection_of_sets_l1950_195047

theorem intersection_of_sets :
  let A : Set ℝ := {x | -2 < x ∧ x < 3}
  let B : Set ℝ := {x | ∃ n : ℤ, x = 2 * n}
  A ∩ B = {0, 2} := by
sorry

end intersection_of_sets_l1950_195047


namespace car_distribution_l1950_195015

def total_production : ℕ := 5650000
def first_supplier : ℕ := 1000000
def second_supplier : ℕ := first_supplier + 500000
def third_supplier : ℕ := first_supplier + second_supplier

theorem car_distribution (fourth_supplier fifth_supplier : ℕ) : 
  fourth_supplier = fifth_supplier ∧
  first_supplier + second_supplier + third_supplier + fourth_supplier + fifth_supplier = total_production →
  fourth_supplier = 325000 := by
sorry

end car_distribution_l1950_195015


namespace polynomial_coefficient_sum_l1950_195002

theorem polynomial_coefficient_sum (a k n : ℤ) : 
  (∀ x : ℝ, (3 * x^2 + 2) * (2 * x^3 - 7) = a * x^5 + k * x^2 + n) →
  a - n + k = -1 := by
  sorry

end polynomial_coefficient_sum_l1950_195002


namespace sine_double_angle_l1950_195001

theorem sine_double_angle (A : ℝ) (h : Real.cos (π/4 + A) = 5/13) : 
  Real.sin (2 * A) = 119/169 := by
  sorry

end sine_double_angle_l1950_195001


namespace f_matches_table_l1950_195020

/-- The function that generates the output values -/
def f (n : ℕ) : ℕ := 2 * n - 1

/-- The proposition that the function f matches the given table for n from 1 to 5 -/
theorem f_matches_table : 
  f 1 = 1 ∧ f 2 = 3 ∧ f 3 = 5 ∧ f 4 = 7 ∧ f 5 = 9 := by
  sorry

#check f_matches_table

end f_matches_table_l1950_195020


namespace abs_x_minus_one_necessary_not_sufficient_l1950_195034

theorem abs_x_minus_one_necessary_not_sufficient :
  (∀ x : ℝ, x * (x + 1) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x + 1) < 0)) :=
by sorry

end abs_x_minus_one_necessary_not_sufficient_l1950_195034


namespace regular_1001_gon_labeling_existence_l1950_195027

theorem regular_1001_gon_labeling_existence :
  ∃ f : Fin 1001 → Fin 1001,
    Function.Bijective f ∧
    ∀ (r : Fin 1001) (b : Bool),
      ∃ i : Fin 1001,
        f ((i + r) % 1001) = if b then i else (1001 - i) % 1001 := by
  sorry

end regular_1001_gon_labeling_existence_l1950_195027


namespace petya_marking_strategy_l1950_195008

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangle that can be placed on the board -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)

/-- The minimum number of cells needed to be marked to uniquely determine 
    the position of a rectangle on a board -/
def min_marked_cells (b : Board) (r : Rectangle) : ℕ := sorry

/-- The main theorem stating the minimum number of cells Petya needs to mark -/
theorem petya_marking_strategy (b : Board) (r : Rectangle) : 
  b.rows = 13 ∧ b.cols = 13 ∧ r.length = 6 ∧ r.width = 1 →
  min_marked_cells b r = 84 := by sorry

end petya_marking_strategy_l1950_195008


namespace sum_plus_even_count_l1950_195094

def sum_of_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_plus_even_count : 
  let x := sum_of_range 50 60
  let y := count_even_in_range 50 60
  x + y = 611 := by sorry

end sum_plus_even_count_l1950_195094


namespace parabola_vertex_l1950_195045

/-- The vertex of the parabola y = x^2 - 2 is at (0, -2) -/
theorem parabola_vertex :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2
  ∃! (h k : ℝ), (∀ x, f x = (x - h)^2 + k) ∧ h = 0 ∧ k = -2 := by
  sorry

end parabola_vertex_l1950_195045


namespace platform_length_l1950_195066

/-- The length of a platform given train passing times -/
theorem platform_length (train_length : ℝ) (time_pass_man : ℝ) (time_cross_platform : ℝ) 
  (h1 : train_length = 178)
  (h2 : time_pass_man = 8)
  (h3 : time_cross_platform = 20) :
  let train_speed := train_length / time_pass_man
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 267 := by
sorry

end platform_length_l1950_195066


namespace no_harmonic_point_on_reciprocal_unique_harmonic_point_range_of_m_l1950_195018

-- Definition of a harmonic point
def is_harmonic_point (x y : ℝ) : Prop := x = y

-- Part 1: No harmonic point on y = -4/x
theorem no_harmonic_point_on_reciprocal : ¬∃ x : ℝ, is_harmonic_point x (-4/x) := by sorry

-- Part 2: Quadratic function with one harmonic point
def quadratic_function (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + 6 * x + c

theorem unique_harmonic_point :
  ∃! (a c : ℝ), a ≠ 0 ∧ 
  (∃! x : ℝ, is_harmonic_point x (quadratic_function a c x)) ∧
  is_harmonic_point (5/2) (quadratic_function a c (5/2)) := by sorry

-- Part 3: Range of m for the modified quadratic function
def modified_quadratic (x : ℝ) : ℝ := -x^2 + 6*x - 6

theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, 1 ≤ x → x ≤ m → -1 ≤ modified_quadratic x ∧ modified_quadratic x ≤ 3) ↔
  (3 ≤ m ∧ m ≤ 5) := by sorry

end no_harmonic_point_on_reciprocal_unique_harmonic_point_range_of_m_l1950_195018


namespace max_b_value_l1950_195071

noncomputable section

variable (a b : ℝ)

def f (x : ℝ) := (3/2) * x^2 - 2*a*x

def g (x : ℝ) := a^2 * Real.log x + b

def common_point (x : ℝ) := f a x = g a b x

def common_tangent (x : ℝ) := deriv (f a) x = deriv (g a b) x

theorem max_b_value (h1 : a > 0) 
  (h2 : ∃ x > 0, common_point a b x ∧ common_tangent a b x) :
  ∃ b_max : ℝ, b_max = 1 / (2 * Real.exp 2) ∧ 
  (∀ b : ℝ, (∃ x > 0, common_point a b x ∧ common_tangent a b x) → b ≤ b_max) :=
sorry

end max_b_value_l1950_195071


namespace average_half_median_l1950_195068

theorem average_half_median (a b c : ℤ) : 
  a < b → b < c → a = 0 → (a + b + c) / 3 = b / 2 → c / b = 1 / 2 := by
  sorry

end average_half_median_l1950_195068


namespace rectangle_ratio_l1950_195082

/-- Configuration of squares and rectangle forming a large square -/
structure SquareConfiguration where
  s : ℝ  -- Side length of small squares
  large_square_side : ℝ  -- Side length of the large square
  rectangle_length : ℝ  -- Length of the rectangle
  rectangle_width : ℝ  -- Width of the rectangle

/-- Properties of the square configuration -/
def valid_configuration (config : SquareConfiguration) : Prop :=
  config.s > 0 ∧
  config.large_square_side = 3 * config.s ∧
  config.rectangle_length = config.large_square_side ∧
  config.rectangle_width = config.s

/-- Theorem stating the ratio of rectangle's length to width -/
theorem rectangle_ratio (config : SquareConfiguration) 
  (h : valid_configuration config) : 
  config.rectangle_length / config.rectangle_width = 3 := by
  sorry

end rectangle_ratio_l1950_195082


namespace abc_inequality_l1950_195005

theorem abc_inequality (a b c : ℝ) (h : a^2 * b * c + a * b^2 * c + a * b * c^2 = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 := by
  sorry

end abc_inequality_l1950_195005


namespace complex_equation_solution_l1950_195079

theorem complex_equation_solution (a : ℝ) (z : ℂ) : 
  z * Complex.I = (a + 1 : ℂ) + 4 * Complex.I → Complex.abs z = 5 → a = 2 ∨ a = -4 := by
  sorry

end complex_equation_solution_l1950_195079


namespace quadratic_polynomial_property_l1950_195016

-- Define a quadratic polynomial with integer coefficients
def QuadraticPolynomial (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

theorem quadratic_polynomial_property (a b c : ℤ) :
  let f := QuadraticPolynomial a b c
  (f (Real.sqrt 3) - f (Real.sqrt 2) = 4) →
  (f (Real.sqrt 10) - f (Real.sqrt 7) = 12) :=
by sorry

end quadratic_polynomial_property_l1950_195016


namespace sequence_existence_l1950_195000

theorem sequence_existence (n : ℕ) (hn : n ≥ 3) :
  (∃ (a : ℕ → ℝ), 
    (∀ i ∈ Finset.range n, a i * a (i + 1) + 1 = a (i + 2)) ∧
    (a (n + 1) = a 1) ∧
    (a (n + 2) = a 2)) ↔ 
  (3 ∣ n) :=
sorry

end sequence_existence_l1950_195000


namespace functional_equation_solution_l1950_195029

/-- Given a fixed positive integer N, prove that any function f satisfying
    the given conditions is identically zero. -/
theorem functional_equation_solution (N : ℕ+) (f : ℤ → ℝ)
  (h1 : ∀ k : ℤ, f (2 * k) = 2 * f k)
  (h2 : ∀ k : ℤ, f (N - k) = f k) :
  ∀ a : ℤ, f a = 0 := by sorry

end functional_equation_solution_l1950_195029


namespace valid_numbers_l1950_195012

def is_valid_number (n : ℕ) : Prop :=
  Odd n ∧
  ∃ (a b : ℕ),
    10 ≤ a ∧ a ≤ 99 ∧
    (∃ (k : ℕ), n = 10^k * a + b) ∧
    n = 149 * b

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n → n = 745 ∨ n = 3725 :=
sorry

end valid_numbers_l1950_195012


namespace polynomial_remainder_theorem_l1950_195076

def f (x : ℝ) : ℝ := x^4 - 8*x^3 + 12*x^2 + 5*x - 20

theorem polynomial_remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x + 2) * q x + 98 :=
sorry

end polynomial_remainder_theorem_l1950_195076


namespace tan_alpha_third_quadrant_l1950_195067

theorem tan_alpha_third_quadrant (α : Real) 
  (h1 : Real.sin (Real.pi + α) = 3/5)
  (h2 : π < α ∧ α < 3*π/2) : 
  Real.tan α = 3/4 := by
sorry

end tan_alpha_third_quadrant_l1950_195067


namespace allocation_schemes_eq_36_l1950_195007

/-- Represents the number of fire drill sites -/
def num_sites : ℕ := 3

/-- Represents the number of fire brigades -/
def num_brigades : ℕ := 4

/-- Represents the condition that each site must have at least one brigade -/
def min_brigade_per_site : ℕ := 1

/-- The number of ways to allocate fire brigades to sites -/
def allocation_schemes : ℕ := sorry

/-- Theorem stating that the number of allocation schemes is 36 -/
theorem allocation_schemes_eq_36 : allocation_schemes = 36 := by sorry

end allocation_schemes_eq_36_l1950_195007


namespace problem_solution_l1950_195052

theorem problem_solution (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end problem_solution_l1950_195052


namespace fuel_calculation_correct_l1950_195085

/-- Calculates the total fuel needed for a plane trip given the specified conditions -/
def total_fuel_needed (base_fuel_per_mile : ℕ) (fuel_increase_per_person : ℕ) 
  (fuel_increase_per_bag : ℕ) (passengers : ℕ) (crew : ℕ) (bags_per_person : ℕ) 
  (trip_distance : ℕ) : ℕ :=
  let total_people := passengers + crew
  let total_bags := total_people * bags_per_person
  let fuel_per_mile := base_fuel_per_mile + 
    total_people * fuel_increase_per_person + 
    total_bags * fuel_increase_per_bag
  fuel_per_mile * trip_distance

/-- Theorem stating that the total fuel needed for the given conditions is 106,000 gallons -/
theorem fuel_calculation_correct : 
  total_fuel_needed 20 3 2 30 5 2 400 = 106000 := by
  sorry

end fuel_calculation_correct_l1950_195085


namespace bill_calculation_correct_l1950_195091

/-- Calculates the final bill amount after late charges and fees --/
def finalBillAmount (originalBill : ℝ) (firstLateChargeRate : ℝ) (secondLateChargeRate : ℝ) (flatFee : ℝ) : ℝ :=
  ((originalBill * (1 + firstLateChargeRate)) * (1 + secondLateChargeRate)) + flatFee

/-- Proves that the final bill amount is correct given the specified conditions --/
theorem bill_calculation_correct :
  finalBillAmount 500 0.01 0.02 5 = 520.1 := by
  sorry

#eval finalBillAmount 500 0.01 0.02 5

end bill_calculation_correct_l1950_195091


namespace harry_terry_calculation_l1950_195062

theorem harry_terry_calculation (x : ℤ) : 
  let H := 12 - (3 + 7) + x
  let T := 12 - 3 + 7 + x
  H - T + x = -14 + x := by
sorry

end harry_terry_calculation_l1950_195062


namespace garden_fencing_l1950_195010

/-- Calculates the perimeter of a rectangular garden with given length and width ratio --/
theorem garden_fencing (length : ℝ) (h1 : length = 80) : 
  2 * (length + length / 2) = 240 := by
  sorry


end garden_fencing_l1950_195010


namespace sqrt_difference_equality_l1950_195004

theorem sqrt_difference_equality : 
  Real.sqrt (9/2) - Real.sqrt (8/5) = (15 * Real.sqrt 2 - 4 * Real.sqrt 10) / 10 := by sorry

end sqrt_difference_equality_l1950_195004


namespace sum_of_squares_and_square_of_sum_l1950_195048

theorem sum_of_squares_and_square_of_sum : (4 + 8)^2 + (4^2 + 8^2) = 224 := by
  sorry

end sum_of_squares_and_square_of_sum_l1950_195048


namespace magnitude_of_z_l1950_195006

/-- The complex number i such that i² = -1 -/
noncomputable def i : ℂ := Complex.I

/-- The given complex number z -/
noncomputable def z : ℂ := (1 - i) / (1 + i) + 4 - 2*i

/-- Theorem stating that the magnitude of z is 5 -/
theorem magnitude_of_z : Complex.abs z = 5 := by sorry

end magnitude_of_z_l1950_195006


namespace sum_of_roots_l1950_195098

theorem sum_of_roots (a b : ℝ) (ha : a * (a - 4) = 12) (hb : b * (b - 4) = 12) (hab : a ≠ b) : a + b = 4 := by
  sorry

end sum_of_roots_l1950_195098


namespace p_18_equals_negative_one_l1950_195046

/-- A quadratic function with specific properties -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ := fun x ↦ d * x^2 + e * x + f

/-- Theorem: For a quadratic function with given properties, p(18) = -1 -/
theorem p_18_equals_negative_one
  (d e f : ℝ)
  (p : ℝ → ℝ)
  (h_quad : p = QuadraticFunction d e f)
  (h_sym : p 6 = p 12)
  (h_max : IsLocalMax p 10)
  (h_p0 : p 0 = -1) :
  p 18 = -1 := by
  sorry

end p_18_equals_negative_one_l1950_195046


namespace intersection_of_P_and_Q_l1950_195054

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 5}

theorem intersection_of_P_and_Q : P ∩ Q = {(4, -1)} := by
  sorry

end intersection_of_P_and_Q_l1950_195054


namespace g_value_at_2_l1950_195009

def g (x : ℝ) : ℝ := 3 * x^8 - 4 * x^4 + 2 * x^2 - 6

theorem g_value_at_2 (h : g (-2) = 10) : g 2 = 1402 := by
  sorry

end g_value_at_2_l1950_195009


namespace inequality_solution_implies_a_range_l1950_195095

theorem inequality_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ((a + 1) * x > 2 * a + 2) ↔ (x < 2)) →
  a < -1 := by
sorry

end inequality_solution_implies_a_range_l1950_195095


namespace picture_frame_problem_l1950_195011

/-- Represents a rectangular picture frame -/
structure Frame where
  outer_length : ℝ
  outer_width : ℝ
  wood_width : ℝ

/-- Calculates the area of the frame material -/
def frame_area (f : Frame) : ℝ :=
  f.outer_length * f.outer_width - (f.outer_length - 2 * f.wood_width) * (f.outer_width - 2 * f.wood_width)

/-- Calculates the sum of the lengths of the four interior edges -/
def interior_perimeter (f : Frame) : ℝ :=
  2 * (f.outer_length - 2 * f.wood_width) + 2 * (f.outer_width - 2 * f.wood_width)

theorem picture_frame_problem :
  ∀ f : Frame,
    f.wood_width = 2 →
    f.outer_length = 7 →
    frame_area f = 34 →
    interior_perimeter f = 9 := by
  sorry

end picture_frame_problem_l1950_195011


namespace series_sum_8_eq_43690_l1950_195081

def series_sum : ℕ → ℕ 
  | 0 => 2
  | n + 1 => 2 * (1 + 4 * series_sum n)

theorem series_sum_8_eq_43690 : series_sum 7 = 43690 := by
  sorry

end series_sum_8_eq_43690_l1950_195081


namespace problem_solution_l1950_195028

theorem problem_solution :
  (∀ x : ℝ, -3 * x * (2 * x^2 - x + 4) = -6 * x^3 + 3 * x^2 - 12 * x) ∧
  (∀ a b : ℝ, (2 * a - b) * (2 * a + b) = 4 * a^2 - b^2) := by
  sorry

end problem_solution_l1950_195028


namespace correct_junior_teachers_in_sample_l1950_195032

/-- Represents the number of teachers in each category -/
structure TeacherPopulation where
  total : Nat
  junior : Nat

/-- Represents a stratified sample -/
structure StratifiedSample where
  populationSize : Nat
  sampleSize : Nat
  juniorInPopulation : Nat
  juniorInSample : Nat

/-- Calculates the number of junior teachers in a stratified sample -/
def calculateJuniorTeachersInSample (pop : TeacherPopulation) (sampleSize : Nat) : Nat :=
  (pop.junior * sampleSize) / pop.total

/-- Theorem stating that the calculated number of junior teachers in the sample is correct -/
theorem correct_junior_teachers_in_sample (pop : TeacherPopulation) (sample : StratifiedSample) 
    (h1 : pop.total = 200)
    (h2 : pop.junior = 80)
    (h3 : sample.populationSize = pop.total)
    (h4 : sample.sampleSize = 50)
    (h5 : sample.juniorInPopulation = pop.junior)
    (h6 : sample.juniorInSample = calculateJuniorTeachersInSample pop sample.sampleSize) :
  sample.juniorInSample = 20 := by
  sorry

#check correct_junior_teachers_in_sample

end correct_junior_teachers_in_sample_l1950_195032


namespace max_value_of_expression_l1950_195080

open Real

theorem max_value_of_expression (t : ℝ) :
  (∃ (max : ℝ), ∀ (t : ℝ), (3^t - 4*t^2)*t / 9^t ≤ max) ∧
  (∃ (t_max : ℝ), (3^t_max - 4*t_max^2)*t_max / 9^t_max = sqrt 3 / 9) :=
by sorry

end max_value_of_expression_l1950_195080


namespace jorges_clay_rich_soil_fraction_l1950_195041

theorem jorges_clay_rich_soil_fraction (total_land : ℝ) (good_soil_yield : ℝ) 
  (clay_rich_soil_yield : ℝ) (total_yield : ℝ) 
  (h1 : total_land = 60)
  (h2 : good_soil_yield = 400)
  (h3 : clay_rich_soil_yield = good_soil_yield / 2)
  (h4 : total_yield = 20000) :
  let clay_rich_fraction := (total_land * good_soil_yield - total_yield) / 
    (total_land * (good_soil_yield - clay_rich_soil_yield))
  clay_rich_fraction = 1/3 := by
  sorry

end jorges_clay_rich_soil_fraction_l1950_195041


namespace gcd_of_198_and_308_l1950_195069

theorem gcd_of_198_and_308 : Nat.gcd 198 308 = 22 := by
  sorry

end gcd_of_198_and_308_l1950_195069


namespace triangle_properties_l1950_195073

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that under certain conditions, we can determine the values of A, b, and c. -/
theorem triangle_properties (t : Triangle) 
  (h1 : ∃ (k : ℝ), k * (Real.sqrt 3 * t.a) = t.c ∧ k * (1 + Real.cos t.A) = Real.sin t.C) 
  (h2 : 3 * t.b * t.c = 16 - t.a^2)
  (h3 : t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry

end triangle_properties_l1950_195073


namespace zeros_of_g_l1950_195088

/-- Given a linear function f(x) = ax + b with a zero at x = 1,
    prove that the zeros of g(x) = bx^2 - ax are 0 and -1 -/
theorem zeros_of_g (a b : ℝ) (h : a + b = 0) (ha : a ≠ 0) :
  let f := λ x : ℝ => a * x + b
  let g := λ x : ℝ => b * x^2 - a * x
  (∀ x : ℝ, g x = 0 ↔ x = 0 ∨ x = -1) :=
by sorry

end zeros_of_g_l1950_195088


namespace inequality_system_solution_set_l1950_195099

theorem inequality_system_solution_set :
  ∀ a : ℝ, (2 * a - 3 < 0 ∧ 1 - a < 0) ↔ (1 < a ∧ a < 3/2) := by
  sorry

end inequality_system_solution_set_l1950_195099


namespace students_in_both_events_l1950_195044

theorem students_in_both_events (total : ℕ) (volleyball : ℕ) (track_field : ℕ) (none : ℕ) :
  total = 45 →
  volleyball = 12 →
  track_field = 20 →
  none = 19 →
  volleyball + track_field - (total - none) = 6 :=
by
  sorry

end students_in_both_events_l1950_195044


namespace donut_selection_problem_l1950_195058

theorem donut_selection_problem :
  let n : ℕ := 6  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 84 := by
  sorry

end donut_selection_problem_l1950_195058


namespace total_volume_is_114_l1950_195049

/-- The volume of a cube with side length s -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- The number of Carl's cubes -/
def carl_cubes : ℕ := 4

/-- The side length of Carl's cubes -/
def carl_side_length : ℝ := 3

/-- The number of Kate's cubes -/
def kate_cubes : ℕ := 6

/-- The side length of Kate's cubes -/
def kate_side_length : ℝ := 1

/-- The total volume of all cubes -/
def total_volume : ℝ :=
  (carl_cubes : ℝ) * cube_volume carl_side_length +
  (kate_cubes : ℝ) * cube_volume kate_side_length

theorem total_volume_is_114 : total_volume = 114 := by
  sorry

end total_volume_is_114_l1950_195049
