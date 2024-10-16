import Mathlib

namespace NUMINAMATH_CALUDE_cubic_function_uniqueness_l477_47710

/-- Given a cubic function f(x) = ax³ + bx² + cx, prove that if it has critical points at x = ±1
    and its derivative at x = 0 is -3, then f(x) = x³ - 3x. -/
theorem cubic_function_uniqueness (a b c : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^3 + b * x^2 + c * x
  let f' := fun (x : ℝ) ↦ 3 * a * x^2 + 2 * b * x + c
  (f' 1 = 0 ∧ f' (-1) = 0 ∧ f' 0 = -3) →
  (∀ x, f x = x^3 - 3*x) := by
sorry

end NUMINAMATH_CALUDE_cubic_function_uniqueness_l477_47710


namespace NUMINAMATH_CALUDE_base_sum_theorem_l477_47773

/-- Represents a fraction in a given base --/
structure FractionInBase where
  numerator : ℕ
  denominator : ℕ
  base : ℕ

/-- Converts a repeating decimal to a fraction in a given base --/
def repeatingDecimalToFraction (digits : List ℕ) (base : ℕ) : FractionInBase :=
  sorry

/-- The sum of the bases R₁ and R₂ --/
def sumOfBases : ℕ := 14

theorem base_sum_theorem (R₁ R₂ : ℕ) :
  let F₁_R₁ := repeatingDecimalToFraction [4, 5] R₁
  let F₂_R₁ := repeatingDecimalToFraction [5, 4] R₁
  let F₁_R₂ := repeatingDecimalToFraction [3, 2] R₂
  let F₂_R₂ := repeatingDecimalToFraction [2, 3] R₂
  F₁_R₁.numerator * F₁_R₂.denominator = F₁_R₂.numerator * F₁_R₁.denominator ∧
  F₂_R₁.numerator * F₂_R₂.denominator = F₂_R₂.numerator * F₂_R₁.denominator →
  R₁ + R₂ = sumOfBases := by
  sorry

end NUMINAMATH_CALUDE_base_sum_theorem_l477_47773


namespace NUMINAMATH_CALUDE_kannon_apples_last_night_l477_47767

/-- The number of apples Kannon had last night -/
def apples_last_night : ℕ := sorry

/-- The number of bananas Kannon had last night -/
def bananas_last_night : ℕ := 1

/-- The number of oranges Kannon had last night -/
def oranges_last_night : ℕ := 4

/-- The number of apples Kannon will have today -/
def apples_today : ℕ := apples_last_night + 4

/-- The number of bananas Kannon will have today -/
def bananas_today : ℕ := 10 * bananas_last_night

/-- The number of oranges Kannon will have today -/
def oranges_today : ℕ := 2 * apples_today

theorem kannon_apples_last_night :
  apples_last_night = 3 ∧
  (apples_last_night + bananas_last_night + oranges_last_night +
   apples_today + bananas_today + oranges_today = 39) :=
by sorry

end NUMINAMATH_CALUDE_kannon_apples_last_night_l477_47767


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_pyramid_l477_47780

/-- The surface area of a sphere given a right square pyramid inscribed in it -/
theorem sphere_surface_area_from_pyramid (h V : ℝ) (h_pos : h > 0) (V_pos : V > 0) :
  let s := Real.sqrt (3 * V / h)
  let r := Real.sqrt ((s^2 + 2 * h^2) / 4)
  h = 4 → V = 16 → 4 * Real.pi * r^2 = 24 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_pyramid_l477_47780


namespace NUMINAMATH_CALUDE_modulus_of_z_l477_47785

-- Define the complex number z
variable (z : ℂ)

-- Define the condition z(i-1) = 4
def condition (z : ℂ) : Prop := z * (Complex.I - 1) = 4

-- Theorem statement
theorem modulus_of_z (h : condition z) : Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l477_47785


namespace NUMINAMATH_CALUDE_set_inclusion_implies_m_bound_l477_47705

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x ≤ 6}
def B : Set ℝ := {x | x^2 - 4*x < 0}
def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- State the theorem
theorem set_inclusion_implies_m_bound (m : ℝ) :
  C m ⊆ (C m ∩ B) → m ≤ 5/2 := by
  sorry


end NUMINAMATH_CALUDE_set_inclusion_implies_m_bound_l477_47705


namespace NUMINAMATH_CALUDE_log_power_sum_l477_47759

theorem log_power_sum (a b : ℝ) (ha : a = Real.log 25) (hb : b = Real.log 36) :
  (5 : ℝ) ^ (a / b) + (6 : ℝ) ^ (b / a) = 11 := by sorry

end NUMINAMATH_CALUDE_log_power_sum_l477_47759


namespace NUMINAMATH_CALUDE_min_value_of_a_l477_47778

theorem min_value_of_a (a : ℝ) (h_a : a > 0) : 
  (∀ (x₁ : ℝ) (x₂ : ℝ), x₁ > 0 → 1 ≤ x₂ → x₂ ≤ Real.exp 1 → 
    x₁ + a^2 / x₁ ≥ x₂ - Real.log x₂) → 
  a ≥ Real.sqrt (Real.exp 1 - 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l477_47778


namespace NUMINAMATH_CALUDE_sqrt_one_third_equality_l477_47748

theorem sqrt_one_third_equality : 3 * Real.sqrt (1/3) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_one_third_equality_l477_47748


namespace NUMINAMATH_CALUDE_odd_quadratic_implies_zero_coefficient_l477_47779

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The quadratic function f(x) = ax^2 + 2x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x

theorem odd_quadratic_implies_zero_coefficient (a : ℝ) :
  IsOdd (f a) → a = 0 := by
  sorry


end NUMINAMATH_CALUDE_odd_quadratic_implies_zero_coefficient_l477_47779


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l477_47727

-- Define the quadratic function
def f (x : ℝ) := -x^2 - 3*x + 4

-- Define the solution set
def S : Set ℝ := {x | f x > 0}

-- Theorem statement
theorem solution_set_is_open_interval : S = Set.Ioo (-4 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l477_47727


namespace NUMINAMATH_CALUDE_square_difference_inapplicable_l477_47730

/-- The square difference formula cannot be applied to (2x+3y)(-3y-2x) -/
theorem square_difference_inapplicable (x y : ℝ) :
  ¬ ∃ (a b : ℝ), (∃ (c₁ c₂ c₃ c₄ : ℝ), a = c₁ * x + c₂ * y ∧ b = c₃ * x + c₄ * y) ∧
    (2 * x + 3 * y) * (-3 * y - 2 * x) = (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_inapplicable_l477_47730


namespace NUMINAMATH_CALUDE_middle_school_students_l477_47723

theorem middle_school_students (band_percentage : ℚ) (band_students : ℕ) 
  (h1 : band_percentage = 1/5) 
  (h2 : band_students = 168) : 
  ∃ total_students : ℕ, 
    (band_percentage * total_students = band_students) ∧ 
    total_students = 840 := by
  sorry

end NUMINAMATH_CALUDE_middle_school_students_l477_47723


namespace NUMINAMATH_CALUDE_molar_ratio_h2_ch4_l477_47700

/-- Represents the heat of reaction for H₂ combustion in kJ/mol -/
def heat_h2 : ℝ := -571.6

/-- Represents the heat of reaction for CH₄ combustion in kJ/mol -/
def heat_ch4 : ℝ := -890

/-- Represents the volume of the gas mixture in liters -/
def mixture_volume : ℝ := 112

/-- Represents the molar volume of gas under standard conditions in L/mol -/
def molar_volume : ℝ := 22.4

/-- Represents the total heat released in kJ -/
def total_heat_released : ℝ := 3695

/-- Theorem stating that the molar ratio of H₂ to CH₄ in the original mixture is 1:3 -/
theorem molar_ratio_h2_ch4 :
  ∃ (x y : ℝ),
    x + y = mixture_volume / molar_volume ∧
    (heat_h2 / 2) * x + heat_ch4 * y = total_heat_released ∧
    x / y = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_molar_ratio_h2_ch4_l477_47700


namespace NUMINAMATH_CALUDE_divisors_of_square_l477_47790

-- Define a function that counts the number of divisors
def count_divisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem divisors_of_square (n : ℕ) :
  count_divisors n = 5 → count_divisors (n^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_square_l477_47790


namespace NUMINAMATH_CALUDE_sqrt_12_similar_to_sqrt_3_l477_47733

/-- Two quadratic radicals are similar if they have the same radicand when simplified. -/
def similar_radicals (a b : ℝ) : Prop :=
  ∃ (k₁ k₂ : ℝ), k₁ > 0 ∧ k₂ > 0 ∧ a = k₁^2 * b

/-- √12 is of the same type as √3 -/
theorem sqrt_12_similar_to_sqrt_3 : similar_radicals 12 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_similar_to_sqrt_3_l477_47733


namespace NUMINAMATH_CALUDE_number_difference_l477_47755

theorem number_difference (n : ℕ) : 
  n / 12 = 25 ∧ n % 12 = 11 → n - 25 = 286 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l477_47755


namespace NUMINAMATH_CALUDE_sum_is_real_product_is_real_sum_and_product_are_real_complex_numbers_satisfying_conditions_l477_47777

-- Define complex numbers
def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

-- Theorem for condition (a)
theorem sum_is_real (a b c : ℝ) :
  ∃ (z₁ z₂ : ℂ), (z₁ + z₂).im = 0 :=
sorry

-- Theorem for condition (b)
theorem product_is_real (a b k : ℝ) :
  ∃ (z₁ z₂ : ℂ), (z₁ * z₂).im = 0 :=
sorry

-- Theorem for condition (c)
theorem sum_and_product_are_real (a b : ℝ) :
  ∃ (z₁ z₂ : ℂ), (z₁ + z₂).im = 0 ∧ (z₁ * z₂).im = 0 :=
sorry

-- Main theorem combining all conditions
theorem complex_numbers_satisfying_conditions :
  (∃ (z₁ z₂ : ℂ), (z₁ + z₂).im = 0) ∧
  (∃ (z₁ z₂ : ℂ), (z₁ * z₂).im = 0) ∧
  (∃ (z₁ z₂ : ℂ), (z₁ + z₂).im = 0 ∧ (z₁ * z₂).im = 0) :=
sorry

end NUMINAMATH_CALUDE_sum_is_real_product_is_real_sum_and_product_are_real_complex_numbers_satisfying_conditions_l477_47777


namespace NUMINAMATH_CALUDE_max_guaranteed_guesses_l477_47717

/- Define the deck of cards -/
def deck_size : Nat := 52

/- Define a function to represent the alternating arrangement of cards -/
def alternating_arrangement (i : Nat) : Bool :=
  i % 2 = 0

/- Define a function to represent the state of the deck after cutting and riffling -/
def riffled_deck (n : Nat) (i : Nat) : Bool :=
  if i < n then alternating_arrangement i else alternating_arrangement (i - n)

/- Theorem: The maximum number of guaranteed correct guesses is 26 -/
theorem max_guaranteed_guesses :
  ∀ n : Nat, n ≤ deck_size →
  ∃ strategy : Nat → Bool,
    (∀ i : Nat, i < deck_size → strategy i = riffled_deck n i) →
    (∃ correct_guesses : Nat, correct_guesses = deck_size / 2 ∧
      ∀ k : Nat, k < correct_guesses → strategy k = riffled_deck n k) :=
by sorry

end NUMINAMATH_CALUDE_max_guaranteed_guesses_l477_47717


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l477_47787

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + a * x - 4

-- Define the property that the solution set is ℝ
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x < 0

-- Theorem statement
theorem quadratic_inequality_range :
  ∀ a : ℝ, solution_set_is_reals a → -16 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l477_47787


namespace NUMINAMATH_CALUDE_prob_same_heads_value_l477_47757

/-- Probability of getting heads for a fair coin -/
def fair_coin_prob : ℚ := 1/2

/-- Probability of getting heads for the biased coin -/
def biased_coin_prob : ℚ := 5/8

/-- Number of coins each person flips -/
def num_coins : ℕ := 3

/-- Number of fair coins each person flips -/
def num_fair_coins : ℕ := 2

/-- Probability of getting k heads when flipping n coins with probability p of heads -/
def prob_k_heads (n : ℕ) (k : ℕ) (p : ℚ) : ℚ := sorry

/-- Probability of Alice and Bob getting the same number of heads -/
def prob_same_heads : ℚ := sorry

theorem prob_same_heads_value : prob_same_heads = 81/256 := by sorry

end NUMINAMATH_CALUDE_prob_same_heads_value_l477_47757


namespace NUMINAMATH_CALUDE_rightmost_four_digits_of_7_to_2023_l477_47766

theorem rightmost_four_digits_of_7_to_2023 :
  7^2023 ≡ 1359 [ZMOD 10000] := by
  sorry

end NUMINAMATH_CALUDE_rightmost_four_digits_of_7_to_2023_l477_47766


namespace NUMINAMATH_CALUDE_james_truck_trip_distance_l477_47760

/-- 
Given:
- James gets paid $0.50 per mile to drive a truck.
- Gas costs $4.00 per gallon.
- The truck gets 20 miles per gallon.
- James made a profit of $180 from a trip.

Prove: The length of the trip was 600 miles.
-/
theorem james_truck_trip_distance : 
  let pay_rate : ℝ := 0.50  -- pay rate in dollars per mile
  let gas_price : ℝ := 4.00  -- gas price in dollars per gallon
  let fuel_efficiency : ℝ := 20  -- miles per gallon
  let profit : ℝ := 180  -- profit in dollars
  ∃ distance : ℝ, 
    distance * pay_rate - (distance / fuel_efficiency) * gas_price = profit ∧ 
    distance = 600 := by
  sorry

end NUMINAMATH_CALUDE_james_truck_trip_distance_l477_47760


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l477_47796

/-- The Euler family children's ages after one year -/
def euler_family_ages : List ℕ := [9, 9, 9, 9, 11, 13, 13]

/-- The number of children in the Euler family -/
def num_children : ℕ := 7

/-- The sum of the Euler family children's ages after one year -/
def sum_ages : ℕ := euler_family_ages.sum

theorem euler_family_mean_age :
  (sum_ages : ℚ) / num_children = 73 / 7 := by sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l477_47796


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l477_47704

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, p < k → is_prime p → ¬(n % p = 0)

theorem smallest_non_prime_non_square_no_small_factors : 
  ∃! n : ℕ, n > 0 ∧ 
    ¬(is_prime n) ∧ 
    ¬(is_perfect_square n) ∧ 
    has_no_prime_factor_less_than n 60 ∧
    ∀ m : ℕ, m > 0 → 
      ¬(is_prime m) → 
      ¬(is_perfect_square m) → 
      has_no_prime_factor_less_than m 60 → 
      n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l477_47704


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l477_47712

theorem smallest_k_no_real_roots : 
  ∀ k : ℤ, (∀ x : ℝ, 2*x*(k*x-4)-x^2+7 ≠ 0) → k ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l477_47712


namespace NUMINAMATH_CALUDE_smallest_reciprocal_sum_l477_47758

/-- A structure representing a quadratic equation with roots satisfying specific conditions -/
structure QuadraticEquation where
  s : ℝ
  p : ℝ
  r₁ : ℝ
  r₂ : ℝ
  root_sum_equal : ∀ n : ℕ, 1 ≤ n → n ≤ 1004 → r₁^n + r₂^n = r₁ + r₂
  vieta1 : r₁ + r₂ = s
  vieta2 : r₁ * r₂ = p

/-- The theorem stating the smallest possible value of 1/r₁¹⁰⁰⁵ + 1/r₂¹⁰⁰⁵ -/
theorem smallest_reciprocal_sum (q : QuadraticEquation) :
  (∀ q' : QuadraticEquation, 1 / q'.r₁^1005 + 1 / q'.r₂^1005 ≥ 1 / q.r₁^1005 + 1 / q.r₂^1005) →
  1 / q.r₁^1005 + 1 / q.r₂^1005 = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_reciprocal_sum_l477_47758


namespace NUMINAMATH_CALUDE_science_price_relation_spending_condition_min_literature_books_proof_l477_47731

-- Define the prices of books
def literature_price : ℚ := 5
def science_price : ℚ := 15 / 2

-- Define the condition that science book price is half higher than literature book price
theorem science_price_relation : science_price = literature_price * (3/2) := by sorry

-- Define the spending condition
theorem spending_condition (lit_count science_count : ℕ) : 
  lit_count * literature_price + science_count * science_price = 15 ∧ lit_count = science_count + 1 := by sorry

-- Define the budget condition
def total_books : ℕ := 10
def total_budget : ℚ := 60

-- Define the function to calculate the minimum number of literature books
def min_literature_books : ℕ := 6

-- Theorem to prove the minimum number of literature books
theorem min_literature_books_proof :
  ∀ m : ℕ, m * literature_price + (total_books - m) * science_price ≤ total_budget → m ≥ min_literature_books := by sorry

end NUMINAMATH_CALUDE_science_price_relation_spending_condition_min_literature_books_proof_l477_47731


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l477_47782

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > -1/2 * p.1 ∧ p.2 > 3 * p.1 + 6}

-- Define the quadrants
def quadrantI : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}
def quadrantII : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}
def quadrantIII : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 < 0}
def quadrantIV : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0}

-- Theorem statement
theorem points_in_quadrants_I_and_II : 
  S ⊆ quadrantI ∪ quadrantII ∧ 
  S ∩ quadrantIII = ∅ ∧ 
  S ∩ quadrantIV = ∅ := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l477_47782


namespace NUMINAMATH_CALUDE_vector_computation_l477_47735

theorem vector_computation : 
  5 • !![3, -9] - 4 • !![2, -6] + !![1, 3] = !![8, -18] := by
  sorry

end NUMINAMATH_CALUDE_vector_computation_l477_47735


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l477_47762

def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
def B (a : ℝ) : Set ℝ := {a - 3, a - 1, a + 1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-2} → a = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l477_47762


namespace NUMINAMATH_CALUDE_decagon_diagonals_l477_47751

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l477_47751


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l477_47738

theorem larger_number_of_pair (x y : ℝ) (h_diff : x - y = 7) (h_sum : x + y = 47) :
  max x y = 27 := by
sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l477_47738


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l477_47737

theorem reciprocal_sum_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  x₁ ≠ 0 → 
  x₂ ≠ 0 → 
  1/x₁ + 1/x₂ = 3/2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l477_47737


namespace NUMINAMATH_CALUDE_triangle_isosceles_if_2cosB_sinA_eq_sinC_l477_47721

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = π

-- Define the property of being isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

-- State the theorem
theorem triangle_isosceles_if_2cosB_sinA_eq_sinC (t : Triangle) :
  2 * Real.cos t.B * Real.sin t.A = Real.sin t.C → isIsosceles t :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_isosceles_if_2cosB_sinA_eq_sinC_l477_47721


namespace NUMINAMATH_CALUDE_distribute_and_simplify_l477_47706

theorem distribute_and_simplify (a b : ℝ) : 3*a*(2*a - b) = 6*a^2 - 3*a*b := by
  sorry

end NUMINAMATH_CALUDE_distribute_and_simplify_l477_47706


namespace NUMINAMATH_CALUDE_profit_starts_fourth_year_option_two_more_profitable_l477_47722

def initial_investment : ℕ := 81
def annual_rental_income : ℕ := 30
def first_year_renovation : ℕ := 1
def yearly_renovation_increase : ℕ := 2

def total_renovation_cost (n : ℕ) : ℕ := n^2

def total_income (n : ℕ) : ℕ := annual_rental_income * n

def profit (n : ℕ) : ℤ := (total_income n : ℤ) - (initial_investment : ℤ) - (total_renovation_cost n : ℤ)

def average_profit (n : ℕ) : ℚ := (profit n : ℚ) / n

theorem profit_starts_fourth_year :
  ∀ n : ℕ, n < 4 → profit n ≤ 0 ∧ profit 4 > 0 := by sorry

theorem option_two_more_profitable :
  profit 15 + 10 < profit 9 + 50 := by sorry

#eval profit 4
#eval profit 15 + 10
#eval profit 9 + 50

end NUMINAMATH_CALUDE_profit_starts_fourth_year_option_two_more_profitable_l477_47722


namespace NUMINAMATH_CALUDE_assignment_ways_for_given_tasks_l477_47791

/-- The number of ways to assign people to tasks --/
def assignment_ways (total_people : ℕ) (selected_people : ℕ) (task_a_people : ℕ) : ℕ :=
  Nat.choose total_people selected_people *
  Nat.choose selected_people task_a_people *
  Nat.factorial (selected_people - task_a_people)

/-- Theorem stating the number of ways to assign 4 people out of 10 to the given tasks --/
theorem assignment_ways_for_given_tasks :
  assignment_ways 10 4 2 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_assignment_ways_for_given_tasks_l477_47791


namespace NUMINAMATH_CALUDE_probability_quarter_or_dime_l477_47769

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Nickel
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def total_value : Coin → ℕ
  | Coin.Quarter => 1500
  | Coin.Nickel => 1500
  | Coin.Dime => 1000
  | Coin.Penny => 500

/-- The number of coins of each type in the jar -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := (coin_count Coin.Quarter) + (coin_count Coin.Nickel) + (coin_count Coin.Dime) + (coin_count Coin.Penny)

/-- The probability of randomly choosing either a quarter or a dime from the jar -/
theorem probability_quarter_or_dime : 
  (coin_count Coin.Quarter + coin_count Coin.Dime : ℚ) / total_coins = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_quarter_or_dime_l477_47769


namespace NUMINAMATH_CALUDE_employment_percentage_l477_47734

theorem employment_percentage (total_population : ℝ) 
  (employed_males_percentage : ℝ) (employed_females_ratio : ℝ) :
  employed_males_percentage = 36 →
  employed_females_ratio = 50 →
  (employed_males_percentage / employed_females_ratio) * 100 = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_employment_percentage_l477_47734


namespace NUMINAMATH_CALUDE_infinite_series_sum_l477_47753

/-- The sum of the infinite series Σ(n=1 to ∞) [(3n - 2) / (n(n + 1)(n + 3))] is equal to 11/12 -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l477_47753


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_l477_47799

/-- Represents the daily rental cost of a canoe -/
def canoe_cost : ℕ := 15

/-- Represents the daily rental cost of a kayak -/
def kayak_cost : ℕ := 18

/-- Represents the total daily revenue -/
def total_revenue : ℕ := 405

/-- Theorem stating that the difference between canoes and kayaks rented is 5 -/
theorem canoe_kayak_difference :
  ∀ (c k : ℕ),
  (c : ℚ) / k = 3 / 2 →
  canoe_cost * c + kayak_cost * k = total_revenue →
  c - k = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_difference_l477_47799


namespace NUMINAMATH_CALUDE_sin_shift_l477_47788

theorem sin_shift (x : ℝ) : 
  Real.sin (4 * x - π / 3) = Real.sin (4 * (x - π / 12)) := by sorry

end NUMINAMATH_CALUDE_sin_shift_l477_47788


namespace NUMINAMATH_CALUDE_largest_common_divisor_l477_47772

theorem largest_common_divisor : ∃ (d : ℕ), 
  d = 98 ∧ 
  (∀ (k : ℕ), k ∈ {28, 49, 98} ∪ {n : ℕ | n > 49 ∧ n % 7 = 0 ∧ n % 2 = 1} ∪ {n : ℕ | n > 98 ∧ n % 7 = 0 ∧ n % 2 = 0} → 
    (13511 % d = 13903 % d ∧ 13511 % d = 14589 % d) → k ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l477_47772


namespace NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l477_47742

theorem cos_pi_4_plus_alpha (α : ℝ) (h : Real.sin (π / 4 - α) = -2 / 5) :
  Real.cos (π / 4 + α) = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l477_47742


namespace NUMINAMATH_CALUDE_car_speed_proof_l477_47736

/-- Proves that the speed of a car is 60 miles per hour given specific conditions -/
theorem car_speed_proof (
  fuel_efficiency : Real
) (
  tank_capacity : Real
) (
  travel_time : Real
) (
  fuel_used_ratio : Real
) (
  h1 : fuel_efficiency = 30 -- miles per gallon
) (
  h2 : tank_capacity = 12 -- gallons
) (
  h3 : travel_time = 5 -- hours
) (
  h4 : fuel_used_ratio = 0.8333333333333334 -- ratio of full tank
) : Real := by
  sorry

end NUMINAMATH_CALUDE_car_speed_proof_l477_47736


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l477_47765

/-- A point P with coordinates (m, 4+2m) is in the third quadrant if and only if m < -2 -/
theorem point_in_third_quadrant (m : ℝ) :
  (m < 0 ∧ 4 + 2 * m < 0) ↔ m < -2 := by
sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l477_47765


namespace NUMINAMATH_CALUDE_constant_value_proof_l477_47752

theorem constant_value_proof (x y a : ℝ) 
  (h1 : (a * x + 4 * y) / (x - 2 * y) = 25)
  (h2 : x / (2 * y) = 3 / 2) : 
  a = 7 := by
sorry

end NUMINAMATH_CALUDE_constant_value_proof_l477_47752


namespace NUMINAMATH_CALUDE_f_sum_negative_l477_47771

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x, f (4 - x) = -f x)
variable (h2 : ∀ x, x < 2 → StrictMonoDecreasing f (Set.Iio 2))

-- Define the conditions on x₁ and x₂
variable (x₁ x₂ : ℝ)
variable (h3 : x₁ + x₂ > 4)
variable (h4 : (x₁ - 2) * (x₂ - 2) < 0)

-- State the theorem
theorem f_sum_negative : f x₁ + f x₂ < 0 :=
sorry

end NUMINAMATH_CALUDE_f_sum_negative_l477_47771


namespace NUMINAMATH_CALUDE_abs_x_plus_one_gt_three_l477_47702

theorem abs_x_plus_one_gt_three (x : ℝ) :
  |x + 1| > 3 ↔ x < -4 ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_abs_x_plus_one_gt_three_l477_47702


namespace NUMINAMATH_CALUDE_factory_workers_count_l477_47794

/-- Represents the number of factory workers in company J -/
def factory_workers : ℕ := sorry

/-- Represents the number of office workers in company J -/
def office_workers : ℕ := 30

/-- Represents the total monthly payroll for factory workers in dollars -/
def factory_payroll : ℕ := 30000

/-- Represents the total monthly payroll for office workers in dollars -/
def office_payroll : ℕ := 75000

/-- Represents the difference in average monthly salary between office and factory workers in dollars -/
def salary_difference : ℕ := 500

theorem factory_workers_count :
  factory_workers = 15 ∧
  factory_workers * (office_payroll / office_workers - salary_difference) = factory_payroll :=
by sorry

end NUMINAMATH_CALUDE_factory_workers_count_l477_47794


namespace NUMINAMATH_CALUDE_prove_initial_number_l477_47775

def initial_number : ℕ := 7899665
def result : ℕ := 7899593
def factor1 : ℕ := 12
def factor2 : ℕ := 3
def factor3 : ℕ := 2

theorem prove_initial_number :
  initial_number - (factor1 * factor2 * factor3) = result :=
by sorry

end NUMINAMATH_CALUDE_prove_initial_number_l477_47775


namespace NUMINAMATH_CALUDE_horner_method_v3_l477_47795

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 + 3x^3 - 2x^2 + x - 1 -/
def f (x : ℝ) : ℝ := 2*x^5 + 3*x^3 - 2*x^2 + x - 1

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [2, 0, 3, -2, 1, -1]

theorem horner_method_v3 :
  let v₃ := horner (f_coeffs.take 4) 2
  v₃ = 20 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l477_47795


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l477_47719

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 + 2*x = 0 ↔ x = 0 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l477_47719


namespace NUMINAMATH_CALUDE_negative_cube_squared_l477_47715

theorem negative_cube_squared (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l477_47715


namespace NUMINAMATH_CALUDE_fraction_equality_l477_47792

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 15) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l477_47792


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l477_47764

theorem complex_fraction_equality : 
  (1 - Complex.I * Real.sqrt 3) / ((Real.sqrt 3 + Complex.I) ^ 2) = 
  -(1/4) - (Real.sqrt 3 / 4) * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l477_47764


namespace NUMINAMATH_CALUDE_stella_stamps_count_l477_47732

/-- The number of stamps in Stella's album -/
def total_stamps : ℕ :=
  let total_pages : ℕ := 50
  let first_pages : ℕ := 10
  let stamps_per_row : ℕ := 30
  let rows_per_first_page : ℕ := 5
  let stamps_per_remaining_page : ℕ := 50
  
  let stamps_in_first_pages : ℕ := first_pages * rows_per_first_page * stamps_per_row
  let remaining_pages : ℕ := total_pages - first_pages
  let stamps_in_remaining_pages : ℕ := remaining_pages * stamps_per_remaining_page
  
  stamps_in_first_pages + stamps_in_remaining_pages

/-- Theorem stating that the total number of stamps in Stella's album is 3500 -/
theorem stella_stamps_count : total_stamps = 3500 := by
  sorry

end NUMINAMATH_CALUDE_stella_stamps_count_l477_47732


namespace NUMINAMATH_CALUDE_trig_expression_equals_zero_l477_47741

theorem trig_expression_equals_zero :
  (Real.sin (15 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (19 * π / 180) * Real.cos (11 * π / 180) + 
   Real.cos (161 * π / 180) * Real.cos (101 * π / 180)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_zero_l477_47741


namespace NUMINAMATH_CALUDE_cos_arcsin_plus_arccos_l477_47745

theorem cos_arcsin_plus_arccos : 
  Real.cos (Real.arcsin (3/5) + Real.arccos (-5/13)) = -56/65 := by sorry

end NUMINAMATH_CALUDE_cos_arcsin_plus_arccos_l477_47745


namespace NUMINAMATH_CALUDE_four_integers_product_2002_sum_less_40_l477_47749

theorem four_integers_product_2002_sum_less_40 :
  ∀ a b c d : ℕ+,
  a * b * c * d = 2002 →
  (a : ℕ) + b + c + d < 40 →
  ((a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨
   (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
   (a = 1 ∧ b = 11 ∧ c = 14 ∧ d = 13) ∨
   (a = 1 ∧ b = 11 ∧ c = 13 ∧ d = 14) ∨
   (a = 2 ∧ b = 11 ∧ c = 7 ∧ d = 13) ∨
   (a = 2 ∧ b = 11 ∧ c = 13 ∧ d = 7) ∨
   (a = 2 ∧ b = 13 ∧ c = 7 ∧ d = 11) ∨
   (a = 2 ∧ b = 13 ∧ c = 11 ∧ d = 7) ∨
   (a = 7 ∧ b = 2 ∧ c = 11 ∧ d = 13) ∨
   (a = 7 ∧ b = 2 ∧ c = 13 ∧ d = 11) ∨
   (a = 7 ∧ b = 11 ∧ c = 2 ∧ d = 13) ∨
   (a = 7 ∧ b = 11 ∧ c = 13 ∧ d = 2) ∨
   (a = 7 ∧ b = 13 ∧ c = 2 ∧ d = 11) ∧
   (a = 7 ∧ b = 13 ∧ c = 11 ∧ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_four_integers_product_2002_sum_less_40_l477_47749


namespace NUMINAMATH_CALUDE_triangle_inequality_violation_l477_47720

theorem triangle_inequality_violation
  (a b c d e : ℝ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (sum_equality : a^2 + b^2 + c^2 + d^2 + e^2 = a*b + a*c + a*d + a*e + b*c + b*d + b*e + c*d + c*e + d*e) :
  ∃ (x y z : ℝ), (x = a ∧ y = b ∧ z = c) ∨ (x = a ∧ y = b ∧ z = d) ∨ (x = a ∧ y = b ∧ z = e) ∨
                 (x = a ∧ y = c ∧ z = d) ∨ (x = a ∧ y = c ∧ z = e) ∨ (x = a ∧ y = d ∧ z = e) ∨
                 (x = b ∧ y = c ∧ z = d) ∨ (x = b ∧ y = c ∧ z = e) ∨ (x = b ∧ y = d ∧ z = e) ∨
                 (x = c ∧ y = d ∧ z = e) ∧
                 (x + y ≤ z ∨ y + z ≤ x ∨ z + x ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_violation_l477_47720


namespace NUMINAMATH_CALUDE_eight_n_even_when_n_seven_l477_47725

theorem eight_n_even_when_n_seven :
  ∃ k : ℤ, 8 * 7 = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_eight_n_even_when_n_seven_l477_47725


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l477_47768

theorem square_sum_equals_one (a b : ℝ) 
  (h1 : a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = 1)
  (h2 : 0 ≤ a ∧ a ≤ 1)
  (h3 : 0 ≤ b ∧ b ≤ 1) : 
  a^2 + b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l477_47768


namespace NUMINAMATH_CALUDE_simplification_and_evaluation_l477_47716

theorem simplification_and_evaluation (a : ℤ) 
  (h1 : -2 ≤ a ∧ a ≤ 2) 
  (h2 : a + 1 ≠ 0) 
  (h3 : a - 2 ≠ 0) : 
  (1 - 3 / (a + 1)) / ((a^2 - 4*a + 4) / (a + 1)) = 1 / (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_simplification_and_evaluation_l477_47716


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l477_47740

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {y | ∃ x, y = x^2 + 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l477_47740


namespace NUMINAMATH_CALUDE_complex_parts_of_z_l477_47797

def z : ℂ := 3 * Complex.I * (Complex.I + 1)

theorem complex_parts_of_z :
  Complex.re z = -3 ∧ Complex.im z = 3 := by sorry

end NUMINAMATH_CALUDE_complex_parts_of_z_l477_47797


namespace NUMINAMATH_CALUDE_garden_table_bench_ratio_l477_47713

theorem garden_table_bench_ratio :
  ∀ (table_cost bench_cost : ℕ),
    bench_cost = 150 →
    table_cost + bench_cost = 450 →
    ∃ (k : ℕ), table_cost = k * bench_cost →
    (table_cost : ℚ) / (bench_cost : ℚ) = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_table_bench_ratio_l477_47713


namespace NUMINAMATH_CALUDE_green_balls_count_l477_47763

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  white = 10 →
  yellow = 10 →
  red = 47 →
  purple = 3 →
  prob_not_red_purple = 1/2 →
  ∃ green : ℕ, green = 30 ∧ total = white + yellow + red + purple + green :=
by sorry

end NUMINAMATH_CALUDE_green_balls_count_l477_47763


namespace NUMINAMATH_CALUDE_max_min_difference_d_l477_47708

theorem max_min_difference_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 20) :
  ∃ (d_min d_max : ℝ), 
    (∀ d', (∃ a' b' c', a' + b' + c' + d' = 3 ∧ a'^2 + b'^2 + c'^2 + d'^2 = 20) → d_min ≤ d' ∧ d' ≤ d_max) ∧
    d_max - d_min = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_d_l477_47708


namespace NUMINAMATH_CALUDE_biker_problem_l477_47744

/-- Two bikers on a circular path problem -/
theorem biker_problem (t1 t2 meet_time : ℕ) : 
  t1 = 12 →  -- First rider completes a round in 12 minutes
  meet_time = 36 →  -- They meet again at the starting point after 36 minutes
  meet_time % t1 = 0 →  -- First rider completes whole number of rounds
  meet_time % t2 = 0 →  -- Second rider completes whole number of rounds
  t2 > t1 →  -- Second rider is slower than the first
  t2 = 36  -- Second rider takes 36 minutes to complete a round
  := by sorry

end NUMINAMATH_CALUDE_biker_problem_l477_47744


namespace NUMINAMATH_CALUDE_largest_five_digit_product_120_l477_47781

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digit_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_five_digit_product_120 :
  ∃ N : ℕ, is_five_digit N ∧ 
           digit_product N = 120 ∧
           (∀ m : ℕ, is_five_digit m → digit_product m = 120 → m ≤ N) ∧
           digit_sum N = 18 := by
sorry

end NUMINAMATH_CALUDE_largest_five_digit_product_120_l477_47781


namespace NUMINAMATH_CALUDE_inequality_theorem_l477_47729

theorem inequality_theorem (a b : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → x^2 + 1 ≥ a*x + b ∧ a*x + b ≥ (3/2)*x^(2/3)) →
  ((2 - Real.sqrt 2) / 4 ≤ b ∧ b ≤ (2 + Real.sqrt 2) / 4) ∧
  (1 / Real.sqrt (2*b) ≤ a ∧ a ≤ 2 * Real.sqrt (1 - b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l477_47729


namespace NUMINAMATH_CALUDE_negation_equivalence_l477_47718

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Ioo (-1 : ℝ) 3, x^2 - 1 ≤ 2*x) ↔
  (∀ x ∈ Set.Ioo (-1 : ℝ) 3, x^2 - 1 > 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l477_47718


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l477_47739

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 - 2 * x - 8 < 0} = {x : ℝ | -4/3 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l477_47739


namespace NUMINAMATH_CALUDE_quadratic_factorization_l477_47784

theorem quadratic_factorization (c d : ℕ) (hc : c > d) :
  (∀ x, x^2 - 18*x + 72 = (x - c) * (x - d)) →
  4*d - c = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l477_47784


namespace NUMINAMATH_CALUDE_cyclic_reciprocal_product_l477_47776

theorem cyclic_reciprocal_product (x y z : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (h_cyclic : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x) : 
  x^2 * y^2 * z^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_cyclic_reciprocal_product_l477_47776


namespace NUMINAMATH_CALUDE_max_m2_plus_n2_l477_47701

theorem max_m2_plus_n2 : ∃ (m n : ℕ),
  1 ≤ m ∧ m ≤ 2005 ∧
  1 ≤ n ∧ n ≤ 2005 ∧
  (n^2 + 2*m*n - 2*m^2)^2 = 1 ∧
  m^2 + n^2 = 702036 ∧
  ∀ (m' n' : ℕ),
    1 ≤ m' ∧ m' ≤ 2005 ∧
    1 ≤ n' ∧ n' ≤ 2005 ∧
    (n'^2 + 2*m'*n' - 2*m'^2)^2 = 1 →
    m'^2 + n'^2 ≤ 702036 :=
by sorry

end NUMINAMATH_CALUDE_max_m2_plus_n2_l477_47701


namespace NUMINAMATH_CALUDE_insect_eggs_base_conversion_l477_47786

theorem insect_eggs_base_conversion : 
  (2 * 7^2 + 3 * 7^1 + 5 * 7^0 : ℕ) = 124 := by sorry

end NUMINAMATH_CALUDE_insect_eggs_base_conversion_l477_47786


namespace NUMINAMATH_CALUDE_sum_of_square_reciprocals_l477_47747

theorem sum_of_square_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a + b = 3 * a * b + 1) : 
  1 / a^2 + 1 / b^2 = (4 * a * b + 10) / (a^2 * b^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_reciprocals_l477_47747


namespace NUMINAMATH_CALUDE_girls_in_class_l477_47728

theorem girls_in_class (total : ℕ) (boys : ℕ) (prob : ℚ) : 
  total = 25 →
  (boys.choose 2 : ℚ) / (total.choose 2 : ℚ) = prob →
  prob = 3/25 →
  total - boys = 16 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_class_l477_47728


namespace NUMINAMATH_CALUDE_choose_three_with_min_diff_3_from_14_l477_47711

/-- The number of ways to choose three integers from 1 to 14 with minimum difference 3 -/
def choose_three_with_min_diff (n : ℕ) (k : ℕ) (min_diff : ℕ) : ℕ :=
  Nat.choose (n - k * (min_diff - 1) + k - 1) (k - 1)

/-- Theorem: There are 120 ways to choose three integers from 1 to 14 
    such that the absolute difference between any two is at least 3 -/
theorem choose_three_with_min_diff_3_from_14 : 
  choose_three_with_min_diff 14 3 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_with_min_diff_3_from_14_l477_47711


namespace NUMINAMATH_CALUDE_cosine_half_angle_in_interval_l477_47756

theorem cosine_half_angle_in_interval (θ m : Real) 
  (h1 : 5/2 * Real.pi < θ) 
  (h2 : θ < 3 * Real.pi) 
  (h3 : |Real.cos θ| = m) : 
  Real.cos (θ/2) = -Real.sqrt ((1 - m)/2) := by
  sorry

end NUMINAMATH_CALUDE_cosine_half_angle_in_interval_l477_47756


namespace NUMINAMATH_CALUDE_number_problem_l477_47789

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 30 → (40/100 : ℝ) * N = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l477_47789


namespace NUMINAMATH_CALUDE_circle_radius_sqrt29_l477_47754

/-- A circle with center on the x-axis passing through two given points has radius √29 -/
theorem circle_radius_sqrt29 (x : ℝ) :
  (x - 1)^2 + 5^2 = (x - 2)^2 + 4^2 →
  Real.sqrt ((x - 1)^2 + 5^2) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_sqrt29_l477_47754


namespace NUMINAMATH_CALUDE_division_problem_l477_47746

theorem division_problem (N : ℕ) : 
  (N / 7 = 12 ∧ N % 7 = 4) → (N / 3 = 29) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l477_47746


namespace NUMINAMATH_CALUDE_speed_limit_excess_l477_47798

/-- Proves that a journey of 150 miles completed in 2 hours exceeds a 60 mph speed limit by 15 mph -/
theorem speed_limit_excess (distance : ℝ) (time : ℝ) (speed_limit : ℝ) : 
  distance = 150 ∧ time = 2 ∧ speed_limit = 60 →
  distance / time - speed_limit = 15 := by
sorry

end NUMINAMATH_CALUDE_speed_limit_excess_l477_47798


namespace NUMINAMATH_CALUDE_special_function_sqrt_5753_l477_47724

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = x * f y + y * f x) ∧
  (∀ x y : ℝ, f (x + y) = f (x * 1993) + f (y * 1993))

/-- The main theorem -/
theorem special_function_sqrt_5753 (f : ℝ → ℝ) (h : special_function f) :
  f (Real.sqrt 5753) = 0 := by sorry

end NUMINAMATH_CALUDE_special_function_sqrt_5753_l477_47724


namespace NUMINAMATH_CALUDE_least_valid_number_l477_47714

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d : ℕ) (m : ℕ), 
    n = 10 * m + d ∧ 
    1 ≤ d ∧ d ≤ 9 ∧ 
    m = n / 25

theorem least_valid_number : 
  (∀ k < 3125, ¬(is_valid_number k)) ∧ is_valid_number 3125 :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l477_47714


namespace NUMINAMATH_CALUDE_min_product_with_98_zeros_l477_47783

/-- The number of trailing zeros in a positive integer -/
def trailingZeros (n : ℕ+) : ℕ := sorry

/-- The concatenation of two positive integers -/
def concat (a b : ℕ+) : ℕ+ := sorry

/-- The statement of the problem -/
theorem min_product_with_98_zeros :
  ∃ (m n : ℕ+),
    (∀ (x y : ℕ+), trailingZeros (x^x.val * y^y.val) = 98 → m.val * n.val ≤ x.val * y.val) ∧
    trailingZeros (m^m.val * n^n.val) = 98 ∧
    trailingZeros (concat (concat m m) (concat n n)) = 98 ∧
    m.val * n.val = 7350 := by
  sorry

end NUMINAMATH_CALUDE_min_product_with_98_zeros_l477_47783


namespace NUMINAMATH_CALUDE_pyramid_prism_sum_max_pyramid_prism_sum_l477_47726

/-- A solid formed by attaching a square pyramid to one rectangular face of a rectangular prism -/
structure PyramidPrism where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_new_faces : ℕ
  pyramid_new_edges : ℕ
  pyramid_new_vertex : ℕ

/-- The total number of exterior faces, edges, and vertices of the combined solid -/
def total_elements (pp : PyramidPrism) : ℕ :=
  (pp.prism_faces - 1 + pp.pyramid_new_faces) +
  (pp.prism_edges + pp.pyramid_new_edges) +
  (pp.prism_vertices + pp.pyramid_new_vertex)

/-- Theorem stating that the sum of faces, edges, and vertices of the combined solid is 34 -/
theorem pyramid_prism_sum (pp : PyramidPrism) 
  (h1 : pp.prism_faces = 6)
  (h2 : pp.prism_edges = 12)
  (h3 : pp.prism_vertices = 8)
  (h4 : pp.pyramid_new_faces = 4)
  (h5 : pp.pyramid_new_edges = 4)
  (h6 : pp.pyramid_new_vertex = 1) :
  total_elements pp = 34 := by
  sorry

/-- The maximum value of the sum of faces, edges, and vertices is 34 -/
theorem max_pyramid_prism_sum :
  ∀ pp : PyramidPrism, total_elements pp ≤ 34 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_prism_sum_max_pyramid_prism_sum_l477_47726


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l477_47750

theorem last_digit_of_sum (n : ℕ) : 
  (54^2019 + 28^2021) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_sum_l477_47750


namespace NUMINAMATH_CALUDE_diane_age_is_16_l477_47793

/-- Represents the current ages of Diane, Alex, and Allison -/
structure Ages where
  diane : ℕ
  alex : ℕ
  allison : ℕ

/-- Conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.alex + ages.allison = 47 ∧
  ages.alex + (30 - ages.diane) = 60 ∧
  ages.allison + (30 - ages.diane) = 15

/-- Theorem stating that Diane's current age is 16 -/
theorem diane_age_is_16 :
  ∃ (ages : Ages), problem_conditions ages ∧ ages.diane = 16 :=
sorry

end NUMINAMATH_CALUDE_diane_age_is_16_l477_47793


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l477_47703

theorem min_value_2x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 1) :
  2 * x + y ≥ 2 * Real.sqrt 2 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x * y = 1 ∧ 2 * x + y = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l477_47703


namespace NUMINAMATH_CALUDE_dinitrogen_monoxide_molecular_weight_l477_47761

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound consisting of two nitrogen atoms and one oxygen atom -/
def dinitrogen_monoxide_weight : ℝ := 2 * nitrogen_weight + oxygen_weight

/-- Theorem stating that the molecular weight of Dinitrogen monoxide is 44.02 amu -/
theorem dinitrogen_monoxide_molecular_weight :
  dinitrogen_monoxide_weight = 44.02 := by sorry

end NUMINAMATH_CALUDE_dinitrogen_monoxide_molecular_weight_l477_47761


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l477_47743

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The probability of a single die showing an even number -/
def prob_even : ℚ := 1/2

/-- The probability of a single die showing an odd number -/
def prob_odd : ℚ := 1/2

/-- The number of dice that need to show even (and odd) for the event to occur -/
def target_even : ℕ := num_dice / 2

-- The theorem statement
theorem equal_even_odd_probability : 
  (Nat.choose num_dice target_even : ℚ) * prob_even ^ num_dice = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l477_47743


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_165_6_l477_47770

theorem percentage_of_360_equals_165_6 : ∃ (p : ℚ), p * 360 = 165.6 ∧ p * 100 = 46 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_165_6_l477_47770


namespace NUMINAMATH_CALUDE_rectangular_park_area_l477_47774

theorem rectangular_park_area (width : ℝ) (length : ℝ) (perimeter : ℝ) :
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_rectangular_park_area_l477_47774


namespace NUMINAMATH_CALUDE_curve_and_tangent_lines_l477_47709

-- Define the curve C
def C (x y : ℝ) : Prop :=
  (x^2 + y^2) / ((x - 3)^2 + y^2) = 1/4

-- Define point N
def N : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem curve_and_tangent_lines :
  (∀ x y : ℝ, C x y ↔ x^2 + y^2 + 2*x - 3 = 0) ∧
  (∀ x y : ℝ, (C x y ∧ (x - N.1)^2 + (y - N.2)^2 = 0) →
    (x = 1 ∨ 5*x - 12*y + 31 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_curve_and_tangent_lines_l477_47709


namespace NUMINAMATH_CALUDE_dvd_player_cost_l477_47707

theorem dvd_player_cost (d m : ℝ) 
  (h1 : d / m = 9 / 2)
  (h2 : d = m + 63) :
  d = 81 := by
sorry

end NUMINAMATH_CALUDE_dvd_player_cost_l477_47707
