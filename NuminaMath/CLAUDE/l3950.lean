import Mathlib

namespace NUMINAMATH_CALUDE_total_egg_collection_l3950_395045

/-- The number of dozen eggs collected by each person -/
structure EggCollection where
  benjamin : ℚ
  carla : ℚ
  trisha : ℚ
  david : ℚ
  emily : ℚ

/-- The conditions of the egg collection problem -/
def eggCollectionConditions (e : EggCollection) : Prop :=
  e.benjamin = 6 ∧
  e.carla = 3 * e.benjamin ∧
  e.trisha = e.benjamin - 4 ∧
  e.david = 2 * e.trisha ∧
  e.david = e.carla / 2 ∧
  e.emily = 3/4 * e.david ∧
  e.emily = e.trisha + e.trisha / 2

/-- The theorem stating that the total number of dozen eggs collected is 33 -/
theorem total_egg_collection (e : EggCollection) 
  (h : eggCollectionConditions e) : 
  e.benjamin + e.carla + e.trisha + e.david + e.emily = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_egg_collection_l3950_395045


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l3950_395009

-- Define the vectors
def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, 1)

-- Define the parallel condition
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem vector_parallel_condition (m : ℝ) :
  are_parallel (a.1 + 2 * (b m).1, a.2 + 2 * (b m).2) (2 * a.1 - (b m).1, 2 * a.2 - (b m).2) →
  m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l3950_395009


namespace NUMINAMATH_CALUDE_limit_fraction_sequence_l3950_395008

theorem limit_fraction_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((n : ℝ) + 20) / (3 * n + 13) - 1/3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_fraction_sequence_l3950_395008


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3950_395030

def is_valid (n : ℕ) : Prop :=
  n % 9 = 0 ∧
  n % 2 = 1 ∧
  n % 3 = 1 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 6 = 1

theorem smallest_valid_number : 
  is_valid 361 ∧ ∀ m : ℕ, m < 361 → ¬is_valid m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3950_395030


namespace NUMINAMATH_CALUDE_first_number_in_ratio_l3950_395003

theorem first_number_in_ratio (A B : ℕ+) : 
  (A : ℚ) / (B : ℚ) = 8 / 9 →
  Nat.lcm A B = 432 →
  A = 48 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_ratio_l3950_395003


namespace NUMINAMATH_CALUDE_siblings_age_problem_l3950_395044

theorem siblings_age_problem (b s : ℕ) : 
  (b - 3 = 7 * (s - 3)) →
  (b - 2 = 4 * (s - 2)) →
  (b - 1 = 3 * (s - 1)) →
  (b = (5 * s) / 2) →
  (b = 10 ∧ s = 4) :=
by sorry

end NUMINAMATH_CALUDE_siblings_age_problem_l3950_395044


namespace NUMINAMATH_CALUDE_spinner_probability_l3950_395065

/-- Given a spinner with three regions A, B, and C, where the probability of
    stopping on A is 1/2 and on B is 1/5, prove that the probability of
    stopping on C is 3/10. -/
theorem spinner_probability (p_A p_B p_C : ℚ) : 
  p_A = 1/2 → p_B = 1/5 → p_A + p_B + p_C = 1 → p_C = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3950_395065


namespace NUMINAMATH_CALUDE_q_function_equality_l3950_395022

/-- Given a function q(x) that satisfies the equation
    q(x) + (2x^6 + 5x^4 + 10x) = (8x^4 + 35x^3 + 40x^2 + 2),
    prove that q(x) = -2x^6 + 3x^4 + 35x^3 + 40x^2 - 10x + 2 -/
theorem q_function_equality (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 5 * x^4 + 10 * x) = (8 * x^4 + 35 * x^3 + 40 * x^2 + 2)) →
  (∀ x, q x = -2 * x^6 + 3 * x^4 + 35 * x^3 + 40 * x^2 - 10 * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_q_function_equality_l3950_395022


namespace NUMINAMATH_CALUDE_total_mail_delivered_l3950_395056

-- Define the number of junk mail pieces
def junk_mail : ℕ := 6

-- Define the number of magazines
def magazines : ℕ := 5

-- Theorem to prove
theorem total_mail_delivered : junk_mail + magazines = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_mail_delivered_l3950_395056


namespace NUMINAMATH_CALUDE_frustum_height_calc_l3950_395046

/-- Represents a pyramid cut by a plane parallel to its base -/
structure CutPyramid where
  height : ℝ              -- Total height of the pyramid
  cut_height : ℝ          -- Height of the cut part
  area_ratio : ℝ          -- Ratio of upper to lower base areas

/-- The height of the frustum in a cut pyramid -/
def frustum_height (p : CutPyramid) : ℝ := p.height - p.cut_height

/-- Theorem stating the height of the frustum given specific conditions -/
theorem frustum_height_calc (p : CutPyramid) 
  (h1 : p.area_ratio = 1 / 4)
  (h2 : p.cut_height = 3) :
  frustum_height p = 3 := by
  sorry

#check frustum_height_calc

end NUMINAMATH_CALUDE_frustum_height_calc_l3950_395046


namespace NUMINAMATH_CALUDE_amy_flash_drive_files_l3950_395028

/-- The number of files on Amy's flash drive -/
def total_files (music_files video_files picture_files : Float) : Float :=
  music_files + video_files + picture_files

/-- Theorem stating the total number of files on Amy's flash drive -/
theorem amy_flash_drive_files : 
  total_files 4.0 21.0 23.0 = 48.0 := by
  sorry

end NUMINAMATH_CALUDE_amy_flash_drive_files_l3950_395028


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3950_395010

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  (X^4 : Polynomial ℝ) + X^3 + 1 = (X^2 - 2*X + 3) * q + r ∧
  r.degree < (X^2 - 2*X + 3).degree ∧
  r = -3*X - 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3950_395010


namespace NUMINAMATH_CALUDE_inverse_exists_iff_a_eq_zero_l3950_395000

-- Define the function f(x) = (x - a)|x|
def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * abs x

-- State the theorem
theorem inverse_exists_iff_a_eq_zero (a : ℝ) :
  Function.Injective (f a) ↔ a = 0 := by sorry

end NUMINAMATH_CALUDE_inverse_exists_iff_a_eq_zero_l3950_395000


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l3950_395089

theorem smallest_number_divisible (n : ℕ) : n = 1013 ↔ 
  (∀ m : ℕ, m < n → 
    ¬(∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
      m - 5 = 12 * k₁ ∧
      m - 5 = 16 * k₂ ∧
      m - 5 = 18 * k₃ ∧
      m - 5 = 21 * k₄ ∧
      m - 5 = 28 * k₅)) ∧
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    n - 5 = 12 * k₁ ∧
    n - 5 = 16 * k₂ ∧
    n - 5 = 18 * k₃ ∧
    n - 5 = 21 * k₄ ∧
    n - 5 = 28 * k₅) :=
by sorry


end NUMINAMATH_CALUDE_smallest_number_divisible_l3950_395089


namespace NUMINAMATH_CALUDE_quadratic_roots_equal_roots_condition_l3950_395021

-- Part 1: Roots of x^2 - 2x - 8 = 0
theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 8
  ∃ (x₁ x₂ : ℝ), x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
sorry

-- Part 2: Value of a when roots of x^2 - ax + 1 = 0 are equal
theorem equal_roots_condition :
  let g : ℝ → ℝ → ℝ := λ a x => x^2 - a*x + 1
  ∀ a : ℝ, (∃! x : ℝ, g a x = 0) → (a = 2 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_equal_roots_condition_l3950_395021


namespace NUMINAMATH_CALUDE_expr_is_symmetrical_l3950_395018

/-- Definition of a symmetrical expression -/
def is_symmetrical (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b, f a b = f b a

/-- The expression we want to prove is symmetrical -/
def expr (a b : ℝ) : ℝ := 4*a^2 + 4*b^2 - 4*a*b

/-- Theorem: The expression 4a^2 + 4b^2 - 4ab is symmetrical -/
theorem expr_is_symmetrical : is_symmetrical expr := by sorry

end NUMINAMATH_CALUDE_expr_is_symmetrical_l3950_395018


namespace NUMINAMATH_CALUDE_special_sequence_property_l3950_395077

/-- A sequence of positive real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  (∀ m n : ℕ, m > 0 → n > 0 → (m + n : ℝ) * a (m + n) ≤ a m + a n) ∧
  (∀ i : ℕ, i > 0 → a i > 0)

/-- The main theorem to be proved -/
theorem special_sequence_property (a : ℕ → ℝ) (h : SpecialSequence a) : 
  1 / a 200 > 4 * 10^7 := by sorry

end NUMINAMATH_CALUDE_special_sequence_property_l3950_395077


namespace NUMINAMATH_CALUDE_faucet_filling_time_l3950_395084

/-- Given that five faucets fill a 150-gallon tub in 9 minutes,
    prove that ten faucets will fill a 75-gallon tub in 135 seconds. -/
theorem faucet_filling_time 
  (initial_faucets : ℕ) 
  (initial_volume : ℝ) 
  (initial_time : ℝ) 
  (target_faucets : ℕ) 
  (target_volume : ℝ) 
  (h1 : initial_faucets = 5) 
  (h2 : initial_volume = 150) 
  (h3 : initial_time = 9) 
  (h4 : target_faucets = 10) 
  (h5 : target_volume = 75) : 
  (target_volume / target_faucets) * (initial_time / (initial_volume / initial_faucets)) * 60 = 135 := by
  sorry

#check faucet_filling_time

end NUMINAMATH_CALUDE_faucet_filling_time_l3950_395084


namespace NUMINAMATH_CALUDE_xyz_value_l3950_395055

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 168) (h2 : y * (z + x) = 180) (h3 : z * (x + y) = 192) :
  x * y * z = 842 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3950_395055


namespace NUMINAMATH_CALUDE_function_composition_result_l3950_395086

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_result (a b : ℝ) :
  (∀ x, h a b x = x + 9) → a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_result_l3950_395086


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3950_395001

theorem quadratic_equation_solutions (x₁ x₂ : ℝ) :
  (x₁ = -1 ∧ x₂ = 3 ∧ x₁^2 - 2*x₁ - 3 = 0 ∧ x₂^2 - 2*x₂ - 3 = 0) →
  ∃ y₁ y₂ : ℝ, y₁ = 1 ∧ y₂ = -1 ∧ (2*y₁ + 1)^2 - 2*(2*y₁ + 1) - 3 = 0 ∧ (2*y₂ + 1)^2 - 2*(2*y₂ + 1) - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3950_395001


namespace NUMINAMATH_CALUDE_infinite_set_A_l3950_395059

/-- Given a function f: ℝ → ℝ satisfying the inequality f²(x) ≤ 2x² f(x/2) for all x,
    and a non-empty set A = {a ∈ ℝ | f(a) > a²}, prove that A is infinite. -/
theorem infinite_set_A (f : ℝ → ℝ) 
    (h1 : ∀ x : ℝ, f x ^ 2 ≤ 2 * x ^ 2 * f (x / 2))
    (A : Set ℝ)
    (h2 : A = {a : ℝ | f a > a ^ 2})
    (h3 : Set.Nonempty A) :
  Set.Infinite A :=
sorry

end NUMINAMATH_CALUDE_infinite_set_A_l3950_395059


namespace NUMINAMATH_CALUDE_min_sum_given_product_l3950_395057

theorem min_sum_given_product (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 8) :
  ∃ (min : ℝ), min = 6 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x * y * z = 8 → x + y + z ≥ min :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l3950_395057


namespace NUMINAMATH_CALUDE_equation_solution_l3950_395023

theorem equation_solution : ∃ x : ℝ, x > 0 ∧ 6 * x^(1/3) - 3 * (x / x^(2/3)) = -1 + 2 * x^(1/3) + 4 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3950_395023


namespace NUMINAMATH_CALUDE_binomial_prob_half_l3950_395054

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: If X ~ B(n, p) with E(X) = 6 and D(X) = 3, then p = 1/2 -/
theorem binomial_prob_half (X : BinomialRV) 
  (h_exp : expectation X = 6)
  (h_var : variance X = 3) : 
  X.p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_prob_half_l3950_395054


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l3950_395087

theorem quadratic_solution_implies_sum (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 2 * a + 4 * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l3950_395087


namespace NUMINAMATH_CALUDE_hank_lawn_mowing_earnings_l3950_395025

/-- Proves that Hank made $50 from mowing lawns given the specified conditions -/
theorem hank_lawn_mowing_earnings :
  let carwash_earnings : ℝ := 100
  let carwash_donation_rate : ℝ := 0.9
  let bake_sale_earnings : ℝ := 80
  let bake_sale_donation_rate : ℝ := 0.75
  let lawn_mowing_donation_rate : ℝ := 1
  let total_donation : ℝ := 200
  let lawn_mowing_earnings : ℝ := 
    total_donation - 
    (carwash_earnings * carwash_donation_rate + 
     bake_sale_earnings * bake_sale_donation_rate)
  lawn_mowing_earnings = 50 := by sorry

end NUMINAMATH_CALUDE_hank_lawn_mowing_earnings_l3950_395025


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l3950_395090

theorem concentric_circles_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * Real.pi * r₁) = 48 / 360 * (2 * Real.pi * r₂)) →
  (r₁ / r₂ = 4 / 5 ∧ (r₁^2 / r₂^2 = 16 / 25)) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l3950_395090


namespace NUMINAMATH_CALUDE_mary_has_ten_marbles_l3950_395016

/-- The number of blue marbles Dan has -/
def dans_marbles : ℕ := 5

/-- The ratio of Mary's marbles to Dan's marbles -/
def mary_to_dan_ratio : ℕ := 2

/-- The number of blue marbles Mary has -/
def marys_marbles : ℕ := mary_to_dan_ratio * dans_marbles

theorem mary_has_ten_marbles : marys_marbles = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_ten_marbles_l3950_395016


namespace NUMINAMATH_CALUDE_plate_cup_cost_l3950_395085

/-- Given that 100 plates and 200 cups cost $7.50, prove that 20 plates and 40 cups cost $1.50 -/
theorem plate_cup_cost (plate_rate cup_rate : ℚ) : 
  100 * plate_rate + 200 * cup_rate = (7.5 : ℚ) → 
  20 * plate_rate + 40 * cup_rate = (1.5 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_plate_cup_cost_l3950_395085


namespace NUMINAMATH_CALUDE_horner_v₃_value_l3950_395033

def f (x : ℝ) : ℝ := 7 * x^5 + 5 * x^4 + 3 * x^3 + x^2 + x + 2

def horner_method (x : ℝ) : ℝ × ℝ × ℝ × ℝ := 
  let v₀ : ℝ := 7
  let v₁ : ℝ := v₀ * x + 5
  let v₂ : ℝ := v₁ * x + 3
  let v₃ : ℝ := v₂ * x + 1
  (v₀, v₁, v₂, v₃)

theorem horner_v₃_value :
  (horner_method 2).2.2.2 = 83 := by sorry

end NUMINAMATH_CALUDE_horner_v₃_value_l3950_395033


namespace NUMINAMATH_CALUDE_consecutive_digits_divisible_by_11_l3950_395078

/-- Given four consecutive digits x, x+1, x+2, x+3, the number formed by
    interchanging the first two digits of (1000x + 100(x+1) + 10(x+2) + (x+3))
    is divisible by 11 for any integer x. -/
theorem consecutive_digits_divisible_by_11 (x : ℤ) :
  ∃ k : ℤ, (1000 * (x + 1) + 100 * x + 10 * (x + 2) + (x + 3)) = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_digits_divisible_by_11_l3950_395078


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3950_395099

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 20)
  (square_diff : x^2 - y^2 = 200)
  (diff_eq : x - y = 10) : x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3950_395099


namespace NUMINAMATH_CALUDE_event_guests_l3950_395052

theorem event_guests (men : ℕ) (women : ℕ) (children : ℕ) : 
  men = 40 →
  women = men / 2 →
  children + 10 = 30 →
  men + women + children = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_event_guests_l3950_395052


namespace NUMINAMATH_CALUDE_union_when_a_eq_2_union_eq_B_iff_l3950_395005

-- Define set A
def A : Set ℝ := {x | (x - 1) / (x - 2) ≤ 1 / 2}

-- Define set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2) * x + 2 * a ≤ 0}

-- Theorem for part (1)
theorem union_when_a_eq_2 : A ∪ B 2 = {x | 0 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for part (2)
theorem union_eq_B_iff (a : ℝ) : A ∪ B a = B a ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_union_when_a_eq_2_union_eq_B_iff_l3950_395005


namespace NUMINAMATH_CALUDE_tank_capacity_l3950_395047

theorem tank_capacity (fill_time_A fill_time_B drain_rate_C combined_fill_time : ℝ) 
  (h1 : fill_time_A = 12)
  (h2 : fill_time_B = 20)
  (h3 : drain_rate_C = 45)
  (h4 : combined_fill_time = 15) :
  ∃ V : ℝ, V = 675 ∧ 
    (V / fill_time_A + V / fill_time_B - drain_rate_C = V / combined_fill_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l3950_395047


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l3950_395051

theorem minimum_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 - b + 4 ≤ 0) :
  ∃ (min : ℝ), min = 14/5 ∧ ∀ x, x = (2*a + 3*b)/(a + b) → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l3950_395051


namespace NUMINAMATH_CALUDE_corner_sum_l3950_395019

/-- Represents a 10x10 array filled with integers from 1 to 100 -/
def CheckerBoard := Fin 10 → Fin 10 → Fin 100

/-- The checkerboard is filled in sequence -/
def is_sequential (board : CheckerBoard) : Prop :=
  ∀ i j, board i j = i.val * 10 + j.val + 1

/-- The sum of the numbers in the four corners of the checkerboard is 202 -/
theorem corner_sum (board : CheckerBoard) (h : is_sequential board) :
  (board 0 0).val + (board 0 9).val + (board 9 0).val + (board 9 9).val = 202 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_l3950_395019


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l3950_395063

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (|x| - 3) / (x + 3) = 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l3950_395063


namespace NUMINAMATH_CALUDE_third_term_is_five_l3950_395038

/-- An arithmetic sequence where the sum of the first and fifth terms is 10 -/
def ArithmeticSequence (a : ℝ) (d : ℝ) : Prop :=
  a + (a + 4 * d) = 10

/-- The third term of the arithmetic sequence -/
def ThirdTerm (a : ℝ) (d : ℝ) : ℝ :=
  a + 2 * d

theorem third_term_is_five {a d : ℝ} (h : ArithmeticSequence a d) :
  ThirdTerm a d = 5 := by
  sorry


end NUMINAMATH_CALUDE_third_term_is_five_l3950_395038


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l3950_395053

/-- The function f(x) = 2bx - 3b + 1 has a zero point in (-1, 1) iff b ∈ (1/5, 1) -/
theorem zero_point_in_interval (b : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, 2 * b * x - 3 * b + 1 = 0) ↔ b ∈ Set.Ioo (1/5 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_zero_point_in_interval_l3950_395053


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l3950_395080

theorem x_range_for_quadratic_inequality :
  ∀ x : ℝ, (∀ a : ℝ, a ∈ Set.Icc (-1) 1 → x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔
    x ∈ Set.Ioi 3 ∪ Set.Iio 1 := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l3950_395080


namespace NUMINAMATH_CALUDE_rational_sum_zero_l3950_395004

theorem rational_sum_zero (x₁ x₂ x₃ x₄ : ℚ) 
  (h₁ : x₁ = x₂ + x₃ + x₄)
  (h₂ : x₂ = x₁ + x₃ + x₄)
  (h₃ : x₃ = x₁ + x₂ + x₄)
  (h₄ : x₄ = x₁ + x₂ + x₃) :
  x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_zero_l3950_395004


namespace NUMINAMATH_CALUDE_ethan_reading_pages_l3950_395020

/-- Represents the number of pages Ethan read on Saturday morning -/
def saturday_morning_pages : ℕ := sorry

/-- Represents the total number of pages in the book -/
def total_pages : ℕ := 360

/-- Represents the number of pages Ethan read on Saturday night -/
def saturday_night_pages : ℕ := 10

/-- Represents the number of pages left to read after Sunday -/
def pages_left : ℕ := 210

/-- The main theorem to prove -/
theorem ethan_reading_pages : 
  saturday_morning_pages = 40 ∧
  (saturday_morning_pages + saturday_night_pages) * 3 = total_pages - pages_left :=
sorry

end NUMINAMATH_CALUDE_ethan_reading_pages_l3950_395020


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3950_395026

/-- Given two vectors a and b in ℝ², if a is perpendicular to (a - b), then the x-coordinate of b is 9. -/
theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.2 = -2 →
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) →
  b.1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3950_395026


namespace NUMINAMATH_CALUDE_circus_ticket_problem_l3950_395002

/-- Circus ticket problem -/
theorem circus_ticket_problem (num_kids : ℕ) (kid_ticket_price : ℚ) (total_cost : ℚ) :
  num_kids = 6 →
  kid_ticket_price = 5 →
  total_cost = 50 →
  ∃ (num_adults : ℕ),
    num_adults = 2 ∧
    total_cost = num_kids * kid_ticket_price + num_adults * (2 * kid_ticket_price) :=
by sorry

end NUMINAMATH_CALUDE_circus_ticket_problem_l3950_395002


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3950_395027

theorem arithmetic_expression_evaluation : 6 + 15 / 3 - 4^2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3950_395027


namespace NUMINAMATH_CALUDE_water_consumption_days_l3950_395014

/-- Represents the daily water consumption of each sibling -/
structure SiblingWaterConsumption where
  theo : ℕ
  mason : ℕ
  roxy : ℕ

/-- Calculates the number of days it takes for the siblings to drink a given amount of water -/
def calculateDays (consumption : SiblingWaterConsumption) (totalWater : ℕ) : ℕ :=
  totalWater / (consumption.theo + consumption.mason + consumption.roxy)

/-- Theorem stating that it takes 7 days for the siblings to drink 168 cups of water -/
theorem water_consumption_days :
  let consumption : SiblingWaterConsumption := ⟨8, 7, 9⟩
  calculateDays consumption 168 = 7 := by
  sorry

#eval calculateDays ⟨8, 7, 9⟩ 168

end NUMINAMATH_CALUDE_water_consumption_days_l3950_395014


namespace NUMINAMATH_CALUDE_money_lending_problem_l3950_395070

theorem money_lending_problem (total : ℝ) (rate_A rate_B : ℝ) (time : ℝ) (interest_diff : ℝ) :
  total = 10000 ∧ 
  rate_A = 15 / 100 ∧ 
  rate_B = 18 / 100 ∧ 
  time = 2 ∧ 
  interest_diff = 360 →
  ∃ (amount_A amount_B : ℝ),
    amount_A + amount_B = total ∧
    amount_A * rate_A * time = amount_B * rate_B * time + interest_diff ∧
    amount_B = 4000 := by
  sorry

end NUMINAMATH_CALUDE_money_lending_problem_l3950_395070


namespace NUMINAMATH_CALUDE_arithmetic_progression_properties_l3950_395012

/-- An arithmetic progression with the property that the sum of its first n terms is 5n² for any n -/
structure ArithmeticProgression where
  /-- The first term of the progression -/
  a₁ : ℝ
  /-- The common difference of the progression -/
  d : ℝ
  /-- Property: The sum of the first n terms is 5n² for any n -/
  sum_property : ∀ n : ℕ, n * (2 * a₁ + (n - 1) * d) / 2 = 5 * n^2

/-- Theorem stating the properties of the arithmetic progression -/
theorem arithmetic_progression_properties (ap : ArithmeticProgression) :
  ap.d = 10 ∧ ap.a₁ = 5 ∧ ap.a₁ + ap.d = 15 ∧ ap.a₁ + 2 * ap.d = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_properties_l3950_395012


namespace NUMINAMATH_CALUDE_volume_rotational_ellipsoid_l3950_395042

/-- The volume of a rotational ellipsoid -/
theorem volume_rotational_ellipsoid (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∫ y in (-b)..b, π * a^2 * (1 - y^2 / b^2)) = (4 / 3) * π * a^2 * b :=
by sorry

end NUMINAMATH_CALUDE_volume_rotational_ellipsoid_l3950_395042


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3950_395088

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (length width : ℝ),
    r = 7 →
    length = 3 * width →
    width = 2 * r →
    length * width = 588 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3950_395088


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l3950_395043

/-- A linear function f(x) = -x + 5 -/
def f (x : ℝ) : ℝ := -x + 5

/-- P₁ is a point on the graph of f with x-coordinate -2 -/
def P₁ (y₁ : ℝ) : Prop := f (-2) = y₁

/-- P₂ is a point on the graph of f with x-coordinate -3 -/
def P₂ (y₂ : ℝ) : Prop := f (-3) = y₂

/-- Theorem: If P₁(-2, y₁) and P₂(-3, y₂) are points on the graph of f, then y₁ < y₂ -/
theorem y₁_less_than_y₂ (y₁ y₂ : ℝ) (h₁ : P₁ y₁) (h₂ : P₂ y₂) : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l3950_395043


namespace NUMINAMATH_CALUDE_function_min_value_l3950_395071

/-- Given a function f(x) = (1/3)x³ - x + m with a maximum value of 1,
    prove that its minimum value is -1/3 -/
theorem function_min_value 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ x, f x = (1/3) * x^3 - x + m) 
  (h2 : ∃ x, f x = 1 ∧ ∀ y, f y ≤ 1) : 
  ∃ x, f x = -1/3 ∧ ∀ y, f y ≥ -1/3 :=
sorry

end NUMINAMATH_CALUDE_function_min_value_l3950_395071


namespace NUMINAMATH_CALUDE_inverse_at_five_l3950_395007

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State that f has an inverse
def f_inv : ℝ → ℝ := sorry

-- Assume f_inv is the inverse of f
axiom f_inverse (x : ℝ) : f (f_inv x) = x
axiom inv_f (x : ℝ) : f_inv (f x) = x

-- Theorem to prove
theorem inverse_at_five : f_inv 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_at_five_l3950_395007


namespace NUMINAMATH_CALUDE_not_always_left_to_right_l3950_395024

theorem not_always_left_to_right : ∃ (a b c : ℕ), a + b * c ≠ (a + b) * c := by sorry

end NUMINAMATH_CALUDE_not_always_left_to_right_l3950_395024


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3950_395083

open Real

noncomputable def seriesTerms (k : ℕ) : ℝ :=
  (7^k) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem infinite_series_sum : 
  ∑' k, seriesTerms k = 7 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3950_395083


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3950_395092

def z : ℂ := Complex.I * (Complex.I + 2)

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3950_395092


namespace NUMINAMATH_CALUDE_negative_abs_negative_five_l3950_395064

theorem negative_abs_negative_five : -|-5| = -5 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_negative_five_l3950_395064


namespace NUMINAMATH_CALUDE_pen_cost_l3950_395029

/-- The cost of a pen, given the cost of a pencil and the total cost of a specific number of pencils and pens. -/
theorem pen_cost (pencil_cost : ℝ) (total_cost : ℝ) (num_pencils : ℕ) (num_pens : ℕ) : 
  pencil_cost = 2.5 →
  total_cost = 291 →
  num_pencils = 38 →
  num_pens = 56 →
  (num_pencils : ℝ) * pencil_cost + (num_pens : ℝ) * ((total_cost - (num_pencils : ℝ) * pencil_cost) / num_pens) = total_cost →
  (total_cost - (num_pencils : ℝ) * pencil_cost) / num_pens = 3.5 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l3950_395029


namespace NUMINAMATH_CALUDE_both_correct_probability_l3950_395073

-- Define the probabilities
def prob_first : ℝ := 0.75
def prob_second : ℝ := 0.55
def prob_neither : ℝ := 0.20

-- Theorem statement
theorem both_correct_probability : 
  prob_first + prob_second - (1 - prob_neither) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_both_correct_probability_l3950_395073


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3950_395081

theorem quadratic_one_root (a : ℝ) : 
  (∃! x : ℝ, a * x^2 - 2 * x + 1 = 0) ↔ (a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3950_395081


namespace NUMINAMATH_CALUDE_negation_of_all_squares_positive_l3950_395017

theorem negation_of_all_squares_positive :
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_positive_l3950_395017


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3950_395031

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2 + 1/600) * (25*x)/(73*y)) :
  Real.sqrt x / Real.sqrt y = 147/43 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3950_395031


namespace NUMINAMATH_CALUDE_bus_probability_l3950_395096

theorem bus_probability (p3 p6 : ℝ) (h1 : p3 = 0.20) (h2 : p6 = 0.60) :
  p3 + p6 = 0.80 := by
  sorry

end NUMINAMATH_CALUDE_bus_probability_l3950_395096


namespace NUMINAMATH_CALUDE_insurance_agents_count_l3950_395095

/-- The number of claims Jan can handle -/
def jan_claims : ℕ := 20

/-- The number of claims John can handle -/
def john_claims : ℕ := jan_claims + jan_claims * 30 / 100

/-- The number of claims Missy can handle -/
def missy_claims : ℕ := john_claims + 15

/-- The total number of agents -/
def num_agents : ℕ := 3

theorem insurance_agents_count :
  missy_claims = 41 → num_agents = 3 := by
  sorry

end NUMINAMATH_CALUDE_insurance_agents_count_l3950_395095


namespace NUMINAMATH_CALUDE_negation_equivalence_l3950_395082

theorem negation_equivalence : 
  (¬ ∃ (x : ℝ), x > 0 ∧ Real.sqrt x ≤ x + 1) ↔ 
  (∀ (x : ℝ), x > 0 → Real.sqrt x > x + 1) := by
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3950_395082


namespace NUMINAMATH_CALUDE_range_of_a_l3950_395034

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (2*x - 1)/(x - 1) < 0 → x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0 ∧ (2*x - 1)/(x - 1) ≥ 0) →
  0 ≤ a ∧ a ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3950_395034


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3950_395048

/-- Given two plane vectors a and b, prove that |a + 2b| = √7 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  (a.fst = 1 ∧ a.snd = 0) →  -- a = (1,0)
  ‖b‖ = 1 →  -- |b| = 1
  Real.cos (Real.pi / 3) = (a.fst * b.fst + a.snd * b.snd) / (‖a‖ * ‖b‖) →  -- angle between a and b is 60°
  ‖a + 2 • b‖ = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3950_395048


namespace NUMINAMATH_CALUDE_additive_inverse_of_2023_l3950_395032

theorem additive_inverse_of_2023 : ∃! x : ℤ, 2023 + x = 0 ∧ x = -2023 := by sorry

end NUMINAMATH_CALUDE_additive_inverse_of_2023_l3950_395032


namespace NUMINAMATH_CALUDE_factorization_proof_l3950_395074

theorem factorization_proof (y : ℝ) : 81 * y^19 + 162 * y^38 = 81 * y^19 * (1 + 2 * y^19) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3950_395074


namespace NUMINAMATH_CALUDE_prism_volume_sum_l3950_395049

theorem prism_volume_sum (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  Nat.lcm a b = 72 →
  Nat.lcm a c = 24 →
  Nat.lcm b c = 18 →
  (∃ (a_min b_min c_min a_max b_max c_max : ℕ),
    (∀ a' b' c' : ℕ, 
      Nat.lcm a' b' = 72 → Nat.lcm a' c' = 24 → Nat.lcm b' c' = 18 →
      a' * b' * c' ≥ a_min * b_min * c_min) ∧
    (∀ a' b' c' : ℕ, 
      Nat.lcm a' b' = 72 → Nat.lcm a' c' = 24 → Nat.lcm b' c' = 18 →
      a' * b' * c' ≤ a_max * b_max * c_max) ∧
    a_min * b_min * c_min + a_max * b_max * c_max = 3024) := by
  sorry

#check prism_volume_sum

end NUMINAMATH_CALUDE_prism_volume_sum_l3950_395049


namespace NUMINAMATH_CALUDE_leona_earnings_l3950_395093

/-- Given an hourly rate calculated from earning $24.75 for 3 hours,
    prove that the earnings for 5 hours at the same rate will be $41.25. -/
theorem leona_earnings (hourly_rate : ℝ) (h1 : hourly_rate * 3 = 24.75) :
  hourly_rate * 5 = 41.25 := by
  sorry

end NUMINAMATH_CALUDE_leona_earnings_l3950_395093


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l3950_395075

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, a * x + 2 * y + 2 = 0 → 3 * x - y - 2 = 0 → 
    (a * x + 2 * y + 2 = 0 ∧ 3 * x - y - 2 = 0) → 
    ((-a/2) * 3 = -1)) → 
  a = 2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l3950_395075


namespace NUMINAMATH_CALUDE_john_kate_penny_difference_l3950_395011

theorem john_kate_penny_difference :
  ∀ (john_pennies kate_pennies : ℕ),
    john_pennies = 388 →
    kate_pennies = 223 →
    john_pennies - kate_pennies = 165 := by
  sorry

end NUMINAMATH_CALUDE_john_kate_penny_difference_l3950_395011


namespace NUMINAMATH_CALUDE_cubic_polynomial_theorem_l3950_395067

def is_monic_cubic (q : ℝ → ℂ) : Prop :=
  ∃ a b c : ℝ, ∀ x, q x = x^3 + a*x^2 + b*x + c

theorem cubic_polynomial_theorem (q : ℝ → ℂ) 
  (h_monic : is_monic_cubic q)
  (h_root : q (2 - 3*I) = 0)
  (h_const : q 0 = -72) :
  ∀ x, q x = x^3 - (100/13)*x^2 + (236/13)*x - 936/13 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_theorem_l3950_395067


namespace NUMINAMATH_CALUDE_book_price_percentage_l3950_395062

/-- Given the original price and current price of a book, prove that the current price is 80% of the original price. -/
theorem book_price_percentage (original_price current_price : ℝ) 
  (h1 : original_price = 25)
  (h2 : current_price = 20) :
  current_price / original_price = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_book_price_percentage_l3950_395062


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_m_greater_than_one_sufficient_m_greater_than_one_not_necessary_l3950_395035

theorem sufficient_not_necessary_condition (m : ℝ) : 
  (∀ x ≥ 1, 3^(x + m) - 3 * Real.sqrt 3 > 0) ↔ m > 1/2 :=
by sorry

theorem m_greater_than_one_sufficient (m : ℝ) :
  m > 1 → ∀ x ≥ 1, 3^(x + m) - 3 * Real.sqrt 3 > 0 :=
by sorry

theorem m_greater_than_one_not_necessary :
  ∃ m, m ≤ 1 ∧ (∀ x ≥ 1, 3^(x + m) - 3 * Real.sqrt 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_m_greater_than_one_sufficient_m_greater_than_one_not_necessary_l3950_395035


namespace NUMINAMATH_CALUDE_unique_prime_satisfying_condition_l3950_395061

theorem unique_prime_satisfying_condition : 
  ∀ p : ℕ, Prime p → (Prime (p^3 + p^2 + 11*p + 2) ↔ p = 3) := by sorry

end NUMINAMATH_CALUDE_unique_prime_satisfying_condition_l3950_395061


namespace NUMINAMATH_CALUDE_money_duration_l3950_395015

def mowing_earnings : ℕ := 14
def weed_eating_earnings : ℕ := 26
def weekly_spending : ℕ := 5

theorem money_duration : 
  (mowing_earnings + weed_eating_earnings) / weekly_spending = 8 := by
sorry

end NUMINAMATH_CALUDE_money_duration_l3950_395015


namespace NUMINAMATH_CALUDE_no_real_roots_for_ff_l3950_395094

/-- A quadratic polynomial function -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The property that f(x) = x has no real roots -/
def NoRealRootsForFX (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ x

theorem no_real_roots_for_ff (a b c : ℝ) :
  let f := QuadraticPolynomial a b c
  NoRealRootsForFX f → NoRealRootsForFX (f ∘ f) := by
  sorry

#check no_real_roots_for_ff

end NUMINAMATH_CALUDE_no_real_roots_for_ff_l3950_395094


namespace NUMINAMATH_CALUDE_prime_composite_property_l3950_395013

theorem prime_composite_property (n : ℕ) :
  (∀ (a : Fin n → ℕ), Function.Injective a →
    ∃ (i j : Fin n), (a i + a j) / Nat.gcd (a i) (a j) ≥ 2 * n - 1) ∨
  (∃ (a : Fin n → ℕ), Function.Injective a ∧
    ∀ (i j : Fin n), (a i + a j) / Nat.gcd (a i) (a j) < 2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_composite_property_l3950_395013


namespace NUMINAMATH_CALUDE_video_game_expenditure_l3950_395066

/-- The cost of the basketball game -/
def basketball_cost : ℚ := 5.20

/-- The cost of the racing game -/
def racing_cost : ℚ := 4.23

/-- The total cost of video games -/
def total_cost : ℚ := basketball_cost + racing_cost

theorem video_game_expenditure : total_cost = 9.43 := by
  sorry

end NUMINAMATH_CALUDE_video_game_expenditure_l3950_395066


namespace NUMINAMATH_CALUDE_min_value_expression_l3950_395079

theorem min_value_expression (a b c d : ℝ) (h1 : b > c) (h2 : c > d) (h3 : d > a) (h4 : b ≠ 0) :
  (a + b)^2 + (b - c)^2 + (c - d)^2 + (d - a)^2 ≥ b^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3950_395079


namespace NUMINAMATH_CALUDE_burger_cost_proof_l3950_395076

/-- The cost of a burger at McDonald's -/
def burger_cost : ℝ := 5

/-- The cost of one pack of fries -/
def fries_cost : ℝ := 2

/-- The cost of a salad -/
def salad_cost : ℝ := 3 * fries_cost

/-- The total cost of the meal -/
def total_cost : ℝ := 15

theorem burger_cost_proof :
  burger_cost + 2 * fries_cost + salad_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_burger_cost_proof_l3950_395076


namespace NUMINAMATH_CALUDE_decrease_in_profit_for_given_scenario_l3950_395036

/-- Represents the financial data of a textile manufacturing firm -/
structure TextileFirm where
  total_looms : ℕ
  sales_value : ℕ
  manufacturing_expenses : ℕ
  establishment_charges : ℕ

/-- Calculates the decrease in profit when one loom is idle for a month -/
def decrease_in_profit (firm : TextileFirm) : ℕ :=
  let sales_per_loom := firm.sales_value / firm.total_looms
  let expenses_per_loom := firm.manufacturing_expenses / firm.total_looms
  sales_per_loom - expenses_per_loom

/-- Theorem stating the decrease in profit for the given scenario -/
theorem decrease_in_profit_for_given_scenario :
  let firm := TextileFirm.mk 125 500000 150000 75000
  decrease_in_profit firm = 2800 := by
  sorry

#eval decrease_in_profit (TextileFirm.mk 125 500000 150000 75000)

end NUMINAMATH_CALUDE_decrease_in_profit_for_given_scenario_l3950_395036


namespace NUMINAMATH_CALUDE_square_area_error_l3950_395091

theorem square_area_error (S : ℝ) (h : S > 0) :
  let measured_side := 1.05 * S
  let actual_area := S^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 10.25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l3950_395091


namespace NUMINAMATH_CALUDE_complete_square_k_value_l3950_395050

theorem complete_square_k_value (x : ℝ) : 
  ∃ (p k : ℝ), (x^2 - 6*x + 5 = 0) ↔ ((x - p)^2 = k) ∧ k = 4 := by
sorry

end NUMINAMATH_CALUDE_complete_square_k_value_l3950_395050


namespace NUMINAMATH_CALUDE_problem_statement_l3950_395006

theorem problem_statement (y : ℝ) (hy : y > 0) : 
  ∃ y, ((3/5 * 2500) * (2/7 * ((5/8 * 4000) + (1/4 * 3600) - ((11/20 * 7200) / (3/10 * y))))) = 25000 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3950_395006


namespace NUMINAMATH_CALUDE_books_from_second_shop_l3950_395097

theorem books_from_second_shop 
  (first_shop_books : ℕ) 
  (first_shop_cost : ℕ) 
  (second_shop_cost : ℕ) 
  (average_price : ℚ) : ℕ :=
  by
    have h1 : first_shop_books = 40 := by sorry
    have h2 : first_shop_cost = 600 := by sorry
    have h3 : second_shop_cost = 240 := by sorry
    have h4 : average_price = 14 := by sorry
    
    -- The number of books from the second shop
    let second_shop_books : ℕ := 20
    
    -- Prove that this satisfies the conditions
    sorry

#check books_from_second_shop

end NUMINAMATH_CALUDE_books_from_second_shop_l3950_395097


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3950_395037

/-- The eccentricity of a hyperbola with equation x^2/a^2 - y^2/b^2 = 1,
    where one of its asymptotes passes through the point (3, -4), is 5/3. -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → c^2 = a^2 + b^2) →
  (∃ k : ℝ, k * 3 = a ∧ k * (-4) = b) →
  c / a = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3950_395037


namespace NUMINAMATH_CALUDE_max_value_of_x_l3950_395058

theorem max_value_of_x (x y z : ℝ) (sum_eq : x + y + z = 7) (prod_eq : x*y + x*z + y*z = 12) :
  x ≤ 1 ∧ ∃ (a b : ℝ), a + b + 1 = 7 ∧ a*b + a*1 + b*1 = 12 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_l3950_395058


namespace NUMINAMATH_CALUDE_negation_of_all_divisible_by_two_are_even_l3950_395098

theorem negation_of_all_divisible_by_two_are_even :
  (¬ ∀ n : ℕ, 2 ∣ n → Even n) ↔ (∃ n : ℕ, 2 ∣ n ∧ ¬Even n) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_divisible_by_two_are_even_l3950_395098


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l3950_395069

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_positive : ∀ x > 0, f x = -x + 1) :
  ∀ x < 0, f x = -x - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l3950_395069


namespace NUMINAMATH_CALUDE_escalator_walking_speed_l3950_395041

theorem escalator_walking_speed 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 11) 
  (h2 : escalator_length = 140) 
  (h3 : time_taken = 10) : 
  ∃ (walking_speed : ℝ), 
    walking_speed = 3 ∧ 
    escalator_length = (walking_speed + escalator_speed) * time_taken := by
  sorry

end NUMINAMATH_CALUDE_escalator_walking_speed_l3950_395041


namespace NUMINAMATH_CALUDE_largest_coefficient_expansion_l3950_395040

theorem largest_coefficient_expansion (x : ℝ) (x_nonzero : x ≠ 0) :
  ∃ (terms : List ℝ), 
    (1/x - 1)^5 = terms.sum ∧ 
    (10/x^3 ∈ terms) ∧
    ∀ (term : ℝ), term ∈ terms → |term| ≤ |10/x^3| :=
by sorry

end NUMINAMATH_CALUDE_largest_coefficient_expansion_l3950_395040


namespace NUMINAMATH_CALUDE_exactly_three_blue_marbles_l3950_395039

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def num_picks : ℕ := 7
def num_blue_picks : ℕ := 3

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem exactly_three_blue_marbles :
  Nat.choose num_picks num_blue_picks *
  (prob_blue ^ num_blue_picks) *
  (prob_red ^ (num_picks - num_blue_picks)) =
  640 / 1547 := by sorry

end NUMINAMATH_CALUDE_exactly_three_blue_marbles_l3950_395039


namespace NUMINAMATH_CALUDE_projection_correct_l3950_395068

/-- Given vectors u and t, prove that the projection of u onto t is correct. -/
theorem projection_correct (u t : ℝ × ℝ) : 
  u = (4, -3) → t = (-6, 8) → 
  let proj := ((u.1 * t.1 + u.2 * t.2) / (t.1 * t.1 + t.2 * t.2)) • t
  proj.1 = 288 / 100 ∧ proj.2 = -384 / 100 := by
  sorry

end NUMINAMATH_CALUDE_projection_correct_l3950_395068


namespace NUMINAMATH_CALUDE_product_quantity_relationship_l3950_395072

/-- The initial budget in yuan -/
def initial_budget : ℝ := 1500

/-- The price increase of product A in yuan -/
def price_increase_A : ℝ := 1.5

/-- The price increase of product B in yuan -/
def price_increase_B : ℝ := 1

/-- The reduction in quantity of product A in the first scenario -/
def quantity_reduction_A1 : ℝ := 10

/-- The budget excess in the first scenario -/
def budget_excess : ℝ := 29

/-- The reduction in quantity of product A in the second scenario -/
def quantity_reduction_A2 : ℝ := 5

/-- The total cost in the second scenario -/
def total_cost_scenario2 : ℝ := 1563.5

theorem product_quantity_relationship (x y a b : ℝ) :
  (a * x + b * y = initial_budget) →
  ((a + price_increase_A) * (x - quantity_reduction_A1) + (b + price_increase_B) * y = initial_budget + budget_excess) →
  ((a + 1) * (x - quantity_reduction_A2) + (b + 1) * y = total_cost_scenario2) →
  (2 * x + y > 205) →
  (2 * x + y < 210) →
  (x + 2 * y = 186) := by
sorry

end NUMINAMATH_CALUDE_product_quantity_relationship_l3950_395072


namespace NUMINAMATH_CALUDE_cases_needed_l3950_395060

def boxes_sold : ℕ := 10
def boxes_per_case : ℕ := 2

theorem cases_needed : boxes_sold / boxes_per_case = 5 := by
  sorry

end NUMINAMATH_CALUDE_cases_needed_l3950_395060
