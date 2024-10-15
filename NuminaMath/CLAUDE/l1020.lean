import Mathlib

namespace NUMINAMATH_CALUDE_barry_sotter_magic_l1020_102043

theorem barry_sotter_magic (n : ℕ) : (n + 3 : ℚ) / 3 = 50 ↔ n = 147 := by sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_l1020_102043


namespace NUMINAMATH_CALUDE_range_of_absolute_value_sum_l1020_102067

theorem range_of_absolute_value_sum (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  ∀ x : ℝ, |x - a| + |x - b| = a - b ↔ b ≤ x ∧ x ≤ a :=
by sorry

end NUMINAMATH_CALUDE_range_of_absolute_value_sum_l1020_102067


namespace NUMINAMATH_CALUDE_expression_evaluation_l1020_102015

theorem expression_evaluation : 6^2 - 4*5 + 2^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1020_102015


namespace NUMINAMATH_CALUDE_lcm_product_hcf_l1020_102018

theorem lcm_product_hcf (x y : ℕ+) : 
  Nat.lcm x y = 560 → x * y = 42000 → Nat.gcd x y = 75 := by
  sorry

end NUMINAMATH_CALUDE_lcm_product_hcf_l1020_102018


namespace NUMINAMATH_CALUDE_reflection_over_y_axis_l1020_102022

theorem reflection_over_y_axis :
  let reflect_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![-1, 0], ![0, 1]]
  ∀ (x y : ℝ), 
    reflect_matrix.mulVec ![x, y] = ![-x, y] := by sorry

end NUMINAMATH_CALUDE_reflection_over_y_axis_l1020_102022


namespace NUMINAMATH_CALUDE_first_year_interest_rate_l1020_102073

/-- Proves that the first-year interest rate is 4% given the problem conditions --/
theorem first_year_interest_rate 
  (initial_amount : ℝ) 
  (final_amount : ℝ) 
  (second_year_rate : ℝ) 
  (h1 : initial_amount = 4000)
  (h2 : final_amount = 4368)
  (h3 : second_year_rate = 0.05)
  : ∃ (R : ℝ), 
    initial_amount * (1 + R) * (1 + second_year_rate) = final_amount ∧ 
    R = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_first_year_interest_rate_l1020_102073


namespace NUMINAMATH_CALUDE_min_value_theorem_l1020_102071

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2 / (3 * a + b) + 1 / (a + 2 * b) = 4) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 / (3 * x + y) + 1 / (x + 2 * y) = 4 → 7 * x + 4 * y ≥ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1020_102071


namespace NUMINAMATH_CALUDE_sqrt2_plus_sqrt3_irrational_l1020_102095

theorem sqrt2_plus_sqrt3_irrational : Irrational (Real.sqrt 2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_plus_sqrt3_irrational_l1020_102095


namespace NUMINAMATH_CALUDE_smallest_sum_a_b_l1020_102010

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_sum_a_b (a b : ℕ) 
  (h1 : a^a % b^b = 0)
  (h2 : ¬(a % b = 0))
  (h3 : is_coprime b 210) :
  (∀ (x y : ℕ), x^x % y^y = 0 → ¬(x % y = 0) → is_coprime y 210 → a + b ≤ x + y) →
  a + b = 374 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_a_b_l1020_102010


namespace NUMINAMATH_CALUDE_jane_hiking_distance_l1020_102008

/-- The distance between two points given a specific path --/
theorem jane_hiking_distance (A B D : ℝ × ℝ) : 
  (A.1 = B.1 ∧ A.2 + 3 = B.2) →  -- AB is 3 units northward
  (Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 8) →  -- BD is 8 units long
  (D.1 - B.1 = D.2 - B.2) →  -- 45 degree angle (isosceles right triangle)
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = Real.sqrt (73 + 24 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_jane_hiking_distance_l1020_102008


namespace NUMINAMATH_CALUDE_power_two_greater_than_sum_of_powers_l1020_102086

theorem power_two_greater_than_sum_of_powers (n : ℕ) (x : ℝ) 
  (h1 : n ≥ 2) (h2 : |x| < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_sum_of_powers_l1020_102086


namespace NUMINAMATH_CALUDE_ski_and_snowboard_intersection_l1020_102079

theorem ski_and_snowboard_intersection (total : ℕ) (ski : ℕ) (snowboard : ℕ) (neither : ℕ)
  (h_total : total = 20)
  (h_ski : ski = 11)
  (h_snowboard : snowboard = 13)
  (h_neither : neither = 3) :
  ski + snowboard - (total - neither) = 7 :=
by sorry

end NUMINAMATH_CALUDE_ski_and_snowboard_intersection_l1020_102079


namespace NUMINAMATH_CALUDE_adams_change_l1020_102041

def adams_money : ℝ := 5.00
def airplane_cost : ℝ := 4.28

theorem adams_change :
  adams_money - airplane_cost = 0.72 := by sorry

end NUMINAMATH_CALUDE_adams_change_l1020_102041


namespace NUMINAMATH_CALUDE_fifth_power_sum_l1020_102076

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l1020_102076


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1020_102020

theorem arithmetic_equality : 2021 - 2223 + 2425 = 2223 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1020_102020


namespace NUMINAMATH_CALUDE_molecular_weight_Al2S3_l1020_102092

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06

-- Define the composition of Al2S3
def Al_atoms_in_Al2S3 : ℕ := 2
def S_atoms_in_Al2S3 : ℕ := 3

-- Define the number of moles
def moles_Al2S3 : ℝ := 3

-- Theorem statement
theorem molecular_weight_Al2S3 :
  let molecular_weight_one_mole := Al_atoms_in_Al2S3 * atomic_weight_Al + S_atoms_in_Al2S3 * atomic_weight_S
  moles_Al2S3 * molecular_weight_one_mole = 450.42 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_Al2S3_l1020_102092


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1020_102044

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1020_102044


namespace NUMINAMATH_CALUDE_sum_of_squares_even_2_to_14_l1020_102050

def evenSquareSum : ℕ → ℕ
| 0 => 0
| n + 1 => if n + 1 ≤ 7 ∧ 2 * (n + 1) ≤ 14 then (2 * (n + 1))^2 + evenSquareSum n else evenSquareSum n

theorem sum_of_squares_even_2_to_14 : evenSquareSum 7 = 560 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_even_2_to_14_l1020_102050


namespace NUMINAMATH_CALUDE_matrix_equation_l1020_102060

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_equation (x y z w : ℝ) 
  (h1 : A * B x y z w = B x y z w * A) 
  (h2 : 4 * y ≠ z) : 
  (x - w) / (z - 4 * y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_l1020_102060


namespace NUMINAMATH_CALUDE_more_solutions_for_first_eq_l1020_102080

/-- The upper bound for x, y, z, and t -/
def upperBound : ℕ := 10^6

/-- The number of integral solutions of x² - y² = z³ - t³ -/
def N : ℕ := sorry

/-- The number of integral solutions of x² - y² = z³ - t³ + 1 -/
def M : ℕ := sorry

/-- Predicate for the first equation -/
def firstEq (x y z t : ℕ) : Prop :=
  x^2 - y^2 = z^3 - t^3 ∧ x ≤ upperBound ∧ y ≤ upperBound ∧ z ≤ upperBound ∧ t ≤ upperBound

/-- Predicate for the second equation -/
def secondEq (x y z t : ℕ) : Prop :=
  x^2 - y^2 = z^3 - t^3 + 1 ∧ x ≤ upperBound ∧ y ≤ upperBound ∧ z ≤ upperBound ∧ t ≤ upperBound

/-- Theorem stating that N > M -/
theorem more_solutions_for_first_eq : N > M := by
  sorry

end NUMINAMATH_CALUDE_more_solutions_for_first_eq_l1020_102080


namespace NUMINAMATH_CALUDE_specific_cube_stack_surface_area_l1020_102005

/-- Represents a three-dimensional shape formed by stacking cubes -/
structure CubeStack where
  num_cubes : ℕ
  edge_length : ℝ
  num_layers : ℕ

/-- Calculates the surface area of a cube stack -/
def surface_area (stack : CubeStack) : ℝ :=
  sorry

/-- Theorem stating that a specific cube stack has a surface area of 72 square meters -/
theorem specific_cube_stack_surface_area :
  let stack : CubeStack := {
    num_cubes := 30,
    edge_length := 1,
    num_layers := 4
  }
  surface_area stack = 72 := by
  sorry

end NUMINAMATH_CALUDE_specific_cube_stack_surface_area_l1020_102005


namespace NUMINAMATH_CALUDE_candy_distribution_l1020_102069

theorem candy_distribution (a b d : ℕ) : 
  (4 * b = 3 * a) →  -- While Andrey eats 4 candies, Boris eats 3
  (6 * d = 7 * a) →  -- While Andrey eats 6 candies, Denis eats 7
  (a + b + d = 70) → -- Total candies eaten
  (a = 24 ∧ b = 18 ∧ d = 28) := by sorry

end NUMINAMATH_CALUDE_candy_distribution_l1020_102069


namespace NUMINAMATH_CALUDE_angle_inequality_l1020_102074

theorem angle_inequality (x y z : Real) 
  (h1 : 0 < x ∧ x < π/2)
  (h2 : 0 < y ∧ y < π/2)
  (h3 : 0 < z ∧ z < π/2)
  (h4 : (Real.sin x + Real.cos x) * (Real.sin y + 2 * Real.cos y) * (Real.sin z + 3 * Real.cos z) = 10) :
  x = π/4 ∧ x > y ∧ y > z := by sorry

end NUMINAMATH_CALUDE_angle_inequality_l1020_102074


namespace NUMINAMATH_CALUDE_cube_volume_proof_l1020_102062

theorem cube_volume_proof (n : ℕ) (m : ℕ) : 
  (n^3 = 98 + m^3) ∧ 
  (m ≠ 1) ∧ 
  (∃ (k : ℕ), n^3 = 99 * k) →
  n^3 = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_proof_l1020_102062


namespace NUMINAMATH_CALUDE_smallest_quotient_is_seven_l1020_102029

/-- A type representing a division of numbers 1 to 10 into two groups -/
def Division := (Finset Nat) × (Finset Nat)

/-- Checks if a division is valid (contains all numbers from 1 to 10 exactly once) -/
def is_valid_division (d : Division) : Prop :=
  d.1 ∪ d.2 = Finset.range 10 ∧ d.1 ∩ d.2 = ∅

/-- Calculates the product of numbers in a Finset -/
def product (s : Finset Nat) : Nat :=
  s.prod id

/-- Checks if the division satisfies the divisibility condition -/
def satisfies_condition (d : Division) : Prop :=
  (product d.1) % (product d.2) = 0

/-- The main theorem stating the smallest possible quotient is 7 -/
theorem smallest_quotient_is_seven :
  ∀ d : Division, 
    is_valid_division d → 
    satisfies_condition d → 
    (product d.1) / (product d.2) ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_quotient_is_seven_l1020_102029


namespace NUMINAMATH_CALUDE_age_sum_problem_l1020_102078

theorem age_sum_problem (leonard_age nina_age jerome_age : ℕ) : 
  leonard_age = 6 →
  nina_age = leonard_age + 4 →
  jerome_age = 2 * nina_age →
  leonard_age + nina_age + jerome_age = 36 := by
sorry

end NUMINAMATH_CALUDE_age_sum_problem_l1020_102078


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1020_102039

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1020_102039


namespace NUMINAMATH_CALUDE_third_term_expansion_l1020_102077

-- Define i as the imaginary unit
axiom i : ℂ
axiom i_squared : i * i = -1

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem third_term_expansion :
  let n : ℕ := 6
  let r : ℕ := 2
  (binomial n r : ℂ) * (1 : ℂ)^(n - r) * i^r = -15 := by sorry

end NUMINAMATH_CALUDE_third_term_expansion_l1020_102077


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1020_102011

theorem min_value_of_function (x : ℝ) (h : x > 2) : x + 4 / (x - 2) ≥ 6 := by
  sorry

theorem equality_condition : ∃ x > 2, x + 4 / (x - 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1020_102011


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1020_102024

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1020_102024


namespace NUMINAMATH_CALUDE_rook_placement_count_l1020_102032

theorem rook_placement_count (n k : ℕ) (h1 : n = 8) (h2 : k = 6) :
  (Nat.choose n k)^2 * Nat.factorial k = 564480 := by
  sorry

end NUMINAMATH_CALUDE_rook_placement_count_l1020_102032


namespace NUMINAMATH_CALUDE_problem_solution_l1020_102013

theorem problem_solution (P Q R : ℚ) : 
  (5 / 8 = P / 56) → 
  (5 / 8 = 80 / Q) → 
  (R = P - 4) → 
  (Q + R = 159) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1020_102013


namespace NUMINAMATH_CALUDE_total_sacks_needed_l1020_102059

/-- The number of sacks of strawberries needed for the first bakery per week -/
def bakery1_weekly_need : ℕ := 2

/-- The number of sacks of strawberries needed for the second bakery per week -/
def bakery2_weekly_need : ℕ := 4

/-- The number of sacks of strawberries needed for the third bakery per week -/
def bakery3_weekly_need : ℕ := 12

/-- The number of weeks for which the supply is calculated -/
def supply_period : ℕ := 4

/-- Theorem stating that the total number of sacks needed for all bakeries in 4 weeks is 72 -/
theorem total_sacks_needed :
  (bakery1_weekly_need + bakery2_weekly_need + bakery3_weekly_need) * supply_period = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_sacks_needed_l1020_102059


namespace NUMINAMATH_CALUDE_pau_total_chicken_l1020_102000

def kobe_order : ℕ := 5

def pau_order (kobe : ℕ) : ℕ := 2 * kobe

def total_pau_order (kobe : ℕ) : ℕ := 2 * pau_order kobe

theorem pau_total_chicken :
  total_pau_order kobe_order = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_pau_total_chicken_l1020_102000


namespace NUMINAMATH_CALUDE_remaining_laps_after_break_l1020_102089

/-- The number of laps Jeff needs to swim over the weekend -/
def total_laps : ℕ := 98

/-- The number of laps Jeff swam on Saturday -/
def saturday_laps : ℕ := 27

/-- The number of laps Jeff swam on Sunday morning -/
def sunday_morning_laps : ℕ := 15

/-- Theorem stating the number of laps remaining after Jeff's break on Sunday -/
theorem remaining_laps_after_break : 
  total_laps - saturday_laps - sunday_morning_laps = 56 := by
  sorry

end NUMINAMATH_CALUDE_remaining_laps_after_break_l1020_102089


namespace NUMINAMATH_CALUDE_odometer_sum_squares_l1020_102056

/-- Represents the odometer reading as a three-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≠ 0 ∧ hundreds + tens + ones = 7

/-- Represents a car journey -/
structure CarJourney where
  duration : Nat
  average_speed : Nat
  initial_reading : OdometerReading
  final_reading : OdometerReading
  speed_constraint : average_speed = 60
  odometer_constraint : final_reading.hundreds = initial_reading.ones ∧
                        final_reading.tens = initial_reading.tens ∧
                        final_reading.ones = initial_reading.hundreds

theorem odometer_sum_squares (journey : CarJourney) :
  journey.initial_reading.hundreds ^ 2 +
  journey.initial_reading.tens ^ 2 +
  journey.initial_reading.ones ^ 2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_odometer_sum_squares_l1020_102056


namespace NUMINAMATH_CALUDE_gcd_228_1995_l1020_102003

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l1020_102003


namespace NUMINAMATH_CALUDE_boys_in_class_l1020_102061

theorem boys_in_class (total : ℕ) (girls_fraction : ℚ) (boys : ℕ) : 
  total = 160 → girls_fraction = 1 / 4 → boys = 120 → 
  boys = total * (1 - girls_fraction) := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l1020_102061


namespace NUMINAMATH_CALUDE_union_contains_1980_l1020_102016

/-- An arithmetic progression of integers -/
def ArithmeticProgression (a₀ d : ℤ) : Set ℤ :=
  {n : ℤ | ∃ k : ℕ, n = a₀ + k * d}

theorem union_contains_1980
  (A B C : Set ℤ)
  (hA : ∃ a₀ d : ℤ, A = ArithmeticProgression a₀ d)
  (hB : ∃ a₀ d : ℤ, B = ArithmeticProgression a₀ d)
  (hC : ∃ a₀ d : ℤ, C = ArithmeticProgression a₀ d)
  (h_union : {1, 2, 3, 4, 5, 6, 7, 8} ⊆ A ∪ B ∪ C) :
  1980 ∈ A ∪ B ∪ C :=
sorry

end NUMINAMATH_CALUDE_union_contains_1980_l1020_102016


namespace NUMINAMATH_CALUDE_student_arrangement_theorem_l1020_102088

/-- The number of ways to arrange 3 male and 2 female students in a row -/
def total_arrangements : ℕ := 120

/-- The number of arrangements where exactly two male students are adjacent -/
def two_male_adjacent : ℕ := 72

/-- The number of arrangements where 3 male students of different heights 
    are arranged in descending order of height -/
def male_descending_height : ℕ := 20

/-- Given 3 male students and 2 female students, prove:
    1. The total number of arrangements
    2. The number of arrangements with exactly two male students adjacent
    3. The number of arrangements with male students in descending height order -/
theorem student_arrangement_theorem 
  (male_count : ℕ) 
  (female_count : ℕ) 
  (h1 : male_count = 3) 
  (h2 : female_count = 2) :
  (total_arrangements = 120) ∧ 
  (two_male_adjacent = 72) ∧ 
  (male_descending_height = 20) := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_theorem_l1020_102088


namespace NUMINAMATH_CALUDE_money_distribution_l1020_102072

/-- Proves that B and C together have Rs. 450 given the conditions of the problem -/
theorem money_distribution (total : ℕ) (ac_sum : ℕ) (c_amount : ℕ) 
  (h1 : total = 600)
  (h2 : ac_sum = 250)
  (h3 : c_amount = 100) : 
  total - (ac_sum - c_amount) + c_amount = 450 := by
  sorry

#check money_distribution

end NUMINAMATH_CALUDE_money_distribution_l1020_102072


namespace NUMINAMATH_CALUDE_lemonade_pitchers_l1020_102053

/-- Represents the number of glasses a pitcher can serve -/
def glasses_per_pitcher : ℕ := 5

/-- Represents the total number of glasses served -/
def total_glasses_served : ℕ := 30

/-- Calculates the number of pitchers needed -/
def pitchers_needed : ℕ := total_glasses_served / glasses_per_pitcher

theorem lemonade_pitchers : pitchers_needed = 6 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_pitchers_l1020_102053


namespace NUMINAMATH_CALUDE_darren_tshirts_l1020_102084

/-- The number of packs of white t-shirts Darren bought -/
def white_packs : ℕ := 5

/-- The number of t-shirts in each pack of white t-shirts -/
def white_per_pack : ℕ := 6

/-- The number of packs of blue t-shirts Darren bought -/
def blue_packs : ℕ := 3

/-- The number of t-shirts in each pack of blue t-shirts -/
def blue_per_pack : ℕ := 9

/-- The total number of t-shirts Darren bought -/
def total_tshirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

theorem darren_tshirts : total_tshirts = 57 := by
  sorry

end NUMINAMATH_CALUDE_darren_tshirts_l1020_102084


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1020_102068

theorem sum_of_fractions (A B : ℕ) (h : (A : ℚ) / 11 + (B : ℚ) / 3 = 17 / 33) : A + B = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1020_102068


namespace NUMINAMATH_CALUDE_intersection_A_B_l1020_102051

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x)}

-- Define set B
def B : Set ℝ := {x | x - 1 > 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1020_102051


namespace NUMINAMATH_CALUDE_subtract_square_equals_two_square_l1020_102002

theorem subtract_square_equals_two_square (x : ℝ) : 3 * x^2 - x^2 = 2 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_square_equals_two_square_l1020_102002


namespace NUMINAMATH_CALUDE_right_triangle_ratio_square_l1020_102099

theorem right_triangle_ratio_square (a c p : ℝ) (h1 : a > 0) (h2 : c > 0) (h3 : p > 0) : 
  (c / a = a / p) → (c^2 = a^2 + p^2) → ((c / a)^2 = (1 + Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_square_l1020_102099


namespace NUMINAMATH_CALUDE_sum_of_angles_F_and_C_l1020_102034

-- Define the circle and points
variable (circle : Circle ℝ)
variable (A B C D E : circle.sphere)

-- Define the arcs and their measures
variable (arc_AB arc_DE : circle.sphere)
variable (measure_AB measure_DE : ℝ)

-- Define point F as intersection of chords
variable (F : circle.sphere)

-- Hypotheses
variable (h1 : measure_AB = 60)
variable (h2 : measure_DE = 72)
variable (h3 : F ∈ (circle.chord A C) ∩ (circle.chord B D))

-- Theorem statement
theorem sum_of_angles_F_and_C :
  ∃ (angle_F angle_C : ℝ),
    angle_F + angle_C = 42 ∧
    angle_F = abs ((measure circle.arc A C - measure circle.arc B D) / 2) ∧
    angle_C = measure_DE / 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_angles_F_and_C_l1020_102034


namespace NUMINAMATH_CALUDE_trig_identity_l1020_102055

theorem trig_identity (α : Real) (h : Real.tan α = 2) :
  7 * Real.sin α ^ 2 + 3 * Real.cos α ^ 2 = 31 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1020_102055


namespace NUMINAMATH_CALUDE_snail_path_count_l1020_102004

theorem snail_path_count (n : ℕ) : 
  (number_of_paths : ℕ) = (Nat.choose (2 * n) n) ^ 2 :=
by
  sorry

where
  number_of_paths : ℕ := 
    count_closed_paths_on_graph_paper (2 * n)

  count_closed_paths_on_graph_paper (steps : ℕ) : ℕ := 
    -- Returns the number of distinct paths on graph paper
    -- that start and end at the same vertex
    -- and have a total length of 'steps'
    sorry

end NUMINAMATH_CALUDE_snail_path_count_l1020_102004


namespace NUMINAMATH_CALUDE_min_value_trig_expression_min_value_achievable_l1020_102031

theorem min_value_trig_expression (θ : Real) :
  (1 / (2 - Real.cos θ ^ 2)) + (1 / (2 - Real.sin θ ^ 2)) ≥ 4 / 3 :=
sorry

theorem min_value_achievable :
  ∃ θ : Real, (1 / (2 - Real.cos θ ^ 2)) + (1 / (2 - Real.sin θ ^ 2)) = 4 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_min_value_achievable_l1020_102031


namespace NUMINAMATH_CALUDE_exists_solution_set_exists_a_range_l1020_102019

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x + 1| + |2*x - a| + a

-- Theorem for part (Ⅰ)
theorem exists_solution_set (a : ℝ) (h : a = 3) :
  ∃ S : Set ℝ, ∀ x ∈ S, f x a > 7 :=
sorry

-- Theorem for part (Ⅱ)
theorem exists_a_range :
  ∃ a_min a_max : ℝ, ∀ a : ℝ, a_min ≤ a ∧ a ≤ a_max →
    ∀ x : ℝ, f x a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_exists_solution_set_exists_a_range_l1020_102019


namespace NUMINAMATH_CALUDE_parabola_vertex_l1020_102007

-- Define the parabola function
def f (x : ℝ) : ℝ := 3 * (x + 4)^2 - 9

-- State the theorem
theorem parabola_vertex :
  ∃ (x y : ℝ), (∀ t : ℝ, f t ≥ f x) ∧ f x = y ∧ x = -4 ∧ y = -9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1020_102007


namespace NUMINAMATH_CALUDE_age_difference_proof_l1020_102096

/-- Given two persons with an age difference of 16 years, where the elder is currently 30 years old,
    this theorem proves that 6 years ago, the elder person was three times as old as the younger one. -/
theorem age_difference_proof :
  ∀ (younger_age elder_age : ℕ) (years_ago : ℕ),
    elder_age = 30 →
    elder_age = younger_age + 16 →
    elder_age - years_ago = 3 * (younger_age - years_ago) →
    years_ago = 6 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1020_102096


namespace NUMINAMATH_CALUDE_complex_calculation_l1020_102070

def a : ℂ := 3 + 2*Complex.I
def b : ℂ := 1 - 2*Complex.I

theorem complex_calculation : 3*a - 4*b = 5 + 14*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l1020_102070


namespace NUMINAMATH_CALUDE_potato_slab_length_l1020_102064

/-- The length of the original uncut potato slab given the lengths of its two pieces -/
theorem potato_slab_length 
  (piece1 : ℕ) 
  (piece2 : ℕ) 
  (h1 : piece1 = 275)
  (h2 : piece2 = piece1 + 50) : 
  piece1 + piece2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_potato_slab_length_l1020_102064


namespace NUMINAMATH_CALUDE_max_rectangle_area_l1020_102052

def is_valid_rectangle (l w : ℕ) : Prop :=
  l + w = 20 ∧ l ≥ w + 3

def rectangle_area (l w : ℕ) : ℕ :=
  l * w

theorem max_rectangle_area :
  ∃ (l w : ℕ), is_valid_rectangle l w ∧
    rectangle_area l w = 91 ∧
    ∀ (l' w' : ℕ), is_valid_rectangle l' w' →
      rectangle_area l' w' ≤ 91 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l1020_102052


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l1020_102036

/-- Calculate the gain percent on a scooter sale given the purchase price, repair costs, and selling price. -/
theorem scooter_gain_percent 
  (purchase_price : ℝ)
  (repair_costs : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 800)
  (h2 : repair_costs = 200)
  (h3 : selling_price = 1400) :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 40 := by
sorry


end NUMINAMATH_CALUDE_scooter_gain_percent_l1020_102036


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1020_102054

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + 6*x₁ - 7 = 0) ∧ 
  (x₂^2 + 6*x₂ - 7 = 0) ∧ 
  x₁ = -7 ∧ 
  x₂ = 1 :=
by
  sorry

#check quadratic_equation_solution

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1020_102054


namespace NUMINAMATH_CALUDE_uno_card_discount_l1020_102037

theorem uno_card_discount (original_price : ℝ) (num_cards : ℕ) (total_paid : ℝ) : 
  original_price = 12 → num_cards = 10 → total_paid = 100 → 
  (original_price * num_cards - total_paid) / num_cards = 2 := by
  sorry

end NUMINAMATH_CALUDE_uno_card_discount_l1020_102037


namespace NUMINAMATH_CALUDE_ant_meeting_point_l1020_102014

/-- Triangle XYZ with given side lengths -/
structure Triangle where
  xy : ℝ
  yz : ℝ
  xz : ℝ

/-- Point P where ants meet -/
def MeetingPoint (t : Triangle) : ℝ := sorry

/-- Theorem stating that YP = 5 in the given triangle -/
theorem ant_meeting_point (t : Triangle) 
  (h_xy : t.xy = 5) 
  (h_yz : t.yz = 7) 
  (h_xz : t.xz = 8) : 
  MeetingPoint t = 5 := by sorry

end NUMINAMATH_CALUDE_ant_meeting_point_l1020_102014


namespace NUMINAMATH_CALUDE_fractions_product_one_l1020_102082

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def irreducible (n d : ℕ) : Prop := Nat.gcd n d = 1

def valid_fraction (n d : ℕ) : Prop :=
  n ∈ S ∧ d ∈ S ∧ n ≠ d ∧ irreducible n d

theorem fractions_product_one :
  ∃ (n₁ d₁ n₂ d₂ n₃ d₃ : ℕ),
    valid_fraction n₁ d₁ ∧
    valid_fraction n₂ d₂ ∧
    valid_fraction n₃ d₃ ∧
    n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₁ ≠ d₂ ∧ n₁ ≠ d₃ ∧
    n₂ ≠ n₃ ∧ n₂ ≠ d₁ ∧ n₂ ≠ d₃ ∧
    n₃ ≠ d₁ ∧ n₃ ≠ d₂ ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧
    d₂ ≠ d₃ ∧
    (n₁ : ℚ) / d₁ * (n₂ : ℚ) / d₂ * (n₃ : ℚ) / d₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_fractions_product_one_l1020_102082


namespace NUMINAMATH_CALUDE_cube_construction_count_l1020_102021

/-- Represents the group of rotations for a 3x3x3 cube -/
def CubeRotations : Type := Unit

/-- The number of elements in the group of rotations for a 3x3x3 cube -/
def rotationGroupSize : ℕ := 27

/-- The total number of ways to arrange 13 white cubes in a 3x3x3 cube -/
def totalArrangements : ℕ := 10400600

/-- The estimated number of fixed points for non-identity rotations -/
def fixedPointsNonIdentity : ℕ := 1000

/-- The total number of fixed points across all rotations -/
def totalFixedPoints : ℕ := totalArrangements + fixedPointsNonIdentity

/-- The number of distinct ways to construct the 3x3x3 cube -/
def distinctConstructions : ℕ := totalFixedPoints / rotationGroupSize

theorem cube_construction_count :
  distinctConstructions = 385244 := by sorry

end NUMINAMATH_CALUDE_cube_construction_count_l1020_102021


namespace NUMINAMATH_CALUDE_all_propositions_true_l1020_102066

-- Proposition 1
def expanded_terms (a b c d p q r m n : ℕ) : ℕ := 24

-- Proposition 2
def five_digit_numbers : ℕ := 36

-- Proposition 3
def seating_arrangements : ℕ := 24

-- Proposition 4
def odd_coefficients (x : ℝ) : ℕ := 2

theorem all_propositions_true :
  (∀ a b c d p q r m n, expanded_terms a b c d p q r m n = 24) ∧
  (five_digit_numbers = 36) ∧
  (seating_arrangements = 24) ∧
  (∀ x, odd_coefficients x = 2) :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_true_l1020_102066


namespace NUMINAMATH_CALUDE_binary_110101_equals_53_l1020_102030

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldr (λ (i, b) acc => acc + if b then 2^i else 0) 0

theorem binary_110101_equals_53 :
  binary_to_decimal [true, false, true, false, true, true] = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_equals_53_l1020_102030


namespace NUMINAMATH_CALUDE_intersection_point_slope_l1020_102035

/-- Given three lines in a plane, if two of them intersect at a point on the third line, 
    then the slope of one of the intersecting lines is 4. -/
theorem intersection_point_slope (k : ℝ) : 
  (∃ x y : ℝ, y = -2*x + 4 ∧ y = k*x ∧ y = x + 2) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_slope_l1020_102035


namespace NUMINAMATH_CALUDE_bridge_length_is_954_l1020_102012

/-- The length of a bridge given train parameters -/
def bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) : ℝ :=
  train_speed * crossing_time - train_length

/-- Theorem: The length of the bridge is 954 meters -/
theorem bridge_length_is_954 :
  bridge_length 90 36 29 = 954 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_is_954_l1020_102012


namespace NUMINAMATH_CALUDE_weighted_average_girl_scouts_permission_l1020_102057

-- Define the structure for each trip
structure Trip where
  total_scouts : ℕ
  boy_scout_percentage : ℚ
  girl_scout_percentage : ℚ
  boy_scout_permission_percentage : ℚ
  girl_scout_participation_percentage : ℚ
  girl_scout_permission_percentage : ℚ

-- Define the three trips
def trip1 : Trip := {
  total_scouts := 100,
  boy_scout_percentage := 60/100,
  girl_scout_percentage := 40/100,
  boy_scout_permission_percentage := 75/100,
  girl_scout_participation_percentage := 50/100,
  girl_scout_permission_percentage := 50/100
}

def trip2 : Trip := {
  total_scouts := 150,
  boy_scout_percentage := 50/100,
  girl_scout_percentage := 50/100,
  boy_scout_permission_percentage := 80/100,
  girl_scout_participation_percentage := 70/100,
  girl_scout_permission_percentage := 60/100
}

def trip3 : Trip := {
  total_scouts := 200,
  boy_scout_percentage := 40/100,
  girl_scout_percentage := 60/100,
  boy_scout_permission_percentage := 85/100,
  girl_scout_participation_percentage := 100/100,
  girl_scout_permission_percentage := 75/100
}

-- Function to calculate the number of Girl Scouts with permission slips for a trip
def girl_scouts_with_permission (trip : Trip) : ℚ :=
  trip.total_scouts * trip.girl_scout_percentage * trip.girl_scout_participation_percentage * trip.girl_scout_permission_percentage

-- Function to calculate the total number of participating Girl Scouts for a trip
def participating_girl_scouts (trip : Trip) : ℚ :=
  trip.total_scouts * trip.girl_scout_percentage * trip.girl_scout_participation_percentage

-- Theorem statement
theorem weighted_average_girl_scouts_permission (ε : ℚ) (h : ε > 0) :
  let total_with_permission := girl_scouts_with_permission trip1 + girl_scouts_with_permission trip2 + girl_scouts_with_permission trip3
  let total_participating := participating_girl_scouts trip1 + participating_girl_scouts trip2 + participating_girl_scouts trip3
  let weighted_average := total_with_permission / total_participating * 100
  |weighted_average - 68| < ε :=
by sorry

end NUMINAMATH_CALUDE_weighted_average_girl_scouts_permission_l1020_102057


namespace NUMINAMATH_CALUDE_perpendicular_travel_time_l1020_102083

theorem perpendicular_travel_time 
  (adam_speed : ℝ) 
  (simon_speed : ℝ) 
  (distance : ℝ) 
  (h1 : adam_speed = 10)
  (h2 : simon_speed = 5)
  (h3 : distance = 75) :
  ∃ (time : ℝ), 
    time = 3 * Real.sqrt 5 ∧ 
    distance^2 = (adam_speed * time)^2 + (simon_speed * time)^2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_travel_time_l1020_102083


namespace NUMINAMATH_CALUDE_carrie_tshirt_purchase_l1020_102048

def tshirt_cost : ℚ := 965 / 100  -- $9.65 represented as a rational number
def number_of_tshirts : ℕ := 12

def total_cost : ℚ := tshirt_cost * number_of_tshirts

theorem carrie_tshirt_purchase :
  total_cost = 11580 / 100 := by sorry

end NUMINAMATH_CALUDE_carrie_tshirt_purchase_l1020_102048


namespace NUMINAMATH_CALUDE_square_root_problem_l1020_102023

theorem square_root_problem (a b : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (a + 3)^2 = x ∧ (2*a - 6)^2 = x) →
  ((-2)^3 = b) →
  (∃ (y : ℝ), y^2 = a - b ∧ (y = 3 ∨ y = -3)) :=
by sorry

end NUMINAMATH_CALUDE_square_root_problem_l1020_102023


namespace NUMINAMATH_CALUDE_set_operations_l1020_102040

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x | -5 < x ∧ x ≤ -1}) ∧
  (A ∪ (U \ B) = {x | -5 < x ∧ x < 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1020_102040


namespace NUMINAMATH_CALUDE_jason_grew_37_watermelons_l1020_102063

/-- The number of watermelons Jason grew -/
def jason_watermelons : ℕ := 37

/-- The number of watermelons Sandy grew -/
def sandy_watermelons : ℕ := 11

/-- The total number of watermelons grown by Jason and Sandy -/
def total_watermelons : ℕ := 48

/-- Theorem stating that Jason grew 37 watermelons -/
theorem jason_grew_37_watermelons : jason_watermelons = total_watermelons - sandy_watermelons := by
  sorry

end NUMINAMATH_CALUDE_jason_grew_37_watermelons_l1020_102063


namespace NUMINAMATH_CALUDE_dogwood_trees_planted_today_l1020_102033

/-- The number of dogwood trees planted today in the park. -/
def trees_planted_today : ℕ := sorry

/-- The current number of dogwood trees in the park. -/
def current_trees : ℕ := 7

/-- The number of dogwood trees to be planted tomorrow. -/
def trees_planted_tomorrow : ℕ := 2

/-- The total number of dogwood trees after planting is finished. -/
def total_trees : ℕ := 12

theorem dogwood_trees_planted_today :
  trees_planted_today = 3 :=
by
  have h : current_trees + trees_planted_today + trees_planted_tomorrow = total_trees := sorry
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_today_l1020_102033


namespace NUMINAMATH_CALUDE_binary_sum_equals_expected_l1020_102097

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

def b1 : List Bool := [false, true, true, false, true]  -- 10110₂
def b2 : List Bool := [false, true, true]               -- 110₂
def b3 : List Bool := [true]                            -- 1₂
def b4 : List Bool := [true, false, true]               -- 101₂

def expected_sum : List Bool := [false, false, false, false, true, true]  -- 110000₂

theorem binary_sum_equals_expected :
  nat_to_binary (binary_to_nat b1 + binary_to_nat b2 + binary_to_nat b3 + binary_to_nat b4) = expected_sum :=
by sorry

end NUMINAMATH_CALUDE_binary_sum_equals_expected_l1020_102097


namespace NUMINAMATH_CALUDE_g_of_3_l1020_102047

theorem g_of_3 (g : ℝ → ℝ) :
  (∀ x, g x = (x^2 + 1) / (4*x - 5)) →
  g 3 = 10/7 := by
sorry

end NUMINAMATH_CALUDE_g_of_3_l1020_102047


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l1020_102094

theorem similar_triangles_leg_sum (a b c A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → A > 0 → B > 0 → C > 0 →
  (1/2) * a * b = 6 →
  (1/2) * A * B = 150 →
  c = 5 →
  a^2 + b^2 = c^2 →
  A^2 + B^2 = C^2 →
  (a/A)^2 = (b/B)^2 →
  (a/A)^2 = (c/C)^2 →
  A + B = 35 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l1020_102094


namespace NUMINAMATH_CALUDE_select_four_from_seven_l1020_102075

theorem select_four_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 4 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_select_four_from_seven_l1020_102075


namespace NUMINAMATH_CALUDE_existence_of_irrational_powers_with_integer_result_l1020_102090

theorem existence_of_irrational_powers_with_integer_result :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Irrational a ∧ Irrational b ∧ ∃ (n : ℤ), a^b = n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_irrational_powers_with_integer_result_l1020_102090


namespace NUMINAMATH_CALUDE_cube_with_cut_corners_has_44_edges_l1020_102093

/-- A cube with cut corners is a polyhedron obtained by cutting off each corner of a cube
    such that no two cutting planes intersect within the cube, and each corner cut
    removes a vertex and replaces it with a quadrilateral face. -/
structure CubeWithCutCorners where
  /-- The number of vertices in the original cube -/
  original_vertices : ℕ
  /-- The number of edges in the original cube -/
  original_edges : ℕ
  /-- The number of new edges introduced by each corner cut -/
  new_edges_per_cut : ℕ
  /-- The condition that the original shape is a cube -/
  is_cube : original_vertices = 8 ∧ original_edges = 12
  /-- The condition that each corner cut introduces 4 new edges -/
  corner_cut : new_edges_per_cut = 4

/-- The number of edges in the resulting figure after cutting off all corners of a cube -/
def num_edges_after_cuts (c : CubeWithCutCorners) : ℕ :=
  c.original_edges + c.original_vertices * c.new_edges_per_cut

/-- Theorem stating that a cube with cut corners has 44 edges -/
theorem cube_with_cut_corners_has_44_edges (c : CubeWithCutCorners) :
  num_edges_after_cuts c = 44 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_cut_corners_has_44_edges_l1020_102093


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1020_102042

/-- The length of the major axis of an ellipse with given foci and tangent to x-axis -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ : ℝ × ℝ),
  F₁ = (2, 10) →
  F₂ = (26, 35) →
  (∃ (X : ℝ), (X, 0) ∈ E) →
  (∀ (P : ℝ × ℝ), P ∈ E ↔ 
    ∃ (k : ℝ), dist P F₁ + dist P F₂ = k ∧ 
    ∀ (Q : ℝ × ℝ), Q ∈ E → dist Q F₁ + dist Q F₂ = k) →
  ∃ (A B : ℝ × ℝ), A ∈ E ∧ B ∈ E ∧ dist A B = 102 ∧
    ∀ (P Q : ℝ × ℝ), P ∈ E → Q ∈ E → dist P Q ≤ 102 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1020_102042


namespace NUMINAMATH_CALUDE_min_w_for_max_sin_l1020_102006

theorem min_w_for_max_sin (y : ℝ → ℝ) (w : ℝ) : 
  (∀ x, y x = Real.sin (w * x)) →  -- Condition 1
  w > 0 →  -- Condition 2
  (∃ n : ℕ, n ≥ 50 ∧ ∀ i : ℕ, i < n → ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ y x = 1) →  -- Condition 3
  w ≥ Real.pi * 100 :=  -- Conclusion
by sorry

end NUMINAMATH_CALUDE_min_w_for_max_sin_l1020_102006


namespace NUMINAMATH_CALUDE_adam_apple_purchase_l1020_102087

/-- The quantity of apples Adam bought on Monday -/
def monday_apples : ℕ := 15

/-- The quantity of apples Adam bought on Tuesday -/
def tuesday_apples : ℕ := 3 * monday_apples

/-- The quantity of apples Adam bought on Wednesday -/
def wednesday_apples : ℕ := 4 * tuesday_apples

/-- The total quantity of apples Adam bought on these three days -/
def total_apples : ℕ := monday_apples + tuesday_apples + wednesday_apples

theorem adam_apple_purchase : total_apples = 240 := by
  sorry

end NUMINAMATH_CALUDE_adam_apple_purchase_l1020_102087


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l1020_102085

theorem reciprocal_of_sum (y : ℚ) : y = 6 + 1/6 → 1/y = 6/37 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l1020_102085


namespace NUMINAMATH_CALUDE_triangle_properties_l1020_102049

theorem triangle_properties (a b c : ℝ) (h_ratio : (a, b, c) = (3, 4, 5)) 
  (h_perimeter : a + b + c = 36) : 
  (a^2 + b^2 = c^2) ∧ (a * b / 2 = 54) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1020_102049


namespace NUMINAMATH_CALUDE_mathematics_partition_ways_l1020_102027

/-- Represents the word "MATHEMATICS" -/
def word : String := "MATHEMATICS"

/-- The positions of vowels in the word -/
def vowel_positions : List Nat := [2, 5, 7, 9]

/-- The number of vowels in the word -/
def num_vowels : Nat := vowel_positions.length

/-- A function to calculate the number of partition ways -/
def num_partition_ways : Nat := 4 * 3 * 3

/-- Theorem stating that the number of ways to partition the word "MATHEMATICS" 
    such that each part contains at least one vowel is 36 -/
theorem mathematics_partition_ways :
  num_partition_ways = 36 := by sorry

end NUMINAMATH_CALUDE_mathematics_partition_ways_l1020_102027


namespace NUMINAMATH_CALUDE_scavenger_hunting_students_l1020_102045

theorem scavenger_hunting_students (total : ℕ) (skiing : ℕ → ℕ) (scavenger : ℕ) :
  total = 12000 →
  skiing scavenger = 2 * scavenger →
  total = skiing scavenger + scavenger →
  scavenger = 4000 := by
sorry

end NUMINAMATH_CALUDE_scavenger_hunting_students_l1020_102045


namespace NUMINAMATH_CALUDE_solve_equation_l1020_102001

theorem solve_equation : ∃ y : ℚ, 2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1020_102001


namespace NUMINAMATH_CALUDE_angle_T_measure_l1020_102058

/-- Represents a heptagon with specific angle properties -/
structure Heptagon :=
  (G E O M Y J R T : ℝ)
  (sum_angles : G + E + O + M + Y + J + R + T = 900)
  (equal_angles : G = E ∧ E = T ∧ T = R)
  (supplementary_M_Y : M + Y = 180)
  (supplementary_J_O : J + O = 180)

/-- The measure of angle T in the specified heptagon is 135° -/
theorem angle_T_measure (h : Heptagon) : h.T = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_T_measure_l1020_102058


namespace NUMINAMATH_CALUDE_jelly_beans_solution_l1020_102028

/-- The number of jelly beans in jar Y -/
def jelly_beans_Y : ℕ := sorry

/-- The number of jelly beans in jar X -/
def jelly_beans_X : ℕ := 3 * jelly_beans_Y - 400

/-- The total number of jelly beans -/
def total_jelly_beans : ℕ := 1200

theorem jelly_beans_solution :
  jelly_beans_X + jelly_beans_Y = total_jelly_beans ∧ jelly_beans_Y = 400 := by sorry

end NUMINAMATH_CALUDE_jelly_beans_solution_l1020_102028


namespace NUMINAMATH_CALUDE_john_racecar_earnings_l1020_102025

/-- The amount of money John made from his racecar after one race -/
def money_made (initial_cost maintenance_cost : ℝ) (discount prize_percent : ℝ) (prize : ℝ) : ℝ :=
  prize * prize_percent - initial_cost * (1 - discount) - maintenance_cost

/-- Theorem stating the amount John made from his racecar -/
theorem john_racecar_earnings (x : ℝ) :
  money_made 20000 x 0.2 0.9 70000 = 47000 - x := by
  sorry

end NUMINAMATH_CALUDE_john_racecar_earnings_l1020_102025


namespace NUMINAMATH_CALUDE_circle_line_bisection_implies_mn_range_l1020_102098

/-- The circle equation: x^2 + y^2 - 4x - 2y - 4 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 4 = 0

/-- The line equation: mx + 2ny - 4 = 0 -/
def line_equation (m n x y : ℝ) : Prop :=
  m*x + 2*n*y - 4 = 0

/-- The line bisects the perimeter of the circle -/
def line_bisects_circle (m n : ℝ) : Prop :=
  ∀ x y, circle_equation x y → line_equation m n x y

/-- The range of mn is (-∞, 1] -/
def mn_range (m n : ℝ) : Prop :=
  m * n ≤ 1

theorem circle_line_bisection_implies_mn_range :
  ∀ m n, line_bisects_circle m n → mn_range m n :=
sorry

end NUMINAMATH_CALUDE_circle_line_bisection_implies_mn_range_l1020_102098


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l1020_102017

theorem floor_abs_negative_real : ⌊|(-56.7 : ℝ)|⌋ = 56 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l1020_102017


namespace NUMINAMATH_CALUDE_smallest_m_dividing_power_minus_one_l1020_102009

theorem smallest_m_dividing_power_minus_one :
  ∃ (m : ℕ+), (2^1990 : ℕ) ∣ (1989^(m : ℕ) - 1) ∧
    ∀ (k : ℕ+), (2^1990 : ℕ) ∣ (1989^(k : ℕ) - 1) → m ≤ k :=
by
  use 2^1988
  sorry

end NUMINAMATH_CALUDE_smallest_m_dividing_power_minus_one_l1020_102009


namespace NUMINAMATH_CALUDE_balloon_arrangements_count_l1020_102038

/-- The number of distinct arrangements of letters in "balloon" -/
def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating the number of distinct arrangements of letters in "balloon" -/
theorem balloon_arrangements_count : balloon_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_count_l1020_102038


namespace NUMINAMATH_CALUDE_hollow_square_students_l1020_102026

/-- Represents a hollow square formation of students -/
structure HollowSquare where
  outer_layer : Nat
  inner_layer : Nat

/-- Calculates the total number of students in a hollow square formation -/
def total_students (hs : HollowSquare) : Nat :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that a hollow square with 52 in the outer layer and 28 in the inner layer has 160 students total -/
theorem hollow_square_students :
  let hs : HollowSquare := { outer_layer := 52, inner_layer := 28 }
  total_students hs = 160 := by
  sorry

end NUMINAMATH_CALUDE_hollow_square_students_l1020_102026


namespace NUMINAMATH_CALUDE_find_multiplier_l1020_102046

theorem find_multiplier : ∃ (m : ℕ), 
  220050 = m * (555 - 445) * (555 + 445) + 50 ∧ 
  m * (555 - 445) = 220050 / (555 + 445) :=
by sorry

end NUMINAMATH_CALUDE_find_multiplier_l1020_102046


namespace NUMINAMATH_CALUDE_fraction_reduction_l1020_102091

theorem fraction_reduction (a b d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hd : d ≠ 0) (hsum : a + b + d ≠ 0) :
  (a^2 + b^2 - d^2 + 2*a*b) / (a^2 + d^2 - b^2 + 2*a*d) = (a + b - d) / (a + d - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_reduction_l1020_102091


namespace NUMINAMATH_CALUDE_memory_card_capacity_l1020_102081

/-- Proves that a memory card with capacity for 3,000 pictures of 8 megabytes
    can hold 4,000 pictures of 6 megabytes -/
theorem memory_card_capacity 
  (initial_count : Nat) 
  (initial_size : Nat) 
  (new_size : Nat) 
  (h1 : initial_count = 3000)
  (h2 : initial_size = 8)
  (h3 : new_size = 6) :
  (initial_count * initial_size) / new_size = 4000 :=
by
  sorry

end NUMINAMATH_CALUDE_memory_card_capacity_l1020_102081


namespace NUMINAMATH_CALUDE_goods_train_length_l1020_102065

/-- The length of a goods train passing a man in another train --/
theorem goods_train_length (man_train_speed goods_train_speed : ℝ) 
  (passing_time : ℝ) : 
  man_train_speed = 36 →
  goods_train_speed = 50.4 →
  passing_time = 10 →
  (man_train_speed + goods_train_speed) * (1000 / 3600) * passing_time = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_goods_train_length_l1020_102065
