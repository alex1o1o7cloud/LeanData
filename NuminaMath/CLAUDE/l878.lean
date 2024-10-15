import Mathlib

namespace NUMINAMATH_CALUDE_arrangements_count_l878_87809

/-- Number of red flags -/
def red_flags : ℕ := 8

/-- Number of white flags -/
def white_flags : ℕ := 8

/-- Number of black flags -/
def black_flags : ℕ := 1

/-- Total number of flags -/
def total_flags : ℕ := red_flags + white_flags + black_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Function to calculate the number of distinguishable arrangements -/
def count_arrangements (r w b p : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of distinguishable arrangements is 315 -/
theorem arrangements_count :
  count_arrangements red_flags white_flags black_flags flagpoles = 315 :=
sorry

end NUMINAMATH_CALUDE_arrangements_count_l878_87809


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l878_87816

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4 * Complex.I) * z = Complex.abs (4 + 3 * Complex.I)) :
  z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l878_87816


namespace NUMINAMATH_CALUDE_limit_f_at_infinity_l878_87833

noncomputable def f (x : ℝ) := (x - Real.sin x) / (x + Real.sin x)

theorem limit_f_at_infinity :
  ∀ ε > 0, ∃ N : ℝ, ∀ x ≥ N, |f x - 1| < ε :=
by
  sorry

/- Assumptions:
   1. x is a real number (implied by the use of ℝ)
   2. sin x is bounded between -1 and 1 (this is a property of sine in Mathlib)
-/

end NUMINAMATH_CALUDE_limit_f_at_infinity_l878_87833


namespace NUMINAMATH_CALUDE_complex_modulus_example_l878_87897

theorem complex_modulus_example : Complex.abs (7/4 + 3*I) = Real.sqrt 193 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l878_87897


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_for_800_l878_87828

def isPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def sumOfDistinctPowersOfTwo (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.map (λ k => 2^k)).sum = n ∧ powers.Nodup

theorem least_sum_of_exponents_for_800 :
  ∀ (powers : List ℕ),
    sumOfDistinctPowersOfTwo 800 powers →
    powers.length ≥ 3 →
    powers.sum ≥ 22 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_for_800_l878_87828


namespace NUMINAMATH_CALUDE_tony_age_proof_l878_87893

/-- Represents Tony's age at the beginning of the work period -/
def initial_age : ℕ := 10

/-- Represents the number of days Tony worked -/
def work_days : ℕ := 80

/-- Represents Tony's daily work hours -/
def daily_hours : ℕ := 3

/-- Represents Tony's base hourly wage in cents -/
def base_wage : ℕ := 75

/-- Represents the age-based hourly wage increase in cents -/
def age_wage_increase : ℕ := 25

/-- Represents Tony's total earnings in cents -/
def total_earnings : ℕ := 84000

/-- Theorem stating that the given initial age satisfies the problem conditions -/
theorem tony_age_proof :
  ∃ (x : ℕ), x ≤ work_days ∧
  (daily_hours * (base_wage + age_wage_increase * initial_age) * x +
   daily_hours * (base_wage + age_wage_increase * (initial_age + 1)) * (work_days - x) =
   total_earnings) :=
sorry

end NUMINAMATH_CALUDE_tony_age_proof_l878_87893


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l878_87807

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 2) ^ (1/3 : ℝ) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l878_87807


namespace NUMINAMATH_CALUDE_cubic_inequality_l878_87896

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 > -36*x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l878_87896


namespace NUMINAMATH_CALUDE_fraction_problem_l878_87887

theorem fraction_problem (x y : ℚ) 
  (h1 : y / (x - 1) = 1 / 3)
  (h2 : (y + 4) / x = 1 / 2) :
  y / x = 7 / 22 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l878_87887


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l878_87812

theorem smallest_gcd_multiple (p q : ℕ+) (h : Nat.gcd p q = 15) :
  (∀ p' q' : ℕ+, Nat.gcd p' q' = 15 → Nat.gcd (8 * p') (18 * q') ≥ 30) ∧
  (∃ p' q' : ℕ+, Nat.gcd p' q' = 15 ∧ Nat.gcd (8 * p') (18 * q') = 30) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l878_87812


namespace NUMINAMATH_CALUDE_sport_participation_theorem_l878_87823

/-- Represents the number of students who play various sports in a class -/
structure SportParticipation where
  total_students : ℕ
  basketball : ℕ
  cricket : ℕ
  baseball : ℕ
  basketball_cricket : ℕ
  cricket_baseball : ℕ
  basketball_baseball : ℕ
  all_three : ℕ

/-- Calculates the number of students who play at least one sport -/
def students_playing_at_least_one_sport (sp : SportParticipation) : ℕ :=
  sp.basketball + sp.cricket + sp.baseball - sp.basketball_cricket - sp.cricket_baseball - sp.basketball_baseball + sp.all_three

/-- Calculates the number of students who don't play any sport -/
def students_not_playing_any_sport (sp : SportParticipation) : ℕ :=
  sp.total_students - students_playing_at_least_one_sport sp

/-- Theorem stating the correct number of students playing at least one sport and not playing any sport -/
theorem sport_participation_theorem (sp : SportParticipation) 
  (h1 : sp.total_students = 40)
  (h2 : sp.basketball = 15)
  (h3 : sp.cricket = 20)
  (h4 : sp.baseball = 12)
  (h5 : sp.basketball_cricket = 5)
  (h6 : sp.cricket_baseball = 7)
  (h7 : sp.basketball_baseball = 3)
  (h8 : sp.all_three = 2) :
  students_playing_at_least_one_sport sp = 32 ∧ students_not_playing_any_sport sp = 8 := by
  sorry

end NUMINAMATH_CALUDE_sport_participation_theorem_l878_87823


namespace NUMINAMATH_CALUDE_intersection_equals_zero_one_l878_87837

def A : Set ℕ := {0, 1, 2, 3}

def B : Set ℕ := {x | ∃ n ∈ A, x = n^2}

def P : Set ℕ := A ∩ B

theorem intersection_equals_zero_one : P = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_zero_one_l878_87837


namespace NUMINAMATH_CALUDE_kitten_weight_l878_87829

theorem kitten_weight (k r p : ℝ) 
  (total_weight : k + r + p = 38)
  (kitten_rabbit_weight : k + r = 3 * p)
  (kitten_parrot_weight : k + p = r) :
  k = 9.5 := by
sorry

end NUMINAMATH_CALUDE_kitten_weight_l878_87829


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2016_n_193_divisible_by_2016_smallest_n_is_193_l878_87888

theorem smallest_n_divisible_by_2016 :
  ∀ n : ℕ, n > 1 → (3 * n^3 + 2013) % 2016 = 0 → n ≥ 193 :=
by sorry

theorem n_193_divisible_by_2016 :
  (3 * 193^3 + 2013) % 2016 = 0 :=
by sorry

theorem smallest_n_is_193 :
  ∃! n : ℕ, n > 1 ∧ (3 * n^3 + 2013) % 2016 = 0 ∧
  ∀ m : ℕ, m > 1 → (3 * m^3 + 2013) % 2016 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2016_n_193_divisible_by_2016_smallest_n_is_193_l878_87888


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l878_87839

/-- Given two adjacent points (1,2) and (4,6) on a square, the area of the square is 25 -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let distance_squared := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
  let area := distance_squared
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l878_87839


namespace NUMINAMATH_CALUDE_price_increase_percentage_l878_87861

theorem price_increase_percentage (old_price new_price : ℝ) (h1 : old_price = 300) (h2 : new_price = 420) :
  ((new_price - old_price) / old_price) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l878_87861


namespace NUMINAMATH_CALUDE_concyclic_intersecting_lines_ratio_l878_87810

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the condition that A, B, C, D are concyclic
def concyclic (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the condition that lines (AB) and (CD) intersect at E
def intersect_at (A B C D E : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem concyclic_intersecting_lines_ratio 
  (h1 : concyclic A B C D) 
  (h2 : intersect_at A B C D E) :
  (distance A C / distance B C) * (distance A D / distance B D) = 
  distance A E / distance B E := by sorry

end NUMINAMATH_CALUDE_concyclic_intersecting_lines_ratio_l878_87810


namespace NUMINAMATH_CALUDE_rachel_homework_l878_87862

theorem rachel_homework (total_pages reading_pages biology_pages : ℕ) 
  (h1 : total_pages = 15)
  (h2 : reading_pages = 3)
  (h3 : biology_pages = 10) : 
  total_pages - reading_pages - biology_pages = 2 := by
sorry

end NUMINAMATH_CALUDE_rachel_homework_l878_87862


namespace NUMINAMATH_CALUDE_range_of_g_l878_87824

-- Define the function g(x)
def g (x : ℝ) : ℝ := 3 * (x + 5)

-- State the theorem
theorem range_of_g :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ 1 ∧ g x = y) ↔ y ≠ 18 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l878_87824


namespace NUMINAMATH_CALUDE_linear_function_proof_l878_87806

theorem linear_function_proof (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x + y) = f x + f y) -- linearity
  (h2 : ∀ x y : ℝ, x < y → f x < f y) -- monotonically increasing
  (h3 : ∀ x : ℝ, f (f x) = 16 * x + 9) : -- given condition
  ∀ x : ℝ, f x = 4 * x + 9/5 := by
sorry

end NUMINAMATH_CALUDE_linear_function_proof_l878_87806


namespace NUMINAMATH_CALUDE_sum_of_rotated_digits_l878_87803

theorem sum_of_rotated_digits : 
  2345 + 3452 + 4523 + 5234 + 3245 + 2453 + 4532 + 5324 = 8888 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rotated_digits_l878_87803


namespace NUMINAMATH_CALUDE_perfect_square_condition_l878_87882

theorem perfect_square_condition (a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + 2*(a+4)*x + 25 = (x + k)^2) → (a = 1 ∨ a = -9) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l878_87882


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l878_87800

/-- Proves that a parallelogram with area 288 sq m and altitude twice the base has a base length of 12 m -/
theorem parallelogram_base_length : 
  ∀ (base altitude : ℝ),
  base > 0 →
  altitude > 0 →
  altitude = 2 * base →
  base * altitude = 288 →
  base = 12 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l878_87800


namespace NUMINAMATH_CALUDE_prob_two_girls_prob_two_girls_five_l878_87848

/-- The probability of selecting two girls from a club with equal numbers of boys and girls -/
theorem prob_two_girls (n : ℕ) (h : n > 0) : 
  (Nat.choose n 2) / (Nat.choose (2*n) 2) = 2 / 9 :=
sorry

/-- The specific case for a club with 5 girls and 5 boys -/
theorem prob_two_girls_five : 
  (Nat.choose 5 2) / (Nat.choose 10 2) = 2 / 9 :=
sorry

end NUMINAMATH_CALUDE_prob_two_girls_prob_two_girls_five_l878_87848


namespace NUMINAMATH_CALUDE_illuminated_cube_surface_area_l878_87834

/-- The illuminated area of a cube's surface when a cylindrical beam of light is directed along its main diagonal --/
theorem illuminated_cube_surface_area
  (a : ℝ) -- Edge length of the cube
  (ρ : ℝ) -- Radius of the cylindrical beam
  (h1 : a = Real.sqrt (2 + Real.sqrt 2)) -- Given edge length
  (h2 : ρ = Real.sqrt 2) -- Given beam radius
  (h3 : ρ > 0) -- Positive radius
  (h4 : a > 0) -- Positive edge length
  : Real.sqrt 3 * π / 2 + 3 * Real.sqrt 6 = 
    (3 : ℝ) * π * ρ^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_illuminated_cube_surface_area_l878_87834


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l878_87841

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h : x₁^2 - x₁ - 1 = 0 ∧ x₂^2 - x₂ - 1 = 0) : 
  x₁ + x₂ = 1 ∧ x₁ * x₂ = -1 ∧ x₁^2 + x₂^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l878_87841


namespace NUMINAMATH_CALUDE_coin_problem_l878_87802

theorem coin_problem (x y : ℕ) : 
  x + y = 40 →  -- Total number of coins
  2 * x + 5 * y = 125 →  -- Total amount of money
  y = 15  -- Number of 5-dollar coins
  := by sorry

end NUMINAMATH_CALUDE_coin_problem_l878_87802


namespace NUMINAMATH_CALUDE_magic_ink_combinations_l878_87842

/-- The number of valid combinations for a magic ink recipe. -/
def validCombinations (herbTypes : ℕ) (essenceTypes : ℕ) (incompatibleHerbs : ℕ) : ℕ :=
  herbTypes * essenceTypes - incompatibleHerbs

/-- Theorem stating that the number of valid combinations for the magic ink is 21. -/
theorem magic_ink_combinations :
  validCombinations 4 6 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_magic_ink_combinations_l878_87842


namespace NUMINAMATH_CALUDE_smallest_difference_l878_87835

/-- Represents the first sequence in the table -/
def first_sequence (n : ℕ) : ℤ := 2 * n - 1

/-- Represents the second sequence in the table -/
def second_sequence (n : ℕ) : ℤ := 5055 - 5 * n

/-- The difference between the two sequences at position n -/
def difference (n : ℕ) : ℤ := (second_sequence n) - (first_sequence n)

/-- The number of terms in each sequence -/
def sequence_length : ℕ := 1010

theorem smallest_difference :
  ∃ (k : ℕ), k ≤ sequence_length ∧ difference k = 2 ∧
  ∀ (n : ℕ), n ≤ sequence_length → difference n ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_l878_87835


namespace NUMINAMATH_CALUDE_smoothie_servings_calculation_l878_87838

/-- Calculates the number of smoothie servings that can be made given the volumes of ingredients and serving size. -/
def smoothie_servings (watermelon_puree : ℕ) (cream : ℕ) (serving_size : ℕ) : ℕ :=
  (watermelon_puree + cream) / serving_size

/-- Theorem: Given 500 ml of watermelon puree, 100 ml of cream, and a serving size of 150 ml, 4 servings of smoothie can be made. -/
theorem smoothie_servings_calculation :
  smoothie_servings 500 100 150 = 4 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_servings_calculation_l878_87838


namespace NUMINAMATH_CALUDE_smallest_number_l878_87884

/-- Convert a number from base 6 to decimal -/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Convert a number from base 4 to decimal -/
def base4ToDecimal (n : ℕ) : ℕ := sorry

/-- Convert a number from base 2 to decimal -/
def base2ToDecimal (n : ℕ) : ℕ := sorry

theorem smallest_number :
  let n1 := base6ToDecimal 210
  let n2 := base4ToDecimal 1000
  let n3 := base2ToDecimal 111111
  n3 < n1 ∧ n3 < n2 := by sorry

end NUMINAMATH_CALUDE_smallest_number_l878_87884


namespace NUMINAMATH_CALUDE_continuous_at_five_l878_87877

def f (x : ℝ) : ℝ := 4 * x^2 - 2

theorem continuous_at_five :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 5| < δ → |f x - f 5| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_continuous_at_five_l878_87877


namespace NUMINAMATH_CALUDE_radio_station_survey_l878_87856

theorem radio_station_survey (total_listeners total_non_listeners female_listeners male_non_listeners : ℕ)
  (h1 : total_listeners = 160)
  (h2 : total_non_listeners = 180)
  (h3 : female_listeners = 72)
  (h4 : male_non_listeners = 88) :
  total_listeners - female_listeners = 92 :=
by
  sorry

#check radio_station_survey

end NUMINAMATH_CALUDE_radio_station_survey_l878_87856


namespace NUMINAMATH_CALUDE_average_weight_problem_l878_87850

/-- Given three weights a, b, c, prove that if their average is 45,
    the average of a and b is 40, and b is 31, then the average of b and c is 43. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 31 →
  (b + c) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l878_87850


namespace NUMINAMATH_CALUDE_divisibility_by_nine_l878_87830

theorem divisibility_by_nine (A : ℕ) (h : A < 10) : 
  (7000 + 200 + 10 * A + 4) % 9 = 0 ↔ A = 5 := by sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_l878_87830


namespace NUMINAMATH_CALUDE_min_c_value_l878_87831

theorem min_c_value (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧  -- consecutive integers
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧  -- consecutive integers
  ∃ m : ℕ, b + c + d = m^2 ∧  -- b + c + d is a perfect square
  ∃ n : ℕ, a + b + c + d + e = n^3 ∧  -- a + b + c + d + e is a perfect cube
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' < b' ∧ b' < c' ∧ c' < d' ∧ d' < e' ∧
    b' = a' + 1 ∧ c' = b' + 1 ∧ d' = c' + 1 ∧ e' = d' + 1 ∧
    ∃ m' : ℕ, b' + c' + d' = m'^2 ∧
    ∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3) →
  c' ≥ c →
  c = 675 :=
sorry

end NUMINAMATH_CALUDE_min_c_value_l878_87831


namespace NUMINAMATH_CALUDE_constant_for_max_n_l878_87847

theorem constant_for_max_n (c : ℝ) : 
  (∀ n : ℤ, n ≤ 8 → c * n^2 ≤ 8100) ∧ 
  (c * 9^2 > 8100) ↔ 
  c = 126.5625 := by
sorry

end NUMINAMATH_CALUDE_constant_for_max_n_l878_87847


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l878_87881

/-- The equations of the asymptotes of the hyperbola x²/16 - y²/9 = 1 -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := λ x y => x^2/16 - y^2/9 = 1
  ∃ (f g : ℝ → ℝ), (∀ x, f x = (3/4) * x) ∧ (∀ x, g x = -(3/4) * x) ∧
    (∀ ε > 0, ∃ M > 0, ∀ x y, h x y → (|x| > M → |y - f x| < ε ∨ |y - g x| < ε)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l878_87881


namespace NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l878_87821

-- Define the repeating decimal 0.3333...
def repeating_decimal : ℚ := 1 / 3

-- Theorem statement
theorem divide_eight_by_repeating_third : 8 / repeating_decimal = 24 := by
  sorry

end NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l878_87821


namespace NUMINAMATH_CALUDE_exchange_rate_theorem_l878_87857

/-- Represents the number of boys in the group -/
def b : ℕ := sorry

/-- Represents the number of girls in the group -/
def g : ℕ := sorry

/-- Represents the exchange rate of yuan to alternative currency -/
def x : ℕ := sorry

/-- The total cost in yuan at the first disco -/
def first_disco_cost : ℕ := b * g

/-- The total cost in alternative currency at the second place -/
def second_place_cost : ℕ := (b + g) * (b + g - 1) + (b + g) + 1

/-- Theorem stating the exchange rate between yuan and alternative currency -/
theorem exchange_rate_theorem : 
  first_disco_cost * x = second_place_cost ∧ x = 5 :=
by sorry

end NUMINAMATH_CALUDE_exchange_rate_theorem_l878_87857


namespace NUMINAMATH_CALUDE_paige_dresser_capacity_l878_87832

/-- Represents the capacity of a dresser in pieces of clothing. -/
def dresser_capacity (pieces_per_drawer : ℕ) (num_drawers : ℕ) : ℕ :=
  pieces_per_drawer * num_drawers

/-- Theorem stating that a dresser with 8 drawers, each holding 5 pieces, has a total capacity of 40 pieces. -/
theorem paige_dresser_capacity :
  dresser_capacity 5 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_paige_dresser_capacity_l878_87832


namespace NUMINAMATH_CALUDE_consecutive_hits_theorem_l878_87826

/-- The number of ways to arrange 8 shots with 3 hits, where exactly 2 hits are consecutive -/
def consecutive_hits_arrangements (total_shots : ℕ) (total_hits : ℕ) (consecutive_hits : ℕ) : ℕ :=
  if total_shots = 8 ∧ total_hits = 3 ∧ consecutive_hits = 2 then
    30
  else
    0

/-- Theorem stating that the number of arrangements is 30 -/
theorem consecutive_hits_theorem :
  consecutive_hits_arrangements 8 3 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_hits_theorem_l878_87826


namespace NUMINAMATH_CALUDE_dividend_calculation_l878_87844

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 8) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 141 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l878_87844


namespace NUMINAMATH_CALUDE_profit_percentage_is_25_l878_87890

/-- 
Given that the cost price of 30 articles is equal to the selling price of 24 articles,
prove that the profit percentage is 25%.
-/
theorem profit_percentage_is_25 (C S : ℝ) (h : 30 * C = 24 * S) : 
  (S - C) / C * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_25_l878_87890


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l878_87840

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {2, 3, 4}

-- Define set B
def B : Set Nat := {1, 2}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l878_87840


namespace NUMINAMATH_CALUDE_xyz_negative_l878_87874

theorem xyz_negative (a b c x y z : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0)
  (h : |x - a| + |y - b| + |z - c| = 0) : 
  x * y * z < 0 := by
sorry

end NUMINAMATH_CALUDE_xyz_negative_l878_87874


namespace NUMINAMATH_CALUDE_advertising_sales_prediction_l878_87871

-- Define the relationship between advertising expenditure and sales revenue
def advertising_sales_relation (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 4

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ :=
  6.5 * x + 17.5

-- Theorem statement
theorem advertising_sales_prediction :
  ∀ x y : ℝ, advertising_sales_relation x y →
  (linear_regression 10 = 82.5) ∧
  (∀ x : ℝ, y = linear_regression x) :=
by sorry

end NUMINAMATH_CALUDE_advertising_sales_prediction_l878_87871


namespace NUMINAMATH_CALUDE_shoe_selection_probability_l878_87853

/-- The number of pairs of shoes in the cabinet -/
def num_pairs : ℕ := 3

/-- The total number of shoes in the cabinet -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes selected -/
def selected_shoes : ℕ := 2

/-- The probability of selecting two shoes that do not form a pair -/
def prob_not_pair : ℚ := 4/5

theorem shoe_selection_probability :
  (Nat.choose total_shoes selected_shoes - num_pairs) / Nat.choose total_shoes selected_shoes = prob_not_pair :=
sorry

end NUMINAMATH_CALUDE_shoe_selection_probability_l878_87853


namespace NUMINAMATH_CALUDE_point_coordinates_l878_87814

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of a 2D coordinate system -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: Given the conditions, prove that the point P has coordinates (-2, 5) -/
theorem point_coordinates (P : Point) 
  (h1 : SecondQuadrant P) 
  (h2 : DistanceToXAxis P = 5) 
  (h3 : DistanceToYAxis P = 2) : 
  P.x = -2 ∧ P.y = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l878_87814


namespace NUMINAMATH_CALUDE_cara_age_l878_87869

/-- Given the age relationships in Cara's family, prove Cara's age --/
theorem cara_age :
  ∀ (cara_age mom_age grandma_age : ℕ),
    cara_age = mom_age - 20 →
    mom_age = grandma_age - 15 →
    grandma_age = 75 →
    cara_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_cara_age_l878_87869


namespace NUMINAMATH_CALUDE_double_burger_cost_l878_87845

/-- Proves that the cost of a double burger is $1.50 given the specified conditions -/
theorem double_burger_cost (total_spent : ℚ) (total_hamburgers : ℕ) (double_burgers : ℕ) (single_burger_cost : ℚ) :
  total_spent = 70.5 ∧
  total_hamburgers = 50 ∧
  double_burgers = 41 ∧
  single_burger_cost = 1 →
  (total_spent - (total_hamburgers - double_burgers : ℚ) * single_burger_cost) / double_burgers = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_double_burger_cost_l878_87845


namespace NUMINAMATH_CALUDE_root_sum_cube_product_l878_87885

theorem root_sum_cube_product (α β : ℝ) : 
  α^2 - 2*α - 4 = 0 → β^2 - 2*β - 4 = 0 → α ≠ β → α^3 + 8*β + 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_cube_product_l878_87885


namespace NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l878_87870

def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 13
def both_drinkers : ℕ := 7

theorem businessmen_neither_coffee_nor_tea :
  total_businessmen - (coffee_drinkers + tea_drinkers - both_drinkers) = 9 :=
by sorry

end NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l878_87870


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l878_87858

theorem smallest_base_perfect_square : ∃ (b : ℕ), 
  b > 3 ∧ 
  (∃ (n : ℕ), n^2 = 2*b + 3 ∧ n^2 < 25) ∧
  (∀ (k : ℕ), k > 3 ∧ k < b → ¬∃ (m : ℕ), m^2 = 2*k + 3 ∧ m^2 < 25) ∧
  b = 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l878_87858


namespace NUMINAMATH_CALUDE_hoseok_flowers_left_l878_87825

/-- Calculates the number of flowers Hoseok has left after giving some away. -/
def flowers_left (initial : ℕ) (to_minyoung : ℕ) (to_yoojeong : ℕ) : ℕ :=
  initial - (to_minyoung + to_yoojeong)

/-- Theorem stating that Hoseok has 7 flowers left after giving some away. -/
theorem hoseok_flowers_left :
  flowers_left 18 5 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_flowers_left_l878_87825


namespace NUMINAMATH_CALUDE_h_is_correct_l878_87895

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := -9*x^3 - x^2 - 4*x + 3

-- State the theorem
theorem h_is_correct : 
  ∀ x : ℝ, 9*x^3 + 6*x^2 - 3*x + 1 + h x = 5*x^2 - 7*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_h_is_correct_l878_87895


namespace NUMINAMATH_CALUDE_biscuit_price_is_two_l878_87899

/-- Represents the bakery order problem --/
def bakery_order (quiche_price croissant_price biscuit_price : ℚ) : Prop :=
  let quiche_count : ℕ := 2
  let croissant_count : ℕ := 6
  let biscuit_count : ℕ := 6
  let discount_rate : ℚ := 1 / 10
  let discounted_total : ℚ := 54

  let original_total : ℚ := quiche_count * quiche_price + 
                            croissant_count * croissant_price + 
                            biscuit_count * biscuit_price

  let discounted_amount : ℚ := original_total * discount_rate
  
  (original_total > 50) ∧ 
  (original_total - discounted_amount = discounted_total) ∧
  (quiche_price = 15) ∧
  (croissant_price = 3) ∧
  (biscuit_price = 2)

/-- Theorem stating that the biscuit price is $2.00 --/
theorem biscuit_price_is_two :
  ∃ (quiche_price croissant_price biscuit_price : ℚ),
    bakery_order quiche_price croissant_price biscuit_price ∧
    biscuit_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_biscuit_price_is_two_l878_87899


namespace NUMINAMATH_CALUDE_second_company_base_rate_l878_87836

/-- The base rate of United Telephone in dollars -/
def united_base_rate : ℝ := 6

/-- The per-minute rate of United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The per-minute rate of the second telephone company in dollars -/
def second_per_minute : ℝ := 0.20

/-- The number of minutes used -/
def minutes_used : ℝ := 120

/-- The base rate of the second telephone company in dollars -/
def second_base_rate : ℝ := 12

theorem second_company_base_rate :
  united_base_rate + united_per_minute * minutes_used =
  second_base_rate + second_per_minute * minutes_used :=
by sorry

end NUMINAMATH_CALUDE_second_company_base_rate_l878_87836


namespace NUMINAMATH_CALUDE_hen_price_calculation_l878_87852

/-- Proves that given 5 goats and 10 hens with a total cost of 2500,
    and an average price of 400 per goat, the average price of a hen is 50. -/
theorem hen_price_calculation (num_goats num_hens total_cost goat_price : ℕ)
    (h1 : num_goats = 5)
    (h2 : num_hens = 10)
    (h3 : total_cost = 2500)
    (h4 : goat_price = 400) :
    (total_cost - num_goats * goat_price) / num_hens = 50 := by
  sorry

end NUMINAMATH_CALUDE_hen_price_calculation_l878_87852


namespace NUMINAMATH_CALUDE_unequal_grandchildren_probability_l878_87878

def num_grandchildren : ℕ := 12

theorem unequal_grandchildren_probability :
  let total_outcomes := 2^num_grandchildren
  let equal_split := Nat.choose num_grandchildren (num_grandchildren / 2)
  (total_outcomes - equal_split) / total_outcomes = 793 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_unequal_grandchildren_probability_l878_87878


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l878_87894

theorem complex_magnitude_product : |(7 + 6*I)*(-5 + 3*I)| = Real.sqrt 2890 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l878_87894


namespace NUMINAMATH_CALUDE_find_number_l878_87866

theorem find_number : ∃ x : ℚ, x * 9999 = 824777405 ∧ x = 82482.5 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l878_87866


namespace NUMINAMATH_CALUDE_subtraction_decimal_proof_l878_87883

theorem subtraction_decimal_proof :
  (12.358 : ℝ) - (7.2943 : ℝ) = 5.0637 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_decimal_proof_l878_87883


namespace NUMINAMATH_CALUDE_subtraction_of_like_terms_l878_87860

theorem subtraction_of_like_terms (a : ℝ) : 4 * a - 3 * a = a := by sorry

end NUMINAMATH_CALUDE_subtraction_of_like_terms_l878_87860


namespace NUMINAMATH_CALUDE_triangle_perimeter_l878_87815

theorem triangle_perimeter (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sum : |a + b - c| + |b + c - a| + |c + a - b| = 12) : 
  a + b + c = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l878_87815


namespace NUMINAMATH_CALUDE_gcd_of_N_is_12_l878_87886

def N (a b c d : ℕ) : ℤ :=
  (a - b) * (c - d) * (a - c) * (b - d) * (a - d) * (b - c)

theorem gcd_of_N_is_12 :
  ∃ (k : ℕ), ∀ (a b c d : ℕ), 
    (∃ (n : ℤ), N a b c d = 12 * n) ∧
    (∀ (m : ℕ), m > 12 → ¬(∃ (l : ℤ), N a b c d = m * l)) :=
sorry

end NUMINAMATH_CALUDE_gcd_of_N_is_12_l878_87886


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l878_87818

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (l₁ l₂ : Line) (α : Plane)
  (h₁ : l₁ ≠ l₂)  -- l₁ and l₂ are non-coincident
  (h₂ : perpendicular l₁ α)
  (h₃ : perpendicular l₂ α) :
  parallel l₁ l₂ :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l878_87818


namespace NUMINAMATH_CALUDE_fund_price_calculation_l878_87808

theorem fund_price_calculation (initial_price : ℝ) 
  (monday_change tuesday_change wednesday_change thursday_change friday_change : ℝ) :
  initial_price = 35 →
  monday_change = 4.5 →
  tuesday_change = 4 →
  wednesday_change = -1 →
  thursday_change = -2.5 →
  friday_change = -6 →
  initial_price + monday_change + tuesday_change + wednesday_change + thursday_change + friday_change = 34 := by
  sorry

end NUMINAMATH_CALUDE_fund_price_calculation_l878_87808


namespace NUMINAMATH_CALUDE_max_period_is_14_l878_87859

/-- A function with symmetry properties and a period -/
structure SymmetricPeriodicFunction where
  f : ℝ → ℝ
  period : ℝ
  periodic : ∀ x, f (x + period) = f x
  sym_1 : ∀ x, f (1 + x) = f (1 - x)
  sym_8 : ∀ x, f (8 + x) = f (8 - x)

/-- The maximum period for a SymmetricPeriodicFunction is 14 -/
theorem max_period_is_14 (spf : SymmetricPeriodicFunction) : 
  spf.period ≤ 14 := by sorry

end NUMINAMATH_CALUDE_max_period_is_14_l878_87859


namespace NUMINAMATH_CALUDE_system_solvability_l878_87873

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x * Real.cos a + y * Real.sin a - 2 ≤ 0 ∧
  x^2 + y^2 + 6*x - 2*y - b^2 + 4*b + 6 = 0

-- Define the solution set for b
def solution_set (b : ℝ) : Prop :=
  b ≤ 4 - Real.sqrt 10 ∨ b ≥ Real.sqrt 10

-- Theorem statement
theorem system_solvability (b : ℝ) :
  (∀ a, ∃ x y, system x y a b) ↔ solution_set b := by
  sorry

end NUMINAMATH_CALUDE_system_solvability_l878_87873


namespace NUMINAMATH_CALUDE_number_problem_l878_87827

theorem number_problem (x : ℚ) : (x / 4) * 12 = 9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l878_87827


namespace NUMINAMATH_CALUDE_inverse_composition_equals_neg_sixteen_ninths_l878_87804

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 7

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (x - 7) / 3

-- Theorem statement
theorem inverse_composition_equals_neg_sixteen_ninths :
  g_inv (g_inv 12) = -16/9 :=
by sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_neg_sixteen_ninths_l878_87804


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l878_87822

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 160 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l878_87822


namespace NUMINAMATH_CALUDE_add_average_score_theorem_singing_competition_scores_l878_87849

/-- Represents a set of scores with their statistical properties -/
structure ScoreSet where
  count : ℕ
  average : ℝ
  variance : ℝ

/-- Represents the result after adding a new score -/
structure NewScoreSet where
  new_average : ℝ
  new_variance : ℝ

/-- 
Given a set of scores and a new score, calculates the new average and variance
-/
def add_score (scores : ScoreSet) (new_score : ℝ) : NewScoreSet :=
  sorry

/-- 
Theorem: Adding a score equal to the original average keeps the average the same
and reduces the variance
-/
theorem add_average_score_theorem (scores : ScoreSet) :
  let new_set := add_score scores scores.average
  new_set.new_average = scores.average ∧ new_set.new_variance < scores.variance :=
  sorry

/-- 
Application of the theorem to the specific problem
-/
theorem singing_competition_scores :
  let original_scores : ScoreSet := ⟨8, 5, 3⟩
  let new_set := add_score original_scores 5
  new_set.new_average = 5 ∧ new_set.new_variance < 3 :=
  sorry

end NUMINAMATH_CALUDE_add_average_score_theorem_singing_competition_scores_l878_87849


namespace NUMINAMATH_CALUDE_cone_sphere_intersection_l878_87817

noncomputable def cone_angle (r : ℝ) (h : ℝ) : ℝ :=
  let α := Real.arcsin ((Real.sqrt 5 - 1) / 2)
  2 * α

theorem cone_sphere_intersection (r : ℝ) (h : ℝ) (hr : r > 0) (hh : h > 0) :
  let α := cone_angle r h / 2
  let sphere_radius := h / 2
  let sphere_cap_area := 4 * Real.pi * sphere_radius^2 * Real.sin α^2
  let cone_cap_area := Real.pi * (2 * sphere_radius * Real.cos α * Real.sin α) * (2 * sphere_radius * Real.cos α)
  sphere_cap_area = cone_cap_area →
  cone_angle r h = 2 * Real.arccos (Real.sqrt 5 - 2) :=
by sorry

end NUMINAMATH_CALUDE_cone_sphere_intersection_l878_87817


namespace NUMINAMATH_CALUDE_chord_length_polar_circle_l878_87867

/-- The length of the chord intercepted by the line tan θ = 1/2 on the circle ρ = 4sin θ is 16/5 -/
theorem chord_length_polar_circle (θ : Real) (ρ : Real) : 
  ρ = 4 * Real.sin θ → Real.tan θ = 1 / 2 → 
  2 * ρ * Real.sin θ = 16 / 5 := by sorry

end NUMINAMATH_CALUDE_chord_length_polar_circle_l878_87867


namespace NUMINAMATH_CALUDE_product_def_l878_87813

theorem product_def (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : (a * f) / (c * d) = 0.75) :
  d * e * f = 250 := by
sorry

end NUMINAMATH_CALUDE_product_def_l878_87813


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l878_87864

/-- Calculates the average speed of a cyclist who drives four laps of equal distance
    at different speeds. -/
theorem cyclist_average_speed (d : ℝ) (h : d > 0) :
  let speeds := [6, 12, 18, 24]
  let total_distance := 4 * d
  let total_time := d / 6 + d / 12 + d / 18 + d / 24
  total_distance / total_time = 288 / 25 := by
sorry

#eval (288 : ℚ) / 25  -- To verify the result is approximately 11.52

end NUMINAMATH_CALUDE_cyclist_average_speed_l878_87864


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l878_87891

theorem intersection_point_of_lines (x y : ℚ) : 
  (5 * x - 2 * y = 8) ∧ (6 * x + 3 * y = 21) ↔ x = 22/9 ∧ y = 19/9 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l878_87891


namespace NUMINAMATH_CALUDE_jessicas_balloons_l878_87855

/-- Given the number of blue balloons for Joan, Sally, and the total,
    prove that Jessica has 2 blue balloons. -/
theorem jessicas_balloons
  (joan_balloons : ℕ)
  (sally_balloons : ℕ)
  (total_balloons : ℕ)
  (h1 : joan_balloons = 9)
  (h2 : sally_balloons = 5)
  (h3 : total_balloons = 16)
  (h4 : ∃ (jessica_balloons : ℕ), joan_balloons + sally_balloons + jessica_balloons = total_balloons) :
  ∃ (jessica_balloons : ℕ), jessica_balloons = 2 ∧ joan_balloons + sally_balloons + jessica_balloons = total_balloons :=
by
  sorry

end NUMINAMATH_CALUDE_jessicas_balloons_l878_87855


namespace NUMINAMATH_CALUDE_rate_percent_calculation_l878_87868

/-- Given that the simple interest on Rs. 25,000 amounts to Rs. 5,500 in 7 years,
    prove that the rate percent is equal to (5500 * 100) / (25000 * 7) -/
theorem rate_percent_calculation (principal : ℝ) (interest : ℝ) (time : ℝ) 
    (h1 : principal = 25000)
    (h2 : interest = 5500)
    (h3 : time = 7)
    (h4 : interest = principal * (rate_percent / 100) * time) :
  rate_percent = (interest * 100) / (principal * time) := by
  sorry

end NUMINAMATH_CALUDE_rate_percent_calculation_l878_87868


namespace NUMINAMATH_CALUDE_quadratic_factorization_l878_87811

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l878_87811


namespace NUMINAMATH_CALUDE_min_PQ_length_l878_87851

/-- Circle C with center (3,4) and radius 2 -/
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

/-- Point P is outside the circle -/
def P_outside_circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 > 4

/-- Length of PQ equals distance from P to origin -/
def PQ_equals_PO (x y : ℝ) : Prop := ∃ (qx qy : ℝ), 
  circle_C qx qy ∧ (x - qx)^2 + (y - qy)^2 = x^2 + y^2

/-- Theorem: Minimum value of |PQ| is 17/2 -/
theorem min_PQ_length (x y : ℝ) : 
  circle_C x y → P_outside_circle x y → PQ_equals_PO x y → 
  ∃ (qx qy : ℝ), circle_C qx qy ∧ (x - qx)^2 + (y - qy)^2 ≥ (17/2)^2 :=
sorry

end NUMINAMATH_CALUDE_min_PQ_length_l878_87851


namespace NUMINAMATH_CALUDE_boat_downstream_speed_l878_87819

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed of a boat given its speed in still water and upstream -/
def downstreamSpeed (boat : BoatSpeed) : ℝ :=
  2 * boat.stillWater - boat.upstream

theorem boat_downstream_speed 
  (boat : BoatSpeed) 
  (h1 : boat.stillWater = 8.5) 
  (h2 : boat.upstream = 4) : 
  downstreamSpeed boat = 13 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_speed_l878_87819


namespace NUMINAMATH_CALUDE_f_decreasing_on_neg_reals_l878_87843

-- Define the function
def f (x : ℝ) : ℝ := |x - 1|

-- State the theorem
theorem f_decreasing_on_neg_reals : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 0 → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_neg_reals_l878_87843


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l878_87898

/-- An increasing arithmetic sequence of integers -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℤ) 
  (h_seq : arithmetic_sequence a) 
  (h_prod : a 4 * a 5 = 12) : 
  a 2 * a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l878_87898


namespace NUMINAMATH_CALUDE_range_of_a_l878_87801

/-- Proposition p: For all x in [1,2], x^2 - a ≥ 0 -/
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

/-- Proposition q: The equation x^2 + 2ax + a + 2 = 0 has solutions -/
def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0

/-- If propositions p and q are both true, then a ∈ (-∞, -1] -/
theorem range_of_a (a : ℝ) (h_p : prop_p a) (h_q : prop_q a) : a ∈ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l878_87801


namespace NUMINAMATH_CALUDE_bad_carrots_count_l878_87875

/-- The number of bad carrots in Faye's garden -/
def bad_carrots (faye_carrots mother_carrots good_carrots : ℕ) : ℕ :=
  faye_carrots + mother_carrots - good_carrots

/-- Theorem: The number of bad carrots is 16 -/
theorem bad_carrots_count : bad_carrots 23 5 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l878_87875


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l878_87820

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 - a - 2) (a^2 - 3*a + 2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l878_87820


namespace NUMINAMATH_CALUDE_a4_square_area_l878_87879

/-- Represents the properties of an A4 sheet of paper -/
structure A4Sheet where
  length : Real
  width : Real
  ratio_preserved : length / width = length / (2 * width)

theorem a4_square_area (sheet : A4Sheet) (h1 : sheet.length = 29.7) :
  ∃ (area : Real), abs (area - sheet.width ^ 2) < 0.05 ∧ abs (area - 441.0) < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_a4_square_area_l878_87879


namespace NUMINAMATH_CALUDE_ratio_sum_squares_to_sum_l878_87805

theorem ratio_sum_squares_to_sum (a b c : ℝ) : 
  (b = 2 * a) → 
  (c = 4 * a) → 
  (a^2 + b^2 + c^2 = 1701) → 
  (a + b + c = 63) := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_to_sum_l878_87805


namespace NUMINAMATH_CALUDE_horner_method_v2_l878_87876

def f (x : ℝ) : ℝ := x^6 - 8*x^5 + 60*x^4 + 16*x^3 + 96*x^2 + 240*x + 64

def horner_v2 (a : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * a - 8
  v1 * a + 60

theorem horner_method_v2 :
  horner_v2 2 = 48 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v2_l878_87876


namespace NUMINAMATH_CALUDE_simplify_fraction_l878_87872

theorem simplify_fraction : (54 : ℚ) / 972 = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l878_87872


namespace NUMINAMATH_CALUDE_class_size_l878_87865

/-- The number of students in Ms. Perez's class -/
def S : ℕ := sorry

/-- The number of students who collected 12 cans each -/
def students_12_cans : ℕ := S / 2

/-- The number of students who collected 4 cans each -/
def students_4_cans : ℕ := 13

/-- The number of students who didn't collect any cans -/
def students_0_cans : ℕ := 2

/-- The total number of cans collected -/
def total_cans : ℕ := 232

theorem class_size :
  S = 30 ∧
  S = students_12_cans + students_4_cans + students_0_cans ∧
  total_cans = students_12_cans * 12 + students_4_cans * 4 + students_0_cans * 0 :=
sorry

end NUMINAMATH_CALUDE_class_size_l878_87865


namespace NUMINAMATH_CALUDE_unique_solution_for_absolute_value_equation_l878_87889

theorem unique_solution_for_absolute_value_equation :
  ∃! x : ℤ, |x - 8 * (3 - 12)| - |5 - 11| = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_absolute_value_equation_l878_87889


namespace NUMINAMATH_CALUDE_eighth_term_is_eight_l878_87854

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference of the sequence
  d : ℝ
  -- The sum of the first six terms is 21
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 21
  -- The seventh term is 7
  seventh_term : a + 6*d = 7

/-- The eighth term of the arithmetic sequence is 8 -/
theorem eighth_term_is_eight (seq : ArithmeticSequence) : seq.a + 7*seq.d = 8 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_eight_l878_87854


namespace NUMINAMATH_CALUDE_song_book_cost_l878_87863

/-- The cost of the song book given the costs of other items and the total spent --/
theorem song_book_cost (trumpet_cost music_tool_cost total_spent : ℚ) : 
  trumpet_cost = 149.16 →
  music_tool_cost = 9.98 →
  total_spent = 163.28 →
  total_spent - (trumpet_cost + music_tool_cost) = 4.14 := by
  sorry

end NUMINAMATH_CALUDE_song_book_cost_l878_87863


namespace NUMINAMATH_CALUDE_wrapping_paper_needed_l878_87892

theorem wrapping_paper_needed (present1 : ℝ) (present2 : ℝ) (present3 : ℝ) :
  present1 = 2 →
  present2 = 3 / 4 * present1 →
  present3 = present1 + present2 →
  present1 + present2 + present3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_needed_l878_87892


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l878_87880

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 15 (2*x + 1) = Nat.choose 15 (x + 2)) ↔ (x = 1 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l878_87880


namespace NUMINAMATH_CALUDE_round_trip_time_l878_87846

/-- Calculates the total time for a round trip boat journey given the boat's speed in standing water,
    the stream speed, and the total distance traveled. -/
theorem round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_distance : ℝ) 
  (h1 : boat_speed = 8) 
  (h2 : stream_speed = 6) 
  (h3 : total_distance = 420) : 
  (total_distance / (boat_speed + stream_speed) + 
   total_distance / (boat_speed - stream_speed)) = 120 := by
  sorry

#check round_trip_time

end NUMINAMATH_CALUDE_round_trip_time_l878_87846
