import Mathlib

namespace NUMINAMATH_CALUDE_marathon_remainder_l526_52657

/-- Represents the distance of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a runner's total distance in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ

/-- Converts a given number of yards to miles and remaining yards -/
def yardsToMilesAndYards (totalYards : ℕ) : TotalDistance :=
  { miles := totalYards / 1760,
    yards := totalYards % 1760 }

theorem marathon_remainder (marathonDistance : Marathon) (numMarathons : ℕ) : 
  marathonDistance.miles = 26 →
  marathonDistance.yards = 395 →
  numMarathons = 15 →
  (yardsToMilesAndYards (numMarathons * (marathonDistance.miles * 1760 + marathonDistance.yards))).yards = 645 := by
  sorry

#check marathon_remainder

end NUMINAMATH_CALUDE_marathon_remainder_l526_52657


namespace NUMINAMATH_CALUDE_only_football_fans_l526_52639

/-- Represents the number of people in different categories in a class --/
structure ClassPreferences where
  total : Nat
  baseballAndFootball : Nat
  onlyBaseball : Nat
  neitherSport : Nat
  onlyFootball : Nat

/-- The theorem stating the number of people who only like football --/
theorem only_football_fans (c : ClassPreferences) : c.onlyFootball = 3 :=
  by
  have h1 : c.total = 16 := by sorry
  have h2 : c.baseballAndFootball = 5 := by sorry
  have h3 : c.onlyBaseball = 2 := by sorry
  have h4 : c.neitherSport = 6 := by sorry
  have h5 : c.total = c.baseballAndFootball + c.onlyBaseball + c.onlyFootball + c.neitherSport := by sorry
  sorry

#check only_football_fans

end NUMINAMATH_CALUDE_only_football_fans_l526_52639


namespace NUMINAMATH_CALUDE_sequence_property_l526_52655

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n > 0) →
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → a n * (n * a n - a (n + 1)) = (n + 1) * (a (n + 1))^2) →
  ∀ n : ℕ, n > 0 → a n = 1 / n :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l526_52655


namespace NUMINAMATH_CALUDE_calzone_time_proof_l526_52636

def calzone_time_calculation (onion_time garlic_time knead_time rest_time assemble_time : ℕ) : Prop :=
  (garlic_time = onion_time / 4) ∧
  (rest_time = 2 * knead_time) ∧
  (assemble_time = (knead_time + rest_time) / 10) ∧
  (onion_time + garlic_time + knead_time + rest_time + assemble_time = 124)

theorem calzone_time_proof :
  ∃ (onion_time garlic_time knead_time rest_time assemble_time : ℕ),
    onion_time = 20 ∧
    knead_time = 30 ∧
    calzone_time_calculation onion_time garlic_time knead_time rest_time assemble_time :=
by
  sorry

end NUMINAMATH_CALUDE_calzone_time_proof_l526_52636


namespace NUMINAMATH_CALUDE_unique_valid_square_l526_52610

/-- A number is a square with exactly two non-zero digits, one of which is 3 -/
def is_valid_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 ∧ 
  (∃ a b : ℕ, a ≠ 0 ∧ b ≠ 0 ∧ (a = 3 ∨ b = 3) ∧ n = 10 * a + b) ∧
  (∀ c d e : ℕ, n ≠ 100 * c + 10 * d + e ∨ c = 0 ∨ d = 0 ∨ e = 0)

theorem unique_valid_square : 
  ∀ n : ℕ, is_valid_square n ↔ n = 36 :=
by sorry

end NUMINAMATH_CALUDE_unique_valid_square_l526_52610


namespace NUMINAMATH_CALUDE_park_pairings_l526_52605

/-- The number of unique pairings in a group of 12 people where two specific individuals do not interact -/
theorem park_pairings (n : ℕ) (h : n = 12) : 
  (n.choose 2) - 1 = 65 := by
  sorry

end NUMINAMATH_CALUDE_park_pairings_l526_52605


namespace NUMINAMATH_CALUDE_laundry_problem_solution_l526_52661

/-- Represents the laundromat problem setup -/
structure LaundryProblem where
  washer_cost : ℚ  -- Cost per washer load in dollars
  dryer_cost : ℚ   -- Cost per 10 minutes of dryer use in dollars
  wash_loads : ℕ   -- Number of wash loads
  num_dryers : ℕ   -- Number of dryers used
  total_spent : ℚ  -- Total amount spent in dollars

/-- Calculates the time each dryer ran in minutes -/
def dryer_time (p : LaundryProblem) : ℚ :=
  let washing_cost := p.washer_cost * p.wash_loads
  let drying_cost := p.total_spent - washing_cost
  let total_drying_time := (drying_cost / p.dryer_cost) * 10
  total_drying_time / p.num_dryers

/-- Theorem stating that for the given problem setup, each dryer ran for 40 minutes -/
theorem laundry_problem_solution (p : LaundryProblem) 
  (h1 : p.washer_cost = 4)
  (h2 : p.dryer_cost = 1/4)
  (h3 : p.wash_loads = 2)
  (h4 : p.num_dryers = 3)
  (h5 : p.total_spent = 11) :
  dryer_time p = 40 := by
  sorry


end NUMINAMATH_CALUDE_laundry_problem_solution_l526_52661


namespace NUMINAMATH_CALUDE_sum_of_ratios_ge_two_l526_52670

theorem sum_of_ratios_ge_two (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / (c + d) + b / (d + a) + c / (a + b) + d / (b + c) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_ge_two_l526_52670


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l526_52645

theorem multiplication_puzzle :
  ∀ (G L D E N : ℕ),
    G ≠ L ∧ G ≠ D ∧ G ≠ E ∧ G ≠ N ∧
    L ≠ D ∧ L ≠ E ∧ L ≠ N ∧
    D ≠ E ∧ D ≠ N ∧
    E ≠ N ∧
    1 ≤ G ∧ G ≤ 9 ∧
    1 ≤ L ∧ L ≤ 9 ∧
    1 ≤ D ∧ D ≤ 9 ∧
    1 ≤ E ∧ E ≤ 9 ∧
    1 ≤ N ∧ N ≤ 9 ∧
    100000 * G + 40000 + 1000 * L + 100 * D + 10 * E + N = 
    (100000 * D + 10000 * E + 1000 * N + 100 * G + 40 + L) * 6 →
    G = 1 ∧ L = 2 ∧ D = 8 ∧ E = 5 ∧ N = 7 :=
by sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l526_52645


namespace NUMINAMATH_CALUDE_bird_flight_problem_l526_52653

theorem bird_flight_problem (h₁ h₂ w : ℝ) (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (w_pos : w > 0)
  (h₁_val : h₁ = 20) (h₂_val : h₂ = 30) (w_val : w = 50) :
  ∃ (d x : ℝ),
    d = 10 * Real.sqrt 13 ∧
    x = 20 ∧
    d = Real.sqrt (x^2 + h₂^2) ∧
    d = Real.sqrt ((w - x)^2 + h₁^2) := by
  sorry

#check bird_flight_problem

end NUMINAMATH_CALUDE_bird_flight_problem_l526_52653


namespace NUMINAMATH_CALUDE_index_card_area_l526_52683

theorem index_card_area (length width : ℝ) : 
  length = 8 ∧ width = 3 →
  (∃ new_length, new_length = length - 2 ∧ new_length * width = 18) →
  (width - 2) * length = 8 :=
by sorry

end NUMINAMATH_CALUDE_index_card_area_l526_52683


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l526_52637

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + 1| - |x - a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 4 x > 2} = {x : ℝ | x < -7 ∨ x > 5/3} := by sorry

-- Part II
theorem range_of_a_part_ii :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 2 3, f a x ≥ |x - 4|) → a ∈ Set.Icc (-1) 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l526_52637


namespace NUMINAMATH_CALUDE_tanners_savings_l526_52607

/-- Tanner's savings problem -/
theorem tanners_savings (september : ℕ) (october : ℕ) (november : ℕ) (spent : ℕ) (left : ℕ) : 
  september = 17 → 
  october = 48 → 
  spent = 49 → 
  left = 41 → 
  september + october + november - spent = left → 
  november = 25 := by
sorry

end NUMINAMATH_CALUDE_tanners_savings_l526_52607


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l526_52608

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |2*x - 1|

-- Theorem for Part I
theorem solution_set_when_a_is_3 :
  {x : ℝ | f x 3 ≤ 6} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/2} := by sorry

-- Theorem for Part II
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ a^2 - a - 13} = {a : ℝ | -Real.sqrt 14 ≤ a ∧ a ≤ 1 + Real.sqrt 13} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l526_52608


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l526_52685

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : (1 - 2*I)*z = 5*I) : 
  z.im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l526_52685


namespace NUMINAMATH_CALUDE_solve_for_y_l526_52623

theorem solve_for_y (x z : ℝ) (h1 : x^2 * z - x * z^2 = 6) (h2 : x = -2) (h3 : z = 1) : 
  ∃ y : ℝ, x^2 * y * z - x * y * z^2 = 6 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l526_52623


namespace NUMINAMATH_CALUDE_functional_equation_solution_l526_52699

/-- The functional equation problem -/
theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y + f x) = x * f y + 2) : 
  ∀ (x : ℝ), x > 0 → f x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l526_52699


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l526_52603

theorem negation_of_existence (f : ℝ → ℝ) : 
  (¬ ∃ x : ℝ, f x = 0) ↔ (∀ x : ℝ, f x ≠ 0) := by sorry

theorem negation_of_cubic_equation :
  (¬ ∃ x : ℝ, x^3 + 5*x - 2 = 0) ↔ (∀ x : ℝ, x^3 + 5*x - 2 ≠ 0) := by
  apply negation_of_existence (λ x => x^3 + 5*x - 2)

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l526_52603


namespace NUMINAMATH_CALUDE_students_playing_neither_l526_52626

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 38 →
  football = 26 →
  tennis = 20 →
  both = 17 →
  total - (football + tennis - both) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l526_52626


namespace NUMINAMATH_CALUDE_fireflies_that_flew_away_l526_52600

def initial_fireflies : ℕ := 3
def additional_fireflies : ℕ := 12 - 4
def remaining_fireflies : ℕ := 9

theorem fireflies_that_flew_away :
  initial_fireflies + additional_fireflies - remaining_fireflies = 2 := by
  sorry

end NUMINAMATH_CALUDE_fireflies_that_flew_away_l526_52600


namespace NUMINAMATH_CALUDE_product_of_ratios_l526_52671

/-- Given three pairs of real numbers satisfying specific equations, 
    prove that the product of their ratios equals a specific value. -/
theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2007) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2006)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2007) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2006)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2007) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2006) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -2011/2006 := by
  sorry

end NUMINAMATH_CALUDE_product_of_ratios_l526_52671


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l526_52698

-- Define the sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l526_52698


namespace NUMINAMATH_CALUDE_greatest_integer_radius_of_circle_exists_greatest_integer_radius_greatest_integer_radius_is_8_l526_52634

theorem greatest_integer_radius_of_circle (r : ℕ) : 
  (r : ℝ) * (r : ℝ) * Real.pi < 75 * Real.pi → r ≤ 8 :=
by sorry

theorem exists_greatest_integer_radius : 
  ∃ (r : ℕ), (r : ℝ) * (r : ℝ) * Real.pi < 75 * Real.pi ∧ 
  ∀ (s : ℕ), (s : ℝ) * (s : ℝ) * Real.pi < 75 * Real.pi → s ≤ r :=
by sorry

theorem greatest_integer_radius_is_8 : 
  ∃! (r : ℕ), (r : ℝ) * (r : ℝ) * Real.pi < 75 * Real.pi ∧ 
  ∀ (s : ℕ), (s : ℝ) * (s : ℝ) * Real.pi < 75 * Real.pi → s ≤ r ∧ r = 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_of_circle_exists_greatest_integer_radius_greatest_integer_radius_is_8_l526_52634


namespace NUMINAMATH_CALUDE_average_sale_is_5500_l526_52609

def sales : List ℕ := [5435, 5927, 5855, 6230, 5562]
def sixth_month_sale : ℕ := 3991
def num_months : ℕ := 6

theorem average_sale_is_5500 :
  (sales.sum + sixth_month_sale) / num_months = 5500 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_is_5500_l526_52609


namespace NUMINAMATH_CALUDE_inequality_proof_l526_52617

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b^2 / a + c^2 / b + a^2 / c ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l526_52617


namespace NUMINAMATH_CALUDE_abc_inequality_l526_52614

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l526_52614


namespace NUMINAMATH_CALUDE_cubic_difference_over_difference_l526_52616

theorem cubic_difference_over_difference (r s : ℝ) : 
  3 * r^2 - 4 * r - 12 = 0 →
  3 * s^2 - 4 * s - 12 = 0 →
  (9 * r^3 - 9 * s^3) / (r - s) = 52 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_over_difference_l526_52616


namespace NUMINAMATH_CALUDE_base_prime_repr_450_l526_52629

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- The base prime representation of 450 is [1, 2, 2] -/
theorem base_prime_repr_450 : base_prime_repr 450 = [1, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_repr_450_l526_52629


namespace NUMINAMATH_CALUDE_divisor_calculation_l526_52658

theorem divisor_calculation (dividend quotient remainder : ℚ) :
  dividend = 13/3 →
  quotient = -61 →
  remainder = -19 →
  ∃ divisor : ℚ, dividend = divisor * quotient + remainder ∧ divisor = -70/183 :=
by
  sorry

end NUMINAMATH_CALUDE_divisor_calculation_l526_52658


namespace NUMINAMATH_CALUDE_min_ratio_case1_min_ratio_case2_min_ratio_case3_min_ratio_case4_l526_52695

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the ratio function
def ratio (n : ℕ) : ℚ := n / (sumOfDigits n)

-- Theorem for case (i)
theorem min_ratio_case1 :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → ratio n ≥ 19/10 := by sorry

-- Theorem for case (ii)
theorem min_ratio_case2 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → ratio n ≥ 119/11 := by sorry

-- Theorem for case (iii)
theorem min_ratio_case3 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → ratio n ≥ 1119/12 := by sorry

-- Theorem for case (iv)
theorem min_ratio_case4 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 → ratio n ≥ 11119/13 := by sorry

end NUMINAMATH_CALUDE_min_ratio_case1_min_ratio_case2_min_ratio_case3_min_ratio_case4_l526_52695


namespace NUMINAMATH_CALUDE_area_G1G2G3_l526_52666

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define point P inside triangle ABC
variable (P : ℝ × ℝ)

-- Define G1, G2, G3 as centroids of triangles PBC, PCA, PAB respectively
def G1 : ℝ × ℝ := sorry
def G2 : ℝ × ℝ := sorry
def G3 : ℝ × ℝ := sorry

-- Define the area function
def area (a b c : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_G1G2G3 (h : area A B C = 24) :
  area G1 G2 G3 = 8/3 := by sorry

end NUMINAMATH_CALUDE_area_G1G2G3_l526_52666


namespace NUMINAMATH_CALUDE_initial_kids_on_soccer_field_l526_52642

theorem initial_kids_on_soccer_field (initial_kids final_kids joined_kids : ℕ) :
  final_kids = initial_kids + joined_kids →
  joined_kids = 22 →
  final_kids = 36 →
  initial_kids = 14 := by
sorry

end NUMINAMATH_CALUDE_initial_kids_on_soccer_field_l526_52642


namespace NUMINAMATH_CALUDE_fred_earnings_l526_52674

/-- The amount of money earned given an hourly rate and number of hours worked -/
def moneyEarned (hourlyRate : ℝ) (hoursWorked : ℝ) : ℝ :=
  hourlyRate * hoursWorked

/-- Proof that working 8 hours at $12.5 per hour results in $100 earned -/
theorem fred_earnings : moneyEarned 12.5 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_fred_earnings_l526_52674


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l526_52628

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + 2*x + 3 > 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l526_52628


namespace NUMINAMATH_CALUDE_even_decreasing_implies_increasing_l526_52601

-- Define the properties of the function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def IsDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem even_decreasing_implies_increasing 
  (f : ℝ → ℝ) (a b : ℝ) 
  (h_pos : 0 < a ∧ a < b) 
  (h_even : IsEven f) 
  (h_decreasing : IsDecreasingOn f a b) : 
  IsIncreasingOn f (-b) (-a) :=
by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_implies_increasing_l526_52601


namespace NUMINAMATH_CALUDE_chairs_per_row_l526_52686

theorem chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) (h1 : total_chairs = 432) (h2 : num_rows = 27) :
  total_chairs / num_rows = 16 := by
  sorry

end NUMINAMATH_CALUDE_chairs_per_row_l526_52686


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l526_52620

def A : Set ℕ := {70, 1946, 1997, 2003}
def B : Set ℕ := {1, 10, 70, 2016}

theorem intersection_of_A_and_B : A ∩ B = {70} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l526_52620


namespace NUMINAMATH_CALUDE_same_grade_percentage_l526_52684

theorem same_grade_percentage (total_students : ℕ) (same_grade_students : ℕ) : 
  total_students = 40 →
  same_grade_students = 17 →
  (same_grade_students : ℚ) / (total_students : ℚ) * 100 = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l526_52684


namespace NUMINAMATH_CALUDE_find_M_l526_52675

theorem find_M : ∃ (M : ℕ+), (12^2 * 45^2 : ℕ) = 15^2 * M^2 ∧ M = 36 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l526_52675


namespace NUMINAMATH_CALUDE_probability_diamond_then_ace_is_one_fiftytwo_l526_52665

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of diamond cards in a standard deck -/
def DiamondCards : ℕ := 13

/-- Represents the number of ace cards in a standard deck -/
def AceCards : ℕ := 4

/-- The probability of drawing a diamond as the first card and an ace as the second card -/
def probability_diamond_then_ace : ℚ :=
  (DiamondCards : ℚ) / StandardDeck * AceCards / (StandardDeck - 1)

theorem probability_diamond_then_ace_is_one_fiftytwo :
  probability_diamond_then_ace = 1 / StandardDeck :=
sorry

end NUMINAMATH_CALUDE_probability_diamond_then_ace_is_one_fiftytwo_l526_52665


namespace NUMINAMATH_CALUDE_room_length_calculation_l526_52678

/-- Given a room with specified width, total paving cost, and paving rate per square meter,
    calculate the length of the room. -/
theorem room_length_calculation (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 3.75 →
  total_cost = 16500 →
  rate_per_sqm = 800 →
  (total_cost / rate_per_sqm) / width = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_room_length_calculation_l526_52678


namespace NUMINAMATH_CALUDE_fourth_root_sixteen_times_cube_root_eight_times_sqrt_four_l526_52648

theorem fourth_root_sixteen_times_cube_root_eight_times_sqrt_four : 
  (16 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (4 : ℝ) ^ (1/2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_sixteen_times_cube_root_eight_times_sqrt_four_l526_52648


namespace NUMINAMATH_CALUDE_gum_distribution_l526_52696

theorem gum_distribution (john_gum cole_gum aubrey_gum : ℕ) 
  (h1 : john_gum = 54)
  (h2 : cole_gum = 45)
  (h3 : aubrey_gum = 0)
  (num_people : ℕ)
  (h4 : num_people = 3) :
  (john_gum + cole_gum + aubrey_gum) / num_people = 33 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l526_52696


namespace NUMINAMATH_CALUDE_book_arrangement_problem_l526_52697

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) (different : ℕ) (adjacent_pair : ℕ) : ℕ :=
  (Nat.factorial (total - identical + 1 - adjacent_pair + 1) * Nat.factorial adjacent_pair) / 
  Nat.factorial identical

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem book_arrangement_problem : 
  arrange_books 7 3 4 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_problem_l526_52697


namespace NUMINAMATH_CALUDE_function_value_difference_bound_l526_52631

theorem function_value_difference_bound
  (f : Set.Icc 0 1 → ℝ)
  (h₁ : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h₂ : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₂ - f x₁| < |x₂.val - x₁.val|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₂ - f x₁| < (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_function_value_difference_bound_l526_52631


namespace NUMINAMATH_CALUDE_sequence_ratio_l526_52618

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₁ = 1 + d ∧ a₂ = 1 + 2*d ∧ 3 = 1 + 3*d

-- Define the geometric sequence
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ = 1 * r ∧ b₂ = 1 * r^2 ∧ b₃ = 1 * r^3 ∧ 4 = 1 * r^4

theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) 
  (h1 : arithmetic_sequence a₁ a₂) 
  (h2 : geometric_sequence b₁ b₂ b₃) : 
  (a₁ + a₂) / b₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_l526_52618


namespace NUMINAMATH_CALUDE_set_inclusion_equivalence_l526_52668

theorem set_inclusion_equivalence (a : ℝ) : 
  let A := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 22}
  (∃ x, x ∈ A) → (A ⊆ A ∩ B ↔ 6 ≤ a ∧ a ≤ 9) := by
sorry

end NUMINAMATH_CALUDE_set_inclusion_equivalence_l526_52668


namespace NUMINAMATH_CALUDE_fixed_point_coordinates_l526_52615

theorem fixed_point_coordinates : ∃! (A : ℝ × ℝ), ∀ (k : ℝ),
  (3 + k) * A.1 + (1 - 2*k) * A.2 + 1 + 5*k = 0 ∧ A = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_coordinates_l526_52615


namespace NUMINAMATH_CALUDE_visitors_in_scientific_notation_l526_52672

-- Define the number of visitors
def visitors : ℕ := 203000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.03 * (10 ^ 5)

-- Theorem statement
theorem visitors_in_scientific_notation :
  (visitors : ℝ) = scientific_notation := by sorry

end NUMINAMATH_CALUDE_visitors_in_scientific_notation_l526_52672


namespace NUMINAMATH_CALUDE_nell_gave_jeff_168_cards_l526_52633

/-- The number of cards Nell gave to Jeff -/
def cards_to_jeff (initial : ℕ) (to_john : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_john - remaining

/-- Proof that Nell gave 168 cards to Jeff -/
theorem nell_gave_jeff_168_cards :
  cards_to_jeff 573 195 210 = 168 := by
  sorry

end NUMINAMATH_CALUDE_nell_gave_jeff_168_cards_l526_52633


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l526_52604

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l526_52604


namespace NUMINAMATH_CALUDE_solve_star_equation_l526_52611

-- Define the ☆ operator
def star (a b : ℝ) : ℝ := a * b + a + b

-- Theorem statement
theorem solve_star_equation : 
  ∃! x : ℝ, star 3 x = -9 ∧ x = -3 :=
sorry

end NUMINAMATH_CALUDE_solve_star_equation_l526_52611


namespace NUMINAMATH_CALUDE_esperanza_salary_l526_52649

def gross_salary (rent food mortgage savings taxes : ℚ) : ℚ :=
  rent + food + mortgage + savings + taxes

theorem esperanza_salary :
  let rent : ℚ := 600
  let food : ℚ := (3/5) * rent
  let mortgage : ℚ := 3 * food
  let savings : ℚ := 2000
  let taxes : ℚ := (2/5) * savings
  gross_salary rent food mortgage savings taxes = 4840 := by
  sorry

end NUMINAMATH_CALUDE_esperanza_salary_l526_52649


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l526_52680

theorem quadratic_root_condition (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*a*x + a + 1 = 0 ∧ y^2 + 2*a*y + a + 1 = 0 ∧ x > 2 ∧ y < 2) → 
  a < -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l526_52680


namespace NUMINAMATH_CALUDE_safe_descent_possible_l526_52647

/-- Represents the cliff and rope setup --/
structure CliffSetup where
  cliff_height : ℝ
  rope_length : ℝ
  branch_height : ℝ

/-- Defines a safe descent --/
def safe_descent (setup : CliffSetup) : Prop :=
  setup.cliff_height > setup.rope_length ∧
  setup.branch_height < setup.cliff_height ∧
  setup.branch_height > 0 ∧
  setup.rope_length ≥ setup.cliff_height - setup.branch_height + setup.branch_height / 2

/-- Theorem stating that a safe descent is possible given the specific measurements --/
theorem safe_descent_possible : 
  ∃ (setup : CliffSetup), 
    setup.cliff_height = 100 ∧ 
    setup.rope_length = 75 ∧ 
    setup.branch_height = 50 ∧ 
    safe_descent setup := by
  sorry


end NUMINAMATH_CALUDE_safe_descent_possible_l526_52647


namespace NUMINAMATH_CALUDE_lawn_length_l526_52644

/-- Given a rectangular lawn with area 20 square feet and width 5 feet, prove its length is 4 feet. -/
theorem lawn_length (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 20 → width = 5 → area = length * width → length = 4 := by
  sorry

end NUMINAMATH_CALUDE_lawn_length_l526_52644


namespace NUMINAMATH_CALUDE_r₂_lower_bound_two_is_greatest_lower_bound_l526_52619

/-- The function f(x) = x² - r₂x + r₃ -/
noncomputable def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂*x + r₃

/-- The sequence {gₙ} defined recursively -/
noncomputable def g (r₂ r₃ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r₂ r₃ (g r₂ r₃ n)

/-- The theorem stating the lower bound on |r₂| -/
theorem r₂_lower_bound (r₂ r₃ : ℝ) :
  (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)) →
  (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i + 1) > g r₂ r₃ i) →
  (∀ M : ℝ, ∃ n : ℕ, |g r₂ r₃ n| > M) →
  |r₂| > 2 :=
sorry

/-- The theorem stating that 2 is the greatest lower bound -/
theorem two_is_greatest_lower_bound :
  ∀ ε > 0, ∃ r₂ r₃ : ℝ,
    (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)) ∧
    (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i + 1) > g r₂ r₃ i) ∧
    (∀ M : ℝ, ∃ n : ℕ, |g r₂ r₃ n| > M) ∧
    |r₂| < 2 + ε :=
sorry

end NUMINAMATH_CALUDE_r₂_lower_bound_two_is_greatest_lower_bound_l526_52619


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l526_52694

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : 2*x + 8*y - x*y = 0) : 
  x*y ≥ 64 ∧ x + y ≥ 18 := by sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l526_52694


namespace NUMINAMATH_CALUDE_sum_of_k_values_l526_52682

theorem sum_of_k_values (a b c k : ℂ) : 
  a ≠ b ∧ b ≠ c ∧ c ≠ a →
  (a + 1) / (2 - b) = k ∧
  (b + 1) / (2 - c) = k ∧
  (c + 1) / (2 - a) = k →
  ∃ k₁ k₂ : ℂ, k = k₁ ∨ k = k₂ ∧ k₁ + k₂ = (3/2 : ℂ) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_k_values_l526_52682


namespace NUMINAMATH_CALUDE_all_equations_have_one_negative_one_positive_root_l526_52606

-- Define the equations
def equation1 (x : ℝ) : Prop := 4 * x^2 - 6 = 34
def equation2 (x : ℝ) : Prop := (3*x-2)^2 = (x+1)^2
def equation3 (x : ℝ) : Prop := (x^2-12).sqrt = (2*x-2).sqrt

-- Define the property of having one negative and one positive root
def has_one_negative_one_positive_root (f : ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x ∧ f y

-- Theorem statement
theorem all_equations_have_one_negative_one_positive_root :
  has_one_negative_one_positive_root equation1 ∧
  has_one_negative_one_positive_root equation2 ∧
  has_one_negative_one_positive_root equation3 :=
sorry

end NUMINAMATH_CALUDE_all_equations_have_one_negative_one_positive_root_l526_52606


namespace NUMINAMATH_CALUDE_inequality_proof_l526_52660

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l526_52660


namespace NUMINAMATH_CALUDE_cubic_root_sum_l526_52687

theorem cubic_root_sum (p q r : ℝ) : 
  0 < p ∧ p < 1 ∧ 
  0 < q ∧ q < 1 ∧ 
  0 < r ∧ r < 1 ∧ 
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  30 * p^3 - 50 * p^2 + 22 * p - 1 = 0 ∧
  30 * q^3 - 50 * q^2 + 22 * q - 1 = 0 ∧
  30 * r^3 - 50 * r^2 + 22 * r - 1 = 0 →
  1 / (1 - p) + 1 / (1 - q) + 1 / (1 - r) = 12 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l526_52687


namespace NUMINAMATH_CALUDE_correct_prime_sum_l526_52630

def isPrime (n : ℕ) : Prop :=
  ∃ m : ℕ, n + 2 = 2^m

def primeSum : ℕ := sorry

theorem correct_prime_sum : primeSum = 2026 := by sorry

end NUMINAMATH_CALUDE_correct_prime_sum_l526_52630


namespace NUMINAMATH_CALUDE_experiment_sequences_l526_52693

/-- The number of procedures in the experiment -/
def total_procedures : ℕ := 6

/-- The number of ways to place procedure A (first or last) -/
def a_placements : ℕ := 2

/-- The number of distinct units to arrange (including BC as one unit) -/
def distinct_units : ℕ := 4

/-- The number of ways to arrange B and C within their unit -/
def bc_arrangements : ℕ := 2

/-- The total number of possible sequences for the experiment procedures -/
def total_sequences : ℕ := a_placements * (distinct_units.factorial) * bc_arrangements

theorem experiment_sequences :
  total_sequences = 96 :=
sorry

end NUMINAMATH_CALUDE_experiment_sequences_l526_52693


namespace NUMINAMATH_CALUDE_equation_solution_l526_52662

theorem equation_solution : ∃ x : ℚ, (1 / 5 + 5 / x = 12 / x + 1 / 12) ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l526_52662


namespace NUMINAMATH_CALUDE_average_increase_is_three_l526_52659

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average after a new inning -/
def averageIncrease (b : Batsman) (newRuns : ℕ) : ℚ :=
  let newAverage := (b.totalRuns + newRuns) / (b.innings + 1)
  newAverage - b.average

/-- The theorem to be proved -/
theorem average_increase_is_three :
  ∀ (b : Batsman),
    b.innings = 16 →
    (b.totalRuns + 84) / 17 = 36 →
    averageIncrease b 84 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_is_three_l526_52659


namespace NUMINAMATH_CALUDE_average_food_expense_percentage_l526_52638

/-- Calculate the average percentage of income spent on food over two months --/
theorem average_food_expense_percentage (jan_income feb_income : ℚ)
  (jan_petrol feb_petrol : ℚ) : 
  jan_income = 3000 →
  feb_income = 4000 →
  jan_petrol = 300 →
  feb_petrol = 400 →
  let jan_remaining := jan_income - jan_petrol
  let feb_remaining := feb_income - feb_petrol
  let jan_rent := jan_remaining * (14 / 100)
  let feb_rent := feb_remaining * (14 / 100)
  let jan_clothing := jan_income * (10 / 100)
  let feb_clothing := feb_income * (10 / 100)
  let jan_utility := jan_income * (5 / 100)
  let feb_utility := feb_income * (5 / 100)
  let jan_food := jan_remaining - jan_rent - jan_clothing - jan_utility
  let feb_food := feb_remaining - feb_rent - feb_clothing - feb_utility
  let total_food := jan_food + feb_food
  let total_income := jan_income + feb_income
  let avg_food_percentage := (total_food / total_income) * 100
  avg_food_percentage = 62.4 := by
  sorry

end NUMINAMATH_CALUDE_average_food_expense_percentage_l526_52638


namespace NUMINAMATH_CALUDE_class_size_l526_52602

theorem class_size (error_increase : ℝ) (average_increase : ℝ) (n : ℕ) : 
  error_increase = 20 →
  average_increase = 1/2 →
  error_increase = n * average_increase →
  n = 40 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l526_52602


namespace NUMINAMATH_CALUDE_brand_x_pen_price_l526_52667

/-- The price of a brand X pen satisfies the given conditions -/
theorem brand_x_pen_price :
  ∀ (total_pens brand_x_pens : ℕ) (brand_y_price total_cost brand_x_price : ℚ),
    total_pens = 12 →
    brand_x_pens = 8 →
    brand_y_price = 14/5 →
    total_cost = 40 →
    brand_x_price * brand_x_pens + brand_y_price * (total_pens - brand_x_pens) = total_cost →
    brand_x_price = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_brand_x_pen_price_l526_52667


namespace NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l526_52635

/-- The area of a circle circumscribing an equilateral triangle with side length 4 is 16π/3 -/
theorem circle_area_equilateral_triangle :
  let s : ℝ := 4  -- side length of the equilateral triangle
  let r : ℝ := s / Real.sqrt 3  -- radius of the circumscribed circle
  let A : ℝ := π * r^2  -- area of the circle
  A = 16 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l526_52635


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l526_52663

theorem cone_lateral_surface_area 
  (base_radius : ℝ) 
  (height : ℝ) 
  (lateral_surface_area : ℝ) 
  (h1 : base_radius = 3) 
  (h2 : height = 4) : 
  lateral_surface_area = 15 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l526_52663


namespace NUMINAMATH_CALUDE_jake_has_nine_peaches_l526_52692

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 16

/-- The number of peaches Jake has fewer than Steven -/
def jake_fewer_than_steven : ℕ := 7

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - jake_fewer_than_steven

/-- Theorem: Jake has 9 peaches -/
theorem jake_has_nine_peaches : jake_peaches = 9 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_nine_peaches_l526_52692


namespace NUMINAMATH_CALUDE_reading_time_calculation_l526_52677

theorem reading_time_calculation (total_time math_time spelling_time : ℕ) 
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : spelling_time = 18) :
  total_time - (math_time + spelling_time) = 27 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l526_52677


namespace NUMINAMATH_CALUDE_minimum_rental_fee_for_360_people_l526_52681

/-- Represents a bus type with its seat capacity and rental fee -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the minimum rental fee for transporting a given number of people -/
def minimumRentalFee (totalPeople : ℕ) (typeA typeB : BusType) : ℕ :=
  sorry

theorem minimum_rental_fee_for_360_people :
  let typeA : BusType := ⟨40, 400⟩
  let typeB : BusType := ⟨50, 480⟩
  minimumRentalFee 360 typeA typeB = 3520 := by
  sorry

end NUMINAMATH_CALUDE_minimum_rental_fee_for_360_people_l526_52681


namespace NUMINAMATH_CALUDE_muirhead_inequality_inequality_chain_l526_52625

/-- Symmetric mean function -/
def T (α : List ℝ) (a b c : ℝ) : ℝ := sorry

/-- Majorization relation -/
def Majorizes (α β : List ℝ) : Prop := sorry

theorem muirhead_inequality {α β : List ℝ} {a b c : ℝ} 
  (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : Majorizes α β) :
  T β a b c ≤ T α a b c := sorry

/-- Main theorem to prove -/
theorem inequality_chain (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) : 
  T [2, 1, 1] a b c ≤ T [3, 1, 0] a b c ∧ T [3, 1, 0] a b c ≤ T [4, 0, 0] a b c := by
  sorry

end NUMINAMATH_CALUDE_muirhead_inequality_inequality_chain_l526_52625


namespace NUMINAMATH_CALUDE_random_events_l526_52621

-- Define the type for events
inductive Event
  | CoinToss
  | ChargeAttraction
  | WaterFreezing
  | DiceRoll

-- Define a function to check if an event is random
def isRandomEvent (e : Event) : Prop :=
  match e with
  | Event.CoinToss => true
  | Event.ChargeAttraction => false
  | Event.WaterFreezing => false
  | Event.DiceRoll => true

-- Theorem stating which events are random
theorem random_events :
  (isRandomEvent Event.CoinToss) ∧
  (¬isRandomEvent Event.ChargeAttraction) ∧
  (¬isRandomEvent Event.WaterFreezing) ∧
  (isRandomEvent Event.DiceRoll) := by
  sorry

#check random_events

end NUMINAMATH_CALUDE_random_events_l526_52621


namespace NUMINAMATH_CALUDE_tangent_line_equation_curve_passes_through_point_l526_52622

/-- The equation of the tangent line to the curve y = x^3 at the point (1,1) -/
theorem tangent_line_equation :
  ∃ (a b c : ℝ), (a * 1 + b * 1 + c = 0) ∧
  (∀ (x y : ℝ), y = x^3 → (x - 1)^2 + (y - 1)^2 ≤ (a * x + b * y + c)^2) ∧
  ((a = 3 ∧ b = -1 ∧ c = -2) ∨ (a = 3 ∧ b = -4 ∧ c = 1)) :=
by sorry

/-- The curve y = x^3 passes through the point (1,1) -/
theorem curve_passes_through_point :
  (1 : ℝ)^3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_curve_passes_through_point_l526_52622


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l526_52669

theorem root_exists_in_interval : ∃! x : ℝ, 1/2 < x ∧ x < 1 ∧ Real.exp x = 1/x := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l526_52669


namespace NUMINAMATH_CALUDE_umar_age_is_10_l526_52627

-- Define the ages as natural numbers
def ali_age : ℕ := 8
def age_difference : ℕ := 3
def umar_age_multiplier : ℕ := 2

-- Theorem to prove
theorem umar_age_is_10 :
  let yusaf_age := ali_age - age_difference
  let umar_age := umar_age_multiplier * yusaf_age
  umar_age = 10 := by sorry

end NUMINAMATH_CALUDE_umar_age_is_10_l526_52627


namespace NUMINAMATH_CALUDE_derivative_lg_over_x_l526_52691

open Real

noncomputable def lg (x : ℝ) : ℝ := log x / log 10

theorem derivative_lg_over_x (x : ℝ) (h : x > 0) :
  deriv (λ x => lg x / x) x = (1 - log 10 * lg x) / (x^2 * log 10) :=
by sorry

end NUMINAMATH_CALUDE_derivative_lg_over_x_l526_52691


namespace NUMINAMATH_CALUDE_total_people_on_boats_l526_52651

/-- The number of boats in the lake -/
def num_boats : ℕ := 5

/-- The number of people on each boat -/
def people_per_boat : ℕ := 3

/-- The total number of people on boats in the lake -/
def total_people : ℕ := num_boats * people_per_boat

theorem total_people_on_boats : total_people = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_people_on_boats_l526_52651


namespace NUMINAMATH_CALUDE_assignment_methods_count_l526_52640

def number_of_teachers : ℕ := 5
def number_of_question_types : ℕ := 3

/- Define a function that calculates the number of ways to assign teachers to question types -/
def assignment_methods : ℕ := sorry

/- Theorem stating that the number of assignment methods is 150 -/
theorem assignment_methods_count : assignment_methods = 150 := by sorry

end NUMINAMATH_CALUDE_assignment_methods_count_l526_52640


namespace NUMINAMATH_CALUDE_cosine_in_triangle_l526_52688

/-- Given a triangle ABC with sides a and b, prove that if a = 4, b = 5, 
    and cos(B-A) = 31/32, then cos B = 9/16 -/
theorem cosine_in_triangle (A B C : ℝ) (a b c : ℝ) : 
  a = 4 → b = 5 → Real.cos (B - A) = 31/32 → Real.cos B = 9/16 := by sorry

end NUMINAMATH_CALUDE_cosine_in_triangle_l526_52688


namespace NUMINAMATH_CALUDE_highest_power_of_three_in_M_l526_52664

def M : ℕ := sorry

theorem highest_power_of_three_in_M : 
  (∃ k : ℕ, M = 3 * k) ∧ ¬(∃ k : ℕ, M = 9 * k) := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_in_M_l526_52664


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l526_52676

theorem quadratic_solution_property (a b : ℝ) : 
  (a * 1^2 + b * 1 + 1 = 0) → (3 - a - b = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l526_52676


namespace NUMINAMATH_CALUDE_women_in_room_l526_52612

theorem women_in_room (x : ℕ) (h1 : 4 * x + 2 = 14) : 2 * (5 * x - 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_women_in_room_l526_52612


namespace NUMINAMATH_CALUDE_cardinality_of_B_l526_52613

def A : Finset Int := {-3, -2, -1, 1, 2, 3, 4}

def f (a : Int) : Int := Int.natAbs a

def B : Finset Int := Finset.image f A

theorem cardinality_of_B : Finset.card B = 4 := by
  sorry

end NUMINAMATH_CALUDE_cardinality_of_B_l526_52613


namespace NUMINAMATH_CALUDE_geometric_series_cube_sum_l526_52673

theorem geometric_series_cube_sum (a r : ℝ) (hr : -1 < r ∧ r < 1) :
  (a / (1 - r) = 2) →
  (a^2 / (1 - r^2) = 6) →
  (a^3 / (1 - r^3) = 96/7) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_cube_sum_l526_52673


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l526_52624

/-- Given a point P (x, 3) on the terminal side of angle θ where cos θ = -4/5, prove that x = -4 -/
theorem point_on_terminal_side (x : ℝ) (θ : ℝ) : 
  (∃ P : ℝ × ℝ, P = (x, 3) ∧ P.1 = x * Real.cos θ ∧ P.2 = x * Real.sin θ) → 
  Real.cos θ = -4/5 → 
  x = -4 := by
sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l526_52624


namespace NUMINAMATH_CALUDE_always_has_real_roots_unique_integer_m_for_distinct_positive_integer_roots_l526_52650

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := m * x^2 - (m + 2) * x + 2

-- Part I: The equation always has real roots
theorem always_has_real_roots :
  ∀ m : ℝ, ∃ x : ℝ, quadratic_equation m x = 0 :=
sorry

-- Part II: Only m = 1 gives two distinct positive integer roots
theorem unique_integer_m_for_distinct_positive_integer_roots :
  ∀ m : ℤ, (∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x ≠ y ∧
    quadratic_equation (m : ℝ) (x : ℝ) = 0 ∧
    quadratic_equation (m : ℝ) (y : ℝ) = 0) ↔ m = 1 :=
sorry

end NUMINAMATH_CALUDE_always_has_real_roots_unique_integer_m_for_distinct_positive_integer_roots_l526_52650


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l526_52654

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the events
def EventAGetsRed (d : Distribution) : Prop := d Person.A = Card.Red
def EventBGetsBlue (d : Distribution) : Prop := d Person.B = Card.Blue

-- State the theorem
theorem events_mutually_exclusive_not_opposite :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventAGetsRed d ∧ EventBGetsBlue d)) ∧
  -- The events are not opposite
  (∃ d : Distribution, ¬EventAGetsRed d ∧ ¬EventBGetsBlue d) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l526_52654


namespace NUMINAMATH_CALUDE_ahn_max_number_l526_52643

theorem ahn_max_number : ∃ (max : ℕ), max = 700 ∧ 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 2 * (500 - n - 50) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ahn_max_number_l526_52643


namespace NUMINAMATH_CALUDE_width_of_identical_rectangles_l526_52689

/-- Given six identical rectangles forming a larger rectangle PQRS, prove that the width of each identical rectangle is 30 -/
theorem width_of_identical_rectangles (w : ℝ) : 
  (6 : ℝ) * w^2 = 5400 ∧ 3 * w = 2 * (2 * w) → w = 30 := by
  sorry

end NUMINAMATH_CALUDE_width_of_identical_rectangles_l526_52689


namespace NUMINAMATH_CALUDE_pizza_night_theorem_l526_52646

/-- Pizza night problem -/
theorem pizza_night_theorem 
  (small_pizza_slices : Nat) 
  (medium_pizza_slices : Nat) 
  (large_pizza_slices : Nat)
  (phil_eaten : Nat)
  (andre_eaten : Nat)
  (phil_ratio : Nat)
  (andre_ratio : Nat)
  (h1 : small_pizza_slices = 8)
  (h2 : medium_pizza_slices = 10)
  (h3 : large_pizza_slices = 14)
  (h4 : phil_eaten = 10)
  (h5 : andre_eaten = 12)
  (h6 : phil_ratio = 3)
  (h7 : andre_ratio = 2) :
  let total_slices := small_pizza_slices + 2 * medium_pizza_slices + large_pizza_slices
  let eaten_slices := phil_eaten + andre_eaten
  let remaining_slices := total_slices - eaten_slices
  let total_ratio := phil_ratio + andre_ratio
  let phil_share := (phil_ratio * remaining_slices) / total_ratio
  let andre_share := (andre_ratio * remaining_slices) / total_ratio
  remaining_slices = 20 ∧ phil_share = 12 ∧ andre_share = 8 := by
  sorry

#check pizza_night_theorem

end NUMINAMATH_CALUDE_pizza_night_theorem_l526_52646


namespace NUMINAMATH_CALUDE_two_red_one_spade_probability_l526_52690

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (red_suits : Nat)
  (black_suits : Nat)

/-- Calculates the probability of drawing two red cards followed by a spade -/
def probability_two_red_one_spade (d : Deck) : Rat :=
  if d.total_cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ d.red_suits = 2 ∧ d.black_suits = 2
  then 13 / 204
  else 0

/-- Theorem stating the probability of drawing two red cards followed by a spade from a standard deck -/
theorem two_red_one_spade_probability (d : Deck) :
  d.total_cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ d.red_suits = 2 ∧ d.black_suits = 2 →
  probability_two_red_one_spade d = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_two_red_one_spade_probability_l526_52690


namespace NUMINAMATH_CALUDE_red_balloons_total_l526_52641

/-- The number of red balloons Sara has -/
def sara_red : ℕ := 31

/-- The number of red balloons Sandy has -/
def sandy_red : ℕ := 24

/-- The total number of red balloons Sara and Sandy have -/
def total_red : ℕ := sara_red + sandy_red

theorem red_balloons_total : total_red = 55 := by
  sorry

end NUMINAMATH_CALUDE_red_balloons_total_l526_52641


namespace NUMINAMATH_CALUDE_smallest_positive_number_l526_52652

theorem smallest_positive_number (a b c d e : ℝ) :
  a = 15 - 4 * Real.sqrt 14 ∧
  b = 4 * Real.sqrt 14 - 15 ∧
  c = 20 - 6 * Real.sqrt 15 ∧
  d = 60 - 12 * Real.sqrt 31 ∧
  e = 12 * Real.sqrt 31 - 60 →
  (0 < a ∧ a ≤ b ∧ a ≤ c ∧ a ≤ d ∧ a ≤ e) ∨
  (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0 ∧ d ≤ 0 ∧ e ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_number_l526_52652


namespace NUMINAMATH_CALUDE_expression_values_l526_52632

theorem expression_values (a b c d x y : ℝ) : 
  (a + b = 0) → 
  (c * d = 1) → 
  (x = 4 ∨ x = -4) → 
  (y = -6) → 
  ((2 * x - c * d + 4 * (a + b) - y^2 = -29 ∧ x = 4) ∨ 
   (2 * x - c * d + 4 * (a + b) - y^2 = -45 ∧ x = -4)) :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l526_52632


namespace NUMINAMATH_CALUDE_largest_angle_of_convex_hexagon_consecutive_angles_l526_52656

-- Define a type for convex hexagons with consecutive integer angle measures
structure ConvexHexagonConsecutiveAngles where
  -- The smallest angle measure
  smallest_angle : ℕ
  -- Ensure the hexagon is convex (all angles are less than 180°)
  convex : smallest_angle + 5 < 180

-- Define the sum of interior angles of a hexagon
def hexagon_angle_sum : ℕ := 720

-- Theorem statement
theorem largest_angle_of_convex_hexagon_consecutive_angles 
  (h : ConvexHexagonConsecutiveAngles) : 
  h.smallest_angle + 5 = 122 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_of_convex_hexagon_consecutive_angles_l526_52656


namespace NUMINAMATH_CALUDE_negative_third_greater_than_negative_half_l526_52679

theorem negative_third_greater_than_negative_half : -1/3 > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_negative_third_greater_than_negative_half_l526_52679
