import Mathlib

namespace NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l3555_355550

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (12/11, 14/11)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y = 6 * x - 4

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines : 
  line1 intersection_point.1 intersection_point.2 ∧ 
  line2 intersection_point.1 intersection_point.2 := by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point (x y : ℚ) : 
  line1 x y ∧ line2 x y → (x, y) = intersection_point := by sorry

end NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l3555_355550


namespace NUMINAMATH_CALUDE_solve_equation_l3555_355529

theorem solve_equation (m : ℝ) : (m - 6) ^ 4 = (1 / 16)⁻¹ ↔ m = 8 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3555_355529


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3555_355519

theorem at_least_one_not_less_than_two (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3555_355519


namespace NUMINAMATH_CALUDE_special_function_zero_l3555_355542

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, |f a - f b| ≤ |a - b|) ∧ (f (f (f 0)) = 0)

/-- Theorem: If f is a special function, then f(0) = 0 -/
theorem special_function_zero (f : ℝ → ℝ) (h : special_function f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_zero_l3555_355542


namespace NUMINAMATH_CALUDE_jovanas_shells_l3555_355597

theorem jovanas_shells (initial_shells : ℕ) : 
  initial_shells + 23 = 28 → initial_shells = 5 := by
  sorry

end NUMINAMATH_CALUDE_jovanas_shells_l3555_355597


namespace NUMINAMATH_CALUDE_files_remaining_l3555_355517

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 27)
  (h2 : video_files = 42)
  (h3 : deleted_files = 11) :
  music_files + video_files - deleted_files = 58 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l3555_355517


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_10_l3555_355539

theorem binomial_coefficient_19_10 (h1 : Nat.choose 17 7 = 19448) (h2 : Nat.choose 17 9 = 24310) :
  Nat.choose 19 10 = 92378 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_10_l3555_355539


namespace NUMINAMATH_CALUDE_center_is_one_l3555_355583

/-- A 3x3 table of positive real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)
  (all_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧ i > 0)

/-- The conditions for the table -/
def TableConditions (t : Table) : Prop :=
  t.a * t.b * t.c = 1 ∧
  t.d * t.e * t.f = 1 ∧
  t.g * t.h * t.i = 1 ∧
  t.a * t.d * t.g = 1 ∧
  t.b * t.e * t.h = 1 ∧
  t.c * t.f * t.i = 1 ∧
  t.a * t.b * t.d * t.e = 2 ∧
  t.b * t.c * t.e * t.f = 2 ∧
  t.d * t.e * t.g * t.h = 2 ∧
  t.e * t.f * t.h * t.i = 2

/-- The theorem stating that the center cell must be 1 -/
theorem center_is_one (t : Table) (h : TableConditions t) : t.e = 1 := by
  sorry


end NUMINAMATH_CALUDE_center_is_one_l3555_355583


namespace NUMINAMATH_CALUDE_books_ratio_proof_l3555_355566

/-- Proves the ratio of books to read this month to books read last month -/
theorem books_ratio_proof (total : ℕ) (last_month : ℕ) : 
  total = 12 → last_month = 4 → (total - last_month) / last_month = 2 := by
  sorry

end NUMINAMATH_CALUDE_books_ratio_proof_l3555_355566


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l3555_355571

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l3555_355571


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3555_355530

theorem arithmetic_sequence_sum : ∃ (n : ℕ), 
  let a := 71  -- first term
  let d := 2   -- common difference
  let l := 99  -- last term
  n = (l - a) / d + 1 ∧ 
  3 * (n * (a + l) / 2) = 3825 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3555_355530


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l3555_355512

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  red : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- The probability of drawing a yellow marble given the conditions -/
def yellowProbability (bagA bagB bagC bagD : Bag) : ℚ :=
  let totalA := bagA.white + bagA.black + bagA.red
  let probWhite := bagA.white / totalA
  let probBlack := bagA.black / totalA
  let probRed := bagA.red / totalA
  let probYellowB := bagB.yellow / (bagB.yellow + bagB.blue)
  let probYellowC := bagC.yellow / (bagC.yellow + bagC.blue)
  let probYellowD := bagD.yellow / (bagD.yellow + bagD.blue)
  probWhite * probYellowB + probBlack * probYellowC + probRed * probYellowD

theorem yellow_marble_probability :
  let bagA : Bag := { white := 4, black := 5, red := 2 }
  let bagB : Bag := { yellow := 7, blue := 5 }
  let bagC : Bag := { yellow := 3, blue := 7 }
  let bagD : Bag := { yellow := 8, blue := 2 }
  yellowProbability bagA bagB bagC bagD = 163 / 330 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marble_probability_l3555_355512


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l3555_355534

/-- Given two planar vectors a and b, prove that the magnitude of (a - 2b) is 5. -/
theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  a = (-2, 1) →
  b = (1, 2) →
  ‖a - 2 • b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l3555_355534


namespace NUMINAMATH_CALUDE_find_p_l3555_355518

theorem find_p : ∃ (d q : ℝ), ∀ (x : ℝ),
  (4 * x^2 - 2 * x + 5/2) * (d * x^2 + p * x + q) = 12 * x^4 - 7 * x^3 + 12 * x^2 - 15/2 * x + 10/2 →
  p = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l3555_355518


namespace NUMINAMATH_CALUDE_coal_shoveling_ratio_l3555_355548

/-- Represents the coal shoveling scenario -/
structure CoalScenario where
  people : ℕ
  days : ℕ
  coal : ℕ

/-- Calculates the daily rate of coal shoveling -/
def daily_rate (s : CoalScenario) : ℚ :=
  s.coal / (s.people * s.days)

theorem coal_shoveling_ratio :
  let original := CoalScenario.mk 10 10 10000
  let new := CoalScenario.mk (10 / 2) 80 40000
  daily_rate original = daily_rate new ∧
  (new.people : ℚ) / original.people = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_coal_shoveling_ratio_l3555_355548


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3555_355561

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    left focus F₁, right focus F₂, and a point P on C such that
    PF₁ ⟂ F₁F₂ and PF₁ = F₁F₂, prove that the eccentricity of C is √2 + 1. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (C : Set (ℝ × ℝ))
  (hC : C = {(x, y) | x^2 / a^2 - y^2 / b^2 = 1})
  (F₁ F₂ P : ℝ × ℝ)
  (hF₁ : F₁ ∈ C) (hF₂ : F₂ ∈ C) (hP : P ∈ C)
  (hLeft : (F₁.1 < F₂.1)) -- F₁ is left of F₂
  (hPerp : (P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2) = 0) -- PF₁ ⟂ F₁F₂
  (hEqual : (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) -- PF₁ = F₁F₂
  : (Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) / (2 * a)) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3555_355561


namespace NUMINAMATH_CALUDE_upload_time_calculation_l3555_355500

/-- Represents the time in minutes required to upload a file -/
def uploadTime (fileSize : ℕ) (uploadSpeed : ℕ) : ℕ :=
  fileSize / uploadSpeed

/-- Proves that uploading a 160 MB file at 8 MB/min takes 20 minutes -/
theorem upload_time_calculation :
  uploadTime 160 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_upload_time_calculation_l3555_355500


namespace NUMINAMATH_CALUDE_puzzle_solution_l3555_355560

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Calculate the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + (n - 1) * seq.diff

theorem puzzle_solution :
  ∀ (row col1 col2 : ArithmeticSequence),
    row.first = 28 →
    nthTerm row 4 = 25 →
    nthTerm row 5 = 32 →
    nthTerm col2 7 = -10 →
    col2.first = nthTerm row 7 →
    col1.first = 28 →
    col1.diff = 7 →
    col2.first = -6 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3555_355560


namespace NUMINAMATH_CALUDE_multiply_monomials_l3555_355554

theorem multiply_monomials (a : ℝ) : 3 * a^3 * (-4 * a^2) = -12 * a^5 := by sorry

end NUMINAMATH_CALUDE_multiply_monomials_l3555_355554


namespace NUMINAMATH_CALUDE_system_solution_exists_l3555_355595

theorem system_solution_exists (m : ℝ) (h : m ≠ 3) :
  ∃ (x y : ℝ), y = (3 * m + 2) * x + 1 ∧ y = (5 * m - 4) * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_l3555_355595


namespace NUMINAMATH_CALUDE_stripes_calculation_l3555_355544

/-- The number of stripes on one of Olga's shoes -/
def olga_stripes_per_shoe : ℕ := 3

/-- The number of stripes on one of Rick's shoes -/
def rick_stripes_per_shoe : ℕ := olga_stripes_per_shoe - 1

/-- The number of stripes on one of Hortense's shoes -/
def hortense_stripes_per_shoe : ℕ := 2 * olga_stripes_per_shoe

/-- The number of stripes on one of Ethan's shoes -/
def ethan_stripes_per_shoe : ℕ := hortense_stripes_per_shoe + 2

/-- The total number of stripes on all shoes -/
def total_stripes : ℕ := 2 * (olga_stripes_per_shoe + rick_stripes_per_shoe + hortense_stripes_per_shoe + ethan_stripes_per_shoe)

/-- The final result after dividing by 2 and rounding up -/
def final_result : ℕ := (total_stripes + 1) / 2

theorem stripes_calculation :
  final_result = 19 := by sorry

end NUMINAMATH_CALUDE_stripes_calculation_l3555_355544


namespace NUMINAMATH_CALUDE_fraction_problem_l3555_355576

theorem fraction_problem (x : ℚ) (h : 75 * x = 37.5) : x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3555_355576


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3555_355564

/-- An isosceles right triangle with perimeter 3p has area (153 - 108√2) / 2 * p^2 -/
theorem isosceles_right_triangle_area (p : ℝ) (h : p > 0) :
  let perimeter := 3 * p
  let leg := (9 * p - 6 * p * Real.sqrt 2)
  let area := (1 / 2) * leg ^ 2
  area = (153 - 108 * Real.sqrt 2) / 2 * p ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3555_355564


namespace NUMINAMATH_CALUDE_work_done_is_four_l3555_355507

-- Define the force vector
def F : Fin 2 → ℝ := ![2, 3]

-- Define points A and B
def A : Fin 2 → ℝ := ![2, 0]
def B : Fin 2 → ℝ := ![4, 0]

-- Define the displacement vector
def displacement : Fin 2 → ℝ := ![B 0 - A 0, B 1 - A 1]

-- Define work as the dot product of force and displacement
def work : ℝ := (F 0 * displacement 0) + (F 1 * displacement 1)

-- Theorem statement
theorem work_done_is_four : work = 4 := by sorry

end NUMINAMATH_CALUDE_work_done_is_four_l3555_355507


namespace NUMINAMATH_CALUDE_abc_sum_sixteen_l3555_355574

theorem abc_sum_sixteen (a b c : ℤ) 
  (h1 : a ≥ 4) (h2 : b ≥ 4) (h3 : c ≥ 4)
  (h4 : ¬(a = b ∧ b = c))
  (h5 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) :
  a + b + c = 16 := by sorry

end NUMINAMATH_CALUDE_abc_sum_sixteen_l3555_355574


namespace NUMINAMATH_CALUDE_f_monotonic_iff_a_in_range_l3555_355569

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x - 7

-- Define the property of being monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem f_monotonic_iff_a_in_range (a : ℝ) :
  MonotonicallyIncreasing (f a) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_iff_a_in_range_l3555_355569


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3555_355559

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that
    ax^2 + bx + c = (px + q)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_condition (m : ℝ) :
  is_perfect_square_trinomial 1 m 16 → m = 8 ∨ m = -8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3555_355559


namespace NUMINAMATH_CALUDE_philatelist_stamps_problem_l3555_355506

theorem philatelist_stamps_problem :
  ∃! x : ℕ, x % 3 = 1 ∧ x % 5 = 3 ∧ x % 7 = 5 ∧ 150 < x ∧ x ≤ 300 ∧ x = 208 := by
  sorry

end NUMINAMATH_CALUDE_philatelist_stamps_problem_l3555_355506


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3555_355599

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (Complex.I : ℂ) / (2 + Complex.I) = ⟨a, b⟩ := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3555_355599


namespace NUMINAMATH_CALUDE_flower_cost_minimization_l3555_355537

/-- The cost of one lily in dollars -/
def lily_cost : ℝ := 5

/-- The cost of one carnation in dollars -/
def carnation_cost : ℝ := 6

/-- The total number of flowers to be bought -/
def total_flowers : ℕ := 12

/-- The minimum number of carnations to be bought -/
def min_carnations : ℕ := 5

/-- The cost function for buying x lilies -/
def cost_function (x : ℝ) : ℝ := -x + 72

theorem flower_cost_minimization :
  ∃ (x : ℝ),
    x ≤ total_flowers - min_carnations ∧
    x ≥ 0 ∧
    ∀ (y : ℝ),
      y ≤ total_flowers - min_carnations ∧
      y ≥ 0 →
      cost_function x ≤ cost_function y ∧
      cost_function x = 65 :=
sorry

end NUMINAMATH_CALUDE_flower_cost_minimization_l3555_355537


namespace NUMINAMATH_CALUDE_fraction_equals_870_l3555_355580

theorem fraction_equals_870 (a : ℕ+) :
  (a : ℚ) / ((a : ℚ) + 50) = 870 / 1000 → a = 335 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_870_l3555_355580


namespace NUMINAMATH_CALUDE_effective_area_percentage_difference_l3555_355533

/-- Calculates the effective area percentage difference between two circular fields -/
theorem effective_area_percentage_difference
  (r1 r2 : ℝ)  -- radii of the two fields
  (sqi1 sqi2 : ℝ)  -- soil quality indices
  (wa1 wa2 : ℝ)  -- water allocations
  (cyf1 cyf2 : ℝ)  -- crop yield factors
  (h_ratio : r2 = (10 / 4) * r1)  -- radius ratio condition
  (h_sqi1 : sqi1 = 0.8)
  (h_sqi2 : sqi2 = 1.2)
  (h_wa1 : wa1 = 15000)
  (h_wa2 : wa2 = 30000)
  (h_cyf1 : cyf1 = 1.5)
  (h_cyf2 : cyf2 = 2) :
  let ea1 := π * r1^2 * sqi1 * wa1 * cyf1
  let ea2 := π * r2^2 * sqi2 * wa2 * cyf2
  (ea2 - ea1) / ea1 * 100 = 1566.67 := by
  sorry

end NUMINAMATH_CALUDE_effective_area_percentage_difference_l3555_355533


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l3555_355575

-- Define the quadratic function
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

-- Define monotonicity in an open interval
def MonotonicIn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

-- Theorem statement
theorem quadratic_monotonicity (t : ℝ) :
  MonotonicIn (f t) 1 3 → t ≤ 1 ∨ t ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l3555_355575


namespace NUMINAMATH_CALUDE_gerbil_weight_l3555_355572

/-- The combined weight of two gerbils given the weights and relationships of three gerbils -/
theorem gerbil_weight (scruffy muffy puffy : ℕ) 
  (h1 : scruffy = 12)
  (h2 : muffy = scruffy - 3)
  (h3 : puffy = muffy + 5) :
  puffy + muffy = 23 := by
  sorry

end NUMINAMATH_CALUDE_gerbil_weight_l3555_355572


namespace NUMINAMATH_CALUDE_cupcakes_theorem_l3555_355546

/-- The number of children sharing the cupcakes -/
def num_children : ℕ := 8

/-- The number of cupcakes each child gets when shared equally -/
def cupcakes_per_child : ℕ := 12

/-- The total number of cupcakes -/
def total_cupcakes : ℕ := num_children * cupcakes_per_child

theorem cupcakes_theorem : total_cupcakes = 96 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_theorem_l3555_355546


namespace NUMINAMATH_CALUDE_isabella_currency_exchange_l3555_355586

/-- The exchange rate from U.S. dollars to Mexican pesos -/
def exchange_rate : ℚ := 13 / 9

/-- The amount spent in pesos -/
def spent : ℕ := 117

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem representing the problem -/
theorem isabella_currency_exchange :
  ∃ (d : ℕ),
    (exchange_rate * d - spent : ℚ) = d ∧
    d % 4 = 0 ∧
    d > 260 ∧
    d < 268 ∧
    sum_of_digits d = 12 := by
  sorry

end NUMINAMATH_CALUDE_isabella_currency_exchange_l3555_355586


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l3555_355515

theorem quadratic_rational_solutions : 
  ∃! (c₁ c₂ : ℕ+), 
    (∃ (x : ℚ), 7 * x^2 + 15 * x + c₁.val = 0) ∧ 
    (∃ (y : ℚ), 7 * y^2 + 15 * y + c₂.val = 0) ∧ 
    c₁ ≠ c₂ ∧ 
    c₁.val * c₂.val = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l3555_355515


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reversal_l3555_355509

/-- Returns true if n is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Returns true if n starts with the digit 3 -/
def starts_with_three (n : ℕ) : Prop := 30 ≤ n ∧ n ≤ 39

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The theorem stating that 53 is the smallest two-digit prime starting with 3 
    whose digit reversal is composite -/
theorem smallest_two_digit_prime_with_composite_reversal : 
  ∃ (n : ℕ), 
    is_two_digit n ∧ 
    starts_with_three n ∧ 
    Nat.Prime n ∧ 
    ¬(Nat.Prime (reverse_digits n)) ∧
    (∀ m : ℕ, m < n → 
      is_two_digit m → 
      starts_with_three m → 
      Nat.Prime m → 
      Nat.Prime (reverse_digits m)) ∧
    n = 53 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reversal_l3555_355509


namespace NUMINAMATH_CALUDE_condition_property_l3555_355522

theorem condition_property : 
  (∀ x : ℝ, x^2 - 2*x < 0 → x < 2) ∧ 
  ¬(∀ x : ℝ, x < 2 → x^2 - 2*x < 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_property_l3555_355522


namespace NUMINAMATH_CALUDE_bicycle_cost_is_150_l3555_355581

/-- The cost of the bicycle Patrick wants to buy. -/
def bicycle_cost : ℕ := 150

/-- The amount Patrick saved, which is half the price of the bicycle. -/
def patricks_savings : ℕ := bicycle_cost / 2

/-- The amount Patrick lent to his friend. -/
def lent_amount : ℕ := 50

/-- The amount Patrick has left after lending money to his friend. -/
def remaining_amount : ℕ := 25

/-- Theorem stating that the bicycle cost is 150, given the conditions. -/
theorem bicycle_cost_is_150 :
  patricks_savings - lent_amount = remaining_amount →
  bicycle_cost = 150 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_is_150_l3555_355581


namespace NUMINAMATH_CALUDE_simplify_expression_l3555_355585

theorem simplify_expression (x : ℝ) (h : x > 3) :
  3 * |3 - x| - |x^2 - 6*x + 10| + |x^2 - 2*x + 1| = 7*x - 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3555_355585


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3555_355547

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℤ, 
    3 * X^4 + 8 * X^3 - 29 * X^2 - 17 * X + 34 = 
    (X^2 + 5 * X - 3) * q + (79 * X - 11) ∧ 
    (79 * X - 11).degree < (X^2 + 5 * X - 3).degree :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3555_355547


namespace NUMINAMATH_CALUDE_right_triangle_area_l3555_355540

theorem right_triangle_area (a b c : ℝ) (h1 : a = 40) (h2 : c = 41) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 180 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3555_355540


namespace NUMINAMATH_CALUDE_angle_triple_complement_l3555_355568

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l3555_355568


namespace NUMINAMATH_CALUDE_electricity_cost_per_watt_l3555_355508

theorem electricity_cost_per_watt 
  (watts : ℕ) 
  (late_fee : ℕ) 
  (total_payment : ℕ) 
  (h1 : watts = 300)
  (h2 : late_fee = 150)
  (h3 : total_payment = 1350) :
  (total_payment - late_fee) / watts = 4 := by
  sorry

end NUMINAMATH_CALUDE_electricity_cost_per_watt_l3555_355508


namespace NUMINAMATH_CALUDE_sector_to_cone_l3555_355563

/-- Given a 300° sector of a circle with radius 12, prove it forms a cone with base radius 10 and slant height 12 -/
theorem sector_to_cone (r : ℝ) (angle : ℝ) :
  r = 12 →
  angle = 300 * (π / 180) →
  ∃ (base_radius slant_height : ℝ),
    base_radius = 10 ∧
    slant_height = r ∧
    2 * π * base_radius = angle * r :=
by sorry

end NUMINAMATH_CALUDE_sector_to_cone_l3555_355563


namespace NUMINAMATH_CALUDE_coefficient_x_10_l3555_355501

/-- The coefficient of x^10 in the expansion of (x^3/3 - 3/x^2)^10 is 17010/729 -/
theorem coefficient_x_10 : 
  let f (x : ℚ) := (x^3 / 3 - 3 / x^2)^10
  ∃ (c : ℚ), c = 17010 / 729 ∧ 
    ∃ (g : ℚ → ℚ), (∀ x, x ≠ 0 → f x = c * x^10 + x * g x) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_10_l3555_355501


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_2_range_of_a_for_solutions_l3555_355520

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + a|

-- Theorem for part I
theorem solution_set_for_a_equals_2 :
  {x : ℝ | f x 2 > 6} = {x : ℝ | x < -3 ∨ x > 3} := by sorry

-- Theorem for part II
theorem range_of_a_for_solutions :
  {a : ℝ | ∃ x, f x a < a^2 - 1} = {a : ℝ | a < -1 - Real.sqrt 2 ∨ a > 1 + Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_2_range_of_a_for_solutions_l3555_355520


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3555_355589

theorem smallest_solution_floor_equation : 
  ∀ x : ℝ, (x ≥ Real.sqrt 119 ∧ ⌊x^2⌋ - ⌊x⌋^2 = 19) ∨ (x < Real.sqrt 119 ∧ ⌊x^2⌋ - ⌊x⌋^2 ≠ 19) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3555_355589


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3555_355553

theorem gcd_of_specific_numbers :
  let m : ℕ := 33333333
  let n : ℕ := 666666666
  Nat.gcd m n = 3 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3555_355553


namespace NUMINAMATH_CALUDE_sqrt_12_plus_3_minus_pi_pow_0_plus_abs_1_minus_sqrt_3_equals_3_sqrt_3_l3555_355578

theorem sqrt_12_plus_3_minus_pi_pow_0_plus_abs_1_minus_sqrt_3_equals_3_sqrt_3 :
  Real.sqrt 12 + (3 - Real.pi) ^ (0 : ℕ) + |1 - Real.sqrt 3| = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_plus_3_minus_pi_pow_0_plus_abs_1_minus_sqrt_3_equals_3_sqrt_3_l3555_355578


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3555_355543

theorem largest_n_satisfying_inequality :
  ∀ n : ℕ, n ≤ 9 ↔ (1 / 4 : ℚ) + (n / 8 : ℚ) < (3 / 2 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3555_355543


namespace NUMINAMATH_CALUDE_common_difference_is_four_l3555_355532

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_correct : ∀ n, S n = (n * (a 0 + a (n - 1))) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ := seq.a 1 - seq.a 0

theorem common_difference_is_four (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 - 3 * seq.S 2 = 12) : 
  common_difference seq = 4 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_four_l3555_355532


namespace NUMINAMATH_CALUDE_sin_equality_in_range_l3555_355552

theorem sin_equality_in_range (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 → Real.sin (n * π / 180) = Real.sin (750 * π / 180) → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_in_range_l3555_355552


namespace NUMINAMATH_CALUDE_height_percentage_difference_l3555_355555

theorem height_percentage_difference (P Q : ℝ) (h : Q = P * (1 + 66.67 / 100)) :
  P = Q * (1 - 40 / 100) :=
sorry

end NUMINAMATH_CALUDE_height_percentage_difference_l3555_355555


namespace NUMINAMATH_CALUDE_number_problem_l3555_355545

theorem number_problem (x : ℝ) : 0.65 * x = 0.8 * x - 21 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3555_355545


namespace NUMINAMATH_CALUDE_derivative_at_point_is_constant_l3555_355535

/-- The derivative of a function at a point is a constant value. -/
theorem derivative_at_point_is_constant (f : ℝ → ℝ) (a : ℝ) : 
  ∃ (c : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |x - a| ≠ 0 → 
    |(f x - f a) / (x - a) - c| < ε :=
sorry

end NUMINAMATH_CALUDE_derivative_at_point_is_constant_l3555_355535


namespace NUMINAMATH_CALUDE_g_of_one_eq_neg_three_l3555_355598

-- Define the function f
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

-- State the theorem
theorem g_of_one_eq_neg_three
  (g : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f g (-x) + f g x = 0)
  (h2 : g (-1) = 1) :
  g 1 = -3 :=
by sorry

end NUMINAMATH_CALUDE_g_of_one_eq_neg_three_l3555_355598


namespace NUMINAMATH_CALUDE_eventually_monotonic_sequence_l3555_355538

/-- An infinite sequence of real numbers where no two members are equal -/
def UniqueMemberSequence (a : ℕ → ℝ) : Prop :=
  ∀ i j, i ≠ j → a i ≠ a j

/-- A monotonic segment of length m starting at index i -/
def MonotonicSegment (a : ℕ → ℝ) (i m : ℕ) : Prop :=
  (∀ k, k < m - 1 → a (i + k) < a (i + k + 1)) ∨
  (∀ k, k < m - 1 → a (i + k) > a (i + k + 1))

/-- For each natural k, the term aₖ is contained in some monotonic segment of length k + 1 -/
def ContainedInMonotonicSegment (a : ℕ → ℝ) : Prop :=
  ∀ k, ∃ i, k ∈ Finset.range (k + 1) ∧ MonotonicSegment a i (k + 1)

/-- The sequence is eventually monotonic -/
def EventuallyMonotonic (a : ℕ → ℝ) : Prop :=
  ∃ N, (∀ n ≥ N, a n < a (n + 1)) ∨ (∀ n ≥ N, a n > a (n + 1))

theorem eventually_monotonic_sequence
  (a : ℕ → ℝ)
  (h1 : UniqueMemberSequence a)
  (h2 : ContainedInMonotonicSegment a) :
  EventuallyMonotonic a :=
sorry

end NUMINAMATH_CALUDE_eventually_monotonic_sequence_l3555_355538


namespace NUMINAMATH_CALUDE_lcm_of_proportional_numbers_l3555_355526

def A : ℕ := 18
def B : ℕ := 24
def C : ℕ := 30

theorem lcm_of_proportional_numbers :
  (A : ℕ) / gcd A B = 3 ∧
  (B : ℕ) / gcd A B = 4 ∧
  (C : ℕ) / gcd A B = 5 ∧
  gcd A (gcd B C) = 6 ∧
  12 ∣ lcm A (lcm B C) →
  lcm A (lcm B C) = 360 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_proportional_numbers_l3555_355526


namespace NUMINAMATH_CALUDE_sqrt_two_f_pi_fourth_plus_f_neg_pi_sixth_pos_l3555_355514

open Real

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define f' as the derivative of f
variable (f' : ℝ → ℝ)

-- Axiom: f is an odd function
axiom f_odd (x : ℝ) : f (-x) = -f x

-- Axiom: f' is the derivative of f
axiom f'_is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Axiom: For x in (0, π/2) ∪ (π/2, π), f(x) + f'(x)tan(x) > 0
axiom f_plus_f'_tan_pos (x : ℝ) (h1 : 0 < x) (h2 : x < π) (h3 : x ≠ π/2) :
  f x + f' x * tan x > 0

-- Theorem to prove
theorem sqrt_two_f_pi_fourth_plus_f_neg_pi_sixth_pos :
  Real.sqrt 2 * f (π/4) + f (-π/6) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_f_pi_fourth_plus_f_neg_pi_sixth_pos_l3555_355514


namespace NUMINAMATH_CALUDE_negation_of_neither_odd_l3555_355565

theorem negation_of_neither_odd (a b : ℤ) :
  ¬(¬(Odd a) ∧ ¬(Odd b)) ↔ Odd a ∨ Odd b := by sorry

end NUMINAMATH_CALUDE_negation_of_neither_odd_l3555_355565


namespace NUMINAMATH_CALUDE_students_per_group_l3555_355516

theorem students_per_group 
  (total_students : ℕ) 
  (num_teachers : ℕ) 
  (h1 : total_students = 256) 
  (h2 : num_teachers = 8) 
  (h3 : num_teachers > 0) :
  total_students / num_teachers = 32 := by
  sorry

end NUMINAMATH_CALUDE_students_per_group_l3555_355516


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_unit_interval_l3555_355596

-- Define the sets M and N
def M : Set ℝ := {x | x^2 ≤ 1}
def N : Set (ℝ × ℝ) := {p | p.2 ∈ M ∧ p.1 = p.2^2}

-- State the theorem
theorem M_intersect_N_eq_unit_interval :
  (M ∩ (N.image Prod.snd)) = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_unit_interval_l3555_355596


namespace NUMINAMATH_CALUDE_three_point_five_million_scientific_notation_l3555_355502

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- Definition of 3.5 million -/
def three_point_five_million : ℝ := 3.5e6

/-- Theorem stating that 3.5 million can be expressed as 3.5 × 10^6 in scientific notation -/
theorem three_point_five_million_scientific_notation :
  ∃ (sn : ScientificNotation), three_point_five_million = sn.a * (10 : ℝ) ^ sn.n :=
sorry

end NUMINAMATH_CALUDE_three_point_five_million_scientific_notation_l3555_355502


namespace NUMINAMATH_CALUDE_income_ratio_l3555_355527

/-- Represents a person's financial information -/
structure Person where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup -/
def problem_setup (p1 p2 : Person) : Prop :=
  p1.income = 5000 ∧
  p1.savings = 2000 ∧
  p2.savings = 2000 ∧
  3 * p2.expenditure = 2 * p1.expenditure ∧
  p1.income = p1.expenditure + p1.savings ∧
  p2.income = p2.expenditure + p2.savings

/-- The theorem to prove -/
theorem income_ratio (p1 p2 : Person) :
  problem_setup p1 p2 → 5 * p2.income = 4 * p1.income :=
by
  sorry


end NUMINAMATH_CALUDE_income_ratio_l3555_355527


namespace NUMINAMATH_CALUDE_rational_equation_proof_l3555_355594

theorem rational_equation_proof (x y : ℚ) 
  (h : |x + 2017| + (y - 2017)^2 = 0) : 
  (x / y)^2017 = -1 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_proof_l3555_355594


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3555_355525

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3555_355525


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3555_355590

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Calculates the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + seq.diff * (n - 1)

theorem arithmetic_sequence_problem :
  ∃ (row col1 col2 : ArithmeticSequence),
    row.first = 15 ∧
    row.nthTerm 4 = 2 ∧
    col1.nthTerm 2 = 14 ∧
    col1.nthTerm 3 = 10 ∧
    col2.nthTerm 5 = -21 ∧
    col2.first = -13.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3555_355590


namespace NUMINAMATH_CALUDE_project_work_difference_l3555_355591

/-- Represents the work hours of three people on a project -/
structure ProjectWork where
  person1 : ℝ
  person2 : ℝ
  person3 : ℝ

/-- The conditions of the project work -/
def validProjectWork (work : ProjectWork) : Prop :=
  work.person1 > 0 ∧ work.person2 > 0 ∧ work.person3 > 0 ∧
  work.person2 = 2 * work.person1 ∧
  work.person3 = 3 * work.person1 ∧
  work.person1 + work.person2 + work.person3 = 120

theorem project_work_difference (work : ProjectWork) 
  (h : validProjectWork work) : 
  work.person3 - work.person1 = 40 := by
  sorry

#check project_work_difference

end NUMINAMATH_CALUDE_project_work_difference_l3555_355591


namespace NUMINAMATH_CALUDE_factor_tree_X_value_l3555_355503

theorem factor_tree_X_value :
  ∀ (X Y Z F G : ℕ),
    X = Y * Z →
    Y = 7 * F →
    Z = 11 * G →
    F = 7 * 3 →
    G = 11 * 3 →
    X = 53361 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_X_value_l3555_355503


namespace NUMINAMATH_CALUDE_max_abs_sum_l3555_355567

theorem max_abs_sum (a b c : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) → 
  |a| + |b| + |c| ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_l3555_355567


namespace NUMINAMATH_CALUDE_chessboard_properties_l3555_355557

/-- Represents a chessboard grid -/
structure ChessboardGrid where
  size : Nat
  perimeter : ℝ

/-- Calculates the number of pencil lifts required to draw the grid -/
def pencilLifts (grid : ChessboardGrid) : Nat :=
  sorry

/-- Calculates the shortest path length to cover the entire grid -/
def shortestPathLength (grid : ChessboardGrid) : ℝ :=
  sorry

/-- Theorem stating the properties of an 8x8 chessboard grid -/
theorem chessboard_properties :
  let grid : ChessboardGrid := ⟨8, 128⟩
  pencilLifts grid = 14 ∧ shortestPathLength grid = 632 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_properties_l3555_355557


namespace NUMINAMATH_CALUDE_special_triangle_angles_l3555_355513

/-- A triangle with a 90° angle that is three times the smallest angle has angles 90°, 60°, and 30° and is right-angled. -/
theorem special_triangle_angles :
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  a = 90 →
  a = 3 * c →
  (a = 90 ∧ b = 60 ∧ c = 30) ∧ (∃ x, x = 90) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_angles_l3555_355513


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3555_355510

theorem min_value_of_expression (a b : ℕ+) (h : a > b) :
  let E := |(a + 2*b : ℝ) / (a - b : ℝ) + (a - b : ℝ) / (a + 2*b : ℝ)|
  ∀ x : ℝ, E ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3555_355510


namespace NUMINAMATH_CALUDE_daniel_and_elsie_crackers_l3555_355593

/-- The amount of crackers Matthew had initially -/
def initial_crackers : ℝ := 27.5

/-- The amount of crackers Ally ate -/
def ally_crackers : ℝ := 3.5

/-- The amount of crackers Bob ate -/
def bob_crackers : ℝ := 4

/-- The amount of crackers Clair ate -/
def clair_crackers : ℝ := 5.5

/-- The amount of crackers Matthew had left after giving to Ally, Bob, and Clair -/
def remaining_crackers : ℝ := 10.5

/-- The theorem stating that Daniel and Elsie ate 4 crackers combined -/
theorem daniel_and_elsie_crackers : 
  initial_crackers - (ally_crackers + bob_crackers + clair_crackers) - remaining_crackers = 4 := by
  sorry

end NUMINAMATH_CALUDE_daniel_and_elsie_crackers_l3555_355593


namespace NUMINAMATH_CALUDE_salary_of_C_salary_C_is_11000_l3555_355523

-- Define the salaries as natural numbers (assuming whole rupees)
def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

-- Define the average salary
def average_salary : ℕ := 8000

-- Theorem to prove
theorem salary_of_C : ℕ :=
  let total_salary := salary_A + salary_B + salary_D + salary_E
  let salary_C := 5 * average_salary - total_salary
  salary_C

-- Proof (skipped)
theorem salary_C_is_11000 : salary_of_C = 11000 := by
  sorry

end NUMINAMATH_CALUDE_salary_of_C_salary_C_is_11000_l3555_355523


namespace NUMINAMATH_CALUDE_ian_painted_cuboids_l3555_355570

/-- The number of cuboids painted by Ian -/
def num_cuboids : ℕ := 8

/-- The total number of faces painted -/
def total_faces : ℕ := 48

/-- The number of faces on one cuboid -/
def faces_per_cuboid : ℕ := 6

/-- Theorem: The number of cuboids painted is equal to 8 -/
theorem ian_painted_cuboids : 
  num_cuboids = total_faces / faces_per_cuboid :=
sorry

end NUMINAMATH_CALUDE_ian_painted_cuboids_l3555_355570


namespace NUMINAMATH_CALUDE_battle_station_staffing_l3555_355524

theorem battle_station_staffing (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (n.descFactorial k) = 30240 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l3555_355524


namespace NUMINAMATH_CALUDE_raghu_investment_l3555_355511

/-- Represents the investment amounts of Raghu, Trishul, and Vishal --/
structure Investments where
  raghu : ℝ
  trishul : ℝ
  vishal : ℝ

/-- Defines the conditions of the investment problem --/
def InvestmentConditions (i : Investments) : Prop :=
  i.trishul = 0.9 * i.raghu ∧
  i.vishal = 1.1 * i.trishul ∧
  i.raghu + i.trishul + i.vishal = 7225

/-- Theorem stating that under the given conditions, Raghu's investment is 2500 --/
theorem raghu_investment (i : Investments) (h : InvestmentConditions i) : i.raghu = 2500 := by
  sorry


end NUMINAMATH_CALUDE_raghu_investment_l3555_355511


namespace NUMINAMATH_CALUDE_expression_equality_l3555_355551

theorem expression_equality : (2^1501 + 5^1500)^2 - (2^1501 - 5^1500)^2 = 8 * 10^1500 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3555_355551


namespace NUMINAMATH_CALUDE_joint_club_afternoon_solution_l3555_355531

/-- Represents the joint club afternoon scenario with two classes -/
structure JointClubAfternoon where
  a : ℕ  -- number of students in class A
  b : ℕ  -- number of students in class B
  K : ℕ  -- the amount each student would pay if one class covered all costs

/-- Conditions for the joint club afternoon -/
def scenario (j : JointClubAfternoon) : Prop :=
  -- Total contribution for the first event
  5 * j.a + 3 * j.b = j.K * j.a
  ∧
  -- Total contribution for the second event
  4 * j.a + 6 * j.b = j.K * j.b
  ∧
  -- Class B has more students than class A
  j.b > j.a

/-- Theorem stating the solution to the problem -/
theorem joint_club_afternoon_solution (j : JointClubAfternoon) 
  (h : scenario j) : j.K = 9 ∧ j.b > j.a := by
  sorry


end NUMINAMATH_CALUDE_joint_club_afternoon_solution_l3555_355531


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3555_355579

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {y | ∃ x, y = x^2}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3555_355579


namespace NUMINAMATH_CALUDE_ali_boxes_calculation_l3555_355592

/-- The number of boxes Ali used for each of his circles -/
def ali_boxes_per_circle : ℕ := 14

/-- The total number of boxes -/
def total_boxes : ℕ := 80

/-- The number of circles Ali made -/
def ali_circles : ℕ := 5

/-- The number of boxes Ernie used for his circle -/
def ernie_boxes : ℕ := 10

theorem ali_boxes_calculation :
  ali_boxes_per_circle * ali_circles + ernie_boxes = total_boxes :=
by sorry

end NUMINAMATH_CALUDE_ali_boxes_calculation_l3555_355592


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l3555_355556

theorem tangent_line_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : ∃ x₀ : ℝ, (2 * x₀ - a = Real.log (2 * x₀ + b)) ∧ 
    (∀ x : ℝ, 2 * x - a ≤ Real.log (2 * x + b))) :
  (4 / a + 1 / b) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l3555_355556


namespace NUMINAMATH_CALUDE_bananas_undetermined_l3555_355505

/-- Represents Philip's fruit collection -/
structure FruitCollection where
  totalOranges : ℕ
  orangeGroups : ℕ
  orangesPerGroup : ℕ
  bananaGroups : ℕ

/-- Philip's actual fruit collection -/
def philipsCollection : FruitCollection := {
  totalOranges := 384,
  orangeGroups := 16,
  orangesPerGroup := 24,
  bananaGroups := 345
}

/-- Predicate to check if the number of bananas can be determined -/
def canDetermineBananas (c : FruitCollection) : Prop :=
  ∃ (bananasPerGroup : ℕ), True  -- Placeholder, always true

/-- Theorem stating that the number of bananas cannot be determined -/
theorem bananas_undetermined (c : FruitCollection) 
  (h1 : c.totalOranges = c.orangeGroups * c.orangesPerGroup) :
  ¬ canDetermineBananas c := by
  sorry

#check bananas_undetermined philipsCollection

end NUMINAMATH_CALUDE_bananas_undetermined_l3555_355505


namespace NUMINAMATH_CALUDE_sum_of_squares_of_consecutive_even_numbers_l3555_355558

theorem sum_of_squares_of_consecutive_even_numbers : 
  ∃ (a b c d : ℕ), 
    (∃ (n : ℕ), a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) ∧ 
    a + b + c + d = 36 → 
    a^2 + b^2 + c^2 + d^2 = 344 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_consecutive_even_numbers_l3555_355558


namespace NUMINAMATH_CALUDE_consecutive_nonprime_integers_l3555_355521

theorem consecutive_nonprime_integers : ∃ (a : ℕ),
  (25 < a) ∧
  (a + 4 < 50) ∧
  (¬ Nat.Prime a) ∧
  (¬ Nat.Prime (a + 1)) ∧
  (¬ Nat.Prime (a + 2)) ∧
  (¬ Nat.Prime (a + 3)) ∧
  (¬ Nat.Prime (a + 4)) ∧
  ((a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) % 10 = 0) ∧
  (a + 4 = 36) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_nonprime_integers_l3555_355521


namespace NUMINAMATH_CALUDE_all_reachable_l3555_355528

def step (x : ℚ) : Set ℚ := {x + 1, -1 / x}

def reachable : Set ℚ → Prop :=
  λ S => ∀ y ∈ S, ∃ n : ℕ, ∃ f : ℕ → ℚ,
    f 0 = 1 ∧ (∀ i < n, f (i + 1) ∈ step (f i)) ∧ f n = y

theorem all_reachable : reachable {-2, 1/2, 5/3, 7} := by
  sorry

end NUMINAMATH_CALUDE_all_reachable_l3555_355528


namespace NUMINAMATH_CALUDE_system_and_linear_equation_solution_l3555_355562

theorem system_and_linear_equation_solution (a : ℝ) :
  (∃ x y : ℝ, x + y = a ∧ x - y = 4*a ∧ 3*x - 5*y - 90 = 0) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_system_and_linear_equation_solution_l3555_355562


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3555_355577

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | 3*x - 2 ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3555_355577


namespace NUMINAMATH_CALUDE_solution_of_equation_l3555_355549

theorem solution_of_equation (x : ℚ) : 2/3 - 1/4 = 1/x → x = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l3555_355549


namespace NUMINAMATH_CALUDE_triangle_problem_l3555_355588

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 2 * Real.sqrt 3 →
  B = π / 3 ∧ 
  (∀ (a' c' : ℝ), a' * c' ≤ 12) ∧
  (∃ (a' c' : ℝ), a' * c' = 12) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3555_355588


namespace NUMINAMATH_CALUDE_sixth_angle_measure_l3555_355504

/-- The sum of internal angles of a hexagon is 720 degrees -/
def hexagon_angle_sum : ℝ := 720

/-- The sum of the five known angles in the hexagon -/
def known_angles_sum : ℝ := 130 + 100 + 105 + 115 + 95

/-- Theorem: In a hexagon where five of the internal angles measure 130°, 100°, 105°, 115°, and 95°,
    the measure of the sixth angle is 175°. -/
theorem sixth_angle_measure :
  hexagon_angle_sum - known_angles_sum = 175 := by sorry

end NUMINAMATH_CALUDE_sixth_angle_measure_l3555_355504


namespace NUMINAMATH_CALUDE_max_elephants_is_1036_l3555_355573

/-- The number of union members --/
def union_members : ℕ := 28

/-- The number of non-union members --/
def non_union_members : ℕ := 37

/-- The function that calculates the total number of elephants given the number
    of elephants per union member and per non-union member --/
def total_elephants (elephants_per_union : ℕ) (elephants_per_non_union : ℕ) : ℕ :=
  union_members * elephants_per_union + non_union_members * elephants_per_non_union

/-- The theorem stating that 1036 is the maximum number of elephants that can be distributed --/
theorem max_elephants_is_1036 :
  ∃ (eu en : ℕ), 
    eu ≠ en ∧ 
    eu > 0 ∧ 
    en > 0 ∧
    total_elephants eu en = 1036 ∧
    (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → total_elephants x y ≤ 1036) :=
sorry

end NUMINAMATH_CALUDE_max_elephants_is_1036_l3555_355573


namespace NUMINAMATH_CALUDE_simple_interest_principal_l3555_355582

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 1000)
  (h2 : rate = 10)
  (h3 : time = 4)
  : interest = (2500 * rate * time) / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l3555_355582


namespace NUMINAMATH_CALUDE_integer_fraction_sum_equals_three_l3555_355536

theorem integer_fraction_sum_equals_three (a b : ℕ+) :
  let A := (a + 1 : ℝ) / b + b / a
  (∃ k : ℤ, A = k) → A = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_sum_equals_three_l3555_355536


namespace NUMINAMATH_CALUDE_pants_cost_is_correct_l3555_355584

/-- The cost of one pair of pants in dollars -/
def pants_cost : ℝ := 80

/-- The cost of one T-shirt in dollars -/
def tshirt_cost : ℝ := 20

/-- The cost of one pair of shoes in dollars -/
def shoes_cost : ℝ := 150

/-- The discount rate applied to all items -/
def discount_rate : ℝ := 0.1

/-- The total cost after discount for Eugene's purchase -/
def total_cost_after_discount : ℝ := 558

theorem pants_cost_is_correct : 
  (4 * tshirt_cost + 3 * pants_cost + 2 * shoes_cost) * (1 - discount_rate) = total_cost_after_discount :=
by sorry

end NUMINAMATH_CALUDE_pants_cost_is_correct_l3555_355584


namespace NUMINAMATH_CALUDE_order_relation_l3555_355587

theorem order_relation (a b c : ℝ) : 
  a = (1 : ℝ) / 2023 →
  b = Real.exp (-(2022 : ℝ) / 2023) →
  c = Real.cos ((1 : ℝ) / 2023) / 2023 →
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_order_relation_l3555_355587


namespace NUMINAMATH_CALUDE_sequence_problem_l3555_355541

theorem sequence_problem (a b : ℝ) : 
  (∃ r : ℝ, 10 * r = a ∧ a * r = 1/2) →  -- geometric sequence condition
  (∃ d : ℝ, a + d = 5 ∧ 5 + d = b) →    -- arithmetic sequence condition
  a = Real.sqrt 5 ∧ b = 10 - Real.sqrt 5 := by
sorry


end NUMINAMATH_CALUDE_sequence_problem_l3555_355541
