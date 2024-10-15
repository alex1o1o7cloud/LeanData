import Mathlib

namespace NUMINAMATH_CALUDE_probability_theorem_l2347_234753

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ (m : ℕ), a * b + a + b = 6 * m - 1

def total_pairs : ℕ := Nat.choose 60 2

def favorable_pairs : ℕ := total_pairs - Nat.choose 50 2

theorem probability_theorem :
  (favorable_pairs : ℚ) / total_pairs = 91 / 295 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2347_234753


namespace NUMINAMATH_CALUDE_base3_to_base10_equality_l2347_234789

/-- Converts a base-3 number to base-10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- The base-3 representation of the number --/
def base3Number : List Nat := [1, 2, 0, 1, 2]

/-- Theorem stating that the base-3 number 12012 is equal to 140 in base-10 --/
theorem base3_to_base10_equality : base3ToBase10 base3Number = 140 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_equality_l2347_234789


namespace NUMINAMATH_CALUDE_average_income_proof_l2347_234778

def family_size : ℕ := 4

def income_1 : ℕ := 8000
def income_2 : ℕ := 15000
def income_3 : ℕ := 6000
def income_4 : ℕ := 11000

def total_income : ℕ := income_1 + income_2 + income_3 + income_4

theorem average_income_proof :
  total_income / family_size = 10000 := by
  sorry

end NUMINAMATH_CALUDE_average_income_proof_l2347_234778


namespace NUMINAMATH_CALUDE_nicks_nacks_nocks_conversion_l2347_234791

/-- Given the conversion rates between nicks, nacks, and nocks, 
    prove that 40 nocks is equal to 160/3 nicks. -/
theorem nicks_nacks_nocks_conversion 
  (h1 : (5 : ℚ) * nick = 3 * nack)
  (h2 : (4 : ℚ) * nack = 5 * nock)
  : (40 : ℚ) * nock = 160 / 3 * nick :=
by sorry

end NUMINAMATH_CALUDE_nicks_nacks_nocks_conversion_l2347_234791


namespace NUMINAMATH_CALUDE_orchard_trees_l2347_234746

theorem orchard_trees (total : ℕ) (pure_fuji : ℕ) (cross_pollinated : ℕ) (pure_gala : ℕ) :
  pure_gala = 39 →
  cross_pollinated = (total : ℚ) * (1 / 10) →
  pure_fuji = (total : ℚ) * (3 / 4) →
  pure_fuji + pure_gala + cross_pollinated = total →
  pure_fuji + cross_pollinated = 221 := by
sorry

end NUMINAMATH_CALUDE_orchard_trees_l2347_234746


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2347_234773

theorem diophantine_equation_solutions (n : ℕ+) :
  ∃ (S : Finset (ℤ × ℤ)), S.card ≥ n ∧ ∀ (p : ℤ × ℤ), p ∈ S → p.1^2 + 15 * p.2^2 = 4^(n : ℕ) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2347_234773


namespace NUMINAMATH_CALUDE_intersection_point_of_linear_function_and_inverse_l2347_234767

-- Define the function f
def f (b : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + b

-- Define the theorem
theorem intersection_point_of_linear_function_and_inverse
  (b : ℤ) (a : ℤ) :
  (f b (-4) = a ∧ f b a = -4) → a = -4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_linear_function_and_inverse_l2347_234767


namespace NUMINAMATH_CALUDE_sqrt_six_and_quarter_equals_five_halves_l2347_234797

theorem sqrt_six_and_quarter_equals_five_halves :
  Real.sqrt (6 + 1/4) = 5/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_six_and_quarter_equals_five_halves_l2347_234797


namespace NUMINAMATH_CALUDE_inverse_exponential_function_l2347_234721

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => Real.log x / Real.log a

theorem inverse_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := f a
  (∀ x, f (a^x) = x) ∧ (∀ y, a^(f y) = y) ∧ f 2 = 1 → f = fun x => Real.log x / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_exponential_function_l2347_234721


namespace NUMINAMATH_CALUDE_vasya_counts_more_apples_CD_l2347_234775

/-- Represents the number of apple trees around the circular lake -/
def n : ℕ := sorry

/-- Represents the total number of apples on all trees -/
def m : ℕ := sorry

/-- Represents the number of trees Vasya counts from A to B -/
def vasya_trees_AB : ℕ := n / 3

/-- Represents the number of trees Petya counts from A to B -/
def petya_trees_AB : ℕ := 2 * n / 3

/-- Represents the number of apples Vasya counts from A to B -/
def vasya_apples_AB : ℕ := m / 8

/-- Represents the number of apples Petya counts from A to B -/
def petya_apples_AB : ℕ := 7 * m / 8

/-- Represents the number of trees Vasya counts from B to C -/
def vasya_trees_BC : ℕ := n / 3

/-- Represents the number of trees Petya counts from B to C -/
def petya_trees_BC : ℕ := 2 * n / 3

/-- Represents the number of apples Vasya counts from B to C -/
def vasya_apples_BC : ℕ := m / 8

/-- Represents the number of apples Petya counts from B to C -/
def petya_apples_BC : ℕ := 7 * m / 8

/-- Represents the number of trees Vasya counts from C to D -/
def vasya_trees_CD : ℕ := n / 3

/-- Represents the number of trees Petya counts from C to D -/
def petya_trees_CD : ℕ := 2 * n / 3

/-- Theorem stating that Vasya counts 3 times more apples than Petya from C to D -/
theorem vasya_counts_more_apples_CD :
  (m - vasya_apples_AB - vasya_apples_BC) = 3 * (m - petya_apples_AB - petya_apples_BC) :=
by sorry

end NUMINAMATH_CALUDE_vasya_counts_more_apples_CD_l2347_234775


namespace NUMINAMATH_CALUDE_common_ratio_is_two_l2347_234770

/-- An increasing geometric sequence with specific conditions -/
structure IncreasingGeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  is_increasing : q > 1
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * q
  a2_eq_2 : a 2 = 2
  a4_minus_a3_eq_4 : a 4 - a 3 = 4

/-- The common ratio of the increasing geometric sequence is 2 -/
theorem common_ratio_is_two (seq : IncreasingGeometricSequence) : seq.q = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_is_two_l2347_234770


namespace NUMINAMATH_CALUDE_fish_disappeared_l2347_234765

def original_goldfish : ℕ := 7
def original_catfish : ℕ := 12
def original_guppies : ℕ := 8
def original_angelfish : ℕ := 5
def current_total : ℕ := 27

theorem fish_disappeared : 
  original_goldfish + original_catfish + original_guppies + original_angelfish - current_total = 5 := by
  sorry

end NUMINAMATH_CALUDE_fish_disappeared_l2347_234765


namespace NUMINAMATH_CALUDE_homework_duration_decrease_l2347_234743

/-- Represents the decrease in homework duration over two adjustments --/
theorem homework_duration_decrease (initial_duration final_duration : ℝ) (x : ℝ) :
  initial_duration = 120 →
  final_duration = 60 →
  initial_duration * (1 - x)^2 = final_duration :=
by sorry

end NUMINAMATH_CALUDE_homework_duration_decrease_l2347_234743


namespace NUMINAMATH_CALUDE_sun_rise_position_l2347_234747

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the visibility of a circle above a line -/
inductive Visibility
  | Small
  | Half
  | Full

/-- Determines the positional relationship between a line and a circle -/
inductive PositionalRelationship
  | Tangent
  | Separate
  | ExternallyTangent
  | Intersecting

/-- 
  Given a circle and a line where only a small portion of the circle is visible above the line,
  prove that the positional relationship between the line and circle is intersecting.
-/
theorem sun_rise_position (c : Circle) (l : Line) (v : Visibility) :
  v = Visibility.Small → PositionalRelationship.Intersecting = 
    (let relationship := sorry -- Define the actual relationship based on c and l
     relationship) := by
  sorry


end NUMINAMATH_CALUDE_sun_rise_position_l2347_234747


namespace NUMINAMATH_CALUDE_linear_equation_integer_solution_l2347_234732

theorem linear_equation_integer_solution : ∃ (x y : ℤ), 2 * x + y - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_integer_solution_l2347_234732


namespace NUMINAMATH_CALUDE_not_geometric_complement_sequence_l2347_234719

/-- Given a geometric sequence a, b, c with common ratio q ≠ 1,
    prove that 1-a, 1-b, 1-c cannot form a geometric sequence. -/
theorem not_geometric_complement_sequence 
  (a b c q : ℝ) 
  (h1 : b = a * q) 
  (h2 : c = b * q) 
  (h3 : q ≠ 1) : 
  ¬ ∃ r : ℝ, (1 - b = (1 - a) * r ∧ 1 - c = (1 - b) * r) :=
sorry

end NUMINAMATH_CALUDE_not_geometric_complement_sequence_l2347_234719


namespace NUMINAMATH_CALUDE_number_of_triangles_triangles_in_figure_l2347_234792

/-- The number of triangles in a figure with 9 lines and 25 intersection points -/
theorem number_of_triangles (num_lines : ℕ) (num_intersections : ℕ) : ℕ :=
  let total_combinations := (num_lines.choose 3)
  total_combinations - num_intersections

/-- Proof that the number of triangles in the given figure is 59 -/
theorem triangles_in_figure : number_of_triangles 9 25 = 59 := by
  sorry

end NUMINAMATH_CALUDE_number_of_triangles_triangles_in_figure_l2347_234792


namespace NUMINAMATH_CALUDE_remaining_water_volume_l2347_234737

/-- Given a cup with 2 liters of water, after pouring out x milliliters 4 times, 
    the remaining volume in milliliters is equal to 2000 - 4x. -/
theorem remaining_water_volume (x : ℝ) : 
  2000 - 4 * x = (2 : ℝ) * 1000 - 4 * x := by sorry

end NUMINAMATH_CALUDE_remaining_water_volume_l2347_234737


namespace NUMINAMATH_CALUDE_probability_from_odds_l2347_234739

/-- Given odds in favor of an event as a ratio of two natural numbers -/
def OddsInFavor : Type := ℕ × ℕ

/-- Calculate the probability of an event given its odds in favor -/
def probability (odds : OddsInFavor) : ℚ :=
  let (favorable, unfavorable) := odds
  favorable / (favorable + unfavorable)

theorem probability_from_odds :
  let odds : OddsInFavor := (3, 5)
  probability odds = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_from_odds_l2347_234739


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2347_234735

theorem last_two_digits_product (n : ℤ) : 
  (∃ k : ℤ, n = 6 * k) →  -- n is divisible by 6
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n % 100 = 10 * a + b ∧ a + b = 12) →  -- sum of last two digits is 12
  (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n % 100 = 10 * x + y ∧ x * y = 32 ∨ x * y = 36) :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2347_234735


namespace NUMINAMATH_CALUDE_sin_2x_equiv_cos_2x_shifted_l2347_234771

theorem sin_2x_equiv_cos_2x_shifted (x : ℝ) : 
  Real.sin (2 * x) = Real.cos (2 * (x - Real.pi / 4)) := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_equiv_cos_2x_shifted_l2347_234771


namespace NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l2347_234744

theorem cubic_polynomial_uniqueness (p q r : ℝ) (Q : ℝ → ℝ) :
  (p^3 + 4*p^2 + 6*p + 8 = 0) →
  (q^3 + 4*q^2 + 6*q + 8 = 0) →
  (r^3 + 4*r^2 + 6*r + 8 = 0) →
  (∃ a b c d : ℝ, ∀ x, Q x = a*x^3 + b*x^2 + c*x + d) →
  (Q p = q + r) →
  (Q q = p + r) →
  (Q r = p + q) →
  (Q (p + q + r) = -20) →
  (∀ x, Q x = 5/4*x^3 + 4*x^2 + 23/4*x + 6) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l2347_234744


namespace NUMINAMATH_CALUDE_population_less_than_15_percent_in_fifth_year_l2347_234790

def population_decrease_rate : ℝ := 0.35
def target_population_ratio : ℝ := 0.15

def population_after_n_years (n : ℕ) : ℝ :=
  (1 - population_decrease_rate) ^ n

theorem population_less_than_15_percent_in_fifth_year :
  (∀ k < 5, population_after_n_years k > target_population_ratio) ∧
  population_after_n_years 5 < target_population_ratio :=
sorry

end NUMINAMATH_CALUDE_population_less_than_15_percent_in_fifth_year_l2347_234790


namespace NUMINAMATH_CALUDE_profit_percentage_l2347_234717

theorem profit_percentage (C P : ℝ) (h : (2/3) * P = 0.9 * C) :
  (P - C) / C = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l2347_234717


namespace NUMINAMATH_CALUDE_village_population_problem_l2347_234723

theorem village_population_problem (final_population : ℕ) : 
  final_population = 5265 → ∃ original : ℕ, 
    (original : ℚ) * (9/10) * (3/4) = final_population ∧ original = 7800 :=
by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l2347_234723


namespace NUMINAMATH_CALUDE_decimal_multiplication_l2347_234741

theorem decimal_multiplication : (0.2 : ℝ) * 0.8 = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l2347_234741


namespace NUMINAMATH_CALUDE_area_ratio_squares_l2347_234711

/-- Given squares A, B, and C with the following properties:
  - The perimeter of square A is 16 units
  - The perimeter of square B is 32 units
  - The side length of square C is 4 times the side length of square B
  Prove that the ratio of the area of square B to the area of square C is 1/16 -/
theorem area_ratio_squares (a b c : ℝ) 
  (ha : 4 * a = 16) 
  (hb : 4 * b = 32) 
  (hc : c = 4 * b) : 
  (b ^ 2) / (c ^ 2) = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_squares_l2347_234711


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l2347_234794

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  (⌊y⌋ - ⌈x⌉ : ℤ) ≤ 5 ∧ ∃ (x' y' : ℝ), 3 < x' ∧ x' < 6 ∧ 6 < y' ∧ y' < 10 ∧ ⌊y'⌋ - ⌈x'⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l2347_234794


namespace NUMINAMATH_CALUDE_mia_tv_watching_time_l2347_234720

def minutes_in_day : ℕ := 1440

def studying_minutes : ℕ := 288

theorem mia_tv_watching_time :
  ∃ (x : ℚ), 
    x > 0 ∧ 
    x < 1 ∧ 
    (1 / 4 : ℚ) * (1 - x) * minutes_in_day = studying_minutes ∧
    x = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_mia_tv_watching_time_l2347_234720


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l2347_234793

/-- The number of ways to divide 12 dogs into specified groups -/
def dog_grouping_count : ℕ := sorry

/-- Total number of dogs -/
def total_dogs : ℕ := 12

/-- Size of the first group (including Fluffy) -/
def group1_size : ℕ := 3

/-- Size of the second group (including Nipper) -/
def group2_size : ℕ := 5

/-- Size of the third group (including Spot) -/
def group3_size : ℕ := 4

/-- Theorem stating the correct number of ways to group the dogs -/
theorem dog_grouping_theorem : 
  dog_grouping_count = 20160 ∧
  total_dogs = group1_size + group2_size + group3_size ∧
  group1_size = 3 ∧
  group2_size = 5 ∧
  group3_size = 4 := by sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l2347_234793


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2347_234705

theorem consecutive_odd_numbers_sum (n₁ n₂ n₃ : ℕ) : 
  n₁ = 9 →
  n₂ = n₁ + 2 →
  n₃ = n₂ + 2 →
  Odd n₁ →
  Odd n₂ →
  Odd n₃ →
  11 * n₁ - (3 * n₃ + 4 * n₂) = 16 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2347_234705


namespace NUMINAMATH_CALUDE_shooting_stars_count_l2347_234728

theorem shooting_stars_count (bridget reginald sam emma max : ℕ) : 
  bridget = 14 →
  reginald = bridget - 2 →
  sam = reginald + 4 →
  emma = sam + 3 →
  max = bridget - 7 →
  sam - ((bridget + reginald + sam + emma + max) / 5 : ℚ) = 2.4 :=
by sorry

end NUMINAMATH_CALUDE_shooting_stars_count_l2347_234728


namespace NUMINAMATH_CALUDE_max_volume_triangular_prism_l2347_234702

/-- Represents a triangular prism with rectangular bases -/
structure TriangularPrism where
  l : ℝ  -- length of the base
  w : ℝ  -- width of the base
  h : ℝ  -- height of the prism

/-- The sum of the areas of two lateral faces and one base is 30 -/
def area_constraint (p : TriangularPrism) : Prop :=
  2 * p.h * p.l + p.l * p.w = 30

/-- The volume of the prism -/
def volume (p : TriangularPrism) : ℝ :=
  p.l * p.w * p.h

/-- Theorem: The maximum volume of the triangular prism is 112.5 -/
theorem max_volume_triangular_prism :
  ∃ (p : TriangularPrism), area_constraint p ∧
    (∀ (q : TriangularPrism), area_constraint q → volume q ≤ volume p) ∧
    volume p = 112.5 :=
sorry

end NUMINAMATH_CALUDE_max_volume_triangular_prism_l2347_234702


namespace NUMINAMATH_CALUDE_quadratic_real_root_and_inequality_l2347_234710

theorem quadratic_real_root_and_inequality (a b c : ℝ) :
  (∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0) ∧
  (a + b + c)^2 ≥ 3 * (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_and_inequality_l2347_234710


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2347_234706

def repeating_decimal_12 : ℚ := 4 / 33
def repeating_decimal_03 : ℚ := 1 / 33
def repeating_decimal_006 : ℚ := 2 / 333

theorem sum_of_repeating_decimals :
  repeating_decimal_12 + repeating_decimal_03 + repeating_decimal_006 = 19041 / 120879 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2347_234706


namespace NUMINAMATH_CALUDE_margaret_fraction_of_dollar_l2347_234785

-- Define the amounts for each person
def lance_cents : ℕ := 70
def guy_cents : ℕ := 50 + 10  -- Two quarters and a dime
def bill_cents : ℕ := 6 * 10  -- Six dimes
def total_cents : ℕ := 265

-- Define Margaret's amount
def margaret_cents : ℕ := total_cents - (lance_cents + guy_cents + bill_cents)

-- Theorem to prove
theorem margaret_fraction_of_dollar : 
  (margaret_cents : ℚ) / 100 = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_margaret_fraction_of_dollar_l2347_234785


namespace NUMINAMATH_CALUDE_solution_set_abs_b_greater_than_two_l2347_234714

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- Part 1: Solution set of the inequality
theorem solution_set (x : ℝ) : f x + f (x + 1) ≥ 5 ↔ x ≥ 4 ∨ x ≤ -1 := by
  sorry

-- Part 2: Proof that |b| > 2
theorem abs_b_greater_than_two (a b : ℝ) (h1 : |a| > 1) (h2 : f (a * b) > |a| * f (b / a)) : |b| > 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_abs_b_greater_than_two_l2347_234714


namespace NUMINAMATH_CALUDE_rational_cube_sum_zero_l2347_234788

theorem rational_cube_sum_zero (x y z : ℚ) 
  (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : 
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_sum_zero_l2347_234788


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2347_234780

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) :
  ∃ ζ : ℂ, ζ^3 = 1 ∧ ζ ≠ 1 ∧ (a^9 + b^9) / (a - b)^9 = 2 / (81 * (ζ - 1)) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2347_234780


namespace NUMINAMATH_CALUDE_remaining_area_ratio_l2347_234730

/-- The ratio of remaining areas of two squares after cutting out smaller squares -/
theorem remaining_area_ratio (side_c side_d cut_side : ℕ) 
  (hc : side_c = 48) 
  (hd : side_d = 60) 
  (hcut : cut_side = 12) : 
  (side_c^2 - cut_side^2) / (side_d^2 - cut_side^2) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_ratio_l2347_234730


namespace NUMINAMATH_CALUDE_tangency_triangle_area_l2347_234762

/-- A right triangle with legs 3 and 4 -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  is_right : leg1 = 3 ∧ leg2 = 4

/-- The incircle of a triangle -/
structure Incircle (t : RightTriangle) where
  center : ℝ × ℝ
  radius : ℝ

/-- Points of tangency of the incircle with the sides of the triangle -/
structure TangencyPoints (t : RightTriangle) (i : Incircle t) where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The area of a triangle -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of the triangle formed by the points of tangency is 6/5 -/
theorem tangency_triangle_area (t : RightTriangle) (i : Incircle t) (tp : TangencyPoints t i) :
  triangleArea tp.point1 tp.point2 tp.point3 = 6/5 := by sorry

end NUMINAMATH_CALUDE_tangency_triangle_area_l2347_234762


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l2347_234772

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 34

/-- The number of pencils Dan took from the drawer -/
def pencils_taken : ℕ := 22

/-- The number of pencils remaining in the drawer -/
def remaining_pencils : ℕ := initial_pencils - pencils_taken

theorem pencils_in_drawer : remaining_pencils = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l2347_234772


namespace NUMINAMATH_CALUDE_boat_length_boat_length_is_three_l2347_234745

/-- The length of a boat given its breadth, sinking depth, and the mass of a man. -/
theorem boat_length (breadth : ℝ) (sinking_depth : ℝ) (man_mass : ℝ) 
  (water_density : ℝ) (gravity : ℝ) : ℝ :=
  let volume := man_mass * gravity / (water_density * gravity)
  volume / (breadth * sinking_depth)

/-- Proof that the length of the boat is 3 meters given specific conditions. -/
theorem boat_length_is_three :
  boat_length 2 0.01 60 1000 9.81 = 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_length_boat_length_is_three_l2347_234745


namespace NUMINAMATH_CALUDE_infinite_triplets_existence_l2347_234782

theorem infinite_triplets_existence : ∀ n : ℕ, ∃ p : ℕ, ∃ q₁ q₂ : ℤ,
  0 < p ∧ p ≤ 2 * n^2 ∧ 
  |p * Real.sqrt 2 - q₁| * |p * Real.sqrt 3 - q₂| ≤ 1 / (4 * ↑n^2) :=
sorry

end NUMINAMATH_CALUDE_infinite_triplets_existence_l2347_234782


namespace NUMINAMATH_CALUDE_stevens_apple_peach_difference_prove_stevens_apple_peach_difference_l2347_234769

/-- Given that Jake has 3 fewer peaches and 4 more apples than Steven, and Steven has 19 apples,
    prove that the difference between Steven's apples and peaches is 19 - P,
    where P is the number of peaches Steven has. -/
theorem stevens_apple_peach_difference (P : ℕ) : ℕ → Prop :=
  let steven_apples : ℕ := 19
  let steven_peaches : ℕ := P
  let jake_peaches : ℕ := P - 3
  let jake_apples : ℕ := steven_apples + 4
  λ _ => steven_apples - steven_peaches = 19 - P

/-- Proof of the theorem -/
theorem prove_stevens_apple_peach_difference (P : ℕ) :
  stevens_apple_peach_difference P P :=
by
  sorry

end NUMINAMATH_CALUDE_stevens_apple_peach_difference_prove_stevens_apple_peach_difference_l2347_234769


namespace NUMINAMATH_CALUDE_farm_animal_difference_l2347_234768

/-- Proves that the difference between the number of goats and pigs is 33 -/
theorem farm_animal_difference : 
  let goats : ℕ := 66
  let chickens : ℕ := 2 * goats
  let ducks : ℕ := (goats + chickens) / 2
  let pigs : ℕ := ducks / 3
  goats - pigs = 33 := by sorry

end NUMINAMATH_CALUDE_farm_animal_difference_l2347_234768


namespace NUMINAMATH_CALUDE_power_product_equality_l2347_234729

theorem power_product_equality : 3^2 * 5 * 7^2 * 11 = 24255 := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l2347_234729


namespace NUMINAMATH_CALUDE_line_perpendicular_to_triangle_sides_l2347_234784

-- Define a triangle in a plane
structure Triangle :=
  (A B C : Point)

-- Define a line
structure Line :=
  (p q : Point)

-- Define perpendicularity between a line and a side of a triangle
def perpendicular (l : Line) (t : Triangle) (side : Fin 3) : Prop := sorry

theorem line_perpendicular_to_triangle_sides 
  (t : Triangle) (l : Line) :
  perpendicular l t 0 → perpendicular l t 1 → perpendicular l t 2 := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_triangle_sides_l2347_234784


namespace NUMINAMATH_CALUDE_sum_at_two_and_neg_two_l2347_234718

/-- A cubic polynomial with specific properties -/
structure CubicPolynomial (k : ℝ) where
  Q : ℝ → ℝ
  is_cubic : ∃ a b c : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + k
  at_zero : Q 0 = k
  at_one : Q 1 = 3 * k
  at_neg_one : Q (-1) = 4 * k

/-- The sum of the polynomial evaluated at 2 and -2 equals 22k -/
theorem sum_at_two_and_neg_two (k : ℝ) (p : CubicPolynomial k) :
  p.Q 2 + p.Q (-2) = 22 * k := by sorry

end NUMINAMATH_CALUDE_sum_at_two_and_neg_two_l2347_234718


namespace NUMINAMATH_CALUDE_quadratic_function_domain_range_conditions_l2347_234704

/-- Given a quadratic function f(x) = -1/2 * x^2 + x with domain [m, n] and range [k*m, k*n],
    prove that m = 2(1 - k) and n = 0 must be satisfied. -/
theorem quadratic_function_domain_range_conditions
  (f : ℝ → ℝ)
  (m n k : ℝ)
  (h_f : ∀ x, f x = -1/2 * x^2 + x)
  (h_domain : Set.Icc m n = {x | f x ∈ Set.Icc (k * m) (k * n)})
  (h_m_lt_n : m < n)
  (h_k_gt_1 : k > 1) :
  m = 2 * (1 - k) ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_domain_range_conditions_l2347_234704


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_l2347_234757

theorem quadratic_roots_distinct (m : ℝ) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_distinct_l2347_234757


namespace NUMINAMATH_CALUDE_cos_30_tan_45_equality_l2347_234758

theorem cos_30_tan_45_equality : 2 * Real.cos (30 * π / 180) - Real.tan (45 * π / 180) = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_30_tan_45_equality_l2347_234758


namespace NUMINAMATH_CALUDE_min_value_of_x_l2347_234754

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x / Real.log 3 ≥ Real.log 9 / Real.log 3 + (1/3) * (Real.log x / Real.log 3)) :
  x ≥ 27 ∧ ∀ y : ℝ, y > 0 → Real.log y / Real.log 3 ≥ Real.log 9 / Real.log 3 + (1/3) * (Real.log y / Real.log 3) → y ≥ x → y ≥ 27 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_l2347_234754


namespace NUMINAMATH_CALUDE_tank_capacity_after_adding_gas_l2347_234774

/-- Given a tank with a capacity of 54 gallons, initially filled to 3/4 of its capacity,
    prove that after adding 9 gallons of gasoline, the tank will be filled to 23/25 of its capacity. -/
theorem tank_capacity_after_adding_gas (tank_capacity : ℚ) (initial_fill : ℚ) (added_gas : ℚ) :
  tank_capacity = 54 →
  initial_fill = 3 / 4 →
  added_gas = 9 →
  (initial_fill * tank_capacity + added_gas) / tank_capacity = 23 / 25 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_after_adding_gas_l2347_234774


namespace NUMINAMATH_CALUDE_perpendicular_distance_extrema_l2347_234783

/-- Given two points on a line, prove that the sum of j values for (6, j) 
    that maximize and minimize squared perpendicular distances to the line is 13 -/
theorem perpendicular_distance_extrema (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁ = 2 ∧ y₁ = 9) (h₂ : x₂ = 14 ∧ y₂ = 20) : 
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  let y_line := m * 6 + b
  let j_max := ⌈y_line⌉ 
  let j_min := ⌊y_line⌋
  j_max + j_min = 13 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_distance_extrema_l2347_234783


namespace NUMINAMATH_CALUDE_emma_numbers_l2347_234738

theorem emma_numbers : ∃ (a b : ℤ), 
  ((a = 17 ∧ b = 31) ∨ (a = 31 ∧ b = 17)) ∧ 3 * a + 4 * b = 161 := by
  sorry

end NUMINAMATH_CALUDE_emma_numbers_l2347_234738


namespace NUMINAMATH_CALUDE_expand_algebraic_expression_l2347_234786

theorem expand_algebraic_expression (a b : ℝ) : 3*a*(5*a - 2*b) = 15*a^2 - 6*a*b := by
  sorry

end NUMINAMATH_CALUDE_expand_algebraic_expression_l2347_234786


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2347_234742

theorem sum_of_decimals : 5.47 + 2.58 + 1.95 = 10.00 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2347_234742


namespace NUMINAMATH_CALUDE_candy_box_problem_l2347_234700

theorem candy_box_problem (n : ℕ) : n ≤ 200 →
  (n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 1 ∧ n % 6 = 1) →
  n % 11 = 0 →
  n = 121 := by
sorry

end NUMINAMATH_CALUDE_candy_box_problem_l2347_234700


namespace NUMINAMATH_CALUDE_jazel_sticks_total_length_l2347_234799

/-- Given Jazel's three sticks with specified lengths, prove that their total length is 14 centimeters. -/
theorem jazel_sticks_total_length :
  let first_stick : ℕ := 3
  let second_stick : ℕ := 2 * first_stick
  let third_stick : ℕ := second_stick - 1
  first_stick + second_stick + third_stick = 14 := by
  sorry

end NUMINAMATH_CALUDE_jazel_sticks_total_length_l2347_234799


namespace NUMINAMATH_CALUDE_points_same_side_of_line_l2347_234759

theorem points_same_side_of_line (a : ℝ) : 
  (∃ (s : ℝ), s * (3 * 3 - 2 * 1 + a) > 0 ∧ s * (3 * (-4) - 2 * 6 + a) > 0) ↔ 
  (a < -7 ∨ a > 24) :=
sorry

end NUMINAMATH_CALUDE_points_same_side_of_line_l2347_234759


namespace NUMINAMATH_CALUDE_sin_negative_390_degrees_l2347_234734

theorem sin_negative_390_degrees : 
  Real.sin ((-390 : ℝ) * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_390_degrees_l2347_234734


namespace NUMINAMATH_CALUDE_at_least_three_correct_guesses_l2347_234707

-- Define the type for colors
inductive Color
| Red | Orange | Yellow | Green | Blue | Indigo | Violet

-- Define the type for dwarves
structure Dwarf where
  id : Fin 6
  seenHats : Finset Color

-- Define the game setup
structure GameSetup where
  allHats : Finset Color
  hiddenHat : Color
  dwarves : Fin 6 → Dwarf

-- Define the guessing strategy
def guessNearestClockwise (d : Dwarf) (allColors : Finset Color) : Color :=
  sorry

-- Theorem statement
theorem at_least_three_correct_guesses 
  (setup : GameSetup)
  (h1 : setup.allHats.card = 7)
  (h2 : ∀ d : Fin 6, (setup.dwarves d).seenHats.card = 5)
  (h3 : ∀ d : Fin 6, (setup.dwarves d).seenHats ⊆ setup.allHats)
  (h4 : setup.hiddenHat ∈ setup.allHats) :
  ∃ (correctGuesses : Finset (Fin 6)), 
    correctGuesses.card ≥ 3 ∧ 
    ∀ d ∈ correctGuesses, guessNearestClockwise (setup.dwarves d) setup.allHats = setup.hiddenHat :=
sorry

end NUMINAMATH_CALUDE_at_least_three_correct_guesses_l2347_234707


namespace NUMINAMATH_CALUDE_garden_perimeter_l2347_234725

theorem garden_perimeter : 
  ∀ (length breadth perimeter : ℝ),
  length = 260 →
  breadth = 190 →
  perimeter = 2 * (length + breadth) →
  perimeter = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l2347_234725


namespace NUMINAMATH_CALUDE_bagged_sugar_weight_recording_l2347_234787

/-- Represents the recording of a bag's weight difference from the standard -/
def weightDifference (standardWeight actual : ℕ) : ℤ :=
  (actual : ℤ) - (standardWeight : ℤ)

/-- Proves that a bag weighing 498 grams should be recorded as -3 grams when the standard is 501 grams -/
theorem bagged_sugar_weight_recording :
  let standardWeight : ℕ := 501
  let actualWeight : ℕ := 498
  weightDifference standardWeight actualWeight = -3 := by
sorry

end NUMINAMATH_CALUDE_bagged_sugar_weight_recording_l2347_234787


namespace NUMINAMATH_CALUDE_soda_cans_calculation_correct_l2347_234736

/-- Given that S cans of soda can be purchased for Q dimes, and 1 dollar is worth 10 dimes,
    this function calculates the number of cans that can be purchased for D dollars. -/
def soda_cans_for_dollars (S Q D : ℚ) : ℚ :=
  10 * D * S / Q

/-- Theorem stating that the number of cans that can be purchased for D dollars
    is correctly calculated by the soda_cans_for_dollars function. -/
theorem soda_cans_calculation_correct (S Q D : ℚ) (hS : S > 0) (hQ : Q > 0) (hD : D ≥ 0) :
  soda_cans_for_dollars S Q D = 10 * D * S / Q :=
by sorry

end NUMINAMATH_CALUDE_soda_cans_calculation_correct_l2347_234736


namespace NUMINAMATH_CALUDE_function_always_one_l2347_234781

theorem function_always_one (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, n > 0 → f (n + f n) = f n)
  (h2 : ∃ n₀ : ℕ, f n₀ = 1) : 
  ∀ n : ℕ, f n = 1 := by
sorry

end NUMINAMATH_CALUDE_function_always_one_l2347_234781


namespace NUMINAMATH_CALUDE_sqrt_calculations_l2347_234752

theorem sqrt_calculations : 
  (2 * Real.sqrt 12 + Real.sqrt 75 - 12 * Real.sqrt (1/3) = 5 * Real.sqrt 3) ∧
  (6 * Real.sqrt (8/5) / (2 * Real.sqrt 2) * (-1/2 * Real.sqrt 60) = -6 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l2347_234752


namespace NUMINAMATH_CALUDE_number_divisibility_l2347_234731

theorem number_divisibility (x : ℝ) : x / 14.5 = 171 → x = 2479.5 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_l2347_234731


namespace NUMINAMATH_CALUDE_michaels_brother_initial_money_l2347_234708

/-- Proof that Michael's brother initially had $17 -/
theorem michaels_brother_initial_money :
  ∀ (michael_money : ℕ) (brother_money_after : ℕ) (candy_cost : ℕ),
    michael_money = 42 →
    brother_money_after = 35 →
    candy_cost = 3 →
    ∃ (brother_initial_money : ℕ),
      brother_initial_money = 17 ∧
      brother_money_after = brother_initial_money + michael_money / 2 - candy_cost :=
by
  sorry

#check michaels_brother_initial_money

end NUMINAMATH_CALUDE_michaels_brother_initial_money_l2347_234708


namespace NUMINAMATH_CALUDE_ellipse_properties_l2347_234726

open Real

theorem ellipse_properties (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  let e := (Real.sqrt 6) / 3
  let d := (Real.sqrt 3) / 2
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let line := fun (k m x : ℝ) => k * x + m
  let A := (0, -b)
  let B := (a, 0)
  let distance_to_AB := d

  (e^2 * a^2 = a^2 - b^2) →
  (distance_to_AB^2 * (a^2 + b^2) = a^2 * b^2) →
  (∃ (C D : ℝ × ℝ) (k m : ℝ), k ≠ 0 ∧ m ≠ 0 ∧
    ellipse C.1 C.2 ∧ ellipse D.1 D.2 ∧
    C.2 = line k m C.1 ∧ D.2 = line k m D.1 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2) →
  (a^2 = 3 ∧ b^2 = 1 ∧
   (let k := (Real.sqrt 6) / 3
    let m := 3 / 2
    let area_ACD := 5 / 4
    ∃ (C D : ℝ × ℝ),
      ellipse C.1 C.2 ∧ ellipse D.1 D.2 ∧
      C.2 = line k m C.1 ∧ D.2 = line k m D.1 ∧
      (C.1 - A.1)^2 + (C.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
      area_ACD = 1/2 * abs ((C.1 - A.1) * (D.2 - A.2) - (C.2 - A.2) * (D.1 - A.1))))
  := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2347_234726


namespace NUMINAMATH_CALUDE_serve_meals_eq_945_l2347_234763

/-- The number of ways to serve meals to 10 people with exactly 2 correct matches -/
def serve_meals : ℕ :=
  let total_people : ℕ := 10
  let pasta_orders : ℕ := 5
  let salad_orders : ℕ := 5
  let correct_matches : ℕ := 2
  -- The actual calculation is not implemented, just the problem statement
  945

/-- Theorem stating that serve_meals equals 945 -/
theorem serve_meals_eq_945 : serve_meals = 945 := by
  sorry

end NUMINAMATH_CALUDE_serve_meals_eq_945_l2347_234763


namespace NUMINAMATH_CALUDE_smallest_possible_campers_l2347_234766

/-- Represents the number of campers participating in different combinations of activities -/
structure CampActivities where
  only_canoeing : ℕ
  canoeing_swimming : ℕ
  only_swimming : ℕ
  canoeing_fishing : ℕ
  swimming_fishing : ℕ
  only_fishing : ℕ

/-- Represents the camp with its activities and camper counts -/
structure Camp where
  activities : CampActivities
  no_activity : ℕ

/-- Calculates the total number of campers in the camp -/
def total_campers (camp : Camp) : ℕ :=
  camp.activities.only_canoeing +
  camp.activities.canoeing_swimming +
  camp.activities.only_swimming +
  camp.activities.canoeing_fishing +
  camp.activities.swimming_fishing +
  camp.activities.only_fishing +
  camp.no_activity

/-- Checks if the camp satisfies the given conditions -/
def satisfies_conditions (camp : Camp) : Prop :=
  (camp.activities.only_canoeing + camp.activities.canoeing_swimming + camp.activities.canoeing_fishing = 15) ∧
  (camp.activities.canoeing_swimming + camp.activities.only_swimming + camp.activities.swimming_fishing = 22) ∧
  (camp.activities.canoeing_fishing + camp.activities.swimming_fishing + camp.activities.only_fishing = 12) ∧
  (camp.no_activity = 9)

theorem smallest_possible_campers :
  ∀ camp : Camp,
    satisfies_conditions camp →
    total_campers camp ≥ 34 :=
by sorry

#check smallest_possible_campers

end NUMINAMATH_CALUDE_smallest_possible_campers_l2347_234766


namespace NUMINAMATH_CALUDE_min_value_f_f_decreasing_sum_lower_bound_l2347_234795

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x^2 + x

-- Part 1
theorem min_value_f (x : ℝ) (h : x > 0) : 
  f (-1) 0 x ≥ 1 :=
sorry

-- Part 2
def f_special (x : ℝ) : ℝ := Real.log x - x^2 + x

theorem f_decreasing (x : ℝ) (h : x > 1) :
  ∀ y > x, f_special y < f_special x :=
sorry

-- Part 3
theorem sum_lower_bound (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0)
  (h : f 1 1 x₁ + f 1 1 x₂ + x₁ * x₂ = 0) :
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_f_decreasing_sum_lower_bound_l2347_234795


namespace NUMINAMATH_CALUDE_crackers_distribution_l2347_234733

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) : 
  total_crackers = 36 →
  num_friends = 18 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 2 := by
  sorry

end NUMINAMATH_CALUDE_crackers_distribution_l2347_234733


namespace NUMINAMATH_CALUDE_set_relationships_l2347_234760

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- Define set B
def B : Set (ℝ × ℝ) := {p | p.2 = p.1^2 + 1}

-- Theorem statement
theorem set_relationships :
  (1 ∉ B) ∧ (2 ∈ A) := by
  sorry

end NUMINAMATH_CALUDE_set_relationships_l2347_234760


namespace NUMINAMATH_CALUDE_lesser_number_problem_l2347_234748

theorem lesser_number_problem (x y : ℝ) (h_sum : x + y = 70) (h_product : x * y = 1050) : 
  min x y = 30 := by
sorry

end NUMINAMATH_CALUDE_lesser_number_problem_l2347_234748


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2347_234740

/-- The slope of a line perpendicular to 4x - 5y = 10 is -5/4 -/
theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 5 * y = 10) → 
  (slope_of_perpendicular_line : ℝ) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2347_234740


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2347_234727

theorem opposite_of_2023 : ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2347_234727


namespace NUMINAMATH_CALUDE_power_product_cube_l2347_234776

theorem power_product_cube (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_cube_l2347_234776


namespace NUMINAMATH_CALUDE_factorial_ratio_l2347_234712

theorem factorial_ratio : (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2347_234712


namespace NUMINAMATH_CALUDE_rhombus_area_l2347_234777

/-- The area of a rhombus with diagonals of length 3 and 4 is 6 -/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 3) (h2 : d2 = 4) : 
  (1 / 2 : ℝ) * d1 * d2 = 6 := by
  sorry

#check rhombus_area

end NUMINAMATH_CALUDE_rhombus_area_l2347_234777


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l2347_234703

-- Define a trapezoid PQRS
structure Trapezoid :=
  (P Q R S : ℝ × ℝ)

-- Define the length function
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

theorem trapezoid_segment_length (PQRS : Trapezoid) :
  length PQRS.P PQRS.S + length PQRS.R PQRS.Q = 270 →
  area_triangle PQRS.P PQRS.Q PQRS.R / area_triangle PQRS.P PQRS.S PQRS.R = 5 / 4 →
  length PQRS.P PQRS.S = 150 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l2347_234703


namespace NUMINAMATH_CALUDE_meet_time_theorem_l2347_234724

def round_time_P : ℕ := 252
def round_time_Q : ℕ := 198
def round_time_R : ℕ := 315

theorem meet_time_theorem : 
  Nat.lcm (Nat.lcm round_time_P round_time_Q) round_time_R = 13860 := by
  sorry

end NUMINAMATH_CALUDE_meet_time_theorem_l2347_234724


namespace NUMINAMATH_CALUDE_union_of_sets_l2347_234749

theorem union_of_sets : 
  let A : Set ℕ := {0, 1, 3}
  let B : Set ℕ := {1, 2, 4}
  A ∪ B = {0, 1, 2, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2347_234749


namespace NUMINAMATH_CALUDE_colonization_theorem_l2347_234756

def blue_planets : ℕ := 7
def orange_planets : ℕ := 8
def blue_cost : ℕ := 3
def orange_cost : ℕ := 2
def total_units : ℕ := 21

def colonization_ways (b o bc oc t : ℕ) : ℕ :=
  (Nat.choose b 7 * Nat.choose o 0) +
  (Nat.choose b 5 * Nat.choose o 3) +
  (Nat.choose b 3 * Nat.choose o 6)

theorem colonization_theorem :
  colonization_ways blue_planets orange_planets blue_cost orange_cost total_units = 2157 := by
  sorry

end NUMINAMATH_CALUDE_colonization_theorem_l2347_234756


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l2347_234761

theorem pirate_treasure_probability : 
  let n : ℕ := 8  -- Total number of islands
  let k : ℕ := 4  -- Number of islands with treasure
  let p_treasure : ℚ := 1/3  -- Probability of treasure and no traps
  let p_neither : ℚ := 1/2  -- Probability of neither treasure nor traps
  Nat.choose n k * p_treasure^k * p_neither^(n-k) = 35/648 := by
sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l2347_234761


namespace NUMINAMATH_CALUDE_lcm_of_five_numbers_l2347_234722

theorem lcm_of_five_numbers : Nat.lcm 53 (Nat.lcm 71 (Nat.lcm 89 (Nat.lcm 103 200))) = 788045800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_five_numbers_l2347_234722


namespace NUMINAMATH_CALUDE_intersection_circle_radius_squared_l2347_234779

/-- The parabolas y = (x - 2)^2 and x + 6 = (y + 1)^2 intersect at four points. 
    All four points lie on a circle. This theorem proves that the radius squared 
    of this circle is 1/4. -/
theorem intersection_circle_radius_squared (x y : ℝ) : 
  (y = (x - 2)^2 ∧ x + 6 = (y + 1)^2) → 
  ((x - 3/2)^2 + (y + 3/2)^2 = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_intersection_circle_radius_squared_l2347_234779


namespace NUMINAMATH_CALUDE_incorrect_inequality_l2347_234751

theorem incorrect_inequality (a b : ℝ) (h : a < b) : ¬(-4*a < -4*b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l2347_234751


namespace NUMINAMATH_CALUDE_exam_marks_percentage_l2347_234755

theorem exam_marks_percentage (full_marks A_marks B_marks C_marks D_marks : ℝ) : 
  full_marks = 500 →
  A_marks = B_marks * 0.9 →
  B_marks = C_marks * 1.25 →
  C_marks = D_marks * 0.8 →
  A_marks = 360 →
  D_marks / full_marks = 0.8 :=
by sorry

end NUMINAMATH_CALUDE_exam_marks_percentage_l2347_234755


namespace NUMINAMATH_CALUDE_percent_of_a_is_4b_l2347_234796

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) : (4 * b) / a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_4b_l2347_234796


namespace NUMINAMATH_CALUDE_hoseok_addition_l2347_234713

theorem hoseok_addition (x : ℤ) : x + 56 = 110 → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_addition_l2347_234713


namespace NUMINAMATH_CALUDE_three_heads_before_four_tails_l2347_234798

/-- The probability of encountering 3 heads before 4 tails in repeated fair coin flips -/
def probability_three_heads_before_four_tails : ℚ := 4/7

/-- A fair coin has equal probability of heads and tails -/
axiom fair_coin : ℚ

/-- The probability of heads for a fair coin is 1/2 -/
axiom fair_coin_probability : fair_coin = 1/2

theorem three_heads_before_four_tails :
  probability_three_heads_before_four_tails = 4/7 :=
by sorry

end NUMINAMATH_CALUDE_three_heads_before_four_tails_l2347_234798


namespace NUMINAMATH_CALUDE_range_of_m_l2347_234701

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) : 
  (f = λ x => 1 + Real.sin (2 * x)) →
  (g = λ x => 2 * (Real.cos x)^2 + m) →
  (∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), f x₀ ≥ g x₀) →
  m ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2347_234701


namespace NUMINAMATH_CALUDE_jills_race_time_l2347_234764

/-- Proves that Jill's race time is 32 seconds -/
theorem jills_race_time (jack_first_half : ℕ) (jack_second_half : ℕ) (time_difference : ℕ) :
  jack_first_half = 19 →
  jack_second_half = 6 →
  time_difference = 7 →
  jack_first_half + jack_second_half + time_difference = 32 :=
by sorry

end NUMINAMATH_CALUDE_jills_race_time_l2347_234764


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2347_234715

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  sample : Finset ℕ
  h_sample_size : sample.card = sample_size
  h_valid_sample : ∀ n ∈ sample, n ≤ population_size

/-- Checks if a given set of numbers forms an arithmetic sequence -/
def is_arithmetic_sequence (s : Finset ℕ) : Prop :=
  ∃ a d : ℤ, ∀ n ∈ s, ∃ k : ℕ, (n : ℤ) = a + k * d

theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop : s.population_size = 52)
  (h_sample : s.sample_size = 4)
  (h_elements : {5, 31, 44} ⊆ s.sample)
  (h_arithmetic : is_arithmetic_sequence s.sample) :
  18 ∈ s.sample := by
  sorry


end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2347_234715


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2347_234709

theorem absolute_value_inequality (x : ℝ) : 
  abs x + abs (2 * x - 3) ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2347_234709


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l2347_234716

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 3) :
  a / c = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l2347_234716


namespace NUMINAMATH_CALUDE_common_divisors_84_90_l2347_234750

theorem common_divisors_84_90 : 
  (Finset.filter (λ x => x ∣ 84 ∧ x ∣ 90) (Finset.range (min 84 90 + 1))).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_84_90_l2347_234750
