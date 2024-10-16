import Mathlib

namespace NUMINAMATH_CALUDE_positive_integer_value_l1452_145289

def first_seven_multiples_of_four : List Nat := [4, 8, 12, 16, 20, 24, 28]

def a : ℚ := (first_seven_multiples_of_four.sum : ℚ) / 7

def b (n : ℕ) : ℚ := 2 * n

theorem positive_integer_value (n : ℕ) (h : n > 0) :
  a^2 - (b n)^2 = 0 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_value_l1452_145289


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1452_145293

/-- A geometric sequence with specified properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  h1 : a 5 - a 3 = 12
  h2 : a 6 - a 4 = 24

/-- Sum of the first n terms of a geometric sequence -/
def S (seq : GeometricSequence) (n : ℕ) : ℝ := sorry

/-- Theorem stating the ratio of S_n to a_n -/
theorem geometric_sequence_ratio (seq : GeometricSequence) (n : ℕ) :
  S seq n / seq.a n = 2 - 2^(1 - n) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1452_145293


namespace NUMINAMATH_CALUDE_sleeping_bag_wholesale_cost_l1452_145206

theorem sleeping_bag_wholesale_cost :
  ∀ (wholesale_cost selling_price : ℝ),
    selling_price = wholesale_cost * 1.12 →
    selling_price = 28 →
    wholesale_cost = 25 := by sorry

end NUMINAMATH_CALUDE_sleeping_bag_wholesale_cost_l1452_145206


namespace NUMINAMATH_CALUDE_correct_total_carrots_l1452_145225

/-- The total number of carrots Bianca has after picking, throwing out, and picking again -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

/-- Theorem stating that the total number of carrots is correct -/
theorem correct_total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ)
  (h1 : initial ≥ thrown_out) :
  total_carrots initial thrown_out picked_next_day = initial - thrown_out + picked_next_day :=
by
  sorry

#eval total_carrots 23 10 47  -- Should evaluate to 60

end NUMINAMATH_CALUDE_correct_total_carrots_l1452_145225


namespace NUMINAMATH_CALUDE_jerrys_coins_l1452_145299

theorem jerrys_coins (n d : ℕ) : 
  n + d = 30 →
  5 * n + 10 * d + 140 = 10 * n + 5 * d →
  5 * n + 10 * d = 155 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_coins_l1452_145299


namespace NUMINAMATH_CALUDE_variance_transformation_l1452_145268

variable {n : ℕ}
variable (x : Fin n → ℝ)
variable (a b : ℝ)

def variance (y : Fin n → ℝ) : ℝ := sorry

theorem variance_transformation (h1 : variance x = 3) 
  (h2 : variance (fun i => a * x i + b) = 12) : a = 2 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_variance_transformation_l1452_145268


namespace NUMINAMATH_CALUDE_opposite_unit_vector_l1452_145219

def a : ℝ × ℝ := (12, 5)

theorem opposite_unit_vector :
  let magnitude := Real.sqrt (a.1^2 + a.2^2)
  (-a.1 / magnitude, -a.2 / magnitude) = (-12/13, -5/13) := by
  sorry

end NUMINAMATH_CALUDE_opposite_unit_vector_l1452_145219


namespace NUMINAMATH_CALUDE_smallest_number_range_l1452_145292

theorem smallest_number_range (a b c d e : ℝ) 
  (distinct : a < b ∧ b < c ∧ c < d ∧ d < e)
  (sum1 : a + b = 20)
  (sum2 : a + c = 200)
  (sum3 : d + e = 2014)
  (sum4 : c + e = 2000) :
  -793 < a ∧ a < 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_range_l1452_145292


namespace NUMINAMATH_CALUDE_xy_value_l1452_145217

theorem xy_value (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 5) : 2 * x * y / 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1452_145217


namespace NUMINAMATH_CALUDE_running_distance_l1452_145298

theorem running_distance (jonathan_distance : ℝ) 
  (h1 : jonathan_distance = 7.5)
  (mercedes_distance : ℝ) 
  (h2 : mercedes_distance = 2 * jonathan_distance)
  (davonte_distance : ℝ) 
  (h3 : davonte_distance = mercedes_distance + 2) :
  mercedes_distance + davonte_distance = 32 := by
sorry

end NUMINAMATH_CALUDE_running_distance_l1452_145298


namespace NUMINAMATH_CALUDE_island_puzzle_l1452_145242

/-- Represents the nature of a person on the island -/
inductive PersonNature
| Knight
| Liar

/-- Represents the statement made by person A -/
def statement (nature : PersonNature) (treasures : Prop) : Prop :=
  (nature = PersonNature.Knight) ↔ treasures

/-- The main theorem about A's statement and the existence of treasures -/
theorem island_puzzle :
  ∀ (A_nature : PersonNature) (treasures : Prop),
    statement A_nature treasures →
    (¬ (A_nature = PersonNature.Knight ∨ A_nature = PersonNature.Liar) ∧ treasures) :=
by sorry

end NUMINAMATH_CALUDE_island_puzzle_l1452_145242


namespace NUMINAMATH_CALUDE_hyperbola_equation_part1_hyperbola_equation_part2_l1452_145209

-- Part 1
theorem hyperbola_equation_part1 (c : ℝ) (h1 : c = Real.sqrt 6) :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (x y : ℝ), (x^2 / a^2 - y^2 / (6 - a^2) = 1) ↔ 
  (x^2 / 5 - y^2 = 1)) ∧
  ((-5)^2 / a^2 - 2^2 / (6 - a^2) = 1) := by
sorry

-- Part 2
theorem hyperbola_equation_part2 (x1 y1 x2 y2 : ℝ) 
  (h1 : x1 = 3 ∧ y1 = -4 * Real.sqrt 2)
  (h2 : x2 = 9/4 ∧ y2 = 5) :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧
  (∀ (x y : ℝ), (m * x^2 - n * y^2 = 1) ↔ 
  (y^2 / 16 - x^2 / 9 = 1)) ∧
  (m * x1^2 - n * y1^2 = 1) ∧
  (m * x2^2 - n * y2^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_part1_hyperbola_equation_part2_l1452_145209


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_cubes_l1452_145232

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed (l w h : ℕ) : ℕ :=
  l + w + h
  - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l)
  + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating the number of cubes passed through by the internal diagonal -/
theorem rectangular_solid_diagonal_cubes :
  cubes_passed 150 324 375 = 768 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_cubes_l1452_145232


namespace NUMINAMATH_CALUDE_infinite_decimal_sqrt_l1452_145280

theorem infinite_decimal_sqrt (x y : ℕ) : 
  x ∈ Finset.range 9 → y ∈ Finset.range 9 → 
  (Real.sqrt (x / 9 : ℝ) = y / 9) ↔ ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) := by
sorry

end NUMINAMATH_CALUDE_infinite_decimal_sqrt_l1452_145280


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_negation_l1452_145271

theorem necessary_not_sufficient_negation (p q : Prop) 
  (h1 : q → p)  -- p is necessary for q
  (h2 : ¬(p → q))  -- p is not sufficient for q
  : (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_negation_l1452_145271


namespace NUMINAMATH_CALUDE_largest_fraction_l1452_145279

theorem largest_fraction :
  let a := (8 + 5) / 3
  let b := 8 / (3 + 5)
  let c := (3 + 5) / 8
  let d := (8 + 3) / 5
  let e := 3 / (8 + 5)
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l1452_145279


namespace NUMINAMATH_CALUDE_thyme_leaves_theorem_l1452_145211

/-- The number of leaves per thyme plant -/
def thyme_leaves_per_plant : ℕ :=
  let basil_pots : ℕ := 3
  let rosemary_pots : ℕ := 9
  let thyme_pots : ℕ := 6
  let basil_leaves_per_pot : ℕ := 4
  let rosemary_leaves_per_pot : ℕ := 18
  let total_leaves : ℕ := 354
  let basil_leaves : ℕ := basil_pots * basil_leaves_per_pot
  let rosemary_leaves : ℕ := rosemary_pots * rosemary_leaves_per_pot
  let thyme_leaves : ℕ := total_leaves - basil_leaves - rosemary_leaves
  thyme_leaves / thyme_pots

theorem thyme_leaves_theorem : thyme_leaves_per_plant = 30 := by
  sorry

end NUMINAMATH_CALUDE_thyme_leaves_theorem_l1452_145211


namespace NUMINAMATH_CALUDE_ages_sum_l1452_145278

theorem ages_sum (a b c : ℕ) 
  (h1 : a = 20 + b + c) 
  (h2 : a^2 = 2000 + (b + c)^2) : 
  a + b + c = 80 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l1452_145278


namespace NUMINAMATH_CALUDE_rachel_apple_trees_l1452_145216

/-- The total number of apples remaining on Rachel's trees -/
def total_apples_remaining (X : ℕ) : ℕ :=
  let first_four_trees := 10 + 40 + 15 + 22
  let remaining_trees := 48 * X
  first_four_trees + remaining_trees

/-- Theorem stating the total number of apples remaining on Rachel's trees -/
theorem rachel_apple_trees (X : ℕ) :
  total_apples_remaining X = 87 + 48 * X := by
  sorry

end NUMINAMATH_CALUDE_rachel_apple_trees_l1452_145216


namespace NUMINAMATH_CALUDE_jeremy_age_l1452_145295

/-- Given the ages of Amy, Jeremy, and Chris, prove Jeremy's age --/
theorem jeremy_age (amy jeremy chris : ℕ) 
  (h1 : amy + jeremy + chris = 132)  -- Combined age
  (h2 : amy = jeremy / 3)            -- Amy's age relation to Jeremy
  (h3 : chris = 2 * amy)             -- Chris's age relation to Amy
  : jeremy = 66 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_age_l1452_145295


namespace NUMINAMATH_CALUDE_marley_fruit_count_l1452_145250

/-- Represents the number of fruits a person has -/
structure FruitCount where
  oranges : ℕ
  apples : ℕ

/-- Calculates the total number of fruits -/
def totalFruits (fc : FruitCount) : ℕ :=
  fc.oranges + fc.apples

/-- The problem statement -/
theorem marley_fruit_count :
  let louis : FruitCount := ⟨5, 3⟩
  let samantha : FruitCount := ⟨8, 7⟩
  let marley : FruitCount := ⟨2 * louis.oranges, 3 * samantha.apples⟩
  totalFruits marley = 31 := by
  sorry


end NUMINAMATH_CALUDE_marley_fruit_count_l1452_145250


namespace NUMINAMATH_CALUDE_right_triangle_area_l1452_145213

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 10) (h3 : a = 6) :
  (1/2) * a * b = 24 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1452_145213


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l1452_145262

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_large_number : sumOfDigits (2^2010 * 5^2008 * 7) = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l1452_145262


namespace NUMINAMATH_CALUDE_pairs_with_female_l1452_145263

theorem pairs_with_female (total : Nat) (males : Nat) (females : Nat) : 
  total = males + females → males = 3 → females = 3 → 
  (Nat.choose total 2) - (Nat.choose males 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_pairs_with_female_l1452_145263


namespace NUMINAMATH_CALUDE_right_angled_triangle_l1452_145235

theorem right_angled_triangle (h₁ h₂ h₃ : ℝ) (h_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0)
  (h_altitudes : h₁ = 12 ∧ h₂ = 15 ∧ h₃ = 20) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * h₁ = 2 * (b * c / 2)) ∧
    (b * h₂ = 2 * (a * c / 2)) ∧
    (c * h₃ = 2 * (a * b / 2)) ∧
    a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l1452_145235


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1452_145254

/-- Given a hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0,
    if one of its asymptotic lines is y = 2x, then b = 2 -/
theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  (∃ (x y : ℝ), x^2 - y^2/b^2 = 1 ∧ y = 2*x) → b = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1452_145254


namespace NUMINAMATH_CALUDE_range_of_m_with_two_integer_solutions_l1452_145274

theorem range_of_m_with_two_integer_solutions (m : ℝ) : 
  (∃ (x y : ℤ), x ≠ y ∧ 
    (∀ z : ℤ, (-1 : ℝ) ≤ z ∧ (z : ℝ) < m ↔ z = x ∨ z = y)) →
  0 < m ∧ m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_with_two_integer_solutions_l1452_145274


namespace NUMINAMATH_CALUDE_sin_18_cos_36_eq_quarter_l1452_145230

theorem sin_18_cos_36_eq_quarter : Real.sin (18 * π / 180) * Real.cos (36 * π / 180) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_cos_36_eq_quarter_l1452_145230


namespace NUMINAMATH_CALUDE_lcm_12_18_25_l1452_145266

theorem lcm_12_18_25 : Nat.lcm (Nat.lcm 12 18) 25 = 900 := by sorry

end NUMINAMATH_CALUDE_lcm_12_18_25_l1452_145266


namespace NUMINAMATH_CALUDE_mirror_area_l1452_145224

/-- Given a rectangular frame with outer dimensions 100 cm by 140 cm and a uniform frame width of 12 cm,
    the area of the rectangular mirror that fits exactly inside the frame is 8816 cm². -/
theorem mirror_area (frame_width frame_height frame_thickness : ℕ) 
  (hw : frame_width = 100)
  (hh : frame_height = 140)
  (ht : frame_thickness = 12) :
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 8816 :=
by sorry

end NUMINAMATH_CALUDE_mirror_area_l1452_145224


namespace NUMINAMATH_CALUDE_max_rectangle_division_ratio_l1452_145229

/-- The number of ways to divide a rectangle with side lengths a and b into smaller rectangles with integer side lengths -/
def D (a b : ℕ+) : ℕ := sorry

/-- The theorem stating that D(a,b)/(2(a+b)) ≤ 3/8 for all positive integers a and b, 
    with equality if and only if a = b = 2 -/
theorem max_rectangle_division_ratio 
  (a b : ℕ+) : 
  (D a b : ℚ) / (2 * ((a:ℚ) + (b:ℚ))) ≤ 3/8 ∧ 
  ((D a b : ℚ) / (2 * ((a:ℚ) + (b:ℚ))) = 3/8 ↔ a = 2 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_division_ratio_l1452_145229


namespace NUMINAMATH_CALUDE_chocolate_theorem_l1452_145215

def chocolate_problem (total : ℕ) (typeA typeB typeC : ℕ) : Prop :=
  let typeD := 2 * typeA
  let typeE := 2 * typeB
  let typeF := typeA + 6
  let typeG := typeB + 6
  let typeH := typeC + 6
  let non_peanut := typeA + typeB + typeC + typeD + typeE + typeF + typeG + typeH
  let peanut := total - non_peanut
  (peanut : ℚ) / total = 3 / 10

theorem chocolate_theorem :
  chocolate_problem 100 5 6 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l1452_145215


namespace NUMINAMATH_CALUDE_all_propositions_false_l1452_145214

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relationships between lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem all_propositions_false :
  ∀ (m n : Line) (α : Plane),
    m ≠ n →
    (¬ (parallel_line_plane m α ∧ parallel_line_plane n α → parallel_lines m n)) ∧
    (¬ (parallel_lines m n ∧ line_in_plane n α → parallel_line_plane m α)) ∧
    (¬ (perpendicular_line_plane m α ∧ perpendicular_lines m n → parallel_line_plane n α)) ∧
    (¬ (parallel_line_plane m α ∧ perpendicular_lines m n → perpendicular_line_plane n α)) :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l1452_145214


namespace NUMINAMATH_CALUDE_seven_sided_die_perfect_square_probability_l1452_145236

/-- Represents a fair seven-sided die with faces numbered 1 through 7 -/
def SevenSidedDie : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- The number of times the die is rolled -/
def numRolls : ℕ := 4

/-- The total number of possible outcomes when rolling the die numRolls times -/
def totalOutcomes : ℕ := SevenSidedDie.card ^ numRolls

/-- The number of favorable outcomes (product of rolls is a perfect square) -/
def favorableOutcomes : ℕ := 164

theorem seven_sided_die_perfect_square_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 164 / 2401 :=
sorry

end NUMINAMATH_CALUDE_seven_sided_die_perfect_square_probability_l1452_145236


namespace NUMINAMATH_CALUDE_product_of_fractions_l1452_145218

theorem product_of_fractions : 
  (7 / 5 : ℚ) * (8 / 16 : ℚ) * (21 / 15 : ℚ) * (14 / 28 : ℚ) * 
  (35 / 25 : ℚ) * (20 / 40 : ℚ) * (49 / 35 : ℚ) * (32 / 64 : ℚ) = 2401 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1452_145218


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l1452_145286

theorem polygon_interior_exterior_angle_relation :
  ∀ n : ℕ, 
  n > 2 →
  (n - 2) * 180 = 2 * 360 →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l1452_145286


namespace NUMINAMATH_CALUDE_not_divisible_power_ten_plus_one_l1452_145226

theorem not_divisible_power_ten_plus_one (m n : ℕ) :
  ¬ ∃ (k : ℕ), (10^m + 1) = k * (10^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_power_ten_plus_one_l1452_145226


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1452_145253

def is_valid_roll (m n : ℕ) : Prop := 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6

def angle_greater_than_90 (m n : ℕ) : Prop := m > n

def count_favorable_outcomes : ℕ := 15

def total_outcomes : ℕ := 36

theorem dice_roll_probability : 
  (count_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 :=
sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1452_145253


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l1452_145245

theorem real_roots_of_polynomial (x : ℝ) :
  x^4 + 2*x^3 - x - 2 = 0 ↔ x = 1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l1452_145245


namespace NUMINAMATH_CALUDE_confetti_area_sum_l1452_145264

/-- The sum of the areas of two square-shaped pieces of confetti, one with side length 11 cm and the other with side length 5 cm, is equal to 146 cm². -/
theorem confetti_area_sum : 
  let red_side : ℝ := 11
  let blue_side : ℝ := 5
  red_side ^ 2 + blue_side ^ 2 = 146 :=
by sorry

end NUMINAMATH_CALUDE_confetti_area_sum_l1452_145264


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l1452_145238

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The theorem states that if (a+i)(2+i) is a pure imaginary number, then a = 1/2 -/
theorem pure_imaginary_product (a : ℝ) : 
  is_pure_imaginary ((a + Complex.I) * (2 + Complex.I)) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l1452_145238


namespace NUMINAMATH_CALUDE_keaton_annual_earnings_l1452_145290

/-- Represents Keaton's farm and calculates annual earnings --/
def farm_earnings : ℕ :=
  let months_per_year : ℕ := 12
  let orange_harvest_interval : ℕ := 2
  let apple_harvest_interval : ℕ := 3
  let orange_harvest_value : ℕ := 50
  let apple_harvest_value : ℕ := 30
  let orange_harvests_per_year : ℕ := months_per_year / orange_harvest_interval
  let apple_harvests_per_year : ℕ := months_per_year / apple_harvest_interval
  let orange_earnings : ℕ := orange_harvests_per_year * orange_harvest_value
  let apple_earnings : ℕ := apple_harvests_per_year * apple_harvest_value
  orange_earnings + apple_earnings

/-- Theorem stating that Keaton's annual farm earnings are $420 --/
theorem keaton_annual_earnings : farm_earnings = 420 := by
  sorry

end NUMINAMATH_CALUDE_keaton_annual_earnings_l1452_145290


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1452_145246

/-- Given two vectors a and b in ℝ³, where a is parallel to b, prove that m + n = 4 -/
theorem parallel_vectors_sum (a b : ℝ × ℝ × ℝ) (m n : ℝ) : 
  a = (2, -1, 3) → b = (4, m, n) → (∃ (k : ℝ), a = k • b) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1452_145246


namespace NUMINAMATH_CALUDE_total_pages_proof_l1452_145210

/-- The number of pages Jairus read -/
def jairus_pages : ℕ := 20

/-- The number of pages Arniel read -/
def arniel_pages : ℕ := 2 * jairus_pages + 2

/-- The total number of pages read by Jairus and Arniel -/
def total_pages : ℕ := jairus_pages + arniel_pages

theorem total_pages_proof : total_pages = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_proof_l1452_145210


namespace NUMINAMATH_CALUDE_distance_between_stations_l1452_145204

/-- The distance between two stations given train travel information -/
theorem distance_between_stations
  (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ)
  (h1 : speed1 = 20)
  (h2 : time1 = 3)
  (h3 : speed2 = 25)
  (h4 : time2 = 2)
  (h5 : speed1 * time1 + speed2 * time2 = speed1 * time1 + speed2 * time2) :
  speed1 * time1 + speed2 * time2 = 110 := by
  sorry

#check distance_between_stations

end NUMINAMATH_CALUDE_distance_between_stations_l1452_145204


namespace NUMINAMATH_CALUDE_base3_to_base10_20123_l1452_145288

/-- Converts a base 3 number to base 10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number --/
def base3Number : List Nat := [3, 2, 1, 0, 2]

/-- Theorem stating that the base 10 equivalent of 20123 (base 3) is 180 --/
theorem base3_to_base10_20123 :
  base3ToBase10 base3Number = 180 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_20123_l1452_145288


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_b_value_l1452_145212

theorem polynomial_equality_implies_b_value 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, (4*x^2 - 2*x + 5/2)*(a*x^2 + b*x + c) = 
                 12*x^4 - 8*x^3 + 15*x^2 - 5*x + 5/2) : 
  b = -1/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_b_value_l1452_145212


namespace NUMINAMATH_CALUDE_smallest_inverse_domain_l1452_145205

-- Define the function g
def g (x : ℝ) : ℝ := (x + 1)^2 - 6

-- State the theorem
theorem smallest_inverse_domain (d : ℝ) :
  (∀ x y, x ∈ Set.Ici d → y ∈ Set.Ici d → g x = g y → x = y) ∧ 
  (∀ d' < d, ∃ x y, x ∈ Set.Ici d' → y ∈ Set.Ici d' → x ≠ y ∧ g x = g y) ↔ 
  d = -1 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_domain_l1452_145205


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1452_145260

theorem complex_number_quadrant (z : ℂ) : z * Complex.I = 2 - Complex.I → z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1452_145260


namespace NUMINAMATH_CALUDE_lunch_break_duration_l1452_145248

/-- Represents the painting team and their work schedule --/
structure PaintingTeam where
  maria_rate : ℝ
  assistants_rate : ℝ
  lunch_break : ℝ

/-- Monday's work schedule --/
def monday_work (team : PaintingTeam) : Prop :=
  (9 - team.lunch_break) * (team.maria_rate + team.assistants_rate) = 0.6

/-- Tuesday's work schedule --/
def tuesday_work (team : PaintingTeam) : Prop :=
  (7 - team.lunch_break) * team.assistants_rate = 0.3

/-- Wednesday's work schedule --/
def wednesday_work (team : PaintingTeam) : Prop :=
  (5 - team.lunch_break) * team.maria_rate = 0.1

/-- The main theorem stating that the lunch break is 42 minutes --/
theorem lunch_break_duration (team : PaintingTeam) :
  monday_work team → tuesday_work team → wednesday_work team →
  team.lunch_break * 60 = 42 := by
  sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l1452_145248


namespace NUMINAMATH_CALUDE_coin_problem_l1452_145277

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50

def total_coins : ℕ := 13
def total_value : ℕ := 163

theorem coin_problem (pennies nickels dimes quarters half_dollars : ℕ) 
  (h1 : pennies + nickels + dimes + quarters + half_dollars = total_coins)
  (h2 : pennies * penny_value + nickels * nickel_value + dimes * dime_value + 
        quarters * quarter_value + half_dollars * half_dollar_value = total_value)
  (h3 : pennies ≥ 1)
  (h4 : nickels ≥ 1)
  (h5 : dimes ≥ 1)
  (h6 : quarters ≥ 1)
  (h7 : half_dollars ≥ 1) :
  dimes = 3 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l1452_145277


namespace NUMINAMATH_CALUDE_tokens_theorem_l1452_145294

/-- The number of tokens Elsa has -/
def elsa_tokens : ℕ := 60

/-- The number of tokens Angus has -/
def x : ℕ := elsa_tokens - (elsa_tokens / 4)

/-- The number of tokens Bella has -/
def y : ℕ := elsa_tokens + (x^2 - 10)

theorem tokens_theorem : x = 45 ∧ y = 2075 := by
  sorry

end NUMINAMATH_CALUDE_tokens_theorem_l1452_145294


namespace NUMINAMATH_CALUDE_touch_football_point_difference_l1452_145243

/-- The point difference between two teams in a touch football game -/
def point_difference (
  touchdown_points : ℕ)
  (extra_point_points : ℕ)
  (field_goal_points : ℕ)
  (team1_touchdowns : ℕ)
  (team1_extra_points : ℕ)
  (team1_field_goals : ℕ)
  (team2_touchdowns : ℕ)
  (team2_extra_points : ℕ)
  (team2_field_goals : ℕ) : ℕ :=
  (team2_touchdowns * touchdown_points +
   team2_extra_points * extra_point_points +
   team2_field_goals * field_goal_points) -
  (team1_touchdowns * touchdown_points +
   team1_extra_points * extra_point_points +
   team1_field_goals * field_goal_points)

theorem touch_football_point_difference :
  point_difference 7 1 3 6 4 2 8 6 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_touch_football_point_difference_l1452_145243


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l1452_145265

/-- Given four square regions with perimeters p₁, p₂, p₃, and p₄, 
    this theorem proves that the ratio of the area of the second square 
    to the area of the fourth square is 9/16 when p₁ = 16, p₂ = 36, p₃ = p₄ = 48. -/
theorem area_ratio_of_squares (p₁ p₂ p₃ p₄ : ℝ) 
    (h₁ : p₁ = 16) (h₂ : p₂ = 36) (h₃ : p₃ = 48) (h₄ : p₄ = 48) :
    (p₂ / 4)^2 / (p₄ / 4)^2 = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l1452_145265


namespace NUMINAMATH_CALUDE_frisbee_price_problem_l1452_145223

/-- The price of the other frisbees in a sporting goods store -/
theorem frisbee_price_problem :
  ∀ (F₃ F_x x : ℕ),
    F₃ + F_x = 64 →
    3 * F₃ + x * F_x = 200 →
    F_x ≥ 8 →
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_frisbee_price_problem_l1452_145223


namespace NUMINAMATH_CALUDE_jack_socks_purchase_l1452_145228

/-- The number of pairs of socks Jack needs to buy -/
def num_socks : ℕ := 2

/-- The cost of each pair of socks in dollars -/
def sock_cost : ℚ := 9.5

/-- The cost of the shoes in dollars -/
def shoe_cost : ℕ := 92

/-- The total amount Jack needs in dollars -/
def total_amount : ℕ := 111

theorem jack_socks_purchase :
  sock_cost * num_socks + shoe_cost = total_amount :=
by sorry

end NUMINAMATH_CALUDE_jack_socks_purchase_l1452_145228


namespace NUMINAMATH_CALUDE_gift_package_combinations_l1452_145221

theorem gift_package_combinations : 
  let wrapping_paper_varieties : ℕ := 10
  let ribbon_colors : ℕ := 5
  let gift_card_types : ℕ := 5
  let gift_tag_types : ℕ := 2
  wrapping_paper_varieties * ribbon_colors * gift_card_types * gift_tag_types = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_gift_package_combinations_l1452_145221


namespace NUMINAMATH_CALUDE_tourist_distribution_theorem_l1452_145284

/-- The number of ways to distribute tourists among guides --/
def distribute_tourists (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  num_guides ^ num_tourists

/-- The number of ways all tourists can choose the same guide --/
def all_same_guide (num_guides : ℕ) : ℕ := num_guides

/-- The number of valid distributions of tourists among guides --/
def valid_distributions (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  distribute_tourists num_tourists num_guides - all_same_guide num_guides

theorem tourist_distribution_theorem :
  valid_distributions 8 3 = 6558 := by
  sorry

end NUMINAMATH_CALUDE_tourist_distribution_theorem_l1452_145284


namespace NUMINAMATH_CALUDE_roberto_outfits_l1452_145234

/-- Represents the number of trousers Roberto has -/
def num_trousers : ℕ := 4

/-- Represents the number of shirts Roberto has -/
def num_shirts : ℕ := 7

/-- Represents the number of jackets Roberto has -/
def num_jackets : ℕ := 5

/-- Represents the number of hat options Roberto has (wear or not wear) -/
def num_hat_options : ℕ := 2

/-- Calculates the total number of outfit combinations -/
def total_outfits : ℕ := num_trousers * num_shirts * num_jackets * num_hat_options

/-- Theorem stating that the total number of outfits Roberto can create is 280 -/
theorem roberto_outfits : total_outfits = 280 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l1452_145234


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1452_145237

theorem quadratic_roots_property (b c : ℝ) : 
  (3 * b^2 + 5 * b - 2 = 0) → 
  (3 * c^2 + 5 * c - 2 = 0) → 
  (b-1)*(c-1) = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1452_145237


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1452_145207

theorem fraction_evaluation : (3 : ℚ) / (2 - 5 / 4) = 4 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1452_145207


namespace NUMINAMATH_CALUDE_different_monotonicity_implies_inequality_l1452_145256

/-- Given a > 1, a ≠ 2, and (a-1)^x and (1/a)^x have different monotonicities,
    prove that (a-1)^(1/3) > (1/a)^3 -/
theorem different_monotonicity_implies_inequality (a : ℝ) 
  (h1 : a > 1) 
  (h2 : a ≠ 2) 
  (h3 : ∀ x y : ℝ, (∃ ε > 0, ∀ δ ∈ Set.Ioo (x - ε) (x + ε), 
    ((a - 1) ^ δ - (a - 1) ^ x) * ((1 / a) ^ δ - (1 / a) ^ x) < 0)) :
  (a - 1) ^ (1 / 3) > (1 / a) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_different_monotonicity_implies_inequality_l1452_145256


namespace NUMINAMATH_CALUDE_apple_eating_theorem_l1452_145297

/-- Represents the rates at which the fish and bird eat the apple -/
structure EatingRates where
  fish : ℝ
  bird : ℝ

/-- Represents the portions of the apple above and below water -/
structure AppleDivision where
  above_water : ℝ
  below_water : ℝ

/-- Represents the portions of the apple eaten by the fish and bird -/
structure EatenPortions where
  fish : ℝ
  bird : ℝ

/-- Theorem stating the portions of apple eaten by fish and bird -/
theorem apple_eating_theorem (rates : EatingRates) (division : AppleDivision) 
    (h1 : rates.fish = 2 * rates.bird) 
    (h2 : division.above_water + division.below_water = 1) :
  ∃ (portions : EatenPortions), 
    portions.fish = 2/3 ∧ 
    portions.bird = 1/3 ∧ 
    portions.fish + portions.bird = 1 := by
  sorry

end NUMINAMATH_CALUDE_apple_eating_theorem_l1452_145297


namespace NUMINAMATH_CALUDE_shooting_events_l1452_145283

-- Define the sample space
variable (Ω : Type)
variable [MeasurableSpace Ω]

-- Define the events
variable (both_hit : Set Ω)
variable (exactly_one_hit : Set Ω)
variable (both_miss : Set Ω)
variable (at_least_one_hit : Set Ω)

-- Define the probability measure
variable (P : Measure Ω)

-- Theorem statement
theorem shooting_events :
  (Disjoint exactly_one_hit both_hit) ∧
  (both_miss = at_least_one_hit.compl) := by
sorry

end NUMINAMATH_CALUDE_shooting_events_l1452_145283


namespace NUMINAMATH_CALUDE_combined_rocket_height_l1452_145222

def first_rocket_height : ℝ := 500

theorem combined_rocket_height :
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 := by
  sorry

end NUMINAMATH_CALUDE_combined_rocket_height_l1452_145222


namespace NUMINAMATH_CALUDE_irrational_plus_five_iff_l1452_145251

theorem irrational_plus_five_iff (a : ℝ) : Irrational (a + 5) ↔ Irrational a := by sorry

end NUMINAMATH_CALUDE_irrational_plus_five_iff_l1452_145251


namespace NUMINAMATH_CALUDE_hundred_to_fifty_equals_ten_to_hundred_l1452_145233

theorem hundred_to_fifty_equals_ten_to_hundred : 100 ^ 50 = 10 ^ 100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_to_fifty_equals_ten_to_hundred_l1452_145233


namespace NUMINAMATH_CALUDE_product_plus_one_is_square_l1452_145208

theorem product_plus_one_is_square (n : ℕ) : 
  ∃ m : ℕ, n * (n + 1) * (n + 2) * (n + 3) + 1 = m ^ 2 := by
  sorry

#check product_plus_one_is_square 7321

end NUMINAMATH_CALUDE_product_plus_one_is_square_l1452_145208


namespace NUMINAMATH_CALUDE_xy_value_l1452_145247

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^(Real.sqrt y) = 27) (h2 : (Real.sqrt x)^y = 9) : 
  x * y = 16 * Real.rpow 3 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1452_145247


namespace NUMINAMATH_CALUDE_women_who_left_l1452_145296

/-- Proves that 3 women left the room given the initial and final conditions --/
theorem women_who_left (initial_men : ℕ) (initial_women : ℕ) 
  (h_ratio : initial_men * 5 = initial_women * 4)
  (h_final_men : initial_men + 2 = 14)
  (h_final_women : 2 * (initial_women - 3) = 24) : 
  ∃ (x : ℕ), x = 3 ∧ 2 * (initial_women - x) = 24 :=
by
  sorry

#check women_who_left

end NUMINAMATH_CALUDE_women_who_left_l1452_145296


namespace NUMINAMATH_CALUDE_median_after_removal_l1452_145257

def room_sequence : List Nat := List.range 26

def remaining_rooms (seq : List Nat) : List Nat :=
  seq.filter (fun n => n ≠ 15 ∧ n ≠ 20 ∧ n ≠ 0)

theorem median_after_removal (seq : List Nat) (h : seq = room_sequence) :
  (remaining_rooms seq).get? ((remaining_rooms seq).length / 2) = some 12 := by
  sorry

end NUMINAMATH_CALUDE_median_after_removal_l1452_145257


namespace NUMINAMATH_CALUDE_polynomial_equality_l1452_145240

/-- Given that 7x^5 + 4x^3 - 3x + p(x) = 2x^4 - 10x^3 + 5x - 2,
    prove that p(x) = -7x^5 + 2x^4 - 6x^3 + 2x - 2 -/
theorem polynomial_equality (x : ℝ) (p : ℝ → ℝ) 
  (h : ∀ x, 7 * x^5 + 4 * x^3 - 3 * x + p x = 2 * x^4 - 10 * x^3 + 5 * x - 2) : 
  p = fun x ↦ -7 * x^5 + 2 * x^4 - 6 * x^3 + 2 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1452_145240


namespace NUMINAMATH_CALUDE_area_of_triangle_qpo_l1452_145282

/-- Represents a parallelogram ABCD with specific properties -/
structure SpecialParallelogram where
  -- The area of the parallelogram
  area : ℝ
  -- DP bisects BC
  dp_bisects_bc : Bool
  -- CQ bisects AD
  cq_bisects_ad : Bool
  -- DP divides triangle BCD into regions of area k/4 and 3k/4
  dp_divides_bcd : Bool

/-- Theorem stating the area of triangle QPO in the special parallelogram -/
theorem area_of_triangle_qpo (ABCD : SpecialParallelogram) :
  let k := ABCD.area
  let area_qpo := (9 : ℝ) / 8 * k
  ABCD.dp_bisects_bc ∧ ABCD.cq_bisects_ad ∧ ABCD.dp_divides_bcd →
  area_qpo = (9 : ℝ) / 8 * k :=
by
  sorry


end NUMINAMATH_CALUDE_area_of_triangle_qpo_l1452_145282


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l1452_145244

theorem solve_system_of_equations :
  ∀ (x y m n : ℝ),
  (4 * x + 3 * y = m) →
  (6 * x - y = n) →
  ((m / 3 + n / 8 = 8) ∧ (m / 6 + n / 2 = 11)) →
  (x = 3 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l1452_145244


namespace NUMINAMATH_CALUDE_count_distinct_arrangements_l1452_145281

/-- A regular five-pointed star with 10 positions for placing objects -/
structure StarArrangement where
  positions : Fin 10 → Fin 10

/-- The group of symmetries of a regular five-pointed star -/
def starSymmetryGroup : Fintype G := sorry

/-- The number of distinct arrangements of 10 different objects on a regular five-pointed star,
    considering rotations and reflections as equivalent -/
def distinctArrangements : ℕ := sorry

/-- Theorem stating the number of distinct arrangements -/
theorem count_distinct_arrangements :
  distinctArrangements = Nat.factorial 10 / 10 := by sorry

end NUMINAMATH_CALUDE_count_distinct_arrangements_l1452_145281


namespace NUMINAMATH_CALUDE_rectangular_array_sum_ratio_l1452_145202

theorem rectangular_array_sum_ratio (a : Fin 50 → Fin 40 → ℝ) :
  let row_sum : Fin 50 → ℝ := λ i => (Finset.univ.sum (λ j => a i j))
  let col_sum : Fin 40 → ℝ := λ j => (Finset.univ.sum (λ i => a i j))
  let C : ℝ := (Finset.univ.sum row_sum) / 50
  let D : ℝ := (Finset.univ.sum col_sum) / 40
  C / D = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_array_sum_ratio_l1452_145202


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l1452_145220

theorem largest_stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 1520)
  (h2 : book2 = 1900)
  (h3 : book3 = 2280) :
  Nat.gcd book1 (Nat.gcd book2 book3) = 380 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l1452_145220


namespace NUMINAMATH_CALUDE_smallest_divisor_cube_sum_l1452_145255

theorem smallest_divisor_cube_sum (n : ℕ) : n ≥ 2 →
  (∃ m : ℕ, m > 0 ∧ m ∣ n ∧
    (∃ d : ℕ, d > 1 ∧ d ∣ n ∧
      (∀ k : ℕ, k > 1 ∧ k ∣ n → k ≥ d) ∧
      n = d^3 + m^3)) →
  n = 16 ∨ n = 72 ∨ n = 520 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_cube_sum_l1452_145255


namespace NUMINAMATH_CALUDE_product_equality_l1452_145272

theorem product_equality (a b c : ℝ) 
  (h : ∀ x y z : ℝ, x * y * z = Real.sqrt ((x + 2) * (y + 3)) / (z + 1)) : 
  6 * 15 * 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1452_145272


namespace NUMINAMATH_CALUDE_sequence_product_l1452_145201

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The line l passing through the origin with normal vector (3,1) -/
def Line (x y : ℝ) : Prop := 3 * x + y = 0

/-- The sequence {a_n} satisfies the condition that (a_{n+1}, a_n) lies on the line for all n -/
def SequenceOnLine (a : Sequence) : Prop := ∀ n : ℕ, Line (a (n + 1)) (a n)

theorem sequence_product (a : Sequence) (h1 : SequenceOnLine a) (h2 : a 2 = 6) :
  a 1 * a 2 * a 3 * a 4 * a 5 = -32 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l1452_145201


namespace NUMINAMATH_CALUDE_bicyclist_average_speed_l1452_145258

/-- The average speed of a bicyclist's trip -/
theorem bicyclist_average_speed :
  let total_distance : ℝ := 250
  let first_part_distance : ℝ := 100
  let first_part_speed : ℝ := 20
  let second_part_distance : ℝ := total_distance - first_part_distance
  let second_part_speed : ℝ := 15
  let average_speed : ℝ := total_distance / (first_part_distance / first_part_speed + second_part_distance / second_part_speed)
  average_speed = 250 / (100 / 20 + 150 / 15) :=
by
  sorry

#eval (250 : Float) / ((100 : Float) / 20 + (150 : Float) / 15)

end NUMINAMATH_CALUDE_bicyclist_average_speed_l1452_145258


namespace NUMINAMATH_CALUDE_cricket_game_overs_l1452_145259

/-- The number of initial overs in a cricket game -/
def initial_overs : ℕ := 20

/-- The initial run rate in runs per over -/
def initial_run_rate : ℚ := 46/10

/-- The target score in runs -/
def target_score : ℕ := 396

/-- The number of remaining overs -/
def remaining_overs : ℕ := 30

/-- The required run rate for the remaining overs -/
def required_run_rate : ℚ := 10133333333333333/1000000000000000

theorem cricket_game_overs :
  initial_overs * initial_run_rate + 
  remaining_overs * required_run_rate = target_score :=
sorry

end NUMINAMATH_CALUDE_cricket_game_overs_l1452_145259


namespace NUMINAMATH_CALUDE_sum_ratio_equals_half_l1452_145241

theorem sum_ratio_equals_half
  (a b c x y z : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 10)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 40)
  (sum_products : a*x + b*y + c*z = 20) :
  (a + b + c) / (x + y + z) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_ratio_equals_half_l1452_145241


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1452_145203

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    ∀ (x : ℝ), x ≠ 1 → x ≠ 4 → x ≠ -2 →
      (x^2 - 18) / ((x - 1) * (x - 4) * (x + 2)) =
      P / (x - 1) + Q / (x - 4) + R / (x + 2) ∧
      P = 17/9 ∧ Q = 1/9 ∧ R = -5/9 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1452_145203


namespace NUMINAMATH_CALUDE_binomial_sum_l1452_145231

theorem binomial_sum : (7 : ℕ).choose 2 + (6 : ℕ).choose 4 = 36 := by sorry

end NUMINAMATH_CALUDE_binomial_sum_l1452_145231


namespace NUMINAMATH_CALUDE_C_power_50_l1452_145200

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l1452_145200


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1452_145291

def A : Set ℝ := {x | x < -3}
def B : Set ℝ := {-5, -4, -3, 1}

theorem intersection_of_A_and_B : A ∩ B = {-5, -4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1452_145291


namespace NUMINAMATH_CALUDE_triangle_properties_l1452_145270

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_properties (t : Triangle) :
  (Real.cos t.C + (Real.cos t.A - Real.sqrt 3 * Real.sin t.A) * Real.cos t.B = 0) →
  (t.a + t.c = 1) →
  (t.B = π / 3 ∧ 1 / 2 ≤ t.b ∧ t.b < 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1452_145270


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1452_145227

theorem right_triangle_side_length (D E F : ℝ) : 
  -- DEF is a right triangle with angle E being right
  (D^2 + E^2 = F^2) →
  -- cos(D) = (8√85)/85
  (Real.cos D = (8 * Real.sqrt 85) / 85) →
  -- EF:DF = 1:2
  (E / F = 1 / 2) →
  -- The length of DF is 2√85
  F = 2 * Real.sqrt 85 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1452_145227


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1452_145239

/-- An isosceles triangle with congruent sides of length 7 cm and perimeter 23 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length : ∀ (base : ℝ), 
  base > 0 → -- The base length is positive
  7 > 0 → -- The congruent side length is positive
  2 * 7 + base = 23 → -- The perimeter is 23 cm
  base = 9 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1452_145239


namespace NUMINAMATH_CALUDE_simplify_fraction_division_l1452_145252

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2)) := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_division_l1452_145252


namespace NUMINAMATH_CALUDE_investment_value_l1452_145276

/-- Proves that the value of the larger investment is $1500 given the specified conditions. -/
theorem investment_value (x : ℝ) : 
  (0.07 * 500 + 0.27 * x = 0.22 * (500 + x)) → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_investment_value_l1452_145276


namespace NUMINAMATH_CALUDE_greatest_b_value_l1452_145269

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, x^2 - 12*x + 32 ≤ 0 → x ≤ 8) ∧ 
  (8^2 - 12*8 + 32 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l1452_145269


namespace NUMINAMATH_CALUDE_minimum_value_implies_m_equals_one_l1452_145287

-- Define the domain D
def D : Set ℝ := Set.Icc 1 2

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x - m^2

-- Theorem statement
theorem minimum_value_implies_m_equals_one :
  ∀ m : ℝ, (∀ x ∈ D, g m x ≥ 2) ∧ (∃ x ∈ D, g m x = 2) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_m_equals_one_l1452_145287


namespace NUMINAMATH_CALUDE_parabola_directrix_l1452_145261

/-- The directrix of a parabola y² = 2x is x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 2*x → (∃ (k : ℝ), k = -1/2 ∧ (∀ (x₀ y₀ : ℝ), y₀^2 = 2*x₀ → x₀ = k)) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1452_145261


namespace NUMINAMATH_CALUDE_y_value_proof_l1452_145249

theorem y_value_proof (y : ℕ) : 
  (∃ (factors : Finset ℕ), factors.card = 24 ∧ (∀ f ∈ factors, f ∣ y) ∧ (∀ f ∈ factors, f > 0)) →
  18 ∣ y →
  20 ∣ y →
  y = 360 :=
by sorry

end NUMINAMATH_CALUDE_y_value_proof_l1452_145249


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1452_145273

def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - k

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

def passes_through_quadrant (f : ℝ → ℝ) (quadrant : ℕ) : Prop :=
  match quadrant with
  | 1 => ∃ x > 0, f x > 0
  | 2 => ∃ x < 0, f x > 0
  | 3 => ∃ x < 0, f x < 0
  | 4 => ∃ x > 0, f x < 0
  | _ => False

theorem linear_function_quadrants (k : ℝ) :
  decreasing_function (linear_function k) →
  (passes_through_quadrant (linear_function k) 1 ∧
   passes_through_quadrant (linear_function k) 2 ∧
   passes_through_quadrant (linear_function k) 4) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l1452_145273


namespace NUMINAMATH_CALUDE_divisible_by_2power10000_within_day_l1452_145285

/-- Represents a card with a natural number -/
structure Card where
  value : ℕ

/-- Represents the state of the table at any given time -/
structure TableState where
  cards : List Card
  time : ℕ

/-- Checks if a number is divisible by 2^10000 -/
def isDivisibleBy2Power10000 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (2^10000)

/-- The process of adding a new card every minute -/
def addNewCard (state : TableState) : TableState :=
  sorry

/-- The main theorem to be proved -/
theorem divisible_by_2power10000_within_day
  (initial_cards : List Card)
  (h1 : initial_cards.length = 100)
  (h2 : (initial_cards.filter (fun c => c.value % 2 = 1)).length = 43) :
  ∃ (final_state : TableState),
    final_state.time ≤ 1440 ∧
    ∃ (c : Card), c ∈ final_state.cards ∧ isDivisibleBy2Power10000 c.value :=
  sorry

end NUMINAMATH_CALUDE_divisible_by_2power10000_within_day_l1452_145285


namespace NUMINAMATH_CALUDE_largest_valid_factor_of_130000_l1452_145275

/-- A function that checks if a natural number contains the digit 0 or 5 --/
def containsZeroOrFive (n : ℕ) : Prop := sorry

/-- The largest factor of 130000 that does not contain the digit 0 or 5 --/
def largestValidFactor : ℕ := 26

theorem largest_valid_factor_of_130000 :
  (largestValidFactor ∣ 130000) ∧ 
  ¬containsZeroOrFive largestValidFactor ∧
  ∀ k : ℕ, k > largestValidFactor → (k ∣ 130000) → containsZeroOrFive k := by sorry

end NUMINAMATH_CALUDE_largest_valid_factor_of_130000_l1452_145275


namespace NUMINAMATH_CALUDE_problem_1_l1452_145267

theorem problem_1 : (1) - 4^2 / (-32) * (2/3)^2 = 2/9 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1452_145267
