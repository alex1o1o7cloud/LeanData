import Mathlib

namespace NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l1782_178236

theorem fourth_power_nested_sqrt : 
  (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt 2)))^4 = 7 + 4 * Real.sqrt 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l1782_178236


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1782_178258

-- Part 1
theorem inequality_solution_1 (x : ℝ) :
  (x + 1) / (x - 2) ≥ 3 ↔ 2 < x ∧ x ≤ 7/2 :=
sorry

-- Part 2
theorem inequality_solution_2 (x a : ℝ) :
  x^2 - a*x - 2*a^2 ≤ 0 ↔
    (a = 0 ∧ x = 0) ∨
    (a > 0 ∧ -a ≤ x ∧ x ≤ 2*a) ∨
    (a < 0 ∧ 2*a ≤ x ∧ x ≤ -a) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1782_178258


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1782_178221

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 - 2^(a+2) * x - 2^(a+3) + 12 > 0) ↔ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1782_178221


namespace NUMINAMATH_CALUDE_cooler_contents_l1782_178259

/-- The number of cherry sodas in the cooler -/
def cherry_sodas : ℕ := 8

/-- The number of orange pops in the cooler -/
def orange_pops : ℕ := 2 * cherry_sodas

/-- The total number of cans in the cooler -/
def total_cans : ℕ := 24

theorem cooler_contents : 
  cherry_sodas + orange_pops = total_cans ∧ cherry_sodas = 8 := by
  sorry

end NUMINAMATH_CALUDE_cooler_contents_l1782_178259


namespace NUMINAMATH_CALUDE_exists_same_color_right_triangle_l1782_178232

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point

-- Define a coloring function
def Coloring := Point → Color

-- Define a property for a right-angled triangle
def isRightAngledTriangle (p q r : Point) : Prop := sorry

-- Theorem statement
theorem exists_same_color_right_triangle 
  (triangle : EquilateralTriangle) 
  (coloring : Coloring) : 
  ∃ (p q r : Point), 
    (coloring p = coloring q) ∧ 
    (coloring q = coloring r) ∧ 
    isRightAngledTriangle p q r :=
sorry

end NUMINAMATH_CALUDE_exists_same_color_right_triangle_l1782_178232


namespace NUMINAMATH_CALUDE_complex_number_problem_l1782_178260

/-- Given a complex number z = bi (b ∈ ℝ) such that (z-2)/(1+i) is real,
    prove that z = -2i and (m+z)^2 is in the first quadrant iff m < -2 -/
theorem complex_number_problem (b : ℝ) (z : ℂ) (h1 : z = Complex.I * b) 
    (h2 : ∃ (r : ℝ), (z - 2) / (1 + Complex.I) = r) :
  z = -2 * Complex.I ∧ 
  ∀ m : ℝ, (Complex.re ((m + z)^2) > 0 ∧ Complex.im ((m + z)^2) > 0) ↔ m < -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1782_178260


namespace NUMINAMATH_CALUDE_divisors_of_n_squared_l1782_178231

def has_exactly_four_divisors (n : ℕ) : Prop :=
  (∃ p : ℕ, Prime p ∧ n = p^3) ∨
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n = p * q)

theorem divisors_of_n_squared (n : ℕ) (h : has_exactly_four_divisors n) :
  (∃ d : ℕ, d = 7 ∧ (∀ x : ℕ, x ∣ n^2 ↔ x ∈ Finset.range (d + 1))) ∨
  (∃ d : ℕ, d = 9 ∧ (∀ x : ℕ, x ∣ n^2 ↔ x ∈ Finset.range (d + 1))) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_n_squared_l1782_178231


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1782_178250

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence -/
def arithmetic_term (a : ℕ → ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a 1 + (n - 1) * d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (d : ℝ) (h : arithmetic_sequence a d) :
  (a 4 + 4)^2 = (a 2 + 2) * (a 6 + 6) → d = -1 := by
  sorry

#check arithmetic_sequence_common_difference

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1782_178250


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l1782_178244

/-- The binomial expansion of (√x + 2/x²)¹⁰ -/
def binomial_expansion (x : ℝ) : ℕ → ℝ :=
  λ r => (Nat.choose 10 r) * (2^r) * (x^((10 - 5*r)/2))

/-- A term in the expansion is rational if its exponent is an integer -/
def is_rational_term (r : ℕ) : Prop :=
  (10 - 5*r) % 2 = 0

theorem binomial_expansion_properties :
  (∃ (S : Finset ℕ), S.card = 6 ∧ ∀ r, r ∈ S ↔ is_rational_term r) ∧
  (∃ r : ℕ, r = 7 ∧ ∀ k : ℕ, k ≠ r → |binomial_expansion 1 r| ≥ |binomial_expansion 1 k|) ∧
  binomial_expansion 1 7 = 15360 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l1782_178244


namespace NUMINAMATH_CALUDE_percentage_problem_l1782_178274

theorem percentage_problem (p : ℝ) (h1 : 0.25 * 680 = p * 1000 - 30) : p = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1782_178274


namespace NUMINAMATH_CALUDE_min_value_expression_l1782_178239

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  b / (3 * a) + 3 / b ≥ 5 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ b / (3 * a) + 3 / b = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1782_178239


namespace NUMINAMATH_CALUDE_height_growth_l1782_178264

theorem height_growth (current_height : ℝ) (growth_rate : ℝ) (original_height : ℝ) : 
  current_height = 147 ∧ growth_rate = 0.05 → original_height = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_height_growth_l1782_178264


namespace NUMINAMATH_CALUDE_ackermann_3_2_l1782_178263

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_3_2 : A 3 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_ackermann_3_2_l1782_178263


namespace NUMINAMATH_CALUDE_equation_solution_l1782_178261

theorem equation_solution :
  ∃ x : ℚ, (5 * x + 6 * x = 360 - 10 * (x - 4)) ∧ x = 400 / 21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1782_178261


namespace NUMINAMATH_CALUDE_inverse_inequality_l1782_178267

theorem inverse_inequality (a b : ℝ) (h1 : 0 > a) (h2 : a > b) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l1782_178267


namespace NUMINAMATH_CALUDE_cubic_local_max_l1782_178224

/-- Given a cubic function with a local maximum, prove the product of two coefficients -/
theorem cubic_local_max (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∧ 
  (f 1 = -3) →
  a * b = 9 := by
sorry

end NUMINAMATH_CALUDE_cubic_local_max_l1782_178224


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1782_178272

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.000136 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.36 ∧ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1782_178272


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l1782_178209

theorem product_from_hcf_lcm (a b : ℕ+) (h_hcf : Nat.gcd a b = 14) (h_lcm : Nat.lcm a b = 183) :
  a * b = 2562 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l1782_178209


namespace NUMINAMATH_CALUDE_intersection_M_N_l1782_178294

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1782_178294


namespace NUMINAMATH_CALUDE_equation_solution_l1782_178234

theorem equation_solution :
  ∃! x : ℝ, x ≠ -3 ∧ (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1782_178234


namespace NUMINAMATH_CALUDE_multinomial_binomial_equality_l1782_178203

theorem multinomial_binomial_equality (n : ℕ) : 
  Nat.choose n 2 * Nat.choose (n - 2) 2 = 3 * Nat.choose n 4 + 3 * Nat.choose n 3 := by
  sorry

end NUMINAMATH_CALUDE_multinomial_binomial_equality_l1782_178203


namespace NUMINAMATH_CALUDE_gohul_independent_time_l1782_178206

/-- Ram's work rate in job completion per day -/
def ram_rate : ℚ := 1 / 10

/-- Time taken when Ram and Gohul work together -/
def combined_time : ℚ := 5.999999999999999

/-- Gohul's independent work time -/
def gohul_time : ℚ := 15

/-- Combined work rate of Ram and Gohul -/
def combined_rate : ℚ := 1 / combined_time

theorem gohul_independent_time :
  ram_rate + (1 / gohul_time) = combined_rate :=
sorry

end NUMINAMATH_CALUDE_gohul_independent_time_l1782_178206


namespace NUMINAMATH_CALUDE_information_spread_time_l1782_178223

theorem information_spread_time (population : ℕ) (h : population = 1000000) :
  ∃ n : ℕ, n ≥ 19 ∧ 2^(n+1) - 1 ≥ population :=
by sorry

end NUMINAMATH_CALUDE_information_spread_time_l1782_178223


namespace NUMINAMATH_CALUDE_functional_sequence_a10_l1782_178297

/-- A sequence satisfying a functional equation -/
def FunctionalSequence (a : ℕ+ → ℤ) : Prop :=
  ∀ p q : ℕ+, a (p + q) = a p + a q

theorem functional_sequence_a10 (a : ℕ+ → ℤ) 
  (h1 : FunctionalSequence a) (h2 : a 2 = -6) : 
  a 10 = -30 := by sorry

end NUMINAMATH_CALUDE_functional_sequence_a10_l1782_178297


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l1782_178208

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l1782_178208


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l1782_178275

theorem alcohol_mixture_problem (x : ℝ) :
  (x * 50 + 30 * 150) / (50 + 150) = 25 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l1782_178275


namespace NUMINAMATH_CALUDE_second_smallest_natural_with_remainder_l1782_178286

theorem second_smallest_natural_with_remainder : ∃ n : ℕ, 
  n > 500 ∧ 
  n % 7 = 3 ∧ 
  (∃! m : ℕ, m > 500 ∧ m % 7 = 3 ∧ m < n) ∧
  n = 514 :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_natural_with_remainder_l1782_178286


namespace NUMINAMATH_CALUDE_guest_bedroom_area_l1782_178200

/-- Proves that the area of each guest bedroom is 200 sq ft given the specified conditions --/
theorem guest_bedroom_area
  (total_rent : ℝ)
  (rent_rate : ℝ)
  (master_area : ℝ)
  (common_area : ℝ)
  (h1 : total_rent = 3000)
  (h2 : rent_rate = 2)
  (h3 : master_area = 500)
  (h4 : common_area = 600)
  : ∃ (guest_bedroom_area : ℝ),
    guest_bedroom_area = 200 ∧
    total_rent / rent_rate = master_area + common_area + 2 * guest_bedroom_area :=
by sorry

end NUMINAMATH_CALUDE_guest_bedroom_area_l1782_178200


namespace NUMINAMATH_CALUDE_intersection_count_l1782_178213

-- Define the two curves
def curve1 (x y : ℝ) : Prop := (x + 2*y - 3) * (4*x - y + 1) = 0
def curve2 (x y : ℝ) : Prop := (2*x - y - 5) * (3*x + 4*y - 8) = 0

-- Define what it means for a point to be on both curves
def intersection_point (x y : ℝ) : Prop := curve1 x y ∧ curve2 x y

-- State the theorem
theorem intersection_count : 
  ∃ (points : Finset (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ points ↔ intersection_point p.1 p.2) ∧ 
    points.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l1782_178213


namespace NUMINAMATH_CALUDE_constant_sum_of_squares_l1782_178240

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of a point being on the major axis -/
def on_major_axis (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

/-- Definition of a line with slope 1/2 passing through (m, 0) -/
def line (m x y : ℝ) : Prop := y = (x - m) / 2

/-- Statement of the theorem -/
theorem constant_sum_of_squares (m x₁ y₁ x₂ y₂ : ℝ) : 
  ellipse 1 (Real.sqrt 3 / 2) →
  on_major_axis m →
  line m x₁ y₁ →
  line m x₂ y₂ →
  ellipse x₁ y₁ →
  ellipse x₂ y₂ →
  (x₁ - m)^2 + y₁^2 + (x₂ - m)^2 + y₂^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_constant_sum_of_squares_l1782_178240


namespace NUMINAMATH_CALUDE_largest_angle_of_hexagon_l1782_178225

/-- Proves that in a convex hexagon with given interior angle measures, the largest angle is 4374/21 degrees -/
theorem largest_angle_of_hexagon (a : ℚ) : 
  (a + 2) + (2 * a - 3) + (3 * a + 1) + (4 * a) + (5 * a - 4) + (6 * a + 2) = 720 →
  max (a + 2) (max (2 * a - 3) (max (3 * a + 1) (max (4 * a) (max (5 * a - 4) (6 * a + 2))))) = 4374 / 21 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_of_hexagon_l1782_178225


namespace NUMINAMATH_CALUDE_rogers_crayons_l1782_178281

theorem rogers_crayons (new_crayons used_crayons broken_crayons : ℕ) 
  (h1 : new_crayons = 2)
  (h2 : used_crayons = 4)
  (h3 : broken_crayons = 8) :
  new_crayons + used_crayons + broken_crayons = 14 := by
  sorry

end NUMINAMATH_CALUDE_rogers_crayons_l1782_178281


namespace NUMINAMATH_CALUDE_gcd_18_30_45_l1782_178235

theorem gcd_18_30_45 : Nat.gcd 18 (Nat.gcd 30 45) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_45_l1782_178235


namespace NUMINAMATH_CALUDE_recreation_area_tents_l1782_178268

/-- Represents the number of tents in different parts of the campsite -/
structure CampsiteTents where
  north : ℕ
  east : ℕ
  center : ℕ
  south : ℕ

/-- Calculates the total number of tents in the campsite -/
def total_tents (c : CampsiteTents) : ℕ :=
  c.north + c.east + c.center + c.south

/-- Theorem stating the total number of tents in the recreation area -/
theorem recreation_area_tents :
  ∃ (c : CampsiteTents),
    c.north = 100 ∧
    c.east = 2 * c.north ∧
    c.center = 4 * c.north ∧
    c.south = 200 ∧
    total_tents c = 900 := by
  sorry

end NUMINAMATH_CALUDE_recreation_area_tents_l1782_178268


namespace NUMINAMATH_CALUDE_pyramid_display_sum_l1782_178205

/-- Proves that the sum of an arithmetic sequence with given parameters is 255 -/
theorem pyramid_display_sum : 
  ∀ (a₁ aₙ d n : ℕ),
  a₁ = 12 →
  aₙ = 39 →
  d = 3 →
  aₙ = a₁ + (n - 1) * d →
  (n : ℕ) * (a₁ + aₙ) / 2 = 255 :=
by
  sorry

end NUMINAMATH_CALUDE_pyramid_display_sum_l1782_178205


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1782_178228

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (2 * z - w) = 20)
  (h2 : Complex.abs (z + 2 * w) = 10)
  (h3 : Complex.abs (z + w) = 5) :
  Complex.abs z = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1782_178228


namespace NUMINAMATH_CALUDE_reciprocal_problem_l1782_178276

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 3) : 200 * (1 / x) = 1600 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l1782_178276


namespace NUMINAMATH_CALUDE_no_eight_consecutive_almost_squares_l1782_178229

/-- Definition of an almost square -/
def is_almost_square (n : ℕ) : Prop :=
  ∃ k p : ℕ, (n = k^2 ∨ n = k^2 * p) ∧ (p = 1 ∨ Nat.Prime p)

/-- Theorem stating that 8 consecutive almost squares are impossible -/
theorem no_eight_consecutive_almost_squares :
  ¬ ∃ n : ℕ, ∀ i : Fin 8, is_almost_square (n + i) :=
sorry

end NUMINAMATH_CALUDE_no_eight_consecutive_almost_squares_l1782_178229


namespace NUMINAMATH_CALUDE_manufacturing_sector_degrees_l1782_178204

/-- Represents the number of degrees in a full circle -/
def full_circle_degrees : ℝ := 360

/-- Represents the percentage of the circle occupied by the manufacturing department -/
def manufacturing_percentage : ℝ := 45

/-- Theorem: The manufacturing department sector in the circle graph occupies 162 degrees -/
theorem manufacturing_sector_degrees : 
  (manufacturing_percentage / 100) * full_circle_degrees = 162 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_sector_degrees_l1782_178204


namespace NUMINAMATH_CALUDE_shot_put_surface_area_l1782_178207

/-- The surface area of a sphere with diameter 9 inches is 81π square inches. -/
theorem shot_put_surface_area :
  ∀ (d : ℝ), d = 9 → 4 * Real.pi * (d / 2)^2 = 81 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shot_put_surface_area_l1782_178207


namespace NUMINAMATH_CALUDE_machinery_expenditure_l1782_178299

/-- Proves that the amount spent on machinery is $2000 --/
theorem machinery_expenditure (total : ℝ) (raw_materials : ℝ) (cash_percentage : ℝ) :
  total = 5555.56 →
  raw_materials = 3000 →
  cash_percentage = 0.1 →
  total = raw_materials + (total * cash_percentage) + 2000 := by
  sorry

end NUMINAMATH_CALUDE_machinery_expenditure_l1782_178299


namespace NUMINAMATH_CALUDE_box_third_side_length_l1782_178282

/-- Proves that the third side of a rectangular box is 6.75 cm given specific conditions -/
theorem box_third_side_length (num_cubes : ℕ) (cube_volume : ℝ) (side1 side2 : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  side1 = 8 →
  side2 = 12 →
  (num_cubes : ℝ) * cube_volume = side1 * side2 * 6.75 :=
by sorry

end NUMINAMATH_CALUDE_box_third_side_length_l1782_178282


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1782_178270

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 7b₃ is -9/14. -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) : 
  b₁ = 2 → 
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) → 
  (∀ b₂' b₃' : ℝ, (∃ r' : ℝ, b₂' = 2 * r' ∧ b₃' = b₂' * r') → 
    3 * b₂ + 7 * b₃ ≤ 3 * b₂' + 7 * b₃') → 
  3 * b₂ + 7 * b₃ = -9/14 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1782_178270


namespace NUMINAMATH_CALUDE_inequality_proof_l1782_178249

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a > b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1782_178249


namespace NUMINAMATH_CALUDE_parallelogram_area_34_18_l1782_178277

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 34 cm and height 18 cm is 612 square centimeters -/
theorem parallelogram_area_34_18 : parallelogram_area 34 18 = 612 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_34_18_l1782_178277


namespace NUMINAMATH_CALUDE_perfect_square_fraction_l1782_178280

theorem perfect_square_fraction (a b : ℕ+) (k : ℕ) 
  (h : k = (a.val^2 + b.val^2) / (a.val * b.val + 1)) : 
  ∃ (n : ℕ), k = n^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_fraction_l1782_178280


namespace NUMINAMATH_CALUDE_peach_basket_problem_l1782_178242

theorem peach_basket_problem (n : ℕ) : 
  n % 4 = 2 →
  n % 6 = 4 →
  (n + 2) % 8 = 0 →
  120 ≤ n →
  n ≤ 150 →
  n = 142 :=
by sorry

end NUMINAMATH_CALUDE_peach_basket_problem_l1782_178242


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l1782_178233

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def ternary_to_decimal (t : List ℕ) : ℕ :=
  t.enum.foldl (fun acc (i, digit) => acc + digit * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary := [true, true, false, true]  -- 1011 in binary (least significant bit first)
  let ternary := [2, 0, 1]  -- 102 in ternary (least significant digit first)
  (binary_to_decimal binary) * (ternary_to_decimal ternary) = 121 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l1782_178233


namespace NUMINAMATH_CALUDE_enclosure_probability_l1782_178248

def is_valid_configuration (c₁ c₂ c₃ d₁ d₂ d₃ : ℕ) : Prop :=
  d₁ ≥ 2 * c₁ ∧ d₁ > d₂ ∧ d₂ > d₃ ∧ c₁ > c₂ ∧ c₂ > c₃ ∧
  d₁ > c₁ ∧ d₂ > c₂ ∧ d₃ > c₃

def probability_of_valid_configuration : ℚ :=
  1 / 2

theorem enclosure_probability :
  ∀ (S : Finset ℕ) (c₁ c₂ c₃ d₁ d₂ d₃ : ℕ),
    S = Finset.range 100 →
    c₁ ∈ S ∧ c₂ ∈ S ∧ c₃ ∈ S →
    c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ →
    d₁ ∈ S.erase c₁ \ {c₂, c₃} ∧ d₂ ∈ S.erase c₁ \ {c₂, c₃, d₁} ∧ d₃ ∈ S.erase c₁ \ {c₂, c₃, d₁, d₂} →
    probability_of_valid_configuration = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_enclosure_probability_l1782_178248


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1782_178238

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + Complex.I) / (1 - Complex.I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1782_178238


namespace NUMINAMATH_CALUDE_odd_function_inequality_l1782_178253

/-- An odd, differentiable function satisfying certain conditions -/
structure OddFunction where
  f : ℝ → ℝ
  odd : ∀ x, f (-x) = -f x
  diff : Differentiable ℝ f
  cond : ∀ x < 0, 2 * f x + x * deriv f x < 0

/-- Theorem stating the relationship between f(1), 2016f(√2016), and 2017f(√2017) -/
theorem odd_function_inequality (f : OddFunction) :
  f.f 1 < 2016 * f.f (Real.sqrt 2016) ∧
  2016 * f.f (Real.sqrt 2016) < 2017 * f.f (Real.sqrt 2017) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l1782_178253


namespace NUMINAMATH_CALUDE_square_times_abs_fraction_equals_three_l1782_178269

theorem square_times_abs_fraction_equals_three :
  (-3)^2 * |-(1/3)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_times_abs_fraction_equals_three_l1782_178269


namespace NUMINAMATH_CALUDE_max_value_of_f_l1782_178289

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ≤ f 0 ∧ f 0 = 3) ∨
  (a > 2 ∧ ∀ x ∈ Set.Icc 0 a, f x ≤ f a ∧ f a = a^2 - 2*a + 3) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1782_178289


namespace NUMINAMATH_CALUDE_special_trapezoid_area_l1782_178212

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- The measure of the adjacent angles in degrees -/
  adjacent_angle : ℝ
  /-- The measure of the angle between diagonals facing the base in degrees -/
  diag_angle : ℝ

/-- The area of a special trapezoid -/
noncomputable def area (t : SpecialTrapezoid) : ℝ := sorry

/-- Theorem stating the area of a specific trapezoid is 2 -/
theorem special_trapezoid_area :
  ∀ t : SpecialTrapezoid,
    t.shorter_base = 2 ∧
    t.adjacent_angle = 135 ∧
    t.diag_angle = 150 →
    area t = 2 :=
by sorry

end NUMINAMATH_CALUDE_special_trapezoid_area_l1782_178212


namespace NUMINAMATH_CALUDE_salary_change_percentage_l1782_178226

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 96 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l1782_178226


namespace NUMINAMATH_CALUDE_thirty_six_hundredths_decimal_l1782_178246

theorem thirty_six_hundredths_decimal : (36 : ℚ) / 100 = 0.36 := by sorry

end NUMINAMATH_CALUDE_thirty_six_hundredths_decimal_l1782_178246


namespace NUMINAMATH_CALUDE_total_production_all_companies_l1782_178252

/-- Represents the production of cars by a company in different continents -/
structure CarProduction where
  northAmerica : ℕ
  europe : ℕ
  asia : ℕ

/-- Calculates the total production for a company -/
def totalProduction (p : CarProduction) : ℕ :=
  p.northAmerica + p.europe + p.asia

/-- The production data for Car Company A -/
def companyA : CarProduction :=
  { northAmerica := 3884
    europe := 2871
    asia := 1529 }

/-- The production data for Car Company B -/
def companyB : CarProduction :=
  { northAmerica := 4357
    europe := 3690
    asia := 1835 }

/-- The production data for Car Company C -/
def companyC : CarProduction :=
  { northAmerica := 2937
    europe := 4210
    asia := 977 }

/-- Theorem stating that the total production of all companies is 26,290 -/
theorem total_production_all_companies :
  totalProduction companyA + totalProduction companyB + totalProduction companyC = 26290 := by
  sorry

end NUMINAMATH_CALUDE_total_production_all_companies_l1782_178252


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1782_178227

theorem sum_of_coefficients (a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x y : ℝ, (x + y)^4 = a₁*x^4 + a₂*x^3*y + a₃*x^2*y^2 + a₄*x*y^3 + a₅*y^4) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1782_178227


namespace NUMINAMATH_CALUDE_division_of_fractions_l1782_178220

theorem division_of_fractions : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l1782_178220


namespace NUMINAMATH_CALUDE_modified_triangle_invalid_zero_area_l1782_178291

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given side lengths can form a valid triangle -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- The original triangle ABC -/
def originalTriangle : Triangle :=
  { a := 12, b := 7, c := 10 }

/-- The modified triangle with doubled AB and AC -/
def modifiedTriangle : Triangle :=
  { a := 24, b := 14, c := 10 }

/-- Theorem stating that the modified triangle is not valid and has zero area -/
theorem modified_triangle_invalid_zero_area :
  ¬(isValidTriangle modifiedTriangle) ∧ 
  (∃ area : ℝ, area = 0 ∧ area ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_modified_triangle_invalid_zero_area_l1782_178291


namespace NUMINAMATH_CALUDE_village_assistants_selection_l1782_178202

-- Define the total number of college graduates
def total_graduates : ℕ := 10

-- Define the number of people to be selected
def selection_size : ℕ := 3

-- Define a function to calculate the number of ways to select k items from n items
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem village_assistants_selection :
  choose (total_graduates - 1) selection_size -
  choose (total_graduates - 3) selection_size = 49 := by
  sorry

end NUMINAMATH_CALUDE_village_assistants_selection_l1782_178202


namespace NUMINAMATH_CALUDE_tangent_product_identity_l1782_178255

theorem tangent_product_identity : 
  (1 + Real.tan (17 * π / 180)) * 
  (1 + Real.tan (18 * π / 180)) * 
  (1 + Real.tan (27 * π / 180)) * 
  (1 + Real.tan (28 * π / 180)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_identity_l1782_178255


namespace NUMINAMATH_CALUDE_bus_rental_problem_l1782_178254

/-- Represents the capacity of buses --/
structure BusCapacity where
  typeA : ℕ
  typeB : ℕ

/-- Represents the rental plan --/
structure RentalPlan where
  bus65 : ℕ
  bus45 : ℕ
  bus30 : ℕ

/-- The main theorem to prove --/
theorem bus_rental_problem 
  (capacity : BusCapacity) 
  (plan : RentalPlan) : 
  (3 * capacity.typeA + 2 * capacity.typeB = 195) →
  (2 * capacity.typeA + 4 * capacity.typeB = 210) →
  (capacity.typeA = 45) →
  (capacity.typeB = 30) →
  (plan.bus65 = 2) →
  (plan.bus45 = 2) →
  (plan.bus30 = 3) →
  (65 * plan.bus65 + 45 * plan.bus45 + 30 * plan.bus30 = 303 + 7) →
  (plan.bus65 + plan.bus45 + plan.bus30 = 7) →
  True := by
  sorry

end NUMINAMATH_CALUDE_bus_rental_problem_l1782_178254


namespace NUMINAMATH_CALUDE_subtraction_from_percentage_l1782_178247

theorem subtraction_from_percentage (n : ℝ) : n = 70 → (n * 0.5 - 10 = 25) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_from_percentage_l1782_178247


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_90_and_m_l1782_178265

theorem greatest_common_divisor_of_90_and_m (m : ℕ) 
  (h1 : ∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
    (∀ (d : ℕ), d ∣ 90 ∧ d ∣ m ↔ d = d1 ∨ d = d2 ∨ d = d3)) :
  ∃ (d : ℕ), d ∣ 90 ∧ d ∣ m ∧ d = 9 ∧ 
    ∀ (x : ℕ), x ∣ 90 ∧ x ∣ m → x ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_90_and_m_l1782_178265


namespace NUMINAMATH_CALUDE_x0_value_l1782_178292

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem x0_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2) → x₀ = exp 1 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l1782_178292


namespace NUMINAMATH_CALUDE_max_value_implies_a_l1782_178216

def f (a x : ℝ) : ℝ := a * x^3 + 2 * a * x + 1

theorem max_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧ 
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = 1/4 ∨ a = -1/11 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l1782_178216


namespace NUMINAMATH_CALUDE_gcd_7654321_6543210_l1782_178285

theorem gcd_7654321_6543210 : Nat.gcd 7654321 6543210 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7654321_6543210_l1782_178285


namespace NUMINAMATH_CALUDE_no_perfect_square_n_n_plus_one_l1782_178296

theorem no_perfect_square_n_n_plus_one (n : ℕ) (hn : n > 0) : 
  ¬∃ (k : ℕ), n * (n + 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_n_n_plus_one_l1782_178296


namespace NUMINAMATH_CALUDE_equation_solution_l1782_178237

theorem equation_solution : ∃ (y₁ y₂ y₃ : ℂ),
  y₁ = -Real.sqrt 3 ∧
  y₂ = -Real.sqrt 3 + Complex.I ∧
  y₃ = -Real.sqrt 3 - Complex.I ∧
  (∀ y : ℂ, (y^3 + 3*y^2*(Real.sqrt 3) + 9*y + 3*(Real.sqrt 3)) + (y + Real.sqrt 3) = 0 ↔ 
    y = y₁ ∨ y = y₂ ∨ y = y₃) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1782_178237


namespace NUMINAMATH_CALUDE_perimeter_of_rearranged_rectangles_l1782_178290

/-- The perimeter of a shape formed by rearranging two equal rectangles cut from a square --/
theorem perimeter_of_rearranged_rectangles (square_side : ℝ) : square_side = 100 → 500 = 3 * square_side + 4 * (square_side / 2) := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_rearranged_rectangles_l1782_178290


namespace NUMINAMATH_CALUDE_range_of_f_l1782_178273

noncomputable def odot (a b : ℝ) : ℝ := if a ≤ b then a else b

noncomputable def f (x : ℝ) : ℝ := odot (2^x) (2^(-x))

theorem range_of_f :
  (∀ y, y ∈ Set.range f → 0 < y ∧ y ≤ 1) ∧
  (∀ y, 0 < y ∧ y ≤ 1 → ∃ x, f x = y) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1782_178273


namespace NUMINAMATH_CALUDE_freds_allowance_l1782_178211

def weekly_allowance (A x y : ℝ) : Prop :=
  -- Fred spent half of his allowance on movie tickets
  let movie_cost := A / 2
  -- Lunch cost y dollars less than the cost of the tickets
  let lunch_cost := x - y
  -- He earned 6 dollars from washing the car and 5 dollars from mowing the lawn
  let earned := 6 + 5
  -- At the end of the day, he had 20 dollars
  movie_cost + lunch_cost + earned + (A - movie_cost - lunch_cost) = 20

theorem freds_allowance :
  ∃ A x y : ℝ, weekly_allowance A x y ∧ A = 9 := by sorry

end NUMINAMATH_CALUDE_freds_allowance_l1782_178211


namespace NUMINAMATH_CALUDE_only_non_algorithm_l1782_178251

/-- A process is a description of a task or method. -/
structure Process where
  description : String

/-- An algorithm is a process that has a sequence of defined steps. -/
structure Algorithm extends Process where
  has_defined_steps : Bool

/-- The property of having defined steps for a process. -/
def has_defined_steps (p : Process) : Prop :=
  ∃ (a : Algorithm), a.description = p.description

/-- The list of processes to be evaluated. -/
def processes : List Process :=
  [{ description := "The process of solving the equation 2x-6=0 involves moving terms and making the coefficient 1" },
   { description := "To get from Jinan to Vancouver, one must first take a train to Beijing, then transfer to a plane" },
   { description := "Solving the equation 2x^2+x-1=0" },
   { description := "Using the formula S=πr^2 to calculate the area of a circle with radius 3 involves computing π×3^2" }]

/-- The theorem stating that "Solving the equation 2x^2+x-1=0" is the only process without defined steps. -/
theorem only_non_algorithm :
  ∃! (p : Process), p ∈ processes ∧ ¬(has_defined_steps p) ∧
    p.description = "Solving the equation 2x^2+x-1=0" :=
  sorry

end NUMINAMATH_CALUDE_only_non_algorithm_l1782_178251


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l1782_178295

theorem sum_first_150_remainder (n : Nat) (h : n = 150) :
  (List.range n).sum % 8000 = 3325 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l1782_178295


namespace NUMINAMATH_CALUDE_slope_intercept_product_specific_line_l1782_178284

/-- A line in a cartesian plane. -/
structure Line where
  /-- The slope of the line. -/
  slope : ℝ
  /-- The y-intercept of the line. -/
  y_intercept : ℝ

/-- The product of the slope and y-intercept of a line. -/
def slope_intercept_product (l : Line) : ℝ := l.slope * l.y_intercept

/-- Theorem: For a line with y-intercept -3 and slope 3, the product of its slope and y-intercept is -9. -/
theorem slope_intercept_product_specific_line :
  ∃ (l : Line), l.y_intercept = -3 ∧ l.slope = 3 ∧ slope_intercept_product l = -9 := by
  sorry

end NUMINAMATH_CALUDE_slope_intercept_product_specific_line_l1782_178284


namespace NUMINAMATH_CALUDE_kendra_cookie_theorem_l1782_178241

/-- Proves that each family member eats 38 chocolate chips given the conditions of Kendra's cookie baking scenario. -/
theorem kendra_cookie_theorem (
  family_members : ℕ)
  (choc_chip_batches : ℕ)
  (double_choc_chip_batches : ℕ)
  (cookies_per_choc_chip_batch : ℕ)
  (chips_per_choc_chip_cookie : ℕ)
  (cookies_per_double_choc_chip_batch : ℕ)
  (chips_per_double_choc_chip_cookie : ℕ)
  (h1 : family_members = 4)
  (h2 : choc_chip_batches = 3)
  (h3 : double_choc_chip_batches = 2)
  (h4 : cookies_per_choc_chip_batch = 12)
  (h5 : chips_per_choc_chip_cookie = 2)
  (h6 : cookies_per_double_choc_chip_batch = 10)
  (h7 : chips_per_double_choc_chip_cookie = 4)
  : (choc_chip_batches * cookies_per_choc_chip_batch * chips_per_choc_chip_cookie +
     double_choc_chip_batches * cookies_per_double_choc_chip_batch * chips_per_double_choc_chip_cookie) / family_members = 38 := by
  sorry

end NUMINAMATH_CALUDE_kendra_cookie_theorem_l1782_178241


namespace NUMINAMATH_CALUDE_distinct_terms_expansion_l1782_178257

/-- The number of distinct terms in the expansion of (a+b+c)(x+y+z+w+t) -/
def distinct_terms (a b c x y z w t : ℝ) : ℕ :=
  3 * 5

theorem distinct_terms_expansion (a b c x y z w t : ℝ) 
  (h_diff : a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ t ∧ 
            y ≠ z ∧ y ≠ w ∧ y ≠ t ∧ z ≠ w ∧ z ≠ t ∧ w ≠ t) : 
  distinct_terms a b c x y z w t = 15 := by
  sorry

end NUMINAMATH_CALUDE_distinct_terms_expansion_l1782_178257


namespace NUMINAMATH_CALUDE_range_of_a_l1782_178262

open Set Real

theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 3}
  let B : Set ℝ := Ioi 5
  (A ∩ B = ∅) → a ∈ Iic 2 ∪ Ici 3 := by
sorry


end NUMINAMATH_CALUDE_range_of_a_l1782_178262


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l1782_178201

theorem hemisphere_surface_area (base_area : ℝ) (Q : ℝ) : 
  base_area = 3 →
  Q = (2 * Real.pi * (Real.sqrt (3 / Real.pi))^2) + base_area →
  Q = 9 := by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l1782_178201


namespace NUMINAMATH_CALUDE_hotdog_distribution_l1782_178215

theorem hotdog_distribution (E : ℚ) 
  (total_hotdogs : E + E + 2*E + 3*E = 14) : E = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotdog_distribution_l1782_178215


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l1782_178222

theorem min_value_squared_sum (x y : ℝ) (h : x * y = 1) :
  x^2 + 4*y^2 ≥ 4 ∧ ∃ (a b : ℝ), a * b = 1 ∧ a^2 + 4*b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l1782_178222


namespace NUMINAMATH_CALUDE_balls_in_boxes_l1782_178218

def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (λ i => (Nat.choose n i) * (k ^ (n - i)))

theorem balls_in_boxes : distribute_balls 6 2 = 665 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l1782_178218


namespace NUMINAMATH_CALUDE_min_c_value_l1782_178230

theorem min_c_value (a b c : ℕ+) (h1 : a < b) (h2 : b < c) (h3 : b = c - 1)
  (h4 : ∃! p : ℝ × ℝ, p.1^2 + p.2 = 2003 ∧ 
    p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - c.val|) :
  c.val ≥ 1006 ∧ ∃ (a' b' : ℕ+), a' < b' ∧ b' < 1006 ∧ b' = 1005 ∧
    ∃! p : ℝ × ℝ, p.1^2 + p.2 = 2003 ∧ 
      p.2 = |p.1 - a'.val| + |p.1 - b'.val| + |p.1 - 1006| := by
  sorry

end NUMINAMATH_CALUDE_min_c_value_l1782_178230


namespace NUMINAMATH_CALUDE_chloe_final_score_l1782_178256

/-- Chloe's points at the end of a trivia game -/
theorem chloe_final_score (first_round second_round last_round : Int) 
  (h1 : first_round = 40)
  (h2 : second_round = 50)
  (h3 : last_round = -4) :
  first_round + second_round + last_round = 86 := by
  sorry

end NUMINAMATH_CALUDE_chloe_final_score_l1782_178256


namespace NUMINAMATH_CALUDE_a_range_characterization_l1782_178298

-- Define the function p
def p (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

-- Define the monotonicity condition for p
def p_monotone (a : ℝ) : Prop :=
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < -2 → p a x₁ < p a x₂) ∧
  (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → p a x₁ < p a x₂)

-- Define the integral function
def integral (x : ℝ) : ℝ := x^2 - 2*x

-- Define the condition for q
def q (a : ℝ) : Prop := ∀ x, integral x > a

-- Define the range of a
def a_range (a : ℝ) : Prop := a < -2 ∨ (-1 ≤ a ∧ a ≤ 2)

-- Theorem statement
theorem a_range_characterization :
  ∀ a : ℝ, (p_monotone a ∧ ¬q a) ∨ (¬p_monotone a ∧ q a) ↔ a_range a :=
sorry

end NUMINAMATH_CALUDE_a_range_characterization_l1782_178298


namespace NUMINAMATH_CALUDE_common_chord_length_l1782_178288

theorem common_chord_length (r : ℝ) (h : r = 12) :
  let chord_length := 2 * (r * Real.sqrt 3)
  chord_length = 12 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_common_chord_length_l1782_178288


namespace NUMINAMATH_CALUDE_expand_polynomial_l1782_178245

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 5 * x + 7) = 4 * x^3 + 7 * x^2 - 8 * x + 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1782_178245


namespace NUMINAMATH_CALUDE_store_earnings_proof_l1782_178271

def store_earnings : ℕ := by
  let graphics_cards := 10 * 600
  let hard_drives := 14 * 80
  let cpus := 8 * 200
  let ram := 4 * 60
  let power_supply_units := 12 * 90
  let monitors := 6 * 250
  let keyboards := 18 * 40
  let mice := 24 * 20
  exact graphics_cards + hard_drives + cpus + ram + power_supply_units + monitors + keyboards + mice

theorem store_earnings_proof : store_earnings = 12740 := by
  sorry

end NUMINAMATH_CALUDE_store_earnings_proof_l1782_178271


namespace NUMINAMATH_CALUDE_lassie_bones_l1782_178217

theorem lassie_bones (initial_bones : ℕ) : 
  (initial_bones / 2 + 10 = 35) → initial_bones = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_lassie_bones_l1782_178217


namespace NUMINAMATH_CALUDE_translation_theorem_l1782_178266

/-- Represents a quadratic function of the form y = a(x-h)^2 + k --/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Translates a quadratic function horizontally --/
def translate (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h - d, k := f.k }

/-- The initial quadratic function y = 3(x-2)^2 + 1 --/
def initial_function : QuadraticFunction :=
  { a := 3, h := 2, k := 1 }

/-- Theorem stating that translating the initial function 2 units right then 2 units left
    results in y = 3x^2 + 3 --/
theorem translation_theorem :
  let f1 := translate initial_function (-2)
  let f2 := translate f1 2
  f2.a * (X - f2.h)^2 + f2.k = 3 * X^2 + 3 := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l1782_178266


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1782_178287

/-- Calculates the average speed of a trip given the following conditions:
  * The trip lasts for 12 hours
  * The car travels at 45 mph for the first 4 hours
  * The car travels at 75 mph for the remaining hours
-/
theorem average_speed_calculation (total_time : ℝ) (initial_speed : ℝ) (initial_duration : ℝ) (final_speed : ℝ) :
  total_time = 12 →
  initial_speed = 45 →
  initial_duration = 4 →
  final_speed = 75 →
  (initial_speed * initial_duration + final_speed * (total_time - initial_duration)) / total_time = 65 := by
  sorry

#check average_speed_calculation

end NUMINAMATH_CALUDE_average_speed_calculation_l1782_178287


namespace NUMINAMATH_CALUDE_fraction_identification_l1782_178219

-- Define what a fraction is
def is_fraction (x : ℚ) : Prop := ∃ (n d : ℤ), d ≠ 0 ∧ x = n / d

-- Define the given expressions
def expr1 (a : ℚ) : ℚ := 2 / a
def expr2 (a : ℚ) : ℚ := 2 * a / 3
def expr3 (b : ℚ) : ℚ := -b / 2
def expr4 (a : ℚ) : ℚ := (3 * a + 1) / 2

-- State the theorem
theorem fraction_identification (a b : ℚ) (ha : a ≠ 0) : 
  is_fraction (expr1 a) ∧ 
  ¬is_fraction (expr2 a) ∧ 
  ¬is_fraction (expr3 b) ∧ 
  ¬is_fraction (expr4 a) :=
sorry

end NUMINAMATH_CALUDE_fraction_identification_l1782_178219


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1782_178278

theorem complex_fraction_evaluation : 
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1782_178278


namespace NUMINAMATH_CALUDE_adams_house_number_range_l1782_178243

/-- Represents a range of house numbers -/
structure Range where
  lower : Nat
  upper : Nat
  valid : lower ≤ upper

/-- Checks if two ranges overlap -/
def overlaps (r1 r2 : Range) : Prop :=
  (r1.lower ≤ r2.upper ∧ r2.lower ≤ r1.upper) ∨
  (r2.lower ≤ r1.upper ∧ r1.lower ≤ r2.upper)

/-- The given ranges -/
def rangeA : Range := ⟨123, 213, by sorry⟩
def rangeB : Range := ⟨132, 231, by sorry⟩
def rangeC : Range := ⟨123, 312, by sorry⟩
def rangeD : Range := ⟨231, 312, by sorry⟩
def rangeE : Range := ⟨312, 321, by sorry⟩

/-- All ranges except E -/
def otherRanges : List Range := [rangeA, rangeB, rangeC, rangeD]

theorem adams_house_number_range :
  (∀ r ∈ otherRanges, ∃ r' ∈ otherRanges, r ≠ r' ∧ overlaps r r') ∧
  (∀ r ∈ otherRanges, ¬overlaps r rangeE) :=
by sorry

end NUMINAMATH_CALUDE_adams_house_number_range_l1782_178243


namespace NUMINAMATH_CALUDE_rope_length_ratio_l1782_178279

def joeys_rope_length : ℕ := 56
def chads_rope_length : ℕ := 21

theorem rope_length_ratio : 
  (joeys_rope_length : ℚ) / (chads_rope_length : ℚ) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_ratio_l1782_178279


namespace NUMINAMATH_CALUDE_factoring_equation_l1782_178214

theorem factoring_equation (m : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + m * x + 1 = (2 * x - 1)^2) → 
  ∃ f : ℝ → ℝ, ∀ x : ℝ, 4 * x^2 + m * x + 1 = f x * f x :=
by sorry

end NUMINAMATH_CALUDE_factoring_equation_l1782_178214


namespace NUMINAMATH_CALUDE_fruit_prices_l1782_178210

/-- Fruit prices problem -/
theorem fruit_prices (total_cost apple_cost orange_cost banana_cost : ℚ) : 
  total_cost = 7.84 ∧ 
  orange_cost = apple_cost + 0.28 ∧ 
  banana_cost = apple_cost - 0.15 ∧ 
  3 * apple_cost + 7 * orange_cost + 5 * banana_cost = total_cost →
  apple_cost = 0.442 ∧ orange_cost = 0.722 ∧ banana_cost = 0.292 := by
sorry

#eval (0.442 : ℚ) + 0.28 -- Should output 0.722
#eval (0.442 : ℚ) - 0.15 -- Should output 0.292
#eval 3 * (0.442 : ℚ) + 7 * 0.722 + 5 * 0.292 -- Should output 7.84

end NUMINAMATH_CALUDE_fruit_prices_l1782_178210


namespace NUMINAMATH_CALUDE_yellow_crayon_count_l1782_178283

/-- Given the number of red, blue, and yellow crayons with specific relationships,
    prove that the number of yellow crayons is 32. -/
theorem yellow_crayon_count :
  ∀ (red blue yellow : ℕ),
  red = 14 →
  blue = red + 5 →
  yellow = 2 * blue - 6 →
  yellow = 32 := by
sorry

end NUMINAMATH_CALUDE_yellow_crayon_count_l1782_178283


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l1782_178293

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 20 + 1) +
  1 / (Real.log 4 / Real.log 15 + 1) +
  1 / (Real.log 7 / Real.log 12 + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l1782_178293
