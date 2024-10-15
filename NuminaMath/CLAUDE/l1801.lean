import Mathlib

namespace NUMINAMATH_CALUDE_girls_in_class_l1801_180141

theorem girls_in_class (total : ℕ) (difference : ℕ) : 
  total = 63 → difference = 7 → (total + difference) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_l1801_180141


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1801_180126

theorem inequality_and_equality_conditions (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hnz : x > 0 ∨ y > 0 ∨ z > 0) : 
  (2*x^2 - x + y + z)/(x + y^2 + z^2) + 
  (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
  (2*z^2 + x + y - z)/(x^2 + y^2 + z) ≥ 3 ∧ 
  ((2*x^2 - x + y + z)/(x + y^2 + z^2) + 
   (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
   (2*z^2 + x + y - z)/(x^2 + y^2 + z) = 3 ↔ 
   (∃ t : ℝ, t > 0 ∧ x = t ∧ y = t ∧ z = t) ∨ 
   (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = t ∧ y = t ∧ z = 1 - t)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1801_180126


namespace NUMINAMATH_CALUDE_sin_780_degrees_l1801_180161

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_780_degrees_l1801_180161


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l1801_180122

theorem not_p_necessary_not_sufficient_for_not_q :
  ∃ (x : ℝ), (¬(x^2 < 1) → ¬(x < 1)) ∧
  ∃ (y : ℝ), ¬(y < 1) ∧ ¬¬(y^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l1801_180122


namespace NUMINAMATH_CALUDE_exposed_sides_count_l1801_180123

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : Nat
  sides_positive : sides > 0

/-- The sequence of regular polygons in the construction. -/
def polygon_sequence : List RegularPolygon :=
  [⟨3, by norm_num⟩, ⟨4, by norm_num⟩, ⟨5, by norm_num⟩, 
   ⟨6, by norm_num⟩, ⟨7, by norm_num⟩, ⟨8, by norm_num⟩, ⟨9, by norm_num⟩]

/-- Calculates the number of exposed sides for a given polygon in the sequence. -/
def exposed_sides (p : RegularPolygon) (index : Nat) : Nat :=
  if index = 0 ∨ index = 6 then p.sides - 1 else p.sides - 2

/-- The total number of exposed sides in the polygon sequence. -/
def total_exposed_sides : Nat :=
  (List.zipWith exposed_sides polygon_sequence (List.range 7)).sum

theorem exposed_sides_count :
  total_exposed_sides = 30 := by
  sorry

end NUMINAMATH_CALUDE_exposed_sides_count_l1801_180123


namespace NUMINAMATH_CALUDE_first_term_of_constant_ratio_l1801_180171

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (h : d = 5) :
  (∃ (c : ℚ), ∀ (n : ℕ), n > 0 → 
    arithmetic_sum a d (5 * n) / arithmetic_sum a d n = c) →
  a = 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_first_term_of_constant_ratio_l1801_180171


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1801_180169

theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) :
  gain = 45 →
  gain_percentage = 30 →
  ∃ (cost_price : ℝ) (selling_price : ℝ),
    gain = (gain_percentage / 100) * cost_price ∧
    selling_price = cost_price + gain ∧
    selling_price = 195 :=
by sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1801_180169


namespace NUMINAMATH_CALUDE_probability_square_divisor_15_factorial_l1801_180136

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of positive integer divisors of n that are perfect squares -/
def num_square_divisors (n : ℕ) : ℕ := sorry

/-- The total number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Two natural numbers are coprime -/
def coprime (a b : ℕ) : Prop := sorry

theorem probability_square_divisor_15_factorial :
  ∃ m n : ℕ, 
    coprime m n ∧ 
    (m : ℚ) / n = (num_square_divisors (factorial 15) : ℚ) / (num_divisors (factorial 15)) ∧
    m = 1 ∧ n = 84 := by
  sorry

end NUMINAMATH_CALUDE_probability_square_divisor_15_factorial_l1801_180136


namespace NUMINAMATH_CALUDE_tire_usage_calculation_l1801_180188

/- Define the problem parameters -/
def total_miles : ℕ := 42000
def total_tires : ℕ := 7
def tires_used_simultaneously : ℕ := 6

/- Theorem statement -/
theorem tire_usage_calculation :
  let total_tire_miles : ℕ := total_miles * tires_used_simultaneously
  let miles_per_tire : ℕ := total_tire_miles / total_tires
  miles_per_tire = 36000 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_calculation_l1801_180188


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1801_180159

/-- The area of a square with diagonal length 20 is 200 -/
theorem square_area_from_diagonal : 
  ∀ s : ℝ, s > 0 → s * s * 2 = 20 * 20 → s * s = 200 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1801_180159


namespace NUMINAMATH_CALUDE_function_value_range_l1801_180120

theorem function_value_range :
  ∀ x : ℝ, -2 ≤ Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 - 1 ∧
           Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 - 1 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_range_l1801_180120


namespace NUMINAMATH_CALUDE_apollonius_circle_symmetric_x_axis_l1801_180146

/-- Apollonius Circle -/
def ApolloniusCircle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (x + 1)^2 + y^2 = a^2 * ((x - 1)^2 + y^2)}

/-- Symmetry about x-axis -/
def SymmetricAboutXAxis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ x y, (x, y) ∈ S ↔ (x, -y) ∈ S

theorem apollonius_circle_symmetric_x_axis (a : ℝ) (ha : a > 1) :
  SymmetricAboutXAxis (ApolloniusCircle a) := by
  sorry

end NUMINAMATH_CALUDE_apollonius_circle_symmetric_x_axis_l1801_180146


namespace NUMINAMATH_CALUDE_exists_x_squared_sum_l1801_180137

theorem exists_x_squared_sum : ∃ x : ℕ, 106 * 106 + x * x = 19872 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_squared_sum_l1801_180137


namespace NUMINAMATH_CALUDE_nested_circles_radius_l1801_180185

theorem nested_circles_radius (B₁ B₃ : ℝ) : 
  B₁ > 0 →
  B₃ > 0 →
  (B₁ + B₃ = π * 6^2) →
  (B₃ - B₁ = (B₁ + B₃) - B₁) →
  (B₁ = π * (3 * Real.sqrt 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_nested_circles_radius_l1801_180185


namespace NUMINAMATH_CALUDE_hcf_problem_l1801_180138

def is_hcf (h : ℕ) (a b : ℕ) : Prop :=
  h ∣ a ∧ h ∣ b ∧ ∀ d : ℕ, d ∣ a → d ∣ b → d ≤ h

def is_lcm (l : ℕ) (a b : ℕ) : Prop :=
  a ∣ l ∧ b ∣ l ∧ ∀ m : ℕ, a ∣ m → b ∣ m → l ≤ m

theorem hcf_problem (a b : ℕ) (h : ℕ) :
  a = 345 →
  (∃ l : ℕ, is_lcm l a b ∧ l = h * 13 * 15) →
  is_hcf h a b →
  h = 15 := by sorry

end NUMINAMATH_CALUDE_hcf_problem_l1801_180138


namespace NUMINAMATH_CALUDE_min_value_abs_diff_l1801_180111

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the theorem
theorem min_value_abs_diff (x y : ℝ) :
  log 4 (x + 2*y) + log 1 (x - 2*y) = 1 ∧ 
  x + 2*y > 0 ∧ 
  x - 2*y > 0 →
  ∃ (min_val : ℝ), min_val = Real.sqrt 3 ∧ ∀ (a b : ℝ), 
    (log 4 (a + 2*b) + log 1 (a - 2*b) = 1 ∧ a + 2*b > 0 ∧ a - 2*b > 0) →
    |a| - |b| ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_abs_diff_l1801_180111


namespace NUMINAMATH_CALUDE_trapezoid_longer_base_l1801_180127

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  midline_segment : ℝ
  height : ℝ

/-- Theorem: The longer base of a trapezoid with specific properties is 90 -/
theorem trapezoid_longer_base 
  (t : Trapezoid) 
  (h1 : t.shorter_base = 80) 
  (h2 : t.midline_segment = 5) 
  (h3 : t.height = 3 * t.midline_segment) 
  (h4 : t.midline_segment = (t.longer_base - t.shorter_base) / 2) : 
  t.longer_base = 90 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_longer_base_l1801_180127


namespace NUMINAMATH_CALUDE_cab_driver_income_l1801_180195

/-- The cab driver's income problem -/
theorem cab_driver_income (day1 day2 day3 day5 : ℕ) (average : ℕ) 
  (h1 : day1 = 400)
  (h2 : day2 = 250)
  (h3 : day3 = 650)
  (h5 : day5 = 500)
  (h_avg : average = 440)
  (h_total : day1 + day2 + day3 + day5 + (5 * average - (day1 + day2 + day3 + day5)) = 5 * average) :
  5 * average - (day1 + day2 + day3 + day5) = 400 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l1801_180195


namespace NUMINAMATH_CALUDE_fraction_division_equality_l1801_180103

theorem fraction_division_equality : (2 / 5) / (3 / 7) = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l1801_180103


namespace NUMINAMATH_CALUDE_symmetry_properties_l1801_180180

/-- Two rational numbers are symmetric about a point with a given symmetric radius. -/
def symmetric (m n p r : ℚ) : Prop :=
  m ≠ n ∧ m ≠ p ∧ n ≠ p ∧ |m - p| = r ∧ |n - p| = r

theorem symmetry_properties :
  (∃ x r : ℚ, symmetric 3 x 1 r ∧ x = -1 ∧ r = 2) ∧
  (∃ a b r : ℚ, symmetric a b 2 r ∧ |a| = 2 * |b| ∧ (r = 2/3 ∨ r = 6)) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_properties_l1801_180180


namespace NUMINAMATH_CALUDE_no_member_divisible_by_four_l1801_180198

-- Define the set T
def T : Set ℤ := {s | ∃ n : ℤ, s = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

-- Theorem statement
theorem no_member_divisible_by_four : ∀ s ∈ T, ¬(4 ∣ s) := by
  sorry

end NUMINAMATH_CALUDE_no_member_divisible_by_four_l1801_180198


namespace NUMINAMATH_CALUDE_water_needed_for_noah_l1801_180129

/-- Represents the recipe ratios and quantities for Noah's orange juice --/
structure OrangeJuiceRecipe where
  orange : ℝ  -- Amount of orange concentrate
  sugar : ℝ   -- Amount of sugar
  water : ℝ   -- Amount of water
  sugar_to_orange_ratio : sugar = 3 * orange
  water_to_sugar_ratio : water = 3 * sugar

/-- Theorem: Given Noah's recipe ratios and 4 cups of orange concentrate, 36 cups of water are needed --/
theorem water_needed_for_noah's_recipe : 
  ∀ (recipe : OrangeJuiceRecipe), 
  recipe.orange = 4 → 
  recipe.water = 36 := by
sorry


end NUMINAMATH_CALUDE_water_needed_for_noah_l1801_180129


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1801_180135

theorem largest_integer_satisfying_inequality :
  ∀ y : ℤ, y ≤ 5 ↔ y / 3 + 5 / 3 < 11 / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1801_180135


namespace NUMINAMATH_CALUDE_caravan_keepers_caravan_keepers_proof_l1801_180140

theorem caravan_keepers : ℕ → Prop :=
  fun k =>
    let hens : ℕ := 50
    let goats : ℕ := 45
    let camels : ℕ := 8
    let total_heads : ℕ := hens + goats + camels + k
    let total_feet : ℕ := hens * 2 + goats * 4 + camels * 4 + k * 2
    total_feet = total_heads + 224 → k = 15

-- The proof goes here
theorem caravan_keepers_proof : ∃ k : ℕ, caravan_keepers k :=
  sorry

end NUMINAMATH_CALUDE_caravan_keepers_caravan_keepers_proof_l1801_180140


namespace NUMINAMATH_CALUDE_only_solution_is_48_l1801_180158

/-- Product of digits function -/
def p (A : ℕ) : ℕ :=
  sorry

/-- Theorem: 48 is the only natural number satisfying A = 1.5 * p(A) -/
theorem only_solution_is_48 :
  ∀ A : ℕ, A = (3/2 : ℚ) * p A ↔ A = 48 :=
by sorry

end NUMINAMATH_CALUDE_only_solution_is_48_l1801_180158


namespace NUMINAMATH_CALUDE_marbles_distribution_l1801_180187

theorem marbles_distribution (x : ℚ) 
  (h1 : x > 0) 
  (h2 : (4 * x + 2) + (2 * x + 1) + 3 * x = 62) : 
  (4 * x + 2 = 254 / 9) ∧ (2 * x + 1 = 127 / 9) ∧ (3 * x = 177 / 9) :=
by sorry

end NUMINAMATH_CALUDE_marbles_distribution_l1801_180187


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l1801_180134

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l1801_180134


namespace NUMINAMATH_CALUDE_ice_cubes_per_tray_l1801_180148

theorem ice_cubes_per_tray (total_ice_cubes : ℕ) (number_of_trays : ℕ) 
  (h1 : total_ice_cubes = 72) 
  (h2 : number_of_trays = 8) 
  (h3 : total_ice_cubes % number_of_trays = 0) : 
  total_ice_cubes / number_of_trays = 9 := by
  sorry

end NUMINAMATH_CALUDE_ice_cubes_per_tray_l1801_180148


namespace NUMINAMATH_CALUDE_twelve_sided_die_expected_value_l1801_180168

/-- The number of sides on the die -/
def n : ℕ := 12

/-- The expected value of rolling an n-sided die with faces numbered from 1 to n -/
def expected_value (n : ℕ) : ℚ :=
  (n + 1 : ℚ) / 2

/-- Theorem: The expected value of rolling a twelve-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem twelve_sided_die_expected_value :
  expected_value n = 13/2 := by sorry

end NUMINAMATH_CALUDE_twelve_sided_die_expected_value_l1801_180168


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l1801_180167

/-- The area of a regular hexagon inscribed in a circle with area 324π -/
theorem inscribed_hexagon_area :
  ∀ (circle_area hexagon_area : ℝ),
  circle_area = 324 * Real.pi →
  hexagon_area = 6 * (((Real.sqrt (circle_area / Real.pi)) ^ 2 * Real.sqrt 3) / 4) →
  hexagon_area = 486 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l1801_180167


namespace NUMINAMATH_CALUDE_garden_fence_posts_l1801_180197

/-- Calculates the minimum number of fence posts needed for a rectangular garden -/
def min_fence_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let wall_side := max length width
  let fenced_perimeter := perimeter - wall_side
  let posts_on_long_side := wall_side / post_spacing + 1
  let posts_on_short_sides := 2 * (fenced_perimeter - wall_side) / post_spacing
  posts_on_long_side + posts_on_short_sides

/-- Proves that for a 50m by 80m garden with 10m post spacing, 17 posts are needed -/
theorem garden_fence_posts :
  min_fence_posts 80 50 10 = 17 := by
  sorry

#eval min_fence_posts 80 50 10

end NUMINAMATH_CALUDE_garden_fence_posts_l1801_180197


namespace NUMINAMATH_CALUDE_no_primes_in_perm_numbers_l1801_180174

/-- A permutation of the digits 1, 2, 3, 4, 5 -/
def Perm5 : Type := Fin 5 → Fin 5

/-- Converts a permutation to a 5-digit number -/
def toNumber (p : Perm5) : ℕ :=
  10000 * (p 0).val + 1000 * (p 1).val + 100 * (p 2).val + 10 * (p 3).val + (p 4).val + 11111

/-- The set of all 5-digit numbers formed by permutations of 1, 2, 3, 4, 5 -/
def PermNumbers : Set ℕ :=
  {n | ∃ p : Perm5, toNumber p = n}

theorem no_primes_in_perm_numbers : ∀ n ∈ PermNumbers, ¬ Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_perm_numbers_l1801_180174


namespace NUMINAMATH_CALUDE_division_sum_theorem_l1801_180181

theorem division_sum_theorem (dividend : Nat) (divisor : Nat) (quotient : Nat) :
  dividend = 82502 →
  divisor ≥ 100 ∧ divisor < 1000 →
  dividend = divisor * quotient →
  divisor + quotient = 723 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l1801_180181


namespace NUMINAMATH_CALUDE_rectangle_count_l1801_180105

/-- The number of different rectangles in a 5x5 grid -/
def num_rectangles : ℕ := 100

/-- The number of rows in the grid -/
def num_rows : ℕ := 5

/-- The number of columns in the grid -/
def num_columns : ℕ := 5

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem rectangle_count :
  num_rectangles = choose_two num_rows * choose_two num_columns :=
sorry

end NUMINAMATH_CALUDE_rectangle_count_l1801_180105


namespace NUMINAMATH_CALUDE_solve_flowers_problem_l1801_180100

def flowers_problem (lilies sunflowers daisies total_flowers : ℕ) : Prop :=
  let other_flowers := lilies + sunflowers + daisies
  let roses := total_flowers - other_flowers
  (lilies = 40) ∧ (sunflowers = 40) ∧ (daisies = 40) ∧ (total_flowers = 160) →
  roses = 40

theorem solve_flowers_problem :
  ∀ (lilies sunflowers daisies total_flowers : ℕ),
  flowers_problem lilies sunflowers daisies total_flowers :=
by
  sorry

end NUMINAMATH_CALUDE_solve_flowers_problem_l1801_180100


namespace NUMINAMATH_CALUDE_product_digit_sum_l1801_180153

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The problem statement -/
theorem product_digit_sum :
  let c : ℕ := 777
  let d : ℕ := 444
  sum_of_digits (7 * c * d) = 27 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l1801_180153


namespace NUMINAMATH_CALUDE_opposite_of_four_l1801_180132

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_four : opposite 4 = -4 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_four_l1801_180132


namespace NUMINAMATH_CALUDE_emily_jumps_in_75_seconds_l1801_180166

/-- Emily's jumping rate in jumps per second -/
def jumping_rate : ℚ := 52 / 60

/-- The number of jumps Emily makes in a given time -/
def jumps (time : ℚ) : ℚ := jumping_rate * time

theorem emily_jumps_in_75_seconds : 
  jumps 75 = 65 := by sorry

end NUMINAMATH_CALUDE_emily_jumps_in_75_seconds_l1801_180166


namespace NUMINAMATH_CALUDE_cube_opposite_face_l1801_180119

structure Cube where
  faces : Finset Char
  adjacent : Char → Finset Char

def is_opposite (c : Cube) (face1 face2 : Char) : Prop :=
  face1 ∈ c.faces ∧ face2 ∈ c.faces ∧ face1 ≠ face2 ∧
  c.adjacent face1 ∩ c.adjacent face2 = ∅

theorem cube_opposite_face (c : Cube) :
  c.faces = {'A', 'B', 'C', 'D', 'E', 'F'} →
  c.adjacent 'E' = {'A', 'B', 'C', 'D'} →
  is_opposite c 'E' 'F' :=
by sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l1801_180119


namespace NUMINAMATH_CALUDE_percentage_not_participating_l1801_180102

theorem percentage_not_participating (total_students : ℕ) (music_and_sports : ℕ) (music_only : ℕ) (sports_only : ℕ) :
  total_students = 50 →
  music_and_sports = 5 →
  music_only = 15 →
  sports_only = 20 →
  (total_students - (music_and_sports + music_only + sports_only)) / total_students * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_participating_l1801_180102


namespace NUMINAMATH_CALUDE_absolute_value_inequality_range_l1801_180155

theorem absolute_value_inequality_range (k : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| < k) ↔ k > -3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_range_l1801_180155


namespace NUMINAMATH_CALUDE_major_axis_length_tangent_ellipse_major_axis_l1801_180156

/-- An ellipse with foci at (4, 1 + 2√3) and (4, 1 - 2√3), tangent to both x and y axes -/
structure TangentEllipse where
  /-- The ellipse is tangent to the x-axis -/
  tangent_x : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_y : Bool
  /-- The x-coordinate of both foci -/
  focus_x : ℝ
  /-- The y-coordinate of the first focus -/
  focus_y1 : ℝ
  /-- The y-coordinate of the second focus -/
  focus_y2 : ℝ
  /-- Ensure the foci are correctly positioned -/
  foci_constraint : focus_x = 4 ∧ focus_y1 = 1 + 2 * Real.sqrt 3 ∧ focus_y2 = 1 - 2 * Real.sqrt 3

/-- The length of the major axis of the ellipse is 2 -/
theorem major_axis_length (e : TangentEllipse) : ℝ :=
  2

/-- The theorem stating that the major axis length of the given ellipse is 2 -/
theorem tangent_ellipse_major_axis (e : TangentEllipse) (h1 : e.tangent_x = true) (h2 : e.tangent_y = true) :
  major_axis_length e = 2 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_tangent_ellipse_major_axis_l1801_180156


namespace NUMINAMATH_CALUDE_nails_needed_l1801_180194

theorem nails_needed (nails_per_plank : ℕ) (num_planks : ℕ) : 
  nails_per_plank = 2 → num_planks = 2 → nails_per_plank * num_planks = 4 := by
  sorry

#check nails_needed

end NUMINAMATH_CALUDE_nails_needed_l1801_180194


namespace NUMINAMATH_CALUDE_clearance_savings_l1801_180142

def coat_price : ℝ := 100
def pants_price : ℝ := 50
def coat_discount : ℝ := 0.30
def pants_discount : ℝ := 0.60

theorem clearance_savings : 
  let total_original := coat_price + pants_price
  let total_savings := coat_price * coat_discount + pants_price * pants_discount
  total_savings / total_original = 0.40 := by sorry

end NUMINAMATH_CALUDE_clearance_savings_l1801_180142


namespace NUMINAMATH_CALUDE_existence_of_sum_equality_l1801_180162

theorem existence_of_sum_equality (n : ℕ) (a : Fin (n + 1) → ℤ)
  (h_n : n > 3)
  (h_a : ∀ i j : Fin (n + 1), i < j → a i < a j)
  (h_lower : a 0 ≥ 1)
  (h_upper : a n ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin (n + 1)),
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧
    k ≠ l ∧ k ≠ m ∧
    l ≠ m ∧
    a i + a j = a k + a l ∧ a k + a l = a m :=
by sorry

end NUMINAMATH_CALUDE_existence_of_sum_equality_l1801_180162


namespace NUMINAMATH_CALUDE_factory_earnings_l1801_180114

/-- Represents a factory with machines producing material -/
structure Factory where
  machines_23h : ℕ  -- Number of machines working 23 hours
  machines_12h : ℕ  -- Number of machines working 12 hours
  production_rate : ℝ  -- Production rate in kg per hour per machine
  price_per_kg : ℝ  -- Selling price per kg of material

/-- Calculates the daily earnings of the factory -/
def daily_earnings (f : Factory) : ℝ :=
  (f.machines_23h * 23 + f.machines_12h * 12) * f.production_rate * f.price_per_kg

/-- Theorem stating that the factory's daily earnings are $8100 -/
theorem factory_earnings :
  let f : Factory := {
    machines_23h := 3,
    machines_12h := 1,
    production_rate := 2,
    price_per_kg := 50
  }
  daily_earnings f = 8100 := by sorry

end NUMINAMATH_CALUDE_factory_earnings_l1801_180114


namespace NUMINAMATH_CALUDE_rhino_horn_segment_area_l1801_180130

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents the "rhino's horn segment" region -/
structure RhinoHornSegment where
  largeCircle : Circle
  smallCircle : Circle
  basePoint : Point
  endPoint : Point

/-- Calculates the area of the "rhino's horn segment" -/
def rhinoHornSegmentArea (r : RhinoHornSegment) : ℝ :=
  sorry

/-- The main theorem stating that the area of the "rhino's horn segment" is 2π -/
theorem rhino_horn_segment_area :
  let r := RhinoHornSegment.mk
    (Circle.mk (Point.mk 0 0) 4)
    (Circle.mk (Point.mk 0 2) 2)
    (Point.mk 0 0)
    (Point.mk 4 0)
  rhinoHornSegmentArea r = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_rhino_horn_segment_area_l1801_180130


namespace NUMINAMATH_CALUDE_rectangle_area_reduction_l1801_180104

theorem rectangle_area_reduction (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧ (l - 1) * w = 24 → l * (w - 1) = 18 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_reduction_l1801_180104


namespace NUMINAMATH_CALUDE_alex_jamie_pairing_probability_l1801_180115

/-- Represents the probability of Alex being paired with Jamie in a class pairing scenario -/
theorem alex_jamie_pairing_probability 
  (total_students : ℕ) 
  (paired_students : ℕ) 
  (h1 : total_students = 50) 
  (h2 : paired_students = 20) 
  (h3 : paired_students < total_students) :
  (1 : ℚ) / (total_students - paired_students - 1 : ℚ) = 1/29 := by
sorry

end NUMINAMATH_CALUDE_alex_jamie_pairing_probability_l1801_180115


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_2005_l1801_180125

def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_m_divisible_by_2005 : 
  ∃ m : ℕ+, (3^100 * (m.val + 1) - 1) % 2005 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_2005_l1801_180125


namespace NUMINAMATH_CALUDE_alternating_sequence_sum_l1801_180112

def sequence_sum (first last : ℤ) (step : ℤ) : ℤ :=
  let n := (last - first) / step + 1
  let sum := (first + last) * n / 2
  if n % 2 = 0 then -sum else sum

theorem alternating_sequence_sum : 
  sequence_sum 2 74 4 = 38 := by sorry

end NUMINAMATH_CALUDE_alternating_sequence_sum_l1801_180112


namespace NUMINAMATH_CALUDE_tangent_line_implies_sum_l1801_180116

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 - x^2 - a*x + b

/-- The derivative of f with respect to x -/
def f' (a x : ℝ) : ℝ := 3*x^2 - 2*x - a

theorem tangent_line_implies_sum (a b : ℝ) :
  f a b 0 = 1 ∧ f' a 0 = 2 → a + b = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_sum_l1801_180116


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_420_l1801_180143

-- Define the sum of divisors function
def sumOfDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

-- State the theorem
theorem sum_of_divisors_of_420 : sumOfDivisors 420 = 1344 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_420_l1801_180143


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l1801_180108

theorem tan_double_angle_special_case (θ : Real) :
  (∃ (x y : Real), y = (1/2) * x ∧ x ≥ 0 ∧ y ≥ 0 ∧ Real.tan θ = y / x) →
  Real.tan (2 * θ) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l1801_180108


namespace NUMINAMATH_CALUDE_cafeteria_apples_theorem_l1801_180178

/-- Given the initial number of apples, the number of pies made, and the number of apples per pie,
    calculate the number of apples handed out to students. -/
def apples_handed_out (initial_apples : ℕ) (pies_made : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - pies_made * apples_per_pie

/-- Theorem stating that for the given problem, 30 apples were handed out to students. -/
theorem cafeteria_apples_theorem :
  apples_handed_out 86 7 8 = 30 := by
  sorry

#eval apples_handed_out 86 7 8

end NUMINAMATH_CALUDE_cafeteria_apples_theorem_l1801_180178


namespace NUMINAMATH_CALUDE_robyn_packs_l1801_180176

def lucy_packs : ℕ := 19
def total_packs : ℕ := 35

theorem robyn_packs : total_packs - lucy_packs = 16 := by sorry

end NUMINAMATH_CALUDE_robyn_packs_l1801_180176


namespace NUMINAMATH_CALUDE_megan_lead_actress_percentage_l1801_180183

def total_plays : ℕ := 100
def not_lead_plays : ℕ := 20

theorem megan_lead_actress_percentage :
  (total_plays - not_lead_plays) * 100 / total_plays = 80 := by
  sorry

end NUMINAMATH_CALUDE_megan_lead_actress_percentage_l1801_180183


namespace NUMINAMATH_CALUDE_condition_for_a_greater_than_b_l1801_180131

-- Define the property of being sufficient but not necessary
def sufficient_but_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

-- Theorem statement
theorem condition_for_a_greater_than_b (a b : ℝ) :
  sufficient_but_not_necessary (a > b + 1) (a > b) := by
  sorry

end NUMINAMATH_CALUDE_condition_for_a_greater_than_b_l1801_180131


namespace NUMINAMATH_CALUDE_power_of_two_equality_l1801_180133

theorem power_of_two_equality (K : ℕ) : 32^5 * 64^2 = 2^K → K = 37 := by
  have h1 : 32 = 2^5 := by sorry
  have h2 : 64 = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l1801_180133


namespace NUMINAMATH_CALUDE_sector_central_angle_l1801_180172

/-- Given a circular sector with perimeter 4 and area 1, 
    its central angle measure is 2 radians. -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) :
  l / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1801_180172


namespace NUMINAMATH_CALUDE_hexagon_diagonals_intersect_l1801_180154

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A hexagon in a 2D plane -/
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A line dividing a side of a triangle into three equal parts -/
def dividingLine (T : Triangle) (vertex : Fin 3) : ℝ × ℝ → ℝ × ℝ → Prop :=
  sorry

/-- The hexagon formed by the dividing lines -/
def formHexagon (T : Triangle) : Hexagon :=
  sorry

/-- The diagonals of a hexagon -/
def diagonals (H : Hexagon) : List (ℝ × ℝ → ℝ × ℝ → Prop) :=
  sorry

/-- The intersection point of lines -/
def intersectionPoint (lines : List (ℝ × ℝ → ℝ × ℝ → Prop)) : Option (ℝ × ℝ) :=
  sorry

/-- Main theorem -/
theorem hexagon_diagonals_intersect (T : Triangle) :
  let H := formHexagon T
  let diag := diagonals H
  ∃ p : ℝ × ℝ, intersectionPoint diag = some p :=
by sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_intersect_l1801_180154


namespace NUMINAMATH_CALUDE_always_two_real_roots_unique_m_value_l1801_180199

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - 4*m*x + 3*m^2 = 0

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ :=
sorry

-- Theorem 2: When m > 0 and the difference between roots is 2, m = 1
theorem unique_m_value (m : ℝ) (h₁ : m > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ x₁ - x₂ = 2) →
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_unique_m_value_l1801_180199


namespace NUMINAMATH_CALUDE_escalator_steps_l1801_180157

/-- The number of steps Al counts walking down the escalator -/
def al_steps : ℕ := 150

/-- The number of steps Bob counts walking up the escalator -/
def bob_steps : ℕ := 75

/-- The ratio of Al's walking speed to Bob's walking speed -/
def speed_ratio : ℕ := 3

/-- The number of steps visible on the escalator at any given time -/
def visible_steps : ℕ := 120

/-- Theorem stating that given the conditions, the number of visible steps on the escalator is 120 -/
theorem escalator_steps : 
  ∀ (al_count bob_count : ℕ) (speed_ratio : ℕ),
    al_count = al_steps →
    bob_count = bob_steps →
    speed_ratio = 3 →
    visible_steps = 120 := by sorry

end NUMINAMATH_CALUDE_escalator_steps_l1801_180157


namespace NUMINAMATH_CALUDE_square_arrangement_sum_l1801_180145

/-- The sum of integers from -12 to 18 inclusive -/
def total_sum : ℤ := 93

/-- The size of the square matrix -/
def matrix_size : ℕ := 6

/-- The common sum for each row, column, and main diagonal -/
def common_sum : ℚ := 15.5

theorem square_arrangement_sum :
  total_sum = matrix_size * (common_sum : ℚ).num / (common_sum : ℚ).den :=
sorry

end NUMINAMATH_CALUDE_square_arrangement_sum_l1801_180145


namespace NUMINAMATH_CALUDE_allen_blocks_count_l1801_180152

/-- The number of blocks for each color -/
def blocks_per_color : ℕ := 7

/-- The number of colors used -/
def number_of_colors : ℕ := 7

/-- The total number of blocks -/
def total_blocks : ℕ := blocks_per_color * number_of_colors

theorem allen_blocks_count : total_blocks = 49 := by
  sorry

end NUMINAMATH_CALUDE_allen_blocks_count_l1801_180152


namespace NUMINAMATH_CALUDE_cake_recipe_ratio_l1801_180179

/-- Given a recipe with 60 eggs and a total of 90 cups of flour and eggs,
    prove that the ratio of cups of flour to eggs is 1:2. -/
theorem cake_recipe_ratio : 
  ∀ (flour eggs : ℕ), 
    eggs = 60 →
    flour + eggs = 90 →
    (flour : ℚ) / (eggs : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cake_recipe_ratio_l1801_180179


namespace NUMINAMATH_CALUDE_split_99_into_four_numbers_l1801_180144

theorem split_99_into_four_numbers : ∃ (a b c d : ℚ),
  a + b + c + d = 99 ∧
  a + 2 = b - 2 ∧
  a + 2 = 2 * c ∧
  a + 2 = d / 2 ∧
  a = 20 ∧ b = 24 ∧ c = 11 ∧ d = 44 := by
  sorry

end NUMINAMATH_CALUDE_split_99_into_four_numbers_l1801_180144


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1801_180182

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

/-- Given two points A(a, 4) and B(-3, b) symmetric with respect to the origin, prove that a + b = -1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin (a, 4) (-3, b)) : 
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1801_180182


namespace NUMINAMATH_CALUDE_isosceles_triangle_smallest_base_l1801_180196

theorem isosceles_triangle_smallest_base 
  (α : ℝ) 
  (q : ℝ) 
  (h_α : 0 < α ∧ α < π) 
  (h_q : q > 0) :
  let base (a : ℝ) := 
    Real.sqrt (q^2 * ((1 - Real.cos α) / 2) + 2 * (1 + Real.cos α) * (a - q/2)^2)
  ∀ a, 0 < a ∧ a < q → base (q/2) ≤ base a :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_smallest_base_l1801_180196


namespace NUMINAMATH_CALUDE_f_2019_l1801_180113

def f : ℕ → ℕ
| x => if x ≤ 2015 then x + 2 else f (x - 5)

theorem f_2019 : f 2019 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_l1801_180113


namespace NUMINAMATH_CALUDE_wire_cutting_l1801_180110

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 21 →
  ratio = 2 / 5 →
  shorter_length + shorter_length / ratio = total_length →
  shorter_length = 6 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l1801_180110


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1801_180107

/-- A geometric sequence with sum of first n terms S_n = 3 * 2^n + m has common ratio 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h : ∀ n, S n = 3 * 2^n + (S 0 - 3)) : 
  ∃ r : ℝ, r = 2 ∧ ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1801_180107


namespace NUMINAMATH_CALUDE_subtraction_of_negative_integers_l1801_180192

theorem subtraction_of_negative_integers : -3 - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_integers_l1801_180192


namespace NUMINAMATH_CALUDE_min_tablets_for_given_box_l1801_180128

/-- Given a box with tablets of two types of medicine, this function calculates
    the minimum number of tablets that must be extracted to guarantee at least
    two tablets of each type. -/
def min_tablets_to_extract (tablets_a tablets_b : ℕ) : ℕ :=
  max ((tablets_b + 1) + 2) ((tablets_a + 1) + 2)

/-- Theorem stating that for a box with 10 tablets of medicine A and 13 tablets
    of medicine B, the minimum number of tablets to extract to guarantee at
    least two of each kind is 15. -/
theorem min_tablets_for_given_box :
  min_tablets_to_extract 10 13 = 15 := by sorry

end NUMINAMATH_CALUDE_min_tablets_for_given_box_l1801_180128


namespace NUMINAMATH_CALUDE_problem_solution_l1801_180177

def f (m : ℝ) (x : ℝ) : ℝ := |x + 3| - m

theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∀ x, f m (x - 3) ≥ 0 ↔ x ≤ -2 ∨ x ≥ 2) →
  m = 2 ∧
  (∃ t : ℝ, ∀ x, ∃ y : ℝ, f 2 y ≥ |2*x - 1| - t^2 + (3/2)*t + 1 ↔ t ≤ 1/2 ∨ t ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1801_180177


namespace NUMINAMATH_CALUDE_smallest_integer_greater_than_root_sum_power_6_l1801_180121

theorem smallest_integer_greater_than_root_sum_power_6 :
  ∃ n : ℕ, n = 970 ∧ (∀ m : ℤ, m > (Real.sqrt 3 + Real.sqrt 2)^6 → m ≥ n) :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_greater_than_root_sum_power_6_l1801_180121


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1801_180139

/-- Given two vectors a and b in R², where a is perpendicular to (a - b),
    prove that the y-coordinate of b must be 3. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 = 1 ∧ a.2 = 2 ∧ b.1 = -1) :
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) → b.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1801_180139


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_plus_inverse_l1801_180147

def z : ℂ := 3 + Complex.I

theorem imaginary_part_of_z_plus_inverse (z : ℂ) (h : z = 3 + Complex.I) :
  Complex.im (z + z⁻¹) = 9 / 10 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_plus_inverse_l1801_180147


namespace NUMINAMATH_CALUDE_bags_given_away_bags_given_away_equals_two_l1801_180191

def initial_purchase : ℕ := 3
def second_purchase : ℕ := 3
def remaining_bags : ℕ := 4

theorem bags_given_away : ℕ := by
  sorry

theorem bags_given_away_equals_two : bags_given_away = 2 := by
  sorry

end NUMINAMATH_CALUDE_bags_given_away_bags_given_away_equals_two_l1801_180191


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1801_180149

/-- Given a polynomial Q with Q(10) = 5 and Q(50) = 15, 
    the remainder when Q is divided by (x - 10)(x - 50) is (1/4)x + 2.5 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 10 = 5) (h2 : Q 50 = 15) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 10) * (x - 50) * R x + 1/4 * x + 5/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1801_180149


namespace NUMINAMATH_CALUDE_living_room_set_cost_l1801_180184

theorem living_room_set_cost (couch_cost sectional_cost other_cost : ℕ)
  (discount_rate : ℚ) (h1 : couch_cost = 2500) (h2 : sectional_cost = 3500)
  (h3 : other_cost = 2000) (h4 : discount_rate = 1/10) :
  (couch_cost + sectional_cost + other_cost) * (1 - discount_rate) = 7200 :=
by sorry

end NUMINAMATH_CALUDE_living_room_set_cost_l1801_180184


namespace NUMINAMATH_CALUDE_multiply_993_879_l1801_180163

theorem multiply_993_879 : 993 * 879 = 872847 := by
  -- Define the method
  let a := 993
  let b := 879
  let n := 7
  
  -- Step 1: Subtract n from b
  let b_minus_n := b - n
  
  -- Step 2: Add n to a
  let a_plus_n := a + n
  
  -- Step 3: Multiply results of steps 1 and 2
  let product_step3 := b_minus_n * a_plus_n
  
  -- Step 4: Calculate the difference
  let diff := a - b_minus_n
  
  -- Step 5: Multiply the difference by n
  let product_step5 := diff * n
  
  -- Step 6: Add results of steps 3 and 5
  let result := product_step3 + product_step5
  
  -- Prove that the result equals 872847
  sorry

end NUMINAMATH_CALUDE_multiply_993_879_l1801_180163


namespace NUMINAMATH_CALUDE_permutations_of_sees_l1801_180151

theorem permutations_of_sees (n : ℕ) (a b : ℕ) (h1 : n = 4) (h2 : a = 2) (h3 : b = 2) :
  (n.factorial) / (a.factorial * b.factorial) = 6 :=
sorry

end NUMINAMATH_CALUDE_permutations_of_sees_l1801_180151


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l1801_180106

theorem sqrt_sum_problem (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l1801_180106


namespace NUMINAMATH_CALUDE_addition_subtraction_ratio_l1801_180173

theorem addition_subtraction_ratio (A B : ℝ) (h : A > 0) (h' : B > 0) (h'' : A / B = 7) : 
  (A + B) / (A - B) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_addition_subtraction_ratio_l1801_180173


namespace NUMINAMATH_CALUDE_ratio_of_x_to_y_l1801_180124

theorem ratio_of_x_to_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 9) (h4 : y = 0.5) :
  x / y = 36 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_to_y_l1801_180124


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1801_180165

theorem sum_of_reciprocals (a b : ℝ) (ha : a^2 - 3*a + 2 = 0) (hb : b^2 - 3*b + 2 = 0) (hab : a ≠ b) :
  1/a + 1/b = 3/2 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1801_180165


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l1801_180193

theorem power_of_power_at_three :
  (3^3)^(3^3) = 27^27 := by sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l1801_180193


namespace NUMINAMATH_CALUDE_min_xyz_l1801_180150

theorem min_xyz (x y z : ℝ) (h1 : x * y + 2 * z = 1) (h2 : x^2 + y^2 + z^2 = 10) : 
  ∀ (a b c : ℝ), a * b * c ≥ -28 → x * y * z ≥ -28 :=
by sorry

end NUMINAMATH_CALUDE_min_xyz_l1801_180150


namespace NUMINAMATH_CALUDE_distance_ratio_is_one_to_one_l1801_180109

def walking_speed : ℝ := 4
def running_speed : ℝ := 8
def total_time : ℝ := 1.5
def total_distance : ℝ := 8

theorem distance_ratio_is_one_to_one :
  ∃ (d_w d_r : ℝ),
    d_w / walking_speed + d_r / running_speed = total_time ∧
    d_w + d_r = total_distance ∧
    d_w / d_r = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_ratio_is_one_to_one_l1801_180109


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l1801_180118

def initial_bottle_caps (current : ℕ) (lost : ℕ) : ℕ := current + lost

theorem danny_bottle_caps : initial_bottle_caps 25 66 = 91 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l1801_180118


namespace NUMINAMATH_CALUDE_min_value_expression_l1801_180170

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 - 1) * (1 / b^2 - 1) ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1801_180170


namespace NUMINAMATH_CALUDE_correct_calculation_l1801_180160

theorem correct_calculation (a b : ℝ) : (a * b)^2 / (-a * b) = -a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1801_180160


namespace NUMINAMATH_CALUDE_similar_polygon_area_sum_l1801_180164

/-- Given two similar polygons, constructs a third similar polygon with area equal to the sum of the given polygons' areas -/
theorem similar_polygon_area_sum 
  (t₁ t₂ : ℝ) 
  (a₁ a₂ : ℝ) 
  (h_positive : t₁ > 0 ∧ t₂ > 0 ∧ a₁ > 0 ∧ a₂ > 0)
  (h_similar : t₁ / (a₁^2) = t₂ / (a₂^2)) :
  let b := Real.sqrt (a₁^2 + a₂^2)
  let t₃ := t₁ + t₂
  t₃ / b^2 = t₁ / a₁^2 := by sorry

end NUMINAMATH_CALUDE_similar_polygon_area_sum_l1801_180164


namespace NUMINAMATH_CALUDE_harrys_age_l1801_180117

theorem harrys_age :
  ∀ (H : ℕ),
  (H + 24 : ℕ) - H / 25 = H + 22 →
  H = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_harrys_age_l1801_180117


namespace NUMINAMATH_CALUDE_birthday_theorem_l1801_180189

def birthday_money (age : ℕ) : ℕ := age * 5

theorem birthday_theorem : 
  ∀ (age : ℕ), age = 3 + 3 * 3 → birthday_money age = 60 := by
  sorry

end NUMINAMATH_CALUDE_birthday_theorem_l1801_180189


namespace NUMINAMATH_CALUDE_intersection_segment_length_l1801_180190

noncomputable section

/-- Curve C in Cartesian coordinates -/
def C (x y : ℝ) : Prop := x^2 = 4*y

/-- Line l in Cartesian coordinates -/
def l (x y : ℝ) : Prop := y = x + 1

/-- Point on both curve C and line l -/
def intersection_point (p : ℝ × ℝ) : Prop :=
  C p.1 p.2 ∧ l p.1 p.2

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_segment_length :
  ∃ (M N : ℝ × ℝ), intersection_point M ∧ intersection_point N ∧ distance M N = 8 :=
sorry

end

end NUMINAMATH_CALUDE_intersection_segment_length_l1801_180190


namespace NUMINAMATH_CALUDE_tangent_ratio_range_l1801_180186

open Real

-- Define the function f(x) = |e^x - 1|
noncomputable def f (x : ℝ) : ℝ := abs (exp x - 1)

-- Define the theorem
theorem tangent_ratio_range 
  (x₁ x₂ : ℝ) 
  (h₁ : x₁ < 0) 
  (h₂ : x₂ > 0) 
  (h_perp : (deriv f x₁) * (deriv f x₂) = -1) :
  ∃ (AM BN : ℝ), 
    AM > 0 ∧ BN > 0 ∧ 
    0 < AM / BN ∧ AM / BN < 1 :=
by sorry


end NUMINAMATH_CALUDE_tangent_ratio_range_l1801_180186


namespace NUMINAMATH_CALUDE_cost_price_percentage_l1801_180101

theorem cost_price_percentage (C S : ℝ) (h : C > 0) (h' : S > 0) :
  (S - C) / C = 3 → C / S = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l1801_180101


namespace NUMINAMATH_CALUDE_battle_station_staffing_l1801_180175

theorem battle_station_staffing (n m : ℕ) (h1 : n = 12) (h2 : m = 4) :
  (n.factorial / ((n - m).factorial * m.factorial)) = 11880 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l1801_180175
