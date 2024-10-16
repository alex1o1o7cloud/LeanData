import Mathlib

namespace NUMINAMATH_CALUDE_garden_walkway_area_l1850_185042

/-- Calculates the total area of walkways in a garden with given specifications -/
def walkway_area (rows : ℕ) (columns : ℕ) (bed_length : ℕ) (bed_width : ℕ) (walkway_width : ℕ) : ℕ :=
  let total_width := columns * bed_length + (columns + 1) * walkway_width
  let total_height := rows * bed_width + (rows + 1) * walkway_width
  let total_area := total_width * total_height
  let bed_area := rows * columns * bed_length * bed_width
  total_area - bed_area

theorem garden_walkway_area :
  walkway_area 4 3 8 3 2 = 416 :=
by sorry

end NUMINAMATH_CALUDE_garden_walkway_area_l1850_185042


namespace NUMINAMATH_CALUDE_words_with_at_least_two_consonants_l1850_185021

/-- The set of all available letters -/
def Letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of consonants -/
def Consonants : Finset Char := {'B', 'C', 'D', 'F'}

/-- The set of vowels -/
def Vowels : Finset Char := {'A', 'E'}

/-- The length of words we're considering -/
def WordLength : Nat := 5

/-- A function that counts the number of 5-letter words with at least two consonants -/
def countWordsWithAtLeastTwoConsonants : Nat := sorry

theorem words_with_at_least_two_consonants :
  countWordsWithAtLeastTwoConsonants = 7424 := by sorry

end NUMINAMATH_CALUDE_words_with_at_least_two_consonants_l1850_185021


namespace NUMINAMATH_CALUDE_triangle_inequality_l1850_185035

theorem triangle_inequality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_triangle : (x + y - z) * (y + z - x) * (z + x - y) > 0) : 
  x * (y + z)^2 + y * (z + x)^2 + z * (x + y)^2 - (x^3 + y^3 + z^3) ≤ 9 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1850_185035


namespace NUMINAMATH_CALUDE_tan_150_degrees_l1850_185066

theorem tan_150_degrees : Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l1850_185066


namespace NUMINAMATH_CALUDE_sin_x_plus_pi_l1850_185039

theorem sin_x_plus_pi (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.tan x = -4/3) :
  Real.sin (x + π) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_x_plus_pi_l1850_185039


namespace NUMINAMATH_CALUDE_final_selling_price_l1850_185077

/-- Calculate the final selling price of three items with given costs, profit/loss percentages, discount, and tax. -/
theorem final_selling_price (cycle_cost scooter_cost motorbike_cost : ℚ)
  (cycle_loss_percent scooter_profit_percent motorbike_profit_percent : ℚ)
  (discount_percent tax_percent : ℚ) :
  let cycle_price := cycle_cost * (1 - cycle_loss_percent)
  let scooter_price := scooter_cost * (1 + scooter_profit_percent)
  let motorbike_price := motorbike_cost * (1 + motorbike_profit_percent)
  let total_price := cycle_price + scooter_price + motorbike_price
  let discounted_price := total_price * (1 - discount_percent)
  let final_price := discounted_price * (1 + tax_percent)
  cycle_cost = 2300 ∧
  scooter_cost = 12000 ∧
  motorbike_cost = 25000 ∧
  cycle_loss_percent = 0.30 ∧
  scooter_profit_percent = 0.25 ∧
  motorbike_profit_percent = 0.15 ∧
  discount_percent = 0.10 ∧
  tax_percent = 0.05 →
  final_price = 41815.20 := by
sorry

end NUMINAMATH_CALUDE_final_selling_price_l1850_185077


namespace NUMINAMATH_CALUDE_shop_width_l1850_185055

/-- Given a rectangular shop with the following properties:
  * Length is 18 feet
  * Monthly rent is Rs. 3600
  * Annual rent per square foot is Rs. 120
  Prove that the width of the shop is 20 feet. -/
theorem shop_width (length : ℝ) (monthly_rent : ℝ) (annual_rent_per_sqft : ℝ) 
  (h1 : length = 18)
  (h2 : monthly_rent = 3600)
  (h3 : annual_rent_per_sqft = 120) :
  (monthly_rent * 12) / (length * annual_rent_per_sqft) = 20 := by
  sorry

end NUMINAMATH_CALUDE_shop_width_l1850_185055


namespace NUMINAMATH_CALUDE_sqrt2_plus_sqrt3_power_2012_decimal_digits_l1850_185029

theorem sqrt2_plus_sqrt3_power_2012_decimal_digits :
  ∃ k : ℤ,
    (k : ℝ) < (Real.sqrt 2 + Real.sqrt 3) ^ 2012 ∧
    (Real.sqrt 2 + Real.sqrt 3) ^ 2012 < (k + 1 : ℝ) ∧
    (Real.sqrt 2 + Real.sqrt 3) ^ 2012 - k > (79 : ℝ) / 100 ∧
    (Real.sqrt 2 + Real.sqrt 3) ^ 2012 - k < (80 : ℝ) / 100 :=
by
  sorry


end NUMINAMATH_CALUDE_sqrt2_plus_sqrt3_power_2012_decimal_digits_l1850_185029


namespace NUMINAMATH_CALUDE_max_and_dog_same_age_in_dog_years_l1850_185074

/-- Conversion rate from human years to dog years -/
def human_to_dog_years : ℕ → ℕ := (· * 7)

/-- Max's age in human years -/
def max_age : ℕ := 3

/-- Max's dog's age in human years -/
def dog_age : ℕ := 3

/-- Theorem: Max and his dog have the same age when expressed in dog years -/
theorem max_and_dog_same_age_in_dog_years :
  human_to_dog_years max_age = human_to_dog_years dog_age :=
by sorry

end NUMINAMATH_CALUDE_max_and_dog_same_age_in_dog_years_l1850_185074


namespace NUMINAMATH_CALUDE_correct_balloons_given_to_fred_l1850_185044

/-- The number of balloons Sam gave to Fred -/
def balloons_given_to_fred (sam_initial : ℕ) (mary : ℕ) (total : ℕ) : ℕ :=
  sam_initial - (total - mary)

theorem correct_balloons_given_to_fred :
  balloons_given_to_fred 6 7 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_balloons_given_to_fred_l1850_185044


namespace NUMINAMATH_CALUDE_judson_contribution_is_500_l1850_185061

def house_painting_problem (judson_contribution : ℝ) : Prop :=
  let kenny_contribution := 1.2 * judson_contribution
  let camilo_contribution := kenny_contribution + 200
  judson_contribution + kenny_contribution + camilo_contribution = 1900

theorem judson_contribution_is_500 :
  ∃ (judson_contribution : ℝ),
    house_painting_problem judson_contribution ∧ judson_contribution = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_judson_contribution_is_500_l1850_185061


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1850_185033

/-- A circle defined by x^2 + y^2 + 2x + 4y + m = 0 has exactly two points at a distance of √2
    from the line x + y + 1 = 0 if and only if m ∈ (-3, 5) -/
theorem circle_line_intersection (m : ℝ) :
  (∃! (p q : ℝ × ℝ),
    p ≠ q ∧
    (p.1^2 + p.2^2 + 2*p.1 + 4*p.2 + m = 0) ∧
    (q.1^2 + q.2^2 + 2*q.1 + 4*q.2 + m = 0) ∧
    (p.1 + p.2 + 1 ≠ 0) ∧
    (q.1 + q.2 + 1 ≠ 0) ∧
    ((p.1 + p.2 + 1)^2 / 2 = 2) ∧
    ((q.1 + q.2 + 1)^2 / 2 = 2))
  ↔
  (-3 < m ∧ m < 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1850_185033


namespace NUMINAMATH_CALUDE_initial_percentage_chemical_x_l1850_185037

/-- Given an 80-liter mixture and adding 20 liters of pure chemical x resulting in a 100-liter mixture that is 44% chemical x, prove that the initial percentage of chemical x was 30%. -/
theorem initial_percentage_chemical_x : 
  ∀ (initial_percentage : ℝ),
  initial_percentage ≥ 0 ∧ initial_percentage ≤ 1 →
  (80 * initial_percentage + 20) / 100 = 0.44 →
  initial_percentage = 0.3 := by
sorry

end NUMINAMATH_CALUDE_initial_percentage_chemical_x_l1850_185037


namespace NUMINAMATH_CALUDE_at_least_two_equations_have_solution_l1850_185012

theorem at_least_two_equations_have_solution (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y : ℝ), (
    ((x - a) * (x - b) = x - c ∧ (y - b) * (y - c) = y - a) ∨
    ((x - a) * (x - b) = x - c ∧ (y - c) * (y - a) = y - b) ∨
    ((x - b) * (x - c) = x - a ∧ (y - c) * (y - a) = y - b)
  ) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_equations_have_solution_l1850_185012


namespace NUMINAMATH_CALUDE_tangent_line_polar_equation_l1850_185065

/-- The polar coordinate equation of the tangent line to the circle ρ = 4sin θ
    that passes through the point (2√2, π/4) is ρ cos θ = 2. -/
theorem tangent_line_polar_equation (ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) →  -- Circle equation
  (∃ (ρ₀ θ₀ : ℝ), ρ₀ = 2 * Real.sqrt 2 ∧ θ₀ = π / 4 ∧ 
    ρ₀ * Real.cos θ₀ = 2 ∧ ρ₀ * Real.sin θ₀ = 2) →  -- Point (2√2, π/4)
  (ρ * Real.cos θ = 2) -- Tangent line equation
:= by sorry

end NUMINAMATH_CALUDE_tangent_line_polar_equation_l1850_185065


namespace NUMINAMATH_CALUDE_carina_coffee_amount_l1850_185002

/-- The amount of coffee Carina has, given the following conditions:
  * She has coffee divided into 5- and 10-ounce packages
  * She has 2 more 5-ounce packages than 10-ounce packages
  * She has 5 10-ounce packages
-/
theorem carina_coffee_amount :
  let num_10oz_packages : ℕ := 5
  let num_5oz_packages : ℕ := num_10oz_packages + 2
  let total_ounces : ℕ := num_10oz_packages * 10 + num_5oz_packages * 5
  total_ounces = 85 := by sorry

end NUMINAMATH_CALUDE_carina_coffee_amount_l1850_185002


namespace NUMINAMATH_CALUDE_bug_visits_24_tiles_l1850_185086

/-- The number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tilesVisited (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- The floor dimensions -/
def floorWidth : ℕ := 12
def floorLength : ℕ := 18

/-- Theorem: A bug walking diagonally across the given rectangular floor visits 24 tiles -/
theorem bug_visits_24_tiles :
  tilesVisited floorWidth floorLength = 24 := by
  sorry


end NUMINAMATH_CALUDE_bug_visits_24_tiles_l1850_185086


namespace NUMINAMATH_CALUDE_actual_speed_proof_l1850_185025

theorem actual_speed_proof (v : ℝ) (h : (v / (v + 10) = 3 / 4)) : v = 30 := by
  sorry

end NUMINAMATH_CALUDE_actual_speed_proof_l1850_185025


namespace NUMINAMATH_CALUDE_orange_box_capacity_l1850_185009

/-- 
Given two boxes for carrying oranges, where:
- The first box has a capacity of 80 and is filled 3/4 full
- The second box has an unknown capacity C and is filled 3/5 full
- The total number of oranges in both boxes is 90

This theorem proves that the capacity C of the second box is 50.
-/
theorem orange_box_capacity 
  (box1_capacity : ℕ) 
  (box1_fill : ℚ) 
  (box2_fill : ℚ) 
  (total_oranges : ℕ) 
  (h1 : box1_capacity = 80)
  (h2 : box1_fill = 3/4)
  (h3 : box2_fill = 3/5)
  (h4 : total_oranges = 90) :
  ∃ (C : ℕ), box1_fill * box1_capacity + box2_fill * C = total_oranges ∧ C = 50 := by
sorry

end NUMINAMATH_CALUDE_orange_box_capacity_l1850_185009


namespace NUMINAMATH_CALUDE_degree_to_radian_15_l1850_185045

theorem degree_to_radian_15 : 
  (15 : ℝ) * (π / 180) = π / 12 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_15_l1850_185045


namespace NUMINAMATH_CALUDE_highway_length_l1850_185027

/-- The length of a highway given two cars starting from opposite ends -/
theorem highway_length (speed1 speed2 : ℝ) (time : ℝ) (h1 : speed1 = 54)
  (h2 : speed2 = 57) (h3 : time = 3) :
  speed1 * time + speed2 * time = 333 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_l1850_185027


namespace NUMINAMATH_CALUDE_area_after_folding_l1850_185019

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (D : Point)
  (R : Point)
  (Q : Point)
  (C : Point)

/-- Calculates the area of a quadrilateral -/
def area_quadrilateral (quad : Quadrilateral) : ℝ := sorry

/-- Creates a rectangle with given dimensions -/
def create_rectangle (width : ℝ) (height : ℝ) : Rectangle := sorry

/-- Performs the folding operation on the rectangle -/
def fold_rectangle (rect : Rectangle) : Quadrilateral := sorry

theorem area_after_folding (width height : ℝ) :
  width = 5 →
  height = 8 →
  let rect := create_rectangle width height
  let folded := fold_rectangle rect
  area_quadrilateral folded = 11.5 := by sorry

end NUMINAMATH_CALUDE_area_after_folding_l1850_185019


namespace NUMINAMATH_CALUDE_quadratic_root_l1850_185059

theorem quadratic_root (a b c : ℝ) (h_arithmetic : b - a = c - b) 
  (h_a : a = 5) (h_c : c = 1) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_one_root : ∃! x, a * x^2 + b * x + c = 0) : 
  ∃ x, a * x^2 + b * x + c = 0 ∧ x = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_l1850_185059


namespace NUMINAMATH_CALUDE_diagonal_smallest_angle_at_midpoints_l1850_185010

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with side length a -/
structure Cube where
  a : ℝ
  center : Point3D

/-- Calculates the angle at which the diagonal is seen from a point on the cube's surface -/
noncomputable def angleFromPoint (c : Cube) (p : Point3D) : ℝ := sorry

/-- Checks if a point is on the surface of the cube -/
def isOnSurface (c : Cube) (p : Point3D) : Prop := sorry

/-- Calculates the midpoints of the cube's faces -/
def faceMidpoints (c : Cube) : List Point3D := sorry

/-- Main theorem: The diagonal is seen at the smallest angle from the midpoints of the cube's faces -/
theorem diagonal_smallest_angle_at_midpoints (c : Cube) :
  ∀ p : Point3D, isOnSurface c p →
    (p ∉ faceMidpoints c → 
      ∀ m ∈ faceMidpoints c, angleFromPoint c p > angleFromPoint c m) :=
sorry

end NUMINAMATH_CALUDE_diagonal_smallest_angle_at_midpoints_l1850_185010


namespace NUMINAMATH_CALUDE_volume_between_concentric_spheres_l1850_185001

theorem volume_between_concentric_spheres :
  let r₁ : ℝ := 4
  let r₂ : ℝ := 8
  let V₁ := (4 / 3) * π * r₁^3
  let V₂ := (4 / 3) * π * r₂^3
  V₂ - V₁ = (1792 / 3) * π := by sorry

end NUMINAMATH_CALUDE_volume_between_concentric_spheres_l1850_185001


namespace NUMINAMATH_CALUDE_variables_related_probability_l1850_185031

/-- The k-value obtained from a 2×2 contingency table -/
def k : ℝ := 4.073

/-- The probability that k^2 is greater than or equal to 3.841 -/
def p_3841 : ℝ := 0.05

/-- The probability that k^2 is greater than or equal to 5.024 -/
def p_5024 : ℝ := 0.025

/-- The theorem stating the probability of two variables being related -/
theorem variables_related_probability : ℝ := by
  sorry

end NUMINAMATH_CALUDE_variables_related_probability_l1850_185031


namespace NUMINAMATH_CALUDE_cos_equality_for_specific_angles_l1850_185038

theorem cos_equality_for_specific_angles :
  ∀ n : ℤ, 0 ≤ n ∧ n ≤ 360 →
    (Real.cos (n * π / 180) = Real.cos (321 * π / 180) ↔ n = 39 ∨ n = 321) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_for_specific_angles_l1850_185038


namespace NUMINAMATH_CALUDE_married_employees_percentage_l1850_185098

-- Define the company
structure Company where
  total_employees : ℕ
  women_percentage : ℚ
  men_single_ratio : ℚ
  women_married_percentage : ℚ

-- Define the conditions
def company_conditions (c : Company) : Prop :=
  c.women_percentage = 64 / 100 ∧
  c.men_single_ratio = 2 / 3 ∧
  c.women_married_percentage = 75 / 100

-- Define the function to calculate the percentage of married employees
def married_percentage (c : Company) : ℚ :=
  let men_percentage := 1 - c.women_percentage
  let married_men := (1 - c.men_single_ratio) * men_percentage
  let married_women := c.women_married_percentage * c.women_percentage
  married_men + married_women

-- Theorem statement
theorem married_employees_percentage (c : Company) :
  company_conditions c → married_percentage c = 60 / 100 := by
  sorry


end NUMINAMATH_CALUDE_married_employees_percentage_l1850_185098


namespace NUMINAMATH_CALUDE_parallelogram_problem_l1850_185092

-- Define a parallelogram
structure Parallelogram :=
  (EF GH FG HE : ℝ)
  (is_parallelogram : EF = GH ∧ FG = HE)

-- Define the problem
theorem parallelogram_problem (EFGH : Parallelogram)
  (h1 : EFGH.EF = 52)
  (h2 : ∃ z : ℝ, EFGH.FG = 2 * z^4)
  (h3 : ∃ w : ℝ, EFGH.GH = 3 * w + 6)
  (h4 : EFGH.HE = 16) :
  ∃ w z : ℝ, w * z = 46 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_problem_l1850_185092


namespace NUMINAMATH_CALUDE_integer_divisibility_equivalence_l1850_185004

theorem integer_divisibility_equivalence (n : ℤ) : 
  (∃ a b : ℤ, 3 * n - 2 = 5 * a ∧ 2 * n + 1 = 7 * b) ↔ 
  (∃ k : ℤ, n = 35 * k + 24) := by
sorry

end NUMINAMATH_CALUDE_integer_divisibility_equivalence_l1850_185004


namespace NUMINAMATH_CALUDE_intersection_of_intervals_l1850_185016

open Set

theorem intersection_of_intervals (A B : Set ℝ) :
  A = {x | -1 < x ∧ x < 2} →
  B = {x | 1 < x ∧ x < 3} →
  A ∩ B = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_intervals_l1850_185016


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1850_185080

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 24 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  ∀ k : ℕ, k ≤ 24 → (k ∣ (n * (n + 1) * (n + 2) * (n + 3))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1850_185080


namespace NUMINAMATH_CALUDE_nested_fraction_equals_nineteen_elevenths_l1850_185094

theorem nested_fraction_equals_nineteen_elevenths :
  1 + 1 / (1 + 1 / (2 + 2 / 3)) = 19 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equals_nineteen_elevenths_l1850_185094


namespace NUMINAMATH_CALUDE_polyhedron_volume_theorem_l1850_185076

/-- A polyhedron consisting of a prism and two pyramids -/
structure Polyhedron where
  prism_volume : ℝ
  pyramid_volume : ℝ
  prism_volume_eq : prism_volume = Real.sqrt 2 - 1
  pyramid_volume_eq : pyramid_volume = 1 / 6

/-- The total volume of the polyhedron -/
def total_volume (p : Polyhedron) : ℝ :=
  p.prism_volume + 2 * p.pyramid_volume

theorem polyhedron_volume_theorem (p : Polyhedron) :
  total_volume p = Real.sqrt 2 - 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_theorem_l1850_185076


namespace NUMINAMATH_CALUDE_fish_tank_water_calculation_l1850_185079

theorem fish_tank_water_calculation (initial_water : ℝ) (added_water : ℝ) : 
  initial_water = 7.75 → added_water = 7 → initial_water + added_water = 14.75 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_water_calculation_l1850_185079


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1850_185089

theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (a 3 * a 3 - 5 * a 3 + 4 = 0) →             -- a_3 is a root of x^2 - 5x + 4 = 0
  (a 5 * a 5 - 5 * a 5 + 4 = 0) →             -- a_5 is a root of x^2 - 5x + 4 = 0
  (a 2 * a 4 * a 6 = 8 ∨ a 2 * a 4 * a 6 = -8) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1850_185089


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l1850_185093

theorem wrong_mark_calculation (total_marks : ℝ) : 
  let n : ℕ := 40
  let correct_mark : ℝ := 63
  let wrong_mark : ℝ := (total_marks - correct_mark + n / 2) / (1 - 1 / n)
  wrong_mark = 43 := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l1850_185093


namespace NUMINAMATH_CALUDE_equation_condition_for_x_equals_4_l1850_185003

theorem equation_condition_for_x_equals_4 :
  (∃ x : ℝ, x^2 - 3*x - 4 = 0) ∧
  (∀ x : ℝ, x = 4 → x^2 - 3*x - 4 = 0) ∧
  (∃ x : ℝ, x^2 - 3*x - 4 = 0 ∧ x ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_condition_for_x_equals_4_l1850_185003


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_primes_l1850_185053

/-- Given four positive prime numbers whose product equals the sum of 55 consecutive positive integers,
    the smallest possible sum of these four primes is 28. -/
theorem smallest_sum_of_four_primes (a b c d : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime c → Nat.Prime d →
  (∃ x : ℕ, a * b * c * d = (55 : ℕ) * (x + 27)) →
  (∀ w x y z : ℕ, Nat.Prime w → Nat.Prime x → Nat.Prime y → Nat.Prime z →
    (∃ n : ℕ, w * x * y * z = (55 : ℕ) * (n + 27)) →
    a + b + c + d ≤ w + x + y + z) →
  a + b + c + d = 28 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_primes_l1850_185053


namespace NUMINAMATH_CALUDE_birdhouse_volume_difference_l1850_185081

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℚ := 12

/-- Volume of a rectangular prism -/
def volume (width height depth : ℚ) : ℚ := width * height * depth

/-- Sara's birdhouse dimensions in feet -/
def sara_width : ℚ := 1
def sara_height : ℚ := 2
def sara_depth : ℚ := 2

/-- Jake's birdhouse dimensions in inches -/
def jake_width : ℚ := 16
def jake_height : ℚ := 20
def jake_depth : ℚ := 18

/-- Theorem stating the difference in volume between Sara's and Jake's birdhouses -/
theorem birdhouse_volume_difference :
  volume (sara_width * feet_to_inches) (sara_height * feet_to_inches) (sara_depth * feet_to_inches) -
  volume jake_width jake_height jake_depth = 1152 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_volume_difference_l1850_185081


namespace NUMINAMATH_CALUDE_shaded_area_circles_l1850_185095

theorem shaded_area_circles (R : ℝ) (r : ℝ) : 
  R^2 * π = 100 * π →
  r = R / 2 →
  (2 / 3) * (π * R^2) + (1 / 3) * (π * r^2) = 75 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_circles_l1850_185095


namespace NUMINAMATH_CALUDE_library_book_sale_l1850_185051

theorem library_book_sale (initial_books : ℕ) (remaining_fraction : ℚ) : 
  initial_books = 9900 →
  remaining_fraction = 4/6 →
  initial_books * (1 - remaining_fraction) = 3300 :=
by sorry

end NUMINAMATH_CALUDE_library_book_sale_l1850_185051


namespace NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_12_l1850_185087

theorem sum_seven_smallest_multiples_of_12 : 
  (Finset.range 7).sum (fun i => 12 * (i + 1)) = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_12_l1850_185087


namespace NUMINAMATH_CALUDE_triangle_sum_proof_l1850_185026

/-- Triangle operation: a + b - 2c --/
def triangle_op (a b c : ℤ) : ℤ := a + b - 2*c

theorem triangle_sum_proof :
  let t1 := triangle_op 3 4 5
  let t2 := triangle_op 6 8 2
  2 * t1 + 3 * t2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_proof_l1850_185026


namespace NUMINAMATH_CALUDE_seed_distribution_l1850_185023

theorem seed_distribution (total : ℕ) (a b c : ℕ) : 
  total = 100 →
  a = b + 10 →
  b = 30 →
  total = a + b + c →
  c = 30 := by
sorry

end NUMINAMATH_CALUDE_seed_distribution_l1850_185023


namespace NUMINAMATH_CALUDE_integer_solution_zero_l1850_185054

theorem integer_solution_zero (x y z t : ℤ) : 
  x^4 - 2*y^4 - 4*z^4 - 8*t^4 = 0 → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_solution_zero_l1850_185054


namespace NUMINAMATH_CALUDE_percentage_equality_l1850_185018

theorem percentage_equality (x y : ℝ) (h1 : 3 * x = 3/4 * y) (h2 : x = 20) : y + 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1850_185018


namespace NUMINAMATH_CALUDE_fraction_zero_at_minus_one_denominator_nonzero_at_minus_one_largest_x_for_zero_fraction_l1850_185013

theorem fraction_zero_at_minus_one (x : ℝ) :
  (x + 1) / (9 * x^2 - 74 * x + 9) = 0 ↔ x = -1 :=
by
  sorry

theorem denominator_nonzero_at_minus_one :
  9 * (-1)^2 - 74 * (-1) + 9 ≠ 0 :=
by
  sorry

theorem largest_x_for_zero_fraction :
  ∀ y > -1, (y + 1) / (9 * y^2 - 74 * y + 9) ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_at_minus_one_denominator_nonzero_at_minus_one_largest_x_for_zero_fraction_l1850_185013


namespace NUMINAMATH_CALUDE_total_customers_is_43_l1850_185032

/-- Represents a table with a number of women and men -/
structure Table where
  women : ℕ
  men : ℕ

/-- Calculates the total number of customers at a table -/
def Table.total (t : Table) : ℕ := t.women + t.men

/-- Represents the waiter's situation -/
structure WaiterSituation where
  table1 : Table
  table2 : Table
  table3 : Table
  table4 : Table
  table5 : Table
  table6 : Table
  walkIn : Table
  table3Left : ℕ
  table4Joined : Table

/-- The initial situation of the waiter -/
def initialSituation : WaiterSituation where
  table1 := { women := 2, men := 4 }
  table2 := { women := 4, men := 3 }
  table3 := { women := 3, men := 5 }
  table4 := { women := 5, men := 2 }
  table5 := { women := 2, men := 1 }
  table6 := { women := 1, men := 2 }
  walkIn := { women := 4, men := 4 }
  table3Left := 2
  table4Joined := { women := 1, men := 2 }

/-- Calculates the total number of customers served by the waiter -/
def totalCustomersServed (s : WaiterSituation) : ℕ :=
  s.table1.total +
  s.table2.total +
  (s.table3.total - s.table3Left) +
  (s.table4.total + s.table4Joined.total) +
  s.table5.total +
  s.table6.total +
  s.walkIn.total

/-- Theorem stating that the total number of customers served is 43 -/
theorem total_customers_is_43 : totalCustomersServed initialSituation = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_customers_is_43_l1850_185032


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_l1850_185020

-- Define the structure of the cube
structure Cube where
  edge_length : ℝ
  small_cubes : ℕ
  white_cubes : ℕ
  black_cubes : ℕ

-- Define the larger cube
def larger_cube : Cube :=
  { edge_length := 4
  , small_cubes := 64
  , white_cubes := 48
  , black_cubes := 16 }

-- Function to calculate the surface area of a cube
def surface_area (c : Cube) : ℝ :=
  6 * c.edge_length ^ 2

-- Function to calculate the number of exposed black faces
def exposed_black_faces (c : Cube) : ℕ :=
  24 + 4  -- 8 corners with 3 faces each, plus 4 along the top edge

-- Theorem stating the fraction of white surface area
theorem white_surface_area_fraction (c : Cube) :
  c = larger_cube →
  (surface_area c - exposed_black_faces c) / surface_area c = 17 / 24 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_area_fraction_l1850_185020


namespace NUMINAMATH_CALUDE_table_tennis_lineups_l1850_185088

/-- Represents a team in the table tennis competition -/
structure Team :=
  (members : Finset Nat)
  (size : members.card = 5)

/-- Represents a lineup for the competition -/
structure Lineup :=
  (singles1 : Nat)
  (singles2 : Nat)
  (doubles1 : Nat)
  (doubles2 : Nat)
  (all_different : singles1 ≠ singles2 ∧ singles1 ≠ doubles1 ∧ singles1 ≠ doubles2 ∧ 
                   singles2 ≠ doubles1 ∧ singles2 ≠ doubles2 ∧ doubles1 ≠ doubles2)

/-- The theorem to be proved -/
theorem table_tennis_lineups (t : Team) (a : Nat) (h : a ∈ t.members) : 
  (∃ l : Finset Lineup, l.card = 60 ∧ ∀ lineup ∈ l, lineup.singles1 ∈ t.members ∧ 
                                                    lineup.singles2 ∈ t.members ∧ 
                                                    lineup.doubles1 ∈ t.members ∧ 
                                                    lineup.doubles2 ∈ t.members) ∧
  (∃ l : Finset Lineup, l.card = 36 ∧ ∀ lineup ∈ l, lineup.singles1 ∈ t.members ∧ 
                                                    lineup.singles2 ∈ t.members ∧ 
                                                    lineup.doubles1 ∈ t.members ∧ 
                                                    lineup.doubles2 ∈ t.members ∧ 
                                                    (lineup.doubles1 ≠ a ∧ lineup.doubles2 ≠ a)) :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_lineups_l1850_185088


namespace NUMINAMATH_CALUDE_last_score_entered_l1850_185008

def scores : List ℕ := [60, 65, 70, 75, 80, 85, 95]

theorem last_score_entered (s : ℕ) :
  s ∈ scores →
  (s = 80 ↔ (List.sum scores - s) % 6 = 0 ∧
    ∀ t ∈ scores, t ≠ s → (List.sum scores - t) % 6 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_last_score_entered_l1850_185008


namespace NUMINAMATH_CALUDE_cats_sold_during_sale_l1850_185046

theorem cats_sold_during_sale 
  (initial_siamese : ℕ) 
  (initial_house : ℕ) 
  (remaining : ℕ) 
  (h1 : initial_siamese = 13) 
  (h2 : initial_house = 5) 
  (h3 : remaining = 8) : 
  initial_siamese + initial_house - remaining = 10 := by
sorry

end NUMINAMATH_CALUDE_cats_sold_during_sale_l1850_185046


namespace NUMINAMATH_CALUDE_cut_pyramid_volume_l1850_185068

/-- The volume of a smaller pyramid cut from a right square pyramid -/
theorem cut_pyramid_volume (base_edge original_height slant_edge cut_height : ℝ) : 
  base_edge = 12 * Real.sqrt 2 →
  slant_edge = 15 →
  original_height = Real.sqrt (slant_edge^2 - (base_edge/2)^2) →
  cut_height = 5 →
  cut_height < original_height →
  (1/3) * (base_edge * (original_height - cut_height) / original_height)^2 * (original_height - cut_height) = 2048/27 :=
by sorry

end NUMINAMATH_CALUDE_cut_pyramid_volume_l1850_185068


namespace NUMINAMATH_CALUDE_cost_at_two_l1850_185085

/-- The cost function for a product -/
def cost (q : ℝ) : ℝ := q^3 + q - 1

/-- Theorem: The cost is 9 when the quantity is 2 -/
theorem cost_at_two : cost 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cost_at_two_l1850_185085


namespace NUMINAMATH_CALUDE_range_of_a_l1850_185014

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1850_185014


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1850_185082

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (30 * π / 180 + α) = 3/5)
  (h2 : 60 * π / 180 < α)
  (h3 : α < 150 * π / 180) :
  Real.cos α = (3 - 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1850_185082


namespace NUMINAMATH_CALUDE_max_at_2_implies_c_6_l1850_185049

/-- The function f(x) = x(x-c)² has a maximum value at x=2 -/
def has_max_at_2 (c : ℝ) : Prop :=
  ∀ x : ℝ, x * (x - c)^2 ≤ 2 * (2 - c)^2

/-- Theorem: If f(x) = x(x-c)² has a maximum value at x=2, then c = 6 -/
theorem max_at_2_implies_c_6 :
  ∀ c : ℝ, has_max_at_2 c → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_at_2_implies_c_6_l1850_185049


namespace NUMINAMATH_CALUDE_villages_with_more_knights_count_l1850_185070

/-- The number of villages on the island -/
def total_villages : ℕ := 1000

/-- The number of inhabitants in each village -/
def inhabitants_per_village : ℕ := 99

/-- The total number of knights on the island -/
def total_knights : ℕ := 54054

/-- The number of people in each village who answered there are more knights -/
def more_knights_answers : ℕ := 66

/-- The number of people in each village who answered there are more liars -/
def more_liars_answers : ℕ := 33

/-- The number of villages with more knights than liars -/
def villages_with_more_knights : ℕ := 638

theorem villages_with_more_knights_count :
  villages_with_more_knights = 
    (total_knights - more_liars_answers * total_villages) / 
    (more_knights_answers - more_liars_answers) :=
by sorry

end NUMINAMATH_CALUDE_villages_with_more_knights_count_l1850_185070


namespace NUMINAMATH_CALUDE_b_share_proof_l1850_185096

/-- The number of days B takes to complete the work alone -/
def b_days : ℕ := 10

/-- The number of days A takes to complete the work alone -/
def a_days : ℕ := 15

/-- The total wages for the work in Rupees -/
def total_wages : ℕ := 5000

/-- The share of wages B should receive when working together with A -/
def b_share : ℕ := 3000

/-- Theorem stating that B's share of the wages when working with A is 3000 Rupees -/
theorem b_share_proof : 
  b_share = (b_days * total_wages) / (a_days + b_days) := by sorry

end NUMINAMATH_CALUDE_b_share_proof_l1850_185096


namespace NUMINAMATH_CALUDE_f_properties_l1850_185097

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_properties :
  ∃ (m : ℝ),
    (∀ (a x₁ x₂ : ℝ), x₁ < x₂ → f a x₁ < f a x₂) ∧
    (∀ (x : ℝ), f 1 x = -f 1 (-x)) ∧
    (m = 12/5 ∧ ∀ (x : ℝ), x ∈ Set.Icc 2 3 → f 1 x ≥ m / 2^x) ∧
    (∀ (m' : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 2 3 → f 1 x ≥ m' / 2^x) → m' ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1850_185097


namespace NUMINAMATH_CALUDE_total_albums_l1850_185040

/-- The number of albums each person has -/
structure Albums where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ
  carlos : ℕ

/-- The conditions given in the problem -/
def album_conditions (a : Albums) : Prop :=
  a.adele = 30 ∧
  a.bridget = a.adele - 15 ∧
  a.katrina = 6 * a.bridget ∧
  a.miriam = 5 * a.katrina ∧
  a.carlos = 3 * a.miriam

/-- The theorem to prove -/
theorem total_albums (a : Albums) (h : album_conditions a) :
  a.adele + a.bridget + a.katrina + a.miriam + a.carlos = 1935 := by
  sorry

end NUMINAMATH_CALUDE_total_albums_l1850_185040


namespace NUMINAMATH_CALUDE_exists_quadrilateral_with_adjacent_colors_l1850_185075

/-- Represents the color of a vertex -/
inductive Color
| Black
| White

/-- Represents a convex polygon -/
structure ConvexPolygon where
  vertices : ℕ
  coloring : ℕ → Color

/-- Represents a quadrilateral formed by dividing the polygon -/
structure Quadrilateral where
  v1 : ℕ
  v2 : ℕ
  v3 : ℕ
  v4 : ℕ

/-- The specific coloring pattern of the 2550-gon -/
def specific_coloring : ℕ → Color := sorry

/-- The 2550-gon with the specific coloring -/
def polygon_2550 : ConvexPolygon :=
  { vertices := 2550,
    coloring := specific_coloring }

/-- Predicate to check if a quadrilateral has two adjacent black vertices and two adjacent white vertices -/
def has_adjacent_colors (q : Quadrilateral) (p : ConvexPolygon) : Prop := sorry

/-- A division of the polygon into quadrilaterals -/
def division : List Quadrilateral := sorry

/-- Theorem stating that there exists a quadrilateral with the required color pattern -/
theorem exists_quadrilateral_with_adjacent_colors :
  ∃ q ∈ division, has_adjacent_colors q polygon_2550 := by sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_with_adjacent_colors_l1850_185075


namespace NUMINAMATH_CALUDE_equation_solution_l1850_185005

theorem equation_solution (a b : ℝ) : 
  a^2 + b^2 + 2*a - 4*b + 5 = 0 → 2*a^2 + 4*b - 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1850_185005


namespace NUMINAMATH_CALUDE_min_triangles_proof_l1850_185063

/-- Represents an 8x8 square with one corner cell removed -/
structure ModifiedSquare where
  side_length : ℕ
  removed_cell_area : ℕ
  total_area : ℕ

/-- Represents a triangulation of the modified square -/
structure Triangulation where
  num_triangles : ℕ
  triangle_area : ℝ

/-- The minimum number of equal-area triangles that can divide the modified square -/
def min_triangles : ℕ := 18

theorem min_triangles_proof (s : ModifiedSquare) (t : Triangulation) :
  s.side_length = 8 ∧ 
  s.removed_cell_area = 1 ∧ 
  s.total_area = s.side_length * s.side_length - s.removed_cell_area ∧
  t.triangle_area = s.total_area / t.num_triangles ∧
  t.triangle_area ≤ 3.5 →
  t.num_triangles ≥ min_triangles :=
sorry

end NUMINAMATH_CALUDE_min_triangles_proof_l1850_185063


namespace NUMINAMATH_CALUDE_jerry_action_figures_count_l1850_185041

/-- Calculates the total number of action figures on Jerry's shelf --/
def total_action_figures (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of action figures is the sum of initial and added figures --/
theorem jerry_action_figures_count (initial : ℕ) (added : ℕ) :
  total_action_figures initial added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_jerry_action_figures_count_l1850_185041


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1850_185022

theorem smallest_prime_divisor_of_sum (p : ℕ) : 
  Prime p ∧ p ∣ (2^12 + 3^10 + 7^15) → p = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1850_185022


namespace NUMINAMATH_CALUDE_vasya_floor_l1850_185062

theorem vasya_floor (steps_per_floor : ℕ) (petya_steps : ℕ) (vasya_steps : ℕ) : 
  steps_per_floor * 2 = petya_steps → 
  vasya_steps = steps_per_floor * 4 → 
  5 = vasya_steps / steps_per_floor + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_vasya_floor_l1850_185062


namespace NUMINAMATH_CALUDE_smallest_multiple_l1850_185073

theorem smallest_multiple (x : ℕ) : x = 32 ↔ 
  (x > 0 ∧ 900 * x % 1152 = 0 ∧ ∀ y : ℕ, (y > 0 ∧ y < x) → 900 * y % 1152 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1850_185073


namespace NUMINAMATH_CALUDE_color_film_fraction_l1850_185091

/-- Given a committee reviewing films for a festival, this theorem proves
    the fraction of selected films that are in color. -/
theorem color_film_fraction
  (x y : ℕ) -- x and y are natural numbers
  (total_bw : ℕ := 40 * x) -- Total number of black-and-white films
  (total_color : ℕ := 10 * y) -- Total number of color films
  (bw_selected_percent : ℚ := y / x) -- Percentage of black-and-white films selected
  (color_selected_percent : ℚ := 1) -- All color films are selected
  : (total_color : ℚ) / ((bw_selected_percent * total_bw + total_color) : ℚ) = 5 / 26 :=
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l1850_185091


namespace NUMINAMATH_CALUDE_backyard_max_area_l1850_185072

theorem backyard_max_area (P : ℝ) (h : P > 0) :
  let A : ℝ → ℝ → ℝ := λ l w => l * w
  let perimeter : ℝ → ℝ → ℝ := λ l w => l + 2 * w
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ perimeter l w = P ∧
    ∀ (l' w' : ℝ), l' > 0 → w' > 0 → perimeter l' w' = P →
      A l w ≥ A l' w' ∧
      A l w = (P / 4) ^ 2 ∧
      w = P / 4 :=
by sorry

end NUMINAMATH_CALUDE_backyard_max_area_l1850_185072


namespace NUMINAMATH_CALUDE_store_a_advantage_l1850_185056

/-- The original price of each computer in yuan -/
def original_price : ℝ := 6000

/-- The cost of buying computers from Store A -/
def cost_store_a (x : ℝ) : ℝ := original_price + (0.75 * original_price) * (x - 1)

/-- The cost of buying computers from Store B -/
def cost_store_b (x : ℝ) : ℝ := 0.8 * original_price * x

/-- Theorem stating when it's more advantageous to buy from Store A -/
theorem store_a_advantage (x : ℝ) : x > 5 → cost_store_a x < cost_store_b x := by
  sorry

#check store_a_advantage

end NUMINAMATH_CALUDE_store_a_advantage_l1850_185056


namespace NUMINAMATH_CALUDE_range_of_m_l1850_185069

theorem range_of_m (m : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < m) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1850_185069


namespace NUMINAMATH_CALUDE_exists_a_with_two_common_tangents_l1850_185007

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C₂ -/
def C₂ (x y a : ℝ) : Prop := (x - 4)^2 + (y + a)^2 = 64

/-- Two circles have exactly two common tangents -/
def have_two_common_tangents (C₁ C₂ : ℝ → ℝ → Prop) : Prop := sorry

/-- Main theorem -/
theorem exists_a_with_two_common_tangents :
  ∃ a : ℕ, a > 0 ∧ have_two_common_tangents C₁ (C₂ · · a) :=
sorry

end NUMINAMATH_CALUDE_exists_a_with_two_common_tangents_l1850_185007


namespace NUMINAMATH_CALUDE_digits_of_2_pow_100_l1850_185043

theorem digits_of_2_pow_100 (N : ℕ) :
  (N = (Nat.digits 10 (2^100)).length) → 29 ≤ N ∧ N ≤ 34 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_2_pow_100_l1850_185043


namespace NUMINAMATH_CALUDE_remaining_sessions_proof_l1850_185017

theorem remaining_sessions_proof (total_patients : Nat) (total_sessions : Nat) 
  (patient1_sessions : Nat) (extra_sessions : Nat) :
  total_patients = 4 →
  total_sessions = 25 →
  patient1_sessions = 6 →
  extra_sessions = 5 →
  total_sessions - (patient1_sessions + (patient1_sessions + extra_sessions)) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_sessions_proof_l1850_185017


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1850_185036

/-- The total surface area of a cylinder with height 15 and radius 2 is 68π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 15
  let r : ℝ := 2
  let circle_area := π * r^2
  let lateral_area := 2 * π * r * h
  circle_area * 2 + lateral_area = 68 * π := by
sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1850_185036


namespace NUMINAMATH_CALUDE_inequality_range_theorem_l1850_185060

/-- The range of a for which |x+3| - |x-1| ≤ a^2 - 3a holds for all real x -/
theorem inequality_range_theorem (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ 
  (a ≤ -1 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_theorem_l1850_185060


namespace NUMINAMATH_CALUDE_complement_M_in_U_l1850_185015

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define the set M
def M : Set ℝ := {x | 2 * x - x^2 > 0}

-- Statement to prove
theorem complement_M_in_U : 
  {x : ℝ | x ∈ U ∧ x ∉ M} = {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l1850_185015


namespace NUMINAMATH_CALUDE_infinitely_many_a_making_n4_plus_a_composite_l1850_185000

theorem infinitely_many_a_making_n4_plus_a_composite :
  ∀ k : ℕ, k > 1 → ∃ a : ℕ, a = 4 * k^4 ∧ ∀ n : ℕ, ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ n^4 + a = x * y :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_a_making_n4_plus_a_composite_l1850_185000


namespace NUMINAMATH_CALUDE_product_of_sums_l1850_185067

theorem product_of_sums (a b c d : ℚ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : (a + c) * (a + d) = 1)
  (h2 : (b + c) * (b + d) = 1) :
  (a + c) * (b + c) = -1 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_l1850_185067


namespace NUMINAMATH_CALUDE_rectangle_largest_side_l1850_185058

/-- Given a rectangle with perimeter 240 feet and area equal to fifteen times its perimeter,
    prove that the length of its largest side is 60 feet. -/
theorem rectangle_largest_side (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧                   -- positive dimensions
  2 * (l + w) = 240 ∧               -- perimeter is 240 feet
  l * w = 15 * 240 →                -- area is fifteen times perimeter
  max l w = 60 := by sorry

end NUMINAMATH_CALUDE_rectangle_largest_side_l1850_185058


namespace NUMINAMATH_CALUDE_four_number_sequence_l1850_185024

theorem four_number_sequence : ∃ (a b c d : ℝ), 
  (∃ (q : ℝ), b = a * q ∧ c = b * q) ∧  -- Geometric progression
  (∃ (r : ℝ), c = b + r ∧ d = c + r) ∧  -- Arithmetic progression
  a + d = 21 ∧                          -- Sum of first and last
  b + c = 18 := by                      -- Sum of middle two
sorry

end NUMINAMATH_CALUDE_four_number_sequence_l1850_185024


namespace NUMINAMATH_CALUDE_divisibility_by_133_l1850_185071

theorem divisibility_by_133 (n : ℕ) : ∃ k : ℤ, 11^(n+2) + 12^(2*n+1) = 133 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_133_l1850_185071


namespace NUMINAMATH_CALUDE_line_parallel_perp_plane_l1850_185034

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perp_plane
  (m n : Line) (α : Plane)
  (h1 : m ≠ n)
  (h2 : parallel m n)
  (h3 : perp m α) :
  perp n α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perp_plane_l1850_185034


namespace NUMINAMATH_CALUDE_expression_values_l1850_185050

theorem expression_values (x y z : ℝ) (h : x * y * z ≠ 0) :
  let expr := |x| / x + y / |y| + |z| / z
  expr = 1 ∨ expr = -1 ∨ expr = 3 ∨ expr = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l1850_185050


namespace NUMINAMATH_CALUDE_tea_consumption_l1850_185047

/-- The total number of cups of tea consumed by three merchants -/
def total_cups (s o p : ℝ) : ℝ := s + o + p

/-- Theorem stating that the total cups of tea consumed is 19.5 -/
theorem tea_consumption (s o p : ℝ) 
  (h1 : s + o = 11) 
  (h2 : p + o = 15) 
  (h3 : p + s = 13) : 
  total_cups s o p = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_tea_consumption_l1850_185047


namespace NUMINAMATH_CALUDE_product_in_M_l1850_185011

/-- The set M of differences of squares of integers -/
def M : Set ℤ := {x | ∃ a b : ℤ, x = a^2 - b^2}

/-- Theorem: The product of any two elements in M is also in M -/
theorem product_in_M (p q : ℤ) (hp : p ∈ M) (hq : q ∈ M) : p * q ∈ M := by
  sorry

end NUMINAMATH_CALUDE_product_in_M_l1850_185011


namespace NUMINAMATH_CALUDE_least_stamps_l1850_185028

theorem least_stamps (n : ℕ) : n = 107 ↔ 
  (n > 0) ∧ 
  (n % 4 = 3) ∧ 
  (n % 5 = 2) ∧ 
  (n % 7 = 1) ∧ 
  (∀ m : ℕ, m > 0 → m % 4 = 3 → m % 5 = 2 → m % 7 = 1 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_least_stamps_l1850_185028


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1850_185090

theorem simplify_polynomial (x : ℝ) :
  2 * x * (5 * x^2 - 3 * x + 1) + 4 * (x^2 - 3 * x + 6) =
  10 * x^3 - 2 * x^2 - 10 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1850_185090


namespace NUMINAMATH_CALUDE_first_term_is_two_l1850_185030

/-- An increasing arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  arithmetic : ∃ d : ℝ, ∀ n, a (n + 1) - a n = d
  sum_first_three : a 1 + a 2 + a 3 = 12
  product_first_three : a 1 * a 2 * a 3 = 48

/-- The first term of the arithmetic sequence is 2 -/
theorem first_term_is_two (seq : ArithmeticSequence) : seq.a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_two_l1850_185030


namespace NUMINAMATH_CALUDE_rectangle_length_from_square_wire_l1850_185006

/-- Given a square with side length 20 cm and a rectangle with width 14 cm made from the same total wire length, the length of the rectangle is 26 cm. -/
theorem rectangle_length_from_square_wire (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 20 →
  rect_width = 14 →
  4 * square_side = 2 * (rect_width + rect_length) →
  rect_length = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_from_square_wire_l1850_185006


namespace NUMINAMATH_CALUDE_quadrilateral_ABCD_area_l1850_185084

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (A B C D : Point) : ℝ := sorry

theorem quadrilateral_ABCD_area :
  let A : Point := ⟨0, 1⟩
  let B : Point := ⟨1, 3⟩
  let C : Point := ⟨5, 2⟩
  let D : Point := ⟨4, 0⟩
  quadrilateralArea A B C D = 9 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_ABCD_area_l1850_185084


namespace NUMINAMATH_CALUDE_centers_form_rectangle_l1850_185064

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup : Prop := ∃ (C C1 C2 C3 C4 : Circle),
  -- C has radius 2
  C.radius = 2 ∧
  -- C1 and C2 have radius 1
  C1.radius = 1 ∧ C2.radius = 1 ∧
  -- C1 and C2 touch at the center of C
  C1.center = C.center + (1, 0) ∧ C2.center = C.center + (-1, 0) ∧
  -- C3 is inside C and touches C, C1, and C2
  (∃ x : ℝ, C3.radius = x ∧
    dist C3.center C.center = 2 - x ∧
    dist C3.center C1.center = 1 + x ∧
    dist C3.center C2.center = 1 + x) ∧
  -- C4 is inside C and touches C, C1, and C3
  (∃ y : ℝ, C4.radius = y ∧
    dist C4.center C.center = 2 - y ∧
    dist C4.center C1.center = 1 + y ∧
    dist C4.center C3.center = C3.radius + y)

-- Define what it means for four points to form a rectangle
def form_rectangle (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let d12 := dist p1 p2
  let d23 := dist p2 p3
  let d34 := dist p3 p4
  let d41 := dist p4 p1
  let d13 := dist p1 p3
  let d24 := dist p2 p4
  d12 = d34 ∧ d23 = d41 ∧ d13 = d24

-- Theorem statement
theorem centers_form_rectangle :
  problem_setup →
  ∃ (C C1 C3 C4 : Circle),
    form_rectangle C.center C1.center C3.center C4.center :=
sorry

end NUMINAMATH_CALUDE_centers_form_rectangle_l1850_185064


namespace NUMINAMATH_CALUDE_complex_simplification_l1850_185048

theorem complex_simplification :
  (4 - 2*Complex.I) - (7 - 2*Complex.I) + (6 - 3*Complex.I) = 3 - 3*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l1850_185048


namespace NUMINAMATH_CALUDE_xy_problem_l1850_185057

theorem xy_problem (x y : ℝ) (h1 : x + y = 7) (h2 : x * y = 6) : 
  ((x - y)^2 = 25) ∧ (x^3 * y + x * y^3 = 222) := by
  sorry

end NUMINAMATH_CALUDE_xy_problem_l1850_185057


namespace NUMINAMATH_CALUDE_extraneous_roots_equation_l1850_185083

theorem extraneous_roots_equation :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧
  (∀ x : ℝ, Real.sqrt (x + 15) - 8 / Real.sqrt (x + 15) = 6 →
    (x = r₁ ∨ x = r₂) ∧
    Real.sqrt (r₁ + 15) - 8 / Real.sqrt (r₁ + 15) ≠ 6 ∧
    Real.sqrt (r₂ + 15) - 8 / Real.sqrt (r₂ + 15) ≠ 6) :=
by
  sorry

end NUMINAMATH_CALUDE_extraneous_roots_equation_l1850_185083


namespace NUMINAMATH_CALUDE_max_volume_at_five_l1850_185052

def box_volume (x : ℝ) : ℝ := (30 - 2*x)^2 * x

def possible_x : Set ℝ := {4, 5, 6, 7}

theorem max_volume_at_five :
  ∀ x ∈ possible_x, x ≠ 5 → box_volume x ≤ box_volume 5 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_at_five_l1850_185052


namespace NUMINAMATH_CALUDE_weight_solution_l1850_185078

def weight_problem (A B C D E : ℝ) : Prop :=
  let avg_ABC := (A + B + C) / 3
  let avg_ABCD := (A + B + C + D) / 4
  let avg_BCDE := (B + C + D + E) / 4
  avg_ABC = 50 ∧ 
  avg_ABCD = 53 ∧ 
  E = D + 3 ∧ 
  avg_BCDE = 51 →
  A = 8

theorem weight_solution :
  ∀ A B C D E : ℝ, weight_problem A B C D E :=
by sorry

end NUMINAMATH_CALUDE_weight_solution_l1850_185078


namespace NUMINAMATH_CALUDE_first_cat_brown_eyed_kittens_l1850_185099

theorem first_cat_brown_eyed_kittens :
  ∀ (brown_eyed_first : ℕ),
  let blue_eyed_first : ℕ := 3
  let blue_eyed_second : ℕ := 4
  let brown_eyed_second : ℕ := 6
  let total_kittens : ℕ := blue_eyed_first + brown_eyed_first + blue_eyed_second + brown_eyed_second
  let total_blue_eyed : ℕ := blue_eyed_first + blue_eyed_second
  (total_blue_eyed : ℚ) / total_kittens = 35 / 100 →
  brown_eyed_first = 7 :=
by sorry

end NUMINAMATH_CALUDE_first_cat_brown_eyed_kittens_l1850_185099
