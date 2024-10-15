import Mathlib

namespace NUMINAMATH_CALUDE_calculate_income_person_income_l2931_293105

/-- Calculates a person's total income based on given distributions --/
theorem calculate_income (children_percentage : ℝ) (wife_percentage : ℝ) (orphan_percentage : ℝ) (remaining_amount : ℝ) : ℝ :=
  let total_children_percentage := 3 * children_percentage
  let remaining_percentage := 1 - (total_children_percentage + wife_percentage)
  let orphan_amount := orphan_percentage * remaining_percentage
  let final_percentage := remaining_percentage - orphan_amount
  remaining_amount / final_percentage

/-- Proves that the person's total income is approximately $168,421.05 --/
theorem person_income : 
  let income := calculate_income 0.15 0.3 0.05 40000
  ∃ ε > 0, |income - 168421.05| < ε :=
sorry

end NUMINAMATH_CALUDE_calculate_income_person_income_l2931_293105


namespace NUMINAMATH_CALUDE_pastries_sold_l2931_293167

def initial_pastries : ℕ := 148
def remaining_pastries : ℕ := 45

theorem pastries_sold : initial_pastries - remaining_pastries = 103 := by
  sorry

end NUMINAMATH_CALUDE_pastries_sold_l2931_293167


namespace NUMINAMATH_CALUDE_special_function_property_l2931_293197

/-- A function satisfying the given condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f (f y))^2)

/-- The main theorem to be proved -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
sorry

end NUMINAMATH_CALUDE_special_function_property_l2931_293197


namespace NUMINAMATH_CALUDE_cos_sin_identity_l2931_293133

theorem cos_sin_identity (α β : Real) :
  (Real.cos (α * π / 180) * Real.cos ((180 - α) * π / 180) + 
   Real.sin (α * π / 180) * Real.sin ((α / 2) * π / 180)) = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l2931_293133


namespace NUMINAMATH_CALUDE_galaxy_composition_l2931_293176

/-- Represents the counts of celestial bodies in a galaxy -/
structure GalaxyComposition where
  planets : ℕ
  solarSystems : ℕ
  stars : ℕ
  moonSystems : ℕ

/-- Calculates the composition of a galaxy based on given ratios and planet count -/
def calculateGalaxyComposition (planetCount : ℕ) : GalaxyComposition :=
  let solarSystems := planetCount * 8
  let stars := solarSystems * 4
  let moonSystems := planetCount * 3 / 5
  { planets := planetCount
  , solarSystems := solarSystems
  , stars := stars
  , moonSystems := moonSystems }

/-- Theorem stating the composition of the galaxy given the conditions -/
theorem galaxy_composition :
  let composition := calculateGalaxyComposition 20
  composition.planets = 20 ∧
  composition.solarSystems = 160 ∧
  composition.stars = 640 ∧
  composition.moonSystems = 12 :=
by sorry

end NUMINAMATH_CALUDE_galaxy_composition_l2931_293176


namespace NUMINAMATH_CALUDE_slopes_equal_necessary_not_sufficient_for_parallel_l2931_293185

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define parallel relation
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

-- Theorem statement
theorem slopes_equal_necessary_not_sufficient_for_parallel :
  -- Given two lines
  ∀ (l1 l2 : Line),
  -- l1 has intercept 1
  l1.intercept = 1 →
  -- Necessary condition
  (parallel l1 l2 → l1.slope = l2.slope) ∧
  -- Not sufficient condition
  ∃ l2 : Line, l1.slope = l2.slope ∧ ¬(parallel l1 l2) :=
by
  sorry

end NUMINAMATH_CALUDE_slopes_equal_necessary_not_sufficient_for_parallel_l2931_293185


namespace NUMINAMATH_CALUDE_box_side_face_area_l2931_293114

/-- Represents a rectangular box with length, width, and height -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box -/
def volume (b : Box) : ℝ := b.length * b.width * b.height

/-- Calculates the area of the top face of a box -/
def topFaceArea (b : Box) : ℝ := b.length * b.width

/-- Calculates the area of the front face of a box -/
def frontFaceArea (b : Box) : ℝ := b.width * b.height

/-- Calculates the area of the side face of a box -/
def sideFaceArea (b : Box) : ℝ := b.length * b.height

theorem box_side_face_area (b : Box) 
  (h1 : volume b = 192)
  (h2 : frontFaceArea b = (1/2) * topFaceArea b)
  (h3 : topFaceArea b = (3/2) * sideFaceArea b) :
  sideFaceArea b = 32 := by
  sorry

end NUMINAMATH_CALUDE_box_side_face_area_l2931_293114


namespace NUMINAMATH_CALUDE_product_equality_implies_sum_l2931_293152

theorem product_equality_implies_sum (m n : ℝ) : 
  (m^2 + 4*m + 5) * (n^2 - 2*n + 6) = 5 → 2*m + 3*n = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_implies_sum_l2931_293152


namespace NUMINAMATH_CALUDE_hyperbola_curve_is_hyperbola_l2931_293144

/-- A curve defined by x = cos^2 u and y = sin^4 u for real u -/
def HyperbolaCurve : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ u : ℝ, p.1 = Real.cos u ^ 2 ∧ p.2 = Real.sin u ^ 4}

/-- The curve defined by HyperbolaCurve is a hyperbola -/
theorem hyperbola_curve_is_hyperbola : 
  ∃ a b c d e f : ℝ, a ≠ 0 ∧ (a * b > 0 ∨ a * b < 0) ∧
  ∀ p : ℝ × ℝ, p ∈ HyperbolaCurve ↔ 
    a * p.1^2 + b * p.2^2 + c * p.1 * p.2 + d * p.1 + e * p.2 + f = 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_curve_is_hyperbola_l2931_293144


namespace NUMINAMATH_CALUDE_jenny_investment_l2931_293157

theorem jenny_investment (total : ℝ) (ratio : ℝ) (real_estate : ℝ) : 
  total = 220000 →
  ratio = 7 →
  real_estate = ratio * (total / (ratio + 1)) →
  real_estate = 192500 := by
sorry

end NUMINAMATH_CALUDE_jenny_investment_l2931_293157


namespace NUMINAMATH_CALUDE_second_square_area_is_676_l2931_293171

/-- An isosceles right triangle with inscribed squares -/
structure TriangleWithSquares where
  /-- Side length of the first inscribed square -/
  a : ℝ
  /-- Area of the first inscribed square is 169 -/
  h_area : a^2 = 169

/-- The area of the second inscribed square -/
def second_square_area (t : TriangleWithSquares) : ℝ :=
  (2 * t.a)^2

theorem second_square_area_is_676 (t : TriangleWithSquares) :
  second_square_area t = 676 := by
  sorry

end NUMINAMATH_CALUDE_second_square_area_is_676_l2931_293171


namespace NUMINAMATH_CALUDE_race_finish_order_l2931_293113

def race_order : List Nat := [1, 7, 9, 10, 8, 11, 2, 5, 3, 4, 6, 12]

theorem race_finish_order :
  ∀ (finish : Nat → Nat),
  (∀ n, n ∈ race_order → finish n ∈ Finset.range 13) →
  (∀ n, n ∈ race_order → ∃ k, n * (finish n) = 13 * k + 1) →
  (∀ n m, n ≠ m → n ∈ race_order → m ∈ race_order → finish n ≠ finish m) →
  (∀ n, n ∈ race_order → finish n = (List.indexOf n race_order).succ) :=
by sorry

#check race_finish_order

end NUMINAMATH_CALUDE_race_finish_order_l2931_293113


namespace NUMINAMATH_CALUDE_mod_equivalence_l2931_293138

theorem mod_equivalence (n : ℕ) : 
  185 * 944 ≡ n [ZMOD 60] → 0 ≤ n → n < 60 → n = 40 := by
sorry

end NUMINAMATH_CALUDE_mod_equivalence_l2931_293138


namespace NUMINAMATH_CALUDE_square_of_sum_product_l2931_293155

theorem square_of_sum_product (a b c d A : ℤ) 
  (h1 : a^2 + A = b^2) (h2 : c^2 + A = d^2) : 
  ∃ n : ℕ, 2 * (a + b) * (c + d) * (a * c + b * d - A) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_product_l2931_293155


namespace NUMINAMATH_CALUDE_square_and_fourth_power_mod_eight_l2931_293153

theorem square_and_fourth_power_mod_eight (n : ℤ) :
  (Even n → n ^ 2 % 8 = 0 ∨ n ^ 2 % 8 = 4) ∧
  (Odd n → n ^ 2 % 8 = 1) ∧
  (Odd n → n ^ 4 % 8 = 1) := by
  sorry

end NUMINAMATH_CALUDE_square_and_fourth_power_mod_eight_l2931_293153


namespace NUMINAMATH_CALUDE_gcd_204_85_l2931_293102

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l2931_293102


namespace NUMINAMATH_CALUDE_red_pencils_count_l2931_293104

theorem red_pencils_count (total_packs : ℕ) (normal_red_per_pack : ℕ) (special_packs : ℕ) (extra_red_per_special : ℕ) : 
  total_packs = 15 → 
  normal_red_per_pack = 1 → 
  special_packs = 3 → 
  extra_red_per_special = 2 → 
  total_packs * normal_red_per_pack + special_packs * extra_red_per_special = 21 := by
sorry

end NUMINAMATH_CALUDE_red_pencils_count_l2931_293104


namespace NUMINAMATH_CALUDE_bees_flew_in_l2931_293141

/-- Given an initial number of bees in a hive and a total number of bees after more flew in,
    this theorem proves that the number of bees that flew in is equal to the difference
    between the total and initial number of bees. -/
theorem bees_flew_in (initial_bees total_bees : ℕ) 
    (h1 : initial_bees = 16) 
    (h2 : total_bees = 26) : 
  total_bees - initial_bees = 10 := by
  sorry

#check bees_flew_in

end NUMINAMATH_CALUDE_bees_flew_in_l2931_293141


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2931_293129

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}
def B : Set ℝ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2931_293129


namespace NUMINAMATH_CALUDE_prime_divides_mn_minus_one_l2931_293119

theorem prime_divides_mn_minus_one (m n p : ℕ) 
  (h_prime : Nat.Prime p)
  (h_order : m < n ∧ n < p)
  (h_div_m : p ∣ m^2 + 1)
  (h_div_n : p ∣ n^2 + 1) :
  p ∣ m * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_mn_minus_one_l2931_293119


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2931_293190

theorem inequality_system_solution :
  {x : ℝ | 2 + x > 7 - 4*x ∧ x < (4 + x) / 2} = {x : ℝ | 1 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2931_293190


namespace NUMINAMATH_CALUDE_circle_radius_is_three_l2931_293115

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 10*y + 32 = 0

/-- The radius of a circle given by its equation -/
def CircleRadius (eq : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem circle_radius_is_three :
  CircleRadius CircleEquation = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_three_l2931_293115


namespace NUMINAMATH_CALUDE_linear_function_properties_l2931_293194

theorem linear_function_properties (m k b : ℝ) (h1 : m > 1) 
  (h2 : k * m + b = 1) (h3 : -k + b = m) : k < 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2931_293194


namespace NUMINAMATH_CALUDE_unique_solution_l2931_293124

theorem unique_solution : ∃! n : ℝ, 7 * n - 15 = 2 * n + 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2931_293124


namespace NUMINAMATH_CALUDE_power_function_through_point_l2931_293108

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = 8) : 
  f 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2931_293108


namespace NUMINAMATH_CALUDE_corner_sum_implies_bottom_right_l2931_293189

/-- Represents a 24 by 24 grid containing numbers 1 to 576 -/
def Grid := Fin 24 → Fin 24 → Nat

/-- Checks if a given number is in the grid -/
def in_grid (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 576

/-- Defines a valid 24 by 24 grid -/
def is_valid_grid (g : Grid) : Prop :=
  ∀ i j, in_grid (g i j) ∧ g i j = i * 24 + j + 1

/-- Represents an 8 by 8 square within the grid -/
structure Square (g : Grid) where
  top_left : Fin 24 × Fin 24
  h_valid : top_left.1 + 7 < 24 ∧ top_left.2 + 7 < 24

/-- Gets the corner values of an 8 by 8 square -/
def corner_values (g : Grid) (s : Square g) : Fin 4 → Nat
| 0 => g s.top_left.1 s.top_left.2
| 1 => g s.top_left.1 (s.top_left.2 + 7)
| 2 => g (s.top_left.1 + 7) s.top_left.2
| 3 => g (s.top_left.1 + 7) (s.top_left.2 + 7)
| _ => 0

/-- The main theorem -/
theorem corner_sum_implies_bottom_right (g : Grid) (s : Square g) :
  is_valid_grid g →
  (corner_values g s 0 + corner_values g s 1 + corner_values g s 2 + corner_values g s 3 = 1646) →
  corner_values g s 3 = 499 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_implies_bottom_right_l2931_293189


namespace NUMINAMATH_CALUDE_obtuse_angle_range_l2931_293116

def vector_AB (x : ℝ) : ℝ × ℝ := (x, 2*x)
def vector_AC (x : ℝ) : ℝ × ℝ := (-3*x, 2)

def is_obtuse_angle (x : ℝ) : Prop :=
  let dot_product := (vector_AB x).1 * (vector_AC x).1 + (vector_AB x).2 * (vector_AC x).2
  dot_product < 0 ∧ x ≠ -1/3

def range_of_x : Set ℝ :=
  {x | x < -1/3 ∨ (-1/3 < x ∧ x < 0) ∨ x > 4/3}

theorem obtuse_angle_range :
  ∀ x, is_obtuse_angle x ↔ x ∈ range_of_x :=
sorry

end NUMINAMATH_CALUDE_obtuse_angle_range_l2931_293116


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l2931_293173

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = b * I) → 
  (∃ c : ℝ, (z + 2)^2 - 8*I = c * I) → 
  z = -2*I :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l2931_293173


namespace NUMINAMATH_CALUDE_insect_count_in_lab_l2931_293135

/-- Given a total number of insect legs and the number of legs per insect, 
    calculates the number of insects. -/
def count_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Theorem stating that given 30 total insect legs and 6 legs per insect, 
    there are 5 insects in the laboratory. -/
theorem insect_count_in_lab : count_insects 30 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_insect_count_in_lab_l2931_293135


namespace NUMINAMATH_CALUDE_mariels_dogs_count_l2931_293177

/-- The number of dogs Mariel is walking -/
def mariels_dogs : ℕ := 5

/-- The number of dogs the other walker has -/
def other_walkers_dogs : ℕ := 3

/-- The number of legs each dog has -/
def dog_legs : ℕ := 4

/-- The number of legs each human has -/
def human_legs : ℕ := 2

/-- The total number of legs tangled in leashes -/
def total_legs : ℕ := 36

/-- The number of dog walkers -/
def num_walkers : ℕ := 2

theorem mariels_dogs_count :
  mariels_dogs * dog_legs + 
  other_walkers_dogs * dog_legs + 
  num_walkers * human_legs = total_legs := by sorry

end NUMINAMATH_CALUDE_mariels_dogs_count_l2931_293177


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2931_293179

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ (-1/2 * x^2 + 2*x > -m*x)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2931_293179


namespace NUMINAMATH_CALUDE_temperature_difference_l2931_293154

/-- The difference between the highest and lowest temperatures of the day -/
theorem temperature_difference (highest lowest : ℤ) (h1 : highest = 1) (h2 : lowest = -9) :
  highest - lowest = 10 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l2931_293154


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2931_293164

open Real

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ w x y z : ℝ, w > 0 → x > 0 → y > 0 → z > 0 → w * x = y * z →
    (f w)^2 + (f x)^2 / (f (y^2) + f (z^2)) = (w^2 + x^2) / (y^2 + z^2)

/-- The main theorem stating the form of functions satisfying the equation -/
theorem functional_equation_solution (f : ℝ → ℝ) 
    (h1 : ∀ x, x > 0 → f x > 0) 
    (h2 : SatisfiesEquation f) : 
    ∀ x, x > 0 → (f x = x ∨ f x = 1 / x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2931_293164


namespace NUMINAMATH_CALUDE_divisible_by_512_l2931_293127

theorem divisible_by_512 (n : ℤ) (h : Odd n) :
  ∃ k : ℤ, n^12 - n^8 - n^4 + 1 = 512 * k := by
sorry

end NUMINAMATH_CALUDE_divisible_by_512_l2931_293127


namespace NUMINAMATH_CALUDE_heath_carrot_planting_rate_l2931_293128

/-- Proves that Heath planted an average of 3000 carrots per hour over the weekend --/
theorem heath_carrot_planting_rate :
  let total_rows : ℕ := 400
  let first_half_rows : ℕ := 200
  let second_half_rows : ℕ := 200
  let plants_per_row_first_half : ℕ := 275
  let plants_per_row_second_half : ℕ := 325
  let hours_first_half : ℕ := 15
  let hours_second_half : ℕ := 25

  let total_plants : ℕ := first_half_rows * plants_per_row_first_half + 
                          second_half_rows * plants_per_row_second_half
  let total_hours : ℕ := hours_first_half + hours_second_half

  (total_plants : ℚ) / (total_hours : ℚ) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_heath_carrot_planting_rate_l2931_293128


namespace NUMINAMATH_CALUDE_exact_selection_probability_l2931_293146

def num_forks : ℕ := 8
def num_spoons : ℕ := 8
def num_knives : ℕ := 8
def total_pieces : ℕ := num_forks + num_spoons + num_knives
def selected_pieces : ℕ := 6

def probability_exact_selection : ℚ :=
  (Nat.choose num_forks 2 * Nat.choose num_spoons 2 * Nat.choose num_knives 2) /
  Nat.choose total_pieces selected_pieces

theorem exact_selection_probability :
  probability_exact_selection = 2744 / 16825 := by
  sorry

#eval probability_exact_selection

end NUMINAMATH_CALUDE_exact_selection_probability_l2931_293146


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2931_293188

theorem complex_equation_solution :
  ∃ (z : ℂ), (4 : ℂ) - 3 * Complex.I * z = (2 : ℂ) + 5 * Complex.I * z ∧ z = -(1/4) * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2931_293188


namespace NUMINAMATH_CALUDE_polynomial_roots_l2931_293139

theorem polynomial_roots : ∃ (p : ℝ → ℝ), 
  (∀ x, p x = 6 * x^4 + 19 * x^3 - 51 * x^2 + 20 * x) ∧ 
  (p 0 = 0) ∧ 
  (p (1/2) = 0) ∧ 
  (p (4/3) = 0) ∧ 
  (p (-5) = 0) := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2931_293139


namespace NUMINAMATH_CALUDE_ant_count_approximation_l2931_293149

/-- Calculates the approximate number of ants in a rectangular field -/
def approximate_ant_count (width_feet : ℝ) (length_feet : ℝ) (ants_per_sq_inch : ℝ) : ℝ :=
  let width_inches := width_feet * 12
  let length_inches := length_feet * 12
  let area_sq_inches := width_inches * length_inches
  area_sq_inches * ants_per_sq_inch

/-- Theorem stating that the number of ants in the given field is approximately 59 million -/
theorem ant_count_approximation :
  let field_width := 250
  let field_length := 330
  let ants_density := 5
  let calculated_count := approximate_ant_count field_width field_length ants_density
  abs (calculated_count - 59000000) / 59000000 < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ant_count_approximation_l2931_293149


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l2931_293126

def lowest_price : ℝ := 10
def highest_price : ℝ := 17

theorem percentage_increase_proof :
  (highest_price - lowest_price) / lowest_price * 100 = 70 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l2931_293126


namespace NUMINAMATH_CALUDE_waiter_customers_l2931_293120

/-- The number of customers a waiter served before the lunch rush -/
def customers_before_rush : ℕ := 29

/-- The number of additional customers during the lunch rush -/
def additional_customers : ℕ := 20

/-- The number of customers who didn't leave a tip -/
def customers_no_tip : ℕ := 34

/-- The number of customers who left a tip -/
def customers_with_tip : ℕ := 15

theorem waiter_customers :
  customers_before_rush + additional_customers =
  customers_no_tip + customers_with_tip :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l2931_293120


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2931_293180

def is_valid_number (a b : Nat) : Prop :=
  a ≤ 9 ∧ b ≤ 9 ∧ (100000 * a + 19880 + b) % 12 = 0

theorem count_valid_numbers :
  ∃ (S : Finset (Nat × Nat)),
    (∀ (p : Nat × Nat), p ∈ S ↔ is_valid_number p.1 p.2) ∧
    S.card = 9 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2931_293180


namespace NUMINAMATH_CALUDE_cupcake_cookie_price_ratio_l2931_293107

theorem cupcake_cookie_price_ratio :
  ∀ (cookie_price cupcake_price : ℚ),
    cookie_price > 0 →
    cupcake_price > 0 →
    5 * cookie_price + 3 * cupcake_price = 23 →
    4 * cookie_price + 4 * cupcake_price = 21 →
    cupcake_price / cookie_price = 13 / 29 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_cookie_price_ratio_l2931_293107


namespace NUMINAMATH_CALUDE_group_size_proof_l2931_293110

theorem group_size_proof (average_increase : ℝ) (new_weight : ℝ) (old_weight : ℝ) :
  average_increase = 6 →
  new_weight = 88 →
  old_weight = 40 →
  (average_increase * (new_weight - old_weight) / average_increase : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2931_293110


namespace NUMINAMATH_CALUDE_triangle_equilateral_condition_l2931_293195

/-- Triangle ABC with angles A, B, C and sides a, b, c -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- A triangle is equilateral if all its sides are equal -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- The theorem stating the conditions and conclusion about the triangle -/
theorem triangle_equilateral_condition (t : Triangle)
    (h1 : t.B = (t.A + t.C) / 2)  -- B is arithmetic mean of A and C
    (h2 : t.b ^ 2 = t.a * t.c)    -- b is geometric mean of a and c
    : t.isEquilateral := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_condition_l2931_293195


namespace NUMINAMATH_CALUDE_two_invariant_lines_l2931_293106

/-- Given a transformation from (x,y) to (x',y'), prove the existence of exactly two lines
    that both (x,y) and (x',y') lie on. -/
theorem two_invariant_lines 
  (x y x' y' : ℝ) 
  (h1 : x' = 3 * x + 2 * y + 1) 
  (h2 : y' = x + 4 * y - 3) :
  ∃! (L1 L2 : ℝ → ℝ → ℝ),
    (∀ x y, L1 x y = 0 ↔ L2 x y = 0 → L1 = L2) ∧
    (∀ x y, L1 x y = 0 → L1 x' y' = 0) ∧
    (∀ x y, L2 x y = 0 → L2 x' y' = 0) ∧
    L1 x y = x - y + 4 ∧
    L2 x y = 4 * x - 8 * y - 5 :=
by sorry

end NUMINAMATH_CALUDE_two_invariant_lines_l2931_293106


namespace NUMINAMATH_CALUDE_space_diagonal_probability_l2931_293198

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of space diagonals in a cube -/
def space_diagonals : ℕ := 4

/-- The probability of selecting two vertices that are endpoints of a space diagonal -/
def probability : ℚ := 1 / 7

/-- Theorem: The probability of randomly selecting two vertices of a cube that are endpoints
    of a space diagonal is 1/7, given that a cube has 8 vertices and 4 space diagonals. -/
theorem space_diagonal_probability :
  (space_diagonals * 2 : ℚ) / (cube_vertices.choose 2) = probability := by
  sorry


end NUMINAMATH_CALUDE_space_diagonal_probability_l2931_293198


namespace NUMINAMATH_CALUDE_three_sequence_inequality_l2931_293161

theorem three_sequence_inequality (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by sorry

end NUMINAMATH_CALUDE_three_sequence_inequality_l2931_293161


namespace NUMINAMATH_CALUDE_baseball_ratio_l2931_293191

theorem baseball_ratio (games_played : ℕ) (games_won : ℕ) 
  (h1 : games_played = 10) (h2 : games_won = 5) :
  (games_played : ℚ) / (games_played - games_won) = 2 := by
  sorry

end NUMINAMATH_CALUDE_baseball_ratio_l2931_293191


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l2931_293165

theorem complex_modulus_equation (n : ℝ) (hn : 0 < n) :
  Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 → n = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l2931_293165


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2931_293145

theorem sqrt_product_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (8 * x) = 30 * x * Real.sqrt (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2931_293145


namespace NUMINAMATH_CALUDE_unit_circle_sector_angle_l2931_293192

/-- In a unit circle, a sector with area 1 has a central angle of 2 radians -/
theorem unit_circle_sector_angle (r : ℝ) (area : ℝ) (angle : ℝ) : 
  r = 1 → area = 1 → angle = 2 * area / r → angle = 2 := by sorry

end NUMINAMATH_CALUDE_unit_circle_sector_angle_l2931_293192


namespace NUMINAMATH_CALUDE_parabola_intersections_l2931_293184

-- Define the parabola
def W (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line x = 4
def L (x : ℝ) : Prop := x = 4

-- Define points A and B
def A : ℝ × ℝ := (4, 4)
def B : ℝ × ℝ := (4, -4)

-- Define point P
structure Point (x₀ y₀ : ℝ) : Prop :=
  (on_parabola : W x₀ y₀)
  (x_constraint : x₀ < 4)
  (y_constraint : y₀ ≥ 0)

-- Define the area of triangle PAB
def area_PAB (x₀ y₀ : ℝ) : ℝ := 4 * (4 - x₀)

-- Define the perpendicularity condition
def perp_condition (x₀ y₀ : ℝ) : Prop :=
  (4 - y₀^2/4)^2 = (4 - y₀) * (4 + y₀)

-- Define the area of triangle PMN
def area_PMN (y₀ : ℝ) : ℝ := y₀^2

theorem parabola_intersections 
  (x₀ y₀ : ℝ) (p : Point x₀ y₀) :
  (area_PAB x₀ y₀ = 4 → x₀ = 3 ∧ y₀ = 2 * Real.sqrt 3) ∧
  (perp_condition x₀ y₀ → Real.sqrt ((4 - x₀)^2 + (4 - y₀)^2) = 4 * Real.sqrt 2) ∧
  (area_PMN y₀ = area_PAB x₀ y₀ → area_PMN y₀ = 8) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersections_l2931_293184


namespace NUMINAMATH_CALUDE_correct_division_result_l2931_293163

theorem correct_division_result (x : ℚ) (h : 9 - x = 3) : 96 / x = 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_result_l2931_293163


namespace NUMINAMATH_CALUDE_triangle_inequality_l2931_293100

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2931_293100


namespace NUMINAMATH_CALUDE_three_true_propositions_l2931_293118

-- Define reciprocals
def reciprocals (x y : ℝ) : Prop := x * y = 1

-- Define triangle congruence and area
def triangle_congruent (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry
def triangle_area (t : Set ℝ × Set ℝ) : ℝ := sorry

-- Define the quadratic equation
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0

theorem three_true_propositions :
  (∀ x y : ℝ, reciprocals x y → x * y = 1) ∧
  (∃ t1 t2 : Set ℝ × Set ℝ, triangle_area t1 = triangle_area t2 ∧ ¬ triangle_congruent t1 t2) ∧
  (∀ m : ℝ, ¬ has_real_roots m → m > 1) :=
by sorry

end NUMINAMATH_CALUDE_three_true_propositions_l2931_293118


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2931_293103

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, 1 < a ∧ a < 2 → a^2 - 3*a ≤ 0) ∧
  (∃ a, a^2 - 3*a ≤ 0 ∧ ¬(1 < a ∧ a < 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2931_293103


namespace NUMINAMATH_CALUDE_vertical_asymptote_at_three_halves_l2931_293131

-- Define the rational function
def f (x : ℚ) : ℚ := (2 * x + 3) / (6 * x - 9)

-- Theorem statement
theorem vertical_asymptote_at_three_halves :
  ∃ (ε : ℚ), ∀ (δ : ℚ), δ > 0 → ε > 0 → 
    ∀ (x : ℚ), 0 < |x - (3/2)| ∧ |x - (3/2)| < δ → |f x| > ε :=
sorry

end NUMINAMATH_CALUDE_vertical_asymptote_at_three_halves_l2931_293131


namespace NUMINAMATH_CALUDE_smallest_positive_omega_l2931_293166

theorem smallest_positive_omega : ∃ ω : ℝ, ω > 0 ∧
  (∀ x : ℝ, Real.sin (ω * x - Real.pi / 4) = Real.cos (ω * (x - Real.pi / 2))) ∧
  (∀ ω' : ℝ, ω' > 0 → 
    (∀ x : ℝ, Real.sin (ω' * x - Real.pi / 4) = Real.cos (ω' * (x - Real.pi / 2))) → 
    ω ≤ ω') ∧
  ω = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_omega_l2931_293166


namespace NUMINAMATH_CALUDE_tv_sets_b_is_30_l2931_293181

/-- The number of electronic shops in the Naza market -/
def num_shops : ℕ := 5

/-- The average number of TV sets in each shop -/
def average_tv_sets : ℕ := 48

/-- The number of TV sets in shop a -/
def tv_sets_a : ℕ := 20

/-- The number of TV sets in shop c -/
def tv_sets_c : ℕ := 60

/-- The number of TV sets in shop d -/
def tv_sets_d : ℕ := 80

/-- The number of TV sets in shop e -/
def tv_sets_e : ℕ := 50

/-- Theorem: The number of TV sets in shop b is 30 -/
theorem tv_sets_b_is_30 : 
  num_shops * average_tv_sets - (tv_sets_a + tv_sets_c + tv_sets_d + tv_sets_e) = 30 := by
  sorry

end NUMINAMATH_CALUDE_tv_sets_b_is_30_l2931_293181


namespace NUMINAMATH_CALUDE_custom_op_result_l2931_293122

-- Define the custom operation
def customOp (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

-- State the theorem
theorem custom_op_result : customOp (customOp 7 5) 4 = 42 + 1/33 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l2931_293122


namespace NUMINAMATH_CALUDE_definite_integral_x_plus_two_cubed_ln_squared_l2931_293162

open Real MeasureTheory

theorem definite_integral_x_plus_two_cubed_ln_squared :
  ∫ x in (-1)..(0), (x + 2)^3 * (log (x + 2))^2 = 4 * (log 2)^2 - 2 * log 2 + 15/32 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_x_plus_two_cubed_ln_squared_l2931_293162


namespace NUMINAMATH_CALUDE_sequence_general_term_l2931_293137

def S (n : ℕ+) : ℚ := 2 * n.val ^ 2 + n.val

def a (n : ℕ+) : ℚ := 4 * n.val - 1

theorem sequence_general_term (n : ℕ+) : 
  (∀ k : ℕ+, S k - S (k - 1) = a k) ∧ S 1 = a 1 := by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2931_293137


namespace NUMINAMATH_CALUDE_golden_ratio_pentagon_l2931_293174

theorem golden_ratio_pentagon (θ : Real) : 
  θ = 108 * Real.pi / 180 →  -- Interior angle of a regular pentagon
  2 * Real.sin (18 * Real.pi / 180) = (Real.sqrt 5 - 1) / 2 →
  Real.sin θ / Real.sin (36 * Real.pi / 180) = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_pentagon_l2931_293174


namespace NUMINAMATH_CALUDE_alice_bob_games_l2931_293109

/-- The number of players in the league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of games two specific players play together -/
def games_together : ℕ := 210

/-- The total number of possible game combinations -/
def total_combinations : ℕ := Nat.choose total_players players_per_game

theorem alice_bob_games :
  games_together = Nat.choose (total_players - 2) (players_per_game - 2) :=
by sorry

#check alice_bob_games

end NUMINAMATH_CALUDE_alice_bob_games_l2931_293109


namespace NUMINAMATH_CALUDE_derivative_sin_squared_minus_cos_squared_l2931_293169

theorem derivative_sin_squared_minus_cos_squared (x : ℝ) :
  (deriv (fun x => Real.sin x ^ 2 - Real.cos x ^ 2)) x = 2 * Real.sin (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_derivative_sin_squared_minus_cos_squared_l2931_293169


namespace NUMINAMATH_CALUDE_first_day_distance_l2931_293187

theorem first_day_distance (total_distance : ℝ) (days : ℕ) (ratio : ℝ) 
  (h1 : total_distance = 378)
  (h2 : days = 6)
  (h3 : ratio = 1/2) :
  (total_distance * (1 - ratio) / (1 - ratio^days)) = 192 :=
sorry

end NUMINAMATH_CALUDE_first_day_distance_l2931_293187


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l2931_293148

theorem quadratic_roots_transformation (a b : ℝ) (r₁ r₂ : ℝ) : 
  r₁^2 + a*r₁ + b = 0 → 
  r₂^2 + a*r₂ + b = 0 → 
  ∃ t : ℝ, (r₁^2 + 2*r₁*r₂ + r₂^2)^2 + (ab - a^2)*(r₁^2 + 2*r₁*r₂ + r₂^2) + t = 0 ∧ 
           (r₁*r₂*(r₁ + r₂))^2 + (ab - a^2)*(r₁*r₂*(r₁ + r₂)) + t = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l2931_293148


namespace NUMINAMATH_CALUDE_prob_at_least_one_karnataka_is_five_sixths_l2931_293170

/-- The probability of selecting at least one student from Karnataka -/
def prob_at_least_one_karnataka : ℚ :=
  let total_students : ℕ := 10
  let maharashtra_students : ℕ := 4
  let goa_students : ℕ := 3
  let karnataka_students : ℕ := 3
  let students_to_select : ℕ := 4
  1 - (Nat.choose (total_students - karnataka_students) students_to_select : ℚ) / 
      (Nat.choose total_students students_to_select : ℚ)

/-- Theorem stating that the probability of selecting at least one student from Karnataka is 5/6 -/
theorem prob_at_least_one_karnataka_is_five_sixths :
  prob_at_least_one_karnataka = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_karnataka_is_five_sixths_l2931_293170


namespace NUMINAMATH_CALUDE_complex_square_simplification_l2931_293175

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l2931_293175


namespace NUMINAMATH_CALUDE_haley_marbles_l2931_293168

/-- The number of marbles Haley had, given the number of boys and marbles per boy -/
def total_marbles (num_boys : ℕ) (marbles_per_boy : ℕ) : ℕ :=
  num_boys * marbles_per_boy

/-- Theorem stating that Haley had 99 marbles -/
theorem haley_marbles : total_marbles 11 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_haley_marbles_l2931_293168


namespace NUMINAMATH_CALUDE_circle_radius_l2931_293134

/-- The radius of a circle given its area and a modified area formula -/
theorem circle_radius (k : ℝ) (A : ℝ) (h1 : k = 4) (h2 : A = 225 * Real.pi) :
  ∃ (r : ℝ), k * Real.pi * r^2 = A ∧ r = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2931_293134


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2931_293136

theorem arithmetic_mean_of_fractions (x b c : ℝ) (hx : x ≠ 0) (hc : c ≠ 0) :
  ((x + b) / (c * x) + (x - b) / (c * x)) / 2 = 1 / c :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2931_293136


namespace NUMINAMATH_CALUDE_age_difference_constant_l2931_293199

/-- Represents a person's age --/
structure Person where
  age : ℕ

/-- Represents the current year --/
def CurrentYear : Type := Unit

/-- Represents a future year --/
structure FutureYear where
  yearsFromNow : ℕ

/-- The age difference between two people --/
def ageDifference (p1 p2 : Person) : ℕ :=
  if p1.age ≥ p2.age then p1.age - p2.age else p2.age - p1.age

/-- The age of a person after a number of years --/
def ageAfterYears (p : Person) (y : ℕ) : ℕ :=
  p.age + y

theorem age_difference_constant
  (a : ℕ)
  (n : ℕ)
  (xiaoShen : Person)
  (xiaoWang : Person)
  (h1 : xiaoShen.age = a)
  (h2 : xiaoWang.age = a - 8)
  : ageDifference
      { age := ageAfterYears xiaoShen (n + 3) }
      { age := ageAfterYears xiaoWang (n + 3) } = 8 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_constant_l2931_293199


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2931_293147

/-- Given an arithmetic sequence {aₙ}, prove that its common difference is 2 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  (h_sum : a 1 + a 5 = 10)  -- Given condition
  (h_S4 : (a 1 + a 2 + a 3 + a 4) = 16)  -- Given condition for S₄
  : a 2 - a 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2931_293147


namespace NUMINAMATH_CALUDE_tan_value_implies_cosine_sine_ratio_l2931_293196

theorem tan_value_implies_cosine_sine_ratio 
  (α : Real) 
  (h : Real.tan α = 1/3) : 
  (Real.cos α)^2 - 2*(Real.sin α)^2 = 7/9 * (Real.cos α)^2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_implies_cosine_sine_ratio_l2931_293196


namespace NUMINAMATH_CALUDE_is_solution_l2931_293121

-- Define the function f(x) = x^2 + x + C
def f (C : ℝ) (x : ℝ) : ℝ := x^2 + x + C

-- State the theorem
theorem is_solution (C : ℝ) : 
  ∀ x : ℝ, deriv (f C) x = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_is_solution_l2931_293121


namespace NUMINAMATH_CALUDE_cube_of_negative_double_l2931_293159

theorem cube_of_negative_double (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_double_l2931_293159


namespace NUMINAMATH_CALUDE_extremum_values_l2931_293158

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- Theorem stating that if f(x) has an extremum of 10 at x = 1, then a = 4 and b = -11 -/
theorem extremum_values (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b 1 ≥ f a b x) ∧ 
  (f a b 1 = 10) →
  a = 4 ∧ b = -11 := by
sorry

end NUMINAMATH_CALUDE_extremum_values_l2931_293158


namespace NUMINAMATH_CALUDE_average_hamburgers_per_day_l2931_293123

-- Define the total number of hamburgers sold
def total_hamburgers : ℕ := 49

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the average number of hamburgers sold per day
def average_hamburgers : ℚ := total_hamburgers / days_in_week

-- Theorem statement
theorem average_hamburgers_per_day :
  average_hamburgers = 7 := by sorry

end NUMINAMATH_CALUDE_average_hamburgers_per_day_l2931_293123


namespace NUMINAMATH_CALUDE_modified_lottery_win_probability_l2931_293130

/-- The number of balls for the MegaBall drawing -/
def megaBallCount : ℕ := 30

/-- The number of balls for the WinnerBalls drawing -/
def winnerBallCount : ℕ := 46

/-- The number of WinnerBalls picked -/
def pickedWinnerBallCount : ℕ := 5

/-- The probability of winning the modified lottery game -/
def winProbability : ℚ := 1 / 34321980

theorem modified_lottery_win_probability :
  winProbability = 1 / (megaBallCount * (Nat.choose winnerBallCount pickedWinnerBallCount)) :=
by sorry

end NUMINAMATH_CALUDE_modified_lottery_win_probability_l2931_293130


namespace NUMINAMATH_CALUDE_sixth_student_matches_l2931_293140

/-- Represents the number of matches played by each student -/
structure MatchCounts where
  student1 : ℕ
  student2 : ℕ
  student3 : ℕ
  student4 : ℕ
  student5 : ℕ
  student6 : ℕ

/-- The total number of matches in a complete tournament with 6 players -/
def totalMatches : ℕ := 15

/-- Theorem stating that if 5 students have played 5, 4, 3, 2, and 1 matches respectively,
    then the 6th student must have played 3 matches -/
theorem sixth_student_matches (mc : MatchCounts) : 
  mc.student1 = 5 ∧ 
  mc.student2 = 4 ∧ 
  mc.student3 = 3 ∧ 
  mc.student4 = 2 ∧ 
  mc.student5 = 1 ∧
  (mc.student1 + mc.student2 + mc.student3 + mc.student4 + mc.student5 + mc.student6 = 2 * totalMatches) →
  mc.student6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sixth_student_matches_l2931_293140


namespace NUMINAMATH_CALUDE_sqrt_two_plus_one_squared_l2931_293143

theorem sqrt_two_plus_one_squared : (Real.sqrt 2 + 1)^2 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_one_squared_l2931_293143


namespace NUMINAMATH_CALUDE_weight_loss_duration_l2931_293132

/-- Represents the weight loss pattern over a 5-month cycle -/
structure WeightLossPattern :=
  (month1 : Int)
  (month2 : Int)
  (month3 : Int)
  (month4and5 : Int)

/-- Calculates the time needed to reach the target weight -/
def timeToReachTarget (initialWeight : Int) (pattern : WeightLossPattern) (targetWeight : Int) : Int :=
  sorry

/-- The theorem statement -/
theorem weight_loss_duration :
  let initialWeight := 222
  let pattern := WeightLossPattern.mk (-12) (-6) 2 (-8)
  let targetWeight := 170
  timeToReachTarget initialWeight pattern targetWeight = 6 :=
sorry

end NUMINAMATH_CALUDE_weight_loss_duration_l2931_293132


namespace NUMINAMATH_CALUDE_solve_equation_l2931_293101

theorem solve_equation (x : ℝ) : 5 * x + 3 = 10 * x - 17 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2931_293101


namespace NUMINAMATH_CALUDE_display_configurations_l2931_293150

/-- The number of holes in the row -/
def num_holes : ℕ := 8

/-- The number of holes that can display at a time -/
def num_display : ℕ := 3

/-- The number of possible states for each displaying hole -/
def num_states : ℕ := 2

/-- A function that calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of possible configurations -/
def total_configurations : ℕ := sorry

theorem display_configurations :
  total_configurations = choose (num_holes - num_display + 1) num_display * num_states ^ num_display :=
sorry

end NUMINAMATH_CALUDE_display_configurations_l2931_293150


namespace NUMINAMATH_CALUDE_num_friends_is_four_l2931_293178

/-- The number of friends who volunteered with James to plant flowers -/
def num_friends : ℕ :=
  let total_flowers : ℕ := 200
  let days : ℕ := 2
  let james_flowers_per_day : ℕ := 20
  (total_flowers - james_flowers_per_day * days) / (james_flowers_per_day * days)

theorem num_friends_is_four : num_friends = 4 := by
  sorry

end NUMINAMATH_CALUDE_num_friends_is_four_l2931_293178


namespace NUMINAMATH_CALUDE_im_z_squared_gt_two_iff_xy_gt_one_l2931_293125

/-- For a complex number z, Im(z^2) > 2 if and only if the product of its real and imaginary parts is greater than 1 -/
theorem im_z_squared_gt_two_iff_xy_gt_one (z : ℂ) :
  Complex.im (z^2) > 2 ↔ Complex.re z * Complex.im z > 1 := by
sorry

end NUMINAMATH_CALUDE_im_z_squared_gt_two_iff_xy_gt_one_l2931_293125


namespace NUMINAMATH_CALUDE_other_number_in_product_l2931_293151

theorem other_number_in_product (P w n : ℕ) : 
  P % 2^4 = 0 →
  P % 3^3 = 0 →
  P % 13^3 = 0 →
  P = n * w →
  w > 0 →
  w = 468 →
  (∀ w' : ℕ, w' > 0 ∧ w' < w → ¬(P % w' = 0)) →
  n = 2028 := by
sorry

end NUMINAMATH_CALUDE_other_number_in_product_l2931_293151


namespace NUMINAMATH_CALUDE_greatest_fourth_term_of_arithmetic_sequence_l2931_293186

theorem greatest_fourth_term_of_arithmetic_sequence 
  (a : ℕ) 
  (d : ℕ) 
  (sum_eq_65 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 65) 
  (a_positive : a > 0) :
  ∀ (b : ℕ) (e : ℕ), 
    b > 0 → 
    b + (b + e) + (b + 2*e) + (b + 3*e) + (b + 4*e) = 65 → 
    b + 3*e ≤ a + 3*d :=
by sorry

end NUMINAMATH_CALUDE_greatest_fourth_term_of_arithmetic_sequence_l2931_293186


namespace NUMINAMATH_CALUDE_blocks_used_in_tower_l2931_293160

/-- Given that Randy initially had 59 blocks and now has 23 blocks left,
    prove that he used 36 blocks to build the tower. -/
theorem blocks_used_in_tower (initial_blocks : ℕ) (remaining_blocks : ℕ) 
  (h1 : initial_blocks = 59)
  (h2 : remaining_blocks = 23) : 
  initial_blocks - remaining_blocks = 36 := by
  sorry

end NUMINAMATH_CALUDE_blocks_used_in_tower_l2931_293160


namespace NUMINAMATH_CALUDE_olivia_baseball_cards_l2931_293156

/-- The number of decks of baseball cards Olivia bought -/
def baseball_decks : ℕ :=
  let basketball_packs : ℕ := 2
  let basketball_price : ℕ := 3
  let baseball_price : ℕ := 4
  let initial_money : ℕ := 50
  let change : ℕ := 24
  let total_spent : ℕ := initial_money - change
  let basketball_cost : ℕ := basketball_packs * basketball_price
  let baseball_cost : ℕ := total_spent - basketball_cost
  baseball_cost / baseball_price

theorem olivia_baseball_cards : baseball_decks = 5 := by
  sorry

end NUMINAMATH_CALUDE_olivia_baseball_cards_l2931_293156


namespace NUMINAMATH_CALUDE_sequence_decreasing_l2931_293182

/-- Given real numbers a and b such that b > a > 1, define the sequence x_n as follows:
    x_n = 2^n * (b^(1/2^n) - a^(1/2^n))
    This theorem states that the sequence is decreasing. -/
theorem sequence_decreasing (a b : ℝ) (h1 : b > a) (h2 : a > 1) :
  ∀ n : ℕ, (2^n * (b^(1/(2^n)) - a^(1/(2^n)))) > (2^(n+1) * (b^(1/(2^(n+1))) - a^(1/(2^(n+1))))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_decreasing_l2931_293182


namespace NUMINAMATH_CALUDE_yoga_studio_women_count_l2931_293112

theorem yoga_studio_women_count :
  let num_men : ℕ := 8
  let avg_weight_men : ℚ := 190
  let avg_weight_women : ℚ := 120
  let total_people : ℕ := 14
  let avg_weight_all : ℚ := 160
  let num_women : ℕ := total_people - num_men
  (num_men : ℚ) * avg_weight_men + (num_women : ℚ) * avg_weight_women = (total_people : ℚ) * avg_weight_all →
  num_women = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_yoga_studio_women_count_l2931_293112


namespace NUMINAMATH_CALUDE_bet_winnings_ratio_l2931_293142

def initial_amount : ℕ := 400
def final_amount : ℕ := 1200

def amount_won : ℕ := final_amount - initial_amount

theorem bet_winnings_ratio :
  (amount_won : ℚ) / initial_amount = 2 := by sorry

end NUMINAMATH_CALUDE_bet_winnings_ratio_l2931_293142


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2931_293183

theorem trigonometric_identity :
  Real.sin (-1071 * π / 180) * Real.sin (99 * π / 180) +
  Real.sin (-171 * π / 180) * Real.sin (-261 * π / 180) +
  Real.tan (-1089 * π / 180) * Real.tan (-540 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2931_293183


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2931_293117

theorem quadratic_equation_solution : ∃ y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2931_293117


namespace NUMINAMATH_CALUDE_certain_number_problem_l2931_293193

theorem certain_number_problem (x : ℝ) : 
  (0.8 * 40 = (4/5) * x + 16) → x = 20 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2931_293193


namespace NUMINAMATH_CALUDE_x_value_proof_l2931_293111

theorem x_value_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x^2 / y = 2)
  (h2 : y^2 / z = 5)
  (h3 : z^2 / x = 7) : 
  x = (2800 : ℝ)^(1/7) := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l2931_293111


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2931_293172

def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)

theorem f_derivative_at_zero : 
  deriv f 0 = 720 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2931_293172
