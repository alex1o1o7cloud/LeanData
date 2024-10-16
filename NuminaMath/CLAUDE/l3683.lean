import Mathlib

namespace NUMINAMATH_CALUDE_trig_expression_approx_value_l3683_368375

theorem trig_expression_approx_value : 
  let expr := (2 * Real.sin (30 * π / 180) * Real.cos (10 * π / 180) + 
               3 * Real.cos (150 * π / 180) * Real.cos (110 * π / 180)) /
              (4 * Real.sin (40 * π / 180) * Real.cos (20 * π / 180) + 
               5 * Real.cos (140 * π / 180) * Real.cos (100 * π / 180))
  ∃ ε > 0, abs (expr - 0.465) < ε := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_approx_value_l3683_368375


namespace NUMINAMATH_CALUDE_alice_bob_meet_time_l3683_368352

def circlePoints : ℕ := 15
def aliceMove : ℕ := 4
def bobMove : ℕ := 8  -- Equivalent clockwise movement

theorem alice_bob_meet_time :
  let relativeMove := (bobMove - aliceMove) % circlePoints
  ∃ n : ℕ, n > 0 ∧ (n * relativeMove) % circlePoints = 0 ∧
  ∀ m : ℕ, 0 < m ∧ m < n → (m * relativeMove) % circlePoints ≠ 0 ∧
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_time_l3683_368352


namespace NUMINAMATH_CALUDE_oprah_band_total_weight_l3683_368325

/-- Represents the Oprah Winfrey High School marching band -/
structure MarchingBand where
  trumpet_count : ℕ
  clarinet_count : ℕ
  trombone_count : ℕ
  tuba_count : ℕ
  drum_count : ℕ
  trumpet_weight : ℕ
  clarinet_weight : ℕ
  trombone_weight : ℕ
  tuba_weight : ℕ
  drum_weight : ℕ

/-- Calculates the total weight carried by the marching band -/
def total_weight (band : MarchingBand) : ℕ :=
  band.trumpet_count * band.trumpet_weight +
  band.clarinet_count * band.clarinet_weight +
  band.trombone_count * band.trombone_weight +
  band.tuba_count * band.tuba_weight +
  band.drum_count * band.drum_weight

/-- The Oprah Winfrey High School marching band configuration -/
def oprah_band : MarchingBand := {
  trumpet_count := 6
  clarinet_count := 9
  trombone_count := 8
  tuba_count := 3
  drum_count := 2
  trumpet_weight := 5
  clarinet_weight := 5
  trombone_weight := 10
  tuba_weight := 20
  drum_weight := 15
}

/-- Theorem stating that the total weight carried by the Oprah Winfrey High School marching band is 245 pounds -/
theorem oprah_band_total_weight :
  total_weight oprah_band = 245 := by
  sorry

end NUMINAMATH_CALUDE_oprah_band_total_weight_l3683_368325


namespace NUMINAMATH_CALUDE_smallest_non_existent_count_l3683_368383

/-- The number of terms in the arithmetic progression -/
def progression_length : ℕ := 1999

/-- 
  Counts the number of integer terms in an arithmetic progression 
  with 'progression_length' terms and common difference 1/m
-/
def count_integer_terms (m : ℕ) : ℕ :=
  1 + (progression_length - 1) / m

/-- 
  Checks if there exists an arithmetic progression of 'progression_length' 
  real numbers containing exactly n integers
-/
def exists_progression_with_n_integers (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ count_integer_terms m = n

theorem smallest_non_existent_count : 
  (∀ k < 70, exists_progression_with_n_integers k) ∧
  ¬exists_progression_with_n_integers 70 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_existent_count_l3683_368383


namespace NUMINAMATH_CALUDE_set_associativity_l3683_368388

theorem set_associativity (A B C : Set α) : 
  (A ∪ (B ∪ C) = (A ∪ B) ∪ C) ∧ (A ∩ (B ∩ C) = (A ∩ B) ∩ C) := by
  sorry

end NUMINAMATH_CALUDE_set_associativity_l3683_368388


namespace NUMINAMATH_CALUDE_jack_jogging_speed_l3683_368307

-- Define the given conditions
def melt_time : ℚ := 10 / 60  -- 10 minutes converted to hours
def num_blocks : ℕ := 16
def block_length : ℚ := 1 / 8  -- in miles

-- Define the total distance
def total_distance : ℚ := num_blocks * block_length

-- Define the required speed
def required_speed : ℚ := total_distance / melt_time

-- Theorem statement
theorem jack_jogging_speed :
  required_speed = 12 := by sorry

end NUMINAMATH_CALUDE_jack_jogging_speed_l3683_368307


namespace NUMINAMATH_CALUDE_dice_surface_sum_l3683_368323

/-- The number of dice in the arrangement -/
def num_dice : Nat := 2012

/-- The sum of points on all faces of a single die -/
def die_sum : Nat := 21

/-- The sum of points on opposite faces of a die -/
def opposite_faces_sum : Nat := 7

/-- A value representing the number of points on one end face of the first die -/
def X : Fin 6 := sorry

/-- The sum of points on the surface of the arranged dice -/
def surface_sum : Nat := 28175 + 2 * X.val

theorem dice_surface_sum :
  surface_sum = num_dice * die_sum - (num_dice - 1) * opposite_faces_sum + 2 * X.val :=
by sorry

end NUMINAMATH_CALUDE_dice_surface_sum_l3683_368323


namespace NUMINAMATH_CALUDE_last_even_number_in_sequence_l3683_368355

theorem last_even_number_in_sequence (n : ℕ) : 
  (4 * (n * (n + 1) * (2 * n + 1)) / 6 = 560) → n = 7 :=
by sorry

end NUMINAMATH_CALUDE_last_even_number_in_sequence_l3683_368355


namespace NUMINAMATH_CALUDE_foil_covered_prism_width_l3683_368395

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (d : PrismDimensions) : ℝ := d.length * d.width * d.height

/-- Represents the properties of the inner prism -/
structure InnerPrism where
  dimensions : PrismDimensions
  cubeCount : ℕ

/-- Represents the properties of the foil-covered prism -/
structure FoilCoveredPrism where
  innerPrism : InnerPrism
  foilThickness : ℝ

theorem foil_covered_prism_width
  (p : FoilCoveredPrism)
  (h1 : p.innerPrism.cubeCount = 128)
  (h2 : p.innerPrism.dimensions.width = 2 * p.innerPrism.dimensions.length)
  (h3 : p.innerPrism.dimensions.width = 2 * p.innerPrism.dimensions.height)
  (h4 : volume p.innerPrism.dimensions = p.innerPrism.cubeCount)
  (h5 : p.foilThickness = 1) :
  p.innerPrism.dimensions.width + 2 * p.foilThickness = 10 := by
  sorry


end NUMINAMATH_CALUDE_foil_covered_prism_width_l3683_368395


namespace NUMINAMATH_CALUDE_standard_form_of_given_equation_l3683_368316

/-- Standard form of a quadratic equation -/
def standard_form (a b c : ℝ) : ℝ → Prop :=
  fun x ↦ a * x^2 + b * x + c = 0

/-- The given quadratic equation -/
def given_equation (x : ℝ) : Prop :=
  3 * x^2 + 1 = 7 * x

/-- Theorem stating that the standard form of 3x^2 + 1 = 7x is 3x^2 - 7x + 1 = 0 -/
theorem standard_form_of_given_equation :
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, given_equation x ↔ standard_form a b c x) ∧ 
  a = 3 ∧ b = -7 ∧ c = 1 :=
sorry

end NUMINAMATH_CALUDE_standard_form_of_given_equation_l3683_368316


namespace NUMINAMATH_CALUDE_ship_ratio_proof_l3683_368300

theorem ship_ratio_proof (total_people : ℕ) (first_ship : ℕ) (ratio : ℚ) : 
  total_people = 847 →
  first_ship = 121 →
  first_ship + first_ship * ratio + first_ship * ratio^2 = total_people →
  ratio = 2 := by
sorry

end NUMINAMATH_CALUDE_ship_ratio_proof_l3683_368300


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3683_368333

def expansion (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  (1 - 2*x)^5 = a₀ + 2*a₁*x + 4*a₂*x^2 + 8*a₃*x^3 + 16*a₄*x^4 + 32*a₅*x^5

theorem sum_of_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x, expansion x a₀ a₁ a₂ a₃ a₄ a₅) : 
  a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3683_368333


namespace NUMINAMATH_CALUDE_rational_inequality_l3683_368359

theorem rational_inequality (x : ℝ) : (x^2 - 9) / (x + 3) < 0 ↔ x ∈ Set.Ioo (-3 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_rational_inequality_l3683_368359


namespace NUMINAMATH_CALUDE_card_sequence_return_l3683_368385

theorem card_sequence_return (n : ℕ) (hn : n > 0) : 
  Nat.totient (2 * n - 1) ≤ 2 * n - 2 := by
  sorry

end NUMINAMATH_CALUDE_card_sequence_return_l3683_368385


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3683_368350

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (48 - b) + c / (72 - c) = 9) :
  4 / (36 - a) + 6 / (48 - b) + 9 / (72 - c) = 13 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3683_368350


namespace NUMINAMATH_CALUDE_inequality_solution_l3683_368391

theorem inequality_solution : 
  ∃! (n : ℕ), n ≥ 3 ∧ 
  (∀ (x : ℝ), x ≥ 3 → 
    (Real.sqrt (5 * x - 11) - Real.sqrt (5 * x^2 - 21 * x + 21) ≥ 5 * x^2 - 26 * x + 32) →
    x = n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3683_368391


namespace NUMINAMATH_CALUDE_unique_prime_solution_l3683_368363

theorem unique_prime_solution : 
  ∀ p q r : ℕ, 
    Prime p ∧ Prime q ∧ Prime r →
    p^2 + 1 = 74 * (q^2 + r^2) →
    p = 31 ∧ q = 2 ∧ r = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l3683_368363


namespace NUMINAMATH_CALUDE_minimum_m_value_l3683_368342

theorem minimum_m_value (m : ℕ) : 
  (∀ n : ℕ, n ≥ 2 → (n.factorial : ℝ) ^ (2 / (n * (n - 1))) < m) ↔ m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_minimum_m_value_l3683_368342


namespace NUMINAMATH_CALUDE_composite_sum_divisibility_l3683_368327

theorem composite_sum_divisibility (s : ℕ) (h1 : s ≥ 4) :
  (∃ (a b c d : ℕ+), (a + b + c + d : ℕ) = s ∧ s ∣ (a * b * c + a * b * d + a * c * d + b * c * d)) ↔
  ¬ Nat.Prime s :=
sorry

end NUMINAMATH_CALUDE_composite_sum_divisibility_l3683_368327


namespace NUMINAMATH_CALUDE_area_of_region_l3683_368386

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 10 ∧ 
   A = Real.pi * (Real.sqrt ((x + 1)^2 + (y - 2)^2)) ^ 2 ∧
   x^2 + y^2 + 2*x - 4*y = 5) := by sorry

end NUMINAMATH_CALUDE_area_of_region_l3683_368386


namespace NUMINAMATH_CALUDE_commodity_cost_proof_l3683_368311

def total_cost (price1 price2 : ℕ) : ℕ := price1 + price2

theorem commodity_cost_proof (price1 price2 : ℕ) 
  (h1 : price1 = 477)
  (h2 : price1 = price2 + 127) :
  total_cost price1 price2 = 827 := by
  sorry

end NUMINAMATH_CALUDE_commodity_cost_proof_l3683_368311


namespace NUMINAMATH_CALUDE_chicken_buying_equation_l3683_368384

/-- Represents the scenario of a group buying chickens -/
structure ChickenBuying where
  people : ℕ
  cost : ℕ

/-- The excess when each person contributes 9 coins -/
def excess (cb : ChickenBuying) : ℤ :=
  9 * cb.people - cb.cost

/-- The shortage when each person contributes 6 coins -/
def shortage (cb : ChickenBuying) : ℤ :=
  cb.cost - 6 * cb.people

/-- The theorem representing the chicken buying scenario -/
theorem chicken_buying_equation (cb : ChickenBuying) 
  (h1 : excess cb = 11) 
  (h2 : shortage cb = 16) : 
  9 * cb.people - 11 = 6 * cb.people + 16 := by
  sorry

end NUMINAMATH_CALUDE_chicken_buying_equation_l3683_368384


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l3683_368340

theorem sqrt_meaningful_iff_geq_two (a : ℝ) : ∃ x : ℝ, x^2 = a - 2 ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l3683_368340


namespace NUMINAMATH_CALUDE_cylinder_diameter_from_sphere_surface_area_l3683_368368

theorem cylinder_diameter_from_sphere_surface_area (r_sphere : ℝ) (h_cylinder : ℝ) :
  r_sphere = 3 →
  h_cylinder = 6 →
  4 * Real.pi * r_sphere^2 = 2 * Real.pi * (6 / 2) * h_cylinder →
  6 = 2 * (6 / 2) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_diameter_from_sphere_surface_area_l3683_368368


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3683_368373

theorem vector_equation_solution :
  ∃ (a b : ℚ),
    (2 : ℚ) * a + (-2 : ℚ) * b = 10 ∧
    (3 : ℚ) * a + (5 : ℚ) * b = -8 ∧
    a = 17/8 ∧ b = -23/8 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3683_368373


namespace NUMINAMATH_CALUDE_complex_fraction_real_minus_imag_l3683_368320

theorem complex_fraction_real_minus_imag (z : ℂ) (a b : ℝ) : 
  z = 5 / (-3 - Complex.I) → 
  a = z.re → 
  b = z.im → 
  a - b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_minus_imag_l3683_368320


namespace NUMINAMATH_CALUDE_max_volume_rectangular_solid_l3683_368313

/-- Given a rectangular solid where the sum of all edges is 18 meters,
    and the length is twice the width, the maximum volume is 3 cubic meters. -/
theorem max_volume_rectangular_solid :
  ∃ (w l h : ℝ),
    w > 0 ∧ l > 0 ∧ h > 0 ∧
    l = 2 * w ∧
    4 * w + 4 * l + 4 * h = 18 ∧
    ∀ (w' l' h' : ℝ),
      w' > 0 → l' > 0 → h' > 0 →
      l' = 2 * w' →
      4 * w' + 4 * l' + 4 * h' = 18 →
      w * l * h ≥ w' * l' * h' ∧
    w * l * h = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_rectangular_solid_l3683_368313


namespace NUMINAMATH_CALUDE_winning_strategy_l3683_368354

/-- Represents the colors used in the chocolate table -/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a cell in the chocolate table -/
structure Cell :=
  (row : Nat)
  (col : Nat)
  (color : Color)

/-- Represents the chocolate table -/
def ChocolateTable (n : Nat) := Array (Array Cell)

/-- Creates a colored n × n chocolate table -/
def createTable (n : Nat) : ChocolateTable n := sorry

/-- Removes a cell from the chocolate table -/
def removeCell (table : ChocolateTable n) (cell : Cell) : ChocolateTable n := sorry

/-- Checks if a 3 × 1 or 1 × 3 rectangle contains one of each color -/
def validRectangle (rect : Array Cell) : Bool := sorry

/-- Checks if the table can be partitioned into valid rectangles -/
def canPartition (table : ChocolateTable n) : Bool := sorry

theorem winning_strategy 
  (n : Nat) 
  (h1 : n > 3) 
  (h2 : ¬(3 ∣ n)) : 
  ∃ (cell : Cell), cell.color ≠ Color.Red ∧ 
    ¬(canPartition (removeCell (createTable n) cell)) := by sorry

end NUMINAMATH_CALUDE_winning_strategy_l3683_368354


namespace NUMINAMATH_CALUDE_natasha_quarters_l3683_368357

theorem natasha_quarters : 
  ∃ (n : ℕ), 8 < n ∧ n < 80 ∧ 
  n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1 ∧
  (∀ (m : ℕ), 8 < m ∧ m < 80 ∧ 
   m % 4 = 1 ∧ m % 5 = 1 ∧ m % 6 = 1 → m = n) ∧
  n = 61 :=
by sorry

end NUMINAMATH_CALUDE_natasha_quarters_l3683_368357


namespace NUMINAMATH_CALUDE_max_pen_area_l3683_368329

/-- The maximum area of a rectangular pen with one side against a wall,
    given 30 meters of fencing for the other three sides. -/
theorem max_pen_area (total_fence : ℝ) (h_total_fence : total_fence = 30) :
  ∃ (width height : ℝ),
    width > 0 ∧
    height > 0 ∧
    width + 2 * height = total_fence ∧
    ∀ (w h : ℝ), w > 0 → h > 0 → w + 2 * h = total_fence →
      w * h ≤ width * height ∧
      width * height = 112 :=
by sorry

end NUMINAMATH_CALUDE_max_pen_area_l3683_368329


namespace NUMINAMATH_CALUDE_solution_set_min_value_min_value_achieved_l3683_368344

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part 1: Theorem for the solution set of f(2x) ≤ f(x + 1)
theorem solution_set (x : ℝ) : f (2 * x) ≤ f (x + 1) ↔ 0 ≤ x ∧ x ≤ 1 := by sorry

-- Part 2: Theorem for the minimum value of f(a²) + f(b²)
theorem min_value (a b : ℝ) (h : a + b = 2) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (a' b' : ℝ), a' + b' = 2 → f (a' ^ 2) + f (b' ^ 2) ≥ m := by sorry

-- Corollary: The minimum is achieved when a = b = 1
theorem min_value_achieved (a b : ℝ) (h : a + b = 2) : 
  f (a ^ 2) + f (b ^ 2) = 2 ↔ a = 1 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_min_value_achieved_l3683_368344


namespace NUMINAMATH_CALUDE_problem_solution_l3683_368309

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 6*x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- Define the line passing through (2,3)
def line_through_2_3 (x y : ℝ) : Prop := x = 2 ∨ 3*x + 4*y = 18

-- Define the line x - y + a = 0
def line_a (x y a : ℝ) : Prop := x - y + a = 0

-- Define perpendicularity
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem problem_solution :
  -- Part 1
  (∀ x y : ℝ, curve x y ∧ (x = 0 ∨ y = 0) → circle_C x y) ∧
  -- Part 2
  (∃ x y : ℝ, line_through_2_3 x y ∧ circle_C x y ∧
    ∃ x' y' : ℝ, line_through_2_3 x' y' ∧ circle_C x' y' ∧
    (x - x')^2 + (y - y')^2 = 32) ∧
  -- Part 3
  (∀ a : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ,
    line_a x₁ y₁ a ∧ line_a x₂ y₂ a ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    perpendicular x₁ y₁ x₂ y₂) → a = -1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3683_368309


namespace NUMINAMATH_CALUDE_rearranged_triple_divisible_by_27_l3683_368318

/-- Given a natural number, rearranging its digits to get a number
    that is three times the original results in a number divisible by 27. -/
theorem rearranged_triple_divisible_by_27 (n m : ℕ) :
  (∃ (f : ℕ → ℕ), f n = m) →  -- n and m have the same digits (rearranged)
  m = 3 * n →                 -- m is three times n
  27 ∣ m :=                   -- m is divisible by 27
by sorry

end NUMINAMATH_CALUDE_rearranged_triple_divisible_by_27_l3683_368318


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3683_368364

theorem quadratic_equation_roots (a b c : ℝ) : 
  (∀ x, a * x * (x + 1) + b * x * (x + 2) + c * (x + 1) * (x + 2) = 0 ↔ x = 1 ∨ x = 2) →
  a + b + c = 2 →
  a = 12 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3683_368364


namespace NUMINAMATH_CALUDE_parabola_vertex_on_axis_l3683_368349

/-- A parabola with equation y = x^2 - kx + k - 1 has its vertex on a coordinate axis if and only if k = 2 or k = 0 -/
theorem parabola_vertex_on_axis (k : ℝ) : 
  (∃ x y : ℝ, (y = x^2 - k*x + k - 1) ∧ 
    ((x = 0 ∧ y = k - 1) ∨ (y = 0 ∧ x = k/2)) ∧
    (∀ x' y' : ℝ, y' = x'^2 - k*x' + k - 1 → y' ≥ y)) ↔ 
  (k = 2 ∨ k = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_axis_l3683_368349


namespace NUMINAMATH_CALUDE_salary_restoration_l3683_368315

theorem salary_restoration (original_salary : ℝ) (reduced_salary : ℝ) : 
  reduced_salary = original_salary * (1 - 0.5) → 
  reduced_salary * 2 = original_salary :=
by sorry

end NUMINAMATH_CALUDE_salary_restoration_l3683_368315


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l3683_368339

theorem least_number_with_remainder (n : ℕ) : n = 266 ↔ 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 33 ≠ 2 ∨ m % 8 ≠ 2)) ∧ 
  n % 33 = 2 ∧ n % 8 = 2 :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l3683_368339


namespace NUMINAMATH_CALUDE_five_dollar_neg_three_l3683_368304

-- Define the $ operation
def dollar_op (a b : Int) : Int := a * (b - 1) + a * b

-- Theorem statement
theorem five_dollar_neg_three : dollar_op 5 (-3) = -35 := by
  sorry

end NUMINAMATH_CALUDE_five_dollar_neg_three_l3683_368304


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l3683_368317

theorem min_value_sum_of_squares (u v w : ℝ) 
  (h_pos_u : u > 0) (h_pos_v : v > 0) (h_pos_w : w > 0)
  (h_sum_squares : u^2 + v^2 + w^2 = 8) :
  (u^4 / 9) + (v^4 / 16) + (w^4 / 25) ≥ 32/25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l3683_368317


namespace NUMINAMATH_CALUDE_sum_squares_quadratic_roots_l3683_368312

theorem sum_squares_quadratic_roots : 
  let a := 1
  let b := -10
  let c := 9
  let s₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let s₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  s₁^2 + s₂^2 = 82 :=
by sorry

end NUMINAMATH_CALUDE_sum_squares_quadratic_roots_l3683_368312


namespace NUMINAMATH_CALUDE_max_value_theorem_l3683_368387

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 3) : 
  2*a*b + 2*b*c*Real.sqrt 3 ≤ 6 ∧ ∃ a b c, 2*a*b + 2*b*c*Real.sqrt 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3683_368387


namespace NUMINAMATH_CALUDE_incorrect_number_value_l3683_368366

theorem incorrect_number_value (n : ℕ) (initial_avg correct_avg incorrect_value : ℚ) 
  (h1 : n = 10)
  (h2 : initial_avg = 46)
  (h3 : correct_avg = 51)
  (h4 : incorrect_value = 25) :
  let correct_value := n * correct_avg - (n * initial_avg - incorrect_value)
  correct_value = 75 := by sorry

end NUMINAMATH_CALUDE_incorrect_number_value_l3683_368366


namespace NUMINAMATH_CALUDE_turtles_on_sand_l3683_368306

theorem turtles_on_sand (x : ℚ) : 
  let nest1 : ℚ := x
  let nest2 : ℚ := 2 * x
  let swept_nest1 : ℚ := (1 / 4) * nest1
  let swept_nest2 : ℚ := (3 / 7) * nest2
  let remaining_nest1 : ℚ := nest1 - swept_nest1
  let remaining_nest2 : ℚ := nest2 - swept_nest2
  remaining_nest1 + remaining_nest2 = (53 / 28) * x :=
by sorry

end NUMINAMATH_CALUDE_turtles_on_sand_l3683_368306


namespace NUMINAMATH_CALUDE_H_surjective_l3683_368399

def H (x : ℝ) : ℝ := |x^2 + 2*x + 1| - |x^2 - 2*x + 1|

theorem H_surjective : Function.Surjective H := by sorry

end NUMINAMATH_CALUDE_H_surjective_l3683_368399


namespace NUMINAMATH_CALUDE_calories_left_for_dinner_l3683_368347

def daily_calorie_limit : ℕ := 2200
def breakfast_calories : ℕ := 353
def lunch_calories : ℕ := 885
def snack_calories : ℕ := 130

theorem calories_left_for_dinner :
  daily_calorie_limit - (breakfast_calories + lunch_calories + snack_calories) = 832 := by
  sorry

end NUMINAMATH_CALUDE_calories_left_for_dinner_l3683_368347


namespace NUMINAMATH_CALUDE_min_sum_a_b_l3683_368343

theorem min_sum_a_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 2 / b = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l3683_368343


namespace NUMINAMATH_CALUDE_fixed_equidistant_point_l3683_368377

-- Define the circles
variable (k₁ k₂ : Set (ℝ × ℝ))

-- Define the intersection point A
variable (A : ℝ × ℝ)

-- Define the particles P₁ and P₂ as functions of time
variable (P₁ P₂ : ℝ → ℝ × ℝ)

-- Define the constant angular speeds
variable (ω₁ ω₂ : ℝ)

-- Axioms
axiom circles_intersect : A ∈ k₁ ∩ k₂

axiom P₁_on_k₁ : ∀ t, P₁ t ∈ k₁
axiom P₂_on_k₂ : ∀ t, P₂ t ∈ k₂

axiom start_at_A : P₁ 0 = A ∧ P₂ 0 = A

axiom constant_speed : ∀ t, ‖(P₁ t).fst - (P₁ 0).fst‖ = ω₁ * t
                    ∧ ‖(P₂ t).fst - (P₂ 0).fst‖ = ω₂ * t

axiom same_direction : ω₁ * ω₂ > 0

axiom simultaneous_arrival : ∃ T > 0, P₁ T = A ∧ P₂ T = A

-- Theorem
theorem fixed_equidistant_point :
  ∃ P : ℝ × ℝ, ∀ t, ‖P - P₁ t‖ = ‖P - P₂ t‖ :=
sorry

end NUMINAMATH_CALUDE_fixed_equidistant_point_l3683_368377


namespace NUMINAMATH_CALUDE_ladder_slide_l3683_368382

theorem ladder_slide (ladder_length : ℝ) (initial_base : ℝ) (slip_distance : ℝ) :
  ladder_length = 25 →
  initial_base = 7 →
  slip_distance = 4 →
  ∃ (slide_distance : ℝ),
    slide_distance = 8 ∧
    (ladder_length ^ 2 = (initial_base + slide_distance) ^ 2 + (Real.sqrt (ladder_length ^ 2 - initial_base ^ 2) - slip_distance) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_ladder_slide_l3683_368382


namespace NUMINAMATH_CALUDE_expected_attacked_squares_l3683_368345

/-- The number of squares on a chessboard -/
def chessboardSquares : ℕ := 64

/-- The number of rooks placed on the chessboard -/
def numberOfRooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook -/
def probNotAttacked : ℚ := (49 : ℚ) / 64

/-- The expected number of squares under attack by three randomly placed rooks on a chessboard -/
def expectedAttackedSquares : ℚ := chessboardSquares * (1 - probNotAttacked ^ numberOfRooks)

/-- Theorem stating the expected number of squares under attack -/
theorem expected_attacked_squares :
  expectedAttackedSquares = 64 * (1 - (49/64)^3) :=
sorry

end NUMINAMATH_CALUDE_expected_attacked_squares_l3683_368345


namespace NUMINAMATH_CALUDE_circus_tent_seating_l3683_368361

theorem circus_tent_seating (total_capacity : ℕ) (num_sections : ℕ) : 
  total_capacity = 984 → num_sections = 4 → 
  (total_capacity / num_sections : ℕ) = 246 := by
  sorry

end NUMINAMATH_CALUDE_circus_tent_seating_l3683_368361


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3683_368336

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^2 + 3*x*y - 2*y^2 = 122 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3683_368336


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l3683_368338

theorem incorrect_observation_value 
  (n : ℕ) 
  (original_mean : ℝ) 
  (new_mean : ℝ) 
  (correct_value : ℝ) 
  (h_n : n = 50) 
  (h_original_mean : original_mean = 36) 
  (h_new_mean : new_mean = 36.02) 
  (h_correct_value : correct_value = 48) :
  ∃ (incorrect_value : ℝ), 
    n * new_mean = n * original_mean - incorrect_value + correct_value ∧ 
    incorrect_value = 47 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l3683_368338


namespace NUMINAMATH_CALUDE_square_ratio_side_lengths_l3683_368390

theorem square_ratio_side_lengths :
  let area_ratio : ℚ := 8 / 125
  let side_ratio : ℝ := Real.sqrt (area_ratio)
  let rationalized_ratio : ℝ := side_ratio * Real.sqrt 5 / Real.sqrt 5
  rationalized_ratio = 2 * Real.sqrt 10 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_lengths_l3683_368390


namespace NUMINAMATH_CALUDE_sample_size_is_75_l3683_368314

/-- Represents the sample size of a stratified sample -/
def sample_size (model_A_count : ℕ) (ratio_A ratio_B ratio_C : ℕ) : ℕ :=
  model_A_count * (ratio_A + ratio_B + ratio_C) / ratio_A

/-- Theorem stating that the sample size is 75 given the problem conditions -/
theorem sample_size_is_75 :
  sample_size 15 2 3 5 = 75 := by
  sorry

#eval sample_size 15 2 3 5

end NUMINAMATH_CALUDE_sample_size_is_75_l3683_368314


namespace NUMINAMATH_CALUDE_angle_addition_theorem_l3683_368362

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)
  (minutes_valid : minutes < 60)

/-- Addition of angles -/
def add_angles (a b : Angle) : Angle :=
  let total_minutes := a.minutes + b.minutes
  let extra_degrees := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  ⟨a.degrees + b.degrees + extra_degrees, remaining_minutes, by sorry⟩

theorem angle_addition_theorem :
  add_angles ⟨36, 28, by sorry⟩ ⟨25, 34, by sorry⟩ = ⟨62, 2, by sorry⟩ :=
by sorry

end NUMINAMATH_CALUDE_angle_addition_theorem_l3683_368362


namespace NUMINAMATH_CALUDE_final_worker_bees_count_l3683_368303

/-- Represents the state of the bee hive --/
structure BeeHive where
  workers : ℕ
  drones : ℕ
  queens : ℕ
  guards : ℕ

/-- Applies the series of events to the bee hive --/
def applyEvents (hive : BeeHive) : BeeHive :=
  let hive1 := { hive with 
    workers := hive.workers - 28,
    drones := hive.drones - 12,
    guards := hive.guards - 5 }
  let hive2 := { hive1 with 
    workers := hive1.workers - 30,
    guards := hive1.guards + 30 }
  let hive3 := { hive2 with 
    workers := hive2.workers + 15 }
  { hive3 with 
    workers := 0 }

/-- The theorem to be proved --/
theorem final_worker_bees_count (initialHive : BeeHive) 
  (h1 : initialHive.workers = 400)
  (h2 : initialHive.drones = 75)
  (h3 : initialHive.queens = 1)
  (h4 : initialHive.guards = 50) :
  (applyEvents initialHive).workers = 0 := by
  sorry

#check final_worker_bees_count

end NUMINAMATH_CALUDE_final_worker_bees_count_l3683_368303


namespace NUMINAMATH_CALUDE_seating_arrangements_l3683_368369

/-- Represents the number of people sitting around the table -/
def total_people : ℕ := 8

/-- Represents the number of people in the special block (leader, vice leader, recorder) -/
def special_block : ℕ := 3

/-- Represents the number of units to arrange (treating the special block as one unit) -/
def units_to_arrange : ℕ := total_people - special_block + 1

/-- Represents the number of ways to arrange the people within the special block -/
def internal_arrangements : ℕ := 2

/-- Calculates the number of unique circular arrangements for n elements -/
def circular_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The main theorem stating the number of unique seating arrangements -/
theorem seating_arrangements : 
  circular_arrangements units_to_arrange * internal_arrangements = 240 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3683_368369


namespace NUMINAMATH_CALUDE_janabel_widget_sales_l3683_368321

theorem janabel_widget_sales (n : ℕ) (a₁ : ℕ) (d : ℕ) : 
  n = 15 → a₁ = 2 → d = 2 → (n * (2 * a₁ + (n - 1) * d)) / 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_janabel_widget_sales_l3683_368321


namespace NUMINAMATH_CALUDE_sine_shift_and_stretch_l3683_368326

/-- Given a function f(x) = sin(x), prove that shifting it right by π/10 units
    and then stretching the x-coordinates by a factor of 2 results in
    the function g(x) = sin(1/2x - π/10) -/
theorem sine_shift_and_stretch (x : ℝ) :
  let f : ℝ → ℝ := λ t ↦ Real.sin t
  let shift : ℝ → ℝ := λ t ↦ t - π / 10
  let stretch : ℝ → ℝ := λ t ↦ t / 2
  let g : ℝ → ℝ := λ t ↦ Real.sin (1/2 * t - π / 10)
  f (stretch (shift x)) = g x := by
  sorry

end NUMINAMATH_CALUDE_sine_shift_and_stretch_l3683_368326


namespace NUMINAMATH_CALUDE_sandy_dog_puppies_l3683_368328

/-- The number of puppies Sandy now has -/
def total_puppies : ℕ := 12

/-- The number of puppies Sandy's friend gave her -/
def friend_puppies : ℕ := 4

/-- The number of puppies Sandy's dog initially had -/
def initial_puppies : ℕ := total_puppies - friend_puppies

theorem sandy_dog_puppies : initial_puppies = 8 := by
  sorry

end NUMINAMATH_CALUDE_sandy_dog_puppies_l3683_368328


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3683_368374

theorem complex_fraction_evaluation : 
  (((1 / 2) * (1 / 3) * (1 / 4) * (1 / 5) + (3 / 2) * (3 / 4) * (3 / 5)) / 
   ((1 / 2) * (2 / 3) * (2 / 5))) = 41 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3683_368374


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l3683_368392

/-- Represents a batsman's cricket performance -/
structure BatsmanStats where
  totalInnings : ℕ
  lastInningsScore : ℕ
  averageIncrease : ℚ
  notOutInnings : ℕ

/-- Calculates the average score of a batsman after their latest innings,
    considering 'not out' innings -/
def calculateAdjustedAverage (stats : BatsmanStats) : ℚ :=
  let totalRuns := stats.totalInnings * (stats.averageIncrease + 
    (stats.lastInningsScore / stats.totalInnings : ℚ))
  totalRuns / (stats.totalInnings - stats.notOutInnings : ℚ)

/-- Theorem stating that for a batsman with given statistics, 
    their adjusted average is approximately 88.64 -/
theorem batsman_average_theorem (stats : BatsmanStats) 
  (h1 : stats.totalInnings = 25)
  (h2 : stats.lastInningsScore = 150)
  (h3 : stats.averageIncrease = 3)
  (h4 : stats.notOutInnings = 3) :
  abs (calculateAdjustedAverage stats - 88.64) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_theorem_l3683_368392


namespace NUMINAMATH_CALUDE_repeating_not_necessarily_periodic_l3683_368396

/-- Definition of the sequence property --/
def has_repeating_property (a : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, ∃ t : ℕ, t > 0 ∧ ∀ n : ℕ, a k = a (k + n * t)

/-- Definition of periodicity --/
def is_periodic (a : ℕ → ℕ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ k : ℕ, a k = a (k + T)

/-- Theorem stating that a sequence with the repeating property is not necessarily periodic --/
theorem repeating_not_necessarily_periodic :
  ∃ a : ℕ → ℕ, has_repeating_property a ∧ ¬ is_periodic a := by
  sorry

end NUMINAMATH_CALUDE_repeating_not_necessarily_periodic_l3683_368396


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3683_368393

/-- If the quadratic x^2 + 6mx + 2m has exactly one real root, then m = 2/9 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x, x^2 + 6*m*x + 2*m = 0) → m = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3683_368393


namespace NUMINAMATH_CALUDE_f_has_minimum_l3683_368394

def f (x : ℝ) := |2*x + 1| - |x - 4|

theorem f_has_minimum : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) := by
  sorry

end NUMINAMATH_CALUDE_f_has_minimum_l3683_368394


namespace NUMINAMATH_CALUDE_smallest_double_multiple_of_2016_l3683_368301

def consecutive_double (n : ℕ) : ℕ :=
  n * 1001 + n

theorem smallest_double_multiple_of_2016 :
  ∀ A : ℕ, A < 288 → ¬(∃ k : ℕ, consecutive_double A = 2016 * k) ∧
  ∃ k : ℕ, consecutive_double 288 = 2016 * k :=
by sorry

end NUMINAMATH_CALUDE_smallest_double_multiple_of_2016_l3683_368301


namespace NUMINAMATH_CALUDE_at_least_one_chooses_probability_l3683_368367

-- Define the probabilities for Students A and B
def prob_A : ℚ := 1/3
def prob_B : ℚ := 1/4

-- Define the event that at least one student chooses the "Inequality Lecture"
def at_least_one_chooses : ℚ := 1 - (1 - prob_A) * (1 - prob_B)

-- Theorem statement
theorem at_least_one_chooses_probability :
  at_least_one_chooses = 1/2 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_chooses_probability_l3683_368367


namespace NUMINAMATH_CALUDE_article_cost_price_l3683_368398

/-- Given an article marked 15% above its cost price, sold at Rs. 462 with a discount of 25.603864734299517%, prove that the cost price of the article is Rs. 540. -/
theorem article_cost_price (cost_price : ℝ) : 
  let markup_percentage : ℝ := 0.15
  let selling_price : ℝ := 462
  let discount_percentage : ℝ := 25.603864734299517
  let marked_price : ℝ := cost_price * (1 + markup_percentage)
  let discounted_price : ℝ := marked_price * (1 - discount_percentage / 100)
  discounted_price = selling_price → cost_price = 540 := by
sorry

#eval (462 : ℚ) / (1 - 25.603864734299517 / 100) / 1.15

end NUMINAMATH_CALUDE_article_cost_price_l3683_368398


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_l3683_368356

theorem no_real_sqrt_negative : ∃ (a b c d : ℝ), 
  (a = (-3)^2 ∧ ∃ x : ℝ, x^2 = a) ∧ 
  (b = 0 ∧ ∃ x : ℝ, x^2 = b) ∧ 
  (c = 1/8 ∧ ∃ x : ℝ, x^2 = c) ∧ 
  (d = -6^3 ∧ ¬∃ x : ℝ, x^2 = d) :=
by sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_l3683_368356


namespace NUMINAMATH_CALUDE_pool_filling_cost_l3683_368310

-- Define the pool dimensions
def pool_depth : ℝ := 10
def pool_width : ℝ := 6
def pool_length : ℝ := 20

-- Define the conversion factor from cubic feet to liters
def cubic_feet_to_liters : ℝ := 25

-- Define the cost per liter of water
def cost_per_liter : ℝ := 3

-- Theorem statement
theorem pool_filling_cost :
  pool_depth * pool_width * pool_length * cubic_feet_to_liters * cost_per_liter = 90000 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_cost_l3683_368310


namespace NUMINAMATH_CALUDE_nina_money_problem_l3683_368376

theorem nina_money_problem (W : ℝ) (M : ℝ) :
  (10 * W = M) →
  (14 * (W - 1.75) = M) →
  M = 61.25 := by
sorry

end NUMINAMATH_CALUDE_nina_money_problem_l3683_368376


namespace NUMINAMATH_CALUDE_total_uniform_cost_is_355_l3683_368389

/-- Calculates the total cost of school uniforms for a student --/
def uniform_cost (num_uniforms : ℕ) (pants_cost : ℚ) (sock_cost : ℚ) : ℚ :=
  let shirt_cost := 2 * pants_cost
  let tie_cost := (1 / 5) * shirt_cost
  let single_uniform_cost := pants_cost + shirt_cost + tie_cost + sock_cost
  num_uniforms * single_uniform_cost

/-- Proves that the total cost of school uniforms for a student is $355 --/
theorem total_uniform_cost_is_355 :
  uniform_cost 5 20 3 = 355 := by
  sorry

#eval uniform_cost 5 20 3

end NUMINAMATH_CALUDE_total_uniform_cost_is_355_l3683_368389


namespace NUMINAMATH_CALUDE_expression_evaluation_l3683_368308

theorem expression_evaluation (a b : ℝ) (h1 : a = 6) (h2 : b = 2) :
  ((3 / (a + b))^2) * (a - b) = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3683_368308


namespace NUMINAMATH_CALUDE_additional_cartons_needed_l3683_368322

/-- Given the total required cartons, the number of strawberry cartons, and the number of blueberry cartons,
    prove that the additional cartons needed is equal to the total required minus the sum of strawberry and blueberry cartons. -/
theorem additional_cartons_needed
  (total_required : ℕ)
  (strawberry_cartons : ℕ)
  (blueberry_cartons : ℕ)
  (h : total_required = 42 ∧ strawberry_cartons = 2 ∧ blueberry_cartons = 7) :
  total_required - (strawberry_cartons + blueberry_cartons) = 33 :=
by sorry

end NUMINAMATH_CALUDE_additional_cartons_needed_l3683_368322


namespace NUMINAMATH_CALUDE_eighth_term_value_l3683_368351

/-- The nth term of a geometric sequence -/
def geometric_term (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r ^ (n - 1)

/-- The 8th term of the specific geometric sequence -/
def eighth_term : ℚ :=
  geometric_term 3 (3/2) 8

theorem eighth_term_value : eighth_term = 6561 / 128 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l3683_368351


namespace NUMINAMATH_CALUDE_new_cost_is_fifty_l3683_368378

/-- Represents the manufacturing cost and profit scenario for Crazy Eddie's key chains --/
structure KeyChainScenario where
  initialCost : ℝ
  initialProfitRate : ℝ
  newProfitRate : ℝ
  sellingPrice : ℝ

/-- Calculates the new manufacturing cost given a KeyChainScenario --/
def newManufacturingCost (scenario : KeyChainScenario) : ℝ :=
  scenario.sellingPrice * (1 - scenario.newProfitRate)

/-- Theorem stating that under the given conditions, the new manufacturing cost is $50 --/
theorem new_cost_is_fifty :
  ∀ (scenario : KeyChainScenario),
    scenario.initialCost = 70 ∧
    scenario.initialProfitRate = 0.3 ∧
    scenario.newProfitRate = 0.5 ∧
    scenario.sellingPrice = scenario.initialCost / (1 - scenario.initialProfitRate) →
    newManufacturingCost scenario = 50 := by
  sorry


end NUMINAMATH_CALUDE_new_cost_is_fifty_l3683_368378


namespace NUMINAMATH_CALUDE_max_value_of_f_l3683_368335

noncomputable def f (x a : ℝ) : ℝ := Real.cos x ^ 2 + a * Real.sin x + (5/8) * a + 1

theorem max_value_of_f (a : ℝ) :
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.pi / 2 → f y a ≤ f x a) →
  (a < 0 → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x a = (5/8) * a + 2) ∧
  (0 ≤ a ∧ a ≤ 2 → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x a = a^2 / 4 + (5/8) * a + 2) ∧
  (2 < a → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x a = (13/8) * a + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3683_368335


namespace NUMINAMATH_CALUDE_inequality_proof_l3683_368381

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3683_368381


namespace NUMINAMATH_CALUDE_square_area_error_l3683_368319

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l3683_368319


namespace NUMINAMATH_CALUDE_inverse_product_positive_implies_one_greater_than_one_l3683_368379

theorem inverse_product_positive_implies_one_greater_than_one 
  (a b c : ℝ) (h : (a⁻¹) * (b⁻¹) * (c⁻¹) > 0) : 
  (a > 1) ∨ (b > 1) ∨ (c > 1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_product_positive_implies_one_greater_than_one_l3683_368379


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_age_l3683_368337

/-- Represents the current age of the son -/
def sonAge : ℕ := 22

/-- Represents the age difference between the man and his son -/
def ageDifference : ℕ := 24

/-- Represents the number of years until the man's age is twice his son's age -/
def yearsUntilTwice : ℕ := 2

/-- Theorem stating that in 'yearsUntilTwice' years, the man's age will be twice his son's age -/
theorem mans_age_twice_sons_age :
  (sonAge + ageDifference + yearsUntilTwice) = 2 * (sonAge + yearsUntilTwice) := by
  sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_age_l3683_368337


namespace NUMINAMATH_CALUDE_evaluate_expression_l3683_368380

theorem evaluate_expression : -(18 / 3 * 8 - 80 + 4^2 * 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3683_368380


namespace NUMINAMATH_CALUDE_divisible_by_seven_l3683_368330

theorem divisible_by_seven (k : ℕ) : ∃ m : ℤ, 2^(6*k+1) + 3^(6*k+1) + 5^(6*k) + 1 = 7*m := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l3683_368330


namespace NUMINAMATH_CALUDE_expression_evaluation_l3683_368371

theorem expression_evaluation (a b : ℤ) (h1 : a = 1) (h2 : b = -2) : 
  -2*a - 2*b^2 + 3*a*b - b^3 = -8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3683_368371


namespace NUMINAMATH_CALUDE_alpha_plus_beta_eq_107_l3683_368341

theorem alpha_plus_beta_eq_107 :
  ∃ α β : ℝ, (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 64*x + 992) / (x^2 + 72*x - 2184)) →
  α + β = 107 := by
  sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_eq_107_l3683_368341


namespace NUMINAMATH_CALUDE_inequality_proof_l3683_368332

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (3 : ℝ) / (a^3 + b^3 + c^3) ≤ 1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ∧
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3683_368332


namespace NUMINAMATH_CALUDE_complement_of_union_l3683_368365

universe u

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {1,3,5,7}
def B : Set ℕ := {2,4,5}

theorem complement_of_union : 
  (U \ (A ∪ B)) = {6,8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3683_368365


namespace NUMINAMATH_CALUDE_total_weight_is_103_2_l3683_368397

/-- The total weight of all books owned by Sandy, Benny, and Tim -/
def total_weight : ℝ :=
  let sandy_books := 10
  let sandy_weight := 1.5
  let benny_books := 24
  let benny_weight := 1.2
  let tim_books := 33
  let tim_weight := 1.8
  sandy_books * sandy_weight + benny_books * benny_weight + tim_books * tim_weight

/-- Theorem stating that the total weight of all books is 103.2 pounds -/
theorem total_weight_is_103_2 : total_weight = 103.2 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_103_2_l3683_368397


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3683_368305

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 3003 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 3003 → 
  A + B + C ≤ X + Y + Z → A + B + C ≤ 117 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3683_368305


namespace NUMINAMATH_CALUDE_base_salary_minimum_l3683_368372

/-- The base salary of Tom's new sales job -/
def base_salary : ℝ := 45000

/-- The salary of Tom's previous job -/
def previous_salary : ℝ := 75000

/-- The commission percentage on each sale -/
def commission_percentage : ℝ := 0.15

/-- The price of each sale -/
def sale_price : ℝ := 750

/-- The minimum number of sales required to not lose money -/
def min_sales : ℝ := 266.67

theorem base_salary_minimum : 
  base_salary + min_sales * (commission_percentage * sale_price) ≥ previous_salary :=
sorry

end NUMINAMATH_CALUDE_base_salary_minimum_l3683_368372


namespace NUMINAMATH_CALUDE_sine_function_properties_l3683_368360

/-- Given a function y = a * sin(x) + 2b where a > 0, with maximum value 4 and minimum value 0,
    prove that a + b = 3 and the minimum positive period of y = b * sin(ax) is π -/
theorem sine_function_properties (a b : ℝ) (h_a_pos : a > 0)
  (h_max : ∀ x, a * Real.sin x + 2 * b ≤ 4)
  (h_min : ∀ x, a * Real.sin x + 2 * b ≥ 0)
  (h_max_achievable : ∃ x, a * Real.sin x + 2 * b = 4)
  (h_min_achievable : ∃ x, a * Real.sin x + 2 * b = 0) :
  (a + b = 3) ∧
  (∀ T > 0, (∀ x, b * Real.sin (a * x) = b * Real.sin (a * (x + T))) → T ≥ π) ∧
  (∀ x, b * Real.sin (a * x) = b * Real.sin (a * (x + π))) := by
  sorry


end NUMINAMATH_CALUDE_sine_function_properties_l3683_368360


namespace NUMINAMATH_CALUDE_jason_initial_money_l3683_368331

theorem jason_initial_money (jason_current : ℕ) (jason_earned : ℕ) 
  (h1 : jason_current = 63) 
  (h2 : jason_earned = 60) : 
  jason_current - jason_earned = 3 := by
sorry

end NUMINAMATH_CALUDE_jason_initial_money_l3683_368331


namespace NUMINAMATH_CALUDE_median_lengths_l3683_368358

/-- Given a triangle with sides a, b, and c, this theorem states the formulas for the lengths of the medians sa, sb, and sc. -/
theorem median_lengths (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (sa sb sc : ℝ),
    sa = Real.sqrt (2 * b^2 + 2 * c^2 - a^2) / 2 ∧
    sb = Real.sqrt (2 * a^2 + 2 * c^2 - b^2) / 2 ∧
    sc = Real.sqrt (2 * a^2 + 2 * b^2 - c^2) / 2 :=
by sorry


end NUMINAMATH_CALUDE_median_lengths_l3683_368358


namespace NUMINAMATH_CALUDE_sqrt_10_irrational_l3683_368370

theorem sqrt_10_irrational : Irrational (Real.sqrt 10) := by sorry

end NUMINAMATH_CALUDE_sqrt_10_irrational_l3683_368370


namespace NUMINAMATH_CALUDE_opposite_faces_sum_seven_l3683_368353

-- Define a type for the faces of a die
inductive DieFace : Type
  | one : DieFace
  | two : DieFace
  | three : DieFace
  | four : DieFace
  | five : DieFace
  | six : DieFace

-- Define a function to get the numeric value of a face
def faceValue : DieFace → Nat
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

-- Define a function to get the opposite face
def oppositeFace : DieFace → DieFace
  | DieFace.one => DieFace.six
  | DieFace.two => DieFace.five
  | DieFace.three => DieFace.four
  | DieFace.four => DieFace.three
  | DieFace.five => DieFace.two
  | DieFace.six => DieFace.one

-- Theorem: The sum of values on opposite faces is always 7
theorem opposite_faces_sum_seven (face : DieFace) :
  faceValue face + faceValue (oppositeFace face) = 7 := by
  sorry


end NUMINAMATH_CALUDE_opposite_faces_sum_seven_l3683_368353


namespace NUMINAMATH_CALUDE_number_of_hens_l3683_368324

theorem number_of_hens (total_heads total_feet num_hens num_cows : ℕ) 
  (total_heads_eq : total_heads = 48)
  (total_feet_eq : total_feet = 144)
  (min_hens : num_hens ≥ 10)
  (min_cows : num_cows ≥ 5)
  (total_animals_eq : num_hens + num_cows = total_heads)
  (total_feet_calc : 2 * num_hens + 4 * num_cows = total_feet) :
  num_hens = 24 := by
sorry

end NUMINAMATH_CALUDE_number_of_hens_l3683_368324


namespace NUMINAMATH_CALUDE_hyperbola_condition_l3683_368334

/-- A hyperbola is represented by an equation of the form ax²/p + by²/q = 1, 
    where a and b are non-zero real numbers with opposite signs, 
    and p and q are non-zero real numbers. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  p : ℝ
  q : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  p_nonzero : p ≠ 0
  q_nonzero : q ≠ 0
  opposite_signs : a * b < 0

/-- The equation x²/(k-1) + y²/(k+1) = 1 -/
def equation (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) + y^2 / (k + 1) = 1

/-- The condition -1 < k < 1 -/
def condition (k : ℝ) : Prop :=
  -1 < k ∧ k < 1

/-- Theorem stating that the condition is necessary and sufficient 
    for the equation to represent a hyperbola -/
theorem hyperbola_condition (k : ℝ) : 
  (∃ h : Hyperbola, equation k ↔ h.a * x^2 / h.p + h.b * y^2 / h.q = 1) ↔ condition k :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l3683_368334


namespace NUMINAMATH_CALUDE_fraction_sum_ratio_l3683_368302

theorem fraction_sum_ratio : (1 / 3 + 1 / 4) / (1 / 2) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_ratio_l3683_368302


namespace NUMINAMATH_CALUDE_rational_numbers_include_integers_and_fractions_l3683_368348

/-- A rational number is a number that can be expressed as the quotient of two integers, where the denominator is non-zero. -/
def IsRational (x : ℚ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

/-- An integer is a whole number (positive, negative, or zero) without a fractional component. -/
def IsInteger (x : ℚ) : Prop := ∃ (n : ℤ), x = n

/-- A fraction is a rational number that is not an integer. -/
def IsFraction (x : ℚ) : Prop := IsRational x ∧ ¬IsInteger x

theorem rational_numbers_include_integers_and_fractions :
  (∀ x : ℚ, IsInteger x → IsRational x) ∧
  (∀ x : ℚ, IsFraction x → IsRational x) :=
sorry

end NUMINAMATH_CALUDE_rational_numbers_include_integers_and_fractions_l3683_368348


namespace NUMINAMATH_CALUDE_select_representatives_count_l3683_368346

/-- The number of ways to select subject representatives -/
def num_ways_to_select_representatives (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2).choose 2

/-- Theorem stating that selecting 4 students from 5 for specific subject representations results in 60 different ways -/
theorem select_representatives_count :
  num_ways_to_select_representatives 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_select_representatives_count_l3683_368346
