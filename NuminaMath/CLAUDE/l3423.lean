import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_range_l3423_342371

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 3| + |x - 1| < a^2 - 3*a) ↔ (a < -1 ∨ a > 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3423_342371


namespace NUMINAMATH_CALUDE_cube_packing_percentage_l3423_342395

/-- Calculates the number of whole cubes that can fit along a dimension -/
def cubesFit (dimension : ℕ) (cubeSize : ℕ) : ℕ :=
  dimension / cubeSize

/-- Calculates the volume of a rectangular box -/
def boxVolume (length width height : ℕ) : ℕ :=
  length * width * height

/-- Calculates the volume of a cube -/
def cubeVolume (size : ℕ) : ℕ :=
  size * size * size

/-- Theorem: The percentage of volume occupied by 4-inch cubes in a box with 
    dimensions 8 × 5 × 14 inches is 24/35 * 100% -/
theorem cube_packing_percentage :
  let boxLength : ℕ := 8
  let boxWidth : ℕ := 5
  let boxHeight : ℕ := 14
  let cubeSize : ℕ := 4
  let cubesAlongLength := cubesFit boxLength cubeSize
  let cubesAlongWidth := cubesFit boxWidth cubeSize
  let cubesAlongHeight := cubesFit boxHeight cubeSize
  let totalCubes := cubesAlongLength * cubesAlongWidth * cubesAlongHeight
  let volumeOccupied := totalCubes * cubeVolume cubeSize
  let totalVolume := boxVolume boxLength boxWidth boxHeight
  (volumeOccupied : ℚ) / totalVolume * 100 = 24 / 35 * 100 := by
  sorry

end NUMINAMATH_CALUDE_cube_packing_percentage_l3423_342395


namespace NUMINAMATH_CALUDE_number_of_scooters_l3423_342338

/-- Represents the number of wheels on a vehicle -/
def wheels (vehicle : String) : ℕ :=
  match vehicle with
  | "bicycle" => 2
  | "tricycle" => 3
  | "scooter" => 2
  | _ => 0

/-- The total number of vehicles -/
def total_vehicles : ℕ := 10

/-- The total number of wheels -/
def total_wheels : ℕ := 26

/-- Proves that the number of scooters is 2 -/
theorem number_of_scooters :
  ∃ (b t s : ℕ),
    b + t + s = total_vehicles ∧
    b * wheels "bicycle" + t * wheels "tricycle" + s * wheels "scooter" = total_wheels ∧
    s = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_scooters_l3423_342338


namespace NUMINAMATH_CALUDE_alyosha_max_gain_l3423_342323

/-- The maximum gain for Alyosha's cube game -/
def max_gain (total_boxes : ℕ) (blue_cubes : ℕ) : ℚ :=
  (2 ^ total_boxes : ℚ) / (total_boxes.choose blue_cubes)

/-- Theorem for Alyosha's maximum gain in the cube game -/
theorem alyosha_max_gain (total_boxes : ℕ) (blue_cubes : ℕ) 
  (h1 : total_boxes = 100) 
  (h2 : blue_cubes ≤ total_boxes) :
  max_gain total_boxes blue_cubes = 
    if blue_cubes = 1 
    then (2 ^ total_boxes : ℚ) / total_boxes
    else (2 ^ total_boxes : ℚ) / (total_boxes.choose blue_cubes) :=
by
  sorry

#check alyosha_max_gain

end NUMINAMATH_CALUDE_alyosha_max_gain_l3423_342323


namespace NUMINAMATH_CALUDE_exactly_five_triangles_l3423_342310

/-- A triangle with integral side lengths -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  sum_eq_8 : a + b + c = 8
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

/-- Count of distinct triangles with perimeter 8 -/
def count_triangles : ℕ := sorry

/-- The main theorem stating there are exactly 5 such triangles -/
theorem exactly_five_triangles : count_triangles = 5 := by sorry

end NUMINAMATH_CALUDE_exactly_five_triangles_l3423_342310


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l3423_342330

/-- Proves that the weight of the replaced person is 45 kg given the conditions -/
theorem weight_of_replaced_person
  (n : ℕ)
  (original_average : ℝ)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : n = 8)
  (h2 : weight_increase = 2.5)
  (h3 : new_person_weight = 65)
  : ∃ (replaced_weight : ℝ),
    n * (original_average + weight_increase) - n * original_average
    = new_person_weight - replaced_weight
    ∧ replaced_weight = 45 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l3423_342330


namespace NUMINAMATH_CALUDE_expression_equals_sixteen_ten_to_five_hundred_l3423_342367

theorem expression_equals_sixteen_ten_to_five_hundred :
  (3^500 + 4^501)^2 - (3^500 - 4^501)^2 = 16 * 10^500 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_sixteen_ten_to_five_hundred_l3423_342367


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l3423_342378

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : 
  (x * y ≥ 4) ∧ (x + y ≥ 9/2) := by
  sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l3423_342378


namespace NUMINAMATH_CALUDE_circle_properties_l3423_342388

/-- Given a circle with equation x^2 - 24x + y^2 - 4y = -36, 
    prove its center, radius, and the sum of center coordinates and radius. -/
theorem circle_properties : 
  let D : Set (ℝ × ℝ) := {p | (p.1^2 - 24*p.1 + p.2^2 - 4*p.2 = -36)}
  ∃ (a b r : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ D ↔ (x - a)^2 + (y - b)^2 = r^2) ∧ 
    a = 12 ∧ 
    b = 2 ∧ 
    r = 4 * Real.sqrt 7 ∧
    a + b + r = 14 + 4 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3423_342388


namespace NUMINAMATH_CALUDE_molecular_weight_3_moles_N2O3_l3423_342372

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in a molecule of Dinitrogen trioxide -/
def N_count : ℕ := 2

/-- The number of Oxygen atoms in a molecule of Dinitrogen trioxide -/
def O_count : ℕ := 3

/-- The number of moles of Dinitrogen trioxide -/
def mole_count : ℝ := 3

/-- The molecular weight of Dinitrogen trioxide in g/mol -/
def molecular_weight_N2O3 : ℝ := N_count * atomic_weight_N + O_count * atomic_weight_O

/-- Theorem: The molecular weight of 3 moles of Dinitrogen trioxide is 228.06 grams -/
theorem molecular_weight_3_moles_N2O3 : 
  mole_count * molecular_weight_N2O3 = 228.06 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_3_moles_N2O3_l3423_342372


namespace NUMINAMATH_CALUDE_min_points_on_circle_l3423_342341

theorem min_points_on_circle (circle_length : ℕ) (h : circle_length = 1956) :
  let min_points := 1304
  ∀ n : ℕ, n < min_points →
    ¬(∀ p : ℕ, p < n →
      (∃! q : ℕ, q < n ∧ (q - p) % circle_length = 1) ∧
      (∃! r : ℕ, r < n ∧ (r - p) % circle_length = 2)) ∧
  (∀ p : ℕ, p < min_points →
    (∃! q : ℕ, q < min_points ∧ (q - p) % circle_length = 1) ∧
    (∃! r : ℕ, r < min_points ∧ (r - p) % circle_length = 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_points_on_circle_l3423_342341


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3423_342327

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3423_342327


namespace NUMINAMATH_CALUDE_range_of_a_l3423_342377

-- Define the system of linear equations
def system (x y a : ℝ) : Prop :=
  (3 * x + 5 * y = 6 * a) ∧ (2 * x + 6 * y = 3 * a + 3)

-- Define the constraint
def constraint (x y : ℝ) : Prop :=
  x - y > 0

-- Theorem statement
theorem range_of_a (x y a : ℝ) :
  system x y a → constraint x y → a > 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3423_342377


namespace NUMINAMATH_CALUDE_radical_axis_properties_l3423_342336

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def power (p : ℝ × ℝ) (c : Circle) : ℝ :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 - c.radius^2

def perpendicular (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let ((x1, y1), (x2, y2)) := l1
  let ((x3, y3), (x4, y4)) := l2
  (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3) = 0

def on_line (p : ℝ × ℝ) (l : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (x, y) := p
  let ((x1, y1), (x2, y2)) := l
  (y2 - y1) * (x - x1) = (x2 - x1) * (y - y1)

-- Define the theorem
theorem radical_axis_properties 
  (k₁ k₂ : Circle) 
  (P Q : ℝ × ℝ) 
  (h_non_intersect : k₁ ≠ k₂) 
  (h_power_P : power P k₁ = power P k₂)
  (h_power_Q : power Q k₁ = power Q k₂) :
  let O₁ := k₁.center
  let O₂ := k₂.center
  (perpendicular (P, Q) (O₁, O₂)) ∧ 
  (∀ S, (power S k₁ = power S k₂) ↔ on_line S (P, Q)) ∧
  (∀ k : Circle, 
    (∃ x y, power (x, y) k = power (x, y) k₁ ∧ power (x, y) k = power (x, y) k₂) →
    (∃ M, (power M k = power M k₁) ∧ (power M k = power M k₂) ∧ 
          (on_line M (P, Q) ∨ perpendicular (P, Q) (k.center, M)))) := by
  sorry

end NUMINAMATH_CALUDE_radical_axis_properties_l3423_342336


namespace NUMINAMATH_CALUDE_simplified_expression_l3423_342304

theorem simplified_expression : -1^2008 + 3*(-1)^2007 + 1^2008 - 2*(-1)^2009 = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l3423_342304


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l3423_342319

theorem trig_expression_equals_four : 
  (1 / Real.sin (10 * π / 180)) - (Real.sqrt 3 / Real.cos (10 * π / 180)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l3423_342319


namespace NUMINAMATH_CALUDE_min_value_of_function_l3423_342303

/-- The function f(x) = 4/(x-2) + x has a minimum value of 6 for x > 2 -/
theorem min_value_of_function (x : ℝ) (h : x > 2) : 
  (4 / (x - 2) + x) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3423_342303


namespace NUMINAMATH_CALUDE_warehouse_length_calculation_l3423_342370

/-- Represents the dimensions and walking pattern around a rectangular warehouse. -/
structure Warehouse :=
  (width : ℝ)
  (length : ℝ)
  (circles : ℕ)
  (total_distance : ℝ)

/-- Theorem stating the length of the warehouse given specific conditions. -/
theorem warehouse_length_calculation (w : Warehouse) 
  (h1 : w.width = 400)
  (h2 : w.circles = 8)
  (h3 : w.total_distance = 16000)
  : w.length = 600 := by
  sorry

#check warehouse_length_calculation

end NUMINAMATH_CALUDE_warehouse_length_calculation_l3423_342370


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3423_342355

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ 3 * Real.sqrt 8 ∧
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 7 ∧
    Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) = 3 * Real.sqrt 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3423_342355


namespace NUMINAMATH_CALUDE_equilateral_triangle_extension_equality_l3423_342320

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the equilateral property
def is_equilateral (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Define points D and E
variable (D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions for D and E
def D_on_AC_extension (A C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, t > 1 ∧ D = A + t • (C - A)

def E_on_BC_extension (B C E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ s : ℝ, s > 1 ∧ E = B + s • (C - B)

-- Define the equality of BD and DE
def BD_equals_DE (B D E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist B D = dist D E

-- State the theorem
theorem equilateral_triangle_extension_equality
  (h1 : is_equilateral A B C)
  (h2 : D_on_AC_extension A C D)
  (h3 : E_on_BC_extension B C E)
  (h4 : BD_equals_DE B D E) :
  dist A D = dist C E :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_extension_equality_l3423_342320


namespace NUMINAMATH_CALUDE_difference_of_percentages_l3423_342315

-- Define the percentage
def percentage : ℚ := 25 / 100

-- Define the two amounts in pence (to avoid floating-point issues)
def amount1 : ℕ := 3700  -- £37 in pence
def amount2 : ℕ := 1700  -- £17 in pence

-- Theorem statement
theorem difference_of_percentages :
  (percentage * amount1 - percentage * amount2 : ℚ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_percentages_l3423_342315


namespace NUMINAMATH_CALUDE_even_function_inequality_l3423_342361

open Real Set

theorem even_function_inequality 
  (f : ℝ → ℝ) 
  (h_even : ∀ x, x ∈ Ioo (-π/2) (π/2) → f x = f (-x))
  (h_deriv : ∀ x, x ∈ Ioo 0 (π/2) → 
    (deriv^[2] f) x * cos x + f x * sin x < 0) :
  ∀ x, x ∈ (Ioo (-π/2) (-π/4) ∪ Ioo (π/4) (π/2)) → 
    f x < Real.sqrt 2 * f (π/4) * cos x := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3423_342361


namespace NUMINAMATH_CALUDE_school_students_count_l3423_342329

theorem school_students_count : ℕ :=
  let below_eight_percent : ℚ := 20 / 100
  let eight_years_count : ℕ := 12
  let above_eight_ratio : ℚ := 2 / 3
  let total_students : ℕ := 40

  have h1 : ↑eight_years_count + (↑eight_years_count * above_eight_ratio) = (1 - below_eight_percent) * total_students := by sorry

  total_students


end NUMINAMATH_CALUDE_school_students_count_l3423_342329


namespace NUMINAMATH_CALUDE_bill_sunday_run_l3423_342311

/-- Represents the miles run by Bill, Julia, and Mark over a weekend -/
structure WeekendRun where
  billSaturday : ℝ
  billSunday : ℝ
  juliaSunday : ℝ
  markSaturday : ℝ
  markSunday : ℝ

/-- Conditions for the weekend run -/
def weekendRunConditions (run : WeekendRun) : Prop :=
  run.billSunday = run.billSaturday + 4 ∧
  run.juliaSunday = 2 * run.billSunday ∧
  run.markSaturday = 5 ∧
  run.markSunday = run.markSaturday + 2 ∧
  run.billSaturday + run.billSunday + run.juliaSunday + run.markSaturday + run.markSunday = 50

/-- Theorem stating that under the given conditions, Bill ran 10.5 miles on Sunday -/
theorem bill_sunday_run (run : WeekendRun) (h : weekendRunConditions run) : 
  run.billSunday = 10.5 := by
  sorry


end NUMINAMATH_CALUDE_bill_sunday_run_l3423_342311


namespace NUMINAMATH_CALUDE_factorization_proof_l3423_342312

theorem factorization_proof (a : ℝ) : 74 * a^2 + 222 * a + 148 = 74 * (a + 2) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3423_342312


namespace NUMINAMATH_CALUDE_investment_problem_l3423_342328

theorem investment_problem (total investment bonds stocks mutual_funds : ℕ) : 
  total = 220000 ∧ 
  stocks = 5 * bonds ∧ 
  mutual_funds = 2 * stocks ∧ 
  total = bonds + stocks + mutual_funds →
  stocks = 68750 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l3423_342328


namespace NUMINAMATH_CALUDE_water_barrel_problem_l3423_342335

theorem water_barrel_problem :
  ∀ (bucket_capacity : ℕ),
    bucket_capacity > 0 →
    bucket_capacity / 2 +
    bucket_capacity / 3 +
    bucket_capacity / 4 +
    bucket_capacity / 5 +
    bucket_capacity / 6 = 29 →
    bucket_capacity ≤ 30 →
    29 ≤ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_water_barrel_problem_l3423_342335


namespace NUMINAMATH_CALUDE_origami_paper_distribution_l3423_342353

theorem origami_paper_distribution (total_papers : ℝ) (num_cousins : ℝ) 
  (h1 : total_papers = 48.0)
  (h2 : num_cousins = 6.0)
  (h3 : num_cousins ≠ 0) :
  total_papers / num_cousins = 8.0 := by
sorry

end NUMINAMATH_CALUDE_origami_paper_distribution_l3423_342353


namespace NUMINAMATH_CALUDE_pizza_cost_per_slice_l3423_342325

-- Define the pizza and topping costs
def large_pizza_cost : ℚ := 10
def first_topping_cost : ℚ := 2
def next_two_toppings_cost : ℚ := 1
def remaining_toppings_cost : ℚ := 0.5

-- Define the number of slices and toppings
def num_slices : ℕ := 8
def num_toppings : ℕ := 7

-- Calculate the total cost of toppings
def total_toppings_cost : ℚ :=
  first_topping_cost +
  2 * next_two_toppings_cost +
  (num_toppings - 3) * remaining_toppings_cost

-- Calculate the total cost of the pizza
def total_pizza_cost : ℚ := large_pizza_cost + total_toppings_cost

-- Theorem to prove
theorem pizza_cost_per_slice :
  total_pizza_cost / num_slices = 2 := by sorry

end NUMINAMATH_CALUDE_pizza_cost_per_slice_l3423_342325


namespace NUMINAMATH_CALUDE_unique_number_with_remainders_l3423_342398

theorem unique_number_with_remainders : ∃! n : ℕ, 
  50 < n ∧ n < 70 ∧ n % 5 = 3 ∧ n % 7 = 2 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_remainders_l3423_342398


namespace NUMINAMATH_CALUDE_matrix_sum_of_squares_l3423_342369

open Matrix

theorem matrix_sum_of_squares (x y z w : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (transpose B = (2 : ℝ) • (B⁻¹)) →
  x^2 + y^2 + z^2 + w^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_of_squares_l3423_342369


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3423_342373

theorem negation_of_proposition (p : Prop) : 
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 < 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3423_342373


namespace NUMINAMATH_CALUDE_cookie_calculation_l3423_342385

theorem cookie_calculation (initial_cookies given_cookies received_cookies : ℕ) :
  initial_cookies ≥ given_cookies →
  initial_cookies - given_cookies + received_cookies =
    initial_cookies - given_cookies + received_cookies :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_calculation_l3423_342385


namespace NUMINAMATH_CALUDE_volume_complete_octagonal_pyramid_l3423_342347

/-- The volume of a complete pyramid with a regular octagonal base, given the dimensions of its truncated version. -/
theorem volume_complete_octagonal_pyramid 
  (lower_base_side : ℝ) 
  (upper_base_side : ℝ) 
  (truncated_height : ℝ) 
  (h_lower : lower_base_side = 0.4) 
  (h_upper : upper_base_side = 0.3) 
  (h_height : truncated_height = 0.5) : 
  ∃ (volume : ℝ), volume = (16/75) * (Real.sqrt 2 + 1) := by
  sorry

#check volume_complete_octagonal_pyramid

end NUMINAMATH_CALUDE_volume_complete_octagonal_pyramid_l3423_342347


namespace NUMINAMATH_CALUDE_equation_solution_l3423_342366

theorem equation_solution : 
  ∃ x : ℝ, (x + 1) / (x - 1) = 1 / (x - 2) + 1 ↔ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3423_342366


namespace NUMINAMATH_CALUDE_saltwater_volume_l3423_342354

/-- Proves that the initial volume of a saltwater solution is 200 gallons, given the conditions stated in the problem. -/
theorem saltwater_volume : ∃ (x : ℝ),
  -- Initial solution is 20% salt by volume
  let initial_salt := 0.2 * x
  -- Volume after evaporation (3/4 of initial volume)
  let volume_after_evap := 0.75 * x
  -- New volume after adding water and salt
  let new_volume := volume_after_evap + 10 + 20
  -- New amount of salt
  let new_salt := initial_salt + 20
  -- The resulting mixture is 33 1/3% salt by volume
  new_salt = (1/3) * new_volume ∧ x = 200 :=
sorry

end NUMINAMATH_CALUDE_saltwater_volume_l3423_342354


namespace NUMINAMATH_CALUDE_expected_ones_value_l3423_342316

/-- The number of magnets --/
def n : ℕ := 50

/-- The probability of a difference of 1 between two randomly chosen numbers --/
def p : ℚ := 49 / 1225

/-- The number of pairs of consecutive magnets --/
def num_pairs : ℕ := n - 1

/-- The expected number of times the difference 1 occurs --/
def expected_ones : ℚ := num_pairs * p

theorem expected_ones_value : expected_ones = 49 / 25 := by sorry

end NUMINAMATH_CALUDE_expected_ones_value_l3423_342316


namespace NUMINAMATH_CALUDE_projection_area_l3423_342383

/-- A polygon in 3D space -/
structure Polygon3D where
  -- Define the polygon structure (this is a simplification)
  area : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane structure (this is a simplification)

/-- The angle between two planes -/
def angle_between_planes (p1 p2 : Plane3D) : ℝ := sorry

/-- The projection of a polygon onto a plane -/
def project_polygon (poly : Polygon3D) (plane : Plane3D) : Polygon3D := sorry

/-- Theorem: The area of a polygon's projection is the original area times the cosine of the angle between planes -/
theorem projection_area (poly : Polygon3D) (plane : Plane3D) : 
  (project_polygon poly plane).area = poly.area * Real.cos (angle_between_planes (Plane3D.mk) plane) := by
  sorry

end NUMINAMATH_CALUDE_projection_area_l3423_342383


namespace NUMINAMATH_CALUDE_sqrt_2_power_n_equals_64_l3423_342375

theorem sqrt_2_power_n_equals_64 (n : ℕ) : Real.sqrt (2^n) = 64 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_power_n_equals_64_l3423_342375


namespace NUMINAMATH_CALUDE_veranda_width_l3423_342313

/-- Veranda width problem -/
theorem veranda_width (room_length room_width veranda_area : ℝ) 
  (h1 : room_length = 17)
  (h2 : room_width = 12)
  (h3 : veranda_area = 132)
  (h4 : veranda_area = (room_length + 2 * w) * (room_width + 2 * w) - room_length * room_width)
  : w = 2 := by
  sorry

end NUMINAMATH_CALUDE_veranda_width_l3423_342313


namespace NUMINAMATH_CALUDE_otimes_equation_roots_l3423_342363

-- Define the new operation
def otimes (a b : ℝ) : ℝ := a * b^2 - b

-- Theorem statement
theorem otimes_equation_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ otimes 1 x = k ∧ otimes 1 y = k) ↔ k > -1/4 :=
sorry

end NUMINAMATH_CALUDE_otimes_equation_roots_l3423_342363


namespace NUMINAMATH_CALUDE_company_workforce_l3423_342391

theorem company_workforce (initial_employees : ℕ) 
  (h1 : (60 : ℚ) / 100 * initial_employees = (55 : ℚ) / 100 * (initial_employees + 30)) :
  initial_employees + 30 = 360 := by
sorry

end NUMINAMATH_CALUDE_company_workforce_l3423_342391


namespace NUMINAMATH_CALUDE_positive_rational_solution_condition_l3423_342334

theorem positive_rational_solution_condition 
  (a b : ℚ) (x y : ℚ) 
  (h_product : x * y = a) 
  (h_sum : x + y = b) : 
  (∃ (k : ℚ), k > 0 ∧ b^2 / 4 - a = k^2) ↔ 
  (x > 0 ∧ y > 0 ∧ ∃ (m n : ℕ), x = m / n) :=
sorry

end NUMINAMATH_CALUDE_positive_rational_solution_condition_l3423_342334


namespace NUMINAMATH_CALUDE_equation_solution_l3423_342358

theorem equation_solution : ∃ x₁ x₂ : ℝ,
  (1 / (x₁ + 10) + 1 / (x₁ + 8) = 1 / (x₁ + 11) + 1 / (x₁ + 7) + 1 / (2 * x₁ + 36)) ∧
  (1 / (x₂ + 10) + 1 / (x₂ + 8) = 1 / (x₂ + 11) + 1 / (x₂ + 7) + 1 / (2 * x₂ + 36)) ∧
  (5 * x₁^2 + 140 * x₁ + 707 = 0) ∧
  (5 * x₂^2 + 140 * x₂ + 707 = 0) ∧
  x₁ ≠ x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3423_342358


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3423_342300

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 2 + 35 / 99) ∧ (x = 233 / 99) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3423_342300


namespace NUMINAMATH_CALUDE_red_balls_count_l3423_342346

/-- Given a bag with 2400 balls of three colors (red, green, blue) distributed
    in the ratio 15:13:17, prove that the number of red balls is 795. -/
theorem red_balls_count (total : ℕ) (red green blue : ℕ) :
  total = 2400 →
  red + green + blue = total →
  red * 13 = green * 15 →
  red * 17 = blue * 15 →
  red = 795 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l3423_342346


namespace NUMINAMATH_CALUDE_tangent_slope_at_two_l3423_342331

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem tangent_slope_at_two
  (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = x / (x - 1))
  : deriv f 2 = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_two_l3423_342331


namespace NUMINAMATH_CALUDE_perpendicular_lines_intersection_l3423_342317

theorem perpendicular_lines_intersection (a b c d : ℝ) : 
  (∀ x y, a * x - 2 * y = d) →  -- First line equation
  (∀ x y, 2 * x + b * y = c) →  -- Second line equation
  (a * 2 - 2 * (-3) = d) →      -- Lines intersect at (2, -3)
  (2 * 2 + b * (-3) = c) →      -- Lines intersect at (2, -3)
  (a * b = -4) →                -- Perpendicular lines condition
  (d = 12) :=                   -- Conclusion
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_intersection_l3423_342317


namespace NUMINAMATH_CALUDE_investment_growth_l3423_342359

/-- Calculates the future value of an investment with compound interest -/
def future_value (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that an initial investment of $313,021.70 at 6% annual interest for 15 years 
    results in approximately $750,000 -/
theorem investment_growth :
  let initial_investment : ℝ := 313021.70
  let interest_rate : ℝ := 0.06
  let years : ℕ := 15
  let target_amount : ℝ := 750000
  
  abs (future_value initial_investment interest_rate years - target_amount) < 1 := by
  sorry


end NUMINAMATH_CALUDE_investment_growth_l3423_342359


namespace NUMINAMATH_CALUDE_equation_solution_l3423_342309

theorem equation_solution :
  ∀ x : ℝ, x ≠ 1 →
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3423_342309


namespace NUMINAMATH_CALUDE_perfect_square_property_l3423_342332

theorem perfect_square_property (n : ℕ) (hn : n ≥ 3) 
  (hx : ∃ x : ℕ, 1 + 3 * n = x ^ 2) : 
  ∃ a b c : ℕ, 1 + (3 * n + 3) / (a ^ 2 + b ^ 2 + c ^ 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_property_l3423_342332


namespace NUMINAMATH_CALUDE_rhombus_area_with_diagonals_6_and_8_l3423_342302

/-- The area of a rhombus with diagonals of lengths 6 and 8 is 24. -/
theorem rhombus_area_with_diagonals_6_and_8 : 
  ∀ (r : ℝ × ℝ → ℝ), 
  (∀ d₁ d₂, r (d₁, d₂) = (1/2) * d₁ * d₂) →
  r (6, 8) = 24 := by
sorry

end NUMINAMATH_CALUDE_rhombus_area_with_diagonals_6_and_8_l3423_342302


namespace NUMINAMATH_CALUDE_quadratic_point_value_l3423_342343

/-- Given a quadratic function y = -ax^2 + 2ax + 3 where a > 0,
    if the point P(m, 3) lies on the graph and m ≠ 0, then m = 2. -/
theorem quadratic_point_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 →
  3 = -a * m^2 + 2 * a * m + 3 →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_point_value_l3423_342343


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3423_342390

theorem smallest_sum_of_sequence (E F G H : ℤ) : 
  E > 0 ∧ F > 0 ∧ G > 0 →  -- E, F, G are positive integers
  ∃ d : ℤ, G - F = F - E ∧ F - E = d →  -- E, F, G form an arithmetic sequence
  ∃ r : ℚ, G = F * r ∧ H = G * r →  -- F, G, H form a geometric sequence
  G / F = 7 / 4 →  -- Given ratio
  ∀ E' F' G' H' : ℤ,
    (E' > 0 ∧ F' > 0 ∧ G' > 0 ∧
     ∃ d' : ℤ, G' - F' = F' - E' ∧ F' - E' = d' ∧
     ∃ r' : ℚ, G' = F' * r' ∧ H' = G' * r' ∧
     G' / F' = 7 / 4) →
    E + F + G + H ≤ E' + F' + G' + H' →
  E + F + G + H = 97 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3423_342390


namespace NUMINAMATH_CALUDE_no_fraction_satisfies_condition_l3423_342352

theorem no_fraction_satisfies_condition : ¬∃ (x y : ℕ+), 
  (Nat.gcd x.val y.val = 1) ∧ 
  ((x + 2 : ℚ) / (y + 2) = 1.2 * (x : ℚ) / y) := by
  sorry

end NUMINAMATH_CALUDE_no_fraction_satisfies_condition_l3423_342352


namespace NUMINAMATH_CALUDE_scientific_notation_of_1_5_million_l3423_342374

/-- Expresses 1.5 million in scientific notation -/
theorem scientific_notation_of_1_5_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1500000 = a * (10 : ℝ) ^ n ∧ a = 1.5 ∧ n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_5_million_l3423_342374


namespace NUMINAMATH_CALUDE_ellipse_equation_l3423_342396

/-- The standard equation of an ellipse passing through (3,0) with eccentricity √6/3 -/
theorem ellipse_equation (x y : ℝ) :
  let e : ℝ := Real.sqrt 6 / 3
  let passes_through : ℝ × ℝ := (3, 0)
  (∃ (a b : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ∧
                 (3^2 / a^2 + 0^2 / b^2 = 1) ∧
                 e^2 = 1 - (min a b)^2 / (max a b)^2)) →
  (x^2 / 9 + y^2 / 3 = 1) ∨ (x^2 / 9 + y^2 / 27 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3423_342396


namespace NUMINAMATH_CALUDE_second_derivative_parametric_function_l3423_342348

/-- The second-order derivative of a parametrically defined function -/
theorem second_derivative_parametric_function (t : ℝ) (h : t ≠ 0) :
  let x := 1 / t
  let y := 1 / (1 + t^2)
  let y''_xx := (2 * (t^2 - 3) * t^4) / ((1 + t^2)^3)
  ∃ (d2y_dx2 : ℝ), d2y_dx2 = y''_xx := by
  sorry

end NUMINAMATH_CALUDE_second_derivative_parametric_function_l3423_342348


namespace NUMINAMATH_CALUDE_estimate_nearsighted_students_l3423_342360

/-- Estimates the number of nearsighted students in a population based on a sample. -/
theorem estimate_nearsighted_students 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (nearsighted_in_sample : ℕ) 
  (h1 : total_students = 400) 
  (h2 : sample_size = 30) 
  (h3 : nearsighted_in_sample = 12) :
  ⌊(total_students : ℚ) * (nearsighted_in_sample : ℚ) / (sample_size : ℚ)⌋ = 160 :=
sorry

end NUMINAMATH_CALUDE_estimate_nearsighted_students_l3423_342360


namespace NUMINAMATH_CALUDE_minimum_red_chips_l3423_342307

theorem minimum_red_chips (w b r : ℕ) 
  (blue_white : b ≥ (1/3 : ℚ) * w)
  (blue_red : b ≤ (1/4 : ℚ) * r)
  (white_blue_total : w + b ≥ 75) :
  r ≥ 76 :=
sorry

end NUMINAMATH_CALUDE_minimum_red_chips_l3423_342307


namespace NUMINAMATH_CALUDE_equation_solution_l3423_342386

theorem equation_solution (a b c : ℤ) : 
  (∀ x, (x - a) * (x - 12) + 4 = (x + b) * (x + c)) → (a = 7 ∨ a = 17) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3423_342386


namespace NUMINAMATH_CALUDE_roots_of_equation1_roots_of_equation2_l3423_342322

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 8*x + 1 = 0
def equation2 (x : ℝ) : Prop := x*(x-2) - x + 2 = 0

-- Theorem for equation 1
theorem roots_of_equation1 : 
  ∃ (x₁ x₂ : ℝ), x₁ = 4 + Real.sqrt 15 ∧ x₂ = 4 - Real.sqrt 15 ∧ 
  equation1 x₁ ∧ equation1 x₂ :=
sorry

-- Theorem for equation 2
theorem roots_of_equation2 : 
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = 1 ∧ 
  equation2 x₁ ∧ equation2 x₂ :=
sorry

end NUMINAMATH_CALUDE_roots_of_equation1_roots_of_equation2_l3423_342322


namespace NUMINAMATH_CALUDE_pony_jeans_discount_rate_l3423_342306

theorem pony_jeans_discount_rate
  (fox_price : ℝ)
  (pony_price : ℝ)
  (total_savings : ℝ)
  (fox_quantity : ℕ)
  (pony_quantity : ℕ)
  (total_discount_rate : ℝ)
  (h1 : fox_price = 15)
  (h2 : pony_price = 18)
  (h3 : total_savings = 8.64)
  (h4 : fox_quantity = 3)
  (h5 : pony_quantity = 2)
  (h6 : total_discount_rate = 22) :
  ∃ (fox_discount : ℝ) (pony_discount : ℝ),
    fox_discount + pony_discount = total_discount_rate ∧
    fox_quantity * fox_price * (fox_discount / 100) +
    pony_quantity * pony_price * (pony_discount / 100) = total_savings ∧
    pony_discount = 14 := by
  sorry

end NUMINAMATH_CALUDE_pony_jeans_discount_rate_l3423_342306


namespace NUMINAMATH_CALUDE_circle_radius_from_chords_l3423_342357

/-- Given a circle with two chords AB and AC, where AB = a, AC = b, and the length of arc AC is twice
    the length of arc AB, prove that the radius R of the circle is equal to a^2 / sqrt(4a^2 - b^2). -/
theorem circle_radius_from_chords (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ (R : ℝ), R > 0 ∧ R = a^2 / Real.sqrt (4 * a^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_chords_l3423_342357


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3423_342333

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 5 * a 7 = 2) →
  (a 2 + a 10 = 3) →
  (a 12 / a 4 = 2 ∨ a 12 / a 4 = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3423_342333


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l3423_342342

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- Theorem 1: Solution set for a = 2
theorem solution_set_a_2 : 
  {x : ℝ | f 2 x ≤ 1} = {x : ℝ | -1/3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2: Range of a for the inequality to hold
theorem range_of_a : 
  {a : ℝ | ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l3423_342342


namespace NUMINAMATH_CALUDE_coffee_break_theorem_coffee_break_converse_l3423_342376

/-- Represents the number of participants who went for coffee -/
def coffee_drinkers : Finset ℕ := {6, 8, 10, 12}

/-- Represents the total number of participants -/
def total_participants : ℕ := 14

theorem coffee_break_theorem (n : ℕ) (hn : n ∈ coffee_drinkers) :
  ∃ (k : ℕ),
    -- k represents the number of pairs of participants who stayed
    0 < k ∧ 
    k < total_participants / 2 ∧
    -- n is the number of participants who left
    n = total_participants - 2 * k ∧
    -- Each remaining participant has exactly one neighbor who left
    ∀ (i : ℕ), i < total_participants → 
      (i % 2 = 0 → (i + 1) % total_participants < n) ∧
      (i % 2 = 1 → i < n) :=
by sorry

theorem coffee_break_converse :
  ∀ (n : ℕ),
    (∃ (k : ℕ),
      0 < k ∧ 
      k < total_participants / 2 ∧
      n = total_participants - 2 * k ∧
      ∀ (i : ℕ), i < total_participants → 
        (i % 2 = 0 → (i + 1) % total_participants < n) ∧
        (i % 2 = 1 → i < n)) →
    n ∈ coffee_drinkers :=
by sorry

end NUMINAMATH_CALUDE_coffee_break_theorem_coffee_break_converse_l3423_342376


namespace NUMINAMATH_CALUDE_greatest_divisor_of_28_l3423_342397

theorem greatest_divisor_of_28 : ∃ d : ℕ, d ∣ 28 ∧ ∀ x : ℕ, x ∣ 28 → x ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_28_l3423_342397


namespace NUMINAMATH_CALUDE_segment_length_is_52_l3423_342340

/-- A right triangle with sides 10, 24, and 26 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 10
  side_b : b = 24
  side_c : c = 26

/-- Three identical circles inscribed in the triangle -/
structure InscribedCircles where
  radius : ℝ
  radius_value : radius = 2
  touches_sides : Bool
  touches_other_circles : Bool

/-- The total length of segments from vertices to tangency points -/
def total_segment_length (t : RightTriangle) (circles : InscribedCircles) : ℝ :=
  (t.a - circles.radius) + (t.b - circles.radius) + (t.c - 2 * circles.radius)

theorem segment_length_is_52 (t : RightTriangle) (circles : InscribedCircles) :
  total_segment_length t circles = 52 :=
sorry

end NUMINAMATH_CALUDE_segment_length_is_52_l3423_342340


namespace NUMINAMATH_CALUDE_candy_given_to_janet_and_emily_l3423_342389

-- Define the initial amount of candy
def initial_candy : ℝ := 78.5

-- Define the amount left after giving to Janet
def left_after_janet : ℝ := 68.75

-- Define the amount given to Emily
def given_to_emily : ℝ := 2.25

-- Theorem to prove
theorem candy_given_to_janet_and_emily :
  initial_candy - left_after_janet + given_to_emily = 12 := by
  sorry

end NUMINAMATH_CALUDE_candy_given_to_janet_and_emily_l3423_342389


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l3423_342305

theorem tan_theta_minus_pi_fourth (θ : ℝ) : 
  (Complex.cos θ - 4/5 = 0) → 
  (Complex.sin θ - 3/5 ≠ 0) → 
  Real.tan (θ - π/4) = -7 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l3423_342305


namespace NUMINAMATH_CALUDE_stone_length_is_four_dm_l3423_342344

/-- Represents the dimensions of a rectangular hall in meters -/
structure HallDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a rectangular stone in decimeters -/
structure StoneDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : HallDimensions) : ℝ := d.length * d.width

/-- Converts meters to decimeters -/
def meterToDecimeter (m : ℝ) : ℝ := m * 10

/-- Theorem: Given a hall of 36m x 15m, 2700 stones required, and stone width of 5dm,
    prove that the length of each stone is 4 dm -/
theorem stone_length_is_four_dm (hall : HallDimensions) (stone : StoneDimensions) 
    (num_stones : ℕ) : 
    hall.length = 36 → 
    hall.width = 15 → 
    num_stones = 2700 → 
    stone.width = 5 → 
    stone.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_stone_length_is_four_dm_l3423_342344


namespace NUMINAMATH_CALUDE_investment_calculation_l3423_342351

/-- Calculates the total investment in shares given the following conditions:
  * Face value of shares is 100 rupees
  * Shares are bought at a 20% premium
  * Company declares a 6% dividend
  * Total dividend received is 720 rupees
-/
def calculate_investment (face_value : ℕ) (premium_percent : ℕ) (dividend_percent : ℕ) (total_dividend : ℕ) : ℕ :=
  let premium_price := face_value + face_value * premium_percent / 100
  let dividend_per_share := face_value * dividend_percent / 100
  let num_shares := total_dividend / dividend_per_share
  num_shares * premium_price

/-- Theorem stating that under the given conditions, the total investment is 14400 rupees -/
theorem investment_calculation :
  calculate_investment 100 20 6 720 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l3423_342351


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l3423_342345

theorem factorization_of_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l3423_342345


namespace NUMINAMATH_CALUDE_book_cost_price_l3423_342399

theorem book_cost_price (profit_10 profit_15 additional_profit : ℝ) 
  (h1 : profit_10 = 0.10)
  (h2 : profit_15 = 0.15)
  (h3 : additional_profit = 120) :
  ∃ cost_price : ℝ, 
    cost_price * (1 + profit_15) - cost_price * (1 + profit_10) = additional_profit ∧ 
    cost_price = 2400 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l3423_342399


namespace NUMINAMATH_CALUDE_parallelogram_probability_theorem_l3423_342384

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- The probability of a point in a parallelogram being not below the x-axis -/
def probability_not_below_x_axis (para : Parallelogram) : ℝ := 
  sorry

theorem parallelogram_probability_theorem (para : Parallelogram) :
  para.P = Point.mk 4 4 →
  para.Q = Point.mk (-2) (-2) →
  para.R = Point.mk (-8) (-2) →
  para.S = Point.mk (-2) 4 →
  probability_not_below_x_axis para = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_probability_theorem_l3423_342384


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_roots_l3423_342394

theorem cubic_polynomial_integer_roots :
  ∀ (a c : ℤ), ∃ (x y z : ℤ),
    ∀ (X : ℤ), X^3 + a*X^2 - X + c = 0 ↔ (X = x ∨ X = y ∨ X = z) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_roots_l3423_342394


namespace NUMINAMATH_CALUDE_valerie_light_bulbs_l3423_342314

theorem valerie_light_bulbs :
  let total_budget : ℕ := 60
  let small_bulb_cost : ℕ := 8
  let large_bulb_cost : ℕ := 12
  let small_bulb_count : ℕ := 3
  let remaining_money : ℕ := 24
  let large_bulb_count : ℕ := (total_budget - remaining_money - small_bulb_cost * small_bulb_count) / large_bulb_cost
  large_bulb_count = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_valerie_light_bulbs_l3423_342314


namespace NUMINAMATH_CALUDE_solve_equation_l3423_342337

theorem solve_equation : ∃ y : ℚ, (2 / 7) * (1 / 8) * y = 12 ∧ y = 336 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3423_342337


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l3423_342393

/-- The minimum distance between a point on the circle (x-2)² + y² = 4
    and a point on the line x - y + 3 = 0 is (5√2)/2 - 2 -/
theorem min_distance_circle_line :
  let circle := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}
  let line := {q : ℝ × ℝ | q.1 - q.2 + 3 = 0}
  ∃ (d : ℝ), d = (5 * Real.sqrt 2) / 2 - 2 ∧
    ∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ circle → q ∈ line →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry


end NUMINAMATH_CALUDE_min_distance_circle_line_l3423_342393


namespace NUMINAMATH_CALUDE_robert_ate_seven_chocolates_l3423_342368

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The difference between Robert's and Nickel's chocolate consumption -/
def robert_nickel_difference : ℕ := 2

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := nickel_chocolates + robert_nickel_difference

theorem robert_ate_seven_chocolates : robert_chocolates = 7 := by
  sorry

end NUMINAMATH_CALUDE_robert_ate_seven_chocolates_l3423_342368


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3423_342350

theorem imaginary_part_of_z (θ : ℝ) : 
  let z : ℂ := Complex.mk (Real.sin (2 * θ) - 1) (Real.sqrt 2 * Real.cos θ - 1)
  (z.re = 0 ∧ z.im ≠ 0) → z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3423_342350


namespace NUMINAMATH_CALUDE_units_digit_of_2_power_2018_l3423_342365

theorem units_digit_of_2_power_2018 : ∃ (f : ℕ → ℕ), 
  (∀ n, f n = n % 4) ∧ 
  (∀ n, n > 0 → (2^n % 10 = 2 ∨ 2^n % 10 = 4 ∨ 2^n % 10 = 8 ∨ 2^n % 10 = 6)) ∧
  (2^2018 % 10 = 4) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_power_2018_l3423_342365


namespace NUMINAMATH_CALUDE_noodle_shop_solution_l3423_342318

/-- Represents the prices and sales of noodles in a shop -/
structure NoodleShop where
  dine_in_price : ℚ
  fresh_price : ℚ
  april_dine_in_sales : ℕ
  april_fresh_sales : ℕ
  may_fresh_price_decrease : ℚ
  may_fresh_sales_increase : ℚ
  may_total_sales_increase : ℚ

/-- Theorem stating the solution to the noodle shop problem -/
theorem noodle_shop_solution (shop : NoodleShop) : 
  (3 * shop.dine_in_price + 2 * shop.fresh_price = 31) →
  (4 * shop.dine_in_price + shop.fresh_price = 33) →
  (shop.april_dine_in_sales = 2500) →
  (shop.april_fresh_sales = 1500) →
  (shop.may_fresh_price_decrease = 3/4 * shop.may_total_sales_increase) →
  (shop.may_fresh_sales_increase = 5/2 * shop.may_total_sales_increase) →
  (shop.dine_in_price = 7) ∧ 
  (shop.fresh_price = 5) ∧ 
  (shop.may_total_sales_increase = 40/9) := by
  sorry


end NUMINAMATH_CALUDE_noodle_shop_solution_l3423_342318


namespace NUMINAMATH_CALUDE_system_solution_l3423_342382

theorem system_solution (a b c x y z : ℝ) : 
  (x + a * y + a^2 * z = a^3) →
  (x + b * y + b^2 * z = b^3) →
  (x + c * y + c^2 * z = c^3) →
  (x = a * b * c ∧ y = -(a * b + b * c + c * a) ∧ z = a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3423_342382


namespace NUMINAMATH_CALUDE_largest_common_term_proof_l3423_342380

/-- The first arithmetic progression with common difference 5 -/
def ap1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression with common difference 11 -/
def ap2 (n : ℕ) : ℕ := 7 + 11 * n

/-- A common term of both arithmetic progressions -/
def common_term (k m : ℕ) : Prop := ap1 k = ap2 m

/-- The largest common term less than 1000 -/
def largest_common_term : ℕ := 964

theorem largest_common_term_proof :
  (∃ k m : ℕ, common_term k m ∧ largest_common_term = ap1 k) ∧
  (∀ n : ℕ, n < 1000 → (∃ k m : ℕ, common_term k m ∧ n = ap1 k) → n ≤ largest_common_term) :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_proof_l3423_342380


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3423_342379

theorem division_remainder_proof (a b : ℕ) 
  (h1 : a - b = 2415)
  (h2 : a = 2520)
  (h3 : a / b = 21) : 
  a % b = 315 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3423_342379


namespace NUMINAMATH_CALUDE_log_inequality_implies_range_l3423_342339

theorem log_inequality_implies_range (x : ℝ) (hx : x > 0) :
  (Real.log x) ^ 2015 < (Real.log x) ^ 2014 ∧ 
  (Real.log x) ^ 2014 < (Real.log x) ^ 2016 →
  0 < x ∧ x < (1/10 : ℝ) := by sorry

end NUMINAMATH_CALUDE_log_inequality_implies_range_l3423_342339


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l3423_342321

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_difference (a b : ℝ) :
  symmetric_wrt_origin (a, -2) (4, b) → a - b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l3423_342321


namespace NUMINAMATH_CALUDE_eliza_basketball_scores_l3423_342392

def first_ten_games : List Nat := [9, 3, 5, 4, 8, 2, 5, 3, 7, 6]

def total_first_ten : Nat := first_ten_games.sum

theorem eliza_basketball_scores :
  ∃ (game11 game12 : Nat),
    game11 < 10 ∧
    game12 < 10 ∧
    (total_first_ten + game11) % 11 = 0 ∧
    (total_first_ten + game11 + game12) % 12 = 0 ∧
    game11 * game12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eliza_basketball_scores_l3423_342392


namespace NUMINAMATH_CALUDE_committee_seating_arrangements_l3423_342326

/-- The number of distinct arrangements of chairs and stools -/
def distinct_arrangements (n_women : ℕ) (n_men : ℕ) : ℕ :=
  Nat.choose (n_women + n_men - 1) (n_men - 1)

/-- Theorem stating the number of distinct arrangements for the given problem -/
theorem committee_seating_arrangements :
  distinct_arrangements 12 3 = 91 := by
  sorry

end NUMINAMATH_CALUDE_committee_seating_arrangements_l3423_342326


namespace NUMINAMATH_CALUDE_car_sale_profit_percentage_l3423_342301

/-- Calculates the profit percentage on a car sale given specific conditions --/
theorem car_sale_profit_percentage (P : ℝ) : 
  let discount_rate : ℝ := 0.1
  let discounted_price : ℝ := P * (1 - discount_rate)
  let first_year_expense_rate : ℝ := 0.05
  let second_year_expense_rate : ℝ := 0.04
  let third_year_expense_rate : ℝ := 0.03
  let selling_price_increase_rate : ℝ := 0.8
  
  let first_year_value : ℝ := discounted_price * (1 + first_year_expense_rate)
  let second_year_value : ℝ := first_year_value * (1 + second_year_expense_rate)
  let third_year_value : ℝ := second_year_value * (1 + third_year_expense_rate)
  
  let selling_price : ℝ := discounted_price * (1 + selling_price_increase_rate)
  let profit : ℝ := selling_price - P
  let profit_percentage : ℝ := (profit / P) * 100
  
  profit_percentage = 62 := by sorry

end NUMINAMATH_CALUDE_car_sale_profit_percentage_l3423_342301


namespace NUMINAMATH_CALUDE_range_of_mu_l3423_342387

theorem range_of_mu (a b μ : ℝ) (ha : a > 0) (hb : b > 0) (hμ : μ > 0) 
  (h : 1/a + 9/b = 1) : 
  (∀ a b, a > 0 → b > 0 → 1/a + 9/b = 1 → a + b ≥ μ) ↔ μ ∈ Set.Ioc 0 16 :=
by sorry

end NUMINAMATH_CALUDE_range_of_mu_l3423_342387


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l3423_342349

/-- 
Given a geometric sequence {a_n} with positive terms and common ratio q > 1,
if a_5 + a_4 - a_3 - a_2 = 5, then a_6 + a_7 ≥ 20.
-/
theorem geometric_sequence_minimum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  q > 1 →
  (∀ n, a (n + 1) = q * a n) →
  a 5 + a 4 - a 3 - a 2 = 5 →
  a 6 + a 7 ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l3423_342349


namespace NUMINAMATH_CALUDE_apple_sales_leftover_l3423_342308

/-- The number of apples left over after selling all possible baskets -/
def leftover_apples (oliver patricia quentin basket_size : ℕ) : ℕ :=
  (oliver + patricia + quentin) % basket_size

theorem apple_sales_leftover :
  leftover_apples 58 36 15 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_apple_sales_leftover_l3423_342308


namespace NUMINAMATH_CALUDE_variance_transform_l3423_342362

/-- The variance of a dataset -/
def variance (data : List ℝ) : ℝ := sorry

/-- Transform a dataset by multiplying each element by a and adding b -/
def transform (data : List ℝ) (a b : ℝ) : List ℝ := sorry

theorem variance_transform (data : List ℝ) (a b : ℝ) :
  variance data = 3 →
  variance (transform data a b) = 12 →
  |a| = 2 := by sorry

end NUMINAMATH_CALUDE_variance_transform_l3423_342362


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3423_342364

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3423_342364


namespace NUMINAMATH_CALUDE_probability_two_females_selected_l3423_342324

def total_contestants : ℕ := 7
def female_contestants : ℕ := 4
def male_contestants : ℕ := 3

theorem probability_two_females_selected :
  (Nat.choose female_contestants 2 : ℚ) / (Nat.choose total_contestants 2 : ℚ) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_selected_l3423_342324


namespace NUMINAMATH_CALUDE_tip_percentage_lower_limit_l3423_342356

theorem tip_percentage_lower_limit 
  (meal_cost : ℝ) 
  (total_paid : ℝ) 
  (tip_percentage : ℝ → Prop) : 
  meal_cost = 35.50 →
  total_paid = 40.825 →
  (∀ x, tip_percentage x → x ≥ 15 ∧ x < 25) →
  total_paid = meal_cost + (meal_cost * (15 / 100)) :=
by sorry

end NUMINAMATH_CALUDE_tip_percentage_lower_limit_l3423_342356


namespace NUMINAMATH_CALUDE_slope_of_l₃_l3423_342381

-- Define the lines and points
def l₁ (x y : ℝ) : Prop := 4 * x - 3 * y = 2
def l₂ (x y : ℝ) : Prop := y = 2
def A : ℝ × ℝ := (0, -3)

-- Define the existence of point B
def B_exists : Prop := ∃ x y : ℝ, l₁ x y ∧ l₂ x y

-- Define the existence of point C
def C_exists : Prop := ∃ x : ℝ, l₂ x (2 : ℝ)

-- Define the properties of line l₃
def l₃_properties (m : ℝ) : Prop :=
  m > 0 ∧ 
  (∃ b : ℝ, ∀ x : ℝ, m * x + b = -3 → x = 0) ∧
  (∃ x : ℝ, l₂ x (m * x + -3))

-- Define the area of triangle ABC
def triangle_area (m : ℝ) : Prop :=
  ∃ B C : ℝ × ℝ, 
    l₁ B.1 B.2 ∧ l₂ B.1 B.2 ∧
    l₂ C.1 C.2 ∧
    (C.2 - A.2) = m * (C.1 - A.1) ∧
    abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) = 10

-- Theorem statement
theorem slope_of_l₃ :
  B_exists → C_exists → ∃ m : ℝ, l₃_properties m ∧ triangle_area m → m = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_l₃_l3423_342381
