import Mathlib

namespace NUMINAMATH_CALUDE_last_number_in_sequence_l1475_147540

theorem last_number_in_sequence (x : ℕ) : 
  1000 + 20 + 1000 + 30 + 1000 + 40 + 1000 + x = 4100 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_last_number_in_sequence_l1475_147540


namespace NUMINAMATH_CALUDE_intersection_M_N_l1475_147515

-- Define set M
def M : Set ℝ := {x | x^2 + 2*x - 3 = 0}

-- Define set N
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (2^x - 1/2)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1475_147515


namespace NUMINAMATH_CALUDE_leo_current_weight_l1475_147559

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 92

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 160 - leo_weight

/-- The combined weight of Leo and Kendra in pounds -/
def combined_weight : ℝ := 160

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = combined_weight) ∧
  (leo_weight = 92) := by
sorry

end NUMINAMATH_CALUDE_leo_current_weight_l1475_147559


namespace NUMINAMATH_CALUDE_usb_cost_problem_l1475_147548

/-- Given that three identical USBs cost $45, prove that seven such USBs cost $105. -/
theorem usb_cost_problem (cost_of_three : ℝ) (h1 : cost_of_three = 45) : 
  (7 / 3) * cost_of_three = 105 := by
  sorry

end NUMINAMATH_CALUDE_usb_cost_problem_l1475_147548


namespace NUMINAMATH_CALUDE_fifth_diagram_shaded_fraction_l1475_147549

/-- Represents the number of shaded triangles in the n-th diagram -/
def shadedTriangles (n : ℕ) : ℕ := 2^(n - 1)

/-- Represents the total number of triangles in the n-th diagram -/
def totalTriangles (n : ℕ) : ℕ := n^2

/-- The fraction of shaded triangles in the n-th diagram -/
def shadedFraction (n : ℕ) : ℚ :=
  (shadedTriangles n : ℚ) / (totalTriangles n : ℚ)

theorem fifth_diagram_shaded_fraction :
  shadedFraction 5 = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fifth_diagram_shaded_fraction_l1475_147549


namespace NUMINAMATH_CALUDE_min_sum_given_product_min_sum_equality_case_l1475_147509

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + 8*b = a*b → a + b ≥ 18 := by
  sorry

-- The equality case
theorem min_sum_equality_case (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + 8*b = a*b → (a + b = 18 ↔ a = 12 ∧ b = 6) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_min_sum_equality_case_l1475_147509


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_l1475_147525

theorem imaginary_part_of_one_plus_i :
  let z : ℂ := 1 + Complex.I
  Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_l1475_147525


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1475_147573

theorem sum_of_cubes_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 5 * x₁ - 2 = 0) → 
  (3 * x₂^2 - 5 * x₂ - 2 = 0) → 
  x₁^3 + x₂^3 = 215 / 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1475_147573


namespace NUMINAMATH_CALUDE_sara_flowers_l1475_147533

/-- Given the number of red flowers and the number of bouquets, 
    calculate the number of yellow flowers needed to create bouquets 
    with an equal number of red and yellow flowers in each. -/
def yellow_flowers (red_flowers : ℕ) (num_bouquets : ℕ) : ℕ :=
  (red_flowers / num_bouquets) * num_bouquets

/-- Theorem stating that given 16 red flowers and 8 bouquets,
    the number of yellow flowers needed is 16. -/
theorem sara_flowers : yellow_flowers 16 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sara_flowers_l1475_147533


namespace NUMINAMATH_CALUDE_cody_tickets_l1475_147539

theorem cody_tickets (initial : ℝ) (lost : ℝ) (spent : ℝ) : 
  initial = 49.0 → lost = 6.0 → spent = 25.0 → initial - lost - spent = 18.0 :=
by sorry

end NUMINAMATH_CALUDE_cody_tickets_l1475_147539


namespace NUMINAMATH_CALUDE_planes_parallel_condition_l1475_147511

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_condition 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β)
  (h3 : perpendicular m α) 
  (h4 : perpendicular n β) 
  (h5 : parallel_lines m n) : 
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_condition_l1475_147511


namespace NUMINAMATH_CALUDE_rectangle_area_l1475_147519

/-- Given a rectangle with a length to width ratio of 0.875 and a width of 24 centimeters,
    its area is 504 square centimeters. -/
theorem rectangle_area (ratio : ℝ) (width : ℝ) (h1 : ratio = 0.875) (h2 : width = 24) :
  ratio * width * width = 504 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1475_147519


namespace NUMINAMATH_CALUDE_average_running_distance_l1475_147584

def monday_distance : ℝ := 4.2
def tuesday_distance : ℝ := 3.8
def wednesday_distance : ℝ := 3.6
def thursday_distance : ℝ := 4.4
def number_of_days : ℕ := 4

def total_distance : ℝ := monday_distance + tuesday_distance + wednesday_distance + thursday_distance

theorem average_running_distance :
  total_distance / number_of_days = 4 := by sorry

end NUMINAMATH_CALUDE_average_running_distance_l1475_147584


namespace NUMINAMATH_CALUDE_expand_expression_l1475_147502

theorem expand_expression (x : ℝ) : 25 * (3 * x - 4) = 75 * x - 100 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1475_147502


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1475_147583

theorem sqrt_product_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (45 * x^2) * Real.sqrt (8 * x^3) * Real.sqrt (22 * x) = 60 * x^3 * Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1475_147583


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l1475_147520

theorem inequality_system_solutions :
  let S : Set ℤ := {x | x ≥ 0 ∧ 2*x + 5 ≤ 3*(x + 2) ∧ 2*x - (1 + 3*x)/2 < 1}
  S = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l1475_147520


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l1475_147571

theorem line_ellipse_intersection (k : ℝ) : ∃ (x y : ℝ), 
  (y = k * x + 1 - k) ∧ (x^2 / 9 + y^2 / 4 = 1) := by
  sorry

#check line_ellipse_intersection

end NUMINAMATH_CALUDE_line_ellipse_intersection_l1475_147571


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l1475_147501

theorem complex_imaginary_part (a : ℝ) (z : ℂ) : 
  z = 1 + a * I →  -- z is of the form 1 + ai
  a > 0 →  -- z is in the first quadrant
  Complex.abs z = Real.sqrt 5 →  -- |z| = √5
  z.im = 2 :=  -- The imaginary part of z is 2
by sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l1475_147501


namespace NUMINAMATH_CALUDE_remainder_3123_div_28_l1475_147593

theorem remainder_3123_div_28 : 3123 % 28 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3123_div_28_l1475_147593


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_l1475_147506

theorem tangent_line_to_ellipse (m : ℝ) :
  (∃! x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 1) →
  m^2 = 35/9 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_l1475_147506


namespace NUMINAMATH_CALUDE_blue_car_fraction_l1475_147585

theorem blue_car_fraction (total : ℕ) (black : ℕ) : 
  total = 516 →
  black = 86 →
  (total - (total / 2 + black)) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_car_fraction_l1475_147585


namespace NUMINAMATH_CALUDE_eagle_speed_proof_l1475_147543

/-- The speed of the eagle in miles per hour -/
def eagle_speed : ℝ := 15

/-- The speed of the falcon in miles per hour -/
def falcon_speed : ℝ := 46

/-- The speed of the pelican in miles per hour -/
def pelican_speed : ℝ := 33

/-- The speed of the hummingbird in miles per hour -/
def hummingbird_speed : ℝ := 30

/-- The time all birds flew in hours -/
def flight_time : ℝ := 2

/-- The total distance covered by all birds in miles -/
def total_distance : ℝ := 248

theorem eagle_speed_proof :
  eagle_speed * flight_time +
  falcon_speed * flight_time +
  pelican_speed * flight_time +
  hummingbird_speed * flight_time =
  total_distance :=
sorry

end NUMINAMATH_CALUDE_eagle_speed_proof_l1475_147543


namespace NUMINAMATH_CALUDE_circle_polar_equation_l1475_147529

/-- The polar coordinate equation of a circle C, given specific conditions -/
theorem circle_polar_equation (C : Set (ℝ × ℝ)) (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) :
  (P = (Real.sqrt 2, π / 4)) →
  (∀ ρ θ, l ρ θ ↔ ρ * Real.sin (θ - π / 3) = -Real.sqrt 3 / 2) →
  (∃ x, x ∈ C ∧ x.1 = 1 ∧ x.2 = 0) →
  (P ∈ C) →
  (∀ ρ θ, (ρ, θ) ∈ C ↔ ρ = 2 * Real.cos θ) :=
by sorry

end NUMINAMATH_CALUDE_circle_polar_equation_l1475_147529


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1475_147596

theorem inequality_solution_set (x : ℝ) : 
  (x / (x + 1) + (x + 3) / (2 * x) ≥ 2) ↔ (0 < x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1475_147596


namespace NUMINAMATH_CALUDE_penny_percentage_theorem_l1475_147598

theorem penny_percentage_theorem (initial_pennies : ℕ) 
                                 (old_pennies : ℕ) 
                                 (final_pennies : ℕ) : 
  initial_pennies = 200 →
  old_pennies = 30 →
  final_pennies = 136 →
  (initial_pennies - old_pennies) * (1 - 20 / 100) = final_pennies :=
by
  sorry

end NUMINAMATH_CALUDE_penny_percentage_theorem_l1475_147598


namespace NUMINAMATH_CALUDE_identifier_count_l1475_147528

/-- The number of English letters -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible characters for the second and third positions -/
def num_chars : ℕ := num_letters + num_digits

/-- The total number of possible identifiers -/
def total_identifiers : ℕ := num_letters + (num_letters * num_chars) + (num_letters * num_chars * num_chars)

theorem identifier_count : total_identifiers = 34658 := by
  sorry

end NUMINAMATH_CALUDE_identifier_count_l1475_147528


namespace NUMINAMATH_CALUDE_line_x_intercept_l1475_147534

/-- Given a line with slope 3/4 passing through (-12, -39), prove its x-intercept is 40 -/
theorem line_x_intercept :
  let m : ℚ := 3/4  -- slope
  let x₀ : ℤ := -12
  let y₀ : ℤ := -39
  let b : ℚ := y₀ - m * x₀  -- y-intercept
  let x_intercept : ℚ := -b / m  -- x-coordinate where y = 0
  x_intercept = 40 := by
sorry

end NUMINAMATH_CALUDE_line_x_intercept_l1475_147534


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_real_x_l1475_147563

theorem inequality_holds_for_all_real_x : ∀ x : ℝ, 2^(Real.sin x)^2 + 2^(Real.cos x)^2 ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_real_x_l1475_147563


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1475_147504

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties (a b c : ℝ) :
  (f' a b (-2/3) = 0 ∧ f' a b 1 = 0) →
  (a = -1/2 ∧ b = -2) ∧
  (∀ x : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f (-1/2) (-2) c x₁ = 0 ∧ f (-1/2) (-2) c x₂ = 0 ∧ f (-1/2) (-2) c x₃ = 0) →
    -22/27 < c ∧ c < 3/2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1475_147504


namespace NUMINAMATH_CALUDE_truncated_cube_edges_l1475_147541

/-- A truncated cube is a polyhedron obtained by truncating each vertex of a cube
    such that a small square face replaces each vertex, and no cutting planes
    intersect each other inside the cube. -/
structure TruncatedCube where
  -- We don't need to define the internal structure, just the concept

/-- The number of edges in a truncated cube -/
def num_edges (tc : TruncatedCube) : ℕ := 16

/-- Theorem stating that the number of edges in a truncated cube is 16 -/
theorem truncated_cube_edges (tc : TruncatedCube) :
  num_edges tc = 16 := by sorry

end NUMINAMATH_CALUDE_truncated_cube_edges_l1475_147541


namespace NUMINAMATH_CALUDE_castle_doors_problem_l1475_147594

theorem castle_doors_problem (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_castle_doors_problem_l1475_147594


namespace NUMINAMATH_CALUDE_sum_equals_four_sqrt_860_l1475_147558

theorem sum_equals_four_sqrt_860 (a b c d : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_squares : a^2 + b^2 = 2016 ∧ c^2 + d^2 = 2016)
  (products : a*c = 1008 ∧ b*d = 1008) :
  a + b + c + d = 4 * Real.sqrt 860 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_four_sqrt_860_l1475_147558


namespace NUMINAMATH_CALUDE_bicycle_selling_prices_l1475_147545

def calculate_selling_price (purchase_price : ℕ) (loss_percentage : ℕ) : ℕ :=
  purchase_price - (purchase_price * loss_percentage / 100)

def bicycle1_purchase_price : ℕ := 1800
def bicycle1_loss_percentage : ℕ := 25

def bicycle2_purchase_price : ℕ := 2700
def bicycle2_loss_percentage : ℕ := 15

def bicycle3_purchase_price : ℕ := 2200
def bicycle3_loss_percentage : ℕ := 20

theorem bicycle_selling_prices :
  (calculate_selling_price bicycle1_purchase_price bicycle1_loss_percentage = 1350) ∧
  (calculate_selling_price bicycle2_purchase_price bicycle2_loss_percentage = 2295) ∧
  (calculate_selling_price bicycle3_purchase_price bicycle3_loss_percentage = 1760) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_selling_prices_l1475_147545


namespace NUMINAMATH_CALUDE_area_of_enclosed_region_l1475_147505

/-- The equation of the curve enclosing the region -/
def curve_equation (x y : ℝ) : Prop :=
  x^2 - 18*x + 3*y + 90 = 33 + 9*y - y^2

/-- The equation of the line bounding the region above -/
def line_equation (x y : ℝ) : Prop :=
  y = x - 5

/-- The region enclosed by the curve and below the line -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve_equation p.1 p.2 ∧ p.2 ≤ p.1 - 5}

/-- The area of the enclosed region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_enclosed_region :
  area_of_region = 33 * Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_area_of_enclosed_region_l1475_147505


namespace NUMINAMATH_CALUDE_artwork_transaction_l1475_147503

/-- Converts a number from base s to base 10 -/
def to_base_10 (digits : List Nat) (s : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * s^i) 0

theorem artwork_transaction (s : Nat) : 
  s > 1 →
  to_base_10 [0, 3, 5] s + to_base_10 [0, 3, 2, 1] s = to_base_10 [0, 0, 0, 2] s →
  s = 8 := by
sorry

end NUMINAMATH_CALUDE_artwork_transaction_l1475_147503


namespace NUMINAMATH_CALUDE_stone_to_crystal_ratio_is_two_to_one_l1475_147546

/-- A bracelet making scenario with Nancy and Rose -/
structure BraceletScenario where
  beads_per_bracelet : ℕ
  nancy_metal_beads : ℕ
  nancy_pearl_beads : ℕ
  rose_crystal_beads : ℕ
  total_bracelets : ℕ

/-- Calculate the ratio of Rose's stone beads to crystal beads -/
def stone_to_crystal_ratio (scenario : BraceletScenario) : ℚ :=
  let total_beads := scenario.total_bracelets * scenario.beads_per_bracelet
  let nancy_total_beads := scenario.nancy_metal_beads + scenario.nancy_pearl_beads
  let rose_total_beads := total_beads - nancy_total_beads
  let rose_stone_beads := rose_total_beads - scenario.rose_crystal_beads
  (rose_stone_beads : ℚ) / scenario.rose_crystal_beads

/-- The given bracelet scenario -/
def given_scenario : BraceletScenario :=
  { beads_per_bracelet := 8
  , nancy_metal_beads := 40
  , nancy_pearl_beads := 60  -- 40 + 20
  , rose_crystal_beads := 20
  , total_bracelets := 20 }

theorem stone_to_crystal_ratio_is_two_to_one :
  stone_to_crystal_ratio given_scenario = 2 := by
  sorry

end NUMINAMATH_CALUDE_stone_to_crystal_ratio_is_two_to_one_l1475_147546


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1475_147536

theorem sufficient_condition_for_inequality (x : ℝ) : 
  (∀ x, x > 1 → 1 - 1/x > 0) ∧ 
  (∃ x, 1 - 1/x > 0 ∧ ¬(x > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1475_147536


namespace NUMINAMATH_CALUDE_pond_area_is_292_l1475_147518

/-- The total surface area of a cuboid-shaped pond, excluding the top surface. -/
def pondSurfaceArea (length width height : ℝ) : ℝ :=
  length * width + 2 * length * height + 2 * width * height

/-- Theorem: The surface area of a pond with given dimensions is 292 square meters. -/
theorem pond_area_is_292 :
  pondSurfaceArea 18 10 2 = 292 := by
  sorry

#eval pondSurfaceArea 18 10 2

end NUMINAMATH_CALUDE_pond_area_is_292_l1475_147518


namespace NUMINAMATH_CALUDE_workshop_workers_l1475_147517

/-- Represents the total number of workers in a workshop -/
def total_workers : ℕ := 20

/-- Represents the number of technicians -/
def technicians : ℕ := 5

/-- Represents the average salary of all workers -/
def avg_salary_all : ℕ := 750

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℕ := 900

/-- Represents the average salary of non-technician workers -/
def avg_salary_others : ℕ := 700

/-- Theorem stating that given the conditions, the total number of workers is 20 -/
theorem workshop_workers : 
  (total_workers * avg_salary_all = technicians * avg_salary_technicians + 
   (total_workers - technicians) * avg_salary_others) → 
  total_workers = 20 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l1475_147517


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l1475_147579

def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_sum_of_coefficients 
  (a b c d : ℝ) 
  (h1 : g a b c d (3*Complex.I) = 0)
  (h2 : g a b c d (3 + Complex.I) = 0) :
  a + b + c + d = 49 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l1475_147579


namespace NUMINAMATH_CALUDE_f_is_decreasing_and_odd_l1475_147521

-- Define the function f(x) = -x
def f (x : ℝ) : ℝ := -x

-- State the theorem
theorem f_is_decreasing_and_odd :
  (∀ x y : ℝ, x < y → f x > f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_f_is_decreasing_and_odd_l1475_147521


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_forty_l1475_147507

theorem fraction_of_fraction_of_forty : (2/3 : ℚ) * ((3/4 : ℚ) * 40) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_forty_l1475_147507


namespace NUMINAMATH_CALUDE_quiz_correct_answers_l1475_147592

theorem quiz_correct_answers
  (wendy_correct : ℕ)
  (campbell_correct : ℕ)
  (kelsey_correct : ℕ)
  (martin_correct : ℕ)
  (h1 : wendy_correct = 20)
  (h2 : campbell_correct = 2 * wendy_correct)
  (h3 : kelsey_correct = campbell_correct + 8)
  (h4 : martin_correct = kelsey_correct - 3) :
  martin_correct = 45 := by
  sorry

end NUMINAMATH_CALUDE_quiz_correct_answers_l1475_147592


namespace NUMINAMATH_CALUDE_f_g_f_3_equals_630_l1475_147564

def f (x : ℝ) : ℝ := 5 * x + 5

def g (x : ℝ) : ℝ := 6 * x + 5

theorem f_g_f_3_equals_630 : f (g (f 3)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_3_equals_630_l1475_147564


namespace NUMINAMATH_CALUDE_rachel_homework_pages_l1475_147574

theorem rachel_homework_pages (math_pages reading_pages biology_pages : ℕ) 
  (h1 : math_pages = 2)
  (h2 : reading_pages = 3)
  (h3 : biology_pages = 10) :
  math_pages + reading_pages + biology_pages = 15 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_pages_l1475_147574


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l1475_147578

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_degree : ∀ v : Fin 20, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that form an edge in a regular dodecahedron -/
def edge_selection_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem: The probability of randomly selecting two vertices that are endpoints 
    of the same edge in a regular dodecahedron is 3/19 -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) : 
  edge_selection_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l1475_147578


namespace NUMINAMATH_CALUDE_functions_properties_l1475_147591

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

-- Theorem statement
theorem functions_properties :
  (∀ x : ℝ, f x < g x) ∧
  (∃ x : ℝ, f x ^ 2 + g x ^ 2 ≥ 1) ∧
  (∀ x : ℝ, f (2 * x) = 2 * f x * g x) := by
  sorry

end NUMINAMATH_CALUDE_functions_properties_l1475_147591


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_l1475_147532

theorem pen_pencil_ratio (num_pencils : ℕ) (num_pens : ℕ) : 
  num_pencils = 36 → 
  num_pencils = num_pens + 6 → 
  (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_l1475_147532


namespace NUMINAMATH_CALUDE_two_three_four_forms_triangle_one_two_three_not_triangle_two_two_four_not_triangle_two_three_six_not_triangle_triangle_formation_theorem_l1475_147599

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem stating that (2, 3, 4) can form a triangle
theorem two_three_four_forms_triangle :
  can_form_triangle 2 3 4 := by sorry

-- Theorem stating that (1, 2, 3) cannot form a triangle
theorem one_two_three_not_triangle :
  ¬ can_form_triangle 1 2 3 := by sorry

-- Theorem stating that (2, 2, 4) cannot form a triangle
theorem two_two_four_not_triangle :
  ¬ can_form_triangle 2 2 4 := by sorry

-- Theorem stating that (2, 3, 6) cannot form a triangle
theorem two_three_six_not_triangle :
  ¬ can_form_triangle 2 3 6 := by sorry

-- Main theorem combining all results
theorem triangle_formation_theorem :
  can_form_triangle 2 3 4 ∧
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 2 2 4 ∧
  ¬ can_form_triangle 2 3 6 := by sorry

end NUMINAMATH_CALUDE_two_three_four_forms_triangle_one_two_three_not_triangle_two_two_four_not_triangle_two_three_six_not_triangle_triangle_formation_theorem_l1475_147599


namespace NUMINAMATH_CALUDE_trig_equation_solution_l1475_147570

theorem trig_equation_solution (x : Real) : 
  (6 * Real.sin x ^ 2 + Real.sin x * Real.cos x - Real.cos x ^ 2 = 2) ↔ 
  (∃ k : Int, x = -π/4 + π * k) ∨ 
  (∃ n : Int, x = Real.arctan (3/4) + π * n) := by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l1475_147570


namespace NUMINAMATH_CALUDE_officer_arrival_time_l1475_147550

/-- The designated arrival time for an officer traveling from A to B -/
noncomputable def designated_arrival_time (s v : ℝ) : ℝ :=
  (v + Real.sqrt (9 * v^2 + 6 * v * s)) / v

theorem officer_arrival_time (s v : ℝ) (h_s : s > 0) (h_v : v > 0) :
  let t := designated_arrival_time s v
  let initial_speed := s / (t + 2)
  s / initial_speed = t + 2 ∧
  s / (2 * initial_speed) + 1 + s / (2 * (initial_speed + v)) = t :=
by sorry

end NUMINAMATH_CALUDE_officer_arrival_time_l1475_147550


namespace NUMINAMATH_CALUDE_cricket_bat_selling_price_l1475_147535

theorem cricket_bat_selling_price 
  (profit : ℝ) 
  (profit_percentage : ℝ) 
  (selling_price : ℝ) : 
  profit = 300 → 
  profit_percentage = 50 → 
  selling_price = profit * (100 / profit_percentage) + profit → 
  selling_price = 900 := by
sorry

end NUMINAMATH_CALUDE_cricket_bat_selling_price_l1475_147535


namespace NUMINAMATH_CALUDE_min_value_ab_l1475_147538

theorem min_value_ab (a b : ℝ) (h : (4 / a) + (1 / b) = Real.sqrt (a * b)) : 
  ∀ x y : ℝ, ((4 / x) + (1 / y) = Real.sqrt (x * y)) → (a * b) ≤ (x * y) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l1475_147538


namespace NUMINAMATH_CALUDE_smallest_cube_for_cone_l1475_147567

/-- Represents a right circular cone -/
structure Cone where
  height : ℝ
  baseDiameter : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- The volume of a cube -/
def cubeVolume (c : Cube) : ℝ :=
  c.sideLength ^ 3

/-- Predicate to check if a cube can contain a cone upright -/
def canContainCone (cube : Cube) (cone : Cone) : Prop :=
  cube.sideLength ≥ cone.height ∧ cube.sideLength ≥ cone.baseDiameter

theorem smallest_cube_for_cone (cone : Cone) 
    (h1 : cone.height = 15)
    (h2 : cone.baseDiameter = 8) :
    ∃ (cube : Cube), 
      canContainCone cube cone ∧ 
      cubeVolume cube = 3375 ∧
      ∀ (c : Cube), canContainCone c cone → cubeVolume c ≥ 3375 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_for_cone_l1475_147567


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l1475_147562

theorem adult_ticket_cost (child_ticket_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (adult_tickets : ℕ) :
  child_ticket_cost = 4 →
  total_tickets = 900 →
  total_revenue = 5100 →
  adult_tickets = 500 →
  ∃ (adult_ticket_cost : ℕ), adult_ticket_cost = 7 ∧
    adult_ticket_cost * adult_tickets + child_ticket_cost * (total_tickets - adult_tickets) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l1475_147562


namespace NUMINAMATH_CALUDE_defective_units_percentage_l1475_147576

theorem defective_units_percentage
  (shipped_defective_ratio : Real)
  (total_shipped_defective_ratio : Real)
  (h1 : shipped_defective_ratio = 0.05)
  (h2 : total_shipped_defective_ratio = 0.0035) :
  ∃ (defective_ratio : Real),
    defective_ratio = 0.07 ∧
    shipped_defective_ratio * defective_ratio = total_shipped_defective_ratio :=
by sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l1475_147576


namespace NUMINAMATH_CALUDE_boys_in_second_grade_is_20_l1475_147542

/-- The number of boys in the second grade -/
def boys_in_second_grade : ℕ := sorry

/-- The number of girls in the second grade -/
def girls_in_second_grade : ℕ := 11

/-- The total number of students in the second grade -/
def students_in_second_grade : ℕ := boys_in_second_grade + girls_in_second_grade

/-- The number of students in the third grade -/
def students_in_third_grade : ℕ := 2 * students_in_second_grade

/-- The total number of students in grades 2 and 3 -/
def total_students : ℕ := 93

theorem boys_in_second_grade_is_20 :
  boys_in_second_grade = 20 ∧
  students_in_second_grade + students_in_third_grade = total_students :=
sorry

end NUMINAMATH_CALUDE_boys_in_second_grade_is_20_l1475_147542


namespace NUMINAMATH_CALUDE_consecutive_product_bound_l1475_147510

theorem consecutive_product_bound (π : Fin 90 → Fin 90) (h : Function.Bijective π) :
  ∃ i : Fin 89, (π i).val * (π (i + 1)).val ≥ 2014 ∨
    (π (Fin.last 89)).val * (π 0).val ≥ 2014 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_bound_l1475_147510


namespace NUMINAMATH_CALUDE_probability_divisible_by_5_l1475_147577

/-- A three-digit number is an integer between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The count of three-digit numbers divisible by 5. -/
def CountDivisibleBy5 : ℕ := 180

/-- The total count of three-digit numbers. -/
def TotalThreeDigitNumbers : ℕ := 900

/-- The probability of a randomly chosen three-digit number being divisible by 5 is 1/5. -/
theorem probability_divisible_by_5 :
  (CountDivisibleBy5 : ℚ) / TotalThreeDigitNumbers = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_probability_divisible_by_5_l1475_147577


namespace NUMINAMATH_CALUDE_star_self_inverse_l1475_147513

/-- The star operation for real numbers -/
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

/-- Theorem: The star operation of (x^2 - y^2) and (y^2 - x^2) is zero -/
theorem star_self_inverse (x y : ℝ) : star (x^2 - y^2) (y^2 - x^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_self_inverse_l1475_147513


namespace NUMINAMATH_CALUDE_correct_factorization_l1475_147580

theorem correct_factorization (a : ℝ) : 2*a^2 - 4*a + 2 = 2*(a-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1475_147580


namespace NUMINAMATH_CALUDE_homework_difference_l1475_147552

theorem homework_difference (reading_pages math_pages biology_pages : ℕ) 
  (h1 : reading_pages = 4)
  (h2 : math_pages = 7)
  (h3 : biology_pages = 19) :
  math_pages - reading_pages = 3 :=
by sorry

end NUMINAMATH_CALUDE_homework_difference_l1475_147552


namespace NUMINAMATH_CALUDE_system_solution_l1475_147526

theorem system_solution :
  let x₁ : ℝ := 5
  let x₂ : ℝ := -5
  let x₃ : ℝ := 0
  let x₄ : ℝ := 2
  let x₅ : ℝ := -1
  let x₆ : ℝ := 1
  (x₁ + x₃ + 2*x₄ + 3*x₅ - 4*x₆ = 20) ∧
  (2*x₁ + x₂ - 3*x₃ + x₅ + 2*x₆ = -13) ∧
  (5*x₁ - x₂ + x₃ + 2*x₄ + 6*x₅ = 20) ∧
  (2*x₁ - 2*x₂ + 3*x₃ + 2*x₅ + 2*x₆ = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1475_147526


namespace NUMINAMATH_CALUDE_remaining_distance_l1475_147590

def total_distance : ℝ := 300
def speed : ℝ := 60
def time : ℝ := 2

theorem remaining_distance : total_distance - speed * time = 180 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l1475_147590


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1475_147561

theorem sqrt_equation_solution :
  ∃! x : ℚ, Real.sqrt (5 - 4 * x) = 6 :=
by
  use -31/4
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1475_147561


namespace NUMINAMATH_CALUDE_largest_divisible_n_l1475_147572

theorem largest_divisible_n : ∃ (n : ℕ), n = 910 ∧ 
  (∀ m : ℕ, m > n → ¬(m - 10 ∣ m^3 - 100)) ∧ 
  (n - 10 ∣ n^3 - 100) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l1475_147572


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1475_147566

def rental_cost : ℝ := 150
def gas_needed : ℝ := 8
def gas_price : ℝ := 3.50
def mileage_expense : ℝ := 0.50
def distance_driven : ℝ := 320

theorem total_cost_calculation :
  rental_cost + gas_needed * gas_price + distance_driven * mileage_expense = 338 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1475_147566


namespace NUMINAMATH_CALUDE_positive_c_in_quadratic_with_no_roots_l1475_147560

/-- A quadratic trinomial with no roots and positive sum of coefficients has a positive constant term. -/
theorem positive_c_in_quadratic_with_no_roots 
  (a b c : ℝ) 
  (no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) 
  (sum_positive : a + b + c > 0) : 
  c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_c_in_quadratic_with_no_roots_l1475_147560


namespace NUMINAMATH_CALUDE_mixture_volume_l1475_147531

/-- Given a mixture of two liquids p and q with an initial ratio of 5:3,
    if adding 15 liters of liquid q changes the ratio to 5:6,
    then the initial volume of the mixture was 40 liters. -/
theorem mixture_volume (p q : ℝ) (h1 : p / q = 5 / 3) 
    (h2 : p / (q + 15) = 5 / 6) : p + q = 40 := by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_l1475_147531


namespace NUMINAMATH_CALUDE_remainder_of_x_l1475_147589

theorem remainder_of_x (x : ℤ) 
  (h1 : (2 + x) % (4^3) = 3^2 % (4^3))
  (h2 : (3 + x) % (5^3) = 5^2 % (5^3))
  (h3 : (6 + x) % (11^3) = 7^2 % (11^3)) :
  x % 220 = 192 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_x_l1475_147589


namespace NUMINAMATH_CALUDE_circle_tangent_to_axes_on_line_l1475_147527

theorem circle_tangent_to_axes_on_line (x y : ℝ) :
  ∃ (a b r : ℝ),
    (∀ t : ℝ, (2 * t - (2 * t + 6) + 6 = 0)) →  -- Center on the line 2x - y + 6 = 0
    (a = -2 ∨ a = -6) →                         -- Possible x-coordinates of the center
    (b = 2 * a + 6) →                           -- y-coordinate of the center
    (r = |a|) →                                 -- Radius equals the absolute value of x-coordinate
    (r = |b|) →                                 -- Radius equals the absolute value of y-coordinate
    ((x + a)^2 + (y - b)^2 = r^2) →             -- Standard form of circle equation
    (((x + 2)^2 + (y - 2)^2 = 4) ∨ ((x + 6)^2 + (y + 6)^2 = 36)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_axes_on_line_l1475_147527


namespace NUMINAMATH_CALUDE_problem_solution_l1475_147582

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then x^2 - 4 else |x - 3| + a

theorem problem_solution (a : ℝ) :
  f a (f a (Real.sqrt 6)) = 3 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1475_147582


namespace NUMINAMATH_CALUDE_jack_payback_l1475_147554

/-- The amount borrowed by Jack from Jill -/
def principal : ℝ := 1200

/-- The interest rate on the loan -/
def interest_rate : ℝ := 0.1

/-- The total amount Jack will pay back -/
def total_amount : ℝ := principal * (1 + interest_rate)

/-- Theorem stating that Jack will pay back $1320 -/
theorem jack_payback : total_amount = 1320 := by
  sorry

end NUMINAMATH_CALUDE_jack_payback_l1475_147554


namespace NUMINAMATH_CALUDE_school_fee_calculation_l1475_147565

/-- Represents the number of bills of each denomination given by a parent -/
structure BillCount where
  fifty : Nat
  twenty : Nat
  ten : Nat

/-- Calculates the total value of bills given by a parent -/
def totalValue (bills : BillCount) : Nat :=
  50 * bills.fifty + 20 * bills.twenty + 10 * bills.ten

theorem school_fee_calculation (mother father : BillCount)
    (h_mother : mother = { fifty := 1, twenty := 2, ten := 3 })
    (h_father : father = { fifty := 4, twenty := 1, ten := 1 }) :
    totalValue mother + totalValue father = 350 := by
  sorry

end NUMINAMATH_CALUDE_school_fee_calculation_l1475_147565


namespace NUMINAMATH_CALUDE_line_point_sum_l1475_147500

/-- The line equation y = -5/3x + 10 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 10

/-- Point P is on the x-axis -/
def P_on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

/-- Point Q is on the y-axis -/
def Q_on_y_axis (Q : ℝ × ℝ) : Prop := Q.1 = 0

/-- Point T is on line segment PQ -/
def T_on_PQ (P Q T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ T = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2)

/-- Area of triangle POQ is 4 times the area of triangle TOP -/
def area_ratio (P Q T : ℝ × ℝ) : Prop :=
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) =
  4 * abs ((T.1 - 0) * (P.2 - 0) - (P.1 - 0) * (T.2 - 0))

theorem line_point_sum (P Q T : ℝ × ℝ) (r s : ℝ) :
  line_equation P.1 P.2 →
  line_equation Q.1 Q.2 →
  line_equation T.1 T.2 →
  P_on_x_axis P →
  Q_on_y_axis Q →
  T_on_PQ P Q T →
  area_ratio P Q T →
  T = (r, s) →
  r + s = 7 := by sorry

end NUMINAMATH_CALUDE_line_point_sum_l1475_147500


namespace NUMINAMATH_CALUDE_xiao_ying_pays_20_yuan_l1475_147551

/-- Represents the price of flowers in yuan -/
structure FlowerPrices where
  rose : ℚ
  carnation : ℚ
  lily : ℚ

/-- The conditions from Xiao Hong's and Xiao Li's purchases -/
def satisfies_conditions (p : FlowerPrices) : Prop :=
  3 * p.rose + 7 * p.carnation + p.lily = 14 ∧
  4 * p.rose + 10 * p.carnation + p.lily = 16

/-- Xiao Ying's purchase -/
def xiao_ying_purchase (p : FlowerPrices) : ℚ :=
  2 * (p.rose + p.carnation + p.lily)

/-- The main theorem to prove -/
theorem xiao_ying_pays_20_yuan (p : FlowerPrices) :
  satisfies_conditions p → xiao_ying_purchase p = 20 := by
  sorry


end NUMINAMATH_CALUDE_xiao_ying_pays_20_yuan_l1475_147551


namespace NUMINAMATH_CALUDE_committee_probability_l1475_147516

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

theorem committee_probability :
  let total_ways := Nat.choose total_members committee_size
  let unwanted_cases := Nat.choose num_girls committee_size +
                        num_boys * Nat.choose num_girls (committee_size - 1) +
                        Nat.choose num_boys committee_size +
                        num_girls * Nat.choose num_boys (committee_size - 1)
  (total_ways - unwanted_cases : ℚ) / total_ways = 457215 / 593775 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l1475_147516


namespace NUMINAMATH_CALUDE_f_range_on_interval_l1475_147524

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem f_range_on_interval :
  ∀ x ∈ Set.Icc (-1 : ℝ) 2,
  2 ≤ f x ∧ f x ≤ 6 ∧
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 2, f x₁ = 2) ∧
  (∃ x₂ ∈ Set.Icc (-1 : ℝ) 2, f x₂ = 6) :=
sorry

end NUMINAMATH_CALUDE_f_range_on_interval_l1475_147524


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1475_147508

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1475_147508


namespace NUMINAMATH_CALUDE_boys_circle_distance_l1475_147595

theorem boys_circle_distance (n : ℕ) (r : ℝ) (h1 : n = 8) (h2 : r = 50) : 
  n * (2 * (2 * r)) = 800 := by
  sorry

end NUMINAMATH_CALUDE_boys_circle_distance_l1475_147595


namespace NUMINAMATH_CALUDE_logarithm_inequality_l1475_147557

theorem logarithm_inequality (x y z : ℝ) 
  (hx : x = Real.log π)
  (hy : y = Real.log π / Real.log (1/2))
  (hz : z = Real.exp (-1/2)) : 
  y < z ∧ z < x := by sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l1475_147557


namespace NUMINAMATH_CALUDE_bird_families_difference_l1475_147586

/-- Given the total number of bird families and the number that flew away,
    prove that the difference between those that stayed and those that flew away is 73. -/
theorem bird_families_difference (total : ℕ) (flew_away : ℕ) 
    (h1 : total = 87) (h2 : flew_away = 7) : total - flew_away - flew_away = 73 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_difference_l1475_147586


namespace NUMINAMATH_CALUDE_steven_route_count_l1475_147553

def central_park_routes : ℕ := 
  let home_to_sw_corner := (Nat.choose 5 2)
  let ne_corner_to_office := (Nat.choose 6 3)
  let park_diagonals := 2
  home_to_sw_corner * park_diagonals * ne_corner_to_office

theorem steven_route_count : central_park_routes = 400 := by
  sorry

end NUMINAMATH_CALUDE_steven_route_count_l1475_147553


namespace NUMINAMATH_CALUDE_waiter_tips_ratio_l1475_147530

theorem waiter_tips_ratio (salary tips : ℝ) 
  (h : tips / (salary + tips) = 0.6363636363636364) :
  tips / salary = 1.75 := by
sorry

end NUMINAMATH_CALUDE_waiter_tips_ratio_l1475_147530


namespace NUMINAMATH_CALUDE_henri_reads_670_words_l1475_147588

def total_time : ℝ := 8
def movie_durations : List ℝ := [3.5, 1.5, 1.25, 0.75]
def reading_speeds : List (ℝ × ℝ) := [(30, 12), (20, 8)]

def calculate_words_read (total_time : ℝ) (movie_durations : List ℝ) (reading_speeds : List (ℝ × ℝ)) : ℕ :=
  sorry

theorem henri_reads_670_words :
  calculate_words_read total_time movie_durations reading_speeds = 670 := by
  sorry

end NUMINAMATH_CALUDE_henri_reads_670_words_l1475_147588


namespace NUMINAMATH_CALUDE_integer_solutions_system_l1475_147569

theorem integer_solutions_system :
  ∀ x y z : ℤ,
  (x + y = 1 - z ∧ x^3 + y^3 = 1 - z^2) ↔
  ((∃ k : ℤ, x = k ∧ y = -k ∧ z = 1) ∨
   (x = 0 ∧ y = 1 ∧ z = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 0) ∨
   (x = 0 ∧ y = -2 ∧ z = 3) ∨
   (x = -2 ∧ y = 0 ∧ z = 3) ∨
   (x = -2 ∧ y = -3 ∧ z = 6) ∨
   (x = -3 ∧ y = -2 ∧ z = 6)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_system_l1475_147569


namespace NUMINAMATH_CALUDE_min_white_surface_area_l1475_147547

/-- Represents a cube with side length 4, composed of unit cubes -/
structure LargeCube :=
  (side_length : Nat)
  (total_cubes : Nat)
  (red_cubes : Nat)
  (white_cubes : Nat)

/-- The fraction of the surface area that is white when minimized -/
def min_white_fraction (c : LargeCube) : Rat :=
  5 / 96

/-- Theorem stating the minimum fraction of white surface area -/
theorem min_white_surface_area (c : LargeCube) 
  (h1 : c.side_length = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.red_cubes = 58)
  (h4 : c.white_cubes = 6) :
  min_white_fraction c = 5 / 96 := by
  sorry

end NUMINAMATH_CALUDE_min_white_surface_area_l1475_147547


namespace NUMINAMATH_CALUDE_initial_fee_value_l1475_147555

/-- The initial fee of the first car rental plan -/
def initial_fee : ℝ := sorry

/-- The cost per mile for the first car rental plan -/
def cost_per_mile_plan1 : ℝ := 0.40

/-- The cost per mile for the second car rental plan -/
def cost_per_mile_plan2 : ℝ := 0.60

/-- The number of miles driven for which both plans cost the same -/
def miles_driven : ℝ := 325

theorem initial_fee_value :
  initial_fee = 65 :=
by
  have h1 : initial_fee + cost_per_mile_plan1 * miles_driven = cost_per_mile_plan2 * miles_driven :=
    sorry
  sorry

end NUMINAMATH_CALUDE_initial_fee_value_l1475_147555


namespace NUMINAMATH_CALUDE_math_club_female_members_l1475_147537

theorem math_club_female_members :
  ∀ (female_members male_members : ℕ),
    female_members > 0 →
    male_members = 2 * female_members →
    female_members + male_members = 18 →
    female_members = 6 := by
  sorry

end NUMINAMATH_CALUDE_math_club_female_members_l1475_147537


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l1475_147512

theorem divisibility_implies_equality (a b : ℕ) (h : (a^2 + b^2) ∣ (a * b)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l1475_147512


namespace NUMINAMATH_CALUDE_oil_price_reduction_theorem_l1475_147523

/-- Represents the price reduction scenario for oil --/
structure OilPriceReduction where
  original_price : ℝ
  reduced_price : ℝ
  total_cost : ℝ
  additional_quantity : ℝ

/-- The reduced price is 80% of the original price --/
def price_reduction (p : OilPriceReduction) : Prop :=
  p.reduced_price = 0.8 * p.original_price

/-- The total cost remains the same before and after the reduction --/
def cost_equality (p : OilPriceReduction) : Prop :=
  ∃ x : ℝ, x * p.original_price = (x + p.additional_quantity) * p.reduced_price

/-- The theorem to be proved --/
theorem oil_price_reduction_theorem (p : OilPriceReduction) 
  (h1 : price_reduction p)
  (h2 : cost_equality p)
  (h3 : p.total_cost = 684)
  (h4 : p.additional_quantity = 4) :
  p.reduced_price = 34.2 := by
  sorry

#check oil_price_reduction_theorem

end NUMINAMATH_CALUDE_oil_price_reduction_theorem_l1475_147523


namespace NUMINAMATH_CALUDE_book_collection_ratio_l1475_147575

theorem book_collection_ratio : ∀ (L S : ℕ), 
  L + S = 3000 →  -- Total books
  S = 600 →       -- Susan's books
  L / S = 4       -- Ratio of Lidia's to Susan's books
  := by sorry

end NUMINAMATH_CALUDE_book_collection_ratio_l1475_147575


namespace NUMINAMATH_CALUDE_f_of_two_equals_two_l1475_147522

theorem f_of_two_equals_two (f : ℝ → ℝ) (h : ∀ x ≥ 0, f (1 + Real.sqrt x) = x + 1) : f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_two_l1475_147522


namespace NUMINAMATH_CALUDE_gcd_of_B_is_five_l1475_147568

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_five : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_five_l1475_147568


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1475_147587

/-- Proves that the simplified form of (9x^9+7x^8+4x^7) + (x^11+x^9+2x^7+3x^3+5x+8)
    is x^11+10x^9+7x^8+6x^7+3x^3+5x+8 -/
theorem simplify_polynomial (x : ℝ) :
  (9 * x^9 + 7 * x^8 + 4 * x^7) + (x^11 + x^9 + 2 * x^7 + 3 * x^3 + 5 * x + 8) =
  x^11 + 10 * x^9 + 7 * x^8 + 6 * x^7 + 3 * x^3 + 5 * x + 8 := by
  sorry

#check simplify_polynomial

end NUMINAMATH_CALUDE_simplify_polynomial_l1475_147587


namespace NUMINAMATH_CALUDE_even_function_properties_l1475_147544

def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

def HasMinimumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≥ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

def HasMaximumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem even_function_properties (f : ℝ → ℝ) :
  EvenFunction f →
  IncreasingOn f 3 7 →
  HasMinimumOn f 3 7 2 →
  DecreasingOn f (-7) (-3) ∧ HasMaximumOn f (-7) (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_properties_l1475_147544


namespace NUMINAMATH_CALUDE_price_reduction_equation_l1475_147581

-- Define the original price
def original_price : ℝ := 200

-- Define the final price after reductions
def final_price : ℝ := 162

-- Define the average percentage reduction
variable (x : ℝ)

-- Theorem statement
theorem price_reduction_equation :
  original_price * (1 - x)^2 = final_price :=
sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l1475_147581


namespace NUMINAMATH_CALUDE_min_distance_line_circle_l1475_147597

/-- The minimum distance between a point on the line y = 2 and a point on the circle (x - 1)² + y² = 1 is 1 -/
theorem min_distance_line_circle : 
  ∃ (d : ℝ), d = 1 ∧ 
  (∀ (p q : ℝ × ℝ), 
    (p.2 = 2) → 
    ((q.1 - 1)^2 + q.2^2 = 1) → 
    d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
sorry

end NUMINAMATH_CALUDE_min_distance_line_circle_l1475_147597


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1475_147556

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1475_147556


namespace NUMINAMATH_CALUDE_orange_harvest_theorem_l1475_147514

theorem orange_harvest_theorem (daily_harvest : ℕ) (harvest_days : ℕ) 
  (h1 : daily_harvest = 76) (h2 : harvest_days = 63) :
  daily_harvest * harvest_days = 4788 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_theorem_l1475_147514
