import Mathlib

namespace NUMINAMATH_CALUDE_train_length_calculation_l3465_346596

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 27 → 
  ∃ (length_m : ℝ), abs (length_m - 450.09) < 0.01 ∧ length_m = speed_kmh * (1000 / 3600) * time_s := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3465_346596


namespace NUMINAMATH_CALUDE_largest_convex_polygon_on_grid_l3465_346528

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  hx : x < 2004
  hy : y < 2004

/-- Represents a convex polygon on the grid -/
structure ConvexPolygon where
  vertices : List GridPoint
  is_convex : Bool  -- We assume there's a function to check convexity

/-- The main theorem stating the largest possible n-gon on the grid -/
theorem largest_convex_polygon_on_grid :
  ∃ (p : ConvexPolygon), p.vertices.length = 561 ∧
  ∀ (q : ConvexPolygon), q.vertices.length ≤ 561 :=
sorry

end NUMINAMATH_CALUDE_largest_convex_polygon_on_grid_l3465_346528


namespace NUMINAMATH_CALUDE_temperature_conversion_l3465_346546

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 105 → k = 221 := by
sorry

end NUMINAMATH_CALUDE_temperature_conversion_l3465_346546


namespace NUMINAMATH_CALUDE_lauras_blocks_l3465_346500

/-- Calculates the total number of blocks given the number of friends and blocks per friend -/
def total_blocks (num_friends : ℕ) (blocks_per_friend : ℕ) : ℕ :=
  num_friends * blocks_per_friend

/-- Proves that given 4 friends and 7 blocks per friend, the total number of blocks is 28 -/
theorem lauras_blocks : total_blocks 4 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_lauras_blocks_l3465_346500


namespace NUMINAMATH_CALUDE_roots_of_quadratic_sum_l3465_346550

theorem roots_of_quadratic_sum (α β : ℝ) : 
  (α^2 - 3*α - 4 = 0) → (β^2 - 3*β - 4 = 0) → 4*α^3 + 9*β^2 = -72 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_sum_l3465_346550


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3465_346512

theorem consecutive_even_numbers_sum (x : ℤ) : 
  (x % 2 = 0) →                   -- x is even
  (x + (x + 2) + (x + 4) = 1194) →  -- sum of three consecutive even numbers is 1194
  x = 396 :=                      -- the first even number is 396
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3465_346512


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3465_346591

theorem completing_square_equivalence (x : ℝ) :
  x^2 - 6*x - 10 = 0 ↔ (x - 3)^2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3465_346591


namespace NUMINAMATH_CALUDE_tommy_balloons_l3465_346569

theorem tommy_balloons (initial : ℝ) : 
  initial + 34.5 - 12.75 = 60.75 → initial = 39 := by sorry

end NUMINAMATH_CALUDE_tommy_balloons_l3465_346569


namespace NUMINAMATH_CALUDE_similarity_transformation_result_l3465_346560

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the similarity transformation
def similarityTransform (p : Point2D) (ratio : ℝ) : Point2D :=
  { x := p.x * ratio, y := p.y * ratio }

-- Define the theorem
theorem similarity_transformation_result :
  let A : Point2D := { x := -4, y := 2 }
  let B : Point2D := { x := -6, y := -4 }
  let ratio : ℝ := 1/2
  let A' := similarityTransform A ratio
  (A'.x = -2 ∧ A'.y = 1) ∨ (A'.x = 2 ∧ A'.y = -1) :=
sorry

end NUMINAMATH_CALUDE_similarity_transformation_result_l3465_346560


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3465_346575

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- The number of Aluminum atoms in the compound -/
def Al_count : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- The number of Hydrogen atoms in the compound -/
def H_count : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := Al_weight * Al_count + O_weight * O_count + H_weight * H_count

theorem compound_molecular_weight : molecular_weight = 78.01 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3465_346575


namespace NUMINAMATH_CALUDE_root_equation_value_l3465_346547

theorem root_equation_value (a : ℝ) : 
  a^2 + 2*a - 2 = 0 → 3*a^2 + 6*a + 2023 = 2029 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l3465_346547


namespace NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l3465_346595

theorem video_recorder_wholesale_cost :
  ∀ (wholesale_cost : ℝ),
  (∃ (retail_price employee_price : ℝ),
    retail_price = 1.20 * wholesale_cost ∧
    employee_price = 0.85 * retail_price ∧
    employee_price = 204) →
  wholesale_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l3465_346595


namespace NUMINAMATH_CALUDE_largest_area_cross_section_passes_through_center_exists_larger_radius_non_center_cross_section_l3465_346530

-- Define a convex centrally symmetric polyhedron
structure ConvexCentrallySymmetricPolyhedron where
  -- Add necessary fields and properties
  is_convex : Bool
  is_centrally_symmetric : Bool

-- Define a cross-section of the polyhedron
structure CrossSection where
  polyhedron : ConvexCentrallySymmetricPolyhedron
  plane : Plane
  passes_through_center : Bool

-- Define the area of a cross-section
def area (cs : CrossSection) : ℝ := sorry

-- Define the radius of the smallest enclosing circle of a cross-section
def smallest_enclosing_circle_radius (cs : CrossSection) : ℝ := sorry

-- Theorem 1: The cross-section with the largest area passes through the center
theorem largest_area_cross_section_passes_through_center 
  (p : ConvexCentrallySymmetricPolyhedron) :
  ∀ (cs : CrossSection), cs.polyhedron = p → 
    ∃ (center_cs : CrossSection), 
      center_cs.polyhedron = p ∧ 
      center_cs.passes_through_center = true ∧
      area center_cs ≥ area cs :=
sorry

-- Theorem 2: There exists a cross-section not passing through the center with a larger 
-- radius of the smallest enclosing circle than the cross-section passing through the center
theorem exists_larger_radius_non_center_cross_section 
  (p : ConvexCentrallySymmetricPolyhedron) :
  ∃ (cs_non_center cs_center : CrossSection), 
    cs_non_center.polyhedron = p ∧ 
    cs_center.polyhedron = p ∧
    cs_non_center.passes_through_center = false ∧
    cs_center.passes_through_center = true ∧
    smallest_enclosing_circle_radius cs_non_center > smallest_enclosing_circle_radius cs_center :=
sorry

end NUMINAMATH_CALUDE_largest_area_cross_section_passes_through_center_exists_larger_radius_non_center_cross_section_l3465_346530


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3465_346519

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a - 1)*x + 1 > 0) → a ∈ Set.Ioo (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3465_346519


namespace NUMINAMATH_CALUDE_stating_b_joined_after_five_months_l3465_346514

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Represents A's initial investment -/
def aInvestment : ℕ := 3500

/-- Represents B's investment -/
def bInvestment : ℕ := 9000

/-- Represents the profit ratio for A -/
def aProfitRatio : ℕ := 2

/-- Represents the profit ratio for B -/
def bProfitRatio : ℕ := 3

/-- 
Theorem stating that B joined 5 months after A started the business,
given the conditions of the problem.
-/
theorem b_joined_after_five_months :
  ∀ (x : ℕ),
  (aInvestment * monthsInYear) / (bInvestment * (monthsInYear - x)) = aProfitRatio / bProfitRatio →
  x = 5 := by
  sorry


end NUMINAMATH_CALUDE_stating_b_joined_after_five_months_l3465_346514


namespace NUMINAMATH_CALUDE_michael_twice_jacob_age_l3465_346507

/-- 
Given that Jacob is 13 years younger than Michael and Jacob will be 8 years old in 4 years,
this theorem proves that Michael will be twice as old as Jacob in 9 years.
-/
theorem michael_twice_jacob_age (jacob_current_age : ℕ) (michael_current_age : ℕ) :
  jacob_current_age + 4 = 8 →
  michael_current_age = jacob_current_age + 13 →
  ∃ (years : ℕ), years = 9 ∧ michael_current_age + years = 2 * (jacob_current_age + years) :=
by sorry

end NUMINAMATH_CALUDE_michael_twice_jacob_age_l3465_346507


namespace NUMINAMATH_CALUDE_planes_parallel_l3465_346598

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perp : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_perp : Line → Plane → Prop)

-- Define the objects
variable (α β γ : Plane) (a b : Line)

-- State the theorem
theorem planes_parallel 
  (h1 : parallel α γ)
  (h2 : parallel β γ)
  (h3 : line_perp a α)
  (h4 : line_perp b β)
  (h5 : line_parallel a b) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_l3465_346598


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l3465_346552

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l3465_346552


namespace NUMINAMATH_CALUDE_ratio_problem_l3465_346578

theorem ratio_problem (p q n : ℚ) (h1 : p / q = 5 / n) (h2 : 2 * p + q = 14) : n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3465_346578


namespace NUMINAMATH_CALUDE_divisible_by_three_and_nineteen_l3465_346538

theorem divisible_by_three_and_nineteen : ∃ x : ℝ, ∃ m n : ℤ, x = 3 * m ∧ x = 19 * n := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_and_nineteen_l3465_346538


namespace NUMINAMATH_CALUDE_power_of_product_l3465_346582

theorem power_of_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3465_346582


namespace NUMINAMATH_CALUDE_square_equation_solution_l3465_346597

theorem square_equation_solution (x y : ℕ) : 
  x^2 = y^2 + 7*y + 6 → x = 6 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3465_346597


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l3465_346534

theorem largest_n_for_equation : 
  (∃ (n : ℕ+), ∀ (m : ℕ+), 
    (∃ (x y z : ℕ+), m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 8) → 
    m ≤ n) ∧ 
  (∃ (x y z : ℕ+), (10 : ℕ+)^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 8) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l3465_346534


namespace NUMINAMATH_CALUDE_fraction_sum_l3465_346545

theorem fraction_sum : (1 : ℚ) / 4 + 2 / 9 + 3 / 6 = 35 / 36 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_l3465_346545


namespace NUMINAMATH_CALUDE_cookie_fraction_with_nuts_l3465_346594

theorem cookie_fraction_with_nuts 
  (total_cookies : ℕ) 
  (nuts_per_cookie : ℕ) 
  (total_nuts : ℕ) 
  (h1 : total_cookies = 60) 
  (h2 : nuts_per_cookie = 2) 
  (h3 : total_nuts = 72) : 
  (total_nuts / nuts_per_cookie : ℚ) / total_cookies = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_fraction_with_nuts_l3465_346594


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3465_346584

/-- Given a geometric sequence with positive terms and common ratio 2, 
    if the product of the 3rd and 11th terms is 16, then the 10th term is 32. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = 2 * a n) →
  a 3 * a 11 = 16 →
  a 10 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3465_346584


namespace NUMINAMATH_CALUDE_kamari_toys_l3465_346527

theorem kamari_toys (kamari_toys : ℕ) (anais_toys : ℕ) :
  anais_toys = kamari_toys + 30 →
  kamari_toys + anais_toys = 160 →
  kamari_toys = 65 := by
sorry

end NUMINAMATH_CALUDE_kamari_toys_l3465_346527


namespace NUMINAMATH_CALUDE_function_expression_l3465_346513

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_expression 
  (ω : ℝ) 
  (φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : -π/2 < φ ∧ φ < π/2) 
  (h_symmetry : f (1/3) ω φ = 0) 
  (h_amplitude : ∃ (x y : ℝ), f x ω φ - f y ω φ = 4) :
  ∀ x, f x ω φ = Real.sqrt 3 * Real.sin (π/2 * x - π/6) :=
sorry

end NUMINAMATH_CALUDE_function_expression_l3465_346513


namespace NUMINAMATH_CALUDE_inequality_proof_l3465_346558

theorem inequality_proof (a b : Real) (ha : 0 < a ∧ a < Real.pi / 2) (hb : 0 < b ∧ b < Real.pi / 2) :
  (5 / Real.cos a ^ 2) + (5 / (Real.sin a ^ 2 * Real.sin b ^ 2 * Real.cos b ^ 2)) ≥ 27 * Real.cos a + 36 * Real.sin a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3465_346558


namespace NUMINAMATH_CALUDE_tablet_count_l3465_346580

theorem tablet_count : 
  ∀ (n : ℕ) (x y : ℕ), -- n: total tablets, x: Lenovo, y: Huawei
  (x + (x + 6) + y < n / 3) →  -- Lenovo, Samsung, Huawei < 1/3 of total
  (n - 2*x - y - 6 = 3*y) →   -- Apple = 3 * Huawei
  (n - 3*x - (x + 6) - y = 59) → -- Tripling Lenovo results in 59 Apple
  (n = 94) := by
sorry

end NUMINAMATH_CALUDE_tablet_count_l3465_346580


namespace NUMINAMATH_CALUDE_roses_in_vase_l3465_346504

theorem roses_in_vase (initial_roses : ℕ) (added_roses : ℕ) (total_roses : ℕ) : 
  added_roses = 11 → total_roses = 14 → initial_roses = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l3465_346504


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3465_346571

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence {a_n} where a_2 * a_3 = 5 and a_5 * a_6 = 10, prove that a_8 * a_9 = 20. -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_23 : a 2 * a 3 = 5) 
    (h_56 : a 5 * a 6 = 10) : 
  a 8 * a 9 = 20 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3465_346571


namespace NUMINAMATH_CALUDE_sum_x_y_equals_one_third_l3465_346585

theorem sum_x_y_equals_one_third 
  (x y a : ℚ) 
  (eq1 : 17 * x + 19 * y = 6 - a) 
  (eq2 : 13 * x - 7 * y = 10 * a + 1) : 
  x + y = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_one_third_l3465_346585


namespace NUMINAMATH_CALUDE_boys_clay_maple_basketball_l3465_346548

/-- Represents a school in the sports camp -/
inductive School
| Jonas
| Clay
| Maple

/-- Represents an activity in the sports camp -/
inductive Activity
| Basketball
| Swimming

/-- Represents the gender of a student -/
inductive Gender
| Boy
| Girl

/-- Data about the sports camp -/
structure SportsData where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  jonas_students : ℕ
  clay_students : ℕ
  maple_students : ℕ
  jonas_boys : ℕ
  swimming_girls : ℕ
  clay_swimming_boys : ℕ

/-- Theorem stating the number of boys from Clay and Maple who attended basketball -/
theorem boys_clay_maple_basketball (data : SportsData)
  (h1 : data.total_students = 120)
  (h2 : data.total_boys = 70)
  (h3 : data.total_girls = 50)
  (h4 : data.jonas_students = 50)
  (h5 : data.clay_students = 40)
  (h6 : data.maple_students = 30)
  (h7 : data.jonas_boys = 28)
  (h8 : data.swimming_girls = 16)
  (h9 : data.clay_swimming_boys = 10) :
  (data.total_boys - data.jonas_boys - data.clay_swimming_boys) = 30 := by
  sorry

end NUMINAMATH_CALUDE_boys_clay_maple_basketball_l3465_346548


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l3465_346502

theorem matrix_not_invertible : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2 + 16/19, 9; 4 - 16/19, 10]
  ¬(IsUnit (Matrix.det A)) := by
sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l3465_346502


namespace NUMINAMATH_CALUDE_division_into_proportional_parts_l3465_346579

theorem division_into_proportional_parts (total : ℝ) (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := total / (a + b + c)
  let middle_part := b * x
  total = 120 ∧ a = 2 ∧ b = 2/3 ∧ c = 2/9 → middle_part = 27.6 := by
  sorry

end NUMINAMATH_CALUDE_division_into_proportional_parts_l3465_346579


namespace NUMINAMATH_CALUDE_maya_total_pages_l3465_346573

def maya_reading_problem (books_last_week : ℕ) (pages_per_book : ℕ) (reading_increase : ℕ) : ℕ :=
  let pages_last_week := books_last_week * pages_per_book
  let pages_this_week := reading_increase * pages_last_week
  pages_last_week + pages_this_week

theorem maya_total_pages :
  maya_reading_problem 5 300 2 = 4500 :=
by
  sorry

end NUMINAMATH_CALUDE_maya_total_pages_l3465_346573


namespace NUMINAMATH_CALUDE_sum_product_equals_negative_one_l3465_346537

theorem sum_product_equals_negative_one 
  (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  a*(b+c) + b*(a+c) + c*(a+b) = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_product_equals_negative_one_l3465_346537


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3465_346590

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 4)) ↔ x ≠ 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3465_346590


namespace NUMINAMATH_CALUDE_red_sweets_count_l3465_346535

theorem red_sweets_count (total : ℕ) (green : ℕ) (neither : ℕ) (red : ℕ) 
  (h1 : total = 285)
  (h2 : green = 59)
  (h3 : neither = 177)
  (h4 : total = red + green + neither) :
  red = 49 := by
  sorry

end NUMINAMATH_CALUDE_red_sweets_count_l3465_346535


namespace NUMINAMATH_CALUDE_rectangle_diagonal_length_l3465_346567

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ
  diagonal : ℝ

-- State the theorem
theorem rectangle_diagonal_length 
  (rect : Rectangle) 
  (h1 : rect.length = 16) 
  (h2 : rect.area = 192) 
  (h3 : rect.area = rect.length * rect.width) 
  (h4 : rect.diagonal ^ 2 = rect.length ^ 2 + rect.width ^ 2) : 
  rect.diagonal = 20 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_diagonal_length_l3465_346567


namespace NUMINAMATH_CALUDE_fraction_1840s_eq_four_fifteenths_l3465_346563

/-- The number of states admitted between 1840 and 1849 -/
def states_1840s : ℕ := 8

/-- The total number of states in Alice's collection -/
def total_states : ℕ := 30

/-- The fraction of states admitted between 1840 and 1849 out of the first 30 states -/
def fraction_1840s : ℚ := states_1840s / total_states

theorem fraction_1840s_eq_four_fifteenths : fraction_1840s = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_1840s_eq_four_fifteenths_l3465_346563


namespace NUMINAMATH_CALUDE_towel_folding_theorem_l3465_346532

/-- Represents the number of towels a person can fold in a given time -/
structure FoldingRate where
  towels : ℕ
  minutes : ℕ

/-- Calculates the number of towels folded in one hour given a folding rate -/
def towelsPerHour (rate : FoldingRate) : ℕ :=
  (60 / rate.minutes) * rate.towels

/-- The total number of towels folded by all three people in one hour -/
def totalTowelsPerHour (jane kyla anthony : FoldingRate) : ℕ :=
  towelsPerHour jane + towelsPerHour kyla + towelsPerHour anthony

theorem towel_folding_theorem (jane kyla anthony : FoldingRate)
  (h1 : jane = ⟨3, 5⟩)
  (h2 : kyla = ⟨5, 10⟩)
  (h3 : anthony = ⟨7, 20⟩) :
  totalTowelsPerHour jane kyla anthony = 87 := by
  sorry

#eval totalTowelsPerHour ⟨3, 5⟩ ⟨5, 10⟩ ⟨7, 20⟩

end NUMINAMATH_CALUDE_towel_folding_theorem_l3465_346532


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l3465_346577

/-- A right triangle with two 45° angles and one 90° angle, and an inscribed circle of radius 8 cm has a hypotenuse of length 16(√2 + 1) cm. -/
theorem isosceles_right_triangle_hypotenuse (r : ℝ) (h : r = 8) :
  ∃ (a : ℝ), a > 0 ∧ 
  (a * a = 2 * r * r * (2 + Real.sqrt 2)) ∧
  (a * Real.sqrt 2 = 16 * (Real.sqrt 2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l3465_346577


namespace NUMINAMATH_CALUDE_milk_drinking_l3465_346539

theorem milk_drinking (total_milk : ℚ) (drunk_fraction : ℚ) : 
  total_milk = 1/4 → drunk_fraction = 3/4 → drunk_fraction * total_milk = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_milk_drinking_l3465_346539


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l3465_346501

theorem no_solutions_for_equation : 
  ¬ ∃ (n : ℕ+), (n + 900) / 60 = ⌊Real.sqrt n⌋ := by
sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l3465_346501


namespace NUMINAMATH_CALUDE_isochronous_growth_law_l3465_346572

theorem isochronous_growth_law (k α : ℝ) (h1 : k > 0) (h2 : α > 0) :
  (∀ (x y : ℝ), y = k * x^α → (16 * x)^α = 8 * y) → α = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_isochronous_growth_law_l3465_346572


namespace NUMINAMATH_CALUDE_smallest_k_value_l3465_346551

/-- Given positive integers a, b, c, d, e, and k satisfying the conditions,
    prove that the smallest possible value for k is 522. -/
theorem smallest_k_value (a b c d e k : ℕ+) 
  (eq1 : a + 2*b + 3*c + 4*d + 5*e = k)
  (eq2 : 5*a = 4*b)
  (eq3 : 4*b = 3*c)
  (eq4 : 3*c = 2*d)
  (eq5 : 2*d = e) :
  k ≥ 522 ∧ (∃ (a' b' c' d' e' : ℕ+), 
    a' + 2*b' + 3*c' + 4*d' + 5*e' = 522 ∧
    5*a' = 4*b' ∧ 4*b' = 3*c' ∧ 3*c' = 2*d' ∧ 2*d' = e') := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_value_l3465_346551


namespace NUMINAMATH_CALUDE_lisa_caffeine_limit_l3465_346559

/-- The amount of caffeine (in mg) in one cup of coffee -/
def caffeine_per_cup : ℕ := 80

/-- The number of cups of coffee Lisa drinks -/
def cups_of_coffee : ℕ := 3

/-- The amount (in mg) by which Lisa exceeds her daily limit when drinking three cups -/
def excess_caffeine : ℕ := 40

/-- Lisa's daily caffeine limit (in mg) -/
def lisas_caffeine_limit : ℕ := 200

theorem lisa_caffeine_limit :
  lisas_caffeine_limit = cups_of_coffee * caffeine_per_cup - excess_caffeine :=
by sorry

end NUMINAMATH_CALUDE_lisa_caffeine_limit_l3465_346559


namespace NUMINAMATH_CALUDE_distance_is_sqrt_5_l3465_346549

/-- A right triangle with sides of length 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- The distance between the centers of the inscribed and circumscribed circles -/
def distance_between_centers (t : RightTriangle) : ℝ := sorry

theorem distance_is_sqrt_5 (t : RightTriangle) :
  distance_between_centers t = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_5_l3465_346549


namespace NUMINAMATH_CALUDE_greatest_common_divisor_630_90_under_35_l3465_346583

theorem greatest_common_divisor_630_90_under_35 : 
  ∃ (n : ℕ), n = 30 ∧ 
  n ∣ 630 ∧ 
  n < 35 ∧ 
  n ∣ 90 ∧ 
  ∀ (m : ℕ), m ∣ 630 → m < 35 → m ∣ 90 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_630_90_under_35_l3465_346583


namespace NUMINAMATH_CALUDE_regression_lines_common_point_l3465_346510

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The average point of a dataset -/
structure AveragePoint where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a regression line -/
def pointOnLine (l : RegressionLine) (p : AveragePoint) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem regression_lines_common_point 
  (l₁ l₂ : RegressionLine) (avg : AveragePoint) : 
  pointOnLine l₁ avg ∧ pointOnLine l₂ avg := by
  sorry

#check regression_lines_common_point

end NUMINAMATH_CALUDE_regression_lines_common_point_l3465_346510


namespace NUMINAMATH_CALUDE_twenty_one_billion_scientific_notation_l3465_346533

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_one_billion_scientific_notation :
  toScientificNotation (21000000000 : ℝ) = ScientificNotation.mk 2.1 10 sorry := by
  sorry

end NUMINAMATH_CALUDE_twenty_one_billion_scientific_notation_l3465_346533


namespace NUMINAMATH_CALUDE_eulers_formula_l3465_346520

/-- A connected planar graph -/
structure PlanarGraph where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  is_connected : Prop
  is_planar : Prop

/-- Euler's formula for connected planar graphs -/
theorem eulers_formula (G : PlanarGraph) : G.V - G.E + G.F = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l3465_346520


namespace NUMINAMATH_CALUDE_initial_apples_correct_l3465_346521

/-- The number of apples the cafeteria had initially -/
def initial_apples : ℕ := 23

/-- The number of apples used for lunch -/
def apples_used : ℕ := 20

/-- The number of apples bought -/
def apples_bought : ℕ := 6

/-- The number of apples remaining after transactions -/
def remaining_apples : ℕ := 9

/-- Theorem stating that the initial number of apples is correct -/
theorem initial_apples_correct : 
  initial_apples - apples_used + apples_bought = remaining_apples := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_correct_l3465_346521


namespace NUMINAMATH_CALUDE_third_term_is_negative_45_l3465_346557

/-- A geometric sequence with common ratio -3 and sum of first 2 terms equal to 10 -/
structure GeometricSequence where
  a₁ : ℝ
  ratio : ℝ
  sum_first_two : ℝ
  ratio_eq : ratio = -3
  sum_eq : a₁ + a₁ * ratio = sum_first_two
  sum_first_two_eq : sum_first_two = 10

/-- The third term of the geometric sequence -/
def third_term (seq : GeometricSequence) : ℝ :=
  seq.a₁ * seq.ratio^2

theorem third_term_is_negative_45 (seq : GeometricSequence) :
  third_term seq = -45 := by
  sorry

#check third_term_is_negative_45

end NUMINAMATH_CALUDE_third_term_is_negative_45_l3465_346557


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3465_346505

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3465_346505


namespace NUMINAMATH_CALUDE_jack_bake_sale_goal_l3465_346531

/-- The price of a brownie that allows Jack to reach his sales goal -/
def brownie_price : ℚ := by sorry

theorem jack_bake_sale_goal (num_brownies : ℕ) (num_lemon_squares : ℕ) (lemon_square_price : ℚ)
  (num_cookies : ℕ) (cookie_price : ℚ) (total_goal : ℚ) :
  num_brownies = 4 →
  num_lemon_squares = 5 →
  lemon_square_price = 2 →
  num_cookies = 7 →
  cookie_price = 4 →
  total_goal = 50 →
  num_brownies * brownie_price + num_lemon_squares * lemon_square_price + num_cookies * cookie_price = total_goal →
  brownie_price = 3 := by sorry

end NUMINAMATH_CALUDE_jack_bake_sale_goal_l3465_346531


namespace NUMINAMATH_CALUDE_stating_triangle_field_q_formula_l3465_346565

/-- Represents a right-angled triangle field with two people walking along its edges -/
structure TriangleField where
  /-- Length of LM -/
  r : ℝ
  /-- Length of LX -/
  p : ℝ
  /-- Length of XN -/
  q : ℝ
  /-- r is positive -/
  r_pos : r > 0
  /-- p is positive -/
  p_pos : p > 0
  /-- q is positive -/
  q_pos : q > 0
  /-- LMN is a right-angled triangle -/
  right_angle : (p + q)^2 + r^2 = (p + r - q)^2

/-- 
Theorem stating that in a TriangleField, the length q can be expressed as pr / (2p + r)
-/
theorem triangle_field_q_formula (tf : TriangleField) : tf.q = (tf.p * tf.r) / (2 * tf.p + tf.r) := by
  sorry

end NUMINAMATH_CALUDE_stating_triangle_field_q_formula_l3465_346565


namespace NUMINAMATH_CALUDE_thomas_needs_2000_more_l3465_346576

/-- Thomas's savings scenario over two years -/
def thomas_savings_scenario (first_year_allowance : ℕ) (second_year_hourly_rate : ℕ) 
  (second_year_weekly_hours : ℕ) (car_cost : ℕ) (weekly_expenses : ℕ) : Prop :=
  let weeks_per_year : ℕ := 52
  let total_weeks : ℕ := 2 * weeks_per_year
  let first_year_earnings : ℕ := first_year_allowance * weeks_per_year
  let second_year_earnings : ℕ := second_year_hourly_rate * second_year_weekly_hours * weeks_per_year
  let total_earnings : ℕ := first_year_earnings + second_year_earnings
  let total_expenses : ℕ := weekly_expenses * total_weeks
  let savings : ℕ := total_earnings - total_expenses
  car_cost - savings = 2000

/-- Theorem stating Thomas needs $2000 more to buy the car -/
theorem thomas_needs_2000_more :
  thomas_savings_scenario 50 9 30 15000 35 := by sorry

end NUMINAMATH_CALUDE_thomas_needs_2000_more_l3465_346576


namespace NUMINAMATH_CALUDE_a_value_l3465_346592

theorem a_value (a : ℝ) : 2 ∈ ({1, a, a^2 - a} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l3465_346592


namespace NUMINAMATH_CALUDE_mari_made_79_buttons_l3465_346562

/-- The number of buttons made by each person -/
structure ButtonCounts where
  kendra : ℕ
  mari : ℕ
  sue : ℕ
  jess : ℕ
  tom : ℕ

/-- The conditions of the button-making problem -/
def ButtonProblem (counts : ButtonCounts) : Prop :=
  counts.kendra = 15 ∧
  counts.mari = 5 * counts.kendra + 4 ∧
  counts.sue = 2 * counts.kendra / 3 ∧
  counts.jess = 2 * (counts.sue + counts.kendra) ∧
  counts.tom = 3 * counts.jess / 4

/-- Mari made 79 buttons -/
theorem mari_made_79_buttons (counts : ButtonCounts) 
  (h : ButtonProblem counts) : counts.mari = 79 := by
  sorry

end NUMINAMATH_CALUDE_mari_made_79_buttons_l3465_346562


namespace NUMINAMATH_CALUDE_cone_surface_area_l3465_346509

theorem cone_surface_area (r l : ℝ) (h1 : r = 3) (h2 : 2 * Real.pi * r = (2 * Real.pi / 3) * l) : 
  r * l * Real.pi + r^2 * Real.pi = 36 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3465_346509


namespace NUMINAMATH_CALUDE_johns_car_trade_in_value_l3465_346526

/-- Calculates the trade-in value of John's car based on his Uber earnings, initial car purchase price, and profit. -/
def trade_in_value (uber_earnings profit initial_car_price : ℕ) : ℕ :=
  initial_car_price - (uber_earnings - profit)

/-- Theorem stating that John's car trade-in value is $6,000 given the provided conditions. -/
theorem johns_car_trade_in_value :
  trade_in_value 30000 18000 18000 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_johns_car_trade_in_value_l3465_346526


namespace NUMINAMATH_CALUDE_train_length_train_length_approx_145m_l3465_346524

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms : ℝ := speed_kmh * (1000 / 3600)
  speed_ms * time_s

/-- Proof that a train's length is approximately 145 meters -/
theorem train_length_approx_145m (speed_kmh : ℝ) (time_s : ℝ)
  (h1 : speed_kmh = 58)
  (h2 : time_s = 9) :
  ∃ ε > 0, |train_length speed_kmh time_s - 145| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_approx_145m_l3465_346524


namespace NUMINAMATH_CALUDE_inequality_proof_l3465_346506

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2*y ∧ y = z) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3465_346506


namespace NUMINAMATH_CALUDE_groups_created_is_four_l3465_346566

/-- The number of groups created when dividing insects equally -/
def number_of_groups (boys_insects : ℕ) (girls_insects : ℕ) (insects_per_group : ℕ) : ℕ :=
  (boys_insects + girls_insects) / insects_per_group

/-- Theorem stating that the number of groups is 4 given the specific conditions -/
theorem groups_created_is_four :
  number_of_groups 200 300 125 = 4 := by
  sorry

end NUMINAMATH_CALUDE_groups_created_is_four_l3465_346566


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_theorem_l3465_346593

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (coplanar : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_theorem 
  (m n : Line) (α : Plane) 
  (h1 : subset m α) 
  (h2 : parallel n α) 
  (h3 : coplanar m n) : 
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_theorem_l3465_346593


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3465_346541

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (r₁ + r₂ = 12 ∧ |r₁ - r₂| = 10) ↔ (a = 1 ∧ b = -12 ∧ c = 11) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3465_346541


namespace NUMINAMATH_CALUDE_bedroom_renovation_time_l3465_346555

theorem bedroom_renovation_time :
  ∀ (bedroom_time : ℝ),
    bedroom_time > 0 →
    (3 * bedroom_time) +                                -- Time for 3 bedrooms
    (1.5 * bedroom_time) +                              -- Time for kitchen (50% longer than a bedroom)
    (2 * ((3 * bedroom_time) + (1.5 * bedroom_time))) = -- Time for living room (twice as everything else)
    54 →                                                -- Total renovation time
    bedroom_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_renovation_time_l3465_346555


namespace NUMINAMATH_CALUDE_total_rowing_campers_l3465_346556

def morning_rowing : ℕ := 13
def afternoon_rowing : ℕ := 21

theorem total_rowing_campers :
  morning_rowing + afternoon_rowing = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_rowing_campers_l3465_346556


namespace NUMINAMATH_CALUDE_range_of_m_l3465_346570

-- Define the conditions
def condition1 (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def condition2 (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10

def q (x m : ℝ) : Prop := condition2 x m

-- State the theorem
theorem range_of_m : 
  (∀ x m : ℝ, condition1 x → (¬(p x) → ¬(q x m)) ∧ ∃ y : ℝ, ¬(p y) ∧ q y m) →
  (∀ m : ℝ, m ≥ 9) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3465_346570


namespace NUMINAMATH_CALUDE_seats_per_row_l3465_346536

/-- Proves that given the specified conditions, the number of seats in each row is 8 -/
theorem seats_per_row (rows : ℕ) (base_cost : ℚ) (discount_rate : ℚ) (discount_group : ℕ) (total_cost : ℚ) :
  rows = 5 →
  base_cost = 30 →
  discount_rate = 1/10 →
  discount_group = 10 →
  total_cost = 1080 →
  ∃ (seats_per_row : ℕ),
    seats_per_row = 8 ∧
    total_cost = rows * (seats_per_row * base_cost - (seats_per_row / discount_group) * (discount_rate * base_cost * discount_group)) :=
by sorry

end NUMINAMATH_CALUDE_seats_per_row_l3465_346536


namespace NUMINAMATH_CALUDE_mystic_four_calculator_theorem_l3465_346554

/-- Represents the possible operations on the Mystic Four Calculator --/
inductive Operation
| replace_one
| divide_two
| subtract_three
| multiply_four

/-- Represents the state of the Mystic Four Calculator --/
structure CalculatorState where
  display : ℕ

/-- Applies an operation to the calculator state --/
def apply_operation (state : CalculatorState) (op : Operation) : CalculatorState :=
  match op with
  | Operation.replace_one => CalculatorState.mk 1
  | Operation.divide_two => 
      if state.display % 2 = 0 then CalculatorState.mk (state.display / 2)
      else state
  | Operation.subtract_three => 
      if state.display ≥ 3 then CalculatorState.mk (state.display - 3)
      else state
  | Operation.multiply_four => 
      if state.display * 4 < 10000 then CalculatorState.mk (state.display * 4)
      else state

/-- Applies a sequence of operations to the calculator state --/
def apply_sequence (initial : CalculatorState) (ops : List Operation) : CalculatorState :=
  ops.foldl apply_operation initial

theorem mystic_four_calculator_theorem :
  (¬ ∃ (ops : List Operation), (apply_sequence (CalculatorState.mk 0) ops).display = 2007) ∧
  (∃ (ops : List Operation), (apply_sequence (CalculatorState.mk 0) ops).display = 2008) :=
sorry

end NUMINAMATH_CALUDE_mystic_four_calculator_theorem_l3465_346554


namespace NUMINAMATH_CALUDE_sin_sum_of_zero_points_l3465_346515

/-- Given that x₁ and x₂ are two zero points of f(x) = 2sin(2x) + cos(2x) - m
    within the interval [0, π/2], prove that sin(x₁ + x₂) = 2√5/5 -/
theorem sin_sum_of_zero_points (x₁ x₂ m : ℝ) : 
  x₁ ∈ Set.Icc 0 (π/2) →
  x₂ ∈ Set.Icc 0 (π/2) →
  2 * Real.sin (2 * x₁) + Real.cos (2 * x₁) - m = 0 →
  2 * Real.sin (2 * x₂) + Real.cos (2 * x₂) - m = 0 →
  Real.sin (x₁ + x₂) = 2 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_sin_sum_of_zero_points_l3465_346515


namespace NUMINAMATH_CALUDE_odometer_skipping_four_l3465_346586

/-- Represents an odometer that skips the digit 4 -/
def SkippingOdometer : Type := ℕ

/-- Converts a regular number to its representation on the skipping odometer -/
def toSkippingOdometer (n : ℕ) : SkippingOdometer :=
  sorry

/-- Converts a skipping odometer reading back to the actual distance -/
def fromSkippingOdometer (s : SkippingOdometer) : ℕ :=
  sorry

/-- The theorem stating the relationship between the odometer reading and actual distance -/
theorem odometer_skipping_four (reading : SkippingOdometer) :
  reading = toSkippingOdometer 2005 →
  fromSkippingOdometer reading = 1462 :=
sorry

end NUMINAMATH_CALUDE_odometer_skipping_four_l3465_346586


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l3465_346525

/-- The curve defined by xy = 2 -/
def curve (x y : ℝ) : Prop := x * y = 2

/-- An arbitrary ellipse in the coordinate plane -/
def ellipse (x y : ℝ) : Prop := sorry

/-- The four points of intersection satisfy both the curve and ellipse equations -/
axiom intersection_points (x y : ℝ) : curve x y ∧ ellipse x y ↔ 
  (x = 3 ∧ y = 2/3) ∨ (x = -4 ∧ y = -1/2) ∨ (x = 1/4 ∧ y = 8) ∨ (x = -2/3 ∧ y = -3)

theorem fourth_intersection_point : 
  ∃ (x y : ℝ), curve x y ∧ ellipse x y ∧ x = -2/3 ∧ y = -3 :=
sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l3465_346525


namespace NUMINAMATH_CALUDE_fourth_red_ball_is_24_l3465_346568

/-- Represents a random number table --/
def RandomTable : List (List Nat) :=
  [[2, 9, 7, 63, 4, 1, 32, 8, 4, 14, 2, 4, 1],
   [8, 3, 0, 39, 8, 2, 25, 8, 8, 82, 4, 1, 0],
   [5, 5, 5, 68, 5, 2, 66, 1, 6, 68, 2, 3, 1]]

/-- Checks if a number is a valid red ball number --/
def isValidRedBall (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 33

/-- Selects valid red ball numbers from a list --/
def selectValidNumbers (numbers : List Nat) : List Nat :=
  numbers.filter isValidRedBall

/-- Flattens the random table into a single list, starting from the specified position --/
def flattenTableFrom (table : List (List Nat)) (startRow startCol : Nat) : List Nat :=
  let rowsFromStart := table.drop startRow
  let firstRow := (rowsFromStart.head!).drop startCol
  firstRow ++ (rowsFromStart.tail!).join

/-- The main theorem to prove --/
theorem fourth_red_ball_is_24 :
  let flattenedTable := flattenTableFrom RandomTable 0 8
  let validNumbers := selectValidNumbers flattenedTable
  validNumbers[3] = 24 := by sorry

end NUMINAMATH_CALUDE_fourth_red_ball_is_24_l3465_346568


namespace NUMINAMATH_CALUDE_equation_solution_l3465_346503

theorem equation_solution :
  let f (x : ℂ) := (x^2 + x + 1) / (x + 1)
  let g (x : ℂ) := x^2 + 2*x + 3
  ∀ x : ℂ, f x = g x ↔ x = -2 ∨ x = Complex.I * Real.sqrt 2 ∨ x = -Complex.I * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3465_346503


namespace NUMINAMATH_CALUDE_persimmons_in_jungkooks_house_l3465_346529

theorem persimmons_in_jungkooks_house : 
  let num_boxes : ℕ := 4
  let persimmons_per_box : ℕ := 5
  num_boxes * persimmons_per_box = 20 := by
  sorry

end NUMINAMATH_CALUDE_persimmons_in_jungkooks_house_l3465_346529


namespace NUMINAMATH_CALUDE_patrol_theorem_l3465_346588

/-- The number of streets patrolled by an officer in one hour -/
def streets_per_hour (streets : ℕ) (hours : ℕ) : ℚ := streets / hours

/-- The total number of streets patrolled by all officers in one hour -/
def total_streets_per_hour (rate_A rate_B rate_C : ℚ) : ℚ := rate_A + rate_B + rate_C

theorem patrol_theorem (a x b y c z : ℕ) 
  (h1 : streets_per_hour a x = 9/1)
  (h2 : streets_per_hour b y = 11/1)
  (h3 : streets_per_hour c z = 7/1) :
  total_streets_per_hour (streets_per_hour a x) (streets_per_hour b y) (streets_per_hour c z) = 27 := by
  sorry

end NUMINAMATH_CALUDE_patrol_theorem_l3465_346588


namespace NUMINAMATH_CALUDE_triangle_construction_exists_l3465_346518

-- Define the necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

def Line (p q : Point) : Set Point :=
  {r : Point | ∃ t : ℝ, r = Point.mk (p.x + t * (q.x - p.x)) (p.y + t * (q.y - p.y))}

def CircumscribedCircle (a b c : Point) : Set Point :=
  sorry -- Definition of circumscribed circle

def Diameter (circle : Set Point) (p q : Point) : Prop :=
  sorry -- Definition of diameter in a circle

def FirstPicturePlane : Set Point :=
  sorry -- Definition of the first picture plane

-- State the theorem
theorem triangle_construction_exists (a b d : Point) (α : ℝ) 
  (h1 : d ∈ Line a b) : 
  ∃ c : Point, 
    c ∈ FirstPicturePlane ∧ 
    d ∈ Line a b ∧ 
    Diameter (CircumscribedCircle a b c) c d := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_exists_l3465_346518


namespace NUMINAMATH_CALUDE_direct_square_variation_problem_l3465_346544

/-- A function representing direct variation with the square of x -/
def direct_square_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem direct_square_variation_problem (y : ℝ → ℝ) :
  (∃ k : ℝ, ∀ x, y x = direct_square_variation k x) →
  y 3 = 18 →
  y 6 = 72 := by sorry

end NUMINAMATH_CALUDE_direct_square_variation_problem_l3465_346544


namespace NUMINAMATH_CALUDE_combined_tax_rate_l3465_346589

theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.3)
  (h2 : mindy_rate = 0.2)
  (h3 : income_ratio = 3) :
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.225 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l3465_346589


namespace NUMINAMATH_CALUDE_seashells_sum_l3465_346587

/-- The number of seashells Mary found -/
def mary_shells : ℕ := 18

/-- The number of seashells Jessica found -/
def jessica_shells : ℕ := 41

/-- The total number of seashells found by Mary and Jessica -/
def total_shells : ℕ := mary_shells + jessica_shells

theorem seashells_sum : total_shells = 59 := by
  sorry

end NUMINAMATH_CALUDE_seashells_sum_l3465_346587


namespace NUMINAMATH_CALUDE_initial_boys_count_l3465_346522

theorem initial_boys_count (t : ℕ) : 
  t > 0 →  -- Ensure the group is non-empty
  (t / 2 : ℚ) = (t : ℚ) * (1 / 2 : ℚ) →  -- Initially 50% boys
  ((t / 2 - 4 : ℚ) / (t + 2 : ℚ) = (2 / 5 : ℚ)) →  -- After changes, 40% boys
  t / 2 = 24 := by  -- Initial number of boys is 24
sorry

end NUMINAMATH_CALUDE_initial_boys_count_l3465_346522


namespace NUMINAMATH_CALUDE_unique_n_with_special_divisors_l3465_346561

def isDivisor (d n : ℕ) : Prop := d ∣ n

def divisors (n : ℕ) : Set ℕ := {d : ℕ | isDivisor d n}

theorem unique_n_with_special_divisors :
  ∃! n : ℕ, n > 0 ∧
  ∃ (d₂ d₃ : ℕ), d₂ ∈ divisors n ∧ d₃ ∈ divisors n ∧
  1 < d₂ ∧ d₂ < d₃ ∧
  n = d₂^2 + d₃^3 ∧
  ∀ d ∈ divisors n, d = 1 ∨ d ≥ d₂ :=
by
  sorry

end NUMINAMATH_CALUDE_unique_n_with_special_divisors_l3465_346561


namespace NUMINAMATH_CALUDE_car_speed_comparison_l3465_346540

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  let x := 3 / (1 / u + 2 / v)
  let y := (2 * u + v) / 3
  x ≤ y := by
sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l3465_346540


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_four_l3465_346599

theorem sum_of_x_and_y_is_four (x y : ℝ) 
  (eq1 : 4 * x - y = 3) 
  (eq2 : x + 6 * y = 17) : 
  x + y = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_four_l3465_346599


namespace NUMINAMATH_CALUDE_equality_of_expressions_l3465_346553

theorem equality_of_expressions : 
  (-2^2 ≠ (-2)^2) ∧ 
  (2^3 ≠ 3^2) ∧ 
  (-3^3 = (-3)^3) ∧ 
  (-|-3| ≠ -(-3)) := by
  sorry

end NUMINAMATH_CALUDE_equality_of_expressions_l3465_346553


namespace NUMINAMATH_CALUDE_pool_capacity_l3465_346508

theorem pool_capacity (additional_water : ℝ) (final_percentage : ℝ) (increase_percentage : ℝ) :
  additional_water = 300 →
  final_percentage = 0.7 →
  increase_percentage = 0.3 →
  ∃ (total_capacity : ℝ),
    total_capacity = 1000 ∧
    additional_water = (final_percentage - increase_percentage) * total_capacity :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_l3465_346508


namespace NUMINAMATH_CALUDE_shirt_cost_theorem_l3465_346543

theorem shirt_cost_theorem :
  let total_shirts : ℕ := 5
  let cheap_shirts : ℕ := 3
  let expensive_shirts : ℕ := total_shirts - cheap_shirts
  let cheap_price : ℕ := 15
  let expensive_price : ℕ := 20
  
  (cheap_shirts * cheap_price + expensive_shirts * expensive_price : ℕ) = 85
  := by sorry

end NUMINAMATH_CALUDE_shirt_cost_theorem_l3465_346543


namespace NUMINAMATH_CALUDE_vacation_cost_division_l3465_346542

theorem vacation_cost_division (total_cost : ℝ) (cost_difference : ℝ) : 
  total_cost = 480 →
  (total_cost / 4 = total_cost / 6 + cost_difference) →
  cost_difference = 40 →
  6 = (total_cost / (total_cost / 4 - cost_difference)) := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l3465_346542


namespace NUMINAMATH_CALUDE_city_parking_fee_l3465_346581

def weekly_salary : ℚ := 450
def federal_tax_rate : ℚ := 1/3
def state_tax_rate : ℚ := 8/100
def health_insurance : ℚ := 50
def life_insurance : ℚ := 20
def final_paycheck : ℚ := 184

theorem city_parking_fee :
  let after_federal := weekly_salary * (1 - federal_tax_rate)
  let after_state := after_federal * (1 - state_tax_rate)
  let after_insurance := after_state - health_insurance - life_insurance
  after_insurance - final_paycheck = 22 := by sorry

end NUMINAMATH_CALUDE_city_parking_fee_l3465_346581


namespace NUMINAMATH_CALUDE_total_handshakes_at_gathering_l3465_346511

def number_of_couples : ℕ := 15

def men_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

def men_women_handshakes (n : ℕ) : ℕ := n * (n - 1)

def women_subset_handshakes : ℕ := 3

theorem total_handshakes_at_gathering :
  men_handshakes number_of_couples +
  men_women_handshakes number_of_couples +
  women_subset_handshakes = 318 := by sorry

end NUMINAMATH_CALUDE_total_handshakes_at_gathering_l3465_346511


namespace NUMINAMATH_CALUDE_luxury_to_suv_ratio_l3465_346516

/-- Represents the number of cars of each type -/
structure CarInventory where
  economy : ℕ
  luxury : ℕ
  suv : ℕ

/-- The ratio of economy cars to luxury cars is 3:2 -/
def economy_to_luxury_ratio (inventory : CarInventory) : Prop :=
  3 * inventory.luxury = 2 * inventory.economy

/-- The ratio of economy cars to SUVs is 4:1 -/
def economy_to_suv_ratio (inventory : CarInventory) : Prop :=
  4 * inventory.suv = inventory.economy

/-- The theorem stating the ratio of luxury cars to SUVs -/
theorem luxury_to_suv_ratio (inventory : CarInventory) 
  (h1 : economy_to_luxury_ratio inventory) 
  (h2 : economy_to_suv_ratio inventory) : 
  8 * inventory.suv = 3 * inventory.luxury := by
  sorry

#check luxury_to_suv_ratio

end NUMINAMATH_CALUDE_luxury_to_suv_ratio_l3465_346516


namespace NUMINAMATH_CALUDE_three_digit_append_divisibility_l3465_346517

theorem three_digit_append_divisibility :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (594000 + n) % 651 = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_three_digit_append_divisibility_l3465_346517


namespace NUMINAMATH_CALUDE_problem_proof_l3465_346574

theorem problem_proof (m n : ℕ) (h1 : m + 9 < n) 
  (h2 : (m + (m + 3) + (m + 9) + n + (n + 1) + (2*n - 1)) / 6 = n - 1) 
  (h3 : (m + 9 + n) / 2 = n - 1) : m + n = 21 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l3465_346574


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l3465_346564

theorem rectangular_prism_volume 
  (top_area : ℝ) 
  (side_area : ℝ) 
  (front_area : ℝ) 
  (h₁ : top_area = 20) 
  (h₂ : side_area = 15) 
  (h₃ : front_area = 12) : 
  ∃ (x y z : ℝ), 
    x * y = top_area ∧ 
    y * z = side_area ∧ 
    x * z = front_area ∧ 
    x * y * z = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l3465_346564


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_eq_one_l3465_346523

theorem min_sum_of_reciprocal_sum_eq_one (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : 1 / a + 1 / b = 1) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = 1 → a + b ≤ x + y ∧ a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_eq_one_l3465_346523
