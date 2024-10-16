import Mathlib

namespace NUMINAMATH_CALUDE_tickets_left_l4092_409219

def tickets_bought : ℕ := 13
def ticket_cost : ℕ := 9
def spent_on_ferris_wheel : ℕ := 81

theorem tickets_left : tickets_bought - (spent_on_ferris_wheel / ticket_cost) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_l4092_409219


namespace NUMINAMATH_CALUDE_corgi_price_calculation_l4092_409269

theorem corgi_price_calculation (x : ℝ) : 
  (2 * (x + 0.3 * x) = 2600) → x = 1000 := by
  sorry

end NUMINAMATH_CALUDE_corgi_price_calculation_l4092_409269


namespace NUMINAMATH_CALUDE_N_equals_negative_fifteen_l4092_409261

/-- A grid with arithmetic sequences in rows and columns -/
structure ArithmeticGrid where
  row_start : ℤ
  col1_second : ℤ
  col1_third : ℤ
  col2_last : ℤ

/-- The value N we're trying to determine -/
def N (grid : ArithmeticGrid) : ℤ :=
  grid.col2_last + (grid.col1_third - grid.col1_second)

/-- Theorem stating that N equals -15 for the given grid -/
theorem N_equals_negative_fifteen (grid : ArithmeticGrid) 
  (h1 : grid.row_start = 25)
  (h2 : grid.col1_second = 10)
  (h3 : grid.col1_third = 18)
  (h4 : grid.col2_last = -23) :
  N grid = -15 := by
  sorry


end NUMINAMATH_CALUDE_N_equals_negative_fifteen_l4092_409261


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l4092_409263

theorem parallel_vectors_sum_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![4, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), a = k • b) →
  ‖a + b‖ = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l4092_409263


namespace NUMINAMATH_CALUDE_not_divisible_1998_pow_minus_1_l4092_409207

theorem not_divisible_1998_pow_minus_1 (m : ℕ) : ¬(1000^m - 1 ∣ 1998^m - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_1998_pow_minus_1_l4092_409207


namespace NUMINAMATH_CALUDE_exponential_properties_l4092_409209

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem exponential_properties (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  (f (x₁ + x₂) = f x₁ * f x₂) ∧ (f (-x₁) = 1 / f x₁) :=
by sorry

end NUMINAMATH_CALUDE_exponential_properties_l4092_409209


namespace NUMINAMATH_CALUDE_min_sum_m_n_min_sum_is_three_l4092_409272

theorem min_sum_m_n (m n : ℕ+) (h : 32 * m = n ^ 5) : 
  ∀ (m' n' : ℕ+), 32 * m' = n' ^ 5 → m + n ≤ m' + n' :=
by
  sorry

theorem min_sum_is_three : 
  ∃ (m n : ℕ+), 32 * m = n ^ 5 ∧ m + n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_m_n_min_sum_is_three_l4092_409272


namespace NUMINAMATH_CALUDE_area_equality_l4092_409281

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of the part of the parallelogram below the x-axis -/
def areaBelow (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of the part of the parallelogram above the x-axis -/
def areaAbove (p : Parallelogram) : ℝ := sorry

/-- Theorem: For the given parallelogram, the area below the x-axis equals the area above -/
theorem area_equality (p : Parallelogram) 
  (h1 : p.E = ⟨-1, 2⟩) 
  (h2 : p.F = ⟨5, 2⟩) 
  (h3 : p.G = ⟨1, -2⟩) 
  (h4 : p.H = ⟨-5, -2⟩) : 
  areaBelow p = areaAbove p := by sorry

end NUMINAMATH_CALUDE_area_equality_l4092_409281


namespace NUMINAMATH_CALUDE_compare_quadratic_expressions_l4092_409258

theorem compare_quadratic_expressions (a : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) := by
  sorry

end NUMINAMATH_CALUDE_compare_quadratic_expressions_l4092_409258


namespace NUMINAMATH_CALUDE_intersection_points_parallel_lines_l4092_409246

/-- Given two parallel lines with m and n points respectively, 
    this theorem states the number of intersection points formed by 
    segments connecting these points. -/
theorem intersection_points_parallel_lines 
  (m n : ℕ) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_parallel_lines_l4092_409246


namespace NUMINAMATH_CALUDE_sqrt_calculations_l4092_409241

theorem sqrt_calculations :
  (∀ (a b c : ℝ), 
    a = 4 * Real.sqrt (1/2) ∧ 
    b = Real.sqrt 32 ∧ 
    c = Real.sqrt 8 →
    a + b - c = 4 * Real.sqrt 2) ∧
  (∀ (d e f g : ℝ),
    d = Real.sqrt 6 ∧
    e = Real.sqrt 3 ∧
    f = Real.sqrt 12 ∧
    g = Real.sqrt 3 →
    d * e + f / g = 3 * Real.sqrt 2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l4092_409241


namespace NUMINAMATH_CALUDE_average_growth_rate_l4092_409233

/-- The average monthly growth rate of CPI food prices -/
def x : ℝ := sorry

/-- The food price increase in January -/
def january_increase : ℝ := 0.028

/-- The predicted food price increase in February -/
def february_increase : ℝ := 0.02

/-- Theorem stating the relationship between the monthly increases and the average growth rate -/
theorem average_growth_rate : 
  (1 + january_increase) * (1 + february_increase) = (1 + x)^2 := by sorry

end NUMINAMATH_CALUDE_average_growth_rate_l4092_409233


namespace NUMINAMATH_CALUDE_max_triangle_area_l4092_409232

theorem max_triangle_area (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 1 ≤ b ∧ b ≤ 2) (hc : 2 ≤ c ∧ c ≤ 3)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  ∃ (area : ℝ), area ≤ 1 ∧ 
  ∀ (area' : ℝ), (∃ (a' b' c' : ℝ), 
    0 ≤ a' ∧ a' ≤ 1 ∧ 
    1 ≤ b' ∧ b' ≤ 2 ∧ 
    2 ≤ c' ∧ c' ≤ 3 ∧ 
    a' + b' > c' ∧ a' + c' > b' ∧ b' + c' > a' ∧
    area' = (a' + b' + c') * (a' + b' - c') * (a' - b' + c') * (-a' + b' + c') / 16) → 
  area' ≤ area :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l4092_409232


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l4092_409253

/-- Given three vectors OA, OB, and OC in ℝ², where points A, B, and C are collinear,
    prove that the x-coordinate of OA is 18. -/
theorem collinear_points_k_value (k : ℝ) :
  let OA : ℝ × ℝ := (k, 12)
  let OB : ℝ × ℝ := (4, 5)
  let OC : ℝ × ℝ := (10, 8)
  (∃ (t : ℝ), OC - OA = t • (OB - OA)) →
  k = 18 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l4092_409253


namespace NUMINAMATH_CALUDE_sum_reciprocals_factors_of_12_l4092_409220

def factors_of_12 : List ℕ := [1, 2, 3, 4, 6, 12]

theorem sum_reciprocals_factors_of_12 :
  (factors_of_12.map (λ n => (1 : ℚ) / n)).sum = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_factors_of_12_l4092_409220


namespace NUMINAMATH_CALUDE_share_of_B_l4092_409211

theorem share_of_B (total : ℚ) (a b c : ℚ) : 
  total = 595 → 
  a = (2/3) * b → 
  b = (1/4) * c → 
  a + b + c = total → 
  b = 105 := by
sorry

end NUMINAMATH_CALUDE_share_of_B_l4092_409211


namespace NUMINAMATH_CALUDE_work_completion_l4092_409285

/-- Represents the number of days it takes to complete the entire work -/
def total_days : ℕ := 40

/-- Represents the number of days y takes to finish the remaining work -/
def remaining_days : ℕ := 32

/-- Represents the fraction of work completed in one day -/
def daily_work_rate : ℚ := 1 / total_days

theorem work_completion (x_days : ℕ) : 
  x_days * daily_work_rate + remaining_days * daily_work_rate = 1 → 
  x_days = 8 := by sorry

end NUMINAMATH_CALUDE_work_completion_l4092_409285


namespace NUMINAMATH_CALUDE_polynomial_equality_l4092_409223

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + Real.sqrt 2) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) →
  (a₀ + a₂ + a₄) ^ 2 - (a₁ + a₃) ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l4092_409223


namespace NUMINAMATH_CALUDE_bobby_candy_remaining_l4092_409236

def candy_problem (initial_count : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) : ℕ :=
  initial_count - (first_eaten + second_eaten)

theorem bobby_candy_remaining :
  candy_problem 36 17 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_remaining_l4092_409236


namespace NUMINAMATH_CALUDE_tornado_distance_l4092_409280

/-- Given a tornado that transported objects as follows:
  * A car was transported 200 feet
  * A lawn chair was blown twice as far as the car
  * A birdhouse flew three times farther than the lawn chair
  This theorem proves that the birdhouse flew 1200 feet. -/
theorem tornado_distance (car_distance : ℕ) (lawn_chair_multiplier : ℕ) (birdhouse_multiplier : ℕ)
  (h1 : car_distance = 200)
  (h2 : lawn_chair_multiplier = 2)
  (h3 : birdhouse_multiplier = 3) :
  birdhouse_multiplier * (lawn_chair_multiplier * car_distance) = 1200 := by
  sorry

#check tornado_distance

end NUMINAMATH_CALUDE_tornado_distance_l4092_409280


namespace NUMINAMATH_CALUDE_min_dot_product_OA_OP_l4092_409237

/-- The minimum dot product of OA and OP -/
theorem min_dot_product_OA_OP : ∃ (min : ℝ),
  (∀ x y : ℝ, x > 0 → y = 9 / x → (1 * x + 1 * y) ≥ min) ∧
  (∃ x y : ℝ, x > 0 ∧ y = 9 / x ∧ 1 * x + 1 * y = min) ∧
  min = 6 := by sorry

end NUMINAMATH_CALUDE_min_dot_product_OA_OP_l4092_409237


namespace NUMINAMATH_CALUDE_simplify_fraction_expression_l4092_409206

theorem simplify_fraction_expression : 
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_expression_l4092_409206


namespace NUMINAMATH_CALUDE_intersection_when_m_3_range_of_m_when_subset_l4092_409202

-- Define sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - 2*x - x^2)}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Theorem for part 1
theorem intersection_when_m_3 :
  A ∩ B 3 = {x | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part 2
theorem range_of_m_when_subset (m : ℝ) :
  m > 0 → A ⊆ B m → m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_3_range_of_m_when_subset_l4092_409202


namespace NUMINAMATH_CALUDE_bucket_capacity_l4092_409201

theorem bucket_capacity (tank_capacity : ℕ) : ∃ (x : ℕ),
  (18 * x = tank_capacity) ∧
  (216 * 5 = tank_capacity) ∧
  (x = 60) := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_l4092_409201


namespace NUMINAMATH_CALUDE_tempo_insurance_fraction_l4092_409225

/-- The fraction of the original value that a tempo is insured for -/
def insured_fraction (premium_rate : ℚ) (premium_amount : ℚ) (original_value : ℚ) : ℚ :=
  (premium_amount / premium_rate) / original_value

/-- Theorem stating that given the specific conditions, the insured fraction is 4/5 -/
theorem tempo_insurance_fraction :
  let premium_rate : ℚ := 13 / 1000
  let premium_amount : ℚ := 910
  let original_value : ℚ := 87500
  insured_fraction premium_rate premium_amount original_value = 4 / 5 := by
sorry


end NUMINAMATH_CALUDE_tempo_insurance_fraction_l4092_409225


namespace NUMINAMATH_CALUDE_speak_both_languages_l4092_409200

theorem speak_both_languages (total : ℕ) (latin : ℕ) (french : ℕ) (neither : ℕ) 
  (h_total : total = 25)
  (h_latin : latin = 13)
  (h_french : french = 15)
  (h_neither : neither = 6) :
  latin + french - (total - neither) = 9 := by
  sorry

end NUMINAMATH_CALUDE_speak_both_languages_l4092_409200


namespace NUMINAMATH_CALUDE_smallest_value_for_y_between_1_and_2_l4092_409296

theorem smallest_value_for_y_between_1_and_2 (y : ℝ) (h1 : 1 < y) (h2 : y < 2) :
  (1 / y < y) ∧ (1 / y < y^2) ∧ (1 / y < 2*y) ∧ (1 / y < Real.sqrt y) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_for_y_between_1_and_2_l4092_409296


namespace NUMINAMATH_CALUDE_strictly_increasing_interval_l4092_409267

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

theorem strictly_increasing_interval
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (h_period : ∀ x : ℝ, f ω (x + π / ω) = f ω x) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * π - π / 3) (k * π + π / 6)) :=
sorry

end NUMINAMATH_CALUDE_strictly_increasing_interval_l4092_409267


namespace NUMINAMATH_CALUDE_shape_to_square_cut_l4092_409226

/-- Represents a shape with a given area -/
structure Shape :=
  (area : ℝ)

/-- Represents a cut of a shape into three parts -/
structure Cut (s : Shape) :=
  (part1 : Shape)
  (part2 : Shape)
  (part3 : Shape)
  (sum_area : part1.area + part2.area + part3.area = s.area)

/-- Predicate to check if three shapes can form a square -/
def CanFormSquare (p1 p2 p3 : Shape) : Prop :=
  ∃ (side : ℝ), side > 0 ∧ p1.area + p2.area + p3.area = side * side

/-- Theorem stating that any shape can be cut into three parts that form a square -/
theorem shape_to_square_cut (s : Shape) : 
  ∃ (c : Cut s), CanFormSquare c.part1 c.part2 c.part3 := by
  sorry

#check shape_to_square_cut

end NUMINAMATH_CALUDE_shape_to_square_cut_l4092_409226


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4092_409279

def M : Set Int := {-1, 1, -2, 2}
def N : Set Int := {1, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4092_409279


namespace NUMINAMATH_CALUDE_allowance_proof_l4092_409254

/-- The student's bi-weekly allowance -/
def allowance : ℝ := 233.89

/-- The amount left after all spending -/
def remaining : ℝ := 2.10

theorem allowance_proof :
  allowance * (4/9) * (1/3) * (4/11) * (1/6) = remaining := by sorry

end NUMINAMATH_CALUDE_allowance_proof_l4092_409254


namespace NUMINAMATH_CALUDE_plane_division_theorem_l4092_409247

/-- The number of regions formed by h horizontal lines and s non-horizontal lines -/
def num_regions (h s : ℕ) : ℕ := h * (s + 1) + 1 + s * (s + 1) / 2

/-- The set of valid solutions for (h, s) -/
def valid_solutions : Set (ℕ × ℕ) :=
  {(995, 1), (176, 10), (80, 21)}

theorem plane_division_theorem :
  ∀ h s : ℕ, h > 0 ∧ s > 0 →
    (num_regions h s = 1992 ↔ (h, s) ∈ valid_solutions) := by
  sorry

#check plane_division_theorem

end NUMINAMATH_CALUDE_plane_division_theorem_l4092_409247


namespace NUMINAMATH_CALUDE_modular_sum_equivalence_l4092_409213

theorem modular_sum_equivalence : ∃ (a b c : ℤ),
  (7 * a) % 80 = 1 ∧
  (13 * b) % 80 = 1 ∧
  (15 * c) % 80 = 1 ∧
  (3 * a + 9 * b + 4 * c) % 80 = 34 := by
  sorry

end NUMINAMATH_CALUDE_modular_sum_equivalence_l4092_409213


namespace NUMINAMATH_CALUDE_total_points_in_game_l4092_409276

def rounds : ℕ := 177
def points_per_round : ℕ := 46

theorem total_points_in_game : rounds * points_per_round = 8142 := by
  sorry

end NUMINAMATH_CALUDE_total_points_in_game_l4092_409276


namespace NUMINAMATH_CALUDE_range_of_a_l4092_409245

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- State the theorem
theorem range_of_a (a : ℝ) (h : a ≠ 0) :
  C a ⊆ (A ∩ (Set.univ \ B)) →
  (0 < a ∧ a ≤ 2/3) ∨ (-2/3 ≤ a ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4092_409245


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l4092_409238

theorem polynomial_root_implies_coefficients 
  (a b : ℝ) 
  (h : (2 - 3*I : ℂ) = -a/3 - (Complex.I * Real.sqrt (Complex.normSq (2 - 3*I) - a^2/9))) :
  a = -3/2 ∧ b = 65/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l4092_409238


namespace NUMINAMATH_CALUDE_mike_initial_marbles_l4092_409231

/-- The number of marbles Mike gave to Sam -/
def marbles_given : ℕ := 4

/-- The number of marbles Mike has left -/
def marbles_left : ℕ := 4

/-- The initial number of marbles Mike had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem mike_initial_marbles : initial_marbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_mike_initial_marbles_l4092_409231


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l4092_409210

theorem trigonometric_expression_equality : 
  (Real.sin (15 * π / 180) * Real.cos (20 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (115 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (5 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 
  2 * (Real.sin (35 * π / 180) - Real.sin (10 * π / 180)) / 
  (1 - Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l4092_409210


namespace NUMINAMATH_CALUDE_solution_analysis_l4092_409260

def system_of_equations (a x y z : ℝ) : Prop :=
  a^3 * x + a * y + z = a^2 ∧
  x + y + z = 1 ∧
  8 * x + 2 * y + z = 4

theorem solution_analysis :
  (∀ x y z : ℝ, system_of_equations 2 x y z ↔ x = 1/5 ∧ y = 8/5 ∧ z = -2/5) ∧
  (∃ x₁ y₁ z₁ x₂ y₂ z₂ : ℝ, x₁ ≠ x₂ ∧ system_of_equations 1 x₁ y₁ z₁ ∧ system_of_equations 1 x₂ y₂ z₂) ∧
  (¬∃ x y z : ℝ, system_of_equations (-3) x y z) :=
sorry

end NUMINAMATH_CALUDE_solution_analysis_l4092_409260


namespace NUMINAMATH_CALUDE_problem_statement_l4092_409222

theorem problem_statement :
  ∀ m n : ℤ,
  m = -(-6) →
  -n = -1 →
  m * n - 7 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4092_409222


namespace NUMINAMATH_CALUDE_ascending_order_l4092_409240

theorem ascending_order (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end NUMINAMATH_CALUDE_ascending_order_l4092_409240


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l4092_409289

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℚ) 
  (num_group1 : Nat) 
  (avg_age_group1 : ℚ) 
  (num_group2 : Nat) 
  (avg_age_group2 : ℚ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 5)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 9)
  (h6 : avg_age_group2 = 16)
  : ℚ :=
  by
    sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_fifteenth_student_age_l4092_409289


namespace NUMINAMATH_CALUDE_zero_is_monomial_l4092_409230

/-- Definition of a monomial as an algebraic expression with only one term -/
def is_monomial (expr : ℚ) : Prop := true

/-- Theorem: 0 is a monomial -/
theorem zero_is_monomial : is_monomial 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_is_monomial_l4092_409230


namespace NUMINAMATH_CALUDE_runners_in_picture_probability_l4092_409277

/-- Represents a runner on a circular track -/
structure Runner where
  lap_time : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Represents the photographer's picture -/
structure Picture where
  coverage : ℝ  -- Fraction of the track covered by the picture

/-- Calculates the probability of both runners being in the picture -/
def probability_both_in_picture (rachel : Runner) (robert : Runner) (pic : Picture) : ℝ :=
  sorry

/-- Main theorem: The probability of both runners being in the picture is 1/6 -/
theorem runners_in_picture_probability
  (rachel : Runner)
  (robert : Runner)
  (pic : Picture)
  (h_rachel_lap : rachel.lap_time = 75)
  (h_robert_lap : robert.lap_time = 90)
  (h_rachel_direction : rachel.direction = true)
  (h_robert_direction : robert.direction = false)
  (h_pic_coverage : pic.coverage = 1/3) :
  probability_both_in_picture rachel robert pic = 1/6 :=
sorry

end NUMINAMATH_CALUDE_runners_in_picture_probability_l4092_409277


namespace NUMINAMATH_CALUDE_simplify_fraction_l4092_409271

theorem simplify_fraction : (5^4 + 5^2 + 5) / (5^3 - 2 * 5) = 27 + 14 / 23 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4092_409271


namespace NUMINAMATH_CALUDE_simplify_expression_l4092_409244

theorem simplify_expression : 
  2 * Real.sqrt 12 + 3 * Real.sqrt (4/3) - Real.sqrt (16/3) - 2/3 * Real.sqrt 48 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4092_409244


namespace NUMINAMATH_CALUDE_circle_omega_area_l4092_409298

/-- Circle ω with points A and B, and tangent lines intersecting on x-axis -/
structure Circle_omega where
  /-- Point A on the circle -/
  A : ℝ × ℝ
  /-- Point B on the circle -/
  B : ℝ × ℝ
  /-- The tangent lines at A and B intersect on the x-axis -/
  tangent_intersection_on_x_axis : Prop

/-- Theorem: Area of circle ω is 120375π/9600 -/
theorem circle_omega_area (ω : Circle_omega) 
  (h1 : ω.A = (5, 15)) 
  (h2 : ω.B = (13, 9)) : 
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = 120375 * π / 9600 := by
  sorry

end NUMINAMATH_CALUDE_circle_omega_area_l4092_409298


namespace NUMINAMATH_CALUDE_discount_order_difference_l4092_409262

def original_price : ℝ := 50
def flat_discount : ℝ := 5
def percentage_discount : ℝ := 0.1

def price_flat_then_percent : ℝ := (original_price - flat_discount) * (1 - percentage_discount)
def price_percent_then_flat : ℝ := original_price * (1 - percentage_discount) - flat_discount

theorem discount_order_difference :
  price_flat_then_percent - price_percent_then_flat = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_discount_order_difference_l4092_409262


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a20_l4092_409297

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a20 (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 3 + a 5 = 105 →
  a 2 + a 4 + a 6 = 99 →
  a 20 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a20_l4092_409297


namespace NUMINAMATH_CALUDE_remainder_theorem_l4092_409239

theorem remainder_theorem (P D Q R D' Q' R' D'' Q'' R'' : ℕ) 
  (h1 : P = D * Q + R) 
  (h2 : Q = D' * Q' + R') 
  (h3 : Q' = D'' * Q'' + R'') : 
  P % (D * D' * D'') = D' * D * R'' + D * R' + R :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4092_409239


namespace NUMINAMATH_CALUDE_root_equation_solution_l4092_409208

theorem root_equation_solution (a : ℝ) (n : ℕ) : 
  a^11 + a^7 + a^3 = 1 → (a^4 + a^3 = a^n + 1 ↔ n = 15) :=
by sorry

end NUMINAMATH_CALUDE_root_equation_solution_l4092_409208


namespace NUMINAMATH_CALUDE_line_slope_l4092_409266

/-- A straight line in the xy-plane with y-intercept 10 and passing through (100, 1000) has slope 9.9 -/
theorem line_slope (f : ℝ → ℝ) (h1 : f 0 = 10) (h2 : f 100 = 1000) :
  (f 100 - f 0) / (100 - 0) = 9.9 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l4092_409266


namespace NUMINAMATH_CALUDE_root_coincidence_problem_l4092_409216

theorem root_coincidence_problem (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℝ, x^3 + a*x^2 + b*x + 16*a = (x - r)^2 * (x - s)) ∧
   (2*r + s = -a) ∧ (r^2 + 2*r*s = b) ∧ (r^2 * s = -16*a)) →
  |a*b| = 2128 :=
sorry

end NUMINAMATH_CALUDE_root_coincidence_problem_l4092_409216


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4092_409264

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1) + x + Real.sin x

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : f (4 * a) + f (b - 9) = 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → f (4 * x) + f (y - 9) = 0 → 1 / x + 1 / y ≥ 1) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ f (4 * x) + f (y - 9) = 0 ∧ 1 / x + 1 / y = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4092_409264


namespace NUMINAMATH_CALUDE_log_expression_equals_one_l4092_409242

noncomputable def a : ℝ := Real.log 5 / Real.log 6
noncomputable def b : ℝ := Real.log 3 / Real.log 10
noncomputable def c : ℝ := Real.log 2 / Real.log 15

theorem log_expression_equals_one :
  (1 - 2 * a * b * c) / (a * b + b * c + c * a) = 1 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_one_l4092_409242


namespace NUMINAMATH_CALUDE_compound_interest_rate_l4092_409275

/-- Given a principal amount, final amount, and time period, 
    calculate the compound interest rate. -/
theorem compound_interest_rate 
  (P : ℝ) (A : ℝ) (n : ℕ) 
  (h_P : P = 453.51473922902494)
  (h_A : A = 500)
  (h_n : n = 2) :
  ∃ r : ℝ, A = P * (1 + r)^n := by
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l4092_409275


namespace NUMINAMATH_CALUDE_true_proposition_l4092_409288

-- Define proposition p
def p : Prop := ∀ x : ℝ, x * (x - 1) ≠ 0 → (x ≠ 0 ∧ x ≠ 1)

-- Define proposition q
def q : Prop := ∀ a b c : ℝ, a > b → a * c > b * c

-- Theorem statement
theorem true_proposition : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_true_proposition_l4092_409288


namespace NUMINAMATH_CALUDE_g_of_3_equals_5_l4092_409252

-- Define the function g
def g (y : ℝ) : ℝ := 2 * (y - 2) + 3

-- State the theorem
theorem g_of_3_equals_5 : g 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_5_l4092_409252


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l4092_409227

/-- A die is represented as a finite type with 6 elements -/
def Die : Type := Fin 6

/-- The sample space of rolling two dice -/
def SampleSpace : Type := Die × Die

/-- Event A: the number on the first die is a multiple of 3 -/
def EventA (outcome : SampleSpace) : Prop :=
  (outcome.1.val + 1) % 3 = 0

/-- Event B: the sum of the numbers on the two dice is greater than 7 -/
def EventB (outcome : SampleSpace) : Prop :=
  outcome.1.val + outcome.2.val + 2 > 7

/-- The probability measure on the sample space -/
def P : Set SampleSpace → ℝ := sorry

/-- Theorem: The conditional probability P(B|A) is 7/12 -/
theorem conditional_probability_B_given_A :
  P {outcome | EventB outcome ∧ EventA outcome} / P {outcome | EventA outcome} = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l4092_409227


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l4092_409290

-- Define the set of people
inductive Person : Type
  | A | B | C | D

-- Define the set of balls
inductive Ball : Type
  | one | two | three | four

-- Define a distribution as a function from Person to Ball
def Distribution := Person → Ball

-- Define the event "Person A gets ball number 1"
def event_A (d : Distribution) : Prop := d Person.A = Ball.one

-- Define the event "Person B gets ball number 1"
def event_B (d : Distribution) : Prop := d Person.B = Ball.one

-- Define mutually exclusive events
def mutually_exclusive (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, ¬(e1 d ∧ e2 d)

-- Define complementary events
def complementary (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, e1 d ↔ ¬(e2 d)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutually_exclusive event_A event_B ∧ ¬(complementary event_A event_B) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l4092_409290


namespace NUMINAMATH_CALUDE_peach_fraction_proof_l4092_409224

theorem peach_fraction_proof (martine_peaches benjy_peaches gabrielle_peaches : ℕ) : 
  martine_peaches = 16 →
  gabrielle_peaches = 15 →
  martine_peaches = 2 * benjy_peaches + 6 →
  (benjy_peaches : ℚ) / gabrielle_peaches = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_peach_fraction_proof_l4092_409224


namespace NUMINAMATH_CALUDE_only_three_satisfies_l4092_409257

theorem only_three_satisfies (n : ℕ) : 
  n > 1 ∧ (∃ k : ℕ, (2^n + 1) = k * n^2) ↔ n = 3 :=
sorry

end NUMINAMATH_CALUDE_only_three_satisfies_l4092_409257


namespace NUMINAMATH_CALUDE_smallest_x_and_y_l4092_409284

theorem smallest_x_and_y (x y : ℕ+) (h : (3 : ℚ) / 4 = y / (242 + x)) : 
  (x = 2 ∧ y = 183) ∧ ∀ (x' y' : ℕ+), ((3 : ℚ) / 4 = y' / (242 + x')) → x ≤ x' :=
sorry

end NUMINAMATH_CALUDE_smallest_x_and_y_l4092_409284


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l4092_409249

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (hypotenuse : ℝ)
  (is_right : side1 = 5 ∧ side2 = 12 ∧ hypotenuse = 13)

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) :=
  (side_length : ℝ)
  (is_inscribed : True)  -- We assume the square is properly inscribed

/-- The side length of the inscribed square is 780/169 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 780 / 169 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l4092_409249


namespace NUMINAMATH_CALUDE_trig_simplification_l4092_409292

theorem trig_simplification (α : Real) (h : π < α ∧ α < 3*π/2) :
  Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) + Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = -2 / Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_trig_simplification_l4092_409292


namespace NUMINAMATH_CALUDE_gomoku_pieces_count_l4092_409228

theorem gomoku_pieces_count :
  ∀ (initial_black : ℕ) (added_black : ℕ),
    initial_black > 0 →
    initial_black ≤ 5 →
    initial_black + added_black + (initial_black + (20 - added_black)) ≤ 30 →
    7 * (initial_black + (20 - added_black)) = 8 * (initial_black + added_black) →
    initial_black + added_black = 16 := by
  sorry

end NUMINAMATH_CALUDE_gomoku_pieces_count_l4092_409228


namespace NUMINAMATH_CALUDE_fiftieth_term_is_248_l4092_409243

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem fiftieth_term_is_248 :
  arithmeticSequenceTerm 3 5 50 = 248 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_248_l4092_409243


namespace NUMINAMATH_CALUDE_chord_length_implies_a_value_l4092_409286

/-- Given a polar coordinate system with a line θ = π/3 and a circle ρ = 2a * sin(θ),
    where the chord length intercepted by the line on the circle is 2√3,
    prove that a = 2. -/
theorem chord_length_implies_a_value (a : ℝ) (h1 : a > 0) : 
  (∃ (ρ : ℝ → ℝ) (θ : ℝ), 
    (θ = π / 3) ∧ 
    (ρ θ = 2 * a * Real.sin θ) ∧
    (∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 3)) → 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_implies_a_value_l4092_409286


namespace NUMINAMATH_CALUDE_power_equation_solution_l4092_409234

theorem power_equation_solution : ∃ x : ℝ, (1/8 : ℝ) * 2^36 = 4^x ∧ x = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l4092_409234


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l4092_409268

theorem triangle_ABC_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  a * Real.cos B + b * Real.cos A = 2 * c * Real.sin C →
  b = 2 * Real.sqrt 3 →
  c = Real.sqrt 19 →
  ((C = π / 6) ∨ (C = 5 * π / 6)) ∧
  (∃ (S : ℝ), (S = (7 * Real.sqrt 3) / 2) ∨ (S = Real.sqrt 3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l4092_409268


namespace NUMINAMATH_CALUDE_symmetric_points_ab_value_l4092_409251

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry across y-axis
def symmetricAcrossYAxis (p q : Point2D) : Prop :=
  p.x = -q.x ∧ p.y = q.y

-- Theorem statement
theorem symmetric_points_ab_value :
  ∀ (a b : ℝ),
  let p : Point2D := ⟨3, -1⟩
  let q : Point2D := ⟨a, 1 - b⟩
  symmetricAcrossYAxis p q →
  a^b = 9 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_ab_value_l4092_409251


namespace NUMINAMATH_CALUDE_lisa_spoons_l4092_409235

/-- The total number of spoons Lisa has -/
def total_spoons (num_children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) 
                 (large_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

/-- Proof that Lisa has 39 spoons in total -/
theorem lisa_spoons : 
  total_spoons 4 3 2 10 15 = 39 := by
  sorry

end NUMINAMATH_CALUDE_lisa_spoons_l4092_409235


namespace NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_times_three_l4092_409256

theorem five_fourths_of_twelve_fifths_times_three (x : ℚ) : x = 12 / 5 → (5 / 4 * x) * 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_times_three_l4092_409256


namespace NUMINAMATH_CALUDE_f_theorem_l4092_409295

/-- Given a function f and a real number a, we define the properties of f -/
def f_properties (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f x = 2 * x - a / x) ∧ 
  (f 1 = 3) ∧
  (a = -1) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂)

/-- Theorem stating the properties of function f -/
theorem f_theorem (f : ℝ → ℝ) (a : ℝ) : f_properties f a := by
  sorry

#check f_theorem

end NUMINAMATH_CALUDE_f_theorem_l4092_409295


namespace NUMINAMATH_CALUDE_seashells_sum_l4092_409255

/-- The number of seashells found by Joan, Jessica, and Jeremy -/
def joan_seashells : ℕ := 6
def jessica_seashells : ℕ := 8
def jeremy_seashells : ℕ := 12

/-- The total number of seashells found by Joan, Jessica, and Jeremy -/
def total_seashells : ℕ := joan_seashells + jessica_seashells + jeremy_seashells

theorem seashells_sum : total_seashells = 26 := by
  sorry

end NUMINAMATH_CALUDE_seashells_sum_l4092_409255


namespace NUMINAMATH_CALUDE_probability_both_genders_selected_l4092_409291

/-- The probability of selecting both boys and girls when randomly choosing 3 students from 2 boys and 3 girls -/
theorem probability_both_genders_selected (total_students : ℕ) (boys : ℕ) (girls : ℕ) (selected : ℕ) : 
  total_students = boys + girls →
  boys = 2 →
  girls = 3 →
  selected = 3 →
  (1 - (Nat.choose girls selected : ℚ) / (Nat.choose total_students selected : ℚ)) = 9/10 :=
by sorry

end NUMINAMATH_CALUDE_probability_both_genders_selected_l4092_409291


namespace NUMINAMATH_CALUDE_min_gennadys_for_festival_l4092_409203

/-- Represents the number of people with a given name -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat

/-- Calculates the minimum number of Gennadys required -/
def min_gennadys (counts : NameCount) : Nat :=
  max 0 (counts.borises - 1 - counts.alexanders - counts.vasilies)

/-- The theorem stating the minimum number of Gennadys required -/
theorem min_gennadys_for_festival (counts : NameCount) 
  (h_alex : counts.alexanders = 45)
  (h_boris : counts.borises = 122)
  (h_vasily : counts.vasilies = 27) :
  min_gennadys counts = 49 := by
  sorry

#eval min_gennadys { alexanders := 45, borises := 122, vasilies := 27 }

end NUMINAMATH_CALUDE_min_gennadys_for_festival_l4092_409203


namespace NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l4092_409294

theorem product_divisible_by_sum_implies_inequality 
  (m n : ℕ) 
  (h : (m * n) % (m + n) = 0) : 
  m + n ≤ n^2 := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l4092_409294


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l4092_409270

theorem matrix_equation_proof : 
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -39/14, 51/14]
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 3]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l4092_409270


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l4092_409259

/-- Given a quadratic expression 8k^2 - 12k + 20, when rewritten in the form d(k + r)^2 + s
    where d, r, and s are constants, prove that r + s = 14.75 -/
theorem quadratic_rewrite_sum (k : ℝ) : 
  ∃ (d r s : ℝ), (∀ k, 8 * k^2 - 12 * k + 20 = d * (k + r)^2 + s) ∧ r + s = 14.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l4092_409259


namespace NUMINAMATH_CALUDE_car_average_speed_l4092_409221

/-- The average speed of a car given its distance traveled in two hours -/
theorem car_average_speed (d1 d2 : ℝ) (h1 : d1 = 98) (h2 : d2 = 60) :
  (d1 + d2) / 2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l4092_409221


namespace NUMINAMATH_CALUDE_line_equation_l4092_409293

/-- Given a line parameterized by (x, y) = (3t + 6, 5t - 10) where t is a real number,
    prove that the equation of this line in the form y = mx + b is y = (5/3)x - 20. -/
theorem line_equation (t x y : ℝ) : 
  (x = 3 * t + 6 ∧ y = 5 * t - 10) → 
  y = (5/3) * x - 20 := by sorry

end NUMINAMATH_CALUDE_line_equation_l4092_409293


namespace NUMINAMATH_CALUDE_mp3_song_count_l4092_409287

/-- Given an initial number of songs, number of deleted songs, and number of added songs,
    calculate the final number of songs on the mp3 player. -/
def final_song_count (initial : ℕ) (deleted : ℕ) (added : ℕ) : ℕ :=
  initial - deleted + added

/-- Theorem stating that given the specific numbers in the problem,
    the final song count is 64. -/
theorem mp3_song_count : final_song_count 34 14 44 = 64 := by
  sorry

end NUMINAMATH_CALUDE_mp3_song_count_l4092_409287


namespace NUMINAMATH_CALUDE_intersection_distance_implies_a_bound_l4092_409217

/-- Given a line and a circle with parameter a, if the distance between
    their intersection points is at least 2√3, then a ≤ -4/3 -/
theorem intersection_distance_implies_a_bound
  (a : ℝ)
  (line : ℝ → ℝ → Prop)
  (circle : ℝ → ℝ → Prop)
  (M N : ℝ × ℝ)
  (h_line : ∀ x y, line x y ↔ a * x - y + 3 = 0)
  (h_circle : ∀ x y, circle x y ↔ (x - 2)^2 + (y - a)^2 = 4)
  (h_intersection : line M.1 M.2 ∧ circle M.1 M.2 ∧ line N.1 N.2 ∧ circle N.1 N.2)
  (h_distance : (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12) :
  a ≤ -4/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_implies_a_bound_l4092_409217


namespace NUMINAMATH_CALUDE_student_rank_l4092_409274

theorem student_rank (total : ℕ) (left_rank : ℕ) (right_rank : ℕ) :
  total = 31 → left_rank = 11 → right_rank = total - left_rank + 1 → right_rank = 21 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_l4092_409274


namespace NUMINAMATH_CALUDE_cube_of_negative_half_x_y_squared_l4092_409265

theorem cube_of_negative_half_x_y_squared (x y : ℝ) :
  (-1/2 * x * y^2)^3 = -1/8 * x^3 * y^6 := by sorry

end NUMINAMATH_CALUDE_cube_of_negative_half_x_y_squared_l4092_409265


namespace NUMINAMATH_CALUDE_distance_in_scientific_notation_l4092_409214

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem distance_in_scientific_notation :
  let distance : ℝ := 38000
  let scientific_form := toScientificNotation distance
  scientific_form.coefficient = 3.8 ∧ scientific_form.exponent = 4 :=
sorry

end NUMINAMATH_CALUDE_distance_in_scientific_notation_l4092_409214


namespace NUMINAMATH_CALUDE_total_earnings_l4092_409299

/-- Calculate total earnings from selling candied apples and grapes -/
theorem total_earnings (num_apples : ℕ) (price_apple : ℚ) 
                       (num_grapes : ℕ) (price_grape : ℚ) : 
  num_apples = 15 → 
  price_apple = 2 → 
  num_grapes = 12 → 
  price_grape = (3/2) → 
  (num_apples : ℚ) * price_apple + (num_grapes : ℚ) * price_grape = 48 := by
sorry

end NUMINAMATH_CALUDE_total_earnings_l4092_409299


namespace NUMINAMATH_CALUDE_tangent_line_equation_l4092_409215

/-- The parabola C: y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The directrix of the parabola -/
def directrix : ℝ → Prop := λ x => x = -1

/-- The axis of symmetry of the parabola -/
def axis_of_symmetry : ℝ → Prop := λ y => y = 0

/-- Point P is the intersection of the directrix and the axis of symmetry -/
def point_P : ℝ × ℝ := (-1, 0)

/-- A tangent line to the parabola C -/
def tangent_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

theorem tangent_line_equation :
  ∃ (s : ℝ), s = 1 ∨ s = -1 ∧
  ∃ (m b : ℝ), m = s ∧ b = 1 ∧
  ∀ (x y : ℝ),
    parabola x y →
    tangent_line m b x y →
    x = point_P.1 ∧ y = point_P.2 →
    x + s * y + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l4092_409215


namespace NUMINAMATH_CALUDE_log_inequality_l4092_409248

theorem log_inequality : ∀ x : ℝ, x > 0 → x + 1/x > 2 := by sorry

end NUMINAMATH_CALUDE_log_inequality_l4092_409248


namespace NUMINAMATH_CALUDE_water_added_to_fill_tank_l4092_409273

/-- Proves that the amount of water added to fill a tank is 16 gallons, given the initial state and capacity. -/
theorem water_added_to_fill_tank (initial_fraction : ℚ) (full_capacity : ℕ) : 
  initial_fraction = 1/3 → full_capacity = 24 → (1 - initial_fraction) * full_capacity = 16 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_fill_tank_l4092_409273


namespace NUMINAMATH_CALUDE_andy_position_after_2023_turns_l4092_409218

/-- Andy's position on the coordinate plane -/
structure Position where
  x : Int
  y : Int

/-- Direction Andy is facing -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Andy's state, including position and direction -/
structure AndyState where
  pos : Position
  dir : Direction

/-- Function to update Andy's state after one move -/
def move (state : AndyState) (distance : Int) : AndyState :=
  sorry

/-- Function to turn Andy 90° right -/
def turnRight (dir : Direction) : Direction :=
  sorry

/-- Function to simulate Andy's movement for a given number of turns -/
def simulateAndy (initialState : AndyState) (turns : Nat) : Position :=
  sorry

theorem andy_position_after_2023_turns :
  let initialState : AndyState := { pos := { x := 10, y := -10 }, dir := Direction.North }
  let finalPosition := simulateAndy initialState 2023
  finalPosition = { x := 1022, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_andy_position_after_2023_turns_l4092_409218


namespace NUMINAMATH_CALUDE_circle_to_ellipse_transformation_l4092_409204

/-- A circle in the xy-plane -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- An ellipse in the x'y'-plane -/
def Ellipse (x' y' : ℝ) : Prop := x'^2/16 + y'^2/4 = 1

/-- The scaling transformation -/
def ScalingTransformation (x' y' : ℝ) : ℝ × ℝ := (4*x', y')

theorem circle_to_ellipse_transformation :
  ∀ (x' y' : ℝ), 
  let (x, y) := ScalingTransformation x' y'
  Circle x y ↔ Ellipse x' y' := by
sorry

end NUMINAMATH_CALUDE_circle_to_ellipse_transformation_l4092_409204


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l4092_409282

theorem fraction_sum_equals_decimal : (2 / 5 : ℚ) + (2 / 50 : ℚ) + (2 / 500 : ℚ) = 0.444 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l4092_409282


namespace NUMINAMATH_CALUDE_limit_of_sequence_l4092_409283

def a (n : ℕ) : ℚ := (5 * n + 1) / (10 * n - 3)

theorem limit_of_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 1/2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l4092_409283


namespace NUMINAMATH_CALUDE_field_trip_students_l4092_409212

/-- The number of people a van can hold -/
def van_capacity : ℕ := 5

/-- The number of adults going on the trip -/
def num_adults : ℕ := 3

/-- The number of vans needed for the trip -/
def num_vans : ℕ := 3

/-- The number of students going on the field trip -/
def num_students : ℕ := van_capacity * num_vans - num_adults

theorem field_trip_students : num_students = 12 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l4092_409212


namespace NUMINAMATH_CALUDE_inequality_proof_l4092_409205

open Real

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq_a : exp a = 2 * a * exp (1/2))
  (eq_b : exp b = 3 * b * exp (1/3))
  (eq_c : exp c = 5 * c * exp (1/5)) :
  b * c * exp a < c * a * exp b ∧ c * a * exp b < a * b * exp c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4092_409205


namespace NUMINAMATH_CALUDE_rainfall_sum_l4092_409229

theorem rainfall_sum (monday1 wednesday1 friday monday2 wednesday2 : ℝ)
  (h1 : monday1 = 0.17)
  (h2 : wednesday1 = 0.42)
  (h3 : friday = 0.08)
  (h4 : monday2 = 0.37)
  (h5 : wednesday2 = 0.51) :
  monday1 + wednesday1 + friday + monday2 + wednesday2 = 1.55 := by
sorry

end NUMINAMATH_CALUDE_rainfall_sum_l4092_409229


namespace NUMINAMATH_CALUDE_roundness_of_hundred_billion_l4092_409250

/-- Roundness of a positive integer is the sum of exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 100,000,000,000 is 22 -/
theorem roundness_of_hundred_billion : roundness 100000000000 = 22 := by sorry

end NUMINAMATH_CALUDE_roundness_of_hundred_billion_l4092_409250


namespace NUMINAMATH_CALUDE_sum_of_m_values_l4092_409278

-- Define the inequality system
def inequality_system (m : ℤ) : Prop :=
  ∀ x : ℝ, (x > 0) ↔ ((x - m) / 2 > 0 ∧ x - 4 < 2 * (x - 2))

-- Define the fractional equation
def fractional_equation (m : ℤ) : Prop :=
  ∃ y : ℕ, (1 - y) / (2 - y) = 3 - m / (y - 2)

-- Theorem statement
theorem sum_of_m_values :
  (∃ S : Finset ℤ, (∀ m : ℤ, m ∈ S ↔ (inequality_system m ∧ fractional_equation m)) ∧
    S.sum id = -8) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_m_values_l4092_409278
