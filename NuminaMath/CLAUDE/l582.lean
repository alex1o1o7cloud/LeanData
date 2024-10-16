import Mathlib

namespace NUMINAMATH_CALUDE_cupcake_packages_l582_58204

/-- Given the initial number of cupcakes, the number of eaten cupcakes, and the number of cupcakes per package,
    calculate the number of complete packages that can be made. -/
def calculate_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) : ℕ :=
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package

/-- Theorem: Given 20 initial cupcakes, 11 eaten cupcakes, and 3 cupcakes per package,
    the number of complete packages that can be made is 3. -/
theorem cupcake_packages : calculate_packages 20 11 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_packages_l582_58204


namespace NUMINAMATH_CALUDE_quartic_arithmetic_sequence_roots_l582_58282

/-- The coefficients of a quartic equation whose roots form an arithmetic sequence -/
theorem quartic_arithmetic_sequence_roots (C D : ℝ) :
  (∃ (a d : ℝ), {a - 3*d, a - d, a + d, a + 3*d} = 
    {x : ℝ | x^4 + 4*x^3 - 34*x^2 + C*x + D = 0}) →
  C = -76 ∧ D = 105 := by
  sorry


end NUMINAMATH_CALUDE_quartic_arithmetic_sequence_roots_l582_58282


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l582_58215

/-- Given that z = (a^2 - 1) + (a - 1)i is a purely imaginary number and a is real,
    prove that (a^2 + i) / (1 + ai) = i -/
theorem complex_fraction_equals_i (a : ℝ) (h : (a^2 - 1 : ℂ) + (a - 1)*I = (0 : ℂ) + I * ((a - 1 : ℝ) : ℂ)) :
  (a^2 + I) / (1 + a*I) = I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l582_58215


namespace NUMINAMATH_CALUDE_baron_munchausen_theorem_l582_58242

theorem baron_munchausen_theorem :
  ∀ (a b : ℕ+), ∃ (n : ℕ+), ∃ (k m : ℕ+), (a * n = k ^ 2) ∧ (b * n = m ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_baron_munchausen_theorem_l582_58242


namespace NUMINAMATH_CALUDE_binary_decimal_base5_conversion_l582_58278

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert decimal to base 5
def decimal_to_base5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_decimal_base5_conversion :
  let binary : List Bool := [true, true, false, false, true, true]
  let decimal : Nat := 51
  let base5 : List Nat := [2, 0, 1]
  binary_to_decimal binary = decimal ∧ decimal_to_base5 decimal = base5 := by
  sorry


end NUMINAMATH_CALUDE_binary_decimal_base5_conversion_l582_58278


namespace NUMINAMATH_CALUDE_point_moved_upwards_l582_58276

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point upwards by a given distance -/
def moveUpwards (p : Point) (distance : ℝ) : Point :=
  { x := p.x, y := p.y + distance }

theorem point_moved_upwards (P : Point) (Q : Point) :
  P.x = -3 ∧ P.y = 1 ∧ Q = moveUpwards P 2 → Q.x = -3 ∧ Q.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_moved_upwards_l582_58276


namespace NUMINAMATH_CALUDE_bottle_caps_remaining_l582_58202

/-- The number of bottle caps Evelyn ends with is equal to the initial number minus the lost number. -/
theorem bottle_caps_remaining (initial : ℝ) (lost : ℝ) : initial - lost = initial - lost := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_remaining_l582_58202


namespace NUMINAMATH_CALUDE_problem_solution_l582_58260

theorem problem_solution (x y : ℝ) (h1 : y = Real.log (2 * x)) (h2 : x + y = 2) :
  (Real.exp x + Real.exp y > 2 * Real.exp 1) ∧ (x * Real.log x + y * Real.log y > 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l582_58260


namespace NUMINAMATH_CALUDE_rachels_age_problem_l582_58243

/-- Rachel's age problem -/
theorem rachels_age_problem 
  (rachel_age : ℕ)
  (grandfather_age : ℕ)
  (mother_age : ℕ)
  (father_age : ℕ)
  (h1 : rachel_age = 12)
  (h2 : grandfather_age = 7 * rachel_age)
  (h3 : mother_age = grandfather_age / 2)
  (h4 : father_age + (25 - rachel_age) = 60) :
  father_age - mother_age = 5 := by
sorry

end NUMINAMATH_CALUDE_rachels_age_problem_l582_58243


namespace NUMINAMATH_CALUDE_pool_volume_calculation_l582_58263

/-- Calculates the total volume of a pool given its draining parameters -/
theorem pool_volume_calculation 
  (drain_rate : ℝ) 
  (drain_time : ℝ) 
  (initial_capacity_percentage : ℝ) : 
  drain_rate * drain_time / initial_capacity_percentage = 90000 :=
by
  sorry

#check pool_volume_calculation 60 1200 0.8

end NUMINAMATH_CALUDE_pool_volume_calculation_l582_58263


namespace NUMINAMATH_CALUDE_circle_inequality_abc_inequality_l582_58241

-- Problem I
theorem circle_inequality (x y : ℝ) (h : x^2 + y^2 = 1) : 
  -Real.sqrt 13 ≤ 2*x + 3*y ∧ 2*x + 3*y ≤ Real.sqrt 13 := by
  sorry

-- Problem II
theorem abc_inequality (a b c : ℝ) (h : a^2 + b^2 + c^2 - 2*a - 2*b - 2*c = 0) :
  2*a - b - c ≤ 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_abc_inequality_l582_58241


namespace NUMINAMATH_CALUDE_decimal_number_calculation_l582_58251

theorem decimal_number_calculation (A B : ℝ) 
  (h1 : B - A = 211.5)
  (h2 : B = 10 * A) : 
  A = 23.5 := by
sorry

end NUMINAMATH_CALUDE_decimal_number_calculation_l582_58251


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l582_58225

theorem problem_1 (a : ℝ) : (-a^3)^2 * (-a^2)^3 / a = -a^11 := by sorry

theorem problem_2 (m n : ℝ) : (m - n)^3 * (n - m)^4 * (n - m)^5 = -(n - m)^12 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l582_58225


namespace NUMINAMATH_CALUDE_waste_recovery_analysis_l582_58207

structure WasteData where
  m : ℕ
  a : ℝ
  freq1 : ℝ
  freq2 : ℝ
  freq5 : ℝ

def WasteAnalysis (data : WasteData) : Prop :=
  data.m > 0 ∧
  0.20 ≤ data.a ∧ data.a ≤ 0.30 ∧
  data.freq1 + data.freq2 + data.a + data.freq5 = 1 ∧
  data.freq1 = 0.05 ∧
  data.freq2 = 0.10 ∧
  data.freq5 = 0.15

theorem waste_recovery_analysis (data : WasteData) 
  (h : WasteAnalysis data) : 
  data.m = 20 ∧ 
  (∃ (median : ℝ), 4 ≤ median ∧ median < 5) ∧
  (∃ (avg : ℝ), avg ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_waste_recovery_analysis_l582_58207


namespace NUMINAMATH_CALUDE_range_of_a_l582_58272

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ 1 → 1 + 2^x + 4^x * a > 0) ↔ a > -1/4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l582_58272


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l582_58210

/-- A geometric sequence with common ratio 2 and all positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : GeometricSequence a) (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l582_58210


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l582_58253

theorem no_perfect_square_in_range : 
  ∀ n : ℕ, 4 ≤ n ∧ n ≤ 12 → ¬∃ m : ℕ, 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l582_58253


namespace NUMINAMATH_CALUDE_min_length_shared_side_l582_58229

/-- Given two triangles ABC and DBC sharing side BC, with AB = 8, AC = 15, DC = 10, and BD = 25,
    the minimum possible integer length of BC is 15. -/
theorem min_length_shared_side (AB AC DC BD BC : ℝ) : 
  AB = 8 → AC = 15 → DC = 10 → BD = 25 → 
  BC > AC - AB → BC > BD - DC → 
  BC ≥ 15 ∧ ∀ n : ℕ, n < 15 → ¬(BC = n) :=
by sorry

end NUMINAMATH_CALUDE_min_length_shared_side_l582_58229


namespace NUMINAMATH_CALUDE_least_three_digit_12_heavy_is_105_12_heavy_is_105_three_digit_least_three_digit_12_heavy_is_105_l582_58279

/-- A number is 12-heavy if its remainder when divided by 12 is greater than 8. -/
def is_12_heavy (n : ℕ) : Prop := n % 12 > 8

/-- The set of three-digit natural numbers. -/
def three_digit_numbers : Set ℕ := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

theorem least_three_digit_12_heavy : 
  ∀ n ∈ three_digit_numbers, is_12_heavy n → n ≥ 105 :=
by sorry

theorem is_105_12_heavy : is_12_heavy 105 :=
by sorry

theorem is_105_three_digit : 105 ∈ three_digit_numbers :=
by sorry

/-- 105 is the least three-digit 12-heavy whole number. -/
theorem least_three_digit_12_heavy_is_105 : 
  ∃ n ∈ three_digit_numbers, is_12_heavy n ∧ ∀ m ∈ three_digit_numbers, is_12_heavy m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_12_heavy_is_105_12_heavy_is_105_three_digit_least_three_digit_12_heavy_is_105_l582_58279


namespace NUMINAMATH_CALUDE_tshirt_company_profit_l582_58284

/-- Calculates the daily profit of a t-shirt company given specific conditions -/
theorem tshirt_company_profit 
  (num_employees : ℕ) 
  (shirts_per_employee : ℕ) 
  (hours_per_shift : ℕ) 
  (hourly_wage : ℚ) 
  (per_shirt_bonus : ℚ) 
  (shirt_price : ℚ) 
  (nonemployee_expenses : ℚ) 
  (h1 : num_employees = 20)
  (h2 : shirts_per_employee = 20)
  (h3 : hours_per_shift = 8)
  (h4 : hourly_wage = 12)
  (h5 : per_shirt_bonus = 5)
  (h6 : shirt_price = 35)
  (h7 : nonemployee_expenses = 1000) :
  (num_employees * shirts_per_employee * shirt_price) - 
  (num_employees * (hours_per_shift * hourly_wage + shirts_per_employee * per_shirt_bonus) + nonemployee_expenses) = 9080 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_company_profit_l582_58284


namespace NUMINAMATH_CALUDE_bee_population_theorem_bee_problem_solution_l582_58217

/-- Represents the daily change in bee population -/
def daily_change (hatch_rate : ℕ) (loss_rate : ℕ) : ℤ :=
  hatch_rate - loss_rate

/-- Calculates the final bee population after a given number of days -/
def final_population (initial : ℕ) (hatch_rate : ℕ) (loss_rate : ℕ) (days : ℕ) : ℤ :=
  initial + days * daily_change hatch_rate loss_rate

/-- Theorem stating the relationship between initial population, hatch rate, loss rate, and final population -/
theorem bee_population_theorem (initial : ℕ) (hatch_rate : ℕ) (loss_rate : ℕ) (days : ℕ) (final : ℕ) :
  final_population initial hatch_rate loss_rate days = final ↔ loss_rate = 899 :=
by
  sorry

#eval final_population 12500 3000 899 7  -- Should evaluate to 27201

/-- Main theorem proving the specific case in the problem -/
theorem bee_problem_solution :
  final_population 12500 3000 899 7 = 27201 :=
by
  sorry

end NUMINAMATH_CALUDE_bee_population_theorem_bee_problem_solution_l582_58217


namespace NUMINAMATH_CALUDE_trig_expression_equality_l582_58265

theorem trig_expression_equality : 
  (Real.sin (110 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (155 * π / 180)^2 - Real.sin (155 * π / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l582_58265


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_l582_58209

theorem rectangular_box_dimensions (X Y Z : ℝ) 
  (h1 : X * Y = 40)
  (h2 : X * Z = 72)
  (h3 : Y * Z = 90) :
  X + Y + Z = 18 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_l582_58209


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l582_58259

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 ∧ y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l582_58259


namespace NUMINAMATH_CALUDE_T_is_three_rays_l582_58232

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (4 = x + 3 ∧ y - 2 ≤ 4) ∨
               (4 = y - 2 ∧ x + 3 ≤ 4) ∨
               (x + 3 = y - 2 ∧ 4 ≤ x + 3)}

-- Define the three rays with common endpoint (1,6)
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ p.2 ≤ 6}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≤ 1 ∧ p.2 = 6}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≥ 1 ∧ p.2 = p.1 + 5}

-- Theorem statement
theorem T_is_three_rays : T = ray1 ∪ ray2 ∪ ray3 :=
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_l582_58232


namespace NUMINAMATH_CALUDE_water_in_altered_solution_l582_58269

/-- Represents the ratios of bleach, detergent, and water in a solution -/
structure SolutionRatio where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- Calculates the new ratio after altering the original ratio -/
def alter_ratio (original : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * original.bleach,
    detergent := original.detergent,
    water := 2 * original.water }

/-- Theorem: Given the conditions, the altered solution contains 150 liters of water -/
theorem water_in_altered_solution :
  let original_ratio : SolutionRatio := ⟨2, 40, 100⟩
  let altered_ratio := alter_ratio original_ratio
  let detergent_volume : ℚ := 60
  (detergent_volume * altered_ratio.water) / altered_ratio.detergent = 150 := by
  sorry

end NUMINAMATH_CALUDE_water_in_altered_solution_l582_58269


namespace NUMINAMATH_CALUDE_weight_of_four_parts_l582_58235

theorem weight_of_four_parts (total_weight : ℚ) (num_parts : ℕ) (parts_of_interest : ℕ) : 
  total_weight = 2 →
  num_parts = 9 →
  parts_of_interest = 4 →
  (parts_of_interest : ℚ) * (total_weight / num_parts) = 8/9 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_four_parts_l582_58235


namespace NUMINAMATH_CALUDE_max_sum_squares_roots_l582_58275

/-- 
For a quadratic equation x^2 + 2ax + 2a^2 + 4a + 3 = 0 with parameter a,
the sum of squares of its roots is maximized when a = -3, and the maximum sum is 18.
-/
theorem max_sum_squares_roots (a : ℝ) : 
  let f := fun x : ℝ => x^2 + 2*a*x + 2*a^2 + 4*a + 3
  let sum_squares := (- (2*a))^2 - 2*(2*a^2 + 4*a + 3)
  (∀ b : ℝ, sum_squares ≤ (-8 * (-3) - 6)) ∧ 
  sum_squares = 18 ↔ a = -3 := by sorry

end NUMINAMATH_CALUDE_max_sum_squares_roots_l582_58275


namespace NUMINAMATH_CALUDE_double_points_on_quadratic_l582_58266

/-- A double point is a point where the ordinate is twice its abscissa. -/
def is_double_point (x y : ℝ) : Prop := y = 2 * x

/-- The quadratic function y = x² + 2mx - m -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x - m

theorem double_points_on_quadratic (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_double_point x₁ y₁ ∧
    is_double_point x₂ y₂ ∧
    y₁ = quadratic_function m x₁ ∧
    y₂ = quadratic_function m x₂ ∧
    x₁ < 1 ∧ 1 < x₂ →
    m < 1 :=
sorry

end NUMINAMATH_CALUDE_double_points_on_quadratic_l582_58266


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l582_58280

theorem six_digit_divisibility (a b c : Nat) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≤ 9) (h4 : c ≤ 9) :
  ∃ k : Nat, (100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c) = 143 * k := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l582_58280


namespace NUMINAMATH_CALUDE_max_value_cos_sin_expression_l582_58233

theorem max_value_cos_sin_expression (a b c : ℝ) :
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c ≤ Real.sqrt (a^2 + b^2) + c) ∧
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c = Real.sqrt (a^2 + b^2) + c) :=
by sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_expression_l582_58233


namespace NUMINAMATH_CALUDE_inequality_solution_set_l582_58201

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 3) / (x - 1) ≤ 0

-- Define the solution set
def solution_set : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Theorem stating that the solution set is correct for the given inequality
theorem inequality_solution_set : 
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l582_58201


namespace NUMINAMATH_CALUDE_cos_two_alpha_minus_pi_sixth_l582_58240

theorem cos_two_alpha_minus_pi_sixth (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/3) 
  (h3 : Real.sin (α + π/6) = 2 * Real.sqrt 5 / 5) : 
  Real.cos (2*α - π/6) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_minus_pi_sixth_l582_58240


namespace NUMINAMATH_CALUDE_three_books_purchase_ways_l582_58257

/-- The number of ways to purchase books given the conditions -/
def purchase_ways (n : ℕ) : ℕ := 2^n - 1

/-- Theorem: There are 7 ways to purchase when there are 3 books -/
theorem three_books_purchase_ways :
  purchase_ways 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_three_books_purchase_ways_l582_58257


namespace NUMINAMATH_CALUDE_vectors_collinear_l582_58277

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- State the theorem
theorem vectors_collinear (h : ¬ ∃ (r : ℝ), e₁ = r • e₂) :
  ∃ (k : ℝ), (3 : ℝ) • e₁ - (2 : ℝ) • e₂ = k • ((4 : ℝ) • e₂ - (6 : ℝ) • e₁) :=
sorry

end NUMINAMATH_CALUDE_vectors_collinear_l582_58277


namespace NUMINAMATH_CALUDE_total_pages_in_book_l582_58250

theorem total_pages_in_book (pages_read : ℕ) (pages_left : ℕ) : 
  pages_read = 147 → pages_left = 416 → pages_read + pages_left = 563 := by
sorry

end NUMINAMATH_CALUDE_total_pages_in_book_l582_58250


namespace NUMINAMATH_CALUDE_phi_expression_l582_58288

/-- A function that is directly proportional to x -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- A function that is inversely proportional to x -/
def InverselyProportional (g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → g x = k / x

/-- The main theorem -/
theorem phi_expression (f g φ : ℝ → ℝ) 
    (h1 : DirectlyProportional f)
    (h2 : InverselyProportional g)
    (h3 : ∀ x : ℝ, φ x = f x + g x)
    (h4 : φ (1/3) = 16)
    (h5 : φ 1 = 8) :
    ∀ x : ℝ, x ≠ 0 → φ x = 3 * x + 5 / x := by
  sorry

end NUMINAMATH_CALUDE_phi_expression_l582_58288


namespace NUMINAMATH_CALUDE_sequence_bounds_l582_58293

variable (n : ℕ)

def a : ℕ → ℚ
  | 0 => 1/2
  | k + 1 => a k + (1/n : ℚ) * (a k)^2

theorem sequence_bounds (hn : n > 0) : 1 - 1/n < a n n ∧ a n n < 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_bounds_l582_58293


namespace NUMINAMATH_CALUDE_vector_triangle_inequality_l582_58221

/-- Given two vectors AB and AC in a Euclidean space, with |AB| = 3 and |AC| = 6,
    prove that the magnitude of BC is between 3 and 9 inclusive. -/
theorem vector_triangle_inequality (A B C : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖B - A‖ = 3) (h2 : ‖C - A‖ = 6) : 
  3 ≤ ‖C - B‖ ∧ ‖C - B‖ ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_vector_triangle_inequality_l582_58221


namespace NUMINAMATH_CALUDE_andrews_hotdogs_l582_58283

theorem andrews_hotdogs (total : ℕ) (cheese_pops : ℕ) (chicken_nuggets : ℕ) 
  (h1 : total = 90)
  (h2 : cheese_pops = 20)
  (h3 : chicken_nuggets = 40)
  (h4 : ∃ hotdogs : ℕ, total = hotdogs + cheese_pops + chicken_nuggets) :
  ∃ hotdogs : ℕ, hotdogs = 30 ∧ total = hotdogs + cheese_pops + chicken_nuggets :=
by
  sorry

end NUMINAMATH_CALUDE_andrews_hotdogs_l582_58283


namespace NUMINAMATH_CALUDE_not_diff_of_squares_l582_58249

theorem not_diff_of_squares (a : ℤ) : ¬ ∃ (x y : ℤ), 4 * a + 2 = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_not_diff_of_squares_l582_58249


namespace NUMINAMATH_CALUDE_square_side_length_l582_58261

/-- Given a square with perimeter 36 cm, prove that the side length is 9 cm -/
theorem square_side_length (perimeter : ℝ) (is_square : Bool) : 
  is_square ∧ perimeter = 36 → (perimeter / 4 : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l582_58261


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l582_58212

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 9

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧
    radius = 3 ∧
    ∀ (x y : ℝ), circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l582_58212


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l582_58291

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1 : ℝ) * seq.d

theorem arithmetic_sequence_product (seq : ArithmeticSequence) 
  (h1 : seq.nthTerm 8 = 20)
  (h2 : seq.d = 2) :
  seq.nthTerm 2 * seq.nthTerm 3 = 80 := by
  sorry

#check arithmetic_sequence_product

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l582_58291


namespace NUMINAMATH_CALUDE_water_amount_in_new_recipe_l582_58226

/-- Represents the ratio of ingredients in a recipe --/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio --/
def original_ratio : RecipeRatio :=
  { flour := 11, water := 8, sugar := 1 }

/-- The new recipe ratio --/
def new_ratio : RecipeRatio :=
  { flour := 22, water := 8, sugar := 4 }

/-- The amount of sugar in the new recipe --/
def new_sugar_amount : ℚ := 2

theorem water_amount_in_new_recipe :
  let water_amount := (new_ratio.water / new_ratio.sugar) * new_sugar_amount
  water_amount = 4 := by
  sorry

#check water_amount_in_new_recipe

end NUMINAMATH_CALUDE_water_amount_in_new_recipe_l582_58226


namespace NUMINAMATH_CALUDE_max_interval_increasing_sin_plus_cos_l582_58247

theorem max_interval_increasing_sin_plus_cos :
  let f : ℝ → ℝ := λ x ↦ Real.sin x + Real.cos x
  ∃ a : ℝ, a = π / 4 ∧ 
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ a → f x < f y) ∧
    (∀ b : ℝ, b > a → ∃ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ b ∧ f x ≥ f y) :=
by sorry

end NUMINAMATH_CALUDE_max_interval_increasing_sin_plus_cos_l582_58247


namespace NUMINAMATH_CALUDE_jack_evening_emails_l582_58219

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 3

/-- The total number of emails Jack received in the morning and evening combined -/
def morning_evening_total : ℕ := 11

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := morning_evening_total - morning_emails

theorem jack_evening_emails : evening_emails = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_evening_emails_l582_58219


namespace NUMINAMATH_CALUDE_water_depth_calculation_l582_58231

def water_depth (dean_height ron_height : ℝ) : ℝ :=
  2 * dean_height

theorem water_depth_calculation (ron_height : ℝ) (h1 : ron_height = 14) :
  water_depth (ron_height - 8) ron_height = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_calculation_l582_58231


namespace NUMINAMATH_CALUDE_monotonic_range_of_a_l582_58220

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 2

theorem monotonic_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, Monotone (f a)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_range_of_a_l582_58220


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l582_58292

/-- The equation of a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to find the symmetric point of a point with respect to a line -/
def symmetricPoint (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

/-- Function to find the symmetric circle -/
def symmetricCircle (c : Circle) (l : Line) : Circle := sorry

theorem symmetric_circle_equation (c : Circle) (l : Line) : 
  c.center = (1, 0) ∧ c.radius = Real.sqrt 2 ∧ 
  l = { a := 2, b := -1, c := 3 } →
  let c' := symmetricCircle c l
  c'.center = (-3, 2) ∧ c'.radius = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l582_58292


namespace NUMINAMATH_CALUDE_probability_red_or_white_marble_l582_58287

theorem probability_red_or_white_marble (total : ℕ) (blue : ℕ) (red : ℕ) 
  (h1 : total = 60) 
  (h2 : blue = 5) 
  (h3 : red = 9) :
  (red : ℚ) / total + ((total - blue - red) : ℚ) / total = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_marble_l582_58287


namespace NUMINAMATH_CALUDE_cos_135_degrees_l582_58230

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l582_58230


namespace NUMINAMATH_CALUDE_three_million_twenty_one_thousand_scientific_notation_l582_58248

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem three_million_twenty_one_thousand_scientific_notation :
  toScientificNotation 3021000 = ScientificNotation.mk 3.021 6 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_three_million_twenty_one_thousand_scientific_notation_l582_58248


namespace NUMINAMATH_CALUDE_hyperbola_quadrilateral_area_ratio_max_l582_58274

theorem hyperbola_quadrilateral_area_ratio_max
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (S₁ : ℝ) (hS₁ : S₁ = 2 * a * b)
  (S₂ : ℝ) (hS₂ : S₂ = 2 * (a^2 + b^2)) :
  (S₁ / S₂) ≤ (1 / 2) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_quadrilateral_area_ratio_max_l582_58274


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l582_58224

theorem square_plus_reciprocal_square (n : ℝ) (h : n + 1/n = 10) : n^2 + 1/n^2 + 4 = 102 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l582_58224


namespace NUMINAMATH_CALUDE_test_examination_l582_58254

/-- The number of boys examined in a test --/
def num_boys : ℕ := 50

/-- The number of girls examined in the test --/
def num_girls : ℕ := 100

/-- The percentage of boys who pass the test --/
def boys_pass_rate : ℚ := 1/2

/-- The percentage of girls who pass the test --/
def girls_pass_rate : ℚ := 2/5

/-- The percentage of total students who fail the test --/
def total_fail_rate : ℚ := 5667/10000

theorem test_examination :
  num_boys = 50 ∧
  (num_boys * (1 - boys_pass_rate) + num_girls * (1 - girls_pass_rate)) / (num_boys + num_girls) = total_fail_rate :=
by sorry

end NUMINAMATH_CALUDE_test_examination_l582_58254


namespace NUMINAMATH_CALUDE_sum_of_x_equals_two_l582_58213

/-- The function f(x) = |x+1| + |x-3| -/
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

/-- Theorem stating that if there exist two distinct real numbers x₁ and x₂ 
    such that f(x₁) = f(x₂) = 101, then their sum is 2 -/
theorem sum_of_x_equals_two (x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ = 101) (h₃ : f x₂ = 101) :
  x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_equals_two_l582_58213


namespace NUMINAMATH_CALUDE_rolls_combination_count_l582_58245

theorem rolls_combination_count :
  let total_rolls : ℕ := 8
  let min_per_kind : ℕ := 2
  let num_kinds : ℕ := 3
  let remaining_rolls : ℕ := total_rolls - (min_per_kind * num_kinds)
  Nat.choose (remaining_rolls + num_kinds - 1) (num_kinds - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_rolls_combination_count_l582_58245


namespace NUMINAMATH_CALUDE_no_nonzero_solution_for_equation_l582_58281

theorem no_nonzero_solution_for_equation (a b c d : ℤ) :
  a^2 + b^2 = 3*(c^2 + d^2) → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_solution_for_equation_l582_58281


namespace NUMINAMATH_CALUDE_missing_village_population_l582_58289

def village_population_problem (total_villages : Nat) 
                               (average_population : Nat) 
                               (known_populations : List Nat) : Prop :=
  total_villages = 7 ∧
  average_population = 1000 ∧
  known_populations = [803, 900, 1023, 945, 980, 1249] ∧
  known_populations.length = 6 ∧
  (List.sum known_populations + 1100) / total_villages = average_population

theorem missing_village_population :
  ∀ (total_villages : Nat) (average_population : Nat) (known_populations : List Nat),
  village_population_problem total_villages average_population known_populations →
  1100 = total_villages * average_population - List.sum known_populations :=
by
  sorry

end NUMINAMATH_CALUDE_missing_village_population_l582_58289


namespace NUMINAMATH_CALUDE_expression_simplification_l582_58294

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2+x) + 4*(2-x) - 5*(2+3*x) = -20*x - 8 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l582_58294


namespace NUMINAMATH_CALUDE_snack_store_spending_l582_58298

/-- The amount Ben spent at the snack store -/
def ben_spent : ℝ := 60

/-- The amount David spent at the snack store -/
def david_spent : ℝ := 45

/-- For every dollar Ben spent, David spent 25 cents less -/
axiom david_spent_less : david_spent = ben_spent - 0.25 * ben_spent

/-- Ben paid $15 more than David -/
axiom ben_paid_more : ben_spent = david_spent + 15

/-- The total amount spent by Ben and David -/
def total_spent : ℝ := ben_spent + david_spent

theorem snack_store_spending : total_spent = 105 := by
  sorry

end NUMINAMATH_CALUDE_snack_store_spending_l582_58298


namespace NUMINAMATH_CALUDE_middle_book_pages_l582_58211

def longest_book : ℕ := 396

def shortest_book : ℕ := longest_book / 4

def middle_book : ℕ := 3 * shortest_book

theorem middle_book_pages : middle_book = 297 := by
  sorry

end NUMINAMATH_CALUDE_middle_book_pages_l582_58211


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l582_58267

open Set

def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}
def N : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l582_58267


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l582_58271

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) * (2 - Complex.I)
  (z.re = 0) → m = -2 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l582_58271


namespace NUMINAMATH_CALUDE_inflection_point_and_concavity_l582_58228

-- Define the function f(x) = x³ - 3x² + 5
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- Define the first derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Define the second derivative of f
def f'' (x : ℝ) : ℝ := 6*x - 6

theorem inflection_point_and_concavity :
  -- The inflection point occurs at x = 1
  (∃ (ε : ℝ), ε > 0 ∧ 
    (∀ x ∈ Set.Ioo (1 - ε) 1, f'' x < 0) ∧
    (∀ x ∈ Set.Ioo 1 (1 + ε), f'' x > 0)) ∧
  -- f(1) = 3
  f 1 = 3 ∧
  -- The function is concave down for x < 1
  (∀ x < 1, f'' x < 0) ∧
  -- The function is concave up for x > 1
  (∀ x > 1, f'' x > 0) :=
by sorry

end NUMINAMATH_CALUDE_inflection_point_and_concavity_l582_58228


namespace NUMINAMATH_CALUDE_max_sum_reciprocal_constraint_l582_58236

theorem max_sum_reciprocal_constraint (a b c : ℕ+) : 
  (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c →
  Nat.gcd a.val (Nat.gcd b.val c.val) = 1 →
  a.val + b.val ≤ 2011 →
  (∀ a' b' c' : ℕ+, (1 : ℚ) / a' + (1 : ℚ) / b' = (1 : ℚ) / c' →
    Nat.gcd a'.val (Nat.gcd b'.val c'.val) = 1 →
    a'.val + b'.val ≤ 2011 →
    a'.val + b'.val ≤ a.val + b.val) →
  a.val + b.val = 1936 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_reciprocal_constraint_l582_58236


namespace NUMINAMATH_CALUDE_proposition_equivalence_l582_58258

theorem proposition_equivalence (p q : Prop) : (p ∨ q) ↔ ¬(¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l582_58258


namespace NUMINAMATH_CALUDE_decimal_point_shift_l582_58255

theorem decimal_point_shift (x : ℝ) :
  (x * 10 = 760.8) → (x = 76.08) :=
by sorry

end NUMINAMATH_CALUDE_decimal_point_shift_l582_58255


namespace NUMINAMATH_CALUDE_division_problem_l582_58252

theorem division_problem : ∃ (q : ℕ), 
  220070 = (555 + 445) * q + 70 ∧ q = 2 * (555 - 445) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l582_58252


namespace NUMINAMATH_CALUDE_quiz_correct_answers_l582_58234

theorem quiz_correct_answers (total : ℕ) (difference : ℕ) (sang_hyeon : ℕ) : 
  total = sang_hyeon + (sang_hyeon + difference) → 
  difference = 5 → 
  total = 43 → 
  sang_hyeon = 19 := by sorry

end NUMINAMATH_CALUDE_quiz_correct_answers_l582_58234


namespace NUMINAMATH_CALUDE_derivative_of_f_l582_58286

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^5 - 3 * x^2 - 4

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 10 * x^4 - 6 * x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l582_58286


namespace NUMINAMATH_CALUDE_cube_volume_and_diagonal_l582_58208

/-- Given a cube with surface area 150 square centimeters, prove its volume and space diagonal. -/
theorem cube_volume_and_diagonal (s : ℝ) (h : 6 * s^2 = 150) : 
  s^3 = 125 ∧ s * Real.sqrt 3 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_and_diagonal_l582_58208


namespace NUMINAMATH_CALUDE_square_roots_equality_l582_58239

theorem square_roots_equality (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2*a + 1)^2 = x ∧ (a + 5)^2 = x) → a = 4 := by
sorry

end NUMINAMATH_CALUDE_square_roots_equality_l582_58239


namespace NUMINAMATH_CALUDE_pizza_toppings_l582_58227

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (h_total : total_slices = 16)
  (h_pepperoni : pepperoni_slices = 8)
  (h_mushroom : mushroom_slices = 14)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l582_58227


namespace NUMINAMATH_CALUDE_two_guests_mixed_probability_l582_58297

/-- The number of guests -/
def num_guests : ℕ := 3

/-- The number of pastry types -/
def num_pastry_types : ℕ := 3

/-- The total number of pastries -/
def total_pastries : ℕ := num_guests * num_pastry_types

/-- The number of pastries each guest receives -/
def pastries_per_guest : ℕ := num_pastry_types

/-- The probability of exactly two guests receiving one of each type of pastry -/
def probability_two_guests_mixed : ℚ := 27 / 280

theorem two_guests_mixed_probability :
  probability_two_guests_mixed = 27 / 280 := by
  sorry

end NUMINAMATH_CALUDE_two_guests_mixed_probability_l582_58297


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_l582_58206

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  focusA : Point
  focusB : Point
  passingPoint : Point

/-- Check if a point satisfies the ellipse equation -/
def satisfiesEllipseEquation (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.h)^2 / e.a^2 + (p.y - e.k)^2 / e.b^2 = 1

theorem ellipse_a_plus_k (e : Ellipse) :
  e.focusA = ⟨0, 1⟩ →
  e.focusB = ⟨0, -3⟩ →
  e.passingPoint = ⟨5, 0⟩ →
  e.a > 0 →
  e.b > 0 →
  satisfiesEllipseEquation e e.passingPoint →
  e.a + e.k = (Real.sqrt 26 + Real.sqrt 34 - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_l582_58206


namespace NUMINAMATH_CALUDE_max_fraction_sum_l582_58237

theorem max_fraction_sum (a b c d : ℕ) 
  (h1 : a + c = 1000) 
  (h2 : b + d = 500) : 
  (∀ a' b' c' d' : ℕ, 
    a' + c' = 1000 → 
    b' + d' = 500 → 
    (a' : ℚ) / b' + (c' : ℚ) / d' ≤ 1 / 499 + 999) := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l582_58237


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_center_l582_58203

-- Define the fixed circle
def fixed_circle (x y : ℝ) : Prop := (x - 5)^2 + (y + 7)^2 = 16

-- Define the moving circle with radius 1
def moving_circle (center_x center_y : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = 1

-- Define the tangency condition
def is_tangent (center_x center_y : ℝ) : Prop :=
  ∃ x y : ℝ, fixed_circle x y ∧ moving_circle center_x center_y x y

-- Theorem statement
theorem trajectory_of_moving_circle_center :
  ∀ center_x center_y : ℝ,
    is_tangent center_x center_y →
    ((center_x - 5)^2 + (center_y + 7)^2 = 25 ∨
     (center_x - 5)^2 + (center_y + 7)^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_center_l582_58203


namespace NUMINAMATH_CALUDE_survey_b_count_l582_58264

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => (firstSample + i * (populationSize / sampleSize)) % populationSize + 1)

/-- Count elements in a list that fall within a given range -/
def countInRange (list : List ℕ) (lower upper : ℕ) : ℕ :=
  list.filter (fun x => lower ≤ x ∧ x ≤ upper) |>.length

theorem survey_b_count :
  let populationSize := 480
  let sampleSize := 16
  let firstSample := 8
  let surveyBLower := 161
  let surveyBUpper := 320
  let sampledNumbers := systematicSample populationSize sampleSize firstSample
  countInRange sampledNumbers surveyBLower surveyBUpper = 5 := by
  sorry


end NUMINAMATH_CALUDE_survey_b_count_l582_58264


namespace NUMINAMATH_CALUDE_break_even_point_l582_58296

/-- The break-even point for a plastic handle molding company -/
theorem break_even_point
  (cost_per_handle : ℝ)
  (fixed_cost : ℝ)
  (selling_price : ℝ)
  (h1 : cost_per_handle = 0.60)
  (h2 : fixed_cost = 7640)
  (h3 : selling_price = 4.60) :
  ∃ x : ℕ, x = 1910 ∧ selling_price * x = fixed_cost + cost_per_handle * x :=
by sorry

end NUMINAMATH_CALUDE_break_even_point_l582_58296


namespace NUMINAMATH_CALUDE_log_inequality_implies_base_inequality_l582_58268

theorem log_inequality_implies_base_inequality (a b : ℝ) 
  (h1 : (Real.log 3 / Real.log a) > (Real.log 3 / Real.log b)) 
  (h2 : (Real.log 3 / Real.log b) > 0) : b > a ∧ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_implies_base_inequality_l582_58268


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l582_58238

theorem inequality_holds_iff (n k : ℕ) : 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → 
    a^k * b^k * (a^2 + b^2)^n ≤ (a + b)^(2*k + 2*n) / 2^(2*k + n)) ↔ 
  k ≥ n := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l582_58238


namespace NUMINAMATH_CALUDE_problem_statement_l582_58216

theorem problem_statement (a : ℤ) 
  (h1 : 0 ≤ a) 
  (h2 : a ≤ 13) 
  (h3 : (51 ^ 2016 - a) % 13 = 0) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l582_58216


namespace NUMINAMATH_CALUDE_min_value_of_f_l582_58205

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + 1/b) / (2*x^2 + 2*x + 1)

theorem min_value_of_f (b : ℝ) (h : b > 0) :
  ∃ c : ℝ, c = -4 ∧ ∀ x : ℝ, f b x ≥ c :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l582_58205


namespace NUMINAMATH_CALUDE_p_and_q_true_l582_58222

theorem p_and_q_true : 
  (∃ x₀ : ℝ, Real.tan x₀ = Real.sqrt 3) ∧ 
  (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_true_l582_58222


namespace NUMINAMATH_CALUDE_gcd_78_36_l582_58285

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_36_l582_58285


namespace NUMINAMATH_CALUDE_cookie_problem_l582_58256

theorem cookie_problem (glenn kenny chris : ℕ) : 
  glenn = 24 →
  glenn = 4 * kenny →
  chris = kenny / 2 →
  glenn + kenny + chris = 33 :=
by sorry

end NUMINAMATH_CALUDE_cookie_problem_l582_58256


namespace NUMINAMATH_CALUDE_function_inequality_l582_58295

theorem function_inequality (f : ℝ → ℝ) (h1 : ∀ x, f x ≥ |x|) (h2 : ∀ x, f x ≥ 2^x) :
  ∀ a b : ℝ, f a ≤ 2^b → a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l582_58295


namespace NUMINAMATH_CALUDE_cosine_in_acute_triangle_l582_58273

theorem cosine_in_acute_triangle (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  (1/2) * a * b * Real.sin C = 5 →
  a = 3 →
  b = 4 →
  Real.cos C = Real.sqrt 11 / 6 := by
sorry

end NUMINAMATH_CALUDE_cosine_in_acute_triangle_l582_58273


namespace NUMINAMATH_CALUDE_nested_squares_shaded_ratio_l582_58200

/-- A nested square figure where inner squares have vertices at midpoints of outer squares' sides -/
structure NestedSquareFigure where
  /-- The number of nested squares in the figure -/
  num_squares : ℕ
  /-- Assertion that inner squares have vertices at midpoints of outer squares' sides -/
  vertices_at_midpoints : num_squares > 1 → True

/-- The ratio of shaded to unshaded area in a nested square figure -/
def shaded_to_unshaded_ratio (figure : NestedSquareFigure) : ℚ :=
  5 / 3

/-- Theorem stating that the ratio of shaded to unshaded area is 5:3 -/
theorem nested_squares_shaded_ratio (figure : NestedSquareFigure) :
    shaded_to_unshaded_ratio figure = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_squares_shaded_ratio_l582_58200


namespace NUMINAMATH_CALUDE_max_min_sum_zero_l582_58223

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_min_sum_zero :
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∃ x, f x = m) ∧ 
               (∀ x, n ≤ f x) ∧ (∃ x, f x = n) ∧
               (m + n = 0) := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_zero_l582_58223


namespace NUMINAMATH_CALUDE_fraction_power_product_l582_58262

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by sorry

end NUMINAMATH_CALUDE_fraction_power_product_l582_58262


namespace NUMINAMATH_CALUDE_topological_subgraph_is_subgraph_max_degree_3_topological_subgraph_iff_subgraph_l582_58244

/-- A graph. -/
structure Graph (V : Type) where
  edge : V → V → Prop

/-- The maximum degree of a graph. -/
def maxDegree {V : Type} (G : Graph V) : ℕ := sorry

/-- A subgraph relation between graphs. -/
def isSubgraph {V : Type} (H G : Graph V) : Prop := sorry

/-- A topological subgraph relation between graphs. -/
def isTopologicalSubgraph {V : Type} (H G : Graph V) : Prop := sorry

theorem topological_subgraph_is_subgraph {V : Type} (G H : Graph V) :
  isTopologicalSubgraph H G → isSubgraph H G := by sorry

theorem max_degree_3_topological_subgraph_iff_subgraph {V : Type} (G H : Graph V) :
  maxDegree G ≤ 3 →
  (isTopologicalSubgraph H G ↔ isSubgraph H G) := by sorry

end NUMINAMATH_CALUDE_topological_subgraph_is_subgraph_max_degree_3_topological_subgraph_iff_subgraph_l582_58244


namespace NUMINAMATH_CALUDE_smallest_n_value_l582_58214

theorem smallest_n_value (r g b : ℕ) (n : ℕ) : 
  (∃ m : ℕ, m > 0 ∧ 10 * r = m ∧ 18 * g = m ∧ 24 * b = m ∧ 25 * n = m) →
  n ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l582_58214


namespace NUMINAMATH_CALUDE_sum_of_non_common_roots_is_zero_l582_58299

/-- Given two quadratic equations with one common root, prove that the sum of the non-common roots is 0 -/
theorem sum_of_non_common_roots_is_zero (m : ℝ) :
  (∃ x : ℝ, x^2 + (m + 1) * x - 3 = 0 ∧ x^2 - 4 * x - m = 0) →
  (∃ α β γ : ℝ, 
    (α^2 + (m + 1) * α - 3 = 0 ∧ β^2 + (m + 1) * β - 3 = 0 ∧ α ≠ β) ∧
    (α^2 - 4 * α - m = 0 ∧ γ^2 - 4 * γ - m = 0 ∧ α ≠ γ) ∧
    β + γ = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_non_common_roots_is_zero_l582_58299


namespace NUMINAMATH_CALUDE_marble_158_is_gray_l582_58246

/-- Represents the color of a marble -/
inductive Color
  | Gray
  | White
  | Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : Color :=
  match n % 12 with
  | 0 | 1 | 2 | 3 | 4 => Color.Gray
  | 5 | 6 | 7 | 8 => Color.White
  | _ => Color.Black

theorem marble_158_is_gray : marbleColor 158 = Color.Gray := by
  sorry

end NUMINAMATH_CALUDE_marble_158_is_gray_l582_58246


namespace NUMINAMATH_CALUDE_square_of_number_ending_in_five_l582_58290

theorem square_of_number_ending_in_five (a : ℤ) : (10 * a + 5)^2 = 100 * a * (a + 1) + 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_number_ending_in_five_l582_58290


namespace NUMINAMATH_CALUDE_diet_soda_count_l582_58218

/-- Represents the number of apples in the grocery store -/
def num_apples : ℕ := 36

/-- Represents the number of regular soda bottles in the grocery store -/
def num_regular_soda : ℕ := 80

/-- Represents the number of diet soda bottles in the grocery store -/
def num_diet_soda : ℕ := 54

/-- The total number of bottles is 98 more than the number of apples -/
axiom total_bottles_relation : num_regular_soda + num_diet_soda = num_apples + 98

theorem diet_soda_count : num_diet_soda = 54 := by sorry

end NUMINAMATH_CALUDE_diet_soda_count_l582_58218


namespace NUMINAMATH_CALUDE_gcd_of_60_and_75_l582_58270

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_60_and_75_l582_58270
