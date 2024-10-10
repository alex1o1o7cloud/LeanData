import Mathlib

namespace undefined_value_expression_undefined_l1677_167759

theorem undefined_value (x : ℝ) : 
  (x^2 - 24*x + 144 = 0) ↔ (x = 12) := by sorry

theorem expression_undefined (x : ℝ) : 
  ¬(∃y : ℝ, y = (3*x^3 + 5) / (x^2 - 24*x + 144)) ↔ (x = 12) := by sorry

end undefined_value_expression_undefined_l1677_167759


namespace geometric_sequence_inequality_l1677_167760

/-- Given a geometric sequence with common ratio q < 0, prove that a₉S₈ > a₈S₉ -/
theorem geometric_sequence_inequality (a₁ : ℝ) (q : ℝ) (hq : q < 0) :
  let a : ℕ → ℝ := λ n => a₁ * q^(n - 1)
  let S : ℕ → ℝ := λ n => a₁ * (1 - q^n) / (1 - q)
  (a 9) * (S 8) > (a 8) * (S 9) :=
by sorry

end geometric_sequence_inequality_l1677_167760


namespace rectangular_solid_surface_area_l1677_167712

theorem rectangular_solid_surface_area
  (a b c : ℝ)
  (sum_edges : a + b + c = 14)
  (diagonal : a^2 + b^2 + c^2 = 11^2) :
  2 * (a * b + b * c + a * c) = 75 := by
  sorry

end rectangular_solid_surface_area_l1677_167712


namespace simultaneous_divisibility_by_17_l1677_167793

theorem simultaneous_divisibility_by_17 : ∃ (x y : ℤ), 
  (17 ∣ (2*x + 3*y)) ∧ (17 ∣ (9*x + 5*y)) := by
  sorry

end simultaneous_divisibility_by_17_l1677_167793


namespace paper_torn_fraction_l1677_167711

theorem paper_torn_fraction (perimeter : ℝ) (remaining_area : ℝ) : 
  perimeter = 32 → remaining_area = 48 → 
  (perimeter / 4)^2 - remaining_area = (1 / 4) * (perimeter / 4)^2 := by
  sorry

end paper_torn_fraction_l1677_167711


namespace tiling_combination_l1677_167768

def interior_angle (n : ℕ) : ℚ := (n - 2 : ℚ) * 180 / n

def can_tile (a b c : ℕ) : Prop :=
  ∃ (m n p : ℕ), m * interior_angle a + n * interior_angle b + p * interior_angle c = 360 ∧
  m + n + p = 4 ∧ m > 0 ∧ n > 0 ∧ p > 0

theorem tiling_combination :
  can_tile 3 4 6 ∧
  ¬can_tile 3 4 5 ∧
  ¬can_tile 3 4 7 ∧
  ¬can_tile 3 4 8 :=
sorry

end tiling_combination_l1677_167768


namespace inequality_solution_l1677_167745

theorem inequality_solution (x : ℝ) : -3 * x^2 + 5 * x + 4 < 0 ∧ x > 0 → x ∈ Set.Ioo 0 1 := by
  sorry

end inequality_solution_l1677_167745


namespace triangle_area_from_inradius_and_perimeter_l1677_167772

/-- Given a triangle with angles A and B, perimeter p, and inradius r, 
    proves that the area of the triangle is equal to r * (p / 2) -/
theorem triangle_area_from_inradius_and_perimeter 
  (A B : Real) (p r : Real) (h1 : A = 40) (h2 : B = 60) (h3 : p = 40) (h4 : r = 2.5) : 
  r * (p / 2) = 50 := by
  sorry

end triangle_area_from_inradius_and_perimeter_l1677_167772


namespace subtraction_decimal_l1677_167754

theorem subtraction_decimal : 3.56 - 1.29 = 2.27 := by sorry

end subtraction_decimal_l1677_167754


namespace common_tangent_line_l1677_167773

/-- Two circles O₁ and O₂ in the Cartesian coordinate system -/
structure TwoCircles where
  m : ℝ
  r₁ : ℝ
  r₂ : ℝ
  h₁ : m > 0
  h₂ : r₁ > 0
  h₃ : r₂ > 0
  h₄ : r₁ * r₂ = 2
  h₅ : (3 : ℝ) = r₁ / m
  h₆ : (1 : ℝ) = r₁
  h₇ : (2 : ℝ) ^ 2 + (2 : ℝ) ^ 2 = (2 - r₁ / m) ^ 2 + (2 - r₁) ^ 2 + r₁ ^ 2
  h₈ : (2 : ℝ) ^ 2 + (2 : ℝ) ^ 2 = (2 - r₂ / m) ^ 2 + (2 - r₂) ^ 2 + r₂ ^ 2

/-- The equation of another common tangent line is y = (4/3)x -/
theorem common_tangent_line (c : TwoCircles) :
  ∃ (k : ℝ), k = 4 / 3 ∧ ∀ (x y : ℝ), y = k * x → 
  (∃ (t : ℝ), (x - 3) ^ 2 + (y - 1) ^ 2 = t ^ 2 ∧ (x - c.r₂ / c.m) ^ 2 + (y - c.r₂) ^ 2 = t ^ 2) :=
by sorry

end common_tangent_line_l1677_167773


namespace roots_transformation_l1677_167725

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 5*r₁^2 + 12 = 0) ∧ 
  (r₂^3 - 5*r₂^2 + 12 = 0) ∧ 
  (r₃^3 - 5*r₃^2 + 12 = 0) → 
  ((3*r₁)^3 - 15*(3*r₁)^2 + 324 = 0) ∧ 
  ((3*r₂)^3 - 15*(3*r₂)^2 + 324 = 0) ∧ 
  ((3*r₃)^3 - 15*(3*r₃)^2 + 324 = 0) := by
sorry

end roots_transformation_l1677_167725


namespace function_identity_l1677_167741

theorem function_identity (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (f m + f n) = m + n) : 
  ∀ x : ℕ, f x = x := by
sorry

end function_identity_l1677_167741


namespace equal_area_triangles_l1677_167778

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem equal_area_triangles :
  triangleArea 25 25 30 = triangleArea 25 25 40 := by sorry

end equal_area_triangles_l1677_167778


namespace curve_intersects_median_l1677_167776

theorem curve_intersects_median (a b c : ℝ) (h : a + c - 2*b ≠ 0) :
  ∃! p : ℂ, 
    (∀ t : ℝ, p ≠ Complex.I * a * (Real.cos t)^4 + 2 * (1/2 + Complex.I * b) * (Real.cos t)^2 * (Real.sin t)^2 + (1 + Complex.I * c) * (Real.sin t)^4) ∧
    (p.re = 1/2) ∧
    (p.im = (a + c + 2*b) / 4) ∧
    (∃ k : ℝ, p.im - (a + b)/2 = (c - a) * (p.re - 1/4) + k * ((3/4 - 1/4) * Complex.I - ((b + c)/2 - (a + b)/2))) := by
  sorry

end curve_intersects_median_l1677_167776


namespace regression_line_not_always_through_point_l1677_167788

/-- A sample data point in a regression analysis -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Linear regression equation -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Check if a point lies on a line defined by a linear regression equation -/
def pointOnLine (p : DataPoint) (reg : LinearRegression) : Prop :=
  p.y = reg.b * p.x + reg.a

/-- Theorem stating that it's not necessarily true that a linear regression line passes through at least one sample point -/
theorem regression_line_not_always_through_point :
  ∃ (n : ℕ) (data : Fin n → DataPoint) (reg : LinearRegression),
    ∀ i : Fin n, ¬(pointOnLine (data i) reg) :=
sorry

end regression_line_not_always_through_point_l1677_167788


namespace quadratic_factor_problem_l1677_167726

theorem quadratic_factor_problem (d e : ℤ) :
  let q : ℝ → ℝ := fun x ↦ x^2 + d*x + e
  (∃ r : ℝ → ℝ, (fun x ↦ x^4 + x^3 + 8*x^2 + 7*x + 18) = q * r) ∧
  (∃ s : ℝ → ℝ, (fun x ↦ 2*x^4 + 3*x^3 + 9*x^2 + 8*x + 20) = q * s) →
  q 1 = -6 := by
sorry

end quadratic_factor_problem_l1677_167726


namespace absolute_value_simplification_l1677_167755

theorem absolute_value_simplification (a b : ℚ) (ha : a < 0) (hb : b > 0) :
  |a - b| + b = -a + 2*b := by sorry

end absolute_value_simplification_l1677_167755


namespace parabola_shift_l1677_167729

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 + 1

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := (x + 2)^2 - 2

/-- Theorem stating that the shifted parabola is equivalent to 
    shifting the original parabola 2 units left and 3 units down -/
theorem parabola_shift : 
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 2) - 3 := by
  sorry

end parabola_shift_l1677_167729


namespace average_of_geometric_sequence_l1677_167742

theorem average_of_geometric_sequence (z : ℝ) :
  let sequence := [0, 2*z, 4*z, 8*z, 16*z]
  (sequence.sum / sequence.length : ℝ) = 6*z :=
by sorry

end average_of_geometric_sequence_l1677_167742


namespace arithmetic_sequence_sum_l1677_167798

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    if 2a_7 - a_8 = 5, then S_11 = 55 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
  (∀ n, S n = n * (a 1 + a n) / 2) →    -- sum formula
  (2 * a 7 - a 8 = 5) →                 -- given condition
  S 11 = 55 := by
sorry

end arithmetic_sequence_sum_l1677_167798


namespace complex_arithmetic_evaluation_l1677_167763

theorem complex_arithmetic_evaluation : (7 - 3*I) - 3*(2 - 5*I) = 1 + 12*I := by
  sorry

end complex_arithmetic_evaluation_l1677_167763


namespace even_sum_condition_l1677_167722

theorem even_sum_condition (m n : ℤ) : 
  (∃ (k l : ℤ), m = 2 * k ∧ n = 2 * l → ∃ (p : ℤ), m + n = 2 * p) ∧ 
  (∃ (m n : ℤ), ∃ (q : ℤ), m + n = 2 * q ∧ ¬(∃ (r s : ℤ), m = 2 * r ∧ n = 2 * s)) :=
by sorry

end even_sum_condition_l1677_167722


namespace only_component_life_uses_experiments_l1677_167784

/-- Represents a method of data collection --/
inductive DataCollectionMethod
  | Observation
  | Experiment
  | Investigation

/-- Represents the different scenarios --/
inductive Scenario
  | TemperatureMeasurement
  | ComponentLifeDetermination
  | TVRatings
  | CounterfeitDetection

/-- Maps each scenario to its typical data collection method --/
def typicalMethod (s : Scenario) : DataCollectionMethod :=
  match s with
  | Scenario.TemperatureMeasurement => DataCollectionMethod.Observation
  | Scenario.ComponentLifeDetermination => DataCollectionMethod.Experiment
  | Scenario.TVRatings => DataCollectionMethod.Investigation
  | Scenario.CounterfeitDetection => DataCollectionMethod.Investigation

theorem only_component_life_uses_experiments :
  ∀ s : Scenario, typicalMethod s = DataCollectionMethod.Experiment ↔ s = Scenario.ComponentLifeDetermination :=
by sorry


end only_component_life_uses_experiments_l1677_167784


namespace consecutive_odd_product_l1677_167706

theorem consecutive_odd_product (n : ℕ) : (2*n - 1) * (2*n + 1) = (2*n)^2 - 1 := by
  sorry

end consecutive_odd_product_l1677_167706


namespace subtract_square_thirty_l1677_167764

theorem subtract_square_thirty : 30 - 5^2 = 5 := by
  sorry

end subtract_square_thirty_l1677_167764


namespace calculator_display_after_50_presses_l1677_167705

def calculator_function (x : ℚ) : ℚ := 1 / (1 - x)

def iterate_function (f : ℚ → ℚ) (x : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_function f x n)

theorem calculator_display_after_50_presses :
  iterate_function calculator_function (1/2) 50 = -1 := by
  sorry

end calculator_display_after_50_presses_l1677_167705


namespace equation_solutions_l1677_167704

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 3*x = 0 ↔ x = 0 ∨ x = 3) ∧
  (∀ y : ℝ, 2*y^2 + 4*y = y + 2 ↔ y = -2 ∨ y = 1/2) ∧
  (∀ y : ℝ, (2*y + 1)^2 - 25 = 0 ↔ y = -3 ∨ y = 2) :=
by sorry

end equation_solutions_l1677_167704


namespace specific_hexagon_area_l1677_167766

/-- A hexagon in 2D space -/
structure Hexagon where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  v5 : ℝ × ℝ
  v6 : ℝ × ℝ

/-- The area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- The specific hexagon from the problem -/
def specificHexagon : Hexagon :=
  { v1 := (0, 0)
    v2 := (1, 4)
    v3 := (3, 4)
    v4 := (4, 0)
    v5 := (3, -4)
    v6 := (1, -4) }

/-- Theorem stating that the area of the specific hexagon is 24 square units -/
theorem specific_hexagon_area :
  hexagonArea specificHexagon = 24 := by sorry

end specific_hexagon_area_l1677_167766


namespace exists_integer_sqrt_8m_l1677_167708

theorem exists_integer_sqrt_8m : ∃ m : ℕ+, ∃ k : ℕ, (8 * m.val : ℕ) = k^2 := by
  sorry

end exists_integer_sqrt_8m_l1677_167708


namespace sum_of_two_numbers_l1677_167751

theorem sum_of_two_numbers (x y : ℝ) : x * y = 437 ∧ |x - y| = 4 → x + y = 42 := by
  sorry

end sum_of_two_numbers_l1677_167751


namespace proposition_count_l1677_167748

theorem proposition_count : ∃! n : ℕ, n = 2 ∧ 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x * y ≥ 0) ∧ 
  (∀ x y : ℝ, x * y ≥ 0 → x ≥ 0 ∧ y ≥ 0 ∨ x ≤ 0 ∧ y ≤ 0) ∧
  (∃ x y : ℝ, ¬(x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)) ∧
  (∀ x y : ℝ, x * y < 0 → x < 0 ∨ y < 0) :=
by
  sorry

end proposition_count_l1677_167748


namespace weight_after_first_week_l1677_167797

/-- Given Jessie's initial weight and weight loss in the first week, 
    calculate her weight after the first week of jogging. -/
theorem weight_after_first_week 
  (initial_weight : ℕ) 
  (weight_loss_first_week : ℕ) 
  (h1 : initial_weight = 92) 
  (h2 : weight_loss_first_week = 56) : 
  initial_weight - weight_loss_first_week = 36 := by
  sorry

end weight_after_first_week_l1677_167797


namespace triangle_inequality_l1677_167733

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a + b - c) * (a - b + c) * (-a + b + c) ≤ a * b * c := by
  sorry

end triangle_inequality_l1677_167733


namespace right_triangle_with_condition_l1677_167785

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def satisfies_condition (a b c : ℕ) : Prop :=
  a + b = c + 6

theorem right_triangle_with_condition :
  ∀ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 →
    a ≤ b →
    is_right_triangle a b c →
    satisfies_condition a b c →
    ((a = 7 ∧ b = 24 ∧ c = 25) ∨
     (a = 8 ∧ b = 15 ∧ c = 17) ∨
     (a = 9 ∧ b = 12 ∧ c = 15)) :=
by
  sorry

#check right_triangle_with_condition

end right_triangle_with_condition_l1677_167785


namespace class_average_problem_l1677_167786

theorem class_average_problem (total_students : Nat) (high_score_students : Nat) 
  (zero_score_students : Nat) (high_score : Nat) (class_average : Rat) :
  total_students = 25 →
  high_score_students = 3 →
  zero_score_students = 3 →
  high_score = 95 →
  class_average = 45.6 →
  let remaining_students := total_students - high_score_students - zero_score_students
  let total_score := (total_students : Rat) * class_average
  let high_score_total := (high_score_students : Rat) * high_score
  let remaining_average := (total_score - high_score_total) / remaining_students
  remaining_average = 45 := by
  sorry

end class_average_problem_l1677_167786


namespace gcd_binomial_integrality_l1677_167757

theorem gcd_binomial_integrality (m n : ℕ) (h1 : 1 ≤ m) (h2 : m ≤ n) :
  ∃ (a b : ℤ), (Nat.gcd m n : ℚ) / n * Nat.choose n m = a * Nat.choose (n-1) (m-1) + b * Nat.choose n m := by
  sorry

end gcd_binomial_integrality_l1677_167757


namespace lg_calculation_l1677_167752

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem lg_calculation : (lg 2)^2 + lg 20 * lg 5 = 1 := by
  sorry

end lg_calculation_l1677_167752


namespace particle_max_elevation_l1677_167744

noncomputable def s (t : ℝ) : ℝ := 200 * t - 17 * t^2 - 3 * t^3

theorem particle_max_elevation :
  ∃ (max_height : ℝ), 
    (∀ t : ℝ, t ≥ 0 → s t ≤ max_height) ∧ 
    (∃ t : ℝ, t ≥ 0 ∧ s t = max_height) ∧
    (abs (max_height - 368.1) < 0.1) := by
  sorry

end particle_max_elevation_l1677_167744


namespace hcf_from_lcm_and_product_l1677_167761

/-- Given two positive integers with LCM 750 and product 18750, prove their HCF is 25 -/
theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750)
  (h_product : a * b = 18750) : 
  Nat.gcd a b = 25 := by
  sorry

end hcf_from_lcm_and_product_l1677_167761


namespace more_students_than_pets_l1677_167727

theorem more_students_than_pets :
  let num_classrooms : ℕ := 5
  let students_per_classroom : ℕ := 25
  let rabbits_per_classroom : ℕ := 3
  let guinea_pigs_per_classroom : ℕ := 3
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_rabbits : ℕ := num_classrooms * rabbits_per_classroom
  let total_guinea_pigs : ℕ := num_classrooms * guinea_pigs_per_classroom
  let total_pets : ℕ := total_rabbits + total_guinea_pigs
  total_students - total_pets = 95 := by
  sorry

end more_students_than_pets_l1677_167727


namespace toy_cost_price_l1677_167716

/-- The cost price of a toy -/
def cost_price : ℕ := sorry

/-- The number of toys sold -/
def toys_sold : ℕ := 18

/-- The total selling price of all toys -/
def total_selling_price : ℕ := 16800

/-- The number of toys whose cost price equals the gain -/
def toys_equal_to_gain : ℕ := 3

theorem toy_cost_price : 
  cost_price * (toys_sold + toys_equal_to_gain) = total_selling_price ∧ 
  cost_price = 800 := by sorry

end toy_cost_price_l1677_167716


namespace problem_1_problem_2_l1677_167789

-- Problem 1
theorem problem_1 (a b : ℝ) (h1 : (a + b)^2 = 6) (h2 : (a - b)^2 = 2) : 
  a^2 + b^2 = 4 ∧ a * b = 1 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x + 1/x = 3) : 
  x^2 + 1/x^2 = 7 := by sorry

end problem_1_problem_2_l1677_167789


namespace complex_fraction_equality_l1677_167719

theorem complex_fraction_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end complex_fraction_equality_l1677_167719


namespace quadratic_minimum_l1677_167771

/-- The quadratic function f(x) = 2(x - 4)² + 6 has a minimum value of 6 -/
theorem quadratic_minimum (x : ℝ) : ∀ y : ℝ, 2 * (x - 4)^2 + 6 ≥ 6 := by
  sorry

end quadratic_minimum_l1677_167771


namespace fibonacci_gcd_2002_1998_l1677_167721

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => fibonacci (n + 2) + fibonacci (n + 1)

theorem fibonacci_gcd_2002_1998 : Nat.gcd (fibonacci 2002) (fibonacci 1998) = 1 := by
  sorry

end fibonacci_gcd_2002_1998_l1677_167721


namespace equation_solution_l1677_167791

theorem equation_solution (x : ℝ) :
  (|Real.cos x| - Real.cos (3 * x)) / (Real.cos x * Real.sin (2 * x)) = 2 / Real.sqrt 3 ↔
  (∃ k : ℤ, x = π / 6 + 2 * k * π ∨ x = 5 * π / 6 + 2 * k * π ∨ x = 4 * π / 3 + 2 * k * π) :=
by sorry

end equation_solution_l1677_167791


namespace journey_time_proof_l1677_167765

theorem journey_time_proof (s : ℝ) (h1 : s > 0) (h2 : s - 1/2 > 0) : 
  (45 / (s - 1/2) - 45 / s = 3/4) → (45 / s = 45 / s) :=
by
  sorry

end journey_time_proof_l1677_167765


namespace complex_equation_sum_l1677_167720

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (1 - Complex.I) * (2 + Complex.I) → a + b = 2 := by
  sorry

end complex_equation_sum_l1677_167720


namespace function_symmetry_l1677_167734

/-- Given a function f and a real number a, 
    if f(x) = x³cos(x) + 1 and f(a) = 11, then f(-a) = -9 -/
theorem function_symmetry (f : ℝ → ℝ) (a : ℝ) 
    (h1 : ∀ x, f x = x^3 * Real.cos x + 1) 
    (h2 : f a = 11) : 
  f (-a) = -9 := by
sorry

end function_symmetry_l1677_167734


namespace z_in_second_quadrant_l1677_167700

noncomputable def z : ℂ := Complex.exp (-4 * Complex.I)

theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 :=
sorry

end z_in_second_quadrant_l1677_167700


namespace even_number_less_than_square_l1677_167777

theorem even_number_less_than_square (m : ℕ) (h1 : m > 1) (h2 : Even m) : m < m^2 := by
  sorry

end even_number_less_than_square_l1677_167777


namespace circle_squares_inequality_l1677_167770

theorem circle_squares_inequality (x y : ℝ) : 
  abs x + abs y ≤ Real.sqrt (2 * (x^2 + y^2)) ∧ 
  Real.sqrt (2 * (x^2 + y^2)) ≤ 2 * max (abs x) (abs y) := by
  sorry

end circle_squares_inequality_l1677_167770


namespace total_cost_of_two_items_l1677_167737

/-- The total cost of two items is the sum of their individual costs -/
theorem total_cost_of_two_items (yoyo_cost whistle_cost : ℕ) :
  yoyo_cost = 24 → whistle_cost = 14 →
  yoyo_cost + whistle_cost = 38 := by
  sorry

end total_cost_of_two_items_l1677_167737


namespace all_children_receive_candy_candy_distribution_works_l1677_167723

/-- Represents the candy distribution function -/
def candyDistribution (n : ℕ+) (k : ℕ) : ℕ :=
  (k * (k + 1) / 2) % n

/-- Theorem stating that all children receive candy iff n is a power of 2 -/
theorem all_children_receive_candy (n : ℕ+) :
  (∀ i : ℕ, i < n → ∃ k : ℕ, candyDistribution n k = i) ↔ ∃ m : ℕ, n = 2^m := by
  sorry

/-- Corollary: The number of children for which the candy distribution works -/
theorem candy_distribution_works (n : ℕ+) :
  (∀ i : ℕ, i < n → ∃ k : ℕ, candyDistribution n k = i) → ∃ m : ℕ, n = 2^m := by
  sorry

end all_children_receive_candy_candy_distribution_works_l1677_167723


namespace maria_ate_two_cookies_l1677_167794

/-- Given Maria's cookie distribution, prove she ate 2 cookies. -/
theorem maria_ate_two_cookies
  (initial_cookies : ℕ)
  (friend_cookies : ℕ)
  (final_cookies : ℕ)
  (h1 : initial_cookies = 19)
  (h2 : friend_cookies = 5)
  (h3 : final_cookies = 5)
  (h4 : ∃ (family_cookies : ℕ), 
    2 * family_cookies = initial_cookies - friend_cookies) :
  initial_cookies - friend_cookies - 
    ((initial_cookies - friend_cookies) / 2) - final_cookies = 2 :=
by sorry


end maria_ate_two_cookies_l1677_167794


namespace solution_set_of_inequality_l1677_167775

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x) / (3 * x - 1) > 1 ↔ 1 / 3 < x ∧ x < 1 := by sorry

end solution_set_of_inequality_l1677_167775


namespace regular_polygon_sides_l1677_167796

theorem regular_polygon_sides (internal_angle : ℝ) (h : internal_angle = 150) :
  (360 : ℝ) / (180 - internal_angle) = 12 := by
  sorry

end regular_polygon_sides_l1677_167796


namespace f_at_negative_five_l1677_167792

/-- Given a function f(x) = x^2 + 2x - 3, prove that f(-5) = 12 -/
theorem f_at_negative_five (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x - 3) : f (-5) = 12 := by
  sorry

end f_at_negative_five_l1677_167792


namespace l_plaque_four_equal_parts_l1677_167701

/-- An L-shaped plaque -/
structure LPlaque where
  width : ℝ
  height : ℝ
  thickness : ℝ

/-- A straight cut on the plaque -/
inductive Cut
  | Vertical (x : ℝ)
  | Horizontal (y : ℝ)

/-- The result of applying cuts to an L-shaped plaque -/
def applyCuts (p : LPlaque) (cuts : List Cut) : List (Set (ℝ × ℝ)) :=
  sorry

/-- Check if all pieces have equal area -/
def equalAreas (pieces : List (Set (ℝ × ℝ))) : Prop :=
  sorry

/-- Main theorem: An L-shaped plaque can be divided into four equal parts using straight cuts -/
theorem l_plaque_four_equal_parts (p : LPlaque) :
  ∃ (cuts : List Cut), (applyCuts p cuts).length = 4 ∧ equalAreas (applyCuts p cuts) :=
sorry

end l_plaque_four_equal_parts_l1677_167701


namespace coefficient_m5n5_in_expansion_l1677_167790

theorem coefficient_m5n5_in_expansion : Nat.choose 10 5 = 252 := by
  sorry

end coefficient_m5n5_in_expansion_l1677_167790


namespace function_zeros_sum_l1677_167781

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.exp (1 - x) - a * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x + Real.exp (1 - x) - a

theorem function_zeros_sum (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : g a x₁ = 0) 
  (h₂ : g a x₂ = 0) 
  (h₃ : f a x₁ + f a x₂ = -4) : 
  a = 4 := by sorry

end function_zeros_sum_l1677_167781


namespace prob_two_slate_is_11_105_l1677_167703

-- Define the number of rocks for each type
def slate_rocks : ℕ := 12
def pumice_rocks : ℕ := 16
def granite_rocks : ℕ := 8

-- Define the total number of rocks
def total_rocks : ℕ := slate_rocks + pumice_rocks + granite_rocks

-- Define the probability of selecting two slate rocks
def prob_two_slate : ℚ := (slate_rocks : ℚ) / total_rocks * (slate_rocks - 1) / (total_rocks - 1)

theorem prob_two_slate_is_11_105 : prob_two_slate = 11 / 105 := by
  sorry

end prob_two_slate_is_11_105_l1677_167703


namespace parallelogram_bisecting_line_slope_l1677_167787

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Determines if a line through the origin cuts a parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (m : ℝ) : Prop := sorry

/-- The specific parallelogram from the problem -/
def problem_parallelogram : Parallelogram :=
  { v1 := { x := 2, y := 5 }
  , v2 := { x := 2, y := 23 }
  , v3 := { x := 7, y := 38 }
  , v4 := { x := 7, y := 20 }
  }

theorem parallelogram_bisecting_line_slope :
  cuts_into_congruent_polygons problem_parallelogram (43/9) := by sorry

end parallelogram_bisecting_line_slope_l1677_167787


namespace system_solution_l1677_167715

theorem system_solution :
  ∃ (x y : ℚ), (4 * x = -10 - 3 * y) ∧ (6 * x = 5 * y - 32) ∧ (x = -73/19) ∧ (y = 34/19) := by
  sorry

end system_solution_l1677_167715


namespace halley_21st_century_appearance_l1677_167728

/-- Represents the year of Halley's Comet's appearance -/
def halley_appearance (n : ℕ) : ℕ := 1682 + 76 * n

/-- Predicate to check if a year is in the 21st century -/
def is_21st_century (year : ℕ) : Prop := 2001 ≤ year ∧ year ≤ 2100

theorem halley_21st_century_appearance :
  ∃ n : ℕ, is_21st_century (halley_appearance n) ∧ halley_appearance n = 2062 :=
sorry

end halley_21st_century_appearance_l1677_167728


namespace largest_integer_x_l1677_167762

theorem largest_integer_x : ∃ x : ℤ, 
  (∀ y : ℤ, (7 - 3 * y > 20 ∧ y ≥ -10) → y ≤ x) ∧ 
  (7 - 3 * x > 20 ∧ x ≥ -10) ∧ 
  x = -5 := by
sorry

end largest_integer_x_l1677_167762


namespace tenth_letter_shift_l1677_167753

def shift_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem tenth_letter_shift :
  ∀ (letter : Char),
  (shift_sum 10) % 26 = 3 :=
by
  sorry

end tenth_letter_shift_l1677_167753


namespace initial_ducks_l1677_167736

theorem initial_ducks (initial additional total : ℕ) 
  (h1 : additional = 20)
  (h2 : total = 33)
  (h3 : initial + additional = total) : 
  initial = 13 := by
sorry

end initial_ducks_l1677_167736


namespace ticket_problem_l1677_167702

/-- Represents the ticket distribution and pricing for a football match --/
structure TicketInfo where
  total : ℕ  -- Total number of tickets
  typeA : ℕ  -- Number of Type A tickets
  m : ℕ      -- Price parameter

/-- Conditions for the ticket distribution and pricing --/
def validTicketInfo (info : TicketInfo) : Prop :=
  info.total = 500 ∧
  info.typeA ≥ 3 * (info.total - info.typeA) ∧
  500 * (1 + (info.m + 10) / 100) * (info.m + 20) = 56000 ∧
  info.m > 0

theorem ticket_problem (info : TicketInfo) (h : validTicketInfo info) :
  info.typeA ≥ 375 ∧ info.m = 50 := by
  sorry


end ticket_problem_l1677_167702


namespace income_expenditure_ratio_l1677_167795

/-- Given a person's income and savings, calculate the ratio of income to expenditure -/
theorem income_expenditure_ratio (income savings : ℕ) (h1 : income = 18000) (h2 : savings = 3600) :
  (income : ℚ) / (income - savings) = 5 / 4 := by
  sorry

end income_expenditure_ratio_l1677_167795


namespace sum_of_series_l1677_167756

theorem sum_of_series : 
  (∑' n : ℕ, (4 * n + 1 : ℝ) / (3 : ℝ) ^ n) = 7 / 2 := by sorry

end sum_of_series_l1677_167756


namespace consecutive_integers_sum_l1677_167739

theorem consecutive_integers_sum (n : ℤ) : n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end consecutive_integers_sum_l1677_167739


namespace converse_and_inverse_false_l1677_167707

-- Define the universe of polygons
variable (Polygon : Type)

-- Define properties of polygons
variable (is_rhombus : Polygon → Prop)
variable (is_parallelogram : Polygon → Prop)

-- Original statement
axiom original_statement : ∀ p : Polygon, is_rhombus p → is_parallelogram p

-- Theorem to prove
theorem converse_and_inverse_false :
  (¬ ∀ p : Polygon, is_parallelogram p → is_rhombus p) ∧
  (¬ ∀ p : Polygon, ¬is_rhombus p → ¬is_parallelogram p) :=
by sorry

end converse_and_inverse_false_l1677_167707


namespace position_vector_coefficients_l1677_167743

/-- Given points A and B, and points P and Q on line segment AB,
    prove that their position vectors have the specified coefficients. -/
theorem position_vector_coefficients
  (A B P Q : ℝ × ℝ) -- A, B, P, Q are points in 2D space
  (h_P : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) -- P is on AB
  (h_Q : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Q = (1 - s) • A + s • B) -- Q is on AB
  (h_P_ratio : (dist A P) / (dist P B) = 3 / 5) -- AP:PB = 3:5
  (h_Q_ratio : (dist A Q) / (dist Q B) = 4 / 3) -- AQ:QB = 4:3
  : (∃ t₁ u₁ : ℝ, P = t₁ • A + u₁ • B ∧ t₁ = 5/8 ∧ u₁ = 3/8) ∧
    (∃ t₂ u₂ : ℝ, Q = t₂ • A + u₂ • B ∧ t₂ = 3/7 ∧ u₂ = 4/7) :=
by sorry

end position_vector_coefficients_l1677_167743


namespace largest_prime_divisor_of_expression_l1677_167774

theorem largest_prime_divisor_of_expression : 
  ∃ p : ℕ, 
    Prime p ∧ 
    p ∣ (Nat.factorial 12 + Nat.factorial 13 + 17) ∧
    ∀ q : ℕ, Prime q → q ∣ (Nat.factorial 12 + Nat.factorial 13 + 17) → q ≤ p :=
by sorry

end largest_prime_divisor_of_expression_l1677_167774


namespace b_eventually_constant_iff_square_l1677_167799

/-- The greatest integer m such that m^2 ≤ n -/
def m (n : ℕ) : ℕ := Nat.sqrt n

/-- d(n) = n - m^2, where m is the greatest integer such that m^2 ≤ n -/
def d (n : ℕ) : ℕ := n - (m n)^2

/-- The sequence b_i defined by b_{k+1} = b_k + d(b_k) -/
def b : ℕ → ℕ → ℕ
  | b_0, 0 => b_0
  | b_0, k + 1 => b b_0 k + d (b b_0 k)

/-- A sequence is eventually constant if there exists an N such that
    for all i ≥ N, the i-th term equals the N-th term -/
def EventuallyConstant (s : ℕ → ℕ) : Prop :=
  ∃ N, ∀ i, N ≤ i → s i = s N

/-- Main theorem: b_i is eventually constant iff b_0 is a perfect square -/
theorem b_eventually_constant_iff_square (b_0 : ℕ) :
  EventuallyConstant (b b_0) ↔ ∃ k, b_0 = k^2 := by sorry

end b_eventually_constant_iff_square_l1677_167799


namespace rectangle_dimension_change_l1677_167782

theorem rectangle_dimension_change (w l : ℝ) (h_w_pos : w > 0) (h_l_pos : l > 0) :
  let new_w := 1.4 * w
  let new_l := l / 1.4
  let area := w * l
  let new_area := new_w * new_l
  new_area = area ∧ (1 - new_l / l) * 100 = 100 * (1 - 1 / 1.4) := by
sorry

end rectangle_dimension_change_l1677_167782


namespace power_eleven_mod_120_l1677_167780

theorem power_eleven_mod_120 : 11^2023 % 120 = 11 := by sorry

end power_eleven_mod_120_l1677_167780


namespace repeating_decimal_equals_fraction_l1677_167740

/-- The repeating decimal 0.3̄45 as a real number -/
def repeating_decimal : ℚ := 3/10 + 45/990

/-- The fraction 83/110 -/
def target_fraction : ℚ := 83/110

/-- Theorem stating that the repeating decimal 0.3̄45 is equal to the fraction 83/110 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end repeating_decimal_equals_fraction_l1677_167740


namespace chord_length_squared_l1677_167758

/-- Two circles with radii 8 and 6, centers 12 units apart, intersecting at P.
    Q and R are points on the circles such that QP = PR. -/
structure CircleConfiguration where
  circle1_radius : ℝ
  circle2_radius : ℝ
  center_distance : ℝ
  chord_length : ℝ
  h1 : circle1_radius = 8
  h2 : circle2_radius = 6
  h3 : center_distance = 12
  h4 : chord_length > 0

/-- The square of the chord length in the given circle configuration is 130. -/
theorem chord_length_squared (config : CircleConfiguration) : 
  config.chord_length ^ 2 = 130 := by
  sorry

end chord_length_squared_l1677_167758


namespace pentagon_rectangle_ratio_l1677_167710

/-- Given a regular pentagon with perimeter 60 inches and a rectangle with perimeter 80 inches
    where the length is twice the width, the ratio of the pentagon's side length to the rectangle's
    width is 9/10. -/
theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side rectangle_width : ℝ),
    pentagon_side * 5 = 60 →
    rectangle_width * 6 = 80 →
    pentagon_side / rectangle_width = 9 / 10 := by
  sorry

end pentagon_rectangle_ratio_l1677_167710


namespace smallest_value_of_root_products_l1677_167724

def g (x : ℝ) : ℝ := x^4 + 16*x^3 + 69*x^2 + 112*x + 64

theorem smallest_value_of_root_products (w₁ w₂ w₃ w₄ : ℝ) 
  (h₁ : g w₁ = 0) (h₂ : g w₂ = 0) (h₃ : g w₃ = 0) (h₄ : g w₄ = 0) :
  ∃ (min : ℝ), min = 8 ∧ ∀ (p : ℝ), p = |w₁*w₂ + w₃*w₄| → p ≥ min :=
by sorry

end smallest_value_of_root_products_l1677_167724


namespace mike_book_count_l1677_167717

/-- The number of books Tim has -/
def tim_books : ℕ := 22

/-- The total number of books Tim and Mike have together -/
def total_books : ℕ := 42

/-- The number of books Mike has -/
def mike_books : ℕ := total_books - tim_books

theorem mike_book_count : mike_books = 20 := by
  sorry

end mike_book_count_l1677_167717


namespace max_value_of_function_l1677_167718

theorem max_value_of_function (x y : ℝ) (h : x^2 + y^2 = 25) :
  (∀ a b : ℝ, a^2 + b^2 = 25 →
    Real.sqrt (8 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50) ≥
    Real.sqrt (8 * b - 6 * a + 50) + Real.sqrt (8 * b + 6 * a + 50)) ∧
  (∃ a b : ℝ, a^2 + b^2 = 25 ∧
    Real.sqrt (8 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50) =
    Real.sqrt (8 * b - 6 * a + 50) + Real.sqrt (8 * b + 6 * a + 50) ∧
    Real.sqrt (8 * b - 6 * a + 50) + Real.sqrt (8 * b + 6 * a + 50) = 6 * Real.sqrt 10) :=
by sorry

end max_value_of_function_l1677_167718


namespace cistern_wet_surface_area_l1677_167750

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem stating the total wet surface area of a specific cistern -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 8 6 1.25 = 83 := by
  sorry

end cistern_wet_surface_area_l1677_167750


namespace circle_center_and_radius_l1677_167713

/-- The polar equation of a circle -/
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The Cartesian equation of a circle with center (h, k) and radius r -/
def cartesian_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the circle ρ = 2cosθ has center (1, 0) and radius 1 -/
theorem circle_center_and_radius :
  ∀ x y ρ θ : ℝ,
  polar_equation ρ θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  cartesian_equation x y 1 0 1 :=
by sorry

end circle_center_and_radius_l1677_167713


namespace second_number_value_l1677_167714

theorem second_number_value (x y : ℕ) (h1 : x + y = 33) (h2 : y = 2 * x) : y = 22 := by
  sorry

end second_number_value_l1677_167714


namespace picture_distance_from_right_end_l1677_167731

/-- Given a wall and a picture with specific dimensions and placement,
    calculate the distance from the right end of the wall to the nearest edge of the picture. -/
theorem picture_distance_from_right_end 
  (wall_width : ℝ) 
  (picture_width : ℝ) 
  (left_gap : ℝ) 
  (h1 : wall_width = 24)
  (h2 : picture_width = 4)
  (h3 : left_gap = 5) :
  wall_width - (left_gap + picture_width) = 15 := by
  sorry

#check picture_distance_from_right_end

end picture_distance_from_right_end_l1677_167731


namespace third_term_range_l1677_167779

/-- A sequence of positive real numbers satisfying certain conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (a 1 = 1) ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1)^2 + a n^2 < (5/2) * a (n + 1) * a n) ∧
  (a 2 = 3/2) ∧
  (a 4 = 4)

/-- The third term of the sequence is within the range (2, 3) -/
theorem third_term_range (a : ℕ → ℝ) (h : SpecialSequence a) :
  ∃ x, a 3 = x ∧ 2 < x ∧ x < 3 := by
  sorry

end third_term_range_l1677_167779


namespace negative_integral_of_negative_function_l1677_167769

theorem negative_integral_of_negative_function 
  {f : ℝ → ℝ} {a b : ℝ} 
  (hf : Continuous f) 
  (hneg : ∀ x, f x < 0) 
  (hab : a < b) : 
  ∫ x in a..b, f x < 0 := by
  sorry

end negative_integral_of_negative_function_l1677_167769


namespace mathematical_run_disqualified_team_size_l1677_167730

theorem mathematical_run_disqualified_team_size 
  (initial_teams : ℕ) 
  (initial_average : ℕ) 
  (final_teams : ℕ) 
  (final_average : ℕ) 
  (h1 : initial_teams = 9)
  (h2 : initial_average = 7)
  (h3 : final_teams = initial_teams - 1)
  (h4 : final_average = 6) :
  initial_teams * initial_average - final_teams * final_average = 15 :=
by sorry

end mathematical_run_disqualified_team_size_l1677_167730


namespace doubled_container_volume_l1677_167735

/-- A cylindrical container that can hold water -/
structure Container :=
  (volume : ℝ)
  (isOriginal : Bool)

/-- Double the dimensions of a container -/
def doubleContainer (c : Container) : Container :=
  { volume := 8 * c.volume, isOriginal := false }

theorem doubled_container_volume (c : Container) 
  (h1 : c.isOriginal = true) 
  (h2 : c.volume = 3) : 
  (doubleContainer c).volume = 24 := by
sorry

end doubled_container_volume_l1677_167735


namespace total_fans_l1677_167747

/-- The number of students who like basketball -/
def basketball_fans : ℕ := 7

/-- The number of students who like cricket -/
def cricket_fans : ℕ := 5

/-- The number of students who like both basketball and cricket -/
def both_fans : ℕ := 3

/-- Theorem: The number of students who like basketball or cricket or both is 9 -/
theorem total_fans : basketball_fans + cricket_fans - both_fans = 9 := by
  sorry

end total_fans_l1677_167747


namespace max_andy_consumption_l1677_167749

def total_cookies : ℕ := 36

def cookie_distribution (andy alexa ann : ℕ) : Prop :=
  ∃ k : ℕ+, alexa = k * andy ∧ ann = 2 * andy ∧ andy + alexa + ann = total_cookies

def max_andy_cookies : ℕ := 9

theorem max_andy_consumption :
  ∀ andy alexa ann : ℕ,
    cookie_distribution andy alexa ann →
    andy ≤ max_andy_cookies :=
by sorry

end max_andy_consumption_l1677_167749


namespace jasons_punch_problem_l1677_167783

/-- Represents the recipe for Jason's punch -/
structure PunchRecipe where
  water : ℝ
  lemon_juice : ℝ
  sugar : ℝ

/-- Represents the actual amounts used in Jason's punch -/
structure PunchIngredients where
  water : ℝ
  lemon_juice : ℝ
  sugar : ℝ

/-- The recipe ratios are correct -/
def recipe_ratios_correct (recipe : PunchRecipe) : Prop :=
  recipe.water = 5 * recipe.lemon_juice ∧ 
  recipe.lemon_juice = 3 * recipe.sugar

/-- The actual ingredients follow the recipe ratios -/
def ingredients_follow_recipe (recipe : PunchRecipe) (ingredients : PunchIngredients) : Prop :=
  ingredients.water / ingredients.lemon_juice = recipe.water / recipe.lemon_juice ∧
  ingredients.lemon_juice / ingredients.sugar = recipe.lemon_juice / recipe.sugar

/-- Jason's punch problem -/
theorem jasons_punch_problem (recipe : PunchRecipe) (ingredients : PunchIngredients) :
  recipe_ratios_correct recipe →
  ingredients_follow_recipe recipe ingredients →
  ingredients.lemon_juice = 5 →
  ingredients.water = 25 := by
  sorry

end jasons_punch_problem_l1677_167783


namespace fencing_championship_medals_l1677_167738

/-- The number of ways to select first and second place winners from n fencers -/
def awardMedals (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 72 ways to award first and second place medals among 9 fencers -/
theorem fencing_championship_medals :
  awardMedals 9 = 72 := by
  sorry

end fencing_championship_medals_l1677_167738


namespace point_on_terminal_side_l1677_167767

theorem point_on_terminal_side (t : ℝ) (θ : ℝ) : 
  ((-2 : ℝ) = Real.cos θ * Real.sqrt (4 + t^2)) →
  (t = Real.sin θ * Real.sqrt (4 + t^2)) →
  (Real.sin θ + Real.cos θ = Real.sqrt 5 / 5) →
  t = 4 := by
  sorry

end point_on_terminal_side_l1677_167767


namespace sum_of_parts_for_specific_complex_l1677_167732

theorem sum_of_parts_for_specific_complex (z : ℂ) (h : z = 1 - Complex.I) : 
  z.re + z.im = 0 := by
  sorry

end sum_of_parts_for_specific_complex_l1677_167732


namespace inequality_constraint_on_a_l1677_167709

theorem inequality_constraint_on_a (a : ℝ) : 
  (∀ x : ℝ, (Real.exp x - a * x) * (x^2 - a * x + 1) ≥ 0) → 
  0 ≤ a ∧ a ≤ 2 := by
  sorry

end inequality_constraint_on_a_l1677_167709


namespace base_8_to_10_fraction_l1677_167746

theorem base_8_to_10_fraction (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- c and d are base-10 digits
  (5 * 8^2 + 6 * 8 + 3 = 3 * 100 + c * 10 + d) →  -- 563_8 = 3cd_10
  (c * d) / 12 = 7 / 4 := by
sorry

end base_8_to_10_fraction_l1677_167746
