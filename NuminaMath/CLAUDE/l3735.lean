import Mathlib

namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l3735_373515

theorem baker_remaining_cakes (total_cakes sold_cakes : ℕ) 
  (h1 : total_cakes = 155)
  (h2 : sold_cakes = 140) :
  total_cakes - sold_cakes = 15 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l3735_373515


namespace NUMINAMATH_CALUDE_tourist_contact_probability_l3735_373533

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 42

theorem tourist_contact_probability (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  contact_probability p =
    1 - (1 - p) ^ (6 * 7) :=
by sorry

end NUMINAMATH_CALUDE_tourist_contact_probability_l3735_373533


namespace NUMINAMATH_CALUDE_log5_of_125_l3735_373501

-- Define the logarithm function for base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log5_of_125 : log5 125 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log5_of_125_l3735_373501


namespace NUMINAMATH_CALUDE_cuboid_diagonal_count_l3735_373556

/-- The number of unit cubes a diagonal passes through in a cuboid -/
def diagonalCubeCount (length width height : ℕ) : ℕ :=
  length + width + height - 2

/-- Theorem: The number of unit cubes a diagonal passes through in a 77 × 81 × 100 cuboid is 256 -/
theorem cuboid_diagonal_count :
  diagonalCubeCount 77 81 100 = 256 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_count_l3735_373556


namespace NUMINAMATH_CALUDE_probability_no_university_in_further_analysis_l3735_373587

/-- Represents the types of schools in the region -/
inductive SchoolType
  | Elementary
  | Middle
  | University

/-- Represents the total number of schools of each type -/
def totalSchools : SchoolType → Nat
  | SchoolType.Elementary => 21
  | SchoolType.Middle => 14
  | SchoolType.University => 7

/-- The total number of schools in the region -/
def totalAllSchools : Nat := 
  totalSchools SchoolType.Elementary + 
  totalSchools SchoolType.Middle + 
  totalSchools SchoolType.University

/-- The number of schools selected in the stratified sample -/
def sampleSize : Nat := 6

/-- The number of schools of each type in the stratified sample -/
def stratifiedSample : SchoolType → Nat
  | SchoolType.Elementary => 3
  | SchoolType.Middle => 2
  | SchoolType.University => 1

/-- The number of schools selected for further analysis -/
def furtherAnalysisSize : Nat := 2

theorem probability_no_university_in_further_analysis : 
  (Nat.choose (stratifiedSample SchoolType.Elementary + stratifiedSample SchoolType.Middle) furtherAnalysisSize : ℚ) / 
  (Nat.choose sampleSize furtherAnalysisSize : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_university_in_further_analysis_l3735_373587


namespace NUMINAMATH_CALUDE_graduating_class_boys_count_l3735_373574

theorem graduating_class_boys_count (total : ℕ) (difference : ℕ) (boys : ℕ) : 
  total = 345 →
  difference = 69 →
  total = boys + (boys + difference) →
  boys = 138 := by
sorry

end NUMINAMATH_CALUDE_graduating_class_boys_count_l3735_373574


namespace NUMINAMATH_CALUDE_sequence_strictly_decreasing_l3735_373572

/-- Given real numbers a and b with b > a > 1, prove that the sequence x_n is strictly monotonically decreasing -/
theorem sequence_strictly_decreasing (a b : ℝ) (h1 : a > 1) (h2 : b > a) : 
  ∀ n : ℕ, (2^n * (b^(1/2^n) - a^(1/2^n))) > (2^(n+1) * (b^(1/2^(n+1)) - a^(1/2^(n+1)))) := by
  sorry

#check sequence_strictly_decreasing

end NUMINAMATH_CALUDE_sequence_strictly_decreasing_l3735_373572


namespace NUMINAMATH_CALUDE_expression_upper_bound_l3735_373517

theorem expression_upper_bound (α β γ δ ε : ℝ) : 
  (1 - α) * Real.exp α + 
  (1 - β) * Real.exp (α + β) + 
  (1 - γ) * Real.exp (α + β + γ) + 
  (1 - δ) * Real.exp (α + β + γ + δ) + 
  (1 - ε) * Real.exp (α + β + γ + δ + ε) ≤ Real.exp 4 := by
  sorry

#check expression_upper_bound

end NUMINAMATH_CALUDE_expression_upper_bound_l3735_373517


namespace NUMINAMATH_CALUDE_angle_calculation_l3735_373550

-- Define a structure for angles in degrees, minutes, and seconds
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)
  (seconds : ℕ)

-- Define multiplication of an angle by a natural number
def multiply_angle (a : Angle) (n : ℕ) : Angle :=
  sorry

-- Define division of an angle by a natural number
def divide_angle (a : Angle) (n : ℕ) : Angle :=
  sorry

-- Define addition of two angles
def add_angles (a b : Angle) : Angle :=
  sorry

-- Theorem statement
theorem angle_calculation :
  let a1 := Angle.mk 50 24 0
  let a2 := Angle.mk 98 12 25
  add_angles (multiply_angle a1 3) (divide_angle a2 5) = Angle.mk 170 50 29 :=
sorry

end NUMINAMATH_CALUDE_angle_calculation_l3735_373550


namespace NUMINAMATH_CALUDE_smallest_harmonic_sum_exceeding_10_l3735_373581

def harmonic_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 1 / (i + 1 : ℚ))

theorem smallest_harmonic_sum_exceeding_10 :
  (∀ k < 12367, harmonic_sum k ≤ 10) ∧ harmonic_sum 12367 > 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_harmonic_sum_exceeding_10_l3735_373581


namespace NUMINAMATH_CALUDE_equation_solution_l3735_373590

/-- The solutions to the equation (8y^2 + 135y + 5) / (3y + 35) = 4y + 2 -/
theorem equation_solution : 
  let y₁ : ℂ := (-11 + Complex.I * Real.sqrt 919) / 8
  let y₂ : ℂ := (-11 - Complex.I * Real.sqrt 919) / 8
  ∀ y : ℂ, (8 * y^2 + 135 * y + 5) / (3 * y + 35) = 4 * y + 2 ↔ y = y₁ ∨ y = y₂ :=
by sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3735_373590


namespace NUMINAMATH_CALUDE_computational_not_basic_l3735_373524

/-- The set of basic algorithmic statements -/
def BasicAlgorithmicStatements : Set String :=
  {"assignment", "conditional", "loop", "input", "output"}

/-- Proposition: Computational statements are not basic algorithmic statements -/
theorem computational_not_basic : "computational" ∉ BasicAlgorithmicStatements := by
  sorry

end NUMINAMATH_CALUDE_computational_not_basic_l3735_373524


namespace NUMINAMATH_CALUDE_race_end_people_count_l3735_373594

/-- The number of people in cars at the end of a race -/
def people_at_race_end (num_cars : ℕ) (initial_people_per_car : ℕ) (additional_passengers : ℕ) : ℕ :=
  num_cars * (initial_people_per_car + additional_passengers)

/-- Theorem stating the number of people at the end of the race -/
theorem race_end_people_count : 
  people_at_race_end 20 3 1 = 80 := by sorry

end NUMINAMATH_CALUDE_race_end_people_count_l3735_373594


namespace NUMINAMATH_CALUDE_pony_jeans_discount_rate_l3735_373529

theorem pony_jeans_discount_rate 
  (fox_price : ℝ) 
  (pony_price : ℝ) 
  (total_savings : ℝ) 
  (fox_quantity : ℕ) 
  (pony_quantity : ℕ) 
  (total_discount_rate : ℝ) :
  fox_price = 15 →
  pony_price = 18 →
  total_savings = 8.55 →
  fox_quantity = 3 →
  pony_quantity = 2 →
  total_discount_rate = 22 →
  ∃ (fox_discount_rate : ℝ) (pony_discount_rate : ℝ),
    fox_discount_rate + pony_discount_rate = total_discount_rate ∧
    fox_quantity * (fox_price * fox_discount_rate / 100) + 
    pony_quantity * (pony_price * pony_discount_rate / 100) = total_savings ∧
    pony_discount_rate = 15 :=
by sorry

end NUMINAMATH_CALUDE_pony_jeans_discount_rate_l3735_373529


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3735_373576

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (1 / a + 4 / b) ≥ 9 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a + b) * (1 / a + 4 / b) < 9 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3735_373576


namespace NUMINAMATH_CALUDE_max_magnitude_c_l3735_373596

open Real

/-- Given vectors a and b, and a vector c satisfying the dot product condition,
    prove that the maximum magnitude of c is √2. -/
theorem max_magnitude_c (a b c : ℝ × ℝ) : 
  a = (1, 0) → 
  b = (0, 1) → 
  (c.1 + a.1, c.2 + a.2) • (c.1 + b.1, c.2 + b.2) = 0 → 
  (∀ c' : ℝ × ℝ, (c'.1 + a.1, c'.2 + a.2) • (c'.1 + b.1, c'.2 + b.2) = 0 → 
    Real.sqrt (c.1^2 + c.2^2) ≥ Real.sqrt (c'.1^2 + c'.2^2)) → 
  Real.sqrt (c.1^2 + c.2^2) = sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_max_magnitude_c_l3735_373596


namespace NUMINAMATH_CALUDE_boat_round_trip_time_l3735_373503

theorem boat_round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 9)
  (h2 : stream_speed = 6)
  (h3 : distance = 170)
  : (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = 68 := by
  sorry

end NUMINAMATH_CALUDE_boat_round_trip_time_l3735_373503


namespace NUMINAMATH_CALUDE_simplify_expression_l3735_373514

theorem simplify_expression (x : ℝ) : x + 3 - 5*x + 6 + 7*x - 2 - 9*x + 8 = -6*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3735_373514


namespace NUMINAMATH_CALUDE_solve_system_l3735_373553

theorem solve_system (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h1 : 1/x + 1/y = 3/2) (h2 : x*y = 9) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3735_373553


namespace NUMINAMATH_CALUDE_original_plums_count_l3735_373545

theorem original_plums_count (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 4 → total = 21 → initial + added = total → initial = 17 := by
sorry

end NUMINAMATH_CALUDE_original_plums_count_l3735_373545


namespace NUMINAMATH_CALUDE_f_of_g_10_l3735_373583

-- Define the functions g and f
def g (x : ℝ) : ℝ := 4 * x + 5
def f (x : ℝ) : ℝ := 6 * x - 8

-- State the theorem
theorem f_of_g_10 : f (g 10) = 262 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_10_l3735_373583


namespace NUMINAMATH_CALUDE_max_value_of_f_l3735_373546

def f (x : ℝ) := x^2 + 2*x

theorem max_value_of_f :
  ∃ (M : ℝ), M = 8 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3735_373546


namespace NUMINAMATH_CALUDE_min_value_theorem_l3735_373561

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) 
  (heq : a^2 * (b + 4*b^2 + 2*a^2) = 8 - 2*b^3) :
  ∃ (m : ℝ), m = 8 * Real.sqrt 3 ∧ 
  (∀ (x y : ℝ), x > 0 → y > 1 → x^2 * (y + 4*y^2 + 2*x^2) = 8 - 2*y^3 → 
    8*x^2 + 4*y^2 + 3*y ≥ m) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 1 ∧ x^2 * (y + 4*y^2 + 2*x^2) = 8 - 2*y^3 ∧ 
    8*x^2 + 4*y^2 + 3*y = m) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3735_373561


namespace NUMINAMATH_CALUDE_remainder_of_M_mod_50_l3735_373528

def M : ℕ := sorry -- Definition of M as concatenation of numbers from 1 to 49

theorem remainder_of_M_mod_50 : M % 50 = 49 := by sorry

end NUMINAMATH_CALUDE_remainder_of_M_mod_50_l3735_373528


namespace NUMINAMATH_CALUDE_sum_of_roots_l3735_373557

theorem sum_of_roots (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ + 2 = 0) → (x₂^2 - 3*x₂ + 2 = 0) → x₁ + x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3735_373557


namespace NUMINAMATH_CALUDE_inequality_solution_l3735_373593

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x + 2) ≥ 1 / (x - 2) + 3 / 4) ↔ (x > -2 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3735_373593


namespace NUMINAMATH_CALUDE_work_completion_time_l3735_373508

/-- Given that A can do a work in 12 days and A and B together can do the work in 8 days,
    prove that B can do the work alone in 24 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 12) (hab : 1 / a + 1 / b = 1 / 8) : b = 24 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3735_373508


namespace NUMINAMATH_CALUDE_cos_difference_of_zeros_l3735_373507

open Real

theorem cos_difference_of_zeros (f g : ℝ → ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, f x = sin (2 * x - π / 3)) →
  (∀ x, g x = f x - 1 / 3) →
  g x₁ = 0 →
  g x₂ = 0 →
  x₁ ≠ x₂ →
  0 ≤ x₁ ∧ x₁ ≤ π →
  0 ≤ x₂ ∧ x₂ ≤ π →
  cos (x₁ - x₂) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_of_zeros_l3735_373507


namespace NUMINAMATH_CALUDE_percent_greater_l3735_373558

theorem percent_greater (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x = 1.2 * y) (hyz : y = 1.2 * z) (hwx : w = 0.8 * x) :
  w = 1.152 * z := by
  sorry

end NUMINAMATH_CALUDE_percent_greater_l3735_373558


namespace NUMINAMATH_CALUDE_diagonals_not_bisect_equiv_not_p_l3735_373511

-- Define the proposition "The diagonals of a trapezoid bisect each other"
def diagonals_bisect_each_other : Prop := sorry

-- Define the proposition "The diagonals of a trapezoid do not bisect each other"
def diagonals_do_not_bisect_each_other : Prop := ¬diagonals_bisect_each_other

-- Theorem stating that the given proposition is equivalent to "not p"
theorem diagonals_not_bisect_equiv_not_p : 
  diagonals_do_not_bisect_each_other ↔ ¬diagonals_bisect_each_other :=
sorry

end NUMINAMATH_CALUDE_diagonals_not_bisect_equiv_not_p_l3735_373511


namespace NUMINAMATH_CALUDE_equation_solution_l3735_373500

theorem equation_solution :
  ∃ x : ℝ, -2 * (x - 1) = 4 ∧ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3735_373500


namespace NUMINAMATH_CALUDE_complex_math_expression_equals_35_l3735_373530

theorem complex_math_expression_equals_35 :
  ((9^2 + (3^3 - 1) * 4^2) % 6 : ℕ) * Real.sqrt 49 + (15 - 3 * 5) = 35 := by
  sorry

end NUMINAMATH_CALUDE_complex_math_expression_equals_35_l3735_373530


namespace NUMINAMATH_CALUDE_trig_problem_l3735_373520

theorem trig_problem (α : Real) (h : Real.tan α = 2) : 
  (2 * Real.sin α ^ 2 + 1) / Real.cos (2 * (α - π/4)) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l3735_373520


namespace NUMINAMATH_CALUDE_factorization_equality_l3735_373538

theorem factorization_equality (x y : ℝ) : x + x^2 - y - y^2 = (x + y + 1) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3735_373538


namespace NUMINAMATH_CALUDE_circle_properties_l3735_373510

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 6*y - 3 = 0

/-- The center of the circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- The radius of the circle -/
def CircleRadius : ℝ := 4

/-- Theorem stating that the given equation represents a circle with the specified center and radius -/
theorem circle_properties :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = CircleRadius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3735_373510


namespace NUMINAMATH_CALUDE_power_function_theorem_l3735_373551

theorem power_function_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^a) →  -- f is a power function with exponent a
  f 2 = 1/4 →         -- f passes through the point (2, 1/4)
  f (-2) = 1/4 :=     -- prove that f(-2) = 1/4
by
  sorry

end NUMINAMATH_CALUDE_power_function_theorem_l3735_373551


namespace NUMINAMATH_CALUDE_exists_circle_with_n_points_l3735_373504

/-- A function that counts the number of lattice points strictly inside a circle -/
def count_lattice_points (center : ℝ × ℝ) (radius : ℝ) : ℕ :=
  sorry

/-- Theorem stating that for any non-negative integer, there exists a circle containing exactly that many lattice points -/
theorem exists_circle_with_n_points (n : ℕ) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), count_lattice_points center radius = n :=
sorry

end NUMINAMATH_CALUDE_exists_circle_with_n_points_l3735_373504


namespace NUMINAMATH_CALUDE_age_problem_solution_l3735_373563

/-- Represents the problem of finding when Anand's age was one-third of Bala's age -/
def age_problem (x : ℕ) : Prop :=
  let anand_current_age : ℕ := 15
  let bala_current_age : ℕ := anand_current_age + 10
  let anand_past_age : ℕ := anand_current_age - x
  let bala_past_age : ℕ := bala_current_age - x
  anand_past_age = bala_past_age / 3

/-- Theorem stating that 10 years ago, Anand's age was one-third of Bala's age -/
theorem age_problem_solution : age_problem 10 := by
  sorry

#check age_problem_solution

end NUMINAMATH_CALUDE_age_problem_solution_l3735_373563


namespace NUMINAMATH_CALUDE_quadratic_solution_l3735_373521

theorem quadratic_solution (x : ℝ) : x^2 - 6*x + 8 = 0 → x = 2 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3735_373521


namespace NUMINAMATH_CALUDE_min_value_problem_l3735_373564

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (8^a * 2^b)) :
  1/a + 2/b ≥ 5 + 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3735_373564


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3735_373512

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the sum of the first and ninth terms is 10,
    the fifth term is equal to 5. -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3735_373512


namespace NUMINAMATH_CALUDE_expression_simplification_l3735_373566

theorem expression_simplification (a b : ℝ) (ha : a = -1) (hb : b = 2) :
  (a + b)^2 + (a^2 * b - 2 * a * b^2 - b^3) / b - (a - b) * (a + b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3735_373566


namespace NUMINAMATH_CALUDE_ping_pong_theorem_l3735_373560

/-- Represents the number of ping-pong balls in the box -/
def total_balls : ℕ := 7

/-- Represents the number of unused balls initially -/
def initial_unused : ℕ := 5

/-- Represents the number of used balls initially -/
def initial_used : ℕ := 2

/-- Represents the number of balls taken out and used -/
def balls_taken : ℕ := 3

/-- Represents the set of possible values for X (number of used balls after the process) -/
def possible_X : Set ℕ := {3, 4, 5}

/-- Represents the probability of X being 3 -/
def prob_X_3 : ℚ := 1/7

theorem ping_pong_theorem :
  (∀ x : ℕ, x ∈ possible_X ↔ (x ≥ initial_used ∧ x ≤ initial_used + balls_taken)) ∧
  (Nat.choose initial_unused 1 * Nat.choose initial_used 2 : ℚ) / Nat.choose total_balls balls_taken = prob_X_3 :=
by sorry

end NUMINAMATH_CALUDE_ping_pong_theorem_l3735_373560


namespace NUMINAMATH_CALUDE_special_ellipse_property_l3735_373544

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  center : ℝ × ℝ := (0, 0)
  focus_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ
  ecc_eq : eccentricity = Real.sqrt (6/3)
  point_eq : passes_through = (Real.sqrt 5, 0)

/-- Line intersecting the ellipse -/
structure IntersectingLine where
  fixed_point : ℝ × ℝ
  point_eq : fixed_point = (-1, 0)

/-- Intersection points of the line with the ellipse -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  midpoint_x : ℝ
  mid_eq : midpoint_x = -1/2

/-- The theorem statement -/
theorem special_ellipse_property
  (e : SpecialEllipse) (l : IntersectingLine) (p : IntersectionPoints) :
  ∃ (M : ℝ × ℝ), M.1 = -7/3 ∧ M.2 = 0 ∧
  (∀ (A B : ℝ × ℝ), 
    ((A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2)) = 4/9) :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_property_l3735_373544


namespace NUMINAMATH_CALUDE_cost_price_percentage_l3735_373586

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Define the gain percent after discount
def gain_percent : ℝ := 0.171875

-- Define the relationship between cost price and marked price
theorem cost_price_percentage (marked_price cost_price : ℝ) 
  (h1 : marked_price > 0) 
  (h2 : cost_price > 0) 
  (h3 : marked_price * (1 - discount_rate) = cost_price * (1 + gain_percent)) : 
  cost_price / marked_price = 0.64 := by
  sorry


end NUMINAMATH_CALUDE_cost_price_percentage_l3735_373586


namespace NUMINAMATH_CALUDE_subtract_two_percent_l3735_373518

theorem subtract_two_percent (a : ℝ) : a - (0.02 * a) = 0.98 * a := by
  sorry

end NUMINAMATH_CALUDE_subtract_two_percent_l3735_373518


namespace NUMINAMATH_CALUDE_symmetric_complex_quotient_l3735_373540

/-- Two complex numbers are symmetric about the y-axis if their real parts are negatives of each other and their imaginary parts are equal -/
def symmetric_about_y_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem symmetric_complex_quotient (z₁ z₂ : ℂ) :
  symmetric_about_y_axis z₁ z₂ → z₁ = 1 + I → z₂ / z₁ = I :=
by
  sorry

#check symmetric_complex_quotient

end NUMINAMATH_CALUDE_symmetric_complex_quotient_l3735_373540


namespace NUMINAMATH_CALUDE_existence_of_index_l3735_373580

theorem existence_of_index (n : ℕ) (x : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_x : ∀ i, i ∈ Finset.range (n + 1) → 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i ∈ Finset.range n, x 1 * (1 - x (i + 1)) ≥ (1 / 4) * x 1 * (1 - x n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_index_l3735_373580


namespace NUMINAMATH_CALUDE_trains_crossing_time_l3735_373573

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 300)
  (h2 : length2 = 400)
  (h3 : speed1 = 36 * 1000 / 3600)
  (h4 : speed2 = 18 * 1000 / 3600)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0) :
  (length1 + length2) / (speed1 + speed2) = 46.67 := by
  sorry

end NUMINAMATH_CALUDE_trains_crossing_time_l3735_373573


namespace NUMINAMATH_CALUDE_class_size_proof_l3735_373571

theorem class_size_proof :
  ∀ n : ℕ,
  20 < n ∧ n < 30 →
  ∃ x : ℕ, n = 3 * x →
  ∃ y : ℕ, n = 4 * y →
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_class_size_proof_l3735_373571


namespace NUMINAMATH_CALUDE_power_of_32_equals_power_of_2_l3735_373567

theorem power_of_32_equals_power_of_2 : ∀ q : ℕ, 32^5 = 2^q → q = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_of_32_equals_power_of_2_l3735_373567


namespace NUMINAMATH_CALUDE_newton_county_population_l3735_373506

theorem newton_county_population (num_cities : ℕ) (lower_bound upper_bound : ℝ) :
  num_cities = 20 →
  lower_bound = 4500 →
  upper_bound = 5000 →
  let avg_population := (lower_bound + upper_bound) / 2
  num_cities * avg_population = 95000 := by
  sorry

end NUMINAMATH_CALUDE_newton_county_population_l3735_373506


namespace NUMINAMATH_CALUDE_f_of_2_equals_3_l3735_373502

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - 1

-- State the theorem
theorem f_of_2_equals_3 : f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_f_of_2_equals_3_l3735_373502


namespace NUMINAMATH_CALUDE_bookshelf_selection_l3735_373536

theorem bookshelf_selection (math_books : ℕ) (chinese_books : ℕ) (english_books : ℕ) 
  (h1 : math_books = 3) (h2 : chinese_books = 5) (h3 : english_books = 8) :
  math_books + chinese_books + english_books = 16 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_selection_l3735_373536


namespace NUMINAMATH_CALUDE_exponential_inequality_l3735_373598

theorem exponential_inequality (m n : ℝ) (a b : ℝ) 
  (h1 : a = (0.2 : ℝ) ^ m) 
  (h2 : b = (0.2 : ℝ) ^ n) 
  (h3 : m > n) : 
  a < b := by
sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3735_373598


namespace NUMINAMATH_CALUDE_contest_probability_l3735_373565

theorem contest_probability (n : ℕ) : n = 4 ↔ n = Nat.succ (Nat.floor (Real.log 10 / Real.log 2)) := by sorry

end NUMINAMATH_CALUDE_contest_probability_l3735_373565


namespace NUMINAMATH_CALUDE_parallel_line_implies_a_value_l3735_373597

/-- Two points are on a line parallel to the y-axis if their x-coordinates are equal -/
def parallel_to_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = x₂

/-- The theorem stating that if M(a-3, a+4) and N(5, 9) form a line segment
    parallel to the y-axis, then a = 8 -/
theorem parallel_line_implies_a_value :
  ∀ a : ℝ,
  parallel_to_y_axis (a - 3) (a + 4) 5 9 →
  a = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_line_implies_a_value_l3735_373597


namespace NUMINAMATH_CALUDE_set_equality_implies_a_values_l3735_373584

theorem set_equality_implies_a_values (a : ℝ) : 
  let A : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}
  let B : Set ℝ := {y | ∃ x ∈ A, y = x + 1}
  let C : Set ℝ := {y | ∃ x ∈ A, y = x^2}
  A.Nonempty → B = C → a = 0 ∨ a = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_values_l3735_373584


namespace NUMINAMATH_CALUDE_train_tunnel_time_l3735_373539

/-- Proves that a train of given length and speed passing through a tunnel of given length takes 1 minute to completely clear the tunnel. -/
theorem train_tunnel_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length_km : ℝ) :
  train_length = 100 →
  train_speed_kmh = 72 →
  tunnel_length_km = 1.1 →
  (tunnel_length_km * 1000 + train_length) / (train_speed_kmh * 1000 / 60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_train_tunnel_time_l3735_373539


namespace NUMINAMATH_CALUDE_wall_length_calculation_l3735_373575

/-- Given a square mirror and a rectangular wall, if the mirror's area is half the wall's area,
    prove the length of the wall. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 18 →
  wall_width = 32 →
  (mirror_side ^ 2) * 2 = wall_width * (20.25 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l3735_373575


namespace NUMINAMATH_CALUDE_jamie_tax_payment_l3735_373555

/-- Calculates the tax amount based on a progressive tax system --/
def calculate_tax (gross_income : ℕ) (deduction : ℕ) : ℕ :=
  let taxable_income := gross_income - deduction
  let first_bracket := min taxable_income 150
  let second_bracket := min (taxable_income - 150) 150
  let third_bracket := max (taxable_income - 300) 0
  0 * first_bracket + 
  (10 * second_bracket) / 100 + 
  (15 * third_bracket) / 100

/-- Theorem stating that Jamie's tax payment is $30 --/
theorem jamie_tax_payment : 
  calculate_tax 450 50 = 30 := by
  sorry

#eval calculate_tax 450 50  -- This should output 30

end NUMINAMATH_CALUDE_jamie_tax_payment_l3735_373555


namespace NUMINAMATH_CALUDE_hiker_distance_l3735_373582

/-- Given a hiker's movements, calculate the final distance from the starting point -/
theorem hiker_distance (north east south west : ℝ) :
  north = 15 ∧ east = 8 ∧ south = 3 ∧ west = 4 →
  Real.sqrt ((north - south)^2 + (east - west)^2) = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_l3735_373582


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_sixteen_l3735_373532

theorem sum_of_solutions_eq_sixteen : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 8)^2 = 36 ∧ (x₂ - 8)^2 = 36 ∧ x₁ + x₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_sixteen_l3735_373532


namespace NUMINAMATH_CALUDE_fred_change_theorem_l3735_373568

/-- Calculates the change received after a purchase -/
def calculate_change (ticket_price : ℚ) (num_tickets : ℕ) (borrowed_movie_price : ℚ) (paid_amount : ℚ) : ℚ :=
  paid_amount - (ticket_price * num_tickets + borrowed_movie_price)

theorem fred_change_theorem :
  let ticket_price : ℚ := 8.25
  let num_tickets : ℕ := 4
  let borrowed_movie_price : ℚ := 9.50
  let paid_amount : ℚ := 50
  calculate_change ticket_price num_tickets borrowed_movie_price paid_amount = 7.50 := by
  sorry

#eval calculate_change 8.25 4 9.50 50

end NUMINAMATH_CALUDE_fred_change_theorem_l3735_373568


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l3735_373591

/-- Given an elevator scenario, prove the initial average weight --/
theorem elevator_weight_problem (initial_people : ℕ) (new_person_weight : ℕ) (new_average : ℕ) :
  initial_people = 6 →
  new_person_weight = 145 →
  new_average = 151 →
  (initial_people * (initial_average : ℕ) + new_person_weight) / (initial_people + 1) = new_average →
  initial_average = 152 :=
by
  sorry

#check elevator_weight_problem

end NUMINAMATH_CALUDE_elevator_weight_problem_l3735_373591


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3735_373522

/-- The angle of inclination of a line passing through (0, 0) and (1, -1) is 135°. -/
theorem line_inclination_angle : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (t, -t)}
  let angle : ℝ := Real.arctan (-1) * (180 / Real.pi)
  angle = 135 := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3735_373522


namespace NUMINAMATH_CALUDE_percentage_of_200_to_50_percentage_proof_l3735_373589

theorem percentage_of_200_to_50 : ℝ → Prop :=
  fun x => (200 / 50) * 100 = x ∧ x = 400

-- The proof would go here
theorem percentage_proof : percentage_of_200_to_50 400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_200_to_50_percentage_proof_l3735_373589


namespace NUMINAMATH_CALUDE_increase_percentage_theorem_l3735_373531

theorem increase_percentage_theorem (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hM : M > 0) (hpq : q < p) :
  M * (1 + p / 100) * (1 + q / 100) > M ↔ (p > 0 ∧ q > 0) :=
by sorry

end NUMINAMATH_CALUDE_increase_percentage_theorem_l3735_373531


namespace NUMINAMATH_CALUDE_function_value_at_pi_over_12_l3735_373547

theorem function_value_at_pi_over_12 (x : Real) (h : x = π / 12) :
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2) - 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_pi_over_12_l3735_373547


namespace NUMINAMATH_CALUDE_derivative_at_negative_one_l3735_373509

/-- Given f(x) = (1/3)x³ + 2x + 1, prove that f'(-1) = 3 -/
theorem derivative_at_negative_one (f : ℝ → ℝ) (hf : ∀ x, f x = (1/3) * x^3 + 2*x + 1) :
  (deriv f) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_one_l3735_373509


namespace NUMINAMATH_CALUDE_ellipse_problem_l3735_373526

-- Define the ellipses and points
def C₁ (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1
def C₂ (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
theorem ellipse_problem (a b : ℝ) (A B H P M N : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (∃ x y, C₂ a b x y ∧ x^2 = 5 ∧ y = 0) ∧
  (∃ x₁ y₁ x₂ y₂, C₂ a b x₁ y₁ ∧ C₂ a b x₂ y₂ ∧ y₂ - y₁ = x₂ - x₁) ∧
  H = (2, -1) ∧
  C₂ a b P.1 P.2 ∧
  C₁ M.1 M.2 ∧
  C₁ N.1 N.2 ∧
  P.1 = M.1 + 2 * N.1 ∧
  P.2 = M.2 + 2 * N.2 →
  (a^2 = 10 ∧ b^2 = 5) ∧
  (M.2 / M.1 * N.2 / N.1 = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_problem_l3735_373526


namespace NUMINAMATH_CALUDE_factorization_proof_l3735_373537

theorem factorization_proof (m n : ℝ) : 12 * m^2 * n - 12 * m * n + 3 * n = 3 * n * (2 * m - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3735_373537


namespace NUMINAMATH_CALUDE_soda_problem_l3735_373505

theorem soda_problem (S : ℝ) : 
  (S / 2 + 2000 = S - (S / 2 - 2000)) → 
  ((S / 2 - 2000) / 2 + 2000 = S / 2 - 2000) → 
  S = 12000 := by
  sorry

end NUMINAMATH_CALUDE_soda_problem_l3735_373505


namespace NUMINAMATH_CALUDE_sarahs_trip_length_l3735_373554

theorem sarahs_trip_length :
  ∀ (x : ℝ),
  (x / 4 : ℝ) + 15 + (x / 3 : ℝ) = x →
  x = 36 := by
sorry

end NUMINAMATH_CALUDE_sarahs_trip_length_l3735_373554


namespace NUMINAMATH_CALUDE_smallest_lattice_triangle_area_is_half_l3735_373535

/-- A lattice triangle is a triangle on a square grid where all vertices are grid points. -/
structure LatticeTriangle where
  vertices : Fin 3 → ℤ × ℤ

/-- The area of a grid square is 1 square unit. -/
def grid_square_area : ℝ := 1

/-- The area of a lattice triangle -/
def lattice_triangle_area (t : LatticeTriangle) : ℝ := sorry

/-- The smallest possible area of a lattice triangle -/
def smallest_lattice_triangle_area : ℝ := sorry

/-- Theorem: The area of the smallest lattice triangle is 1/2 square unit -/
theorem smallest_lattice_triangle_area_is_half :
  smallest_lattice_triangle_area = 1/2 := by sorry

end NUMINAMATH_CALUDE_smallest_lattice_triangle_area_is_half_l3735_373535


namespace NUMINAMATH_CALUDE_factor_x4_plus_16_l3735_373525

theorem factor_x4_plus_16 (x : ℝ) : x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_x4_plus_16_l3735_373525


namespace NUMINAMATH_CALUDE_train_crossing_time_l3735_373595

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (signal_pole_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 300) 
  (h2 : signal_pole_time = 18) 
  (h3 : platform_length = 600.0000000000001) : 
  (train_length + platform_length) / (train_length / signal_pole_time) = 54.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3735_373595


namespace NUMINAMATH_CALUDE_h_range_l3735_373599

-- Define the function h
def h (x : ℝ) : ℝ := 3 * (x - 5)

-- State the theorem
theorem h_range :
  {y : ℝ | ∃ x : ℝ, x ≠ -9 ∧ h x = y} = {y : ℝ | y < -42 ∨ y > -42} :=
by sorry

end NUMINAMATH_CALUDE_h_range_l3735_373599


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3735_373588

theorem trig_identity_proof (α : Real) (h : Real.tan α = 3) : 
  (Real.cos (α + π/4))^2 - (Real.cos (α - π/4))^2 = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3735_373588


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3735_373513

theorem cube_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a * b + a * c + b * c = 3)
  (h3 : a * b * c = 5) :
  a^3 + b^3 + c^3 = 15 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3735_373513


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_four_is_solution_four_is_smallest_l3735_373548

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 3 → (∃ n : ℕ, 3 * b + 4 = n^2) → b ≥ 4 :=
by sorry

theorem four_is_solution : 
  ∃ n : ℕ, 3 * 4 + 4 = n^2 :=
by sorry

theorem four_is_smallest : 
  ∀ b : ℕ, b > 3 ∧ (∃ n : ℕ, 3 * b + 4 = n^2) → b = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_four_is_solution_four_is_smallest_l3735_373548


namespace NUMINAMATH_CALUDE_power_eleven_mod_hundred_l3735_373516

theorem power_eleven_mod_hundred : 11^2023 % 100 = 31 := by
  sorry

end NUMINAMATH_CALUDE_power_eleven_mod_hundred_l3735_373516


namespace NUMINAMATH_CALUDE_interval_intersection_l3735_373519

-- Define the conditions
def condition1 (x : ℝ) : Prop := 2 < 5 * x ∧ 5 * x < 3
def condition2 (x : ℝ) : Prop := 4 < 7 * x ∧ 7 * x < 6

-- Define the theorem
theorem interval_intersection :
  ∀ x : ℝ, (condition1 x ∧ condition2 x) ↔ (4/7 < x ∧ x < 3/5) :=
sorry

end NUMINAMATH_CALUDE_interval_intersection_l3735_373519


namespace NUMINAMATH_CALUDE_prove_z_value_l3735_373534

theorem prove_z_value (z : ℝ) : 
  (Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt z / Real.sqrt 0.49 = 2.650793650793651) → 
  z = 1.00 := by
sorry

end NUMINAMATH_CALUDE_prove_z_value_l3735_373534


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3735_373570

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x > 2, x^3 - 8 > 0) ↔ (∃ x > 2, x^3 - 8 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3735_373570


namespace NUMINAMATH_CALUDE_specific_parallelepiped_volume_l3735_373541

/-- A right parallelepiped with a parallelogram base -/
structure RightParallelepiped where
  /-- Length of one side of the base -/
  sideA : ℝ
  /-- Length of the other side of the base -/
  sideB : ℝ
  /-- Angle between the sides of the base in radians -/
  baseAngle : ℝ
  /-- The smaller diagonal of the parallelepiped -/
  smallerDiagonal : ℝ

/-- The volume of the right parallelepiped -/
def volume (p : RightParallelepiped) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific parallelepiped -/
theorem specific_parallelepiped_volume :
  ∃ (p : RightParallelepiped),
    p.sideA = 3 ∧
    p.sideB = 4 ∧
    p.baseAngle = 2 * π / 3 ∧
    p.smallerDiagonal = Real.sqrt (p.sideA ^ 2 + p.sideB ^ 2 - 2 * p.sideA * p.sideB * Real.cos p.baseAngle) ∧
    volume p = 36 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_specific_parallelepiped_volume_l3735_373541


namespace NUMINAMATH_CALUDE_range_of_f_range_of_m_l3735_373523

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

-- Theorem for the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-3) 3 :=
sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x, f x + 2 * m - 1 ≥ 0) ↔ m ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_m_l3735_373523


namespace NUMINAMATH_CALUDE_weekly_rainfall_sum_l3735_373559

def monday_rainfall : ℝ := 0.12962962962962962
def tuesday_rainfall : ℝ := 0.35185185185185186
def wednesday_rainfall : ℝ := 0.09259259259259259
def thursday_rainfall : ℝ := 0.25925925925925924
def friday_rainfall : ℝ := 0.48148148148148145
def saturday_rainfall : ℝ := 0.2222222222222222
def sunday_rainfall : ℝ := 0.4444444444444444

theorem weekly_rainfall_sum :
  monday_rainfall + tuesday_rainfall + wednesday_rainfall + thursday_rainfall +
  friday_rainfall + saturday_rainfall + sunday_rainfall = 1.9814814814814815 := by
  sorry

end NUMINAMATH_CALUDE_weekly_rainfall_sum_l3735_373559


namespace NUMINAMATH_CALUDE_discount_percentages_l3735_373579

/-- Merchant's markup percentage -/
def markup : ℚ := 75 / 100

/-- Profit percentage for 65 items -/
def profit65 : ℚ := 575 / 1000

/-- Profit percentage for 30 items -/
def profit30 : ℚ := 525 / 1000

/-- Profit percentage for 5 items -/
def profit5 : ℚ := 48 / 100

/-- Calculate discount percentage given profit percentage -/
def calcDiscount (profit : ℚ) : ℚ :=
  (markup - profit) / (1 + markup) * 100

/-- Round to nearest integer -/
def roundToInt (q : ℚ) : ℤ :=
  (q + 1/2).floor

/-- Theorem stating the discount percentages -/
theorem discount_percentages :
  let x := roundToInt (calcDiscount profit5)
  let y := roundToInt (calcDiscount profit30)
  let z := roundToInt (calcDiscount profit65)
  x = 15 ∧ y = 13 ∧ z = 10 ∧
  (5 ≤ x ∧ x ≤ 25) ∧ (5 ≤ y ∧ y ≤ 25) ∧ (5 ≤ z ∧ z ≤ 25) :=
by sorry


end NUMINAMATH_CALUDE_discount_percentages_l3735_373579


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_parabola_standard_equation_l3735_373562

-- Ellipse
def ellipse_equation (x y : ℝ) := x^2 / 25 + y^2 / 9 = 1

theorem ellipse_standard_equation 
  (foci_on_x_axis : Bool) 
  (major_axis_length : ℝ) 
  (eccentricity : ℝ) :
  foci_on_x_axis ∧ 
  major_axis_length = 10 ∧ 
  eccentricity = 4/5 →
  ∀ x y : ℝ, ellipse_equation x y :=
sorry

-- Parabola
def parabola_equation (x y : ℝ) := x^2 = -8*y

theorem parabola_standard_equation 
  (vertex : ℝ × ℝ) 
  (directrix : ℝ → ℝ) :
  vertex = (0, 0) ∧ 
  (∀ x : ℝ, directrix x = 2) →
  ∀ x y : ℝ, parabola_equation x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_parabola_standard_equation_l3735_373562


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3735_373578

theorem complex_equation_solution (z : ℂ) :
  (1 + Complex.I) * z = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3735_373578


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3735_373577

def isGeometricSequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

theorem geometric_sequence_product (a b : ℝ) :
  isGeometricSequence 2 a b 16 → a * b = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3735_373577


namespace NUMINAMATH_CALUDE_sequence_sum_l3735_373549

/-- Given a sequence {a_n} where the sum of its first n terms S_n = n^2,
    and a sequence {b_n} defined as b_n = 2^(a_n),
    prove that the sum of the first n terms of {b_n}, T_n, is (2/3) * (4^n - 1) -/
theorem sequence_sum (n : ℕ) (a b : ℕ → ℕ) (S T : ℕ → ℚ)
  (h_S : ∀ k, S k = k^2)
  (h_b : ∀ k, b k = 2^(a k)) :
  T n = 2/3 * (4^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3735_373549


namespace NUMINAMATH_CALUDE_robert_claire_photo_difference_l3735_373592

/-- 
Given that:
- Lisa and Robert have taken the same number of photos
- Lisa has taken 3 times as many photos as Claire
- Claire has taken 6 photos

Prove that Robert has taken 12 more photos than Claire.
-/
theorem robert_claire_photo_difference : 
  ∀ (lisa robert claire : ℕ),
  robert = lisa →
  lisa = 3 * claire →
  claire = 6 →
  robert - claire = 12 :=
by sorry

end NUMINAMATH_CALUDE_robert_claire_photo_difference_l3735_373592


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_range_l3735_373527

/-- A triangle with sides a, a+2, and a+4 is obtuse if and only if 2 < a < 6 -/
theorem obtuse_triangle_side_range (a : ℝ) : 
  (∃ (x y z : ℝ), x = a ∧ y = a + 2 ∧ z = a + 4 ∧ 
   x > 0 ∧ y > 0 ∧ z > 0 ∧
   x + y > z ∧ x + z > y ∧ y + z > x ∧
   z^2 > x^2 + y^2) ↔ 
  (2 < a ∧ a < 6) :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_range_l3735_373527


namespace NUMINAMATH_CALUDE_functional_equation_properties_l3735_373569

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (f 0 = 0) ∧ (f 1 = 0) ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l3735_373569


namespace NUMINAMATH_CALUDE_cleaning_time_ratio_with_help_cleaning_time_ratio_l3735_373542

/-- Represents the grove of trees -/
structure Grove where
  rows : Nat
  columns : Nat

/-- Represents the time spent cleaning trees -/
structure CleaningTime where
  minutes : Nat

theorem cleaning_time_ratio_with_help (g : Grove) 
  (time_per_tree_without_help : Nat) 
  (total_time_with_help : CleaningTime) : 
  2 * (total_time_with_help.minutes / (g.rows * g.columns)) = time_per_tree_without_help :=
by
  sorry

#check cleaning_time_ratio_with_help

/-- Main theorem that proves the ratio of cleaning time with help to without help is 1:2 -/
theorem cleaning_time_ratio (g : Grove) 
  (time_per_tree_without_help : Nat) 
  (total_time_with_help : CleaningTime) : 
  (total_time_with_help.minutes / (g.rows * g.columns)) / time_per_tree_without_help = 1 / 2 :=
by
  sorry

#check cleaning_time_ratio

end NUMINAMATH_CALUDE_cleaning_time_ratio_with_help_cleaning_time_ratio_l3735_373542


namespace NUMINAMATH_CALUDE_divisibility_property_l3735_373585

theorem divisibility_property (a b : ℕ+) 
  (h : ∀ k : ℕ+, k < b → (b + k) ∣ (a + k)) :
  ∀ k : ℕ+, k < b → (b - k) ∣ (a - k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3735_373585


namespace NUMINAMATH_CALUDE_decreasing_quadratic_function_m_range_l3735_373543

/-- A function f(x) = mx^2 + (m-1)x + 1 is decreasing on (-∞, 1] if and only if m ∈ [0, 1/3] -/
theorem decreasing_quadratic_function_m_range (m : ℝ) : 
  (∀ x ≤ 1, ∀ y ≤ 1, x < y → m * x^2 + (m - 1) * x + 1 > m * y^2 + (m - 1) * y + 1) ↔ 
  0 ≤ m ∧ m ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_function_m_range_l3735_373543


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l3735_373552

theorem least_positive_integer_with_remainder_one (n : ℕ) : n = 421 ↔ 
  (n > 1) ∧ 
  (∀ d ∈ ({3, 4, 5, 6, 7, 10, 12} : Set ℕ), n % d = 1) ∧
  (∀ m : ℕ, m > 1 → (∀ d ∈ ({3, 4, 5, 6, 7, 10, 12} : Set ℕ), m % d = 1) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l3735_373552
