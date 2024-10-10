import Mathlib

namespace square_octagon_exterior_angle_l250_25013

/-- The measure of an interior angle of a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ := 180 * (n - 2) / n

/-- The configuration of a square and regular octagon sharing a side -/
structure SquareOctagonConfig where
  square_angle : ℚ  -- Interior angle of the square
  octagon_angle : ℚ -- Interior angle of the octagon
  common_side : ℚ   -- Length of the common side (not used in this problem, but included for completeness)

/-- The exterior angle formed by the non-shared sides of the square and octagon -/
def exterior_angle (config : SquareOctagonConfig) : ℚ :=
  360 - config.square_angle - config.octagon_angle

/-- Theorem: The exterior angle in the square-octagon configuration is 135° -/
theorem square_octagon_exterior_angle :
  ∀ (config : SquareOctagonConfig),
    config.square_angle = 90 ∧
    config.octagon_angle = interior_angle 8 →
    exterior_angle config = 135 := by
  sorry


end square_octagon_exterior_angle_l250_25013


namespace box_counting_l250_25070

theorem box_counting (initial_boxes : ℕ) (boxes_per_operation : ℕ) (final_nonempty_boxes : ℕ) : 
  initial_boxes = 2013 → 
  boxes_per_operation = 13 → 
  final_nonempty_boxes = 2013 →
  initial_boxes + boxes_per_operation * final_nonempty_boxes = 28182 := by
  sorry

#check box_counting

end box_counting_l250_25070


namespace beautiful_equations_proof_l250_25015

/-- Two linear equations are "beautiful equations" if the sum of their solutions is 1 -/
def beautiful_equations (eq1 eq2 : ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), eq1 x ∧ eq2 y ∧ x + y = 1

/-- The first pair of equations -/
def eq1 (x : ℝ) : Prop := 4*x - (x + 5) = 1
def eq2 (y : ℝ) : Prop := -2*y - y = 3

/-- The second pair of equations with parameter n -/
def eq3 (n : ℝ) (x : ℝ) : Prop := 2*x - n + 3 = 0
def eq4 (n : ℝ) (x : ℝ) : Prop := x + 5*n - 1 = 0

theorem beautiful_equations_proof :
  (beautiful_equations eq1 eq2) ∧
  (∃ (n : ℝ), n = -1/3 ∧ beautiful_equations (eq3 n) (eq4 n)) :=
by sorry

end beautiful_equations_proof_l250_25015


namespace train_speed_l250_25028

/-- The speed of a train given its length and time to cross a post -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 250.02) (h2 : time = 22.5) :
  (length * 3600) / (time * 1000) = 40.0032 := by
  sorry

end train_speed_l250_25028


namespace some_number_value_l250_25054

theorem some_number_value : ∃ (some_number : ℝ), 
  (0.0077 * 3.6) / (0.04 * some_number * 0.007) = 990.0000000000001 ∧ some_number = 10 := by
  sorry

end some_number_value_l250_25054


namespace smallest_number_with_conditions_l250_25055

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (∀ k : ℕ, k < n → ¬(11 ∣ k ∧ (∀ i : ℕ, 3 ≤ i ∧ i ≤ 7 → k % i = 2))) ∧ 
  11 ∣ n ∧ 
  (∀ i : ℕ, 3 ≤ i ∧ i ≤ 7 → n % i = 2) ∧ 
  n = 2102 :=
by sorry

end smallest_number_with_conditions_l250_25055


namespace pen_cost_calculation_l250_25077

theorem pen_cost_calculation (pack_size : ℕ) (pack_cost : ℚ) (desired_pens : ℕ) : 
  pack_size = 150 → pack_cost = 45 → desired_pens = 3600 →
  (desired_pens : ℚ) * (pack_cost / pack_size) = 1080 := by
  sorry

end pen_cost_calculation_l250_25077


namespace inequality_assignment_exists_l250_25007

/-- Represents the inequality symbols on even-positioned cards -/
def InequalitySequence := Fin 50 → Bool

/-- Represents the assignment of numbers to odd-positioned cards -/
def NumberAssignment := Fin 51 → Fin 51

/-- Checks if a number assignment satisfies the given inequality sequence -/
def is_valid_assignment (ineq : InequalitySequence) (assign : NumberAssignment) : Prop :=
  ∀ i : Fin 50, 
    (ineq i = true → assign i < assign (i + 1)) ∧
    (ineq i = false → assign i > assign (i + 1))

/-- The main theorem stating that a valid assignment always exists -/
theorem inequality_assignment_exists (ineq : InequalitySequence) :
  ∃ (assign : NumberAssignment), is_valid_assignment ineq assign ∧ Function.Bijective assign :=
sorry

end inequality_assignment_exists_l250_25007


namespace sqrt_two_minus_x_real_range_l250_25079

theorem sqrt_two_minus_x_real_range :
  {x : ℝ | ∃ y : ℝ, y ^ 2 = 2 - x} = {x : ℝ | x ≤ 2} := by sorry

end sqrt_two_minus_x_real_range_l250_25079


namespace height_relation_l250_25012

/-- Two right circular cylinders with equal volumes and related radii -/
structure TwoCylinders where
  r₁ : ℝ  -- radius of the first cylinder
  h₁ : ℝ  -- height of the first cylinder
  r₂ : ℝ  -- radius of the second cylinder
  h₂ : ℝ  -- height of the second cylinder
  r₁_pos : 0 < r₁
  h₁_pos : 0 < h₁
  r₂_pos : 0 < r₂
  h₂_pos : 0 < h₂
  equal_volume : r₁^2 * h₁ = r₂^2 * h₂
  radius_relation : r₂ = 1.2 * r₁

theorem height_relation (c : TwoCylinders) : c.h₁ = 1.44 * c.h₂ := by
  sorry

end height_relation_l250_25012


namespace hyperbola_eccentricity_l250_25069

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±2√2x -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_equation : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptotes : ∀ x, ∃ y, y = 2 * Real.sqrt 2 * x ∨ y = -2 * Real.sqrt 2 * x) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = 3 := by sorry

end hyperbola_eccentricity_l250_25069


namespace double_acute_angle_less_than_180_degrees_l250_25045

theorem double_acute_angle_less_than_180_degrees (α : Real) :
  (0 < α ∧ α < Real.pi / 2) → 2 * α < Real.pi := by
  sorry

end double_acute_angle_less_than_180_degrees_l250_25045


namespace english_math_only_count_l250_25041

/-- The number of students taking at least one subject -/
def total_students : ℕ := 28

/-- The number of students taking Mathematics and History, but not English -/
def math_history_only : ℕ := 6

theorem english_math_only_count :
  ∀ (math_only english_math_only math_english_history english_history_only : ℕ),
  -- The number taking Mathematics and English only equals the number taking Mathematics only
  math_only = english_math_only →
  -- No student takes English only or History only
  -- Six students take Mathematics and History, but not English (already defined as math_history_only)
  -- The number taking English and History only is five times the number taking all three subjects
  english_history_only = 5 * math_english_history →
  -- The number taking all three subjects is even and non-zero
  math_english_history % 2 = 0 ∧ math_english_history > 0 →
  -- The total number of students is correct
  total_students = math_only + english_math_only + math_history_only + english_history_only + math_english_history →
  -- Prove that the number of students taking English and Mathematics only is 5
  english_math_only = 5 := by
sorry

end english_math_only_count_l250_25041


namespace geometric_sequence_property_l250_25089

theorem geometric_sequence_property (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence definition
  a 4 = 1.5 →                   -- 4th term is 1.5
  a 10 = 1.62 →                 -- 10th term is 1.62
  a 7 = Real.sqrt 2.43 :=        -- 7th term is √2.43
by
  sorry


end geometric_sequence_property_l250_25089


namespace cube_sum_l250_25060

theorem cube_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cube_sum_l250_25060


namespace linear_function_not_in_second_quadrant_l250_25044

/-- A linear function f(x) = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The four quadrants of the Cartesian plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Determine if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I  => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- A linear function passes through a quadrant if there exists a point (x, y) in that quadrant satisfying the function equation -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  ∃ x y : ℝ, y = f.m * x + f.b ∧ inQuadrant x y q

/-- The main theorem: the graph of y = 2x - 3 does not pass through the second quadrant -/
theorem linear_function_not_in_second_quadrant :
  ¬ passesThrough ⟨2, -3⟩ Quadrant.II := by
  sorry


end linear_function_not_in_second_quadrant_l250_25044


namespace min_value_binomial_distribution_l250_25040

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The expected value of a binomial distribution -/
def expectedValue (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The minimum value of 1/p + 1/q for a binomial distribution
    with E(X) = 4 and D(X) = q is 9/4 -/
theorem min_value_binomial_distribution 
  (X : BinomialDistribution) 
  (h_exp : expectedValue X = 4)
  (h_var : variance X = q)
  : (1 / X.p + 1 / q) ≥ 9/4 :=
sorry

end min_value_binomial_distribution_l250_25040


namespace triangle_sum_l250_25084

theorem triangle_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + (1/3)*y^2 = 25)
  (eq2 : (1/3)*y^2 + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 16) :
  x*y + 2*y*z + 3*z*x = 24*Real.sqrt 3 := by
sorry

end triangle_sum_l250_25084


namespace inequality_implication_l250_25073

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end inequality_implication_l250_25073


namespace not_right_triangle_l250_25003

theorem not_right_triangle (a b c : ℝ) (ha : a = 1/3) (hb : b = 1/4) (hc : c = 1/5) :
  ¬ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) :=
by sorry

end not_right_triangle_l250_25003


namespace area_of_region_l250_25051

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 10 ∧ 
   A = Real.pi * (Real.sqrt ((x + 1)^2 + (y - 2)^2)) ^ 2 ∧
   x^2 + y^2 + 2*x - 4*y = 5) := by sorry

end area_of_region_l250_25051


namespace function_expression_l250_25091

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 1) :
  ∀ x, f x = x^2 - 2*x :=
by sorry

end function_expression_l250_25091


namespace carpet_exchange_theorem_l250_25092

theorem carpet_exchange_theorem (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  ∃ c : ℝ, c > 0 ∧ ((c > 1 ∧ a / c < 1) ∨ (c < 1 ∧ a / c > 1)) := by
  sorry

end carpet_exchange_theorem_l250_25092


namespace range_of_trig_function_l250_25093

theorem range_of_trig_function :
  ∀ x : ℝ, (3 / 8 : ℝ) ≤ Real.sin x ^ 6 + Real.cos x ^ 4 ∧
            Real.sin x ^ 6 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end range_of_trig_function_l250_25093


namespace point_division_theorem_l250_25008

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 3:5,
    prove that Q = (5/8)*C + (3/8)*D -/
theorem point_division_theorem (C D Q : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D) 
  (h2 : ∃ k : ℝ, k > 0 ∧ (Q - C) = k • (3 • (D - C))) :
  Q = (5/8) • C + (3/8) • D :=
sorry

end point_division_theorem_l250_25008


namespace max_value_problem_l250_25061

theorem max_value_problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  ∃ (max : ℝ), max = 1 ∧ x + y^3 + z^2 ≤ max ∧ ∃ (x' y' z' : ℝ), x' + y'^3 + z'^2 = max :=
sorry

end max_value_problem_l250_25061


namespace bigger_part_is_34_l250_25068

theorem bigger_part_is_34 (x y : ℝ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) :
  max x y = 34 := by
  sorry

end bigger_part_is_34_l250_25068


namespace intersection_A_B_l250_25090

-- Define set A
def A : Set ℝ := {x | x^2 - x - 6 > 0}

-- Define set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | -3 ≤ x ∧ x < -2} := by sorry

end intersection_A_B_l250_25090


namespace max_value_theorem_l250_25052

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 3) : 
  2*a*b + 2*b*c*Real.sqrt 3 ≤ 6 ∧ ∃ a b c, 2*a*b + 2*b*c*Real.sqrt 3 = 6 := by
  sorry

end max_value_theorem_l250_25052


namespace discount_calculation_l250_25039

/-- Calculates the final amount paid after applying a discount -/
def finalAmount (initialAmount : ℕ) (discountPer100 : ℕ) : ℕ :=
  let fullDiscountUnits := initialAmount / 100
  let totalDiscount := fullDiscountUnits * discountPer100
  initialAmount - totalDiscount

/-- Theorem stating that for a $250 purchase with a $10 discount per $100 spent, the final amount is $230 -/
theorem discount_calculation :
  finalAmount 250 10 = 230 := by
  sorry

end discount_calculation_l250_25039


namespace quadratic_equation_properties_l250_25011

-- Define the quadratic equation
def quadratic_equation (m n x : ℝ) : Prop := x^2 + m*x + n = 0

-- Define the condition for two real roots
def has_two_real_roots (m n : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m n x₁ ∧ quadratic_equation m n x₂

-- Define the condition for negative roots
def has_negative_roots (m n : ℝ) : Prop := ∀ x : ℝ, quadratic_equation m n x → x < 0

-- Define the inequality
def inequality (m n t : ℝ) : Prop := t ≤ (m-1)^2 + (n-1)^2 + (m-n)^2

-- Theorem statement
theorem quadratic_equation_properties :
  ∀ m n : ℝ, has_two_real_roots m n →
  (∃ t : ℝ, (n = 3 - m ∧ has_negative_roots m n) → 2 ≤ m ∧ m < 3) ∧
  (∃ t_max : ℝ, t_max = 9/8 ∧ ∀ t : ℝ, inequality m n t → t ≤ t_max) :=
by sorry

end quadratic_equation_properties_l250_25011


namespace black_ball_probability_l250_25098

theorem black_ball_probability (total : ℕ) (white yellow black : ℕ) :
  total = white + yellow + black →
  white = 10 →
  yellow = 5 →
  black = 10 →
  (black : ℚ) / (yellow + black) = 2 / 3 :=
by sorry

end black_ball_probability_l250_25098


namespace salary_restoration_l250_25017

theorem salary_restoration (original_salary : ℝ) (reduced_salary : ℝ) : 
  reduced_salary = original_salary * (1 - 0.5) → 
  reduced_salary * 2 = original_salary :=
by sorry

end salary_restoration_l250_25017


namespace sine_special_angle_l250_25004

theorem sine_special_angle (α : Real) 
  (h1 : π / 2 < α ∧ α < π) 
  (h2 : Real.sin (-π - α) = Real.sqrt 5 / 5) : 
  Real.sin (α - 3 * π / 2) = 2 * Real.sqrt 5 / 5 := by
sorry

end sine_special_angle_l250_25004


namespace log_eight_three_equals_five_twelve_l250_25019

theorem log_eight_three_equals_five_twelve (x : ℝ) : 
  Real.log x / Real.log 8 = 3 → x = 512 := by
  sorry

end log_eight_three_equals_five_twelve_l250_25019


namespace avg_temp_MTWT_is_48_l250_25059

/-- The average temperature for Monday, Tuesday, Wednesday, and Thursday -/
def avg_temp_MTWT : ℝ := sorry

/-- The average temperature for some days -/
def avg_temp_some_days : ℝ := 48

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday -/
def avg_temp_TWTF : ℝ := 40

/-- The temperature on Monday -/
def temp_Monday : ℝ := 42

/-- The temperature on Friday -/
def temp_Friday : ℝ := 10

/-- The theorem stating that the average temperature for Monday, Tuesday, Wednesday, and Thursday is 48 degrees -/
theorem avg_temp_MTWT_is_48 : avg_temp_MTWT = 48 := by sorry

end avg_temp_MTWT_is_48_l250_25059


namespace symmetric_circle_equation_l250_25034

/-- Given a circle with equation x^2 + y^2 - 4x = 0, its symmetric circle
    with respect to the line x = 0 has the equation x^2 + y^2 + 4x = 0 -/
theorem symmetric_circle_equation : 
  ∀ (x y : ℝ), (x^2 + y^2 - 4*x = 0) → 
  ∃ (x' y' : ℝ), (x'^2 + y'^2 + 4*x' = 0) ∧ (x' = -x) ∧ (y' = y) := by
sorry

end symmetric_circle_equation_l250_25034


namespace circle_chord_theorem_l250_25049

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y + 2*a = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + y + 2 = 0

-- Define the chord length
def chord_length : ℝ := 4

-- Theorem statement
theorem circle_chord_theorem (a : ℝ) :
  (∀ x y : ℝ, circle_equation x y a ∧ line_equation x y) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ a ∧ line_equation x₁ y₁ ∧
    circle_equation x₂ y₂ a ∧ line_equation x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2) →
  a = -2 := by
  sorry

end circle_chord_theorem_l250_25049


namespace select_students_with_female_l250_25065

/-- The number of male students -/
def num_male : ℕ := 5

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students to be selected -/
def num_selected : ℕ := 3

/-- The number of ways to select students with at least one female -/
def num_ways_with_female : ℕ := Nat.choose (num_male + num_female) num_selected - Nat.choose num_male num_selected

theorem select_students_with_female :
  num_ways_with_female = 25 := by
  sorry

end select_students_with_female_l250_25065


namespace number_of_divisors_of_36_l250_25009

theorem number_of_divisors_of_36 : Finset.card (Finset.filter (· ∣ 36) (Finset.range 37)) = 9 := by
  sorry

end number_of_divisors_of_36_l250_25009


namespace specific_arrangement_surface_area_l250_25067

/-- Represents a cube arrangement with two layers --/
structure CubeArrangement where
  totalCubes : Nat
  layerSize : Nat
  cubeEdgeLength : Real

/-- Calculates the exposed surface area of the cube arrangement --/
def exposedSurfaceArea (arrangement : CubeArrangement) : Real :=
  sorry

/-- Theorem stating that the exposed surface area of the specific arrangement is 49 square meters --/
theorem specific_arrangement_surface_area :
  let arrangement : CubeArrangement := {
    totalCubes := 18,
    layerSize := 9,
    cubeEdgeLength := 1
  }
  exposedSurfaceArea arrangement = 49 := by sorry

end specific_arrangement_surface_area_l250_25067


namespace min_sum_squares_l250_25064

def S : Finset Int := {-8, -6, -4, -1, 3, 5, 7, 10}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 18 :=
by sorry

end min_sum_squares_l250_25064


namespace equation_solution_l250_25018

theorem equation_solution : 
  ∃ x : ℝ, (x + 1) / 6 = 4 / 3 - x ∧ x = 1 := by
  sorry

end equation_solution_l250_25018


namespace find_original_number_l250_25000

/-- A four-digit number is between 1000 and 9999 inclusive -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem find_original_number (N : ℕ) (h1 : FourDigitNumber N) (h2 : N - 3 - 57 = 1819) : N = 1879 := by
  sorry

end find_original_number_l250_25000


namespace largest_k_phi_sigma_power_two_l250_25076

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem largest_k_phi_sigma_power_two :
  (∀ k : ℕ, k > 31 → phi (sigma (2^k)) ≠ 2^k) ∧
  phi (sigma (2^31)) = 2^31 := by sorry

end largest_k_phi_sigma_power_two_l250_25076


namespace three_vectors_with_zero_sum_and_unit_difference_l250_25081

theorem three_vectors_with_zero_sum_and_unit_difference (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :
  ∃ (a b c : α), 
    a + b + c = 0 ∧ 
    ‖a + b - c‖ = 1 ∧ 
    ‖b + c - a‖ = 1 ∧ 
    ‖c + a - b‖ = 1 ∧
    ‖a‖ = (1 : ℝ) / 2 ∧ 
    ‖b‖ = (1 : ℝ) / 2 ∧ 
    ‖c‖ = (1 : ℝ) / 2 := by
  sorry

end three_vectors_with_zero_sum_and_unit_difference_l250_25081


namespace total_commute_time_is_16_l250_25030

-- Define the time it takes to walk and bike to work
def walk_time : ℕ := 2
def bike_time : ℕ := 1

-- Define the number of times Roque walks and bikes to work per week
def walk_trips : ℕ := 3
def bike_trips : ℕ := 2

-- Define the total commuting time
def total_commute_time : ℕ := 
  2 * (walk_time * walk_trips + bike_time * bike_trips)

-- Theorem statement
theorem total_commute_time_is_16 : total_commute_time = 16 := by
  sorry

end total_commute_time_is_16_l250_25030


namespace basil_seed_cost_l250_25035

/-- Represents the cost structure and profit for Burt's basil plant business -/
structure BasilBusiness where
  seed_cost : ℝ
  soil_cost : ℝ
  plants : ℕ
  price_per_plant : ℝ
  net_profit : ℝ

/-- Calculates the total revenue from selling basil plants -/
def total_revenue (b : BasilBusiness) : ℝ :=
  b.plants * b.price_per_plant

/-- Calculates the total expenses for the basil business -/
def total_expenses (b : BasilBusiness) : ℝ :=
  b.seed_cost + b.soil_cost

/-- Theorem stating that given the conditions, the seed cost is $2.00 -/
theorem basil_seed_cost (b : BasilBusiness) 
  (h1 : b.soil_cost = 8)
  (h2 : b.plants = 20)
  (h3 : b.price_per_plant = 5)
  (h4 : b.net_profit = 90)
  (h5 : total_revenue b - total_expenses b = b.net_profit) :
  b.seed_cost = 2 := by
  sorry

end basil_seed_cost_l250_25035


namespace gordons_second_restaurant_meals_l250_25086

/-- Given Gordon's restaurants and their meal serving information, prove that the second restaurant serves 40 meals per day. -/
theorem gordons_second_restaurant_meals (total_weekly_meals : ℕ)
  (first_restaurant_daily_meals : ℕ) (third_restaurant_daily_meals : ℕ)
  (h1 : total_weekly_meals = 770)
  (h2 : first_restaurant_daily_meals = 20)
  (h3 : third_restaurant_daily_meals = 50) :
  ∃ (second_restaurant_daily_meals : ℕ),
    second_restaurant_daily_meals = 40 ∧
    total_weekly_meals = 7 * (first_restaurant_daily_meals + second_restaurant_daily_meals + third_restaurant_daily_meals) :=
by sorry

end gordons_second_restaurant_meals_l250_25086


namespace point_on_line_extension_l250_25021

theorem point_on_line_extension (A B C D : EuclideanSpace ℝ (Fin 2)) :
  (D - A) = 2 • (B - A) - (C - A) →
  ∃ t : ℝ, t > 1 ∧ D = C + t • (B - C) :=
by sorry

end point_on_line_extension_l250_25021


namespace product_of_numbers_l250_25027

theorem product_of_numbers (x y : ℝ) 
  (h1 : (x - y)^2 / (x + y)^3 = 4 / 27)
  (h2 : x + y = 5 * (x - y) + 3) : 
  x * y = 15.75 := by
  sorry

end product_of_numbers_l250_25027


namespace square_difference_square_difference_40_l250_25042

theorem square_difference (n : ℕ) : (n + 1)^2 - (n - 1)^2 = 4 * n := by
  -- The proof goes here
  sorry

-- Define the specific case for n = 40
def n : ℕ := 40

-- State the theorem for the specific case
theorem square_difference_40 : (n + 1)^2 - (n - 1)^2 = 160 := by
  -- The proof goes here
  sorry

end square_difference_square_difference_40_l250_25042


namespace students_walking_home_l250_25087

theorem students_walking_home (bus_fraction automobile_fraction bicycle_fraction skateboard_fraction : ℚ) :
  bus_fraction = 1/3 →
  automobile_fraction = 1/5 →
  bicycle_fraction = 1/10 →
  skateboard_fraction = 1/15 →
  1 - (bus_fraction + automobile_fraction + bicycle_fraction + skateboard_fraction) = 3/10 := by
sorry

end students_walking_home_l250_25087


namespace factorization_of_quadratic_l250_25024

theorem factorization_of_quadratic (x : ℝ) : x^2 - 3*x = x*(x - 3) := by
  sorry

end factorization_of_quadratic_l250_25024


namespace perfect_square_solution_l250_25020

theorem perfect_square_solution (t n : ℤ) : 
  (n > 0) → (n^2 + (4*t - 1)*n + 4*t^2 = 0) → ∃ k : ℤ, n = k^2 := by
  sorry

end perfect_square_solution_l250_25020


namespace fixed_fee_calculation_l250_25074

/-- Represents the billing system for an online service provider -/
structure BillingSystem where
  fixed_fee : ℝ
  hourly_charge : ℝ

/-- Calculates the total bill given the connect time -/
def total_bill (bs : BillingSystem) (connect_time : ℝ) : ℝ :=
  bs.fixed_fee + bs.hourly_charge * connect_time

theorem fixed_fee_calculation (bs : BillingSystem) :
  total_bill bs 1 = 15.75 ∧ total_bill bs 3 = 24.45 → bs.fixed_fee = 11.40 := by
  sorry

end fixed_fee_calculation_l250_25074


namespace largest_angle_in_3_4_5_ratio_triangle_l250_25047

theorem largest_angle_in_3_4_5_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    b = (4/3) * a →
    c = (5/3) * a →
    a + b + c = 180 →
    c = 75 := by
  sorry

end largest_angle_in_3_4_5_ratio_triangle_l250_25047


namespace equilateral_triangle_perimeter_l250_25006

/-- The perimeter of an equilateral triangle with side length 13/12 meters is 3.25 meters. -/
theorem equilateral_triangle_perimeter :
  let side_length : ℚ := 13/12
  let perimeter : ℚ := 3 * side_length
  perimeter = 13/4 := by sorry

end equilateral_triangle_perimeter_l250_25006


namespace number_guessing_game_l250_25072

theorem number_guessing_game (a b c d : ℕ) 
  (ha : a ≥ 10) 
  (hb : b < 10) (hc : c < 10) (hd : d < 10) : 
  ((((((a * 2 + 1) * 5 + b) * 2 + 1) * 5 + c) * 2 + 1) * 5 + d) - 555 = 1000 * a + 100 * b + 10 * c + d :=
by sorry

#check number_guessing_game

end number_guessing_game_l250_25072


namespace quadratic_function_through_point_l250_25083

theorem quadratic_function_through_point (a b : ℝ) :
  (∀ t : ℝ, (t^2 + t + 1) * 1^2 - 2*(a+t)^2 * 1 + t^2 + 3*a*t + b = 0) →
  a = 1 ∧ b = 1 := by
sorry

end quadratic_function_through_point_l250_25083


namespace max_gcd_consecutive_b_l250_25088

def b (n : ℕ) : ℕ := n.factorial + n^2

theorem max_gcd_consecutive_b : ∀ n : ℕ, Nat.gcd (b n) (b (n + 1)) ≤ 2 ∧ 
  ∃ m : ℕ, Nat.gcd (b m) (b (m + 1)) = 2 := by
  sorry

end max_gcd_consecutive_b_l250_25088


namespace range_of_m_l250_25046

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 < x ∧ x < m + 1}

-- State the theorem
theorem range_of_m (m : ℝ) : (B m ⊆ A) → m ≥ -1 := by
  sorry

end range_of_m_l250_25046


namespace tabitha_money_to_mom_l250_25094

/-- The amount of money Tabitha gave her mom -/
def money_given_to_mom (initial_amount : ℚ) (item_cost : ℚ) (num_items : ℕ) (final_amount : ℚ) : ℚ :=
  initial_amount - 2 * (final_amount + item_cost * num_items)

/-- Theorem stating the amount of money Tabitha gave her mom -/
theorem tabitha_money_to_mom :
  money_given_to_mom 25 0.5 5 6 = 8 := by
  sorry

#eval money_given_to_mom 25 0.5 5 6

end tabitha_money_to_mom_l250_25094


namespace kiras_cat_kibble_l250_25082

/-- Calculates the amount of kibble Kira initially filled her cat's bowl with. -/
def initial_kibble_amount (eating_rate : ℚ) (time_away : ℚ) (kibble_left : ℚ) : ℚ :=
  (time_away / 4) * eating_rate + kibble_left

/-- Theorem stating that given the conditions, Kira initially filled the bowl with 3 pounds of kibble. -/
theorem kiras_cat_kibble : initial_kibble_amount 1 8 1 = 3 := by
  sorry

#eval initial_kibble_amount 1 8 1

end kiras_cat_kibble_l250_25082


namespace sample_size_is_75_l250_25016

/-- Represents the sample size of a stratified sample -/
def sample_size (model_A_count : ℕ) (ratio_A ratio_B ratio_C : ℕ) : ℕ :=
  model_A_count * (ratio_A + ratio_B + ratio_C) / ratio_A

/-- Theorem stating that the sample size is 75 given the problem conditions -/
theorem sample_size_is_75 :
  sample_size 15 2 3 5 = 75 := by
  sorry

#eval sample_size 15 2 3 5

end sample_size_is_75_l250_25016


namespace product_of_three_numbers_l250_25031

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_condition : a + b + c = 24)
  (sum_squares_condition : a^2 + b^2 + c^2 = 392)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 2760) :
  a * b * c = 1844 := by sorry

end product_of_three_numbers_l250_25031


namespace preimage_of_3_1_l250_25036

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (x+2y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2*p.2, 2*p.1 - p.2)

/-- The theorem stating that (-1/3, 5/3) is the pre-image of (3, 1) under the mapping f -/
theorem preimage_of_3_1 :
  f (-1/3, 5/3) = (3, 1) :=
by sorry

end preimage_of_3_1_l250_25036


namespace hexagon_area_equal_perimeter_l250_25096

theorem hexagon_area_equal_perimeter (s t : ℝ) : 
  s > 0 → 
  t > 0 → 
  3 * s = 6 * t → -- Equal perimeters condition
  s^2 * Real.sqrt 3 / 4 = 16 → -- Triangle area condition
  6 * (t^2 * Real.sqrt 3 / 4) = 24 := by
  sorry

end hexagon_area_equal_perimeter_l250_25096


namespace monotonically_decreasing_range_l250_25025

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + a*x - 1

-- Theorem statement
theorem monotonically_decreasing_range (a : ℝ) : 
  (∀ x : ℝ, f_derivative a x ≤ 0) → a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
by
  sorry

#check monotonically_decreasing_range

end monotonically_decreasing_range_l250_25025


namespace maria_stamp_collection_l250_25014

/-- Given that Maria has 40 stamps and wants to increase her collection by 20%,
    prove that she will have a total of 48 stamps. -/
theorem maria_stamp_collection (initial_stamps : ℕ) (increase_percentage : ℚ) : 
  initial_stamps = 40 → 
  increase_percentage = 20 / 100 → 
  initial_stamps + (initial_stamps * increase_percentage).floor = 48 := by
  sorry

end maria_stamp_collection_l250_25014


namespace negation_at_most_three_l250_25056

theorem negation_at_most_three (x : ℝ) : ¬(x ≤ 3) ↔ x > 3 := by sorry

end negation_at_most_three_l250_25056


namespace car_speed_problem_l250_25001

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_fraction : ℝ) 
  (h1 : distance = 720)
  (h2 : original_time = 8)
  (h3 : new_time_fraction = 5/8) :
  let new_time := new_time_fraction * original_time
  let new_speed := distance / new_time
  new_speed = 144 := by sorry

end car_speed_problem_l250_25001


namespace mathematician_contemporaries_probability_l250_25048

theorem mathematician_contemporaries_probability :
  let total_years : ℕ := 600
  let lifespan1 : ℕ := 120
  let lifespan2 : ℕ := 100
  let total_area : ℕ := total_years * total_years
  let overlap_area : ℕ := total_area - (lifespan1 * lifespan1 / 2 + lifespan2 * lifespan2 / 2)
  (overlap_area : ℚ) / total_area = 193 / 200 :=
by sorry

end mathematician_contemporaries_probability_l250_25048


namespace jerry_thermostat_problem_l250_25078

/-- Calculates the final temperature after a series of adjustments --/
def finalTemperature (initial : ℝ) : ℝ :=
  let doubled := initial * 2
  let afterDad := doubled - 30
  let afterMom := afterDad * 0.7  -- Reducing by 30% is equivalent to multiplying by 0.7
  let final := afterMom + 24
  final

/-- Theorem stating that the final temperature is 59 degrees --/
theorem jerry_thermostat_problem : finalTemperature 40 = 59 := by
  sorry

end jerry_thermostat_problem_l250_25078


namespace set_associativity_l250_25053

theorem set_associativity (A B C : Set α) : 
  (A ∪ (B ∪ C) = (A ∪ B) ∪ C) ∧ (A ∩ (B ∩ C) = (A ∩ B) ∩ C) := by
  sorry

end set_associativity_l250_25053


namespace blue_balloon_count_l250_25097

/-- The number of blue balloons owned by Joan, Melanie, Alex, and Gary, respectively --/
def joan_balloons : ℕ := 60
def melanie_balloons : ℕ := 85
def alex_balloons : ℕ := 37
def gary_balloons : ℕ := 48

/-- The total number of blue balloons --/
def total_blue_balloons : ℕ := joan_balloons + melanie_balloons + alex_balloons + gary_balloons

theorem blue_balloon_count : total_blue_balloons = 230 := by
  sorry

end blue_balloon_count_l250_25097


namespace seating_probability_is_two_sevenths_l250_25037

/-- The number of boys to be seated -/
def num_boys : ℕ := 5

/-- The number of girls to be seated -/
def num_girls : ℕ := 6

/-- The total number of chairs -/
def total_chairs : ℕ := 11

/-- The probability of seating boys and girls with the given condition -/
def seating_probability : ℚ :=
  2 / 7

/-- Theorem stating that the probability of seating boys and girls
    such that there are no more boys than girls at any point is 2/7 -/
theorem seating_probability_is_two_sevenths :
  seating_probability = 2 / 7 := by sorry

end seating_probability_is_two_sevenths_l250_25037


namespace range_of_a_l250_25063

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

theorem range_of_a (m n : ℝ) (h1 : 0 < m) (h2 : m < n) (h3 : 0 ≤ a) 
  (h4 : Set.Icc m n ⊆ Set.range (f a))
  (h5 : Set.Icc m n ⊆ Set.range (f a ∘ f a)) :
  1 - Real.exp (-1) ≤ a ∧ a < 1 := by sorry

end range_of_a_l250_25063


namespace calculation_proof_l250_25032

theorem calculation_proof :
  (1 * (-8) - 9 - (-3) + (-6) = -20) ∧
  (-2^2 + 3 * (-1)^2023 - |1 - 5| / 2 = -9) := by
  sorry

end calculation_proof_l250_25032


namespace gabriel_forgotten_days_l250_25022

/-- The number of days in July -/
def days_in_july : ℕ := 31

/-- The number of days Gabriel took his capsules -/
def days_capsules_taken : ℕ := 28

/-- The number of days Gabriel forgot to take his capsules -/
def days_forgotten : ℕ := days_in_july - days_capsules_taken

theorem gabriel_forgotten_days :
  days_forgotten = 3 := by sorry

end gabriel_forgotten_days_l250_25022


namespace order_of_values_l250_25029

theorem order_of_values : ∃ (a b c : ℝ),
  a = Real.exp 0.2 - 1 ∧
  b = Real.log 1.2 ∧
  c = Real.tan 0.2 ∧
  a > c ∧ c > b := by
  sorry

end order_of_values_l250_25029


namespace ali_baba_maximum_value_l250_25010

/-- Represents the problem of maximizing the value of gold and diamonds in one trip --/
theorem ali_baba_maximum_value :
  let gold_weight : ℝ := 200
  let diamond_weight : ℝ := 40
  let max_carry_weight : ℝ := 100
  let gold_value_per_kg : ℝ := 20
  let diamond_value_per_kg : ℝ := 60
  
  ∀ x y : ℝ,
  x ≥ 0 → y ≥ 0 →
  x + y = max_carry_weight →
  x * gold_value_per_kg + y * diamond_value_per_kg ≤ 3000 :=
by sorry

end ali_baba_maximum_value_l250_25010


namespace remainder_problem_l250_25005

theorem remainder_problem (d r : ℤ) : 
  d > 1 →
  1223 % d = r →
  1625 % d = r →
  2513 % d = r →
  d - r = 1 := by
  sorry

end remainder_problem_l250_25005


namespace cubic_root_function_l250_25057

theorem cubic_root_function (k : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = k * x^(1/3)) →
  (∃ y : ℝ, y = 4 * Real.sqrt 3 ∧ 64^(1/3) * k = y) →
  (∃ y : ℝ, y = 2 * Real.sqrt 3 ∧ 8^(1/3) * k = y) :=
by sorry

end cubic_root_function_l250_25057


namespace negative_power_six_interpretation_l250_25033

theorem negative_power_six_interpretation :
  -2^6 = -(2 * 2 * 2 * 2 * 2 * 2) := by
  sorry

end negative_power_six_interpretation_l250_25033


namespace lilith_water_bottle_price_l250_25066

/-- The regular price per water bottle in Lilith's town -/
def regularPrice : ℚ := 185 / 100

theorem lilith_water_bottle_price :
  let initialBottles : ℕ := 60
  let initialPrice : ℚ := 2
  let shortfall : ℚ := 9
  (initialBottles : ℚ) * regularPrice = initialBottles * initialPrice - shortfall :=
by sorry

end lilith_water_bottle_price_l250_25066


namespace sufficient_but_not_necessary_l250_25080

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a set of four points
def FourPoints := Fin 4 → Point3D

-- Define collinearity for three points
def collinear (p q r : Point3D) : Prop := sorry

-- Define coplanarity for four points
def coplanar (points : FourPoints) : Prop := sorry

-- No three points are collinear
def no_three_collinear (points : FourPoints) : Prop :=
  ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬collinear (points i) (points j) (points k)

theorem sufficient_but_not_necessary :
  (∀ (points : FourPoints), no_three_collinear points → ¬coplanar points) ∧
  (∃ (points : FourPoints), ¬coplanar points ∧ ¬no_three_collinear points) := by
  sorry

end sufficient_but_not_necessary_l250_25080


namespace directrix_of_given_parabola_l250_25026

/-- A parabola in the xy-plane -/
structure Parabola where
  /-- The equation of the parabola in the form y = ax^2 + bx + c -/
  equation : ℝ → ℝ → Prop

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → ℝ → Prop :=
  sorry

/-- The given parabola y = -3x^2 + 6x - 5 -/
def given_parabola : Parabola :=
  { equation := λ x y => y = -3 * x^2 + 6 * x - 5 }

theorem directrix_of_given_parabola :
  directrix given_parabola = λ x y => y = -35/18 :=
sorry

end directrix_of_given_parabola_l250_25026


namespace unique_root_condition_l250_25058

/-- The system of equations has only one root if and only if m is 0 or 2 -/
theorem unique_root_condition (m : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^2 = 2*|p.1| ∧ |p.1| - p.2 - m = 1 - p.2^2) ↔ (m = 0 ∨ m = 2) :=
sorry

end unique_root_condition_l250_25058


namespace belle_biscuits_l250_25085

/-- The number of dog biscuits Belle eats every evening -/
def num_biscuits : ℕ := 4

/-- The number of rawhide bones Belle eats every evening -/
def num_bones : ℕ := 2

/-- The cost of one rawhide bone in dollars -/
def cost_bone : ℚ := 1

/-- The cost of one dog biscuit in dollars -/
def cost_biscuit : ℚ := 1/4

/-- The total cost to feed Belle these treats for a week in dollars -/
def total_cost : ℚ := 21

theorem belle_biscuits :
  num_biscuits = 4 ∧
  (7 : ℚ) * (num_bones * cost_bone + num_biscuits * cost_biscuit) = total_cost :=
sorry

end belle_biscuits_l250_25085


namespace optimal_sales_distribution_l250_25043

/-- Represents the sales and profit model for a company selling robots in two locations --/
structure RobotSales where
  x : ℝ  -- Monthly sales volume in both locations
  production_cost : ℝ := 200
  price_A : ℝ := 500
  price_B : ℝ → ℝ := λ x => 1200 - x
  advert_cost_A : ℝ → ℝ := λ x => 100 * x + 10000
  advert_cost_B : ℝ := 50000
  total_sales : ℝ := 1000

/-- Calculates the profit for location A --/
def profit_A (model : RobotSales) : ℝ :=
  model.x * model.price_A - model.x * model.production_cost - model.advert_cost_A model.x

/-- Calculates the profit for location B --/
def profit_B (model : RobotSales) : ℝ :=
  model.x * model.price_B model.x - model.x * model.production_cost - model.advert_cost_B

/-- Calculates the total profit for both locations --/
def total_profit (model : RobotSales) : ℝ :=
  profit_A model + profit_B model

/-- Theorem stating the optimal sales distribution --/
theorem optimal_sales_distribution (model : RobotSales) :
  ∃ (x_A x_B : ℝ),
    x_A + x_B = model.total_sales ∧
    x_A = 600 ∧
    x_B = 400 ∧
    ∀ (y_A y_B : ℝ),
      y_A + y_B = model.total_sales →
      total_profit { model with x := y_A } + total_profit { model with x := y_B } ≤
      total_profit { model with x := x_A } + total_profit { model with x := x_B } :=
sorry

end optimal_sales_distribution_l250_25043


namespace plumbing_job_washers_remaining_l250_25071

/-- Calculates the number of washers remaining after a plumbing job. -/
def washers_remaining (copper_pipe : ℕ) (pvc_pipe : ℕ) (steel_pipe : ℕ) 
  (copper_bolt_length : ℕ) (pvc_bolt_length : ℕ) (steel_bolt_length : ℕ)
  (copper_washers_per_bolt : ℕ) (pvc_washers_per_bolt : ℕ) (steel_washers_per_bolt : ℕ)
  (total_washers : ℕ) : ℕ :=
  let copper_bolts := (copper_pipe + copper_bolt_length - 1) / copper_bolt_length
  let pvc_bolts := (pvc_pipe + pvc_bolt_length - 1) / pvc_bolt_length * 2
  let steel_bolts := (steel_pipe + steel_bolt_length - 1) / steel_bolt_length
  let washers_used := copper_bolts * copper_washers_per_bolt + 
                      pvc_bolts * pvc_washers_per_bolt + 
                      steel_bolts * steel_washers_per_bolt
  total_washers - washers_used

theorem plumbing_job_washers_remaining :
  washers_remaining 40 30 20 5 10 8 2 3 4 80 = 43 := by
  sorry

end plumbing_job_washers_remaining_l250_25071


namespace yearly_income_is_130_l250_25023

/-- Calculates the yearly simple interest income given principal and rate -/
def simple_interest (principal : ℕ) (rate : ℕ) : ℕ :=
  principal * rate / 100

/-- Proves that the yearly annual income is 130 given the specified conditions -/
theorem yearly_income_is_130 (total : ℕ) (part1 : ℕ) (rate1 : ℕ) (rate2 : ℕ) 
  (h1 : total = 2500)
  (h2 : part1 = 2000)
  (h3 : rate1 = 5)
  (h4 : rate2 = 6) :
  simple_interest part1 rate1 + simple_interest (total - part1) rate2 = 130 := by
  sorry

#eval simple_interest 2000 5 + simple_interest 500 6

end yearly_income_is_130_l250_25023


namespace quadratic_transformation_has_integer_roots_l250_25075

/-- Represents a quadratic polynomial ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic polynomial has integer roots -/
def has_integer_roots (p : QuadraticPolynomial) : Prop :=
  ∃ (x : ℤ), p.a * x^2 + p.b * x + p.c = 0

/-- Represents a single step in the transformation process -/
inductive TransformationStep
  | IncreaseX
  | DecreaseX
  | IncreaseConstant
  | DecreaseConstant

/-- Applies a transformation step to a polynomial -/
def apply_step (p : QuadraticPolynomial) (step : TransformationStep) : QuadraticPolynomial :=
  match step with
  | TransformationStep.IncreaseX => { a := p.a, b := p.b + 1, c := p.c }
  | TransformationStep.DecreaseX => { a := p.a, b := p.b - 1, c := p.c }
  | TransformationStep.IncreaseConstant => { a := p.a, b := p.b, c := p.c + 1 }
  | TransformationStep.DecreaseConstant => { a := p.a, b := p.b, c := p.c - 1 }

theorem quadratic_transformation_has_integer_roots 
  (initial : QuadraticPolynomial)
  (final : QuadraticPolynomial)
  (h_initial : initial = { a := 1, b := 10, c := 20 })
  (h_final : final = { a := 1, b := 20, c := 10 })
  (h_transform : ∃ (steps : List TransformationStep), 
    final = steps.foldl apply_step initial) :
  ∃ (intermediate : QuadraticPolynomial),
    (∃ (steps : List TransformationStep), intermediate = steps.foldl apply_step initial) ∧
    has_integer_roots intermediate :=
  sorry

end quadratic_transformation_has_integer_roots_l250_25075


namespace power_of_2_probability_l250_25002

/-- A number is a four-digit number in base 4 if it's between 1000₄ and 3333₄ inclusive -/
def IsFourDigitBase4 (n : ℕ) : Prop :=
  64 ≤ n ∧ n ≤ 255

/-- The count of four-digit numbers in base 4 -/
def CountFourDigitBase4 : ℕ := 255 - 64 + 1

/-- A number is a power of 2 if its log base 2 is an integer -/
def IsPowerOf2 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- The count of powers of 2 that are four-digit numbers in base 4 -/
def CountPowerOf2FourDigitBase4 : ℕ := 2

/-- The probability of a randomly chosen four-digit number in base 4 being a power of 2 -/
def ProbabilityPowerOf2FourDigitBase4 : ℚ :=
  CountPowerOf2FourDigitBase4 / CountFourDigitBase4

theorem power_of_2_probability :
  ProbabilityPowerOf2FourDigitBase4 = 1 / 96 := by
  sorry

end power_of_2_probability_l250_25002


namespace milan_phone_bill_l250_25038

/-- Calculates the number of minutes billed given the total bill, monthly fee, and cost per minute -/
def minutes_billed (total_bill monthly_fee cost_per_minute : ℚ) : ℚ :=
  (total_bill - monthly_fee) / cost_per_minute

/-- Proves that given the specified conditions, the number of minutes billed is 178 -/
theorem milan_phone_bill :
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let cost_per_minute : ℚ := 0.12
  minutes_billed total_bill monthly_fee cost_per_minute = 178 := by
  sorry

end milan_phone_bill_l250_25038


namespace f_composition_fixed_points_l250_25050

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem f_composition_fixed_points :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} := by sorry

end f_composition_fixed_points_l250_25050


namespace four_common_divisors_l250_25062

/-- The number of positive integer divisors that simultaneously divide 60, 84, and 126 -/
def common_divisors : Nat :=
  (Nat.divisors 60 ∩ Nat.divisors 84 ∩ Nat.divisors 126).card

/-- Theorem stating that there are exactly 4 positive integers that simultaneously divide 60, 84, and 126 -/
theorem four_common_divisors : common_divisors = 4 := by
  sorry

end four_common_divisors_l250_25062


namespace field_division_l250_25095

theorem field_division (total_area : ℝ) (smaller_area larger_area : ℝ) : 
  total_area = 900 →
  smaller_area + larger_area = total_area →
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 405 := by
sorry

end field_division_l250_25095


namespace count_numbers_with_6_or_7_correct_l250_25099

/-- The number of integers from 1 to 729 (inclusive) in base 9 that contain at least one digit 6 or 7 -/
def count_numbers_with_6_or_7 : ℕ := 386

/-- The total number of integers we're considering -/
def total_numbers : ℕ := 729

/-- The base of the number system we're using -/
def base : ℕ := 9

/-- The number of digits available that are neither 6 nor 7 -/
def digits_without_6_or_7 : ℕ := 7

theorem count_numbers_with_6_or_7_correct :
  count_numbers_with_6_or_7 = total_numbers - digits_without_6_or_7^3 :=
sorry

end count_numbers_with_6_or_7_correct_l250_25099
