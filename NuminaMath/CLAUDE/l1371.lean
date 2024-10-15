import Mathlib

namespace NUMINAMATH_CALUDE_matrix_equation_holds_l1371_137189

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 2; 1, 0, 1; 2, 1, 0]

theorem matrix_equation_holds :
  let s : ℤ := -2
  let t : ℤ := -6
  let u : ℤ := -14
  let v : ℤ := -13
  A^4 + s • A^3 + t • A^2 + u • A + v • (1 : Matrix (Fin 3) (Fin 3) ℤ) = (0 : Matrix (Fin 3) (Fin 3) ℤ) := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_holds_l1371_137189


namespace NUMINAMATH_CALUDE_second_boy_probability_l1371_137117

/-- Represents a student in the classroom -/
inductive Student : Type
| Boy : Student
| Girl : Student

/-- The type of all possible orders in which students can leave -/
def LeaveOrder := List Student

/-- Generate all possible leave orders for 2 boys and 2 girls -/
def allLeaveOrders : List LeaveOrder :=
  sorry

/-- Check if the second student in a leave order is a boy -/
def isSecondBoy (order : LeaveOrder) : Bool :=
  sorry

/-- Count the number of leave orders where the second student is a boy -/
def countSecondBoy (orders : List LeaveOrder) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem second_boy_probability (orders : List LeaveOrder) 
  (h1 : orders = allLeaveOrders) 
  (h2 : orders.length = 6) : 
  (countSecondBoy orders : ℚ) / (orders.length : ℚ) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_second_boy_probability_l1371_137117


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l1371_137135

/-- Two 3x3 matrices that are inverses of each other -/
def matrix1 (a b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![a, 1, b; 2, 2, 3; c, 5, d]
def matrix2 (e f g h : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![-5, e, -11; f, -13, g; 2, h, 4]

/-- The theorem stating that the sum of all variables is 45 -/
theorem inverse_matrices_sum (a b c d e f g h : ℝ) :
  (matrix1 a b c d) * (matrix2 e f g h) = 1 →
  a + b + c + d + e + f + g + h = 45 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l1371_137135


namespace NUMINAMATH_CALUDE_extremal_point_implies_k_range_l1371_137165

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (Real.exp x) / (x^2) + 2*k*(Real.log x) - k*x

theorem extremal_point_implies_k_range :
  ∀ k : ℝ, (∀ x : ℝ, x > 0 → (x ≠ 2 → (deriv (f k)) x ≠ 0)) →
  k ∈ Set.Iic ((Real.exp 2) / 4) :=
sorry

end NUMINAMATH_CALUDE_extremal_point_implies_k_range_l1371_137165


namespace NUMINAMATH_CALUDE_range_of_m_correct_l1371_137127

/-- The range of m satisfying the given conditions -/
def range_of_m : Set ℝ :=
  {m | m ≥ 3 ∨ (1 < m ∧ m ≤ 2)}

/-- Condition p: x^2 + mx + 1 = 0 has two distinct negative roots -/
def condition_p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
    x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Condition q: 4x^2 + 4(m-2)x + 1 = 0 has no real roots -/
def condition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m_correct :
  ∀ m : ℝ, (condition_p m ∨ condition_q m) ∧ ¬(condition_p m ∧ condition_q m) ↔ m ∈ range_of_m :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_correct_l1371_137127


namespace NUMINAMATH_CALUDE_vectors_problem_l1371_137138

def a : ℝ × ℝ := (3, -1)
def b (k : ℝ) : ℝ × ℝ := (1, k)

theorem vectors_problem (k : ℝ) 
  (h : a.1 * (b k).1 + a.2 * (b k).2 = 0) : 
  k = 3 ∧ 
  (a.1 + (b k).1) * (a.1 - (b k).1) + (a.2 + (b k).2) * (a.2 - (b k).2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_problem_l1371_137138


namespace NUMINAMATH_CALUDE_complement_of_A_l1371_137155

-- Define the universal set U
def U : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Define the set A
def A : Set ℝ := {x | 0 < x ∧ x < 1/3}

-- Define the complement of A with respect to U
def complementU (A : Set ℝ) : Set ℝ := {x ∈ U | x ∉ A}

-- Theorem statement
theorem complement_of_A : complementU A = {x | x = 0 ∨ 1/3 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l1371_137155


namespace NUMINAMATH_CALUDE_problem_statement_l1371_137118

theorem problem_statement (number : ℝ) (value : ℝ) : 
  number = 1.375 →
  0.6667 * number + 0.75 = value →
  value = 1.666675 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1371_137118


namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l1371_137107

theorem beef_weight_loss_percentage 
  (initial_weight : ℝ) 
  (processed_weight : ℝ) 
  (h1 : initial_weight = 1500) 
  (h2 : processed_weight = 750) : 
  (initial_weight - processed_weight) / initial_weight * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l1371_137107


namespace NUMINAMATH_CALUDE_ellipse_y_equation_ellipse_x_equation_l1371_137122

/-- An ellipse with foci on the y-axis -/
structure EllipseY where
  c : ℝ
  e : ℝ

/-- An ellipse passing through a point on the x-axis -/
structure EllipseX where
  x : ℝ
  e : ℝ

/-- Standard equation of an ellipse -/
def standardEquation (a b : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_y_equation (E : EllipseY) (h1 : E.c = 6) (h2 : E.e = 2/3) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ standardEquation a b = standardEquation 9 (Real.sqrt 45) :=
sorry

theorem ellipse_x_equation (E : EllipseX) (h1 : E.x = 2) (h2 : E.e = Real.sqrt 3 / 2) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ standardEquation a b = standardEquation 2 1) ∨
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ standardEquation a b = standardEquation 4 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_y_equation_ellipse_x_equation_l1371_137122


namespace NUMINAMATH_CALUDE_symmetry_implies_ratio_l1371_137111

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℚ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if points A(m+4, -1) and B(1, n-3) are symmetric with respect to the origin,
    then m/n = -5/4 -/
theorem symmetry_implies_ratio (m n : ℚ) :
  symmetric_wrt_origin (m + 4) (-1) 1 (n - 3) →
  m / n = -5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_ratio_l1371_137111


namespace NUMINAMATH_CALUDE_expression_evaluation_l1371_137100

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 11)
  (h2 : b = a + 3)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7) = 72 / 35 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1371_137100


namespace NUMINAMATH_CALUDE_function_f_form_l1371_137170

/-- A function from positive integers to non-negative integers satisfying the given property -/
def FunctionF (f : ℕ+ → ℕ) : Prop :=
  f ≠ 0 ∧ ∀ a b : ℕ+, 2 * f (a * b) = (↑b + 1) * f a + (↑a + 1) * f b

/-- The main theorem stating the existence of c such that f(n) = c(n-1) -/
theorem function_f_form (f : ℕ+ → ℕ) (hf : FunctionF f) :
  ∃ c : ℕ, ∀ n : ℕ+, f n = c * (↑n - 1) :=
sorry

end NUMINAMATH_CALUDE_function_f_form_l1371_137170


namespace NUMINAMATH_CALUDE_decreasing_cubic_implies_nonpositive_a_l1371_137186

/-- A function f: ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The cubic function f(x) = ax³ - x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x + 1

theorem decreasing_cubic_implies_nonpositive_a :
  ∀ a : ℝ, DecreasingFunction (f a) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_cubic_implies_nonpositive_a_l1371_137186


namespace NUMINAMATH_CALUDE_zoey_finishes_on_friday_l1371_137103

def days_to_read (n : ℕ) : ℕ := n + 1

def total_days (sets : ℕ) : ℕ :=
  (List.range sets).map (λ i => days_to_read (i + 1)) |>.sum

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem zoey_finishes_on_friday :
  let sets := 8
  let start_day := 3  -- Wednesday (0 = Sunday, 1 = Monday, ..., 6 = Saturday)
  day_of_week start_day (total_days sets) = 5  -- Friday
  := by sorry

end NUMINAMATH_CALUDE_zoey_finishes_on_friday_l1371_137103


namespace NUMINAMATH_CALUDE_power_equality_l1371_137177

theorem power_equality : (243 : ℕ)^4 = 3^12 * 3^8 := by sorry

end NUMINAMATH_CALUDE_power_equality_l1371_137177


namespace NUMINAMATH_CALUDE_investment_time_calculation_l1371_137125

/-- Investment time calculation for partners P and Q -/
theorem investment_time_calculation 
  (investment_ratio_p : ℝ) 
  (investment_ratio_q : ℝ)
  (profit_ratio_p : ℝ) 
  (profit_ratio_q : ℝ)
  (time_q : ℝ) :
  investment_ratio_p = 7 →
  investment_ratio_q = 5.00001 →
  profit_ratio_p = 7.00001 →
  profit_ratio_q = 10 →
  time_q = 9.999965714374696 →
  ∃ (time_p : ℝ), abs (time_p - 50) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_investment_time_calculation_l1371_137125


namespace NUMINAMATH_CALUDE_distance_AE_BF_is_19_2_l1371_137181

/-- A rectangular parallelepiped with given dimensions and midpoints -/
structure Parallelepiped where
  -- Edge lengths
  ab : ℝ
  ad : ℝ
  aa1 : ℝ
  -- Ensure it's a rectangular parallelepiped
  is_rectangular : True
  -- Ensure E is midpoint of A₁B₁
  e_is_midpoint_a1b1 : True
  -- Ensure F is midpoint of B₁C₁
  f_is_midpoint_b1c1 : True

/-- The distance between lines AE and BF in the parallelepiped -/
def distance_AE_BF (p : Parallelepiped) : ℝ := sorry

/-- Theorem: The distance between AE and BF is 19.2 -/
theorem distance_AE_BF_is_19_2 (p : Parallelepiped) 
  (h1 : p.ab = 30) (h2 : p.ad = 32) (h3 : p.aa1 = 20) : 
  distance_AE_BF p = 19.2 := by sorry

end NUMINAMATH_CALUDE_distance_AE_BF_is_19_2_l1371_137181


namespace NUMINAMATH_CALUDE_right_tangential_trapezoid_shorter_leg_l1371_137113

/-- In a right tangential trapezoid, the shorter leg equals 2ac/(a+c) where a and c are the lengths of the bases. -/
theorem right_tangential_trapezoid_shorter_leg
  (a c d : ℝ)
  (h_positive : a > 0 ∧ c > 0 ∧ d > 0)
  (h_right_tangential : d^2 + (a - c)^2 = (a + c - d)^2)
  (h_shorter_leg : d ≤ a + c - d) :
  d = 2 * a * c / (a + c) := by
sorry

end NUMINAMATH_CALUDE_right_tangential_trapezoid_shorter_leg_l1371_137113


namespace NUMINAMATH_CALUDE_cannot_obtain_703_from_604_l1371_137184

/-- Represents the computer operations -/
inductive Operation
  | square : Operation
  | split : Operation

/-- Applies the given operation to a natural number -/
def apply_operation (op : Operation) (n : ℕ) : ℕ :=
  match op with
  | Operation.square => n * n
  | Operation.split => 
      if n < 1000 then n
      else (n % 1000) + (n / 1000)

/-- Checks if it's possible to transform start into target using the given operations -/
def can_transform (start target : ℕ) : Prop :=
  ∃ (seq : List Operation), 
    (seq.foldl (λ acc op => apply_operation op acc) start) = target

/-- The main theorem stating that 703 cannot be obtained from 604 using the given operations -/
theorem cannot_obtain_703_from_604 : ¬ can_transform 604 703 := by
  sorry


end NUMINAMATH_CALUDE_cannot_obtain_703_from_604_l1371_137184


namespace NUMINAMATH_CALUDE_equation_solutions_l1371_137121

theorem equation_solutions :
  (∃ x : ℚ, (5*x - 1)/4 = (3*x + 1)/2 - (2 - x)/3 ↔ x = -1/7) ∧
  (∃ x : ℚ, (3*x + 2)/2 - 1 = (2*x - 1)/4 - (2*x + 1)/5 ↔ x = -9/28) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1371_137121


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1371_137163

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1371_137163


namespace NUMINAMATH_CALUDE_heptagon_angle_sums_l1371_137115

/-- A heptagon is a polygon with 7 sides -/
def Heptagon : Nat := 7

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : Nat) : ℝ := (n - 2) * 180

/-- The sum of exterior angles of any polygon -/
def sum_exterior_angles : ℝ := 360

theorem heptagon_angle_sums :
  (sum_interior_angles Heptagon = 900) ∧ (sum_exterior_angles = 360) := by
  sorry

#check heptagon_angle_sums

end NUMINAMATH_CALUDE_heptagon_angle_sums_l1371_137115


namespace NUMINAMATH_CALUDE_matrix_N_properties_l1371_137185

def N : Matrix (Fin 2) (Fin 2) ℚ := !![5, -3; -1/2, 2]

theorem matrix_N_properties :
  let v1 : Matrix (Fin 2) (Fin 1) ℚ := !![2; -1]
  let v2 : Matrix (Fin 2) (Fin 1) ℚ := !![0; 3]
  let r1 : Matrix (Fin 2) (Fin 1) ℚ := !![5; -3]
  let r2 : Matrix (Fin 2) (Fin 1) ℚ := !![-9; 6]
  (N * v1 = r1) ∧
  (N * v2 = r2) ∧
  (N 0 0 - N 1 1 = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_matrix_N_properties_l1371_137185


namespace NUMINAMATH_CALUDE_jason_balloons_l1371_137148

/-- Given an initial number of violet balloons and a number of lost violet balloons,
    calculate the remaining number of violet balloons. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Jason's remaining violet balloons is 4,
    given he started with 7 and lost 3. -/
theorem jason_balloons : remaining_balloons 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_balloons_l1371_137148


namespace NUMINAMATH_CALUDE_simplify_expression_l1371_137123

theorem simplify_expression (x : ℝ) : (2 * x + 30) + (150 * x + 45) + 5 = 152 * x + 80 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1371_137123


namespace NUMINAMATH_CALUDE_not_always_greater_quotient_l1371_137157

theorem not_always_greater_quotient : ¬ ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → (∃ n : ℤ, b = n / 10) → a / b > a := by
  sorry

end NUMINAMATH_CALUDE_not_always_greater_quotient_l1371_137157


namespace NUMINAMATH_CALUDE_a_minus_b_equals_negative_nine_l1371_137101

theorem a_minus_b_equals_negative_nine
  (a b : ℝ)
  (h : |a + 5| + Real.sqrt (2 * b - 8) = 0) :
  a - b = -9 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_negative_nine_l1371_137101


namespace NUMINAMATH_CALUDE_equation_one_solution_l1371_137183

theorem equation_one_solution : 
  {x : ℝ | (x + 3)^2 - 9 = 0} = {0, -6} := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_l1371_137183


namespace NUMINAMATH_CALUDE_marys_potatoes_l1371_137153

def potatoes_problem (initial_potatoes : ℕ) (eaten_potatoes : ℕ) : Prop :=
  initial_potatoes - eaten_potatoes = 5

theorem marys_potatoes : potatoes_problem 8 3 :=
sorry

end NUMINAMATH_CALUDE_marys_potatoes_l1371_137153


namespace NUMINAMATH_CALUDE_lloyd_earnings_correct_l1371_137171

/-- Calculates Lloyd's earnings for the given work days -/
def lloyd_earnings (regular_rate : ℚ) (normal_hours : ℚ) (overtime_rate : ℚ) (saturday_rate : ℚ) 
  (monday_hours : ℚ) (tuesday_hours : ℚ) (saturday_hours : ℚ) : ℚ :=
  let monday_earnings := 
    min normal_hours monday_hours * regular_rate + 
    max 0 (monday_hours - normal_hours) * regular_rate * overtime_rate
  let tuesday_earnings := 
    min normal_hours tuesday_hours * regular_rate + 
    max 0 (tuesday_hours - normal_hours) * regular_rate * overtime_rate
  let saturday_earnings := saturday_hours * regular_rate * saturday_rate
  monday_earnings + tuesday_earnings + saturday_earnings

theorem lloyd_earnings_correct : 
  lloyd_earnings 5 8 (3/2) 2 (21/2) 9 6 = 665/4 := by
  sorry

end NUMINAMATH_CALUDE_lloyd_earnings_correct_l1371_137171


namespace NUMINAMATH_CALUDE_middle_number_not_unique_l1371_137136

/-- Represents a configuration of three cards with positive integers. -/
structure CardConfiguration where
  left : Nat
  middle : Nat
  right : Nat
  sum_is_15 : left + middle + right = 15
  increasing : left < middle ∧ middle < right

/-- Predicate to check if Alan can determine the other two numbers. -/
def alan_cant_determine (config : CardConfiguration) : Prop :=
  ∃ (other_config : CardConfiguration), other_config.left = config.left ∧ other_config ≠ config

/-- Predicate to check if Carlos can determine the other two numbers. -/
def carlos_cant_determine (config : CardConfiguration) : Prop :=
  ∃ (other_config : CardConfiguration), other_config.right = config.right ∧ other_config ≠ config

/-- Predicate to check if Brenda can determine the other two numbers. -/
def brenda_cant_determine (config : CardConfiguration) : Prop :=
  ∃ (other_config : CardConfiguration), other_config.middle = config.middle ∧ other_config ≠ config

/-- The main theorem stating that the middle number cannot be uniquely determined. -/
theorem middle_number_not_unique : ∃ (config1 config2 : CardConfiguration),
  config1.middle ≠ config2.middle ∧
  alan_cant_determine config1 ∧
  alan_cant_determine config2 ∧
  carlos_cant_determine config1 ∧
  carlos_cant_determine config2 ∧
  brenda_cant_determine config1 ∧
  brenda_cant_determine config2 :=
sorry

end NUMINAMATH_CALUDE_middle_number_not_unique_l1371_137136


namespace NUMINAMATH_CALUDE_obtain_a_to_six_l1371_137143

/-- Given a^4 and a^6 - 1, prove that a^6 can be obtained using +, -, and · operations -/
theorem obtain_a_to_six (a : ℝ) : ∃ f : ℝ → ℝ → ℝ → ℝ, 
  f (a^4) (a^6 - 1) 1 = a^6 ∧ 
  (∀ x y z, f x y z = x + y ∨ f x y z = x - y ∨ f x y z = x * y ∨ 
            f x y z = y + z ∨ f x y z = y - z ∨ f x y z = y * z ∨
            f x y z = z + x ∨ f x y z = z - x ∨ f x y z = z * x) :=
by
  sorry

end NUMINAMATH_CALUDE_obtain_a_to_six_l1371_137143


namespace NUMINAMATH_CALUDE_jane_rejection_proof_l1371_137172

/-- Represents the percentage of products Jane rejected -/
def jane_rejection_rate : ℝ := 0.9

/-- Represents the percentage of products John rejected -/
def john_rejection_rate : ℝ := 0.5

/-- Represents the fraction of total products Jane inspected -/
def jane_inspection_fraction : ℝ := 0.625

/-- Represents the total percentage of rejected products -/
def total_rejection_rate : ℝ := 0.75

/-- Theorem stating that given the conditions, Jane's rejection rate is 0.9% -/
theorem jane_rejection_proof :
  john_rejection_rate * (1 - jane_inspection_fraction) +
  jane_rejection_rate * jane_inspection_fraction / 100 =
  total_rejection_rate / 100 := by
  sorry

#check jane_rejection_proof

end NUMINAMATH_CALUDE_jane_rejection_proof_l1371_137172


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1371_137187

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | x > -6 - 2*x ∧ x ≤ (3 + x) / 4}
  S = {x | -2 < x ∧ x ≤ 1} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1371_137187


namespace NUMINAMATH_CALUDE_cube_division_l1371_137151

theorem cube_division (n : ℕ) (small_edge : ℝ) : 
  12 / n = small_edge ∧ 
  n * 6 * small_edge^2 = 8 * 6 * 12^2 → 
  n^3 = 512 ∧ small_edge = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_cube_division_l1371_137151


namespace NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l1371_137160

theorem triangle_perimeter_impossibility (a b x : ℝ) : 
  a = 20 → b = 15 → x > 0 → a + b + x ≠ 72 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l1371_137160


namespace NUMINAMATH_CALUDE_junk_items_remaining_l1371_137104

/-- Represents the distribution of items in Mason's attic -/
structure AtticItems where
  useful : ℕ
  valuable : ℕ
  junk : ℕ

/-- Calculates the total number of items in the attic -/
def AtticItems.total (items : AtticItems) : ℕ := items.useful + items.valuable + items.junk

/-- The initial distribution of items in the attic -/
def initial_distribution : AtticItems := {
  useful := 20,
  valuable := 10,
  junk := 70
}

/-- The number of useful items given away -/
def useful_items_given_away : ℕ := 4

/-- The number of valuable items sold -/
def valuable_items_sold : ℕ := 20

/-- The number of useful items remaining after giving some away -/
def remaining_useful_items : ℕ := 16

/-- Theorem stating the number of junk items remaining in the attic -/
theorem junk_items_remaining (h1 : initial_distribution.total = 100)
  (h2 : remaining_useful_items = initial_distribution.useful - useful_items_given_away)
  (h3 : valuable_items_sold > initial_distribution.valuable) :
  initial_distribution.junk - (valuable_items_sold - initial_distribution.valuable) = 60 := by
  sorry


end NUMINAMATH_CALUDE_junk_items_remaining_l1371_137104


namespace NUMINAMATH_CALUDE_median_of_special_arithmetic_sequence_l1371_137156

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h1 : d ≠ 0 -- Non-zero common difference
  h2 : ∀ n, a (n + 1) = a n + d -- Arithmetic sequence property
  h3 : a 3 = 8 -- Third term is 8
  h4 : ∃ r, r ≠ 0 ∧ a 1 * r = a 3 ∧ a 3 * r = a 7 -- Geometric sequence property for a₁, a₃, a₇

/-- The median of a 9-term arithmetic sequence with specific properties is 24 -/
theorem median_of_special_arithmetic_sequence (seq : ArithmeticSequence) : 
  seq.a 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_median_of_special_arithmetic_sequence_l1371_137156


namespace NUMINAMATH_CALUDE_salary_B_is_5000_l1371_137139

/-- Calculates the salary of person B given the salaries of other people and the average salary -/
def calculate_salary_B (salary_A salary_C salary_D salary_E average_salary : ℕ) : ℕ :=
  5 * average_salary - (salary_A + salary_C + salary_D + salary_E)

/-- Proves that B's salary is 5000 given the conditions in the problem -/
theorem salary_B_is_5000 :
  let salary_A : ℕ := 8000
  let salary_C : ℕ := 11000
  let salary_D : ℕ := 7000
  let salary_E : ℕ := 9000
  let average_salary : ℕ := 8000
  calculate_salary_B salary_A salary_C salary_D salary_E average_salary = 5000 := by
  sorry

#eval calculate_salary_B 8000 11000 7000 9000 8000

end NUMINAMATH_CALUDE_salary_B_is_5000_l1371_137139


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l1371_137129

def total_budget : ℝ := 60
def hummus_cost : ℝ := 5
def hummus_quantity : ℕ := 2
def chicken_cost : ℝ := 20
def bacon_cost : ℝ := 10
def vegetables_cost : ℝ := 10
def apple_quantity : ℕ := 5

theorem apple_cost_calculation :
  let other_items_cost := hummus_cost * hummus_quantity + chicken_cost + bacon_cost + vegetables_cost
  let remaining_budget := total_budget - other_items_cost
  remaining_budget / apple_quantity = 2 := by sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l1371_137129


namespace NUMINAMATH_CALUDE_gcd_420_135_l1371_137132

theorem gcd_420_135 : Nat.gcd 420 135 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_420_135_l1371_137132


namespace NUMINAMATH_CALUDE_vector_decomposition_l1371_137130

def x : Fin 3 → ℝ := ![6, -1, 7]
def p : Fin 3 → ℝ := ![1, -2, 0]
def q : Fin 3 → ℝ := ![-1, 1, 3]
def r : Fin 3 → ℝ := ![1, 0, 4]

theorem vector_decomposition :
  x = λ i => -p i - 3 * q i + 4 * r i :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l1371_137130


namespace NUMINAMATH_CALUDE_square_in_right_triangle_l1371_137108

/-- Given a right triangle PQR with PQ = 9, PR = 12, and right angle at P,
    if a square is fitted with one side on the hypotenuse QR and other vertices
    touching the legs of the triangle, then the length of the square's side is 3. -/
theorem square_in_right_triangle (P Q R : ℝ × ℝ) (s : ℝ) :
  let pq : ℝ := 9
  let pr : ℝ := 12
  -- P is the origin (0, 0)
  P = (0, 0) →
  -- Q is on the x-axis
  Q.2 = 0 →
  -- R is on the y-axis
  R.1 = 0 →
  -- PQ = 9
  Q.1 = pq →
  -- PR = 12
  R.2 = pr →
  -- s is positive
  s > 0 →
  -- One vertex of the square is on QR
  ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ x + y = s ∧ x^2 + y^2 = (pq - s)^2 + (pr - s)^2 →
  -- The square's side length is 3
  s = 3 :=
by sorry

end NUMINAMATH_CALUDE_square_in_right_triangle_l1371_137108


namespace NUMINAMATH_CALUDE_fifth_employee_speed_correct_l1371_137116

/-- Calculates the typing speed of the 5th employee given the team's average and the typing speeds of the other 4 employees. -/
def calculate_fifth_employee_speed (team_size : Nat) (team_average : Nat) (employee1_speed : Nat) (employee2_speed : Nat) (employee3_speed : Nat) (employee4_speed : Nat) : Nat :=
  team_size * team_average - (employee1_speed + employee2_speed + employee3_speed + employee4_speed)

/-- Theorem stating that the calculated speed of the 5th employee is correct given the team's average and the speeds of the other 4 employees. -/
theorem fifth_employee_speed_correct (team_average : Nat) (employee1_speed : Nat) (employee2_speed : Nat) (employee3_speed : Nat) (employee4_speed : Nat) :
  let team_size : Nat := 5
  let fifth_employee_speed := calculate_fifth_employee_speed team_size team_average employee1_speed employee2_speed employee3_speed employee4_speed
  (employee1_speed + employee2_speed + employee3_speed + employee4_speed + fifth_employee_speed) / team_size = team_average :=
by
  sorry

#eval calculate_fifth_employee_speed 5 80 64 76 91 80

end NUMINAMATH_CALUDE_fifth_employee_speed_correct_l1371_137116


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l1371_137140

/-- The distance between Town X and Town Y in miles -/
def distance : ℝ := 90

/-- The speed difference between cyclists D and C in miles per hour -/
def speed_difference : ℝ := 5

/-- The distance from Town Y where cyclists C and D meet on D's return trip in miles -/
def meeting_point : ℝ := 15

/-- The speed of Cyclist C in miles per hour -/
def speed_C : ℝ := 12.5

theorem cyclist_speed_proof :
  ∃ (speed_D : ℝ),
    speed_D = speed_C + speed_difference ∧
    distance / speed_C = (distance + meeting_point) / speed_D :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l1371_137140


namespace NUMINAMATH_CALUDE_calculation_proof_l1371_137162

theorem calculation_proof : (0.8 * 60 - 2/5 * 35) * Real.sqrt 144 = 408 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1371_137162


namespace NUMINAMATH_CALUDE_line_equation_60_degrees_l1371_137133

/-- The equation of a line with a slope of 60° and a y-intercept of -1 -/
theorem line_equation_60_degrees (x y : ℝ) :
  let slope : ℝ := Real.tan (60 * π / 180)
  let y_intercept : ℝ := -1
  slope * x - y - y_intercept = 0 ↔ Real.sqrt 3 * x - y - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_60_degrees_l1371_137133


namespace NUMINAMATH_CALUDE_chord_equation_l1371_137152

theorem chord_equation (x y : ℝ) :
  (x^2 + y^2 - 2*x = 0) →  -- Circle equation
  (∃ (t : ℝ), x = 1/2 + t ∧ y = 1/2 + t) →  -- Midpoint condition
  (x - y = 0) :=  -- Line equation
by sorry

end NUMINAMATH_CALUDE_chord_equation_l1371_137152


namespace NUMINAMATH_CALUDE_cyclic_triples_count_l1371_137106

/-- Represents a round-robin tournament -/
structure Tournament where
  n : ℕ  -- number of teams
  wins : Fin n → ℕ  -- number of wins for each team
  losses : Fin n → ℕ  -- number of losses for each team

/-- The number of sets of three teams with a cyclic winning relationship -/
def cyclic_triples (t : Tournament) : ℕ := sorry

/-- Main theorem about the number of cyclic triples in the specific tournament -/
theorem cyclic_triples_count (t : Tournament) 
  (h1 : t.n > 0)
  (h2 : ∀ i : Fin t.n, t.wins i = 9)
  (h3 : ∀ i : Fin t.n, t.losses i = 9)
  (h4 : ∀ i j : Fin t.n, i ≠ j → (t.wins i + t.losses i = t.wins j + t.losses j)) :
  cyclic_triples t = 969 := by sorry

end NUMINAMATH_CALUDE_cyclic_triples_count_l1371_137106


namespace NUMINAMATH_CALUDE_max_value_of_f_l1371_137112

/-- The quadratic function f(x) = -2x^2 + 9 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 9

/-- Theorem: The maximum value of f(x) = -2x^2 + 9 is 9 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1371_137112


namespace NUMINAMATH_CALUDE_comparison_and_inequality_l1371_137128

theorem comparison_and_inequality (x y m : ℝ) 
  (h1 : x > y) (h2 : y > 0) (h3 : m > 0) : 
  y / x < (y + m) / (x + m) ∧ Real.sqrt (x * y) * (2 - Real.sqrt (x * y)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_comparison_and_inequality_l1371_137128


namespace NUMINAMATH_CALUDE_inequality_proof_l1371_137150

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  1 / (x^3 * y) + 1 / (y^3 * z) + 1 / (z^3 * x) ≥ x * y + y * z + z * x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1371_137150


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l1371_137188

def income : ℕ := 10000
def savings : ℕ := 3000
def expenditure : ℕ := income - savings

theorem income_expenditure_ratio :
  (income : ℚ) / (expenditure : ℚ) = 10 / 7 := by sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l1371_137188


namespace NUMINAMATH_CALUDE_brenda_money_brenda_money_proof_l1371_137190

/-- Proof that Brenda has 8 dollars given the conditions about Emma, Daya, Jeff, and Brenda's money. -/
theorem brenda_money : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun emma daya jeff brenda =>
    emma = 8 ∧
    daya = emma * 1.25 ∧
    jeff = daya * (2/5) ∧
    brenda = jeff + 4 →
    brenda = 8

-- Proof
theorem brenda_money_proof : brenda_money 8 10 4 8 := by
  sorry

end NUMINAMATH_CALUDE_brenda_money_brenda_money_proof_l1371_137190


namespace NUMINAMATH_CALUDE_five_digit_palindromes_count_l1371_137119

/-- A five-digit palindromic number -/
def FiveDigitPalindrome (a b c : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * c + 10 * b + a

/-- The count of five-digit palindromic numbers -/
def CountFiveDigitPalindromes : ℕ := 90

theorem five_digit_palindromes_count :
  (∀ n : ℕ, 10000 ≤ n ∧ n < 100000 →
    (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ n = FiveDigitPalindrome a b c) ↔
    (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ n = 10000 * a + 1000 * b + 100 * c + 10 * b + a)) →
  CountFiveDigitPalindromes = (9 : ℕ) * (10 : ℕ) * (1 : ℕ) :=
sorry

end NUMINAMATH_CALUDE_five_digit_palindromes_count_l1371_137119


namespace NUMINAMATH_CALUDE_smallest_integer_bound_l1371_137199

theorem smallest_integer_bound (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d) / 4 = 74 →
  d = 90 →
  max a (max b c) ≤ d →
  min a (min b c) ≥ 29 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_bound_l1371_137199


namespace NUMINAMATH_CALUDE_tank_b_circumference_l1371_137196

/-- The circumference of Tank B given the conditions of the problem -/
theorem tank_b_circumference : 
  ∀ (h_a h_b c_a c_b r_a r_b v_a v_b : ℝ),
  h_a = 7 →
  h_b = 8 →
  c_a = 8 →
  c_a = 2 * Real.pi * r_a →
  v_a = Real.pi * r_a^2 * h_a →
  v_b = Real.pi * r_b^2 * h_b →
  v_a = 0.5600000000000001 * v_b →
  c_b = 2 * Real.pi * r_b →
  c_b = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_b_circumference_l1371_137196


namespace NUMINAMATH_CALUDE_smallest_multiple_of_84_with_6_and_7_l1371_137126

def is_multiple_of_84 (n : ℕ) : Prop := n % 84 = 0

def contains_only_6_and_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 6 ∨ d = 7

theorem smallest_multiple_of_84_with_6_and_7 :
  (is_multiple_of_84 76776) ∧
  (contains_only_6_and_7 76776) ∧
  (∀ n : ℕ, n < 76776 → ¬(is_multiple_of_84 n ∧ contains_only_6_and_7 n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_84_with_6_and_7_l1371_137126


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1371_137158

/-- Calculates the total cost of beef and vegetables -/
theorem total_cost_calculation (beef_weight : ℝ) (veg_weight : ℝ) (veg_price : ℝ) :
  beef_weight = 4 →
  veg_weight = 6 →
  veg_price = 2 →
  beef_weight * (3 * veg_price) + veg_weight * veg_price = 36 := by
  sorry

#check total_cost_calculation

end NUMINAMATH_CALUDE_total_cost_calculation_l1371_137158


namespace NUMINAMATH_CALUDE_project_completion_time_l1371_137149

/-- The number of days it takes B to complete the project alone -/
def B_days : ℝ := 30

/-- The number of days A and B work together -/
def AB_work_days : ℝ := 10

/-- The number of days B works alone after A quits -/
def B_alone_days : ℝ := 5

/-- The number of days it takes A to complete the project alone -/
def A_days : ℝ := 20

theorem project_completion_time :
  (AB_work_days * (1 / A_days + 1 / B_days) + B_alone_days * (1 / B_days)) = 1 :=
sorry

end NUMINAMATH_CALUDE_project_completion_time_l1371_137149


namespace NUMINAMATH_CALUDE_divisibility_property_l1371_137159

theorem divisibility_property (p n q : ℕ) : 
  Prime p → 
  n > 0 → 
  q > 0 → 
  q ∣ ((n + 1)^p - n^p) → 
  p ∣ (q - 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l1371_137159


namespace NUMINAMATH_CALUDE_dance_step_ratio_l1371_137167

theorem dance_step_ratio : 
  ∀ (N J : ℕ),
  (∃ (k : ℕ), N = k * J) →  -- Nancy steps k times as often as Jason
  N + J = 32 →              -- Total steps
  J = 8 →                   -- Jason's steps
  N / J = 3 :=              -- Ratio of Nancy's to Jason's steps
by sorry

end NUMINAMATH_CALUDE_dance_step_ratio_l1371_137167


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_and_parabola_l1371_137182

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 2

/-- Parabola C₂ -/
def C₂ (x y : ℝ) : Prop := y^2 = 4*x

/-- Line l -/
def l (x y : ℝ) : Prop := y = Real.sqrt 2

theorem line_tangent_to_circle_and_parabola :
  ∃! p : ℝ × ℝ, C₁ p.1 p.2 ∧ l p.1 p.2 ∧
  ∃! q : ℝ × ℝ, C₂ q.1 q.2 ∧ l q.1 q.2 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_and_parabola_l1371_137182


namespace NUMINAMATH_CALUDE_cube_root_product_l1371_137102

theorem cube_root_product : (4^9 * 5^6 * 7^3 : ℝ)^(1/3) = 11200 := by sorry

end NUMINAMATH_CALUDE_cube_root_product_l1371_137102


namespace NUMINAMATH_CALUDE_vectors_are_coplanar_l1371_137193

def a : ℝ × ℝ × ℝ := (1, -2, 6)
def b : ℝ × ℝ × ℝ := (1, 0, 1)
def c : ℝ × ℝ × ℝ := (2, -6, 17)

def coplanar (u v w : ℝ × ℝ × ℝ) : Prop :=
  let (u₁, u₂, u₃) := u
  let (v₁, v₂, v₃) := v
  let (w₁, w₂, w₃) := w
  (u₁ * (v₂ * w₃ - v₃ * w₂) - u₂ * (v₁ * w₃ - v₃ * w₁) + u₃ * (v₁ * w₂ - v₂ * w₁)) = 0

theorem vectors_are_coplanar : coplanar a b c := by
  sorry

end NUMINAMATH_CALUDE_vectors_are_coplanar_l1371_137193


namespace NUMINAMATH_CALUDE_least_subtrahend_l1371_137145

def is_valid (x : ℕ) : Prop :=
  (997 - x) % 5 = 3 ∧ (997 - x) % 9 = 3 ∧ (997 - x) % 11 = 3

theorem least_subtrahend :
  ∃ (x : ℕ), is_valid x ∧ ∀ (y : ℕ), y < x → ¬is_valid y :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_l1371_137145


namespace NUMINAMATH_CALUDE_general_term_correct_l1371_137191

/-- A geometric sequence with given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  S : ℕ → ℚ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  sum_formula : ∀ n : ℕ, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))
  S_3 : S 3 = 13/9
  S_6 : S 6 = 364/9

/-- The general term of the geometric sequence -/
def general_term (seq : GeometricSequence) (n : ℕ) : ℚ :=
  (1/6) * 3^(n-1)

/-- Theorem stating that the general term is correct -/
theorem general_term_correct (seq : GeometricSequence) :
  ∀ n : ℕ, seq.a n = general_term seq n :=
sorry

end NUMINAMATH_CALUDE_general_term_correct_l1371_137191


namespace NUMINAMATH_CALUDE_complex_magnitude_l1371_137144

theorem complex_magnitude (z : ℂ) : z + 2 * Complex.I = (3 - Complex.I ^ 3) / (1 + Complex.I) → Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1371_137144


namespace NUMINAMATH_CALUDE_honey_production_l1371_137179

/-- The amount of honey (in grams) produced by a single bee in 60 days -/
def single_bee_honey : ℕ := 1

/-- The number of bees in the group -/
def num_bees : ℕ := 60

/-- The amount of honey (in grams) produced by the group of bees in 60 days -/
def group_honey : ℕ := num_bees * single_bee_honey

theorem honey_production :
  group_honey = 60 := by sorry

end NUMINAMATH_CALUDE_honey_production_l1371_137179


namespace NUMINAMATH_CALUDE_domino_placement_l1371_137173

/-- The maximum number of 1 × k dominos that can be placed on an n × n chessboard. -/
def max_dominos (n k : ℕ) : ℕ :=
  if n = k ∨ n = 2*k - 1 then n
  else if k < n ∧ n < 2*k - 1 then 2*n - 2*k + 2
  else 0

theorem domino_placement (n k : ℕ) (h1 : k ≤ n) (h2 : n < 2*k) :
  max_dominos n k = if n = k ∨ n = 2*k - 1 then n else 2*n - 2*k + 2 :=
by sorry

end NUMINAMATH_CALUDE_domino_placement_l1371_137173


namespace NUMINAMATH_CALUDE_function_and_composition_proof_l1371_137110

def f (x b c : ℝ) : ℝ := x^2 - b*x + c

theorem function_and_composition_proof 
  (b c : ℝ) 
  (h1 : f 1 b c = 0) 
  (h2 : f 2 b c = -3) :
  (∀ x, f x b c = x^2 - 6*x + 5) ∧ 
  (∀ x, x > -1 → f (1 / Real.sqrt (x + 1)) b c = 1 / (x + 1) - 6 / Real.sqrt (x + 1) + 5) :=
by sorry

end NUMINAMATH_CALUDE_function_and_composition_proof_l1371_137110


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l1371_137197

theorem parallelogram_side_sum (x y : ℝ) : 
  (12 * y - 2 = 10) → (5 * x + 15 = 20) → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l1371_137197


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1371_137192

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I)*z = 4 + 3*I) : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1371_137192


namespace NUMINAMATH_CALUDE_complex_set_equals_zero_two_neg_two_l1371_137109

def complex_set : Set ℂ := {z | ∃ n : ℤ, z = Complex.I ^ n + Complex.I ^ (-n)}

theorem complex_set_equals_zero_two_neg_two : 
  complex_set = {0, 2, -2} :=
sorry

end NUMINAMATH_CALUDE_complex_set_equals_zero_two_neg_two_l1371_137109


namespace NUMINAMATH_CALUDE_no_real_solutions_for_complex_product_l1371_137195

theorem no_real_solutions_for_complex_product : 
  ¬∃ (x : ℝ), (Complex.I : ℂ).im * ((x + 2 + Complex.I) * ((x + 3) + 2 * Complex.I) * ((x + 4) + Complex.I)).im = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_complex_product_l1371_137195


namespace NUMINAMATH_CALUDE_multiple_properties_l1371_137154

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a - b = 4 * p) ∧ 
  (∃ q : ℤ, a - b = 2 * q) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l1371_137154


namespace NUMINAMATH_CALUDE_quadratic_discriminant_theorem_l1371_137124

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The discriminant of a quadratic polynomial -/
def discriminant (p : QuadraticPolynomial) : ℝ := p.b^2 - 4*p.a*p.c

/-- A function to check if a quadratic equation has exactly one root -/
def has_one_root (a b c : ℝ) : Prop := (b^2 - 4*a*c = 0)

theorem quadratic_discriminant_theorem (p : QuadraticPolynomial) 
  (h1 : has_one_root p.a p.b (p.c + 2))
  (h2 : has_one_root p.a (p.b + 1/2) (p.c - 1)) :
  discriminant p = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_theorem_l1371_137124


namespace NUMINAMATH_CALUDE_paving_stone_width_l1371_137137

/-- Given a rectangular courtyard and paving stones with specific dimensions,
    prove that the width of each paving stone is 1 meter. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (stone_count : ℕ)
  (h1 : courtyard_length = 30)
  (h2 : courtyard_width = 16)
  (h3 : stone_length = 2)
  (h4 : stone_count = 240)
  : ∃ (stone_width : ℝ), 
    stone_width = 1 ∧ 
    courtyard_length * courtyard_width = ↑stone_count * stone_length * stone_width :=
by sorry

end NUMINAMATH_CALUDE_paving_stone_width_l1371_137137


namespace NUMINAMATH_CALUDE_parabola_tangent_intersection_l1371_137114

/-- Proves that for a parabola y^2 = 2px, the intersection point of two tangent lines
    has a y-coordinate equal to the average of the y-coordinates of the tangent points. -/
theorem parabola_tangent_intersection
  (p : ℝ) (x₁ y₁ x₂ y₂ x y : ℝ)
  (h_parabola₁ : y₁^2 = 2*p*x₁)
  (h_parabola₂ : y₂^2 = 2*p*x₂)
  (h_tangent₁ : y*y₁ = p*(x + x₁))
  (h_tangent₂ : y*y₂ = p*(x + x₂))
  (h_distinct : y₁ ≠ y₂) :
  y = (y₁ + y₂) / 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_intersection_l1371_137114


namespace NUMINAMATH_CALUDE_solution_implies_sum_equals_four_l1371_137166

-- Define the operation ⊗
def otimes (x y : ℝ) := x * (1 - y)

-- Define the theorem
theorem solution_implies_sum_equals_four 
  (h : ∀ x : ℝ, (otimes (x - a) (x - b) > 0) ↔ (2 < x ∧ x < 3)) :
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_sum_equals_four_l1371_137166


namespace NUMINAMATH_CALUDE_original_equals_scientific_l1371_137175

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 284000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { significand := 2.84
    exponent := 8
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.significand * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l1371_137175


namespace NUMINAMATH_CALUDE_combined_cost_theorem_l1371_137131

/-- Calculate the total cost of a stock given its face value, discount/premium rate, and brokerage rate -/
def stockCost (faceValue : ℚ) (discountPremiumRate : ℚ) (brokerageRate : ℚ) : ℚ :=
  let adjustedValue := faceValue * (1 + discountPremiumRate)
  adjustedValue * (1 + brokerageRate)

/-- The combined cost of stocks A, B, and C -/
def combinedCost : ℚ :=
  stockCost 100 (-0.02) (1/500) +  -- Stock A
  stockCost 150 0.015 (1/600) +    -- Stock B
  stockCost 200 (-0.03) (1/200)    -- Stock C

theorem combined_cost_theorem :
  combinedCost = 445669750/1000000 := by sorry

end NUMINAMATH_CALUDE_combined_cost_theorem_l1371_137131


namespace NUMINAMATH_CALUDE_correct_sampling_order_l1371_137164

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Stratified
| Systematic

-- Define the characteristics of each scenario
structure Scenario where
  population_size : ℕ
  has_subgroups : Bool
  has_orderly_numbering : Bool

-- Define the three given scenarios
def scenario1 : Scenario := ⟨8, false, false⟩
def scenario2 : Scenario := ⟨2100, true, false⟩
def scenario3 : Scenario := ⟨700, false, true⟩

-- Function to determine the most appropriate sampling method for a given scenario
def appropriate_method (s : Scenario) : SamplingMethod :=
  if s.population_size ≤ 10 && !s.has_subgroups && !s.has_orderly_numbering then
    SamplingMethod.SimpleRandom
  else if s.has_subgroups then
    SamplingMethod.Stratified
  else if s.has_orderly_numbering then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom

-- Theorem stating that the given order of sampling methods is correct for the three scenarios
theorem correct_sampling_order :
  (appropriate_method scenario1 = SamplingMethod.SimpleRandom) ∧
  (appropriate_method scenario2 = SamplingMethod.Stratified) ∧
  (appropriate_method scenario3 = SamplingMethod.Systematic) := by
  sorry

end NUMINAMATH_CALUDE_correct_sampling_order_l1371_137164


namespace NUMINAMATH_CALUDE_genevieve_errors_fixed_l1371_137180

/-- Represents a programmer's coding and debugging process -/
structure Programmer where
  total_lines : ℕ
  debug_interval : ℕ
  errors_per_debug : ℕ

/-- Calculates the total number of errors fixed by a programmer -/
def total_errors_fixed (p : Programmer) : ℕ :=
  (p.total_lines / p.debug_interval) * p.errors_per_debug

/-- Theorem stating that under given conditions, the programmer fixes 129 errors -/
theorem genevieve_errors_fixed :
  ∀ (p : Programmer),
    p.total_lines = 4300 →
    p.debug_interval = 100 →
    p.errors_per_debug = 3 →
    total_errors_fixed p = 129 := by
  sorry


end NUMINAMATH_CALUDE_genevieve_errors_fixed_l1371_137180


namespace NUMINAMATH_CALUDE_basketball_tournament_l1371_137120

/-- Number of classes in the tournament -/
def num_classes : ℕ := 10

/-- Total number of matches in the tournament -/
def total_matches : ℕ := 45

/-- Points earned for winning a game -/
def win_points : ℕ := 2

/-- Points earned for losing a game -/
def lose_points : ℕ := 1

/-- Minimum points target for a class -/
def min_points : ℕ := 14

/-- Theorem stating the number of classes and minimum wins required -/
theorem basketball_tournament :
  (num_classes * (num_classes - 1)) / 2 = total_matches ∧
  ∃ (min_wins : ℕ), 
    min_wins * win_points + (num_classes - 1 - min_wins) * lose_points ≥ min_points ∧
    ∀ (wins : ℕ), wins < min_wins → 
      wins * win_points + (num_classes - 1 - wins) * lose_points < min_points :=
by sorry

end NUMINAMATH_CALUDE_basketball_tournament_l1371_137120


namespace NUMINAMATH_CALUDE_motorcycle_cyclist_meeting_times_l1371_137194

theorem motorcycle_cyclist_meeting_times 
  (angle : Real) 
  (cyclist_speed : Real) 
  (motorcyclist_speed : Real) 
  (t : Real) : 
  angle = π / 3 →
  cyclist_speed = 36 →
  motorcyclist_speed = 72 →
  (motorcyclist_speed^2 * t^2 + cyclist_speed^2 * (t - 1)^2 - 
   2 * motorcyclist_speed * cyclist_speed * |t| * |t - 1| * (1/2) = 252^2) →
  (t = 4 ∨ t = -4) :=
by sorry

end NUMINAMATH_CALUDE_motorcycle_cyclist_meeting_times_l1371_137194


namespace NUMINAMATH_CALUDE_emily_candy_problem_l1371_137161

/-- The number of candy pieces Emily received from neighbors -/
def candy_from_neighbors : ℕ := 5

/-- The number of candy pieces Emily ate per day -/
def candy_eaten_per_day : ℕ := 9

/-- The number of days the candy lasted -/
def days_candy_lasted : ℕ := 2

/-- The number of candy pieces Emily received from her older sister -/
def candy_from_sister : ℕ := 13

theorem emily_candy_problem :
  candy_from_sister = (candy_eaten_per_day * days_candy_lasted) - candy_from_neighbors := by
  sorry

end NUMINAMATH_CALUDE_emily_candy_problem_l1371_137161


namespace NUMINAMATH_CALUDE_division_ways_correct_l1371_137134

/-- The number of ways to divide 6 distinct objects into three groups,
    where one group has 4 objects and the other two groups have 1 object each. -/
def divisionWays : ℕ := 15

/-- The total number of objects to be divided. -/
def totalObjects : ℕ := 6

/-- The number of objects in the largest group. -/
def largestGroupSize : ℕ := 4

/-- The number of groups. -/
def numberOfGroups : ℕ := 3

/-- Theorem stating that the number of ways to divide the objects is correct. -/
theorem division_ways_correct :
  divisionWays = Nat.choose totalObjects largestGroupSize :=
sorry

end NUMINAMATH_CALUDE_division_ways_correct_l1371_137134


namespace NUMINAMATH_CALUDE_triangle_existence_from_bisector_and_segments_l1371_137141

/-- Given an angle bisector and the segments it divides a side into,
    prove the existence of a triangle satisfying these conditions. -/
theorem triangle_existence_from_bisector_and_segments
  (l_c a' b' : ℝ) (h_positive : l_c > 0 ∧ a' > 0 ∧ b' > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    l_c ^ 2 = a * b - a' * b' ∧
    a' / b' = a / b :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_from_bisector_and_segments_l1371_137141


namespace NUMINAMATH_CALUDE_school_sampling_is_systematic_l1371_137174

/-- Represents a student with a unique student number -/
structure Student where
  number : ℕ

/-- Represents the sampling method used -/
inductive SamplingMethod
  | Stratified
  | Lottery
  | Random
  | Systematic

/-- Function to check if a student number ends in 4 -/
def endsInFour (n : ℕ) : Bool :=
  n % 10 = 4

/-- The sampling method used in the school -/
def schoolSamplingMethod (students : List Student) : SamplingMethod :=
  SamplingMethod.Systematic

/-- Theorem stating that the school's sampling method is systematic sampling -/
theorem school_sampling_is_systematic (students : List Student) :
  schoolSamplingMethod students = SamplingMethod.Systematic :=
by sorry

end NUMINAMATH_CALUDE_school_sampling_is_systematic_l1371_137174


namespace NUMINAMATH_CALUDE_money_distribution_l1371_137105

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 400)
  (AC_sum : A + C = 300)
  (BC_sum : B + C = 150) : 
  C = 50 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1371_137105


namespace NUMINAMATH_CALUDE_sum_reciprocal_and_sum_squares_l1371_137147

theorem sum_reciprocal_and_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (1 / a + 1 / b ≥ 4) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → a^2 + b^2 ≤ x^2 + y^2) := by
  sorry

#check sum_reciprocal_and_sum_squares

end NUMINAMATH_CALUDE_sum_reciprocal_and_sum_squares_l1371_137147


namespace NUMINAMATH_CALUDE_cube_inequality_l1371_137176

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l1371_137176


namespace NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l1371_137146

theorem cos_pi_sixth_plus_alpha (α : Real) 
  (h : Real.sin (α - π/3) = 1/3) : 
  Real.cos (π/6 + α) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l1371_137146


namespace NUMINAMATH_CALUDE_parallel_properties_l1371_137142

-- Define a type for lines
variable {Line : Type}

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Define a relation for two lines being parallel to the same line
def parallel_to_same (l1 l2 : Line) : Prop :=
  ∃ l3 : Line, parallel l1 l3 ∧ parallel l2 l3

theorem parallel_properties :
  (∀ l1 l2 : Line, parallel_to_same parallel l1 l2 → parallel l1 l2) ∧
  (∀ l1 l2 : Line, parallel l1 l2 → parallel_to_same parallel l1 l2) ∧
  (∀ l1 l2 : Line, ¬parallel_to_same parallel l1 l2 → ¬parallel l1 l2) ∧
  (∀ l1 l2 : Line, ¬parallel l1 l2 → ¬parallel_to_same parallel l1 l2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_properties_l1371_137142


namespace NUMINAMATH_CALUDE_total_tiles_is_44_l1371_137169

-- Define the room dimensions
def room_length : ℕ := 20
def room_width : ℕ := 15

-- Define tile sizes
def border_tile_size : ℕ := 2
def inner_tile_size : ℕ := 3

-- Function to calculate the number of border tiles
def border_tiles : ℕ :=
  2 * (room_length / border_tile_size + room_width / border_tile_size) - 4

-- Function to calculate the inner area
def inner_area : ℕ :=
  (room_length - 2 * border_tile_size) * (room_width - 2 * border_tile_size)

-- Function to calculate the number of inner tiles
def inner_tiles : ℕ :=
  (inner_area + inner_tile_size^2 - 1) / inner_tile_size^2

-- Theorem stating the total number of tiles
theorem total_tiles_is_44 :
  border_tiles + inner_tiles = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_tiles_is_44_l1371_137169


namespace NUMINAMATH_CALUDE_circle_radius_implies_a_value_l1371_137168

theorem circle_radius_implies_a_value (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*a*x + 4*a*y = 0) → 
  (∃ (c_x c_y : ℝ), ∀ (x y : ℝ), (x - c_x)^2 + (y - c_y)^2 = 5) →
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_implies_a_value_l1371_137168


namespace NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l1371_137178

/-- The sum of the series ∑_{n=1}^∞ 1/(n(n+3)) is equal to 11/18. -/
theorem sum_reciprocal_n_n_plus_three : 
  (∑' n : ℕ+, (1 : ℝ) / (n * (n + 3))) = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l1371_137178


namespace NUMINAMATH_CALUDE_fraction_reduction_l1371_137198

theorem fraction_reduction (a b c : ℝ) (h : a + b + c ≠ 0) :
  (a^2 + b^2 - c^2 + 2*a*b) / (a^2 + c^2 - b^2 + 2*a*c) = (a + b - c) / (a - b + c) :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l1371_137198
