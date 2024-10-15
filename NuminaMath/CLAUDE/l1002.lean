import Mathlib

namespace NUMINAMATH_CALUDE_magnitude_of_z_l1002_100237

theorem magnitude_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1002_100237


namespace NUMINAMATH_CALUDE_optimal_playground_max_area_l1002_100249

/-- Represents a rectangular playground with given constraints -/
structure Playground where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 190
  length_constraint : length ≥ 100
  width_constraint : width ≥ 60

/-- The area of a playground -/
def area (p : Playground) : ℝ := p.length * p.width

/-- The optimal playground dimensions -/
def optimal_playground : Playground := {
  length := 100,
  width := 90,
  perimeter_constraint := by sorry,
  length_constraint := by sorry,
  width_constraint := by sorry
}

/-- Theorem stating that the optimal playground has the maximum area -/
theorem optimal_playground_max_area :
  ∀ p : Playground, area p ≤ area optimal_playground := by sorry

end NUMINAMATH_CALUDE_optimal_playground_max_area_l1002_100249


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l1002_100277

-- Define the condition for a hyperbola with foci on the x-axis
def is_hyperbola_x_axis (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0

-- Define the equation
def equation (m n x y : ℝ) : Prop :=
  m * x^2 - n * y^2 = 1

-- Theorem statement
theorem mn_positive_necessary_not_sufficient :
  ∀ m n : ℝ,
    (is_hyperbola_x_axis m n → m * n > 0) ∧
    ¬(m * n > 0 → is_hyperbola_x_axis m n) :=
by sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l1002_100277


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1002_100274

def satisfies_inequalities (x : ℤ) : Prop :=
  (2 * (x - 1) < x + 3) ∧ ((2 * x + 1) / 3 > x - 1)

def non_negative_integer_solutions : Set ℤ :=
  {x : ℤ | x ≥ 0 ∧ satisfies_inequalities x}

theorem inequality_system_solution :
  non_negative_integer_solutions = {0, 1, 2, 3} :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1002_100274


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l1002_100268

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 13
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersection :
  ∃ (A B : ℝ × ℝ), 
    (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧ 
    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧ 
    A ≠ B ∧
    (∀ (x y : ℝ), perp_bisector x y ↔ 
      (x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = 
      ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l1002_100268


namespace NUMINAMATH_CALUDE_mean_of_four_numbers_with_sum_half_l1002_100279

theorem mean_of_four_numbers_with_sum_half (a b c d : ℚ) 
  (sum_condition : a + b + c + d = 1/2) : 
  (a + b + c + d) / 4 = 1/8 := by
sorry

end NUMINAMATH_CALUDE_mean_of_four_numbers_with_sum_half_l1002_100279


namespace NUMINAMATH_CALUDE_percentage_excess_l1002_100241

/-- Given two positive real numbers A and B with a specific ratio and sum condition,
    this theorem proves the formula for the percentage by which B exceeds A. -/
theorem percentage_excess (x y A B : ℝ) : 
  x > 0 → y > 0 → A > 0 → B > 0 →
  A / B = (5 * y^2) / (6 * x) →
  2 * x + 3 * y = 42 →
  ((B - A) / A) * 100 = ((126 - 9*y - 5*y^2) / (5*y^2)) * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_excess_l1002_100241


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1002_100209

/-- The number of ways to arrange n distinct objects into k distinct positions --/
def arrangements (n k : ℕ) : ℕ := (k.factorial) / ((k - n).factorial)

/-- The number of seating arrangements for three people in a row of eight chairs
    with an empty seat on either side of each person --/
def seatingArrangements : ℕ :=
  let totalChairs : ℕ := 8
  let peopleToSeat : ℕ := 3
  let availablePositions : ℕ := totalChairs - 2 - (peopleToSeat - 1)
  arrangements peopleToSeat availablePositions

theorem seating_arrangements_count :
  seatingArrangements = 24 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l1002_100209


namespace NUMINAMATH_CALUDE_triangle_consecutive_numbers_l1002_100204

/-- Represents the state of the triangle cells -/
def TriangleState := List Int

/-- Represents an operation on two adjacent cells -/
inductive Operation
| Add : Nat → Nat → Operation
| Subtract : Nat → Nat → Operation

/-- Checks if two cells are adjacent in the triangle -/
def are_adjacent (i j : Nat) : Bool := sorry

/-- Applies an operation to the triangle state -/
def apply_operation (state : TriangleState) (op : Operation) : TriangleState := sorry

/-- Checks if a list contains consecutive integers from n to n+8 -/
def is_consecutive_from_n (l : List Int) (n : Int) : Prop := sorry

/-- The main theorem -/
theorem triangle_consecutive_numbers :
  ∀ (initial_state : TriangleState),
  (initial_state.length = 9 ∧ initial_state.all (· = 0)) →
  ∃ (n : Int) (final_state : TriangleState),
  (∃ (ops : List Operation), 
    (∀ op ∈ ops, ∃ i j, are_adjacent i j ∧ (op = Operation.Add i j ∨ op = Operation.Subtract i j)) ∧
    (final_state = ops.foldl apply_operation initial_state)) ∧
  is_consecutive_from_n final_state n →
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_consecutive_numbers_l1002_100204


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_sum_l1002_100290

theorem cube_roots_of_unity_sum (ω ω_conj : ℂ) : 
  ω = (-1 + Complex.I * Real.sqrt 3) / 2 →
  ω_conj = (-1 - Complex.I * Real.sqrt 3) / 2 →
  ω^3 = 1 →
  ω_conj^3 = 1 →
  ω^4 + ω_conj^4 - 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_sum_l1002_100290


namespace NUMINAMATH_CALUDE_ln_lower_bound_l1002_100291

theorem ln_lower_bound (n : ℕ) (k : ℕ) (h : k = (Nat.factorization n).support.card) :
  Real.log n ≥ k * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ln_lower_bound_l1002_100291


namespace NUMINAMATH_CALUDE_min_minutes_for_plan_b_cheaper_l1002_100281

def plan_a_cost (minutes : ℕ) : ℚ := 15 + (12 / 100) * minutes
def plan_b_cost (minutes : ℕ) : ℚ := 30 + (6 / 100) * minutes

theorem min_minutes_for_plan_b_cheaper : 
  ∀ m : ℕ, m ≥ 251 → plan_b_cost m < plan_a_cost m ∧
  ∀ n : ℕ, n < 251 → plan_a_cost n ≤ plan_b_cost n :=
by sorry

end NUMINAMATH_CALUDE_min_minutes_for_plan_b_cheaper_l1002_100281


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1002_100205

def polynomial (x : ℂ) : ℂ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

theorem root_sum_theorem (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h₁ : polynomial z₁ = 0)
  (h₂ : polynomial z₂ = 0)
  (h₃ : polynomial z₃ = 0)
  (h₄ : polynomial z₄ = 0)
  (h₅ : polynomial z₅ = 0) :
  (z₁ / (z₁^2 + 1) + z₂ / (z₂^2 + 1) + z₃ / (z₃^2 + 1) + z₄ / (z₄^2 + 1) + z₅ / (z₅^2 + 1)) = 4/17 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1002_100205


namespace NUMINAMATH_CALUDE_equation_proof_l1002_100218

theorem equation_proof : 3889 + 12.808 - 47.806 = 3854.002 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1002_100218


namespace NUMINAMATH_CALUDE_robotics_club_max_participants_l1002_100286

theorem robotics_club_max_participants 
  (physics : Finset ℕ)
  (math : Finset ℕ)
  (programming : Finset ℕ)
  (h1 : physics.card = 8)
  (h2 : math.card = 7)
  (h3 : programming.card = 11)
  (h4 : (physics ∩ math).card ≥ 2)
  (h5 : (math ∩ programming).card ≥ 3)
  (h6 : (physics ∩ programming).card ≥ 4) :
  (physics ∪ math ∪ programming).card ≤ 19 :=
sorry

end NUMINAMATH_CALUDE_robotics_club_max_participants_l1002_100286


namespace NUMINAMATH_CALUDE_logarithm_simplification_l1002_100227

theorem logarithm_simplification 
  (p q r s y z : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (hy : y > 0) (hz : z > 0) : 
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * z / (s * y)) = Real.log (y / z) :=
sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l1002_100227


namespace NUMINAMATH_CALUDE_expression_always_positive_l1002_100223

theorem expression_always_positive (x : ℝ) : (x - 3) * (x - 5) + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_positive_l1002_100223


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l1002_100224

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x - 5

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, x < y ∧ y ≤ 1 → f x < f y :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l1002_100224


namespace NUMINAMATH_CALUDE_sin_870_degrees_l1002_100230

theorem sin_870_degrees : Real.sin (870 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_870_degrees_l1002_100230


namespace NUMINAMATH_CALUDE_valid_prices_count_l1002_100206

def valid_digits : List Nat := [1, 1, 4, 5, 6, 6]

def is_valid_start (n : Nat) : Bool :=
  n ≥ 4

def count_valid_prices (digits : List Nat) : Nat :=
  digits.filter is_valid_start
    |>.map (λ d => (digits.erase d).permutations.length)
    |>.sum

theorem valid_prices_count :
  count_valid_prices valid_digits = 90 := by
  sorry

end NUMINAMATH_CALUDE_valid_prices_count_l1002_100206


namespace NUMINAMATH_CALUDE_special_cubes_in_4x5x6_prism_l1002_100288

/-- Represents a rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of unit cubes in a rectangular prism that have
    either exactly one face on the surface or no faces on the surface -/
def count_special_cubes (prism : RectangularPrism) : ℕ :=
  let interior_cubes := (prism.length - 2) * (prism.width - 2) * (prism.height - 2)
  let one_face_cubes := 2 * ((prism.width - 2) * (prism.height - 2) +
                             (prism.length - 2) * (prism.height - 2) +
                             (prism.length - 2) * (prism.width - 2))
  interior_cubes + one_face_cubes

/-- The main theorem stating that a 4x5x6 prism has 76 special cubes -/
theorem special_cubes_in_4x5x6_prism :
  count_special_cubes ⟨4, 5, 6⟩ = 76 := by
  sorry

end NUMINAMATH_CALUDE_special_cubes_in_4x5x6_prism_l1002_100288


namespace NUMINAMATH_CALUDE_contract_completion_hours_l1002_100242

/-- Represents the contract completion problem -/
structure ContractProblem where
  total_days : ℕ
  initial_men : ℕ
  initial_hours : ℕ
  days_passed : ℕ
  work_completed : ℚ
  additional_men : ℕ

/-- Calculates the required daily work hours to complete the contract on time -/
def required_hours (p : ContractProblem) : ℚ :=
  let total_man_hours := p.initial_men * p.initial_hours * p.total_days
  let remaining_man_hours := (1 - p.work_completed) * total_man_hours
  let remaining_days := p.total_days - p.days_passed
  let total_men := p.initial_men + p.additional_men
  remaining_man_hours / (total_men * remaining_days)

/-- Theorem stating that the required work hours for the given problem is approximately 7.16 -/
theorem contract_completion_hours (p : ContractProblem) 
  (h1 : p.total_days = 46)
  (h2 : p.initial_men = 117)
  (h3 : p.initial_hours = 8)
  (h4 : p.days_passed = 33)
  (h5 : p.work_completed = 4/7)
  (h6 : p.additional_men = 81) :
  ∃ ε > 0, abs (required_hours p - 7.16) < ε :=
sorry

end NUMINAMATH_CALUDE_contract_completion_hours_l1002_100242


namespace NUMINAMATH_CALUDE_range_of_x_when_m_is_2_range_of_m_when_p_necessary_not_sufficient_l1002_100299

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 6*x + 5 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Part 1
theorem range_of_x_when_m_is_2 :
  ∀ x : ℝ, (p x ∧ q x 2) → (1 ≤ x ∧ x ≤ 3) :=
sorry

-- Part 2
theorem range_of_m_when_p_necessary_not_sufficient :
  ∀ m : ℝ, (m > 0 ∧ (∀ x : ℝ, q x m → p x) ∧ (∃ x : ℝ, p x ∧ ¬(q x m))) → m ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_m_is_2_range_of_m_when_p_necessary_not_sufficient_l1002_100299


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_specific_equation_l1002_100264

theorem sum_of_squares_of_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a ≠ 0 → a*x^2 + b*x + c = 0 → r₁^2 + r₂^2 = (b^2 - 2*a*c) / a^2 :=
by
  sorry

theorem sum_of_squares_of_specific_equation :
  let r₁ := (-(-14) + Real.sqrt ((-14)^2 - 4*1*8)) / (2*1)
  let r₂ := (-(-14) - Real.sqrt ((-14)^2 - 4*1*8)) / (2*1)
  r₁^2 + r₂^2 = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_specific_equation_l1002_100264


namespace NUMINAMATH_CALUDE_three_heads_probability_l1002_100221

/-- A fair coin has a probability of 1/2 for heads on a single flip -/
def fair_coin_prob : ℚ := 1 / 2

/-- The probability of getting three heads in three independent flips of a fair coin -/
def three_heads_prob : ℚ := fair_coin_prob * fair_coin_prob * fair_coin_prob

/-- Theorem: The probability of getting three heads in three independent flips of a fair coin is 1/8 -/
theorem three_heads_probability : three_heads_prob = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_three_heads_probability_l1002_100221


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l1002_100273

/-- Given vectors a, b, and c in ℝ³, prove they are not coplanar. -/
theorem vectors_not_coplanar :
  let a : ℝ × ℝ × ℝ := (3, 10, 5)
  let b : ℝ × ℝ × ℝ := (-2, -2, -3)
  let c : ℝ × ℝ × ℝ := (2, 4, 3)
  ¬(∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l1002_100273


namespace NUMINAMATH_CALUDE_max_side_length_exists_max_side_length_l1002_100263

/-- A triangle with integer side lengths and perimeter 24 -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter : a + b + c = 24

/-- The maximum side length of a triangle is 11 -/
theorem max_side_length (t : Triangle) : t.a ≤ 11 ∧ t.b ≤ 11 ∧ t.c ≤ 11 :=
sorry

/-- There exists a triangle with maximum side length 11 -/
theorem exists_max_side_length : ∃ (t : Triangle), t.a = 11 ∨ t.b = 11 ∨ t.c = 11 :=
sorry

end NUMINAMATH_CALUDE_max_side_length_exists_max_side_length_l1002_100263


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l1002_100270

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_f_at_2 : 
  deriv f 2 = 3 * Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l1002_100270


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l1002_100252

theorem complex_in_second_quadrant : 
  let z : ℂ := Complex.mk (Real.cos 3) (Real.sin 3)
  Complex.re z < 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l1002_100252


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1002_100296

theorem simplify_fraction_product : 8 * (15 / 4) * (-40 / 45) = -80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1002_100296


namespace NUMINAMATH_CALUDE_project_completion_time_l1002_100295

/-- The number of days it takes for person A to complete the project alone -/
def A_days : ℕ := 20

/-- The number of days it takes for both A and B to complete the project together,
    with A quitting 10 days before completion -/
def total_days : ℕ := 18

/-- The number of days A works before quitting -/
def A_work_days : ℕ := total_days - 10

/-- The rate at which person A completes the project per day -/
def A_rate : ℚ := 1 / A_days

theorem project_completion_time (B_days : ℕ) :
  (A_work_days : ℚ) * (A_rate + 1 / B_days) + (10 : ℚ) * (1 / B_days) = 1 →
  B_days = 30 := by sorry

end NUMINAMATH_CALUDE_project_completion_time_l1002_100295


namespace NUMINAMATH_CALUDE_refrigerator_loss_percentage_l1002_100284

/-- Represents the problem of calculating the loss percentage on a refrigerator. -/
def RefrigeratorLossProblem (refrigerator_cp mobile_cp : ℕ) (mobile_profit overall_profit : ℕ) : Prop :=
  let refrigerator_sp := refrigerator_cp + mobile_cp + overall_profit - (mobile_cp + mobile_cp * mobile_profit / 100)
  let loss_percentage := (refrigerator_cp - refrigerator_sp) * 100 / refrigerator_cp
  loss_percentage = 5

/-- The main theorem stating that given the problem conditions, the loss percentage on the refrigerator is 5%. -/
theorem refrigerator_loss_percentage :
  RefrigeratorLossProblem 15000 8000 10 50 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_loss_percentage_l1002_100284


namespace NUMINAMATH_CALUDE_jose_age_is_19_l1002_100294

-- Define the ages of the individuals
def inez_age : ℕ := 18
def alice_age : ℕ := inez_age - 3
def zack_age : ℕ := inez_age + 5
def jose_age : ℕ := zack_age - 4

-- Theorem to prove Jose's age
theorem jose_age_is_19 : jose_age = 19 := by
  sorry

end NUMINAMATH_CALUDE_jose_age_is_19_l1002_100294


namespace NUMINAMATH_CALUDE_line_segments_not_in_proportion_l1002_100202

theorem line_segments_not_in_proportion :
  let a : ℝ := 4
  let b : ℝ := 5
  let c : ℝ := 6
  let d : ℝ := 10
  (a / b) ≠ (c / d) :=
by sorry

end NUMINAMATH_CALUDE_line_segments_not_in_proportion_l1002_100202


namespace NUMINAMATH_CALUDE_bert_spending_l1002_100283

theorem bert_spending (n : ℝ) : 
  (3/4 * n - 9) / 2 = 12 → n = 44 := by sorry

end NUMINAMATH_CALUDE_bert_spending_l1002_100283


namespace NUMINAMATH_CALUDE_specific_tetrahedron_properties_l1002_100251

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Theorem stating the volume and height of a specific tetrahedron -/
theorem specific_tetrahedron_properties :
  let A₁ : Point3D := ⟨-2, 0, -4⟩
  let A₂ : Point3D := ⟨-1, 7, 1⟩
  let A₃ : Point3D := ⟨4, -8, -4⟩
  let A₄ : Point3D := ⟨1, -4, 6⟩
  (tetrahedronVolume A₁ A₂ A₃ A₄ = 250 / 3) ∧
  (tetrahedronHeight A₁ A₂ A₃ A₄ = 5 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_properties_l1002_100251


namespace NUMINAMATH_CALUDE_trapezium_height_l1002_100239

theorem trapezium_height (a b area : ℝ) (ha : a = 20) (hb : b = 18) (harea : area = 247) :
  (2 * area) / (a + b) = 13 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l1002_100239


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1002_100292

-- First expression
theorem simplify_expression_1 : 
  (Real.sqrt 12 + Real.sqrt 20) - (3 - Real.sqrt 5) = 2 * Real.sqrt 3 + 3 * Real.sqrt 5 - 3 := by
  sorry

-- Second expression
theorem simplify_expression_2 : 
  Real.sqrt 8 * Real.sqrt 6 - 3 * Real.sqrt 6 + Real.sqrt 2 = 4 * Real.sqrt 3 - 3 * Real.sqrt 6 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1002_100292


namespace NUMINAMATH_CALUDE_square_of_binomial_b_value_l1002_100253

theorem square_of_binomial_b_value (p b : ℝ) : 
  (∃ q : ℝ, ∀ x : ℝ, x^2 + p*x + b = (x + q)^2) → 
  p = -10 → 
  b = 25 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_b_value_l1002_100253


namespace NUMINAMATH_CALUDE_difference_greater_than_one_l1002_100255

theorem difference_greater_than_one (x : ℕ+) :
  (x.val + 3 : ℚ) / 2 - (2 * x.val - 1 : ℚ) / 3 > 1 ↔ x.val < 5 := by
  sorry

end NUMINAMATH_CALUDE_difference_greater_than_one_l1002_100255


namespace NUMINAMATH_CALUDE_triangle_conditions_l1002_100216

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a triangle is right-angled
def is_right_triangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

-- Condition A
def condition_A (t : Triangle) : Prop :=
  t.a = 1/3 ∧ t.b = 1/4 ∧ t.c = 1/5

-- Condition B (using angle ratios)
def condition_B (A B C : ℝ) : Prop :=
  A / B = 1/3 ∧ A / C = 1/2 ∧ B / C = 3/2

-- Condition C
def condition_C (t : Triangle) : Prop :=
  (t.b + t.c) * (t.b - t.c) = t.a^2

theorem triangle_conditions :
  (∃ t1 t2 : Triangle, condition_A t1 ∧ is_right_triangle t1 ∧
                       condition_A t2 ∧ ¬is_right_triangle t2) ∧
  (∀ A B C : ℝ, condition_B A B C → A + B + C = 180 → B = 90) ∧
  (∀ t : Triangle, condition_C t → is_right_triangle t) :=
sorry

end NUMINAMATH_CALUDE_triangle_conditions_l1002_100216


namespace NUMINAMATH_CALUDE_particular_number_addition_l1002_100257

theorem particular_number_addition : ∃ x : ℝ, 0.46 + x = 0.72 ∧ x = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_addition_l1002_100257


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l1002_100245

theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) :
  area = 300 ∧ d2 = 20 ∧ area = (d1 * d2) / 2 → d1 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l1002_100245


namespace NUMINAMATH_CALUDE_sqrt_42_minus_1_range_l1002_100215

theorem sqrt_42_minus_1_range : 5 < Real.sqrt 42 - 1 ∧ Real.sqrt 42 - 1 < 6 := by
  have h1 : 36 < 42 := by sorry
  have h2 : 42 < 49 := by sorry
  have h3 : Real.sqrt 36 = 6 := by sorry
  have h4 : Real.sqrt 49 = 7 := by sorry
  sorry

end NUMINAMATH_CALUDE_sqrt_42_minus_1_range_l1002_100215


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_equivalent_to_range_l1002_100298

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem intersection_when_a_is_one :
  A 1 ∩ B = {x | 1 < x ∧ x < 2} := by sorry

theorem subset_condition_equivalent_to_range :
  ∀ a : ℝ, A a ⊆ A a ∩ B ↔ 2 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_equivalent_to_range_l1002_100298


namespace NUMINAMATH_CALUDE_car_wash_price_l1002_100259

theorem car_wash_price (oil_change_price : ℕ) (repair_price : ℕ) (oil_changes : ℕ) (repairs : ℕ) (car_washes : ℕ) (total_earnings : ℕ) :
  oil_change_price = 20 →
  repair_price = 30 →
  oil_changes = 5 →
  repairs = 10 →
  car_washes = 15 →
  total_earnings = 475 →
  (oil_change_price * oil_changes + repair_price * repairs + car_washes * 5 = total_earnings) :=
by
  sorry


end NUMINAMATH_CALUDE_car_wash_price_l1002_100259


namespace NUMINAMATH_CALUDE_equation_solution_l1002_100285

theorem equation_solution (y : ℝ) (h : (1 : ℝ) / 3 + 1 / y = 7 / 9) : y = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1002_100285


namespace NUMINAMATH_CALUDE_second_draw_probability_l1002_100220

/-- Represents the probability of drawing a red sweet in the second draw -/
def probability_second_red (x y : ℕ) : ℚ :=
  y / (x + y)

/-- Theorem stating that the probability of drawing a red sweet in the second draw
    is equal to the initial ratio of red sweets to total sweets -/
theorem second_draw_probability (x y : ℕ) (hxy : x + y > 0) :
  probability_second_red x y = y / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_second_draw_probability_l1002_100220


namespace NUMINAMATH_CALUDE_mean_temperature_l1002_100200

def temperatures : List ℤ := [-8, -3, -3, -6, 2, 4, 1]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1002_100200


namespace NUMINAMATH_CALUDE_fraction_product_power_l1002_100247

theorem fraction_product_power : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_power_l1002_100247


namespace NUMINAMATH_CALUDE_eva_is_speed_skater_l1002_100207

-- Define the people and sports
inductive Person : Type
| Ben : Person
| Filip : Person
| Eva : Person
| Andrea : Person

inductive Sport : Type
| SpeedSkating : Sport
| Skiing : Sport
| Hockey : Sport
| Snowboarding : Sport

-- Define the positions at the table
inductive Position : Type
| Top : Position
| Right : Position
| Bottom : Position
| Left : Position

-- Define the seating arrangement
def SeatingArrangement : Type := Person → Position

-- Define the sport assignment
def SportAssignment : Type := Person → Sport

-- Define the conditions
def Conditions (seating : SeatingArrangement) (sports : SportAssignment) : Prop :=
  ∃ (skier hockey_player : Person),
    -- The skier sat at Andrea's left hand
    seating Person.Andrea = Position.Top ∧ seating skier = Position.Left
    -- The speed skater sat opposite Ben
    ∧ seating Person.Ben = Position.Left
    ∧ sports Person.Ben ≠ Sport.SpeedSkating
    -- Eva and Filip sat next to each other
    ∧ (seating Person.Eva = Position.Right ∧ seating Person.Filip = Position.Bottom
    ∨ seating Person.Eva = Position.Bottom ∧ seating Person.Filip = Position.Right)
    -- A woman sat at the hockey player's left hand
    ∧ ((seating hockey_player = Position.Right ∧ seating Person.Andrea = Position.Top)
    ∨ (seating hockey_player = Position.Bottom ∧ seating Person.Eva = Position.Right))

-- The theorem to prove
theorem eva_is_speed_skater (seating : SeatingArrangement) (sports : SportAssignment) :
  Conditions seating sports → sports Person.Eva = Sport.SpeedSkating :=
sorry

end NUMINAMATH_CALUDE_eva_is_speed_skater_l1002_100207


namespace NUMINAMATH_CALUDE_tourist_growth_rate_l1002_100269

theorem tourist_growth_rate (x : ℝ) : 
  (1 - 0.4) * (1 - 0.5) * (1 + x) = 2 :=
by sorry

end NUMINAMATH_CALUDE_tourist_growth_rate_l1002_100269


namespace NUMINAMATH_CALUDE_grace_garden_medium_bed_rows_l1002_100293

/-- Represents a raised bed garden with large and medium beds -/
structure RaisedBedGarden where
  large_beds : Nat
  medium_beds : Nat
  large_bed_rows : Nat
  large_bed_seeds_per_row : Nat
  medium_bed_seeds_per_row : Nat
  total_seeds : Nat

/-- Calculates the number of rows in medium beds -/
def medium_bed_rows (garden : RaisedBedGarden) : Nat :=
  let large_bed_seeds := garden.large_beds * garden.large_bed_rows * garden.large_bed_seeds_per_row
  let medium_bed_seeds := garden.total_seeds - large_bed_seeds
  medium_bed_seeds / garden.medium_bed_seeds_per_row

/-- Theorem stating that for the given garden configuration, medium beds have 6 rows -/
theorem grace_garden_medium_bed_rows :
  let garden : RaisedBedGarden := {
    large_beds := 2,
    medium_beds := 2,
    large_bed_rows := 4,
    large_bed_seeds_per_row := 25,
    medium_bed_seeds_per_row := 20,
    total_seeds := 320
  }
  medium_bed_rows garden = 6 := by
  sorry

end NUMINAMATH_CALUDE_grace_garden_medium_bed_rows_l1002_100293


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l1002_100254

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a b c d e f : ℝ) : Prop :=
  a / b = d / e

/-- Given two lines l₁: ax + 3y - 3 = 0 and l₂: 4x + 6y - 1 = 0,
    if they are parallel, then a = 2 -/
theorem parallel_lines_theorem (a : ℝ) :
  parallel_lines a 3 (-3) 4 6 (-1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l1002_100254


namespace NUMINAMATH_CALUDE_red_balls_count_l1002_100250

/-- Given a bag with 15 balls, prove that there are 6 red balls if the probability
    of drawing two red balls at random at the same time is 2/35. -/
theorem red_balls_count (total : ℕ) (prob : ℚ) (h1 : total = 15) (h2 : prob = 2/35) :
  ∃ r : ℕ, r = 6 ∧
  (r : ℚ) / total * ((r - 1) : ℚ) / (total - 1) = prob :=
sorry

end NUMINAMATH_CALUDE_red_balls_count_l1002_100250


namespace NUMINAMATH_CALUDE_final_amount_calculation_l1002_100217

-- Define the variables
def initial_amount : ℕ := 45
def amount_spent : ℕ := 20
def additional_amount : ℕ := 46

-- Define the theorem
theorem final_amount_calculation :
  initial_amount - amount_spent + additional_amount = 71 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_calculation_l1002_100217


namespace NUMINAMATH_CALUDE_no_three_digit_multiple_base_l1002_100256

/-- Definition of a valid base for a number x -/
def valid_base (x : ℕ) (b : ℕ) : Prop :=
  b ≥ 2 ∧ b ≤ 10 ∧ (b - 1)^4 < x ∧ x < b^4

/-- Definition of a three-digit number in base b -/
def three_digit (x : ℕ) (b : ℕ) : Prop :=
  b^2 ≤ x ∧ x < b^3

/-- Main theorem: No three-digit number represents multiple values in different bases -/
theorem no_three_digit_multiple_base :
  ¬ ∃ (x : ℕ) (b1 b2 : ℕ), x < 10000 ∧ b1 < b2 ∧
    valid_base x b1 ∧ valid_base x b2 ∧
    three_digit x b1 ∧ three_digit x b2 :=
sorry

end NUMINAMATH_CALUDE_no_three_digit_multiple_base_l1002_100256


namespace NUMINAMATH_CALUDE_rental_problem_l1002_100266

/-- Rental problem theorem -/
theorem rental_problem (first_hour_rate : ℕ) (additional_hour_rate : ℕ) (total_paid : ℕ) :
  first_hour_rate = 25 →
  additional_hour_rate = 10 →
  total_paid = 125 →
  ∃ (hours : ℕ), hours = 11 ∧ 
    total_paid = first_hour_rate + (hours - 1) * additional_hour_rate :=
by
  sorry

end NUMINAMATH_CALUDE_rental_problem_l1002_100266


namespace NUMINAMATH_CALUDE_lansing_elementary_students_l1002_100234

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

theorem lansing_elementary_students :
  total_students = 6175 :=
by sorry

end NUMINAMATH_CALUDE_lansing_elementary_students_l1002_100234


namespace NUMINAMATH_CALUDE_sandbox_width_l1002_100265

theorem sandbox_width (perimeter : ℝ) (width : ℝ) (length : ℝ) : 
  perimeter = 30 →
  length = 2 * width →
  perimeter = 2 * width + 2 * length →
  width = 5 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_width_l1002_100265


namespace NUMINAMATH_CALUDE_min_value_expression_l1002_100236

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min : ℝ), min = (1 / 2 : ℝ) ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 →
    (1 / a) - (4 * b / (b + 1)) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1002_100236


namespace NUMINAMATH_CALUDE_line_intersects_AB_CD_l1002_100228

/-- Given points A, B, C, D, prove that the line x = 8t, y = 2t, z = 11t passes through
    the origin and intersects both lines AB and CD. -/
theorem line_intersects_AB_CD :
  let A : ℝ × ℝ × ℝ := (1, 0, 1)
  let B : ℝ × ℝ × ℝ := (-2, 2, 1)
  let C : ℝ × ℝ × ℝ := (2, 0, 3)
  let D : ℝ × ℝ × ℝ := (0, 4, -2)
  let line (t : ℝ) : ℝ × ℝ × ℝ := (8*t, 2*t, 11*t)
  (∃ t : ℝ, line t = (0, 0, 0)) ∧ 
  (∃ t₁ s₁ : ℝ, line t₁ = (1-3*s₁, 2*s₁, 1)) ∧
  (∃ t₂ s₂ : ℝ, line t₂ = (2-2*s₂, 4*s₂, 3+5*s₂)) :=
by
  sorry


end NUMINAMATH_CALUDE_line_intersects_AB_CD_l1002_100228


namespace NUMINAMATH_CALUDE_instrument_players_l1002_100212

theorem instrument_players (total_people : ℕ) 
  (at_least_one_ratio : ℚ) (exactly_one_prob : ℝ) 
  (h1 : total_people = 800)
  (h2 : at_least_one_ratio = 1/5)
  (h3 : exactly_one_prob = 0.12) : 
  ℕ := by
  sorry

#check instrument_players

end NUMINAMATH_CALUDE_instrument_players_l1002_100212


namespace NUMINAMATH_CALUDE_ben_remaining_amount_l1002_100289

/-- Calculates the remaining amount after a series of transactions -/
def remaining_amount (initial: Int) (supplier_payment: Int) (debtor_payment: Int) (maintenance_cost: Int) : Int :=
  initial - supplier_payment + debtor_payment - maintenance_cost

/-- Proves that given the specified transactions, the remaining amount is $1000 -/
theorem ben_remaining_amount :
  remaining_amount 2000 600 800 1200 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ben_remaining_amount_l1002_100289


namespace NUMINAMATH_CALUDE_sally_lost_balloons_l1002_100238

theorem sally_lost_balloons (initial_orange : ℕ) (current_orange : ℕ) 
  (h1 : initial_orange = 9) 
  (h2 : current_orange = 7) : 
  initial_orange - current_orange = 2 := by
  sorry

end NUMINAMATH_CALUDE_sally_lost_balloons_l1002_100238


namespace NUMINAMATH_CALUDE_isosceles_points_count_l1002_100271

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the property of being acute
def isAcute (t : Triangle) : Prop := sorry

-- Define the ordering of side lengths
def sideOrdering (t : Triangle) : Prop := sorry

-- Define the property of a point P making isosceles triangles with AB and BC
def makesIsosceles (P : ℝ × ℝ) (t : Triangle) : Prop := sorry

-- The main theorem
theorem isosceles_points_count (t : Triangle) : 
  isAcute t → sideOrdering t → ∃! (points : Finset (ℝ × ℝ)), 
    Finset.card points = 15 ∧ 
    ∀ P ∈ points, makesIsosceles P t := by
  sorry

end NUMINAMATH_CALUDE_isosceles_points_count_l1002_100271


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1002_100246

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 6 * (2 * x - 1) - 3 * (5 + 2 * x) = 6 * x - 21 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : (4 * a^2 - 8 * a - 9) + 3 * (2 * a^2 - 2 * a - 5) = 10 * a^2 - 14 * a - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1002_100246


namespace NUMINAMATH_CALUDE_dentist_age_problem_l1002_100232

/-- Proves that given a dentist's current age of 32 years, if one-sixth of his age 8 years ago
    equals one-tenth of his age at a certain time in the future, then that future time is 8 years from now. -/
theorem dentist_age_problem (future_years : ℕ) : 
  (1/6 : ℚ) * (32 - 8) = (1/10 : ℚ) * (32 + future_years) → future_years = 8 := by
  sorry

end NUMINAMATH_CALUDE_dentist_age_problem_l1002_100232


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1002_100226

theorem arithmetic_computation : -9 * 3 - (-7 * -4) + (-11 * -6) = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1002_100226


namespace NUMINAMATH_CALUDE_intersection_point_l1002_100297

-- Define the line
def line (x y : ℝ) : Prop := 5 * y - 2 * x = 10

-- Define a point on the x-axis
def on_x_axis (x y : ℝ) : Prop := y = 0

-- Theorem: The point (-5, 0) is on the line and the x-axis
theorem intersection_point : 
  line (-5) 0 ∧ on_x_axis (-5) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l1002_100297


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1002_100211

/-- A function f: ℝ₊ → ℝ₊ satisfying the given functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x > 0 → f y > 0 → f (f x + y) = α * x + 1 / f (1 / y)

/-- The main theorem stating the solution to the functional equation. -/
theorem functional_equation_solution :
  ∀ α : ℝ, α ≠ 0 →
    (∃ f : ℝ → ℝ, FunctionalEquation f α) ↔ (α = 1 ∧ ∃ f : ℝ → ℝ, FunctionalEquation f 1 ∧ ∀ x, x > 0 → f x = x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1002_100211


namespace NUMINAMATH_CALUDE_contradiction_proof_l1002_100287

theorem contradiction_proof (x a b : ℝ) : 
  x^2 - (a + b)*x - a*b ≠ 0 → x ≠ a ∧ x ≠ b := by
  sorry

end NUMINAMATH_CALUDE_contradiction_proof_l1002_100287


namespace NUMINAMATH_CALUDE_sector_area_ratio_l1002_100225

/-- Given a circular sector AOB with central angle α (in radians),
    and a line drawn through point B and the midpoint C of radius OA,
    the ratio of the area of triangle BCO to the area of figure ABC
    is sin(α) / (2α - sin(α)). -/
theorem sector_area_ratio (α : Real) :
  let R : Real := 1  -- Assume unit radius for simplicity
  let S : Real := (1/2) * R^2 * α  -- Area of sector AOB
  let S_BCO : Real := (1/4) * R^2 * Real.sin α  -- Area of triangle BCO
  let S_ABC : Real := S - S_BCO  -- Area of figure ABC
  S_BCO / S_ABC = Real.sin α / (2 * α - Real.sin α) := by
sorry

end NUMINAMATH_CALUDE_sector_area_ratio_l1002_100225


namespace NUMINAMATH_CALUDE_monic_polynomial_square_decomposition_l1002_100275

theorem monic_polynomial_square_decomposition
  (P : Polynomial ℤ)
  (h_monic : P.Monic)
  (h_even_degree : Even P.degree)
  (h_infinite_squares : ∃ S : Set ℤ, Infinite S ∧ ∀ x ∈ S, ∃ y : ℤ, 0 < y ∧ P.eval x = y^2) :
  ∃ Q : Polynomial ℤ, P = Q^2 :=
sorry

end NUMINAMATH_CALUDE_monic_polynomial_square_decomposition_l1002_100275


namespace NUMINAMATH_CALUDE_riverside_park_adjustment_plans_l1002_100248

/-- Represents the number of riverside theme parks -/
def total_parks : ℕ := 7

/-- Represents the number of parks to be removed -/
def parks_to_remove : ℕ := 2

/-- Represents the number of parks that can be adjusted (excluding the ends) -/
def adjustable_parks : ℕ := total_parks - 2

/-- Represents the number of adjacent park pairs that cannot be removed together -/
def adjacent_pairs : ℕ := adjustable_parks - 1

theorem riverside_park_adjustment_plans :
  (adjustable_parks.choose parks_to_remove) - adjacent_pairs = 6 := by
  sorry

end NUMINAMATH_CALUDE_riverside_park_adjustment_plans_l1002_100248


namespace NUMINAMATH_CALUDE_trapezoid_area_l1002_100229

-- Define the trapezoid
def trapezoid := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ x ≤ 15 ∧ 10 ≤ y ∧ y ≤ 15 ∧ (y = x ∨ y = 10 ∨ y = 15)}

-- Define the area function
def area (T : Set (ℝ × ℝ)) : ℝ := 62.5

-- Theorem statement
theorem trapezoid_area : area trapezoid = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1002_100229


namespace NUMINAMATH_CALUDE_gcd_problem_l1002_100267

theorem gcd_problem (b : ℤ) (h : ∃ (k : ℤ), b = 7 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 34 * b + 76) (b + 16) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1002_100267


namespace NUMINAMATH_CALUDE_combined_weight_l1002_100203

/-- The combined weight of John, Mary, and Jamison is 540 lbs -/
theorem combined_weight (mary_weight jamison_weight john_weight : ℝ) :
  mary_weight = 160 →
  jamison_weight = mary_weight + 20 →
  john_weight = mary_weight * (5/4) →
  mary_weight + jamison_weight + john_weight = 540 := by
sorry

end NUMINAMATH_CALUDE_combined_weight_l1002_100203


namespace NUMINAMATH_CALUDE_ages_sum_l1002_100280

/-- Represents the ages of three people A, B, and C --/
structure Ages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The conditions of the problem --/
def problem_conditions (ages : Ages) : Prop :=
  ages.b = 30 ∧ 
  ∃ x : ℕ, x > 0 ∧ 
    ages.a - 10 = x ∧
    ages.b - 10 = 2 * x ∧
    ages.c - 10 = 3 * x

/-- The theorem to prove --/
theorem ages_sum (ages : Ages) : 
  problem_conditions ages → ages.a + ages.b + ages.c = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l1002_100280


namespace NUMINAMATH_CALUDE_weight_of_A_l1002_100244

theorem weight_of_A (a b c d : ℝ) : 
  (a + b + c) / 3 = 70 →
  (a + b + c + d) / 4 = 70 →
  ((b + c + d + (d + 3)) / 4 = 68) →
  a = 81 := by sorry

end NUMINAMATH_CALUDE_weight_of_A_l1002_100244


namespace NUMINAMATH_CALUDE_frog_climb_time_l1002_100231

/-- Represents the frog's climbing scenario -/
structure FrogClimb where
  wellDepth : ℝ
  climbUp : ℝ
  slipDown : ℝ
  slipTime : ℝ
  timeAt3mBelow : ℝ

/-- Calculates the time taken for the frog to reach the top of the well -/
def timeToReachTop (f : FrogClimb) : ℝ :=
  sorry

/-- Theorem stating that the frog takes 22 minutes to reach the top -/
theorem frog_climb_time (f : FrogClimb) 
  (h1 : f.wellDepth = 12)
  (h2 : f.climbUp = 3)
  (h3 : f.slipDown = 1)
  (h4 : f.slipTime = f.climbUp / 3)
  (h5 : f.timeAt3mBelow = 17) :
  timeToReachTop f = 22 :=
sorry

end NUMINAMATH_CALUDE_frog_climb_time_l1002_100231


namespace NUMINAMATH_CALUDE_number_problem_l1002_100262

theorem number_problem (x : ℝ) : 0.20 * x - 4 = 6 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1002_100262


namespace NUMINAMATH_CALUDE_other_communities_count_l1002_100214

theorem other_communities_count (total_boys : ℕ) 
  (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 850 →
  muslim_percent = 40 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ↑⌊(1 - (muslim_percent + hindu_percent + sikh_percent)) * total_boys⌋ = 187 :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l1002_100214


namespace NUMINAMATH_CALUDE_endangered_animal_population_l1002_100260

/-- The population of an endangered animal after n years, given an initial population and annual decrease rate. -/
def population (m : ℝ) (r : ℝ) (n : ℕ) : ℝ := m * (1 - r) ^ n

/-- Theorem stating that given specific conditions, the population after 3 years will be 5832. -/
theorem endangered_animal_population :
  let m : ℝ := 8000  -- Initial population
  let r : ℝ := 0.1   -- Annual decrease rate (10%)
  let n : ℕ := 3     -- Number of years
  population m r n = 5832 := by
  sorry

end NUMINAMATH_CALUDE_endangered_animal_population_l1002_100260


namespace NUMINAMATH_CALUDE_f_inequality_l1002_100278

noncomputable def f (x : ℝ) : ℝ := Real.log (|x| + 1) / Real.log (1/2) + 1 / (x^2 + 1)

theorem f_inequality (x : ℝ) : f x > f (2*x - 1) ↔ x > 1 ∨ x < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1002_100278


namespace NUMINAMATH_CALUDE_two_digit_odd_integers_count_l1002_100222

def odd_digits : Finset Nat := {1, 3, 5, 7, 9}

theorem two_digit_odd_integers_count :
  (Finset.filter
    (fun n => n ≥ 10 ∧ n < 100 ∧ n % 2 = 1 ∧
      (n / 10) ∈ odd_digits ∧ (n % 10) ∈ odd_digits ∧
      (n / 10) ≠ (n % 10))
    (Finset.range 100)).card = 20 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_odd_integers_count_l1002_100222


namespace NUMINAMATH_CALUDE_field_length_calculation_l1002_100201

theorem field_length_calculation (width : ℝ) (pond_side : ℝ) : 
  width > 0 →
  pond_side = 4 →
  2 * width * width = 8 * (pond_side * pond_side) →
  2 * width = 16 := by
sorry

end NUMINAMATH_CALUDE_field_length_calculation_l1002_100201


namespace NUMINAMATH_CALUDE_percentage_married_students_l1002_100261

theorem percentage_married_students (T : ℝ) (T_pos : T > 0) : 
  let male_students := 0.7 * T
  let female_students := 0.3 * T
  let married_male_students := (2/7) * male_students
  let married_female_students := (1/3) * female_students
  let total_married_students := married_male_students + married_female_students
  (total_married_students / T) * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_percentage_married_students_l1002_100261


namespace NUMINAMATH_CALUDE_system_solution_unique_l1002_100210

theorem system_solution_unique : 
  ∃! (x y : ℚ), (6 * x = -9 - 3 * y) ∧ (4 * x = 5 * y - 34) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1002_100210


namespace NUMINAMATH_CALUDE_percent_of_x_l1002_100235

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 10 + x / 25) / x * 100 = 14 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_l1002_100235


namespace NUMINAMATH_CALUDE_jim_buicks_count_l1002_100258

/-- The number of model cars Jim has for each brand -/
structure ModelCars where
  ford : ℕ
  buick : ℕ
  chevy : ℕ

/-- Jim's collection of model cars satisfying the given conditions -/
def jim_collection : ModelCars → Prop
  | ⟨f, b, c⟩ => f + b + c = 301 ∧ b = 4 * f ∧ f = 2 * c + 3

theorem jim_buicks_count (cars : ModelCars) (h : jim_collection cars) : cars.buick = 220 := by
  sorry

end NUMINAMATH_CALUDE_jim_buicks_count_l1002_100258


namespace NUMINAMATH_CALUDE_complex_roots_cubic_l1002_100219

theorem complex_roots_cubic (a b c : ℂ) 
  (h1 : a + b + c = 1)
  (h2 : a * b + a * c + b * c = 0)
  (h3 : a * b * c = -1) :
  (∀ x : ℂ, x^3 - x^2 + 1 = 0 ↔ x = a ∨ x = b ∨ x = c) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_cubic_l1002_100219


namespace NUMINAMATH_CALUDE_barbara_candies_l1002_100213

theorem barbara_candies (original_boxes : Nat) (original_candies_per_box : Nat)
                         (new_boxes : Nat) (new_candies_per_box : Nat) :
  original_boxes = 9 →
  original_candies_per_box = 25 →
  new_boxes = 18 →
  new_candies_per_box = 35 →
  original_boxes * original_candies_per_box + new_boxes * new_candies_per_box = 855 :=
by sorry

end NUMINAMATH_CALUDE_barbara_candies_l1002_100213


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l1002_100243

def numbers : List ℝ := [18, 27, 45]

theorem arithmetic_mean_of_numbers : 
  (List.sum numbers) / (List.length numbers) = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l1002_100243


namespace NUMINAMATH_CALUDE_base_eight_digits_of_512_l1002_100240

theorem base_eight_digits_of_512 : ∃ n : ℕ, n > 0 ∧ 8^(n-1) ≤ 512 ∧ 512 < 8^n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_digits_of_512_l1002_100240


namespace NUMINAMATH_CALUDE_scientific_notation_450_million_l1002_100208

theorem scientific_notation_450_million :
  (450000000 : ℝ) = 4.5 * (10 : ℝ)^8 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_450_million_l1002_100208


namespace NUMINAMATH_CALUDE_square_cut_in_half_l1002_100272

/-- A square with side length 8 is cut in half to create two congruent rectangles. -/
theorem square_cut_in_half (square_side : ℝ) (rect_width rect_height : ℝ) : 
  square_side = 8 →
  rect_width * rect_height = square_side * square_side / 2 →
  rect_width = square_side ∨ rect_height = square_side →
  (rect_width = 4 ∧ rect_height = 8) ∨ (rect_width = 8 ∧ rect_height = 4) := by
  sorry

end NUMINAMATH_CALUDE_square_cut_in_half_l1002_100272


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l1002_100233

def is_valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def is_unique_assignment (K P O C S R T : ℕ) : Prop :=
  is_valid_digit K ∧ is_valid_digit P ∧ is_valid_digit O ∧ 
  is_valid_digit C ∧ is_valid_digit S ∧ is_valid_digit R ∧
  is_valid_digit T ∧
  K ≠ P ∧ K ≠ O ∧ K ≠ C ∧ K ≠ S ∧ K ≠ R ∧ K ≠ T ∧
  P ≠ O ∧ P ≠ C ∧ P ≠ S ∧ P ≠ R ∧ P ≠ T ∧
  O ≠ C ∧ O ≠ S ∧ O ≠ R ∧ O ≠ T ∧
  C ≠ S ∧ C ≠ R ∧ C ≠ T ∧
  S ≠ R ∧ S ≠ T ∧
  R ≠ T

def satisfies_equation (K P O C S R T : ℕ) : Prop :=
  10000 * K + 1000 * P + 100 * O + 10 * C + C +
  10000 * K + 1000 * P + 100 * O + 10 * C + C =
  10000 * S + 1000 * P + 100 * O + 10 * R + T

theorem cryptarithm_solution :
  ∃! (K P O C S R T : ℕ),
    is_unique_assignment K P O C S R T ∧
    satisfies_equation K P O C S R T ∧
    K = 3 ∧ P = 5 ∧ O = 9 ∧ C = 7 ∧ S = 7 ∧ R = 5 ∧ T = 4 :=
sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l1002_100233


namespace NUMINAMATH_CALUDE_road_greening_costs_l1002_100276

/-- Represents a road greening project with two plans. -/
structure RoadGreeningProject where
  total_length : ℝ
  plan_a_type_a : ℝ
  plan_a_type_b : ℝ
  plan_a_cost : ℝ
  plan_b_type_a : ℝ
  plan_b_type_b : ℝ
  plan_b_cost : ℝ

/-- Calculates the cost per stem of type A and B flowers. -/
def calculate_flower_costs (project : RoadGreeningProject) : ℝ × ℝ := sorry

/-- Calculates the minimum total cost of the project. -/
def calculate_min_cost (project : RoadGreeningProject) : ℝ := sorry

/-- Theorem stating the correct flower costs and minimum project cost. -/
theorem road_greening_costs (project : RoadGreeningProject) 
  (h1 : project.total_length = 1500)
  (h2 : project.plan_a_type_a = 2)
  (h3 : project.plan_a_type_b = 3)
  (h4 : project.plan_a_cost = 22)
  (h5 : project.plan_b_type_a = 1)
  (h6 : project.plan_b_type_b = 5)
  (h7 : project.plan_b_cost = 25) :
  let (cost_a, cost_b) := calculate_flower_costs project
  calculate_flower_costs project = (5, 4) ∧ 
  calculate_min_cost project = 36000 := by
  sorry

end NUMINAMATH_CALUDE_road_greening_costs_l1002_100276


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1002_100282

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a b : Line) (α : Plane) :
  parallel a α → perpendicular b α → perpendicularLines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1002_100282
