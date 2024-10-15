import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_hypotenuse_length_l433_43311

theorem right_triangle_hypotenuse (a b : ℝ) (h_right : a > 0 ∧ b > 0) :
  (∃ (m_a m_b : ℝ), 
    m_a^2 = b^2 + (a/2)^2 ∧
    m_b^2 = a^2 + (b/2)^2 ∧
    m_a = 6 ∧
    m_b = Real.sqrt 34) →
  a^2 + b^2 = 56 :=
by sorry

theorem hypotenuse_length (a b : ℝ) (h_right : a > 0 ∧ b > 0) :
  a^2 + b^2 = 56 →
  Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_hypotenuse_length_l433_43311


namespace NUMINAMATH_CALUDE_physics_marks_l433_43313

/-- Represents the marks obtained in each subject --/
structure Marks where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ
  biology : ℝ
  computerScience : ℝ

/-- The conditions of the problem --/
def ProblemConditions (m : Marks) : Prop :=
  -- Average score across all subjects is 75
  (m.physics + m.chemistry + m.mathematics + m.biology + m.computerScience) / 5 = 75 ∧
  -- Average score in Physics, Mathematics, and Biology is 85
  (m.physics + m.mathematics + m.biology) / 3 = 85 ∧
  -- Average score in Physics, Chemistry, and Computer Science is 70
  (m.physics + m.chemistry + m.computerScience) / 3 = 70 ∧
  -- Weightages sum to 100%
  0.20 + 0.25 + 0.20 + 0.15 + 0.20 = 1

theorem physics_marks (m : Marks) (h : ProblemConditions m) : m.physics = 90 := by
  sorry

end NUMINAMATH_CALUDE_physics_marks_l433_43313


namespace NUMINAMATH_CALUDE_line_quadrant_theorem_l433_43330

/-- Represents a line in the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a line passes through a given quadrant -/
def passes_through_quadrant (l : Line) (q : ℕ) : Prop :=
  match q with
  | 1 => ∃ x > 0, l.slope * x + l.intercept > 0
  | 2 => ∃ x < 0, l.slope * x + l.intercept > 0
  | 3 => ∃ x < 0, l.slope * x + l.intercept < 0
  | 4 => ∃ x > 0, l.slope * x + l.intercept < 0
  | _ => False

/-- The main theorem -/
theorem line_quadrant_theorem (a b : ℝ) (h1 : a < 0) (h2 : b > 0) 
  (h3 : passes_through_quadrant (Line.mk a b) 1)
  (h4 : passes_through_quadrant (Line.mk a b) 2)
  (h5 : passes_through_quadrant (Line.mk a b) 4) :
  ¬ passes_through_quadrant (Line.mk b a) 2 := by
  sorry

end NUMINAMATH_CALUDE_line_quadrant_theorem_l433_43330


namespace NUMINAMATH_CALUDE_f_domain_correct_l433_43334

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * x^2) / Real.sqrt (1 - x) + Real.log (3 * x + 1)

-- Define the domain of f
def domain_f : Set ℝ := {x | -1/3 < x ∧ x < 1}

-- Theorem stating that domain_f is the correct domain for f
theorem f_domain_correct : 
  ∀ x : ℝ, x ∈ domain_f ↔ (∃ y : ℝ, f x = y) :=
sorry

end NUMINAMATH_CALUDE_f_domain_correct_l433_43334


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l433_43391

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides. -/
def pentagon_sides : ℕ := 5

/-- The sum of the interior angles of a pentagon is 540 degrees. -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon_sides = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l433_43391


namespace NUMINAMATH_CALUDE_markers_per_box_l433_43328

theorem markers_per_box (initial_markers : ℕ) (new_boxes : ℕ) (total_markers : ℕ)
  (h1 : initial_markers = 32)
  (h2 : new_boxes = 6)
  (h3 : total_markers = 86) :
  (total_markers - initial_markers) / new_boxes = 9 :=
by sorry

end NUMINAMATH_CALUDE_markers_per_box_l433_43328


namespace NUMINAMATH_CALUDE_expected_deviation_10_gt_100_l433_43324

/-- Represents the outcome of a coin toss experiment -/
structure CoinTossExperiment where
  n : ℕ  -- number of tosses
  m : ℕ  -- number of heads
  h_m_le_n : m ≤ n  -- ensure m is not greater than n

/-- The frequency of heads in a coin toss experiment -/
def frequency (e : CoinTossExperiment) : ℚ :=
  e.m / e.n

/-- The deviation of the frequency from the probability of a fair coin (0.5) -/
def deviation (e : CoinTossExperiment) : ℚ :=
  frequency e - 1/2

/-- The absolute deviation of the frequency from the probability of a fair coin (0.5) -/
def absoluteDeviation (e : CoinTossExperiment) : ℚ :=
  |deviation e|

/-- The expected value of the absolute deviation for n coin tosses -/
noncomputable def expectedAbsoluteDeviation (n : ℕ) : ℝ :=
  sorry  -- Definition not provided in the problem, so we leave it as sorry

/-- Theorem stating that the expected absolute deviation for 10 tosses
    is greater than for 100 tosses -/
theorem expected_deviation_10_gt_100 :
  expectedAbsoluteDeviation 10 > expectedAbsoluteDeviation 100 :=
by sorry

end NUMINAMATH_CALUDE_expected_deviation_10_gt_100_l433_43324


namespace NUMINAMATH_CALUDE_function_bounds_l433_43355

/-- Given a function f(x) = 1 - a cos x - b sin x - A cos 2x - B sin 2x,
    where a, b, A, B are real constants, and f(x) ≥ 0 for all real x,
    prove that a² + b² ≤ 2 and A² + B² ≤ 1. -/
theorem function_bounds (a b A B : ℝ) 
    (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_bounds_l433_43355


namespace NUMINAMATH_CALUDE_no_solution_implies_a_geq_6_l433_43377

theorem no_solution_implies_a_geq_6 (a : ℝ) : 
  (∀ x : ℝ, ¬(2*x - a > 0 ∧ 3*x - 4 < 5)) → a ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_geq_6_l433_43377


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l433_43385

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l433_43385


namespace NUMINAMATH_CALUDE_probability_no_adjacent_red_in_ring_l433_43305

/-- The number of red marbles -/
def num_red : ℕ := 4

/-- The number of blue marbles -/
def num_blue : ℕ := 8

/-- The total number of marbles -/
def total_marbles : ℕ := num_red + num_blue

/-- The probability of no two red marbles being adjacent when arranged in a ring -/
def probability_no_adjacent_red : ℚ := 7 / 33

/-- Theorem: The probability of no two red marbles being adjacent when 4 red marbles
    and 8 blue marbles are randomly arranged in a ring is 7/33 -/
theorem probability_no_adjacent_red_in_ring :
  probability_no_adjacent_red = 7 / 33 :=
by sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_red_in_ring_l433_43305


namespace NUMINAMATH_CALUDE_congruence_problem_l433_43392

theorem congruence_problem (y : ℤ) 
  (h1 : (2 + y) % (2^4) = 2^3 % (2^4))
  (h2 : (4 + y) % (4^3) = 4^2 % (4^3))
  (h3 : (6 + y) % (6^3) = 6^2 % (6^3)) :
  y % 48 = 44 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l433_43392


namespace NUMINAMATH_CALUDE_square_and_cube_roots_l433_43319

theorem square_and_cube_roots : 
  (∀ x : ℝ, x^2 = 81 → x = 3 ∨ x = -3) ∧ 
  (∀ y : ℝ, y^3 = -64/125 → y = -4/5) := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_roots_l433_43319


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l433_43343

/-- Given that the monomials 2a^(4)b^(-2m+7) and 3a^(2m)b^(n+2) are like terms, prove that m + n = 3 -/
theorem like_terms_exponent_sum (a b : ℝ) (m n : ℤ) 
  (h : ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → 2 * a^4 * b^(-2*m+7) = 3 * a^(2*m) * b^(n+2)) : 
  m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l433_43343


namespace NUMINAMATH_CALUDE_equation_solution_l433_43317

theorem equation_solution : ∃ x : ℝ, 20 - 3 * (x + 4) = 2 * (x - 1) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l433_43317


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l433_43382

theorem matrix_equation_proof : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0.5, 1]
  M^3 - 3 • M^2 + 4 • M = !![7, 14; 3.5, 7] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l433_43382


namespace NUMINAMATH_CALUDE_amusement_park_ticket_cost_l433_43316

/-- The cost of a single ticket to an amusement park -/
def ticket_cost : ℕ := sorry

/-- The number of people in the group -/
def num_people : ℕ := 4

/-- The cost of a set of snacks -/
def snack_cost : ℕ := 5

/-- The total cost for the group, including tickets and snacks -/
def total_cost : ℕ := 92

theorem amusement_park_ticket_cost :
  ticket_cost = 18 ∧
  total_cost = num_people * ticket_cost + num_people * snack_cost :=
sorry

end NUMINAMATH_CALUDE_amusement_park_ticket_cost_l433_43316


namespace NUMINAMATH_CALUDE_sandbox_width_l433_43390

/-- The width of a rectangle given its length and area -/
theorem sandbox_width (length : ℝ) (area : ℝ) (h1 : length = 312) (h2 : area = 45552) :
  area / length = 146 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_width_l433_43390


namespace NUMINAMATH_CALUDE_farm_dogs_left_l433_43398

/-- Given a farm with dogs and farmhands, calculates the number of dogs left after a morning walk. -/
def dogs_left_after_walk (total_dogs : ℕ) (dog_houses : ℕ) (farmhands : ℕ) (dogs_per_farmhand : ℕ) : ℕ :=
  total_dogs - farmhands * dogs_per_farmhand

/-- Proves that given the specific conditions of the farm, 144 dogs are left after the morning walk. -/
theorem farm_dogs_left : dogs_left_after_walk 156 22 6 2 = 144 := by
  sorry

#eval dogs_left_after_walk 156 22 6 2

end NUMINAMATH_CALUDE_farm_dogs_left_l433_43398


namespace NUMINAMATH_CALUDE_properties_of_A_l433_43309

def A : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2^k}

theorem properties_of_A :
  (∀ a ∈ A, ∀ b : ℕ, b < 2*a - 1 → ¬(2*a ∣ b*(b+1))) ∧
  (∀ a : ℕ, a ∉ A → a ≠ 1 → ∃ b : ℕ, b < 2*a - 1 ∧ (2*a ∣ b*(b+1))) :=
sorry

end NUMINAMATH_CALUDE_properties_of_A_l433_43309


namespace NUMINAMATH_CALUDE_residual_plot_ordinate_l433_43320

/-- Represents a residual plot used in residual analysis -/
structure ResidualPlot where
  /-- The ordinate of the residual plot -/
  ordinate : ℝ
  /-- The abscissa of the residual plot (could be sample number, height data, or estimated weight) -/
  abscissa : ℝ

/-- Represents a residual in statistical analysis -/
def Residual : Type := ℝ

/-- Theorem stating that the ordinate of a residual plot represents the residual -/
theorem residual_plot_ordinate (plot : ResidualPlot) : 
  ∃ (r : Residual), plot.ordinate = r :=
sorry

end NUMINAMATH_CALUDE_residual_plot_ordinate_l433_43320


namespace NUMINAMATH_CALUDE_union_equals_M_l433_43325

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = 2^x}

-- Define the set S
def S : Set ℝ := {x | ∃ y, y = x - 1}

-- State the theorem
theorem union_equals_M : M ∪ S = M :=
sorry

end NUMINAMATH_CALUDE_union_equals_M_l433_43325


namespace NUMINAMATH_CALUDE_divisor_probability_l433_43340

/-- The number of positive divisors of 10^99 -/
def total_divisors : ℕ := 10000

/-- The number of positive divisors of 10^99 that are multiples of 10^88 -/
def favorable_divisors : ℕ := 144

/-- The probability of a randomly chosen positive divisor of 10^99 being a multiple of 10^88 -/
def probability : ℚ := favorable_divisors / total_divisors

theorem divisor_probability :
  probability = 9 / 625 :=
sorry

end NUMINAMATH_CALUDE_divisor_probability_l433_43340


namespace NUMINAMATH_CALUDE_f_properties_l433_43397

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |x + 1|

-- Define the range of f
def range_f : Set ℝ := {y : ℝ | ∃ x, f x = y}

-- Theorem statement
theorem f_properties :
  (range_f = {y : ℝ | y ≥ 3/2}) ∧
  (∀ a : ℝ, a ∈ range_f → |a - 1| + |a + 1| > 3/(2*a) ∧ 3/(2*a) > 7/2 - 2*a) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l433_43397


namespace NUMINAMATH_CALUDE_min_value_x2_2xy_y2_l433_43337

theorem min_value_x2_2xy_y2 :
  (∀ x y : ℝ, x^2 + 2*x*y + y^2 ≥ 0) ∧
  (∃ x y : ℝ, x^2 + 2*x*y + y^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x2_2xy_y2_l433_43337


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l433_43326

/-- Given two plane vectors satisfying certain conditions, prove that the magnitude of their linear combination is √13. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  ‖a‖ = Real.sqrt 2 →
  b = (1, 0) →
  a • (a - 2 • b) = 0 →
  ‖2 • a + b‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l433_43326


namespace NUMINAMATH_CALUDE_expression_evaluation_l433_43314

theorem expression_evaluation : 
  ∃ ε > 0, |((10 * 1.8 - 2 * 1.5) / 0.3 + Real.rpow 3 (2/3) - Real.log 4) - 50.6938| < ε :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l433_43314


namespace NUMINAMATH_CALUDE_isabel_albums_l433_43301

theorem isabel_albums (phone_pics camera_pics pics_per_album : ℕ) 
  (h1 : phone_pics = 2)
  (h2 : camera_pics = 4)
  (h3 : pics_per_album = 2)
  : (phone_pics + camera_pics) / pics_per_album = 3 := by
  sorry

end NUMINAMATH_CALUDE_isabel_albums_l433_43301


namespace NUMINAMATH_CALUDE_difference_calculations_l433_43380

theorem difference_calculations (d1 d2 d3 : Int) 
  (h1 : d1 = -15)
  (h2 : d2 = 405)
  (h3 : d3 = 1280) :
  let sum := d1 + d2 + d3
  let product := d1 * d2 * d3
  let avg_squares := ((d1^2 + d2^2 + d3^2) : ℚ) / 3
  sum = 1670 ∧ 
  product = -7728000 ∧ 
  avg_squares = 600883 + 1/3 ∧
  (product : ℚ) - avg_squares = -8328883 - 1/3 := by
sorry

#eval (-15 : Int) + 405 + 1280
#eval (-15 : Int) * 405 * 1280
#eval ((-15 : ℚ)^2 + 405^2 + 1280^2) / 3
#eval (-7728000 : ℚ) - (((-15 : ℚ)^2 + 405^2 + 1280^2) / 3)

end NUMINAMATH_CALUDE_difference_calculations_l433_43380


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_l433_43357

def sum_odd_integers (n : ℕ) : ℕ :=
  let count := (n + 1) / 2
  count * (1 + n) / 2

def sum_even_integers (n : ℕ) : ℕ :=
  let count := n / 2
  count * (2 + n) / 2

theorem odd_even_sum_difference : sum_odd_integers 215 - sum_even_integers 100 = 9114 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_l433_43357


namespace NUMINAMATH_CALUDE_bess_frisbee_throws_l433_43395

/-- The problem of determining how many times Bess throws the Frisbee -/
theorem bess_frisbee_throws :
  ∀ (bess_throw_distance : ℕ) 
    (holly_throw_distance : ℕ) 
    (holly_throw_count : ℕ) 
    (total_distance : ℕ),
  bess_throw_distance = 20 →
  holly_throw_distance = 8 →
  holly_throw_count = 5 →
  total_distance = 200 →
  ∃ (bess_throw_count : ℕ),
    bess_throw_count * (2 * bess_throw_distance) + 
    holly_throw_count * holly_throw_distance = total_distance ∧
    bess_throw_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_bess_frisbee_throws_l433_43395


namespace NUMINAMATH_CALUDE_f_negative_five_equals_negative_five_l433_43342

/-- Given a function f(x) = a * sin(x) + b * tan(x) + 1 where f(5) = 7, prove that f(-5) = -5 -/
theorem f_negative_five_equals_negative_five 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * Real.sin x + b * Real.tan x + 1)
  (h2 : f 5 = 7) :
  f (-5) = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_five_equals_negative_five_l433_43342


namespace NUMINAMATH_CALUDE_smallest_number_l433_43381

-- Define the base conversion function
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def A : Nat := to_decimal [1, 1, 1, 1] 2
def B : Nat := to_decimal [0, 1, 2] 6
def C : Nat := to_decimal [0, 0, 0, 1] 4
def D : Nat := to_decimal [1, 0, 1] 8

-- Theorem statement
theorem smallest_number : A < B ∧ A < C ∧ A < D := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l433_43381


namespace NUMINAMATH_CALUDE_problem_statement_l433_43370

theorem problem_statement : ((16^15 / 16^14)^3 * 8^3) / 2^9 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l433_43370


namespace NUMINAMATH_CALUDE_cubic_roots_theorem_l433_43351

open Complex

-- Define the cubic equation
def cubic_equation (p q x : ℂ) : Prop := x^3 + p*x + q = 0

-- Define the condition for roots forming an equilateral triangle
def roots_form_equilateral_triangle (r₁ r₂ r₃ : ℂ) : Prop :=
  abs (r₁ - r₂) = Real.sqrt 3 ∧
  abs (r₂ - r₃) = Real.sqrt 3 ∧
  abs (r₃ - r₁) = Real.sqrt 3

theorem cubic_roots_theorem (p q : ℂ) :
  (∃ r₁ r₂ r₃ : ℂ, 
    cubic_equation p q r₁ ∧
    cubic_equation p q r₂ ∧
    cubic_equation p q r₃ ∧
    roots_form_equilateral_triangle r₁ r₂ r₃) →
  arg q = 2 * Real.pi / 3 →
  p + q = -1/2 + (Real.sqrt 3 / 2) * I :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_theorem_l433_43351


namespace NUMINAMATH_CALUDE_cafeteria_sales_comparison_l433_43367

def arithmetic_growth (initial : ℝ) (increment : ℝ) (periods : ℕ) : ℝ :=
  initial + increment * periods

def geometric_growth (initial : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  initial * (1 + rate) ^ periods

theorem cafeteria_sales_comparison
  (initial : ℝ)
  (increment : ℝ)
  (rate : ℝ)
  (h1 : initial > 0)
  (h2 : increment > 0)
  (h3 : rate > 0)
  (h4 : arithmetic_growth initial increment 8 = geometric_growth initial rate 8) :
  arithmetic_growth initial increment 4 > geometric_growth initial rate 4 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_sales_comparison_l433_43367


namespace NUMINAMATH_CALUDE_henry_skittles_l433_43353

theorem henry_skittles (bridget_initial : ℕ) (bridget_final : ℕ) (henry : ℕ) : 
  bridget_initial = 4 → 
  bridget_final = 8 → 
  bridget_final = bridget_initial + henry → 
  henry = 4 := by
sorry

end NUMINAMATH_CALUDE_henry_skittles_l433_43353


namespace NUMINAMATH_CALUDE_continuous_compound_interest_interest_rate_problem_l433_43341

/-- The annual interest rate for a continuously compounded investment --/
noncomputable def annual_interest_rate (initial_investment : ℝ) (final_amount : ℝ) (years : ℝ) : ℝ :=
  (Real.log (final_amount / initial_investment)) / years

/-- Theorem stating the relationship between the initial investment, final amount, time, and interest rate --/
theorem continuous_compound_interest
  (initial_investment : ℝ)
  (final_amount : ℝ)
  (years : ℝ)
  (h1 : initial_investment > 0)
  (h2 : final_amount > initial_investment)
  (h3 : years > 0) :
  final_amount = initial_investment * Real.exp (years * annual_interest_rate initial_investment final_amount years) :=
by sorry

/-- The specific problem instance --/
theorem interest_rate_problem :
  let initial_investment : ℝ := 5000
  let final_amount : ℝ := 8500
  let years : ℝ := 10
  8500 = 5000 * Real.exp (10 * annual_interest_rate 5000 8500 10) :=
by sorry

end NUMINAMATH_CALUDE_continuous_compound_interest_interest_rate_problem_l433_43341


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l433_43358

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 3 + a 4 = 9 →
  a 7 = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l433_43358


namespace NUMINAMATH_CALUDE_chemistry_marks_proof_l433_43335

def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 72
def biology_marks : ℕ := 82
def average_marks : ℕ := 71
def total_subjects : ℕ := 5

theorem chemistry_marks_proof :
  ∃ (chemistry_marks : ℕ),
    chemistry_marks = average_marks * total_subjects - (english_marks + math_marks + physics_marks + biology_marks) ∧
    chemistry_marks = 65 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_marks_proof_l433_43335


namespace NUMINAMATH_CALUDE_hands_count_l433_43372

/-- The number of students in Peter's class, including Peter. -/
def total_students : ℕ := 11

/-- The number of hands each student has. -/
def hands_per_student : ℕ := 2

/-- The number of hands in Peter's class, not including his. -/
def hands_in_class : ℕ := (total_students - 1) * hands_per_student

theorem hands_count : hands_in_class = 20 := by
  sorry

end NUMINAMATH_CALUDE_hands_count_l433_43372


namespace NUMINAMATH_CALUDE_equation_root_implies_a_value_l433_43315

theorem equation_root_implies_a_value (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 * x - 1) / (x - 3) = a / (3 - x) - 1) →
  a = -8 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_implies_a_value_l433_43315


namespace NUMINAMATH_CALUDE_regular_hexagon_most_symmetry_l433_43373

-- Define the types of polygons
inductive Polygon
  | RegularPentagon
  | IrregularHexagon
  | RegularHexagon
  | IrregularPentagon
  | EquilateralTriangle

-- Function to get the number of lines of symmetry for each polygon
def linesOfSymmetry (p : Polygon) : ℕ :=
  match p with
  | Polygon.RegularPentagon => 5
  | Polygon.IrregularHexagon => 0
  | Polygon.RegularHexagon => 6
  | Polygon.IrregularPentagon => 0
  | Polygon.EquilateralTriangle => 3

-- Theorem stating that the regular hexagon has the most lines of symmetry
theorem regular_hexagon_most_symmetry :
  ∀ p : Polygon, linesOfSymmetry Polygon.RegularHexagon ≥ linesOfSymmetry p :=
by sorry

end NUMINAMATH_CALUDE_regular_hexagon_most_symmetry_l433_43373


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l433_43383

theorem four_digit_number_problem :
  ∀ N : ℕ,
  (1000 ≤ N) ∧ (N < 10000) →
  (∃ x y : ℕ,
    (1 ≤ x) ∧ (x ≤ 9) ∧
    (100 ≤ y) ∧ (y < 1000) ∧
    (N = 1000 * x + y) ∧
    (N / y = 3) ∧
    (N % y = 8)) →
  (N = 1496 ∨ N = 2996) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l433_43383


namespace NUMINAMATH_CALUDE_f_monotone_increasing_and_g_bound_l433_43306

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (2 * x) / (x + 2)

noncomputable def g (x : ℝ) : ℝ := f x - 4 / (x + 2)

theorem f_monotone_increasing_and_g_bound (a : ℝ) :
  (∀ x > 0, Monotone f) ∧
  (∀ x > 0, g x < x + a ↔ a > -3) := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_and_g_bound_l433_43306


namespace NUMINAMATH_CALUDE_distance_Cara_approx_l433_43300

/-- The distance between two skaters on a frozen lake --/
def distance_CD : ℝ := 100

/-- Cara's skating speed in meters per second --/
def speed_Cara : ℝ := 9

/-- Danny's skating speed in meters per second --/
def speed_Danny : ℝ := 6

/-- The angle between Cara's path and the line CD in degrees --/
def angle_Cara : ℝ := 75

/-- The time it takes for Cara and Danny to meet --/
noncomputable def meeting_time : ℝ := 
  let a : ℝ := 45
  let b : ℝ := -1800 * Real.cos (angle_Cara * Real.pi / 180)
  let c : ℝ := 10000
  (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)

/-- The distance Cara skates before meeting Danny --/
noncomputable def distance_Cara : ℝ := speed_Cara * meeting_time

/-- Theorem stating that the distance Cara skates is approximately 27.36144 meters --/
theorem distance_Cara_approx : 
  ∃ ε > 0, abs (distance_Cara - 27.36144) < ε :=
by sorry

end NUMINAMATH_CALUDE_distance_Cara_approx_l433_43300


namespace NUMINAMATH_CALUDE_intersection_sum_l433_43362

theorem intersection_sum (a b : ℝ) : 
  (3 = (1/3) * 6 + a) → 
  (6 = (1/3) * 3 + b) → 
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l433_43362


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l433_43339

theorem isosceles_triangle_base_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- triangle sides are positive
  a = b →                  -- isosceles triangle condition
  a = 5 →                  -- given leg length
  a + b > c →              -- triangle inequality
  c + a > b →              -- triangle inequality
  c ≠ 11                   -- base cannot be 11
  := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l433_43339


namespace NUMINAMATH_CALUDE_pears_for_20_apples_is_13_l433_43332

/-- The price of fruits in an arbitrary unit -/
structure FruitPrices where
  apple : ℚ
  orange : ℚ
  pear : ℚ

/-- Given the conditions of the problem, calculate the number of pears
    that can be bought for the price of 20 apples -/
def pears_for_20_apples (prices : FruitPrices) : ℕ :=
  sorry

/-- Theorem stating the result of the calculation -/
theorem pears_for_20_apples_is_13 (prices : FruitPrices) 
  (h1 : 10 * prices.apple = 5 * prices.orange)
  (h2 : 3 * prices.orange = 4 * prices.pear) :
  pears_for_20_apples prices = 13 := by
  sorry

end NUMINAMATH_CALUDE_pears_for_20_apples_is_13_l433_43332


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l433_43368

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 20 - 7 * a) : 
  a = 20 / 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l433_43368


namespace NUMINAMATH_CALUDE_matrix_property_implies_k_one_and_n_even_l433_43346

open Matrix

theorem matrix_property_implies_k_one_and_n_even 
  (k n : ℕ) 
  (hk : k ≥ 1) 
  (hn : n ≥ 2) 
  (A B : Matrix (Fin n) (Fin n) ℤ) 
  (h1 : A ^ 3 = 0)
  (h2 : A ^ k * B + B * A = 1) :
  k = 1 ∧ Even n :=
sorry

end NUMINAMATH_CALUDE_matrix_property_implies_k_one_and_n_even_l433_43346


namespace NUMINAMATH_CALUDE_library_shelves_l433_43321

theorem library_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 14240) (h2 : books_per_shelf = 8) :
  total_books / books_per_shelf = 1780 := by
sorry

end NUMINAMATH_CALUDE_library_shelves_l433_43321


namespace NUMINAMATH_CALUDE_problem_statement_l433_43354

theorem problem_statement (a b : ℝ) (h : 2 * a - 3 * b = 5) :
  4 * a^2 - 9 * b^2 - 30 * b + 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l433_43354


namespace NUMINAMATH_CALUDE_yi_jianlian_shots_l433_43323

/-- Given the basketball game statistics of Yi Jianlian, prove the number of two-point shots and free throws --/
theorem yi_jianlian_shots (total_shots : ℕ) (total_points : ℕ) (three_pointers : ℕ) 
  (h1 : total_shots = 16)
  (h2 : total_points = 28)
  (h3 : three_pointers = 3) :
  ∃ (two_pointers free_throws : ℕ),
    two_pointers + free_throws + three_pointers = total_shots ∧
    2 * two_pointers + free_throws + 3 * three_pointers = total_points ∧
    two_pointers = 6 ∧
    free_throws = 7 := by
  sorry

end NUMINAMATH_CALUDE_yi_jianlian_shots_l433_43323


namespace NUMINAMATH_CALUDE_total_money_after_redistribution_l433_43386

/-- Represents the money redistribution process among three friends. -/
def moneyRedistribution (initialAmy : ℝ) (initialJan : ℝ) (initialToy : ℝ) : ℝ :=
  let afterAmy := initialAmy - 2 * (initialJan + initialToy) + 3 * initialJan + 3 * initialToy
  let afterJan := 3 * (initialAmy - 2 * (initialJan + initialToy)) + 
                  (3 * initialJan - 2 * (initialAmy - 2 * (initialJan + initialToy) + 3 * initialToy)) + 
                  3 * 3 * initialToy
  let afterToy := 27  -- Given condition
  afterAmy + afterJan + afterToy

/-- Theorem stating that the total amount after redistribution is 243 when Toy starts and ends with 27. -/
theorem total_money_after_redistribution :
  ∀ (initialAmy : ℝ) (initialJan : ℝ),
  moneyRedistribution initialAmy initialJan 27 = 243 :=
by
  sorry

#eval moneyRedistribution 0 0 27  -- For verification

end NUMINAMATH_CALUDE_total_money_after_redistribution_l433_43386


namespace NUMINAMATH_CALUDE_taco_castle_parking_lot_l433_43375

/-- The number of Volkswagen Bugs in the parking lot of Taco Castle -/
def volkswagen_bugs (dodge ford toyota : ℕ) : ℕ :=
  toyota / 2

theorem taco_castle_parking_lot (dodge ford toyota : ℕ) 
  (h1 : ford = dodge / 3)
  (h2 : ford = toyota * 2)
  (h3 : dodge = 60) :
  volkswagen_bugs dodge ford toyota = 5 := by
sorry

end NUMINAMATH_CALUDE_taco_castle_parking_lot_l433_43375


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l433_43329

/-- Given two similar right triangles, where one has legs 12 and 9, and the other has legs x and 6, prove that x = 8 -/
theorem similar_triangles_leg_length : 
  ∀ x : ℝ, 
  (12 : ℝ) / x = (9 : ℝ) / 6 → 
  x = 8 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l433_43329


namespace NUMINAMATH_CALUDE_shortest_distance_proof_l433_43389

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 - y - 2 * Real.log (Real.sqrt x) = 0

-- Define the line
def line (x y : ℝ) : Prop := 4*x + 4*y + 1 = 0

-- Define the shortest distance function
noncomputable def shortest_distance : ℝ := (Real.sqrt 2 / 2) * (1 + Real.log 2)

-- Theorem statement
theorem shortest_distance_proof :
  ∀ (x y : ℝ), curve x y →
  ∃ (d : ℝ), d ≥ 0 ∧ 
    (∀ (x' y' : ℝ), line x' y' → 
      d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) ∧
    d = shortest_distance :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_proof_l433_43389


namespace NUMINAMATH_CALUDE_max_distance_ratio_l433_43304

/-- The maximum ratio of distances from a point on a circle to two fixed points -/
theorem max_distance_ratio : 
  let A : ℝ × ℝ := (0, -2)
  let B : ℝ × ℝ := (1, -1)
  let circle := {P : ℝ × ℝ | P.1^2 + P.2^2 = 2}
  ∃ (max : ℝ), max = (3 * Real.sqrt 2) / 2 ∧ 
    ∀ P ∈ circle, 
      Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) / Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) ≤ max :=
by sorry


end NUMINAMATH_CALUDE_max_distance_ratio_l433_43304


namespace NUMINAMATH_CALUDE_a3_range_l433_43359

/-- A sequence {aₙ} is convex if (aₙ + aₙ₊₂)/2 ≤ aₙ₊₁ for all positive integers n. -/
def is_convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a n + a (n + 2)) / 2 ≤ a (n + 1)

/-- The function bₙ = n² - 6n + 10 -/
def b (n : ℕ) : ℝ := (n : ℝ)^2 - 6*(n : ℝ) + 10

theorem a3_range (a : ℕ → ℝ) 
  (h_convex : is_convex_sequence a)
  (h_a1 : a 1 = 1)
  (h_a10 : a 10 = 28)
  (h_bound : ∀ n : ℕ, 1 ≤ n → n < 10 → |a n - b n| ≤ 20) :
  7 ≤ a 3 ∧ a 3 ≤ 19 := by sorry

end NUMINAMATH_CALUDE_a3_range_l433_43359


namespace NUMINAMATH_CALUDE_min_value_theorem_l433_43303

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z > 0 → z + y = 1 →
    2 / (z + 3 * y) + 1 / (z - y) ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l433_43303


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l433_43387

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b, (a - b)^3 * b^2 > 0 → a > b) ∧
  (∃ a b, a > b ∧ (a - b)^3 * b^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l433_43387


namespace NUMINAMATH_CALUDE_weight_difference_e_d_l433_43366

/-- Given weights of individuals A, B, C, D, and E, prove that E weighs 3 kg more than D -/
theorem weight_difference_e_d (w_a w_b w_c w_d w_e : ℝ) : 
  (w_a + w_b + w_c) / 3 = 60 →
  (w_a + w_b + w_c + w_d) / 4 = 65 →
  (w_b + w_c + w_d + w_e) / 4 = 64 →
  w_a = 87 →
  w_e - w_d = 3 := by
sorry

end NUMINAMATH_CALUDE_weight_difference_e_d_l433_43366


namespace NUMINAMATH_CALUDE_tank_volume_ratio_l433_43333

theorem tank_volume_ratio : 
  ∀ (tank1_volume tank2_volume : ℝ), 
  tank1_volume > 0 → tank2_volume > 0 →
  (3/4 : ℝ) * tank1_volume = (5/8 : ℝ) * tank2_volume →
  tank1_volume / tank2_volume = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_tank_volume_ratio_l433_43333


namespace NUMINAMATH_CALUDE_fake_to_total_handbags_ratio_l433_43312

theorem fake_to_total_handbags_ratio
  (total_purses : ℕ)
  (total_handbags : ℕ)
  (authentic_items : ℕ)
  (h1 : total_purses = 26)
  (h2 : total_handbags = 24)
  (h3 : authentic_items = 31)
  (h4 : total_purses / 2 = total_purses - authentic_items + total_handbags - authentic_items) :
  (total_handbags - (authentic_items - total_purses / 2)) / total_handbags = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_fake_to_total_handbags_ratio_l433_43312


namespace NUMINAMATH_CALUDE_first_number_is_five_l433_43308

/-- A sequence where each sum is 1 less than the actual sum of two numbers -/
def SpecialSequence (seq : List (ℕ × ℕ × ℕ)) : Prop :=
  ∀ (a b c : ℕ), (a, b, c) ∈ seq → a + b = c + 1

/-- The first equation in the sequence is x + 7 = 12 -/
def FirstEquation (x : ℕ) : Prop :=
  x + 7 = 12

theorem first_number_is_five (seq : List (ℕ × ℕ × ℕ)) (x : ℕ) 
  (h1 : SpecialSequence seq) (h2 : FirstEquation x) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_number_is_five_l433_43308


namespace NUMINAMATH_CALUDE_functional_equation_solution_l433_43310

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  f 0 = 1 ∧ ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2

/-- The main theorem stating that any function satisfying the functional equation
    must be of the form f(x) = x + 1 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
    ∀ x : ℝ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l433_43310


namespace NUMINAMATH_CALUDE_average_hours_upside_down_per_month_l433_43394

/-- The number of inches Alex needs to grow to ride the roller coaster -/
def height_difference : ℚ := 54 - 48

/-- Alex's normal growth rate in inches per month -/
def normal_growth_rate : ℚ := 1 / 3

/-- Alex's growth rate in inches per hour when hanging upside down -/
def upside_down_growth_rate : ℚ := 1 / 12

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Theorem stating the average number of hours Alex needs to hang upside down per month -/
theorem average_hours_upside_down_per_month :
  (height_difference - normal_growth_rate * months_per_year) / (upside_down_growth_rate * months_per_year) = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_hours_upside_down_per_month_l433_43394


namespace NUMINAMATH_CALUDE_no_mutually_exclusive_sets_l433_43348

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Yellow

/-- Represents the outcome of drawing two balls -/
def TwoBallDraw := (BallColor × BallColor)

/-- The set of all possible outcomes when drawing two balls from a bag with two white and two yellow balls -/
def SampleSpace : Set TwoBallDraw := sorry

/-- Event: At least one white ball -/
def AtLeastOneWhite (draw : TwoBallDraw) : Prop := 
  draw.1 = BallColor.White ∨ draw.2 = BallColor.White

/-- Event: At least one yellow ball -/
def AtLeastOneYellow (draw : TwoBallDraw) : Prop := 
  draw.1 = BallColor.Yellow ∨ draw.2 = BallColor.Yellow

/-- Event: Both balls are yellow -/
def BothYellow (draw : TwoBallDraw) : Prop := 
  draw.1 = BallColor.Yellow ∧ draw.2 = BallColor.Yellow

/-- Event: Exactly one white ball and one yellow ball -/
def OneWhiteOneYellow (draw : TwoBallDraw) : Prop := 
  (draw.1 = BallColor.White ∧ draw.2 = BallColor.Yellow) ∨
  (draw.1 = BallColor.Yellow ∧ draw.2 = BallColor.White)

/-- The three sets of events -/
def EventSet1 := {draw : TwoBallDraw | AtLeastOneWhite draw ∧ AtLeastOneYellow draw}
def EventSet2 := {draw : TwoBallDraw | AtLeastOneYellow draw ∧ BothYellow draw}
def EventSet3 := {draw : TwoBallDraw | OneWhiteOneYellow draw}

/-- Two events are mutually exclusive if their intersection is empty -/
def MutuallyExclusive (A B : Set TwoBallDraw) : Prop := A ∩ B = ∅

theorem no_mutually_exclusive_sets : 
  ¬(MutuallyExclusive EventSet1 EventSet2) ∧ 
  ¬(MutuallyExclusive EventSet1 EventSet3) ∧ 
  ¬(MutuallyExclusive EventSet2 EventSet3) := by sorry

end NUMINAMATH_CALUDE_no_mutually_exclusive_sets_l433_43348


namespace NUMINAMATH_CALUDE_exists_n_plus_Sn_eq_1980_consecutive_n_plus_Sn_l433_43356

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem 1: There exists a natural number n such that n + S(n) = 1980
theorem exists_n_plus_Sn_eq_1980 : ∃ n : ℕ, n + S n = 1980 := by sorry

-- Theorem 2: For any natural number k, either k or k+1 can be expressed as n + S(n)
theorem consecutive_n_plus_Sn : ∀ k : ℕ, (∃ n : ℕ, n + S n = k) ∨ (∃ n : ℕ, n + S n = k + 1) := by sorry

end NUMINAMATH_CALUDE_exists_n_plus_Sn_eq_1980_consecutive_n_plus_Sn_l433_43356


namespace NUMINAMATH_CALUDE_total_games_proof_l433_43338

/-- The number of baseball games Benny's high school played -/
def total_games (games_attended games_missed : ℕ) : ℕ :=
  games_attended + games_missed

/-- Theorem stating that the total number of games is the sum of attended and missed games -/
theorem total_games_proof (games_attended games_missed : ℕ) :
  total_games games_attended games_missed = games_attended + games_missed :=
by sorry

end NUMINAMATH_CALUDE_total_games_proof_l433_43338


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l433_43307

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -2)
  (z_ge : z ≥ -3) :
  ∃ (max : ℝ), max = 6 * Real.sqrt 3 ∧
  ∀ (a b c : ℝ), a + b + c = 3 → a ≥ -1 → b ≥ -2 → c ≥ -3 →
  Real.sqrt (4 * a + 4) + Real.sqrt (4 * b + 8) + Real.sqrt (4 * c + 12) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l433_43307


namespace NUMINAMATH_CALUDE_wendy_full_face_time_l433_43349

/-- Calculates the total time for Wendy's "full face" routine -/
def fullFaceTime (numProducts : ℕ) (waitTime : ℕ) (makeupTime : ℕ) : ℕ :=
  numProducts * waitTime + makeupTime

/-- Theorem: Wendy's "full face" routine takes 55 minutes -/
theorem wendy_full_face_time :
  fullFaceTime 5 5 30 = 55 := by
  sorry

end NUMINAMATH_CALUDE_wendy_full_face_time_l433_43349


namespace NUMINAMATH_CALUDE_sin_cos_sum_one_l433_43371

theorem sin_cos_sum_one (x : ℝ) :
  0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0 ∨ x = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_one_l433_43371


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l433_43364

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - 7*I) / (4 - I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l433_43364


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l433_43393

theorem arithmetic_calculations :
  ((-8) - 5 + (-4) - (-10) = -7) ∧
  (18 - 6 / (-2) * (-1/3) = 17) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l433_43393


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l433_43365

/-- Given a geometric sequence with first term a and common ratio r,
    this function returns the nth term of the sequence. -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- Theorem stating that the 7th term of a geometric sequence
    with first term 5 and second term -1 is 1/3125 -/
theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 5
  let a₂ : ℚ := -1
  let r : ℚ := a₂ / a₁
  geometric_term a₁ r 7 = 1 / 3125 := by
  sorry


end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l433_43365


namespace NUMINAMATH_CALUDE_banana_apple_worth_l433_43345

theorem banana_apple_worth (banana_worth : ℚ) :
  (3 / 4 * 12 : ℚ) * banana_worth = 6 →
  (1 / 4 * 8 : ℚ) * banana_worth = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_banana_apple_worth_l433_43345


namespace NUMINAMATH_CALUDE_curve_equation_represents_quadrants_l433_43322

-- Define the circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the right quadrant of the circle
def right_quadrant (x y : ℝ) : Prop := x = Real.sqrt (1 - y^2) ∧ x ≥ 0

-- Define the lower quadrant of the circle
def lower_quadrant (x y : ℝ) : Prop := y = -Real.sqrt (1 - x^2) ∧ y ≤ 0

-- Theorem stating the equation represents the right and lower quadrants of the unit circle
theorem curve_equation_represents_quadrants :
  ∀ x y : ℝ, unit_circle x y →
  ((x - Real.sqrt (1 - y^2)) * (y + Real.sqrt (1 - x^2)) = 0) ↔
  (right_quadrant x y ∨ lower_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_curve_equation_represents_quadrants_l433_43322


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l433_43361

theorem smallest_k_no_real_roots : ∀ k : ℤ,
  (∀ x : ℝ, (1/2 : ℝ) * x^2 + 3*x + (k : ℝ) ≠ 0) ↔ k ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l433_43361


namespace NUMINAMATH_CALUDE_initial_hno3_concentration_l433_43378

/-- Proves that the initial concentration of HNO3 is 35% given the problem conditions -/
theorem initial_hno3_concentration
  (initial_volume : ℝ)
  (pure_hno3_added : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_volume = 60)
  (h2 : pure_hno3_added = 18)
  (h3 : final_concentration = 50)
  : ∃ (initial_concentration : ℝ),
    initial_concentration = 35 ∧
    (initial_concentration / 100) * initial_volume + pure_hno3_added =
    (final_concentration / 100) * (initial_volume + pure_hno3_added) :=
by sorry

end NUMINAMATH_CALUDE_initial_hno3_concentration_l433_43378


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_i_minus_one_l433_43374

theorem imaginary_part_of_i_squared_times_i_minus_one (i : ℂ) : 
  i^2 = -1 → Complex.im (i^2 * (i - 1)) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_i_minus_one_l433_43374


namespace NUMINAMATH_CALUDE_eunji_uncle_money_l433_43376

/-- The amount of money Eunji received from her uncle -/
def uncle_money : ℕ := sorry

/-- The amount of money Eunji received from her mother -/
def mother_money : ℕ := 550

/-- The total amount of money Eunji has after receiving money from her mother -/
def total_money : ℕ := 1000

/-- Theorem stating that Eunji received 900 won from her uncle -/
theorem eunji_uncle_money :
  uncle_money = 900 ∧
  uncle_money / 2 + mother_money = total_money :=
sorry

end NUMINAMATH_CALUDE_eunji_uncle_money_l433_43376


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l433_43363

theorem modulo_residue_problem :
  (250 * 15 - 337 * 5 + 22) % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l433_43363


namespace NUMINAMATH_CALUDE_committee_selection_probability_l433_43369

theorem committee_selection_probability :
  let total_members : ℕ := 9
  let english_teachers : ℕ := 3
  let select_count : ℕ := 2

  let total_combinations := total_members.choose select_count
  let english_combinations := english_teachers.choose select_count

  (english_combinations : ℚ) / total_combinations = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_committee_selection_probability_l433_43369


namespace NUMINAMATH_CALUDE_pear_count_l433_43331

theorem pear_count (initial_apples : ℕ) (apple_removal_rate : ℚ) (pear_removal_rate : ℚ) :
  initial_apples = 160 →
  apple_removal_rate = 3/4 →
  pear_removal_rate = 1/3 →
  (initial_apples * (1 - apple_removal_rate) : ℚ) = (1/2 : ℚ) * (initial_pears * (1 - pear_removal_rate) : ℚ) →
  initial_pears = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_pear_count_l433_43331


namespace NUMINAMATH_CALUDE_f_neg_two_eq_eleven_l433_43336

/-- The function f(x) = x^2 - 3x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem: f(-2) = 11 -/
theorem f_neg_two_eq_eleven : f (-2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_eleven_l433_43336


namespace NUMINAMATH_CALUDE_acute_angles_trig_identities_l433_43344

theorem acute_angles_trig_identities (α β : Real) 
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_tan_α : Real.tan α = 4/3)
  (h_cos_sum : Real.cos (α + β) = -Real.sqrt 5 / 5) :
  Real.cos (2*α) = -7/25 ∧ Real.tan (α - β) = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_trig_identities_l433_43344


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_plus_one_l433_43318

theorem cube_plus_reciprocal_cube_plus_one (m : ℝ) (h : m + 1/m = 10) : 
  m^3 + 1/m^3 + 1 = 971 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_plus_one_l433_43318


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_five_sixths_l433_43302

theorem sum_of_fractions_equals_five_sixths :
  let sum : ℚ := (1 / (1 * 2)) + (1 / (2 * 3)) + (1 / (3 * 4)) + (1 / (4 * 5)) + (1 / (5 * 6))
  sum = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_five_sixths_l433_43302


namespace NUMINAMATH_CALUDE_percentage_difference_l433_43396

theorem percentage_difference (w q y z P : ℝ) : 
  w = q * (1 - P / 100) →
  q = y * 0.6 →
  z = y * 0.54 →
  z = w * 1.5 →
  P = 78.4 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l433_43396


namespace NUMINAMATH_CALUDE_power_multiplication_l433_43352

theorem power_multiplication (x : ℝ) : x^2 * x^4 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l433_43352


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l433_43327

theorem abs_sum_inequality (x : ℝ) : |x + 3| + |x - 4| < 8 ↔ 4 ≤ x ∧ x < 4.5 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l433_43327


namespace NUMINAMATH_CALUDE_line_intersection_l433_43360

theorem line_intersection : ∃! p : ℚ × ℚ, 
  5 * p.1 - 3 * p.2 = 7 ∧ 
  8 * p.1 + 2 * p.2 = 22 :=
by
  -- The point (40/17, 27/17) satisfies both equations
  have h1 : 5 * (40/17) - 3 * (27/17) = 7 := by sorry
  have h2 : 8 * (40/17) + 2 * (27/17) = 22 := by sorry

  -- Prove uniqueness
  sorry

end NUMINAMATH_CALUDE_line_intersection_l433_43360


namespace NUMINAMATH_CALUDE_division_problem_l433_43388

theorem division_problem (A : ℕ) : 
  (11 / A = 3) ∧ (11 % A = 2) → A = 3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l433_43388


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l433_43384

theorem min_value_of_expression (x y : ℝ) : (x * y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

theorem lower_bound_achievable : ∃ x y : ℝ, (x * y - 1)^2 + (x + y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l433_43384


namespace NUMINAMATH_CALUDE_sixth_score_achieves_target_mean_l433_43350

def existing_scores : List ℝ := [76, 82, 79, 84, 91]
def target_mean : ℝ := 85
def sixth_score : ℝ := 98

theorem sixth_score_achieves_target_mean :
  let all_scores := existing_scores ++ [sixth_score]
  (all_scores.sum / all_scores.length : ℝ) = target_mean := by sorry

end NUMINAMATH_CALUDE_sixth_score_achieves_target_mean_l433_43350


namespace NUMINAMATH_CALUDE_punch_bowl_capacity_l433_43399

/-- Proves that the total capacity of a punch bowl is 72 cups given the specified conditions -/
theorem punch_bowl_capacity 
  (lemonade : ℕ) 
  (cranberry : ℕ) 
  (h1 : lemonade * 5 = cranberry * 3) 
  (h2 : cranberry = lemonade + 18) : 
  lemonade + cranberry = 72 := by
  sorry

#check punch_bowl_capacity

end NUMINAMATH_CALUDE_punch_bowl_capacity_l433_43399


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_5_5_minus_5_3_l433_43347

theorem smallest_prime_factor_of_5_5_minus_5_3 :
  Nat.minFac (5^5 - 5^3) = 2 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_5_5_minus_5_3_l433_43347


namespace NUMINAMATH_CALUDE_min_pizzas_for_johns_van_l433_43379

/-- The minimum whole number of pizzas needed to recover the van's cost -/
def min_pizzas (van_cost : ℕ) (earnings_per_pizza : ℕ) (gas_cost : ℕ) : ℕ :=
  (van_cost + (earnings_per_pizza - gas_cost - 1)) / (earnings_per_pizza - gas_cost)

theorem min_pizzas_for_johns_van :
  min_pizzas 8000 15 4 = 728 := by sorry

end NUMINAMATH_CALUDE_min_pizzas_for_johns_van_l433_43379
