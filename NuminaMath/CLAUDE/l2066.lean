import Mathlib

namespace NUMINAMATH_CALUDE_circle_op_not_commutative_l2066_206670

/-- Defines the "☉" operation for plane vectors -/
def circle_op (a b : ℝ × ℝ) : ℝ :=
  let (m, n) := a
  let (p, q) := b
  m * q - n * p

/-- Theorem stating that the "☉" operation is not commutative -/
theorem circle_op_not_commutative :
  ∃ (a b : ℝ × ℝ), circle_op a b ≠ circle_op b a :=
sorry

end NUMINAMATH_CALUDE_circle_op_not_commutative_l2066_206670


namespace NUMINAMATH_CALUDE_increasing_positive_function_inequality_l2066_206611

theorem increasing_positive_function_inequality 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y)
  (h_positive : ∀ x, f x > 0)
  (h_differentiable : Differentiable ℝ f) :
  3 * f (-2) > 2 * f (-3) := by
  sorry

end NUMINAMATH_CALUDE_increasing_positive_function_inequality_l2066_206611


namespace NUMINAMATH_CALUDE_rectangle_arrangement_exists_l2066_206663

theorem rectangle_arrangement_exists : ∃ (a b c d : ℕ+), 
  (a * b + c * d = 49) ∧ 
  ((2 * (a + b) = 4 * (c + d)) ∨ (2 * (c + d) = 4 * (a + b))) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_exists_l2066_206663


namespace NUMINAMATH_CALUDE_camdens_dogs_legs_l2066_206651

def number_of_dogs (name : String) : ℕ :=
  match name with
  | "Justin" => 14
  | "Rico" => 24
  | "Camden" => 18
  | _ => 0

theorem camdens_dogs_legs : 
  (∀ (name : String), number_of_dogs name ≥ 0) →
  number_of_dogs "Rico" = number_of_dogs "Justin" + 10 →
  number_of_dogs "Camden" = (3 * number_of_dogs "Rico") / 4 →
  number_of_dogs "Camden" * 4 = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_camdens_dogs_legs_l2066_206651


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2066_206664

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 = 1 →                     -- first term condition
  d ≠ 0 →                       -- non-zero common difference
  (a 2) ^ 2 = a 1 * a 5 →       -- geometric sequence condition
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2066_206664


namespace NUMINAMATH_CALUDE_series_general_term_l2066_206671

theorem series_general_term (n : ℕ) (a : ℕ → ℚ) :
  (∀ k, a k = 1 / (k^2 : ℚ)) →
  a n = 1 / (n^2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_series_general_term_l2066_206671


namespace NUMINAMATH_CALUDE_sum_of_ages_l2066_206666

/-- Proves that the sum of Henry and Jill's present ages is 43 years -/
theorem sum_of_ages (henry_age jill_age : ℕ) : 
  henry_age = 27 →
  jill_age = 16 →
  henry_age - 5 = 2 * (jill_age - 5) →
  henry_age + jill_age = 43 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2066_206666


namespace NUMINAMATH_CALUDE_sum_of_divisors_theorem_l2066_206676

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j * 5^k is 3780, then i + j + k = 8 -/
theorem sum_of_divisors_theorem (i j k : ℕ) : 
  sum_of_divisors (2^i * 3^j * 5^k) = 3780 → i + j + k = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_theorem_l2066_206676


namespace NUMINAMATH_CALUDE_blackboard_sum_l2066_206627

def Operation : Type := List ℕ → List ℕ → (List ℕ × List ℕ)

def performOperations (initialBoard : List ℕ) (n : ℕ) (op : Operation) : (List ℕ × List ℕ) :=
  sorry

theorem blackboard_sum (initialBoard : List ℕ) (finalBoard : List ℕ) (paperNumbers : List ℕ) :
  initialBoard = [1, 3, 5, 7, 9] →
  (∃ op : Operation, performOperations initialBoard 4 op = (finalBoard, paperNumbers)) →
  finalBoard.length = 1 →
  paperNumbers.length = 4 →
  paperNumbers.sum = 230 :=
  sorry

end NUMINAMATH_CALUDE_blackboard_sum_l2066_206627


namespace NUMINAMATH_CALUDE_monomial_properties_l2066_206692

-- Define the monomial structure
structure Monomial (α : Type*) [Ring α] where
  coeff : α
  vars : List (Nat × Nat)

-- Define the monomial -2x^2y
def monomial : Monomial ℤ :=
  { coeff := -2,
    vars := [(1, 2), (2, 1)] }  -- Representing x^2 and y^1

-- Theorem statement
theorem monomial_properties :
  (monomial.coeff = -2) ∧
  (List.sum (monomial.vars.map (λ (_, exp) => exp)) = 3) :=
by sorry

end NUMINAMATH_CALUDE_monomial_properties_l2066_206692


namespace NUMINAMATH_CALUDE_product_inequality_l2066_206688

theorem product_inequality (a b c d : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
  (hab : a + b = 2) (hcd : c + d = 2) : 
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2066_206688


namespace NUMINAMATH_CALUDE_chemistry_class_size_l2066_206615

theorem chemistry_class_size 
  (total_students : ℕ) 
  (chem_only : ℕ) 
  (bio_only : ℕ) 
  (both : ℕ) 
  (h1 : total_students = 70)
  (h2 : total_students = chem_only + bio_only + both)
  (h3 : chem_only + both = 2 * (bio_only + both))
  (h4 : both = 8) : 
  chem_only + both = 52 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_class_size_l2066_206615


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2066_206685

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : Prop :=
  k * x^2 + 2 * x + 1 = 0

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x ∧ quadratic_equation k y

-- Theorem statement
theorem quadratic_roots_condition (k : ℝ) :
  has_two_distinct_real_roots k ↔ k < 1 ∧ k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2066_206685


namespace NUMINAMATH_CALUDE_haydens_earnings_l2066_206693

/-- Represents Hayden's work day --/
structure WorkDay where
  totalHours : ℕ
  peakHours : ℕ
  totalRides : ℕ
  longDistanceRides : ℕ
  shortDistanceGallons : ℕ
  longDistanceGallons : ℕ
  maintenanceCost : ℕ
  tollCount : ℕ
  parkingExpense : ℕ
  positiveReviews : ℕ
  excellentReviews : ℕ

/-- Calculate Hayden's earnings for a given work day --/
def calculateEarnings (day : WorkDay) : ℚ :=
  sorry

/-- Theorem stating that Hayden's earnings for the given day equal $411.75 --/
theorem haydens_earnings : 
  let day : WorkDay := {
    totalHours := 12,
    peakHours := 3,
    totalRides := 6,
    longDistanceRides := 3,
    shortDistanceGallons := 10,
    longDistanceGallons := 20,
    maintenanceCost := 30,
    tollCount := 2,
    parkingExpense := 10,
    positiveReviews := 2,
    excellentReviews := 1
  }
  calculateEarnings day = 411.75 := by sorry

end NUMINAMATH_CALUDE_haydens_earnings_l2066_206693


namespace NUMINAMATH_CALUDE_smallest_representable_difference_l2066_206678

theorem smallest_representable_difference : ∃ (m n : ℕ+), 
  14 = 19^(n : ℕ) - 5^(m : ℕ) ∧ 
  ∀ (k : ℕ+) (m' n' : ℕ+), k < 14 → k ≠ 19^(n' : ℕ) - 5^(m' : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_representable_difference_l2066_206678


namespace NUMINAMATH_CALUDE_triangle_area_l2066_206622

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) : 
  (1/2) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2066_206622


namespace NUMINAMATH_CALUDE_juice_cost_calculation_l2066_206634

theorem juice_cost_calculation (orange_cost apple_cost total_bottles orange_bottles : ℕ) 
  (h1 : orange_cost = 70)
  (h2 : apple_cost = 60)
  (h3 : total_bottles = 70)
  (h4 : orange_bottles = 42) :
  orange_cost * orange_bottles + apple_cost * (total_bottles - orange_bottles) = 4620 := by
  sorry

#check juice_cost_calculation

end NUMINAMATH_CALUDE_juice_cost_calculation_l2066_206634


namespace NUMINAMATH_CALUDE_paint_fraction_proof_l2066_206695

def paint_problem (initial_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) : Prop :=
  let remaining_after_first_week : ℚ := initial_paint - (first_week_fraction * initial_paint)
  let used_second_week : ℚ := total_used - (first_week_fraction * initial_paint)
  (used_second_week / remaining_after_first_week) = 1 / 6

theorem paint_fraction_proof :
  paint_problem 360 (1 / 4) 135 := by
  sorry

end NUMINAMATH_CALUDE_paint_fraction_proof_l2066_206695


namespace NUMINAMATH_CALUDE_smallest_marble_count_l2066_206658

/-- Represents the number of marbles of each color -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles -/
def totalMarbles (m : MarbleCount) : ℕ :=
  m.red + m.white + m.blue + m.green + m.yellow

/-- Represents the probability of selecting a specific combination of marbles -/
def selectProbability (m : MarbleCount) (r w b g : ℕ) : ℚ :=
  (m.red.choose r * m.white.choose w * m.blue.choose b * m.green.choose g : ℚ) /
  (totalMarbles m).choose 5

/-- The conditions for the marble selection probabilities to be equal -/
def equalProbabilities (m : MarbleCount) : Prop :=
  selectProbability m 5 0 0 0 = selectProbability m 4 1 0 0 ∧
  selectProbability m 5 0 0 0 = selectProbability m 3 1 1 0 ∧
  selectProbability m 5 0 0 0 = selectProbability m 2 1 1 1 ∧
  selectProbability m 5 0 0 0 = selectProbability m 1 1 1 1

/-- The theorem stating the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : 
  ∃ (m : MarbleCount), 
    m.yellow = 4 ∧ 
    equalProbabilities m ∧
    totalMarbles m = 27 ∧
    (∀ (m' : MarbleCount), m'.yellow = 4 → equalProbabilities m' → totalMarbles m' ≥ 27) :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l2066_206658


namespace NUMINAMATH_CALUDE_biology_enrollment_percentage_l2066_206684

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) 
  (h1 : total_students = 880) (h2 : not_enrolled = 616) :
  (((total_students - not_enrolled : ℝ) / total_students) * 100 : ℝ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_biology_enrollment_percentage_l2066_206684


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2066_206631

/-- Simple interest calculation -/
theorem simple_interest_principal (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 4.8)
  (h2 : T = 12)
  (h3 : R = 0.05)
  (h4 : SI = P * R * T) :
  P = 8 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2066_206631


namespace NUMINAMATH_CALUDE_B_2_2_equals_9_l2066_206614

def B : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => B m 2
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_2_equals_9 : B 2 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_B_2_2_equals_9_l2066_206614


namespace NUMINAMATH_CALUDE_floor_division_equality_l2066_206668

theorem floor_division_equality (α : ℝ) (d : ℕ) (h_α : α > 0) :
  ⌊α / d⌋ = ⌊⌊α⌋ / d⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_division_equality_l2066_206668


namespace NUMINAMATH_CALUDE_arithmetic_mean_product_l2066_206626

theorem arithmetic_mean_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 ∧ 
  a = 14 ∧ 
  b = 25 ∧ 
  c + 3 = d → 
  c * d = 418 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_product_l2066_206626


namespace NUMINAMATH_CALUDE_all_turbans_zero_price_l2066_206641

/-- Represents a servant's employment details -/
structure Servant where
  fullYearSalary : ℚ
  monthsWorked : ℚ
  actualPayment : ℚ

/-- Calculates the price of a turban given a servant's details -/
def turbanPrice (s : Servant) : ℚ :=
  s.actualPayment - (s.monthsWorked / 12) * s.fullYearSalary

/-- The main theorem proving that all turbans have zero price -/
theorem all_turbans_zero_price (servantA servantB servantC : Servant)
  (hA : servantA = { fullYearSalary := 120, monthsWorked := 8, actualPayment := 80 })
  (hB : servantB = { fullYearSalary := 150, monthsWorked := 7, actualPayment := 87.5 })
  (hC : servantC = { fullYearSalary := 180, monthsWorked := 10, actualPayment := 150 }) :
  turbanPrice servantA = 0 ∧ turbanPrice servantB = 0 ∧ turbanPrice servantC = 0 := by
  sorry


end NUMINAMATH_CALUDE_all_turbans_zero_price_l2066_206641


namespace NUMINAMATH_CALUDE_polygon_exterior_interior_angles_equal_l2066_206637

theorem polygon_exterior_interior_angles_equal (n : ℕ) : 
  (n ≥ 3) → (360 = (n - 2) * 180) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_interior_angles_equal_l2066_206637


namespace NUMINAMATH_CALUDE_age_sum_proof_l2066_206623

theorem age_sum_proof (patrick michael monica : ℕ) : 
  3 * michael = 5 * patrick →
  3 * monica = 5 * michael →
  monica - patrick = 80 →
  patrick + michael + monica = 245 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l2066_206623


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l2066_206667

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l2066_206667


namespace NUMINAMATH_CALUDE_shekar_average_marks_l2066_206642

def shekar_scores : List Nat := [76, 65, 82, 67, 55]

theorem shekar_average_marks :
  let total_marks := shekar_scores.sum
  let num_subjects := shekar_scores.length
  (total_marks / num_subjects : ℚ) = 69 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l2066_206642


namespace NUMINAMATH_CALUDE_marble_weight_problem_l2066_206617

theorem marble_weight_problem (piece1 piece2 total : ℝ) 
  (h1 : piece1 = 0.3333333333333333)
  (h2 : piece2 = 0.3333333333333333)
  (h3 : total = 0.75) :
  total - (piece1 + piece2) = 0.08333333333333337 := by
  sorry

end NUMINAMATH_CALUDE_marble_weight_problem_l2066_206617


namespace NUMINAMATH_CALUDE_similar_triangles_exist_l2066_206661

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorFunction : Point → Color := sorry

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define similarity ratio between two triangles
def similarityRatio (T1 T2 : Triangle) : ℝ := sorry

-- Define a predicate to check if all vertices of a triangle have the same color
def sameColor (T : Triangle) : Prop :=
  colorFunction T.A = colorFunction T.B ∧ colorFunction T.B = colorFunction T.C

-- The main theorem
theorem similar_triangles_exist :
  ∃ (T1 T2 : Triangle), similarityRatio T1 T2 = 1995 ∧ sameColor T1 ∧ sameColor T2 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_exist_l2066_206661


namespace NUMINAMATH_CALUDE_no_function_satisfies_composite_condition_l2066_206686

theorem no_function_satisfies_composite_condition :
  ∀ f : ℝ → ℝ, ∃ x : ℝ, f (f x) ≠ x^2 - 1996 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_composite_condition_l2066_206686


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_l2066_206674

theorem right_triangle_max_ratio (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_a : a = 3) :
  (∀ x y z, x^2 + y^2 = z^2 → x = 3 → (x^2 + y^2 + z^2) / z^2 ≤ 2) ∧
  (∃ x y z, x^2 + y^2 = z^2 ∧ x = 3 ∧ (x^2 + y^2 + z^2) / z^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_l2066_206674


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2066_206643

theorem complex_modulus_problem (z : ℂ) : (1 - Complex.I) * z = Complex.I → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2066_206643


namespace NUMINAMATH_CALUDE_set_of_positive_rationals_l2066_206687

def is_closed_under_addition_and_multiplication (S : Set ℚ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a * b) ∈ S

def has_trichotomy_property (S : Set ℚ) : Prop :=
  ∀ r : ℚ, (r ∈ S ∧ -r ∉ S ∧ r ≠ 0) ∨ (-r ∈ S ∧ r ∉ S ∧ r ≠ 0) ∨ (r = 0 ∧ r ∉ S ∧ -r ∉ S)

theorem set_of_positive_rationals (S : Set ℚ) 
  (h1 : is_closed_under_addition_and_multiplication S)
  (h2 : has_trichotomy_property S) :
  S = {r : ℚ | 0 < r} :=
sorry

end NUMINAMATH_CALUDE_set_of_positive_rationals_l2066_206687


namespace NUMINAMATH_CALUDE_a_work_days_l2066_206613

/-- The number of days B takes to finish the work alone -/
def b_days : ℝ := 8

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B works alone after A leaves -/
def b_alone_days : ℝ := 2

/-- The total amount of work to be done -/
def total_work : ℝ := 1

theorem a_work_days : ∃ (a : ℝ), 
  a > 0 ∧ 
  together_days * (1/a + 1/b_days) + b_alone_days * (1/b_days) = total_work ∧ 
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_work_days_l2066_206613


namespace NUMINAMATH_CALUDE_train_speed_l2066_206662

/-- The speed of a train given its length, time to cross a walking man, and the man's speed. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 500 →
  crossing_time = 29.997600191984642 →
  man_speed_kmh = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l2066_206662


namespace NUMINAMATH_CALUDE_fermat_little_theorem_l2066_206682

theorem fermat_little_theorem (p : ℕ) (a : ℤ) (h : Nat.Prime p) :
  ∃ k : ℤ, a^p - a = k * p :=
sorry

end NUMINAMATH_CALUDE_fermat_little_theorem_l2066_206682


namespace NUMINAMATH_CALUDE_trailing_zeros_remainder_l2066_206603

/-- Calculate the number of trailing zeros in the product of factorials from 1 to n -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The remainder when the number of trailing zeros in 1!2!3!...50! is divided by 500 -/
theorem trailing_zeros_remainder : trailingZeros 50 % 500 = 12 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_remainder_l2066_206603


namespace NUMINAMATH_CALUDE_max_abs_u_for_unit_circle_l2066_206609

theorem max_abs_u_for_unit_circle (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^4 - z^3 - 3*z^2*Complex.I - z + 1) ≤ 5 ∧
  Complex.abs ((-1 : ℂ)^4 - (-1 : ℂ)^3 - 3*(-1 : ℂ)^2*Complex.I - (-1 : ℂ) + 1) = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_u_for_unit_circle_l2066_206609


namespace NUMINAMATH_CALUDE_henry_collection_cost_l2066_206697

/-- The amount of money Henry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem stating that Henry needs $30 to finish his collection -/
theorem henry_collection_cost :
  let current := 3  -- Henry's current number of action figures
  let total := 8    -- Total number of action figures needed for a complete collection
  let cost := 6     -- Cost of each action figure in dollars
  money_needed current total cost = 30 := by
sorry

end NUMINAMATH_CALUDE_henry_collection_cost_l2066_206697


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l2066_206604

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l2066_206604


namespace NUMINAMATH_CALUDE_prob_two_nondefective_pens_l2066_206619

/-- Given a box of 8 pens with 2 defective pens, the probability of selecting 2 non-defective pens at random is 15/28. -/
theorem prob_two_nondefective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 8) 
  (h2 : defective_pens = 2) : 
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 15 / 28 := by
  sorry

#check prob_two_nondefective_pens

end NUMINAMATH_CALUDE_prob_two_nondefective_pens_l2066_206619


namespace NUMINAMATH_CALUDE_base7_product_sum_theorem_l2066_206650

/-- Converts a base-7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Sums the digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := sorry

theorem base7_product_sum_theorem :
  let a := 35
  let b := 21
  let product := multiplyBase7 a b
  let digitSum := sumDigitsBase7 product
  multiplyBase7 digitSum 3 = 63
  := by sorry

end NUMINAMATH_CALUDE_base7_product_sum_theorem_l2066_206650


namespace NUMINAMATH_CALUDE_correct_calculation_l2066_206669

theorem correct_calculation : 3 * Real.sqrt 2 - (Real.sqrt 2) / 2 = (5 / 2) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2066_206669


namespace NUMINAMATH_CALUDE_shelf_board_length_l2066_206605

/-- Proves that given a board of 143 cm, after cutting 25 cm and then 7 cm, 
    the length of other boards before the 7 cm cut was 125 cm. -/
theorem shelf_board_length 
  (initial_length : ℕ) 
  (first_cut : ℕ) 
  (final_adjustment : ℕ) 
  (h1 : initial_length = 143)
  (h2 : first_cut = 25)
  (h3 : final_adjustment = 7) :
  initial_length - first_cut - final_adjustment + final_adjustment = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_shelf_board_length_l2066_206605


namespace NUMINAMATH_CALUDE_triangle_area_l2066_206633

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos x)^2 + Real.sin x * Real.cos x

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  f (A / 2) = Real.sqrt 3 ∧
  a = 4 ∧
  b + c = 5 →
  (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2066_206633


namespace NUMINAMATH_CALUDE_part_one_part_two_l2066_206690

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x + a|
def g (x : ℝ) : ℝ := |x + 3| - x

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | f a x < g x}

-- Statement for part (1)
theorem part_one (a : ℝ) : (a - 3) ∈ M a → a ∈ Set.Ioo 0 3 := by sorry

-- Statement for part (2)
theorem part_two (a : ℝ) : Set.Icc (-1) 1 ⊆ M a → a ∈ Set.Ioo (-2) 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2066_206690


namespace NUMINAMATH_CALUDE_watermelon_sale_proof_l2066_206600

/-- Calculates the total money made from selling watermelons -/
def total_money_from_watermelons (weight : ℕ) (price_per_pound : ℕ) (num_watermelons : ℕ) : ℕ :=
  weight * price_per_pound * num_watermelons

/-- Proves that selling 18 watermelons weighing 23 pounds each at $2 per pound yields $828 -/
theorem watermelon_sale_proof :
  total_money_from_watermelons 23 2 18 = 828 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_sale_proof_l2066_206600


namespace NUMINAMATH_CALUDE_green_pieces_count_l2066_206649

theorem green_pieces_count (amber : ℕ) (clear : ℕ) (green : ℕ) :
  amber = 20 →
  clear = 85 →
  green = (25 : ℚ) / 100 * (amber + green + clear) →
  green = 35 := by
sorry

end NUMINAMATH_CALUDE_green_pieces_count_l2066_206649


namespace NUMINAMATH_CALUDE_thermodynamic_expansion_l2066_206610

/-- First law of thermodynamics --/
def first_law (Q Δu A : ℝ) : Prop := Q = Δu + A

/-- Ideal gas law --/
def ideal_gas_law (P V R T : ℝ) : Prop := P * V = R * T

theorem thermodynamic_expansion 
  (Q Δu A cᵥ T T₀ k x P S n R P₀ V₀ : ℝ) 
  (h_Q : Q = 0)
  (h_Δu : Δu = cᵥ * (T - T₀))
  (h_A : A = (k * x^2) / 2)
  (h_kx : k * x = P * S)
  (h_V : S * x = V₀ * (n - 1) / n)
  (h_first_law : first_law Q Δu A)
  (h_ideal_gas_initial : ideal_gas_law P₀ V₀ R T₀)
  (h_ideal_gas_final : ideal_gas_law P (n * V₀) R T)
  (h_positive : cᵥ > 0 ∧ n > 1 ∧ R > 0 ∧ T₀ > 0 ∧ P₀ > 0) :
  P = P₀ / (n * (1 + ((n - 1) * R) / (2 * n * cᵥ))) :=
sorry

end NUMINAMATH_CALUDE_thermodynamic_expansion_l2066_206610


namespace NUMINAMATH_CALUDE_divisibility_property_l2066_206628

theorem divisibility_property (a b c : ℕ) : 
  a ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) → 
  b ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) → 
  c ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) → 
  a ≠ b → b ≠ c → a ≠ c →
  ∃ k : ℤ, a * b * c + (7 - a) * (7 - b) * (7 - c) = 7 * k := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l2066_206628


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2066_206679

theorem consecutive_odd_numbers_sum (a b c d e : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1 ∧ 
            b = 2*k + 3 ∧ 
            c = 2*k + 5 ∧ 
            d = 2*k + 7 ∧ 
            e = 2*k + 9) →
  a + b + c + d + e = 130 →
  c = 26 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2066_206679


namespace NUMINAMATH_CALUDE_xyz_product_l2066_206624

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 100 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l2066_206624


namespace NUMINAMATH_CALUDE_income_calculation_l2066_206625

/-- Calculates a person's income given the income to expenditure ratio and savings amount. -/
def calculate_income (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : ℕ :=
  (income_ratio * savings) / (income_ratio - expenditure_ratio)

/-- Proves that given a person's income to expenditure ratio of 5:4 and savings of Rs. 3000, 
    the person's income is Rs. 15000. -/
theorem income_calculation (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) 
  (h1 : income_ratio = 5) 
  (h2 : expenditure_ratio = 4) 
  (h3 : savings = 3000) : 
  calculate_income income_ratio expenditure_ratio savings = 15000 := by
  sorry

end NUMINAMATH_CALUDE_income_calculation_l2066_206625


namespace NUMINAMATH_CALUDE_largest_angle_in_hexagon_l2066_206636

-- Define the hexagon and its angles
def Hexagon (A B C D E F : ℝ) : Prop :=
  -- Sum of angles in a hexagon is 720°
  A + B + C + D + E + F = 720 ∧
  -- Given conditions
  A = 90 ∧
  B = 120 ∧
  C = 95 ∧
  D = E ∧
  F = 2 * D + 25

-- Theorem statement
theorem largest_angle_in_hexagon (A B C D E F : ℝ) 
  (h : Hexagon A B C D E F) : 
  max A (max B (max C (max D (max E F)))) = 220 := by
  sorry


end NUMINAMATH_CALUDE_largest_angle_in_hexagon_l2066_206636


namespace NUMINAMATH_CALUDE_masters_sample_size_l2066_206672

/-- Calculates the sample size for a specific stratum in stratified sampling -/
def stratifiedSampleSize (totalRatio : ℕ) (stratumRatio : ℕ) (totalSample : ℕ) : ℕ :=
  (stratumRatio * totalSample) / totalRatio

/-- Proves that the sample size for master's students is 36 given the conditions -/
theorem masters_sample_size :
  let totalRatio : ℕ := 5 + 15 + 9 + 1
  let mastersRatio : ℕ := 9
  let totalSample : ℕ := 120
  stratifiedSampleSize totalRatio mastersRatio totalSample = 36 := by
  sorry

#eval stratifiedSampleSize 30 9 120

end NUMINAMATH_CALUDE_masters_sample_size_l2066_206672


namespace NUMINAMATH_CALUDE_truck_kinetic_energy_l2066_206601

/-- The initial kinetic energy of a truck with mass m, initial velocity v, and braking force F
    that stops after traveling a distance x, is equal to Fx. -/
theorem truck_kinetic_energy
  (m : ℝ) (v : ℝ) (F : ℝ) (x : ℝ) (t : ℝ)
  (h1 : m > 0)
  (h2 : v > 0)
  (h3 : F > 0)
  (h4 : x > 0)
  (h5 : t > 0)
  (h6 : F * x = (1/2) * m * v^2) :
  (1/2) * m * v^2 = F * x := by
sorry

end NUMINAMATH_CALUDE_truck_kinetic_energy_l2066_206601


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2066_206620

theorem contrapositive_equivalence :
  (∀ x y : ℝ, (x = 3 ∧ y = 5) → x + y = 8) ↔
  (∀ x y : ℝ, x + y ≠ 8 → (x ≠ 3 ∨ y ≠ 5)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2066_206620


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2066_206618

/-- Given a rectangle with width 10 m and area 150 square meters, 
    if its length is increased such that the new area is 1 (1/3) times the original area, 
    then the new perimeter of the rectangle is 60 meters. -/
theorem rectangle_perimeter (width : ℝ) (original_area : ℝ) (new_area : ℝ) : 
  width = 10 →
  original_area = 150 →
  new_area = original_area * (4/3) →
  let original_length := original_area / width
  let new_length := new_area / width
  2 * (new_length + width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2066_206618


namespace NUMINAMATH_CALUDE_half_times_two_thirds_times_three_fourths_l2066_206644

theorem half_times_two_thirds_times_three_fourths :
  (1 / 2 : ℚ) * (2 / 3 : ℚ) * (3 / 4 : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_half_times_two_thirds_times_three_fourths_l2066_206644


namespace NUMINAMATH_CALUDE_choose_seven_two_l2066_206648

theorem choose_seven_two : Nat.choose 7 2 = 21 := by sorry

end NUMINAMATH_CALUDE_choose_seven_two_l2066_206648


namespace NUMINAMATH_CALUDE_equal_utility_days_l2066_206655

/-- Utility function --/
def utility (math reading painting : ℝ) : ℝ := math^2 + reading * painting

/-- The problem statement --/
theorem equal_utility_days (t : ℝ) : 
  utility 4 t (12 - t) = utility 3 (t + 1) (11 - t) → t = 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_utility_days_l2066_206655


namespace NUMINAMATH_CALUDE_gcd_10010_20020_l2066_206654

theorem gcd_10010_20020 : Nat.gcd 10010 20020 = 10010 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10010_20020_l2066_206654


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2066_206638

theorem cubic_root_sum (u v w : ℝ) : 
  u^3 - 6*u^2 + 11*u - 6 = 0 →
  v^3 - 6*v^2 + 11*v - 6 = 0 →
  w^3 - 6*w^2 + 11*w - 6 = 0 →
  u * v / w + v * w / u + w * u / v = 49 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2066_206638


namespace NUMINAMATH_CALUDE_power_calculation_l2066_206632

theorem power_calculation (n : ℝ) : 
  (3/5 : ℝ) * (14.500000000000002 : ℝ)^n = 126.15 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2066_206632


namespace NUMINAMATH_CALUDE_honey_production_l2066_206629

theorem honey_production (num_hives : ℕ) (jar_capacity : ℚ) (jars_for_half : ℕ) 
  (h1 : num_hives = 5)
  (h2 : jar_capacity = 1/2)
  (h3 : jars_for_half = 100) :
  (2 * jars_for_half : ℚ) * jar_capacity / num_hives = 20 := by
  sorry

end NUMINAMATH_CALUDE_honey_production_l2066_206629


namespace NUMINAMATH_CALUDE_average_of_first_25_odd_primes_l2066_206696

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def first_25_odd_primes : List ℕ := 
  [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]

theorem average_of_first_25_odd_primes : 
  (∀ p ∈ first_25_odd_primes, is_prime p ∧ is_odd p) → 
  (List.sum first_25_odd_primes).toFloat / 25 = 47.48 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_25_odd_primes_l2066_206696


namespace NUMINAMATH_CALUDE_project_completion_equation_l2066_206689

/-- Represents the number of days required for a person to complete the project alone -/
structure ProjectTime where
  person_a : ℝ
  person_b : ℝ

/-- Represents the work schedule for the project -/
structure WorkSchedule where
  solo_days : ℝ
  total_days : ℝ

/-- Theorem stating the equation for the total number of days required to complete the project -/
theorem project_completion_equation (pt : ProjectTime) (ws : WorkSchedule) :
  pt.person_a = 12 →
  pt.person_b = 8 →
  ws.solo_days = 3 →
  ws.total_days / pt.person_a + (ws.total_days - ws.solo_days) / pt.person_b = 1 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_equation_l2066_206689


namespace NUMINAMATH_CALUDE_no_perfect_square_9999_xxxx_l2066_206657

theorem no_perfect_square_9999_xxxx : 
  ¬ ∃ x : ℕ, 99990000 ≤ x ∧ x ≤ 99999999 ∧ ∃ y : ℕ, x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_9999_xxxx_l2066_206657


namespace NUMINAMATH_CALUDE_combined_average_age_l2066_206602

theorem combined_average_age (room_a_count room_b_count room_c_count : ℕ)
                             (room_a_avg room_b_avg room_c_avg : ℝ) :
  room_a_count = 8 →
  room_b_count = 5 →
  room_c_count = 7 →
  room_a_avg = 30 →
  room_b_avg = 35 →
  room_c_avg = 40 →
  let total_count := room_a_count + room_b_count + room_c_count
  let total_age := room_a_count * room_a_avg + room_b_count * room_b_avg + room_c_count * room_c_avg
  (total_age / total_count : ℝ) = 34.75 := by
sorry

end NUMINAMATH_CALUDE_combined_average_age_l2066_206602


namespace NUMINAMATH_CALUDE_closest_ratio_is_one_to_one_l2066_206645

def admission_fee_adult : ℕ := 30
def admission_fee_child : ℕ := 15
def total_collected : ℕ := 2250

def is_valid_combination (adults children : ℕ) : Prop :=
  adults ≥ 1 ∧ children ≥ 1 ∧
  adults * admission_fee_adult + children * admission_fee_child = total_collected

def ratio_difference_from_one (adults children : ℕ) : ℚ :=
  |((adults : ℚ) / (children : ℚ)) - 1|

theorem closest_ratio_is_one_to_one :
  ∃ (a c : ℕ), is_valid_combination a c ∧
    ∀ (x y : ℕ), is_valid_combination x y →
      ratio_difference_from_one a c ≤ ratio_difference_from_one x y :=
by sorry

end NUMINAMATH_CALUDE_closest_ratio_is_one_to_one_l2066_206645


namespace NUMINAMATH_CALUDE_certain_number_problem_l2066_206616

theorem certain_number_problem (x : ℝ) : ((2 * (x + 5)) / 5) - 5 = 22 → x = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2066_206616


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2066_206659

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2066_206659


namespace NUMINAMATH_CALUDE_circle_diameter_theorem_l2066_206652

/-- A circle with two intersecting perpendicular chords -/
structure CircleWithChords where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The lengths of the segments of the first chord -/
  chord1_seg1 : ℝ
  chord1_seg2 : ℝ
  /-- The lengths of the segments of the second chord -/
  chord2_seg1 : ℝ
  chord2_seg2 : ℝ
  /-- The chords are perpendicular -/
  chords_perpendicular : True
  /-- The product of segments of each chord equals the square of the radius -/
  chord1_property : chord1_seg1 * chord1_seg2 = radius ^ 2
  chord2_property : chord2_seg1 * chord2_seg2 = radius ^ 2

/-- The theorem to be proved -/
theorem circle_diameter_theorem (c : CircleWithChords) 
  (h1 : c.chord1_seg1 = 3 ∧ c.chord1_seg2 = 4) 
  (h2 : c.chord2_seg1 = 6 ∧ c.chord2_seg2 = 2) : 
  2 * c.radius = 4 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_theorem_l2066_206652


namespace NUMINAMATH_CALUDE_average_of_quadratic_solutions_l2066_206635

theorem average_of_quadratic_solutions (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + 3 * a * x + b
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (x₁ + x₂) / 2 = -3 / 2 := by
sorry

end NUMINAMATH_CALUDE_average_of_quadratic_solutions_l2066_206635


namespace NUMINAMATH_CALUDE_team_formation_count_l2066_206675

theorem team_formation_count (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  (Nat.choose (n - 1) (k - 1)) = 406 :=
by sorry

end NUMINAMATH_CALUDE_team_formation_count_l2066_206675


namespace NUMINAMATH_CALUDE_max_visible_blue_cubes_l2066_206646

/-- Represents a column of cubes with red and blue colors -/
structure CubeColumn :=
  (total : Nat)
  (blue : Nat)
  (red : Nat)
  (h_sum : blue + red = total)

/-- Represents a row of three columns on the board -/
structure BoardRow :=
  (left : CubeColumn)
  (middle : CubeColumn)
  (right : CubeColumn)

/-- The entire 3x3 board configuration -/
structure Board :=
  (front : BoardRow)
  (middle : BoardRow)
  (back : BoardRow)

/-- Calculates the maximum number of visible blue cubes in a row -/
def maxVisibleBlueInRow (row : BoardRow) : Nat :=
  row.left.blue + max 0 (row.middle.total - row.left.total) + max 0 (row.right.total - max row.left.total row.middle.total)

/-- The main theorem stating the maximum number of visible blue cubes -/
theorem max_visible_blue_cubes (board : Board) : 
  maxVisibleBlueInRow board.front + maxVisibleBlueInRow board.middle + maxVisibleBlueInRow board.back ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_visible_blue_cubes_l2066_206646


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_attainable_l2066_206660

theorem max_value_inequality (x y : ℝ) :
  (2 * x + 3 * y + 5) / Real.sqrt (2 * x^2 + 3 * y^2 + 7) ≤ Real.sqrt 38 :=
by sorry

theorem max_value_attainable :
  ∃ x y : ℝ, (2 * x + 3 * y + 5) / Real.sqrt (2 * x^2 + 3 * y^2 + 7) = Real.sqrt 38 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_attainable_l2066_206660


namespace NUMINAMATH_CALUDE_f_max_value_l2066_206640

def f (a b c : Real) : Real :=
  a * (1 - a + a * b) * (1 - a * b + a * b * c) * (1 - c)

theorem f_max_value (a b c : Real) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  f a b c ≤ 8/27 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l2066_206640


namespace NUMINAMATH_CALUDE_gcd_324_243_135_l2066_206653

theorem gcd_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by sorry

end NUMINAMATH_CALUDE_gcd_324_243_135_l2066_206653


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2066_206665

/-- Given a hyperbola with equation x²/121 - y²/81 = 1, 
    prove that the positive value n in its asymptote equations y = ±nx is 9/11 -/
theorem hyperbola_asymptote_slope (x y : ℝ) :
  (x^2 / 121 - y^2 / 81 = 1) →
  (∃ (n : ℝ), n > 0 ∧ (y = n*x ∨ y = -n*x) ∧ n = 9/11) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2066_206665


namespace NUMINAMATH_CALUDE_A_power_50_l2066_206699

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 2; -8, -5]

theorem A_power_50 : A ^ 50 = !![(-199 : ℤ), -100; 400, 201] := by sorry

end NUMINAMATH_CALUDE_A_power_50_l2066_206699


namespace NUMINAMATH_CALUDE_circles_intersect_circles_satisfy_intersection_condition_l2066_206606

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 8*p.2 - 8 = 0}

/-- Circle C₂ with equation x² + y² - 4x - 4y - 1 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 - 1 = 0}

/-- The center of circle C₁ -/
def center₁ : ℝ × ℝ := (-1, -4)

/-- The center of circle C₂ -/
def center₂ : ℝ × ℝ := (2, 2)

/-- The radius of circle C₁ -/
def radius₁ : ℝ := 5

/-- The radius of circle C₂ -/
def radius₂ : ℝ := 3

/-- Theorem stating that circles C₁ and C₂ intersect -/
theorem circles_intersect : ∃ p : ℝ × ℝ, p ∈ C₁ ∩ C₂ := by
  sorry

/-- Lemma stating the condition for intersecting circles -/
lemma intersecting_circles_condition (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) :
  let d := Real.sqrt ((c₂.1 - c₁.1)^2 + (c₂.2 - c₁.2)^2)
  abs (r₁ - r₂) < d ∧ d < r₁ + r₂ → ∃ p : ℝ × ℝ, p ∈ C₁ ∩ C₂ := by
  sorry

/-- Proof that C₁ and C₂ satisfy the intersecting circles condition -/
theorem circles_satisfy_intersection_condition :
  let d := Real.sqrt ((center₂.1 - center₁.1)^2 + (center₂.2 - center₁.2)^2)
  abs (radius₁ - radius₂) < d ∧ d < radius₁ + radius₂ := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_circles_satisfy_intersection_condition_l2066_206606


namespace NUMINAMATH_CALUDE_simplest_square_root_l2066_206694

theorem simplest_square_root :
  let options := [Real.sqrt 8, (Real.sqrt 2)⁻¹, Real.sqrt 2, Real.sqrt (1/2)]
  ∃ (x : ℝ), x ∈ options ∧ 
    (∀ y ∈ options, x ≠ y → (∃ z : ℝ, z ≠ 1 ∧ y = z * x ∨ y = x / z ∨ y = Real.sqrt (z * x^2))) ∧
    x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplest_square_root_l2066_206694


namespace NUMINAMATH_CALUDE_range_of_a_l2066_206680

theorem range_of_a (x y a : ℝ) : 
  x - y = 2 → 
  x + y = a → 
  x > -1 → 
  y < 0 → 
  -4 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2066_206680


namespace NUMINAMATH_CALUDE_abs_x_bound_inequality_x_y_l2066_206656

-- Part 1
theorem abs_x_bound (x y : ℝ) 
  (h1 : |x - 3*y| < 1/2) (h2 : |x + 2*y| < 1/6) : 
  |x| < 3/10 := by sorry

-- Part 2
theorem inequality_x_y (x y : ℝ) :
  x^4 + 16*y^4 ≥ 2*x^3*y + 8*x*y^3 := by sorry

end NUMINAMATH_CALUDE_abs_x_bound_inequality_x_y_l2066_206656


namespace NUMINAMATH_CALUDE_solution_part_I_solution_part_II_l2066_206612

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |x + 1|

-- Theorem for part I
theorem solution_part_I :
  ∀ x : ℝ, f 1 x < 3 ↔ -1 < x ∧ x < 1 :=
by sorry

-- Theorem for part II
theorem solution_part_II :
  (∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x ∧ f a x = 1) ↔ a = -4 ∨ a = 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_part_I_solution_part_II_l2066_206612


namespace NUMINAMATH_CALUDE_integers_between_cubes_l2066_206698

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(10.6 : ℝ)^3⌋ - ⌈(10.5 : ℝ)^3⌉ + 1) ∧ n = 33 := by
  sorry

end NUMINAMATH_CALUDE_integers_between_cubes_l2066_206698


namespace NUMINAMATH_CALUDE_monge_circle_theorem_monge_circle_tangent_point_l2066_206691

/-- The Monge circle theorem for an ellipse with semi-major axis a and semi-minor axis b --/
theorem monge_circle_theorem (a b : ℝ) (h : 0 < b ∧ b < a) :
  ∃ (r : ℝ), r^2 = a^2 + b^2 ∧ 
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 → 
    ∃ (t s : ℝ), (t * x + s * y = 0 ∧ t^2 + s^2 ≠ 0) → 
      x^2 + y^2 = r^2) :=
sorry

/-- The main theorem about the value of b --/
theorem monge_circle_tangent_point (b : ℝ) : 
  (∃ (x y : ℝ), x^2/3 + y^2 = 1) →  -- Ellipse exists
  (∃ (x y : ℝ), (x-3)^2 + (y-b)^2 = 9) →  -- Circle exists
  (∃! (x y : ℝ), x^2 + y^2 = 4 ∧ (x-3)^2 + (y-b)^2 = 9) →  -- Exactly one common point
  b = 4 :=
sorry

end NUMINAMATH_CALUDE_monge_circle_theorem_monge_circle_tangent_point_l2066_206691


namespace NUMINAMATH_CALUDE_sector_area_l2066_206621

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 3) (h2 : θ = 120 * π / 180) :
  (θ / (2 * π)) * π * r^2 = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2066_206621


namespace NUMINAMATH_CALUDE_quadratic_intercept_problem_l2066_206630

/-- Given two quadratic functions with specific y-intercepts and rational x-intercepts, prove that h = 1 -/
theorem quadratic_intercept_problem (j k : ℚ) : 
  (∃ x₁ x₂ : ℚ, 4 * (x₁ - 1)^2 + j = 0 ∧ 4 * (x₂ - 1)^2 + j = 0 ∧ x₁ ≠ x₂) →
  (∃ y₁ y₂ : ℚ, 3 * (y₁ - 1)^2 + k = 0 ∧ 3 * (y₂ - 1)^2 + k = 0 ∧ y₁ ≠ y₂) →
  4 * 1^2 + j = 2021 →
  3 * 1^2 + k = 2022 →
  (∀ h : ℚ, (∃ x₁ x₂ : ℚ, 4 * (x₁ - h)^2 + j = 0 ∧ 4 * (x₂ - h)^2 + j = 0 ∧ x₁ ≠ x₂) →
             (∃ y₁ y₂ : ℚ, 3 * (y₁ - h)^2 + k = 0 ∧ 3 * (y₂ - h)^2 + k = 0 ∧ y₁ ≠ y₂) →
             4 * h^2 + j = 2021 →
             3 * h^2 + k = 2022 →
             h = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intercept_problem_l2066_206630


namespace NUMINAMATH_CALUDE_coin_toss_sequence_count_l2066_206681

/-- Represents a coin toss sequence. -/
def CoinSequence := List Bool

/-- Counts the number of occurrences of a given subsequence in a coin sequence. -/
def countSubsequence (seq : CoinSequence) (subseq : List Bool) : Nat :=
  sorry

/-- Checks if a coin sequence satisfies the given conditions. -/
def isValidSequence (seq : CoinSequence) : Prop :=
  seq.length = 20 ∧
  countSubsequence seq [true, true] = 3 ∧
  countSubsequence seq [true, false] = 4 ∧
  countSubsequence seq [false, true] = 5 ∧
  countSubsequence seq [false, false] = 7

/-- The number of valid coin toss sequences. -/
def validSequenceCount : Nat :=
  sorry

theorem coin_toss_sequence_count :
  validSequenceCount = 11550 :=
sorry

end NUMINAMATH_CALUDE_coin_toss_sequence_count_l2066_206681


namespace NUMINAMATH_CALUDE_log_equality_l2066_206677

-- Define the logarithm base 2 (lg)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_equality : lg (5/2) + 2 * lg 2 + 2^(Real.log 3 / Real.log 4) = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l2066_206677


namespace NUMINAMATH_CALUDE_daughter_weight_l2066_206608

/-- Proves that the weight of the daughter is 48 kg given the conditions of the problem -/
theorem daughter_weight (M D C : ℝ) 
  (total_weight : M + D + C = 120)
  (daughter_child_weight : D + C = 60)
  (child_grandmother_ratio : C = (1/5) * M) :
  D = 48 := by sorry

end NUMINAMATH_CALUDE_daughter_weight_l2066_206608


namespace NUMINAMATH_CALUDE_unique_solution_l2066_206683

-- Define the machine's rule
def machineRule (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 5 * n + 2

-- Define a function that applies the rule n times
def applyNTimes (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => applyNTimes n (machineRule x)

-- Theorem statement
theorem unique_solution : ∀ n : ℕ, n > 0 → (applyNTimes 6 n = 4 ↔ n = 256) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2066_206683


namespace NUMINAMATH_CALUDE_cases_in_1990_l2066_206607

def linearDecrease (initial : ℕ) (final : ℕ) (totalYears : ℕ) (yearsPassed : ℕ) : ℕ :=
  initial - (initial - final) * yearsPassed / totalYears

theorem cases_in_1990 : 
  let initial := 600000
  let final := 2000
  let totalYears := 30
  let yearsPassed := 20
  linearDecrease initial final totalYears yearsPassed = 201333 :=
by sorry

end NUMINAMATH_CALUDE_cases_in_1990_l2066_206607


namespace NUMINAMATH_CALUDE_scrabble_middle_letter_value_l2066_206639

/-- Given a three-letter word in Scrabble with known conditions, 
    prove the value of the middle letter. -/
theorem scrabble_middle_letter_value 
  (first_letter_value : ℕ) 
  (third_letter_value : ℕ) 
  (total_score : ℕ) 
  (h1 : first_letter_value = 1)
  (h2 : third_letter_value = 1)
  (h3 : total_score = 30)
  (h4 : ∃ (middle_letter_value : ℕ), 
    3 * (first_letter_value + middle_letter_value + third_letter_value) = total_score) :
  ∃ (middle_letter_value : ℕ), middle_letter_value = 8 := by
  sorry

end NUMINAMATH_CALUDE_scrabble_middle_letter_value_l2066_206639


namespace NUMINAMATH_CALUDE_blackboard_remainder_l2066_206673

theorem blackboard_remainder (a : ℕ) : 
  a < 10 → (a + 100) % 7 = 5 → a = 5 := by sorry

end NUMINAMATH_CALUDE_blackboard_remainder_l2066_206673


namespace NUMINAMATH_CALUDE_subscription_cost_l2066_206647

theorem subscription_cost (reduction_percentage : ℝ) (reduction_amount : ℝ) (original_cost : ℝ) : 
  reduction_percentage = 0.30 →
  reduction_amount = 658 →
  reduction_percentage * original_cost = reduction_amount →
  original_cost = 2193 := by
  sorry

end NUMINAMATH_CALUDE_subscription_cost_l2066_206647
