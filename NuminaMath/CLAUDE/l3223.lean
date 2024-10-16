import Mathlib

namespace NUMINAMATH_CALUDE_solution_exists_l3223_322337

theorem solution_exists : ∃ a : ℝ, (-6) * (a^2) = 3 * (4*a + 2) ∧ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l3223_322337


namespace NUMINAMATH_CALUDE_sum_of_roots_of_unity_l3223_322326

def is_root_of_unity (z : ℂ) : Prop := ∃ n : ℕ, n > 0 ∧ z^n = 1

theorem sum_of_roots_of_unity (x y z : ℂ) :
  is_root_of_unity x ∧ is_root_of_unity y ∧ is_root_of_unity z →
  (is_root_of_unity (x + y + z) ↔ (x + y = 0 ∨ y + z = 0 ∨ z + x = 0)) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_unity_l3223_322326


namespace NUMINAMATH_CALUDE_fraction_count_l3223_322334

-- Define a function to check if an expression is a fraction
def is_fraction (expr : String) : Bool :=
  match expr with
  | "1/x" => true
  | "x^2+5x" => false
  | "1/2x" => false
  | "a/(3-2a)" => true
  | "3.14/π" => false
  | _ => false

-- Define the list of expressions
def expressions : List String := ["1/x", "x^2+5x", "1/2x", "a/(3-2a)", "3.14/π"]

-- Theorem statement
theorem fraction_count : (expressions.filter is_fraction).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_count_l3223_322334


namespace NUMINAMATH_CALUDE_exists_divisible_by_33_l3223_322378

def original_number : ℕ := 975312468

def insert_digit (n : ℕ) (d : ℕ) (pos : ℕ) : ℕ :=
  let digits := n.digits 10
  let (before, after) := digits.splitAt pos
  ((before ++ [d] ++ after).foldl (fun acc x => acc * 10 + x) 0)

theorem exists_divisible_by_33 :
  ∃ (d : ℕ) (pos : ℕ), d < 10 ∧ pos ≤ 9 ∧ 
  (insert_digit original_number d pos) % 33 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_by_33_l3223_322378


namespace NUMINAMATH_CALUDE_insurance_coverage_percentage_l3223_322370

def mri_cost : ℝ := 1200
def doctor_rate : ℝ := 300
def doctor_time : ℝ := 0.5
def fee_for_seen : ℝ := 150
def tim_payment : ℝ := 300

def total_cost : ℝ := mri_cost + doctor_rate * doctor_time + fee_for_seen

def insurance_coverage : ℝ := total_cost - tim_payment

theorem insurance_coverage_percentage : 
  insurance_coverage / total_cost * 100 = 80 := by sorry

end NUMINAMATH_CALUDE_insurance_coverage_percentage_l3223_322370


namespace NUMINAMATH_CALUDE_parabolas_cyclic_quadrilateral_l3223_322305

/-- A parabola in the xy-plane --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop

/-- Two parabolas have perpendicular axes --/
def perpendicular_axes (p1 p2 : Parabola) : Prop := sorry

/-- Two parabolas intersect at four distinct points --/
def four_distinct_intersections (p1 p2 : Parabola) : Prop := sorry

/-- Four points in the plane form a cyclic quadrilateral --/
def cyclic_quadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

/-- The main theorem --/
theorem parabolas_cyclic_quadrilateral (p1 p2 : Parabola) :
  perpendicular_axes p1 p2 →
  four_distinct_intersections p1 p2 →
  ∃ q1 q2 q3 q4 : ℝ × ℝ,
    (p1.eq q1.1 q1.2 ∧ p2.eq q1.1 q1.2) ∧
    (p1.eq q2.1 q2.2 ∧ p2.eq q2.1 q2.2) ∧
    (p1.eq q3.1 q3.2 ∧ p2.eq q3.1 q3.2) ∧
    (p1.eq q4.1 q4.2 ∧ p2.eq q4.1 q4.2) ∧
    cyclic_quadrilateral q1 q2 q3 q4 :=
by sorry

end NUMINAMATH_CALUDE_parabolas_cyclic_quadrilateral_l3223_322305


namespace NUMINAMATH_CALUDE_work_efficiency_ratio_l3223_322330

/-- Given a road that can be repaired by A in 4 days or by B in 5 days,
    the ratio of A's work efficiency to B's work efficiency is 5/4. -/
theorem work_efficiency_ratio (road : ℝ) (days_A days_B : ℕ) 
  (h_A : road / days_A = road / 4)
  (h_B : road / days_B = road / 5) :
  (road / days_A) / (road / days_B) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_work_efficiency_ratio_l3223_322330


namespace NUMINAMATH_CALUDE_probability_same_activity_l3223_322382

/-- The probability that two specific students participate in the same activity
    when four students are divided into two groups. -/
theorem probability_same_activity (n : ℕ) (m : ℕ) : 
  n = 4 → m = 2 → (m : ℚ) / (Nat.choose n 2) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_same_activity_l3223_322382


namespace NUMINAMATH_CALUDE_circle_problem_l3223_322374

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 = c.radius^2

def tangent_to_line (c : Circle) (a b d : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x + b * y = d ∧ (c.center.1 - x)^2 + (c.center.2 - y)^2 = c.radius^2

def center_on_line (c : Circle) (m b : ℝ) : Prop :=
  c.center.2 = m * c.center.1 + b

-- Define the theorem
theorem circle_problem :
  ∃ (c : Circle),
    passes_through c (2, -1) ∧
    tangent_to_line c 1 1 1 ∧
    center_on_line c (-2) 0 ∧
    c.center = (1, -2) ∧
    c.radius^2 = 2 ∧
    (∀ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = 2 ↔ passes_through c (x, y)) ∧
    (let chord_length := 2 * Real.sqrt (c.radius^2 - (3 * c.center.1 + 4 * c.center.2)^2 / 25);
     chord_length = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_problem_l3223_322374


namespace NUMINAMATH_CALUDE_sum_coefficients_expansion_l3223_322340

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the sum of coefficients function
def sumCoefficients (x : ℕ) : ℕ :=
  (C x 1 + C (x+1) 1 + C (x+2) 1 + C (x+3) 1) ^ 2

-- Theorem statement
theorem sum_coefficients_expansion :
  ∃ x : ℕ, sumCoefficients x = 225 :=
sorry

end NUMINAMATH_CALUDE_sum_coefficients_expansion_l3223_322340


namespace NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l3223_322307

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the specific bridge length problem -/
theorem specific_bridge_length :
  bridge_length 140 45 30 = 235 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l3223_322307


namespace NUMINAMATH_CALUDE_q_value_proof_l3223_322343

theorem q_value_proof (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1/p + 1/q = 3/2) 
  (h4 : p * q = 12) : 
  q = 9 + 3 * Real.sqrt 23 := by
sorry

end NUMINAMATH_CALUDE_q_value_proof_l3223_322343


namespace NUMINAMATH_CALUDE_complex_equation_l3223_322393

theorem complex_equation (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l3223_322393


namespace NUMINAMATH_CALUDE_fibonacci_ratio_property_fibonacci_ratio_periodic_fibonacci_ratio_distinct_in_period_l3223_322310

/-- Fibonacci sequence -/
def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

/-- Ratio of consecutive Fibonacci numbers -/
def fibonacciRatio (n : ℕ) : ℚ :=
  if n = 0 then 0 else (fibonacci (n + 1) : ℚ) / (fibonacci n : ℚ)

theorem fibonacci_ratio_property (n : ℕ) (h : n > 1) :
  fibonacciRatio n = 1 + 1 / (fibonacciRatio (n - 1)) :=
sorry

theorem fibonacci_ratio_periodic :
  ∃ (p : ℕ) (h : p > 0), ∀ (n : ℕ), fibonacciRatio (n + p) = fibonacciRatio n :=
sorry

theorem fibonacci_ratio_distinct_in_period (p : ℕ) (h : p > 0) :
  ∀ (i j : ℕ), i < j → j < p → fibonacciRatio i ≠ fibonacciRatio j :=
sorry

end NUMINAMATH_CALUDE_fibonacci_ratio_property_fibonacci_ratio_periodic_fibonacci_ratio_distinct_in_period_l3223_322310


namespace NUMINAMATH_CALUDE_polynomial_zero_at_sqrt_three_halves_l3223_322341

def q (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ x y : ℝ) : ℝ :=
  b₀ + b₁*x + b₂*y + b₃*x^2 + b₄*x*y + b₅*y^2 + b₆*x^3 + b₇*x^2*y + b₈*x*y^2 + b₉*y^3

theorem polynomial_zero_at_sqrt_three_halves 
  (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) 
  (h₁ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 0 = 0)
  (h₂ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 0 = 0)
  (h₃ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) 0 = 0)
  (h₄ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 1 = 0)
  (h₅ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 (-1) = 0)
  (h₆ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 1 = 0)
  (h₇ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 (-1) = 0)
  (h₈ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) 1 = 0) :
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (Real.sqrt (3/2)) (Real.sqrt (3/2)) = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_zero_at_sqrt_three_halves_l3223_322341


namespace NUMINAMATH_CALUDE_prime_quadratic_roots_l3223_322321

theorem prime_quadratic_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 222*p = 0 ∧ y^2 + p*y - 222*p = 0) → 
  31 < p ∧ p ≤ 41 := by
sorry

end NUMINAMATH_CALUDE_prime_quadratic_roots_l3223_322321


namespace NUMINAMATH_CALUDE_vector_properties_l3223_322361

def a : ℝ × ℝ := (1, 2)
def b (t : ℝ) : ℝ × ℝ := (-4, t)

theorem vector_properties :
  (∀ t : ℝ, (∃ k : ℝ, a = k • b t) → t = -8) ∧
  (∃ t_min : ℝ, ∀ t : ℝ, ‖a - b t‖ ≥ ‖a - b t_min‖ ∧ ‖a - b t_min‖ = 5) ∧
  (∀ t : ℝ, ‖a + b t‖ = ‖a - b t‖ → t = 2) ∧
  (∀ t : ℝ, (a • b t < 0) → t < 2) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l3223_322361


namespace NUMINAMATH_CALUDE_best_fit_model_l3223_322300

/-- Represents a regression model with a correlation coefficient -/
structure RegressionModel where
  R : ℝ
  h_R_range : R ≥ 0 ∧ R ≤ 1

/-- Defines when one model has a better fit than another -/
def better_fit (m1 m2 : RegressionModel) : Prop := m1.R > m2.R

theorem best_fit_model (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.R = 0.98)
  (h2 : model2.R = 0.80)
  (h3 : model3.R = 0.50)
  (h4 : model4.R = 0.25) :
  better_fit model1 model2 ∧ better_fit model1 model3 ∧ better_fit model1 model4 := by
  sorry


end NUMINAMATH_CALUDE_best_fit_model_l3223_322300


namespace NUMINAMATH_CALUDE_bill_face_value_l3223_322379

/-- Proves that given a true discount of 360 and a banker's discount of 432, 
    the face value of the bill is 1800. -/
theorem bill_face_value (TD : ℕ) (BD : ℕ) (FV : ℕ) : 
  TD = 360 → BD = 432 → FV = (TD^2) / (BD - TD) → FV = 1800 := by
  sorry

#check bill_face_value

end NUMINAMATH_CALUDE_bill_face_value_l3223_322379


namespace NUMINAMATH_CALUDE_die_rolls_for_most_likely_32_twos_l3223_322371

/-- The number of rolls needed for the most likely number of twos to be 32 -/
theorem die_rolls_for_most_likely_32_twos :
  ∃ n : ℕ, 191 ≤ n ∧ n ≤ 197 ∧
  (∀ k : ℕ, (Nat.choose n k * (1/6)^k * (5/6)^(n-k)) ≤ (Nat.choose n 32 * (1/6)^32 * (5/6)^(n-32))) :=
by sorry

end NUMINAMATH_CALUDE_die_rolls_for_most_likely_32_twos_l3223_322371


namespace NUMINAMATH_CALUDE_lecture_series_arrangements_l3223_322308

theorem lecture_series_arrangements (n : ℕ) (pair_constraints : ℕ) : 
  n = 6 → pair_constraints = 2 → (n.factorial / 2^pair_constraints) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lecture_series_arrangements_l3223_322308


namespace NUMINAMATH_CALUDE_restaurant_group_size_l3223_322358

/-- Calculates the total number of people in a restaurant group given the following conditions:
  * The cost of an adult meal is $7
  * Kids eat for free
  * There are 9 kids in the group
  * The total cost for the group is $28
-/
theorem restaurant_group_size :
  let adult_meal_cost : ℕ := 7
  let kids_count : ℕ := 9
  let total_cost : ℕ := 28
  let adult_count : ℕ := total_cost / adult_meal_cost
  let total_people : ℕ := adult_count + kids_count
  total_people = 13 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_size_l3223_322358


namespace NUMINAMATH_CALUDE_one_third_of_number_l3223_322356

theorem one_third_of_number (x : ℝ) : 
  (1 / 3 : ℝ) * x = 130.00000000000003 → x = 390.0000000000001 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_number_l3223_322356


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3223_322312

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ), 
    a > 3 ∧ b > 3 ∧
    n = 2 * a + 2 ∧
    n = 3 * b + 3 ∧
    (∀ (m : ℕ) (c d : ℕ), c > 3 → d > 3 → m = 2 * c + 2 → m = 3 * d + 3 → m ≥ n) ∧
    n = 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3223_322312


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l3223_322380

theorem intersection_points_on_circle :
  ∀ (x y : ℝ), 
    ((x + 2*y = 19 ∨ y + 2*x = 98) ∧ y = 1/x) →
    (x - 34)^2 + (y - 215/4)^2 = 49785/16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l3223_322380


namespace NUMINAMATH_CALUDE_intersection_point_l3223_322339

/-- The point of intersection for the lines 8x - 5y = 10 and 6x + 2y = 20 -/
theorem intersection_point :
  ∃! p : ℚ × ℚ, 
    (8 * p.1 - 5 * p.2 = 10) ∧ 
    (6 * p.1 + 2 * p.2 = 20) ∧ 
    p = (60/23, 50/23) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3223_322339


namespace NUMINAMATH_CALUDE_six_students_five_lectures_l3223_322365

/-- The number of ways to assign students to lectures -/
def assignment_count (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: The number of ways to assign 6 students to 5 lectures is 5^6 -/
theorem six_students_five_lectures :
  assignment_count 6 5 = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_six_students_five_lectures_l3223_322365


namespace NUMINAMATH_CALUDE_simplify_expression_l3223_322349

theorem simplify_expression (x : ℝ) : 105 * x - 58 * x = 47 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3223_322349


namespace NUMINAMATH_CALUDE_toms_original_portion_l3223_322391

theorem toms_original_portion (tom uma vicky : ℝ) : 
  tom + uma + vicky = 2000 →
  (tom - 200) + 3 * uma + 3 * vicky = 3500 →
  tom = 1150 := by
sorry

end NUMINAMATH_CALUDE_toms_original_portion_l3223_322391


namespace NUMINAMATH_CALUDE_first_month_sale_l3223_322395

def average_sale : ℕ := 5500
def month2_sale : ℕ := 5927
def month3_sale : ℕ := 5855
def month4_sale : ℕ := 6230
def month5_sale : ℕ := 5562
def month6_sale : ℕ := 3991

theorem first_month_sale :
  let total_sale := 6 * average_sale
  let known_sales := month2_sale + month3_sale + month4_sale + month5_sale + month6_sale
  total_sale - known_sales = 5435 := by
sorry

end NUMINAMATH_CALUDE_first_month_sale_l3223_322395


namespace NUMINAMATH_CALUDE_inscribed_square_area_ratio_l3223_322320

/-- A circle with a square inscribed in it, where the square's vertices touch the circle
    and the side of the square intersects the circle such that each intersection segment
    equals twice the radius of the circle. -/
structure InscribedSquare where
  r : ℝ  -- radius of the circle
  s : ℝ  -- side length of the square
  h1 : s = r * Real.sqrt 2  -- relationship between side length and radius
  h2 : s * Real.sqrt 2 = 2 * r  -- diagonal of square equals diameter of circle

/-- The ratio of the area of the inscribed square to the area of the circle is 2/π. -/
theorem inscribed_square_area_ratio (square : InscribedSquare) :
  (square.s ^ 2) / (Real.pi * square.r ^ 2) = 2 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_ratio_l3223_322320


namespace NUMINAMATH_CALUDE_problem1_l3223_322354

theorem problem1 (x y : ℝ) : (-3 * x * y)^2 * (4 * x^2) = 36 * x^4 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l3223_322354


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3223_322315

theorem inequality_system_solution (x : ℝ) :
  (1 / x < 1 ∧ |4 * x - 1| > 2) ↔ (x < -1/4 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3223_322315


namespace NUMINAMATH_CALUDE_certain_value_proof_l3223_322303

theorem certain_value_proof (x y : ℕ) : 
  x + y = 50 → x = 30 → y = 20 → 2 * (x - y) = 20 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l3223_322303


namespace NUMINAMATH_CALUDE_oak_trees_planted_l3223_322392

/-- Given the initial and final number of oak trees in a park, 
    prove that the number of new trees planted is their difference -/
theorem oak_trees_planted (initial final : ℕ) (h : final ≥ initial) :
  final - initial = final - initial :=
by sorry

end NUMINAMATH_CALUDE_oak_trees_planted_l3223_322392


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_l3223_322384

/-- A geometric sequence with positive first term -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ∧ a 1 > 0

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_sequence_increasing_iff (a : ℕ → ℝ) :
  GeometricSequence a → (a 2 > a 1 ↔ IncreasingSequence a) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_l3223_322384


namespace NUMINAMATH_CALUDE_f_lower_bound_l3223_322368

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + 2 * Real.exp (-x) + (a - 2) * x

theorem f_lower_bound (a : ℝ) :
  (∀ x > 0, f a x ≥ (a + 2) * Real.cos x) → a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_f_lower_bound_l3223_322368


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l3223_322353

/-- 
For a two-digit number where:
- The number is 26
- The product of the number and the sum of its digits is 208
Prove that the difference between the unit's digit and the 10's digit is 4.
-/
theorem two_digit_number_difference (n : ℕ) (h1 : n = 26) 
  (h2 : n * (n / 10 + n % 10) = 208) : n % 10 - n / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l3223_322353


namespace NUMINAMATH_CALUDE_min_value_fraction_l3223_322360

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a + b = 2) :
  (∃ (x : ℝ), ∀ (y : ℝ), (3*a - b) / (a^2 + 2*a*b - 3*b^2) ≥ x) ∧
  (∃ (z : ℝ), (3*z - (2-z)) / (z^2 + 2*z*(2-z) - 3*(2-z)^2) = (3 + Real.sqrt 5) / 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3223_322360


namespace NUMINAMATH_CALUDE_elsa_final_marbles_l3223_322317

/-- Calculates the final number of marbles Elsa has at the end of the day. -/
def elsas_marbles (initial : ℕ) (lost_breakfast : ℕ) (given_to_susie : ℕ) (received_from_mom : ℕ) : ℕ :=
  initial - lost_breakfast - given_to_susie + received_from_mom + 2 * given_to_susie

/-- Theorem stating that Elsa ends up with 54 marbles given the conditions of the problem. -/
theorem elsa_final_marbles :
  elsas_marbles 40 3 5 12 = 54 :=
by sorry

end NUMINAMATH_CALUDE_elsa_final_marbles_l3223_322317


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3223_322372

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  (2 / (1 - i)).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3223_322372


namespace NUMINAMATH_CALUDE_linear_function_properties_l3223_322377

-- Define the linear function
def f (x : ℝ) : ℝ := -3 * x + 2

-- Define the original line before moving up
def g (x : ℝ) : ℝ := 2 * x - 4

-- Define the line after moving up by 5 units
def h (x : ℝ) : ℝ := g x + 5

theorem linear_function_properties :
  (∀ x y : ℝ, x < 0 ∧ y < 0 → f x ≠ y) ∧
  (∀ x : ℝ, h x = 2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l3223_322377


namespace NUMINAMATH_CALUDE_paper_width_covering_cube_l3223_322381

/-- Given a rectangular piece of paper covering a cube, prove the width of the paper. -/
theorem paper_width_covering_cube 
  (paper_length : ℝ) 
  (cube_volume : ℝ) 
  (h1 : paper_length = 48)
  (h2 : cube_volume = 8) : 
  ∃ (paper_width : ℝ), paper_width = 72 ∧ 
    paper_length * paper_width = 6 * (12 * (cube_volume ^ (1/3)))^2 :=
by sorry

end NUMINAMATH_CALUDE_paper_width_covering_cube_l3223_322381


namespace NUMINAMATH_CALUDE_unique_valid_denomination_l3223_322306

def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 120 → ∃ (a b c : ℕ), k = 7 * a + n * b + (n + 2) * c

def is_greatest_unformable (n : ℕ) : Prop :=
  ¬∃ (a b c : ℕ), 120 = 7 * a + n * b + (n + 2) * c

theorem unique_valid_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n ∧ is_greatest_unformable n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_denomination_l3223_322306


namespace NUMINAMATH_CALUDE_piecewise_function_proof_l3223_322346

theorem piecewise_function_proof :
  ∀ x : ℝ, 
    (|0| - |-x| + (-x + 2) = 
      if x < -1 then -1
      else if x ≤ 0 then 2
      else -2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_piecewise_function_proof_l3223_322346


namespace NUMINAMATH_CALUDE_gain_percentage_proof_l3223_322373

/-- Proves that the gain percentage is 20% when selling 20 articles for $60,
    given that selling 29.99999625000047 articles for $60 would result in a 20% loss. -/
theorem gain_percentage_proof (articles_sold : ℝ) (total_price : ℝ) (loss_articles : ℝ) 
  (h1 : articles_sold = 20)
  (h2 : total_price = 60)
  (h3 : loss_articles = 29.99999625000047)
  (h4 : (0.8 * (loss_articles * (total_price / articles_sold))) = total_price) :
  (((total_price / articles_sold) - (total_price / loss_articles)) / (total_price / loss_articles)) * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_gain_percentage_proof_l3223_322373


namespace NUMINAMATH_CALUDE_subtracted_number_l3223_322386

theorem subtracted_number (a b : ℕ) (x : ℚ) 
  (h1 : a / b = 6 / 5)
  (h2 : (a - x) / (b - x) = 5 / 4)
  (h3 : a - b = 5) :
  x = 5 := by sorry

end NUMINAMATH_CALUDE_subtracted_number_l3223_322386


namespace NUMINAMATH_CALUDE_regular_dinosaur_count_l3223_322366

theorem regular_dinosaur_count :
  ∀ (barney_weight : ℕ) (regular_dino_weight : ℕ) (total_weight : ℕ) (num_regular_dinos : ℕ),
    regular_dino_weight = 800 →
    barney_weight = regular_dino_weight * num_regular_dinos + 1500 →
    total_weight = barney_weight + regular_dino_weight * num_regular_dinos →
    total_weight = 9500 →
    num_regular_dinos = 5 := by
sorry

end NUMINAMATH_CALUDE_regular_dinosaur_count_l3223_322366


namespace NUMINAMATH_CALUDE_circle_circumference_when_equal_to_area_l3223_322314

/-- 
For a circle where the circumference and area are numerically equal,
if the diameter is 4, then the circumference is 4π.
-/
theorem circle_circumference_when_equal_to_area (d : ℝ) (C : ℝ) (A : ℝ) : 
  C = A →  -- Circumference equals area
  d = 4 →  -- Diameter is 4
  C = π * d →  -- Definition of circumference
  A = π * (d/2)^2 →  -- Definition of area
  C = 4 * π := by
sorry

end NUMINAMATH_CALUDE_circle_circumference_when_equal_to_area_l3223_322314


namespace NUMINAMATH_CALUDE_min_cubes_for_3x9x5_hollow_block_l3223_322355

/-- The minimum number of cubes needed to create a hollow block -/
def min_cubes_for_hollow_block (length width depth : ℕ) : ℕ :=
  length * width * depth - (length - 2) * (width - 2) * (depth - 2)

/-- Theorem stating that the minimum number of cubes for a 3x9x5 hollow block is 114 -/
theorem min_cubes_for_3x9x5_hollow_block :
  min_cubes_for_hollow_block 3 9 5 = 114 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_3x9x5_hollow_block_l3223_322355


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l3223_322375

theorem remainder_of_large_number (n : Nat) (d : Nat) (h : d = 180) :
  n = 1234567890123 → n % d = 123 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l3223_322375


namespace NUMINAMATH_CALUDE_smaller_number_proof_l3223_322309

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 44) (h2 : 5 * x = 6 * y) : min x y = 20 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l3223_322309


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3223_322331

/-- Given that the solution set of ax² - bx + c > 0 is (-1, 2), prove the following properties -/
theorem quadratic_inequality_properties 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, ax^2 - b*x + c > 0 ↔ -1 < x ∧ x < 2) : 
  (a + b + c = 0) ∧ (a < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3223_322331


namespace NUMINAMATH_CALUDE_turkey_roasting_time_l3223_322363

/-- Represents the turkey roasting problem --/
structure TurkeyRoasting where
  num_turkeys : ℕ
  weight_per_turkey : ℕ
  start_time : ℕ
  end_time : ℕ

/-- Calculates the roasting time per pound --/
def roasting_time_per_pound (tr : TurkeyRoasting) : ℚ :=
  let total_time := tr.end_time - tr.start_time
  let total_weight := tr.num_turkeys * tr.weight_per_turkey
  (total_time : ℚ) / total_weight

/-- The main theorem stating that the roasting time per pound is 15 minutes --/
theorem turkey_roasting_time (tr : TurkeyRoasting)
  (h1 : tr.num_turkeys = 2)
  (h2 : tr.weight_per_turkey = 16)
  (h3 : tr.start_time = 10 * 60)  -- 10:00 am in minutes
  (h4 : tr.end_time = 18 * 60)    -- 6:00 pm in minutes
  : roasting_time_per_pound tr = 15 := by
  sorry

#eval roasting_time_per_pound { num_turkeys := 2, weight_per_turkey := 16, start_time := 10 * 60, end_time := 18 * 60 }

end NUMINAMATH_CALUDE_turkey_roasting_time_l3223_322363


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l3223_322324

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem eighth_term_of_sequence (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 4 = 25 →
  arithmetic_sequence a₁ d 6 = 49 →
  arithmetic_sequence a₁ d 8 = 73 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l3223_322324


namespace NUMINAMATH_CALUDE_complex_number_problem_l3223_322329

theorem complex_number_problem (z : ℂ) :
  Complex.abs z = 5 ∧ (Complex.I * Complex.im ((3 + 4 * Complex.I) * z) = (3 + 4 * Complex.I) * z) →
  z = 4 + 3 * Complex.I ∨ z = -4 - 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3223_322329


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_reciprocal_squares_l3223_322350

theorem quadratic_roots_sum_of_reciprocal_squares :
  ∀ (r s : ℝ), 
    (2 * r^2 + 3 * r - 5 = 0) →
    (2 * s^2 + 3 * s - 5 = 0) →
    (r ≠ s) →
    (1 / r^2 + 1 / s^2 = 29 / 25) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_reciprocal_squares_l3223_322350


namespace NUMINAMATH_CALUDE_amc_distinct_scores_l3223_322388

/-- Represents the scoring system for an exam -/
structure ScoringSystem where
  totalQuestions : Nat
  correctPoints : Nat
  incorrectPoints : Nat
  unansweredPoints : Nat

/-- Calculates the number of distinct possible scores for a given scoring system -/
def distinctScores (s : ScoringSystem) : Nat :=
  sorry

/-- The AMC exam scoring system -/
def amcScoring : ScoringSystem :=
  { totalQuestions := 30
  , correctPoints := 5
  , incorrectPoints := 0
  , unansweredPoints := 2 }

/-- Theorem stating that the number of distinct possible scores for the AMC exam is 145 -/
theorem amc_distinct_scores : distinctScores amcScoring = 145 := by
  sorry

end NUMINAMATH_CALUDE_amc_distinct_scores_l3223_322388


namespace NUMINAMATH_CALUDE_sum_middle_m_value_l3223_322399

/-- An arithmetic sequence with 3m terms -/
structure ArithmeticSequence (m : ℕ) where
  sum_first_2m : ℝ
  sum_last_2m : ℝ

/-- The sum of the middle m terms in an arithmetic sequence -/
def sum_middle_m (seq : ArithmeticSequence m) : ℝ := sorry

theorem sum_middle_m_value {m : ℕ} (seq : ArithmeticSequence m)
  (h1 : seq.sum_first_2m = 100)
  (h2 : seq.sum_last_2m = 200) :
  sum_middle_m seq = 75 := by sorry

end NUMINAMATH_CALUDE_sum_middle_m_value_l3223_322399


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3223_322344

theorem smallest_solution_of_equation (x : ℝ) :
  x^4 - 40*x^2 + 400 = 0 → x ≥ -2*Real.sqrt 5 ∧ (∃ y, y^4 - 40*y^2 + 400 = 0 ∧ y = -2*Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3223_322344


namespace NUMINAMATH_CALUDE_new_crew_weight_l3223_322327

/-- The combined weight of two new crew members in a sailboat scenario -/
theorem new_crew_weight (n : ℕ) (avg_increase w1 w2 : ℝ) : 
  n = 12 → 
  avg_increase = 2.2 →
  w1 = 78 →
  w2 = 65 →
  (n : ℝ) * avg_increase + w1 + w2 = 169.4 :=
by sorry

end NUMINAMATH_CALUDE_new_crew_weight_l3223_322327


namespace NUMINAMATH_CALUDE_coffee_mix_solution_l3223_322316

/-- Represents the coffee mix problem -/
structure CoffeeMix where
  total_mix : ℝ
  columbian_price : ℝ
  brazilian_price : ℝ
  ethiopian_price : ℝ
  mix_price : ℝ
  ratio_columbian : ℝ
  ratio_brazilian : ℝ
  ratio_ethiopian : ℝ

/-- Theorem stating the correct amounts of each coffee type -/
theorem coffee_mix_solution (mix : CoffeeMix)
  (h_total : mix.total_mix = 150)
  (h_columbian_price : mix.columbian_price = 9.5)
  (h_brazilian_price : mix.brazilian_price = 4.25)
  (h_ethiopian_price : mix.ethiopian_price = 7.25)
  (h_mix_price : mix.mix_price = 6.7)
  (h_ratio : mix.ratio_columbian = 2 ∧ mix.ratio_brazilian = 3 ∧ mix.ratio_ethiopian = 5) :
  ∃ (columbian brazilian ethiopian : ℝ),
    columbian = 30 ∧
    brazilian = 45 ∧
    ethiopian = 75 ∧
    columbian + brazilian + ethiopian = mix.total_mix ∧
    columbian / mix.ratio_columbian = brazilian / mix.ratio_brazilian ∧
    columbian / mix.ratio_columbian = ethiopian / mix.ratio_ethiopian :=
by
  sorry


end NUMINAMATH_CALUDE_coffee_mix_solution_l3223_322316


namespace NUMINAMATH_CALUDE_perimeter_of_problem_pentagon_l3223_322332

/-- A pentagon ABCDE with given side lengths and a right angle -/
structure Pentagon :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DE : ℝ)
  (AE : ℝ)
  (right_angle_AED : AE^2 + DE^2 = AB^2 + BC^2 + DE^2)

/-- The perimeter of a pentagon -/
def perimeter (p : Pentagon) : ℝ :=
  p.AB + p.BC + p.CD + p.DE + p.AE

/-- The specific pentagon from the problem -/
def problem_pentagon : Pentagon :=
  { AB := 4
  , BC := 2
  , CD := 2
  , DE := 6
  , AE := 6
  , right_angle_AED := by sorry }

/-- Theorem: The perimeter of the problem pentagon is 14 + 6√2 -/
theorem perimeter_of_problem_pentagon :
  perimeter problem_pentagon = 14 + 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_problem_pentagon_l3223_322332


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l3223_322390

/-- The total number of handshakes -/
def total_handshakes : ℕ := 325

/-- The number of gymnasts -/
def n : ℕ := 26

/-- The number of handshakes between gymnasts -/
def gymnast_handshakes : ℕ := n * (n - 1) / 2

/-- The number of handshakes by the first coach -/
def coach1_handshakes : ℕ := 0

/-- The number of handshakes by the second coach -/
def coach2_handshakes : ℕ := total_handshakes - gymnast_handshakes - coach1_handshakes

theorem min_coach_handshakes :
  gymnast_handshakes + coach1_handshakes + coach2_handshakes = total_handshakes ∧
  coach1_handshakes = 0 ∧
  coach2_handshakes ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l3223_322390


namespace NUMINAMATH_CALUDE_power_quotient_square_l3223_322357

theorem power_quotient_square : (19^12 / 19^8)^2 = 130321 := by sorry

end NUMINAMATH_CALUDE_power_quotient_square_l3223_322357


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l3223_322311

/-- The line 2x - y = 0 passes through the center of the circle (x-a)² + (y-2a)² = 1 for all real a -/
theorem line_passes_through_circle_center (a : ℝ) : 2 * a - 2 * a = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l3223_322311


namespace NUMINAMATH_CALUDE_function_period_l3223_322335

/-- Given a constant a and a function f: ℝ → ℝ that satisfies
    f(x) = (f(x-a) - 1) / (f(x-a) + 1) for all x ∈ ℝ,
    prove that f has period 4a. -/
theorem function_period (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x, f x = (f (x - a) - 1) / (f (x - a) + 1)) :
  ∀ x, f (x + 4*a) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_period_l3223_322335


namespace NUMINAMATH_CALUDE_missing_digit_is_one_l3223_322338

/-- Converts a number from base 3 to base 10 -/
def base3ToBase10 (digit1 digit2 : ℕ) : ℕ :=
  digit1 * 3 + digit2

/-- Converts a number from base 12 to base 10 -/
def base12ToBase10 (digit1 digit2 : ℕ) : ℕ :=
  digit1 * 12 + digit2

/-- The main theorem stating that the missing digit is 1 -/
theorem missing_digit_is_one :
  ∃ (triangle : ℕ), 
    triangle < 10 ∧ 
    base3ToBase10 5 triangle = base12ToBase10 triangle 4 ∧ 
    triangle = 1 := by
  sorry

#check missing_digit_is_one

end NUMINAMATH_CALUDE_missing_digit_is_one_l3223_322338


namespace NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l3223_322385

theorem cos_squared_alpha_plus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + π / 4) ^ 2 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l3223_322385


namespace NUMINAMATH_CALUDE_intersection_line_l3223_322362

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line : 
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_l3223_322362


namespace NUMINAMATH_CALUDE_sara_marbles_count_l3223_322313

/-- The number of black marbles Sara has after receiving marbles from Fred -/
def saras_final_marbles (initial : ℝ) (received : ℝ) : ℝ :=
  initial + received

/-- Theorem: Sara's final number of marbles is 1025.0 -/
theorem sara_marbles_count :
  saras_final_marbles 792.0 233.0 = 1025.0 := by
  sorry

end NUMINAMATH_CALUDE_sara_marbles_count_l3223_322313


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l3223_322318

theorem cubic_sum_problem (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 2) 
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a * b * c = 1/6 ∧ a^4 + b^4 + c^4 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l3223_322318


namespace NUMINAMATH_CALUDE_white_spotted_mushrooms_count_l3223_322367

/-- The number of red mushrooms Bill gathered -/
def red_mushrooms : ℕ := 12

/-- The number of brown mushrooms Bill gathered -/
def brown_mushrooms : ℕ := 6

/-- The number of green mushrooms Ted gathered -/
def green_mushrooms : ℕ := 14

/-- The number of blue mushrooms Ted gathered -/
def blue_mushrooms : ℕ := 6

/-- The fraction of blue mushrooms with white spots -/
def blue_spotted_fraction : ℚ := 1/2

/-- The fraction of red mushrooms with white spots -/
def red_spotted_fraction : ℚ := 2/3

/-- The fraction of brown mushrooms with white spots -/
def brown_spotted_fraction : ℚ := 1

theorem white_spotted_mushrooms_count : 
  ⌊blue_spotted_fraction * blue_mushrooms⌋ + 
  ⌊red_spotted_fraction * red_mushrooms⌋ + 
  ⌊brown_spotted_fraction * brown_mushrooms⌋ = 17 := by
  sorry

end NUMINAMATH_CALUDE_white_spotted_mushrooms_count_l3223_322367


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l3223_322394

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l3223_322394


namespace NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l3223_322396

-- Define the cough rates and time
def georgia_cough_rate : ℕ := 5
def robert_cough_rate : ℕ := 2 * georgia_cough_rate
def time_minutes : ℕ := 20

-- Define the total coughs function
def total_coughs (georgia_rate : ℕ) (robert_rate : ℕ) (time : ℕ) : ℕ :=
  georgia_rate * time + robert_rate * time

-- Theorem statement
theorem total_coughs_after_20_minutes :
  total_coughs georgia_cough_rate robert_cough_rate time_minutes = 300 := by
  sorry


end NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l3223_322396


namespace NUMINAMATH_CALUDE_spongebob_burger_price_l3223_322302

/-- The price of a burger in Spongebob's shop -/
def burger_price : ℝ := 2

/-- The number of burgers sold -/
def burgers_sold : ℕ := 30

/-- The number of large fries sold -/
def fries_sold : ℕ := 12

/-- The price of each large fries -/
def fries_price : ℝ := 1.5

/-- The total earnings for the day -/
def total_earnings : ℝ := 78

theorem spongebob_burger_price :
  burger_price * burgers_sold + fries_price * fries_sold = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_spongebob_burger_price_l3223_322302


namespace NUMINAMATH_CALUDE_cookie_scaling_l3223_322398

/-- Given a recipe for cookies, calculate the required ingredients for a larger batch -/
theorem cookie_scaling (base_cookies : ℕ) (target_cookies : ℕ) 
  (base_flour : ℚ) (base_sugar : ℚ) 
  (target_flour : ℚ) (target_sugar : ℚ) : 
  base_cookies > 0 → 
  (target_flour = (target_cookies : ℚ) / base_cookies * base_flour) ∧ 
  (target_sugar = (target_cookies : ℚ) / base_cookies * base_sugar) →
  (base_cookies = 40 ∧ 
   base_flour = 3 ∧ 
   base_sugar = 1 ∧ 
   target_cookies = 200) →
  (target_flour = 15 ∧ target_sugar = 5) := by
  sorry

end NUMINAMATH_CALUDE_cookie_scaling_l3223_322398


namespace NUMINAMATH_CALUDE_cuboid_volume_l3223_322348

/-- Given a cuboid with face areas 3, 5, and 15 sharing a common vertex, its volume is 15 -/
theorem cuboid_volume (a b c : ℝ) 
  (h1 : a * b = 3) 
  (h2 : a * c = 5) 
  (h3 : b * c = 15) : 
  a * b * c = 15 := by
  sorry

#check cuboid_volume

end NUMINAMATH_CALUDE_cuboid_volume_l3223_322348


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3223_322301

theorem quadratic_equation_solution (x y z t : ℝ) :
  x^2 + y^2 + z^2 + t^2 = x*(y + z + t) → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3223_322301


namespace NUMINAMATH_CALUDE_second_wood_weight_l3223_322389

/-- Represents a square piece of wood -/
structure Wood where
  side : ℝ
  weight : ℝ

/-- The weight of a square piece of wood is proportional to its area -/
axiom weight_prop_area {w1 w2 : Wood} :
  w1.weight / w2.weight = (w1.side ^ 2) / (w2.side ^ 2)

/-- Given two pieces of wood with specific properties, prove the weight of the second piece -/
theorem second_wood_weight (w1 w2 : Wood)
  (h1 : w1.side = 4)
  (h2 : w1.weight = 16)
  (h3 : w2.side = 6) :
  w2.weight = 36 := by
  sorry

#check second_wood_weight

end NUMINAMATH_CALUDE_second_wood_weight_l3223_322389


namespace NUMINAMATH_CALUDE_total_books_count_l3223_322345

/-- The number of books Susan has -/
def susan_books : ℕ := 600

/-- The number of books Lidia has -/
def lidia_books : ℕ := 4 * susan_books

/-- The total number of books Susan and Lidia have -/
def total_books : ℕ := susan_books + lidia_books

theorem total_books_count : total_books = 3000 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l3223_322345


namespace NUMINAMATH_CALUDE_no_integer_solution_l3223_322304

theorem no_integer_solution (n : ℝ) (hn : n ≠ 0) :
  ¬ ∃ z : ℤ, n / (z : ℝ) = n / ((z : ℝ) + 1) + n / ((z : ℝ) + 25) :=
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3223_322304


namespace NUMINAMATH_CALUDE_shortest_side_of_triangle_l3223_322359

theorem shortest_side_of_triangle (a b c : ℕ) (area : ℕ) : 
  a = 21 →
  a + b + c = 48 →
  area * area = 24 * 3 * (24 - b) * (b - 3) →
  b ≤ c →
  b = 10 :=
sorry

end NUMINAMATH_CALUDE_shortest_side_of_triangle_l3223_322359


namespace NUMINAMATH_CALUDE_composite_equal_if_same_greatest_divisors_l3223_322319

/-- The set of greatest divisors of a natural number, excluding the number itself -/
def greatestDivisors (n : ℕ) : Set ℕ :=
  {d | d ∣ n ∧ d ≠ n ∧ ∀ k, k ∣ n ∧ k ≠ n → k ≤ d}

/-- Two natural numbers are composite if they are greater than 1 and not prime -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

theorem composite_equal_if_same_greatest_divisors (a b : ℕ) 
    (ha : isComposite a) (hb : isComposite b) 
    (h : greatestDivisors a = greatestDivisors b) : 
  a = b := by
  sorry

end NUMINAMATH_CALUDE_composite_equal_if_same_greatest_divisors_l3223_322319


namespace NUMINAMATH_CALUDE_problem_solution_l3223_322352

theorem problem_solution : 
  ∃ x : ℝ, (28 + x / 69) * 69 = 1980 ∧ x = 1952 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3223_322352


namespace NUMINAMATH_CALUDE_probability_not_special_number_l3223_322369

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a^b = n

def is_power_of_three_halves (n : ℕ) : Prop :=
  ∃ (k : ℕ), (3/2)^k = n

def count_special_numbers : ℕ := 20

theorem probability_not_special_number :
  (200 - count_special_numbers) / 200 = 9 / 10 := by
  sorry

#check probability_not_special_number

end NUMINAMATH_CALUDE_probability_not_special_number_l3223_322369


namespace NUMINAMATH_CALUDE_bob_profit_l3223_322322

/-- Calculates the profit from breeding and selling show dogs -/
def dogBreedingProfit (numDogs : ℕ) (dogCost : ℕ) (numPuppies : ℕ) (puppyPrice : ℕ) 
                      (foodVaccinationCost : ℕ) (advertisingCost : ℕ) : ℤ :=
  (numPuppies * puppyPrice : ℤ) - (numDogs * dogCost + foodVaccinationCost + advertisingCost)

theorem bob_profit : 
  dogBreedingProfit 2 250 6 350 500 150 = 950 := by
  sorry

end NUMINAMATH_CALUDE_bob_profit_l3223_322322


namespace NUMINAMATH_CALUDE_division_problem_l3223_322336

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3223_322336


namespace NUMINAMATH_CALUDE_divisor_problem_l3223_322351

theorem divisor_problem (range_start : Nat) (range_end : Nat) (divisible_count : Nat) : 
  range_start = 10 → 
  range_end = 1000000 → 
  divisible_count = 111110 → 
  ∃ (d : Nat), d = 9 ∧ 
    (∀ n : Nat, range_start ≤ n ∧ n ≤ range_end → 
      (n % d = 0 ↔ ∃ k : Nat, k ≤ divisible_count ∧ n = range_start + (k - 1) * d)) :=
by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3223_322351


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3223_322397

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ bridge_length : ℝ,
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length ∧
    bridge_length = 217.5 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3223_322397


namespace NUMINAMATH_CALUDE_mike_total_games_l3223_322387

/-- The total number of basketball games Mike attended over two years -/
def total_games (games_this_year games_last_year : ℕ) : ℕ :=
  games_this_year + games_last_year

/-- Theorem stating that Mike attended 54 games in total -/
theorem mike_total_games : 
  total_games 15 39 = 54 := by
  sorry

end NUMINAMATH_CALUDE_mike_total_games_l3223_322387


namespace NUMINAMATH_CALUDE_tom_profit_is_8798_l3223_322323

/-- Calculates the profit for Tom's dough ball project -/
def dough_ball_profit (
  flour_needed : ℕ)  -- Amount of flour needed in pounds
  (flour_bag_size : ℕ)  -- Size of each flour bag in pounds
  (flour_bag_cost : ℕ)  -- Cost of each flour bag in dollars
  (salt_needed : ℕ)  -- Amount of salt needed in pounds
  (salt_cost_per_pound : ℚ)  -- Cost of salt per pound in dollars
  (promotion_cost : ℕ)  -- Cost of promotion in dollars
  (tickets_sold : ℕ)  -- Number of tickets sold
  (ticket_price : ℕ)  -- Price of each ticket in dollars
  : ℤ :=
  let flour_bags := (flour_needed + flour_bag_size - 1) / flour_bag_size
  let flour_cost := flour_bags * flour_bag_cost
  let salt_cost := (salt_needed : ℚ) * salt_cost_per_pound
  let total_cost := flour_cost + salt_cost.ceil + promotion_cost
  let revenue := tickets_sold * ticket_price
  revenue - total_cost

/-- Theorem stating that Tom's profit is $8798 -/
theorem tom_profit_is_8798 :
  dough_ball_profit 500 50 20 10 (2/10) 1000 500 20 = 8798 := by
  sorry

end NUMINAMATH_CALUDE_tom_profit_is_8798_l3223_322323


namespace NUMINAMATH_CALUDE_sum_of_three_distinct_divisors_l3223_322342

theorem sum_of_three_distinct_divisors (n : ℕ+) :
  (∃ (d₁ d₂ d₃ : ℕ+), d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃ ∧
    d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧
    d₁ + d₂ + d₃ = n) ↔
  (∃ k : ℕ+, n = 6 * k) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_distinct_divisors_l3223_322342


namespace NUMINAMATH_CALUDE_expand_expression_l3223_322364

theorem expand_expression (a b c : ℝ) : (a + b - c) * (a - b - c) = a^2 - 2*a*c + c^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3223_322364


namespace NUMINAMATH_CALUDE_no_integers_between_sqrt_bounds_l3223_322325

theorem no_integers_between_sqrt_bounds (n : ℕ+) :
  ¬∃ (x y : ℕ+), (Real.sqrt n + Real.sqrt (n + 1) < Real.sqrt x + Real.sqrt y) ∧
                  (Real.sqrt x + Real.sqrt y < Real.sqrt (4 * n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_no_integers_between_sqrt_bounds_l3223_322325


namespace NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l3223_322347

theorem product_of_symmetric_complex_numbers :
  ∀ (z₁ z₂ : ℂ),
  (z₁.im = -z₂.im) →  -- Symmetry condition with respect to real axis
  (z₁.re = z₂.re) →   -- Symmetry condition with respect to real axis
  (z₁ = 2 - I) →      -- Given condition for z₁
  (z₁ * z₂ = 5) :=    -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l3223_322347


namespace NUMINAMATH_CALUDE_eva_second_semester_maths_score_l3223_322376

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science

theorem eva_second_semester_maths_score :
  ∀ (first second : SemesterScores),
    first.maths = second.maths + 10 →
    first.arts + 15 = second.arts →
    first.science + (first.science / 3) = second.science →
    second.arts = 90 →
    second.science = 90 →
    totalScore first + totalScore second = 485 →
    second.maths = 80 := by
  sorry

end NUMINAMATH_CALUDE_eva_second_semester_maths_score_l3223_322376


namespace NUMINAMATH_CALUDE_simplify_expression_l3223_322328

theorem simplify_expression (y : ℝ) :
  4 * y - 6 * y^2 + 8 - (3 + 5 * y - 9 * y^2) = 3 * y^2 - y + 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3223_322328


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3223_322383

theorem complex_fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (1 / y) / (1 / x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3223_322383


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3223_322333

theorem polynomial_factorization (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3223_322333
