import Mathlib

namespace NUMINAMATH_CALUDE_g_has_unique_zero_l1058_105890

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * (f a x + a + 1)

theorem g_has_unique_zero (a : ℝ) (h : a > 1 / Real.exp 1) :
  ∃! x, x > 0 ∧ g a x = 0 :=
sorry

end NUMINAMATH_CALUDE_g_has_unique_zero_l1058_105890


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1058_105843

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 - 4*y + m = 0 ∧ y = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1058_105843


namespace NUMINAMATH_CALUDE_rope_length_increase_l1058_105829

/-- Proves that increasing a circular area with initial radius 10 m by 942.8571428571429 m² results in a new radius of 20 m -/
theorem rope_length_increase (π : Real) (initial_radius : Real) (area_increase : Real) (new_radius : Real) : 
  π > 0 → 
  initial_radius = 10 → 
  area_increase = 942.8571428571429 → 
  new_radius = 20 → 
  π * new_radius^2 = π * initial_radius^2 + area_increase := by
  sorry

#check rope_length_increase

end NUMINAMATH_CALUDE_rope_length_increase_l1058_105829


namespace NUMINAMATH_CALUDE_smallest_k_divides_k_210_divides_smallest_k_is_210_l1058_105897

def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides : 
  ∀ k : ℕ, k > 0 → (∀ z : ℂ, f z = 0 → z^k = 1) → k ≥ 210 :=
by sorry

theorem k_210_divides : 
  ∀ z : ℂ, f z = 0 → z^210 = 1 :=
by sorry

theorem smallest_k_is_210 : 
  (∃ k : ℕ, k > 0 ∧ (∀ z : ℂ, f z = 0 → z^k = 1)) ∧
  (∀ k : ℕ, k > 0 → (∀ z : ℂ, f z = 0 → z^k = 1) → k ≥ 210) ∧
  (∀ z : ℂ, f z = 0 → z^210 = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_divides_k_210_divides_smallest_k_is_210_l1058_105897


namespace NUMINAMATH_CALUDE_senior_junior_ratio_l1058_105816

theorem senior_junior_ratio (j k : ℕ) (hj : j > 0) (hk : k > 0)
  (h_junior_contestants : (3 * j) / 5 = (j * 3) / 5)
  (h_senior_contestants : k / 5 = (k * 1) / 5)
  (h_equal_contestants : (3 * j) / 5 = k / 5) :
  k = 3 * j :=
sorry

end NUMINAMATH_CALUDE_senior_junior_ratio_l1058_105816


namespace NUMINAMATH_CALUDE_cutoff_portion_area_l1058_105856

/-- Regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  lateralSurfaceArea : ℝ

/-- Plane intersecting the pyramid -/
structure IntersectingPlane where
  triangleArea : ℝ

/-- The portion of the pyramid cut off by the plane -/
def cutOffPortion (pyramid : RegularQuadrilateralPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

theorem cutoff_portion_area 
  (pyramid : RegularQuadrilateralPyramid) 
  (plane : IntersectingPlane) 
  (h1 : pyramid.lateralSurfaceArea = 25) 
  (h2 : plane.triangleArea = 4) : 
  cutOffPortion pyramid plane = 20.25 := by
  sorry

end NUMINAMATH_CALUDE_cutoff_portion_area_l1058_105856


namespace NUMINAMATH_CALUDE_condition_relationship_l1058_105857

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a / b ≥ 1 → b * (b - a) ≤ 0) ∧
  (∃ a b, b * (b - a) ≤ 0 ∧ ¬(a / b ≥ 1)) :=
sorry

end NUMINAMATH_CALUDE_condition_relationship_l1058_105857


namespace NUMINAMATH_CALUDE_cos_sq_plus_two_sin_double_l1058_105870

theorem cos_sq_plus_two_sin_double (α : Real) (h : Real.tan α = 3 / 4) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_sq_plus_two_sin_double_l1058_105870


namespace NUMINAMATH_CALUDE_min_additional_bureaus_l1058_105825

theorem min_additional_bureaus (total_bureaus offices : ℕ) : 
  total_bureaus = 192 → offices = 36 → 
  (∃ (additional_bureaus : ℕ), 
    (total_bureaus + additional_bureaus) % offices = 0 ∧
    ∀ (x : ℕ), x < additional_bureaus → 
      (total_bureaus + x) % offices ≠ 0) →
  24 = (offices * ((total_bureaus + 23) / offices + 1)) - total_bureaus :=
by sorry

end NUMINAMATH_CALUDE_min_additional_bureaus_l1058_105825


namespace NUMINAMATH_CALUDE_unique_divisible_number_l1058_105889

def original_number : Nat := 20172018

theorem unique_divisible_number :
  ∃! n : Nat,
    (∃ a b : Nat, a < 10 ∧ b < 10 ∧ n = a * 1000000000 + original_number * 10 + b) ∧
    n % 8 = 0 ∧
    n % 9 = 0 :=
  by sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l1058_105889


namespace NUMINAMATH_CALUDE_square_sum_equals_twice_square_a_l1058_105855

theorem square_sum_equals_twice_square_a 
  (x y a θ : ℝ) 
  (h1 : x * Real.cos θ - y * Real.sin θ = a) 
  (h2 : (x - a * Real.sin θ)^2 + (y - a * Real.cos θ)^2 = a^2) : 
  x^2 + y^2 = 2 * a^2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_twice_square_a_l1058_105855


namespace NUMINAMATH_CALUDE_journey_distance_l1058_105821

/-- Given a journey that takes 3 hours and can be completed in half the time
    at a speed of 293.3333333333333 kmph, prove that the distance traveled is 440 km. -/
theorem journey_distance (original_time : ℝ) (new_speed : ℝ) (distance : ℝ) : 
  original_time = 3 →
  new_speed = 293.3333333333333 →
  distance = new_speed * (original_time / 2) →
  distance = 440 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l1058_105821


namespace NUMINAMATH_CALUDE_james_total_points_l1058_105808

/-- Quiz Bowl Scoring System -/
structure QuizBowl where
  correct_points : ℕ := 2
  incorrect_penalty : ℕ := 1
  quick_answer_bonus : ℕ := 1
  rounds : ℕ := 5
  questions_per_round : ℕ := 5

/-- James' Performance -/
structure Performance where
  correct_answers : ℕ
  missed_questions : ℕ
  quick_answers : ℕ

/-- Calculate total points for a given performance in the quiz bowl -/
def calculate_points (qb : QuizBowl) (perf : Performance) : ℕ :=
  qb.correct_points * perf.correct_answers + qb.quick_answer_bonus * perf.quick_answers

/-- Theorem: James' total points in the quiz bowl -/
theorem james_total_points (qb : QuizBowl) (james : Performance) 
  (h1 : james.correct_answers = qb.rounds * qb.questions_per_round - james.missed_questions)
  (h2 : james.missed_questions = 1)
  (h3 : james.quick_answers = 4) :
  calculate_points qb james = 52 := by
  sorry

end NUMINAMATH_CALUDE_james_total_points_l1058_105808


namespace NUMINAMATH_CALUDE_sticker_distribution_l1058_105883

theorem sticker_distribution (gold : ℕ) (students : ℕ) : 
  gold = 50 →
  students = 5 →
  (gold + 2 * gold + (2 * gold - 20)) / students = 46 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1058_105883


namespace NUMINAMATH_CALUDE_syrup_dilution_l1058_105813

theorem syrup_dilution (x : ℝ) : 
  (0 < x) ∧ 
  (x < 1000) ∧ 
  ((1000 - 2*x) * (1000 - x) = 120000) → 
  x = 400 :=
by sorry

end NUMINAMATH_CALUDE_syrup_dilution_l1058_105813


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l1058_105837

theorem solution_implies_a_value (x a : ℝ) : 
  x = 5 → 2 * x + 3 * a = 4 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l1058_105837


namespace NUMINAMATH_CALUDE_vector_b_calculation_l1058_105838

theorem vector_b_calculation (a b : ℝ × ℝ) : 
  a = (1, 2) → (2 • a + b = (3, 2)) → b = (1, -2) := by sorry

end NUMINAMATH_CALUDE_vector_b_calculation_l1058_105838


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1058_105878

-- Define the equation
def equation (x : ℝ) : Prop :=
  x / 3 + x / Real.sqrt (x^2 - 9) = 35 / 12

-- Define the set of solutions
def solution_set : Set ℝ :=
  {x | equation x ∧ x^2 > 9}

-- Theorem statement
theorem sum_of_solutions :
  ∃ (s : Finset ℝ), s.toSet = solution_set ∧ s.sum id = 35 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1058_105878


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1058_105841

theorem inequality_equivalence (x : ℝ) : (x - 2) / 3 ≤ x ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1058_105841


namespace NUMINAMATH_CALUDE_prime_count_200_to_220_l1058_105823

theorem prime_count_200_to_220 : ∃! p, Nat.Prime p ∧ 200 < p ∧ p < 220 := by
  sorry

end NUMINAMATH_CALUDE_prime_count_200_to_220_l1058_105823


namespace NUMINAMATH_CALUDE_square_reciprocal_sum_equality_l1058_105812

theorem square_reciprocal_sum_equality (n m k : ℕ+) : 
  (1 : ℚ) / n.val^2 + (1 : ℚ) / m.val^2 = (k : ℚ) / (n.val^2 + m.val^2) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_reciprocal_sum_equality_l1058_105812


namespace NUMINAMATH_CALUDE_custom_operation_theorem_l1058_105877

-- Define the custom operation *
def star (a b : ℝ) : ℝ := (a + b)^2

-- State the theorem
theorem custom_operation_theorem (x y : ℝ) : 
  star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by sorry

end NUMINAMATH_CALUDE_custom_operation_theorem_l1058_105877


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l1058_105868

/-- A pyramid with a square base and known face areas -/
structure Pyramid where
  base_area : ℝ
  face_area1 : ℝ
  face_area2 : ℝ

/-- The volume of the pyramid -/
def pyramid_volume (p : Pyramid) : ℝ := sorry

/-- Theorem: The volume of the specific pyramid is 784 -/
theorem specific_pyramid_volume :
  let p : Pyramid := { base_area := 196, face_area1 := 105, face_area2 := 91 }
  pyramid_volume p = 784 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l1058_105868


namespace NUMINAMATH_CALUDE_segment_length_l1058_105845

/-- Given a line segment AB with points P and Q, prove that AB has length 35 -/
theorem segment_length (A B P Q : ℝ × ℝ) : 
  (P.1 - A.1) / (B.1 - A.1) = 1 / 5 →  -- P divides AB in ratio 1:4
  (Q.1 - A.1) / (B.1 - A.1) = 2 / 7 →  -- Q divides AB in ratio 2:5
  abs (Q.1 - P.1) = 3 →                -- Distance between P and Q is 3
  abs (B.1 - A.1) = 35 := by            -- Length of AB is 35
sorry

end NUMINAMATH_CALUDE_segment_length_l1058_105845


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1058_105800

theorem arithmetic_mean_of_fractions : 
  let a := 9/12
  let b := 5/6
  let c := 7/8
  b = (a + c) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1058_105800


namespace NUMINAMATH_CALUDE_digit_150_is_6_l1058_105839

/-- The decimal representation of 17/270 as a sequence of digits -/
def decimalRepresentation : ℕ → Fin 10 := sorry

/-- The decimal representation of 17/270 is periodic with period 5 -/
axiom period_five : ∀ n : ℕ, decimalRepresentation n = decimalRepresentation (n + 5)

/-- The first period of the decimal representation -/
axiom first_period : 
  (decimalRepresentation 0 = 0) ∧
  (decimalRepresentation 1 = 6) ∧
  (decimalRepresentation 2 = 2) ∧
  (decimalRepresentation 3 = 9) ∧
  (decimalRepresentation 4 = 6)

/-- The 150th digit after the decimal point in 17/270 is 6 -/
theorem digit_150_is_6 : decimalRepresentation 149 = 6 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_6_l1058_105839


namespace NUMINAMATH_CALUDE_equal_roots_C_value_l1058_105847

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4 * eq.a * eq.c

/-- Checks if a quadratic equation has equal roots -/
def hasEqualRoots (eq : QuadraticEquation) : Prop :=
  discriminant eq = 0

/-- The specific quadratic equation from the problem -/
def problemEquation (k C : ℝ) : QuadraticEquation where
  a := 2 * k
  b := 6 * k
  c := C

/-- The theorem to be proved -/
theorem equal_roots_C_value :
  ∃ C : ℝ, hasEqualRoots (problemEquation 0.4444444444444444 C) ∧ C = 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_C_value_l1058_105847


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l1058_105879

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l1058_105879


namespace NUMINAMATH_CALUDE_tailor_time_calculation_l1058_105896

-- Define the time ratios
def shirt_time : ℚ := 1
def pants_time : ℚ := 2
def jacket_time : ℚ := 3

-- Define the reference quantities
def ref_shirts : ℕ := 2
def ref_pants : ℕ := 3
def ref_jackets : ℕ := 4
def ref_total_time : ℚ := 10

-- Define the quantities to calculate
def calc_shirts : ℕ := 14
def calc_pants : ℕ := 10
def calc_jackets : ℕ := 2

-- Theorem statement
theorem tailor_time_calculation :
  let base_time := ref_total_time / (ref_shirts * shirt_time + ref_pants * pants_time + ref_jackets * jacket_time)
  calc_shirts * (base_time * shirt_time) + calc_pants * (base_time * pants_time) + calc_jackets * (base_time * jacket_time) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tailor_time_calculation_l1058_105896


namespace NUMINAMATH_CALUDE_point_between_parallel_lines_l1058_105861

theorem point_between_parallel_lines (b : ℤ) : 
  (6 * 5 - 8 * b + 1 > 0 ∧ 3 * 5 - 4 * b + 5 < 0) → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_between_parallel_lines_l1058_105861


namespace NUMINAMATH_CALUDE_abs_ratio_equals_sqrt_eleven_sevenths_l1058_105811

theorem abs_ratio_equals_sqrt_eleven_sevenths (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + b^2 = 9*a*b) : 
  |((a+b)/(a-b))| = Real.sqrt (11/7) := by
sorry

end NUMINAMATH_CALUDE_abs_ratio_equals_sqrt_eleven_sevenths_l1058_105811


namespace NUMINAMATH_CALUDE_intersection_point_l1058_105815

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 5 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the line C2
def C2 (x y : ℝ) : Prop := y = x - 1

-- Theorem statement
theorem intersection_point : 
  ∃! (x y : ℝ), C1 x y ∧ C2 x y ∧ x = 2 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l1058_105815


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l1058_105849

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - 8*x + m = 0

-- Define an isosceles triangle
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

-- Define the triangle inequality
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

-- Main theorem
theorem isosceles_triangle_quadratic_roots (m : ℝ) : 
  (∃ x y : ℝ, 
    quadratic_equation x m ∧ 
    quadratic_equation y m ∧ 
    x ≠ y ∧
    is_isosceles_triangle 6 x y ∧
    satisfies_triangle_inequality 6 x y) ↔ 
  (m = 12 ∨ m = 16) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l1058_105849


namespace NUMINAMATH_CALUDE_quinn_free_donuts_l1058_105881

/-- The number of free donuts Quinn is eligible for based on his summer reading --/
def free_donuts (books_per_donut : ℕ) (books_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  (books_per_week * num_weeks) / books_per_donut

/-- Theorem stating that Quinn is eligible for 4 free donuts --/
theorem quinn_free_donuts :
  free_donuts 5 2 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quinn_free_donuts_l1058_105881


namespace NUMINAMATH_CALUDE_recurrence_relation_initial_conditions_sequence_satisfies_conditions_l1058_105853

/-- Sequence definition -/
def a (n : ℕ) : ℚ := 3 * 2^n + 3^n + (1/2) * n + 11/4

/-- Recurrence relation -/
theorem recurrence_relation (n : ℕ) (h : n ≥ 2) :
  a n = 5 * a (n-1) - 6 * a (n-2) + n + 2 := by sorry

/-- Initial conditions -/
theorem initial_conditions :
  a 0 = 27/4 ∧ a 1 = 49/4 := by sorry

/-- Main theorem: The sequence satisfies the recurrence relation and initial conditions -/
theorem sequence_satisfies_conditions :
  (∀ n : ℕ, n ≥ 2 → a n = 5 * a (n-1) - 6 * a (n-2) + n + 2) ∧
  a 0 = 27/4 ∧ a 1 = 49/4 := by sorry

end NUMINAMATH_CALUDE_recurrence_relation_initial_conditions_sequence_satisfies_conditions_l1058_105853


namespace NUMINAMATH_CALUDE_x_value_proof_l1058_105869

theorem x_value_proof (x y z a b d : ℝ) 
  (h1 : x * y / (x + y) = a) 
  (h2 : x * z / (x + z) = b) 
  (h3 : y * z / (y - z) = d) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hd : d ≠ 0) : 
  x = a * b / (a + b) := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l1058_105869


namespace NUMINAMATH_CALUDE_mental_competition_result_l1058_105884

/-- Represents the number of students who correctly answered each problem -/
structure ProblemCounts where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the scores for each problem -/
def problem_scores : Fin 3 → ℕ
  | 0 => 20  -- Problem a
  | 1 => 25  -- Problem b
  | 2 => 25  -- Problem c
  | _ => 0   -- This case should never occur due to Fin 3

theorem mental_competition_result 
  (counts : ProblemCounts)
  (h1 : counts.a + counts.b = 29)
  (h2 : counts.a + counts.c = 25)
  (h3 : counts.b + counts.c = 20)
  (h4 : counts.a + counts.b + counts.c ≥ 1 + 3 * 15 + 1)  -- At least one correct + 15 with two correct + one with all correct
  (h5 : counts.a + counts.b + counts.c - (3 + 2 * 15) ≥ 0)  -- Non-negative number of students with only one correct
  : 
  (counts.a + counts.b + counts.c - (3 + 2 * 15) = 4) ∧  -- 4 students answered only one question correctly
  (((counts.a * problem_scores 0) + (counts.b * problem_scores 1) + (counts.c * problem_scores 2) + 70) / (counts.a + counts.b + counts.c - (3 + 2 * 15) + 15 + 1) = 42) -- Average score is 42
  := by sorry


end NUMINAMATH_CALUDE_mental_competition_result_l1058_105884


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1058_105824

theorem sqrt_equation_solution (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 4/3) :
  ∀ x : ℝ, (Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x) ↔ 
  (x = (4 - p) / Real.sqrt (16 - 8*p)) := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1058_105824


namespace NUMINAMATH_CALUDE_yellow_highlighters_l1058_105833

theorem yellow_highlighters (total : ℕ) (pink : ℕ) (blue : ℕ) 
  (h1 : total = 15) 
  (h2 : pink = 3) 
  (h3 : blue = 5) : 
  total - pink - blue = 7 := by
  sorry

end NUMINAMATH_CALUDE_yellow_highlighters_l1058_105833


namespace NUMINAMATH_CALUDE_hansel_raise_percentage_l1058_105809

/-- Proves that Hansel's raise percentage is 10% given the problem conditions --/
theorem hansel_raise_percentage : 
  ∀ (hansel_initial gretel_initial hansel_final gretel_final gretel_raise hansel_raise : ℝ),
  hansel_initial = 30000 →
  gretel_initial = 30000 →
  gretel_raise = 0.15 →
  gretel_final = gretel_initial * (1 + gretel_raise) →
  hansel_final = gretel_final - 1500 →
  hansel_raise = (hansel_final - hansel_initial) / hansel_initial →
  hansel_raise = 0.1 := by
sorry


end NUMINAMATH_CALUDE_hansel_raise_percentage_l1058_105809


namespace NUMINAMATH_CALUDE_min_value_expressions_l1058_105891

theorem min_value_expressions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (max x (1/y) + max y (2/x) ≥ 2 * Real.sqrt 2) ∧
  (max x (1/y) + max y (2/z) + max z (3/x) ≥ 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expressions_l1058_105891


namespace NUMINAMATH_CALUDE_profit_calculation_l1058_105818

def trees : ℕ := 30
def planks_per_tree : ℕ := 25
def planks_per_table : ℕ := 15
def selling_price : ℕ := 300
def labor_cost : ℕ := 3000

theorem profit_calculation :
  let total_planks := trees * planks_per_tree
  let tables_made := total_planks / planks_per_table
  let revenue := tables_made * selling_price
  revenue - labor_cost = 12000 := by
sorry

end NUMINAMATH_CALUDE_profit_calculation_l1058_105818


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l1058_105892

/-- Calculate the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percentage for a radio with
    cost price 1500 and selling price 1290 is 14% -/
theorem radio_loss_percentage :
  loss_percentage 1500 1290 = 14 := by sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l1058_105892


namespace NUMINAMATH_CALUDE_percentage_relation_l1058_105851

theorem percentage_relation (x y z w : ℝ) 
  (h1 : x = 1.2 * y)
  (h2 : y = 0.4 * z)
  (h3 : z = 0.7 * w) :
  x = 0.336 * w := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l1058_105851


namespace NUMINAMATH_CALUDE_divisibility_condition_l1058_105862

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number (A B : ℕ) : ℕ := 3000000 + 100000 * B + 46200 + 10 * A + 7

theorem divisibility_condition (A B : ℕ) 
  (h1 : is_single_digit A) 
  (h2 : is_single_digit B) 
  (h3 : number A B % 9 = 0) : 
  A + B = 5 ∨ A + B = 14 := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1058_105862


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l1058_105874

/-- Calculate the gain percent on a scooter sale given the purchase price, repair costs, and selling price. -/
theorem scooter_gain_percent 
  (purchase_price : ℝ)
  (repair_costs : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 800)
  (h2 : repair_costs = 200)
  (h3 : selling_price = 1400) :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 40 := by
sorry


end NUMINAMATH_CALUDE_scooter_gain_percent_l1058_105874


namespace NUMINAMATH_CALUDE_proportionality_check_l1058_105886

/-- Represents a relationship between x and y --/
inductive Relationship
  | DirectProp
  | InverseProp
  | Neither

/-- Determines the relationship between x and y for a given equation --/
def determineRelationship (equation : ℝ → ℝ → Prop) : Relationship :=
  sorry

/-- Equation A: x^2 + y^2 = 16 --/
def equationA (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- Equation B: 2xy = 5 --/
def equationB (x y : ℝ) : Prop := 2*x*y = 5

/-- Equation C: x = 3y --/
def equationC (x y : ℝ) : Prop := x = 3*y

/-- Equation D: x^2 = 4y --/
def equationD (x y : ℝ) : Prop := x^2 = 4*y

/-- Equation E: 5x + 2y = 20 --/
def equationE (x y : ℝ) : Prop := 5*x + 2*y = 20

theorem proportionality_check :
  (determineRelationship equationA = Relationship.Neither) ∧
  (determineRelationship equationB = Relationship.InverseProp) ∧
  (determineRelationship equationC = Relationship.DirectProp) ∧
  (determineRelationship equationD = Relationship.Neither) ∧
  (determineRelationship equationE = Relationship.Neither) :=
sorry

end NUMINAMATH_CALUDE_proportionality_check_l1058_105886


namespace NUMINAMATH_CALUDE_g_of_3_l1058_105863

theorem g_of_3 (g : ℝ → ℝ) :
  (∀ x, g x = (x^2 + 1) / (4*x - 5)) →
  g 3 = 10/7 := by
sorry

end NUMINAMATH_CALUDE_g_of_3_l1058_105863


namespace NUMINAMATH_CALUDE_committee_selection_count_l1058_105803

def club_size : ℕ := 12
def women_count : ℕ := 7
def men_count : ℕ := 5
def committee_size : ℕ := 5
def min_women : ℕ := 2

theorem committee_selection_count :
  (Finset.sum (Finset.range (committee_size - min_women + 1))
    (fun k => Nat.choose women_count (k + min_women) * 
              Nat.choose (club_size - k - min_women) (committee_size - k - min_women))) = 2520 :=
sorry

end NUMINAMATH_CALUDE_committee_selection_count_l1058_105803


namespace NUMINAMATH_CALUDE_orange_weight_l1058_105831

theorem orange_weight (apple_weight : ℕ) (bag_capacity : ℕ) (num_bags : ℕ) (total_apple_weight : ℕ) :
  apple_weight = 4 →
  bag_capacity = 49 →
  num_bags = 3 →
  total_apple_weight = 84 →
  ∃ (orange_weight : ℕ),
    orange_weight * (total_apple_weight / apple_weight) = total_apple_weight ∧
    orange_weight = 4 :=
by
  sorry

#check orange_weight

end NUMINAMATH_CALUDE_orange_weight_l1058_105831


namespace NUMINAMATH_CALUDE_sum_of_eleventh_powers_l1058_105880

theorem sum_of_eleventh_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_eleventh_powers_l1058_105880


namespace NUMINAMATH_CALUDE_fourth_month_sale_l1058_105850

/-- Calculates the missing sale amount given the other sales and desired average -/
def calculate_missing_sale (sale1 sale2 sale3 sale5 desired_average : ℕ) : ℕ :=
  5 * desired_average - (sale1 + sale2 + sale3 + sale5)

/-- Theorem: Given the sales for 5 consecutive months, where 4 of the 5 sales are known,
    and the desired average sale, the sale in the fourth month must be 7720. -/
theorem fourth_month_sale 
  (sale1 : ℕ) (sale2 : ℕ) (sale3 : ℕ) (sale5 : ℕ) (desired_average : ℕ)
  (h1 : sale1 = 5420)
  (h2 : sale2 = 5660)
  (h3 : sale3 = 6200)
  (h4 : sale5 = 6500)
  (h5 : desired_average = 6300) :
  calculate_missing_sale sale1 sale2 sale3 sale5 desired_average = 7720 := by
  sorry

#eval calculate_missing_sale 5420 5660 6200 6500 6300

end NUMINAMATH_CALUDE_fourth_month_sale_l1058_105850


namespace NUMINAMATH_CALUDE_blake_change_l1058_105885

theorem blake_change (oranges apples mangoes initial : ℕ) 
  (h_oranges : oranges = 40)
  (h_apples : apples = 50)
  (h_mangoes : mangoes = 60)
  (h_initial : initial = 300) :
  initial - (oranges + apples + mangoes) = 150 := by
  sorry

end NUMINAMATH_CALUDE_blake_change_l1058_105885


namespace NUMINAMATH_CALUDE_carrie_tshirt_purchase_l1058_105864

def tshirt_cost : ℚ := 965 / 100  -- $9.65 represented as a rational number
def number_of_tshirts : ℕ := 12

def total_cost : ℚ := tshirt_cost * number_of_tshirts

theorem carrie_tshirt_purchase :
  total_cost = 11580 / 100 := by sorry

end NUMINAMATH_CALUDE_carrie_tshirt_purchase_l1058_105864


namespace NUMINAMATH_CALUDE_abs_zero_iff_eq_l1058_105805

theorem abs_zero_iff_eq (y : ℚ) : |5 * y - 7| = 0 ↔ y = 7 / 5 := by sorry

end NUMINAMATH_CALUDE_abs_zero_iff_eq_l1058_105805


namespace NUMINAMATH_CALUDE_slope_of_line_l1058_105867

theorem slope_of_line (x y : ℝ) :
  y = (Real.sqrt 3 / 3) * x - (Real.sqrt 7 / 3) →
  (y - (-(Real.sqrt 7 / 3))) / x = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l1058_105867


namespace NUMINAMATH_CALUDE_veranda_area_l1058_105810

/-- Given a rectangular room with length 17 m and width 12 m, surrounded by a veranda of width 2 m on all sides, the area of the veranda is 132 square meters. -/
theorem veranda_area (room_length : ℝ) (room_width : ℝ) (veranda_width : ℝ) :
  room_length = 17 →
  room_width = 12 →
  veranda_width = 2 →
  let total_length := room_length + 2 * veranda_width
  let total_width := room_width + 2 * veranda_width
  let total_area := total_length * total_width
  let room_area := room_length * room_width
  total_area - room_area = 132 :=
by sorry

end NUMINAMATH_CALUDE_veranda_area_l1058_105810


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1058_105846

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x - 48 = 0 → 
  (x = 16 ∨ x = -3) → 
  ∃ y : ℝ, y^2 - 13*y - 48 = 0 ∧ y ≠ x ∧ x ≤ y → x = 16 :=
by sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1058_105846


namespace NUMINAMATH_CALUDE_fathers_age_l1058_105854

theorem fathers_age (n m f : ℕ) (h1 : n * m = f / 7) (h2 : (n + 3) * (m + 3) = f + 3) : f = 21 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_l1058_105854


namespace NUMINAMATH_CALUDE_fraction_equality_l1058_105895

theorem fraction_equality : 48 / (7 + 3/4) = 192/31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1058_105895


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1058_105860

def M : Set ℕ := {3, 5, 6, 8}
def N : Set ℕ := {4, 5, 7, 8}

theorem intersection_of_M_and_N : M ∩ N = {5, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1058_105860


namespace NUMINAMATH_CALUDE_inscribed_square_properties_l1058_105820

theorem inscribed_square_properties (circle_area : ℝ) (h : circle_area = 324 * Real.pi) :
  let r : ℝ := Real.sqrt (circle_area / Real.pi)
  let d : ℝ := 2 * r
  let s : ℝ := d / Real.sqrt 2
  let square_area : ℝ := s ^ 2
  let total_diagonal_length : ℝ := 2 * d
  (square_area = 648) ∧ (total_diagonal_length = 72) := by
  sorry

#check inscribed_square_properties

end NUMINAMATH_CALUDE_inscribed_square_properties_l1058_105820


namespace NUMINAMATH_CALUDE_playground_boys_count_l1058_105871

/-- Given a playground with children, prove that the number of boys is 44 -/
theorem playground_boys_count (total_children : ℕ) (girls : ℕ) (boys : ℕ) : 
  total_children = 97 → girls = 53 → total_children = girls + boys → boys = 44 := by
sorry

end NUMINAMATH_CALUDE_playground_boys_count_l1058_105871


namespace NUMINAMATH_CALUDE_carla_book_count_l1058_105804

/-- The number of times Carla counted the books in a row on Tuesday -/
def book_count_tuesday (tiles : ℕ) (books : ℕ) (total_count : ℕ) : ℕ :=
  (total_count - 2 * tiles) / books

theorem carla_book_count 
  (tiles : ℕ) 
  (books : ℕ) 
  (total_count : ℕ) 
  (h1 : tiles = 38) 
  (h2 : books = 75) 
  (h3 : total_count = 301) : 
  book_count_tuesday tiles books total_count = 3 := by
sorry

end NUMINAMATH_CALUDE_carla_book_count_l1058_105804


namespace NUMINAMATH_CALUDE_product_of_sums_l1058_105866

theorem product_of_sums (x y : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h1 : x * y = 1 / 9)
  (h2 : x * (y + 1) = 7 / 9)
  (h3 : y * (x + 1) = 5 / 18) :
  (x + 1) * (y + 1) = 35 / 18 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_l1058_105866


namespace NUMINAMATH_CALUDE_other_players_score_l1058_105822

theorem other_players_score (total_score : ℕ) (faye_score : ℕ) (total_players : ℕ) :
  total_score = 68 →
  faye_score = 28 →
  total_players = 5 →
  ∃ (other_player_score : ℕ),
    other_player_score * (total_players - 1) = total_score - faye_score ∧
    other_player_score = 10 := by
  sorry

end NUMINAMATH_CALUDE_other_players_score_l1058_105822


namespace NUMINAMATH_CALUDE_sum_18_29_in_base3_l1058_105898

/-- Converts a natural number from base 10 to base 3 -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def fromBase3 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_18_29_in_base3 :
  toBase3 (18 + 29) = [1, 2, 0, 2] :=
sorry

end NUMINAMATH_CALUDE_sum_18_29_in_base3_l1058_105898


namespace NUMINAMATH_CALUDE_sarah_homework_problem_l1058_105888

theorem sarah_homework_problem (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  math_pages = 4 →
  reading_pages = 6 →
  problems_per_page = 4 →
  (math_pages + reading_pages) * problems_per_page = 40 :=
by sorry

end NUMINAMATH_CALUDE_sarah_homework_problem_l1058_105888


namespace NUMINAMATH_CALUDE_raft_travel_time_l1058_105865

/-- Given a distance between two docks, prove that if a motor ship travels this distance downstream
    in 5 hours and upstream in 6 hours, then a raft traveling at the speed of the current will take
    60 hours to cover the same distance downstream. -/
theorem raft_travel_time (s : ℝ) (h_s : s > 0) : 
  (∃ (v_s v_c : ℝ), v_s > v_c ∧ v_c > 0 ∧ s / (v_s + v_c) = 5 ∧ s / (v_s - v_c) = 6) →
  s / v_c = 60 :=
by sorry

end NUMINAMATH_CALUDE_raft_travel_time_l1058_105865


namespace NUMINAMATH_CALUDE_dress_designs_count_l1058_105802

/-- The number of different fabric colors available -/
def num_colors : ℕ := 5

/-- The number of different patterns available -/
def num_patterns : ℕ := 4

/-- The number of different sizes available -/
def num_sizes : ℕ := 3

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_patterns * num_sizes

/-- Theorem stating that the total number of possible dress designs is 60 -/
theorem dress_designs_count : total_designs = 60 := by
  sorry

end NUMINAMATH_CALUDE_dress_designs_count_l1058_105802


namespace NUMINAMATH_CALUDE_ae_length_is_fifteen_l1058_105858

/-- Represents a rectangle ABCD with a line EF dividing it into two equal areas -/
structure DividedRectangle where
  AB : ℝ
  AD : ℝ
  EB : ℝ
  EF : ℝ
  AE : ℝ
  area_AEFCD : ℝ
  area_EBCF : ℝ
  equal_areas : area_AEFCD = area_EBCF
  rectangle_area : AB * AD = area_AEFCD + area_EBCF

/-- The theorem stating that under given conditions, AE = 15 -/
theorem ae_length_is_fifteen (r : DividedRectangle)
  (h1 : r.EB = 40)
  (h2 : r.AD = 80)
  (h3 : r.EF = 30) :
  r.AE = 15 := by
  sorry

#check ae_length_is_fifteen

end NUMINAMATH_CALUDE_ae_length_is_fifteen_l1058_105858


namespace NUMINAMATH_CALUDE_pitcher_distribution_percentage_l1058_105801

/-- Represents the contents of a pitcher -/
structure Pitcher :=
  (capacity : ℝ)
  (orange_juice : ℝ)
  (apple_juice : ℝ)

/-- Represents the distribution of the pitcher contents into cups -/
structure Distribution :=
  (pitcher : Pitcher)
  (num_cups : ℕ)

/-- The theorem stating the percentage of the pitcher's capacity in each cup -/
theorem pitcher_distribution_percentage (d : Distribution) 
  (h1 : d.pitcher.orange_juice = 2/3 * d.pitcher.capacity)
  (h2 : d.pitcher.apple_juice = 1/3 * d.pitcher.capacity)
  (h3 : d.num_cups = 6)
  (h4 : d.pitcher.capacity > 0) :
  (d.pitcher.capacity / d.num_cups) / d.pitcher.capacity = 1/6 := by
  sorry

#check pitcher_distribution_percentage

end NUMINAMATH_CALUDE_pitcher_distribution_percentage_l1058_105801


namespace NUMINAMATH_CALUDE_exists_good_submatrix_l1058_105840

/-- Definition of a binary matrix -/
def BinaryMatrix (n : ℕ) := Matrix (Fin n) (Fin n) Bool

/-- Definition of a good matrix -/
def IsGoodMatrix {n : ℕ} (A : BinaryMatrix n) : Prop :=
  ∃ (x y : Bool), ∀ (i j : Fin n),
    (i < j → A i j = x) ∧
    (j < i → A i j = y)

/-- Main theorem -/
theorem exists_good_submatrix :
  ∃ (M : ℕ), ∀ (n : ℕ) (A : BinaryMatrix n),
    n > M →
    ∃ (m : ℕ) (indices : Fin m → Fin n),
      Function.Injective indices ∧
      IsGoodMatrix (Matrix.submatrix A indices indices) :=
by sorry

end NUMINAMATH_CALUDE_exists_good_submatrix_l1058_105840


namespace NUMINAMATH_CALUDE_uno_card_discount_l1058_105875

theorem uno_card_discount (original_price : ℝ) (num_cards : ℕ) (total_paid : ℝ) : 
  original_price = 12 → num_cards = 10 → total_paid = 100 → 
  (original_price * num_cards - total_paid) / num_cards = 2 := by
  sorry

end NUMINAMATH_CALUDE_uno_card_discount_l1058_105875


namespace NUMINAMATH_CALUDE_sum_of_120_terms_l1058_105899

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  first_term : ℚ
  common_difference : ℚ

/-- Sum of the first n terms of an arithmetic progression. -/
def sum_of_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.first_term + (n - 1 : ℚ) * ap.common_difference)

/-- Theorem stating the sum of the first 120 terms of a specific arithmetic progression. -/
theorem sum_of_120_terms (ap : ArithmeticProgression) 
  (h1 : sum_of_terms ap 15 = 150)
  (h2 : sum_of_terms ap 115 = 5) :
  sum_of_terms ap 120 = -2620 / 77 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_120_terms_l1058_105899


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1058_105817

/-- Given a line passing through points (-3,1) and (1,3), prove that the sum of its slope and y-intercept is 3. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → (x = -3 ∧ y = 1) ∨ (x = 1 ∧ y = 3)) → 
  m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1058_105817


namespace NUMINAMATH_CALUDE_circle_radius_l1058_105872

/-- The radius of a circle given by the equation x^2 + y^2 - 4x + 2y - 4 = 0 is 3 -/
theorem circle_radius (x y : ℝ) : 
  x^2 + y^2 - 4*x + 2*y - 4 = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 3^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1058_105872


namespace NUMINAMATH_CALUDE_dogwood_trees_in_park_l1058_105835

theorem dogwood_trees_in_park (current_trees : ℕ) : current_trees = 34 :=
  by
  have h1 : current_trees + 49 = 83 := by sorry
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_in_park_l1058_105835


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1058_105828

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ 
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1058_105828


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l1058_105873

theorem smallest_staircase_steps : ∃ n : ℕ,
  n > 20 ∧
  n % 5 = 4 ∧
  n % 6 = 3 ∧
  n % 7 = 5 ∧
  (∀ m : ℕ, m > 20 → m % 5 = 4 → m % 6 = 3 → m % 7 = 5 → m ≥ n) ∧
  n = 159 :=
by sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l1058_105873


namespace NUMINAMATH_CALUDE_calculation_proof_l1058_105842

theorem calculation_proof : (((15^15 / 15^14)^3 * 8^3) / 2^9) = 3375 := by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1058_105842


namespace NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l1058_105832

/-- Represents a right circular cone -/
structure RightCircularCone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ

/-- Calculates the shortest distance between two points on the surface of a cone -/
noncomputable def shortestDistanceOnCone (cone : RightCircularCone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem shortest_distance_on_specific_cone :
  let cone : RightCircularCone := { baseRadius := 600, height := 200 * Real.sqrt 7 }
  let p1 : ConePoint := { distanceFromVertex := 125 }
  let p2 : ConePoint := { distanceFromVertex := 375 * Real.sqrt 2 }
  shortestDistanceOnCone cone p1 p2 = 125 * Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l1058_105832


namespace NUMINAMATH_CALUDE_max_sum_three_consecutive_l1058_105852

/-- A circular arrangement of numbers from 1 to 10 -/
def CircularArrangement := Fin 10 → Fin 10

/-- The sum of three consecutive numbers in a circular arrangement -/
def sumThreeConsecutive (arr : CircularArrangement) (i : Fin 10) : Nat :=
  arr i + arr ((i + 1) % 10) + arr ((i + 2) % 10)

/-- The theorem stating the maximum sum of three consecutive numbers -/
theorem max_sum_three_consecutive :
  (∀ arr : CircularArrangement, ∃ i : Fin 10, sumThreeConsecutive arr i ≥ 18) ∧
  ¬(∀ arr : CircularArrangement, ∃ i : Fin 10, sumThreeConsecutive arr i ≥ 19) :=
sorry

end NUMINAMATH_CALUDE_max_sum_three_consecutive_l1058_105852


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1058_105893

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 2*(m-1)*x + 16 = (x + a)^2) →
  (m = 5 ∨ m = -3) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1058_105893


namespace NUMINAMATH_CALUDE_pipe_stack_total_l1058_105844

/-- Calculates the total number of pipes in a trapezoidal stack -/
def total_pipes (layers : ℕ) (bottom : ℕ) (top : ℕ) : ℕ :=
  (bottom + top) * layers / 2

/-- Proves that a trapezoidal stack of pipes with given parameters contains 88 pipes -/
theorem pipe_stack_total : total_pipes 11 13 3 = 88 := by
  sorry

end NUMINAMATH_CALUDE_pipe_stack_total_l1058_105844


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l1058_105814

theorem simultaneous_equations_solution :
  ∃ (x y : ℚ), 3 * x - 4 * y = -2 ∧ 4 * x + 5 * y = 23 ∧ x = 82/31 ∧ y = 77/31 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l1058_105814


namespace NUMINAMATH_CALUDE_unique_solution_is_two_l1058_105859

def cyclic_index (i n : ℕ) : ℕ := i % n + 1

theorem unique_solution_is_two (x : ℕ → ℕ) (n : ℕ) (hn : n = 20) :
  (∀ i, x i > 0) →
  (∀ i, x (cyclic_index (i + 2) n)^2 = Nat.lcm (x (cyclic_index (i + 1) n)) (x (cyclic_index i n)) + 
                                       Nat.lcm (x (cyclic_index i n)) (x (cyclic_index (i - 1) n))) →
  (∀ i, x i = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_is_two_l1058_105859


namespace NUMINAMATH_CALUDE_different_color_chip_probability_l1058_105826

theorem different_color_chip_probability :
  let total_chips : ℕ := 7 + 5
  let red_chips : ℕ := 7
  let green_chips : ℕ := 5
  let prob_red : ℚ := red_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  let prob_different_colors : ℚ := prob_red * prob_green + prob_green * prob_red
  prob_different_colors = 35 / 72 :=
by sorry

end NUMINAMATH_CALUDE_different_color_chip_probability_l1058_105826


namespace NUMINAMATH_CALUDE_unique_solution_trig_equation_l1058_105894

theorem unique_solution_trig_equation :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ 180 ∧
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trig_equation_l1058_105894


namespace NUMINAMATH_CALUDE_maximize_product_l1058_105887

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 50) :
  x^7 * y^3 ≤ 35^7 * 15^3 ∧
  (x^7 * y^3 = 35^7 * 15^3 ↔ x = 35 ∧ y = 15) :=
sorry

end NUMINAMATH_CALUDE_maximize_product_l1058_105887


namespace NUMINAMATH_CALUDE_balloon_arrangements_count_l1058_105876

/-- The number of distinct arrangements of letters in "balloon" -/
def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating the number of distinct arrangements of letters in "balloon" -/
theorem balloon_arrangements_count : balloon_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_count_l1058_105876


namespace NUMINAMATH_CALUDE_chocolate_candy_cost_difference_l1058_105882

/-- The difference in cost between chocolate and candy bar --/
def cost_difference (chocolate_cost candy_cost : ℕ) : ℕ :=
  chocolate_cost - candy_cost

/-- Theorem stating the difference in cost between chocolate and candy bar --/
theorem chocolate_candy_cost_difference :
  let chocolate_cost : ℕ := 7
  let candy_cost : ℕ := 2
  cost_difference chocolate_cost candy_cost = 5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_candy_cost_difference_l1058_105882


namespace NUMINAMATH_CALUDE_traffic_accident_emergency_number_correct_l1058_105830

def emergency_numbers : List ℕ := [122, 110, 120, 114]

def traffic_accident_emergency_number : ℕ := 122

theorem traffic_accident_emergency_number_correct :
  traffic_accident_emergency_number ∈ emergency_numbers ∧
  traffic_accident_emergency_number = 122 := by
  sorry

end NUMINAMATH_CALUDE_traffic_accident_emergency_number_correct_l1058_105830


namespace NUMINAMATH_CALUDE_max_sum_of_squares_difference_l1058_105834

theorem max_sum_of_squares_difference (x y : ℕ+) : 
  x^2 - y^2 = 2016 → x + y ≤ 1008 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_difference_l1058_105834


namespace NUMINAMATH_CALUDE_bounds_of_W_l1058_105807

/-- Given conditions on x, y, and z, prove the bounds of W = 2x + 6y + 4z -/
theorem bounds_of_W (x y z : ℝ) 
  (sum_eq_one : x + y + z = 1)
  (ineq_one : 3 * y + z ≥ 2)
  (x_bounds : 0 ≤ x ∧ x ≤ 1)
  (y_bounds : 0 ≤ y ∧ y ≤ 2) :
  let W := 2 * x + 6 * y + 4 * z
  ∃ (W_min W_max : ℝ), W_min = 4 ∧ W_max = 6 ∧ W_min ≤ W ∧ W ≤ W_max :=
by sorry

end NUMINAMATH_CALUDE_bounds_of_W_l1058_105807


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1058_105827

/-- A geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

/-- Theorem: For a geometric sequence with a₁ = 2 and a₄ = 16, the common ratio q is 2 -/
theorem geometric_sequence_ratio : 
  ∀ (q : ℝ), 
    (geometric_sequence 2 q 1 = 2) ∧ 
    (geometric_sequence 2 q 4 = 16) → 
    q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1058_105827


namespace NUMINAMATH_CALUDE_inheritance_problem_l1058_105806

/-- Proves the unique solution for the inheritance problem --/
theorem inheritance_problem (A B C : ℕ) : 
  (A + B + C = 30000) →  -- Total inheritance
  (A - B = B - C) →      -- B's relationship to A and C
  (A = B + C) →          -- A's relationship to B and C
  (A = 15000 ∧ B = 10000 ∧ C = 5000) := by
  sorry

end NUMINAMATH_CALUDE_inheritance_problem_l1058_105806


namespace NUMINAMATH_CALUDE_members_not_playing_specific_club_l1058_105836

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  badminton : ℕ
  tennis : ℕ
  both : ℕ

/-- The number of members who don't play either badminton or tennis -/
def members_not_playing (club : SportsClub) : ℕ :=
  club.total - (club.badminton + club.tennis - club.both)

/-- Theorem stating the number of members not playing either sport in the given scenario -/
theorem members_not_playing_specific_club :
  let club : SportsClub := {
    total := 30,
    badminton := 17,
    tennis := 19,
    both := 8
  }
  members_not_playing club = 2 := by
  sorry

end NUMINAMATH_CALUDE_members_not_playing_specific_club_l1058_105836


namespace NUMINAMATH_CALUDE_oh_squared_value_l1058_105848

/-- Given a triangle ABC with circumcenter O, orthocenter H, side lengths a, b, c, and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ

/-- The squared distance between the circumcenter and orthocenter -/
def OH_squared (t : Triangle) : ℝ := 9 * t.R^2 - (t.a^2 + t.b^2 + t.c^2)

theorem oh_squared_value (t : Triangle) 
  (h1 : t.R = 5) 
  (h2 : t.a^2 + t.b^2 + t.c^2 = 50) : 
  OH_squared t = 175 := by
  sorry

end NUMINAMATH_CALUDE_oh_squared_value_l1058_105848


namespace NUMINAMATH_CALUDE_equation_has_four_solutions_l1058_105819

/-- The number of integer solutions to the equation 6y² + 3xy + x + 2y - 72 = 0 -/
def num_solutions : ℕ := 4

/-- The equation 6y² + 3xy + x + 2y - 72 = 0 -/
def equation (x y : ℤ) : Prop :=
  6 * y^2 + 3 * x * y + x + 2 * y - 72 = 0

theorem equation_has_four_solutions :
  ∃! (s : Finset (ℤ × ℤ)), s.card = num_solutions ∧ 
  ∀ (p : ℤ × ℤ), p ∈ s ↔ equation p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_four_solutions_l1058_105819
