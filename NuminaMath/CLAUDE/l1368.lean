import Mathlib

namespace NUMINAMATH_CALUDE_diameter_eq_hypotenuse_l1368_136856

/-- Triangle PQR with sides PQ = 15, QR = 36, and RP = 39 -/
structure RightTriangle where
  PQ : ℝ
  QR : ℝ
  RP : ℝ
  pq_eq : PQ = 15
  qr_eq : QR = 36
  rp_eq : RP = 39
  right_angle : PQ^2 + QR^2 = RP^2

/-- The diameter of the circumscribed circle of a right triangle is equal to the length of its hypotenuse -/
theorem diameter_eq_hypotenuse (t : RightTriangle) : 
  2 * (t.RP / 2) = t.RP := by sorry

end NUMINAMATH_CALUDE_diameter_eq_hypotenuse_l1368_136856


namespace NUMINAMATH_CALUDE_gear_rpm_problem_l1368_136852

/-- The number of revolutions per minute for gear q -/
def q_rpm : ℝ := 40

/-- The duration in minutes -/
def duration : ℝ := 0.5

/-- The difference in revolutions between gear q and gear p after 30 seconds -/
def revolution_difference : ℝ := 15

/-- The number of revolutions per minute for gear p -/
def p_rpm : ℝ := 10

theorem gear_rpm_problem :
  q_rpm * duration - revolution_difference = p_rpm * duration :=
by sorry

end NUMINAMATH_CALUDE_gear_rpm_problem_l1368_136852


namespace NUMINAMATH_CALUDE_negative_sqrt_point_eight_one_equals_negative_point_nine_l1368_136827

theorem negative_sqrt_point_eight_one_equals_negative_point_nine :
  -Real.sqrt 0.81 = -0.9 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_point_eight_one_equals_negative_point_nine_l1368_136827


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1368_136855

/-- The value of p for a parabola y^2 = 2px where the distance between (-2, 3) and the focus is 5 -/
theorem parabola_focus_distance (p : ℝ) : 
  p > 0 → -- Condition that p is positive
  let focus : ℝ × ℝ := (p/2, 0) -- Definition of focus for parabola y^2 = 2px
  (((-2 : ℝ) - p/2)^2 + 3^2).sqrt = 5 → -- Distance formula between (-2, 3) and focus is 5
  p = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1368_136855


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_11_l1368_136867

theorem smallest_multiple_of_45_and_75_not_11 : 
  (∃ n : ℕ+, n * 45 = 225 ∧ n * 75 = 225) ∧ 
  (¬ ∃ m : ℕ+, m * 11 = 225) ∧
  (∀ k : ℕ+, k < 225 → ¬(∃ p : ℕ+, p * 45 = k ∧ p * 75 = k) ∨ (∃ q : ℕ+, q * 11 = k)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_11_l1368_136867


namespace NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l1368_136810

theorem no_real_solutions_for_sqrt_equation :
  ¬∃ x : ℝ, Real.sqrt (4 + 2*x) + Real.sqrt (6 + 3*x) + Real.sqrt (8 + 4*x) = 9 + 3*x/2 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l1368_136810


namespace NUMINAMATH_CALUDE_x_value_in_set_A_l1368_136844

-- Define the set A
def A (x : ℝ) : Set ℝ := {0, -1, x}

-- Theorem statement
theorem x_value_in_set_A (x : ℝ) (h1 : x^2 ∈ A x) (h2 : 0 ≠ -1 ∧ 0 ≠ x ∧ -1 ≠ x) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_set_A_l1368_136844


namespace NUMINAMATH_CALUDE_sqrt_sum_max_value_l1368_136868

theorem sqrt_sum_max_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 2 → Real.sqrt x + Real.sqrt y ≤ m :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_max_value_l1368_136868


namespace NUMINAMATH_CALUDE_sqrt_six_diamond_sqrt_six_l1368_136808

-- Define the operation ¤
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- Theorem statement
theorem sqrt_six_diamond_sqrt_six : diamond (Real.sqrt 6) (Real.sqrt 6) = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_diamond_sqrt_six_l1368_136808


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1368_136846

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1368_136846


namespace NUMINAMATH_CALUDE_first_quadrant_sufficient_not_necessary_l1368_136876

-- Define the complex number z
def z (a : ℝ) : ℂ := a + (a + 1) * Complex.I

-- Define the condition for a point to be in the first quadrant
def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

-- Statement of the theorem
theorem first_quadrant_sufficient_not_necessary (a : ℝ) :
  (is_in_first_quadrant (z a) → a > -1) ∧
  ¬(a > -1 → is_in_first_quadrant (z a)) :=
sorry

end NUMINAMATH_CALUDE_first_quadrant_sufficient_not_necessary_l1368_136876


namespace NUMINAMATH_CALUDE_distance_between_trees_l1368_136834

/-- Given a yard of length 1565 metres with 356 trees planted at equal distances
    (including one at each end), the distance between two consecutive trees
    is equal to 1565 / (356 - 1) metres. -/
theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) 
    (h1 : yard_length = 1565)
    (h2 : num_trees = 356) :
    (yard_length : ℚ) / (num_trees - 1) = 1565 / 355 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_trees_l1368_136834


namespace NUMINAMATH_CALUDE_complement_of_A_range_of_m_for_subset_range_of_m_for_disjoint_l1368_136803

-- Define the sets A and B
def A : Set ℝ := {x | -x^2 - 3*x > 0}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- Theorem for the complement of A
theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x | x ≤ -3 ∨ x ≥ 0} := by sorry

-- Theorem for the range of m when A is a subset of B
theorem range_of_m_for_subset : 
  ∀ m : ℝ, A ⊆ B m → m ≥ 0 := by sorry

-- Theorem for the range of m when A and B are disjoint
theorem range_of_m_for_disjoint : 
  ∀ m : ℝ, A ∩ B m = ∅ → m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_range_of_m_for_subset_range_of_m_for_disjoint_l1368_136803


namespace NUMINAMATH_CALUDE_sebastians_high_school_students_l1368_136872

theorem sebastians_high_school_students (s m : ℕ) : 
  s = 4 * m →  -- Sebastian's high school has 4 times as many students as Mia's
  s + m = 3000 →  -- The total number of students in both schools is 3000
  s = 2400 :=  -- Sebastian's high school has 2400 students
by sorry

end NUMINAMATH_CALUDE_sebastians_high_school_students_l1368_136872


namespace NUMINAMATH_CALUDE_square_side_length_average_l1368_136813

theorem square_side_length_average : 
  let areas : List ℝ := [25, 64, 121, 196]
  let side_lengths := areas.map Real.sqrt
  (side_lengths.sum / side_lengths.length : ℝ) = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l1368_136813


namespace NUMINAMATH_CALUDE_seating_theorem_l1368_136886

/-- Represents a group of people seated around a round table -/
structure SeatingArrangement where
  num_men : ℕ
  num_women : ℕ

/-- A man is satisfied if at least one woman is sitting next to him -/
def is_satisfied (s : SeatingArrangement) : Prop :=
  ∃ (p : ℝ), p = 1 - (s.num_men - 1) / (s.num_men + s.num_women - 1) * (s.num_men - 2) / (s.num_men + s.num_women - 2)

/-- The probability of a specific man being satisfied -/
def satisfaction_probability (s : SeatingArrangement) : ℚ :=
  25 / 33

/-- The expected number of satisfied men -/
def expected_satisfied_men (s : SeatingArrangement) : ℚ :=
  (s.num_men : ℚ) * (satisfaction_probability s)

/-- The main theorem about the seating arrangement -/
theorem seating_theorem (s : SeatingArrangement) 
  (h1 : s.num_men = 50) 
  (h2 : s.num_women = 50) : 
  is_satisfied s ∧ 
  satisfaction_probability s = 25 / 33 ∧ 
  expected_satisfied_men s = 1250 / 33 := by
  sorry


end NUMINAMATH_CALUDE_seating_theorem_l1368_136886


namespace NUMINAMATH_CALUDE_average_of_expressions_l1368_136836

theorem average_of_expressions (x : ℚ) : 
  (1/3 : ℚ) * ((2*x + 8) + (5*x + 3) + (3*x + 9)) = 3*x + 2 → x = -14 := by
sorry

end NUMINAMATH_CALUDE_average_of_expressions_l1368_136836


namespace NUMINAMATH_CALUDE_find_divisor_l1368_136878

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 125 → 
  quotient = 8 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder →
  divisor = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1368_136878


namespace NUMINAMATH_CALUDE_intersection_M_N_l1368_136839

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1368_136839


namespace NUMINAMATH_CALUDE_bookcase_weight_theorem_l1368_136809

/-- Represents the weight of the bookcase and items -/
def BookcaseWeightProblem : Prop :=
  let bookcaseLimit : ℕ := 80
  let hardcoverCount : ℕ := 70
  let hardcoverWeight : ℚ := 1/2
  let textbookCount : ℕ := 30
  let textbookWeight : ℕ := 2
  let knickknackCount : ℕ := 3
  let knickknackWeight : ℕ := 6
  let totalWeight : ℚ := 
    hardcoverCount * hardcoverWeight + 
    textbookCount * textbookWeight + 
    knickknackCount * knickknackWeight
  totalWeight - bookcaseLimit = 33

theorem bookcase_weight_theorem : BookcaseWeightProblem := by
  sorry

end NUMINAMATH_CALUDE_bookcase_weight_theorem_l1368_136809


namespace NUMINAMATH_CALUDE_quadratic_root_value_l1368_136873

theorem quadratic_root_value (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 - 2*x + a^2 - 1 = 0 ↔ x = 0 ∨ x ≠ 0) →
  (a - 1 ≠ 0) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l1368_136873


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_k_l1368_136885

theorem quadratic_roots_imply_k (k : ℝ) : 
  (∀ x : ℂ, 8 * x^2 + 4 * x + k = 0 ↔ x = (-4 + Complex.I * Real.sqrt 380) / 16 ∨ x = (-4 - Complex.I * Real.sqrt 380) / 16) →
  k = 12.375 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_k_l1368_136885


namespace NUMINAMATH_CALUDE_gcd_9125_4277_l1368_136887

theorem gcd_9125_4277 : Nat.gcd 9125 4277 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9125_4277_l1368_136887


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1368_136879

theorem arithmetic_sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 4) (h2 : ∀ n : ℕ, a (n + 1) - a n = 3) :
  ∀ n : ℕ, a n = 3 * n + 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1368_136879


namespace NUMINAMATH_CALUDE_book_pages_theorem_l1368_136884

/-- Calculates the number of pages with text in a book with given specifications. -/
def pages_with_text (total_pages : ℕ) (image_pages : ℕ) (intro_pages : ℕ) : ℕ :=
  let remaining_pages := total_pages - image_pages - intro_pages
  remaining_pages / 2

/-- Theorem stating that a book with 98 pages, half images, 11 intro pages, 
    and remaining pages split equally between blank and text, has 19 pages of text. -/
theorem book_pages_theorem : 
  pages_with_text 98 (98 / 2) 11 = 19 := by
sorry

#eval pages_with_text 98 (98 / 2) 11

end NUMINAMATH_CALUDE_book_pages_theorem_l1368_136884


namespace NUMINAMATH_CALUDE_f_has_two_roots_l1368_136832

/-- The function f(x) = x^4 + 5x^3 + 6x^2 - 4x - 16 -/
def f (x : ℝ) : ℝ := x^4 + 5*x^3 + 6*x^2 - 4*x - 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4*x^3 + 15*x^2 + 12*x - 4

theorem f_has_two_roots :
  ∃! (a b : ℝ), a < b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_roots_l1368_136832


namespace NUMINAMATH_CALUDE_race_time_proof_l1368_136825

/-- The time A takes to complete the race -/
def race_time_A : ℝ := 390

/-- The distance of the race in meters -/
def race_distance : ℝ := 1000

/-- The difference in distance between A and B at the finish -/
def distance_diff_AB : ℝ := 25

/-- The time difference between A and B -/
def time_diff_AB : ℝ := 10

/-- The difference in distance between A and C at the finish -/
def distance_diff_AC : ℝ := 40

/-- The time difference between A and C -/
def time_diff_AC : ℝ := 8

/-- The difference in distance between B and C at the finish -/
def distance_diff_BC : ℝ := 15

/-- The time difference between B and C -/
def time_diff_BC : ℝ := 2

theorem race_time_proof :
  let v_a := race_distance / race_time_A
  let v_b := (race_distance - distance_diff_AB) / race_time_A
  let v_c := (race_distance - distance_diff_AC) / race_time_A
  (v_b * (race_time_A + time_diff_AB) = race_distance) ∧
  (v_c * (race_time_A + time_diff_AC) = race_distance) ∧
  (v_c * (race_time_A + time_diff_AB + time_diff_BC) = race_distance) →
  race_time_A = 390 := by
sorry

end NUMINAMATH_CALUDE_race_time_proof_l1368_136825


namespace NUMINAMATH_CALUDE_max_steps_17_steps_17_possible_l1368_136890

/-- Represents the number of toothpicks used for n steps in Mandy's staircase -/
def toothpicks (n : ℕ) : ℕ := n * (n + 5)

/-- Theorem stating that 17 is the maximum number of steps that can be built with 380 toothpicks -/
theorem max_steps_17 :
  ∀ n : ℕ, toothpicks n ≤ 380 → n ≤ 17 :=
by
  sorry

/-- Theorem stating that 17 steps can indeed be built with 380 toothpicks -/
theorem steps_17_possible :
  toothpicks 17 ≤ 380 :=
by
  sorry

end NUMINAMATH_CALUDE_max_steps_17_steps_17_possible_l1368_136890


namespace NUMINAMATH_CALUDE_total_amount_shared_l1368_136840

theorem total_amount_shared (z y x : ℝ) : 
  z = 150 →
  y = 1.2 * z →
  x = 1.25 * y →
  x + y + z = 555 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_shared_l1368_136840


namespace NUMINAMATH_CALUDE_problem_solution_l1368_136829

def f (x : ℝ) := |3*x - 2| + |x - 2|

theorem problem_solution :
  (∀ x : ℝ, f x ≤ 8 ↔ x ∈ Set.Icc (-1) 3) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ≠ 0 → f x ≥ (m^2 - m + 2) * |x|) → m ∈ Set.Icc 0 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1368_136829


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l1368_136871

theorem least_number_for_divisibility (n m : ℕ) (h : n = 1056 ∧ m = 26) :
  ∃ x : ℕ, (x = 10 ∧ (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l1368_136871


namespace NUMINAMATH_CALUDE_trig_simplification_l1368_136850

open Real

theorem trig_simplification (α : ℝ) (n : ℤ) :
  ((-sin (α + π) + sin (-α) - tan (2*π + α)) / 
   (tan (α + π) + cos (-α) + cos (π - α)) = -1) ∧
  ((sin (α + n*π) + sin (α - n*π)) / 
   (sin (α + n*π) * cos (α - n*π)) = 
     if n % 2 = 0 then 2 / cos α else -2 / cos α) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l1368_136850


namespace NUMINAMATH_CALUDE_f_max_at_a_l1368_136805

/-- The function f(x) = x^3 - 12x -/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- The maximum value point of f(x) -/
def a : ℝ := -2

theorem f_max_at_a : IsLocalMax f a := by sorry

end NUMINAMATH_CALUDE_f_max_at_a_l1368_136805


namespace NUMINAMATH_CALUDE_inequality_solution_l1368_136826

theorem inequality_solution (x : ℕ+) : 4 - (x : ℝ) > 1 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1368_136826


namespace NUMINAMATH_CALUDE_largest_divisor_of_cube_difference_l1368_136833

theorem largest_divisor_of_cube_difference (n : ℤ) (h : 5 ∣ n) :
  (∃ (m : ℤ), m ∣ (n^3 - n) ∧ ∀ (k : ℤ), k ∣ (n^3 - n) → k ≤ m) → 
  (∃ (m : ℤ), m ∣ (n^3 - n) ∧ ∀ (k : ℤ), k ∣ (n^3 - n) → k ≤ m) ∧ m = 10 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_cube_difference_l1368_136833


namespace NUMINAMATH_CALUDE_triangle_formation_l1368_136812

/-- A line in 2D space represented by ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines form a triangle --/
def form_triangle (l1 l2 l3 : Line) : Prop :=
  ∃ (x y : ℝ), (l1.a * x + l1.b * y = l1.c) ∧ 
                (l2.a * x + l2.b * y = l2.c) ∧ 
                (l3.a * x + l3.b * y = l3.c)

theorem triangle_formation (m : ℝ) : 
  ¬(form_triangle 
      ⟨1, 1, 2⟩  -- x + y = 2
      ⟨m, 1, 0⟩  -- mx + y = 0
      ⟨1, -1, 4⟩ -- x - y = 4
    ) ↔ m = 1/3 ∨ m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_formation_l1368_136812


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1368_136824

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, x > 1 ↔ (x - 1 > 0 ∧ 2*x - a > 0)) →
  a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1368_136824


namespace NUMINAMATH_CALUDE_quintic_integer_root_counts_l1368_136897

/-- The set of possible numbers of integer roots (counting multiplicity) for a quintic polynomial with integer coefficients -/
def QuinticIntegerRootCounts : Set ℕ := {0, 1, 2, 3, 4, 5}

/-- A quintic polynomial with integer coefficients -/
structure QuinticPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The number of integer roots (counting multiplicity) of a quintic polynomial -/
def integerRootCount (p : QuinticPolynomial) : ℕ := sorry

theorem quintic_integer_root_counts (p : QuinticPolynomial) :
  integerRootCount p ∈ QuinticIntegerRootCounts := by sorry

end NUMINAMATH_CALUDE_quintic_integer_root_counts_l1368_136897


namespace NUMINAMATH_CALUDE_intersection_at_most_one_point_f_composition_half_l1368_136895

-- Statement B
theorem intersection_at_most_one_point (f : ℝ → ℝ) :
  ∃ (y : ℝ), ∀ (y' : ℝ), f 1 = y' → y = y' :=
sorry

-- Statement D
def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem f_composition_half : f (f (1/2)) = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_at_most_one_point_f_composition_half_l1368_136895


namespace NUMINAMATH_CALUDE_other_divisor_problem_l1368_136865

theorem other_divisor_problem (n : ℕ) (h1 : n = 266) (h2 : n % 33 = 2) : 
  ∃ (x : ℕ), x ≠ 33 ∧ n % x = 2 ∧ x = 132 ∧ ∀ y : ℕ, y ≠ 33 → n % y = 2 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_other_divisor_problem_l1368_136865


namespace NUMINAMATH_CALUDE_painters_work_days_l1368_136807

theorem painters_work_days (painters_initial : ℕ) (painters_new : ℕ) (days_initial : ℚ) : 
  painters_initial = 5 → 
  painters_new = 4 → 
  days_initial = 3/2 → 
  ∃ (days_new : ℚ), days_new = 15/8 ∧ 
    painters_initial * days_initial = painters_new * days_new :=
by sorry

end NUMINAMATH_CALUDE_painters_work_days_l1368_136807


namespace NUMINAMATH_CALUDE_geometric_mean_relationship_l1368_136838

theorem geometric_mean_relationship (m : ℝ) : 
  (m = 4 → m^2 = 2 * 8) ∧ ¬(m^2 = 2 * 8 → m = 4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_relationship_l1368_136838


namespace NUMINAMATH_CALUDE_ratio_sum_max_l1368_136816

theorem ratio_sum_max (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 3 / 4) (h4 : a + b = 21) : 
  max a b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_max_l1368_136816


namespace NUMINAMATH_CALUDE_function_condition_l1368_136893

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem function_condition (a : ℝ) : f a (f a 0) = 3 * a → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_condition_l1368_136893


namespace NUMINAMATH_CALUDE_park_rose_bushes_l1368_136823

/-- The number of rose bushes in a park after planting new ones -/
def total_rose_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The park will have 6 rose bushes after planting -/
theorem park_rose_bushes : total_rose_bushes 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_park_rose_bushes_l1368_136823


namespace NUMINAMATH_CALUDE_perpendicular_implies_cos_value_triangle_implies_f_range_l1368_136854

noncomputable section

-- Define the vectors m and n
def m (x : ℝ) : Fin 2 → ℝ := ![Real.sqrt 3 * Real.sin (x/4), 1]
def n (x : ℝ) : Fin 2 → ℝ := ![Real.cos (x/4), Real.cos (x/4)^2]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define perpendicularity
def perpendicular (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

-- Define the function f
def f (x : ℝ) : ℝ := dot_product (m x) (n x)

-- Theorem 1
theorem perpendicular_implies_cos_value (x : ℝ) :
  perpendicular (m x) (n x) → Real.cos (2 * Real.pi / 3 - x) = -1/2 := by sorry

-- Theorem 2
theorem triangle_implies_f_range (A B C a b c : ℝ) :
  A + B + C = Real.pi →
  (2 * a - c) * Real.cos B = b * Real.cos C →
  0 < A →
  A < 2 * Real.pi / 3 →
  ∃ (y : ℝ), 1 < f A ∧ f A < 3/2 := by sorry

end

end NUMINAMATH_CALUDE_perpendicular_implies_cos_value_triangle_implies_f_range_l1368_136854


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l1368_136860

theorem two_digit_number_problem : ∃ x : ℕ, 
  10 ≤ x ∧ x < 100 ∧ 10 * x + 6 = x + 474 → x = 52 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l1368_136860


namespace NUMINAMATH_CALUDE_great_circle_bisects_angle_l1368_136894

-- Define the sphere
def Sphere : Type := ℝ × ℝ × ℝ

-- Define the north pole
def N : Sphere := (0, 0, 1)

-- Define a great circle
def GreatCircle (p q : Sphere) : Type := sorry

-- Define a point on the equator
def OnEquator (p : Sphere) : Prop := sorry

-- Define equidistance from a point
def Equidistant (a b c : Sphere) : Prop := sorry

-- Define angle bisection on a sphere
def AngleBisector (a b c d : Sphere) : Prop := sorry

-- Theorem statement
theorem great_circle_bisects_angle 
  (A B C : Sphere) 
  (h1 : GreatCircle N A)
  (h2 : GreatCircle N B)
  (h3 : Equidistant N A B)
  (h4 : OnEquator C) :
  AngleBisector C N A B :=
sorry

end NUMINAMATH_CALUDE_great_circle_bisects_angle_l1368_136894


namespace NUMINAMATH_CALUDE_softball_team_ratio_l1368_136889

theorem softball_team_ratio :
  ∀ (men women : ℕ),
  men + women = 16 →
  women = men + 2 →
  (men : ℚ) / women = 7 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l1368_136889


namespace NUMINAMATH_CALUDE_smallest_difference_l1368_136853

def Digits : Finset Nat := {0, 3, 4, 7, 8}

def isValidArrangement (a b c d e : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧ e ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a ≠ 0

def difference (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) - (10 * d + e)

theorem smallest_difference :
  ∀ a b c d e,
    isValidArrangement a b c d e →
    difference a b c d e ≥ 339 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_l1368_136853


namespace NUMINAMATH_CALUDE_train_length_calculation_l1368_136851

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) :
  train_speed = 36 * (1000 / 3600) →
  time_to_cross = 82.49340052795776 →
  bridge_length = 660 →
  train_speed * time_to_cross - bridge_length = 164.9340052795776 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1368_136851


namespace NUMINAMATH_CALUDE_max_sum_sides_is_ten_l1368_136848

/-- Represents a configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  
/-- Represents a region formed by the intersection of lines -/
structure Region where
  num_sides : ℕ

/-- Represents two neighboring regions -/
structure NeighboringRegions where
  region1 : Region
  region2 : Region

/-- The maximum sum of sides for two neighboring regions in a configuration with 7 lines -/
def max_sum_sides (config : LineConfiguration) : ℕ :=
  10

/-- Theorem: The maximum sum of sides for two neighboring regions in a configuration with 7 lines is 10 -/
theorem max_sum_sides_is_ten (config : LineConfiguration) 
  (h : config.num_lines = 7) : 
  ∀ (neighbors : NeighboringRegions), 
    neighbors.region1.num_sides + neighbors.region2.num_sides ≤ max_sum_sides config :=
by
  sorry

#check max_sum_sides_is_ten

end NUMINAMATH_CALUDE_max_sum_sides_is_ten_l1368_136848


namespace NUMINAMATH_CALUDE_kenneth_earnings_l1368_136843

/-- Kenneth's earnings problem -/
theorem kenneth_earnings (earnings : ℝ) 
  (h1 : earnings * 0.1 + earnings * 0.15 + 75 + 80 + 405 = earnings) : 
  earnings = 746.67 := by
sorry

end NUMINAMATH_CALUDE_kenneth_earnings_l1368_136843


namespace NUMINAMATH_CALUDE_triangle_properties_l1368_136801

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B = 3 * t.C ∧
  2 * Real.sin (t.A - t.C) = Real.sin t.B ∧
  t.AB = 5

-- Define the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.A = 3 * (10 ^ (1/2 : ℝ)) / 10 ∧
  ∃ (height : ℝ), height = 6 ∧ 
    height * t.AB / 2 = Real.sin t.C * (Real.sin t.A * t.AB / Real.sin t.C) * (Real.sin t.B * t.AB / Real.sin t.C) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1368_136801


namespace NUMINAMATH_CALUDE_y_value_proof_l1368_136881

theorem y_value_proof (y : ℝ) (h : (9 : ℝ) / y^3 = y / 81) : y = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l1368_136881


namespace NUMINAMATH_CALUDE_c_used_car_for_13_hours_l1368_136828

/-- Represents the car rental scenario -/
structure CarRental where
  totalCost : ℝ
  aHours : ℝ
  bHours : ℝ
  bPaid : ℝ
  cHours : ℝ

/-- Theorem stating that under the given conditions, c used the car for 13 hours -/
theorem c_used_car_for_13_hours (rental : CarRental) 
  (h1 : rental.totalCost = 720)
  (h2 : rental.aHours = 9)
  (h3 : rental.bHours = 10)
  (h4 : rental.bPaid = 225) :
  rental.cHours = 13 := by
  sorry

#check c_used_car_for_13_hours

end NUMINAMATH_CALUDE_c_used_car_for_13_hours_l1368_136828


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_126_l1368_136849

theorem sqrt_nine_factorial_over_126 : 
  Real.sqrt (Nat.factorial 9 / 126) = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_126_l1368_136849


namespace NUMINAMATH_CALUDE_average_stream_speed_theorem_l1368_136806

/-- Represents the swimming scenario with given parameters. -/
structure SwimmingScenario where
  swimmer_speed : ℝ  -- Speed of the swimmer in still water (km/h)
  upstream_time_ratio : ℝ  -- Ratio of upstream time to downstream time
  stream_speed_increase : ℝ  -- Increase in stream speed per 100 meters (km/h)
  upstream_distance : ℝ  -- Total upstream distance (meters)

/-- Calculates the average stream speed over the given distance. -/
def average_stream_speed (scenario : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating the average stream speed for the given scenario. -/
theorem average_stream_speed_theorem (scenario : SwimmingScenario) 
  (h1 : scenario.swimmer_speed = 1.5)
  (h2 : scenario.upstream_time_ratio = 2)
  (h3 : scenario.stream_speed_increase = 0.2)
  (h4 : scenario.upstream_distance = 500) :
  average_stream_speed scenario = 0.7 :=
sorry

end NUMINAMATH_CALUDE_average_stream_speed_theorem_l1368_136806


namespace NUMINAMATH_CALUDE_equation_solution_l1368_136842

theorem equation_solution : ∃ x : ℝ, (5 + 3.5 * x = 2.5 * x - 25) ∧ (x = -30) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1368_136842


namespace NUMINAMATH_CALUDE_apples_per_box_l1368_136898

theorem apples_per_box (total_apples : ℕ) (num_boxes : ℕ) (apples_per_box : ℕ) : 
  total_apples = 49 → num_boxes = 7 → total_apples = num_boxes * apples_per_box → apples_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_box_l1368_136898


namespace NUMINAMATH_CALUDE_sum_of_coefficients_for_factored_form_l1368_136892

theorem sum_of_coefficients_for_factored_form : ∃ (a b c d e f : ℤ),
  (2401 : ℤ) * x^4 + 16 = (a * x + b) * (c * x^3 + d * x^2 + e * x + f) ∧
  a + b + c + d + e + f = 274 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_for_factored_form_l1368_136892


namespace NUMINAMATH_CALUDE_locus_of_midpoint_l1368_136837

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 13

-- Define a point P on the circle
def point_P (x y : ℝ) : Prop := circle_O x y

-- Define Q as the foot of the perpendicular from P to the y-axis
def point_Q (x y : ℝ) : Prop := x = 0

-- Define M as the midpoint of PQ
def point_M (x y px py : ℝ) : Prop := x = px / 2 ∧ y = py

-- Theorem statement
theorem locus_of_midpoint :
  ∀ (x y px py : ℝ),
  point_P px py →
  point_Q 0 py →
  point_M x y px py →
  (x^2 / (13/4) + y^2 / 13 = 1) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_midpoint_l1368_136837


namespace NUMINAMATH_CALUDE_thirteenth_result_l1368_136814

theorem thirteenth_result (total_count : Nat) (total_average : ℝ) 
  (first_twelve_average : ℝ) (last_twelve_average : ℝ) :
  total_count = 25 →
  total_average = 20 →
  first_twelve_average = 14 →
  last_twelve_average = 17 →
  (12 * first_twelve_average + 12 * last_twelve_average + 
    (total_count * total_average - 12 * first_twelve_average - 12 * last_twelve_average)) / 1 = 128 := by
  sorry

#check thirteenth_result

end NUMINAMATH_CALUDE_thirteenth_result_l1368_136814


namespace NUMINAMATH_CALUDE_trajectory_length_l1368_136869

/-- The curve y = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The line x = 2 on which point A moves -/
def line_x_eq_2 (x : ℝ) : Prop := x = 2

/-- The tangent line to the curve at point (x₀, f x₀) -/
def tangent_line (x₀ : ℝ) (x a : ℝ) : Prop :=
  a = (3 * x₀^2 - 1) * (x - x₀) + f x₀

/-- The condition for point A(2, a) to have a tangent line to the curve -/
def has_tangent (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, tangent_line x₀ 2 a

/-- The statement to be proved -/
theorem trajectory_length :
  ∀ a : ℝ, line_x_eq_2 2 →
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    has_tangent a ∧
    (∀ x : ℝ, has_tangent a → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  (∃ a_min a_max : ℝ, 
    (∀ a' : ℝ, has_tangent a' → a_min ≤ a' ∧ a' ≤ a_max) ∧
    a_max - a_min = 8) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_length_l1368_136869


namespace NUMINAMATH_CALUDE_price_per_drawing_l1368_136863

-- Define the variables
def saturday_sales : ℕ := 24
def sunday_sales : ℕ := 16
def total_revenue : ℕ := 800

-- Define the theorem
theorem price_per_drawing : 
  ∃ (price : ℚ), price * (saturday_sales + sunday_sales) = total_revenue ∧ price = 20 := by
  sorry

end NUMINAMATH_CALUDE_price_per_drawing_l1368_136863


namespace NUMINAMATH_CALUDE_notebook_packages_l1368_136861

theorem notebook_packages (L : ℕ) : L > 4 →
  (∃ a b : ℕ, a > 0 ∧ a * L + 4 * b = 69) →
  L = 23 := by sorry

end NUMINAMATH_CALUDE_notebook_packages_l1368_136861


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1368_136888

theorem greatest_distance_between_circle_centers 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (circle_diameter : ℝ) 
  (h1 : rectangle_width = 16) 
  (h2 : rectangle_height = 20) 
  (h3 : circle_diameter = 8) :
  ∃ (d : ℝ), d = 4 * Real.sqrt 13 ∧ 
  ∀ (d' : ℝ), d' ≤ d ∧ 
  ∃ (x1 y1 x2 y2 : ℝ),
    0 ≤ x1 ∧ x1 ≤ rectangle_width ∧
    0 ≤ y1 ∧ y1 ≤ rectangle_height ∧
    0 ≤ x2 ∧ x2 ≤ rectangle_width ∧
    0 ≤ y2 ∧ y2 ≤ rectangle_height ∧
    (x1 - circle_diameter / 2 ≥ 0) ∧ (x1 + circle_diameter / 2 ≤ rectangle_width) ∧
    (y1 - circle_diameter / 2 ≥ 0) ∧ (y1 + circle_diameter / 2 ≤ rectangle_height) ∧
    (x2 - circle_diameter / 2 ≥ 0) ∧ (x2 + circle_diameter / 2 ≤ rectangle_width) ∧
    (y2 - circle_diameter / 2 ≥ 0) ∧ (y2 + circle_diameter / 2 ≤ rectangle_height) ∧
    d' = Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) := by
  sorry


end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1368_136888


namespace NUMINAMATH_CALUDE_same_direction_condition_l1368_136899

/-- Two vectors are in the same direction if one is a positive scalar multiple of the other -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m > 0 ∧ a = (m * b.1, m * b.2)

/-- The condition for vectors a and b to be in the same direction -/
theorem same_direction_condition (k : ℝ) :
  same_direction (k, 2) (1, 1) ↔ k = 2 := by
  sorry

#check same_direction_condition

end NUMINAMATH_CALUDE_same_direction_condition_l1368_136899


namespace NUMINAMATH_CALUDE_race_time_calculation_l1368_136866

/-- 
Given a 100-meter race where:
- Runner A beats runner B by 20 meters
- Runner B finishes the race in 45 seconds

This theorem proves that runner A finishes the race in 36 seconds.
-/
theorem race_time_calculation (race_distance : ℝ) (b_time : ℝ) (distance_difference : ℝ) 
  (h1 : race_distance = 100)
  (h2 : b_time = 45)
  (h3 : distance_difference = 20) : 
  ∃ (a_time : ℝ), a_time = 36 ∧ 
  (race_distance / a_time = (race_distance - distance_difference) / a_time) ∧
  ((race_distance - distance_difference) / a_time = race_distance / b_time) :=
sorry

end NUMINAMATH_CALUDE_race_time_calculation_l1368_136866


namespace NUMINAMATH_CALUDE_smallest_integer_divisible_l1368_136862

theorem smallest_integer_divisible (n : ℕ) : n = 43179 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 48 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 64 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 75 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 108 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 21) = 48 * k₁ ∧ (n + 21) = 64 * k₂ ∧ (n + 21) = 75 * k₃ ∧ (n + 21) = 108 * k₄) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_divisible_l1368_136862


namespace NUMINAMATH_CALUDE_pie_cost_l1368_136845

def mary_initial_amount : ℕ := 58
def mary_remaining_amount : ℕ := 52

theorem pie_cost : mary_initial_amount - mary_remaining_amount = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_cost_l1368_136845


namespace NUMINAMATH_CALUDE_prob_at_least_6_heads_value_l1368_136818

/-- The probability of getting at least 6 heads when flipping a fair coin 8 times -/
def prob_at_least_6_heads : ℚ :=
  (Nat.choose 8 6 + Nat.choose 8 7 + Nat.choose 8 8) / 2^8

/-- Theorem stating that the probability of getting at least 6 heads
    when flipping a fair coin 8 times is 37/256 -/
theorem prob_at_least_6_heads_value :
  prob_at_least_6_heads = 37 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_6_heads_value_l1368_136818


namespace NUMINAMATH_CALUDE_distance_difference_l1368_136880

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Grayson's first leg speed in mph -/
def grayson_speed1 : ℝ := 25

/-- Grayson's first leg time in hours -/
def grayson_time1 : ℝ := 1

/-- Grayson's second leg speed in mph -/
def grayson_speed2 : ℝ := 20

/-- Grayson's second leg time in hours -/
def grayson_time2 : ℝ := 0.5

/-- Rudy's speed in mph -/
def rudy_speed : ℝ := 10

/-- Rudy's time in hours -/
def rudy_time : ℝ := 3

/-- The difference in distance traveled between Grayson and Rudy -/
theorem distance_difference : 
  distance grayson_speed1 grayson_time1 + distance grayson_speed2 grayson_time2 - 
  distance rudy_speed rudy_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l1368_136880


namespace NUMINAMATH_CALUDE_fourteen_n_divisibility_l1368_136896

theorem fourteen_n_divisibility (n d : ℕ) (p₁ p₂ p₃ : ℕ) 
  (h1 : 0 < n ∧ n < 200)
  (h2 : Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃)
  (h3 : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h4 : n = p₁ * p₂ * p₃)
  (h5 : (14 * n) % d = 0) : 
  d = n := by
sorry

end NUMINAMATH_CALUDE_fourteen_n_divisibility_l1368_136896


namespace NUMINAMATH_CALUDE_range_of_a_l1368_136875

-- Define a monotonically decreasing function
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : MonotonicallyDecreasing f) 
  (h2 : f (2 - a^2) > f a) : 
  a > 1 ∨ a < -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1368_136875


namespace NUMINAMATH_CALUDE_ackermann_3_2_l1368_136817

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_3_2 : A 3 2 = 11 := by sorry

end NUMINAMATH_CALUDE_ackermann_3_2_l1368_136817


namespace NUMINAMATH_CALUDE_parrot_guinea_pig_ownership_l1368_136800

theorem parrot_guinea_pig_ownership (total : ℕ) (parrot : ℕ) (guinea_pig : ℕ) :
  total = 48 →
  parrot = 30 →
  guinea_pig = 35 →
  ∃ (both : ℕ), both = 17 ∧ total = parrot + guinea_pig - both :=
by
  sorry

end NUMINAMATH_CALUDE_parrot_guinea_pig_ownership_l1368_136800


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l1368_136859

/-- A line in the plane can be represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if and only if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := 3, point := (0, 2) } →
  b.point = (5, 10) →
  y_intercept b = -5 := by
  sorry

#check y_intercept_of_parallel_line

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l1368_136859


namespace NUMINAMATH_CALUDE_mairead_exercise_distance_l1368_136874

theorem mairead_exercise_distance :
  let run_distance : ℝ := 40
  let walk_distance : ℝ := (3/5) * run_distance
  let jog_distance : ℝ := (1/5) * walk_distance
  let total_distance : ℝ := run_distance + walk_distance + jog_distance
  total_distance = 64.8 := by
  sorry

end NUMINAMATH_CALUDE_mairead_exercise_distance_l1368_136874


namespace NUMINAMATH_CALUDE_atMostOneHead_exactlyTwoHeads_mutuallyExclusive_l1368_136882

/-- Represents the outcome of tossing a coin -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the result of tossing two coins simultaneously -/
def TwoCoinsResult := (CoinOutcome × CoinOutcome)

/-- The set of all possible outcomes when tossing two coins -/
def sampleSpace : Set TwoCoinsResult := {(CoinOutcome.Heads, CoinOutcome.Heads),
                                         (CoinOutcome.Heads, CoinOutcome.Tails),
                                         (CoinOutcome.Tails, CoinOutcome.Heads),
                                         (CoinOutcome.Tails, CoinOutcome.Tails)}

/-- The event of getting at most 1 head -/
def atMostOneHead : Set TwoCoinsResult := {(CoinOutcome.Heads, CoinOutcome.Tails),
                                           (CoinOutcome.Tails, CoinOutcome.Heads),
                                           (CoinOutcome.Tails, CoinOutcome.Tails)}

/-- The event of getting exactly 2 heads -/
def exactlyTwoHeads : Set TwoCoinsResult := {(CoinOutcome.Heads, CoinOutcome.Heads)}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set TwoCoinsResult) : Prop := A ∩ B = ∅

theorem atMostOneHead_exactlyTwoHeads_mutuallyExclusive :
  mutuallyExclusive atMostOneHead exactlyTwoHeads := by
  sorry

end NUMINAMATH_CALUDE_atMostOneHead_exactlyTwoHeads_mutuallyExclusive_l1368_136882


namespace NUMINAMATH_CALUDE_trapezoid_area_is_half_sq_dm_l1368_136820

/-- A trapezoid with specific measurements -/
structure Trapezoid where
  smallBase : ℝ
  adjacentAngle : ℝ
  diagonalAngle : ℝ

/-- The area of a trapezoid with given measurements -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  0.5

/-- Theorem stating that a trapezoid with the given measurements has an area of 0.5 square decimeters -/
theorem trapezoid_area_is_half_sq_dm (t : Trapezoid) 
    (h1 : t.smallBase = 1)
    (h2 : t.adjacentAngle = 135 * π / 180)
    (h3 : t.diagonalAngle = 150 * π / 180) :
    trapezoidArea t = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_half_sq_dm_l1368_136820


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l1368_136822

/-- The minimum distance between a point on the circle (x + 1)² + y² = 1 
    and a point on the line 3x + 4y + 13 = 0 is equal to 1. -/
theorem min_distance_circle_line : 
  ∃ (d : ℝ), d = 1 ∧ 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    ((x₁ + 1)^2 + y₁^2 = 1) →
    (3*x₂ + 4*y₂ + 13 = 0) →
    ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_line_l1368_136822


namespace NUMINAMATH_CALUDE_corner_sum_equality_l1368_136857

/-- A matrix satisfying the given condition for any 2x2 sub-matrix -/
def SpecialMatrix (n : ℕ) := Matrix (Fin n) (Fin n) ℝ

/-- The condition that must hold for any 2x2 sub-matrix -/
def satisfies_condition (A : SpecialMatrix 2000) : Prop :=
  ∀ i j, i.val < 1999 → j.val < 1999 →
    A i j + A (Fin.succ i) (Fin.succ j) = A i (Fin.succ j) + A (Fin.succ i) j

/-- The theorem to be proved -/
theorem corner_sum_equality (A : SpecialMatrix 2000) (h : satisfies_condition A) :
  A 0 0 + A 1999 1999 = A 0 1999 + A 1999 0 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_equality_l1368_136857


namespace NUMINAMATH_CALUDE_anoop_joining_time_l1368_136847

/-- Proves that Anoop joined after 6 months given the investment conditions -/
theorem anoop_joining_time (arjun_investment anoop_investment : ℕ) 
  (total_months : ℕ) (x : ℕ) :
  arjun_investment = 20000 →
  anoop_investment = 40000 →
  total_months = 12 →
  arjun_investment * total_months = anoop_investment * (total_months - x) →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_anoop_joining_time_l1368_136847


namespace NUMINAMATH_CALUDE_nancy_keeps_ten_l1368_136802

def nancy_chips : ℕ := 22
def brother_chips : ℕ := 7
def sister_chips : ℕ := 5

theorem nancy_keeps_ten : 
  nancy_chips - (brother_chips + sister_chips) = 10 := by
  sorry

end NUMINAMATH_CALUDE_nancy_keeps_ten_l1368_136802


namespace NUMINAMATH_CALUDE_ellipse_area_ratio_range_l1368_136804

/-- An ellipse with given properties --/
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  passesThrough : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- A line intersecting the ellipse --/
structure IntersectingLine where
  passingThrough : ℝ × ℝ
  intersectionPoints : (ℝ × ℝ) × (ℝ × ℝ)

/-- The ratio of triangle areas --/
def areaRatio (e : Ellipse) (l : IntersectingLine) : ℝ := sorry

theorem ellipse_area_ratio_range 
  (e : Ellipse) 
  (l : IntersectingLine) 
  (h1 : e.foci = ((-Real.sqrt 3, 0), (Real.sqrt 3, 0)))
  (h2 : e.passesThrough = (1, Real.sqrt 3 / 2))
  (h3 : e.equation = fun x y ↦ x^2 / 4 + y^2 = 1)
  (h4 : l.passingThrough = (0, 2))
  (h5 : ∃ (M N : ℝ × ℝ), l.intersectionPoints = (M, N) ∧ 
        e.equation M.1 M.2 ∧ e.equation N.1 N.2 ∧ 
        (∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = (t * l.passingThrough.1 + (1 - t) * N.1, 
                                       t * l.passingThrough.2 + (1 - t) * N.2))) :
  1/3 < areaRatio e l ∧ areaRatio e l < 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_area_ratio_range_l1368_136804


namespace NUMINAMATH_CALUDE_marble_arrangement_theorem_l1368_136811

/-- The number of blue marbles -/
def blue_marbles : ℕ := 7

/-- The maximum number of yellow marbles that can be arranged with the blue marbles
    such that the number of marbles with same-color right neighbors equals
    the number with different-color right neighbors -/
def max_yellow_marbles : ℕ := 19

/-- The total number of marbles -/
def total_marbles : ℕ := blue_marbles + max_yellow_marbles

/-- The number of ways to arrange the marbles satisfying the condition -/
def arrangement_count : ℕ := Nat.choose (max_yellow_marbles + blue_marbles + 1) blue_marbles

theorem marble_arrangement_theorem :
  arrangement_count % 1000 = 970 := by
  sorry

end NUMINAMATH_CALUDE_marble_arrangement_theorem_l1368_136811


namespace NUMINAMATH_CALUDE_range_of_a_l1368_136831

theorem range_of_a (p q : Prop) 
  (hp : p ↔ ∀ x : ℝ, x > 0 → x + 1/x > a)
  (hq : q ↔ ∃ x₀ : ℝ, x₀^2 - 2*a*x₀ + 1 ≤ 0)
  (hnq : ¬¬q)
  (hpq : ¬(p ∧ q)) :
  a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1368_136831


namespace NUMINAMATH_CALUDE_fraction_non_negative_l1368_136815

theorem fraction_non_negative (x : ℝ) : (x + 7) / (x^2 + 2*x + 8) ≥ 0 ↔ x ≥ -7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_non_negative_l1368_136815


namespace NUMINAMATH_CALUDE_distance_difference_around_block_l1368_136821

/-- The difference in distance run around a square block -/
theorem distance_difference_around_block (block_side_length street_width : ℝ) :
  block_side_length = 500 →
  street_width = 25 →
  (4 * (block_side_length + 2 * street_width)) - (4 * block_side_length) = 200 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_around_block_l1368_136821


namespace NUMINAMATH_CALUDE_proportional_function_value_l1368_136877

/-- A function f is proportional if it can be written as f(x) = kx for some constant k -/
def IsProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function f(x) = (m-2)x + m^2 - 4 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x + m^2 - 4

theorem proportional_function_value :
  ∀ m : ℝ, IsProportional (f m) → f m (-2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_value_l1368_136877


namespace NUMINAMATH_CALUDE_fraction_proportion_l1368_136870

theorem fraction_proportion (x y : ℚ) (h : y ≠ 0) :
  (x / y) / (2 / 5) = (3 / 7) / (6 / 5) → x / y = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_proportion_l1368_136870


namespace NUMINAMATH_CALUDE_implication_q_not_p_l1368_136864

theorem implication_q_not_p (x : ℝ) : x^2 - x - 2 > 0 → x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_implication_q_not_p_l1368_136864


namespace NUMINAMATH_CALUDE_fraction_denominator_problem_l1368_136835

theorem fraction_denominator_problem (n d : ℤ) : 
  d = n - 4 ∧ n + 6 = 3 * d → d = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_problem_l1368_136835


namespace NUMINAMATH_CALUDE_sugar_measurement_l1368_136819

theorem sugar_measurement (required_sugar : Rat) (cup_capacity : Rat) (fills : Nat) : 
  required_sugar = 15/4 ∧ cup_capacity = 1/3 → fills = 12 := by
  sorry

end NUMINAMATH_CALUDE_sugar_measurement_l1368_136819


namespace NUMINAMATH_CALUDE_sam_money_value_l1368_136891

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The number of pennies Sam has -/
def num_pennies : ℕ := 9

/-- The number of quarters Sam has -/
def num_quarters : ℕ := 7

/-- The total value of Sam's money in dollars -/
def total_value : ℚ := num_pennies * penny_value + num_quarters * quarter_value

theorem sam_money_value : total_value = 184 / 100 := by sorry

end NUMINAMATH_CALUDE_sam_money_value_l1368_136891


namespace NUMINAMATH_CALUDE_evaluate_expression_l1368_136883

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) :
  y * (y - 2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1368_136883


namespace NUMINAMATH_CALUDE_race_percentage_l1368_136830

theorem race_percentage (v_Q : ℝ) (h : v_Q > 0) : 
  let v_P := v_Q * (1 + 25/100)
  (300 / v_P = (300 - 60) / v_Q) → 
  ∃ (p : ℝ), v_P = v_Q * (1 + p/100) ∧ p = 25 :=
by sorry

end NUMINAMATH_CALUDE_race_percentage_l1368_136830


namespace NUMINAMATH_CALUDE_local_minimum_condition_l1368_136841

/-- The function f(x) = x(x - m)² has a local minimum at x = 2 if and only if m = 6 -/
theorem local_minimum_condition (m : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), x * (x - m)^2 ≥ 2 * (2 - m)^2) ↔ m = 6 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_condition_l1368_136841


namespace NUMINAMATH_CALUDE_triangle_and_function_properties_l1368_136858

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m and n that are parallel, prove the angle B and properties of function f. -/
theorem triangle_and_function_properties
  (a b c : ℝ)
  (A B C : ℝ)
  (m : ℝ × ℝ)
  (n : ℝ × ℝ)
  (ω : ℝ)
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_m : m = (b, 2*a - c))
  (h_n : n = (Real.cos B, Real.cos C))
  (h_parallel : ∃ (k : ℝ), m = k • n)
  (h_ω : ω > 0)
  (f : ℝ → ℝ)
  (h_f : f = λ x => Real.cos (ω * x - π/6) + Real.sin (ω * x))
  (h_period : ∀ x, f (x + π) = f x) :
  (B = π/3) ∧
  (∃ x₀ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x ≤ f x₀ ∧ f x₀ = Real.sqrt 3) ∧
  (∃ x₁ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x₁ ≤ f x ∧ f x₁ = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_and_function_properties_l1368_136858
