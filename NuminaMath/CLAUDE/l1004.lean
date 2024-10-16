import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_l1004_100464

theorem quadratic_roots_reciprocal (b : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - b*x₁ + 1 = 0 ∧ x₂^2 - b*x₂ + 1 = 0 →
  (x₂ = 1 / x₁ ∨ (b = 2 ∧ x₁ = 1 ∧ x₂ = 1) ∨ (b = -2 ∧ x₁ = -1 ∧ x₂ = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_l1004_100464


namespace NUMINAMATH_CALUDE_equation_real_roots_a_range_l1004_100415

theorem equation_real_roots_a_range :
  ∀ a : ℝ, (∃ x : ℝ, 2 - 2^(-|x-2|) = 2 + a) → -1 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_real_roots_a_range_l1004_100415


namespace NUMINAMATH_CALUDE_election_result_proof_l1004_100448

theorem election_result_proof (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 5500 →
  candidate_percentage = 35 / 100 →
  (total_votes : ℚ) * candidate_percentage - (total_votes : ℚ) * (1 - candidate_percentage) = -1650 := by
  sorry

end NUMINAMATH_CALUDE_election_result_proof_l1004_100448


namespace NUMINAMATH_CALUDE_fuwa_selection_theorem_l1004_100421

/-- The number of types of "Chinese Fuwa" mascots -/
def num_types : ℕ := 5

/-- The total number of Fuwa mascots -/
def total_fuwa : ℕ := 10

/-- The number of Fuwa to be selected -/
def select_num : ℕ := 5

/-- The number of ways to select Fuwa mascots -/
def ways_to_select : ℕ := 160

/-- Theorem stating the number of ways to select Fuwa mascots -/
theorem fuwa_selection_theorem :
  (num_types = 5) →
  (total_fuwa = 10) →
  (select_num = 5) →
  (ways_to_select = 
    2 * (Nat.choose num_types 1) * (2^(num_types - 1))) :=
by sorry

end NUMINAMATH_CALUDE_fuwa_selection_theorem_l1004_100421


namespace NUMINAMATH_CALUDE_cubic_roots_determinant_l1004_100460

/-- Given a cubic equation x^3 - px^2 + qx - r = 0 with roots a, b, c,
    the determinant of the matrix
    |a 0 1|
    |0 b 1|
    |1 1 c|
    is equal to r - a - b -/
theorem cubic_roots_determinant (p q r a b c : ℝ) : 
  a^3 - p*a^2 + q*a - r = 0 →
  b^3 - p*b^2 + q*b - r = 0 →
  c^3 - p*c^2 + q*c - r = 0 →
  Matrix.det !![a, 0, 1; 0, b, 1; 1, 1, c] = r - a - b :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_determinant_l1004_100460


namespace NUMINAMATH_CALUDE_union_of_A_and_B_when_m_is_3_necessary_but_not_sufficient_condition_l1004_100440

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 2*m + 1}

-- Part 1
theorem union_of_A_and_B_when_m_is_3 :
  A ∪ B 3 = {x : ℝ | -1 ≤ x ∧ x < 7} := by sorry

-- Part 2
theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m) ↔ m ≤ -2 ∨ (0 ≤ m ∧ m ≤ 1) := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_when_m_is_3_necessary_but_not_sufficient_condition_l1004_100440


namespace NUMINAMATH_CALUDE_angle_measure_with_special_supplement_complement_l1004_100411

theorem angle_measure_with_special_supplement_complement : 
  ∀ x : ℝ, 
    (0 < x) ∧ (x < 90) →
    (180 - x = 7 * (90 - x)) → 
    x = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_with_special_supplement_complement_l1004_100411


namespace NUMINAMATH_CALUDE_part_one_part_two_l1004_100424

-- Define the conditions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 6*x + 8 ≤ 0

-- Part 1
theorem part_one : 
  ∀ x : ℝ, (a = -3 ∧ a < 0 ∧ p x a ∧ q x) → -4 ≤ x ∧ x < -3 :=
sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, a < 0 → 
  ((∀ x : ℝ, p x a ↔ q x) ↔ (-2 < a ∧ a < -4/3)) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1004_100424


namespace NUMINAMATH_CALUDE_f_max_and_g_dominance_l1004_100410

open Real

noncomputable def f (x : ℝ) : ℝ := log x - 2 * x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * x^2 + (m - 3) * x - 1

theorem f_max_and_g_dominance :
  (∃ (c : ℝ), c = -log 2 - 1 ∧ ∀ x > 0, f x ≤ c) ∧
  (∀ m : ℤ, (∀ x > 0, f x ≤ g m x) → m ≥ 2) ∧
  (∀ x > 0, f x ≤ g 2 x) :=
sorry

end NUMINAMATH_CALUDE_f_max_and_g_dominance_l1004_100410


namespace NUMINAMATH_CALUDE_correct_change_l1004_100481

/-- Calculates the change received when purchasing frames -/
def change_received (num_frames : ℕ) (frame_cost : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_frames * frame_cost)

/-- Proves that the change received is correct for the given problem -/
theorem correct_change : change_received 3 3 20 = 11 := by
  sorry

end NUMINAMATH_CALUDE_correct_change_l1004_100481


namespace NUMINAMATH_CALUDE_third_term_value_l1004_100407

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the specific sequence with given conditions
def specific_sequence (a : ℕ → ℝ) : Prop :=
  geometric_sequence a ∧ a 1 = -2 ∧ a 5 = -8

-- Theorem statement
theorem third_term_value (a : ℕ → ℝ) (h : specific_sequence a) : a 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_third_term_value_l1004_100407


namespace NUMINAMATH_CALUDE_expression_simplification_l1004_100423

theorem expression_simplification (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1004_100423


namespace NUMINAMATH_CALUDE_craig_apples_l1004_100496

/-- Theorem: If Craig shares 7 apples and has 13 apples left after sharing,
    then Craig initially had 20 apples. -/
theorem craig_apples (initial : ℕ) (shared : ℕ) (remaining : ℕ)
    (h1 : shared = 7)
    (h2 : remaining = 13)
    (h3 : initial = shared + remaining) :
  initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_craig_apples_l1004_100496


namespace NUMINAMATH_CALUDE_angle_A_value_min_side_a_value_l1004_100406

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom angle_side_relation : (2 * c - b) * Real.cos A = a * Real.cos B
axiom triangle_area : (1 / 2) * b * c * Real.sin A = 2 * Real.sqrt 3

-- Theorem statements
theorem angle_A_value : A = π / 3 := by sorry

theorem min_side_a_value : ∃ (a_min : ℝ), a_min = 2 * Real.sqrt 2 ∧ ∀ (a : ℝ), a ≥ a_min := by sorry

end NUMINAMATH_CALUDE_angle_A_value_min_side_a_value_l1004_100406


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l1004_100483

theorem lcm_gcf_ratio : 
  (Nat.lcm 180 594) / (Nat.gcd 180 594) = 330 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l1004_100483


namespace NUMINAMATH_CALUDE_first_group_size_is_eight_l1004_100456

/-- The number of men in the first group -/
def first_group_size : ℕ := 8

/-- The number of hours worked per day -/
def hours_per_day : ℕ := 8

/-- The number of days the first group works -/
def days_first_group : ℕ := 24

/-- The number of men in the second group -/
def second_group_size : ℕ := 12

/-- The number of days the second group works -/
def days_second_group : ℕ := 16

theorem first_group_size_is_eight :
  first_group_size * hours_per_day * days_first_group =
  second_group_size * hours_per_day * days_second_group :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_is_eight_l1004_100456


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1004_100446

theorem yellow_balls_count (total : ℕ) (white : ℕ) (green : ℕ) (red : ℕ) (purple : ℕ) (prob : ℚ) :
  total = 100 →
  white = 50 →
  green = 20 →
  red = 17 →
  purple = 3 →
  prob = 4/5 →
  (white + green + (total - white - green - red - purple) : ℚ) / total = prob →
  total - white - green - red - purple = 10 :=
by sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1004_100446


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l1004_100489

theorem wrapping_paper_fraction (total_used : ℚ) (num_presents : ℕ) (fraction_per_present : ℚ) :
  total_used = 1/2 →
  num_presents = 5 →
  total_used = fraction_per_present * num_presents →
  fraction_per_present = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l1004_100489


namespace NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l1004_100437

theorem probability_neither_red_nor_purple (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 60) (h2 : red = 5) (h3 : purple = 7) :
  (total - (red + purple)) / total = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l1004_100437


namespace NUMINAMATH_CALUDE_train_length_l1004_100468

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  speed = 72 * 1000 / 3600 → 
  platform_length = 300 → 
  crossing_time = 26 → 
  speed * crossing_time - platform_length = 220 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1004_100468


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_pair_l1004_100470

theorem sum_of_reciprocal_pair (a b : ℝ) : 
  a > 0 → b > 0 → a * b = 1 → (3 * a + 2 * b) * (3 * b + 2 * a) = 295 → a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_pair_l1004_100470


namespace NUMINAMATH_CALUDE_remainder_polynomial_division_l1004_100405

theorem remainder_polynomial_division (x : ℝ) : 
  let g (x : ℝ) := x^5 + x^4 + x^3 + x^2 + x + 1
  (g (x^12)) % (g x) = 6 := by sorry

end NUMINAMATH_CALUDE_remainder_polynomial_division_l1004_100405


namespace NUMINAMATH_CALUDE_segment_ae_length_segment_ae_length_value_l1004_100442

/-- Given points on a coordinate grid, prove the length of segment AE --/
theorem segment_ae_length :
  let A : ℝ × ℝ := (0, 4)
  let B : ℝ × ℝ := (8, 0)
  let C : ℝ × ℝ := (6, 3)
  let D : ℝ × ℝ := (3, 0)
  let E := (14/3, 2/3)  -- Intersection point of AB and CD
  (A.1 - E.1)^2 + (A.2 - E.2)^2 = 296/9 :=
by sorry

/-- The length of segment AE is √(296)/3 --/
theorem segment_ae_length_value :
  let A : ℝ × ℝ := (0, 4)
  let B : ℝ × ℝ := (8, 0)
  let C : ℝ × ℝ := (6, 3)
  let D : ℝ × ℝ := (3, 0)
  let E := (14/3, 2/3)  -- Intersection point of AB and CD
  Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2) = Real.sqrt 296 / 3 :=
by sorry

end NUMINAMATH_CALUDE_segment_ae_length_segment_ae_length_value_l1004_100442


namespace NUMINAMATH_CALUDE_whole_number_between_l1004_100485

theorem whole_number_between : 
  ∃ (M : ℕ), (8 : ℚ) < (M : ℚ) / 4 ∧ (M : ℚ) / 4 < 9 → M = 33 :=
by sorry

end NUMINAMATH_CALUDE_whole_number_between_l1004_100485


namespace NUMINAMATH_CALUDE_roots_on_circle_l1004_100443

theorem roots_on_circle (a : ℝ) : 
  (∃ (z₁ z₂ z₃ z₄ : ℂ), z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    (z₁^2 - 2*z₁ + 5)*(z₁^2 + 2*a*z₁ + 1) = 0 ∧
    (z₂^2 - 2*z₂ + 5)*(z₂^2 + 2*a*z₂ + 1) = 0 ∧
    (z₃^2 - 2*z₃ + 5)*(z₃^2 + 2*a*z₃ + 1) = 0 ∧
    (z₄^2 - 2*z₄ + 5)*(z₄^2 + 2*a*z₄ + 1) = 0 ∧
    ∃ (c : ℂ) (r : ℝ), r > 0 ∧ 
      Complex.abs (z₁ - c) = r ∧
      Complex.abs (z₂ - c) = r ∧
      Complex.abs (z₃ - c) = r ∧
      Complex.abs (z₄ - c) = r) ↔
  (a > -1 ∧ a < 1) ∨ a = -3 :=
by sorry

end NUMINAMATH_CALUDE_roots_on_circle_l1004_100443


namespace NUMINAMATH_CALUDE_average_daily_low_temperature_l1004_100433

theorem average_daily_low_temperature (temperatures : List ℝ) : 
  temperatures = [40, 47, 45, 41, 39] → 
  (temperatures.sum / temperatures.length : ℝ) = 42.4 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_low_temperature_l1004_100433


namespace NUMINAMATH_CALUDE_sequence_problem_l1004_100462

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a_sum : a 1 + a 5 + a 9 = 9)
  (h_b_prod : b 2 * b 5 * b 8 = 3 * Real.sqrt 3) :
  (a 2 + a 8) / (1 + b 2 * b 8) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1004_100462


namespace NUMINAMATH_CALUDE_ana_salary_calculation_l1004_100469

/-- Calculates the final salary after a raise, pay cut, and bonus -/
def final_salary (initial_salary : ℝ) (raise_percent : ℝ) (cut_percent : ℝ) (bonus : ℝ) : ℝ :=
  initial_salary * (1 + raise_percent) * (1 - cut_percent) + bonus

theorem ana_salary_calculation :
  final_salary 2500 0.25 0.25 200 = 2543.75 := by
  sorry

end NUMINAMATH_CALUDE_ana_salary_calculation_l1004_100469


namespace NUMINAMATH_CALUDE_fraction_simplification_l1004_100428

theorem fraction_simplification :
  (270 : ℚ) / 18 * 7 / 140 * 9 / 4 = 27 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1004_100428


namespace NUMINAMATH_CALUDE_remainder_theorem_l1004_100422

theorem remainder_theorem (P E M S F N T : ℕ) 
  (h1 : P = E * M + S) 
  (h2 : M = N * F + T) : 
  P % (E * F + 1) = E * T + S - N :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1004_100422


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l1004_100494

theorem binomial_coefficient_sum (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ : ℝ) :
  (∀ x : ℝ, (1 + x)^14 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + 
    a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12 + a₁₃*x^13 + a₁₄*x^14) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ + 7*a₇ + 8*a₈ + 9*a₉ + 10*a₁₀ + 
    11*a₁₁ + 12*a₁₂ + 13*a₁₃ + 14*a₁₄ = 7 * 2^14 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l1004_100494


namespace NUMINAMATH_CALUDE_constant_term_of_product_l1004_100431

/-- The constant term in the expansion of (x^6 + x^2 + 3)(x^4 + x^3 + 20) is 60 -/
theorem constant_term_of_product (x : ℝ) : 
  (x^6 + x^2 + 3) * (x^4 + x^3 + 20) = x^10 + x^9 + 20*x^6 + x^7 + x^6 + 20*x^2 + 3*x^4 + 3*x^3 + 60 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_of_product_l1004_100431


namespace NUMINAMATH_CALUDE_range_of_f_l1004_100404

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a = 0 ∧ b = 4 ∧
  (∀ y, (∃ x, x ∈ [0, 3] ∧ f x = y) ↔ y ∈ [a, b]) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1004_100404


namespace NUMINAMATH_CALUDE_monge_point_properties_l1004_100461

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron with vertices A, B, C, and D -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- The Monge point of a tetrahedron -/
def mongePoint (t : Tetrahedron) : Point3D := sorry

/-- Checks if a point lies on a plane defined by three other points -/
def isOnPlane (p q r s : Point3D) : Prop := sorry

/-- The projection of a point onto a plane -/
def projection (p : Point3D) (plane : Point3D × Point3D × Point3D) : Point3D := sorry

/-- The intersection point of the altitudes of a triangular face -/
def altitudeIntersection (face : Point3D × Point3D × Point3D) : Point3D := sorry

/-- The center of the circumscribed circle of a triangular face -/
def circumcenter (face : Point3D × Point3D × Point3D) : Point3D := sorry

/-- Checks if four points are coplanar -/
def areCoplanar (p q r s : Point3D) : Prop := sorry

theorem monge_point_properties (t : Tetrahedron) : 
  isOnPlane (mongePoint t) t.A t.B t.C →
  let D1 := projection t.D (t.A, t.B, t.C)
  (areCoplanar t.D 
    (altitudeIntersection (t.D, t.A, t.B))
    (altitudeIntersection (t.D, t.B, t.C))
    (altitudeIntersection (t.D, t.A, t.C))) ∧
  (areCoplanar t.D
    (circumcenter (t.D, t.A, t.B))
    (circumcenter (t.D, t.B, t.C))
    (circumcenter (t.D, t.A, t.C))) := by
  sorry

end NUMINAMATH_CALUDE_monge_point_properties_l1004_100461


namespace NUMINAMATH_CALUDE_video_archive_space_theorem_l1004_100409

/-- Represents the number of days in the video archive -/
def days : ℕ := 15

/-- Represents the total disk space used by the archive in megabytes -/
def total_space : ℕ := 30000

/-- Calculates the total number of minutes in a given number of days -/
def total_minutes (d : ℕ) : ℕ := d * 24 * 60

/-- Calculates the average disk space per minute of video -/
def avg_space_per_minute : ℚ :=
  total_space / total_minutes days

theorem video_archive_space_theorem :
  abs (avg_space_per_minute - 1.388) < 0.001 :=
sorry

end NUMINAMATH_CALUDE_video_archive_space_theorem_l1004_100409


namespace NUMINAMATH_CALUDE_pure_imaginary_real_part_zero_l1004_100458

/-- A complex number z is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_real_part_zero (a : ℝ) :
  isPureImaginary (Complex.mk a 1) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_real_part_zero_l1004_100458


namespace NUMINAMATH_CALUDE_quadratic_equation_rewrite_l1004_100471

theorem quadratic_equation_rewrite :
  ∀ x : ℝ, (-5 * x^2 = 2 * x + 10) ↔ (x^2 + (2/5) * x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_rewrite_l1004_100471


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_l1004_100476

theorem square_area_from_rectangle (r l b : ℝ) : 
  l = r / 4 →  -- length of rectangle is 1/4 of circle radius
  l * b = 35 → -- area of rectangle is 35
  b = 5 →      -- breadth of rectangle is 5
  r^2 = 784 := by sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_l1004_100476


namespace NUMINAMATH_CALUDE_triangle_area_l1004_100452

theorem triangle_area (a b c : ℝ) (ha : a = 10) (hb : b = 24) (hc : c = 26) :
  (1 / 2) * a * b = 120 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1004_100452


namespace NUMINAMATH_CALUDE_max_quotient_value_l1004_100441

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 250) (hb : 700 ≤ b ∧ b ≤ 1400) :
  (∀ x y, 100 ≤ x ∧ x ≤ 250 → 700 ≤ y ∧ y ≤ 1400 → y / x ≤ b / a) →
  b / a = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l1004_100441


namespace NUMINAMATH_CALUDE_root_difference_quadratic_specific_quadratic_root_difference_l1004_100490

theorem root_difference_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  2*a*root1^2 + b*root1 = c ∧
  2*a*root2^2 + b*root2 = c ∧
  root1 ≥ root2 →
  root1 - root2 = Real.sqrt discriminant / a :=
by sorry

theorem specific_quadratic_root_difference :
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := 12
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  root1 - root2 = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_specific_quadratic_root_difference_l1004_100490


namespace NUMINAMATH_CALUDE_library_book_count_l1004_100408

/-- Represents the library with its bookshelves and books. -/
structure Library where
  num_bookshelves : Nat
  floors_per_bookshelf : Nat
  left_position : Nat
  right_position : Nat

/-- Calculates the total number of books in the library. -/
def total_books (lib : Library) : Nat :=
  let books_per_floor := lib.left_position + lib.right_position - 1
  let books_per_bookshelf := books_per_floor * lib.floors_per_bookshelf
  books_per_bookshelf * lib.num_bookshelves

/-- Theorem stating the total number of books in the library. -/
theorem library_book_count :
  ∀ (lib : Library),
    lib.num_bookshelves = 28 →
    lib.floors_per_bookshelf = 6 →
    lib.left_position = 9 →
    lib.right_position = 11 →
    total_books lib = 3192 := by
  sorry

#eval total_books ⟨28, 6, 9, 11⟩

end NUMINAMATH_CALUDE_library_book_count_l1004_100408


namespace NUMINAMATH_CALUDE_current_calculation_l1004_100497

theorem current_calculation (Q R t I : ℝ) 
  (heat_eq : Q = I^2 * R * t)
  (resistance : R = 8)
  (heat_generated : Q = 72)
  (time : t = 2) :
  I = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_current_calculation_l1004_100497


namespace NUMINAMATH_CALUDE_dog_food_cost_l1004_100467

def initial_amount : ℕ := 167
def meat_cost : ℕ := 17
def chicken_cost : ℕ := 22
def veggie_cost : ℕ := 43
def egg_cost : ℕ := 5
def remaining_amount : ℕ := 35

theorem dog_food_cost : 
  initial_amount - (meat_cost + chicken_cost + veggie_cost + egg_cost + remaining_amount) = 45 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_cost_l1004_100467


namespace NUMINAMATH_CALUDE_water_transfer_height_l1004_100455

/-- Represents a rectangular tank with given dimensions -/
structure Tank where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of water in a tank given the water height -/
def waterVolume (t : Tank) (waterHeight : ℝ) : ℝ :=
  t.length * t.width * waterHeight

/-- Calculates the base area of a tank -/
def baseArea (t : Tank) : ℝ :=
  t.length * t.width

/-- Represents the problem setup -/
structure ProblemSetup where
  tankA : Tank
  tankB : Tank
  waterHeightB : ℝ

/-- The main theorem to prove -/
theorem water_transfer_height (setup : ProblemSetup) 
  (h1 : setup.tankA = { length := 4, width := 3, height := 5 })
  (h2 : setup.tankB = { length := 4, width := 2, height := 8 })
  (h3 : setup.waterHeightB = 1.5) :
  (waterVolume setup.tankB setup.waterHeightB) / (baseArea setup.tankA) = 1 := by
  sorry

end NUMINAMATH_CALUDE_water_transfer_height_l1004_100455


namespace NUMINAMATH_CALUDE_janice_age_proof_l1004_100465

/-- Calculates a person's age given their birth year and the current year -/
def calculate_age (birth_year : ℕ) (current_year : ℕ) : ℕ :=
  current_year - birth_year

theorem janice_age_proof (current_year : ℕ) (mark_birth_year : ℕ) :
  current_year = 2021 →
  mark_birth_year = 1976 →
  let mark_age := calculate_age mark_birth_year current_year
  let graham_age := mark_age - 3
  let janice_age := graham_age / 2
  janice_age = 21 := by sorry

end NUMINAMATH_CALUDE_janice_age_proof_l1004_100465


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1004_100484

def set_A : Set ℝ := {x | Real.cos x = 0}
def set_B : Set ℝ := {x | x^2 - 5*x ≤ 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {Real.pi/2, 3*Real.pi/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1004_100484


namespace NUMINAMATH_CALUDE_base_equivalence_l1004_100479

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The theorem stating the equivalence of the numbers in different bases --/
theorem base_equivalence (k : Nat) :
  toBase10 [5, 2, 4] 8 = toBase10 [6, 6, 4] k → k = 7 ∧ toBase10 [6, 6, 4] 7 = toBase10 [5, 2, 4] 8 := by
  sorry

end NUMINAMATH_CALUDE_base_equivalence_l1004_100479


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_312_l1004_100472

/-- The sum of the digits in the binary representation of 312 is 3 -/
theorem sum_of_binary_digits_312 : 
  (Nat.digits 2 312).sum = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_312_l1004_100472


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1004_100432

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 ∨ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1004_100432


namespace NUMINAMATH_CALUDE_min_T_minus_S_and_max_T_l1004_100478

/-- Given non-negative real numbers a, b, and c, S and T are defined as follows:
    S = a + 2b + 3c
    T = a + b^2 + c^3 -/
def S (a b c : ℝ) : ℝ := a + 2*b + 3*c
def T (a b c : ℝ) : ℝ := a + b^2 + c^3

theorem min_T_minus_S_and_max_T (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (∀ a' b' c' : ℝ, 0 ≤ a' → 0 ≤ b' → 0 ≤ c' → -3 ≤ T a' b' c' - S a' b' c') ∧
  (S a b c = 4 → T a b c ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_min_T_minus_S_and_max_T_l1004_100478


namespace NUMINAMATH_CALUDE_triangle_special_cosine_identity_l1004_100466

theorem triangle_special_cosine_identity (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  Real.sin A = Real.cos B ∧ 
  Real.sin A = Real.tan C → 
  Real.cos A ^ 3 + Real.cos A ^ 2 - Real.cos A = 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_cosine_identity_l1004_100466


namespace NUMINAMATH_CALUDE_range_of_m_l1004_100492

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -Real.sqrt (9 - p.1^2)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2}

-- Define the point A
def A (m : ℝ) : ℝ × ℝ := (0, m)

-- Define the vector from A to P
def AP (m : ℝ) (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - 0, P.2 - m)

-- Define the vector from A to Q
def AQ (m : ℝ) (Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - 0, Q.2 - m)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∃ P ∈ C, ∃ Q ∈ l, AP m P + AQ m Q = (0, 0)) → m ∈ Set.Icc (-1/2) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1004_100492


namespace NUMINAMATH_CALUDE_difference_of_squares_262_258_l1004_100493

theorem difference_of_squares_262_258 : 262^2 - 258^2 = 2080 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_262_258_l1004_100493


namespace NUMINAMATH_CALUDE_max_table_sum_l1004_100417

def numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def is_valid_partition (l1 l2 : List ℕ) : Prop :=
  l1.length = 4 ∧ l2.length = 3 ∧ (l1 ++ l2).toFinset = numbers.toFinset

def table_sum (l1 l2 : List ℕ) : ℕ := (l1.sum * l2.sum)

theorem max_table_sum :
  ∃ (l1 l2 : List ℕ), is_valid_partition l1 l2 ∧
    (∀ (m1 m2 : List ℕ), is_valid_partition m1 m2 →
      table_sum m1 m2 ≤ table_sum l1 l2) ∧
    table_sum l1 l2 = 841 := by sorry

end NUMINAMATH_CALUDE_max_table_sum_l1004_100417


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1004_100463

-- Define sets A and B
def A : Set ℝ := {x | x * (x - 2) < 3}
def B : Set ℝ := {x | 5 / (x + 1) ≥ 1}

-- Theorem statement
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1004_100463


namespace NUMINAMATH_CALUDE_meeting_percentage_is_35_percent_l1004_100419

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 35

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 2 * first_meeting_minutes

/-- Represents the duration of the third meeting in minutes -/
def third_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Represents the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes + third_meeting_minutes

/-- Represents the percentage of the work day spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100

theorem meeting_percentage_is_35_percent : meeting_percentage = 35 := by
  sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_35_percent_l1004_100419


namespace NUMINAMATH_CALUDE_fraction_inequality_l1004_100486

theorem fraction_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1004_100486


namespace NUMINAMATH_CALUDE_range_of_a_l1004_100402

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x + a ≥ 0) ↔ a ∈ Set.Ici (-8) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1004_100402


namespace NUMINAMATH_CALUDE_representable_integers_l1004_100425

theorem representable_integers (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2004) : 
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ k = (m * n + 1) / (m + n) := by
  sorry

end NUMINAMATH_CALUDE_representable_integers_l1004_100425


namespace NUMINAMATH_CALUDE_three_number_sum_l1004_100426

theorem three_number_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  b = 7 →
  (a + b + c) / 3 = a + 15 →
  (a + b + c) / 3 = c - 20 →
  a + b + c = 36 := by
sorry

end NUMINAMATH_CALUDE_three_number_sum_l1004_100426


namespace NUMINAMATH_CALUDE_regression_line_intercept_l1004_100450

/-- Prove that a regression line with slope 1.23 passing through (4, 5) has y-intercept 0.08 -/
theorem regression_line_intercept (slope : ℝ) (center_x center_y : ℝ) (y_intercept : ℝ) : 
  slope = 1.23 → center_x = 4 → center_y = 5 → 
  center_y = slope * center_x + y_intercept →
  y_intercept = 0.08 := by
sorry

end NUMINAMATH_CALUDE_regression_line_intercept_l1004_100450


namespace NUMINAMATH_CALUDE_smallest_difference_sides_l1004_100439

theorem smallest_difference_sides (DE EF FD : ℕ) : 
  (DE < EF ∧ EF ≤ FD) →
  (DE + EF + FD = 1801) →
  (DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF) →
  (∀ DE' EF' FD' : ℕ, 
    (DE' < EF' ∧ EF' ≤ FD') →
    (DE' + EF' + FD' = 1801) →
    (DE' + EF' > FD' ∧ EF' + FD' > DE' ∧ FD' + DE' > EF') →
    EF' - DE' ≥ EF - DE) →
  EF - DE = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_difference_sides_l1004_100439


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l1004_100403

def original_earnings : ℝ := 30
def new_earnings : ℝ := 40

theorem percentage_increase_proof :
  (new_earnings - original_earnings) / original_earnings * 100 =
  (40 - 30) / 30 * 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l1004_100403


namespace NUMINAMATH_CALUDE_rearrangements_count_l1004_100457

def word : String := "Alejandro"
def subwords : List String := ["ned", "den"]

theorem rearrangements_count : 
  (List.length word.data - 2) * (Nat.factorial (List.length word.data - 2) / 2) * (List.length subwords) = 40320 := by
  sorry

end NUMINAMATH_CALUDE_rearrangements_count_l1004_100457


namespace NUMINAMATH_CALUDE_inequality_implication_l1004_100475

theorem inequality_implication (p q r : ℝ) 
  (hr : r > 0) (hpq : p * q ≠ 0) (hineq : p * r < q * r) : 
  1 < q / p :=
sorry

end NUMINAMATH_CALUDE_inequality_implication_l1004_100475


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1004_100416

/-- A linear function passing through (-4, 3) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 5

/-- The point (-4, 3) lies on the graph of f -/
def point_condition (a : ℝ) : Prop := f a (-4) = 3

/-- The equation ax - 5 = 3 -/
def equation (a x : ℝ) : Prop := a * x - 5 = 3

theorem linear_equation_solution (a : ℝ) (h : point_condition a) :
  ∃ x, equation a x ∧ x = -4 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1004_100416


namespace NUMINAMATH_CALUDE_sin_double_sum_eq_four_sin_product_l1004_100454

/-- Given that α + β + γ = π, prove that sin 2α + sin 2β + sin 2γ = 4 sin α sin β sin γ -/
theorem sin_double_sum_eq_four_sin_product (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 4 * Real.sin α * Real.sin β * Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_sin_double_sum_eq_four_sin_product_l1004_100454


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l1004_100435

-- Define the original function g(x)
def g (x : ℝ) : ℝ := x^2 - 4

-- Define the reflected function h(x)
def h (x : ℝ) : ℝ := -x^2 + 4

-- Theorem stating that h(x) is the reflection of g(x) across the x-axis
theorem reflection_across_x_axis :
  ∀ x : ℝ, h x = -(g x) :=
by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l1004_100435


namespace NUMINAMATH_CALUDE_product_digit_sum_is_nine_l1004_100488

/-- Represents a strictly increasing sequence of 5 digits -/
def StrictlyIncreasingDigits (a b c d e : Nat) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved -/
theorem product_digit_sum_is_nine 
  (a b c d e : Nat) 
  (h : StrictlyIncreasingDigits a b c d e) : 
  sumOfDigits (9 * (a * 10000 + b * 1000 + c * 100 + d * 10 + e)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_is_nine_l1004_100488


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l1004_100491

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 9*x^2 > -27*x ↔ (0 < x ∧ x < 3) ∨ (6 < x) :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l1004_100491


namespace NUMINAMATH_CALUDE_initial_water_percentage_l1004_100430

/-- 
Given a mixture of 180 liters, if adding 12 liters of water results in a new mixture 
where water is 25% of the total, then the initial percentage of water in the mixture was 20%.
-/
theorem initial_water_percentage (initial_volume : ℝ) (added_water : ℝ) (final_water_percentage : ℝ) :
  initial_volume = 180 →
  added_water = 12 →
  final_water_percentage = 25 →
  (initial_volume * (20 / 100) + added_water) / (initial_volume + added_water) = final_water_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l1004_100430


namespace NUMINAMATH_CALUDE_smallest_multiple_with_last_four_digits_l1004_100418

theorem smallest_multiple_with_last_four_digits (n : ℕ) : 
  (n % 10000 = 2020) → (n % 77 = 0) → (∀ m : ℕ, m < n → (m % 10000 ≠ 2020 ∨ m % 77 ≠ 0)) → n = 722020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_last_four_digits_l1004_100418


namespace NUMINAMATH_CALUDE_fifteenths_in_fraction_l1004_100434

theorem fifteenths_in_fraction : 
  let whole_number : ℚ := 82
  let fraction : ℚ := 3 / 5
  let divisor : ℚ := 1 / 15
  let multiplier : ℕ := 3
  let subtrahend_whole : ℕ := 42
  let subtrahend_fraction : ℚ := 7 / 10
  
  ((whole_number + fraction) / divisor * multiplier) - 
  (subtrahend_whole + subtrahend_fraction) = 3674.3 := by sorry

end NUMINAMATH_CALUDE_fifteenths_in_fraction_l1004_100434


namespace NUMINAMATH_CALUDE_student_line_arrangements_l1004_100413

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def arrangements_with_two_together (n : ℕ) : ℕ := (n - 1).factorial * 2

theorem student_line_arrangements :
  let total_students : ℕ := 5
  let total_arrangements := number_of_arrangements total_students
  let arrangements_with_specific_two_together := arrangements_with_two_together total_students
  total_arrangements - arrangements_with_specific_two_together = 72 := by
sorry

end NUMINAMATH_CALUDE_student_line_arrangements_l1004_100413


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_l1004_100474

/-- The probability of selecting at least one woman when choosing 3 people
    at random from a group of 8 men and 4 women is 41/55. -/
theorem prob_at_least_one_woman (n_men : ℕ) (n_women : ℕ) (n_select : ℕ) :
  n_men = 8 → n_women = 4 → n_select = 3 →
  (1 : ℚ) - (Nat.choose n_men n_select : ℚ) / (Nat.choose (n_men + n_women) n_select : ℚ) = 41 / 55 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_woman_l1004_100474


namespace NUMINAMATH_CALUDE_existence_of_pairs_l1004_100412

theorem existence_of_pairs : ∃ (f : Fin 2018 → ℕ × ℕ),
  (∀ i : Fin 2018, (f i).1 ≠ (f i).2) ∧
  (∀ i : Fin 2017, (f i.succ).1 = (f i).1 + 1) ∧
  (∀ i : Fin 2017, (f i.succ).2 = (f i).2 + 1) ∧
  (∀ i : Fin 2018, (f i).1 % (f i).2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_pairs_l1004_100412


namespace NUMINAMATH_CALUDE_age_ratio_l1004_100495

/-- Given that Billy is 4 years old and you were 12 years older than Billy when he was born,
    prove that the ratio of your current age to Billy's current age is 4:1. -/
theorem age_ratio (billy_age : ℕ) (age_difference : ℕ) : 
  billy_age = 4 → age_difference = 12 → (age_difference + billy_age) / billy_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l1004_100495


namespace NUMINAMATH_CALUDE_point_inside_circle_l1004_100487

/-- A point is inside a circle if its distance from the center is less than the radius -/
def is_inside_circle (center_distance radius : ℝ) : Prop :=
  center_distance < radius

/-- Given a circle with radius 5 and a point A at distance 4 from the center,
    prove that point A is inside the circle -/
theorem point_inside_circle (center_distance radius : ℝ)
  (h1 : center_distance = 4)
  (h2 : radius = 5) :
  is_inside_circle center_distance radius := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l1004_100487


namespace NUMINAMATH_CALUDE_kindWizardCanAchieveGoal_l1004_100427

/-- Represents a gnome -/
structure Gnome where
  id : Nat

/-- Represents a friendship between two gnomes -/
structure Friendship where
  gnome1 : Gnome
  gnome2 : Gnome

/-- Represents a round table with gnomes -/
structure RoundTable where
  gnomes : List Gnome

/-- The kind wizard's action of making gnomes friends -/
def makeGnomesFriends (pairs : List (Gnome × Gnome)) : List Friendship :=
  sorry

/-- The evil wizard's action of making gnomes unfriends -/
def makeGnomesUnfriends (friendships : List Friendship) (n : Nat) : List Friendship :=
  sorry

/-- Check if a seating arrangement is valid (all adjacent gnomes are friends) -/
def isValidSeating (seating : List Gnome) (friendships : List Friendship) : Prop :=
  sorry

theorem kindWizardCanAchieveGoal (n : Nat) (hn : n > 1 ∧ Odd n) :
  ∃ (table1 table2 : RoundTable),
    table1.gnomes.length = n ∧
    table2.gnomes.length = n ∧
    (∀ (pairs : List (Gnome × Gnome)),
      pairs.length = 2 * n →
      ∀ (evilAction : List Friendship → List Friendship),
        ∃ (finalSeating : List Gnome),
          finalSeating.length = 2 * n ∧
          isValidSeating finalSeating (evilAction (makeGnomesFriends pairs))) :=
by sorry

end NUMINAMATH_CALUDE_kindWizardCanAchieveGoal_l1004_100427


namespace NUMINAMATH_CALUDE_system_solution_l1004_100445

theorem system_solution : ∃ (x y z : ℚ), 
  (7 * x + 3 * y = z - 10) ∧ 
  (2 * x - 4 * y = 3 * z + 20) := by
  use 0, -50/13, -20/13
  sorry

end NUMINAMATH_CALUDE_system_solution_l1004_100445


namespace NUMINAMATH_CALUDE_leftmost_box_value_l1004_100444

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i, i < n - 2 → a i + a (i + 1) + a (i + 2) = 2005

theorem leftmost_box_value (a : ℕ → ℕ) :
  sequence_sum a 9 →
  a 1 = 888 →
  a 2 = 999 →
  a 0 = 118 :=
by sorry

end NUMINAMATH_CALUDE_leftmost_box_value_l1004_100444


namespace NUMINAMATH_CALUDE_line_y_intercept_l1004_100429

/-- Proves that for a line ax + y + 2 = 0 with an inclination angle of 3π/4, the y-intercept is -2 -/
theorem line_y_intercept (a : ℝ) : 
  (∀ x y : ℝ, a * x + y + 2 = 0) → 
  (Real.tan (3 * Real.pi / 4) = -a) → 
  (∃ x : ℝ, 0 * x + (-2) + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l1004_100429


namespace NUMINAMATH_CALUDE_gala_luncheon_croissant_cost_l1004_100453

/-- Calculates the cost of croissants for a gala luncheon --/
theorem gala_luncheon_croissant_cost
  (people : ℕ)
  (sandwiches_per_person : ℕ)
  (croissants_per_set : ℕ)
  (cost_per_set : ℚ)
  (h1 : people = 24)
  (h2 : sandwiches_per_person = 2)
  (h3 : croissants_per_set = 12)
  (h4 : cost_per_set = 8) :
  (people * sandwiches_per_person / croissants_per_set : ℚ) * cost_per_set = 32 := by
  sorry

#check gala_luncheon_croissant_cost

end NUMINAMATH_CALUDE_gala_luncheon_croissant_cost_l1004_100453


namespace NUMINAMATH_CALUDE_number_comparisons_l1004_100473

-- Define the numbers
def a : ℚ := -7/2
def b : ℚ := -7/3
def c : ℚ := -3/4
def d : ℚ := -4/5

-- Theorem to prove the comparisons
theorem number_comparisons :
  (a < b) ∧ (c > d) := by sorry

end NUMINAMATH_CALUDE_number_comparisons_l1004_100473


namespace NUMINAMATH_CALUDE_solution_value_l1004_100438

theorem solution_value (x m : ℤ) : 
  x = -2 → 3 * x + 5 = x - m → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1004_100438


namespace NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l1004_100436

theorem sqrt_plus_reciprocal_inequality (x : ℝ) (h : x > 0) : Real.sqrt x + 1 / Real.sqrt x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l1004_100436


namespace NUMINAMATH_CALUDE_average_weight_of_class_l1004_100477

theorem average_weight_of_class (num_male num_female : ℕ) 
                                (avg_weight_male avg_weight_female : ℚ) :
  num_male = 20 →
  num_female = 20 →
  avg_weight_male = 42 →
  avg_weight_female = 38 →
  (num_male * avg_weight_male + num_female * avg_weight_female) / (num_male + num_female) = 40 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_class_l1004_100477


namespace NUMINAMATH_CALUDE_ellipse_sum_a_k_l1004_100480

-- Define the ellipse
def Ellipse (f₁ f₂ p : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2)
  let d₂ := Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2)
  let c := Real.sqrt ((f₂.1 - f₁.1)^2 + (f₂.2 - f₁.2)^2) / 2
  let a := (d₁ + d₂) / 2
  let b := Real.sqrt (a^2 - c^2)
  let h := (f₁.1 + f₂.1) / 2
  let k := (f₁.2 + f₂.2) / 2
  ∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem ellipse_sum_a_k :
  let f₁ : ℝ × ℝ := (2, 1)
  let f₂ : ℝ × ℝ := (2, 5)
  let p : ℝ × ℝ := (-3, 3)
  Ellipse f₁ f₂ p →
  let a := (Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
            Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2)) / 2
  let k := (f₁.2 + f₂.2) / 2
  a + k = Real.sqrt 29 + 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_a_k_l1004_100480


namespace NUMINAMATH_CALUDE_closest_point_l1004_100401

/-- The curve y = 3 - x^2 for x > 0 -/
def curve (x : ℝ) : ℝ := 3 - x^2

/-- The fixed point P(0, 2) -/
def P : ℝ × ℝ := (0, 2)

/-- A point Q on the curve -/
def Q (x : ℝ) : ℝ × ℝ := (x, curve x)

/-- The squared distance between P and Q -/
def distance_squared (x : ℝ) : ℝ := (x - P.1)^2 + (curve x - P.2)^2

/-- The theorem stating that (√2/2, 5/2) is the closest point to P on the curve -/
theorem closest_point :
  ∃ (x : ℝ), x > 0 ∧ 
  ∀ (y : ℝ), y > 0 → distance_squared x ≤ distance_squared y ∧
  Q x = (Real.sqrt 2 / 2, 5 / 2) :=
sorry

end NUMINAMATH_CALUDE_closest_point_l1004_100401


namespace NUMINAMATH_CALUDE_robin_additional_cupcakes_l1004_100414

/-- Calculates the number of additional cupcakes made given the initial number,
    the number sold, and the final number of cupcakes. -/
def additional_cupcakes (initial : ℕ) (sold : ℕ) (final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Proves that Robin made 39 additional cupcakes given the problem conditions. -/
theorem robin_additional_cupcakes :
  additional_cupcakes 42 22 59 = 39 := by
  sorry

end NUMINAMATH_CALUDE_robin_additional_cupcakes_l1004_100414


namespace NUMINAMATH_CALUDE_password_decryption_probability_l1004_100482

theorem password_decryption_probability 
  (p1 : ℝ) (p2 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/4) 
  (h3 : 0 ≤ p1 ∧ p1 ≤ 1) (h4 : 0 ≤ p2 ∧ p2 ≤ 1) : 
  1 - (1 - p1) * (1 - p2) = 0.4 := by
sorry

end NUMINAMATH_CALUDE_password_decryption_probability_l1004_100482


namespace NUMINAMATH_CALUDE_original_to_circle_l1004_100400

/-- The original curve in polar coordinates -/
def original_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * (Real.cos θ)^2 + 4 * (Real.sin θ)^2)

/-- The transformation applied to the curve -/
def transformation (x y x'' y'' : ℝ) : Prop :=
  x'' = (1/2) * x ∧ y'' = (Real.sqrt 3 / 3) * y

/-- The resulting curve after transformation -/
def resulting_curve (x'' y'' : ℝ) : Prop :=
  x''^2 + y''^2 = 1

/-- Theorem stating that the original curve transforms into a circle -/
theorem original_to_circle :
  ∀ (ρ θ x y x'' y'' : ℝ),
    original_curve ρ θ →
    x = ρ * Real.cos θ →
    y = ρ * Real.sin θ →
    transformation x y x'' y'' →
    resulting_curve x'' y'' :=
sorry

end NUMINAMATH_CALUDE_original_to_circle_l1004_100400


namespace NUMINAMATH_CALUDE_net_amount_calculation_l1004_100499

/-- Calculates the net amount received after selling a stock and deducting brokerage -/
def net_amount_after_brokerage (sale_amount : ℚ) (brokerage_rate : ℚ) : ℚ :=
  sale_amount - (sale_amount * brokerage_rate)

/-- Theorem stating that the net amount received after selling a stock for Rs. 108.25 
    with a 1/4% brokerage rate is Rs. 107.98 -/
theorem net_amount_calculation :
  let sale_amount : ℚ := 108.25
  let brokerage_rate : ℚ := 1 / 400  -- 1/4% expressed as a fraction
  net_amount_after_brokerage sale_amount brokerage_rate = 107.98 := by
  sorry

#eval net_amount_after_brokerage 108.25 (1 / 400)

end NUMINAMATH_CALUDE_net_amount_calculation_l1004_100499


namespace NUMINAMATH_CALUDE_inequality_proof_l1004_100449

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1004_100449


namespace NUMINAMATH_CALUDE_cartesian_to_polar_equivalence_l1004_100498

theorem cartesian_to_polar_equivalence :
  ∀ (x y ρ θ : ℝ),
  x = 2 ∧ y = -2 →
  ρ = Real.sqrt (x^2 + y^2) →
  θ = Real.arctan (y / x) →
  (ρ = 2 * Real.sqrt 2 ∧ θ = -π/4) := by
  sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_equivalence_l1004_100498


namespace NUMINAMATH_CALUDE_sqrt_x_over_5_increase_l1004_100420

theorem sqrt_x_over_5_increase (x : ℝ) (hx : x > 0) :
  let x_new := x * 1.69
  let original := Real.sqrt (x / 5)
  let new_value := Real.sqrt (x_new / 5)
  (new_value - original) / original * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_over_5_increase_l1004_100420


namespace NUMINAMATH_CALUDE_school_gender_difference_l1004_100447

theorem school_gender_difference 
  (initial_girls : ℕ) 
  (initial_boys : ℕ) 
  (additional_girls : ℕ) 
  (h1 : initial_girls = 632)
  (h2 : initial_boys = 410)
  (h3 : additional_girls = 465) :
  initial_girls + additional_girls - initial_boys = 687 := by
  sorry

end NUMINAMATH_CALUDE_school_gender_difference_l1004_100447


namespace NUMINAMATH_CALUDE_divisors_of_squared_number_l1004_100451

theorem divisors_of_squared_number (n : ℕ) (h : n > 1) :
  (Finset.card (Nat.divisors n) = 4) → (Finset.card (Nat.divisors (n^2)) = 9) := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_squared_number_l1004_100451


namespace NUMINAMATH_CALUDE_fraction_is_integer_l1004_100459

theorem fraction_is_integer (b t : ℤ) (h : b ≠ 1) :
  ∃ k : ℤ, (t^5 - 5*b + 4) / (b^2 - 2*b + 1) = k :=
sorry

end NUMINAMATH_CALUDE_fraction_is_integer_l1004_100459
