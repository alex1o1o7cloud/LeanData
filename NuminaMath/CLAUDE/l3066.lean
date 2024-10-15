import Mathlib

namespace NUMINAMATH_CALUDE_power_of_fraction_l3066_306652

theorem power_of_fraction : (5 / 3 : ℚ) ^ 3 = 125 / 27 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_l3066_306652


namespace NUMINAMATH_CALUDE_partition_existence_l3066_306699

theorem partition_existence (p q : ℕ+) (h_coprime : Nat.Coprime p q) (h_neq : p ≠ q) :
  (∃ (A B C : Set ℕ+),
    (∀ z : ℕ+, (z ∈ A ∧ z + p ∈ B ∧ z + q ∈ C) ∨
               (z ∈ B ∧ z + p ∈ C ∧ z + q ∈ A) ∨
               (z ∈ C ∧ z + p ∈ A ∧ z + q ∈ B)) ∧
    (A ∪ B ∪ C = Set.univ) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅)) ↔
  (3 ∣ p + q) :=
sorry

end NUMINAMATH_CALUDE_partition_existence_l3066_306699


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3066_306648

theorem solution_satisfies_system : ∃ (a b c : ℝ), 
  (a^3 + 3*a*b^2 + 3*a*c^2 - 6*a*b*c = 1) ∧
  (b^3 + 3*b*a^2 + 3*b*c^2 - 6*a*b*c = 1) ∧
  (c^3 + 3*c*a^2 + 3*c*b^2 - 6*a*b*c = 1) ∧
  (a = 1 ∧ b = 1 ∧ c = 1) := by
sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3066_306648


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3066_306622

/-- A quadratic equation x^2 - 2kx + k^2 + k + 1 = 0 with two real roots -/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  x^2 - 2*k*x + k^2 + k + 1 = 0

/-- The equation has two real roots -/
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ ∧ quadratic_equation k x₂

theorem quadratic_equation_properties :
  (∀ k : ℝ, has_two_real_roots k → k ≤ -1) ∧
  (∀ k : ℝ, has_two_real_roots k → (∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ x₁^2 + x₂^2 = 10) → k = -2) ∧
  (∀ k : ℝ, has_two_real_roots k → (∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ |x₁| + |x₂| = 2) → k = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3066_306622


namespace NUMINAMATH_CALUDE_line_through_origin_and_negative_one_l3066_306696

/-- The angle of inclination (in degrees) of a line passing through two points -/
def angleOfInclination (x1 y1 x2 y2 : ℝ) : ℝ := sorry

/-- A line passes through the origin (0, 0) and the point (-1, -1) -/
theorem line_through_origin_and_negative_one : 
  angleOfInclination 0 0 (-1) (-1) = 45 := by sorry

end NUMINAMATH_CALUDE_line_through_origin_and_negative_one_l3066_306696


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_plus_one_l3066_306658

theorem sum_of_reciprocals_plus_one (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → (b^3 - b + 1 = 0) → (c^3 - c + 1 = 0) → 
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_plus_one_l3066_306658


namespace NUMINAMATH_CALUDE_cos_is_even_l3066_306603

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem cos_is_even : IsEven Real.cos := by
  sorry

end NUMINAMATH_CALUDE_cos_is_even_l3066_306603


namespace NUMINAMATH_CALUDE_train_speed_l3066_306612

/-- The speed of a train given its length and time to pass an observer -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) :
  train_length = 100 →
  passing_time = 12 →
  (train_length / 1000) / (passing_time / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3066_306612


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3066_306629

theorem smallest_number_with_remainders : ∃ b : ℕ, 
  b > 0 ∧
  b % 5 = 3 ∧
  b % 4 = 2 ∧
  b % 6 = 2 ∧
  (∀ c : ℕ, c > 0 ∧ c % 5 = 3 ∧ c % 4 = 2 ∧ c % 6 = 2 → b ≤ c) ∧
  b = 38 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3066_306629


namespace NUMINAMATH_CALUDE_base_representation_equivalence_l3066_306662

/-- Represents a positive integer in different bases -/
structure BaseRepresentation where
  base8 : ℕ  -- Representation in base 8
  base5 : ℕ  -- Representation in base 5
  base10 : ℕ -- Representation in base 10
  is_valid : base8 ≥ 10 ∧ base8 < 100 ∧ base5 ≥ 10 ∧ base5 < 100

/-- Converts a two-digit number in base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ :=
  8 * (n / 10) + (n % 10)

/-- Converts a two-digit number in base 5 to base 10 -/
def base5_to_base10 (n : ℕ) : ℕ :=
  5 * (n % 10) + (n / 10)

/-- Theorem stating the equivalence of the representations -/
theorem base_representation_equivalence (n : BaseRepresentation) : 
  base8_to_base10 n.base8 = base5_to_base10 n.base5 ∧ 
  base8_to_base10 n.base8 = n.base10 ∧ 
  n.base10 = 39 := by
  sorry

#check base_representation_equivalence

end NUMINAMATH_CALUDE_base_representation_equivalence_l3066_306662


namespace NUMINAMATH_CALUDE_min_shift_for_monotonic_decrease_l3066_306618

open Real

theorem min_shift_for_monotonic_decrease (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = sin (2*x + 2*m + π/6)) →
  (∀ x ∈ [-π/12, 5*π/12], ∀ y ∈ [-π/12, 5*π/12], x < y → f x > f y) →
  m > 0 →
  m ≥ π/4 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_for_monotonic_decrease_l3066_306618


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l3066_306623

theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  (R^2 - r^2 = 20) →
  ∃ c : ℝ, c > 0 ∧ c^2 / 4 + r^2 = R^2 ∧ c = 4 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l3066_306623


namespace NUMINAMATH_CALUDE_slope_of_line_l3066_306611

theorem slope_of_line (x y : ℝ) :
  3 * x + 4 * y + 12 = 0 → (y - 0) / (x - 0) = -3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l3066_306611


namespace NUMINAMATH_CALUDE_surface_area_increase_l3066_306633

/-- Given a cube with edge length a that is cut into 27 identical smaller cubes,
    the increase in surface area is 12a². -/
theorem surface_area_increase (a : ℝ) (h : a > 0) : 
  27 * 6 * (a / 3)^2 - 6 * a^2 = 12 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_increase_l3066_306633


namespace NUMINAMATH_CALUDE_max_planes_four_points_l3066_306686

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in three-dimensional space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Function to determine the number of planes formed by four points -/
def numPlanes (p1 p2 p3 p4 : Point3D) : ℕ := sorry

/-- Theorem stating the maximum number of planes determined by four points -/
theorem max_planes_four_points :
  ∃ (p1 p2 p3 p4 : Point3D), numPlanes p1 p2 p3 p4 = 4 ∧
  ∀ (q1 q2 q3 q4 : Point3D), numPlanes q1 q2 q3 q4 ≤ 4 := by sorry

end NUMINAMATH_CALUDE_max_planes_four_points_l3066_306686


namespace NUMINAMATH_CALUDE_function_and_range_proof_l3066_306640

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B (f : ℝ → ℝ) : Set ℝ := {x | 1 < f x ∧ f x < 3}

-- Define the theorem
theorem function_and_range_proof 
  (a b : ℝ) 
  (h_a_nonzero : a ≠ 0)
  (h_f : ∀ x, f x = a * x + b)
  (h_f_condition : ∀ x, f (2 * x + 1) = 4 * x + 1)
  (h_subset : B f ⊆ A a) :
  (∀ x, f x = 2 * x - 1) ∧ (1/2 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_function_and_range_proof_l3066_306640


namespace NUMINAMATH_CALUDE_smallest_multiple_l3066_306602

theorem smallest_multiple (n : ℕ) (h : n = 5) : 
  (∃ m : ℕ, m * n - 15 > 2 * n ∧ ∀ k : ℕ, k < m → k * n - 15 ≤ 2 * n) → 
  (∃ m : ℕ, m * n - 15 > 2 * n ∧ ∀ k : ℕ, k < m → k * n - 15 ≤ 2 * n ∧ m = 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3066_306602


namespace NUMINAMATH_CALUDE_set_intersection_example_l3066_306669

theorem set_intersection_example : 
  let A : Set ℕ := {1, 3, 9}
  let B : Set ℕ := {1, 5, 9}
  A ∩ B = {1, 9} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l3066_306669


namespace NUMINAMATH_CALUDE_quiz_show_probability_l3066_306620

-- Define the number of questions and choices
def num_questions : ℕ := 4
def num_choices : ℕ := 4

-- Define the minimum number of correct answers needed to win
def min_correct : ℕ := 3

-- Define the probability of guessing a single question correctly
def prob_correct : ℚ := 1 / num_choices

-- Define the probability of guessing a single question incorrectly
def prob_incorrect : ℚ := 1 - prob_correct

-- Define the function to calculate the probability of winning
def prob_win : ℚ :=
  (num_questions.choose min_correct) * (prob_correct ^ min_correct) * (prob_incorrect ^ (num_questions - min_correct)) +
  (prob_correct ^ num_questions)

-- State the theorem
theorem quiz_show_probability :
  prob_win = 13 / 256 := by sorry

end NUMINAMATH_CALUDE_quiz_show_probability_l3066_306620


namespace NUMINAMATH_CALUDE_work_completion_l3066_306654

theorem work_completion (x : ℕ) : 
  (x * 40 = (x - 5) * 60) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l3066_306654


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l3066_306621

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Theorem for part (1)
theorem intersection_A_complement_B (m : ℝ) (h : m = 3) :
  A ∩ (Set.univ \ B m) = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem for part (2)
theorem intersection_A_B_empty (m : ℝ) :
  A ∩ B m = ∅ ↔ m ≤ -2 := by sorry

-- Theorem for part (3)
theorem intersection_A_B_equals_A (m : ℝ) :
  A ∩ B m = A ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l3066_306621


namespace NUMINAMATH_CALUDE_negative_operation_l3066_306698

theorem negative_operation (a b c d : ℤ) : a = (-7) * (-6) ∧ b = (-7) - (-15) ∧ c = 0 * (-2) * (-3) ∧ d = (-6) + (-4) → d < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_operation_l3066_306698


namespace NUMINAMATH_CALUDE_badminton_cost_equality_l3066_306660

/-- Represents the cost calculation for two stores selling badminton equipment -/
theorem badminton_cost_equality (x : ℝ) : x ≥ 5 → (125 + 5*x = 135 + 4.5*x ↔ x = 20) :=
by
  sorry

#check badminton_cost_equality

end NUMINAMATH_CALUDE_badminton_cost_equality_l3066_306660


namespace NUMINAMATH_CALUDE_m_range_when_only_one_proposition_true_l3066_306679

def proposition_p (m : ℝ) : Prop := 0 < m ∧ m < 1/3

def proposition_q (m : ℝ) : Prop := 0 < m ∧ m < 15

theorem m_range_when_only_one_proposition_true :
  ∀ m : ℝ, (proposition_p m ∨ proposition_q m) ∧ ¬(proposition_p m ∧ proposition_q m) →
  1/3 ≤ m ∧ m < 15 :=
sorry

end NUMINAMATH_CALUDE_m_range_when_only_one_proposition_true_l3066_306679


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l3066_306627

/-- 
Given two varieties of rice mixed in a specific ratio to create a mixture with a known cost,
this theorem proves the cost of the first variety of rice.
-/
theorem rice_mixture_cost 
  (cost_second : ℝ) 
  (cost_mixture : ℝ) 
  (mix_ratio : ℝ) 
  (h1 : cost_second = 8.75)
  (h2 : cost_mixture = 7.50)
  (h3 : mix_ratio = 0.625)
  : ∃ (cost_first : ℝ), cost_first = 8.28125 := by
  sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l3066_306627


namespace NUMINAMATH_CALUDE_jennys_money_l3066_306639

theorem jennys_money (initial_money : ℚ) : 
  (initial_money * (1 - 3/7) = 24) → 
  (initial_money / 2 = 21) := by
sorry

end NUMINAMATH_CALUDE_jennys_money_l3066_306639


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l3066_306616

-- Define the universe set U
def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | 3 < x ∧ x ≤ 7}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 3 < x ∧ x < 5} :=
sorry

-- Theorem for the union of complement of A and B
theorem union_complement_A_B : (U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 < x ∧ x ≤ 7)} :=
sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l3066_306616


namespace NUMINAMATH_CALUDE_rachelle_gpa_probability_l3066_306675

def grade_points (grade : Char) : ℕ :=
  match grade with
  | 'A' => 5
  | 'B' => 4
  | 'C' => 3
  | 'D' => 2
  | _ => 0

def gpa (total_points : ℕ) : ℚ := total_points / 5

def english_prob (grade : Char) : ℚ :=
  match grade with
  | 'A' => 1 / 7
  | 'B' => 1 / 5
  | 'C' => 1 - 1 / 7 - 1 / 5
  | _ => 0

def history_prob (grade : Char) : ℚ :=
  match grade with
  | 'B' => 1 / 3
  | 'C' => 1 / 6
  | 'D' => 1 - 1 / 3 - 1 / 6
  | _ => 0

theorem rachelle_gpa_probability :
  let assured_points := 3 * grade_points 'A'
  let min_total_points := 20
  let required_points := min_total_points - assured_points
  let prob_a_english := english_prob 'A'
  let prob_b_english := english_prob 'B'
  let prob_b_history := history_prob 'B'
  let prob_c_history := history_prob 'C'
  (prob_a_english + prob_b_english * prob_b_history + prob_b_english * prob_c_history) = 17 / 70 := by
  sorry

end NUMINAMATH_CALUDE_rachelle_gpa_probability_l3066_306675


namespace NUMINAMATH_CALUDE_elective_course_schemes_l3066_306649

theorem elective_course_schemes (n : ℕ) (k : ℕ) : n = 4 ∧ k = 2 → Nat.choose n k = 6 := by
  sorry

end NUMINAMATH_CALUDE_elective_course_schemes_l3066_306649


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3066_306607

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, ax - b > 0 ↔ x > 1) →
  (∀ x, (a * x + b) * (x - 3) > 0 ↔ x < -1 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3066_306607


namespace NUMINAMATH_CALUDE_train_speed_problem_l3066_306688

/-- Proves that given two trains of equal length 70 meters, where one train travels at 50 km/hr
    and passes the other train in 36 seconds, the speed of the slower train is 36 km/hr. -/
theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 70 →
  faster_speed = 50 →
  passing_time = 36 →
  ∃ slower_speed : ℝ,
    slower_speed > 0 ∧
    slower_speed < faster_speed ∧
    train_length * 2 = (faster_speed - slower_speed) * passing_time * (1000 / 3600) ∧
    slower_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3066_306688


namespace NUMINAMATH_CALUDE_person_age_puzzle_l3066_306676

theorem person_age_puzzle (x : ℤ) : 3 * (x + 5) - 3 * (x - 5) = x → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_person_age_puzzle_l3066_306676


namespace NUMINAMATH_CALUDE_f_composition_sqrt2_l3066_306601

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 3 * x + 1 else |x|

theorem f_composition_sqrt2 :
  f (f (-Real.sqrt 2)) = 3 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_sqrt2_l3066_306601


namespace NUMINAMATH_CALUDE_max_distance_sum_l3066_306671

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the point M
def M : ℝ × ℝ := (6, 4)

-- Statement of the theorem
theorem max_distance_sum (a b : ℝ) (F₁ : ℝ × ℝ) :
  ∃ (max : ℝ), ∀ (P : ℝ × ℝ),
    ellipse a b P.1 P.2 →
    dist P M + dist P F₁ ≤ max ∧
    (∃ (Q : ℝ × ℝ), ellipse a b Q.1 Q.2 ∧ dist Q M + dist Q F₁ = max) ∧
    max = 15 :=
sorry

end NUMINAMATH_CALUDE_max_distance_sum_l3066_306671


namespace NUMINAMATH_CALUDE_penny_difference_l3066_306609

theorem penny_difference (kate_pennies john_pennies : ℕ) 
  (h1 : kate_pennies = 223) 
  (h2 : john_pennies = 388) : 
  john_pennies - kate_pennies = 165 := by
sorry

end NUMINAMATH_CALUDE_penny_difference_l3066_306609


namespace NUMINAMATH_CALUDE_second_pedal_triangle_rotation_l3066_306677

/-- Represents a triangle with angles in degrees -/
structure Triangle where
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  sum_180 : angle_a + angle_b + angle_c = 180

/-- Computes the angles of the first pedal triangle -/
def first_pedal_triangle (t : Triangle) : Triangle :=
  { angle_a := 2 * t.angle_a,
    angle_b := 2 * t.angle_b,
    angle_c := 2 * t.angle_c - 180,
    sum_180 := by sorry }

/-- Computes the angles of the second pedal triangle -/
def second_pedal_triangle (t : Triangle) : Triangle :=
  let pt := first_pedal_triangle t
  { angle_a := 180 - 2 * pt.angle_a,
    angle_b := 180 - 2 * pt.angle_b,
    angle_c := 180 - 2 * pt.angle_c,
    sum_180 := by sorry }

/-- Computes the rotation angle between two triangles -/
def rotation_angle (t1 t2 : Triangle) : ℝ :=
  (180 - t1.angle_c) + t2.angle_b

/-- Theorem statement -/
theorem second_pedal_triangle_rotation (t : Triangle)
  (h1 : t.angle_a = 12)
  (h2 : t.angle_b = 36)
  (h3 : t.angle_c = 132) :
  rotation_angle t (second_pedal_triangle t) = 120 := by sorry

end NUMINAMATH_CALUDE_second_pedal_triangle_rotation_l3066_306677


namespace NUMINAMATH_CALUDE_sum_first_three_terms_l3066_306672

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_first_three_terms
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_fifth : a 5 = 7)
  (h_sixth : a 6 = 12)
  (h_seventh : a 7 = 17) :
  a 1 + a 2 + a 3 = -24 :=
sorry

end NUMINAMATH_CALUDE_sum_first_three_terms_l3066_306672


namespace NUMINAMATH_CALUDE_crease_lines_form_ellipse_l3066_306695

/-- Given a circle with radius R and an interior point A at distance a from the center,
    this theorem states that the set of points on all crease lines formed by folding
    the circle so that a point on the circumference coincides with A is described by
    the equation of an ellipse. -/
theorem crease_lines_form_ellipse (R a : ℝ) (h : 0 < a ∧ a < R) :
  ∀ x y : ℝ, (x - a / 2)^2 / (R / 2)^2 + y^2 / ((R / 2)^2 - (a / 2)^2) = 1 ↔ 
  (∃ A' : ℝ × ℝ, (A'.1^2 + A'.2^2 = R^2) ∧ 
   ((x - a)^2 + y^2 = (x - A'.1)^2 + (y - A'.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_crease_lines_form_ellipse_l3066_306695


namespace NUMINAMATH_CALUDE_trig_identity_l3066_306617

theorem trig_identity (x y : ℝ) :
  Real.cos x ^ 2 + Real.cos (x + y) ^ 2 - 2 * Real.cos x * Real.cos y * Real.cos (x + y) = Real.sin y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3066_306617


namespace NUMINAMATH_CALUDE_quadratic_solution_l3066_306604

theorem quadratic_solution (b : ℚ) : 
  ((-8 : ℚ)^2 + b * (-8) - 15 = 0) → b = 49/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3066_306604


namespace NUMINAMATH_CALUDE_opposite_of_2022_l3066_306685

-- Define the opposite of an integer
def opposite (n : ℤ) : ℤ := -n

-- Theorem stating that the opposite of 2022 is -2022
theorem opposite_of_2022 : opposite 2022 = -2022 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2022_l3066_306685


namespace NUMINAMATH_CALUDE_quadratic_trinomial_factorization_l3066_306681

theorem quadratic_trinomial_factorization 
  (a b c x x₁ x₂ : ℝ) 
  (ha : a ≠ 0) 
  (hx₁ : a * x₁^2 + b * x₁ + c = 0) 
  (hx₂ : a * x₂^2 + b * x₂ + c = 0) : 
  a * x^2 + b * x + c = a * (x - x₁) * (x - x₂) := by
sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_factorization_l3066_306681


namespace NUMINAMATH_CALUDE_propositions_analysis_l3066_306644

-- Proposition 1
def has_real_roots (q : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + q = 0

-- Proposition 2
def both_zero (x y : ℝ) : Prop := x = 0 ∧ y = 0

theorem propositions_analysis :
  -- Proposition 1
  (¬ (∀ q : ℝ, has_real_roots q → q < 1)) ∧  -- Converse is false
  (∀ q : ℝ, ¬(has_real_roots q) → q ≥ 1) ∧  -- Contrapositive is true
  -- Proposition 2
  (∀ x y : ℝ, both_zero x y → x^2 + y^2 = 0) ∧  -- Converse is true
  (∀ x y : ℝ, ¬(both_zero x y) → x^2 + y^2 ≠ 0)  -- Contrapositive is true
  := by sorry

end NUMINAMATH_CALUDE_propositions_analysis_l3066_306644


namespace NUMINAMATH_CALUDE_yoongi_initial_money_l3066_306690

/-- The amount of money Yoongi had initially -/
def initial_money : ℕ := 590

/-- The cost of the candy Yoongi bought -/
def candy_cost : ℕ := 250

/-- The amount of pocket money Yoongi received -/
def pocket_money : ℕ := 500

/-- The amount of money Yoongi had left after all transactions -/
def money_left : ℕ := 420

theorem yoongi_initial_money :
  ∃ (pencil_cost : ℕ),
    initial_money = candy_cost + pencil_cost + money_left ∧
    initial_money + pocket_money - candy_cost = 2 * money_left :=
by
  sorry


end NUMINAMATH_CALUDE_yoongi_initial_money_l3066_306690


namespace NUMINAMATH_CALUDE_simplify_fraction_l3066_306643

theorem simplify_fraction : 24 * (8 / 15) * (5 / 18) = 32 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3066_306643


namespace NUMINAMATH_CALUDE_equal_perimeters_shapes_l3066_306635

theorem equal_perimeters_shapes (x y : ℝ) : 
  (4 * (x + 2) = 6 * x) ∧ (6 * x = 2 * Real.pi * y) → x = 4 ∧ y = 12 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_equal_perimeters_shapes_l3066_306635


namespace NUMINAMATH_CALUDE_quadratic_factorability_l3066_306692

theorem quadratic_factorability : ∃ (a b c p q : ℤ),
  (∀ x : ℝ, 3 * (x - 3)^2 = x^2 - 9 ↔ a * x^2 + b * x + c = 0) ∧
  (a * x^2 + b * x + c = (x - p) * (x - q)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_factorability_l3066_306692


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_36_495_l3066_306637

theorem gcd_lcm_sum_36_495 : Nat.gcd 36 495 + Nat.lcm 36 495 = 1989 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_36_495_l3066_306637


namespace NUMINAMATH_CALUDE_student_b_speed_l3066_306646

theorem student_b_speed (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 12 →
  speed_ratio = 1.2 →
  time_difference = 1/6 →
  ∃ (speed_b : ℝ),
    distance / speed_b - time_difference = distance / (speed_ratio * speed_b) ∧
    speed_b = 12 :=
by sorry

end NUMINAMATH_CALUDE_student_b_speed_l3066_306646


namespace NUMINAMATH_CALUDE_vasya_driving_distance_l3066_306665

theorem vasya_driving_distance
  (total_distance : ℝ)
  (anton_distance : ℝ)
  (vasya_distance : ℝ)
  (sasha_distance : ℝ)
  (dima_distance : ℝ)
  (h1 : anton_distance = vasya_distance / 2)
  (h2 : sasha_distance = anton_distance + dima_distance)
  (h3 : dima_distance = total_distance / 10)
  (h4 : anton_distance + vasya_distance + sasha_distance + dima_distance = total_distance) :
  vasya_distance = (2 : ℝ) / 5 * total_distance :=
by sorry

end NUMINAMATH_CALUDE_vasya_driving_distance_l3066_306665


namespace NUMINAMATH_CALUDE_product_of_fractions_l3066_306624

theorem product_of_fractions :
  (3 : ℚ) / 5 * (4 : ℚ) / 7 * (5 : ℚ) / 9 = (4 : ℚ) / 21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3066_306624


namespace NUMINAMATH_CALUDE_fifth_inequality_holds_l3066_306697

theorem fifth_inequality_holds : 
  1 + (1 : ℝ) / 2^2 + 1 / 3^2 + 1 / 4^2 + 1 / 5^2 + 1 / 6^2 < (2 * 5 + 1) / (5 + 1) :=
by sorry

end NUMINAMATH_CALUDE_fifth_inequality_holds_l3066_306697


namespace NUMINAMATH_CALUDE_lesser_solution_quadratic_l3066_306670

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 10*x - 24 = 0 ∧ ∀ y, y^2 + 10*y - 24 = 0 → x ≤ y → x = -12 :=
by sorry

end NUMINAMATH_CALUDE_lesser_solution_quadratic_l3066_306670


namespace NUMINAMATH_CALUDE_maximal_planar_iff_3n_minus_6_edges_l3066_306647

structure PlanarGraph where
  n : ℕ
  e : ℕ
  h_vertices : n ≥ 3

def is_maximal_planar (G : PlanarGraph) : Prop :=
  ∀ H : PlanarGraph, G.n = H.n → G.e ≥ H.e

theorem maximal_planar_iff_3n_minus_6_edges (G : PlanarGraph) :
  is_maximal_planar G ↔ G.e = 3 * G.n - 6 := by sorry

end NUMINAMATH_CALUDE_maximal_planar_iff_3n_minus_6_edges_l3066_306647


namespace NUMINAMATH_CALUDE_intersection_on_y_axis_l3066_306673

/-- Given two lines in the xy-plane defined by equations 2x + 3y - k = 0 and x - ky + 12 = 0,
    if their intersection point lies on the y-axis, then k = 6 or k = -6. -/
theorem intersection_on_y_axis (k : ℝ) : 
  (∃ y : ℝ, 2 * 0 + 3 * y - k = 0 ∧ 0 - k * y + 12 = 0) →
  k = 6 ∨ k = -6 := by
sorry

end NUMINAMATH_CALUDE_intersection_on_y_axis_l3066_306673


namespace NUMINAMATH_CALUDE_company_profit_ratio_l3066_306667

/-- Represents the revenues of a company in a given year -/
structure Revenue where
  amount : ℝ

/-- Calculates the profit given a revenue and a profit percentage -/
def profit (revenue : Revenue) (percentage : ℝ) : ℝ := revenue.amount * percentage

/-- Company N's revenues over three years -/
structure CompanyN where
  revenue2008 : Revenue
  revenue2009 : Revenue
  revenue2010 : Revenue
  revenue2009_eq : revenue2009.amount = 0.8 * revenue2008.amount
  revenue2010_eq : revenue2010.amount = 1.3 * revenue2009.amount

/-- Company M's revenues over three years -/
structure CompanyM where
  revenue : Revenue

theorem company_profit_ratio (n : CompanyN) (m : CompanyM) :
  (profit n.revenue2008 0.08 + profit n.revenue2009 0.15 + profit n.revenue2010 0.10) /
  (profit m.revenue 0.12 + profit m.revenue 0.18 + profit m.revenue 0.14) =
  (0.304 * n.revenue2008.amount) / (0.44 * m.revenue.amount) := by
  sorry

end NUMINAMATH_CALUDE_company_profit_ratio_l3066_306667


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l3066_306619

theorem pie_crust_flour_calculation (initial_crusts : ℕ) (new_crusts : ℕ) (initial_flour : ℚ) :
  initial_crusts = 36 →
  new_crusts = 24 →
  initial_flour = 1/8 →
  (initial_crusts : ℚ) * initial_flour = (new_crusts : ℚ) * ((3:ℚ)/16) :=
by sorry

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l3066_306619


namespace NUMINAMATH_CALUDE_total_jumps_l3066_306656

theorem total_jumps (ronald_jumps : ℕ) (rupert_extra_jumps : ℕ) 
  (h1 : ronald_jumps = 157)
  (h2 : rupert_extra_jumps = 86) : 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 :=
by sorry

end NUMINAMATH_CALUDE_total_jumps_l3066_306656


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l3066_306645

/-- Proves that the ratio of a square pond's area to a rectangular field's area is 1:50 
    given specific dimensions -/
theorem pond_to_field_area_ratio 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (pond_side : ℝ) 
  (h1 : field_length = 80) 
  (h2 : field_width = 40) 
  (h3 : pond_side = 8) : 
  (pond_side ^ 2) / (field_length * field_width) = 1 / 50 := by
  sorry


end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l3066_306645


namespace NUMINAMATH_CALUDE_not_divisible_by_seven_l3066_306610

theorem not_divisible_by_seven (n : ℤ) : ¬(7 ∣ (n^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_seven_l3066_306610


namespace NUMINAMATH_CALUDE_julio_mocktail_days_l3066_306632

/-- The number of days Julio made mocktails given the specified conditions -/
def mocktail_days (lime_juice_per_mocktail : ℚ) (juice_per_lime : ℚ) (limes_per_dollar : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent * limes_per_dollar * juice_per_lime) / lime_juice_per_mocktail

/-- Theorem stating that Julio made mocktails for 30 days under the given conditions -/
theorem julio_mocktail_days :
  mocktail_days 1 2 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_julio_mocktail_days_l3066_306632


namespace NUMINAMATH_CALUDE_prism_properties_l3066_306694

/-- A right triangular prism with rectangular base ABCD and height DE -/
structure RightTriangularPrism where
  AB : ℝ
  BC : ℝ
  DE : ℝ
  ab_eq_dc : AB = 8
  bc_eq : BC = 15
  de_eq : DE = 7

/-- The perimeter of the base ABCD -/
def basePerimeter (p : RightTriangularPrism) : ℝ :=
  2 * (p.AB + p.BC)

/-- The area of the base ABCD -/
def baseArea (p : RightTriangularPrism) : ℝ :=
  p.AB * p.BC

/-- The volume of the right triangular prism -/
def volume (p : RightTriangularPrism) : ℝ :=
  p.AB * p.BC * p.DE

theorem prism_properties (p : RightTriangularPrism) :
  basePerimeter p = 46 ∧ baseArea p = 120 ∧ volume p = 840 := by
  sorry

end NUMINAMATH_CALUDE_prism_properties_l3066_306694


namespace NUMINAMATH_CALUDE_tangent_circles_constant_l3066_306678

/-- Two circles are tangent if the distance between their centers equals the sum of their radii -/
def are_tangent (c1_center : ℝ × ℝ) (c1_radius : ℝ) (c2_center : ℝ × ℝ) (c2_radius : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (c1_radius + c2_radius)^2

/-- The theorem stating the value of 'a' for which the given circles are tangent -/
theorem tangent_circles_constant (a : ℝ) : 
  are_tangent (0, 0) 1 (-4, a) 5 ↔ a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_constant_l3066_306678


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l3066_306600

/-- Given vectors p and q in ℝ², where p is parallel to q, prove that |p + q| = √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) :
  p = (2, -3) →
  q.1 = x ∧ q.2 = 6 →
  (∃ (k : ℝ), q = k • p) →
  ‖p + q‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l3066_306600


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3066_306630

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - 7
  ∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 11 ∧ x2 = 2 - Real.sqrt 11 ∧ f x1 = 0 ∧ f x2 = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3066_306630


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3066_306628

theorem triangle_perimeter (a b c : ℕ) : 
  a = 2 → b = 4 → Even c → 
  a + b > c ∧ b + c > a ∧ c + a > b → 
  a + b + c = 10 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3066_306628


namespace NUMINAMATH_CALUDE_salary_problem_l3066_306634

theorem salary_problem (total : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ) 
  (h1 : total = 6000)
  (h2 : a_spend_rate = 0.95)
  (h3 : b_spend_rate = 0.85)
  (h4 : (1 - a_spend_rate) * a = (1 - b_spend_rate) * (total - a)) :
  a = 4500 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l3066_306634


namespace NUMINAMATH_CALUDE_maryann_client_call_time_l3066_306626

theorem maryann_client_call_time (total_time accounting_time client_time : ℕ) : 
  total_time = 560 →
  accounting_time = 7 * client_time →
  total_time = accounting_time + client_time →
  client_time = 70 := by
sorry

end NUMINAMATH_CALUDE_maryann_client_call_time_l3066_306626


namespace NUMINAMATH_CALUDE_band_percentage_is_twenty_percent_l3066_306661

-- Define the number of students in the band
def students_in_band : ℕ := 168

-- Define the total number of students
def total_students : ℕ := 840

-- Define the percentage of students in the band
def percentage_in_band : ℚ := (students_in_band : ℚ) / total_students * 100

-- Theorem statement
theorem band_percentage_is_twenty_percent :
  percentage_in_band = 20 := by
  sorry

end NUMINAMATH_CALUDE_band_percentage_is_twenty_percent_l3066_306661


namespace NUMINAMATH_CALUDE_log_inequality_iff_x_range_l3066_306608

-- Define the domain constraints
def domain (x : ℝ) : Prop := x > -2 ∧ x ≠ -1

-- Define the logarithmic inequality
def log_inequality (x : ℝ) : Prop :=
  Real.log (8 + x^3) / Real.log (2 + x) ≤ Real.log ((2 + x)^3) / Real.log (2 + x)

-- State the theorem
theorem log_inequality_iff_x_range (x : ℝ) :
  domain x → (log_inequality x ↔ (-2 < x ∧ x < -1) ∨ x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_iff_x_range_l3066_306608


namespace NUMINAMATH_CALUDE_valentines_day_theorem_l3066_306615

theorem valentines_day_theorem (male_students female_students : ℕ) : 
  (male_students * female_students = male_students + female_students + 42) → 
  (male_students * female_students = 88) :=
by
  sorry

end NUMINAMATH_CALUDE_valentines_day_theorem_l3066_306615


namespace NUMINAMATH_CALUDE_two_solutions_l3066_306682

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x * (x - 6) = 7

-- Theorem statement
theorem two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ 
  quadratic_equation a ∧ 
  quadratic_equation b ∧
  ∀ (c : ℝ), quadratic_equation c → (c = a ∨ c = b) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_l3066_306682


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3066_306666

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (h : r ≠ 1) :
  let S := (a * (1 - r^n)) / (1 - r)
  a = -1 → r = -3 → n = 10 → S = 14762 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3066_306666


namespace NUMINAMATH_CALUDE_second_planner_cheaper_l3066_306614

/-- Represents the cost function for an event planner -/
structure EventPlanner where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for an event planner given the number of people -/
def totalCost (planner : EventPlanner) (people : ℕ) : ℕ :=
  planner.basicFee + planner.perPersonFee * people

/-- The first event planner's pricing structure -/
def planner1 : EventPlanner := ⟨120, 18⟩

/-- The second event planner's pricing structure -/
def planner2 : EventPlanner := ⟨250, 15⟩

/-- Theorem stating the conditions for when the second planner becomes less expensive -/
theorem second_planner_cheaper (n : ℕ) :
  (n < 44 → totalCost planner1 n ≤ totalCost planner2 n) ∧
  (n ≥ 44 → totalCost planner2 n < totalCost planner1 n) :=
sorry

end NUMINAMATH_CALUDE_second_planner_cheaper_l3066_306614


namespace NUMINAMATH_CALUDE_delores_purchase_shortfall_l3066_306664

def initial_amount : ℚ := 450
def computer_cost : ℚ := 500
def computer_discount_rate : ℚ := 0.2
def printer_cost : ℚ := 50
def printer_tax_rate : ℚ := 0.2

def computer_discount : ℚ := computer_cost * computer_discount_rate
def discounted_computer_cost : ℚ := computer_cost - computer_discount
def printer_tax : ℚ := printer_cost * printer_tax_rate
def total_printer_cost : ℚ := printer_cost + printer_tax
def total_spent : ℚ := discounted_computer_cost + total_printer_cost

theorem delores_purchase_shortfall :
  initial_amount - total_spent = -10 := by sorry

end NUMINAMATH_CALUDE_delores_purchase_shortfall_l3066_306664


namespace NUMINAMATH_CALUDE_fruit_sales_calculation_l3066_306606

/-- Calculate the total money collected from selling fruits with price increases -/
theorem fruit_sales_calculation (lemon_price grape_price orange_price apple_price : ℚ)
  (lemon_count grape_count orange_count apple_count : ℕ)
  (lemon_increase grape_increase orange_increase apple_increase : ℚ) :
  let new_lemon_price := lemon_price * (1 + lemon_increase)
  let new_grape_price := grape_price * (1 + grape_increase)
  let new_orange_price := orange_price * (1 + orange_increase)
  let new_apple_price := apple_price * (1 + apple_increase)
  lemon_count * new_lemon_price + grape_count * new_grape_price +
  orange_count * new_orange_price + apple_count * new_apple_price = 2995 :=
by
  sorry

#check fruit_sales_calculation 8 7 5 4 80 140 60 100 (1/2) (1/4) (1/10) (1/5)

end NUMINAMATH_CALUDE_fruit_sales_calculation_l3066_306606


namespace NUMINAMATH_CALUDE_veranda_area_is_196_l3066_306631

/-- Represents the dimensions and characteristics of a room with a trapezoidal veranda. -/
structure RoomWithVeranda where
  room_length : ℝ
  room_width : ℝ
  veranda_short_side : ℝ
  veranda_long_side : ℝ

/-- Calculates the area of the trapezoidal veranda surrounding the room. -/
def verandaArea (r : RoomWithVeranda) : ℝ :=
  (r.room_length + 2 * r.veranda_long_side) * (r.room_width + 2 * r.veranda_short_side) - r.room_length * r.room_width

/-- Theorem stating that the area of the trapezoidal veranda is 196 m² for the given dimensions. -/
theorem veranda_area_is_196 (r : RoomWithVeranda)
    (h1 : r.room_length = 17)
    (h2 : r.room_width = 12)
    (h3 : r.veranda_short_side = 2)
    (h4 : r.veranda_long_side = 4) :
    verandaArea r = 196 := by
  sorry

#eval verandaArea { room_length := 17, room_width := 12, veranda_short_side := 2, veranda_long_side := 4 }

end NUMINAMATH_CALUDE_veranda_area_is_196_l3066_306631


namespace NUMINAMATH_CALUDE_residential_building_capacity_l3066_306693

/-- The number of households that can be accommodated in multiple identical residential buildings. -/
def total_households (floors_per_building : ℕ) (households_per_floor : ℕ) (num_buildings : ℕ) : ℕ :=
  floors_per_building * households_per_floor * num_buildings

/-- Theorem stating that 10 buildings with 16 floors and 12 households per floor can accommodate 1920 households. -/
theorem residential_building_capacity :
  total_households 16 12 10 = 1920 := by
  sorry

end NUMINAMATH_CALUDE_residential_building_capacity_l3066_306693


namespace NUMINAMATH_CALUDE_jelly_bean_matching_probability_l3066_306651

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Abe's jelly bean distribution -/
def abe_jelly_beans : JellyBeans :=
  { green := 2, red := 1, blue := 1, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob_jelly_beans : JellyBeans :=
  { green := 3, red := 2, blue := 1, yellow := 2 }

/-- Calculates the probability of both people showing the same color -/
def matching_probability (jb1 jb2 : JellyBeans) : ℚ :=
  let total1 := jb1.total
  let total2 := jb2.total
  (jb1.green * jb2.green + jb1.red * jb2.red + jb1.blue * jb2.blue) / (total1 * total2)

theorem jelly_bean_matching_probability :
  matching_probability abe_jelly_beans bob_jelly_beans = 9 / 32 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_matching_probability_l3066_306651


namespace NUMINAMATH_CALUDE_evaluate_fraction_l3066_306668

theorem evaluate_fraction : 
  (3 * Real.sqrt 7) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 11) = 
  -(1/6) * (Real.sqrt 21 + Real.sqrt 35 - Real.sqrt 77) - 7/3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_fraction_l3066_306668


namespace NUMINAMATH_CALUDE_intersection_A_B_l3066_306641

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_A_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3066_306641


namespace NUMINAMATH_CALUDE_nursery_school_students_nursery_school_students_is_50_l3066_306683

/-- The number of students in a nursery school satisfying specific age distribution conditions -/
theorem nursery_school_students : ℕ :=
  let S : ℕ := 50  -- Total number of students
  let four_and_older : ℕ := S / 10  -- Students 4 years old or older
  let younger_than_three : ℕ := 20  -- Students younger than 3 years old
  let not_between_three_and_four : ℕ := 25  -- Students not between 3 and 4 years old
  have h1 : four_and_older = S / 10 := by sorry
  have h2 : younger_than_three = 20 := by sorry
  have h3 : not_between_three_and_four = 25 := by sorry
  have h4 : four_and_older + younger_than_three = not_between_three_and_four := by sorry
  S

/-- Proof that the number of students in the nursery school is 50 -/
theorem nursery_school_students_is_50 : nursery_school_students = 50 := by sorry

end NUMINAMATH_CALUDE_nursery_school_students_nursery_school_students_is_50_l3066_306683


namespace NUMINAMATH_CALUDE_last_segment_speed_l3066_306613

/-- Proves that the average speed for the last segment is 67 mph given the conditions of the problem -/
theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) 
  (first_segment_speed : ℝ) (second_segment_speed : ℝ) : ℝ :=
  by
  have h1 : total_distance = 96 := by sorry
  have h2 : total_time = 90 / 60 := by sorry
  have h3 : first_segment_speed = 60 := by sorry
  have h4 : second_segment_speed = 65 := by sorry
  
  let overall_average_speed := total_distance / total_time
  have h5 : overall_average_speed = 64 := by sorry
  
  let last_segment_speed := 3 * overall_average_speed - first_segment_speed - second_segment_speed
  
  exact last_segment_speed

end NUMINAMATH_CALUDE_last_segment_speed_l3066_306613


namespace NUMINAMATH_CALUDE_total_wage_calculation_l3066_306663

-- Define the basic parameters
def basic_rate : ℝ := 20
def regular_hours : ℕ := 40
def total_hours : ℕ := 48
def overtime_rate_increase : ℝ := 0.25

-- Define the calculation functions
def regular_pay (rate : ℝ) (hours : ℕ) : ℝ := rate * hours
def overtime_rate (rate : ℝ) (increase : ℝ) : ℝ := rate * (1 + increase)
def overtime_hours (total : ℕ) (regular : ℕ) : ℕ := total - regular
def overtime_pay (rate : ℝ) (hours : ℕ) : ℝ := rate * hours

-- Theorem statement
theorem total_wage_calculation :
  let reg_pay := regular_pay basic_rate regular_hours
  let ot_rate := overtime_rate basic_rate overtime_rate_increase
  let ot_hours := overtime_hours total_hours regular_hours
  let ot_pay := overtime_pay ot_rate ot_hours
  reg_pay + ot_pay = 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_wage_calculation_l3066_306663


namespace NUMINAMATH_CALUDE_integer_root_condition_l3066_306691

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, x^3 + 3*x^2 + a*x + 11 = 0

theorem integer_root_condition (a : ℤ) :
  has_integer_root a ↔ a = -155 ∨ a = -15 ∨ a = 13 ∨ a = 87 :=
sorry

end NUMINAMATH_CALUDE_integer_root_condition_l3066_306691


namespace NUMINAMATH_CALUDE_roundness_of_1280000_l3066_306653

/-- Roundness of a positive integer is the sum of exponents in its prime factorization. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The number we're calculating the roundness for -/
def our_number : ℕ+ := 1280000

/-- Theorem stating that the roundness of 1,280,000 is 19 -/
theorem roundness_of_1280000 : roundness our_number = 19 := by
  sorry

end NUMINAMATH_CALUDE_roundness_of_1280000_l3066_306653


namespace NUMINAMATH_CALUDE_base_6_to_10_54123_l3066_306625

def base_6_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

theorem base_6_to_10_54123 :
  base_6_to_10 [3, 2, 1, 4, 5] = 7395 := by
  sorry

end NUMINAMATH_CALUDE_base_6_to_10_54123_l3066_306625


namespace NUMINAMATH_CALUDE_log_problem_l3066_306642

theorem log_problem (x : ℝ) : 
  x = (Real.log 4 / Real.log 16) ^ (Real.log 16 / Real.log 4) →
  Real.log x / Real.log 7 = -2 * Real.log 2 / Real.log 7 := by
sorry

end NUMINAMATH_CALUDE_log_problem_l3066_306642


namespace NUMINAMATH_CALUDE_count_f_50_eq_18_l3066_306674

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range n)).card + 1

-- Define f₁(n)
def f₁ (n : ℕ) : ℕ := 3 * num_divisors n

-- Define fⱼ(n) recursively
def f (j n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | j + 1 => f₁ (f j n)

-- Define the set of n ≤ 60 for which f₅₀(n) = 18
def set_f_50_eq_18 : Finset ℕ :=
  Finset.filter (λ n => f 50 n = 18) (Finset.range 61)

-- State the theorem
theorem count_f_50_eq_18 : (set_f_50_eq_18.card : ℕ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_count_f_50_eq_18_l3066_306674


namespace NUMINAMATH_CALUDE_tan_sum_product_equals_one_l3066_306657

theorem tan_sum_product_equals_one :
  ∀ (x y : Real),
  (x = 17 * π / 180 ∧ y = 28 * π / 180) →
  (∀ (A B : Real), Real.tan (A + B) = (Real.tan A + Real.tan B) / (1 - Real.tan A * Real.tan B)) →
  (x + y = π / 4) →
  (Real.tan (π / 4) = 1) →
  Real.tan x + Real.tan y + Real.tan x * Real.tan y = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_product_equals_one_l3066_306657


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l3066_306659

theorem sum_of_four_numbers : 3456 + 4563 + 5634 + 6345 = 19998 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l3066_306659


namespace NUMINAMATH_CALUDE_button_probability_l3066_306636

theorem button_probability (initial_red : ℕ) (initial_blue : ℕ) 
  (removed_red : ℕ) (removed_blue : ℕ) :
  initial_red = 8 →
  initial_blue = 12 →
  removed_red = removed_blue →
  (initial_red + initial_blue - removed_red - removed_blue : ℚ) = 
    (5 / 8 : ℚ) * (initial_red + initial_blue : ℚ) →
  ((initial_red - removed_red : ℚ) / (initial_red + initial_blue - removed_red - removed_blue : ℚ)) *
  (removed_red : ℚ) / (removed_red + removed_blue : ℚ) = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_button_probability_l3066_306636


namespace NUMINAMATH_CALUDE_triathlete_average_speed_l3066_306655

/-- Triathlete's average speed for a multi-segment trip -/
theorem triathlete_average_speed (total_distance : ℝ) 
  (run_flat_speed run_uphill_speed run_downhill_speed swim_speed bike_speed : ℝ)
  (run_flat_distance run_uphill_distance run_downhill_distance swim_distance bike_distance : ℝ)
  (h1 : total_distance = run_flat_distance + run_uphill_distance + run_downhill_distance + swim_distance + bike_distance)
  (h2 : run_flat_speed > 0 ∧ run_uphill_speed > 0 ∧ run_downhill_speed > 0 ∧ swim_speed > 0 ∧ bike_speed > 0)
  (h3 : run_flat_distance > 0 ∧ run_uphill_distance > 0 ∧ run_downhill_distance > 0 ∧ swim_distance > 0 ∧ bike_distance > 0)
  (h4 : total_distance = 9)
  (h5 : run_flat_speed = 10)
  (h6 : run_uphill_speed = 6)
  (h7 : run_downhill_speed = 14)
  (h8 : swim_speed = 4)
  (h9 : bike_speed = 12)
  (h10 : run_flat_distance = 1)
  (h11 : run_uphill_distance = 1)
  (h12 : run_downhill_distance = 1)
  (h13 : swim_distance = 3)
  (h14 : bike_distance = 3) :
  ∃ (average_speed : ℝ), abs (average_speed - 0.1121) < 0.0001 ∧
    average_speed = total_distance / (run_flat_distance / run_flat_speed + 
                                      run_uphill_distance / run_uphill_speed + 
                                      run_downhill_distance / run_downhill_speed + 
                                      swim_distance / swim_speed + 
                                      bike_distance / bike_speed) / 60 :=
by sorry

end NUMINAMATH_CALUDE_triathlete_average_speed_l3066_306655


namespace NUMINAMATH_CALUDE_expanded_volume_of_problem_box_l3066_306650

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of space inside and within one unit of a box -/
def expandedVolume (b : Box) : ℝ := sorry

/-- The specific box in the problem -/
def problemBox : Box := ⟨2, 3, 4⟩

theorem expanded_volume_of_problem_box :
  expandedVolume problemBox = (228 + 31 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_expanded_volume_of_problem_box_l3066_306650


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l3066_306605

def satisfies_inequality (x : ℤ) : Prop :=
  |7 * x - 3| - 2 * x < 5 - 3 * x

theorem greatest_integer_satisfying_inequality :
  satisfies_inequality 0 ∧
  ∀ y : ℤ, y > 0 → ¬(satisfies_inequality y) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l3066_306605


namespace NUMINAMATH_CALUDE_bug_return_probability_l3066_306680

/-- Probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The probability of the bug returning to its starting vertex on the twelfth move -/
theorem bug_return_probability : Q 12 = 44287 / 177147 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l3066_306680


namespace NUMINAMATH_CALUDE_jennifers_cans_count_l3066_306689

/-- The number of cans Jennifer brought home from the store -/
def jennifers_total_cans (initial_cans : ℕ) (marks_cans : ℕ) : ℕ :=
  initial_cans + (6 * marks_cans) / 5

/-- Theorem stating the total number of cans Jennifer brought home -/
theorem jennifers_cans_count : jennifers_total_cans 40 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_jennifers_cans_count_l3066_306689


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3066_306638

theorem rectangular_field_area (width length perimeter area : ℝ) : 
  width = (1/3) * length →
  perimeter = 2 * (width + length) →
  perimeter = 60 →
  area = width * length →
  area = 168.75 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3066_306638


namespace NUMINAMATH_CALUDE_no_single_digit_quadratic_solution_l3066_306687

theorem no_single_digit_quadratic_solution :
  ¬∃ (A : ℕ), 1 ≤ A ∧ A ≤ 9 ∧
  ∃ (x : ℕ), x > 0 ∧ x^2 - (2*A)*x + (A^2 + 1) = 0 :=
sorry

end NUMINAMATH_CALUDE_no_single_digit_quadratic_solution_l3066_306687


namespace NUMINAMATH_CALUDE_work_completion_time_l3066_306684

theorem work_completion_time (x : ℕ) : 
  (50 * x = 25 * (x + 20)) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3066_306684
