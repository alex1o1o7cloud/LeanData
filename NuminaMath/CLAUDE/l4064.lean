import Mathlib

namespace NUMINAMATH_CALUDE_maggie_yellow_packs_l4064_406418

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The number of packs of red bouncy balls Maggie bought -/
def red_packs : ℕ := 4

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs : ℕ := 4

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := 160

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℕ := (total_balls - (red_packs + green_packs) * balls_per_pack) / balls_per_pack

theorem maggie_yellow_packs : yellow_packs = 8 := by
  sorry

end NUMINAMATH_CALUDE_maggie_yellow_packs_l4064_406418


namespace NUMINAMATH_CALUDE_prob_no_consecutive_heads_is_half_l4064_406424

/-- The probability of heads not appearing consecutively when tossing a fair coin four times -/
def prob_no_consecutive_heads : ℚ := 1/2

/-- A fair coin is tossed four times -/
def num_tosses : ℕ := 4

/-- The total number of possible outcomes when tossing a fair coin four times -/
def total_outcomes : ℕ := 2^num_tosses

/-- The number of outcomes where heads do not appear consecutively -/
def favorable_outcomes : ℕ := 8

theorem prob_no_consecutive_heads_is_half :
  prob_no_consecutive_heads = favorable_outcomes / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_prob_no_consecutive_heads_is_half_l4064_406424


namespace NUMINAMATH_CALUDE_simplify_fraction_l4064_406453

theorem simplify_fraction (a : ℚ) (h : a = 2) : 24 * a^5 / (72 * a^3) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4064_406453


namespace NUMINAMATH_CALUDE_password_count_l4064_406449

def password_length : ℕ := 4
def available_digits : ℕ := 9  -- 10 digits minus 1 (7 is excluded)

def total_passwords : ℕ := available_digits ^ password_length

def all_different_passwords : ℕ := Nat.choose available_digits password_length * Nat.factorial password_length

theorem password_count : 
  total_passwords - all_different_passwords = 3537 :=
by sorry

end NUMINAMATH_CALUDE_password_count_l4064_406449


namespace NUMINAMATH_CALUDE_inequality_proof_l4064_406463

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d)
  (hsum : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4064_406463


namespace NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l4064_406492

theorem smallest_divisible_by_one_to_ten : ∃ n : ℕ,
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧
  (∀ m : ℕ, m < n → ∃ j : ℕ, 1 ≤ j ∧ j ≤ 10 ∧ ¬(j ∣ m)) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l4064_406492


namespace NUMINAMATH_CALUDE_rationalize_denominator_l4064_406442

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (4 * Real.sqrt 6 + 5 * Real.sqrt 7) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧ B = 6 ∧ C = 15 ∧ D = 7 ∧ E = 79 ∧
    Int.gcd A E = 1 ∧ Int.gcd C E = 1 ∧
    Int.gcd B D = 1 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l4064_406442


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l4064_406414

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 6*x + 7 = 0) ↔ ((x - 3)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l4064_406414


namespace NUMINAMATH_CALUDE_product_of_max_min_sum_l4064_406416

theorem product_of_max_min_sum (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  (4 : ℝ)^(Real.sqrt (5*x + 9*y + 4*z)) - 68 * 2^(Real.sqrt (5*x + 9*y + 4*z)) + 256 = 0 →
  ∃ (min_sum max_sum : ℝ),
    (∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → 
      (4 : ℝ)^(Real.sqrt (5*a + 9*b + 4*c)) - 68 * 2^(Real.sqrt (5*a + 9*b + 4*c)) + 256 = 0 →
      min_sum ≤ a + b + c ∧ a + b + c ≤ max_sum) ∧
    min_sum * max_sum = 4 :=
by sorry

end NUMINAMATH_CALUDE_product_of_max_min_sum_l4064_406416


namespace NUMINAMATH_CALUDE_translation_theorem_l4064_406482

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem (A B C : Point) (t : Translation) :
  A.x = -1 ∧ A.y = 4 ∧
  B.x = -4 ∧ B.y = -1 ∧
  C.x = 4 ∧ C.y = 7 ∧
  C = applyTranslation t A →
  applyTranslation t B = { x := 1, y := 2 } := by
  sorry


end NUMINAMATH_CALUDE_translation_theorem_l4064_406482


namespace NUMINAMATH_CALUDE_sequence_nth_term_l4064_406466

theorem sequence_nth_term (n : ℕ+) (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h_sum : ∀ k : ℕ+, S k = k^2) :
  a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_nth_term_l4064_406466


namespace NUMINAMATH_CALUDE_function_property_l4064_406469

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ p q : ℝ, f (p + q) = f p * f q) 
  (h2 : f 1 = 3) : 
  (f 1 * f 1 + f 2) / f 1 + (f 2 * f 2 + f 4) / f 3 + 
  (f 3 * f 3 + f 6) / f 5 + (f 4 * f 4 + f 8) / f 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l4064_406469


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1984_l4064_406458

theorem smallest_divisible_by_1984 :
  ∃ (a : ℕ), (a > 0) ∧
  (∀ (n : ℕ), Odd n → (47^n + a * 15^n) % 1984 = 0) ∧
  (∀ (b : ℕ), 0 < b ∧ b < a → ∃ (m : ℕ), Odd m ∧ (47^m + b * 15^m) % 1984 ≠ 0) ∧
  (a = 1055) := by
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1984_l4064_406458


namespace NUMINAMATH_CALUDE_pool_capacity_l4064_406438

theorem pool_capacity (initial_percentage : ℚ) (final_percentage : ℚ) (added_water : ℚ) :
  initial_percentage = 0.4 →
  final_percentage = 0.8 →
  added_water = 300 →
  (∃ (total_capacity : ℚ), 
    total_capacity * final_percentage = total_capacity * initial_percentage + added_water ∧
    total_capacity = 750) :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_l4064_406438


namespace NUMINAMATH_CALUDE_average_age_of_class_l4064_406481

theorem average_age_of_class (total_students : ℕ) 
  (group1_count group2_count : ℕ) 
  (group1_avg group2_avg last_student_age : ℝ) : 
  total_students = group1_count + group2_count + 1 →
  group1_count = 8 →
  group2_count = 6 →
  group1_avg = 14 →
  group2_avg = 16 →
  last_student_age = 17 →
  (group1_count * group1_avg + group2_count * group2_avg + last_student_age) / total_students = 15 :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_class_l4064_406481


namespace NUMINAMATH_CALUDE_correct_factorization_l4064_406427

theorem correct_factorization (a : ℝ) :
  a^2 - a + (1/4 : ℝ) = (a - 1/2)^2 := by sorry

end NUMINAMATH_CALUDE_correct_factorization_l4064_406427


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_managers_l4064_406406

theorem stratified_sampling_middle_managers :
  ∀ (total_employees : ℕ) 
    (middle_managers : ℕ) 
    (sample_size : ℕ),
  total_employees = 160 →
  middle_managers = 30 →
  sample_size = 32 →
  (middle_managers * sample_size) / total_employees = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_managers_l4064_406406


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l4064_406483

theorem exterior_angle_measure (a b : ℝ) (ha : a = 40) (hb : b = 30) : 
  180 - (180 - a - b) = 70 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l4064_406483


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l4064_406445

theorem cylinder_radius_problem (r : ℝ) (y : ℝ) : 
  r > 0 →
  (π * ((r + 4)^2 * 4 - r^2 * 4) = y) →
  (π * (r^2 * 8 - r^2 * 4) = y) →
  r = 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l4064_406445


namespace NUMINAMATH_CALUDE_min_segment_length_in_right_angle_l4064_406467

/-- Given a point inside a 90° angle, located 8 units from one side and 1 unit from the other side,
    the minimum length of a segment passing through this point with ends on the sides of the angle is 10 units. -/
theorem min_segment_length_in_right_angle (P : ℝ × ℝ) 
  (inside_angle : P.1 > 0 ∧ P.2 > 0) 
  (dist_to_sides : P.1 = 1 ∧ P.2 = 8) : 
  Real.sqrt ((P.1 + P.1)^2 + (P.2 + P.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_segment_length_in_right_angle_l4064_406467


namespace NUMINAMATH_CALUDE_max_area_height_l4064_406486

/-- A right trapezoid with an acute angle of 30° and perimeter 6 -/
structure RightTrapezoid where
  height : ℝ
  sumOfBases : ℝ
  acuteAngle : ℝ
  perimeter : ℝ
  area : ℝ
  acuteAngle_eq : acuteAngle = π / 6
  perimeter_eq : perimeter = 6
  area_eq : area = (3 * sumOfBases * height) / 2
  perimeter_constraint : sumOfBases + 3 * height = 6

/-- The height that maximizes the area of the right trapezoid is 1 -/
theorem max_area_height (t : RightTrapezoid) : 
  t.area ≤ (3 : ℝ) / 2 ∧ (t.area = (3 : ℝ) / 2 ↔ t.height = 1) :=
sorry

end NUMINAMATH_CALUDE_max_area_height_l4064_406486


namespace NUMINAMATH_CALUDE_six_points_fifteen_segments_l4064_406473

/-- The number of line segments formed by connecting n distinct points on a circle --/
def lineSegments (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 6 distinct points on a circle, the number of line segments is 15 --/
theorem six_points_fifteen_segments : lineSegments 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_six_points_fifteen_segments_l4064_406473


namespace NUMINAMATH_CALUDE_oil_cylinder_capacity_l4064_406400

theorem oil_cylinder_capacity (capacity : ℚ) 
  (h1 : (3 : ℚ) / 4 * capacity = 27.5)
  (h2 : (9 : ℚ) / 10 * capacity = 35) : 
  capacity = 110 / 3 := by sorry

end NUMINAMATH_CALUDE_oil_cylinder_capacity_l4064_406400


namespace NUMINAMATH_CALUDE_sherry_opposite_vertex_probability_l4064_406475

/-- The probability that Sherry is at the opposite vertex after k minutes on a triangle -/
def P (k : ℕ) : ℚ :=
  1/6 + 1/(3 * (-2)^k)

/-- Theorem: For k > 0, the probability that Sherry is at the opposite vertex after k minutes on a triangle is P(k) -/
theorem sherry_opposite_vertex_probability (k : ℕ) (h : k > 0) : 
  (1/6 : ℚ) + 1/(3 * (-2)^k) = P k := by
sorry

end NUMINAMATH_CALUDE_sherry_opposite_vertex_probability_l4064_406475


namespace NUMINAMATH_CALUDE_exists_non_increasing_function_with_condition_increasing_on_subset_not_implies_entire_interval_inverse_function_decreasing_intervals_monotonic_function_extrema_on_closed_interval_l4064_406411

-- 1
theorem exists_non_increasing_function_with_condition :
  ∃ f : ℝ → ℝ, f (-1) < f 3 ∧ ¬(∀ x y : ℝ, x < y → f x < f y) := by sorry

-- 2
theorem increasing_on_subset_not_implies_entire_interval
  (f : ℝ → ℝ) (h : ∀ x y : ℝ, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y) :
  ¬(∀ a b : ℝ, a < b → (∀ x y : ℝ, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x < f y) →
    Set.Icc a b = Set.Ici 1) := by sorry

-- 3
theorem inverse_function_decreasing_intervals :
  ¬(∀ x y : ℝ, (x < y ∧ x < 0 ∧ y < 0) ∨ (x < y ∧ x > 0 ∧ y > 0) → 1/x > 1/y) := by sorry

-- 4
theorem monotonic_function_extrema_on_closed_interval
  {a b : ℝ} (f : ℝ → ℝ) (h : Monotone f) :
  ∃ x y : ℝ, x ∈ Set.Icc a b ∧ y ∈ Set.Icc a b ∧
    (∀ z : ℝ, z ∈ Set.Icc a b → f x ≤ f z) ∧
    (∀ z : ℝ, z ∈ Set.Icc a b → f z ≤ f y) ∧
    (x = a ∨ x = b) ∧ (y = a ∨ y = b) := by sorry

end NUMINAMATH_CALUDE_exists_non_increasing_function_with_condition_increasing_on_subset_not_implies_entire_interval_inverse_function_decreasing_intervals_monotonic_function_extrema_on_closed_interval_l4064_406411


namespace NUMINAMATH_CALUDE_track_circumference_l4064_406478

/-- Represents the circular track and the runners' positions --/
structure TrackSystem where
  circumference : ℝ
  first_meeting_distance : ℝ
  second_meeting_distance : ℝ

/-- The conditions of the problem --/
def problem_conditions (t : TrackSystem) : Prop :=
  t.first_meeting_distance = 150 ∧
  t.second_meeting_distance = t.circumference - 90 ∧
  2 * t.circumference = t.first_meeting_distance * 2 + t.second_meeting_distance

/-- The theorem stating that the circumference is 300 yards --/
theorem track_circumference (t : TrackSystem) :
  problem_conditions t → t.circumference = 300 :=
by
  sorry


end NUMINAMATH_CALUDE_track_circumference_l4064_406478


namespace NUMINAMATH_CALUDE_overall_average_score_problem_solution_l4064_406437

/-- Calculates the overall average score of two classes -/
theorem overall_average_score 
  (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) : 
  (n1 : ℝ) * avg1 + (n2 : ℝ) * avg2 = ((n1 + n2) : ℝ) * ((n1 * avg1 + n2 * avg2) / (n1 + n2)) :=
by sorry

/-- Proves that the overall average score for the given problem is 74 -/
theorem problem_solution 
  (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) 
  (h1 : n1 = 20) (h2 : n2 = 30) (h3 : avg1 = 80) (h4 : avg2 = 70) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 74 :=
by sorry

end NUMINAMATH_CALUDE_overall_average_score_problem_solution_l4064_406437


namespace NUMINAMATH_CALUDE_amanda_notebooks_l4064_406477

/-- Calculates the final number of notebooks Amanda has -/
def final_notebooks (initial : ℕ) (ordered : ℕ) (lost : ℕ) : ℕ :=
  initial + ordered - lost

theorem amanda_notebooks :
  final_notebooks 10 6 2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_amanda_notebooks_l4064_406477


namespace NUMINAMATH_CALUDE_joan_missed_games_l4064_406474

/-- The number of baseball games Joan missed -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Theorem stating that Joan missed 469 games -/
theorem joan_missed_games :
  let total_games : ℕ := 864
  let attended_games : ℕ := 395
  games_missed total_games attended_games = 469 := by
  sorry

end NUMINAMATH_CALUDE_joan_missed_games_l4064_406474


namespace NUMINAMATH_CALUDE_tangent_line_equality_l4064_406499

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Define the derivatives of f and g
def f' (a b x : ℝ) : ℝ := 3*x^2 + 4*a*x + b
def g' (x : ℝ) : ℝ := 2*x - 3

-- State the theorem
theorem tangent_line_equality (a b : ℝ) :
  f a b 2 = g 2 ∧ f' a b 2 = g' 2 →
  a = -2 ∧ b = 5 ∧ ∀ x y, y = x - 2 ↔ x - y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equality_l4064_406499


namespace NUMINAMATH_CALUDE_average_weight_problem_l4064_406491

/-- The average weight problem -/
theorem average_weight_problem 
  (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84)
  (h2 : (A + B + C + D) / 4 = 80)
  (h3 : E = D + 8)
  (h4 : A = 80) :
  (B + C + D + E) / 4 = 79 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l4064_406491


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l4064_406413

theorem polynomial_root_problem (a b c d : ℝ) 
  (h1 : ∃ x1 x2 x3 x4 : ℂ, x1^4 + a*x1^3 + b*x1^2 + c*x1 + d = 0 ∧ 
                           x2^4 + a*x2^3 + b*x2^2 + c*x2 + d = 0 ∧ 
                           x3^4 + a*x3^3 + b*x3^2 + c*x3 + d = 0 ∧ 
                           x4^4 + a*x4^3 + b*x4^2 + c*x4 + d = 0)
  (h2 : ∀ x : ℂ, x^4 + a*x^3 + b*x^2 + c*x + d = 0 → x.im ≠ 0)
  (h3 : ∃ x1 x2 : ℂ, (x1^4 + a*x1^3 + b*x1^2 + c*x1 + d = 0) ∧ 
                     (x2^4 + a*x2^3 + b*x2^2 + c*x2 + d = 0) ∧ 
                     (x1 * x2 = 13 + I))
  (h4 : ∃ x3 x4 : ℂ, (x3^4 + a*x3^3 + b*x3^2 + c*x3 + d = 0) ∧ 
                     (x4^4 + a*x4^3 + b*x4^2 + c*x4 + d = 0) ∧ 
                     (x3 + x4 = 3 + 4*I)) :
  b = 51 := by sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l4064_406413


namespace NUMINAMATH_CALUDE_quadratic_solution_l4064_406433

theorem quadratic_solution : ∃ x₁ x₂ : ℝ, x₁ = 6 ∧ x₂ = -1 ∧ ∀ x : ℝ, x^2 - 5*x - 6 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l4064_406433


namespace NUMINAMATH_CALUDE_julie_school_work_hours_l4064_406457

-- Define the given parameters
def summer_weeks : ℕ := 10
def summer_hours_per_week : ℕ := 36
def summer_earnings : ℕ := 4500
def school_weeks : ℕ := 45
def school_earnings : ℕ := 4500

-- Define the function to calculate required hours per week
def required_hours_per_week (weeks : ℕ) (total_earnings : ℕ) (hourly_rate : ℚ) : ℚ :=
  (total_earnings : ℚ) / (weeks : ℚ) / hourly_rate

-- Theorem statement
theorem julie_school_work_hours :
  let hourly_rate : ℚ := (summer_earnings : ℚ) / ((summer_weeks * summer_hours_per_week) : ℚ)
  required_hours_per_week school_weeks school_earnings hourly_rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_julie_school_work_hours_l4064_406457


namespace NUMINAMATH_CALUDE_team_photo_arrangements_l4064_406412

/-- The number of students in each group (boys and girls) -/
def group_size : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := 2 * group_size

/-- The number of possible arrangements for the team photo -/
def photo_arrangements : ℕ := (Nat.factorial group_size) * (Nat.factorial group_size)

/-- Theorem stating that the number of possible arrangements is 36 -/
theorem team_photo_arrangements :
  photo_arrangements = 36 :=
by sorry

end NUMINAMATH_CALUDE_team_photo_arrangements_l4064_406412


namespace NUMINAMATH_CALUDE_cost_difference_70_copies_l4064_406465

/-- Calculates the cost for color copies at print shop X -/
def costX (copies : ℕ) : ℚ :=
  if copies ≤ 50 then
    1.2 * copies
  else
    1.2 * 50 + 0.9 * (copies - 50)

/-- Calculates the cost for color copies at print shop Y -/
def costY (copies : ℕ) : ℚ :=
  10 + 1.7 * copies

/-- The difference in cost between print shop Y and X for 70 color copies is $51 -/
theorem cost_difference_70_copies : costY 70 - costX 70 = 51 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_70_copies_l4064_406465


namespace NUMINAMATH_CALUDE_point_symmetry_l4064_406422

def f (x : ℝ) : ℝ := x^3

theorem point_symmetry (a b : ℝ) : 
  (f a = b) → (f (-a) = -b) := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l4064_406422


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l4064_406436

theorem geometric_to_arithmetic_sequence (a₁ a₂ a₃ a₄ q : ℝ) : 
  q > 0 ∧ q ≠ 1 ∧
  a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧
  ((a₁ + a₃ = 2 * a₂) ∨ (a₁ + a₄ = 2 * a₃)) →
  q = ((-1 + Real.sqrt 5) / 2) ∨ q = ((1 + Real.sqrt 5) / 2) := by
sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l4064_406436


namespace NUMINAMATH_CALUDE_fraction_equality_unique_solution_l4064_406493

theorem fraction_equality_unique_solution :
  ∃! (C D : ℝ), ∀ x : ℝ, x ≠ 4 ∧ x ≠ 5 →
    (D * x - 23) / (x^2 - 9*x + 20) = C / (x - 4) + 7 / (x - 5) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_fraction_equality_unique_solution_l4064_406493


namespace NUMINAMATH_CALUDE_max_temperature_range_l4064_406417

theorem max_temperature_range (temps : Finset ℝ) (avg : ℝ) (min_temp : ℝ) :
  temps.card = 5 →
  Finset.sum temps id / temps.card = avg →
  avg = 60 →
  min_temp = 40 →
  min_temp ∈ temps →
  ∀ t ∈ temps, t ≥ min_temp →
  ∃ max_temp ∈ temps, max_temp - min_temp ≤ 100 ∧
    ∀ t ∈ temps, t - min_temp ≤ max_temp - min_temp :=
by sorry

end NUMINAMATH_CALUDE_max_temperature_range_l4064_406417


namespace NUMINAMATH_CALUDE_legos_lost_l4064_406454

theorem legos_lost (initial_legos current_legos : ℕ) 
  (h1 : initial_legos = 380) 
  (h2 : current_legos = 323) : 
  initial_legos - current_legos = 57 := by
  sorry

end NUMINAMATH_CALUDE_legos_lost_l4064_406454


namespace NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_30_l4064_406455

theorem smallest_of_three_consecutive_sum_30 (x : ℕ) :
  x + (x + 1) + (x + 2) = 30 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_30_l4064_406455


namespace NUMINAMATH_CALUDE_eight_thousand_eight_place_values_l4064_406439

/-- Represents the place value of a digit in a number -/
inductive PlaceValue
  | Ones
  | Tens
  | Hundreds
  | Thousands

/-- Returns the place value of a digit based on its position from the right -/
def getPlaceValue (position : Nat) : PlaceValue :=
  match position with
  | 1 => PlaceValue.Ones
  | 2 => PlaceValue.Tens
  | 3 => PlaceValue.Hundreds
  | 4 => PlaceValue.Thousands
  | _ => PlaceValue.Ones  -- Default to Ones for other positions

/-- Represents a digit in a specific position of a number -/
structure Digit where
  value : Nat
  position : Nat

/-- Theorem: In the number 8008, the 8 in the first position from the right represents 8 units of ones,
    and the 8 in the fourth position from the right represents 8 units of thousands -/
theorem eight_thousand_eight_place_values :
  let num := 8008
  let rightmost_eight : Digit := { value := 8, position := 1 }
  let leftmost_eight : Digit := { value := 8, position := 4 }
  (getPlaceValue rightmost_eight.position = PlaceValue.Ones) ∧
  (getPlaceValue leftmost_eight.position = PlaceValue.Thousands) :=
by sorry

end NUMINAMATH_CALUDE_eight_thousand_eight_place_values_l4064_406439


namespace NUMINAMATH_CALUDE_continuous_and_strictly_monotone_function_l4064_406431

-- Define a function type from reals to reals
def RealFunction := ℝ → ℝ

-- Define the property of having limits at any point
def has_limits_at_any_point (f : RealFunction) : Prop :=
  ∀ a : ℝ, ∃ L : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - a| < δ → |f x - L| < ε

-- Define the property of having no local extrema
def has_no_local_extrema (f : RealFunction) : Prop :=
  ∀ a : ℝ, ∀ ε > 0, ∃ x y : ℝ, |x - a| < ε ∧ |y - a| < ε ∧ f x < f a ∧ f a < f y

-- State the theorem
theorem continuous_and_strictly_monotone_function 
  (f : RealFunction) 
  (h1 : has_limits_at_any_point f) 
  (h2 : has_no_local_extrema f) : 
  Continuous f ∧ StrictMono f :=
by sorry

end NUMINAMATH_CALUDE_continuous_and_strictly_monotone_function_l4064_406431


namespace NUMINAMATH_CALUDE_equal_area_trapezoid_kp_l4064_406405

/-- Represents a trapezoid with two bases and a point that divides it into equal areas -/
structure EqualAreaTrapezoid where
  /-- Length of the longer base KL -/
  base_kl : ℝ
  /-- Length of the shorter base MN -/
  base_mn : ℝ
  /-- Length of segment KP, where P divides the trapezoid into equal areas when connected to N -/
  kp : ℝ
  /-- Assumption that base_kl is greater than base_mn -/
  h_base : base_kl > base_mn
  /-- Assumption that all lengths are positive -/
  h_positive : base_kl > 0 ∧ base_mn > 0 ∧ kp > 0

/-- Theorem stating that for a trapezoid with given dimensions, KP = 28 when P divides the area equally -/
theorem equal_area_trapezoid_kp
  (t : EqualAreaTrapezoid)
  (h_kl : t.base_kl = 40)
  (h_mn : t.base_mn = 16) :
  t.kp = 28 := by
  sorry

#check equal_area_trapezoid_kp

end NUMINAMATH_CALUDE_equal_area_trapezoid_kp_l4064_406405


namespace NUMINAMATH_CALUDE_derek_added_water_l4064_406410

theorem derek_added_water (initial_amount final_amount : ℝ) (h1 : initial_amount = 3) (h2 : final_amount = 9.8) :
  final_amount - initial_amount = 6.8 := by
  sorry

end NUMINAMATH_CALUDE_derek_added_water_l4064_406410


namespace NUMINAMATH_CALUDE_line_points_l4064_406456

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def lies_on_line (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

/-- The main theorem -/
theorem line_points : 
  let p1 : Point := ⟨8, 16⟩
  let p2 : Point := ⟨2, 4⟩
  let points_on_line : List Point := [⟨5, 10⟩, ⟨7, 14⟩, ⟨10, 20⟩, ⟨3, 6⟩]
  let point_not_on_line : Point := ⟨4, 7⟩
  (∀ p ∈ points_on_line, lies_on_line p p1 p2) ∧
  ¬(lies_on_line point_not_on_line p1 p2) := by
  sorry

end NUMINAMATH_CALUDE_line_points_l4064_406456


namespace NUMINAMATH_CALUDE_histogram_area_sum_is_one_l4064_406446

/-- Represents a histogram of sample frequency distribution -/
structure Histogram where
  rectangles : List ℝ
  -- Each element in the list represents the area of a small rectangle

/-- The sum of areas of all rectangles in a histogram equals 1 -/
theorem histogram_area_sum_is_one (h : Histogram) : 
  h.rectangles.sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_histogram_area_sum_is_one_l4064_406446


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l4064_406490

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 21) : 
  max x (max (x + 1) (x + 2)) = 8 := by
sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l4064_406490


namespace NUMINAMATH_CALUDE_point_on_line_coordinates_l4064_406448

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line passing through two points in 3D space -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Function to get a point on a line given an x-coordinate -/
def pointOnLine (l : Line3D) (x : ℝ) : Point3D :=
  sorry

theorem point_on_line_coordinates (l : Line3D) :
  l.p1 = ⟨1, 3, 4⟩ →
  l.p2 = ⟨4, 2, 1⟩ →
  let p := pointOnLine l 7
  p.y = 1 ∧ p.z = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_coordinates_l4064_406448


namespace NUMINAMATH_CALUDE_birds_on_trees_l4064_406404

theorem birds_on_trees (n : ℕ) (h : n = 44) : 
  let initial_sum := n * (n + 1) / 2
  ∀ (current_sum : ℕ), current_sum % 4 ≠ 0 →
    ∃ (next_sum : ℕ), (next_sum = current_sum ∨ next_sum = current_sum + n - 1 ∨ next_sum = current_sum - (n - 1)) ∧
      next_sum % 4 ≠ 0 :=
by sorry

#check birds_on_trees

end NUMINAMATH_CALUDE_birds_on_trees_l4064_406404


namespace NUMINAMATH_CALUDE_ellipse_k_range_l4064_406497

def is_ellipse (k : ℝ) : Prop :=
  ∀ x y : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
    x^2 / (k - 4) + y^2 / (9 - k) = 1 ↔ (x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_k_range (k : ℝ) :
  is_ellipse k ↔ (k ∈ Set.Ioo 4 (13/2) ∪ Set.Ioo (13/2) 9) :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l4064_406497


namespace NUMINAMATH_CALUDE_polynomial_not_equal_33_l4064_406464

theorem polynomial_not_equal_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_equal_33_l4064_406464


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4064_406440

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 5 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 3*x₀ + 5 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4064_406440


namespace NUMINAMATH_CALUDE_passing_percentage_is_33_percent_l4064_406487

def total_marks : ℕ := 400
def obtained_marks : ℕ := 92
def failing_margin : ℕ := 40

theorem passing_percentage_is_33_percent :
  (obtained_marks + failing_margin) / total_marks * 100 = 33 := by sorry

end NUMINAMATH_CALUDE_passing_percentage_is_33_percent_l4064_406487


namespace NUMINAMATH_CALUDE_line_contains_point_l4064_406403

/-- 
Given a line represented by the equation 1-kx = -3y that contains the point (4, -3),
prove that the value of k is -2.
-/
theorem line_contains_point (k : ℝ) : 
  (1 - k * 4 = -3 * (-3)) → k = -2 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l4064_406403


namespace NUMINAMATH_CALUDE_equation_solution_l4064_406462

theorem equation_solution : 
  ∃ x₁ x₂ : ℚ, (x₁ = 5/2 ∧ x₂ = -1/2) ∧ 
  (∀ x : ℚ, 4 * (x - 1)^2 = 9 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4064_406462


namespace NUMINAMATH_CALUDE_power_mod_prime_remainder_5_1000_mod_29_l4064_406432

theorem power_mod_prime (p : Nat) (a : Nat) (m : Nat) (h : Prime p) :
  a ^ m % p = (a ^ (m % (p - 1))) % p :=
sorry

theorem remainder_5_1000_mod_29 : 5^1000 % 29 = 21 := by
  have h1 : Prime 29 := by sorry
  have h2 : 5^1000 % 29 = (5^(1000 % 28)) % 29 := by
    apply power_mod_prime 29 5 1000 h1
  have h3 : 1000 % 28 = 20 := by sorry
  have h4 : (5^20) % 29 = 21 := by sorry
  rw [h2, h3, h4]

end NUMINAMATH_CALUDE_power_mod_prime_remainder_5_1000_mod_29_l4064_406432


namespace NUMINAMATH_CALUDE_a_51_value_l4064_406434

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem a_51_value (a : ℕ → ℤ) (h : arithmetic_sequence a) : a 51 = 101 := by
  sorry

end NUMINAMATH_CALUDE_a_51_value_l4064_406434


namespace NUMINAMATH_CALUDE_chess_tournament_theorem_l4064_406485

/-- Represents a single-elimination chess tournament -/
structure ChessTournament where
  participants : ℕ
  winner_games : ℕ
  is_power_of_two : ∃ n : ℕ, participants = 2^n
  winner_played_six : winner_games = 6

/-- Number of participants who won at least 2 more games than they lost -/
def participants_with_two_more_wins (t : ChessTournament) : ℕ :=
  8

theorem chess_tournament_theorem (t : ChessTournament) :
  participants_with_two_more_wins t = 8 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_theorem_l4064_406485


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l4064_406476

theorem min_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hsum : x + y = 15) :
  ∃ (min : ℝ), min = 4/15 ∧ ∀ (a b : ℝ), 0 < a → 0 < b → a + b = 15 → min ≤ 1/a + 1/b :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l4064_406476


namespace NUMINAMATH_CALUDE_books_sold_l4064_406484

theorem books_sold (initial_books : ℕ) (remaining_books : ℕ) (h1 : initial_books = 242) (h2 : remaining_books = 105) :
  initial_books - remaining_books = 137 := by
sorry

end NUMINAMATH_CALUDE_books_sold_l4064_406484


namespace NUMINAMATH_CALUDE_buses_needed_l4064_406419

theorem buses_needed (classrooms : ℕ) (students_per_classroom : ℕ) (seats_per_bus : ℕ) : 
  classrooms = 67 → students_per_classroom = 66 → seats_per_bus = 6 →
  (classrooms * students_per_classroom + seats_per_bus - 1) / seats_per_bus = 738 := by
  sorry

end NUMINAMATH_CALUDE_buses_needed_l4064_406419


namespace NUMINAMATH_CALUDE_rectangle_length_l4064_406447

/-- Given a rectangle with width 6 inches and area 48 square inches, prove its length is 8 inches -/
theorem rectangle_length (width : ℝ) (area : ℝ) (h1 : width = 6) (h2 : area = 48) :
  area / width = 8 := by sorry

end NUMINAMATH_CALUDE_rectangle_length_l4064_406447


namespace NUMINAMATH_CALUDE_eighth_odd_multiple_of_five_l4064_406460

def is_odd_multiple_of_five (n : ℕ) : Prop := n % 2 = 1 ∧ n % 5 = 0

def nth_odd_multiple_of_five (n : ℕ) : ℕ :=
  (2 * n - 1) * 5

theorem eighth_odd_multiple_of_five :
  nth_odd_multiple_of_five 8 = 75 ∧ is_odd_multiple_of_five (nth_odd_multiple_of_five 8) :=
sorry

end NUMINAMATH_CALUDE_eighth_odd_multiple_of_five_l4064_406460


namespace NUMINAMATH_CALUDE_train_speed_l4064_406459

/-- The speed of a train given specific conditions -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_length = 500 →
  man_speed = 12 →
  passing_time = 10 →
  ∃ (train_speed : ℝ), train_speed = 168 ∧ 
    (train_speed + man_speed) * passing_time / 3.6 = train_length + man_speed * passing_time / 3.6 :=
by sorry


end NUMINAMATH_CALUDE_train_speed_l4064_406459


namespace NUMINAMATH_CALUDE_shoes_to_sandals_ratio_l4064_406451

def shoes_sold : ℕ := 72
def sandals_sold : ℕ := 40

theorem shoes_to_sandals_ratio :
  (shoes_sold / sandals_sold : ℚ) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_shoes_to_sandals_ratio_l4064_406451


namespace NUMINAMATH_CALUDE_simplify_expression_l4064_406401

theorem simplify_expression (m n : ℝ) : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4064_406401


namespace NUMINAMATH_CALUDE_fence_perimeter_is_236_l4064_406409

/-- Represents the configuration of a rectangular fence --/
structure FenceConfig where
  total_posts : ℕ
  post_width : ℚ
  gap_width : ℕ
  long_short_ratio : ℕ

/-- Calculates the perimeter of the fence given its configuration --/
def calculate_perimeter (config : FenceConfig) : ℚ :=
  let short_side_posts := config.total_posts / (config.long_short_ratio + 1)
  let long_side_posts := short_side_posts * config.long_short_ratio
  let short_side_length := short_side_posts * config.post_width + (short_side_posts - 1) * config.gap_width
  let long_side_length := long_side_posts * config.post_width + (long_side_posts - 1) * config.gap_width
  2 * (short_side_length + long_side_length)

/-- The main theorem stating that the fence configuration results in a perimeter of 236 feet --/
theorem fence_perimeter_is_236 :
  let config : FenceConfig := {
    total_posts := 36,
    post_width := 1/2,  -- 6 inches = 1/2 foot
    gap_width := 6,
    long_short_ratio := 3
  }
  calculate_perimeter config = 236 := by sorry


end NUMINAMATH_CALUDE_fence_perimeter_is_236_l4064_406409


namespace NUMINAMATH_CALUDE_new_student_weights_l4064_406435

/-- Proves that given the class size changes and average weights, the weights of the four new students are as calculated. -/
theorem new_student_weights
  (original_size : ℕ)
  (original_avg : ℝ)
  (avg_after_first : ℝ)
  (avg_after_second : ℝ)
  (avg_after_third : ℝ)
  (final_avg : ℝ)
  (h_original_size : original_size = 29)
  (h_original_avg : original_avg = 28)
  (h_avg_after_first : avg_after_first = 27.2)
  (h_avg_after_second : avg_after_second = 27.8)
  (h_avg_after_third : avg_after_third = 27.6)
  (h_final_avg : final_avg = 28) :
  ∃ (w1 w2 w3 w4 : ℝ),
    w1 = 4 ∧
    w2 = 45.8 ∧
    w3 = 21.4 ∧
    w4 = 40.8 ∧
    (original_size : ℝ) * original_avg + w1 = (original_size + 1 : ℝ) * avg_after_first ∧
    (original_size + 1 : ℝ) * avg_after_first + w2 = (original_size + 2 : ℝ) * avg_after_second ∧
    (original_size + 2 : ℝ) * avg_after_second + w3 = (original_size + 3 : ℝ) * avg_after_third ∧
    (original_size + 3 : ℝ) * avg_after_third + w4 = (original_size + 4 : ℝ) * final_avg :=
by
  sorry

end NUMINAMATH_CALUDE_new_student_weights_l4064_406435


namespace NUMINAMATH_CALUDE_monotone_decreasing_range_l4064_406472

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then (2*a - 1)*x + a
  else Real.log x / Real.log a

-- State the theorem
theorem monotone_decreasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂) ↔ 0 < a ∧ a ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_monotone_decreasing_range_l4064_406472


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l4064_406444

/-- Given two parallel vectors p and q, prove that their sum has magnitude √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) (h_parallel : p.1 * q.2 = p.2 * q.1) :
  p = (2, -3) → q.2 = 6 → ‖p + q‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l4064_406444


namespace NUMINAMATH_CALUDE_figure2_segment_length_l4064_406415

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Calculates the total length of visible segments after cutting a square from a rectangle -/
def visibleSegmentsLength (rect : Rectangle) (sq : Square) : ℝ :=
  rect.length + (rect.width - sq.side) + (rect.length - sq.side) + sq.side

/-- Theorem stating that the total length of visible segments in Figure 2 is 23 units -/
theorem figure2_segment_length :
  let rect : Rectangle := { length := 10, width := 6 }
  let sq : Square := { side := 3 }
  visibleSegmentsLength rect sq = 23 := by
  sorry

end NUMINAMATH_CALUDE_figure2_segment_length_l4064_406415


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l4064_406471

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (-2, 3)
  let b : ℝ × ℝ := (1, m - 3/2)
  are_parallel a b → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l4064_406471


namespace NUMINAMATH_CALUDE_gift_card_value_l4064_406494

theorem gift_card_value (coffee_price : ℝ) (pounds_bought : ℝ) (remaining_balance : ℝ) :
  coffee_price = 8.58 →
  pounds_bought = 4 →
  remaining_balance = 35.68 →
  coffee_price * pounds_bought + remaining_balance = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_gift_card_value_l4064_406494


namespace NUMINAMATH_CALUDE_parabola_C₃_expression_l4064_406430

/-- The parabola C₁ -/
def C₁ (x y : ℝ) : Prop := y = x^2 - 2*x + 3

/-- The parabola C₂, shifted 1 unit to the left from C₁ -/
def C₂ (x y : ℝ) : Prop := C₁ (x + 1) y

/-- The parabola C₃, symmetric to C₂ with respect to the y-axis -/
def C₃ (x y : ℝ) : Prop := C₂ (-x) y

/-- The theorem stating the analytical expression of C₃ -/
theorem parabola_C₃_expression : ∀ x y : ℝ, C₃ x y ↔ y = x^2 + 2 := by sorry

end NUMINAMATH_CALUDE_parabola_C₃_expression_l4064_406430


namespace NUMINAMATH_CALUDE_fraction_of_b_equal_to_third_of_a_prove_fraction_of_b_equal_to_third_of_a_l4064_406420

theorem fraction_of_b_equal_to_third_of_a : ℝ → ℝ → ℝ → Prop :=
  fun a b x =>
    a + b = 1210 →
    b = 484 →
    (1/3) * a = x * b →
    x = 1/2

-- Proof
theorem prove_fraction_of_b_equal_to_third_of_a :
  ∃ (a b x : ℝ), fraction_of_b_equal_to_third_of_a a b x :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_of_b_equal_to_third_of_a_prove_fraction_of_b_equal_to_third_of_a_l4064_406420


namespace NUMINAMATH_CALUDE_log_function_value_l4064_406421

-- Define the logarithmic function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_function_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (f a (1/8) = 3) → (f a (1/4) = 2) :=
by sorry

end NUMINAMATH_CALUDE_log_function_value_l4064_406421


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4064_406450

/-- Given that a² and √b vary inversely, a = 3 when b = 36, and ab = 108, prove that b = 36 -/
theorem inverse_variation_problem (a b : ℝ) (h1 : ∃ k : ℝ, a^2 * Real.sqrt b = k)
  (h2 : a = 3 ∧ b = 36) (h3 : a * b = 108) : b = 36 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4064_406450


namespace NUMINAMATH_CALUDE_locus_of_point_on_moving_segment_l4064_406402

/-- 
Given two perpendicular lines with moving points M and N, where the distance MN 
remains constant, and P is an arbitrary point on segment MN, prove that the locus 
of points P(x,y) forms an ellipse described by the equation x²/a² + y²/b² = 1, 
where a and b are constants related to the distance MN.
-/
theorem locus_of_point_on_moving_segment (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (M N P : ℝ × ℝ) (dist_MN : ℝ),
    (∀ t : ℝ, ∃ (Mt Nt : ℝ × ℝ), 
      (Mt.1 = 0 ∧ Nt.2 = 0) ∧  -- M and N move on perpendicular lines
      (Mt.2 - Nt.2)^2 + (Mt.1 - Nt.1)^2 = dist_MN^2 ∧  -- constant distance MN
      (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ 
        P = (s * Mt.1 + (1 - s) * Nt.1, s * Mt.2 + (1 - s) * Nt.2))) →  -- P on segment MN
    P.1^2 / a^2 + P.2^2 / b^2 = 1  -- locus is an ellipse
  := by sorry

end NUMINAMATH_CALUDE_locus_of_point_on_moving_segment_l4064_406402


namespace NUMINAMATH_CALUDE_original_station_count_l4064_406480

/-- The number of combinations of 2 items from a set of k items -/
def combinations (k : ℕ) : ℕ := k * (k - 1) / 2

/-- 
Given:
- m is the original number of stations
- n is the number of new stations added (n > 1)
- The increase in types of passenger tickets is 58

Prove that m = 14
-/
theorem original_station_count (m n : ℕ) 
  (h1 : n > 1) 
  (h2 : combinations (m + n) - combinations m = 58) : 
  m = 14 := by sorry

end NUMINAMATH_CALUDE_original_station_count_l4064_406480


namespace NUMINAMATH_CALUDE_x_equals_negative_x_and_abs_x_equals_two_l4064_406488

theorem x_equals_negative_x_and_abs_x_equals_two (x : ℝ) :
  (x = -x → x = 0) ∧ (|x| = 2 → x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_x_equals_negative_x_and_abs_x_equals_two_l4064_406488


namespace NUMINAMATH_CALUDE_base6_addition_theorem_l4064_406495

/-- Converts a base 6 number represented as a list of digits to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 represented as a list of digits -/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc else aux (n / 6) ((n % 6) :: acc)
    aux n []

/-- Adds two base 6 numbers represented as lists of digits -/
def addBase6 (a b : List Nat) : List Nat :=
  let sum := base6ToBase10 a + base6ToBase10 b
  base10ToBase6 sum

theorem base6_addition_theorem :
  let a := [2, 4, 5, 3]  -- 2453₆
  let b := [1, 6, 4, 3, 2]  -- 16432₆
  addBase6 a b = [2, 5, 5, 4, 5] ∧  -- 25545₆
  base6ToBase10 (addBase6 a b) = 3881 := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_theorem_l4064_406495


namespace NUMINAMATH_CALUDE_museum_visit_permutations_l4064_406441

theorem museum_visit_permutations :
  Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_museum_visit_permutations_l4064_406441


namespace NUMINAMATH_CALUDE_urn_probability_l4064_406407

/-- Represents the contents of the urn -/
structure UrnContents where
  red : ℕ
  blue : ℕ

/-- Represents a single operation of drawing and adding balls -/
inductive Operation
  | DrawRed
  | DrawBlue

/-- The initial state of the urn -/
def initial_urn : UrnContents := ⟨2, 1⟩

/-- Perform a single operation on the urn -/
def perform_operation (urn : UrnContents) (op : Operation) : UrnContents :=
  match op with
  | Operation.DrawRed => ⟨urn.red + 2, urn.blue⟩
  | Operation.DrawBlue => ⟨urn.red, urn.blue + 2⟩

/-- Perform a sequence of operations on the urn -/
def perform_operations (urn : UrnContents) (ops : List Operation) : UrnContents :=
  ops.foldl perform_operation urn

/-- Calculate the probability of a specific sequence of operations -/
def sequence_probability (ops : List Operation) : ℚ :=
  sorry

/-- Calculate the total probability of all valid sequences -/
def total_probability (valid_sequences : List (List Operation)) : ℚ :=
  sorry

theorem urn_probability : 
  ∃ (valid_sequences : List (List Operation)),
    (∀ seq ∈ valid_sequences, seq.length = 5) ∧
    (∀ seq ∈ valid_sequences, 
      let final_urn := perform_operations initial_urn seq
      final_urn.red + final_urn.blue = 12 ∧
      final_urn.red = 7 ∧ final_urn.blue = 5) ∧
    total_probability valid_sequences = 25 / 224 :=
  sorry

end NUMINAMATH_CALUDE_urn_probability_l4064_406407


namespace NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l4064_406452

theorem cucumber_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percentage : ℝ) 
  (final_water_percentage : ℝ) :
  initial_weight = 100 →
  initial_water_percentage = 0.99 →
  final_water_percentage = 0.95 →
  ∃ (final_weight : ℝ), 
    final_weight * (1 - final_water_percentage) = initial_weight * (1 - initial_water_percentage) ∧
    final_weight = 20 :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l4064_406452


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l4064_406443

theorem smallest_solution_of_equation (x : ℝ) :
  (((3 * x) / (x - 3)) + ((3 * x^2 - 27) / x) = 15) →
  (x ≥ -1 ∧ (∀ y : ℝ, y < -1 → ((3 * y) / (y - 3)) + ((3 * y^2 - 27) / y) ≠ 15)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l4064_406443


namespace NUMINAMATH_CALUDE_intersection_complement_A_and_B_l4064_406425

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- Define the complement of A in ℝ
def complementA : Set ℝ := {x : ℝ | x ≤ 3}

-- State the theorem
theorem intersection_complement_A_and_B :
  (complementA ∩ B) = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_A_and_B_l4064_406425


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4064_406408

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 7 → b = 24 → c^2 = a^2 + b^2 → c = 25 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4064_406408


namespace NUMINAMATH_CALUDE_quadrilateral_ratio_theorem_l4064_406496

-- Define the quadrilateral and points
variable (A B C D K L M N P : ℝ × ℝ)
variable (α β : ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def point_on_side (X Y Z : ℝ × ℝ) : Prop := sorry

def ratio_equals (A B X Y : ℝ × ℝ) (r : ℝ) : Prop := sorry

def intersection_point (K M L N P : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_ratio_theorem 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_K : point_on_side K A B)
  (h_L : point_on_side L B C)
  (h_M : point_on_side M C D)
  (h_N : point_on_side N D A)
  (h_AK_KB : ratio_equals A K K B α)
  (h_DM_MC : ratio_equals D M M C α)
  (h_BL_LC : ratio_equals B L L C β)
  (h_AN_ND : ratio_equals A N N D β)
  (h_P : intersection_point K M L N P) :
  ratio_equals N P P L α ∧ ratio_equals K P P M β := by sorry

end NUMINAMATH_CALUDE_quadrilateral_ratio_theorem_l4064_406496


namespace NUMINAMATH_CALUDE_fahrenheit_for_40_celsius_l4064_406426

-- Define the relationship between C and F
def celsius_to_fahrenheit (C F : ℝ) : Prop :=
  C = (5/9) * (F - 32)

-- Theorem statement
theorem fahrenheit_for_40_celsius :
  ∃ F : ℝ, celsius_to_fahrenheit 40 F ∧ F = 104 :=
by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_for_40_celsius_l4064_406426


namespace NUMINAMATH_CALUDE_circle_condition_l4064_406498

/-- The equation of a potential circle with a parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 4*y + m = 0

/-- A predicate to check if an equation represents a circle -/
def is_circle (m : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating the condition for the equation to represent a circle -/
theorem circle_condition (m : ℝ) : is_circle m ↔ m < 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l4064_406498


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l4064_406428

theorem friend_lunch_cost (total : ℕ) (difference : ℕ) (friend_cost : ℕ) : 
  total = 15 → difference = 5 → 
  (∃ (your_cost : ℕ), your_cost + friend_cost = total ∧ friend_cost = your_cost + difference) →
  friend_cost = 10 := by sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l4064_406428


namespace NUMINAMATH_CALUDE_max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l4064_406470

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits possible in a 24-hour format digital watch display -/
def maxSumOfDigits : Nat := 24

/-- Theorem stating that the maximum sum of digits in a 24-hour format digital watch display is 24 -/
theorem max_sum_of_digits_is_24 :
  ∀ t : Time24, timeSumOfDigits t ≤ maxSumOfDigits :=
sorry

/-- Theorem stating that there exists a time that achieves the maximum sum of digits -/
theorem max_sum_of_digits_is_achievable :
  ∃ t : Time24, timeSumOfDigits t = maxSumOfDigits :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l4064_406470


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l4064_406429

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l4064_406429


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l4064_406423

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 3}
def B : Set Nat := {3, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l4064_406423


namespace NUMINAMATH_CALUDE_square_root_squared_l4064_406461

theorem square_root_squared (x : ℝ) (h : x ≥ 0) : (Real.sqrt x) ^ 2 = x := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_l4064_406461


namespace NUMINAMATH_CALUDE_probability_of_mathematics_letter_l4064_406468

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of unique letters in "MATHEMATICS" -/
def unique_letters : ℕ := 8

/-- The probability of selecting a letter from "MATHEMATICS" -/
def probability : ℚ := unique_letters / alphabet_size

theorem probability_of_mathematics_letter :
  probability = 4 / 13 := by sorry

end NUMINAMATH_CALUDE_probability_of_mathematics_letter_l4064_406468


namespace NUMINAMATH_CALUDE_circle_center_correct_l4064_406479

/-- The polar equation of a circle -/
def polar_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

/-- The Cartesian equation of a circle -/
def cartesian_equation (x y : ℝ) : Prop := x^2 + y^2 = 4 * y

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (0, 2)

theorem circle_center_correct :
  (∀ ρ θ : ℝ, polar_equation ρ θ ↔ ∃ x y : ℝ, cartesian_equation x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (∀ x y : ℝ, cartesian_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 4) :=
sorry

end NUMINAMATH_CALUDE_circle_center_correct_l4064_406479


namespace NUMINAMATH_CALUDE_school_population_l4064_406489

/-- Given a school with boys, girls, and teachers, prove that the total number
    of people is 41b/32 when there are 4 times as many boys as girls and 8 times
    as many girls as teachers. -/
theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) :
  b + g + t = (41 * b) / 32 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l4064_406489
