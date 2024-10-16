import Mathlib

namespace NUMINAMATH_CALUDE_sum_plus_count_theorem_l917_91723

def sum_of_integers (a b : ℕ) : ℕ := ((b - a + 1) * (a + b)) / 2

def count_even_integers (a b : ℕ) : ℕ := ((b - a) / 2) + 1

theorem sum_plus_count_theorem : 
  sum_of_integers 50 70 + count_even_integers 50 70 = 1271 := by
  sorry

end NUMINAMATH_CALUDE_sum_plus_count_theorem_l917_91723


namespace NUMINAMATH_CALUDE_interior_angles_sum_l917_91749

/-- If the sum of the interior angles of a convex polygon with n sides is 1800°,
    then the sum of the interior angles of a convex polygon with n + 4 sides is 2520°. -/
theorem interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 1800) → (180 * ((n + 4) - 2) = 2520) := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l917_91749


namespace NUMINAMATH_CALUDE_union_M_complement_N_equals_R_l917_91784

-- Define the sets M and N
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | x^2 - x < 0}

-- State the theorem
theorem union_M_complement_N_equals_R : M ∪ (Set.univ \ N) = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_M_complement_N_equals_R_l917_91784


namespace NUMINAMATH_CALUDE_absolute_value_sum_l917_91759

theorem absolute_value_sum (y q : ℝ) (h1 : |y - 5| = q) (h2 : y > 5) : y + q = 2*q + 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l917_91759


namespace NUMINAMATH_CALUDE_min_odd_integers_l917_91730

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 22)
  (sum2 : a + b + c + d = 36)
  (sum3 : a + b + c + d + e + f = 50) :
  ∃ (a' b' c' d' e' f' : ℤ), 
    a' % 2 = 0 ∧ b' % 2 = 0 ∧ c' % 2 = 0 ∧ d' % 2 = 0 ∧ e' % 2 = 0 ∧ f' % 2 = 0 ∧
    a' + b' = 22 ∧
    a' + b' + c' + d' = 36 ∧
    a' + b' + c' + d' + e' + f' = 50 :=
by sorry

end NUMINAMATH_CALUDE_min_odd_integers_l917_91730


namespace NUMINAMATH_CALUDE_expression_evaluation_l917_91777

theorem expression_evaluation (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (2*x + y)^2 + (x + y)*(x - y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l917_91777


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l917_91703

theorem trig_expression_simplification (α : ℝ) :
  (Real.sin (π - α) * Real.sin (3 * π - α) + Real.sin (-α - π) * Real.sin (α - 2 * π)) /
  (Real.sin (4 * π - α) * Real.sin (5 * π + α)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l917_91703


namespace NUMINAMATH_CALUDE_probability_all_white_balls_l917_91729

def total_balls : ℕ := 15
def white_balls : ℕ := 7
def black_balls : ℕ := 8
def drawn_balls : ℕ := 7

theorem probability_all_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls) = 1 / 6435 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_white_balls_l917_91729


namespace NUMINAMATH_CALUDE_simplify_expression_l917_91711

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^3 + b^3 = 3*(a + b)) : a/b + b/a - 3/(a*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l917_91711


namespace NUMINAMATH_CALUDE_cube_product_equals_728_39_l917_91776

theorem cube_product_equals_728_39 : 
  (((4^3 - 1) / (4^3 + 1)) * 
   ((5^3 - 1) / (5^3 + 1)) * 
   ((6^3 - 1) / (6^3 + 1)) * 
   ((7^3 - 1) / (7^3 + 1)) * 
   ((8^3 - 1) / (8^3 + 1)) * 
   ((9^3 - 1) / (9^3 + 1))) = 728 / 39 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_equals_728_39_l917_91776


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l917_91738

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 7 * 17^2) :
  ∃ k : ℕ, k = 63 ∧ 
  (∀ m : ℕ, n^m ∣ n.factorial → m ≤ k) ∧
  n^k ∣ n.factorial :=
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l917_91738


namespace NUMINAMATH_CALUDE_mod_congruence_unique_solution_l917_91794

theorem mod_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -300 ≡ n [ZMOD 23] := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_solution_l917_91794


namespace NUMINAMATH_CALUDE_equation_holds_for_x_equals_three_l917_91743

theorem equation_holds_for_x_equals_three : 
  ∀ x : ℝ, x = 3 → 3 * x - 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_for_x_equals_three_l917_91743


namespace NUMINAMATH_CALUDE_least_number_with_conditions_l917_91719

theorem least_number_with_conditions : ∃ n : ℕ, 
  (n = 1262) ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 7 → n % k = 2) ∧
  (n % 13 = 0) ∧
  (∀ m : ℕ, m < n → ¬(∀ k : ℕ, 2 ≤ k ∧ k ≤ 7 → m % k = 2) ∨ m % 13 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_conditions_l917_91719


namespace NUMINAMATH_CALUDE_photo_arrangements_eq_288_l917_91735

/-- The number of ways to arrange teachers and students in a photo. -/
def photoArrangements (numTeachers numMaleStudents numFemaleStudents : ℕ) : ℕ :=
  2 * (numMaleStudents.factorial * numFemaleStudents.factorial * (numFemaleStudents + 1).choose numMaleStudents)

/-- Theorem stating the number of photo arrangements under given conditions. -/
theorem photo_arrangements_eq_288 :
  photoArrangements 2 3 3 = 288 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_eq_288_l917_91735


namespace NUMINAMATH_CALUDE_candy_challenge_solution_l917_91752

/-- Represents the candy-eating challenge over three days -/
def candy_challenge (initial_candies : ℚ) : Prop :=
  let day1_after_eating := (3/4) * initial_candies
  let day1_remaining := day1_after_eating - 3
  let day2_after_eating := (4/5) * day1_remaining
  let day2_remaining := day2_after_eating - 5
  day2_remaining = 10

theorem candy_challenge_solution :
  ∃ (x : ℚ), candy_challenge x ∧ x = 52 :=
sorry

end NUMINAMATH_CALUDE_candy_challenge_solution_l917_91752


namespace NUMINAMATH_CALUDE_negative_sqrt_three_is_quadratic_radical_l917_91766

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ a : ℝ, a ≥ 0 ∧ x = Real.sqrt a ∨ x = -Real.sqrt a

-- Theorem statement
theorem negative_sqrt_three_is_quadratic_radical :
  is_quadratic_radical (-Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_three_is_quadratic_radical_l917_91766


namespace NUMINAMATH_CALUDE_santiagos_number_l917_91745

theorem santiagos_number (amelia santiago : ℂ) : 
  amelia * santiago = 20 + 15 * Complex.I ∧ 
  amelia = 4 - 5 * Complex.I →
  santiago = (5 : ℚ) / 41 + (160 : ℚ) / 41 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_santiagos_number_l917_91745


namespace NUMINAMATH_CALUDE_max_square_plots_is_48_l917_91700

/-- Represents the dimensions and constraints of the field --/
structure FieldParameters where
  length : ℝ
  width : ℝ
  pathwayWidth : ℝ
  availableFencing : ℝ

/-- Calculates the maximum number of square test plots --/
def maxSquarePlots (params : FieldParameters) : ℕ :=
  sorry

/-- Theorem stating the maximum number of square test plots --/
theorem max_square_plots_is_48 (params : FieldParameters) :
  params.length = 45 ∧ 
  params.width = 30 ∧ 
  params.pathwayWidth = 5 ∧ 
  params.availableFencing = 2700 →
  maxSquarePlots params = 48 :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_is_48_l917_91700


namespace NUMINAMATH_CALUDE_screw_nut_production_l917_91753

theorem screw_nut_production (total_workers : ℕ) (screws_per_worker : ℕ) (nuts_per_worker : ℕ) 
  (screw_workers : ℕ) (nut_workers : ℕ) : 
  total_workers = 22 →
  screws_per_worker = 1200 →
  nuts_per_worker = 2000 →
  screw_workers = 10 →
  nut_workers = 12 →
  screw_workers + nut_workers = total_workers ∧
  2 * (screw_workers * screws_per_worker) = nut_workers * nuts_per_worker :=
by
  sorry

#check screw_nut_production

end NUMINAMATH_CALUDE_screw_nut_production_l917_91753


namespace NUMINAMATH_CALUDE_coefficient_b_value_l917_91733

-- Define the polynomial P(x)
def P (a b d c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + d*x + c

-- Define the sum of zeros
def sum_of_zeros (a : ℝ) : ℝ := -a

-- Define the product of zeros taken three at a time
def product_of_three_zeros (d : ℝ) : ℝ := d

-- Define the sum of coefficients
def sum_of_coefficients (a b d c : ℝ) : ℝ := 1 + a + b + d + c

-- State the theorem
theorem coefficient_b_value (a b d c : ℝ) :
  sum_of_zeros a = product_of_three_zeros d ∧
  sum_of_zeros a = sum_of_coefficients a b d c ∧
  P a b d c 0 = 8 →
  b = -17 := by sorry

end NUMINAMATH_CALUDE_coefficient_b_value_l917_91733


namespace NUMINAMATH_CALUDE_sum_between_14_and_14_half_l917_91748

theorem sum_between_14_and_14_half :
  let sum := (3 + 3/8) + (4 + 3/4) + (6 + 2/23)
  14 < sum ∧ sum < 14.5 := by
sorry

end NUMINAMATH_CALUDE_sum_between_14_and_14_half_l917_91748


namespace NUMINAMATH_CALUDE_complex_equation_solution_l917_91799

def i : ℂ := Complex.I

theorem complex_equation_solution (a : ℝ) :
  (2 + a * i) / (1 + Real.sqrt 2 * i) = -Real.sqrt 2 * i →
  a = -Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l917_91799


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l917_91796

-- Define the number of sibling pairs
def num_sibling_pairs : ℕ := 4

-- Define the number of seats in each row
def seats_per_row : ℕ := 4

-- Define the number of rows in the van
def num_rows : ℕ := 2

-- Define the derangement function for 4 objects
def derangement_4 : ℕ := 9

-- Theorem statement
theorem seating_arrangements_count :
  (seats_per_row.factorial) * derangement_4 * (2^num_sibling_pairs) = 3456 := by
  sorry


end NUMINAMATH_CALUDE_seating_arrangements_count_l917_91796


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_largest_inscribed_circle_radius_proof_l917_91755

/-- The radius of the largest inscribed circle in a square with side length 15,
    outside two congruent equilateral triangles sharing one side and each having
    one vertex on a vertex of the square. -/
theorem largest_inscribed_circle_radius : ℝ :=
  let square_side : ℝ := 15
  let triangle_side : ℝ := (square_side * Real.sqrt 6 - square_side * Real.sqrt 2) / 2
  let circle_radius : ℝ := square_side / 2 - (square_side * Real.sqrt 6 - square_side * Real.sqrt 2) / 8
  circle_radius

/-- Proof that the radius of the largest inscribed circle is correct. -/
theorem largest_inscribed_circle_radius_proof :
  largest_inscribed_circle_radius = 7.5 - (15 * Real.sqrt 6 - 15 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_largest_inscribed_circle_radius_proof_l917_91755


namespace NUMINAMATH_CALUDE_perpendicular_planes_line_parallel_l917_91769

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Line : Type
  Plane : Type
  parallel_line_plane : Line → Plane → Prop
  perpendicular_plane_plane : Plane → Plane → Prop
  perpendicular_line_plane : Line → Plane → Prop
  line_in_plane : Line → Plane → Prop

variable {S : Space3D}

/-- The main theorem -/
theorem perpendicular_planes_line_parallel 
  (α β : S.Plane) (m : S.Line)
  (h1 : S.perpendicular_plane_plane α β)
  (h2 : S.perpendicular_line_plane m β)
  (h3 : ¬ S.line_in_plane m α) :
  S.parallel_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_line_parallel_l917_91769


namespace NUMINAMATH_CALUDE_triangle_area_from_smaller_triangles_l917_91751

/-- Given a triangle divided into six parts by lines parallel to its sides,
    this theorem states that the area of the original triangle is equal to
    (√t₁ + √t₂ + √t₃)², where t₁, t₂, and t₃ are the areas of three of the
    smaller triangles formed. -/
theorem triangle_area_from_smaller_triangles 
  (t₁ t₂ t₃ : ℝ) 
  (h₁ : t₁ > 0) 
  (h₂ : t₂ > 0) 
  (h₃ : t₃ > 0) :
  ∃ T : ℝ, T > 0 ∧ T = (Real.sqrt t₁ + Real.sqrt t₂ + Real.sqrt t₃)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_smaller_triangles_l917_91751


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l917_91739

theorem solution_satisfies_system :
  let x : ℚ := 43 / 9
  let y : ℚ := 16 / 3
  (3 * x - 4 * y = -7) ∧ (6 * x - 5 * y = 2) := by
sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l917_91739


namespace NUMINAMATH_CALUDE_inequality_proof_l917_91798

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l917_91798


namespace NUMINAMATH_CALUDE_problem_solution_l917_91768

theorem problem_solution : (2^(1/2) * 4^(1/2)) + (18 / 3 * 3) - 8^(3/2) = 18 - 14 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l917_91768


namespace NUMINAMATH_CALUDE_total_distance_approx_l917_91728

/-- The speed at which Tammy drove, in miles per hour -/
def speed : ℝ := 1.527777778

/-- The duration of Tammy's drive, in hours -/
def duration : ℝ := 36.0

/-- The total distance Tammy drove, in miles -/
def total_distance : ℝ := speed * duration

/-- Theorem stating that the total distance is approximately 55.0 miles -/
theorem total_distance_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |total_distance - 55.0| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_total_distance_approx_l917_91728


namespace NUMINAMATH_CALUDE_no_solution_iff_m_leq_neg_one_l917_91781

theorem no_solution_iff_m_leq_neg_one (m : ℝ) :
  (∀ x : ℝ, ¬(x - m < 0 ∧ 3*x - 1 > 2*(x - 1))) ↔ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_leq_neg_one_l917_91781


namespace NUMINAMATH_CALUDE_stating_prob_three_students_same_group_l917_91793

/-- Represents the total number of students -/
def total_students : ℕ := 800

/-- Represents the number of lunch groups -/
def num_groups : ℕ := 4

/-- Represents the size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- Represents the probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- 
Theorem stating that the probability of three specific students 
being assigned to the same lunch group is 1/16
-/
theorem prob_three_students_same_group : 
  (prob_assigned_to_group * prob_assigned_to_group : ℚ) = 1 / 16 := by
  sorry

#check prob_three_students_same_group

end NUMINAMATH_CALUDE_stating_prob_three_students_same_group_l917_91793


namespace NUMINAMATH_CALUDE_gcd_102_238_l917_91742

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l917_91742


namespace NUMINAMATH_CALUDE_max_attachable_squares_l917_91765

/-- A unit square in 2D space -/
structure UnitSquare where
  center : ℝ × ℝ

/-- Represents the configuration of unit squares attached to a central square -/
structure SquareConfiguration where
  central : UnitSquare
  attached : List UnitSquare

/-- Checks if two unit squares overlap -/
def squaresOverlap (s1 s2 : UnitSquare) : Prop := sorry

/-- Checks if a configuration is valid (no overlaps) -/
def isValidConfiguration (config : SquareConfiguration) : Prop := sorry

/-- The main theorem: maximum number of attachable squares is 8 -/
theorem max_attachable_squares (K : UnitSquare) :
  (∃ (config : SquareConfiguration),
    config.central = K ∧
    isValidConfiguration config ∧
    config.attached.length = 8) ∧
  (∀ (config : SquareConfiguration),
    config.central = K →
    isValidConfiguration config →
    config.attached.length ≤ 8) := by sorry

end NUMINAMATH_CALUDE_max_attachable_squares_l917_91765


namespace NUMINAMATH_CALUDE_geometric_number_difference_l917_91746

/-- A function that checks if a 3-digit number has distinct digits forming a geometric sequence --/
def is_geometric_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    b * b = a * c

/-- The largest 3-digit number with distinct digits forming a geometric sequence --/
def largest_geometric_number : ℕ := 964

/-- The smallest 3-digit number with distinct digits forming a geometric sequence --/
def smallest_geometric_number : ℕ := 124

theorem geometric_number_difference :
  is_geometric_number largest_geometric_number ∧
  is_geometric_number smallest_geometric_number ∧
  (∀ n : ℕ, is_geometric_number n → 
    smallest_geometric_number ≤ n ∧ n ≤ largest_geometric_number) ∧
  largest_geometric_number - smallest_geometric_number = 840 := by
  sorry

end NUMINAMATH_CALUDE_geometric_number_difference_l917_91746


namespace NUMINAMATH_CALUDE_nines_count_to_500_l917_91775

/-- Count of digit 9 appearances in a number -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Sum of digit 9 appearances in a range of numbers -/
def sum_nines (start finish : ℕ) : ℕ := sorry

/-- The count of digit 9 appearances in all integers from 1 to 500 is 100 -/
theorem nines_count_to_500 : sum_nines 1 500 = 100 := by sorry

end NUMINAMATH_CALUDE_nines_count_to_500_l917_91775


namespace NUMINAMATH_CALUDE_no_condition_satisfies_equations_l917_91708

theorem no_condition_satisfies_equations (a b c : ℤ) : 
  a + b + c = 3 →
  (∀ (condition : Prop), 
    (condition = (a = b ∧ b = c ∧ c = 1) ∨
     condition = (a = b - 1 ∧ b = c - 1) ∨
     condition = (a = b ∧ b = c) ∨
     condition = (a > c ∧ c = b - 1)) →
    ¬(condition → a*(a-b)^3 + b*(b-c)^3 + c*(c-a)^3 = 3)) :=
by sorry

end NUMINAMATH_CALUDE_no_condition_satisfies_equations_l917_91708


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l917_91704

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 1 / 2 →
  n + k = 9 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l917_91704


namespace NUMINAMATH_CALUDE_memory_cell_increment_prime_or_one_l917_91763

def memory_cell_sequence (k : ℕ) : ℕ → ℕ
  | 0 => 6
  | (n + 1) => memory_cell_sequence k n + Nat.gcd (memory_cell_sequence k n) (n + 1)

theorem memory_cell_increment_prime_or_one (k : ℕ) (hk : k ≤ 1000000) :
  ∀ n ≤ k, (memory_cell_sequence k n) - (memory_cell_sequence k (n - 1)) = 1 ∨
           Nat.Prime ((memory_cell_sequence k n) - (memory_cell_sequence k (n - 1))) :=
by sorry

end NUMINAMATH_CALUDE_memory_cell_increment_prime_or_one_l917_91763


namespace NUMINAMATH_CALUDE_largest_number_l917_91779

theorem largest_number (a b c : ℝ) : 
  a = (1 : ℝ) / 2 →
  b = Real.log 3 / Real.log 4 →
  c = Real.sin (π / 8) →
  b ≥ a ∧ b ≥ c :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l917_91779


namespace NUMINAMATH_CALUDE_billy_homework_problem_l917_91788

theorem billy_homework_problem (first_hour second_hour third_hour total : ℕ) : 
  first_hour > 0 →
  second_hour = 2 * first_hour →
  third_hour = 3 * first_hour →
  third_hour = 132 →
  total = first_hour + second_hour + third_hour →
  total = 264 := by
  sorry

end NUMINAMATH_CALUDE_billy_homework_problem_l917_91788


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l917_91721

theorem magical_red_knights_fraction (total : ℕ) (red : ℕ) (blue : ℕ) (magical : ℕ) 
  (h1 : red = (3 * total) / 7)
  (h2 : blue = total - red)
  (h3 : magical = total / 4)
  (h4 : ∃ (r s : ℕ), (r * blue * 3 = s * red) ∧ (r * red + r * blue = s * magical)) :
  ∃ (r s : ℕ), (r * red = s * magical) ∧ (r = 21 ∧ s = 52) :=
sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l917_91721


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l917_91710

theorem min_distance_to_origin (x y : ℝ) : 
  3 * x + 4 * y = 24 → 
  x - 2 * y = 0 → 
  ∃ (min_dist : ℝ), 
    min_dist = Real.sqrt 28.8 ∧ 
    ∀ (x' y' : ℝ), 3 * x' + 4 * y' = 24 → x' - 2 * y' = 0 → 
      Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_dist := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l917_91710


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l917_91712

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im (i * (i - 3)) = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l917_91712


namespace NUMINAMATH_CALUDE_digit_equation_solution_l917_91731

/-- Represents a base-ten digit -/
def Digit := Fin 10

/-- Checks if three digits are all different -/
def all_different (d1 d2 d3 : Digit) : Prop :=
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- Converts a pair of digits to a two-digit number -/
def to_two_digit (tens ones : Digit) : Nat :=
  10 * tens.val + ones.val

/-- Converts a digit to a three-digit number with all digits the same -/
def to_three_digit (d : Digit) : Nat :=
  111 * d.val

theorem digit_equation_solution :
  ∃ (V E A : Digit),
    all_different V E A ∧
    (to_two_digit V E) * (to_two_digit A E) = to_three_digit A ∧
    E.val + A.val + A.val + V.val = 26 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l917_91731


namespace NUMINAMATH_CALUDE_roller_coaster_runs_l917_91713

def people_in_line : ℕ := 1532
def num_cars : ℕ := 8
def seats_per_car : ℕ := 3

def capacity_per_ride : ℕ := num_cars * seats_per_car

theorem roller_coaster_runs : 
  ∃ (runs : ℕ), runs * capacity_per_ride ≥ people_in_line ∧ 
  ∀ (k : ℕ), k * capacity_per_ride ≥ people_in_line → k ≥ runs :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_runs_l917_91713


namespace NUMINAMATH_CALUDE_interval_intersection_l917_91715

theorem interval_intersection (x : ℝ) : (|5 - x| < 5 ∧ x^2 < 25) ↔ (0 < x ∧ x < 5) := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l917_91715


namespace NUMINAMATH_CALUDE_rational_sqrt_two_equation_l917_91756

theorem rational_sqrt_two_equation (x y : ℚ) (h : x + Real.sqrt 2 * y = 0) : x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt_two_equation_l917_91756


namespace NUMINAMATH_CALUDE_octagon_side_length_l917_91705

/-- Given a regular pentagon with side length 16 cm, prove that if the same total length of yarn
    is used to make a regular octagon, then the length of one side of the octagon is 10 cm. -/
theorem octagon_side_length (pentagon_side : ℝ) (octagon_side : ℝ) : 
  pentagon_side = 16 → 5 * pentagon_side = 8 * octagon_side → octagon_side = 10 := by
  sorry

end NUMINAMATH_CALUDE_octagon_side_length_l917_91705


namespace NUMINAMATH_CALUDE_fraction_equality_l917_91789

/-- Given two integers A and B satisfying the equation for all real x except 0, 3, and roots of x^2 + 2x + 1 = 0, prove that B/A = 0 -/
theorem fraction_equality (A B : ℤ) 
  (h : ∀ x : ℝ, x ≠ 0 → x ≠ 3 → x^2 + 2*x + 1 ≠ 0 → 
    (A / (x - 3) : ℝ) + (B / (x^2 + 2*x + 1) : ℝ) = (x^3 - x^2 + 3*x + 1) / (x^3 - x - 3)) : 
  (B : ℚ) / A = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l917_91789


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l917_91744

theorem smallest_integer_with_remainders : ∃! M : ℕ,
  M > 0 ∧
  M % 4 = 3 ∧
  M % 5 = 4 ∧
  M % 6 = 5 ∧
  M % 7 = 6 ∧
  ∀ n : ℕ, n > 0 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 → M ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l917_91744


namespace NUMINAMATH_CALUDE_face_mask_cost_per_box_l917_91782

/-- Represents the problem of calculating the cost per box of face masks --/
theorem face_mask_cost_per_box :
  ∀ (total_boxes : ℕ) 
    (masks_per_box : ℕ) 
    (repacked_boxes : ℕ) 
    (repacked_price : ℚ) 
    (repacked_quantity : ℕ) 
    (remaining_masks : ℕ) 
    (baggie_price : ℚ) 
    (baggie_quantity : ℕ) 
    (profit : ℚ),
  total_boxes = 12 →
  masks_per_box = 50 →
  repacked_boxes = 6 →
  repacked_price = 5 →
  repacked_quantity = 25 →
  remaining_masks = 300 →
  baggie_price = 3 →
  baggie_quantity = 10 →
  profit = 42 →
  ∃ (cost_per_box : ℚ),
    cost_per_box = 9 ∧
    (repacked_boxes * masks_per_box / repacked_quantity * repacked_price +
     remaining_masks / baggie_quantity * baggie_price) - 
    (total_boxes * cost_per_box) = profit :=
by sorry


end NUMINAMATH_CALUDE_face_mask_cost_per_box_l917_91782


namespace NUMINAMATH_CALUDE_five_circles_theorem_l917_91727

/-- A circle in a plane -/
structure Circle where
  -- We don't need to define the internal structure of a circle for this problem

/-- A point in a plane -/
structure Point where
  -- We don't need to define the internal structure of a point for this problem

/-- Predicate to check if a point is on a circle -/
def PointOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a point is common to a list of circles -/
def CommonPoint (p : Point) (circles : List Circle) : Prop :=
  ∀ c ∈ circles, PointOnCircle p c

theorem five_circles_theorem (circles : List Circle) :
  circles.length = 5 →
  (∀ (subset : List Circle), subset.length = 4 ∧ subset ⊆ circles →
    ∃ (p : Point), CommonPoint p subset) →
  ∃ (p : Point), CommonPoint p circles := by
  sorry

end NUMINAMATH_CALUDE_five_circles_theorem_l917_91727


namespace NUMINAMATH_CALUDE_jerry_zinc_consumption_l917_91750

/-- The amount of zinc Jerry eats from antacids -/
def zinc_consumed (big_antacid_weight : ℝ) (big_antacid_count : ℕ) (big_antacid_zinc_percent : ℝ)
                  (small_antacid_weight : ℝ) (small_antacid_count : ℕ) (small_antacid_zinc_percent : ℝ) : ℝ :=
  (big_antacid_weight * big_antacid_count * big_antacid_zinc_percent +
   small_antacid_weight * small_antacid_count * small_antacid_zinc_percent) * 1000

/-- Theorem stating the amount of zinc Jerry consumes -/
theorem jerry_zinc_consumption :
  zinc_consumed 2 2 0.05 1 3 0.15 = 650 := by
  sorry

end NUMINAMATH_CALUDE_jerry_zinc_consumption_l917_91750


namespace NUMINAMATH_CALUDE_factor_polynomial_l917_91717

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l917_91717


namespace NUMINAMATH_CALUDE_fish_cost_is_80_l917_91741

/-- The cost of fish per kilogram in pesos -/
def fish_cost : ℝ := sorry

/-- The cost of pork per kilogram in pesos -/
def pork_cost : ℝ := sorry

/-- First condition: 530 pesos can buy 4 kg of fish and 2 kg of pork -/
axiom condition1 : 4 * fish_cost + 2 * pork_cost = 530

/-- Second condition: 875 pesos can buy 7 kg of fish and 3 kg of pork -/
axiom condition2 : 7 * fish_cost + 3 * pork_cost = 875

/-- Theorem: The cost of a kilogram of fish is 80 pesos -/
theorem fish_cost_is_80 : fish_cost = 80 := by sorry

end NUMINAMATH_CALUDE_fish_cost_is_80_l917_91741


namespace NUMINAMATH_CALUDE_kittens_and_mice_count_l917_91772

/-- The number of children carrying baskets -/
def num_children : ℕ := 12

/-- The number of baskets each child carries -/
def baskets_per_child : ℕ := 3

/-- The number of cats in each basket -/
def cats_per_basket : ℕ := 1

/-- The number of kittens each cat has -/
def kittens_per_cat : ℕ := 12

/-- The number of mice each kitten carries -/
def mice_per_kitten : ℕ := 4

/-- The total number of kittens and mice carried by the children -/
def total_kittens_and_mice : ℕ :=
  let total_baskets := num_children * baskets_per_child
  let total_cats := total_baskets * cats_per_basket
  let total_kittens := total_cats * kittens_per_cat
  let total_mice := total_kittens * mice_per_kitten
  total_kittens + total_mice

theorem kittens_and_mice_count : total_kittens_and_mice = 2160 := by
  sorry

end NUMINAMATH_CALUDE_kittens_and_mice_count_l917_91772


namespace NUMINAMATH_CALUDE_work_done_circular_path_l917_91791

/-- The work done by a force field on a mass point moving along a circular path -/
theorem work_done_circular_path (m a : ℝ) (h : a > 0) : 
  let force (x y : ℝ) := (x + y, -x)
  let path (t : ℝ) := (a * Real.cos t, -a * Real.sin t)
  let work := ∫ t in (0)..(2 * Real.pi), 
    m * (force (path t).1 (path t).2).1 * (-a * Real.sin t) + 
    m * (force (path t).1 (path t).2).2 * (-a * Real.cos t)
  work = 0 :=
sorry

end NUMINAMATH_CALUDE_work_done_circular_path_l917_91791


namespace NUMINAMATH_CALUDE_clothing_production_l917_91780

theorem clothing_production (fabric_A B : ℝ) (sets_B : ℕ) : 
  (fabric_A + 2 * B = 5) →
  (3 * fabric_A + B = 7) →
  (∀ m : ℕ, m + sets_B = 100 → fabric_A * m + B * sets_B ≤ 168) →
  (fabric_A = 1.8 ∧ B = 1.6 ∧ sets_B ≥ 60) :=
by sorry

end NUMINAMATH_CALUDE_clothing_production_l917_91780


namespace NUMINAMATH_CALUDE_chimney_bricks_count_l917_91761

/-- Represents the time taken by Brenda to build the chimney alone -/
def brenda_time : ℝ := 6

/-- Represents the time taken by Brandon to build the chimney alone -/
def brandon_time : ℝ := 8

/-- Represents the reduction in combined output when working together -/
def output_reduction : ℝ := 15

/-- Represents the time taken to build the chimney when working together -/
def combined_time : ℝ := 4

/-- Represents the number of bricks in the chimney -/
def chimney_bricks : ℝ := 360

theorem chimney_bricks_count :
  (combined_time * (chimney_bricks / brenda_time + chimney_bricks / brandon_time - output_reduction) = chimney_bricks) :=
sorry

end NUMINAMATH_CALUDE_chimney_bricks_count_l917_91761


namespace NUMINAMATH_CALUDE_illegal_parking_percentage_l917_91722

theorem illegal_parking_percentage
  (total_cars : ℝ)
  (towed_percentage : ℝ)
  (not_towed_percentage : ℝ)
  (h1 : towed_percentage = 0.02)
  (h2 : not_towed_percentage = 0.80)
  (h3 : total_cars > 0) :
  let towed_cars := towed_percentage * total_cars
  let illegally_parked_cars := towed_cars / (1 - not_towed_percentage)
  illegally_parked_cars / total_cars = 0.10 := by
sorry

end NUMINAMATH_CALUDE_illegal_parking_percentage_l917_91722


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_neg_four_squared_l917_91774

theorem arithmetic_sqrt_of_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_neg_four_squared_l917_91774


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l917_91758

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l917_91758


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l917_91778

theorem largest_n_satisfying_inequality :
  ∀ n : ℤ, (1 / 4 : ℚ) + (n : ℚ) / 5 < 7 / 4 ↔ n ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l917_91778


namespace NUMINAMATH_CALUDE_triangle_sin_A_l917_91740

theorem triangle_sin_A (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (a > 0) → (b > 0) → (c > 0) →
  -- cos B = -1/4
  (Real.cos B = -1/4) →
  -- a = 6
  (a = 6) →
  -- Area of triangle ABC is 3√15
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 15) →
  -- Then sin A = 3√15/16
  Real.sin A = (3 * Real.sqrt 15) / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sin_A_l917_91740


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l917_91716

theorem quadratic_roots_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    |x₁ - x₂| = 2) →
  p = 2 * Real.sqrt (q + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l917_91716


namespace NUMINAMATH_CALUDE_min_squares_to_exceed_500_l917_91732

def square (n : ℕ) : ℕ := n * n

def repeated_square (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => square (repeated_square n k)

theorem min_squares_to_exceed_500 :
  (∃ k : ℕ, repeated_square 2 k > 500) ∧
  (∀ k : ℕ, k < 4 → repeated_square 2 k ≤ 500) ∧
  (repeated_square 2 4 > 500) :=
by sorry

end NUMINAMATH_CALUDE_min_squares_to_exceed_500_l917_91732


namespace NUMINAMATH_CALUDE_smallest_element_100th_set_l917_91790

/-- Defines the smallest element of the nth set in the sequence -/
def smallest_element (n : ℕ) : ℕ := 
  (n - 1) * (n + 2) / 2 + 1

/-- The sequence of sets where the nth set contains n+1 consecutive integers -/
def set_sequence (n : ℕ) : Set ℕ :=
  {k : ℕ | smallest_element n ≤ k ∧ k < smallest_element (n + 1)}

/-- Theorem stating that the smallest element of the 100th set is 5050 -/
theorem smallest_element_100th_set : 
  smallest_element 100 = 5050 := by sorry

end NUMINAMATH_CALUDE_smallest_element_100th_set_l917_91790


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l917_91786

open Set

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {1, 2, 3}
def B : Finset Nat := {1, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l917_91786


namespace NUMINAMATH_CALUDE_negative_reciprocal_inequality_l917_91787

theorem negative_reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  -1/a < -1/b := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_inequality_l917_91787


namespace NUMINAMATH_CALUDE_tangent_line_equation_l917_91757

def parabola (x : ℝ) : ℝ := x^2 + x + 1

theorem tangent_line_equation :
  let f := parabola
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m := (deriv f) x₀
  ∀ x y, y - y₀ = m * (x - x₀) ↔ x - y + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l917_91757


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l917_91764

theorem quadratic_inequality_solution_condition (d : ℝ) :
  d > 0 ∧ (∃ x : ℝ, x^2 - 8*x + d < 0) ↔ 0 < d ∧ d < 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l917_91764


namespace NUMINAMATH_CALUDE_units_digit_of_j_squared_plus_three_to_j_l917_91702

/-- Given j = 2017^3 + 3^2017 - 1, prove that j^2 + 3^j ≡ 8 (mod 10) -/
theorem units_digit_of_j_squared_plus_three_to_j (j : ℕ) : 
  j = 2017^3 + 3^2017 - 1 → (j^2 + 3^j) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_j_squared_plus_three_to_j_l917_91702


namespace NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l917_91718

-- Define a circle with radius R
variable (R : ℝ) (hR : R > 0)

-- Define an inscribed rectangle with side lengths x and y
def inscribed_rectangle (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x^2 + y^2 = (2*R)^2

-- Define the area of a rectangle
def rectangle_area (x y : ℝ) : ℝ := x * y

-- Theorem: The area of any inscribed rectangle is less than or equal to 2R^2
theorem max_area_inscribed_rectangle (x y : ℝ) 
  (h : inscribed_rectangle R x y) : rectangle_area x y ≤ 2 * R^2 := by
  sorry

-- Note: The actual proof is omitted and replaced with 'sorry'

end NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l917_91718


namespace NUMINAMATH_CALUDE_unique_modular_residue_l917_91724

theorem unique_modular_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n < 11 ∧ -1234 ≡ n [ZMOD 11] :=
by sorry

end NUMINAMATH_CALUDE_unique_modular_residue_l917_91724


namespace NUMINAMATH_CALUDE_unique_solution_is_four_l917_91771

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem unique_solution_is_four :
  ∃! x : ℝ, 2 * (f x) - 19 = f (x - 4) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_four_l917_91771


namespace NUMINAMATH_CALUDE_sum_positive_implies_one_positive_l917_91797

theorem sum_positive_implies_one_positive (a b : ℝ) : a + b > 0 → a > 0 ∨ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_one_positive_l917_91797


namespace NUMINAMATH_CALUDE_inverse_of_5_mod_31_l917_91714

theorem inverse_of_5_mod_31 : ∃ x : ℕ, x ≤ 30 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_inverse_of_5_mod_31_l917_91714


namespace NUMINAMATH_CALUDE_doll_production_theorem_l917_91720

/-- The number of non-defective dolls produced per day -/
def non_defective_dolls : ℕ := 4800

/-- The ratio of total dolls to non-defective dolls -/
def total_to_non_defective_ratio : ℚ := 133 / 100

/-- The total number of dolls produced per day -/
def total_dolls : ℕ := 6384

/-- Theorem stating the relationship between non-defective dolls, the ratio, and total dolls -/
theorem doll_production_theorem :
  (non_defective_dolls : ℚ) * total_to_non_defective_ratio = total_dolls := by
  sorry

end NUMINAMATH_CALUDE_doll_production_theorem_l917_91720


namespace NUMINAMATH_CALUDE_inevitable_not_random_l917_91725

-- Define the Event type
inductive Event
| Random : Event
| Inevitable : Event
| Impossible : Event

-- Define properties of events
def mayOccur (e : Event) : Prop :=
  match e with
  | Event.Random => true
  | Event.Inevitable => true
  | Event.Impossible => false

def willDefinitelyOccur (e : Event) : Prop :=
  match e with
  | Event.Inevitable => true
  | _ => false

-- Theorem: An inevitable event is not a random event
theorem inevitable_not_random (e : Event) :
  willDefinitelyOccur e → e ≠ Event.Random := by
  sorry

end NUMINAMATH_CALUDE_inevitable_not_random_l917_91725


namespace NUMINAMATH_CALUDE_bus_driver_hours_l917_91726

-- Define constants
def regular_rate : ℝ := 15
def regular_hours : ℝ := 40
def overtime_rate_factor : ℝ := 1.75
def total_compensation : ℝ := 976

-- Define functions
def overtime_rate : ℝ := regular_rate * overtime_rate_factor

def total_hours (overtime_hours : ℝ) : ℝ :=
  regular_hours + overtime_hours

def compensation (overtime_hours : ℝ) : ℝ :=
  regular_rate * regular_hours + overtime_rate * overtime_hours

-- Theorem to prove
theorem bus_driver_hours :
  ∃ (overtime_hours : ℝ),
    compensation overtime_hours = total_compensation ∧
    total_hours overtime_hours = 54 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_hours_l917_91726


namespace NUMINAMATH_CALUDE_tim_running_hours_l917_91736

def days_per_week : ℕ := 7

def previous_running_days : ℕ := 3
def added_running_days : ℕ := 2
def hours_per_run : ℕ := 2

def total_running_days : ℕ := previous_running_days + added_running_days
def total_running_hours : ℕ := total_running_days * hours_per_run

theorem tim_running_hours : total_running_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_tim_running_hours_l917_91736


namespace NUMINAMATH_CALUDE_expand_polynomial_l917_91737

theorem expand_polynomial (x : ℝ) : (3*x^2 + 7*x + 4) * (5*x - 2) = 15*x^3 + 29*x^2 + 6*x - 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l917_91737


namespace NUMINAMATH_CALUDE_cookies_problem_l917_91760

theorem cookies_problem (millie mike frank : ℕ) : 
  millie = 4 →
  mike = 3 * millie →
  frank = mike / 2 - 3 →
  frank = 3 := by
sorry

end NUMINAMATH_CALUDE_cookies_problem_l917_91760


namespace NUMINAMATH_CALUDE_twenty_fifth_term_of_sequence_l917_91762

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twenty_fifth_term_of_sequence : 
  let a₁ := 2
  let a₂ := 5
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 25 = 74 := by
sorry

end NUMINAMATH_CALUDE_twenty_fifth_term_of_sequence_l917_91762


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_three_l917_91785

theorem greatest_common_divisor_of_three (n : ℕ) : 
  (∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
    {d : ℕ | d ∣ 180 ∧ d ∣ n} = {d1, d2, d3}) →
  (Nat.gcd 180 n = 9) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_three_l917_91785


namespace NUMINAMATH_CALUDE_factor_implies_a_value_l917_91747

theorem factor_implies_a_value (a b : ℤ) (x : ℝ) :
  (∀ x, (x^2 - x - 1) ∣ (a*x^17 + b*x^16 + 1)) →
  a = 987 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_a_value_l917_91747


namespace NUMINAMATH_CALUDE_minimum_spend_equal_fruits_l917_91795

/-- Represents a fruit set with apples, oranges, and cost -/
structure FruitSet where
  apples : ℕ
  oranges : ℕ
  cost : ℕ

/-- Calculates the total cost of buying multiple fruit sets -/
def totalCost (set : FruitSet) (quantity : ℕ) : ℕ :=
  set.cost * quantity

/-- Calculates the total number of apples in multiple fruit sets -/
def totalApples (set : FruitSet) (quantity : ℕ) : ℕ :=
  set.apples * quantity

/-- Calculates the total number of oranges in multiple fruit sets -/
def totalOranges (set : FruitSet) (quantity : ℕ) : ℕ :=
  set.oranges * quantity

theorem minimum_spend_equal_fruits : 
  let set1 : FruitSet := ⟨3, 15, 360⟩
  let set2 : FruitSet := ⟨20, 5, 500⟩
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧
    totalApples set1 x + totalApples set2 y = totalOranges set1 x + totalOranges set2 y ∧
    ∀ (a b : ℕ), 
      (a > 0 ∧ b > 0 ∧ 
       totalApples set1 a + totalApples set2 b = totalOranges set1 a + totalOranges set2 b) →
      totalCost set1 x + totalCost set2 y ≤ totalCost set1 a + totalCost set2 b ∧
    totalCost set1 x + totalCost set2 y = 3800 :=
by
  sorry


end NUMINAMATH_CALUDE_minimum_spend_equal_fruits_l917_91795


namespace NUMINAMATH_CALUDE_new_students_count_l917_91706

/-- Represents the problem of calculating the number of new students joining a school --/
theorem new_students_count (initial_avg_age initial_count new_students_avg_age final_avg_age final_count : ℕ) : 
  initial_avg_age = 48 →
  new_students_avg_age = 32 →
  final_avg_age = 44 →
  final_count = 160 →
  ∃ new_students : ℕ,
    new_students = 40 ∧
    final_count = initial_count + new_students ∧
    final_avg_age * final_count = initial_avg_age * initial_count + new_students_avg_age * new_students :=
by sorry

end NUMINAMATH_CALUDE_new_students_count_l917_91706


namespace NUMINAMATH_CALUDE_problem_statement_l917_91701

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x else Real.log x / Real.log 0.2

-- Theorem statement
theorem problem_statement (a : ℝ) (h : f (a + 5) = -1) : f a = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l917_91701


namespace NUMINAMATH_CALUDE_percentage_reduction_l917_91767

theorem percentage_reduction (P : ℝ) : (200 * (P / 100)) - 12 = 178 → P = 95 := by
  sorry

end NUMINAMATH_CALUDE_percentage_reduction_l917_91767


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l917_91792

/-- Given a function f(x) = a*tan³(x) - b*sin(3x) + cx + 7 where f(1) = 14, 
    prove that f(-1) = 0 -/
theorem function_value_at_negative_one 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * (Real.tan x)^3 - b * Real.sin (3 * x) + c * x + 7)
  (h2 : f 1 = 14) : 
  f (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l917_91792


namespace NUMINAMATH_CALUDE_triangle_area_l917_91783

/-- Given a triangle ABC with side lengths AB = 1 and BC = 3, and the dot product of vectors AB and BC equal to -1, 
    prove that the area of the triangle is √2. -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let AB := ((B.1 - A.1), (B.2 - A.2))
  let BC := ((C.1 - B.1), (C.2 - B.2))
  (AB.1^2 + AB.2^2 = 1) →
  (BC.1^2 + BC.2^2 = 9) →
  (AB.1 * BC.1 + AB.2 * BC.2 = -1) →
  (abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) / 2 = Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_area_l917_91783


namespace NUMINAMATH_CALUDE_distance_to_focus_l917_91707

def parabola (x y : ℝ) : Prop := y^2 = 4*x

def focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

theorem distance_to_focus (x y : ℝ) (h1 : parabola x y) (h2 : x = 3) : 
  ∃ (fx fy : ℝ), focus fx fy ∧ Real.sqrt ((x - fx)^2 + (y - fy)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_distance_to_focus_l917_91707


namespace NUMINAMATH_CALUDE_delta_f_P0_approx_df_P0_l917_91709

-- Define the function f
def f (x y : ℝ) : ℝ := x^2 * y

-- Define the point P0
def P0 : ℝ × ℝ := (5, 4)

-- Define Δx and Δy
def Δx : ℝ := 0.1
def Δy : ℝ := -0.2

-- Theorem for Δf(P0)
theorem delta_f_P0_approx : 
  let (x0, y0) := P0
  abs (f (x0 + Δx) (y0 + Δy) - f x0 y0 + 1.162) < 0.001 := by sorry

-- Theorem for df(P0)
theorem df_P0 : 
  let (x0, y0) := P0
  (2 * x0 * y0) * Δx + x0^2 * Δy = -1 := by sorry

end NUMINAMATH_CALUDE_delta_f_P0_approx_df_P0_l917_91709


namespace NUMINAMATH_CALUDE_factor_sum_18_with_2_l917_91754

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem factor_sum_18_with_2 (x : ℕ) 
  (h1 : x > 0) 
  (h2 : sum_of_factors x = 18) 
  (h3 : 2 ∣ x) : 
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_factor_sum_18_with_2_l917_91754


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l917_91770

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) / n = 156 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l917_91770


namespace NUMINAMATH_CALUDE_x_value_l917_91773

theorem x_value (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hx : x = y + 7) : x = 134 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l917_91773


namespace NUMINAMATH_CALUDE_theater_ticket_pricing_l917_91734

/-- Theorem: Theater Ticket Pricing --/
theorem theater_ticket_pricing
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (orchestra_cost : ℕ)
  (balcony_surplus : ℕ)
  (h1 : total_tickets = 370)
  (h2 : total_cost = 3320)
  (h3 : orchestra_cost = 12)
  (h4 : balcony_surplus = 190)
  : ∃ (balcony_cost : ℕ),
    balcony_cost = 8 ∧
    balcony_cost * (total_tickets - (total_tickets - balcony_surplus) / 2) +
    orchestra_cost * ((total_tickets - balcony_surplus) / 2) = total_cost :=
by sorry


end NUMINAMATH_CALUDE_theater_ticket_pricing_l917_91734
