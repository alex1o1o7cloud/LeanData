import Mathlib

namespace NUMINAMATH_CALUDE_number_division_problem_l2898_289879

theorem number_division_problem (x : ℝ) : (x + 17) / 5 = 25 → x / 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2898_289879


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2898_289830

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  ((a = 4 ∧ b = 5) ∨ (a = 4 ∧ c = 5) ∨ (b = 4 ∧ c = 5)) →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c = 3 ∨ c = Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2898_289830


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l2898_289886

theorem quadratic_equations_common_root (a b : ℝ) : 
  (∃! x, x^2 + a*x + b = 0 ∧ x^2 + b*x + a = 0) → 
  (a + b + 1 = 0 ∧ a ≠ b) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l2898_289886


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2898_289837

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop := x^2 - 2*x + a < 0

-- Define the solution set
def solution_set (t : ℝ) (x : ℝ) : Prop := -2 < x ∧ x < t

-- Define the second inequality
def second_inequality (c a : ℝ) (x : ℝ) : Prop := (c+a)*x^2 + 2*(c+a)*x - 1 < 0

theorem quadratic_inequality_solution :
  ∃ (a t : ℝ),
    (∀ x, quadratic_inequality a x ↔ solution_set t x) ∧
    a = -8 ∧
    t = 4 ∧
    ∀ c, (∀ x, second_inequality c a x) ↔ (7 < c ∧ c ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2898_289837


namespace NUMINAMATH_CALUDE_work_completed_together_l2898_289860

/-- The amount of work that can be completed by two workers in one day, given their individual work rates. -/
theorem work_completed_together 
  (days_A : ℝ) -- Number of days A takes to complete the work
  (days_B : ℝ) -- Number of days B takes to complete the work
  (h1 : days_A = 10) -- A can finish the work in 10 days
  (h2 : days_B = days_A / 2) -- B can do the same work in half the time taken by A
  : (1 / days_A + 1 / days_B) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completed_together_l2898_289860


namespace NUMINAMATH_CALUDE_max_working_groups_l2898_289892

theorem max_working_groups (total_instructors : ℕ) (group_size : ℕ) (max_membership : ℕ) :
  total_instructors = 36 →
  group_size = 4 →
  max_membership = 2 →
  (∃ (n : ℕ), n ≤ 18 ∧ 
    n * group_size ≤ total_instructors * max_membership ∧
    ∀ (m : ℕ), m > n → m * group_size > total_instructors * max_membership) :=
by sorry

end NUMINAMATH_CALUDE_max_working_groups_l2898_289892


namespace NUMINAMATH_CALUDE_triangle_side_length_l2898_289849

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define a median
def is_median (t : Triangle) (M : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ × ℝ), m = ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2) ∧ M = m

theorem triangle_side_length (t : Triangle) :
  length t.A t.B = 7 →
  length t.B t.C = 10 →
  (∃ (M : ℝ × ℝ), is_median t M ∧ length t.A M = 5) →
  length t.A t.C = Real.sqrt 51 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2898_289849


namespace NUMINAMATH_CALUDE_equilateral_pyramid_cross_section_l2898_289858

/-- Represents a pyramid with an equilateral triangular base -/
structure EquilateralPyramid where
  /-- Side length of the base triangle -/
  base_side : ℝ
  /-- Height of the pyramid -/
  height : ℝ

/-- Represents a plane that intersects the pyramid -/
structure IntersectingPlane where
  /-- Angle between the plane and the base of the pyramid -/
  angle_with_base : ℝ

/-- Calculates the area of the cross-section of the pyramid -/
noncomputable def cross_section_area (p : EquilateralPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

theorem equilateral_pyramid_cross_section
  (p : EquilateralPyramid)
  (plane : IntersectingPlane) :
  p.base_side = 3 ∧
  p.height = Real.sqrt 3 ∧
  plane.angle_with_base = π / 3 →
  cross_section_area p plane = 11 * Real.sqrt 3 / 10 := by
    sorry

end NUMINAMATH_CALUDE_equilateral_pyramid_cross_section_l2898_289858


namespace NUMINAMATH_CALUDE_intersection_A_B_solution_set_quadratic_l2898_289856

-- Define sets A and B
def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

-- Define the quadratic inequality
def quadratic_inequality (x : ℝ) : Prop := 2*x^2 + 4*x - 6 < 0

-- Theorem for the solution set of the quadratic inequality
theorem solution_set_quadratic : {x | quadratic_inequality x} = B := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_solution_set_quadratic_l2898_289856


namespace NUMINAMATH_CALUDE_total_distance_walked_and_run_l2898_289822

/-- Calculates the total distance traveled when walking and running at different speeds for different durations. -/
theorem total_distance_walked_and_run 
  (walk_time : ℝ) (walk_speed : ℝ) (run_time : ℝ) (run_speed : ℝ) :
  walk_time = 60 →  -- 60 minutes walking
  walk_speed = 3 →  -- 3 mph walking speed
  run_time = 45 →   -- 45 minutes running
  run_speed = 8 →   -- 8 mph running speed
  (walk_time + run_time) / 60 = 1.75 →  -- Total time in hours
  walk_time / 60 * walk_speed + run_time / 60 * run_speed = 9 := by
  sorry

#check total_distance_walked_and_run

end NUMINAMATH_CALUDE_total_distance_walked_and_run_l2898_289822


namespace NUMINAMATH_CALUDE_domain_transformation_l2898_289838

/-- Given that the domain of f(x^2 - 1) is [0, 3], prove that the domain of f(2x - 1) is [0, 9/2] -/
theorem domain_transformation (f : ℝ → ℝ) :
  (∀ y, f y ≠ 0 → 0 ≤ y + 1 ∧ y + 1 ≤ 3) →
  (∀ x, f (2*x - 1) ≠ 0 → 0 ≤ x ∧ x ≤ 9/2) :=
sorry

end NUMINAMATH_CALUDE_domain_transformation_l2898_289838


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2898_289842

/-- A cubic function with a linear term -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x

theorem cubic_function_properties (m : ℝ) (h : f m 1 = 5) :
  m = 4 ∧ ∀ x : ℝ, f m (-x) = -(f m x) := by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2898_289842


namespace NUMINAMATH_CALUDE_intersection_M_N_l2898_289839

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set N
def N : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2898_289839


namespace NUMINAMATH_CALUDE_vectors_linearly_dependent_iff_l2898_289890

/-- Two vectors in ℝ² -/
def v1 : Fin 2 → ℝ := ![2, 5]
def v2 (m : ℝ) : Fin 2 → ℝ := ![4, m]

/-- Definition of linear dependence for two vectors -/
def linearlyDependent (u v : Fin 2 → ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (∀ i, a * u i + b * v i = 0)

/-- Theorem: The vectors v1 and v2 are linearly dependent iff m = 10 -/
theorem vectors_linearly_dependent_iff (m : ℝ) :
  linearlyDependent v1 (v2 m) ↔ m = 10 := by
  sorry

end NUMINAMATH_CALUDE_vectors_linearly_dependent_iff_l2898_289890


namespace NUMINAMATH_CALUDE_inequality_condition_l2898_289824

theorem inequality_condition (a b : ℝ) : 
  (a > b → ((a + b) / 2)^2 > a * b) ∧ 
  (∃ a b : ℝ, ((a + b) / 2)^2 > a * b ∧ a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l2898_289824


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l2898_289867

theorem quadratic_equation_value (x : ℝ) : 2*x^2 + 3*x + 7 = 8 → 9 - 4*x^2 - 6*x = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l2898_289867


namespace NUMINAMATH_CALUDE_angle_sum_inequality_l2898_289834

theorem angle_sum_inequality (θ₁ θ₂ θ₃ θ₄ : Real) 
  (h₁ : 0 < θ₁ ∧ θ₁ < π/2)
  (h₂ : 0 < θ₂ ∧ θ₂ < π/2)
  (h₃ : 0 < θ₃ ∧ θ₃ < π/2)
  (h₄ : 0 < θ₄ ∧ θ₄ < π/2)
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = π) :
  (Real.sqrt 2 * Real.sin θ₁ - 1) / Real.cos θ₁ +
  (Real.sqrt 2 * Real.sin θ₂ - 1) / Real.cos θ₂ +
  (Real.sqrt 2 * Real.sin θ₃ - 1) / Real.cos θ₃ +
  (Real.sqrt 2 * Real.sin θ₄ - 1) / Real.cos θ₄ ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_inequality_l2898_289834


namespace NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l2898_289874

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between a line and three circles -/
def max_line_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between three circles and one line -/
theorem max_intersections_three_circles_one_line :
  max_circle_intersections + max_line_circle_intersections = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l2898_289874


namespace NUMINAMATH_CALUDE_jiaqi_pe_grade_l2898_289827

/-- Calculates the total grade based on component scores and weights -/
def calculate_total_grade (extracurricular_score : ℝ) (midterm_score : ℝ) (final_score : ℝ) : ℝ :=
  extracurricular_score * 0.2 + midterm_score * 0.3 + final_score * 0.5

/-- Jiaqi's physical education grade calculation -/
theorem jiaqi_pe_grade :
  calculate_total_grade 96 92 97 = 95.3 := by
  sorry

end NUMINAMATH_CALUDE_jiaqi_pe_grade_l2898_289827


namespace NUMINAMATH_CALUDE_xyz_congruence_l2898_289899

theorem xyz_congruence (x y z : Int) : 
  x < 7 → y < 7 → z < 7 →
  (x + 3*y + 2*z) % 7 = 2 →
  (3*x + 2*y + z) % 7 = 5 →
  (2*x + y + 3*z) % 7 = 3 →
  (x * y * z) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_congruence_l2898_289899


namespace NUMINAMATH_CALUDE_factorization_condition_l2898_289854

theorem factorization_condition (a : ℤ) : 
  (∃ b c : ℤ, ∀ x, (x - a) * (x - 10) + 1 = (x + b) * (x + c)) ↔ (a = 8 ∨ a = 12) :=
sorry

end NUMINAMATH_CALUDE_factorization_condition_l2898_289854


namespace NUMINAMATH_CALUDE_bertha_descendants_no_daughters_l2898_289887

/-- Represents a person in Bertha's family tree -/
inductive Person
| bertha : Person
| child : Person → Person
| grandchild : Person → Person
| greatgrandchild : Person → Person

/-- Represents the gender of a person -/
inductive Gender
| male
| female

/-- Function to determine the gender of a person -/
def gender : Person → Gender
| Person.bertha => Gender.female
| _ => sorry

/-- Function to count the number of daughters a person has -/
def daughterCount : Person → Nat
| Person.bertha => 7
| _ => sorry

/-- Function to count the number of sons a person has -/
def sonCount : Person → Nat
| Person.bertha => 3
| _ => sorry

/-- Function to count the total number of female descendants of a person -/
def femaleDescendantCount : Person → Nat
| Person.bertha => 40
| _ => sorry

/-- Function to determine if a person has exactly three daughters -/
def hasThreeDaughters : Person → Bool
| _ => sorry

/-- Function to count the number of descendants (including the person) who have no daughters -/
def descendantsWithNoDaughters : Person → Nat
| _ => sorry

/-- Theorem stating that the number of Bertha's descendants with no daughters is 28 -/
theorem bertha_descendants_no_daughters :
  descendantsWithNoDaughters Person.bertha = 28 := by sorry

end NUMINAMATH_CALUDE_bertha_descendants_no_daughters_l2898_289887


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l2898_289895

theorem chocolate_box_problem (N : ℕ) (rows columns : ℕ) :
  -- Initial conditions
  N > 0 ∧ rows > 0 ∧ columns > 0 ∧
  -- After operations, one-third remains
  N / 3 > 0 ∧
  -- Three rows minus one can be filled at one point
  (3 * columns - 1 ≤ N ∧ 3 * columns > N / 3) ∧
  -- Five columns minus one can be filled at another point
  (5 * rows - 1 ≤ N ∧ 5 * rows > N / 3) →
  -- Conclusions
  N = 60 ∧ N - (3 * columns - 1) = 25 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l2898_289895


namespace NUMINAMATH_CALUDE_sin_1035_degrees_l2898_289808

theorem sin_1035_degrees : Real.sin (1035 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1035_degrees_l2898_289808


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2898_289898

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) ↔ 
  ((a^2 + b^2 = 1 → ∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) ∧
   (∃ a b : ℝ, a^2 + b^2 ≠ 1 ∧ ∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2898_289898


namespace NUMINAMATH_CALUDE_car_owners_without_motorcycle_l2898_289807

theorem car_owners_without_motorcycle (total_adults : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) (no_vehicle_owners : ℕ) 
  (h1 : total_adults = 560)
  (h2 : car_owners = 520)
  (h3 : motorcycle_owners = 80)
  (h4 : no_vehicle_owners = 10) :
  car_owners - (total_adults - no_vehicle_owners - (car_owners + motorcycle_owners - (total_adults - no_vehicle_owners))) = 470 := by
  sorry

end NUMINAMATH_CALUDE_car_owners_without_motorcycle_l2898_289807


namespace NUMINAMATH_CALUDE_max_N_is_six_l2898_289850

/-- Definition of I_k -/
def I (k : ℕ) : ℕ := 10^(k+1) + 32

/-- Definition of N(k) -/
def N (k : ℕ) : ℕ := (I k).factors.count 2

/-- Theorem: The maximum value of N(k) is 6 -/
theorem max_N_is_six :
  (∀ k : ℕ, N k ≤ 6) ∧ (∃ k : ℕ, N k = 6) := by sorry

end NUMINAMATH_CALUDE_max_N_is_six_l2898_289850


namespace NUMINAMATH_CALUDE_division_multiplication_result_l2898_289823

theorem division_multiplication_result : 
  let x : ℝ := 6.5
  let y : ℝ := (x / 6) * 12
  y = 13 := by sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l2898_289823


namespace NUMINAMATH_CALUDE_remainder_98_pow_24_mod_100_l2898_289891

theorem remainder_98_pow_24_mod_100 : 98^24 % 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_pow_24_mod_100_l2898_289891


namespace NUMINAMATH_CALUDE_a_range_l2898_289859

/-- Given a > 0, if the function y = a^x is not monotonically increasing on ℝ
    or the inequality ax^2 - ax + 1 > 0 does not hold for ∀x ∈ ℝ,
    and at least one of these conditions is true,
    then a ∈ (0,1] ∪ [4,+∞) -/
theorem a_range (a : ℝ) (h_a_pos : a > 0) : 
  (¬∀ x y : ℝ, x < y → a^x < a^y) ∨ 
  (¬∀ x : ℝ, a*x^2 - a*x + 1 > 0) ∧ 
  ((∀ x y : ℝ, x < y → a^x < a^y) ∨ 
   (∀ x : ℝ, a*x^2 - a*x + 1 > 0)) → 
  a ∈ Set.Ioc 0 1 ∪ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_a_range_l2898_289859


namespace NUMINAMATH_CALUDE_ln_101_100_gt_2_201_l2898_289832

theorem ln_101_100_gt_2_201 : Real.log (101/100) > 2/201 := by
  sorry

end NUMINAMATH_CALUDE_ln_101_100_gt_2_201_l2898_289832


namespace NUMINAMATH_CALUDE_chemistry_textbook_weight_l2898_289869

/-- The weight of the geometry textbook in pounds -/
def geometry_weight : ℝ := 0.62

/-- The additional weight of the chemistry textbook compared to the geometry textbook in pounds -/
def additional_weight : ℝ := 6.5

/-- The weight of the chemistry textbook in pounds -/
def chemistry_weight : ℝ := geometry_weight + additional_weight

theorem chemistry_textbook_weight : chemistry_weight = 7.12 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_textbook_weight_l2898_289869


namespace NUMINAMATH_CALUDE_cube_construction_count_l2898_289897

/-- The number of distinguishable ways to construct a cube from colored squares -/
def distinguishable_cube_constructions : ℕ := 1260

/-- The number of faces on a cube -/
def cube_faces : ℕ := 6

/-- The number of colored squares available -/
def colored_squares : ℕ := 8

/-- The number of rotational symmetries when one face is fixed -/
def rotational_symmetries : ℕ := 4

theorem cube_construction_count :
  distinguishable_cube_constructions = (colored_squares - 1).factorial / rotational_symmetries :=
sorry

end NUMINAMATH_CALUDE_cube_construction_count_l2898_289897


namespace NUMINAMATH_CALUDE_value_of_d_l2898_289878

theorem value_of_d (a b c d e : ℝ) 
  (h : 3 * (a^2 + b^2 + c^2) + 4 = 2*d + Real.sqrt (a + b + c - d + e)) 
  (he : e = 1) : 
  d = 7/4 := by
sorry

end NUMINAMATH_CALUDE_value_of_d_l2898_289878


namespace NUMINAMATH_CALUDE_square_side_length_l2898_289894

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * side * 2 = diagonal * diagonal ∧ side = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2898_289894


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l2898_289861

theorem complex_equation_real_solution (a : ℝ) : 
  (((a : ℂ) / (1 + Complex.I) + 1 + Complex.I).im = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l2898_289861


namespace NUMINAMATH_CALUDE_julia_bought_399_balls_l2898_289847

/-- The number of balls Julia bought -/
def total_balls (red_packs yellow_packs green_packs balls_per_pack : ℕ) : ℕ :=
  (red_packs + yellow_packs + green_packs) * balls_per_pack

/-- Proof that Julia bought 399 balls -/
theorem julia_bought_399_balls :
  total_balls 3 10 8 19 = 399 := by
  sorry

end NUMINAMATH_CALUDE_julia_bought_399_balls_l2898_289847


namespace NUMINAMATH_CALUDE_f_plus_one_is_odd_l2898_289826

-- Define the property of the function f
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ + 1

-- Define what it means for a function to be odd
def is_odd (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

-- Theorem statement
theorem f_plus_one_is_odd (f : ℝ → ℝ) (h : satisfies_property f) :
  is_odd (fun x => f x + 1) :=
sorry

end NUMINAMATH_CALUDE_f_plus_one_is_odd_l2898_289826


namespace NUMINAMATH_CALUDE_sequence_properties_l2898_289802

def a (n : ℕ) : ℤ := 2^n - (-1)^n

theorem sequence_properties :
  (∀ k : ℕ, k > 0 →
    (a k + a (k + 2) = 2 * a (k + 1)) ↔ k = 2) ∧
  (∀ r s : ℕ, r > 1 ∧ s > r →
    (a 1 + a s = 2 * a r) → s = r + 1) ∧
  (∀ q r s t : ℕ, 0 < q ∧ q < r ∧ r < s ∧ s < t →
    ¬(a q + a t = a r + a s)) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2898_289802


namespace NUMINAMATH_CALUDE_largest_base7_five_digit_to_base10_l2898_289882

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ (digits.length - 1 - i))) 0

/-- The largest five-digit number in base-7 --/
def largestBase7FiveDigit : List Nat := [6, 6, 6, 6, 6]

theorem largest_base7_five_digit_to_base10 :
  base7ToBase10 largestBase7FiveDigit = 16806 := by
  sorry

end NUMINAMATH_CALUDE_largest_base7_five_digit_to_base10_l2898_289882


namespace NUMINAMATH_CALUDE_surface_area_after_corner_removal_l2898_289876

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (c : CubeDimensions) : ℝ :=
  6 * c.length * c.width

/-- Represents the problem setup -/
structure ProblemSetup where
  originalCube : CubeDimensions
  cornerCube : CubeDimensions
  numCorners : ℕ

/-- Theorem stating that the surface area remains unchanged after removing corner cubes -/
theorem surface_area_after_corner_removal (p : ProblemSetup) 
  (h1 : p.originalCube.length = 4)
  (h2 : p.originalCube.width = 4)
  (h3 : p.originalCube.height = 4)
  (h4 : p.cornerCube.length = 2)
  (h5 : p.cornerCube.width = 2)
  (h6 : p.cornerCube.height = 2)
  (h7 : p.numCorners = 8) :
  surfaceArea p.originalCube = 96 := by
  sorry

#eval surfaceArea { length := 4, width := 4, height := 4 }

end NUMINAMATH_CALUDE_surface_area_after_corner_removal_l2898_289876


namespace NUMINAMATH_CALUDE_abc_equation_solutions_l2898_289857

/-- Given integers a, b, c ≥ 2, prove that a b c - 1 = (a - 1)(b - 1)(c - 1) 
    if and only if (a, b, c) is a permutation of (2, 2, 2), (2, 2, 4), (2, 4, 8), or (3, 5, 15) -/
theorem abc_equation_solutions (a b c : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  a * b * c - 1 = (a - 1) * (b - 1) * (c - 1) ↔ 
  List.Perm [a, b, c] [2, 2, 2] ∨ 
  List.Perm [a, b, c] [2, 2, 4] ∨ 
  List.Perm [a, b, c] [2, 4, 8] ∨ 
  List.Perm [a, b, c] [3, 5, 15] :=
by sorry


end NUMINAMATH_CALUDE_abc_equation_solutions_l2898_289857


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2898_289862

theorem complex_magnitude_product : 
  Complex.abs ((5 - 3*Complex.I) * (7 + 24*Complex.I)) = 25 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2898_289862


namespace NUMINAMATH_CALUDE_car_braking_distance_l2898_289829

/-- Represents the braking sequence of a car -/
def brakingSequence (initial : ℕ) (decrease : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => max 0 (brakingSequence initial decrease n - decrease)

/-- Calculates the total distance traveled during braking -/
def totalDistance (initial : ℕ) (decrease : ℕ) : ℕ :=
  (List.range 100).foldl (λ acc n => acc + brakingSequence initial decrease n) 0

/-- Theorem stating the total braking distance for the given conditions -/
theorem car_braking_distance :
  totalDistance 36 8 = 108 := by
  sorry


end NUMINAMATH_CALUDE_car_braking_distance_l2898_289829


namespace NUMINAMATH_CALUDE_mathcounts_teach_probability_l2898_289843

def mathcounts_letters : Finset Char := {'M', 'A', 'T', 'H', 'C', 'O', 'U', 'N', 'T', 'S'}
def teach_letters : Finset Char := {'T', 'E', 'A', 'C', 'H'}

theorem mathcounts_teach_probability :
  let common_letters := mathcounts_letters ∩ teach_letters
  (common_letters.card : ℚ) / mathcounts_letters.card = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_mathcounts_teach_probability_l2898_289843


namespace NUMINAMATH_CALUDE_f_3_is_even_l2898_289804

/-- Given a function f(x) = a(x-1)³ + bx + c where a is real and b, c are integers,
    if f(-1) = 2, then f(3) must be even. -/
theorem f_3_is_even (a : ℝ) (b c : ℤ) :
  let f : ℝ → ℝ := λ x => a * (x - 1)^3 + b * x + c
  (f (-1) = 2) → ∃ k : ℤ, f 3 = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_f_3_is_even_l2898_289804


namespace NUMINAMATH_CALUDE_ratio_problem_l2898_289875

theorem ratio_problem (a b c d : ℝ) (h1 : b / a = 4) (h2 : d / c = 2) : (a + b) / (c + d) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2898_289875


namespace NUMINAMATH_CALUDE_constant_k_value_l2898_289864

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, 
  -x^2 - (k + 7)*x - 8 = -(x - 2)*(x - 4) ↔ k = -13 := by
  sorry

end NUMINAMATH_CALUDE_constant_k_value_l2898_289864


namespace NUMINAMATH_CALUDE_integer_triple_solution_l2898_289871

theorem integer_triple_solution (x y z : ℤ) :
  x * y * z + 4 * (x + y + z) = 2 * (x * y + x * z + y * z) + 7 ↔
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 3 ∧ y = 3 ∧ z = 1) ∨
  (x = 3 ∧ y = 1 ∧ z = 3) ∨
  (x = 1 ∧ y = 3 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_solution_l2898_289871


namespace NUMINAMATH_CALUDE_movie_collection_size_l2898_289818

theorem movie_collection_size :
  ∀ (dvd_count blu_count : ℕ),
  (dvd_count : ℚ) / blu_count = 17 / 4 →
  (dvd_count : ℚ) / (blu_count - 4) = 9 / 2 →
  dvd_count + blu_count = 378 :=
by
  sorry

end NUMINAMATH_CALUDE_movie_collection_size_l2898_289818


namespace NUMINAMATH_CALUDE_ratio_transformation_l2898_289851

theorem ratio_transformation (x : ℚ) : 
  ((2 : ℚ) + 2) / (x + 2) = 4 / 5 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transformation_l2898_289851


namespace NUMINAMATH_CALUDE_simplify_expression_l2898_289813

theorem simplify_expression (a : ℝ) (h : -1 < a ∧ a < 0) :
  Real.sqrt ((a + 1/a)^2 - 4) + Real.sqrt ((a - 1/a)^2 + 4) = -2/a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2898_289813


namespace NUMINAMATH_CALUDE_two_sum_of_four_reciprocals_l2898_289853

/-- A function that checks if a positive integer can be expressed as the sum of reciprocals of four different positive integers -/
def is_sum_of_four_reciprocals (n : ℕ+) : Prop :=
  ∃ (a b c d : ℕ+), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = n

/-- The theorem stating that there are exactly two positive integers that can be expressed as the sum of reciprocals of four different positive integers -/
theorem two_sum_of_four_reciprocals :
  ∃! (s : Finset ℕ+), (∀ n ∈ s, is_sum_of_four_reciprocals n) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_two_sum_of_four_reciprocals_l2898_289853


namespace NUMINAMATH_CALUDE_max_sum_2_by_1009_l2898_289877

/-- Represents a grid of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the maximum sum of numbers in white squares for a given grid -/
def maxSumWhiteSquares (g : Grid) : ℕ :=
  if g.rows ≠ 2 ∨ g.cols ≠ 1009 then 0
  else
    let interiorContribution := (g.cols - 2) * 3
    let endpointContribution := 2 * 2
    interiorContribution + endpointContribution

/-- The theorem stating the maximum sum for a 2 by 1009 grid -/
theorem max_sum_2_by_1009 :
  ∀ g : Grid, g.rows = 2 ∧ g.cols = 1009 → maxSumWhiteSquares g = 3025 :=
by
  sorry

#eval maxSumWhiteSquares ⟨2, 1009⟩

end NUMINAMATH_CALUDE_max_sum_2_by_1009_l2898_289877


namespace NUMINAMATH_CALUDE_max_expression_value_l2898_289814

/-- Represents the count of integers equal to each value from 1 to 2003 -/
def IntegerCounts := Fin 2003 → ℕ

/-- The sum of all integers is 2003 -/
def SumConstraint (counts : IntegerCounts) : Prop :=
  (Finset.range 2003).sum (fun i => (i + 1) * counts i) = 2003

/-- The expression to be maximized -/
def ExpressionToMaximize (counts : IntegerCounts) : ℕ :=
  (Finset.range 2002).sum (fun i => i * counts (i + 1))

/-- There are at least two integers in the set -/
def AtLeastTwoIntegers (counts : IntegerCounts) : Prop :=
  (Finset.range 2003).sum (fun i => counts i) ≥ 2

theorem max_expression_value (counts : IntegerCounts) 
  (h1 : SumConstraint counts) (h2 : AtLeastTwoIntegers counts) :
  ExpressionToMaximize counts ≤ 2001 := by
  sorry

end NUMINAMATH_CALUDE_max_expression_value_l2898_289814


namespace NUMINAMATH_CALUDE_roots_of_equation_l2898_289848

theorem roots_of_equation : 
  {x : ℝ | x * (x + 5)^3 * (5 - x) = 0} = {-5, 0, 5} := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2898_289848


namespace NUMINAMATH_CALUDE_three_cell_shapes_count_l2898_289888

/-- Represents the number of cells in a shape -/
inductive ShapeSize
| Three : ShapeSize
| Four : ShapeSize

/-- Represents a configuration of shapes -/
structure Configuration :=
  (threeCell : ℕ)
  (fourCell : ℕ)

/-- Checks if a configuration is valid -/
def isValidConfiguration (config : Configuration) : Prop :=
  3 * config.threeCell + 4 * config.fourCell = 22

/-- Checks if a configuration matches the desired solution -/
def isDesiredSolution (config : Configuration) : Prop :=
  config.threeCell = 6 ∧ config.fourCell = 1

/-- The main theorem to prove -/
theorem three_cell_shapes_count :
  ∃ (config : Configuration),
    isValidConfiguration config ∧ isDesiredSolution config :=
sorry

end NUMINAMATH_CALUDE_three_cell_shapes_count_l2898_289888


namespace NUMINAMATH_CALUDE_probability_tamika_greater_carlos_l2898_289873

def tamika_set : Finset ℕ := {8, 9, 10, 11}
def carlos_set : Finset ℕ := {3, 5, 6, 7}

def tamika_result (a b : ℕ) : ℕ := a + b

def carlos_result (a b : ℕ) : ℕ := a * b - 2

def valid_pair (s : Finset ℕ) (a b : ℕ) : Prop :=
  a ∈ s ∧ b ∈ s ∧ a ≠ b

def favorable_outcomes : ℕ := 26
def total_outcomes : ℕ := 36

theorem probability_tamika_greater_carlos :
  (↑favorable_outcomes / ↑total_outcomes : ℚ) = 13 / 18 := by sorry

end NUMINAMATH_CALUDE_probability_tamika_greater_carlos_l2898_289873


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2898_289805

-- Define the quadratic function
def f (t x : ℝ) : ℝ := x^2 - 2*t*x + 3

-- State the theorem
theorem quadratic_function_properties (t : ℝ) (h_t : t > 0) :
  -- Part 1
  (f t 2 = 1 → t = 3/2) ∧
  -- Part 2
  (∃ (x_min : ℝ), 0 ≤ x_min ∧ x_min ≤ 3 ∧
    (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f t x ≥ f t x_min) ∧
    f t x_min = -2 → t = Real.sqrt 5) ∧
  -- Part 3
  (∀ (m a b : ℝ),
    f t (m - 2) = a ∧ f t 4 = b ∧ f t m = a ∧ a < b ∧ b < 3 →
    (3 < m ∧ m < 4) ∨ m > 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2898_289805


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2898_289868

/-- Represents an ellipse with the given equation -/
structure Ellipse (k : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 / (k - 4) + y^2 / (10 - k) = 1)

/-- Condition for the foci to be on the x-axis -/
def foci_on_x_axis (k : ℝ) : Prop :=
  k - 4 > 10 - k

/-- The main theorem stating that 4 < k < 10 is necessary but not sufficient -/
theorem necessary_but_not_sufficient :
  ∃ k : ℝ, 4 < k ∧ k < 10 ∧
  (∀ k' : ℝ, (∃ e : Ellipse k', foci_on_x_axis k') → 4 < k' ∧ k' < 10) ∧
  ¬(∀ k' : ℝ, 4 < k' ∧ k' < 10 → ∃ e : Ellipse k', foci_on_x_axis k') :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2898_289868


namespace NUMINAMATH_CALUDE_problem_solution_l2898_289880

theorem problem_solution (s t : ℝ) 
  (eq1 : 12 * s + 8 * t = 160)
  (eq2 : s = t^2 + 2) : 
  t = (Real.sqrt 103 - 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2898_289880


namespace NUMINAMATH_CALUDE_garden_area_l2898_289811

theorem garden_area (width : ℝ) (length : ℝ) (perimeter : ℝ) :
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l2898_289811


namespace NUMINAMATH_CALUDE_ferry_speed_difference_l2898_289889

theorem ferry_speed_difference (speed_p time_p distance_q_factor time_difference : ℝ) 
  (h1 : speed_p = 6)
  (h2 : time_p = 3)
  (h3 : distance_q_factor = 3)
  (h4 : time_difference = 3) : 
  let distance_p := speed_p * time_p
  let distance_q := distance_q_factor * distance_p
  let time_q := time_p + time_difference
  let speed_q := distance_q / time_q
  speed_q - speed_p = 3 := by sorry

end NUMINAMATH_CALUDE_ferry_speed_difference_l2898_289889


namespace NUMINAMATH_CALUDE_fishing_problem_l2898_289801

theorem fishing_problem (jordan_catch : ℕ) (perry_catch : ℕ) (total_catch : ℕ) (fish_lost : ℕ) (fish_remaining : ℕ) : 
  jordan_catch = 4 →
  perry_catch = 2 * jordan_catch →
  total_catch = jordan_catch + perry_catch →
  fish_lost = total_catch / 4 →
  fish_remaining = total_catch - fish_lost →
  fish_remaining = 9 := by
sorry

end NUMINAMATH_CALUDE_fishing_problem_l2898_289801


namespace NUMINAMATH_CALUDE_circle_contains_origin_l2898_289831

theorem circle_contains_origin
  (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ a b c : ℝ)
  (h₁ : x₁ > 0) (h₂ : y₁ > 0)
  (h₃ : x₂ < 0) (h₄ : y₂ > 0)
  (h₅ : x₃ < 0) (h₆ : y₃ < 0)
  (h₇ : x₄ > 0) (h₈ : y₄ < 0)
  (h₉ : (x₁ - a)^2 + (y₁ - b)^2 ≤ c^2)
  (h₁₀ : (x₂ - a)^2 + (y₂ - b)^2 ≤ c^2)
  (h₁₁ : (x₃ - a)^2 + (y₃ - b)^2 ≤ c^2)
  (h₁₂ : (x₄ - a)^2 + (y₄ - b)^2 ≤ c^2)
  (h₁₃ : c > 0) :
  a^2 + b^2 < c^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_contains_origin_l2898_289831


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2898_289833

theorem complex_equation_solution :
  ∃ (x : ℂ), (5 : ℂ) + 2 * Complex.I * x = (3 : ℂ) - 4 * Complex.I * x ∧ x = Complex.I / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2898_289833


namespace NUMINAMATH_CALUDE_last_digit_of_powers_l2898_289819

theorem last_digit_of_powers (n : Nat) :
  (∃ k : Nat, n = 2^1000 ∧ n % 10 = 6) ∧
  (∃ k : Nat, n = 3^1000 ∧ n % 10 = 1) ∧
  (∃ k : Nat, n = 7^1000 ∧ n % 10 = 1) :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_powers_l2898_289819


namespace NUMINAMATH_CALUDE_doll_production_time_l2898_289812

/-- Represents the production details of dolls and accessories in a factory --/
structure DollProduction where
  total_dolls : ℕ
  accessories_per_doll : ℕ
  accessory_time : ℕ
  total_operation_time : ℕ

/-- Calculates the time required to make each doll --/
def time_per_doll (prod : DollProduction) : ℕ :=
  (prod.total_operation_time - prod.total_dolls * prod.accessories_per_doll * prod.accessory_time) / prod.total_dolls

/-- Theorem stating that the time to make each doll is 45 seconds --/
theorem doll_production_time (prod : DollProduction) 
  (h1 : prod.total_dolls = 12000)
  (h2 : prod.accessories_per_doll = 11)
  (h3 : prod.accessory_time = 10)
  (h4 : prod.total_operation_time = 1860000) :
  time_per_doll prod = 45 := by
  sorry

#eval time_per_doll { total_dolls := 12000, accessories_per_doll := 11, accessory_time := 10, total_operation_time := 1860000 }

end NUMINAMATH_CALUDE_doll_production_time_l2898_289812


namespace NUMINAMATH_CALUDE_bennys_work_hours_l2898_289865

/-- Given that Benny worked for 6 days and a total of 18 hours, 
    prove that he worked 3 hours each day. -/
theorem bennys_work_hours (days : ℕ) (total_hours : ℕ) 
    (h1 : days = 6) (h2 : total_hours = 18) : 
    total_hours / days = 3 := by
  sorry

end NUMINAMATH_CALUDE_bennys_work_hours_l2898_289865


namespace NUMINAMATH_CALUDE_rectangle_area_with_circles_l2898_289846

/-- The area of a rectangle containing 8 circles arranged in a 2x4 grid, 
    where each circle has a radius of 3 inches. -/
theorem rectangle_area_with_circles (radius : ℝ) (width_circles : ℕ) (length_circles : ℕ) :
  radius = 3 →
  width_circles = 2 →
  length_circles = 4 →
  (2 * radius * width_circles) * (2 * radius * length_circles) = 288 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_circles_l2898_289846


namespace NUMINAMATH_CALUDE_some_magical_beings_are_mystical_creatures_l2898_289870

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Dragon : U → Prop)
variable (MagicalBeing : U → Prop)
variable (MysticalCreature : U → Prop)

-- State the theorem
theorem some_magical_beings_are_mystical_creatures :
  (∀ x, Dragon x → MagicalBeing x) →  -- All dragons are magical beings
  (∃ x, MysticalCreature x ∧ Dragon x) →  -- Some mystical creatures are dragons
  (∃ x, MagicalBeing x ∧ MysticalCreature x)  -- Some magical beings are mystical creatures
:= by sorry

end NUMINAMATH_CALUDE_some_magical_beings_are_mystical_creatures_l2898_289870


namespace NUMINAMATH_CALUDE_dara_jane_age_ratio_l2898_289884

-- Define the given conditions
def minimum_employment_age : ℕ := 25
def jane_current_age : ℕ := 28
def years_until_dara_minimum_age : ℕ := 14
def years_in_future : ℕ := 6

-- Define Dara's current age
def dara_current_age : ℕ := minimum_employment_age - years_until_dara_minimum_age

-- Define Dara's and Jane's ages in 6 years
def dara_future_age : ℕ := dara_current_age + years_in_future
def jane_future_age : ℕ := jane_current_age + years_in_future

-- Theorem to prove
theorem dara_jane_age_ratio : 
  dara_future_age * 2 = jane_future_age := by sorry

end NUMINAMATH_CALUDE_dara_jane_age_ratio_l2898_289884


namespace NUMINAMATH_CALUDE_sqrt_three_plus_two_range_l2898_289809

theorem sqrt_three_plus_two_range :
  ∃ (x : ℝ), x = Real.sqrt 3 ∧ Irrational x ∧ 1 < x ∧ x < 2 → 3.5 < x + 2 ∧ x + 2 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_plus_two_range_l2898_289809


namespace NUMINAMATH_CALUDE_smallest_multiple_l2898_289840

theorem smallest_multiple (n : ℕ) : n = 1628 ↔ 
  (∃ k : ℕ, n = 37 * k) ∧ 
  (∃ m : ℕ, n - 3 = 101 * m) ∧ 
  (∀ x : ℕ, x < n → ¬((∃ k : ℕ, x = 37 * k) ∧ (∃ m : ℕ, x - 3 = 101 * m))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2898_289840


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l2898_289820

theorem product_and_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 12 → (1 / x) = 3 * (1 / y) → x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l2898_289820


namespace NUMINAMATH_CALUDE_joe_paint_usage_l2898_289896

/-- The amount of paint Joe used given the initial amount and usage fractions -/
def paint_used (initial : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) : ℚ :=
  let first_week := initial * first_week_fraction
  let remaining := initial - first_week
  let second_week := remaining * second_week_fraction
  first_week + second_week

/-- Theorem stating that Joe used 168 gallons of paint -/
theorem joe_paint_usage :
  paint_used 360 (1/3) (1/5) = 168 := by
  sorry

end NUMINAMATH_CALUDE_joe_paint_usage_l2898_289896


namespace NUMINAMATH_CALUDE_food_distributor_comparison_l2898_289810

theorem food_distributor_comparison (p₁ p₂ : ℝ) 
  (h₁ : 0 < p₁) (h₂ : 0 < p₂) (h₃ : p₁ < p₂) :
  (2 * p₁ * p₂) / (p₁ + p₂) < (p₁ + p₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_food_distributor_comparison_l2898_289810


namespace NUMINAMATH_CALUDE_heptagon_angles_l2898_289844

/-- The number of sides in a heptagon -/
def n : ℕ := 7

/-- The measure of an interior angle of a regular heptagon -/
def interior_angle : ℚ := (5 * 180) / n

/-- The measure of an exterior angle of a regular heptagon -/
def exterior_angle : ℚ := 180 - interior_angle

theorem heptagon_angles :
  (interior_angle = (5 * 180) / n) ∧
  (exterior_angle = 180 - ((5 * 180) / n)) := by
  sorry

end NUMINAMATH_CALUDE_heptagon_angles_l2898_289844


namespace NUMINAMATH_CALUDE_count_divisors_not_divisible_by_three_of_180_l2898_289863

def divisors_not_divisible_by_three (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ x => x ∣ n ∧ ¬(3 ∣ x))

theorem count_divisors_not_divisible_by_three_of_180 :
  (divisors_not_divisible_by_three 180).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_divisors_not_divisible_by_three_of_180_l2898_289863


namespace NUMINAMATH_CALUDE_dog_eaten_cost_calculation_l2898_289872

def cake_cost (flour_cost sugar_cost butter_cost eggs_cost : ℚ) : ℚ :=
  flour_cost + sugar_cost + butter_cost + eggs_cost

def dog_eaten_cost (total_cost : ℚ) (total_slices mother_eaten_slices : ℕ) : ℚ :=
  (total_cost * (total_slices - mother_eaten_slices : ℚ)) / total_slices

theorem dog_eaten_cost_calculation :
  let flour_cost : ℚ := 4
  let sugar_cost : ℚ := 2
  let butter_cost : ℚ := 2.5
  let eggs_cost : ℚ := 0.5
  let total_slices : ℕ := 6
  let mother_eaten_slices : ℕ := 2
  let total_cost := cake_cost flour_cost sugar_cost butter_cost eggs_cost
  dog_eaten_cost total_cost total_slices mother_eaten_slices = 6 := by
  sorry

#eval dog_eaten_cost (cake_cost 4 2 2.5 0.5) 6 2

end NUMINAMATH_CALUDE_dog_eaten_cost_calculation_l2898_289872


namespace NUMINAMATH_CALUDE_root_equivalence_l2898_289855

theorem root_equivalence (r : ℝ) : r^2 - 2*r - 1 = 0 → r^5 - 29*r - 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_equivalence_l2898_289855


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l2898_289828

/-- Given two vectors in ℝ², prove that their linear combination results in the expected vector. -/
theorem vector_subtraction_and_scalar_multiplication (a b : ℝ × ℝ) :
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l2898_289828


namespace NUMINAMATH_CALUDE_probability_theorem_l2898_289815

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ k : ℕ, a * b + a + b = 7 * k - 2

def total_pairs : ℕ := Nat.choose 100 2

def valid_pairs : ℕ := 105

theorem probability_theorem :
  (valid_pairs : ℚ) / total_pairs = 7 / 330 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2898_289815


namespace NUMINAMATH_CALUDE_ratio_proof_l2898_289866

def problem (A B : ℕ) : Prop :=
  A = 45 ∧ Nat.lcm A B = 180

theorem ratio_proof (A B : ℕ) (h : problem A B) : 
  A / B = 45 / 4 := by sorry

end NUMINAMATH_CALUDE_ratio_proof_l2898_289866


namespace NUMINAMATH_CALUDE_infinite_prime_divisors_of_derived_set_l2898_289841

/-- A subset of natural numbers with infinite members -/
def InfiniteNatSubset (S : Set ℕ) : Prop := Set.Infinite S

/-- The set S' derived from S -/
def DerivedSet (S : Set ℕ) : Set ℕ :=
  {n | ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ n = x^y + y^x}

/-- The set of prime divisors of a set of natural numbers -/
def PrimeDivisors (S : Set ℕ) : Set ℕ :=
  {p | Nat.Prime p ∧ ∃ n ∈ S, p ∣ n}

/-- Main theorem: The set of prime divisors of S' is infinite -/
theorem infinite_prime_divisors_of_derived_set (S : Set ℕ) 
  (h : InfiniteNatSubset S) : Set.Infinite (PrimeDivisors (DerivedSet S)) :=
sorry

end NUMINAMATH_CALUDE_infinite_prime_divisors_of_derived_set_l2898_289841


namespace NUMINAMATH_CALUDE_iphone_price_proof_l2898_289885

/-- The original price of an iPhone X -/
def original_price : ℝ := sorry

/-- The discount rate for buying at least 2 smartphones -/
def discount_rate : ℝ := 0.05

/-- The number of people buying iPhones -/
def num_buyers : ℕ := 3

/-- The amount saved by buying together -/
def amount_saved : ℝ := 90

theorem iphone_price_proof :
  (original_price * num_buyers * discount_rate = amount_saved) →
  original_price = 600 := by
  sorry

end NUMINAMATH_CALUDE_iphone_price_proof_l2898_289885


namespace NUMINAMATH_CALUDE_committee_selection_ways_l2898_289806

-- Define the total number of team owners
def total_owners : ℕ := 30

-- Define the number of owners who don't want to serve
def ineligible_owners : ℕ := 3

-- Define the size of the committee
def committee_size : ℕ := 5

-- Define the number of eligible owners
def eligible_owners : ℕ := total_owners - ineligible_owners

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem committee_selection_ways : 
  combination eligible_owners committee_size = 65780 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l2898_289806


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2898_289893

def i : ℂ := Complex.I

theorem complex_magnitude_problem (z : ℂ) (h : z * (i + 1) = i) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2898_289893


namespace NUMINAMATH_CALUDE_pizza_combinations_l2898_289845

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  (n.choose 1) + (n.choose 2) + (n.choose 3) = 92 :=
by sorry

end NUMINAMATH_CALUDE_pizza_combinations_l2898_289845


namespace NUMINAMATH_CALUDE_abs_sqrt3_minus_2_l2898_289816

theorem abs_sqrt3_minus_2 : |Real.sqrt 3 - 2| = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sqrt3_minus_2_l2898_289816


namespace NUMINAMATH_CALUDE_always_odd_l2898_289883

theorem always_odd (n : ℤ) : ∃ k : ℤ, n^2 + n + 5 = 2*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l2898_289883


namespace NUMINAMATH_CALUDE_simplify_cube_roots_l2898_289852

theorem simplify_cube_roots : (512 : ℝ)^(1/3) * (343 : ℝ)^(1/3) = 56 := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_l2898_289852


namespace NUMINAMATH_CALUDE_no_divisors_congruent_to_3_mod_4_and_unique_solution_l2898_289821

theorem no_divisors_congruent_to_3_mod_4_and_unique_solution : 
  (∀ x : ℤ, ∀ d : ℤ, d ∣ (x^2 + 1) → d % 4 ≠ 3) ∧ 
  (∀ x y : ℕ, x^2 - y^3 = 7 ↔ x = 23 ∧ y = 8) := by
  sorry

end NUMINAMATH_CALUDE_no_divisors_congruent_to_3_mod_4_and_unique_solution_l2898_289821


namespace NUMINAMATH_CALUDE_simplify_expression_l2898_289800

theorem simplify_expression (y : ℝ) :
  4 * y + 8 * y^2 + 6 - (3 - 4 * y - 8 * y^2) = 16 * y^2 + 8 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2898_289800


namespace NUMINAMATH_CALUDE_employee_savings_l2898_289881

/-- Calculates the combined savings of three employees over a given period. -/
def combinedSavings (hourlyWage : ℚ) (hoursPerDay : ℚ) (daysPerWeek : ℚ) (weeks : ℚ)
  (savingsRate1 savingsRate2 savingsRate3 : ℚ) : ℚ :=
  let weeklyWage := hourlyWage * hoursPerDay * daysPerWeek
  let totalPeriod := weeklyWage * weeks
  totalPeriod * (savingsRate1 + savingsRate2 + savingsRate3)

/-- The combined savings of three employees with given work conditions and savings rates
    over four weeks is $3000. -/
theorem employee_savings : 
  combinedSavings 10 10 5 4 (2/5) (3/5) (1/2) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_employee_savings_l2898_289881


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l2898_289803

theorem partial_fraction_decomposition_product (M₁ M₂ : ℝ) :
  (∀ x : ℝ, x ≠ 1 → x ≠ 3 → (45 * x - 36) / (x^2 - 4*x + 3) = M₁ / (x - 1) + M₂ / (x - 3)) →
  M₁ * M₂ = -222.75 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l2898_289803


namespace NUMINAMATH_CALUDE_one_fourth_in_five_eighths_l2898_289825

theorem one_fourth_in_five_eighths : (5 / 8 : ℚ) / (1 / 4 : ℚ) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_in_five_eighths_l2898_289825


namespace NUMINAMATH_CALUDE_distance_origin_to_point_l2898_289836

theorem distance_origin_to_point :
  let x : ℝ := 20
  let y : ℝ := 21
  Real.sqrt (x^2 + y^2) = 29 :=
by sorry

end NUMINAMATH_CALUDE_distance_origin_to_point_l2898_289836


namespace NUMINAMATH_CALUDE_rectangle_area_l2898_289817

/-- Given a rectangle where the length is 15% more than the breadth and the breadth is 20 meters,
    prove that its area is 460 square meters. -/
theorem rectangle_area (b l a : ℝ) : 
  b = 20 →                  -- The breadth is 20 meters
  l = b * 1.15 →            -- The length is 15% more than the breadth
  a = l * b →               -- Area formula
  a = 460 := by sorry       -- The area is 460 square meters

end NUMINAMATH_CALUDE_rectangle_area_l2898_289817


namespace NUMINAMATH_CALUDE_starters_count_l2898_289835

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 5 starters from a team of 12 players,
    including a set of twins, with at most one of the twins in the starting lineup -/
def chooseStarters : ℕ :=
  choose 10 5 + 2 * choose 10 4

theorem starters_count : chooseStarters = 672 := by sorry

end NUMINAMATH_CALUDE_starters_count_l2898_289835
