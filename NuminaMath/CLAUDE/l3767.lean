import Mathlib

namespace NUMINAMATH_CALUDE_conference_handshakes_l3767_376799

/-- Represents a conference with specific group dynamics -/
structure Conference where
  total_people : Nat
  group_a_size : Nat
  group_b_size : Nat
  exceptions : Nat
  unknown_per_exception : Nat

/-- Calculates the number of handshakes in the conference -/
def handshakes (c : Conference) : Nat :=
  let group_a_b_handshakes := c.group_a_size * c.group_b_size
  let group_b_internal_handshakes := c.group_b_size * (c.group_b_size - 1) / 2
  let exception_handshakes := c.exceptions * c.unknown_per_exception
  group_a_b_handshakes + group_b_internal_handshakes + exception_handshakes

/-- The theorem to be proved -/
theorem conference_handshakes :
  let c := Conference.mk 40 25 15 5 3
  handshakes c = 495 := by
  sorry

#eval handshakes (Conference.mk 40 25 15 5 3)

end NUMINAMATH_CALUDE_conference_handshakes_l3767_376799


namespace NUMINAMATH_CALUDE_triangle_side_square_sum_bound_l3767_376746

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_radius : 0 < R
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The sum of squares of triangle sides is less than or equal to 9 times the square of its circumradius -/
theorem triangle_side_square_sum_bound (t : Triangle) : t.a^2 + t.b^2 + t.c^2 ≤ 9 * t.R^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_square_sum_bound_l3767_376746


namespace NUMINAMATH_CALUDE_quadratic_root_constant_l3767_376789

/-- 
Given a quadratic equation 5x^2 + 6x + k = 0 with roots (-3 ± √69) / 10,
prove that k = -1.65
-/
theorem quadratic_root_constant (k : ℝ) : 
  (∀ x : ℝ, 5 * x^2 + 6 * x + k = 0 ↔ x = (-3 - Real.sqrt 69) / 10 ∨ x = (-3 + Real.sqrt 69) / 10) →
  k = -1.65 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_constant_l3767_376789


namespace NUMINAMATH_CALUDE_largest_sum_is_923_l3767_376705

def digits : List Nat := [3, 5, 7, 8, 0]

def is_valid_partition (a b : List Nat) : Prop :=
  a.length = 3 ∧ b.length = 2 ∧ (a ++ b).toFinset = digits.toFinset

def to_number (l : List Nat) : Nat :=
  l.foldl (fun acc d => acc * 10 + d) 0

theorem largest_sum_is_923 :
  ∀ a b : List Nat,
    is_valid_partition a b →
    to_number a + to_number b ≤ 923 :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_is_923_l3767_376705


namespace NUMINAMATH_CALUDE_club_selection_theorem_l3767_376788

/-- The number of ways to choose a president, vice-president, and secretary from a club -/
def club_selection_ways (total_members boys girls : ℕ) : ℕ :=
  let president_vp_ways := boys * girls + girls * boys
  let secretary_ways := boys * (boys - 1) + girls * (girls - 1)
  president_vp_ways * secretary_ways

/-- Theorem stating the number of ways to choose club positions under specific conditions -/
theorem club_selection_theorem :
  club_selection_ways 25 15 10 = 90000 :=
by sorry

end NUMINAMATH_CALUDE_club_selection_theorem_l3767_376788


namespace NUMINAMATH_CALUDE_tan_15_plus_3sin_15_l3767_376798

theorem tan_15_plus_3sin_15 : 
  Real.tan (15 * π / 180) + 3 * Real.sin (15 * π / 180) = 
    (Real.sqrt 6 - Real.sqrt 2 + 3) / (Real.sqrt 6 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_tan_15_plus_3sin_15_l3767_376798


namespace NUMINAMATH_CALUDE_one_third_and_three_eightyone_in_cantor_cantor_iteration_length_l3767_376730

/-- The Cantor set constructed by repeatedly removing the middle third of each interval --/
def CantorSet : Set ℝ :=
  sorry

/-- The nth iteration in the Cantor set construction --/
def CantorIteration (n : ℕ) : Set (Set ℝ) :=
  sorry

/-- The length of the nth iteration in the Cantor set construction --/
def CantorIterationLength (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating that 1/3 and 3/81 belong to the Cantor set --/
theorem one_third_and_three_eightyone_in_cantor :
  (1/3 : ℝ) ∈ CantorSet ∧ (3/81 : ℝ) ∈ CantorSet :=
sorry

/-- Theorem stating the length of the nth iteration in the Cantor set construction --/
theorem cantor_iteration_length (n : ℕ) :
  CantorIterationLength n = (2/3 : ℝ) ^ (n - 1) :=
sorry

end NUMINAMATH_CALUDE_one_third_and_three_eightyone_in_cantor_cantor_iteration_length_l3767_376730


namespace NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l3767_376702

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 10*x^3 + 25*x^2 + 2*x - 12

-- State the theorem
theorem monic_quartic_with_specific_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 10*x^3 + 25*x^2 + 2*x - 12) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- 3 + √5 is a root
  p (3 + Real.sqrt 5) = 0 ∧
  -- 2 - √7 is a root
  p (2 - Real.sqrt 7) = 0 :=
by sorry


end NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l3767_376702


namespace NUMINAMATH_CALUDE_hexagon_side_length_l3767_376754

/-- The length of one side of a regular hexagon with perimeter 43.56 -/
theorem hexagon_side_length : ∃ (s : ℝ), s > 0 ∧ s * 6 = 43.56 ∧ s = 7.26 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l3767_376754


namespace NUMINAMATH_CALUDE_extreme_value_conditions_max_min_values_l3767_376764

/-- The function f(x) = x^3 + 3ax^2 + bx -/
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x

/-- The derivative of f(x) -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + b

theorem extreme_value_conditions (a b : ℝ) :
  f a b (-1) = 0 ∧ f_deriv a b (-1) = 0 →
  a = 2/3 ∧ b = 1 :=
sorry

theorem max_min_values (a b : ℝ) :
  a = 2/3 ∧ b = 1 →
  (∀ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x ≤ 0) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x = 0) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x = -2) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_conditions_max_min_values_l3767_376764


namespace NUMINAMATH_CALUDE_terms_before_four_l3767_376748

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

theorem terms_before_four (a₁ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 100 ∧ d = -6 ∧ arithmetic_sequence a₁ d n = 4 → n - 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_terms_before_four_l3767_376748


namespace NUMINAMATH_CALUDE_fraction_problem_l3767_376775

theorem fraction_problem (f : ℚ) : f * 50 - 4 = 6 → f = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3767_376775


namespace NUMINAMATH_CALUDE_cafeteria_pies_l3767_376712

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) :
  initial_apples = 75 →
  handed_out = 19 →
  apples_per_pie = 8 →
  (initial_apples - handed_out) / apples_per_pie = 7 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l3767_376712


namespace NUMINAMATH_CALUDE_obtuse_triangles_in_100gon_l3767_376771

/-- The number of vertices in the regular polygon -/
def n : ℕ := 100

/-- A function that determines if three vertices form an obtuse triangle in a regular n-gon -/
def is_obtuse (k l m : Fin n) : Prop :=
  (m - k : ℕ) % n > n / 4

/-- The number of ways to choose three vertices forming an obtuse triangle in a regular n-gon -/
def num_obtuse_triangles : ℕ := n * (n / 2 - 1).choose 2

/-- Theorem stating the number of obtuse triangles in a regular 100-gon -/
theorem obtuse_triangles_in_100gon :
  num_obtuse_triangles = 117600 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangles_in_100gon_l3767_376771


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_f_lower_bound_l3767_376710

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x - 1|

-- Theorem for part I
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≥ 2} = Set.Ici (4/3) ∪ Set.Iic 0 := by sorry

-- Theorem for part II
theorem f_lower_bound (a x : ℝ) :
  f a x ≥ |a - 1/2| := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_f_lower_bound_l3767_376710


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3767_376741

def polynomial (p q : ℚ) (x : ℚ) : ℚ :=
  x^6 - x^5 + x^4 - p*x^3 + q*x^2 + 6*x - 8

theorem polynomial_divisibility (p q : ℚ) :
  (∀ x, (x + 2 = 0 ∨ x - 1 = 0 ∨ x - 3 = 0) → polynomial p q x = 0) ↔
  (p = -26/3 ∧ q = -26/3) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3767_376741


namespace NUMINAMATH_CALUDE_smallest_distance_complex_circles_l3767_376797

theorem smallest_distance_complex_circles (z w : ℂ) 
  (hz : Complex.abs (z + 1 + 3*I) = 1)
  (hw : Complex.abs (w - 7 - 8*I) = 3) :
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), 
      Complex.abs (z' + 1 + 3*I) = 1 → 
      Complex.abs (w' - 7 - 8*I) = 3 → 
      Complex.abs (z' - w') ≥ min_dist) ∧
    (∃ (z₀ w₀ : ℂ), 
      Complex.abs (z₀ + 1 + 3*I) = 1 ∧ 
      Complex.abs (w₀ - 7 - 8*I) = 3 ∧ 
      Complex.abs (z₀ - w₀) = min_dist) ∧
    min_dist = Real.sqrt 185 - 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_complex_circles_l3767_376797


namespace NUMINAMATH_CALUDE_correct_selection_ways_l3767_376717

def total_students : ℕ := 50
def class_leaders : ℕ := 2
def students_to_select : ℕ := 5

def selection_ways : ℕ := sorry

theorem correct_selection_ways :
  selection_ways = Nat.choose class_leaders 1 * Nat.choose (total_students - class_leaders) 4 +
                   Nat.choose class_leaders 2 * Nat.choose (total_students - class_leaders) 3 ∧
  selection_ways = Nat.choose total_students students_to_select - 
                   Nat.choose (total_students - class_leaders) students_to_select ∧
  selection_ways = Nat.choose class_leaders 1 * Nat.choose (total_students - 1) 4 - 
                   Nat.choose (total_students - class_leaders) 3 ∧
  selection_ways ≠ Nat.choose class_leaders 1 * Nat.choose (total_students - 1) 4 :=
by sorry

end NUMINAMATH_CALUDE_correct_selection_ways_l3767_376717


namespace NUMINAMATH_CALUDE_triangle_rotation_l3767_376778

theorem triangle_rotation (α β γ : ℝ) (k m : ℤ) (h1 : 15 * α = 360 * k)
    (h2 : 6 * β = 360 * m) (h3 : α + β + γ = 180) :
  ∃ (n : ℕ+), n * γ = 360 * (n / 5 : ℤ) ∧ ∀ (n' : ℕ+), n' < n → ¬(∃ (l : ℤ), n' * γ = 360 * l) := by
  sorry

end NUMINAMATH_CALUDE_triangle_rotation_l3767_376778


namespace NUMINAMATH_CALUDE_square_of_binomial_l3767_376758

theorem square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 20 * x + 9 = (r * x + s)^2) → 
  a = 100 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3767_376758


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3767_376701

/-- Given two quadratic equations where the roots of one are three times the roots of the other,
    prove that the ratio of certain coefficients is 27. -/
theorem quadratic_root_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ (s₁ s₂ : ℝ),
    (s₁ + s₂ = -p ∧ s₁ * s₂ = m) ∧
    (3*s₁ + 3*s₂ = -m ∧ 9*s₁ * s₂ = n)) →
  n / p = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3767_376701


namespace NUMINAMATH_CALUDE_sphere_only_identical_views_l3767_376757

-- Define the possible geometric bodies
inductive GeometricBody
  | Sphere
  | Cube
  | RegularTetrahedron

-- Define a function that checks if all views are identical
def hasIdenticalViews (body : GeometricBody) : Prop :=
  match body with
  | GeometricBody.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_only_identical_views :
  ∀ (body : GeometricBody),
    hasIdenticalViews body ↔ body = GeometricBody.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_only_identical_views_l3767_376757


namespace NUMINAMATH_CALUDE_sticker_problem_l3767_376731

theorem sticker_problem (x : ℝ) : 
  (x * (1 - 0.25) * (1 - 0.20) = 45) → x = 75 := by
sorry

end NUMINAMATH_CALUDE_sticker_problem_l3767_376731


namespace NUMINAMATH_CALUDE_joan_apple_count_l3767_376725

/-- Theorem: Given Joan picked 43 apples initially and Melanie gave her 27 more apples, Joan now has 70 apples. -/
theorem joan_apple_count (initial_apples : ℕ) (given_apples : ℕ) (total_apples : ℕ) : 
  initial_apples = 43 → given_apples = 27 → total_apples = initial_apples + given_apples → total_apples = 70 := by
  sorry

end NUMINAMATH_CALUDE_joan_apple_count_l3767_376725


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3767_376790

theorem arithmetic_sequence_solution (x : ℝ) (h1 : x ≠ 0) :
  (x - Int.floor x) + (Int.floor x + 1) + x = 3 * ((Int.floor x + 1)) →
  x = -2 ∨ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3767_376790


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3767_376793

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3767_376793


namespace NUMINAMATH_CALUDE_perfect_square_function_characterization_l3767_376745

/-- A function g: ℕ → ℕ satisfies the perfect square property if 
    (g(m) + n)(m + g(n)) is a perfect square for all m, n ∈ ℕ -/
def PerfectSquareProperty (g : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, ∃ k : ℕ, (g m + n) * (m + g n) = k * k

/-- The main theorem characterizing functions with the perfect square property -/
theorem perfect_square_function_characterization :
  ∀ g : ℕ → ℕ, PerfectSquareProperty g ↔ ∃ c : ℕ, ∀ n : ℕ, g n = n + c :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_function_characterization_l3767_376745


namespace NUMINAMATH_CALUDE_count_is_58_l3767_376794

/-- A function that generates all permutations of a list -/
def permutations (l : List ℕ) : List (List ℕ) :=
  sorry

/-- A function that converts a list of digits to a number -/
def list_to_number (l : List ℕ) : ℕ :=
  sorry

/-- The set of digits we're working with -/
def digits : List ℕ := [1, 2, 3, 4, 5]

/-- All possible five-digit numbers from the given digits -/
def all_numbers : List ℕ :=
  (permutations digits).map list_to_number

/-- The count of numbers satisfying our conditions -/
def count_numbers : ℕ :=
  (all_numbers.filter (λ n => n > 23145 ∧ n < 43521)).length

theorem count_is_58 : count_numbers = 58 :=
  sorry

end NUMINAMATH_CALUDE_count_is_58_l3767_376794


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_two_sqrt_five_l3767_376742

theorem cube_root_sum_equals_two_sqrt_five :
  (((17 * Real.sqrt 5 + 38) ^ (1/3 : ℝ)) + ((17 * Real.sqrt 5 - 38) ^ (1/3 : ℝ))) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_two_sqrt_five_l3767_376742


namespace NUMINAMATH_CALUDE_opposite_of_abs_neg_half_l3767_376723

theorem opposite_of_abs_neg_half : 
  -(|(-0.5 : ℝ)|) = -0.5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_abs_neg_half_l3767_376723


namespace NUMINAMATH_CALUDE_odd_function_parallelicity_l3767_376709

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → f (-x) = -f x

/-- A function has parallelicity if there exist two distinct points with parallel tangent lines -/
def HasParallelicity (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
    DifferentiableAt ℝ f x₁ ∧ DifferentiableAt ℝ f x₂ ∧
    deriv f x₁ = deriv f x₂

/-- Theorem: Any odd function defined on (-∞,0)∪(0,+∞) has parallelicity -/
theorem odd_function_parallelicity (f : ℝ → ℝ) (hf : IsOdd f) : HasParallelicity f := by
  sorry


end NUMINAMATH_CALUDE_odd_function_parallelicity_l3767_376709


namespace NUMINAMATH_CALUDE_f_recursive_relation_l3767_376749

def f (n : ℕ) : ℕ := (Finset.range (2 * n + 1)).sum (λ i => i * i)

theorem f_recursive_relation (k : ℕ) : f (k + 1) = f k + (2 * k + 1)^2 + (2 * k + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_f_recursive_relation_l3767_376749


namespace NUMINAMATH_CALUDE_train_speed_l3767_376724

/-- The speed of a train given specific passing times -/
theorem train_speed (t_pole : ℝ) (t_stationary : ℝ) (l_stationary : ℝ) 
  (h_pole : t_pole = 10)
  (h_stationary : t_stationary = 30)
  (h_length : l_stationary = 600) :
  ∃ v : ℝ, v = 30 ∧ v * t_pole = v * t_stationary - l_stationary :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l3767_376724


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3767_376785

theorem necessary_but_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0) ∧
  (∃ x y : ℝ, x = 0 ∧ x^2 + y^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3767_376785


namespace NUMINAMATH_CALUDE_probability_one_person_two_days_l3767_376777

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of days
def num_days : ℕ := 2

-- Define the number of students required each day
def students_per_day : ℕ := 2

-- Define the function to calculate combinations
def C (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the total number of ways to select students for two days
def total_ways : ℕ := (C num_students students_per_day) * (C num_students students_per_day)

-- Define the number of ways exactly 1 person participates for two consecutive days
def favorable_ways : ℕ := (C num_students 1) * (C (num_students - 1) 1) * (C (num_students - 2) 1)

-- State the theorem
theorem probability_one_person_two_days :
  (favorable_ways : ℚ) / total_ways = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_one_person_two_days_l3767_376777


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l3767_376759

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l3767_376759


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3767_376750

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ, (2 * a^2 + 7 * a - 15 = 0) ∧ (2 * b^2 + 7 * b - 15 = 0) → (a - b)^2 = 169/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3767_376750


namespace NUMINAMATH_CALUDE_composition_ratio_l3767_376780

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : (f (g (f 2))) / (g (f (g 2))) = 115 / 73 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l3767_376780


namespace NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l3767_376779

/-- Polar to Cartesian Coordinate Conversion Theorem -/
theorem polar_to_cartesian_conversion (x y ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) ∧
  (x = ρ * Real.cos θ) ∧
  (y = ρ * Real.sin θ) ∧
  (ρ^2 = x^2 + y^2) →
  (x^2 + (y - 2)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l3767_376779


namespace NUMINAMATH_CALUDE_external_diagonal_inequality_l3767_376704

theorem external_diagonal_inequality (a b c x y z : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  x^2 = a^2 + b^2 ∧ y^2 = b^2 + c^2 ∧ z^2 = a^2 + c^2 →
  x^2 + y^2 ≥ z^2 ∧ y^2 + z^2 ≥ x^2 ∧ z^2 + x^2 ≥ y^2 := by sorry

end NUMINAMATH_CALUDE_external_diagonal_inequality_l3767_376704


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l3767_376753

theorem sum_of_roots_cubic_equation : 
  let p (x : ℝ) := 3 * x^3 - 15 * x^2 - 36 * x + 7
  ∃ r s t : ℝ, (p r = 0 ∧ p s = 0 ∧ p t = 0) ∧ (r + s + t = 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l3767_376753


namespace NUMINAMATH_CALUDE_cooking_time_for_remaining_potatoes_l3767_376767

/-- Given a chef cooking potatoes with the following conditions:
  - The total number of potatoes to cook is 15
  - The number of potatoes already cooked is 6
  - Each potato takes 8 minutes to cook
  This theorem proves that the time required to cook the remaining potatoes is 72 minutes. -/
theorem cooking_time_for_remaining_potatoes :
  let total_potatoes : ℕ := 15
  let cooked_potatoes : ℕ := 6
  let cooking_time_per_potato : ℕ := 8
  let remaining_potatoes : ℕ := total_potatoes - cooked_potatoes
  remaining_potatoes * cooking_time_per_potato = 72 := by
  sorry

end NUMINAMATH_CALUDE_cooking_time_for_remaining_potatoes_l3767_376767


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_negation_equivalence_necessary_sufficient_condition_l3767_376726

-- Statement 1
theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  ¬(∀ x : ℝ, |x| > 1 → x > 1) :=
sorry

-- Statement 2
theorem negation_equivalence :
  ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
sorry

-- Statement 3
theorem necessary_sufficient_condition (a b c : ℝ) :
  (a + b + c = 0) ↔ (a * 1^2 + b * 1 + c = 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_negation_equivalence_necessary_sufficient_condition_l3767_376726


namespace NUMINAMATH_CALUDE_dog_age_difference_l3767_376700

/-- The ratio of dog years to human years -/
def dogYearRatio : ℕ := 7

/-- The age of Max (the human) in years -/
def maxAge : ℕ := 3

/-- The age of Max's dog in human years -/
def dogAgeHuman : ℕ := 3

/-- Calculates the age of a dog in dog years given its age in human years -/
def dogAgeInDogYears (humanYears : ℕ) : ℕ := humanYears * dogYearRatio

/-- The difference in years between a dog's age in dog years and its owner's age in human years -/
def ageDifference (humanAge : ℕ) (dogAgeHuman : ℕ) : ℕ :=
  dogAgeInDogYears dogAgeHuman - humanAge

theorem dog_age_difference :
  ageDifference maxAge dogAgeHuman = 18 := by
  sorry

end NUMINAMATH_CALUDE_dog_age_difference_l3767_376700


namespace NUMINAMATH_CALUDE_abigail_report_time_l3767_376752

/-- Calculates the time needed to finish a report given the total words required,
    words already written, and typing speed. -/
def timeToFinishReport (totalWords : ℕ) (writtenWords : ℕ) (wordsPerHalfHour : ℕ) : ℕ :=
  let remainingWords := totalWords - writtenWords
  let wordsPerMinute := wordsPerHalfHour / 30
  remainingWords / wordsPerMinute

/-- Proves that given the conditions in the problem, 
    it will take 80 minutes to finish the report. -/
theorem abigail_report_time : 
  timeToFinishReport 1000 200 300 = 80 := by
  sorry

end NUMINAMATH_CALUDE_abigail_report_time_l3767_376752


namespace NUMINAMATH_CALUDE_circle_center_sum_l3767_376718

/-- Given a circle with equation x^2 + y^2 = 6x + 8y - 48, 
    the sum of the coordinates of its center is 7 -/
theorem circle_center_sum : 
  ∀ (h k : ℝ), 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 8*y - 48 ↔ (x - h)^2 + (y - k)^2 = 2) →
  h + k = 7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3767_376718


namespace NUMINAMATH_CALUDE_car_speed_proof_l3767_376747

theorem car_speed_proof (v : ℝ) : v > 0 →
  (3600 / v - 3600 / 225 = 2) ↔ v = 200 :=
by
  sorry

#check car_speed_proof

end NUMINAMATH_CALUDE_car_speed_proof_l3767_376747


namespace NUMINAMATH_CALUDE_least_number_of_trees_sixty_divisible_by_four_five_six_least_number_of_trees_is_sixty_l3767_376768

theorem least_number_of_trees (n : ℕ) : n > 0 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n → n ≥ 60 := by
  sorry

theorem sixty_divisible_by_four_five_six : 4 ∣ 60 ∧ 5 ∣ 60 ∧ 6 ∣ 60 := by
  sorry

theorem least_number_of_trees_is_sixty :
  ∃ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 4 ∣ m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_trees_sixty_divisible_by_four_five_six_least_number_of_trees_is_sixty_l3767_376768


namespace NUMINAMATH_CALUDE_jessica_attended_games_l3767_376703

theorem jessica_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 6)
  (h2 : missed_games = 4) :
  total_games - missed_games = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_attended_games_l3767_376703


namespace NUMINAMATH_CALUDE_M_subset_N_l3767_376791

def M : Set ℝ := {-1, 1}

def N : Set ℝ := {x | (1 : ℝ) / x < 2}

theorem M_subset_N : M ⊆ N := by sorry

end NUMINAMATH_CALUDE_M_subset_N_l3767_376791


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l3767_376770

theorem jacket_price_reduction (x : ℝ) : 
  (1 - x) * (1 - 0.3) * (1 + 0.9047619047619048) = 1 → x = 0.25 := by
sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l3767_376770


namespace NUMINAMATH_CALUDE_probability_of_specific_three_card_arrangement_l3767_376756

/-- The number of possible arrangements of n distinct objects -/
def numberOfArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The probability of a specific arrangement given n distinct objects -/
def probabilityOfSpecificArrangement (n : ℕ) : ℚ :=
  1 / (numberOfArrangements n)

theorem probability_of_specific_three_card_arrangement :
  probabilityOfSpecificArrangement 3 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_three_card_arrangement_l3767_376756


namespace NUMINAMATH_CALUDE_trajectory_is_hyperbola_l3767_376795

-- Define the two fixed circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define the moving circle
def movingCircle (cx cy r : ℝ) : Prop := ∀ (x y : ℝ), (x - cx)^2 + (y - cy)^2 = r^2

-- Define the tangency condition
def isTangent (cx cy r : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), circle x y ∧ movingCircle cx cy r ∧ (x - cx)^2 + (y - cy)^2 = r^2

-- Define the trajectory of the center of the moving circle
def trajectory (x y : ℝ) : Prop :=
  ∃ (r : ℝ), isTangent x y r circle1 ∧ isTangent x y r circle2

-- Theorem statement
theorem trajectory_is_hyperbola :
  ∃ (a b : ℝ), ∀ (x y : ℝ), trajectory x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_hyperbola_l3767_376795


namespace NUMINAMATH_CALUDE_angle_trigonometry_l3767_376774

theorem angle_trigonometry (a : ℝ) (θ : ℝ) (h : a < 0) :
  let P : ℝ × ℝ := (4*a, 3*a)
  (∃ (r : ℝ), r > 0 ∧ P.1 = r * Real.cos θ ∧ P.2 = r * Real.sin θ) →
  (Real.sin θ = -3/5 ∧ Real.cos θ = -4/5) ∧
  ((1 + 2 * Real.sin (π + θ) * Real.cos (2023 * π - θ)) / 
   (Real.sin (π/2 + θ)^2 - Real.cos (5*π/2 - θ)^2) = 7) :=
by sorry

end NUMINAMATH_CALUDE_angle_trigonometry_l3767_376774


namespace NUMINAMATH_CALUDE_triangle_problem_l3767_376766

theorem triangle_problem (A B C : Real) (BC AB AC : Real) :
  BC = 7 →
  AB = 3 →
  (Real.sin C) / (Real.sin B) = 3/5 →
  AC = 5 ∧ Real.cos A = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3767_376766


namespace NUMINAMATH_CALUDE_average_speed_three_sections_l3767_376729

/-- The average speed of a person traveling on a 1 km street divided into three equal sections,
    with speeds of 4 km/h, 10 km/h, and 6 km/h in each section respectively. -/
theorem average_speed_three_sections (total_distance : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_distance = 1 →
  speed1 = 4 →
  speed2 = 10 →
  speed3 = 6 →
  let section_distance := total_distance / 3
  let time1 := section_distance / speed1
  let time2 := section_distance / speed2
  let time3 := section_distance / speed3
  let total_time := time1 + time2 + time3
  total_distance / total_time = 180 / 31 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_three_sections_l3767_376729


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3767_376743

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 90 ∧ percentage = 50 ∧ final = initial * (1 + percentage / 100) →
  final = 135 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3767_376743


namespace NUMINAMATH_CALUDE_correct_time_to_write_rearrangements_l3767_376706

/-- The number of unique letters in the name --/
def num_letters : ℕ := 8

/-- The number of rearrangements that can be written per minute --/
def rearrangements_per_minute : ℕ := 10

/-- The number of minutes in an hour --/
def minutes_per_hour : ℕ := 60

/-- Calculates the time in hours to write all rearrangements of a name --/
def time_to_write_all_rearrangements : ℚ :=
  (Nat.factorial num_letters : ℚ) / (rearrangements_per_minute * minutes_per_hour)

theorem correct_time_to_write_rearrangements :
  time_to_write_all_rearrangements = 67.2 := by sorry

end NUMINAMATH_CALUDE_correct_time_to_write_rearrangements_l3767_376706


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3767_376784

def i : ℂ := Complex.I

theorem complex_fraction_simplification :
  (1 + 2*i) / i = -2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3767_376784


namespace NUMINAMATH_CALUDE_power_inequality_l3767_376763

theorem power_inequality (x : ℝ) (h : x < 27) : 27^9 > x^24 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3767_376763


namespace NUMINAMATH_CALUDE_sqrt_equation_root_l3767_376728

theorem sqrt_equation_root : 
  ∃ x : ℝ, x = 35.0625 ∧ Real.sqrt (x - 2) + Real.sqrt (x + 4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_root_l3767_376728


namespace NUMINAMATH_CALUDE_set_union_problem_l3767_376792

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 2, 3} →
  B = {2, a} →
  A ∪ B = {0, 1, 2, 3} →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l3767_376792


namespace NUMINAMATH_CALUDE_problem_solution_l3767_376735

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + a

-- State the theorem
theorem problem_solution :
  -- Part 1: Find the value of a
  (∃ (a : ℝ), ∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  -- Part 2: Find the minimum value of m
  (let a := 1 -- Use the value of a found in part 1
   ∃ (m : ℝ), (∃ (n : ℝ), f n a ≤ m - f (-n) a) ∧
              ∀ (m' : ℝ), (∃ (n : ℝ), f n a ≤ m' - f (-n) a) → m' ≥ m) ∧
  -- The actual solutions
  (let a := 1
   let m := 4
   (∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
   (∃ (n : ℝ), f n a ≤ m - f (-n) a) ∧
   ∀ (m' : ℝ), (∃ (n : ℝ), f n a ≤ m' - f (-n) a) → m' ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3767_376735


namespace NUMINAMATH_CALUDE_midpoints_form_equilateral_triangle_l3767_376776

/-- A hexagon inscribed in a unit circle with alternate sides of length 1 -/
structure InscribedHexagon where
  /-- The vertices of the hexagon -/
  vertices : Fin 6 → ℝ × ℝ
  /-- The hexagon is inscribed in a unit circle -/
  inscribed : ∀ i, (vertices i).1^2 + (vertices i).2^2 = 1
  /-- Alternate sides have length 1 -/
  alt_sides_length : ∀ i, dist (vertices i) (vertices ((i + 1) % 6)) = 1 ∨ 
                           dist (vertices ((i + 1) % 6)) (vertices ((i + 2) % 6)) = 1

/-- The midpoints of the three sides that don't have length 1 -/
def midpoints (h : InscribedHexagon) : Fin 3 → ℝ × ℝ := sorry

/-- The theorem statement -/
theorem midpoints_form_equilateral_triangle (h : InscribedHexagon) : 
  ∀ i j, dist (midpoints h i) (midpoints h j) = dist (midpoints h 0) (midpoints h 1) :=
sorry

end NUMINAMATH_CALUDE_midpoints_form_equilateral_triangle_l3767_376776


namespace NUMINAMATH_CALUDE_circle_passes_through_intersections_and_tangent_to_line_l3767_376783

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0
def l (x y : ℝ) : Prop := x + 2*y = 0

-- Define the desired circle
def desiredCircle (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1)^2 = 5/4

-- Theorem statement
theorem circle_passes_through_intersections_and_tangent_to_line :
  ∀ x y : ℝ,
  (C₁ x y ∧ C₂ x y → desiredCircle x y) ∧
  (∃ t : ℝ, l (1/2 + t) (1 - t/2) ∧
    ∀ s : ℝ, s ≠ t → ¬(desiredCircle (1/2 + s) (1 - s/2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_intersections_and_tangent_to_line_l3767_376783


namespace NUMINAMATH_CALUDE_goldfish_cost_price_l3767_376727

theorem goldfish_cost_price (selling_price : ℝ) (goldfish_sold : ℕ) (tank_cost : ℝ) (profit_percentage : ℝ) :
  selling_price = 0.75 →
  goldfish_sold = 110 →
  tank_cost = 100 →
  profit_percentage = 0.55 →
  ∃ (cost_price : ℝ),
    cost_price = 0.25 ∧
    (goldfish_sold : ℝ) * (selling_price - cost_price) = profit_percentage * tank_cost :=
by sorry

end NUMINAMATH_CALUDE_goldfish_cost_price_l3767_376727


namespace NUMINAMATH_CALUDE_kite_cost_l3767_376736

theorem kite_cost (initial_amount : ℕ) (frisbee_cost : ℕ) (remaining_amount : ℕ) (kite_cost : ℕ) : 
  initial_amount = 78 →
  frisbee_cost = 9 →
  remaining_amount = 61 →
  initial_amount = kite_cost + frisbee_cost + remaining_amount →
  kite_cost = 8 := by
sorry

end NUMINAMATH_CALUDE_kite_cost_l3767_376736


namespace NUMINAMATH_CALUDE_total_bookmark_sales_l3767_376734

/-- Represents the sales of bookmarks over two days -/
structure BookmarkSales where
  /-- Number of bookmarks sold on the first day -/
  day1 : ℕ
  /-- Number of bookmarks sold on the second day -/
  day2 : ℕ

/-- Theorem stating that the total number of bookmarks sold over two days is 3m-3 -/
theorem total_bookmark_sales (m : ℕ) (sales : BookmarkSales)
    (h1 : sales.day1 = m)
    (h2 : sales.day2 = 2 * m - 3) :
    sales.day1 + sales.day2 = 3 * m - 3 := by
  sorry

end NUMINAMATH_CALUDE_total_bookmark_sales_l3767_376734


namespace NUMINAMATH_CALUDE_set_D_is_empty_l3767_376720

def set_D : Set ℝ := {x : ℝ | x^2 - x + 1 = 0}

theorem set_D_is_empty : set_D = ∅ := by
  sorry

end NUMINAMATH_CALUDE_set_D_is_empty_l3767_376720


namespace NUMINAMATH_CALUDE_at_least_one_negative_l3767_376740

theorem at_least_one_negative (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 + 1/b = b^2 + 1/a) :
  a < 0 ∨ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l3767_376740


namespace NUMINAMATH_CALUDE_integer_part_sqrt_39_minus_3_l3767_376713

theorem integer_part_sqrt_39_minus_3 : 
  ⌊Real.sqrt 39 - 3⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_integer_part_sqrt_39_minus_3_l3767_376713


namespace NUMINAMATH_CALUDE_opposite_numbers_and_reciprocal_l3767_376760

theorem opposite_numbers_and_reciprocal (a b c : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : 1 / c = 4)  -- the reciprocal of c is 4
  : 3 * a + 3 * b - 4 * c = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_and_reciprocal_l3767_376760


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3767_376737

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  (∀ x y : ℝ, 2 * x + y = 2 → x * y > 0 → 1 / m + 2 / n ≤ 1 / x + 2 / y) ∧
  (∃ x y : ℝ, 2 * x + y = 2 ∧ x * y > 0 ∧ 1 / x + 2 / y = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3767_376737


namespace NUMINAMATH_CALUDE_simplify_fraction_l3767_376765

theorem simplify_fraction : 15 * (16 / 9) * (-45 / 32) = -25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3767_376765


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3767_376711

/-- The complex number z = (2-i)/(1-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - I) / (1 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3767_376711


namespace NUMINAMATH_CALUDE_volume_of_sphere_wedge_l3767_376782

/-- The volume of a wedge when a sphere with circumference 18π is cut into 6 congruent parts -/
theorem volume_of_sphere_wedge : 
  ∀ (r : ℝ) (V : ℝ),
  (2 * Real.pi * r = 18 * Real.pi) →  -- Circumference condition
  (V = (4/3) * Real.pi * r^3) →       -- Volume of sphere formula
  (V / 6 = 162 * Real.pi) :=          -- Volume of one wedge
by sorry

end NUMINAMATH_CALUDE_volume_of_sphere_wedge_l3767_376782


namespace NUMINAMATH_CALUDE_problem_statement_l3767_376716

theorem problem_statement (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3767_376716


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3767_376722

theorem sum_of_fractions : (1 : ℚ) / 9 + (1 : ℚ) / 11 = 20 / 99 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3767_376722


namespace NUMINAMATH_CALUDE_rectangle_area_divisible_by_12_l3767_376738

theorem rectangle_area_divisible_by_12 (a b c : ℕ) 
  (h1 : a * a + b * b = c * c) : 
  12 ∣ (a * b) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_divisible_by_12_l3767_376738


namespace NUMINAMATH_CALUDE_parabolas_same_vertex_l3767_376773

/-- 
Two parabolas have the same vertex if and only if their coefficients satisfy specific relations.
-/
theorem parabolas_same_vertex (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (∃ (x y : ℝ), 
    (x = -b / (2 * a) ∧ y = a * x^2 + b * x + c) ∧
    (x = -c / (2 * b) ∧ y = b * x^2 + c * x + a)) ↔
  (b = -2 * a ∧ c = 4 * a) :=
sorry

end NUMINAMATH_CALUDE_parabolas_same_vertex_l3767_376773


namespace NUMINAMATH_CALUDE_road_cleaning_problem_l3767_376762

/-- The distance between East City and West City in kilometers -/
def total_distance : ℝ := 60

/-- The time it takes Vehicle A to clean the entire road alone in hours -/
def time_A : ℝ := 10

/-- The time it takes Vehicle B to clean the entire road alone in hours -/
def time_B : ℝ := 15

/-- The additional distance cleaned by Vehicle A compared to Vehicle B when they meet, in kilometers -/
def extra_distance_A : ℝ := 12

theorem road_cleaning_problem :
  let speed_A := total_distance / time_A
  let speed_B := total_distance / time_B
  let combined_speed := speed_A + speed_B
  let meeting_time := total_distance / combined_speed
  speed_A * meeting_time - speed_B * meeting_time = extra_distance_A :=
by sorry

#check road_cleaning_problem

end NUMINAMATH_CALUDE_road_cleaning_problem_l3767_376762


namespace NUMINAMATH_CALUDE_figure_perimeter_l3767_376796

theorem figure_perimeter (total_area : ℝ) (square_area : ℝ) (rect_width rect_length : ℝ) :
  total_area = 130 →
  3 * square_area + rect_width * rect_length = total_area →
  rect_length = 2 * rect_width →
  square_area = rect_width ^ 2 →
  (3 * square_area.sqrt + rect_width + rect_length) * 2 = 11 * Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_figure_perimeter_l3767_376796


namespace NUMINAMATH_CALUDE_kate_museum_visits_cost_l3767_376733

/-- Calculates the total amount spent on museum visits over 3 years -/
def total_spent (initial_fee : ℕ) (increased_fee : ℕ) (visits_first_year : ℕ) (visits_per_year_after : ℕ) : ℕ :=
  initial_fee * visits_first_year + increased_fee * visits_per_year_after * 2

/-- Theorem stating the total amount Kate spent on museum visits over 3 years -/
theorem kate_museum_visits_cost :
  let initial_fee := 5
  let increased_fee := 7
  let visits_first_year := 12
  let visits_per_year_after := 4
  total_spent initial_fee increased_fee visits_first_year visits_per_year_after = 116 := by
  sorry

#eval total_spent 5 7 12 4

end NUMINAMATH_CALUDE_kate_museum_visits_cost_l3767_376733


namespace NUMINAMATH_CALUDE_complex_real_condition_l3767_376732

theorem complex_real_condition (m : ℝ) :
  (Complex.I * (m^2 - 2*m - 15) : ℂ).im = 0 → m = 5 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3767_376732


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3767_376755

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 5*x - 20 = 4*x + 25

-- Define a function to represent the sum of solutions
def sum_of_solutions : ℝ := 9

-- Theorem statement
theorem sum_of_quadratic_solutions :
  ∃ (x₁ x₂ : ℝ), 
    quadratic_equation x₁ ∧ 
    quadratic_equation x₂ ∧ 
    x₁ ≠ x₂ ∧
    x₁ + x₂ = sum_of_solutions :=
sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3767_376755


namespace NUMINAMATH_CALUDE_age_difference_l3767_376751

-- Define the ages as natural numbers
def rona_age : ℕ := 8
def rachel_age : ℕ := 2 * rona_age
def collete_age : ℕ := rona_age / 2

-- Theorem statement
theorem age_difference : rachel_age - collete_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3767_376751


namespace NUMINAMATH_CALUDE_sum_local_values_2345_l3767_376714

def local_value (digit : Nat) (place : Nat) : Nat := digit * (10 ^ place)

theorem sum_local_values_2345 :
  let thousands := local_value 2 3
  let hundreds := local_value 3 2
  let tens := local_value 4 1
  let ones := local_value 5 0
  thousands + hundreds + tens + ones = 2345 := by
sorry

end NUMINAMATH_CALUDE_sum_local_values_2345_l3767_376714


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l3767_376715

theorem square_area_equal_perimeter_triangle (a b c : ℝ) (h_triangle : a = 7.2 ∧ b = 9.5 ∧ c = 11.3) :
  let triangle_perimeter := a + b + c
  let square_side := triangle_perimeter / 4
  square_side ^ 2 = 49 := by
sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l3767_376715


namespace NUMINAMATH_CALUDE_f_difference_at_five_l3767_376787

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x

-- State the theorem
theorem f_difference_at_five : f 5 - f (-5) = 50 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_five_l3767_376787


namespace NUMINAMATH_CALUDE_tan_x_minus_pi_sixth_l3767_376761

theorem tan_x_minus_pi_sixth (x : Real) 
  (h : Real.sin (π / 3 - x) = (1 / 2) * Real.cos (x - π / 2)) : 
  Real.tan (x - π / 6) = Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_minus_pi_sixth_l3767_376761


namespace NUMINAMATH_CALUDE_instantaneous_acceleration_at_3s_l3767_376707

-- Define the displacement function
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 6 * t^2

-- Define the acceleration function as the derivative of velocity
def acceleration (t : ℝ) : ℝ := 12 * t

-- Theorem statement
theorem instantaneous_acceleration_at_3s :
  acceleration 3 = 36 := by
  sorry


end NUMINAMATH_CALUDE_instantaneous_acceleration_at_3s_l3767_376707


namespace NUMINAMATH_CALUDE_john_daily_earnings_l3767_376781

/-- Calculate daily earnings from website visits -/
def daily_earnings (visits_per_month : ℕ) (days_per_month : ℕ) (earnings_per_visit : ℚ) : ℚ :=
  (visits_per_month : ℚ) * earnings_per_visit / (days_per_month : ℚ)

/-- Prove that John's daily earnings are $10 -/
theorem john_daily_earnings :
  daily_earnings 30000 30 (1 / 100) = 10 := by
  sorry

end NUMINAMATH_CALUDE_john_daily_earnings_l3767_376781


namespace NUMINAMATH_CALUDE_tangent_line_at_2_sum_formula_min_value_nSn_l3767_376721

/-- The original function -/
def g (x : ℝ) : ℝ := x^2 - 2*x - 11

/-- The tangent line to g(x) at x = 2 -/
def f (x : ℝ) : ℝ := 2*x - 15

/-- The sequence a_n -/
def a (n : ℕ) : ℝ := f n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ := n^2 - 14*n

theorem tangent_line_at_2 : 
  ∀ x, f x = (2 : ℝ) * (x - 2) + g 2 :=
sorry

theorem sum_formula : 
  ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2 :=
sorry

theorem min_value_nSn : 
  ∃ n : ℕ, ∀ m : ℕ, m ≥ 1 → (n : ℝ) * S n ≤ (m : ℝ) * S m ∧ 
  (n : ℝ) * S n = -405 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_sum_formula_min_value_nSn_l3767_376721


namespace NUMINAMATH_CALUDE_given_terms_are_like_l3767_376769

/-- Two algebraic terms are like terms if they have the same variables with the same exponents. -/
def are_like_terms (term1 term2 : String) : Prop := sorry

/-- The first term in the pair. -/
def term1 : String := "-m^2n^3"

/-- The second term in the pair. -/
def term2 : String := "-3n^3m^2"

/-- Theorem stating that the given terms are like terms. -/
theorem given_terms_are_like : are_like_terms term1 term2 := by sorry

end NUMINAMATH_CALUDE_given_terms_are_like_l3767_376769


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l3767_376739

theorem abs_inequality_equivalence (x : ℝ) : 
  |2*x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l3767_376739


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3767_376786

/-- Given vectors a and b in ℝ², prove that the cosine of the angle between them is √5 / 5 -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) : 
  a = (2, -4) → b = (-3, -4) → 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3767_376786


namespace NUMINAMATH_CALUDE_no_horizontal_asymptote_l3767_376719

noncomputable def f (x : ℝ) : ℝ :=
  (18 * x^5 + 12 * x^4 + 4 * x^3 + 9 * x^2 + 5 * x + 3) /
  (3 * x^4 + 2 * x^3 + 8 * x^2 + 3 * x + 1)

theorem no_horizontal_asymptote :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ N, ∀ x > N, |f x - L| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_no_horizontal_asymptote_l3767_376719


namespace NUMINAMATH_CALUDE_gcd_seven_factorial_six_factorial_l3767_376744

theorem gcd_seven_factorial_six_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 6) = Nat.factorial 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_seven_factorial_six_factorial_l3767_376744


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3767_376708

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_function_properties :
  ∀ a b c : ℝ,
  (f' a b 2 = 0) →  -- Extremum at x = 2
  (f' a b 1 = -3) →  -- Tangent line parallel to 3x + y + 2 = 0 at x = 1
  (a = -1 ∧ b = 0) ∧  -- Values of a and b
  (∃ x₁ x₂ : ℝ, f a b c x₁ - f a b c x₂ = 4)  -- Difference between max and min is 4
  := by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3767_376708


namespace NUMINAMATH_CALUDE_conditional_probability_suitable_joint_structure_l3767_376772

/-- The probability of a child having a suitable joint structure given that they have a suitable physique -/
theorem conditional_probability_suitable_joint_structure 
  (total : ℕ) 
  (physique : ℕ) 
  (joint : ℕ) 
  (both : ℕ) 
  (h_total : total = 20)
  (h_physique : physique = 4)
  (h_joint : joint = 5)
  (h_both : both = 2) :
  (both : ℚ) / physique = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_conditional_probability_suitable_joint_structure_l3767_376772
