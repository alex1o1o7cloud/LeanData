import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_expression_l3756_375623

theorem max_value_of_expression (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 2) : 
  (Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2-a)^2 * (2-b)^2 * (2-c)^2)) ≤ 16 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ a' ≤ 2 ∧ 
                    0 ≤ b' ∧ b' ≤ 2 ∧ 
                    0 ≤ c' ∧ c' ≤ 2 ∧ 
                    Real.sqrt (a'^2 * b'^2 * c'^2) + Real.sqrt ((2-a')^2 * (2-b')^2 * (2-c')^2) = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3756_375623


namespace NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l3756_375661

/-- Represents a labeling of a tetrahedron's vertices -/
def TetrahedronLabeling := Fin 4 → Fin 4

/-- Checks if a labeling uses each number exactly once -/
def is_valid_labeling (l : TetrahedronLabeling) : Prop :=
  ∀ i : Fin 4, ∃! j : Fin 4, l j = i

/-- Calculates the sum of labels on a face -/
def face_sum (l : TetrahedronLabeling) (face : Fin 4 → Fin 3) : ℕ :=
  (face 0).val + (face 1).val + (face 2).val

/-- Checks if all face sums are equal -/
def all_face_sums_equal (l : TetrahedronLabeling) (faces : Fin 4 → (Fin 4 → Fin 3)) : Prop :=
  ∀ i j : Fin 4, face_sum l (faces i) = face_sum l (faces j)

/-- The main theorem stating that no valid labeling exists -/
theorem no_valid_tetrahedron_labeling (faces : Fin 4 → (Fin 4 → Fin 3)) :
  ¬∃ l : TetrahedronLabeling, is_valid_labeling l ∧ all_face_sums_equal l faces :=
sorry

end NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l3756_375661


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3756_375657

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem monotonic_decreasing_interval :
  ∀ x y : ℝ, x < y → x < 0 → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3756_375657


namespace NUMINAMATH_CALUDE_eldest_age_is_32_l3756_375633

/-- Represents the ages of three people A, B, and C -/
structure Ages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The present ages are in the ratio 5:7:8 -/
def present_ratio (ages : Ages) : Prop :=
  7 * ages.a = 5 * ages.b ∧ 8 * ages.a = 5 * ages.c

/-- The sum of ages 7 years ago was 59 -/
def past_sum (ages : Ages) : Prop :=
  (ages.a - 7) + (ages.b - 7) + (ages.c - 7) = 59

/-- Theorem stating that given the conditions, the eldest person's age is 32 -/
theorem eldest_age_is_32 (ages : Ages) 
  (h1 : present_ratio ages) 
  (h2 : past_sum ages) : 
  ages.c = 32 := by
  sorry


end NUMINAMATH_CALUDE_eldest_age_is_32_l3756_375633


namespace NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3756_375628

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ
  angle : ℝ  -- Angle from a reference line on the surface

/-- Calculates the shortest distance between two points on the surface of a cone -/
def shortestDistanceOnCone (c : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem shortest_distance_on_specific_cone :
  let c : Cone := { baseRadius := 500, height := 400 }
  let p1 : ConePoint := { distanceFromVertex := 150, angle := 0 }
  let p2 : ConePoint := { distanceFromVertex := 400 * Real.sqrt 2, angle := π }
  shortestDistanceOnCone c p1 p2 = 25 * Real.sqrt 741 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3756_375628


namespace NUMINAMATH_CALUDE_identical_solutions_quadratic_linear_l3756_375602

theorem identical_solutions_quadratic_linear (k : ℝ) :
  (∃! x y : ℝ, y = x^2 ∧ y = 4*x + k ∧ 
   ∀ x' y' : ℝ, y' = x'^2 ∧ y' = 4*x' + k → x' = x ∧ y' = y) ↔ k = -4 :=
by sorry

end NUMINAMATH_CALUDE_identical_solutions_quadratic_linear_l3756_375602


namespace NUMINAMATH_CALUDE_equation_transformation_l3756_375648

theorem equation_transformation (x y : ℝ) (h : y = x - 1/x) :
  x^6 + x^5 - 5*x^4 + 2*x^3 - 5*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l3756_375648


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l3756_375609

/-- Represents the taxi fare structure and proves the cost for a 100-mile ride -/
theorem taxi_fare_calculation (base_fare : ℝ) (rate : ℝ) 
  (h1 : base_fare = 10)
  (h2 : base_fare + 80 * rate = 150) :
  base_fare + 100 * rate = 185 := by
  sorry

#check taxi_fare_calculation

end NUMINAMATH_CALUDE_taxi_fare_calculation_l3756_375609


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3756_375601

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 2 * i) / (1 + 4 * i) = -6 / 17 - (10 / 17) * i :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3756_375601


namespace NUMINAMATH_CALUDE_fermat_like_theorem_l3756_375693

theorem fermat_like_theorem : ∀ (x y z k : ℕ), x < k → y < k → x^k + y^k ≠ z^k := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_theorem_l3756_375693


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3756_375641

theorem sum_of_x_and_y (x y : ℝ) (h : x - 1 = 1 - y) : x + y = 2 := by
  sorry

#check sum_of_x_and_y

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3756_375641


namespace NUMINAMATH_CALUDE_expression_decrease_value_decrease_l3756_375669

theorem expression_decrease (x y : ℝ) (h : x > 0 ∧ y > 0) : 
  (3/4 * x) * (3/4 * y)^2 = (27/64) * (x * y^2) := by
  sorry

theorem value_decrease (x y : ℝ) (h : x > 0 ∧ y > 0) : 
  1 - (3/4 * x) * (3/4 * y)^2 / (x * y^2) = 37/64 := by
  sorry

end NUMINAMATH_CALUDE_expression_decrease_value_decrease_l3756_375669


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3756_375697

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3
  let θ : ℝ := 3 * π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (-3 / Real.sqrt 2, 3 / Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3756_375697


namespace NUMINAMATH_CALUDE_range_of_a_l3756_375615

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + 4 * x + a > 0) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3756_375615


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l3756_375629

/-- Given a line passing through (3, -5) and (k, 21) that is parallel to 4x - 5y = 20, prove k = 35.5 -/
theorem parallel_line_k_value (k : ℝ) :
  (∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ (x = 3 ∧ y = -5) ∨ (x = k ∧ y = 21)) ∧
                 (∀ x y : ℝ, y = (4/5) * x - 4 ↔ 4*x - 5*y = 20)) →
  k = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l3756_375629


namespace NUMINAMATH_CALUDE_positive_reals_inequality_l3756_375642

theorem positive_reals_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = x - y) : x^2 + 4*y^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequality_l3756_375642


namespace NUMINAMATH_CALUDE_bigger_number_problem_l3756_375674

theorem bigger_number_problem (x y : ℝ) 
  (sum_eq : x + y = 77)
  (ratio_eq : 5 * x = 6 * y)
  (x_geq_y : x ≥ y) : 
  x = 42 := by
  sorry

end NUMINAMATH_CALUDE_bigger_number_problem_l3756_375674


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l3756_375649

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a cubic polynomial at a point -/
def CubicPolynomial.eval (P : CubicPolynomial) (x : ℤ) : ℤ :=
  P.a * x^3 + P.b * x^2 + P.c * x + P.d

/-- Property of having infinitely many pairs of distinct integers (x, y) such that xP(x) = yP(y) -/
def has_infinitely_many_equal_products (P : CubicPolynomial) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ x * P.eval x = y * P.eval y ∧ (abs x > n ∨ abs y > n)

/-- Main theorem: If a cubic polynomial with integer coefficients has infinitely many pairs of 
    distinct integers (x, y) such that xP(x) = yP(y), then it has an integer root -/
theorem cubic_polynomial_integer_root (P : CubicPolynomial) 
    (h : has_infinitely_many_equal_products P) : 
    ∃ k : ℤ, P.eval k = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l3756_375649


namespace NUMINAMATH_CALUDE_akeno_spent_more_l3756_375607

def akeno_expenditure : ℕ := 2985
def lev_expenditure : ℕ := akeno_expenditure / 3
def ambrocio_expenditure : ℕ := lev_expenditure - 177

theorem akeno_spent_more :
  akeno_expenditure - (lev_expenditure + ambrocio_expenditure) = 1172 := by
  sorry

end NUMINAMATH_CALUDE_akeno_spent_more_l3756_375607


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l3756_375687

theorem base_conversion_theorem (n A B : ℕ) : 
  (0 < n) →
  (0 ≤ A) ∧ (A < 8) →
  (0 ≤ B) ∧ (B < 6) →
  (n = 8 * A + B) →
  (n = 6 * B + A) →
  n = 47 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l3756_375687


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_range_of_a_l3756_375651

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem 1
theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2
theorem union_A_complement_B : A ∪ (U \ B) = {x | x < 4} := by sorry

-- Theorem 3
theorem range_of_a (h : A ⊆ C a) : a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_range_of_a_l3756_375651


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3756_375681

theorem greatest_prime_factor_of_expression : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (3^8 + 6^7) ∧ ∀ q : ℕ, q.Prime → q ∣ (3^8 + 6^7) → q ≤ p ∧ p = 131 :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3756_375681


namespace NUMINAMATH_CALUDE_max_min_difference_l3756_375643

def f (x : ℝ) : ℝ := x^3 - 3*x - 1

theorem max_min_difference (M N : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f x ≤ M) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = M) ∧
  (∀ x ∈ Set.Icc (-3) 2, N ≤ f x) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = N) →
  M - N = 20 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_l3756_375643


namespace NUMINAMATH_CALUDE_ms_cole_students_l3756_375685

/-- Represents the number of students in each math level class taught by Ms. Cole -/
structure MathClasses where
  sixth_level : ℕ
  fourth_level : ℕ
  seventh_level : ℕ

/-- Calculates the total number of students Ms. Cole teaches -/
def total_students (classes : MathClasses) : ℕ :=
  classes.sixth_level + classes.fourth_level + classes.seventh_level

/-- Theorem stating the total number of students Ms. Cole teaches -/
theorem ms_cole_students : ∃ (classes : MathClasses), 
  classes.sixth_level = 40 ∧ 
  classes.fourth_level = 4 * classes.sixth_level ∧
  classes.seventh_level = 2 * classes.fourth_level ∧
  total_students classes = 520 := by
  sorry

end NUMINAMATH_CALUDE_ms_cole_students_l3756_375685


namespace NUMINAMATH_CALUDE_town_population_theorem_l3756_375655

theorem town_population_theorem (total_population : ℕ) 
  (females_with_glasses : ℕ) (female_glasses_percentage : ℚ) :
  total_population = 5000 →
  females_with_glasses = 900 →
  female_glasses_percentage = 30/100 →
  (females_with_glasses : ℚ) / female_glasses_percentage = 3000 →
  total_population - 3000 = 2000 := by
sorry

end NUMINAMATH_CALUDE_town_population_theorem_l3756_375655


namespace NUMINAMATH_CALUDE_number_line_relations_l3756_375647

/-- Definition of "A is k related to B" --/
def is_k_related (A B C : ℝ) (k : ℝ) : Prop :=
  |C - A| = k * |C - B| ∧ k > 1

/-- Problem statement --/
theorem number_line_relations (x t k : ℝ) : 
  let A := -3
  let B := 6
  let P := x
  let Q := 6 - 2*t
  (
    /- Part 1 -/
    (is_k_related A B P 2 → x = 3) ∧ 
    
    /- Part 2 -/
    (|x + 2| + |x - 1| = 3 ∧ is_k_related A B P k → 1/8 ≤ k ∧ k ≤ 4/5) ∧
    
    /- Part 3 -/
    (is_k_related (-3 + t) A Q 3 → t = 3/2)
  ) := by sorry

end NUMINAMATH_CALUDE_number_line_relations_l3756_375647


namespace NUMINAMATH_CALUDE_smallest_b_value_l3756_375617

theorem smallest_b_value (a b c : ℕ+) (h : (31 : ℚ) / 72 = a / 8 + b / 9 - c) : 
  ∀ b' : ℕ+, b' < b → ¬∃ (a' c' : ℕ+), (31 : ℚ) / 72 = a' / 8 + b' / 9 - c' :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3756_375617


namespace NUMINAMATH_CALUDE_min_value_of_f_l3756_375686

def f (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 7

theorem min_value_of_f :
  ∃ (y_min : ℝ), ∀ (y : ℝ), f y ≥ f y_min ∧ y_min = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3756_375686


namespace NUMINAMATH_CALUDE_pond_length_l3756_375644

/-- The length of a rectangular pond given its width, depth, and volume. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) 
  (h_width : width = 15)
  (h_depth : depth = 5)
  (h_volume : volume = 1500)
  : volume / (width * depth) = 20 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l3756_375644


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3756_375650

def arrange_books (math_books : ℕ) (english_books : ℕ) : ℕ :=
  let math_group := 1
  let english_groups := 2
  let total_groups := math_group + english_groups
  let math_arrangements := Nat.factorial math_books
  let english_group_size := english_books / english_groups
  let english_group_arrangements := Nat.factorial english_group_size
  Nat.factorial total_groups * math_arrangements * english_group_arrangements * english_group_arrangements

theorem book_arrangement_theorem :
  arrange_books 4 6 = 5184 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3756_375650


namespace NUMINAMATH_CALUDE_sophomore_sample_size_l3756_375678

/-- Represents the number of students selected in a stratified sample -/
def stratifiedSample (totalPopulation : ℕ) (sampleSize : ℕ) (strataSize : ℕ) : ℕ :=
  (strataSize * sampleSize) / totalPopulation

/-- The problem statement -/
theorem sophomore_sample_size :
  let totalStudents : ℕ := 2800
  let sophomores : ℕ := 930
  let sampleSize : ℕ := 280
  stratifiedSample totalStudents sampleSize sophomores = 93 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_sample_size_l3756_375678


namespace NUMINAMATH_CALUDE_circle_area_l3756_375698

theorem circle_area (x y : ℝ) : 
  (4 * x^2 + 4 * y^2 - 8 * x + 24 * y + 60 = 0) → 
  (∃ (center_x center_y radius : ℝ), 
    ((x - center_x)^2 + (y - center_y)^2 = radius^2) ∧ 
    (π * radius^2 = 5 * π)) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l3756_375698


namespace NUMINAMATH_CALUDE_solution_equation_l3756_375604

theorem solution_equation (p q : ℝ) (h1 : p ≠ q) (h2 : p ≠ 0) (h3 : q ≠ 0) :
  ∃ x : ℝ, (x + p)^2 - (x + q)^2 = 4*(p-q)^2 ∧ x = 2*p - 2*q :=
by sorry

end NUMINAMATH_CALUDE_solution_equation_l3756_375604


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l3756_375670

theorem parallelogram_side_length 
  (s : ℝ) 
  (side1 : ℝ) 
  (side2 : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h : side1 = 3 * s) 
  (h' : side2 = s) 
  (h'' : angle = π / 3) 
  (h''' : area = 9 * Real.sqrt 3) 
  (h'''' : area = side2 * side1 * Real.sin angle) : 
  s = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l3756_375670


namespace NUMINAMATH_CALUDE_find_m_l3756_375608

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

-- State the theorem
theorem find_m : ∃ m : ℝ, 
  (∀ x : ℝ, f (x - 1) = |x| - |x - 2|) ∧ 
  f (f m) = f 2002 - 7/2 → 
  m = -3/8 := by sorry

end NUMINAMATH_CALUDE_find_m_l3756_375608


namespace NUMINAMATH_CALUDE_min_distance_complex_l3756_375688

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ w, Complex.abs (z + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l3756_375688


namespace NUMINAMATH_CALUDE_hoseok_position_l3756_375680

theorem hoseok_position (n : Nat) (h : n = 9) :
  ∀ (position_tallest : Nat), position_tallest = 5 →
    n + 1 - position_tallest = 5 :=
by sorry

end NUMINAMATH_CALUDE_hoseok_position_l3756_375680


namespace NUMINAMATH_CALUDE_yeonseo_skirt_count_l3756_375632

/-- Given that Yeonseo has more than two types of skirts and pants each,
    there are 4 types of pants, and 7 ways to choose pants or skirts,
    prove that the number of types of skirts is 3. -/
theorem yeonseo_skirt_count :
  ∀ (S P : ℕ),
  S > 2 →
  P > 2 →
  P = 4 →
  S + P = 7 →
  S = 3 := by
sorry

end NUMINAMATH_CALUDE_yeonseo_skirt_count_l3756_375632


namespace NUMINAMATH_CALUDE_log_stack_sum_l3756_375621

/-- 
Given a stack of logs where:
- The bottom row has 15 logs
- Each successive row has one less log
- The top row has 5 logs
Prove that the total number of logs in the stack is 110.
-/
theorem log_stack_sum : 
  ∀ (a l n : ℕ), 
    a = 15 → 
    l = 5 → 
    n = a - l + 1 → 
    (n : ℚ) / 2 * (a + l) = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l3756_375621


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3756_375639

/-- Given a point (a,b) outside a circle x^2 + y^2 = r^2 (r ≠ 0),
    prove that the line ax + by = r^2 intersects the circle. -/
theorem line_intersects_circle 
  (a b r : ℝ) 
  (r_nonzero : r ≠ 0)
  (point_outside : a^2 + b^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ a*x + b*y = r^2 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3756_375639


namespace NUMINAMATH_CALUDE_B_equals_D_l3756_375616

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}

-- Define set D (real numbers not less than 1)
def D : Set ℝ := {y : ℝ | y ≥ 1}

-- Theorem statement
theorem B_equals_D : B = D := by sorry

end NUMINAMATH_CALUDE_B_equals_D_l3756_375616


namespace NUMINAMATH_CALUDE_equation_solution_l3756_375694

theorem equation_solution : ∃ X : ℝ, 
  (1.5 * ((3.6 * 0.48 * X) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002) ∧ 
  (abs (X - 2.5) < 0.0000000000000005) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3756_375694


namespace NUMINAMATH_CALUDE_inequality_range_l3756_375690

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l3756_375690


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3756_375636

/-- An arithmetic sequence starting with 1, having a common difference of -2, and ending with -89, has 46 terms. -/
theorem arithmetic_sequence_length :
  ∀ (a : ℕ → ℤ), 
    (a 0 = 1) →  -- First term is 1
    (∀ n, a (n + 1) - a n = -2) →  -- Common difference is -2
    (∃ N, a N = -89 ∧ ∀ k, k > N → a k < -89) →  -- Sequence ends at -89
    (∃ N, N = 46 ∧ a (N - 1) = -89) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3756_375636


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3756_375638

theorem fraction_sum_equality : 
  (2 + 4 + 6 + 8 + 10) / (1 + 3 + 5 + 7 + 9) + 
  (1 + 3 + 5 + 7 + 9) / (2 + 4 + 6 + 8 + 10) = 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3756_375638


namespace NUMINAMATH_CALUDE_toms_age_l3756_375653

theorem toms_age (j t : ℕ) 
  (h1 : j - 6 = 3 * (t - 6))  -- John was thrice as old as Tom 6 years ago
  (h2 : j + 4 = 2 * (t + 4))  -- John will be 2 times as old as Tom in 4 years
  : t = 16 := by  -- Tom's current age is 16
  sorry

end NUMINAMATH_CALUDE_toms_age_l3756_375653


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l3756_375673

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  blue_white : ℕ

/-- Given the initial triangle counts and the number of coinciding pairs of various types,
    calculates the number of white-white pairs that coincide when the figure is folded -/
def calculate_white_pairs (counts : TriangleCounts) (pairs : CoincidingPairs) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, 5 white pairs coincide -/
theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) 
  (h1 : counts.red = 5)
  (h2 : counts.blue = 6)
  (h3 : counts.white = 9)
  (h4 : pairs.red_red = 3)
  (h5 : pairs.blue_blue = 2)
  (h6 : pairs.red_white = 3)
  (h7 : pairs.blue_white = 1) :
  calculate_white_pairs counts pairs = 5 :=
by sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l3756_375673


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3756_375626

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a (-2))
  (h_sum : (Finset.range 33).sum (fun i => a (3 * i + 1)) = 50) :
  (Finset.range 33).sum (fun i => a (3 * i + 3)) = -82 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3756_375626


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_relation_l3756_375600

-- Define a structure for a trapezoid
structure Trapezoid where
  a : ℝ  -- larger base
  c : ℝ  -- smaller base
  e : ℝ  -- diagonal
  f : ℝ  -- diagonal
  d : ℝ  -- side
  b : ℝ  -- side
  h_ac : a > c  -- condition that a > c

-- Theorem statement
theorem trapezoid_diagonal_relation (T : Trapezoid) :
  (T.e^2 + T.f^2) / (T.a^2 - T.b^2) = (T.a + T.c) / (T.a - T.c) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_relation_l3756_375600


namespace NUMINAMATH_CALUDE_x_value_l3756_375640

variables (x y z k l m : ℝ)

theorem x_value (h1 : x * y = k * (x + y))
                (h2 : x * z = l * (x + z))
                (h3 : y * z = m * (y + z))
                (hk : k ≠ 0) (hl : l ≠ 0) (hm : m ≠ 0)
                (hkl : k * l + k * m - l * m ≠ 0) :
  x = (2 * k * l * m) / (k * l + k * m - l * m) :=
by sorry

end NUMINAMATH_CALUDE_x_value_l3756_375640


namespace NUMINAMATH_CALUDE_no_discount_possible_l3756_375675

theorem no_discount_possible (purchase_price : ℝ) (marked_price_each : ℝ) 
  (h1 : purchase_price = 50)
  (h2 : marked_price_each = 22.5) :
  2 * marked_price_each < purchase_price := by
  sorry

#eval 2 * 22.5 -- This will output 45.0, confirming the contradiction

end NUMINAMATH_CALUDE_no_discount_possible_l3756_375675


namespace NUMINAMATH_CALUDE_product_35_42_base7_l3756_375619

/-- Converts a base-7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a base-7 number --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Main theorem --/
theorem product_35_42_base7 :
  let a := toDecimal 35
  let b := toDecimal 42
  let product := a * b
  let base7Product := toBase7 product
  let digitSum := sumOfDigitsBase7 base7Product
  digitSum = 5 ∧ digitSum % 5 = 5 := by sorry

end NUMINAMATH_CALUDE_product_35_42_base7_l3756_375619


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3756_375691

theorem cube_root_equation_solution :
  ∀ x : ℝ, (5 + x / 3) ^ (1/3 : ℝ) = 2 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3756_375691


namespace NUMINAMATH_CALUDE_expression_evaluation_l3756_375610

theorem expression_evaluation :
  let d : ℕ := 4
  (d^d - d*(d-2)^d + d^2)^(d-1) = 9004736 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3756_375610


namespace NUMINAMATH_CALUDE_pentagon_condition_l3756_375625

/-- Represents the lengths of five segments cut from a wire -/
structure WireSegments where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  sum_eq_two : a + b + c + d + e = 2
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e

/-- Checks if the given segments can form a pentagon -/
def can_form_pentagon (segments : WireSegments) : Prop :=
  segments.a + segments.b + segments.c + segments.d > segments.e ∧
  segments.a + segments.b + segments.c + segments.e > segments.d ∧
  segments.a + segments.b + segments.d + segments.e > segments.c ∧
  segments.a + segments.c + segments.d + segments.e > segments.b ∧
  segments.b + segments.c + segments.d + segments.e > segments.a

/-- Theorem stating the necessary and sufficient condition for forming a pentagon -/
theorem pentagon_condition (segments : WireSegments) :
  can_form_pentagon segments ↔ segments.a < 1 ∧ segments.b < 1 ∧ segments.c < 1 ∧ segments.d < 1 ∧ segments.e < 1 :=
sorry

end NUMINAMATH_CALUDE_pentagon_condition_l3756_375625


namespace NUMINAMATH_CALUDE_angle_range_l3756_375652

def a (x : ℝ) : Fin 2 → ℝ := ![2, x]
def b : Fin 2 → ℝ := ![1, 3]

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

def is_acute_angle (v w : Fin 2 → ℝ) : Prop := dot_product v w > 0

theorem angle_range (x : ℝ) :
  is_acute_angle (a x) b → x ∈ {y : ℝ | y > -2/3 ∧ y ≠ -2/3} := by
  sorry

end NUMINAMATH_CALUDE_angle_range_l3756_375652


namespace NUMINAMATH_CALUDE_current_year_is_2021_l3756_375683

-- Define the given conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := kelsey_birth_year - 3
def sister_current_age : ℕ := 50

-- Define the theorem
theorem current_year_is_2021 :
  sister_birth_year + sister_current_age = 2021 :=
sorry

end NUMINAMATH_CALUDE_current_year_is_2021_l3756_375683


namespace NUMINAMATH_CALUDE_two_white_marbles_probability_l3756_375663

/-- The probability of drawing two white marbles consecutively without replacement from a bag containing 5 red marbles and 7 white marbles is 7/22. -/
theorem two_white_marbles_probability :
  let red_marbles : ℕ := 5
  let white_marbles : ℕ := 7
  let total_marbles : ℕ := red_marbles + white_marbles
  let prob_first_white : ℚ := white_marbles / total_marbles
  let prob_second_white : ℚ := (white_marbles - 1) / (total_marbles - 1)
  prob_first_white * prob_second_white = 7 / 22 :=
by sorry

end NUMINAMATH_CALUDE_two_white_marbles_probability_l3756_375663


namespace NUMINAMATH_CALUDE_probability_same_color_girls_marbles_l3756_375692

/-- The probability of all 4 girls selecting the same colored marble -/
def probability_same_color (total_marbles : ℕ) (white_marbles : ℕ) (black_marbles : ℕ) (num_girls : ℕ) : ℚ :=
  let prob_all_white := (white_marbles.factorial * (total_marbles - num_girls).factorial) / 
                        (total_marbles.factorial * (white_marbles - num_girls).factorial)
  let prob_all_black := (black_marbles.factorial * (total_marbles - num_girls).factorial) / 
                        (total_marbles.factorial * (black_marbles - num_girls).factorial)
  prob_all_white + prob_all_black

/-- The theorem stating the probability of all 4 girls selecting the same colored marble -/
theorem probability_same_color_girls_marbles : 
  probability_same_color 8 4 4 4 = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_girls_marbles_l3756_375692


namespace NUMINAMATH_CALUDE_problem_solution_l3756_375660

def P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}
def Q (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem problem_solution :
  (∀ x, x ∉ P ↔ (x < -2 ∨ x > 10)) ∧
  (∀ m, P ⊆ Q m ↔ m ≥ 9) ∧
  (∀ m, P ∩ Q m = Q m ↔ m ≤ 9) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3756_375660


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_r_value_l3756_375611

-- Define the sum of the first n terms of the geometric sequence
def S (n : ℕ) (r : ℚ) : ℚ := 3^(n-1) - r

-- Define the geometric sequence
def a (n : ℕ) (r : ℚ) : ℚ := S n r - S (n-1) r

-- Theorem statement
theorem geometric_sequence_sum_r_value :
  ∃ (r : ℚ), ∀ (n : ℕ), n ≥ 2 → a n r = 2 * 3^(n-2) ∧ a 1 r = 1 - r → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_r_value_l3756_375611


namespace NUMINAMATH_CALUDE_x_value_from_fraction_equality_l3756_375664

theorem x_value_from_fraction_equality (x y : ℝ) :
  x / (x - 1) = (y^2 + 2*y + 3) / (y^2 + 2*y - 2) →
  x = (y^2 + 2*y + 3) / 5 := by
sorry

end NUMINAMATH_CALUDE_x_value_from_fraction_equality_l3756_375664


namespace NUMINAMATH_CALUDE_val_coin_ratio_l3756_375695

theorem val_coin_ratio :
  -- Define the number of nickels Val has initially
  let initial_nickels : ℕ := 20
  -- Define the value of a nickel in cents
  let nickel_value : ℕ := 5
  -- Define the value of a dime in cents
  let dime_value : ℕ := 10
  -- Define the total value in cents after finding additional nickels
  let total_value_after : ℕ := 900
  -- Define the function to calculate the number of additional nickels
  let additional_nickels (n : ℕ) : ℕ := 2 * n
  -- Define the function to calculate the total number of nickels after finding additional ones
  let total_nickels (n : ℕ) : ℕ := n + additional_nickels n
  -- Define the function to calculate the value of nickels in cents
  let nickel_value_cents (n : ℕ) : ℕ := n * nickel_value
  -- Define the function to calculate the value of dimes in cents
  let dime_value_cents (d : ℕ) : ℕ := d * dime_value
  -- Define the function to calculate the number of dimes
  let num_dimes (n : ℕ) : ℕ := (total_value_after - nickel_value_cents (total_nickels n)) / dime_value
  -- The ratio of dimes to nickels is 3:1
  num_dimes initial_nickels / initial_nickels = 3 := by
  sorry

end NUMINAMATH_CALUDE_val_coin_ratio_l3756_375695


namespace NUMINAMATH_CALUDE_specific_glued_cubes_surface_area_l3756_375613

/-- Represents a 3D shape formed by gluing two cubes --/
structure GluedCubes where
  large_edge_length : ℝ
  small_edge_length : ℝ

/-- Calculates the surface area of the GluedCubes shape --/
def surface_area (shape : GluedCubes) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the specific GluedCubes shape is 136 --/
theorem specific_glued_cubes_surface_area :
  ∃ (shape : GluedCubes),
    shape.large_edge_length = 4 ∧
    shape.small_edge_length = 1 ∧
    surface_area shape = 136 :=
  sorry

end NUMINAMATH_CALUDE_specific_glued_cubes_surface_area_l3756_375613


namespace NUMINAMATH_CALUDE_component_qualification_l3756_375603

def lower_limit : ℝ := 20 - 0.05
def upper_limit : ℝ := 20 + 0.02

def is_qualified (diameter : ℝ) : Prop :=
  lower_limit ≤ diameter ∧ diameter ≤ upper_limit

theorem component_qualification :
  is_qualified 19.96 ∧
  ¬is_qualified 19.50 ∧
  ¬is_qualified 20.2 ∧
  ¬is_qualified 20.05 := by
  sorry

end NUMINAMATH_CALUDE_component_qualification_l3756_375603


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l3756_375684

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l3756_375684


namespace NUMINAMATH_CALUDE_equilateral_is_isosceles_l3756_375614

-- Define a triangle type
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

-- Define what it means for a triangle to be equilateral
def IsEquilateral (t : Triangle) : Prop :=
  t.side1 = t.side2 ∧ t.side2 = t.side3

-- Define what it means for a triangle to be isosceles
def IsIsosceles (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side2 = t.side3 ∨ t.side3 = t.side1

-- Theorem: Every equilateral triangle is isosceles
theorem equilateral_is_isosceles (t : Triangle) :
  IsEquilateral t → IsIsosceles t := by
  sorry


end NUMINAMATH_CALUDE_equilateral_is_isosceles_l3756_375614


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l3756_375624

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The 11th term of a geometric sequence is 648, given that its 5th term is 8 and its 8th term is 72. -/
theorem geometric_sequence_11th_term (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h5 : a 5 = 8) (h8 : a 8 = 72) : a 11 = 648 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l3756_375624


namespace NUMINAMATH_CALUDE_lindsay_squat_weight_l3756_375665

/-- The total weight Lindsey will squat -/
def total_weight (num_bands : ℕ) (resistance_per_band : ℕ) (dumbbell_weight : ℕ) : ℕ :=
  num_bands * resistance_per_band + dumbbell_weight

/-- Theorem stating the total weight Lindsey will squat -/
theorem lindsay_squat_weight :
  let num_bands : ℕ := 2
  let resistance_per_band : ℕ := 5
  let dumbbell_weight : ℕ := 10
  total_weight num_bands resistance_per_band dumbbell_weight = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_lindsay_squat_weight_l3756_375665


namespace NUMINAMATH_CALUDE_f_properties_l3756_375679

def f (a x : ℝ) : ℝ := |1 - x - a| + |2 * a - x|

theorem f_properties (a x : ℝ) :
  (f a 1 < 3 ↔ a > -2/3 ∧ a < 4/3) ∧
  (a ≥ 2/3 → f a x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3756_375679


namespace NUMINAMATH_CALUDE_win_sector_area_l3756_375696

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/3) :
  p * π * r^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l3756_375696


namespace NUMINAMATH_CALUDE_sum_of_roots_of_equation_l3756_375668

theorem sum_of_roots_of_equation (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, (r₁ - 7)^2 = 16 ∧ (r₂ - 7)^2 = 16 ∧ r₁ + r₂ = 14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_equation_l3756_375668


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l3756_375666

theorem halfway_between_fractions :
  (3 / 4 + 5 / 6) / 2 = 19 / 24 := by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l3756_375666


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_ending_25_is_90_l3756_375699

/-- A function that returns the count of four-digit numbers divisible by 5 with 25 as their last two digits -/
def count_four_digit_numbers_ending_25 : ℕ :=
  let first_number := 1025
  let last_number := 9925
  (last_number - first_number) / 100 + 1

/-- Theorem stating that the count of four-digit numbers divisible by 5 with 25 as their last two digits is 90 -/
theorem count_four_digit_numbers_ending_25_is_90 :
  count_four_digit_numbers_ending_25 = 90 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_ending_25_is_90_l3756_375699


namespace NUMINAMATH_CALUDE_rectangle_new_perimeter_l3756_375605

/-- Given a rectangle with width 10 meters and original area 150 square meters,
    if its length is increased so that the new area is 1 (1/3) times the original area,
    then the new perimeter is 60 meters. -/
theorem rectangle_new_perimeter (width : ℝ) (original_area : ℝ) (new_area : ℝ) :
  width = 10 →
  original_area = 150 →
  new_area = original_area * (4/3) →
  2 * (width + new_area / width) = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_new_perimeter_l3756_375605


namespace NUMINAMATH_CALUDE_expected_correct_answers_l3756_375667

theorem expected_correct_answers 
  (total_problems : ℕ) 
  (katya_probability : ℚ) 
  (pen_probability : ℚ) 
  (katya_problems : ℕ) :
  total_problems = 20 →
  katya_probability = 4/5 →
  pen_probability = 1/2 →
  katya_problems ≥ 10 →
  katya_problems ≤ total_problems →
  (katya_problems : ℚ) * katya_probability + 
  (total_problems - katya_problems : ℚ) * pen_probability ≥ 13 := by
sorry

end NUMINAMATH_CALUDE_expected_correct_answers_l3756_375667


namespace NUMINAMATH_CALUDE_cos_315_degrees_l3756_375622

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l3756_375622


namespace NUMINAMATH_CALUDE_r₂_bound_r₂_bound_tight_l3756_375634

/-- A function f(x) = x² - r₂x + r₃ -/
def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂*x + r₃

/-- Sequence g_n defined recursively -/
def g (r₂ r₃ : ℝ) : ℕ → ℝ
| 0 => 0
| n + 1 => f r₂ r₃ (g r₂ r₃ n)

/-- The statement that needs to be proved -/
theorem r₂_bound (r₂ r₃ : ℝ) :
  (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i+1) ∧ g r₂ r₃ (2*i+1) > g r₂ r₃ (2*i+2)) →
  (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i+1) > g r₂ r₃ i) →
  (∀ M : ℝ, ∃ n : ℕ, g r₂ r₃ n > M) →
  abs r₂ ≥ 2 :=
by sorry

/-- The bound is tight -/
theorem r₂_bound_tight : ∀ ε > 0, ∃ r₂ r₃ : ℝ,
  (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i+1) ∧ g r₂ r₃ (2*i+1) > g r₂ r₃ (2*i+2)) ∧
  (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i+1) > g r₂ r₃ i) ∧
  (∀ M : ℝ, ∃ n : ℕ, g r₂ r₃ n > M) ∧
  abs r₂ < 2 + ε :=
by sorry

end NUMINAMATH_CALUDE_r₂_bound_r₂_bound_tight_l3756_375634


namespace NUMINAMATH_CALUDE_sparrow_seeds_count_l3756_375662

theorem sparrow_seeds_count : ∃ n : ℕ+, 
  (9 * n < 1001) ∧ 
  (10 * n > 1100) ∧ 
  (n = 111) := by
sorry

end NUMINAMATH_CALUDE_sparrow_seeds_count_l3756_375662


namespace NUMINAMATH_CALUDE_log_inequality_l3756_375676

theorem log_inequality : 
  let a := Real.log 2 / Real.log (1/3)
  let b := (1/3)^2
  let c := 2^(1/3)
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_log_inequality_l3756_375676


namespace NUMINAMATH_CALUDE_state_a_selection_percentage_l3756_375631

theorem state_a_selection_percentage :
  ∀ (total_candidates : ℕ) (state_b_percentage : ℚ) (additional_selected : ℕ),
    total_candidates = 8000 →
    state_b_percentage = 7 / 100 →
    additional_selected = 80 →
    ∃ (state_a_percentage : ℚ),
      state_a_percentage * total_candidates + additional_selected = state_b_percentage * total_candidates ∧
      state_a_percentage = 6 / 100 := by
  sorry

end NUMINAMATH_CALUDE_state_a_selection_percentage_l3756_375631


namespace NUMINAMATH_CALUDE_total_amount_correct_l3756_375646

/-- Represents the total amount lent out in rupees -/
def total_amount : ℝ := 11501.6

/-- Represents the amount lent at 8% p.a. in rupees -/
def amount_at_8_percent : ℝ := 15008

/-- Represents the total interest received after one year in rupees -/
def total_interest : ℝ := 850

/-- Theorem stating that the total amount lent out is correct given the conditions -/
theorem total_amount_correct :
  ∃ (amount_at_10_percent : ℝ),
    amount_at_8_percent + amount_at_10_percent = total_amount ∧
    0.08 * amount_at_8_percent + 0.1 * amount_at_10_percent = total_interest :=
by sorry

end NUMINAMATH_CALUDE_total_amount_correct_l3756_375646


namespace NUMINAMATH_CALUDE_x_value_l3756_375630

theorem x_value (x : ℝ) (h_pos : x > 0) (h_percent : x * (x / 100) = 9) (h_multiple : ∃ k : ℤ, x = 3 * k) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3756_375630


namespace NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l3756_375618

theorem ceiling_fraction_evaluation :
  (⌈(19 : ℚ) / 11 - ⌈(35 : ℚ) / 22⌉⌉) / (⌈(35 : ℚ) / 11 + ⌈(11 * 22 : ℚ) / 35⌉⌉) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l3756_375618


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l3756_375635

/-- Definition of the sequence sum -/
def S (n : ℕ) : ℕ := n^2

/-- Definition of the sequence terms -/
def a (n : ℕ) : ℕ := S (n + 1) - S n

/-- Proposition: The sequence {a_n} is arithmetic -/
theorem sequence_is_arithmetic : ∃ (d : ℕ), ∀ (n : ℕ), a (n + 1) = a n + d := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l3756_375635


namespace NUMINAMATH_CALUDE_sum_of_hundred_consecutive_integers_l3756_375620

theorem sum_of_hundred_consecutive_integers : ∃ k : ℕ, 
  50 * (2 * k + 99) = 1627384950 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_hundred_consecutive_integers_l3756_375620


namespace NUMINAMATH_CALUDE_vector_inequality_l3756_375656

theorem vector_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_vector_inequality_l3756_375656


namespace NUMINAMATH_CALUDE_correct_categorization_l3756_375671

def given_numbers : List ℚ := [-13.5, 5, 0, -10, 3.14, 27, -4/5, -15/100, 21/3]

def is_negative (x : ℚ) : Prop := x < 0
def is_non_negative (x : ℚ) : Prop := x ≥ 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ ¬(is_integer x)

def negative_numbers : List ℚ := [-13.5, -10, -4/5, -15/100]
def non_negative_numbers : List ℚ := [5, 0, 3.14, 27, 21/3]
def integers : List ℚ := [5, 0, -10, 27]
def negative_fractions : List ℚ := [-13.5, -4/5, -15/100]

theorem correct_categorization :
  (∀ x ∈ negative_numbers, is_negative x) ∧
  (∀ x ∈ non_negative_numbers, is_non_negative x) ∧
  (∀ x ∈ integers, is_integer x) ∧
  (∀ x ∈ negative_fractions, is_negative_fraction x) ∧
  (∀ x ∈ given_numbers, 
    (x ∈ negative_numbers ∨ x ∈ non_negative_numbers) ∧
    (x ∈ integers ∨ x ∈ negative_fractions ∨ (is_non_negative x ∧ ¬(is_integer x)))) := by
  sorry

end NUMINAMATH_CALUDE_correct_categorization_l3756_375671


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3756_375677

theorem unique_triple_solution : 
  ∃! (p q n : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    p > 0 ∧ q > 0 ∧ n > 0 ∧
    p * (p + 3) + q * (q + 3) = n * (n + 3) ∧
    p = 3 ∧ q = 2 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3756_375677


namespace NUMINAMATH_CALUDE_first_month_sale_l3756_375612

def sales_month_2_to_6 : List ℕ := [6927, 6855, 7230, 6562, 4891]
def average_sale : ℕ := 6500
def number_of_months : ℕ := 6

theorem first_month_sale :
  (average_sale * number_of_months - sales_month_2_to_6.sum) = 6535 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_l3756_375612


namespace NUMINAMATH_CALUDE_domino_less_than_trimino_l3756_375682

/-- A domino tiling of a 2n × 2n grid -/
def DominoTiling (n : ℕ) := Fin (2*n) → Fin (2*n) → Bool

/-- A trimino tiling of a 3n × 3n grid -/
def TriminoTiling (n : ℕ) := Fin (3*n) → Fin (3*n) → Bool

/-- The number of domino tilings of a 2n × 2n grid -/
def numDominoTilings (n : ℕ) : ℕ := sorry

/-- The number of trimino tilings of a 3n × 3n grid -/
def numTriminoTilings (n : ℕ) : ℕ := sorry

/-- Theorem: The number of domino tilings of a 2n × 2n grid is less than
    the number of trimino tilings of a 3n × 3n grid for all positive n -/
theorem domino_less_than_trimino (n : ℕ) (h : n > 0) : 
  numDominoTilings n < numTriminoTilings n := by
  sorry

end NUMINAMATH_CALUDE_domino_less_than_trimino_l3756_375682


namespace NUMINAMATH_CALUDE_power_product_l3756_375606

theorem power_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_l3756_375606


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l3756_375658

theorem complex_magnitude_proof (i a : ℂ) : 
  i ^ 2 = -1 →
  a.im = 0 →
  (∃ k : ℝ, (2 - i) / (a + i) = k * i) →
  Complex.abs ((2 * a + 1) + Real.sqrt 2 * i) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l3756_375658


namespace NUMINAMATH_CALUDE_fundraiser_total_l3756_375672

def fundraiser (sasha_muffins : ℕ) (sasha_price : ℕ)
               (melissa_multiplier : ℕ) (melissa_price : ℕ)
               (tiffany_price : ℕ)
               (sarah_muffins : ℕ) (sarah_price : ℕ)
               (damien_dozens : ℕ) (damien_price : ℕ) : ℕ :=
  let melissa_muffins := melissa_multiplier * sasha_muffins
  let tiffany_muffins := (sasha_muffins + melissa_muffins) / 2
  let damien_muffins := damien_dozens * 12
  (sasha_muffins * sasha_price) +
  (melissa_muffins * melissa_price) +
  (tiffany_muffins * tiffany_price) +
  (sarah_muffins * sarah_price) +
  (damien_muffins * damien_price)

theorem fundraiser_total :
  fundraiser 30 4 4 3 5 50 2 2 6 = 1099 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_total_l3756_375672


namespace NUMINAMATH_CALUDE_cell_division_3_hours_l3756_375645

/-- The number of cells after a given number of 30-minute intervals -/
def num_cells (n : ℕ) : ℕ := 2^n

/-- The number of 30-minute intervals in 3 hours -/
def intervals_in_3_hours : ℕ := 6

theorem cell_division_3_hours : 
  num_cells intervals_in_3_hours = 128 := by
  sorry

end NUMINAMATH_CALUDE_cell_division_3_hours_l3756_375645


namespace NUMINAMATH_CALUDE_compound_proposition_falsehood_l3756_375627

theorem compound_proposition_falsehood (p q : Prop) 
  (hp : p) (hq : q) : 
  (p ∨ q) ∧ (p ∧ q) ∧ ¬(¬p ∧ q) ∧ (¬p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_falsehood_l3756_375627


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_9_l3756_375654

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_factorials (n : ℕ) : ℕ :=
  (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_9 :
  units_digit (sum_factorials 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_9_l3756_375654


namespace NUMINAMATH_CALUDE_inverse_function_point_and_sum_l3756_375659

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverses of each other
axiom inverse_relation : ∀ x, f_inv (f x) = x

-- Given condition: (2,3) is on the graph of y = f(x)/2
axiom point_on_f : f 2 = 6

-- Theorem to prove
theorem inverse_function_point_and_sum :
  (f_inv 6 = 2) ∧ (6, 1) ∈ {p : ℝ × ℝ | p.2 = f_inv p.1 / 2} ∧ (6 + 1 = 7) :=
sorry

end NUMINAMATH_CALUDE_inverse_function_point_and_sum_l3756_375659


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3756_375637

theorem imaginary_part_of_z (z : ℂ) : z = Complex.I * (1 + Complex.I) → Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3756_375637


namespace NUMINAMATH_CALUDE_simplify_expression_l3756_375689

theorem simplify_expression : 18 * (8 / 12) * (1 / 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3756_375689
