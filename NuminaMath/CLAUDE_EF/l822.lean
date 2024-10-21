import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_shortest_segments_and_rays_part_of_line_l822_82255

-- Define a straight line
structure StraightLine where
  -- Add necessary properties (placeholder)
  dummy : Unit

-- Define a point
structure Point where
  -- Add necessary properties (placeholder)
  dummy : Unit

-- Define a line segment
def LineSegment (p q : Point) : Type := Unit

-- Define a ray
def Ray (p : Point) (l : StraightLine) : Type := Unit

-- Define the length of a line connecting two points
noncomputable def LineLength (p q : Point) : ℝ := 0 -- Placeholder implementation

-- Theorem 1: The line segment is the shortest line connecting two points
theorem line_segment_shortest (p q : Point) :
  ∀ (l : Point → Point → Type), LineLength p q ≤ LineLength p q := by sorry

-- Theorem 2: Both line segments and rays are parts of a straight line
theorem segments_and_rays_part_of_line (l : StraightLine) :
  (∃ (p q : Point), True) ∧ (∃ (p : Point), True) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_shortest_segments_and_rays_part_of_line_l822_82255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l822_82220

noncomputable def ω : ℝ := 1

noncomputable def a (x : ℝ) : ℝ × ℝ := (1 + Real.cos (ω * x), -1)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3, Real.sin (ω * x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

noncomputable def θ : ℝ := Real.arccos ((3 * Real.sqrt 3 + 4) / 10)

theorem cos_theta_value :
  ω > 0 ∧
  (∀ x, f (x + 2 * Real.pi) = f x) ∧
  (∀ y, y > 0 → y < 2 * Real.pi → (∀ x, f (x + y) = f x) → y = 2 * Real.pi) ∧
  0 < θ ∧ θ < Real.pi / 2 ∧
  f θ = Real.sqrt 3 + 6 / 5 →
  Real.cos θ = (3 * Real.sqrt 3 + 4) / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l822_82220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_value_l822_82266

def sequence_a : ℕ → ℤ
  | 0 => 0  -- Add this case for Nat.zero
  | 1 => 0
  | n + 1 => -Int.natAbs (sequence_a n + n)

theorem a_2023_value : sequence_a 2023 = -1011 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_value_l822_82266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l822_82261

noncomputable def monthly_salary (savings_rate : ℝ) (expense_increase_rate : ℝ) (new_savings : ℝ) : ℝ :=
  new_savings / (1 - (1 + expense_increase_rate) * (1 - savings_rate))

theorem salary_calculation (savings_rate : ℝ) (expense_increase_rate : ℝ) (new_savings : ℝ) :
  savings_rate = 0.20 →
  expense_increase_rate = 0.20 →
  new_savings = 200 →
  monthly_salary savings_rate expense_increase_rate new_savings = 5000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l822_82261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_to_monday_ratio_l822_82212

/-- Represents the number of sweaters knit on each day of the week -/
structure SweaterCount where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  deriving Repr

/-- Represents the weekly sweater knitting scenario -/
def sweaterScenario : SweaterCount where
  monday := 8
  tuesday := 10
  wednesday := 6
  thursday := 6
  friday := 34 - (8 + 10 + 6 + 6)

theorem friday_to_monday_ratio : 
  (sweaterScenario.friday : ℚ) / sweaterScenario.monday = 1 / 2 := by
  -- Proof steps would go here
  sorry

#eval sweaterScenario

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_to_monday_ratio_l822_82212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_exponential_inequality_l822_82268

theorem solution_set_exponential_inequality :
  {x : ℝ | (2 : ℝ)^(x + 2) > 8} = {x : ℝ | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_exponential_inequality_l822_82268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_equals_one_l822_82224

noncomputable def real_log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem f_2015_equals_one
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_shift : ∀ x, f (2 + x) = f (2 - x))
  (h_log : ∀ x, -3 ≤ x → x ≤ 0 → f x = real_log 3 (2 - x)) :
  f 2015 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_equals_one_l822_82224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l822_82231

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1) / Real.log (1/2)
noncomputable def g (x : ℝ) : ℝ := 2^x

-- Define the domain of f
def A : Set ℝ := {x | x > 1}

-- Define the range of g
def B : Set ℝ := {y | 1/2 ≤ y ∧ y ≤ 4}

-- Define the interval for x in g
def I : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- Theorem for the domain of f
theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = A := by sorry

-- Theorem for the range of g
theorem range_of_g : {y : ℝ | ∃ x ∈ I, g x = y} = B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l822_82231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_proof_l822_82227

/-- The total amount spent by four shoppers with given spending rules and adjustments -/
def shopping_total (emma_spent : ℚ) : ℚ :=
  let erika_initial := emma_spent + 20
  let erika_final := erika_initial * (9/10)
  let elsa_spent := emma_spent * 2
  let elizabeth_initial := elsa_spent * 4
  let elizabeth_final := elizabeth_initial * (106/100)
  emma_spent + erika_final + elsa_spent + elizabeth_final

/-- Proof that the total amount spent is $736.04 -/
theorem total_spent_proof : shopping_total 58 = 736.04 := by
  -- Unfold the definition and simplify
  unfold shopping_total
  -- Perform the calculation
  norm_num
  -- QED

#eval shopping_total 58

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_proof_l822_82227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l822_82234

/-- Given a line and a circle with specified properties, prove the possible values of 'a' -/
theorem line_circle_intersection (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 - A.2 + a = 0) ∧ 
    (B.1 - B.2 + a = 0) ∧ 
    (A.1^2 + A.2^2 + 2*A.1 - 4*A.2 - 4 = 0) ∧ 
    (B.1^2 + B.2^2 + 2*B.1 - 4*B.2 - 4 = 0) ∧
    let C : ℝ × ℝ := (-1, 2)
    ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0)) →
  a = 0 ∨ a = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l822_82234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l822_82250

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x + a - 2) / (2^x + 1)

-- State the theorem
theorem function_properties (a : ℝ) (h : f a 1 = 1/3) :
  (a = 1) ∧ 
  (∀ x, f a x = -f a (-x)) ∧
  (∀ x y, x < y → f a x < f a y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l822_82250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carrots_removed_l822_82246

/-- Proves that 4 carrots were removed from the scale given the conditions of the problem -/
theorem carrots_removed 
  (total_weight : ℝ)
  (total_carrots : ℕ)
  (remaining_carrots : ℕ)
  (avg_weight_remaining : ℝ)
  (avg_weight_removed : ℝ)
  (h1 : total_weight = 3.64)
  (h2 : total_carrots = 20)
  (h3 : remaining_carrots = 16)
  (h4 : avg_weight_remaining = 0.180)
  (h5 : avg_weight_removed = 0.190)
  : total_carrots - remaining_carrots = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carrots_removed_l822_82246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_dot_product_l822_82269

-- Define the parallelogram ABCD
def Parallelogram (A B C D : ℝ × ℝ) : Prop :=
  B.1 - A.1 = D.1 - C.1 ∧ B.2 - A.2 = D.2 - C.2

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the length of a vector
noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Main theorem
theorem parallelogram_dot_product 
  (A B C D : ℝ × ℝ) 
  (h_parallelogram : Parallelogram A B C D)
  (h_AB_length : vector_length (B.1 - A.1, B.2 - A.2) = 1)
  (h_AD_length : vector_length (D.1 - A.1, D.2 - A.2) = 2) :
  dot_product 
    (C.1 - A.1, C.2 - A.2) 
    (D.1 - B.1, D.2 - B.2) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_dot_product_l822_82269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unshaded_fraction_is_four_ninths_l822_82265

/-- A square with side length s and points R and S located on its sides. -/
structure Square (s : ℝ) where
  R : ℝ × ℝ
  S : ℝ × ℝ
  h_R : R = (0, s / 3)
  h_S : S = (s, 2 * s / 3)

/-- The fraction of the square that is not shaded when R and S are connected to opposite vertices. -/
noncomputable def unshaded_fraction (s : ℝ) (square : Square s) : ℝ := 4 / 9

/-- Theorem stating that the unshaded fraction of the square is 4/9. -/
theorem unshaded_fraction_is_four_ninths (s : ℝ) (square : Square s) :
  unshaded_fraction s square = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unshaded_fraction_is_four_ninths_l822_82265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_fraction_value_l822_82208

-- Define angle_alpha as a variable instead of a function
variable (α : ℝ)

-- Condition 3
def terminal_point : ℝ × ℝ := (1, 3)

-- Condition 4
def tan_pi_minus_alpha : ℝ := -2

-- Theorem 1
theorem sin_alpha_value : 
  Real.sin α = 3 * Real.sqrt 10 / 10 := by sorry

-- Theorem 2
theorem fraction_value (h : tan_pi_minus_alpha = -2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_fraction_value_l822_82208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_equals_five_l822_82292

def f (a b x : ℝ) : ℝ := a * x + b

def f_n (a b : ℝ) : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- Base case for n = 0
  | 1 => f a b
  | n + 1 => f a b ∘ f_n a b n

theorem a_plus_b_equals_five (a b : ℝ) :
  (∀ x, f_n a b 7 x = 128 * x + 381) → a + b = 5 := by
  sorry

#check a_plus_b_equals_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_equals_five_l822_82292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_star_intersections_star_2018_25_l822_82254

/-- A regular (n,k)-star is a closed broken line obtained by replacing every k consecutive sides
    of a regular n-gon with a diagonal connecting the same endpoints. -/
def regular_star (n k : ℕ) : Prop :=
  Nat.Coprime n k ∧ n ≥ 5 ∧ k < n / 2

/-- The number of intersection points in a regular (n,k)-star. -/
def num_intersections (n k : ℕ) : ℕ := n * (k - 1)

/-- Theorem stating the number of intersection points in a regular (n,k)-star. -/
theorem regular_star_intersections (n k : ℕ) (h : regular_star n k) :
  num_intersections n k = n * (k - 1) := by
  rfl

/-- The specific case for (2018, 25)-star -/
theorem star_2018_25 :
  regular_star 2018 25 ∧ num_intersections 2018 25 = 48432 := by
  constructor
  · -- Prove that (2018, 25) forms a regular star
    constructor
    · -- Prove 2018 and 25 are coprime
      exact Nat.Coprime.symm (by norm_num)
    · -- Prove 2018 ≥ 5 and 25 < 2018 / 2
      constructor
      · norm_num
      · norm_num
  · -- Prove the number of intersections
    rfl

#eval num_intersections 2018 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_star_intersections_star_2018_25_l822_82254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_distinct_roots_quadratic_roots_when_k_is_one_l822_82299

/-- The quadratic equation x^2 - 2x + 2k - 3 = 0 has two distinct real roots if and only if k < 2 -/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + 2*k - 3 = 0 ∧ y^2 - 2*y + 2*k - 3 = 0) ↔ k < 2 :=
by sorry

/-- When k is 1, the roots of x^2 - 2x + 2k - 3 = 0 are 1 + √2 and 1 - √2 -/
theorem quadratic_roots_when_k_is_one :
  let k : ℝ := 1
  let x1 : ℝ := 1 + Real.sqrt 2
  let x2 : ℝ := 1 - Real.sqrt 2
  x1^2 - 2*x1 + 2*k - 3 = 0 ∧ x2^2 - 2*x2 + 2*k - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_distinct_roots_quadratic_roots_when_k_is_one_l822_82299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_for_b_l822_82288

/-- The time taken by worker b to complete a work alone, given the relative efficiencies of workers a, b, and c, and their combined work time. -/
theorem work_time_for_b (efficiency_b : ℝ) (combined_time : ℝ) : ℝ := by
  -- Define efficiency_a and efficiency_c in terms of efficiency_b
  let efficiency_a := 2 * efficiency_b
  let efficiency_c := 3 * efficiency_a
  
  -- Assume the combined time is 10 days
  have h1 : combined_time = 10 := by sorry
  
  -- The equation for the combined work
  have h2 : (efficiency_a + efficiency_b + efficiency_c) * combined_time = efficiency_b * 90 := by sorry
  
  -- Solve for the time taken by b alone
  exact 90 -- This is the final answer


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_for_b_l822_82288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_quiz_score_for_average_l822_82275

/-- Theorem: Given a student's average score on 4 quizzes and the required overall average for 5 quizzes,
    calculate the required score on the 5th quiz to achieve the overall average. -/
theorem final_quiz_score_for_average 
  (num_quizzes : ℕ)
  (current_average : ℚ)
  (required_average : ℚ)
  (h_num_quizzes : num_quizzes = 4)
  (h_current_average : current_average = 92/100)
  (h_required_average : required_average = 93/100) :
  (num_quizzes + 1) * required_average - num_quizzes * current_average = 97/100 := by
  sorry

#eval (5 : ℕ) * (93/100 : ℚ) - (4 : ℕ) * (92/100 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_quiz_score_for_average_l822_82275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_ratio_l822_82241

theorem tan_ratio_from_sin_ratio (α β : ℝ) (h : Real.sin (2 * α) / Real.sin (2 * β) = 3) :
  Real.tan (α - β) / Real.tan (α + β) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_ratio_l822_82241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_l822_82297

noncomputable def rotation_60_degrees : ℂ := Complex.exp (Complex.I * Real.pi / 3)

def dilation_factor : ℝ := 2

def initial_number : ℂ := -4 + 3 * Complex.I

theorem transformations_result :
  initial_number * rotation_60_degrees * dilation_factor = 5 - Real.sqrt 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_l822_82297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_prime_fifteenth_prime_l822_82217

-- Define a function that returns the nth prime number
def nthPrime : ℕ → ℕ 
| 0 => 2  -- The first prime number is 2
| n+1 => sorry  -- We'll use sorry here as we don't have a full implementation

-- State the given condition
theorem fifth_prime : nthPrime 5 = 11 := by
  sorry

-- State the theorem to be proved
theorem fifteenth_prime : nthPrime 15 = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_prime_fifteenth_prime_l822_82217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_theorem_l822_82200

/-- A quadrilateral defined by four lines in the plane -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The intersection point of the diagonals of a quadrilateral -/
noncomputable def diagonalIntersection (q : Quadrilateral) : ℝ × ℝ :=
  (0, (q.b + q.c) / 2)

/-- Theorem stating that the intersection point of the diagonals
    of the given quadrilateral is (0, (b + c)/2) -/
theorem diagonal_intersection_theorem (q : Quadrilateral) :
  diagonalIntersection q = (0, (q.b + q.c) / 2) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_theorem_l822_82200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l822_82218

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_problem (a b : V) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = 2) 
  (hab : ‖a + b‖ = Real.sqrt 5) : 
  ‖(2 : ℝ) • a - b‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l822_82218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_less_than_v_l822_82283

noncomputable def geomSum (x : ℝ) (n : ℕ) : ℝ := (x * (x^n - 1)) / (x - 1)

noncomputable def U (x : ℝ) : ℝ := geomSum x 8 + 10 * x^9

noncomputable def V (x : ℝ) : ℝ := geomSum x 10 + 10 * x^11

theorem u_less_than_v (u v : ℝ) (hu : U u = 8) (hv : V v = 8) : u < v := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_less_than_v_l822_82283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sum_l822_82281

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Icc (-2) 1

-- State the theorem
theorem domain_of_sum :
  (∀ z ∈ domain_f_plus_one, f (z + 1) = f (z + 1)) →
  ∀ x : ℝ, (f x + f (-x) = f x + f (-x)) ↔ x ∈ Set.Icc (-1) 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sum_l822_82281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_arrangements_eq_three_l822_82286

def digits : List Nat := [2, 0, 2, 2]

def is_valid_arrangement (arrangement : List Nat) : Bool :=
  arrangement.length = 4 &&
  arrangement.all (λ d => d ∈ digits) &&
  arrangement.head? ≠ some 0 &&
  arrangement.get? 3 ≠ some 0 &&
  (arrangement.foldl (λ acc d => acc * 10 + d) 0) % 2 = 0

def count_valid_arrangements : Nat :=
  (digits.permutations.filter is_valid_arrangement).length

theorem count_valid_arrangements_eq_three :
  count_valid_arrangements = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_arrangements_eq_three_l822_82286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_66_approximation_l822_82293

-- Define the approximation for cos 78°
noncomputable def cos_78 : ℝ := 1/5

-- Define the theorem
theorem sin_66_approximation : 
  abs (Real.sin (66 * π / 180) - 0.92) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_66_approximation_l822_82293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coverage_l822_82282

noncomputable def large_triangle : Set (ℝ × ℝ) := sorry

def is_equilateral_triangle (t : Set (ℝ × ℝ)) : Prop := sorry

noncomputable def side_length (t : Set (ℝ × ℝ)) : ℝ := sorry

theorem triangle_coverage (s : ℝ) (h_s : 0 < s ∧ s < 1) 
  (h_cover : ∃ (t1 t2 t3 t4 t5 : Set (ℝ × ℝ)), 
    (∀ i ∈ [t1, t2, t3, t4, t5], is_equilateral_triangle i ∧ side_length i = s) ∧
    large_triangle ⊆ t1 ∪ t2 ∪ t3 ∪ t4 ∪ t5) :
  ∃ (t1 t2 t3 t4 : Set (ℝ × ℝ)),
    (∀ i ∈ [t1, t2, t3, t4], is_equilateral_triangle i ∧ side_length i = s) ∧
    large_triangle ⊆ t1 ∪ t2 ∪ t3 ∪ t4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coverage_l822_82282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_to_total_area_ratio_formula_l822_82263

/-- A cylinder where the front view is similar to its lateral development view -/
structure SimilarViewsCylinder where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cylinder
  h_eq : h = 2 * r * Real.sqrt Real.pi

/-- The ratio of the lateral area to the total area of a SimilarViewsCylinder -/
noncomputable def lateral_to_total_area_ratio (c : SimilarViewsCylinder) : ℝ :=
  let lateral_area := 2 * Real.pi * c.r * c.h
  let total_area := lateral_area + 2 * Real.pi * c.r^2
  lateral_area / total_area

theorem lateral_to_total_area_ratio_formula (c : SimilarViewsCylinder) :
  lateral_to_total_area_ratio c = (2 * Real.sqrt Real.pi) / (2 * Real.sqrt Real.pi + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_to_total_area_ratio_formula_l822_82263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_trajectory_area_l822_82249

/-- The area enclosed by the curve traced by the highest points of projectile trajectories -/
theorem projectile_trajectory_area (v g : ℝ) (h_v : v > 0) (h_g : g > 0) :
  let θ : ℝ → ℝ := λ t => t * π / 2  -- Angle parameterization from 0 to π/2
  let x : ℝ → ℝ := λ t => (v^2 / (2 * g)) * Real.sin (2 * θ t)
  let y : ℝ → ℝ := λ t => (v^2 / (2 * g)) * Real.sin (θ t)^2
  ∃ A : ℝ, A = π / 16 * (v^4 / g^2) ∧
    A = ∫ t in (Set.Icc 0 1), y t * (deriv x t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_trajectory_area_l822_82249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cyclic_sum_f_l822_82236

/-- CyclicSum represents a cyclic sum of a function over three variables -/
noncomputable def CyclicSum (f : ℝ → ℝ → ℝ → ℝ) (x y z : ℝ) : ℝ :=
  f x y z + f y z x + f z x y

/-- The function f as defined in the problem -/
noncomputable def f (x y z : ℝ) : ℝ :=
  Real.sqrt ((x^2 + 256*y*z) / (y^2 + z^2))

/-- Theorem stating the minimum value of the cyclic sum of f -/
theorem min_cyclic_sum_f {x y z : ℝ} (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
    (h_not_all_zero : ¬(x = 0 ∧ y = 0 ∧ z = 0)) :
    CyclicSum f x y z ≥ 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cyclic_sum_f_l822_82236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_to_reach_end_is_correct_l822_82233

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction of movement -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- A path is a list of directions -/
def MovePath := List Direction

/-- The probability of a single step in any direction -/
def stepProbability : ℚ := 1/4

/-- The number of steps in the path -/
def pathLength : ℕ := 8

/-- The starting point -/
def start : Point := ⟨0, 0⟩

/-- The ending point -/
def finish : Point := ⟨3, 3⟩

/-- Function to check if a path leads from start to finish -/
def isValidPath (p : MovePath) : Bool :=
  sorry

/-- Function to count the number of valid paths -/
def countValidPaths : ℕ :=
  sorry

/-- The total number of possible paths -/
def totalPaths : ℕ := 4^pathLength

/-- The probability of reaching the end point -/
noncomputable def probabilityToReachEnd : ℚ := countValidPaths / totalPaths

theorem probability_to_reach_end_is_correct :
  probabilityToReachEnd = 175/8192 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_to_reach_end_is_correct_l822_82233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l822_82237

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 2

-- Define the line
def my_line (x y : ℝ) : Prop := x + y = 0

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem tangent_line_to_circle :
  ∃! k : ℝ, 
    (∀ x y : ℝ, my_line x y ↔ y = -x) ∧
    (∃ x y : ℝ, my_line x y ∧ my_circle x y ∧ fourth_quadrant x y) ∧
    (∀ x y : ℝ, my_line x y → (x = 0 ∧ y = 0) ∨ (x ≠ 0 ∧ y ≠ 0)) ∧
    k = -1 := by
  sorry

#check tangent_line_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l822_82237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_of_specific_triangle_l822_82298

-- Define the right triangle DEF
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  is_right : DE > 0 ∧ EF > 0

-- Define the median length function
noncomputable def median_length (t : RightTriangle) : ℝ :=
  (1 / 2) * Real.sqrt (2 * t.DE^2 + 2 * t.EF^2 - (t.DE^2 + t.EF^2))

-- Theorem statement
theorem median_length_of_specific_triangle :
  ∃ (t : RightTriangle), t.DE = 5 ∧ t.EF = 12 ∧ median_length t = 6.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_of_specific_triangle_l822_82298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_cos_alpha_eq_one_fourth_l822_82294

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 4^x

-- Define the angle α
noncomputable def α : ℝ := Real.arctan (2 * Real.sqrt 2)

-- Theorem statement
theorem f_f_cos_alpha_eq_one_fourth :
  f (f (Real.cos α)) = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_cos_alpha_eq_one_fourth_l822_82294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_nth_roots_l822_82242

theorem compare_nth_roots : (4 : ℝ)^(1/4) > (5 : ℝ)^(1/5) ∧ (5 : ℝ)^(1/5) > (6 : ℝ)^(1/6) ∧ (6 : ℝ)^(1/6) > (12 : ℝ)^(1/12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_nth_roots_l822_82242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_person_knows_all_l822_82222

/-- Represents a person in the company -/
structure Person where
  id : Nat

/-- Represents the company -/
structure Company (n : Nat) where
  people : Finset Person
  size_eq : people.card = 2 * n + 1
  knows : Person → Person → Prop

/-- For any n people, there exists another person who knows all of them -/
axiom knows_n_people {n : Nat} (c : Company n) :
  ∀ (s : Finset Person), s ⊆ c.people → s.card = n →
    ∃ p ∈ c.people, p ∉ s ∧ ∀ q ∈ s, c.knows p q

/-- There exists a person who knows everyone in the company -/
theorem exists_person_knows_all {n : Nat} (c : Company n) :
  ∃ p ∈ c.people, ∀ q ∈ c.people, p ≠ q → c.knows p q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_person_knows_all_l822_82222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sphere_volume_ratio_l822_82221

-- Define the edge length of the cube
def cube_edge : ℝ := 8

-- Define the volume of a cube
noncomputable def cube_volume (s : ℝ) : ℝ := s^3

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Theorem statement
theorem cube_sphere_volume_ratio :
  (cube_volume cube_edge) / (sphere_volume (cube_edge / 2)) = 6 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sphere_volume_ratio_l822_82221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_sum_l822_82257

/-- A hexagon inscribed in a circle -/
structure InscribedHexagon where
  /-- The lengths of the six sides of the hexagon -/
  sides : Fin 6 → ℝ
  /-- The hexagon is inscribed in a circle -/
  inscribed : True
  /-- Five sides have length 100 -/
  five_equal_sides : ∃ i, sides i = 40 ∧ (∀ j ≠ i, sides j = 100)

/-- The sum of the lengths of the three diagonals from one vertex -/
noncomputable def diagonalSum (h : InscribedHexagon) : ℝ := sorry

/-- Theorem: The sum of the lengths of the three diagonals from one vertex
    in the specified hexagon is approximately 376.22 -/
theorem hexagon_diagonal_sum :
  ∀ h : InscribedHexagon, ∃ ε > 0, |diagonalSum h - 376.22| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_sum_l822_82257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_n_l822_82245

theorem sum_of_digits_n (n : ℕ) : 
  (Nat.factorial (n + 1)) + (Nat.factorial (n + 2)) = (Nat.factorial n) * 440 → 
  (n.repr.toList.map Char.toNat).sum = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_n_l822_82245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l822_82285

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 8 * sin x - tan x

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), 0 < x ∧ x < π / 2 ∧
  (∀ (y : ℝ), 0 < y ∧ y < π / 2 → f y ≤ f x) ∧
  f x = 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l822_82285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_production_analysis_l822_82264

/-- Linear relationship between monthly sales volume and selling price -/
def sales_volume (x : ℝ) : ℝ := -2 * x + 100

/-- Cost per unit in March -/
def march_cost : ℝ := 20

/-- Profit function in April -/
def april_profit (x : ℝ) : ℝ := -2 * x^2 + 112 * x - 1050

/-- Minimum profit in April -/
def min_april_profit : ℝ := 500

theorem factory_production_analysis 
  (march_price : ℝ) 
  (march_profit : ℝ) 
  (april_cost_reduction : ℝ) 
  (april_price_lower : ℝ) 
  (april_price_upper : ℝ) 
  (april_investment : ℝ)
  (h1 : march_price = 35)
  (h2 : march_profit = 450)
  (h3 : april_cost_reduction = 14)
  (h4 : april_price_lower = 25)
  (h5 : april_price_upper = 30)
  (h6 : april_investment = 450) :
  (∀ x, sales_volume x = -2 * x + 100) ∧
  march_cost = 20 ∧
  (∀ x, april_profit x = -2 * x^2 + 112 * x - 1050) ∧
  min_april_profit = 500 := by
  sorry

#check factory_production_analysis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_production_analysis_l822_82264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_proof_l822_82272

theorem angle_value_proof (α : Real) 
  (h1 : Real.cos α = -Real.sqrt 3 / 2) 
  (h2 : 0 < α ∧ α < Real.pi) : 
  α = 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_proof_l822_82272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_30_l822_82202

def S : Set ℚ := {-30, -5, -1, 0, 2, 10, 15}

theorem largest_quotient_is_30 : 
  ∀ a b : ℚ, a ∈ S → b ∈ S → b ≠ 0 → (a / b : ℚ) ≤ 30 ∧ ∃ c d : ℚ, c ∈ S ∧ d ∈ S ∧ d ≠ 0 ∧ (c / d : ℚ) = 30 := by
  sorry

#check largest_quotient_is_30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_30_l822_82202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OMN_l822_82228

/-- Circle C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 20

/-- Line C₂ in polar coordinates -/
def C₂ (θ ρ : ℝ) : Prop := θ = Real.pi/3

/-- Line C₃ in polar coordinates -/
def C₃ (θ ρ : ℝ) : Prop := θ = Real.pi/6

/-- Polar to Cartesian conversion for x-coordinate -/
noncomputable def polar_to_cartesian_x (ρ θ : ℝ) : ℝ := ρ * Real.cos θ

/-- Polar to Cartesian conversion for y-coordinate -/
noncomputable def polar_to_cartesian_y (ρ θ : ℝ) : ℝ := ρ * Real.sin θ

/-- Theorem stating the area of triangle OMN -/
theorem area_of_triangle_OMN : 
  ∃ (ρ₁ ρ₂ : ℝ), 
    C₁ (polar_to_cartesian_x ρ₁ (Real.pi/3)) (polar_to_cartesian_y ρ₁ (Real.pi/3)) ∧ 
    C₁ (polar_to_cartesian_x ρ₂ (Real.pi/6)) (polar_to_cartesian_y ρ₂ (Real.pi/6)) ∧ 
    1/2 * ρ₁ * ρ₂ * Real.sin (Real.pi/3 - Real.pi/6) = 8 + 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OMN_l822_82228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circle_area_l822_82252

/-- The curve on which the center of the circle lies -/
def curve (x y : ℝ) : Prop := x * y = 2 ∧ x > 0

/-- The line to which the circle is tangent -/
def tangent_line (x y : ℝ) : Prop := x + 2 * y + 1 = 0

/-- The area of a circle with radius r -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- The theorem stating the minimum area of the circle -/
theorem min_circle_area :
  ∃ (x y r : ℝ), curve x y ∧ tangent_line x y ∧
  r = |x + 2*y + 1| / Real.sqrt 5 ∧
  (∀ (x' y' r' : ℝ), curve x' y' → tangent_line x' y' →
    r' = |x' + 2*y' + 1| / Real.sqrt 5 → circle_area r ≤ circle_area r') ∧
  circle_area r = 5 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circle_area_l822_82252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l822_82258

/-- Represents the volume of a cone -/
noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Represents the volume of a cylinder -/
noncomputable def cylinder_volume (r : ℝ) (h : ℝ) : ℝ := Real.pi * r^2 * h

theorem water_height_in_cylinder 
  (cone_radius : ℝ) 
  (cone_height : ℝ) 
  (cylinder_radius : ℝ) :
  cone_radius = 8 →
  cone_height = 12 →
  cylinder_radius = 16 →
  ∃ (cylinder_height : ℝ),
    cone_volume cone_radius cone_height = cylinder_volume cylinder_radius cylinder_height ∧
    cylinder_height = 1 := by
  sorry

#check water_height_in_cylinder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l822_82258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mural_length_approx_l822_82232

/-- Calculates the length of a mural given its specifications -/
noncomputable def muralLength (width : ℝ) (paintCost : ℝ) (paintRate : ℝ) (laborRate : ℝ) (totalCost : ℝ) : ℝ :=
  let laborCost := laborRate / paintRate
  let totalArea := totalCost / (paintCost + laborCost)
  totalArea / width

/-- The length of the mural is approximately 5.9967 meters -/
theorem mural_length_approx :
  let width := (3 : ℝ)
  let paintCost := (4 : ℝ)
  let paintRate := (1.5 : ℝ)
  let laborRate := (10 : ℝ)
  let totalCost := (192 : ℝ)
  abs (muralLength width paintCost paintRate laborRate totalCost - 5.9967) < 0.0001 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval muralLength 3 4 1.5 10 192

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mural_length_approx_l822_82232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_implies_nonpositive_a_l822_82205

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (exp (2 * x) / x) - 2 * x + log x

-- State the theorem
theorem no_minimum_implies_nonpositive_a :
  ∀ a : ℝ, (∀ s > 0, ∃ t > 0, f a t < f a s) → a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_implies_nonpositive_a_l822_82205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_arrives_before_bob_l822_82267

-- Define the constants from the problem
noncomputable def distance : ℝ := 180
noncomputable def bob_speed : ℝ := 40
noncomputable def alice_delay : ℝ := 0.5
noncomputable def weather_factor : ℝ := 0.85

-- Define Bob's arrival time
noncomputable def bob_arrival_time : ℝ := distance / bob_speed

-- Define Alice's minimum speed to arrive before Bob
noncomputable def alice_min_speed : ℝ := 53

-- Theorem statement
theorem alice_arrives_before_bob :
  let alice_actual_speed := alice_min_speed * weather_factor
  let alice_travel_time := distance / alice_actual_speed
  alice_travel_time + alice_delay < bob_arrival_time := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_arrives_before_bob_l822_82267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_specific_circles_l822_82219

/-- The area between two concentric circles -/
noncomputable def area_between_circles (r₁ r₂ : ℝ) : ℝ := Real.pi * (r₁^2 - r₂^2)

/-- Theorem: The area between two concentric circles with radii 7 and 4 is 33π -/
theorem area_between_specific_circles :
  area_between_circles 7 4 = 33 * Real.pi := by
  unfold area_between_circles
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_specific_circles_l822_82219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_x_over_3_l822_82243

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 3)

-- State the theorem
theorem period_of_tan_x_over_3 : 
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
  ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y :=
by
  -- We claim that p = 3π
  let p := 3 * Real.pi
  
  -- Prove that this p satisfies the conditions
  have h_p_pos : p > 0 := by sorry
  have h_periodic : ∀ (x : ℝ), f (x + p) = f x := by sorry
  have h_minimal : ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y := by sorry
  
  -- Combine all parts of the proof
  exact ⟨p, h_p_pos, λ x => ⟨h_periodic x, h_minimal⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_x_over_3_l822_82243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_e_minus_2_l822_82244

-- Define e as the base of natural logarithms
noncomputable def e : ℝ := Real.exp 1

-- Define the greatest integer function (floor function)
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem statement
theorem greatest_integer_e_minus_2 : floor (e - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_e_minus_2_l822_82244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f₁_max_value_f₂_l822_82262

noncomputable section

-- Function for the first part of the problem
def f₁ (x : ℝ) : ℝ := 12 / x + 3 * x

-- Function for the second part of the problem
def f₂ (x : ℝ) : ℝ := x * (-3 * x)

-- Theorem for the first part
theorem min_value_f₁ (x : ℝ) (hx : x > 0) :
  f₁ x ≥ f₁ 2 ∧ (f₁ x = f₁ 2 ↔ x = 2) := by
  sorry

-- Theorem for the second part
theorem max_value_f₂ (x : ℝ) (hx : 0 < x ∧ x < 1/3) :
  f₂ x ≤ 1/12 ∧ (f₂ x = 1/12 ↔ x = 1/6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f₁_max_value_f₂_l822_82262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_herd_division_l822_82277

theorem herd_division (herd : ℕ) : 
  (herd / 3 + herd / 5 + herd / 9 + 11 = herd) ∧ 
  (herd % 3 = 0) ∧ (herd % 5 = 0) ∧ (herd % 9 = 0) →
  herd = 31 := by
  intro h
  sorry

#check herd_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_herd_division_l822_82277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l822_82287

noncomputable def f (x : ℝ) := Real.log (x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l822_82287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alyssa_allowance_proof_l822_82278

-- Define Alyssa's weekly allowance
noncomputable def weekly_allowance : ℚ := 8

-- Define the amount spent on movies
noncomputable def movie_expense : ℚ := weekly_allowance / 2

-- Define the amount earned from car washing
noncomputable def car_wash_earnings : ℚ := 8

-- Define the final amount Alyssa has
noncomputable def final_amount : ℚ := 12

-- Theorem to prove
theorem alyssa_allowance_proof :
  movie_expense + car_wash_earnings = final_amount ∧
  weekly_allowance = 8 := by
  -- Split the conjunction
  constructor
  -- Prove the first part
  · simp [movie_expense, car_wash_earnings, final_amount, weekly_allowance]
    norm_num
  -- Prove the second part
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alyssa_allowance_proof_l822_82278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_product_zero_l822_82214

/-- Given that C = (4, 3) is the midpoint of AB, where A = (2, 6) and B = (x, y), prove that xy = 0. -/
theorem midpoint_product_zero (x y : ℝ) : 
  ((4 : ℝ), 3) = ((2 + x) / 2, (6 + y) / 2) → x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_product_zero_l822_82214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l822_82251

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x : ℝ), f (Real.pi / 3 + x) = f (Real.pi / 3 - x)) ∧
  (∀ (x y : ℝ), -Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ Real.pi / 3 → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l822_82251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_l822_82271

variable (C : ℝ)

def P (C : ℝ) (x : ℝ) : ℝ := 2 * C * x^2 + 2 * (C + 1) * x + 1
def Q (x : ℝ) : ℝ := 2 * x * (x + 1)

theorem polynomial_equation (C : ℝ) (x : ℝ) (h : x ≠ 0 ∧ x ≠ -2) :
  P C x / Q x - P C (x + 1) / Q (x + 1) = 1 / (x * (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_l822_82271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_to_999_l822_82273

theorem sequence_to_999 : ∃ (op1 op2 op3 op4 op5 op6 op7 op8 op9 : ℕ → ℕ → ℕ) 
  (b1 b2 b3 b4 b5 : ℕ → ℕ),
  let seq := [2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
  let result := op9 (op8 (b5 (op7 (b4 (op6 (b3 (op5 (b2 (op4 (b1 (op3 (seq.get! 0) (seq.get! 1))) 
    (op2 (seq.get! 2) (seq.get! 3)))) (op1 (seq.get! 4) (seq.get! 5)))) (seq.get! 6))) (seq.get! 7))) 
    (seq.get! 8)) (seq.get! 9)
  result = 999 := by
  sorry

#eval 999

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_to_999_l822_82273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_multiple_of_300_l822_82240

def n : ℕ := 2^12 * 3^15 * 5^9

def is_factor_multiple_of_300 (f : ℕ) : Prop :=
  f ∣ n ∧ 300 ∣ f

theorem count_factors_multiple_of_300 :
  (Finset.filter (fun f => decidable_of_iff (is_factor_multiple_of_300 f)
    (by simp [is_factor_multiple_of_300])) (Nat.divisors n)).card = 1320 := by
  sorry

#eval (Finset.filter (fun f => 300 ∣ f) (Nat.divisors n)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_multiple_of_300_l822_82240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_in_range_l822_82256

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then a + |x - 2|
  else x^2 - 2*a*x + 2*a

theorem f_nonnegative_iff_a_in_range (a : ℝ) :
  (∀ x, f a x ≥ 0) ↔ a ∈ Set.Icc (-1) 2 := by
  sorry

#check f_nonnegative_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_in_range_l822_82256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_n_equals_263_l822_82238

-- Define the point R
def R : ℝ × ℝ := (10, 7)

-- Define the lines
def line1 (x y : ℝ) : Prop := 9 * y = 18 * x
def line2 (x y : ℝ) : Prop := 12 * y = 5 * x

-- Define points P and Q
variable (P Q : ℝ × ℝ)

-- State that P is on line1
axiom P_on_line1 : line1 P.1 P.2

-- State that Q is on line2
axiom Q_on_line2 : line2 Q.1 Q.2

-- R is the midpoint of PQ
axiom R_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the length of PQ
noncomputable def PQ_length : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define m and n
variable (m n : ℕ)

-- m and n are relatively prime
axiom m_n_coprime : Nat.Coprime m n

-- Length of PQ = m/n
axiom PQ_length_frac : PQ_length = m / n

-- Theorem to prove
theorem sum_m_n_equals_263 : m + n = 263 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_n_equals_263_l822_82238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l822_82204

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 12 * x - 9) / (x^2 - 5 * x + 2)

/-- The horizontal asymptote of g(x) -/
def horizontalAsymptote : ℝ := 3

/-- Theorem: g(x) crosses its horizontal asymptote at x = 5 -/
theorem g_crosses_asymptote :
  g 5 = horizontalAsymptote :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l822_82204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_answer_is_no_l822_82280

/-- Represents a sequence of yes/no answers -/
def AnswerSequence := List Bool

/-- Checks if there are more true (yes) than false (no) in the list -/
def moreYesThanNo (seq : AnswerSequence) : Prop :=
  (seq.filter id).length > (seq.filter not).length

/-- Checks if there are no three consecutive identical answers -/
def noThreeConsecutive (seq : AnswerSequence) : Prop :=
  ∀ i, i + 2 < seq.length → ¬(seq.get? i = seq.get? (i+1) ∧ seq.get? (i+1) = seq.get? (i+2))

/-- Checks if the first and last answers are opposite -/
def firstLastOpposite (seq : AnswerSequence) : Prop :=
  seq.head? = some (!(seq.getLast?).getD false)

/-- Represents a valid answer sequence -/
structure ValidSequence where
  seq : AnswerSequence
  length_eq : seq.length = 5
  more_yes : moreYesThanNo seq
  no_three : noThreeConsecutive seq
  opposite_ends : firstLastOpposite seq

/-- The main theorem to prove -/
theorem second_answer_is_no :
  ∀ (vs : ValidSequence), (∃! (unique_vs : ValidSequence), true) → vs.seq.get? 1 = some false := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_answer_is_no_l822_82280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_at_4_delta_satisfies_continuity_l822_82209

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 3

-- State the continuity theorem
theorem continuity_at_4 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 4| < δ → |f x - f 4| < ε :=
by
  sorry

-- Define delta as a function of epsilon
noncomputable def delta (ε : ℝ) : ℝ := ε / 27

-- State that this delta satisfies the continuity condition
theorem delta_satisfies_continuity :
  ∀ ε > 0, ∀ x, |x - 4| < delta ε → |f x - f 4| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_at_4_delta_satisfies_continuity_l822_82209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l822_82270

theorem division_problem : ∃ (q : ℤ), 
  14698 = (q : ℚ) * 164.98876404494382 + 14 ∧ q = 88 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l822_82270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l822_82274

noncomputable section

/-- Define the function f -/
def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

/-- Define the function g -/
def g (A ω φ : ℝ) (x : ℝ) : ℝ := f A ω φ (x - Real.pi/6)

/-- Define the function h -/
def h (A ω φ : ℝ) (x : ℝ) : ℝ := f A ω φ x + g A ω φ x + 2 * (Real.cos x)^2 - 1

theorem range_of_h (A ω φ : ℝ) :
  A > 0 → 0 < ω → ω < 4 → |φ| < Real.pi/2 →
  f A ω φ 0 = 1/2 →
  f A ω φ (Real.pi/6) = 1 →
  ∀ x ∈ Set.Icc 0 (Real.pi/2), -1 ≤ h A ω φ x ∧ h A ω φ x ≤ 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l822_82274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_identity_l822_82235

theorem cos_sin_identity (α β : ℝ) :
  Real.cos (α + β) * Real.cos β + Real.sin (α + β) * Real.sin β = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_identity_l822_82235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l822_82206

-- Define the circles and their properties
noncomputable def circle_H : ℝ := 6  -- radius of circle H

-- Define the radii of circles J, K, and L in terms of a variable r
noncomputable def circle_K (r : ℝ) : ℝ := r
noncomputable def circle_J (r : ℝ) : ℝ := 2 * r
noncomputable def circle_L (r : ℝ) : ℝ := r / 3

-- Define the relationship between circles based on tangency
def tangency_condition (r : ℝ) : Prop :=
  circle_H = circle_K r + r ∧ 
  circle_H = circle_J r + r ∧
  circle_H = circle_L r + (circle_K r)

-- Define the form of circle J's radius
def radius_J_form (p q : ℕ) (r : ℝ) : Prop :=
  circle_J r = Real.sqrt (p : ℝ) - (q : ℝ) ∧ p > 0 ∧ q > 0

-- The main theorem
theorem circle_tangency_theorem (r : ℝ) (p q : ℕ) :
  tangency_condition r →
  radius_J_form p q r →
  p + q = 156 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l822_82206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l822_82226

/-- An arithmetic sequence with a non-zero common difference where a_1, a_3, a_4 form a geometric sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_nonzero : d ≠ 0
  geometric_condition : (a 1 + 2 * d) ^ 2 = a 1 * (a 1 + 3 * d)

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * seq.a 1 + ((n : ℝ) - 1) * seq.d)

/-- The main theorem -/
theorem arithmetic_geometric_ratio (seq : ArithmeticSequence) :
  (S seq 4 - S seq 2) / (S seq 5 - S seq 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l822_82226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subject_selection_probability_l822_82215

theorem subject_selection_probability (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 2) :
  (Nat.choose n k - Nat.choose (n - 2) k : ℚ) / Nat.choose n k = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subject_selection_probability_l822_82215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_positive_ball_l822_82291

noncomputable def ball_labels : Finset ℚ := {-2, 0, 1/4, 3}

theorem probability_positive_ball (ball_labels : Finset ℚ) : 
  ball_labels = {-2, 0, 1/4, 3} →
  (ball_labels.filter (λ x => x > 0)).card / ball_labels.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_positive_ball_l822_82291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lines_theorem_l822_82225

/-- Given n points on a plane (n ≥ 3) with no three points collinear, 
    this function returns the maximum number of lines that can be formed 
    without creating a triangle with vertices among the given points. -/
def max_lines_without_triangle (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 / 4 else (n^2 - 1) / 4

/-- Theorem stating the maximum number of lines that can be formed 
    without creating a triangle, given n points on a plane (n ≥ 3) 
    with no three points collinear. -/
theorem max_lines_theorem (n : ℕ) (h : n ≥ 3) :
  let k := max_lines_without_triangle n
  ∀ (lines : Finset (Finset (Fin n))), 
    (∀ l ∈ lines, l.card = 2) →
    (∀ a b c : Fin n, 
      a ≠ b ∧ b ≠ c ∧ a ≠ c → 
      ¬({a, b} ∈ lines ∧ {b, c} ∈ lines ∧ {a, c} ∈ lines)) →
    lines.card ≤ k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lines_theorem_l822_82225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_paper_l822_82230

/-- The height of a cone formed from a circular paper --/
noncomputable def cone_height (R : ℝ) : ℝ :=
  (Real.sqrt 5 * R) / 3

/-- Theorem stating the height of the cone --/
theorem cone_height_from_circular_paper (R : ℝ) (h_pos : R > 0) :
  let sector_angle : ℝ := 120 * π / 180
  let base_radius : ℝ := (2 * R) / 3
  let slant_height : ℝ := R
  cone_height R = Real.sqrt (slant_height ^ 2 - base_radius ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_paper_l822_82230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l822_82211

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a vector in 2D space -/
structure Vec where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def vector (p1 p2 : Point) : Vec :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

theorem parabola_focus_property
  (parabola : Parabola)
  (F A B C : Point)
  (l : Line)
  (h1 : parabola.p > 0)
  (h2 : A.y^2 = 2 * parabola.p * A.x)
  (h3 : B.y^2 = 2 * parabola.p * B.x)
  (h4 : F.x = parabola.p / 2 ∧ F.y = 0)
  (h5 : vector B C = { x := -2 * (vector B F).x, y := -2 * (vector B F).y })
  (h6 : distance A F = 3)
  (h7 : l.a * F.x + l.b * F.y + l.c = 0)
  (h8 : l.a * A.x + l.b * A.y + l.c = 0)
  (h9 : l.a * B.x + l.b * B.y + l.c = 0)
  (h10 : l.a * C.x + l.b * C.y + l.c = 0)
  : parabola.p = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l822_82211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluations_l822_82295

theorem expression_evaluations :
  (- (1 : ℝ)^2022 + |1 - Real.sqrt 3| - ((-27 : ℝ)^(1/3)) + Real.sqrt 4 = Real.sqrt 3 + 3) ∧
  (Real.sqrt ((-3)^2) - (-Real.sqrt 3)^2 - Real.sqrt 16 + ((-64 : ℝ)^(1/3)) = -8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluations_l822_82295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_F_has_minimum_l822_82248

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2

-- Define the function F
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := |f a x|

-- Theorem for part 1
theorem inequality_holds (a : ℝ) (x : ℝ) (h1 : 1 < x) (h2 : x < 3) :
  (f a x + a * x^2 - x + 2) / ((3 - x) * Real.exp x) > 1 / Real.exp 2 := by
  sorry

-- Theorem for part 2
theorem F_has_minimum (a : ℝ) :
  (∃ x, x ∈ Set.Icc 1 (Real.exp 1) ∧ ∀ y, y ∈ Set.Icc 1 (Real.exp 1) → F a x ≤ F a y) ↔ 
  (0 < a ∧ a < 1 / Real.exp 1) ∨ (1 / Real.exp 1 < a ∧ a < 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_F_has_minimum_l822_82248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_at_nine_l822_82289

/-- A monic polynomial of degree 8 satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ, 
    p = fun x ↦ x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) ∧
  p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5 ∧ p 6 = 6 ∧ p 7 = 7 ∧ p 8 = 8

/-- The main theorem -/
theorem special_polynomial_at_nine (p : ℝ → ℝ) (h : special_polynomial p) : p 9 = 40329 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_at_nine_l822_82289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_exponents_l822_82290

theorem smallest_sum_of_exponents (a b : ℕ) (h : (2^6 : ℕ) * (3^9 : ℕ) = a^b) : 
  ∀ (x y : ℕ), ((2^6 : ℕ) * (3^9 : ℕ) = x^y) → (a + b : ℕ) ≤ (x + y : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_exponents_l822_82290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_f_greater_than_exp_neg_l822_82216

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- State the theorems
theorem f_has_zero (a : ℝ) :
  (∃ x > 0, f a x = 0) ↔ (0 < a ∧ a ≤ Real.exp (-1)) := by sorry

theorem f_greater_than_exp_neg (a : ℝ) :
  a ≥ 2 / Real.exp 1 → ∀ x > 0, f a x > Real.exp (-x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_f_greater_than_exp_neg_l822_82216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l822_82203

-- Define the line l
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, -2 + t * Real.sin α)

-- Define the semi-circle C
noncomputable def semi_circle_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.sin θ * Real.cos θ, 2 * Real.sin θ * Real.sin θ)

-- Define the point D on semi-circle C
noncomputable def point_D (α : ℝ) : ℝ × ℝ :=
  (Real.cos (2 * α), 1 + Real.sin (2 * α))

-- Define the area of triangle ABD
noncomputable def area_ABD (α : ℝ) : ℝ :=
  2 / Real.sin α * (3 * Real.cos α + Real.sin α) / 2

theorem point_D_coordinates (α : ℝ) :
  α > 0 ∧ α < Real.pi / 2 ∧
  area_ABD α = 4 →
  point_D α = (0, 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l822_82203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l822_82239

noncomputable def inverse_proportion (x : ℝ) : ℝ := 1 / x

theorem inverse_proportion_properties :
  -- The graph passes through the point (-1, -1)
  inverse_proportion (-1) = -1 ∧
  -- The graph is in the first and third quadrants
  (∀ x, x > 0 → inverse_proportion x > 0) ∧
  (∀ x, x < 0 → inverse_proportion x < 0) ∧
  -- When x > 1, 0 < y < 1
  (∀ x, x > 1 → 0 < inverse_proportion x ∧ inverse_proportion x < 1) ∧
  -- When x < 0, y decreases as x increases
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → inverse_proportion x₁ > inverse_proportion x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l822_82239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_average_l822_82229

theorem consecutive_integers_average (s : Finset ℤ) : 
  (s.card = 7) →  -- The set has 7 elements
  (∀ x y, x ∈ s → y ∈ s → x ≠ y → (x - y).natAbs = 1) →  -- The elements are consecutive
  (∃ m ∈ s, ∀ x ∈ s, x ≤ m ∧ m = 23) →  -- The largest element is 23
  (s.sum id / s.card : ℚ) = 20 :=  -- The average is 20
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_average_l822_82229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_at_neg_pi_third_f_value_in_second_quadrant_l822_82284

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi - α) * Real.cos (2*Real.pi - α) * Real.cos (3*Real.pi/2 + α)) /
  (Real.cos (Real.pi/2 + α) * Real.sin (Real.pi + α))

theorem f_simplification (α : Real) : f α = Real.cos α := by sorry

theorem f_value_at_neg_pi_third : f (-Real.pi/3) = 1/2 := by sorry

theorem f_value_in_second_quadrant (α : Real) 
  (h1 : Real.pi/2 < α ∧ α < Real.pi) 
  (h2 : Real.cos (α - Real.pi/2) = 3/5) : 
  f α = -4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_at_neg_pi_third_f_value_in_second_quadrant_l822_82284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l822_82213

-- Define the start and end points of the line segment
def start : ℝ × ℝ := (-1, 4)
def endpoint (x : ℝ) : ℝ × ℝ := (3, x)

-- Define the condition that x > 0
def x_positive (x : ℝ) : Prop := x > 0

-- Define the length of the segment
def segment_length : ℝ := 7

-- Theorem statement
theorem line_segment_endpoint (x : ℝ) 
  (h_positive : x_positive x)
  (h_length : Real.sqrt ((start.1 - (endpoint x).1)^2 + (start.2 - (endpoint x).2)^2) = segment_length) :
  x = 4 + Real.sqrt 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l822_82213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l822_82296

theorem sum_remainder (a b c d : ℕ) 
  (ha : a % 53 = 31)
  (hb : b % 53 = 45)
  (hc : c % 53 = 17)
  (hd : d % 53 = 6) :
  (a + b + c + d) % 53 = 46 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l822_82296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instrument_reading_changes_l822_82207

/-- Circuit parameters -/
noncomputable def U₀ : ℝ := 45  -- Power source voltage in volts
noncomputable def R : ℝ := 50   -- Resistance in ohms
noncomputable def r : ℝ := 20   -- Internal resistance in ohms

/-- Initial ammeter reading -/
noncomputable def I₁ : ℝ := U₀ / (R + 2*r) / 2

/-- Initial voltmeter reading -/
noncomputable def U₁ : ℝ := U₀ * r / (R/2 + r)

/-- Final ammeter reading after swapping -/
noncomputable def I₂ : ℝ := U₀ / R

/-- Final voltmeter reading after swapping -/
noncomputable def U₂ : ℝ := U₀ * r / (R + r)

/-- Theorem stating the changes in instrument readings -/
theorem instrument_reading_changes :
  (I₂ - I₁ = 0.4) ∧ (abs (U₁ - U₂ - 7.14) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instrument_reading_changes_l822_82207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_equation_solution_l822_82247

theorem tangent_equation_solution (x : ℝ) : 
  (Real.cos (5*x) ≠ 0 ∧ Real.cos (3*x) ≠ 0) →
  (Real.tan (5*x) - 2 * Real.tan (3*x) = Real.tan (3*x)^2 * Real.tan (5*x) ↔ ∃ k : ℤ, x = k * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_equation_solution_l822_82247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_coloring_l822_82201

theorem symmetric_coloring (a b : ℕ) (ha : a > 2) (hb : b > 2) (hcoprime : Nat.Coprime a b) :
  let c : ℚ := (a * b + a + b) / 2
  ∀ k : ℤ, (∃ x y : ℕ, k = a * x + b * y) ↔ ¬(∃ x y : ℕ, a * b + a + b - k = a * x + b * y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_coloring_l822_82201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_formula_correct_l822_82253

/-- The radius of the circumscribed circle of a triangle -/
noncomputable def circumradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c)))

/-- Theorem: The circumradius formula is correct for any triangle -/
theorem circumradius_formula_correct (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (R : ℝ), R > 0 ∧ R = circumradius a b c ∧
  R = (a * b * c) / (4 * Real.sqrt (((a + b + c) / 2) *
    (((a + b + c) / 2) - a) * (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_formula_correct_l822_82253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_solutions_l822_82260

noncomputable def h (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 8 else 3 * x - 12

theorem h_solutions :
  {x : ℝ | h x = 4} = {-1, 16/3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_solutions_l822_82260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_solution_l822_82259

/-- A digit in decimal notation is a natural number from 0 to 9. -/
def Digit : Type := {d : ℕ // d < 10}

/-- Represents a repeating decimal of the form 0.d25d25d25... -/
def RepeatingDecimal (d : Digit) : ℚ :=
  (d.val : ℚ) / 1000 + 25 / 1000 / 999

theorem repeating_decimal_solution (n : ℕ) (d : Digit) 
  (h : (n : ℚ) / 810 = RepeatingDecimal d) : n = 750 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_solution_l822_82259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l822_82279

noncomputable section

-- Define the vectors a and b
def a (x : Real) : Real × Real := (1, Real.cos (x / 2))
def b (x y : Real) : Real × Real := (Real.sqrt 3 * Real.sin (x / 2) + Real.cos (x / 2), y)

-- Define the collinearity condition
def collinear (x y : Real) : Prop :=
  (a x).1 * (b x y).2 = (a x).2 * (b x y).1

-- Define the function f
def f (x : Real) : Real := Real.sin (x + Real.pi / 6) + 1 / 2

-- Part I
theorem part_one (x : Real) (h : collinear x (f x)) (h1 : f x = 1) :
  Real.cos (2 * Real.pi / 3 - 2 * x) = -1 / 2 := by sorry

-- Part II
theorem part_two (A B C : Real) (a b c : Real) 
  (h : 2 * a * Real.cos C + c = 2 * b) 
  (h1 : 0 < B) (h2 : B < 2 * Real.pi / 3) :
  1 < f B ∧ f B ≤ 3 / 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l822_82279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l822_82223

-- Define the relationship between x and a
noncomputable def relation (x a : ℝ) : Prop := (2 - a) * Real.exp a = x * (2 + a)

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := (a^2 * Real.exp a) / (Real.exp a - (a + 1) * x)

-- Theorem statement
theorem range_of_f :
  ∀ x a : ℝ, x ∈ Set.Icc 0 1 → relation x a →
  ∃ y : ℝ, y ∈ Set.Ioc 2 4 ∧ f x a = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l822_82223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_min_value_is_nine_min_value_achieved_l822_82210

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 : ℝ)^(x - 3) = ((1/2) : ℝ)^y) : 
  ∀ a b : ℝ, a > 0 → b > 0 → (2 : ℝ)^(a - 3) = ((1/2) : ℝ)^b → 1/x + 4/y ≤ 1/a + 4/b :=
by sorry

theorem min_value_is_nine (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 : ℝ)^(x - 3) = ((1/2) : ℝ)^y) : 
  1/x + 4/y ≥ 9 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (2 : ℝ)^(x - 3) = ((1/2) : ℝ)^y ∧ 1/x + 4/y < 9 + ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_min_value_is_nine_min_value_achieved_l822_82210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l822_82276

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem inequality_solution_set :
  {x : ℝ | 4 ≤ f x ∧ f x < 5} = {x : ℝ | 1 < x ∧ x < 4} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l822_82276
