import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l928_92897

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 0 1

-- State the theorem
theorem domain_transformation :
  (∀ y ∈ domain_f, f y = f y) →
  (Set.Icc (-1/2) 0 = {x : ℝ | ∃ y ∈ domain_f, y = 2*x + 1}) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l928_92897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l928_92813

-- Define the equation
def equation (x y θ : ℝ) : Prop := x^2 + y^2 / Real.cos θ = 4

-- Theorem statement
theorem not_parabola :
  ∀ θ : ℝ, ¬∃ a b c : ℝ, ∀ x y : ℝ, 
    equation x y θ ↔ (y - b)^2 = 4*a*(x - c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l928_92813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_on_ellipse_circumcircle_through_point_l928_92849

-- Define the ellipse and parabola
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def parabola (x y p : ℝ) : Prop := x^2 = 2 * p * y

-- Define the intersection points A and B
def intersection_points (p : ℝ) : Prop :=
  ∃ (x_a y_a x_b y_b : ℝ),
    ellipse x_a y_a ∧ parabola x_a y_a p ∧
    ellipse x_b y_b ∧ parabola x_b y_b p

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Helper function to determine if a point is the circumcenter of a triangle
def is_circumcenter (c a b o : ℝ × ℝ) : Prop := sorry

-- Helper function to determine if a point is on the circle defined by three points
def on_circle (n a b o : ℝ × ℝ) : Prop := sorry

-- Theorem for the first question
theorem circumcenter_on_ellipse (p : ℝ) :
  p > 0 →
  intersection_points p →
  (∃ (x_c y_c : ℝ), ellipse x_c y_c ∧ 
    (∀ (x_a y_a x_b y_b : ℝ), 
      ellipse x_a y_a ∧ parabola x_a y_a p ∧
      ellipse x_b y_b ∧ parabola x_b y_b p →
      is_circumcenter (x_c, y_c) (x_a, y_a) (x_b, y_b) origin)) →
  p = (7 - Real.sqrt 13) / 6 := by sorry

-- Theorem for the second question
theorem circumcircle_through_point (p : ℝ) :
  p > 0 →
  intersection_points p →
  (∃ (x_a y_a x_b y_b : ℝ), 
    ellipse x_a y_a ∧ parabola x_a y_a p ∧
    ellipse x_b y_b ∧ parabola x_b y_b p ∧
    on_circle (0, 13/2) (x_a, y_a) (x_b, y_b) origin) →
  p = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_on_ellipse_circumcircle_through_point_l928_92849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statements_describe_relationships_l928_92889

-- Define the basic geometric entities
def Point : Type := Unit
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relationships described in the statements
def belongs_to (p : Point) (l : Line) : Prop := sorry
def belongs_to_plane (p : Point) (pl : Plane) : Prop := sorry
def lies_in (l : Line) (pl : Plane) : Prop := sorry
def divides_into_convex_regions (l : Line) (pl : Plane) : Prop := sorry

-- Define the statements
def statement1 (l : Line) (pl : Plane) : Prop :=
  ∀ (p1 p2 : Point), belongs_to p1 l → belongs_to p2 l →
  belongs_to_plane p1 pl → belongs_to_plane p2 pl →
  ∀ (p : Point), belongs_to p l → belongs_to_plane p pl

def statement2 (l : Line) (pl : Plane) : Prop :=
  lies_in l pl → divides_into_convex_regions l pl

-- Define a proposition to represent the conclusion
def conclusion : Prop :=
  (∀ (l : Line) (pl : Plane), statement1 l pl) ∧
  (∀ (l : Line) (pl : Plane), statement2 l pl)

-- Theorem to prove
theorem statements_describe_relationships :
  conclusion →
  "Both statements describe relationships between lines and planes" = "Both statements describe relationships between lines and planes" :=
by
  intro h
  rfl

#check statements_describe_relationships

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statements_describe_relationships_l928_92889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l928_92859

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

-- State the theorem
theorem f_sum_property (x₁ x₂ : ℝ) (h : x₁ + x₂ = 1) :
  f x₁ + f x₂ = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l928_92859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_4_or_6_l928_92826

def is_multiple_of_4_or_6 (n : ℕ) : Bool := n % 4 = 0 ∨ n % 6 = 0

def count_multiples (n : ℕ) : ℕ := (Finset.range n).filter (fun x => is_multiple_of_4_or_6 x) |>.card

theorem probability_multiple_4_or_6 :
  count_multiples 71 / 70 = 23 / 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_4_or_6_l928_92826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l928_92832

-- Define x as noncomputable
noncomputable def x : ℝ := 1 / (2 - Real.sqrt 3)

-- Theorem for part 1
theorem part_one : x + 1/x = 4 := by sorry

-- Theorem for part 2
theorem part_two : (7 - 4*Real.sqrt 3)*x^2 + (2 - Real.sqrt 3)*x + Real.sqrt 3 = 2 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l928_92832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximatePi_valid_l928_92816

/-- A pair of real numbers (x,y) where x and y are in [0,1] and x²+y² < 1 -/
structure UnitCirclePair where
  x : ℝ
  y : ℝ
  x_in_range : x ∈ Set.Icc 0 1
  y_in_range : y ∈ Set.Icc 0 1
  in_unit_circle : x^2 + y^2 < 1

/-- The approximate value of π given a sample of points -/
noncomputable def approximatePi (n m : ℕ) : ℝ :=
  4 * (m : ℝ) / (n : ℝ)

/-- Theorem stating that the approximation of π is valid -/
theorem approximatePi_valid (n m : ℕ) (pairs : Finset UnitCirclePair) 
    (h_n : n > 0)
    (h_m : m > 0)
    (h_m_le_n : m ≤ n)
    (h_pairs_card : pairs.card = n)
    (h_m_count : m = (pairs.filter (λ p => p.x^2 + p.y^2 < 1)).card) :
  ∃ (ε : ℝ), ε > 0 ∧ |approximatePi n m - π| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximatePi_valid_l928_92816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_l928_92882

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the first transformation (halving the x-coordinate)
noncomputable def transform1 (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (2 * x)

-- Define the second transformation (shifting left by π/12)
noncomputable def transform2 (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x + Real.pi / 12)

-- The final transformed function
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

-- Theorem stating that the transformations result in the expected function
theorem transformations_result (x : ℝ) : 
  (transform2 (transform1 f)) x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_l928_92882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l928_92827

/-- Calculates the average speed of a trip given the total distance, speed of the first half, and the time ratio of the second half to the first half. -/
noncomputable def averageSpeed (totalDistance : ℝ) (firstHalfSpeed : ℝ) (secondHalfTimeRatio : ℝ) : ℝ :=
  let firstHalfDistance := totalDistance / 2
  let firstHalfTime := firstHalfDistance / firstHalfSpeed
  let totalTime := firstHalfTime * (1 + secondHalfTimeRatio)
  totalDistance / totalTime

/-- Theorem stating that for a 640-mile trip with the given conditions, the average speed is 40 mph. -/
theorem average_speed_theorem :
  averageSpeed 640 80 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l928_92827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l928_92885

theorem triangle_angle_measure (a b c : ℝ) (h : b^2 + c^2 - a^2 = b*c) :
  Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l928_92885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_B_implies_not_all_correct_l928_92860

/-- Represents whether a student answered all mathematics questions correctly -/
def answered_all_correctly (student : String) : Prop := sorry

/-- Represents whether a student received at least a B grade -/
def received_at_least_B (student : String) : Prop := sorry

/-- The given condition: if a student answers all mathematics questions correctly, 
    they receive at least a B -/
axiom condition (student : String) : 
  answered_all_correctly student → received_at_least_B student

/-- Theorem to prove: if a student did not receive at least a B, 
    then they must have answered at least one mathematics question incorrectly -/
theorem not_B_implies_not_all_correct (student : String) :
  ¬(received_at_least_B student) → ¬(answered_all_correctly student) := by
  intro h
  contrapose! h
  exact condition student h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_B_implies_not_all_correct_l928_92860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_nested_polynomial_l928_92805

/-- The degree of the polynomial ((2x^3 + 5)^8)^7 is 168 -/
theorem degree_of_nested_polynomial : 
  let p : Polynomial ℚ := (2 * X^3 + 5)^8^7
  Polynomial.degree p = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_nested_polynomial_l928_92805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_super_ball_distance_l928_92834

noncomputable def initial_height : ℝ := 20
noncomputable def bounce_ratio : ℝ := 3/5
def num_bounces : ℕ := 4

noncomputable def bounce_height (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

noncomputable def total_distance : ℝ :=
  2 * (initial_height + 
       bounce_height 1 + 
       bounce_height 2 + 
       bounce_height 3) + 
  bounce_height 4

theorem super_ball_distance :
  total_distance = 90.112 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_super_ball_distance_l928_92834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l928_92864

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 6*x + 8)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≠ 2 ∧ x ≠ 4}

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l928_92864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_l928_92876

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define the relation for a line lying in a plane
variable (lies_in : Line → Plane → Prop)

-- Define the intersection relation for lines
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem planes_parallel
  (α α₁ : Plane) (a b a₁ b₁ : Line)
  (h1 : lies_in a α)
  (h2 : lies_in b α)
  (h3 : lies_in a₁ α₁)
  (h4 : lies_in b₁ α₁)
  (h5 : intersects a b)
  (h6 : intersects a₁ b₁)
  (h7 : parallel a a₁)
  (h8 : parallel b b₁) :
  α = α₁ ∨ plane_parallel α α₁ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_l928_92876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_m_range_l928_92800

noncomputable def f (x : ℝ) : ℝ := 1 - (2 * Real.exp (Real.log 5 * x)) / (Real.exp (Real.log 5 * x) + 1)

theorem function_properties_imply_m_range :
  (∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → f (-x) = -f x) →  -- f is odd
  (∀ x y, x ∈ Set.Ioo (-2 : ℝ) 2 → y ∈ Set.Ioo (-2 : ℝ) 2 → x < y → f x > f y) →  -- f is decreasing
  (∀ m : ℝ, f (m - 1) + f (2 * m + 1) > 0) →
  ∀ m : ℝ, m ∈ Set.Ioo (-1 : ℝ) 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_m_range_l928_92800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_average_cost_l928_92874

/-- The optimal number of floors that minimizes the average comprehensive cost -/
def optimal_floors : ℕ := 15

/-- The cost function representing the average comprehensive cost per square meter -/
noncomputable def cost_function (x : ℝ) : ℝ := 560 + 48 * x + 10800 / x

/-- Theorem stating that the optimal_floors minimizes the average comprehensive cost -/
theorem minimize_average_cost :
  ∀ x : ℕ, x ≥ 10 →
  cost_function (↑x) ≥ cost_function (↑optimal_floors) :=
by
  sorry

#check minimize_average_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_average_cost_l928_92874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_symmetry_l928_92812

-- Define an inverse proportion function
noncomputable def inverse_proportion (k : ℝ) : ℝ → ℝ := fun x => k / x

-- State the theorem
theorem inverse_proportion_symmetry (k : ℝ) :
  (inverse_proportion k 1 = 2) → (inverse_proportion k (-1) = -2) :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_symmetry_l928_92812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_drive_more_suitable_remaining_time_formula_l928_92866

-- Define the given parameters
def file_size : ℝ := 1.5
def e_drive_used : ℝ := 11.52
def e_drive_unused_percent : ℝ := 0.10
def c_drive_total : ℝ := 9.75
def c_drive_used_percent : ℝ := 0.80

-- Define the functions to calculate unused space
noncomputable def e_drive_unused_space : ℝ :=
  e_drive_used / (1 - e_drive_unused_percent) * e_drive_unused_percent

noncomputable def c_drive_unused_space : ℝ :=
  c_drive_total * (1 - c_drive_used_percent)

-- Theorem to prove C drive is more suitable
theorem c_drive_more_suitable :
  c_drive_unused_space > e_drive_unused_space ∧ c_drive_unused_space > file_size :=
by sorry

-- Define the function to calculate remaining download time
noncomputable def remaining_download_time (downloaded_percent : ℝ) (time_elapsed : ℝ) : ℝ :=
  (1 - downloaded_percent) / (downloaded_percent / time_elapsed)

-- Theorem to prove the formula for remaining download time
theorem remaining_time_formula (downloaded_percent : ℝ) (time_elapsed : ℝ) :
  remaining_download_time downloaded_percent time_elapsed =
  (1 - downloaded_percent) * time_elapsed / downloaded_percent :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_drive_more_suitable_remaining_time_formula_l928_92866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_two_roots_l928_92820

noncomputable section

-- Define f as a function from ℝ to ℝ
def f : ℝ → ℝ := sorry

-- f is an even function
axiom f_even : ∀ x : ℝ, f x = f (-x)

-- f(x+1) = f(1-x) for all real x
axiom f_property : ∀ x : ℝ, f (x + 1) = f (1 - x)

-- f(x) = ln x for 1 ≤ x ≤ 2
axiom f_ln : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = Real.log x

theorem a_range_for_two_roots :
  ∃ a : ℝ, (1 - Real.log 2) / 4 < a ∧ a ≤ 1 / 5 ∧
  (∃ x y : ℝ, 3 ≤ x ∧ x < y ∧ y ≤ 5 ∧
   f x + a * x - 1 = 0 ∧ f y + a * y - 1 = 0) ∧
  (∀ a' : ℝ, ((1 - Real.log 2) / 4 < a' ∧ a' ≤ 1 / 5) →
   (∃ x y : ℝ, 3 ≤ x ∧ x < y ∧ y ≤ 5 ∧
    f x + a' * x - 1 = 0 ∧ f y + a' * y - 1 = 0)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_two_roots_l928_92820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l928_92831

/-- Converts cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ := (10, Real.pi / 4, -3)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ × ℝ := (5 * Real.sqrt 2, 5 * Real.sqrt 2, -3)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l928_92831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_k_is_one_no_zeros_condition_l928_92895

-- Define the function f(x) with parameter k
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log (x - 1) - k * (x - 1) + 1

-- Theorem 1: Maximum value of f(x) when k=1
theorem max_value_when_k_is_one :
  ∃ (M : ℝ), M = 0 ∧ ∀ x > 1, f 1 x ≤ M := by
  sorry

-- Theorem 2: Condition for f(x) to have no zeros
theorem no_zeros_condition (k : ℝ) :
  (∀ x > 1, f k x ≠ 0) ↔ k > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_k_is_one_no_zeros_condition_l928_92895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_main_theorem_l928_92844

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

-- Define the derivative of the function
noncomputable def f_deriv (x : ℝ) : ℝ := (x^2 - 1) / x

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, x > 0 → (f_deriv x < 0 ↔ x < 1) := by
  sorry

-- State the main theorem about the monotonic decreasing interval
theorem main_theorem :
  ∀ a b : ℝ, 0 < a ∧ a < b → 
  (∀ x : ℝ, a < x ∧ x < b → f_deriv x < 0) ↔ (a = 0 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_main_theorem_l928_92844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l928_92888

theorem power_equation_solution :
  ∃ x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x = (256 : ℝ)^5 ∧ x = 5/2 := by
  use (5/2 : ℝ)
  constructor
  · -- Proof of the equation
    sorry
  · -- Proof that x = 5/2
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l928_92888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_with_valid_answers_l928_92811

/-- Represents a student's answers to the exam questions -/
def StudentAnswers := Fin 4 → Fin 3

/-- The property that for any 3 students, there is at least 1 question where their answers are all different -/
def ValidAnswerSet (answers : Finset StudentAnswers) : Prop :=
  ∀ s1 s2 s3, s1 ∈ answers → s2 ∈ answers → s3 ∈ answers →
    s1 ≠ s2 → s2 ≠ s3 → s1 ≠ s3 →
    ∃ q : Fin 4, s1 q ≠ s2 q ∧ s2 q ≠ s3 q ∧ s1 q ≠ s3 q

/-- The theorem stating the maximum number of students satisfying the condition -/
theorem max_students_with_valid_answers :
  (∃ (answers : Finset StudentAnswers), ValidAnswerSet answers ∧ answers.card = 9) ∧
  (∀ (answers : Finset StudentAnswers), ValidAnswerSet answers → answers.card ≤ 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_with_valid_answers_l928_92811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_side_is_correct_largest_rectangle_dimensions_are_correct_l928_92851

variable (a b : ℝ)

/-- The side length of the largest square with vertex C inside the right triangle ABC -/
noncomputable def largest_square_side (a b : ℝ) : ℝ := a * b / (a + b)

/-- The width of the largest rectangle with vertex C inside the right triangle ABC -/
noncomputable def largest_rectangle_width (a : ℝ) : ℝ := a / 2

/-- The height of the largest rectangle with vertex C inside the right triangle ABC -/
noncomputable def largest_rectangle_height (b : ℝ) : ℝ := b / 2

/-- Theorem: The side length of the largest square with vertex C inside the right triangle ABC is ab / (a + b) -/
theorem largest_square_side_is_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  largest_square_side a b = a * b / (a + b) := by
  -- The proof is omitted
  sorry

/-- Theorem: The dimensions of the largest rectangle with vertex C inside the right triangle ABC are a/2 and b/2 -/
theorem largest_rectangle_dimensions_are_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (largest_rectangle_width a, largest_rectangle_height b) = (a / 2, b / 2) := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_side_is_correct_largest_rectangle_dimensions_are_correct_l928_92851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_score_is_111_l928_92842

noncomputable def matthew_scores : List ℝ := [88, 77, 82, 90, 75, 85]

noncomputable def current_average : ℝ := (matthew_scores.sum) / matthew_scores.length

noncomputable def target_average : ℝ := current_average + 4

noncomputable def minimum_score (scores : List ℝ) (target : ℝ) : ℝ :=
  (target * (scores.length + 1)) - scores.sum

theorem minimum_score_is_111 :
  ⌈minimum_score matthew_scores target_average⌉ = 111 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_score_is_111_l928_92842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_sided_polygon_diagonals_l928_92810

structure Polygon where
  vertices : ℕ
  diagonals : ℕ

def Regular (p : Polygon) : Prop := sorry

theorem seven_sided_polygon_diagonals :
  ∀ (p : Polygon), 
    Regular p → 
    p.vertices = 7 → 
    p.diagonals = 14 :=
by
  intro p reg_p vert_7
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_sided_polygon_diagonals_l928_92810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_problem_l928_92881

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of white balls in the bag -/
def num_white_balls : ℕ := 6

/-- The number of balls to be drawn -/
def num_drawn : ℕ := 5

/-- The score of a red ball -/
def red_score : ℕ := 2

/-- The score of a white ball -/
def white_score : ℕ := 1

/-- The minimum total score required in part 1 -/
def min_score : ℕ := 7

/-- The exact total score required in part 2 -/
def exact_score : ℕ := 8

/-- Function to represent the number of ways to draw balls with a certain score -/
def ways_to_draw (score : ℕ) : ℕ := sorry

/-- Function to represent the number of arrangements with only 2 red balls adjacent -/
def arrangements_with_two_adjacent (score : ℕ) : ℕ := sorry

theorem ball_drawing_problem :
  (∃ (ways : ℕ), ways = 186 ∧ ways = ways_to_draw min_score) ∧
  (∃ (arrangements : ℕ), arrangements = 4320 ∧ arrangements = arrangements_with_two_adjacent exact_score) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_problem_l928_92881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_cut_probability_l928_92899

/-- Represents the length of a wire in centimeters -/
def WireLength : ℝ := 80

/-- Represents the minimum length of each segment in centimeters -/
def MinSegmentLength : ℝ := 20

/-- Represents the probability space of cutting the wire -/
def CuttingSpace : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1 + p.2 ≤ WireLength}

/-- Represents the event where each segment is at least MinSegmentLength -/
def ValidCutEvent : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ CuttingSpace ∧ 
    p.1 ≥ MinSegmentLength ∧ 
    p.2 ≥ MinSegmentLength ∧ 
    WireLength - p.1 - p.2 ≥ MinSegmentLength}

/-- The probability of the ValidCutEvent -/
theorem valid_cut_probability : 
  (MeasureTheory.volume ValidCutEvent) / (MeasureTheory.volume CuttingSpace) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_cut_probability_l928_92899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_sum_68_l928_92884

theorem count_pairs_sum_68 : 
  let S := Finset.filter (fun p : ℕ × ℕ => 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 ≤ 64 ∧ p.1 + p.2 = 68) (Finset.range 65 ×ˢ Finset.range 65)
  S.card = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_sum_68_l928_92884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_leg_ratio_l928_92841

theorem right_triangle_leg_ratio (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_area : (Real.sqrt 3 / 4) * c^2 = a * b) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) : 
  max (a / b) (b / a) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_leg_ratio_l928_92841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l928_92896

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/2) * tan (5*x + π/4)

-- Define the domain
def domain : Set ℝ := {x | ∀ k : ℤ, x ≠ k*π/5 + π/20}

-- Define the monotonic intervals
def monotonicIntervals : Set (Set ℝ) := 
  {I | ∃ k : ℤ, I = Set.Ioo (k*π/5 - 3*π/20) (k*π/5 + π/20)}

-- Define the symmetry centers
def symmetryCenters : Set (ℝ × ℝ) := 
  {p | ∃ k : ℤ, p = (k*π/10 - π/20, 0)}

theorem function_properties :
  (∀ x ∈ domain, f x = f x) ∧
  (∀ I ∈ monotonicIntervals, ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) ∧
  (∀ c ∈ symmetryCenters, ∀ x, f (c.1 + x) = -f (c.1 - x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l928_92896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_solution_l928_92861

-- Define the function G
noncomputable def G (a b c d e : ℝ) : ℝ := a^b + c * d - e

-- Define the problem statement
theorem nearest_integer_solution :
  ∃ x : ℝ, G 3 x 5 12 10 = 500 ∧ 
  ∀ y : ℤ, |y - x| < 1 → |G 3 (↑y) 5 12 10 - 500| ≥ |G 3 6 5 12 10 - 500| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_solution_l928_92861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equivalence_ellipse_equivalence_l928_92883

-- Define the parameterized equations for the straight line
def line_param (t : ℝ) : ℝ × ℝ :=
  (1 - 3*t, 4*t)

-- Define the standard form equation for the straight line
def line_standard (x y : ℝ) : Prop :=
  4*x + 3*y - 4 = 0

-- Theorem for the straight line
theorem line_equivalence :
  ∀ t x y, line_param t = (x, y) → line_standard x y :=
by sorry

-- Define the parameterized equations for the ellipse
noncomputable def ellipse_param (θ : ℝ) : ℝ × ℝ :=
  (5 * Real.cos θ, 4 * Real.sin θ)

-- Define the standard form equation for the ellipse
def ellipse_standard (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1

-- Theorem for the ellipse
theorem ellipse_equivalence :
  ∀ θ x y, ellipse_param θ = (x, y) → ellipse_standard x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equivalence_ellipse_equivalence_l928_92883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_circle_area_l928_92863

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem -/
def problemConditions (t : Triangle) : Prop :=
  t.a = 2 ∧ f t.A = 2

/-- The theorem to be proved -/
theorem max_inscribed_circle_area (t : Triangle) (h : problemConditions t) :
  (∃ r : ℝ, r ≤ Real.sqrt 3 / 3 ∧
    ∀ s : ℝ, s * (t.a + t.b + t.c) / 2 = t.b * t.c * Real.sin t.A / 2 → s ≤ r) ∧
  Real.pi * (Real.sqrt 3 / 3)^2 = Real.pi / 3 := by
  sorry

#check max_inscribed_circle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_circle_area_l928_92863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_L_l928_92868

-- Define the circles and line
def circle_A : ℝ × ℝ → Prop := λ p ↦ (p.1 - 3)^2 + (p.2 - 3)^2 = 9
def circle_B : ℝ × ℝ → Prop := λ p ↦ (p.1 - 3)^2 + (p.2 - 1)^2 = 1

-- Define tangency conditions
def tangent_to_x_axis (circle : ℝ × ℝ → Prop) : Prop := ∃ x, circle (x, 0)
def tangent_to_y_axis (circle : ℝ × ℝ → Prop) : Prop := ∃ y, circle (0, y)
def circles_tangent (circle1 circle2 : ℝ × ℝ → Prop) : Prop :=
  ∃ p, circle1 p ∧ circle2 p

-- Define line L
def line_L : ℝ → ℝ → Prop := λ x y ↦ ∃ m b, y = m * x + b ∧
  (∃ p, circle_A p ∧ (p.2 = m * p.1 + b)) ∧
  (∃ q, circle_B q ∧ (q.2 = m * q.1 + b))

-- Theorem statement
theorem y_intercept_of_L :
  tangent_to_x_axis circle_A →
  tangent_to_y_axis circle_A →
  tangent_to_x_axis circle_B →
  circles_tangent circle_A circle_B →
  ∃ y, line_L 0 y ∧ y = 10 + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_L_l928_92868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_formula_l928_92823

/-- Sequence a_n -/
def a : ℕ → ℝ := sorry

/-- Sequence b_n -/
def b : ℕ → ℝ := sorry

/-- Sum of first n terms of a_n -/
def S : ℕ → ℝ := sorry

/-- Sum of first n terms of b_n -/
def T : ℕ → ℝ := sorry

/-- b_n is an arithmetic sequence -/
axiom b_arithmetic : ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- All terms of b_n are positive -/
axiom b_positive : ∀ n : ℕ, b n > 0

/-- Definition of a_1 -/
axiom a_1 : a 1 = 1

/-- Recurrence relation for a_n -/
axiom a_recurrence : ∀ n : ℕ+, a (n + 1) = 2 * S n + 1

/-- Sum of first three terms of b_n -/
axiom b_sum_3 : b 1 + b 2 + b 3 = 15

/-- a_n + b_n forms a geometric sequence for first three terms -/
axiom ab_geometric : ∃ r : ℝ, (a 2 + b 2) = r * (a 1 + b 1) ∧ (a 3 + b 3) = r * (a 2 + b 2)

/-- Main theorem: T_n = n^2 + 2n -/
theorem T_formula : ∀ n : ℕ, T n = n^2 + 2*n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_formula_l928_92823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_greater_than_original_number_l928_92801

/-- Represents a 100-digit number as a list of natural numbers -/
def Digits := List ℕ

/-- Checks if a list of digits represents a valid 100-digit number with non-zero first digit -/
def is_valid_100_digit_number (d : Digits) : Prop :=
  d.length = 100 ∧ d.head? ≠ some 0

/-- Calculates the product of sums of all possible pairs of digits -/
def product_of_digit_pair_sums (d : Digits) : ℕ :=
  d.foldl (fun acc x ↦ d.foldl (fun acc' y ↦ acc' * (x + y)) acc) 1

/-- The original 100-digit number as a natural number -/
def number_from_digits (d : Digits) : ℕ :=
  d.foldl (fun acc x ↦ acc * 10 + x) 0

theorem product_greater_than_original_number (d : Digits) 
  (h : is_valid_100_digit_number d) : 
  product_of_digit_pair_sums d > number_from_digits d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_greater_than_original_number_l928_92801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_theorem_l928_92879

-- Define the quadrants
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to determine the quadrant of an angle
noncomputable def angle_quadrant (θ : ℝ) : Quadrant :=
  if Real.sin θ > 0 && Real.cos θ > 0 then Quadrant.first
  else if Real.sin θ > 0 && Real.cos θ < 0 then Quadrant.second
  else if Real.sin θ < 0 && Real.cos θ < 0 then Quadrant.third
  else Quadrant.fourth

-- Theorem statement
theorem second_quadrant_theorem (θ : ℝ) :
  Real.sin θ > 0 → Real.cos θ < 0 → angle_quadrant θ = Quadrant.second :=
by
  intro h_sin h_cos
  unfold angle_quadrant
  simp [h_sin, h_cos]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_theorem_l928_92879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_in_solution_set_l928_92833

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7))

-- Define the solution set
def solution_set : Set ℝ := Set.Iio 2 ∪ Set.Ioo 4 5 ∪ Set.Ioo 6 7 ∪ Set.Ioi 7

-- State the theorem
theorem f_positive_iff_in_solution_set :
  ∀ x : ℝ, f x > 0 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_in_solution_set_l928_92833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_of_data_set_l928_92836

noncomputable def is_ascending (l : List ℝ) : Prop :=
  ∀ i j, i < j → i < l.length → j < l.length → l[i]! ≤ l[j]!

noncomputable def median (l : List ℝ) : ℝ :=
  if l.length % 2 = 0
  then (l[l.length / 2 - 1]! + l[l.length / 2]!) / 2
  else l[l.length / 2]!

noncomputable def mode (l : List ℝ) : ℝ :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) (l.head!)

theorem mode_of_data_set (data : List ℝ) (x : ℝ) :
  data = [1, 2, 4, x, 6, 9] →
  is_ascending data →
  median data = 5 →
  mode data = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_of_data_set_l928_92836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l928_92817

noncomputable def A : ℝ × ℝ := (-4.5, -1.5)
noncomputable def B : ℝ × ℝ := (8.5, 2.5)
noncomputable def C : ℝ × ℝ := (-3, 4.5)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def perimeter : ℝ :=
  distance A B + distance B C + distance C A

theorem triangle_perimeter :
  perimeter = Real.sqrt 185 + Real.sqrt 136.25 + Real.sqrt 38.25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l928_92817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l928_92880

/-- Represents the expansion of (√x - 1/(2⁴√x))^n -/
noncomputable def expansion (x : ℝ) (n : ℕ) := (Real.sqrt x - 1 / (2 * x^(1/4))) ^ n

/-- The general term of the expansion -/
noncomputable def general_term (x : ℝ) (n r : ℕ) : ℝ := 
  (n.choose r) * (-1/2)^r * x^((2*n - 3*r : ℝ)/4)

theorem expansion_properties :
  ∃ (n : ℕ), 
    (∀ x, general_term x n 4 = (15/16 : ℝ)) ∧ 
    (∀ x, general_term x n 0 = x^3) ∧
    (∀ r, r ≠ 0 → r ≠ 4 → ∃ m : ℤ, (2*n - 3*r : ℝ)/4 = m → False) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l928_92880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_friends_pairs_2000_1000_l928_92870

/-- Given a group of n people, where each person invites m others,
    and friendship is mutual (both must invite each other),
    the minimum number of pairs of friends is at least 1000. -/
def min_friends_pairs (n m : ℕ) : ℕ := 1000

/-- Theorem stating that for 2000 people, each inviting 1000 others,
    the minimum number of pairs of friends is 1000. -/
theorem min_friends_pairs_2000_1000 : 
  min_friends_pairs 2000 1000 = 1000 := by
  -- Define key values
  let total_invitations : ℕ := 2000 * 1000
  let total_pairs : ℕ := 2000 * 1999 / 2

  -- Proof sketch (to be completed)
  sorry

/-- Helper lemma: The number of friend pairs is bounded below by the difference
    between total invitations and total possible pairs, plus the minimum. -/
lemma friend_pairs_bound (n m : ℕ) (h1 : n = 2000) (h2 : m = 1000) :
  (min_friends_pairs n m : ℤ) ≤ (n * m : ℤ) - (n * (n - 1) / 2 : ℤ) + (min_friends_pairs n m : ℤ) := by
  -- Proof sketch (to be completed)
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_friends_pairs_2000_1000_l928_92870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l928_92854

-- Define the arithmetic sequences and their partial sums
noncomputable def a : ℕ → ℝ := sorry
noncomputable def b : ℕ → ℝ := sorry
noncomputable def S : ℕ → ℝ := sorry
noncomputable def T : ℕ → ℝ := sorry

-- Define the property of partial sums
axiom partial_sum_property : ∀ n : ℕ, S n / T n = (2 * n + 1) / (4 * n - 2)

-- Define the property of arithmetic sequences
axiom arithmetic_sequence_property : 
  ∀ n : ℕ, b 3 + b 18 = b 6 + b 15 ∧ b 3 + b 18 = b 10 + b 11

-- State the theorem
theorem arithmetic_sequence_sum : 
  a 10 / (b 3 + b 18) + a 11 / (b 6 + b 15) = 41 / 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l928_92854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_product_l928_92822

/-- The distance from the center to the foci of the ellipse -/
def c_ellipse : ℝ := 5

/-- The distance from the center to the foci of the hyperbola -/
def c_hyperbola : ℝ := 8

/-- The semi-major axis of the ellipse -/
noncomputable def a_ellipse : ℝ := Real.sqrt 19.5

/-- The semi-minor axis of the ellipse -/
noncomputable def b_ellipse : ℝ := Real.sqrt 44.5

/-- The semi-major axis of the hyperbola -/
noncomputable def a_hyperbola : ℝ := Real.sqrt 19.5

/-- The semi-minor axis of the hyperbola -/
noncomputable def b_hyperbola : ℝ := Real.sqrt 44.5

theorem conic_sections_product :
  c_ellipse ^ 2 = b_ellipse ^ 2 - a_ellipse ^ 2 ∧
  c_hyperbola ^ 2 = a_hyperbola ^ 2 + b_hyperbola ^ 2 ∧
  |a_ellipse * b_ellipse| = Real.sqrt 868.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_product_l928_92822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separate_implies_obtuse_triangle_l928_92878

/-- Given a line ax-by+c=0 (abc≠0) and a circle x²+y²=1 that are separate,
    and |a|+|b| > |c|, the triangle with side lengths |a|, |b|, and |c| is obtuse. -/
theorem line_circle_separate_implies_obtuse_triangle 
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hsep : ∀ (x y : ℝ), a*x - b*y + c ≠ 0 ∨ x^2 + y^2 ≠ 1) 
  (htri : |a| + |b| > |c|) : 
  |a|^2 + |b|^2 < |c|^2 := by
  sorry

#check line_circle_separate_implies_obtuse_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separate_implies_obtuse_triangle_l928_92878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_less_than_u_l928_92856

-- Define variables
variable (y u w z : ℝ)

-- Define the conditions
def u_less_than_y (u y : ℝ) : Prop := u = 0.6 * y
def z_less_than_y (z y : ℝ) : Prop := z = 0.54 * y
def z_greater_than_w (z w : ℝ) : Prop := z = 1.5 * w

-- Define the theorem
theorem w_less_than_u 
  (h1 : u_less_than_y u y)
  (h2 : z_less_than_y z y)
  (h3 : z_greater_than_w z w) :
  w = 0.6 * u := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_less_than_u_l928_92856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_covered_l928_92828

noncomputable def walk_rate_1 : ℝ := 3
noncomputable def walk_time_1 : ℝ := 30 / 60
noncomputable def run_rate : ℝ := 8
noncomputable def run_time : ℝ := 20 / 60
noncomputable def walk_rate_2 : ℝ := 2
noncomputable def walk_time_2 : ℝ := 10 / 60

theorem total_distance_covered : 
  walk_rate_1 * walk_time_1 + run_rate * run_time + walk_rate_2 * walk_time_2 = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_covered_l928_92828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_speed_theorem_l928_92821

/-- The speed of a wheel in miles per hour -/
noncomputable def speed : ℝ → ℝ := λ r => r

/-- The time for one complete rotation of the wheel in hours -/
noncomputable def rotation_time (r : ℝ) : ℝ := 15 / (5280 * r)

/-- The condition that reducing rotation time by 1/3 second increases speed by 4 mph -/
def speed_increase_condition (r : ℝ) : Prop :=
  (r + 4) * (rotation_time r - 1 / (3 * 3600)) = 15 / 5280

theorem wheel_speed_theorem (r : ℝ) :
  speed_increase_condition r → r = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_speed_theorem_l928_92821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillation_period_proof_l928_92865

/-- The period of small oscillations for a system with given mass and spring stiffnesses -/
noncomputable def period_of_oscillation (m k₁ k₂ : ℝ) : ℝ :=
  2 * Real.pi / Real.sqrt ((k₁ + 4 * k₂) / (9 * m))

/-- Theorem stating the period of oscillation for the given system -/
theorem oscillation_period_proof (m k₁ k₂ : ℝ) 
  (h_m : m = 1.6)
  (h_k₁ : k₁ = 10)
  (h_k₂ : k₂ = 7.5) :
  period_of_oscillation m k₁ k₂ = 6 * Real.pi / 5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval period_of_oscillation 1.6 10 7.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillation_period_proof_l928_92865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instructor_lessons_l928_92819

/-- Represents the number of shirts the instructor has -/
def shirts : ℕ := sorry

/-- Represents the number of pairs of trousers the instructor has -/
def trousers : ℕ := sorry

/-- Represents the number of pairs of shoes the instructor has -/
def shoes : ℕ := sorry

/-- The number of jackets the instructor has -/
def jackets : ℕ := 2

/-- The number of additional lessons possible with one more shirt -/
def additional_lessons_shirt : ℕ := 18

/-- The number of additional lessons possible with one more pair of trousers -/
def additional_lessons_trousers : ℕ := 63

/-- The number of additional lessons possible with one more pair of shoes -/
def additional_lessons_shoes : ℕ := 42

/-- The maximum number of lessons the instructor can conduct -/
def max_lessons : ℕ := 126

theorem instructor_lessons :
  (3 * shirts * trousers = additional_lessons_shirt) ∧
  (3 * shirts * shoes = additional_lessons_trousers) ∧
  (3 * trousers * shoes = additional_lessons_shoes) →
  3 * shirts * trousers * shoes = max_lessons := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instructor_lessons_l928_92819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_trig_identity_l928_92873

/-- Given vectors a and b where a is parallel to b, prove that 2 * sin α * cos α = -4/5 -/
theorem parallel_vectors_trig_identity (α : ℝ) :
  let a : Fin 2 → ℝ := ![Real.cos α, -2]
  let b : Fin 2 → ℝ := ![Real.sin α, 1]
  (∃ (k : ℝ), a = k • b) →
  2 * Real.sin α * Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_trig_identity_l928_92873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l928_92893

-- Define the hyperbola
noncomputable def hyperbola (k : ℝ) (x y : ℝ) : Prop := k * x^2 - y^2 = 1

-- Define the line
noncomputable def line (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Define the asymptote of the hyperbola
noncomputable def asymptote (a : ℝ) (x y : ℝ) : Prop := y = (1/a) * x

-- Define perpendicularity condition
noncomputable def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Define eccentricity
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity (k : ℝ) :
  (∃ a : ℝ, perpendicular (1/a) (-2) ∧
    (∀ x y : ℝ, hyperbola k x y ↔ hyperbola (1/a^2) x y)) →
  (∃ e : ℝ, e = Real.sqrt 5 / 2 ∧
    e = eccentricity 2 (Real.sqrt (2^2 + 1^2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l928_92893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_k_value_l928_92867

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given two non-zero vectors that are not collinear, if ke₁ + 2e₂ is collinear with 3e₁ + ke₂,
    then k = ± √6 -/
theorem collinear_vectors_k_value (e₁ e₂ : V) (k : ℝ) 
  (h₁ : e₁ ≠ 0)
  (h₂ : e₂ ≠ 0)
  (h₃ : ¬ ∃ (c : ℝ), e₁ = c • e₂)
  (h₄ : ∃ (l : ℝ), k • e₁ + 2 • e₂ = l • (3 • e₁ + k • e₂)) :
  k = Real.sqrt 6 ∨ k = -Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_k_value_l928_92867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_given_l928_92829

theorem fish_given (initial : ℝ) (final : ℕ) (h : initial = 49) (h' : final = 67) :
  final - Int.floor initial = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_given_l928_92829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l928_92838

/-- The line equation -/
def line (x y : ℝ) : Prop := x - y + 4 = 0

/-- The circle equations -/
def circle_eq (x y θ : ℝ) : Prop := x = 1 + 2 * Real.cos θ ∧ y = 1 + 2 * Real.sin θ

/-- The distance function from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y + 4| / Real.sqrt 2

/-- The main theorem stating the minimum distance -/
theorem min_distance_circle_to_line : 
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 - 2 ∧ 
  ∀ (x y θ : ℝ), circle_eq x y θ → distance_to_line x y ≥ d :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l928_92838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_max_ratio_l928_92802

/-- Represents a participant's scores in a two-day tournament --/
structure TournamentScore where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ

/-- Calculates the success ratio for a given day --/
def daily_ratio (score : ℕ) (total : ℕ) : ℚ :=
  if total = 0 then 0 else ↑score / ↑total

/-- Calculates the overall success ratio --/
def overall_ratio (ts : TournamentScore) : ℚ :=
  ↑(ts.day1_score + ts.day2_score) / ↑(ts.day1_total + ts.day2_total)

theorem gamma_max_ratio
  (alpha : TournamentScore)
  (gamma : TournamentScore)
  (h1 : alpha.day1_score = 210 ∧ alpha.day1_total = 350)
  (h2 : alpha.day2_score = 150 ∧ alpha.day2_total = 250)
  (h3 : gamma.day1_total + gamma.day2_total = 600)
  (h4 : gamma.day1_total ≠ 350)
  (h5 : gamma.day1_score > 0 ∧ gamma.day2_score > 0)
  (h6 : daily_ratio gamma.day1_score gamma.day1_total < daily_ratio alpha.day1_score alpha.day1_total)
  (h7 : daily_ratio gamma.day2_score gamma.day2_total < daily_ratio alpha.day2_score alpha.day2_total)
  (h8 : overall_ratio alpha = 3/5)
  : overall_ratio gamma ≤ 359/600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_max_ratio_l928_92802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l928_92809

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line with slope k and y-intercept m
def line (k m x y : ℝ) : Prop := y = k*x + m

-- Define tangency condition for C₁
def tangent_C₁ (k m : ℝ) : Prop := 2*k^2 - m^2 + 1 = 0

-- Define tangency condition for C₂
def tangent_C₂ (k m : ℝ) : Prop := k*m = 1

theorem tangent_line_equation :
  ∀ k m : ℝ,
  tangent_C₁ k m → tangent_C₂ k m →
  (k = Real.sqrt 2/2 ∧ m = Real.sqrt 2) ∨ (k = -Real.sqrt 2/2 ∧ m = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l928_92809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_symmetric_f_l928_92825

/-- A function f(x) that is symmetric about x = -2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- The property of f being symmetric about x = -2 -/
def isSymmetricAboutNegativeTwo (f : ℝ → ℝ) : Prop :=
  ∀ h : ℝ, f (-2 - h) = f (-2 + h)

/-- Theorem: If f is symmetric about x = -2, its maximum value is 16 -/
theorem max_value_of_symmetric_f (a b : ℝ) 
    (h : isSymmetricAboutNegativeTwo (f a b)) : 
    ∃ M, M = 16 ∧ ∀ x, f a b x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_symmetric_f_l928_92825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_squared_l928_92848

noncomputable def radius : ℝ := 8

noncomputable def central_angle : ℝ := Real.pi / 2

theorem longest_chord_squared (r : ℝ) (θ : ℝ) (h1 : r = radius) (h2 : θ = central_angle) :
  2 * r^2 * (1 - Real.cos θ) = 128 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_squared_l928_92848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_specific_ellipse_l928_92804

/-- The eccentricity of an ellipse defined by parametric equations -/
noncomputable def eccentricity_of_ellipse (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b / a) ^ 2)

/-- Theorem: The eccentricity of the ellipse defined by x = 3cos(φ) and y = 5sin(φ) is 4/5 -/
theorem eccentricity_of_specific_ellipse :
  eccentricity_of_ellipse 5 3 = 4 / 5 := by
  sorry

#check eccentricity_of_specific_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_specific_ellipse_l928_92804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_value_l928_92850

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
variable (h1 : A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1)
variable (h2 : (B.1 - A.1) = (D.1 - C.1))
variable (h3 : C.1 - B.1 = 10)
variable (h4 : E.2 ≠ 0)
variable (h5 : ((B.1 - E.1)^2 + (B.2 - E.2)^2) = 12^2)
variable (h6 : ((C.1 - E.1)^2 + (C.2 - E.2)^2) = 12^2)
variable (h7 : ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 12^2)

-- Define the perimeter relationship
noncomputable def perimeter_AED : ℝ := Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2) + 
                         Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) + 
                         (D.1 - A.1)

def perimeter_BEC : ℝ := 36

variable (h8 : perimeter_AED = 1.5 * perimeter_BEC)

-- The theorem to prove
theorem AB_value : B.1 - A.1 = 157 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_value_l928_92850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l928_92843

-- Define the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

-- Define the line l passing through point A(1,0) with slope k
def lineL (k x y : ℝ) : Prop := y = k*(x - 1)

-- Define the condition that the line intersects the circle
def intersects (k : ℝ) : Prop := ∃ x y : ℝ, circleC x y ∧ lineL k x y

-- Theorem statement
theorem slope_range :
  ∀ k : ℝ, intersects k → -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l928_92843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l928_92807

theorem sum_remainder_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 11) 
  (hb : b % 15 = 13) 
  (hc : c % 15 = 14) : 
  (a + b + c) % 15 = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l928_92807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_fourth_power_l928_92872

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 4 * (x^4 - 2*x^5) + 3 * (x^2 - x^4 - 2*x^6) - (5*x^5 - 2*x^4)

/-- The coefficient of x^4 in the simplified expression is 3 -/
theorem coefficient_of_x_fourth_power : 
  ∃ f : ℝ → ℝ, ∀ x : ℝ, expression x = 3 * x^4 + f x ∧ (λ y : ℝ ↦ y^4) ∘ f = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_fourth_power_l928_92872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l928_92890

/-- Represents the probability of ending on a vertical edge from a given point -/
noncomputable def P (x y : ℕ) : ℚ := sorry

/-- The rectangular region -/
def rectangle : Set (ℕ × ℕ) :=
  {p | p.1 ≤ 5 ∧ p.2 ≤ 5}

/-- The boundary of the rectangle -/
def boundary : Set (ℕ × ℕ) :=
  {p ∈ rectangle | p.1 = 0 ∨ p.1 = 5 ∨ p.2 = 0 ∨ p.2 = 5}

/-- The vertical edges of the rectangle -/
def verticalEdge : Set (ℕ × ℕ) :=
  {p ∈ boundary | p.1 = 0 ∨ p.1 = 5}

/-- Jumping rules -/
axiom jump_rules (x y : ℕ) :
  (x, y) ∉ boundary →
  (x ≠ y → P x y = (P (x-1) y + P (x+1) y + P x (y-1) + P x (y+1)) / 4) ∧
  (x = y → P x y = (2 * P (x-1) y + 2 * P (x+1) y + P x (y-1) + P x (y+1)) / 6)

/-- Boundary conditions -/
axiom boundary_conditions :
  (∀ y, 0 ≤ y ∧ y ≤ 5 → P 0 y = 1) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 5 → P 5 y = 1) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 5 → P x 0 = 0) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 5 → P x 5 = 0)

/-- The main theorem to prove -/
theorem frog_jump_probability :
  P 2 3 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l928_92890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_sin_cos_relation_l928_92871

theorem tan_value_given_sin_cos_relation (x : ℝ) :
  Real.sin x - 2 * Real.cos x = Real.sqrt 5 → Real.tan x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_sin_cos_relation_l928_92871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mystery_compound_has_one_hydrogen_l928_92837

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Atomic weights of elements in atomic mass units (amu) -/
def atomic_weight : Compound → ℚ
  | ⟨c, h, o⟩ => 12 * c + 1 * h + 16 * o

/-- The compound in question -/
def mystery_compound : Compound := ⟨4, 1, 1⟩

/-- Theorem stating that the mystery compound has 1 hydrogen atom -/
theorem mystery_compound_has_one_hydrogen :
  atomic_weight mystery_compound = 65 → mystery_compound.hydrogen = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mystery_compound_has_one_hydrogen_l928_92837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l928_92814

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 3)

-- State the theorem
theorem phase_shift_of_f :
  ∃ (shift : ℝ), shift = Real.pi / 6 ∧
  ∀ (x : ℝ), f (x + shift) = 2 * Real.cos (2 * x) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l928_92814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygonal_domain_rearrangement_l928_92875

-- Define a polygonal domain in the Euclidean plane
structure PolygonalDomain where
  vertices : Set (ℝ × ℝ)
  is_polygonal : Prop  -- Changed from 'sorry' to 'Prop'

-- Define the concept of superposition through translations and rotations
def can_superpose (d1 d2 : PolygonalDomain) : Prop := 
  ∃ (t : ℝ × ℝ) (θ : ℝ), sorry  -- Placeholder for actual condition

-- Define the concept of splitting a domain into subdomains
def can_split_into (d : PolygonalDomain) (subdomains : Set PolygonalDomain) : Prop :=
  sorry  -- Placeholder for actual condition

-- Define the concept of rearranging subdomains
def can_rearrange_to (subdomains : Set PolygonalDomain) (target : PolygonalDomain) : Prop :=
  sorry  -- Placeholder for actual condition

-- The main theorem
theorem polygonal_domain_rearrangement 
  (d1 d2 : PolygonalDomain) 
  (h_identical : d1 = d2)  -- Condition that d1 and d2 are identical
  (h_not_superpose : ¬ can_superpose d1 d2) :
  ∃ (subdomains : Set PolygonalDomain),
    can_split_into d1 subdomains ∧ 
    can_rearrange_to subdomains d2 := by
  sorry  -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygonal_domain_rearrangement_l928_92875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_to_cylinder_volume_ratio_l928_92803

open Real

-- Define the parameters
noncomputable def radius : ℝ := 8
noncomputable def cylinder_height : ℝ := 24
noncomputable def cone_height : ℝ := 0.75 * cylinder_height

-- Define the volumes
noncomputable def cylinder_volume : ℝ := Real.pi * radius^2 * cylinder_height
noncomputable def cone_volume : ℝ := (1/3) * Real.pi * radius^2 * cone_height

-- The theorem to prove
theorem cone_to_cylinder_volume_ratio :
  cone_volume / cylinder_volume = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_to_cylinder_volume_ratio_l928_92803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_expression_x_squared_coefficient_l928_92886

noncomputable def nested_expression (k : ℕ) (x : ℝ) : ℝ :=
  match k with
  | 0 => x
  | n + 1 => ((nested_expression n x - 2)^2 - 2)

noncomputable def coefficient_x_squared (k : ℕ) : ℝ :=
  4^(k-1) * (4^k - 1) / 3

theorem nested_expression_x_squared_coefficient (k : ℕ) :
  ∃ (P : ℝ → ℝ), (∀ x, nested_expression k x = P x) ∧
  (∃ (a b c : ℝ) (Q : ℝ → ℝ), P = (λ x ↦ a + b*x + c*x^2 + Q x) ∧ c = coefficient_x_squared k) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_expression_x_squared_coefficient_l928_92886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_a_faster_company_b_cheaper_l928_92839

-- Define the companies
inductive Company
| A
| B

-- Define the parameters
def total_work : ℚ := 1 -- Representing 100% of the work
def joint_time : ℚ := 6 -- 6 weeks when working together
def joint_cost : ℚ := 52000 -- 52,000 yuan when working together
def a_solo_time : ℚ := 4 -- 4 weeks of A working alone
def total_split_time : ℚ := 13 -- 13 weeks total when A works 4 weeks then B finishes
def total_split_cost : ℚ := 48000 -- 48,000 yuan when A works 4 weeks then B finishes

-- Define work rates and cost rates
noncomputable def work_rate (c : Company) : ℚ := 
  match c with
  | Company.A => (total_work / joint_time - (total_work - a_solo_time / joint_time) / (total_split_time - a_solo_time))
  | Company.B => (total_work - a_solo_time / joint_time) / (total_split_time - a_solo_time)

noncomputable def cost_rate (c : Company) : ℚ :=
  match c with
  | Company.A => (joint_cost * a_solo_time / joint_time) / a_solo_time
  | Company.B => (total_split_cost - joint_cost * a_solo_time / joint_time) / (total_split_time - a_solo_time)

-- Theorem statements
theorem company_a_faster : work_rate Company.A > work_rate Company.B := by sorry

theorem company_b_cheaper : cost_rate Company.B < cost_rate Company.A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_a_faster_company_b_cheaper_l928_92839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_l928_92818

noncomputable section

/-- The parabola in the xy-plane with vertex at the origin and focus at (1,0) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- A point on the parabola -/
def PointOnParabola (M : ℝ × ℝ) : Prop :=
  M ∈ Parabola

/-- The line perpendicular to OM at O -/
def PerpendicularLine (M : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -(M.1 / M.2) * p.1}

/-- The line through M parallel to the y-axis -/
def ParallelLine (M : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = M.1}

/-- The intersection point of the perpendicular and parallel lines -/
def IntersectionPoint (M : ℝ × ℝ) : ℝ × ℝ :=
  (M.1, -(M.1 / M.2) * M.1)

/-- The theorem stating the geometric locus of intersection points -/
theorem intersection_locus (M : ℝ × ℝ) (h : PointOnParabola M) (h' : M ≠ (0, 0)) :
  (IntersectionPoint M).1 = -4 ∧ IntersectionPoint M ≠ (-4, 0) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_l928_92818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_length_example_l928_92853

/-- The average length of two strings -/
noncomputable def average_length (length1 : ℝ) (length2 : ℝ) : ℝ :=
  (length1 + length2) / 2

/-- Theorem: The average length of two strings with lengths 3.2 inches and 4.8 inches is 4.0 inches -/
theorem average_length_example : average_length 3.2 4.8 = 4.0 := by
  -- Unfold the definition of average_length
  unfold average_length
  -- Simplify the arithmetic
  simp [add_div]
  -- The rest of the proof (which we'll skip for now)
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_length_example_l928_92853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducibility_criterion_l928_92855

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- The sum of absolute values of coefficients -/
def sumAbsCoeffs {n : ℕ} (f : IntPolynomial n) : ℤ :=
  (Finset.univ : Finset (Fin n)).sum fun i => |f i|

/-- Irreducibility of a polynomial over integers -/
def isIrreducible {n : ℕ} (f : IntPolynomial n) : Prop :=
  ∀ g h : IntPolynomial n, (∀ i, f i = g i * h i) → 
    (∀ i, g i = 0 ∨ ∀ i, h i = 0)

theorem irreducibility_criterion 
  {n : ℕ} 
  (f : IntPolynomial (n + 1)) 
  (p : ℕ) 
  (hp : Nat.Prime p) 
  (h_leading : f (Fin.last n) ≠ 0)
  (h_constant : f 0 = p)
  (h_sum : sumAbsCoeffs (fun i => f (Fin.castSucc i)) < p) :
  isIrreducible f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducibility_criterion_l928_92855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_a_coordinates_l928_92869

-- Define the vector type
def MyVector := ℝ × ℝ

-- Define the magnitude of a vector
noncomputable def magnitude (v : MyVector) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define parallel vectors
def parallel (v w : MyVector) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem vector_a_coordinates :
  ∀ (a : MyVector),
    magnitude a = Real.sqrt 5 →
    parallel a (1, 2) →
    (a = (1, 2) ∨ a = (-1, -2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_a_coordinates_l928_92869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactlyOneWhiteAndTwoWhiteMutuallyExclusiveButNotComplementary_l928_92852

-- Define the bag of balls
def Bag := Finset (Fin 4)

-- Define the color of balls
inductive Color
| Red
| White
deriving DecidableEq

-- Define the coloring function
def color : Fin 4 → Color
| 0 => Color.Red
| 1 => Color.Red
| 2 => Color.White
| 3 => Color.White

-- Define the event of drawing exactly one white ball
def exactlyOneWhite (draw : Bag) : Prop :=
  (draw.filter (λ b => color b = Color.White)).card = 1

-- Define the event of drawing exactly two white balls
def exactlyTwoWhite (draw : Bag) : Prop :=
  (draw.filter (λ b => color b = Color.White)).card = 2

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : Bag → Prop) : Prop :=
  ∀ draw : Bag, ¬(e1 draw ∧ e2 draw)

-- Define complementarity
def complementary (e1 e2 : Bag → Prop) : Prop :=
  ∀ draw : Bag, e1 draw ∨ e2 draw

-- Theorem statement
theorem exactlyOneWhiteAndTwoWhiteMutuallyExclusiveButNotComplementary :
  mutuallyExclusive exactlyOneWhite exactlyTwoWhite ∧
  ¬complementary exactlyOneWhite exactlyTwoWhite := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactlyOneWhiteAndTwoWhiteMutuallyExclusiveButNotComplementary_l928_92852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l928_92862

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x - 1)

-- Define the domain of f
def domain (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 1

-- State that f is an odd function
axiom f_odd (a : ℝ) : ∀ x, domain x → f a (-x) = -(f a x)

-- Define the range of f
def range_f (y : ℝ) : Prop := 
  (y ≥ -3/2 ∧ y < -1/2) ∨ (y > 1/2 ∧ y ≤ 3/2)

-- Theorem stating the range of f
theorem f_range (a : ℝ) : 
  (∀ x, domain x → range_f (f a x)) ∧ 
  (∀ y, range_f y → ∃ x, domain x ∧ f a x = y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l928_92862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l928_92892

-- Define the hyperbola
def hyperbola (a b x y : ℝ) := y^2 / a^2 - x^2 / b^2 = 1

-- Define the circle (renamed to avoid conflict)
def hyperbola_circle (a c x y : ℝ) := x^2 + y^2 - (2*c/3)*y + a^2/9 = 0

-- Define the theorem
theorem hyperbola_asymptotes 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (M : ℝ × ℝ) 
  (hM : hyperbola a b M.1 M.2 ∧ M.2 < 0) 
  (D : ℝ × ℝ) 
  (hD : hyperbola_circle a c D.1 D.2) 
  (hTangent : ∃ (t : ℝ), D = (t * M.1, t * M.2 + (1 - t) * c)) 
  (hRatio : (M.1^2 + (M.2 - c)^2) = 9 * (D.1^2 + (D.2 - c)^2)) :
  ∃ (k : ℝ), k * M.1 = 2 * M.2 ∨ k * M.1 = -2 * M.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l928_92892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_minus_c_is_negative_l928_92847

/-- For all integers n > 1, a_n is defined as 1 / (log_n 1024) -/
noncomputable def a (n : ℕ) : ℝ := 1 / (Real.log 1024 / Real.log n)

/-- b is defined as the sum of a_3, a_4, a_5, and a_6 -/
noncomputable def b : ℝ := a 3 + a 4 + a 5 + a 6

/-- c is defined as the sum of a_15, a_16, a_17, a_18, and a_19 -/
noncomputable def c : ℝ := a 15 + a 16 + a 17 + a 18 + a 19

/-- Theorem stating that b - c is negative -/
theorem b_minus_c_is_negative : b - c < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_minus_c_is_negative_l928_92847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_monthly_savings_l928_92808

-- Define the savings per month for Thomas and Joseph
def thomas_savings : ℚ := 0
def joseph_savings : ℚ := 0

-- Define the relationship between Thomas and Joseph's savings
axiom savings_relation : joseph_savings = (3/5) * thomas_savings

-- Define the total savings period in months
def savings_period : ℕ := 6 * 12

-- Define the total savings amount
def total_savings : ℚ := 4608

-- Theorem stating the problem to be proved
theorem thomas_monthly_savings : 
  thomas_savings * savings_period + joseph_savings * savings_period = total_savings →
  thomas_savings = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_monthly_savings_l928_92808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l928_92858

open Real

theorem trig_identity (α β : ℝ) :
  sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β →
  tan (α - β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l928_92858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_water_usage_for_min_bill_l928_92887

/-- Represents the water usage and billing system --/
structure WaterBilling where
  usage : ℝ
  baseRate : ℝ
  excessRate : ℝ
  baseCutoff : ℝ
  minBill : ℝ

/-- Calculates the water bill based on usage and rates --/
noncomputable def calculateBill (wb : WaterBilling) : ℝ :=
  if wb.usage ≤ wb.baseCutoff then
    wb.usage * wb.baseRate
  else
    wb.baseCutoff * wb.baseRate + (wb.usage - wb.baseCutoff) * wb.excessRate

/-- Theorem stating the minimum water usage that results in a bill of at least 29 yuan --/
theorem min_water_usage_for_min_bill (wb : WaterBilling) 
  (h1 : wb.baseRate = 2.8)
  (h2 : wb.excessRate = 3)
  (h3 : wb.baseCutoff = 5)
  (h4 : wb.minBill = 29)
  : (∀ x, calculateBill { wb with usage := x } ≥ wb.minBill → x ≥ 10) ∧ 
    calculateBill { wb with usage := 10 } ≥ wb.minBill := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_water_usage_for_min_bill_l928_92887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l928_92815

theorem expression_evaluation : 
  (0.002 : ℝ)^(-(1/2 : ℝ)) - 10 * (Real.sqrt 5 - 2)^(-(1 : ℝ)) + (Real.sqrt 2 - Real.sqrt 3)^(0 : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l928_92815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_is_fifty_l928_92806

/-- The distance from Sky Falls to the city in miles -/
noncomputable def sky_falls_distance : ℝ := 8

/-- The distance from Rocky Mist Mountains to the city in miles -/
noncomputable def rocky_mist_distance : ℝ := 400

/-- The ratio of the distance from Rocky Mist Mountains to the city
    to the distance from Sky Falls to the city -/
noncomputable def distance_ratio : ℝ := rocky_mist_distance / sky_falls_distance

theorem distance_ratio_is_fifty :
  distance_ratio = 50 := by
  unfold distance_ratio rocky_mist_distance sky_falls_distance
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_is_fifty_l928_92806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_locus_definition_l928_92835

-- Define the space we're working in (e.g., a 2D plane)
variable {P : Type} [NormedAddCommGroup P] [InnerProductSpace ℝ P] [FiniteDimensional ℝ P]

-- Define a point from which we measure distances
variable (center : P)

-- Define the locus as a set of points
def locus (center : P) : Set P :=
  {p : P | ∃ r : ℝ, ∀ q : P, ‖q - center‖ = r ↔ q = p}

-- Define the geometric condition (equidistant from the center)
def satisfies_condition (p : P) (center : P) : Prop :=
  ∃ r : ℝ, ‖p - center‖ = r

-- The incorrect statement
theorem incorrect_locus_definition (center : P) :
  ¬(∀ p : P, p ∉ locus center → ¬satisfies_condition p center) ∧
  ¬(∀ p : P, satisfies_condition p center → p ∈ locus center) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_locus_definition_l928_92835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f₁_lower_bound_f₂_l928_92846

-- Part I
def f₁ (x : ℝ) := |x - 1| + |x + 2|

theorem solution_set_f₁ : 
  {x : ℝ | f₁ x ≤ 5} = Set.Icc (-3) 2 := by sorry

-- Part II
def f₂ (a b x : ℝ) := |x - a| + |x + b|

theorem lower_bound_f₂ (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 4*b = 2*a*b) :
  ∀ x, f₂ a b x ≥ 9/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f₁_lower_bound_f₂_l928_92846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_rule_approximation_l928_92840

-- Define the function to be integrated
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- Define the interval
def a : ℝ := 2
def b : ℝ := 12

-- Define the number of subintervals
def n : ℕ := 10

-- Define the width of each subinterval
noncomputable def Δx : ℝ := (b - a) / n

-- Define the trapezoidal rule approximation
noncomputable def trapezoidalRule (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : ℝ :=
  let Δx := (b - a) / n
  Δx * ((f a + f b) / 2 + (Finset.range (n - 1)).sum (λ i => f (a + (i + 1) * Δx)))

-- Define the exact integral value
noncomputable def exactIntegral : ℝ := Real.log (b / a)

-- State the theorem
theorem trapezoidal_rule_approximation :
  abs (trapezoidalRule f a b n - exactIntegral) / exactIntegral < 0.011 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_rule_approximation_l928_92840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l928_92830

/-- Compound interest calculation function -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- Theorem stating the relationship between initial investment and final amount -/
theorem investment_problem (P : ℝ) :
  let r := 0.0396  -- 3.96% annual rate
  let n := 2       -- compounded semi-annually
  let t := 2       -- 2 years
  let A := 10815.83  -- final amount
  abs (compound_interest P r n t - A) < 0.01 ↔ abs (P - 10000) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l928_92830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisible_number_l928_92898

theorem unique_divisible_number : ∃! n : ℕ,
  (∃ x y z : ℕ, x < 10 ∧ y < 10 ∧ z < 10 ∧
    n = 13000000 + x * 10000 + y * 1000 + 450 + z) ∧
  792 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisible_number_l928_92898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l928_92857

-- Define the parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the vertex of a parabola
def has_vertex (p : Parabola) (x y : ℝ) : Prop :=
  ∀ t, p.equation t 0 ↔ t = x

-- Define the axis of symmetry of a parabola
def has_axis_of_symmetry (p : Parabola) (x : ℝ) : Prop :=
  ∀ y t, p.equation (x - t) y ↔ p.equation (x + t) y

-- Theorem statement
theorem parabola_equation (p : Parabola) :
  has_vertex p 0 0 →
  has_axis_of_symmetry p (-2) →
  (∀ x y, p.equation x y ↔ y^2 = 8*x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l928_92857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l928_92877

-- Define the set T
def T : Set (ℝ × ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 ∧ p.1 + p.2.1 + p.2.2 = 1}

-- Define the supports relation
def supports (x y z a b c : ℝ) : Prop :=
  (x ≥ a ∧ y ≥ b) ∨ (x ≥ a ∧ z ≥ c) ∨ (y ≥ b ∧ z ≥ c)

-- Define the set S
def S : Set (ℝ × ℝ × ℝ) :=
  {p ∈ T | supports p.1 p.2.1 p.2.2 (1/4) (1/2) (1/4) ∨
           supports p.1 p.2.1 p.2.2 (1/3) (1/4) (2/5)}

-- Define the area function (assuming it exists)
noncomputable def area : Set (ℝ × ℝ × ℝ) → ℝ := sorry

-- State the theorem
theorem area_ratio : area S / area T = 1/16 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l928_92877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_A_l928_92894

theorem min_value_A (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  let A := (Real.sqrt (3 * x^4 + y) + Real.sqrt (3 * y^4 + z) + Real.sqrt (3 * z^4 + x) - 3) / (x * y + y * z + z * x)
  A ≥ 1 ∧ ∃ x y z, x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ A = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_A_l928_92894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_min_distance_l928_92845

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem ellipse_min_distance (p : ℝ) :
  ∃ (min_dist : ℝ) (mx my : ℝ),
    ellipse mx my ∧
    (∀ x y, ellipse x y → distance mx my p 0 ≤ distance x y p 0) ∧
    min_dist = distance mx my p 0 ∧
    ((abs p ≤ 1 ∧ min_dist = Real.sqrt (2 - p^2) ∧ mx = 2*p ∧ my^2 = 2 - 2*p^2) ∨
     (p > 1 ∧ min_dist = abs (p - 2) ∧ mx = 2 ∧ my = 0) ∨
     (p < -1 ∧ min_dist = abs (p + 2) ∧ mx = -2 ∧ my = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_min_distance_l928_92845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l928_92891

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 - 2*x + 8)

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  ∃ (a b : ℝ), a = -1 ∧ b = 2 ∧
  (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b → f x₁ > f x₂) ∧
  (∀ c d, c < a ∨ b < d → ¬(∀ x₁ x₂, c < x₁ ∧ x₁ < x₂ ∧ x₂ < d → f x₁ > f x₂)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l928_92891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_is_zero_l928_92824

def points : List (ℚ × ℚ) := [(4, 15), (7, 25), (13, 38), (19, 45), (21, 52)]

def is_above_line (point : ℚ × ℚ) : Bool :=
  point.2 > 3 * point.1 + 5

def sum_x_above_line (points : List (ℚ × ℚ)) : ℚ :=
  (points.filter is_above_line).map Prod.fst |>.sum

theorem sum_x_above_line_is_zero : sum_x_above_line points = 0 := by
  -- Unfold the definition of sum_x_above_line
  unfold sum_x_above_line
  -- Simplify the filter operation
  simp [points, is_above_line]
  -- The filter results in an empty list, so the sum is 0
  rfl

#eval sum_x_above_line points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_is_zero_l928_92824
