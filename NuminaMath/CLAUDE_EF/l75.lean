import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_in_valid_configuration_l75_7517

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Color of a point -/
inductive Color where
  | Red
  | Green

/-- A configuration of colored points in the plane -/
structure Configuration where
  points : Finset Point
  colors : Point → Color

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if a point is inside a triangle -/
def inside_triangle (p q r s : Point) : Prop := sorry

/-- Predicate to check if a configuration is valid -/
def valid_configuration (config : Configuration) : Prop :=
  (∀ p q r, p ∈ config.points → q ∈ config.points → r ∈ config.points →
    p ≠ q → q ≠ r → p ≠ r → ¬collinear p q r) ∧
  (∀ p q r, p ∈ config.points → q ∈ config.points → r ∈ config.points →
    config.colors p = config.colors q → config.colors q = config.colors r →
    ∃ s, s ∈ config.points ∧ config.colors s ≠ config.colors p ∧ inside_triangle p q r s)

/-- Theorem stating the maximum number of points in a valid configuration -/
theorem max_points_in_valid_configuration :
  (¬∃ (config : Configuration), valid_configuration config ∧ config.points.card = 9) ∧
  (∃ (config : Configuration), valid_configuration config ∧ config.points.card = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_in_valid_configuration_l75_7517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_six_equals_43_over_16_l75_7551

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9

noncomputable def g (x : ℝ) : ℝ := 
  let y := (x + 9) / 4  -- This is f⁻¹(x)
  3 * y^2 + 4 * y - 2

-- Theorem to prove
theorem g_of_neg_six_equals_43_over_16 : g (-6) = 43 / 16 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_six_equals_43_over_16_l75_7551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_planes_l75_7587

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of four points in 3D space -/
def FourPoints := Fin 4 → Point3D

/-- Predicate to check if three points are collinear -/
def areCollinear (p q r : Point3D) : Prop := sorry

/-- Predicate to check if four points are coplanar -/
def areCoplanar (p q r s : Point3D) : Prop := sorry

/-- The number of distinct planes formed by four points -/
noncomputable def numPlanes (points : FourPoints) : ℕ := sorry

/-- Theorem stating that given four non-collinear points, 
    the number of distinct planes they form is either 1 or 4 -/
theorem four_points_planes (points : FourPoints) 
  (h : ∀ (i j k : Fin 4), i ≠ j → j ≠ k → i ≠ k → ¬areCollinear (points i) (points j) (points k)) :
  numPlanes points = 1 ∨ numPlanes points = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_planes_l75_7587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grazing_area_is_quarter_circle_example_field_grazing_area_l75_7573

/-- Represents a rectangular field with a horse tethered to one corner. -/
structure GrazingField where
  length : ℝ
  width : ℝ
  rope_length : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  rope_shorter_than_dimensions : rope_length < length ∧ rope_length < width

/-- Calculates the area over which the horse can graze. -/
noncomputable def grazing_area (field : GrazingField) : ℝ :=
  (1 / 4) * Real.pi * field.rope_length ^ 2

/-- Theorem stating that the grazing area is a quarter circle with radius equal to the rope length. -/
theorem grazing_area_is_quarter_circle (field : GrazingField) :
  grazing_area field = (1 / 4) * Real.pi * field.rope_length ^ 2 :=
by rfl

/-- Example field with given dimensions -/
def example_field : GrazingField where
  length := 45
  width := 25
  rope_length := 22
  length_pos := by norm_num
  width_pos := by norm_num
  rope_shorter_than_dimensions := by norm_num

-- Remove the #eval statement as it's not computable
theorem example_field_grazing_area :
  grazing_area example_field = (1 / 4) * Real.pi * 22 ^ 2 :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grazing_area_is_quarter_circle_example_field_grazing_area_l75_7573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petri_dish_count_l75_7572

def total_germs : ℝ := 0.036 * 10^5
def germs_per_dish : ℝ := 99.99999999999999

def rounded_germs_per_dish : ℕ := 100

theorem petri_dish_count : 
  (Int.floor (total_germs / rounded_germs_per_dish : ℝ)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petri_dish_count_l75_7572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_b_n_relation_l75_7544

open BigOperators Finset Real

/-- For a positive integer n, a_n is defined as the sum of 1 / (n choose k) for k from 0 to n. -/
noncomputable def a_n (n : ℕ+) : ℝ := ∑ k in range (n + 1), 1 / (n.val.choose k)

/-- For a positive integer n, b_n is defined as the sum of k^2 / (n choose k) for k from 0 to n. -/
noncomputable def b_n (n : ℕ+) : ℝ := ∑ k in range (n + 1), k^2 / (n.val.choose k)

/-- For a positive integer n, c_n is defined as the sum of (k - n/2)^2 / (n choose k) for k from 0 to n. -/
noncomputable def c_n (n : ℕ+) : ℝ := ∑ k in range (n + 1), (k - n.val / 2)^2 / (n.val.choose k)

/-- The main theorem stating the relationship between a_n, b_n, and c_n. -/
theorem a_n_b_n_relation (n : ℕ+) : 
  a_n n / b_n n = 4 / (n.val^2 + 4 * c_n n / a_n n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_b_n_relation_l75_7544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_ln_and_cube_root_l75_7550

theorem derivative_ln_and_cube_root (x : ℝ) (h : x > 0) :
  (deriv (λ y => Real.log (3 * y + 1))) x = 3 / (3 * x + 1) ∧
  (deriv (λ y => 1 / Real.rpow y (1/3))) x = -1 / (3 * Real.rpow x (4/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_ln_and_cube_root_l75_7550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_not_basis_l75_7539

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the basis vectors
variable (a b c : V)

-- Hypothesis that {a, b, c} form a basis
variable (h : LinearIndependent ℝ ![a, b, c])

-- Define the sets of vectors
def set_A (a b c : V) : Fin 3 → V
  | 0 => b + c
  | 1 => b
  | 2 => b - c

def set_C (a b c : V) : Fin 3 → V
  | 0 => a
  | 1 => a + b
  | 2 => a - b

def set_D (a b c : V) : Fin 3 → V
  | 0 => a + b
  | 1 => a + b + c
  | 2 => c

-- Theorem statement
theorem sets_not_basis (h : LinearIndependent ℝ ![a, b, c]) :
  ¬(LinearIndependent ℝ (set_A a b c)) ∧
  ¬(LinearIndependent ℝ (set_C a b c)) ∧
  ¬(LinearIndependent ℝ (set_D a b c)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_not_basis_l75_7539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_range_l75_7516

-- Part 1
def f₁ (x : ℝ) := 4 * x^2 - 6 * x + 3

theorem f_expression (f : ℝ → ℝ) :
  (∀ x, f (x + 1) = 4 * x^2 + 2 * x + 1) →
  (∀ x, f x = f₁ x) :=
sorry

-- Part 2
def f₂ (x : ℝ) := -x^2 + x - 2

theorem f_range (f : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (∀ x, f (x + 2) - 2 * f x = x^2 - 5 * x) →
  Set.range f = Set.Iic (-7/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_range_l75_7516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l75_7523

def a (x : ℝ) : ℝ × ℝ := (x, 2)
def b : ℝ × ℝ := (2, -1)

theorem vector_magnitude_problem (x : ℝ) 
  (h : a x • b = 0) : 
  ‖a x - b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l75_7523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_C2_l75_7505

/-- Curve C1 in parametric form -/
noncomputable def C1 (α : ℝ) : ℝ × ℝ :=
  (2 * Real.sqrt 2 * Real.cos α, 2 * Real.sin α)

/-- Curve C2 in polar form -/
def C2 (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ - Real.sqrt 2 * ρ * Real.sin θ - 5 = 0

/-- The minimum distance between a point on C1 and a point on C2 -/
theorem min_distance_C1_C2 :
    ∃ (d : ℝ), d = Real.sqrt 3 / 3 ∧
    ∀ (α θ ρ : ℝ), C2 ρ θ →
    ∀ (P Q : ℝ × ℝ), P = C1 α → Q.1 = ρ * Real.cos θ ∧ Q.2 = ρ * Real.sin θ →
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_C2_l75_7505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_l75_7538

theorem sum_of_powers (x : ℝ) (h : (4 : ℝ)^x + (4 : ℝ)^(-x) = 23) : (2 : ℝ)^x + (2 : ℝ)^(-x) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_l75_7538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_sqrt_two_l75_7518

noncomputable def max_distance_to_line : ℝ := 
  sSup {d : ℝ | ∃ (x y : ℝ), y = x + 1 ∧ d = Real.sqrt ((x - 0)^2 + (y + 1)^2)}

theorem max_distance_is_sqrt_two : max_distance_to_line = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_sqrt_two_l75_7518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_median_and_bisector_l75_7567

-- Define a right-angled triangle ABC with ∠C = 90° and tan(α/2) = 1/∛³√3
def triangle_ABC (α : Real) : Prop :=
  0 < α ∧ α < Real.pi/2 ∧ Real.tan (α/2) = 1 / Real.rpow 3 (1/4)

-- Define θ as the angle between the median and angle bisector from A to BC
noncomputable def angle_θ (α : Real) (θ : Real) : Prop :=
  triangle_ABC α → 
  Real.tan θ = (1/2) * ((Real.tan (2*α) - Real.tan α) / (1 + Real.tan (2*α) * Real.tan α))

-- The theorem to be proved
theorem angle_between_median_and_bisector (α θ : Real) :
  triangle_ABC α → angle_θ α θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_median_and_bisector_l75_7567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l75_7519

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - 2*x) + (x - 1)^0 + 1 / (x + 1)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < -1 ∨ (-1 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2)}

-- Define the alternative function g(x) = f(x+1)
def g (x : ℝ) : ℝ := x^2 - 2*x

-- Theorem statement
theorem function_properties :
  (∀ x ∈ domain_f, f x = Real.sqrt (4 - 2*x) + (x - 1)^0 + 1 / (x + 1)) ∧
  (∀ x, g x = x^2 - 2*x) ∧
  (∀ x, f (x + 1) = x^2 - 2*x) ∧
  (∀ x, f x = x^2 - 4*x + 3) ∧
  (f 3 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l75_7519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BRS_l75_7529

-- Define the point B
def B : ℝ × ℝ := (4, 10)

-- Define the y-intercepts R and S as functions of y₁ and y₂
def R (y₁ : ℝ) : ℝ × ℝ := (0, y₁)
def S (y₂ : ℝ) : ℝ × ℝ := (0, y₂)

-- Define area_triangle function (this should be defined in Mathlib, but we'll define it here for completeness)
noncomputable def area_triangle (p q r : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_BRS :
  ∀ m₁ m₂ y₁ y₂ : ℝ, 
  -- Two lines are perpendicular
  m₁ * m₂ = -1 →
  -- The lines pass through B
  (10 = m₁ * 4 + y₁) →
  (10 = m₂ * 4 + y₂) →
  -- The sum of y-coordinates of R and S is zero
  y₁ + y₂ = 0 →
  -- The area of triangle BRS is 8√29
  area_triangle B (R y₁) (S y₂) = 8 * Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BRS_l75_7529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_B_is_algorithm_l75_7586

/-- Definition of an algorithm --/
def is_algorithm (s : String) : Prop :=
  ∃ (clear_instructions : String → Bool) 
    (solves_problem : String → Bool) 
    (finite_steps : String → Bool),
  clear_instructions s ∧ solves_problem s ∧ finite_steps s

/-- Statement A --/
def statement_A : String :=
  "Divide the students of Class 5, Grade 1 into two groups, with taller students participating in a basketball game and shorter students in a tug-of-war."

/-- Statement B --/
def statement_B : String :=
  "Divide the students of Class 5, Grade 1 into two groups, with students taller than 170cm participating in a basketball game and those shorter than 170cm in a tug-of-war."

/-- Statement C --/
def statement_C : String :=
  "Cooking must have rice."

/-- Statement D --/
def statement_D : String :=
  "Starting from 2, write the next number as the sum of the previous number and 2, continuously write to list all even numbers."

/-- Theorem: Statement B is an algorithm --/
theorem statement_B_is_algorithm :
  is_algorithm statement_B ∧
  ¬is_algorithm statement_A ∧
  ¬is_algorithm statement_C ∧
  ¬is_algorithm statement_D := by
  sorry

#check statement_B_is_algorithm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_B_is_algorithm_l75_7586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_billing_theorem_l75_7508

/-- Water billing system -/
noncomputable def WaterBilling (a : ℝ) : ℝ → ℝ :=
  fun n =>
    if n ≤ 12 then n * a
    else if n ≤ 20 then 12 * a + (n - 12) * 1.5 * a
    else 12 * a + 8 * 1.5 * a + (n - 20) * 2 * a

theorem water_billing_theorem (a : ℝ) :
  (WaterBilling a 18 = 21 * a) ∧
  (WaterBilling a 26 = 36 * a) ∧
  (∀ n : ℝ, n > 20 → WaterBilling a n = (2 * n - 16) * a) ∧
  (WaterBilling 1.5 28 = 60) := by
  sorry

#check water_billing_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_billing_theorem_l75_7508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_of_composite_function_l75_7510

theorem fixed_points_of_composite_function 
  {a b : ℝ} (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) 
  (h_map : Set.MapsTo f (Set.Icc a b) (Set.Icc a b))
  (h_alpha : ∃ α, α ∈ Set.Ioo a b ∧ f α = a) 
  (h_beta : ∃ β, β ∈ Set.Ioo a b ∧ f β = b) :
  ∃ x y z, x ∈ Set.Icc a b ∧ y ∈ Set.Icc a b ∧ z ∈ Set.Icc a b ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (f ∘ f) x = x ∧ (f ∘ f) y = y ∧ (f ∘ f) z = z :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_of_composite_function_l75_7510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_y_axis_intersection_l75_7588

/-- Represents a parabola in the form y = 1/2 * (x + h)^2 + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
noncomputable def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  1/2 * (x + p.h)^2 + p.k

theorem parabola_y_axis_intersection
  (p : Parabola)
  (vertex_x : p.y_coord 2 = 8) :
  p.y_coord 0 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_y_axis_intersection_l75_7588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_scores_l75_7569

/-- Represents the number of baskets made for each point value -/
structure BasketCounts where
  two_points : ℕ
  three_points : ℕ
  four_points : ℕ

/-- Calculates the total score for a given set of basket counts -/
def total_score (bc : BasketCounts) : ℕ :=
  2 * bc.two_points + 3 * bc.three_points + 4 * bc.four_points

/-- Generates all valid combinations of basket counts -/
def valid_combinations : List BasketCounts :=
  List.foldr (λ x acc =>
    List.foldr (λ y inner_acc =>
      let z := 7 - x - y
      if z ≥ 0 then {two_points := x, three_points := y, four_points := z} :: inner_acc
      else inner_acc
    ) acc (List.range 8)
  ) [] (List.range 8)

theorem basketball_scores :
  (valid_combinations.map total_score).toFinset.card = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_scores_l75_7569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_equation_l75_7500

-- Define the * operation
noncomputable def star (a b : ℝ) : ℝ :=
  if a ≥ b then a^2 + b^2 else a^2 - b^2

-- Theorem statement
theorem solution_of_equation (x : ℝ) :
  star x 2 = 12 ↔ x = 2 ∨ x = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_equation_l75_7500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spy_kidnap_theorem_l75_7554

/-- Represents a gentleman in the city -/
structure Gentleman where
  id : Nat

/-- Represents a club in the city -/
structure Club where
  id : Nat
  members : Finset Gentleman

/-- The city of London with its clubs and gentlemen -/
structure London where
  clubs : Finset Club
  gentlemen : Finset Gentleman
  club_count : clubs.card = 10^10
  club_size : ∀ c, c ∈ clubs → c.members.card = 10
  common_member : ∀ c1 c2, c1 ∈ clubs → c2 ∈ clubs → c1 ≠ c2 → 
    ∃ g, g ∈ gentlemen ∧ g ∈ c1.members ∧ g ∈ c2.members

theorem spy_kidnap_theorem (london : London) :
  ∃ kidnapped : Finset Gentleman,
    kidnapped.card = 9 ∧
    (∀ c, c ∈ london.clubs → ∃ g, g ∈ kidnapped ∧ g ∈ c.members) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spy_kidnap_theorem_l75_7554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_circle_parabola_l75_7598

/-- The circle with equation x^2 + y^2 - 4x - 6y + 9 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p | (p.1^2 + p.2^2 - 4*p.1 - 6*p.2 + 9) = 0}

/-- The parabola with equation y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 8*p.1}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The smallest possible distance between a point on the Circle and a point on the Parabola -/
noncomputable def smallestDistance : ℝ := Real.sqrt (49/64)

theorem smallest_distance_circle_parabola :
  ∀ (A : ℝ × ℝ) (B : ℝ × ℝ),
    A ∈ Circle → B ∈ Parabola →
    distance A B ≥ smallestDistance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_circle_parabola_l75_7598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l75_7597

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 6*x + y^2 - 8*y + 18 = 0

/-- The shortest distance from the origin to the circle -/
noncomputable def shortest_distance : ℝ := 5 - Real.sqrt 7

theorem shortest_distance_to_circle :
  ∃ (p : ℝ × ℝ), circle_equation p.1 p.2 ∧
  ∀ (q : ℝ × ℝ), circle_equation q.1 q.2 →
  Real.sqrt (p.1^2 + p.2^2) ≤ Real.sqrt (q.1^2 + q.2^2) ∧
  Real.sqrt (p.1^2 + p.2^2) = shortest_distance :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l75_7597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l75_7566

variable (a b : ℝ × ℝ)

-- Define the angle between vectors a and b
noncomputable def angle_between (a b : ℝ × ℝ) : ℝ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem vector_sum_magnitude
  (h1 : angle_between a b = 2 * Real.pi / 3)
  (h2 : a = (1/2, Real.sqrt 3 / 2))
  (h3 : magnitude b = 2) :
  magnitude (2 • a + 3 • b) = 2 * Real.sqrt 7 := by
  sorry

#check vector_sum_magnitude

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l75_7566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_factorial_problem_l75_7562

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_factorial_problem : 
  Nat.gcd (factorial 5) ((factorial 10) / (factorial 4)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_factorial_problem_l75_7562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_calculation_l75_7541

noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

theorem trapezium_area_calculation :
  let a : ℝ := 20
  let b : ℝ := 18
  let h : ℝ := 12
  trapezium_area a b h = 228 :=
by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_calculation_l75_7541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_difference_l75_7595

/-- Represents a cyclist with their speeds on different terrains -/
structure Cyclist where
  flat_speed : ℚ
  downhill_speed : ℚ
  uphill_speed : ℚ

/-- Represents a segment of the route -/
structure RouteSegment where
  distance : ℚ
  terrain : String

/-- Calculates the time taken for a cyclist to complete a route segment -/
noncomputable def time_for_segment (c : Cyclist) (s : RouteSegment) : ℚ :=
  match s.terrain with
  | "flat" => s.distance / c.flat_speed
  | "downhill" => s.distance / c.downhill_speed
  | "uphill" => s.distance / c.uphill_speed
  | _ => 0  -- Default case, should not occur in this problem

/-- Calculates the total time for a cyclist to complete the entire route -/
noncomputable def total_time (c : Cyclist) (route : List RouteSegment) : ℚ :=
  route.foldr (fun segment acc => acc + time_for_segment c segment) 0

/-- The main theorem statement -/
theorem journey_time_difference :
  let minnie : Cyclist := { flat_speed := 25, downhill_speed := 35, uphill_speed := 10 }
  let penny : Cyclist := { flat_speed := 35, downhill_speed := 45, uphill_speed := 15 }
  let route : List RouteSegment := [
    { distance := 15, terrain := "uphill" },
    { distance := 20, terrain := "downhill" },
    { distance := 25, terrain := "flat" }
  ]
  (total_time minnie route - total_time penny route) * 60 = 87 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_difference_l75_7595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_product_squares_l75_7520

/-- Given a geometric progression with n terms, first term a, and common ratio r,
    prove that the product of the squares of these n terms (P) is equal to (S · S')^(n/2),
    where S is the sum of the terms and S' is the sum of their reciprocals. -/
theorem geometric_progression_product_squares (n : ℕ) (a r : ℝ) (hr : r ≠ 0) (hr1 : r ≠ 1) :
  let S := a * (1 - r^n) / (1 - r)
  let S' := (1 / a) * (1 - (1/r)^n) / (1 - 1/r)
  let P := (a^2) * (r^2)^((n * (n - 1)) / 2)
  P = (S * S')^(n / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_product_squares_l75_7520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l75_7570

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 3 * Real.sqrt 3 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the distance function from a point to the line
noncomputable def dist_to_line (x y : ℝ) : ℝ := 
  (|Real.sqrt 3 * x - y + 3 * Real.sqrt 3|) / 2

-- Theorem statement
theorem distance_range : 
  ∀ x y : ℝ, circle_C x y → 
  (5 * Real.sqrt 3) / 2 - 1 ≤ dist_to_line x y ∧ dist_to_line x y ≤ (5 * Real.sqrt 3) / 2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l75_7570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_theorem_l75_7579

def is_valid_arrangement (n : ℕ) (arr : List ℕ) : Prop :=
  (arr.length = n) ∧ 
  (∀ i ∈ arr, 1 ≤ i ∧ i ≤ n) ∧
  (∀ i, i < arr.length → arr[i]! ∣ (arr[(i+1) % arr.length]! + arr[(i-1+arr.length) % arr.length]!))

def are_equivalent_arrangements (n : ℕ) (arr1 arr2 : List ℕ) : Prop :=
  ∃ k, (List.rotateLeft arr1 k = arr2) ∨ (List.reverse (List.rotateLeft arr1 k) = arr2)

theorem circle_arrangement_theorem (n : ℕ) (h_odd : Odd n) (h_gt_10 : n > 10) :
  ∃ (arr1 arr2 : List ℕ), 
    is_valid_arrangement n arr1 ∧
    is_valid_arrangement n arr2 ∧
    ¬(are_equivalent_arrangements n arr1 arr2) ∧
    (∀ arr, is_valid_arrangement n arr → 
      (are_equivalent_arrangements n arr arr1) ∨ (are_equivalent_arrangements n arr arr2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_theorem_l75_7579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l75_7561

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

theorem symmetry_axis_of_f :
  ∀ (k : ℤ), (∀ (x : ℝ), f x = f ((Real.pi / 6 + ↑k * Real.pi) - x)) ↔ 
  (∀ (x : ℝ), f x = f ((Real.pi / 6 + ↑k * Real.pi / 2) - (x - (Real.pi / 12 + ↑k * Real.pi / 2)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l75_7561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_min_a_value_l75_7540

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - 3 * x^2 - 11 * x

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (m b : ℝ), m = -15 ∧ b = 1 ∧
  ∀ x, f x = m * (x - 1) + f 1 := by sorry

-- Theorem for the minimum value of a
theorem min_a_value :
  ∃ (a : ℕ), a = 1 ∧
  (∀ x > 0, f x ≤ (a - 3) * x^2 + (2 * a - 13) * x - 2) ∧
  (∀ a' : ℕ, a' < a → ∃ x > 0, f x > (a' - 3) * x^2 + (2 * a' - 13) * x - 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_min_a_value_l75_7540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_theorem_l75_7528

/-- Investment problem -/
def investment_problem (total_investment : ℕ) 
  (company_a_investment company_a_share_price company_a_premium company_a_dividend : ℕ)
  (company_b_investment company_b_share_price company_b_premium company_b_dividend : ℕ)
  (company_c_share_price company_c_premium company_c_dividend : ℕ) : Prop :=
  let company_c_investment := total_investment - company_a_investment - company_b_investment
  let company_a_shares := company_a_investment / (company_a_share_price + company_a_premium * company_a_share_price / 100)
  let company_b_shares := company_b_investment / (company_b_share_price + company_b_premium * company_b_share_price / 100)
  let company_c_shares := company_c_investment / (company_c_share_price + company_c_premium * company_c_share_price / 100)
  let company_a_dividend_amount := company_a_shares * company_a_dividend * company_a_share_price / 100
  let company_b_dividend_amount := company_b_shares * company_b_dividend * company_b_share_price / 100
  let company_c_dividend_amount := company_c_shares * company_c_dividend * company_c_share_price / 100
  let total_dividend := company_a_dividend_amount + company_b_dividend_amount + company_c_dividend_amount
  total_dividend = 2096

theorem investment_theorem : 
  investment_problem 50000 14400 100 20 5 22000 50 10 3 200 5 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_theorem_l75_7528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_force_on_dam_value_l75_7521

/-- Calculates the force exerted by water on a trapezoidal dam section. -/
noncomputable def water_force_on_dam (ρ g a b h : ℝ) : ℝ :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

/-- The force exerted by water on a trapezoidal dam section with given dimensions and water properties is equal to 544000 N. -/
theorem water_force_on_dam_value :
  let ρ : ℝ := 1000  -- water density in kg/m³
  let g : ℝ := 10    -- acceleration due to gravity in m/s²
  let a : ℝ := 5.7   -- bottom width of trapezoid in m
  let b : ℝ := 9.0   -- top width of trapezoid in m
  let h : ℝ := 4.0   -- height of trapezoid in m
  water_force_on_dam ρ g a b h = 544000 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_force_on_dam_value_l75_7521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_is_two_l75_7501

-- Define the function f(x) = e^x + 4/e^x - 2
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 / Real.exp x - 2

-- Theorem statement
theorem min_value_of_f_is_two :
  ∀ x : ℝ, f x ≥ 2 ∧ ∃ y : ℝ, f y = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_is_two_l75_7501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l75_7584

theorem vector_magnitude (α β : ℝ × ℝ) : 
  (‖α‖ = 1) → 
  (‖β‖ = 2) → 
  (α • (α - 2 • β) = 0) → 
  ‖2 • α + β‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l75_7584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recycling_drive_sections_l75_7522

/-- Proves that the number of sections participating in the recycling drive is 3 --/
theorem recycling_drive_sections : 
  (let kilos_per_section_two_weeks : ℕ := 280
   let target_kilos : ℕ := 2000
   let remaining_kilos : ℕ := 320
   (target_kilos - remaining_kilos) / (kilos_per_section_two_weeks * 3 / 2) = 3) := by
  -- Introduce the local variables
  let kilos_per_section_two_weeks : ℕ := 280
  let target_kilos : ℕ := 2000
  let remaining_kilos : ℕ := 320
  
  -- Calculate the number of sections
  have sections : ℕ := (target_kilos - remaining_kilos) / (kilos_per_section_two_weeks * 3 / 2)
  
  -- Prove that the number of sections is 3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recycling_drive_sections_l75_7522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_evil_league_l75_7553

/-- The shortest distance from (5,1) to visit lines y = x and x = 7 and return -/
theorem shortest_distance_evil_league : ∃ d : ℝ, d = 4 * Real.sqrt 5 := by
  -- Define the starting point
  let start : ℝ × ℝ := (5, 1)
  
  -- Define the first pipe line
  let pipe1 : Set (ℝ × ℝ) := {p | p.2 = p.1}
  
  -- Define the second pipe line
  let pipe2 : Set (ℝ × ℝ) := {p | p.1 = 7}
  
  -- Define the path
  let path (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
    {x | ∃ t : ℝ, x = (1 - t) • p + t • q ∧ 0 ≤ t ∧ t ≤ 1}
  
  -- State the shortest distance
  let shortest_distance : ℝ := 4 * Real.sqrt 5

  -- Prove the theorem
  sorry

#check shortest_distance_evil_league

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_evil_league_l75_7553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l75_7513

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define vectors m and n
noncomputable def m (C : Real) : Real × Real := (Real.cos (C/2), Real.sin (C/2))
noncomputable def n (C : Real) : Real × Real := (Real.cos (C/2), -Real.sin (C/2))

-- Define the angle between two vectors
noncomputable def angle_between (v w : Real × Real) : Real :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

-- Define the area of a triangle
noncomputable def area (t : Triangle) : Real := 1/2 * t.a * t.b * Real.sin t.C

-- Main theorem
theorem triangle_properties (t : Triangle) :
  angle_between (m t.C) (n t.C) = π/3 ∧
  t.c = 7/2 ∧
  area t = 3/2 * Real.sqrt 3 →
  t.C = π/3 ∧ t.a + t.b = 11/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l75_7513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unwound_rope_length_l75_7581

/-- A point mass connected to a stationary cylinder by a rope on a frictionless surface. -/
structure RopeSystem where
  m : ℝ  -- Mass of the point object
  R : ℝ  -- Radius of the cylinder
  v₀ : ℝ  -- Initial velocity of the object (perpendicular to the rope)
  L₀ : ℝ  -- Initial length of the rope
  Tₘₐₓ : ℝ  -- Maximum tension before the rope breaks

/-- The length of the unwound rope when it breaks. -/
noncomputable def unwoundLength (s : RopeSystem) : ℝ := s.m * s.v₀^2 / (2 * s.Tₘₐₓ)

/-- 
Theorem: The length of the rope not yet wound when it breaks is equal to mv₀²/(2Tₘₐₓ).
-/
theorem unwound_rope_length (s : RopeSystem) 
    (h₁ : s.m > 0) 
    (h₂ : s.R > 0) 
    (h₃ : s.v₀ > 0) 
    (h₄ : s.L₀ > s.R) 
    (h₅ : s.Tₘₐₓ > 0) : 
  ∃ (L : ℝ), L = unwoundLength s ∧ L > 0 ∧ L < s.L₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unwound_rope_length_l75_7581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_2015_2016_occurrences_l75_7542

/-- Represents the circular number sequence -/
def CircularSequence := List Nat

/-- Initial sequence with 1 and 2 -/
def initial_sequence : CircularSequence := [1, 2]

/-- Operation that triples the sum of the sequence -/
def triple_sum_operation (seq : CircularSequence) : CircularSequence :=
  sorry

/-- Counts the occurrences of a number in the sequence -/
def count_occurrences (n : Nat) (seq : CircularSequence) : Nat :=
  sorry

/-- Theorem stating the sum of occurrences of 2015 and 2016 equals 2016 -/
theorem sum_of_2015_2016_occurrences (final_seq : CircularSequence) :
  (∃ n : Nat, (Nat.iterate triple_sum_operation n initial_sequence = final_seq)) →
  count_occurrences 2015 final_seq + count_occurrences 2016 final_seq = 2016 :=
by
  sorry

#check sum_of_2015_2016_occurrences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_2015_2016_occurrences_l75_7542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moment_of_inertia_equals_14k_l75_7590

-- Define the solid Ω
def Ω : Set (Fin 3 → ℝ) :=
  {p | 0 ≤ p 0 ∧ p 0 ≤ 2 ∧ 0 ≤ p 1 ∧ p 1 ≤ 1 ∧ 0 ≤ p 2 ∧ (p 2)^2 ≤ 6 * p 0}

-- Define the volume density function
def ρ (k : ℝ) (p : Fin 3 → ℝ) : ℝ := k * p 2

-- Define the moment of inertia function
noncomputable def momentOfInertia (k : ℝ) : ℝ :=
  ∫ p in Ω, ((p 0)^2 + (p 1)^2) * ρ k p

-- Theorem statement
theorem moment_of_inertia_equals_14k (k : ℝ) :
  momentOfInertia k = 14 * k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moment_of_inertia_equals_14k_l75_7590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commercial_reduction_l75_7564

/-- Calculates the number of seconds to cut from a commercial given its original length in minutes and the required percentage reduction. -/
noncomputable def seconds_to_cut (original_length : ℝ) (reduction_percentage : ℝ) : ℝ :=
  original_length * 60 * (reduction_percentage / 100)

/-- Proves that for a 0.5-minute commercial that needs to be shortened by 30%, the number of seconds to be cut is equal to 9. -/
theorem commercial_reduction : seconds_to_cut 0.5 30 = 9 := by
  -- Unfold the definition of seconds_to_cut
  unfold seconds_to_cut
  -- Simplify the arithmetic expression
  simp [mul_assoc, mul_comm, mul_div_assoc]
  -- The result should now be obvious to Lean
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_commercial_reduction_l75_7564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l75_7557

theorem sin_pi_plus_alpha (α : ℝ) (h1 : Real.cos (2 * Real.pi - α) = 2 * Real.sqrt 2 / 3)
  (h2 : α ∈ Set.Ioo (-Real.pi/2) 0) : Real.sin (Real.pi + α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l75_7557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_wins_l75_7560

/-- Represents the race between a rabbit and a snail --/
structure Race where
  distance : ℝ
  rabbit_initial_speed : ℝ
  rabbit_rest_time : ℝ
  rabbit_final_speed : ℝ
  snail_speed : ℝ

/-- Calculates the time taken by the rabbit to complete the race --/
noncomputable def rabbit_time (race : Race) : ℝ :=
  let initial_distance := race.rabbit_initial_speed * (race.rabbit_rest_time - 3)
  let remaining_distance := race.distance - initial_distance
  (race.rabbit_rest_time - 3) + 3 + (remaining_distance / race.rabbit_final_speed)

/-- Calculates the time taken by the snail to complete the race --/
noncomputable def snail_time (race : Race) : ℝ :=
  race.distance / race.snail_speed

/-- Theorem stating that the rabbit wins the race --/
theorem rabbit_wins (race : Race) 
  (h1 : race.distance = 100)
  (h2 : race.rabbit_initial_speed = 20)
  (h3 : race.rabbit_rest_time = 4.5)
  (h4 : race.rabbit_final_speed = 30)
  (h5 : race.snail_speed = 2) : 
  rabbit_time race < snail_time race := by
  sorry

#eval "Rabbit wins the race"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_wins_l75_7560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_minimum_l75_7583

/-- Triangle with sides a, b, c and area T -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  T : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_T : T > 0

/-- Point inside a triangle with perpendicular distances a₁, b₁, c₁ to the sides -/
structure PointInTriangle (tri : Triangle) where
  a₁ : ℝ
  b₁ : ℝ
  c₁ : ℝ
  pos_a₁ : a₁ > 0
  pos_b₁ : b₁ > 0
  pos_c₁ : c₁ > 0
  in_triangle : tri.T = (tri.a * a₁ + tri.b * b₁ + tri.c * c₁) / 2

/-- The expression to be minimized -/
noncomputable def expression (tri : Triangle) (p : PointInTriangle tri) : ℝ :=
  tri.a / p.a₁ + tri.b / p.b₁ + tri.c / p.c₁

/-- Definition of the incenter (equidistant from all sides) -/
def is_incenter (tri : Triangle) (p : PointInTriangle tri) : Prop :=
  p.a₁ = p.b₁ ∧ p.b₁ = p.c₁

theorem expression_minimum (tri : Triangle) :
  ∀ p : PointInTriangle tri,
    expression tri p ≥ (tri.a + tri.b + tri.c)^2 / (2 * tri.T) ∧
    (expression tri p = (tri.a + tri.b + tri.c)^2 / (2 * tri.T) ↔ is_incenter tri p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_minimum_l75_7583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_divisors_cube_l75_7594

/-- If a positive integer n has exactly 3 divisors, then n^3 has exactly 7 divisors. -/
theorem three_divisors_cube (n : ℕ+) (h : (Nat.divisors n.val).card = 3) : 
  (Nat.divisors (n^3).val).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_divisors_cube_l75_7594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_calculation_l75_7546

-- Define the operation *
noncomputable def star (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

-- Define a function to represent the nested operation
noncomputable def nestedStar : ℕ → ℝ
| 0 => 1001
| (n + 1) => star (n + 2) (nestedStar n)

-- Define y
noncomputable def y : ℝ := nestedStar 999

-- State the theorem
theorem nested_star_calculation (h : y ≠ 1/2) :
  star 2 y = (2 - y) / (1 - 2 * y) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_calculation_l75_7546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C2_to_l_l75_7545

/-- The curve C2 -/
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, 3 + 3 * Real.sin θ)

/-- The line l -/
def l (x y : ℝ) : Prop := x - 2 * y - 1 = 0

/-- The distance function from a point to the line l -/
noncomputable def distance_to_l (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  |x - 2 * y - 1| / Real.sqrt 5

/-- The theorem stating the maximum distance from C2 to l -/
theorem max_distance_C2_to_l :
  ∃ (θ : ℝ), ∀ (φ : ℝ), distance_to_l (C2 θ) ≥ distance_to_l (C2 φ) ∧
  distance_to_l (C2 θ) = 3 + 7 * Real.sqrt 5 / 5 := by
  sorry

#check max_distance_C2_to_l

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C2_to_l_l75_7545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l75_7576

theorem problem_statement (x y : ℤ) (h1 : x = 3) (h2 : y = -1) :
  x - (y : ℚ)^(x - 2*y) + 2*x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l75_7576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_calculation_l75_7534

-- Define the given number
def given_number : ℝ := 3.615

-- Define the multiplier
def multiplier : ℝ := 1.3333

-- Define the result (rounded to two decimal places)
def rounded_result : ℝ := 4.82

-- Define a small margin of error for rounding
def margin_of_error : ℝ := 0.005

-- Theorem statement
theorem value_calculation :
  ∃ (exact_result : ℝ), 
    exact_result = given_number * multiplier ∧ 
    |exact_result - rounded_result| < margin_of_error := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_calculation_l75_7534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_count_l75_7548

-- Define a convex pentagon
structure ConvexPentagon where
  vertices : Fin 5 → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

-- Define a point inside the pentagon
def PointInside (p : ConvexPentagon) := { point : ℝ × ℝ // point ∈ interior (Set.range p.vertices) }

-- Define a function that counts the number of intersecting lines
noncomputable def CountIntersectingLines (p : ConvexPentagon) (m : PointInside p) : ℕ :=
  sorry

-- Theorem statement
theorem intersecting_lines_count (p : ConvexPentagon) (m : PointInside p) :
  ∃ n : ℕ, n ∈ ({1, 3, 5} : Set ℕ) ∧ CountIntersectingLines p m = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_count_l75_7548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l75_7532

/-- The time it takes for the first pipe to fill the tank -/
noncomputable def T : ℝ := 16.8

/-- The time it takes for the second pipe to empty the tank -/
noncomputable def emptyTime : ℝ := 24

/-- The time after which the second pipe is closed when both pipes are open -/
noncomputable def closeTime : ℝ := 36

/-- The total time to fill the tank -/
noncomputable def totalTime : ℝ := 30

/-- The rate at which the first pipe fills the tank -/
noncomputable def fillRate : ℝ := 1 / T

/-- The rate at which the second pipe empties the tank -/
noncomputable def emptyRate : ℝ := 1 / emptyTime

theorem pipe_fill_time : 
  (fillRate - emptyRate) * closeTime + fillRate * (totalTime - closeTime) = 1 := by
  sorry

#check pipe_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l75_7532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equation_solution_l75_7503

noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_sum_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 
    (Finset.sum (Finset.range 10) (fun k => log_base (Real.rpow 10 (1 / (k + 1 : ℝ))) x)) = 5.5 ∧
    x = Real.rpow 10 (1 / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equation_solution_l75_7503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_graph_l75_7565

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define the transformed function
noncomputable def f (x : ℝ) : ℝ := (1/3) * g x - 2

-- Theorem statement
theorem transform_graph :
  ∀ x y : ℝ, y = g x ↔ (1/3) * y - 2 = f g x :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_graph_l75_7565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_problem_solution_l75_7575

/-- Represents a friend's guess about the number of apples --/
structure Guess where
  friend : Nat
  value : Nat

/-- Represents the actual number of apples and the guesses --/
structure AppleProblem where
  actual : Nat
  guesses : List Guess

/-- Calculates the absolute difference between two natural numbers --/
def absDiff (a b : Nat) : Nat :=
  if a ≥ b then a - b else b - a

/-- Checks if the guesses satisfy the error conditions --/
def validGuesses (p : AppleProblem) : Prop :=
  p.guesses.length = 3 ∧
  (∃ (g1 g2 g3 : Guess), 
    g1 ∈ p.guesses ∧ g2 ∈ p.guesses ∧ g3 ∈ p.guesses ∧
    g1 ≠ g2 ∧ g2 ≠ g3 ∧ g1 ≠ g3 ∧
    ({absDiff p.actual g1.value, absDiff p.actual g2.value, absDiff p.actual g3.value} : Finset Nat) = {1, 2, 3})

theorem apple_problem_solution :
  ∀ (p : AppleProblem),
    p.guesses = [Guess.mk 1 19, Guess.mk 2 22, Guess.mk 3 23] →
    validGuesses p →
    p.actual = 20 := by
  intro p guesses_eq valid_guesses
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_problem_solution_l75_7575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equations_equivalence_l75_7593

-- Define the parametric equations for C₁
noncomputable def C₁ (θ : Real) : Real × Real :=
  (1 + Real.cos θ, 1 + Real.sin θ)

-- Define the standard form equation for C₁
def C₁_standard (x y : Real) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 1

-- Define the polar equation for C₂
def C₂_polar (ρ : Real) : Prop :=
  ρ = 1

-- Define the Cartesian equation for C₂
def C₂_cartesian (x y : Real) : Prop :=
  x^2 + y^2 = 1

theorem curve_equations_equivalence :
  (∀ θ x y, C₁ θ = (x, y) → C₁_standard x y) ∧
  (∀ ρ θ, C₂_polar ρ → C₂_cartesian (ρ * Real.cos θ) (ρ * Real.sin θ)) := by
  sorry

#check curve_equations_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equations_equivalence_l75_7593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l75_7526

-- Define the line l: 3x + y - 6 = 0
def line (x y : ℝ) : Prop := 3 * x + y - 6 = 0

-- Define the circle C: x^2 + y^2 - 2y - 4 = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 4 = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | line p.1 p.2 ∧ circle_eq p.1 p.2}

-- Theorem statement
theorem distance_between_intersection_points :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧
  A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l75_7526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l75_7556

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of circles C, D, and E -/
structure CircleConfiguration where
  C : Circle
  D : Circle
  E : Circle

/-- The main theorem statement -/
theorem circle_tangency_theorem (config : CircleConfiguration) : 
  (config.C.radius = 6) →
  (config.D.radius = 4 * config.E.radius) →
  (∃ (p q : ℕ), config.D.radius = Real.sqrt (p : ℝ) - q) →
  (config.C.radius = config.D.radius + config.E.radius) →
  (config.C.radius = config.D.radius + 2 * config.E.radius) →
  (∃ (p q : ℕ), config.D.radius = Real.sqrt (p : ℝ) - q ∧ p + q = 598) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l75_7556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_third_implications_l75_7599

theorem tan_one_third_implications (α : ℝ) (h : Real.tan α = 1/3) :
  (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 ∧
  Real.cos α ^ 2 - Real.sin (2 * α) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_third_implications_l75_7599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_removing_thirteen_increases_probability_l75_7504

def T : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

def sumPairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
  S.product S |>.filter (fun p => p.1 + p.2 = 15 ∧ p.1 ≠ p.2)

def probability (S : Finset ℕ) : ℚ :=
  (sumPairs S).card / Nat.choose S.card 2

theorem removing_thirteen_increases_probability :
  probability (T.erase 13) > probability T :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_removing_thirteen_increases_probability_l75_7504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_covering_length_eq_three_sqrt_three_quarters_l75_7506

/-- The minimum length of line segments required to cover at least one point
    inside an equilateral triangle with side length 1 -/
noncomputable def min_covering_length : ℝ := (3 * Real.sqrt 3) / 4

/-- Theorem stating that the minimum length of line segments required to cover
    at least one point inside an equilateral triangle with side length 1
    is equal to 3√3/4 -/
theorem min_covering_length_eq_three_sqrt_three_quarters :
  min_covering_length = (3 * Real.sqrt 3) / 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_covering_length_eq_three_sqrt_three_quarters_l75_7506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_range_of_a_l75_7537

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 2) * Real.exp x + Real.log x + 1 / x

/-- The derivative of f(x) with respect to x -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a * (x - 1) * Real.exp x + 1 / x - 1 / (x^2)

/-- Predicate indicating that f has two extreme points in the interval (0, 2) -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ f' a x₁ = 0 ∧ f' a x₂ = 0

/-- The range of a -/
def range_of_a : Set ℝ := {x | x < -1 / Real.exp 1 ∨ (-1 / Real.exp 1 < x ∧ x < -1 / (4 * Real.exp 2))}

/-- Theorem stating the relationship between the existence of two extreme points and the range of a -/
theorem extreme_points_imply_range_of_a :
  ∀ a : ℝ, has_two_extreme_points a → a ∈ range_of_a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_range_of_a_l75_7537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_third_kilometer_time_speed_inverse_square_time_formula_correct_l75_7571

/-- The speed for the nth kilometer, where n ≥ 3 -/
noncomputable def speed (n : ℕ) : ℝ := 4 / (3 * (n - 1)^2)

/-- The time needed to traverse the nth kilometer, where n ≥ 3 -/
noncomputable def time (n : ℕ) : ℝ := 3 * (n - 1)^2 / 4

theorem car_travel_time (n : ℕ) (h : n ≥ 3) :
  time n = 1 / speed n := by
  sorry

theorem third_kilometer_time :
  time 3 = 3 := by
  sorry

theorem speed_inverse_square (n m : ℕ) (hn : n ≥ 3) (hm : m ≥ 3) :
  speed n / speed m = (m - 1)^2 / (n - 1)^2 := by
  sorry

theorem time_formula_correct (n : ℕ) (h : n ≥ 3) :
  time n = 3 * (n - 1)^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_third_kilometer_time_speed_inverse_square_time_formula_correct_l75_7571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_a_axis_of_symmetry_l75_7507

-- Define a continuous function f
variable (f : ℝ → ℝ)

-- Define a real number a
variable (a : ℝ)

-- Theorem 1: Symmetry about x = a
theorem symmetry_about_a :
  ∀ x y, f (x - a) = y ↔ f (a - x) = y :=
by sorry

-- Given condition for Theorem 2
axiom functional_equality : ∀ x, f (1 + 2*x) = f (1 - 2*x)

-- Theorem 2: Axis of symmetry is x = 1
theorem axis_of_symmetry :
  ∀ x, f x = f (2 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_a_axis_of_symmetry_l75_7507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_range_l75_7515

theorem x_squared_range (x : ℝ) (h : (x + 16)^(1/3) - (x - 16)^(1/3) = 4) :
  240 < x^2 ∧ x^2 < 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_range_l75_7515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_at_135_chord_equation_when_bisected_l75_7589

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define chord AB
def chord_AB (slope_AB : ℝ) (x y : ℝ) : Prop := 
  ∃ (t : ℝ), x = -1 + t ∧ y = 2 + t * slope_AB

-- Define the inclination angle and bisected condition
def bisected : Prop := 
  ∃ (x y : ℝ), chord_AB (1/2) x y ∧ my_circle x y ∧ 
    (x + 1)^2 + (y - 2)^2 = (-1 - x)^2 + (2 - y)^2

def inclination_angle (α : ℝ) : Prop := α = 135 ∨ bisected

-- Theorem 1
theorem chord_length_at_135 : 
  ∀ (x y : ℝ), my_circle x y → chord_AB (-1) x y → inclination_angle 135 → 
    ∃ (a b : ℝ), chord_AB (-1) a b ∧ (a - x)^2 + (b - y)^2 = 30 :=
sorry

-- Theorem 2
theorem chord_equation_when_bisected : 
  bisected → 
    ∃ (x y : ℝ), chord_AB (1/2) x y ∧ x - 2*y + 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_at_135_chord_equation_when_bisected_l75_7589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpicks_stage_15_l75_7580

/-- The number of toothpicks at a given stage of the pattern -/
def toothpicks : ℕ → ℕ
  | 0 => 3  -- We define 0 as the first stage
  | n + 1 => toothpicks n + if n % 2 = 0 then 3 else 2

/-- The 15th stage of the pattern has 38 toothpicks -/
theorem toothpicks_stage_15 : toothpicks 14 = 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpicks_stage_15_l75_7580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l75_7512

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1/x - 17/4) / Real.log (1/2)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := (x > 4) ∨ (0 < x ∧ x < 1/4)

-- Theorem statement
theorem f_monotonic_increasing_interval :
  ∀ x y : ℝ, domain x → domain y → (0 < x ∧ x < y ∧ y < 1/4 → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l75_7512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l75_7596

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) : Prop :=
  -- One internal angle is 120°
  ∃ (angle : ℝ), angle = 120 * Real.pi / 180 ∧
  -- Side lengths form an arithmetic sequence with common difference 2
  b = a + 2 ∧ c = b + 2 ∧
  -- Cosine rule for the angle of 120°
  Real.cos angle = (a^2 + b^2 - c^2) / (2 * a * b)

-- Theorem statement
theorem triangle_ABC_area :
  ∀ (a b c : ℝ), triangle_ABC a b c →
  (1/2 * a * b * Real.sin (120 * Real.pi / 180) = 15 * Real.sqrt 3 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l75_7596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_calculation_l75_7531

theorem gain_percent_calculation (marked_price : ℝ) (cost_price : ℝ) 
  (h1 : cost_price = 0.64 * marked_price) 
  (h2 : marked_price > 0) : 
  (((0.88 * marked_price - cost_price) / cost_price) * 100) = 37.5 := by
  have selling_price : ℝ := 0.88 * marked_price
  have profit : ℝ := selling_price - cost_price
  have gain_percent : ℝ := (profit / cost_price) * 100
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_calculation_l75_7531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_champion_is_chinese_l75_7552

-- Define the set of players
inductive Player : Type
| A : Player  -- Chinese player A
| B : Player  -- Chinese player B

-- Define the property of being a Chinese player
def isChinese : Player → Prop
| Player.A => True
| Player.B => True

-- Define the champion
variable (champion : Player)

-- Theorem stating that the champion is certainly a Chinese player
theorem champion_is_chinese : isChinese champion := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_champion_is_chinese_l75_7552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_minus_x_real_l75_7524

theorem sqrt_two_minus_x_real (x : ℝ) : 0 ≤ 2 - x ↔ x ≤ 2 := by
  apply Iff.intro
  · intro h
    linarith
  · intro h
    linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_minus_x_real_l75_7524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_integer_expression_l75_7514

noncomputable def is_integer (x : ℝ) : Prop := ∃ (n : ℤ), x = n

noncomputable def expression (n : ℕ) : ℝ := (Real.sqrt 7 + 2 * Real.sqrt (n : ℝ)) / (2 * Real.sqrt 7 - Real.sqrt (n : ℝ))

theorem largest_n_for_integer_expression :
  (∀ n : ℕ, n > 343 → ¬ is_integer (expression n)) ∧
  is_integer (expression 343) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_integer_expression_l75_7514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_f_omega_primitive_root_l75_7525

/-- Define ω as a complex number -/
noncomputable def ω : ℂ := Complex.exp (Complex.I * Real.pi / 5)

/-- The polynomial with roots ω, ω³, ω⁷, ω⁹ -/
def f (x : ℂ) : ℂ := x^4 - x^3 + x^2 - x + 1

/-- Theorem stating that f has roots ω, ω³, ω⁷, ω⁹ -/
theorem roots_of_f :
  f ω = 0 ∧ f (ω^3) = 0 ∧ f (ω^7) = 0 ∧ f (ω^9) = 0 := by
  sorry

/-- Theorem stating that ω is a primitive 10th root of unity -/
theorem omega_primitive_root : ω^10 = 1 ∧ ∀ k : ℕ, 0 < k → k < 10 → ω^k ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_f_omega_primitive_root_l75_7525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_theorem_l75_7536

/-- Represents a math exam with two papers -/
structure MathExam where
  paper1 : Finset ℕ
  paper2 : Finset ℕ
  total_questions : ℕ
  pupil_attempts : Finset (Finset ℕ)

/-- Conditions for the math exam -/
def ValidMathExam (exam : MathExam) : Prop :=
  exam.paper1.Nonempty ∧
  exam.paper2.Nonempty ∧
  exam.total_questions = 28 ∧
  exam.paper1 ∪ exam.paper2 = Finset.range exam.total_questions ∧
  (∀ p ∈ exam.pupil_attempts, p.card = 7) ∧
  (∀ q1 q2, q1 ∈ Finset.range exam.total_questions → q2 ∈ Finset.range exam.total_questions → q1 ≠ q2 →
    (exam.pupil_attempts.filter (λ p => q1 ∈ p ∧ q2 ∈ p)).card = 2)

/-- Main theorem to prove -/
theorem exam_theorem (exam : MathExam) (h : ValidMathExam exam) :
  ∃ p ∈ exam.pupil_attempts, (p ∩ exam.paper1).card = 0 ∨ (p ∩ exam.paper1).card ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_theorem_l75_7536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l75_7577

/-- Given two parallel lines in the form ax + by + c = 0, calculate their distance -/
noncomputable def distance_between_parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : ℝ :=
  Real.sqrt ((c1 - c2)^2 / (a1^2 + b1^2))

theorem parallel_lines_distance (x y : ℝ) :
  (2 * x + 3 * y - 9 = 0) →
  (6 * x + 9 * y + 12 = 0) →
  distance_between_parallel_lines 2 3 (-9) 6 9 12 = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l75_7577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_result_l75_7559

def operation (a b : ℕ) : ℕ := a * b + a + b

theorem board_game_result :
  (List.range 20).foldl operation 1 = Nat.factorial 21 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_result_l75_7559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_6_l75_7543

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ := 
  c / a

def hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

theorem hyperbola_eccentricity_sqrt_6 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (M N : ℝ × ℝ) (hM : M ∈ hyperbola a b) (hN : N ∈ hyperbola a b)
  (h_parallel : ∃ (k : ℝ), N.1 - M.1 = k * (2*c) ∧ N.2 - M.2 = 0)
  (h_distance : dist (-c, 0) (c, 0) = 4 * dist M N)
  (Q : ℝ × ℝ) (hQ : Q ∈ hyperbola a b)
  (h_intersect : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ Q = (1 - t) • (-c, 0) + t • N)
  (h_equal_dist : dist (-c, 0) Q = dist Q N) :
  hyperbola_eccentricity a b c h1 h2 = Real.sqrt 6 := by
  sorry

#check hyperbola_eccentricity_sqrt_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_6_l75_7543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_weights_l75_7582

/-- The weight of the chihuahua in pounds -/
def chihuahua_weight : ℝ := sorry

/-- The weight of the pitbull in pounds -/
def pitbull_weight : ℝ := sorry

/-- The weight of the great dane in pounds -/
def great_dane_weight : ℝ := sorry

/-- The combined weight of all three dogs in pounds -/
def combined_weight : ℝ := chihuahua_weight + pitbull_weight + great_dane_weight

theorem dog_weights :
  (pitbull_weight = 3 * chihuahua_weight) →
  (great_dane_weight = 3 * pitbull_weight + 10) →
  (great_dane_weight = 307) →
  (combined_weight = 439) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_weights_l75_7582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_dissection_estimate_l75_7574

/-- The number of vertices in a cube -/
def num_vertices : ℕ := 8

/-- The number of edges in a cube -/
def num_edges : ℕ := 12

/-- The total number of marked points on the cube -/
def total_marked_points : ℕ := num_vertices + num_edges

/-- The minimum number of points a plane must pass through -/
def min_points_per_plane : ℕ := 4

/-- Estimate of the number of pieces the cube is cut into -/
def estimated_pieces : ℕ := 15600

/-- Theorem stating the estimated number of pieces -/
theorem cube_dissection_estimate :
  ∃ (N : ℕ), N ≥ estimated_pieces ∧ N ≤ estimated_pieces + 100 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_dissection_estimate_l75_7574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l75_7511

def sequence_rule (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = (a n + 2028) / (1 + a (n + 1))

theorem min_sum_first_two_terms (a : ℕ → ℕ) :
  sequence_rule a → (∀ n, a n > 0) → ∃ a₁ a₂ : ℕ, 
    a 1 = a₁ ∧ a 2 = a₂ ∧ 
    (∀ b₁ b₂ : ℕ, b₁ > 0 → b₂ > 0 → sequence_rule (λ n ↦ if n = 1 then b₁ else if n = 2 then b₂ else a n) 
      → a₁ + a₂ ≤ b₁ + b₂) ∧
    a₁ + a₂ = 104 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l75_7511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_difference_in_first_quadrant_l75_7558

def Z1 : ℂ := 1 + Complex.I
def Z2 : ℂ := -2 - 3*Complex.I

theorem complex_difference_in_first_quadrant :
  let Z := Z1 - Z2
  (Z.re > 0) ∧ (Z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_difference_in_first_quadrant_l75_7558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_approximation_l75_7549

/-- The median of contest answers -/
noncomputable def M : ℝ := 8 + 8 * Real.rpow 3 (1/4)

/-- Theorem stating that M is approximately equal to 18.528592 -/
theorem median_approximation : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000001 ∧ |M - 18.528592| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_approximation_l75_7549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l75_7591

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (v w : V) : Prop := ∃ (k : ℝ), v = k • w ∨ w = k • v

theorem vector_collinearity 
  (e₁ e₂ : V) (lambda : ℝ) (a b : V)
  (h₁ : e₁ ≠ 0)
  (h₂ : a = e₁ + lambda • e₂)
  (h₃ : b = 2 • e₁)
  (h₄ : collinear a b) :
  (∃ (k : ℝ), e₁ = k • e₂) ∨ lambda = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l75_7591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l75_7509

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the shifted function
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

-- Theorem stating the equivalence of the graphs
theorem sin_shift_equivalence :
  ∀ x : ℝ, g x = f (x + Real.pi / 8) := by
  intro x
  simp [f, g]
  apply congr_arg Real.sin
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l75_7509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marcus_shoe_purchase_l75_7555

theorem marcus_shoe_purchase (original_price discount_percent amount_saved : ℚ) 
  (h1 : original_price = 120)
  (h2 : discount_percent = 30)
  (h3 : amount_saved = 46) : 
  (original_price - original_price * (discount_percent / 100)) + amount_saved = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marcus_shoe_purchase_l75_7555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roberto_downhill_speed_l75_7592

noncomputable def trail_length : ℝ := 5
noncomputable def uphill_percentage : ℝ := 0.6
noncomputable def downhill_percentage : ℝ := 0.4
noncomputable def uphill_speed : ℝ := 2
noncomputable def total_time_minutes : ℝ := 130

noncomputable def uphill_distance : ℝ := trail_length * uphill_percentage
noncomputable def downhill_distance : ℝ := trail_length * downhill_percentage

noncomputable def uphill_time_hours : ℝ := uphill_distance / uphill_speed
noncomputable def uphill_time_minutes : ℝ := uphill_time_hours * 60
noncomputable def downhill_time_minutes : ℝ := total_time_minutes - uphill_time_minutes
noncomputable def downhill_time_hours : ℝ := downhill_time_minutes / 60

noncomputable def downhill_speed : ℝ := downhill_distance / downhill_time_hours

theorem roberto_downhill_speed :
  downhill_speed = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roberto_downhill_speed_l75_7592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_exist_no_fixed_point_l75_7578

-- Define the functions
def f1 (x : ℝ) := x^2 - x - 3
noncomputable def f2 (x : ℝ) := Real.sqrt x + 1
noncomputable def f3 (x : ℝ) := (2 : ℝ)^x - 2
noncomputable def f4 (x : ℝ) := (2 : ℝ)^x + x

-- Theorem for functions with fixed points
theorem fixed_points_exist :
  (∃ x : ℝ, f1 x = x) ∧
  (∃ x : ℝ, x ≥ 0 → f2 x = x) ∧
  (∃ x : ℝ, f3 x = x) :=
by sorry

-- Theorem for function without fixed point
theorem no_fixed_point :
  ¬(∃ x : ℝ, f4 x = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_exist_no_fixed_point_l75_7578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_P_l75_7530

noncomputable def angle : ℝ := 2 * Real.pi / 3

def point_P (x y : ℝ) : Prop :=
  x = 2 * Real.cos angle ∧ y = 2 * Real.sin angle

theorem coordinates_of_P :
  point_P (-1) (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_P_l75_7530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_minimum_area_l75_7533

theorem circle_minimum_area (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 3 / (2 + x) + 3 / (2 + y) = 1) :
  ∃ (x₀ y₀ : ℝ), x₀ = 4 ∧ y₀ = 4 ∧
  ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 3 / (2 + x') + 3 / (2 + y') = 1 →
  x' * y' ≥ x₀ * y₀ := by
  sorry

#check circle_minimum_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_minimum_area_l75_7533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sequence_values_l75_7547

/-- The function g(x) = x^2 - 6x + 8 -/
def g (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- The sequence defined by x_n = g(x_{n-1}) -/
def seq (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => g (seq x₀ n)

/-- The set of values in the sequence -/
def seqValues (x₀ : ℝ) : Set ℝ :=
  {x | ∃ n : ℕ, seq x₀ n = x}

theorem infinite_sequence_values : ∀ x₀ : ℝ, Set.Infinite (seqValues x₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sequence_values_l75_7547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_and_max_value_l75_7585

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.cos x)
def c : ℝ × ℝ := (-1, 0)

noncomputable def f (x : ℝ) : ℝ := 2 * (a x).1 * (b x).1 + 2 * (a x).2 * (b x).2 + 1

noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem vector_angle_and_max_value :
  (angle (a (π/6)) c = 5*π/6) ∧
  (∀ x ∈ Set.Icc (π/2) (9*π/8), f x ≤ 1) ∧
  (f (π/2) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_and_max_value_l75_7585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l75_7535

/-- Proves that a train with given parameters has a speed of 60 km/h -/
theorem train_speed (train_length bridge_length time_to_cross : ℝ) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 200)
  (h3 : time_to_cross = 18.598512119030477) : ℝ :=
by
  -- Define the total distance covered
  let total_distance := train_length + bridge_length
  
  -- Calculate the speed in m/s
  let speed_ms := total_distance / time_to_cross
  
  -- Convert speed to km/h
  let speed_kmh := speed_ms * 3.6
  
  -- Prove that the speed is 60 km/h
  sorry

-- Example usage (commented out to avoid evaluation errors)
-- #eval train_speed 110 200 18.598512119030477

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l75_7535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l75_7563

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / (Real.sin x)^2 - 2 * Real.cos x - 3 * Real.log (Real.tan (x/2))

-- State the theorem
theorem derivative_of_f (x : ℝ) (h : Real.sin x ≠ 0) :
  deriv f x = -(2 + 3 * Real.sin x^2) / Real.sin x^3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l75_7563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_odd_numbers_minus_remainder_l75_7527

/-- Theorem about the difference between the minimum sum of squares of ten odd numbers and its remainder when divided by 4 -/
theorem min_sum_squares_odd_numbers_minus_remainder : ℕ := by
  -- Define a function to generate the first n odd numbers
  let first_n_odd_numbers (n : ℕ) : List ℕ :=
    List.range n |> List.map (fun i => 2 * i + 1)

  -- Define the sum of squares of the first n odd numbers
  let sum_squares_odd_numbers (n : ℕ) : ℕ :=
    (first_n_odd_numbers n).map (fun x => x * x) |> List.sum

  -- Define the minimum sum of squares for 10 different odd numbers
  let min_sum_squares : ℕ := sum_squares_odd_numbers 10

  -- Define the remainder when divided by 4
  let remainder : ℕ := min_sum_squares % 4

  -- The theorem
  have : min_sum_squares - remainder = 1328 := by sorry

  exact 1328

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_odd_numbers_minus_remainder_l75_7527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doubled_mean_of_reciprocals_l75_7502

def first_four_primes : List ℕ := [2, 3, 5, 7]

def reciprocal_sum (list : List ℕ) : ℚ :=
  (list.map (fun x => (1 : ℚ) / x)).sum

theorem doubled_mean_of_reciprocals :
  2 * (reciprocal_sum first_four_primes / first_four_primes.length) = 247 / 420 := by
  sorry

#eval reciprocal_sum first_four_primes
#eval 2 * (reciprocal_sum first_four_primes / first_four_primes.length)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doubled_mean_of_reciprocals_l75_7502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l75_7568

/-- Given a parabola y^2 = 2px passing through (2,2), 
    the distance from (2,2) to the focus is 5/2 -/
theorem parabola_focus_distance : 
  ∀ (p : ℝ), 
  (2:ℝ)^2 = 2*p*2 → -- parabola passes through (2,2)
  Real.sqrt ((2 - p/2)^2 + (2 - 0)^2) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l75_7568
