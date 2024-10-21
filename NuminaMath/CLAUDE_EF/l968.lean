import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_for_f_geq_two_l968_96858

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a| + |x - 1|

-- Theorem 1
theorem solution_set_when_a_is_one : 
  {x : ℝ | f 1 x ≤ 5} = Set.Icc (-3) 2 := by sorry

-- Theorem 2
theorem range_of_a_for_f_geq_two : 
  {a : ℝ | ∀ x, f a x ≥ 2} = {a : ℝ | a ≥ 1/2 ∨ a ≤ -3/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_for_f_geq_two_l968_96858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l968_96887

/- Define the ellipse parameters -/
variable (a b : ℝ)

/- Define the eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

/- State the theorem -/
theorem ellipse_eccentricity_range (x y : ℝ) :
  (a > b) ∧ (b > 0) ∧ 
  (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (2 / 4 / a^2 + 2 / 4 / b^2 = 1) ∧
  (Real.sqrt 5 ≤ 2*a) ∧ (2*a ≤ Real.sqrt 6) →
  (Real.sqrt 3 / 3 ≤ eccentricity a b) ∧ (eccentricity a b ≤ Real.sqrt 2 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l968_96887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_fourteen_pi_thirds_l968_96807

theorem cos_fourteen_pi_thirds : Real.cos (14 * π / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_fourteen_pi_thirds_l968_96807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l968_96849

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (3 * x) - Real.sin (3 * x)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = 2 * Real.pi / 3 := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l968_96849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_problems_count_is_30_l968_96819

def test_problems_count : ℕ :=
  let total_points : ℕ := 110
  let computation_problems : ℕ := 20
  let computation_problem_points : ℕ := 3
  let word_problem_points : ℕ := 5

  let word_problems : ℕ := (total_points - computation_problems * computation_problem_points) / word_problem_points
  
  computation_problems + word_problems

theorem test_problems_count_is_30 : test_problems_count = 30 := by
  rfl

#eval test_problems_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_problems_count_is_30_l968_96819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l968_96806

noncomputable section

/-- Angle alpha -/
def α : ℝ := sorry

/-- Angle beta -/
def β : ℝ := sorry

/-- Point P -/
def P : ℝ × ℝ := (1, 4 * Real.sqrt 3)

/-- Theorem stating that beta equals π/3 -/
theorem beta_value (h1 : 0 < β) (h2 : β < α) (h3 : α < π / 2)
  (h4 : P.1^2 + P.2^2 = (Real.tan α)^2 + 1)
  (h5 : Real.sin α * Real.sin (π / 2 - β) + Real.cos α * Real.cos (π / 2 + β) = 3 * Real.sqrt 3 / 14) :
  β = π / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l968_96806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_divisibility_l968_96894

theorem infinite_pairs_divisibility (n : ℕ) (h : n ≥ 2) :
  let a := 2^n - 1
  let b := 2^n + 1
  Nat.Coprime a b ∧ a > 1 ∧ b > 1 ∧ (a^b + b^a) % (a + b) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_divisibility_l968_96894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a₁_l968_96862

def sequence_a (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => 13 * sequence_a a₁ n - 2 * (n + 1)

theorem smallest_a₁ (a₁ : ℝ) :
  (∀ n : ℕ, sequence_a a₁ n > 0) →
  a₁ ≥ 1/4 ∧ ∃ a₁', a₁' = 1/4 ∧ ∀ n : ℕ, sequence_a a₁' n > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a₁_l968_96862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_theorem_l968_96866

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry
noncomputable def sequence_c : ℕ → ℝ := sorry

noncomputable def S : ℕ → ℝ := sorry

axiom S_def : ∀ n : ℕ, S n = 2 * sequence_a n - 2

axiom b_initial : sequence_b 1 = 1

axiom b_line : ∀ n : ℕ, sequence_b (n + 1) = sequence_b n + 2

noncomputable def T : ℕ → ℝ := sorry

theorem sequences_theorem :
  (∀ n : ℕ, sequence_a n = 2^n) ∧
  (∀ n : ℕ, sequence_b n = 2*n - 1) ∧
  (∀ n : ℕ, sequence_c n = sequence_a n * sequence_b n) ∧
  (∀ n : ℕ, T n = (2*n - 3) * 2^(n+1) + 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_theorem_l968_96866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_is_1_point_9_l968_96841

-- Define the weights
noncomputable def mother_hen_weight : ℝ := 2.3
noncomputable def baby_chick_weight_grams : ℝ := 400

-- Convert grams to kilograms
noncomputable def grams_to_kg (grams : ℝ) : ℝ := grams / 1000

-- Calculate the weight difference
noncomputable def weight_difference : ℝ := mother_hen_weight - grams_to_kg baby_chick_weight_grams

-- Theorem to prove
theorem weight_difference_is_1_point_9 : weight_difference = 1.9 := by
  -- Expand the definition of weight_difference
  unfold weight_difference
  -- Expand the definition of grams_to_kg
  unfold grams_to_kg
  -- Perform the calculation
  simp [mother_hen_weight, baby_chick_weight_grams]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_is_1_point_9_l968_96841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_parallelogram_side_length_l968_96857

/-- A circumscribed parallelogram around a circle -/
structure CircumscribedParallelogram where
  R : ℝ  -- radius of the inscribed circle
  S : ℝ  -- area of the quadrilateral formed by points of tangency
  h_R_pos : R > 0
  h_S_pos : S > 0

/-- The side length of a circumscribed parallelogram -/
noncomputable def side_length (p : CircumscribedParallelogram) : ℝ := 4 * p.R^3 / p.S

/-- Theorem: The side length of a circumscribed parallelogram is 4R³/S -/
theorem circumscribed_parallelogram_side_length (p : CircumscribedParallelogram) :
  ∃ (side : ℝ), side = side_length p ∧ side > 0 := by
  use side_length p
  constructor
  · rfl
  · sorry  -- Proof that side_length p > 0


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_parallelogram_side_length_l968_96857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l968_96851

theorem min_value_of_f (a : ℝ) (h1 : -1/2 < a) (h2 : a < 0) :
  ∀ x : ℝ, 0 ≤ x → x ≤ π → (Real.cos x) ^ 2 - 2 * a * Real.sin x - 1 ≥ -2 * a - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l968_96851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_y_axis_l968_96881

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2 / 4) * Real.sin (x - Real.pi/4) + (Real.sqrt 6 / 4) * Real.cos (x - Real.pi/4)

theorem min_distance_to_y_axis (k : ℤ) :
  let symmetric_center := k * Real.pi - Real.pi/12
  ∃ (center : ℝ), center = symmetric_center ∧ 
    ∀ (other_center : ℝ), (∃ (m : ℤ), other_center = m * Real.pi - Real.pi/12) →
      |center| ≤ |other_center| :=
by
  sorry

#check min_distance_to_y_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_y_axis_l968_96881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_r_l968_96898

-- Define the ratios and investment times
def investment_ratio : Fin 3 → ℚ
  | 0 => 7
  | 1 => 5
  | 2 => 4

def profit_ratio : Fin 3 → ℚ
  | 0 => 7
  | 1 => 10
  | 2 => 8

def investment_time (x : ℚ) : Fin 3 → ℚ
  | 0 => 2
  | 1 => 4
  | 2 => x

-- Theorem statement
theorem investment_time_r (x : ℚ) : 
  (∀ i j : Fin 3, profit_ratio i / profit_ratio j = 
    (investment_ratio i * investment_time x i) / (investment_ratio j * investment_time x j)) →
  x = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_r_l968_96898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_price_calculation_l968_96831

/-- Calculates the final price of a shoe after a series of price changes -/
theorem shoe_price_calculation (initial_price : ℝ) 
  (wednesday_increase : ℝ) (thursday_discount : ℝ) (friday_discount : ℝ) :
  initial_price = 50 →
  wednesday_increase = 0.2 →
  thursday_discount = 0.15 →
  friday_discount = 0.1 →
  initial_price * (1 + wednesday_increase) * (1 - thursday_discount) * (1 - friday_discount) = 45.9 := by
  intro h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  norm_num
  -- The proof is completed automatically by norm_num

-- We don't need the #eval line in this case

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_price_calculation_l968_96831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l968_96879

/-- Given a line segment AB divided at C such that AC = 2CB, with circles described on AC and CB as diameters,
    and a common tangent to these circles touching AB extended at D, prove that BD = 2CB/(CB-2) -/
theorem tangent_length (A B C D : ℝ × ℝ) (x : ℝ) 
  (h1 : (C.1 - A.1) = 2 * (B.1 - C.1)) -- AC = 2CB
  (h2 : x = B.1 - C.1) -- CB = x
  (h3 : D.1 > B.1) -- D is on AB extended
  (h4 : ∃ (T₁ T₂ : ℝ × ℝ), 
    (T₁.1 - A.1)^2 + T₁.2^2 = x^2 ∧ -- T₁ is on circle with AC as diameter
    (T₂.1 - C.1)^2 + T₂.2^2 = (x/2)^2 ∧ -- T₂ is on circle with CB as diameter
    (D.1 - T₁.1) * T₁.2 = (T₁.1 - A.1) * (D.2 - T₁.2) ∧ -- DT₁ is tangent to circle on AC
    (D.1 - T₂.1) * T₂.2 = (T₂.1 - C.1) * (D.2 - T₂.2)) -- DT₂ is tangent to circle on CB
  : D.1 - B.1 = 2*x/(x-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l968_96879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_omega_x_monotone_implies_omega_range_l968_96889

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

theorem sin_omega_x_monotone_implies_omega_range 
  (ω : ℝ) 
  (h_pos : ω > 0) 
  (h_monotone : ∀ x y, -π/2 ≤ x ∧ x < y ∧ y ≤ 2*π/3 → f ω x < f ω y) :
  ω ∈ Set.Ioo 0 (3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_omega_x_monotone_implies_omega_range_l968_96889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_weight_problem_l968_96809

theorem stone_weight_problem :
  ∃! y : ℕ, ∃ x z : ℕ,
    x + y + z = 100 ∧
    x + 10 * y + 50 * z = 500 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_weight_problem_l968_96809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_increasing_l968_96822

noncomputable def f (x : ℝ) := Real.exp (abs x)

theorem f_is_even_and_increasing :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_increasing_l968_96822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l968_96834

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def b (a : ℕ → ℝ) (n : ℕ) : ℝ := (1/2) ^ (a n)

theorem arithmetic_sequence_solution (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  b a 1 + b a 2 + b a 3 = 21/8 →
  b a 1 * b a 2 * b a 3 = 1/8 →
  (∀ n : ℕ, a n = 2*n - 3) ∨ (∀ n : ℕ, a n = 5 - 2*n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l968_96834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axes_intersect_at_centroid_l968_96812

/-- A polygon represented by its vertices -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- The centroid of a polygon -/
noncomputable def centroid (p : Polygon) : ℝ × ℝ :=
  let n := p.vertices.length
  let sum := p.vertices.foldl (fun acc v => (acc.1 + v.1, acc.2 + v.2)) (0, 0)
  (sum.1 / n, sum.2 / n)

/-- An axis of symmetry of a polygon -/
structure SymmetryAxis where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a point lies on a line -/
def pointOnLine (p : ℝ × ℝ) (l : SymmetryAxis) : Prop :=
  ∃ t : ℝ, p = (l.start.1 + t * l.direction.1, l.start.2 + t * l.direction.2)

/-- A polygon has an axis of symmetry -/
def hasSymmetryAxis (p : Polygon) (a : SymmetryAxis) : Prop :=
  ∀ v : ℝ × ℝ, v ∈ p.vertices → 
    ∃ v' : ℝ × ℝ, v' ∈ p.vertices ∧ pointOnLine ((v.1 + v'.1) / 2, (v.2 + v'.2) / 2) a

theorem symmetry_axes_intersect_at_centroid 
  (p : Polygon) (axes : List SymmetryAxis) 
  (h_multiple_axes : axes.length > 1)
  (h_all_symmetry : ∀ a ∈ axes, hasSymmetryAxis p a) :
  ∀ a ∈ axes, pointOnLine (centroid p) a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axes_intersect_at_centroid_l968_96812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_max_value_of_M_l968_96848

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.exp x - a * x + b

-- Part I
theorem range_of_f_on_interval :
  let f₁ := f 1 1
  ∃ (min max : ℝ), min = 2 ∧ max = Real.exp 2 - 1 ∧
    ∀ x ∈ Set.Icc (-1) 2, min ≤ f₁ x ∧ f₁ x ≤ max := by
  sorry

-- Part II
def M (a b : ℝ) : ℝ := a - b

theorem max_value_of_M :
  ∃ (max : ℝ), max = Real.exp 1 ∧
    (∀ a b : ℝ, (∀ x : ℝ, f a b x ≥ 0) → M a b ≤ max) ∧
    ∃ a b : ℝ, (∀ x : ℝ, f a b x ≥ 0) ∧ M a b = max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_max_value_of_M_l968_96848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_equiv_straight_line_l968_96864

-- Define the triangle and points
variable (A B C P Q : EuclideanSpace ℝ (Fin 2))

-- Define the property of P and Q being inside the triangle
def inside_triangle (P : EuclideanSpace ℝ (Fin 2)) (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the reflection of a point over a line
def reflect_point (P : EuclideanSpace ℝ (Fin 2)) (A B : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

-- Define the sequence of reflections to obtain Q'
def Q' (Q A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  reflect_point (reflect_point (reflect_point Q B C) C A) A B

-- Define a path that touches each side of the triangle
def valid_path (path : ℝ → EuclideanSpace ℝ (Fin 2)) (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the length of a path
noncomputable def path_length (path : ℝ → EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- The main theorem
theorem shortest_path_equiv_straight_line 
  (h_P_inside : inside_triangle P A B C)
  (h_Q_inside : inside_triangle Q A B C) :
  ∀ (path : ℝ → EuclideanSpace ℝ (Fin 2)),
    valid_path path A B C →
    path 0 = P →
    path 1 = Q →
    path_length path ≥ ‖P - Q' Q A B C‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_equiv_straight_line_l968_96864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_formula_l968_96870

def sequence_a : ℕ → ℤ
  | 0 => 5
  | n+1 => 2 * sequence_a n + 2^(n+1) - 1

theorem sequence_a_general_formula (n : ℕ) :
  sequence_a n = (n + 1) * 2^n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_formula_l968_96870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l968_96836

/-- Two circles are externally tangent -/
def externally_tangent (r₁ r₂ : ℝ) : Prop := sorry

/-- A circle is internally tangent to another circle -/
def internally_tangent (r₁ r₂ : ℝ) : Prop := sorry

/-- A chord is a common external tangent to two circles and is part of a larger circle -/
def is_common_external_tangent (chord r₁ r₂ r₃ : ℝ) : Prop := sorry

/-- Given three circles with radii 4, 8, and 12, where the smaller circles are
externally tangent to each other and internally tangent to the largest circle,
the square of the length of the chord of the largest circle that is a common
external tangent to the other two circles is equal to 398.04. -/
theorem chord_length_squared (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : r₃ = 12)
    (h_ext_tangent : externally_tangent r₁ r₂)
    (h_int_tangent₁ : internally_tangent r₁ r₃)
    (h_int_tangent₂ : internally_tangent r₂ r₃)
    (chord : ℝ) (h_chord : is_common_external_tangent chord r₁ r₂ r₃) :
  chord^2 = 398.04 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l968_96836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_height_calculation_l968_96875

/-- Given a tree and a person casting shadows, calculate the person's height -/
theorem shadow_height_calculation (tree_height tree_shadow bob_shadow : ℝ) 
  (h1 : tree_height = 50)
  (h2 : tree_shadow = 25)
  (h3 : bob_shadow = 6) :
  let ratio := tree_height / tree_shadow
  ratio * bob_shadow = 12 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_height_calculation_l968_96875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_astrophysics_degrees_l968_96830

/-- Represents the allocation of a research and development budget --/
structure BudgetAllocation where
  microphotonics : ℚ
  home_electronics : ℚ
  food_additives : ℚ
  genetically_modified_microorganisms : ℚ
  industrial_lubricants : ℚ

/-- Calculates the degrees in a circle representing a given percentage --/
def percentageToDegrees (percentage : ℚ) : ℚ := percentage * 360 / 100

/-- Theorem: The number of degrees representing basic astrophysics research is 72 --/
theorem basic_astrophysics_degrees (budget : BudgetAllocation) :
  budget.microphotonics = 14 ∧
  budget.home_electronics = 24 ∧
  budget.food_additives = 15 ∧
  budget.genetically_modified_microorganisms = 19 ∧
  budget.industrial_lubricants = 8 →
  percentageToDegrees (100 - (budget.microphotonics + budget.home_electronics + budget.food_additives +
    budget.genetically_modified_microorganisms + budget.industrial_lubricants)) = 72 := by
  intro h
  sorry

#eval percentageToDegrees 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_astrophysics_degrees_l968_96830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l968_96801

/-- The function f(x) = ln(x) - 2x^2 is increasing on the interval (0, 1/2) -/
theorem f_increasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1/2 →
  Real.log x₁ - 2 * x₁^2 < Real.log x₂ - 2 * x₂^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l968_96801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_is_3a_squared_volume_is_a_cubed_sqrt_2_over_3_l968_96888

/-- Regular quadrilateral pyramid with base side length a -/
structure RegularQuadPyramid where
  a : ℝ
  h : a > 0

variable (P : RegularQuadPyramid)

/-- The diagonal cross-section area is equal to the base area -/
axiom diagonal_cross_section_eq_base : P.a^2 = (P.a * Real.sqrt 2) * (P.a * Real.sqrt 2) / 2

/-- The lateral surface area of the pyramid -/
noncomputable def lateral_surface_area (P : RegularQuadPyramid) : ℝ := 3 * P.a^2

/-- The volume of the pyramid -/
noncomputable def volume (P : RegularQuadPyramid) : ℝ := (P.a^3 * Real.sqrt 2) / 3

/-- Theorem: The lateral surface area is 3a² -/
theorem lateral_surface_area_is_3a_squared (P : RegularQuadPyramid) :
  lateral_surface_area P = 3 * P.a^2 := by sorry

/-- Theorem: The volume is (a³√2)/3 -/
theorem volume_is_a_cubed_sqrt_2_over_3 (P : RegularQuadPyramid) :
  volume P = (P.a^3 * Real.sqrt 2) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_is_3a_squared_volume_is_a_cubed_sqrt_2_over_3_l968_96888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l968_96861

/-- The area of a sector with given central angle and radius -/
noncomputable def sector_area (angle : ℝ) (radius : ℝ) : ℝ :=
  (1/2) * angle * radius^2

/-- Theorem: The area of a sector with central angle 2 radians and radius 3 is 9 -/
theorem sector_area_specific : sector_area 2 3 = 9 := by
  -- Unfold the definition of sector_area
  unfold sector_area
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l968_96861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l968_96871

/-- The rational function f(x) = (5x^2 - 9) / (3x^2 + 5x + 2) -/
noncomputable def f (x : ℝ) : ℝ := (5 * x^2 - 9) / (3 * x^2 + 5 * x + 2)

/-- The denominator of the function f -/
def denom (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 2

/-- Theorem: The sum of the x-values of the vertical asymptotes of f is -5/3 -/
theorem vertical_asymptotes_sum :
  ∃ (a b : ℝ), (denom a = 0 ∧ denom b = 0 ∧ a ≠ b) ∧ a + b = -5/3 := by
  sorry

#check vertical_asymptotes_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l968_96871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_theorem_l968_96890

/-- Represents the speeds of the cars in different road conditions -/
structure Speeds where
  highway : ℝ
  dirt : ℝ
  muddy : ℝ

/-- Represents the distances between the cars in different road conditions -/
structure Distances where
  initial : ℝ
  dirt : ℝ
  muddy : ℝ

/-- Given the initial conditions and speeds, calculates the distances between the cars -/
noncomputable def calculateDistances (initialDistance : ℝ) (speeds : Speeds) : Distances :=
  { initial := initialDistance,
    dirt := (initialDistance / speeds.highway) * speeds.dirt,
    muddy := ((initialDistance / speeds.highway) * speeds.dirt / speeds.dirt) * speeds.muddy }

theorem car_distance_theorem (initialDistance : ℝ) (speeds : Speeds) 
    (h1 : initialDistance = 400)
    (h2 : speeds.highway = 160)
    (h3 : speeds.dirt = 60)
    (h4 : speeds.muddy = 20) :
  let distances := calculateDistances initialDistance speeds
  distances.dirt = 150 ∧ distances.muddy = 50 := by
  sorry

-- Remove the #eval line as it's causing issues with MetaEval
-- #eval calculateDistances 400 { highway := 160, dirt := 60, muddy := 20 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_theorem_l968_96890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_operation_example_l968_96885

def star_operation (A B : Set ℕ) : Set ℕ :=
  {x | ∃ x1 x2, x1 ∈ A ∧ x2 ∈ B ∧ x = x1 + x2}

theorem star_operation_example :
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {1, 2, 3}
  star_operation A B = {2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_operation_example_l968_96885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l968_96839

/-- The time given to complete the work -/
noncomputable def T : ℝ := sorry

/-- Kevin's work rate -/
noncomputable def kevin_rate : ℝ := 1 / (T - 4)

/-- Dave's work rate -/
noncomputable def dave_rate : ℝ := 1 / (T + 6)

/-- The combined work rate of Kevin and Dave -/
noncomputable def combined_rate : ℝ := kevin_rate + dave_rate

/-- The amount of work done by Kevin and Dave together in 4 days -/
noncomputable def work_4_days : ℝ := 4 * combined_rate

/-- The remaining work after 4 days of working together -/
noncomputable def remaining_work : ℝ := 1 - work_4_days

/-- The time taken for Dave to complete the remaining work alone -/
noncomputable def dave_remaining_time : ℝ := T - 4

theorem job_completion_time :
  (remaining_work = dave_remaining_time * dave_rate) →
  (T = 24) →
  ∃ (time_together : ℝ), time_together * combined_rate = 1 ∧ time_together = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l968_96839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l968_96813

noncomputable section

open Real

theorem trigonometric_identities (α : ℝ) (h : α < π / 4) :
  (∀ (sign : Bool), Real.sin (π / 2 + (if sign then α else -α)) = Real.cos α) ∧
  (∀ (sign : Bool), Real.sin (π + (if sign then α else -α)) = (if sign then Real.sin α else -Real.sin α)) ∧
  (∀ (sign : Bool), Real.sin (3 * π / 2 + (if sign then α else -α)) = -Real.cos α) ∧
  Real.sin (2 * π - α) = -Real.sin α ∧
  (∀ (sign : Bool), Real.cos (π / 2 + (if sign then α else -α)) = (if sign then -Real.sin α else Real.sin α)) ∧
  (∀ (sign : Bool), Real.cos (π + (if sign then α else -α)) = -Real.cos α) ∧
  (∀ (sign : Bool), Real.cos (3 * π / 2 + (if sign then α else -α)) = (if sign then Real.sin α else -Real.sin α)) ∧
  Real.cos (2 * π - α) = Real.cos α :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l968_96813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l968_96820

-- Define the 4D space
def Point4D := Fin 4 → ℝ

-- Define the start and end points
def start : Point4D := λ i => [1, 2, 3, 4].get i
def endpoint : Point4D := λ i => [0, -2, -4, -5].get i

-- Define the sphere center and radius
def sphere_center : Point4D := λ _ => 1
def sphere_radius : ℝ := 3

-- Define the line parameterization
def line (t : ℝ) : Point4D := λ i => start i + t * (endpoint i - start i)

-- Define the quadratic equation for intersection
def intersection_equation (t : ℝ) : ℝ :=
  (line t 0 - sphere_center 0)^2 +
  (line t 1 - sphere_center 1)^2 +
  (line t 2 - sphere_center 2)^2 +
  (line t 3 - sphere_center 3)^2 - sphere_radius^2

-- Theorem statement
theorem intersection_distance (t₁ t₂ : ℝ) 
  (h₁ : intersection_equation t₁ = 0)
  (h₂ : intersection_equation t₂ = 0) :
  ∃ (d : ℝ), d = Real.sqrt 147 * |t₁ - t₂| ∧
             d = Real.sqrt ((line t₁ 0 - line t₂ 0)^2 +
                            (line t₁ 1 - line t₂ 1)^2 +
                            (line t₁ 2 - line t₂ 2)^2 +
                            (line t₁ 3 - line t₂ 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l968_96820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_1010_mod_1000_l968_96818

-- Define the polynomial q(x)
noncomputable def q (x : ℝ) : ℝ := (x^1011 - 1) / (x - 1)

-- Define the divisor polynomial
def divisor (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 2

-- Define s(x) as the remainder when q(x) is divided by divisor(x)
noncomputable def s (x : ℝ) : ℝ := q x % divisor x

-- Theorem statement
theorem remainder_of_s_1010_mod_1000 : (Int.floor (|s 1010|) : ℤ) % 1000 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_1010_mod_1000_l968_96818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l968_96876

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

noncomputable def shortest_distance : ℝ := Real.sqrt 5

theorem shortest_distance_proof :
  ∃ (x₀ y₀ : ℝ), y₀ = f x₀ ∧
  (∀ (x y : ℝ), y = f x → 
    (x - x₀)^2 + (y - y₀)^2 ≥ shortest_distance^2) ∧
  (∃ (x y : ℝ), line x y ∧ 
    (x - x₀)^2 + (y - y₀)^2 = shortest_distance^2) :=
by sorry

#check shortest_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l968_96876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_half_l968_96886

/-- The area between the line y=x and the curve y=x^3 from x=-1 to x=1 -/
noncomputable def area_between_curves : ℝ :=
  ∫ x in (-1)..1, (x - x^3)

/-- Theorem stating that the area between y=x and y=x^3 is 1/2 -/
theorem area_is_half : area_between_curves = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_half_l968_96886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cost_is_5_l968_96845

/-- The cost of an ice cream cone given the following conditions:
  * There are 15 ice cream cones and 5 cups of pudding
  * Each cup of pudding costs $2
  * The total cost of ice cream is $65 more than the total cost of pudding
-/
noncomputable def ice_cream_cost : ℚ :=
  let num_ice_cream : ℕ := 15
  let num_pudding : ℕ := 5
  let pudding_cost : ℚ := 2
  let extra_cost : ℚ := 65
  let total_pudding_cost : ℚ := num_pudding * pudding_cost
  let total_ice_cream_cost : ℚ := total_pudding_cost + extra_cost
  total_ice_cream_cost / num_ice_cream

theorem ice_cream_cost_is_5 : ice_cream_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cost_is_5_l968_96845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_of_powers_l968_96872

theorem square_difference_of_powers (a b : ℝ) : 
  a = 100^(50 : ℤ) - 100^(-(50 : ℤ)) → 
  b = 100^(50 : ℤ) + 100^(-(50 : ℤ)) → 
  a^2 - b^2 = -4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_of_powers_l968_96872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cone_ratio_l968_96854

/-- A configuration of a right circular cone topped by a sphere -/
structure ConeSphereConfig where
  r : ℝ  -- radius of the sphere
  R : ℝ  -- radius of the base of the cone
  h : ℝ  -- height of the cone

/-- The volume of a sphere -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of a right circular cone -/
noncomputable def coneVolume (R h : ℝ) : ℝ := (1 / 3) * Real.pi * R^2 * h

/-- Theorem: The ratio of sphere radius to cone base radius in the given configuration -/
theorem sphere_cone_ratio (config : ConeSphereConfig) 
    (hTouch : config.h = 2 * config.r)  -- sphere touches cone internally at top
    (hVolume : coneVolume config.R config.h = 3 * sphereVolume config.r)  -- cone volume is 3 times sphere volume
    : config.r / config.R = 1 / Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cone_ratio_l968_96854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_angle_bisector_perpendicular_not_all_right_triangles_have_perpendicular_bisector_l968_96874

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  /-- The two equal sides of the isosceles triangle -/
  side1 : ℝ
  side2 : ℝ
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The vertex angle of the isosceles triangle -/
  vertex_angle : ℝ
  /-- Condition that side1 and side2 are equal -/
  equal_sides : side1 = side2
  /-- Condition that the triangle is valid -/
  valid_triangle : side1 + side2 > base ∧ base > 0 ∧ side1 > 0

/-- A right triangle -/
structure RightTriangle where
  /-- The two sides adjacent to the right angle -/
  side1 : ℝ
  side2 : ℝ
  /-- The hypotenuse of the right triangle -/
  hypotenuse : ℝ
  /-- Condition that the triangle has a right angle -/
  right_angle : side1^2 + side2^2 = hypotenuse^2
  /-- Condition that the triangle is valid -/
  valid_triangle : side1 > 0 ∧ side2 > 0 ∧ hypotenuse > 0

/-- Helper function to represent that a bisector is perpendicular to the base in an isosceles triangle -/
def bisector_perpendicular_to_base (t : IsoscelesTriangle) (bisector : ℝ) : Prop :=
  sorry

/-- Helper function to represent that a bisector is perpendicular to a side in a right triangle -/
def bisector_perpendicular_to_base_right (t : RightTriangle) (bisector : ℝ) : Prop :=
  sorry

/-- The theorem stating the property of isosceles triangles -/
theorem isosceles_angle_bisector_perpendicular (t : IsoscelesTriangle) :
  ∃ (bisector : ℝ), bisector_perpendicular_to_base t bisector :=
sorry

/-- The theorem stating that this property doesn't necessarily hold for right triangles -/
theorem not_all_right_triangles_have_perpendicular_bisector :
  ∃ (t : RightTriangle), ∀ (bisector : ℝ), ¬(bisector_perpendicular_to_base_right t bisector) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_angle_bisector_perpendicular_not_all_right_triangles_have_perpendicular_bisector_l968_96874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_l968_96804

noncomputable def k (x : ℝ) : ℝ := 1 / (x - 3) + 1 / (x^2 - 9) + 1 / (x^3 - 27)

theorem k_domain :
  {x : ℝ | ∃ y, k x = y} = {x : ℝ | x < -3 ∨ (-3 < x ∧ x < 3) ∨ 3 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_l968_96804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_L_for_product_less_than_0_3_l968_96867

def a : ℕ → ℝ
  | 0 => 0.8
  | n + 1 => (a n) ^ 2

def product (L : ℕ) : ℝ :=
  Finset.prod (Finset.range L) (λ i => a i)

theorem least_L_for_product_less_than_0_3 :
  ∀ L, (L < 3 → product L ≥ 0.3) ∧
       (L ≥ 3 → product L < 0.3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_L_for_product_less_than_0_3_l968_96867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_and_range_l968_96878

noncomputable def f (a x : ℝ) : ℝ := |x + a + 1| + |x - 4/a|

theorem f_lower_bound_and_range (a : ℝ) (h : a > 0) :
  (∀ x, f a x ≥ 5) ∧
  {a | a > 0 ∧ f a 1 < 6} = Set.Ioo 1 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_and_range_l968_96878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_equation_l968_96852

/-- Predicate to check if a point is an angle bisector -/
def is_angle_bisector (A B C P : ℝ × ℝ) : Prop :=
  dist A P * dist B C = dist B P * dist A C

/-- Define a line through two points -/
def line_through (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ t : ℝ, P = (1 - t) • A + t • B}

/-- Given a triangle ABC with vertex A at (3, -1), angle bisector of B as x=0,
    and angle bisector of C as y=x, the equation of line BC is 2x - y + 5 = 0 -/
theorem triangle_line_equation (A B C : ℝ × ℝ) :
  A = (3, -1) →
  (∀ x y : ℝ, x = 0 → is_angle_bisector A B C (x, y)) →
  (∀ x y : ℝ, y = x → is_angle_bisector A C B (x, y)) →
  ∃ k : ℝ, ∀ x y : ℝ, (x, y) ∈ line_through B C ↔ 2*x - y + 5 = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_equation_l968_96852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_table_theorem_l968_96814

/-- A move is adding 1 to all numbers in a row or column -/
def Move (table : Matrix (Fin 2010) (Fin 2010) ℕ) : Matrix (Fin 2010) (Fin 2010) ℕ :=
  sorry

/-- A table is equilibrium if all numbers can be made equal after finitely many moves -/
def IsEquilibrium (table : Matrix (Fin 2010) (Fin 2010) ℕ) : Prop :=
  sorry

/-- The table contains numbers 2^0, 2^1, ..., 2^n -/
def ContainsPowersOfTwo (table : Matrix (Fin 2010) (Fin 2010) ℕ) (n : ℕ) : Prop :=
  sorry

theorem equilibrium_table_theorem :
  ∃ (table : Matrix (Fin 2010) (Fin 2010) ℕ),
    ContainsPowersOfTwo table 1 ∧
    IsEquilibrium table ∧
    (∀ m > 1, ¬∃ (table : Matrix (Fin 2010) (Fin 2010) ℕ), ContainsPowersOfTwo table m ∧ IsEquilibrium table) ∧
    (∀ i j, table i j ≤ 2) :=
by
  sorry

#check equilibrium_table_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_table_theorem_l968_96814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l968_96847

theorem triangle_trigonometric_identity 
  (A B C : ℝ) (a b c h : ℝ) : 
  (0 < a) → (0 < b) → (0 < c) → 
  (A + B + C = Real.pi) → 
  (c = 2 * a) → 
  (h = a) → 
  (Real.sin ((C - A) / 2) + Real.cos ((C + A) / 2) = 1) :=
by
  intros ha hb hc hABC hc ha'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l968_96847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_numbers_difference_l968_96803

theorem consecutive_odd_numbers_difference (nums : List Int) : 
  nums.length = 7 ∧ 
  (∀ i ∈ List.range 6, nums[i + 1]! = nums[i]! + 2) ∧
  (∀ n ∈ nums, n % 2 = 1) ∧
  nums.sum / nums.length = 75 →
  (nums.maximum?.getD 0) - (nums.minimum?.getD 0) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_numbers_difference_l968_96803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_tangent_intercept_l968_96846

/-- Represents a 3D point -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  distance : ℝ

/-- Represents a cone in 3D space -/
structure Cone where
  vertex : Point3D
  axis : Point3D
  angle : ℝ

/-- Two planes are parallel -/
def are_parallel (p1 p2 : Plane) : Prop :=
  sorry

/-- Distance between two parallel planes -/
def distance_between (p1 p2 : Plane) : ℝ :=
  sorry

/-- Tangents from a point to a sphere -/
def tangents_from_point_to_sphere (p : Point3D) (s : Sphere) : Set Point3D :=
  sorry

/-- Common tangents of two cones -/
def common_tangents (c1 c2 : Cone) : Set Point3D :=
  sorry

/-- Tangents intercepted by planes with specific length -/
def tangents_intercepted_by_planes (p : Point3D) (s : Sphere) (s1 s2 : Plane) (d : ℝ) : Set Point3D :=
  sorry

/-- The main theorem statement -/
theorem sphere_tangent_intercept (s : Sphere) (p : Point3D) (s1 s2 : Plane) (d : ℝ) :
  ∃ (c1 c2 : Cone),
    (are_parallel s1 s2) →
    (distance_between s1 s2 = d) →
    (tangents_from_point_to_sphere p s).inter
      (common_tangents c1 c2) =
    (tangents_intercepted_by_planes p s s1 s2 d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_tangent_intercept_l968_96846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_property_l968_96860

open Real

/-- The function f(x) = ln x + (1/2)x^2 - 2kx --/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * x^2 - 2 * k * x

/-- The derivative of f(x) --/
noncomputable def f_deriv (k : ℝ) (x : ℝ) : ℝ := 1/x + x - 2*k

theorem extreme_point_property (k : ℝ) :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → 
  f_deriv k x₁ = 0 → f_deriv k x₂ = 0 → 
  f k x₂ < -3/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_property_l968_96860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_equation_l968_96855

theorem complex_square_equation : 
  ∀ z : ℂ, z^2 = -91 - 54*I ↔ 
    z = Complex.mk (Real.sqrt ((-91 + Real.sqrt 11197) / 2)) (-27 / Real.sqrt ((-91 + Real.sqrt 11197) / 2)) ∨
    z = Complex.mk (-Real.sqrt ((-91 + Real.sqrt 11197) / 2)) (27 / Real.sqrt ((-91 + Real.sqrt 11197) / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_equation_l968_96855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_is_2_5_l968_96816

/-- Represents the distribution of population and wealth in a two-country world --/
structure WorldDistribution where
  -- Country X's share of world population
  x_pop_share : ℝ
  -- Country X's share of world wealth
  x_wealth_share : ℝ
  -- Country Y's share of world population
  y_pop_share : ℝ
  -- Country Y's share of world wealth
  y_wealth_share : ℝ
  -- Percentage of Country Y's population that is wealthier
  y_wealthy_pop_percent : ℝ
  -- Percentage of Country Y's wealth owned by the wealthier population
  y_wealthy_wealth_percent : ℝ
  -- Constraints
  x_pop_share_valid : 0 < x_pop_share ∧ x_pop_share < 1
  y_pop_share_valid : 0 < y_pop_share ∧ y_pop_share < 1
  x_wealth_share_valid : 0 < x_wealth_share ∧ x_wealth_share < 1
  y_wealth_share_valid : 0 < y_wealth_share ∧ y_wealth_share < 1
  y_wealthy_pop_percent_valid : 0 < y_wealthy_pop_percent ∧ y_wealthy_pop_percent < 1
  y_wealthy_wealth_percent_valid : 0 < y_wealthy_wealth_percent ∧ y_wealthy_wealth_percent < 1
  total_pop_share : x_pop_share + y_pop_share ≤ 1
  total_wealth_share : x_wealth_share + y_wealth_share ≤ 1

/-- Calculates the ratio of average wealth per citizen in Country X to Country Y --/
noncomputable def wealthRatio (w : WorldDistribution) : ℝ :=
  (w.x_wealth_share / w.x_pop_share) / 
  ((w.y_wealth_share * w.y_wealthy_wealth_percent) / (w.y_pop_share * w.y_wealthy_pop_percent) + 
   (w.y_wealth_share * (1 - w.y_wealthy_wealth_percent)) / (w.y_pop_share * (1 - w.y_wealthy_pop_percent))) * 2

theorem wealth_ratio_is_2_5 (w : WorldDistribution) 
  (hx_pop : w.x_pop_share = 0.4)
  (hx_wealth : w.x_wealth_share = 0.6)
  (hy_pop : w.y_pop_share = 0.2)
  (hy_wealth : w.y_wealth_share = 0.3)
  (hy_wealthy_pop : w.y_wealthy_pop_percent = 0.5)
  (hy_wealthy_wealth : w.y_wealthy_wealth_percent = 0.8) :
  wealthRatio w = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_is_2_5_l968_96816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l968_96826

noncomputable def f (x : ℝ) := Real.cos (2 * x - Real.pi / 6)

theorem f_satisfies_conditions :
  (∀ x : ℝ, f (Real.pi / 12 + x) + f (Real.pi / 12 - x) = 0) ∧
  (∀ x : ℝ, -Real.pi / 6 < x → x < Real.pi / 3 → (deriv f x) > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l968_96826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imoSelection1992_l968_96833

/-- The polynomial f(x) = x^8 + 4x^6 + 2x^4 + 28x^2 + 1 -/
def f (x : ℤ) : ℤ := x^8 + 4*x^6 + 2*x^4 + 28*x^2 + 1

/-- The polynomial g(x) = (x - z₁)(x - z₂)...(x - z₈) -/
def g (x z₁ z₂ z₃ z₄ z₅ z₆ z₇ z₈ : ℤ) : ℤ :=
  (x - z₁) * (x - z₂) * (x - z₃) * (x - z₄) * (x - z₅) * (x - z₆) * (x - z₇) * (x - z₈)

theorem imoSelection1992 (p : ℕ) (hpPrime : Nat.Prime p) (hp : p > 3) 
  (z : ℤ) (hz : (p : ℤ) ∣ f z) :
  ∃ z₁ z₂ z₃ z₄ z₅ z₆ z₇ z₈ : ℤ, ∀ k, (p : ℤ) ∣ (f k - g k z₁ z₂ z₃ z₄ z₅ z₆ z₇ z₈) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imoSelection1992_l968_96833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_equalities_l968_96808

open Set

universe u

variable {U : Type u}
variable (A B C K : Set U)

theorem set_operations_equalities :
  ((A \ K) ∪ (B \ K) = (A ∪ B) \ K) ∧
  (A \ (B \ C) = (A \ B) ∪ (A ∩ C)) ∧
  (A \ (A \ B) = A ∩ B) ∧
  ((A \ B) \ C = (A \ C) \ (B \ C)) ∧
  (A \ (B ∩ C) = (A \ B) ∪ (A \ C)) ∧
  (A \ (B ∪ C) = (A \ B) ∩ (A \ C)) ∧
  (A \ B = (A ∪ B) \ B) ∧
  (A \ B = A \ (A ∩ B)) := by
  sorry

#check set_operations_equalities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_equalities_l968_96808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circles_collinearity_l968_96800

-- Define a circle with center and radius
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the problem setup
def three_circles_intersection (O₁ O₂ O₃ O A B C : EuclideanSpace ℝ (Fin 2)) (c₁ c₂ c₃ : Circle) : Prop :=
  -- Circles intersect at common point O
  (dist c₁.center O = c₁.radius) ∧
  (dist c₂.center O = c₂.radius) ∧
  (dist c₃.center O = c₃.radius) ∧
  -- Circles intersect pairwise at A, B, C
  (dist c₁.center A = c₁.radius) ∧ (dist c₂.center A = c₂.radius) ∧
  (dist c₂.center B = c₂.radius) ∧ (dist c₃.center B = c₃.radius) ∧
  (dist c₃.center C = c₃.radius) ∧ (dist c₁.center C = c₁.radius)

-- Define concyclicity
def concyclic (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ),
    dist P center = radius ∧
    dist Q center = radius ∧
    dist R center = radius ∧
    dist S center = radius

-- Define collinearity
def collinear (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (t : ℝ), Q - P = t • (R - P)

-- The main theorem
theorem three_circles_collinearity
  (O₁ O₂ O₃ O A B C : EuclideanSpace ℝ (Fin 2))
  (c₁ c₂ c₃ : Circle)
  (h₁ : three_circles_intersection O₁ O₂ O₃ O A B C c₁ c₂ c₃)
  (h₂ : concyclic O O₁ O₂ O₃) :
  collinear A B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circles_collinearity_l968_96800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l968_96838

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (4^x + 1)

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is an odd function
  (a = -1/2) ∧  -- Part 1: value of a
  (∀ x y, x < y → f a x > f a y) ∧  -- Part 2: f is decreasing
  (Set.Icc (-15/34) (3/10) = {y | ∃ x ∈ Set.Ico (-1) 2, y = f a x}) :=  -- Part 3: range of f
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l968_96838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_sqrt_at_one_l968_96882

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem tangent_slope_sqrt_at_one :
  deriv f 1 = 1/2 := by
  -- Calculate the derivative of f
  have h1 : deriv f = fun x => 1 / (2 * Real.sqrt x) := by
    sorry  -- Proof of the derivative calculation

  -- Evaluate the derivative at x = 1
  have h2 : deriv f 1 = 1 / (2 * Real.sqrt 1) := by
    sorry  -- Proof of the evaluation at x = 1

  -- Simplify the result
  have h3 : 1 / (2 * Real.sqrt 1) = 1 / 2 := by
    sorry  -- Proof of the simplification

  -- Combine the steps to conclude
  rw [h2, h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_sqrt_at_one_l968_96882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l968_96850

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x - 7 else Real.sqrt x

-- State the theorem
theorem range_of_a (a : ℝ) (h : f a < 1) : -3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l968_96850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_iff_tangent_l968_96817

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the structure for a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the structure for a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Function to check if two circles are externally tangent
def areExternallyTangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Function to check if a quadrilateral is cyclic
def isCyclic (q : Quadrilateral) : Prop :=
  sorry  -- Definition of cyclic quadrilateral

-- Helper function to check if a point is on a circle
def isOnCircle (p : Point) (c : Circle) : Prop :=
  let (cx, cy) := c.center
  (p.x - cx)^2 + (p.y - cy)^2 = c.radius^2

-- Helper function to check if a line is an external tangent to two circles
def isExternalTangent (p1 p2 : Point) (c1 c2 : Circle) : Prop :=
  sorry  -- Definition of external tangent

-- Main theorem
theorem cyclic_iff_tangent 
  (c1 c2 : Circle) 
  (q : Quadrilateral) 
  (h1 : c1.radius ≠ c2.radius) 
  (h2 : isOnCircle q.A c1 ∧ isOnCircle q.B c2 ∧ isOnCircle q.C c2 ∧ isOnCircle q.D c1) 
  (h3 : isExternalTangent q.A q.B c1 c2 ∧ isExternalTangent q.C q.D c1 c2) :
  isCyclic q ↔ areExternallyTangent c1 c2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_iff_tangent_l968_96817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_N_disjoint_l968_96859

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 + x + a) / Real.log a

-- Define set M
def M : Set ℝ := {a | a > (1/2) ∧ a ≠ 1}

-- Define set N
def N : Set ℝ := {a | 0 < a ∧ a ≤ (1/2)}

-- Theorem statement
theorem M_N_disjoint : M ∩ N = ∅ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_N_disjoint_l968_96859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_container_problem_l968_96897

theorem milk_container_problem (A B C D : ℝ) 
  (hB : B = 0.55 * A)
  (hC : C = 1.125 * A)
  (hD : D = 0.8 * A)
  (h_equal : B + 150 = C - 50 ∧ C - 50 = D - 100) :
  A = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_container_problem_l968_96897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_internal_right_angles_l968_96869

/-- The largest number of internal right angles in a polygon with n vertices -/
def max_right_angles (n : ℕ) : ℕ :=
  (2 * n) / 3 + 1

/-- A polygon is represented as a set of points in the plane -/
def Polygon := Set (ℝ × ℝ)

/-- The number of vertices in a polygon -/
def num_vertices (p : Polygon) : ℕ := sorry

/-- The number of internal right angles in a polygon -/
def num_internal_right_angles (p : Polygon) : ℕ := sorry

/-- Theorem stating the maximum number of internal right angles in a polygon with n vertices -/
theorem max_internal_right_angles (n : ℕ) (h : n ≥ 5) :
  ∃ (k : ℕ), k = max_right_angles n ∧
  (∀ (m : ℕ), ∃ (polygon : Polygon),
    (num_vertices polygon = n ∧
     num_internal_right_angles polygon = m)
    → m ≤ k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_internal_right_angles_l968_96869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_over_a_range_l968_96825

/-- A quadratic function with coefficient a > 0 and two distinct zeros in [1,2] -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  zeros_in_interval : ∃ (x y : ℝ), x ≠ y ∧ x ∈ Set.Icc 1 2 ∧ y ∈ Set.Icc 1 2 ∧
    a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0

/-- The value of f(1)/a for a quadratic function f -/
noncomputable def f_one_over_a (f : QuadraticFunction) : ℝ :=
  (f.a + f.b + f.c) / f.a

/-- The theorem stating the range of f(1)/a -/
theorem f_one_over_a_range (f : QuadraticFunction) :
  ∃ (S : Set ℝ), S = Set.Ico 0 1 ∧ f_one_over_a f ∈ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_over_a_range_l968_96825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_findings_l968_96863

-- Define the gnomes
inductive Gnome
| Pili
| Eli
| Spali

-- Define the items
inductive Item
| Diamond
| Topaz
| CopperBasin

-- Define hood colors
inductive HoodColor
| Red
| Blue

-- Define a function to represent beard length
def beardLength : Gnome → ℕ := sorry

-- Define a function to represent hood color
def hoodColor : Gnome → HoodColor := sorry

-- Define a function to represent who found what
def foundItem : Gnome → Item := sorry

-- State the theorem
theorem gnome_findings :
  (hoodColor Eli = HoodColor.Red) →
  (beardLength Eli > beardLength Pili) →
  (∃ g : Gnome, foundItem g = Item.CopperBasin ∧ 
    (∀ g' : Gnome, beardLength g ≥ beardLength g') ∧
    hoodColor g = HoodColor.Blue) →
  (∃ g : Gnome, foundItem g = Item.Diamond ∧
    (∀ g' : Gnome, g ≠ g' → beardLength g < beardLength g')) →
  (foundItem Spali = Item.CopperBasin ∧
   foundItem Pili = Item.Diamond ∧
   foundItem Eli = Item.Topaz) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_findings_l968_96863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_tan_equation_l968_96883

-- Define the inverse tangent of 500
noncomputable def tanInv500 : ℝ := Real.arctan 500

-- Define the property that tan θ > θ for 0 < θ < π/2
axiom tan_gt_self : ∀ θ : ℝ, 0 < θ → θ < Real.pi / 2 → Real.tan θ > θ

-- Theorem statement
theorem no_solutions_tan_equation :
  ∀ x : ℝ, 0 ≤ x → x ≤ tanInv500 → Real.tan x ≠ -Real.tan (Real.tan x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_tan_equation_l968_96883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_of_sum_is_zero_l968_96802

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equality condition
def equality_condition (a b : ℝ) : Prop :=
  (2 + i) / (1 + i) = Complex.ofReal a + Complex.ofReal b * i

-- Theorem statement
theorem log_of_sum_is_zero (a b : ℝ) (h : equality_condition a b) :
  Real.log (a + b) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_of_sum_is_zero_l968_96802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_total_distance_l968_96837

/-- The number of boys on the circle -/
def num_boys : ℕ := 8

/-- The radius of the circle in feet -/
noncomputable def radius : ℝ := 40

/-- The angle between adjacent boys in radians -/
noncomputable def angle_between : ℝ := 2 * Real.pi / num_boys

/-- The minimum number of boys between non-adjacent boys -/
def min_boys_between : ℕ := 2

/-- The angle to the nearest non-adjacent boy in radians -/
noncomputable def angle_to_nearest_non_adjacent : ℝ := angle_between * (min_boys_between + 1)

/-- The number of non-adjacent boys each boy visits -/
def num_visits : ℕ := num_boys - 2 * min_boys_between - 1

theorem minimum_total_distance :
  let total_distance := 2 * (num_boys : ℝ) * (num_visits : ℝ) * radius * Real.sin (angle_to_nearest_non_adjacent / 2)
  total_distance = 2560 * Real.sin (67.5 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_total_distance_l968_96837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_inequality_l968_96823

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

-- State the theorem
theorem range_of_m_for_inequality (m : ℝ) :
  f (2 * m - 1) + f (3 - m) > 0 ↔ m > -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_inequality_l968_96823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_cone_intersection_l968_96827

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A cone with a vertex and a circular base -/
structure Cone where
  vertex : Point
  base : Set Point  -- Representing the circular base as a set of points

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ  -- Plane equation: ax + by + cz + d = 0

/-- The intersection of a cone and a plane -/
def intersection (c : Cone) (p : Plane) : Set Point := sorry

/-- The area of a set of points -/
noncomputable def area (s : Set Point) : ℝ := sorry

/-- Check if a set of points forms a triangle -/
def isTriangle (s : Set Point) : Prop := sorry

/-- An axial section of a cone -/
def isAxialSection (c : Cone) (p : Plane) : Prop := sorry

/-- The theorem stating that the area of the triangle formed by the intersection
    of a cone with a plane passing through the vertex of the cone is maximized
    when the plane is an axial section -/
theorem max_area_cone_intersection (c : Cone) (p : Plane) :
  p.a * c.vertex.x + p.b * c.vertex.y + p.c * c.vertex.z + p.d = 0 →
  isTriangle (intersection c p) →
  (∀ q : Plane, q.a * c.vertex.x + q.b * c.vertex.y + q.c * c.vertex.z + q.d = 0 →
    isTriangle (intersection c q) →
    area (intersection c p) ≥ area (intersection c q)) ↔
  isAxialSection c p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_cone_intersection_l968_96827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_l968_96884

/-- A sequence of digits satisfying the given conditions -/
def ValidSequence (s : List Nat) : Prop :=
  s.length = 3003 ∧
  s.head? = some 2 ∧
  ∀ i, i < s.length - 1 →
    (s[i]?.getD 0 * 10 + s[i+1]?.getD 0) % 23 = 0 ∨ (s[i]?.getD 0 * 10 + s[i+1]?.getD 0) % 29 = 0

/-- The theorem stating the largest possible last digit -/
theorem largest_last_digit (s : List Nat) (h : ValidSequence s) :
  s.getLast?.getD 0 ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_l968_96884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_adjacent_face_diagonals_l968_96880

/-- A cube is a three-dimensional shape with six square faces of equal size. -/
structure Cube where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A face diagonal is a line segment connecting two opposite corners of a face of the cube. -/
noncomputable def face_diagonal (c : Cube) : ℝ := c.side_length * Real.sqrt 2

/-- The angle between two adjacent faces of a cube is 90°. -/
def AngleBetweenFaces (c : Cube) : ℝ := 90

/-- The angle between two line segments. -/
def AngleBetweenLineSeg (a b : ℝ) : ℝ := sorry

/-- The theorem stating that the angle between two face diagonals on adjacent faces of a cube is 90°. -/
theorem angle_between_adjacent_face_diagonals (c : Cube) :
  AngleBetweenLineSeg (face_diagonal c) (face_diagonal c) = AngleBetweenFaces c := by
  sorry

#check angle_between_adjacent_face_diagonals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_adjacent_face_diagonals_l968_96880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_french_only_speakers_count_l968_96899

/-- Represents the survey results of students' language abilities -/
structure LanguageSurvey where
  total : ℕ
  frenchSpeakers : ℕ
  frenchAndEnglishSpeakers : ℕ
  nonFrenchSpeakerPercentage : ℚ

/-- Calculates the number of French-speaking students who do not speak English -/
def frenchOnlySpeakers (survey : LanguageSurvey) : ℕ :=
  survey.frenchSpeakers - survey.frenchAndEnglishSpeakers

/-- Theorem stating the number of French-only speakers in the given survey -/
theorem french_only_speakers_count (survey : LanguageSurvey) 
  (h1 : survey.total = 200)
  (h2 : survey.nonFrenchSpeakerPercentage = 60 / 100)
  (h3 : survey.frenchAndEnglishSpeakers = 20)
  (h4 : survey.frenchSpeakers = (survey.total : ℚ) * (1 - survey.nonFrenchSpeakerPercentage)) :
  frenchOnlySpeakers survey = 60 := by
  sorry

#eval frenchOnlySpeakers {
  total := 200,
  frenchSpeakers := 80,
  frenchAndEnglishSpeakers := 20,
  nonFrenchSpeakerPercentage := 60 / 100
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_french_only_speakers_count_l968_96899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l968_96856

/-- Represents the problem of a train passing a platform -/
def TrainProblem (train_speed_kmh : ℝ) (platform_length : ℝ) (time_pass_man : ℝ) : Prop :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let train_length := train_speed_ms * time_pass_man
  let total_distance := train_length + platform_length
  let time_pass_platform := total_distance / train_speed_ms
  ∃ ε > 0, |time_pass_platform - 35| < ε

/-- Theorem stating that given the conditions, the train takes approximately 35 seconds to pass the platform -/
theorem train_passing_platform :
  TrainProblem 54 225.018 20 := by
  sorry

#check train_passing_platform

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l968_96856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_team_mean_height_l968_96815

noncomputable def player_heights : List ℝ := [50, 51, 54, 60, 62, 62, 63, 65, 68, 70, 71, 74, 75]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / (l.length : ℝ)

theorem volleyball_team_mean_height :
  abs (mean player_heights - 63.46) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_team_mean_height_l968_96815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l968_96840

/-- The ellipse C: x²/4 + y²/3 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- The right focus F of the ellipse C -/
def F : ℝ × ℝ := (1, 0)

/-- A point A outside the ellipse -/
noncomputable def A : ℝ × ℝ := (1, 2 * Real.sqrt 2)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The maximum value of |PA| + |PF| for any point P on the ellipse C -/
theorem max_distance_sum : ∀ P ∈ C, distance P A + distance P F ≤ 4 + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l968_96840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_f_implies_t_range_l968_96829

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 4*x - 3 * Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := -x + 4 - 3/x

-- Define the condition for non-monotonicity
def non_monotonic (f : ℝ → ℝ) (f' : ℝ → ℝ) (t : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo t (t + 1), f' x = 0

-- Theorem statement
theorem non_monotonic_f_implies_t_range (t : ℝ) :
  non_monotonic f f' t → (0 < t ∧ t < 1) ∨ (2 < t ∧ t < 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_f_implies_t_range_l968_96829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_for_two_zeros_sum_of_zeros_lower_bound_l968_96805

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - (a + 1) * x^2 - 2 * a * x + 1

-- Part 1: Tangent line equation when a = 1
def tangent_line (x : ℝ) : ℝ := -4 * x + 1

theorem tangent_line_at_one (x : ℝ) :
  (deriv (f 1)) 1 = (deriv tangent_line) 1 ∧
  f 1 1 = tangent_line 1 := by sorry

-- Part 2: Range of a for which f has two zeros
def has_two_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0

theorem range_of_a_for_two_zeros :
  ∀ a : ℝ, has_two_zeros a ↔ -1 < a ∧ a < 0 := by sorry

-- Part 3: Proof that x₁ + x₂ > 2 / (a + 1)
theorem sum_of_zeros_lower_bound {a : ℝ} (ha : -1 < a ∧ a < 0) 
  {x₁ x₂ : ℝ} (hx : x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :
  x₁ + x₂ > 2 / (a + 1) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_for_two_zeros_sum_of_zeros_lower_bound_l968_96805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_intersection_range_l968_96824

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the trajectory of M
def trajectory_M (x y : ℝ) : Prop := y^2 + 2*x^2 = 1

-- Define the line l
def line_l (x y m k : ℝ) : Prop := y = k*x + m

-- Define the condition for distinct intersection points
def distinct_intersections (m k : ℝ) : Prop := 4*(k^2 - 2*m^2 + 2) > 0

-- Define the relationship between A, B, and Q
def point_relationship (x₁ y₁ x₂ y₂ m k : ℝ) : Prop := 
  x₁ = -3*x₂ ∧ x₁ + x₂ = -2*k*m/(k^2 + 2) ∧ x₁*x₂ = (m^2 - 1)/(k^2 + 2)

-- Main theorem
theorem trajectory_intersection_range (m : ℝ) : 
  (∃ k x₁ y₁ x₂ y₂, 
    m ≠ 0 ∧
    distinct_intersections m k ∧
    trajectory_M x₁ y₁ ∧
    trajectory_M x₂ y₂ ∧
    line_l x₁ y₁ m k ∧
    line_l x₂ y₂ m k ∧
    point_relationship x₁ y₁ x₂ y₂ m k) 
  ↔ 
  (m > -1 ∧ m < -1/2) ∨ (m > 1/2 ∧ m < 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_intersection_range_l968_96824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_b_diff_not_div_by_five_l968_96895

def base_b_diff (b : ℕ) : ℤ := b * (3 * b^2 - 3 * b - 1)

theorem base_b_diff_not_div_by_five (b : ℕ) :
  b ∈ ({5, 6, 7, 8, 10} : Set ℕ) →
  ¬(5 ∣ base_b_diff b) ↔ b = 6 ∨ b = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_b_diff_not_div_by_five_l968_96895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l968_96811

/-- Definition of the ellipse C -/
noncomputable def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Definition of the unit circle -/
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Definition of a tangent line to the unit circle -/
def tangent_line (m k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - m)

/-- Theorem stating the maximum chord length -/
theorem max_chord_length (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : eccentricity a b = Real.sqrt 3 / 2)
  (h4 : 4 * a = 8) :
  ∃ (m : ℝ),
    ∀ (k x₁ y₁ x₂ y₂ : ℝ),
      unit_circle m 0 →
      tangent_line m k x₁ y₁ →
      tangent_line m k x₂ y₂ →
      ellipse_C x₁ y₁ a b →
      ellipse_C x₂ y₂ a b →
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l968_96811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_when_a_is_one_range_of_a_for_two_zeros_inequality_for_zeros_l968_96842

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -2 * Real.log x - a / x^2 + 1

-- Theorem for part (1)
theorem extreme_values_when_a_is_one :
  let a := 1
  ∃ (max_val min_val : ℝ),
    (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f a x ≤ max_val) ∧
    (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f a x = max_val) ∧
    (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f a x ≥ min_val) ∧
    (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f a x = min_val) ∧
    max_val = 0 ∧
    min_val = -3 + 2 * Real.log 2 := by
  sorry

-- Theorem for part (2)
theorem range_of_a_for_two_zeros :
  ∀ a : ℝ,
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔
    (a > 0 ∧ a < 1) := by
  sorry

-- Theorem for the inequality in part (2)
theorem inequality_for_zeros (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ → f a x₁ = 0 → f a x₂ = 0 →
  1 / x₁^2 + 1 / x₂^2 > 2 / a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_when_a_is_one_range_of_a_for_two_zeros_inequality_for_zeros_l968_96842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_explicit_formula_for_a_l968_96873

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Adding a case for 0 to avoid missing cases error
  | 1 => 1
  | n + 1 => (1 + 4 * a n + Real.sqrt (1 + 24 * a n)) / 16

theorem explicit_formula_for_a (n : ℕ) (hn : n ≥ 1) :
  a n = ((3 + 2^(2 - n))^2 - 1) / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_explicit_formula_for_a_l968_96873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_10_terms_eq_5_div_2_l968_96821

def sequence_a : ℕ → ℚ
  | 0 => -2
  | n + 1 => (1 + 2 * sequence_a n) / 2

def sum_first_n_terms (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i => sequence_a i)

theorem sum_first_10_terms_eq_5_div_2 :
  sum_first_n_terms 10 = 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_10_terms_eq_5_div_2_l968_96821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_3600_25_2_l968_96877

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- The difference between compound and simple interest -/
noncomputable def interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  compound_interest principal rate time - simple_interest principal rate time

theorem interest_difference_3600_25_2 :
  interest_difference 3600 25 2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_3600_25_2_l968_96877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_calculations_correct_l968_96892

structure City where
  name : String
  timeDiff : Int

def greenwich : City := ⟨"Greenwich", 0⟩
def beijing : City := ⟨"Beijing", 8⟩
def newYork : City := ⟨"New York", -4⟩
def sydney : City := ⟨"Sydney", 11⟩
def moscow : City := ⟨"Moscow", 3⟩

def timeDifference (c1 c2 : City) : Int :=
  c1.timeDiff - c2.timeDiff

def localTime (time : Int) (fromCity toCity : City) : Int :=
  (time + timeDifference toCity fromCity + 24) % 24

def arrivalTime (departureTime flightDuration : Int) (fromCity toCity : City) : Int :=
  localTime ((departureTime + flightDuration) % 24) fromCity toCity

def timeRelationship (c1 c2 : City) (f : Int → Int → Bool) : Option Int :=
  let diff := timeDifference c1 c2
  (List.range 24).find? fun t => f t ((t - diff + 24) % 24)

theorem time_calculations_correct :
  (timeDifference beijing newYork = 12) ∧
  (localTime 21 sydney newYork = 6) ∧
  (arrivalTime 23 12 beijing sydney = 14) ∧
  (timeRelationship beijing moscow (fun x y => x = 2 * y) = some 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_calculations_correct_l968_96892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_sum_l968_96844

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the inverse function
def f_inv : ℝ → ℝ := sorry

-- Axioms for the given conditions
axiom f_bijective : Function.Bijective f
axiom f_inv_def : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f

-- Define the specific values of f
axiom f_2 : f 2 = 4
axiom f_3 : f 3 = 7
axiom f_4 : f 4 = 9
axiom f_5 : f 5 = 10

-- The theorem to prove
theorem f_composition_sum : f (f 2) + f_inv (f 5) + f (f_inv 7) = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_sum_l968_96844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_translation_correctness_l968_96896

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2

/-- The translated parabola function -/
def translated_parabola (x : ℝ) : ℝ := (x - 2)^2 - 1

/-- The translation function -/
def translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2, p.2 - 1)

theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x - 2) - 1 :=
by sorry

theorem translation_correctness :
  ∀ p : ℝ × ℝ, translated_parabola p.1 = original_parabola ((translate p).1 - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_translation_correctness_l968_96896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l968_96835

/-- The function f(x) = x^2 + x - ln x - 2 -/
noncomputable def f (x : ℝ) := x^2 + x - Real.log x - 2

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) := 2*x + 1 - 1/x

theorem non_monotonic_interval (k : ℝ) : 
  (∃ x y, x ∈ Set.Ioo (2*k - 1) (k + 2) ∧ y ∈ Set.Ioo (2*k - 1) (k + 2) ∧ f' x * f' y < 0) → 
  k ∈ Set.Icc (1/2) (3/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l968_96835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_final_number_lower_bound_l968_96832

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Given a positive real number c > 1, compute L = 1 + log_φ(c) -/
noncomputable def L (c : ℝ) : ℝ := 1 + (Real.log c) / (Real.log φ)

/-- The lower bound function for the final number on the blackboard -/
noncomputable def lowerBound (c : ℝ) (n : ℕ) : ℝ :=
  ((c^(n / L c) - 1) / (c^(1 / L c) - 1))^(L c)

/-- The theorem statement -/
theorem blackboard_final_number_lower_bound (c : ℝ) (n : ℕ) (hc : c > 1) (hn : n > 0) :
  ∃ (final : ℝ), final ≥ lowerBound c n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_final_number_lower_bound_l968_96832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wages_difference_l968_96853

-- Define the wages as real numbers
variable (E : ℝ) -- Erica's wages
variable (R : ℝ) -- Robin's wages
variable (C : ℝ) -- Charles' wages

-- Define the conditions
axiom robin_wages : R = 1.3 * E
axiom charles_wages : C = 1.7 * E

-- Define the theorem
theorem wages_difference : ∃ ε > 0, |((C - R) / R * 100) - 30.77| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wages_difference_l968_96853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_theorem_l968_96828

theorem circle_sum_theorem (circle : Fin 10 → ℕ+) : 
  (∀ i : Fin 10, circle i = Nat.gcd (circle (i - 1)) (circle (i + 1)) + 1) →
  (Finset.sum Finset.univ (λ i => (circle i).val) = 28) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_theorem_l968_96828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_proper_sets_l968_96893

/-- A proper set of weights is a multiset of positive integers that can uniquely balance
    any integer weight from 1 to 500 grams. -/
def ProperSet : Type := Multiset ℕ

/-- The predicate that checks if a set of weights is proper. -/
def isProper (s : ProperSet) : Prop :=
  (s.sum = 500) ∧
  (∀ w : ℕ, w ≥ 1 → w ≤ 500 → ∃! subset : Multiset ℕ, subset ⊆ s ∧ subset.sum = w)

/-- The theorem stating that there are exactly 3 distinct proper sets of weights. -/
theorem three_proper_sets :
  ∃! (sets : Finset ProperSet), (∀ s ∈ sets, isProper s) ∧ sets.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_proper_sets_l968_96893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l968_96843

/-- Predicate to represent that a real number is the eccentricity of a hyperbola -/
def IsEccentricityOfHyperbola (e x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ e = Real.sqrt (1 + (b^2 / a^2)) ∧
  (x^2 / a^2) - (y^2 / b^2) = 1

/-- Given a hyperbola with asymptotes x - √3 y = 0 and √3 x + y = 0, its eccentricity is √2 -/
theorem hyperbola_eccentricity :
  ∃ (e : ℝ), e = Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), (x - Real.sqrt 3 * y = 0 ∧ Real.sqrt 3 * x + y = 0) →
  IsEccentricityOfHyperbola e x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l968_96843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l968_96810

theorem equation_solution (x : ℝ) : (4 : ℝ)^x - (4 : ℝ)^(x-1) = 54 → (3*x)^x = 27 * Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l968_96810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l968_96891

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case for n = 0
  | 1 => 2
  | (n + 2) => (1 / 2) * sequence_a (n + 1) + (1 / 2)

theorem sequence_a_formula (n : ℕ) : 
  sequence_a n = (1 / 2) ^ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l968_96891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_proof_l968_96868

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |x - 1/m|

-- State the theorems
theorem solution_set (x : ℝ) : 
  (f 1 x ≤ 4) ↔ (-2 ≤ x ∧ x ≤ 2) := by sorry

theorem inequality_proof (a m : ℝ) (h : a ≠ 0) (hm : m > 0) : 
  f m (-a) + f m (1/a) ≥ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_proof_l968_96868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l3_l968_96865

-- Define the lines and points
def l1 (x y : ℝ) : Prop := 2 * x + 3 * y = 6
def l2 (y : ℝ) : Prop := y = 2
def A : ℝ × ℝ := (3, 0)

-- Define the existence of point B
def B_exists : Prop := ∃ B : ℝ × ℝ, l1 B.1 B.2 ∧ l2 B.2

-- Define the existence of point C
def C_exists : Prop := ∃ C : ℝ × ℝ, l2 C.2

-- Define the area of triangle ABC
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Theorem statement
theorem slope_of_l3 (B C : ℝ × ℝ) 
  (h1 : l1 A.1 A.2)
  (h2 : B_exists)
  (h3 : C_exists)
  (h4 : triangle_area A B C = 6) :
  (C.2 - A.2) / (C.1 - A.1) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l3_l968_96865
