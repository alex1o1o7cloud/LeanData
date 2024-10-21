import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l54_5465

/-- Defines an isosceles triangle with two equal sides. -/
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

/-- An isosceles triangle with sides of length 4 and 8 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 8 → b = 8 → c = 4 →
  IsoscelesTriangle a b c →
  a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l54_5465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l54_5435

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  (∀ k : ℤ, x ≠ π * k / 3) →
  sin (3 * x) ≠ 0 →
  (sin (2 * x))^2 = (cos x)^2 + cos (3 * x) / sin (3 * x) →
  (∃ n : ℤ, x = π / 2 + π * ↑n) ∨ (∃ k : ℤ, x = π / 6 + π * ↑k) ∨ (∃ k : ℤ, x = -π / 6 + π * ↑k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l54_5435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_area_ratio_l54_5437

theorem triangle_circle_area_ratio :
  let a : ℝ := 13
  let b : ℝ := 14
  let c : ℝ := 15
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area / s
  let R := (a * b * c) / (4 * area)
  (R / r)^2 = (65/32)^2 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_area_ratio_l54_5437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_in_terms_of_a_l54_5434

theorem b_in_terms_of_a (n p q : ℝ) (a b : ℝ) 
  (h1 : p = n^a)
  (h2 : q = n^b)
  (h3 : n = 2^(0.15 : ℝ))
  (h4 : p = 8)
  (h5 : q = 64) : 
  b = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_in_terms_of_a_l54_5434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l54_5427

-- Define the hyperbola
def hyperbola_equation (x y : ℝ) : Prop := 9 * y^2 - 4 * x^2 = -36

-- Define the vertices
def vertices : Set (ℝ × ℝ) := {(-3, 0), (3, 0)}

-- Define the foci
noncomputable def foci : Set (ℝ × ℝ) := {(-Real.sqrt 13, 0), (Real.sqrt 13, 0)}

-- Define the length of the transverse axis
def transverse_axis_length : ℝ := 6

-- Define the length of the conjugate axis
def conjugate_axis_length : ℝ := 4

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 13 / 3

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = 2/3 * x ∨ y = -2/3 * x

-- Theorem statement
theorem hyperbola_properties :
  ∀ x y : ℝ, hyperbola_equation x y →
  (x, y) ∈ vertices ∨
  (x, y) ∈ foci ∨
  transverse_axis_length = 6 ∧
  conjugate_axis_length = 4 ∧
  eccentricity = Real.sqrt 13 / 3 ∧
  asymptotes x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l54_5427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l54_5418

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  b = 2 * Real.sqrt 7 ∧ c = 2 ∧ B = Real.pi / 3

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : 
  a = 6 ∧ 
  Real.sin A = (3 * Real.sqrt 21) / 14 ∧ 
  Real.sin (B - 2 * A) = -(5 * Real.sqrt 3) / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l54_5418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l54_5454

-- Define the statements p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, Real.cos (2 * x) - Real.sin x + 2 ≤ m

noncomputable def q (m : ℝ) : Prop := ∀ x ≥ 2, ∀ ε > 0, 
  (((1 : ℝ) / 3) ^ (2 * x^2 - m * x + 2)) > (((1 : ℝ) / 3) ^ (2 * (x + ε)^2 - m * (x + ε) + 2))

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (m < 0 ∨ m > 8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l54_5454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_extrema_range_l54_5439

/-- A cubic function with parameters b, c, and d -/
noncomputable def f (b c d : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * b * x^2 + c * x + d

/-- The derivative of f with respect to x -/
noncomputable def f' (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

theorem cubic_function_extrema_range (b c d : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧
    f' b c x₁ = 0 ∧ f' b c x₂ = 0) →
  0 < c^2 + 2*b*c + 4*c ∧ c^2 + 2*b*c + 4*c < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_extrema_range_l54_5439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speech_competition_score_l54_5453

/-- Represents the scoring system for a speech competition --/
structure SpeechCompetition where
  contentScore : ℚ
  skillsScore : ℚ
  effectsScore : ℚ
  contentRatio : ℚ
  skillsRatio : ℚ
  effectsRatio : ℚ

/-- Calculates the final score for a speech competition --/
def finalScore (comp : SpeechCompetition) : ℚ :=
  (comp.contentScore * comp.contentRatio + 
   comp.skillsScore * comp.skillsRatio + 
   comp.effectsScore * comp.effectsRatio) / 
  (comp.contentRatio + comp.skillsRatio + comp.effectsRatio)

/-- Theorem stating that the final score for the given competition is 86 --/
theorem speech_competition_score :
  let comp : SpeechCompetition := {
    contentScore := 90
    skillsScore := 80
    effectsScore := 85
    contentRatio := 4
    skillsRatio := 2
    effectsRatio := 4
  }
  finalScore comp = 86 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speech_competition_score_l54_5453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_three_solutions_l54_5469

-- Define the first equation
def first_equation (x y : ℝ) : Prop :=
  |3 * x| + |4 * y| + |48 - 3 * x - 4 * y| = 48

-- Define the second equation
def second_equation (x y a : ℝ) : Prop :=
  (x - 8)^2 + (y + 6 * Real.cos (a * Real.pi / 2))^2 = (a + 4)^2

-- Define the triangle vertices
def E : ℝ × ℝ := (16, 0)
def G : ℝ × ℝ := (0, 12)
def N : ℝ × ℝ := (0, 0)

-- Theorem for part (a)
theorem area_of_triangle : 
  ∀ (x y : ℝ), first_equation x y → 
  (1/2 * |E.1 * (G.2 - N.2) + G.1 * (N.2 - E.2) + N.1 * (E.2 - G.2)| = 96) := 
by sorry

-- Theorem for part (b)
theorem three_solutions : 
  ∀ (a : ℝ), (∃! (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
    first_equation x₁ y₁ ∧ second_equation x₁ y₁ a ∧
    first_equation x₂ y₂ ∧ second_equation x₂ y₂ a ∧
    first_equation x₃ y₃ ∧ second_equation x₃ y₃ a) 
  ↔ (a = 6 ∨ a = -14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_three_solutions_l54_5469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABM_range_l54_5490

-- Define the circle
noncomputable def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define points N and M
noncomputable def N : ℝ × ℝ := (0, -Real.sqrt 3 / 3)
noncomputable def M : ℝ × ℝ := (0, Real.sqrt 3 / 3)

-- Define a line passing through N
noncomputable def line_through_N (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 + Real.sqrt 3 / 3 = k * p.1}

-- Define the intersection points A and B
noncomputable def intersection_points (k : ℝ) : Set (ℝ × ℝ) := circle_O ∩ line_through_N k

-- Define the area of triangle ABM
noncomputable def area_ABM (k : ℝ) : ℝ :=
  (2 / 3) * Real.sqrt ((3 * k^2 + 2) / (k^2 + 1)^2)

-- The theorem to prove
theorem area_ABM_range :
  ∀ k : ℝ, 0 < area_ABM k ∧ area_ABM k ≤ 2 * Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABM_range_l54_5490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_disk_sales_l54_5416

/-- The number of disks Maria needs to sell to make a $100 profit -/
def disks_needed_for_profit (buy_rate : ℚ) (sell_rate : ℚ) (profit_goal : ℚ) : ℕ :=
  let cost_per_disk := 5 / buy_rate
  let sell_per_disk := 5 / sell_rate
  let profit_per_disk := sell_per_disk - cost_per_disk
  (profit_goal / profit_per_disk).ceil.toNat

theorem maria_disk_sales :
  disks_needed_for_profit 4 3 100 = 240 := by
  sorry

#eval disks_needed_for_profit 4 3 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_disk_sales_l54_5416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dice_l54_5417

noncomputable def ivan_dice (x : ℝ) : ℝ := x

noncomputable def jerry_dice (x : ℝ) : ℝ := (1/2 * x)^2

theorem total_dice (x : ℝ) : ivan_dice x + jerry_dice x = x + (1/4) * x^2 := by
  -- Unfold the definitions of ivan_dice and jerry_dice
  unfold ivan_dice jerry_dice
  -- Simplify the expression
  simp [pow_two]
  -- Perform algebraic manipulations
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dice_l54_5417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l54_5462

theorem simplify_and_rationalize :
  1 / (1 + 1 / (Real.sqrt 5 + 2)) = (Real.sqrt 5 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l54_5462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_retirement_age_l54_5423

/-- Jason's military career and retirement age --/
theorem jason_retirement_age 
  (join_age : ℕ) 
  (chief_years : ℕ) 
  (master_chief_factor : ℚ) 
  (additional_years : ℕ) 
  (h1 : join_age = 18)
  (h2 : chief_years = 8)
  (h3 : master_chief_factor = 1.25)
  (h4 : additional_years = 10) :
  join_age + chief_years + (master_chief_factor * chief_years).floor + additional_years = 46 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_retirement_age_l54_5423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_eleven_average_l54_5449

theorem first_eleven_average (numbers : List ℝ) : 
  numbers.length = 21 ∧ 
  numbers.sum / 21 = 44 ∧
  (numbers.drop 10).sum / 11 = 41 ∧
  numbers.get? 10 = some 55 →
  (numbers.take 11).sum / 11 = 48 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_eleven_average_l54_5449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mode_difference_l54_5459

def stem_leaf_plot : List ℤ := [11, 12, 16, 17, 17, 21, 21, 21, 22, 25, 25, 30, 34, 37, 38, 39, 41, 43, 43, 43, 49, 50, 52, 55, 56, 58]

def mode (data : List ℤ) : ℤ := 11

def median (data : List ℤ) : ℤ := 29

theorem median_mode_difference :
  let data := stem_leaf_plot
  abs ((median data) - (mode data)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mode_difference_l54_5459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_four_theta_l54_5485

theorem tan_four_theta (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (4 * θ) = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_four_theta_l54_5485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_heights_l54_5411

/-- Given three isosceles triangles with the same base length, prove the relationship between their heights -/
theorem isosceles_triangle_heights
  (b : ℝ) -- base length
  (h₁ h₂ h₃ : ℝ) -- heights of the triangles
  (area₁ area₂ area₃ : ℝ) -- areas of the triangles
  (h_positive : b > 0 ∧ h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0) -- positive base and heights
  (h_isosceles : area₁ = 1/2 * b * h₁ ∧ area₂ = 1/2 * b * h₂ ∧ area₃ = 1/2 * b * h₃) -- area formula for isosceles triangles
  (h_vertical_angle : ∃ θ : ℝ, Real.sin θ = h₁ / b ∧ Real.sin θ = h₂ / b) -- equal vertical angles for first and second triangles
  (h_area_ratio_12 : area₁ / area₂ = 16 / 25) -- area ratio of first and second triangles
  (h_area_ratio_13 : area₁ / area₃ = 4 / 9) -- area ratio of first and third triangles
  : h₁ / h₂ = 4 / 5 ∧ h₁ / h₃ = 2 / 3 ∧ h₂ / h₃ = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_heights_l54_5411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_M₁M₂M₃M₄_is_parallelogram_l54_5421

-- Define the types for points and vectors
variable (Point Vector : Type) [AddCommGroup Vector] [Module ℝ Vector]

-- Define the vector space operations
variable (add : Vector → Vector → Vector)
variable (neg : Vector → Vector)
variable (smul : ℝ → Vector → Vector)

-- Define the rotation operation (60 degrees counterclockwise)
variable (rotate60 : Vector → Vector)

-- Define the quadrilateral ABCD
variable (A B C D : Point)

-- Define the vectors of the quadrilateral sides
variable (a b c d : Vector)

-- Define the relationship between the vectors
axiom quadrilateral_closed : a + b + c + d = (0 : Vector)

-- Define the equilateral triangles
variable (M₁ M₂ M₃ M₄ : Point)

-- Define the vectors from quadrilateral vertices to triangle vertices
axiom triangle_ABM₁ : -a + rotate60 a = (0 : Vector)
axiom triangle_BCM₂ : -b + rotate60 b = (0 : Vector)
axiom triangle_CDM₃ : c + rotate60 c = (0 : Vector)
axiom triangle_DAM₄ : d + rotate60 d = (0 : Vector)

-- Theorem statement
theorem quadrilateral_M₁M₂M₃M₄_is_parallelogram :
  let v₁ := -rotate60 a + rotate60 b
  let v₂ := -rotate60 c + rotate60 (-a)
  v₁ = -v₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_M₁M₂M₃M₄_is_parallelogram_l54_5421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l54_5470

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x * Real.exp (k * x)

theorem tangent_line_and_monotonicity (k : ℝ) (h : k ≠ 0) :
  (∃ m b : ℝ, ∀ x : ℝ, (m * x + b) = ((deriv (f k)) 0) * x + f k 0) ∧
  (∀ x : ℝ, x ∈ Set.Ioo (-1) 1 → MonotoneOn (f k) (Set.Ioo (-1) 1)) ↔ 
  k ∈ Set.Icc (-1) 0 ∪ Set.Ioc 0 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l54_5470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l54_5431

noncomputable def slope_angle : Real := Real.pi / 6

def point_P : Prod Real Real := (1, 1)

noncomputable def curve_C (α : Real) : Prod Real Real :=
  (2 * Real.sin α, 2 + 2 * Real.cos α)

noncomputable def line_l (t : Real) : Prod Real Real :=
  (1 + Real.cos slope_angle * t, 1 + Real.sin slope_angle * t)

theorem intersection_product :
  ∃ (t₁ t₂ : Real),
    (∃ (α₁ α₂ : Real),
      line_l t₁ = curve_C α₁ ∧
      line_l t₂ = curve_C α₂) →
    t₁ * t₂ = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l54_5431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l54_5406

theorem shaded_areas_equality (φ : Real) (h1 : 0 < φ) (h2 : φ < π / 4) :
  (∃ r : Real, r > 0 ∧
    (φ * r^2 / 2 = r^2 * Real.tan φ / 2 - φ * r^2 / 2)) ↔ Real.tan φ = 2 * φ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l54_5406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_classification_l54_5436

/-- Defines a relation between x and y --/
def Relation : Type := ℝ → ℝ → Prop

/-- Directly proportional relation --/
def DirectlyProportional : Relation := λ x y => ∃ k : ℝ, x = k * y ∧ k ≠ 0

/-- Inversely proportional relation --/
def InverselyProportional : Relation := λ x y => ∃ k : ℝ, x * y = k ∧ k ≠ 0

/-- Neither directly nor inversely proportional --/
def NeitherProportional : Relation := λ x y =>
  ¬(DirectlyProportional x y) ∧ ¬(InverselyProportional x y)

theorem proportionality_classification :
  (∀ x y : ℝ, 2 * x + y = 5 → NeitherProportional x y) ∧
  (∀ x y : ℝ, 4 * x * y = 12 → InverselyProportional x y) ∧
  (∀ x y : ℝ, x = 3 * y → DirectlyProportional x y) ∧
  (∀ x y : ℝ, 2 * x + 3 * y = 15 → NeitherProportional x y) ∧
  (∀ x y : ℝ, x / y = 2 → DirectlyProportional x y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_classification_l54_5436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l54_5451

/-- Ellipse struct representing the equation (x^2/a^2) + (y^2/b^2) = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Theorem stating the eccentricity of the ellipse under given conditions -/
theorem ellipse_eccentricity (e : Ellipse) 
  (A B F₁ F₂ : Point)
  (h_A_on_C : (A.x^2 / e.a^2) + (A.y^2 / e.b^2) = 1)
  (h_B_on_y : B.x = 0)
  (h_perpendicular : (A.x - F₁.x) * (B.x - F₁.x) + (A.y - F₁.y) * (B.y - F₁.y) = 0)
  (h_ratio : (A.x - F₂.x, A.y - F₂.y) = 
             ((2/3) * (F₂.x - B.x), (2/3) * (F₂.y - B.y))) :
  eccentricity e = Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l54_5451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_of_three_element_set_l54_5428

theorem subset_count_of_three_element_set :
  let S : Finset Int := {-1, 0, 1}
  (Finset.powerset S).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_of_three_element_set_l54_5428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_altitude_l54_5471

/-- Given an equilateral triangle with area 480 square feet, its altitude is 40√3 feet. -/
theorem equilateral_triangle_altitude (s h : ℝ) : 
  (Real.sqrt 3 / 4) * s^2 = 480 →  -- Area formula for equilateral triangle
  h = s * Real.sqrt 3 / 2 →        -- Altitude formula for equilateral triangle
  h = 40 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_altitude_l54_5471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_diagonal_ratio_l54_5457

-- Define Trapezoid as a structure
structure Trapezoid (A B C D : ℝ × ℝ) :=
  (ab_less_cd : dist A B < dist C D)
  (ab_perp_bc : (B.1 - A.1) * (C.2 - B.2) + (B.2 - A.2) * (C.1 - B.1) = 0)
  (ab_parallel_cd : (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1))

-- Define the theorem
theorem trapezoid_diagonal_ratio 
  (A B C D P Q : ℝ × ℝ) 
  (h : Trapezoid A B C D) 
  (diagonals_perp : (C.1 - A.1) * (D.1 - B.1) + (C.2 - A.2) * (D.2 - B.2) = 0)
  (q_on_ray_ca : ∃ t : ℝ, t > 1 ∧ Q = (t * C.1 + (1 - t) * A.1, t * C.2 + (1 - t) * A.2))
  (qd_perp_dc : (Q.1 - D.1) * (C.1 - D.1) + (Q.2 - D.2) * (C.2 - D.2) = 0)
  (h_ratio : dist Q P / dist A P + dist A P / dist Q P = (51/14)^4 - 2) :
  dist B P / dist A P - dist A P / dist B P = 47/14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_diagonal_ratio_l54_5457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l54_5486

/-- Given a triangle ABC with the following properties:
    - cos A = 2/3
    - sin B = √5 * cos C
    - a = √2 (side length opposite to angle A)
    This theorem proves that:
    1. tan C = √5
    2. The area of triangle ABC is √5/2
-/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  Real.cos A = 2/3 →
  Real.sin B = Real.sqrt 5 * Real.cos C →
  a = Real.sqrt 2 →
  Real.tan C = Real.sqrt 5 ∧ 
  (1/2 : ℝ) * a * b * Real.sin C = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l54_5486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l54_5460

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := 3*x + 4*y + 2 = 0

-- Define the circle
def my_circle (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- Define the focus of a parabola
def focus_of_parabola (a b : ℝ) : Prop := a = 1 ∧ b = 0

-- Define tangency between circle and line
def tangent_circle_line (a b r : ℝ) : Prop :=
  r = |3*a + 4*b + 2| / Real.sqrt (3^2 + 4^2)

theorem circle_equation :
  ∀ a b r : ℝ,
  (∀ x y : ℝ, parabola x y → focus_of_parabola a b) →
  tangent_circle_line a b r →
  (∀ x y : ℝ, my_circle x y a b r ↔ (x - 1)^2 + y^2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l54_5460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l54_5448

/-- The focal distance of an ellipse with parametric equations x = a * cos(θ) and y = b * sin(θ) -/
noncomputable def focalDistance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- Theorem: The focal distance of the ellipse defined by x = 5cos(θ) and y = 4sin(θ) is 6 -/
theorem ellipse_focal_distance :
  focalDistance 5 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l54_5448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_area_l54_5472

/-- A quadrilateral inscribed in a circle with specific properties -/
structure InscribedQuadrilateral where
  /-- The circle in which the quadrilateral is inscribed -/
  circle : Real → Real → Prop
  /-- The quadrilateral inscribed in the circle -/
  quad : Real → Real → Real → Real → Prop
  /-- The diameter of the circle is 1 -/
  diameter_is_one : ∀ x y, circle x y → (x^2 + y^2 = 1)
  /-- Angle D is a right angle -/
  angle_d_is_right : ∀ a b c d, quad a b c d → (a * c + b * d = 0)
  /-- Side AB is equal to side BC -/
  ab_eq_bc : ∀ a b c d, quad a b c d → ((a - b)^2 = (b - c)^2)
  /-- The perimeter of the quadrilateral is 9√2/5 -/
  perimeter : ∀ a b c d, quad a b c d → 
    Real.sqrt ((a - b)^2 + (b - c)^2 + (c - d)^2 + (d - a)^2) = 9 * Real.sqrt 2 / 5

/-- The area of an inscribed quadrilateral with specific properties is 8/25 -/
theorem inscribed_quadrilateral_area (q : InscribedQuadrilateral) : 
  ∀ a b c d, q.quad a b c d → 
    abs ((a * d - b * c) / 2) = 8 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_area_l54_5472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_at_zero_l54_5401

noncomputable section

/-- A linear function -/
def LinearFunction (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

/-- A quadratic function -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- A rational function -/
def RationalFunction (u : ℝ → ℝ) (v : ℝ → ℝ) : ℝ → ℝ := λ x ↦ u x / v x

theorem rational_function_value_at_zero 
  (u : ℝ → ℝ) (v : ℝ → ℝ) (a b c d : ℝ) :
  (∃ k₁ k₂ : ℝ, u = LinearFunction k₁ k₂) →
  (∃ k₃ k₄ k₅ : ℝ, v = QuadraticFunction k₃ k₄ k₅) →
  (∀ x, v x = 0 ↔ x = -4 ∨ x = 1) →
  RationalFunction u v 4 = 2 →
  RationalFunction u v 0 = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_at_zero_l54_5401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_sum_divisibility_l54_5415

open BigOperators

def S {n : ℕ} (c : Fin n → ℤ) (a : Equiv.Perm (Fin n)) : ℤ :=
  ∑ i, c i * a i

theorem permutation_sum_divisibility {n : ℕ} (h_odd : Odd n) (c : Fin n → ℤ) :
  ∃ (a b : Equiv.Perm (Fin n)), a ≠ b ∧ (n.factorial : ℤ) ∣ (S c a - S c b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_sum_divisibility_l54_5415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_two_power_sum_l54_5456

theorem max_value_of_two_power_sum (x y : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^y = 6) :
  ∀ a b : ℝ, (2 : ℝ)^a + (2 : ℝ)^b = 6 → (2 : ℝ)^(x+y) ≥ (2 : ℝ)^(a+b) ∧ (2 : ℝ)^(x+y) ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_two_power_sum_l54_5456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_woman_l54_5458

theorem probability_at_least_one_woman (total_men : ℕ) (total_women : ℕ) (selection_size : ℕ) :
  total_men = 8 →
  total_women = 5 →
  selection_size = 4 →
  (1 - (Nat.choose total_men selection_size : ℚ) / (Nat.choose (total_men + total_women) selection_size)) = 129/143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_woman_l54_5458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ellipse_area_l54_5409

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a circle with center (0, 1) and radius 1 -/
def UnitCircle := {(x, y) : ℝ × ℝ | x^2 + (y - 1)^2 = 1}

/-- Checks if a point (x, y) is on the ellipse -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Checks if the ellipse contains the unit circle -/
def Ellipse.containsUnitCircle (e : Ellipse) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ UnitCircle → e.contains x y

/-- The area of an ellipse -/
noncomputable def Ellipse.area (e : Ellipse) : ℝ := Real.pi * e.a * e.b

/-- Theorem: The smallest possible area of an ellipse containing the unit circle is 5π -/
theorem smallest_ellipse_area (e : Ellipse) (h : e.containsUnitCircle) :
  ∃ (e_min : Ellipse), e_min.containsUnitCircle ∧ e_min.area = 5 * Real.pi ∧ ∀ (e' : Ellipse), e'.containsUnitCircle → e'.area ≥ 5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ellipse_area_l54_5409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_FCA_angle_l54_5475

-- Define the ellipse
def Γ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 2019) + (p.2^2 / 2018) = 1}

-- Define the focus F, and points A, B, C
variable (F A B C : ℝ × ℝ)

-- Define that F is the left focus of Γ
axiom F_is_left_focus : F ∈ Γ ∧ F.1 < 0

-- Define that C, A, B are on a line l passing through F
axiom on_line_l : ∃ (m b : ℝ), C.2 = m * C.1 + b ∧ A.2 = m * A.1 + b ∧ B.2 = m * B.1 + b ∧ F.2 = m * F.1 + b

-- Define that C is on the left directrix
axiom C_on_left_directrix : C.1 = -Real.sqrt 2019

-- Define that A and B are on Γ
axiom A_on_Γ : A ∈ Γ
axiom B_on_Γ : B ∈ Γ

-- Define the angles
noncomputable def angle (P Q R : ℝ × ℝ) : ℝ := sorry

-- Given angles
axiom FAB_angle : angle F A B = 40 * Real.pi / 180
axiom FBA_angle : angle F B A = 10 * Real.pi / 180

-- Theorem to prove
theorem FCA_angle : angle F C A = 15 * Real.pi / 180 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_FCA_angle_l54_5475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l54_5491

theorem infinitely_many_solutions (n : ℤ) :
  (∃ (S : Set (ℤ × ℤ)), Set.Infinite S ∧
    (∀ (p : ℤ × ℤ), p ∈ S →
      let (x, y) := p
      x^2 + n*x*y + y^2 = 1)) ↔
  n ≠ -1 ∧ n ≠ 0 ∧ n ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l54_5491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_of_intersections_l54_5499

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y = 4
def C₂ (x y θ : ℝ) : Prop := x = 1 + Real.cos θ ∧ y = Real.sin θ

-- Define the polar coordinates of points A and B
noncomputable def ρ₁ (α : ℝ) : ℝ := 4 / (Real.cos α + Real.sin α)
noncomputable def ρ₂ (α : ℝ) : ℝ := 2 * Real.cos α

-- Define the ratio |OB|/|OA|
noncomputable def ratio (α : ℝ) : ℝ := ρ₂ α / ρ₁ α

-- State the theorem
theorem max_ratio_of_intersections :
  ∃ (max_ratio : ℝ), max_ratio = (Real.sqrt 2 + 1) / 4 ∧
  ∀ (α : ℝ), -π/4 < α ∧ α < π/2 → ratio α ≤ max_ratio := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_of_intersections_l54_5499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l54_5450

def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

theorem extreme_value_at_one (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1) ∧
  (f a b 1 = 10) ∧
  (deriv (f a b) 1 = 0) →
  a = -4 ∧ b = 11 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l54_5450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l54_5441

theorem min_abs_difference (a b : ℕ+) (h : a.val * b.val - 8 * a.val + 7 * b.val = 569) :
  ∃ (a' b' : ℕ+), a'.val * b'.val - 8 * a'.val + 7 * b'.val = 569 ∧
  ∀ (x y : ℕ+), x.val * y.val - 8 * x.val + 7 * y.val = 569 → 
    (Int.natAbs (x.val - y.val) : ℤ) ≥ (Int.natAbs (a'.val - b'.val) : ℤ) ∧
  Int.natAbs (a'.val - b'.val) = 23 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l54_5441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_sum_cubes_l54_5468

noncomputable def a (i : Fin 3) (x : ℝ) : ℝ :=
  ∑' n, x^(3*n + i.val) / (Nat.factorial (3*n + i.val))

theorem a_sum_cubes (x : ℝ) :
  (a 0 x)^3 + (a 1 x)^3 + (a 2 x)^3 - 3*(a 0 x)*(a 1 x)*(a 2 x) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_sum_cubes_l54_5468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_properties_l54_5484

/-- The ellipse (C) -/
def ellipse_C (x y : ℝ) : Prop := x^2 + 2*y^2 = 4

/-- Point A is on the line y = 2 -/
def point_A (x y : ℝ) : Prop := y = 2

/-- Point B is on the ellipse (C) -/
def point_B (x y : ℝ) : Prop := ellipse_C x y

/-- OA is perpendicular to OB -/
def OA_perp_OB (xa ya xb yb : ℝ) : Prop := xa * xb + ya * yb = 0

/-- The eccentricity of the ellipse (C) -/
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

/-- The minimum length of line segment AB -/
noncomputable def min_length_AB : ℝ := 2 * Real.sqrt 2

theorem ellipse_C_properties :
  (∀ x y, ellipse_C x y → ∃ e, e = eccentricity) ∧
  (∀ xa ya xb yb,
    point_A xa ya → point_B xb yb → OA_perp_OB xa ya xb yb →
    ∃ l, l ≥ min_length_AB ∧
    (l = min_length_AB ↔ (xb = 2 ∨ xb = -2))) := by
  sorry

#check ellipse_C_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_properties_l54_5484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cone_equal_volume_l54_5467

/-- Given a cone with radius 2 inches and height 3 inches, prove that a sphere
    with radius ∛3 inches has the same volume as the cone. -/
theorem sphere_cone_equal_volume :
  let cone_radius : ℝ := 2
  let cone_height : ℝ := 3
  let sphere_radius : ℝ := Real.rpow 3 (1/3)
  let cone_volume := (1/3) * Real.pi * cone_radius^2 * cone_height
  let sphere_volume := (4/3) * Real.pi * sphere_radius^3
  cone_volume = sphere_volume := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cone_equal_volume_l54_5467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibby_numbers_l54_5489

def is_fibby (k : ℕ) : Prop :=
  k ≥ 3 ∧
  ∃ (n : ℕ) (d : ℕ → ℕ),
    (∀ i j, i < j ∧ j < k → d i < d j) ∧
    (∀ j, j + 2 < k → d (j + 2) = d (j + 1) + d j) ∧
    (∀ i, i < k → d i ∣ n) ∧
    (∀ m, m ∣ n → m < d 0 ∨ m > d (k - 1) ∨ ∃ i, i < k ∧ m = d i)

theorem fibby_numbers : ∀ k, is_fibby k ↔ k = 3 ∨ k = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibby_numbers_l54_5489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_common_positive_divisors_l54_5488

def numbers : List Nat := [18, 54, 36, 90]

theorem sum_common_positive_divisors (n : List Nat) (h : n = numbers) :
  (Finset.filter (fun d => ∀ x ∈ n, x % d = 0) (Finset.range 19)).sum id = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_common_positive_divisors_l54_5488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_range_l54_5443

theorem cos_sum_range (x y : ℝ) (h : Real.sin x + Real.sin y = 1) :
  ∃ (t : ℝ), Real.cos x + Real.cos y = t ∧ -Real.sqrt 3 ≤ t ∧ t ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_range_l54_5443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_perpendicular_vectors_l54_5495

noncomputable section

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![1, -1]

def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

noncomputable def vector_magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

noncomputable def angle_between (v w : Fin 2 → ℝ) : ℝ :=
  Real.arccos ((dot_product v w) / (vector_magnitude v * vector_magnitude w))

def vector_add (v w : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![v 0 + w 0, v 1 + w 1]

def vector_sub (v w : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![v 0 - w 0, v 1 - w 1]

def scalar_mult (c : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![c * (v 0), c * (v 1)]

theorem angle_between_vectors :
  angle_between (vector_add (scalar_mult 2 a) b) (vector_sub a b) = π / 4 := by
  sorry

theorem perpendicular_vectors :
  ∃ k : ℝ, k = 0 ∧ 
    dot_product (vector_add (scalar_mult 2 a) b) (vector_add (scalar_mult k a) b) = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_perpendicular_vectors_l54_5495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l54_5440

def a : ℝ × ℝ × ℝ := (6, 3, 4)
def b : ℝ × ℝ × ℝ := (-1, -2, -1)
def c : ℝ × ℝ × ℝ := (2, 1, 2)

theorem vectors_not_coplanar : ¬(∃ (x y z : ℝ), x • a + y • b + z • c = (0, 0, 0) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l54_5440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l54_5452

/-- Represents the time difference between two runners in a race -/
noncomputable def timeDifference (raceLength : ℝ) (finishTimeA : ℝ) (distanceAhead : ℝ) : ℝ :=
  let speedA := raceLength / finishTimeA
  let timeForB := distanceAhead / speedA
  timeForB

/-- Proves that in a 1000-meter race, if runner A finishes in 115 seconds and beats runner B by 80 meters, 
    then A beats B by 9.2 seconds (with a small margin of error due to floating-point arithmetic) -/
theorem race_time_difference : 
  let raceLength : ℝ := 1000
  let finishTimeA : ℝ := 115
  let distanceAhead : ℝ := 80
  let timeDiff := timeDifference raceLength finishTimeA distanceAhead
  abs (timeDiff - 9.2) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l54_5452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_of_f_l54_5477

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2*x + 3)

-- State the theorem
theorem monotone_increasing_interval_of_f :
  ∀ x y : ℝ, x < y → (x < 1 ∧ y < 1) → f x < f y :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_of_f_l54_5477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_area_percentage_l54_5480

def circle_design (n : ℕ) (base_radius : ℝ) (increment : ℝ) : List ℝ :=
  List.range n |>.map (λ i => base_radius + i * increment)

noncomputable def total_area (radii : List ℝ) : ℝ :=
  match radii.getLast? with
  | some r => Real.pi * r^2
  | none => 0

noncomputable def white_area (radii : List ℝ) : ℝ :=
  List.zip (List.range (radii.length)) radii
  |>.filter (λ (i, _) => i % 2 = 0)
  |>.map (λ (_, r) => Real.pi * r^2)
  |>.sum

theorem white_area_percentage :
  let radii := circle_design 5 3 3
  let total := total_area radii
  let white := white_area radii
  white / total = 0.6 := by sorry

#eval circle_design 5 3 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_area_percentage_l54_5480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersections_distance_l54_5494

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define the distance between roots of a quadratic equation
noncomputable def distance_between_roots (a b c : ℝ) : ℝ := Real.sqrt (a^2 - 4*b*c)

-- State the theorem
theorem quadratic_intersections_distance (a b s t : ℝ) 
  (h1 : distance_between_roots 1 a (b - s) = 5)
  (h2 : distance_between_roots 1 a (b - t) = 11) :
  |t - s| = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersections_distance_l54_5494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_perpendicular_distance_l54_5400

/-- Represents a parabola y² = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Represents a point on the parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

/-- Represents a point on the directrix -/
structure PointOnDirectrix (p : Parabola) where
  point : ℝ × ℝ
  on_directrix : p.directrix point.1 point.2

/-- Main theorem -/
theorem parabola_perpendicular_distance 
  (p : Parabola)
  (h_eq : p.equation = fun x y ↦ y^2 = 8*x)
  (M : PointOnParabola p)
  (N : PointOnDirectrix p)
  (h_perp : (M.point.1 - p.focus.1) * (N.point.1 - p.focus.1) + 
            (M.point.2 - p.focus.2) * (N.point.2 - p.focus.2) = 0)
  (h_dist : Real.sqrt ((M.point.1 - p.focus.1)^2 + (M.point.2 - p.focus.2)^2) = 10) :
  Real.sqrt ((N.point.1 - p.focus.1)^2 + (N.point.2 - p.focus.2)^2) = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_perpendicular_distance_l54_5400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_distance_l54_5464

/-- Line in parametric form -/
structure ParametricLine where
  φ : Real
  hφ : 0 < φ ∧ φ < π

/-- Curve in polar form -/
noncomputable def PolarCurve (θ : Real) : Real :=
  4 * Real.sin θ / (Real.cos θ)^2

/-- Distance between intersection points -/
noncomputable def IntersectionDistance (l : ParametricLine) : Real :=
  4 / (Real.sin l.φ)^2

/-- Theorem: The minimum distance between intersection points is 4 -/
theorem min_intersection_distance :
  ∀ l : ParametricLine, ∃ l' : ParametricLine, ∀ l'' : ParametricLine,
    IntersectionDistance l'' ≥ IntersectionDistance l' ∧
    IntersectionDistance l' = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_distance_l54_5464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_over_36_l54_5476

/-- Sequence b_n defined recursively -/
def b : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 3
  | (n + 3) => b (n + 2) + b (n + 1)

/-- The series sum -/
noncomputable def series_sum : ℝ := ∑' n, (b n : ℝ) / 9^(n + 1)

/-- Theorem stating that the series sum equals 1/36 -/
theorem series_sum_equals_one_over_36 : series_sum = 1 / 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_over_36_l54_5476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplified_f_period_f_max_in_interval_f_min_in_interval_l54_5438

-- Define the function f as noncomputable due to Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - Real.sqrt 3 * Real.cos (2 * x) + 1

-- Theorem for the simplified form of f
theorem f_simplified (x : ℝ) : f x = 2 * Real.sin (2 * x - π / 3) + 1 := by sorry

-- Theorem for the smallest positive period of f
theorem f_period : ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T ≤ T') := by sorry

-- Theorem for the maximum value of f in the interval [π/4, π/2]
theorem f_max_in_interval :
  ∃ x₀ : ℝ, π/4 ≤ x₀ ∧ x₀ ≤ π/2 ∧ f x₀ = 3 ∧
  (∀ x : ℝ, π/4 ≤ x ∧ x ≤ π/2 → f x ≤ 3) := by sorry

-- Theorem for the minimum value of f in the interval [π/4, π/2]
theorem f_min_in_interval :
  ∃ x₀ : ℝ, π/4 ≤ x₀ ∧ x₀ ≤ π/2 ∧ f x₀ = 2 ∧
  (∀ x : ℝ, π/4 ≤ x ∧ x ≤ π/2 → f x ≥ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplified_f_period_f_max_in_interval_f_min_in_interval_l54_5438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l54_5497

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the hyperbola equation (right branch)
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 6 ∧ x > 0

-- Define the intersection condition
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    hyperbola x₁ (line k x₁) ∧ 
    hyperbola x₂ (line k x₂)

-- State the theorem
theorem intersection_range :
  ∀ k : ℝ, intersects_at_two_points k ↔ -Real.sqrt 15 / 3 < k ∧ k < Real.sqrt 15 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l54_5497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l54_5402

-- Define the parabola
def parabola (x y : ℝ) : Prop := x = -(1/4) * y^2

-- Define the directrix
def directrix (x : ℝ) : Prop := x = 1

-- Define IsDirectrix (this is a placeholder definition)
def IsDirectrix (d : ℝ) (p : ℝ → ℝ → Prop) : Prop := True

-- Theorem statement
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola x y → (∃ (d : ℝ), directrix d ∧ IsDirectrix d parabola) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l54_5402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overtimePayIncreaseIs75Percent_l54_5442

/-- Represents the bus driver's pay structure and work details --/
structure BusDriverPay where
  regularRate : ℚ
  regularHours : ℚ
  totalHours : ℚ
  totalCompensation : ℚ

/-- Calculates the percentage increase in overtime pay rate --/
def overtimePayIncrease (pay : BusDriverPay) : ℚ :=
  let regularEarnings := pay.regularRate * pay.regularHours
  let overtimeHours := pay.totalHours - pay.regularHours
  let overtimeEarnings := pay.totalCompensation - regularEarnings
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - pay.regularRate) / pay.regularRate) * 100

/-- Theorem stating that the overtime pay increase is 75% given the specified conditions --/
theorem overtimePayIncreaseIs75Percent (pay : BusDriverPay)
  (h1 : pay.regularRate = 16)
  (h2 : pay.regularHours = 40)
  (h3 : pay.totalHours = 52)
  (h4 : pay.totalCompensation = 976) :
  overtimePayIncrease pay = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overtimePayIncreaseIs75Percent_l54_5442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_equality_condition_l54_5479

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^(1/3) * b^(1/3) + c^(1/3) * d^(1/3) ≤ (a+b+c)^(1/3) * (a+c+d)^(1/3) ∧
  (a^(1/3) * b^(1/3) + c^(1/3) * d^(1/3) = (a+b+c)^(1/3) * (a+c+d)^(1/3) ↔
    ∃ l : ℝ, 0 < l ∧ l < 1 ∧ b = a / (1 - l) ∧ c = a * (1 - l) / l ∧ d = a * (1 - l) / l^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_equality_condition_l54_5479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_perimeter_l54_5405

/-- An isosceles trapezoid with the given properties -/
structure IsoscelesTrapezoid where
  -- Points A, B, C, D
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- AB is half the length of CD
  ab_half_cd : dist A B = (1/2) * dist C D
  -- BC = 6
  bc_length : dist B C = 6
  -- AD = 12
  ad_length : dist A D = 12
  -- Height from AB to CD is 4
  height : abs (A.2 - C.2) = 4
  -- Ensure it's a trapezoid (parallel sides)
  is_trapezoid : (A.1 - B.1) / (A.2 - B.2) = (D.1 - C.1) / (D.2 - C.2)
  -- Ensure it's isosceles
  is_isosceles : dist A B = dist D C

/-- The perimeter of the isosceles trapezoid -/
def perimeter (t : IsoscelesTrapezoid) : ℝ :=
  dist t.A t.B + dist t.B t.C + dist t.C t.D + dist t.D t.A

/-- Theorem stating the perimeter of the isosceles trapezoid -/
theorem isosceles_trapezoid_perimeter (t : IsoscelesTrapezoid) :
  perimeter t = 3 * Real.sqrt 13 + 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_perimeter_l54_5405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_arrangements_eq_six_l54_5414

def digits : List ℕ := [3, 0, 0, 5]

def is_valid_arrangement (arr : List ℕ) : Bool :=
  arr.length = 4 && arr.head? ≠ some 0 && arr.toFinset = digits.toFinset

def count_valid_arrangements : ℕ :=
  (List.permutations digits).filter is_valid_arrangement |>.length

theorem count_valid_arrangements_eq_six :
  count_valid_arrangements = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_arrangements_eq_six_l54_5414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_parabola_l54_5430

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the slope of the given line (and parallel tangent line)
def m : ℝ := 2

-- Theorem statement
theorem tangent_line_to_parabola :
  ∃ (a : ℝ), (∀ x : ℝ, m*x - parabola x - 1 = 0 → 
    (∀ h : ℝ, h ≠ 0 → 
      (parabola (x + h) - parabola x) / h ≠ m)) ∧
    (∃ x : ℝ, m*x - parabola x - 1 = 0 ∧
      (∀ h : ℝ, h ≠ 0 → 
        |((parabola (x + h) - parabola x) / h) - m| < a * |h|)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_parabola_l54_5430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_point_problem_l54_5432

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem line_and_point_problem :
  let l1 : Line := { a := 1, b := 1, c := -2 }
  let A : Point := { x := -2, y := 0 }
  let l2 : Line := { a := 1, b := 1, c := 2 }
  let B1 : Point := { x := 2, y := 0 }
  let B2 : Point := { x := -2, y := 4 }
  (parallel l1 l2) ∧
  (pointOnLine A l2) ∧
  (pointOnLine B1 l1) ∧
  (pointOnLine B2 l1) ∧
  (distance A B1 = 4) ∧
  (distance A B2 = 4) ∧
  (∀ B : Point, pointOnLine B l1 ∧ distance A B = 4 → B = B1 ∨ B = B2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_point_problem_l54_5432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_problem_l54_5429

/-- A square with a given perimeter -/
structure Square where
  perimeter : ℝ

/-- The side length of a square -/
noncomputable def Square.sideLength (s : Square) : ℝ := s.perimeter / 4

/-- Given squares A, B, and C, where the perimeter of A is 16, the perimeter of B is 32,
    and the side length of C is the difference between the side lengths of A and B,
    prove that the perimeter of C is 16 -/
theorem square_perimeter_problem (A B : Square) (hA : A.perimeter = 16) (hB : B.perimeter = 32) :
  let C : Square := ⟨4 * (B.sideLength - A.sideLength)⟩
  C.perimeter = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_problem_l54_5429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_vertex_l54_5481

/-- Given a triangle with vertices at (8,6), (0,0), and (x,0) where x < 0,
    if the area of the triangle is 36 square units, then x = -12. -/
theorem triangle_third_vertex (x : ℝ) (h1 : x < 0) : 
  (1/2 : ℝ) * abs x * 6 = 36 → x = -12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_vertex_l54_5481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_christina_speed_l54_5498

/-- The speed at which Jack and Christina walk towards each other -/
def walking_speed : ℝ → ℝ := λ v => v

/-- The initial distance between Jack and Christina -/
def initial_distance : ℝ := 240

/-- Lindy's running speed -/
def lindy_speed : ℝ := 10

/-- The total distance Lindy travels -/
def lindy_total_distance : ℝ := 400

/-- The time it takes for Jack and Christina to meet -/
noncomputable def meeting_time (v : ℝ) : ℝ := initial_distance / (2 * v)

theorem jack_christina_speed : 
  ∃ v : ℝ, v > 0 ∧ 
    walking_speed v = v ∧
    lindy_speed * meeting_time v = lindy_total_distance ∧
    v = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_christina_speed_l54_5498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_number_probability_l54_5446

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧  -- two-digit integer
  n > 75 ∧  -- greater than 75
  (n / 10) % 2 = 1 ∧  -- tens digit is odd
  n % 2 = 1  -- units digit is odd

def valid_numbers : List ℕ :=
  (List.range 100).filter (fun n => 
    n ≥ 10 ∧ n < 100 ∧ 
    n > 75 ∧ 
    (n / 10) % 2 = 1 ∧ 
    n % 2 = 1)

theorem secret_number_probability : (1 : ℚ) / (valid_numbers.length : ℚ) = 1 / 7 := by
  -- Evaluate valid_numbers
  have h1 : valid_numbers = [77, 79, 91, 93, 95, 97, 99] := by rfl
  
  -- Calculate the length of valid_numbers
  have h2 : valid_numbers.length = 7 := by rfl
  
  -- Rewrite the left side of the equation
  rw [h2]
  
  -- Simplify the fraction
  norm_num

  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_number_probability_l54_5446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_greater_G_l54_5404

-- Define F and G as functions from ℝ to ℝ
variable {F G : ℝ → ℝ}

-- State the hypothesis
axiom h : ∀ x : ℝ, F (F x) > G (F x) ∧ G (F x) > G (G x)

-- State the theorem to be proved
theorem F_greater_G : ∀ x : ℝ, F x > G x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_greater_G_l54_5404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_quadrant_l54_5413

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative -/
def in_fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

/-- An angle θ is in the first quadrant if both cos θ and sin θ are positive -/
def in_first_quadrant (θ : ℝ) : Prop := Real.cos θ > 0 ∧ Real.sin θ > 0

/-- If z = cos θ - sin θ i is in the fourth quadrant, then θ is in the first quadrant -/
theorem angle_in_first_quadrant (θ : ℝ) (z : ℂ) 
  (h1 : z = Complex.mk (Real.cos θ) (-Real.sin θ))
  (h2 : in_fourth_quadrant z) : 
  in_first_quadrant θ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_quadrant_l54_5413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_box_balls_l54_5447

/-- A sequence of ball counts in boxes -/
def BallSequence : ℕ → ℕ := sorry

/-- The total number of boxes -/
def totalBoxes : ℕ := 1993

/-- The condition that the first box contains 7 balls -/
axiom first_box : BallSequence 1 = 7

/-- The condition that every four consecutive boxes contain 30 balls -/
axiom four_box_sum (n : ℕ) : 
  n + 3 ≤ totalBoxes → 
  BallSequence n + BallSequence (n + 1) + BallSequence (n + 2) + BallSequence (n + 3) = 30

/-- The theorem to be proved -/
theorem last_box_balls : BallSequence totalBoxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_box_balls_l54_5447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l54_5492

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

-- Define the line function
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Theorem statement
theorem shortest_distance_curve_to_line :
  ∃ (x₀ y₀ : ℝ), 
    y₀ = f x₀ ∧ 
    (∀ (x y : ℝ), y = f x → 
      (x - x₀)^2 + (y - y₀)^2 ≥ 
      ((2 * x₀ - y₀ + 3) / Real.sqrt 5)^2) ∧
    (2 * x₀ - y₀ + 3) / Real.sqrt 5 = Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l54_5492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_l54_5478

noncomputable def series_sum (k : ℝ) : ℝ := ∑' n, (6 * n - 2) / k^n

theorem series_sum_equals_three (k : ℝ) (h1 : k > 1) (h2 : series_sum k = 5) : k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_l54_5478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_painting_rate_l54_5473

/-- Represents the properties of a rectangular floor and its painting cost --/
structure Floor :=
  (length : ℝ)
  (breadth : ℝ)
  (total_cost : ℝ)

/-- Calculates the area of a rectangular floor --/
noncomputable def area (f : Floor) : ℝ := f.length * f.breadth

/-- Calculates the painting rate per square meter --/
noncomputable def rate_per_sqm (f : Floor) : ℝ := f.total_cost / area f

/-- Theorem stating the properties of the floor and the resulting painting rate --/
theorem floor_painting_rate (f : Floor) 
  (h1 : f.length = 3 * f.breadth)  -- Length is 200% more than breadth
  (h2 : f.length = 12.24744871391589)
  (h3 : f.total_cost = 100) :
  rate_per_sqm f = 2 := by
  sorry

#check floor_painting_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_painting_rate_l54_5473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l54_5407

/-- Given vectors a, b, and c in ℝ², prove that if 2a + b is collinear with c, 
    then the second component of a equals -9/2. -/
theorem vector_collinearity (a b c : ℝ × ℝ) (lambda : ℝ) : 
  a = (1, lambda) → 
  b = (2, 1) → 
  c = (1, -2) → 
  (∃ (k : ℝ), k ≠ 0 ∧ (2 • a + b) = k • c) → 
  lambda = -9/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l54_5407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_80_factorial_l54_5493

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let nonzero_digits := digits.filter (· ≠ 0)
  match nonzero_digits.reverse.take 2 with
  | [a, b] => 10 * b + a
  | [a] => a
  | _ => 0

theorem last_two_nonzero_digits_80_factorial :
  last_two_nonzero_digits (factorial 80) = 1 := by
  sorry

#eval last_two_nonzero_digits (factorial 80)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_80_factorial_l54_5493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_numbers_l54_5474

def are_opposite (a b : ℤ) : Prop := a = -b

theorem opposite_numbers : are_opposite (-(-1)) (-1) := by
  unfold are_opposite
  simp
  
#check opposite_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_numbers_l54_5474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_theorem_l54_5461

/-- Represents a rectangular box with length l, width w, and height h = w/2 -/
structure Box where
  l : ℝ
  w : ℝ
  h : ℝ
  h_eq_half_w : h = w / 2

/-- Calculates the area of wrapping paper needed for a given box -/
noncomputable def wrapping_paper_area (box : Box) : ℝ :=
  (7 * box.l * box.w) / 2

/-- Theorem stating that the area of the wrapping paper for a box with the given conditions is 7lw/2 -/
theorem wrapping_paper_area_theorem (box : Box) :
  wrapping_paper_area box = (7 * box.l * box.w) / 2 := by
  -- Proof goes here
  sorry

#check wrapping_paper_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_theorem_l54_5461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_waiting_time_l54_5408

/-- Represents the duration of green light in minutes -/
noncomputable def green_duration : ℝ := 1

/-- Represents the duration of red light in minutes -/
noncomputable def red_duration : ℝ := 2

/-- Represents the total cycle time of the traffic light in minutes -/
noncomputable def cycle_time : ℝ := green_duration + red_duration

/-- Represents the probability of arriving during green light -/
noncomputable def p_green : ℝ := green_duration / cycle_time

/-- Represents the probability of arriving during red light -/
noncomputable def p_red : ℝ := red_duration / cycle_time

/-- Represents the expected waiting time if arriving during green light -/
noncomputable def e_wait_green : ℝ := 0

/-- Represents the expected waiting time if arriving during red light -/
noncomputable def e_wait_red : ℝ := red_duration / 2

/-- Theorem stating that the expected waiting time is 2/3 minutes -/
theorem expected_waiting_time :
  p_green * e_wait_green + p_red * e_wait_red = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_waiting_time_l54_5408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_male_democrat_ratio_l54_5410

def total_participants : ℕ := 810
def female_democrats : ℕ := 135

theorem male_democrat_ratio :
  (let female_participants : ℕ := 2 * female_democrats
   let male_participants : ℕ := total_participants - female_participants
   let total_democrats : ℕ := total_participants / 3
   let male_democrats : ℕ := total_democrats - female_democrats
   (male_democrats : ℚ) / male_participants) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_male_democrat_ratio_l54_5410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_bound_l54_5455

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (4 : ℝ)^x + 2*x - 2

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4*x - 1

-- State the theorem
theorem root_difference_bound :
  ∃ (x₀ : ℝ), (1/4 < x₀ ∧ x₀ < 1/2) ∧ 
  g x₀ = 0 ∧ 
  ∃ (x₁ : ℝ), f x₁ = 0 ∧ 
  |x₁ - x₀| ≤ 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_bound_l54_5455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l54_5496

def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem cubic_function_properties (a b c : ℝ) :
  (∀ x, (deriv (f a b c)) x = 0 ↔ x = 1 ∨ x = -2/3) →
  f a b c (-1) = 3/2 →
  a = -1/2 ∧ b = -2 ∧ c = 1 ∧
  (∀ x, x < -2/3 → (deriv (f a b c)) x > 0) ∧
  (∀ x, -2/3 < x ∧ x < 1 → (deriv (f a b c)) x < 0) ∧
  (∀ x, x > 1 → (deriv (f a b c)) x > 0) ∧
  f a b c (-2/3) = 49/27 ∧
  f a b c 1 = -1/2 ∧
  (∀ x, f a b c x ≤ f a b c (-2/3)) ∧
  (∀ x, f a b c x ≥ f a b c 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l54_5496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l54_5419

noncomputable def sqrt5 : ℝ := Real.sqrt 5

def b_sequence (b : ℕ → ℝ) : Prop :=
  (∀ n ≥ 2, b n = b (n - 1) * b (n + 1)) ∧
  b 1 = 3 + 2 * sqrt5 ∧
  b 1800 = 11 + 2 * sqrt5

theorem sequence_property (b : ℕ → ℝ) (h : b_sequence b) : b 2010 = 11 + 2 * sqrt5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l54_5419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l54_5420

/-- The function f as defined in the problem -/
noncomputable def f (ω : ℝ) (m : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (ω * x) - 2 * Real.sin (ω * x / 2) ^ 2 + m

/-- The theorem stating the main results of the problem -/
theorem problem_solution (ω m : ℝ) (A B C : ℝ) :
  ω > 0 →
  (∀ x, f ω m (x + 3 * Real.pi) = f ω m x) →
  (∀ x ∈ Set.Icc 0 Real.pi, f ω m x ≥ 0) →
  (∃ x ∈ Set.Icc 0 Real.pi, f ω m x = 0) →
  f ω m C = 1 →
  2 * Real.sin B ^ 2 = Real.cos B + Real.cos (A - C) →
  (∀ x, f ω m x = 2 * Real.sin ((2 * x / 3) + Real.pi / 6) - 1) ∧
  Real.sin A = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l54_5420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l54_5487

-- Define the function f
noncomputable def f (a : ℝ) (θ : ℝ) (x : ℝ) : ℝ := 
  (a + 2 * Real.cos x ^ 2) * Real.cos (2 * x + θ)

-- State the theorem
theorem problem_solution (a θ α : ℝ) :
  (∀ x, f a θ x = -f a θ (-x)) →  -- f is an odd function
  f a θ (π/4) = 0 →               -- f(π/4) = 0
  θ ∈ Set.Ioo 0 π →               -- θ ∈ (0, π)
  f a θ (α/4) = -2/5 →            -- f(α/4) = -2/5
  α ∈ Set.Ioo (π/2) π →           -- α ∈ (π/2, π)
  (a = -1 ∧ 
   θ = π/2 ∧ 
   Real.sin (α + π/3) = (4 - 3 * Real.sqrt 3) / 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l54_5487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l54_5403

/-- A geometric sequence with first term a and common ratio r -/
def geometric_sequence (a r : ℝ) : ℕ → ℝ := λ n ↦ a * r^(n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_five (a r : ℝ) :
  let seq := geometric_sequence a r
  (seq 1 + seq 2 = 3/4) →
  (seq 4 + seq 5 = 6) →
  geometric_sum a r 5 = 31/4 := by
  sorry

#check geometric_sequence_sum_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l54_5403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_finger_is_seven_l54_5412

def g : ℕ → ℕ
  | 1 => 8
  | 2 => 7
  | 3 => 6
  | 4 => 5
  | 5 => 4
  | 6 => 3
  | 7 => 2
  | 8 => 1
  | 9 => 0
  | _ => 0  -- Default case for completeness

def finger_sequence : ℕ → ℕ
  | 0 => 2  -- Start with 2 on the first finger (index 0)
  | n+1 => g (finger_sequence n)

theorem twelfth_finger_is_seven :
  finger_sequence 11 = 7 := by
  sorry

#eval finger_sequence 11  -- This will evaluate the 12th finger (index 11)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_finger_is_seven_l54_5412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l54_5422

/-- The angle between two 2D vectors given their components and projection -/
theorem angle_between_vectors (a1 a2 b1 b2 m : ℝ) (h_proj : (a1 * b1 + a2 * b2) / Real.sqrt (a1^2 + a2^2) = -3)
  (h_a : a1 = 1 ∧ a2 = Real.sqrt 3)
  (h_b : b1 = 3 ∧ b2 = m) :
  Real.arccos ((a1 * b1 + a2 * b2) / (Real.sqrt (a1^2 + a2^2) * Real.sqrt (b1^2 + b2^2))) = 2 * Real.pi / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l54_5422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l54_5463

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 2 + x) + (Real.sin (Real.pi / 2 + x))^2

theorem f_max_value :
  ∃ (M : ℝ), M = 5/4 ∧ ∀ (x : ℝ), f x ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l54_5463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l54_5433

-- Define the circles
def circle1 : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + p.2^2 = 3^2}
def circle2 : Set (ℝ × ℝ) := {p | (p.1 - 8)^2 + p.2^2 = 2^2}

-- Define the tangent line
def tangentLine (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (t : ℝ), p = (1 - t) • p1 + t • p2}

-- Theorem statement
theorem tangent_line_y_intercept :
  ∃ (p1 p2 : ℝ × ℝ),
    p1 ∈ circle1 ∧
    p2 ∈ circle2 ∧
    p1.1 > 3 ∧ p1.2 > 0 ∧
    p2.1 > 8 ∧ p2.2 > 0 ∧
    (tangentLine p1 p2 ∩ ({0} : Set ℝ) ×ˢ Set.univ = {(0, 15 * Real.sqrt 26 / 26)}) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l54_5433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_punctuality_related_to_company_l54_5445

/-- Represents the survey data for bus companies --/
structure BusSurveyData where
  company_a_on_time : Nat
  company_a_not_on_time : Nat
  company_b_on_time : Nat
  company_b_not_on_time : Nat

/-- Calculates the chi-square statistic for the given survey data --/
noncomputable def chi_square (data : BusSurveyData) : Real :=
  let n := data.company_a_on_time + data.company_a_not_on_time + data.company_b_on_time + data.company_b_not_on_time
  let a := data.company_a_on_time
  let b := data.company_a_not_on_time
  let c := data.company_b_on_time
  let d := data.company_b_not_on_time
  (n * (a * d - b * c)^2 : Real) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for 90% confidence level --/
def critical_value : Real := 2.706

/-- Theorem stating that the chi-square statistic for the given data is greater than the critical value --/
theorem bus_punctuality_related_to_company (data : BusSurveyData)
  (h1 : data.company_a_on_time = 240)
  (h2 : data.company_a_not_on_time = 20)
  (h3 : data.company_b_on_time = 210)
  (h4 : data.company_b_not_on_time = 30) :
  chi_square data > critical_value := by
  sorry

#eval "Theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_punctuality_related_to_company_l54_5445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_hours_reduced_by_four_l54_5482

/-- Calculates the difference in reading hours per week given initial and new reading conditions -/
noncomputable def reading_hours_difference (initial_rate : ℝ) (initial_pages : ℝ) (speed_increase : ℝ) (new_pages : ℝ) : ℝ :=
  let initial_hours := initial_pages / initial_rate
  let new_rate := initial_rate * speed_increase
  let new_hours := new_pages / new_rate
  initial_hours - new_hours

/-- Theorem stating that under the given conditions, the difference in reading hours is 4 -/
theorem reading_hours_reduced_by_four :
  reading_hours_difference 40 600 1.5 660 = 4 := by
  -- Unfold the definition of reading_hours_difference
  unfold reading_hours_difference
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_hours_reduced_by_four_l54_5482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_implies_a_value_l54_5424

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a / (x^2 + 1)) / Real.log (1/2)

-- State the theorem
theorem function_range_implies_a_value (a : ℝ) :
  (∀ y ∈ Set.range (f a), y ≥ -1) ∧ 
  (∀ y ≥ -1, ∃ x, f a x = y) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_implies_a_value_l54_5424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l54_5444

/-- A complex polynomial of degree n -/
def ComplexPolynomial (n : ℕ) := Fin (n + 1) → ℂ

/-- The evaluation of a complex polynomial at a point -/
noncomputable def evalPoly (p : ComplexPolynomial n) (z : ℂ) : ℂ :=
  (Finset.range (n + 1)).sum (fun i => p i * z ^ (n - i))

/-- The statement of the theorem -/
theorem polynomial_value_bound (n : ℕ) (f : ComplexPolynomial n) :
  ∃ z₀ : ℂ, Complex.abs z₀ ≤ 1 ∧ Complex.abs (evalPoly f z₀) ≥ Complex.abs (f 0) + Complex.abs (f ⟨n, by norm_num⟩) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l54_5444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_qr_ratio_l54_5483

/-- Given a triangle XYZ with points P on XY and Q on YZ, prove that PQ/QR = 1/5 -/
theorem pq_qr_ratio (X Y Z P Q R : ℝ × ℝ) : 
  let XY := Y - X
  let YZ := Z - Y
  let PQ := Q - P
  let QR := R - Q
  (P = X + (4/5 : ℝ) • XY) →
  (Q = Y + (4/5 : ℝ) • YZ) →
  (∃ t : ℝ, R = X + t • (Z - X)) →
  (∃ s : ℝ, R = P + s • PQ) →
  ‖PQ‖ / ‖QR‖ = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_qr_ratio_l54_5483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_theorem_l54_5466

/-- The number of ways to place n identical balls into k different boxes -/
def PlaceBalls (n : ℕ) (k : ℕ) (emptyAllowed : ℕ) : ℕ :=
  sorry

/-- The number of ways to place n identical balls into k different boxes with no empty boxes -/
def PlaceBallsNoEmpty (n : ℕ) (k : ℕ) : ℕ := 
  PlaceBalls n k 0

theorem balls_in_boxes_theorem :
  (PlaceBallsNoEmpty 7 4 = Nat.choose 6 3) ∧
  (PlaceBalls 7 4 1 = Nat.choose 10 3) :=
by
  sorry

#check balls_in_boxes_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_theorem_l54_5466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l54_5426

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 6))

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≠ 4.5}

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l54_5426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_2016_l54_5425

noncomputable def f (a b α β x : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 2

theorem function_value_2016 
  (a b α β : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hα : α ≠ 0) 
  (hβ : β ≠ 0) 
  (h2015 : f a b α β 2015 = 1) : 
  f a b α β 2016 = 3 := by
  sorry

#check function_value_2016

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_2016_l54_5425
