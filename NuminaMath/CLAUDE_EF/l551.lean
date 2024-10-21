import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l551_55123

/-- The function f(x) = x^2 + x -/
noncomputable def f (x : ℝ) : ℝ := x^2 + x

/-- The minimum point of f(x) -/
noncomputable def min_point : ℝ := -1/2

/-- The minimum value of f(x) -/
noncomputable def min_value : ℝ := -1/4

theorem f_minimum :
  (∀ x : ℝ, f x ≥ f min_point) ∧ f min_point = min_value := by
  sorry

#check f_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l551_55123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_ratio_l551_55160

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_ratio (θ : ℝ) :
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (1, -2)
  parallel a b →
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_ratio_l551_55160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_intersection_varies_l551_55164

-- Define the circle
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the points and chord
structure CircleConfiguration where
  circle : Circle
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)
  D : EuclideanSpace ℝ (Fin 2)
  O : EuclideanSpace ℝ (Fin 2)

-- Define the properties of the configuration
def validConfiguration (config : CircleConfiguration) : Prop :=
  let circle := config.circle
  (dist config.A circle.center = circle.radius) ∧
  (dist config.B circle.center = circle.radius) ∧
  (dist config.C circle.center = circle.radius) ∧
  (dist config.D circle.center = circle.radius) ∧
  (config.O = circle.center) ∧
  (dist config.A config.B = 2 * circle.radius) ∧
  (dist config.A config.O = dist config.B config.O)

-- Define the bisector of angle OCD
noncomputable def bisectorOCD (config : CircleConfiguration) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry -- Definition of the bisector

-- Define the intersection point of the bisector with the circle
noncomputable def intersectionPoint (config : CircleConfiguration) : EuclideanSpace ℝ (Fin 2) :=
  sorry -- Definition of the intersection point

-- Theorem statement
theorem bisector_intersection_varies (circle : Circle) :
  ∃ (config1 config2 : CircleConfiguration),
    validConfiguration config1 ∧
    validConfiguration config2 ∧
    config1.circle = circle ∧
    config2.circle = circle ∧
    config1.A = config2.A ∧
    config1.B = config2.B ∧
    config1.O = config2.O ∧
    intersectionPoint config1 ≠ intersectionPoint config2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_intersection_varies_l551_55164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arcsin_cos_l551_55133

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.cos x)

-- Define the interval
def interval : Set ℝ := Set.Icc 0 (4 * Real.pi)

-- State the theorem
theorem area_arcsin_cos : 
  ∫ x in interval, |f x| = Real.pi ^ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arcsin_cos_l551_55133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_shared_focus_l551_55184

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

-- Define the focus of the parabola
noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the right focus of the ellipse
noncomputable def ellipse_right_focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem parabola_ellipse_shared_focus :
  ∀ p : ℝ, (∃ x y : ℝ, parabola p x y) ∧ 
           (∃ x y : ℝ, ellipse x y) ∧ 
           parabola_focus p = ellipse_right_focus →
           p = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_shared_focus_l551_55184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_factorial_divisible_by_840_l551_55119

theorem least_n_factorial_divisible_by_840 :
  ∀ n : ℕ, n > 0 → (∀ k : ℕ, 0 < k → k < 8 → ¬(840 ∣ Nat.factorial k)) ∧ (840 ∣ Nat.factorial 8) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_factorial_divisible_by_840_l551_55119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gender_world_cup_relationship_l551_55187

/-- Represents the contingency table data --/
structure ContingencyTable where
  male_interested : ℕ
  female_interested : ℕ
  male_not_interested : ℕ
  female_not_interested : ℕ

/-- Calculates the K² statistic --/
noncomputable def calculate_k_squared (table : ContingencyTable) : ℝ :=
  let n := table.male_interested + table.female_interested + table.male_not_interested + table.female_not_interested
  let a := table.male_interested
  let b := table.female_interested
  let c := table.male_not_interested
  let d := table.female_not_interested
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for 1% significance level --/
def critical_value : ℝ := 6.635

/-- Theorem stating the relationship between gender and World Cup interest --/
theorem gender_world_cup_relationship (data : ContingencyTable)
  (h1 : data.male_interested = 50)
  (h2 : data.female_not_interested = 20)
  (h3 : data.male_interested + data.female_interested + data.male_not_interested + data.female_not_interested = 110)
  (h4 : data.female_interested + data.female_not_interested = 30) :
  calculate_k_squared data > critical_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gender_world_cup_relationship_l551_55187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l551_55169

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sqrt 3 * Real.cos (2 * x + φ) + Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) (h1 : |φ| < π/2) 
  (h2 : ∀ x, f x φ = f (-x) φ) : 
  (∃ T > 0, (∀ x, f (x + T) φ = f x φ) ∧ 
   (∀ S, 0 < S → S < T → ∃ y, f (y + S) φ ≠ f y φ)) ∧
  (∀ x y, x ∈ Set.Ioo 0 (π/2) → y ∈ Set.Ioo 0 (π/2) → x < y → f y φ < f x φ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l551_55169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_selected_is_20_l551_55122

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Represents the population size --/
def PopulationSize : Nat := 50

/-- Represents the starting row in the random number table --/
def StartRow : Nat := 6

/-- Represents the starting column in the random number table --/
def StartColumn : Nat := 9

/-- Represents the number of individuals to be selected --/
def SelectionSize : Nat := 6

/-- The given random number table --/
def givenTable : RandomNumberTable := [
  [2635, 7900, 3370, 9160, 1620, 3882, 7757, 4950],
  [3211, 4919, 7306, 4916, 7677, 8733, 9974, 6732],
  [2748, 6198, 7164, 4148, 7086, 2888, 8519, 1620],
  [7477, 0111, 1630, 2404, 2979, 7991, 9683, 5125]
]

/-- Function to select individuals based on the random number table --/
def selectIndividuals (table : RandomNumberTable) (startRow : Nat) (startColumn : Nat) (popSize : Nat) (selectionSize : Nat) : List Nat :=
  sorry -- Implementation details omitted

/-- Theorem stating that the 3rd selected individual is 20 --/
theorem third_selected_is_20 :
  let selected := selectIndividuals givenTable StartRow StartColumn PopulationSize SelectionSize
  (selected.length ≥ 3) → (selected.get ⟨2, sorry⟩ = 20) := by
  sorry

#check third_selected_is_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_selected_is_20_l551_55122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_sums_l551_55111

theorem sandy_sums : ∃ (marks_per_correct marks_per_incorrect total_marks correct_sums total_sums : ℕ),
  marks_per_correct = 3 ∧
  marks_per_incorrect = 2 ∧
  total_marks = 60 ∧
  correct_sums = 24 ∧
  total_sums = correct_sums + (total_marks - marks_per_correct * correct_sums) / marks_per_incorrect ∧
  total_sums = 30 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_sums_l551_55111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l551_55108

-- Define the triangle ABC
def A : ℝ × ℝ := (-4, 0)
def C : ℝ × ℝ := (4, 0)

-- Define the ellipse equation
def on_ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Define vertex B to be on the ellipse
noncomputable def B : ℝ × ℝ := sorry
axiom B_on_ellipse : on_ellipse B.1 B.2

-- Define the angles of the triangle
noncomputable def angle_A : ℝ := Real.arccos ((B.1 - (-4))^2 + B.2^2 - 8^2) / (2 * Real.sqrt ((B.1 - (-4))^2 + B.2^2) * 8)
noncomputable def angle_B : ℝ := Real.arccos (((-4 - B.1)^2 + B.2^2 + (4 - B.1)^2 + B.2^2 - 8^2) / (2 * Real.sqrt (((-4 - B.1)^2 + B.2^2) * ((4 - B.1)^2 + B.2^2))))
noncomputable def angle_C : ℝ := Real.arccos ((B.1 - 4)^2 + B.2^2 - 8^2) / (2 * Real.sqrt ((B.1 - 4)^2 + B.2^2) * 8)

-- State the theorem
theorem triangle_ratio : 
  (Real.sin angle_A + Real.sin angle_C) / Real.sin angle_B = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l551_55108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_is_twelve_l551_55191

-- Define Ana and Bonita's ages this year
def ana_age (year : ℕ) : ℕ := sorry
def bonita_age (year : ℕ) : ℕ := sorry

-- n is the age difference between Ana and Bonita
def n : ℕ := ana_age 0 - bonita_age 0

-- Ana and Bonita were born n years apart
axiom age_difference : ∀ year, ana_age year = bonita_age year + n

-- Last year Ana was 5 times as old as Bonita
axiom last_year_relation : ana_age 0 - 1 = 5 * (bonita_age 0 - 1)

-- This year Ana's age is the square of Bonita's age
axiom this_year_relation : ana_age 0 = (bonita_age 0) ^ 2

-- The theorem to prove
theorem age_difference_is_twelve : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_is_twelve_l551_55191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemys_theorem_l551_55127

/-- A quadrilateral is cyclic if all four vertices lie on a circle -/
def is_cyclic_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ), 
    ‖A - center‖ = radius ∧
    ‖B - center‖ = radius ∧
    ‖C - center‖ = radius ∧
    ‖D - center‖ = radius

/-- Ptolemy's theorem for cyclic quadrilaterals -/
theorem ptolemys_theorem (A B C D : EuclideanSpace ℝ (Fin 2)) 
  (h : is_cyclic_quadrilateral A B C D) :
  ‖A - B‖ * ‖C - D‖ + ‖A - D‖ * ‖B - C‖ = ‖A - C‖ * ‖B - D‖ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemys_theorem_l551_55127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_theorem_l551_55157

/-- Given a parabola y = ax^2 with directrix y = -2, prove that a = 1/8 -/
theorem parabola_directrix_theorem (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x^2) ∧ 
  (∃ (y : ℝ), y = -2 ∧ y = -1 / (4 * a)) → 
  a = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_theorem_l551_55157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l551_55158

theorem problem_statement (b : ℝ) : 
  ((2023 - b) - b^2 / (2023 - b)) / ((2023 - b - b) / (2023 - b)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l551_55158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_theorem_l551_55113

/-- Represents a segment on the side of a square --/
structure Segment where
  length : ℚ
  side : Fin 4

/-- Represents a division of a square's sides --/
structure SquareDivision where
  segments : List Segment
  segment_count : ℕ
  segment_count_eq : segment_count = 400

/-- Checks if a list of segments can form a rectangle --/
def can_form_rectangle (segments : List Segment) : Prop :=
  ∃ (a b : ℚ), a ≠ b ∧ 
    ∃ (side1 side2 : List Segment),
      segments = side1 ++ side2 ∧
      (side1.map (λ s => s.length)).sum = a ∧
      (side2.map (λ s => s.length)).sum = b

/-- The main theorem stating that there exists a way to divide a square
    such that no rectangle other than the original square can be formed --/
theorem square_division_theorem :
  ∃ (sd : SquareDivision),
    sd.segment_count = 400 ∧
    (∀ (rect_segments : List Segment),
      rect_segments ⊆ sd.segments →
      can_form_rectangle rect_segments →
      rect_segments = sd.segments) := by
  sorry

#check square_division_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_theorem_l551_55113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_2011_equals_two_l551_55178

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | (n + 2) => 1 / sequence_a (n + 1)

def product_pi (n : ℕ) : ℚ :=
  (Finset.range n).prod (λ i => sequence_a (i + 1))

theorem product_2011_equals_two : product_pi 2011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_2011_equals_two_l551_55178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_theorem_l551_55144

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 5 then x - 3 else Real.rpow x (1/3)

-- State that f has an inverse
axiom f_has_inverse : Function.Bijective f

-- Define the inverse function f⁻¹
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- State the theorem
theorem inverse_sum_theorem :
  (Finset.range 11).sum (λ i => f_inv (i - 6 : ℝ)) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_theorem_l551_55144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l551_55185

theorem equation_solution : ∃! x : ℚ, (5 * x - 20) / 3 = (7 - 3 * x) / 4 := by
  use 101 / 29
  constructor
  · -- Prove that 101/29 satisfies the equation
    sorry
  · -- Prove uniqueness
    intro y h
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l551_55185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_approaches_two_l551_55143

theorem limit_fraction_approaches_two :
  ∀ ε > 0, ∃ N : ℝ, ∀ n ≥ N, |((2 * n + 3) / (n + 1)) - 2| < ε :=
by
  intro ε hε
  use 1/ε
  intro n hn
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_approaches_two_l551_55143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upgraded_sensors_fraction_l551_55112

/-- Represents the number of modular units for each satellite -/
def units_A : ℚ := 24
def units_B : ℚ := 36
def units_C : ℚ := 48

/-- Represents the total number of upgraded sensors on Satellite A -/
def U_A : ℚ := 1  -- We assign a value here, but it could be any non-zero rational

/-- Represents the number of non-upgraded sensors on one unit of Satellite A -/
def N_A_unit : ℚ := U_A / 4

/-- Represents the total number of upgraded sensors on Satellite B -/
def U_B : ℚ := 3 * U_A

/-- Represents the number of non-upgraded sensors on one unit of Satellite B -/
def N_B_unit : ℚ := 2 * N_A_unit

/-- Represents the total number of upgraded sensors on Satellite C -/
def U_C : ℚ := 4 * U_A

/-- Represents the number of non-upgraded sensors on one unit of Satellite C -/
def N_C_unit : ℚ := 3 * N_A_unit

/-- The fraction of upgraded sensors for all satellites combined is 1/8.5 -/
theorem upgraded_sensors_fraction :
  let total_upgraded := U_A + U_B + U_C
  let total_A := U_A + units_A * N_A_unit
  let total_B := U_B + units_B * N_B_unit
  let total_C := U_C + units_C * N_C_unit
  let total_sensors := total_A + total_B + total_C
  total_upgraded / total_sensors = 1 / 8.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upgraded_sensors_fraction_l551_55112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l551_55179

/-- Parabola with focus F, fixed point M, and origin O -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- The focus of the parabola -/
noncomputable def Parabola.focus (c : Parabola) : ℝ × ℝ := (c.p / 2, 0)

/-- The fixed point M on the parabola -/
def Parabola.M (c : Parabola) (a : ℝ) : ℝ × ℝ := (a, 4)

/-- The origin O -/
def O : ℝ × ℝ := (0, 0)

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem parabola_properties (c : Parabola) (a : ℝ) :
  triangleArea (Parabola.M c a) O (Parabola.focus c) = 4 →
  c.p = 4 ∧ ∀ (x y : ℝ), y^2 = 8*x ↔ y^2 = 2*c.p*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l551_55179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_increasing_l551_55125

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define what it means for a function to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

-- State the theorem
theorem log2_increasing :
  (∀ a > 1, IncreasingOn (log a) (Set.Ioo 0 Real.pi)) →
  IncreasingOn (log 2) (Set.Ioo 0 Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_increasing_l551_55125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_coordinates_l551_55188

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (x - 5)^2 / 4^2 - (y - 10)^2 / 9^2 = 1

/-- The coordinates of the focus with larger x-coordinate -/
noncomputable def focus_coordinates : ℝ × ℝ := (5 + Real.sqrt 97, 10)

/-- Theorem: The coordinates of the focus with larger x-coordinate for the given hyperbola -/
theorem hyperbola_focus_coordinates :
  let (fx, fy) := focus_coordinates
  hyperbola_equation fx fy ∧
  ∀ x y, hyperbola_equation x y → x ≤ fx :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_coordinates_l551_55188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_rotation_similarity_l551_55139

-- Define the structure for a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the structure for a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the structure for a rhombus
structure Rhombus where
  p : Point
  q : Point
  r : Point
  s : Point

-- Define the perpendicular relation between lines
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the parallel relation between lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the inscribed relation between a rhombus and a set of lines
def inscribed (r : Rhombus) (l1 l2 l3 l4 : Line) : Prop := sorry

-- Define the rotation of a rhombus around a point by an angle
noncomputable def rotate (r : Rhombus) (center : Point) (angle : ℝ) : Rhombus := sorry

-- Define the similarity relation between rhombuses
def similar (r1 r2 : Rhombus) : Prop := sorry

-- Define the diagonal intersection of a rhombus
noncomputable def diagonal_intersection (r : Rhombus) : Point := sorry

-- The main theorem
theorem rhombus_rotation_similarity 
  (l1 l2 l3 l4 : Line) 
  (r : Rhombus) 
  (h1 : perpendicular l1 l3) 
  (h2 : perpendicular l2 l4) 
  (h3 : parallel l1 l2) 
  (h4 : parallel l3 l4) 
  (h5 : inscribed r l1 l2 l3 l4) 
  (m : Point) 
  (h6 : m = diagonal_intersection r) 
  (α : ℝ) : 
  similar r (rotate r m α) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_rotation_similarity_l551_55139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_difference_l551_55199

/-- Given the investments of three partners and the profit share of one partner,
    calculate the difference between the profit shares of the other two partners. -/
theorem profit_share_difference (a b c : ℕ) (b_profit : ℕ) :
  a = 8000 → b = 10000 → c = 12000 → b_profit = 1800 →
  (c / 2000 * (b_profit / (b / 2000))) - (a / 2000 * (b_profit / (b / 2000))) = 720 :=
by
  intro ha hb hc hb_profit
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_difference_l551_55199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_center_max_height_l551_55154

/-- The maximum height reached by the center of a spring connecting two identical masses -/
theorem spring_center_max_height
  (m : ℝ) -- Mass of each object
  (V₁ V₂ : ℝ) -- Initial velocities of upper and lower masses
  (α β : ℝ) -- Angles of initial velocities with respect to horizontal
  (g : ℝ) -- Acceleration due to gravity
  (h_m_pos : m > 0)
  (h_g_pos : g > 0) :
  (((V₁ * Real.sin β + V₂ * Real.sin α) / 2)^2) / (2 * g) = (V₁ * Real.sin β + V₂ * Real.sin α)^2 / (8 * g) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_center_max_height_l551_55154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_2_sqrt_70_l551_55155

-- Define the point P
def P : Fin 3 → ℝ := ![(-7), (-13), 10]

-- Define the direction vector of the line l
def l_direction : Fin 3 → ℝ := ![(-2), 1, 0]

-- Define a point on the line l
def l_point : Fin 3 → ℝ := ![1, (-2), 0]

-- Define the distance function
noncomputable def distance_point_to_line (P l_point l_direction : Fin 3 → ℝ) : ℝ :=
  let v := fun i => P i - l_point i
  let cross_product := ![
    v 1 * l_direction 2 - v 2 * l_direction 1,
    v 2 * l_direction 0 - v 0 * l_direction 2,
    v 0 * l_direction 1 - v 1 * l_direction 0
  ]
  Real.sqrt (cross_product 0^2 + cross_product 1^2 + cross_product 2^2) /
    Real.sqrt (l_direction 0^2 + l_direction 1^2 + l_direction 2^2)

-- Theorem statement
theorem distance_point_to_line_is_2_sqrt_70 :
  distance_point_to_line P l_point l_direction = 2 * Real.sqrt 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_2_sqrt_70_l551_55155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_correct_l551_55181

/-- Represents a runner with uphill and downhill speeds -/
structure Runner where
  uphillSpeed : ℝ
  downhillSpeed : ℝ

/-- The problem setup -/
structure HillProblem where
  totalDistance : ℝ 
  upHillDistance : ℝ 
  jack : Runner
  jill : Runner
  headStart : ℝ 

/-- The specific problem instance -/
def problemInstance : HillProblem := {
  totalDistance := 12
  upHillDistance := 6
  jack := ⟨12, 18⟩
  jill := ⟨14, 20⟩
  headStart := 0.25 -- 15 minutes in hours
}

/-- The meeting point of Jack and Jill -/
noncomputable def meetingPoint (p : HillProblem) : ℝ := 15.75

/-- Theorem stating that the meeting point is correct -/
theorem meeting_point_correct (p : HillProblem) :
  ∃ t : ℝ, 
    t > p.upHillDistance / p.jack.uphillSpeed ∧
    p.jack.downhillSpeed * (t - p.upHillDistance / p.jack.uphillSpeed) = 
    p.jill.uphillSpeed * (t - p.headStart) ∧
    p.jack.downhillSpeed * (t - p.upHillDistance / p.jack.uphillSpeed) + p.upHillDistance - meetingPoint p = 0 :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_correct_l551_55181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_with_5_cups_l551_55117

-- Define the number of cookies Alice can make with 4 cups of flour
def cookies_with_4_cups : ℕ := 24

-- Define the number of cups of flour for the original recipe
def original_flour_cups : ℕ := 4

-- Define the number of cups of flour for the new recipe
def new_flour_cups : ℕ := 5

-- Theorem to prove
theorem cookies_with_5_cups : 
  (cookies_with_4_cups * new_flour_cups) / original_flour_cups = 30 := by
  -- We need to prove that the calculation results in 30
  sorry

#eval (cookies_with_4_cups * new_flour_cups) / original_flour_cups

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_with_5_cups_l551_55117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l551_55197

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | Real.log (x + 2) ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l551_55197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_100_simplify_expression_equals_one_l551_55128

-- Problem 1
theorem complex_expression_equals_100 :
  (2 + 7/9 : ℝ)^(1/2 : ℝ) + (1/10 : ℝ)^(-2 : ℝ) + (2 + 10/27 : ℝ)^(-2/3 : ℝ) - 3*(π : ℝ)^(0 : ℝ) + 37/48 = 100 := by sorry

-- Problem 2
theorem simplify_expression_equals_one (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^(8/5 : ℝ) * b^(-6/5 : ℝ))^(-1/2 : ℝ) * 5*a^4 / (5*b^3) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_100_simplify_expression_equals_one_l551_55128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l551_55182

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : Fin 2 → ℚ) : Prop :=
  a 0 * b 1 = a 1 * b 0

/-- Vector pair A -/
def vector_pair_A : (Fin 2 → ℚ) × (Fin 2 → ℚ) :=
  (λ i => if i = 0 then -1 else 2, λ i => if i = 0 then 3 else 5)

/-- Vector pair B -/
def vector_pair_B : (Fin 2 → ℚ) × (Fin 2 → ℚ) :=
  (λ i => if i = 0 then 1 else 2, λ i => if i = 0 then 2 else 1)

/-- Vector pair C -/
def vector_pair_C : (Fin 2 → ℚ) × (Fin 2 → ℚ) :=
  (λ i => if i = 0 then 2 else -1, λ i => if i = 0 then 3 else 4)

/-- Vector pair D -/
def vector_pair_D : (Fin 2 → ℚ) × (Fin 2 → ℚ) :=
  (λ i => if i = 0 then -2 else 1, λ i => if i = 0 then 4 else -2)

theorem parallel_vectors :
  ¬(are_parallel vector_pair_A.1 vector_pair_A.2) ∧
  ¬(are_parallel vector_pair_B.1 vector_pair_B.2) ∧
  ¬(are_parallel vector_pair_C.1 vector_pair_C.2) ∧
  (are_parallel vector_pair_D.1 vector_pair_D.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l551_55182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_point_l551_55189

/-- The trajectory of a point M(x,y) satisfying |MF₁| - |MF₂| = 4 where F₁(-2,0) and F₂(2,0) are fixed points -/
theorem trajectory_of_moving_point (x y : ℝ) : 
  (Real.sqrt ((x + 2)^2 + y^2) - Real.sqrt ((x - 2)^2 + y^2) = 4) →
  (y = 0 ∧ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_point_l551_55189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_B_win_probability_l551_55124

/-- A game where two players roll a fair die with the following rules:
    - Player A wins if 3 comes up
    - Player B wins if 4, 5, or 6 comes up
    - If 1 or 2 comes up, the die is rolled again until 3, 4, 5, or 6 is rolled -/
def DiceGame : Type := Unit

/-- The probability of rolling a specific number on a fair die -/
noncomputable def fairDieProbability : ℝ := 1 / 6

/-- The probability of player A winning on a single roll -/
noncomputable def probA (game : DiceGame) : ℝ := fairDieProbability

/-- The probability of player B winning on a single roll -/
noncomputable def probB (game : DiceGame) : ℝ := 3 * fairDieProbability

/-- The probability of neither player winning on a single roll (i.e., rolling 1 or 2) -/
noncomputable def probNeither (game : DiceGame) : ℝ := 2 * fairDieProbability

/-- The theorem stating that the probability of player B winning the game is 3/4 -/
theorem player_B_win_probability (game : DiceGame) : 
  probB game / (1 - probNeither game) = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_B_win_probability_l551_55124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleTransformation_l551_55168

-- Define the color type
inductive Color
| Red
| Blue

-- Define the position type (even or odd)
inductive Position
| Even
| Odd

-- Define a function to convert a piece to its numerical value
def pieceValue (c : Color) (p : Position) : Int :=
  match c, p with
  | Color.Red, _ => 0
  | Color.Blue, Position.Even => 1
  | Color.Blue, Position.Odd => -1

-- Define the type of operations
inductive Operation
| Insert (c : Color) (p : Position)
| Remove (c : Color) (p : Position)

-- Define a function to apply an operation to a sequence
def applyOperation (seq : List Int) (op : Operation) : List Int :=
  match op with
  | Operation.Insert c p => 
    let v := pieceValue c p
    seq ++ [v, v]
  | Operation.Remove c p =>
    let v := pieceValue c p
    seq.filter (λ x => x ≠ v)

-- Define a function to apply a list of operations
def applyOperations (seq : List Int) (ops : List Operation) : List Int :=
  ops.foldl applyOperation seq

-- State the theorem
theorem impossibleTransformation :
  ∀ (ops : List Operation),
    let initial := [0, 1]
    let final := applyOperations initial ops
    final ≠ [-1, 0] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleTransformation_l551_55168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_range_l551_55134

theorem ellipse_ratio_range (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a^2 = b^2 + c^2) :
  1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_range_l551_55134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_option_is_D_l551_55194

/-- Represents a programming statement --/
inductive Statement
| Input (var : String)
| Print (expr : String)

/-- Checks if a statement is valid based on given rules --/
def is_valid_statement : Statement → Prop
| Statement.Input var => ∃ val : String, var ≠ ""
| Statement.Print expr => expr ≠ ""

/-- Represents the options in the multiple choice question --/
inductive MCOption
| A | B | C | D

/-- Maps an option to its corresponding statement --/
def option_to_statement : MCOption → Statement
| MCOption.A => Statement.Input "A"
| MCOption.B => Statement.Input "B=3"
| MCOption.C => Statement.Print "y=2*x+1"
| MCOption.D => Statement.Print "4*x"

theorem correct_option_is_D :
  (∀ var : String, ∃ val : String, Statement.Input var = Statement.Input var) →  -- INPUT is an input statement
  (∀ expr : String, Statement.Print expr = Statement.Print expr) →  -- PRINT is an output statement
  (∀ expr var : String, Statement.Print expr ≠ Statement.Input var) →  -- PRINT does not have the function of assignment
  (∀ o : MCOption, is_valid_statement (option_to_statement o) ↔ o = MCOption.D) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_option_is_D_l551_55194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_plan_fee_is_75_l551_55100

/-- Represents a cellular phone plan -/
structure PhonePlan where
  baseFee : ℚ  -- Monthly base fee
  baseMinutes : ℚ  -- Minutes included in the base fee
  overageRate : ℚ  -- Rate per minute over the base minutes

/-- Calculates the total cost for a given number of minutes -/
def totalCost (plan : PhonePlan) (minutes : ℚ) : ℚ :=
  plan.baseFee + max 0 (minutes - plan.baseMinutes) * plan.overageRate

theorem second_plan_fee_is_75 :
  let plan1 : PhonePlan := { baseFee := 50, baseMinutes := 500, overageRate := 35/100 }
  let plan2 : PhonePlan := { baseFee := x, baseMinutes := 1000, overageRate := 45/100 }
  ∃ x : ℚ, x > 0 ∧ totalCost plan1 2500 = totalCost plan2 2500 → x = 75 := by
  sorry

#eval totalCost { baseFee := 50, baseMinutes := 500, overageRate := 35/100 } 2500
#eval totalCost { baseFee := 75, baseMinutes := 1000, overageRate := 45/100 } 2500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_plan_fee_is_75_l551_55100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_specific_angles_l551_55116

-- Define a right triangle
structure RightTriangle where
  a : ℝ  -- side a
  b : ℝ  -- side b
  c : ℝ  -- hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Define the properties of the triangle
def triangle_properties (t : RightTriangle) : Prop :=
  ∃ (r m : ℝ),
    r > 0 ∧ m > 0 ∧  -- r is inradius, m is altitude to hypotenuse
    r = 0.45 * m ∧   -- given condition
    r = (t.a + t.b - t.c) / 2  -- inradius formula for right triangle

-- Define the acute angles
noncomputable def acute_angles (t : RightTriangle) : ℝ × ℝ :=
  (Real.arcsin (t.a / t.c), Real.arcsin (t.b / t.c))

-- Theorem statement
theorem right_triangle_specific_angles 
  (t : RightTriangle) 
  (h : triangle_properties t) : 
  let (angle1, angle2) := acute_angles t
  abs (angle1 - 14.805 * π / 180) < 0.001 ∧ 
  abs (angle2 - 75.195 * π / 180) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_specific_angles_l551_55116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_solution_verifies_differential_equation_l551_55180

/-- Parametric function for x -/
noncomputable def x (t : ℝ) : ℝ := Real.log t + Real.sin t

/-- Parametric function for y -/
noncomputable def y (t : ℝ) : ℝ := t * (1 + Real.sin t) + Real.cos t

/-- Derivative of x with respect to t -/
noncomputable def dx_dt (t : ℝ) : ℝ := 1 / t + Real.cos t

/-- Derivative of y with respect to t -/
noncomputable def dy_dt (t : ℝ) : ℝ := t * Real.cos t + 1

/-- The derivative y' -/
noncomputable def y_prime (t : ℝ) : ℝ := dy_dt t / dx_dt t

theorem parametric_solution_verifies_differential_equation (t : ℝ) (h : t > 0) :
  x t = Real.log (y_prime t) + Real.sin (y_prime t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_solution_verifies_differential_equation_l551_55180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_disjoint_infinite_sets_l551_55183

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Three points are collinear if the area of the triangle they form is zero -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- A point is inside a triangle if it's on the same side of all three edges -/
def pointInTriangle (p a b c : Point) : Prop :=
  let sideAB := (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x)
  let sideBC := (p.x - b.x) * (c.y - b.y) - (p.y - b.y) * (c.x - b.x)
  let sideCA := (p.x - c.x) * (a.y - c.y) - (p.y - c.y) * (a.x - c.x)
  (sideAB > 0 ∧ sideBC > 0 ∧ sideCA > 0) ∨ (sideAB < 0 ∧ sideBC < 0 ∧ sideCA < 0)

/-- Main theorem -/
theorem no_disjoint_infinite_sets : 
  ¬∃ (A B : Set Point), 
    (Set.Infinite A ∧ Set.Infinite B ∧ A ∩ B = ∅) ∧
    (∀ p q r, p ∈ A ∪ B → q ∈ A ∪ B → r ∈ A ∪ B → ¬collinear p q r) ∧
    (∀ p q, p ∈ A ∪ B → q ∈ A ∪ B → distance p q ≥ 1) ∧
    (∀ a b c, a ∈ B → b ∈ B → c ∈ B → ∃ p, p ∈ A ∧ pointInTriangle p a b c) ∧
    (∀ a b c, a ∈ A → b ∈ A → c ∈ A → ∃ p, p ∈ B ∧ pointInTriangle p a b c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_disjoint_infinite_sets_l551_55183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_is_zero_l551_55190

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  h1 : d ≠ 0 -- Non-zero common difference
  h2 : ∀ n : ℕ, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum_10_is_zero (seq : ArithmeticSequence) 
    (h : (seq.a 4)^2 + (seq.a 5)^2 = (seq.a 6)^2 + (seq.a 7)^2) : 
    sum_n seq 10 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_is_zero_l551_55190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l551_55120

noncomputable def f (a : ℝ) (φ : ℝ) (x : ℝ) : ℝ := 2 * a * Real.sin (2 * x + φ)

theorem function_properties (a : ℝ) (φ : ℝ) :
  (∀ x, f a φ x ∈ Set.Icc (-2) 2) →
  (∀ x ∈ Set.Icc (-5 * π / 12) (π / 12),
    ∀ y ∈ Set.Icc (-5 * π / 12) (π / 12),
    x < y → f a φ x > f a φ y) →
  0 < φ →
  φ < 2 * π →
  ((a = 1 ∧ φ = 4 * π / 3) ∨ (a = -1 ∧ φ = π / 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l551_55120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_decrease_l551_55132

noncomputable def initial_price : ℝ := 2000
noncomputable def final_price : ℝ := 1280
def num_reductions : ℕ := 2

noncomputable def total_decrease_percentage : ℝ :=
  (initial_price - final_price) / initial_price * 100

noncomputable def average_decrease_percentage : ℝ :=
  total_decrease_percentage / num_reductions

theorem average_price_decrease :
  average_decrease_percentage = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_decrease_l551_55132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_distribution_l551_55162

theorem water_distribution (players : ℕ) (initial_water : ℕ) (spilled_water : ℕ) (leftover_water : ℕ) : 
  players = 30 →
  initial_water = 8000 →
  spilled_water = 250 →
  leftover_water = 1750 →
  (initial_water - spilled_water - leftover_water) / players = 200 := by
  sorry

#check water_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_distribution_l551_55162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_van_distance_theorem_l551_55166

/-- The distance covered by a van -/
noncomputable def distance : ℝ := 450

/-- The initial time taken by the van to cover the distance -/
noncomputable def initial_time : ℝ := 5

/-- The speed the van should maintain to cover the distance in 3/2 of the initial time -/
noncomputable def required_speed : ℝ := 60

/-- The ratio of the new time to the initial time -/
noncomputable def time_ratio : ℝ := 3 / 2

theorem van_distance_theorem :
  distance = required_speed * (time_ratio * initial_time) :=
by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_van_distance_theorem_l551_55166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_formula_correct_l551_55153

/-- A curve in 2D space parameterized by t -/
structure Curve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The first derivative of a curve -/
noncomputable def Curve.derivative (γ : Curve) : Curve where
  x := fun t => deriv γ.x t
  y := fun t => deriv γ.y t

/-- The second derivative of a curve -/
noncomputable def Curve.secondDerivative (γ : Curve) : Curve where
  x := fun t => deriv (deriv γ.x) t
  y := fun t => deriv (deriv γ.y) t

/-- The curvature formula for a parameterized curve -/
noncomputable def curvature (γ : Curve) (t : ℝ) : ℝ :=
  let γ' := γ.derivative
  let γ'' := γ.secondDerivative
  (γ''.x t * γ'.y t - γ''.y t * γ'.x t)^2 / ((γ'.x t)^2 + (γ'.y t)^2)^3

/-- Theorem stating that the given formula correctly calculates the curvature -/
theorem curvature_formula_correct (γ : Curve) (t : ℝ) :
  (curvature γ t)^2 = (γ.secondDerivative.x t * γ.derivative.y t - γ.secondDerivative.y t * γ.derivative.x t)^2 /
    ((γ.derivative.x t)^2 + (γ.derivative.y t)^2)^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_formula_correct_l551_55153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_f_property_l551_55150

open Real

/-- Define the function f --/
noncomputable def f (α : ℝ) : ℝ := 
  cos α * Real.sqrt ((1 / tan α - cos α) / (1 / tan α + cos α)) + 
  sin α * Real.sqrt ((tan α - sin α) / (tan α + sin α))

/-- Main theorem --/
theorem function_f_property (α : ℝ) 
  (h1 : π < α ∧ α < 3*π/2) -- α is in the second quadrant
  (h2 : f (-α) = 1/5) : 
  1 / tan α - 1 / (1 / tan α) = -7/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_f_property_l551_55150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_relationship_l551_55115

open Real MeasureTheory Interval

/-- The value of the integral of x from 0 to 1 -/
noncomputable def a : ℝ := ∫ x in Set.Icc 0 1, x

/-- The value of the integral of x^2 from 0 to 1 -/
noncomputable def b : ℝ := ∫ x in Set.Icc 0 1, x^2

/-- The value of the integral of √x from 0 to 1 -/
noncomputable def c : ℝ := ∫ x in Set.Icc 0 1, Real.sqrt x

/-- Theorem stating the relationship between a, b, and c -/
theorem integral_relationship : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_relationship_l551_55115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_words_per_day_is_ten_l551_55101

/-- Calculates the number of new words learned per day given initial vocabulary,
    percentage increase, and time period. -/
def wordsPerDay (initialWords : ℕ) (percentageIncrease : ℚ) (days : ℕ) : ℚ :=
  (initialWords * percentageIncrease) / days

/-- Theorem stating that given the problem conditions, 10 words are learned per day. -/
theorem words_per_day_is_ten :
  wordsPerDay 14600 (1/2) 730 = 10 := by
  -- Unfold the definition of wordsPerDay
  unfold wordsPerDay
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_words_per_day_is_ten_l551_55101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_sum_l551_55172

def productEvenIntegers (x : Nat) : Nat :=
  if x % 2 = 0 then
    (List.range (x/2)).foldl (fun acc i => acc * (2 * (i + 1))) 1
  else
    0

theorem greatest_prime_factor_of_sum (n : Nat) :
  ∃ (p : Nat), Nat.Prime p ∧ 
  p = (Nat.factors (productEvenIntegers 18 + productEvenIntegers 16)).maximum?.getD 0 ∧ 
  p = 19 := by
  sorry

#eval productEvenIntegers 18 + productEvenIntegers 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_sum_l551_55172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_x100_l551_55148

/-- Function f defined as f(x) = ax + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

/-- Sequence x_n defined recursively -/
def x (a b c : ℝ) : ℕ → ℝ
  | 0 => c  -- Add this case for n = 0
  | 1 => c
  | n + 1 => f a b (x a b c n)

theorem arithmetic_sequence_and_x100 (a b c : ℝ) :
  (∃ d : ℝ, ∀ n : ℕ, n ≥ 2 → x a b c (n + 1) - x a b c n = d) ∧
  x a b c 100 = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_x100_l551_55148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_roots_l551_55140

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the equation
def equation (x : ℝ) : Prop :=
  x^2 - (floor x : ℝ) - 2 = 0

-- Theorem statement
theorem equation_has_two_roots :
  ∃ (a b : ℝ), a ≠ b ∧ equation a ∧ equation b ∧
  (∀ (c : ℝ), equation c → c = a ∨ c = b) :=
by
  -- The proof goes here
  sorry

#check equation_has_two_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_roots_l551_55140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_is_five_l551_55138

def S : Finset ℕ := {1, 2, 3, 4, 5}

def pairs : Finset (ℕ × ℕ) := S.product S

def valid_pairs : Finset (ℕ × ℕ) := pairs.filter (λ p => p.1 < p.2)

def sum_is_five (p : ℕ × ℕ) : Bool := p.1 + p.2 = 5

theorem probability_sum_is_five :
  (valid_pairs.filter (λ p => sum_is_five p)).card / valid_pairs.card = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_is_five_l551_55138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicularity_l551_55146

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relations
def parallel (l : Line) (p : Plane) : Prop := sorry

def perpendicular (l : Line) (p : Plane) : Prop := sorry

def perpendicular_planes (p1 : Plane) (p2 : Plane) : Prop := sorry

-- State the theorem
theorem line_plane_perpendicularity 
  (l : Line) (α β : Plane) (h_diff : α ≠ β) :
  parallel l α → perpendicular l β → perpendicular_planes α β :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicularity_l551_55146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_three_fourths_l551_55198

/-- The sum of the infinite series ∑(k/3^k) for k from 1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' k, k / (3 : ℝ) ^ k

/-- The theorem states that the sum of the infinite series ∑(k/3^k) for k from 1 to infinity equals 3/4 -/
theorem infinite_series_sum_equals_three_fourths : infinite_series_sum = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_three_fourths_l551_55198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selenas_tip_l551_55137

theorem selenas_tip (steak_price burger_price icecream_price : ℚ)
  (steak_quantity burger_quantity icecream_quantity : ℕ)
  (remaining_money : ℚ)
  (h1 : steak_price = 24)
  (h2 : burger_price = 7/2)
  (h3 : icecream_price = 2)
  (h4 : steak_quantity = 2)
  (h5 : burger_quantity = 2)
  (h6 : icecream_quantity = 3)
  (h7 : remaining_money = 38) :
  steak_price * steak_quantity +
  burger_price * burger_quantity +
  icecream_price * icecream_quantity +
  remaining_money = 99 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_selenas_tip_l551_55137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_arrive_together_l551_55149

structure Student where
  name : String
  speed : ℚ
  deriving Repr

def walkTogether (s1 s2 : Student) : Student :=
  { name := s1.name ++ " and " ++ s2.name, speed := min s1.speed s2.speed }

def canCatchUp (s1 s2 : Student) (headStart : ℚ) : Prop :=
  s1.speed > s2.speed

structure DayScenario where
  order : List Student
  distanceToSchool : ℚ

noncomputable def simulateDay (scenario : DayScenario) : List Student :=
  sorry

theorem all_arrive_together (anya borya vasya : Student) 
  (h1 : borya.speed > anya.speed)
  (h2 : anya.speed > vasya.speed)
  (day1 : DayScenario)
  (day2 : DayScenario)
  (h3 : day1.order = [anya, borya, vasya])
  (h4 : day2.order = [vasya, borya, anya]) :
  ∃ (result : List Student), simulateDay day2 = result ∧ result.length = 1 :=
by
  sorry

#check all_arrive_together

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_arrive_together_l551_55149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_min_S_value_l551_55152

-- Define the sequence a_n and its sum S_n
def a : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry

-- The given condition
axiom condition (n : ℕ) : 2 * S n / n + n = 2 * a n + 1

-- a_4, a_7, and a_9 form a geometric sequence
axiom geometric_seq : (a 7) ^ 2 = (a 4) * (a 9)

-- Theorem 1: {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d := by
  sorry

-- Theorem 2: The minimum value of S_n is -78
theorem min_S_value : ∃ n : ℕ, S n = -78 ∧ ∀ m : ℕ, S m ≥ -78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_min_S_value_l551_55152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_bound_l551_55105

/-- The ellipse with equation x²/4 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 4 + p.2^2 = 1}

/-- The foci of the ellipse -/
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

/-- The dot product of vectors PF₁ and PF₂ -/
noncomputable def dotProduct (p : ℝ × ℝ) : ℝ :=
  (F₁.1 - p.1) * (F₂.1 - p.1) + (F₁.2 - p.2) * (F₂.2 - p.2)

theorem ellipse_dot_product_bound :
  ∀ p ∈ Ellipse, -2 ≤ dotProduct p ∧ dotProduct p ≤ 1 := by
  sorry

#check ellipse_dot_product_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_bound_l551_55105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_implies_five_green_marbles_l551_55159

def bag1_total : ℕ := 4
def bag1_red : ℕ := 2
def bag1_blue : ℕ := 2

def bag2_red : ℕ := 2
def bag2_blue : ℕ := 2

def probability_same_color (total : ℕ) (red : ℕ) (blue : ℕ) (green : ℕ) : ℚ :=
  (red * (red - 1) / 2 + blue * (blue - 1) / 2 + green * (green - 1) / 2) / (total * (total - 1) / 2)

theorem equal_probability_implies_five_green_marbles :
  ∀ g : ℕ, g > 0 →
  probability_same_color bag1_total bag1_red bag1_blue 0 =
  probability_same_color (bag2_red + bag2_blue + g) bag2_red bag2_blue g →
  g = 5 := by
  sorry

#check equal_probability_implies_five_green_marbles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_implies_five_green_marbles_l551_55159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_m_neg_one_union_equals_B_iff_intersection_empty_iff_l551_55175

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem 1
theorem union_when_m_neg_one :
  A ∪ B (-1) = Set.Ioo (-2) 3 := by sorry

-- Theorem 2
theorem union_equals_B_iff (m : ℝ) :
  A ∪ B m = B m ↔ m ≤ -2 := by sorry

-- Theorem 3
theorem intersection_empty_iff (m : ℝ) :
  A ∩ B m = ∅ ↔ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_m_neg_one_union_equals_B_iff_intersection_empty_iff_l551_55175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l551_55109

theorem percentage_difference : 
  let x : ℝ := 820
  let percentage_x : ℝ := 0.25
  let y : ℝ := 1500
  let percentage_y : ℝ := 0.15
  let difference : ℝ := percentage_y * y - percentage_x * x
  difference = 20 := by
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l551_55109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l551_55161

/-- A structure representing a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A function to check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- A function to check if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

/-- A theorem stating that under the given conditions, the radius of circle B is 20/9 -/
theorem circle_radius_problem (A B C D : Circle) : 
  are_externally_tangent A B → 
  are_externally_tangent A C → 
  are_externally_tangent B C → 
  is_internally_tangent A D → 
  is_internally_tangent B D → 
  is_internally_tangent C D → 
  B.radius = C.radius →
  A.radius = 2 →
  (D.center.1 - A.center.1)^2 + (D.center.2 - A.center.2)^2 = A.radius^2 →
  B.radius = 20/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l551_55161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l551_55121

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define the line L
def L (a b x y : ℝ) : Prop := 2*a*x + b*y + 6 = 0

-- Define the symmetry condition
def symmetry (a b : ℝ) : Prop := ∃ (x y : ℝ), C x y ∧ L a b (-1) 2

-- Define the tangent length function
noncomputable def tangent_length (a b : ℝ) : ℝ := 
  Real.sqrt ((a + 1)^2 + (b - 2)^2) - Real.sqrt 2

-- Theorem statement
theorem min_tangent_length (a b : ℝ) : 
  (∀ x y, C x y → L a b x y → symmetry a b) → 
  ∃ (min_length : ℝ), min_length = 4 ∧ ∀ (a' b' : ℝ), tangent_length a' b' ≥ min_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l551_55121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_initial_black_marbles_l551_55151

/-- The number of black marbles Sara had initially -/
def initial_black_marbles : ℕ := sorry

/-- The number of red marbles Sara had initially -/
def initial_red_marbles : ℕ := 122

/-- The number of black marbles Fred took -/
def taken_black_marbles : ℕ := 233

/-- The number of black marbles Sara has now -/
def current_black_marbles : ℕ := 559

theorem sara_initial_black_marbles : 
  initial_black_marbles = current_black_marbles + taken_black_marbles :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_initial_black_marbles_l551_55151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l551_55145

-- Define the basic concepts
def Line : Type := Unit
def Plane : Type := Unit
def Point : Type := Unit

-- Define the relations
def Parallel (a b : Line) : Prop := sorry
def Perpendicular (l : Line) (α : Plane) : Prop := sorry
def SkewLines (a b : Line) : Prop := sorry
def ParallelPlanes (α β : Plane) : Prop := sorry
def LiesIn (l : Line) (α : Plane) : Prop := sorry
def Intersect (a b : Line) : Prop := sorry
def Equidistant (p : Point) (α : Plane) : Prop := sorry
def NonCollinear (p q r : Point) : Prop := sorry

-- State the theorem
theorem correct_propositions :
  -- Proposition ②
  (∀ (l : Line) (α : Plane),
    (∀ (m : Line), LiesIn m α → Perpendicular l m) ↔ Perpendicular l α) ∧
  -- Proposition ④
  (∀ (α β : Plane),
    ParallelPlanes α β →
    ∃ (p q r : Point),
      NonCollinear p q r ∧
      Equidistant p β ∧ Equidistant q β ∧ Equidistant r β) ∧
  -- Proposition ① is incorrect
  ¬(∀ (a b : Line) (α : Plane),
    LiesIn b α →
    (Parallel a b ↔ ∀ (c : Line), LiesIn c α → Parallel a c)) ∧
  -- Proposition ③ is incorrect
  ¬(∀ (a b : Line),
    ¬Intersect a b → SkewLines a b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l551_55145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_book_pairs_l551_55142

/-- Represents the set of textbooks -/
inductive Textbook
  | Chinese
  | Mathematics
  | English
  | Biology
  | History
deriving Repr, DecidableEq, Fintype

/-- The number of textbooks -/
def num_textbooks : Nat := 5

/-- Function to count pairs with Mathematics -/
def count_pairs_with_math (books : Finset Textbook) : Nat :=
  (books.filter (· ≠ Textbook.Mathematics)).card

/-- Theorem: The number of ways to select two books where one is Mathematics is 4 -/
theorem math_book_pairs :
  count_pairs_with_math (Finset.univ : Finset Textbook) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_book_pairs_l551_55142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l551_55196

-- Define the ellipse C passing through A(√3, 0) and B(0, 2)
def ellipse_C (x y : ℝ) : Prop := y^2 / 4 + x^2 / 3 = 1

-- Define points A and B
noncomputable def point_A : ℝ × ℝ := (Real.sqrt 3, 0)
def point_B : ℝ × ℝ := (0, 2)

-- Define the area of triangle ABP
noncomputable def triangle_area (p : ℝ × ℝ) : ℝ :=
  abs ((p.1 - point_A.1) * (point_B.2 - point_A.2) - (point_B.1 - point_A.1) * (p.2 - point_A.2)) / 2

-- Theorem statement
theorem ellipse_properties :
  -- 1. The equation of the ellipse
  (∀ x y : ℝ, ellipse_C x y ↔ y^2 / 4 + x^2 / 3 = 1) ∧
  -- 2. The maximum area of triangle ABP
  (∃ max_area : ℝ, max_area = Real.sqrt 6 + Real.sqrt 3 ∧
    ∀ p : ℝ × ℝ, ellipse_C p.1 p.2 → triangle_area p ≤ max_area) ∧
  -- 3. The coordinates of P when the area is maximum
  (∃ p : ℝ × ℝ, p = (-Real.sqrt 6 / 2, -Real.sqrt 2) ∧
    ellipse_C p.1 p.2 ∧
    ∀ q : ℝ × ℝ, ellipse_C q.1 q.2 → triangle_area q ≤ triangle_area p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l551_55196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l551_55110

/-- Double absolute difference operation -/
def double_abs_diff (a b c d : ℝ) : ℝ := |a - b| - |c - d|

theorem problem_statement :
  /- Statement 1 -/
  (Finset.image (λ p : Fin 4 × Fin 4 ↦ double_abs_diff 24 25 29 30) Finset.univ).card = 3 ∧
  /- Statement 2 -/
  ∀ x : ℝ, x ≥ 2 → (double_abs_diff (x^2) (2*x) 1 1 = 7 → x^4 + 2401 / x^4 = 226) ∧
  /- Statement 3 -/
  ∀ x : ℝ, x ≥ -2 → 
    double_abs_diff (2*x - 5) (3*x - 2) (4*x - 1) (5*x + 3) ≠ 0 ∧
    double_abs_diff (2*x - 5) (4*x - 1) (3*x - 2) (5*x + 3) ≠ 0 ∧
    double_abs_diff (2*x - 5) (5*x + 3) (3*x - 2) (4*x - 1) ≠ 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l551_55110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l551_55174

-- Define the lines
def line_through_origin (x y : ℝ) : Prop := ∃ (m : ℝ), y = m * x
def vertical_line (x₀ : ℝ) (x : ℝ) : Prop := x = x₀
def sloped_line (x y : ℝ) : Prop := y = 2 - (1/2) * x

-- Define the triangle
def right_triangle (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  line_through_origin x₁ y₁ ∧
  vertical_line 2 x₂ ∧
  sloped_line x₃ y₃ ∧
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0  -- Right angle condition

-- Calculate distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Define the perimeter of the triangle
noncomputable def perimeter (A B C : ℝ × ℝ) : ℝ :=
  distance A B + distance B C + distance C A

-- Theorem statement
theorem triangle_perimeter :
  ∀ A B C : ℝ × ℝ, right_triangle A B C → perimeter A B C = 3 + 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l551_55174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repaired_shoes_lifespan_l551_55106

/-- The lifespan of repaired shoes given specific cost conditions -/
theorem repaired_shoes_lifespan :
  let repair_cost : ℝ := 11.50
  let new_shoes_cost : ℝ := 28.00
  let new_shoes_lifespan : ℝ := 2
  let cost_difference_percentage : ℝ := 21.73913043478261

  let new_shoes_yearly_cost : ℝ := new_shoes_cost / new_shoes_lifespan
  let repaired_shoes_lifespan : ℝ := repair_cost / (new_shoes_yearly_cost / (1 + cost_difference_percentage / 100))

  ∃ (ε : ℝ), ε > 0 ∧ |repaired_shoes_lifespan - 0.6745| < ε := by
  sorry

#eval Float.toString ((11.50 : Float) / (14.00 / 1.2173913043478261))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repaired_shoes_lifespan_l551_55106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_girls_at_park_correct_l551_55186

def number_of_girls_at_park (num_boys : ℕ) (num_parents : ℕ) (num_groups : ℕ) (group_size : ℕ) : ℕ :=
  let total_people := num_groups * group_size
  total_people - num_boys - num_parents

theorem number_of_girls_at_park_correct 
  (num_boys : ℕ) (num_parents : ℕ) (num_groups : ℕ) (group_size : ℕ) :
  number_of_girls_at_park num_boys num_parents num_groups group_size = 
  num_groups * group_size - num_boys - num_parents := by
  unfold number_of_girls_at_park
  rfl

#eval number_of_girls_at_park 11 50 3 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_girls_at_park_correct_l551_55186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l551_55103

theorem sin_cos_identity (θ : ℝ) (a : ℝ) (h1 : 0 < θ ∧ θ < π/2) (h2 : Real.cos (2*θ) = a) :
  Real.sin θ * Real.cos θ = Real.sqrt (1 - a^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l551_55103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_equals_4_l551_55136

def sequence_a : ℕ → ℕ
  | 0 => 2  -- We define a value for 0 to cover all natural numbers
  | 1 => 2
  | 2 => 7
  | n + 3 => (sequence_a (n + 1) * sequence_a (n + 2)) % 10

theorem a_2013_equals_4 : sequence_a 2013 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_equals_4_l551_55136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_capacity_X_l551_55118

/-- Represents the capacity of a drum -/
structure Drum where
  capacity : ℝ
  filled : ℝ
  h_filled_le_capacity : filled ≤ capacity

/-- The problem setup -/
structure DrumProblem where
  X : Drum
  Y : Drum
  h_Y_capacity : Y.capacity = 2 * X.capacity
  h_initial_fill : X.filled = Y.filled
  h_final_fill : X.filled + Y.filled = 0.75 * Y.capacity

theorem initial_capacity_X (problem : DrumProblem) : 
  problem.X.filled = 0.75 * problem.X.capacity := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_capacity_X_l551_55118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l551_55193

/-- Helper function to calculate the volume of a cone -/
noncomputable def volume_of_cone (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The volume of a cone with base radius R and lateral surface sector angle of 90° -/
theorem cone_volume (R : ℝ) (h : R > 0) : 
  ∃ V : ℝ, V = (Real.pi * R^3 * Real.sqrt 15) / 3 ∧ 
  V = volume_of_cone R (R * Real.sqrt 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l551_55193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_approximation_accurate_l551_55141

open Real MeasureTheory

-- Define the function to be integrated
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

-- Define the trapezoidal rule
noncomputable def trapezoidalRule (a b : ℝ) (n : ℕ) (f : ℝ → ℝ) : ℝ :=
  let h := (b - a) / n
  let sumTerms := (Finset.range (n+1)).sum (fun i => 
    let x := a + i * h
    if i = 0 || i = n then f x / 2 else f x)
  h * sumTerms

-- State the theorem
theorem trapezoidal_approximation_accurate :
  let a := 2
  let b := 3
  let n := 5
  let approx := trapezoidalRule a b n f
  abs (approx - 0.6956) < 0.01 ∧ 
  abs (approx - ∫ x in a..b, f x) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_approximation_accurate_l551_55141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_gt_3_sufficient_not_necessary_l551_55163

-- Define the circle
def myCircle (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = r^2}

-- Define the line
def myLine : Set (ℝ × ℝ) := {p | p.1 + Real.sqrt 3 * p.2 - 2 = 0}

-- Define the intersection condition
def intersects (r : ℝ) : Prop := ∃ p, p ∈ myCircle r ∩ myLine

-- Theorem statement
theorem r_gt_3_sufficient_not_necessary (r : ℝ) (h : r > 0) :
  (r > 3 → intersects r) ∧ ¬(intersects r → r > 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_gt_3_sufficient_not_necessary_l551_55163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l551_55129

/-- The function f(x) = x³ + ax² - (4/3)a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - (4/3)*a

/-- The derivative of f with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x

theorem function_property (a : ℝ) : 
  (∃ x : ℝ, f_derivative a x = 0 ∧ f a x = 0) → 
  (a = 0 ∨ a = 3 ∨ a = -3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l551_55129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l551_55195

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 else x^3 - (a-1)*x + a^2 - 3*a - 4

theorem f_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

#check f_increasing_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l551_55195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_characterization_l551_55173

def is_valid_sequence (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∃ (d : ℕ → ℕ), 
    (∀ i, d i ∣ n) ∧ 
    (∀ i j, 1 ≤ i → i < j → j < Nat.card {x : ℕ | x ∣ n} → d i ≠ d j) ∧
    (∃ r, ∀ i, 1 ≤ i → i < Nat.card {x : ℕ | x ∣ n} → a i = a 1 + (i - 1) * r) ∧
    (∀ i, 1 ≤ i → i < Nat.card {x : ℕ | x ∣ n} → d i = Nat.gcd (a i) n)

theorem valid_n_characterization (n : ℕ) : 
  (4 ≤ Nat.card {x : ℕ | x ∣ n}) → 
  (∃ a, is_valid_sequence n a) ↔ 
  (n = 8 ∨ n = 12 ∨ ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_characterization_l551_55173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_property_l551_55114

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_symmetry : ∀ x : ℝ, f (4 - x) = f x
axiom f_derivative_property : ∀ x : ℝ, (x - 2) * deriv f x < 0

-- State the theorem
theorem f_decreasing_property (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₁ + x₂ > 4) :
  f x₁ > f x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_property_l551_55114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_ratio_l551_55170

-- Define the time it takes for Pipe A to fill the tank
noncomputable def pipe_a_time : ℝ := 30

-- Define the time it takes for both pipes to fill the tank
noncomputable def both_pipes_time : ℝ := 5

-- Define Pipe A's filling rate
noncomputable def pipe_a_rate : ℝ := 1 / pipe_a_time

-- Define the combined filling rate of both pipes
noncomputable def combined_rate : ℝ := 1 / both_pipes_time

-- Define Pipe B's filling rate
noncomputable def pipe_b_rate : ℝ := combined_rate - pipe_a_rate

-- Theorem: The ratio of Pipe B's rate to Pipe A's rate is 5:1
theorem pipe_ratio : pipe_b_rate / pipe_a_rate = 5 := by
  -- Expand definitions
  unfold pipe_b_rate pipe_a_rate combined_rate pipe_a_time both_pipes_time
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_ratio_l551_55170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_three_l551_55177

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function f
noncomputable def f (x : ℂ) : ℂ :=
  if x.im = 0 then 1 + x else (1 + i) * x

-- State the theorem
theorem f_composition_equals_three :
  f (f (1 - i)) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_three_l551_55177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertices_without_diagonal_l551_55135

/-- A non-convex n-gon is a polygon with n sides that is not convex. -/
structure NonConvexNGon (n : ℕ) where
  n_ge_4 : n ≥ 4  -- A non-convex polygon must have at least 4 sides

/-- A diagonal in a polygon is a line segment that connects two non-adjacent vertices. -/
def has_diagonal (n : ℕ) (p : NonConvexNGon n) (v : Fin n) : Prop :=
  -- Definition of a diagonal from vertex v
  sorry

/-- The maximum number of vertices in a non-convex n-gon from which no diagonal can be drawn is ⌊n/2⌋. -/
theorem max_vertices_without_diagonal (n : ℕ) (p : NonConvexNGon n) :
  (∃ (S : Finset (Fin n)), (∀ v ∈ S, ¬has_diagonal n p v) ∧ S.card = n / 2) ∧
  (∀ (T : Finset (Fin n)), (∀ v ∈ T, ¬has_diagonal n p v) → T.card ≤ n / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertices_without_diagonal_l551_55135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_characterization_l551_55156

/-- Triangle ABC with side lengths a, b, c opposite to vertices A, B, C respectively -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point inside a triangle -/
def InternalPoint (t : Triangle) := ℝ × ℝ

/-- Distance from a point to a line segment -/
noncomputable def distanceToSide (p : ℝ × ℝ) (s : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- Sum of squares of distances from a point to the sides of a triangle -/
noncomputable def sumSquaredDistances (t : Triangle) (p : InternalPoint t) : ℝ :=
  (distanceToSide p (t.B, t.C))^2 + (distanceToSide p (t.C, t.A))^2 + (distanceToSide p (t.A, t.B))^2

/-- Point that divides a line segment in a given ratio -/
noncomputable def divideSegment (s : (ℝ × ℝ) × (ℝ × ℝ)) (r : ℝ) : ℝ × ℝ := sorry

/-- Theorem: The point that minimizes the sum of squared distances is at the intersection of specific lines -/
theorem min_distance_point_characterization (t : Triangle) :
  ∃ (O : InternalPoint t),
    (∀ (P : InternalPoint t), sumSquaredDistances t O ≤ sumSquaredDistances t P) ∧
    (O = (divideSegment (t.C, divideSegment (t.A, t.B) (t.a^2 / (t.a^2 + t.b^2))) (t.c^2 / (t.b^2 + t.c^2)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_characterization_l551_55156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l551_55147

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem statement -/
theorem ellipse_triangle_perimeter
  (C : Ellipse)
  (A D E : Point)
  (h_eccentricity : Real.sqrt (1 - C.b^2 / C.a^2) = 1/2)
  (h_upper_vertex : A.y = C.b ∧ A.x = 0)
  (h_DE_perpendicular : ∃ (F₁ F₂ : Point), 
    F₁.y = 0 ∧ F₂.y = 0 ∧ 
    (D.y - F₁.y) * (A.x - F₂.x) = (A.y - F₂.y) * (D.x - F₁.x))
  (h_DE_on_ellipse : 
    D.x^2 / C.a^2 + D.y^2 / C.b^2 = 1 ∧
    E.x^2 / C.a^2 + E.y^2 / C.b^2 = 1)
  (h_DE_length : distance D E = 6) :
  distance A D + distance D E + distance E A = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l551_55147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_l551_55171

/-- The equation of circle D -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 2*y - 8 = -y^2 + 6*x

/-- The center of circle D -/
def center : ℝ × ℝ := (3, -1)

/-- The radius of circle D -/
noncomputable def radius : ℝ := 3 * Real.sqrt 2

/-- Theorem: The sum of the center coordinates and radius equals 2 + 3√2 -/
theorem circle_sum :
  let (c, d) := center
  c + d + radius = 2 + 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_l551_55171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_percentage_change_l551_55165

/-- Given an original price, calculate the final price after a series of percentage changes -/
noncomputable def finalPrice (originalPrice : ℝ) : ℝ :=
  originalPrice * 1.15 * 1.20 * 0.90

/-- Calculate the equivalent single percentage change -/
noncomputable def equivalentPercentageChange (originalPrice : ℝ) : ℝ :=
  (finalPrice originalPrice - originalPrice) / originalPrice * 100

/-- Theorem: The equivalent single percentage change is 24.2% -/
theorem equivalent_percentage_change :
  ∀ (p : ℝ), p > 0 → equivalentPercentageChange p = 24.2 := by
  intro p hp
  -- The proof steps would go here
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_percentage_change_l551_55165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_407_list_l551_55176

def repeat_407 (n : ℕ) : ℕ :=
  let s := String.join (List.replicate n "407")
  s.toNat!

def list_407 : List ℕ := List.map repeat_407 (List.range (Nat.succ 0))

theorem no_primes_in_407_list : ∀ n ∈ list_407, ¬ Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_407_list_l551_55176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_AB_is_correct_l551_55126

noncomputable def A : ℝ × ℝ := (1, 3)
noncomputable def B : ℝ × ℝ := (4, -1)

noncomputable def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

noncomputable def magnitude_AB : ℝ := Real.sqrt (vector_AB.1^2 + vector_AB.2^2)

noncomputable def unit_vector_AB : ℝ × ℝ := (vector_AB.1 / magnitude_AB, vector_AB.2 / magnitude_AB)

theorem unit_vector_AB_is_correct : unit_vector_AB = (3/5, -4/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_AB_is_correct_l551_55126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_pi_4_l551_55130

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x ^ 8 + Real.cos x ^ 8 + Real.sin x ^ 2 * Real.cos x ^ 2) / 
  (Real.sin x ^ 6 + Real.cos x ^ 6 + 2)

theorem f_min_at_pi_4 : ∀ x : ℝ, f x ≥ f (Real.pi / 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_pi_4_l551_55130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_values_l551_55107

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sqrt 3 * Real.sin x + Real.cos x ^ 3) + Real.sin x * (Real.sqrt 3 * Real.cos x - Real.sin x ^ 3)

theorem triangle_side_values (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  Real.sin A / a = Real.sin B / b →
  Real.sin A / a = Real.sin C / c →
  a^2 + c^2 = a*c + b^2 →
  f A = 0 →
  b + c = Real.sqrt 2 + Real.sqrt 3 →
  b = Real.sqrt 3 ∧ c = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_values_l551_55107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_minus_one_pow_zero_plus_cube_root_neg_27_l551_55192

theorem sqrt_minus_one_pow_zero_plus_cube_root_neg_27 :
  (Real.sqrt 2 - 1)^0 + ((-27 : ℝ) ^ (1/3 : ℝ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_minus_one_pow_zero_plus_cube_root_neg_27_l551_55192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_perpendicular_condition_l551_55104

-- Define the vectors
def a : ℝ × ℝ := (5, -12)
def b : ℝ × ℝ := (-3, 4)

-- Theorem for the cosine of the angle between vectors
theorem cosine_of_angle (a b : ℝ × ℝ) : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -63/65 := by
  sorry

-- Theorem for the value of t when a + tb is perpendicular to a - b
theorem perpendicular_condition (a b : ℝ × ℝ) (t : ℝ) :
  (a.1 + t * b.1) * (a.1 - b.1) + (a.2 + t * b.2) * (a.2 - b.2) = 0 → t = 29/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_perpendicular_condition_l551_55104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_weight_l551_55131

theorem peanut_weight (total_weight raisin_weight peanut_weight : ℚ) 
  (h1 : total_weight = 1/2)
  (h2 : raisin_weight = 2/5)
  (h3 : total_weight = raisin_weight + peanut_weight) :
  peanut_weight = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_weight_l551_55131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_theorem_l551_55167

theorem chord_cosine_theorem (r : ℝ) (α β : ℝ) : 
  0 < r ∧ 
  0 < α ∧ 
  0 < β ∧ 
  α + β < π ∧
  2 * r * Real.sin (α/2) = 2 ∧
  2 * r * Real.sin (β/2) = 3 ∧
  2 * r * Real.sin ((α+β)/2) = 4 →
  Real.cos α = 17/32 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_theorem_l551_55167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_person_speed_l551_55102

/-- Calculates the speed of the second person given the speed of the first person,
    the total distance traveled, and the time elapsed. -/
noncomputable def calculate_speed (speed1 : ℝ) (total_distance : ℝ) (time : ℝ) : ℝ :=
  (total_distance - speed1 * time) / time

/-- Theorem stating that given the conditions of the problem,
    the speed of the second person is 27 km/h. -/
theorem second_person_speed :
  let speed1 : ℝ := 18 -- km/h
  let total_distance : ℝ := 45 -- km
  let time : ℝ := 1 -- hour
  calculate_speed speed1 total_distance time = 27 := by
  -- Unfold the definition of calculate_speed
  unfold calculate_speed
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_person_speed_l551_55102
