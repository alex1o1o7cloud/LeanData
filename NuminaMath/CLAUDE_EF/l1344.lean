import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arccos_cos_l1344_134442

open MeasureTheory

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.arccos (Real.cos x)

-- Define the interval
def interval : Set ℝ := Set.Icc 0 (2 * Real.pi)

-- State the theorem
theorem area_arccos_cos : ∫ x in interval, f x = Real.pi ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arccos_cos_l1344_134442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_less_than_M_over_100_l1344_134403

def M : ℚ := 20 * (2^19 - (1 + 20 + 190))

theorem greatest_integer_less_than_M_over_100 :
  let sum := (1 / (3 * 2 * 1 * Nat.factorial 18)) + (1 / (4 * 3 * 2 * 1 * Nat.factorial 17)) +
             (1 / (5 * 4 * 3 * 2 * 1 * Nat.factorial 16)) + (1 / (6 * 5 * 4 * 3 * 2 * 1 * Nat.factorial 15)) +
             (1 / (7 * 6 * 5 * 4 * 3 * 2 * 1 * Nat.factorial 14)) + (1 / (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 * Nat.factorial 13)) +
             (1 / (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 * Nat.factorial 12)) + (1 / (10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 * Nat.factorial 11))
  sum = M / (1 * Nat.factorial 19) →
  ⌊M / 100⌋ = 262 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_less_than_M_over_100_l1344_134403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l1344_134484

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, a * x^2 + 2 * a * x + 1 ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, a * x^2 + 2 * a * x + 1 = 4) →
  a = -3 ∨ a = 1/3 := by
  sorry

#check quadratic_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l1344_134484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1344_134476

theorem power_equality (a b : ℝ) (h1 : (30 : ℝ)^a = 4) (h2 : (30 : ℝ)^b = 9) :
  (18 : ℝ)^((1 - a - b) / (2 * (1 - b))) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1344_134476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_l1344_134473

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x/2) * Real.cos (x/2) + (Real.cos (x/2))^2 - 1/2

-- Define the theorem
theorem angle_C_value (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  A + B + C = Real.pi →
  a > 0 → b > 0 → c > 0 →
  -- Given conditions
  f (B + C) = 1 →
  a = Real.sqrt 3 →
  b = 1 →
  -- Law of sines
  Real.sin A / a = Real.sin B / b →
  Real.sin A / a = Real.sin C / c →
  -- Conclusion
  C = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_l1344_134473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_time_correct_l1344_134452

/-- The time when two ships are at the shortest distance from each other -/
noncomputable def shortest_distance_time (initial_distance : ℝ) (speed1 speed2 : ℝ) (angle : ℝ) : ℝ :=
  5 / 14

theorem shortest_distance_time_correct (initial_distance speed1 speed2 angle : ℝ) 
  (h1 : initial_distance = 10)
  (h2 : speed1 = 4)
  (h3 : speed2 = 6)
  (h4 : angle = 60 * Real.pi / 180) :
  shortest_distance_time initial_distance speed1 speed2 angle = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_time_correct_l1344_134452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_CD_length_l1344_134478

/-- Represents a tetrahedron with vertices A, B, C, and D -/
structure Tetrahedron where
  AB : ℝ
  AC : ℝ
  AD : ℝ
  BC : ℝ
  BD : ℝ
  CD : ℝ

/-- The set of edge lengths for our specific tetrahedron -/
def tetrahedron_edges : Set ℝ := {8, 15, 17, 29, 34, 40}

/-- Theorem stating that for a tetrahedron with the given edge lengths and AB = 40, CD must be 8 -/
theorem tetrahedron_CD_length 
  (t : Tetrahedron) 
  (h1 : t.AB = 40) 
  (h2 : t.AB ∈ tetrahedron_edges)
  (h3 : t.AC ∈ tetrahedron_edges)
  (h4 : t.AD ∈ tetrahedron_edges)
  (h5 : t.BC ∈ tetrahedron_edges)
  (h6 : t.BD ∈ tetrahedron_edges)
  (h7 : t.CD ∈ tetrahedron_edges)
  (h8 : t.AB + t.AC > t.BC) -- Triangle inequality for ABC
  (h9 : t.AB + t.BC > t.AC)
  (h10 : t.AC + t.BC > t.AB)
  (h11 : t.AB + t.AD > t.BD) -- Triangle inequality for ABD
  (h12 : t.AB + t.BD > t.AD)
  (h13 : t.AD + t.BD > t.AB)
  (h14 : t.AC + t.AD > t.CD) -- Triangle inequality for ACD
  (h15 : t.AC + t.CD > t.AD)
  (h16 : t.AD + t.CD > t.AC)
  (h17 : t.BC + t.BD > t.CD) -- Triangle inequality for BCD
  (h18 : t.BC + t.CD > t.BD)
  (h19 : t.BD + t.CD > t.BC)
  : t.CD = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_CD_length_l1344_134478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_theorem_l1344_134449

/-- An arithmetic sequence with first term 1 and common difference d > 1 -/
noncomputable def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ :=
  1 + (n - 1 : ℝ) * d

/-- Sum of first n terms of the arithmetic sequence -/
noncomputable def S (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 + (n - 1 : ℝ) * d) / 2

/-- Sequence c_n defined as 1 / (a_n * a_n+1) -/
noncomputable def c (d : ℝ) (n : ℕ) : ℝ :=
  1 / (arithmetic_sequence d n * arithmetic_sequence d (n + 1))

/-- Sum of first n terms of the sequence c_n -/
noncomputable def T (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / (2 * n + 1 : ℝ)

theorem arithmetic_sequence_theorem (d : ℝ) (h1 : d > 1) 
  (h2 : S d 4 - 2 * arithmetic_sequence d 2 * arithmetic_sequence d 3 + 14 = 0) :
  (∀ n : ℕ, arithmetic_sequence d n = 2 * n - 1) ∧
  (∀ n : ℕ, T d n = (n : ℝ) / (2 * n + 1 : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_theorem_l1344_134449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1344_134406

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def Line.passes_through (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

noncomputable def Line.x_intercept (l : Line) : ℝ :=
  -l.c / l.a

noncomputable def Line.y_intercept (l : Line) : ℝ :=
  -l.c / l.b

noncomputable def Line.triangle_area (l : Line) : ℝ :=
  abs (l.x_intercept * l.y_intercept) / 2

theorem line_equation (l : Line) :
  l.passes_through 2 3 ∧
  l.x_intercept + l.y_intercept = 0 ∧
  l.triangle_area = 16 →
  (l.a = 1 ∧ l.b = 2 ∧ l.c = -8) ∨ (l.a = 9 ∧ l.b = 2 ∧ l.c = -24) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1344_134406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l1344_134491

/-- Represents the composition of lemonade -/
structure LemonadeRecipe where
  lemonJuice : ℝ
  honey : ℝ
  water : ℝ

/-- Calculates the total weight of the lemonade -/
noncomputable def totalWeight (recipe : LemonadeRecipe) : ℝ :=
  recipe.lemonJuice + recipe.honey + recipe.water

/-- Calculates the total calories in the lemonade -/
noncomputable def totalCalories (recipe : LemonadeRecipe) (lemonJuiceCaloriesPer100g : ℝ) (honeyCaloriesPer100g : ℝ) : ℝ :=
  (recipe.lemonJuice * lemonJuiceCaloriesPer100g / 100) + (recipe.honey * honeyCaloriesPer100g / 100)

/-- Calculates the calories in a given weight of lemonade -/
noncomputable def caloriesInWeight (recipe : LemonadeRecipe) (lemonJuiceCaloriesPer100g : ℝ) (honeyCaloriesPer100g : ℝ) (weight : ℝ) : ℝ :=
  (totalCalories recipe lemonJuiceCaloriesPer100g honeyCaloriesPer100g * weight) / totalWeight recipe

/-- Theorem: The number of calories in 250g of Veronica's lemonade is approximately 192 -/
theorem lemonade_calories : 
  let recipe := LemonadeRecipe.mk 150 200 500
  let lemonJuiceCaloriesPer100g := 30
  let honeyCaloriesPer100g := 304
  let targetWeight := 250
  abs (caloriesInWeight recipe lemonJuiceCaloriesPer100g honeyCaloriesPer100g targetWeight - 192) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l1344_134491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_speed_l1344_134458

noncomputable section

/-- The speed of sound in m/s -/
def c : ℝ := 343

/-- The distance from the origin to the first siren in meters -/
def L : ℝ := 50

/-- The speed of the boat and bicycle in m/s -/
def U_b : ℝ := 20 * 1000 / 3600

/-- The distance from the shore where we want to calculate the current speed, in meters -/
def h : ℝ := 62

/-- The speed of the river current at distance h from the shore -/
def U_v : ℝ := 3.44

theorem river_current_speed :
  ∀ (x y : ℝ),
  (y = 0) →  -- Gavrila's path is along y = 0
  (x^2 + L^2 = (x + L)^2) →  -- Sound reaches Gavrila simultaneously
  (y * U_v / (2 * L) = U_b) →  -- Velocity relationship
  U_v = 3.44 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_speed_l1344_134458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_height_l1344_134499

/-- Given a building and a pole, prove the height of the building -/
theorem building_height
  (building_shadow : ℝ)
  (pole_height : ℝ)
  (pole_shadow : ℝ)
  (h1 : building_shadow = 20)
  (h2 : pole_height = 2)
  (h3 : pole_shadow = 3) :
  ∃ building_height : ℝ, 
    pole_height / pole_shadow = building_height / building_shadow ∧
    building_height = 40 / 3 :=
by
  -- Introduce the building_height as an existential variable
  use 40 / 3
  constructor
  · -- Prove the ratio equality
    rw [h1, h2, h3]
    norm_num
  · -- Prove the height equality
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_height_l1344_134499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1344_134457

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos (abs x)

-- State the theorem
theorem f_properties :
  (∀ x, f (x + 2 * Real.pi) = f x) ∧  -- Period is 2π
  (∀ x, f x ≥ -Real.sqrt 2) ∧         -- Minimum value is -√2
  (∃ x, f x = -Real.sqrt 2) :=        -- The minimum value is attained
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1344_134457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l1344_134445

theorem tv_price_change (P : ℝ) (h : P > 0) : 
  let price_after_decrease : ℝ := P * (1 - 0.2)
  let final_price : ℝ := price_after_decrease * (1 + 0.55)
  final_price = P * 1.24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l1344_134445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l1344_134464

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem log_inequality_range (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_incr : is_increasing_on f (Set.Ici 0))
  (h_zero : f (1/3) = 0) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = Set.union (Set.Ioo 0 (1/2)) (Set.Ioi 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l1344_134464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1344_134493

/-- The distance between the foci of a hyperbola xy = 4 is 4√2 -/
theorem hyperbola_foci_distance : 
  ∀ (f₁ f₂ : ℝ × ℝ), 
  (∀ (x y : ℝ), x * y = 4 → (x = f₁.1 ∧ y = f₁.2) ∨ (x = f₂.1 ∧ y = f₂.2)) →
  Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 4 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1344_134493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_triangle_tangent_ratio_theorem_l1344_134480

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  sum_angles : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

theorem triangle_ratio_theorem (t : Triangle) :
  (2 * Real.sin t.A * Real.cos t.C = Real.sin t.B) → t.a / t.c = 1 := by sorry

theorem triangle_tangent_ratio_theorem (t : Triangle) :
  (Real.sin (2 * t.A + t.B) = 3 * Real.sin t.B) →
  Real.tan t.A / Real.tan t.C = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_triangle_tangent_ratio_theorem_l1344_134480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1344_134469

variable (a b c d : ℝ)

noncomputable def f (x : ℝ) : ℝ := 
  (a + b * x) / (b + c * x) + 
  (b + c * x) / (c + d * x) + 
  (c + d * x) / (d + a * x) + 
  (d + a * x) / (a + b * x)

theorem f_max_value (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  ∀ x ≥ 0, f a b c d x ≤ a / b + b / c + c / d + d / a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1344_134469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_decomposition_l1344_134454

-- Define a permutation as a bijective function from ℕ to ℕ
def Permutation (n : ℕ) := {f : Fin n → Fin n // Function.Bijective f}

-- Define a transposition as a function that swaps two elements and leaves others unchanged
def Transposition (n : ℕ) (a b : Fin n) : Fin n → Fin n :=
  λ x => if x = a then b else if x = b then a else x

-- State the theorem
theorem permutation_decomposition (n : ℕ) (σ : Permutation n) :
  ∃ (k : ℕ) (t : Fin k → Fin n × Fin n), 
    σ.val = (List.foldl (λ f (pair : Fin n × Fin n) => 
      (Transposition n pair.1 pair.2) ∘ f) id (List.ofFn t)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_decomposition_l1344_134454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1344_134424

noncomputable def distancePointToLine (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

def circleCenter : ℝ × ℝ := (-1, 0)

def lineCoefficients : ℝ × ℝ × ℝ := (2, -1, 3)

theorem distance_circle_center_to_line :
  let (x₀, y₀) := circleCenter
  let (A, B, C) := lineCoefficients
  distancePointToLine x₀ y₀ A B C = Real.sqrt 5 / 5 := by
  sorry

#eval circleCenter
#eval lineCoefficients

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1344_134424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l1344_134459

/-- An ellipse C with equation x²/4 + y² = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- A line l with equation mx + ny = 1 -/
def line_l (m n x y : ℝ) : Prop := m*x + n*y = 1

/-- The unit circle with equation x² + y² = 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The area of a triangle given two side lengths and the angle between them -/
noncomputable def triangle_area (a b θ : ℝ) : ℝ := (1/2) * a * b * Real.sin θ

theorem ellipse_max_triangle_area :
  ∃ (m n : ℝ), 
    ellipse_C m n ∧ 
    (∃ (x₁ y₁ x₂ y₂ : ℝ), 
      x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
      line_l m n x₁ y₁ ∧ 
      line_l m n x₂ y₂ ∧ 
      unit_circle x₁ y₁ ∧ 
      unit_circle x₂ y₂ ∧
      triangle_area 1 1 (Real.pi/2) = (1/2) ∧
      m^2 = 4/3 ∧ 
      n^2 = 2/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l1344_134459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ages_l1344_134455

-- Define the ages as natural numbers
variable (Patrick Michael Monica : ℕ)

-- Define the ratios and age difference
axiom ratio_Patrick_Michael : 3 * Michael = 5 * Patrick
axiom ratio_Michael_Monica : 3 * Monica = 5 * Michael
axiom age_difference : Monica - Patrick = 64

-- Theorem to prove
theorem sum_of_ages : Patrick + Michael + Monica = 196 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ages_l1344_134455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l1344_134448

/-- A quadrilateral in a rectangular coordinate system -/
structure Quadrilateral where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ
  x3 : ℝ
  y3 : ℝ
  x4 : ℝ
  y4 : ℝ

/-- The area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ := sorry

/-- The theorem stating the relationship between the quadrilateral's coordinates and its area -/
theorem quadrilateral_area_theorem (q : Quadrilateral) (h1 : q.x1 = 4) (h2 : q.y1 = -3)
    (h3 : q.x2 = 4) (h4 : q.y2 = 7) (h5 : q.x3 = q.x4) (h6 : q.y3 = 2) (h7 : q.y4 = -7)
    (h8 : area q = 76) : ∃ (ε : ℝ), ε > 0 ∧ |q.x3 - 12.444| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l1344_134448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_parabola_l1344_134483

/-- The focus of the parabola y = 4x^2 -/
noncomputable def parabola_focus : ℝ × ℝ := (0, 1/16)

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 4 * x^2

/-- Theorem stating that parabola_focus is the focus of the parabola defined by parabola_equation -/
theorem focus_of_parabola :
  ∀ (x y : ℝ), parabola_equation x y →
  (x - parabola_focus.1)^2 + (y - parabola_focus.2)^2 = 
  (y + parabola_focus.2)^2 :=
by
  sorry

#check focus_of_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_parabola_l1344_134483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vectors_projection_b_minus_2a_onto_a_l1344_134461

noncomputable def a : Fin 2 → ℝ := ![1, 2]

def b_magnitude : ℝ := 1

noncomputable def angle_a_b : ℝ := Real.pi / 3 -- 60° in radians

theorem perpendicular_unit_vectors (v : Fin 2 → ℝ) :
  (v 0 * a 0 + v 1 * a 1 = 0) ∧ (v 0^2 + v 1^2 = 1) →
  (v = ![- 2 / 5 * Real.sqrt 5, 1 / 5 * Real.sqrt 5]) ∨
  (v = ![2 / 5 * Real.sqrt 5, - 1 / 5 * Real.sqrt 5]) :=
by sorry

theorem projection_b_minus_2a_onto_a :
  let a_magnitude : ℝ := Real.sqrt ((a 0)^2 + (a 1)^2)
  let a_dot_b : ℝ := a_magnitude * b_magnitude * Real.cos angle_a_b
  ((a_dot_b - 2 * ((a 0)^2 + (a 1)^2)) / a_magnitude) = - 19 / (2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vectors_projection_b_minus_2a_onto_a_l1344_134461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_l1344_134456

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the fixed points and radius
variable (P Q : V) (a : ℝ)

-- Define the distance between P and Q
def dist_PQ (P Q : V) : ℝ := ‖P - Q‖

-- Define the locus of centers
def locus (P Q : V) (a : ℝ) (O : V) : Prop :=
  ‖O - P‖ = a ∧ ‖O - Q‖ = a

-- Theorem statement
theorem locus_characterization (P Q : V) (a : ℝ) :
  (∃! O, locus P Q a O) ↔ dist_PQ P Q = 2 * a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_l1344_134456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_diagonal_length_l1344_134496

/-- A trapezoid with specific side lengths -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  side1 : ℝ
  side2 : ℝ
  parallel : EF ≥ GH
  EF_length : EF = 28
  GH_length : GH = 18
  side1_length : side1 = 15
  side2_length : side2 = 12

/-- The length of the shorter diagonal in the trapezoid -/
noncomputable def shorter_diagonal (t : Trapezoid) : ℝ := Real.sqrt (69639 / 400)

/-- Theorem stating that the calculated diagonal is indeed the shorter one -/
theorem shorter_diagonal_length (t : Trapezoid) :
  ∃ d : ℝ, d = shorter_diagonal t ∧ 
  d ≤ Real.sqrt ((t.EF - t.GH + d)^2 + t.side1^2) ∧
  d ≤ Real.sqrt ((t.EF - t.GH - d)^2 + t.side2^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_diagonal_length_l1344_134496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_rate_calculation_l1344_134413

/-- Calculates the second year's interest rate given initial investment, first year rate, and final amount -/
noncomputable def calculate_second_year_rate (initial_investment : ℝ) (first_year_rate : ℝ) (final_amount : ℝ) : ℝ :=
  let first_year_amount := initial_investment * (1 + first_year_rate)
  (final_amount / first_year_amount - 1) * 100

/-- Theorem stating that given the problem conditions, the second year's interest rate is approximately 4.17% -/
theorem second_year_rate_calculation :
  let initial_investment := (12000 : ℝ)
  let first_year_rate := (0.08 : ℝ)
  let final_amount := (13500 : ℝ)
  let calculated_rate := calculate_second_year_rate initial_investment first_year_rate final_amount
  abs (calculated_rate - 4.17) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_rate_calculation_l1344_134413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_OC_value_l1344_134414

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The origin point (0,0) -/
def O : Point := ⟨0, 0⟩

/-- Point A at (2,0) -/
def A : Point := ⟨2, 0⟩

/-- Predicate for points on the curve y = √(1 - x²) -/
def onCurve (p : Point) : Prop :=
  p.y = Real.sqrt (1 - p.x^2)

/-- Predicate for isosceles right triangle with A as right angle -/
def isIsoscelesRightTriangle (b c : Point) : Prop :=
  distance A b = distance A c ∧ 
  (b.x - 2) * (c.x - 2) + b.y * c.y = 0

/-- Theorem: Maximum value of |OC| is 2√2 + 1 -/
theorem max_OC_value :
  ∃ (b c : Point), onCurve b ∧ isIsoscelesRightTriangle b c ∧
  ∀ (b' c' : Point), onCurve b' ∧ isIsoscelesRightTriangle b' c' →
  distance O c ≤ 2 * Real.sqrt 2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_OC_value_l1344_134414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_of_container2_l1344_134408

-- Define the containers
structure Container where
  diameter : ℝ
  height : ℝ

-- Define the problem parameters
def container1 : Container := { diameter := 5, height := 10 }
def container2 : Container := { diameter := 10, height := 15 }
def price1 : ℝ := 2.5

-- Function to calculate the volume of a container
noncomputable def volume (c : Container) : ℝ := Real.pi * (c.diameter / 2) ^ 2 * c.height

-- Theorem stating the price of the second container
theorem price_of_container2 : 
  ∃ (price2 : ℝ), price2 = 15 ∧ 
  price2 / price1 = volume container2 / volume container1 :=
by
  -- Proof goes here
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_of_container2_l1344_134408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1344_134417

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  /-- The asymptote slope -/
  asymptote_slope : ℝ
  /-- A point on the hyperbola -/
  point_on_curve : ℝ × ℝ

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The equation of the hyperbola -/
def equation (h : Hyperbola) : ℝ → ℝ → Prop := sorry

theorem hyperbola_properties (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = Real.sqrt 2 / 2)
  (h_point : h.point_on_curve = (4, 2)) :
  eccentricity h = Real.sqrt 6 / 2 ∧ 
  ∀ x y, equation h x y ↔ x^2 / 8 - y^2 / 4 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1344_134417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_equivalence_l1344_134402

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Angle where
  value : ℝ

-- Define the given points and angles
variable (A B C P A' B' C' P' Q' : Point)
variable (α β : Angle)

-- Define the conditions from the problem
axiom terrain_points : A ≠ B ∧ B ≠ C ∧ C ≠ A
axiom map_points : A' ≠ B' ∧ B' ≠ C' ∧ C' ≠ A'

-- Define a function for angle between points
def angle_between (p1 p2 p3 : Point) : Angle :=
  ⟨0⟩  -- Placeholder implementation

axiom angle_APC : angle_between A P C = α
axiom angle_BPC : angle_between B P C = β

-- Define the arc intersection method
noncomputable def arc_intersection (A' B' C' : Point) (α β : Angle) : Point :=
  sorry

-- Define the alternative construction method
noncomputable def alternative_construction (A' B' C' : Point) (α β : Angle) : Point :=
  sorry

-- State the theorem to be proved
theorem construction_equivalence :
  ∀ (A' B' C' : Point) (α β : Angle),
    arc_intersection A' B' C' α β = alternative_construction A' B' C' α β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_equivalence_l1344_134402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1344_134494

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 3*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 2*x + 3

-- Theorem statement
theorem tangent_line_slope :
  let curve : Set (ℝ × ℝ) := {p | ∃ t, p.1 = t ∧ p.2 = f t}
  let p : ℝ × ℝ := (1, 3)
  p ∈ curve →
  (HasDerivAt f (f' 1) 1) →
  f' 1 = 5 := by
  intros
  simp [f']
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1344_134494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1344_134418

-- Define the equation
def equation (x a : ℝ) : Prop := x * (abs (x - a)) = a

-- Define the number of solutions
noncomputable def num_solutions (a : ℝ) : ℕ :=
  if -4 < a ∧ a < 4 then 1
  else if a = -4 ∨ a = 4 then 2
  else 3

-- State the theorem
theorem equation_solutions :
  ∀ a : ℝ, (∃ x : ℝ, equation x a) ∧
    (∀ x y z : ℝ, equation x a ∧ equation y a ∧ equation z a →
      (x = y ∨ x = z ∨ y = z ∨ num_solutions a = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1344_134418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l1344_134482

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Define for 0 to cover all natural numbers
  | n+1 => (1 + sequence_a n) / (1 - sequence_a n)

theorem a_2016_value : sequence_a 2016 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l1344_134482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_YZX_measure_l1344_134431

open Real

-- Define the circle Γ
noncomputable def Γ : EuclideanSpace ℝ (Fin 2) → Prop := sorry

-- Define the triangles
noncomputable def triangle_ABC : Set (EuclideanSpace ℝ (Fin 2)) := sorry
noncomputable def triangle_XYZ : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the points
noncomputable def A : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def B : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def C : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def X : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def Y : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def Z : EuclideanSpace ℝ (Fin 2) := sorry

-- Define necessary predicates
def IsIncircle (c : EuclideanSpace ℝ (Fin 2) → Prop) (t : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry
def IsCircumcircle (c : EuclideanSpace ℝ (Fin 2) → Prop) (t : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry
def OnSegment (p q r : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def MeasureAngle (p q r : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- State the theorem
theorem angle_YZX_measure :
  -- Γ is the incircle of ABC
  IsIncircle Γ triangle_ABC →
  -- Γ is the circumcircle of XYZ
  IsCircumcircle Γ triangle_XYZ →
  -- X is on BC
  OnSegment X B C →
  -- Y is on AB
  OnSegment Y A B →
  -- Z is on AC
  OnSegment Z A C →
  -- Angle A measures 50°
  MeasureAngle B A C = 50 →
  -- Angle B measures 70°
  MeasureAngle A B C = 70 →
  -- Then angle YZX measures 60°
  MeasureAngle Y Z X = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_YZX_measure_l1344_134431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1344_134465

noncomputable def f (x : ℝ) : ℝ := (2 * Real.sin (x + Real.pi / 3) + Real.sin x) * Real.cos x - Real.sqrt 3 * Real.sin x ^ 2

theorem f_properties :
  -- 1. The smallest positive period of f is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- 2. Range of m for which mf(x₀) - 2 = 0, where x₀ ∈ [0, 5π/12]
  (∀ (m : ℝ), (∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀ ≤ 5*Real.pi/12 ∧ m * f x₀ - 2 = 0) ↔ 
    (m ≤ -2 ∨ m ≥ 1)) ∧
  -- 3. Range of f(C/2 - π/6) / f(B/2 - π/6) in acute triangle ABC where ∠B = 2∠A
  (∀ (A B C : ℝ), 0 < A ∧ A < Real.pi/2 ∧ B = 2*A ∧ C = Real.pi - 3*A →
    Real.sqrt 2 / 2 < f (C/2 - Real.pi/6) / f (B/2 - Real.pi/6) ∧ 
    f (C/2 - Real.pi/6) / f (B/2 - Real.pi/6) < 2 * Real.sqrt 3 / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1344_134465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1344_134436

noncomputable def f (x : ℝ) := Real.log (2 - x)

theorem domain_of_f :
  {x : ℝ | f x ≠ 0 ∨ f x = 0} = Set.Iio 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1344_134436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l1344_134410

theorem cube_surface_area_increase : 
  ∀ (s : ℝ), s > 0 → 
  (6 * (1.3 * s)^2 - 6 * s^2) / (6 * s^2) = 0.69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l1344_134410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_width_calculation_l1344_134479

/-- The width of a wall built with bricks -/
noncomputable def wall_width (brick_length brick_width brick_height : ℝ)
               (wall_length wall_height : ℝ)
               (num_bricks : ℕ) : ℝ :=
  (brick_length * brick_width * brick_height * (num_bricks : ℝ)) / (wall_length * wall_height)

/-- Theorem stating the width of the wall given specific dimensions and number of bricks -/
theorem wall_width_calculation :
  wall_width 25 11.25 6 850 600 6800 = 22.5 := by
  -- Expand the definition of wall_width
  unfold wall_width
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_width_calculation_l1344_134479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_45_deg_l1344_134488

open Real

-- Define the rotation matrix S
noncomputable def S : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos (π/4), -sin (π/4)],
    ![sin (π/4),  cos (π/4)]]

-- State the theorem
theorem det_rotation_45_deg :
  Matrix.det S = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_45_deg_l1344_134488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_pyramid_volume_l1344_134433

/-- A pyramid with an isosceles right triangle base -/
structure IsoscelesRightPyramid where
  leg : ℝ
  height : ℝ

/-- The volume of an isosceles right pyramid -/
noncomputable def volume (p : IsoscelesRightPyramid) : ℝ :=
  (1/3) * (1/2 * p.leg * p.leg) * p.height

/-- Theorem: The volume of a pyramid with base leg 3 and height 4 is 6 -/
theorem isosceles_right_pyramid_volume :
  ∃ (p : IsoscelesRightPyramid), p.leg = 3 ∧ p.height = 4 ∧ volume p = 6 := by
  -- Construct the pyramid
  let p : IsoscelesRightPyramid := ⟨3, 4⟩
  -- Show that it satisfies the conditions
  have h1 : p.leg = 3 := rfl
  have h2 : p.height = 4 := rfl
  -- Calculate the volume
  have h3 : volume p = 6 := by
    unfold volume
    simp [h1, h2]
    norm_num
  -- Prove the existence
  exact ⟨p, h1, h2, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_pyramid_volume_l1344_134433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moles_cl2_required_l1344_134451

/-- Represents the number of moles of a substance -/
def Moles := ℝ

/-- Represents the chemical reaction CH4 + Cl2 → CH3Cl + HCl -/
structure Reaction where
  ch4 : Moles
  cl2 : Moles
  ch3cl : Moles
  hcl : Moles

/-- The reaction is balanced when the moles of reactants and products are correct -/
def is_balanced (r : Reaction) : Prop :=
  r.ch4 = r.cl2 ∧ r.ch3cl = r.hcl ∧ r.ch4 = r.ch3cl

/-- The theorem stating the number of moles of Cl2 required -/
theorem moles_cl2_required (r : Reaction) 
  (h1 : r.ch3cl = (1 : ℝ)) 
  (h2 : r.hcl = (1 : ℝ)) 
  (h3 : is_balanced r) : 
  r.cl2 = (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moles_cl2_required_l1344_134451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_in_65_69_interval_l1344_134450

/-- Represents a score interval with its lower bound and count of students -/
structure ScoreInterval where
  lower_bound : Nat
  count : Nat
deriving Repr

/-- Finds the interval containing the median score -/
def find_median_interval (intervals : List ScoreInterval) (total_students : Nat) : Option ScoreInterval :=
  let median_position := (total_students + 1) / 2
  let rec find_interval (acc : Nat) (remaining : List ScoreInterval) : Option ScoreInterval :=
    match remaining with
    | [] => none
    | i :: is => 
      if acc + i.count ≥ median_position then some i
      else find_interval (acc + i.count) is
  find_interval 0 intervals

theorem median_in_65_69_interval : 
  let intervals := [
    ScoreInterval.mk 50 5,
    ScoreInterval.mk 55 7,
    ScoreInterval.mk 60 22,
    ScoreInterval.mk 65 19,
    ScoreInterval.mk 70 15,
    ScoreInterval.mk 75 10,
    ScoreInterval.mk 80 18,
    ScoreInterval.mk 85 5
  ]
  let total_students := 101
  find_median_interval intervals total_students = some (ScoreInterval.mk 65 19) := by
  sorry

#eval find_median_interval [
  ScoreInterval.mk 50 5,
  ScoreInterval.mk 55 7,
  ScoreInterval.mk 60 22,
  ScoreInterval.mk 65 19,
  ScoreInterval.mk 70 15,
  ScoreInterval.mk 75 10,
  ScoreInterval.mk 80 18,
  ScoreInterval.mk 85 5
] 101

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_in_65_69_interval_l1344_134450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identities_l1344_134460

/-- Given a triangle ABC with cos A = 4/5 and tan B = 2, this theorem proves
    the values of tan(2A) and tan(2A - 2B). -/
theorem triangle_trig_identities (A B C : ℝ) (a b c : ℝ) :
  Real.cos A = 4/5 →
  Real.tan B = 2 →
  Real.tan (2*A) = 24/7 ∧ Real.tan (2*A - 2*B) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identities_l1344_134460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_star_equation_l1344_134435

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt (a - b)

-- Theorem statement
theorem solve_star_equation (x : ℝ) (h : star x 24 = 7) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_star_equation_l1344_134435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_difference_l1344_134416

/-- Xanthia's reading speed in pages per hour -/
noncomputable def xanthia_speed : ℝ := 120

/-- Molly's reading speed in pages per hour -/
noncomputable def molly_speed : ℝ := 60

/-- Number of pages in the book -/
noncomputable def book_pages : ℝ := 360

/-- Time difference in minutes between Molly and Xanthia reading the book -/
noncomputable def time_difference : ℝ := (book_pages / molly_speed - book_pages / xanthia_speed) * 60

theorem reading_time_difference : time_difference = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_difference_l1344_134416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l1344_134490

theorem polynomial_equality : 
  5 * (X + 3) * (X + 7) * (X + 11) * (X + 13) - 4 * X^2 = 
  5 * X^4 + 180 * X^3 + 1431 * X^2 + 4900 * X + 5159 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l1344_134490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_15pi_over_2_l1344_134489

theorem tan_alpha_plus_15pi_over_2 (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α = 1 / 4) : 
  Real.tan (α + 15 * π / 2) = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_15pi_over_2_l1344_134489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1344_134470

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem f_properties :
  (∀ x > 0, DifferentiableAt ℝ f x) ∧
  f 1 = 1/2 ∧
  deriv f 1 = 0 ∧
  (∀ x, 0 < x → x < 1 → deriv f x < 0) ∧
  (∀ x, x > 1 → deriv f x > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1344_134470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_inequality_l1344_134422

open Real

/-- The function f(x) = a(x^2 - 1) - ln x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - 1) - log x

/-- The derivative of f(x) -/
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - 1 / x

theorem extreme_value_inequality {a : ℝ} (ha : 0 < a) (ha2 : a < 1/2) :
  ∃ x_0 : ℝ, x_0 > 1 ∧ f a x_0 = 0 ∧ f_prime a x_0 < 1 - 2 * a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_inequality_l1344_134422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_numbers_sum_less_than_75_l1344_134401

theorem ten_numbers_sum_less_than_75 :
  ∃ (S : Finset ℕ),
    Finset.card S = 10 ∧
    (∀ x y, x ∈ S → y ∈ S → x ≠ y) ∧
    (∃ A : Finset ℕ, A ⊆ S ∧ Finset.card A = 3 ∧ ∀ x ∈ A, x % 5 = 0) ∧
    (∃ B : Finset ℕ, B ⊆ S ∧ Finset.card B = 4 ∧ ∀ x ∈ B, x % 4 = 0) ∧
    Finset.sum S id < 75 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_numbers_sum_less_than_75_l1344_134401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_combinations_l1344_134420

theorem sum_of_combinations (n : ℕ) : 
  (Finset.range (n + 1)).sum (λ k => Nat.choose (4*n + 1) (4*k + 1)) = 2^(4*n - 1) - 2^(2*n - 1) :=
by
  have h1 : Nat.choose 5 1 + Nat.choose 5 5 = 2^3 - 2 := by sorry
  have h2 : Nat.choose 9 1 + Nat.choose 9 5 + Nat.choose 9 9 = 2^7 - 2^3 := by sorry
  have h3 : Nat.choose 13 1 + Nat.choose 13 5 + Nat.choose 13 9 + Nat.choose 13 13 = 2^11 - 2^5 := by sorry
  have h4 : Nat.choose 17 1 + Nat.choose 17 5 + Nat.choose 17 9 + Nat.choose 17 13 + Nat.choose 17 17 = 2^15 - 2^7 := by sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_combinations_l1344_134420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_five_l1344_134404

theorem divisors_multiple_of_five (n : ℕ) (h : n = 3960) :
  (Finset.filter (λ d ↦ d ∣ n ∧ 5 ∣ d) (Finset.range (n + 1))).card = 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_five_l1344_134404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1344_134487

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given conditions
def satisfies_conditions (t : Triangle) : Prop :=
  Real.sin t.A = t.a * Real.cos t.C ∧
  t.c = Real.sqrt 3

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_conditions t) : 
  t.C = Real.pi / 3 ∧ 
  3/2 < t.a * Real.sin t.A + t.b * Real.sin t.B ∧ 
  t.a * Real.sin t.A + t.b * Real.sin t.B ≤ 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1344_134487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_theorem_l1344_134453

def vector_angle_problem (a b : ℝ × ℝ) : Prop :=
  let norm_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let norm_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  let norm_a_plus_b := Real.sqrt (((a.1 + b.1) ^ 2) + ((a.2 + b.2) ^ 2))
  norm_a = 1 ∧
  norm_a_plus_b = Real.sqrt 7 ∧
  b = (Real.sqrt 3, -1) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (norm_a * norm_b)) = Real.pi / 3

theorem vector_angle_theorem :
  ∃ a : ℝ × ℝ, vector_angle_problem a (Real.sqrt 3, -1) := by
  sorry

#check vector_angle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_theorem_l1344_134453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_width_theorem_l1344_134447

-- Define the domain
def D : Set ℝ := {x | x ≥ 1}

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := 1 / x
noncomputable def f₂ (x : ℝ) : ℝ := Real.sin x
noncomputable def f₃ (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)
def f₄ (x : ℝ) : ℝ := x^3 + 1

-- Define what it means for a function to have a channel width of d
def has_channel_width (f : ℝ → ℝ) (d : ℝ) (domain : Set ℝ) : Prop :=
  ∃ (k m₁ m₂ : ℝ), ∀ x ∈ domain,
    k * x + m₁ ≤ f x ∧ f x ≤ k * x + m₂ ∧ m₂ - m₁ = d

-- State the theorem
theorem channel_width_theorem :
  (has_channel_width f₁ 1 D ∧ has_channel_width f₃ 1 D) ∧
  (¬has_channel_width f₂ 1 D ∧ ¬has_channel_width f₄ 1 D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_width_theorem_l1344_134447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_completion_days_l1344_134477

/-- Represents the number of days required for Arun to complete the remaining work alone -/
noncomputable def daysForArunToComplete (totalDays : ℝ) (arunDays : ℝ) (workingDays : ℝ) : ℝ :=
  let totalWork := 1
  let combinedRate := totalWork / totalDays
  let arunRate := totalWork / arunDays
  let workDone := workingDays * combinedRate
  let remainingWork := totalWork - workDone
  remainingWork / arunRate

/-- Theorem stating that Arun will require 36 days to complete the remaining work alone -/
theorem arun_completion_days :
  daysForArunToComplete 10 60 4 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_completion_days_l1344_134477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_diagonal_length_l1344_134407

-- Define the trapezoid WXYZ
structure Trapezoid :=
  (W X Y Z : ℝ × ℝ)
  (parallel_WZ_XY : (Z.2 - W.2) / (Z.1 - W.1) = (Y.2 - X.2) / (Y.1 - X.1))
  (WX_eq_YZ : dist W X = dist Y Z)
  (WX_length : dist W X = 24)
  (WY_perp_XY : (Y.2 - W.2) * (Y.1 - X.1) + (Y.1 - W.1) * (X.2 - Y.2) = 0)

-- Define the intersection point O and midpoint Q
noncomputable def O (t : Trapezoid) : ℝ × ℝ := sorry
noncomputable def Q (t : Trapezoid) : ℝ × ℝ := ((t.X.1 + t.Y.1) / 2, (t.X.2 + t.Y.2) / 2)

-- State the theorem
theorem trapezoid_diagonal_length (t : Trapezoid) 
  (h_OQ : dist (O t) (Q t) = 9) :
  dist t.W t.Y = 6 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_diagonal_length_l1344_134407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1344_134419

theorem problem_solution (a b : ℝ) (h : Set.toFinset {a, b/a, 1} = Set.toFinset {a^2, a+b, 0}) :
  a^2003 + b^2004 = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1344_134419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_second_quadrant_l1344_134421

/-- An angle in the second quadrant -/
def SecondQuadrantAngle (α : ℝ) : Prop :=
  Real.pi / 2 < α ∧ α < Real.pi

/-- A point on the terminal side of an angle -/
def TerminalSidePoint (α : ℝ) (x y : ℝ) : Prop :=
  x = Real.cos α ∧ y = Real.sin α

theorem sin_value_second_quadrant (α x : ℝ) 
  (h1 : SecondQuadrantAngle α)
  (h2 : TerminalSidePoint α x (Real.sqrt 5))
  (h3 : Real.cos α = (Real.sqrt 2 / 4) * x) :
  Real.sin α = Real.sqrt 10 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_second_quadrant_l1344_134421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1344_134471

theorem divisibility_condition (n : ℕ) : 
  (2^n + n) ∣ (8^n + n) ↔ n ∈ ({1, 2, 4, 6} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1344_134471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_pattern_area_l1344_134423

/-- The area of shaded region formed by semicircles in a pattern --/
theorem semicircle_pattern_area (diameter : ℝ) (pattern_length : ℝ) : 
  diameter = 3 →
  pattern_length = 18 →
  let radius := diameter / 2
  let num_semicircles := pattern_length / diameter
  let semicircle_area := π * radius^2 / 2
  let total_area := num_semicircles * 2 * semicircle_area
  total_area = 27 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_pattern_area_l1344_134423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1344_134462

theorem power_equation_solution (P : ℝ) : 
  Real.sqrt (P^3) = 9 * (9^(1/9)) → P = 3^(14/9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1344_134462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_proof_l1344_134498

/-- Given a triangle ABC and a point M, prove that AM = (1/3)b + (2/3)c -/
theorem triangle_vector_proof (A B C M : EuclideanSpace ℝ (Fin 2)) (b c : EuclideanSpace ℝ (Fin 2)) : 
  (B - A = c) → 
  (C - A = b) → 
  (C - M = 2 • (M - B)) → 
  (M - A = (1/3) • b + (2/3) • c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_proof_l1344_134498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_encryption_decryption_theorem_l1344_134409

-- Define the encryption function for a single digit
def encrypt_digit (d : Nat) : Nat :=
  10 - ((7 * d) % 10)

-- Define the decryption function for a single digit
def decrypt_digit (d : Nat) : Nat :=
  match d with
  | 0 => 0
  | 1 => 7
  | 2 => 4
  | 3 => 1
  | 4 => 8
  | 5 => 5
  | 6 => 2
  | 7 => 9
  | 8 => 6
  | 9 => 3
  | _ => d

-- Define the encryption function for a number
def encrypt (n : Nat) : Nat :=
  let digits := Nat.digits 10 n
  let encrypted_digits := List.map encrypt_digit digits
  List.foldl (fun acc d => acc * 10 + d) 0 encrypted_digits

-- Define the decryption function for a number
def decrypt (n : Nat) : Nat :=
  let digits := Nat.digits 10 n
  let decrypted_digits := List.map decrypt_digit digits
  List.foldl (fun acc d => acc * 10 + d) 0 decrypted_digits

-- Theorem statement
theorem encryption_decryption_theorem :
  decrypt 473392 = 891134 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_encryption_decryption_theorem_l1344_134409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_term_position_l1344_134485

-- Define the sequence
noncomputable def a (n : ℕ) : ℝ := Real.sqrt (3 * n - 1)

-- State the theorem
theorem term_position :
  ∃ n : ℕ, a n = 2 * Real.sqrt 17 ∧ n = 23 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_term_position_l1344_134485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_points_properties_l1344_134439

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) * Real.sin x

-- Define the sequence of extremum points
noncomputable def extremum_points (a : ℝ) : ℕ → ℝ := 
  fun n => n * Real.pi - Real.arctan (1 / a)

-- Statement of the theorem
theorem extremum_points_properties (a : ℝ) (h : a > 0) :
  -- Part I: The sequence {f(x_n)} is a geometric sequence
  (∃ r : ℝ, ∀ n : ℕ, f a (extremum_points a (n + 1)) = r * f a (extremum_points a n)) ∧
  -- Part II: If a ≥ 1/√(e^2 - 1), then for all n ∈ ℕ*, x_n < |f(x_n)|
  (a ≥ 1 / Real.sqrt (Real.exp 2 - 1) →
    ∀ n : ℕ, n > 0 → extremum_points a n < |f a (extremum_points a n)|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_points_properties_l1344_134439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circles_theorem_l1344_134425

-- Define the points
variable (A₁ A₂ B₁ B₂ C₁ C₂ : EuclideanSpace ℝ (Fin 2))

-- Define the circles (we represent them by their centers and radii)
variable (circle1 circle2 circle3 : EuclideanSpace ℝ (Fin 2) × ℝ)

-- Define the condition that the circles intersect pairwise at the given points
def circles_intersect (c1 c2 : EuclideanSpace ℝ (Fin 2) × ℝ) (P Q : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (x : EuclideanSpace ℝ (Fin 2)), x ∈ Metric.sphere c1.1 c1.2 ∧ x ∈ Metric.sphere c2.1 c2.2 ∧
  (x = P ∨ x = Q)

-- Define the length of a line segment
noncomputable def length (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  ‖P - Q‖

-- State the theorem
theorem three_circles_theorem
  (h1 : circles_intersect circle1 circle2 A₁ A₂)
  (h2 : circles_intersect circle2 circle3 B₁ B₂)
  (h3 : circles_intersect circle3 circle1 C₁ C₂) :
  length A₁ B₂ * length B₁ C₂ * length C₁ A₂ =
  length A₂ B₁ * length B₂ C₁ * length C₂ A₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circles_theorem_l1344_134425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_by_force_l1344_134437

/-- The force function F(x) in Newtons, where x is in meters -/
def F (x : ℝ) : ℝ := 5 * x + 2

/-- The work done by a force F over a displacement from a to b -/
noncomputable def work (F : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, F x

/-- Theorem stating that the work done by the force F from x=0 to x=4 is 48 Joules -/
theorem work_done_by_force : work F 0 4 = 48 := by
  -- Unfold the definition of work
  unfold work
  -- Evaluate the integral
  simp [F]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_by_force_l1344_134437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_composite_bound_l1344_134427

/-- Sum of digits in base k -/
def S (k : ℕ) (n : ℕ) : ℕ := sorry

/-- The set of composite numbers -/
def Composite : Set ℕ := {n : ℕ | n > 1 ∧ ¬Prime n}

theorem digit_sum_composite_bound :
  ∃ (C : Finset ℕ), C.toSet ⊆ Composite ∧ C.card ≤ 2 ∧
  ∀ p : ℕ, p < 20000 → Prime p → S 31 p ∈ C ∨ S 31 p ∉ Composite := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_composite_bound_l1344_134427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1344_134486

theorem problem_statement (a b : ℝ) (ha : a > 0) (heq : a + Real.exp b = 2) :
  (a + b ≤ 1) ∧ (Real.log a + Real.exp b ≤ 1) ∧ (Real.log a - abs b ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1344_134486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_eight_six_thousand_scientific_notation_l1344_134411

-- Define scientific notation
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

-- Define the property of scientific notation
def is_scientific_notation (x : ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℤ), x = scientific_notation a n ∧ 1 ≤ |a| ∧ |a| < 10

-- Theorem: 686,000 in scientific notation is 6.86 × 10^5
theorem six_eight_six_thousand_scientific_notation :
  is_scientific_notation 686000 ∧ 
  686000 = scientific_notation 6.86 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_eight_six_thousand_scientific_notation_l1344_134411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l1344_134441

theorem largest_expression : 
  let a := (35 : ℝ)^(1/6)
  let b := (7 * 5^(1/4))^(1/2)
  let c := (5 * 7^(1/4))^(1/2)
  let d := (7 * 5^(1/2))^(1/4)
  let e := (5 * 7^(1/2))^(1/4)
  b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e := by
  sorry

#check largest_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l1344_134441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l1344_134428

-- Define the circles and their properties
def circle_P : ℝ → Prop := λ r ↦ r = 3
def circle_Q : ℝ → ℝ → Prop := λ r_Q r_P ↦ r_Q < r_P
def circle_R : ℝ → ℝ → ℝ → Prop := λ r_R r_Q r_P ↦ r_R < r_P ∧ r_R < r_Q

-- Define the tangency conditions
def internally_tangent : ℝ → ℝ → Prop := λ _ _ ↦ True
def externally_tangent : ℝ → ℝ → Prop := λ _ _ ↦ True
def tangent_to_diameter : ℝ → Prop := λ _ ↦ True

-- Define the relationship between radii of Q and R
def radius_relationship : ℝ → ℝ → Prop := λ r_Q r_R ↦ r_Q = 2 * r_R

-- Define the form of radius of Q
noncomputable def radius_Q_form : ℝ → ℕ → ℕ → Prop := λ r_Q p q ↦ r_Q = Real.sqrt (p : ℝ) - q

theorem circle_tangency_theorem (r_P r_Q r_R : ℝ) (p q : ℕ) :
  circle_P r_P →
  circle_Q r_Q r_P →
  circle_R r_R r_Q r_P →
  internally_tangent r_Q r_P →
  externally_tangent r_R r_P →
  externally_tangent r_R r_Q →
  tangent_to_diameter r_R →
  radius_relationship r_Q r_R →
  radius_Q_form r_Q p q →
  p + q = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l1344_134428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_exists_l1344_134472

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A direction vector in 2D space -/
structure Direction where
  dx : ℝ
  dy : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- A rectangle defined by four points -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Check if a point lies on a line defined by two other points -/
def pointOnLine (p : Point) (a : Point) (b : Point) : Prop :=
  (p.y - a.y) * (b.x - a.x) = (b.y - a.y) * (p.x - a.x)

/-- Check if two line segments are parallel -/
def areParallel (p1 : Point) (p2 : Point) (q1 : Point) (q2 : Point) : Prop :=
  (p2.y - p1.y) * (q2.x - q1.x) = (q2.y - q1.y) * (p2.x - p1.x)

/-- Check if a line segment is parallel to a direction -/
def isParallelToDirection (p1 : Point) (p2 : Point) (dir : Direction) : Prop :=
  (p2.y - p1.y) * dir.dx = dir.dy * (p2.x - p1.x)

/-- Main theorem: There exists an inscribed rectangle with specified directions -/
theorem inscribed_rectangle_exists (quad : Quadrilateral) (dir1 dir2 : Direction) :
  ∃ (rect : Rectangle),
    pointOnLine rect.P quad.A quad.B ∧
    pointOnLine rect.R quad.C quad.D ∧
    pointOnLine rect.Q quad.B quad.C ∧
    areParallel rect.P rect.Q rect.R rect.S ∧
    areParallel rect.Q rect.R rect.S rect.P ∧
    isParallelToDirection rect.P rect.S dir1 ∧
    isParallelToDirection rect.Q rect.R dir2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_exists_l1344_134472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_teaches_latin_in_y_l1344_134426

-- Define the types for teachers, schools, and subjects
inductive Teacher : Type
| A | B | C

inductive School : Type
| X | Y | Z

inductive Subject : Type
| Mathematics | Latin | Music

-- Define the assignment function
def assignment : Teacher → School × Subject := sorry

-- State the conditions
axiom condition1 : ∀ s : School, assignment Teacher.A ≠ (s, Subject.Mathematics)

axiom condition2 : ∀ s : Subject, assignment Teacher.B ≠ (School.Z, s)

axiom condition3 : ∃ t : Teacher, assignment t = (School.Z, Subject.Music)

axiom condition4 : ∀ t : Teacher, (assignment t).1 = School.X → 
                   (assignment t).2 ≠ Subject.Latin

axiom condition5 : ∀ s : School, assignment Teacher.B ≠ (s, Subject.Mathematics)

-- State the theorem to be proved
theorem b_teaches_latin_in_y : 
  assignment Teacher.B = (School.Y, Subject.Latin) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_teaches_latin_in_y_l1344_134426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_for_f_less_than_five_l1344_134495

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 3

theorem necessary_condition_for_f_less_than_five :
  ∀ x : ℝ, f x < 5 → x > -4 ∧ ∃ y : ℝ, y > -4 ∧ f y ≥ 5 :=
by
  intro x h
  have h1 : x > -3 := by
    -- Proof steps here
    sorry
  have h2 : x > -4 := by
    -- Proof steps here
    sorry
  have h3 : ∃ y : ℝ, y > -4 ∧ f y ≥ 5 := by
    -- Proof steps here
    sorry
  exact ⟨h2, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_for_f_less_than_five_l1344_134495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_greater_than_07_l1344_134432

noncomputable def numbers : List ℚ := [8/10, 1/2, 9/10, 1/3]

def sum_greater_than_threshold (lst : List ℚ) (threshold : ℚ) : ℚ :=
  lst.filter (λ x => x > threshold) |>.sum

theorem sum_greater_than_07 : sum_greater_than_threshold numbers (7/10) = 17/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_greater_than_07_l1344_134432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_shortest_distance_l1344_134467

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

/-- Line intersecting the parabola -/
noncomputable def intersecting_line (C : Parabola) (x : ℝ) : ℝ :=
  2*Real.sqrt 2*(x - C.p/2)

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Distance from a point to a line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y + 3| / Real.sqrt 2

theorem parabola_and_shortest_distance 
  (C : Parabola) 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : C.eq x₁ y₁) 
  (h₂ : C.eq x₂ y₂) 
  (h₃ : y₁ = intersecting_line C x₁) 
  (h₄ : y₂ = intersecting_line C x₂) 
  (h₅ : distance x₁ y₁ x₂ y₂ = 9/2) :
  (C.p = 2 ∧ C.eq = λ x y => y^2 = 4*x) ∧
  (∀ x y, C.eq x y → distance_to_line x y ≥ distance_to_line 1 2) ∧
  C.eq 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_shortest_distance_l1344_134467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distances_midpoint_tangent_l1344_134492

-- Define the hyperbola
def Hyperbola : Set (ℝ × ℝ) := {p | p.1 * p.2 = 1}

-- Define points A and B on the hyperbola
variable (A B : ℝ × ℝ)
variable (hA : A ∈ Hyperbola)
variable (hB : B ∈ Hyperbola)

-- Define line AB
def LineAB (A B : ℝ × ℝ) : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (1 - t) • A + t • B}

-- Define asymptotes of the hyperbola
def Asymptotes : Set (ℝ × ℝ) := {p | p.1 = 0 ∨ p.2 = 0}

-- Define intersection points A₁ and B₁
variable (A₁ B₁ : ℝ × ℝ)
variable (hA₁ : A₁ ∈ LineAB A B ∩ Asymptotes)
variable (hB₁ : B₁ ∈ LineAB A B ∩ Asymptotes)

-- Define point X
variable (X : ℝ × ℝ)
variable (hX : X ∈ Hyperbola)

-- Define line A₁B₁
def LineA₁B₁ (A₁ B₁ : ℝ × ℝ) : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (1 - t) • A₁ + t • B₁}

-- Define tangent line condition
def IsTangent (l : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ l ∧ ∀ q ∈ l, q ≠ p → q ∉ Hyperbola

-- Theorem statements
theorem equal_distances :
  dist A A₁ = dist B B₁ ∧ dist A B₁ = dist B A₁ :=
sorry

theorem midpoint_tangent (h : IsTangent (LineA₁B₁ A₁ B₁) X) :
  X = (A₁ + B₁) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distances_midpoint_tangent_l1344_134492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_FM_FN_A_Q_N_collinear_l1344_134434

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define points A, B, F
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)
def F : ℝ × ℝ := (1, 0)

-- Define point P on the ellipse
noncomputable def P (x₀ y₀ : ℝ) : Prop := 
  ellipse x₀ y₀ ∧ (x₀, y₀) ≠ A ∧ (x₀, y₀) ≠ B

-- Define points M and N
noncomputable def M (x₀ y₀ : ℝ) : ℝ × ℝ := (3, 5 * y₀ / (x₀ + 2))
noncomputable def N (x₀ y₀ : ℝ) : ℝ × ℝ := (3, y₀ / (x₀ - 2))

-- Define point Q
noncomputable def Q : ℝ × ℝ := sorry

-- Theorem statements
theorem dot_product_FM_FN (x₀ y₀ : ℝ) (h : P x₀ y₀) :
  (M x₀ y₀ - F) • (N x₀ y₀ - F) = 1/4 := by
  sorry

theorem A_Q_N_collinear (x₀ y₀ : ℝ) (h : P x₀ y₀) :
  ∃ (t : ℝ), Q = t • A + (1 - t) • N x₀ y₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_FM_FN_A_Q_N_collinear_l1344_134434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_wallet_problem_l1344_134463

theorem anna_wallet_problem (total_bills : ℕ) (total_value : ℕ) 
  (small_bills : ℕ) (ten_bills : ℕ) (small_value : ℕ) :
  total_bills = 12 →
  total_value = 100 →
  small_bills = 4 →
  ten_bills = 8 →
  total_bills = small_bills + ten_bills →
  total_value = small_bills * small_value + ten_bills * 10 →
  small_value = 5 := by
  intro h1 h2 h3 h4 h5 h6
  -- The proof steps would go here
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_wallet_problem_l1344_134463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_CD_l1344_134444

/-- Given a line and a circle with specific properties, prove the distance between
    certain points on the x-axis. -/
theorem distance_CD (m : ℝ) (A B C D : ℝ × ℝ) : 
  (∀ x y, m * x + y + 3 * m - Real.sqrt 3 = 0 → x^2 + y^2 = 12) →  -- Line l intersects circle
  A ∈ {p : ℝ × ℝ | ∃ x, p = (x, -(m * x + 3 * m - Real.sqrt 3))} →  -- A is on line l
  B ∈ {p : ℝ × ℝ | ∃ x, p = (x, -(m * x + 3 * m - Real.sqrt 3))} →  -- B is on line l
  A.1^2 + A.2^2 = 12 →  -- A is on the circle
  B.1^2 + B.2^2 = 12 →  -- B is on the circle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 →  -- |AB| = 2√3
  C.2 = 0 →  -- C is on x-axis
  D.2 = 0 →  -- D is on x-axis
  m * (C.1 - A.1) = -(C.2 - A.2) →  -- AC perpendicular to l
  m * (D.1 - B.1) = -(D.2 - B.2) →  -- BD perpendicular to l
  (C.1 - D.1)^2 = 16  -- |CD| = 4
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_CD_l1344_134444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_birds_in_tree_l1344_134400

-- Define the initial number of birds
def initial_sparrows : ℕ := 7
def initial_robins : ℕ := 5
def initial_blue_jays : ℕ := 2

-- Define the additional number of birds
def additional_sparrows : ℕ := 12
def additional_robins : ℕ := 4
def additional_blue_jays : ℕ := 5

-- Define the weight of hummingbirds in grams and their total weight in kilograms
def hummingbird_weight : ℕ := 60
def total_hummingbird_weight : ℚ := 0.3

-- Define the weight of cardinals in ounces and their total weight in ounces
def cardinal_weight : ℕ := 16  -- 1 pound = 16 ounces
def total_cardinal_weight : ℕ := 80

-- Theorem to prove
theorem total_birds_in_tree : 
  (initial_sparrows + additional_sparrows) +
  (initial_robins + additional_robins) +
  (initial_blue_jays + additional_blue_jays) +
  (Nat.floor (total_hummingbird_weight * 1000 / hummingbird_weight)) +
  (total_cardinal_weight / cardinal_weight) = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_birds_in_tree_l1344_134400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_allocation_l1344_134412

/-- Profit function for product A -/
noncomputable def profit_A (m : ℝ) : ℝ := (1/3) * m + 65

/-- Profit function for product B -/
noncomputable def profit_B (m : ℝ) : ℝ := 76 + 4 * Real.sqrt m

/-- Total profit function -/
noncomputable def total_profit (x : ℝ) : ℝ := profit_A (150 - x) + profit_B x

/-- The domain of the total profit function -/
def valid_investment (x : ℝ) : Prop := 25 ≤ x ∧ x ≤ 125

theorem max_profit_allocation :
  ∃ x : ℝ, valid_investment x ∧ 
    (∀ y : ℝ, valid_investment y → total_profit y ≤ total_profit x) ∧
    total_profit x = 203 ∧
    x = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_allocation_l1344_134412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_cookie_radius_l1344_134481

theorem scrap_cookie_radius (r_large r_small r_center : ℝ) (n_small : ℕ) :
  r_large = 5 →
  r_small = 1 →
  r_center = 2 →
  n_small = 10 →
  let a_large := π * r_large^2
  let a_small := n_small * π * r_small^2
  let a_center := π * r_center^2
  let a_scrap := a_large - (a_small + a_center)
  Real.sqrt (a_scrap / π) = Real.sqrt 11 := by
  sorry

#check scrap_cookie_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_cookie_radius_l1344_134481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a6_plus_b6_l1344_134446

-- Define the arithmetic sequence a_n
def a : ℕ → ℝ := sorry

-- Define the sum of first n terms of a_n
def S : ℕ → ℝ := sorry

-- Define the geometric sequence b_n
def b : ℕ → ℝ := sorry

-- State the theorem
theorem tan_a6_plus_b6 (h1 : ∀ n : ℕ, S (n + 1) - S n = a (n + 1))
                       (h2 : S 11 = 22 * Real.pi / 3)
                       (h3 : ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1)
                       (h4 : b 5 * b 7 = Real.pi^2 / 4) :
  Real.tan (a 6 + b 6) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a6_plus_b6_l1344_134446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1344_134415

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 1171) :
  Int.gcd (3 * b ^ 2 + 47 * b + 79) (b + 17) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1344_134415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_135_l1344_134475

/-- Given a triangle with sides a, b, c and area S, if S = 1/4 * (c^2 - a^2 - b^2), 
    then the angle C opposite to side c is 135°. -/
theorem triangle_angle_135 (a b c S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
    (h_area : S = (1/4) * (c^2 - a^2 - b^2)) : 
    Real.cos (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = 
    -Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) := by
  sorry

#check triangle_angle_135

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_135_l1344_134475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1344_134497

noncomputable def f (a x : ℝ) : ℝ := (x + 1) / x + Real.sin x - a^2

theorem sufficient_not_necessary_condition :
  (∀ a : ℝ, a = 1 → ∀ x : ℝ, x ≠ 0 → f a x = -f a (-x)) ∧
  ¬(∀ a : ℝ, (∀ x : ℝ, x ≠ 0 → f a x = -f a (-x)) → a = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1344_134497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_circle_l1344_134440

-- Define the line l: 3x - 4y - 3 = 0
def line_l (x y : ℝ) : Prop := 3 * x - 4 * y - 3 = 0

-- Define the circle C: (x+1)² + (y-1)² = 2
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 2

-- Define a point P on line l
def point_on_line (P : ℝ × ℝ) : Prop := line_l P.1 P.2

-- Define the distance from a point to the circle
noncomputable def distance_to_circle (P : ℝ × ℝ) : ℝ := 
  Real.sqrt ((P.1 + 1)^2 + (P.2 - 1)^2 - 2)

-- Theorem: The minimum distance from any point on line l to circle C is √2
theorem min_distance_to_circle : 
  ∃ (P : ℝ × ℝ), point_on_line P ∧ 
    (∀ (Q : ℝ × ℝ), point_on_line Q → distance_to_circle P ≤ distance_to_circle Q) ∧
    distance_to_circle P = Real.sqrt 2 := by
  sorry

#check min_distance_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_circle_l1344_134440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_solution_l1344_134468

noncomputable def vector_a : ℝ × ℝ × ℝ := (1, -2, 5)

def in_xy_plane (v : ℝ × ℝ × ℝ) : Prop :=
  v.2.2 = 0

def perpendicular (v w : ℝ × ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2.1 * w.2.1 + v.2.2 * w.2.2 = 0

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

theorem vector_b_solution :
  ∀ b : ℝ × ℝ × ℝ,
    in_xy_plane b ∧
    perpendicular vector_a b ∧
    magnitude b = 2 * Real.sqrt 5 →
    b = (4, 2, 0) ∨ b = (-4, -2, 0) :=
by sorry

#check vector_b_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_solution_l1344_134468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_equals_sin_l1344_134438

open Real

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => sin
  | (n + 1) => deriv (f n)

-- State the theorem
theorem f_2012_equals_sin : f 2012 = sin := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_equals_sin_l1344_134438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l1344_134429

/-- The differential equation y^(IV) - 4y''' + 8y'' - 8y' + 4y = 0 -/
def DiffEq (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv^[4] y) x - 4 * (deriv^[3] y) x + 8 * (deriv^[2] y) x - 8 * (deriv y) x + 4 * y x = 0

/-- The general solution to the differential equation -/
noncomputable def GeneralSolution (C₁ C₂ C₃ C₄ : ℝ) (x : ℝ) : ℝ :=
  Real.exp x * ((C₁ + C₃ * x) * Real.cos x + (C₂ + C₄ * x) * Real.sin x)

/-- Theorem stating that the general solution satisfies the differential equation -/
theorem general_solution_satisfies_diff_eq (C₁ C₂ C₃ C₄ : ℝ) :
  DiffEq (GeneralSolution C₁ C₂ C₃ C₄) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l1344_134429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_half_equals_neg_two_l1344_134443

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the specific function
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (4 : ℝ) ^ x else -((4 : ℝ) ^ (-x))

-- State the theorem
theorem f_neg_half_equals_neg_two (h : odd_function f) : f (-1/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_half_equals_neg_two_l1344_134443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_or_blue_l1344_134430

-- Define the total number of marbles
def total_marbles : ℕ := 90

-- Define the probability of drawing a white marble
def prob_white : ℚ := 1/6

-- Define the probability of drawing a green marble
def prob_green : ℚ := 1/5

-- Theorem statement
theorem prob_red_or_blue : 1 - (prob_white + prob_green) = 19/30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_or_blue_l1344_134430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_area_is_144pi_l1344_134474

-- Define the radii of the two circles
def r₁ : ℝ := 15
def r₂ : ℝ := 9

-- Define the area of a circle
noncomputable def circleArea (r : ℝ) : ℝ := Real.pi * r^2

-- Define the area of the ring
noncomputable def ringArea : ℝ := circleArea r₁ - circleArea r₂

-- Theorem statement
theorem ring_area_is_144pi : ringArea = 144 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_area_is_144pi_l1344_134474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_first_greater_second_l1344_134466

def fair_8_sided_die : Finset ℕ := Finset.range 8

def roll_pair : Finset (ℕ × ℕ) := fair_8_sided_die.product fair_8_sided_die

def first_greater (pair : ℕ × ℕ) : Prop := pair.fst > pair.snd

theorem probability_first_greater_second :
  (roll_pair.filter (fun p => p.fst > p.snd)).card / roll_pair.card = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_first_greater_second_l1344_134466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocals_l1344_134405

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Define the line l
def line_l (x y t : ℝ) : Prop := x = 1 + (1/2) * t ∧ y = -1 + (Real.sqrt 3 / 2) * t

-- Define the point P
def point_P : ℝ × ℝ := (1, -1)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    line_l A.1 A.2 t₁ ∧ curve_C A.1 A.2 ∧
    line_l B.1 B.2 t₂ ∧ curve_C B.1 B.2 ∧
    t₁ ≠ t₂

-- State the theorem
theorem intersection_sum_reciprocals 
  (A B : ℝ × ℝ) 
  (h_intersection : intersection_points A B) :
  1 / dist point_P A + 1 / dist point_P B = (Real.sqrt (16 + 2 * Real.sqrt 3)) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocals_l1344_134405
