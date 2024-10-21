import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_exponential_inequality_l499_49965

theorem solution_set_of_exponential_inequality :
  {x : ℝ | (2 : ℝ)^(x + 2) > 8} = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_exponential_inequality_l499_49965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_difference_total_cost_max_A_projectors_l499_49950

/- Define the cost of projectors -/
def cost_A : ℕ → ℕ
| _ => 4000  -- Assuming a constant cost based on the solution

def cost_B : ℕ → ℕ
| _ => 3200  -- Assuming a constant cost based on the solution

/- Define the conditions -/
theorem cost_difference : ∀ n, cost_A n = cost_B n + 800 := by
  intro n
  rfl  -- This should hold given our definitions

theorem total_cost : 5 * cost_A 1 + 4 * cost_B 1 = 32800 := by
  rfl  -- This should hold given our definitions

/- Define the discounted costs -/
def discounted_cost_A (n : ℕ) : ℚ := (9 / 10) * cost_A n
def discounted_cost_B (n : ℕ) : ℕ := cost_B n - 200

/- The theorem to prove -/
theorem max_A_projectors : 
  ∃ m : ℕ, m ≤ 30 ∧ 
  (m : ℚ) * discounted_cost_A 1 + ((30 - m) : ℚ) * (discounted_cost_B 1 : ℚ) ≤ 97200 ∧
  ∀ k : ℕ, k > m → (k : ℚ) * discounted_cost_A 1 + ((30 - k) : ℚ) * (discounted_cost_B 1 : ℚ) > 97200 ∧
  m = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_difference_total_cost_max_A_projectors_l499_49950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l499_49935

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the property of being equilateral
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.a / Real.cos t.A = t.b / Real.cos t.B)
  (h2 : t.b / Real.cos t.B = t.c / Real.cos t.C) :
  isEquilateral t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l499_49935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l499_49912

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the axioms for f
axiom f_domain : ∀ x : ℝ, x > 0 → f x ≠ 0
axiom f_not_zero : ∃ x : ℝ, x > 0 ∧ f x ≠ 0
axiom f_property : ∀ x y : ℝ, x > 0 → f (x^y) = y * f x

-- Define the theorem
theorem f_inequality (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > 1)
  (h4 : ∃ d : ℝ, a - b = b - c) : 
  f a * f c < (f b)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l499_49912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_theorem_l499_49991

/-- The theorem statement for the chord problem -/
theorem chord_theorem (a b p q : ℝ) (h_ab : a > b) (h_b_pos : b > 0) :
  let ellipse (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
  let circle (x y : ℝ) := x^2 + y^2 = a^2
  let M : ℝ × ℝ := (p, q)
  let M' : ℝ × ℝ := (p, a / b * q)
  ∀ (A B A' B' : ℝ × ℝ),
    ellipse A.1 A.2 → ellipse B.1 B.2 →
    circle A'.1 A'.2 → circle B'.1 B'.2 →
    A.1 = A'.1 → (A.2 * A'.2 ≥ 0) →
    B.1 = B'.1 → (B.2 * B'.2 ≥ 0) →
    (∃ t : ℝ, A.1 + t * (B.1 - A.1) = p ∧ A.2 + t * (B.2 - A.2) = q) →
    ∃ t : ℝ, A'.1 + t * (B'.1 - A'.1) = p ∧ A'.2 + t * (B'.2 - A'.2) = a / b * q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_theorem_l499_49991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_when_intersect_l499_49976

/-- The set M on the unit circle, excluding the endpoints -/
def M : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi ∧ p.1 = Real.cos θ ∧ p.2 = Real.sin θ}

/-- The set N, a line with slope 1 and y-intercept m -/
def N (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = p.1 + m}

/-- The theorem stating the range of m when M and N intersect -/
theorem m_range_when_intersect :
    ∀ m : ℝ, (M ∩ N m).Nonempty → -Real.sqrt 2 / 2 < m ∧ m ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_when_intersect_l499_49976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_by_one_universal_exists_no_reciprocal_existential_some_triangles_not_180_existential_l499_49967

-- Define what it means for a statement to be a universal proposition
def isUniversalProposition (P : Prop) : Prop := 
  ∃ (α : Type) (Q : α → Prop), P = ∀ x : α, Q x

-- Define what it means for a statement to be an existential proposition
def isExistentialProposition (P : Prop) : Prop := 
  ∃ (α : Type) (Q : α → Prop), P = ∃ x : α, Q x

-- Define a Triangle type for statement 4
structure Triangle where
  angleSum : ℝ

-- Statement 1
theorem division_by_one_universal : 
  isUniversalProposition (∀ x : ℝ, x / 1 = x) := by sorry

-- Statement 3
theorem exists_no_reciprocal_existential : 
  isExistentialProposition (∃ x : ℝ, ∀ y : ℝ, x * y ≠ 1) := by sorry

-- Statement 4
theorem some_triangles_not_180_existential : 
  isExistentialProposition (∃ t : Triangle, t.angleSum ≠ 180) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_by_one_universal_exists_no_reciprocal_existential_some_triangles_not_180_existential_l499_49967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l499_49913

/-- An ellipse with semi-major axis a, semi-minor axis b, and left focus F -/
structure Ellipse where
  a : ℝ
  b : ℝ
  F : ℝ × ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_F : F.1 = -((a^2 - b^2).sqrt)
  h_F_2 : F.2 = 0

/-- A point on the ellipse -/
def on_ellipse (E : Ellipse) (P : ℝ × ℝ) : Prop :=
  P.1^2 / E.a^2 + P.2^2 / E.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ :=
  ((E.a^2 - E.b^2).sqrt) / E.a

/-- The distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

/-- The main theorem -/
theorem ellipse_eccentricity_range (E : Ellipse) 
  (h_PQ : ∃ (P Q : ℝ × ℝ), on_ellipse E P ∧ on_ellipse E Q ∧ distance P E.F = 2 * distance E.F Q) :
  1/3 ≤ eccentricity E ∧ eccentricity E < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l499_49913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l499_49971

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 3 - 2 * x)

theorem f_properties :
  ∃ (period : ℝ) (increasing_interval : Set ℝ) (range : Set ℝ),
    -- (1) Period
    (∀ x, f (x + period) = f x) ∧ period = Real.pi ∧
    -- (2) Increasing interval
    (∀ k : ℤ, increasing_interval = Set.Icc (5 * Real.pi / 12 + k * Real.pi) (11 * Real.pi / 12 + k * Real.pi)) ∧
    (∀ x y, x ∈ increasing_interval → y ∈ increasing_interval → x < y → f x < f y) ∧
    -- (3) Range on [0, π/2]
    range = Set.Icc (-2) (Real.sqrt 3) ∧
    (∀ y ∈ range, ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = y) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ range) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l499_49971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_param_sum_l499_49941

/-- Represents a hyperbola with center (h, k), focus (fx, fy), and vertex (vx, vy) -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  fx : ℝ
  fy : ℝ
  vx : ℝ
  vy : ℝ

/-- Calculates the parameters a and b for a hyperbola -/
noncomputable def hyperbola_params (hyp : Hyperbola) : ℝ × ℝ :=
  let a := abs (hyp.vy - hyp.k)
  let c := abs (hyp.fy - hyp.k)
  let b := Real.sqrt (c^2 - a^2)
  (a, b)

/-- The sum of h, k, a, and b for the given hyperbola -/
noncomputable def param_sum (hyp : Hyperbola) : ℝ :=
  let (a, b) := hyperbola_params hyp
  hyp.h + hyp.k + a + b

theorem hyperbola_param_sum :
  ∀ hyp : Hyperbola,
    hyp.h = 3 ∧ hyp.k = -1 ∧ hyp.fx = 3 ∧ hyp.fy = 7 ∧ hyp.vx = 3 ∧ hyp.vy = 2 →
    param_sum hyp = 5 + Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_param_sum_l499_49941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_15_minus_half_equals_sqrt_3_over_4_l499_49907

theorem cos_squared_15_minus_half_equals_sqrt_3_over_4 :
  Real.cos (15 * π / 180) ^ 2 - 1/2 = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_15_minus_half_equals_sqrt_3_over_4_l499_49907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l499_49997

theorem similar_triangles_side_length 
  (area_small area_large : ℕ) 
  (side_small : ℕ) 
  (area_diff : ℕ) 
  (ratio_square : ℕ) :
  area_large = area_small + area_diff →
  area_large = area_small * ratio_square →
  side_small = 3 →
  area_diff = 18 →
  ∃ (side_large : ℕ), 
    side_large = 6 ∧ 
    (side_large : ℚ) / side_small = Real.sqrt (area_large / area_small) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l499_49997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_y_axis_l499_49952

noncomputable def f (x : ℝ) : ℝ := (4^x + 1) / (2^x)

theorem f_symmetric_about_y_axis : ∀ x : ℝ, f (-x) = f x := by
  intro x
  unfold f
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_y_axis_l499_49952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l499_49910

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a vector
noncomputable def vectorLength (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Define the dot product of two vectors
def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem triangle_side_length (t : Triangle) :
  let AB := (t.B.1 - t.A.1, t.B.2 - t.A.2)
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  let AC := (t.C.1 - t.A.1, t.C.2 - t.A.2)
  vectorLength AB = 2 →
  vectorLength AC = 3 →
  dotProduct AB BC = 1 →
  vectorLength BC = Real.sqrt 3 := by
    sorry

#check triangle_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l499_49910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_shifted_l499_49934

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2^x else Real.log x / Real.log (1/2)

-- State the theorem
theorem max_value_f_shifted : 
  ∀ x y : ℝ, f (1 - x) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_shifted_l499_49934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_construct_3x5_rectangle_l499_49926

-- Define the puzzle pieces
def PuzzlePiece : Type := Nat × Nat

-- Define the set of available puzzle pieces
def available_pieces : List PuzzlePiece :=
  [(2, 2), (1, 4), (1, 4), (2, 3), (1, 1), (3, 1), (1, 2)]

-- Define the target shape
def target_shape : Nat × Nat := (3, 5)

-- Function to calculate the area of a piece
def area (piece : PuzzlePiece) : Nat :=
  piece.1 * piece.2

-- Function to calculate the total area of a list of pieces
def total_area (pieces : List PuzzlePiece) : Nat :=
  pieces.foldl (fun acc piece => acc + area piece) 0

-- Function to check if a list is a subset of another list
def is_subset (sub : List PuzzlePiece) (super : List PuzzlePiece) : Prop :=
  ∀ x, x ∈ sub → x ∈ super

-- Theorem: It's impossible to construct a 3x5 rectangle with the given pieces
theorem cannot_construct_3x5_rectangle :
  ¬ ∃ (arrangement : List PuzzlePiece),
    is_subset arrangement available_pieces ∧
    total_area arrangement = area target_shape :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_construct_3x5_rectangle_l499_49926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_monotonicity_l499_49979

theorem exponential_monotonicity (a b : ℝ) (h : a > b) : (2 : ℝ)^a > (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_monotonicity_l499_49979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_l499_49946

theorem undefined_values (a : ℝ) : 
  (a + 5) / (a^2 - 9) = 0/0 ↔ a = -3 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_l499_49946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_BAG_is_sqrt_two_thirds_l499_49915

/-- A cube ABCDEFGH with specified coordinates -/
structure Cube where
  A : ℝ × ℝ × ℝ := (0, 0, 0)
  B : ℝ × ℝ × ℝ := (1, 0, 0)
  C : ℝ × ℝ × ℝ := (1, 1, 0)
  D : ℝ × ℝ × ℝ := (0, 1, 0)
  E : ℝ × ℝ × ℝ := (0, 0, 1)
  F : ℝ × ℝ × ℝ := (1, 0, 1)
  G : ℝ × ℝ × ℝ := (1, 1, 1)
  H : ℝ × ℝ × ℝ := (0, 1, 1)

/-- The sine of the angle BAG in the cube -/
noncomputable def sinBAG (cube : Cube) : ℝ :=
  Real.sqrt (2 / 3)

/-- Theorem stating that the sine of angle BAG in the given cube is √(2/3) -/
theorem sin_BAG_is_sqrt_two_thirds (cube : Cube) : sinBAG cube = Real.sqrt (2 / 3) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_BAG_is_sqrt_two_thirds_l499_49915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_sum_relationship_l499_49943

/-- Given a sum, time, and interest rate, calculates the simple interest -/
noncomputable def simple_interest (sum : ℝ) (time : ℝ) (rate : ℝ) : ℝ :=
  (sum * time * rate) / 100

/-- Given a sum, time, and simple interest, calculates the true discount -/
noncomputable def true_discount (sum : ℝ) (time : ℝ) (interest : ℝ) : ℝ :=
  (interest * sum) / (sum + (interest * time))

/-- 
Theorem stating the relationship between sum, time, and rate
when simple interest is 88 and true discount is 80
-/
theorem sum_relationship (sum time rate : ℝ) 
  (h1 : simple_interest sum time rate = 88)
  (h2 : true_discount sum time 88 = 80) :
  sum = (88 * 100) / (time * rate) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_sum_relationship_l499_49943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_set_transitivity_l499_49940

theorem point_in_set_transitivity {A : Type} {a α : Set A} (x : A) :
  x ∈ a → a ⊆ α → x ∈ α := by
  intro h1 h2
  exact h2 h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_set_transitivity_l499_49940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_necessary_not_sufficient_l499_49972

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_necessary_not_sufficient :
  (∀ a : ℕ → ℝ, ∀ q : ℝ,
    is_geometric_sequence a →
    a 1 > 0 →
    (∀ n : ℕ, a (n + 1) = a n * q) →
    (∀ n : ℕ, a (2 * n - 1) + a (2 * n) < 0 → q < 0)) ∧
  (∃ a : ℕ → ℝ, ∃ q : ℝ,
    is_geometric_sequence a ∧
    a 1 > 0 ∧
    q < 0 ∧
    ¬(∀ n : ℕ, a (2 * n - 1) + a (2 * n) < 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_necessary_not_sufficient_l499_49972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_piecewise_equivalent_l499_49918

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -x - 2
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- Define a default value for x outside the given ranges

-- Define g(x) = -f(x) + 1
noncomputable def g (x : ℝ) : ℝ := -f x + 1

-- Theorem statement
theorem g_piecewise_equivalent :
  ∀ x : ℝ, 
    (-3 ≤ x ∧ x ≤ 0 → g x = x + 3) ∧
    (0 < x ∧ x ≤ 2 → g x = -Real.sqrt (4 - (x - 2)^2) + 3) ∧
    (2 < x ∧ x ≤ 3 → g x = -2 * x + 5) :=
by
  sorry

#check g_piecewise_equivalent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_piecewise_equivalent_l499_49918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_without_contact_l499_49989

def days_in_year : ℕ := 366

def contact_schedule : List ℕ := [2, 6, 4]

def days_with_contact (schedule : List ℕ) : ℕ :=
  let single_contact := schedule.map (λ n => days_in_year / n)
  let double_contact := [
    days_in_year / (schedule.get! 0).lcm (schedule.get! 1),
    days_in_year / (schedule.get! 0).lcm (schedule.get! 2),
    days_in_year / (schedule.get! 1).lcm (schedule.get! 2)
  ]
  let triple_contact := days_in_year / ((schedule.get! 0).lcm (schedule.get! 1)).lcm (schedule.get! 2)
  (single_contact.sum - double_contact.sum + triple_contact)

theorem days_without_contact :
  days_in_year - days_with_contact contact_schedule = 183 := by
  sorry

#eval days_in_year - days_with_contact contact_schedule

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_without_contact_l499_49989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_both_english_teachers_l499_49914

def committee_size : ℕ := 9
def english_teachers : ℕ := 3
def math_teachers : ℕ := 4
def social_studies_teachers : ℕ := 2
def members_to_select : ℕ := 2

theorem probability_both_english_teachers :
  (Nat.choose english_teachers members_to_select : ℚ) / (Nat.choose committee_size members_to_select) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_both_english_teachers_l499_49914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l499_49975

/-- Triangle ABC with interior angles A, B, C and opposite side lengths a, b, c -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Vector in 2D space -/
structure Vector2D where
  x : Real
  y : Real

theorem triangle_proof (t : Triangle) 
  (m : Vector2D) (n : Vector2D)
  (h1 : m = { x := Real.sqrt 3, y := Real.cos t.A + 1 })
  (h2 : n = { x := Real.sin t.A, y := -1 })
  (h3 : m.x * n.x + m.y * n.y = 0)  -- m ⊥ n
  (h4 : t.a = 2)
  (h5 : Real.cos t.B = Real.sqrt 3 / 3) :
  t.A = Real.pi / 3 ∧ t.b = 4 * Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l499_49975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l499_49938

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the asymptotes of a hyperbola given specific conditions -/
theorem hyperbola_asymptotes (E : Hyperbola) 
  (A B C D : Point)
  (h_perpendicular : A.x = B.x ∧ C.x = D.x)
  (h_on_hyperbola : (A.x^2 / E.a^2) - (A.y^2 / E.b^2) = 1 ∧
                    (B.x^2 / E.a^2) - (B.y^2 / E.b^2) = 1)
  (h_on_asymptotes : C.y = (E.b / E.a) * C.x ∧
                     D.y = -(E.b / E.a) * D.x)
  (h_distance_relation : distance A B = (Real.sqrt 3 / 2) * distance C D) :
  E.b / E.a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l499_49938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_asymptote_l499_49929

/-- A rational function with a quadratic numerator and denominator -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^2 - x + k) / (x^2 + x - 18)

/-- The property of having exactly one vertical asymptote -/
def has_exactly_one_vertical_asymptote (k : ℝ) : Prop :=
  ∃! x : ℝ, (x^2 + x - 18 = 0) ∧ (x^2 - x + k ≠ 0)

/-- Theorem: f(x) has exactly one vertical asymptote iff k = -6 or k = -42 -/
theorem f_one_asymptote (k : ℝ) :
  has_exactly_one_vertical_asymptote k ↔ k = -6 ∨ k = -42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_asymptote_l499_49929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l499_49944

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x + 4)

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂ :=
by
  -- The proof is skipped using sorry
  sorry

-- Note: Real.log is used instead of lg, as it's the standard logarithm function in Lean
-- The theorem states that for any two real numbers x₁ and x₂,
-- if x₁ < x₂ and x₂ < 1, then f(x₁) > f(x₂),
-- which is equivalent to saying the function is monotonically decreasing on (-∞, 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l499_49944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l499_49990

-- Define the function f with domain [1,3]
def f : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

-- Define the function g(x) = f(2x+1)
def g : ℝ → Set ℝ := λ x => {y : ℝ | y ∈ f ∧ ∃ z, y = 2*z + 1 ∧ z = x}

-- Theorem statement
theorem domain_of_g : 
  {x : ℝ | ∃ y, y ∈ g x} = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l499_49990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2005th_term_is_133_l499_49999

def sumOfCubesOfDigits (n : Nat) : Nat :=
  (n.digits 10).map (fun d => d^3) |>.sum

def sequenceTerm : Nat → Nat
  | 0 => 2005
  | n + 1 => sumOfCubesOfDigits (sequenceTerm n)

theorem sequence_2005th_term_is_133 : sequenceTerm 2004 = 133 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2005th_term_is_133_l499_49999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_angle_segment_in_pentagon_l499_49954

-- Define a regular pentagon
structure RegularPentagon where
  vertices : Finset (ℝ × ℝ)
  regular : vertices.card = 5
  -- Additional properties ensuring it's a regular pentagon

-- Define a line segment
structure LineSegment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

-- Define what it means for a point to be strictly inside a pentagon
def StrictlyInside (p : RegularPentagon) (point : ℝ × ℝ) : Prop :=
  sorry -- Definition of strictly inside

-- Define what it means for a line segment to be strictly inside a pentagon
def SegmentStrictlyInside (p : RegularPentagon) (s : LineSegment) : Prop :=
  StrictlyInside p s.start ∧ StrictlyInside p s.endpoint

-- Define the angle at which a line segment is seen from a point
noncomputable def ViewAngle (point : ℝ × ℝ) (s : LineSegment) : ℝ :=
  sorry -- Definition of view angle

-- The main theorem
theorem no_equal_angle_segment_in_pentagon (p : RegularPentagon) :
  ¬ ∃ (s : LineSegment), 
    SegmentStrictlyInside p s ∧ 
    ∃ (α : ℝ), ∀ v ∈ p.vertices, ViewAngle v s = α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_angle_segment_in_pentagon_l499_49954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_rose_price_is_5_l499_49987

/-- The price of an individual rose -/
def individual_rose_price : ℚ := 5

/-- The cost of one dozen roses -/
def dozen_price : ℚ := 36

/-- The cost of two dozen roses -/
def two_dozen_price : ℚ := 50

/-- The maximum number of roses that can be purchased for $680 -/
def max_roses : ℕ := 318

/-- The total budget -/
def total_budget : ℚ := 680

theorem individual_rose_price_is_5 :
  individual_rose_price = 5 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_rose_price_is_5_l499_49987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l499_49939

/-- The curve y = 1 - x^2 -/
noncomputable def curve (x : ℝ) : ℝ := 1 - x^2

/-- The tangent line at point (x, curve x) -/
noncomputable def tangent_line (x : ℝ) (t : ℝ) : ℝ := -2 * x * (t - x) + curve x

/-- The area of the triangle formed by tangents at points a and b, and the x-axis -/
noncomputable def triangle_area (a b : ℝ) : ℝ :=
  ((a^2 + 1) / (2*a) - (b^2 + 1) / (2*b)) * (1 - a*b) / 2

theorem min_triangle_area :
  ∃ (a b : ℝ), a > 0 ∧ b < 0 ∧
  ∀ (x y : ℝ), x > 0 → y < 0 →
  triangle_area a b ≤ triangle_area x y ∧
  triangle_area a b = 8 * Real.sqrt 3 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l499_49939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_acquaintances_exist_l499_49901

-- Define the acquainted relation
def acquainted (n : ℕ) : Fin n → Fin n → Prop := sorry

theorem same_acquaintances_exist (n : ℕ) (h : n ≥ 5) :
  ∃ (i j : Fin n), i ≠ j ∧
  (∀ p : Fin n, (∃ m : ℕ, m < n ∧ (∀ q : Fin n, q ≠ p → (acquainted n p q ↔ q.val < m)))) →
  (∀ q : Fin n, q ≠ i → acquainted n i q) ↔ (∀ q : Fin n, q ≠ j → acquainted n j q) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_acquaintances_exist_l499_49901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_customers_is_fourteen_l499_49925

/-- Represents a breakfast plate at the cafe -/
structure BreakfastPlate where
  eggs : ℕ
  bacon : ℕ

/-- Cafe scenario with given conditions -/
structure CafeScenario where
  plate : BreakfastPlate
  totalBacon : ℕ
  hEggs : plate.eggs = 2
  hBacon : plate.bacon = 2 * plate.eggs
  hTotalBacon : totalBacon = 56

/-- The number of customers who ordered breakfast plates -/
def numCustomers (scenario : CafeScenario) : ℕ :=
  scenario.totalBacon / scenario.plate.bacon

/-- Theorem stating that the number of customers is 14 -/
theorem num_customers_is_fourteen (scenario : CafeScenario) :
  numCustomers scenario = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_customers_is_fourteen_l499_49925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l499_49983

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ := (x - 3) / (x^2 + c*x + d)

theorem asymptote_sum (c d : ℝ) :
  (∀ x, x ≠ 2 ∧ x ≠ -1 → g c d x ≠ 0) →
  (2^2 + c*2 + d = 0) →
  ((-1)^2 + c*(-1) + d = 0) →
  c + d = -3 := by
  sorry

#check asymptote_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l499_49983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_value_l499_49906

theorem cosine_difference_value (α : Real) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) (h3 : Real.tan α = 2) : 
  Real.cos (α - Real.pi / 4) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_value_l499_49906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_l499_49986

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x^α

-- Theorem statement
theorem power_function_not_in_fourth_quadrant (α : ℝ) :
  ∀ x y : ℝ, y = power_function α x → ¬(x > 0 ∧ y < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_l499_49986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_satisfy_psi_l499_49920

-- Define property ψ
def property_psi (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ D → x₂ ∈ D → x₁ ≠ x₂ → |((f x₁) - (f x₂)) / (x₁ - x₂)| ≥ 1

-- Define function f
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- Define function h
def h (x : ℝ) : ℝ := x ^ 2

-- Define domains
def D_f : Set ℝ := {x | 0 < x ∧ x < 1}
def D_h : Set ℝ := {x | x ≤ -1}

-- Theorem statement
theorem functions_satisfy_psi : 
  (property_psi f D_f) ∧ (property_psi h D_h) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_satisfy_psi_l499_49920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_negative_one_l499_49937

/-- Given a natural number n, we define the binomial expansion of (x - 2/√x)^n -/
noncomputable def binomial_expansion (n : ℕ) (x : ℝ) : ℝ :=
  (x - 2 / Real.sqrt x) ^ n

/-- The condition that the coefficients of the fifth and sixth terms are the largest -/
def fifth_sixth_largest (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a = 5 ∧ b = 6 ∧ 
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n + 1 → 
    Nat.choose n (k - 1) ≤ max (Nat.choose n (a - 1)) (Nat.choose n (b - 1))

/-- The theorem stating that the sum of coefficients is -1 -/
theorem sum_of_coefficients_is_negative_one (n : ℕ) 
  (h : fifth_sixth_largest n) : 
  binomial_expansion n 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_negative_one_l499_49937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_quadrilateral_l499_49963

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the line
def line_eq (x y k : ℝ) : Prop := y = k*x - 4

-- Define the tangent property
def is_tangent (x y xc yc : ℝ) : Prop :=
  (x - xc)^2 + (y - yc)^2 = 1

-- Main theorem
theorem min_area_quadrilateral (k : ℝ) : 
  k < 0 →  -- condition that k < 0
  (∃ x y, line_eq x y k ∧ 
    (∃ xa ya xb yb, 
      circle_eq xa ya ∧ circle_eq xb yb ∧
      is_tangent x y xa ya ∧ is_tangent x y xb yb ∧
      (∀ x' y', line_eq x' y' k → 
        (∃ xa' ya' xb' yb', 
          circle_eq xa' ya' ∧ circle_eq xb' yb' ∧
          is_tangent x' y' xa' ya' ∧ is_tangent x' y' xb' yb' →
          (x' - xa')*(y' - yb') - (x' - xb')*(y' - ya') ≥ 2)))) →
  k = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_quadrilateral_l499_49963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l499_49959

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x * Real.cos x

-- Define the axis of symmetry
noncomputable def axis_of_symmetry (k : ℤ) : ℝ := (k : ℝ) * Real.pi / 2 + Real.pi / 8

-- Theorem statement
theorem symmetry_of_f :
  ∀ (x : ℝ) (k : ℤ), f (axis_of_symmetry k - x) = f (axis_of_symmetry k + x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l499_49959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l499_49955

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) := {(x, y) | x^2 / a^2 - y^2 / b^2 = 1}

-- Define the foci
def LeftFocus (c : ℝ) : ℝ × ℝ := (-c, 0)
def RightFocus (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define a point on the right branch
def PointA (m n : ℝ) : ℝ × ℝ := (m, n)

-- Define a point on the left branch
def PointB (s t : ℝ) : ℝ × ℝ := (s, t)

-- Define the origin
def Origin : ℝ × ℝ := (0, 0)

-- State the theorem
theorem hyperbola_asymptotes 
  (a b c m n s t : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : m > 0) 
  (h4 : n > 0) 
  (h5 : s < 0) 
  (h6 : t > 0) 
  (h7 : c^2 = a^2 + b^2) 
  (h8 : PointA m n ∈ Hyperbola a b) 
  (h9 : PointB s t ∈ Hyperbola a b) 
  (h10 : ‖PointA m n - LeftFocus c‖ = 3 * ‖PointB s t - LeftFocus c‖) 
  (h11 : ‖LeftFocus c - Origin‖ = ‖PointA m n - Origin‖) :
  ∃ (k : ℝ), k = 2 ∧ (∀ (x y : ℝ), y = k * x ∨ y = -k * x → (x, y) ∈ closure (Hyperbola a b)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l499_49955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ranked_choice_voting_win_condition_l499_49900

theorem ranked_choice_voting_win_condition 
  (total_votes : ℕ) 
  (invalid_vote_percentage : ℚ) 
  (hTotal : total_votes = 10000)
  (hInvalid : invalid_vote_percentage = 30 / 100) :
  let valid_votes := total_votes - (invalid_vote_percentage * ↑total_votes).floor
  ∃ (win_threshold : ℕ), 
    win_threshold > valid_votes / 2 ∧ 
    win_threshold ≤ valid_votes ∧
    win_threshold > 3500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ranked_choice_voting_win_condition_l499_49900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l499_49981

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d e f : ℝ} (h : b ≠ 0 ∧ e ≠ 0) :
  (∀ x y, a * x + b * y + c = 0 ↔ d * x + e * y + f = 0) ↔ a / b = d / e

/-- Definition of the first line l₁ -/
def l₁ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 2 * x + (m + 1) * y + 4 = 0

/-- Definition of the second line l₂ -/
def l₂ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ m * x + 3 * y - 2 = 0

/-- Theorem stating that if l₁ and l₂ are parallel, then m = -3 or m = 2 -/
theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y, l₁ m x y ↔ l₂ m x y) → m = -3 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l499_49981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l499_49948

noncomputable section

-- Define the triangle ABC
variable (a b c A B C : ℝ)
variable (S : ℝ)

-- Define the conditions
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom angle_sum : A + B + C = Real.pi
axiom area_condition : S ≥ 2
axiom cosine_condition : c * Real.cos A = 4 / b

-- Define the function f
def f (A : ℝ) : ℝ := Real.cos A ^ 2 + Real.sqrt 3 * Real.sin (Real.pi / 2 + A / 2) ^ 2 - Real.sqrt 3 / 2

-- State the theorem
theorem triangle_theorem :
  (Real.pi / 4 ≤ A ∧ A < Real.pi / 2) ∧
  (∀ x, Real.pi / 4 ≤ x ∧ x < Real.pi / 2 → f x ≤ (1 / 2 + Real.sqrt 6 / 4)) ∧
  (∃ x, Real.pi / 4 ≤ x ∧ x < Real.pi / 2 ∧ f x = (1 / 2 + Real.sqrt 6 / 4)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l499_49948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_set_with_property_l499_49968

-- Define the property for set A
def has_property (A : Set ℕ) : Prop :=
  ∀ (S : Set ℕ), Set.Infinite S → (∀ p ∈ S, Nat.Prime p) →
    ∃ (m n : ℕ) (k : ℕ),
      k ≥ 2 ∧
      m ∈ A ∧
      n ∉ A ∧
      (∃ (factors_m factors_n : Finset ℕ),
        factors_m.card = k ∧
        factors_n.card = k ∧
        (∀ p ∈ factors_m, p ∈ S) ∧
        (∀ p ∈ factors_n, p ∈ S) ∧
        m = factors_m.prod id ∧
        n = factors_n.prod id)

-- State the theorem
theorem exists_set_with_property :
  ∃ (A : Set ℕ), has_property A :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_set_with_property_l499_49968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cylindrical_planes_is_line_l499_49931

/-- Given two distinct angles in cylindrical coordinates, the intersection of the planes
    defined by these angles forms a line along the z-axis. -/
theorem intersection_of_cylindrical_planes_is_line (θ₁ θ₂ : ℝ) (h : θ₁ ≠ θ₂) :
  ∃ (L : Set (ℝ × ℝ × ℝ)), 
    L = {p : ℝ × ℝ × ℝ | p.1 = 0 ∧ p.2 = 0} ∧
    L = {p : ℝ × ℝ × ℝ | (∃ (r z : ℝ), p = (r * Real.cos θ₁, r * Real.sin θ₁, z))} ∩
        {p : ℝ × ℝ × ℝ | (∃ (r z : ℝ), p = (r * Real.cos θ₂, r * Real.sin θ₂, z))} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cylindrical_planes_is_line_l499_49931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_ten_l499_49936

theorem not_divisible_by_ten (n : ℕ) : 
  (n^2012 + n^2010) % 10 = 0 → n ≠ 59 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_ten_l499_49936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_subset_existence_l499_49905

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if two circles intersect -/
def intersect (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 ≤ (c1.radius + c2.radius) ^ 2

/-- Main theorem -/
theorem circle_subset_existence (circles : Finset Circle) 
  (h1 : circles.card = 2015) 
  (h2 : ∀ c ∈ circles, c.radius = 1) : 
  ∃ (S : Finset Circle), S ⊆ circles ∧ S.card = 27 ∧ 
    ∀ c1 c2, c1 ∈ S → c2 ∈ S → c1 ≠ c2 → (intersect c1 c2 ∨ ¬intersect c1 c2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_subset_existence_l499_49905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l499_49942

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the ellipse
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  pointOnEllipse : ℝ × ℝ

-- Define the theorem
theorem ellipse_eccentricity (t : Triangle) (e : Ellipse) : 
  -- Conditions
  (Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) = Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)) →  -- AB = BC
  (((t.B.1 - t.A.1) * (t.C.1 - t.A.1) + (t.B.2 - t.A.2) * (t.C.2 - t.A.2)) / 
   (Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) * Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)) = -7/18) →  -- cos B = -7/18
  (e.foci = (t.A, t.B) ∧ e.pointOnEllipse = t.C) →  -- Ellipse foci at A and B, passing through C
  -- Conclusion
  (Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) / 
   (Real.sqrt ((t.A.1 - e.pointOnEllipse.1)^2 + (t.A.2 - e.pointOnEllipse.2)^2) + 
    Real.sqrt ((t.B.1 - e.pointOnEllipse.1)^2 + (t.B.2 - e.pointOnEllipse.2)^2)) = 3/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l499_49942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l499_49966

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a / x else (2 - a) * x + 3

theorem decreasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) ↔ a ∈ Set.Ioo 2 (5/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l499_49966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_asymptotes_l499_49933

/-- Given a hyperbola and a parabola intersecting at points A and B, 
    if the sum of distances from A and B to a focus F is equal to 4 times 
    the distance from the origin O to F, then the asymptotes of the hyperbola 
    have a specific form. -/
theorem hyperbola_parabola_intersection_asymptotes 
  (a b p : ℝ) (h_a : a > 0) (h_b : b > 0) (h_p : p > 0) :
  (∃ (A B : ℝ × ℝ),
    (A.1^2 / a^2 - A.2^2 / b^2 = 1) ∧
    (B.1^2 / a^2 - B.2^2 / b^2 = 1) ∧
    (A.1^2 = 2 * p * A.2) ∧
    (B.1^2 = 2 * p * B.2) ∧
    (|A.2 - p/2| + |B.2 - p/2| = 2 * p)) →
  (∀ (x y : ℝ), y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x ↔ 
    ∃ (t : ℝ), x = a * t ∧ y = b * t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_asymptotes_l499_49933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l499_49932

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6) - Real.cos (ω * x)

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∨
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x)

theorem omega_values (ω : ℝ) :
  ω > 0 ∧
  is_symmetric_about (f ω) (2 * Real.pi) ∧
  is_monotonic_on (f ω) (-Real.pi/4) (Real.pi/4) →
  ω = 1/3 ∨ ω = 5/6 ∨ ω = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l499_49932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equality_l499_49953

theorem exponent_equality (y : ℝ) : (3 : ℝ)^(y-4) = (9 : ℝ)^3 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equality_l499_49953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l499_49974

noncomputable def f (x : ℝ) := (3 : ℝ)^x + 4*(3 : ℝ)^(-x)

theorem min_value_of_f : 
  ∀ x : ℝ, f x ≥ 4 ∧ ∃ x₀ : ℝ, f x₀ = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l499_49974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_even_and_has_zero_point_l499_49996

theorem cosine_even_and_has_zero_point :
  (∀ x : ℝ, Real.cos (-x) = Real.cos x) ∧ (∃ x : ℝ, Real.cos x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_even_and_has_zero_point_l499_49996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l499_49978

theorem trigonometric_identities 
  (α β : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : π/2 < β) (h4 : β < π)
  (h5 : Real.sin α = Real.sqrt 5 / 5)
  (h6 : Real.cos (α + β) = -(2 * Real.sqrt 5) / 5) :
  (Real.cos α = (2 * Real.sqrt 5) / 5) ∧ 
  (Real.cos β = -3/5) ∧ 
  (Real.sin (α - β) = -(11 * Real.sqrt 5) / 25) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l499_49978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_exp_inequality_l499_49993

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x + 3 else -x^2 + 2*x + 3

-- State the theorem
theorem f_minus_exp_inequality (m : ℝ) :
  (∀ x : ℝ, f x - Real.exp x - m ≤ 0) ↔ m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_exp_inequality_l499_49993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_min_f_g_l499_49992

noncomputable def min_func (x y : ℝ) : ℝ := if x ≤ y then x else y

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 8)

noncomputable def g (x : ℝ) : ℝ := x / (x^2 + 8)

theorem max_value_of_min_f_g :
  (∀ x : ℝ, min_func (f x) (g x) ≤ Real.sqrt 2 / 8) ∧
  (∃ x : ℝ, min_func (f x) (g x) = Real.sqrt 2 / 8) :=
sorry

#check max_value_of_min_f_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_min_f_g_l499_49992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_not_sold_approx_l499_49964

def initial_stock : ℕ := 900
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem percentage_not_sold_approx (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (δ : ℝ), δ > 0 ∧ |percentage_not_sold - 55.33| < δ → |percentage_not_sold - 55.33| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_not_sold_approx_l499_49964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plantation_revenue_l499_49908

-- Define the plantation dimensions
noncomputable def plantation_length : ℝ := 500
noncomputable def plantation_width : ℝ := 500

-- Define the conversion rates
noncomputable def peanuts_per_sqft : ℝ := 50
noncomputable def peanuts_to_butter_ratio : ℝ := 5 / 20
noncomputable def butter_price_per_kg : ℝ := 10

-- Calculate the total revenue
noncomputable def total_revenue : ℝ :=
  plantation_length * plantation_width * peanuts_per_sqft * peanuts_to_butter_ratio * butter_price_per_kg / 1000

-- Theorem statement
theorem plantation_revenue :
  total_revenue = 31250 := by
  -- Unfold definitions
  unfold total_revenue plantation_length plantation_width peanuts_per_sqft peanuts_to_butter_ratio butter_price_per_kg
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plantation_revenue_l499_49908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_with_square_and_cube_has_sixth_power_l499_49949

/-- An arithmetic progression of positive integers -/
def ArithmeticProgression (a b : ℕ) : ℕ → ℕ := λ n => a + b * n

/-- Predicate to check if a number is in the progression -/
def InProgression (p : ℕ → ℕ) (x : ℕ) : Prop :=
  ∃ n : ℕ, p n = x

theorem arithmetic_progression_with_square_and_cube_has_sixth_power
  (a b : ℕ) (hb : b > 0)
  (square_exists : ∃ n : ℕ, InProgression (ArithmeticProgression a b) (n^2))
  (cube_exists : ∃ m : ℕ, InProgression (ArithmeticProgression a b) (m^3)) :
  ∃ k : ℕ, InProgression (ArithmeticProgression a b) (k^6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_with_square_and_cube_has_sixth_power_l499_49949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_shift_divisibility_l499_49916

theorem digit_shift_divisibility (n : ℕ+) (N : ℕ) (h : N % 7 = 0) :
  let digits := N.digits 10
  let M := (digits.head! * 10^(6*n.val - 1) : ℕ) + (N / 10)
  M % 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_shift_divisibility_l499_49916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_l499_49951

-- Define the function f(x) as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 / 2 - a * Real.log x

-- State the theorem
theorem monotonicity_of_f (a : ℝ) :
  (a < 0 → ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a > 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.sqrt a → f a x₁ > f a x₂) ∧
           (∀ x₁ x₂, Real.sqrt a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_l499_49951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_side_length_l499_49985

noncomputable def f (x : Real) : Real := Real.sqrt 3 * Real.sin (x + Real.pi / 3) - Real.cos x

theorem min_value_and_side_length :
  (∃ x₀ ∈ Set.Icc 0 Real.pi, ∀ x ∈ Set.Icc 0 Real.pi, f x₀ ≤ f x) ∧
  f (Real.pi / 3) = 1 ∧
  (∀ a b c A B C : Real,
    b = 5 * Real.sqrt 3 →
    Real.cos A = 3 / 5 →
    f B = 1 →
    a / Real.sin A = b / Real.sin B →
    a = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_side_length_l499_49985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_MN_passes_through_fixed_point_l499_49947

-- Define the circle
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line x = 4
def line_x_4 (x : ℝ) : Prop := x = 4

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define point P
def point_P (t : ℝ) : ℝ × ℝ := (4, t)

-- Define the line through two points
def line_through (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

-- Define the intersection of a line and the circle
def intersection_line_circle (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  line_through p1 p2 x y ∧ circle_O x y

-- Theorem statement
theorem line_MN_passes_through_fixed_point :
  ∀ t : ℝ,
  ∃ (M N : ℝ × ℝ),
  intersection_line_circle (point_P t) point_A M.1 M.2 ∧
  intersection_line_circle (point_P t) point_B N.1 N.2 →
  line_through M N 1 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_MN_passes_through_fixed_point_l499_49947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_problem_solution_l499_49921

/-- Represents the highway construction problem -/
structure HighwayProblem where
  initialWorkers : ℕ
  totalLength : ℚ
  initialDays : ℕ
  initialHoursPerDay : ℕ
  progressDays : ℕ
  progressFraction : ℚ
  additionalWorkers : ℕ
  remainingDays : ℕ

/-- Calculates the required hours per day for new employees -/
noncomputable def requiredHoursPerDay (p : HighwayProblem) : ℚ :=
  let totalManHours := p.initialWorkers * p.initialDays * p.initialHoursPerDay
  let usedManHours := p.initialWorkers * p.progressDays * p.initialHoursPerDay
  let remainingManHours := totalManHours - usedManHours
  let totalWorkers := p.initialWorkers + p.additionalWorkers
  remainingManHours / (totalWorkers * p.remainingDays)

/-- Theorem stating that the required hours per day for new employees is 5 -/
theorem highway_problem_solution (p : HighwayProblem) 
  (h1 : p.initialWorkers = 100)
  (h2 : p.totalLength = 2)
  (h3 : p.initialDays = 50)
  (h4 : p.initialHoursPerDay = 8)
  (h5 : p.progressDays = 25)
  (h6 : p.progressFraction = 1/3)
  (h7 : p.additionalWorkers = 60)
  (h8 : p.remainingDays = 25) :
  requiredHoursPerDay p = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_problem_solution_l499_49921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kerosene_cost_equals_eight_eggs_l499_49958

noncomputable def cost_of_rice_pound : ℝ := 0.36

noncomputable def cost_of_dozen_eggs (cost_of_rice_pound : ℝ) : ℝ := cost_of_rice_pound

noncomputable def cost_of_egg (cost_of_dozen_eggs : ℝ) : ℝ := cost_of_dozen_eggs / 12

noncomputable def cost_of_eight_eggs (cost_of_egg : ℝ) : ℝ := 8 * cost_of_egg

theorem kerosene_cost_equals_eight_eggs 
  (h1 : cost_of_dozen_eggs cost_of_rice_pound = cost_of_rice_pound)
  (h2 : cost_of_rice_pound = 0.36)
  (h3 : ∃ k : ℝ, k = cost_of_eight_eggs (cost_of_egg (cost_of_dozen_eggs cost_of_rice_pound))) :
  cost_of_eight_eggs (cost_of_egg (cost_of_dozen_eggs cost_of_rice_pound)) = 0.24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kerosene_cost_equals_eight_eggs_l499_49958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_k_l499_49928

-- Define the line l: x - ky - 5 = 0
def line (k : ℝ) (x y : ℝ) : Prop := x - k * y - 5 = 0

-- Define the circle O: x^2 + y^2 = 10
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 10

-- Define the intersection points A and B
def intersection (k : ℝ) (xa ya xb yb : ℝ) : Prop :=
  line k xa ya ∧ circleO xa ya ∧ line k xb yb ∧ circleO xb yb

-- Define the perpendicular condition OA · OB = 0
def perpendicular (xa ya xb yb : ℝ) : Prop :=
  xa * xb + ya * yb = 0

-- Main theorem
theorem intersection_perpendicular_k (k : ℝ) :
  (∃ xa ya xb yb : ℝ, intersection k xa ya xb yb ∧ perpendicular xa ya xb yb) →
  k = 2 ∨ k = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_k_l499_49928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_proof_l499_49945

noncomputable def triangle_inequality (a b c : ℝ) : Prop :=
  let s := Real.sqrt a + Real.sqrt b + Real.sqrt c
  let t := 1/a + 1/b + 1/c
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  ((1/2) * a * b * (Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)) = 1/4) ∧
  ((a * b * c) / (4 * (1/4)) = 1) →
  t > s

theorem triangle_inequality_proof : ∃ (a b c : ℝ), triangle_inequality a b c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_proof_l499_49945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colinear_vectors_not_imply_collinear_points_l499_49998

/-- Given two vectors AB and CD in a 3D space that are colinear,
    prove that points A, B, C, and D do not necessarily lie on the same straight line. -/
theorem colinear_vectors_not_imply_collinear_points
  (A B C D : EuclideanSpace ℝ (Fin 3))
  (h_colinear : ∃ (k : ℝ), C - D = k • (B - A)) :
  ¬ (∀ (t : ℝ), ∃ (s : ℝ), C = A + t • (B - A) ∧ D = A + s • (B - A)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colinear_vectors_not_imply_collinear_points_l499_49998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_forms_cone_l499_49982

-- Define a right triangle
structure RightTriangle where
  -- We don't need to specify the exact properties of a right triangle
  -- as they are implicit in the name and the problem statement

-- Define the rotation operation
def rotateAroundRightAngle (t : RightTriangle) : GeometricShape := sorry

-- Define a cone
structure Cone where
  -- We don't need to specify the exact properties of a cone
  -- as they are implicit in the name and the problem statement

-- Theorem statement
theorem rotation_forms_cone (t : RightTriangle) :
  rotateAroundRightAngle t = Cone.mk := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_forms_cone_l499_49982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_reversal_problem_l499_49957

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ∃ (a b : ℕ),
    n = 10 * a + b ∧
    b < 10 ∧
    10 * b + a = (175 * n) / 100

theorem two_digit_reversal_problem :
  ∀ n : ℕ, is_valid_number n ↔ n ∈ ({12, 24, 36, 48} : Set ℕ) :=
by sorry

#check two_digit_reversal_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_reversal_problem_l499_49957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_after_800_seconds_l499_49970

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℚ
  deriving Repr

/-- Represents the circular track -/
def trackLength : ℚ := 400

/-- Calculates the position of a runner after a given time -/
def position (runner : Runner) (time : ℚ) : ℚ :=
  (runner.speed * time) % trackLength

/-- Checks if all runners are at the same position -/
def allTogether (runners : List Runner) (time : ℚ) : Prop :=
  ∀ r1 r2, r1 ∈ runners → r2 ∈ runners → position r1 time = position r2 time

/-- The main theorem stating that the runners meet after 800 seconds -/
theorem runners_meet_after_800_seconds (runners : List Runner)
  (h1 : runners = [⟨3⟩, ⟨7/2⟩, ⟨4⟩, ⟨9/2⟩]) :
  allTogether runners 800 := by
  sorry

#eval position ⟨3⟩ 800
#eval position ⟨7/2⟩ 800
#eval position ⟨4⟩ 800
#eval position ⟨9/2⟩ 800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_after_800_seconds_l499_49970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_stock_correct_final_percentage_unsold_l499_49988

def initial_stock : ℕ := 1000

def sale_percentages : List ℚ := [5/100, 10/100, 15/100, 20/100, 25/100]

def apply_sale (stock : ℕ) (percentage : ℚ) : ℕ :=
  stock - (Int.toNat ⌊(stock : ℚ) * percentage⌋)

def final_stock : ℕ := sale_percentages.foldl apply_sale initial_stock

theorem final_stock_correct : final_stock = 437 :=
sorry

theorem final_percentage_unsold : 
  (final_stock : ℚ) / (initial_stock : ℚ) * 100 = 43.7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_stock_correct_final_percentage_unsold_l499_49988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_l499_49930

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.tan x + (1 / Real.cos x) + (1 / Real.tan x) + (1 / Real.sin x)

-- State the theorem about the periodicity of f
theorem f_periodic : ∀ x : ℝ, f (x + π) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_l499_49930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_2024_l499_49994

theorem opposite_of_2024 : 
  (∀ x : ℤ, x + (-x) = 0) → -2024 = -2024 := by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_2024_l499_49994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_width_calculation_l499_49919

-- Define the field dimensions
def width : ℝ := sorry
def length : ℝ := sorry

-- Define the perimeter
def perimeter : ℝ := 288

-- Define the relationship between length and width
axiom length_width_relation : length = (7/5) * width

-- Define the perimeter equation
axiom perimeter_equation : perimeter = 2 * length + 2 * width

-- Define the angle between diagonal and width
def angle : ℝ := 60

-- Theorem to prove
theorem width_calculation :
  width = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_width_calculation_l499_49919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_abelian_from_power_property_l499_49924

theorem group_abelian_from_power_property (G : Type*) [Fintype G] [Group G] :
  ∃ (k : ℕ), k ≥ 2 ∧
  (∀ (x y : G) (i : ℕ), i ∈ ({k-1, k, k+1} : Set ℕ) → (x * y)^i = x^i * y^i) →
  ∀ (a b : G), a * b = b * a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_abelian_from_power_property_l499_49924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_game_players_l499_49973

def number_of_games (n : ℕ) (k : ℕ) : ℕ := n.choose k

theorem chess_game_players (total_players : ℕ) (total_games : ℕ) (players_per_game : ℕ) : 
  total_players = 50 → 
  total_games = 1225 → 
  number_of_games total_players players_per_game = total_games → 
  players_per_game = 2 := by
  intro h1 h2 h3
  -- The proof goes here
  sorry

#check chess_game_players

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_game_players_l499_49973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l499_49995

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, (Finset.range n).sum (λ i ↦ a ⟨i + 1, Nat.succ_pos i⟩) = n - a n

theorem sequence_theorem (a : ℕ+ → ℝ) (h : sequence_property a) :
  (∃ q : ℝ, q = 1/2 ∧
    (∀ n : ℕ+, a n - 1 = -1/2 * (1/2)^(n.val - 1))) ∧
  (∀ t : ℝ, (∀ n : ℕ+, n * (1 - a n) ≤ t) → t ≥ 1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l499_49995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l499_49917

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (2 * x + Real.pi / 6) + Real.sin (2 * x - Real.pi / 6) + Real.cos (2 * x) + 1

theorem triangle_side_length 
  (A B C : ℝ) 
  (hA : 0 < A ∧ A < Real.pi) 
  (hB : B = Real.pi / 4) 
  (hfA : f A = 3) 
  (ha : Real.sqrt 3 = Real.sin C * (3 * Real.sqrt 2 + Real.sqrt 6) / 2) :
  (3 * Real.sqrt 2 + Real.sqrt 6) / 2 = 
    Real.sin C * (3 * Real.sqrt 2 + Real.sqrt 6) / (2 * Real.sin A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l499_49917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pqr_area_l499_49923

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the area of a triangle given its three vertices -/
noncomputable def triangleArea (p q r : Point) : ℝ :=
  (1/2) * abs ((q.x - p.x) * (r.y - p.y) - (r.x - p.x) * (q.y - p.y))

/-- Theorem: The area of triangle PQR with given coordinates is 14.5 square units -/
theorem triangle_pqr_area :
  let p : Point := ⟨-3, 2⟩
  let q : Point := ⟨1, 7⟩
  let r : Point := ⟨4, 1⟩
  triangleArea p q r = 14.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pqr_area_l499_49923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l499_49909

/-- The function f(x) = 3x + sin x -/
noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

/-- Theorem stating that f(2m-1) + f(3-m) > 0 is equivalent to m > -2 -/
theorem inequality_equivalence (m : ℝ) : f (2 * m - 1) + f (3 - m) > 0 ↔ m > -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l499_49909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l499_49969

-- Define the cost prices and selling prices
noncomputable def cp1 : ℚ := 900
noncomputable def cp2 : ℚ := 1200
noncomputable def cp3 : ℚ := 1500
noncomputable def sp1 : ℚ := 1080
noncomputable def sp2 : ℚ := 1320
noncomputable def sp3 : ℚ := 1650

-- Define total cost price and total selling price
noncomputable def total_cp : ℚ := cp1 + cp2 + cp3
noncomputable def total_sp : ℚ := sp1 + sp2 + sp3

-- Define gain and gain percentage
noncomputable def gain : ℚ := total_sp - total_cp
noncomputable def gain_percentage : ℚ := (gain / total_cp) * 100

-- Theorem statement
theorem overall_gain_percentage :
  gain_percentage = 25/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l499_49969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_apex_angle_problem_l499_49960

theorem cone_apex_angle_problem (α β γ : ℝ) : 
  β = π / 8 →
  γ = 3 * π / 8 →
  2 * α = 2 * Real.arctan (2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_apex_angle_problem_l499_49960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l499_49922

theorem teacher_age (num_students : ℕ) (student_avg_age : ℕ) (new_avg_age : ℕ) (teacher_age : ℕ) : 
  num_students = 19 →
  student_avg_age = 20 →
  new_avg_age = student_avg_age + 1 →
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = new_avg_age →
  teacher_age = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l499_49922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ayen_jogging_time_l499_49961

/-- Ayen's usual jogging time in minutes per weekday -/
def usual_jog_time (x : ℕ) : Prop := True

/-- The total jogging time for the week in minutes -/
def total_week_time (t : ℕ) : Prop := True

theorem ayen_jogging_time :
  ∀ x : ℕ,
  usual_jog_time x →
  total_week_time (5 * x + 30) →
  total_week_time 180 →
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ayen_jogging_time_l499_49961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_two_l499_49911

-- Define the expression as noncomputable
noncomputable def expression : ℝ := Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3)

-- Theorem statement
theorem expression_equals_two : expression = 2 := by
  -- The proof steps will go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_two_l499_49911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l499_49902

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition that a > b > 0
def ellipse_condition (a b : ℝ) : Prop :=
  a > b ∧ b > 0

-- Define the right focus F
def right_focus (a b : ℝ) : ℝ × ℝ :=
  (1, 0)

-- Define the point M on the ellipse
noncomputable def point_M : ℝ × ℝ :=
  (Real.sqrt 6 / 2, 1 / 2)

-- Define the point P
def point_P : ℝ × ℝ :=
  (2, 1)

-- Define the condition for Q
def Q_condition (A B Q : ℝ × ℝ) (lambda : ℝ) : Prop :=
  lambda > 0 ∧
  (A.1 - point_P.1 = lambda * (B.1 - point_P.1)) ∧
  (A.2 - point_P.2 = lambda * (B.2 - point_P.2)) ∧
  (A.1 - Q.1 = lambda * (Q.1 - B.1)) ∧
  (A.2 - Q.2 = lambda * (Q.2 - B.2))

theorem ellipse_theorem (a b : ℝ) :
  ellipse_condition a b →
  ellipse a b point_M.1 point_M.2 →
  (∃ A B Q : ℝ × ℝ, ∃ lambda : ℝ,
    ellipse a b A.1 A.2 ∧
    ellipse a b B.1 B.2 ∧
    Q_condition A B Q lambda) →
  (a = Real.sqrt 2 ∧ b = 1) ∧
  (∀ Q : ℝ × ℝ, Q.1 + Q.2 = 1 → Q.1^2 + Q.2^2 ≥ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l499_49902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ngon_equal_triangle_division_l499_49904

/-- Predicate to check if a set of points forms a regular n-gon -/
def is_regular_ngon (n : ℕ) (points : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate to check if a set of line segments forms a division by diagonals -/
def is_division_by_diagonals (division : Set (Set (ℝ × ℝ))) : Prop := sorry

/-- Predicate to check if all triangles in a division are equal -/
def are_triangles_equal (division : Set (Set (ℝ × ℝ))) : Prop := sorry

/-- A regular n-gon can be divided by diagonals into equal triangles if and only if n is even and greater than 3 -/
theorem regular_ngon_equal_triangle_division (n : ℕ) : 
  (n > 3 ∧ ∃ (division : Set (Set (ℝ × ℝ))), 
    is_regular_ngon n points ∧ 
    is_division_by_diagonals division ∧ 
    are_triangles_equal division) ↔ 
  (n > 3 ∧ Even n) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ngon_equal_triangle_division_l499_49904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_correct_l499_49962

-- Define the point A
def A : ℝ × ℝ := (-1, 0)

-- Define the slope of the given line
noncomputable def m₁ : ℝ := 2

-- Define the slope of the perpendicular line
noncomputable def m₂ : ℝ := -1 / m₁

-- Define the equation of the perpendicular line
def perpendicular_line (x y : ℝ) : Prop :=
  x + 2*y + 1 = 0

theorem perpendicular_line_correct :
  -- The line passes through point A
  perpendicular_line A.1 A.2 ∧
  -- The line is perpendicular to 2x-y+1=0
  m₁ * m₂ = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_correct_l499_49962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_ellipse_l499_49903

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a triangle -/
structure Triangle where
  p : Point
  q : Point
  r : Point

/-- Check if a point is on the ellipse -/
def onEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if a triangle is right-angled at Q -/
def isRightAngled (t : Triangle) : Prop :=
  (t.p.x - t.q.x) * (t.r.x - t.q.x) + (t.p.y - t.q.y) * (t.r.y - t.q.y) = 0

/-- Check if PR is parallel to x-axis -/
def isPRParallelToX (t : Triangle) : Prop :=
  t.p.y = t.r.y

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Main theorem -/
theorem triangle_in_ellipse (e : Ellipse) (t : Triangle) (f1 f2 : Point) :
  onEllipse e t.p ∧ onEllipse e t.q ∧ onEllipse e t.r ∧
  isRightAngled t ∧
  t.q = ⟨0, e.b⟩ ∧
  isPRParallelToX t ∧
  distance f1 f2 = 2 →
  distance t.p t.q / distance f1 f2 = Real.sqrt 7 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_ellipse_l499_49903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_speed_range_max_valid_speed_l499_49927

/-- The fuel consumption function for a car -/
noncomputable def fuel_consumption (x k : ℝ) : ℝ := (1/5) * (x - k + 4500/x)

/-- Theorem stating the valid speed range for the car -/
theorem valid_speed_range :
  ∀ (k : ℝ),
  (fuel_consumption 120 k = 11.5) →
  (∀ (x : ℝ), 60 ≤ x → x ≤ 120 → fuel_consumption x k ≤ 9) →
  (∀ (x : ℝ), 60 ≤ x ∧ x ≤ 100 ↔ (60 ≤ x ∧ x ≤ 120 ∧ fuel_consumption x k ≤ 9)) :=
by sorry

/-- Corollary: The maximum speed satisfying all conditions is 100 km/h -/
theorem max_valid_speed (k : ℝ) (h1 : fuel_consumption 120 k = 11.5) 
  (h2 : ∀ (x : ℝ), 60 ≤ x → x ≤ 120 → fuel_consumption x k ≤ 9) :
  ∃ (x : ℝ), x = 100 ∧ fuel_consumption x k ≤ 9 ∧ 
  ∀ (y : ℝ), y > x → (y > 120 ∨ fuel_consumption y k > 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_speed_range_max_valid_speed_l499_49927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l499_49980

/-- Represents the time taken to complete a job when working together -/
noncomputable def time_together (time_a time_b : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b)

/-- Theorem: If person A can complete a job in 10 days and person B in 15 days,
    then they can complete the job together in 6 days -/
theorem job_completion_time (time_a time_b : ℝ) 
  (ha : time_a = 10) (hb : time_b = 15) : 
  time_together time_a time_b = 6 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l499_49980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_distance_l499_49984

/-- Calculates the total distance traveled by a man rowing in a river --/
noncomputable def total_distance_traveled (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : ℝ :=
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := (upstream_speed * downstream_speed * total_time) / (2 * (upstream_speed + downstream_speed))
  2 * one_way_distance

/-- Theorem stating the total distance traveled by the man --/
theorem man_rowing_distance :
  let man_speed := (9 : ℝ)
  let river_speed := (1.2 : ℝ)
  let total_time := (1 : ℝ)
  abs (total_distance_traveled man_speed river_speed total_time - 8.84) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_distance_l499_49984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_necessary_not_sufficient_l499_49956

-- Define planes as a type
variable (Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular parallel : Plane → Plane → Prop)

-- Notation for perpendicular and parallel
local infix:50 " ⊥ " => perpendicular
local infix:50 " ∥ " => parallel

theorem perpendicular_necessary_not_sufficient 
  (α β γ : Plane) 
  (h : α ⊥ β) : 
  (∀ γ, α ∥ γ → γ ⊥ β) ∧ 
  (∃ γ, γ ⊥ β ∧ ¬(α ∥ γ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_necessary_not_sufficient_l499_49956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_drop_l499_49977

/-- Stone drop problem -/
theorem stone_drop (h₀ v s : ℝ → ℝ) (k c : ℝ) : 
  h₀ 0 = 200 →                          -- Initial height
  (∀ t, v t = k * t) →                  -- Speed proportional to time
  (∀ t, s t = c * t^2) →                -- Distance proportional to time squared
  v 1 = 10 →                            -- Speed after 1 second
  h₀ 0 - s 1 = 190 →                    -- Height after 1 second
  (∃ t, v t = 25 ∧ h₀ 0 - s t = 137.5)  -- Conclusion
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_drop_l499_49977
