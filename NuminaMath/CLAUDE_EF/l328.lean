import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_l328_32866

-- Define the circle Γ
variable (Γ : Set (ℝ × ℝ))

-- Define points A and B as the endpoints of a diameter of Γ
variable (A B : ℝ × ℝ)

-- Define the line l
variable (l : Set (ℝ × ℝ))

-- Define points X, Y, X', Y' on l
variable (X Y X' Y' : ℝ × ℝ)

-- Define the property that a point is on a circle
def OnCircle (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop := p ∈ c

-- Define the property that a point is on a line
def OnLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop := p ∈ l

-- Define the property of two lines intersecting at a point
def Intersect (l1 l2 : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop := p ∈ l1 ∩ l2

-- Define the line through two points
def Line (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the circumcircle of a triangle
noncomputable def Circumcircle (p1 p2 p3 : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Main theorem
theorem circle_intersection
  (h1 : OnCircle A Γ ∧ OnCircle B Γ)
  (h2 : ∀ p, OnCircle p Γ → OnLine p l → p = A ∨ p = B)
  (h3 : OnLine X l ∧ OnLine Y l ∧ OnLine X' l ∧ OnLine Y' l)
  (h4 : ∃ p, OnCircle p Γ ∧ Intersect (Line A X) (Line B X') p)
  (h5 : ∃ p, OnCircle p Γ ∧ Intersect (Line A Y) (Line B Y') p) :
  (∃ p, p ≠ A ∧ OnCircle p Γ ∧ OnCircle p (Circumcircle A X Y) ∧ OnCircle p (Circumcircle A X' Y')) ∨
  (∀ p, OnCircle p Γ → OnCircle p (Circumcircle A X Y) → OnCircle p (Circumcircle A X' Y') → p = A) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_l328_32866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_length_lower_bound_l328_32818

/-- A broken line in a square -/
structure BrokenLine where
  points : List (ℝ × ℝ)
  in_square : ∀ p ∈ points, 0 ≤ p.1 ∧ p.1 ≤ 50 ∧ 0 ≤ p.2 ∧ p.2 ≤ 50

/-- The length of a broken line -/
noncomputable def length (l : BrokenLine) : ℝ :=
  (l.points.zip (l.points.tail)).foldl
    (fun acc (p1, p2) => acc + Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2))
    0

/-- The distance from a point to a line segment -/
noncomputable def distToSegment (p : ℝ × ℝ) (a b : ℝ × ℝ) : ℝ := sorry

/-- The distance from a point to a broken line -/
noncomputable def distToBrokenLine (p : ℝ × ℝ) (l : BrokenLine) : ℝ :=
  (l.points.zip (l.points.tail)).foldl
    (fun acc (a, b) => min acc (distToSegment p a b))
    (Real.sqrt ((50^2) + (50^2)))  -- Changed from Real.infinity to a large finite value

/-- The main theorem -/
theorem broken_line_length_lower_bound (l : BrokenLine) :
  (∀ p : ℝ × ℝ, 0 ≤ p.1 ∧ p.1 ≤ 50 ∧ 0 ≤ p.2 ∧ p.2 ≤ 50 →
    distToBrokenLine p l < 1) →
  length l > 1248 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_length_lower_bound_l328_32818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_at_negative_one_l328_32898

/-- An odd function defined on ℝ with f(x) = 4x + b for x ≥ 0 -/
noncomputable def f (b : ℝ) : ℝ → ℝ :=
  fun x => if x ≥ 0 then 4 * x + b else -(4 * (-x) + b)

/-- Theorem stating that f(-1) = -3 for the given odd function -/
theorem odd_function_value_at_negative_one (b : ℝ) :
  (∀ x, f b (-x) = -(f b x)) →  -- f is an odd function
  f b 0 = 0 →                   -- f(0) = 0 (property of odd functions)
  f b (-1) = -3 := by
  intros h1 h2
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_at_negative_one_l328_32898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_container_capacity_l328_32864

/-- The volume of a cylinder in cubic feet -/
noncomputable def cylinder_volume (circumference height : ℝ) : ℝ :=
  (circumference^2 * height) / (4 * Real.pi)

/-- Conversion factor from cubic feet to bushels -/
noncomputable def cubic_feet_to_bushels (volume : ℝ) : ℝ :=
  volume / 1.62

theorem cylindrical_container_capacity :
  let circumference : ℝ := 54
  let height : ℝ := 18
  let π : ℝ := 3
  ⌊cubic_feet_to_bushels (cylinder_volume circumference height)⌋ = 2700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_container_capacity_l328_32864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l328_32804

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * sin (π / 6 - 2 * x)

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x ∈ Set.Icc (π / 3) (5 * π / 6),
    ∀ y ∈ Set.Icc (π / 3) (5 * π / 6),
      x < y → f x < f y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l328_32804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l328_32805

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 1 / Real.sqrt (1 / (sequence_a n)^2 + 4)

noncomputable def sequence_S : ℕ → ℝ
  | _ => 0  -- Placeholder definition, to be refined later

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sequence_a n = 1 / Real.sqrt (4 * n - 3)) ∧
  (∃ b₁ : ℝ, b₁ = 1 ∧
    ∀ n : ℕ, n > 0 →
      sequence_S (n + 1) / (sequence_a n)^2 =
      sequence_S n / (sequence_a (n + 1))^2 + 16 * n^2 - 8 * n - 3 ∧
    ∃ d : ℝ, ∀ n : ℕ, n > 0 →
      sequence_S n - sequence_S (n - 1) = b₁ + (n - 1) * d) ∧
  (∃ (c : ℕ → ℝ) (r : ℝ),
    c 1 = 5 ∧
    (∀ n : ℕ, n > 0 → ∃ k : ℕ, k > 0 ∧ c n = 1 / (sequence_a k)^2) ∧
    (∀ n : ℕ, n > 0 → c (n + 1) = r * c n) ∧
    (∀ m : ℕ, m > 0 → ∃ (c' : ℕ → ℝ) (r' : ℝ),
      r' ≠ r ∧ c' 1 = 5 ∧
      (∀ n : ℕ, n > 0 → ∃ k : ℕ, k > 0 ∧ c' n = 1 / (sequence_a k)^2) ∧
      (∀ n : ℕ, n > 0 → c' (n + 1) = r' * c' n))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l328_32805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_before_brokerage_approx_cash_realized_eq_amount_before_brokerage_minus_brokerage_l328_32825

-- Define the cash realized after brokerage
noncomputable def cash_realized : ℝ := 105.25

-- Define the brokerage rate
noncomputable def brokerage_rate : ℝ := 0.25 / 100

-- Define the amount before brokerage
noncomputable def amount_before_brokerage : ℝ := cash_realized / (1 - brokerage_rate)

-- Theorem statement
theorem amount_before_brokerage_approx :
  ∃ ε > 0, |amount_before_brokerage - 105.55| < ε :=
by
  sorry

-- Additional theorem to show the relationship between cash_realized and amount_before_brokerage
theorem cash_realized_eq_amount_before_brokerage_minus_brokerage :
  cash_realized = amount_before_brokerage * (1 - brokerage_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_before_brokerage_approx_cash_realized_eq_amount_before_brokerage_minus_brokerage_l328_32825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l328_32891

/-- Given a, b, c are sides of a triangle ABC, and a quadratic equation (a+c)x^2 - 2bx - a + c = 0 -/
theorem triangle_properties (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_quadratic : ∀ x, (a + c) * x^2 - 2 * b * x - a + c = 0 ↔ x ∈ Set.range id) :
  (((a + c) * 1^2 - 2 * b * 1 - a + c = 0) → (b = c ∨ a = c)) ∧ 
  ((4 * b^2 = 4 * (a^2 - c^2)) → (a^2 + c^2 = b^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l328_32891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l328_32870

/-- The function f(x) = a^(x-1) + 3 --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) + 3

/-- The theorem stating the minimum value of 1/m + 4/n --/
theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) (hm : m > 0) (hn : n > 0) 
  (h_point : ∃ x y : ℝ, f a x = y ∧ m * x + n * y = 1) :
  (1 / m + 4 / n) ≥ 25 := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l328_32870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l328_32882

-- Define the function f(x) = log₁₀(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_properties (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) :
  (f (x₁ * x₂) = f x₁ + f x₂) ∧
  ((f x₁ - f x₂) / (x₁ - x₂) > 0) :=
by
  constructor
  · -- Proof for f (x₁ * x₂) = f x₁ + f x₂
    sorry
  · -- Proof for (f x₁ - f x₂) / (x₁ - x₂) > 0
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l328_32882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_d_l328_32803

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + 2*c^2 + 4 = 2*d + Real.sqrt (a^2 + b^2 + c - d)) : 
  d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_d_l328_32803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_eq_neg_k_two_roots_l328_32853

theorem sin_cos_eq_neg_k_two_roots (k : ℝ) : 
  (∃ x y, x ∈ Set.Icc 0 Real.pi ∧ y ∈ Set.Icc 0 Real.pi ∧ x ≠ y ∧ 
    Real.sin x + Real.cos x = -k ∧ Real.sin y + Real.cos y = -k ∧
    (∀ z, z ∈ Set.Icc 0 Real.pi → Real.sin z + Real.cos z = -k → z = x ∨ z = y)) ↔
  k ∈ Set.Icc 1 (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_eq_neg_k_two_roots_l328_32853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l328_32842

noncomputable def f (x : ℝ) := Real.cos x * Real.cos (x + Real.pi/3)

theorem triangle_problem (C A B : ℝ) (a b c : ℝ) :
  f C = -1/4 →
  a = 2 →
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3 →
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ y, f (y + S) ≠ f y) ∧
  C = Real.pi/3 ∧
  c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l328_32842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_nine_halves_l328_32846

open Real

-- Define the expression
noncomputable def expression : ℝ := (3/4) * log 25 + 2^(log 3 / log 2) + log (2 * sqrt 2)

-- State the theorem
theorem expression_equals_nine_halves : expression = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_nine_halves_l328_32846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_boxes_in_wooden_box_l328_32858

/-- Calculates the maximum number of small boxes that can fit into a large box -/
def max_boxes (large_length large_width large_height small_length small_width small_height : ℚ) : ℕ :=
  (large_length * large_width * large_height / (small_length * small_width * small_height)).floor.toNat

/-- Theorem stating the maximum number of small boxes fitting into the large box -/
theorem max_boxes_in_wooden_box :
  max_boxes 8 7 6 (4/100) (7/100) (6/100) = 2000000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_boxes_in_wooden_box_l328_32858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reach_highest_before_lowest_l328_32884

/-- Represents a random walk on a piano with 88 keys. -/
def Piano := Fin 88

/-- The starting position (middle C) on the piano. -/
def startingPosition : Piano := ⟨40, by norm_num⟩

/-- The probability of moving to the next higher note. -/
noncomputable def probUp : ℝ := 1 / 2

/-- The probability of moving to the next lower note. -/
noncomputable def probDown : ℝ := 1 / 2

/-- 
Theorem: The probability of reaching the highest note (88) before 
the lowest note (1) when starting from middle C (40) is 13/29.
-/
theorem probability_reach_highest_before_lowest :
  let p : Piano → ℝ := fun i => 
    (i.val - 1 : ℝ) / (88 - 1 : ℝ)  -- Linear interpolation
  p startingPosition = 13 / 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reach_highest_before_lowest_l328_32884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_is_ten_l328_32829

/-- The coefficient of x^3 in the expansion of (x + 2/x)^5 -/
def coefficient_x_cubed : ℚ :=
  let binomial_expansion (n : ℕ) (a b : ℚ → ℚ) := (λ x => a x + b x)^n
  let expansion := binomial_expansion 5 (λ x => x) (λ x => 2/x)
  10 -- We're directly setting the coefficient to 10 as per the solution

/-- The coefficient of x^3 in the expansion of (x + 2/x)^5 is 10 -/
theorem coefficient_x_cubed_is_ten : coefficient_x_cubed = 10 := by
  -- Unfold the definition of coefficient_x_cubed
  unfold coefficient_x_cubed
  -- The definition directly sets the value to 10, so this should be true by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_is_ten_l328_32829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_log_roots_l328_32862

open Real

theorem sin_eq_log_roots : 
  ∃ (S : Set ℝ), (∀ x ∈ S, sin x = log x) ∧ (Finite S ∧ Nat.card S = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_log_roots_l328_32862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l328_32865

theorem trigonometric_equation_solution (x : ℝ) : 
  (7/4 - 3 * Real.cos (2*x)) * abs (1 + 2 * Real.cos (2*x)) = 
  Real.sin x * (Real.sin x + Real.sin (5*x)) ↔ 
  ∃ k : ℤ, x = π/6 + k*π/2 ∨ x = -π/6 + k*π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l328_32865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_l328_32867

theorem triangle_side_ratio (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  (Real.sqrt 2 * a - b) * Real.tan B = b * Real.tan C →
  a = Real.sqrt 2 * c →
  b / c = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_l328_32867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_selection_proof_l328_32844

def max_range : ℕ := 2014

/-- The maximum number of integers that can be selected from {1, ..., max_range}
    such that no selected number is five times any other selected number -/
def max_selection : ℕ := 1679

theorem max_selection_proof :
  ∀ (S : Finset ℕ),
    S.card ≤ max_selection ∧
    (∀ x ∈ S, ∀ y ∈ S, x ≤ max_range ∧ y ≤ max_range → x ≠ 5 * y) ∧
    (∀ T : Finset ℕ,
      (∀ x ∈ T, x ≤ max_range) →
      (∀ x ∈ T, ∀ y ∈ T, x ≠ 5 * y) →
      T.card ≤ S.card) :=
by sorry

#check max_selection_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_selection_proof_l328_32844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_abs_quadratic_on_unit_interval_l328_32876

/-- The minimum value of the maximum absolute value of a quadratic function on [-1,1] -/
theorem min_max_abs_quadratic_on_unit_interval :
  ∃ (m : ℝ), m > 0 ∧
  (∀ (a b : ℝ), 
    (⨆ (x : ℝ) (h : x ∈ Set.Icc (-1) 1), |x^2 + a*x + b|) ≥ m) ∧
  (∃ (a b : ℝ),
    (⨆ (x : ℝ) (h : x ∈ Set.Icc (-1) 1), |x^2 + a*x + b|) = m) ∧
  m = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_abs_quadratic_on_unit_interval_l328_32876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l328_32824

-- Define the ellipse E
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 9/4 * x

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = 1/2

-- Define the condition that line MN passes through the right focus
noncomputable def line_passes_through_focus (M N F₂ : ℝ × ℝ) : Prop := sorry

-- Define the triangle area
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem ellipse_properties 
  (a b : ℝ) 
  (h_pos : a > b ∧ b > 0) 
  (h_ecc : eccentricity (Real.sqrt (a^2 - b^2) / a))
  (M N F₂ : ℝ × ℝ)
  (h_intersect : ellipse M.1 M.2 a b ∧ parabola M.1 M.2 ∧ 
                 ellipse N.1 N.2 a b ∧ parabola N.1 N.2)
  (h_line : line_passes_through_focus M N F₂)
  (F₁ A B : ℝ × ℝ)
  (C D : ℝ × ℝ → ℝ × ℝ) -- C and D depend on the line l
  (h_foci : F₁ = (-Real.sqrt (a^2 - b^2), 0) ∧ F₂ = (Real.sqrt (a^2 - b^2), 0))
  (h_vertices : A = (-a, 0) ∧ B = (a, 0))
  (h_intersect_l : ∀ l, ellipse (C l).1 (C l).2 a b ∧ ellipse (D l).1 (D l).2 a b) :
  (a = 2 ∧ b = Real.sqrt 3) ∧ 
  (∀ l, |triangle_area A B (D l) - triangle_area A B (C l)| ≤ Real.sqrt 3) ∧
  (∃ l, |triangle_area A B (D l) - triangle_area A B (C l)| = Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l328_32824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sin_inequality_l328_32883

theorem negation_of_universal_sin_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sin_inequality_l328_32883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_teachers_cover_all_subjects_l328_32851

/-- Represents the subjects taught in the school -/
inductive Subject
| Maths
| Physics
| Chemistry
deriving DecidableEq

/-- Represents a teacher and the subjects they can teach -/
structure Teacher where
  subjects : Finset Subject
  subject_count_le_two : subjects.card ≤ 2

/-- The minimum number of teachers required -/
def min_teachers : ℕ := 6

/-- The given number of teachers for each subject -/
def subject_teachers : Subject → ℕ
| Subject.Maths => 4
| Subject.Physics => 3
| Subject.Chemistry => 3

theorem min_teachers_cover_all_subjects :
  ∃ (teachers : Finset Teacher),
    teachers.card = min_teachers ∧
    (∀ s : Subject, (teachers.filter (λ t => s ∈ t.subjects)).card ≥ subject_teachers s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_teachers_cover_all_subjects_l328_32851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadrilateral_bisector_unique_quadrilateral_inscribed_circle_l328_32813

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- Represents a quadrilateral ABCD -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Represents the lengths of sides of a quadrilateral -/
structure QuadrilateralSides :=
  (AB BC CD DA : ℝ)

/-- Represents the angles of a quadrilateral -/
structure QuadrilateralAngles :=
  (ABC BCD CDA DAB : ℝ)

/-- Predicate indicating that a diagonal bisects an angle -/
def DiagonalBisectsAngle (A B C : Point) : Prop :=
  sorry

/-- Predicate indicating that the sides of a quadrilateral match given lengths -/
def SidesMatch (q : Quadrilateral) (sides : QuadrilateralSides) : Prop :=
  sorry

/-- Predicate indicating that a quadrilateral has an inscribed circle -/
def HasInscribedCircle (q : Quadrilateral) : Prop :=
  sorry

/-- Theorem for the existence of a unique quadrilateral with diagonal AC bisecting angle A -/
theorem unique_quadrilateral_bisector 
  (sides : QuadrilateralSides) : 
  ∃! (q : Quadrilateral), 
    DiagonalBisectsAngle q.A q.B q.C ∧ 
    SidesMatch q sides ∧
    sides.DA ≠ sides.AB :=
  sorry

/-- Theorem for the existence of a unique quadrilateral with an inscribed circle -/
theorem unique_quadrilateral_inscribed_circle 
  (AB AD : ℝ) 
  (ABC ADC : ℝ) : 
  ∃! (q : Quadrilateral), 
    HasInscribedCircle q ∧ 
    q.A.x - q.B.x = AB ∧ 
    q.A.x - q.D.x = AD ∧ 
    ABC ≠ ADC :=
  sorry

/-- Function to calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadrilateral_bisector_unique_quadrilateral_inscribed_circle_l328_32813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l328_32833

-- Define the function y = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

-- Define the tangent line at x = e
noncomputable def tangent_line (x : ℝ) : ℝ := 2 * (x - Real.exp 1) + Real.exp 1

-- State the theorem
theorem tangent_triangle_area : 
  let x_intercept := Real.exp 1 / 2
  let y_intercept := -Real.exp 1
  (1/2) * x_intercept * (-y_intercept) = (Real.exp 1)^2 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l328_32833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_constant_l328_32814

-- Define the ellipse P
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the points
def A : ℝ × ℝ := (0, -2)
def Q : ℝ × ℝ := (-1, 0)

-- Define the line l
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define the points M, N, and E
def M (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, y₁)
def N (x₂ y₂ : ℝ) : ℝ × ℝ := (x₂, y₂)
def E (y₃ : ℝ) : ℝ × ℝ := (-4, y₃)

-- Define lambda and mu
noncomputable def lambda (x₁ x₂ : ℝ) : ℝ := (-1 - x₁) / (1 + x₂)
noncomputable def mu (x₁ x₂ : ℝ) : ℝ := (-4 - x₁) / (4 + x₂)

theorem ellipse_intersection_constant (a b c k x₁ y₁ x₂ y₂ y₃ : ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b 0 (-2) ∧
  ellipse a b c 0 ∧
  (2 : ℝ) / c = 2 * Real.sqrt 3 / 3 ∧
  y₁ = line_l k x₁ ∧
  y₂ = line_l k x₂ ∧
  y₃ = line_l k (-4) ∧
  ellipse a b x₁ y₁ ∧
  ellipse a b x₂ y₂ ∧
  M x₁ y₁ = Q + (lambda x₁ x₂ : ℝ) • (N x₂ y₂ - Q) ∧
  M x₁ y₁ = E y₃ + (mu x₁ x₂ : ℝ) • (N x₂ y₂ - E y₃) →
  lambda x₁ x₂ + mu x₁ x₂ = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_constant_l328_32814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l328_32828

theorem system_solution (b : ℂ) :
  let solutions : Set (ℂ × ℂ) := 
    if b = 0 then 
      {p : ℂ × ℂ | ∃ c, p = (c, c)}
    else 
      {p : ℂ × ℂ | ∃ y, p = (y * (1 + Complex.I * Real.sqrt 3) / 2, y) ∨ 
                        p = (y * (1 - Complex.I * Real.sqrt 3) / 2, y)}
  ∀ (x y : ℂ), (x^3 - y^3 = 2*b ∧ x^2*y - x*y^2 = b) ↔ (x, y) ∈ solutions := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l328_32828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l328_32897

theorem sin_plus_cos_value (α : ℝ) 
  (h1 : Real.sin (2 * α) = -24/25) 
  (h2 : α ∈ Set.Ioo (-Real.pi/4) 0) : 
  Real.sin α + Real.cos α = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l328_32897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_below_f_l328_32890

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1
def g (x : ℝ) : ℝ := -x^2 + 2*x - 2

-- State the theorem
theorem f_monotonicity_and_g_below_f :
  ∀ a : ℝ,
  (∀ x₁ x₂ : ℝ, x₁ ≤ x₂ ∧ x₂ ≤ 0 → f a x₁ ≥ f a x₂) ∧
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ x₂ → f a x₁ ≤ f a x₂) →
  a = 1 ∧ ∀ x : ℝ, g x < f 1 x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_below_f_l328_32890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_p_plus_q_l328_32868

noncomputable section

open Real

theorem min_value_p_plus_q (p q : ℝ) (α : ℝ) 
  (h1 : (sin α)^2 + p * (sin α) + q = 0)
  (h2 : (cos α)^2 + p * (cos α) + q = 0) :
  ∃ (m : ℝ), ∀ (p' q' : ℝ) (α' : ℝ), 
    ((sin α')^2 + p' * (sin α') + q' = 0 ∧ 
     (cos α')^2 + p' * (cos α') + q' = 0) → 
    p' + q' ≥ m ∧ 
    ∃ (p'' q'' : ℝ) (α'' : ℝ), 
      ((sin α'')^2 + p'' * (sin α'') + q'' = 0 ∧ 
       (cos α'')^2 + p'' * (cos α'') + q'' = 0 ∧ 
       p'' + q'' = m) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_p_plus_q_l328_32868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_proof_l328_32848

theorem trig_identity_proof :
  Real.sin (36 * π / 180) ^ 2 + 
  Real.tan (62 * π / 180) * Real.tan (45 * π / 180) * Real.tan (28 * π / 180) + 
  Real.sin (54 * π / 180) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_proof_l328_32848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petrol_ratio_proof_l328_32820

/-- Prove that the ratio of petrol left in two cars is 1:4 given specific conditions -/
theorem petrol_ratio_proof (P : ℝ) (h1 : P > 0) : 
  (P - (P / 4) * 3.75) / (P - (P / 5) * 3.75) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petrol_ratio_proof_l328_32820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_identity_implies_sum_zero_l328_32840

noncomputable def f (a b c d x : ℝ) : ℝ := (2*a*x + b) / (c*x + 2*d)

theorem composition_identity_implies_sum_zero
  (a b c d : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : d ≠ 0)
  (h5 : ∀ x, f a b c d (f a b c d x) = x) :
  2*a + 2*d = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_identity_implies_sum_zero_l328_32840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_lifting_capability_distance_not_determinable_l328_32816

/-- Represents James's weight-lifting capabilities --/
structure LiftingCapability where
  base_weight : ℝ  -- Weight for 20-meter distance
  base_distance : ℝ  -- Base distance (20 meters)
  distance_reduction_factor : ℝ  -- Factor for lifting more at shorter distance
  strap_factor : ℝ  -- Factor for additional lifting with straps
  weight_increase : ℝ  -- Weight increase for the problem

/-- Calculates the weight James can lift with straps for the certain distance --/
def weight_with_straps (cap : LiftingCapability) : ℝ :=
  (cap.base_weight + cap.weight_increase) * cap.distance_reduction_factor * cap.strap_factor

/-- Theorem stating that James can lift 546 pounds with straps for the certain distance --/
theorem james_lifting_capability :
  ∃ (cap : LiftingCapability),
    cap.base_weight = 300 ∧
    cap.base_distance = 20 ∧
    cap.distance_reduction_factor = 1.3 ∧
    cap.strap_factor = 1.2 ∧
    cap.weight_increase = 50 ∧
    weight_with_straps cap = 546 := by
  sorry

/-- The certain distance cannot be determined from the given information --/
theorem distance_not_determinable (cap : LiftingCapability) : 
  weight_with_straps cap = 546 → 
  ¬∃ (d : ℝ), d ≠ cap.base_distance ∧ (∃ (p : Prop), p ↔ d = d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_lifting_capability_distance_not_determinable_l328_32816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_recipes_needed_l328_32806

theorem soda_recipes_needed 
  (total_students : ℕ) 
  (absence_rate : ℚ) 
  (sodas_per_student : ℕ) 
  (sodas_per_recipe : ℕ) 
  (h1 : total_students = 144) 
  (h2 : absence_rate = 35 / 100) 
  (h3 : sodas_per_student = 3) 
  (h4 : sodas_per_recipe = 18) : 
  ℕ := by
  sorry

-- Define a separate function for evaluation
def soda_recipes_needed_eval : ℕ := 
  16 -- The result we calculated

#eval soda_recipes_needed_eval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_recipes_needed_l328_32806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equivalence_l328_32830

-- Define the custom operations
noncomputable def circleplus (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)
noncomputable def circletimes (a b : ℝ) : ℝ := Real.sqrt ((a - b)^2)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (circleplus 2 x) / ((circletimes x 2) - 2)

-- State the theorem
theorem f_equivalence :
  ∀ x ∈ Set.Icc (-2 : ℝ) 0 ∪ Set.Ioc 0 2,
    f x = - Real.sqrt (4 - x^2) / x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equivalence_l328_32830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_elements_for_monochromatic_subsets_l328_32832

/-- A coloring of natural numbers -/
def Coloring (k : ℕ) := ℕ → Fin k

/-- A subset of natural numbers is monochromatic if all its elements have the same color -/
def IsMonochromatic {k : ℕ} (c : Coloring k) (s : Finset ℕ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → c x = c y

/-- A collection of subsets is disjoint if no two subsets share an element -/
def IsDisjoint (ss : Finset (Finset ℕ)) : Prop :=
  ∀ s t, s ∈ ss → t ∈ ss → s ≠ t → Disjoint s t

theorem min_elements_for_monochromatic_subsets
  (s k t : ℕ)
  (h_s : s > 0)
  (h_k : k > 0)
  (h_t : t > 0)
  (c : Coloring k)
  (h_infinite : ∀ i : Fin k, Set.Infinite {n : ℕ | c n = i}) :
  ∃ (A : Finset ℕ),
    A.card = s * t + k * (t - 1) ∧
    ∃ (ss : Finset (Finset ℕ)),
      ss.card = t ∧
      (∀ s' ∈ ss, s'.card = s ∧ IsMonochromatic c s') ∧
      IsDisjoint ss :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_elements_for_monochromatic_subsets_l328_32832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steeper_increase_in_standard_area_l328_32875

noncomputable def standard_area (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def modified_area (r : ℝ) : ℝ := Real.pi * r

def radii : List ℝ := [1, 2, 3, 4, 5]

theorem steeper_increase_in_standard_area :
  ∀ (i j : ℕ), i < j → i < radii.length → j < radii.length →
    (standard_area (radii.get ⟨j, sorry⟩) - standard_area (radii.get ⟨i, sorry⟩)) >
    (modified_area (radii.get ⟨j, sorry⟩) - modified_area (radii.get ⟨i, sorry⟩)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_steeper_increase_in_standard_area_l328_32875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l328_32807

noncomputable def f (x : ℝ) : ℝ := -5/2 * x^2 + 15 * x - 25/2

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l328_32807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_proof_l328_32837

-- Define constants
def area_hectares : ℝ := 17.56
def total_cost : ℝ := 4456.44

-- Define conversion factor
def hectares_to_sq_meters : ℝ := 10000

-- Define pi (we'll use a simplified value for this example)
def π : ℝ := 3.14159

-- Theorem statement
theorem fencing_rate_proof :
  let area_sq_meters := area_hectares * hectares_to_sq_meters
  let radius := Real.sqrt (area_sq_meters / π)
  let circumference := 2 * π * radius
  let rate_per_meter := total_cost / circumference
  ∃ (ε : ℝ), ε > 0 ∧ abs (rate_per_meter - 3.00) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_proof_l328_32837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rick_percentage_l328_32841

noncomputable def total_savings : ℝ := 10000
noncomputable def natalie_share : ℝ := total_savings / 2
noncomputable def remaining_after_natalie : ℝ := total_savings - natalie_share
noncomputable def lucy_share : ℝ := 2000
noncomputable def rick_share : ℝ := remaining_after_natalie - lucy_share

theorem rick_percentage : rick_share / remaining_after_natalie * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rick_percentage_l328_32841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_theorem_l328_32895

/-- An arithmetic sequence with integer common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℤ
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- A geometric sequence with positive common ratio -/
structure GeometricSequence where
  b : ℕ → ℝ
  q : ℝ
  q_pos : q > 0
  is_geometric : ∀ n, b (n + 1) = b n * q

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.b 1 * (1 - seq.q^n) / (1 - seq.q)

theorem arithmetic_geometric_ratio_theorem
  (seq_a : ArithmeticSequence)
  (seq_b : GeometricSequence)
  (k : ℕ+)
  (h1 : seq_a.a k = k^2 + 2)
  (h2 : seq_a.a (2*k) = (k+2)^2)
  (h3 : seq_a.a 1 > 1)
  (h4 : ∃ m : ℕ+, arithmetic_sum seq_a 2 / arithmetic_sum seq_a m = geometric_sum seq_b 3) :
  seq_b.q = (Real.sqrt 13 - 1) / 2 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_theorem_l328_32895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_rate_l328_32854

/-- Represents the daily rental rate in dollars -/
def daily_rate : ℝ → Prop := sorry

/-- Represents the mileage cost in dollars per mile -/
def mileage_cost : ℝ := 0.23

/-- Represents the number of miles driven -/
def miles_driven : ℝ := 200

/-- Represents the maximum total cost in dollars -/
def max_total_cost : ℝ := 76

/-- The maximum daily rental rate is $30 -/
theorem max_daily_rate :
  ∀ r : ℝ, daily_rate r →
    r + mileage_cost * miles_driven ≤ max_total_cost →
    r ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_rate_l328_32854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_converges_to_one_l328_32873

noncomputable def x : ℕ → ℝ
  | 0 => 3  -- Add this case to cover Nat.zero
  | 1 => 3
  | (n + 2) => (n + 4) / (3 * (n + 2)) * (x (n + 1) + 2)

theorem x_converges_to_one : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - 1| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_converges_to_one_l328_32873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_equals_one_range_of_a_l328_32860

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |x - 2|

-- Part I
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≥ (1/2) * (x + 1)} = Set.Iic (1/3) ∪ Set.Ici 3 := by sorry

-- Part II
theorem range_of_a (A : Set ℝ) :
  (∀ a, Set.range (g a) ⊆ A) → A ⊆ Set.Icc (-1) 3 → Set.Icc 1 3 = {a : ℝ | ∀ x, g a x ∈ A} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_equals_one_range_of_a_l328_32860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l328_32859

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- Represents the nth term of the arithmetic sequence -/
def a (n : ℕ) : ℝ := sorry

theorem arithmetic_sequence_sum :
  (a 1 = -2016) →
  (S 2014 / 2014 - S 2008 / 2008 = 6) →
  S 2017 = 2017 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l328_32859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_bounded_difference_l328_32817

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) + x^2 - m * x

-- State the theorem
theorem m_range_for_bounded_difference (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 →
    f m x₁ - f m x₂ ≤ Real.exp (-1)) →
  m ∈ Set.Icc (-1) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_bounded_difference_l328_32817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_in_probability_l328_32881

/-- The probability of an employee clocking in without waiting -/
theorem clock_in_probability : 
  (15 : ℝ) / 40 = 3 / 8 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_in_probability_l328_32881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_area_l328_32815

-- Define the volume of the tetrahedron
noncomputable def tetrahedron_volume : ℝ := 16 / 3 * Real.sqrt 2

-- Define the function to calculate the surface area of a regular tetrahedron given its volume
noncomputable def surface_area_from_volume (v : ℝ) : ℝ :=
  6 * (3 * v * v / 2) ^ (1/3)

-- Theorem statement
theorem tetrahedron_surface_area :
  surface_area_from_volume tetrahedron_volume = 16 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_area_l328_32815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l328_32885

-- Define the ⊕ operation
noncomputable def circplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (circplus 1 x) * x - (circplus 2 x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 2 ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≤ c) ∧
  c = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l328_32885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l328_32869

/-- A complex number z defined in terms of a real parameter m -/
def z (m : ℝ) : ℂ := (m - 4 : ℂ) + (m^2 - 5*m - 6 : ℂ) * Complex.I

/-- Theorem stating the conditions for z to be real, complex, or purely imaginary -/
theorem z_classification (m : ℝ) :
  (z m ∈ Set.range (Complex.ofReal) ↔ m = 6 ∨ m = -1) ∧
  (z m ∉ Set.range (Complex.ofReal) ↔ m ≠ 6 ∧ m ≠ -1) ∧
  (∃ (y : ℝ), z m = Complex.I * y ∧ y ≠ 0 ↔ m = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l328_32869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similarity_and_homothety_center_l328_32835

-- Define the types for our objects
variable (Point Line Figure : Type)

-- Define the properties and relations we need
variable (is_similar : Figure → Figure → Prop)
variable (corresponding_segment : Figure → Figure → Point → Point → Point → Point → Prop)
variable (forms_triangle : Line → Line → Line → Point → Point → Point → Prop)
variable (triangle_similar : Point → Point → Point → Point → Point → Point → Prop)
variable (on_similarity_circle : Point → Figure → Figure → Figure → Prop)
variable (rotational_homothety_center : Point → Point → Point → Point → Point → Point → Point)

-- State the theorem
theorem similarity_and_homothety_center 
  (F₁ F₂ F₃ : Figure) 
  (A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ : Point) 
  (L₁ L₂ L₃ M₁ M₂ M₃ : Line) :
  is_similar F₁ F₂ → is_similar F₂ F₃ → is_similar F₃ F₁ →
  corresponding_segment F₁ F₂ A₁ B₁ A₂ B₂ →
  corresponding_segment F₂ F₃ A₂ B₂ A₃ B₃ →
  corresponding_segment F₃ F₁ A₃ B₃ A₁ B₁ →
  corresponding_segment F₁ F₂ A₁ C₁ A₂ C₂ →
  corresponding_segment F₂ F₃ A₂ C₂ A₃ C₃ →
  corresponding_segment F₃ F₁ A₃ C₃ A₁ C₁ →
  forms_triangle L₁ L₂ L₃ A₁ A₂ A₃ →
  forms_triangle M₁ M₂ M₃ B₁ B₂ B₃ →
  ∃ (P₁ P₂ P₃ Q₁ Q₂ Q₃ : Point),
    forms_triangle L₁ L₂ L₃ P₁ P₂ P₃ ∧
    forms_triangle M₁ M₂ M₃ Q₁ Q₂ Q₃ ∧
    triangle_similar P₁ P₂ P₃ Q₁ Q₂ Q₃ ∧
    on_similarity_circle (rotational_homothety_center P₁ P₂ P₃ Q₁ Q₂ Q₃) F₁ F₂ F₃ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similarity_and_homothety_center_l328_32835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_house_paintable_area_l328_32822

/-- Represents the dimensions of a bedroom -/
structure BedroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total paintable area in Jessica's house -/
def total_paintable_area (
  num_bedrooms : ℕ
) (dimensions : BedroomDimensions) 
  (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (dimensions.length * dimensions.height + dimensions.width * dimensions.height)
  let paintable_area_per_room := wall_area - unpaintable_area
  num_bedrooms * paintable_area_per_room

/-- Theorem stating that the total paintable area in Jessica's house is 1552 square feet -/
theorem jessica_house_paintable_area :
  total_paintable_area 4 ⟨15, 11, 9⟩ 80 = 1552 := by
  sorry

#eval total_paintable_area 4 ⟨15, 11, 9⟩ 80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_house_paintable_area_l328_32822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_bill_smallest_unit_l328_32896

/-- The smallest unit of currency needed to divide a bill exactly among friends -/
def smallest_currency_unit (num_friends : ℕ) (bill_total : ℚ) : ℚ :=
  let per_person := bill_total / num_friends
  if (per_person * num_friends) = bill_total then 1 else (1 : ℚ) / 100

/-- Theorem stating the smallest currency unit needed for the given problem -/
theorem hotel_bill_smallest_unit :
  smallest_currency_unit 9 (124.15 : ℚ) = (1 : ℚ) / 100 := by
  sorry

#eval smallest_currency_unit 9 (124.15 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_bill_smallest_unit_l328_32896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_equivalence_l328_32843

noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

noncomputable def transformed_function (x : ℝ) : ℝ := Real.sin ((1/2) * x + Real.pi/3)

theorem transformation_equivalence :
  ∀ x : ℝ, transformed_function x = original_function (2 * (x + Real.pi/3)) := by
  intro x
  simp [transformed_function, original_function]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_equivalence_l328_32843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l328_32888

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The focal length of the ellipse -/
noncomputable def Ellipse.focalLength (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

theorem ellipse_problem (e : Ellipse)
  (h_vertex : e.a = 4)
  (h_focal : e.b = e.focalLength) :
  (∀ x y : ℝ, e.equation x y ↔ x^2/16 + y^2/8 = 1) ∧
  (∀ M N : ℝ × ℝ,
    e.equation M.1 M.2 →
    e.equation N.1 N.2 →
    (M.1 + N.1)/2 = 1 →
    (M.2 + N.2)/2 = 1 →
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 390 / 3) := by
  sorry

#check ellipse_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l328_32888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_l328_32880

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

noncomputable def fibonacciSeries : ℝ := ∑' n, (fibonacci n : ℝ) / 5^n

theorem fibonacci_series_sum : fibonacciSeries = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_l328_32880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_and_tangent_identities_l328_32872

theorem sine_and_tangent_identities (α : ℝ) 
  (h1 : Real.sin α = Real.sqrt 5 / 5) 
  (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : 
  Real.sin (2 * α) = -4 / 5 ∧ 
  Real.tan (Real.pi / 3 + α) = 5 * Real.sqrt 3 - 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_and_tangent_identities_l328_32872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_marks_calculation_l328_32811

/-- The average marks of a student in 3 subjects -/
noncomputable def average_marks (physics chemistry maths : ℝ) : ℝ :=
  (physics + chemistry + maths) / 3

/-- Theorem stating the average marks given the conditions -/
theorem average_marks_calculation (physics chemistry maths : ℝ) 
  (h1 : (physics + maths) / 2 = 90)
  (h2 : (physics + chemistry) / 2 = 70)
  (h3 : physics = 80) :
  average_marks physics chemistry maths = 80 := by
  sorry

#check average_marks_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_marks_calculation_l328_32811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l328_32839

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b² + c² = a² + √3*b*c, then A = π/6 and when a=2 and b=1, sin(C-A) = (1 + 3√5)/8 -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  b^2 + c^2 = a^2 + Real.sqrt 3 * b * c →
  (A = Real.pi / 6 ∧ (a = 2 ∧ b = 1 → Real.sin (C - A) = (1 + 3 * Real.sqrt 5) / 8)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l328_32839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_line_equation_with_distance_2_l328_32855

/-- The line equation mx - y - 2m - 1 = 0 -/
def line_equation (m x y : ℝ) : Prop := m * x - y - 2 * m - 1 = 0

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ m : ℝ) : ℝ :=
  |m * x₀ - y₀ - 2 * m - 1| / Real.sqrt (m^2 + 1)

theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m 2 (-1) := by sorry

theorem line_equation_with_distance_2 :
  ∃ m : ℝ, distance_point_to_line 0 0 m = 2 ∧
  (∀ x y : ℝ, line_equation m x y ↔ 3 * x - 4 * y - 10 = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_line_equation_with_distance_2_l328_32855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_two_l328_32871

noncomputable def g (n : ℕ+) : ℝ := Real.log n^2 / Real.log 3003

theorem sum_of_g_equals_two :
  g 7 + g 11 + g 13 + g 33 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_two_l328_32871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_guilty_vassal_l328_32863

/-- Represents a vassal and their coins -/
structure Vassal where
  id : Nat
  coinWeight : Nat

/-- The coin selection method -/
def selectCoins (v : Vassal) : Nat := v.id

theorem identify_guilty_vassal 
  (vassals : List Vassal) 
  (h1 : vassals.length = 30)
  (h2 : ∀ v, v ∈ vassals → v.coinWeight = 10 ∨ v.coinWeight = 9)
  (h3 : ∃! v, v ∈ vassals ∧ v.coinWeight = 9)
  : ∃ (weight : Nat), 
    weight < 4650 ∧ 
    4650 - weight ≤ 30 ∧
    ∃ (guilty : Vassal), 
      guilty ∈ vassals ∧
      guilty.coinWeight = 9 ∧
      guilty.id = 4650 - weight :=
by sorry

#check identify_guilty_vassal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_guilty_vassal_l328_32863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_path_length_divisible_by_four_l328_32809

/-- Represents a robot's move in the plane -/
inductive Move
| North
| East
| South
| West

/-- Represents a robot's path as a list of moves -/
def RobotPath := List Move

/-- Checks if a path returns to the origin -/
def returns_to_origin (p : RobotPath) : Bool :=
  sorry

/-- Checks if a path visits any point more than once -/
def no_revisits (p : RobotPath) : Bool :=
  sorry

/-- Theorem: Any valid path for the robot has a length divisible by 4 -/
theorem robot_path_length_divisible_by_four (p : RobotPath) 
  (h1 : returns_to_origin p = true) 
  (h2 : no_revisits p = true) : 
  ∃ k : ℕ, p.length = 4 * k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_path_length_divisible_by_four_l328_32809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_of_reflection_line_l328_32878

def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ := !![3/5, -4/5; -4/5, -3/5]

def direction_vector : Fin 2 → ℚ := ![3, -4]

theorem direction_vector_of_reflection_line :
  let v := direction_vector
  (reflection_matrix.mulVec v = v) ∧
  (v 0 > 0) ∧
  (Nat.gcd (Int.natAbs (v 0).num) (Int.natAbs (v 1).num) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_of_reflection_line_l328_32878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_times_distance_circle_l328_32812

/-- Given points A and B in a plane, prove that the set of all points M,
    which are three times farther away from A than from B, is a circle. -/
theorem three_times_distance_circle (A B M : ℝ × ℝ) :
  A = (0, 0) →
  B = (1, 0) →
  (let (x, y) := M
   (x^2 + y^2) = 9 * ((x - 1)^2 + y^2)) →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (9/8, 0) ∧
    radius = 3/8 ∧
    (let (x, y) := M
     (x - 9/8)^2 + y^2 = (3/8)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_times_distance_circle_l328_32812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_leq_one_l328_32827

open Real Set

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * sin x

-- State the theorem
theorem f_inequality_iff_a_leq_one :
  ∀ a : ℝ, (∀ x ∈ Icc 0 π, f a x ≥ 2 - cos x) ↔ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_leq_one_l328_32827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_cube_if_ratio_06_l328_32857

/-- Represents a three-dimensional solid -/
structure Solid where
  surface_area : ℝ

/-- Represents a cube -/
structure Cube extends Solid where
  side_length : ℝ
  surface_area_eq : surface_area = 6 * side_length^2

/-- Given two solids where the second is derived from the first by doubling its length -/
def doubled_length_solid (s : Solid) : Solid :=
  { surface_area := 4 * s.surface_area }

/-- Helper function to convert Cube to Solid -/
def cube_to_solid (c : Cube) : Solid :=
  { surface_area := c.surface_area }

/-- The theorem stating that if a cube has a surface area ratio of 0.6 
    when compared to another solid identical in all ways except its length 
    is doubled, then the second solid cannot be a cube -/
theorem not_cube_if_ratio_06 (c : Cube) :
  c.surface_area / (doubled_length_solid (cube_to_solid c)).surface_area = 0.6 →
  ¬∃ (c' : Cube), (doubled_length_solid (cube_to_solid c)).surface_area = c'.surface_area :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_cube_if_ratio_06_l328_32857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l328_32823

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- Define the function to calculate the area of a triangle given its base and height
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

-- Define the function to calculate the area of the right part of the triangle
noncomputable def rightTriangleArea (d : ℝ) : ℝ := triangleArea (10 - d) 4

-- Theorem statement
theorem equal_area_division :
  ∃ d : ℝ, d = 5 ∧ rightTriangleArea d = triangleArea 10 4 / 2 := by
  -- Proof goes here
  sorry

#eval A
#eval B
#eval C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l328_32823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_kindergarten_percentage_is_11_5_l328_32831

/-- Represents the student distribution for a school -/
structure SchoolDistribution where
  total_students : ℕ
  kindergarten_percentage : ℚ

/-- Calculates the combined kindergarten percentage for two schools -/
def combined_kindergarten_percentage (school1 school2 : SchoolDistribution) : ℚ :=
  let total_students := school1.total_students + school2.total_students
  let kindergarten_students1 := (school1.kindergarten_percentage * school1.total_students) / 100
  let kindergarten_students2 := (school2.kindergarten_percentage * school2.total_students) / 100
  let total_kindergarten_students := kindergarten_students1 + kindergarten_students2
  (total_kindergarten_students / total_students) * 100

/-- The main theorem stating that the combined kindergarten percentage is 11.5% -/
theorem combined_kindergarten_percentage_is_11_5 
  (annville : SchoolDistribution) 
  (cleona : SchoolDistribution) :
  annville.total_students = 150 →
  cleona.total_students = 250 →
  annville.kindergarten_percentage = 14 →
  cleona.kindergarten_percentage = 10 →
  combined_kindergarten_percentage annville cleona = 11.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_kindergarten_percentage_is_11_5_l328_32831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_whole_number_to_ratio_l328_32874

noncomputable def ratio : ℝ := (10^3000 + 10^3004) / (10^3002 + 10^3002)

theorem closest_whole_number_to_ratio : 
  ∃ (n : ℕ), n = 50 ∧ ∀ (m : ℕ), m ≠ n → |ratio - (n : ℝ)| < |ratio - (m : ℝ)| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_whole_number_to_ratio_l328_32874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_value_l328_32856

theorem ab_value (a b : ℝ) 
  (h1 : (2 : ℝ) ^ a = (16 : ℝ) ^ (b + 3))
  (h2 : (64 : ℝ) ^ b = (8 : ℝ) ^ (a - 2)) : 
  a * b = 40 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_value_l328_32856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l328_32834

noncomputable def triangle_area (base length : ℝ) : ℝ := (1 / 2) * base * length

theorem quadrilateral_area (AF BF CF DF AB BC CD : ℝ) : 
  AF = 30 →
  triangle_area BF AB + triangle_area CF BC + triangle_area DF CD = 147.65625 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l328_32834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_cost_l328_32894

theorem pen_cost (notebooks_count : ℕ) (notebooks_price : ℚ) 
                  (folders_count : ℕ) (folders_price : ℚ) 
                  (pens_count : ℕ) (paid_amount : ℚ) (change : ℚ) : 
  let total_spent := paid_amount - change
  let notebooks_cost := notebooks_count * notebooks_price
  let folders_cost := folders_count * folders_price
  let pens_total_cost := total_spent - notebooks_cost - folders_cost
  let pen_cost := pens_total_cost / pens_count
  pen_cost = 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_cost_l328_32894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_to_O_l328_32886

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

-- Define the major axis length
def major_axis_length : ℝ := 10

-- Define point A on the ellipse
noncomputable def A : ℝ × ℝ := sorry

-- Define focus F
noncomputable def F : ℝ × ℝ := sorry

-- Define point B as the midpoint of AF
noncomputable def B : ℝ × ℝ := ((A.1 + F.1) / 2, (A.2 + F.2) / 2)

-- Define the center of the ellipse
def O : ℝ × ℝ := (0, 0)

-- State the theorem
theorem distance_B_to_O : 
  ellipse A.1 A.2 →  -- A is on the ellipse
  dist A F = 2 →     -- A is 2 units away from F
  dist B O = 4       -- Distance from B to O is 4
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_to_O_l328_32886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_l328_32826

/-- Represents the time taken to cover a distance with different walking and running proportions -/
noncomputable def time_taken (total_distance : ℝ) (walk_speed : ℝ) (walk_proportion : ℝ) : ℝ :=
  (walk_proportion * total_distance) / walk_speed + 
  ((1 - walk_proportion) * total_distance) / (2 * walk_speed)

/-- Theorem stating the relationship between times taken with different walking and running proportions -/
theorem time_difference 
  (total_distance : ℝ) 
  (walk_speed : ℝ) 
  (h1 : walk_speed > 0) 
  (h2 : time_taken total_distance walk_speed (2/3) = 30) :
  time_taken total_distance walk_speed (1/3) = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_l328_32826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_theorem_l328_32887

theorem complex_sum_theorem (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1)
  (h2 : Complex.abs z₂ = 2)
  (h3 : 3 * z₁ - z₂ = Complex.ofReal 2 + Complex.I * Complex.ofReal (Real.sqrt 3)) :
  2 * z₁ + z₂ = Complex.ofReal 3 - Complex.I * Complex.ofReal (Real.sqrt 3) ∨ 
  2 * z₁ + z₂ = Complex.ofReal (-9/7) + Complex.I * Complex.ofReal (13 * Real.sqrt 3 / 7) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_theorem_l328_32887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l328_32861

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem first_term_of_geometric_sequence 
  (a : ℝ) (r : ℝ) 
  (h1 : geometric_sequence a r 4 = 120)  -- 5! = 120
  (h2 : geometric_sequence a r 7 = 5040) : -- 7! = 5040
  a = 120 / (42 ^ (1/3 : ℝ)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l328_32861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l328_32821

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = Real.pi

theorem triangle_problem 
  (A B C a b c : ℝ)
  (h_triangle : Triangle A B C a b c)
  (h_trig : Real.cos A * Real.cos C + Real.sin A * Real.sin C + Real.cos B = 3/2)
  (h_geom_prog : ∃ r : ℝ, a * r = b ∧ b * r = c)
  (h_tan_relation : a / Real.tan A + c / Real.tan C = 2 * b / Real.tan B)
  (h_a : a = 2) :
  B = Real.pi/3 ∧ A = Real.pi/3 ∧ C = Real.pi/3 := by
  sorry

#check triangle_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l328_32821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gujarati_speakers_count_l328_32849

/-- Represents the number of students who can speak a given language or combination of languages -/
structure StudentCount where
  count : Nat

/-- The total number of students in the class -/
def total_students : StudentCount := ⟨22⟩

/-- The number of students who can speak Hindi -/
def hindi_speakers : StudentCount := ⟨15⟩

/-- The number of students who can speak Marathi -/
def marathi_speakers : StudentCount := ⟨6⟩

/-- The number of students who can speak two languages -/
def bilingual_students : StudentCount := ⟨2⟩

/-- The number of students who can speak all three languages -/
def trilingual_students : StudentCount := ⟨1⟩

/-- Addition for StudentCount -/
instance : Add StudentCount where
  add a b := ⟨a.count + b.count⟩

/-- Subtraction for StudentCount -/
instance : Sub StudentCount where
  sub a b := ⟨a.count - b.count⟩

/-- Theorem stating that the number of Gujarati speakers is 2 -/
theorem gujarati_speakers_count : 
  ∃ (gujarati_speakers : StudentCount), 
    gujarati_speakers = ⟨2⟩ ∧
    total_students = gujarati_speakers + hindi_speakers + marathi_speakers - bilingual_students + trilingual_students :=
by
  use ⟨2⟩
  constructor
  · rfl
  · simp [total_students, hindi_speakers, marathi_speakers, bilingual_students, trilingual_students]
    rfl

#check gujarati_speakers_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gujarati_speakers_count_l328_32849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_negation_of_sin_positive_l328_32892

theorem negation_of_universal_proposition {α : Type*} (p : α → Prop) :
  (¬ ∀ x : α, p x) ↔ (∃ x : α, ¬ p x) :=
by sorry

theorem negation_of_sin_positive :
  (¬ ∀ x : ℝ, Real.sin x > 0) ↔ (∃ x : ℝ, Real.sin x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_negation_of_sin_positive_l328_32892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_staff_sample_size_l328_32847

/-- Calculates the number of general staff to be sampled in a stratified sampling scenario -/
theorem general_staff_sample_size
  (total_employees : ℕ)
  (general_staff_percentage : ℚ)
  (sample_size : ℕ)
  (h1 : total_employees = 1000)
  (h2 : general_staff_percentage = 80 / 100)
  (h3 : sample_size = 120) :
  Int.floor (general_staff_percentage * sample_size) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_staff_sample_size_l328_32847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_key_chain_manufacturing_cost_l328_32802

theorem key_chain_manufacturing_cost 
  (selling_price : ℝ)
  (original_profit_percentage : ℝ)
  (new_profit_percentage : ℝ)
  (new_manufacturing_cost : ℝ)
  (h1 : original_profit_percentage = 0.20)
  (h2 : new_profit_percentage = 0.50)
  (h3 : new_manufacturing_cost = 50)
  (h4 : selling_price = new_manufacturing_cost / (1 - new_profit_percentage)) :
  selling_price * (1 - original_profit_percentage) = 80 := by
  sorry

#check key_chain_manufacturing_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_key_chain_manufacturing_cost_l328_32802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_tshirt_expenditure_l328_32893

/-- The amount Lisa spent on t-shirts -/
def T : ℝ := sorry

/-- The total amount spent by Lisa and Carly -/
def total_spent : ℝ := 230

theorem lisa_tshirt_expenditure :
  T + (1/2 * T) + (2 * T) + (1/4 * T) + (3/2 * T) + (1/2 * T) = total_spent →
  T = total_spent / 4.75 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_tshirt_expenditure_l328_32893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_iff_three_right_angles_l328_32836

/-- A quadrilateral is a polygon with four sides -/
structure Quadrilateral where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  -- Add necessary conditions for a valid quadrilateral

/-- A rectangle is a quadrilateral with four right angles -/
def isRectangle (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, q.angles i = Real.pi / 2

/-- A quadrilateral has three right angles -/
def hasThreeRightAngles (q : Quadrilateral) : Prop :=
  ∃ i : Fin 4, ∀ j : Fin 4, j ≠ i → q.angles j = Real.pi / 2

theorem rectangle_iff_three_right_angles (q : Quadrilateral) :
  isRectangle q ↔ hasThreeRightAngles q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_iff_three_right_angles_l328_32836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_upper_bound_l328_32899

theorem function_upper_bound (f : ℕ → ℝ) :
  (∀ n, f n > 0) →
  (∀ n, f n ^ 2 ≤ f n - f (n + 1)) →
  ∀ n, f n < 1 / (n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_upper_bound_l328_32899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_cost_savings_l328_32850

theorem fuel_cost_savings 
  (old_efficiency : ℝ) 
  (efficiency_improvement : ℝ) 
  (fuel_price_increase : ℝ) : 
  let new_efficiency := old_efficiency * (1 + efficiency_improvement)
  let old_fuel_cost := (1 : ℝ)
  let new_fuel_cost := old_fuel_cost * (1 + fuel_price_increase)
  let old_trip_cost := old_fuel_cost / old_efficiency
  let new_trip_cost := new_fuel_cost / new_efficiency
  let savings_percentage := (old_trip_cost - new_trip_cost) / old_trip_cost * 100
  efficiency_improvement = 0.7 ∧ 
  fuel_price_increase = 0.25 →
  abs (savings_percentage - 26.5) < 0.1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_cost_savings_l328_32850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_ratio_is_three_l328_32877

/-- Represents the business partnership between A and B -/
structure Partnership where
  a_investment : ℚ
  b_investment : ℚ
  a_period : ℚ
  b_period : ℚ
  b_profit : ℚ
  total_profit : ℚ

/-- The ratio of A's investment to B's investment in the partnership -/
def investment_ratio (p : Partnership) : ℚ := p.a_investment / p.b_investment

/-- The conditions of the partnership as described in the problem -/
def partnership_conditions (p : Partnership) : Prop :=
  p.a_investment > 0 ∧
  p.b_investment > 0 ∧
  p.a_period = 2 * p.b_period ∧
  p.b_profit = 4000 ∧
  p.total_profit = 28000 ∧
  p.a_investment = (investment_ratio p) * p.b_investment

/-- Theorem stating that under the given conditions, the investment ratio is 3 -/
theorem investment_ratio_is_three (p : Partnership) 
  (h : partnership_conditions p) : investment_ratio p = 3 := by
  sorry

#check investment_ratio_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_ratio_is_three_l328_32877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_one_third_area_exists_l328_32852

/-- A point on the grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A line on the grid --/
structure GridLine where
  start : GridPoint
  end_ : GridPoint

/-- Predicate to check if a line passes through at least two grid points --/
def passesThoughTwoPoints (l : GridLine) : Prop :=
  l.start ≠ l.end_

/-- Predicate to check if a line does not coincide with grid lines --/
def notOnGridLine (l : GridLine) : Prop :=
  (l.start.x ≠ l.end_.x ∨ l.start.y ≠ l.end_.y) ∧
  (l.end_.x - l.start.x ≠ l.end_.y - l.start.y)

/-- The area of a triangle formed by three points --/
def triangleArea (p1 p2 p3 : GridPoint) : ℚ :=
  let x1 := p1.x
  let y1 := p1.y
  let x2 := p2.x
  let y2 := p2.y
  let x3 := p3.x
  let y3 := p3.y
  (((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) : ℤ).natAbs : ℚ) / 2

/-- Theorem: There exists a triangle with area 1/3 of a grid cell --/
theorem triangle_with_one_third_area_exists :
  ∃ (l1 l2 l3 : GridLine) (p1 p2 p3 : GridPoint),
    passesThoughTwoPoints l1 ∧
    passesThoughTwoPoints l2 ∧
    passesThoughTwoPoints l3 ∧
    notOnGridLine l1 ∧
    notOnGridLine l2 ∧
    notOnGridLine l3 ∧
    triangleArea p1 p2 p3 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_one_third_area_exists_l328_32852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_in_semicircle_l328_32800

/-- The distance between two points -/
def distance (A B : Point) : ℝ := sorry

/-- A rectangle is inscribed in a semicircle -/
def InscribedInSemicircle (A B C D G H : Point) : Prop := sorry

/-- The area of a rectangle -/
def Area (A B C D : Point) : ℝ := sorry

/-- The area of a rectangle inscribed in a semicircle -/
theorem rectangle_area_in_semicircle (A B C D G H : Point) 
  (h1 : InscribedInSemicircle A B C D G H) 
  (h2 : distance D A = 20) 
  (h3 : distance G D = 5) 
  (h4 : distance H D = 5) : 
  Area A B C D = 200 * Real.sqrt 2 := by
  sorry

#check rectangle_area_in_semicircle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_in_semicircle_l328_32800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_is_twenty_percent_l328_32889

/-- Represents the tradesman's business with three types of items -/
structure TradesmanBusiness where
  itemA_defraud : ℚ  -- Defraud percentage for Item A (both buying and selling)
  itemB_buy_defraud : ℚ  -- Defraud percentage for Item B when buying
  itemB_sell_defraud : ℚ  -- Defraud percentage for Item B when selling
  itemC_buy_defraud : ℚ  -- Defraud percentage for Item C when buying
  itemC_sell_defraud : ℚ  -- Defraud percentage for C when selling

/-- Calculates the overall percentage gain for the tradesman's business -/
def overallPercentageGain (business : TradesmanBusiness) : ℚ :=
  let itemA_gain := business.itemA_defraud
  let itemB_gain := business.itemB_sell_defraud - business.itemB_buy_defraud
  let itemC_gain := business.itemC_sell_defraud - business.itemC_buy_defraud
  (itemA_gain + itemB_gain + itemC_gain) / 3 * 100

/-- Theorem stating that the overall percentage gain is 20% -/
theorem overall_gain_is_twenty_percent (business : TradesmanBusiness) 
  (h1 : business.itemA_defraud = 3/10)
  (h2 : business.itemB_buy_defraud = 1/5)
  (h3 : business.itemB_sell_defraud = 1/10)
  (h4 : business.itemC_buy_defraud = 1/10)
  (h5 : business.itemC_sell_defraud = 1/5) :
  overallPercentageGain business = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_is_twenty_percent_l328_32889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_51_numbers_from_100_l328_32845

theorem divisibility_in_51_numbers_from_100 :
  ∀ (S : Finset ℕ), S.card = 51 → (∀ n, n ∈ S → 1 ≤ n ∧ n ≤ 100) →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_51_numbers_from_100_l328_32845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l328_32808

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - x^2 + a * x

-- Define the interval [1/e, e]
def interval : Set ℝ := { x | 1 / Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 }

-- Part 1
theorem part_one (a : ℝ) :
  (∃ m : ℝ, ∃ x y : ℝ, x ∈ interval ∧ y ∈ interval ∧ x ≠ y ∧
    f a x - a * x + m = 0 ∧ f a y - a * y + m = 0) →
  ∃ m : ℝ, 1 < m ∧ m ≤ 2 + 1 / (Real.exp 1)^2 :=
by
  sorry

-- Part 2
theorem part_two (a : ℝ) (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂)
  (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) :
  ∀ p q : ℝ, 0 < p → p ≤ q → p + q = 1 →
    (deriv (f a)) (p * x₁ + q * x₂) < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l328_32808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avery_work_time_l328_32879

/-- The time it takes Avery to build a complete wall alone -/
noncomputable def avery_time : ℝ := 3

/-- The time it takes Tom to build a complete wall alone -/
noncomputable def tom_time : ℝ := 3

/-- The time it takes Tom to complete the wall after Avery leaves (in hours) -/
noncomputable def tom_completion_time : ℝ := 60.000000000000014 / 60

/-- The fraction of the wall built per hour by each person -/
noncomputable def build_rate : ℝ := 1 / 3

/-- The theorem stating that Avery worked for 1 hour before leaving -/
theorem avery_work_time : 
  ∃ (x : ℝ), x = 1 ∧ 
  build_rate * x + build_rate * x + build_rate * tom_completion_time = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_avery_work_time_l328_32879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_is_385_l328_32801

noncomputable def geometricSum (x : ℝ) (n : ℕ) : ℝ := (1 - x^(n+1)) / (1 - x)

noncomputable def evenPowerSum (x : ℝ) (n : ℕ) : ℝ := (1 - x^(n+2)) / (1 - x^2)

noncomputable def coefficientX21 (x : ℝ) : ℝ :=
  (geometricSum x 20) * (geometricSum x 10)^2 * (evenPowerSum x 10)

theorem coefficient_is_385 :
  ⌊coefficientX21 1⌋ = 385 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_is_385_l328_32801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berries_left_for_mom_l328_32810

-- Define the number of berries picked in dozens
def strawberries_picked : ℚ := 2.5
def blueberries_picked : ℚ := 1.75
def raspberries_picked : ℚ := 1.25

-- Define the number of berries eaten
def strawberries_eaten : ℕ := 6
def blueberries_eaten : ℕ := 4

-- Define the conversion factor from dozens to individual berries
def dozen : ℕ := 12

-- Theorem statement
theorem berries_left_for_mom :
  (⌊strawberries_picked * ↑dozen⌋ - strawberries_eaten) +
  (⌊blueberries_picked * ↑dozen⌋ - blueberries_eaten) +
  ⌊raspberries_picked * ↑dozen⌋ = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_berries_left_for_mom_l328_32810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_competition_cost_theorem_l328_32819

/-- Badminton competition cost model -/
structure CostModel where
  /-- Venue fee (in yuan) -/
  b : ℝ
  /-- Proportionality constant for material cost per participant -/
  k : ℝ
  /-- Total cost function (in yuan) given number of participants -/
  y : ℝ → ℝ
  /-- Condition: y is a linear function of x with slope k and y-intercept b -/
  linear : ∀ x, y x = k * x + b
  /-- Condition: When x = 20, y = 1600 -/
  cond1 : y 20 = 1600
  /-- Condition: When x = 30, y = 2000 -/
  cond2 : y 30 = 2000

/-- Main theorem about the badminton competition cost model -/
theorem badminton_competition_cost_theorem (model : CostModel) :
  (∀ x, model.y x = 40 * x + 800) ∧
  (∀ x, model.y x ≤ 3000 → x ≤ 55) ∧
  (∀ x, 50 ≤ x ∧ x ≤ 70 → 2800 ≤ model.y x ∧ model.y x ≤ 3600) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_competition_cost_theorem_l328_32819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l328_32838

/-- A tetrahedron is a three-dimensional geometric shape with four triangular faces. -/
structure Tetrahedron where
  /-- The areas of the four faces of the tetrahedron -/
  face_areas : Fin 4 → ℝ
  /-- The radius of the inscribed sphere of the tetrahedron -/
  inscribed_radius : ℝ

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ :=
  (1 / 3) * (t.face_areas 0 + t.face_areas 1 + t.face_areas 2 + t.face_areas 3) * t.inscribed_radius

/-- Theorem: The volume of a tetrahedron is equal to one-third of the product of the sum of its face areas and the radius of its inscribed sphere -/
theorem tetrahedron_volume (t : Tetrahedron) :
  volume t = (1 / 3) * (t.face_areas 0 + t.face_areas 1 + t.face_areas 2 + t.face_areas 3) * t.inscribed_radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l328_32838
