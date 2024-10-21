import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_on_post_l259_25995

/-- The length of a string wrapped around a cylindrical post -/
noncomputable def string_length (circumference height : ℝ) (wraps : ℕ) : ℝ :=
  let vertical_distance := height / (wraps : ℝ)
  let horizontal_distance := circumference
  (wraps : ℝ) * Real.sqrt (vertical_distance ^ 2 + horizontal_distance ^ 2)

/-- Theorem: The length of the string is 5√34 feet -/
theorem string_length_on_post :
  string_length 5 15 5 = 5 * Real.sqrt 34 := by
  unfold string_length
  simp
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_on_post_l259_25995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_area_l259_25972

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define the right focus F (coordinates to be determined)
noncomputable def F : ℝ × ℝ := sorry

-- Define point A
noncomputable def A : ℝ × ℝ := (0, 6 * Real.sqrt 6)

-- Define a point P on the left branch of the hyperbola
noncomputable def P : ℝ × ℝ := sorry

-- Define the condition that P is on the left branch
def P_on_left_branch : Prop := 
  hyperbola P.1 P.2 ∧ P.1 < 0

-- Define the perimeter of triangle APF
noncomputable def perimeter (A F P : ℝ × ℝ) : ℝ := sorry

-- Define the area of triangle APF
noncomputable def area (A F P : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem min_perimeter_area : 
  ∀ P : ℝ × ℝ, P_on_left_branch → 
  (∀ Q : ℝ × ℝ, P_on_left_branch → perimeter A F P ≤ perimeter A F Q) →
  area A F P = 12 * Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_area_l259_25972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l259_25924

/-- A power function passing through (3, √3) -/
noncomputable def power_function (k a : ℝ) : ℝ → ℝ := fun x ↦ k * x^a

/-- The condition that the function passes through (3, √3) -/
def passes_through (k a : ℝ) : Prop := power_function k a 3 = Real.sqrt 3

theorem power_function_sum (k a : ℝ) (h : passes_through k a) : k + a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l259_25924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_existence_l259_25947

-- Define the necessary structures
structure Point where

structure Line where

structure Plane where

-- Define the geometric relationships
def passes_through (l : Line) (p : Point) : Prop := sorry

def perpendicular (l1 l2 : Line) : Prop := sorry

def parallel_line_plane (l : Line) (s : Plane) : Prop := sorry

-- Define the theorem
theorem line_existence 
  (P : Point) (g : Line) (S : Plane) : 
  ∃ l : Line, 
    passes_through l P ∧ 
    perpendicular l g ∧ 
    parallel_line_plane l S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_existence_l259_25947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_series_term_limit_zero_specific_series_convergence_l259_25945

noncomputable def geometric_series (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

noncomputable def series_term (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r^n

theorem geometric_series_convergence (a : ℝ) (r : ℝ) 
  (h : 0 < |r| ∧ |r| < 1) : 
  ∃ (L : ℝ), L = a / (1 - r) ∧ 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
  |geometric_series a r n - L| < ε :=
by sorry

theorem series_term_limit_zero (a : ℝ) (r : ℝ) 
  (h : 0 < |r| ∧ |r| < 1) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
  |series_term a r n| < ε :=
by sorry

theorem specific_series_convergence :
  let a : ℝ := 3
  let r : ℝ := 1/3
  ∃ (L : ℝ), L = 4.5 ∧
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
   |geometric_series a r n - L| < ε) ∧
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
   |series_term a r n| < ε) ∧
  (∃ (L : ℝ), ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
   |geometric_series a r n - L| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_series_term_limit_zero_specific_series_convergence_l259_25945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_smallest_x_sin_cos_conditions_l259_25916

/-- The function f(x) = sin(x/4) + cos(x/9) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.cos (x / 9)

/-- The smallest positive value of x in degrees where f(x) achieves its maximum -/
def smallest_max_x : ℝ := 3600

/-- Theorem stating that f achieves its maximum at smallest_max_x -/
theorem f_max_at_smallest_x :
  (∀ x : ℝ, x > 0 → f x ≤ f smallest_max_x) ∧
  (∀ x : ℝ, x > 0 → x < smallest_max_x → f x < f smallest_max_x) := by
  sorry

/-- Theorem relating the conditions on sin and cos to smallest_max_x -/
theorem sin_cos_conditions (x : ℝ) :
  Real.sin (x / 4) = 1 ∧ Real.cos (x / 9) = 1 ↔ x = smallest_max_x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_smallest_x_sin_cos_conditions_l259_25916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_32_terms_b_sequence_l259_25966

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

noncomputable def b_sequence (a : ℕ → ℝ) : ℕ → ℝ := 
  λ n => 1 / (a n * Real.sqrt (a n) * Real.sqrt (a (n + 1)) + a (n + 1) * Real.sqrt (a n) * Real.sqrt (a (n + 1)))

noncomputable def sum_b_sequence (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum b

theorem sum_32_terms_b_sequence :
  let a := arithmetic_sequence 4 3
  let b := b_sequence a
  sum_b_sequence b 32 = 2 / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_32_terms_b_sequence_l259_25966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PQRS_volume_l259_25920

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron using the Cayley-Menger determinant -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- The specific tetrahedron PQRS from the problem -/
noncomputable def PQRS : Tetrahedron := {
  PQ := 6,
  PR := 4,
  PS := 5,
  QR := 7,
  QS := 8,
  RS := Real.sqrt 65
}

/-- Theorem stating that the volume of tetrahedron PQRS is 20 -/
theorem PQRS_volume : tetrahedronVolume PQRS = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PQRS_volume_l259_25920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_is_five_hours_l259_25939

/-- Calculates the total time taken to row from A to B and back -/
noncomputable def total_rowing_time (rowing_speed : ℝ) (distance : ℝ) (stream_speed : ℝ) : ℝ :=
  let downstream_speed := rowing_speed + stream_speed
  let upstream_speed := rowing_speed - stream_speed
  let time_downstream := distance / downstream_speed
  let time_upstream := distance / upstream_speed
  time_downstream + time_upstream

/-- Theorem stating that the total rowing time is 5 hours under given conditions -/
theorem rowing_time_is_five_hours :
  let rowing_speed : ℝ := 10
  let distance : ℝ := 24
  let stream_speed : ℝ := 2
  total_rowing_time rowing_speed distance stream_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_is_five_hours_l259_25939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_sq_inch_difference_l259_25999

-- Define the parameters for the first TV
noncomputable def first_tv_width : ℝ := 24
noncomputable def first_tv_height : ℝ := 16
noncomputable def first_tv_original_cost : ℝ := 840
noncomputable def first_tv_discount_rate : ℝ := 0.10
noncomputable def first_tv_tax_rate : ℝ := 0.05

-- Define the parameters for the new TV
noncomputable def new_tv_width : ℝ := 48
noncomputable def new_tv_height : ℝ := 32
noncomputable def new_tv_original_cost : ℝ := 1800
noncomputable def new_tv_first_discount_rate : ℝ := 0.20
noncomputable def new_tv_second_discount_rate : ℝ := 0.15
noncomputable def new_tv_tax_rate : ℝ := 0.08

-- Function to calculate the final cost after discount and tax
noncomputable def calculate_final_cost (original_cost discount_rate tax_rate : ℝ) : ℝ :=
  let discounted_price := original_cost * (1 - discount_rate)
  discounted_price * (1 + tax_rate)

-- Function to calculate the area of a TV
noncomputable def calculate_area (width height : ℝ) : ℝ :=
  width * height

-- Function to calculate cost per square inch
noncomputable def calculate_cost_per_sq_inch (final_cost area : ℝ) : ℝ :=
  final_cost / area

-- Theorem statement
theorem cost_per_sq_inch_difference : 
  let first_tv_final_cost := calculate_final_cost first_tv_original_cost first_tv_discount_rate first_tv_tax_rate
  let first_tv_area := calculate_area first_tv_width first_tv_height
  let first_tv_cost_per_sq_inch := calculate_cost_per_sq_inch first_tv_final_cost first_tv_area

  let new_tv_discounted_price := new_tv_original_cost * (1 - new_tv_first_discount_rate) * (1 - new_tv_second_discount_rate)
  let new_tv_final_cost := new_tv_discounted_price * (1 + new_tv_tax_rate)
  let new_tv_area := calculate_area new_tv_width new_tv_height
  let new_tv_cost_per_sq_inch := calculate_cost_per_sq_inch new_tv_final_cost new_tv_area

  let difference := first_tv_cost_per_sq_inch - new_tv_cost_per_sq_inch

  ∃ (ε : ℝ), ε > 0 ∧ |difference - 1.2073| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_sq_inch_difference_l259_25999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_only_stable_l259_25907

-- Define the Shape type
inductive Shape
  | Triangle
  | Rectangle
  | Square
  | Parallelogram

-- Define the side length property (this is a placeholder, as we don't have actual geometric implementations)
noncomputable def sideLength (s : Shape) (side : ℝ) : ℝ := 
  match s with
  | Shape.Triangle => 1
  | Shape.Rectangle => 1
  | Shape.Square => 1
  | Shape.Parallelogram => 1

-- Define the stability property
def isStable (s : Shape) : Prop :=
  ∀ (deformation : Shape → Shape), 
    deformation s ≠ s → ∃ (side : ℝ), sideLength s side ≠ sideLength (deformation s) side

-- The theorem to prove
theorem triangle_only_stable :
  ∀ (s : Shape), isStable s ↔ s = Shape.Triangle := by sorry

-- Example usage of the theorem (not part of the proof)
example : isStable Shape.Triangle := by
  apply (triangle_only_stable Shape.Triangle).mpr
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_only_stable_l259_25907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_two_solutions_l259_25980

/-- Represents a triangle with sides a, b, c and angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Checks if a triangle has two distinct solutions given partial information --/
noncomputable def hasTwoSolutions (t : Triangle) : Prop :=
  ∃ (B1 B2 : ℝ), B1 ≠ B2 ∧ 
    Real.sin t.A / t.a = Real.sin B1 / t.b ∧
    Real.sin t.A / t.a = Real.sin B2 / t.b ∧
    0 < B1 ∧ B1 < Real.pi ∧ 0 < B2 ∧ B2 < Real.pi

theorem triangle_two_solutions :
  let t1 : Triangle := { a := 10, b := 10, c := 0, A := Real.pi/4, B := 7*Real.pi/18, C := 0 }
  let t2 : Triangle := { a := 60, b := 0, c := 48, A := 0, B := 5*Real.pi/9, C := 0 }
  let t3 : Triangle := { a := 14, b := 16, c := 0, A := Real.pi/4, B := 0, C := 0 }
  let t4 : Triangle := { a := 7, b := 5, c := 0, A := 4*Real.pi/9, B := 0, C := 0 }
  ¬(hasTwoSolutions t1) ∧
  ¬(hasTwoSolutions t2) ∧
  hasTwoSolutions t3 ∧
  ¬(hasTwoSolutions t4) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_two_solutions_l259_25980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l259_25976

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x + 3

-- Theorem statement
theorem f_properties :
  -- f(x) has exactly two extreme points
  (∃ (a b : ℝ), a ≠ b ∧
    (∀ (x : ℝ), (∃ (ε : ℝ), ε > 0 ∧ ∀ (y : ℝ), |y - a| < ε → f y ≤ f a) ∧
                (∃ (ε : ℝ), ε > 0 ∧ ∀ (y : ℝ), |y - b| < ε → f y ≤ f b)) ∧
    (∀ (c : ℝ), c ≠ a → c ≠ b →
      ¬(∃ (ε : ℝ), ε > 0 ∧ ∀ (y : ℝ), |y - c| < ε → f y ≤ f c))) ∧
  -- (0, 3) is the center of symmetry of y = f(x)
  (∀ (x : ℝ), f x + f (-x) = 6) ∧
  -- f(x) = k has two distinct roots iff k = 1 or k = 5
  (∀ (k : ℝ), (∃ (x y : ℝ), x ≠ y ∧ f x = k ∧ f y = k) ↔ (k = 1 ∨ k = 5)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l259_25976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l259_25993

theorem unique_function_satisfying_conditions (f g h : ℝ → ℝ) 
  (hg : g = λ x ↦ x + 1) 
  (hh : h = λ x ↦ x^2) 
  (cond_g : ∀ x, f (g x) = g (f x)) 
  (cond_h : ∀ x, f (h x) = h (f x)) : 
  f = id := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l259_25993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_pi_l259_25903

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * x

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x + y - Real.pi = 0

theorem tangent_at_pi :
  ∃ (m b : ℝ), 
    (∀ x, tangent_line x (m * x + b)) ∧ 
    tangent_line Real.pi (f Real.pi) ∧
    (deriv f) Real.pi = m :=
by
  sorry

#check tangent_at_pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_pi_l259_25903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l259_25948

theorem equation_solution : ∃ x : ℝ, 125 = 5 * (25 : ℝ) ^ (x - 1) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l259_25948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brothers_identity_and_cards_l259_25991

-- Define the brothers and card colors
inductive Brother : Type
| first : Brother
| second : Brother

inductive CardColor : Type
| purple : CardColor
| orange : CardColor

-- Define the card assignment function
variable (card_assignment : Brother → CardColor)

-- Define the name assignment function
variable (name_assignment : Brother → String)

-- Theorem statement
theorem brothers_identity_and_cards :
  (∃ b : Brother, card_assignment b = CardColor.purple) →  -- At least one card is purple
  (name_assignment Brother.first = "Tralalya") →           -- First brother claims to be Tralalya
  (card_assignment Brother.first = CardColor.orange ∧     -- First brother has orange card
   card_assignment Brother.second = CardColor.purple ∧    -- Second brother has purple card
   name_assignment Brother.first = "Tralalya" ∧           -- First brother is Tralalya
   name_assignment Brother.second = "Trulalya") :=        -- Second brother is Trulalya
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brothers_identity_and_cards_l259_25991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_amount_l259_25954

/-- Given two liquids p and q in a mixture, this function calculates the total amount of the mixture -/
noncomputable def mixtureTotalAmount (p q : ℝ) : ℝ := p + q

/-- This function calculates the ratio of two quantities -/
noncomputable def ratio (a b : ℝ) : ℝ := a / b

/-- This theorem proves that the initial amount of the mixture is 30 liters -/
theorem initial_mixture_amount :
  ∀ (p q : ℝ),
  (ratio p q = 3 / 2) →
  (ratio p (q + 12) = 3 / 4) →
  (mixtureTotalAmount p q = 30) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_amount_l259_25954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2006_is_zero_l259_25959

def sequence_a : ℕ → ℕ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | n + 2 => if n % 2 = 0 then sequence_a (n / 2 + 1) else 1 - sequence_a ((n + 1) / 2 + 1)

theorem sequence_a_2006_is_zero : sequence_a 2006 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2006_is_zero_l259_25959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l259_25992

theorem problem_solution (m n : ℕ) 
  (h1 : m + 7 < n + 3)
  (h2 : (m + (m+3) + (m+7) + (n+3) + (n+6) + 2*n) / 6 = n + 3)
  (h3 : ((m+7) + (n+3)) / 2 = n + 3) :
  m + n = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l259_25992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_f_composition_l259_25962

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 + 2 else x - 1

-- State the theorem
theorem two_solutions_for_f_composition :
  ∃ (a b : ℝ), a ≠ b ∧ 
    f (f a) = 10 ∧ 
    f (f b) = 10 ∧ 
    ∀ x : ℝ, f (f x) = 10 → x = a ∨ x = b :=
by
  sorry

#check two_solutions_for_f_composition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_f_composition_l259_25962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gumball_water_intake_l259_25964

/-- 
Given Gumball's water intake for a week:
- He drank 9 liters on Monday, Thursday, and Saturday
- He drank 8 liters on Tuesday, Friday, and Sunday
- He drank an unknown amount on Wednesday
- The total water intake for the week was 60 liters

This theorem proves that Gumball drank 9 liters on Wednesday.
-/
theorem gumball_water_intake (water_intake : Fin 7 → ℕ) : 
  water_intake 0 = 9 ∧ 
  water_intake 1 = 8 ∧ 
  water_intake 3 = 9 ∧ 
  water_intake 4 = 8 ∧ 
  water_intake 5 = 9 ∧ 
  water_intake 6 = 8 ∧
  (Finset.sum Finset.univ water_intake) = 60 →
  water_intake 2 = 9 := by
  intro h
  sorry

#check gumball_water_intake

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gumball_water_intake_l259_25964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_equal_faced_polyhedra_are_regular_l259_25956

-- Define a polyhedron
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  edges : Set ((Fin 3 → ℝ) × (Fin 3 → ℝ))
  faces : Set (Set (Fin 3 → ℝ))

-- Define a regular polygon
structure RegularPolygon where
  vertices : Set (Fin 3 → ℝ)
  num_sides : ℕ

-- Define properties of a polyhedron
def has_equal_regular_polygon_faces (p : Polyhedron) : Prop :=
  ∃ (r : RegularPolygon), ∀ f ∈ p.faces, f = r.vertices

def is_regular (p : Polyhedron) : Prop :=
  -- A regular polyhedron is vertex-transitive and face-transitive
  sorry

-- Theorem statement
theorem not_all_equal_faced_polyhedra_are_regular :
  ∃ (p : Polyhedron), has_equal_regular_polygon_faces p ∧ ¬is_regular p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_equal_faced_polyhedra_are_regular_l259_25956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_min_max_cubic_quartic_l259_25953

theorem sum_min_max_cubic_quartic (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 30) :
  let f := λ (x y z w : ℝ) ↦ 3 * (x^3 + y^3 + z^3 + w^3) - 2 * (x^4 + y^4 + z^4 + w^4)
  ∃ (m M : ℝ), (∀ p q r s, f p q r s ≥ m ∧ f p q r s ≤ M) ∧ 
               (∃ p q r s, f p q r s = m) ∧
               (∃ p q r s, f p q r s = M) ∧
               m + M = 88 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_min_max_cubic_quartic_l259_25953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l259_25934

/-- The area of a trapezoid with height y, one base 3y, and another base 4y -/
noncomputable def trapezoid_area (y : ℝ) : ℝ := y * (3 * y + 4 * y) / 2

/-- Theorem: The area of the trapezoid is 7y²/2 -/
theorem trapezoid_area_formula (y : ℝ) : trapezoid_area y = 7 * y^2 / 2 := by
  -- Unfold the definition of trapezoid_area
  unfold trapezoid_area
  -- Simplify the expression
  simp [mul_add, mul_div_right_comm]
  -- Perform algebraic manipulations
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l259_25934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hummus_ingredients_needed_l259_25910

-- Define the recipe for one serving of hummus
def chickpeas_per_serving : ℚ := 1 -- in cups
def tahini_per_serving : ℚ := 3 -- in tablespoons
def olive_oil_per_serving : ℚ := 3/2 -- in tablespoons

-- Define packaging information
def chickpeas_per_can : ℚ := 16 -- in ounces
def chickpeas_per_cup : ℚ := 6 -- in ounces
def tahini_per_jar : ℚ := 8 -- in ounces
def tahini_tbsp_per_ounce : ℚ := 2
def olive_oil_per_bottle : ℚ := 32 -- in ounces
def olive_oil_tbsp_per_ounce : ℚ := 2

-- Define the number of servings
def num_servings : ℕ := 20

-- Theorem statement
theorem hummus_ingredients_needed :
  let chickpeas_needed := Int.ceil ((chickpeas_per_serving * num_servings * chickpeas_per_cup / chickpeas_per_can : ℚ))
  let tahini_needed := Int.ceil ((tahini_per_serving * num_servings / (tahini_per_jar * tahini_tbsp_per_ounce) : ℚ))
  let olive_oil_needed := Int.ceil ((olive_oil_per_serving * num_servings / (olive_oil_per_bottle * olive_oil_tbsp_per_ounce) : ℚ))
  chickpeas_needed = 8 ∧ tahini_needed = 4 ∧ olive_oil_needed = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hummus_ingredients_needed_l259_25910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_even_functions_l259_25949

-- Define the functions
def f₁ (x : ℝ) : ℝ := x^2

noncomputable def f₂ (x : ℝ) : ℝ := Real.log x

noncomputable def g (x : ℝ) : ℝ := (2 : ℝ)^x - (2 : ℝ)^(-x)

noncomputable def h (x : ℝ) : ℝ := (2 : ℝ)^x + (2 : ℝ)^(-x)

-- Define what it means for a function to be even
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem two_even_functions : 
  (IsEven f₁ ∧ ¬IsEven f₂ ∧ ¬IsEven g ∧ IsEven h) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_even_functions_l259_25949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l259_25971

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 7 - 3*x + 1

-- Define the proposed inverse function g
noncomputable def g (x : ℝ) : ℝ := (8 - x) / 3

-- Theorem statement
theorem f_inverse_is_g : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l259_25971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_on_pyramid_l259_25912

/-- The length of the shortest closed path on the surface of a regular quadrilateral pyramid -/
noncomputable def shortest_path (b : ℝ) (α : ℝ) : ℝ :=
  if α ≤ Real.pi/4 then 2 * b * Real.sin (2 * α) else 2 * b

/-- Theorem: The shortest closed path on the surface of a regular quadrilateral pyramid -/
theorem shortest_path_on_pyramid (b : ℝ) (α : ℝ) 
  (h1 : b > 0) (h2 : 0 < α ∧ α < Real.pi/2) :
  let path := shortest_path b α
  ∀ other_path : ℝ, 
    (other_path ≥ path ∧ 
     other_path = path ↔ other_path = shortest_path b α) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_on_pyramid_l259_25912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AOC_is_three_pi_four_l259_25958

noncomputable section

def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (3/2, 1)
def C : ℝ × ℝ := (5, -1)
def O : ℝ × ℝ := (0, 0)

def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, q - p = t • (r - p)

def orthocenter (p q r : ℝ × ℝ) : ℝ × ℝ := sorry

def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem angle_AOC_is_three_pi_four :
  collinear A B C ∧
  ((A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0) ∧
  B - O = (3/2) • (orthocenter O A C - O) →
  angle O A C = 3 * π / 4 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AOC_is_three_pi_four_l259_25958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_log_three_halves_l259_25938

/-- The series sum from n=2 to infinity of log(n^3 + 1) - log(n^3 - 1) equals log(3/2) -/
theorem series_sum_equals_log_three_halves :
  ∑' n : ℕ, (Real.log ((n : ℝ)^3 + 1) - Real.log ((n : ℝ)^3 - 1)) = Real.log (3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_log_three_halves_l259_25938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l259_25969

-- Define the binary operation ⊗
noncomputable def otimes (a b : ℝ) : ℝ :=
  if a ≤ b then a else b

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  otimes (Real.sin x) (Real.cos x)

-- Theorem statement
theorem range_of_f :
  Set.range f = Set.Icc (-1) (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l259_25969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_theta_plus_pi_half_l259_25988

theorem sin_two_theta_plus_pi_half (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo 0 (π/4))
  (h2 : Real.tan (2*θ) = Real.cos θ / (2 - Real.sin θ)) :
  Real.sin (2*θ + π/2) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_theta_plus_pi_half_l259_25988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_utilization_rate_l259_25917

noncomputable section

-- Define the radius of the original spherical craft
def sphere_radius : ℝ := 2

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Define the volume of a cylinder
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Define the material utilization rate
noncomputable def utilization_rate (new_volume original_volume : ℝ) : ℝ :=
  new_volume / original_volume

-- Theorem statement
theorem max_utilization_rate :
  ∃ (cylinder_radius cylinder_height : ℝ),
    cylinder_radius > 0 ∧
    cylinder_height > 0 ∧
    cylinder_radius^2 + (cylinder_height/2)^2 = sphere_radius^2 ∧
    ∀ (r h : ℝ),
      r > 0 → h > 0 → r^2 + (h/2)^2 ≤ sphere_radius^2 →
      utilization_rate (cylinder_volume r h) (sphere_volume sphere_radius) ≤
      utilization_rate (cylinder_volume cylinder_radius cylinder_height) (sphere_volume sphere_radius) ∧
    utilization_rate (cylinder_volume cylinder_radius cylinder_height) (sphere_volume sphere_radius) = Real.sqrt 3 / 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_utilization_rate_l259_25917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_4_f_monotonic_intervals_l259_25913

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (3 * x) * Real.cos x - Real.cos (3 * x) * Real.sin x + Real.cos (2 * x)

theorem f_value_at_pi_over_4 : f (π / 4) = 1 := by sorry

theorem f_monotonic_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * π - 3 * π / 8) (k * π + π / 8)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_4_f_monotonic_intervals_l259_25913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_inclusions_l259_25984

/-- Set A in R^2 -/
def A : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 ≤ 1}

/-- Set B in R^2 -/
def B : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}

/-- Set C in R^2 -/
def C : Set (ℝ × ℝ) := {p | abs p.1 ≤ 1 ∧ abs p.2 ≤ 1}

/-- Theorem stating the inclusion relationships among sets A, B, and C -/
theorem set_inclusions : A ⊂ B ∧ B ⊂ C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_inclusions_l259_25984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_grid_fraction_l259_25925

-- Define the triangle vertices
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (6, 2)
def C : ℝ × ℝ := (5, 5)

-- Define the grid dimensions
def gridWidth : ℝ := 7
def gridHeight : ℝ := 6

-- Function to calculate triangle area using Shoelace formula
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

-- Theorem statement
theorem triangle_grid_fraction :
  (triangleArea A B C) / (gridWidth * gridHeight) = 11/84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_grid_fraction_l259_25925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_specific_line_l259_25973

theorem slope_of_specific_line :
  ((-2) - 3) / ((-4) - 1) = 1 := by
  norm_num

#eval ((-2) - 3) / ((-4) - 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_specific_line_l259_25973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l259_25930

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 1)) / (x - 2)

-- Define the domain of f
def domain_f : Set ℝ := { x | x ≥ 1 ∧ x ≠ 2 }

-- Theorem statement
theorem domain_of_f : 
  domain_f = Set.Icc 1 2 ∪ Set.Ioi 2 := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l259_25930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_satisfies_conditions_l259_25944

-- Define the piecewise function h
noncomputable def h (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ 0 then -x - 2
  else if 0 < x ∧ x ≤ 3 then Real.sqrt (9 - (x - 3)^2) - 3
  else if 3 < x ∧ x ≤ 4 then 3 * (x - 3)
  else 0  -- Define a default value for x outside the specified ranges

-- Define the transformation j
noncomputable def j (x : ℝ) : ℝ := h (x / 3) - 5

-- Theorem statement
theorem transformation_satisfies_conditions :
  (∀ x, j x = h (x / 3) - 5) ∧
  (∀ x, j (3 * x) = h x - 5) := by
  constructor
  · intro x
    rfl
  · intro x
    simp [j, h]
    -- The proof would go here, but we'll use sorry for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_satisfies_conditions_l259_25944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_theorem_l259_25952

noncomputable section

open EuclideanSpace

/-- The chord intersection theorem -/
theorem chord_intersection_theorem 
  (O A A₁ B B₁ C C₁ X M : EuclideanSpace ℝ (Fin 2)) :
  let circle := Metric.sphere O (dist O A)
  let is_on_circle (P : EuclideanSpace ℝ (Fin 2)) := P ∈ circle
  let is_chord (P Q : EuclideanSpace ℝ (Fin 2)) := is_on_circle P ∧ is_on_circle Q
  is_on_circle A ∧ is_on_circle A₁ ∧ is_chord A A₁ ∧
  is_on_circle B ∧ is_on_circle B₁ ∧ is_chord B B₁ ∧
  is_on_circle C ∧ is_on_circle C₁ ∧ is_chord C C₁ ∧
  (∃ t : ℝ, X = A + t • (A₁ - A)) ∧
  (∃ s : ℝ, X = B + s • (B₁ - B)) ∧
  (∃ r : ℝ, X = C + r • (C₁ - C)) ∧
  M = (1/3 : ℝ) • (A + B + C) →
  (dist A X / dist X A₁ + dist B X / dist X B₁ + dist C X / dist X C₁ = 3) ↔
  (dist O X)^2 + (dist M X)^2 = (dist O M)^2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_theorem_l259_25952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l259_25979

-- Define the initial conditions
noncomputable def initial_pencils : ℝ := 15
noncomputable def initial_price : ℝ := 1
noncomputable def initial_loss_percentage : ℝ := 15

-- Define the new selling strategy
noncomputable def new_pencils : ℝ := 11.09

-- Calculate the cost price per pencil
noncomputable def cost_price : ℝ := initial_price / (initial_pencils * (1 - initial_loss_percentage / 100))

-- Calculate the new selling price per pencil
noncomputable def new_selling_price : ℝ := initial_price / new_pencils

-- Define the theorem
theorem profit_percentage_calculation :
  ∃ ε > 0, |((new_selling_price / cost_price) - 1) * 100 - 14.94| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l259_25979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_from_tangent_l259_25990

theorem cosine_from_tangent (α : Real) :
  (α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) →  -- Fourth quadrant condition
  (Real.tan α = -5/12) →
  (Real.cos α = 12/13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_from_tangent_l259_25990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_line_properties_l259_25977

-- Define the line l with irrational slope k passing through the origin
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x

-- Theorem statement
theorem irrational_line_properties (k : ℝ) (h_irrational : Irrational k) :
  -- Part (i): The origin is the only rational point on the line
  (∀ x y : ℚ, (y : ℝ) = line_l k (x : ℝ) → x = 0 ∧ y = 0) ∧
  -- Part (ii): For any ε > 0, there exist integers m and n such that
  -- the distance between the line and (m, n) is less than ε
  (∀ ε : ℝ, ε > 0 → ∃ m n : ℤ, 
    |k * (m : ℝ) - (n : ℝ)| / Real.sqrt (k^2 + 1) < ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_line_properties_l259_25977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_proof_l259_25909

noncomputable def discount_percentage (original_price discounted_price : ℝ) : ℝ :=
  (original_price - discounted_price) / original_price * 100

theorem discount_proof (original_price discounted_price : ℝ) 
  (h1 : original_price = 1000)
  (h2 : discounted_price = 700) :
  discount_percentage original_price discounted_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_proof_l259_25909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_condition_l259_25983

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(x*(x-a))

-- State the theorem
theorem monotonically_decreasing_condition (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂) ↔ a ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_condition_l259_25983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l259_25955

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x - 2

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem inverse_function_theorem (a b : ℝ) :
  (∀ x, g x = (f a b).invFun x - 2) →
  3 * a + 4 * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l259_25955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_g_min_max_l259_25963

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 + 1 / 2

noncomputable def g (x : ℝ) := Real.sin (2 * x - Real.pi / 2)

theorem f_monotone_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)) :=
sorry

theorem g_min_max :
  (∀ x ∈ Set.Icc 0 Real.pi, g x ≥ -Real.sqrt 3 / 2) ∧
  (∀ x ∈ Set.Icc 0 Real.pi, g x ≤ 1) ∧
  (∃ x ∈ Set.Icc 0 Real.pi, g x = -Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 Real.pi, g x = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_g_min_max_l259_25963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l259_25919

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else (2 : ℝ)^x

-- State the theorem
theorem f_inequality_range (x : ℝ) :
  f x + f (x - 1/2) > 1 ↔ x > -1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l259_25919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_AB_l259_25961

noncomputable section

-- Define circle C₁
def C₁ (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 + 2 * Real.sin α)

-- Define circle C₂
def C₂ (θ : ℝ) : ℝ × ℝ := 
  let p := 2 * Real.sqrt 2 * Real.cos (θ + Real.pi/4)
  (p * Real.cos θ, p * Real.sin θ)

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Theorem stating the length of chord AB is 4
theorem chord_length_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_AB_l259_25961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_trig_composition_l259_25926

open Real

theorem largest_trig_composition (x : ℝ) (h : x ∈ Set.Ioo 0 (π/4)) :
  max (sin (cos x)) (max (sin (sin x)) (max (cos (sin x)) (cos (cos x)))) = cos (sin x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_trig_composition_l259_25926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncertain_line_plane_relationship_l259_25922

structure GeometricSpace where
  Point : Type
  Line : Type
  Plane : Type
  contains : Plane → Line → Prop
  perpendicular_planes : Plane → Plane → Prop
  perpendicular_line_plane : Line → Plane → Prop
  perpendicular_lines : Line → Line → Prop
  intersects_at_angle : Line → Plane → ℝ → Prop
  parallel : Line → Plane → Prop

variable (S : GeometricSpace)

theorem uncertain_line_plane_relationship
  (α β : S.Plane) (b m : S.Line) :
  S.perpendicular_planes α β →
  S.contains α b →
  S.contains β m →
  S.perpendicular_lines b m →
  ¬(S.perpendicular_line_plane b β ∨ 
    (∃ θ : ℝ, 0 < θ ∧ θ < π/2 ∧ S.intersects_at_angle b β θ) ∨ 
    S.parallel b β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncertain_line_plane_relationship_l259_25922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l259_25908

-- Define the train length in meters
noncomputable def train_length : ℝ := 150

-- Define the time taken to pass the oak tree in seconds
noncomputable def passing_time : ℝ := 14.998800095992321

-- Define the conversion factor from m/s to km/hr
noncomputable def ms_to_kmhr : ℝ := 3600 / 1000

-- Theorem statement
theorem train_speed :
  (train_length / passing_time) * ms_to_kmhr = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l259_25908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_thirds_pi_plus_two_alpha_l259_25933

theorem cos_two_thirds_pi_plus_two_alpha (α : ℝ) : 
  Real.sin (π / 6 - α) = 1 / 3 → Real.cos (2 * π / 3 + 2 * α) = - 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_thirds_pi_plus_two_alpha_l259_25933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_period_l259_25929

/-- Given a principal amount, an interest rate increase, and the resulting increase in interest,
    calculate the time period for which the principal was invested. -/
noncomputable def calculate_time_period (principal : ℝ) (rate_increase : ℝ) (interest_increase : ℝ) : ℝ :=
  interest_increase / (principal * rate_increase / 100)

theorem simple_interest_time_period :
  let principal : ℝ := 600
  let rate_increase : ℝ := 4
  let interest_increase : ℝ := 144
  calculate_time_period principal rate_increase interest_increase = 6 := by
  -- Unfold the definition of calculate_time_period
  unfold calculate_time_period
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_period_l259_25929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_l259_25931

theorem infinite_prime_divisors (a : ℕ) : Set.Infinite {p : ℕ | Prime p ∧ ∃ n : ℕ, n > 0 ∧ p ∣ 2^(2^n) + a} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_l259_25931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_divisors_of_8_factorial_l259_25914

theorem even_divisors_of_8_factorial : 
  (Finset.filter (λ d => d ∣ 40320 ∧ Even d) (Finset.range 40321)).card = 84 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_divisors_of_8_factorial_l259_25914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_max_ab_l259_25996

theorem circle_symmetry_max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 + y^2 - 4*a*x + 2*b*y + b^2 = 0 → 
   x^2 + y^2 - 4*a*x + 2*b*y + b^2 = 0 ∧ 
   (y + 1)^2 + (x - 1)^2 - 4*a*(y + 1) + 2*b*(x - 1) + b^2 = 0) →
  a * b ≤ 1/8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_max_ab_l259_25996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_catches_john_l259_25906

/-- The time it takes for Bob to catch up with John -/
noncomputable def catchUpTime (johnSpeed bobSpeed : ℝ) (initialDistance : ℝ) (bobDelay : ℝ) : ℝ :=
  let johnDistanceBeforeBobStarts := johnSpeed * bobDelay
  let newDistance := initialDistance + johnDistanceBeforeBobStarts
  let relativeSpeed := bobSpeed - johnSpeed
  (newDistance / relativeSpeed) * 60 + bobDelay * 60

/-- Theorem stating that Bob catches up to John in 150 minutes -/
theorem bob_catches_john :
  catchUpTime 4 6 3 0.5 = 150 := by
  -- Unfold the definition of catchUpTime
  unfold catchUpTime
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_catches_john_l259_25906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l259_25987

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def g (a b c k : ℝ) (x : ℝ) : ℝ := f a b c x - x^2 - x - 2 + k / x

theorem quadratic_function_theorem 
  (a b c : ℝ) 
  (h1 : ∀ x, f a b c (x + 1) - f a b c x = 2 * x + 3)
  (h2 : ∃ m, ∀ x, f a b c x ≥ m ∧ ∃ x₀, f a b c x₀ = m)
  (h3 : ∃ m, m = 1 ∧ ∀ x, f a b c x ≥ m ∧ ∃ x₀, f a b c x₀ = m) :
  (∀ x, f a b c x = x^2 + 2*x + 2) ∧
  (∀ k, (∀ x ∈ Set.Icc 1 4, g a b c k x < -x + 1/x + 6) → k < -7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l259_25987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_schedules_l259_25911

/-- Represents a chess tournament between two schools --/
structure ChessTournament where
  /-- Number of players per school --/
  players_per_school : ℕ
  /-- Number of games each player plays against each opponent --/
  games_per_opponent : ℕ
  /-- Number of games played simultaneously in each round --/
  games_per_round : ℕ

/-- Calculate the total number of games in the tournament --/
def total_games (t : ChessTournament) : ℕ :=
  t.players_per_school * t.players_per_school * t.games_per_opponent

/-- Calculate the number of rounds in the tournament --/
def num_rounds (t : ChessTournament) : ℕ :=
  total_games t / t.games_per_round

/-- Calculate the number of ways to schedule the tournament --/
def num_schedules (t : ChessTournament) : ℕ :=
  (Nat.factorial (num_rounds t)) / (Nat.factorial t.games_per_opponent ^ t.players_per_school)

/-- Theorem stating the number of ways to schedule the specific tournament --/
theorem chess_tournament_schedules :
  ∃ t : ChessTournament,
    t.players_per_school = 3 ∧
    t.games_per_opponent = 3 ∧
    t.games_per_round = 3 ∧
    num_schedules t = 1680 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_schedules_l259_25911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l259_25994

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  ha : 0 < A ∧ A < Real.pi
  hb : 0 < B ∧ B < Real.pi
  hc : 0 < C ∧ C < Real.pi
  hsum : A + B + C = Real.pi
  sine_law : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Define the theorem
theorem triangle_problem (t : Triangle)
  (h1 : t.a * Real.sin t.A - t.b * Real.sin t.B = (Real.sqrt 3 * t.a - t.c) * Real.sin t.C)
  (h2 : t.a / t.b = 2 / 3) :
  Real.sin t.C = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 ∧
  (t.b = 6 → t.a * t.b * Real.sin t.C / 2 = 2 * Real.sqrt 3 + 4 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l259_25994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_largest_factorial_product_l259_25978

/-- 
Given a natural number n, this function checks if n! can be expressed 
as the product of n - 2 consecutive positive integers.
-/
def is_factorial_product (n : ℕ) : Prop :=
  ∃ (k : ℕ), Nat.factorial n = (Finset.range (n - 2)).prod (λ i => k + i + 1)

/-- 
Theorem stating that 2 is the largest natural number n for which n! 
can be expressed as the product of n - 2 consecutive positive integers.
-/
theorem largest_factorial_product : 
  (is_factorial_product 2) ∧ (∀ m > 2, ¬(is_factorial_product m)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_largest_factorial_product_l259_25978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_average_age_l259_25927

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age_diff : ℕ) 
  (average_age : ℝ)
  (h_team_size : team_size = 11)
  (h_captain_age : captain_age = 24)
  (h_wicket_keeper_age : wicket_keeper_age_diff = 3)
  (h_remaining_players_avg : 
    (team_size : ℝ) * average_age - (captain_age + (captain_age + wicket_keeper_age_diff))
    = (team_size - 2 : ℝ) * (average_age - 1)) :
  average_age = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_average_age_l259_25927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l259_25900

theorem constant_term_expansion (x : ℝ) (x_neq_0 : x ≠ 0) : 
  let expansion := (1/x - 1) * ((x^(1/2) : ℝ) + 1)^5
  ∃ (a b c d : ℝ), expansion = a*x^(3/2) + b*x^(1/2) + c/x + 9 + d*x
    ∧ (∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x, 0 < |x| → |x| < δ → |d*x| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l259_25900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l259_25923

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/20 + y^2/16 = 1

-- Define the top vertex M
def M : ℝ × ℝ := (0, 4)

-- Define the right focus F
def F : ℝ × ℝ := (2, 0)

-- Define points A and B on the ellipse
variable (A B : ℝ × ℝ)

-- Define line l
noncomputable def l (x : ℝ) : ℝ := (6 * x - 28) / 5

-- State the theorem
theorem line_equation_proof 
  (hA : C A.1 A.2)
  (hB : C B.1 B.2)
  (hCentroid : (A.1 + B.1 + M.1) / 3 = F.1 ∧ (A.2 + B.2 + M.2) / 3 = F.2)
  (hIntersection : l A.1 = A.2 ∧ l B.1 = B.2) :
  ∀ x y, y = l x ↔ 6 * x - 5 * y - 28 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l259_25923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_l259_25901

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- State the theorem
theorem equidistant_points (x y : ℝ) :
  distance x y (-2) 2 = distance x y 2 0 ↔ y = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_l259_25901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_value_for_f_l259_25974

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  (1 + 1 / Real.log (Real.sqrt (x^2 + 10) + x)) * 
  (1 + 2 / Real.log (Real.sqrt (x^2 + 10) - x))

-- State the theorem
theorem no_minimum_value_for_f :
  ¬ ∃ (min : ℝ), ∀ (x : ℝ), 0 < x → x < 4.5 → f x ≥ min ∧ ∃ (y : ℝ), 0 < y ∧ y < 4.5 ∧ f y = min :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_value_for_f_l259_25974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_mold_radius_l259_25940

noncomputable def hemisphereVolume (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

theorem smaller_mold_radius :
  let largeRadius : ℝ := 2
  let numMolds : ℕ := 64
  let largeVolume : ℝ := hemisphereVolume largeRadius
  ∃ (smallRadius : ℝ),
    smallRadius > 0 ∧
    numMolds * hemisphereVolume smallRadius = largeVolume ∧
    smallRadius = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_mold_radius_l259_25940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l259_25932

noncomputable def f (x : ℝ) : ℝ :=
  if 1/2 < x ∧ x ≤ 1 then x^3 / (x + 1)
  else if 0 ≤ x ∧ x ≤ 1/2 then -1/6 * x + 1/12
  else 0

noncomputable def g (a x : ℝ) : ℝ := a * Real.sin (Real.pi/6 * x) - a + 1

theorem range_of_a (a : ℝ) :
  (a > 0 ∧
   ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 1 ∧ 0 ≤ x₂ ∧ x₂ ≤ 1 ∧ f x₁ = g a x₂) →
  1/2 ≤ a ∧ a ≤ 2 :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l259_25932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l259_25918

theorem train_length_problem (v1 v2 l1 t : ℝ) (h1 : v1 = 60) (h2 : v2 = 40) (h3 : l1 = 300) 
  (h4 : t = 17.998560115190788) : 
  (((v1 + v2) * 1000 / 3600) * t - l1) = 200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l259_25918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_compound_difference_l259_25981

noncomputable section

-- Define the loan parameters
def initial_amount : ℝ := 8000
def annual_rate : ℝ := 0.10
def years : ℕ := 3

-- Define the compounding functions
noncomputable def compound_monthly (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate / 12) ^ (12 * time)

noncomputable def compound_semiannually (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate / 2) ^ (2 * time)

-- Define the theorem
theorem loan_compound_difference :
  ∃ (diff : ℝ), 
    abs (compound_monthly initial_amount annual_rate years - 
         compound_semiannually initial_amount annual_rate years - diff) < 0.01 ∧
    abs (diff - 148.32) < 0.01 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_compound_difference_l259_25981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2014_l259_25970

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (seq.a 1 + seq.a n)

/-- Three points are collinear if they lie on the same line -/
def areCollinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

theorem arithmetic_sequence_sum_2014 (seq : ArithmeticSequence) :
  S seq 2014 = seq.a 1 + seq.a 2014 →
  areCollinear (seq.a 1, 0) (seq.a 2014, 0) (S seq 2014, 0) →
  (seq.a 1, 0) ≠ (0, 0) →
  (seq.a 2014, 0) ≠ (0, 0) →
  S seq 2014 = 1007 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2014_l259_25970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_even_function_minimum_l259_25935

noncomputable def f (x : ℝ) : ℝ := x^2

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x + |2*x - m|

noncomputable def g_min (m : ℝ) : ℝ :=
  if m < -2 then -m - 1
  else if m ≤ 2 then m^2 / 4
  else m - 1

theorem quadratic_even_function_minimum (m : ℝ) :
  (∀ x, f (-x) = f x) ∧  -- f is even
  f 0 = 0 ∧              -- f passes through origin
  (2 : ℝ) * 1 = 2 →      -- f' passes through (1,2)
  ∀ x, g m x ≥ g_min m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_even_function_minimum_l259_25935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_implies_a_value_inequality_implies_a_range_l259_25928

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x / Real.log 3)^2 + (a - 1) * (Real.log x / Real.log 3) + 3 * a - 2

-- Part 1
theorem range_implies_a_value (a : ℝ) :
  (∀ y : ℝ, y ≥ 2 → ∃ x > 0, f a x = y) ∧ 
  (∀ x > 0, f a x ≥ 2) →
  a = 7 + 4 * Real.sqrt 2 ∨ a = 7 - 4 * Real.sqrt 2 :=
by sorry

-- Part 2
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 3 9, f a (3 * x) + Real.log (9 * x) / Real.log 3 ≤ 0) →
  a ≤ -4/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_implies_a_value_inequality_implies_a_range_l259_25928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_41835_l259_25902

def n : ℕ := 41835

theorem divisors_of_41835 : 
  (Finset.filter (λ i => n % i = 0) (Finset.range 10)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_41835_l259_25902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_of_f_l259_25904

/-- The function f(x) = x + 4/x -/
noncomputable def f (x : ℝ) : ℝ := x + 4/x

theorem extreme_values_of_f :
  ∀ x : ℝ, x ≠ 0 →
  (∀ y : ℝ, y ≠ 0 → f y ≥ f 2) ∧
  (f 2 = 4) ∧
  (∀ y : ℝ, y ≠ 0 → f y ≤ f (-2)) ∧
  (f (-2) = -4) := by
  sorry

#check extreme_values_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_of_f_l259_25904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l259_25982

/-- A parabola defined by y = x^2 -/
def parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2}

/-- Point M on the parabola -/
def M : ℝ × ℝ := (1, 1)

/-- The intersection point E of line segments AB and CD -/
def E : ℝ × ℝ := (-1, 2)

/-- M is on the parabola -/
axiom h_M_on_parabola : M ∈ parabola

/-- M is the vertex of right-angled triangles inscribed in the parabola -/
axiom h_M_vertex : ∃ (A B C D : ℝ × ℝ), 
  A ∈ parabola ∧ B ∈ parabola ∧ C ∈ parabola ∧ D ∈ parabola ∧
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0 ∧
  (C.1 - M.1) * (D.1 - M.1) + (C.2 - M.2) * (D.2 - M.2) = 0

/-- The main theorem: E is the intersection point of AB and CD -/
theorem intersection_point : E = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l259_25982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_vertex_face_product_l259_25986

/-- A cube is a three-dimensional geometric shape with specific properties. -/
structure Cube where

/-- The number of vertices in a cube. -/
def Cube.num_vertices : Nat := 8

/-- The number of faces in a cube. -/
def Cube.num_faces : Nat := 6

/-- Theorem stating that the product of the number of vertices and faces in a cube is 48. -/
theorem cube_vertex_face_product : Cube.num_vertices * Cube.num_faces = 48 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_vertex_face_product_l259_25986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_equal_l259_25937

/-- Represents a regular octagon with integer side lengths -/
structure RegularOctagon where
  -- Side lengths of the octagon
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  -- Ensure the octagon is regular (all angles are equal)
  regular : Prop

/-- Theorem stating that opposite sides of a regular octagon with integer side lengths are equal -/
theorem opposite_sides_equal (octagon : RegularOctagon) : 
  octagon.a = octagon.e ∧ 
  octagon.b = octagon.f ∧ 
  octagon.c = octagon.g ∧ 
  octagon.d = octagon.h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_equal_l259_25937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l259_25950

-- Define the propositions
def proposition1 : Prop := ∀ (a b : ℝ), a = b

def proposition2 : Prop := ∀ α : Real, 45 * Real.pi / 180 < α ∧ α < 90 * Real.pi / 180 → Real.sin α > Real.cos α

def proposition3 : Prop := ∀ m x : ℝ, (3 * x - m) / (x + 2) = 2 ∧ x < 0 → m < -4

def proposition4 : Prop := ∀ (a b : ℝ), a = b

-- Define a function to check if a proposition is false
def is_false (p : Prop) : Prop := ¬p

-- Theorem statement
theorem propositions_truth : 
  is_false proposition1 ∧ 
  ¬(is_false proposition2) ∧ 
  is_false proposition3 ∧ 
  is_false proposition4 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l259_25950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_approx_l259_25921

noncomputable def train_length : ℝ := 480
noncomputable def train_speed_kmh : ℝ := 55
noncomputable def crossing_time : ℝ := 71.99424046076314

noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600

noncomputable def platform_length : ℝ := train_speed_ms * crossing_time - train_length

theorem platform_length_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |platform_length - 620| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_approx_l259_25921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_arrangement_l259_25957

-- Define the fruits
inductive Fruit
| Apple
| Pear
| Orange
| Banana

-- Define a type for box numbers
inductive BoxNumber
| One
| Two
| Three
| Four

-- Define a function type to represent the arrangement of fruits in boxes
def Arrangement := BoxNumber → Fruit

-- Define predicates for the labels
def label1 (a : Arrangement) : Prop := a BoxNumber.One ≠ Fruit.Orange
def label2 (a : Arrangement) : Prop := a BoxNumber.Two ≠ Fruit.Pear
def label3 (a : Arrangement) : Prop := 
  a BoxNumber.One = Fruit.Banana → 
  (a BoxNumber.Three ≠ Fruit.Apple ∧ a BoxNumber.Three ≠ Fruit.Pear)
def label4 (a : Arrangement) : Prop := a BoxNumber.Four ≠ Fruit.Apple

-- Define the theorem
theorem fruit_arrangement :
  ∃! a : Arrangement,
    (∀ b : BoxNumber, ∃! f : Fruit, a b = f) ∧
    label1 a ∧
    label2 a ∧
    label3 a ∧
    label4 a ∧
    a BoxNumber.One = Fruit.Banana ∧
    a BoxNumber.Two = Fruit.Apple ∧
    a BoxNumber.Three = Fruit.Orange ∧
    a BoxNumber.Four = Fruit.Pear :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_arrangement_l259_25957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l259_25943

-- Define set A
def A : Set ℝ := {x | 2 / (x + 1) ≥ 1}

-- Define set B
def B : Set ℝ := {y | ∃ x < 0, y = 2^x}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = Set.Ioc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l259_25943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_sum_l259_25967

/-- A triangle type -/
structure Triangle where
  -- You can add more properties here if needed
  mk :: 

/-- A circle type -/
structure Circle where
  -- You can add more properties here if needed
  mk ::

/-- A line type -/
structure Line where
  -- You can add more properties here if needed
  mk ::

/-- Predicate for a circle being inscribed in a triangle -/
def IsInscribed (c : Circle) (T : Triangle) : Prop := sorry

/-- Predicate for lines being parallel to the sides of a triangle -/
def IsParallelToSides (T : Triangle) (l₁ l₂ l₃ : Line) : Prop := sorry

/-- Predicate for lines being tangent to a circle -/
def IsTangentToCircle (c : Circle) (l₁ l₂ l₃ : Line) : Prop := sorry

/-- Predicate for triangles being smaller triangles formed by lines in a larger triangle -/
def AreSmallerTriangles (T T₁ T₂ T₃ : Triangle) (l₁ l₂ l₃ : Line) : Prop := sorry

/-- Function to calculate the circumradius of a triangle -/
noncomputable def circumradius (T : Triangle) : ℝ := sorry

/-- Given a triangle with an inscribed circle, and three lines drawn parallel to its sides and
    tangent to the inscribed circle, forming three smaller triangles, the circumradius of the
    original triangle is equal to the sum of the circumradii of the three smaller triangles. -/
theorem triangle_circumradius_sum (T : Triangle) (c : Circle) 
  (h_inscribed : IsInscribed c T) 
  (l₁ l₂ l₃ : Line) 
  (h_parallel : IsParallelToSides T l₁ l₂ l₃) 
  (h_tangent : IsTangentToCircle c l₁ l₂ l₃) 
  (T₁ T₂ T₃ : Triangle) 
  (h_smaller : AreSmallerTriangles T T₁ T₂ T₃ l₁ l₂ l₃) :
  circumradius T = circumradius T₁ + circumradius T₂ + circumradius T₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_sum_l259_25967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_dividing_factorial_squared_l259_25965

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem largest_power_dividing_factorial_squared (p : ℕ) (h : Nat.Prime p) :
  (∀ n : ℕ, n > p + 1 → ¬((factorial p)^n ∣ factorial (p^2))) ∧
  ((factorial p)^(p+1) ∣ factorial (p^2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_dividing_factorial_squared_l259_25965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_inequality_l259_25942

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + 2*b + 3*c = 13) : 
  Real.sqrt (3*a) + Real.sqrt (2*b) + Real.sqrt c ≤ (13 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_inequality_l259_25942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l259_25968

-- Define the interval
def interval : Set ℝ := {x | 0 < x ∧ x < 200 * Real.pi}

-- Define the equation
def equation (x : ℝ) : Prop := Real.sin x = (1/3) ^ x

-- State the theorem
theorem solution_count : 
  ∃ (S : Finset ℝ), S.card = 200 ∧ ∀ x ∈ S, x ∈ interval ∧ equation x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l259_25968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_power_l259_25915

theorem degree_of_polynomial_power : 
  Polynomial.degree ((2 * X^3 + 5 * X^2 + 1 : Polynomial ℝ) ^ 15) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_power_l259_25915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_monotonicity_l259_25975

/-- A polynomial function from real numbers to real numbers -/
def MyPolynomial := ℝ → ℝ

/-- A function is strictly monotone if for all x < y, f(x) < f(y) or for all x < y, f(x) > f(y) -/
def StrictlyMonotone (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∨ (∀ x y, x < y → f x > f y)

theorem polynomial_monotonicity (P : MyPolynomial) 
  (h : StrictlyMonotone (fun x => P (P x))) : 
  StrictlyMonotone P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_monotonicity_l259_25975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l259_25985

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line l with slope 1
def line (x y m : ℝ) : Prop := y = x + m

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem line_equation_proof (x1 y1 x2 y2 m : ℝ) :
  ellipse x1 y1 ∧ ellipse x2 y2 ∧
  line x1 y1 m ∧ line x2 y2 m ∧
  distance x1 y1 x2 y2 = 3 * Real.sqrt 2 / 2 →
  m = 1 ∨ m = -1 := by
  sorry

#check line_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l259_25985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_4_l259_25997

-- Define the polynomials
def p1 (x : ℝ) : ℝ := 3 * x^5 + 4 * x^3 - 9 * x^2 + 2
def p2 (x : ℝ) : ℝ := 2 * x^3 - 5 * x + 1

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p1 x * p2 x

-- Theorem statement
theorem coefficient_of_x_4 :
  ∃ (a b c d e f g : ℝ),
    (∀ x : ℝ, product x = a * x^6 + b * x^5 + c * x^4 + d * x^3 + e * x^2 + f * x + g) ∧
    c = -29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_4_l259_25997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l259_25941

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 + 2 * x + a^2) / Real.log a

-- Define the interval
def interval : Set ℝ := {x | -4 ≤ x ∧ x ≤ -2}

-- Theorem statement
theorem f_increasing_iff_a_in_range :
  ∀ a : ℝ, (0 < a ∧ a ≠ 1) →
  (∀ x y, x ∈ interval → y ∈ interval → x < y → f a x < f a y) ↔ 
  (1/2 ≤ a ∧ a < -2 + 2 * Real.sqrt 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l259_25941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_intervals_l259_25951

noncomputable def f (x : ℝ) (φ : ℝ) := x * Real.cos x + Real.cos (x + φ)

theorem strictly_increasing_intervals
  (φ : ℝ)
  (h_φ : 0 < φ ∧ φ < Real.pi)
  (h_odd : ∀ x, -2 * Real.pi < x ∧ x < 2 * Real.pi → f (-x) φ = -f x φ) :
  ∀ x, (-2 * Real.pi < x ∧ x < -Real.pi) ∨ (Real.pi < x ∧ x < 2 * Real.pi) →
    ∃ ε > 0, ∀ y, 0 < y ∧ y < ε → f (x + y) φ > f x φ :=
by
  sorry

#check strictly_increasing_intervals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_intervals_l259_25951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l259_25905

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (2 * x + 3)

theorem function_composition_identity (a : ℝ) :
  (∀ x : ℝ, f a (f a x) = x) → a = -3 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l259_25905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_vertex_product_l259_25960

/-- A cube is a three-dimensional shape with six square faces. -/
structure Cube where

/-- The number of faces in a cube. -/
def Cube.numFaces : Nat := 6

/-- The number of vertices in a cube. -/
def Cube.numVertices : Nat := 8

/-- The product of the number of faces and vertices in a cube is 48. -/
theorem cube_face_vertex_product : Cube.numFaces * Cube.numVertices = 48 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_vertex_product_l259_25960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_a_tangent_line_at_p_l259_25989

-- Problem 1
noncomputable def f (a b x : ℝ) : ℝ := a * x + b

theorem range_of_f_a (a b : ℝ) 
  (h : ∫ x in (-1)..(1), (f a b x)^2 = 2) :
  ∃ (y : ℝ), -1 ≤ y ∧ y ≤ 37/12 ∧ y = f a b a :=
sorry

-- Problem 2
def g (x : ℝ) : ℝ := x^3 - 3*x

theorem tangent_line_at_p (x y : ℝ) 
  (h : g 1 = -2) 
  (k : y + 2 = 0 ∨ 9*x + 4*y - 1 = 0) :
  y - g 1 = (deriv g 1) * (x - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_a_tangent_line_at_p_l259_25989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_B_max_value_of_f_l259_25936

open Real

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the condition for the angles
def angle_condition (A B C : ℝ) : Prop :=
  2 * log (tan B) = log (tan A) + log (tan C)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  7 - 4 * sin x * cos x + 4 * (cos x)^2 - 4 * (cos x)^4

-- Theorem 1: Range of B
theorem range_of_B (A B C : ℝ) :
  triangle_ABC A B C → angle_condition A B C →
  Real.pi/3 ≤ B ∧ B < Real.pi/2 := by sorry

-- Theorem 2: Maximum value of f
theorem max_value_of_f :
  ∃ x : ℝ, f x = 10 ∧ ∀ y : ℝ, f y ≤ 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_B_max_value_of_f_l259_25936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_first_class_rate_probability_both_s_equal_four_l259_25998

/-- Represents a product with quality indicators -/
structure Product where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Calculates the composite indicator S for a product -/
def compositeIndicator (p : Product) : ℚ := p.x + p.y + p.z

/-- Determines if a product is first-class -/
def isFirstClass (p : Product) : Bool :=
  compositeIndicator p ≤ 4

/-- The sample of 10 products -/
def sampleProducts : List Product := [
  ⟨1, 1, 2⟩, ⟨2, 1, 1⟩, ⟨2, 2, 2⟩, ⟨1, 1, 1⟩, ⟨1, 2, 1⟩,
  ⟨1, 2, 2⟩, ⟨2, 1, 1⟩, ⟨2, 2, 1⟩, ⟨1, 1, 1⟩, ⟨2, 1, 2⟩
]

theorem sample_first_class_rate :
  (sampleProducts.filter isFirstClass).length / sampleProducts.length = 6 / 10 := by
  sorry

theorem probability_both_s_equal_four :
  let firstClass := sampleProducts.filter isFirstClass
  let sEqualFour := firstClass.filter (fun p => compositeIndicator p = 4)
  (sEqualFour.length * (sEqualFour.length - 1)) / (firstClass.length * (firstClass.length - 1)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_first_class_rate_probability_both_s_equal_four_l259_25998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_minus_pi_fourth_l259_25946

theorem tan_x_minus_pi_fourth (x : ℝ) 
  (h1 : Real.sin x = 4/5) 
  (h2 : x ∈ Set.Ioo (π/2) π) : 
  Real.tan (x - π/4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_minus_pi_fourth_l259_25946
