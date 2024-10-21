import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_stationary_points_order_l494_49484

-- Define the concept of "new stationary point"
def new_stationary_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = (deriv (deriv f)) x

-- Define the functions
noncomputable def g (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := Real.log x
def φ (x : ℝ) : ℝ := x^3

-- State the theorem
theorem new_stationary_points_order :
  ∃ (a b c : ℝ),
    (0 < a ∧ a < Real.pi) ∧
    (0 < b) ∧
    (c ≠ 0) ∧
    new_stationary_point g a ∧
    new_stationary_point h b ∧
    new_stationary_point φ c ∧
    c > b ∧ b > a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_stationary_points_order_l494_49484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vartan_recreation_spending_l494_49418

/-- Represents Vartan's wages from last week -/
noncomputable def last_week_wages : ℝ := 1

/-- Percentage of wages spent on recreation last week -/
noncomputable def last_week_recreation_percentage : ℝ := 0.20

/-- Percentage decrease in wages this week -/
noncomputable def wage_decrease_percentage : ℝ := 0.20

/-- Percentage increase in recreation spending this week compared to last week -/
noncomputable def recreation_increase_percentage : ℝ := 1.60

/-- Calculates the percentage of wages spent on recreation this week -/
noncomputable def this_week_recreation_percentage : ℝ :=
  (recreation_increase_percentage * last_week_recreation_percentage) /
  (1 - wage_decrease_percentage)

theorem vartan_recreation_spending :
  this_week_recreation_percentage = 0.40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vartan_recreation_spending_l494_49418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l494_49489

/-- The time it takes for two trains to completely pass each other -/
noncomputable def train_passing_time (l1 l2 v1 v2 angle : ℝ) : ℝ :=
  let v1_ms := v1 * 1000 / 3600
  let v2_ms := v2 * 1000 / 3600
  let v1_component := v1_ms * Real.cos (angle * Real.pi / 180)
  let v2_component := v2_ms * Real.cos (angle * Real.pi / 180)
  let relative_speed := v1_component + v2_component
  let total_length := l1 + l2
  total_length / relative_speed

/-- Theorem stating that the time for two specific trains to pass each other is approximately 12.99 seconds -/
theorem train_passing_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_passing_time 140 160 57.3 38.7 30 - 12.99| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l494_49489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_range_theorem_l494_49442

noncomputable section

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def minorAxisLength (b : ℝ) : ℝ := 2 * b

def triangleF1ABArea (a c : ℝ) : ℝ := (a - c) / 2

theorem ellipse_range_theorem (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : minorAxisLength b = 2) (h4 : triangleF1ABArea a c = (2 - Real.sqrt 3) / 2) :
  ∀ P : ℝ × ℝ, ellipse a b P.1 P.2 →
    (1 : ℝ) ≤ (1 / |P.1 + c| + 1 / |P.1 - c|) ∧
    (1 / |P.1 + c| + 1 / |P.1 - c|) ≤ 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_range_theorem_l494_49442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_theorem_l494_49463

/-- Represents the distance traveled by a car in yards -/
noncomputable def distance_in_yards (b t : ℝ) : ℝ := 25 * b / (12 * t)

/-- Theorem stating that the distance traveled by the car in 5 minutes is equal to the calculated formula -/
theorem car_distance_theorem (b t : ℝ) (hb : b > 0) (ht : t > 0) :
  let inches_per_t_seconds := b / 4
  let seconds_in_5_minutes := 5 * 60
  let inches_in_5_minutes := (inches_per_t_seconds / t) * seconds_in_5_minutes
  let yards_in_5_minutes := inches_in_5_minutes / 36
  yards_in_5_minutes = distance_in_yards b t := by
  sorry

#check car_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_theorem_l494_49463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_l494_49466

/-- The number of ordered pairs of positive integers (x, y) satisfying xy = 3430 -/
def num_ordered_pairs : ℕ := 16

/-- The prime factorization of 3430 -/
def prime_factorization : List ℕ := [2, 5, 7, 7, 7]

theorem count_ordered_pairs :
  (∀ (x y : ℕ), x * y = 3430 ↔ (x, y) ∈ (Finset.filter (fun p => p.1 * p.2 = 3430) (Finset.product (Finset.range (3430 + 1)) (Finset.range (3430 + 1))))) ∧
  (prime_factorization.count 2 = 1 ∧
   prime_factorization.count 5 = 1 ∧
   prime_factorization.count 7 = 3) →
  Finset.card (Finset.filter (fun p => p.1 * p.2 = 3430) (Finset.product (Finset.range (3430 + 1)) (Finset.range (3430 + 1)))) = num_ordered_pairs :=
by
  sorry

#eval num_ordered_pairs
#eval prime_factorization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_l494_49466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_telephone_number_theorem_l494_49441

def TelephoneNumber := Fin 10 → Fin 10

def isDecreasing (x y z : Fin 10) : Prop := x > y ∧ y > z

def isDecreasing4 (w x y z : Fin 10) : Prop := w > x ∧ x > y ∧ y > z

def areConsecutiveEven (x y z : Fin 10) : Prop :=
  x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧ x = y + 2 ∧ y = z + 2

def areConsecutiveOdd (w x y z : Fin 10) : Prop :=
  w % 2 = 1 ∧ x % 2 = 1 ∧ y % 2 = 1 ∧ z % 2 = 1 ∧
  w = x + 2 ∧ x = y + 2 ∧ y = z + 2

theorem telephone_number_theorem (t : TelephoneNumber) :
  (∀ i j, i ≠ j → t i ≠ t j) →
  isDecreasing (t 0) (t 1) (t 2) →
  isDecreasing (t 3) (t 4) (t 5) →
  isDecreasing4 (t 6) (t 7) (t 8) (t 9) →
  areConsecutiveEven (t 3) (t 4) (t 5) →
  areConsecutiveOdd (t 6) (t 7) (t 8) (t 9) →
  (t 0).val + (t 1).val + (t 2).val = 9 →
  (t 0).val = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_telephone_number_theorem_l494_49441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_product_l494_49433

/-- Given two single-digit natural numbers A and B, 
    the sum of the digits in the product of (A * 111111111) and (B * 111111111) is 81. -/
theorem sum_of_digits_product (A B : ℕ) (hA : A < 10) (hB : B < 10) :
  (Nat.digits 10 ((A * 111111111) * (B * 111111111))).sum = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_product_l494_49433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_6_magnitude_l494_49448

/-- Recursive definition of the sequence z_n -/
def z : ℕ → ℂ
  | 0 => 1
  | n + 1 => (z n)^2 - 1 + Complex.I

/-- The magnitude of the 6th term of the sequence is 291 -/
theorem z_6_magnitude : Complex.abs (z 6) = 291 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_6_magnitude_l494_49448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_y_coordinates_specific_rectangle_l494_49474

/-- A rectangle in a 2D plane --/
structure Rectangle where
  a : ℝ × ℝ
  c : ℝ × ℝ

/-- The sum of y-coordinates of the other two vertices of a rectangle --/
noncomputable def sum_other_y_coordinates (r : Rectangle) : ℝ :=
  2 * ((r.a.2 + r.c.2) / 2)

/-- Theorem: For a rectangle with opposite vertices (4,20) and (10,-6),
    the sum of y-coordinates of the other two vertices is 14 --/
theorem sum_y_coordinates_specific_rectangle :
  let r : Rectangle := { a := (4, 20), c := (10, -6) }
  sum_other_y_coordinates r = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_y_coordinates_specific_rectangle_l494_49474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l494_49468

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x else x^3 - 1/x + 1

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  f (6 - x^2) > f x ↔ -3 < x ∧ x < 2 :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l494_49468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circles_for_regular_2011_gon_l494_49455

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

/-- A regular polygon --/
structure RegularPolygon where
  sides : ℕ
  center : ℝ × ℝ
  radius : ℝ

/-- Defines if a line is tangent to a circle --/
def is_tangent (l : Line) (c : Circle) : Prop := sorry

/-- Defines if a line is a side of a regular polygon --/
def is_side (l : Line) (p : RegularPolygon) : Prop := sorry

/-- The main theorem --/
theorem min_circles_for_regular_2011_gon :
  ∀ (circles : List Circle) (polygon : RegularPolygon),
    polygon.sides = 2011 →
    (∀ l : Line, is_side l polygon → 
      ∃ c1 c2 : Circle, c1 ∈ circles ∧ c2 ∈ circles ∧ c1 ≠ c2 ∧ 
        is_tangent l c1 ∧ is_tangent l c2) →
    circles.length ≥ 504 := by
  sorry

#check min_circles_for_regular_2011_gon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circles_for_regular_2011_gon_l494_49455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l494_49469

theorem trig_problem (α : ℝ) (h1 : Real.cos α = -4/5) (h2 : π < α ∧ α < 3*π/2) :
  (Real.sin α = 3/5) ∧ 
  (Real.tan α = -3/4) ∧ 
  ((2*Real.sin α + 3*Real.cos α) / (Real.cos α - Real.sin α) = 6/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l494_49469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graveyard_bone_difference_l494_49486

theorem graveyard_bone_difference : 
  ∀ (total_skeletons : ℕ) 
    (total_bones : ℕ) 
    (woman_bones : ℕ),
  total_skeletons = 20 →
  total_bones = 375 →
  woman_bones = 20 →
  let adult_women := total_skeletons / 2
  let adult_men := (total_skeletons - adult_women) / 2
  let children := (total_skeletons - adult_women) / 2
  let child_bones := woman_bones / 2
  let total_woman_bones := adult_women * woman_bones
  let total_child_bones := children * child_bones
  let remaining_bones := total_bones - total_woman_bones - total_child_bones
  let man_bones := remaining_bones / adult_men
  man_bones - woman_bones = 5 :=
by
  intro total_skeletons total_bones woman_bones h1 h2 h3
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graveyard_bone_difference_l494_49486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_growth_l494_49436

/-- Represents the annual increase in height of a tree -/
def annual_increase : ℝ → Prop := sorry

theorem tree_growth (h : ℝ) : 
  annual_increase h →
  (4 + 6 * h = (4 + 4 * h) * (5/4)) →
  h = 1 := by
  intro h_annual h_equation
  -- The proof steps would go here
  sorry

#check tree_growth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_growth_l494_49436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l494_49408

noncomputable def original_function (x : Real) : Real := Real.sin (-2 * x)

noncomputable def target_function (x : Real) : Real := Real.sin (-2 * x + Real.pi / 4) + 2

noncomputable def shifted_function (x : Real) : Real := original_function (x - Real.pi / 8) + 2

theorem function_transformation :
  ∀ x : Real, shifted_function x = target_function x := by
  intro x
  simp [shifted_function, original_function, target_function]
  -- The proof steps would go here
  sorry

#check function_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l494_49408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_and_sum_l494_49459

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 4 * x - 10) / (x - 5)

theorem slant_asymptote_and_sum (m b : ℝ) :
  (∀ ε > 0, ∃ M, ∀ x, abs x > M → abs (f x - (m * x + b)) < ε) →
  m = 3 ∧ b = 19 ∧ m + b = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_and_sum_l494_49459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_f_l494_49415

noncomputable def f (x : ℝ) := 3 - Real.sin x - 2 * (Real.cos x)^2

theorem max_min_difference_f :
  let domain := Set.Icc (π / 6) (7 * π / 6)
  ∃ (max min : ℝ), 
    (∀ x ∈ domain, f x ≤ max) ∧ 
    (∃ x ∈ domain, f x = max) ∧
    (∀ x ∈ domain, min ≤ f x) ∧ 
    (∃ x ∈ domain, f x = min) ∧
    max - min = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_f_l494_49415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_l494_49496

/-- Conversion from rectangular to spherical coordinates -/
theorem rectangular_to_spherical 
  (x y z : ℝ) 
  (ρ θ φ : ℝ) 
  (h_x : x = -4) 
  (h_y : y = 4 * Real.sqrt 3) 
  (h_z : z = 2) 
  (h_ρ_pos : ρ > 0) 
  (h_θ_range : 0 ≤ θ ∧ θ < 2 * π) 
  (h_φ_range : 0 ≤ φ ∧ φ ≤ π) : 
  (x = ρ * Real.sin φ * Real.cos θ ∧ 
   y = ρ * Real.sin φ * Real.sin θ ∧ 
   z = ρ * Real.cos φ) ↔ 
  (ρ = 2 * Real.sqrt 17 ∧ 
   θ = 2 * π / 3 ∧ 
   φ = Real.arccos (1 / Real.sqrt 17)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_l494_49496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l494_49437

open Real

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := log (x^2 + x + sqrt (1 + (x + 1)^2))

-- Theorem stating that g is neither even nor odd
theorem g_neither_even_nor_odd :
  ¬(∀ x, g (-x) = g x) ∧ ¬(∀ x, g (-x) = -g x) :=
by
  apply And.intro
  · intro h
    -- Proof that g is not even
    sorry
  · intro h
    -- Proof that g is not odd
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l494_49437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_correct_all_lines_pass_through_C_l494_49499

/-- The fixed point C through which all lines pass -/
def C : ℝ × ℝ := (-1, 2)

/-- The line equation parameterized by a -/
def line (a : ℝ) (x y : ℝ) : Prop :=
  (a - 1) * x - y + a + 1 = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- Theorem stating that the circle equation represents the circle with center C -/
theorem circle_equation_correct :
  ∃ (r : ℝ), ∀ (x y : ℝ), 
    circle_equation x y ↔ (x - C.1)^2 + (y - C.2)^2 = r^2 :=
by
  sorry

/-- Theorem stating that all lines pass through the fixed point C -/
theorem all_lines_pass_through_C :
  ∀ (a : ℝ), line a C.1 C.2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_correct_all_lines_pass_through_C_l494_49499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023rd_term_l494_49446

def sequence_mean (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (Finset.sum (Finset.range n) a) / n = n + 1

theorem sequence_2023rd_term (a : ℕ → ℕ) (h : sequence_mean a) : a 2022 = 4046 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023rd_term_l494_49446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_guessing_probability_l494_49434

-- Define the number of questions
def num_questions : ℕ := 15

-- Define the probability of getting a single question correct
def prob_correct : ℚ := 1/2

-- Define the minimum number of correct answers needed for "at least half"
def min_correct : ℕ := 8

-- Statement to prove
theorem random_guessing_probability :
  (Finset.sum (Finset.range (num_questions - min_correct + 1))
    (λ k ↦ (num_questions.choose (min_correct + k)) * prob_correct^(min_correct + k) * (1 - prob_correct)^(num_questions - (min_correct + k)))) = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_guessing_probability_l494_49434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_at_op_equation_l494_49403

-- Define the operation @
noncomputable def at_op (a b : ℝ) : ℝ := (a ^ b) / 2

-- State the theorem
theorem solve_at_op_equation :
  ∀ x : ℝ, at_op 3 x = 4.5 → x = 2 := by
  intro x h
  -- The proof steps would go here
  sorry

#check solve_at_op_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_at_op_equation_l494_49403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_f_range_l494_49411

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

-- Theorem 1: f(x) is always increasing
theorem f_increasing (a : ℝ) : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ := by sorry

-- Theorem 2: f(x) is an odd function iff a = 1/2
theorem f_odd_iff (a : ℝ) : (∀ x : ℝ, f a (-x) = -(f a x)) ↔ a = 1/2 := by sorry

-- Theorem 3: When a = 1/2, the range of f(x) is (-1/2, 1/2)
theorem f_range : ∀ y : ℝ, (∃ x : ℝ, f (1/2) x = y) ↔ -1/2 < y ∧ y < 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_f_range_l494_49411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l494_49410

theorem trig_problem (α β : Real) 
  (h1 : 0 < α ∧ α < π/3)
  (h2 : -π/2 < β ∧ β < -π/3)
  (h3 : Real.cos (α + π/6) = Real.sqrt 3 / 3)
  (h4 : Real.sin (α - β + π/6) = 2/3) :
  Real.sin α = (3 * Real.sqrt 2 - Real.sqrt 3) / 6 ∧
  Real.cos (3*α - β) = -(2 * Real.sqrt 10 + 2) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l494_49410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l494_49460

-- Define the triangle and its properties
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi

-- Define the vectors
noncomputable def m (t : Triangle) : Real × Real := (1/2, Real.cos t.A)
noncomputable def n (t : Triangle) : Real × Real := (Real.sin t.A, -Real.sqrt 3 / 2)

-- Perpendicularity condition
def perpendicular (v w : Real × Real) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h_perp : perpendicular (m t) (n t))
  (h_a : t.a = 7)
  (h_b : t.b = 8) :
  t.A = Real.pi / 3 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l494_49460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_value_l494_49406

/-- The function f(x) = 1 - 2x - 3/x for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := 1 - 2*x - 3/x

/-- The theorem stating the maximum value of f(x) and where it occurs -/
theorem f_maximum_value :
  ∃ (x_max : ℝ), x_max > 0 ∧
    (∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
    x_max = Real.sqrt 6 / 2 ∧
    f x_max = 1 - 2 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_value_l494_49406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l494_49447

/-- Predicate to check if a point is on the right branch of the hyperbola -/
def is_right_branch (P : ℝ × ℝ) : Prop := P.1 > 0

/-- Predicate to check if a point is the left focus of the hyperbola -/
def is_left_focus (F : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is the right focus of the hyperbola -/
def is_right_focus (F : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is the incenter of a triangle -/
def is_incenter (I F₁ P F₂ : ℝ × ℝ) : Prop := sorry

/-- Function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Function to calculate the eccentricity of a hyperbola given a and b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola with the given properties is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (P F₁ F₂ I : ℝ × ℝ) : 
  a > 0 → b > 0 → 
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) → 
  is_right_branch P →
  is_left_focus F₁ →
  is_right_focus F₂ →
  is_incenter I F₁ P F₂ →
  2 * (area_triangle P F₁ I - area_triangle P F₂ I) = area_triangle F₁ F₂ I →
  eccentricity a b = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l494_49447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paths_to_church_l494_49457

/-- Represents a point on the grid --/
structure Point where
  x : Nat
  y : Nat

/-- Calculates the number of paths from (0,0) to a given point --/
def numPaths : Point → Nat
  | ⟨0, _⟩ => 1  -- First column
  | ⟨_, 0⟩ => 1  -- First row
  | ⟨x+1, y+1⟩ => numPaths ⟨x, y+1⟩ + numPaths ⟨x+1, y⟩ + numPaths ⟨x, y⟩

/-- The specific end point that represents the church --/
def churchPoint : Point := ⟨8, 6⟩  -- Example values, adjust as needed

theorem paths_to_church :
  numPaths churchPoint = 321 := by
  sorry

#eval numPaths churchPoint

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paths_to_church_l494_49457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l494_49471

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (Real.sqrt 3 * (a^2 + c^2 - b^2) = 2 * b * Real.sin A) →
  (B = π / 3 ∧ 
   (Real.cos A = 1 / 3 → Real.sin (2 * A - B) = (4 * Real.sqrt 2 + 7 * Real.sqrt 3) / 18)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l494_49471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l494_49413

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^(1/3)

-- State the theorem
theorem root_exists_in_interval :
  ∃ x ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ), f x = 0 :=
by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l494_49413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l494_49488

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

-- State the theorem
theorem min_value_of_f :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ f x₀ = 1/2 ∧ ∀ (x : ℝ), x > 0 → f x ≥ f x₀ := by
  -- Proof goes here
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l494_49488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_theorem_l494_49423

/-- A function representing the inverse variation of a with respect to b³ -/
noncomputable def inverse_variation (k : ℝ) (b : ℝ) : ℝ := k / (b^3)

theorem inverse_variation_theorem (k : ℝ) :
  inverse_variation k 1 = 5 →
  inverse_variation k 2 = 5/8 := by
  intro h
  -- Proof steps would go here
  sorry

#check inverse_variation_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_theorem_l494_49423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l494_49473

-- Define the function g(x)
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := (t - 1) * x - 4 / x

-- Define the maximum value function f(t)
noncomputable def f (t : ℝ) : ℝ :=
  if t ≤ -3 then t - 5
  else if t < 0 then -4 * Real.sqrt (1 - t)
  else 2 * t - 4

-- Theorem statement
theorem max_value_theorem (t : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 2 → g t x ≤ m) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ g t x = f t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l494_49473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_k_range_l494_49416

/-- If f(x) = (k^2 - 3k + 2)x + b is decreasing on ℝ, then 1 < k < 2 -/
theorem decreasing_function_k_range (k b : ℝ) : 
  (∀ x y : ℝ, x < y → ((k^2 - 3*k + 2)*y + b) < ((k^2 - 3*k + 2)*x + b)) → 
  1 < k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_k_range_l494_49416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l494_49421

/-- Given a natural number n, we define a set A containing elements from 1 to 2n+1 -/
def A (n : ℕ) : Finset ℕ := Finset.range (2 * n + 2)

/-- A predicate that checks if a subset B of A satisfies the condition that
    for any three different elements x, y, z in B, x + y ≠ z -/
def valid_subset (n : ℕ) (B : Finset ℕ) : Prop :=
  B ⊆ A n ∧ ∀ x y z, x ∈ B → y ∈ B → z ∈ B → x ≠ y → y ≠ z → x ≠ z → x + y ≠ z

/-- The theorem stating that the maximum size of a valid subset B is n + 1 -/
theorem max_subset_size (n : ℕ) :
  (∃ B : Finset ℕ, valid_subset n B ∧ B.card = n + 1) ∧
  (∀ B : Finset ℕ, valid_subset n B → B.card ≤ n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l494_49421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_classification_l494_49407

def complex_number (m : ℝ) : ℂ := Complex.mk (m^2 + 2*m - 8) (m^2 - 2*m)

theorem complex_number_classification (m : ℝ) :
  ((complex_number m).im = 0 ↔ m = 0 ∨ m = 2) ∧
  ((complex_number m).im ≠ 0 ↔ m ≠ 0 ∧ m ≠ 2) ∧
  ((complex_number m).re = 0 ∧ (complex_number m).im ≠ 0 ↔ m = -4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_classification_l494_49407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_in_right_triangle_l494_49426

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vector operations
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem length_AB_in_right_triangle (t : Triangle) : 
  -- Condition 1: Angle A is 90 degrees (right angle)
  (vector t.A t.B).1 * (vector t.A t.C).1 + (vector t.A t.B).2 * (vector t.A t.C).2 = 0 →
  -- Condition 2: Dot product of AB and BC is -2
  dot_product (vector t.A t.B) (vector t.B t.C) = -2 →
  -- Conclusion: Length of AB is √2
  vector_length (vector t.A t.B) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_in_right_triangle_l494_49426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_circle_center_l494_49483

-- Define the line equation
noncomputable def line_equation (a x y : ℝ) : Prop :=
  a * x + y + 1 = 0

-- Define the circle equation
noncomputable def circle_equation (a x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - a*y - 2 = 0

-- Define the center of the circle
noncomputable def circle_center (a : ℝ) : ℝ × ℝ :=
  (-1, a/2)

-- Theorem statement
theorem line_passes_through_circle_center (a : ℝ) :
  (∀ x y : ℝ, circle_equation a x y → (x, y) = circle_center a) →
  line_equation a (-1) (a/2) →
  a = 2 :=
by
  intros h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_circle_center_l494_49483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l494_49491

def point_A : ℝ × ℝ × ℝ := (1, 3, 0)
def point_B : ℝ × ℝ × ℝ := (-3, 6, 12)

theorem distance_AB : Real.sqrt (((point_B.1 - point_A.1)^2 + (point_B.2.1 - point_A.2.1)^2) + (point_B.2.2 - point_A.2.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l494_49491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_after_scaling_l494_49428

variable (x₁ x₂ x₃ x₄ x₅ m n a : ℝ)

noncomputable def average (x₁ x₂ x₃ x₄ x₅ : ℝ) : ℝ := (x₁ + x₂ + x₃ + x₄ + x₅) / 5

noncomputable def variance (x₁ x₂ x₃ x₄ x₅ m : ℝ) : ℝ := 
  ((x₁ - m)^2 + (x₂ - m)^2 + (x₃ - m)^2 + (x₄ - m)^2 + (x₅ - m)^2) / 5

theorem standard_deviation_after_scaling 
  (h_avg : average x₁ x₂ x₃ x₄ x₅ = m)
  (h_var : variance x₁ x₂ x₃ x₄ x₅ m = n)
  (h_pos : a > 0) :
  Real.sqrt (variance (a * x₁) (a * x₂) (a * x₃) (a * x₄) (a * x₅) (a * m)) = a * Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_after_scaling_l494_49428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antiderivative_of_f_l494_49476

noncomputable section

-- Define the integrand
def f (x : ℝ) : ℝ := 1 / Real.sqrt (1 - 9 * x^2)

-- Define the antiderivative
def F (x : ℝ) : ℝ := (1/3) * Real.arcsin (3*x)

-- Theorem statement
theorem antiderivative_of_f :
  ∀ x : ℝ, x ∈ Set.Ioo (-1/3) (1/3) → (deriv F) x = f x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antiderivative_of_f_l494_49476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l494_49427

open BigOperators Real

theorem infinite_series_sum : 
  let S : ℕ → ℝ := λ n => (2*n + 1) * (1/1001)^n
  (∑' n, S n) = 1001/1000000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l494_49427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_angle_l494_49494

theorem line_equation_through_point_with_angle :
  let point : ℝ × ℝ := (1, 2)
  let slope_angle : ℝ := 30 * π / 180
  let slope : ℝ := Real.tan slope_angle
  let line_equation := fun (x y : ℝ) ↦ Real.sqrt 3 * x - 3 * y + 6 - Real.sqrt 3 = 0
  ∀ x y : ℝ, (line_equation x y ↔ y - point.2 = slope * (x - point.1)) ∧
  line_equation point.1 point.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_angle_l494_49494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_distance_l494_49419

/-- The total distance covered by a bouncing ball -/
noncomputable def totalDistance (h : ℝ) : ℝ :=
  let firstTerm := h + 2 * (0.8 * h)
  let ratio := 0.8
  firstTerm / (1 - ratio)

/-- Theorem: The total distance covered by a ball dropped from height h,
    bouncing to 80% of its previous height each time, is 13h -/
theorem ball_bounce_distance (h : ℝ) (h_pos : h > 0) :
  totalDistance h = 13 * h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_distance_l494_49419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_K_for_divisibility_condition_l494_49404

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 50}

theorem smallest_K_for_divisibility_condition : 
  ∃ K : ℕ, K = 39 ∧ 
  (∀ A : Finset ℕ, ↑A ⊆ S → A.card ≥ K → 
    ∃ a b : ℕ, a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ (a + b) ∣ (a * b)) ∧
  (∀ K' : ℕ, K' < K → 
    ∃ A : Finset ℕ, ↑A ⊆ S ∧ A.card = K' ∧
    ∀ a b : ℕ, a ∈ A → b ∈ A → a ≠ b → ¬((a + b) ∣ (a * b))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_K_for_divisibility_condition_l494_49404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_draw_possible_l494_49444

/-- Represents a player in the game -/
inductive Player
| A
| B

/-- Represents a color of a token -/
inductive Color
| Red
| Blue

/-- Represents a vertex on the triangular grid -/
structure Vertex where
  x : ℕ
  y : ℕ

/-- Represents the state of the game board -/
def GameBoard := Vertex → Option Color

/-- Represents a path on the board -/
def GamePath := List Vertex

/-- Checks if a path is valid for a given player -/
def is_valid_path (player : Player) (path : GamePath) (board : GameBoard) : Prop :=
  match player with
  | Player.A => sorry  -- Path from PS to QR with red tokens
  | Player.B => sorry  -- Path from PQ to SR with blue tokens

/-- The main theorem stating that a draw is impossible -/
theorem no_draw_possible (board : GameBoard) :
  (∃ path_A : GamePath, is_valid_path Player.A path_A board) ∨
  (∃ path_B : GamePath, is_valid_path Player.B path_B board) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_draw_possible_l494_49444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l494_49450

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 + log x
noncomputable def g (x : ℝ) : ℝ := 2/3 * x^3

-- State the theorem
theorem f_properties :
  (∀ x ∈ Set.Icc 1 (exp 1), f x ≤ 1/2 * (exp 1)^2 + 1) ∧
  (∀ x ∈ Set.Icc 1 (exp 1), f x ≥ 1/2) ∧
  (∃ x ∈ Set.Icc 1 (exp 1), f x = 1/2 * (exp 1)^2 + 1) ∧
  (∃ x ∈ Set.Icc 1 (exp 1), f x = 1/2) ∧
  (∀ x > 1, f x < g x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l494_49450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_cardinality_l494_49498

def U : Finset ℕ := Finset.filter (fun n => n > 0 ∧ n ≤ 9) (Finset.range 10)
def A : Finset ℕ := {2, 5}
def B : Finset ℕ := {1, 2, 4, 5}

theorem complement_cardinality : Finset.card (U \ (A ∪ B)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_cardinality_l494_49498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_modification_theorem_l494_49425

/-- Represents the increase in miles traveled per full tank after modification --/
noncomputable def increased_miles_per_tank (current_efficiency : ℝ) (tank_capacity : ℝ) (fuel_reduction_factor : ℝ) : ℝ :=
  let new_efficiency := current_efficiency / fuel_reduction_factor
  let current_range := current_efficiency * tank_capacity
  let new_range := new_efficiency * tank_capacity
  new_range - current_range

/-- Theorem stating the increased miles per tank after modification --/
theorem car_modification_theorem :
  increased_miles_per_tank 28 15 0.8 = 84 := by
  -- Unfold the definition of increased_miles_per_tank
  unfold increased_miles_per_tank
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_modification_theorem_l494_49425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_of_special_triangle_l494_49430

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    where 2b = a = 4 and sinC - √3 * sinB = 0, the area of the circumcircle of triangle ABC is 4π. -/
theorem circumcircle_area_of_special_triangle (A B C : ℝ) (a b c : ℝ) :
  2 * b = a → a = 4 → Real.sin C - Real.sqrt 3 * Real.sin B = 0 →
  let R := c / (2 * Real.sin C)
  π * R^2 = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_of_special_triangle_l494_49430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_fraction_meaningful_alt_l494_49409

-- Define the fraction as a noncomputable function
noncomputable def fraction (x : ℝ) : ℝ := 8 / (x - 1)

-- Theorem stating when the fraction is meaningful
theorem fraction_meaningful (x : ℝ) : 
  ¬(x = 1) ↔ ∃ y, fraction x = y := by
  sorry

-- Equivalent statement to match the problem's options
theorem fraction_meaningful_alt (x : ℝ) :
  (∃ y, fraction x = y) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_fraction_meaningful_alt_l494_49409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l494_49431

-- Define the bounds of the integral
noncomputable def lowerBound : ℝ := 0
noncomputable def upperBound : ℝ := 2 * Real.pi / 3

-- Define the function representing the curve
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

-- Theorem statement
theorem area_enclosed_by_curve : 
  ∫ x in lowerBound..upperBound, f x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l494_49431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l494_49417

theorem count_integer_pairs : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.2 < 50) (Finset.product (Finset.range 50) (Finset.range 50))).card = 190 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l494_49417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l494_49439

/-- The function f(x) = |x-a| + |x+3| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

/-- Part 1: Solution set of f(x) ≥ 6 when a = 1 -/
theorem solution_set_part1 : 
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

/-- Part 2: Range of a for which f(x) > -a holds for all x -/
theorem range_of_a_part2 : 
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l494_49439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reward_is_320_l494_49481

/-- Represents the possible grades a student can receive. -/
inductive Grade
| BPlus
| A
| APlus
deriving BEq, Repr

/-- Represents a course category. -/
inductive CourseCategory
| Math
| Science
| History
| English
| ForeignLanguage
| SocialStudies
deriving BEq, Repr

/-- Represents the distribution of courses. -/
def courseDistribution : List (CourseCategory × Nat) :=
  [(CourseCategory.Math, 3), (CourseCategory.Science, 2), (CourseCategory.History, 2),
   (CourseCategory.English, 2), (CourseCategory.ForeignLanguage, 1), (CourseCategory.SocialStudies, 2)]

/-- The total number of courses. -/
def totalCourses : Nat := courseDistribution.foldl (fun acc (_, count) => acc + count) 0

/-- The reward for each grade. -/
def gradeReward (g : Grade) : Nat :=
  match g with
  | Grade.BPlus => 5
  | Grade.A => 10
  | Grade.APlus => 20

/-- Calculate the bonus based on the grade distribution. -/
def calculateBonus (grades : List Grade) : Nat :=
  let aPlusCount := grades.filter (· == Grade.APlus) |>.length
  let aCount := grades.filter (· == Grade.A) |>.length
  if aPlusCount ≥ 3 && aCount ≥ 2 then 50 else 0

/-- Calculate the total reward for a given list of grades. -/
def calculateTotalReward (grades : List Grade) : Nat :=
  grades.foldl (fun acc g => acc + gradeReward g) 0 + calculateBonus grades

/-- Theorem: The maximum reward Paul can receive is $320. -/
theorem max_reward_is_320 :
  ∃ (grades : List Grade),
    grades.length = totalCourses ∧
    calculateTotalReward grades = 320 ∧
    ∀ (otherGrades : List Grade),
      otherGrades.length = totalCourses →
      calculateTotalReward otherGrades ≤ calculateTotalReward grades := by
  sorry

#eval totalCourses
#eval calculateTotalReward (List.replicate totalCourses Grade.APlus)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reward_is_320_l494_49481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l494_49420

/-- Given a triangle ABC with side a = 15 cm and altitude h_a = 6 cm dividing side a in the ratio 4:1, 
    prove that the angle α is 90 degrees. -/
theorem triangle_right_angle (a h_a : ℝ) (h1 : a = 15) (h2 : h_a = 6) 
  (h3 : ∃ (x y : ℝ), x + y = a ∧ x / y = 4) : ∃ α : ℝ, α = 90 ∧ Real.tan α = h_a / (4 * a / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l494_49420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l494_49477

theorem min_value_trig_expression (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.tan x + 1 / Real.tan x)^2 + (Real.sin x + Real.cos x)^2 ≥ 5 ∧
  ((Real.tan x + 1 / Real.tan x)^2 + (Real.sin x + Real.cos x)^2 = 5 ↔ x = Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l494_49477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_angle_triangle_area_l494_49458

/-- Given an ellipse with semi-major axis 5 and semi-minor axis 3, 
    and a point P on the ellipse forming a right angle with the foci,
    the area of the triangle formed by P and the foci is 9. -/
theorem ellipse_right_angle_triangle_area :
  ∀ (F₁ F₂ P : ℝ × ℝ),
  let a : ℝ := 5
  let b : ℝ := 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) →  -- P is on the ellipse
  (F₁ = (c, 0) ∧ F₂ = (-c, 0)) →    -- F₁ and F₂ are the foci
  (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
   Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 * a) →  -- Ellipse property
  ((P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0) →  -- Right angle at P
  (1/2 * Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) * 
   Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
   Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) / 
   (Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)) = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_angle_triangle_area_l494_49458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_values_for_2014_division_l494_49495

def count_divisors_minus_one (n : Nat) : Nat :=
  (Nat.divisors n).filter (· > 1) |>.card

theorem positive_integer_values_for_2014_division : 
  count_divisors_minus_one 2014 = 7 := by
  rw [count_divisors_minus_one]
  simp [Nat.divisors]
  -- The actual proof would go here
  sorry

#eval count_divisors_minus_one 2014

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_values_for_2014_division_l494_49495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l494_49440

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- Represents a line with slope m -/
structure Line where
  m : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- Theorem: For an ellipse intersected by a line with slope √2/2 at two points
    whose x-axis projections are the foci, the eccentricity is √2/2 -/
theorem ellipse_eccentricity_theorem (e : Ellipse) (l : Line) (p q : Point) :
  l.m = Real.sqrt 2 / 2 →
  (∃ t : ℝ, p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1 ∧
             q.x^2 / e.a^2 + q.y^2 / e.b^2 = 1 ∧
             p.y = l.m * p.x ∧ q.y = l.m * q.x ∧
             t ≠ 0 ∧ q.x = p.x + t ∧ q.y = p.y + l.m * t) →
  (∃ c : ℝ, c > 0 ∧ p.x = c ∧ q.x = -c ∧ 
            c^2 = e.a^2 - e.b^2) →
  eccentricity e = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l494_49440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l494_49452

/-- Calculates the time for a train to cross a platform given its length, the platform length, and the time to cross a signal pole. -/
noncomputable def time_to_cross_platform (train_length platform_length signal_time : ℝ) : ℝ :=
  (train_length + platform_length) / (train_length / signal_time)

/-- Theorem stating that a 300m train crossing a 450m platform takes approximately 45 seconds,
    given that it takes 18 seconds to cross a signal pole. -/
theorem train_crossing_time :
  ∀ (ε : ℝ), ε > 0 →
  ∃ (time : ℝ),
    time_to_cross_platform 300 450 18 = time ∧
    abs (time - 45) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l494_49452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_length_l494_49490

/-- A sequence of numbers satisfying specific average conditions -/
def SpecialSequence (s : List ℝ) : Prop :=
  ∃ n : ℕ,
    n > 0 ∧
    s.length = n ∧
    (s.take 4).sum / 4 = 4 ∧
    (s.drop (n - 4)).sum / 4 = 4 ∧
    s.sum / n = 3 ∧
    s[3]! = 11

/-- The length of a special sequence is 7 -/
theorem special_sequence_length :
  ∀ s : List ℝ, SpecialSequence s → s.length = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_length_l494_49490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l494_49412

/-- Proves that shifting the graph of y = sin(2x - π/6) to the left by π/12 units 
    results in the graph of y = sin 2x -/
theorem sin_graph_shift (x : ℝ) : 
  Real.sin (2 * (x + π/12) - π/6) = Real.sin (2 * x) := by
  have h1 : 2 * (x + π/12) - π/6 = 2 * x := by
    ring
  rw [h1]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l494_49412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_probability_distribution_l494_49402

/-- A random variable ξ with a discrete distribution --/
noncomputable def ξ : ℝ → ℝ := sorry

/-- The probability mass function of ξ --/
noncomputable def P (x y : ℝ) : ℝ → ℝ := sorry

/-- The expected value of ξ --/
noncomputable def E (x y : ℝ) : ℝ := 1 * P x y 1 + 2 * P x y 2 + 3 * P x y 3

/-- The objective function to be minimized --/
noncomputable def f (x y : ℝ) : ℝ := 1/x + 4/y

theorem minimize_probability_distribution (x y : ℝ) :
  (x > 0) →
  (y > 0) →
  (x + 1/2 + y = 1) →
  (P x y 1 = x) →
  (P x y 2 = 1/2) →
  (P x y 3 = y) →
  (∀ x' y', x' > 0 → y' > 0 → x' + 1/2 + y' = 1 → f x' y' ≥ f x y) →
  (x = 1/6 ∧ E x y = 13/6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_probability_distribution_l494_49402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_sale_savings_l494_49487

/-- The amount saved when buying 3 shirts under given discount conditions -/
theorem shirt_sale_savings 
  (regular_price : ℝ)
  (second_shirt_discount : ℝ)
  (third_shirt_discount : ℝ)
  (h1 : regular_price = 10)
  (h2 : second_shirt_discount = 0.5)
  (h3 : third_shirt_discount = 0.6) :
  3 * regular_price - (regular_price + (1 - second_shirt_discount) * regular_price + (1 - third_shirt_discount) * regular_price) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_sale_savings_l494_49487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stephan_name_rearrangement_time_l494_49432

def stephan_rearrangement_time (name_length : ℕ) (rearrangements_per_minute : ℕ) (minutes_per_hour : ℕ) : ℕ :=
  let total_rearrangements := Nat.factorial name_length
  let total_minutes := total_rearrangements / rearrangements_per_minute
  total_minutes / minutes_per_hour

theorem stephan_name_rearrangement_time :
  stephan_rearrangement_time 7 12 60 = 7 :=
by
  unfold stephan_rearrangement_time
  simp
  norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stephan_name_rearrangement_time_l494_49432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_diagonals_distance_theorem_l494_49465

/-- The distance between two skew diagonals of adjacent faces of a cube -/
noncomputable def skew_diagonals_distance (a : ℝ) : ℝ :=
  (1/3) * a * Real.sqrt 3

/-- Representation of a line in 3D space -/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Calculate the distance between two lines in 3D space -/
noncomputable def distance (l1 l2 : Line3D) : ℝ :=
  sorry  -- Placeholder for the actual distance calculation

/-- Generate a skew diagonal of an adjacent face given the cube edge length -/
noncomputable def skew_diagonal_of_adjacent_face (a : ℝ) : Line3D :=
  sorry  -- Placeholder for the actual line generation

/-- Generate another skew diagonal of an adjacent face given the cube edge length -/
noncomputable def another_skew_diagonal_of_adjacent_face (a : ℝ) : Line3D :=
  sorry  -- Placeholder for the actual line generation

/-- Theorem: The distance between two skew diagonals of adjacent faces of a cube
    with edge length a is (1/3)a√3 -/
theorem skew_diagonals_distance_theorem (a : ℝ) (ha : a > 0) :
  let cube_edge := a
  let diag1 := skew_diagonal_of_adjacent_face cube_edge
  let diag2 := another_skew_diagonal_of_adjacent_face cube_edge
  distance diag1 diag2 = skew_diagonals_distance cube_edge :=
by
  sorry  -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_diagonals_distance_theorem_l494_49465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_two_unique_parallel_lines_at_distance_two_l494_49492

/-- The distance between two parallel lines with equations ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₂ - c₁) / Real.sqrt (a^2 + b^2)

/-- Theorem: The lines 3x - 4y - 11 = 0 and 3x - 4y + 9 = 0 are at distance 2 from 3x - 4y - 1 = 0 -/
theorem parallel_lines_at_distance_two :
  (distance_parallel_lines 3 (-4) (-1) (-11) = 2) ∧
  (distance_parallel_lines 3 (-4) (-1) 9 = 2) := by
  sorry

/-- The equations 3x - 4y - 11 = 0 and 3x - 4y + 9 = 0 are the only lines at distance 2 from 3x - 4y - 1 = 0 -/
theorem unique_parallel_lines_at_distance_two (c : ℝ) :
  distance_parallel_lines 3 (-4) (-1) c = 2 ↔ c = -11 ∨ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_two_unique_parallel_lines_at_distance_two_l494_49492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigEquationSolutions_l494_49435

noncomputable def trigEquation (x : ℝ) : Prop :=
  (Real.sin x)^3 + (Real.sin (2 * Real.pi / 3 + x))^3 + (Real.sin (4 * Real.pi / 3 + x))^3 + (3/4) * Real.cos (2*x) = 0

noncomputable def generalSolution (k : ℤ) : ℝ :=
  (4*k + 1) * Real.pi / 10

def isDodecagonVertex (k : ℤ) : Prop :=
  k % 5 = 1

def isRegularNGonVertex (n : ℕ) (x : ℝ) : Prop :=
  ∃ m : ℤ, x = (2 * m * Real.pi) / n

theorem trigEquationSolutions :
  (∀ x : ℝ, trigEquation x ↔ ∃ k : ℤ, x = generalSolution k) ∧
  (∀ k : ℤ, isDodecagonVertex k → isRegularNGonVertex 12 (generalSolution k)) ∧
  (∀ n : ℕ, (¬ 2 ∣ n ∧ ¬ 3 ∣ n) ∨ Nat.Prime n → 
    ∀ k : ℤ, ¬ isRegularNGonVertex n (generalSolution k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigEquationSolutions_l494_49435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_dodecahedron_volume_ratio_l494_49461

/-- The volume of a regular dodecahedron with side length s -/
noncomputable def dodecahedron_volume (s : ℝ) : ℝ := (15 + 7 * Real.sqrt 5) * s^3 / 4

/-- The volume of a regular icosahedron with side length a -/
noncomputable def icosahedron_volume (a : ℝ) : ℝ := 5 * (3 + Real.sqrt 5) * a^3 / 12

/-- The side length of an icosahedron formed from a dodecahedron with side length s -/
noncomputable def icosahedron_side (s : ℝ) : ℝ := s * Real.sqrt 3 / 2

/-- The theorem stating the ratio of volumes of icosahedron to dodecahedron -/
theorem icosahedron_dodecahedron_volume_ratio (s : ℝ) (h : s > 0) :
  icosahedron_volume (icosahedron_side s) / dodecahedron_volume s =
  45 * Real.sqrt 3 * (3 + Real.sqrt 5) / (384 * (15 + 7 * Real.sqrt 5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_dodecahedron_volume_ratio_l494_49461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l494_49422

/-- The polar equation of the circle -/
def polar_equation (r θ : ℝ) : Prop := r = 3 * Real.cos θ - 4 * Real.sin θ

/-- The area of the circle described by the polar equation -/
noncomputable def circle_area : ℝ := 13 * Real.pi / 4

theorem circle_area_from_polar_equation :
  ∀ r θ : ℝ, polar_equation r θ → circle_area = (13 : ℝ) * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l494_49422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_positional_relationship_intersection_l494_49478

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 4 = 0

-- Define the centers and radii of the circles
def center_C1 : ℝ × ℝ := (0, 0)
def radius_C1 : ℝ := 1
def center_C2 : ℝ × ℝ := (-2, 3)
def radius_C2 : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 13

-- Theorem: The circles C1 and C2 intersect
theorem circles_intersect :
  radius_C2 - radius_C1 < distance_between_centers ∧
  distance_between_centers < radius_C2 + radius_C1 := by
  sorry

-- Theorem: The positional relationship between C1 and C2 is intersection
theorem positional_relationship_intersection :
  ∃ (x y : ℝ), circle_C1 x y ∧ circle_C2 x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_positional_relationship_intersection_l494_49478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_equations_l494_49497

noncomputable def reciprocal (n : ℝ) : ℝ := 1 / n

theorem reciprocal_equations :
  let star := reciprocal
  (star 4 + star 8 ≠ star 12) ∧
  (star 8 - star 6 ≠ star 2) ∧
  (star 3 / star 9 ≠ star 3) ∧
  (star 15 / star 3 = star 5) :=
by
  -- Introduce the local definition of star
  intro star

  -- Split the conjunction into four parts
  apply And.intro
  · sorry -- Proof for star 4 + star 8 ≠ star 12
  apply And.intro
  · sorry -- Proof for star 8 - star 6 ≠ star 2
  apply And.intro
  · sorry -- Proof for star 3 / star 9 ≠ star 3
  · sorry -- Proof for star 15 / star 3 = star 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_equations_l494_49497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l494_49472

/-- Represents a quadratic function of the form f(x) = x^2 + bx + c -/
structure QuadraticFunction where
  b : ℝ
  c : ℝ

/-- The x-coordinate of the axis of symmetry for a quadratic function -/
noncomputable def axisOfSymmetry (f : QuadraticFunction) : ℝ := -f.b / 2

theorem parabola_properties (f : QuadraticFunction) 
  (h1 : f.c = -3)  -- Intersects y-axis at (0, -3)
  (h2 : ∃ x1 x2 : ℝ, 
    x1^2 + f.b * x1 + f.c = 0 ∧ 
    x2^2 + f.b * x2 + f.c = 0 ∧ 
    x1^2 + x2^2 = 15) :  -- Sum of squares of x-intercepts is 15
  (f.b = 3 ∨ f.b = -3) ∧ 
  (axisOfSymmetry f = -3/2 ∨ axisOfSymmetry f = 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l494_49472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celia_cell_phone_cost_l494_49482

/-- Celia's monthly budget --/
structure MonthlyBudget where
  food_weekly : ℚ
  rent : ℚ
  streaming : ℚ
  savings_percent : ℚ
  savings_amount : ℚ
  weeks_in_month : ℕ

/-- Calculate Celia's cell phone usage cost --/
def cell_phone_cost (budget : MonthlyBudget) : ℚ :=
  let total_spending := budget.savings_amount / budget.savings_percent
  let known_expenses := budget.food_weekly * budget.weeks_in_month + budget.rent + budget.streaming
  total_spending - known_expenses

/-- Theorem: Celia's cell phone usage cost is $50 --/
theorem celia_cell_phone_cost :
  let budget : MonthlyBudget := {
    food_weekly := 100,
    rent := 1500,
    streaming := 30,
    savings_percent := 1/10,
    savings_amount := 198,
    weeks_in_month := 4
  }
  cell_phone_cost budget = 50 := by sorry

#eval cell_phone_cost {
  food_weekly := 100,
  rent := 1500,
  streaming := 30,
  savings_percent := 1/10,
  savings_amount := 198,
  weeks_in_month := 4
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_celia_cell_phone_cost_l494_49482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l494_49401

/-- The system of equations --/
def system (x y z : ℝ) : Prop :=
  x^3 + x*(y - z)^2 = 2 ∧
  y^3 + y*(z - x)^2 = 30 ∧
  z^3 + z*(x - y)^2 = 16

/-- The solutions to the system of equations --/
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ 
    ((x = (2 : ℝ)^(1/3) ∧ y = (2 : ℝ)^(1/3) ∧ z = (2 : ℝ)^(1/3)) ∨
     (x = -(2 : ℝ)^(1/3) ∧ y = (2 : ℝ)^(1/3) ∧ z = 3 * (2 : ℝ)^(1/3))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l494_49401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_must_contain_sequential_not_all_algorithms_contain_selection_not_all_algorithms_contain_loop_sequential_structure_only_correct_statement_l494_49405

-- Define the basic logical structures
inductive LogicalStructure
| Sequential
| Selection
| Loop

-- Define an algorithm as a set of logical structures
def Algorithm := Set LogicalStructure

-- Define a membership instance for LogicalStructure and Algorithm
instance : Membership LogicalStructure Algorithm := 
  { mem := λ s a => a s }

-- Theorem stating that every algorithm must contain a sequential structure
theorem algorithm_must_contain_sequential (a : Algorithm) : 
  LogicalStructure.Sequential ∈ a :=
sorry

-- Theorem stating that not every algorithm must contain a selection structure
theorem not_all_algorithms_contain_selection : 
  ∃ a : Algorithm, LogicalStructure.Selection ∉ a :=
sorry

-- Theorem stating that not every algorithm must contain a loop structure
theorem not_all_algorithms_contain_loop : 
  ∃ a : Algorithm, LogicalStructure.Loop ∉ a :=
sorry

-- Main theorem proving that the statement about sequential structure is the only correct one
theorem sequential_structure_only_correct_statement : 
  (∀ a : Algorithm, LogicalStructure.Sequential ∈ a) ∧
  (∃ a : Algorithm, LogicalStructure.Selection ∉ a) ∧
  (∃ a : Algorithm, LogicalStructure.Loop ∉ a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_must_contain_sequential_not_all_algorithms_contain_selection_not_all_algorithms_contain_loop_sequential_structure_only_correct_statement_l494_49405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l494_49456

-- Define the slope angle of a line
noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan (-m)

-- Define our line equation
def line_equation (x y a : ℝ) : Prop := Real.sqrt 3 * x + 3 * y + a = 0

-- Theorem statement
theorem line_slope_angle : 
  slope_angle (Real.sqrt 3 / 3) = 150 * (Real.pi / 180) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l494_49456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inf_f_solution_set_l494_49485

-- Define the function f
def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

-- Theorem 1: The infimum of f(x) is -3
theorem inf_f : ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∀ (ε : ℝ), ε > 0 → ∃ (y : ℝ), f y < m + ε) :=
sorry

-- Theorem 2: The solution set of x^2 - 8x + 15 + f(x) ≤ 0 is {x | 5 - √3 ≤ x ≤ 6}
theorem solution_set : 
  ∀ (x : ℝ), x^2 - 8*x + 15 + f x ≤ 0 ↔ 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inf_f_solution_set_l494_49485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l494_49445

/-- A three-digit number -/
def A : ℕ := 171

/-- First two-digit number -/
def B : ℕ := 37

/-- Second two-digit number -/
def C : ℕ := 39

/-- Predicate to check if a number contains the digit 7 -/
def contains_seven (n : ℕ) : Prop := ∃ (k m : ℕ), n = 10 * k + 7 + 10 * m

/-- Predicate to check if a number contains the digit 3 -/
def contains_three (n : ℕ) : Prop := ∃ (k m : ℕ), n = 10 * k + 3 + 10 * m

theorem sum_of_numbers (h1 : 100 ≤ A ∧ A < 1000)
                       (h2 : 10 ≤ B ∧ B < 100)
                       (h3 : 10 ≤ C ∧ C < 100)
                       (h4 : contains_seven A ∧ contains_seven B)
                       (h5 : contains_three B ∧ contains_three C)
                       (h6 : B + C = 76)
                       (h7 : A + B = 208) :
  A + B + C = 247 := by
  sorry

#eval A + B + C -- This will evaluate to 247

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l494_49445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_three_four_l494_49449

-- Define the operation ⊕
noncomputable def oplus (x y : ℝ) : ℝ := (x^2 + y^2) / (1 + x * y^2)

-- State the theorem
theorem oplus_three_four :
  oplus 3 4 = 25 / 49 := by
  -- Unfold the definition of oplus
  unfold oplus
  -- Simplify the numerator and denominator
  simp [pow_two]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_three_four_l494_49449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l494_49480

theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (current_speed : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : current_speed = 3) 
  (h3 : time_minutes = 15) : 
  (boat_speed + current_speed) * (time_minutes / 60) = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l494_49480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_plus_inv_w_traces_ellipse_l494_49493

-- Define the complex number w
variable (w : ℂ)

-- Define the condition that w traces a circle with radius 3
def traces_circle (w : ℂ) : Prop := Complex.abs w = 3

-- Define the function f(w) = w + 1/w
noncomputable def f (w : ℂ) : ℂ := w + w⁻¹

-- State the theorem
theorem w_plus_inv_w_traces_ellipse :
  traces_circle w → ∃ a b : ℝ, ∀ z ∈ Set.range f, 
    (z.re / a)^2 + (z.im / b)^2 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_plus_inv_w_traces_ellipse_l494_49493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l494_49475

theorem sin_double_angle (α : Real) 
  (h1 : Real.sin (α + π/2) = 3/5) 
  (h2 : 0 < α) 
  (h3 : α < π) : 
  Real.sin (2 * α) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l494_49475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_length_is_sqrt_13_l494_49424

/-- A right-angled polyhedron with vertices X and Y -/
structure RightAngledPolyhedron where
  /-- The length of the first edge between X and Y -/
  a : ℝ
  /-- The length of the second edge between X and Y -/
  b : ℝ
  /-- The length of the third edge between X and Y -/
  c : ℝ

/-- The shortest path length between vertices X and Y on the surface of a right-angled polyhedron -/
noncomputable def shortestPathLength (p : RightAngledPolyhedron) : ℝ :=
  Real.sqrt (p.a^2 + p.b^2 + p.c^2)

/-- Theorem: The shortest path length between X and Y on the surface of a right-angled polyhedron
    with edge lengths a = 2, b = 2, and c = 1 is √13 -/
theorem shortest_path_length_is_sqrt_13 :
  let p : RightAngledPolyhedron := ⟨2, 2, 1⟩
  shortestPathLength p = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_length_is_sqrt_13_l494_49424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_increase_l494_49400

/-- Given a sequence of four equilateral triangles where each subsequent triangle's
    side length is 125% of the previous one, starting with a side length of 4 units,
    this theorem states that the percent increase in perimeter from the first to the
    fourth triangle is approximately 95.3%. -/
theorem triangle_perimeter_increase (initial_side : ℝ) (growth_factor : ℝ) :
  initial_side = 4 →
  growth_factor = 1.25 →
  abs ((initial_side * growth_factor^3 - initial_side) / initial_side * 100 - 95.3) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_increase_l494_49400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_nineteen_pi_over_four_l494_49414

theorem cos_nineteen_pi_over_four :
  Real.cos (19 * Real.pi / 4) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_nineteen_pi_over_four_l494_49414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l494_49429

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the slope of a line -/
noncomputable def slopeOfLine (l : Line) : ℝ := -l.a / l.b

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the distance between a point and a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  |l.a * p.x + l.b * p.y + l.c| / Real.sqrt (l.a^2 + l.b^2)

/-- Main theorem -/
theorem line_equations (P : Point) (l m : Line) :
  P.x = 2 ∧ P.y = 7/4 ∧ 
  slopeOfLine l = 3/4 ∧
  pointOnLine P l ∧
  slopeOfLine m = slopeOfLine l ∧
  distancePointToLine P m = 3 →
  (l.a = 3 ∧ l.b = -4 ∧ l.c = 5) ∧
  ((m.a = 3 ∧ m.b = -4 ∧ m.c = -16) ∨ (m.a = 3 ∧ m.b = -4 ∧ m.c = 14)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l494_49429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l494_49467

def sequence_a : ℕ → ℝ
  | 0 => 2  -- Adding the base case for 0
  | 1 => 2
  | n + 2 => 3 * sequence_a (n + 1) + 2

theorem min_k_value (k : ℝ) :
  (∀ n : ℕ, n ≥ 1 → k * (sequence_a n + 1) ≥ 2 * n - 3) →
  k ≥ 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l494_49467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_and_inequality_l494_49454

noncomputable def f (a b x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

def isOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem odd_function_values_and_inequality 
  (h₁ : isOdd (f a b))
  (h₂ : ∀ x, (f a b x) ∈ Set.univ) :
  (a = 2 ∧ b = 1) ∧
  (∀ t, f 2 1 (t^2 - 2*t) + f 2 1 (2*t^2 - 1) < 0 ↔ t > 1 ∨ t < -1/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_and_inequality_l494_49454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_when_a_4_a_values_when_min_is_neg_5_l494_49438

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - a * x + 5

-- Define the domain
def domain : Set ℝ := Set.Icc (-1) 2

-- Part 1: Extreme values when a = 4
theorem extreme_values_when_a_4 :
  (∃ x ∈ domain, ∀ y ∈ domain, f 4 x ≥ f 4 y) ∧
  (∃ x ∈ domain, ∀ y ∈ domain, f 4 x ≤ f 4 y) ∧
  (∀ x ∈ domain, f 4 x ≤ 11) ∧
  (∀ x ∈ domain, f 4 x ≥ 3) ∧
  (∃ x ∈ domain, f 4 x = 11) ∧
  (∃ x ∈ domain, f 4 x = 3) :=
by sorry

-- Part 2: Value of a when minimum is -5
theorem a_values_when_min_is_neg_5 :
  (∃ a : ℝ, ∃ x ∈ domain, ∀ y ∈ domain, f a x ≤ f a y ∧ f a x = -5) →
  (a = -12 ∨ a = 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_when_a_4_a_values_when_min_is_neg_5_l494_49438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_specific_matrix_l494_49470

theorem det_specific_matrix (x y z : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, x, y+z; 2, x+y, y; 2, x, x+z]
  Matrix.det A = 2*x^2 + 2*x*z + 2*y*z - 2*y^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_specific_matrix_l494_49470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_parallelepiped_volume_l494_49462

/-- A parallelepiped with rhombus faces -/
structure RhombusParallelepiped where
  /-- Length of the first diagonal of the rhombus faces -/
  diagonal1 : ℝ
  /-- Length of the second diagonal of the rhombus faces -/
  diagonal2 : ℝ
  /-- Assertion that the parallelepiped has trihedral angles formed by three acute angles of the rhombuses -/
  has_trihedral_angles : Prop

/-- The volume of a rhombus parallelepiped -/
noncomputable def volume (p : RhombusParallelepiped) : ℝ :=
  9 * Real.sqrt 39 / 4

/-- Theorem stating the volume of the specific rhombus parallelepiped -/
theorem rhombus_parallelepiped_volume (p : RhombusParallelepiped) 
    (h1 : p.diagonal1 = 3) 
    (h2 : p.diagonal2 = 4) : 
  volume p = 9 * Real.sqrt 39 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_parallelepiped_volume_l494_49462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_function_l494_49443

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem symmetry_of_sine_function 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_bound : |φ| < π/2) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x)
  (h_odd_shift : ∀ x, f ω φ (x + π/6) = -f ω φ (-x + π/6)) :
  ∀ x, f ω φ (π/6 + x) = f ω φ (π/6 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_function_l494_49443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_formula_orthogonal_iff_chord_length_l494_49453

/-- Two intersecting circles with radii r₁ and r₂ -/
structure IntersectingCircles (r₁ r₂ : ℝ) where
  h : r₁ < r₂

/-- The chord PQ formed by intersecting lines from R to A and B -/
noncomputable def chord_length (ic : IntersectingCircles r₁ r₂) (β : ℝ) : ℝ :=
  r₁ * Real.sin (2 * β) + 2 * Real.sqrt (r₂^2 - r₁^2 * (Real.sin β)^2) * Real.sin β

/-- The circles are orthogonal -/
def orthogonal (ic : IntersectingCircles r₁ r₂) : Prop :=
  ∃ β, chord_length ic β = 2 * r₂

theorem chord_length_formula (ic : IntersectingCircles r₁ r₂) (β : ℝ) :
  chord_length ic β = r₁ * Real.sin (2 * β) + 2 * Real.sqrt (r₂^2 - r₁^2 * (Real.sin β)^2) * Real.sin β := by
  sorry

theorem orthogonal_iff_chord_length (ic : IntersectingCircles r₁ r₂) :
  orthogonal ic ↔ ∃ β, chord_length ic β = 2 * r₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_formula_orthogonal_iff_chord_length_l494_49453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_sum_equality_l494_49451

theorem tan_cot_sum_equality (θ φ : Real) 
  (h : (Real.tan θ^4)/(Real.tan φ^2) + (1/Real.tan θ^4)/(1/Real.tan φ^2) = 2) :
  (Real.tan φ^4)/(Real.tan θ^2) + (1/Real.tan φ^4)/(1/Real.tan θ^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_sum_equality_l494_49451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_corrals_area_ratio_l494_49464

theorem equilateral_triangle_corrals_area_ratio :
  ∀ (s : ℝ),
  s > 0 →
  let small_corral_side := s
  let small_corral_perimeter := 3 * small_corral_side
  let total_fencing := 6 * small_corral_perimeter
  let large_corral_side := total_fencing / 3
  let small_corral_area := (Real.sqrt 3 / 4) * small_corral_side^2
  let large_corral_area := (Real.sqrt 3 / 4) * large_corral_side^2
  (6 * small_corral_area) / large_corral_area = 1 / 6 :=
by
  intro s s_pos
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_corrals_area_ratio_l494_49464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_first_term_l494_49479

/-- An arithmetic sequence with 1000 terms -/
noncomputable def arithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- The sum of the first k terms of an arithmetic sequence -/
noncomputable def sumFirstK (a₁ d : ℝ) (k : ℕ) : ℝ := k * (2 * a₁ + (k - 1 : ℝ) * d) / 2

/-- The sum of the last k terms of an arithmetic sequence with n terms -/
noncomputable def sumLastK (a₁ d : ℝ) (n k : ℕ) : ℝ := 
  k * (2 * (a₁ + (n - k : ℝ) * d) + (k - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_first_term (a₁ d : ℝ) :
  sumFirstK a₁ d 100 = 100 →
  sumLastK a₁ d 1000 100 = 1000 →
  a₁ = 0.505 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_first_term_l494_49479
