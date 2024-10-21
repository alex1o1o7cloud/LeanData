import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l349_34923

theorem trigonometric_identities (x : ℝ) 
  (h1 : Real.sin (x - π/4) = 7*Real.sqrt 2/10) 
  (h2 : Real.cos (2*x) = 7/25) : 
  Real.cos (7*π/12 - x) = (7*Real.sqrt 6 - Real.sqrt 2)/20 ∧ 
  (Real.sin (2*x) + 2*Real.sin x^2) / (1 - Real.tan x) = -24/175 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l349_34923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_monotonicity_l349_34943

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.log x + (1/2) * x^2 - 5 * x

-- Define the derivative of f
noncomputable def f_prime (x : ℝ) : ℝ := 4 / x + x - 5

-- State the theorem
theorem function_and_monotonicity (a b : ℝ) :
  (∀ x > 0, f x = 4 * Real.log x + a * x^2 + b * x) →
  (∀ x > 0, f_prime x = (4 / x) + 2 * a * x + b) →
  (f_prime 1 = 0 ∧ f_prime 4 = 0) →
  (∀ m ∈ Set.Icc (1/2) 1, ∀ x ∈ Set.Ioo (2*m) (m+1), f_prime x < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_monotonicity_l349_34943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l349_34910

noncomputable section

-- Define the point P
def P : ℝ × ℝ := (-2, 5)

-- Define the slope of line l
def slope_l : ℝ := -3/4

-- Define the distance between lines l and m
def distance_lm : ℝ := 3

-- Define the general form of a line equation
def line_equation (a b c : ℝ) : ℝ × ℝ → Prop :=
  λ p ↦ a * p.1 + b * p.2 + c = 0

-- Define line l
def line_l : ℝ × ℝ → Prop :=
  line_equation 3 4 (-17)

-- Define the possible equations for line m
def line_m1 : ℝ × ℝ → Prop :=
  line_equation 3 4 1

def line_m2 : ℝ × ℝ → Prop :=
  line_equation 3 4 (-29)

-- State the theorem
theorem parallel_line_equation :
  ∃ (m : ℝ × ℝ → Prop), 
    (∀ p, line_l p ↔ 3 * p.1 + 4 * p.2 - 17 = 0) ∧
    (m = line_m1 ∨ m = line_m2) ∧
    (∀ p, line_l p → P.1 * p.1 + P.2 * p.2 = -2 * (-2) + 5 * 5) ∧
    (∀ p q, line_l p → m q → (p.1 - q.1)^2 + (p.2 - q.2)^2 ≥ distance_lm^2) ∧
    (∃ p q, line_l p ∧ m q ∧ (p.1 - q.1)^2 + (p.2 - q.2)^2 = distance_lm^2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l349_34910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_diagonal_quadrilateral_sum_squares_l349_34968

/-- A quadrilateral with perpendicular diagonals -/
structure PerpendicularDiagonalQuadrilateral :=
  (A B C D : ℝ × ℝ)
  (perpendicular_diagonals : (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0)

/-- The length of a side of a quadrilateral -/
noncomputable def side_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- Theorem: In a quadrilateral with perpendicular diagonals, 
    if two opposite sides have lengths 6 and 8, 
    then the sum of squares of the other two sides is 100 -/
theorem perpendicular_diagonal_quadrilateral_sum_squares
  (ABCD : PerpendicularDiagonalQuadrilateral)
  (h1 : side_length ABCD.A ABCD.B = 6)
  (h2 : side_length ABCD.C ABCD.D = 8) :
  (side_length ABCD.B ABCD.C)^2 + (side_length ABCD.D ABCD.A)^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_diagonal_quadrilateral_sum_squares_l349_34968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_is_all_reals_l349_34981

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := (x^3 + 8*x - 4) / (|x - 4| + |x + 2|)

-- Theorem stating that the domain of h is all real numbers
theorem h_domain_is_all_reals :
  ∀ x : ℝ, ∃ y : ℝ, h x = y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_is_all_reals_l349_34981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_general_equation_l349_34935

open Real

-- Define the curve C
noncomputable def C (α : ℝ) : ℝ × ℝ := (sin α - cos α, sin (2 * α))

-- State the theorem
theorem curve_C_general_equation :
  ∀ (x y : ℝ), (∃ α : ℝ, C α = (x, y)) →
  y = -x^2 + 1 ∧ x ∈ Set.Icc (-sqrt 2) (sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_general_equation_l349_34935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l349_34901

/-- Given a sphere and a triangle tangent to it, calculate the distance from the sphere's center to the triangle's plane --/
theorem sphere_triangle_distance (r : ℝ) (a b c : ℝ) (h_sphere : r = 6) (h_triangle : a = 15 ∧ b = 15 ∧ c = 24) :
  ∃ d : ℝ, d = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l349_34901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l349_34955

def A : Set ℤ := {-1, 1, 3, 5, 7}
def B : Set ℤ := {x : ℤ | (2 : ℝ)^(x : ℝ) > 2 * Real.sqrt 2}

theorem A_intersect_B : A ∩ B = {3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l349_34955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_to_cone_surface_area_l349_34925

/-- The surface area of a cone formed by rolling a semicircle with radius R -/
noncomputable def cone_surface_area (R : ℝ) : ℝ := (3/4) * Real.pi * R^2

/-- Theorem: The surface area of a cone formed by rolling a semicircle with radius R
    is equal to 3/4 * π * R^2 -/
theorem semicircle_to_cone_surface_area (R : ℝ) (h : R > 0) :
  cone_surface_area R = (3/4) * Real.pi * R^2 := by
  -- Unfold the definition of cone_surface_area
  unfold cone_surface_area
  -- The equality is trivial by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_to_cone_surface_area_l349_34925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_day_distance_l349_34927

/-- Represents the total distance of the journey in li -/
noncomputable def total_distance : ℝ := 378

/-- Represents the number of days of the journey -/
def num_days : ℕ := 6

/-- Represents the ratio of distance walked each day compared to the previous day -/
noncomputable def ratio : ℝ := 1/2

/-- Calculates the distance walked on the nth day -/
noncomputable def distance_on_day (n : ℕ) : ℝ :=
  if n = 1 then
    total_distance * (1 - ratio) / (1 - ratio^num_days)
  else
    (total_distance * (1 - ratio) / (1 - ratio^num_days)) * ratio^(n-1)

/-- Theorem stating that the distance walked on the last day is 6 li -/
theorem last_day_distance :
  distance_on_day num_days = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_day_distance_l349_34927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l349_34950

noncomputable def f (x : ℝ) : ℝ := 5 / (x + 1) - 1

theorem f_satisfies_conditions :
  (∀ x, x > 0 → f x ∈ Set.Ioo (-1) 4) ∧
  (∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → f x₁ + f x₂ < 2 * f ((x₁ + x₂) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l349_34950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_l349_34956

/-- The height of a right pyramid with a rectangular base -/
theorem pyramid_height (length width perimeter apex_distance : ℝ) (h1 : length = 2 * width) 
  (h2 : perimeter = 2 * (length + width)) (h3 : perimeter = 32) 
  (h4 : apex_distance = 10) : 
  Real.sqrt (apex_distance^2 - ((length^2 + width^2).sqrt / 2)^2) = 10 * Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_l349_34956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_third_statement_correct_l349_34959

theorem only_third_statement_correct : 
  -- First statement is incorrect
  (∃ p q : Prop, ¬p ∧ ¬(¬q) ∧ (p ∨ q)) ∧
  -- Second statement is incorrect
  (∃ x y : ℝ, ¬(x * y = 0 → x = 0 ∨ y = 0) ↔ (x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  -- Third statement is correct
  (¬(∀ x : ℝ, (2 : ℝ)^x > 0) ↔ (∃ x : ℝ, (2 : ℝ)^x ≤ 0)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_third_statement_correct_l349_34959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bubble_sort_largest_floats_up_l349_34915

/-- Represents a list of numbers to be sorted --/
def SortList := List ℕ

/-- Represents a single pass of bubble sort --/
def BubblePass (l : SortList) : SortList :=
  sorry

/-- Checks if a list is sorted in ascending order --/
def IsSorted (l : SortList) : Prop :=
  sorry

/-- Theorem stating that bubble sort moves the largest number to the end with each pass --/
theorem bubble_sort_largest_floats_up (l : SortList) :
  ∀ n, n > 0 → n ≤ l.length →
  (BubblePass^[n] l).getLast? = l.maximum :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bubble_sort_largest_floats_up_l349_34915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_2_l349_34977

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then (1/2)^x else 1/(x-1)

theorem f_composition_at_2 : f (f 2) = -4/3 := by
  -- Evaluate f(2)
  have h1 : f 2 = 1/4 := by
    simp [f]
    norm_num
  
  -- Evaluate f(1/4)
  have h2 : f (1/4) = -4/3 := by
    simp [f]
    norm_num
  
  -- Combine the results
  rw [h1, h2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_2_l349_34977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_a_distance_at_meeting_l349_34934

/-- Represents the meeting point of two trains traveling towards each other. -/
noncomputable def train_meeting_point (total_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) : ℝ :=
  (total_distance * speed_a) / (speed_a + speed_b)

/-- Proves that Train A travels approximately 50 miles when it meets Train B. -/
theorem train_a_distance_at_meeting :
  let total_distance : ℝ := 125
  let time_a : ℝ := 12
  let time_b : ℝ := 8
  let speed_a : ℝ := total_distance / time_a
  let speed_b : ℝ := total_distance / time_b
  let meeting_distance : ℝ := train_meeting_point total_distance speed_a speed_b
  ∃ ε > 0, |meeting_distance - 50| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_a_distance_at_meeting_l349_34934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_branch_l349_34985

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define a function to represent a circle with center (a, b) and radius r
def tangent_circle (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- Define the locus of centers of circles tangent to both given circles
def locus (x y : ℝ) : Prop :=
  ∃ (r : ℝ), (∀ (x' y' : ℝ), circle1 x' y' → (x - x')^2 + (y - y')^2 = (r + 1)^2) ∧
             (∀ (x' y' : ℝ), circle2 x' y' → (x - x')^2 + (y - y')^2 = (r + 2)^2)

-- Theorem stating that the locus is one branch of a hyperbola
theorem locus_is_hyperbola_branch : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (∀ (x y : ℝ), locus x y ↔ (x - c)^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_branch_l349_34985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_k_for_coplanar_lines_l349_34947

/-- Two lines in 3D space -/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Check if two lines are coplanar -/
def are_coplanar (l1 l2 : Line3D) : Prop :=
  let v1 := l1.direction
  let v2 := l2.direction
  let v3 := λ i => l2.point i - l1.point i
  ∃ (a b c : ℝ), (∀ i : Fin 3, a * v1 i + b * v2 i + c * v3 i = 0) ∧
                 (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

/-- The main theorem -/
theorem no_real_k_for_coplanar_lines :
  ¬∃ (k : ℝ), are_coplanar
    (Line3D.mk (λ i => [3, 2, 7].get i) (λ i => [2, 1, -k].get i))
    (Line3D.mk (λ i => [4, 5, 6].get i) (λ i => [k, 3, 2].get i)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_k_for_coplanar_lines_l349_34947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_area_and_tiles_l349_34919

-- Define the room dimensions
noncomputable def room_length : ℚ := 52/10
noncomputable def room_width : ℚ := room_length / (13/10)

-- Define the tile size
noncomputable def tile_side : ℚ := 4/10

-- Theorem statement
theorem room_area_and_tiles :
  (room_length * room_width = 208/10) ∧
  (Int.ceil (room_length * room_width / (tile_side * tile_side)) = 130) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_area_and_tiles_l349_34919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slot_distribution_l349_34958

-- Define the function for number of ways to distribute
def number_of_ways_to_distribute (n m : ℕ) : ℕ := Nat.choose (n - 1) (m - 1)

-- State the theorem
theorem slot_distribution (n m : ℕ) (hn : n ≥ m) (hm : m > 0) :
  number_of_ways_to_distribute n m = Nat.choose (n - 1) (m - 1) :=
by
  -- Unfold the definition of number_of_ways_to_distribute
  unfold number_of_ways_to_distribute
  -- The equality is now trivial
  rfl

-- Evaluate the specific case for the problem
#eval number_of_ways_to_distribute 10 7  -- Should output 84

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slot_distribution_l349_34958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l349_34928

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-4, 7)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ := (dot_product v w) / (magnitude w)

theorem projection_a_on_b : projection a b = Real.sqrt 65 / 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l349_34928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_multiplication_property_l349_34988

/-- Reverses the digits of a positive integer -/
def rev (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of numbers with the special multiplication property -/
theorem special_multiplication_property :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  ∃ (c : ℕ), c = a * b ∧ rev c = rev a * rev b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_multiplication_property_l349_34988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_bijective_l349_34924

/-- Given a function f : ℝ → ℝ satisfying f(f(x) - 1) = x + 1 for all x ∈ ℝ,
    prove that f is bijective. -/
theorem f_is_bijective (f : ℝ → ℝ) (h : ∀ x, f (f x - 1) = x + 1) :
  Function.Bijective f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_bijective_l349_34924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_value_sin_cos_difference_l349_34946

theorem trig_expression_value : 
  Real.sin (120 * π / 180)^2 + Real.cos (180 * π / 180) + Real.tan (45 * π / 180) - 
  Real.cos (-330 * π / 180)^2 + Real.sin (-210 * π / 180) = -1/2 := by
  sorry

theorem sin_cos_difference (α : Real) 
  (h : Real.sin (π + α) = 1/2) 
  (h_range : π < α ∧ α < 3*π/2) : 
  Real.sin α - Real.cos α = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_value_sin_cos_difference_l349_34946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l349_34970

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the angle
structure Angle where
  vertex : ℝ × ℝ
  side1 : (ℝ × ℝ) → Prop
  side2 : (ℝ × ℝ) → Prop

-- Define the tangency condition
def isTangent (c : Circle) (l : (ℝ × ℝ) → Prop) : Prop :=
  ∃ p, l p ∧ (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Main theorem
theorem fixed_point_exists (c1 c2 : Circle) (α : Angle) 
  (h1 : isTangent c1 α.side1) (h2 : isTangent c2 α.side2) 
  (h3 : (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 > (c1.radius + c2.radius)^2) :
  ∃ p : ℝ × ℝ, ∀ θ : ℝ, 
    let rotated_angle := Angle.mk α.vertex 
      (λ q ↦ (q.1 - α.vertex.1) * Real.cos θ - (q.2 - α.vertex.2) * Real.sin θ + α.vertex.1 = q.1 ∧
            (q.1 - α.vertex.1) * Real.sin θ + (q.2 - α.vertex.2) * Real.cos θ + α.vertex.2 = q.2)
      (λ q ↦ (q.1 - α.vertex.1) * Real.cos θ - (q.2 - α.vertex.2) * Real.sin θ + α.vertex.1 = q.1 ∧
            (q.1 - α.vertex.1) * Real.sin θ + (q.2 - α.vertex.2) * Real.cos θ + α.vertex.2 = q.2)
    ∃ t : ℝ, rotated_angle.side1 (α.vertex.1 + t * (p.1 - α.vertex.1), α.vertex.2 + t * (p.2 - α.vertex.2)) ∨
           rotated_angle.side2 (α.vertex.1 + t * (p.1 - α.vertex.1), α.vertex.2 + t * (p.2 - α.vertex.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l349_34970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l349_34996

/-- The length of the chord intercepted by a circle on a line -/
noncomputable def chord_length (a b c d e f g : ℝ) : ℝ :=
  let center := (c, d)
  let radius := Real.sqrt e
  let distance_to_line := abs (a * c + b * d - f) / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (radius^2 - distance_to_line^2)

/-- Theorem stating the length of the chord intercepted by the given circle on the given line -/
theorem chord_length_specific_case :
  chord_length 3 4 2 1 4 5 0 = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l349_34996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_a_gt_b_necessary_not_sufficient_l349_34922

-- Define the function f(x) = log₂(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_condition (a b : ℝ) :
  (∀ x > 0, f x > f b → x > b) ∧
  ¬(∀ x > 0, x > b → f x > f b) :=
by
  sorry

-- Prove that "a > b" is a necessary but not sufficient condition for "f(a) > f(b)"
theorem a_gt_b_necessary_not_sufficient :
  (∀ a b : ℝ, a > 0 → b > 0 → f a > f b → a > b) ∧
  ¬(∀ a b : ℝ, a > 0 → b > 0 → a > b → f a > f b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_a_gt_b_necessary_not_sufficient_l349_34922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_with_property_l349_34952

def digits_appear_once (n : ℕ) : Prop :=
  ∃ (s : List ℕ), s.length = 10 ∧ 
  (∀ d, d ∈ s ↔ d < 10) ∧
  (s.Nodup) ∧
  (s = (n^3).digits 10 ++ (n^4).digits 10)

theorem unique_n_with_property : 
  ∃! n : ℕ, digits_appear_once n ∧ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_with_property_l349_34952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_inscribed_cube_l349_34914

/-- The volume of a cube inscribed in a sphere of radius R is (8R^3) / (3√3) -/
theorem volume_of_inscribed_cube (R : ℝ) (R_pos : R > 0) :
  ∃ V : ℝ, V = (8 * R^3) / (3 * Real.sqrt 3) ∧
    V = (λ a : ℝ ↦ a^3) ((2 * R) / Real.sqrt 3) :=
by
  -- Introduce the volume V
  let V := (8 * R^3) / (3 * Real.sqrt 3)
  
  -- Show that V satisfies both conditions
  have h1 : V = (8 * R^3) / (3 * Real.sqrt 3) := by rfl
  
  have h2 : V = (λ a : ℝ ↦ a^3) ((2 * R) / Real.sqrt 3) := by
    -- This step requires algebraic manipulation
    sorry
  
  -- Conclude the proof
  exact ⟨V, ⟨h1, h2⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_inscribed_cube_l349_34914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fractions_satisfying_conditions_l349_34938

theorem no_fractions_satisfying_conditions :
  ¬∃ (x y : ℕ+), 
    (x < y) ∧ 
    (Nat.Coprime x.val y.val) ∧ 
    ((2 * x + 1 : ℚ) / (2 * y + 1 : ℚ) = 4/5 * (x : ℚ) / (y : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fractions_satisfying_conditions_l349_34938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_roots_implies_a_equals_5_l349_34918

/-- A natural number is prime if it's greater than 1 and its only positive divisors are 1 and itself. -/
def IsPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The quadratic equation x^2 - ax + a + 1 = 0 -/
def QuadraticEq (a x : ℝ) : Prop :=
  x^2 - a*x + a + 1 = 0

theorem prime_roots_implies_a_equals_5 (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    QuadraticEq a x₁ ∧ QuadraticEq a x₂ ∧ 
    IsPrime (Int.toNat ⌈x₁⌉) ∧ IsPrime (Int.toNat ⌈x₂⌉)) →
  a = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_roots_implies_a_equals_5_l349_34918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_29325_div_182_l349_34948

/-- Triangle with inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The length of DP, where P is the tangent point on DE -/
  dp : ℝ
  /-- The length of PE, where P is the tangent point on DE -/
  pe : ℝ

/-- Calculate the perimeter of the triangle -/
noncomputable def perimeter (t : TriangleWithInscribedCircle) : ℝ :=
  29325 / 182

/-- Theorem stating that the perimeter of the triangle is 29325/182 -/
theorem triangle_perimeter_is_29325_div_182 (t : TriangleWithInscribedCircle) 
  (h1 : t.radius = 15) 
  (h2 : t.dp = 19) 
  (h3 : t.pe = 31) : 
  perimeter t = 29325 / 182 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_29325_div_182_l349_34948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_closed_form_inverse_inverse_requires_numerical_methods_l349_34953

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 2 * x^2 + 5 * x - 9 + 3 * Real.sin x + 2 * Real.exp x

-- Theorem stating that f does not have a closed-form inverse
theorem no_closed_form_inverse :
  ¬ ∃ g : ℝ → ℝ, ∀ x : ℝ, g (f x) = x ∧ f (g x) = x := by
  sorry

-- Theorem stating that finding f⁻¹(-3.5) requires numerical methods
theorem inverse_requires_numerical_methods :
  ∀ y : ℝ, f y = -3.5 → (∀ ε > 0, ∃ δ > 0, |y - (f⁻¹ (-3.5))| < ε → |f y - (-3.5)| < δ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_closed_form_inverse_inverse_requires_numerical_methods_l349_34953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_myopia_study_results_l349_34973

/-- Data for the myopia study -/
structure MyopiaData where
  school_a_myopic : ℕ
  school_a_non_myopic : ℕ
  school_b_myopic : ℕ
  school_b_non_myopic : ℕ

/-- Calculate K² statistic -/
noncomputable def calculate_k_squared (data : MyopiaData) : ℝ :=
  let a := data.school_a_myopic
  let b := data.school_a_non_myopic
  let c := data.school_b_myopic
  let d := data.school_b_non_myopic
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating the results of the myopia study -/
theorem myopia_study_results (data : MyopiaData)
  (h_data : data = {
    school_a_myopic := 250,
    school_a_non_myopic := 250,
    school_b_myopic := 300,
    school_b_non_myopic := 200
  }) :
  let freq_a := (data.school_a_myopic : ℝ) / (data.school_a_myopic + data.school_a_non_myopic)
  let freq_b := (data.school_b_myopic : ℝ) / (data.school_b_myopic + data.school_b_non_myopic)
  freq_a = 0.5 ∧ freq_b = 0.6 ∧ calculate_k_squared data > 6.635 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_myopia_study_results_l349_34973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l349_34936

theorem problem_solution : 
  ((-2)^2 - Real.rpow 27 (1/3) + Real.sqrt 16 + (-1)^2023 = 4) ∧ 
  (|Real.sqrt 2 - Real.sqrt 3| + 2 * Real.sqrt 2 = Real.sqrt 3 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l349_34936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_relation_l349_34911

def is_median (s : Finset ℤ) (m : ℤ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧ 2 * (s.filter (· ≥ m)).card ≥ s.card

def is_mean (s : Finset ℤ) (μ : ℚ) : Prop :=
  μ = (s.sum (fun x => (x : ℚ))) / s.card

theorem median_mean_relation (x : ℤ) :
  x < 0 →
  let s : Finset ℤ := {20, 50, 55, x, 22}
  ∃ (m : ℤ) (μ : ℚ), is_median s m ∧ is_mean s μ ∧ (m : ℚ) + 7 = μ →
  x = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_relation_l349_34911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_congruent_to_two_mod_four_l349_34974

theorem three_digit_integers_congruent_to_two_mod_four :
  (Finset.filter (fun n => n % 4 = 2) (Finset.range 900)).card = 225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_congruent_to_two_mod_four_l349_34974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_inequality_l349_34991

-- Define the interval (1, 2]
def OpenClosedInterval : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Define the inequality condition
def InequalityCondition (a : ℝ) : Prop :=
  ∀ x ∈ OpenClosedInterval, (x - 1)^2 ≤ Real.log x / Real.log a

-- Define the range of a
def RangeOfA : Set ℝ := {a | 1 < a ∧ a ≤ 2}

-- Theorem statement
theorem range_of_a_given_inequality :
  (∃ a : ℝ, InequalityCondition a) → (∀ a : ℝ, InequalityCondition a ↔ a ∈ RangeOfA) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_inequality_l349_34991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_XG_GY_ratio_l349_34994

-- Define the triangle and points
variable (X Y Z E G Q : ℝ × ℝ)

-- Define the condition that E is on XZ and G is on XY
def E_on_XZ (X Z E : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = X + t • (Z - X)
def G_on_XY (X Y G : ℝ × ℝ) : Prop := ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ G = X + s • (Y - X)

-- Define the condition that Q is the intersection of XE and YG
def Q_intersection (X Y Z E G Q : ℝ × ℝ) : Prop := ∃ u v : ℝ, 0 < u ∧ u < 1 ∧ 0 < v ∧ v < 1 ∧
  Q = X + u • (E - X) ∧ Q = Y + v • (G - Y)

-- Define the given ratios
def XQ_QE_ratio (X E Q : ℝ × ℝ) : Prop := ∃ k : ℝ, k > 0 ∧ Q = X + (3/5) • k • (E - X)
def GQ_QZ_ratio (G Z Q : ℝ × ℝ) : Prop := ∃ m : ℝ, m > 0 ∧ Q = G + (2/5) • m • (Z - G)

-- State the theorem
theorem XG_GY_ratio
  (h1 : E_on_XZ X Z E)
  (h2 : G_on_XY X Y G)
  (h3 : Q_intersection X Y Z E G Q)
  (h4 : XQ_QE_ratio X E Q)
  (h5 : GQ_QZ_ratio G Z Q) :
  ∃ t : ℝ, t = 1/3 ∧ G = X + t • (Y - X) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_XG_GY_ratio_l349_34994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_count_l349_34984

-- Define the basic geometric structures
structure Point where
  x : ℝ
  y : ℝ

def dist (p q : Point) : ℝ := sorry

def angle (p q r : Point) : ℝ := sorry

def on_line (p q r : Point) : Prop := sorry

-- Define the triangle ABC
structure Triangle (A B C : Point) : Prop where
  ab_eq_ac : dist A B = dist A C
  angle_abc : angle A B C = 72

-- Define the bisector BD
structure Bisector (B D : Point) (ABC : Triangle A B C) : Prop where
  bisects : angle A B D = angle D B C
  d_on_ac : on_line A C D

-- Define parallel segments
def parallel (p1 p2 q1 q2 : Point) : Prop := sorry

-- Define the configuration
structure Configuration (A B C D E F : Point) : Prop where
  triangle : Triangle A B C
  bisector : Bisector B D triangle
  e_on_bc : on_line B C E
  de_parallel_ab : parallel D E A B
  f_on_ac : on_line A C F
  ef_parallel_bd : parallel E F B D

-- Define an isosceles triangle
def isosceles (A B C : Point) : Prop := 
  (dist A B = dist A C) ∨ (dist B A = dist B C) ∨ (dist C A = dist C B)

-- The main theorem
theorem isosceles_count (A B C D E F : Point) (config : Configuration A B C D E F) :
  ∃ (triangles : Finset (Point × Point × Point)), 
    triangles.card = 7 ∧ 
    ∀ t ∈ triangles, isosceles t.1 t.2.1 t.2.2 ∧
    ∀ p q r, isosceles p q r → (p, q, r) ∈ triangles :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_count_l349_34984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l349_34912

/-- Calculate the volume of a cylindrical tube formed by rolling a rectangular sheet along its longer side -/
noncomputable def cylinderVolume (width : ℝ) (length : ℝ) : ℝ :=
  (width * length^2) / (4 * Real.pi)

/-- The problem statement -/
theorem cylinder_volume_difference : 
  Real.pi * (cylinderVolume 9 12 - cylinderVolume 7 10) = 149 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l349_34912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x₀_l349_34969

-- Define the circle C
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line that P is on
def my_line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the angle condition
noncomputable def angle_condition (x₀ y₀ : ℝ) : Prop :=
  ∃ (x y : ℝ), my_circle x y ∧ Real.sqrt ((x - x₀)^2 + (y - y₀)^2) * Real.sqrt (x₀^2 + y₀^2) * Real.cos (30 * Real.pi / 180) =
    x * x₀ + y * y₀

-- Theorem statement
theorem range_of_x₀ (x₀ y₀ : ℝ) : 
  my_line x₀ y₀ → angle_condition x₀ y₀ → 0 ≤ x₀ ∧ x₀ ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x₀_l349_34969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l349_34937

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := y - 1/2 = -3/4 * (x - 1)

-- Define the line AB
def line_AB (x y : ℝ) : Prop := y = -2 * (x - 1)

-- Define the ellipse
def my_ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_equation :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  (∃ c : ℝ, c > 0 ∧ a^2 = b^2 + c^2) →
  (∃ x₀ y₀ : ℝ, my_circle x₀ y₀ ∧ tangent_line x₀ y₀) →
  (∃ x₁ y₁ : ℝ, my_circle x₁ y₁ ∧ line_AB x₁ y₁) →
  line_AB 1 0 →
  line_AB 0 2 →
  a^2 = 5 ∧ b^2 = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l349_34937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_excircle_area_relation_l349_34987

-- Define the structure for a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the structure for circle contact points
structure CircleContacts where
  A : Point
  B : Point
  C : Point

-- Define the function to calculate triangle area
noncomputable def triangleArea (t : CircleContacts) : ℝ := sorry

-- Define the acute triangle ABC
variable (ABC : Triangle)

-- Assume ABC is acute (we'll use a placeholder for now)
axiom ABC_is_acute : True  -- Placeholder, replace with appropriate condition if available

-- Define the incircle contact points
variable (incircle : CircleContacts)

-- Define the excircle contact points
variable (excircle1 excircle2 excircle3 : CircleContacts)

-- Define areas
noncomputable def T₀ : ℝ := triangleArea incircle
noncomputable def T₁ : ℝ := triangleArea excircle1
noncomputable def T₂ : ℝ := triangleArea excircle2
noncomputable def T₃ : ℝ := triangleArea excircle3

-- State the theorem
theorem incircle_excircle_area_relation :
  1 / T₀ = 1 / T₁ + 1 / T₂ + 1 / T₃ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_excircle_area_relation_l349_34987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_simplification_l349_34904

theorem log_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log (x^2) / Real.log (y^4)) * (Real.log (y^3) / Real.log (x^3)) * 
  (Real.log (x^4) / Real.log (y^5)) * (Real.log (y^5) / Real.log (x^2)) * 
  (Real.log (x^3) / Real.log (y^3)) = (1/3) * (Real.log x / Real.log y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_simplification_l349_34904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_l349_34951

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6 × Fin 6

-- Define event A: all three numbers are different
def A : Set Ω := {ω | ω.1 ≠ ω.2.1 ∧ ω.1 ≠ ω.2.2 ∧ ω.2.1 ≠ ω.2.2}

-- Define event B: at least one 6 appears
def B : Set Ω := {ω | ω.1 = 5 ∨ ω.2.1 = 5 ∨ ω.2.2 = 5}

-- Define a probability measure on Ω
variable (P : Set Ω → ℝ)
variable (hP : Probability P)

-- State the theorem
theorem dice_probability :
  P (A ∩ B) / P B = 60 / 91 ∧ P (A ∩ B) / P A = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_l349_34951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_l349_34929

/-- The diameter of each cylindrical pipe in centimeters -/
def pipe_diameter : ℝ := 8

/-- The number of rows in Crate A -/
def crate_a_rows : ℕ := 25

/-- The number of rows in Crate B -/
def crate_b_rows : ℕ := 24

/-- The height of Crate A in centimeters -/
def crate_a_height : ℝ := crate_a_rows * pipe_diameter

/-- The vertical distance between rows in Crate B -/
noncomputable def crate_b_row_distance : ℝ := (Real.sqrt 3 / 2) * pipe_diameter

/-- The height of Crate B in centimeters -/
noncomputable def crate_b_height : ℝ := crate_b_rows * crate_b_row_distance

/-- The difference in height between Crate A and Crate B -/
theorem height_difference : crate_a_height - crate_b_height = 200 - 96 * Real.sqrt 3 := by
  sorry

#eval crate_a_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_l349_34929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_x_intercept_l349_34980

noncomputable def circle_center (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def circle_radius (center : ℝ × ℝ) (endpoint : ℝ × ℝ) : ℝ :=
  Real.sqrt ((center.1 - endpoint.1)^2 + (center.2 - endpoint.2)^2)

def circle_equation (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

theorem circle_x_intercept :
  let p1 : ℝ × ℝ := (-2, 0)
  let p2 : ℝ × ℝ := (6, 4)
  let center := circle_center p1 p2
  let radius := circle_radius center p1
  ∃ x : ℝ, x ≠ -2 ∧ circle_equation center radius x 0 ∧ x = 6 :=
by
  sorry

#check circle_x_intercept

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_x_intercept_l349_34980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l349_34933

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^m - 2/x

-- State the theorem
theorem function_properties :
  ∃ m : ℝ,
    (f m 4 = 7/2) ∧
    (m = 1) ∧
    (∀ x : ℝ, x ≠ 0 → f m (-x) = -(f m x)) ∧
    (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 0 → f m x₁ > f m x₂) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l349_34933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_2_3_l349_34986

-- Define the function f(x) = x + ln x - 4
noncomputable def f (x : ℝ) : ℝ := x + Real.log x - 4

-- State the theorem
theorem zero_of_f_in_interval_2_3 :
  ∃ x : ℝ, x > 2 ∧ x < 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_2_3_l349_34986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_reach_54_from_12_in_60_steps_l349_34965

def is_valid_operation (n : ℕ) : ℕ → Prop :=
  λ m => m = n * 2 ∨ m = n * 3 ∨ m = n / 2 ∨ m = n / 3

def sequence_of_operations (start : ℕ) (n : ℕ) : (Fin (n + 1) → ℕ) → Prop :=
  λ seq => seq 0 = start ∧ ∀ i : Fin n, is_valid_operation (seq i) (seq i.succ)

theorem cannot_reach_54_from_12_in_60_steps :
  ¬ ∃ (seq : Fin 61 → ℕ), sequence_of_operations 12 60 seq ∧ seq 60 = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_reach_54_from_12_in_60_steps_l349_34965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_F_to_line_l349_34949

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance from point F(1, 0) to the line θ = π/4 (ρ ∈ R) in polar coordinates -/
noncomputable def distance_to_line (F : Point) (line : Set Point) : ℝ :=
  sorry

/-- Point F in polar coordinates -/
def F : Point :=
  { x := 1, y := 0 }

/-- The line θ = π/4 (ρ ∈ R) in polar coordinates -/
def line : Set Point :=
  { p : Point | p.y = p.x }

theorem distance_from_F_to_line : distance_to_line F line = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_F_to_line_l349_34949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_of_coefficients_l349_34979

theorem cubic_polynomial_sum_of_coefficients (p q r : ℝ) (w : ℂ) :
  let Q : ℂ → ℂ := λ z => z^3 + p*z^2 + q*z + r
  (∀ z, Q z = 0 ↔ z ∈ ({w + 4*Complex.I, w + 10*Complex.I, 3*w - 5} : Set ℂ)) →
  p + q + r = -235/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_of_coefficients_l349_34979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l349_34917

theorem power_equation_solution (x : ℝ) : (5 : ℝ)^(x+2) = 625 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l349_34917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_2011_l349_34916

/-- Represents a star with 11 numbers -/
structure StarStruct where
  numbers : Fin 11 → ℕ
  distinct : ∀ i j, i ≠ j → numbers i ≠ numbers j

/-- The sequence of all stars -/
def starSequence : ℕ → StarStruct := sorry

/-- The position of a number within a star (1-based index) -/
def positionInStar (n : ℕ) : Fin 11 := sorry

/-- The star index containing a given number (1-based index) -/
def starIndex (n : ℕ) : ℕ := sorry

theorem star_2011 :
  let s := starSequence 183
  starIndex 2011 = 183 ∧
  positionInStar 2011 = ⟨9, sorry⟩ ∧
  (∀ i : Fin 11, s.numbers i = 2003 + i) := by
  sorry

#check star_2011

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_2011_l349_34916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_with_restriction_l349_34960

def seating_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

def adjacent_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 2)

theorem circular_seating_with_restriction (n : ℕ) (h : n = 8) :
  seating_arrangements n - adjacent_arrangements n = 3600 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_with_restriction_l349_34960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_visit_permutations_l349_34995

/-- The number of permutations of n elements with two precedence constraints -/
def permutations_with_two_constraints (n : ℕ) : ℕ :=
  (Nat.factorial n) / 4

theorem hotel_visit_permutations :
  permutations_with_two_constraints 5 = 30 := by
  rfl

#eval permutations_with_two_constraints 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_visit_permutations_l349_34995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_sequence_properties_l349_34964

def a : ℕ → ℕ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 1 => if n % 2 = 0 then a n + 2 else a n + 1

def b (n : ℕ) : ℕ := a (2 * n)

theorem b_sequence_properties :
  (b 1 = 2) ∧ (∀ n : ℕ, n ≥ 1 → b n = 3 * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_sequence_properties_l349_34964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l349_34907

/-- The slope of the line -/
def line_slope : ℚ := -3/4

/-- The y-intercept of the line -/
def line_y_intercept : ℚ := 3

/-- The x-coordinate of the initial point in the parameterization -/
def x_init : ℚ := 6

/-- The y-component of the direction vector in the parameterization -/
def y_dir : ℚ := 7

/-- The theorem stating that the parameterization of the line y = -3/4x + 3
    as (x, y) = (6, s) + t(m, 7) yields s = -3/2 and m = -7/3 -/
theorem line_parameterization :
  ∃ (s m : ℚ),
    (∀ (t : ℚ), line_y_intercept + line_slope * (x_init + t * m) = s + t * y_dir) ∧
    s = -3/2 ∧
    m = -7/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l349_34907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_integral_half_to_three_halves_pi_l349_34920

theorem sin_integral_half_to_three_halves_pi : ∫ x in (π / 2)..(3 * π / 2), Real.sin x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_integral_half_to_three_halves_pi_l349_34920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_base_l349_34992

theorem log_equality_implies_base (x : ℝ) :
  x > 0 → (Real.log 16 / Real.log x = Real.log 4 / Real.log 64) → x = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_base_l349_34992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l349_34909

/-- Represents an arithmetic sequence -/
def ArithmeticSequence : Type := ℝ × ℝ

/-- The nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.1 + seq.2 * (n - 1 : ℝ)

theorem arithmetic_sequence_difference : 
  let C : ArithmeticSequence := (20, 15)
  let D : ArithmeticSequence := (20, -15)
  |nthTerm C 61 - nthTerm D 61| = 1800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l349_34909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l349_34982

/-- Given a point P(-8m, -3) on the terminal side of angle α where cos α = -4/5, prove that m = 3/4 -/
theorem point_on_terminal_side (m : ℝ) (α : ℝ) : 
  (∃ P : ℝ × ℝ, P = (-8*m, -3) ∧ P.1 < 0 ∧ P.2 < 0) →  -- Point P(-8m, -3) is on the terminal side (third quadrant)
  Real.cos α = -4/5 →                                 -- cos α = -4/5
  m = 3/4                                             -- Prove m = 3/4
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l349_34982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_of_given_ellipse_l349_34932

/-- The distance between the foci of an ellipse -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- The ellipse equation -/
def is_ellipse (x y : ℝ) : Prop :=
  25 * x^2 - 125 * x + 4 * y^2 + 8 * y + 16 = 0

/-- Theorem: The distance between the foci of the given ellipse is 2√4.62 -/
theorem distance_between_foci_of_given_ellipse :
  ∃ (a b : ℝ), (∀ x y, is_ellipse x y ↔ (x - 5/2)^2 / a^2 + (y + 1)^2 / b^2 = 1) ∧
  distance_between_foci a b = 2 * Real.sqrt 4.62 := by
  sorry

#check distance_between_foci_of_given_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_of_given_ellipse_l349_34932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_half_implies_fraction_l349_34903

theorem tan_alpha_half_implies_fraction (α : ℝ) (h : Real.tan α = 1 / 2) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_half_implies_fraction_l349_34903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_and_cylinder_volumes_l349_34913

/-- Given a cone with base diameter and height both equal to 6 cm,
    prove the volume of the cone and the volume of a cylinder with the same base and height. -/
theorem cone_and_cylinder_volumes :
  ∃ (cone_volume cylinder_volume : ℝ),
    cone_volume = π * 6^2 / 4 ∧ cylinder_volume = 3 * π * 6^2 / 4 := by
  let cone_height : ℝ := 6
  let cone_base_diameter : ℝ := 6
  let cone_volume := (π * cone_base_diameter^2 * cone_height) / 12
  let cylinder_volume := (π * cone_base_diameter^2 * cone_height) / 4
  use cone_volume, cylinder_volume
  constructor
  · -- Proof for cone volume
    sorry
  · -- Proof for cylinder volume
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_and_cylinder_volumes_l349_34913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l349_34972

def arithmetic_sequence_sum (a : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1)) / 2

def sequence_difference : ℕ :=
  let first_sequence_sum := arithmetic_sequence_sum 1901 93
  let second_sequence_sum := arithmetic_sequence_sum 101 93
  first_sequence_sum - second_sequence_sum

#eval sequence_difference

theorem main_theorem : sequence_difference = 167400 := by
  -- Unfold the definitions
  unfold sequence_difference
  unfold arithmetic_sequence_sum
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l349_34972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equal_if_floor_equal_l349_34939

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_equal_if_floor_equal
  (f g : ℝ → ℝ)
  (hf : is_quadratic f)
  (hg : is_quadratic g)
  (h : ∀ x, floor (f x) = floor (g x)) :
  ∀ x, f x = g x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equal_if_floor_equal_l349_34939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pies_differ_in_both_l349_34975

/-- Enumeration of pie fillings -/
inductive Filling
  | Apple
  | Cherry

/-- Enumeration of pie preparation methods -/
inductive Preparation
  | Fried
  | Baked

/-- A pie is characterized by its filling and preparation method -/
structure Pie where
  filling : Filling
  preparation : Preparation

/-- The set of all possible pies -/
def AllPies : Set Pie :=
  {p : Pie | p.filling = Filling.Apple ∨ p.filling = Filling.Cherry} ∩
  {p : Pie | p.preparation = Preparation.Fried ∨ p.preparation = Preparation.Baked}

/-- Two pies differ in both filling and preparation -/
def DifferInBoth (p1 p2 : Pie) : Prop :=
  p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation

/-- Main theorem: If there are at least three types of pies, 
    we can find two that differ in both filling and preparation -/
theorem pies_differ_in_both 
  (h : ∃ (s : Finset Pie), s.toSet ⊆ AllPies ∧ s.card ≥ 3) : 
  ∃ (p1 p2 : Pie), p1 ∈ AllPies ∧ p2 ∈ AllPies ∧ DifferInBoth p1 p2 := by
  sorry

/-- AllPies is finite -/
instance : Finite AllPies := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pies_differ_in_both_l349_34975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_function_a_range_l349_34963

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((3 - a) * x - a) / Real.log a

-- State the theorem
theorem increasing_log_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Ioo 1 3 ∪ {3} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_function_a_range_l349_34963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_restoration_l349_34997

noncomputable def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount / 100)

noncomputable def apply_tax (price : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 + tax_rate / 100)

noncomputable def percentage_increase (original_price : ℝ) (final_price : ℝ) : ℝ :=
  (original_price / final_price - 1) * 100

theorem jacket_price_restoration (P : ℝ) (h : P > 0) :
  let discounted_price := apply_discount (apply_discount (apply_discount (apply_discount P 25) 15) 10) 5
  let taxed_price := apply_tax discounted_price 7
  abs (percentage_increase P taxed_price - 71.45) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_restoration_l349_34997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_points_with_integer_distances_l349_34967

/-- A type representing a point in the plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- The distance between two points is an integer -/
def integerDistance (p1 p2 : Point) : Prop :=
  ∃ d : ℤ, d^2 = (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Not all points in a list lie on the same line -/
def notAllCollinear (points : List Point) : Prop :=
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    (p2.y - p1.y) * (p3.x - p1.x) ≠ (p3.y - p1.y) * (p2.x - p1.x)

/-- The main theorem statement -/
theorem exists_n_points_with_integer_distances (n : ℕ) (h : n > 2) :
  ∃ (points : List Point),
    points.length = n ∧
    (∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → integerDistance p1 p2) ∧
    notAllCollinear points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_points_with_integer_distances_l349_34967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_l349_34978

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_ab : a + b = 26)
  (sum_abcd : a + b + c + d = 41)
  (sum_all : a + b + c + d + e + f = 57) :
  ∃ (odd_count : ℕ), odd_count = 1 ∧ 
  odd_count ≤ (a % 2).natAbs + (b % 2).natAbs + 
              (c % 2).natAbs + (d % 2).natAbs + 
              (e % 2).natAbs + (f % 2).natAbs :=
by
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_l349_34978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AOB_area_l349_34940

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Calculates the area of a triangle given two points in polar coordinates -/
noncomputable def triangleArea (a b : PolarPoint) : ℝ :=
  (1/2) * a.r * b.r * Real.sin (abs (b.θ - a.θ))

theorem triangle_AOB_area :
  let a : PolarPoint := ⟨2, 2*Real.pi/3⟩
  let b : PolarPoint := ⟨3, Real.pi/6⟩
  triangleArea a b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AOB_area_l349_34940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_polar_equation_l349_34942

/-- A point in the polar coordinate system -/
structure Punkt where
  ρ : ℝ
  θ : ℝ

/-- The equation of the circle in polar coordinates -/
def circle_equation (p : Punkt) : Prop := p.ρ = 4

/-- The polar axis in the polar coordinate system -/
def polar_axis : Set Punkt := {p | p.θ = 0 ∨ p.θ = Real.pi}

/-- A line in the polar coordinate system -/
noncomputable def l : Set Punkt := {p | p.ρ * Real.cos p.θ = 2 * Real.sqrt 3}

/-- The distance between two points -/
noncomputable def dist (A B : Punkt) : ℝ :=
  Real.sqrt ((A.ρ * Real.cos A.θ - B.ρ * Real.cos B.θ)^2 + (A.ρ * Real.sin A.θ - B.ρ * Real.sin B.θ)^2)

/-- Given a polar coordinate system with a circle ρ = 4, and a line l that:
    1) Intersects perpendicularly with the polar axis
    2) Intersects the circle at points A and B
    3) The length of AB is 4
    Then the polar equation of line l is ρcos(θ) = 2√3 -/
theorem line_polar_equation (A B : Punkt) :
  (∀ p, circle_equation p → p.ρ = 4) →
  (∀ p ∈ l, p.ρ * Real.cos p.θ = 2 * Real.sqrt 3) →
  A ∈ l ∧ circle_equation A →
  B ∈ l ∧ circle_equation B →
  dist A B = 4 →
  ∀ p ∈ l, p.ρ * Real.cos p.θ = 2 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_polar_equation_l349_34942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l349_34930

/-- The angle between the asymptotes of the hyperbola y²/3 - x² = 1 is 60° -/
theorem hyperbola_asymptote_angle :
  let hyperbola := {(x, y) : ℝ × ℝ | y^2 / 3 - x^2 = 1}
  let asymptote1 := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * x}
  let asymptote2 := {(x, y) : ℝ × ℝ | y = -Real.sqrt 3 * x}
  ∃ (angle : ℝ), angle = π / 3 ∧ 
    angle = Real.arccos ((1 + Real.sqrt 3 * (-Real.sqrt 3)) / 
      (Real.sqrt (1 + Real.sqrt 3 * Real.sqrt 3) * Real.sqrt (1 + (-Real.sqrt 3) * (-Real.sqrt 3)))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l349_34930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_proof_l349_34983

noncomputable def line (x : ℝ) : ℝ := (2 * x + 3) / 5

def parameterization (v d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (v.1 + t * d.1, v.2 + t * d.2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem direction_vector_proof (d : ℝ × ℝ) :
  (∀ x ≤ -3, ∃ t,
    let p := parameterization (-3, -1) d t
    line p.1 = p.2 ∧
    distance p (-3, -1) = t) →
  d = (5/2, 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_proof_l349_34983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l349_34906

/-- The focus of a parabola y² = 8x -/
def parabola_focus : ℝ × ℝ := (2, 0)

/-- The eccentricity of the hyperbola -/
noncomputable def hyperbola_eccentricity : ℝ := Real.sqrt 2

/-- The general equation of a hyperbola -/
def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The distance from the center to a focus of a hyperbola -/
noncomputable def hyperbola_c (a : ℝ) : ℝ := a * hyperbola_eccentricity

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (x y : ℝ), is_hyperbola a b x y) →
  (hyperbola_c a = parabola_focus.1) →
  (a^2 = 2 ∧ b^2 = 2) := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l349_34906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_C_highest_prob_prob_two_students_earn_credits_l349_34931

-- Define the probabilities for each student passing each exam
def prob_theo_A : ℚ := 5/6
def prob_theo_B : ℚ := 4/5
def prob_theo_C : ℚ := 3/4
def prob_prac_A : ℚ := 1/2
def prob_prac_B : ℚ := 2/3
def prob_prac_C : ℚ := 5/6

-- Define the probability of each student earning credits
def prob_credit (theo : ℚ) (prac : ℚ) : ℚ := theo * prac

-- Theorem 1: Student C has the highest probability of earning credits
theorem student_C_highest_prob :
  prob_credit prob_theo_C prob_prac_C > prob_credit prob_theo_A prob_prac_A ∧
  prob_credit prob_theo_C prob_prac_C > prob_credit prob_theo_B prob_prac_B := by
  sorry

-- Theorem 2: Probability of exactly two students earning credits
theorem prob_two_students_earn_credits :
  let p_A := prob_credit prob_theo_A prob_prac_A
  let p_B := prob_credit prob_theo_B prob_prac_B
  let p_C := prob_credit prob_theo_C prob_prac_C
  (1 - p_A) * p_B * p_C + p_A * (1 - p_B) * p_C + p_A * p_B * (1 - p_C) = 115/288 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_C_highest_prob_prob_two_students_earn_credits_l349_34931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_arc_circumference_l349_34900

-- Define the circle and points
def Circle : Type := ℝ × ℝ → Prop
def Point : Type := ℝ × ℝ

-- Define the radius of the circle
def radius : ℝ := 24

-- Define the circle with the given radius
def circleDefinition : Circle := λ (x, y) ↦ x^2 + y^2 = radius^2

-- Declare points P, Q, and R on the circle
variable (P Q R : Point)
axiom P_on_circle : circleDefinition P
axiom Q_on_circle : circleDefinition Q
axiom R_on_circle : circleDefinition R

-- Define angle PRQ
noncomputable def angle_PRQ : ℝ := 60 * Real.pi / 180

-- Define the circumference of the minor arc PQ
noncomputable def minor_arc_PQ : ℝ := 8 * Real.pi

-- State the theorem
theorem minor_arc_circumference :
  circleDefinition P ∧ circleDefinition Q ∧ circleDefinition R ∧ angle_PRQ = 60 * Real.pi / 180 →
  minor_arc_PQ = 8 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_arc_circumference_l349_34900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_power_polynomial_exists_l349_34908

theorem perfect_power_polynomial_exists (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 1) :
  ∃ (P : Polynomial ℤ), 
    (Polynomial.degree P = n) ∧ 
    (∀ i : ℕ, i ≤ n → ∃ k : ℕ, (P.eval (↑i : ℤ)) = m ^ k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_power_polynomial_exists_l349_34908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_increasing_on_positive_f_bounds_on_interval_l349_34905

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - 4 / x

-- Theorem for odd function property
theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by
  sorry

-- Theorem for monotonicity on (0, +∞)
theorem f_increasing_on_positive : ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Theorem for maximum and minimum values on [1, 4]
theorem f_bounds_on_interval : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → -3 ≤ f x ∧ f x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_increasing_on_positive_f_bounds_on_interval_l349_34905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_relation_l349_34990

/-- Given two infinite geometric series with specified conditions, prove that m = 4 -/
theorem geometric_series_relation (m : ℝ) : 
  let first_series := fun n : ℕ => 12 * (1/2)^n
  let second_series := fun n : ℕ => 12 * ((6 + m)/12)^n
  (∑' n, first_series n) * 3 = ∑' n, second_series n → m = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_relation_l349_34990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_fifths_l349_34945

-- Define the sequence b_n
def b : ℕ → ℚ
  | 0 => 2  -- Adding case for 0
  | 1 => 2
  | 2 => 3
  | (n + 3) => b (n + 2) + 2 * b (n + 1)

-- Define the series
noncomputable def series_sum : ℚ := ∑' n, b n / 3^(n + 1)

-- Theorem statement
theorem series_sum_equals_two_fifths : series_sum = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_fifths_l349_34945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milkshake_cost_l349_34941

noncomputable def initial_amount : ℝ := 20

noncomputable def cupcake_fraction : ℝ := 1/4
noncomputable def sandwich_fraction : ℝ := 0.30
noncomputable def toy_fraction : ℝ := 1/5

noncomputable def final_amount : ℝ := 3

theorem milkshake_cost :
  let amount_after_cupcakes := initial_amount * (1 - cupcake_fraction)
  let amount_after_sandwich := amount_after_cupcakes * (1 - sandwich_fraction)
  let amount_after_toy := amount_after_sandwich * (1 - toy_fraction)
  amount_after_toy - final_amount = 5.40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milkshake_cost_l349_34941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_and_mode_of_data_set_l349_34961

noncomputable def data_set : List ℚ := [50, 50, 60, 60, 60, 70, 70]

noncomputable def mean (xs : List ℚ) : ℚ := (xs.sum) / xs.length

noncomputable def mode (xs : List ℚ) : ℚ :=
  xs.foldl (λ acc x => if xs.count x > xs.count acc then x else acc) (xs.head!)

theorem mean_and_mode_of_data_set :
  mean data_set = 60 ∧ mode data_set = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_and_mode_of_data_set_l349_34961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l349_34926

theorem cosine_of_angle_through_point :
  ∀ α : ℝ,
  (∃ (t : ℝ), t > 0 ∧ t * (Real.cos α) = -1 ∧ t * (Real.sin α) = 2) →
  Real.cos α = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l349_34926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piglet_gave_four_balloons_l349_34944

-- Define the number of balloons each character prepared
variable (piglet_prepared : ℕ)
variable (pooh_prepared : ℕ)
variable (owl_prepared : ℕ)

-- Define the number of balloons that burst
variable (burst_balloons : ℕ)

-- Define the total number of balloons Eeyore received
variable (total_balloons : ℕ)

-- Axioms based on the problem conditions
axiom pooh_prep : pooh_prepared = 3 * piglet_prepared
axiom owl_prep : owl_prepared = 4 * piglet_prepared
axiom total_received : total_balloons = 60
axiom burst_count : burst_balloons = 4

-- Theorem to prove
theorem piglet_gave_four_balloons : 
  piglet_prepared - burst_balloons = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piglet_gave_four_balloons_l349_34944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_nine_terms_l349_34999

/-- An arithmetic sequence with given first and ninth terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 3
  ninth_term : a 9 = 11

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The sum of the first 9 terms of the given arithmetic sequence is 63 -/
theorem sum_nine_terms (seq : ArithmeticSequence) : sum_n_terms seq 9 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_nine_terms_l349_34999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exists_l349_34921

-- Define the function f(x)
noncomputable def f (x : ℝ) := Real.sqrt (x + 3) + Real.sqrt (3 * x - 2)

-- State the theorem
theorem unique_solution_exists :
  ∃! x₁ : ℝ, f x₁ = 7 ∧ x₁ ≥ 2/3 :=
by
  -- Assume the following conditions
  have h1 : ∀ x ≥ 2/3, f x ≥ 0 := by sorry
  have h2 : Continuous f := by sorry
  have h3 : StrictMono f := by sorry
  
  sorry -- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exists_l349_34921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_in_sequence_l349_34902

def sequenceQ (n : ℕ) : ℚ :=
  (9720 : ℚ) / (4 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ z : ℤ, q = ↑z

theorem integer_count_in_sequence :
  (∃ n : ℕ, is_integer (sequenceQ n) ∧ ¬is_integer (sequenceQ (n + 1))) ∧
  (∀ m : ℕ, m > 0 → ¬is_integer (sequenceQ m)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_in_sequence_l349_34902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l349_34962

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 16*x - 6*y + 89 = 0

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := y^2 = 8*x

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The theorem stating the smallest possible distance -/
theorem smallest_distance :
  ∃ (min_dist : ℝ),
    (∀ (x1 y1 x2 y2 : ℝ),
      circle_eq x1 y1 → parabola_eq x2 y2 →
        distance x1 y1 x2 y2 ≥ min_dist) ∧
    min_dist = 6 * Real.sqrt 2 - 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l349_34962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l349_34954

/-- The function f(x) = x - a√x is monotonically increasing on [1,4] -/
def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 4 → f x < f y

/-- The function f defined in terms of a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.sqrt x

theorem max_a_value :
  (∃ a : ℝ, is_monotone_increasing (f a)) →
  (∀ a : ℝ, is_monotone_increasing (f a) → a ≤ 2) ∧
  (∃ a : ℝ, a = 2 ∧ is_monotone_increasing (f a)) :=
by
  sorry

#check max_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l349_34954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l349_34998

/-- The function f(x) = |x^2 - 1| / (x - 1) -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 ∨ x ≥ 1 then x + 1 else -x - 1

/-- The function g(x, k) = kx - 2 -/
def g (x k : ℝ) : ℝ := k * x - 2

/-- The proposition that f and g intersect at two points iff k ∈ (0, 1) ∪ (1, 4) -/
theorem intersection_condition (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g x₁ k ∧ f x₂ = g x₂ k) ↔
  (k > 0 ∧ k < 1) ∨ (k > 1 ∧ k < 4) := by
  sorry

#check intersection_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l349_34998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_cos_x_squared_l349_34993

-- Define the function (marked as noncomputable due to Real.cos)
noncomputable def f (x : ℝ) : ℝ := Real.cos (1 + x^2)

-- State the theorem
theorem derivative_of_cos_x_squared (x : ℝ) :
  deriv f x = -2 * x * Real.sin (1 + x^2) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_cos_x_squared_l349_34993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_all_reals_l349_34976

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / (2 - x)^3

-- State the theorem about the range of f
theorem range_of_f_is_all_reals :
  ∀ y : ℝ, ∃ x : ℝ, f x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_all_reals_l349_34976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleonoras_age_l349_34989

theorem eleonoras_age (e m : ℕ) 
  (first_condition : m - e = 3 * (e - (m - e)))
  (second_condition : 3 * e + (m + 2 * e) = 100) : 
  e = 15 := by
  sorry

#check eleonoras_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleonoras_age_l349_34989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_profit_percent_l349_34966

/-- Calculates the profit percent for a retailer selling a radio -/
theorem radio_profit_percent (purchase_price overhead_expenses selling_price : ℝ) 
  (h1 : purchase_price = 225)
  (h2 : overhead_expenses = 20)
  (h3 : selling_price = 300) :
  abs (((selling_price - (purchase_price + overhead_expenses)) / (purchase_price + overhead_expenses)) * 100 - 22.45) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_profit_percent_l349_34966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sees_arrangements_l349_34971

/-- The number of unique arrangements of letters in a word with repeated letters -/
def arrangements (total : ℕ) (repeats : List ℕ) : ℕ :=
  Nat.factorial total / (repeats.foldl (λ acc x => acc * Nat.factorial x) 1)

/-- Theorem: The number of unique arrangements of "SEES" is 6 -/
theorem sees_arrangements :
  arrangements 4 [2, 2] = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sees_arrangements_l349_34971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l349_34957

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * t.a * Real.cos t.A = t.c * Real.cos t.B + t.b * Real.cos t.C)
  (h2 : t.a = 1)
  (h3 : Real.cos (t.B / 2) ^ 2 + Real.cos (t.C / 2) ^ 2 = 1 + Real.sqrt 3 / 4) :
  Real.cos t.A = 1 / 2 ∧ (t.c = 2 * Real.sqrt 3 / 3 ∨ t.c = Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l349_34957
