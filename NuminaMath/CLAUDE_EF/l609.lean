import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l609_60973

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := { x | x ≥ -1 }

-- State that f is monotonically increasing on its domain
axiom f_monotone : ∀ {x y : ℝ}, x ∈ domain → y ∈ domain → x ≤ y → f x ≤ f y

-- Define the inequality
def inequality (x : ℝ) : Prop := f (Real.exp (x - 2)) ≥ f (2 - x / 2)

-- State the theorem
theorem solution_set : 
  { x : ℝ | inequality x } = { x : ℝ | 2 ≤ x ∧ x ≤ 6 } :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l609_60973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colbert_treehouse_planks_l609_60929

theorem colbert_treehouse_planks (total : ℕ) (storage_fraction : ℚ) (parents_fraction : ℚ) (friends : ℕ) 
  (h1 : total = 200)
  (h2 : storage_fraction = 1/4)
  (h3 : parents_fraction = 1/2)
  (h4 : friends = 20) : 
  total - (total * storage_fraction.num / storage_fraction.den).toNat 
        - (total * parents_fraction.num / parents_fraction.den).toNat 
        - friends = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colbert_treehouse_planks_l609_60929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_tetrahedron_theorem_l609_60918

/-- A tetrahedron with specific properties -/
structure SpecialTetrahedron where
  -- Edge lengths
  e : ℝ
  f : ℝ
  -- Constraints
  e_positive : 0 < e
  f_positive : 0 < f
  e_less_f : e < f
  -- AB = BC = CD = e, AC = BD = f
  ab : ℝ
  bc : ℝ
  cd : ℝ
  ac : ℝ
  bd : ℝ
  ab_eq_e : ab = e
  bc_eq_e : bc = e
  cd_eq_e : cd = e
  ac_eq_f : ac = f
  bd_eq_f : bd = f
  -- Angle between planes ADB and ADC is 60°
  angle_adb_adc : ℝ
  angle_eq_60 : angle_adb_adc = 60

/-- Properties of the special tetrahedron -/
def SpecialTetrahedronProperties (t : SpecialTetrahedron) : Prop :=
  ∃ (ad : ℝ),
    -- AD = √(3(f² - e²))
    ad = Real.sqrt (3 * (t.f^2 - t.e^2)) ∧
    -- Dihedral angles at edges of length f are right angles
    (∀ edge : ℝ, edge = t.f → edge = t.ac ∨ edge = t.bd) ∧
    -- Can be dissected into a regular triangular prism
    ∃ (planes : List ℝ), 
      planes.length ≥ 2 ∧ True  -- Placeholder for dissection property

/-- Main theorem: The special tetrahedron has the specified properties -/
theorem special_tetrahedron_theorem (t : SpecialTetrahedron) : 
  SpecialTetrahedronProperties t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_tetrahedron_theorem_l609_60918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_25_l609_60906

theorem divisible_by_25 (n : ℕ) : ∃ k : ℤ, (2^(n + 2) * 3^n + 5 * n - 4 : ℤ) = 25 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_25_l609_60906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_angle_l609_60942

/-- The angle turned by a clock's minute hand when moved back by 10 minutes -/
theorem minute_hand_angle : ∃ (angle : ℝ), angle = Real.pi / 3 := by
  let full_rotation : ℝ := 2 * Real.pi
  let minutes_in_full_rotation : ℝ := 60
  let minutes_moved : ℝ := 10
  let angle : ℝ := minutes_moved / minutes_in_full_rotation * full_rotation
  use angle
  calc
    angle = minutes_moved / minutes_in_full_rotation * full_rotation := rfl
    _ = 10 / 60 * (2 * Real.pi) := by rfl
    _ = (1 / 6) * (2 * Real.pi) := by norm_num
    _ = Real.pi / 3 := by ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_angle_l609_60942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sleep_time_comparison_l609_60994

noncomputable def sleep_time_xiao_yu : List ℝ := [8, 9, 9, 9, 10, 9, 9]
noncomputable def sleep_time_xiao_zhong : List ℝ := [10, 10, 9, 9, 8, 8, 9]

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length : ℝ)

noncomputable def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (fun x => (x - m) ^ 2)).sum / (data.length : ℝ)

theorem sleep_time_comparison :
  (mean sleep_time_xiao_yu = mean sleep_time_xiao_zhong) ∧
  (variance sleep_time_xiao_yu ≠ variance sleep_time_xiao_zhong) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sleep_time_comparison_l609_60994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_sine_graph_l609_60948

open Real

theorem translated_sine_graph (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π/2) : 
  (∃ (x₁ x₂ : ℝ), 
    |sin (2*x₁) - sin (2*x₂ - 2*φ)| = 2 ∧ 
    ∀ (y₁ y₂ : ℝ), |sin (2*y₁) - sin (2*y₂ - 2*φ)| = 2 → |x₁ - x₂| ≤ |y₁ - y₂|) →
  φ = π/6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_sine_graph_l609_60948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_3_l609_60935

open Real

noncomputable def center_x : ℝ := 3 * cos (π / 6)
noncomputable def center_y : ℝ := 3 * sin (π / 6)

def radius : ℝ := 1

noncomputable def line_x (t : ℝ) : ℝ := -1 + (sqrt 3 / 2) * t
noncomputable def line_y (t : ℝ) : ℝ := (1 / 2) * t

theorem chord_length_is_sqrt_3 :
  ∃ (t₁ t₂ : ℝ),
    (line_x t₁ - center_x)^2 + (line_y t₁ - center_y)^2 = radius^2 ∧
    (line_x t₂ - center_x)^2 + (line_y t₂ - center_y)^2 = radius^2 ∧
    (line_x t₁ - line_x t₂)^2 + (line_y t₁ - line_y t₂)^2 = 3 :=
by
  sorry

#check chord_length_is_sqrt_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_3_l609_60935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_centers_of_symmetry_arithmetic_sequence_property_l609_60936

-- Define the function f(x) = x + sin(2x)
noncomputable def f (x : ℝ) : ℝ := x + Real.sin (2 * x)

-- Statement about centers of symmetry
theorem infinitely_many_centers_of_symmetry :
  ∀ n : ℤ, ∃ c : ℝ, ∀ x : ℝ, f (c + x) = f (c - x) :=
by sorry

-- Statement about arithmetic sequence
theorem arithmetic_sequence_property (a₁ a₂ a₃ : ℝ) :
  (∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d) →
  f a₁ + f a₂ + f a₃ = 3 * π →
  a₂ = π :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_centers_of_symmetry_arithmetic_sequence_property_l609_60936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_theorem_l609_60915

/-- The area of a circle given its radius -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- The area of a rectangle given its width and height -/
def rectangle_area (w h : ℝ) : ℝ := w * h

/-- The distance between the centers of two externally tangent circles -/
def dist_between_centers (r₁ r₂ : ℝ) : ℝ := r₁ + r₂

/-- The area outside two overlapping circles within a bounding rectangle -/
noncomputable def area_outside_circles (r₁ r₂ : ℝ) : ℝ :=
  rectangle_area (r₁ + 2*r₂) (2*r₂) - circle_area r₁ - circle_area r₂

/-- The area outside two overlapping circles within a bounding rectangle -/
theorem area_outside_circles_theorem (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 5) 
  (h₃ : r₁ + r₂ = dist_between_centers r₁ r₂) : 
  area_outside_circles r₁ r₂ = 130 - 34 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_theorem_l609_60915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quad_side_sum_l609_60920

/-- A quadrilateral ABCD with specific side lengths and angles -/
structure Quadrilateral where
  BC : ℝ
  CD : ℝ
  AD : ℝ
  angleA : ℝ
  angleB : ℝ
  AB : ℝ

/-- The specific quadrilateral from the problem -/
noncomputable def problemQuad : Quadrilateral where
  BC := 10
  CD := 15
  AD := 8
  angleA := Real.pi / 2  -- 90 degrees in radians
  angleB := Real.pi / 3  -- 60 degrees in radians
  AB := 20  -- This is derived from the conditions, not given directly

/-- Theorem stating the result to be proved -/
theorem quad_side_sum (r s : ℕ) (h : r > 0 ∧ s > 0) :
  problemQuad.AB = r + Real.sqrt (s : ℝ) → r + s = 356 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quad_side_sum_l609_60920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l609_60938

theorem complex_number_in_second_quadrant (θ : ℝ) 
  (h : θ ∈ Set.Ioo (3/4 * Real.pi) (5/4 * Real.pi)) : 
  let z : ℂ := Complex.ofReal (Real.cos θ + Real.sin θ) + Complex.I * Complex.ofReal (Real.sin θ - Real.cos θ)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l609_60938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l609_60914

/-- The area of a triangle with side lengths 9, 12, and 15 is 54 square units. -/
theorem triangle_area : ℝ := by
  -- Define the side lengths
  let a : ℝ := 9
  let b : ℝ := 12
  let c : ℝ := 15

  -- Define the area
  let area : ℝ := 54

  -- State that the area of the triangle with sides a, b, and c is equal to 'area'
  have h : (1/2) * a * b = area := by
    -- Proof steps would go here
    sorry

  -- Return the area
  exact area

#check triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l609_60914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_equation_l609_60943

theorem unique_solution_equation (x y z n : ℕ) : 
  n ≥ 2 ∧ 
  y ≤ 5 * 2^(2*n) ∧ 
  x^(2*n+1) - y^(2*n+1) = x * y * z + 2^(2*n+1) → 
  x = 3 ∧ y = 1 ∧ z = 70 ∧ n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_equation_l609_60943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_false_l609_60910

-- Define the propositions
def inverse_true_implies_negation_true : Prop :=
  ∀ (P Q : Prop), (¬Q → ¬P) → (P ∧ ¬Q)

def negation_of_universal_quantifier : Prop :=
  (∀ x : ℝ, x^2 - x ≤ 0) ↔ (∃ x : ℝ, x^2 - x ≥ 0)

-- Remove the Quadrilateral-related proposition as it's not defined in the standard library
-- Instead, we'll use a placeholder proposition
def placeholder_proposition : Prop :=
  ∀ x : ℝ, x > 0 → x^2 > 0

def sufficient_condition_for_absolute_value : Prop :=
  ∀ x : ℝ, x ≠ 3 → |x| ≠ 3

-- Theorem stating that all propositions are false
theorem all_statements_false :
  ¬inverse_true_implies_negation_true ∧
  ¬negation_of_universal_quantifier ∧
  ¬placeholder_proposition ∧
  ¬sufficient_condition_for_absolute_value :=
by
  constructor
  · intro h
    -- Proof for inverse_true_implies_negation_true
    sorry
  constructor
  · intro h
    -- Proof for negation_of_universal_quantifier
    sorry
  constructor
  · intro h
    -- Proof for placeholder_proposition
    sorry
  · intro h
    -- Proof for sufficient_condition_for_absolute_value
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_false_l609_60910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birdhouse_cost_l609_60975

/-- Cost of building birdhouses -/
theorem birdhouse_cost : 
  (3 : ℚ) * (7 * 3 + 20 * (5 / 100 : ℚ)) +
  (2 : ℚ) * (10 * 5 + 36 * (5 / 100 : ℚ)) = 169.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birdhouse_cost_l609_60975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_equals_PJ_l609_60904

-- Define the basic geometric structures
structure Point where
  x : ℝ
  y : ℝ

-- Define the triangle AKL
def Triangle (A K L : Point) : Prop := True

-- Define a line
def Line (P Q : Point) : Set Point :=
  {R : Point | ∃ t : ℝ, R = ⟨P.x + t * (Q.x - P.x), P.y + t * (Q.y - P.y)⟩}

-- Define parallel lines
def Parallel (l1 l2 : Set Point) : Prop := True

-- Define the intersection of two lines
def Intersect (l1 l2 : Set Point) : Set Point := l1 ∩ l2

-- Define the distance between two points
noncomputable def Distance (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Theorem statement
theorem PA_equals_PJ 
  (A K L B C P J : Point) 
  (triangle : Triangle A K L)
  (bc : Set Point)
  (kl_parallel_bc : Parallel (Line K L) bc)
  (intersect_A : ∃ l : Set Point, l ⊆ Intersect (Line K L) (Line A L) ∧ A ∈ l)
  (intersect_P : P ∈ bc)
  : Distance P A = Distance P J := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_equals_PJ_l609_60904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_over_1_minus_tan_22_5_squared_equals_half_l609_60900

-- Define 22.5 degrees in radians
noncomputable def angle_22_5 : ℝ := Real.pi / 8

-- State the theorem
theorem tan_22_5_over_1_minus_tan_22_5_squared_equals_half :
  Real.tan angle_22_5 / (1 - Real.tan angle_22_5 ^ 2) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_over_1_minus_tan_22_5_squared_equals_half_l609_60900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l609_60988

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  b = 6 →
  (1/2) * a * c * Real.sin B = 15 →
  (1/2) * b / Real.sin B = 5 →
  (Real.sin (2 * B) = 24/25) ∧ (a + b + c = 6 + 6 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l609_60988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l609_60913

theorem lottery_probability : 
  let total_tickets : ℕ := 3
  let prize_tickets : ℕ := 2
  let people_drawing : ℕ := 2
  
  let probability_both_win_prize : ℚ :=
    (prize_tickets : ℚ) / total_tickets * ((prize_tickets - 1) / (total_tickets - 1))
  
  probability_both_win_prize = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l609_60913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_donation_average_l609_60901

theorem school_donation_average (total_donation : ℕ) (num_teachers : ℕ) (num_classes : ℕ) 
  (students_per_class : ℕ) (h1 : total_donation = 1995) (h2 : num_teachers = 35) 
  (h3 : num_classes = 14) (h4 : students_per_class > 30) (h5 : students_per_class ≤ 45) 
  (h6 : total_donation % (num_teachers + num_classes * students_per_class) = 0) :
  total_donation / (num_teachers + num_classes * students_per_class) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_donation_average_l609_60901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_B_is_incorrect_l609_60957

/-- Represents the correlation coefficient in linear regression -/
def correlation_coefficient : ℝ → ℝ := sorry

/-- Represents the strength of correlation between variables -/
def correlation_strength : ℝ → ℝ := sorry

/-- The statement about correlation coefficient and correlation strength -/
def statement_B : Prop :=
  ∀ r₁ r₂ : ℝ, r₁ > r₂ → correlation_strength (correlation_coefficient r₁) > correlation_strength (correlation_coefficient r₂)

theorem statement_B_is_incorrect : ¬statement_B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_B_is_incorrect_l609_60957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l609_60949

theorem ascending_order : Real.cos (3/4 * Real.pi) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l609_60949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_of_H_l609_60917

/-- The mass percentage of an element in a compound -/
noncomputable def mass_percentage (mass_element : ℝ) (total_mass : ℝ) : ℝ :=
  (mass_element / total_mass) * 100

/-- The given mass percentage of H in the compound -/
def given_percentage : ℝ := 4.84

theorem mass_percentage_of_H (mass_H : ℝ) (total_mass : ℝ) 
  (h : mass_percentage mass_H total_mass = given_percentage) : 
  mass_percentage mass_H total_mass = 4.84 := by
  rw [h]
  rfl

#check mass_percentage_of_H

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_of_H_l609_60917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_on_line_l609_60963

/-- The line equation 4x + 3y = 12 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y = 12

/-- The distance from a point (x, y) to the origin -/
noncomputable def distance_to_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- The distance from a point (x, y) to the line 4x + 3y = 12 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |4 * x + 3 * y - 12| / 5

/-- A point is equidistant from the origin and the line -/
def is_equidistant (x y : ℝ) : Prop :=
  distance_to_origin x y = distance_to_line x y

theorem equidistant_points_on_line :
  ∀ x y : ℝ, line_equation x y ∧ is_equidistant x y ↔ (x = 3 ∧ y = 0) ∨ (x = -3 ∧ y = 4) := by
  sorry

#check equidistant_points_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_on_line_l609_60963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outside_area_gt_nine_l609_60990

/-- A rhombus with one diagonal of length 10 and an inscribed circle of radius 3 -/
structure SpecialRhombus where
  diagonal : ℝ
  radius : ℝ
  diagonal_eq : diagonal = 10
  radius_eq : radius = 3

/-- The area of the rhombus outside the inscribed circle -/
noncomputable def outside_area (r : SpecialRhombus) : ℝ :=
  (r.diagonal^2) / 4 - Real.pi * r.radius^2

/-- The area of the rhombus outside the inscribed circle is greater than 9 -/
theorem outside_area_gt_nine (r : SpecialRhombus) : outside_area r > 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_outside_area_gt_nine_l609_60990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l609_60951

theorem cos_alpha_value (α : Real) (h1 : π/2 < α ∧ α < π) (h2 : Real.tan α = -1/2) :
  Real.cos α = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l609_60951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_is_triangular_prism_l609_60961

/-- A solid shape with the following properties:
    1. Front view is an isosceles triangle
    2. Left view is an isosceles triangle
    3. Top view is a circle
-/
structure Shape where
  front_view : IsoscelesTriangle
  left_view : IsoscelesTriangle
  top_view : Circle

/-- Isosceles triangle -/
structure IsoscelesTriangle

/-- Circle -/
structure Circle

/-- Triangular prism shape -/
structure TriangularPrism

theorem shape_is_triangular_prism (s : Shape) : TriangularPrism := by
  sorry

#check shape_is_triangular_prism

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_is_triangular_prism_l609_60961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_properties_l609_60965

open Real Set

noncomputable def f (x : ℝ) : ℝ := x / (x^2 - 4)

theorem odd_decreasing_function_properties :
  ∃ (f : ℝ → ℝ),
    (∀ x, x ∈ Ioo (-2) 2 → f x = x / (x^2 - 4)) ∧
    (∀ x, x ∈ Ioo (-2) 2 → f (-x) = -f x) ∧
    (∀ x y, x ∈ Ioo (-2) 2 → y ∈ Ioo (-2) 2 → x < y → f x > f y) ∧
    (∀ t : ℝ, f (t - 2) + f t > 0 ↔ t ∈ Ioo 0 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_properties_l609_60965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sets_A_l609_60919

def U : Finset Nat := {1, 2, 3}
def B : Finset Nat := {1, 2}

theorem number_of_sets_A : 
  ∃! n : Nat, n = (Finset.filter (λ X : Finset Nat => X ⊆ U ∧ X ∩ B = {1}) (Finset.powerset U)).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sets_A_l609_60919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_dilution_result_l609_60955

/-- Calculates the amount of pure milk remaining after two successive dilutions -/
noncomputable def milk_after_dilutions (initial_capacity : ℝ) (initial_milk : ℝ) (removal_amount : ℝ) : ℝ :=
  let first_dilution := initial_milk - removal_amount
  let milk_removed_second := (first_dilution / initial_capacity) * removal_amount
  first_dilution - milk_removed_second

/-- Theorem stating the final amount of pure milk after two dilutions -/
theorem milk_dilution_result :
  milk_after_dilutions 45 45 9 = 28.8 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_dilution_result_l609_60955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_equals_0_l609_60986

def f : ℕ → ℝ
| 0 => 0
| 1 => 1
| n + 2 => f n

theorem f_4_equals_0 : f 4 = 0 := by
  -- Expand the definition of f
  unfold f
  -- Simplify
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_equals_0_l609_60986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_calculation_l609_60922

/-- The total capacity of a pool in gallons -/
def pool_capacity : ℝ → Prop := λ C => C > 0

/-- The amount of water added to the pool in gallons -/
def water_added : ℝ := 300

/-- The initial fill percentage of the pool -/
def initial_fill_percentage : ℝ := 0.70

/-- The final fill percentage of the pool after adding water -/
def final_fill_percentage : ℝ := 0.85

/-- The percentage increase in water volume after adding water -/
def water_increase_percentage : ℝ := 0.30

theorem pool_capacity_calculation (C : ℝ) :
  pool_capacity C ↔ 
    (water_added = (final_fill_percentage - initial_fill_percentage) * C) ∧
    (water_added = water_increase_percentage * (initial_fill_percentage * C)) ∧
    (C = 2000) :=
by
  sorry

#check pool_capacity_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_calculation_l609_60922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l609_60909

/-- The repeating decimal 0.̅56 is equal to the fraction 56/99. -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 56 / 99) ∧ (∀ n : ℕ, x = (56 * (100^n - 1)) / (99 * 100^n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l609_60909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_bounds_l609_60923

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x + 1) * Real.exp x

def monotone_increasing (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x < f y

def monotone_decreasing (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x > f y

theorem f_monotonicity (a : ℝ) (h : a ≥ 0) :
  (a = 0 ∧ monotone_increasing (f a) Set.univ) ∨
  (a > 0 ∧ 
    (monotone_increasing (f a) { x | x < -1 } ∧
     monotone_decreasing (f a) { x | -1 < x ∧ x < a - 1 } ∧
     monotone_increasing (f a) { x | x > a - 1 })) :=
sorry

theorem f_bounds (a : ℝ) :
  (∀ x, x ∈ Set.Icc 0 1 → f a x ≥ 1) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_bounds_l609_60923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_315_degrees_l609_60970

theorem csc_315_degrees : (1 / Real.sin (315 * π / 180)) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_315_degrees_l609_60970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unkind_manager_proposal_result_l609_60953

/-- Represents the total salary of employees earning up to $500 -/
def salary_up_to_500 : ℝ := sorry

/-- Represents the total salary of employees earning more than $500 -/
def salary_over_500 : ℝ := sorry

/-- Represents the number of employees earning more than $500 -/
def num_employees_over_500 : ℕ := sorry

/-- The total current salary of all employees -/
def total_current_salary : ℝ := 10000

/-- The total salary after the kind manager's proposal -/
def total_salary_kind_proposal : ℝ := 17000

theorem unkind_manager_proposal_result :
  salary_up_to_500 + 500 * (num_employees_over_500 : ℝ) = 7000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unkind_manager_proposal_result_l609_60953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_set_is_line_l609_60978

/-- Defines a set of points in the plane satisfying θ = π/4 in polar coordinates -/
def polar_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ r : ℝ, p.1 = r * Real.cos (Real.pi/4) ∧ p.2 = r * Real.sin (Real.pi/4)}

/-- Theorem stating that the polar_set forms a straight line through the origin -/
theorem polar_set_is_line : 
  ∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ 
    polar_set = {p : ℝ × ℝ | a * p.1 + b * p.2 = 0} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_set_is_line_l609_60978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_existence_l609_60958

theorem unique_triangle_existence (a c : ℝ) (A : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 / 3 →
  A = 2 * π / 3 →
  ∃! (B C : ℝ), 
    0 < B ∧ 0 < C ∧
    A + B + C = π ∧
    Real.sin C = c * Real.sin A / a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_existence_l609_60958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_special_triangle_l609_60952

/-- The diameter of the inscribed circle in a triangle with given side lengths -/
noncomputable def inscribed_circle_diameter (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  2 * area / s

theorem inscribed_circle_diameter_special_triangle :
  inscribed_circle_diameter 13 14 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_special_triangle_l609_60952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l609_60945

-- Define the volume of the sphere
noncomputable def sphere_volume : ℝ := 9 * Real.pi / 2

-- Define the relationship between the cube and the sphere
def cube_inscribed_in_sphere (cube_side : ℝ) (sphere_radius : ℝ) : Prop :=
  cube_side * Real.sqrt 3 = 2 * sphere_radius

-- Theorem statement
theorem cube_surface_area (cube_side : ℝ) (sphere_radius : ℝ) 
  (h1 : (4/3) * Real.pi * sphere_radius^3 = sphere_volume)
  (h2 : cube_inscribed_in_sphere cube_side sphere_radius) : 
  6 * cube_side^2 = 18 := by
  sorry

#check cube_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l609_60945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_accident_location_l609_60977

theorem accident_location : True := by
  trivial

#check accident_location

end NUMINAMATH_CALUDE_ERRORFEEDBACK_accident_location_l609_60977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_above_median_contestants_l609_60939

/-- The number of contestants who performed at or above the median score in at least one of three individual tests -/
def N : ℕ := 516

/-- The total number of contestants -/
def total_contestants : ℕ := 679

/-- The number of individual tests -/
def num_tests : ℕ := 3

/-- A custom approximation relation for natural numbers -/
def approx (a b : ℕ) : Prop := (a : ℤ) - b ≤ 2 ∧ (b : ℤ) - a ≤ 2

notation:50 a " ≈ " b:50 => approx a b

/-- The main theorem stating the approximation of N -/
theorem estimate_above_median_contestants :
  N ≈ total_contestants - (total_contestants / 2 ^ num_tests) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_above_median_contestants_l609_60939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_A_is_16_l609_60903

-- Define the set A of prime numbers between 62 and 85
def A : Finset Nat := Finset.filter (λ n => 62 < n ∧ n < 85 ∧ Nat.Prime n) (Finset.range 86)

-- Define the range of a finite set
def range (S : Finset Nat) : Nat :=
  if h : S.Nonempty then
    S.max' h - S.min' h
  else
    0

-- Theorem statement
theorem range_of_A_is_16 : range A = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_A_is_16_l609_60903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_miles_in_rods_l609_60954

-- Define our units as types
structure Mile : Type
structure Furlong : Type
structure Rod : Type

-- Define conversion functions
def mile_to_furlong : Mile → Furlong
  | _ => ⟨⟩

def furlong_to_rod : Furlong → Rod
  | _ => ⟨⟩

-- Define numeric conversions
def Mile.toNat : Mile → Nat
  | _ => 1

def Furlong.toNat : Furlong → Nat
  | _ => 1

def Rod.toNat : Rod → Nat
  | _ => 1

-- Define conversion rates
axiom mile_to_furlong_rate : ∀ m : Mile, (mile_to_furlong m).toNat = 10 * m.toNat
axiom furlong_to_rod_rate : ∀ f : Furlong, (furlong_to_rod f).toNat = 50 * f.toNat

-- Main theorem
theorem two_miles_in_rods : 
  ∃ m : Mile, m.toNat = 2 ∧ 
  (furlong_to_rod (mile_to_furlong m)).toNat = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_miles_in_rods_l609_60954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_for_24_70_l609_60983

/-- The side length of a rhombus with given diagonals -/
noncomputable def rhombus_side (d1 d2 : ℝ) : ℝ :=
  Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)

/-- Theorem: The side length of a rhombus with diagonals 24 and 70 is 37 -/
theorem rhombus_side_for_24_70 : rhombus_side 24 70 = 37 := by
  -- Unfold the definition of rhombus_side
  unfold rhombus_side
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_for_24_70_l609_60983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_phi_monotone_decreasing_l609_60925

/-- The function f(x) = sin(x + φ) -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (x + φ)

/-- Theorem: There exists a φ in [0, 2π) such that f(x) is monotonically decreasing on [π/3, π] -/
theorem exists_phi_monotone_decreasing :
  ∃ φ : ℝ, 0 ≤ φ ∧ φ < 2 * Real.pi ∧
  (∀ x y : ℝ, π / 3 ≤ x ∧ x ≤ y ∧ y ≤ π → f φ y ≤ f φ x) :=
by sorry

/-- Lemma: π/6 is a valid solution for φ -/
lemma pi_sixth_is_valid_solution :
  let φ := π / 6
  0 ≤ φ ∧ φ < 2 * Real.pi ∧
  (∀ x y : ℝ, π / 3 ≤ x ∧ x ≤ y ∧ y ≤ π → f φ y ≤ f φ x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_phi_monotone_decreasing_l609_60925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_five_l609_60959

def number_set : Finset ℕ := {1, 2, 3, 4}

theorem probability_sum_five (number_set : Finset ℕ) : 
  (number_set = {1, 2, 3, 4}) → 
  (Finset.card (Finset.filter (λ p : ℕ × ℕ => p.1 ∈ number_set ∧ p.2 ∈ number_set ∧ p.1 + p.2 = 5) (number_set.product number_set))) / 
  (Finset.card (number_set.product number_set)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_five_l609_60959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_parabola_hyperbola_l609_60976

/-- The minimum dot product of OP and FP for a parabola and hyperbola with shared focus -/
theorem min_dot_product_parabola_hyperbola :
  ∃ (P : ℝ × ℝ),
    P.1 ≥ 0 ∧
    P.2^2/3 - P.1^2 = 1 ∧
    (∀ Q : ℝ × ℝ, Q.1 ≥ 0 → Q.2^2/3 - Q.1^2 = 1 →
      P.1 * P.1 + P.2 * (P.2 - 2) ≤ Q.1 * Q.1 + Q.2 * (Q.2 - 2)) ∧
    P.1 * P.1 + P.2 * (P.2 - 2) = 3 - 2 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_parabola_hyperbola_l609_60976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l609_60911

/-- A function f is monotonically increasing on [0, +∞) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

/-- The function f(x) = e^x + 2x^2 + mx + 1 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x + 2 * x^2 + m * x + 1

theorem sufficient_not_necessary (m : ℝ) :
  (MonotonicallyIncreasing (f m) → m ≥ -5) ∧
  (∃ m : ℝ, m ≥ -5 ∧ ¬MonotonicallyIncreasing (f m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l609_60911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_and_side_sum_l609_60924

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0

theorem smallest_period_and_side_sum 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_f_A : f A = 3/2)
  (h_a : a = 3)
  (h_area : (1/2) * b * c * Real.sin A = Real.sqrt 3) :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∀ T : ℝ, T > 0 → (∀ x : ℝ, f (x + T) = f x) → T ≥ Real.pi) ∧
  b^2 + c^2 = 21 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_and_side_sum_l609_60924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_lower_bound_l609_60971

theorem polynomial_degree_lower_bound (p : ℕ) (f : Polynomial ℤ) (d : ℕ) :
  Prime p →
  f.degree = some d →
  f.eval 0 = 0 →
  f.eval 1 = 1 →
  (∀ n : ℕ, n > 0 → (f.eval (n : ℤ)) % p = 0 ∨ (f.eval (n : ℤ)) % p = 1) →
  d ≥ p - 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_lower_bound_l609_60971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metro_visiting_plans_l609_60921

/-- The number of valid assignments when n people choose from m places,
    with the condition that at least one person must choose a special place. -/
def number_of_valid_assignments (n m : ℕ) (special : Fin m) : ℕ :=
  m^n - (m-1)^n

theorem metro_visiting_plans (n m : ℕ) (special : Fin m) :
  n ≥ 1 → m ≥ 2 →
  (m^n - (m-1)^n) = number_of_valid_assignments n m special :=
by
  intros hn hm
  rfl  -- reflexivity, as the left side is definitionally equal to the right side

#eval number_of_valid_assignments 4 3 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metro_visiting_plans_l609_60921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_l609_60947

/-- The area of wrapping paper required for a rectangular box -/
theorem wrapping_paper_area 
  (l w h : ℝ) 
  (hl : l > 0) 
  (hw : w > 0) 
  (hh : h > 0) 
  (hlw : l > w) : 
  ∃ (area : ℝ), area = 3 * (l + w) * h ∧ 
  area = (fun L W H => 
    let side_area := L * H + W * H
    let corner_area := (L * H / 2) + (W * H / 2)
    2 * (side_area + corner_area)) l w h :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_l609_60947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_b_less_than_one_l609_60946

/-- Geometric sequence with common ratio 2 -/
def a (n : ℕ) : ℝ := 2^n

/-- Sum of first n terms of the geometric sequence -/
def S (n : ℕ) : ℝ := 2^(n+1) - 2

/-- Sequence b_n -/
noncomputable def b (n : ℕ) : ℝ := n * Real.log (S n + 2) / Real.log 2

/-- Theorem: Sum of reciprocals of b_n is less than 1 -/
theorem sum_reciprocal_b_less_than_one (n : ℕ) :
  (Finset.range n).sum (fun i => 1 / b (i + 1)) < 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_b_less_than_one_l609_60946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_route_time_l609_60991

/-- Represents the time taken for Joey to run the route one way -/
noncomputable def route_time (route_length : ℝ) (avg_speed : ℝ) (return_speed : ℝ) : ℝ :=
  let total_distance := 2 * route_length
  let return_time := route_length / return_speed
  (total_distance / avg_speed) - return_time

theorem joey_route_time :
  route_time 2 3 6.000000000000002 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_route_time_l609_60991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l609_60934

/-- Proves that the repeating decimal 4.054054... is equal to 150/37 -/
theorem repeating_decimal_to_fraction : 4 + (54 : ℚ) / 999 = 150 / 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l609_60934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_implies_a_l609_60933

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then 1 + 1/x
  else if -1 ≤ x ∧ x ≤ 1 then x^2 + 1
  else 2*x + 3

theorem f_value_implies_a (a : ℝ) :
  f a = 3/2 → a = 2 ∨ a = Real.sqrt 2/2 ∨ a = -Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_implies_a_l609_60933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abcd_equals_13_l609_60926

/-- The tangent of 7.5 degrees -/
noncomputable def tan_7_5 : ℝ := Real.tan (7.5 * Real.pi / 180)

/-- The theorem stating the sum of a, b, c, d given the conditions -/
theorem sum_abcd_equals_13 
  (a b c d : ℕ+) 
  (h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d)
  (h_eq : tan_7_5 = Real.sqrt a - Real.sqrt b + Real.sqrt c - d) :
  a + b + c + d = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abcd_equals_13_l609_60926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_example_l609_60989

/-- The height of a right cylinder with given radius and surface area -/
noncomputable def cylinder_height (r : ℝ) (sa : ℝ) : ℝ :=
  (sa - 2 * Real.pi * r^2) / (2 * Real.pi * r)

/-- Theorem: The height of a right cylinder with radius 3 feet and surface area 27π square feet is 3/2 feet -/
theorem cylinder_height_example : cylinder_height 3 (27 * Real.pi) = 3/2 := by
  -- Unfold the definition of cylinder_height
  unfold cylinder_height
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_example_l609_60989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_seven_seven_times_l609_60927

/-- The number of faces on the die -/
def num_faces : ℕ := 8

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The minimum number of "at least seven" rolls required -/
def min_success : ℕ := 7

/-- The probability of rolling at least a seven on a single roll -/
def p_success : ℚ := 1 / 4

/-- The probability of not rolling at least a seven on a single roll -/
def p_failure : ℚ := 3 / 4

/-- The probability of rolling at least a seven at least seven times in eight rolls -/
theorem prob_at_least_seven_seven_times : 
  (Finset.sum (Finset.range 2) (λ k => 
    (Nat.choose num_rolls (num_rolls - k) : ℚ) * 
    p_success ^ (num_rolls - k) * 
    p_failure ^ k)) = 385 / 65536 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_seven_seven_times_l609_60927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_angle_l609_60995

theorem perpendicular_vectors_angle (α : Real) 
  (h1 : 0 < α) (h2 : α < π) 
  (h3 : (1 : Real) * Real.sin α + Real.cos α * 1 = 0) : α = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_angle_l609_60995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_divisor_digit_sum_l609_60968

def n : ℕ := 16382

theorem greatest_prime_divisor_digit_sum (p : ℕ) : 
  Nat.Prime p → p ∣ n → (∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) → 
  (Nat.digits 10 p).sum = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_divisor_digit_sum_l609_60968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_is_zero_l609_60956

-- Define the region R
def R : Set (ℝ × ℝ) :=
  {p | let (x, y) := p
       (|x - y| + y ≤ 12) ∧ 
       (2 * y - x ≥ 12) ∧ 
       (4 ≤ x) ∧ (x ≤ 8)}

-- Define the axis of revolution
def axis (p : ℝ × ℝ) : Prop := 2 * p.2 - p.1 = 12

-- Define the volume of revolution (placeholder)
noncomputable def volume_of_revolution (S : Set (ℝ × ℝ)) (a : (ℝ × ℝ) → Prop) : ℝ := sorry

-- Theorem statement
theorem volume_of_solid_is_zero :
  (∀ p ∈ R, ∃ q, axis q ∧ (p.1 - q.1)^2 + (p.2 - q.2)^2 = 0) →
  (volume_of_revolution R axis = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_is_zero_l609_60956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evans_initial_money_l609_60982

/-- The problem of calculating Evan's initial amount of money --/
theorem evans_initial_money (watch_cost money_from_david money_still_needed evans_initial : ℕ) :
  watch_cost = 20 →
  money_from_david = 12 →
  money_still_needed = 7 →
  watch_cost = money_from_david + money_still_needed + evans_initial →
  evans_initial = 13 :=
by
  sorry

#check evans_initial_money

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evans_initial_money_l609_60982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_solutions_count_l609_60908

theorem natural_number_solutions_count (n : ℕ) : 
  (Finset.card (Finset.filter (λ tuple : ℕ × ℕ × ℕ => tuple.1 + tuple.2.1 + tuple.2.2 = n) (Finset.product (Finset.range (n + 1)) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))))) = 
  Nat.choose (n + 2) 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_solutions_count_l609_60908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l609_60996

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the distance from a point on a hyperbola to its foci -/
theorem hyperbola_foci_distance (h : Hyperbola) (p f1 f2 : Point) :
  isOnHyperbola h p →
  distance p f1 = 4 →
  f1.x < f2.x →  -- Ensuring f1 is the left focus
  distance p f2 = 10 := by
  sorry

#check hyperbola_foci_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l609_60996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l609_60998

noncomputable def f (x m : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else -x^2 + m

theorem range_of_m (m : ℝ) :
  (∀ y, y ∈ Set.range (f · m) ↔ y ∈ Set.Iic 1) →
  m ∈ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l609_60998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l609_60928

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 / Real.sqrt (x - 1)

-- State the theorem
theorem f_domain : Set.Ioi 1 = {x : ℝ | f x ∈ Set.univ} := by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l609_60928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_thirds_in_nine_halves_l609_60985

theorem one_thirds_in_nine_halves : ⌊(9 : ℝ) / 2 / (1 / 3)⌋ = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_thirds_in_nine_halves_l609_60985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_x_coordinate_l609_60905

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop :=
  parabola p.1 p.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem midpoint_x_coordinate
  (M N : ℝ × ℝ)
  (hM : point_on_parabola M)
  (hN : point_on_parabola N)
  (h_distance : distance M focus + distance N focus = 6) :
  (M.1 + N.1) / 2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_x_coordinate_l609_60905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l609_60950

def A : Set ℝ := {x | x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3}

def B : Set ℝ := {x | x ≥ 2}

theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x | x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l609_60950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selection_uses_golden_ratio_l609_60931

/-- The optimal selection method popularized by Hua Luogeng -/
def optimalSelectionMethod : Type := Unit

/-- The optimal selection method uses the golden ratio -/
theorem optimal_selection_uses_golden_ratio :
  ∃ (f : optimalSelectionMethod → ℝ), f = λ _ ↦ ((1 + Real.sqrt 5) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selection_uses_golden_ratio_l609_60931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertices_is_21_l609_60969

/-- Represents a vertex in the triangular grid --/
structure Vertex where
  x : Nat
  y : Nat
  z : Nat
  sum_eq_30 : x + y + z = 30

/-- The maximum number of vertices in the triangular grid --/
def max_vertices : Nat := 21

/-- States that the maximum number of vertices in the triangular grid is 21 --/
theorem max_vertices_is_21 :
  ∀ (vertices : Finset Vertex),
    (∀ v1 v2, v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 → 
      (v1.x ≠ v2.x ∨ v1.y ≠ v2.y ∨ v1.z ≠ v2.z)) →
    vertices.card ≤ max_vertices :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertices_is_21_l609_60969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_roots_sum_l609_60964

theorem tan_roots_sum (α β : Real) (hα : α ∈ Set.Ioo 0 (π/2)) (hβ : β ∈ Set.Ioo 0 (π/2))
  (h_roots : (∃ x, x^2 - 3*Real.sqrt 3*x + 4 = 0 ∧ x = Real.tan α) ∧
             (∃ x, x^2 - 3*Real.sqrt 3*x + 4 = 0 ∧ x = Real.tan β)) :
  α + β = 2*π/3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_roots_sum_l609_60964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_5km_l609_60993

/-- The distance from the warehouse to the station in kilometers -/
def x : Real := Real.mk 0  -- We define x as a Real number

/-- The monthly occupancy fee in ten thousand yuan -/
noncomputable def y₁ (x : ℝ) : ℝ := 20 / x

/-- The monthly freight cost in ten thousand yuan -/
noncomputable def y₂ (x : ℝ) : ℝ := (4 / 5) * x

/-- The total monthly cost in ten thousand yuan -/
noncomputable def total_cost (x : ℝ) : ℝ := y₁ x + y₂ x

/-- Theorem stating that the total cost is minimized when x = 5 -/
theorem min_cost_at_5km :
  ∀ x > 0, total_cost 5 ≤ total_cost x :=
by
  sorry  -- We use sorry to skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_5km_l609_60993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_equivalence_l609_60999

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2^(-x)
noncomputable def g (x : ℝ) : ℝ := 2^(-x+1) + 3

-- State the theorem
theorem translation_equivalence : ∀ x : ℝ, f x = g (x - 1) - 3 := by
  intro x
  simp [f, g]
  ring_nf
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_equivalence_l609_60999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l609_60972

-- Define the piecewise function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x ≥ m then 4 else x^2 + 4*x - 3

-- Define function g
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x - 2*x

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g m x = 0 ∧ g m y = 0 ∧ g m z = 0) →
  (m > 1 ∧ m ≤ 2) := by
  sorry

#check range_of_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l609_60972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_simplest_l609_60974

def is_simplest_sqrt (x : ℝ) (options : List ℝ) : Prop :=
  x ∈ options ∧ ∀ y ∈ options, (∀ a b : ℚ, y ≠ a * Real.sqrt b ∨ b = 1) → y = x

theorem sqrt_5_simplest : 
  is_simplest_sqrt (Real.sqrt 5) [Real.sqrt 5, Real.sqrt 18, Real.sqrt (2/3), Real.sqrt 0.6] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_simplest_l609_60974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_pricing_l609_60916

theorem dealer_pricing (cost_price : ℝ) (original_price : ℝ) 
  (h1 : original_price > 0) 
  (h2 : cost_price > 0) 
  (h3 : original_price * 0.8 = cost_price * 1.2) 
  (h4 : (10 : ℝ) * cost_price = (9 : ℝ) * original_price * 0.8) : 
  original_price = cost_price * 1.5 := by
  sorry

#check dealer_pricing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_pricing_l609_60916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2594_to_hundredth_l609_60902

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ / 100 : ℝ)

/-- The problem statement -/
theorem round_2594_to_hundredth :
  roundToHundredth 2.594 = 2.59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2594_to_hundredth_l609_60902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_describes_parabola_l609_60962

/-- The polar equation r = 1 / (1 - sin θ) describes a parabola. -/
theorem polar_equation_describes_parabola :
  ∃ (a b c : ℝ) (h : a ≠ 0),
    ∀ (x y : ℝ),
      (∃ (r θ : ℝ), r > 0 ∧ r = 1 / (1 - Real.sin θ) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
      a * x^2 + b * x + c * y = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_describes_parabola_l609_60962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_triangle_problem_l609_60997

noncomputable section

/-- Line represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

noncomputable def x_intercept (l : Line) : ℝ := -l.c / l.a

noncomputable def y_intercept (l : Line) : ℝ := -l.c / l.b

noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

theorem line_and_triangle_problem (l1 l2 : Line) (p : Point) :
  l1 = Line.mk 2 (-3) 1 →
  p = Point.mk 1 1 →
  passes_through l2 p →
  perpendicular l1 l2 →
  (l2 = Line.mk 3 2 (-5) ∧
   triangle_area (x_intercept l2) (y_intercept l2) = 25 / 12) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_triangle_problem_l609_60997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_triangle_problem_l609_60940

/-- Represents the shaded area in the semicircle-triangle problem -/
noncomputable def shaded_area (side_length : ℝ) : ℝ :=
  50 / 3 * Real.pi - 25 * Real.sqrt 3

/-- Represents the sum of coefficients in the expression a*π - b*√c -/
def coefficient_sum (a b c : ℕ) : ℕ :=
  a + b + c

/-- Theorem stating the existence of integer coefficients a, b, c that satisfy the problem conditions -/
theorem semicircle_triangle_problem :
  ∃ (a b c : ℕ),
    shaded_area 10 = a * Real.pi - b * Real.sqrt c ∧
    coefficient_sum a b c = 78 := by
  sorry

#eval coefficient_sum 50 25 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_triangle_problem_l609_60940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polynomial_with_properties_l609_60960

/-- Definition of a polynomial with at least one negative coefficient -/
def has_negative_coeff (p : Polynomial ℝ) : Prop :=
  ∃ i, p.coeff i < 0

/-- Definition of a polynomial with all positive coefficients -/
def all_positive_coeffs (p : Polynomial ℝ) : Prop :=
  ∀ i, p.coeff i > 0

/-- Theorem stating the existence of a polynomial with the required properties -/
theorem exists_polynomial_with_properties : 
  ∃ p : Polynomial ℝ, has_negative_coeff p ∧ ∀ n > 1, all_positive_coeffs (p^n) := by
  sorry

#check exists_polynomial_with_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polynomial_with_properties_l609_60960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_enough_for_six_days_l609_60979

/-- Amount of food in a large package -/
noncomputable def B : ℝ := sorry

/-- Amount of food in a small package -/
noncomputable def S : ℝ := sorry

/-- A large package contains more food than a small one -/
axiom large_more_than_small : B > S

/-- A large package contains less food than two small packages -/
axiom large_less_than_two_small : B < 2 * S

/-- Daily consumption of cat food -/
noncomputable def daily_consumption : ℝ := (B + 2 * S) / 2

/-- One large and two small packages are enough for exactly two days -/
axiom two_day_consumption : B + 2 * S = 2 * daily_consumption

/-- Theorem: 4 large and 4 small packages are not enough for six days -/
theorem not_enough_for_six_days : 4 * B + 4 * S < 6 * daily_consumption := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_enough_for_six_days_l609_60979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_circles_l609_60987

/-- The area of the shaded region formed by two identical smaller circles 
    touching each other at the center of a larger circle with radius 8 
    is equal to 32π. -/
theorem shaded_area_circles (r : ℝ) (h : r = 8) : 
  π * r^2 - 2 * (π * (r/2)^2) = 32 * π := by
  -- Substitute r = 8
  rw [h]
  -- Simplify the expression
  simp [Real.pi_pos]
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_circles_l609_60987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base3_addition_theorem_l609_60984

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/-- Converts a decimal number to its base 3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

theorem base3_addition_theorem :
  let a := [2]
  let b := [2, 1, 1]
  let c := [2, 0, 1, 2]
  let d := [1, 0, 1, 1]
  let e := [2, 1, 2, 1, 2]
  let sum := [1, 1, 2, 2, 1, 1]
  base3ToDecimal a + base3ToDecimal b + base3ToDecimal c + base3ToDecimal d + base3ToDecimal e =
  base3ToDecimal sum :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base3_addition_theorem_l609_60984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_after_4040_steps_l609_60907

/-- Represents the state of the game as a list of three integers -/
def GameState := List Nat

/-- The initial state of the game -/
def initialState : GameState := [2, 2, 2]

/-- A single step of the game -/
noncomputable def gameStep (state : GameState) : GameState :=
  sorry

/-- The probability of transitioning from one state to another in a single step -/
noncomputable def transitionProbability (start finish : GameState) : ℝ :=
  sorry

/-- The probability of being in the initial state after n steps -/
noncomputable def probabilityAfterSteps (n : Nat) : ℝ :=
  sorry

theorem probability_after_4040_steps :
  probabilityAfterSteps 4040 = 1 / 4 := by
  sorry

#check probability_after_4040_steps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_after_4040_steps_l609_60907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_storage_methods_l609_60980

/-- Represents a pyramid with 8 edges and 5 vertices -/
structure Pyramid where
  vertices : Fin 5
  edges : Fin 8

/-- Represents a grouping of edges that don't share vertices -/
def ValidGrouping (p : Pyramid) := List (List (Fin 8))

/-- Represents an assignment of edge groups to warehouses -/
def WarehouseAssignment (p : Pyramid) := Fin 4 → List (Fin 8)

/-- Predicate to check if a grouping is valid (no shared vertices) -/
def is_valid_grouping (p : Pyramid) (g : ValidGrouping p) : Prop := sorry

/-- Predicate to check if a warehouse assignment is valid -/
def is_valid_assignment (p : Pyramid) (a : WarehouseAssignment p) : Prop := sorry

/-- The number of valid groupings for a pyramid -/
noncomputable def num_valid_groupings (p : Pyramid) : ℕ := sorry

/-- The number of ways to assign 4 groups to 4 warehouses -/
noncomputable def num_warehouse_assignments : ℕ := sorry

/-- The main theorem to prove -/
theorem pyramid_storage_methods (p : Pyramid) :
  (num_valid_groupings p) * num_warehouse_assignments = 48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_storage_methods_l609_60980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_left_to_read_l609_60912

theorem pages_left_to_read (total_pages : ℕ) (read_fraction : ℚ) (pages_left : ℕ) : 
  total_pages = 396 → 
  read_fraction = 1/3 → 
  pages_left = total_pages - (read_fraction * total_pages).floor → 
  pages_left = 264 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_left_to_read_l609_60912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximation_l609_60966

-- Define the train's characteristics
noncomputable def train_length : ℝ := 250  -- in meters
noncomputable def crossing_time : ℝ := 9   -- in seconds

-- Define the conversion factor from m/s to km/hr
noncomputable def conversion_factor : ℝ := 3.6

-- Define the function to calculate speed in km/hr
noncomputable def calculate_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * conversion_factor

-- Theorem statement
theorem train_speed_approximation :
  ∃ ε > 0, |calculate_speed train_length crossing_time - 100| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximation_l609_60966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2016_l609_60944

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => 1 / (1 - mySequence n)

theorem mySequence_2016 : mySequence 2015 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2016_l609_60944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_x_squared_cos_x_l609_60930

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

-- State the theorem
theorem derivative_x_squared_cos_x :
  deriv f = λ x => 2 * x * Real.cos x - x^2 * Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_x_squared_cos_x_l609_60930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_existence_l609_60941

noncomputable def f (x : ℝ) : ℝ := (1 + x) ^ (1/3) + (1 - x) ^ (1/3)

theorem equation_solution_existence (a : ℝ) :
  (a ≥ 0 ∧ ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f x = a) ↔ (2 ^ (1/3) ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_existence_l609_60941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_angle_parallel_iff_collinear_l609_60981

-- Define a vector type
structure Vec3 where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define collinearity for vectors
def collinear (v w : Vec3) : Prop :=
  ∃ (k : ℝ), v = Vec3.mk (k * w.x) (k * w.y) (k * w.z) ∨ 
              w = Vec3.mk (k * v.x) (k * v.y) (k * v.z)

-- Define parallelism for vectors
def parallel (v w : Vec3) : Prop :=
  ∃ (k : ℝ), v = Vec3.mk (k * w.x) (k * w.y) (k * w.z)

-- Define the angle between two vectors
noncomputable def angle (v w : Vec3) : ℝ := sorry

-- Statement C
theorem collinear_angle (v w : Vec3) (hv : v ≠ Vec3.mk 0 0 0) (hw : w ≠ Vec3.mk 0 0 0) :
  collinear v w → (angle v w = 0 ∨ angle v w = Real.pi) := by
  sorry

-- Statement D
theorem parallel_iff_collinear (v w : Vec3) :
  parallel v w ↔ collinear v w := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_angle_parallel_iff_collinear_l609_60981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_socks_remaining_l609_60937

def total_socks : ℕ := 1200

def white_fraction : ℚ := 1/4
def blue_fraction : ℚ := 3/8
def red_fraction : ℚ := 1/6
def green_fraction : ℚ := 1/12

def white_lost_fraction : ℚ := 1/3
def blue_sold_fraction : ℚ := 1/2

theorem socks_remaining : 
  (300 - 100) + (450 - 225) + 200 + 100 = 725 := by
  -- Proof steps would go here
  sorry

#check socks_remaining

end NUMINAMATH_CALUDE_ERRORFEEDBACK_socks_remaining_l609_60937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l609_60992

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else (a + 1) / x

-- State the theorem
theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  -7/2 ≤ a ∧ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l609_60992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l609_60967

open Real

theorem log_properties (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hne : x₁ ≠ x₂) :
  let f := fun x => log x
  (f (x₁ * x₂) = f x₁ + f x₂) ∧ 
  (f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l609_60967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l609_60932

/-- Rectangle type with side1, side2, area, and diagonal -/
structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  area : ℝ
  diagonal : ℝ

/-- Similarity relation between two rectangles -/
def Similar (R1 R2 : Rectangle) : Prop :=
  R1.side1 / R1.side2 = R2.side1 / R2.side2

/-- Given a rectangle R1 with one side of 2 inches and area of 12 square inches,
    and a similar rectangle R2 with a diagonal of 15 inches,
    the area of R2 is 135/2 square inches. -/
theorem area_of_similar_rectangle (R1 R2 : Rectangle) : 
  R1.side1 = 2 →
  R1.area = 12 →
  R2.diagonal = 15 →
  Similar R1 R2 →
  R2.area = 135 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l609_60932
