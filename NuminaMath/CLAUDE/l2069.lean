import Mathlib

namespace NUMINAMATH_CALUDE_equation_solutions_l2069_206903

theorem equation_solutions :
  (∃ y : ℝ, 6 - 3*y = 15 + 6*y ∧ y = -1) ∧
  (∃ x : ℝ, (1 - 2*x) / 3 = (3*x + 1) / 7 - 2 ∧ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2069_206903


namespace NUMINAMATH_CALUDE_trapezoid_EN_squared_l2069_206907

/-- Trapezoid ABCD with given side lengths and point N -/
structure Trapezoid :=
  (A B C D E M N : ℝ × ℝ)
  (AB_parallel_CD : (A.2 - B.2) / (A.1 - B.1) = (C.2 - D.2) / (C.1 - D.1))
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5)
  (BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 9)
  (CD_length : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 10)
  (DA_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 7)
  (E_on_BC : ∃ t, E = (1 - t) • B + t • C)
  (E_on_DA : ∃ s, E = (1 - s) • D + s • A)
  (M_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
  (N_on_BMC : (N.1 - B.1)^2 + (N.2 - B.2)^2 = (N.1 - M.1)^2 + (N.2 - M.2)^2 ∧
               (N.1 - M.1)^2 + (N.2 - M.2)^2 = (N.1 - C.1)^2 + (N.2 - C.2)^2)
  (N_on_DMA : (N.1 - D.1)^2 + (N.2 - D.2)^2 = (N.1 - M.1)^2 + (N.2 - M.2)^2 ∧
               (N.1 - M.1)^2 + (N.2 - M.2)^2 = (N.1 - A.1)^2 + (N.2 - A.2)^2)
  (N_not_M : N ≠ M)

/-- The main theorem -/
theorem trapezoid_EN_squared (t : Trapezoid) : 
  (t.E.1 - t.N.1)^2 + (t.E.2 - t.N.2)^2 = 900 / 11 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_EN_squared_l2069_206907


namespace NUMINAMATH_CALUDE_inequality_proof_l2069_206950

theorem inequality_proof (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) : 
  a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2069_206950


namespace NUMINAMATH_CALUDE_line_equation_l2069_206997

/-- Given two points A(x₁,y₁) and B(x₂,y₂) satisfying the equations 3x₁ - 4y₁ - 2 = 0 and 3x₂ - 4y₂ - 2 = 0,
    the line passing through these points has the equation 3x - 4y - 2 = 0. -/
theorem line_equation (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 3 * x₁ - 4 * y₁ - 2 = 0) 
  (h₂ : 3 * x₂ - 4 * y₂ - 2 = 0) : 
  ∀ (x y : ℝ), (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) → 3 * x - 4 * y - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2069_206997


namespace NUMINAMATH_CALUDE_subset_sum_equality_l2069_206965

theorem subset_sum_equality (n : ℕ) (A : Finset ℕ) :
  A.card = n →
  (∀ a ∈ A, a > 0) →
  A.sum id < 2^n - 1 →
  ∃ (B C : Finset ℕ), B ⊆ A ∧ C ⊆ A ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ B ∩ C = ∅ ∧ B.sum id = C.sum id := by
  sorry

end NUMINAMATH_CALUDE_subset_sum_equality_l2069_206965


namespace NUMINAMATH_CALUDE_total_bottles_l2069_206990

theorem total_bottles (juice : ℕ) (water : ℕ) : 
  juice = 34 → 
  water = (3 * juice) / 2 + 3 → 
  juice + water = 88 := by
sorry

end NUMINAMATH_CALUDE_total_bottles_l2069_206990


namespace NUMINAMATH_CALUDE_dragon_lion_equivalence_l2069_206922

-- Define the propositions
variable (P Q : Prop)

-- State the theorem
theorem dragon_lion_equivalence :
  (P → Q) ↔ (¬Q → ¬P) ∧ (¬P ∨ Q) :=
sorry

end NUMINAMATH_CALUDE_dragon_lion_equivalence_l2069_206922


namespace NUMINAMATH_CALUDE_triangle_kite_property_l2069_206973

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))
-- Define points D, H, M, N
variable (D H M N : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_acute_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_angle_bisector (A D B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_altitude (A H B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def on_circle (M B D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_kite (A M H N : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem triangle_kite_property 
  (h_acute : is_acute_triangle A B C)
  (h_bisector : is_angle_bisector A D B C)
  (h_altitude : is_altitude A H B C)
  (h_circle_M : on_circle M B D)
  (h_circle_N : on_circle N C D) :
  is_kite A M H N :=
sorry

end NUMINAMATH_CALUDE_triangle_kite_property_l2069_206973


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l2069_206983

/-- Given x is inversely proportional to y, prove that y₁/y₂ = 4/3 when x₁/x₂ = 3/4 -/
theorem inverse_proportion_ratio (x y : ℝ → ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h_inverse : ∀ t : ℝ, t ≠ 0 → x t * y t = x₁ * y₁)
  (h_x₁_nonzero : x₁ ≠ 0)
  (h_x₂_nonzero : x₂ ≠ 0)
  (h_y₁_nonzero : y₁ ≠ 0)
  (h_y₂_nonzero : y₂ ≠ 0)
  (h_x_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_ratio_l2069_206983


namespace NUMINAMATH_CALUDE_camel_height_is_28_feet_l2069_206945

/-- The height of a hare in inches -/
def hare_height : ℕ := 14

/-- The factor by which a camel is taller than a hare -/
def camel_height_factor : ℕ := 24

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Calculates the height of a camel in feet -/
def camel_height_in_feet : ℕ :=
  (hare_height * camel_height_factor) / inches_per_foot

/-- Theorem stating that the camel's height is 28 feet -/
theorem camel_height_is_28_feet : camel_height_in_feet = 28 := by
  sorry

end NUMINAMATH_CALUDE_camel_height_is_28_feet_l2069_206945


namespace NUMINAMATH_CALUDE_bus_encounters_l2069_206933

-- Define the schedule and travel time
def austin_departure_interval : ℕ := 2
def sanantonio_departure_interval : ℕ := 2
def sanantonio_departure_offset : ℕ := 1
def travel_time : ℕ := 7

-- Define the number of encounters
def encounters : ℕ := 4

-- Theorem statement
theorem bus_encounters :
  (austin_departure_interval = 2) →
  (sanantonio_departure_interval = 2) →
  (sanantonio_departure_offset = 1) →
  (travel_time = 7) →
  (encounters = 4) := by
  sorry

end NUMINAMATH_CALUDE_bus_encounters_l2069_206933


namespace NUMINAMATH_CALUDE_circle_E_and_tangents_l2069_206966

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x - 5)^2 + (y - 1)^2 = 25

-- Define points A, B, C, and P
def point_A : ℝ × ℝ := (0, 1)
def point_B : ℝ × ℝ := (1, 4)
def point_C : ℝ × ℝ := (10, 1)
def point_P : ℝ × ℝ := (10, 11)

-- Define lines l1 and l2
def line_l1 (x y : ℝ) : Prop := x - 5*y - 5 = 0
def line_l2 (x y : ℝ) : Prop := x - 2*y - 8 = 0

-- Define tangent lines
def tangent_line1 (x : ℝ) : Prop := x = 10
def tangent_line2 (x y : ℝ) : Prop := 3*x - 4*y + 14 = 0

theorem circle_E_and_tangents :
  (∀ x y : ℝ, line_l1 x y ∧ line_l2 x y → (x, y) = point_C) →
  circle_E point_A.1 point_A.2 →
  circle_E point_B.1 point_B.2 →
  circle_E point_C.1 point_C.2 →
  (∀ x y : ℝ, circle_E x y ∧ (tangent_line1 x ∨ tangent_line2 x y) →
    ((x - point_P.1)^2 + (y - point_P.2)^2) * 25 = ((x - 5)^2 + (y - 1)^2) * ((point_P.1 - 5)^2 + (point_P.2 - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_E_and_tangents_l2069_206966


namespace NUMINAMATH_CALUDE_girls_trying_out_l2069_206977

theorem girls_trying_out (girls : ℕ) (boys : ℕ) (called_back : ℕ) (didnt_make_cut : ℕ) :
  boys = 32 →
  called_back = 10 →
  didnt_make_cut = 39 →
  girls + boys = called_back + didnt_make_cut →
  girls = 17 := by
  sorry

end NUMINAMATH_CALUDE_girls_trying_out_l2069_206977


namespace NUMINAMATH_CALUDE_distance_between_5th_and_29th_red_light_l2069_206976

/-- Represents the color of a light in the sequence -/
inductive Color
  | Red
  | Blue
  | Green

/-- Defines the repeating pattern of lights -/
def pattern : List Color := [Color.Red, Color.Red, Color.Red, Color.Blue, Color.Blue, Color.Green, Color.Green]

/-- The distance between each light in inches -/
def light_distance : ℕ := 8

/-- Calculates the position of the nth red light in the sequence -/
def red_light_position (n : ℕ) : ℕ :=
  (n - 1) / 3 * 7 + (n - 1) % 3 + 1

/-- Calculates the distance between two positions in the sequence -/
def distance_between (pos1 pos2 : ℕ) : ℕ :=
  (pos2 - pos1) * light_distance

/-- Converts a distance in inches to feet -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem distance_between_5th_and_29th_red_light :
  inches_to_feet (distance_between (red_light_position 5) (red_light_position 29)) = 37 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_5th_and_29th_red_light_l2069_206976


namespace NUMINAMATH_CALUDE_line_segment_coordinates_l2069_206938

theorem line_segment_coordinates (y : ℝ) : 
  y > 0 → 
  ((2 - 6)^2 + (y - 10)^2 = 10^2) →
  (y = 10 - 2 * Real.sqrt 21 ∨ y = 10 + 2 * Real.sqrt 21) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_coordinates_l2069_206938


namespace NUMINAMATH_CALUDE_conference_children_count_l2069_206991

theorem conference_children_count :
  let total_men : ℕ := 700
  let total_women : ℕ := 500
  let indian_men_percentage : ℚ := 20 / 100
  let indian_women_percentage : ℚ := 40 / 100
  let indian_children_percentage : ℚ := 10 / 100
  let non_indian_percentage : ℚ := 79 / 100
  ∃ (total_children : ℕ),
    (indian_men_percentage * total_men +
     indian_women_percentage * total_women +
     indian_children_percentage * total_children : ℚ) =
    ((1 - non_indian_percentage) * (total_men + total_women + total_children) : ℚ) ∧
    total_children = 800 :=
by sorry

end NUMINAMATH_CALUDE_conference_children_count_l2069_206991


namespace NUMINAMATH_CALUDE_last_digit_of_expression_l2069_206988

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ := n % 10

-- Define the main theorem
theorem last_digit_of_expression : lastDigit (33 * 3 - (1984^1984 - 1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_expression_l2069_206988


namespace NUMINAMATH_CALUDE_daniel_driving_speed_l2069_206934

/-- The speed at which Daniel drove on Monday for the first 32 miles -/
def monday_first_speed (x : ℝ) : ℝ := 2 * x

theorem daniel_driving_speed (x : ℝ) (h_x_pos : x > 0) :
  let total_distance : ℝ := 100
  let monday_first_distance : ℝ := 32
  let monday_second_distance : ℝ := total_distance - monday_first_distance
  let sunday_time : ℝ := total_distance / x
  let monday_time : ℝ := monday_first_distance / (monday_first_speed x) + monday_second_distance / (x / 2)
  let time_increase_ratio : ℝ := 1.52
  monday_time = time_increase_ratio * sunday_time :=
by sorry

#check daniel_driving_speed

end NUMINAMATH_CALUDE_daniel_driving_speed_l2069_206934


namespace NUMINAMATH_CALUDE_trajectory_equation_l2069_206928

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -2)

-- Define the vector equation
def vector_equation (C : ℝ × ℝ) (s t : ℝ) : Prop :=
  C = (s * A.1 + t * B.1, s * A.2 + t * B.2)

-- Define the constraint
def constraint (s t : ℝ) : Prop := s + t = 1

-- Theorem statement
theorem trajectory_equation :
  ∀ (C : ℝ × ℝ) (s t : ℝ),
  vector_equation C s t → constraint s t →
  C.1 - C.2 - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2069_206928


namespace NUMINAMATH_CALUDE_emily_subtraction_l2069_206930

theorem emily_subtraction : 50^2 - 49^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_emily_subtraction_l2069_206930


namespace NUMINAMATH_CALUDE_smallest_common_term_l2069_206995

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

def is_in_sequence (x : ℤ) (a₁ : ℤ) (d : ℤ) : Prop :=
  ∃ n : ℕ, arithmetic_sequence a₁ d n = x

theorem smallest_common_term (a₁ b₁ d₁ d₂ : ℤ) (h₁ : a₁ = 1) (h₂ : b₁ = 2) (h₃ : d₁ = 3) (h₄ : d₂ = 10) :
  (∀ x : ℤ, x > 2023 ∧ x < 2032 → ¬(is_in_sequence x a₁ d₁ ∧ is_in_sequence x b₁ d₂)) ∧
  (is_in_sequence 2032 a₁ d₁ ∧ is_in_sequence 2032 b₁ d₂) := by sorry

end NUMINAMATH_CALUDE_smallest_common_term_l2069_206995


namespace NUMINAMATH_CALUDE_inequality_solution_l2069_206967

theorem inequality_solution (a : ℝ) :
  4 ≤ a / (3 * a - 6) ∧ a / (3 * a - 6) > 12 → a < 72 / 35 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2069_206967


namespace NUMINAMATH_CALUDE_sin_alpha_minus_cos_alpha_l2069_206947

theorem sin_alpha_minus_cos_alpha (α : Real) (h : Real.tan α = -3/4) :
  Real.sin α * (Real.sin α - Real.cos α) = 21/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_cos_alpha_l2069_206947


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2069_206931

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - i) / (3 + 4 * i) = 2 / 5 - 11 / 25 * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2069_206931


namespace NUMINAMATH_CALUDE_z_squared_minus_norm_squared_l2069_206974

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- Theorem statement
theorem z_squared_minus_norm_squared :
  z^2 - Complex.abs z^2 = 2 * Complex.I - 2 := by
  sorry

end NUMINAMATH_CALUDE_z_squared_minus_norm_squared_l2069_206974


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l2069_206960

theorem arithmetic_equalities :
  (-16 - (-12) - 24 + 18 = -10) ∧
  (0.125 + 1/4 + (-2 - 1/8) + (-0.25) = -2) ∧
  ((-1/12 - 1/36 + 1/6) * (-36) = -2) ∧
  ((-2 + 3) * 3 - (-2)^3 / 4 = 5) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l2069_206960


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l2069_206992

open Real

-- Define the type of continuous functions from ℝ⁺ to ℝ⁺
def ContinuousPosFun := {f : ℝ → ℝ // Continuous f ∧ ∀ x, x > 0 → f x > 0}

-- Define the property that the function satisfies the given equation
def SatisfiesEquation (f : ContinuousPosFun) : Prop :=
  ∀ x, x > 0 → x + 1/x = f.val x + 1/(f.val x)

-- Define the set of possible solutions
def PossibleSolutions (x : ℝ) : Set ℝ :=
  {x, 1/x, max x (1/x), min x (1/x)}

-- State the theorem
theorem functional_equation_solutions (f : ContinuousPosFun) 
  (h : SatisfiesEquation f) :
  ∀ x, x > 0 → f.val x ∈ PossibleSolutions x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l2069_206992


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l2069_206900

-- Part 1
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*a*(x-1) + 4) →
  a ∈ Set.Iic (1/3) :=
sorry

-- Part 2
def f' (x : ℝ) : ℝ := x^2 - 3*x + 2
def g (m : ℝ) (x : ℝ) : ℝ := -x + m

theorem range_of_m :
  ∀ m : ℝ, (∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Ioo 1 8, f' x₁ = g m x₂) →
  m ∈ Set.Ioo 7 (31/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l2069_206900


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2069_206986

theorem inscribed_cube_volume (large_cube_edge : ℝ) (small_cube_edge : ℝ) 
  (h1 : large_cube_edge = 12)
  (h2 : small_cube_edge * Real.sqrt 3 = large_cube_edge) : 
  small_cube_edge ^ 3 = 192 * Real.sqrt 3 := by
  sorry

#check inscribed_cube_volume

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2069_206986


namespace NUMINAMATH_CALUDE_simplify_expressions_l2069_206970

variable (a b x y : ℝ)

theorem simplify_expressions :
  (2 * a - (a + b) = a - b) ∧
  ((x^2 - 2*y^2) - 2*(3*y^2 - 2*x^2) = 5*x^2 - 8*y^2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2069_206970


namespace NUMINAMATH_CALUDE_car_speed_proof_l2069_206915

theorem car_speed_proof (v : ℝ) : v > 0 →
  (3600 / v - 3600 / 225 = 2) ↔ v = 200 :=
by
  sorry

#check car_speed_proof

end NUMINAMATH_CALUDE_car_speed_proof_l2069_206915


namespace NUMINAMATH_CALUDE_sufficient_necessary_equivalence_l2069_206912

theorem sufficient_necessary_equivalence (A B : Prop) :
  (A → B) ↔ (¬B → ¬A) :=
sorry

end NUMINAMATH_CALUDE_sufficient_necessary_equivalence_l2069_206912


namespace NUMINAMATH_CALUDE_flagpole_height_l2069_206958

/-- Given a right triangle with hypotenuse 5 meters, where a person of height 1.6 meters
    touches the hypotenuse at a point 4 meters from one end of the base,
    prove that the height of the triangle (perpendicular to the base) is 8 meters. -/
theorem flagpole_height (base : ℝ) (hypotenuse : ℝ) (person_height : ℝ) (person_distance : ℝ) :
  base = 5 →
  person_height = 1.6 →
  person_distance = 4 →
  ∃ (height : ℝ), height = 8 ∧ height * base = person_height * person_distance :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_l2069_206958


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2069_206957

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2069_206957


namespace NUMINAMATH_CALUDE_simplify_radicals_l2069_206978

theorem simplify_radicals : 
  (Real.sqrt 440 / Real.sqrt 55) - (Real.sqrt 210 / Real.sqrt 70) = 2 * Real.sqrt 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l2069_206978


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l2069_206940

theorem tic_tac_toe_tie_probability (amy_win_prob lily_win_prob : ℚ) :
  amy_win_prob = 4/9 ∧ lily_win_prob = 1/3 →
  1 - (amy_win_prob + lily_win_prob) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l2069_206940


namespace NUMINAMATH_CALUDE_calculation_proof_l2069_206926

def mixed_to_improper (whole : Int) (num : Int) (denom : Int) : Rat :=
  (whole * denom + num) / denom

theorem calculation_proof :
  let a := mixed_to_improper 2 3 7
  let b := mixed_to_improper 5 1 3
  let c := mixed_to_improper 3 1 5
  let d := mixed_to_improper 2 1 6
  75 * (a - b) / (c + d) = -208 - 7/9 := by sorry

end NUMINAMATH_CALUDE_calculation_proof_l2069_206926


namespace NUMINAMATH_CALUDE_angle_B_is_70_l2069_206962

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)

-- Define the properties of the triangle
def rightTriangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = 180 ∧ t.A = 20 ∧ t.C = 90

-- Theorem statement
theorem angle_B_is_70 (t : Triangle) (h : rightTriangle t) : t.B = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_70_l2069_206962


namespace NUMINAMATH_CALUDE_unreachable_y_value_l2069_206959

theorem unreachable_y_value (x : ℝ) (h : x ≠ -5/4) :
  ¬∃y : ℝ, y = -3/4 ∧ y = (2 - 3*x) / (4*x + 5) :=
by sorry

end NUMINAMATH_CALUDE_unreachable_y_value_l2069_206959


namespace NUMINAMATH_CALUDE_six_grade_assignments_l2069_206904

/-- Number of ways to assign n grades, where each grade is 2, 3, or 4, and no two consecutive grades can both be 2 -/
def gradeAssignments : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => 2 * gradeAssignments (n + 1) + 2 * gradeAssignments n

/-- The number of ways to assign 6 grades under the given conditions is 448 -/
theorem six_grade_assignments : gradeAssignments 6 = 448 := by
  sorry

end NUMINAMATH_CALUDE_six_grade_assignments_l2069_206904


namespace NUMINAMATH_CALUDE_chord_length_circle_line_l2069_206944

/-- The length of the chord intersected by a line on a circle -/
theorem chord_length_circle_line (x y : ℝ) : 
  let circle := fun (x y : ℝ) => (x - 2)^2 + y^2 = 4
  let line := fun (x y : ℝ) => 4*x - 3*y - 3 = 0
  let center := (2, 0)
  let radius := 2
  let d := |4*2 - 3*0 - 3| / Real.sqrt (4^2 + (-3)^2)
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle x₁ y₁ ∧ circle x₂ y₂ ∧ 
    line x₁ y₁ ∧ line x₂ y₂ ∧
    ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 * (radius^2 - d^2) :=
by
  sorry

#check chord_length_circle_line

end NUMINAMATH_CALUDE_chord_length_circle_line_l2069_206944


namespace NUMINAMATH_CALUDE_sample_size_is_80_l2069_206985

/-- Represents the ratio of quantities for products A, B, and C -/
structure ProductRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a stratified sample -/
structure StratifiedSample where
  ratio : ProductRatio
  units_of_a : ℕ

/-- Theorem stating that given the specific conditions, the sample size is 80 -/
theorem sample_size_is_80 (sample : StratifiedSample) 
  (h_ratio : sample.ratio = ProductRatio.mk 2 3 5)
  (h_units_a : sample.units_of_a = 16) : 
  (sample.units_of_a / sample.ratio.a) * (sample.ratio.a + sample.ratio.b + sample.ratio.c) = 80 := by
  sorry

#check sample_size_is_80

end NUMINAMATH_CALUDE_sample_size_is_80_l2069_206985


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2069_206964

/-- The line mx + y - m - 1 = 0 passes through the point (1, 1) for all real m -/
theorem line_passes_through_point (m : ℝ) : m + 1 - m - 1 = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2069_206964


namespace NUMINAMATH_CALUDE_product_xy_equals_four_l2069_206929

-- Define variables
variable (a b x y : ℕ)

-- State the theorem
theorem product_xy_equals_four
  (h1 : x = a)
  (h2 : y = b)
  (h3 : a + a = b * a)
  (h4 : y = a)
  (h5 : a * a = a + a)
  (h6 : b = 3) :
  x * y = 4 := by
sorry

end NUMINAMATH_CALUDE_product_xy_equals_four_l2069_206929


namespace NUMINAMATH_CALUDE_smallest_k_with_remainders_l2069_206905

theorem smallest_k_with_remainders : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 → m % 19 = 1 → m % 7 = 1 → m % 3 = 1 → k ≤ m :=
by
  use 400
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainders_l2069_206905


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2069_206936

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / Real.cos A = b / (2 * Real.cos B) ∧
  a / Real.cos A = c / (3 * Real.cos C) →
  A = π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2069_206936


namespace NUMINAMATH_CALUDE_m_range_proof_l2069_206954

theorem m_range_proof (h : ∀ x, (|x - m| < 1) ↔ (1/3 < x ∧ x < 1/2)) :
  -1/2 ≤ m ∧ m ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_proof_l2069_206954


namespace NUMINAMATH_CALUDE_total_points_is_1320_l2069_206942

def freshman_points : ℕ := 260

def sophomore_points : ℕ := freshman_points + (freshman_points * 15 / 100)

def junior_points : ℕ := sophomore_points + (sophomore_points * 20 / 100)

def senior_points : ℕ := junior_points + (junior_points * 12 / 100)

def total_points : ℕ := freshman_points + sophomore_points + junior_points + senior_points

theorem total_points_is_1320 : total_points = 1320 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_1320_l2069_206942


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2069_206906

theorem fraction_evaluation : (15 - 3^2) / 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2069_206906


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l2069_206952

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.6 * P) 
  (hP : P ≠ 0) : M / N = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l2069_206952


namespace NUMINAMATH_CALUDE_largest_angle_and_sinC_l2069_206902

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle with a = 7, b = 3, c = 5 -/
def givenTriangle : Triangle where
  a := 7
  b := 3
  c := 5
  A := sorry
  B := sorry
  C := sorry

theorem largest_angle_and_sinC (t : Triangle) (h : t = givenTriangle) :
  (t.A > t.B ∧ t.A > t.C) ∧ t.A = Real.pi * (2/3) ∧ Real.sin t.C = 5 * Real.sqrt 3 / 14 := by
  sorry

#check largest_angle_and_sinC

end NUMINAMATH_CALUDE_largest_angle_and_sinC_l2069_206902


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2069_206996

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_third_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = 4)
  (h_tenth : a 10 = 22) :
  a 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2069_206996


namespace NUMINAMATH_CALUDE_points_collinear_l2069_206972

/-- Three points in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of collinearity for three points -/
def collinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem points_collinear : 
  let p1 : Point2D := ⟨1, 2⟩
  let p2 : Point2D := ⟨3, 8⟩
  let p3 : Point2D := ⟨4, 11⟩
  collinear p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_points_collinear_l2069_206972


namespace NUMINAMATH_CALUDE_hyperbola_dimensions_l2069_206955

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the length of the real axis is 2 units greater than the length of the imaginary axis
    and the focal length is 10, then a = 4 and b = 3. -/
theorem hyperbola_dimensions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2*a - 2*b = 2 → a^2 + b^2 = 25 → a = 4 ∧ b = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_dimensions_l2069_206955


namespace NUMINAMATH_CALUDE_single_digit_square_equals_5929_l2069_206979

theorem single_digit_square_equals_5929 (A : ℕ) : 
  A < 10 → (10 * A + A) * (10 * A + A) = 5929 → A = 7 := by
sorry

end NUMINAMATH_CALUDE_single_digit_square_equals_5929_l2069_206979


namespace NUMINAMATH_CALUDE_two_inscribed_cube_lengths_l2069_206910

/-- A regular tetrahedron with unit edge length -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_unit : edge_length = 1

/-- A cube inscribed in a tetrahedron such that each vertex lies on a face of the tetrahedron -/
structure InscribedCube where
  edge_length : ℝ
  vertices_on_faces : True  -- This is a placeholder for the geometric condition

/-- The set of all possible edge lengths for inscribed cubes in a unit regular tetrahedron -/
def inscribed_cube_edge_lengths (t : RegularTetrahedron) : Set ℝ :=
  {l | ∃ c : InscribedCube, c.edge_length = l}

/-- Theorem stating that there are exactly two distinct edge lengths for inscribed cubes -/
theorem two_inscribed_cube_lengths (t : RegularTetrahedron) :
  ∃ l₁ l₂ : ℝ, l₁ ≠ l₂ ∧ inscribed_cube_edge_lengths t = {l₁, l₂} :=
sorry

end NUMINAMATH_CALUDE_two_inscribed_cube_lengths_l2069_206910


namespace NUMINAMATH_CALUDE_range_of_m_l2069_206989

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2*y - x/Real.exp 1)*(Real.log x - Real.log y) - y/m ≤ 0) →
  0 < m ∧ m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2069_206989


namespace NUMINAMATH_CALUDE_rachel_winter_clothing_l2069_206917

theorem rachel_winter_clothing (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) : 
  num_boxes = 7 → scarves_per_box = 3 → mittens_per_box = 4 → 
  num_boxes * scarves_per_box + num_boxes * mittens_per_box = 49 := by
  sorry

end NUMINAMATH_CALUDE_rachel_winter_clothing_l2069_206917


namespace NUMINAMATH_CALUDE_sine_even_function_phi_l2069_206920

theorem sine_even_function_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x + π / 6)) →
  (0 < φ) →
  (φ < π / 2) →
  (∀ x, f (x - φ) = f (φ - x)) →
  φ = π / 3 := by sorry

end NUMINAMATH_CALUDE_sine_even_function_phi_l2069_206920


namespace NUMINAMATH_CALUDE_income_data_mean_difference_l2069_206911

/-- Represents the income data for a group of families -/
structure IncomeData where
  num_families : ℕ
  min_income : ℕ
  max_income : ℕ
  incorrect_max_income : ℕ

/-- Calculates the difference between the mean of incorrect data and actual data -/
def mean_difference (data : IncomeData) : ℚ :=
  (data.incorrect_max_income - data.max_income) / data.num_families

/-- Theorem stating the difference in means for the given scenario -/
theorem income_data_mean_difference :
  ∀ (data : IncomeData),
  data.num_families = 500 →
  data.min_income = 12000 →
  data.max_income = 150000 →
  data.incorrect_max_income = 1500000 →
  mean_difference data = 2700 := by
  sorry

end NUMINAMATH_CALUDE_income_data_mean_difference_l2069_206911


namespace NUMINAMATH_CALUDE_trapezoid_median_l2069_206909

/-- Given a triangle with base 24 inches and area 192 square inches, and a trapezoid with the same 
    height and area as the triangle, the median of the trapezoid is 12 inches. -/
theorem trapezoid_median (triangle_base : ℝ) (triangle_area : ℝ) (trapezoid_height : ℝ) 
  (trapezoid_median : ℝ) : 
  triangle_base = 24 → 
  triangle_area = 192 → 
  triangle_area = (1/2) * triangle_base * trapezoid_height → 
  triangle_area = trapezoid_median * trapezoid_height → 
  trapezoid_median = 12 := by
  sorry

#check trapezoid_median

end NUMINAMATH_CALUDE_trapezoid_median_l2069_206909


namespace NUMINAMATH_CALUDE_count_zeros_up_to_2500_l2069_206998

/-- A function that returns true if a natural number contains the digit 0 in its decimal representation -/
def containsZero (n : ℕ) : Bool :=
  sorry

/-- The count of numbers less than or equal to 2500 that contain the digit 0 -/
def countZeros : ℕ := (List.range 2501).filter containsZero |>.length

/-- Theorem stating that the count of numbers less than or equal to 2500 containing 0 is 591 -/
theorem count_zeros_up_to_2500 : countZeros = 591 := by
  sorry

end NUMINAMATH_CALUDE_count_zeros_up_to_2500_l2069_206998


namespace NUMINAMATH_CALUDE_bus_stop_time_l2069_206937

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 60 → speed_with_stops = 50 → 
  (60 - (60 * speed_with_stops / speed_without_stops)) = 10 := by
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l2069_206937


namespace NUMINAMATH_CALUDE_susan_playground_area_l2069_206951

/-- Represents a rectangular playground with fence posts -/
structure Playground where
  total_posts : ℕ
  post_spacing : ℕ
  longer_side_post_ratio : ℕ

/-- Calculates the area of the playground in square yards -/
def playground_area (p : Playground) : ℕ :=
  let shorter_side := p.post_spacing * (((p.total_posts / 2) / (p.longer_side_post_ratio + 1)) - 1)
  let longer_side := p.post_spacing * (p.longer_side_post_ratio * ((p.total_posts / 2) / (p.longer_side_post_ratio + 1)) - 1)
  shorter_side * longer_side

/-- Theorem stating the area of Susan's playground -/
theorem susan_playground_area :
  ∃ (p : Playground), p.total_posts = 30 ∧ p.post_spacing = 6 ∧ p.longer_side_post_ratio = 3 ∧
  playground_area p = 1188 :=
by
  sorry


end NUMINAMATH_CALUDE_susan_playground_area_l2069_206951


namespace NUMINAMATH_CALUDE_minyoung_fruit_sale_l2069_206927

theorem minyoung_fruit_sale :
  ∀ (tangerines apples : ℕ),
    tangerines = 2 →
    apples = 7 →
    tangerines + apples = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_minyoung_fruit_sale_l2069_206927


namespace NUMINAMATH_CALUDE_complex_magnitude_l2069_206994

theorem complex_magnitude (w : ℂ) (h : w^2 = -48 + 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2069_206994


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2069_206961

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y = 48
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y = -12
def centerLine (x y : ℝ) : Prop := x - y = 0

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  isTangentToLine1 : line1 center.1 center.2
  isTangentToLine2 : line2 center.1 center.2
  centerOnLine : centerLine center.1 center.2

-- Theorem statement
theorem circle_center_coordinates :
  ∀ (c : Circle), c.center = (18/7, 18/7) :=
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2069_206961


namespace NUMINAMATH_CALUDE_log_inequality_l2069_206923

theorem log_inequality (a b : ℝ) 
  (ha : a = Real.log 0.4 / Real.log 0.2) 
  (hb : b = 1 - 1 / Real.log 4) : 
  a * b < a + b ∧ a + b < 0 := by sorry

end NUMINAMATH_CALUDE_log_inequality_l2069_206923


namespace NUMINAMATH_CALUDE_n_accurate_to_hundred_thousandth_l2069_206935

/-- The number we're considering -/
def n : ℝ := 5.374e8

/-- Definition of accuracy to the hundred thousandth place -/
def accurate_to_hundred_thousandth (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k : ℝ) * 1e5

/-- Theorem stating that our number is accurate to the hundred thousandth place -/
theorem n_accurate_to_hundred_thousandth : accurate_to_hundred_thousandth n := by
  sorry

end NUMINAMATH_CALUDE_n_accurate_to_hundred_thousandth_l2069_206935


namespace NUMINAMATH_CALUDE_fourth_square_area_l2069_206924

-- Define the triangles and their properties
structure Triangle :=
  (P Q R : ℝ × ℝ)
  (isRightAngle : Bool)

-- Define the squares on the sides
structure Square :=
  (side : ℝ)
  (area : ℝ)

-- Theorem statement
theorem fourth_square_area
  (PQR PRM : Triangle)
  (square1 square2 square3 : Square)
  (h1 : PQR.isRightAngle = true)
  (h2 : PRM.isRightAngle = true)
  (h3 : square1.area = 25)
  (h4 : square2.area = 81)
  (h5 : square3.area = 64)
  : ∃ (square4 : Square), square4.area = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_square_area_l2069_206924


namespace NUMINAMATH_CALUDE_problem_solution_l2069_206908

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + a

-- State the theorem
theorem problem_solution :
  -- Part 1: Find the value of a
  (∃ (a : ℝ), ∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  -- Part 2: Find the minimum value of m
  (let a := 1 -- Use the value of a found in part 1
   ∃ (m : ℝ), (∃ (n : ℝ), f n a ≤ m - f (-n) a) ∧
              ∀ (m' : ℝ), (∃ (n : ℝ), f n a ≤ m' - f (-n) a) → m' ≥ m) ∧
  -- The actual solutions
  (let a := 1
   let m := 4
   (∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
   (∃ (n : ℝ), f n a ≤ m - f (-n) a) ∧
   ∀ (m' : ℝ), (∃ (n : ℝ), f n a ≤ m' - f (-n) a) → m' ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2069_206908


namespace NUMINAMATH_CALUDE_triangle_side_square_sum_bound_l2069_206914

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_radius : 0 < R
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The sum of squares of triangle sides is less than or equal to 9 times the square of its circumradius -/
theorem triangle_side_square_sum_bound (t : Triangle) : t.a^2 + t.b^2 + t.c^2 ≤ 9 * t.R^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_square_sum_bound_l2069_206914


namespace NUMINAMATH_CALUDE_mary_speed_calculation_l2069_206919

/-- Mary's running speed in miles per hour -/
def mary_speed : ℝ := sorry

/-- Jimmy's running speed in miles per hour -/
def jimmy_speed : ℝ := 4

/-- Time elapsed in hours -/
def time : ℝ := 1

/-- Distance between Mary and Jimmy after 1 hour in miles -/
def distance : ℝ := 9

theorem mary_speed_calculation :
  mary_speed = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_speed_calculation_l2069_206919


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_capacity_l2069_206982

/-- The total capacity of a Ferris wheel -/
def ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) : ℕ :=
  num_seats * people_per_seat

/-- Theorem: The capacity of a Ferris wheel with 14 seats and 6 people per seat is 84 -/
theorem paradise_park_ferris_wheel_capacity :
  ferris_wheel_capacity 14 6 = 84 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_capacity_l2069_206982


namespace NUMINAMATH_CALUDE_lending_time_problem_l2069_206956

/-- The problem of finding the lending time for the second part of a sum --/
theorem lending_time_problem (total_sum : ℝ) (second_part : ℝ) (rate1 : ℝ) (time1 : ℝ) (rate2 : ℝ) :
  total_sum = 2743 →
  second_part = 1688 →
  rate1 = 0.03 →
  time1 = 8 →
  rate2 = 0.05 →
  (total_sum - second_part) * rate1 * time1 = second_part * rate2 * 3 :=
by
  sorry

#check lending_time_problem

end NUMINAMATH_CALUDE_lending_time_problem_l2069_206956


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l2069_206963

/-- Given three squares A, B, and C, where the perimeter of A is 20, 
    the perimeter of B is 40, and each side of C is equal to the sum 
    of the side lengths of A and B, the perimeter of C is 60. -/
theorem square_perimeter_problem (A B C : Set ℝ) : 
  (∃ (sA sB sC : ℝ),
    (∀ x ∈ A, ∃ y ∈ A, |x - y| = sA) ∧
    (∀ x ∈ B, ∃ y ∈ B, |x - y| = sB) ∧
    (∀ x ∈ C, ∃ y ∈ C, |x - y| = sC) ∧
    (4 * sA = 20) ∧
    (4 * sB = 40) ∧
    (sC = sA + sB)) →
  (∃ (p : ℝ), (∀ x ∈ C, ∃ y ∈ C, |x - y| = p / 4) ∧ p = 60) := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l2069_206963


namespace NUMINAMATH_CALUDE_scientific_notation_of_billion_l2069_206921

theorem scientific_notation_of_billion (x : ℝ) (h : x = 61345.05) :
  x * (10 : ℝ)^9 = 6.134505 * (10 : ℝ)^12 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_billion_l2069_206921


namespace NUMINAMATH_CALUDE_conner_start_rocks_l2069_206984

/-- Represents the number of rocks collected by each person on each day -/
structure RockCollection where
  sydney_start : ℕ
  conner_start : ℕ
  sydney_day1 : ℕ
  conner_day1 : ℕ
  sydney_day2 : ℕ
  conner_day2 : ℕ
  sydney_day3 : ℕ
  conner_day3 : ℕ

/-- The rock collecting contest scenario -/
def contest_scenario : RockCollection where
  sydney_start := 837
  conner_start := 723  -- This is what we want to prove
  sydney_day1 := 4
  conner_day1 := 8 * 4
  sydney_day2 := 0
  conner_day2 := 123
  sydney_day3 := 2 * (8 * 4)
  conner_day3 := 27

/-- Calculates the total rocks for each person at the end of the contest -/
def total_rocks (rc : RockCollection) : ℕ × ℕ :=
  (rc.sydney_start + rc.sydney_day1 + rc.sydney_day2 + rc.sydney_day3,
   rc.conner_start + rc.conner_day1 + rc.conner_day2 + rc.conner_day3)

/-- Theorem stating that Conner must have started with 723 rocks to at least tie Sydney -/
theorem conner_start_rocks : 
  let (sydney_total, conner_total) := total_rocks contest_scenario
  conner_total ≥ sydney_total ∧ contest_scenario.conner_start = 723 := by
  sorry


end NUMINAMATH_CALUDE_conner_start_rocks_l2069_206984


namespace NUMINAMATH_CALUDE_tree_height_l2069_206901

theorem tree_height (hop_distance : ℕ) (slip_distance : ℕ) (total_hours : ℕ) (tree_height : ℕ) : 
  hop_distance = 3 →
  slip_distance = 2 →
  total_hours = 17 →
  tree_height = (total_hours - 1) * (hop_distance - slip_distance) + hop_distance := by
sorry

#eval (17 - 1) * (3 - 2) + 3

end NUMINAMATH_CALUDE_tree_height_l2069_206901


namespace NUMINAMATH_CALUDE_complex_number_theorem_l2069_206918

theorem complex_number_theorem (z : ℂ) :
  (∃ (k : ℝ), z / 4 = k * I) →
  Complex.abs z = 2 * Real.sqrt 5 →
  z = 2 * I ∨ z = -2 * I := by sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l2069_206918


namespace NUMINAMATH_CALUDE_library_visitors_on_sunday_l2069_206939

/-- The average number of visitors on non-Sunday days -/
def avg_visitors_non_sunday : ℕ := 240

/-- The total number of days in the month -/
def total_days : ℕ := 30

/-- The number of Sundays in the month -/
def num_sundays : ℕ := 5

/-- The average number of visitors per day in the month -/
def avg_visitors_per_day : ℕ := 300

/-- The average number of visitors on Sundays -/
def avg_visitors_sunday : ℕ := 600

theorem library_visitors_on_sunday :
  num_sundays * avg_visitors_sunday + (total_days - num_sundays) * avg_visitors_non_sunday =
  total_days * avg_visitors_per_day := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_on_sunday_l2069_206939


namespace NUMINAMATH_CALUDE_kevin_kangaroo_hops_l2069_206949

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem kevin_kangaroo_hops :
  let a : ℚ := 1/4
  let r : ℚ := 7/16
  let n : ℕ := 5
  geometric_sum a r n = 1031769/2359296 := by sorry

end NUMINAMATH_CALUDE_kevin_kangaroo_hops_l2069_206949


namespace NUMINAMATH_CALUDE_root_sum_and_square_l2069_206913

theorem root_sum_and_square (α β : ℝ) : 
  (α^2 - α - 2006 = 0) → 
  (β^2 - β - 2006 = 0) → 
  (α + β = 1) →
  α + β^2 = 2007 := by
sorry

end NUMINAMATH_CALUDE_root_sum_and_square_l2069_206913


namespace NUMINAMATH_CALUDE_quadratic_expansion_sum_l2069_206946

theorem quadratic_expansion_sum (a b : ℝ) : 
  (∀ x : ℝ, x^2 + 4*x + 3 = (x - 1)^2 + a*(x - 1) + b) → a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expansion_sum_l2069_206946


namespace NUMINAMATH_CALUDE_product_xyz_is_one_l2069_206916

theorem product_xyz_is_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) 
  (hy_nonzero : y ≠ 0) 
  (hz_nonzero : z ≠ 0) : 
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_is_one_l2069_206916


namespace NUMINAMATH_CALUDE_larger_share_theorem_l2069_206987

/-- Given two investments and a total profit, calculates the share of profit for the larger investment -/
def calculate_larger_share (investment1 : ℕ) (investment2 : ℕ) (total_profit : ℕ) : ℕ :=
  let larger_investment := max investment1 investment2
  let total_investment := investment1 + investment2
  (larger_investment * total_profit) / total_investment

theorem larger_share_theorem (investment1 investment2 total_profit : ℕ) 
  (h1 : investment1 = 22500) 
  (h2 : investment2 = 35000) 
  (h3 : total_profit = 13800) :
  calculate_larger_share investment1 investment2 total_profit = 8400 := by
  sorry

#eval calculate_larger_share 22500 35000 13800

end NUMINAMATH_CALUDE_larger_share_theorem_l2069_206987


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2069_206925

/-- The perimeter of a rhombus with diagonals measuring 20 feet and 16 feet is 8√41 feet. -/
theorem rhombus_perimeter (d₁ d₂ : ℝ) (h₁ : d₁ = 20) (h₂ : d₂ = 16) :
  let side := Real.sqrt ((d₁/2)^2 + (d₂/2)^2)
  4 * side = 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2069_206925


namespace NUMINAMATH_CALUDE_pages_copied_for_30_dollars_l2069_206968

/-- The number of pages that can be copied for a given amount of money -/
def pages_copied (cost_per_2_pages : ℚ) (amount : ℚ) : ℚ :=
  (amount / cost_per_2_pages) * 2

/-- Theorem: Given that it costs 4 cents to copy 2 pages, 
    the number of pages that can be copied for $30 is 1500 -/
theorem pages_copied_for_30_dollars : 
  pages_copied (4/100) 30 = 1500 := by
  sorry

#eval pages_copied (4/100) 30

end NUMINAMATH_CALUDE_pages_copied_for_30_dollars_l2069_206968


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2069_206941

theorem parallel_vectors_x_value (x : ℝ) (a b : ℝ × ℝ) :
  a = (2, x) →
  b = (4, -1) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2069_206941


namespace NUMINAMATH_CALUDE_perfect_squares_closed_under_multiplication_perfect_squares_not_closed_under_addition_perfect_squares_not_closed_under_subtraction_perfect_squares_not_closed_under_division_l2069_206969

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def perfect_squares : Set ℕ := {n : ℕ | is_perfect_square n ∧ n > 0}

theorem perfect_squares_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ perfect_squares → b ∈ perfect_squares → (a * b) ∈ perfect_squares :=
sorry

theorem perfect_squares_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ (a + b) ∉ perfect_squares :=
sorry

theorem perfect_squares_not_closed_under_subtraction :
  ∃ a b : ℕ, a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ a > b ∧ (a - b) ∉ perfect_squares :=
sorry

theorem perfect_squares_not_closed_under_division :
  ∃ a b : ℕ, a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ b ≠ 0 ∧ (a / b) ∉ perfect_squares :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_closed_under_multiplication_perfect_squares_not_closed_under_addition_perfect_squares_not_closed_under_subtraction_perfect_squares_not_closed_under_division_l2069_206969


namespace NUMINAMATH_CALUDE_decimal_to_base_conversion_l2069_206932

theorem decimal_to_base_conversion (x : ℕ) : 
  (4 * x + 7 = 71) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_base_conversion_l2069_206932


namespace NUMINAMATH_CALUDE_program_output_is_44_l2069_206971

/-- The output value of the program -/
def program_output : ℕ := 44

/-- Theorem stating that the program output is 44 -/
theorem program_output_is_44 : program_output = 44 := by
  sorry

end NUMINAMATH_CALUDE_program_output_is_44_l2069_206971


namespace NUMINAMATH_CALUDE_m_subset_p_subset_n_l2069_206993

/-- Set M definition -/
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

/-- Set N definition -/
def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

/-- Set P definition -/
def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

/-- Theorem stating M ⊂ P ⊂ N -/
theorem m_subset_p_subset_n : M ⊆ P ∧ P ⊆ N := by sorry

end NUMINAMATH_CALUDE_m_subset_p_subset_n_l2069_206993


namespace NUMINAMATH_CALUDE_kamals_math_marks_l2069_206943

def english_marks : ℕ := 96
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 79
def total_subjects : ℕ := 5

theorem kamals_math_marks :
  let total_marks := average_marks * total_subjects
  let known_marks := english_marks + physics_marks + chemistry_marks + biology_marks
  let math_marks := total_marks - known_marks
  math_marks = 65 := by sorry

end NUMINAMATH_CALUDE_kamals_math_marks_l2069_206943


namespace NUMINAMATH_CALUDE_toy_pile_ratio_l2069_206980

theorem toy_pile_ratio : 
  let total_toys : ℕ := 120
  let larger_pile : ℕ := 80
  let smaller_pile : ℕ := total_toys - larger_pile
  (larger_pile : ℚ) / smaller_pile = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_toy_pile_ratio_l2069_206980


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2069_206953

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2069_206953


namespace NUMINAMATH_CALUDE_pepsi_amount_l2069_206948

/-- Represents the drink inventory and packing constraints -/
structure DrinkInventory where
  maaza : ℕ
  sprite : ℕ
  total_cans : ℕ
  pepsi : ℕ

/-- Calculates the greatest common divisor of two natural numbers -/
def gcd (a b : ℕ) : ℕ := sorry

/-- Theorem: Given the inventory and constraints, the amount of Pepsi is 144 liters -/
theorem pepsi_amount (inventory : DrinkInventory) 
  (h1 : inventory.maaza = 80)
  (h2 : inventory.sprite = 368)
  (h3 : inventory.total_cans = 37)
  (h4 : ∃ (can_size : ℕ), can_size > 0 ∧ 
        inventory.maaza % can_size = 0 ∧ 
        inventory.sprite % can_size = 0 ∧
        inventory.pepsi % can_size = 0 ∧
        inventory.total_cans = inventory.maaza / can_size + inventory.sprite / can_size + inventory.pepsi / can_size)
  : inventory.pepsi = 144 := by
  sorry

end NUMINAMATH_CALUDE_pepsi_amount_l2069_206948


namespace NUMINAMATH_CALUDE_max_min_on_interval_l2069_206981

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_on_interval :
  ∃ (a b : ℝ), a ∈ Set.Icc 0 3 ∧ b ∈ Set.Icc 0 3 ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f a) ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 :=
sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l2069_206981


namespace NUMINAMATH_CALUDE_sum_of_median_scores_l2069_206999

-- Define the type for basketball scores
def Score := ℕ

-- Define a function to calculate the median of a list of scores
noncomputable def median (scores : List Score) : ℝ := sorry

-- Define the scores for player A
def scoresA : List Score := sorry

-- Define the scores for player B
def scoresB : List Score := sorry

-- Theorem to prove
theorem sum_of_median_scores : 
  median scoresA + median scoresB = 64 := by sorry

end NUMINAMATH_CALUDE_sum_of_median_scores_l2069_206999


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2069_206975

theorem average_speed_calculation (distance1 : ℝ) (distance2 : ℝ) (time1 : ℝ) (time2 : ℝ) 
  (h1 : distance1 = 100) 
  (h2 : distance2 = 60) 
  (h3 : time1 = 1) 
  (h4 : time2 = 1) : 
  (distance1 + distance2) / (time1 + time2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2069_206975
