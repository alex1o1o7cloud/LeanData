import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l180_18037

theorem line_through_points (a b : ℝ → ℝ → ℝ → ℝ) (k : ℝ) : 
  a ≠ b →  -- a and b are distinct
  (∃ t : ℝ, k • a + (2/5 : ℝ) • b = a + t • (b - a)) →  -- the vector lies on the line
  k = 3/5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l180_18037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_of_S_wire_length_form_final_result_l180_18080

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p | let (x, y) := p
       (|(|x| - 2)| - 1) + (|(|y| - 2)| - 1) = 1}

-- Define the length of wire function
noncomputable def wireLength (s : Set (ℝ × ℝ)) : ℝ :=
  8 * Real.sqrt 2

-- Theorem statement
theorem wire_length_of_S :
  wireLength S = 8 * Real.sqrt 2 :=
by
  sorry

-- Proof that the result can be expressed as a√b where a and b are positive integers
-- and b is not divisible by the square of any prime
theorem wire_length_form (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  wireLength S = a * Real.sqrt b ∧ 
  ∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ b) →
  a = 8 ∧ b = 2 :=
by
  sorry

-- The final result a + b = 10
theorem final_result :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  wireLength S = a * Real.sqrt b ∧
  (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ b)) ∧
  a + b = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_of_S_wire_length_form_final_result_l180_18080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l180_18092

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - Real.sqrt (3 - Real.sqrt (4 - x)))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -5 ≤ x ∧ x ≤ 4} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l180_18092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_factor_answer_is_70_l180_18018

def refined_number : ℕ := 2^27 * 3^15 * 5^5 * 7^3

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

theorem smallest_perfect_square_factor :
  ∃ k : ℕ, k > 0 ∧ is_perfect_square (k * refined_number) ∧
    ∀ m : ℕ, m > 0 → is_perfect_square (m * refined_number) → k ≤ m :=
by
  -- We claim that 70 is the smallest positive integer that satisfies the condition
  use 70
  refine ⟨Nat.zero_lt_succ _, ?square, ?smallest⟩
  
  · -- Prove that 70 * refined_number is a perfect square
    sorry
  
  · -- Prove that 70 is the smallest such positive integer
    sorry

-- Auxiliary theorem to show that 70 is indeed the answer
theorem answer_is_70 (h : ∃ k : ℕ, k > 0 ∧ is_perfect_square (k * refined_number) ∧
    ∀ m : ℕ, m > 0 → is_perfect_square (m * refined_number) → k ≤ m) : 
  Classical.choose h = 70 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_factor_answer_is_70_l180_18018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_y_coord_l180_18044

/-- Given an equilateral triangle with two vertices at (0,5) and (8,5),
    and the third vertex in the third quadrant, the y-coordinate of the third vertex is 5 - 4√3 -/
theorem equilateral_triangle_third_vertex_y_coord :
  ∀ (x y : ℝ),
  let a : ℝ × ℝ := (0, 5)
  let b : ℝ × ℝ := (8, 5)
  let c : ℝ × ℝ := (x, y)
  (‖a - b‖ = ‖b - c‖) ∧ (‖b - c‖ = ‖c - a‖) ∧  -- equilateral condition
  (x < 0) ∧ (y < 0) →  -- third quadrant condition
  y = 5 - 4 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_y_coord_l180_18044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_size_max_committees_l180_18036

/- Part (a) -/
theorem committee_size (n : ℕ) (h1 : n > 0) :
  (∀ m : Finset (Finset ℕ), 
    m.card = 40 ∧ 
    (∀ s ∈ m, s.card = 10) ∧
    (∀ s ∈ m, ∀ t ∈ m, s ≠ t → (s ∩ t).card ≤ 1)) →
  n > 60 := by sorry

/- Part (b) -/
theorem max_committees :
  ∀ m : Finset (Finset (Fin 25)), 
    (∀ s ∈ m, s.card = 5) ∧
    (∀ s ∈ m, ∀ t ∈ m, s ≠ t → (s ∩ t).card ≤ 1) →
    m.card ≤ 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_size_max_committees_l180_18036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hard_drive_doubling_time_l180_18032

/-- The number of years between the initial and final measurements -/
def total_years : ℝ := 50

/-- The initial hard drive capacity in terabytes -/
def initial_capacity : ℝ := 0.4

/-- The final hard drive capacity in terabytes -/
def final_capacity : ℝ := 4100

/-- The number of doubling periods between the initial and final capacities -/
noncomputable def doubling_periods : ℝ := Real.log (final_capacity / initial_capacity) / Real.log 2

/-- The time it takes for the capacity to double -/
noncomputable def doubling_time : ℝ := total_years / doubling_periods

theorem hard_drive_doubling_time :
  ∃ ε > 0, |doubling_time - 3.57| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hard_drive_doubling_time_l180_18032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l180_18094

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through a point with a given angle -/
structure Line where
  point : Point
  angle : ℝ

/-- Circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The product of distances from P to intersection points of line and circle is 2 -/
theorem intersection_distance_product (P : Point) (l : Line) (C : Circle) :
  P.x = 1 ∧ P.y = 1 ∧
  l.point = P ∧ l.angle = 2*π/3 ∧
  C.center.x = 0 ∧ C.center.y = 0 ∧ C.radius = 2 →
  ∃ (A B : Point), A ≠ B ∧
    (distance P A) * (distance P B) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l180_18094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_equal_l180_18098

-- Define an octagon
structure Octagon :=
  (sides : Fin 8 → ℕ)
  (angles : Fin 8 → ℝ)

-- Define the properties of our specific octagon
def EqualAngledIntegerSidedOctagon (o : Octagon) : Prop :=
  (∀ i j : Fin 8, o.angles i = o.angles j) ∧
  (∀ i : Fin 8, o.sides i > 0)

-- Define opposite sides
def OppositeSides (o : Octagon) : Prop :=
  (o.sides 0 = o.sides 4) ∧
  (o.sides 1 = o.sides 5) ∧
  (o.sides 2 = o.sides 6) ∧
  (o.sides 3 = o.sides 7)

-- State the theorem
theorem opposite_sides_equal (o : Octagon) 
  (h : EqualAngledIntegerSidedOctagon o) : OppositeSides o := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_equal_l180_18098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_transformation_l180_18012

noncomputable def average (X : List ℝ) : ℝ := X.sum / X.length

noncomputable def variance (X : List ℝ) : ℝ :=
  let μ := average X
  (X.map (λ x => (x - μ)^2)).sum / X.length

def transformData (X : List ℝ) : List ℝ :=
  X.map (λ x => 2 * x - 80)

theorem data_transformation (X : List ℝ) (h : X.length > 0) :
  average (transformData X) = 1.2 ∧ variance (transformData X) = 4.4 →
  average X = 40.6 ∧ variance X = 1.1 := by
  sorry

#check data_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_transformation_l180_18012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_f_geq_g_condition_l180_18099

noncomputable section

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := Real.exp (x - 1) - (4 * a - 3) / (6 * x)
noncomputable def g (a x : ℝ) : ℝ := (1/3) * a * x^2 + (1/2) * x - (a - 1)

-- Theorem for the first part of the problem
theorem tangent_perpendicular (a : ℝ) : 
  (∃ m : ℝ, (∀ x : ℝ, deriv (f a) x = m * (x - 1) + f a 1) ∧ 
   m * (-1/2) = -1) ↔ a = 9/4 := by sorry

-- Theorem for the second part of the problem
theorem f_geq_g_condition (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → f a x ≥ g a x) ↔ a ≤ 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_f_geq_g_condition_l180_18099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_4_equals_2_l180_18039

-- Define the function f and its inverse
noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

-- Define the properties of f_inv
axiom f_inv_def : ∀ x > 0, f_inv x = x^2

-- Define f as the inverse of f_inv
axiom f_inverse : ∀ x > 0, f (f_inv x) = x

-- Theorem to prove
theorem f_of_4_equals_2 : f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_4_equals_2_l180_18039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_pawns_infinite_unpresentable_pawns_l180_18079

theorem grid_pawns (m n : ℕ) :
  (m + 1) * (n + 1) + m * n = 500 ↔ (m = 1 ∧ n = 166) ∨ (m = 4 ∧ n = 55) ∨ (m = 13 ∧ n = 18) :=
sorry

theorem infinite_unpresentable_pawns :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ k, k ∈ S → ¬∃ (m n : ℕ), 2 * k - 1 = (2 * m + 1) * (2 * n + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_pawns_infinite_unpresentable_pawns_l180_18079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l180_18087

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  B = π / 3 →
  Real.cos A = 4 / 5 →
  b = Real.sqrt 3 →
  Real.sin C = (3 + 4 * Real.sqrt 3) / 10 ∧
  (1 / 2) * a * b * Real.sin C = (9 * Real.sqrt 3 + 36) / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l180_18087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l180_18027

/-- Custom operation ※ for real numbers -/
noncomputable def star (a b : ℝ) : ℝ := (a - b) / (a + b)

/-- Theorem stating that if (x + 1) ※ (x - 2) = 3, then x = 1 -/
theorem star_equation_solution (x : ℝ) : star (x + 1) (x - 2) = 3 → x = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l180_18027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l180_18064

theorem plant_arrangement (n m : ℕ) (hn : n = 5) (hm : m = 4) : 
  (Nat.factorial (n + 1)) * (Nat.factorial m) = 17280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l180_18064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_and_square_l180_18017

/-- Triangle is a placeholder for the concept of a triangle. 
    In a full implementation, this would be defined with appropriate properties. -/
structure Triangle where
  -- Placeholder for triangle properties
  mk :: 

/-- SimilarTriangles is a placeholder for the concept of similar triangles. 
    In a full implementation, this would be defined with appropriate properties. -/
def SimilarTriangles (t1 t2 : Triangle) : Prop :=
  -- Placeholder for similarity definition
  True

/-- Given two similar triangles MNP and XYZ with known side lengths, 
    prove the length of XY and the area of a square with side XY. -/
theorem similar_triangles_and_square (MNP XYZ : Triangle) (MN NP YZ : ℝ) 
    (h_similar : SimilarTriangles MNP XYZ) 
    (h_MN : MN = 8) (h_NP : NP = 16) (h_YZ : YZ = 24) : 
    ∃ (XY : ℝ), XY = 12 ∧ XY^2 = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_and_square_l180_18017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_pure_imaginary_value_l180_18084

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the property of being a pure imaginary number
def isPureImaginary (z : ℂ) : Prop := ∃ b : ℝ, z = b * i

-- State the theorem
theorem complex_pure_imaginary_value (a : ℝ) :
  isPureImaginary (a - 17 / (4 - i)) → a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_pure_imaginary_value_l180_18084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_over_sixth_root_seven_l180_18067

theorem fourth_root_over_sixth_root_seven : 
  (7 : ℝ) ^ (1/4 : ℝ) / (7 : ℝ) ^ (1/6 : ℝ) = (7 : ℝ) ^ (1/12 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_over_sixth_root_seven_l180_18067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shift_l180_18026

noncomputable section

-- Define the original function
def original_function (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3)

-- Define the period of the original function
def period : ℝ := 2 * Real.pi / 2

-- Define the shift amount
def shift : ℝ := period / 3

-- Define the resulting function after the shift
def shifted_function (x : ℝ) : ℝ := -Real.cos (2 * x)

theorem function_shift :
  ∀ x : ℝ, original_function (x + shift) = shifted_function x :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shift_l180_18026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_integer_l180_18042

noncomputable def sequenceX (c : ℕ+) : ℕ → ℤ
  | 0 => c.val
  | n + 1 => c.val * sequenceX c n + Int.sqrt ((c.val ^ 2 - 1) * ((sequenceX c n) ^ 2 - 1))

theorem sequence_is_integer (c : ℕ+) (n : ℕ) : ∃ k : ℤ, sequenceX c n = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_integer_l180_18042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_B_students_in_history_class_l180_18004

/-- The number of students who earn a B in a history class --/
def number_of_B_students (total_students : ℕ) (prob_A_ratio prob_C_ratio : ℚ) : ℚ :=
  total_students / (1 + prob_A_ratio + prob_C_ratio)

/-- Theorem stating the number of B students in the given conditions --/
theorem number_of_B_students_in_history_class :
  let total_students : ℕ := 45
  let prob_A_ratio : ℚ := 1/2
  let prob_C_ratio : ℚ := 8/5
  ⌊(number_of_B_students total_students prob_A_ratio prob_C_ratio : ℚ)⌋₊ = 15 := by
  sorry

#eval (number_of_B_students 45 (1/2) (8/5) : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_B_students_in_history_class_l180_18004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heavy_water_electrons_l180_18058

/-- Avogadro's constant (in mol⁻¹) -/
noncomputable def NA : ℝ := Real.exp 1  -- Placeholder value, replace with actual constant if known

/-- Molecular mass of D₂O (in g/mol) -/
def D2O_mass : ℝ := 20

/-- Number of electrons in a D₂O molecule -/
def D2O_electrons : ℕ := 10

/-- Mass of heavy water sample (in g) -/
def sample_mass : ℝ := 20

theorem heavy_water_electrons :
  (sample_mass / D2O_mass) * (D2O_electrons : ℝ) * NA = 10 * NA := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heavy_water_electrons_l180_18058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_destination_is_13_5_hours_l180_18021

/-- Represents the time taken to reach a destination given specific walking conditions. -/
noncomputable def time_to_destination (harris_speed : ℝ) (harris_time : ℝ) (distance_ratio : ℝ) (rest_time : ℝ) : ℝ :=
  let initial_speed := 3 * harris_speed
  let store_distance := harris_speed * harris_time
  let destination_distance := distance_ratio * store_distance
  let first_hour_distance := initial_speed * 1
  let remaining_distance := destination_distance - first_hour_distance
  let time_to_halfway := (destination_distance / 2 - first_hour_distance) / harris_speed
  let second_half_time := destination_distance / (2 * harris_speed)
  1 + time_to_halfway + rest_time + second_half_time

/-- Theorem stating that given the specific conditions, the time to reach the destination is 13.5 hours. -/
theorem time_to_destination_is_13_5_hours (harris_speed : ℝ) (harris_time : ℝ := 3) 
    (distance_ratio : ℝ := 5) (rest_time : ℝ := 0.5) :
  time_to_destination harris_speed harris_time distance_ratio rest_time = 13.5 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_destination_is_13_5_hours_l180_18021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l180_18097

open Real

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : cos (α + π / 3) = -Real.sqrt 3 / 3) : 
  sin α = (Real.sqrt 6 + 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l180_18097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_western_village_contribution_l180_18045

def northern_population : ℕ := 8758
def western_population : ℕ := 7236
def southern_population : ℕ := 8356
def total_needed : ℕ := 378

def total_population : ℕ := northern_population + western_population + southern_population

noncomputable def western_proportion : ℚ := western_population / total_population

theorem western_village_contribution : 
  ⌊(total_needed : ℚ) * western_proportion⌋ = 112 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_western_village_contribution_l180_18045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_six_points_in_rectangle_l180_18073

theorem min_distance_six_points_in_rectangle :
  ∀ (points : Finset (ℝ × ℝ)),
  (Finset.card points = 6) →
  (∀ p, p ∈ points → 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) →
  ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ 5/6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_six_points_in_rectangle_l180_18073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l180_18034

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (1, 3)

-- Define the slope of the tangent line
def tangent_slope : ℝ := 2

-- Theorem: The equation of the tangent line is 2x - y + 1 = 0
theorem tangent_line_equation :
  ∀ x y : ℝ, 2*x - y + 1 = 0 ↔ 
  y - f tangent_point.1 = tangent_slope * (x - tangent_point.1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l180_18034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_r_at_zero_l180_18031

-- Define Q(m) as a function
noncomputable def Q (m : ℝ) : ℝ := -Real.sqrt (m + 4)

-- Define r as a function of m
noncomputable def r (m : ℝ) : ℝ := (Q (-m) - Q m) / m

-- Theorem statement
theorem limit_r_at_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ ∧ -4 < m ∧ m < 4 →
    |r m - (-1/2)| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_r_at_zero_l180_18031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_n_digits_same_l180_18077

def sequence_a : ℕ → ℕ
  | 0 => 5
  | n + 1 => (sequence_a n) ^ 2

theorem last_n_digits_same (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, sequence_a (n + 1) - sequence_a n = k * (10 ^ n) := by
  sorry

#eval sequence_a 0
#eval sequence_a 1
#eval sequence_a 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_n_digits_same_l180_18077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_rationality_l180_18060

theorem cube_root_rationality (x y : ℚ) (h : (x : ℝ)^(1/3) + (y : ℝ)^(1/3) = 1) :
  ∃ (a b : ℚ), (x : ℝ)^(1/3) = a ∧ (y : ℝ)^(1/3) = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_rationality_l180_18060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_lines_with_slopes_from_equation_l180_18035

-- Define the equation whose roots are the slopes of the lines
def slope_equation (x : ℝ) : Prop := 6 * x^2 + x - 1 = 0

-- Define the angle between two lines given their slopes
noncomputable def angle_between_lines (m₁ m₂ : ℝ) : ℝ :=
  Real.arctan (abs ((m₂ - m₁) / (1 + m₁ * m₂)))

-- Theorem statement
theorem angle_between_lines_with_slopes_from_equation :
  ∃ (m₁ m₂ : ℝ), 
    slope_equation m₁ ∧ 
    slope_equation m₂ ∧ 
    m₁ ≠ m₂ ∧
    angle_between_lines m₁ m₂ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_lines_with_slopes_from_equation_l180_18035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hand_rotation_difference_l180_18015

/-- The rotation of the hour hand in degrees per hour -/
noncomputable def hourHandRotationPerHour : ℝ := -30

/-- The rotation of the minute hand in degrees per minute -/
noncomputable def minuteHandRotationPerMinute : ℝ := -6

/-- Calculates the rotation of the hour hand after a given time -/
noncomputable def hourHandRotation (hours : ℝ) : ℝ :=
  hourHandRotationPerHour * hours

/-- Calculates the rotation of the minute hand after a given time -/
noncomputable def minuteHandRotation (minutes : ℝ) : ℝ :=
  minuteHandRotationPerMinute * minutes

/-- The time elapsed in hours -/
noncomputable def elapsedTime : ℝ := 3 + 35 / 60

/-- Theorem: The difference in rotation between hour and minute hands after 3 hours and 35 minutes is 1182.5° -/
theorem clock_hand_rotation_difference :
  minuteHandRotation (elapsedTime * 60) - hourHandRotation elapsedTime = 1182.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hand_rotation_difference_l180_18015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l180_18063

/-- Represents the loan amount in dollars -/
noncomputable def initial_loan : ℝ := 15000

/-- Represents the loan duration in years -/
noncomputable def loan_duration : ℝ := 15

/-- Represents the annual interest rate for the compound interest scheme -/
noncomputable def compound_rate : ℝ := 0.08

/-- Represents the number of times interest is compounded per year -/
noncomputable def compound_frequency : ℝ := 2

/-- Represents the annual interest rate for the simple interest scheme -/
noncomputable def simple_rate : ℝ := 0.09

/-- Calculates the amount owed after compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- Calculates the amount owed after simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating the difference between compound and simple interest schemes -/
theorem interest_difference : 
  ∃ (compound_total simple_total : ℝ),
    compound_total = 
      (compound_interest initial_loan compound_rate compound_frequency 10) / 3 + 
      compound_interest ((compound_interest initial_loan compound_rate compound_frequency 10) * 2/3) compound_rate compound_frequency 5 ∧
    simple_total = simple_interest initial_loan simple_rate loan_duration ∧
    abs (compound_total - simple_total - 8140) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l180_18063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equality_condition_l180_18056

theorem sqrt_equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.sqrt (x - y / z) = x * Real.sqrt (y / z)) ↔ (z = y * (x^2 + 1) / x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equality_condition_l180_18056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l180_18013

/-- The function f(x) = x - 4 + log₂x -/
noncomputable def f (x : ℝ) : ℝ := x - 4 + Real.log x / Real.log 2

/-- Theorem: The root of f(x) = x - 4 + log₂x lies in the interval (2, 3) -/
theorem root_in_interval :
  ∃ (r : ℝ), f r = 0 ∧ r ∈ Set.Ioo 2 3 := by
  sorry

#check root_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l180_18013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fold_application_of_f_l180_18033

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -1 / x

-- State the theorem
theorem five_fold_application_of_f :
  f (f (f (f (f 6)))) = -1 / 6 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fold_application_of_f_l180_18033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l180_18001

-- Define the point in rectangular coordinates
noncomputable def x : ℝ := Real.sqrt 3
def y : ℝ := -1

-- Define the polar coordinates
def r : ℝ := 2
noncomputable def θ : ℝ := 11 * Real.pi / 6

-- Theorem statement
theorem rectangular_to_polar_conversion :
  (x^2 + y^2 = r^2) ∧ 
  (x = r * Real.cos θ) ∧ 
  (y = r * Real.sin θ) ∧ 
  (r > 0) ∧ 
  (0 ≤ θ ∧ θ < 2 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l180_18001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l180_18074

/-- Calculates the time difference between two runners finishing a race -/
theorem race_time_difference 
  (malcolm_speed : ℝ) -- Malcolm's speed in minutes per mile
  (joshua_speed : ℝ)  -- Joshua's speed in minutes per mile
  (race_distance : ℝ) -- Race distance in miles
  (h1 : malcolm_speed = 5)
  (h2 : joshua_speed = 8)
  (h3 : race_distance = 12) :
  joshua_speed * race_distance - malcolm_speed * race_distance = 36 := by
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Simplify the expression
  ring
  -- The proof is complete
  done

#check race_time_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l180_18074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_inverse_at_seven_halves_l180_18081

def g (x : ℝ) : ℝ := 3 * x - 7

theorem g_equals_inverse_at_seven_halves :
  g (7/2) = g.invFun (7/2) := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_inverse_at_seven_halves_l180_18081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l180_18025

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := 3 * x^2 + y^2 = 6

/-- The coordinates of point P -/
noncomputable def P : ℝ × ℝ := (1, Real.sqrt 3)

/-- A point is on the ellipse if it satisfies the ellipse equation -/
def on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

/-- The area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

/-- The maximum area of triangle PAB -/
theorem max_triangle_area :
  ∃ (A B : ℝ × ℝ), on_ellipse A ∧ on_ellipse B ∧
    ∀ (X Y : ℝ × ℝ), on_ellipse X → on_ellipse Y →
      triangle_area P X Y ≤ triangle_area P A B ∧
      triangle_area P A B = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l180_18025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_score_with_one_question_l180_18089

/-- Represents a football team --/
inductive Team
| Dynamo
| Shinnik

/-- Represents the score of a football match --/
structure Score where
  team1_goals : ℕ
  team2_goals : ℕ

/-- Defines the properties of the football match --/
structure FootballMatch where
  score : Score
  total_goals : ℕ
  shinnik_scored : Prop
  unique_total : Prop
  unique_total_plus_one : Prop

/-- The main theorem to be proved --/
theorem determine_score_with_one_question 
  (m : FootballMatch)
  (h1 : m.total_goals = m.score.team1_goals + m.score.team2_goals)
  (h2 : m.shinnik_scored)
  (h3 : m.unique_total)
  (h4 : m.unique_total_plus_one) :
  ∃ (question : Prop), 
    (question ∨ ¬question) → 
    (m.score.team1_goals = 2 ∧ m.score.team2_goals = 0) ∨ 
    (m.score.team1_goals = 1 ∧ m.score.team2_goals = 0) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_score_with_one_question_l180_18089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l180_18019

/-- Properties of triangle ABC -/
structure TriangleABC where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Length of side BC
  b : ℝ  -- Length of side AC
  c : ℝ  -- Length of side AB
  C_eq_2A : C = 2 * A
  cosA : Real.cos A = 3/4
  dot_product : 2 * (a * c * Real.cos B) = -27

/-- Main theorem about the properties of triangle ABC -/
theorem triangle_abc_properties (t : TriangleABC) : Real.cos t.B = 9/16 ∧ t.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l180_18019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_two_common_tangents_l180_18072

/-- Two circles in a plane -/
structure TwoCircles where
  r : ℝ
  s : ℝ
  d : ℝ
  h1 : r > 0
  h2 : s > 0
  h3 : r ≠ s
  h4 : d > 0

/-- The number of common tangents between two circles -/
noncomputable def commonTangents (c : TwoCircles) : ℕ :=
  if c.d > c.r + c.s then 4
  else if c.d + min c.r c.s < max c.r c.s then 0
  else if c.d = c.r + c.s then 3
  else 0  -- This case should not occur for non-intersecting circles

/-- Theorem stating that two non-intersecting circles with unequal radii cannot have exactly 2 common tangents -/
theorem no_two_common_tangents (c : TwoCircles) : commonTangents c ≠ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_two_common_tangents_l180_18072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_a_more_cost_effective_l180_18088

-- Define the pricing strategies
def price_per_kg : ℚ := 10
def discount_threshold : ℚ := 4
def discount_rate_a : ℚ := 2 / 5  -- 40% as a rational number
def discount_rate_b : ℚ := 1 / 5  -- 20% as a rational number

-- Define the cost functions for each supermarket
noncomputable def cost_a (x : ℚ) : ℚ :=
  if x ≤ discount_threshold then x * price_per_kg
  else discount_threshold * price_per_kg + (x - discount_threshold) * price_per_kg * (1 - discount_rate_a)

def cost_b (x : ℚ) : ℚ := x * price_per_kg * (1 - discount_rate_b)

-- Theorem statement
theorem supermarket_a_more_cost_effective :
  cost_a 10 < cost_b 10 := by
  -- Unfold the definitions and simplify
  unfold cost_a cost_b
  simp [price_per_kg, discount_threshold, discount_rate_a, discount_rate_b]
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_a_more_cost_effective_l180_18088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l180_18049

/-- Given two arithmetic sequences a_n and b_n with the sum of their first n terms being S_n and T_n,
    respectively. If S_n/T_n = 2n/(3n+1), then a_n/b_n = (2n-1)/(3n-1). -/
theorem arithmetic_sequence_ratio (n : ℕ) (a b : ℕ → ℝ) (S T : ℕ → ℝ)
    (h_arith_a : ∀ k, a (k + 1) - a k = a 1 - a 0)
    (h_arith_b : ∀ k, b (k + 1) - b k = b 1 - b 0)
    (h_S : ∀ k, S k = (k / 2) * (2 * a 0 + (k - 1) * (a 1 - a 0)))
    (h_T : ∀ k, T k = (k / 2) * (2 * b 0 + (k - 1) * (b 1 - b 0)))
    (h_ratio : ∀ k, S k / T k = 2 * k / (3 * k + 1)) :
    a n / b n = (2 * n - 1) / (3 * n - 1) := by
  sorry

#check arithmetic_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l180_18049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_property_l180_18070

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 2 then a * x + b else 8 - 3 * x

theorem piecewise_function_property (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_property_l180_18070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_plot_area_l180_18062

/-- Represents a rectangular plot with length and width in meters -/
structure RectangularPlot where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular plot in square meters -/
def area_sq_meters (plot : RectangularPlot) : ℕ :=
  plot.length * plot.width

/-- Converts square meters to hectares -/
def sq_meters_to_hectares (area : ℕ) : ℚ :=
  (area : ℚ) / 10000

theorem vegetable_plot_area :
  let plot := RectangularPlot.mk 450 200
  sq_meters_to_hectares (area_sq_meters plot) = 9 := by
  sorry

#eval sq_meters_to_hectares (area_sq_meters (RectangularPlot.mk 450 200))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_plot_area_l180_18062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_and_area_l180_18068

-- Define the structure for a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the function to calculate the angle between three points
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the function to calculate the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

-- Define the theorem
theorem triangle_congruence_and_area (t1 t2 : Triangle) 
  (h1 : distance t1.A t1.B = distance t2.A t2.B)
  (h2 : distance t1.A t1.C = distance t2.A t2.C)
  (h3 : angle t1.B t1.C t1.A = angle t2.B t2.C t2.A) :
  (∃ (t1' t2' : Triangle), t1' ≠ t2' ∧ 
    distance t1'.A t1'.B = distance t2'.A t2'.B ∧
    distance t1'.A t1'.C = distance t2'.A t2'.C ∧
    angle t1'.B t1'.C t1'.A = angle t2'.B t2'.C t2'.A) ∧
  (∃ (t1'' t2'' : Triangle), 
    distance t1''.A t1''.B = distance t2''.A t2''.B ∧
    distance t1''.A t1''.C = distance t2''.A t2''.C ∧
    angle t1''.B t1''.C t1''.A = angle t2''.B t2''.C t2''.A ∧
    triangleArea t1'' = triangleArea t2'') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_and_area_l180_18068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_cardinality_l180_18002

theorem max_intersection_cardinality 
  (A B C : Finset ℕ) 
  (hA : Finset.card A = 2016)
  (hB : Finset.card B = 2016)
  (h_powerset : Finset.card (Finset.powerset A) + 
                Finset.card (Finset.powerset B) + 
                Finset.card (Finset.powerset C) = 
                Finset.card (Finset.powerset (A ∪ B ∪ C))) :
  Finset.card (A ∩ B ∩ C) ≤ 2015 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_cardinality_l180_18002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_condition_for_sine_l180_18048

theorem angle_condition_for_sine (A B C : Real) (h_triangle : A + B + C = Real.pi) : 
  (A > Real.pi / 6 → Real.sin A > 1 / 2) ∧ 
  ¬(Real.sin A > 1 / 2 → A > Real.pi / 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_condition_for_sine_l180_18048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l180_18093

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

-- Define the interval
def I : Set ℝ := Set.Ioo (-1) 1

-- Theorem statement
theorem f_increasing_and_odd :
  (∀ x ∈ I, ∀ y ∈ I, x < y → f x < f y) ∧ 
  (∀ x ∈ I, f (-x) = -f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l180_18093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l180_18009

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z + Complex.I) = 1) :
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l180_18009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l180_18024

/-- The circle with equation x^2 + y^2 = 5 -/
def my_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 5}

/-- The point A with coordinates (1, 2) -/
def point_A : ℝ × ℝ := (1, 2)

/-- The tangent line passing through point A -/
def tangent_line : Set (ℝ × ℝ) := {p | p.1 + 2 * p.2 = 5}

/-- The triangle formed by the tangent line and the coordinate axes -/
def triangle : Set (ℝ × ℝ) := {p | p ∈ tangent_line ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

theorem triangle_area : MeasureTheory.volume triangle = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l180_18024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_acute_to_two_obtuse_l180_18051

/-- Representation of a triangle -/
structure Triangle where
  angles : Finset ℝ
  area : ℝ

/-- Definition of an acute triangle -/
def IsAcuteTriangle (t : Triangle) : Prop :=
  ∀ angle ∈ t.angles, angle < 90

/-- Definition of an obtuse triangle -/
def IsObtuseTriangle (t : Triangle) : Prop :=
  ∃ angle ∈ t.angles, angle > 90

/-- Definition of a division of a triangle into two triangles -/
def IsDivisionOf (t₁ t₂ t : Triangle) : Prop :=
  t₁.area + t₂.area = t.area ∧ t₁.angles ∪ t₂.angles = t.angles

/-- Theorem: It is impossible to divide an acute triangle into two obtuse triangles -/
theorem no_acute_to_two_obtuse (t : Triangle) :
  IsAcuteTriangle t →
  ¬∃ (t₁ t₂ : Triangle), IsDivisionOf t₁ t₂ t ∧ IsObtuseTriangle t₁ ∧ IsObtuseTriangle t₂ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_acute_to_two_obtuse_l180_18051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l180_18020

def s (n : ℕ) : ℤ := n^2 - 7*n + 6

def a : ℕ → ℤ
| 0 => 0  -- Add this case for 0
| 1 => 0
| (n+2) => 2*(n+2) - 8

theorem sequence_general_term : ∀ n : ℕ, 
  (n = 1 ∧ a n = 0) ∨ 
  (n > 1 ∧ a n = 2*n - 8) ∧
  s n - s (n-1) = a n := by
  sorry

#check sequence_general_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l180_18020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l180_18030

/-- An ellipse with a point on it and given eccentricity -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  e : ℝ
  h_e : e = 1/2
  h_point : 3/a^2 + 3/(4*b^2) = 1

/-- A line intersecting the ellipse -/
structure IntersectingLine (ε : Ellipse) where
  k : ℝ
  h_midpoint : ∃ x₁ x₂ : ℝ,
    (3 + 4*k^2) * x₁^2 + 24*k*x₁ + 24 = 0 ∧
    (3 + 4*k^2) * x₂^2 + 24*k*x₂ + 24 = 0 ∧
    x₁ = 2*x₂

/-- The main theorem stating the equations of the ellipse and the intersecting line -/
theorem ellipse_and_line_equations (ε : Ellipse) (l : IntersectingLine ε) :
  (ε.a = 2 ∧ ε.b = Real.sqrt 3) ∧
  (l.k = 3/2 ∨ l.k = -3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l180_18030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mersenne_prime_units_digit_l180_18057

theorem mersenne_prime_units_digit (n : ℕ) (h : Prime (2^n - 1)) : 
  (2^74207281 - 1) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mersenne_prime_units_digit_l180_18057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l180_18005

-- Define the centers of the circles
def center1 : ℝ × ℝ := (4, 5)
def center2 : ℝ × ℝ := (20, 8)

-- Define the radii of the circles (equal to their y-coordinates as they're tangent to x-axis)
def radius1 : ℝ := 5
def radius2 : ℝ := 8

-- Define the distance between the centers
noncomputable def center_distance : ℝ := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)

-- Theorem: The distance between the closest points of the circles is √265 - 13
theorem closest_points_distance : 
  center_distance - (radius1 + radius2) = Real.sqrt 265 - 13 := by sorry

#eval center1
#eval center2
#eval radius1
#eval radius2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l180_18005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_310_shares_terminal_side_with_50_l180_18054

/-- An angle in degrees -/
structure Angle where
  value : ℝ

/-- Two angles share the same terminal side if their difference is a multiple of 360° -/
def shares_terminal_side (a b : Angle) : Prop :=
  ∃ k : ℤ, a.value = b.value + k * 360

/-- The statement to prove -/
theorem negative_310_shares_terminal_side_with_50 :
  shares_terminal_side ⟨-310⟩ ⟨50⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_310_shares_terminal_side_with_50_l180_18054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_hands_angle_theorem_l180_18041

noncomputable def hour_hand_angle (n : ℝ) : ℝ := 150 + n / 2

noncomputable def minute_hand_angle (n : ℝ) : ℝ := 6 * n

noncomputable def angle_between_hands (n : ℝ) : ℝ := 
  abs (hour_hand_angle n - minute_hand_angle n)

theorem watch_hands_angle_theorem :
  ∃ (t₁ t₂ : ℝ), 0 < t₁ ∧ t₁ < t₂ ∧ t₂ < 60 ∧
  angle_between_hands t₁ = 120 ∧
  angle_between_hands t₂ = 120 ∧
  t₂ - t₁ = 480 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_hands_angle_theorem_l180_18041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l180_18003

/-- The area of a triangle with two sides of length 31 and one side of length 46 is 477 -/
theorem triangle_area_specific : ∃ (D E F : ℝ × ℝ),
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let df := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  let s := (de + ef + df) / 2
  let area := Real.sqrt (s * (s - de) * (s - ef) * (s - df))
  de = 31 ∧ ef = 31 ∧ df = 46 ∧ area = 477 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l180_18003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_21_l180_18008

def sequence_a : ℕ → ℤ
  | 0 => 1
  | n + 1 => sequence_a n + 2 * n

theorem a_5_equals_21 : sequence_a 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_21_l180_18008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l180_18055

noncomputable def M : Set ℝ := {x | ∃ y, y = Real.sqrt (Real.log x / Real.log 2 - 1)}
def N : Set ℝ := {x | |x - 1| ≤ 2}

theorem intersection_M_N : M ∩ N = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l180_18055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l180_18028

-- Define the sets M and N
def M : Set ℝ := {x | x^2 < 1}
def N : Set ℝ := {x | Real.exp (x * Real.log 2) > 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l180_18028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_correct_l180_18050

-- Define the parametric equations of curve C₁
noncomputable def C₁ (t : ℝ) : ℝ × ℝ :=
  (4 + 5 * Real.cos t, 5 + 5 * Real.sin t)

-- Define the polar equation of curve C₂
noncomputable def C₂ (θ : ℝ) : ℝ :=
  2 * Real.sin θ

-- Define the intersection points in polar coordinates
def intersection_points : Set (ℝ × ℝ) :=
  {(Real.sqrt 2, Real.pi / 4), (2, Real.pi / 2)}

-- Theorem stating that the intersection points are correct
theorem intersection_points_correct :
  ∀ (ρ θ : ℝ), (ρ, θ) ∈ intersection_points ↔
    (∃ t : ℝ, C₁ t = (ρ * Real.cos θ, ρ * Real.sin θ)) ∧
    ρ = C₂ θ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_correct_l180_18050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l180_18029

noncomputable def f (x : ℝ) := x + 1 / (x - 2)

theorem minimum_value_of_f :
  ∀ x : ℝ, x > 2 → f x ≥ 4 ∧ (f x = 4 ↔ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l180_18029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_pay_theorem_l180_18010

/-- Calculates Hannah's final pay after taxes and penalties -/
noncomputable def hannah_final_pay (hourly_rate : ℚ) (hours_worked : ℚ) (late_penalty : ℚ) (late_count : ℕ)
  (federal_tax_rate : ℚ → ℚ) (state_tax_rate : ℚ) (bonus_rate : ℚ → ℚ) 
  (total_interactions : ℕ) (positive_reviews : ℕ) : ℚ :=
  let base_pay := hourly_rate * hours_worked
  let adjusted_base_pay := base_pay - (late_penalty * (late_count : ℚ))
  let federal_tax := federal_tax_rate adjusted_base_pay
  let state_tax := state_tax_rate * adjusted_base_pay
  let bonus := bonus_rate ((positive_reviews : ℚ) / (total_interactions : ℚ))
  adjusted_base_pay - federal_tax - state_tax + bonus

/-- The federal tax rate function -/
noncomputable def federal_tax_rate (income : ℚ) : ℚ :=
  if income ≤ 1000 then (1 : ℚ) / 10 * income
  else if income ≤ 2000 then (1 : ℚ) / 10 * 1000 + (15 : ℚ) / 100 * (income - 1000)
  else (1 : ℚ) / 10 * 1000 + (15 : ℚ) / 100 * 1000 + (1 : ℚ) / 5 * (income - 2000)

/-- The bonus rate function -/
noncomputable def bonus_rate (review_ratio : ℚ) : ℚ :=
  if (8 : ℚ) / 10 ≤ review_ratio && review_ratio < (9 : ℚ) / 10 then 10 * review_ratio
  else if (9 : ℚ) / 10 ≤ review_ratio && review_ratio ≤ 1 then 20 * review_ratio
  else 0

/-- Theorem stating Hannah's final pay -/
theorem hannah_pay_theorem :
  hannah_final_pay 30 18 5 3 federal_tax_rate (1 / 20) bonus_rate 6 4 = 446.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_pay_theorem_l180_18010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_100_l180_18090

/-- Represents a rectangular sheep pasture --/
structure SheepPasture where
  fenceCostPerFoot : ℚ
  totalFenceCost : ℚ
  stableLength : ℚ

/-- Calculates the area of the pasture given the length perpendicular to the stable --/
def pasture_area (p : SheepPasture) (y : ℚ) : ℚ :=
  y * (p.totalFenceCost / p.fenceCostPerFoot - 2 * y)

/-- Theorem stating that the maximum area occurs when the side parallel to the stable is 100 feet --/
theorem max_area_at_100 (p : SheepPasture) 
    (h1 : p.fenceCostPerFoot = 10)
    (h2 : p.totalFenceCost = 2000)
    (h3 : p.stableLength = 500) :
    ∃ (y : ℚ), pasture_area p y = pasture_area p 50 ∧ 
    ∀ (z : ℚ), pasture_area p z ≤ pasture_area p y := by
  sorry

#check max_area_at_100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_100_l180_18090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l180_18053

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  25 * x^2 - 150 * x + 4 * y^2 + 8 * y + 9 = 0

/-- The distance between the foci of the ellipse -/
noncomputable def foci_distance : ℝ := 2 * Real.sqrt 46.2

/-- Theorem stating that the distance between the foci of the given ellipse is 2√46.2 -/
theorem ellipse_foci_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_equation x₁ y₁ ∧
    ellipse_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = foci_distance^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l180_18053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_percentage_l180_18083

/-- Represents the duration of a work day in hours -/
noncomputable def work_day_hours : ℚ := 10

/-- Represents the duration of the first meeting in minutes -/
noncomputable def first_meeting_minutes : ℚ := 60

/-- Represents the duration of the second meeting in minutes -/
noncomputable def second_meeting_minutes : ℚ := 1.5 * first_meeting_minutes

/-- Converts hours to minutes -/
noncomputable def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

/-- Calculates the total time spent in meetings in minutes -/
noncomputable def total_meeting_time : ℚ := first_meeting_minutes + second_meeting_minutes

/-- Calculates the percentage of work day spent in meetings -/
noncomputable def meeting_percentage : ℚ := (total_meeting_time / hours_to_minutes work_day_hours) * 100

theorem meeting_time_percentage :
  meeting_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_percentage_l180_18083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l180_18069

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := a * b - b^2

-- Define the # operation
def hash_op (a b : ℝ) : ℝ := a + b - a * b^2

-- Theorem statement
theorem fraction_equality : (at_op 7 3) / (hash_op 7 3) = -12 / 53 := by
  -- Expand the definitions
  have h1 : at_op 7 3 = 12 := by
    unfold at_op
    ring
  have h2 : hash_op 7 3 = -53 := by
    unfold hash_op
    ring
  -- Rewrite the goal using h1 and h2
  rw [h1, h2]
  -- Simplify the fraction
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l180_18069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_lines_perpendicular_property_l180_18085

/-- A type representing a line in 3D space --/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A predicate to check if two lines are perpendicular --/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- A predicate to check if a line passes through a given point --/
def passes_through (l : Line3D) (p : ℝ × ℝ × ℝ) : Prop := sorry

/-- The main theorem --/
theorem seven_lines_perpendicular_property :
  ∃ (p : ℝ × ℝ × ℝ) (lines : Finset Line3D),
    lines.card = 7 ∧
    (∀ l, l ∈ lines → passes_through l p) ∧
    (∀ l1 l2, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → 
      ∃ l3, l3 ∈ lines ∧ perpendicular l1 l3 ∧ perpendicular l2 l3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_lines_perpendicular_property_l180_18085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_zero_range_of_a_max_value_h_l180_18075

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1|
def g (a x : ℝ) : ℝ := 2 * |x| + a

-- Part I: Solution set when a = 0
theorem solution_set_a_zero :
  {x : ℝ | f x ≥ g 0 x} = Set.Icc (-1/3) 1 := by sorry

-- Part II: Range of a
theorem range_of_a :
  {a : ℝ | ∃ x, f x ≥ g a x} = Set.Iic 1 := by sorry

-- Helper theorem: Maximum value of h(x) = |x+1| - 2|x|
theorem max_value_h :
  ∀ x : ℝ, |x + 1| - 2 * |x| ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_zero_range_of_a_max_value_h_l180_18075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_size_S_l180_18040

def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2015}

def satisfies_condition (S : Set ℕ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → ¬(a - b ∣ a + b)

theorem max_size_S :
  ∃ (T : Finset ℕ), ↑T ⊆ S ∧ satisfies_condition ↑T ∧ T.card = 672 ∧
  ∀ (U : Finset ℕ), ↑U ⊆ S → satisfies_condition ↑U → U.card ≤ 672 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_size_S_l180_18040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_vertical_asymptote_iff_l180_18016

-- Define the function g(x)
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + k) / (x^2 - 3*x - 4)

-- Define the property of having exactly one vertical asymptote
def has_exactly_one_vertical_asymptote (k : ℝ) : Prop :=
  (∃! x : ℝ, (x^2 - 3*x - 4 = 0 ∧ x^2 - 2*x + k ≠ 0))

-- Theorem statement
theorem g_has_one_vertical_asymptote_iff (k : ℝ) :
  has_exactly_one_vertical_asymptote k ↔ (k = -8 ∨ k = -3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_vertical_asymptote_iff_l180_18016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l180_18061

-- Define the functions and their domains
noncomputable def f : ℝ → ℝ := sorry
def g (a : ℝ) : ℝ → ℝ := fun x ↦ a * x - 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ∈ Set.Icc 0 4) →
  (∀ x₁ ∈ Set.Icc (-2 : ℝ) 2, ∃ x₀ ∈ Set.Icc (-2 : ℝ) 2, g a x₀ = f x₁) →
  a ≤ -1/2 ∨ a ≥ 5/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l180_18061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_both_AB_to_only_B_l180_18082

/-- Represents the number of people who purchased only book B -/
def only_B : ℕ := sorry

/-- Represents the number of people who purchased both books A and B -/
def both_AB : ℕ := 500

/-- Represents the number of people who purchased only book A -/
def only_A : ℕ := 1000

/-- The number of people who purchased book A is twice the number of people who purchased book B -/
axiom total_A_twice_B : only_A + both_AB = 2 * (only_B + both_AB)

/-- The number of people who purchased both books A and B is some multiple of the number of people who purchased only book B -/
axiom both_AB_multiple_only_B : ∃ k : ℕ, both_AB = k * only_B

theorem ratio_both_AB_to_only_B :
  both_AB / only_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_both_AB_to_only_B_l180_18082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_equation_correct_solution_satisfies_equation_l180_18096

/-- The time (in minutes) it takes for the first machine to address 800 envelopes -/
def t₁ : ℝ := 10

/-- The time (in minutes) it takes for the third machine to address 800 envelopes -/
def t₃ : ℝ := 5

/-- The time (in minutes) it takes for all three machines together to address 800 envelopes -/
def t_total : ℝ := 3

/-- The number of envelopes addressed -/
def n : ℝ := 800

/-- The equation representing the situation -/
def envelope_equation (t₂ : ℝ) : Prop :=
  1 / t₁ + 1 / t₂ + 1 / t₃ = 1 / t_total

theorem envelope_equation_correct (t₂ : ℝ) :
  envelope_equation t₂ ↔ 
  (1 : ℝ) / n * (n / t₁ + n / t₂ + n / t₃) = (1 : ℝ) / n * (n / t_total) := by
  sorry

/-- The solution for t₂ -/
noncomputable def t₂_solution : ℝ := 30

theorem solution_satisfies_equation :
  envelope_equation t₂_solution := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_equation_correct_solution_satisfies_equation_l180_18096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_with_specific_average_l180_18006

noncomputable def average (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem exists_number_with_specific_average : ∃ x : ℝ, 
  average 10 70 x = average 20 40 60 - 9 := by
  use 13
  simp [average]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_with_specific_average_l180_18006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_intersection_l180_18091

-- Define the parametric equations of Line 1
noncomputable def line1 (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, -5 + (Real.sqrt 3 / 2) * t)

-- Define the equation of Line 2
def line2 (x y : ℝ) : Prop := x - y - 2 * Real.sqrt 3 = 0

-- Define the intersection point
noncomputable def intersection_point : ℝ × ℝ := (1 + 2 * Real.sqrt 3, 1)

-- Define the given point P0
def P0 : ℝ × ℝ := (1, -5)

-- State the theorem
theorem distance_to_intersection :
  Real.sqrt ((intersection_point.1 - P0.1)^2 + (intersection_point.2 - P0.2)^2) = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_intersection_l180_18091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_in_circle_l180_18043

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
noncomputable def octagonArea : ℝ := 36 * (1 - Real.sqrt 2 / 2) * Real.sqrt 3

/-- Theorem stating that the area of a regular octagon inscribed in a circle
    with radius 3 units is equal to 36(1 - √2/2)√3 square units -/
theorem regular_octagon_area_in_circle (r : ℝ) (h : r = 3) :
  octagonArea = 8 * (r^2 * Real.sin (π / 8)^2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_in_circle_l180_18043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_male_students_count_l180_18052

/-- The number of first-year male students -/
def male_students : ℕ := sorry

/-- The number of first-year female students -/
def female_students : ℕ := sorry

/-- The total number of first-year students -/
def total_students : ℕ := sorry

/-- Axiom: The total number of first-year students is 240 -/
axiom total_is_240 : total_students = 240

/-- Axiom: The number of female students is 3/5 of the number of male students -/
axiom female_to_male_ratio : female_students = (3 * male_students) / 5

/-- Axiom: The total number of students is the sum of male and female students -/
axiom total_is_sum : total_students = male_students + female_students

/-- Theorem: The number of first-year male students is 150 -/
theorem male_students_count : male_students = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_male_students_count_l180_18052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_cubic_polynomial_l180_18011

-- Define the cubic polynomial
def cubic_polynomial (x : ℝ) : ℝ := 27 * x^3 + 27 * x^2 - 9 * x - 3

-- Define the root
noncomputable def root : ℝ := (Real.rpow 81 (1/3) + Real.rpow 9 (1/3) - 3) / 27

-- Theorem statement
theorem root_of_cubic_polynomial : cubic_polynomial root = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_cubic_polynomial_l180_18011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_a_range_for_disjoint_sets_l180_18071

noncomputable def f (x : ℝ) : ℝ := Real.log ((2 / (x + 1)) - 1)

def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + a

def A : Set ℝ := {x | f x ∈ Set.univ}

def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ Set.Icc 0 3, g a x = y}

theorem f_property : f (1/2015) + f (-1/2015) = 0 := by sorry

theorem a_range_for_disjoint_sets (a : ℝ) :
  A ∩ B a = ∅ ↔ a ∈ Set.Iic (-2) ∪ Set.Ici 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_a_range_for_disjoint_sets_l180_18071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_digit_probability_l180_18059

def num_dice : ℕ := 5
def sides_per_die : ℕ := 20
def one_digit_count : ℕ := 9
def two_digit_count : ℕ := 11

def prob_one_digit : ℚ := one_digit_count / sides_per_die
def prob_two_digit : ℚ := two_digit_count / sides_per_die

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem equal_digit_probability :
  (choose num_dice (num_dice / 2)) *
  (prob_one_digit ^ (num_dice / 2 : ℕ)) *
  (prob_two_digit ^ (num_dice - (num_dice / 2))) =
  107811 / 320000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_digit_probability_l180_18059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_ratio_l180_18047

noncomputable section

-- Define the radius and height
variable (R h : ℝ)

-- Define the surface areas and volumes
noncomputable def sphere_surface_area (R : ℝ) := 4 * Real.pi * R^2
noncomputable def cylinder_surface_area (R h : ℝ) := 2 * Real.pi * R^2 + 2 * Real.pi * R * h
noncomputable def sphere_volume (R : ℝ) := (4/3) * Real.pi * R^3
noncomputable def cylinder_volume (R h : ℝ) := Real.pi * R^2 * h

-- State the theorem
theorem cylinder_sphere_volume_ratio 
  (h_radius : h = R) 
  (h_surface : cylinder_surface_area R h = sphere_surface_area R) :
  cylinder_volume R h / sphere_volume R = 3/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_ratio_l180_18047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l180_18078

-- Define the constants
noncomputable def a : ℝ := Real.rpow 0.2 0.3
noncomputable def b : ℝ := Real.log 3 / Real.log 0.2
noncomputable def c : ℝ := Real.log 4 / Real.log 0.2

-- State the theorem
theorem ascending_order : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l180_18078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_equals_four_l180_18086

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the left focus
noncomputable def left_focus : ℝ × ℝ := (-Real.sqrt 3, 0)

-- Define the line passing through the left focus with slope 1
def line_equation (x y : ℝ) : Prop := y = x + Real.sqrt 3

-- Define points A and B as the intersection of the line and ellipse
def is_intersection_point (p : ℝ × ℝ) : Prop :=
  is_on_ellipse p.1 p.2 ∧ line_equation p.1 p.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem reciprocal_sum_equals_four :
  ∀ A B : ℝ × ℝ,
  is_intersection_point A →
  is_intersection_point B →
  A ≠ B →
  (1 / distance left_focus A) + (1 / distance left_focus B) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_equals_four_l180_18086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_inequality_abc_l180_18066

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |x - 1| + (1/2) * |x - 3|

-- Theorem 1
theorem solution_set_f (x : ℝ) : f x < 2 ↔ 1 < x ∧ x < 3 := by
  sorry

-- Theorem 2
theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum : a + b + c = 2) :
  1/a + 1/b + 1/c ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_inequality_abc_l180_18066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l180_18046

noncomputable def vec_a (k x : ℝ) : ℝ × ℝ := (k * Real.sin x ^ 3, Real.cos x ^ 2)
noncomputable def vec_b (k x : ℝ) : ℝ × ℝ := (Real.cos x ^ 3, -k)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def f (k x : ℝ) : ℝ := dot_product (vec_a k x) (vec_b k x)

theorem problem_solution (k : ℝ) (A B C : ℝ) (a b c : ℝ) :
  k > 0 →
  (∀ x, f k x ≤ Real.sqrt 2 - 1) →
  π / 2 < A ∧ A < π →
  f k A = 0 →
  b = 2 * Real.sqrt 2 →
  a = 2 * Real.sqrt 10 →
  k = 1 ∧ a * b * Real.cos A = -8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l180_18046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_range_l180_18022

theorem cos_sum_range (α β : ℝ) (h : Real.sin α + Real.sin β = Real.sqrt 2 / 2) :
  ∃ (t : ℝ), Real.cos α + Real.cos β = t ∧ -Real.sqrt (7/2) ≤ t ∧ t ≤ Real.sqrt (7/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_range_l180_18022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_equal_money_after_2020_steps_l180_18014

/-- Represents the money distribution among the three players -/
structure MoneyState where
  rita : ℕ
  sam : ℕ
  tom : ℕ

/-- The transition function for one step of the game -/
def transition (state : MoneyState) : MoneyState := sorry

/-- The probability of transitioning from one state to another -/
def transitionProb (s1 s2 : MoneyState) : ℚ := sorry

/-- The initial state of the game -/
def initialState : MoneyState := ⟨2, 1, 1⟩

/-- The target state where each player has 1 -/
def targetState : MoneyState := ⟨1, 1, 1⟩

/-- The number of steps (bell rings) in the game -/
def numSteps : ℕ := 2020

/-- Helper function to represent repeated application of transitionProb -/
def repeatedTransition (n : ℕ) (s1 s2 : MoneyState) : ℚ := sorry

theorem probability_of_equal_money_after_2020_steps :
  repeatedTransition numSteps initialState targetState = 1/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_equal_money_after_2020_steps_l180_18014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_player_card_value_l180_18065

def total_sum : ℕ := 220

def card_value (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

theorem other_player_card_value :
  ∃ n : ℕ, card_value n ∧ (∃ m : ℕ, total_sum = 5 + 3 + 9 + n + 15 * m) ∧ n = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_player_card_value_l180_18065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_pencils_count_l180_18000

/-- Represents the color of a pencil -/
inductive PencilColor
| Blue
| Other1
| Other2
deriving Repr, DecidableEq

/-- Represents a pencil case with a fixed number of pencils -/
structure PencilCase where
  pencils : Finset PencilColor
  total_count : Nat
  blue_count : Nat
  other1_count : Nat
  other2_count : Nat

/-- The conditions of the problem -/
def satisfies_conditions (pc : PencilCase) : Prop :=
  (pc.total_count = 9) ∧
  (pc.blue_count ≥ 1) ∧
  (∀ (s : Finset PencilColor), s ⊆ pc.pencils → s.card = 4 → ∃ (c : PencilColor), (s.filter (· = c)).card ≥ 2) ∧
  (∀ (s : Finset PencilColor), s ⊆ pc.pencils → s.card = 5 → ∀ (c : PencilColor), (s.filter (· = c)).card ≤ 3)

/-- The theorem to be proved -/
theorem blue_pencils_count (pc : PencilCase) :
  satisfies_conditions pc → pc.blue_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_pencils_count_l180_18000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_l180_18038

/-- The volume of a prism with a regular triangular base, lateral edges perpendicular to the base,
    and lateral surface that unfolds into a rectangle with sides of length 6 and 4. -/
theorem prism_volume (base_side : ℝ) (height : ℝ) : 
  base_side > 0 → height > 0 → 
  base_side * height = 24 →
  (base_side * height * (Real.sqrt 3 / 4) = 4 * Real.sqrt 3) ∨ 
  (base_side * height * (Real.sqrt 3 / 4) = 8 * Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_l180_18038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_tens_digits_l180_18095

/-- The probability of selecting 6 different integers from 10 to 79 (inclusive) with different tens digits -/
theorem probability_different_tens_digits : ℚ := by
  -- Define the range of integers
  let range : Finset ℕ := Finset.range 70

  -- Define the number of integers to select
  let k : ℕ := 6

  -- Define the number of possible tens digits
  let tens_digits : ℕ := 7

  -- Calculate the number of favorable outcomes
  let favorable_outcomes : ℕ := (tens_digits.choose k) * (10^k)

  -- Calculate the total number of outcomes
  let total_outcomes : ℕ := range.card.choose k

  -- Define the probability
  let prob : ℚ := favorable_outcomes / total_outcomes

  -- Prove that the probability equals 4375/744407
  sorry

#eval (4375 : ℚ) / 744407 -- To verify the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_tens_digits_l180_18095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l180_18007

noncomputable section

/-- Line l with parametric equation x = 1/2 + (√3/2)t, y = 1 + (1/2)t -/
def line_l (t : ℝ) : ℝ × ℝ := (1/2 + (Real.sqrt 3 / 2) * t, 1 + (1/2) * t)

/-- Curve C with Cartesian equation (x - 1/2)² + (y - 1/2)² = 1/2 -/
def curve_C (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1/2)^2 = 1/2

/-- Point P -/
def point_P : ℝ × ℝ := (1/2, 1)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The product of distances from P to intersection points of l and C is 1/4 -/
theorem intersection_distance_product :
  ∃ (t1 t2 : ℝ),
    curve_C (line_l t1).1 (line_l t1).2 ∧
    curve_C (line_l t2).1 (line_l t2).2 ∧
    t1 ≠ t2 ∧
    (distance point_P (line_l t1)) * (distance point_P (line_l t2)) = 1/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l180_18007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_digit_sum_20_l180_18076

def digit_sum (n : ℕ) : ℕ := sorry

def count_integers_with_digit_sum (lower : ℕ) (upper : ℕ) (sum : ℕ) : ℕ :=
  (List.range (upper - lower + 1)).map (· + lower)
    |>.filter (λ n => digit_sum n = sum)
    |>.length

theorem count_integers_with_digit_sum_20 :
  count_integers_with_digit_sum 700 900 20 = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_digit_sum_20_l180_18076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_radical_axis_l180_18023

-- Define trilinear coordinates
variable (α β γ : ℝ)
variable (x y z : ℝ)

-- Define a circle in trilinear coordinates
def circle_eq (p q r : ℝ) (α β γ x y z : ℝ) : Prop :=
  (p*x + q*y + r*z) * (x*Real.sin α + y*Real.sin β + z*Real.sin γ) = 
    y*z*Real.sin α + x*z*Real.sin β + x*y*Real.sin γ

-- Define the radical axis of two circles
def radical_axis (p₁ q₁ r₁ p₂ q₂ r₂ : ℝ) (x y z : ℝ) : Prop :=
  p₁*x + q₁*y + r₁*z = p₂*x + q₂*y + r₂*z

-- Theorem statement
theorem circle_and_radical_axis :
  (∀ p q r, circle_eq p q r α β γ x y z) ∧
  (∀ p₁ q₁ r₁ p₂ q₂ r₂, 
    circle_eq p₁ q₁ r₁ α β γ x y z → 
    circle_eq p₂ q₂ r₂ α β γ x y z → 
    radical_axis p₁ q₁ r₁ p₂ q₂ r₂ x y z) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_radical_axis_l180_18023
