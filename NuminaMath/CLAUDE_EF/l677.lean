import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l677_67749

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 2 - x) * Real.cos x + Real.sqrt 3 * (Real.sin x) ^ 2

def is_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem f_properties :
  (∃ T > 0, is_period T f ∧ ∀ S, 0 < S ∧ S < T → ¬ is_period S f) ∧
  (∀ k : ℤ, is_monotone_decreasing_on f (k * Real.pi + 5 * Real.pi / 12) (k * Real.pi + 11 * Real.pi / 12)) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l677_67749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_l677_67722

theorem quadratic_coefficient (s : ℤ) :
  (∃ (x y : ℤ), x ≠ y ∧ x^2 + s*x + 72 = 0 ∧ y^2 + s*y + 72 = 0) →
  (∃ (t : Finset ℤ), t.card = 12 ∧ s ∈ t) →
  (∃ (a : ℤ), ∀ (x : ℤ), a*x^2 + s*x + 72 = 0 ↔ x^2 + s*x + 72 = 0) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_l677_67722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_sufficient_nor_necessary_condition_l677_67781

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * a 1 + (n * (n - 1) : ℝ) / 2 * (a 2 - a 1)

def is_increasing_sequence (S : ℕ → ℝ) : Prop :=
  ∀ n, S (n + 1) > S n

theorem not_sufficient_nor_necessary_condition
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_sum : ∀ n, S n = sum_of_arithmetic_sequence a n) :
  ¬(d > 0 → is_increasing_sequence S) ∧
  ¬(is_increasing_sequence S → d > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_sufficient_nor_necessary_condition_l677_67781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_painting_time_l677_67724

noncomputable def two_people_time (rate : ℝ) : ℝ := 20 / rate

theorem house_painting_time (people_rate : ℕ → ℝ) 
  (h1 : people_rate 5 * 4 = people_rate 2 * 10) 
  (h2 : ∀ n : ℕ, n > 0 → people_rate n > 0) : 
  two_people_time (people_rate 2) = 10 :=
by
  -- Proof steps go here
  sorry

#check house_painting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_painting_time_l677_67724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_of_first_thousand_l677_67726

/-- Represents a sequence of digits formed by writing positive integers
    starting with 1 in increasing order -/
def digitSequence : ℕ → ℕ := sorry

/-- The number of digits in the first n positive integers starting with 1 -/
def digitCount (n : ℕ) : ℕ := sorry

/-- Returns the nth digit in the sequence -/
def nthDigit (n : ℕ) : ℕ :=
  digitSequence n

theorem last_three_digits_of_first_thousand :
  (nthDigit 998, nthDigit 999, nthDigit 1000) = (1, 1, 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_of_first_thousand_l677_67726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_60_l677_67739

/-- The set of positive integer solutions less than or equal to 30 
    to the congruence 7(5x-3) ≡ 35 (mod 10) -/
def solution_set : Finset ℕ :=
  Finset.filter (fun x => x ≤ 30 ∧ (7 * (5 * x - 3)) % 10 = 35 % 10) (Finset.range 31)

/-- The sum of all elements in the solution set -/
def sum_of_solutions : ℕ := solution_set.sum id

theorem sum_of_solutions_is_60 : sum_of_solutions = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_60_l677_67739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_adjacent_part_theorem_l677_67709

/-- RegularQuadrilateralPyramid represents a regular quadrilateral pyramid. -/
structure RegularQuadrilateralPyramid where
  base_side : ℝ
  lateral_angle : ℝ

/-- Defines the volume of a part of the pyramid adjacent to the vertex. -/
noncomputable def volume_adjacent_part (pyramid : RegularQuadrilateralPyramid) : ℝ :=
  (Real.sqrt 6 * pyramid.base_side ^ 3) / 18

/-- Theorem stating the volume of the part of the pyramid adjacent to the vertex. -/
theorem volume_adjacent_part_theorem (pyramid : RegularQuadrilateralPyramid) 
  (h1 : pyramid.base_side > 0)
  (h2 : pyramid.lateral_angle = π / 6) : 
  volume_adjacent_part pyramid = (Real.sqrt 6 * pyramid.base_side ^ 3) / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_adjacent_part_theorem_l677_67709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_rain_is_three_l677_67772

/-- The amount of water in gallons collected per inch of rain -/
noncomputable def gallons_per_inch : ℝ := 15

/-- The amount of rain in inches collected on Monday -/
noncomputable def monday_rain : ℝ := 4

/-- The price in dollars per gallon of water -/
noncomputable def price_per_gallon : ℝ := 1.2

/-- The total amount of money made from selling all the water -/
noncomputable def total_money : ℝ := 126

/-- The amount of rain in inches collected on Tuesday -/
noncomputable def tuesday_rain : ℝ := (total_money - monday_rain * gallons_per_inch * price_per_gallon) / (gallons_per_inch * price_per_gallon)

theorem tuesday_rain_is_three : tuesday_rain = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_rain_is_three_l677_67772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_three_consecutive_multiples_l677_67707

theorem largest_of_three_consecutive_multiples (n : ℕ) :
  (3 * n + 3 * (n + 1) + 3 * (n + 2) = 72) →
  max (3 * n) (max (3 * (n + 1)) (3 * (n + 2))) = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_three_consecutive_multiples_l677_67707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_properties_l677_67753

-- Define the function f
noncomputable def f (x : ℝ) := Real.sin x * Real.sin (Real.pi / 2 + x) + Real.cos x ^ 2

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

theorem function_and_triangle_properties :
  -- Part 1: Maximum value of f
  (∀ x, f x ≤ (Real.sqrt 2 + 1) / 2) ∧
  (∃ x, f x = (Real.sqrt 2 + 1) / 2) ∧
  -- Part 2: Triangle properties
  ∀ ABC : Triangle,
    f ABC.A = 1 →
    ABC.A + ABC.B = 7 * Real.pi / 12 →
    ABC.b = Real.sqrt 6 →
    ABC.A = Real.pi / 4 ∧
    ABC.a = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_properties_l677_67753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_trig_ratio_l677_67775

theorem perpendicular_line_trig_ratio (α : Real) :
  -- Line l with inclination angle α is perpendicular to x + 2y - 3 = 0
  (Real.tan α = 2) →
  -- Prove that (sin α - cos α) / (sin α + cos α) = 1/3
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_trig_ratio_l677_67775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_parabola_chord_constant_l677_67777

/-- A parabola in the xy-plane defined by y = x^2 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2}

/-- The point C on the y-axis -/
def C (d : ℝ) : ℝ × ℝ := (0, d)

/-- The distance between two points in the plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The constant t defined for a chord AB passing through C -/
noncomputable def t (A B : ℝ × ℝ) (d : ℝ) : ℝ :=
  1 / distance A (C d) + 1 / distance B (C d)

/-- The theorem stating that t is constant and equal to 4 -/
theorem parabola_chord_theorem (d : ℝ) :
  ∃ (t_const : ℝ), ∀ (A B : ℝ × ℝ),
    A ∈ Parabola → B ∈ Parabola →
    (∃ (m : ℝ), A.2 = m * A.1 + d ∧ B.2 = m * B.1 + d) →
    t A B d = t_const := by
  sorry

/-- The specific value of the constant t -/
theorem parabola_chord_constant : 
  ∃ (d : ℝ), ∃ (t_const : ℝ), (∀ (A B : ℝ × ℝ),
    A ∈ Parabola → B ∈ Parabola →
    (∃ (m : ℝ), A.2 = m * A.1 + d ∧ B.2 = m * B.1 + d) →
    t A B d = t_const) ∧ t_const = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_parabola_chord_constant_l677_67777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l677_67792

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point M
def M : ℝ × ℝ := (2, 4)

-- Define the ellipse T
def ellipse_T (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1 ∧ a > b ∧ b > 0

-- Define line l
def line_l (x y k : ℝ) : Prop := y = k*x + Real.sqrt 3 ∧ k > 0

-- Helper function for triangle area
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem ellipse_and_triangle_area :
  ∃ (a b : ℝ), 
    (∀ x y, ellipse_T x y a b ↔ x^2/4 + y^2 = 1) ∧
    (∃ (P Q : ℝ × ℝ) (k : ℝ),
      line_l P.1 P.2 k ∧
      line_l Q.1 Q.2 k ∧
      ellipse_T P.1 P.2 a b ∧
      ellipse_T Q.1 Q.2 a b ∧
      (∀ O P' Q' : ℝ × ℝ,
        line_l P'.1 P'.2 k →
        line_l Q'.1 Q'.2 k →
        ellipse_T P'.1 P'.2 a b →
        ellipse_T Q'.1 Q'.2 a b →
        O = (0, 0) →
        area_triangle O P' Q' ≤ 1) ∧
      area_triangle (0, 0) P Q = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l677_67792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_ellipse_l677_67794

/-- The minimum distance between a point on a circle centered at (0,0) with radius 2
    and a point on an ellipse centered at (2,0) with semi-major and semi-minor axes
    both equal to 3 is (17 - 6√5) / (3√5) -/
theorem min_distance_circle_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let ellipse := {p : ℝ × ℝ | (p.1 - 2)^2 / 9 + p.2^2 / 9 = 1}
  (⨅ (a ∈ circle) (b ∈ ellipse), Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)) =
    (17 - 6 * Real.sqrt 5) / (3 * Real.sqrt 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_ellipse_l677_67794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_div_c_le_n_plus_one_l677_67713

-- Define polynomials f and g
noncomputable def f (n : ℕ) : Polynomial ℝ := sorry
noncomputable def g (n : ℕ) : Polynomial ℝ := sorry

-- Define the relationship between f and g
axiom g_eq_xr_mul_f (n : ℕ) (r : ℝ) : g n = (Polynomial.X + Polynomial.C r) * f n

-- Define a and c
noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def c (n : ℕ) : ℝ := sorry

-- Axioms for a and c being the maximum absolute values of coefficients
axiom a_is_max (n : ℕ) : ∀ i ≤ n, |Polynomial.coeff (f n) i| ≤ a n
axiom c_is_max (n : ℕ) : ∀ i ≤ n + 1, |Polynomial.coeff (g n) i| ≤ c n

-- Axiom for f and g being non-zero polynomials
axiom f_nonzero (n : ℕ) : f n ≠ 0
axiom g_nonzero (n : ℕ) : g n ≠ 0

-- The theorem to prove
theorem a_div_c_le_n_plus_one (n : ℕ) : (a n) / (c n) ≤ n + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_div_c_le_n_plus_one_l677_67713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_impossible_side_length_l677_67762

-- Define Triangle as a structure
structure Triangle where
  sides : Finset ℝ
  side_count : sides.card = 3
  positive_sides : ∀ s ∈ sides, s > 0

-- Triangle inequality theorem
theorem triangle_inequality {a b c : ℝ} (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  ∃ (t : Triangle), t.sides = {a, b, c} := by
  sorry

-- Theorem proving that a triangle with sides 4, 5, and 10 is impossible
theorem impossible_side_length : ¬∃ (t : Triangle), t.sides = {4, 5, 10} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_impossible_side_length_l677_67762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_decreasing_interval_l677_67727

-- Define the function f(x) = x^2 - ln(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the derivative of f(x)
noncomputable def f_deriv (x : ℝ) : ℝ := (2 * x^2 - 1) / x

-- Theorem statement
theorem monotonically_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x ≤ Real.sqrt 2 / 2 ↔ f_deriv x ≤ 0 :=
by
  sorry

-- Additional theorem to state the interval explicitly
theorem decreasing_interval :
  Set.Ioo (0 : ℝ) (Real.sqrt 2 / 2) ⊆ {x | f_deriv x ≤ 0} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_decreasing_interval_l677_67727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_proof_l677_67712

/-- The distance between the foci of an ellipse given by the equation 9x^2 - 36x + 4y^2 + 8y + 16 = 0 -/
noncomputable def ellipse_foci_distance : ℝ :=
  2 * Real.sqrt 30 / 3

/-- Theorem stating that the distance between the foci of the given ellipse is 2√(30)/3 -/
theorem ellipse_foci_distance_proof (x y : ℝ) :
  9 * x^2 - 36 * x + 4 * y^2 + 8 * y + 16 = 0 →
  ∃ (f1 f2 : ℝ × ℝ), 
    (f1.1 - f2.1)^2 + (f1.2 - f2.2)^2 = ellipse_foci_distance^2 :=
by
  sorry

#check ellipse_foci_distance
#check ellipse_foci_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_proof_l677_67712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_function_is_negative_inverse_of_negative_l677_67756

-- Define φ as a function from ℝ to ℝ
variable (φ : ℝ → ℝ)

-- Define the inverse of φ
noncomputable def φ_inv (φ : ℝ → ℝ) : ℝ → ℝ := Function.invFun φ

-- Define the third function f
variable (f : ℝ → ℝ)

-- Define the symmetry condition
def symmetric_to_inverse (f : ℝ → ℝ) (φ_inv : ℝ → ℝ) : Prop :=
  ∀ x y, φ_inv x = y ↔ f (-y) = -x

-- Theorem statement
theorem third_function_is_negative_inverse_of_negative (h : symmetric_to_inverse f (φ_inv φ)) :
  ∀ x, f x = -(φ_inv φ (-x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_function_is_negative_inverse_of_negative_l677_67756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l677_67729

def projection_problem (a b : ℝ × ℝ) : Prop :=
  let angle_ab := Real.arccos (-1/2)  -- 120° in radians
  let magnitude_a := 2
  let magnitude_b := 3
  let v1 := (2 * a.1 + 3 * b.1, 2 * a.2 + 3 * b.2)
  let v2 := (2 * a.1 + b.1, 2 * a.2 + b.2)
  let dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2
  let magnitude (v : ℝ × ℝ) := Real.sqrt (dot_product v v)
  let projection := (dot_product v1 v2) / (magnitude v2)
  (magnitude a = magnitude_a) ∧ 
  (magnitude b = magnitude_b) ∧
  (Real.arccos ((dot_product a b) / (magnitude a * magnitude b)) = angle_ab) →
  projection = 19 * Real.sqrt 13 / 13

theorem projection_theorem (a b : ℝ × ℝ) : projection_problem a b := by
  sorry

#check projection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l677_67729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_starters_l677_67795

def total_players : ℕ := 18
def num_quadruplets : ℕ := 4
def num_starters : ℕ := 8

theorem soccer_team_starters :
  (Finset.univ.filter (λ lineup : Finset (Fin total_players) =>
    lineup.card = num_starters ∧
    (lineup.filter (λ i => i.val < num_quadruplets)).card < num_quadruplets
  )).card = 42757 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_starters_l677_67795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sequence_520_to_523_l677_67735

/-- A function representing the arrow sequence in a path with a repeating pattern every 5 points -/
def arrowSequence (n : ℕ) : ℕ :=
  n % 5

/-- The sequence of arrows from one point to another -/
def sequenceBetween (start finish : ℕ) : List ℕ :=
  List.range (finish - start + 1) |>.map (fun i => arrowSequence (start + i))

theorem correct_sequence_520_to_523 :
  sequenceBetween 520 523 = [0, 1, 2, 3] := by
  sorry

#eval sequenceBetween 520 523

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sequence_520_to_523_l677_67735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_at_most_one_obtuse_angle_l677_67743

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  sum_angles : angle_a + angle_b + angle_c = Real.pi
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define an obtuse angle
def isObtuseAngle (angle : ℝ) : Prop := angle > Real.pi / 2

-- Theorem: A triangle cannot have two or more obtuse angles
theorem triangle_at_most_one_obtuse_angle (t : Triangle) : 
  ¬(isObtuseAngle t.angle_a ∧ isObtuseAngle t.angle_b) ∧
  ¬(isObtuseAngle t.angle_b ∧ isObtuseAngle t.angle_c) ∧
  ¬(isObtuseAngle t.angle_a ∧ isObtuseAngle t.angle_c) := by
  sorry

#check triangle_at_most_one_obtuse_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_at_most_one_obtuse_angle_l677_67743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_conservation_mode_l677_67733

def water_usage : List ℝ := [0.20, 0.25, 0.3, 0.4, 0.50]
def frequencies : List ℕ := [2, 4, 4, 8, 2]

def mode (data : List ℝ) (freq : List ℕ) : ℝ :=
  let paired_data := data.zip freq
  let max_freq := freq.maximum?
  (paired_data.filter (λ p => p.2 = max_freq.getD 0)).head?.map Prod.fst |>.getD 0

theorem water_conservation_mode :
  mode water_usage frequencies = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_conservation_mode_l677_67733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_larger_than_octagon_l677_67700

noncomputable section

-- Define the side length
def side_length : ℝ := 3

-- Define the number of sides for pentagon and octagon
def pentagon_sides : ℕ := 5
def octagon_sides : ℕ := 8

-- Define the central angle for a regular polygon
noncomputable def central_angle (n : ℕ) : ℝ := 2 * Real.pi / n

-- Define the apothem of a regular polygon
noncomputable def apothem (n : ℕ) (s : ℝ) : ℝ := s / (2 * Real.tan (central_angle n / 2))

-- Define the circumradius of a regular polygon
noncomputable def circumradius (n : ℕ) (s : ℝ) : ℝ := s / (2 * Real.sin (central_angle n / 2))

-- Define the area between inscribed and circumscribed circles
noncomputable def area_between_circles (n : ℕ) (s : ℝ) : ℝ :=
  Real.pi * ((circumradius n s)^2 - (apothem n s)^2)

-- Theorem statement
theorem pentagon_area_larger_than_octagon :
  area_between_circles pentagon_sides side_length > area_between_circles octagon_sides side_length := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_larger_than_octagon_l677_67700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dry_person_exists_l677_67798

/-- A directed graph where each vertex has exactly one outgoing edge -/
structure OneOutGraph (α : Type) where
  vertices : Set α
  edge : α → α
  edge_in_vertices : ∀ v ∈ vertices, edge v ∈ vertices

/-- There exists a dry person in a OneOutGraph with an odd number of vertices -/
theorem dry_person_exists {α : Type} [Fintype α] (G : OneOutGraph α) (h_odd : Odd (Fintype.card α)) :
  ∃ v ∈ G.vertices, ∀ u ∈ G.vertices, G.edge u ≠ v :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dry_person_exists_l677_67798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiders_cannot_catch_fly_l677_67778

structure Cube where
  side_length : ℝ
  side_length_pos : side_length > 0

structure Spider where
  max_speed : ℝ
  max_speed_pos : max_speed > 0

structure Fly where
  max_speed : ℝ
  max_speed_pos : max_speed > 0

noncomputable def diagonal_length (c : Cube) : ℝ := c.side_length * Real.sqrt 3

theorem spiders_cannot_catch_fly (c : Cube) (s : Spider) (f : Fly) 
  (h_speed : f.max_speed = 3 * s.max_speed) :
  ¬∃ (t : ℝ), t ≥ 0 ∧ 
  (s.max_speed * t ≥ 3 * c.side_length ∧ f.max_speed * t ≤ diagonal_length c) := by
  sorry

#check spiders_cannot_catch_fly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiders_cannot_catch_fly_l677_67778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_juice_volume_is_pi_over_four_l677_67780

/-- Represents a cylindrical glass with lemonade -/
structure LemonadeGlass where
  height : ℝ
  diameter : ℝ
  fullness : ℝ
  lemonJuiceRatio : ℝ

/-- Calculates the volume of lemon juice in the glass -/
noncomputable def lemonJuiceVolume (glass : LemonadeGlass) : ℝ :=
  let radius := glass.diameter / 2
  let liquidVolume := Real.pi * radius^2 * (glass.height * glass.fullness)
  liquidVolume * (glass.lemonJuiceRatio / (1 + glass.lemonJuiceRatio))

/-- Theorem stating that the volume of lemon juice in the specified glass is π/4 cubic inches -/
theorem lemon_juice_volume_is_pi_over_four :
  let glass : LemonadeGlass := {
    height := 6,
    diameter := 2,
    fullness := 1/2,
    lemonJuiceRatio := 1/11
  }
  lemonJuiceVolume glass = Real.pi/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_juice_volume_is_pi_over_four_l677_67780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_modulo_p_l677_67760

theorem function_identity_modulo_p (p : ℕ) (hp : Nat.Prime p) (hp2 : p > 2) 
  (f : ℤ → Fin p) 
  (h1 : ∀ n : ℤ, (f (f n) - f (n + 1) + 1 : ℤ) ≡ 0 [ZMOD p])
  (h2 : ∀ n : ℤ, f (n + p) = f n) :
  ∀ x : ℤ, f x = x.toNat % p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_modulo_p_l677_67760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l677_67771

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Triangle inequality
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b
  -- Angle sum property
  angle_sum : A + B + C = π
  -- Law of cosines
  cos_A : Real.cos A = (b^2 + c^2 - a^2) / (2*b*c)
  cos_B : Real.cos B = (a^2 + c^2 - b^2) / (2*a*c)
  cos_C : Real.cos C = (a^2 + b^2 - c^2) / (2*a*b)

-- Define the specific conditions for our triangle
def SpecialTriangle (t : Triangle) : Prop :=
  2 * t.a * Real.cos t.C = 2 * t.b - t.c ∧ t.b + t.c = 2

-- State the theorem
theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.A = π / 3 ∧ 1 ≤ t.a ∧ t.a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l677_67771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_max_value_l677_67710

noncomputable def f (a b x : ℝ) : ℝ := a^x + b * a^(-x)

theorem odd_function_max_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ b, ∀ x, f a b x = -f a b (-x)) →
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, f a (-1) x ≥ f a (-1) y) →
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a (-1) x = 8/3) →
  a = 3 ∨ a = 1/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_max_value_l677_67710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_l677_67768

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)*x^2 - 2*x + 5

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - x - 2

-- Theorem statement
theorem f_increasing_intervals :
  (∀ x, x < -2/3 → (f' x > 0)) ∧ (∀ x, x > 1 → (f' x > 0)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_l677_67768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cab_driver_average_income_l677_67774

noncomputable def daily_incomes : List ℚ := [600, 250, 450, 400, 800]

def num_days : ℕ := 5

noncomputable def average_income : ℚ := (daily_incomes.sum) / num_days

theorem cab_driver_average_income :
  average_income = 500 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cab_driver_average_income_l677_67774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l677_67715

-- Define the parabola C: y² = 8x
def C (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus F
def F : ℝ × ℝ := (2, 0)

-- Define point P
noncomputable def P (m : ℝ) : ℝ × ℝ := (Real.sqrt 3, m)

-- Theorem statement
theorem parabola_properties :
  ∀ m : ℝ, C (P m).1 (P m).2 →
  F = (2, 0) ∧
  Real.sqrt ((P m).1 - F.1)^2 + ((P m).2 - F.2)^2 = Real.sqrt 3 + 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l677_67715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_275_l677_67701

/-- Calculates the length of a bridge given train length, speed, and crossing time. -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

/-- The length of the bridge is 275 meters. -/
theorem bridge_length_is_275 :
  bridge_length 475 90 30 = 275 := by
  unfold bridge_length
  simp
  -- The actual proof would go here
  sorry

/-- Compute the bridge length using rational numbers for precise calculation -/
def bridge_length_rat (train_length : ℚ) (train_speed_kmh : ℚ) (crossing_time_s : ℚ) : ℚ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

#eval bridge_length_rat 475 90 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_275_l677_67701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_conditions_imply_m_range_l677_67711

/-- A quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*(m-1)*x - 5*m - 2

/-- The roots of the quadratic function -/
def roots (m : ℝ) : Set ℝ := {x | f m x = 0}

theorem root_conditions_imply_m_range (m : ℝ) :
  (∃ x₁ x₂, x₁ ∈ roots m ∧ x₂ ∈ roots m ∧ x₁ < 1 ∧ x₂ > 1) → m > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_conditions_imply_m_range_l677_67711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_points_exist_l677_67783

-- Define the circle
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

-- Define the diameter endpoints
def DiameterEndpoints : (ℝ × ℝ) × (ℝ × ℝ) := ((-2, 0), (2, 0))

-- Define the condition for points P
def SatisfiesCondition (p : ℝ × ℝ) : Prop :=
  let (a, b) := DiameterEndpoints
  (p.1 - a.1)^2 + (p.2 - a.2)^2 + (p.1 - b.1)^2 + (p.2 - b.2)^2 = 5

-- Theorem statement
theorem infinite_points_exist :
  ∃ (S : Set (ℝ × ℝ)), S ⊆ Circle ∧ (∀ p ∈ S, SatisfiesCondition p) ∧ Set.Infinite S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_points_exist_l677_67783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sellable_fruit_count_l677_67755

def orange_crates : ℕ := 30
def oranges_per_crate : ℕ := 300
def nectarine_boxes : ℕ := 45
def nectarines_per_box : ℕ := 80
def apple_baskets : ℕ := 20
def apples_per_basket : ℕ := 120
def orange_damage_percent : ℚ := 10 / 100
def customers_taking_nectarines : ℕ := 5
def nectarines_taken_per_customer : ℕ := 20
def bad_apples : ℕ := 50

def total_sellable_fruit : ℕ := 13950

theorem sellable_fruit_count :
  (orange_crates * oranges_per_crate) -
  Int.floor ((orange_crates * oranges_per_crate : ℚ) * orange_damage_percent) +
  (nectarine_boxes * nectarines_per_box) -
  (customers_taking_nectarines * nectarines_taken_per_customer) +
  (apple_baskets * apples_per_basket) - bad_apples = total_sellable_fruit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sellable_fruit_count_l677_67755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_relation_point_on_curve_sum_b_less_than_next_b_l677_67787

def sequence_a : ℕ+ → ℝ := fun n => 3 * (2 : ℝ)^(n.val - 1)

def sequence_b : ℕ+ → ℝ := fun n => (2 : ℝ)^(n.val - 1)

def sum_a (n : ℕ+) : ℝ := 3 * (2 : ℝ)^n.val - 3

def sum_b (n : ℕ+) : ℝ := (2 : ℝ)^n.val - 1

theorem sequence_relation (n : ℕ+) : 
  sequence_b n + sequence_b (n + 1) = sequence_a n := by sorry

theorem point_on_curve (n : ℕ+) : 
  sum_a n + 3 = 3 * (2 : ℝ)^n.val := by sorry

theorem sum_b_less_than_next_b (n : ℕ+) : 
  sum_b n < sequence_b (n + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_relation_point_on_curve_sum_b_less_than_next_b_l677_67787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_l677_67786

noncomputable def a (n : ℕ) (t : ℝ) : ℝ := -n + t

noncomputable def b (n : ℕ) : ℝ := 3^(n-3)

noncomputable def c (n : ℕ) (t : ℝ) : ℝ := (a n t + b n) / 2 + |a n t - b n| / 2

theorem t_range (t : ℝ) :
  (∀ n : ℕ, n ≥ 1 → c n t ≥ c 3 t) ↔ (10/3 < t ∧ t < 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_l677_67786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_relations_l677_67741

open Matrix

variable {α : Type*} [CommRing α]

variable {A B : Matrix (Fin 2) (Fin 2) α}

theorem matrix_determinant_relations (h : (A - B)^2 = 0) :
  (Matrix.det (A^2 - B^2) = (Matrix.det A - Matrix.det B)^2) ∧
  (Matrix.det (A * B - B * A) = 0 ↔ Matrix.det A = Matrix.det B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_relations_l677_67741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l677_67782

-- Define the triangle
noncomputable def Triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0

-- Define the right triangle condition
noncomputable def RightTriangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Define the ratio condition
noncomputable def SidesInRatio (a b c : ℝ) : Prop := ∃ (k : ℝ), a = 5*k ∧ b = 12*k ∧ c = 13*k

-- Define the area of the triangle
noncomputable def TriangleArea (a b : ℝ) : ℝ := (1/2) * a * b

-- Theorem statement
theorem triangle_perimeter (a b c : ℝ) 
  (h_triangle : Triangle a b c)
  (h_right : RightTriangle a b c)
  (h_ratio : SidesInRatio a b c)
  (h_area : TriangleArea a b = 3000) :
  a + b + c = 300 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l677_67782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l677_67740

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x - Real.pi / 3)

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

def is_shifted_right (f g : ℝ → ℝ) (shift : ℝ) : Prop :=
  ∀ x, g x = f (x - shift)

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_properties :
  (is_odd (fun x ↦ f (x + Real.pi / 6))) ∧
  ¬(is_shifted_right f (fun x ↦ 4 * Real.sin (2 * x)) (Real.pi / 3)) ∧
  (is_symmetric_about f (-Real.pi / 12)) ∧
  (is_monotone_increasing f 0 (5 * Real.pi / 12)) ∧
  ¬(is_odd f) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l677_67740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_sum_l677_67745

theorem cubic_root_equation_sum (a b c : ℕ+) :
  (3 * Real.sqrt (Real.rpow 5 (1/3) - Real.rpow 4 (1/3)) = Real.rpow a (1/3) + Real.rpow b (1/3) - Real.rpow c (1/3)) →
  (a : ℕ) + b + c = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_sum_l677_67745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_relation_l677_67796

theorem square_area_relation (c : ℝ) :
  let square_A := λ (d : ℝ) => (d / Real.sqrt 2) ^ 2
  let square_B := λ (a : ℝ) => 3 * a
  square_B (square_A (2 * c)) = 6 * c ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_relation_l677_67796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_50_cents_l677_67789

def box : Finset (ℕ × ℕ) := {(1, 2), (5, 4), (10, 6)}

def total_coins : ℕ := (box.sum (λ (_, count) => count))

def draw_coins (n : ℕ) : Finset (Finset (ℕ × ℕ)) :=
  Finset.powerset box |>.filter (λ s => s.sum (λ (_, count) => count) = n)

def value (s : Finset (ℕ × ℕ)) : ℕ :=
  s.sum (λ (value, count) => value * count)

def favorable_outcomes : Finset (Finset (ℕ × ℕ)) :=
  (draw_coins 6).filter (λ s => value s ≥ 50)

theorem probability_at_least_50_cents :
  (favorable_outcomes.card : ℚ) / (draw_coins 6).card = 127 / 924 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_50_cents_l677_67789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_example_l677_67717

/-- The volume of a cone given its radius and height -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The ratio of volumes of two cones -/
noncomputable def cone_volume_ratio (r_A h_A r_B h_B : ℝ) : ℝ :=
  (cone_volume r_A h_A) / (cone_volume r_B h_B)

theorem cone_volume_ratio_example :
  cone_volume_ratio 10 25 25 10 = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_example_l677_67717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l677_67793

namespace TriangleProof

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

/-- The radius of the circumcircle of a triangle -/
noncomputable def circumRadius (t : Triangle) : ℝ := t.a / (2 * Real.sin t.A)

/-- Theorem about the properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 5)
  (h2 : t.b = 6)
  (h3 : Real.cos t.B = -4/5)
  (h4 : triangleArea t = 15 * Real.sqrt 7 / 4) :
  t.A = π/6 ∧ 
  circumRadius t = 5 ∧ 
  (t.c = 4 ∨ t.c = Real.sqrt 106) := by sorry

end TriangleProof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l677_67793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_decreasing_digit_numbers_l677_67751

/-- A function that checks if a natural number has digits in strictly decreasing order -/
def hasDecreasingDigits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.repr.get! i).toNat > (n.repr.get! j).toNat

/-- The count of natural numbers with at least two digits and strictly decreasing digits -/
def countDecreasingDigitNumbers : ℕ :=
  (Finset.range 9).sum (fun k => Nat.choose 10 (k + 2)) + 1

/-- Theorem stating that the count of natural numbers with at least two digits 
    and strictly decreasing digits is 1013 -/
theorem count_decreasing_digit_numbers :
  countDecreasingDigitNumbers = 1013 := by
  sorry

#eval countDecreasingDigitNumbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_decreasing_digit_numbers_l677_67751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l677_67704

/-- An ellipse with given major axis length and eccentricity -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_major_axis : 2 * a = 6
  h_eccentricity : e = 1 / 3

/-- The equation of the ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- Point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : ellipse_equation E x y

/-- Focus of the ellipse -/
structure Focus (E : Ellipse) where
  x : ℝ
  y : ℝ

/-- Angle between three points -/
def angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Area of a triangle given three points -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the properties of the ellipse and the area of the triangle -/
theorem ellipse_properties (E : Ellipse) (P : PointOnEllipse E) (F₁ F₂ : Focus E) 
    (h_angle : angle (P.x, P.y) (F₁.x, F₁.y) (F₂.x, F₂.y) = 60 * π / 180) :
  (∀ x y, ellipse_equation E x y ↔ x^2 / 9 + y^2 / 8 = 1) ∧ 
  area_triangle (P.x, P.y) (F₁.x, F₁.y) (F₂.x, F₂.y) = 8 * Real.sqrt 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l677_67704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_midpoints_is_circle_l677_67779

-- Define the circle K
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point P inside the circle K
noncomputable def P (K : Circle) : ℝ × ℝ := 
  (K.center.1 + 2/3 * K.radius, K.center.2)

-- Define the distance from P to the center of K
noncomputable def distance_PO (K : Circle) : ℝ := 
  Real.sqrt ((P K).1 - K.center.1)^2 + ((P K).2 - K.center.2)^2

-- Define a chord of K
def Chord (K : Circle) := (ℝ × ℝ) × (ℝ × ℝ)

-- Define the midpoint of a chord
noncomputable def chord_midpoint (c : Chord K) : ℝ × ℝ := 
  ((c.1.1 + c.2.1) / 2, (c.1.2 + c.2.2) / 2)

-- Define the property of a chord subtending a right angle at P
def subtends_right_angle_at_P (K : Circle) (c : Chord K) : Prop := 
  let v1 := (c.1.1 - (P K).1, c.1.2 - (P K).2)
  let v2 := (c.2.1 - (P K).1, c.2.2 - (P K).2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Theorem statement
theorem locus_of_midpoints_is_circle (K : Circle) 
  (h : distance_PO K = (2/3) * K.radius) :
  ∃ C : Circle, ∀ c : Chord K, 
    subtends_right_angle_at_P K c → 
    chord_midpoint c ∈ {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_midpoints_is_circle_l677_67779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_transitive_unit_vector_direction_parallel_projection_magnitude_l677_67736

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : V) : Prop := ∃ (k : ℝ), v = k • w

/-- If a is parallel to b and b is parallel to c, then a is parallel to c -/
theorem vector_parallel_transitive 
  (a b c : V) (hab : parallel a b) (hbc : parallel b c) : parallel a c :=
sorry

/-- a/‖a‖ is a unit vector in the direction of a -/
theorem unit_vector_direction 
  (a : V) (ha : a ≠ 0) : ‖(1 / ‖a‖) • a‖ = 1 ∧ parallel ((1 / ‖a‖) • a) a :=
sorry

/-- If a is parallel to b, the magnitude of the projection of a onto b is |a| -/
theorem parallel_projection_magnitude 
  (a b : V) (hab : parallel a b) (hb : b ≠ 0) : 
  ‖(inner a b / ‖b‖^2) • b‖ = ‖a‖ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_transitive_unit_vector_direction_parallel_projection_magnitude_l677_67736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l677_67763

theorem trig_identity (x y z : ℝ) : 
  Real.cos (x + y) * Real.sin z + Real.sin (x + y) * Real.cos z = Real.sin (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l677_67763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_integral_of_differential_equation_l677_67799

/-- The general integral of the differential equation (x^2 + y^2) dx - xy dy = 0 -/
theorem general_integral_of_differential_equation
  (x y : ℝ → ℝ) (C : ℝ) (h : C ≠ 0) :
  (∀ t, x t = C * Real.exp ((y t)^2 / (2 * (x t)^2))) →
  (∀ t, ((x t)^2 + (y t)^2) * (deriv x t) - (x t) * (y t) * (deriv y t) = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_integral_of_differential_equation_l677_67799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_conversion_l677_67703

theorem map_distance_conversion (map_distance : ℝ) (actual_distance : ℝ) 
  (map_scale_inches : ℝ) (map_scale_km : ℝ) :
  map_scale_inches = 34 →
  map_scale_km = 14.916129032258064 →
  map_distance = 310 →
  actual_distance = (map_distance * map_scale_km) / map_scale_inches →
  abs (actual_distance - 136) < 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_conversion_l677_67703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_calculation_l677_67725

/-- Calculates the round trip time given the speed to work, speed back home, and time to work -/
noncomputable def roundTripTime (speedToWork : ℝ) (speedBackHome : ℝ) (timeToWork : ℝ) : ℝ :=
  let distanceToWork := speedToWork * timeToWork
  let timeBackHome := distanceToWork / speedBackHome
  timeToWork + timeBackHome

theorem round_trip_time_calculation :
  let speedToWork : ℝ := 75
  let speedBackHome : ℝ := 105
  let timeToWork : ℝ := 70 / 60
  roundTripTime speedToWork speedBackHome timeToWork = 2 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_calculation_l677_67725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_ratio_is_two_l677_67750

/-- Represents the financial situation of a person over two years -/
structure FinancialSituation where
  first_year_income : ℝ
  first_year_savings_rate : ℝ
  second_year_income_increase_rate : ℝ
  second_year_savings_increase_rate : ℝ

/-- Calculates the ratio of total expenditure in 2 years to expenditure in the 1st year -/
noncomputable def expenditure_ratio (fs : FinancialSituation) : ℝ :=
  let first_year_expenditure := fs.first_year_income * (1 - fs.first_year_savings_rate)
  let second_year_income := fs.first_year_income * (1 + fs.second_year_income_increase_rate)
  let second_year_savings := fs.first_year_income * fs.first_year_savings_rate * (1 + fs.second_year_savings_increase_rate)
  let second_year_expenditure := second_year_income - second_year_savings
  let total_expenditure := first_year_expenditure + second_year_expenditure
  total_expenditure / first_year_expenditure

/-- Theorem stating that under given conditions, the expenditure ratio is 2 -/
theorem expenditure_ratio_is_two (fs : FinancialSituation)
  (h1 : fs.first_year_savings_rate = 0.3)
  (h2 : fs.second_year_income_increase_rate = 0.3)
  (h3 : fs.second_year_savings_increase_rate = 1) :
  expenditure_ratio fs = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_ratio_is_two_l677_67750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l677_67721

theorem book_price_change (x : ℝ) (h : x > 0) : 
  1.30 * 0.85 * 1.05 * x = 1.16025 * x :=
by
  -- Arithmetic simplification
  calc
    1.30 * 0.85 * 1.05 * x = 1.16025 * x := by
      -- Perform the multiplication
      ring
      -- The equality holds by arithmetic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l677_67721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l677_67738

-- Define the points
def P : ℝ × ℝ := (-6, 4)
def Q : ℝ × ℝ := (0, 7)
def R : ℝ × ℝ := (3, -3)

-- Function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem area_of_triangle_PQR :
  triangleArea P Q R = 39 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l677_67738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_slope_angle_l677_67737

-- Define the curve
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (1 + 5 * Real.cos θ, 5 * Real.sin θ)

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the theorem
theorem chord_slope_angle 
  (h₁ : ∀ θ, 0 ≤ θ ∧ θ < 2 * Real.pi → ∃ (a b : ℝ), curve θ = (a, b))
  (h₂ : ∃ (θ₁ θ₂ : ℝ), 0 ≤ θ₁ ∧ θ₁ < 2 * Real.pi ∧ 
                        0 ≤ θ₂ ∧ θ₂ < 2 * Real.pi ∧ 
                        P.1 = ((curve θ₁).1 + (curve θ₂).1) / 2 ∧ 
                        P.2 = ((curve θ₁).2 + (curve θ₂).2) / 2) :
  ∃ (m : ℝ), Real.arctan m = Real.pi / 4 ∧ 
    ∀ (θ₁ θ₂ : ℝ), 0 ≤ θ₁ ∧ θ₁ < 2 * Real.pi ∧ 
                    0 ≤ θ₂ ∧ θ₂ < 2 * Real.pi ∧ 
                    P.1 = ((curve θ₁).1 + (curve θ₂).1) / 2 ∧ 
                    P.2 = ((curve θ₁).2 + (curve θ₂).2) / 2 → 
      m = ((curve θ₂).2 - (curve θ₁).2) / ((curve θ₂).1 - (curve θ₁).1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_slope_angle_l677_67737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_to_asymptote_distance_l677_67758

/-- The distance from the focus to the asymptote of the hyperbola x² - y²/3 = 1 is √3 -/
theorem hyperbola_focus_to_asymptote_distance :
  let hyperbola := {p : ℝ × ℝ | (p.1)^2 - (p.2)^2/3 = 1}
  let focus : ℝ × ℝ := (2, 0)
  let asymptote := {p : ℝ × ℝ | p.2 = Real.sqrt 3 * p.1 ∨ p.2 = -(Real.sqrt 3) * p.1}
  (∃ p ∈ asymptote, dist focus p = Real.sqrt 3) ∧
  (∀ q ∈ asymptote, dist focus q ≥ Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_to_asymptote_distance_l677_67758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_triangles_in_cube_l677_67747

/-- The side length of the cube -/
noncomputable def cube_side : ℝ := 2

/-- The number of vertices in the cube -/
def num_vertices : ℕ := 8

/-- The number of faces in the cube -/
def num_faces : ℕ := 6

/-- The area of a triangle within a single face of the cube -/
noncomputable def area_face_triangle : ℝ := (1 / 2) * cube_side * cube_side

/-- The number of triangles within a single face of the cube -/
def num_face_triangles_per_face : ℕ := 4

/-- The area of a triangle in a plane perpendicular to a face -/
noncomputable def area_perp_triangle : ℝ := (1 / 2) * cube_side * (Real.sqrt 8)

/-- The number of triangles in planes perpendicular to faces -/
def num_perp_triangles : ℕ := 24

/-- The area of a triangle with sides of three face diagonals -/
noncomputable def area_diagonal_triangle : ℝ := 4 * Real.sqrt 3

/-- The number of triangles with sides of three face diagonals -/
def num_diagonal_triangles : ℕ := 8

/-- The theorem stating the total area of all triangles in the cube -/
theorem total_area_triangles_in_cube :
  (num_faces * num_face_triangles_per_face * area_face_triangle) +
  (num_perp_triangles * area_perp_triangle) +
  (num_diagonal_triangles * area_diagonal_triangle) =
  48 + 48 * Real.sqrt 2 + 32 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_triangles_in_cube_l677_67747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_value_at_4_l677_67790

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the properties of h
def h_properties (h : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, h = λ x ↦ -(x - a^2) * (x - b^2) * (x - c^2)) ∧
  (h 0 = 1) ∧
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, h s = 0 ∧ s = r^2)

-- The theorem to prove
theorem h_value_at_4 (h : ℝ → ℝ) (hprops : h_properties h) : h 4 = -3599 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_value_at_4_l677_67790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_pqrs_is_two_l677_67705

/-- Represents a tetrahedron with vertices P, Q, R, S -/
structure Tetrahedron where
  pq : ℝ
  pr : ℝ
  ps : ℝ
  qr : ℝ
  qs : ℝ
  rs : ℝ

/-- Calculates the volume of a tetrahedron using the Cayley-Menger determinant -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  (1/6) * Real.sqrt (abs (
    Matrix.det ![
      ![0,  1,    1,    1,    1],
      ![1,  0,    t.pq^2, t.pr^2, t.ps^2],
      ![1,  t.pq^2, 0,    t.qr^2, t.qs^2],
      ![1,  t.pr^2, t.qr^2, 0,    t.rs^2],
      ![1,  t.ps^2, t.qs^2, t.rs^2, 0]
    ]
  ))

/-- The specific tetrahedron PQRS with given edge lengths -/
def pqrs : Tetrahedron :=
  { pq := 6
    pr := 4
    ps := 5
    qr := 5
    qs := 4
    rs := 3 }

/-- Theorem: The volume of tetrahedron PQRS is 2 -/
theorem volume_of_pqrs_is_two : tetrahedronVolume pqrs = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_pqrs_is_two_l677_67705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_property_for_small_positive_values_l677_67784

theorem log_property_for_small_positive_values (b w : ℝ) (h1 : b > 1) (h2 : 0 < w) (h3 : w < 1) :
  let z := Real.log w / Real.log b
  z < 0 ∧ ∀ ε > 0, ∃ δ > 0, ∀ w' > 0, w' < δ → z < -ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_property_for_small_positive_values_l677_67784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_from_same_pairwise_sums_l677_67716

/-- Two sequences of rational numbers with the same pairwise sums property -/
def SamePairwiseSums {n : ℕ} (a b : Fin n → ℚ) : Prop :=
  ∀ (i j k l : Fin n), i < j → k < l → 
    (∃ (p q : Fin n), p < q ∧ a i + a j = b p + b q) ∧
    (∃ (p q : Fin n), p < q ∧ b k + b l = a p + a q)

theorem power_of_two_from_same_pairwise_sums
  {n : ℕ} 
  (a b : Fin n → ℚ) 
  (h_neq : (Finset.univ : Finset (Fin n)).image a ≠ (Finset.univ : Finset (Fin n)).image b)
  (h_sums : SamePairwiseSums a b) :
  ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_from_same_pairwise_sums_l677_67716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_extrema_range_l677_67797

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

/-- Theorem stating the range of a for which f(x) has both a local maximum and minimum in (1/2, 3) -/
theorem f_local_extrema_range :
  ∀ a : ℝ, (∃ x y : ℝ, (1/2 < x ∧ x < 3) ∧ (1/2 < y ∧ y < 3) ∧ 
    (∀ z : ℝ, 1/2 < z ∧ z < 3 → f a z ≤ f a x) ∧
    (∀ w : ℝ, 1/2 < w ∧ w < 3 → f a y ≤ f a w) ∧ x ≠ y) ↔
  (2 < a ∧ a < 5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_extrema_range_l677_67797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_property_l677_67731

/-- Sum of the first n terms of a geometric series -/
def S (n : ℕ) : ℝ := sorry

/-- Theorem: If S_200 + 1 = (S_100 + 1)^2, then S_600 + 1 = (S_300 + 1)^2 -/
theorem geometric_series_sum_property (h : S 200 + 1 = (S 100 + 1)^2) :
  S 600 + 1 = (S 300 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_property_l677_67731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_approximation_l677_67759

/-- The base number whose 17th power, when cube-rooted, is closest to 50 -/
noncomputable def base_number : ℝ :=
  (125000 : ℝ) ^ (1 / 17)

/-- The cube root of the 17th power of the base number -/
def cube_root_of_power (b : ℝ) : ℝ :=
  (b ^ 17) ^ (1 / 3)

theorem base_number_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  ∀ x, x ≠ base_number →
  |cube_root_of_power base_number - 50| < 
  |cube_root_of_power x - 50| + ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_approximation_l677_67759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_m_value_l677_67730

def x : List ℚ := [0, 2, 4, 6, 8]

def y (m : ℚ) : List ℚ := [1, m+1, 2*m+1, 3*m+3, 11]

def mean (l : List ℚ) : ℚ := (l.sum) / l.length

def regression_line (x : ℚ) : ℚ := 1.4 * x + 1.4

theorem linear_regression_m_value :
  ∃ m : ℚ, 
    (mean x = 4) ∧ 
    (mean (y m) = 1.2 * m + 3.4) ∧ 
    (regression_line (mean x) = mean (y m)) ∧ 
    m = 3 := by
  sorry

#eval mean x
#eval regression_line (mean x)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_m_value_l677_67730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l677_67702

def mySequence (n : ℕ) : ℚ :=
  if n % 2 = 1 then 3 else 10

theorem fifteenth_term_is_three :
  let s := mySequence
  (∀ n : ℕ, n ≥ 1 → s n * s (n + 1) = 30) →
  s 0 = 3 →
  s 1 = 10 →
  s 14 = 3 := by
  sorry

#eval mySequence 14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l677_67702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_scenarios_highest_probability_sums_l677_67761

/-- Represents the possible numbers on a ball -/
inductive BallNumber
| one
| two
deriving Fintype, Repr

/-- Represents a box containing two balls -/
structure Box where
  ball1 : BallNumber
  ball2 : BallNumber
deriving Fintype, Repr

/-- Represents the result of drawing one ball from each of three boxes -/
structure DrawResult where
  x : BallNumber
  y : BallNumber
  z : BallNumber
deriving Fintype, Repr

/-- Calculates the sum of the numbers on the drawn balls -/
def sum_of_draw (result : DrawResult) : Nat :=
  match result.x, result.y, result.z with
  | BallNumber.one, BallNumber.one, BallNumber.one => 3
  | BallNumber.one, BallNumber.one, BallNumber.two => 4
  | BallNumber.one, BallNumber.two, BallNumber.one => 4
  | BallNumber.two, BallNumber.one, BallNumber.one => 4
  | BallNumber.one, BallNumber.two, BallNumber.two => 5
  | BallNumber.two, BallNumber.one, BallNumber.two => 5
  | BallNumber.two, BallNumber.two, BallNumber.one => 5
  | BallNumber.two, BallNumber.two, BallNumber.two => 6

/-- The set of all possible draw results -/
def all_possible_draws : Finset DrawResult := Finset.univ

/-- Theorem stating the number of possible scenarios -/
theorem number_of_scenarios : Finset.card all_possible_draws = 8 := by sorry

/-- Theorem stating the sums with highest probability -/
theorem highest_probability_sums :
  ∀ s : Nat, s ≠ 4 ∧ s ≠ 5 →
    (Finset.filter (λ r => sum_of_draw r = s) all_possible_draws).card <
    (Finset.filter (λ r => sum_of_draw r = 4) all_possible_draws).card ∧
    (Finset.filter (λ r => sum_of_draw r = s) all_possible_draws).card <
    (Finset.filter (λ r => sum_of_draw r = 5) all_possible_draws).card := by sorry

#eval all_possible_draws
#eval Finset.card all_possible_draws

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_scenarios_highest_probability_sums_l677_67761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_quarter_circle_l677_67728

open MeasureTheory

-- Define the sample space
def Ω : Set (ℝ × ℝ) := Set.prod (Set.Icc 0 1) (Set.Icc 0 1)

-- Define the probability measure on the sample space
variable (μ : Measure (ℝ × ℝ))

-- Assume the measure is uniform over the sample space
axiom μ_uniform : μ = volume.restrict Ω

-- Define the event
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 1}

-- State the theorem
theorem probability_quarter_circle (μ : Measure (ℝ × ℝ)) (h : μ = volume.restrict Ω) :
  μ (E ∩ Ω) / μ Ω = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_quarter_circle_l677_67728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_length_sB_onto_sA_l677_67791

noncomputable def sA : ℝ × ℝ := (4, 3)
noncomputable def sB : ℝ × ℝ := (-2, 6)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection_length (v w : ℝ × ℝ) : ℝ :=
  (dot_product v w) / (vector_magnitude w)

theorem projection_length_sB_onto_sA :
  projection_length sB sA = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_length_sB_onto_sA_l677_67791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l677_67769

theorem sin_double_angle_special_case (α : ℝ) 
  (h1 : Real.sin α = -4/5)
  (h2 : α ∈ Set.Ioo (-π/2) (π/2)) :
  Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l677_67769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_l677_67785

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1) * Real.exp (a * x)

-- Define the theorem
theorem extremum_and_monotonicity (a : ℝ) :
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| → |h| < ε → f a (1/2 + h) ≤ f a (1/2)) →
  (f a (1/2) = (5/4) * Real.exp (-2/5) ∧
   ∀ (x : ℝ), (0 < a ∧ a < 1) →
     (∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
       (∀ (y : ℝ), y < x₁ → (deriv (f a)) y > 0) ∧
       (∀ (y : ℝ), x₁ < y ∧ y < x₂ → (deriv (f a)) y < 0) ∧
       (∀ (y : ℝ), x₂ < y → (deriv (f a)) y > 0))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_l677_67785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_cartesian_line_l_rectangular_intersection_condition_l677_67748

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (1 + 3 * Real.cos α, 3 + 3 * Real.sin α)

-- Define the line l
noncomputable def line_l (m : ℝ) (θ : ℝ) : ℝ := Real.cos θ + Real.sin θ - m

-- Theorem for the Cartesian equation of curve C
theorem curve_C_cartesian : 
  ∀ (x y : ℝ), (∃ α : ℝ, curve_C α = (x, y)) ↔ (x - 1)^2 + (y - 3)^2 = 9 := by sorry

-- Theorem for the rectangular equation of line l
theorem line_l_rectangular : 
  ∀ (x y m : ℝ), (∃ ρ θ : ℝ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ line_l m θ = 0) 
  ↔ x + y - m = 0 := by sorry

-- Theorem for the intersection condition
theorem intersection_condition : 
  ∀ (m : ℝ), (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - 1)^2 + (y₁ - 3)^2 = 9 ∧ x₁ + y₁ - m = 0 ∧
    (x₂ - 1)^2 + (y₂ - 3)^2 = 9 ∧ x₂ + y₂ - m = 0 ∧
    (x₁, y₁) ≠ (x₂, y₂)) 
  ↔ 4 - 3 * Real.sqrt 2 < m ∧ m < 4 + 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_cartesian_line_l_rectangular_intersection_condition_l677_67748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l677_67706

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def g (a x : ℝ) : ℝ := Real.sqrt x + Real.sqrt (a - x)

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (a > 0) →
  (∀ x₁ ∈ Set.Icc 0 a, ∃ x₂ ∈ Set.Icc 4 16, g a x₁ = f x₂) →
  a ∈ Set.Icc 4 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l677_67706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_exists_l677_67770

-- Define a brick as a specific shape (truncated cube)
structure Brick where
  is_truncated_cube : Bool

-- Define a polyhedron
structure Polyhedron where
  is_convex : Bool

-- Define a function to construct a polyhedron from bricks
def construct_polyhedron (bricks : List Brick) : Polyhedron :=
  { is_convex := sorry } -- Placeholder implementation

-- Theorem statement
theorem convex_polyhedron_exists :
  ∃ (n : ℕ) (bricks : List Brick),
    (∀ b ∈ bricks, b.is_truncated_cube) ∧
    (bricks.length = n) ∧
    (construct_polyhedron bricks).is_convex := by
  -- Proof sketch
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_exists_l677_67770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l677_67718

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a * x

theorem function_properties (a : ℝ) :
  (∀ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), 
    (HasDerivAt (f a) ((Real.log x + a + 1) : ℝ) x) → Real.log x + a + 1 ≤ 0) →
  (a ≤ -3) ∧
  (∃ k : ℕ, k = 3 ∧ 
    (∀ x : ℝ, x > 1 → f a x > (k : ℝ) * (x - 1) + a * x - x) ∧
    (∀ m : ℕ, (∀ x : ℝ, x > 1 → f a x > (m : ℝ) * (x - 1) + a * x - x) → m ≤ k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l677_67718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_addition_solution_l677_67723

/-- Represents the mixture of milk and water in a can -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Calculates the ratio of milk to water in a mixture -/
def ratio (m : Mixture) : ℚ := m.milk / m.water

/-- Represents the problem of adding milk to a mixture -/
structure MilkAdditionProblem where
  initial : Mixture
  final : Mixture
  capacity : ℚ

/-- The main theorem to be proved -/
theorem milk_addition_solution (p : MilkAdditionProblem) :
  p.initial.milk + p.initial.water < p.capacity →
  ratio p.initial = 4/3 →
  ratio p.final = 5/2 →
  p.final.milk + p.final.water = p.capacity →
  p.final.water = p.initial.water →
  p.capacity = 30 →
  p.final.milk - p.initial.milk = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_addition_solution_l677_67723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l677_67766

/-- Definition of the function f(x) -/
def f (c : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + c * x + 1

/-- Condition that f has two extreme points in (0, +∞) -/
def has_two_extreme_points (c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 x₁ → (deriv (f c)) x ≠ 0) ∧
  (∀ x : ℝ, x ∈ Set.Ioo x₁ x₂ → (deriv (f c)) x ≠ 0) ∧
  (∀ x : ℝ, x > x₂ → (deriv (f c)) x ≠ 0) ∧
  (deriv (f c)) x₁ = 0 ∧ (deriv (f c)) x₂ = 0

/-- The main theorem -/
theorem f_range_theorem (c : ℝ) (h : has_two_extreme_points c) :
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ (f c x₁) / x₂ ∈ Set.Ioo 1 (5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l677_67766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l677_67714

/-- Curve C defined by x²/3 + y² = 1 -/
def curve_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- Line l defined by x - y + 2 = 0 -/
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

/-- Line l' parallel to l passing through (-1, 0) -/
def line_l' (x y : ℝ) : Prop := ∃ (t : ℝ), x = -1 + Real.sqrt 2 / 2 * t ∧ y = Real.sqrt 2 / 2 * t

/-- Point M at (-1, 0) -/
def point_M : ℝ × ℝ := (-1, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_properties :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    line_l' A.1 A.2 ∧ line_l' B.1 B.2 ∧
    distance A B = Real.sqrt 10 ∧
    distance point_M A * distance point_M B = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l677_67714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_divisible_by_three_tyoma_wins_l677_67765

/-- Represents the initial number on the board -/
def initial_number : ℕ := 123456789

/-- The number of times the sequence is repeated -/
def repetitions : ℕ := 2015

/-- The sum of digits in the initial sequence -/
def sequence_sum : ℕ := 45

/-- Represents a player's move: removing a digit and adding it to their total -/
def player_move (board : ℕ) (digit : ℕ) : ℕ :=
  board - digit + digit

/-- The theorem stating that the last remaining digit is divisible by 3 -/
theorem last_digit_divisible_by_three :
  ∃ (final_digit : ℕ), final_digit ∈ ({3, 6, 9} : Set ℕ) ∧
  (∀ (board : ℕ) (move : ℕ → ℕ),
    (board % 3 = 0) →
    (move board) % 3 = 0) →
  (initial_number * repetitions) % 3 = 0 →
  final_digit % 3 = 0 :=
by
  sorry

/-- The main theorem stating that Tyoma wins the game -/
theorem tyoma_wins :
  ∃ (final_digit : ℕ), final_digit ∈ ({3, 6, 9} : Set ℕ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_divisible_by_three_tyoma_wins_l677_67765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_coefficient_sum_l677_67757

/-- Given a function f: ℝ → ℝ satisfying f(x+2) = 2x^2 + 6x + 5 and f(x) = ax^2 + bx + c 
    for some real numbers a, b, and c, prove that a + b + c = 1 -/
theorem function_coefficient_sum (f : ℝ → ℝ) (a b c : ℝ) 
    (h1 : ∀ x, f (x + 2) = 2 * x^2 + 6 * x + 5)
    (h2 : ∀ x, f x = a * x^2 + b * x + c) : 
  a + b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_coefficient_sum_l677_67757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_non_square_differences_l677_67776

theorem subset_with_non_square_differences :
  ∃ (c : ℝ) (α : ℝ), c > 0 ∧ α > (1/2 : ℝ) ∧
    ∀ (n : ℕ), n > 0 →
      ∃ (A : Finset ℕ), A ⊆ Finset.range n ∧
        (A.card : ℝ) ≥ c * (n : ℝ) ^ α ∧
        ∀ (x y : ℕ), x ∈ A → y ∈ A → x ≠ y →
          ¬∃ (z : ℕ), z * z = x - y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_non_square_differences_l677_67776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_circle_has_min_area_l677_67720

/-- A circle tangent to y = 2/x (x > 0) and 2x + y + 1 = 0 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_x_pos : center.1 > 0
  tangent_to_curve : (center.1 - radius)*(center.2 + radius) = 2
  tangent_to_line : |2*center.1 + center.2 + 1| = radius * Real.sqrt 5

/-- The area of a circle -/
noncomputable def circleArea (c : TangentCircle) : ℝ := Real.pi * c.radius^2

/-- The specific circle with equation (x - 1)^2 + (y - 2)^2 = 5 -/
noncomputable def specificCircle : TangentCircle := {
  center := (1, 2),
  radius := Real.sqrt 5,
  center_x_pos := by simp
  tangent_to_curve := by sorry
  tangent_to_line := by sorry
}

/-- Theorem stating that the specific circle has the minimum area -/
theorem specific_circle_has_min_area :
  ∀ c : TangentCircle, circleArea specificCircle ≤ circleArea c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_circle_has_min_area_l677_67720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_is_three_l677_67773

/-- Calculates the length of a boat given its properties and the effect of added weight -/
noncomputable def boatLength (breadth : ℝ) (sinkDepth : ℝ) (manMass : ℝ) (waterDensity : ℝ) (gravity : ℝ) : ℝ :=
  (manMass * gravity) / (waterDensity * gravity * breadth * sinkDepth)

/-- Proves that the length of the boat is 3 meters given the specified conditions -/
theorem boat_length_is_three :
  let breadth : ℝ := 2
  let sinkDepth : ℝ := 0.01
  let manMass : ℝ := 60
  let waterDensity : ℝ := 1000
  let gravity : ℝ := 9.81
  boatLength breadth sinkDepth manMass waterDensity gravity = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_is_three_l677_67773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_6_seconds_l677_67744

/-- The time (in seconds) for two trains moving in opposite directions to pass each other completely. -/
noncomputable def trainPassingTime (speed1 speed2 : ℝ) (length1 length2 : ℝ) : ℝ :=
  let relativeSpeed := speed1 + speed2
  let relativeSpeedMS := (relativeSpeed * 1000) / 3600
  let combinedLength := length1 + length2
  combinedLength / relativeSpeedMS

/-- Theorem stating that the time for the trains to pass each other is approximately 6 seconds. -/
theorem train_passing_time_approx_6_seconds :
  ∃ ε > 0, |trainPassingTime 80 70 150 100 - 6| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_6_seconds_l677_67744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_composite_factorial_minus_one_l677_67767

theorem infinite_composite_factorial_minus_one (d : ℕ+) :
  {n : ℕ+ | ∃ (k l : ℕ+), k ≠ l ∧ k * l = d * (n : ℕ).factorial - 1}.Infinite :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_composite_factorial_minus_one_l677_67767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_l677_67742

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a median of a triangle
noncomputable def median (t : Triangle) (vertex : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define a line passing through two points
def line (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define perpendicular relation between a line and a vector
def perpendicular (l : Set (ℝ × ℝ)) (v : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem locus_of_points (t : Triangle) :
  ∀ M : ℝ × ℝ, 
    distance M t.A ^ 2 + distance M t.B ^ 2 = 2 * distance M t.C ^ 2 →
    M ∈ line (circumcenter t) (median t t.A) ∧ 
    perpendicular (line (circumcenter t) (median t t.A)) (median t t.A - t.A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_l677_67742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l677_67708

theorem circles_intersect (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) :
  c1 = (0, 0) →
  c2 = (2, 2) →
  r1 = 1 →
  r2 = Real.sqrt 5 →
  let d := Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2)
  d > r2 - r1 ∧ d < r2 + r1 →
  ∃ (p : ℝ × ℝ), p.1^2 + p.2^2 = r1^2 ∧ (p.1 - c2.1)^2 + (p.2 - c2.2)^2 = r2^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l677_67708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l677_67734

theorem sin_cos_relation (α : ℝ) (m n : ℝ) 
  (h1 : Real.sin α + Real.cos α = m) 
  (h2 : Real.sin α * Real.cos α = n) : 
  m^2 = 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l677_67734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_in_binomial_expansion_l677_67732

theorem odd_terms_in_binomial_expansion (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  (Finset.filter (fun k => Odd (Nat.choose 4 k * p^(4-k) * q^k)) (Finset.range 5)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_in_binomial_expansion_l677_67732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_proof_l677_67754

/-- Proves that the current height of a tree is 52 feet given specific growth conditions. -/
theorem tree_height_proof (growth_rate : ℕ) (future_years : ℕ) (future_height_inches : ℕ) 
  (inches_per_foot : ℕ) (h1 : growth_rate = 5)
  (h2 : future_years = 8) (h3 : future_height_inches = 1104) (h4 : inches_per_foot = 12) :
  (future_height_inches / inches_per_foot) - (growth_rate * future_years) = 52 := by
  sorry

#check tree_height_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_proof_l677_67754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sides_equal_diagonal_polygon_l677_67719

/-- A convex polygon with all diagonals of equal length -/
structure EqualDiagonalPolygon where
  n : ℕ  -- number of sides
  vertices : Fin n → ℝ × ℝ  -- vertices of the polygon
  convex : Convex ℝ (Set.range vertices)
  equal_diagonals : ∀ (i j k l : Fin n), 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l → 
    dist (vertices i) (vertices k) = dist (vertices j) (vertices l)

/-- The maximum number of sides in a convex polygon with all diagonals of equal length is 5 -/
theorem max_sides_equal_diagonal_polygon : 
  ∀ p : EqualDiagonalPolygon, p.n ≤ 5 := by
  sorry

#check max_sides_equal_diagonal_polygon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sides_equal_diagonal_polygon_l677_67719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_correct_l677_67752

/-- Calculates the length of a bridge given train parameters --/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- The bridge length calculation is correct for the given parameters --/
theorem bridge_length_correct :
  let train_length := (110 : ℝ)
  let train_speed_kmh := (72 : ℝ)
  let crossing_time := (14.248860091192705 : ℝ)
  let calculated_length := bridge_length train_length train_speed_kmh crossing_time
  abs (calculated_length - 174.98) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_correct_l677_67752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_calculation_l677_67764

/-- Calculates the speed of the second train given the lengths of two trains,
    the speed of the first train, and the time they take to cross each other. -/
noncomputable def second_train_speed (length1 length2 : ℝ) (speed1 time_to_cross : ℝ) : ℝ :=
  let total_distance := length1 + length2
  let relative_speed := total_distance / time_to_cross
  let relative_speed_kmh := relative_speed * 3.6
  relative_speed_kmh - speed1

/-- Theorem stating that under the given conditions, the speed of the second train
    is approximately 48.00287997120036 km/h. -/
theorem second_train_speed_calculation :
  let length1 := (140 : ℝ)
  let length2 := (160 : ℝ)
  let speed1 := (60 : ℝ)
  let time_to_cross := (9.99920006399488 : ℝ)
  abs (second_train_speed length1 length2 speed1 time_to_cross - 48.00287997120036) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_calculation_l677_67764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_half_second_l677_67746

/-- The height function of the athlete -/
noncomputable def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

/-- The instantaneous velocity at time t -/
noncomputable def v (t : ℝ) : ℝ := deriv h t

theorem instantaneous_velocity_at_half_second :
  v 0.5 = 1.6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_half_second_l677_67746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acquaintance_theorem_l677_67788

/-- Represents a company with employees and their acquaintance relationships -/
structure Company where
  employees : Finset Nat
  knows : employees → employees → Prop

/-- The property that among any 9 employees, at least two know each other -/
def has_acquaintance_property (c : Company) : Prop :=
  ∀ (group : Finset c.employees), group.card = 9 → ∃ (x y : c.employees), x ∈ group ∧ y ∈ group ∧ x ≠ y ∧ c.knows x y

/-- The existence of a group of 8 employees such that each remaining employee knows someone from this group -/
def has_eight_person_group (c : Company) : Prop :=
  ∃ (group : Finset c.employees), group.card = 8 ∧
    ∀ (x : c.employees), x ∉ group → ∃ (y : c.employees), y ∈ group ∧ c.knows x y

/-- The main theorem statement -/
theorem acquaintance_theorem (c : Company) :
  has_acquaintance_property c → has_eight_person_group c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acquaintance_theorem_l677_67788
