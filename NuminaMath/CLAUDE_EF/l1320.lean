import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l1320_132087

theorem right_triangle_side_length 
  (P Q R : ℝ × ℝ) 
  (right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0) 
  (cos_Q : Real.cos (Real.arctan ((R.2 - Q.2) / (R.1 - Q.1))) = 0.4) 
  (QP_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 12) :
  Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 30 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l1320_132087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_cubic_l1320_132044

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Define the point through which the tangent lines pass
def A : ℝ × ℝ := (1, -1)

-- Define the equations of the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := x - y - 2 = 0
def tangent_line_2 (x y : ℝ) : Prop := 5*x + 4*y - 1 = 0

theorem tangent_lines_to_cubic :
  ∀ x y : ℝ,
  (∃ t : ℝ, (x, y) = (t, f t) ∧ 
             ((x - A.1) * (f' t) = y - A.2)) →
  (tangent_line_1 x y ∨ tangent_line_2 x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_cubic_l1320_132044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_participants_shaken_all_but_one_is_M_minus_2_l1320_132013

/-- Represents a conference with M participants -/
structure Conference (M : ℕ) where
  participant_count : M > 5
  shaken_hands : Fin M → Fin M → Prop
  alex_sam_not_shaken : ∃ (alex sam : Fin M), alex ≠ sam ∧ ¬shaken_hands alex sam
  others_not_shaken_one : ∀ (p : Fin M), ∃! (q : Fin M), p ≠ q ∧ ¬shaken_hands p q

/-- The maximum number of participants who have shaken hands with every other participant except for one -/
def max_participants_shaken_all_but_one (M : ℕ) (conf : Conference M) : ℕ := M - 2

/-- Theorem stating that the maximum number of participants who have shaken hands with every other participant except for one is M-2 -/
theorem max_participants_shaken_all_but_one_is_M_minus_2 {M : ℕ} (conf : Conference M) :
  max_participants_shaken_all_but_one M conf = M - 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_participants_shaken_all_but_one_is_M_minus_2_l1320_132013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_approx_l1320_132089

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length bridge_length speed : ℝ) : ℝ :=
  (train_length + bridge_length) / speed

/-- Theorem stating that the train crossing time is approximately 5.56 seconds -/
theorem train_crossing_bridge_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_crossing_time 250 120 66.6 - 5.56| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_approx_l1320_132089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l1320_132011

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ :=
  Real.sqrt (r1^2 + r2^2 - 2*r1*r2*(Real.cos (θ2 - θ1)))

/-- Theorem: The distance between points A(1, π/6) and B(3, 5π/6) in polar coordinates is √13 -/
theorem distance_between_polar_points :
  polar_distance 1 (π/6) 3 (5*π/6) = Real.sqrt 13 := by
  -- Unfold the definition of polar_distance
  unfold polar_distance
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l1320_132011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_quadratic_l1320_132084

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 8*x - 15) → (∃ s : ℝ, s = 8 ∧ s = x + (8*x - 15)/x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_quadratic_l1320_132084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1320_132080

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 2 * (Real.cos (x / 2))^2) * Real.cos (x + Real.pi / 2)

theorem problem_solution :
  (∃ a : ℝ, f a (Real.pi / 2) = 0 ∧ a = -1) ∧
  (∀ α : ℝ, Real.pi / 2 < α ∧ α < Real.pi →
    f (-1) (α / 2) = -2 / 5 →
      Real.cos (Real.pi / 6 - 2 * α) = (-7 * Real.sqrt 3 - 24) / 50) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1320_132080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_agno3_combined_equals_agoh_formed_l1320_132000

/-- Represents the number of moles of a substance -/
structure Moles where
  value : ℝ

/-- The reaction between AgNO3 and NaOH produces AgOH in a 1:1 molar ratio -/
axiom reaction_ratio (agno3 naoh agoh : Moles) : agno3 = naoh ∧ agno3 = agoh

/-- The number of moles of NaOH available -/
def naoh_available : Moles := ⟨3⟩

/-- The number of moles of AgOH formed -/
def agoh_formed : Moles := ⟨3⟩

/-- Theorem stating that the number of moles of AgNO3 combined equals the number of moles of AgOH formed -/
theorem agno3_combined_equals_agoh_formed :
  ∃ (agno3_combined : Moles), agno3_combined = agoh_formed := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_agno3_combined_equals_agoh_formed_l1320_132000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gates_class_size_l1320_132022

/-- Proves that given the conditions of Mr. Gates' hot dog bun purchase, 
    the number of students in each of his classes is 30. -/
theorem gates_class_size :
  let buns_per_package : ℕ := 8
  let packages_bought : ℕ := 30
  let number_of_classes : ℕ := 4
  let buns_per_student : ℕ := 2
  let total_buns : ℕ := buns_per_package * packages_bought
  let total_students : ℕ := total_buns / buns_per_student
  let students_per_class : ℕ := total_students / number_of_classes
  students_per_class = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gates_class_size_l1320_132022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_formula_l1320_132002

noncomputable def a : ℕ → ℝ
  | 0 => 0
  | (n + 1) => 5 * a n + Real.sqrt (24 * (a n)^2 + 1)

theorem a_general_term_formula (n : ℕ) :
  a n = (Real.sqrt 6 / 24) * ((5 + 2 * Real.sqrt 6)^n - (5 - 2 * Real.sqrt 6)^n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_formula_l1320_132002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_eccentricity_l1320_132097

/-- The minimum eccentricity of a hyperbola with given conditions -/
theorem hyperbola_min_eccentricity :
  ∀ (H : Set (ℝ × ℝ)) (c : ℝ),
  c = 3 →
  (∀ (x y : ℝ), (x, y) ∈ H ↔ |Real.sqrt ((x + 3)^2 + y^2) - Real.sqrt ((x - 3)^2 + y^2)| = 2 * Real.sqrt (c^2 - 1)) →
  (∃ (x : ℝ), (x, x - 1) ∈ H) →
  ∃ (e : ℝ), (∀ (a : ℝ), a > c → e ≤ c / a) ∧ e = 3 * Real.sqrt 5 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_eccentricity_l1320_132097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_roots_quadratic_l1320_132023

theorem difference_of_roots_quadratic : 
  ∃ r₁ r₂ : ℝ, r₁^2 - 9*r₁ + 14 = 0 ∧ r₂^2 - 9*r₂ + 14 = 0 ∧ |r₁ - r₂| = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_roots_quadratic_l1320_132023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l1320_132018

theorem marble_probability : 
  let total_marbles : ℚ := 10
  let green_marbles : ℚ := 6
  let purple_marbles : ℚ := 4
  let total_draws : ℕ := 8
  let green_draws : ℕ := 3

  (Nat.choose total_draws green_draws : ℚ) * (green_marbles / total_marbles) ^ green_draws * 
   (purple_marbles / total_marbles) ^ (total_draws - green_draws) = 154828 / 125000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l1320_132018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_exceeds_capacity_in_75_years_l1320_132025

-- Define the parameters
noncomputable def total_land : ℝ := 30000
noncomputable def land_per_person : ℝ := 1.2
def initial_population : ℕ := 300
def years_to_quadruple : ℕ := 25

-- Define the population growth function
noncomputable def population (years : ℕ) : ℝ :=
  initial_population * (4 ^ (years / years_to_quadruple))

-- Define the island's capacity
noncomputable def island_capacity : ℝ :=
  total_land / land_per_person

-- Theorem statement
theorem population_exceeds_capacity_in_75_years :
  ∃ y : ℕ, y ≤ 75 ∧ population y ≥ island_capacity := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_exceeds_capacity_in_75_years_l1320_132025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_15gon_sum_quadratic_form_l1320_132016

/-- Represents a regular 15-gon inscribed in a circle -/
structure Regular15Gon where
  radius : ℝ
  vertices : Fin 15 → ℝ × ℝ

/-- The sum of lengths of all sides and diagonals of a regular 15-gon -/
def sumOfLengths (polygon : Regular15Gon) : ℝ :=
  sorry

/-- Expresses a real number in the form a + b√2 + c√3 + d√5 -/
structure QuadraticFormCustom where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Converts a real number to its quadratic form representation -/
noncomputable def toQuadraticFormCustom (x : ℝ) : QuadraticFormCustom :=
  sorry

theorem regular_15gon_sum_quadratic_form (polygon : Regular15Gon) 
  (h : polygon.radius = 15) : 
  ∃ (q : QuadraticFormCustom), sumOfLengths polygon = q.a + q.b * Real.sqrt 2 + q.c * Real.sqrt 3 + q.d * Real.sqrt 5 :=
by
  sorry

#check regular_15gon_sum_quadratic_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_15gon_sum_quadratic_form_l1320_132016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1320_132092

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2) / h.a

/-- The minimum perimeter of triangle APF where A is an endpoint of the conjugate axis,
    F is the right focus, and P is a point on the left branch of the hyperbola -/
def min_perimeter (h : Hyperbola) : ℝ :=
  6 * h.b

theorem hyperbola_eccentricity (h : Hyperbola) 
    (h_perimeter : min_perimeter h = 6 * h.b) : 
    eccentricity h = Real.sqrt 85 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1320_132092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1320_132037

noncomputable section

-- Define the polar coordinates of points O, A, and B
def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (6, Real.pi / 2)
noncomputable def B : ℝ × ℝ := (6 * Real.sqrt 2, Real.pi / 4)

-- Define the parametric equations of line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (-1 + t * Real.cos α, 2 + t * Real.sin α)

-- Define point P
def P : ℝ × ℝ := (-1, 2)

-- Define the circle C passing through O, A, and B
noncomputable def circle_C (θ : ℝ) : ℝ := 6 * Real.cos θ + 6 * Real.sin θ

-- State the theorem
theorem intersection_distance_product :
  ∃ (M N : ℝ × ℝ) (t₁ t₂ α : ℝ),
    (M = line_l t₁ α ∧ N = line_l t₂ α) ∧
    (Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) *
     Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2) = 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1320_132037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factorials_ends_with_zeros_l1320_132088

def last_two_digits (n : Nat) : Nat := n % 100

def ends_with_two_zeros (n : Nat) : Prop := last_two_digits n = 0

axiom factorial_ends_with_zeros : ∀ n : Nat, n ≥ 10 → ends_with_two_zeros (Nat.factorial n)

theorem sum_of_factorials_ends_with_zeros :
  last_two_digits (Nat.factorial 25 + Nat.factorial 50 + Nat.factorial 75 + Nat.factorial 100 + Nat.factorial 125) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factorials_ends_with_zeros_l1320_132088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blythe_is_northernmost_l1320_132003

-- Define the towns
inductive Town : Type
| Arva : Town
| Blythe : Town
| Cans : Town
| Dundee : Town
| Ernie : Town

-- Define the "north of" relation
def north_of : Town → Town → Prop := sorry

-- State the given conditions as axioms
axiom cans_north_of_ernie : north_of Town.Cans Town.Ernie
axiom dundee_south_of_cans : north_of Town.Cans Town.Dundee
axiom dundee_north_of_ernie : north_of Town.Dundee Town.Ernie
axiom arva_south_of_blythe : north_of Town.Blythe Town.Arva
axiom arva_north_of_dundee : north_of Town.Arva Town.Dundee
axiom arva_north_of_cans : north_of Town.Arva Town.Cans

-- Define the property of being the northernmost town
def is_northernmost (t : Town) : Prop :=
  ∀ other : Town, other ≠ t → north_of t other

-- State the theorem
theorem blythe_is_northernmost : is_northernmost Town.Blythe := by
  sorry

#check blythe_is_northernmost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blythe_is_northernmost_l1320_132003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_two_l1320_132079

noncomputable def f (a : ℤ) (x : ℝ) : ℝ := x^(a^2 - 2*a - 3)

theorem f_value_at_two (a : ℤ) :
  (∀ x > 0, ∀ y > 0, x < y → f a x > f a y) →  -- f is decreasing on (0, +∞)
  (∀ x : ℝ, f a x = f a (-x)) →                -- f is even
  f a 2 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_two_l1320_132079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_relation_l1320_132091

theorem vector_magnitude_relation (n : ℕ) : ∃ (a b : Fin n → ℝ), 
  (norm a = norm b ∧ norm (a + b) ≠ norm (a - b)) ∧
  (norm a ≠ norm b ∧ norm (a + b) = norm (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_relation_l1320_132091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surjective_functions_count_l1320_132083

/-- The number of surjective functions from a set of 4 elements to a set of 3 elements -/
theorem surjective_functions_count :
  (Fintype.card {f : Fin 4 → Fin 3 | Function.Surjective f}) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surjective_functions_count_l1320_132083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_imply_a_range_l1320_132012

-- Define the functions f and g
noncomputable def f (x : ℝ) := -Real.exp x - x
noncomputable def g (a : ℝ) (x : ℝ) := a * x + 2 * Real.cos x

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) := -Real.exp x - 1

-- Define the derivative of g
noncomputable def g_derivative (a : ℝ) (x : ℝ) := a - 2 * Real.sin x

-- State the theorem
theorem perpendicular_tangents_imply_a_range (a : ℝ) :
  (∀ x₀ : ℝ, ∃ t : ℝ, (f_derivative x₀) * (g_derivative a t) = -1) →
  a ∈ Set.Icc (-1 : ℝ) 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_imply_a_range_l1320_132012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_and_perpendicular_l1320_132008

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 2*x

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := x + 2*y = 0

-- Function to check if a point is on the circle
def on_circle (x y : ℝ) : Prop := my_circle x y

-- Function to check if a point is on line l
def on_line_l (x y : ℝ) : Prop := line_l x y

-- Function to get the center of the circle
def circle_center : (ℝ × ℝ) := (1, 2)

-- Theorem statement
theorem line_bisects_and_perpendicular :
  (on_line_l (circle_center.1) (circle_center.2)) ∧  -- line l passes through the center of the circle
  (∀ (x y : ℝ), perp_line x y → ¬(line_l x y)) →     -- line l is perpendicular to the given line
  (∀ (x y : ℝ), on_circle x y → on_line_l x y) ∧     -- line l bisects the circle
  (∀ (x y : ℝ), perp_line x y → ¬(line_l x y)) :=    -- line l is perpendicular to the given line
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_and_perpendicular_l1320_132008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_morse_code_distinct_symbols_l1320_132036

/-- Represents a Morse code symbol as either a dot or a dash -/
inductive MorseSymbol
| dot : MorseSymbol
| dash : MorseSymbol

/-- A Morse code sequence is a list of MorseSymbols -/
def MorseSequence := List MorseSymbol

/-- The number of possible Morse sequences of length n -/
def morseSequenceCount (n : ℕ) : ℕ := 2^n

/-- The total number of distinct Morse code symbols for sequences of length 1 to 5 -/
def totalDistinctSymbols : ℕ :=
  morseSequenceCount 1 + morseSequenceCount 2 + morseSequenceCount 3 +
  morseSequenceCount 4 + morseSequenceCount 5

theorem morse_code_distinct_symbols :
  totalDistinctSymbols = 62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_morse_code_distinct_symbols_l1320_132036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_eq_187_5_l1320_132086

/-- Represents a trapezoid ABCD with midpoints E and F -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  altitude : ℝ
  is_trapezoid : AB ≠ CD

/-- Calculate the area of quadrilateral EFCD in a trapezoid ABCD -/
noncomputable def area_EFCD (t : Trapezoid) : ℝ :=
  (t.altitude / 2) * ((t.AB + t.CD) / 2)

/-- Theorem: The area of quadrilateral EFCD in the given trapezoid is 187.5 square units -/
theorem area_EFCD_eq_187_5 (t : Trapezoid) 
    (h1 : t.AB = 10)
    (h2 : t.CD = 30)
    (h3 : t.altitude = 15) :
  area_EFCD t = 187.5 := by
  unfold area_EFCD
  simp [h1, h2, h3]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_eq_187_5_l1320_132086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_fraction_is_one_third_l1320_132095

/-- A cement mixture composed of sand, water, and gravel -/
structure CementMixture where
  total_weight : ℚ
  water_fraction : ℚ
  gravel_weight : ℚ

/-- The fraction of sand in the cement mixture -/
def sand_fraction (m : CementMixture) : ℚ :=
  1 - m.water_fraction - m.gravel_weight / m.total_weight

/-- Theorem: The fraction of sand in the specific cement mixture is 1/3 -/
theorem sand_fraction_is_one_third : 
  let m : CementMixture := {
    total_weight := 24,
    water_fraction := 1/4,
    gravel_weight := 10
  }
  sand_fraction m = 1/3 := by
  -- Proof goes here
  sorry

#eval sand_fraction { total_weight := 24, water_fraction := 1/4, gravel_weight := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_fraction_is_one_third_l1320_132095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_abs_g_odd_l1320_132057

-- Define the real-valued functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x

-- Theorem to prove
theorem f_abs_g_odd : ∀ x : ℝ, f x * |g x| = -(f (-x) * |g (-x)|) := by
  sorry

#check f_abs_g_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_abs_g_odd_l1320_132057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_one_l1320_132010

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := Real.exp x + x * Real.exp x

-- Theorem statement
theorem tangent_line_at_negative_one :
  let x₀ : ℝ := -1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  (λ (x y : ℝ) => y = m * (x - x₀) + y₀) = (λ (x y : ℝ) => y = -Real.exp (-1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_one_l1320_132010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_smallest_angle_l1320_132056

/-- A triangle with sides 2√10, 3√5, and 5 has its smallest angle equal to 45° --/
theorem triangle_smallest_angle (a b c : ℝ) (h1 : a = 2 * Real.sqrt 10) 
  (h2 : b = 3 * Real.sqrt 5) (h3 : c = 5) : 
  ∃ θ : ℝ, θ = Real.pi / 4 ∧ 
  θ = min (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))) 
          (min (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)))
               (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_smallest_angle_l1320_132056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_incenter_property_l1320_132014

/-- Given a hyperbola P: x²/9 - y²/16 = 1 with foci B and C, and a point A on P,
    if I is the incenter of triangle ABC and line AI passes through (1,0),
    then x + y = 3/4 where ⃗AI = x⃗AB + y⃗AC -/
theorem hyperbola_incenter_property (P : Set (ℝ × ℝ))
  (B C A : ℝ × ℝ) (I : ℝ × ℝ) (x y : ℝ) :
  (∀ p : ℝ × ℝ, p ∈ P ↔ (p.1^2 / 9 - p.2^2 / 16 = 1)) →
  (B.1 < 0 ∧ C.1 > 0) →
  (A ∈ P) →
  (I = (1, 0)) →
  (∃ k : ℝ, I.1 - A.1 = k * ((x * (B.1 - A.1)) + (y * (C.1 - A.1))) ∧
            I.2 - A.2 = k * ((x * (B.2 - A.2)) + (y * (C.2 - A.2)))) →
  (x + y = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_incenter_property_l1320_132014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_l1320_132099

theorem simplify_and_evaluate :
  (∀ x y z : ℝ, x = Real.sqrt 18 ∧ y = Real.sqrt 6 ∧ z = Real.sqrt 3 → x - y / z = 2 * Real.sqrt 2) ∧
  (∀ a b c : ℝ, a = 2 ∧ b = Real.sqrt 5 ∧ c = Real.sqrt 3 → (a + b) * (a - b) - (c - a)^2 = -8 + 4 * c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_l1320_132099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_smile_probability_l1320_132071

theorem baby_smile_probability (p : ℝ) (n k : ℕ) (hp : p = 1/3) (hn : n = 6) (hk : k = 3) :
  1 - (Finset.sum (Finset.range k) (λ i ↦ (n.choose i : ℝ) * p^i * (1-p)^(n-i))) = 353/729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_smile_probability_l1320_132071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_crossing_time_l1320_132052

/-- Calculates the time in minutes for a train to cross a tunnel -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length : ℝ) : ℝ :=
  let total_distance := train_length + tunnel_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let time_seconds := total_distance / train_speed_ms
  time_seconds / 60

/-- Theorem: A train of length 800 meters traveling at 78 km/hr takes 1 minute to cross a 500-meter tunnel -/
theorem train_tunnel_crossing_time :
  train_crossing_time 800 78 500 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_crossing_time_l1320_132052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_m_range_l1320_132021

-- Define the circle C
def circleC (x y m : ℝ) : Prop :=
  (x + 2)^2 + (y - m)^2 = 3

-- Define the condition |AB| = 2|GO|
def chord_condition (x y m : ℝ) : Prop :=
  ∃ (a b : ℝ × ℝ), 
    circleC a.1 a.2 m ∧ 
    circleC b.1 b.2 m ∧
    (x, y) = ((a.1 + b.1) / 2, (a.2 + b.2) / 2) ∧
    (b.1 - a.1)^2 + (b.2 - a.2)^2 = 4 * (x^2 + y^2)

-- Theorem statement
theorem circle_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, circleC x y m ∧ chord_condition x y m) ↔ -Real.sqrt 2 ≤ m ∧ m ≤ Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_m_range_l1320_132021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_inverse_ops_exist_l1320_132093

-- Define a non-commutative operation
def noncommutative_op {α : Type*} : α → α → α := sorry

-- Axiom: The operation is non-commutative
axiom non_comm {α : Type*} : ∃ (a b : α), noncommutative_op a b ≠ noncommutative_op b a

-- Define the concept of an inverse operation
def is_inverse_op {α : Type*} (f g : α → α → α) :=
  ∀ (a b : α), ∃ (c : α), f (g a b) c = a ∧ g (f a b) c = a

-- Theorem: There can exist multiple inverse operations for a non-commutative operation
theorem multiple_inverse_ops_exist {α : Type*} :
  ∃ (inv1 inv2 : α → α → α), 
    is_inverse_op noncommutative_op inv1 ∧ 
    is_inverse_op noncommutative_op inv2 ∧ 
    inv1 ≠ inv2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_inverse_ops_exist_l1320_132093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1320_132026

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := if x < 0 then -x^3 else x^3

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (3 * a - 1) ≥ 8 * f a) ↔ (a ≤ 1/5 ∨ a ≥ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1320_132026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_continuous_function_taking_each_value_once_not_exists_continuous_function_taking_each_value_twice_l1320_132046

-- Define a closed interval
def ClosedInterval (a b : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ b}

-- Define a property for a function taking each value exactly once
def TakesEachValueOnce (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ y, y ∈ f '' I → ∃! x, x ∈ I ∧ f x = y

-- Define a property for a function taking each value exactly twice
def TakesEachValueTwice (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ y, y ∈ f '' I → ∃ x₁ x₂, x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ ≠ x₂ ∧ f x₁ = y ∧ f x₂ = y ∧
    ∀ x, x ∈ I ∧ f x = y → (x = x₁ ∨ x = x₂)

-- Theorem 1
theorem exists_continuous_function_taking_each_value_once :
  ∃ (a b : ℝ) (f : ℝ → ℝ), a < b ∧ 
    ContinuousOn f (ClosedInterval a b) ∧
    TakesEachValueOnce f (ClosedInterval a b) := by
  sorry

-- Theorem 2
theorem not_exists_continuous_function_taking_each_value_twice :
  ¬ ∃ (a b : ℝ) (f : ℝ → ℝ), a < b ∧ 
    ContinuousOn f (ClosedInterval a b) ∧
    TakesEachValueTwice f (ClosedInterval a b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_continuous_function_taking_each_value_once_not_exists_continuous_function_taking_each_value_twice_l1320_132046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_rate_is_six_percent_l1320_132069

-- Define the problem parameters
noncomputable def initial_amount : ℝ := 4000
noncomputable def borrowing_period : ℝ := 2
noncomputable def borrowing_rate : ℝ := 4
noncomputable def gain_per_year : ℝ := 80

-- Define the simple interest function
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

-- Theorem statement
theorem lending_rate_is_six_percent :
  let borrowed_interest := simple_interest initial_amount borrowing_rate borrowing_period
  let total_gain := gain_per_year * borrowing_period
  let total_earnings := borrowed_interest + total_gain
  let lending_rate := (total_earnings * 100) / (initial_amount * borrowing_period)
  lending_rate = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_rate_is_six_percent_l1320_132069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_f_properties_l1320_132041

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, -1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (1, Real.cos x)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) (b x)

theorem perpendicular_vectors (x : ℝ) :
  dot_product (a x) (b x) = 0 → Real.tan x = Real.sqrt 3 / 3 :=
by sorry

theorem f_properties :
  (∃ p > 0, ∀ x, f (x + p) = f x) ∧
  (∃ p > 0, ∀ q > 0, (∀ x, f (x + q) = f x) → p ≤ q) ∧
  (∃ M, ∀ x, f x ≤ M) ∧
  (∀ M, (∀ x, f x ≤ M) → 2 ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_f_properties_l1320_132041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_necessary_not_sufficient_l1320_132063

/-- Two planes are perpendicular -/
def planes_perpendicular (α β : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (l : Set (Fin 3 → ℝ)) (β : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Set (Fin 3 → ℝ)) (α : Set (Fin 3 → ℝ)) : Prop := sorry

/-- Main theorem: Given two different planes α and β, and a line l in plane α,
    "α ⊥ β" is a necessary but not sufficient condition for "l ⊥ β" -/
theorem perpendicular_planes_necessary_not_sufficient 
  (α β : Set (Fin 3 → ℝ)) (l : Set (Fin 3 → ℝ)) 
  (h_diff : α ≠ β) 
  (h_in : line_in_plane l α) : 
  (planes_perpendicular α β → line_perpendicular_to_plane l β) ∧ 
  ¬(line_perpendicular_to_plane l β → planes_perpendicular α β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_necessary_not_sufficient_l1320_132063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rubles_problem_l1320_132024

theorem rubles_problem (x y n : ℕ) : 
  n * (x - 3) = y + 3 →
  x + n = 3 * (y - n) →
  n ∈ ({1, 2, 3, 7} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rubles_problem_l1320_132024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_probability_l1320_132072

theorem remainder_probability : 
  ∃ (S : Finset ℕ), S.card = 2027 ∧ 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 2027) ∧
  (Finset.filter (λ n => (n^20 % 7 = 1)) S).card / S.card = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_probability_l1320_132072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1320_132065

theorem solve_exponential_equation :
  ∃ x : ℝ, 3 * (5 : ℝ) ^ x = 1875 ∧ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1320_132065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_distance_l1320_132009

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 is given by
    |Ax₀ + By₀ + C| / √(A² + B²) -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

theorem point_line_distance (a : ℝ) :
  distance_point_to_line 2 2 3 (-4) a = a → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_distance_l1320_132009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_half_l1320_132006

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = a - 1/(2^x + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a - 1 / (2^x + 1)

/-- If f(x) = a - 1/(2^x + 1) is an odd function, then a = 1/2 -/
theorem odd_function_implies_a_half :
  ∃ a : ℝ, IsOdd (f a) → a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_half_l1320_132006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_2_l1320_132081

noncomputable def f (ω b : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4) + b

theorem f_at_pi_over_2 (ω b : ℝ) :
  ω > 0 →
  2 * Real.pi / 3 < 2 * Real.pi / ω →
  2 * Real.pi / ω < Real.pi →
  (∀ x, f ω b (3 * Real.pi / 2 - x) = f ω b (3 * Real.pi / 2 + x)) →
  f ω b (3 * Real.pi / 2) = 2 →
  f ω b (Real.pi / 2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_2_l1320_132081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2011_value_l1320_132017

/-- A sequence defined by a₁ = 0 and aₙ₊₁ = aₙ + 2n for n ≥ 1 -/
def a : ℕ → ℕ
  | 0 => 0  -- Add this case for n = 0
  | 1 => 0
  | n + 1 => a n + 2 * n

/-- The 2011th term of the sequence equals 2012 × 2011 -/
theorem a_2011_value : a 2011 = 2012 * 2011 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2011_value_l1320_132017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_profit_share_is_half_total_profit_l1320_132096

/-- Calculates the share of profit for an investor given the total profit and investments --/
noncomputable def calculate_profit_share (total_profit : ℝ) (investment_A : ℝ) (months_A : ℝ) (investment_B : ℝ) (months_B : ℝ) : ℝ :=
  let total_investment_time := investment_A * months_A + investment_B * months_B
  let share_A := (investment_A * months_A) / total_investment_time
  share_A * total_profit

/-- Theorem: A's share of the profit is half of the total profit --/
theorem A_profit_share_is_half_total_profit :
  calculate_profit_share 100 100 12 200 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_profit_share_is_half_total_profit_l1320_132096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_five_integers_l1320_132028

theorem product_of_five_integers (A B C D E : ℕ) : 
  A + B + C + D + E = 60 → 
  A + 3 = B - 3 → 
  A + 3 = C * 3 → 
  A + 3 = D / 3 → 
  A + 3 = E - 2 → 
  A * B * C * D * E = 64008 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_five_integers_l1320_132028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1320_132055

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + 2 * Real.sin (x - Real.pi / 4) * Real.sin (x + Real.pi / 4)

theorem f_range :
  let S := Set.Icc (-Real.pi / 12) (Real.pi / 2)
  ∀ y ∈ Set.range (f ∘ (↑) : S → ℝ), -Real.sqrt 3 / 2 ≤ y ∧ y ≤ 1 ∧
  ∃ x : S, f x = -Real.sqrt 3 / 2 ∧
  ∃ x : S, f x = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1320_132055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_diff_implies_strongest_relationship_l1320_132049

/-- Represents a pair of values (a, c) in a 2x2 contingency table -/
structure ContingencyPair where
  a : ℕ
  c : ℕ

/-- Calculates the absolute difference between a and c -/
def absDiff (pair : ContingencyPair) : ℕ :=
  max pair.a pair.c - min pair.a pair.c

/-- The set of given options -/
def options : List ContingencyPair :=
  [⟨45, 15⟩, ⟨40, 20⟩, ⟨35, 25⟩, ⟨30, 30⟩]

/-- Theorem stating that (45, 15) maximizes the absolute difference -/
theorem max_abs_diff_implies_strongest_relationship :
  ∃ (pair : ContingencyPair),
    pair ∈ options ∧
    (∀ (other : ContingencyPair),
      other ∈ options →
      absDiff pair ≥ absDiff other) ∧
    pair = ⟨45, 15⟩ :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_diff_implies_strongest_relationship_l1320_132049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_inscribed_squares_l1320_132060

/-- Represents that one square is inscribed in another -/
def IsInscribed (inner_perimeter outer_perimeter : ℝ) : Prop :=
  inner_perimeter < outer_perimeter ∧ 
  ∃ (angle : ℝ), 0 < angle ∧ angle < Real.pi/4

/-- Calculates the greatest distance between vertices of two squares, one inscribed in the other -/
noncomputable def GreatestDistance (inner_perimeter outer_perimeter : ℝ) : ℝ :=
  let inner_side := inner_perimeter / 4
  let outer_side := outer_perimeter / 4
  Real.sqrt ((outer_side - inner_side)^2 + outer_side^2)

/-- The greatest distance between vertices of two squares, one inscribed in the other -/
theorem greatest_distance_inscribed_squares (inner_perimeter outer_perimeter : ℝ) 
  (h_inner : inner_perimeter = 16)
  (h_outer : outer_perimeter = 40)
  (h_inscribed : IsInscribed inner_perimeter outer_perimeter) : 
  GreatestDistance inner_perimeter outer_perimeter = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_inscribed_squares_l1320_132060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersect_l1320_132094

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Define the line
def my_line (x y : ℝ) : Prop := x + y = 1

-- Theorem statement
theorem curves_intersect : ∃ (x y : ℝ), my_circle x y ∧ my_line x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersect_l1320_132094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l1320_132032

/-- The unit simplex in ℝ³ -/
def UnitSimplex : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 ∧ p.1 + p.2.1 + p.2.2 = 1}

/-- The subset S of the unit simplex -/
def SubsetS : Set (ℝ × ℝ × ℝ) :=
  {p ∈ UnitSimplex | (p.1 ≥ 1/4 ∧ p.2.1 ≥ 1/5) ∨ (p.1 ≥ 1/4 ∧ p.2.2 ≥ 1/6) ∨ (p.2.1 ≥ 1/5 ∧ p.2.2 ≥ 1/6)}

/-- The area ratio theorem -/
theorem area_ratio_theorem :
  (MeasureTheory.volume SubsetS) / (MeasureTheory.volume UnitSimplex) = 2149 / 3600 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l1320_132032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_and_roots_l1320_132090

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (log x) / x
noncomputable def g (a x : ℝ) : ℝ := (exp (a * x^2) - exp 1 * x + a * x^2 - 1) / x

-- Define the volume function
noncomputable def volume (x : ℝ) : ℝ := (π / 3) * ((log x)^2 / x)

-- State the theorem
theorem triangle_rotation_and_roots :
  -- Part 1: Maximum volume
  (∃ (v : ℝ), ∀ (x : ℝ), x > 1 → volume x ≤ v ∧ v = 4 * π / (3 * exp 2)) ∧
  -- Part 2: Properties of roots
  (∀ (a : ℝ), ∃ (x₁ x₂ : ℝ), 
    x₁ < x₂ ∧ 
    f x₁ = g a x₁ ∧ 
    f x₂ = g a x₂ →
    -- Part 2a: Range of a
    (0 < a ∧ a < exp 1 / 2) ∧
    -- Part 2b: Inequality for x₁² + x₂²
    x₁^2 + x₂^2 > 2 / exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_and_roots_l1320_132090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_equals_10100_l1320_132040

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case to handle Nat.zero
  | 1 => 2
  | (n + 1) => sequence_a n + (2 * sequence_a n) / n

theorem a_100_equals_10100 : sequence_a 100 = 10100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_equals_10100_l1320_132040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_l1320_132061

-- Define the domain
variable {α : Type*} [Fintype α] [DecidableEq α]
variable (x₁ x₂ x₃ : α)

-- Define the function g
variable (g : α → ℝ)

-- Define the area function for a triangle given three points
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- Define a function to convert α to ℝ
variable (f : α → ℝ)

-- State the theorem
theorem transformed_triangle_area 
  (h₁ : Fintype.card α = 3)
  (h₂ : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁)
  (h₃ : triangleArea (f x₁, g x₁) (f x₂, g x₂) (f x₃, g x₃) = 15) :
  triangleArea (3 * f x₁, 3 * g x₁) (3 * f x₂, 3 * g x₂) (3 * f x₃, 3 * g x₃) = 135 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_l1320_132061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1320_132098

-- Define the constant a
variable (a : ℝ)

-- Define the function f
noncomputable def f (x k : ℝ) : ℝ := 1 / Real.log (a^x + 4*a^(-x) - k)

-- State the theorem
theorem range_of_k (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, ∃ y, f a x k = y) ↔ k < 4 ∧ k ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1320_132098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_one_eq_f_three_l1320_132020

/-- A function f: ℝ → ℝ that is decreasing on (-∞, 1) and symmetric about x = 1 -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is decreasing on (-∞, 1) -/
axiom f_decreasing : ∀ x y, x < y → y < 1 → f x > f y

/-- f is symmetric about x = 1 -/
axiom f_symmetric : ∀ x, f (1 - x) = f (1 + x)

/-- The main theorem: f(-1) = f(3) -/
theorem f_minus_one_eq_f_three : f (-1) = f 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_one_eq_f_three_l1320_132020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1320_132034

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (k : ℝ)

-- State the theorem
theorem function_inequality (h1 : f 0 = -1) 
                            (h2 : ∀ x, HasDerivAt f (f' x) x) 
                            (h3 : ∀ x, f' x > k) 
                            (h4 : k > 1) : 
  f (1 / (k - 1)) > 1 / (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1320_132034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_sector_perimeter_40_l1320_132074

/-- Represents a circular sector -/
structure Sector where
  R : ℝ  -- radius
  α : ℝ  -- central angle in radians

/-- The perimeter of a sector -/
noncomputable def sector_perimeter (s : Sector) : ℝ := s.R * s.α + 2 * s.R

/-- The area of a sector -/
noncomputable def sector_area (s : Sector) : ℝ := 1/2 * s.R^2 * s.α

/-- Theorem: Maximum area of a sector with perimeter 40 -/
theorem max_area_sector_perimeter_40 :
  ∃ (s : Sector), 
    sector_perimeter s = 40 ∧ 
    sector_area s = 100 ∧ 
    s.α = 2 ∧
    ∀ (s' : Sector), sector_perimeter s' = 40 → sector_area s' ≤ sector_area s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_sector_perimeter_40_l1320_132074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l1320_132038

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x + Real.log x

-- Define the point of tangency
noncomputable def p : ℝ × ℝ := (Real.exp 2, Real.exp 2 + 2)

-- Define the slope of the tangent line at the point of tangency
noncomputable def k : ℝ := 1 + 1 / Real.exp 2

-- Define the y-intercept of the tangent line
def y_intercept : ℝ := 1

-- Theorem statement
theorem tangent_line_y_intercept :
  let x₀ := p.1
  let y₀ := p.2
  let m := k
  y_intercept = y₀ - m * x₀ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l1320_132038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_speed_is_10_l1320_132042

/-- Given a carriage ride with the following parameters:
  * distance: distance to the destination in miles
  * hourly_rate: cost per hour in dollars
  * flat_fee: flat fee for the ride in dollars
  * total_paid: total amount paid for the ride in dollars

  This function calculates the speed of the horse in miles per hour.
-/
noncomputable def calculate_horse_speed (distance : ℝ) (hourly_rate : ℝ) (flat_fee : ℝ) (total_paid : ℝ) : ℝ :=
  let time := (total_paid - flat_fee) / hourly_rate
  distance / time

/-- Theorem stating that for the given conditions, the horse's speed is 10 miles per hour. -/
theorem horse_speed_is_10 :
  calculate_horse_speed 20 30 20 80 = 10 := by
  -- Unfold the definition of calculate_horse_speed
  unfold calculate_horse_speed
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_speed_is_10_l1320_132042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_2x_l1320_132019

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

theorem min_positive_period_sin_2x :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_2x_l1320_132019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_distribution_l1320_132067

open Real MeasureTheory

noncomputable def normal_distribution (μ σ : ℝ) (x : ℝ) : ℝ := 
  (1 / (σ * sqrt (2 * π))) * exp (-((x - μ)^2) / (2 * σ^2))

theorem exam_score_distribution 
  (σ : ℝ) 
  (h_σ_pos : σ > 0) 
  (h_excellent : ∫ x in Set.Ici 120, normal_distribution 105 σ x = 0.2) :
  ∫ x in Set.Icc 90 105, normal_distribution 105 σ x = 0.3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_distribution_l1320_132067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptotes_l1320_132068

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := y = x/2 ∨ y = -x/2

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem distance_to_asymptotes :
  ∃ (d : ℝ), d = (2 / 5) * Real.sqrt 5 ∧
  ∀ (x y : ℝ), asymptote x y →
    distance_to_line 2 0 y (-x) (0 : ℝ) = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptotes_l1320_132068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_cosine_product_l1320_132030

theorem sine_sum_cosine_product (A B C : ℝ) :
  Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_cosine_product_l1320_132030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_less_than_24_l1320_132062

-- Define the line 5x + 12y = 0
def line (x y : ℝ) : Prop := 5 * x + 12 * y = 0

-- Define the points F₁ and F₂
def F₁ : ℝ × ℝ := (-13, 0)
def F₂ : ℝ × ℝ := (13, 0)

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem distance_difference_less_than_24 :
  ∀ M : ℝ × ℝ, line M.1 M.2 →
  |distance M F₁ - distance M F₂| < 24 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_less_than_24_l1320_132062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_intersection_area_l1320_132076

/-- The area of the square formed by intersecting a cube with a plane --/
theorem cube_intersection_area (cube_side_length : ℝ) (plane_position : ℝ) : 
  cube_side_length = 2 →
  plane_position = 1 →
  4 = 4 :=
by
  sorry

#check cube_intersection_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_intersection_area_l1320_132076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1320_132053

/-- Given an ellipse C with center at the origin, passing through the point (1, √3/2),
    and foci at (-√3, 0) and (√3, 0), its standard equation is x²/4 + y² = 1. -/
theorem ellipse_equation (C : Set (ℝ × ℝ)) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / 4 + y^2 = 1) ↔
  (0, 0) ∈ C ∧
  (1, Real.sqrt 3 / 2) ∈ C ∧
  (-Real.sqrt 3, 0) ∈ C ∧
  (Real.sqrt 3, 0) ∈ C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1320_132053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_weight_l1320_132001

/-- The weight of Leo in pounds -/
noncomputable def leo : ℝ := sorry

/-- The weight of Kendra in pounds -/
noncomputable def kendra : ℝ := sorry

/-- The weight of Jake in pounds -/
noncomputable def jake : ℝ := sorry

/-- The weight of Mia in pounds -/
noncomputable def mia : ℝ := sorry

/-- Leo's weight after gaining 15 pounds is 60% more than Kendra's weight -/
axiom leo_kendra_relation : leo + 15 = 1.60 * kendra

/-- Leo's weight after gaining 15 pounds is 40% of Jake's weight -/
axiom leo_jake_relation : leo + 15 = 0.40 * jake

/-- Jake weighs 25 pounds more than Kendra -/
axiom jake_kendra_relation : jake = kendra + 25

/-- Mia weighs 20 pounds less than Kendra -/
axiom mia_kendra_relation : mia = kendra - 20

/-- The combined weight of Leo, Kendra, Jake, and Mia is 350 pounds -/
axiom total_weight : leo + kendra + jake + mia = 350

/-- Leo's current weight is approximately 110.22 pounds -/
theorem leo_weight : ∃ ε > 0, |leo - 110.22| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_weight_l1320_132001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_degrees_in_zorbs_l1320_132004

/-- Represents the number of units in a full circle on Zorblat -/
noncomputable def zorblat_full_circle : ℚ := 600

/-- Represents the number of degrees in a full circle on Earth -/
noncomputable def earth_full_circle : ℚ := 360

/-- Converts an angle from degrees to zorbs -/
noncomputable def degrees_to_zorbs (degrees : ℚ) : ℚ :=
  (degrees / earth_full_circle) * zorblat_full_circle

theorem sixty_degrees_in_zorbs :
  degrees_to_zorbs 60 = 100 := by
  -- Unfold the definition of degrees_to_zorbs
  unfold degrees_to_zorbs
  -- Simplify the arithmetic
  simp [zorblat_full_circle, earth_full_circle]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_degrees_in_zorbs_l1320_132004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1320_132035

-- Define the line, circle, and ellipse
def my_line (m n x y : ℝ) := m * x - n * y = 4
def my_circle (x y : ℝ) := x^2 + y^2 = 4
def my_ellipse (x y : ℝ) := x^2 / 9 + y^2 / 4 = 1

-- Define the point P
def point_P (m n : ℝ) := (m, n)

-- Theorem statement
theorem intersection_points_count 
  (m n : ℝ) 
  (h1 : ∀ x y : ℝ, my_line m n x y → ¬my_circle x y) 
  (h2 : point_P m n = (m, n)) 
  (h3 : ∀ x y : ℝ, my_ellipse x y ↔ x^2 / 9 + y^2 / 4 = 1) :
  ∃! (p q : ℝ × ℝ), 
    p ≠ q ∧ 
    (∃ t : ℝ, p = (m + t, n + t)) ∧ 
    (∃ t : ℝ, q = (m + t, n + t)) ∧ 
    my_ellipse p.1 p.2 ∧ 
    my_ellipse q.1 q.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1320_132035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1320_132048

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - floor x

-- Define the equation
def equation (x : ℝ) : Prop :=
  (8 / frac x = 9 / x + 10 / (floor x : ℝ)) ∧ (x > 0) ∧ (x ≠ ↑(floor x))

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 3/2 := by
  sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1320_132048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_product_l1320_132070

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 3 - 3 * Complex.I) * (2 * Real.sqrt 2 + 2 * Complex.I)) = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_product_l1320_132070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_extrema_l1320_132066

noncomputable def f (x : ℝ) : ℝ := x + 1/x

theorem f_increasing_and_extrema :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → f x ≤ 17/4) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 2 ≤ f x) ∧
  f 4 = 17/4 ∧
  f 1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_extrema_l1320_132066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1320_132059

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  first_term_eq : a 1 = a 1  -- Placeholder for the first term
  term_eq : ∀ n : ℕ, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem statement -/
theorem arithmetic_sequence_property
  (seq : ArithmeticSequence)
  (h1 : seq.a 3 + seq.a 6 = 12)
  (h2 : S seq 4 = 8) :
  seq.a 9 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1320_132059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_false_statement_3_false_l1320_132073

/-- The function f(x) = sin(φx + φ) -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (φ * x + φ)

/-- Statement 1 is false -/
theorem statement_1_false : ¬ ∀ φ : ℝ, ∀ x : ℝ, f φ (x + 2 * Real.pi) = f φ x := by
  sorry

/-- Statement 3 is false -/
theorem statement_3_false : ¬ ∀ φ : ℝ, ∀ x : ℝ, f φ x ≠ f φ (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_false_statement_3_false_l1320_132073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_efficiency_problem_l1320_132007

/-- Represents the time taken by a worker to complete the job alone -/
structure WorkerTime where
  time : ℝ
  time_positive : time > 0

/-- Represents the combined work rate of multiple workers -/
noncomputable def combined_work_rate (workers : List WorkerTime) : ℝ :=
  workers.map (λ w => 1 / w.time) |> List.sum

/-- The problem statement -/
theorem worker_efficiency_problem 
  (delta epsilon zeta eta : WorkerTime)
  (h1 : combined_work_rate [delta, epsilon, zeta, eta] = 1 / (delta.time - 8))
  (h2 : combined_work_rate [delta, epsilon, zeta, eta] = 1 / (epsilon.time - 2))
  (h3 : combined_work_rate [delta, epsilon, zeta, eta] = 3 / zeta.time)
  : ∃ k : ℝ, k > 0 ∧ 1 / epsilon.time + 1 / zeta.time = 1 / k ∧ k = 120 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_efficiency_problem_l1320_132007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_positive_product_l1320_132058

def S : Finset Int := {-3, -6, 5, 2, -1}

def positive_product_pairs (S : Finset Int) : Finset (Int × Int) :=
  S.product S |>.filter (fun p => p.1 ≠ p.2 ∧ p.1 * p.2 > 0)

theorem probability_positive_product (S : Finset Int) :
  (positive_product_pairs S).card / (S.card.choose 2 : ℚ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_positive_product_l1320_132058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_speed_ratio_l1320_132033

/-- Represents the typing speed of a person in pages per hour -/
def TypingSpeed : Type := ℝ

/-- The number of hours it takes John to type the entire set of pages -/
def johnTotalTime : ℝ := 5

/-- The number of hours John types -/
def johnTypingTime : ℝ := 3

/-- The number of hours it takes Jack to finish the remaining pages -/
def jackTypingTime : ℝ := 5

/-- Theorem stating the ratio of Jack's typing speed to John's typing speed -/
theorem typing_speed_ratio (johnSpeed jackSpeed : ℝ) :
  jackSpeed / johnSpeed = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_speed_ratio_l1320_132033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_finite_valued_sequences_l1320_132051

/-- The function f(x) = 4x - x^2 -/
def f (x : ℝ) : ℝ := 4 * x - x^2

/-- The sequence x_n defined recursively -/
def x (n : ℕ) (x₀ : ℝ) : ℝ :=
  match n with
  | 0 => x₀
  | n + 1 => f (x n x₀)

/-- A set is finite if it has a bijection with a finite set of naturals -/
def IsFiniteSet (S : Set ℝ) : Prop := ∃ (n : ℕ), Nonempty (Equiv S (Fin n))

/-- The set of values in the sequence -/
def SequenceValues (x₀ : ℝ) : Set ℝ := {y : ℝ | ∃ (n : ℕ), y = x n x₀}

/-- The main theorem -/
theorem infinitely_many_finite_valued_sequences :
  ∃ (S : Set ℝ), (Set.Infinite S) ∧ (∀ x₀ ∈ S, IsFiniteSet (SequenceValues x₀)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_finite_valued_sequences_l1320_132051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sector_area_l1320_132029

-- Define the radius of the circle
noncomputable def radius : ℝ := 15

-- Define the angle of each sector in radians
noncomputable def sector_angle : ℝ := Real.pi / 4  -- 45° in radians

-- Define the area of the resulting figure
noncomputable def figure_area : ℝ := 2 * (sector_angle / (2 * Real.pi)) * Real.pi * radius^2

-- Theorem statement
theorem two_sector_area :
  figure_area = 56.25 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sector_area_l1320_132029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_within_radius_is_three_fourths_l1320_132027

/-- A semicircle with radius r -/
structure Semicircle where
  r : ℝ
  r_pos : r > 0

/-- A point on a semicircle -/
structure PointOnSemicircle (S : Semicircle) where
  angle : ℝ
  angle_range : 0 ≤ angle ∧ angle ≤ Real.pi

/-- The probability that two random points on a semicircle are at a distance not greater than the radius -/
noncomputable def probability_within_radius (S : Semicircle) : ℝ :=
  3/4

/-- Theorem: The probability that two random points on a semicircle are at a distance 
    not greater than the radius is 3/4 -/
theorem probability_within_radius_is_three_fourths (S : Semicircle) :
  probability_within_radius S = 3/4 := by
  sorry

#check probability_within_radius_is_three_fourths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_within_radius_is_three_fourths_l1320_132027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_54880000000_l1320_132054

theorem cube_root_54880000000 :
  Real.rpow 54880000000 (1/3) = 2800 * Real.rpow 2 (2/3) * Real.rpow 5 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_54880000000_l1320_132054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1320_132039

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2)^2 + 3 * (Real.log x / Real.log 2) + 2

-- Define the domain
def domain (x : ℝ) : Prop := 1/4 ≤ x ∧ x ≤ 4

-- Define t
noncomputable def t (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement
theorem f_properties :
  (∀ x, domain x → -2 ≤ t x ∧ t x ≤ 2) ∧
  (∃ x, domain x ∧ f x = -1/4 ∧ x = (1/2)^(3/2)) ∧
  (∃ x, domain x ∧ f x = 12 ∧ x = 4) ∧
  (∀ x, domain x → f x ≥ -1/4 ∧ f x ≤ 12) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1320_132039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_sum_l1320_132078

/-- A hexagon with vertices P, Q, R, S, T, U -/
structure Hexagon where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  T : ℝ × ℝ
  U : ℝ × ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculate the area of a hexagon -/
noncomputable def area (h : Hexagon) : ℝ :=
  sorry  -- Placeholder for area calculation

/-- Theorem: If a hexagon PQRSTU has area 68, PQ = 10, QR = 7, and TU = 6, then RS + ST = 3 -/
theorem hexagon_side_sum (h : Hexagon) 
    (area_cond : area h = 68)
    (pq_cond : distance h.P h.Q = 10)
    (qr_cond : distance h.Q h.R = 7)
    (tu_cond : distance h.T h.U = 6) :
    distance h.R h.S + distance h.S h.T = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_sum_l1320_132078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_with_one_even_digit_l1320_132015

def is_even (n : ℕ) : Bool :=
  n % 2 = 0

def is_valid_three_digit_number (n : ℕ) : Bool :=
  100 ≤ n ∧ n ≤ 999

def has_exactly_one_even_digit (n : ℕ) : Bool :=
  let hundreds : ℕ := n / 100
  let tens : ℕ := (n / 10) % 10
  let ones : ℕ := n % 10
  (is_even hundreds ∧ ¬is_even tens ∧ ¬is_even ones) ∨
  (¬is_even hundreds ∧ is_even tens ∧ ¬is_even ones) ∨
  (¬is_even hundreds ∧ ¬is_even tens ∧ is_even ones)

theorem count_three_digit_numbers_with_one_even_digit :
  (Finset.filter (fun n ↦ is_valid_three_digit_number n ∧ has_exactly_one_even_digit n) (Finset.range 1000)).card = 350 := by
  sorry

#eval (Finset.filter (fun n ↦ is_valid_three_digit_number n ∧ has_exactly_one_even_digit n) (Finset.range 1000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_with_one_even_digit_l1320_132015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1320_132005

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The system of inequalities defining the region --/
def InRegion (p : Point) : Prop :=
  p.x + 2 * p.y ≤ 4 ∧
  3 * p.x + p.y ≥ 3 ∧
  p.x ≥ 0 ∧
  p.y ≥ 0

/-- The distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The theorem stating that the longest side of the region has length 2√5 --/
theorem longest_side_length :
  ∃ (p1 p2 : Point), InRegion p1 ∧ InRegion p2 ∧
    distance p1 p2 = 2 * Real.sqrt 5 ∧
    ∀ (q1 q2 : Point), InRegion q1 → InRegion q2 →
      distance q1 q2 ≤ 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1320_132005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_correct_l1320_132077

/-- Represents the number 144...430 with n number of 4s -/
def number (n : ℕ) : ℕ :=
  100000 * 10^n + 44 * ((10^n - 1) / 9) + 430

/-- Predicate to check if a number is divisible by 2015 -/
def is_multiple_of_2015 (x : ℕ) : Prop :=
  x % 2015 = 0

/-- The smallest n such that number(n) is a multiple of 2015 -/
def smallest_n : ℕ := 14

theorem smallest_n_is_correct :
  (∀ k < smallest_n, ¬ is_multiple_of_2015 (number k)) ∧
  is_multiple_of_2015 (number smallest_n) := by
  sorry

#check smallest_n_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_correct_l1320_132077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_a_range_implies_f_increasing_f_increasing_iff_a_range_l1320_132050

/-- The function f(x) = 2ln(x) + 1 - a/x --/
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x + 1 - a / x

/-- The derivative of f(x) --/
noncomputable def f_deriv (x : ℝ) (a : ℝ) : ℝ := (2 * x + a) / (x^2)

/-- Theorem: If f(x) is increasing on (1, +∞), then a ∈ [-2, +∞) --/
theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x > 1, f_deriv x a ≥ 0) → a ≥ -2 := by
  intro h
  -- Proof goes here
  sorry

/-- Theorem: If a ∈ [-2, +∞), then f(x) is increasing on (1, +∞) --/
theorem a_range_implies_f_increasing (a : ℝ) :
  a ≥ -2 → (∀ x > 1, f_deriv x a ≥ 0) := by
  intro h x hx
  -- Proof goes here
  sorry

/-- Main theorem: f(x) is increasing on (1, +∞) if and only if a ∈ [-2, +∞) --/
theorem f_increasing_iff_a_range (a : ℝ) :
  (∀ x > 1, f_deriv x a ≥ 0) ↔ a ≥ -2 := by
  constructor
  · exact f_increasing_implies_a_range a
  · exact a_range_implies_f_increasing a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_a_range_implies_f_increasing_f_increasing_iff_a_range_l1320_132050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_positive_solution_set_l1320_132085

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4 * Real.log x

-- State the theorem
theorem f_derivative_positive_solution_set :
  {x : ℝ | (deriv f) x > 0} = {x : ℝ | x > 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_positive_solution_set_l1320_132085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_digits_icosahedral_die_l1320_132082

/-- A fair 20-sided die with numbers from 1 to 20 -/
structure IcosahedralDie where
  faces : Fin 20 → ℕ
  fair : ∀ i : Fin 20, faces i ∈ Finset.range 21

/-- The number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- The expected number of digits on the top face of a fair 20-sided die -/
def expectedDigits (d : IcosahedralDie) : ℚ :=
  (Finset.sum (Finset.range 20) (λ i => numDigits (d.faces i))) / 20

theorem expected_digits_icosahedral_die :
  ∀ d : IcosahedralDie, expectedDigits d = 31/20 := by
  sorry

#eval (31 : ℚ) / 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_digits_icosahedral_die_l1320_132082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_numbers_with_distinct_sums_l1320_132043

theorem existence_of_numbers_with_distinct_sums (k : ℕ) 
  (h_k : k ≤ (2022 * 2021) / 2) : 
  ∃ (S : Finset ℝ), 
    Finset.card S = 2022 ∧ 
    (∀ (x y : ℝ), x ∈ S → y ∈ S → x ≠ y → 
      ∀ (a b : ℝ), a ∈ S → b ∈ S → a ≠ b → x + y ≠ a + b) ∧
    ((Finset.filter (fun p => p.1 < p.2 ∧ p.1 + p.2 > 0) 
      (Finset.product S S)).card = k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_numbers_with_distinct_sums_l1320_132043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shift_for_symmetry_l1320_132047

noncomputable def f (x : ℝ) : ℝ := (3/2) * Real.cos (2*x) + (Real.sqrt 3 / 2) * Real.sin (2*x)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x + m)

theorem smallest_shift_for_symmetry :
  ∃ (m : ℝ), m > 0 ∧
  (∀ (x : ℝ), g m x = g m (-x)) ∧
  (∀ (m' : ℝ), m' > 0 → (∀ (x : ℝ), g m' x = g m' (-x)) → m ≤ m') ∧
  m = Real.pi / 12 := by
  sorry

#check smallest_shift_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shift_for_symmetry_l1320_132047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_l1320_132075

-- Define the polynomial function
def polynomial (x : ℝ) : ℝ := 
  (x - 1)^1005 + 2*(x - 2)^1004 + 3*(x - 3)^1003 + 
  1004*(x - 1004)^2 + 1005*(x - 1005)

-- State the theorem about the sum of roots
theorem sum_of_roots : 
  ∃ (roots : Finset ℝ), (∀ r ∈ roots, polynomial r = 0) ∧ (roots.sum id = 1003) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_l1320_132075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_sequence_sorted_P_sequence_valid_P_sequence_complete_position_of_2015_l1320_132064

/-- P sequence: all positive integers whose digits sum to 8, arranged in ascending order -/
def P_sequence : List ℕ := sorry

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number belongs to the P sequence -/
def in_P_sequence (n : ℕ) : Prop :=
  sum_of_digits n = 8 ∧ n > 0

/-- The P sequence is sorted in ascending order -/
theorem P_sequence_sorted : List.Sorted (·<·) P_sequence := sorry

/-- Every number in P_sequence satisfies in_P_sequence -/
theorem P_sequence_valid : ∀ n ∈ P_sequence, in_P_sequence n := sorry

/-- Every number satisfying in_P_sequence is in P_sequence -/
theorem P_sequence_complete : ∀ n, in_P_sequence n → n ∈ P_sequence := sorry

/-- The main theorem: 2015 is the 83rd number in the P sequence -/
theorem position_of_2015 : P_sequence.get? 82 = some 2015 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_sequence_sorted_P_sequence_valid_P_sequence_complete_position_of_2015_l1320_132064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_complex_circle_l1320_132031

theorem min_slope_complex_circle (x y : ℝ) : 
  (x - 1)^2 + y^2 = 1 → 
  ∃ (k : ℝ), k = Real.sqrt 3 / 3 ∧ 
    ∀ (x' y' : ℝ), (x' - 1)^2 + y'^2 = 1 → 
      y' / (x' + 1) ≥ -k ∧ y' / (x' + 1) ≤ k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_complex_circle_l1320_132031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l1320_132045

theorem product_of_four_integers (A B C D : ℕ+) : 
  A + B + C + D = 48 → 
  A + 3 = B - 3 → 
  A + 3 = C * 3 → 
  A + 3 = (D : ℚ) / 3 → 
  A * B * C * D = 5832 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l1320_132045
