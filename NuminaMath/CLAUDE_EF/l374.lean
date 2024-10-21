import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_properties_l374_37435

def sequence_b : ℕ → ℝ
  | 0 => 5  -- We define b₀ as 5 to match b₁ in the problem
  | 1 => 3
  | (n + 2) => 2 * sequence_b n + sequence_b (n + 1)

def is_arithmetic_progression (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

theorem sequence_b_properties :
  (¬ is_arithmetic_progression sequence_b) ∧ (sequence_b 4 = 45) := by
  sorry

#eval sequence_b 4  -- This will evaluate the 5th term (index 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_properties_l374_37435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_theta_for_symmetry_l374_37466

noncomputable def f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x

noncomputable def g (x θ : ℝ) := 2 * Real.sin (2 * (x - θ) + Real.pi / 3)

theorem min_theta_for_symmetry :
  ∃ (θ : ℝ), θ > 0 ∧
  (∀ (x : ℝ), g (3 * Real.pi / 4 + x) θ = g (3 * Real.pi / 4 - x) θ) ∧
  (∀ (θ' : ℝ), θ' > 0 → 
    (∀ (x : ℝ), g (3 * Real.pi / 4 + x) θ' = g (3 * Real.pi / 4 - x) θ') → 
    θ' ≥ θ) ∧
  θ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_theta_for_symmetry_l374_37466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_a_value_l374_37499

/-- A function f is symmetric about a line x = c if for all x, f(c + d) = f(c - d) for any d --/
def SymmetricAboutLine (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ d : ℝ, f (c + d) = f (c - d)

/-- The main theorem --/
theorem symmetric_function_a_value (a : ℝ) :
  SymmetricAboutLine (fun x ↦ |x + 1| + |x - a|) 1 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_a_value_l374_37499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l374_37493

/-- The line equation y = (x - 3) / 3 -/
def line_equation (x y : ℝ) : Prop := y = (x - 3) / 3

/-- The point we're finding the closest point to -/
noncomputable def target_point : ℝ × ℝ := (0, 2)

/-- The claimed closest point on the line -/
noncomputable def closest_point : ℝ × ℝ := (9/10, -7/10)

/-- Theorem stating that the closest_point is indeed the closest point on the line to the target_point -/
theorem closest_point_is_closest :
  line_equation closest_point.1 closest_point.2 ∧
  ∀ (p : ℝ × ℝ), line_equation p.1 p.2 →
    ‖p - target_point‖ ≥ ‖closest_point - target_point‖ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l374_37493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l374_37492

/-- Given two vectors in R³ that are parallel, prove their components satisfy x + y = 2 -/
theorem parallel_vectors_sum (x y : ℝ) :
  let a : Fin 3 → ℝ := ![2 - x, -1, y]
  let b : Fin 3 → ℝ := ![-1, x, -1]
  (∃ (k : ℝ), a = k • b) →
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l374_37492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l374_37414

noncomputable def f (x : ℝ) : ℝ := 
  Real.exp x - Real.exp 0 * x + (1/2) * x^2

theorem f_derivative_at_one : 
  (deriv f) 1 = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l374_37414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_colors_l374_37470

/-- The probability of drawing one white, one red, and one green ball from a bag containing 6 black, 5 white, 4 red, and 3 green balls when drawing 3 balls without replacement -/
theorem probability_three_colors (black white red green : ℕ) 
  (h_black : black = 6)
  (h_white : white = 5)
  (h_red : red = 4)
  (h_green : green = 3) :
  let total := black + white + red + green
  let ways_to_choose_three := (total.choose 3)
  let ways_to_choose_colors := white * red * green
  (ways_to_choose_colors : ℚ) / ways_to_choose_three = 5 / 68 :=
by
  -- Unfold the let bindings
  simp_all only [h_black, h_white, h_red, h_green]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_colors_l374_37470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_to_199_l374_37471

/-- The sum of the alternating series from 1 to n -/
def alternating_sum : ℕ → ℤ
  | 0 => 0
  | n + 1 => alternating_sum n + if n % 2 = 0 then (n + 1 : ℤ) else -(n + 1 : ℤ)

/-- The theorem stating that the sum of the alternating series from 1 to 199 is 100 -/
theorem alternating_sum_to_199 : alternating_sum 199 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_to_199_l374_37471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_circle_complete_sin_circle_minimal_l374_37450

/-- The smallest angle that generates a complete circle for r = sin θ -/
noncomputable def smallest_complete_circle_angle : ℝ := Real.pi

theorem sin_circle_complete (t : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = Real.sin θ) →
  (∀ r θ : ℝ, r = Real.sin θ → ∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = (Real.sin θ) * (Real.sin θ)) →
  t ≥ smallest_complete_circle_angle :=
by sorry

theorem sin_circle_minimal (t : ℝ) :
  t < smallest_complete_circle_angle →
  ∃ x y : ℝ, x^2 + y^2 ≤ 1 ∧
    ¬∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = (Real.sin θ) * (Real.sin θ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_circle_complete_sin_circle_minimal_l374_37450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l374_37416

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
noncomputable def special_triangle : Triangle where
  a := Real.sqrt 5
  b := 3
  c := 2 * Real.sqrt 5
  A := Real.arcsin ((Real.sqrt 5) / 5)
  B := Real.arcsin (3 / (2 * Real.sqrt 5))
  C := Real.arcsin ((2 * Real.sqrt 5) / 5)

/-- Theorem stating the properties of the special triangle -/
theorem special_triangle_properties (t : Triangle) 
  (h1 : t.a = Real.sqrt 5)
  (h2 : t.b = 3)
  (h3 : Real.sin t.C = 2 * Real.sin t.A) :
  t.c = 2 * Real.sqrt 5 ∧ 
  Real.cos (2 * t.A) = 3 / 5 ∧ 
  (1 / 2) * t.b * t.c * Real.sin t.A = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l374_37416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_a_votes_l374_37483

/-- Represents an election with given parameters -/
structure Election where
  total_votes : ℕ
  invalid_percent : ℚ
  candidate_a_percent : ℚ

/-- Calculates the number of valid votes for Candidate A in the given election -/
def valid_votes_for_candidate_a (e : Election) : ℕ :=
  ⌊(e.total_votes : ℚ) * (1 - e.invalid_percent) * e.candidate_a_percent⌋.toNat

/-- Theorem stating that for the given election parameters, 
    Candidate A receives 405,000 valid votes -/
theorem candidate_a_votes (e : Election) 
  (h1 : e.total_votes = 1200000)
  (h2 : e.invalid_percent = 1/4)
  (h3 : e.candidate_a_percent = 45/100) :
  valid_votes_for_candidate_a e = 405000 := by
  sorry

#eval valid_votes_for_candidate_a ⟨1200000, 1/4, 45/100⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_a_votes_l374_37483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l374_37400

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + 2

noncomputable def tangent_line (f : ℝ → ℝ) (x₀ : ℝ) : ℝ → ℝ :=
  λ x => (deriv f x₀) * (x - x₀) + f x₀

theorem tangent_line_at_zero :
  tangent_line f 0 = λ x => 2 * x + 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l374_37400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_set_C_values_l374_37484

-- Define the universal set Z as the set of all real numbers
def Z : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 + 2*x - 15 = 0}

-- Define set B as a function of a
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

-- Define the complement of B with respect to Z
def complement_Z_B (a : ℝ) : Set ℝ := Z \ B a

-- Statement for question 1
theorem intersection_A_complement_B :
  A ∩ (complement_Z_B (1/5)) = {-5, 3} := by sorry

-- Define set C
def C : Set ℝ := {a | B a ⊆ A}

-- Statement for question 2
theorem set_C_values : C = {-1/5, 1/3, 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_set_C_values_l374_37484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_for_even_f_l374_37451

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (x + θ) + Real.cos (x + θ)

theorem theta_value_for_even_f (θ : ℝ) (h1 : θ ∈ Set.Icc 0 (Real.pi / 2)) 
  (h2 : ∀ x, f θ x = f θ (-x)) : θ = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_for_even_f_l374_37451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angle_planes_l374_37422

/-- Represents a line in 3D space -/
structure Line3D where
  direction : Fin 3 → ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  normal : Fin 3 → ℝ
  point : Fin 3 → ℝ

/-- Represents the projection of a line onto a plane -/
noncomputable def project_line_to_plane (l : Line3D) (p : Plane3D) : Line3D :=
  sorry

/-- The angle between a line and a plane -/
noncomputable def angle_line_plane (l : Line3D) (p : Plane3D) : ℝ :=
  sorry

/-- The fourth projection plane -/
noncomputable def P₄ : Plane3D :=
  sorry

/-- The reference plane -/
noncomputable def reference_plane : Plane3D :=
  sorry

theorem equal_angle_planes
  (g : Line3D)
  (g' : Line3D)
  (g'' : Line3D)
  (h1 : g' = project_line_to_plane g reference_plane)
  (h2 : g'' = project_line_to_plane g (Plane3D.mk (λ _ => 0) (λ _ => 0)))
  (h3 : project_line_to_plane g P₄ = g')
  (h4 : angle_line_plane g reference_plane = Real.pi / 4) :
  angle_line_plane g P₄ = angle_line_plane g reference_plane :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angle_planes_l374_37422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_g_symmetry_center_f_l374_37402

noncomputable def f (x φ₁ : ℝ) : ℝ := Real.sin (2 * x + φ₁)
noncomputable def g (x φ₂ : ℝ) : ℝ := Real.cos (4 * x + φ₂)

theorem symmetry_axis_g (φ φ₁ φ₂ : ℝ) (k : ℤ) 
  (h₁ : |φ₁| ≤ π/2) (h₂ : |φ₂| ≤ π/2)
  (h₃ : ∀ x, f x φ₁ = f (2*φ - x) φ₁) 
  (h₄ : ∀ x, g x φ₂ = g (2*φ - x) φ₂) :
  ∀ x, g x φ₂ = g (2*(1/2 * ↑k * π + φ) - x) φ₂ :=
by
  sorry

theorem symmetry_center_f (φ φ₁ φ₂ : ℝ) (k : ℤ) 
  (h₁ : |φ₁| ≤ π/2) (h₂ : |φ₂| ≤ π/2)
  (h₃ : ∀ x, f (φ + x) φ₁ = f (φ - x) φ₁) 
  (h₄ : ∀ x, g (φ + x) φ₂ = g (φ - x) φ₂) :
  ∃ k : ℤ, ∀ x, f ((k * π / 4 + φ) + x) φ₁ = f ((k * π / 4 + φ) - x) φ₁ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_g_symmetry_center_f_l374_37402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_trig_function_l374_37455

/-- Given a function f(x) = a sin x + cos x where x = π/6 is one of its symmetry axes,
    prove that its maximum value is 2√3/3 -/
theorem max_value_of_trig_function (a : ℝ) :
  let f : ℝ → ℝ := fun x => a * Real.sin x + Real.cos x
  (∃ (c : ℝ), f (π/6) = c ∧ f (π/3) = c) →
  (∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x) ∧
  (∃ (x : ℝ), f x = 2*Real.sqrt 3/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_trig_function_l374_37455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_by_six_l374_37446

def is_valid_number (n : ℕ) : Prop :=
  (10000 ≤ n ∧ n < 100000) ∧
  ∃ (a b c d e : ℕ),
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    ({a, b, c, d, e} : Finset ℕ) = {1, 2, 3, 7, 8}

theorem smallest_valid_divisible_by_six :
  ∀ n : ℕ, is_valid_number n → n % 6 = 0 → n ≥ 13782 :=
by
  intro n h1 h2
  sorry

#check smallest_valid_divisible_by_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_by_six_l374_37446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l374_37487

/-- Given a square with side length z and an inscribed square with side length w,
    where the larger square is divided into the smaller square and four identical rectangles,
    the perimeter of one of these rectangles is w + z. -/
theorem rectangle_perimeter (z w : ℝ) (hz : z > 0) (hw : w > 0) (hw_lt_z : w < z) : 
  2 * w + 2 * ((z - w) / 2) = w + z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l374_37487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_problem_l374_37408

/-- The population problem for Wellington, Port Perry, Lazy Harbor, and Newbridge. -/
theorem population_problem (wellington : ℚ) (port_perry lazy_harbor newbridge : ℚ) 
  (h1 : wellington = 900)
  (h2 : port_perry = 7 * wellington)
  (h3 : lazy_harbor = 2 * wellington + 600)
  (h4 : newbridge = 3 * (port_perry - wellington))
  : ⌊(1 + 0.125) * newbridge⌋ = 18225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_problem_l374_37408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_burned_dancing_per_week_l374_37448

/-- Calories burned per hour while walking -/
def walking_calories_per_hour : ℝ := 300

/-- Calories burned per hour while dancing -/
def dancing_calories_per_hour : ℝ := 2 * walking_calories_per_hour

/-- Duration of each dance session in hours -/
def dance_session_duration : ℝ := 0.5

/-- Number of dance sessions per day -/
def dance_sessions_per_day : ℕ := 2

/-- Number of days James dances per week -/
def dance_days_per_week : ℕ := 4

/-- Theorem: James burns 2400 calories a week from dancing -/
theorem calories_burned_dancing_per_week : 
  dancing_calories_per_hour * dance_session_duration * 
  (dance_sessions_per_day : ℝ) * (dance_days_per_week : ℝ) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_burned_dancing_per_week_l374_37448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_in_cube_l374_37437

def X : ℝ × ℝ × ℝ := (0, 0, 0)
def Y : ℝ × ℝ × ℝ := (5, 5, 5)

def cube_edge_length : ℝ := 4

def cube_min : ℝ × ℝ × ℝ := (-2, -2, -2)
def cube_max : ℝ × ℝ × ℝ := (2, 2, 2)

theorem segment_length_in_cube :
  let segment_length := Real.sqrt (
    (cube_max.1 - cube_min.1)^2 + 
    (cube_max.2.1 - cube_min.2.1)^2 + 
    (cube_max.2.2 - cube_min.2.2)^2
  )
  segment_length = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_in_cube_l374_37437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_bounds_specific_values_l374_37457

def original_number : ℚ := 0.20120415

-- Define a function to generate all possible repeating decimals
noncomputable def generate_repeating_decimals (x : ℚ) : Set ℚ := sorry

-- Define the largest repeating decimal
noncomputable def largest_repeating_decimal (x : ℚ) : ℚ := sorry

-- Define the smallest repeating decimal
noncomputable def smallest_repeating_decimal (x : ℚ) : ℚ := sorry

theorem repeating_decimal_bounds :
  let repeating_decimals := generate_repeating_decimals original_number
  ∀ y ∈ repeating_decimals,
    smallest_repeating_decimal original_number ≤ y ∧
    y ≤ largest_repeating_decimal original_number ∧
    largest_repeating_decimal original_number = 0.20120415 ∧
    smallest_repeating_decimal original_number = 0.20120415 :=
by sorry

-- Additional theorem to state the specific values
theorem specific_values :
  largest_repeating_decimal original_number = 0.20120415 ∧
  smallest_repeating_decimal original_number = 0.20120415 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_bounds_specific_values_l374_37457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_fourth_in_expansion_l374_37475

theorem coefficient_x_fourth_in_expansion : 
  (Polynomial.coeff (Polynomial.X^2 * (1 - Polynomial.X)^6 : Polynomial ℚ) 4) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_fourth_in_expansion_l374_37475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_ranking_sequences_l374_37465

/-- Represents a team in the tournament -/
inductive Team
| A | B | C | D | E

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the tournament structure -/
structure Tournament where
  saturday_match1 : Match
  saturday_match2 : Match
  semifinal_team : Team

/-- Represents a possible ranking sequence -/
def RankingSequence := List Team

/-- Calculates the number of possible ranking sequences -/
def countRankingSequences (t : Tournament) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem tournament_ranking_sequences (t : Tournament) : 
  t.saturday_match1 = ⟨Team.A, Team.B⟩ ∧ 
  t.saturday_match2 = ⟨Team.C, Team.D⟩ ∧ 
  t.semifinal_team = Team.E → 
  countRankingSequences t = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_ranking_sequences_l374_37465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_l374_37432

theorem fraction_sum (x : ℚ) : 
  x / 9 + 3 / 4 + 5 * x / 12 = (19 * x + 27) / 36 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_l374_37432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l374_37477

/-- Properties of an equilateral triangle with side length 10 --/
theorem equilateral_triangle_properties :
  let s : ℝ := 10  -- side length
  let p : ℝ := 3 * s  -- perimeter
  let a : ℝ := (Real.sqrt 3 / 4) * s^2  -- area
  let h_generic : ℝ := (Real.sqrt 3 / 2) * s  -- height using generic property
  let h_heron : ℝ := 2 * a / s  -- height using Heron's formula and area
  (a / p = 5 * Real.sqrt 3 / 6) ∧  -- ratio of area to perimeter
  (h_heron = 5 * Real.sqrt 3) ∧  -- height using Heron's formula
  (h_generic = h_heron)  -- heights are equal
  := by
  sorry

#check equilateral_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l374_37477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_sufficient_not_necessary_for_B_l374_37444

-- Define set A
def A : Set ℝ := {x | |x - 2| < 1}

-- Define set B
def B : Set ℝ := {x | Real.exp (x * Real.log 2) > 1/2}

-- Theorem statement
theorem A_sufficient_not_necessary_for_B :
  (∀ x, x ∈ A → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_sufficient_not_necessary_for_B_l374_37444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_ratio_l374_37495

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the total length of wire needed for a rectangular prism frame -/
def prismWireLength (d : PrismDimensions) : ℝ :=
  4 * (d.length + d.width + d.height)

/-- Calculates the total length of wire needed for a given number of unit cubes -/
def unitCubeWireLength (n : ℝ) : ℝ :=
  12 * n

theorem wire_length_ratio :
  let bonnieDimensions : PrismDimensions := ⟨10, 5, 2⟩
  let bonnieWireLength : ℝ := 8 * 10
  let roarkCubeCount : ℝ := prismVolume bonnieDimensions
  bonnieWireLength / (unitCubeWireLength roarkCubeCount) = 1 / 15 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_ratio_l374_37495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_consecutive_prob_l374_37453

noncomputable def lottery_prob : ℝ :=
  1 - (Nat.choose 86 5 : ℝ) / (Nat.choose 90 5 : ℝ)

theorem lottery_consecutive_prob :
  ∃ (p : ℝ), (abs (p - lottery_prob) < 0.001) ∧ (abs (p - 0.21) < 0.001) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_consecutive_prob_l374_37453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l374_37430

theorem sum_of_angles (α β : Real) (h1 : Real.sin α = Real.sqrt 5 / 5)
  (h2 : Real.sin β = Real.sqrt 10 / 10) (h3 : π / 2 < α) (h4 : α < π)
  (h5 : π / 2 < β) (h6 : β < π) : α + β = 7 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l374_37430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_l374_37464

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The conditions on the polynomial P -/
def satisfies_conditions (P : IntPolynomial) (a : ℤ) : Prop :=
  a > 0 ∧
  P.eval (1 : ℤ) = a ∧ P.eval (3 : ℤ) = a ∧ P.eval (5 : ℤ) = a ∧ P.eval (7 : ℤ) = a ∧ P.eval (9 : ℤ) = a ∧
  P.eval (2 : ℤ) = -a ∧ P.eval (4 : ℤ) = -a ∧ P.eval (6 : ℤ) = -a ∧ P.eval (8 : ℤ) = -a ∧ P.eval (10 : ℤ) = -a

/-- The theorem stating that 1680 is the smallest possible value of a -/
theorem smallest_a : 
  ∀ P : IntPolynomial, ∀ a : ℤ, 
    satisfies_conditions P a → a ≥ 1680 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_l374_37464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_cake_slices_l374_37428

theorem birthday_cake_slices : ∃ (total_slices : ℕ), 
  let eaten_fraction : ℚ := 1/4
  let kept_fraction : ℚ := 1 - eaten_fraction
  let kept_slices : ℕ := 9
  kept_fraction * (total_slices : ℚ) = kept_slices ∧ kept_fraction = 3/4 ∧ total_slices = 12 := by
  use 12
  have eaten_fraction : ℚ := 1/4
  have kept_fraction : ℚ := 1 - eaten_fraction
  have kept_slices : ℕ := 9
  refine ⟨?h1, ?h2, rfl⟩
  · -- Proof that kept_fraction * 12 = kept_slices
    sorry
  · -- Proof that kept_fraction = 3/4
    sorry

#check birthday_cake_slices

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_cake_slices_l374_37428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_comparison_l374_37449

/-- Represents the return on investment for a project --/
structure ProjectReturn where
  profit : ℚ
  loss : ℚ
  noChange : ℚ := 0

/-- Represents the probability distribution for a project's outcomes --/
structure ProjectProbability where
  profitProb : ℚ
  lossProb : ℚ
  noChangeProb : ℚ := 0

/-- Calculates the expected return for a project --/
def expectedReturn (r : ProjectReturn) (p : ProjectProbability) : ℚ :=
  r.profit * p.profitProb + r.loss * p.lossProb + r.noChange * p.noChangeProb

/-- Project A's return and probability --/
def projectA : ProjectReturn × ProjectProbability :=
  (⟨1, -1, 0⟩, ⟨1/2, 1/4, 1/4⟩)

/-- Project B's return (probability is parameterized by α) --/
def projectB (α : ℚ) : ProjectReturn × ProjectProbability :=
  (⟨2, -2, 0⟩, ⟨α, 1 - α, 0⟩)

theorem investment_comparison (α : ℚ) :
  expectedReturn projectA.1 projectA.2 = 1/4 ∧
  (expectedReturn (projectB α).1 (projectB α).2 ≥ 1/4 ↔ 9/16 ≤ α ∧ α ≤ 1) := by
  sorry

#eval expectedReturn projectA.1 projectA.2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_comparison_l374_37449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_melting_l374_37460

theorem ice_melting (original_volume : ℚ) (first_hour_loss_ratio : ℚ) (second_hour_loss_ratio : ℚ) :
  original_volume = 12 →
  first_hour_loss_ratio = 3/4 →
  second_hour_loss_ratio = 3/4 →
  (original_volume * (1 - first_hour_loss_ratio) * (1 - second_hour_loss_ratio)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_melting_l374_37460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l374_37423

-- Define the original function
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

-- Define the translated function
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi/3 + φ)

-- Theorem statement
theorem symmetry_implies_phi_value (φ : ℝ) :
  (∀ x, g φ x = -g φ (-x)) → φ = Real.pi/3 := by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l374_37423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_with_120_degree_inclination_l374_37486

-- Define a structure for Line2D
structure Line2D where
  slope : ℝ
  angle_of_inclination : ℝ

theorem slope_of_line_with_120_degree_inclination :
  ∀ (line : Line2D),
  line.angle_of_inclination = 120 * π / 180 →
  line.slope = -Real.sqrt 3 :=
by
  intro line h
  -- The proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_with_120_degree_inclination_l374_37486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l374_37473

open Real

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The main theorem about the triangle properties -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A ∈ Set.Ioo 0 π)
  (h2 : t.B ∈ Set.Ioo 0 π)
  (h3 : t.a * (cos t.B) = t.b * (cos t.A)) : 
  (t.A = t.B) ∧ 
  (Set.Ioo (-3/2 : ℝ) 0 = { x | ∃ A : ℝ, A ∈ Set.Ioo 0 (π/2) ∧ x = sin (2*A + π/6) - 2*(cos A)^2 }) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l374_37473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_20_max_sum_value_l374_37474

/-- An arithmetic sequence with a₁ = 29 and S₁₀ = S₂₀ -/
noncomputable def arithmetic_seq (n : ℕ) : ℝ := 29 + (n - 1 : ℝ) * 2

/-- Sum of the first n terms of the arithmetic sequence -/
noncomputable def S (n : ℕ) : ℝ := (n : ℝ) * (29 + arithmetic_seq n) / 2

/-- The condition that S₁₀ = S₂₀ -/
axiom S_10_eq_S_20 : S 10 = S 20

/-- The maximum sum occurs at n = 20 -/
theorem max_sum_at_20 : ∀ n : ℕ, S n ≤ S 20 := by
  sorry

/-- The maximum sum is 960 -/
theorem max_sum_value : S 20 = 960 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_20_max_sum_value_l374_37474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l374_37439

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + 2 * (Real.cos x) ^ 2

theorem f_properties :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ M = 1 + Real.sqrt 3) ∧
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ T = Real.pi) ∧
  (∀ (A B C : ℝ) (a b c : ℝ),
    f (C / 2) = 1 →
    c = 2 →
    c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
    ∃ (S : ℝ), S = 1 / 2 * a * b * Real.sin C ∧ S ≤ Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l374_37439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_lines_theorem_l374_37488

noncomputable def F : ℝ × ℝ := (Real.sqrt 3, 0)

def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

def circle_O (x y r : ℝ) : Prop := x^2 + y^2 = r^2

def line_l1 (m n x y : ℝ) : Prop := m * x + n * y = 1
def line_l2 (m n x y : ℝ) : Prop := m * x + n * y = 4

theorem ellipse_and_lines_theorem :
  (∀ k : ℝ, ∃ x y : ℝ, (Real.sqrt 3 * k + 1) * x + (k - Real.sqrt 3) * y - (3 * k + Real.sqrt 3) = 0 ∧ (x, y) = F) →
  (∀ x y : ℝ, ellipse_C x y → (x - F.1)^2 + (y - F.2)^2 ≤ (2 + Real.sqrt 3)^2) →
  (∃ r : ℝ, 1 < r ∧ r < 2 ∧ (∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℝ,
    ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧ ellipse_C x3 y3 ∧ ellipse_C x4 y4 ∧
    circle_O x1 y1 r ∧ circle_O x2 y2 r ∧ circle_O x3 y3 r ∧ circle_O x4 y4 r ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧
    (x2, y2) ≠ (x3, y3) ∧ (x2, y2) ≠ (x4, y4) ∧ (x3, y3) ≠ (x4, y4))) →
  (∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ ellipse_C x y) ∧
  (∀ m n r : ℝ, ellipse_C m n → 1 < r → r < 2 →
    (∃ x y : ℝ, line_l1 m n x y ∧ circle_O x y r) ∧
    (∀ x y : ℝ, line_l2 m n x y → ¬circle_O x y r)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_lines_theorem_l374_37488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_package_price_calculation_l374_37433

/-- Represents the price of a package of bags -/
def package_price : ℚ := 16/5

/-- The number of students in the class -/
def num_students : ℕ := 25

/-- The number of students who want vampire-themed bags -/
def vampire_bags : ℕ := 11

/-- The number of students who want pumpkin-themed bags -/
def pumpkin_bags : ℕ := 14

/-- The cost of an individual bag -/
def individual_bag_cost : ℚ := 1

/-- The total amount spent by the teacher -/
def total_spent : ℚ := 17

/-- The number of packages of vampire-themed bags needed -/
def vampire_packages : ℕ := (vampire_bags + 4) / 5

/-- The number of packages of pumpkin-themed bags needed -/
def pumpkin_packages : ℕ := (pumpkin_bags + 4) / 5

/-- The number of individual vampire-themed bags needed -/
def individual_vampire_bags : ℕ := vampire_bags % 5

theorem package_price_calculation :
  package_price * (vampire_packages + pumpkin_packages : ℚ) + 
  individual_bag_cost * (individual_vampire_bags : ℚ) = total_spent ∧
  package_price = 16/5 := by
  sorry

#eval package_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_package_price_calculation_l374_37433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l374_37461

theorem ellipse_k_range (k : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
    (∀ (x y : ℝ), x^2 + k*y^2 = 4 ↔ (x^2 / (2^2)) + (y^2 / ((2/Real.sqrt k)^2)) = 1) ∧
    (∀ (c : ℝ), c^2 = (2/Real.sqrt k)^2 - 2^2 → c > 0)) →
  0 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l374_37461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_a_l374_37490

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2  -- Define the base case for n = 0
  | n + 1 => (sequence_a n - 4) / 3

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sequence_a n - (-2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_a_l374_37490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebus_solution_l374_37409

theorem rebus_solution : ∃ (a b c d e : ℕ), 
  (Even a ∧ Even c) ∧ 
  (Odd b ∧ Odd d ∧ Odd e) ∧
  (100 * a + 40 + b = 285) ∧
  (10 * c + d = 39) ∧
  ((100 * a + 40 + b) * (10 * c + d) = 11115) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebus_solution_l374_37409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_18_l374_37438

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n, a (n + 1) - a n = d

noncomputable def sum_arithmetic (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_18 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum9 : sum_arithmetic a 9 = 81)
  (h_sum23 : a 2 + a 3 = 8)
  (h_geom : ∃ (r : ℝ), (sum_arithmetic a 3) * (sum_arithmetic a 9) = r^2 ∧ 
                       (sum_arithmetic a 3) * r = a 14 ∧
                       r * (sum_arithmetic a 9) = sum_arithmetic a 9) :
  sum_arithmetic a 18 = 324 := by
  sorry

#check arithmetic_sequence_sum_18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_18_l374_37438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l374_37479

theorem train_speed_problem (train_length crossing_time : Real) : Real := by
  -- Define the conditions
  have h1 : train_length = 120 := by sorry
  have h2 : crossing_time = 24 := by sorry

  -- Define the total distance covered
  let total_distance := 2 * train_length

  -- Define the relative speed
  let relative_speed := total_distance / crossing_time

  -- Define the speed of each train
  let train_speed := relative_speed / 2

  -- Convert speed from m/s to km/hr
  let train_speed_kmh := train_speed * 3.6

  -- Theorem statement
  have : train_speed_kmh = 18 := by sorry

  exact train_speed_kmh


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l374_37479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_l374_37496

-- Define the ellipse C
noncomputable def C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define a point on the ellipse
def PointOnC (p : ℝ × ℝ) : Prop := C p.1 p.2

-- Define a line not passing through (0,1)
def Line (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b ∧ (k * 0 + b ≠ 1)

-- Define the slope of a line passing through two points
noncomputable def Slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- The main theorem
theorem line_passes_through_fixed_point 
  (k b : ℝ) (A B : ℝ × ℝ) : 
  PointOnC A → PointOnC B → 
  Line k b A.1 A.2 → Line k b B.1 B.2 → 
  Slope (0, 1) A + Slope (0, 1) B = -1 → 
  Line k b 2 (-1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_l374_37496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_new_earning_l374_37468

/-- Given an original weekly earning and a percentage increase, 
    calculate the new weekly earning after the raise. -/
noncomputable def new_weekly_earning (original : ℝ) (percentage_increase : ℝ) : ℝ :=
  original * (1 + percentage_increase / 100)

/-- Theorem: John's new weekly earning after a 50% raise from $50 is $75. -/
theorem johns_new_earning :
  new_weekly_earning 50 50 = 75 := by
  -- Unfold the definition of new_weekly_earning
  unfold new_weekly_earning
  -- Simplify the arithmetic
  simp [mul_add, mul_div_right_comm]
  -- Check that 50 * (1 + 50 / 100) = 75
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_new_earning_l374_37468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_problem_l374_37415

theorem complex_power_problem (x y : ℝ) (h : (x - 2) * Complex.I - y = -1 + Complex.I) :
  (1 + Complex.I) ^ (Complex.ofReal (x + y)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_problem_l374_37415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_l374_37447

/-- Represents the rate of the train in miles per second -/
noncomputable def train_rate : ℚ := 2 / 160

/-- Represents the time of 40 minutes in seconds -/
def total_time : ℚ := 40 * 60

/-- Represents the distance traveled by the train in 40 minutes -/
noncomputable def distance_traveled : ℚ := train_rate * total_time

theorem train_distance : distance_traveled = 30 := by
  -- Unfold the definitions
  unfold distance_traveled train_rate total_time
  
  -- Simplify the expression
  simp [mul_div_assoc, mul_comm, mul_assoc]
  
  -- Perform the final calculation
  norm_num
  
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_l374_37447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_prime_square_iff_l374_37404

def p (n : ℤ) : ℤ := n^3 - n^2 - 5*n + 2

def is_prime_square (m : ℤ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ (m = (p : ℤ)^2 ∨ m = (-p : ℤ)^2)

theorem p_prime_square_iff (n : ℤ) :
  is_prime_square ((p n)^2) ↔ n = -3 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_prime_square_iff_l374_37404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_maximum_l374_37431

theorem sequence_sum_maximum : 
  ∃ (m : ℕ), ∀ (k : ℕ), 
    ((-2 : ℤ) * k^3 + 21 * k^2 + 23 * k) ≤ ((-2 : ℤ) * m^3 + 21 * m^2 + 23 * m) ∧ 
    ((-2 : ℤ) * m^3 + 21 * m^2 + 23 * m) = 504 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_maximum_l374_37431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_regular_decagon_l374_37413

/-- The measure of each interior angle of a regular decagon is 144 degrees. -/
theorem interior_angle_regular_decagon : ℝ := by
  -- Define the number of sides in a decagon
  let n : ℕ := 10

  -- Define the sum of exterior angles for any polygon
  let sum_exterior_angles : ℝ := 360

  -- Define the relationship between interior and exterior angles
  let interior_exterior_sum : ℝ := 180

  -- Calculate the measure of each exterior angle
  let exterior_angle : ℝ := sum_exterior_angles / n

  -- Calculate the measure of each interior angle
  let interior_angle : ℝ := interior_exterior_sum - exterior_angle

  -- State and prove the theorem
  have : interior_angle = 144 := by
    -- Proof goes here
    sorry

  -- Return the result
  exact interior_angle


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_regular_decagon_l374_37413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_sixteen_percent_l374_37441

/-- Calculates the annual interest rate given the face value, time period, and true discount of a bill. -/
noncomputable def calculate_interest_rate (face_value : ℝ) (time_months : ℝ) (true_discount : ℝ) : ℝ :=
  let present_value := face_value - true_discount
  let time_years := time_months / 12
  (true_discount * 100) / (present_value * time_years)

/-- Theorem stating that for a bill with face value 1960 Rs, due in 9 months, 
    and a true discount of 210 Rs, the annual interest rate is 16%. -/
theorem interest_rate_is_sixteen_percent :
  calculate_interest_rate 1960 9 210 = 16 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_sixteen_percent_l374_37441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_image_center_theorem_l374_37411

-- Define the types for points and geometric objects
structure Point where
  x : ℝ
  y : ℝ

inductive GeometricObject
  | Circle (center : Point) (radius : ℝ)
  | Line (a : ℝ) (b : ℝ) (c : ℝ)  -- ax + by + c = 0

-- Define the circle k centered at C
def k (C : Point) (radius : ℝ) : GeometricObject :=
  GeometricObject.Circle C radius

-- Define the condition that g does not pass through C
def notPassesThrough (g : GeometricObject) (C : Point) : Prop :=
  match g with
  | GeometricObject.Circle center r => center ≠ C ∧ r ≠ 0
  | GeometricObject.Line a b c => a * C.x + b * C.y + c ≠ 0

-- Define the inverse image of a point with respect to a circle
noncomputable def inverseImage (P : Point) (circle : GeometricObject) : Point :=
  sorry

-- Define the reflection of a point with respect to a line
noncomputable def reflectPoint (P : Point) (line : GeometricObject) : Point :=
  sorry

-- Define the center of the inverse image of g with respect to k
noncomputable def centerOfInverseImage (g : GeometricObject) (k : GeometricObject) : Point :=
  sorry

-- Theorem statement
theorem inverse_image_center_theorem (C : Point) (radius : ℝ) (g : GeometricObject) :
  notPassesThrough g C →
  centerOfInverseImage g (k C radius) =
    match g with
    | GeometricObject.Circle _ _ => inverseImage (inverseImage C g) (k C radius)
    | GeometricObject.Line _ _ _ => inverseImage (reflectPoint C g) (k C radius)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_image_center_theorem_l374_37411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l374_37463

/-- The angle between two curves y = 2x^2 and y = x^3 + 2x^2 - 1 at their intersection point -/
def angle_between_curves : ℝ → Prop :=
  fun φ => ∃ (x y : ℝ),
    -- Point of intersection
    y = 2 * x^2 ∧
    y = x^3 + 2 * x^2 - 1 ∧
    -- Slopes at the intersection point
    let k₁ := (fun x => 4 * x) x
    let k₂ := (fun x => 3 * x^2 + 4 * x) x
    -- Angle formula
    φ = Real.arctan ((k₂ - k₁) / (1 + k₁ * k₂)) ∧
    φ = Real.arctan (3 / 29)

/-- The main theorem stating that the angle between the curves is arctan(3/29) -/
theorem main_theorem : angle_between_curves (Real.arctan (3 / 29)) := by
  sorry

#check main_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l374_37463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_domain_and_range_l374_37498

-- Define the function h
noncomputable def h : ℝ → ℝ := sorry

-- Define the properties of h
axiom h_domain : ∀ x, h x ≠ 0 → -1 ≤ x ∧ x ≤ 3
axiom h_range : ∀ x, 0 ≤ h x ∧ h x ≤ 2

-- Define the function q
noncomputable def q (x : ℝ) : ℝ := 2 - h (x - 2)

-- Theorem statement
theorem q_domain_and_range :
  (∀ x, q x ≠ 0 → 1 ≤ x ∧ x ≤ 5) ∧
  (∀ x, 0 ≤ q x ∧ q x ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_domain_and_range_l374_37498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l374_37403

/-- Represents the scenario of a train journey with an accident -/
structure TrainJourney where
  initialSpeed : ℝ  -- Initial speed of the train in mph
  totalDistance : ℝ  -- Total distance of the trip in miles
  accidentTime : ℝ  -- Time of accident in hours after start
  detentionTime : ℝ  -- Time detained due to accident in hours
  speedReductionFactor : ℝ  -- Factor by which speed is reduced after accident
  lateness : ℝ  -- Hours late the train arrives

/-- Calculates the expected travel time for the actual scenario -/
noncomputable def expectedTravelTime (j : TrainJourney) : ℝ :=
  j.accidentTime + j.detentionTime + (j.totalDistance - j.initialSpeed * j.accidentTime) / (j.initialSpeed * j.speedReductionFactor)

/-- Calculates the expected travel time if the accident happened later -/
noncomputable def expectedTravelTimeLaterAccident (j : TrainJourney) (additionalDistance : ℝ) : ℝ :=
  (j.initialSpeed * j.accidentTime + additionalDistance) / j.initialSpeed + j.detentionTime +
  (j.totalDistance - (j.initialSpeed * j.accidentTime + additionalDistance)) / (j.initialSpeed * j.speedReductionFactor)

/-- The main theorem stating the conditions and the result to be proved -/
theorem train_journey_distance : ∃ (j : TrainJourney),
  j.accidentTime = 1 ∧
  j.detentionTime = 0.75 ∧
  j.speedReductionFactor = 2/3 ∧
  j.lateness = 4 ∧
  expectedTravelTime j = expectedTravelTime j - j.lateness ∧
  expectedTravelTimeLaterAccident j 120 = expectedTravelTimeLaterAccident j 120 - 2.75 ∧
  j.totalDistance = 550 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l374_37403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vinegar_concentration_l374_37407

/-- Given a vinegar solution dilution problem, prove the original concentration. -/
theorem vinegar_concentration (original_amount : ℝ) (water_amount : ℝ) (final_concentration : ℝ) :
  original_amount = 12 →
  water_amount = 50 →
  final_concentration = 7 →
  ∃ (original_concentration : ℝ),
    (original_concentration / 100) * original_amount =
    (final_concentration / 100) * (original_amount + water_amount) ∧
    abs (original_concentration - 36.17) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vinegar_concentration_l374_37407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_weight_after_four_days_l374_37476

noncomputable def initial_weight : ℚ := 80

def halve_weight (w : ℚ) : ℚ := w / 2

def weight_after_days (w : ℚ) : ℕ → ℚ
  | 0 => w
  | n + 1 => halve_weight (weight_after_days w n)

theorem soup_weight_after_four_days :
  weight_after_days initial_weight 4 = 5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_weight_after_four_days_l374_37476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_product_equals_one_l374_37440

-- Define the angles in radians
noncomputable def angle23 : Real := 23 * Real.pi / 180
noncomputable def angle67 : Real := 67 * Real.pi / 180

-- State the theorem
theorem trig_product_equals_one :
  (1 - 1 / Real.cos angle23) * (1 + 1 / Real.sin angle67) * 
  (1 - 1 / Real.sin angle23) * (1 + 1 / Real.cos angle67) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_product_equals_one_l374_37440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_become_fully_weedy_l374_37436

/-- Represents a rectangular field divided into plots -/
structure RectangularField where
  plots : ℕ
  weedy_plots : ℕ
  free_sides : ℕ

/-- The spreading rule for weeds -/
def can_spread (f : RectangularField) : Prop :=
  f.free_sides ≥ 2 * (Nat.ceil (Real.sqrt (f.plots : ℝ)))

/-- The initial state of the field -/
def initial_field : RectangularField :=
  { plots := 100,
    weedy_plots := 9,
    free_sides := 36 }

/-- Theorem stating that the entire field cannot become weedy -/
theorem cannot_become_fully_weedy (f : RectangularField) :
  f = initial_field → ¬(can_spread f) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_become_fully_weedy_l374_37436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_approximation_l374_37494

noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * (2 * Real.pi * r)

theorem circle_radius_approximation (L : ℝ) (θ : ℝ) (h1 : L = 2000) (h2 : θ = 300) :
  Int.toNat (Int.floor ((L * 360) / (θ * 2 * Real.pi) + 0.5)) = 382 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_approximation_l374_37494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_count_l374_37458

def count_sequences (f : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n > 0 → f (f n) = 4 * n) ∧
  (∀ n : ℕ, n > 0 → f (n + 1) > f n) ∧
  (∀ n : ℕ, n > 0 → f n > 0)

theorem sequence_count :
  ∃! count : ℕ, ∀ f : ℕ → ℕ,
    count_sequences f →
    (∃! seq : Fin 16 → ℕ, (∀ i : Fin 16, seq i = f (i.val.succ)) ∧ count = 118) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_count_l374_37458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cut_theorem_l374_37410

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the result of cutting a rectangle in half along its length -/
noncomputable def cut_rectangle_in_half (r : Rectangle) : Rectangle :=
  { length := r.length / 2, width := r.width }

theorem rectangle_cut_theorem (original : Rectangle) 
  (h1 : original.length = 10)
  (h2 : original.width = 6) :
  let new_rectangle := cut_rectangle_in_half original
  new_rectangle.length = 5 ∧ new_rectangle.width = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cut_theorem_l374_37410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_inequalities_l374_37452

theorem logarithm_inequalities :
  (Real.log 3 / Real.log 2 > Real.log 4 / Real.log 3) ∧
  ((2022 : Real) ^ (Real.log 2233 / Real.log 10) > (2023 : Real) ^ (Real.log 2022 / Real.log 10)) ∧
  (2 ^ (Real.log 2 / Real.log 10) + 2 ^ (Real.log 5 / Real.log 10) > 2 * Real.sqrt 2) ∧
  (Real.log 3 + 4 / Real.log 3 > 2 * Real.log 2 + 2 / Real.log 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_inequalities_l374_37452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalent_l374_37418

/-- A rectangle is a quadrilateral with four right angles. -/
structure Rectangle where
  /-- The length of the rectangle -/
  length : ℝ
  /-- The width of the rectangle -/
  width : ℝ
  /-- Assumption that length and width are positive -/
  length_pos : length > 0
  width_pos : width > 0

/-- The length of a diagonal in a rectangle -/
noncomputable def diagonal (r : Rectangle) : ℝ :=
  Real.sqrt (r.length ^ 2 + r.width ^ 2)

/-- The statement "The diagonals of a rectangle are equal" -/
def diagonals_equal : Prop :=
  ∀ r : Rectangle, diagonal r = diagonal r

/-- The negation of "The diagonals of a rectangle are equal" -/
def negation_diagonals_equal : Prop :=
  ∃ r : Rectangle, diagonal r ≠ diagonal r

/-- Theorem stating that the negation of "The diagonals of a rectangle are equal"
    is equivalent to "There exists a rectangle whose diagonals are not equal" -/
theorem negation_equivalent :
  ¬diagonals_equal ↔ negation_diagonals_equal := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalent_l374_37418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_descent_time_is_two_hours_l374_37443

/-- Represents a mountain climbing scenario with two routes --/
structure MountainClimb where
  shortRoute : ℝ
  longRoute : ℝ
  uphillSpeed : ℝ
  downhillSpeed : ℝ
  totalTime : ℝ

/-- Calculates the time taken to come down the mountain --/
noncomputable def timeToDescend (climb : MountainClimb) : ℝ :=
  climb.longRoute / climb.downhillSpeed

/-- Theorem stating that under given conditions, the time to descend is 2 hours --/
theorem descent_time_is_two_hours (climb : MountainClimb) :
  climb.longRoute = climb.shortRoute + 2 ∧
  climb.uphillSpeed = 3 ∧
  climb.downhillSpeed = 4 ∧
  climb.totalTime = 4 ∧
  climb.shortRoute / climb.uphillSpeed + climb.longRoute / climb.downhillSpeed = climb.totalTime
  →
  timeToDescend climb = 2 := by
  sorry

#check descent_time_is_two_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_descent_time_is_two_hours_l374_37443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_into_isosceles_l374_37420

-- Define a triangle type
structure Triangle where
  -- Add necessary fields for a triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Add triangle inequality constraints
  hab : a + b > c
  hbc : b + c > a
  hca : c + a > b

-- Define an isosceles triangle type
structure IsoscelesTriangle extends Triangle where
  -- Add condition for isosceles triangle
  is_isosceles : (a = b) ∨ (b = c) ∨ (c = a)

-- Define a function to represent the division of a triangle into isosceles triangles
def can_divide_into_isosceles (t : Triangle) (n : ℕ) : Prop :=
  ∃ (divisions : Fin n → IsoscelesTriangle), 
    -- Additional conditions to ensure the divisions form the original triangle
    True  -- Placeholder, replace with actual conditions

-- State the theorem
theorem triangle_division_into_isosceles (t : Triangle) (n : ℕ) (h : n ≥ 4) :
  can_divide_into_isosceles t n := by
  sorry  -- The proof is omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_into_isosceles_l374_37420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l374_37489

theorem trigonometric_inequality (α β γ : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : 0 < γ ∧ γ < π/2)
  (h4 : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  (Real.sin α ^ 3 / Real.sin β) + (Real.sin β ^ 3 / Real.sin γ) + (Real.sin γ ^ 3 / Real.sin α) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l374_37489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_invariant_l374_37491

def transform (t : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (a, b, c) := t
  (b + c, a + c, a + b)

def initial_triplet : ℕ × ℕ × ℕ := (70, 61, 20)

def difference (t : ℕ × ℕ × ℕ) : ℕ :=
  let (a, b, c) := t
  max a (max b c) - min a (min b c)

theorem difference_invariant (n : ℕ) :
  difference (Nat.iterate transform n initial_triplet) = difference initial_triplet :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_invariant_l374_37491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_product_ABC_l374_37405

-- Define the original expression
noncomputable def original_expression : ℝ := (2 + Real.sqrt 5) / (3 - Real.sqrt 5)

-- Define the rationalized and simplified expression
noncomputable def rationalized_expression : ℝ := 2 + 5 * Real.sqrt 5

-- Theorem statement
theorem rationalize_denominator :
  original_expression = rationalized_expression := by
  sorry

-- Define A, B, and C as integers
def A : ℤ := 2
def B : ℤ := 5
def C : ℤ := 5

-- Theorem to prove the product ABC
theorem product_ABC :
  A * B * C = 50 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_product_ABC_l374_37405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l374_37469

noncomputable def f (x a : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + Real.sqrt 3 * Real.sin (2 * x) + 2 * a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x a ≥ 0) →
  (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x a = 0) →
  (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x a = 1/2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x a ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l374_37469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_sixteen_to_fourth_power_l374_37412

theorem fourth_root_of_sixteen_to_fourth_power : (16 : ℝ) ^ (1/4) ^ 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_sixteen_to_fourth_power_l374_37412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_x_plus_20_l374_37480

theorem cube_root_of_x_plus_20 (x : ℝ) (h : Real.sqrt (x + 2) = 3) :
  (x + 20) ^ (1/3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_x_plus_20_l374_37480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardboard_prism_diagonal_l374_37419

/-- The length of the diagonal of a right rectangular prism. -/
noncomputable def prism_diagonal (a b c : ℝ) : ℝ := Real.sqrt (a^2 + b^2 + c^2)

/-- The possible diagonal lengths of a right rectangular prism formed from an 8cm by 4cm rectangular cardboard. -/
def possible_diagonals : Set ℝ := {2 * Real.sqrt 6, Real.sqrt 66}

/-- Theorem: The diagonal of a right rectangular prism formed from an 8cm by 4cm rectangular cardboard is either 2√6 cm or √66 cm. -/
theorem cardboard_prism_diagonal :
  ∀ l w h : ℝ,
  (l * w = 32 ∧ (l = 8 ∧ w = 4 ∨ l = 4 ∧ w = 8)) →
  (h = 4 ∨ h = 8) →
  prism_diagonal (l/4) (w/4) h ∈ possible_diagonals :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardboard_prism_diagonal_l374_37419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l374_37462

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 then x^3 + x + 1 
  else if x < 0 then x^3 + x - 1 
  else 0  -- We define f(0) = 0 to make it total

-- State the theorem
theorem odd_function_property : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x : ℝ, x > 0 → f x = x^3 + x + 1) → 
  (∀ x : ℝ, x < 0 → f x = x^3 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l374_37462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_is_correct_l374_37417

/-- The coefficient of x^3 in the expansion of ((x - 2/x)^5) -/
def coefficient_x_cubed : ℤ := -10

/-- The binomial expression ((x - 2/x)^5) -/
noncomputable def binomial_expression (x : ℝ) : ℝ := (x - 2/x)^5

/-- Theorem stating that the coefficient of x^3 in the expansion is correct -/
theorem coefficient_x_cubed_is_correct :
  ∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → binomial_expression x = f x) ∧
  (∃ (g : ℝ → ℝ), (∀ x, f x = coefficient_x_cubed * x^3 + x * g x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_is_correct_l374_37417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_comparison_l374_37467

theorem sqrt_difference_comparison : Real.sqrt 11 - Real.sqrt 12 < Real.sqrt 12 - Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_comparison_l374_37467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l374_37434

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin (2 * x - Real.pi / 6) - 1 / 2

noncomputable def g (x : ℝ) := f (x - Real.pi / 12)

theorem function_and_range_proof :
  (∀ x ∈ Set.Icc (5 * Real.pi / 12) (7 * Real.pi / 6), f x ≤ 1) ∧
  (Set.Icc (-2) 1 = {m : ℝ | ∀ x ∈ Set.Icc 0 (Real.pi / 3), g x - 3 ≤ m ∧ m ≤ g x + 3}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l374_37434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l374_37401

-- Define the triangle and its properties
structure RightTriangle where
  DE : ℝ
  DF : ℝ
  EF : ℝ
  is_right : DE^2 + DF^2 = EF^2

-- Define the points Z and W
noncomputable def Z (t : RightTriangle) : ℝ := t.DE / 4
noncomputable def W (t : RightTriangle) : ℝ := t.DF / 4

-- State the theorem
theorem hypotenuse_length (t : RightTriangle) 
  (h1 : (t.DE - Z t)^2 + t.DF^2 = 18^2)
  (h2 : (t.DF - W t)^2 + t.DE^2 = 24^2) :
  t.EF = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l374_37401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_not_sufficient_nor_necessary_l374_37427

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The partial sum sequence of a given sequence -/
def PartialSum (a : Sequence) : Sequence :=
  fun n => (Finset.range n).sum a

/-- A sequence is increasing -/
def IsIncreasing (s : Sequence) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

theorem increasing_sequence_not_sufficient_nor_necessary :
  ¬(∀ a : Sequence, IsIncreasing a → IsIncreasing (PartialSum a)) ∧
  ¬(∀ a : Sequence, IsIncreasing (PartialSum a) → IsIncreasing a) := by
  sorry

#check increasing_sequence_not_sufficient_nor_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_not_sufficient_nor_necessary_l374_37427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_proportions_l374_37445

theorem lemonade_proportions (lemons_60 sugar_60 gallons_60 gallons_15 : ℚ)
  (h1 : lemons_60 = 40)
  (h2 : sugar_60 = 80)
  (h3 : gallons_60 = 60)
  (h4 : gallons_15 = 15) :
  (lemons_60 * gallons_15 / gallons_60 = 10) ∧ 
  (sugar_60 * gallons_15 / gallons_60 = 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_proportions_l374_37445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_defined_iff_x_neq_one_l374_37424

/-- The expression e^(3-x) / (1/(x-1)) is defined if and only if x ≠ 1 -/
theorem expression_defined_iff_x_neq_one (x : ℝ) :
  (∃ y : ℝ, y = Real.exp (3 - x) / (1 / (x - 1))) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_defined_iff_x_neq_one_l374_37424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_30_factorial_is_10_l374_37442

/-- The number of distinct prime factors of 30! -/
def distinct_prime_factors_30_factorial : ℕ := 10

/-- 30! is the product of all positive integers from 1 to 30 -/
def factorial_30 : ℕ := (Finset.range 30).prod (λ i => i + 1)

/-- Theorem: The number of distinct prime factors of 30! is 10 -/
theorem distinct_prime_factors_30_factorial_is_10 :
  (Nat.factors factorial_30).toFinset.card = distinct_prime_factors_30_factorial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_30_factorial_is_10_l374_37442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_problem_l374_37478

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 30) (h3 : x - y = 10) :
  ∃ y', x = 4 → y' = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_problem_l374_37478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_lake_speed_l374_37421

/-- Represents the speed of a boat in various conditions --/
structure BoatSpeed where
  lake : ℝ
  upstream : ℝ
  downstream : ℝ

/-- Represents the distances traveled by the boat --/
structure BoatDistance where
  lake : ℝ
  upstream : ℝ
  downstream : ℝ

/-- Calculates the average speed given total distance and total time --/
noncomputable def averageSpeed (totalDistance : ℝ) (totalTime : ℝ) : ℝ :=
  totalDistance / totalTime

/-- Theorem stating the boat's speed across the lake --/
theorem boat_lake_speed (speed : BoatSpeed) (distance : BoatDistance) 
  (h1 : speed.upstream = 4)
  (h2 : speed.downstream = 6)
  (h3 : distance.upstream = 2 * distance.lake)
  (h4 : distance.downstream = distance.lake)
  (h5 : averageSpeed (distance.lake + distance.upstream + distance.downstream) 
    ((distance.lake / speed.lake) + (distance.upstream / speed.upstream) + (distance.downstream / speed.downstream)) = 3.6) :
  ∃ ε > 0, |speed.lake - 2.25| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_lake_speed_l374_37421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l374_37497

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1)
noncomputable def g (x m : ℝ) : ℝ := (1/2)^x - m

-- State the theorem
theorem m_range_theorem (m : ℝ) : 
  (∀ x₁ ∈ Set.Icc 0 3, ∀ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g x₂ m) ↔ 
  m ∈ Set.Ici (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l374_37497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l374_37482

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y + 3 = 0

-- Define the centers and radii of the circles
def center₁ : ℝ × ℝ := (1, 0)
def center₂ : ℝ × ℝ := (-2, -1)
def radius₁ : ℝ := 2

noncomputable def radius₂ : ℝ := Real.sqrt 2

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 10

-- Theorem stating that the circles intersect
theorem circles_intersect :
  distance_between_centers > abs (radius₁ - radius₂) ∧
  distance_between_centers < radius₁ + radius₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l374_37482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_implies_m_range_l374_37456

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m-3 ≤ x ∧ x ≤ m+3}

-- Part 1
theorem intersection_implies_m_value (m : ℝ) :
  A ∩ B m = Set.Icc 2 3 → m = 5 := by sorry

-- Part 2
theorem subset_complement_implies_m_range (m : ℝ) :
  A ⊆ (B m)ᶜ → m < -4 ∨ m > 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_implies_m_range_l374_37456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l374_37485

-- Define the spherical coordinates
noncomputable def ρ : ℝ := 3
noncomputable def θ : ℝ := Real.pi / 3
noncomputable def φ : ℝ := Real.pi / 6

-- Define the conversion functions
noncomputable def x : ℝ := ρ * Real.sin φ * Real.cos θ
noncomputable def y : ℝ := ρ * Real.sin φ * Real.sin θ
noncomputable def z : ℝ := ρ * Real.cos φ

-- Theorem statement
theorem spherical_to_rectangular_conversion :
  (x, y, z) = (3/4, 3*Real.sqrt 3/4, 3*Real.sqrt 3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l374_37485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_four_seat_more_cost_effective_l374_37429

/-- Represents the number of people on the trip -/
def total_people : ℕ := sorry

/-- Represents the number of 48-seat buses needed -/
def buses_48 : ℕ := sorry

/-- Represents the number of 64-seat buses needed -/
def buses_64 : ℕ := sorry

/-- The cost of renting a 48-seat bus -/
def cost_48 : ℕ := 260

/-- The cost of renting a 64-seat bus -/
def cost_64 : ℕ := 300

/-- Assumption: The number of people can fill 48-seat buses exactly -/
axiom fill_48 : total_people = 48 * buses_48

/-- Assumption: 64-seat buses require one less bus than 48-seat buses -/
axiom less_64 : buses_64 = buses_48 - 1

/-- Assumption: With 64-seat buses, one bus is not fully occupied but more than half full -/
axiom partially_full_64 : 32 < total_people - 64 * (buses_64 - 1) ∧ total_people - 64 * (buses_64 - 1) < 64

/-- The total cost of renting 48-seat buses -/
def total_cost_48 : ℕ := buses_48 * cost_48

/-- The total cost of renting 64-seat buses -/
def total_cost_64 : ℕ := buses_64 * cost_64

/-- Theorem: 64-seat buses are more cost-effective -/
theorem sixty_four_seat_more_cost_effective : total_cost_64 < total_cost_48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_four_seat_more_cost_effective_l374_37429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_lowest_terms_count_l374_37426

theorem not_lowest_terms_count : 
  (Finset.filter (fun N : ℕ => Nat.gcd (N^3 + 11) (N + 5) > 1) 
    (Finset.range 2000)).card = 164 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_lowest_terms_count_l374_37426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_bound_l374_37481

theorem subset_intersection_bound {n : ℕ} {A : Finset ℕ} {m : ℕ} 
  (h_card_A : A.card = 2 * n)
  (A_subsets : Fin m → Finset ℕ)
  (h_subset : ∀ i : Fin m, A_subsets i ⊆ A)
  (h_card_subset : ∀ i : Fin m, (A_subsets i).card = n)
  (h_intersection : ∀ i j k : Fin m, i ≠ j → j ≠ k → i ≠ k → 
    ((A_subsets i) ∩ (A_subsets j) ∩ (A_subsets k)).card ≤ 1) :
  (m % 2 = 0 → m ≤ 4 * (n - 1) / (n - 4)) ∧
  (m % 2 = 1 → m ≤ (3 * n - 2) / (n - 4)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_bound_l374_37481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lattice_points_l374_37406

/-- The number of lattice points at distance 5 from the origin with x = 1 -/
def lattice_points_count : ℕ := 24

/-- A lattice point at distance 5 from the origin with x = 1 -/
structure LatticePoint where
  y : ℤ
  z : ℤ
  dist_eq : (1 : ℤ)^2 + y^2 + z^2 = 25

def lattice_points : List LatticePoint := [
  ⟨0, 4, by sorry⟩,
  ⟨0, -4, by sorry⟩,
  ⟨4, 0, by sorry⟩,
  ⟨-4, 0, by sorry⟩,
  ⟨2, 4, by sorry⟩,
  ⟨2, -4, by sorry⟩,
  ⟨-2, 4, by sorry⟩,
  ⟨-2, -4, by sorry⟩,
  ⟨4, 2, by sorry⟩,
  ⟨4, -2, by sorry⟩,
  ⟨-4, 2, by sorry⟩,
  ⟨-4, -2, by sorry⟩,
  ⟨2, 2, by sorry⟩,
  ⟨2, -2, by sorry⟩,
  ⟨-2, 2, by sorry⟩,
  ⟨-2, -2, by sorry⟩
]

theorem count_lattice_points :
  lattice_points.length = lattice_points_count ∧
  ∀ p : LatticePoint, p ∈ lattice_points := by
  sorry

#check count_lattice_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lattice_points_l374_37406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_length_proof_l374_37425

/-- The length of fence that can be painted by a given number of boys in a certain time. -/
def fenceLength (boys : ℕ) (days : ℚ) (rate : ℚ) : ℚ :=
  rate * (boys : ℚ) * days

theorem fence_length_proof (rate : ℚ) :
  (5 : ℚ) * rate * (1.8 : ℚ) = 30 →
  fenceLength 2 3 rate = 20 := by
  intro h
  -- The proof steps would go here
  sorry

#check fence_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_length_proof_l374_37425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_l374_37454

/-- An arithmetic progression with a first term and common difference -/
structure ArithmeticProgression where
  a : ℝ  -- first term
  d : ℝ  -- common difference

/-- The nth term of an arithmetic progression -/
noncomputable def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.a + (n - 1 : ℝ) * ap.d

/-- The sum of the first n terms of an arithmetic progression -/
noncomputable def ArithmeticProgression.sum (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * ap.a + (n - 1 : ℝ) * ap.d)

/-- Theorem: In an arithmetic progression where the sum of the first 15 terms is 225,
    the sum of the 4th and 12th terms is 30 -/
theorem arithmetic_progression_sum (ap : ArithmeticProgression) :
  ap.sum 15 = 225 →
  ap.nthTerm 4 + ap.nthTerm 12 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_l374_37454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l374_37459

-- Define the train's length in meters
noncomputable def train_length : ℝ := 450

-- Define the train's speed in km/hr
noncomputable def train_speed_kmh : ℝ := 120

-- Convert km/hr to m/s
noncomputable def km_to_m_per_s (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

-- Calculate the time taken for the train to pass the pole
noncomputable def time_to_pass_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  length / km_to_m_per_s speed_kmh

-- Theorem statement
theorem train_passing_pole_time :
  ∃ ε > 0, abs (time_to_pass_pole train_length train_speed_kmh - 13.5) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l374_37459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_minimum_value_condition_monotonicity_condition_l374_37472

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

-- Part I
theorem tangent_line_at_one (a : ℝ) :
  a = 1 → (∃ (m : ℝ), ∀ x, m * (x - 1) - 2 = f 1 x - f 1 1) := by
  sorry

-- Part II
theorem minimum_value_condition (a : ℝ) :
  a > 0 → (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ -2) → (f a 1 = -2) → a ≥ 1 := by
  sorry

-- Part III
theorem monotonicity_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f a x₁ + 2*x₁ < f a x₂ + 2*x₂) →
  0 ≤ a ∧ a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_minimum_value_condition_monotonicity_condition_l374_37472
