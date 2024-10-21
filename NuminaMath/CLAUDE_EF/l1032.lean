import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_impossibility_l1032_103204

theorem partition_impossibility :
  ¬ ∃ (partition : Finset (Finset Nat)),
    (partition.card = 11) ∧
    (∀ group, group ∈ partition → group.card = 3) ∧
    (∀ group, group ∈ partition → ∃ a b c, a ∈ group ∧ b ∈ group ∧ c ∈ group ∧ a + b = c) ∧
    (partition.biUnion id = Finset.range 33) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_impossibility_l1032_103204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_positive_y_negative_l1032_103294

-- Define the function y as noncomputable
noncomputable def y (x : ℝ) : ℝ := Real.log (1 / (x + 3)) / Real.log (1 / Real.sqrt 2)

-- Theorem for y > 0
theorem y_positive (x : ℝ) : y x > 0 ↔ x > -2 := by sorry

-- Theorem for y < 0
theorem y_negative (x : ℝ) : y x < 0 ↔ -3 < x ∧ x < -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_positive_y_negative_l1032_103294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1032_103248

/-- The angle between two vectors in radians -/
noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 3)
  (h2 : a.1^2 + a.2^2 = 1)
  (h3 : b = (1, 1)) :
  angle_between a b = π * 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1032_103248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1032_103254

noncomputable def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def UpperVertex (a b : ℝ) : ℝ × ℝ := (0, b)

noncomputable def Distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ p ∈ Ellipse a b, Distance p (UpperVertex a b) ≤ 2 * b) →
  (0 < Eccentricity a b ∧ Eccentricity a b ≤ Real.sqrt 2 / 2) := by
  sorry

#check ellipse_eccentricity_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1032_103254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_c_draws_l1032_103272

structure Team where
  games_played : ℕ
  win_percentage : ℚ
  draw_percentage : ℚ
  loss_percentage : ℚ

def calculate_draws (team : Team) : ℕ :=
  team.games_played - 
  (team.win_percentage * ↑team.games_played).floor.toNat - 
  (team.loss_percentage * ↑team.games_played).floor.toNat

def team_c : Team := {
  games_played := 210
  win_percentage := 45/100
  draw_percentage := 15/100  -- Calculated from remaining percentage
  loss_percentage := 40/100
}

theorem team_c_draws : calculate_draws team_c = 31 := by
  unfold calculate_draws
  unfold team_c
  simp
  norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_c_draws_l1032_103272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_five_twentyfourths_l1032_103229

/-- Configuration of three mutually tangent semi-circles -/
structure SemiCircleConfig where
  AB : ℝ
  AC : ℝ
  CB : ℝ
  CD : ℝ
  h_AB : AB = AC + CB
  h_tangent : AC * CB = CD^2
  h_positive : AB > 0 ∧ AC > 0 ∧ CB > 0 ∧ CD > 0

/-- The ratio of shaded area to the area of circle with CD as radius -/
noncomputable def areaRatio (config : SemiCircleConfig) : ℝ :=
  let largestSemiCircle := Real.pi * (config.AB / 2)^2 / 2
  let smallerSemiCircle1 := Real.pi * (config.AC / 2)^2 / 2
  let smallerSemiCircle2 := Real.pi * (config.CB / 2)^2 / 2
  let shadedArea := largestSemiCircle - smallerSemiCircle1 - smallerSemiCircle2
  let circleArea := Real.pi * config.CD^2
  shadedArea / circleArea

/-- The main theorem stating the ratio is 5/24 -/
theorem area_ratio_is_five_twentyfourths (config : SemiCircleConfig) :
  areaRatio config = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_five_twentyfourths_l1032_103229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sin_cos_l1032_103208

theorem max_distance_sin_cos (m : ℝ) : 
  |Real.sin m - Real.sin (π/2 - m)| ≤ Real.sqrt 2 ∧ 
  ∃ m₀ : ℝ, |Real.sin m₀ - Real.sin (π/2 - m₀)| = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sin_cos_l1032_103208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1032_103252

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  D : Real × Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.C + (2 * t.b + t.c) * Real.cos t.A = 0 ∧
  t.D = ((t.B + t.C) / 2, 0) ∧  -- Assuming BC is on x-axis for simplicity
  Real.sqrt ((t.D.1 - t.A)^2 + t.D.2^2) = 7/2 ∧
  t.c = 3

-- State the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.A = 2 * Real.pi / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A = 6 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1032_103252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l1032_103266

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - Complex.I)^2 * (z + Complex.I)) ≤ 5 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l1032_103266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_good_arrangements_l1032_103287

def is_good_arrangement (x : Fin 12 → ℝ) : Prop :=
  x 0 = 1 ∧ x 11 = 1 ∧ x 1 = x 2 ∧
  ∀ n : Fin 9, (x n + x (n + 3)) / 2 = x (n + 1) * x (n + 2)

-- Define the set of good arrangements
def good_arrangements : Set (Fin 12 → ℝ) :=
  {x | is_good_arrangement x}

-- State the theorem without using Fintype.card
theorem count_good_arrangements :
  ∃! (count : ℕ), (∃ (f : good_arrangements → Fin count), Function.Bijective f) ∧ count = 89 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_good_arrangements_l1032_103287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_second_quadrant_l1032_103290

def second_quadrant (α : ℝ) : Prop :=
  Real.sin α > 0 ∧ Real.cos α < 0

theorem trigonometric_expression_second_quadrant (α : ℝ) 
  (h : second_quadrant α) : 
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) + 
  Real.sqrt (1 - Real.sin α ^ 2) / Real.cos α = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_second_quadrant_l1032_103290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_totient_second_solution_l1032_103233

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- Theorem: If Euler's totient function has one solution, it always has a second solution -/
theorem totient_second_solution (n : ℕ) :
  (∃ x₁ : ℕ, phi x₁ = n) → (∃ x₂ : ℕ, phi x₂ = n ∧ x₂ ≠ x₁) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_totient_second_solution_l1032_103233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l1032_103257

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The distance from a point to a line ax + by + c = 0 -/
noncomputable def distanceToLine (p : Point2D) (a b c : ℝ) : ℝ :=
  |a * p.x + b * p.y + c| / Real.sqrt (a^2 + b^2)

/-- The definition of an ellipse -/
def isEllipse (P : Point2D → Prop) : Prop :=
  ∃ (f : Point2D) (a b c : ℝ),
    ∀ p, P p ↔ distance p f = 0.5 * distanceToLine p a b c

/-- The given equation for point P -/
def satisfiesEquation (p : Point2D) : Prop :=
  10 * distance p ⟨1, 2⟩ = |3 * p.x + 4 * p.y + 2|

/-- The theorem to prove -/
theorem trajectory_is_ellipse :
  isEllipse satisfiesEquation := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l1032_103257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_order_change_l1032_103230

-- Define the function f
variable (f : ℝ → ℝ → ℝ)

-- Define the original integral
noncomputable def original_integral (f : ℝ → ℝ → ℝ) : ℝ :=
  ∫ y in Set.Icc 0 1, (∫ x in Set.Icc 0 (Real.sqrt y), f x y) +
  ∫ y in Set.Icc 1 (Real.sqrt 2), (∫ x in Set.Icc 0 (Real.sqrt (2 - y^2)), f x y)

-- Define the changed integral
noncomputable def changed_integral (f : ℝ → ℝ → ℝ) : ℝ :=
  ∫ x in Set.Icc 0 1, (∫ y in Set.Icc (x^2) (Real.sqrt (2 - x^2)), f x y)

-- Theorem statement
theorem integral_order_change (f : ℝ → ℝ → ℝ) :
  original_integral f = changed_integral f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_order_change_l1032_103230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_med_b_amount_is_425_l1032_103286

/-- Represents the composition of a medication mixture -/
structure MedicationMixture where
  total : ℚ
  med_a_percent : ℚ
  med_b_percent : ℚ
  total_painkiller : ℚ

/-- Calculates the amount of medication B in the mixture -/
def amount_of_med_b (m : MedicationMixture) : ℚ :=
  m.total - (m.total_painkiller - m.med_b_percent * m.total) / (m.med_a_percent - m.med_b_percent)

/-- Theorem stating the amount of medication B in the given mixture -/
theorem med_b_amount_is_425 (m : MedicationMixture) 
  (h1 : m.total = 750)
  (h2 : m.med_a_percent = 2/5)
  (h3 : m.med_b_percent = 1/5)
  (h4 : m.total_painkiller = 215) :
  amount_of_med_b m = 425 := by
  sorry

#eval amount_of_med_b { total := 750, med_a_percent := 2/5, med_b_percent := 1/5, total_painkiller := 215 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_med_b_amount_is_425_l1032_103286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_equals_one_range_of_a_for_g_leq_four_l1032_103226

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a| + |x - 1|

-- Define the function g
noncomputable def g (a : ℝ) : ℝ := f a (1/a)

-- Theorem for part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for part 2
theorem range_of_a_for_g_leq_four :
  {a : ℝ | a ≠ 0 ∧ g a ≤ 4} = {a : ℝ | 1/2 ≤ a ∧ a ≤ 3/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_equals_one_range_of_a_for_g_leq_four_l1032_103226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l1032_103251

theorem tan_value_from_sin_cos_sum (θ : ℝ) (h1 : θ ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.sin θ + Real.cos θ = 17/13) : Real.tan θ = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l1032_103251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l1032_103238

-- Define an intelligent function
def is_intelligent (f : ℝ → ℝ) : Prop := ∃ m : ℝ, f m = m

-- Define a quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Part 1
theorem part1 : is_intelligent (λ x => 2 * x - 3) := by sorry

-- Part 2
theorem part2 (a b c : ℝ) (x₁ x₂ : ℝ) 
  (h1 : quadratic a b c x₁ = 0) 
  (h2 : quadratic a b c x₂ = 0)
  (h3 : x₁ * x₂ + x₁ + x₂ = -2 / a) :
  (∀ b : ℝ, is_intelligent (quadratic a b c)) ↔ (0 < a ∧ a ≤ 1) := by sorry

-- Part 3
theorem part3 (a b c : ℝ) (x₁ x₂ : ℝ) 
  (h1 : quadratic a b c x₁ = 0) 
  (h2 : quadratic a b c x₂ = 0)
  (h3 : x₁ * x₂ + x₁ + x₂ = -2 / a)
  (h4 : 0 < a ∧ a ≤ 1)
  (h5 : ∃ m₁ m₂ : ℝ, is_intelligent (quadratic a b c) ∧ 
                     quadratic a b c m₁ = m₁ ∧ 
                     quadratic a b c m₂ = m₂ ∧
                     m₁ + m₂ = 2 * (a / (2 * a^2 + a + 1))) :
  b ≥ 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l1032_103238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l1032_103278

theorem min_value_of_function :
  ∃ (min_val : ℝ), 
    (∀ x > 2, x + 9 / (x - 2) ≥ min_val) ∧ 
    (∃ z > 2, z + 9 / (z - 2) = min_val) ∧ 
    min_val = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l1032_103278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_77_l1032_103206

theorem floor_sqrt_77 : ⌊Real.sqrt 77⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_77_l1032_103206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_identical_grouping_l1032_103296

/-- Given a natural number k, it's impossible to divide the set of natural numbers
    from 1 to k into two non-empty groups such that when the numbers in each group
    are listed in some order, the resulting sequences are identical. -/
theorem no_identical_grouping (k : ℕ) : ¬ ∃ (A B : Finset ℕ),
  (A ∪ B = Finset.range (k + 1) \ {0}) ∧
  (A ∩ B = ∅) ∧
  (A.Nonempty) ∧
  (B.Nonempty) ∧
  (∃ (f : ℕ → ℕ) (g : ℕ → ℕ),
    (∀ n, n < A.card → f n ∈ A) ∧
    (∀ n, n < B.card → g n ∈ B) ∧
    (∀ n, n < A.card → f n = g n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_identical_grouping_l1032_103296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_with_two_rational_points_l1032_103298

-- Define a rational point
def RationalPoint (p : ℚ × ℚ) : Prop := True

-- Define a line passing through a point
def LineThrough (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  p ∈ l ∧ ∃ (m b : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ y = m * x + b

-- Convert rational point to real point
def toReal (p : ℚ × ℚ) : ℝ × ℝ := (↑p.1, ↑p.2)

-- Theorem statement
theorem unique_line_with_two_rational_points (a : ℝ) (h : Irrational a) :
  ∃! (l : Set (ℝ × ℝ)), LineThrough (a, 0) l ∧
    (∃ (p₁ p₂ : ℚ × ℚ), toReal p₁ ≠ toReal p₂ ∧ 
      toReal p₁ ∈ l ∧ toReal p₂ ∈ l ∧ RationalPoint p₁ ∧ RationalPoint p₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_with_two_rational_points_l1032_103298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_llama_roam_area_specific_l1032_103288

/-- The area a llama can roam when tethered to a shed corner -/
noncomputable def llamaRoamArea (shedWidth shedLength leashLength : ℝ) : ℝ :=
  (3/4) * Real.pi * leashLength^2 + (1/2) * Real.pi * (leashLength - shedWidth)^2

/-- Theorem stating the area a llama can roam under specific conditions -/
theorem llama_roam_area_specific (shedWidth shedLength leashLength : ℝ) 
  (h1 : shedWidth = 4)
  (h2 : shedLength = 6)
  (h3 : leashLength = 5) :
  llamaRoamArea shedWidth shedLength leashLength = 20.75 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_llama_roam_area_specific_l1032_103288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_value_properties_l1032_103236

/-- Definition of beautiful two-variable linear equation -/
def beautiful_equation (a b : ℚ) (x y : ℚ) : Prop := a * x + y = b

/-- Definition of beautiful value -/
def beautiful_value (a b : ℚ) : ℚ := 
  (b / (a + 2))

theorem beautiful_value_properties : 
  (beautiful_value 5 1 = 1/3) ∧ 
  (beautiful_value (1/3) (-7) = -3) ∧
  (∃ n : ℚ, beautiful_value (5/2) n = beautiful_value 4 (n-2) ∧ 
            beautiful_value (5/2) n = 4/5 ∧ 
            n = 18/5) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_value_properties_l1032_103236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_perpendicular_equality_l1032_103246

/-- Given a circle with diameter AB and chord CD, where perpendiculars to CD through C and D
    intersect AB at K and M respectively, AK equals BM. -/
theorem chord_perpendicular_equality (O A B C D K M : ℝ × ℝ) :
  let r := dist O A
  (dist O A = dist O B) →  -- AB is a diameter
  (dist O C = r) →         -- C is on the circle
  (dist O D = r) →         -- D is on the circle
  ((C.1 - D.1) * (A.1 - B.1) + (C.2 - D.2) * (A.2 - B.2) = 0) →  -- CD is perpendicular to AB
  ((K.1 - C.1) * (C.1 - D.1) + (K.2 - C.2) * (C.2 - D.2) = 0) →  -- CK is perpendicular to CD
  ((M.1 - D.1) * (C.1 - D.1) + (M.2 - D.2) * (C.2 - D.2) = 0) →  -- DM is perpendicular to CD
  (∃ t : ℝ, K = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))) →  -- K is on line AB
  (∃ s : ℝ, M = (A.1 + s * (B.1 - A.1), A.2 + s * (B.2 - A.2))) →  -- M is on line AB
  dist A K = dist B M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_perpendicular_equality_l1032_103246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_n_for_product_six_satisfies_condition_six_is_smallest_l1032_103277

theorem smallest_even_n_for_product (n : ℕ) : 
  (n % 2 = 0 ∧ n > 0 ∧ 3^((n*(n+1))/8) > 500) → n ≥ 6 :=
by sorry

theorem six_satisfies_condition : 
  6 % 2 = 0 ∧ 6 > 0 ∧ 3^((6*(6+1))/8) > 500 :=
by sorry

theorem six_is_smallest : 
  ∀ m : ℕ, m < 6 → ¬(m % 2 = 0 ∧ m > 0 ∧ 3^((m*(m+1))/8) > 500) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_n_for_product_six_satisfies_condition_six_is_smallest_l1032_103277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_permutable_powers_of_two_l1032_103297

/-- Represents the decimal digits of a natural number as a list. -/
def digits (n : ℕ) : List ℕ :=
  sorry

/-- Predicate to check if a function is a permutation of a list. -/
def IsPermutation (f : List ℕ → List ℕ) (l : List ℕ) : Prop :=
  sorry

theorem no_permutable_powers_of_two : ∀ m n : ℕ, m ≠ n →
  ¬∃ (perm : List ℕ → List ℕ), 
    perm (digits (2^m)) = digits (2^n) ∧ IsPermutation perm (digits (2^m)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_permutable_powers_of_two_l1032_103297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_triangles_similar_l1032_103283

-- Define the triangles
structure Triangle :=
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ)

-- Define similarity relation
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the triangles P, Q, and R
noncomputable def P (a : ℝ) : Triangle :=
  ⟨a, Real.sqrt 3 * a, 2 * a⟩

noncomputable def Q (a : ℝ) : Triangle :=
  ⟨3 * a, 2 * Real.sqrt 3 * a, Real.sqrt 3 * a⟩

-- R is defined by its angles, not its sides
def R : Triangle := sorry

-- Theorem statement
theorem all_triangles_similar (a : ℝ) (h : a > 0) :
  similar (P a) (Q a) ∧ similar (Q a) R ∧ similar R (P a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_triangles_similar_l1032_103283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l1032_103211

theorem cosine_inequality (α β γ : ℝ) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < γ) (h4 : γ ≤ π/2) :
  (Real.cos β - Real.cos γ) / (Real.cos α - Real.cos β) > (8/π^2) * (γ - β) / (β - α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l1032_103211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l1032_103200

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ + Real.pi / 3)

theorem phi_value :
  (∀ x, f x (2 * Real.pi / 3) = -f (-x) (2 * Real.pi / 3)) →  -- f is odd
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 4 → f y (2 * Real.pi / 3) ≤ f x (2 * Real.pi / 3)) →  -- f is decreasing in [0, π/4]
  ∃ φ, φ = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l1032_103200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defeat_dragon_l1032_103259

/-- Represents the number of heads a warrior can cut off -/
structure WarriorAbility where
  fraction : Nat
  bonus : Nat

/-- Represents the state of the dragon -/
structure DragonState where
  heads : Nat

/-- Defines the abilities of the three warriors -/
def warriors : List WarriorAbility := [
  { fraction := 2, bonus := 1 },  -- Ilya Muromets
  { fraction := 3, bonus := 2 },  -- Dobrynya Nikitich
  { fraction := 4, bonus := 3 }   -- Alyosha Popovich
]

/-- Applies a warrior's ability to the dragon state -/
def applyAbility (state : DragonState) (ability : WarriorAbility) : Option DragonState :=
  if state.heads % ability.fraction = 0 then
    some { heads := state.heads - (state.heads / ability.fraction + ability.bonus) }
  else
    none

/-- Checks if the warriors can defeat the dragon -/
def canDefeatDragon (initialHeads : Nat) : Prop :=
  ∃ (sequence : List WarriorAbility), 
    (∀ ability ∈ sequence, ability ∈ warriors) ∧
    (sequence.foldl 
      (λ state ability => 
        match state with
        | some s => applyAbility s ability
        | none => none
      ) 
      (some { heads := initialHeads }) = some { heads := 0 })

/-- Theorem: The warriors can defeat a dragon with 20^20 heads -/
theorem defeat_dragon : canDefeatDragon (20^20) := by
  sorry

/-- Lemma: If the number of heads is divisible by 2 or 3, the warriors can always reduce it -/
lemma reduce_heads (n : Nat) (h : n % 2 = 0 ∨ n % 3 = 0) : 
  ∃ (ability : WarriorAbility), ability ∈ warriors ∧ 
    ∃ (m : Nat), applyAbility { heads := n } ability = some { heads := m } ∧ (m % 2 = 0 ∨ m % 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defeat_dragon_l1032_103259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_dogs_l1032_103295

/-- The number of dogs in a house with given conditions -/
theorem number_of_dogs : ℕ := by
  -- Define the number of birds, cats, and family members
  let birds : ℕ := 4
  let cats : ℕ := 18
  let family_members : ℕ := 7

  -- Define the number of feet for each type of creature
  let human_feet : ℕ := 2
  let bird_feet : ℕ := 2
  let cat_feet : ℕ := 4
  let dog_feet : ℕ := 4

  -- Define the relation between total feet and heads
  let feet_head_difference : ℕ := 74

  -- The number of dogs that satisfies the conditions
  have h : ∃ dogs : ℕ, 
    (family_members * human_feet + birds * bird_feet + cats * cat_feet + dogs * dog_feet) = 
    (family_members + birds + cats + dogs + feet_head_difference) := by
    use 3
    ring

  -- Prove that the number of dogs is 3
  exact 3


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_dogs_l1032_103295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1032_103279

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + (a - 1/2) * x^2 + 2*(1-a)*x + a

/-- Theorem stating that f(x) has exactly 3 zeros when a < -2 -/
theorem f_has_three_zeros (a : ℝ) (h : a < -2) : 
  ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    (∀ x > 0, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1032_103279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_35_values_g_of_f_l1032_103244

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 4 * x^2 - 3

noncomputable def g (y : ℝ) : ℝ := 
  let x := Real.sqrt ((y + 3) / 4)
  x^2 - x + 2

-- Theorem statement
theorem sum_of_g_35_values :
  (g 35) + (g 35) = 23 := by
  sorry

-- Additional theorem to show the relationship between f and g
theorem g_of_f (x : ℝ) :
  g (f x) = x^2 - x + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_35_values_g_of_f_l1032_103244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_value_l1032_103281

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => (m + 3) * x + y - 1 = 0

def l₂ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => 4 * x + m * y + 3 * m - 4 = 0

-- Define perpendicularity of two lines
def perpendicular (l₁ l₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ (m₁ m₂ n₁ n₂ : ℝ), 
    (∀ x y, l₁ x y ↔ y = m₁ * x + m₂) ∧
    (∀ x y, l₂ x y ↔ y = n₁ * x + n₂) ∧
    m₁ * n₁ = -1

-- Theorem statement
theorem perpendicular_lines_m_value :
  ∀ m : ℝ, perpendicular (l₁ m) (l₂ m) → m = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_value_l1032_103281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_count_l1032_103222

theorem strawberry_count (total : ℕ) (kiwi_fraction : ℚ) (strawberry_count : ℕ) : 
  total = 78 → 
  kiwi_fraction = 1/3 →
  strawberry_count = total - (kiwi_fraction * ↑total).floor →
  strawberry_count = 52 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_count_l1032_103222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1032_103202

noncomputable def f (x y : ℝ) : ℝ := (x + y) / (Int.floor x * Int.floor y + Int.floor x + Int.floor y + 1)

theorem range_of_f :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 →
  ∃ z ∈ {1/2} ∪ Set.Icc (5/6) (5/4), f x y = z :=
by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1032_103202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1032_103217

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sin x}
def B : Set ℝ := {y | ∃ x, y = x^2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1032_103217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_implies_a_equals_two_condition_implies_m_range_l1032_103284

noncomputable section

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := x + m
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m^2/2 + 2*m - 3

-- Theorem 1: If the solution set of g(x) < m^2/2 + 1 is (1, a), then a = 2
theorem solution_set_implies_a_equals_two (m : ℝ) (a : ℝ) :
  (∀ x, 1 < x ∧ x < a ↔ g m x < m^2/2 + 1) → a = 2 := by sorry

-- Theorem 2: If for all x₁ ∈ [0,1], there exists x₂ ∈ [1,2], such that f(x₁) > g(x₂), then -2 < m < 2
theorem condition_implies_m_range (m : ℝ) :
  (∀ x₁ ∈ Set.Icc 0 1, ∃ x₂ ∈ Set.Icc 1 2, f m x₁ > g m x₂) → 
  -2 < m ∧ m < 2 := by sorry

end


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_implies_a_equals_two_condition_implies_m_range_l1032_103284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tau_product_of_coprime_l1032_103293

/-- The number of divisors function -/
def tau (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- Theorem: For coprime natural numbers a and b, τ(ab) = τ(a)τ(b) -/
theorem tau_product_of_coprime (a b : ℕ) (h : Nat.Coprime a b) : 
  tau (a * b) = tau a * tau b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tau_product_of_coprime_l1032_103293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_with_min_difference_l1032_103209

theorem max_sum_with_min_difference (f g h j : ℕ) : 
  f ∈ ({4, 5, 6, 7} : Set ℕ) → 
  g ∈ ({4, 5, 6, 7} : Set ℕ) → 
  h ∈ ({4, 5, 6, 7} : Set ℕ) → 
  j ∈ ({4, 5, 6, 7} : Set ℕ) → 
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j →
  (∀ (f' g' h' j' : ℕ), 
    f' ∈ ({4, 5, 6, 7} : Set ℕ) → 
    g' ∈ ({4, 5, 6, 7} : Set ℕ) → 
    h' ∈ ({4, 5, 6, 7} : Set ℕ) → 
    j' ∈ ({4, 5, 6, 7} : Set ℕ) → 
    f' ≠ g' → f' ≠ h' → f' ≠ j' → g' ≠ h' → g' ≠ j' → h' ≠ j' →
    f * h - g * j ≤ f' * h' - g' * j') →
  f * g + g * h + h * j + f * j ≤ 141 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_with_min_difference_l1032_103209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_points_l1032_103299

-- Define the equilateral triangle ABC
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  side_length : ℝ
  is_equilateral : side_length = 1 ∧
                   dist A B = side_length ∧
                   dist B C = side_length ∧
                   dist C A = side_length

-- Define a point P in the plane
def Point : Type := ℝ × ℝ

-- Define the distance function
noncomputable def dist (p1 p2 : Point) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the set of points P satisfying the condition
def SatisfyingPoints (t : EquilateralTriangle) : Set Point :=
  {P : Point | max (dist P t.A) (max (dist P t.B) (dist P t.C)) = 1}

-- Define a 120° arc on a circle
def Arc120 (center : Point) (radius : ℝ) : Set Point :=
  {P : Point | dist P center = radius ∧ 
               ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi / 3 ∧
               P.1 = center.1 + radius * Real.cos θ ∧
               P.2 = center.2 + radius * Real.sin θ}

-- State the theorem
theorem equilateral_triangle_points (t : EquilateralTriangle) :
  SatisfyingPoints t = 
    Arc120 t.A 1 ∪ Arc120 t.B 1 ∪ Arc120 t.C 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_points_l1032_103299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1032_103270

/-- The eccentricity of a hyperbola with given conditions is 5 -/
theorem hyperbola_eccentricity (a b : ℝ) (F₁ F₂ P O : ℝ × ℝ) : 
  a > 0 → b > 0 →
  -- F₁ and F₂ are foci of the hyperbola
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ 
    ‖(x, y) - F₁‖ - ‖(x, y) - F₂‖ = 2 * a) →
  -- P is on the right branch of the hyperbola
  P.1 > 0 →
  P.1^2 / a^2 - P.2^2 / b^2 = 1 →
  -- Orthogonality condition
  ((P - O) + (F₂ - O)) • (F₂ - P) = 0 →
  -- Distance ratio condition
  3 * ‖P - F₁‖ = 4 * ‖P - F₂‖ →
  -- Eccentricity definition
  let c := ‖F₂ - O‖
  c / a = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1032_103270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_building_floors_l1032_103292

/-- The number of entrances in the building -/
def p : ℕ := sorry

/-- The number of floors per entrance -/
def f : ℕ := sorry

/-- The number of apartments per floor -/
def k : ℕ := sorry

/-- The total number of apartments in the building -/
def total_apartments : ℕ := 105

theorem apartment_building_floors :
  (1 < p) ∧ (p < k) ∧ (k < f) ∧ (p * f * k = total_apartments) → f = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_building_floors_l1032_103292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_101_equals_52_l1032_103234

def F : ℕ → ℚ
  | 0 => 2  -- Add a case for 0 to cover all natural numbers
  | 1 => 2
  | (n + 2) => (2 * F (n + 1) + 1) / 2

theorem F_101_equals_52 : F 101 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_101_equals_52_l1032_103234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_male_attendees_on_time_l1032_103247

theorem male_attendees_on_time (total : ℝ) (h1 : total > 0) : 
  (1/2 : ℝ) * ((2/3 : ℝ) * total) = 
  (total - (2/9 : ℝ) * total) - (5/6 : ℝ) * ((1/3 : ℝ) * total) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_male_attendees_on_time_l1032_103247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l1032_103213

-- Define the function f(x) = x + 1/x
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- Theorem statement
theorem f_odd_and_decreasing :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l1032_103213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1032_103273

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (a r : ℝ) :
  (∃ (S₃₀₀₀ S₆₀₀₀ : ℝ),
    geometric_sequence a r 3000 = S₃₀₀₀ ∧
    geometric_sequence a r 6000 = S₆₀₀₀ ∧
    S₃₀₀₀ = 1000 ∧
    S₆₀₀₀ = 1900) →
  geometric_sequence a r 9000 = 2710 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1032_103273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l1032_103253

-- Define the ellipse C
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define eccentricity
noncomputable def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Define the line segment length
noncomputable def line_segment_length (b : ℝ) : ℝ :=
  Real.sqrt 2 * b

-- Define the triangle area
noncomputable def triangle_area (x₁ x₂ k : ℝ) : ℝ :=
  Real.sqrt ((8 * (2 * k^2 - 3)) / (2 * k^2 + 1)^2)

theorem ellipse_and_triangle_properties
  (a b c : ℝ)
  (h₁ : a > b)
  (h₂ : b > 0)
  (h₃ : eccentricity a c = Real.sqrt 2 / 2)
  (h₄ : line_segment_length b = Real.sqrt 2)
  (h₅ : a^2 = b^2 + c^2) :
  (∃ (x y : ℝ), ellipse (Real.sqrt 2) 1 x y) ∧
  (∃ (k : ℝ), ∀ (x₁ x₂ : ℝ), triangle_area x₁ x₂ k ≤ Real.sqrt 2 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l1032_103253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_magnitude_l1032_103245

noncomputable def z : ℂ := 2 + 2 * Real.sqrt 2 * Complex.I

theorem complex_power_magnitude : Complex.abs (z^6) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_magnitude_l1032_103245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lincoln_memorial_statue_ratio_l1032_103243

/-- Represents the ratio between a statue and its scale model -/
structure StatueModelRatio where
  statue_height : ℚ
  model_height : ℚ
  statue_unit : String
  model_unit : String

/-- Calculates how many units of the statue one unit of the model represents -/
def feet_per_inch (ratio : StatueModelRatio) : ℚ :=
  ratio.statue_height / ratio.model_height

theorem lincoln_memorial_statue_ratio :
  let ratio : StatueModelRatio := {
    statue_height := 60,
    model_height := 4,
    statue_unit := "feet",
    model_unit := "inches"
  }
  feet_per_inch ratio = 15 := by
  sorry

#eval feet_per_inch {
  statue_height := 60,
  model_height := 4,
  statue_unit := "feet",
  model_unit := "inches"
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lincoln_memorial_statue_ratio_l1032_103243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_point_exists_l1032_103215

/-- Represents a point on the road -/
structure RoadPoint where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : RoadPoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The distance from a point to the power line y = d -/
def distanceToPowerLine (p : RoadPoint) (d : ℝ) : ℝ :=
  |p.y - d|

/-- The factory point -/
def factory : RoadPoint :=
  ⟨0, 0⟩

/-- Theorem: For any natural number n and non-zero real d, 
    there exists a point on the road equidistant from the factory and power line -/
theorem road_point_exists (n : ℕ) (d : ℝ) (h : d ≠ 0) : 
  ∃ (p : RoadPoint), 
    distance p factory = n * d ∧ 
    distanceToPowerLine p d = n * d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_point_exists_l1032_103215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1032_103203

/-- Hyperbola with given properties has eccentricity √3 + 1 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) →  -- P is on the hyperbola
  (F₁.1 < 0 ∧ F₁.2 = 0) →            -- F₁ is the left focus
  (F₂.1 > 0 ∧ F₂.2 = 0) →            -- F₂ is the right focus
  (P.1 > 0) →                        -- P is on the right branch
  ((P.1 + F₂.1) * (P.1 - F₂.1) + P.2 * P.2 = 0) →  -- (OP + OF₂) ⋅ PF₂ = 0
  ((P.1 - F₁.1)^2 + P.2^2 = 3 * ((P.1 - F₂.1)^2 + P.2^2)) →  -- |PF₁| = √3 |PF₂|
  (Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 3 + 1) :=  -- eccentricity = √3 + 1
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1032_103203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_theorem_l1032_103271

/-- Calculate the simple interest for a given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculate the gain per year in the transaction -/
noncomputable def gainPerYear (
  borrowedAmount : ℝ)
  (borrowingPeriod : ℝ)
  (borrowingRate : ℝ)
  (lendingPeriod : ℝ)
  (lendingRate : ℝ) : ℝ :=
  let interestEarned := simpleInterest borrowedAmount lendingRate lendingPeriod
  let interestPaid := simpleInterest borrowedAmount borrowingRate borrowingPeriod
  (interestEarned - interestPaid) / borrowingPeriod

theorem transaction_gain_theorem :
  gainPerYear 9000 2 4 2 6 = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_theorem_l1032_103271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_period_of_f_l1032_103261

noncomputable def f (x : ℝ) : ℝ := (Real.arcsin (Real.sin (Real.arccos (Real.cos (3 * x)))))^(-5:ℤ)

theorem principal_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_period_of_f_l1032_103261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_sum_positive_l1032_103221

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function F
noncomputable def F (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then f a b x else -f a b x

-- State the theorem
theorem F_sum_positive
  (a b m n : ℝ)
  (hm : m > 0)
  (hn : n < 0)
  (hmn : m + n > 0)
  (ha : a > 0)
  (hf_even : ∀ x, f a b x = f a b (-x)) :
  F a b m + F a b n > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_sum_positive_l1032_103221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_mark_l1032_103268

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) (initial_avg : ℝ) : 
  total_students = 30 →
  excluded_students = 5 →
  excluded_avg = 20 →
  remaining_avg = 92 →
  (total_students * initial_avg - excluded_students * excluded_avg) / (total_students - excluded_students) = remaining_avg →
  initial_avg = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_mark_l1032_103268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1032_103285

-- Define the inequality condition
def inequality_condition (a : ℝ) : Prop :=
  ∀ x > 0, (a - 5) / x < |1 + 1/x| - |1 - 2/x| ∧ |1 + 1/x| - |1 - 2/x| < (a + 2) / x

-- Define the set A
def set_A (a : ℝ) : Set ℝ :=
  {x : ℝ | |x - 1| + |x + 1| ≤ a}

-- Define the set B
def set_B : Set ℝ :=
  {x : ℝ | 4 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 8}

theorem problem_solution :
  (∃ a : ℝ, inequality_condition a ∧ 1 < a ∧ a < 8) ∧
  (∃ x : ℝ, x ∈ set_A 8 ∧ x ∈ set_B) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1032_103285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_comparison_l1032_103237

theorem sqrt_difference_comparison : Real.sqrt 10 - Real.sqrt 3 > Real.sqrt 14 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_comparison_l1032_103237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greene_family_amusement_park_expense_l1032_103205

def admission_cost : ℝ := 45
def discount_rate : ℝ := 0.10
def food_cost_difference : ℝ := 13
def food_tax_rate : ℝ := 0.08
def transportation_cost : ℝ := 25
def souvenir_cost : ℝ := 40
def games_cost : ℝ := 28

theorem greene_family_amusement_park_expense :
  (let discounted_admission := admission_cost * (1 - discount_rate)
   let food_cost := discounted_admission - food_cost_difference
   let food_cost_with_tax := food_cost * (1 + food_tax_rate)
   discounted_admission + food_cost_with_tax + transportation_cost + souvenir_cost + games_cost) = 163.20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greene_family_amusement_park_expense_l1032_103205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_expression_simplification_equation_solutions_l1032_103276

-- Problem 1
theorem system_solution :
  ∃ (x y : ℝ), 5*x + 2*y = 25 ∧ 3*x + 4*y = 15 ∧ x = 5 ∧ y = 0 := by sorry

-- Problem 2
theorem expression_simplification :
  2*(Real.sqrt 3 - 1) - |Real.sqrt 3 - 2| - ((-64 : ℝ) ^ (1/3 : ℝ)) = 3 * Real.sqrt 3 := by sorry

-- Problem 3
theorem equation_solutions :
  ∀ x : ℝ, 2*(x - 1)^2 - 49 = 1 ↔ x = -4 ∨ x = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_expression_simplification_equation_solutions_l1032_103276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_squared_l1032_103264

/-- A polynomial of the form z^3 + qz + r -/
def CubicPolynomial (q r : ℂ) (z : ℂ) : ℂ := z^3 + q*z + r

theorem right_triangle_hypotenuse_squared
  (a b c q r : ℂ)
  (h_zeros : ∀ z, CubicPolynomial q r z = 0 ↔ z = a ∨ z = b ∨ z = c)
  (h_sum_squares : Complex.abs a ^ 2 + Complex.abs b ^ 2 + Complex.abs c ^ 2 = 250)
  (h_right_triangle : ∃ (p : Equiv.Perm (Fin 3)),
    Complex.abs ((p.toFun 0 : ℂ) - (p.toFun 1 : ℂ))^2 +
    Complex.abs ((p.toFun 1 : ℂ) - (p.toFun 2 : ℂ))^2 =
    Complex.abs ((p.toFun 0 : ℂ) - (p.toFun 2 : ℂ))^2)
  : Complex.abs (a - b)^2 = 375 ∨
    Complex.abs (b - c)^2 = 375 ∨
    Complex.abs (c - a)^2 = 375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_squared_l1032_103264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_polar_equation_l1032_103265

-- Define the point in polar coordinates
noncomputable def point : ℝ × ℝ := (4, Real.pi/4)

-- Define the condition that the line is perpendicular to the polar axis
def perpendicular_to_polar_axis (θ : ℝ) : Prop := θ = Real.pi/2

-- State the theorem
theorem line_polar_equation :
  ∃ (ρ θ : ℝ), 
    (ρ * Real.cos θ = point.1 ∧ ρ * Real.sin θ = point.2) ∧ 
    perpendicular_to_polar_axis θ → 
    ρ * Real.sin θ = Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_polar_equation_l1032_103265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_line_equation_derivative_zero_at_x₀_l1032_103256

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - 3 * x ^ (1/3)

-- Define the point of interest
def x₀ : ℝ := 64

-- Theorem statement
theorem normal_line_equation : 
  ∀ x : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < |h| ∧ |h| < δ → 
    |((f (x₀ + h) - f x₀) / h) - 0| < ε) → 
  (x = x₀) :=
by
  sorry

-- Additional theorem to show that the derivative at x₀ is indeed 0
theorem derivative_zero_at_x₀ :
  ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < |h| ∧ |h| < δ → 
    |((f (x₀ + h) - f x₀) / h) - 0| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_line_equation_derivative_zero_at_x₀_l1032_103256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scheme_two_probability_l1032_103224

/-- The probability of group A correctly recognizing a piece of music -/
noncomputable def p_a : ℝ := 2/3

/-- The probability of group B correctly recognizing a piece of music -/
noncomputable def p_b : ℝ := 1/2

/-- The number of attempts each group makes in scheme two -/
def attempts : ℕ := 2

/-- The minimum number of correct recognitions required to pass scheme two -/
def min_correct : ℕ := 3

/-- The probability of passing scheme two in one test -/
noncomputable def prob_pass_scheme_two : ℝ :=
  (p_a ^ 2 * (1 - (1 - p_b) ^ 2)) +
  (2 * p_a * (1 - p_a) * p_b ^ 2) +
  (p_a ^ 2 * p_b ^ 2)

theorem scheme_two_probability :
  prob_pass_scheme_two = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scheme_two_probability_l1032_103224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_1989_l1032_103267

theorem partition_1989 :
  ∃ (partition : Finset (Finset Nat)),
    (∀ A ∈ partition, A.card = 17) ∧
    (partition.card = 117) ∧
    (∀ A B, A ∈ partition → B ∈ partition → A ≠ B → A ∩ B = ∅) ∧
    (∀ A B, A ∈ partition → B ∈ partition → (A.sum id) = (B.sum id)) ∧
    (partition.biUnion id = Finset.range 1990 \ {0}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_1989_l1032_103267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_sum_l1032_103260

def systematic_sample (n : ℕ) (k : ℕ) (start : ℕ) : List ℕ :=
  List.range k |>.map (fun i => start + i * (n / k))

theorem systematic_sample_sum (n k start : ℕ) (h1 : n = 55) (h2 : k = 5) (h3 : start = 6) :
  let sample := systematic_sample n k start
  (sample.get? 1).isSome ∧ (sample.get? 3).isSome →
  ((sample.get? 1).getD 0 + (sample.get? 3).getD 0 = 56) :=
by
  sorry

#eval systematic_sample 55 5 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_sum_l1032_103260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1032_103262

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := log x + 2 * x^2 + 6 * m * x + 1

-- Define monotonically increasing
def MonotonicallyIncreasing (g : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → g x < g y

-- State the theorem
theorem sufficient_but_not_necessary :
  (∃ m : ℝ, MonotonicallyIncreasing (f m) 0 π) ∧
  (∃ m : ℝ, m ≥ -5 ∧ ¬MonotonicallyIncreasing (f m) 0 π) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1032_103262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_12_l1032_103228

/-- A triangle with side lengths 6, x, and 2x, where 2 < x < 6 -/
structure SpecialTriangle where
  x : ℝ
  h1 : 2 < x
  h2 : x < 6

/-- The area of a SpecialTriangle -/
noncomputable def area (t : SpecialTriangle) : ℝ :=
  let s := (6 + t.x + 2*t.x) / 2
  Real.sqrt (s * (s - 6) * (s - t.x) * (s - 2*t.x))

/-- The maximum area of a SpecialTriangle is 12 -/
theorem max_area_is_12 :
  ∀ t : SpecialTriangle, area t ≤ 12 ∧ ∃ t' : SpecialTriangle, area t' = 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_12_l1032_103228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chelsea_percentage_proof_l1032_103223

def total_candles : ℕ := 40
def alyssa_fraction : ℚ := 1/2
def remaining_candles : ℕ := 6

theorem chelsea_percentage_proof :
  let candles_after_alyssa := total_candles - (total_candles * alyssa_fraction).floor
  let chelsea_used := candles_after_alyssa - remaining_candles
  (chelsea_used : ℚ) / candles_after_alyssa * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chelsea_percentage_proof_l1032_103223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1032_103258

-- Define proposition P
def P : Prop := ∀ a b : ℝ, (a > b → (2 : ℝ)^a > (2 : ℝ)^b) ∧ ¬((2 : ℝ)^a > (2 : ℝ)^b → a > b)

-- Define proposition q
def q : Prop := ∃ x : ℝ, |x + 1| ≤ x

theorem problem_statement : ¬P ∧ ¬q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1032_103258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pc_smartphone_price_difference_l1032_103275

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 300

/-- The difference in price between a personal computer and a smartphone -/
def pc_difference : ℕ := 500

/-- The price of a personal computer in dollars -/
def pc_price : ℕ := smartphone_price + pc_difference

/-- The price of an advanced tablet in dollars -/
def tablet_price : ℕ := smartphone_price + pc_price

/-- The total cost of buying one of each product in dollars -/
def total_cost : ℕ := 2200

theorem pc_smartphone_price_difference :
  smartphone_price + pc_price + tablet_price = total_cost →
  pc_difference = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pc_smartphone_price_difference_l1032_103275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1032_103239

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x else x^3 - 1/x + 1

-- Theorem statement
theorem f_composition_value :
  f (1 / f 2) = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1032_103239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l1032_103231

-- Define the game
structure Game where
  n : Nat
  digits : Fin 101 → Fin 10
  h : n = 101

-- Define the sum of digits
def sum_of_digits (g : Game) : Nat :=
  (Finset.range 101).sum (λ i => (g.digits i).val)

-- Define the winning condition for the first player
def first_player_wins (g : Game) : Prop :=
  sum_of_digits g % 11 = 0

-- Theorem statement
theorem first_player_winning_strategy :
  ∃ (strategy : Fin 51 → Fin 10),
    ∀ (opponent_moves : Fin 50 → Fin 10),
      first_player_wins {
        n := 101,
        digits := λ i =>
          if i.val % 2 = 0
          then strategy ⟨i.val / 2, by sorry⟩
          else opponent_moves ⟨i.val / 2, by sorry⟩,
        h := rfl
      } :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l1032_103231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OB_OC_and_cos_alpha_l1032_103227

noncomputable section

open Real

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 2)
def C (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)

def O : ℝ × ℝ := (0, 0)

def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def angle (v w : ℝ × ℝ) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem angle_OB_OC_and_cos_alpha (α : ℝ) 
  (h1 : 0 < α ∧ α < π) 
  (h2 : (vector O A + vector O (C α)).1^2 + (vector O A + vector O (C α)).2^2 = 7) 
  (h3 : perpendicular (vector A (C α)) (vector B (C α))) :
  (angle (vector O B) (vector O (C α)) = π / 6) ∧ 
  (Real.cos α = (1 - Real.sqrt 7) / 4) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OB_OC_and_cos_alpha_l1032_103227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_M_l1032_103218

/-- Given that M^2 = 9^50 * 6^75, prove that the sum of digits of M is 18 -/
theorem sum_of_digits_of_M (M : ℕ) (h : M ^ 2 = 9 ^ 50 * 6 ^ 75) : 
  (Nat.digits 10 M).sum = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_M_l1032_103218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_measurement_double_inequality_l1032_103250

theorem measurement_double_inequality (a₀ : ℝ) : 
  (∃ (x : ℝ), x = 9.3 ∧ |a₀ - x| ≤ 0.5) → 8.8 ≤ a₀ ∧ a₀ ≤ 9.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_measurement_double_inequality_l1032_103250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l1032_103249

theorem max_true_statements (a b : ℝ) (ha : a < 0) (hb : b < 0) : 
  (∃ (s : Finset Prop), s.card = 3 ∧ 
    (∀ p ∈ s, p ∈ ({1/a < 1/b, a^2 > b^2, Real.sqrt a < Real.sqrt b, a^3 > b^3, a < 0, b < 0} : Set Prop)) ∧ 
    (∀ t : Finset Prop, t.card > 3 → 
      ¬(∀ q ∈ t, q ∈ ({1/a < 1/b, a^2 > b^2, Real.sqrt a < Real.sqrt b, a^3 > b^3, a < 0, b < 0} : Set Prop)))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l1032_103249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assignment_impossibility_l1032_103274

/-- A function that assigns natural numbers to points with integer coordinates on a plane. -/
def assignment : ℤ × ℤ → ℕ := sorry

/-- Predicate to check if three points are collinear. -/
def collinear (p₁ p₂ p₃ : ℤ × ℤ) : Prop :=
  (p₂.2 - p₁.2) * (p₃.1 - p₁.1) = (p₃.2 - p₁.2) * (p₂.1 - p₁.1)

/-- Predicate to check if three numbers have a common divisor greater than one. -/
def have_common_divisor (a b c : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d ∣ a ∧ d ∣ b ∧ d ∣ c

/-- Theorem stating the impossibility of the assignment. -/
theorem assignment_impossibility : ¬∃ (assignment : ℤ × ℤ → ℕ),
  ∀ (p₁ p₂ p₃ : ℤ × ℤ), collinear p₁ p₂ p₃ ↔ 
    have_common_divisor (assignment p₁) (assignment p₂) (assignment p₃) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_assignment_impossibility_l1032_103274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_twelve_m_squared_l1032_103291

/-- Given an odd integer m with exactly 7 positive divisors,
    the number of positive divisors of 12m^2 is 78. -/
theorem divisors_of_twelve_m_squared (m : ℕ) 
  (h_odd : Odd m) 
  (h_divisors : (Nat.divisors m).card = 7) : 
  (Nat.divisors (12 * m^2)).card = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_twelve_m_squared_l1032_103291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_open_interval_l1032_103269

def A (a : ℝ) : Set ℝ := {-1, 0, a}
def B : Set ℝ := {x : ℝ | 1 < (3 : ℝ)^x ∧ (3 : ℝ)^x < 3}

theorem a_in_open_interval (a : ℝ) : (A a ∩ B).Nonempty → a ∈ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_open_interval_l1032_103269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_and_triangle_areas_l1032_103216

/-- Rectangle ABCD with points M on AB and N on AD -/
structure Rectangle :=
  (A B C D M N : ℝ × ℝ)
  (AN : ℝ)
  (NC : ℝ)
  (AM : ℝ)
  (MB : ℝ)

/-- Conditions for the rectangle -/
def rectangle_conditions (r : Rectangle) : Prop :=
  r.AN = 3 ∧ r.NC = 39 ∧ r.AM = 10 ∧ r.MB = 5

/-- Area of rectangle ABCD -/
def area_rectangle (r : Rectangle) : ℝ :=
  (r.AM + r.MB) * (r.AN + r.NC)

/-- Area of triangle MNC -/
noncomputable def area_triangle (r : Rectangle) : ℝ :=
  (1/2) * Real.sqrt 109 * r.NC

theorem rectangle_and_triangle_areas (r : Rectangle) 
  (h : rectangle_conditions r) : 
  area_rectangle r = 630 ∧ 
  area_triangle r = (39 * Real.sqrt 109) / 2 := by
  sorry

#check rectangle_and_triangle_areas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_and_triangle_areas_l1032_103216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_S_l1032_103225

-- Define the set S
def S : Set ℂ := {z | Complex.abs z.re ≤ 2 ∧ Complex.abs z.im ≤ 2}

-- Define the transformation
noncomputable def transform (z : ℂ) : ℂ := (1/2 + 1/2 * Complex.I) * z

-- Theorem statement
theorem transform_stays_in_S : ∀ z ∈ S, transform z ∈ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_S_l1032_103225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1032_103235

noncomputable section

variable (f : ℝ → ℝ)

axiom periodic_property : ∀ x : ℝ, f (x + 3) = -f (x + 1)
axiom f_2_value : f 2 = 2014

theorem f_composition_value : f (f 2014 + 2) + 3 = -2011 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1032_103235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_simplification_l1032_103212

theorem trigonometric_expression_simplification (a b : ℝ) :
  4 * a^2 * (Real.sin (π/6))^4 - 6 * a * b * (Real.tan (π/6))^2 + (b * (1 / Real.tan (π/4)))^2 = a^2/4 - 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_simplification_l1032_103212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1032_103255

-- Part I
theorem part_one (a b c n p q : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a^2 + b^2 + c^2 = 2) (h5 : n^2 + p^2 + q^2 = 2) :
  n^4 / a^2 + p^4 / b^2 + q^4 / c^2 ≥ 2 := by sorry

-- Part II
theorem part_two (a b c : ℝ) 
  (f : ℝ → ℝ) (hf : f = λ x ↦ a*x^2 + b*x + c)
  (h1 : |f (-1)| ≤ 1) (h2 : |f 0| ≤ 1) (h3 : |f 1| ≤ 1) :
  ∀ x, x ∈ Set.Icc (-1) 1 → |a*x + b| ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1032_103255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_mold_radius_is_one_l1032_103280

noncomputable def hemisphereVolume (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

noncomputable def smallerMoldRadius (largeBowlRadius : ℝ) (numMolds : ℕ) : ℝ :=
  let largeVolume := hemisphereVolume largeBowlRadius
  let distributedVolume := (1 / 3) * largeVolume
  let smallMoldVolume := distributedVolume / (numMolds : ℝ)
  (smallMoldVolume / ((2 / 3) * Real.pi))^(1 / 3)

theorem smaller_mold_radius_is_one :
  smallerMoldRadius 3 9 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_mold_radius_is_one_l1032_103280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_neg_three_point_five_l1032_103263

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem floor_neg_three_point_five :
  floor (-3.5) = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_neg_three_point_five_l1032_103263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_problem_l1032_103207

theorem triangle_sine_problem (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = Real.pi ∧  -- Angle sum property of triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  Real.sin A / a = Real.sin B / b ∧ Real.sin B / b = Real.sin C / c ∧  -- Sine law
  Real.sin (A + B) = 1/3 ∧
  a = 3 ∧
  c = 4 →
  Real.sin A = 1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_problem_l1032_103207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1032_103219

def a : ℝ × ℝ := (3, 2)
def b (lambda : ℝ) : ℝ × ℝ := (-7, lambda + 1)

theorem parallel_vectors_lambda (lambda : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (11 • a - 2018 • b lambda) = k • (10 • a + 2017 • b lambda)) →
  lambda = -17/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1032_103219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_undefined_value_l1032_103201

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (8 * x^2 - 65 * x + 8)

-- Theorem statement
theorem largest_undefined_value :
  ∃ (x : ℝ), x = 8 ∧ 
  (∀ y : ℝ, y > x → f y ≠ 0⁻¹) ∧
  (∀ ε > 0, ∃ y : ℝ, x - ε < y ∧ y < x ∧ f y = 0⁻¹) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_undefined_value_l1032_103201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_C_in_arithmetic_sine_triangle_l1032_103214

theorem min_cos_C_in_arithmetic_sine_triangle :
  ∀ (A B C : ℝ) (a b c : ℝ),
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  Real.sin A / a = Real.sin B / b ∧ Real.sin B / b = Real.sin C / c ∧
  Real.sin A + Real.sin B = 2 * Real.sin C →
  Real.cos C ≥ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_C_in_arithmetic_sine_triangle_l1032_103214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1032_103232

noncomputable def time_to_complete_together (time_A time_B : ℝ) : ℝ :=
  1 / (1 / time_A + 1 / time_B)

theorem work_completion_time (time_john time_rose : ℝ) 
  (h1 : time_john = 10)
  (h2 : time_rose = 40) :
  time_to_complete_together time_john time_rose = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1032_103232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_and_form_l1032_103210

-- Define the curves
def curve1 (x y : ℝ) : Prop := x = y^3
def curve2 (x y : ℝ) : Prop := x + y = 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | curve1 p.1 p.2 ∧ curve2 p.1 p.2}

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem intersection_distance_and_form :
  (∃! p, p ∈ intersection_points) ∧
  (∀ p q, p ∈ intersection_points → q ∈ intersection_points → distance p q = 0) ∧
  (∃ u v w : ℝ, u = 0 ∧ v = 0 ∧ w = 1 ∧
    (∀ p q, p ∈ intersection_points → q ∈ intersection_points →
      distance p q = Real.sqrt (u + v * Real.sqrt w))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_and_form_l1032_103210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1032_103242

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The equation of the ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The foci and upper vertex of the ellipse -/
noncomputable def foci_and_vertex (E : Ellipse) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  let c := (E.a^2 - E.b^2).sqrt
  (-c, 0, c, 0, 0, E.b)

/-- The triangle formed by the foci and upper vertex is an isosceles right triangle with area 1 -/
def triangle_property (E : Ellipse) : Prop :=
  let (x1, y1, x2, y2, x3, y3) := foci_and_vertex E
  (x2 - x1)^2 + (y2 - y1)^2 = (x3 - x1)^2 + (y3 - y1)^2 ∧
  (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) = 0 ∧
  (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) = 2

/-- The equation of the line intersecting the ellipse -/
def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  y = -x + m

/-- The circle passing through the intersection points is tangent to the y-axis -/
def circle_tangent_property (E : Ellipse) (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    ellipse_equation E x1 y1 ∧
    ellipse_equation E x2 y2 ∧
    line_equation m x1 y1 ∧
    line_equation m x2 y2 ∧
    (x1 + x2)^2 = 4 * ((x1 - x2)^2 + (y1 - y2)^2) / 9

/-- The main theorem -/
theorem ellipse_theorem (E : Ellipse) :
  triangle_property E →
  (∀ x y, ellipse_equation E x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ m, circle_tangent_property E m → m = (6:ℝ).sqrt / 2 ∨ m = -(6:ℝ).sqrt / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1032_103242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezium_area_l1032_103240

/-- The area of a trapezium with given dimensions -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: Area of a specific trapezium -/
theorem specific_trapezium_area :
  let a : ℝ := 28 -- Length of the longer parallel side
  let b : ℝ := 20 -- Length of the shorter parallel side
  let h : ℝ := 21 -- Distance between parallel sides
  trapezium_area a b h = 504 := by
  -- Unfold the definition and simplify
  unfold trapezium_area
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezium_area_l1032_103240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1032_103289

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.B = 3/5

-- Theorem for part I
theorem part_one (t : Triangle) :
  triangle_conditions t → t.A = π/6 → t.a = 5/4 := by sorry

-- Theorem for part II
theorem part_two (t : Triangle) :
  triangle_conditions t → (1/2 * t.a * t.c * Real.sin t.B = 3) → t.a + t.c = 2 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1032_103289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_equals_63_sum_of_b_is_neg_one_or_zero_l1032_103220

def f (n m : ℕ) : ℚ := (Nat.factorial n) / ((Nat.factorial (n - m)) * (Nat.factorial m))

def a (m : ℕ) : ℚ := f 6 m

def b (n m : ℕ) : ℤ := (-1)^m * m * (f n m).num

theorem sum_of_a_equals_63 :
  (Finset.range 12).sum (λ i => a (i + 1)) = 63 := by sorry

theorem sum_of_b_is_neg_one_or_zero (n : ℕ) :
  (Finset.range (2 * n)).sum (λ i => b n (i + 1)) ∈ ({-1, 0} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_equals_63_sum_of_b_is_neg_one_or_zero_l1032_103220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1032_103241

theorem problem_statement (a b c : ℤ) 
  (ha : a = 1)
  (hb : b = -b)
  (hc : c = 2)
  : (2*a + 3*c) * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1032_103241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_line_through_points_l1032_103282

/-- The angle corresponding to the slope of a line passing through two points -/
noncomputable def angle_from_slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  let slope := (y2 - y1) / (x2 - x1)
  Real.arctan slope * (180 / Real.pi) + 180 * (if slope < 0 then 1 else 0)

/-- Theorem: The angle corresponding to the slope of a line passing through (1,0) and (-2,3) is 135° -/
theorem angle_of_line_through_points : 
  angle_from_slope 1 0 (-2) 3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_line_through_points_l1032_103282
