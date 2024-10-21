import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_equals_one_l481_48189

def A : Set Int := {-1, 0, 2}

def B : Set Int := {x | ∃ y ∈ A, x = -y ∧ (2 - y) ∉ A}

theorem B_equals_one : B = {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_equals_one_l481_48189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l481_48135

-- Define the two curves
def curve1 (x : ℝ) : ℝ := x^2
def curve2 (x : ℝ) : ℝ := x^3

-- Define the area of the closed figure
noncomputable def area : ℝ := ∫ x in (0:ℝ)..1, (curve1 x - curve2 x)

-- Theorem statement
theorem area_between_curves : area = 1/12 := by
  -- Unfold the definition of area
  unfold area
  -- Evaluate the integral
  simp [curve1, curve2]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l481_48135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_transformation_l481_48113

-- Define the solution set type
def SolutionSet (P : ℝ → Prop) : Set ℝ := {x : ℝ | P x}

-- Define the quadratic inequality type
def QuadraticInequality (a b c : ℝ) : ℝ → Prop := λ x => a * x^2 - b * x + c < 0

-- Define the second quadratic inequality type
def SecondQuadraticInequality (a b c : ℝ) : ℝ → Prop := λ x => b * x^2 + a * x + c < 0

theorem solution_set_transformation 
  (a b c : ℝ) 
  (h : SolutionSet (QuadraticInequality a b c) = Set.Ioo (-2) 3) :
  SolutionSet (SecondQuadraticInequality a b c) = Set.Ioo (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_transformation_l481_48113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_vertical_asymptote_l481_48127

noncomputable section

/-- The function g(x) with parameter c -/
def g (c : ℝ) (x : ℝ) : ℝ := (x^2 - 3*x + c) / (x^2 - 5*x + 6)

/-- The denominator of g(x) -/
def denominator (x : ℝ) : ℝ := x^2 - 5*x + 6

/-- A function has a vertical asymptote at x = a if the limit of the function as x approaches a is undefined -/
def has_vertical_asymptote (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - a| ∧ |x - a| < δ → |f x - L| < ε

/-- The main theorem stating that g(x) has exactly one vertical asymptote iff c = 2 -/
theorem g_has_one_vertical_asymptote (c : ℝ) :
  (∃! (a : ℝ), has_vertical_asymptote (g c) a) ↔ c = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_vertical_asymptote_l481_48127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_when_floor_sqrt_n_is_5_l481_48128

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define our theorem
theorem largest_n_when_floor_sqrt_n_is_5 :
  ∃ (n : ℕ), (floor (Real.sqrt (n : ℝ)) = 5) ∧ 
  (∀ m : ℕ, floor (Real.sqrt (m : ℝ)) = 5 → m ≤ n) ∧
  (n = 35) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_when_floor_sqrt_n_is_5_l481_48128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l481_48193

theorem union_cardinality (M N : Finset ℕ) : 
  M = {0, 1, 2, 3, 4} → N = {1, 3, 5} → Finset.card (M ∪ N) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l481_48193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l481_48119

noncomputable def line_l (x : ℝ) : ℝ := Real.sqrt 3 * x

def curve_C (x y : ℝ) : Prop := (x - 2)^2 + (y - Real.sqrt 3)^2 = 3

def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ y = line_l x ∧ curve_C x y}

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

theorem intersection_distance_product :
  ∀ (A B : ℝ × ℝ), A ∈ intersection_points → B ∈ intersection_points → 
  distance_from_origin A * distance_from_origin B = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l481_48119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_unit_circle_l481_48183

theorem max_distance_on_unit_circle (z : ℂ) (h : Complex.abs z = 1) :
  ∃ w : ℂ, Complex.abs w = 1 ∧ Complex.abs (w - (3 - 4*I)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_unit_circle_l481_48183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_iff_m_in_range_union_when_intersection_has_one_integer_l481_48192

-- Define sets A and B
def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*m*x + m^2 - 1 < 0}

-- Define the complement of A
def A_complement : Set ℝ := {x | x ≤ -2 ∨ x ≥ 1}

-- Theorem 1
theorem intersection_empty_iff_m_in_range (m : ℝ) :
  (A_complement ∩ B m) = ∅ ↔ 0 ≤ m ∧ m ≤ 1 :=
sorry

-- Theorem 2
theorem union_when_intersection_has_one_integer (m : ℝ) :
  (∃ (n : ℤ), A ∩ B m = {(n : ℝ)}) →
  ((1 ≤ m ∧ m < 2 → A ∪ B m = Set.Ioo (-1 - m) 1) ∧
   (-1 < m ∧ m ≤ 0 → A ∪ B m = Set.Ioo (-2) (1 - m))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_iff_m_in_range_union_when_intersection_has_one_integer_l481_48192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_special_list_l481_48104

/-- The sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list where each integer n from 1 to 150 appears n times -/
def special_list : List ℕ := sorry

theorem median_of_special_list :
  let total_elements : ℕ := triangular_number 150
  let median_position : ℕ := total_elements / 2 + 1
  List.length special_list = total_elements ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 150 → List.count n special_list = n) →
  special_list.get? (median_position - 1) = some 106 :=
by sorry

#check median_of_special_list

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_special_list_l481_48104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l481_48105

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (n / 10) % 10 = 4 ∧ 
  (∃ a b : ℕ, ({a, b} : Finset ℕ) = {3, 7} ∧ n = 100 * a + 40 + b)

theorem smallest_valid_number : 
  ∀ n : ℕ, is_valid_number n → n ≥ 347 :=
by
  intro n h
  -- The proof steps would go here
  sorry

#check smallest_valid_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l481_48105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l481_48134

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / (x^2 - 1)

-- Define the interval [2, 6]
def interval : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}

-- State the theorem
theorem max_value_of_f :
  ∃ (y : ℝ), y = 2/3 ∧ ∀ (x : ℝ), x ∈ interval → f x ≤ y := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l481_48134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_complement_equality_l481_48112

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem union_complement_equality :
  A ∪ (U \ B) = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_complement_equality_l481_48112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_count_l481_48163

-- Define the type for cube faces
inductive Face : Type
| one | two | three | four | five | six

-- Define the type for colors
inductive Color : Type
| red | blue | green | yellow | purple | orange

-- Define the function that pairs opposite faces
def oppositeFace : Face → Face
| Face.one => Face.six
| Face.two => Face.five
| Face.three => Face.four
| Face.four => Face.three
| Face.five => Face.two
| Face.six => Face.one

-- Define a coloring of the cube
def Coloring := Face → Color

-- Define a valid coloring
def validColoring (c : Coloring) : Prop :=
  ∀ f1 f2 : Face, f1 ≠ f2 → c f1 ≠ c f2

-- Define equivalent colorings under cube rotations
def equivalentColoring (c1 c2 : Coloring) : Prop :=
  ∃ (perm : Face → Face), ∀ f : Face, c1 f = c2 (perm f)

-- The main theorem
theorem cube_coloring_count :
  ∃ (colorings : Finset Coloring),
    (∀ c, c ∈ colorings → validColoring c) ∧
    (∀ c1 c2, c1 ∈ colorings → c2 ∈ colorings → c1 ≠ c2 → ¬equivalentColoring c1 c2) ∧
    colorings.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_count_l481_48163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l481_48188

-- Define the slopes of the lines
noncomputable def slope_l1 (m : ℝ) : ℝ := -1 / m
noncomputable def slope_l2 (m : ℝ) : ℝ := -(m - 2) / 3

-- Define the condition for parallel lines
def parallel_condition (m : ℝ) : Prop := slope_l1 m = slope_l2 m

-- Theorem statement
theorem parallel_lines_m_value :
  ∀ m : ℝ, m ≠ 0 → parallel_condition m → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l481_48188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_of_nine_moles_l481_48136

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in C7H6O2 -/
def carbon_count : ℕ := 7

/-- The number of Hydrogen atoms in C7H6O2 -/
def hydrogen_count : ℕ := 6

/-- The number of Oxygen atoms in C7H6O2 -/
def oxygen_count : ℕ := 2

/-- The number of moles of C7H6O2 -/
def mole_count : ℝ := 9

/-- The molecular weight of C7H6O2 in g/mol -/
def molecular_weight : ℝ :=
  carbon_weight * (carbon_count : ℝ) +
  hydrogen_weight * (hydrogen_count : ℝ) +
  oxygen_weight * (oxygen_count : ℝ)

/-- Theorem stating the molecular weight of 9 moles of C7H6O2 -/
theorem molecular_weight_of_nine_moles :
  molecular_weight * mole_count = 1099.062 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_of_nine_moles_l481_48136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_f_is_odd_l481_48180

-- Define the function f(x) = tan(1/2 * x)
noncomputable def f (x : ℝ) : ℝ := Real.tan (1/2 * x)

-- Statement for the smallest positive period
theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = 2 * Real.pi := by
  sorry

-- Statement for f being an odd function
theorem f_is_odd : ∀ (x : ℝ), f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_f_is_odd_l481_48180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_current_age_l481_48195

-- Define the ages at the time Pat built the pigsty
def pat_age_at_pigsty : ℚ := 0 -- We'll leave this as 0 for now
def mike_age_at_pigsty : ℚ := 10/3
def biddy_age_at_pigsty : ℚ := 0 -- We'll leave this as 0 for now

-- Define current ages
def pat_current_age : ℚ := 0 -- We'll leave this as 0 for now
def mike_current_age : ℚ := 0 -- We'll leave this as 0 for now
def biddy_current_age : ℚ := 0 -- We'll leave this as 0 for now

-- State the theorem
theorem mikes_current_age :
  -- Pat's current age is 1 1/3 times his age when he built the pigsty
  pat_current_age = 4/3 * pat_age_at_pigsty →
  -- Mike is now 2 years older than half of Biddy's age when Pat built the pigsty
  mike_current_age = biddy_age_at_pigsty / 2 + 2 →
  -- When Mike reaches Pat's age at pigsty, the combined age will be 100
  pat_current_age + biddy_current_age + (pat_age_at_pigsty - mike_current_age + mike_current_age) = 100 →
  -- Mike's current age is 10 16/21 years
  mike_current_age = 226/21 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_current_age_l481_48195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportionality_check_l481_48161

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := x + 1
noncomputable def f2 (x : ℝ) : ℝ := -x^2
noncomputable def f3 (x : ℝ) : ℝ := x / 5
noncomputable def f4 (x : ℝ) : ℝ := 5 / x

-- Define direct proportionality
def is_direct_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

-- Theorem statement
theorem direct_proportionality_check :
  ¬ is_direct_proportional f1 ∧
  ¬ is_direct_proportional f2 ∧
  is_direct_proportional f3 ∧
  ¬ is_direct_proportional f4 := by
  sorry

#check direct_proportionality_check

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportionality_check_l481_48161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_10_solutions_l481_48103

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else 2*x

-- State the theorem
theorem f_equals_10_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 10 ∧ f x₂ = 10 ∧
  ∀ (y : ℝ), f y = 10 → y = x₁ ∨ y = x₂ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_10_solutions_l481_48103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_t_l481_48175

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^2)) + 3

-- State the theorem
theorem f_negative_t (t : ℝ) (h : f t = 7) : f (-t) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_t_l481_48175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l481_48154

noncomputable def a (k : ℝ) : Fin 2 → ℝ := ![2, k]
def b : Fin 2 → ℝ := ![1, 1]

def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2)

theorem vector_problem (k : ℝ) :
  (dot_product b (fun i => (a k i) - 3 * (b i)) = 0) →
  (k = 4 ∧ 
   (dot_product (a 4) b) / (magnitude (a 4) * magnitude b) = 3 * Real.sqrt 10 / 10) := by
  sorry

#check vector_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l481_48154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_l481_48170

/-- The function f(x) = -1/3x³ + 1/2x² + 2x takes its minimum value when x = -1 -/
theorem function_minimum :
  let f := fun (x : ℝ) ↦ -1/3 * x^3 + 1/2 * x^2 + 2*x
  ∀ y : ℝ, f (-1) ≤ f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_l481_48170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_odd_f_not_odd_l481_48178

noncomputable def f (x : ℝ) := Real.sin (x^2 + 1)

theorem sine_odd (x : ℝ) : Real.sin (-x) = -Real.sin x := by sorry

theorem f_not_odd : ¬(∀ x : ℝ, f (-x) = -f x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_odd_f_not_odd_l481_48178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l481_48139

theorem problem_statement :
  (∀ x : ℝ, x^2 + 1 > 0) ∧
  ¬(∀ x : ℝ, Real.sin x = 2) ∧
  ((∀ x : ℝ, x^2 + 1 > 0) ∨ (∀ x : ℝ, Real.sin x = 2)) ∧
  ¬¬(∀ x : ℝ, x^2 + 1 > 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l481_48139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_chord_length_when_m_zero_l481_48166

noncomputable section

-- Define the line l
def line_l (m : ℝ) (t : ℝ) : ℝ × ℝ :=
  (1/2 * t, m + (Real.sqrt 3)/2 * t)

-- Define the curve C in polar coordinates
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 2*ρ*(Real.cos θ) - 4 = 0

-- Define the condition for no common points
def no_common_points (m : ℝ) : Prop :=
  ∀ t, ¬ ∃ θ ρ, curve_C ρ θ ∧ line_l m t = (ρ * Real.cos θ, ρ * Real.sin θ)

-- Theorem for part (1)
theorem range_of_m (m : ℝ) :
  no_common_points m ↔ (m < -Real.sqrt 3 - 2 * Real.sqrt 5 ∨ m > -Real.sqrt 3 + 2 * Real.sqrt 5) :=
by sorry

-- Define the chord length function
def chord_length (m : ℝ) : ℝ :=
  let ρ₁ := (1 + Real.sqrt 17) / 2
  let ρ₂ := (1 - Real.sqrt 17) / 2
  abs (ρ₁ - ρ₂)

-- Theorem for part (2)
theorem chord_length_when_m_zero :
  chord_length 0 = Real.sqrt 17 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_chord_length_when_m_zero_l481_48166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_probability_l481_48184

/-- The probability of not eating a green candy after a red candy when selecting 5 from 12 candies -/
def prob_not_green_after_red (green red selected : ℕ) : ℚ :=
  1 - (green / (green + red - 1))

/-- The sum of the numerator and denominator of the simplified fraction representing the probability -/
def sum_fraction_parts (p : ℚ) : ℕ :=
  (p.num.natAbs.gcd p.den) + (p.den / p.num.natAbs.gcd p.den)

theorem candy_probability :
  let green := 8
  let red := 4
  let selected := 5
  let p := prob_not_green_after_red green red selected
  sum_fraction_parts p = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_probability_l481_48184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_technicians_average_salary_l481_48152

/-- Proves that given 21 workers with an average salary of $8000, and 14 non-technicians
    with an average salary of $6000, the average salary of the remaining 7 technicians is $12,000. -/
theorem technicians_average_salary
  (total_workers : ℕ)
  (total_average : ℝ)
  (non_technicians : ℕ)
  (non_technicians_average : ℝ)
  (technicians : ℕ)
  (h_total_workers : total_workers = 21)
  (h_total_average : total_average = 8000)
  (h_non_technicians : non_technicians = 14)
  (h_non_technicians_average : non_technicians_average = 6000)
  (h_technicians : technicians = total_workers - non_technicians) :
  (total_average * (total_workers : ℝ) - non_technicians_average * (non_technicians : ℝ)) / (technicians : ℝ) = 12000 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_technicians_average_salary_l481_48152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_half_cube_l481_48179

open Real

theorem max_volume_half_cube (a : ℝ) (h : a > 0) :
  let V (b m : ℝ) := b^2 * m
  let constraint (b m : ℝ) := b + m = a / sqrt 2
  ∃ (b m : ℝ), constraint b m ∧ 
    (∀ (b' m' : ℝ), constraint b' m' → V b m ≥ V b' m') ∧
    V b m = (a^3 * sqrt 2) / 27 :=
by
  sorry

#check max_volume_half_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_half_cube_l481_48179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_inverse_l481_48116

theorem function_composition_inverse (a c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x + c
  let g : ℝ → ℝ := λ x ↦ -4 * x + 6
  let h : ℝ → ℝ := λ x ↦ f (g x)
  (∀ x, (h (x + 8) = x)) → a - c = 25 / 4 := by
  sorry

#check function_composition_inverse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_inverse_l481_48116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_theorem_l481_48160

/-- A class of students with friendship relations -/
structure FriendshipClass where
  n : ℕ
  s : ℕ
  t : ℕ
  friends : Finset (Fin n × Fin n)
  triples : Finset (Fin n × Fin n × Fin n)
  n_ge_3 : n ≥ 3
  s_ge_1 : s ≥ 1
  t_ge_1 : t ≥ 1
  friends_symmetric : ∀ x y, (x, y) ∈ friends ↔ (y, x) ∈ friends
  friends_count : s = (friends.filter (fun (x, y) => x < y)).card
  triples_count : t = triples.card
  triples_are_friends : ∀ x y z, (x, y, z) ∈ triples → (x, y) ∈ friends ∧ (y, z) ∈ friends ∧ (z, x) ∈ friends

/-- The number of common friends between two students -/
def common_friends (fc : FriendshipClass) (x y : Fin fc.n) : ℕ :=
  (fc.friends.filter (fun (z, _) => (z, x) ∈ fc.friends ∧ (z, y) ∈ fc.friends)).card

/-- The main theorem -/
theorem friendship_theorem (fc : FriendshipClass) :
  ∃ u v w : Fin fc.n, (u, v, w) ∈ fc.triples ∧
    common_friends fc u v + common_friends fc v w + common_friends fc w u ≥ 9 * fc.t / fc.s :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_theorem_l481_48160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_iff_distinct_products_l481_48101

theorem prime_iff_distinct_products (n : ℕ) (n₁ n₂ n₃ n₄ : ℕ) 
  (h₁ : n ≥ 5)
  (h₂ : n = n₁ + n₂ + n₃ + n₄)
  (h₃ : n₁ > 0 ∧ n₂ > 0 ∧ n₃ > 0 ∧ n₄ > 0) :
  Nat.Prime n ↔ 
  ∀ (σ : Equiv.Perm (Fin 4)), 
    (σ 0 ≠ σ 1 ∧ σ 2 ≠ σ 3) → 
    (List.get! [n₁, n₂, n₃, n₄] (σ 0).val * List.get! [n₁, n₂, n₃, n₄] (σ 1).val ≠
     List.get! [n₁, n₂, n₃, n₄] (σ 2).val * List.get! [n₁, n₂, n₃, n₄] (σ 3).val) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_iff_distinct_products_l481_48101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_sqrt_is_sqrt_2_l481_48167

-- Define a function to check if a square root is in its simplest form
noncomputable def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y ^ 2 * x ≠ (⌊y ^ 2 * x⌋ : ℝ)

-- Define the set of given square roots
def sqrt_options : Set ℝ := {Real.sqrt 20, Real.sqrt 2, Real.sqrt (1/2), Real.sqrt 0.2}

-- Theorem statement
theorem simplest_sqrt_is_sqrt_2 :
  ∃ (x : ℝ), x ∈ sqrt_options ∧ is_simplest_sqrt x ∧ x = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_sqrt_is_sqrt_2_l481_48167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_periodic_two_digits_denominators_l481_48159

def is_purely_periodic_two_digits (a : ℕ) : Prop :=
  ∃ (b c : ℕ), b < 10 ∧ c < 10 ∧ (1 : ℚ) / a = (10 * b + c : ℚ) / 99

theorem purely_periodic_two_digits_denominators :
  ∀ a : ℕ, a > 0 → (is_purely_periodic_two_digits a ↔ a ∈ ({11, 33, 99} : Set ℕ)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_periodic_two_digits_denominators_l481_48159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_fourth_plus_square_plus_one_eq_zero_l481_48190

-- Define ω as a complex number
noncomputable def ω : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I

-- Theorem statement
theorem omega_fourth_plus_square_plus_one_eq_zero :
  ω^4 + ω^2 + 1 = 0 := by
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_fourth_plus_square_plus_one_eq_zero_l481_48190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selling_decision_l481_48110

/-- Represents the selling decision for a batch of goods -/
inductive SellingDecision
  | ThisMonth
  | NextMonth
  | EitherMonth

/-- Determines the optimal selling decision based on the cost price -/
noncomputable def optimalSellingDecision (a : ℝ) : SellingDecision :=
  if a > 2900 then SellingDecision.ThisMonth
  else if a < 2900 then SellingDecision.NextMonth
  else SellingDecision.EitherMonth

/-- Calculates the profit if sold this month -/
noncomputable def profitThisMonth (a : ℝ) : ℝ := 5 / 1000 * a + 100.5

/-- Calculates the profit if sold next month -/
def profitNextMonth : ℝ := 115

theorem optimal_selling_decision (a : ℝ) :
  (optimalSellingDecision a = SellingDecision.ThisMonth → profitThisMonth a > profitNextMonth) ∧
  (optimalSellingDecision a = SellingDecision.NextMonth → profitThisMonth a < profitNextMonth) ∧
  (optimalSellingDecision a = SellingDecision.EitherMonth → profitThisMonth a = profitNextMonth) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selling_decision_l481_48110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l481_48146

theorem rationalize_denominator :
  ∀ x : ℝ, x > 0 → Real.sqrt (5 / (3 - Real.sqrt 2)) = (Real.sqrt (105 + 35 * Real.sqrt 2)) / 7 :=
by
  intros x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l481_48146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_fixed_point_of_tangent_chord_l481_48155

-- Define the circle E
def E (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l with slope k
def L (k : ℝ) (x y : ℝ) : Prop := y = k * x - 4

-- Define the line l: x - y - 4 = 0
def L' (x y : ℝ) : Prop := x - y - 4 = 0

-- Define a point on the circle
def OnCircle (x y : ℝ) : Prop := E x y

-- Define a point on the line with slope k
def OnLine (k : ℝ) (x y : ℝ) : Prop := L k x y

-- Define a point on the line x - y - 4 = 0
def OnLine' (x y : ℝ) : Prop := L' x y

-- Define the angle between two points and the origin
noncomputable def Angle (x1 y1 x2 y2 : ℝ) : ℝ := sorry

-- Define a tangent line to the circle
def Tangent (x0 y0 x y : ℝ) : Prop := sorry

-- Part 1 theorem
theorem slope_of_line (k : ℝ) :
  (∃ x1 y1 x2 y2, OnCircle x1 y1 ∧ OnCircle x2 y2 ∧
    OnLine k x1 y1 ∧ OnLine k x2 y2 ∧ 
    x1 ≠ x2 ∧ y1 ≠ y2 ∧
    Angle x1 y1 x2 y2 = π/6) →
  k = Real.sqrt 15 ∨ k = -Real.sqrt 15 := by
  sorry

-- Part 2 theorem
theorem fixed_point_of_tangent_chord :
  (∀ x0 y0, OnLine' x0 y0 →
    ∃ x1 y1 x2 y2, OnCircle x1 y1 ∧ OnCircle x2 y2 ∧
      Tangent x0 y0 x1 y1 ∧ Tangent x0 y0 x2 y2 →
      ∃ t, x1 + t*(x2 - x1) = 1 ∧ y1 + t*(y2 - y1) = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_fixed_point_of_tangent_chord_l481_48155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l481_48165

/-- The speed of a train in km/h, given its length and time to cross a stationary point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (3600 / 1000)

/-- Theorem: A 240-meter long train that takes 6 seconds to cross a man has a speed of 144 km/h -/
theorem train_speed_calculation :
  train_speed 240 6 = 144 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l481_48165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_midpoints_l481_48158

noncomputable section

-- Define the circle S
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
def PointOnCircle (S : Circle) (A : ℝ × ℝ) : Prop :=
  let (x, y) := A
  let (cx, cy) := S.center
  (x - cx)^2 + (y - cy)^2 = S.radius^2

-- Define a chord passing through A
def ChordThroughA (S : Circle) (A : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  PointOnCircle S A ∧ PointOnCircle S P ∧ PointOnCircle S Q ∧
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ A = (t • P + (1 - t) • Q)

-- Define the midpoint of a chord
def MidpointOfChord (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- The main theorem
theorem locus_of_midpoints (S : Circle) (A : ℝ × ℝ) :
  PointOnCircle S A →
  ∃ S' : Circle, S'.center = ((S.center.1 + A.1) / 2, (S.center.2 + A.2) / 2) ∧
                 S'.radius = S.radius / 2 ∧
                 ∀ P Q : ℝ × ℝ, ChordThroughA S A P Q →
                   PointOnCircle S' (MidpointOfChord P Q) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_midpoints_l481_48158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_a_half_unique_zero_point_range_decreasing_function_condition_l481_48123

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (4 : ℝ) ^ x - (2 : ℝ) ^ x + 1

theorem range_when_a_half :
  let a : ℝ := 1/2
  ∀ y : ℝ, y ≥ 1/2 → ∃ x : ℝ, f a x = y := by
  sorry

theorem unique_zero_point_range (a : ℝ) :
  (∃! x : ℝ, x ∈ Set.Ioo 0 1 ∧ f a x = 0) → 0 < a ∧ a < 1/4 := by
  sorry

theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_a_half_unique_zero_point_range_decreasing_function_condition_l481_48123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_proposition_l481_48162

theorem negation_of_existence {α : Type*} (p : α → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) := by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x > 0 ∧ |x| ≤ 1) ↔ (∀ x : ℝ, x > 0 → |x| > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_proposition_l481_48162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l481_48114

theorem trig_inequality : 
  Real.tan (7 * Real.pi / 5) > Real.sin (2 * Real.pi / 5) ∧ 
  Real.sin (2 * Real.pi / 5) > Real.cos (6 * Real.pi / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l481_48114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l481_48117

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  -- Period of f is π
  (∀ x, f (x + Real.pi) = f x) ∧
  -- (2π/3, -2) is a lowest point
  f (2 * Real.pi / 3) = -2 ∧
  -- Minimum value on [0, π/12]
  (∀ x ∈ Set.Icc 0 (Real.pi / 12), f x ≥ 1) ∧
  f 0 = 1 ∧
  -- Maximum value on [0, π/12]
  (∀ x ∈ Set.Icc 0 (Real.pi / 12), f x ≤ Real.sqrt 3) ∧
  f (Real.pi / 12) = Real.sqrt 3 :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l481_48117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_angle_relation_l481_48107

theorem triangle_side_angle_relation (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  c * Real.sin A = a * Real.sin C →
  c * Real.sin B = b * Real.sin A →
  2 * c^2 - 2 * a^2 = b^2 →
  2 * c * Real.cos A - 2 * a * Real.cos C = b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_angle_relation_l481_48107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_opposite_parts_l481_48131

theorem complex_number_opposite_parts (b : ℝ) : 
  let z := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = - z.im) → b = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_opposite_parts_l481_48131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_grass_area_l481_48140

theorem remaining_grass_area (plot_diameter : ℝ) (path_width : ℝ) 
  (h1 : plot_diameter = 12)
  (h2 : path_width = 3) :
  30 * Real.pi - 9 * Real.sqrt 3 = 
    let plot_radius : ℝ := plot_diameter / 2
    let plot_area : ℝ := Real.pi * plot_radius^2
    let chord_length : ℝ := 2 * plot_radius * Real.sqrt 3 / 2
    let sector_area : ℝ := (1/3) * plot_area
    let triangle_area : ℝ := (1/2) * chord_length * (path_width / 2)
    let segment_area : ℝ := sector_area - triangle_area
    plot_area - segment_area :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_grass_area_l481_48140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l481_48111

/-- The probability of the specific outcome in the game -/
theorem game_probability : 
  ∀ (p_alex p_mel p_chelsea : ℝ),
  p_alex = 3/5 →
  p_mel = 3 * p_chelsea →
  p_alex + p_mel + p_chelsea = 1 →
  (p_alex^4 * p_mel^3 * p_chelsea) * (Nat.choose 8 4 * Nat.choose 4 3) = 61242/625000 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l481_48111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l481_48176

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3*x + 2)

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y, x < y ∧ y < 1 → f x > f y :=
by
  -- The proof is omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l481_48176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_medians_condition_right_triangle_medians_ratio_l481_48126

/-- The condition for a triangle, constructed from the medians of a right triangle as its sides, to also be a right triangle. -/
theorem right_triangle_medians_condition (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_order : a < b ∧ b < c) :
  (∃ k : ℝ, a = k ∧ b = k * Real.sqrt 2) ↔
  (b^2 + (a/2)^2)^2 = (a^2 + (b/2)^2)^2 + ((a^2 + b^2)/4)^2 :=
sorry

/-- The ratio of the legs of the right triangle must be 1 : √2 -/
theorem right_triangle_medians_ratio (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_order : a < b ∧ b < c) :
  (∃ k : ℝ, a = k ∧ b = k * Real.sqrt 2) ↔
  b = a * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_medians_condition_right_triangle_medians_ratio_l481_48126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l481_48122

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 - a*x + a else (4 - 2*a)^x

-- State the theorem
theorem monotonic_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x ≤ f a y) →
  (3/2 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l481_48122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_b_onto_a_is_correct_l481_48129

noncomputable def vector_a : ℝ × ℝ := (-4, 3)
noncomputable def vector_b : ℝ × ℝ := (5, 12)

/-- The projection of vector b onto vector a -/
noncomputable def projection_b_onto_a : ℝ × ℝ := (-64/25, 48/25)

/-- Theorem stating that the projection of b onto a is correct -/
theorem projection_b_onto_a_is_correct :
  let a := vector_a
  let b := vector_b
  let proj := projection_b_onto_a
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  let scalar := (dot_product / (magnitude_a * magnitude_a)) * magnitude_b
  (scalar * a.1, scalar * a.2) = proj :=
by sorry

#check projection_b_onto_a_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_b_onto_a_is_correct_l481_48129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l481_48125

noncomputable def f (x : ℝ) : ℝ := (2*x - 3) / Real.sqrt (x - 7)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 7} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l481_48125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_is_zero_l481_48120

def is_valid_rectangle (l w : ℕ) : Prop :=
  (l + w) * 2 = 200 ∧ (l ≥ 25 ∨ w ≥ 25)

def area (l w : ℕ) : ℕ := l * w

theorem max_area_difference_is_zero :
  ∀ l₁ w₁ l₂ w₂ : ℕ,
    is_valid_rectangle l₁ w₁ →
    is_valid_rectangle l₂ w₂ →
    (area l₁ w₁ : ℤ) - (area l₂ w₂ : ℤ) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_is_zero_l481_48120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scale_model_height_theorem_l481_48102

/-- Represents a lighthouse with its height and water capacity -/
structure Lighthouse where
  height : ℝ
  waterCapacity : ℝ

/-- Calculates the height of a scale model lighthouse -/
noncomputable def scaleModelHeight (original : Lighthouse) (modelCapacity : ℝ) : ℝ :=
  original.height / (original.waterCapacity / modelCapacity) ^ (1/3)

theorem scale_model_height_theorem (original : Lighthouse) (modelCapacity : ℝ) :
  original.height = 90 →
  original.waterCapacity = 500000 →
  modelCapacity = 0.2 →
  ∃ ε > 0, |scaleModelHeight original modelCapacity - 0.66| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scale_model_height_theorem_l481_48102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_adjusted_sale_price_l481_48124

/-- The adjusted sale price for an article with variable production costs and bulk discounts -/
noncomputable def adjusted_sale_price (X Y : ℝ) : ℝ :=
  (992 + 3.1 * X) * (1 - Y / 100)

/-- Theorem stating the correct adjusted sale price for 55% profit -/
theorem correct_adjusted_sale_price 
  (cost_price : ℝ) 
  (X : ℝ) -- production cost variation per 100 articles
  (Y : ℝ) -- bulk discount percentage
  (h1 : cost_price = (832 + 448) / 2) -- cost price is midpoint between profit and loss prices
  (h2 : cost_price = 640) -- derived from h1
  (h3 : ∀ quantity : ℝ, quantity ≥ 200 → 
    adjusted_sale_price X Y = (cost_price + 2 * X) * 1.55 * (1 - Y / 100)) :
  adjusted_sale_price X Y = (992 + 3.1 * X) * (1 - Y / 100) := by
  sorry

#check correct_adjusted_sale_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_adjusted_sale_price_l481_48124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_price_reduction_l481_48198

/-- Proves that given a 40% reduction in the price of bananas, if a man can buy 64 more bananas
    for Rs. 40.00001 after the reduction, then the reduced price per dozen is Rs. 3.00000075. -/
theorem banana_price_reduction (original_price : ℚ) :
  let reduction_percentage : ℚ := 40 / 100
  let reduced_price := original_price * (1 - reduction_percentage)
  let original_quantity := (40.00001 / original_price).floor
  let new_quantity := original_quantity + 64 / 12
  (40.00001 / original_price).floor * original_price = new_quantity * reduced_price →
  reduced_price = 3.00000075 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_price_reduction_l481_48198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_intervals_l481_48147

-- Define the function f(x) = |x^2 - 2x - 3|
def f (x : ℝ) : ℝ := abs (x^2 - 2*x - 3)

-- Define the property of being monotonically increasing on an interval
def MonotonicIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- State the theorem
theorem f_monotonic_increasing_intervals :
  (MonotonicIncreasing f (-1) 1) ∧ 
  (∀ x : ℝ, x ≥ 3 → MonotonicIncreasing f 3 x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_intervals_l481_48147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l481_48172

-- Define the point P
noncomputable def P : ℝ × ℝ := (2, 5)

-- Define the slopes of the lines
noncomputable def slope_PQ : ℝ := 1/2
noncomputable def slope_PR : ℝ := 3

-- Define Q and R as points on the x-axis
noncomputable def Q : ℝ × ℝ := (-8, 0)
noncomputable def R : ℝ × ℝ := (1/3, 0)

-- Theorem statement
theorem area_of_triangle_PQR :
  let base := R.1 - Q.1
  let height := P.2
  (1/2 : ℝ) * base * height = 125/6 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l481_48172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l481_48151

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Theorem: The area of a triangle with vertices at (1, 1), (6, 1), and (3, -4) is 12.5 square units -/
theorem triangle_area_specific : triangleArea (1, 1) (6, 1) (3, -4) = 12.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l481_48151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_implies_sum_l481_48149

theorem log_equation_implies_sum (x y : ℝ) 
  (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^3 + (Real.log y / Real.log 3)^3 + 6 = 
       6 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^Real.sqrt 3 + y^Real.sqrt 3 = 35 := by
  sorry

#check log_equation_implies_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_implies_sum_l481_48149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_theorem_l481_48133

/-- A function that varies in inverse proportion to the square of its input -/
noncomputable def inverse_square_prop (k : ℝ) (x : ℝ) : ℝ := k / (x * x)

theorem inverse_square_theorem (k : ℝ) :
  inverse_square_prop k 12 = 40 →
  inverse_square_prop k 24 = 10 := by
  intro h
  -- Proof steps would go here
  sorry

#check inverse_square_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_theorem_l481_48133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l481_48173

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

theorem compound_interest_calculation (P : ℝ) :
  simple_interest P 5 2 = 50 →
  compound_interest P 5 2 = 51.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l481_48173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_two_l481_48109

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the constant a
def a : ℝ := sorry

-- State the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_def : ∀ x, x ≥ 0 → f x = 2^x + x + a

-- State the theorem
theorem f_neg_one_eq_neg_two : f (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_two_l481_48109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_three_is_closed_l481_48199

def is_closed_set (A : Set ℤ) : Prop :=
  ∀ (a b : ℤ), a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

def multiples_of_three : Set ℤ :=
  {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem multiples_of_three_is_closed : is_closed_set multiples_of_three := by
  intro a b ha hb
  have ⟨ka, ha⟩ := ha
  have ⟨kb, hb⟩ := hb
  constructor
  · use ka + kb
    rw [ha, hb]
    ring
  · use ka - kb
    rw [ha, hb]
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_three_is_closed_l481_48199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_x_measure_l481_48130

-- Define a triangle with an internal point
structure TriangleWithInternalPoint where
  A : Real
  B : Real
  C : Real
  X : Real
  sum_ABC : A + B + C = 180
  sum_with_x : A + B + C + X = 360

-- Theorem statement
theorem angle_x_measure (t : TriangleWithInternalPoint) 
  (h1 : t.A = 85) (h2 : t.B = 35) (h3 : t.C = 30) : t.X = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_x_measure_l481_48130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersects_parallel_plane_l481_48132

/-- A plane in 3D space -/
structure Plane where

/-- Two planes are parallel -/
def parallel (p q : Plane) : Prop :=
  sorry

/-- A plane intersects another plane -/
def intersects (p q : Plane) : Prop :=
  sorry

/-- Theorem: If a plane intersects one of two parallel planes, it also intersects the other -/
theorem intersects_parallel_plane (α β γ : Plane) 
  (h1 : parallel α β) (h2 : intersects γ α) : intersects γ β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersects_parallel_plane_l481_48132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinate_l481_48185

/-- A triangle with vertices at (3,3), (0,0), and (x, 0) where x < 0 -/
structure ObtuseTriangle where
  x : ℝ
  h_negative : x < 0

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- The theorem stating that if the area of the obtuse triangle is 12,
    then the x-coordinate of the third vertex is -8 -/
theorem third_vertex_coordinate (t : ObtuseTriangle) :
  triangleArea (3 - t.x) 3 = 12 → t.x = -8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinate_l481_48185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_2017_l481_48137

theorem complex_sum_2017 : 
  let i : ℂ := Complex.I
  ((1 + i) / (1 - i)) ^ 2017 + ((1 - i) / (1 + i)) ^ 2017 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_2017_l481_48137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_l481_48145

-- Define the function f as noncomputable due to Real.sqrt and power operation
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else (2 : ℝ)^x

-- State the theorem
theorem f_composition_negative_two :
  f (f (-2)) = 1/2 := by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_l481_48145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l481_48157

noncomputable section

-- Define the curves
def C₁ (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.sqrt 2 / Real.sin (θ + Real.pi/4)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem min_distance_between_curves :
  (∃ (α θ : ℝ), distance (C₁ α) (C₂ θ) = Real.sqrt 2) ∧
  (∀ (α θ : ℝ), distance (C₁ α) (C₂ θ) ≥ Real.sqrt 2) ∧
  (∃ (α : ℝ), C₁ α = (3/2, 1/2) ∧
    ∀ (θ : ℝ), distance (C₁ α) (C₂ θ) ≥ Real.sqrt 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l481_48157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_top_width_l481_48164

/-- Represents the cross-section of a water channel -/
structure WaterChannel where
  bottomWidth : ℝ
  depth : ℝ
  area : ℝ
  topWidth : ℝ

/-- The area of a trapezium is given by (a + b) * h / 2, where a and b are parallel sides and h is height -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

theorem water_channel_top_width (channel : WaterChannel) 
  (h1 : channel.bottomWidth = 6)
  (h2 : channel.depth = 70)
  (h3 : channel.area = 630)
  (h4 : channel.area = trapeziumArea channel.topWidth channel.bottomWidth channel.depth) :
  channel.topWidth = 12 := by
  sorry

#check water_channel_top_width

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_top_width_l481_48164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_radius_l481_48191

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 35 = 6*x + 22*y

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := Real.sqrt 95

/-- Theorem stating the existence of a center (h, k) such that the equation
    represents a circle with radius √95 -/
theorem cookie_radius :
  ∃ (h k : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = circle_radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_radius_l481_48191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_personal_tax_150k_tax_function_correct_l481_48118

/-- Personal tax calculation based on annual comprehensive income --/
noncomputable def personal_tax (annual_income : ℝ) : ℝ :=
  let basic_deduction := 60000
  let special_deduction_rate := 0.2
  let special_additional_deduction := 36000
  let other_deductions := 4000
  let taxable_income := annual_income - basic_deduction - 
                        (annual_income * special_deduction_rate) - 
                        special_additional_deduction - other_deductions
  if taxable_income ≤ 0 then 0
  else if taxable_income ≤ 36000 then taxable_income * 0.03
  else if taxable_income ≤ 144000 then taxable_income * 0.1 - 2520
  else taxable_income * 0.2 - 16920

/-- Theorem: Personal tax for 150,000 yuan annual income is 600 yuan --/
theorem personal_tax_150k :
  ∀ ε > 0, |personal_tax 150000 - 600| < ε := by
  sorry

/-- Functional expression of tax based on annual comprehensive income --/
noncomputable def tax_function (x : ℝ) : ℝ :=
  if x ≤ 125000 then 0
  else if x ≤ 170000 then 0.024 * x - 3000
  else if x ≤ 305000 then 0.08 * x - 12520
  else 0.16 * x - 36920

/-- Theorem: The tax_function correctly represents the tax calculation --/
theorem tax_function_correct :
  ∀ x, 0 ≤ x ∧ x ≤ 500000 → tax_function x = personal_tax x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_personal_tax_150k_tax_function_correct_l481_48118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hunter_cannot_guarantee_catch_l481_48106

-- Define the game setup
structure GameState where
  rabbitPos : ℝ × ℝ
  hunterPos : ℝ × ℝ
  round : ℕ

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the game rules
def validRabbitMove (oldPos newPos : ℝ × ℝ) : Prop :=
  distance oldPos newPos = 1

def validHunterMove (oldPos newPos : ℝ × ℝ) : Prop :=
  distance oldPos newPos = 1

def validLocatorFeedback (rabbitPos feedbackPos : ℝ × ℝ) : Prop :=
  distance rabbitPos feedbackPos ≤ 1

-- Define the theorem
theorem hunter_cannot_guarantee_catch (initialState : GameState)
  (h_initial : initialState.rabbitPos = initialState.hunterPos ∧ initialState.round = 0) :
  ∃ (rabbitStrategy : GameState → ℝ × ℝ)
    (locatorStrategy : GameState → ℝ × ℝ),
    ∀ (hunterStrategy : GameState → ℝ × ℝ),
    ∃ (finalState : GameState),
    finalState.round = 10^9 ∧
    distance finalState.rabbitPos finalState.hunterPos > 100 ∧
    (∀ n < 10^9,
      let state := -- state after n rounds
        { rabbitPos := rabbitStrategy state,
          hunterPos := hunterStrategy state,
          round := n }
      validRabbitMove state.rabbitPos state.rabbitPos ∧
      validLocatorFeedback state.rabbitPos (locatorStrategy state) ∧
      validHunterMove state.hunterPos state.hunterPos) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hunter_cannot_guarantee_catch_l481_48106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_square_fraction_l481_48142

/-- The area of a square with side length 1 -/
def unit_square_area : ℝ := 1

/-- The side length of the larger square in the 5x5 grid -/
def larger_square_side : ℝ := 4

/-- The side length of the shaded square (diagonal of a unit square) -/
noncomputable def shaded_square_side : ℝ := Real.sqrt 2

/-- The area of the larger square -/
def larger_square_area : ℝ := larger_square_side ^ 2

/-- The area of the shaded square -/
noncomputable def shaded_square_area : ℝ := shaded_square_side ^ 2

/-- The fraction of the larger square's area that is inside the shaded square -/
theorem shaded_square_fraction :
  shaded_square_area / larger_square_area = 1 / 8 := by
  -- Expand definitions
  unfold shaded_square_area larger_square_area shaded_square_side larger_square_side
  -- Simplify the expression
  simp [pow_two]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_square_fraction_l481_48142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_relation_l481_48108

theorem sine_cosine_relation (α : ℝ) : 
  Real.sin (π / 8 + α) = 3 / 4 → Real.cos (3 * π / 8 - α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_relation_l481_48108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_increasing_l481_48174

noncomputable def seq_A (n : ℕ) : ℚ := n / (n + 1)

noncomputable def seq_B (n : ℕ) : ℝ := -(1/2)^n

def seq_C : ℕ → ℚ
| 0 => 1
| 1 => 1
| (n + 2) => 3 - seq_C (n + 1)

noncomputable def seq_D : ℕ → ℝ
| 0 => 1
| 1 => 1
| (n + 2) => seq_D (n + 1)^2 - seq_D (n + 1) + 2

theorem sequences_increasing :
  (∀ n : ℕ, seq_A (n + 1) > seq_A n) ∧
  (∀ n : ℕ, seq_B (n + 1) > seq_B n) ∧
  (∀ n : ℕ, seq_D (n + 1) > seq_D n) ∧
  ¬(∀ n : ℕ, seq_C (n + 1) > seq_C n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_increasing_l481_48174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_vertices_coincide_m_range_l481_48144

-- Define the ellipse equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (9 - m) + y^2 / (2 * m) = 1 ∧ 9 - m > 2 * m ∧ m > 0

-- Define the hyperbola equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / 5 - y^2 / m = 1 ∧ m > 0

-- Define the eccentricity condition for the hyperbola
noncomputable def hyperbola_eccentricity (m : ℝ) : Prop :=
  let e := Real.sqrt ((5 + m) / 5)
  Real.sqrt 6 / 2 < e ∧ e < Real.sqrt 2

-- Theorem 1: Coincidence of foci and vertices
theorem foci_vertices_coincide (m : ℝ) :
  is_ellipse m ∧ is_hyperbola m → m = 4/3 := by
  sorry

-- Theorem 2: Range of m when both conditions are true
theorem m_range (m : ℝ) :
  is_ellipse m ∧ is_hyperbola m ∧ hyperbola_eccentricity m → 5/2 < m ∧ m < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_vertices_coincide_m_range_l481_48144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sylvia_height_in_meters_l481_48138

/-- Converts inches to meters -/
def inchesToMeters (inches : ℝ) : ℝ := inches * 0.0254

/-- Rounds a real number to three decimal places -/
noncomputable def roundToThreeDecimalPlaces (x : ℝ) : ℝ := 
  ⌊x * 1000 + 0.5⌋ / 1000

theorem sylvia_height_in_meters :
  roundToThreeDecimalPlaces (inchesToMeters 74) = 1.880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sylvia_height_in_meters_l481_48138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_washes_is_23_l481_48168

/-- Represents the discount tiers for car washes -/
inductive DiscountTier
  | Tier1  -- 10-14 car washes
  | Tier2  -- 15-19 car washes
  | Tier3  -- 20+ car washes

/-- Returns the discount rate for a given tier -/
def discountRate (tier : DiscountTier) : Rat :=
  match tier with
  | DiscountTier.Tier1 => 9/10
  | DiscountTier.Tier2 => 8/10
  | DiscountTier.Tier3 => 7/10

/-- Returns the minimum number of car washes for a given tier -/
def minWashes (tier : DiscountTier) : Nat :=
  match tier with
  | DiscountTier.Tier1 => 10
  | DiscountTier.Tier2 => 15
  | DiscountTier.Tier3 => 20

def normalPrice : Rat := 15
def budget : Rat := 250

/-- Calculates the number of car washes that can be purchased for a given tier -/
def washesForTier (tier : DiscountTier) : Nat :=
  (budget / (normalPrice * discountRate tier)).floor.toNat

/-- Finds the maximum number of car washes that can be purchased -/
def maxWashes : Nat :=
  max (washesForTier DiscountTier.Tier1)
    (max (washesForTier DiscountTier.Tier2)
      (washesForTier DiscountTier.Tier3))

theorem max_washes_is_23 :
  maxWashes = 23 ∧
  ∀ (n : Nat), n > 23 → n * (normalPrice * discountRate DiscountTier.Tier3) > budget :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_washes_is_23_l481_48168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_size_l481_48148

-- Define the number of subsets function
def n (S : Finset α) : ℕ := 2^(S.card)

-- Define the theorem
theorem min_intersection_size
  (A B C : Finset ℕ)
  (h1 : n A + n B + n C = n (A ∪ B ∪ C))
  (h2 : A.card = 100)
  (h3 : B.card = 100) :
  ∃ (min_size : ℕ), min_size = 97 ∧
    ∀ (intersection_size : ℕ),
      intersection_size = (A ∩ B ∩ C).card →
      intersection_size ≥ min_size :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_size_l481_48148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_square_area_difference_l481_48115

/-- A regular hexagon with side length 2 units --/
def regular_hexagon : ℝ := 2

/-- The area of a regular hexagon with side length s --/
noncomputable def hexagon_area (s : ℝ) : ℝ := 3 * Real.sqrt 3 * s^2 / 2

/-- The area of a square with side length s --/
def square_area (s : ℝ) : ℝ := s^2

/-- The number of squares attached to the hexagon --/
def num_squares : ℕ := 18

/-- The side length of the larger hexagon S --/
def large_hexagon_side : ℝ := 2 * regular_hexagon

/-- The theorem to prove --/
theorem hexagon_square_area_difference : 
  hexagon_area large_hexagon_side - (hexagon_area regular_hexagon + num_squares * square_area regular_hexagon) = 42 * Real.sqrt 3 - 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_square_area_difference_l481_48115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angled_l481_48181

theorem triangle_right_angled (h₁ h₂ h₃ : ℝ) 
  (height_1 : h₁ = 12)
  (height_2 : h₂ = 15)
  (height_3 : h₃ = 20) :
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angled_l481_48181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_odd_f_specific_value_l481_48197

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^x + 1) / (2^x - 1)

-- Theorem for the domain of f
theorem f_domain (x : ℝ) : f x ≠ 0 ↔ x ≠ 0 := by
  sorry

-- Theorem for the parity of f
theorem f_odd : ∀ x, f (-x) = -f x := by
  sorry

-- Theorem for the specific value of x when f(x) = -5/3
theorem f_specific_value : f (-2) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_odd_f_specific_value_l481_48197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_driving_time_comparison_l481_48141

/-- Proves that the time taken to drive 100 miles in two parts (32 miles at 2x mph and 68 miles at x/2 mph) is 52% longer than driving 100 miles at a constant speed x mph. -/
theorem driving_time_comparison (x : ℝ) (h : x > 0) :
  (32 / (2 * x) + 68 / (x / 2) - 100 / x) / (100 / x) = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_driving_time_comparison_l481_48141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculation_system_of_equations_solution_l481_48186

-- Problem 1
theorem sqrt_calculation : Real.sqrt 9 + ((-8) ^ (1/3 : ℝ)) + abs (1 - Real.sqrt 3) = Real.sqrt 3 := by sorry

-- Problem 2
theorem system_of_equations_solution :
  ∃ (x y : ℝ), x + y = 3 ∧ 3 * x - 2 * y = 4 ∧ x = 2 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculation_system_of_equations_solution_l481_48186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_two_l481_48177

noncomputable section

-- Define the circle C
def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the line l
def line_l (ρ θ : ℝ) : Prop := 2 * ρ * Real.sin (θ + Real.pi/3) = 3 * Real.sqrt 3

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop := θ = Real.pi/3

-- Define points O, P, and Q
def point_O : ℝ × ℝ := (0, 0)
def point_P (ρ : ℝ) : ℝ × ℝ := (ρ, Real.pi/3)
def point_Q (ρ : ℝ) : ℝ × ℝ := (ρ, Real.pi/3)

-- Theorem statement
theorem length_PQ_is_two :
  ∀ (ρ_P ρ_Q : ℝ),
    circle_C (Real.pi/3) = ρ_P →
    line_l ρ_Q (Real.pi/3) →
    ray_OM (Real.pi/3) →
    point_P ρ_P = (ρ_P, Real.pi/3) →
    point_Q ρ_Q = (ρ_Q, Real.pi/3) →
    ρ_Q - ρ_P = 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_two_l481_48177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_sum_l481_48121

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of circles C, D, and E -/
structure CircleConfiguration where
  C : Circle
  D : Circle
  E : Circle
  p : ℕ
  q : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (config : CircleConfiguration) : Prop :=
  config.C.radius = 6 ∧
  config.D.radius = 2 * config.E.radius ∧
  config.D.radius = Real.sqrt (config.p : ℝ) - (config.q : ℝ) ∧
  config.p > 0 ∧
  config.q > 0

/-- The theorem to be proved -/
theorem circle_configuration_sum (config : CircleConfiguration) 
  (h : satisfiesConditions config) : config.p + config.q = 186 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_sum_l481_48121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l481_48156

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x + a * Real.log (x - 2)

-- Theorem statement
theorem function_properties (a : ℝ) (h_a : a < 1) :
  -- 1. f(x) is decreasing when a ≤ 0
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, x₁ > 2 ∧ x₂ > 2 ∧ x₁ < x₂ → f a x₁ > f a x₂) ∧
  -- 2. f(x) is increasing then decreasing when 0 < a < 1
  (0 < a → ∃ c : ℝ, c > 2 ∧
    (∀ x₁ x₂ : ℝ, 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < c → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂ : ℝ, c < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂)) ∧
  -- 3. When a < 0, if f(x₁) - f(x₂) < -4 for any x₁, x₂ ∈ (2, +∞), then a ≤ -3
  (a < 0 → (∀ x₁ x₂ : ℝ, x₁ > 2 ∧ x₂ > 2 → f a x₁ - f a x₂ < -4) → a ≤ -3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l481_48156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_left_faces_dots_l481_48171

/-- Represents a single die face with a number of dots -/
structure DieFace where
  dots : Nat
  h : dots ≥ 1 ∧ dots ≤ 6

/-- Represents a die with six faces -/
structure Die where
  faces : Fin 6 → DieFace
  h : ∀ i j, i ≠ j → (faces i).dots ≠ (faces j).dots

/-- Represents the structure formed by four glued dice -/
structure DiceStructure where
  dice : Fin 4 → Die
  h : ∀ i j, i ≠ j → dice i = dice j  -- All dice are identical

/-- The four left faces of the structure -/
def LeftFaces : Fin 4 → DieFace
| 0 => { dots := 3, h := by simp [Nat.le_refl, Nat.le_succ] }  -- Face A
| 1 => { dots := 5, h := by simp [Nat.le_refl] }  -- Face B
| 2 => { dots := 6, h := by simp [Nat.le_refl] }  -- Face C
| 3 => { dots := 5, h := by simp [Nat.le_refl] }  -- Face D

/-- Theorem stating the number of dots on the left faces -/
theorem left_faces_dots :
  (LeftFaces 0).dots = 3 ∧
  (LeftFaces 1).dots = 5 ∧
  (LeftFaces 2).dots = 6 ∧
  (LeftFaces 3).dots = 5 := by
  simp [LeftFaces]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_left_faces_dots_l481_48171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l481_48194

-- Define set A
def A : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x + 2)}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (-2 : ℝ) 1 ∪ {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l481_48194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_equivalence_l481_48150

theorem log_equation_equivalence (a b x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (ha1 : a ≠ 1) (hb1 : b ≠ 1) (hx1 : x ≠ 1) :
  4 * (Real.log x / Real.log a)^2 + 3 * (Real.log x / Real.log b)^2 = 
  8 * (Real.log x / Real.log a) * (Real.log x / Real.log b) ↔ 
  a = b^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_equivalence_l481_48150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_90_degrees_counterclockwise_l481_48182

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the rotated function
noncomputable def g (x : ℝ) : ℝ := 10^(-x)

-- Theorem statement
theorem rotation_90_degrees_counterclockwise (x y : ℝ) :
  y = f x ↔ x = g (-y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_90_degrees_counterclockwise_l481_48182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_quadratic_equation_l481_48196

theorem correct_quadratic_equation 
  (h1 : ∃ k : ℝ, 5 * 3 = k ∧ k ≠ 24)
  (h2 : ∃ m : ℝ, 5 + 3 = -m ∧ m ≠ -8)
  (h3 : ∃ n : ℝ, -6 + -4 = -n ∧ n ≠ -8)
  (h4 : (-6 : ℝ) * (-4 : ℝ) = 24) :
  (λ x : ℝ ↦ x^2 - 8*x + 24) = (λ x : ℝ ↦ (x - 5) * (x - 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_quadratic_equation_l481_48196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_circles_pass_through_D_l481_48143

/-- A quadratic function that intersects the coordinate axes at three distinct points -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  distinct_intersections : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ q ≠ 0 ∧ x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0

/-- The point D(0, 1) -/
def point_D : ℝ × ℝ := (0, 1)

/-- Predicate to check if a set of points in ℝ² forms a circle circumscribed around the triangle
    formed by the intersection points of the quadratic function with the coordinate axes -/
def is_circumscribed_circle (f : QuadraticFunction) (circle : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- The theorem stating that all circumscribed circles pass through point D -/
theorem all_circles_pass_through_D (f : QuadraticFunction) :
  ∀ (circle : Set (ℝ × ℝ)), is_circumscribed_circle f circle → point_D ∈ circle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_circles_pass_through_D_l481_48143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l481_48187

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 75 →
  person_speed = 3 →
  passing_time = 24.998 →
  let relative_speed := (train_speed - person_speed) * 1000 / 3600
  ∃ (ε : ℝ), abs (relative_speed * passing_time - 499.96) < ε ∧ ε > 0 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l481_48187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_eq_7_5_sqrt_580_l481_48169

noncomputable section

/-- Triangle vertices -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (26, 0)
def C : ℝ × ℝ := (10, 18)

/-- Midpoints of triangle sides -/
def M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def N : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def P : ℝ × ℝ := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)

/-- Centroid of the triangle -/
def G : ℝ × ℝ := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Height of the pyramid (length of the longest side) -/
def h : ℝ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)

/-- Area of the midpoint triangle -/
def area_MNP : ℝ := (1/2) * abs (M.1 * N.2 + N.1 * P.2 + P.1 * M.2 - M.2 * N.1 - N.2 * P.1 - P.2 * M.1)

/-- Volume of the pyramid -/
def pyramid_volume : ℝ := (1/3) * area_MNP * h

/-- Theorem: The volume of the pyramid is 7.5 √580 -/
theorem pyramid_volume_eq_7_5_sqrt_580 : pyramid_volume = 7.5 * Real.sqrt 580 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_eq_7_5_sqrt_580_l481_48169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l481_48100

theorem relationship_abc : 
  let a := Real.rpow 0.6 4.2
  let b := Real.rpow 7 0.6
  let c := Real.log 7 / Real.log 0.6
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l481_48100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_160_degree_angles_l481_48153

-- Define interior angle of a regular n-gon
noncomputable def interior_angle (n : ℕ) : ℝ :=
  180 * (n - 2) / n

theorem regular_polygon_with_160_degree_angles (n : ℕ) : 
  (n ≥ 3) →  -- Ensure it's a valid polygon
  (∀ i : ℕ, i < n → interior_angle n = 160) →
  n = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_160_degree_angles_l481_48153
