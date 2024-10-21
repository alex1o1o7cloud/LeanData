import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_f_l111_11150

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log x

-- Define the theorem
theorem min_interval_for_f (t s : ℝ) (h : f t = g s) :
  ∃ (a : ℝ), a ∈ Set.Ioo (1/2 : ℝ) (Real.log 2) ∧
  (∀ (t' s' : ℝ), f t' = g s' → s' - t' ≥ s - t) →
  f t = a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_f_l111_11150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_exists_point_at_shortest_distance_l111_11108

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 18*x + y^2 - 8*y + 153 = 0

/-- The shortest distance from the origin to the circle -/
noncomputable def shortest_distance : ℝ := Real.sqrt 97 - 2 * Real.sqrt 14

/-- Theorem stating that the shortest_distance is indeed the shortest distance to any point on the circle -/
theorem shortest_distance_to_circle :
  ∀ (x y : ℝ), circle_equation x y →
  shortest_distance ≤ Real.sqrt (x^2 + y^2) := by
  sorry

/-- Theorem stating that there exists a point on the circle at exactly the shortest_distance from the origin -/
theorem exists_point_at_shortest_distance :
  ∃ (x y : ℝ), circle_equation x y ∧ 
  shortest_distance = Real.sqrt (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_exists_point_at_shortest_distance_l111_11108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_directions_norm_sum_eq_norm_diff_l111_11159

/-- Two vectors are in opposite directions if their sum has a smaller magnitude than their difference -/
def opposite_directions {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] (a b : E) : Prop :=
  ‖a + b‖ < ‖a - b‖

theorem opposite_directions_norm_sum_eq_norm_diff 
  {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] (a b : E) 
  (h : opposite_directions a b) : 
  ‖a‖ + ‖b‖ = ‖a - b‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_directions_norm_sum_eq_norm_diff_l111_11159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_x_exists_solution_l111_11164

theorem greatest_integer_x (x : ℤ) : (3 * x.natAbs ^ 2 + 5 < 32) → x ≤ 2 :=
by sorry

theorem exists_solution : ∃ x : ℤ, (3 * x.natAbs ^ 2 + 5 < 32) ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_x_exists_solution_l111_11164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l111_11179

/-- Calculate the overall gain percentage for three articles --/
theorem overall_gain_percentage
  (cost_A : ℝ) (sell_A : ℝ)
  (cost_B : ℝ) (sell_B : ℝ)
  (cost_C : ℝ) (sell_C : ℝ)
  (h1 : cost_A = 100)
  (h2 : sell_A = 125)
  (h3 : cost_B = 200)
  (h4 : sell_B = 250)
  (h5 : cost_C = 150)
  (h6 : sell_C = 180) :
  let total_cost := cost_A + cost_B + cost_C
  let total_sell := sell_A + sell_B + sell_C
  let gain := total_sell - total_cost
  let gain_percentage := (gain / total_cost) * 100
  ∃ (ε : ℝ), abs (gain_percentage - 23.33) < ε ∧ ε > 0 := by
  sorry

#check overall_gain_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l111_11179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cl_in_BaCl2_l111_11100

/-- The molar mass of barium in g/mol -/
noncomputable def molar_mass_Ba : ℝ := 137.33

/-- The molar mass of chlorine in g/mol -/
noncomputable def molar_mass_Cl : ℝ := 35.45

/-- The number of barium atoms in BaCl2 -/
def num_Ba_atoms : ℕ := 1

/-- The number of chlorine atoms in BaCl2 -/
def num_Cl_atoms : ℕ := 2

/-- The molar mass of BaCl2 in g/mol -/
noncomputable def molar_mass_BaCl2 : ℝ := molar_mass_Ba + num_Cl_atoms * molar_mass_Cl

/-- The mass percentage of Cl in BaCl2 -/
noncomputable def mass_percentage_Cl : ℝ := (num_Cl_atoms * molar_mass_Cl / molar_mass_BaCl2) * 100

/-- Theorem stating that the mass percentage of Cl in BaCl2 is approximately 34.04% -/
theorem mass_percentage_Cl_in_BaCl2 :
  |mass_percentage_Cl - 34.04| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cl_in_BaCl2_l111_11100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_heart_calculation_l111_11170

-- Define the spade operation
noncomputable def spade (a b : ℝ) : ℝ := a - 1 / (b^2)

-- Define the heart operation
noncomputable def heart (a b : ℝ) : ℝ := a + b^2

-- Theorem statement
theorem spade_heart_calculation :
  spade 3 (heart 3 2) = 146 / 49 :=
by
  -- Expand the definitions of spade and heart
  unfold spade heart
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_heart_calculation_l111_11170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_is_one_fourth_l111_11102

/-- The set of positive integers less than or equal to 36 -/
def S : Finset Nat := Finset.filter (fun n => 1 ≤ n ∧ n ≤ 36) (Finset.range 37)

/-- The set of factors of 36 -/
def factors_of_36 : Finset Nat := Finset.filter (fun n => n ∣ 36) S

/-- The probability of a number in S being a factor of 36 -/
def prob : ℚ := (factors_of_36.card : ℚ) / (S.card : ℚ)

theorem prob_is_one_fourth : prob = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_is_one_fourth_l111_11102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_circle_minus_square_l111_11162

/-- The area of the region inside a circle but outside an inscribed square -/
theorem area_circle_minus_square (r : ℝ) (h : r = 5) :
  π * r^2 - (r * Real.sqrt 2)^2 = 25 * π - 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_circle_minus_square_l111_11162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_satisfies_projections_l111_11107

noncomputable def v1 : ℝ × ℝ := (3, 2)
noncomputable def v2 : ℝ × ℝ := (3, 4)
noncomputable def w : ℝ × ℝ := (-48433/975, 2058/325)

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1 * v.1 + v.2 * v.2
  ((dot / norm_sq) * v.1, (dot / norm_sq) * v.2)

theorem w_satisfies_projections :
  proj w v1 = (47/13, 31/13) ∧ proj w v2 = (85/25, 113/25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_satisfies_projections_l111_11107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_tangent_sum_bound_l111_11147

theorem acute_angles_tangent_sum_bound 
  (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_sum : Real.sin α + Real.sin β + Real.sin γ = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_tangent_sum_bound_l111_11147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l111_11182

noncomputable section

-- Define the circle C
def circle_C (x y r : ℝ) : Prop := (x + 1)^2 + y^2 = r^2

-- Define the parabola D
def parabola_D (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the intersection points A and B
def intersectionPoints (A B : ℝ × ℝ) (r : ℝ) : Prop :=
  circle_C A.1 A.2 r ∧ parabola_D A.1 A.2 ∧
  circle_C B.1 B.2 r ∧ parabola_D B.1 B.2

-- Define the distance between A and B
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem circle_area (r : ℝ) (A B : ℝ × ℝ) :
  circle_C A.1 A.2 r → circle_C B.1 B.2 r →
  parabola_D A.1 A.2 → parabola_D B.1 B.2 →
  distance A B = 8 →
  π * r^2 = 25 * π :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l111_11182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_term_equals_initial_l111_11130

/-- A sequence defined recursively -/
noncomputable def v (b : ℝ) : ℕ → ℝ
  | 0 => b  -- Adding the base case for 0
  | 1 => b
  | n + 2 => -1 / (v b (n + 1) + 2)

/-- The theorem stating that the 13th term of the sequence equals the initial value -/
theorem thirteenth_term_equals_initial (b : ℝ) (h : b > 0) : v b 13 = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_term_equals_initial_l111_11130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l111_11178

theorem trigonometric_identities (θ : Real) 
  (h1 : Real.cos θ = 3/5) 
  (h2 : θ ∈ Set.Ioo 0 (Real.pi/2)) : 
  (Real.sin θ)^2 = 24/25 ∧ 
  Real.cos (θ + Real.pi/3) = (3 - 4 * Real.sqrt 3) / 10 ∧ 
  Real.tan (θ + Real.pi/4) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l111_11178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_t_l111_11114

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * ((x - 1) / 2) - 2

-- State the theorem
theorem find_t : ∃ t : ℝ, f t = 4 ∧ t = 5 := by
  -- Introduce t
  let t : ℝ := 5
  
  -- Show that f t = 4
  have h1 : f t = 4 := by
    -- Expand the definition of f
    unfold f
    -- Simplify the expression
    simp [t]
    -- Prove the equality
    norm_num
  
  -- Show that t = 5 (trivial since we defined t as 5)
  have h2 : t = 5 := rfl
  
  -- Combine the two facts to prove the existence
  exact ⟨t, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_t_l111_11114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_parts_l111_11160

/-- Represents a part of the 8x8 square -/
structure SquarePart where
  area : ℕ
  perimeter : ℕ

/-- The original 8x8 square -/
def originalSquare : ℕ := 64

/-- A function that checks if a list of parts is valid according to the problem conditions -/
def isValidPartition (parts : List SquarePart) : Prop :=
  (parts.map SquarePart.area).sum = originalSquare ∧
  ∃ p, ∀ part, part ∈ parts → part.perimeter = p ∧
  ∃ part1 part2, part1 ∈ parts ∧ part2 ∈ parts ∧ part1.area ≠ part2.area

/-- The theorem stating the maximum number of parts -/
theorem max_parts : 
  (∃ parts : List SquarePart, isValidPartition parts ∧ parts.length = 21) ∧
  (∀ parts : List SquarePart, isValidPartition parts → parts.length ≤ 21) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_parts_l111_11160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_theorem_l111_11109

/-- Represents an ellipse with foci on the y-axis -/
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- focal distance from center
  h_positive : 0 < b ∧ b < a
  h_focal : c^2 = a^2 - b^2

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.b^2 + y^2 / e.a^2 = 1

/-- Condition: The point on the short axis and the two foci form an equilateral triangle -/
def equilateral_triangle_condition (e : Ellipse) : Prop :=
  2 * e.c = e.b * Real.sqrt 3

/-- Condition: The shortest distance from a focus to the endpoint of the major axis is √3 -/
def focus_to_endpoint_condition (e : Ellipse) : Prop :=
  e.a - e.c = Real.sqrt 3

theorem ellipse_equation_theorem (e : Ellipse) 
  (h1 : equilateral_triangle_condition e)
  (h2 : focus_to_endpoint_condition e) :
  ∀ x y : ℝ, e.equation x y ↔ x^2 / 9 + y^2 / 12 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_theorem_l111_11109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_proof_l111_11184

/-- Represents the length of a candle stub after burning for a given time. -/
noncomputable def candleStubLength (initialLength : ℝ) (burnTime : ℝ) (elapsedTime : ℝ) : ℝ :=
  initialLength * (burnTime - elapsedTime) / burnTime

theorem candle_lighting_time_proof (initialLength : ℝ) (h : initialLength > 0) :
  let burnTime1 : ℝ := 3 * 60  -- 3 hours in minutes
  let burnTime2 : ℝ := 4 * 60  -- 4 hours in minutes
  let lightingTime : ℝ := 144  -- 2 hours and 24 minutes before 4 PM
  candleStubLength initialLength burnTime2 lightingTime = 
    2 * candleStubLength initialLength burnTime1 lightingTime := by
  sorry

#check candle_lighting_time_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_proof_l111_11184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curves_and_locus_l111_11190

-- Define the polar curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ := -2 * Real.cos θ
noncomputable def C₂ (θ : ℝ) : ℝ := 1 / Real.cos (θ + Real.pi/3)

-- Define the condition for point P
def P_condition (ρ ρ₀ : ℝ) : Prop := ρ * ρ₀ = 2

-- Theorem statement
theorem polar_curves_and_locus :
  -- Part 1: No common points between C₁ and C₂
  (∀ θ : ℝ, C₁ θ ≠ C₂ θ) ∧
  -- Part 2: Locus of P is a circle
  (∀ θ : ℝ, ∃ ρ : ℝ, P_condition ρ (C₂ θ) →
    (ρ * Real.cos θ - 1/2)^2 + (ρ * Real.sin θ + Real.sqrt 3/2)^2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curves_and_locus_l111_11190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_eq_2006_l111_11104

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Assume f and g have inverse functions
axiom f_has_inverse : Function.Bijective f
axiom g_has_inverse : Function.Bijective g

-- Define the symmetry condition
axiom symmetry : ∀ x y : ℝ, f (x - 1) = y ↔ g⁻¹ (y - 2) = x

-- Define the given condition
axiom g_5_eq_2004 : g 5 = 2004

-- State the theorem to be proved
theorem f_4_eq_2006 : f 4 = 2006 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_eq_2006_l111_11104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l111_11138

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Define F(x) = xf(x)
def F (x : ℝ) : ℝ := x * f x

-- State the theorem
theorem odd_function_inequality (h1 : ∀ x, f (-x) = -f x) 
  (h2 : ∀ x ≤ 0, x * f' x < f (-x))
  (h3 : ∀ x, F x = x * f x) :
  ∀ x, F 3 > F (2*x - 1) ↔ -1 < x ∧ x < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l111_11138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_projection_theorem_l111_11155

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_projection_theorem (seq : ArithmeticSequence) :
  seq.a 1009 = 0 →
  (2 : ℚ) * seq.a 1009 + 2 = 2 * 2 - 1 * 2 →
  sum_n seq 2017 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_projection_theorem_l111_11155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l111_11156

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The first complex number -/
noncomputable def z₁ : ℂ := 2 + i

/-- The second complex number -/
noncomputable def z₂ : ℂ := 1 - i

/-- The result of division -/
noncomputable def z : ℂ := z₁ / z₂

/-- Theorem: The point corresponding to z is in the first quadrant -/
theorem z_in_first_quadrant : 0 < z.re ∧ 0 < z.im := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l111_11156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l111_11169

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - Real.sqrt (2 - Real.sqrt (3 - x)))

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l111_11169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l111_11111

theorem power_equation_solution (x : ℝ) : (3 : ℝ)^6 = 27^x → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l111_11111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_polar_equation_intersection_distance_l111_11152

-- Define the parametric equations of line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, Real.sqrt 3 * t)

-- Define the polar equation of curve C
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos θ^2 + ρ^2 * Real.sin θ^2 - 2 * ρ * Real.sin θ - 3 = 0

-- Theorem 1: The polar equation of line l is θ = π/3
theorem line_l_polar_equation :
  ∀ ρ : ℝ, ∃ t : ℝ, line_l t = (ρ * Real.cos (π/3), ρ * Real.sin (π/3)) := by
  sorry

-- Theorem 2: The distance between intersection points is √15
theorem intersection_distance :
  ∃ ρ₁ ρ₂ : ℝ,
    curve_C ρ₁ (π/3) ∧
    curve_C ρ₂ (π/3) ∧
    (ρ₁ - ρ₂)^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_polar_equation_intersection_distance_l111_11152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_of_three_l111_11166

def sequence_a : ℕ → ℤ
  | 0 => 3  -- Adding case for 0 to cover all natural numbers
  | 1 => 3
  | 2 => 7
  | n + 3 => 3 * sequence_a (n + 2) - sequence_a (n + 1)

theorem prime_power_of_three (n : ℕ) :
  (∀ k : ℕ, k ≥ 2 → (sequence_a k)^2 + 5 = sequence_a (k - 1) * sequence_a (k + 1)) →
  Nat.Prime (Int.natAbs (sequence_a n + (-1)^n)) →
  ∃ m : ℕ, n = 3^m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_of_three_l111_11166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_erdos_number_equals_project_value_l111_11183

/-- Represents the distribution of Erdős numbers -/
noncomputable def erdos_distribution : List ℝ := [1, 10, 50, 125, 156, 97, 30, 4, 0.3]

/-- Calculates the average Erdős number given a distribution -/
noncomputable def average_erdos_number (dist : List ℝ) : ℝ :=
  let weighted_sum := (List.enumFrom 1 dist).map (fun (i, x) => i * x) |>.sum
  let total := dist.sum
  weighted_sum / total

/-- The average Erdős number according to the Erdős Number Project -/
def erdos_project_average : ℝ := 4.65

/-- Theorem stating that the calculated average Erdős number matches the Erdős Number Project's value -/
theorem average_erdos_number_equals_project_value :
  average_erdos_number erdos_distribution = erdos_project_average := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_erdos_number_equals_project_value_l111_11183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l111_11110

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + x - 1 else -2^(-x) + x + 1

theorem problem_solution :
  ∀ (m : ℝ),
  (∀ x, f (-x) = -f x) →
  (∀ x ≥ 0, f x = 2^x + x - m) →
  (m = 1 ∧
   (∀ x ≥ 0, f x = 2^x + x - 1) ∧
   (∀ x < 0, f x = -2^(-x) + x + 1) ∧
   (∀ k : ℝ, (∀ x ∈ Set.Icc (-3 : ℝ) (-2 : ℝ), f (k * 4^x) + f (1 - 2^(x+1)) > 0) ↔ k > -8)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l111_11110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l111_11140

open Real

-- Define the function f(x) = ln x - kx
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log x - k * x

theorem f_properties (k : ℝ) :
  -- Part 1: Tangent line when k = 2
  (k = 2 → ∃ m b : ℝ, m = -1 ∧ b = 1 ∧ ∀ x y : ℝ, y = f 2 x → y = m * (x - 1) + f 2 1) ∧
  -- Part 2: Condition for no zero points
  (∀ x : ℝ, x > 0 → f k x ≠ 0 ↔ k > (exp 1)⁻¹) ∧
  -- Part 3: Property when two distinct zero points exist
  (∃ x₁ x₂ : ℝ, 0 < x₂ ∧ x₂ < x₁ ∧ f k x₁ = 0 ∧ f k x₂ = 0 → log x₁ + log x₂ > 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l111_11140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_vertex_angle_theorem_l111_11188

/-- The vertex angle of the axial cross-section of n equal cones sharing a common vertex,
    each touching two others along a generatrix, and all touching a single plane. -/
noncomputable def coneVertexAngle (n : ℕ) : ℝ :=
  2 * Real.arcsin (Real.tan (Real.pi / n) / Real.sqrt (1 + 2 * Real.tan (Real.pi / n) ^ 2))

/-- Theorem stating the vertex angle of the axial cross-section of the cones. -/
theorem cone_vertex_angle_theorem (n : ℕ) (h : n ≥ 3) :
  coneVertexAngle n = 2 * Real.arcsin (Real.tan (Real.pi / n) / Real.sqrt (1 + 2 * Real.tan (Real.pi / n) ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_vertex_angle_theorem_l111_11188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l111_11196

noncomputable def f (a b x : ℝ) : ℝ := a * (2 * (Real.cos (x / 2))^2 + Real.sin x) + b

theorem f_properties :
  ∀ (a b : ℝ) (k : ℤ),
    -- Part 1
    (a = 1 →
      (∀ x : ℝ, 2 * k * Real.pi - 3 * Real.pi / 4 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 4 →
        ∀ y : ℝ, x < y → f a b x < f a b y) ∧
      (∀ x : ℝ, f a b (x + (Real.pi / 4 + k * Real.pi)) = f a b (Real.pi / 4 + k * Real.pi - x))) ∧
    -- Part 2
    (a > 0 →
      (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → 3 ≤ f a b x ∧ f a b x ≤ 4) →
        a = Real.sqrt 2 - 1 ∧ b = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l111_11196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l111_11112

/-- Given a projection that takes (2, -3) to (1, -3/2), prove that it takes (3, -1) to (18/13, -27/13) -/
theorem projection_theorem (P : ℝ × ℝ → ℝ × ℝ) (h : P (2, -3) = (1, -3/2)) :
  P (3, -1) = (18/13, -27/13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l111_11112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_product_lower_bound_sum_of_squares_lower_bound_l111_11168

noncomputable section

variable (a b c : ℝ)

-- Definitions
noncomputable def u (a b c : ℝ) : ℝ := a/b + b/c + c/a
noncomputable def v (a b c : ℝ) : ℝ := a/c + b/a + c/b

-- Assumptions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom c_pos : c > 0

-- Theorems to prove
theorem sum_lower_bound : u a b c + v a b c ≥ 6 := by sorry

theorem product_lower_bound : u a b c * v a b c ≥ 9 := by sorry

theorem sum_of_squares_lower_bound : 
  (∀ a b c, a > 0 → b > 0 → c > 0 → (u a b c)^2 + (v a b c)^2 ≥ 18) ∧ 
  (∃ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ (u a b c)^2 + (v a b c)^2 = 18) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_product_lower_bound_sum_of_squares_lower_bound_l111_11168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_point_l111_11195

theorem sin_double_angle_special_point :
  ∀ α : ℝ,
  (∃ r : ℝ, r > 0 ∧ r * Real.cos α = 1 ∧ r * Real.sin α = -2) →
  Real.sin (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_point_l111_11195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_proof_l111_11106

theorem fraction_equality_proof (a : ℕ) (ha : a > 0) : (a : ℚ) / (a + 36 : ℚ) = 9 / 10 → a = 324 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_proof_l111_11106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l111_11175

noncomputable def curve (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + 2015

theorem slope_angle_range :
  ∀ α : ℝ,
  (∃ x : ℝ, 0 ≤ α ∧ α < π ∧
   α = Real.arctan (-Real.sqrt 3 * Real.sin x)) →
  (0 ≤ α ∧ α ≤ π/3) ∨ (2*π/3 ≤ α ∧ α < π) :=
by
  intro α h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l111_11175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ones_digits_div_by_8_ones_digits_div_by_8_correct_l111_11191

/-- The set of possible ones digits for numbers divisible by 8 -/
def ones_digits_div_by_8 : Finset Nat :=
  {0, 2, 4, 6, 8}

/-- There are exactly 5 different possible ones digits in numbers divisible by 8 -/
theorem count_ones_digits_div_by_8 : Finset.card ones_digits_div_by_8 = 5 := by
  rfl

/-- Proof that the set contains all and only the correct digits -/
theorem ones_digits_div_by_8_correct (d : Nat) :
  d ∈ ones_digits_div_by_8 ↔ ∃ n : Nat, 8 ∣ n ∧ n % 10 = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ones_digits_div_by_8_ones_digits_div_by_8_correct_l111_11191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_two_equals_negative_eight_l111_11165

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (3 : ℝ)^x - 1 else -(3 : ℝ)^(-x) + 1

-- State the theorem
theorem f_negative_two_equals_negative_eight :
  f (-2) = -8 ∧ 
  (∀ x, f (-x) = -f x) ∧  -- odd function property
  (∀ x, x ≥ 0 → f x = (3 : ℝ)^x - 1) ∧  -- definition for x ≥ 0
  (∀ x, x < 0 → f x = -(3 : ℝ)^(-x) + 1) :=  -- definition for x < 0
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_two_equals_negative_eight_l111_11165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_sqrt_four_l111_11117

theorem sqrt_of_sqrt_four : ∃ (x : ℝ), x^2 = Real.sqrt 4 ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_sqrt_four_l111_11117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_if_all_lines_parallel_l111_11141

/-- A plane in 3D space -/
structure Plane where

/-- A line in 3D space -/
structure Line where

/-- Relation indicating that a line is inside a plane -/
def Line.insidePlane (l : Line) (p : Plane) : Prop :=
  sorry

/-- Relation indicating that a line is parallel to a plane -/
def Line.parallelToPlane (l : Line) (p : Plane) : Prop :=
  sorry

/-- Relation indicating that two planes are parallel -/
def Plane.parallel (p1 p2 : Plane) : Prop :=
  sorry

/-- Theorem: If any line inside plane α is parallel to plane β, then α and β are parallel -/
theorem planes_parallel_if_all_lines_parallel (α β : Plane) :
  (∀ l : Line, l.insidePlane α → l.parallelToPlane β) →
  α.parallel β :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_if_all_lines_parallel_l111_11141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_numbers_sum_120_l111_11172

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, n = m^2

def is_prime (n : Nat) : Prop := Nat.Prime n

def is_composite (n : Nat) : Prop := n > 1 ∧ ¬(Nat.Prime n)

def digits (n : Nat) : List Nat :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

theorem unique_numbers_sum_120 :
  ∃! (A B C : Nat),
    A ≥ 10 ∧ A < 100 ∧
    B ≥ 10 ∧ B < 100 ∧
    C ≥ 10 ∧ C < 100 ∧
    is_perfect_square A ∧
    (∀ d ∈ digits A, is_perfect_square d) ∧
    is_prime B ∧
    (∀ d ∈ digits B, is_prime d) ∧
    is_prime (List.sum (digits B)) ∧
    is_composite C ∧
    (∀ d ∈ digits C, is_composite d) ∧
    is_composite (Int.natAbs ((digits C).get! 0 - (digits C).get! 1)) ∧
    A < C ∧ C < B ∧
    A + B + C = 120 :=
by
  sorry

#eval digits 49
#eval digits 23
#eval digits 48

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_numbers_sum_120_l111_11172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_distance_theorem_l111_11161

/-- The part of the distance AB that the motorboat had traveled by the time it met the second boat -/
noncomputable def motorboat_distance (S : ℝ) (v : ℝ) (u : ℝ) : ℝ :=
  (5*v - u) * S / (6*v) + 20*S / 81

theorem motorboat_distance_theorem (S : ℝ) (v : ℝ) (u : ℝ) 
  (h1 : S > 0) 
  (h2 : v > 0)
  (h3 : (v + u) * (5*v - u) / (36*v^2) = 20 / 81) :
  motorboat_distance S v u = 56 / 81 ∨ motorboat_distance S v u = 65 / 81 := by
  sorry

#check motorboat_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_distance_theorem_l111_11161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diminished_value_proof_l111_11128

theorem diminished_value_proof (n : ℕ) (h : n = 1014) :
  ∃ (x : ℕ), (∀ d : ℕ, d ∈ ({12, 16, 18, 21, 28} : Finset ℕ) → (n - x) % d = 0) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diminished_value_proof_l111_11128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_five_thirds_unique_solution_g_equals_g_inv_l111_11192

/-- The function g -/
noncomputable def g (x : ℝ) : ℝ := 4 * x - 5

/-- The inverse function of g -/
noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

/-- Theorem stating that g(5/3) = g^(-1)(5/3) -/
theorem g_equals_g_inv_at_five_thirds :
  g (5/3) = g_inv (5/3) := by sorry

/-- Theorem stating that 5/3 is the unique solution to g(x) = g^(-1)(x) -/
theorem unique_solution_g_equals_g_inv :
  ∀ x : ℝ, g x = g_inv x ↔ x = 5/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_five_thirds_unique_solution_g_equals_g_inv_l111_11192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_l111_11125

/-- The cost per kg of mangos -/
def M : ℝ := sorry

/-- The cost per kg of rice -/
def R : ℝ := sorry

/-- The cost per kg of flour -/
def F : ℝ := sorry

/-- The cost per kg of grapes -/
def G : ℝ := sorry

/-- The cost of 10 kg of mangos is equal to the cost of 24 kg of rice -/
axiom mango_rice_relation : 10 * M = 24 * R

/-- The cost of 6 kg of flour equals the cost of 2 kg of rice -/
axiom flour_rice_relation : 6 * F = 2 * R

/-- The cost of each kg of flour is $23 -/
axiom flour_cost : F = 23

/-- The cost of 5 kg of grapes is equal to the cost of 15 kg of rice -/
axiom grape_rice_relation : 5 * G = 15 * R

/-- The total cost of 4 kg of mangos, 3 kg of rice, 5 kg of flour, and 2 kg of grapes is $1398.4 -/
theorem total_cost : 4 * M + 3 * R + 5 * F + 2 * G = 1398.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_l111_11125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_data_l111_11137

noncomputable def sample_data : List ℝ := [3, 4, 4, 5, 5, 6, 6, 7]

noncomputable def sample_mean (data : List ℝ) : ℝ :=
  (data.sum) / (data.length : ℝ)

noncomputable def sample_variance (data : List ℝ) : ℝ :=
  let mean := sample_mean data
  let squared_diff_sum := (data.map (λ x => (x - mean) ^ 2)).sum
  squared_diff_sum / (data.length : ℝ)

theorem variance_of_sample_data :
  sample_mean sample_data = 5 →
  sample_variance sample_data = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_data_l111_11137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_l111_11177

/-- Given an ellipse and a circle, prove the minimum area of a specific triangle --/
theorem min_area_triangle (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : 
  (x₀^2 / 8 + y₀^2 / 4 = 1) →  -- P(x₀, y₀) is on the ellipse
  (x₀ > 0 ∧ y₀ > 0) →  -- P is in the first quadrant
  (x₁^2 + y₁^2 = 4) →  -- A(x₁, y₁) is on the circle
  (x₂^2 + y₂^2 = 4) →  -- B(x₂, y₂) is on the circle
  (x₁*x₀ + y₁*y₀ = 4) →  -- PA is tangent to the circle
  (x₂*x₀ + y₂*y₀ = 4) →  -- PB is tangent to the circle
  (∀ S : ℝ, S = (4/x₀) * (4/y₀) / 2 → S ≥ Real.sqrt 2) := 
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_l111_11177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_sum_and_exponent_difference_l111_11116

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the binary logarithm (base 2)
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem logarithm_sum_and_exponent_difference :
  (lg 2 + lg 5 = 1) ∧ ((2 : ℝ) ^ (log2 3) - (8 : ℝ) ^ (1/3 : ℝ) = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_sum_and_exponent_difference_l111_11116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_coordinates_l111_11127

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 2, 0)

def O : ℝ × ℝ := (0, 0)

def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem hyperbola_point_coordinates (x y : ℝ) :
  hyperbola x y →
  x ≥ 1 →
  (distance (x, y) F₁ + distance (x, y) F₂) / distance (x, y) O = Real.sqrt 6 →
  (x, y) = (Real.sqrt 6 / 2, Real.sqrt 2 / 2) ∨ (x, y) = (Real.sqrt 6 / 2, -Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_coordinates_l111_11127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_necessary_not_sufficient_l111_11149

/-- Fixed points in a 2D plane -/
structure FixedPoint where
  x : ℝ
  y : ℝ

/-- A moving point in a 2D plane -/
structure MovingPoint where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : MovingPoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Sum of distances from a moving point to two fixed points -/
noncomputable def sumOfDistances (m : MovingPoint) (a b : MovingPoint) : ℝ :=
  distance m a + distance m b

/-- Predicate for a point being on an ellipse with foci A and B -/
def isOnEllipse (m : MovingPoint) (a b : MovingPoint) (c : ℝ) : Prop :=
  sumOfDistances m a b = c

/-- Predicate for a trajectory being an ellipse -/
def isEllipseTrajectory (trajectory : Set MovingPoint) (a b : MovingPoint) (c : ℝ) : Prop :=
  ∀ m ∈ trajectory, isOnEllipse m a b c

theorem ellipse_necessary_not_sufficient 
  (a b : MovingPoint) (c : ℝ) (h : c > 0) :
  (∃ trajectory : Set MovingPoint, isEllipseTrajectory trajectory a b c) →
  (∃ m : MovingPoint, sumOfDistances m a b = c) ∧
  (∃ trajectory : Set MovingPoint, (∀ m ∈ trajectory, sumOfDistances m a b = c) ∧ 
    ¬isEllipseTrajectory trajectory a b c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_necessary_not_sufficient_l111_11149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l111_11124

noncomputable def f (x : ℝ) := Real.log x / Real.log 3 + 2

noncomputable def y (x : ℝ) := (f x)^2 + f (x^2)

theorem max_value_of_y :
  ∃ (M : ℝ), M = 13 ∧ 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → y x ≤ M) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 9 ∧ y x = M) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l111_11124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exprC_not_square_difference_l111_11134

/-- The square difference formula -/
def squareDifference (a b : ℝ) : ℝ := a^2 - b^2

/-- Expressions to consider -/
def exprA (m n : ℝ) : ℝ := (m - n) * (-m - n)
def exprB (m n : ℝ) : ℝ := (-1 + m * n) * (1 + m * n)
def exprC (x y : ℝ) : ℝ := (-x + y) * (x - y)
def exprD (a b : ℝ) : ℝ := (2 * a - b) * (2 * a + b)

/-- Theorem stating that exprC cannot be directly represented as a square difference -/
theorem exprC_not_square_difference :
  ¬∃ (a b : ℝ → ℝ → ℝ), ∀ (x y : ℝ), exprC x y = squareDifference (a x y) (b x y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exprC_not_square_difference_l111_11134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_recurrence_relation_l111_11198

def x : ℕ → ℚ
  | 0 => 2  -- Add a case for 0 to cover all natural numbers
  | n + 1 => (3 * x n - 1) / (3 - x n)

theorem x_recurrence_relation (n : ℕ) :
  x (n + 1) = (3 * x n - 1) / (3 - x n) := by
  cases n
  · -- Case n = 0
    rfl
  · -- Case n = m + 1
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_recurrence_relation_l111_11198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maddie_bought_four_lipsticks_l111_11132

/-- Represents the purchase of beauty products by Maddie -/
structure BeautyPurchase where
  num_palettes : ℕ
  palette_price : ℚ
  num_lipsticks : ℕ
  lipstick_price : ℚ
  num_hair_color : ℕ
  hair_color_price : ℚ
  total_paid : ℚ

/-- The specific purchase made by Maddie -/
def maddie_purchase : BeautyPurchase :=
  { num_palettes := 3
  , palette_price := 15
  , num_lipsticks := 0  -- This is what we need to prove
  , lipstick_price := 5/2
  , num_hair_color := 3
  , hair_color_price := 4
  , total_paid := 67
  }

/-- Theorem stating that Maddie bought 4 lipsticks -/
theorem maddie_bought_four_lipsticks :
  ∃ (purchase : BeautyPurchase),
    purchase.num_palettes = maddie_purchase.num_palettes ∧
    purchase.palette_price = maddie_purchase.palette_price ∧
    purchase.lipstick_price = maddie_purchase.lipstick_price ∧
    purchase.num_hair_color = maddie_purchase.num_hair_color ∧
    purchase.hair_color_price = maddie_purchase.hair_color_price ∧
    purchase.total_paid = maddie_purchase.total_paid ∧
    purchase.num_lipsticks = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maddie_bought_four_lipsticks_l111_11132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_iff_equidistant_l111_11151

-- Define the types for points, planes, and lines
variable (Point Plane Line : Type)

-- Define the belongs_to relation
variable (belongs_to : Point → Plane → Prop)

-- Define the intersect_along relation
variable (intersect_along : Plane → Plane → Line → Prop)

-- Define the forms_equal_angles_with relation
variable (forms_equal_angles_with : Line → Plane → Plane → Prop)

-- Define the equidistant_from relation
variable (equidistant_from : Point → Point → Line → Prop)

-- Define the line_through function
variable (line_through : Point → Point → Line)

theorem equal_angles_iff_equidistant 
  (A1 A2 : Point) (Pi1 Pi2 : Plane) (l : Line) :
  belongs_to A1 Pi1 →
  belongs_to A2 Pi2 →
  intersect_along Pi1 Pi2 l →
  (forms_equal_angles_with (line_through A1 A2) Pi1 Pi2 ↔ 
   equidistant_from A1 A2 l) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_iff_equidistant_l111_11151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_value_l111_11146

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + Real.sqrt 3 * Real.cos x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x - Real.pi/3)

theorem symmetry_implies_a_value (a : ℝ) :
  (∀ x, g a (Real.pi/6 - x) = g a (Real.pi/6 + x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_value_l111_11146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l111_11123

-- Define the parameters of the frustum
def vertex_angle : ℝ := 60
def lower_radius : ℝ := 8
def upper_radius : ℝ := 4
def frustum_height : ℝ := 6

-- Define the lateral surface area of the frustum
noncomputable def lateral_surface_area (vertex_angle lower_radius upper_radius frustum_height : ℝ) : ℝ :=
  let semi_vertical_angle := vertex_angle / 2
  let full_height := lower_radius / Real.tan (semi_vertical_angle * Real.pi / 180)
  let small_height := full_height * (upper_radius / lower_radius)
  let full_slant := Real.sqrt (full_height^2 + lower_radius^2)
  let small_slant := Real.sqrt (small_height^2 + upper_radius^2)
  Real.pi * (lower_radius * full_slant - upper_radius * small_slant)

-- Theorem statement
theorem frustum_lateral_surface_area :
  lateral_surface_area vertex_angle lower_radius upper_radius frustum_height = 96 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l111_11123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_l111_11153

/-- A rectangle with area 2018 and sides parallel to coordinate axes -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)
  (area_eq : length * width = 2018)

/-- The four vertices of the rectangle -/
structure Vertex :=
  (x : ℤ)
  (y : ℤ)

/-- The four quadrants -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Function to determine the quadrant of a vertex -/
def get_quadrant (v : Vertex) : Quadrant :=
  if v.x > 0 && v.y > 0 then Quadrant.I
  else if v.x < 0 && v.y > 0 then Quadrant.II
  else if v.x < 0 && v.y < 0 then Quadrant.III
  else if v.x > 0 && v.y < 0 then Quadrant.IV
  else Quadrant.I  -- Default case, should not occur in our problem

theorem rectangle_diagonal (r : Rectangle) 
  (v1 v2 v3 v4 : Vertex)
  (h1 : get_quadrant v1 ≠ get_quadrant v2 ∧ 
        get_quadrant v1 ≠ get_quadrant v3 ∧ 
        get_quadrant v1 ≠ get_quadrant v4 ∧
        get_quadrant v2 ≠ get_quadrant v3 ∧
        get_quadrant v2 ≠ get_quadrant v4 ∧
        get_quadrant v3 ≠ get_quadrant v4) :
  (r.length ^ 2 + r.width ^ 2 : ℚ) = 1018085 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_l111_11153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bounds_l111_11186

theorem angle_sum_bounds (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_sin_sum : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  π / 2 < α + β + γ ∧ α + β + γ < 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bounds_l111_11186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l111_11174

noncomputable section

/-- The line is defined by this vector-valued function -/
def line (s : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 6 + 3 * s
  | 1 => 2 - 9 * s
  | 2 => 6 * s

/-- The point we're finding the closest point to -/
def target : Fin 3 → ℝ := ![1, 4, 5]

/-- The proposed closest point on the line -/
def closestPoint : Fin 3 → ℝ := ![249/42, 95/42, -1/7]

theorem closest_point_on_line :
  ∃ (s : ℝ), line s = closestPoint ∧
  ∀ (t : ℝ), ‖line t - target‖ ≥ ‖closestPoint - target‖ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l111_11174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l111_11197

-- Define the function f(x) = lg(1-x)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x) / Real.log 10

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, x ∈ Set.Iio 1 ↔ (∃ y : ℝ, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l111_11197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_bound_on_sine_l111_11101

theorem negation_of_universal_bound_on_sine :
  (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_bound_on_sine_l111_11101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l111_11180

-- Define the function f
noncomputable def f (x a b : ℝ) : ℝ := Real.log x + x - a + b / x

-- Part 1
theorem part1 (b : ℝ) :
  (∀ x > 0, f x 3 b ≥ 0) → b ≥ 2 := by
  sorry

-- Part 2
theorem part2 (a b : ℝ) :
  a > 0 → (∀ x > 0, f x a b ≥ 0) → a / (b + 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l111_11180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_polygon_areas_l111_11167

/-- Represents a rectangle on a grid paper --/
structure GridRectangle where
  width : ℕ
  height : ℕ
  total_squares : ℕ
  h_total_squares : width * height = total_squares

/-- Represents a closed polygonal line on a grid --/
structure ClosedPolygonalLine where
  rectangle : GridRectangle
  passes_all_points : Bool
  stays_inside : Bool

/-- Calculates the area of the polygon enclosed by the polygonal line --/
def enclosed_area (line : ClosedPolygonalLine) : ℕ :=
  (line.rectangle.width + 1) * (line.rectangle.height + 1) / 2 - 1

/-- The main theorem --/
theorem enclosed_polygon_areas 
  (rect : GridRectangle)
  (line : ClosedPolygonalLine)
  (h_rect : rect.total_squares = 72)
  (h_line : line.rectangle = rect ∧ line.passes_all_points ∧ line.stays_inside) :
  enclosed_area line ∈ ({72, 49, 44} : Set ℕ) := by
  sorry

#check enclosed_polygon_areas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_polygon_areas_l111_11167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l111_11126

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F and directrix intersection H
def focus : ℝ × ℝ := (1, 0)
def directrix_intersection : ℝ × ℝ := (-1, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_point_x_coordinate (p : ℝ × ℝ) :
  parabola p.1 p.2 →
  distance p directrix_intersection = Real.sqrt 2 * distance p focus →
  p.1 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l111_11126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_grammys_house_l111_11131

/-- Prove the distance to Grammy's house given the following conditions:
  * Cost of filling a car fuel tank is $45
  * One full tank covers 500 miles
  * Food expenses are 3/5 of fuel expenses
  * Total expenses are $288
-/
theorem distance_to_grammys_house : ℝ := by
  let fuel_cost : ℝ := 45
  let miles_per_tank : ℝ := 500
  let food_to_fuel_ratio : ℝ := 3 / 5
  let total_expenses : ℝ := 288

  let food_expenses : ℝ := food_to_fuel_ratio * fuel_cost
  let fuel_expenses : ℝ := total_expenses - food_expenses
  let num_tanks : ℝ := fuel_expenses / fuel_cost
  let distance : ℝ := num_tanks * miles_per_tank

  -- The actual computation
  have : distance = 2900 := by
    -- Here we would normally provide the proof steps
    sorry

  -- Return the final result
  exact 2900


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_grammys_house_l111_11131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l111_11158

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- The two lines are parallel -/
def are_parallel (A₁ B₁ A₂ B₂ : ℝ) : Prop :=
  A₁ / B₁ = A₂ / B₂

theorem parallel_lines_distance : 
  let line1 : ℝ → ℝ → ℝ := fun x y => 3*x + 4*y - 12
  let line2 : ℝ → ℝ → ℝ := fun x y => 6*x + 8*y + 11
  are_parallel 3 4 6 8 →
  distance_between_parallel_lines 6 8 (-24) 11 = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l111_11158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_m_value_l111_11193

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

theorem geometric_series_m_value :
  ∀ (m : ℝ),
  let s1 := geometric_series_sum 18 (1/3)
  let s2 := geometric_series_sum 18 ((6 + m) / 18)
  s2 = 3 * s1 →
  m = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_m_value_l111_11193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_triangles_2003_than_2000_l111_11121

/-- A triangle with integer sides -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a

/-- The set of triangles with integer sides and perimeter 2000 -/
def Triangles2000 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 2000}

/-- The set of triangles with integer sides and perimeter 2003 -/
def Triangles2003 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 2003}

/-- The theorem stating that there are more triangles with perimeter 2003 than 2000 -/
theorem more_triangles_2003_than_2000 : 
  ∃ f : Triangles2000 → Triangles2003, Function.Injective f ∧ ¬Function.Surjective f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_triangles_2003_than_2000_l111_11121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_tan_diff_l111_11181

noncomputable def z (θ : ℝ) : ℂ := (Real.cos θ - 3/5) + (Real.sin θ - 4/5) * Complex.I

theorem purely_imaginary_tan_diff (θ : ℝ) 
  (h : z θ = Complex.I * (Real.sin θ - 4/5)) : 
  Real.tan (θ - π/4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_tan_diff_l111_11181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l111_11139

theorem polynomial_identity (f : ℝ → ℝ) (h : ∀ x, f (x^2 + 2) = x^4 + 6*x^2 + 4) :
  ∀ x, f (x^2 - 2) = x^4 - 2*x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l111_11139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_to_cos_l111_11113

noncomputable section

-- Define the original function
def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/4)

-- Define the transformation
def transform (g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ g (2*x + Real.pi/4)

-- State the theorem
theorem transform_sin_to_cos :
  transform f = Real.cos := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_to_cos_l111_11113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l111_11122

theorem log_inequality (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → x^2 < Real.log (x + 1) / Real.log a) ↔ a ∈ Set.Ioc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l111_11122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_series_sum_l111_11194

noncomputable def b : ℕ → ℚ
  | 0 => 2  -- Adding this case to cover Nat.zero
  | 1 => 2
  | 2 => 1
  | k + 3 => (1/2) * b (k + 2) + (1/3) * b (k + 1)

noncomputable def series_sum : ℚ := ∑' n, b n

theorem b_series_sum : series_sum = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_series_sum_l111_11194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l111_11173

theorem sin_double_angle_special (α : ℝ) : 
  0 < α ∧ α < π / 2 →
  Real.cos (α + π / 6) = 3 / 5 →
  Real.sin (2 * α + π / 3) = 24 / 25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l111_11173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_square_l111_11145

/-- If a natural number n has exactly 4 divisors, then n^2 has either 7 or 9 divisors. -/
theorem divisors_of_square (n : ℕ) (h : (Finset.card (Nat.divisors n)) = 4) :
  (Finset.card (Nat.divisors (n^2))) = 7 ∨ (Finset.card (Nat.divisors (n^2))) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_square_l111_11145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_is_700_div_3_l111_11120

/-- Represents a trapezoid ABCD with points E and F on its sides -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  altitude : ℝ
  E_ratio : ℝ
  F_ratio : ℝ

/-- The area of quadrilateral EFCD in the given trapezoid -/
noncomputable def area_EFCD (t : Trapezoid) : ℝ :=
  let EF := t.E_ratio * t.AB + (1 - t.E_ratio) * t.CD
  let altitude_EFCD := (1 - t.E_ratio) * t.altitude
  altitude_EFCD * (EF + t.CD) / 2

/-- Theorem stating that the area of EFCD is 700/3 for the given trapezoid -/
theorem area_EFCD_is_700_div_3 :
  let t : Trapezoid := {
    AB := 10,
    CD := 26,
    altitude := 15,
    E_ratio := 1/3,
    F_ratio := 1/3
  }
  area_EFCD t = 700/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_is_700_div_3_l111_11120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_go_tournament_computer_assistance_l111_11154

/-- Represents a Go tournament with computer assistance. -/
structure GoTournament where
  num_players : Nat
  rankings : Fin num_players → Nat
  used_computer : Fin num_players → Bool

/-- The number of wins for a player in the tournament. -/
def wins (t : GoTournament) (player : Fin t.num_players) : Nat :=
  Finset.card (Finset.filter (λ opponent => 
    (t.used_computer player ∧ ¬t.used_computer opponent) ∨
    (t.used_computer player = t.used_computer opponent ∧ t.rankings player < t.rankings opponent)
  ) (Finset.univ))

/-- The statement of the Go tournament problem. -/
theorem go_tournament_computer_assistance 
  (t : GoTournament)
  (h_players : t.num_players = 55)
  (h_rankings : ∀ i j : Fin t.num_players, i ≠ j → t.rankings i ≠ t.rankings j)
  (h_winners : ∃ p q : Fin t.num_players, p ≠ q ∧ 
    ∀ i : Fin t.num_players, t.rankings i ≤ 2 → wins t p > wins t i ∧ wins t q > wins t i)
  : Finset.card (Finset.filter (λ i => ¬t.used_computer i) Finset.univ) ≤ 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_go_tournament_computer_assistance_l111_11154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_transformation_l111_11199

theorem cubic_equation_roots_transformation (q r : ℝ) (a b c : ℝ) :
  (a^3 + q*a + r = 0) ∧ (b^3 + q*b + r = 0) ∧ (c^3 + q*c + r = 0) →
  (r * ((b + c) / a^2)^3 - q * ((b + c) / a^2)^2 - 1 = 0) ∧
  (r * ((c + a) / b^2)^3 - q * ((c + a) / b^2)^2 - 1 = 0) ∧
  (r * ((a + b) / c^2)^3 - q * ((a + b) / c^2)^2 - 1 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_transformation_l111_11199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_concyclic_l111_11136

noncomputable section

-- Define the basic geometric objects
variable (C₁ C₂ : Set (EuclideanSpace ℝ (Fin 2))) -- Circles
variable (P Q A B E F H K : EuclideanSpace ℝ (Fin 2)) -- Points

-- Define the conditions
axiom intersect : P ∈ C₁ ∧ P ∈ C₂ ∧ Q ∈ C₁ ∧ Q ∈ C₂

-- We'll need to define these concepts
def IsTangentLine (C : Set (EuclideanSpace ℝ (Fin 2))) (P : EuclideanSpace ℝ (Fin 2)) (L : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

def Line.throughPoints (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

axiom common_tangent : IsTangentLine C₁ A (Line.throughPoints A B) ∧ 
                       IsTangentLine C₂ B (Line.throughPoints A B)
axiom tangent_P_C₁ : IsTangentLine C₁ P (Line.throughPoints P E)
axiom tangent_P_C₂ : IsTangentLine C₂ P (Line.throughPoints P F)
axiom E_not_P : E ≠ P
axiom F_not_P : F ≠ P

def Ray.fromPoints (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

axiom H_on_AF : H ∈ Ray.fromPoints A F
axiom K_on_BE : K ∈ Ray.fromPoints B E
axiom AH_eq_AP : dist A H = dist A P
axiom BK_eq_BP : dist B K = dist B P

def IsCircle (S : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Theorem to prove
theorem points_concyclic : ∃ (C : Set (EuclideanSpace ℝ (Fin 2))), IsCircle C ∧ A ∈ C ∧ H ∈ C ∧ Q ∈ C ∧ K ∈ C ∧ B ∈ C := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_concyclic_l111_11136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_percentage_rounded_l111_11103

/-- Represents the grocery items and their prices --/
structure Groceries where
  broccoli_price : ℚ
  broccoli_quantity : ℚ
  orange_price : ℚ
  orange_quantity : ℕ
  cabbage_price : ℚ
  bacon_price : ℚ
  bacon_quantity : ℚ
  chicken_price : ℚ
  chicken_quantity : ℚ

/-- Calculates the total cost of groceries --/
def total_cost (g : Groceries) : ℚ :=
  g.broccoli_price * g.broccoli_quantity +
  g.orange_price * g.orange_quantity +
  g.cabbage_price +
  g.bacon_price * g.bacon_quantity +
  g.chicken_price * g.chicken_quantity

/-- Calculates the total cost of meat (bacon and chicken) --/
def meat_cost (g : Groceries) : ℚ :=
  g.bacon_price * g.bacon_quantity +
  g.chicken_price * g.chicken_quantity

/-- Calculates the percentage of grocery budget spent on meat --/
def meat_percentage (g : Groceries) : ℚ :=
  (meat_cost g / total_cost g) * 100

/-- Rounds a rational number to the nearest integer --/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊(x + 1/2 : ℚ)⌋

/-- Theorem: The percentage of grocery budget spent on meat is 33% when rounded to the nearest percent --/
theorem meat_percentage_rounded (g : Groceries) : 
  g.broccoli_price = 4 ∧
  g.broccoli_quantity = 3 ∧
  g.orange_price = 3/4 ∧
  g.orange_quantity = 3 ∧
  g.cabbage_price = 15/4 ∧
  g.bacon_price = 3 ∧
  g.bacon_quantity = 1 ∧
  g.chicken_price = 3 ∧
  g.chicken_quantity = 2 →
  round_to_nearest (meat_percentage g) = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_percentage_rounded_l111_11103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l111_11163

/-- The area of a quadrilateral given its diagonal and two offsets -/
noncomputable def quadrilateralArea (d h₁ h₂ : ℝ) : ℝ := (1/2) * d * (h₁ + h₂)

/-- Theorem: The area of a quadrilateral with diagonal 28 cm and offsets 8 cm and 2 cm is 140 cm² -/
theorem quadrilateral_area_example : quadrilateralArea 28 8 2 = 140 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the expression
  simp [mul_add, mul_comm, mul_assoc]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l111_11163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l111_11115

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope-intercept form of a line
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Theorem: The slope of any line parallel to 3x - 6y = 12 is 1/2
theorem parallel_line_slope :
  ∃ (m : ℝ), ∀ (x y : ℝ), line_equation x y → 
    ∃ (b : ℝ), slope_intercept_form m b x y ∧ m = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l111_11115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l111_11119

noncomputable def f (x : ℝ) : ℝ := 2 / (x + 2) + 4 / (x + 8) - 1 / 2

theorem solution_set :
  {x : ℝ | f x ≥ 0} = Set.Ioc (-8 : ℝ) (-4 : ℝ) ∪ Set.Ioc (-2 : ℝ) (12 : ℝ) :=
by
  sorry

#check solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l111_11119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_cylinder_l111_11189

/-- The volume of a cylinder formed by rotating a square about its horizontal line of symmetry -/
noncomputable def cylinder_volume_from_square (side_length : ℝ) : ℝ :=
  Real.pi * (side_length / 2)^2 * side_length

/-- Theorem stating that the volume of the cylinder formed by rotating a square 
    with side length 20 cm about its horizontal line of symmetry is 2000π cubic centimeters -/
theorem volume_of_specific_cylinder : 
  cylinder_volume_from_square 20 = 2000 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_cylinder_l111_11189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_correct_l111_11144

/-- Regular quadrilateral pyramid with inscribed spheres -/
structure PyramidWithSpheres where
  a : ℝ  -- Side length of the base
  r : ℝ  -- Radius of the spheres
  h : ℝ  -- Height of the pyramid
  regular : a > 0  -- The pyramid is regular
  spheres_fit : a > 2*r  -- The spheres fit inside the pyramid

/-- The height of the pyramid in terms of a and r -/
noncomputable def pyramidHeight (p : PyramidWithSpheres) : ℝ :=
  (4 * p.a * p.r + Real.sqrt (p.a^2 + 32 * p.r^2)) / (p.a^2 - p.r^2)

/-- Theorem: The height of the pyramid is correctly calculated by pyramidHeight -/
theorem pyramid_height_correct (p : PyramidWithSpheres) :
  p.h = pyramidHeight p := by
  sorry

#check pyramid_height_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_correct_l111_11144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_games_played_l111_11171

theorem team_games_played (c d : ℕ) : 
  (3 * c = 4 * ((2 * d) / 3 - 3)) →  -- For every 4 games C wins, D wins 3 more
  (4 * c = 3 * d) →                  -- C wins 3/4 of its games, D wins 2/3
  (d = c + 12) →                     -- D played 12 more games than C
  c = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_games_played_l111_11171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_point_l111_11135

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define a point on the parabola
def point_on_parabola (p : ℝ) (M : ℝ × ℝ) : Prop :=
  parabola p M.1 M.2

-- Define the circumcircle of a triangle
def circumcircle (O F M : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | ∃ r : ℝ, (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2 ∧
                       (P.1 - F.1)^2 + (P.2 - F.2)^2 = r^2 ∧
                       (P.1 - M.1)^2 + (P.2 - M.2)^2 = r^2}

-- Define the directrix of the parabola
def directrix (p : ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | P.1 = -p/2}

-- Define tangency of a circle to a line
def circle_tangent_to_line (circle : Set (ℝ × ℝ)) (line : Set (ℝ × ℝ)) : Prop :=
  ∃ P : ℝ × ℝ, P ∈ circle ∧ P ∈ line ∧ 
    ∀ Q : ℝ × ℝ, Q ∈ circle → Q ∈ line → Q = P

-- Define the area of a circle
def circle_area (circle : Set (ℝ × ℝ)) (area : ℝ) : Prop :=
  ∃ r : ℝ, area = Real.pi * r^2 ∧
    ∀ P Q : ℝ × ℝ, P ∈ circle → Q ∈ circle →
      (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ≤ 4 * r^2

theorem parabola_focus_point (p : ℝ) :
  ∀ M : ℝ × ℝ,
  point_on_parabola p M →
  circle_tangent_to_line (circumcircle (0, 0) (focus p) M) (directrix p) →
  circle_area (circumcircle (0, 0) (focus p) M) (36 * Real.pi) →
  p = 8 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_point_l111_11135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l111_11148

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => -1/2
  | n + 1 => 1 / (1 - sequenceA n)

theorem sequence_properties :
  (sequenceA 1 = 2/3) ∧
  (sequenceA 2 = 3) ∧
  (sequenceA 3 = -1/2) ∧
  (∀ k, sequenceA (k + 3) = sequenceA k) ∧
  (sequenceA 1997 = 3) ∧
  (sequenceA 1999 = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l111_11148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_value_l111_11129

-- Define the vectors
noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
def b (m : ℝ) : ℝ × ℝ := (3, m)

-- Define the projection formula
noncomputable def projection (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2) / Real.sqrt (u.1^2 + u.2^2)

-- State the theorem
theorem projection_value (m : ℝ) :
  projection (b m) a = 3 → m = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_value_l111_11129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_7_pow_2016_l111_11105

def last_two_digits (n : ℕ) : ℕ := n % 100

def last_two_digits_pattern (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 01
  | 1 => 07
  | 2 => 49
  | 3 => 43
  | _ => 0  -- Add a catch-all case to handle all possible inputs

theorem last_two_digits_7_pow_2016 :
  last_two_digits (7^2016) = last_two_digits_pattern 2016 := by
  sorry

#eval last_two_digits_pattern 2016  -- This will evaluate to 01

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_7_pow_2016_l111_11105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_distance_l111_11133

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle3D) : Prop :=
  distance t.a t.b = distance t.b t.c ∧ distance t.b t.c = distance t.c t.a

/-- Calculates the dihedral angle between two triangles sharing an edge -/
noncomputable def dihedralAngle (t1 t2 : Triangle3D) : ℝ := sorry

theorem equidistant_point_distance 
  (x y z r s m : Point3D) 
  (triangle : Triangle3D) 
  (h1 : triangle.a = x ∧ triangle.b = y ∧ triangle.c = z)
  (h2 : isEquilateral triangle)
  (h3 : distance x y = 300)
  (h4 : distance r x = distance r y ∧ distance r y = distance r z)
  (h5 : distance s x = distance s y ∧ distance s y = distance s z)
  (h6 : dihedralAngle (Triangle3D.mk r x z) (Triangle3D.mk s x z) = Real.pi / 2)
  (h7 : distance m x = distance m y ∧ distance m y = distance m z ∧ 
        distance m z = distance m r ∧ distance m r = distance m s) :
  distance m x = 100 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_distance_l111_11133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_sum_l111_11185

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Compound interest calculation function -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating the sum on which compound interest is calculated -/
theorem compound_interest_sum : ∃ (p : ℝ),
  simple_interest 1750 8 3 = (1/2) * compound_interest p 10 2 ∧ p = 4000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_sum_l111_11185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_determination_l111_11118

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  h : r ≠ 1

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def S (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a * (1 - seq.r^n) / (1 - seq.r)

/-- Sum of S_1 to S_n -/
noncomputable def T (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a * n - seq.a * (seq.r - seq.r^(n+1)) / ((1 - seq.r)^2)

theorem unique_determination (seq : GeometricSequence) :
  ∃! n : ℕ, ∀ seq' : GeometricSequence, S seq 20 = S seq' 20 → T seq n = T seq' n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_determination_l111_11118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_square_divisors_l111_11187

/-- A triple of consecutive integers (a, a+1, a+2) -/
structure ConsecutiveTriple where
  first : ℤ
  second : ℤ
  third : ℤ
  consecutive : second = first + 1 ∧ third = first + 2

/-- Predicate to check if a number has a square divisor greater than 1 -/
def hasSquareDivisor (n : ℤ) : Prop :=
  ∃ (k : ℤ), k > 1 ∧ k * k ∣ n

/-- Predicate to check if a triple satisfies the condition -/
def isValidTriple (t : ConsecutiveTriple) : Prop :=
  hasSquareDivisor t.first ∧ hasSquareDivisor t.second ∧ hasSquareDivisor t.third

theorem consecutive_integers_square_divisors :
  ∀ (start : ℤ),
  ∃ (triples : Finset ConsecutiveTriple),
  (∀ t ∈ triples, start ≤ t.first ∧ t.third < start + 2000) ∧
  (∀ t ∈ triples, isValidTriple t) ∧
  triples.card ≥ 18 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_square_divisors_l111_11187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_fraction_l111_11176

def is_positive_integer (x : ℝ) : Prop := ∃ n : ℕ, x = n ∧ n > 0

theorem positive_integer_fraction (x : ℝ) (hx : x ≠ 0) :
  is_positive_integer (|(x - |x| + 2)| / x) ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_fraction_l111_11176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_set_bounds_l111_11142

/-- Represents a set of five n-digit numbers formed from digits 1 and 2 -/
def DigitSet (n : ℕ) := Fin 5 → Fin n → Fin 2

/-- The property that every two numbers in the set match in exactly m places -/
def MatchInMPlaces (s : DigitSet n) (m : ℕ) : Prop :=
  ∀ i j, i ≠ j → (Finset.filter (λ k : Fin n => s i k = s j k) Finset.univ).card = m

/-- The property that there is no position where all five numbers have the same digit -/
def NoAllSameDigit (s : DigitSet n) : Prop :=
  ∀ k : Fin n, ∃ i j : Fin 5, s i k ≠ s j k

theorem digit_set_bounds (n m : ℕ) (s : DigitSet n) 
  (h1 : MatchInMPlaces s m) (h2 : NoAllSameDigit s) : 
  2 / 5 ≤ (m : ℚ) / n ∧ (m : ℚ) / n ≤ 8 / 5 := by
  sorry

#check digit_set_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_set_bounds_l111_11142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l111_11143

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x b : ℝ) : ℝ := Real.log x + b

-- Define the derivative of f at x = 0
def f_derivative_at_0 : ℝ := 1

-- State the theorem
theorem tangent_line_intersection (b : ℝ) :
  (∃ (m : ℝ), m > 0 ∧
    (f_derivative_at_0 * m + f 0 = g m b) ∧
    (f_derivative_at_0 = (g m b - g 1 b) / (m - 1))) →
  b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l111_11143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_not_in_naturals_l111_11157

-- Define the set of natural numbers
def myNat : Set ℕ := {n : ℕ | n > 0}

-- Define the empty set
def emptySet : Set α := ∅

-- Define subset relation
def subset (A B : Set α) : Prop := ∀ x, x ∈ A → x ∈ B

-- Define superset relation
def superset (A B : Set α) : Prop := ∀ x, x ∈ B → x ∈ A

-- Theorem to prove
theorem zero_not_in_naturals : 0 ∉ myNat := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_not_in_naturals_l111_11157
