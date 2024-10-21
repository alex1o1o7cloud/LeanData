import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l21_2147

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (3, 0)

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- Main theorem: The distance from the right focus of the hyperbola to the line is √5 -/
theorem distance_right_focus_to_line :
  distance_point_to_line right_focus.1 right_focus.2 1 2 (-8) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l21_2147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_exist_l21_2150

theorem no_solutions_exist : ¬∃ (x y : ℝ), (64 : ℝ)^(x^2 + y + x) + (64 : ℝ)^(x + y^2 + y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_exist_l21_2150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l21_2136

def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x : ℤ | (x : ℝ)^2 - 2*(x : ℝ) > 0}

theorem intersection_M_N : M ∩ N = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l21_2136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l21_2175

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop :=
  (x + 2)^2 / 7^2 - (y - 5)^2 / 3^2 = 1

/-- The focus with larger x-coordinate -/
noncomputable def focus : ℝ × ℝ := (-2 + Real.sqrt 58, 5)

/-- Theorem stating that the given focus is correct for the hyperbola -/
theorem focus_of_hyperbola :
  ∀ x y : ℝ, hyperbola x y → 
  ∀ f : ℝ × ℝ, (f.1 > (-2 + Real.sqrt 58) ∨ f = focus) → 
  ¬(hyperbola f.1 f.2 ∧ 
    ∃ c : ℝ, c > 0 ∧ 
    ∀ p : ℝ × ℝ, hyperbola p.1 p.2 → 
    (p.1 - f.1)^2 + (p.2 - f.2)^2 = 
    (p.1 - (-2 + Real.sqrt 58))^2 + (p.2 - 5)^2 + c^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l21_2175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_supplementary_angle_l21_2158

theorem cosine_supplementary_angle (VTX VTU : Real) : 
  Real.cos VTX = 4/5 → VTX + VTU = Real.pi → Real.cos VTU = -4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_supplementary_angle_l21_2158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_even_increasing_l21_2162

-- Define the power function
noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x^(-m^2 + 2*m + 3)

-- State the theorem
theorem power_function_even_increasing (m : ℤ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- f is an even function
  (∀ x y : ℝ, 0 < x → x < y → f m x < f m y) →  -- f is monotonically increasing on (0, +∞)
  m = 1 →  -- Add this condition to specify m = 1
  (∀ x : ℝ, f m x = x^4) :=  -- f(x) = x^4
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_even_increasing_l21_2162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_sector_area_l21_2169

theorem hexagon_sector_area (s r θ : ℝ) : 
  s = 8 → r = 4 → θ = 120 →
  (6 * (Real.sqrt 3 / 4 * s^2) - 6 * (θ / 360 * Real.pi * r^2)) = 96 * Real.sqrt 3 - 32 * Real.pi :=
by
  intros hs hr hθ
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_sector_area_l21_2169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_division_l21_2172

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / 3 = q / 7 ∧ q / 7 = r / 12 →
  q - p = 3600 →
  r - q = 4500 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_division_l21_2172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l21_2195

noncomputable section

/-- The original function f(x) -/
def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 3)

/-- The shifted function g(x) -/
def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x + Real.pi / 6)

theorem min_omega_value :
  ∀ ω : ℝ, ω > 0 →
  g ω (Real.pi / 2) = 1 →
  (∀ ω' : ℝ, ω' > 0 → g ω' (Real.pi / 2) = 1 → ω ≤ ω') →
  ω = 3 / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l21_2195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l21_2134

/-- The distance from the center of the circle (x+1)^2 + y^2 = 2 to the line y = x + 3 is √2 -/
theorem distance_from_circle_center_to_line : 
  let circle : Set (ℝ × ℝ) := {p | (p.1 + 1)^2 + p.2^2 = 2}
  let line : Set (ℝ × ℝ) := {p | p.2 = p.1 + 3}
  let center : ℝ × ℝ := (-1, 0)
  ∃ (point : ℝ × ℝ), point ∈ line ∧ dist center point = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l21_2134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l21_2187

-- Define function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x else 0

-- Define function g
noncomputable def g (x : ℝ) : ℝ :=
  if x < 2 then 2 - x else 0

-- Theorem statement
theorem inequality_solution (x : ℝ) : f (g x) > g (f x) ↔ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l21_2187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l21_2145

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 5 / (3 * x^8 - 7)

-- State the theorem
theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  unfold g
  -- The actual proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l21_2145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l21_2192

noncomputable section

-- Define points A and B
def A : ℝ × ℝ := (-3, -1)
def B : ℝ × ℝ := (-4, 4)

-- Define the line l: x - y - 1 = 0
def l (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the reflection point C
def C : ℝ × ℝ := (-1, -2)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem reflection_problem :
  (l C.1 C.2) ∧ 
  (∃ (t : ℝ), (1 - t) • A + t • C = B) ∧
  (triangleArea A B C = 9/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l21_2192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_good_sequence_l21_2157

def is_good (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n > 0 → a (n.factorial) = (Finset.range n).prod (λ i ↦ a (i + 1))) ∧
  (∀ n : ℕ, n > 0 → ∃ b : ℕ, a n = b ^ n)

theorem unique_good_sequence :
  ∀ a : ℕ → ℕ, is_good a ↔ (∀ n : ℕ, n > 0 → a n = 1) :=
by
  sorry

#check unique_good_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_good_sequence_l21_2157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hno3_concentration_after_addition_l21_2116

/-- Represents a chemical solution --/
structure Solution where
  volume : ℝ
  concentration : ℝ

/-- Calculates the resultant solution after adding pure HNO3 --/
noncomputable def add_pure_hno3 (initial : Solution) (added_volume : ℝ) : Solution :=
  let total_volume := initial.volume + added_volume
  let total_hno3 := initial.volume * initial.concentration + added_volume
  { volume := total_volume,
    concentration := total_hno3 / total_volume }

/-- The theorem to be proved --/
theorem hno3_concentration_after_addition :
  let initial := Solution.mk 60 0.4
  let resultant := add_pure_hno3 initial 12
  resultant.concentration = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hno3_concentration_after_addition_l21_2116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_greater_negation_of_proposition_l21_2144

theorem negation_of_existence {α : Type*} (p : α → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by
  apply Iff.intro
  · intro h x px
    apply h
    exact ⟨x, px⟩
  · intro h ⟨x, px⟩
    exact h x px

theorem negation_of_greater {α : Type*} [LinearOrder α] (a b : α) :
  (¬ (a > b)) ↔ (a ≤ b) :=
by
  apply Iff.intro
  · intro h
    exact le_of_not_gt h
  · intro h h'
    exact not_le_of_gt h' h

theorem negation_of_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by
  apply Iff.intro
  · intro h n
    apply le_of_not_gt
    intro h'
    apply h
    exact ⟨n, h'⟩
  · intro h ⟨n, hn⟩
    exact not_lt_of_ge (h n) hn

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_greater_negation_of_proposition_l21_2144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wage_is_42675_l21_2105

/-- Represents the production deviation for each day of the week -/
def production_deviation : Fin 7 → Int
  | 0 => 5    -- Monday
  | 1 => -2   -- Tuesday
  | 2 => -4   -- Wednesday
  | 3 => 13   -- Thursday
  | 4 => -10  -- Friday
  | 5 => 16   -- Saturday
  | 6 => -9   -- Sunday

/-- The planned weekly production -/
def planned_weekly_production : Nat := 700

/-- The average daily production -/
def average_daily_production : Nat := 100

/-- The base wage per bicycle -/
def base_wage : Nat := 60

/-- The bonus wage for each overproduced bicycle -/
def bonus_wage : Nat := 15

/-- The deduction for each underproduced bicycle -/
def deduction_wage : Nat := 20

/-- Calculate the total wage for the week -/
def calculate_total_wage : Int :=
  let total_deviation := (List.sum (List.map production_deviation (List.range 7)))
  let actual_production := planned_weekly_production + total_deviation
  let base_total := actual_production * base_wage
  let bonus_total := total_deviation * (base_wage + bonus_wage)
  base_total + bonus_total

/-- The theorem stating that the total wage for the week is 42675 元 -/
theorem total_wage_is_42675 : calculate_total_wage = 42675 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wage_is_42675_l21_2105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l21_2140

/-- A hyperbola passing through (2,3) with asymptotes y = ±√3x has the standard equation x^2 - y^2/3 = 1 -/
theorem hyperbola_standard_equation (h : Set (ℝ × ℝ)) 
  (passes_through : (2, 3) ∈ h)
  (asymptotes : ∀ x y, (x, y) ∈ h → y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) :
  ∀ x y, (x, y) ∈ h ↔ x^2 - y^2/3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l21_2140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_tangent_to_lines_l21_2193

theorem circle_radius_tangent_to_lines (k : ℝ) (h : k > 8) :
  let center := (0, k)
  let radius := λ (x y : ℝ) ↦ Real.sqrt ((x - 0)^2 + (y - k)^2)
  let tangent_to_line := λ (a b c : ℝ) (x y : ℝ) ↦ |a*x + b*y + c| / Real.sqrt (a^2 + b^2) = radius x y
  (∀ x y, y = x → tangent_to_line 1 (-1) 0 x y) ∧
  (∀ x y, y = -x → tangent_to_line 1 1 0 x y) ∧
  (∀ x y, y = 8 → tangent_to_line 0 1 (-8) x y) →
  ∀ x y, radius x y = 8 * Real.sqrt 2 + 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_tangent_to_lines_l21_2193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l21_2177

theorem calculation_proof :
  (|(-2 + 1/4)| - (-3/4) + 1 - |1 - 1/2|) = 7/2 ∧
  (-3^2 - (8 / (-2)^3 - 1) + 3 / 2 * 1/2) = -25/4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l21_2177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_in_sqrt_function_l21_2190

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x - 3)

-- State the theorem
theorem range_of_x_in_sqrt_function :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x ≥ 3/2 :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_in_sqrt_function_l21_2190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l21_2100

noncomputable def f (n : ℤ) (x : ℝ) : ℝ := (n^2 + 2*n - 2) * x^(n^2 - 3*n)

theorem decreasing_power_function (n : ℤ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f n x₁ > f n x₂) →
  n = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l21_2100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_max_k_l21_2131

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 2 * x

noncomputable def g (x : ℝ) : ℝ := log ((x + 2) / (x - 2))

theorem f_increasing_and_max_k :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₁ > 0 ∧ x₂ > 0 → f x₁ > f x₂) ∧
  (∃ k : ℕ, k = 2 ∧
    (∀ k' : ℕ, 
      (∀ x₁ : ℝ, x₁ ∈ Set.Ioo 0 1 → 
        ∃ x₂ : ℝ, x₂ ∈ Set.Ioo (k' : ℝ) (k' + 1) ∧ f x₁ < g x₂) →
      k' ≤ k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_max_k_l21_2131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l21_2176

theorem sqrt_equation_solution (n : ℝ) : (Real.sqrt (2 * n))^2 + 12 * n / 4 - 7 = 64 → n = 14.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l21_2176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_1_trig_identity_2_l21_2103

-- Problem 1
theorem trig_identity_1 :
  Real.sin (120 * π / 180) ^ 2 + Real.cos (180 * π / 180) + Real.tan (45 * π / 180) -
  Real.cos (-330 * π / 180) ^ 2 + Real.sin (-210 * π / 180) = 1/2 := by sorry

-- Problem 2
theorem trig_identity_2 (α : Real) (h1 : Real.sin (π + α) = 1/2) (h2 : π < α) (h3 : α < 3*π/2) :
  Real.sin α - Real.cos α = (Real.sqrt 3 - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_1_trig_identity_2_l21_2103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l21_2153

/-- The value of m for a hyperbola with given equation and eccentricity -/
theorem hyperbola_m_value (x y m : ℝ) :
  (x^2 / 16 - y^2 / m = 1) →  -- hyperbola equation
  (Real.sqrt (1 + m/16) = 5/4) →       -- eccentricity condition
  m = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l21_2153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_saved_time_specific_values_l21_2115

-- Define the given conditions
def distance : ℝ := 100
def speed : ℝ → ℝ := λ v => v
def speed_increase : ℝ → ℝ := λ a => a

-- Define the time function
noncomputable def time (v : ℝ) : ℝ := distance / v

-- Define the time saved function
noncomputable def time_saved (v a : ℝ) : ℝ := time v - time (v + a)

-- Theorem for the time to travel
theorem travel_time (v : ℝ) (h : v > 0) : time v = 100 / v := by
  simp [time, distance]

-- Theorem for the time saved with increased speed
theorem saved_time (v a : ℝ) (h : v > 0) (h' : v + a > 0) :
  time_saved v a = 100 / v - 100 / (v + a) := by
  simp [time_saved, time, distance]

-- Theorem for specific values
theorem specific_values :
  time 40 = 2.5 ∧ time_saved 40 10 = 0.5 := by
  apply And.intro
  · simp [time, distance]
    norm_num
  · simp [time_saved, time, distance]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_saved_time_specific_values_l21_2115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_wins_probability_l21_2125

noncomputable def game_probability (p1 p2 : ℝ) : ℝ :=
  p1 / (1 - (1 - p1) * (1 - p2))

theorem larry_wins_probability :
  game_probability (1/3) (1/2) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_wins_probability_l21_2125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_chord_triangle_perimeter_l21_2151

/-- Represents a hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ  -- Left focus
  F₂ : ℝ × ℝ  -- Right focus
  a : ℝ       -- Half of the real axis length

/-- Represents a chord of the hyperbola -/
structure Chord where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The perimeter of a triangle given its three vertices -/
noncomputable def trianglePerimeter (p q r : ℝ × ℝ) : ℝ :=
  distance p q + distance q r + distance r p

theorem hyperbola_chord_triangle_perimeter 
  (h : Hyperbola) 
  (c : Chord) 
  (h_real_axis : h.a = 4)
  (c_length : distance c.A c.B = 5)
  (c_through_F₁ : c.A = h.F₁ ∨ c.B = h.F₁)
  : trianglePerimeter c.A c.B h.F₂ = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_chord_triangle_perimeter_l21_2151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_OA_perp_OB_circle_property_circle_center_locus_l21_2174

-- Define the Cartesian plane
variable (x y : ℝ)

-- Define points M and N
def M : ℝ × ℝ := (1, -3)
def N : ℝ × ℝ := (5, 1)

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the trajectory of point C
def trajectory_C (t : ℝ) : ℝ × ℝ := 
  (t * M.fst + (1-t) * N.fst, t * M.snd + (1-t) * N.snd)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Theorem 1: OA ⊥ OB
theorem OA_perp_OB : 
  A.fst * B.fst + A.snd * B.snd = 0 := by sorry

-- Define point P
def P : ℝ × ℝ := (4, 0)

-- Theorem 2: Circle property for chords through P
theorem circle_property (chord : Set (ℝ × ℝ)) :
  ∀ D E, D ∈ chord → E ∈ chord → parabola D.fst D.snd → parabola E.fst E.snd → 
  P ∈ chord →
  ∃ center : ℝ × ℝ, ∃ radius : ℝ, 
    (center.fst - 0)^2 + (center.snd - 0)^2 = radius^2 ∧
    (center.fst - D.fst)^2 + (center.snd - D.snd)^2 = radius^2 ∧
    (center.fst - E.fst)^2 + (center.snd - E.snd)^2 = radius^2 := by sorry

-- Theorem 3: Locus of circle centers
theorem circle_center_locus (center : ℝ × ℝ) :
  (∃ D E : ℝ × ℝ, parabola D.fst D.snd ∧ parabola E.fst E.snd ∧ 
   (D.fst - P.fst) * (E.snd - P.snd) = (E.fst - P.fst) * (D.snd - P.snd) ∧
   (center.fst - 0)^2 + (center.snd - 0)^2 = 
   ((D.fst - E.fst)^2 + (D.snd - E.snd)^2) / 4) →
  center.snd^2 = 2 * center.fst - 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_OA_perp_OB_circle_property_circle_center_locus_l21_2174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_count_total_prime_factors_in_expression_l21_2148

-- Define the expression
def expression (a b c : ℕ) : ℕ := (4^a) * (7^b) * (11^c)

-- Define the function to count prime factors
def count_prime_factors (a b c : ℕ) : ℕ := 2*a + b + c

-- Theorem statement
theorem prime_factors_count : 
  count_prime_factors 11 5 2 = 29 := by sorry

-- Main theorem
theorem total_prime_factors_in_expression : 
  ∃ n : ℕ, n = count_prime_factors 11 5 2 ∧ 
  n = (if expression 11 5 2 % 2 = 0 then 1 else 0) + 
      (if expression 11 5 2 % 3 = 0 then 1 else 0) + 
      (if expression 11 5 2 % 5 = 0 then 1 else 0) + 
      (if expression 11 5 2 % 7 = 0 then 1 else 0) + 
      (if expression 11 5 2 % 11 = 0 then 1 else 0) + 
      (if expression 11 5 2 % 13 = 0 then 1 else 0) := by sorry

-- Note: The original comment about continuing for all primes up to the square root
-- has been removed as it was causing a syntax error.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_count_total_prime_factors_in_expression_l21_2148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jose_bottle_caps_l21_2102

/-- Given Jose starts with 7 bottle caps and receives 2 more from Rebecca,
    prove that he ends up with 9 bottle caps. -/
theorem jose_bottle_caps : 7 + 2 = 9 := by
  rfl

#check jose_bottle_caps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jose_bottle_caps_l21_2102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_prime_odd_proposition_l21_2199

theorem negation_of_prime_odd_proposition :
  (¬ ∀ p : ℕ, Nat.Prime p → Odd p) ↔ (∃ p : ℕ, Nat.Prime p ∧ ¬ Odd p) := by
  apply Iff.intro
  · intro h
    push_neg at h
    exact h
  · intro h
    push_neg
    exact h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_prime_odd_proposition_l21_2199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_acute_angle_clock_l21_2124

/-- Represents a clock with hour and minute hands -/
structure Clock where
  hour : ℝ -- Hour hand position (0 ≤ hour < 12)
  minute : ℝ -- Minute hand position (0 ≤ minute < 60)

/-- The angle between the hour and minute hands of a clock -/
noncomputable def angle_between_hands (c : Clock) : ℝ :=
  sorry

/-- Predicate to check if an angle is acute -/
def is_acute (angle : ℝ) : Prop :=
  0 < angle ∧ angle < Real.pi/2

/-- The probability of an event occurring in a continuous uniform distribution -/
noncomputable def probability (event : Clock → Prop) : ℝ :=
  sorry

/-- The main theorem: probability of acute angle between clock hands -/
theorem prob_acute_angle_clock :
  probability (λ c : Clock => is_acute (angle_between_hands c)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_acute_angle_clock_l21_2124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l21_2123

noncomputable def f (x : ℝ) := 2^x + x - 8

theorem root_in_interval (x : ℝ) (k : ℤ) :
  f x = 0 → x > k → x < k + 1 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l21_2123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_lost_time_l21_2164

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_minutes_valid : minutes < 60

/-- Converts a Time to minutes since midnight -/
def Time.toMinutes (t : Time) : ℕ :=
  t.hours * 60 + t.minutes

/-- Calculates the difference in minutes between two Times -/
def timeDifference (t1 t2 : Time) : ℕ :=
  if t2.toMinutes ≥ t1.toMinutes then
    t2.toMinutes - t1.toMinutes
  else
    (24 * 60) - (t1.toMinutes - t2.toMinutes)

/-- Represents a clock that loses time -/
structure LosingClock where
  minutesLostPerHour : ℚ

theorem clock_lost_time (c : LosingClock) 
  (clockTime actualTime : Time) :
  c.minutesLostPerHour = 10 →
  clockTime.hours = 15 ∧ clockTime.minutes = 0 →
  actualTime.hours = 15 ∧ actualTime.minutes = 36 →
  (timeDifference clockTime actualTime : ℚ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_lost_time_l21_2164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_n_multiple_of_seven_l21_2119

theorem base_n_multiple_of_seven : 
  (Finset.filter (fun n => (2*n^5 + 3*n^4 + 5*n^3 + 2*n^2 + 3*n + 6) % 7 = 0) 
    (Finset.range 99)).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_n_multiple_of_seven_l21_2119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_theorem_l21_2183

/-- A quadrilateral -/
structure Quadrilateral where
  -- Define the properties of a quadrilateral here
  mk :: -- You can add necessary fields

/-- A triangle -/
structure Triangle where
  -- Define the properties of a triangle here
  mk :: -- You can add necessary fields

/-- A quadrilateral is circumscriptible if it has an inscribed circle -/
def is_circumscriptible (q : Quadrilateral) : Prop := sorry

/-- A quadrilateral is exscriptible if it has a circumscribed circle -/
def is_exscriptible (q : Quadrilateral) : Prop := sorry

/-- A division of a triangle into quadrilaterals -/
def TriangleDivision (t : Triangle) := List Quadrilateral

/-- Predicate to check if all quadrilaterals in a division are both circumscriptible and exscriptible -/
def all_circumscriptible_and_exscriptible (d : List Quadrilateral) : Prop :=
  ∀ q ∈ d, is_circumscriptible q ∧ is_exscriptible q

theorem triangle_division_theorem (t : Triangle) :
  ∃ (d : TriangleDivision t), d.length = 2019 ∧ all_circumscriptible_and_exscriptible d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_theorem_l21_2183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swiss_cross_theorem_l21_2149

/-- A Swiss cross is a shape consisting of 5 unit squares: one in the center and four on the sides. -/
def SwissCross : Set (ℝ × ℝ) :=
  {p | (p.1 ≥ -1 ∧ p.1 ≤ 2) ∧ (p.2 ≥ -1 ∧ p.2 ≤ 2) ∧ (|p.1| ≤ 1 ∨ |p.2| ≤ 1)}

/-- The property that for any n points in a Swiss cross, there exist two with distance less than 1. -/
def HasClosePoints (n : ℕ) : Prop :=
  ∀ (points : Finset (ℝ × ℝ)), points.card = n → points.toSet ⊆ SwissCross →
    ∃ p q : ℝ × ℝ, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) < 1

/-- 13 is the smallest natural number satisfying the HasClosePoints property. -/
theorem swiss_cross_theorem : (∀ m, m ≥ 13 → HasClosePoints m) ∧ ¬HasClosePoints 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swiss_cross_theorem_l21_2149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_theorem_l21_2112

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line passing through (1,0)
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the condition that OM is perpendicular to ON
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Theorem statement
theorem ellipse_line_theorem :
  ∀ k x1 y1 x2 y2 : ℝ,
  ellipse x1 y1 → ellipse x2 y2 →
  line k x1 y1 → line k x2 y2 →
  perpendicular x1 y1 x2 y2 →
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 :=
by
  sorry

#check ellipse_line_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_theorem_l21_2112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_tank_time_l21_2146

/-- Represents the tank and its properties -/
structure Tank where
  volume : ℚ  -- volume in cubic feet
  inlet_rate : ℚ  -- inlet rate in cubic inches per minute
  outlet_rate1 : ℚ  -- first outlet rate in cubic inches per minute
  outlet_rate2 : ℚ  -- second outlet rate in cubic inches per minute

/-- Calculates the time to empty the tank in minutes -/
def time_to_empty (t : Tank) : ℚ :=
  (t.volume * 12 * 12 * 12) / ((t.outlet_rate1 + t.outlet_rate2) - t.inlet_rate)

/-- Theorem stating the time to empty the specific tank -/
theorem empty_tank_time :
  let tank := Tank.mk 30 3 9 6
  time_to_empty tank = 4320 := by
  sorry

#eval time_to_empty (Tank.mk 30 3 9 6)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_tank_time_l21_2146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_alternate_interior_angles_contrapositive_l21_2197

/-- Two lines in a plane -/
structure Line where

/-- Angle between two lines -/
def Angle (l1 l2 : Line) : ℝ := sorry

/-- Two lines are parallel -/
def Parallel (l1 l2 : Line) : Prop := sorry

/-- Alternate interior angles of two lines -/
def AlternateInteriorAngles (l1 l2 : Line) : Prop :=
  ∃ a1 a2 : ℝ, Angle l1 l2 = a1 ∧ Angle l1 l2 = a2 ∧ a1 = a2

theorem parallel_alternate_interior_angles_contrapositive (l1 l2 : Line) :
  (¬ AlternateInteriorAngles l1 l2 → ¬ Parallel l1 l2) ↔
  (Parallel l1 l2 → AlternateInteriorAngles l1 l2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_alternate_interior_angles_contrapositive_l21_2197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_neg6_to_7_l21_2191

theorem arithmetic_mean_neg6_to_7 : 
  let sequence := List.range 14
  let shifted_sequence := sequence.map (λ x => (x : ℤ) - 6)
  (shifted_sequence.sum : ℚ) / shifted_sequence.length = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_neg6_to_7_l21_2191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l21_2127

theorem triangle_shape (A B C : Real) (h1 : 0 < A ∧ A < π/2) 
  (h2 : 0 < B ∧ B < π/2) (h3 : Real.cos A > Real.sin B) (h4 : A + B + C = π) :
  C > π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l21_2127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l21_2138

theorem inequality_solution_set 
  (a b : ℝ) 
  (h1 : Set.Ioi 1 = {x : ℝ | a * x + b > 0}) :
  Set.Ioo 1 2 = {x : ℝ | (a * x + b) * (x - 2) < 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l21_2138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_sample_size_l21_2194

/-- Represents the total number of students in the sample -/
def total_students : ℕ := sorry

/-- Represents the number of freshmen in the sample -/
def freshmen : ℕ := sorry

/-- Represents the number of sophomores in the sample -/
def sophomores : ℕ := sorry

/-- Represents the number of juniors in the sample -/
def juniors : ℕ := sorry

/-- Represents the number of seniors in the sample -/
def seniors : ℕ := sorry

theorem student_sample_size :
  -- All students are either freshmen, sophomores, juniors, or seniors
  (total_students = freshmen + sophomores + juniors + seniors) →
  -- 22% are juniors
  (juniors = (22 * total_students) / 100) →
  -- 75% are not sophomores (which means 25% are sophomores)
  (sophomores = (25 * total_students) / 100) →
  -- There are 160 seniors
  (seniors = 160) →
  -- There are 64 more freshmen than sophomores
  (freshmen = sophomores + 64) →
  -- The total number of students in the sample is 800
  total_students = 800 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_sample_size_l21_2194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_range_l21_2155

theorem triangle_inequality_range (a : ℝ) : 
  (∃ (t : Set (ℝ × ℝ × ℝ)), t.Nonempty ∧ (∀ x y z, (x, y, z) ∈ t → x + y > z ∧ y + z > x ∧ z + x > y) ∧ (7, 12, a) ∈ t) ↔ 5 < a ∧ a < 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_range_l21_2155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_hyperbola_a_l21_2120

-- Define the golden ratio
noncomputable def golden_ratio : ℝ := (Real.sqrt 5 - 1) / 2

-- Define the eccentricity of a hyperbola
noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := Real.sqrt ((a + 1) / a)

-- Theorem statement
theorem golden_hyperbola_a (a : ℝ) :
  (hyperbola_eccentricity a = 1 / golden_ratio) → a = golden_ratio := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_hyperbola_a_l21_2120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_x0_l21_2179

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem value_of_x0 (x0 : ℝ) (h : x0 > 0) :
  f x0 + (deriv f) x0 = 1 → x0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_x0_l21_2179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l21_2128

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

-- Define the domain
def domain : Set ℝ := { x | Real.pi / 6 ≤ x ∧ x ≤ 5 * Real.pi / 6 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -1 ≤ y ∧ y ≤ 1/2 } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l21_2128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l21_2167

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (1, -1)

-- Define the function to calculate the area of a triangle given its vertices
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))

-- Theorem statement
theorem triangle_ABC_area : triangleArea A B C = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l21_2167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_l21_2114

open Real

/-- The function f(x) = ln x + x² + x -/
noncomputable def f (x : ℝ) : ℝ := log x + x^2 + x

/-- Theorem: If x₁ and x₂ are positive real numbers satisfying
    f(x₁) + f(x₂) + x₁x₂ = 0, then x₁ + x₂ ≥ (√5 - 1)/2 -/
theorem sum_lower_bound (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0)
    (h : f x₁ + f x₂ + x₁ * x₂ = 0) :
    x₁ + x₂ ≥ (sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_l21_2114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l21_2126

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + 2 * (Real.cos x)^2 - 2

theorem f_properties :
  ∃ (T : ℝ) (max_value : ℝ) (max_set : Set ℝ) (range : Set ℝ),
    -- Smallest positive period
    (∀ x, f (x + T) = f x) ∧ (∀ t > 0, (∀ x, f (x + t) = f x) → T ≤ t) ∧ T = π ∧
    -- Maximum value
    (∀ x, f x ≤ max_value) ∧ max_value = Real.sqrt 2 ∧
    -- Set where maximum occurs
    (∀ x, x ∈ max_set ↔ f x = max_value) ∧
    (∀ x, x ∈ max_set ↔ ∃ k : ℤ, x = k * π + π / 8) ∧
    -- Range when x ∈ [π/4, 3π/4]
    (∀ y, y ∈ range ↔ ∃ x, x ∈ Set.Icc (π / 4) (3 * π / 4) ∧ f x = y) ∧
    range = Set.Icc (-Real.sqrt 2) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l21_2126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_cost_l21_2188

/-- The total cost of a road trip shared by friends -/
def total_cost : ℝ := 74.67

/-- The number of friends initially planning the trip -/
def initial_friends : ℕ := 4

/-- The number of friends who joined later -/
def additional_friends : ℕ := 3

/-- The total number of friends after others joined -/
def total_friends : ℕ := initial_friends + additional_friends

/-- The amount by which the cost decreased for each original friend -/
def cost_decrease : ℝ := 8

theorem road_trip_cost : 
  (total_cost / initial_friends) - (total_cost / total_friends) = cost_decrease := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_cost_l21_2188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_grid_sizes_l21_2196

/-- A move on the grid removes exactly three edges surrounding a cell. -/
def ValidMove (n : ℕ) (grid : Fin n → Fin n → ℕ) : Prop :=
  ∃ (i j : Fin n), grid i j ≥ 3 ∧ ∃ (grid' : Fin n → Fin n → ℕ), 
    grid' i j = grid i j - 3 ∧ 
    ∀ (x y : Fin n), (x ≠ i ∨ y ≠ j) → grid' x y = grid x y

/-- The goal is to remove all edges from the grid. -/
def AllEdgesRemoved (n : ℕ) (grid : Fin n → Fin n → ℕ) : Prop :=
  ∀ (i j : Fin n), grid i j = 0

/-- The sequence of moves that removes all edges from the grid. -/
def ValidSequence (n : ℕ) : Prop :=
  ∃ (steps : ℕ) (sequence : Fin (steps + 1) → Fin n → Fin n → ℕ),
    (∀ (k : Fin steps), ValidMove n (sequence k)) ∧
    AllEdgesRemoved n (sequence ⟨steps, by simp⟩)

/-- The main theorem: characterization of valid grid sizes. -/
theorem valid_grid_sizes (n : ℕ) (h : n > 0) :
  ValidSequence n ↔ (n = 2 ∨ (n > 2 ∧ (n % 6 = 3 ∨ n % 6 = 5))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_grid_sizes_l21_2196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_coprime_sum_l21_2161

/-- A function that checks if three positive integers are pairwise coprime -/
def are_pairwise_coprime (a b c : ℕ) : Prop :=
  Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime c a

/-- A function that checks if a natural number can be expressed as the sum of three
    pairwise coprime integers, each greater than 1 -/
def has_coprime_sum (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 1 ∧ b > 1 ∧ c > 1 ∧ are_pairwise_coprime a b c ∧ a + b + c = n

/-- Theorem stating that 17 is the largest positive integer that cannot be expressed
    as the sum of three pairwise coprime integers, each greater than 1 -/
theorem largest_non_coprime_sum :
  (∀ n : ℕ, n > 17 → has_coprime_sum n) ∧
  ¬(has_coprime_sum 17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_coprime_sum_l21_2161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l21_2107

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

theorem tangent_slope_angle (x₀ : ℝ) (h : x₀ = 1) :
  let y₀ := f x₀
  let m := deriv f x₀
  Real.arctan m = π/4 := by
    -- We'll use 'sorry' to skip the proof for now
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l21_2107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l21_2110

/-- Represents a contestant in the race -/
structure Contestant where
  speed : ℚ
  startPoint : ℚ

/-- Calculates the time taken by a contestant to finish the race -/
def timeTaken (c : Contestant) (raceDistance : ℚ) : ℚ :=
  (raceDistance - c.startPoint) / c.speed

/-- The main theorem about the race -/
theorem race_result (x : ℚ) (hx : x > 0) :
  let raceDistance : ℚ := 500
  let a := Contestant.mk (3 * x) 140
  let b := Contestant.mk (4 * x) 0
  let c := Contestant.mk (5 * x) 60
  let d := Contestant.mk (6 * x) 20
  timeTaken b raceDistance = timeTaken a raceDistance - 5 / x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l21_2110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_prime_permutation_characterization_l21_2189

theorem fermat_prime_permutation_characterization (k : ℕ) :
  let n := 2^k + 1
  Prime n ↔ ∃ (a : Fin n → Fin n) (g : Fin n → ℕ),
    Function.Bijective a ∧
    ∀ i : Fin n, n ∣ (g i)^((a i).val + 1) - ((a (i + 1)).val + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_prime_permutation_characterization_l21_2189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_minimum_l21_2173

/-- The function representing the curve -/
noncomputable def f (x : ℝ) : ℝ := 4 * x^3 + 3 * Real.log x

/-- The derivative of the function -/
noncomputable def f' (x : ℝ) : ℝ := 12 * x^2 + 3 / x

theorem tangent_slope_minimum (x : ℝ) (h : x > 0) :
  f' x ≥ 9 ∧ ∃ y : ℝ, y > 0 ∧ f' y = 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_minimum_l21_2173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_max_probability_l21_2142

/-- The number of trials in the binomial distribution -/
def n : ℕ := 100

/-- The probability of success in each trial -/
noncomputable def p : ℚ := 1/2

/-- The binomial distribution random variable -/
noncomputable def ζ : ℕ → ℝ := sorry

/-- The probability mass function of the binomial distribution -/
noncomputable def pmf (k : ℕ) : ℝ := (n.choose k) * p^k * (1-p)^(n-k)

/-- The value of k that maximizes the probability mass function -/
def k_max : ℕ := n / 2

theorem binomial_max_probability :
  ∀ k : ℕ, k ≠ k_max → pmf k ≤ pmf k_max := by
  sorry

#check binomial_max_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_max_probability_l21_2142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_of_five_eq_seven_point_five_l21_2165

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := 5 / (3 - x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := 3 - 5 / x

-- Define the function p
noncomputable def p (x : ℝ) : ℝ := 1 / h_inv x + 7

-- Theorem statement
theorem p_of_five_eq_seven_point_five : p 5 = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_of_five_eq_seven_point_five_l21_2165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_is_112_l21_2185

/-- Work done by a force function F(x) moving a body from point a to point b -/
noncomputable def work (F : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, F x

/-- The specific force function for this problem -/
def F (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 3

/-- Theorem stating that the work done by F(x) from x=1 to x=5 is 112 -/
theorem work_done_is_112 :
  work F 1 5 = 112 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_is_112_l21_2185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cylinder_properties_l21_2106

/-- Represents a homogeneous hollow truncated cylinder -/
structure TruncatedCylinder where
  R : ℝ  -- External radius
  r : ℝ  -- Internal radius
  H : ℝ  -- Height
  δ : ℝ  -- Density
  h_R_pos : R > 0
  h_r_pos : r > 0
  h_H_pos : H > 0
  h_R_gt_r : R > r
  h_δ_pos : δ > 0

/-- Center of gravity of a truncated cylinder -/
noncomputable def centerOfGravity (c : TruncatedCylinder) : ℝ × ℝ :=
  (-(c.R^2 + c.r^2) / (4 * c.R), (c.H * (3 * c.R^2 - c.r^2)) / (16 * c.R^2))

/-- Moment of inertia of a truncated cylinder with respect to its axis -/
noncomputable def momentOfInertia (c : TruncatedCylinder) : ℝ :=
  (Real.pi * c.δ * (c.R^4 - c.r^4)) / 4

theorem truncated_cylinder_properties (c : TruncatedCylinder) (h_δ_one : c.δ = 1) :
  centerOfGravity c = (-(c.R^2 + c.r^2) / (4 * c.R), (c.H * (3 * c.R^2 - c.r^2)) / (16 * c.R^2)) ∧
  momentOfInertia c = (Real.pi * (c.R^4 - c.r^4)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cylinder_properties_l21_2106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_five_points_l21_2113

def bag_size : ℕ := 2
def num_draws : ℕ := 3
def red_score : ℕ := 2
def black_score : ℕ := 1
def target_score : ℕ := 5

def probability_of_score (score : ℕ) : ℚ :=
  (Nat.choose num_draws (score / red_score) * (1 : ℚ) ^ (score / red_score) * (1 : ℚ) ^ (num_draws - score / red_score)) / (bag_size ^ num_draws)

theorem probability_of_five_points :
  probability_of_score target_score = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_five_points_l21_2113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_l21_2154

/-- Given vectors a and b in a real inner product space, with |a| = 2, |b| = 1,
    and the angle between a and b is 2π/3, prove that |a + 2b| = 2 -/
theorem magnitude_of_sum {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (a b : V)
    (ha : ‖a‖ = 2)
    (hb : ‖b‖ = 1)
    (hab : inner a b = ‖a‖ * ‖b‖ * Real.cos (2 * Real.pi / 3)) :
    ‖a + 2 • b‖ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_l21_2154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_without_goblet_l21_2122

/-- Represents the type of wine in a goblet -/
inductive Wine
| Red
| White

/-- Represents a knight at the round table -/
structure Knight where
  position : Fin 50
  wine : Wine

/-- Represents the state of the round table -/
def RoundTable := Fin 50 → Knight

/-- Defines how goblets are passed based on the wine type -/
def passGoblet (table : RoundTable) (i : Fin 50) : Fin 50 :=
  match (table i).wine with
  | Wine.Red => i + 1
  | Wine.White => i - 2

theorem at_least_one_without_goblet (table : RoundTable) 
  (h1 : ∃ i j : Fin 50, (table i).wine = Wine.Red ∧ (table j).wine = Wine.White) :
  ∃ k : Fin 50, ∀ i : Fin 50, passGoblet table i ≠ k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_without_goblet_l21_2122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l21_2108

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (x^2 + 1)

-- State the theorem
theorem odd_function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1) 1 → f a b x = -f a b (-x)) →  -- f is odd on (-1, 1)
  f a b (1/2) = 2/5 →                             -- f(1/2) = 2/5
  (∃ g : ℝ → ℝ, 
    (∀ x, x ∈ Set.Ioo (-1) 1 → g x = x / (x^2 + 1)) ∧  -- Explicit formula
    (∀ x y, x ∈ Set.Ioo (-1) 1 → y ∈ Set.Ioo (-1) 1 → x < y → g x < g y) ∧  -- Monotonicity
    (∀ x, x ∈ Set.Ioo 0 (1/3) → g (2*x-1) + g x < 0) ∧ -- Inequality solution
    (∀ x, x ∉ Set.Ioo 0 (1/3) → g (2*x-1) + g x ≥ 0) ∧
    (∀ x, x ∈ Set.Ioo (-1) 1 → g x = f a b x)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l21_2108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l21_2184

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

noncomputable def line_slope : ℝ := 2 * Real.sqrt 2

-- Define the intersection points
structure Point where
  x : ℝ
  y : ℝ

def intersect (p : ℝ) (A B : Point) : Prop :=
  parabola p A.x A.y ∧ parabola p B.x B.y ∧
  A.y = line_slope * (A.x - p/2) ∧ B.y = line_slope * (B.x - p/2)

-- Define the distance between points
noncomputable def distance (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

-- The main theorem
theorem parabola_intersection 
  (p : ℝ) (A B : Point) 
  (h_pos : p > 0)
  (h_intersect : intersect p A B)
  (h_order : A.x < B.x)
  (h_distance : distance A B = 9) :
  p = 4 ∧ 
  A = ⟨1, -2 * Real.sqrt 2⟩ ∧ 
  B = ⟨4, 4 * Real.sqrt 2⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l21_2184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l21_2182

/-- Binary operation star for positive real numbers -/
noncomputable def star (a b : ℝ) : ℝ := a^(b + 1)

/-- Theorem stating the properties of the star operation -/
theorem star_properties (a b n : ℝ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) : 
  (star a b = a^(b + 1)) ∧ (star a (b^n) = a^(b^n + 1)) := by
  constructor
  · -- Proof of the first part
    rfl
  · -- Proof of the second part
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l21_2182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l21_2117

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let line := λ (x : ℝ) ↦ x + Real.sqrt 2
  let ellipse := λ (x y : ℝ) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let O := (0 : ℝ × ℝ)
  ∃ M N : ℝ × ℝ, 
    ellipse M.1 M.2 ∧ 
    ellipse N.1 N.2 ∧ 
    M.2 = line M.1 ∧ 
    N.2 = line N.1 ∧ 
    (M.1 * N.1 + M.2 * N.2 = 0) ∧ 
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = 6 →
  a^2 = 4 + 2 * Real.sqrt 2 ∧ b^2 = 4 - 2 * Real.sqrt 2 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l21_2117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reservation_charge_is_six_l21_2180

/-- Represents the cost of a railway ticket --/
structure TicketCost where
  fullFare : ℚ
  reservationCharge : ℚ

/-- Calculates the total cost of a full ticket --/
def fullTicketCost (t : TicketCost) : ℚ :=
  t.fullFare + t.reservationCharge

/-- Calculates the total cost of a half ticket --/
def halfTicketCost (t : TicketCost) : ℚ :=
  t.fullFare / 2 + t.reservationCharge

/-- The theorem stating the reservation charge based on given conditions --/
theorem reservation_charge_is_six (t : TicketCost) 
  (h1 : fullTicketCost t = 216)
  (h2 : fullTicketCost t + halfTicketCost t = 327) : 
  t.reservationCharge = 6 := by
  sorry

#check reservation_charge_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reservation_charge_is_six_l21_2180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l21_2133

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if tan A = 7 tan B and (a² - b²) / c = 4, then c = 16/3 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  Real.sin A / a = Real.sin B / b →
  Real.sin A / a = Real.sin C / c →
  Real.tan A = 7 * Real.tan B →
  (a^2 - b^2) / c = 4 →
  c = 16/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l21_2133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_speed_l21_2141

/-- The speed of the river current in miles per hour -/
def river_speed : ℝ := 2

/-- The distance traveled downstream and upstream -/
def d : ℝ := sorry

/-- The man's regular rowing speed in still water -/
def r : ℝ := sorry

/-- Time equation for normal speed -/
axiom normal_speed_eq : r^2 - river_speed^2 - d * river_speed = 0

/-- Time equation for tripled speed -/
axiom tripled_speed_eq : 9 * r^2 - river_speed^2 - d * river_speed = 0

/-- The speed of the river current is 2 miles per hour -/
theorem river_current_speed : river_speed = 2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_speed_l21_2141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_ratio_for_given_cycle_l21_2104

/-- Represents a thermodynamic cycle with two isochoric and two adiabatic processes -/
structure ThermodynamicCycle where
  Tmax : ℝ
  Tmin : ℝ
  η : ℝ

/-- The ratio of final to initial absolute temperatures during the isochoric heating process -/
noncomputable def temperatureRatio (cycle : ThermodynamicCycle) : ℝ :=
  (cycle.Tmax / cycle.Tmin) * (1 - cycle.η)

/-- Theorem stating the temperature ratio for the given cycle parameters -/
theorem temperature_ratio_for_given_cycle :
  let cycle : ThermodynamicCycle := {
    Tmax := 900,
    Tmin := 350,
    η := 0.4
  }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |temperatureRatio cycle - 1.54| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_ratio_for_given_cycle_l21_2104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_4_l21_2130

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if abs x ≤ 1 then Real.sqrt x else 1 / x

-- Theorem statement
theorem f_of_f_4 : f (f 4) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_4_l21_2130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l21_2152

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.b = 1 ∧ Real.cos t.C + (2 * t.a + t.c) * Real.cos t.B = 0

/-- The theorem stating the angle B and maximum area -/
theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.B = Real.pi / 3 ∧ 
  ∃ (max_area : ℝ), max_area = Real.sqrt 3 / 12 ∧ 
    ∀ (area : ℝ), area = 1 / 2 * t.a * Real.sin t.B → area ≤ max_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l21_2152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_ratio_triangle_l21_2171

theorem largest_angle_in_ratio_triangle : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  5 * a = 4 * b →
  9 * a = 4 * c →
  c = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_ratio_triangle_l21_2171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l21_2160

-- Define the parameters
noncomputable def train_A_length : ℝ := 90
noncomputable def train_B_length : ℝ := 410.04
noncomputable def train_A_speed_kmph : ℝ := 120
noncomputable def train_B_speed_kmph : ℝ := 80

-- Define the function to calculate the time taken for trains to cross
noncomputable def time_to_cross (l_A l_B v_A v_B : ℝ) : ℝ :=
  let total_length := l_A + l_B
  let relative_speed := (v_A + v_B) * 1000 / 3600
  total_length / relative_speed

-- State the theorem
theorem trains_crossing_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |time_to_cross train_A_length train_B_length train_A_speed_kmph train_B_speed_kmph - 9| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l21_2160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l21_2198

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin (3 * x) + 5 * Real.sqrt 3 * Real.cos (3 * x)

/-- Theorem: f(x) is monotonically increasing on [0, π/20] -/
theorem f_monotone_increasing :
  MonotoneOn f (Set.Icc 0 (Real.pi / 20)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l21_2198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_power_sum_l21_2137

theorem not_prime_power_sum (x y : ℕ) (hx : 2 ≤ x) (hy : 2 ≤ y) (hx_upper : x ≤ 100) (hy_upper : y ≤ 100) :
  ∃ n : ℕ+, ¬ (Nat.Prime ((x^(2^n.val) : ℕ) + (y^(2^n.val) : ℕ))) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_power_sum_l21_2137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_closure_gcd_sum_div_l21_2163

theorem set_closure_gcd_sum_div (M : Set ℕ) :
  (∀ a b, a ∈ M → b ∈ M → (a + b) / Nat.gcd a b ∈ M) →
  (M = Set.univ ∨ M = Set.univ \ {1}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_closure_gcd_sum_div_l21_2163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_equation_solution_l21_2132

open Real MeasureTheory

/-- The heat equation solution -/
noncomputable def u (x t : ℝ) : ℝ :=
  3/4 * Real.exp (-4 * Real.pi^2 * t) * Real.sin (2 * Real.pi * x) -
  1/4 * Real.exp (-36 * Real.pi^2 * t) * Real.sin (6 * Real.pi * x)

/-- The heat equation -/
theorem heat_equation_solution (x t : ℝ) (hx : x ∈ Set.Ioo 0 1) (ht : t > 0) :
  deriv (fun t => u x t) t = deriv (fun x => deriv (u x) t) x ∧
  u x 0 = Real.sin (2 * Real.pi * x)^3 ∧
  u 0 t = 0 ∧ u 1 t = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_equation_solution_l21_2132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_six_digit_square_split_l21_2109

theorem unique_six_digit_square_split : ∃! n : ℕ, 
  (100000 ≤ n ∧ n < 1000000) ∧  -- six-digit number
  (∃ m : ℕ, n = m^2) ∧  -- perfect square
  (∃ a b : ℕ, 
    10 ≤ a ∧ a < 100 ∧  -- a is two-digit
    10 ≤ b ∧ b < 100 ∧  -- b is two-digit
    n = 100*a + b + a ∧  -- split into three two-digit numbers
    2*b = a) ∧  -- middle number is half of outer numbers
  n = 763876 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_six_digit_square_split_l21_2109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_with_given_angle_and_perimeter_l21_2181

/-- The area of a sector with central angle α and perimeter p -/
noncomputable def sectorArea (α : ℝ) (p : ℝ) : ℝ :=
  let r := p / (α + 2)
  (1 / 2) * α * r ^ 2

theorem sector_area_with_given_angle_and_perimeter :
  sectorArea 2 3 = 9 / 16 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_with_given_angle_and_perimeter_l21_2181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_a_2_monotonicity_intervals_l21_2168

noncomputable section

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 2*a*(Real.log x)

-- Theorem for the extreme values when a = 2
theorem extreme_values_a_2 :
  let a := 2
  ∃ x_max x_min : ℝ, x_max > 0 ∧ x_min > 0 ∧
    (∀ x : ℝ, x > 0 → f a x ≤ f a x_max) ∧
    (∀ x : ℝ, x > 0 → f a x ≥ f a x_min) ∧
    f a x_max = -5 ∧ f a x_min = 4 * Real.log 2 - 8 :=
by sorry

-- Theorem for intervals of monotonicity
theorem monotonicity_intervals (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 1 →
    (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < a → f a x < f a y) ∧
    (∀ x y : ℝ, a < x ∧ x < y ∧ y < 1 → f a x > f a y) ∧
    (∀ x y : ℝ, 1 < x ∧ x < y → f a x < f a y)) ∧
  (a = 1 →
    ∀ x y : ℝ, 0 < x ∧ x < y → f a x < f a y) ∧
  (a > 1 →
    (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 → f a x < f a y) ∧
    (∀ x y : ℝ, 1 < x ∧ x < y ∧ y < a → f a x > f a y) ∧
    (∀ x y : ℝ, a < x ∧ x < y → f a x < f a y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_a_2_monotonicity_intervals_l21_2168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equality_l21_2118

theorem complex_modulus_equality (a : ℝ) : 
  (Complex.abs (2 + a * Complex.I) = Complex.abs (3 - Complex.I)) → 
  (a = Real.sqrt 6 ∨ a = -Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equality_l21_2118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l21_2111

theorem problem_solution : 
  ((∀ (x y : ℝ), x = Real.sqrt 12 ∧ y = Real.sqrt (4/3) → x - y = (4*Real.sqrt 3)/3) ∧
   (∀ (a b : ℝ), a = Real.sqrt 5 ∧ b = Real.sqrt 3 → (a - b)^2 + (a + b)*(a - b) = 10 - 2*Real.sqrt 15)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l21_2111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l21_2129

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Assumptions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x < 0, f' x = deriv f x)
variable (h3 : ∀ x < 0, (2 * f x) / x + f' x < 0)

-- Theorem statement
theorem solution_set :
  {x : ℝ | (x + 2015)^2 * f (x + 2015) - 4 * f (-2) > 0} = Set.Ioi (-2017) := by
  sorry

#check solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l21_2129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l21_2156

-- Define the line
def line (x : ℝ) : ℝ := 2 * x + 1

-- Define the circle
def circleC (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Theorem stating that the line intersects the circle
theorem line_intersects_circle : ∃ (x y : ℝ), y = line x ∧ circleC x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l21_2156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l21_2166

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

-- Define the moving line l through the origin
def l (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define the trajectory C
def C (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = 9/4 ∧ 5/3 < x ∧ x ≤ 3

-- Define the line L
def L (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

-- Main theorem
theorem circle_intersection_theorem :
  -- The center of C₁ is at (3, 0)
  (∀ x y, C₁ x y ↔ (x - 3)^2 + y^2 = 4) ∧
  -- The trajectory C is correct
  (∀ k x y, (∃ x₁ y₁ x₂ y₂, C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ l k x₁ y₁ ∧ l k x₂ y₂ ∧ 
    x = (x₁ + x₂)/2 ∧ y = (y₁ + y₂)/2) ↔ C x y) ∧
  -- The condition for L to intersect C at only one point
  (∀ k, (∃! x y, C x y ∧ L k x y) ↔ 
    (k ∈ Set.Icc (-2 * Real.sqrt 5 / 7) (2 * Real.sqrt 5 / 7) ∨ k = -3/4 ∨ k = 3/4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l21_2166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_ratio_l21_2178

-- Define the points X, Y, Z in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the distance between two points
noncomputable def distance (A B : Point3D) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2)

theorem midpoint_triangle_ratio 
  (p q r : ℝ) (X Y Z : Point3D)
  (h1 : (Y.x + Z.x) / 2 = p ∧ (Y.y + Z.y) / 2 = 0 ∧ (Y.z + Z.z) / 2 = 0)
  (h2 : (X.x + Z.x) / 2 = 0 ∧ (X.y + Z.y) / 2 = q ∧ (X.z + Z.z) / 2 = 0)
  (h3 : (X.x + Y.x) / 2 = 0 ∧ (X.y + Y.y) / 2 = 0 ∧ (X.z + Y.z) / 2 = r) :
  (distance X Y)^2 + (distance X Z)^2 + (distance Y Z)^2 = 8 * (p^2 + q^2 + r^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_ratio_l21_2178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_triple_well_defined_and_nonzero_l21_2143

noncomputable def sequence_triple : ℕ → ℝ × ℝ × ℝ
| 0 => (2, 4, 6/7)
| n + 1 =>
  let (x, y, z) := sequence_triple n
  (2*x / (x^2 - 1), 2*y / (y^2 - 1), 2*z / (z^2 - 1))

theorem sequence_triple_well_defined_and_nonzero :
  ∀ n : ℕ,
    let (x, y, z) := sequence_triple n
    (x^2 ≠ 1 ∧ y^2 ≠ 1 ∧ z^2 ≠ 1) ∧ x + y + z ≠ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_triple_well_defined_and_nonzero_l21_2143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l21_2101

theorem cos_alpha_minus_pi_fourth (α : Real) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) (h3 : Real.tan α = 2) : 
  Real.cos (α - Real.pi / 4) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l21_2101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_time_l21_2186

/-- Represents the speed and time characteristics of a bus journey -/
structure BusJourney where
  speed_without_stops : ℝ
  speed_with_stops : ℝ
  total_time : ℝ
  stop_time : ℝ

/-- Calculates the stop time for a bus given its speeds with and without stops -/
noncomputable def calculate_stop_time (speed_without_stops speed_with_stops : ℝ) : ℝ :=
  60 * (1 - speed_with_stops / speed_without_stops)

/-- Theorem: A bus with average speeds of 80 km/hr without stops and 40 km/hr with stops
    will stop for 30 minutes per hour -/
theorem bus_stop_time :
  let bus : BusJourney := {
    speed_without_stops := 80
    speed_with_stops := 40
    total_time := 60
    stop_time := 30
  }
  calculate_stop_time bus.speed_without_stops bus.speed_with_stops = bus.stop_time := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_time_l21_2186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_is_6pi_l21_2159

-- Define the cylinder
structure Cylinder where
  height : ℝ
  baseRadius : ℝ

-- Define the sphere
structure Sphere where
  diameter : ℝ

-- Helper function to calculate volume
noncomputable def volume (c : Cylinder) : ℝ :=
  Real.pi * c.baseRadius^2 * c.height

-- Theorem statement
theorem cylinder_volume_is_6pi 
  (c : Cylinder) 
  (s : Sphere) 
  (h_height : c.height = 2)
  (h_diameter : s.diameter = 4)
  (h_base_on_sphere : c.baseRadius^2 + 1 = (s.diameter / 2)^2) :
  volume c = 6 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_is_6pi_l21_2159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l21_2139

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2 else -z^2

-- State the theorem
theorem f_composition_result :
  f (f (f (f (2 + I)))) = 164833 + 354192 * I :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l21_2139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gdp_growth_and_cpi_l21_2170

-- Define the structure for a good
structure Good where
  name : String
  price2009 : ℝ
  quantity2009 : ℝ
  price2015 : ℝ
  quantity2015 : ℝ

-- Define the goods
def alpha : Good := ⟨"Alpha", 5, 12, 6, 15⟩
def beta : Good := ⟨"Beta", 7, 8, 5, 10⟩
def gamma : Good := ⟨"Gamma", 9, 6, 10, 2⟩

-- Define the list of goods
def goods : List Good := [alpha, beta, gamma]

-- Calculate nominal GDP for a given year
noncomputable def nominalGDP (year : Nat) (goods : List Good) : ℝ :=
  goods.foldl (fun acc g => 
    acc + if year = 2009 then g.price2009 * g.quantity2009
          else g.price2015 * g.quantity2015) 0

-- Calculate real GDP for 2015 using 2009 prices
noncomputable def realGDP2015 (goods : List Good) : ℝ :=
  goods.foldl (fun acc g => acc + g.price2009 * g.quantity2015) 0

-- Calculate the growth rate of real GDP
noncomputable def realGDPGrowthRate (goods : List Good) : ℝ :=
  (realGDP2015 goods - nominalGDP 2009 goods) / nominalGDP 2009 goods

-- Calculate the CPI for 2015
noncomputable def CPI2015 (goods : List Good) : ℝ :=
  100 * (goods.foldl (fun acc g => acc + g.price2015 * g.quantity2009) 0) / 
        (goods.foldl (fun acc g => acc + g.price2009 * g.quantity2009) 0)

theorem gdp_growth_and_cpi :
  ∃ (ε : ℝ), ε > 0 ∧ 
  abs (realGDPGrowthRate goods + 0.0412) < ε ∧
  abs (CPI2015 goods - 101.17) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gdp_growth_and_cpi_l21_2170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l21_2121

-- Define the nabla operation
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Theorem statement
theorem nabla_calculation : nabla (nabla 2 3) 4 = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l21_2121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_majorAxisLength_of_parametric_ellipse_l21_2135

/-- The length of the major axis of an ellipse defined by parametric equations -/
noncomputable def majorAxisLength (a b : ℝ) : ℝ :=
  2 * max a b

/-- The ellipse defined by parametric equations x = a * cos(φ) and y = b * sin(φ) -/
noncomputable def ellipseParametric (a b : ℝ) (φ : ℝ) : ℝ × ℝ :=
  (a * Real.cos φ, b * Real.sin φ)

theorem majorAxisLength_of_parametric_ellipse :
  majorAxisLength 3 5 = 10 := by
  unfold majorAxisLength
  simp [max]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_majorAxisLength_of_parametric_ellipse_l21_2135
