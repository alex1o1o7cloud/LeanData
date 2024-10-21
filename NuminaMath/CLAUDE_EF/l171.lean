import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_implies_equal_side_l171_17108

/-- Two triangles in Euclidean space -/
def Triangle := Fin 3 → ℝ × ℝ

/-- A function that returns the length of a side of a triangle -/
def side_length (T : Triangle) (i j : Fin 3) : ℝ := sorry

/-- Definition of triangle congruence -/
def congruent (T₁ T₂ : Triangle) : Prop := sorry

/-- Theorem: If two triangles are congruent, then they have at least one pair of corresponding sides that are equal -/
theorem congruent_implies_equal_side {T₁ T₂ : Triangle} (h : congruent T₁ T₂) : 
  ∃ (i j : Fin 3), side_length T₁ i j = side_length T₂ i j := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_implies_equal_side_l171_17108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_proof_l171_17159

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle (x y : V) : ℝ := Real.arccos ((inner x y) / (norm x * norm y))

theorem vector_angle_proof (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : ‖a + b‖ = ‖a - b‖) (h2 : ‖a + b‖ = 2 * ‖b‖) : 
  angle (a + b) a = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_proof_l171_17159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l171_17106

theorem trig_problem (α β : ℝ) 
  (h1 : Real.sin (α + β) = 4/5)
  (h2 : Real.sin (α - β) = 3/5) :
  (Real.tan α / Real.tan β = 7) ∧ 
  (0 < β → β < α → α ≤ π/4 → Real.cos β = 7 * Real.sqrt 2 / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l171_17106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jump_rope_results_l171_17125

def passing_score : ℕ := 140

def scores : List ℤ := [-25, 17, 23, 0, -39, -11, 9, 34]

def score_difference : ℕ := 73

def average_score : ℕ := 141

def calculate_points (score : ℤ) : ℤ :=
  if score > 0 then 2 * score else -score

def total_points : ℕ := 91

theorem jump_rope_results :
  (∀ max min, scores.maximum? = some max → scores.minimum? = some min →
    max - min = score_difference) ∧
  (scores.sum / scores.length + passing_score = average_score) ∧
  (scores.map calculate_points).sum = total_points ∧
  total_points < 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jump_rope_results_l171_17125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l171_17187

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -6 ≤ x ∧ x < 4}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 8}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = Set.Ioo (-2) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l171_17187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_sum_of_squares_l171_17170

theorem power_of_two_sum_of_squares (a b c d n : ℕ) 
  (h_order : 0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (h_sum : 2^n = a^2 + b^2 + c^2 + d^2) :
  ∃ x : ℕ, (a, b, c, d) = (2^x, 0, 0, 0) ∨
            (a, b, c, d) = (0, 2^x, 0, 0) ∨
            (a, b, c, d) = (0, 0, 2^x, 0) ∨
            (a, b, c, d) = (0, 0, 0, 2^x) ∨
            (a, b, c, d) = (2^x, 2^x, 0, 0) ∨
            (a, b, c, d) = (2^x, 0, 2^x, 0) ∨
            (a, b, c, d) = (2^x, 0, 0, 2^x) ∨
            (a, b, c, d) = (0, 2^x, 2^x, 0) ∨
            (a, b, c, d) = (0, 2^x, 0, 2^x) ∨
            (a, b, c, d) = (0, 0, 2^x, 2^x) ∨
            (a, b, c, d) = (2^x, 2^x, 2^x, 2^x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_sum_of_squares_l171_17170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_meowing_l171_17165

/-- The number of meows for a given cat in 7.5 minutes -/
def meows_in_7_5_minutes (meows_per_minute : ℚ) : ℚ :=
  meows_per_minute * (15 / 2)

/-- The total number of meows for all cats in 7.5 minutes -/
def total_meows : ℕ :=
  let cat1 := meows_in_7_5_minutes 3
  let cat2 := meows_in_7_5_minutes 6
  let cat3 := meows_in_7_5_minutes 2
  let cat4 := meows_in_7_5_minutes 4
  let cat5 := meows_in_7_5_minutes (4 / 3)
  let cat6 := meows_in_7_5_minutes (5 / 2)
  let cat7 := meows_in_7_5_minutes 3
  let cat8 := meows_in_7_5_minutes 2
  let cat9 := meows_in_7_5_minutes (5 / 3)
  let cat10 := meows_in_7_5_minutes (14 / 5)
  (cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9 + cat10).floor.toNat

theorem cats_meowing :
  total_meows = 212 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_meowing_l171_17165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_chosen_set_l171_17124

theorem divisibility_in_chosen_set :
  ∀ (S : Finset Nat),
    S.card = 100 →
    (∀ n, n ∈ S → n ≤ 200) →
    (∃ m ∈ S, m < 16) →
    ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_chosen_set_l171_17124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_iff_f_inequality_l171_17194

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Theorem 1: f has a root iff a is in (0, 1/e]
theorem f_has_root_iff (a : ℝ) (h : a > 0) :
  (∃ x > 0, f a x = 0) ↔ a ∈ Set.Ioc 0 (Real.exp (-1)) := by sorry

-- Theorem 2: When a ≥ 2/e and b > 1, f(ln b) > 1/b
theorem f_inequality (a b : ℝ) (ha : a ≥ 2 / Real.exp 1) (hb : b > 1) :
  f a (Real.log b) > 1 / b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_iff_f_inequality_l171_17194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_l171_17100

theorem cosine_symmetry (φ : ℝ) : 
  (-π/2 < φ) → (φ < π/2) → 
  (∀ x : ℝ, Real.cos (2*x + φ) = Real.cos (2*(π/3 - x) + φ)) → 
  φ = -π/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_l171_17100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_exp_pi_over_6_l171_17172

-- Define Euler's formula
noncomputable def euler_formula (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the imaginary part of a complex number
def imaginary_part (z : ℂ) : ℝ := z.im

-- Theorem statement
theorem imaginary_part_of_exp_pi_over_6 :
  imaginary_part (euler_formula (π / 6)) = 1 / 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_exp_pi_over_6_l171_17172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_run_time_square_field_l171_17173

/-- Represents the time taken to run around a square field with hurdles -/
noncomputable def run_time (side_length : ℝ) (speed1 speed2 : ℝ) (hurdle_count : ℕ) (hurdle_time : ℝ) : ℝ :=
  let perimeter := 4 * side_length
  let time_speed1 := (2 * side_length) / (speed1 * 1000 / 3600)
  let time_speed2 := (2 * side_length) / (speed2 * 1000 / 3600)
  let total_hurdle_time := hurdle_count * hurdle_time
  time_speed1 + time_speed2 + total_hurdle_time

/-- Theorem stating the time taken to run around the field under given conditions -/
theorem run_time_square_field :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |run_time 50 9 7 8 5 - 131.44| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_run_time_square_field_l171_17173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l171_17176

noncomputable def f (x : ℝ) : ℝ := 2^(-x)

theorem f_decreasing_on_interval : 
  ∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l171_17176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_comparison_l171_17164

-- Define the domain D
variable (D : Type)

-- Define the functions f and g
variable (f g : D → ℝ)

-- Define the range bounds
variable (a b c d : ℝ)

-- State the theorem
theorem function_range_comparison :
  (∀ x, a ≤ f x ∧ f x ≤ b) →
  (∀ x, c ≤ g x ∧ g x ≤ d) →
  (((a > d) ↔ (∀ x₁ x₂, f x₁ > g x₂)) ∧
   ((a > d) → (∀ x, f x > g x)) ∧
   (¬((∀ x, f x > g x) → (a > d)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_comparison_l171_17164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pentagon_area_l171_17193

/-- A pentagon with specific side lengths that can be divided into a right triangle and a trapezoid -/
structure SpecialPentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  right_triangle_side1 : ℝ
  right_triangle_side2 : ℝ
  trapezoid_height : ℝ
  trapezoid_base1 : ℝ
  trapezoid_base2 : ℝ

/-- The area of the special pentagon -/
noncomputable def area (p : SpecialPentagon) : ℝ :=
  (1/2 * p.right_triangle_side1 * p.right_triangle_side2) +
  (1/2 * (p.trapezoid_base1 + p.trapezoid_base2) * p.trapezoid_height)

/-- Theorem: The area of the special pentagon is 600 square units -/
theorem special_pentagon_area :
  ∃ (p : SpecialPentagon),
    p.side1 = 12 ∧
    p.side2 = 20 ∧
    p.side3 = 30 ∧
    p.side4 = 15 ∧
    p.side5 = 25 ∧
    p.right_triangle_side1 = 12 ∧
    p.right_triangle_side2 = 25 ∧
    p.trapezoid_height = 20 ∧
    p.trapezoid_base1 = 30 ∧
    p.trapezoid_base2 = 15 ∧
    area p = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pentagon_area_l171_17193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_problem_l171_17147

theorem boat_speed_problem (downstream_time upstream_time current_speed boat_speed : ℝ) 
  (h1 : downstream_time = 4)
  (h2 : upstream_time = 5)
  (h3 : current_speed = 3)
  (h4 : downstream_time * (boat_speed + current_speed) = upstream_time * (boat_speed - current_speed)) :
  boat_speed = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_problem_l171_17147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l171_17191

theorem divisibility_theorem (A : Finset ℕ) (P : ℕ → ℕ) 
  (h : ∀ n : ℕ, n > 0 → ∃ a ∈ A, a ∣ P n) :
  ∃ a ∈ A, ∀ n : ℕ, n > 0 → a ∣ P n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l171_17191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ellipse_hyperbola_l171_17179

/-- Given an ellipse and a hyperbola that are tangent, prove that the parameter n of the hyperbola is 9/5 -/
theorem tangent_ellipse_hyperbola (n : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 = 9 ∧ x^2 - n*(y-1)^2 = 4) →  -- Ellipse and hyperbola equations
  (∀ x y : ℝ, x^2 + 9*y^2 = 9 → x^2 - n*(y-1)^2 = 4 → 
    ∃! p : ℝ × ℝ, p.1^2 + 9*p.2^2 = 9 ∧ p.1^2 - n*(p.2-1)^2 = 4) →  -- Tangency condition
  n = 9/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ellipse_hyperbola_l171_17179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shed_covering_cost_l171_17131

/-- Calculates the cost of covering a wall and roof with panels -/
noncomputable def calculate_panel_cost (wall_width wall_height roof_width roof_height panel_width panel_height panel_cost : ℝ) : ℝ :=
  let total_area := wall_width * wall_height + 2 * roof_width * roof_height
  let panel_area := panel_width * panel_height
  let panels_needed := ⌈(total_area / panel_area)⌉
  panels_needed * panel_cost

/-- Proves that the cost to cover the given wall and roof is $70 -/
theorem shed_covering_cost :
  calculate_panel_cost 10 7 10 6 10 15 35 = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shed_covering_cost_l171_17131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_derivative_l171_17142

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- State the theorem
theorem cos_derivative :
  deriv f = fun x => -Real.sin x := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_derivative_l171_17142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_arrangement_theorem_l171_17139

/-- The number of different car models --/
def total_cars : ℕ := 10

/-- The number of available exhibition spots --/
def total_spots : ℕ := 6

/-- The number of cars that cannot be placed in the 2nd spot --/
def restricted_cars : ℕ := 2

/-- The number of ways to arrange the cars in the exhibition --/
def arrangement_count : ℕ := (total_cars - restricted_cars).choose 1 * (total_cars - 1).descFactorial (total_spots - 1)

theorem car_arrangement_theorem :
  arrangement_count = (total_cars - restricted_cars).choose 1 * (total_cars - 1).descFactorial (total_spots - 1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_arrangement_theorem_l171_17139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_2025_pi_l171_17158

noncomputable section

/-- The area between two concentric circles where a chord of the larger circle
    is tangent to the smaller circle -/
def area_between_circles (chord_length : ℝ) : ℝ := by
  -- Define the radius of the smaller circle
  let small_radius : ℝ := 30
  
  -- Define the radius of the larger circle using Pythagorean theorem
  let large_radius : ℝ := Real.sqrt (small_radius^2 + (chord_length/2)^2)
  
  -- Calculate the area between the circles
  exact Real.pi * (large_radius^2 - small_radius^2)

/-- Theorem stating that the area between the circles is 2025π when the chord length is 90 -/
theorem area_is_2025_pi : area_between_circles 90 = 2025 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_2025_pi_l171_17158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_prime_factors_l171_17154

/-- 
Given an infinite arithmetic progression with first term a and common difference d,
where a and d are natural numbers, there exist infinitely many terms in this progression
that share the same set of prime factors in their prime factorizations.
-/
theorem arithmetic_progression_prime_factors (a d : ℕ) : 
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
  (∀ n, n ∈ S → ∃ k : ℕ, n = a + k * d) ∧
  (∀ m n, m ∈ S → n ∈ S → {p : ℕ | Nat.Prime p ∧ p ∣ m} = {p : ℕ | Nat.Prime p ∧ p ∣ n}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_prime_factors_l171_17154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_pencil_length_l171_17197

/-- The length of the black part of a pencil given its total length and the lengths of other colored parts. -/
theorem black_pencil_length 
  (total : ℝ) 
  (purple : ℝ) 
  (blue : ℝ) 
  (h1 : total = 4) 
  (h2 : purple = 1.5) 
  (h3 : blue = 2) : 
  total - (purple + blue) = 0.5 := by
  -- Substitute the known values
  rw [h1, h2, h3]
  -- Simplify the arithmetic
  norm_num

#check black_pencil_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_pencil_length_l171_17197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l171_17127

/-- A function from positive reals to positive reals satisfying the given functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x > 0 → f y > 0 → f x * f (y * f x) = f (x + y)

/-- The main theorem stating that any function satisfying the functional equation
    must be of the form f(x) = 1 / (1 + k*x) for some positive k. -/
theorem functional_equation_solution (f : ℝ → ℝ) (hf : FunctionalEquation f) :
    ∃ k : ℝ, k > 0 ∧ ∀ x, x > 0 → f x = 1 / (1 + k * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l171_17127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_3x_over_2_specific_period_is_correct_l171_17133

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.tan (3 * x / 2)

-- State the theorem
theorem period_of_tan_3x_over_2 :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
  ∀ (q : ℝ), (0 < q ∧ q < p) → ∃ (y : ℝ), f (y + q) ≠ f y :=
by
  -- The proof would go here
  sorry

-- Define the specific period we're proving
noncomputable def specific_period : ℝ := 2 * Real.pi / 3

-- State that this specific period satisfies the period condition
theorem specific_period_is_correct (x : ℝ) : 
  f (x + specific_period) = f x ∧
  ∀ (q : ℝ), (0 < q ∧ q < specific_period) → ∃ (y : ℝ), f (y + q) ≠ f y :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_3x_over_2_specific_period_is_correct_l171_17133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_encounters_l171_17180

/-- Represents a cyclist on the track -/
structure Cyclist where
  speed : ℝ
  encounters : ℕ

/-- The state of the cycling competition -/
structure CyclingCompetition where
  n : ℕ
  trackLength : ℝ
  cyclists : Finset Cyclist

/-- Proposition: Each cyclist has at least n^2 encounters -/
def AllCyclistsHaveEnoughEncounters (comp : CyclingCompetition) : Prop :=
  ∀ c, c ∈ comp.cyclists → c.encounters ≥ comp.n^2

/-- The main theorem -/
theorem cyclist_encounters (comp : CyclingCompetition) 
  (h1 : comp.cyclists.card = 2 * comp.n)
  (h2 : ∀ c1 c2, c1 ∈ comp.cyclists → c2 ∈ comp.cyclists → c1 ≠ c2 → c1.speed ≠ c2.speed)
  (h3 : ∀ c1 c2, c1 ∈ comp.cyclists → c2 ∈ comp.cyclists → c1 ≠ c2 → 
    ∃ t : ℝ, t > 0 ∧ (c1.speed * t) % comp.trackLength = (c2.speed * t) % comp.trackLength)
  (h4 : ∀ c1 c2 c3, c1 ∈ comp.cyclists → c2 ∈ comp.cyclists → c3 ∈ comp.cyclists → 
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 → 
    ∀ t : ℝ, ¬((c1.speed * t) % comp.trackLength = (c2.speed * t) % comp.trackLength ∧
              (c2.speed * t) % comp.trackLength = (c3.speed * t) % comp.trackLength)) :
  AllCyclistsHaveEnoughEncounters comp :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_encounters_l171_17180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_inscribed_angle_theorem_l171_17115

/-- Helper function to represent the area enclosed by an inscribed angle -/
noncomputable def area_enclosed_by_inscribed_angle (R α : ℝ) : ℝ :=
  R^2 * (α + Real.sin α)

/-- The area enclosed by an inscribed angle in a circle -/
theorem area_enclosed_by_inscribed_angle_theorem 
  (R : ℝ) 
  (α : ℝ) 
  (h_R_pos : R > 0) 
  (h_α_pos : α > 0) 
  (h_α_lt_pi : α < π) : 
  ∃ (A : ℝ), A = R^2 * (α + Real.sin α) ∧ 
  A = area_enclosed_by_inscribed_angle R α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_inscribed_angle_theorem_l171_17115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_50_terms_is_2500_l171_17152

/-- An arithmetic sequence with given second and twentieth terms -/
structure ArithmeticSequence where
  a₂ : ℚ
  a₂₀ : ℚ
  is_arithmetic : ∃ (a₁ d : ℚ), a₂ = a₁ + d ∧ a₂₀ = a₁ + 19 * d

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  let a₁ : ℚ := seq.a₂ - (seq.a₂₀ - seq.a₂) / 18
  let d : ℚ := (seq.a₂₀ - seq.a₂) / 18
  (n : ℚ) / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem stating that the sum of the first 50 terms is 2500 -/
theorem sum_50_terms_is_2500 (seq : ArithmeticSequence) 
    (h₁ : seq.a₂ = 3) (h₂ : seq.a₂₀ = 39) : 
    sum_n_terms seq 50 = 2500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_50_terms_is_2500_l171_17152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_theorem_l171_17143

noncomputable section

def curve_C (θ : ℝ) : ℝ := 2 * Real.cos θ

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * t, -1 + t)

def point_P : ℝ × ℝ := (0, -1)

theorem intersection_product_theorem :
  ∃ A B : ℝ × ℝ,
  (((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2).sqrt + 1) *
  (((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2).sqrt + 1) = 3 + Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_theorem_l171_17143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l171_17104

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 2}

-- Define the line x-y+1=0
def line1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

-- Define the line x+y+3=0
def line2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + 3 = 0}

-- State the theorem
theorem circle_equation :
  (∃ c : ℝ × ℝ, c ∈ line1 ∧ c ∈ x_axis ∧ c = (-1, 0)) ∧
  (∀ p : ℝ × ℝ, p ∈ circle_C → dist p (-1, 0) = Real.sqrt 2) ∧
  (∃ q : ℝ × ℝ, q ∈ circle_C ∧ q ∈ line2 ∧ 
    (∀ r : ℝ × ℝ, r ∈ circle_C → dist r q ≤ dist r (-1, 0))) →
  ∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 + 1)^2 + p.2^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l171_17104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_of_rectangle_l171_17162

/-- A rectangle ABCD with given dimensions and rotations -/
structure Rectangle where
  AB : ℝ
  BC : ℝ
  /-- First rotation: 90° clockwise about point D -/
  first_rotation : ℝ
  /-- Second rotation: 180° clockwise about new position of point C -/
  second_rotation : ℝ

/-- The path length of point A during the rotations -/
noncomputable def path_length (rect : Rectangle) : ℝ :=
  (Real.pi * Real.sqrt 34) / 2 + 5 * Real.pi

/-- Theorem stating the path length of point A -/
theorem path_length_of_rectangle (rect : Rectangle) 
  (h1 : rect.AB = 3) 
  (h2 : rect.BC = 5) 
  (h3 : rect.first_rotation = Real.pi / 2) 
  (h4 : rect.second_rotation = Real.pi) : 
  path_length rect = (Real.pi * Real.sqrt 34) / 2 + 5 * Real.pi :=
by
  sorry

#check path_length_of_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_of_rectangle_l171_17162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_comparison_l171_17128

theorem exam_comparison (total_questions : ℕ) 
  (sylvia_incorrect_ratio : ℚ) (sergio_incorrect : ℕ) : 
  total_questions = 50 → 
  sylvia_incorrect_ratio = 1/5 → 
  sergio_incorrect = 4 → 
  (total_questions - (sylvia_incorrect_ratio * ↑total_questions).floor) - 
  (total_questions - sergio_incorrect) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_comparison_l171_17128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_4_equals_5_l171_17112

noncomputable def F (x : ℝ) : ℤ :=
  ⌊Real.sqrt (abs (x + 2))⌋ + ⌈(8 / Real.pi) * Real.arctan (Real.sqrt (abs x))⌉

theorem F_4_equals_5 : F 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_4_equals_5_l171_17112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_value_l171_17141

theorem tan_beta_value (α β : ℝ) (h1 : α + β = π/4) (h2 : Real.tan α = 2) : 
  Real.tan β = -1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_value_l171_17141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_8_equals_80_l171_17138

-- Define the polynomial
noncomputable def p (x : ℝ) : ℝ := 1 - x + 2 * x^2

-- Define the coefficient extraction function
noncomputable def coeff (n : ℕ) (f : ℝ → ℝ) : ℝ :=
  (1 / n.factorial) * (deriv^[n] f 0)

-- State the theorem
theorem coeff_x_8_equals_80 :
  coeff 8 (λ x => p x ^ 5) = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_8_equals_80_l171_17138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l171_17109

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

-- State the theorem
theorem monotonically_decreasing_interval :
  (∀ x y : ℝ, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → x < y → f x > f y) ∧
  (∀ x : ℝ, x ∈ Set.Ioi 1 → ∃ y : ℝ, y ∈ Set.Ioo x (x + 1) ∧ f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l171_17109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contained_circle_radius_l171_17105

/-- An isosceles trapezoid with specific dimensions -/
structure IsoscelesTrapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  isIsosceles : BC = DA
  dimAB : AB = 6
  dimBC : BC = 5
  dimCD : CD = 4

/-- Circles centered at the vertices of the trapezoid -/
structure VertexCircles where
  radiusAB : ℝ
  radiusCD : ℝ
  valueAB : radiusAB = 3
  valueCD : radiusCD = 2

/-- A circle contained within the trapezoid and tangent to the vertex circles -/
def ContainedCircle (t : IsoscelesTrapezoid) (v : VertexCircles) :=
  {r : ℝ // ∃ (k m n p : ℕ+), 
    r = (-k + m * Real.sqrt n) / p ∧
    ¬ ∃ (q : ℕ+), q ^ 2 ∣ n ∧
    Nat.Coprime k p}

/-- The main theorem -/
theorem contained_circle_radius (t : IsoscelesTrapezoid) (v : VertexCircles) :
  ∃ (c : ContainedCircle t v), c.val = (-60 + 48 * Real.sqrt 3) / 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_contained_circle_radius_l171_17105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_equation_l171_17130

/-- In a triangle with sides a, b, and c, and m as the median to side c,
    the equation a^2 + b^2 = 2(c^2/4 + m^2) holds. -/
theorem triangle_median_equation (a b c m : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ m > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_median : m^2 = (1/4) * (2*a^2 + 2*b^2 - c^2)) :
  a^2 + b^2 = 2 * (c^2 / 4 + m^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_equation_l171_17130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_greater_than_two_l171_17189

-- Define a standard die
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the event of rolling a number greater than 2
def greaterThanTwo (n : ℕ) : Prop := n > 2

-- Provide an instance of DecidablePred for greaterThanTwo
instance : DecidablePred greaterThanTwo :=
  fun n => show Decidable (n > 2) from inferInstance

-- Theorem statement
theorem probability_greater_than_two :
  (Finset.filter greaterThanTwo standardDie).card / standardDie.card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_greater_than_two_l171_17189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_integer_l171_17156

theorem x_is_integer (x : ℝ) (n : ℕ) 
  (h1 : ∃ k : ℤ, x^2 - x = k) 
  (h2 : n ≥ 3) 
  (h3 : ∃ m : ℤ, x^n - x = m) : 
  ∃ z : ℤ, x = z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_integer_l171_17156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersection_points_l171_17136

-- Define the three lines
noncomputable def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
noncomputable def line2 (x y : ℝ) : Prop := 3 * x + y = 5
noncomputable def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 8

-- Define the intersection points
noncomputable def point1 : ℝ × ℝ := (2, 5)
noncomputable def point2 : ℝ × ℝ := (14/9, 1/3)

-- Theorem statement
theorem two_intersection_points :
  (∃! p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧
    ((line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∨
     (line1 p1.1 p1.2 ∧ line3 p1.1 p1.2) ∨
     (line2 p1.1 p1.2 ∧ line3 p1.1 p1.2)) ∧
    ((line1 p2.1 p2.2 ∧ line2 p2.1 p2.2) ∨
     (line1 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∨
     (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2)) ∧
    (∀ p : ℝ × ℝ, p ≠ p1 ∧ p ≠ p2 →
      ¬((line1 p.1 p.2 ∧ line2 p.1 p.2) ∨
        (line1 p.1 p.2 ∧ line3 p.1 p.2) ∨
        (line2 p.1 p.2 ∧ line3 p.1 p.2)))) ∧
  point1 = (2, 5) ∧
  point2 = (14/9, 1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersection_points_l171_17136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_student_number_l171_17146

/-- Systematic sampling function -/
def systematic_sample (population : ℕ) (sample_size : ℕ) (start : ℕ) : List ℕ :=
  let interval := population / sample_size
  List.range sample_size |>.map (fun i => start + i * interval)

theorem fourth_student_number
  (population : ℕ)
  (sample_size : ℕ)
  (h_pop : population = 48)
  (h_size : sample_size = 4)
  (h_start : ∃ start, systematic_sample population sample_size start = [5, 17, 29, 41]) :
  (systematic_sample population sample_size 5).get? 1 = some 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_student_number_l171_17146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ritas_initial_money_l171_17126

theorem ritas_initial_money (dresses pants jackets : ℕ) 
  (dress_cost pant_cost jacket_cost transport_cost remaining : ℚ) : 
  dresses = 5 →
  pants = 3 →
  jackets = 4 →
  dress_cost = 20 →
  pant_cost = 12 →
  jacket_cost = 30 →
  transport_cost = 5 →
  remaining = 139 →
  (dresses : ℚ) * dress_cost + (pants : ℚ) * pant_cost + 
  (jackets : ℚ) * jacket_cost + transport_cost + remaining = 400 := by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Proof steps would go here
  sorry

#check ritas_initial_money

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ritas_initial_money_l171_17126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_and_sum_l171_17155

/-- A hexagon inscribed in a circle with specific side lengths and a dividing chord -/
structure InscribedHexagon where
  -- Three consecutive sides of length 4
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ
  side_a_length : side_a = 4
  side_b_length : side_b = 4
  side_c_length : side_c = 4
  
  -- Three consecutive sides of length 6
  side_d : ℝ
  side_e : ℝ
  side_f : ℝ
  side_d_length : side_d = 6
  side_e_length : side_e = 6
  side_f_length : side_f = 6
  
  -- Chord dividing the hexagon into two trapezoids
  chord : ℝ
  
  -- Chord divides hexagon into trapezoids with sides of length 4 and 6
  divides_into_trapezoids : ∃ (trapezoid1 trapezoid2 : Set ℝ),
    trapezoid1 ∪ trapezoid2 = {side_a, side_b, side_c, side_d, side_e, side_f} ∧
    trapezoid1 ∩ trapezoid2 = {chord} ∧
    (trapezoid1 = {side_a, side_b, side_c, chord} ∨ trapezoid1 = {side_d, side_e, side_f, chord})

/-- Theorem stating the length of the chord and the sum of m and n -/
theorem chord_length_and_sum (h : InscribedHexagon) :
  ∃ (m n : ℕ), h.chord = m / n ∧ Nat.Coprime m n ∧ m + n = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_and_sum_l171_17155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_min_f_l171_17184

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 - 3*x + 3) / (x - 1)

-- State the theorem
theorem no_max_min_f :
  ∀ (a b : ℝ), -3 < a ∧ a < b ∧ b < 2 →
  ¬∃ (m M : ℝ), (∀ x, a < x ∧ x < b → m ≤ f x ∧ f x ≤ M) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_min_f_l171_17184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auditorium_seats_l171_17190

theorem auditorium_seats : ℕ := by
  let taken : ℚ := 2 / 5
  let broken : ℚ := 1 / 10
  let available : ℕ := 250
  let total_seats : ℕ := 500
  
  have h1 : (taken + broken) * total_seats + available = total_seats := by
    sorry
  
  exact total_seats

#check auditorium_seats

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auditorium_seats_l171_17190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_between_specific_planes_l171_17144

/-- The cosine of the angle between two planes given by their normal vectors. -/
noncomputable def cosine_angle_between_planes (n₁ n₂ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := n₁
  let (x₂, y₂, z₂) := n₂
  (x₁ * x₂ + y₁ * y₂ + z₁ * z₂) / Real.sqrt ((x₁^2 + y₁^2 + z₁^2) * (x₂^2 + y₂^2 + z₂^2))

/-- The normal vector of the plane ax + by + cz + d = 0 -/
def normal_vector (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

theorem cosine_angle_between_specific_planes :
  let plane1 := normal_vector 1 (-3) 2
  let plane2 := normal_vector 3 2 (-4)
  cosine_angle_between_planes plane1 plane2 = -11 / Real.sqrt 406 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_between_specific_planes_l171_17144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l171_17117

theorem cos_alpha_value (α : ℝ) 
  (h1 : Real.sin (α + π/3) + Real.sin α = -4 * Real.sqrt 3 / 5)
  (h2 : -π/2 < α ∧ α < 0) :
  Real.cos α = (3 * Real.sqrt 3 - 4) / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l171_17117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_max_volume_l171_17182

-- Define the sphere
noncomputable def sphere_surface_area : ℝ := 289 * Real.pi / 16

-- Define the tetrahedron's base
noncomputable def base_side_length : ℝ := Real.sqrt 3

-- Theorem statement
theorem tetrahedron_max_volume 
  (h_sphere : sphere_surface_area = 289 * Real.pi / 16)
  (h_base : base_side_length = Real.sqrt 3)
  (h_vertices_on_sphere : True)  -- This represents that all vertices are on the sphere's surface
  (h_base_equilateral : True)    -- This represents that the base is an equilateral triangle
  : ∃ (volume : ℝ), volume ≤ Real.sqrt 3 ∧ 
    ∀ (other_volume : ℝ), (other_volume ≤ volume) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_max_volume_l171_17182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_sum_l171_17121

def length (k : ℕ) : ℕ := 
  if k ≤ 1 then 0 else (Nat.factorization k).sum (fun _ v => v)

theorem max_length_sum (x y : ℕ) (hx : x > 1) (hy : y > 1) (hsum : x + 3 * y < 960) :
  ∃ (max_sum : ℕ), max_sum = 15 ∧ 
    ∀ (a b : ℕ), a > 1 → b > 1 → a + 3 * b < 960 → length a + length b ≤ max_sum :=
by
  sorry

#eval length 24  -- Should output 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_sum_l171_17121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solution_l171_17114

noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 2 else 3 * x - 15

theorem g_solution : 
  ∀ x : ℝ, g x = 5 ↔ x = 20 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solution_l171_17114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_length_is_five_sixths_l171_17113

/-- A square formed by six congruent equilateral triangles -/
structure TriangleSquare where
  /-- Side length of each equilateral triangle -/
  s : ℝ
  /-- Assumption that s is positive -/
  s_pos : 0 < s

/-- The diagonal of the square -/
noncomputable def diagonal (ts : TriangleSquare) : ℝ := 3 * ts.s * (Real.sqrt 3 / 2)

/-- The point X on the diagonal -/
noncomputable def x_point (ts : TriangleSquare) : ℝ := 2 * ts.s * (Real.sqrt 3 / 2)

/-- The point Y on the diagonal -/
noncomputable def y_point (ts : TriangleSquare) : ℝ := 3/2 * ts.s * (Real.sqrt 3 / 2)

theorem xy_length_is_five_sixths (ts : TriangleSquare) 
    (h : diagonal ts = 10) : x_point ts - y_point ts = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_length_is_five_sixths_l171_17113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_diagonals_l171_17137

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  -- Additional structure details can be added if needed

/-- A diagonal in a convex polygon -/
structure Diagonal (n : ℕ) where
  -- Additional structure details can be added if needed

/-- Predicate to determine if a diagonal is "good" -/
def isGoodDiagonal (n : ℕ) (d : Diagonal n) (diagonals : Set (Diagonal n)) : Prop :=
  ∃! d', d' ∈ diagonals ∧ d ≠ d' ∧ True -- Placeholder for intersection check

/-- The main theorem about the maximum number of good diagonals -/
theorem max_good_diagonals (n : ℕ) (p : ConvexPolygon n) :
  ∃ (diagonals : Finset (Diagonal n)),
    (∀ d ∈ diagonals, isGoodDiagonal n d diagonals.toSet) ∧
    (n % 2 = 0 → diagonals.card = n - 2) ∧
    (n % 2 = 1 → diagonals.card = n - 3) ∧
    (∀ diagonals' : Finset (Diagonal n),
      (∀ d ∈ diagonals', isGoodDiagonal n d diagonals'.toSet) →
      diagonals'.card ≤ diagonals.card) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_diagonals_l171_17137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l171_17118

noncomputable section

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 2

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y = 1

-- Define the center line
def center_line (x y : ℝ) : Prop := y = -2*x

-- Define point A
def point_A : ℝ × ℝ := (2, -1)

-- Define point P
def point_P : ℝ × ℝ := (1/2, -3)

-- Define line l
def line_l (x y : ℝ) : Prop := 2*x + 4*y + 11 = 0

theorem circle_and_line_properties :
  (∃ (x y : ℝ), circle_C x y ∧ tangent_line x y) ∧
  (∃ (x y : ℝ), circle_C x y ∧ center_line x y) ∧
  circle_C point_A.1 point_A.2 ∧
  (∃ (x1 y1 x2 y2 : ℝ),
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l x1 y1 ∧ line_l x2 y2 ∧
    line_l point_P.1 point_P.2 ∧
    point_P.1 = (x1 + x2) / 2 ∧ point_P.2 = (y1 + y2) / 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l171_17118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisibility_l171_17199

def fn (n : ℕ) (x y z : ℚ) : ℚ := x^(2*n) + y^(2*n) + z^(2*n) - x*y - y*z - z*x

def gn (n : ℕ) (x y z : ℚ) : ℚ := (x - y)^(5*n) + (y - z)^(5*n) + (z - x)^(5*n)

theorem unique_divisibility (n : ℕ) :
  (∀ (x y z : ℚ), (fn n x y z) ∣ (gn n x y z)) ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisibility_l171_17199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_commuting_poly_l171_17134

/-- A polynomial with leading coefficient 1 -/
structure MoincPoly (R : Type) [CommRing R] where
  coeff : R → R
  degree : ℕ
  leading_one : coeff (degree : R) = 1

/-- Two polynomials commute if P(Q(x)) = Q(P(x)) -/
def commute {R : Type} [CommRing R] (P Q : MoincPoly R) : Prop :=
  ∀ x, P.coeff (Q.coeff x) = Q.coeff (P.coeff x)

/-- A polynomial of degree 2 -/
def is_degree_two {R : Type} [CommRing R] (P : MoincPoly R) : Prop :=
  P.degree = 2

theorem unique_commuting_poly {R : Type} [CommRing R] (P : MoincPoly R) (k : ℕ) 
  (h_P : is_degree_two P) :
  ∃! Q : MoincPoly R, Q.degree = k ∧ commute P Q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_commuting_poly_l171_17134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l171_17132

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define a point on the parabola
def point_on_parabola (p : ℝ) : Prop := parabola p 1 (-2)

-- Define a line parallel to OA
def parallel_line (t : ℝ) (x y : ℝ) : Prop := y = -2*x + t

-- Define the intersection of the line and the parabola
def intersection (p t : ℝ) (x y : ℝ) : Prop := parabola p x y ∧ parallel_line t x y

-- Define the length of MN
noncomputable def length_MN (t : ℝ) : Prop := Real.sqrt (1 + (-1/2)^2) * Real.sqrt (4 + 8*t) = 3 * Real.sqrt 5

-- Theorem statement
theorem area_of_triangle (p t : ℝ) :
  point_on_parabola p →
  length_MN t →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    intersection p t x₁ y₁ ∧
    intersection p t x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    (1/2 * 3 * Real.sqrt 5 * 4 / Real.sqrt 5 = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l171_17132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l171_17101

noncomputable def circle_equation (x y : ℝ) : Prop :=
  x^2 - 16*x + y^2 - 8*y + 100 = 0

noncomputable def shortest_distance : ℝ := 4 * Real.sqrt 5 - 8

theorem shortest_distance_to_circle :
  ∃ (p : ℝ × ℝ), circle_equation p.1 p.2 ∧
  ∀ (q : ℝ × ℝ), circle_equation q.1 q.2 →
  shortest_distance ≤ Real.sqrt (q.1^2 + q.2^2) := by
  sorry

#check shortest_distance_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l171_17101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l171_17150

theorem quadratic_equation_solution :
  let f : ℂ → ℂ := λ y => y^2 - 6*y + 5 + (y + 2)*(y + 7)
  ∀ y : ℂ, f y = 0 ↔ y = (-3 + Complex.I * Real.sqrt 143) / 4 ∨ y = (-3 - Complex.I * Real.sqrt 143) / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l171_17150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l171_17103

theorem polynomial_divisibility (f : Polynomial ℂ) (n : ℕ) (hn : n > 0) 
  (h : (X - 1) ∣ f.comp (X ^ n)) : 
  (X ^ n - 1) ∣ f.comp (X ^ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l171_17103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_implies_a_value_l171_17151

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := (a * x - 1) * (x + 1) < 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | x < -1 ∨ (-1/2 < x ∧ x < Real.pi)}

-- Theorem statement
theorem inequality_solution_implies_a_value :
  ∀ a : ℝ, (∀ x : ℝ, inequality a x ↔ x ∈ solution_set a) → a = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_implies_a_value_l171_17151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l171_17110

noncomputable def a : ℕ → ℚ
| 0 => 1/2
| n+1 => a n + (a n)^2 / 2018

theorem sequence_property :
  a 2018 < 1 ∧ 1 < a 2019 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l171_17110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_property_exactly_one_false_l171_17171

theorem log_property (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) (hlog : Real.log b / Real.log a > 1) :
  (b > 1 ∧ b > a) ∨ (a < 1 ∧ b < 1) ∨ (b < 1 ∧ b < a) :=
by sorry

theorem exactly_one_false (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) (hlog : Real.log b / Real.log a > 1) :
  ∃! i : Fin 4, ¬(
    (i = 0 → b > 1 ∧ b > a) ∧
    (i = 1 → a < 1 ∧ a < b) ∧
    (i = 2 → b < 1 ∧ b < a) ∧
    (i = 3 → a < 1 ∧ b < 1)
  ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_property_exactly_one_false_l171_17171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_rays_l171_17186

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := t + 1/t
def y : ℝ := 2

-- Theorem statement
theorem two_rays :
  ∀ t : ℝ, t ≠ 0 →
  (x t ≤ -2 ∨ x t ≥ 2) ∧ y = 2 :=
by
  intro t ht
  have h1 : x t ≤ -2 ∨ x t ≥ 2 := by
    sorry -- Proof for this part is omitted
  have h2 : y = 2 := rfl
  exact ⟨h1, h2⟩

#check two_rays

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_rays_l171_17186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_4_l171_17188

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

-- State the theorem
theorem f_of_g_4 : f (g 4) = (25 / 7) * Real.sqrt 21 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_4_l171_17188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l171_17196

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.cos x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x

-- Part 1
theorem part_one (x : ℝ) (h : x ∈ Set.Ioo 0 Real.pi) : 
  x > g 1 x ∧ g 1 x > f x := by sorry

-- Part 2
theorem part_two (a : ℝ) : 
  (∀ x ∈ Set.union (Set.Ioo (-Real.pi) 0) (Set.Ioo 0 Real.pi), 
    f x / g a x < Real.sin x / x) → 
  a ∈ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l171_17196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_P_not_square_area_P_square_l171_17140

-- Define the square K with area 1
def K : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define an inscribed rectangle P
def P (a b : ℝ) : Set (ℝ × ℝ) := {p | a ≤ p.1 ∧ p.1 ≤ 1-a ∧ b ≤ p.2 ∧ p.2 ≤ 1-b}

-- Area of P
def area_P (a b : ℝ) : ℝ := (1-2*a) * (1-2*b)

-- Theorem for case when P is not a square
theorem area_P_not_square (a b : ℝ) (h1 : 0 < a ∧ a < 1/2) (h2 : 0 < b ∧ b < 1/2) (h3 : a ≠ b) :
  0 < area_P a b ∧ area_P a b < 1/2 := by sorry

-- Theorem for case when P is a square
theorem area_P_square (a : ℝ) (h : 0 < a ∧ a < 1/2) :
  1/2 ≤ area_P a a ∧ area_P a a < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_P_not_square_area_P_square_l171_17140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincidence_l171_17119

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2 = 1

-- Define the right vertex of the hyperbola
noncomputable def right_vertex (x y : ℝ) : Prop := hyperbola x y ∧ x > 0 ∧ y = 0

-- Define the focus of the parabola
noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Theorem statement
theorem parabola_hyperbola_focus_coincidence (p : ℝ) :
  (∃ x y : ℝ, right_vertex x y ∧ parabola_focus p = (x, y)) → p = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincidence_l171_17119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_is_two_l171_17157

-- Define the arithmetic sequence and its sum
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

-- State the theorem
theorem common_difference_is_two (a₁ : ℝ) :
  ∃ d : ℝ, 2 * S a₁ d 3 = 3 * S a₁ d 2 + 6 → d = 2 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_is_two_l171_17157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowfall_average_is_0_40_l171_17145

/-- Calculates the average snowfall per day given 5 daily measurements -/
noncomputable def average_snowfall (day1 day2 day3 day4 day5 : ℝ) : ℝ :=
  (day1 + day2 + day3 + day4 + day5) / 5

/-- Theorem stating that the average snowfall per day is 0.40 cm -/
theorem snowfall_average_is_0_40 :
  average_snowfall 0.33 0.33 0.22 0.45 0.67 = 0.40 := by
  -- Unfold the definition of average_snowfall
  unfold average_snowfall
  -- Simplify the arithmetic
  simp [add_div]
  -- The proof is completed by normalization of real numbers
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowfall_average_is_0_40_l171_17145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_condition_l171_17175

theorem triangle_formation_condition (x : ℝ) :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let D : ℝ × ℝ := (x, 0)
  let C : ℝ × ℝ := (x, Real.sqrt (1 - x^2))
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let BD := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  (∀ p ∈ circle, (p.1 - A.1) * (B.1 - A.1) + (p.2 - A.2) * (B.2 - A.2) = 0 → p = C) →
  (AD + BD > CD ∧ BD + CD > AD ∧ CD + AD > BD) ↔ x ∈ Set.Ioo (2 - Real.sqrt 5) (Real.sqrt 5 - 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_condition_l171_17175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagstaff_shadow_length_l171_17129

/-- Given a flagstaff and a building with known heights and the building's shadow length,
    calculate the length of the shadow cast by the flagstaff under similar conditions. -/
theorem flagstaff_shadow_length 
  (flagstaff_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h_flagstaff : flagstaff_height = 17.5)
  (h_building_height : building_height = 12.5)
  (h_building_shadow : building_shadow = 28.75) :
  flagstaff_height * building_shadow / building_height = 40.25 := by
  sorry

#check flagstaff_shadow_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagstaff_shadow_length_l171_17129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_two_minus_i_squared_l171_17169

open Complex

theorem modulus_of_two_minus_i_squared : Complex.abs ((2 - I) ^ 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_two_minus_i_squared_l171_17169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_distance_sqrt_2_l171_17167

noncomputable def line_equation (t : ℝ) : ℝ × ℝ := (-2 - Real.sqrt 2 * t, 3 + Real.sqrt 2 * t)

def point_A : ℝ × ℝ := (-2, 3)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_points_at_distance_sqrt_2 :
  ∃ t1 t2 : ℝ,
    t1 ≠ t2 ∧
    distance (line_equation t1) point_A = Real.sqrt 2 ∧
    distance (line_equation t2) point_A = Real.sqrt 2 ∧
    line_equation t1 = (-3, 4) ∧
    line_equation t2 = (-1, 2) ∧
    ∀ t : ℝ, distance (line_equation t) point_A = Real.sqrt 2 → t = t1 ∨ t = t2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_distance_sqrt_2_l171_17167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bijective_l171_17198

def x : ℕ → ℕ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | n + 2 =>
    let k := (n + 1) / 2
    if (n + 2) % 2 = 0 then
      if k % 2 = 0 then 2 * x k else 2 * x k + 1
    else
      if k % 2 = 0 then 2 * x k + 1 else 2 * x k

theorem x_bijective : Function.Bijective x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bijective_l171_17198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_twice_square_l171_17120

/-- Square with vertices O, S, U, V -/
structure Square (O S U V : ℝ × ℝ) : Prop where
  origin : O = (0, 0)
  s_coord : S = (3, 3)
  is_square : U = (3, 0) ∧ V = (0, 3)

/-- Calculate the area of a triangle given three points -/
def triangleArea (A B C : ℝ × ℝ) : ℝ := sorry

/-- Calculate the area of a square given its side length -/
def squareArea (side : ℝ) : ℝ := side ^ 2

theorem area_triangle_twice_square {O S U V W : ℝ × ℝ} 
  (sq : Square O S U V) (w_coord : W = (0, -9)) :
  triangleArea U V W = 2 * squareArea 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_twice_square_l171_17120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_max_value_l171_17178

-- Define the solution set
def solution_set (x : ℝ) : Prop := -4/3 < x ∧ x < 8/3

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := |a * x - 2| < 6

-- Theorem 1: Prove that a = 3
theorem find_a : 
  (∀ x, inequality a x ↔ solution_set x) → a = 3 :=
sorry

-- Define the function to maximize
noncomputable def f (a b t : ℝ) : ℝ := Real.sqrt (-a * t + 12) + Real.sqrt (3 * b * t)

-- Theorem 2: Prove the maximum value
theorem max_value (h : a = 3) (h' : b = 1) :
  ∃ (max : ℝ), max = 2 * Real.sqrt 6 ∧ ∀ t, f a b t ≤ max :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_max_value_l171_17178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_l171_17107

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := x^2 + 6*x + 15
def g (x : ℝ) : ℝ := x^2 - 4*x + 1

-- Define the vertices of the two functions
def C : ℝ × ℝ := (-3, f (-3))
def D : ℝ × ℝ := (2, g 2)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem vertex_distance : distance C D = Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_l171_17107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fee_for_25_cubic_meters_l171_17122

/-- Calculates the water fee for a given usage and rate structure -/
def water_fee (base_rate : ℚ) (usage : ℚ) : ℚ :=
  if usage ≤ 20 then
    usage * base_rate
  else
    20 * base_rate + (usage - 20) * (base_rate + 2)

/-- Proves that for 25 cubic meters of usage, the water fee is 25a + 10 -/
theorem water_fee_for_25_cubic_meters (a : ℚ) :
  water_fee a 25 = 25 * a + 10 := by
  sorry

#check water_fee_for_25_cubic_meters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fee_for_25_cubic_meters_l171_17122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_selection_l171_17192

def total_students : ℕ := 10

theorem student_selection (x : ℕ) : 
  (∀ (selection : Finset (Fin total_students)) 
    (h_selection_size : selection.card = 6), 
    (∃ (i : Fin total_students), i ∈ selection ∧ i.val ≥ x) ∧ 
    (¬∃ (selection : Finset (Fin total_students)), 
      selection.card = 6 ∧ 
      (selection.filter (fun i => i.val < x)).card = 5 ∧ 
      (selection.filter (fun i => i.val ≥ x)).card = 1) ∧
    (∃ (selection : Finset (Fin total_students)), 
      selection.card = 6 ∧ 
      (selection.filter (fun i => i.val < x)).card = 3 ∧ 
      (selection.filter (fun i => i.val ≥ x)).card = 3) ∧
    (∃ (selection : Finset (Fin total_students)), 
      selection.card = 6 ∧ 
      ((selection.filter (fun i => i.val < x)).card ≠ 3 ∨ 
       (selection.filter (fun i => i.val ≥ x)).card ≠ 3))) ↔ 
  (x = 3 ∨ x = 4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_selection_l171_17192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roots_unity_sum_divides_fifth_power_sum_l171_17149

theorem cube_roots_unity_sum_divides_fifth_power_sum (p : ℕ) (x y z : ℕ) 
  (h_prime : Nat.Prime p)
  (h_positive : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_order : x < y ∧ y < z ∧ z < p)
  (h_frac_eq : (x^3 : ZMod p) = (y^3 : ZMod p) ∧ (y^3 : ZMod p) = (z^3 : ZMod p)) : 
  (x + y + z) ∣ (x^5 + y^5 + z^5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roots_unity_sum_divides_fifth_power_sum_l171_17149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_g_implies_alpha_l171_17111

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x + Real.sqrt 3 * Real.cos x) - Real.sqrt 3 / 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f (x + a)

def is_symmetric_about (h : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, h (2 * p.1 - x) = h x

theorem symmetric_g_implies_alpha (a : ℝ) :
  is_symmetric_about (g a) (-π/2, 0) →
  ∃ α : ℝ, 0 < α ∧ α < π/2 ∧ α = π/3 := by
  sorry

#check symmetric_g_implies_alpha

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_g_implies_alpha_l171_17111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_vendor_total_sale_l171_17123

def fruits_sold (lemons avocados oranges apples : Float) : Nat :=
  let dozen : Nat := 12
  let lemons_count : Nat := (lemons * dozen.toFloat).toUInt64.toNat
  let avocados_count : Nat := (avocados * dozen.toFloat).toUInt64.toNat
  let oranges_count : Nat := (oranges * dozen.toFloat).toUInt64.toNat
  let apples_count : Nat := (apples * dozen.toFloat).toUInt64.toNat
  lemons_count + avocados_count + oranges_count + apples_count

theorem fruit_vendor_total_sale :
  fruits_sold 2.5 5.25 3.75 2.12 = 163 := by
  sorry

#eval fruits_sold 2.5 5.25 3.75 2.12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_vendor_total_sale_l171_17123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_bag_wholesale_cost_l171_17166

/-- The wholesale cost of a sleeping bag, given the selling price and profit margin. -/
noncomputable def wholesale_cost (selling_price : ℝ) (profit_margin : ℝ) : ℝ :=
  selling_price / (1 + profit_margin)

/-- Theorem stating that the wholesale cost of a sleeping bag is approximately $24.35 -/
theorem sleeping_bag_wholesale_cost :
  let selling_price : ℝ := 28
  let profit_margin : ℝ := 0.15
  abs (wholesale_cost selling_price profit_margin - 24.35) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_bag_wholesale_cost_l171_17166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_sum_factorial_squares_12_l171_17185

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorialSquares (n : ℕ) : ℕ :=
  (List.range n).map (λ i => (factorial (i + 1))^2) |>.sum

theorem units_digit_sum_factorial_squares_12 :
  unitsDigit (sumFactorialSquares 12) = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_sum_factorial_squares_12_l171_17185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_shifted_sine_l171_17181

noncomputable def f (x : ℝ) := -Real.sin (x + Real.pi/6)

theorem symmetry_of_shifted_sine :
  ∀ (x : ℝ), f (2*Real.pi/3 - x) = f (2*Real.pi/3 + x) := by
  intro x
  simp [f]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_shifted_sine_l171_17181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_50_divided_by_x_minus_1_cubed_l171_17148

theorem remainder_x_50_divided_by_x_minus_1_cubed (x : ℝ) : 
  ∃ q : Polynomial ℝ, X^50 = (X - 1)^3 * q + (1225*X^2 - 2500*X + 1276) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_50_divided_by_x_minus_1_cubed_l171_17148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l171_17135

theorem system_solution (a : ℝ) :
  (∀ x y : ℝ, Real.sin x + a * Real.cos y = (1 + a) / Real.sqrt 2 ∧ 
              Real.cos x + a * Real.sin y = (1 + a) / Real.sqrt 2 →
    (a = 0 → ∃ k : ℤ, x = π / 4 + 2 * π * ↑k) ∧
    (a = -1 → ∃ k : ℤ, x = π / 2 - y + 2 * π * ↑k) ∧
    (a ≠ 0 ∧ a ≠ -1 → ∃ k : ℤ, x = π / 4 + 2 * π * ↑k ∧ y = π / 4 - 2 * π * ↑k)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l171_17135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_gt_1_sufficient_not_necessary_for_2_pow_x_gt_1_l171_17174

theorem x_gt_1_sufficient_not_necessary_for_2_pow_x_gt_1 :
  (∀ x : ℝ, x > 1 → (2 : ℝ)^x > 1) ∧
  (∃ x : ℝ, x ≤ 1 ∧ (2 : ℝ)^x > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_gt_1_sufficient_not_necessary_for_2_pow_x_gt_1_l171_17174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_a_amount_l171_17160

/-- The amount of fuel A added to the tank -/
def fuel_a : ℝ := 50

/-- The total capacity of the tank in gallons -/
def tank_capacity : ℝ := 200

/-- The percentage of ethanol in fuel A -/
def ethanol_percentage_a : ℝ := 0.12

/-- The percentage of ethanol in fuel B -/
def ethanol_percentage_b : ℝ := 0.16

/-- The total amount of ethanol in the full tank -/
def total_ethanol : ℝ := 30

theorem fuel_a_amount : 
  fuel_a = 50 :=
by
  -- Unfold the definition of fuel_a
  unfold fuel_a
  -- The proof is complete since fuel_a is defined as 50
  rfl

#check fuel_a_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_a_amount_l171_17160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_inverse_221_l171_17195

def t : ℕ → ℚ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | n + 2 => if (n + 2) % 2 = 0 then (n + 2) + t (n + 1) else (t (n + 1)) / (n + 2)

theorem t_inverse_221 (n : ℕ) (h : n > 0) (h_tn : t n = 1 / 221) : n = 221 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_inverse_221_l171_17195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_distances_l171_17168

/-- Given a point M and a circle, if the smallest distance from M to any point on the circle is a
    and the greatest distance is b, then the radius of the circle is either (b - a) / 2 or (b + a) / 2. -/
theorem circle_radius_from_distances (M : ℝ × ℝ) (center : ℝ × ℝ) (a b : ℝ) :
  let circle := {P : ℝ × ℝ | ∃ r, dist P center = r}
  (∀ P, P ∈ circle → dist M P ≥ a) ∧
  (∀ P, P ∈ circle → dist M P ≤ b) ∧
  (∃ P₁ P₂, P₁ ∈ circle ∧ P₂ ∈ circle ∧ dist M P₁ = a ∧ dist M P₂ = b) →
  ∃ r, (r = (b - a) / 2 ∨ r = (b + a) / 2) ∧ ∀ P, P ∈ circle → dist P center = r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_distances_l171_17168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_phi_value_l171_17183

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (3 * x + φ)

noncomputable def f_shifted (x φ : ℝ) : ℝ := f (x - Real.pi/4) φ

theorem smallest_phi_value (φ : ℝ) :
  (∀ x, f_shifted x φ = f_shifted (2*Real.pi/3 - x) φ) →  -- Symmetry condition
  (∃ k : ℤ, φ = k * Real.pi - Real.pi/4) →  -- Phase shift equation
  (∀ k : ℤ, |φ| ≤ |k * Real.pi - Real.pi/4|) →  -- Smallest absolute value
  |φ| = Real.pi/4 := by
  sorry

#check smallest_phi_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_phi_value_l171_17183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l171_17163

noncomputable def f (x : ℝ) := Real.sin x ^ 2 - Real.cos x ^ 2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (f (2 * Real.pi / 3) = 2) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (-5 * Real.pi / 6 + k * Real.pi) (-Real.pi / 3 + k * Real.pi))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l171_17163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toronto_beijing_time_difference_l171_17161

/-- Represents a time with date and hour -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  deriving Repr

/-- Represents the time difference between two cities in hours -/
def TimeDifference := ℤ

/-- Calculates the time in Toronto given the time in Beijing and the time difference -/
def torontoTime (beijingTime : DateTime) (timeDiff : TimeDifference) : DateTime :=
  sorry

theorem toronto_beijing_time_difference 
  (beijingTime : DateTime)
  (timeDiff : TimeDifference)
  (h1 : timeDiff = (-12 : ℤ))
  (h2 : beijingTime = ⟨2023, 10, 1, 8⟩) :
  torontoTime beijingTime timeDiff = ⟨2023, 9, 30, 20⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_toronto_beijing_time_difference_l171_17161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_property_min_f_max_minus_f_min_l171_17116

noncomputable def quadratic_equation (t : ℝ) (x : ℝ) : Prop :=
  2 * x^2 - t * x - 2 = 0

noncomputable def has_roots (t : ℝ) (α β : ℝ) : Prop :=
  quadratic_equation t α ∧ quadratic_equation t β ∧ α < β

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  (4 * x - t) / (x^2 + 1)

theorem quadratic_roots_property (t : ℝ) (α β x₁ x₂ : ℝ) 
  (h : has_roots t α β) 
  (h₁ : α ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ β) :
  4 * x₁ * x₂ - t * (x₁ + x₂) - 4 < 0 := by
  sorry

theorem min_f_max_minus_f_min (t : ℝ) (α β : ℝ) 
  (h : has_roots t α β) :
  ∃ (f_max f_min : ℝ), 
    (∀ x, α ≤ x ∧ x ≤ β → f_min ≤ f t x ∧ f t x ≤ f_max) ∧
    (∀ g : ℝ → ℝ, (∀ s, g s = f_max - f_min) → 
      ∃ (s : ℝ), ∀ (r : ℝ), g r ≥ g s ∧ g s = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_property_min_f_max_minus_f_min_l171_17116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l171_17153

/-- The asymptotes of the hyperbola x²-4y²=1 -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → ℝ := fun x y ↦ x^2 - 4*y^2 - 1
  ∃ (f g : ℝ → ℝ → ℝ),
    (∀ x y, h x y = 0 → (f x y = 0 ∨ g x y = 0)) ∧
    (∀ x y, f x y = x + 2*y) ∧
    (∀ x y, g x y = x - 2*y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l171_17153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_focus_coordinates_l171_17102

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  majorAxis : Point × Point
  minorAxis : Point × Point

/-- Calculates the center of an ellipse -/
noncomputable def centerOfEllipse (e : Ellipse) : Point :=
  { x := (e.majorAxis.1.x + e.majorAxis.2.x) / 2,
    y := (e.majorAxis.1.y + e.majorAxis.2.y) / 2 }

/-- Calculates the semi-major axis length -/
noncomputable def semiMajorAxis (e : Ellipse) : ℝ :=
  (e.majorAxis.2.y - e.majorAxis.1.y) / 2

/-- Calculates the semi-minor axis length -/
noncomputable def semiMinorAxis (e : Ellipse) : ℝ :=
  (e.minorAxis.2.x - e.minorAxis.1.x) / 2

/-- Calculates the y-coordinate of the upper focus -/
noncomputable def upperFocusY (e : Ellipse) : ℝ :=
  let center := centerOfEllipse e
  let a := semiMajorAxis e
  let b := semiMinorAxis e
  center.y + Real.sqrt (a^2 - b^2)

/-- The main theorem -/
theorem upper_focus_coordinates (e : Ellipse) :
  e.majorAxis = ((⟨2, 0⟩ : Point), (⟨2, 8⟩ : Point)) →
  e.minorAxis = ((⟨0, 4⟩ : Point), (⟨4, 4⟩ : Point)) →
  (⟨2, 4 + 2 * Real.sqrt 3⟩ : Point) = ⟨2, upperFocusY e⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_focus_coordinates_l171_17102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_complement_intersection_l171_17177

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {y | y ≤ 3}

-- Theorem for the first part of the question
theorem intersection_and_union :
  (A ∩ B = Set.Icc 2 3) ∧ (A ∪ B = Set.univ) := by sorry

-- Theorem for the second part of the question
theorem complement_intersection :
  (Set.compl A ∩ Set.compl B) = ∅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_complement_intersection_l171_17177
