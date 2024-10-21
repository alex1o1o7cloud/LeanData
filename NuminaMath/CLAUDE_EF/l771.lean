import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nines_in_cube_l771_77194

/-- Helper function to count the number of '9's in the decimal representation of a natural number -/
def number_of_nines_in_decimal (n : ℕ) : ℕ :=
  sorry

/-- The number of '9's in the decimal representation of (10^2007 - 1)^3 is 4015 -/
theorem count_nines_in_cube : ∃ (n : ℕ), n = 10^2007 - 1 ∧ (number_of_nines_in_decimal (n^3) = 4015) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nines_in_cube_l771_77194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rate_is_11_l771_77151

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  withStream : ℚ
  againstStream : ℚ

/-- Calculates the man's rate in still water given his speeds with and against the stream -/
def manRate (speed : RowingSpeed) : ℚ :=
  (speed.withStream + speed.againstStream) / 2

/-- Theorem stating that given the specific speeds, the man's rate in still water is 11 km/h -/
theorem man_rate_is_11 (speed : RowingSpeed) 
  (h1 : speed.withStream = 16)
  (h2 : speed.againstStream = 6) : 
  manRate speed = 11 := by
  unfold manRate
  rw [h1, h2]
  norm_num

/-- Example calculation -/
def example_speed : RowingSpeed := { withStream := 16, againstStream := 6 }

#eval manRate example_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rate_is_11_l771_77151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l771_77118

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

-- Define the system of equations
def system (x a : ℝ) : Prop :=
  (2 * x - (floor x : ℝ) = 4 * a + 1) ∧
  (4 * (floor x : ℝ) - 3 * frac x = 5 * a + 15)

theorem system_solution :
  ∀ a : ℝ, (∃ x : ℝ, system x a) ↔ (a = 1 ∨ a = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l771_77118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PM_is_3_l771_77174

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := ((Real.sqrt 2 / 2) * t, 3 + (Real.sqrt 2 / 2) * t)

-- Define the curve C in polar coordinates
def curve_C (θ : ℝ) : ℝ := 2 * Real.sin θ / (Real.cos θ)^2

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the intersection points A and B (existence assumed)
axiom exists_intersection : ∃ (t₁ t₂ : ℝ), 
  let (x₁, y₁) := line_l t₁
  let (x₂, y₂) := line_l t₂
  x₁^2 = 2*y₁ ∧ x₂^2 = 2*y₂ ∧ t₁ ≠ t₂

-- Define point M as the midpoint of A and B
def point_M : ℝ × ℝ := (1, 4)

-- State the theorem
theorem distance_PM_is_3 : 
  let (px, py) := point_P
  let (mx, my) := point_M
  ((px - mx)^2 + (py - my)^2) = 3^2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PM_is_3_l771_77174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_intervals_l771_77102

open Real Set

theorem strictly_increasing_intervals :
  let f : ℝ → ℝ := λ x => x * sin x + cos x
  StrictMonoOn f (Ioo (-π) (-π/2)) ∧ StrictMonoOn f (Ioo 0 (π/2)) :=
by
  -- Define the function
  let f : ℝ → ℝ := λ x => x * sin x + cos x
  
  -- State that we'll prove both parts of the conjunction
  have h1 : StrictMonoOn f (Ioo (-π) (-π/2)) := by sorry
  have h2 : StrictMonoOn f (Ioo 0 (π/2)) := by sorry
  
  -- Combine the two parts
  exact ⟨h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_intervals_l771_77102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l771_77145

noncomputable def f (x : ℝ) : ℝ := x^3 - Real.sqrt 3 * x + 3/5

noncomputable def tangent_slope (x : ℝ) : ℝ := 3 * x^2 - Real.sqrt 3

noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan m

theorem slope_angle_range : 
  ∀ x : ℝ, slope_angle (tangent_slope x) ∈ Set.union 
    (Set.Icc 0 (Real.pi / 2)) 
    (Set.Icc ((2 * Real.pi) / 3) Real.pi) :=
by
  sorry

#check slope_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l771_77145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_inequality_l771_77177

theorem cos_inequality (x y : ℝ) : Real.cos (x^2) + Real.cos (y^2) - Real.cos (x*y) < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_inequality_l771_77177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_three_point_four_five_l771_77126

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem round_three_point_four_five :
  roundToNearestTenth 3.45 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_three_point_four_five_l771_77126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_when_OP_minimum_l771_77153

/-- Given a point P(t/2 + 2/t, 1) where t < 0, prove that cos α = -2√5/5 when |OP| is minimum -/
theorem cos_alpha_when_OP_minimum (t : ℝ) (h : t < 0) :
  let x := t/2 + 2/t
  let y := 1
  let OP := Real.sqrt (x^2 + y^2)
  (∀ s, s < 0 → OP ≤ Real.sqrt ((s/2 + 2/s)^2 + 1)) →
  -2 * Real.sqrt 5 / 5 = x / OP := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_when_OP_minimum_l771_77153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_solution_l771_77168

def a (x : ℝ) : Fin 3 → ℝ := ![2*x, 1, 3]
def b (y : ℝ) : Fin 3 → ℝ := ![1, -2*y, 9]

theorem parallel_vectors_solution (x y : ℝ) :
  (∃ (k : ℝ), a x = k • b y) → x = 1/6 ∧ y = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_solution_l771_77168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_stack_distance_l771_77140

/-- Represents a stack of linked rings -/
structure RingStack where
  topDiameter : ℚ
  bottomDiameter : ℚ
  thickness : ℚ
  diameterDecrement : ℚ

/-- Calculates the vertical distance of a ring stack -/
def verticalDistance (stack : RingStack) : ℚ :=
  let insideTopDiameter := stack.topDiameter - 2 * stack.thickness
  let insideBottomDiameter := stack.bottomDiameter - 2 * stack.thickness
  let numRings := (insideTopDiameter - insideBottomDiameter) / stack.diameterDecrement + 1
  numRings * stack.thickness

/-- Theorem stating the vertical distance of the given ring stack is 260 cm -/
theorem ring_stack_distance : 
  let stack : RingStack := {
    topDiameter := 36,
    bottomDiameter := 12,
    thickness := 2,
    diameterDecrement := 2
  }
  verticalDistance stack = 260 := by sorry

#eval verticalDistance {
  topDiameter := 36,
  bottomDiameter := 12,
  thickness := 2,
  diameterDecrement := 2
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_stack_distance_l771_77140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toyota_honda_ratio_l771_77135

/-- Represents the number of Toyota vehicles -/
def T : ℕ := sorry

/-- Represents the number of Honda vehicles -/
def H : ℕ := sorry

/-- The total number of SUVs bought -/
def total_suvs : ℕ := 52

/-- The percentage of Toyota vehicles that are SUVs -/
def toyota_suv_percentage : ℚ := 2/5

/-- The percentage of Honda vehicles that are SUVs -/
def honda_suv_percentage : ℚ := 3/5

/-- Theorem stating the ratio of Toyota to Honda vehicles -/
theorem toyota_honda_ratio :
  (toyota_suv_percentage * ↑T + honda_suv_percentage * ↑H : ℚ) = total_suvs →
  T = 5 * H := by
  sorry

#check toyota_honda_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toyota_honda_ratio_l771_77135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_alpha_l771_77115

theorem sin_pi_minus_alpha (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = 4/5) :
  Real.sin (Real.pi - α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_alpha_l771_77115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_cube_root_sum_l771_77189

noncomputable def f (x : ℝ) : ℝ := (13 + Real.sqrt x)^(1/3) + (13 - Real.sqrt x)^(1/3)

theorem integer_cube_root_sum (x : ℝ) : 
  x ≥ 0 ∧ ∃ n : ℤ, f x = n ↔ 
  x = 137 + 53/216 ∨ 
  x = 168 + 728/729 ∨ 
  x = 196 ∨ 
  x = 747 + 19/27 := by
  sorry

#check integer_cube_root_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_cube_root_sum_l771_77189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalent_to_inverse_l771_77165

/-- Two angles are supplementary -/
def supplementary (α β : ℝ) : Prop := α + β = 180

/-- Two angles are congruent -/
def congruent_angles (α β : ℝ) : Prop := α = β

/-- Two lines are parallel -/
def parallel_lines (l₁ l₂ : Set (ℝ × ℝ)) : Prop := ∃ (a b c d : ℝ), l₁ = {(x, y) | a*x + b*y = c} ∧ l₂ = {(x, y) | a*x + b*y = d}

/-- The proposition "Supplementary angles are congruent, and two lines are parallel" -/
def proposition (α β : ℝ) (l₁ l₂ : Set (ℝ × ℝ)) : Prop :=
  supplementary α β ∧ congruent_angles α β ∧ parallel_lines l₁ l₂

/-- The inverse proposition "Two lines are parallel, and supplementary angles are congruent" -/
def inverse_proposition (α β : ℝ) (l₁ l₂ : Set (ℝ × ℝ)) : Prop :=
  parallel_lines l₁ l₂ ∧ supplementary α β ∧ congruent_angles α β

theorem proposition_equivalent_to_inverse :
  ∀ (α β : ℝ) (l₁ l₂ : Set (ℝ × ℝ)),
    proposition α β l₁ l₂ ↔ inverse_proposition α β l₁ l₂ := by
  intros α β l₁ l₂
  apply Iff.intro
  · intro h
    exact ⟨h.2.2, h.1, h.2.1⟩
  · intro h
    exact ⟨h.2.1, h.2.2, h.1⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalent_to_inverse_l771_77165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_multiple_of_three_smallest_n_for_2017_l771_77162

def f : ℕ → ℤ
  | 0 => 0
  | n + 1 => if n % 2 = 0 then -f (n / 2) else f n + 1

theorem f_multiple_of_three (n : ℕ) : 3 ∣ f n ↔ 3 ∣ n := by sorry

theorem smallest_n_for_2017 : ∃ n : ℕ, f n = 2017 ∧ ∀ m : ℕ, m < n → f m ≠ 2017 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_multiple_of_three_smallest_n_for_2017_l771_77162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_to_base_formula_l771_77182

/-- A cylinder with a square axial section -/
structure SquareAxialCylinder where
  /-- Length of the side of the square axial section -/
  side : ℝ
  /-- Segment AB connecting points on upper and lower base circumferences -/
  segment_length : ℝ
  /-- Distance of segment AB from cylinder axis -/
  segment_distance : ℝ
  /-- The side length is positive -/
  side_pos : 0 < side
  /-- The segment length is positive -/
  segment_length_pos : 0 < segment_length
  /-- The segment distance is non-negative and less than half the side length -/
  segment_distance_bounds : 0 ≤ segment_distance ∧ segment_distance < side / 2

/-- The angle between the segment AB and the base plane of the cylinder -/
noncomputable def angle_to_base (c : SquareAxialCylinder) : ℝ :=
  (1 / 2) * Real.arccos (-4 * c.segment_distance^2 / c.segment_length^2)

/-- Theorem: The angle between segment AB and the base plane is as defined -/
theorem angle_to_base_formula (c : SquareAxialCylinder) :
  angle_to_base c = (1 / 2) * Real.arccos (-4 * c.segment_distance^2 / c.segment_length^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_to_base_formula_l771_77182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_is_22_l771_77108

/-- Represents the properties of a rectangular floor. -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  paintCost : ℝ
  paintRate : ℝ

/-- Checks if the floor satisfies the given conditions. -/
def isValidFloor (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth ∧
  floor.paintCost = 484 ∧
  floor.paintRate = 3 ∧
  floor.paintCost = floor.paintRate * floor.length * floor.breadth

/-- Theorem stating that a valid floor has a length of approximately 22 meters. -/
theorem floor_length_is_22 (floor : RectangularFloor) 
    (h : isValidFloor floor) : 
    ∃ ε > 0, |floor.length - 22| < ε := by
  sorry

#check floor_length_is_22

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_is_22_l771_77108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_hyperbola_to_line_l771_77107

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - 2*y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := Real.sqrt 2 * x - 2*y + 2 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (Real.sqrt 2 * x - 2*y + 2) / Real.sqrt 6

-- Theorem statement
theorem max_distance_hyperbola_to_line :
  ∃ (max_t : ℝ), max_t = Real.sqrt 6 / 3 ∧
  (∀ (x y : ℝ), hyperbola x y → x > 0 → distance_to_line x y ≤ max_t) ∧
  (∃ (x y : ℝ), hyperbola x y ∧ x > 0 ∧ distance_to_line x y = max_t) := by
  sorry

#check max_distance_hyperbola_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_hyperbola_to_line_l771_77107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosecant_minus_cosine_equality_l771_77199

theorem cosecant_minus_cosine_equality : 
  1 / Real.sin (π / 18) - 6 * Real.cos (2 * π / 9) = 
  (2 * Real.cos (π / 18)) / (2 * Real.sin (π / 18) * Real.cos (π / 18)) - 
  6 * (2 * (2 * Real.cos (π / 18) ^ 2 - 1) ^ 2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosecant_minus_cosine_equality_l771_77199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grandma_calls_l771_77154

def days_with_at_least_one_call (total_days : ℕ) (call_periods : List ℕ) : ℕ :=
  sorry

def days_without_call (total_days : ℕ) (call_periods : List ℕ) : ℕ :=
  total_days - (days_with_at_least_one_call total_days call_periods)

theorem grandma_calls (total_days : ℕ) (call_periods : List ℕ) :
  total_days = 366 ∧ call_periods = [4, 5, 6, 8] →
  days_without_call total_days call_periods = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grandma_calls_l771_77154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_values_l771_77181

noncomputable def f (a b c : ℕ) (x : ℝ) : ℝ :=
  if x > 0 then a * x + 2
  else if x = 0 then 4 * b
  else 2 * b * x + c

theorem piecewise_function_values (a b c : ℕ) :
  f a b c 1 = 3 ∧ f a b c 0 = 8 ∧ f a b c (-1) = -4 → a = 1 ∧ b = 2 ∧ c = 0 :=
by
  intro h
  sorry

#check piecewise_function_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_values_l771_77181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liar_count_in_kingdom_l771_77125

def Island := Nat
def Population := Nat
def KnightCount := Nat

structure Kingdom where
  islands : Nat
  population : Nat
  first_yes : Nat
  second_no : Nat

def count_liars (k : Kingdom) : Nat :=
  k.islands * k.population - 
    (60 * k.second_no + 59 * (k.islands - k.second_no - (k.first_yes - k.second_no)))

theorem liar_count_in_kingdom (k : Kingdom) 
  (h1 : k.islands = 17)
  (h2 : k.population = 119)
  (h3 : k.first_yes = 7)
  (h4 : k.second_no = 7)
  : count_liars k = 1013 := by
  sorry

#eval count_liars {islands := 17, population := 119, first_yes := 7, second_no := 7}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liar_count_in_kingdom_l771_77125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l771_77161

def n : ℕ := 2^5 * 3^7 * 5^3 * 15^2

theorem number_of_factors : (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l771_77161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_terms_l771_77114

def our_sequence (a : ℕ → ℕ) : Prop :=
  a 2 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = 3 * n

theorem sum_of_specific_terms (a : ℕ → ℕ) (h : our_sequence a) :
  a 2 + a 4 + a 6 + a 8 + a 10 + a 12 = 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_terms_l771_77114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_and_minimum_l771_77112

-- Define the quadratic function
noncomputable def q (a b x : ℝ) : ℝ := x^2 - 5*a*x + b

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, q a b x > 0 ↔ (x > 4 ∨ x < 1)

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a/x + b/(1-x)

-- Theorem statement
theorem quadratic_and_minimum :
  ∃ a b : ℝ,
    solution_set a b ∧
    a = 1 ∧
    b = 4 ∧
    ∀ x, 0 < x → x < 1 →
      f a b x ≥ 9 ∧
      (∃ x₀, 0 < x₀ ∧ x₀ < 1 ∧ f a b x₀ = 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_and_minimum_l771_77112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_exp_3x_l771_77186

open Real MeasureTheory

theorem integral_x_squared_exp_3x : ∫ x in (Set.Icc 0 1), x^2 * Real.exp (3 * x) = (5 * Real.exp 3 - 2) / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_exp_3x_l771_77186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_fourth_quadrant_condition_l771_77131

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (1 + Complex.I) * m^2 - (8 + 5*Complex.I) * m + (15 - 14*Complex.I)

-- Part I: Pure imaginary number condition
theorem pure_imaginary_condition (m : ℝ) :
  z m = Complex.I * (z m).im → m = 3 ∨ m = 5 :=
by sorry

-- Part II: Fourth quadrant condition
theorem fourth_quadrant_condition (m : ℝ) :
  (z m).re > 0 ∧ (z m).im < 0 → (-2 < m ∧ m < 3) ∨ (5 < m ∧ m < 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_fourth_quadrant_condition_l771_77131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_line_l771_77120

-- Define the curve and line functions
def curve (x : ℝ) : ℝ := 3 - x^2
def line (x : ℝ) : ℝ := 2*x

-- Define the enclosed area
noncomputable def enclosed_area : ℝ := ∫ x in Set.Icc (-3) 1, (curve x - line x)

-- Theorem statement
theorem area_enclosed_by_curve_and_line : enclosed_area = 32/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_line_l771_77120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_width_for_truck_l771_77139

/-- Represents a parabola with equation x^2 = -2py -/
structure Parabola where
  p : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Checks if a rectangle fits inside a parabola -/
def fits_inside (r : Rectangle) (p : Parabola) : Prop :=
  r.width^2 / 4 = p.p * (p.p / 2 - r.height)

theorem parabola_width_for_truck (p : Parabola) (r : Rectangle) :
  r.width = 1.6 →
  r.height = 3 →
  fits_inside r p →
  ∃ ε > 0, |2 * p.p - 12.21| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_width_for_truck_l771_77139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pennys_canoe_capacity_l771_77143

/-- The number of people Penny's canoe can carry without the dog -/
def P : ℕ := sorry

/-- The weight of each person in pounds -/
def person_weight : ℕ := 140

/-- The weight of the dog in pounds -/
def dog_weight : ℕ := person_weight / 4

/-- The total weight the canoe was carrying with the dog in pounds -/
def total_weight : ℕ := 595

/-- Theorem stating that Penny's canoe can carry 6 people without the dog -/
theorem pennys_canoe_capacity :
  (2 * P / 3 * person_weight + dog_weight = total_weight) → P = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pennys_canoe_capacity_l771_77143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_base_l771_77121

-- Define the function
noncomputable def f (a x : ℝ) : ℝ := (a^2 - 3*a + 3) * a^x

-- State the theorem
theorem exponential_function_base (a : ℝ) :
  (∀ x, ∃ k, f a x = k * a^x) →  -- f is an exponential function
  (a > 0) →                      -- base of exponential function is positive
  (a ≠ 1) →                      -- base of exponential function is not 1
  a = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_base_l771_77121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_condition_l771_77185

theorem tangent_perpendicular_condition (b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + b*x
  let f' : ℝ → ℝ := λ x ↦ 2*x + b
  let perpendicular_line : ℝ → ℝ := λ x ↦ -(1/3)*x + 2/3
  (f' 1 = (perpendicular_line 1)⁻¹) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_condition_l771_77185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_matrix_det_is_one_l771_77193

/-- The determinant of the given trigonometric matrix is 1 for all real α and θ. -/
theorem trig_matrix_det_is_one (α θ : ℝ) : 
  Matrix.det ![
    ![Real.cos α * Real.cos θ, Real.cos α * Real.sin θ, Real.sin α],
    ![Real.sin θ, -Real.cos θ, 0],
    ![Real.sin α * Real.cos θ, Real.sin α * Real.sin θ, -Real.cos α]
  ] = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_matrix_det_is_one_l771_77193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_dihedral_angles_gt_pi_l771_77132

/-- A trihedral angle with dihedral angles α, β, and γ -/
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real

/-- The sum of dihedral angles in a trihedral angle is greater than π -/
theorem sum_dihedral_angles_gt_pi (t : TrihedralAngle) : t.α + t.β + t.γ > Real.pi := by
  sorry

/-- The sum of plane angles in a spherical triangle is less than 2π -/
axiom sum_plane_angles_lt_two_pi (a' b' c' : Real) : a' + b' + c' < 2 * Real.pi

/-- Relation between plane angles and dihedral angles in a trihedral angle -/
axiom dihedral_plane_angle_relation (t : TrihedralAngle) (a' b' c' : Real) :
  b' + c' = Real.pi - t.α ∧ c' + a' = Real.pi - t.β ∧ a' + b' = Real.pi - t.γ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_dihedral_angles_gt_pi_l771_77132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_mileage_approx_l771_77130

/-- Represents the routes and conditions for two bikers --/
structure BikeRoutes where
  boston_distance : ℝ
  atlanta_distance : ℝ
  total_days : ℕ
  boston_hilly_days : ℕ
  boston_rainy_days : ℕ
  boston_rest_days : ℕ
  atlanta_hilly_days : ℕ
  atlanta_rainy_days : ℕ
  atlanta_rest_days : ℕ
  hilly_reduction : ℝ
  rainy_reduction : ℝ

/-- Calculates the maximum daily mileage for both bikers --/
noncomputable def max_daily_mileage (routes : BikeRoutes) : ℝ :=
  let boston_factor := routes.total_days - routes.boston_hilly_days * routes.hilly_reduction -
                       routes.boston_rainy_days * routes.rainy_reduction - routes.boston_rest_days
  let atlanta_factor := routes.total_days - routes.atlanta_hilly_days * routes.hilly_reduction -
                        routes.atlanta_rainy_days * routes.rainy_reduction - routes.atlanta_rest_days
  min (routes.boston_distance / boston_factor) (routes.atlanta_distance / atlanta_factor)

/-- Theorem stating the maximum daily mileage for the given conditions --/
theorem max_daily_mileage_approx (routes : BikeRoutes)
  (h1 : routes.boston_distance = 840)
  (h2 : routes.atlanta_distance = 440)
  (h3 : routes.total_days = 7)
  (h4 : routes.boston_hilly_days = 2)
  (h5 : routes.boston_rainy_days = 2)
  (h6 : routes.boston_rest_days = 1)
  (h7 : routes.atlanta_hilly_days = 1)
  (h8 : routes.atlanta_rainy_days = 1)
  (h9 : routes.atlanta_rest_days = 1)
  (h10 : routes.hilly_reduction = 0.2)
  (h11 : routes.rainy_reduction = 0.1) :
  ∃ ε > 0, abs (max_daily_mileage routes - 77.19) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_mileage_approx_l771_77130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_problem_l771_77142

open Real Set

theorem isosceles_triangle_problem (α β : Real) (h_isosceles : β = π - 2*α) 
  (h_cos_β : cos β = 3/5) (h_α_range : 0 < α ∧ α < π/2) :
  (sin α = 2 * Real.sqrt 5 / 5) ∧
  (∀ m : Real, (Icc (-π/3) α).image tan = 
                (Icc 0 m).image (fun x => 2 * sin (2*x - π/3)) → 
                (5*π/12 ≤ m ∧ m ≤ 5*π/6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_problem_l771_77142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ca_approx_l771_77104

/-- The molar mass of calcium in g/mol -/
noncomputable def molar_mass_Ca : ℝ := 40.08

/-- The molar mass of bromine in g/mol -/
noncomputable def molar_mass_Br : ℝ := 79.904

/-- The number of bromine atoms in calcium bromide -/
def num_Br_atoms : ℕ := 2

/-- The molar mass of calcium bromide in g/mol -/
noncomputable def molar_mass_CaBr2 : ℝ := molar_mass_Ca + num_Br_atoms * molar_mass_Br

/-- The mass percentage of calcium in calcium bromide -/
noncomputable def mass_percentage_Ca : ℝ := (molar_mass_Ca / molar_mass_CaBr2) * 100

theorem mass_percentage_Ca_approx :
  ∃ ε > 0, |mass_percentage_Ca - 20.04| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ca_approx_l771_77104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_mean_score_l771_77172

theorem exam_mean_score (morning_mean afternoon_mean : ℝ) 
  (morning_students afternoon_students : ℕ) : 
  morning_mean = 84 →
  afternoon_mean = 70 →
  morning_students = (3 : ℕ) * afternoon_students / 4 →
  (morning_mean * (morning_students : ℝ) + afternoon_mean * (afternoon_students : ℝ)) / 
    ((morning_students : ℝ) + (afternoon_students : ℝ)) = 76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_mean_score_l771_77172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_quadratic_l771_77156

/-- A function f(x) is quadratic if it can be written as f(x) = ax^2 + bx + c, where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given function -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * (x ^ (m^2 - 7))

theorem unique_m_for_quadratic :
  ∃! m : ℝ, IsQuadratic (f m) ∧ m - 3 ≠ 0 := by
  sorry

#check unique_m_for_quadratic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_quadratic_l771_77156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_deriv_2017_l771_77127

/-- The sequence of functions defined by repeated differentiation of sine -/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => Real.sin
  | n + 1 => deriv (f n)

/-- Theorem stating that the 2017th derivative of sine is cosine -/
theorem sine_deriv_2017 : f 2017 = Real.cos := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_deriv_2017_l771_77127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_number_properties_l771_77198

theorem rational_number_properties :
  (∃ a b : ℚ, |a| = |b| ∧ a ≠ b) ∧
  (∀ a b : ℚ, a = -b → (-a)^2 = b^2) ∧
  (∃ a b : ℚ, |a| > b ∧ |a| ≤ |b|) ∧
  (∃ a b : ℚ, |a| < |b| ∧ a ≥ b) :=
by
  sorry

#check rational_number_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_number_properties_l771_77198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_a_given_sum_condition_l771_77197

theorem max_sin_a_given_sum_condition (a b : ℝ) :
  Real.sin (a + b) = Real.sin a + Real.sin b → (∀ x, Real.sin x ≤ Real.sin a → Real.sin x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_a_given_sum_condition_l771_77197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_congruence_l771_77117

def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1

def S_q (p : ℕ) : ℚ :=
  let q := (3 * p - 5) / 2
  (Finset.range q).sum (λ k ↦ 1 / ((2 * k + 2) * (2 * k + 3) * (2 * k + 4)))

theorem odd_prime_congruence (p : ℕ) (h : is_odd_prime p) :
  let q := (3 * p - 5) / 2
  let frac := 1 / p - 2 * S_q p
  ∃ (m n : ℤ), frac = m / n ∧ m ≡ n [ZMOD p] := by
  sorry

#check odd_prime_congruence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_congruence_l771_77117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_eight_l771_77191

-- Define the type for our 3x3 grid
def Grid := Matrix (Fin 3) (Fin 3) ℕ

-- Define a function to check if two positions share an edge
def sharesEdge (i j i' j' : Fin 3) : Prop :=
  (i = i' ∧ (j = j' + 1 ∨ j' = j + 1)) ∨ (j = j' ∧ (i = i' + 1 ∨ i' = i + 1))

-- Define the property of consecutive numbers
def consecutive (a b : ℕ) : Prop := a + 1 = b ∨ b + 1 = a

-- Define the property that 1 is in a corner
def oneInCorner (g : Grid) : Prop :=
  g 0 0 = 1 ∨ g 0 2 = 1 ∨ g 2 0 = 1 ∨ g 2 2 = 1

-- Define the sum of corners
def cornerSum (g : Grid) : ℕ := g 0 0 + g 0 2 + g 2 0 + g 2 2

-- Main theorem
theorem center_is_eight (g : Grid) 
  (unique : ∀ i j i' j', g i j = g i' j' → i = i' ∧ j = j')
  (range : ∀ i j, 1 ≤ g i j ∧ g i j ≤ 9)
  (consec : ∀ i j i' j', consecutive (g i j) (g i' j') → sharesEdge i j i' j')
  (corner_one : oneInCorner g)
  (corner_sum : cornerSum g = 20) :
  g 1 1 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_eight_l771_77191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_exists_l771_77128

/-- Given a finite line segment AB, there exists a point C such that triangle ABC is equilateral. -/
theorem equilateral_triangle_exists (A B : EuclideanSpace ℝ (Fin 2)) : 
  ∃ C : EuclideanSpace ℝ (Fin 2), 
    dist A C = dist A B ∧ 
    dist B C = dist A B ∧ 
    dist A C = dist B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_exists_l771_77128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l771_77160

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line ax + y - 1 = 0
def my_line (a x y : ℝ) : Prop := a * x + y - 1 = 0

-- Define the tangent line passing through (1,2)
def my_tangent_line (k x y : ℝ) : Prop := k * x - y + 2 - k = 0

-- Define the perpendicularity condition
def my_perpendicular (a k : ℝ) : Prop := a * k = -1 ∨ (a = 0 ∧ k = 0)

-- Define the tangency condition
def my_is_tangent (k : ℝ) : Prop := (2 - k)^2 / (k^2 + 1) = 1

theorem tangent_line_theorem (a : ℝ) :
  (∃ k : ℝ, my_tangent_line k 1 2 ∧ 
            my_is_tangent k ∧ 
            my_perpendicular a k) →
  a = 0 ∨ a = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l771_77160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l771_77175

-- Define the function
noncomputable def f (x : ℝ) := Real.log (x^2 - 2*x) / Real.log 3

-- Define the derivative of the function
noncomputable def f_derivative (x : ℝ) := (2*(x - 1)) / ((x^2 - 2*x) * Real.log 3)

-- Theorem stating the monotonic decreasing interval
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, x < 0 → (∀ y : ℝ, y < 0 → f_derivative y < 0) ∧ 
  (∀ z : ℝ, z ≥ 0 → z ≤ 2 → f_derivative z ≥ 0) :=
by
  sorry

#check monotonic_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l771_77175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l771_77184

/-- Represents the fuel efficiency of an SUV in different driving conditions -/
structure SUVFuelEfficiency where
  highway_mpg : ℚ
  city_mpg : ℚ

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available fuel -/
def max_distance (efficiency : SUVFuelEfficiency) (fuel_gallons : ℚ) : ℚ :=
  max efficiency.highway_mpg efficiency.city_mpg * fuel_gallons

/-- Theorem: The maximum distance the SUV can travel on 25 gallons of gasoline is 305 miles -/
theorem suv_max_distance :
  let efficiency := SUVFuelEfficiency.mk (22 / 10 : ℚ) (19 / 5 : ℚ)
  max_distance efficiency 25 = 305 := by
  -- Unfold the definitions
  unfold max_distance
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

#eval max_distance (SUVFuelEfficiency.mk (22 / 10 : ℚ) (19 / 5 : ℚ)) 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l771_77184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_justine_sheets_used_l771_77167

-- Define the given quantities
def total_sheets : ℕ := 4600
def num_binders : ℕ := 11
def fraction_colored : ℚ := 3 / 5

-- Define the function to calculate sheets used
def sheets_used (total : ℕ) (binders : ℕ) (fraction : ℚ) : ℕ :=
  (fraction * (total / binders : ℚ)).floor.toNat

-- State the theorem
theorem justine_sheets_used :
  sheets_used total_sheets num_binders fraction_colored = 250 := by
  -- Unfold the definition of sheets_used
  unfold sheets_used
  -- Evaluate the expression
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_justine_sheets_used_l771_77167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_specific_M_coordinates_l771_77179

-- Define the circle A
def circle_A (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8

-- Define point B
def point_B : ℝ × ℝ := (1, 0)

-- Define a moving point P on circle A
def point_P (x y : ℝ) : Prop := circle_A x y

-- Define point M as the intersection of perpendicular bisector of PB and radius PA
def point_M (x y : ℝ) (px py : ℝ) : Prop :=
  point_P px py ∧ 
  (x - (-1))^2 + y^2 = (px - (-1))^2 + py^2 ∧ 
  (x - 1)^2 + y^2 = (x - px)^2 + (y - py)^2

-- Theorem 1: The trajectory of point M forms an ellipse
theorem trajectory_is_ellipse :
  ∀ x y : ℝ, (∃ px py : ℝ, point_M x y px py) → (x^2 / 2 + y^2 = 1) :=
sorry

-- Theorem 2: Coordinates of M when P is in first quadrant and cos∠BAP = 2√2/3
theorem specific_M_coordinates :
  ∀ px py : ℝ, 
    point_P px py ∧ 
    px > -1 ∧ py > 0 ∧ 
    ((px + 1) / (2 * Real.sqrt 2)) = 2 * Real.sqrt 2 / 3 →
    (∃ x y : ℝ, point_M x y px py ∧ x = 1 ∧ y = Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_specific_M_coordinates_l771_77179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_triangle_area_ratio_l771_77149

theorem hexagon_triangle_area_ratio (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  let triangle_area := Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4
  let hexagon_area := triangle_area + 
    (a * (a + b) * (a + c) + b * (a + b) * (b + c) + c * (b + c) * (c + a)) / (4 * triangle_area)
  hexagon_area / triangle_area ≥ 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_triangle_area_ratio_l771_77149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_m_for_odd_function_l771_77133

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x) - Real.sin (2 * x)

theorem minimum_m_for_odd_function (m : ℝ) (h_m : m > 0) :
  (∀ x, f (x - m) = -f (-x - m)) → m ≥ π / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_m_for_odd_function_l771_77133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_person_speed_l771_77187

/-- Calculates the walking speed of the second person given the track circumference,
    the first person's speed, and the time they meet. -/
noncomputable def calculate_second_speed (track_circumference : ℝ) (first_speed : ℝ) (meet_time : ℝ) : ℝ :=
  let total_distance := track_circumference / 1000 -- Convert m to km
  let time_hours := meet_time / 60 -- Convert minutes to hours
  let first_distance := first_speed * time_hours
  let second_distance := total_distance - first_distance
  second_distance / time_hours

/-- Theorem stating that given the specified conditions, the second person's
    walking speed is 3.75 km/hr. -/
theorem second_person_speed
  (track_circumference : ℝ)
  (first_speed : ℝ)
  (meet_time : ℝ)
  (h1 : track_circumference = 528)
  (h2 : first_speed = 4.5)
  (h3 : meet_time = 3.84) :
  calculate_second_speed track_circumference first_speed meet_time = 3.75 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_person_speed_l771_77187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l771_77158

theorem cos_alpha_minus_pi_fourth (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) (h2 : Real.tan α = 2) :
  Real.cos (α - π/4) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l771_77158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l771_77180

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l771_77180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_5_sqrt_2_l771_77183

/-- A circle passing through two points with its center on the x-axis -/
structure CircleWithCenterOnXAxis where
  center : ℝ  -- x-coordinate of the center
  radius : ℝ
  point1 : ℝ × ℝ := (0, 5)
  point2 : ℝ × ℝ := (2, 1)
  center_on_x_axis : True  -- This is always true since center is just an x-coordinate
  point1_on_circle : (center - point1.1)^2 + point1.2^2 = radius^2
  point2_on_circle : (center - point2.1)^2 + point2.2^2 = radius^2

/-- The radius of the circle is 5√2 -/
theorem circle_radius_is_5_sqrt_2 :
  ∃ (c : CircleWithCenterOnXAxis), c.radius = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_5_sqrt_2_l771_77183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_51_l771_77100

noncomputable section

-- Define the travel segments
def segment1_speed : ℝ := 40
def segment1_distance : ℝ := 20
def segment2_speed : ℝ := 50
def segment2_distance : ℝ := 25
def segment3_speed : ℝ := 60
def segment3_time : ℝ := 45 / 60  -- 45 minutes in hours
def segment4_speed : ℝ := 48
def segment4_time : ℝ := 15 / 60  -- 15 minutes in hours

-- Define the theorem
theorem average_speed_is_51 :
  let total_distance := segment1_distance + segment2_distance + 
                        segment3_speed * segment3_time + 
                        segment4_speed * segment4_time
  let total_time := segment1_distance / segment1_speed + 
                    segment2_distance / segment2_speed + 
                    segment3_time + segment4_time
  (total_distance / total_time) = 51 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_51_l771_77100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_range_l771_77152

-- Define the power function f(x)
noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x^(-m^2 + 2*m + 3)

-- Define the function g(x)
noncomputable def g (q : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt (f 1 x) - 8*x + q - 1

-- State the theorem
theorem power_function_and_range :
  (∀ x > 0, Monotone (f 1)) ∧  -- f(x) is monotonically increasing in (0, +∞)
  (∀ x, f 1 x = f 1 (-x)) →    -- f(x) is an even function
  (∀ x, f 1 x = x^4) ∧         -- f(x) = x^4
  {q : ℝ | ∀ x ∈ Set.Icc (-1) 1, g q x > 0} = Set.Ioi 7  -- Range of q is (7, +∞)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_range_l771_77152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_eq_seven_l771_77141

/-- The number of integers n such that 200 < n < 300 and n mod 7 = n mod 9 -/
def count_special_integers : Nat :=
  Finset.card (Finset.filter (fun n => 200 < n ∧ n < 300 ∧ n % 7 = n % 9) (Finset.range 300))

/-- Theorem stating that there are exactly 7 integers satisfying the conditions -/
theorem count_special_integers_eq_seven : count_special_integers = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_eq_seven_l771_77141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_circle_parameter_l771_77188

-- Define the polar function
noncomputable def r (θ : Real) : Real := Real.cos (2 * θ)

-- Theorem statement
theorem smallest_complete_circle_parameter :
  ∃ (t : Real), t > 0 ∧ 
  (∀ (θ : Real), 0 ≤ θ ∧ θ ≤ t → r θ ∈ Set.range r) ∧
  (∀ (t' : Real), 0 < t' ∧ t' < t → 
    ∃ (θ : Real), θ > t' ∧ r θ ∉ Set.range (r ∘ (fun x => x * (t' / t)))) ∧
  t = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_circle_parameter_l771_77188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_speedometer_calculation_l771_77157

/-- Represents a faulty speedometer with a constant error rate -/
structure FaultySpeedometer where
  /-- The reading on the speedometer -/
  reading : ℚ
  /-- The actual distance traveled -/
  actual : ℚ
  /-- The error rate is constant -/
  constant_error : reading / actual = 144 / 150

/-- Calculates the actual distance traveled given a speedometer reading -/
def actualDistance (s : FaultySpeedometer) (newReading : ℚ) : ℚ :=
  newReading * s.actual / s.reading

/-- Theorem: Given a faulty speedometer that reads 144 km for an actual distance of 150 km,
    the actual distance traveled when the speedometer shows 1200 km is 1250 km -/
theorem faulty_speedometer_calculation (s : FaultySpeedometer)
    (h1 : s.reading = 144)
    (h2 : s.actual = 150) :
    actualDistance s 1200 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_speedometer_calculation_l771_77157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_bound_l771_77155

/-- Given a triangle ABC with side lengths a, b, c, when reflected over its incenter
    to form triangle A₁B₁C₁, the perimeter of the common hexagon formed by the
    intersection of ABC and A₁B₁C₁ is bounded above by 2(ab + bc + ca) / (a + b + c). -/
theorem hexagon_perimeter_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (p : ℝ), p ≤ 2 * (a * b + b * c + c * a) / (a + b + c) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_bound_l771_77155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l771_77176

/-- The function g resulting from the transformation of sin x -/
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 6))

/-- The set of intervals where g is monotonically increasing -/
def increasing_intervals : Set (Set ℝ) :=
  {I | ∃ k : ℤ, I = Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)}

/-- Theorem stating that g is monotonically increasing on the specified intervals -/
theorem g_monotone_increasing : 
  ∀ I ∈ increasing_intervals, StrictMonoOn g I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l771_77176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_rate_l771_77169

/-- Represents the annual growth rate of investment -/
def annual_growth_rate (r : ℝ) : Prop :=
  (500 * (1 + r)^2 = 720) ∧ (r > 0)

/-- The annual growth rate of investment is 20% -/
theorem investment_growth_rate : annual_growth_rate 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_rate_l771_77169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_2023_rearrangements_l771_77123

/-- The set of digits in 2023 -/
def digits : Finset Nat := {2, 0, 2, 3}

/-- A function that checks if a list of digits forms a valid four-digit number -/
def is_valid_four_digit (l : List Nat) : Bool :=
  l.length = 4 ∧ l.head? ≠ some 0

/-- The number of valid four-digit numbers formed by rearranging digits in 2023 -/
noncomputable def count_valid_numbers : Nat :=
  (digits.toList.permutations.filter is_valid_four_digit).length

theorem count_2023_rearrangements : count_valid_numbers = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_2023_rearrangements_l771_77123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AC_l771_77163

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (AB : ℝ)
  (DC : ℝ)
  (AD : ℝ)

-- Define the theorem
theorem length_of_AC (q : Quadrilateral) (h1 : q.AB = 15) (h2 : q.DC = 24) (h3 : q.AD = 7) :
  ∃ AC : ℝ, abs (AC - 30.1) < 0.05 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AC_l771_77163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_zero_monotonicity_condition_l771_77195

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (x^2 + a*x) * Real.exp x

-- Part 1
theorem monotonicity_when_a_zero :
  ∀ x : ℝ, 
    (x < -2 ∨ x > 0 → (deriv (f 0)) x > 0) ∧ 
    (-2 < x ∧ x < 0 → (deriv (f 0)) x < 0) := by sorry

-- Part 2
theorem monotonicity_condition :
  ∀ a : ℝ, 
    (∀ x ∈ Set.Ioo 1 2, (deriv (f a)) x ≤ 0) ↔ 
    a ≤ -8/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_zero_monotonicity_condition_l771_77195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l771_77110

noncomputable def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ 4 * f ((x + Real.pi/2) / 2)

theorem function_transformation (f : ℝ → ℝ) :
  (∀ x, transform f x = 2 * Real.sin x) → 
  (∀ x, f x = -1/2 * Real.cos (2*x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l771_77110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnus_third_game_score_l771_77109

/-- Represents a player in the game -/
inductive Player : Type where
  | Magnus : Player
  | Viswanathan : Player
deriving Repr, DecidableEq

/-- Represents a game number -/
inductive GameNumber : Type where
  | First : GameNumber
  | Second : GameNumber
  | Third : GameNumber
deriving Repr, DecidableEq

/-- Function to get a player's score in a specific game -/
def score : Player → GameNumber → ℕ := sorry

/-- Function to determine the winner of a game -/
def winner : GameNumber → Player := sorry

/-- Theorem stating Magnus's score in the third game is 19 -/
theorem magnus_third_game_score :
  (∀ p g, score p g > 0) →  -- All scores are positive integers
  (∀ p1 g1 p2 g2, (p1 ≠ p2 ∨ g1 ≠ g2) → score p1 g1 ≠ score p2 g2) →  -- All six scores are different
  (∀ g, score (winner g) g ≥ 25) →  -- Winner's score is at least 25
  (∀ g, score (winner g) g = 25 → 
    score (if winner g = Player.Magnus then Player.Viswanathan else Player.Magnus) g ≤ 23) →  -- If winner scores 25, opponent scores at most 23
  (∀ g, score (winner g) g > 25 → 
    score (if winner g = Player.Magnus then Player.Viswanathan else Player.Magnus) g = score (winner g) g - 2) →  -- If winner scores > 25, opponent scores 2 less
  (winner GameNumber.First = Player.Viswanathan ↔ winner GameNumber.Second = Player.Magnus) →  -- Viswanathan wins either first or second game, not both
  (winner GameNumber.Third = Player.Viswanathan) →  -- Viswanathan wins third game
  (score Player.Viswanathan GameNumber.Third = 25) →  -- Viswanathan's score in third game is 25
  (∀ p, 2 * score p GameNumber.Second = score p GameNumber.First + score p GameNumber.Third) →  -- Second game score is average of first and third
  score Player.Magnus GameNumber.Third = 19  -- Magnus's score in third game is 19
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnus_third_game_score_l771_77109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_initial_number_l771_77134

theorem largest_initial_number : 
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℕ), 
    89 + a₁ + a₂ + a₃ + a₄ + a₅ = 100 ∧ 
    ¬(a₁ ∣ 89) ∧ 
    ¬(a₂ ∣ (89 + a₁)) ∧ 
    ¬(a₃ ∣ (89 + a₁ + a₂)) ∧ 
    ¬(a₄ ∣ (89 + a₁ + a₂ + a₃)) ∧ 
    ¬(a₅ ∣ (89 + a₁ + a₂ + a₃ + a₄)) ∧ 
    ∀ n > 89, ¬∃ (b₁ b₂ b₃ b₄ b₅ : ℕ), 
      n + b₁ + b₂ + b₃ + b₄ + b₅ = 100 ∧ 
      ¬(b₁ ∣ n) ∧ 
      ¬(b₂ ∣ (n + b₁)) ∧ 
      ¬(b₃ ∣ (n + b₁ + b₂)) ∧ 
      ¬(b₄ ∣ (n + b₁ + b₂ + b₃)) ∧ 
      ¬(b₅ ∣ (n + b₁ + b₂ + b₃ + b₄)) := by
  sorry

#check largest_initial_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_initial_number_l771_77134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_intersects_x_axis_l771_77144

-- Define the logarithm function to base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log2_intersects_x_axis : ∃ x : ℝ, x > 0 ∧ log2 x = 0 := by
  -- Use 2 as the value of x that satisfies the condition
  use 1
  constructor
  · -- Prove 1 > 0
    exact one_pos
  · -- Prove log2 1 = 0
    unfold log2
    simp [Real.log_one]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_intersects_x_axis_l771_77144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l771_77103

theorem sin_shift_equivalence (x : ℝ) : 
  Real.sin (2 * (x - π / 12) + π / 6) = Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l771_77103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_at_21_l771_77190

/-- A function satisfying f(x + f(x)) = 4f(x) for all x, and f(1) = 3 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + f x) = 4 * f x) ∧ f 1 = 3

/-- If f is a special function, then f(21) = 192 -/
theorem special_function_at_21 (f : ℝ → ℝ) (h : special_function f) : f 21 = 192 := by
  sorry

#check special_function_at_21

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_at_21_l771_77190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_672_irrational_l771_77159

noncomputable def sequenceX (x : ℕ → ℝ) : Prop :=
  x 1 > 0 ∧ ∀ n : ℕ, x (n + 1) = Real.sqrt 5 * x n + 2 * Real.sqrt (x n ^ 2 + 1)

theorem at_least_672_irrational (x : ℕ → ℝ) (h : sequenceX x) :
  ∃ (S : Finset ℕ), S.card ≥ 672 ∧ (∀ n ∈ S, n ≤ 2016 ∧ Irrational (x n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_672_irrational_l771_77159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2008_is_6_l771_77136

/-- A function that checks if a two-digit number has exactly three distinct prime factors -/
def has_three_prime_factors (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ n = p * q * r

/-- Definition of our 2009-digit number as a function from its index to its digit -/
def digit_sequence : ℕ → ℕ := sorry

/-- The property that each pair of adjacent digits forms a two-digit number with three prime factors -/
axiom adjacent_digits_property : ∀ i : ℕ, i ≥ 1 → i ≤ 2007 → 
  has_three_prime_factors (digit_sequence i * 10 + digit_sequence (i + 1))

/-- The theorem stating that the 2008th digit must be 6 -/
theorem digit_2008_is_6 : digit_sequence 2008 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2008_is_6_l771_77136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l771_77192

theorem perpendicular_vectors_lambda (a b c : ℝ × ℝ) (l : ℝ) : 
  a = (1, 2) → 
  b = (2, 3) → 
  c = (-4, 6) → 
  (l • a.1 + b.1, l • a.2 + b.2) • c = 0 → 
  l = -5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l771_77192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_endpoint_product_l771_77148

/-- Definition of midpoint in 2D space -/
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

/-- Given that M(4,8) is the midpoint of line segment AB and A(1,-2) is one endpoint,
    prove that the product of the coordinates of point B is 126. -/
theorem midpoint_endpoint_product (A B M : ℝ × ℝ) : 
  A = (1, -2) → M = (4, 8) → is_midpoint M A B → (B.1 * B.2 = 126) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_endpoint_product_l771_77148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_dissection_theorem_l771_77166

/-- A rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- A dissection of a square into rectangles -/
def SquareDissection (m : ℕ) (rectangles : List Rectangle) : Prop :=
  rectangles.length = 5 ∧
  (rectangles.map (λ r => r.width) ++ rectangles.map (λ r => r.height)).toFinset = Finset.range 10 ∧
  (rectangles.map (λ r => r.width * r.height)).sum = m * m

/-- The theorem stating that only 11x11 and 13x13 squares can be dissected as required -/
theorem square_dissection_theorem :
  ∀ m : ℕ, (∃ rectangles : List Rectangle, SquareDissection m rectangles) ↔ m = 11 ∨ m = 13 := by
  sorry

#check square_dissection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_dissection_theorem_l771_77166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_36_l771_77111

/-- The sum of all positive divisors of 36 is 91 -/
theorem sum_of_divisors_36 : (Finset.filter (λ x => 36 % x = 0) (Finset.range 37)).sum id = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_36_l771_77111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_labeling_single_cycle_l771_77101

/-- Represents a tree with n vertices -/
structure TreeGraph (n : ℕ) where
  vertices : Finset (Fin n)
  edges : Finset (Fin n × Fin n)
  is_tree : sorry  -- Additional conditions to ensure it's a tree

/-- Represents a labeling of the tree vertices -/
def Labeling (n : ℕ) := Fin n → Fin n

/-- Represents a single label-swapping operation -/
def swap_labels (n : ℕ) (l : Labeling n) (e : Fin n × Fin n) : Labeling n :=
  sorry

/-- Performs n-1 label-swapping operations -/
def perform_swaps (n : ℕ) (t : TreeGraph n) (l : Labeling n) : Labeling n :=
  sorry

/-- Checks if a permutation consists of a single cycle -/
def is_single_cycle (n : ℕ) (p : Labeling n) : Prop :=
  sorry

theorem tree_labeling_single_cycle (n : ℕ) (t : TreeGraph n) (l : Labeling n) :
  is_single_cycle n (perform_swaps n t l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_labeling_single_cycle_l771_77101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l771_77164

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv y)) x - 5 * (deriv y x) + 6 * (y x) = 0

-- Define the general solution
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  C₁ * Real.exp (2 * x) + C₂ * Real.exp (3 * x)

-- Define the particular solution
noncomputable def particular_solution (x : ℝ) : ℝ :=
  Real.exp (2 * x)

-- Theorem statement
theorem differential_equation_solution :
  -- The general solution satisfies the differential equation
  (∀ C₁ C₂, diff_eq (general_solution C₁ C₂)) ∧
  -- The particular solution satisfies the differential equation
  (diff_eq particular_solution) ∧
  -- The particular solution satisfies the initial conditions
  (particular_solution 0 = 1 ∧ (deriv particular_solution) 0 = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l771_77164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_utube_height_difference_l771_77150

/-- Represents a U-tube with mercury and an additional liquid -/
structure UTube where
  crossSectionalArea : ℝ
  mercurySpecificGravity : ℝ
  liquidSpecificGravity : ℝ
  liquidHeight : ℝ

/-- Calculates the vertical height difference between liquid surfaces in a U-tube -/
noncomputable def heightDifference (tube : UTube) : ℝ :=
  (tube.liquidHeight * (tube.mercurySpecificGravity - tube.liquidSpecificGravity)) / tube.mercurySpecificGravity

/-- Theorem stating the vertical height difference in a U-tube -/
theorem utube_height_difference (tube : UTube) 
  (h1 : tube.crossSectionalArea > 0)
  (h2 : tube.mercurySpecificGravity > 0)
  (h3 : tube.liquidSpecificGravity > 0)
  (h4 : tube.liquidHeight > 0) :
  heightDifference tube = (tube.liquidHeight * (tube.mercurySpecificGravity - tube.liquidSpecificGravity)) / tube.mercurySpecificGravity :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_utube_height_difference_l771_77150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_twelfth_l771_77122

-- Define the parabola function
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define the tangent line function at x = 1
noncomputable def tangent_line (x : ℝ) : ℝ := 2*x - 1

-- Define the area of the figure
noncomputable def area : ℝ :=
  ∫ x in (0)..(1), f x - ∫ x in (1/2)..(1), tangent_line x

-- Theorem statement
theorem area_is_one_twelfth : area = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_twelfth_l771_77122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangency_concurrent_lines_l771_77116

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given circles and triangle
def S : Circle := sorry
def S1 : Circle := sorry
def S2 : Circle := sorry
def S3 : Circle := sorry

def A : Point := sorry
def B : Point := sorry
def C : Point := sorry

def A1 : Point := sorry
def B1 : Point := sorry
def C1 : Point := sorry

-- Define the conditions
def externally_tangent (c1 c2 : Circle) (p : Point) : Prop := sorry
def tangent_to_triangle_sides (c : Circle) (p1 p2 p3 : Point) : Prop := sorry

-- Define a function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Define the theorem
theorem circles_tangency_concurrent_lines 
  (h1 : externally_tangent S S1 A1)
  (h2 : externally_tangent S S2 B1)
  (h3 : externally_tangent S S3 C1)
  (h4 : tangent_to_triangle_sides S1 A B C)
  (h5 : tangent_to_triangle_sides S2 A B C)
  (h6 : tangent_to_triangle_sides S3 A B C) :
  ∃ (P : Point) (l1 l2 l3 : Line), 
    point_on_line A l1 ∧ point_on_line A1 l1 ∧ point_on_line P l1 ∧
    point_on_line B l2 ∧ point_on_line B1 l2 ∧ point_on_line P l2 ∧
    point_on_line C l3 ∧ point_on_line C1 l3 ∧ point_on_line P l3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangency_concurrent_lines_l771_77116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_lines_distance_l771_77171

-- Define the lines l₁ and l₂
def l₁ (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y + 1 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + y + a = 0

-- Part 1
theorem perpendicular_lines (a : ℝ) :
  (∀ x y, l₁ a 0 x y ↔ a * x + 1 = 0) →
  (∀ x y, l₂ a x y ↔ (a - 2) * x + y + a = 0) →
  (∀ x y, l₁ a 0 x y → l₂ a x y → (a * (a - 2) + 0 * 1 = 0)) →
  a = 2 :=
by
  sorry

-- Part 2
theorem parallel_lines_distance (a : ℝ) :
  (∀ x y, l₁ a 3 x y ↔ a * x + 3 * y + 1 = 0) →
  (∀ x y, l₂ a x y ↔ (a - 2) * x + y + a = 0) →
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y, a * x + 3 * y + 1 = k * ((a - 2) * x + y + a)) →
  let d := |1 - (a + 3)| / Real.sqrt (3^2 + 3^2)
  d = 4 * Real.sqrt 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_lines_distance_l771_77171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PACB_l771_77138

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x - 4 * y + 11 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define a point P on the line
def P : Type := {p : ℝ × ℝ // line_equation p.1 p.2}

-- Define the circle C with center P
def C (p : P) : Type := {c : ℝ × ℝ // circle_equation (c.1 - p.val.1) (c.2 - p.val.2)}

-- Define points A and B as intersections of the line and circle
axiom A : P → ℝ × ℝ
axiom B : P → ℝ × ℝ

-- Define the quadrilateral PACB
def quadrilateral_PACB (p : P) : Set (ℝ × ℝ) :=
  {x | x = p.val ∨ x = A p ∨ x = (1, 1) ∨ x = B p}

-- Define the area of the quadrilateral
noncomputable def area_PACB (p : P) : ℝ := sorry

-- Theorem statement
theorem min_area_PACB :
  ∃ (min_area : ℝ), min_area = Real.sqrt 3 ∧
  ∀ (p : P), area_PACB p ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PACB_l771_77138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l771_77129

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (-abs x)

-- Define a, b, and c
noncomputable def a : ℝ := f (Real.log (1/3))
noncomputable def b : ℝ := f (Real.log (1/Real.exp 1) / Real.log 3)
noncomputable def c : ℝ := f (Real.log 9 / Real.log (Real.exp 1))

-- State the theorem
theorem f_inequality : b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l771_77129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_perimeter_quadrilateral_ABCD_l771_77113

/-- The minimum perimeter of quadrilateral ABCD -/
noncomputable def min_perimeter_ABCD : ℝ := 8 + 2 * Real.sqrt 26

theorem minimum_perimeter_quadrilateral_ABCD :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 4)
  let area_ABCD : ℝ := 20
  min_perimeter_ABCD = 8 + 2 * Real.sqrt 26 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_perimeter_quadrilateral_ABCD_l771_77113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_period_l771_77137

/-- The minimum positive period of y = cos(2x) is π -/
theorem cos_2x_period : ∃ p > 0, ∀ x, Real.cos (2 * x) = Real.cos (2 * (x + p)) ∧ 
  ∀ q, 0 < q → q < p → ∃ x, Real.cos (2 * x) ≠ Real.cos (2 * (x + q)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_period_l771_77137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l771_77105

-- Define the ellipse
def C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

-- Define the maximum area of triangle PF₁F₂
def max_area_PF₁F₂ : ℝ := 1

-- Define the product of slopes
noncomputable def slope_product (x₀ y₀ : ℝ) : ℝ := 
  let k_PA := y₀ / (x₀ + Real.sqrt 2)
  let k_PB := y₀ / (x₀ - Real.sqrt 2)
  k_PA * k_PB

-- State the theorem
theorem ellipse_properties :
  ∀ x y : ℝ, C x y → 
    ∃ (F₁ F₂ A B : ℝ × ℝ), 
      eccentricity = Real.sqrt 2 / 2 ∧
      max_area_PF₁F₂ = 1 ∧
      slope_product x y = -1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l771_77105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_tan_phi_l771_77119

theorem right_triangle_tan_phi (β : Real) (φ : Real) : 
  -- Given conditions
  0 < β → β < π / 2 →  -- β is an acute angle
  Real.sin β = 1 / Real.sqrt 3 →  -- sin β = 1/√3
  φ = β / 2 →  -- φ is half of β (as explained in the solution)
  -- Conclusion
  Real.tan φ = Real.sqrt ((5 - 2 * Real.sqrt 6) / (5 + 2 * Real.sqrt 6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_tan_phi_l771_77119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l771_77147

/-- A circle with equation (x-4)^2 + (y-5)^2 = 10 -/
def myCircle (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 5)^2 = 10

/-- The line with equation 2x - y - 3 = 0 -/
def myLine (x y : ℝ) : Prop :=
  2*x - y - 3 = 0

theorem circle_properties :
  (∃ x y : ℝ, myLine x y ∧ myCircle x y) ∧  -- Center is on the line
  myCircle 5 2 ∧                            -- Passes through A(5,2)
  myCircle 3 2 :=                           -- Passes through B(3,2)
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l771_77147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_k_range_l771_77178

/-- The function f(x) = kx - ln x -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x - Real.log x

/-- Theorem: If f(x) = kx - ln x is monotonically increasing on (1, +∞), then k ∈ [1, +∞) -/
theorem monotone_increasing_f_implies_k_range (k : ℝ) :
  (∀ x y, 1 < x ∧ x < y → f k x < f k y) →
  k ∈ Set.Ici 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_k_range_l771_77178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_in_options_l771_77106

def options : List String := ["everyone else", "the other", "someone else", "the rest"]

def correctAnswer : String := "everyone else"

theorem correct_answer_in_options : correctAnswer ∈ options := by
  simp [options, correctAnswer]

#check correct_answer_in_options

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_in_options_l771_77106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_calculation_l771_77173

/-- The cost in dollars per kilogram for NASA Space Shuttle transport -/
noncomputable def cost_per_kg : ℚ := 25000

/-- The weight of the scientific instrument in grams -/
noncomputable def instrument_weight_g : ℚ := 500

/-- Conversion factor from grams to kilograms -/
noncomputable def g_to_kg : ℚ := 1000

/-- The cost of transporting the scientific instrument -/
noncomputable def transport_cost : ℚ := (instrument_weight_g / g_to_kg) * cost_per_kg

theorem transport_cost_calculation : transport_cost = 12500 := by
  -- Unfold the definitions
  unfold transport_cost cost_per_kg instrument_weight_g g_to_kg
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_calculation_l771_77173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l771_77196

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Given conditions for the geometric sequence -/
structure GeometricSequenceConditions where
  q : ℝ
  S_3 : ℝ
  S_5 : ℝ
  S_6 : ℝ
  h1 : S_6 = 9 * S_3
  h2 : S_5 = 62

/-- Theorem stating the first term of the geometric sequence -/
theorem geometric_sequence_first_term (c : GeometricSequenceConditions) :
  ∃ a : ℝ, 
    geometric_sum a c.q 3 = c.S_3 ∧ 
    geometric_sum a c.q 5 = c.S_5 ∧ 
    geometric_sum a c.q 6 = c.S_6 ∧ 
    a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l771_77196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_line_fits_l771_77170

noncomputable def point := ℝ × ℝ

noncomputable def trisection_points (p1 p2 : point) : List point :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let dx := (x2 - x1) / 3
  let dy := (y2 - y1) / 3
  [(x1 + dx, y1 + dy), (x1 + 2*dx, y1 + 2*dy)]

def line_equation (a b c : ℝ) (p : point) : Prop :=
  let (x, y) := p
  a * x + b * y + c = 0

theorem no_line_fits (p : point) (p1 p2 : point) : 
  let trisects := trisection_points p1 p2
  ¬ (∃ (a b c : ℝ), 
    (line_equation a b c p ∧ 
     (∃ t ∈ trisects, line_equation a b c t) ∧
     ((a = 3 ∧ b = -2 ∧ c = -1) ∨
      (a = 4 ∧ b = -5 ∧ c = 8) ∨
      (a = 5 ∧ b = 2 ∧ c = -23) ∨
      (a = 1 ∧ b = 7 ∧ c = -31) ∨
      (a = 1 ∧ b = -4 ∧ c = 13)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_line_fits_l771_77170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_reduction_with_speed_increase_l771_77146

theorem time_reduction_with_speed_increase :
  ∀ (distance : ℝ) (original_time : ℝ),
    original_time > 0 →
    distance = 42 * original_time →
    let increased_speed := 42 + 21;
    let new_time := distance / increased_speed;
    (original_time - new_time) / original_time = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_reduction_with_speed_increase_l771_77146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_lines_theorem_l771_77124

/-- Two lines in barycentric coordinates -/
structure BarycentricLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The determinant of a 2x2 matrix -/
def det2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- Intersection point of two lines -/
def intersection_point (l₁ l₂ : BarycentricLine) : Prod ℝ (Prod ℝ ℝ) :=
  (det2 l₁.b l₁.c l₂.b l₂.c, (det2 l₁.c l₁.a l₂.c l₂.a, det2 l₁.a l₁.b l₂.a l₂.b))

/-- Check if two lines are parallel -/
def are_parallel (l₁ l₂ : BarycentricLine) : Prop :=
  det2 l₁.b l₁.c l₂.b l₂.c + det2 l₁.c l₁.a l₂.c l₂.a + det2 l₁.a l₁.b l₂.a l₂.b = 0

/-- Theorem about intersection point of two lines in barycentric coordinates
    and the condition for parallel lines -/
theorem barycentric_lines_theorem (l₁ l₂ : BarycentricLine) :
  intersection_point l₁ l₂ = (det2 l₁.b l₁.c l₂.b l₂.c, (det2 l₁.c l₁.a l₂.c l₂.a, det2 l₁.a l₁.b l₂.a l₂.b))
  ∧
  are_parallel l₁ l₂ ↔ (det2 l₁.b l₁.c l₂.b l₂.c + det2 l₁.c l₁.a l₂.c l₂.a + det2 l₁.a l₁.b l₂.a l₂.b = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_lines_theorem_l771_77124
