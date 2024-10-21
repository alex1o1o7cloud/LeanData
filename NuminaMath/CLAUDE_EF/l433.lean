import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_level_conversation_effective_sound_pressure_classroom_l433_43379

/-- Sound pressure level calculation -/
noncomputable def sound_pressure_level (p_e : ℝ) (p_ref : ℝ) : ℝ :=
  20 * Real.log (p_e / p_ref) / Real.log 10

theorem sound_pressure_level_conversation (p_e p_ref : ℝ) :
  p_e = 0.002 → p_ref = 2e-5 → sound_pressure_level p_e p_ref = 40 := by
  sorry

theorem effective_sound_pressure_classroom (S_PL p_ref : ℝ) :
  S_PL = 90 → p_ref = 2e-5 → 
  ∃ p_e : ℝ, sound_pressure_level p_e p_ref = S_PL ∧ p_e = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_level_conversation_effective_sound_pressure_classroom_l433_43379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_x_intercept_l433_43355

/-- Given a line y = (2/3)x + 2 rotated counterclockwise by π/4 around (0, 2),
    the x-intercept of the resulting line is -2/5 -/
theorem rotated_line_x_intercept :
  let initial_slope : ℝ := 2/3
  let initial_y_intercept : ℝ := 2
  let rotation_angle : ℝ := π/4
  let rotation_point : ℝ × ℝ := (0, 2)
  let new_slope : ℝ := (initial_slope + Real.tan rotation_angle) / (1 - initial_slope * Real.tan rotation_angle)
  let new_line (x : ℝ) : ℝ := new_slope * x + initial_y_intercept
  ∃ x : ℝ, new_line x = 0 ∧ x = -2/5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_x_intercept_l433_43355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_6_and_distance_l433_43332

def z : ℕ → ℂ
  | 0 => 1
  | n + 1 => (z n)^2 - 1 + Complex.I

theorem z_6_and_distance : z 6 = -289 + 35 * Complex.I ∧ Complex.abs (z 6) = 291 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_6_and_distance_l433_43332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_vector_of_plane_l433_43358

def A : Fin 3 → ℝ := ![1, 0, 0]
def B : Fin 3 → ℝ := ![2, 0, 1]
def C : Fin 3 → ℝ := ![0, 1, 2]

def normal_vector : Fin 3 → ℝ := ![1, 3, -1]

def vector_subtract (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i => v i - w i

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

theorem normal_vector_of_plane :
  let AB := vector_subtract B A
  let AC := vector_subtract C A
  dot_product normal_vector AB = 0 ∧ dot_product normal_vector AC = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_vector_of_plane_l433_43358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_square_arrangement_l433_43382

-- Define the type for squares
inductive Square : Type
| One | Nine | A | B | C | D | E | F | G

-- Define the function that maps squares to their numbers
def square_number : Square → ℕ
| Square.One => 1
| Square.Nine => 9
| Square.A => 6
| Square.B => 2
| Square.C => 4
| Square.D => 5
| Square.E => 3
| Square.F => 8
| Square.G => 7

-- Define the sequence of squares
def square_sequence : List Square :=
  [Square.One, Square.B, Square.E, Square.C, Square.D, Square.A, Square.G, Square.F, Square.Nine]

-- Define a function to check if two squares are adjacent in the sequence
def are_adjacent (s1 s2 : Square) : Prop :=
  ∃ i, (square_sequence.get? i = some s1) ∧ (square_sequence.get? (i+1) = some s2)

-- Main theorem
theorem valid_square_arrangement :
  (∀ i : Fin 8, are_adjacent (square_sequence[i]) (square_sequence[i+1])) ∧
  (∀ s : Square, s ∈ square_sequence) ∧
  (∀ n : Fin 9, ∃ s ∈ square_sequence, square_number s = n.val + 1) :=
by
  sorry

#check valid_square_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_square_arrangement_l433_43382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_switch_setting_l433_43378

/-- Represents the effect of a switch on the time machine --/
def switchEffect (n : ℕ) (state : Bool) : ℤ :=
  if state then
    if n % 2 = 1 then (2 ^ (n - 1) : ℤ) else -(2 ^ (n - 1) : ℤ)
  else 0

/-- Calculates the total time change for a given switch setting --/
def totalTimeChange (setting : List Bool) : ℤ :=
  List.sum (List.zipWith switchEffect (List.range 10) setting)

/-- The theorem to be proved --/
theorem correct_switch_setting :
  totalTimeChange [false, false, false, true, false, false, true, false, true, true] = -200 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_switch_setting_l433_43378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_neg_half_slope_l433_43305

/-- Triangle ABO with given properties -/
structure TriangleABO where
  A : ℝ × ℝ
  B : ℝ × ℝ
  O : ℝ × ℝ
  right_angle : (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0
  ab_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 1
  bo_length : (B.1 - O.1)^2 + (B.2 - O.2)^2 = 1

/-- Point P on the triangle -/
noncomputable def P : ℝ × ℝ := (1/2, 1/4)

/-- Line MN passing through P with slope k -/
noncomputable def line_MN (k : ℝ) (x : ℝ) : ℝ :=
  k * (x - P.1) + P.2

/-- Area of triangle AMN given slope k -/
noncomputable def area_AMN (t : TriangleABO) (k : ℝ) : ℝ :=
  sorry  -- Area calculation would go here

/-- Theorem stating that the area is maximized when k = -1/2 -/
theorem max_area_at_neg_half_slope (t : TriangleABO) :
  ∀ k : ℝ, area_AMN t (-1/2) ≥ area_AMN t k := by
  sorry

#check max_area_at_neg_half_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_neg_half_slope_l433_43305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_approx_five_l433_43377

/-- The rate per meter for fencing a circular field -/
noncomputable def fencing_rate (diameter : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (Real.pi * diameter)

/-- Theorem: The fencing rate for a circular field with diameter 42 m and total cost Rs. 659.73 is approximately Rs. 5 per meter -/
theorem fencing_rate_approx_five :
  let diameter := (42 : ℝ)
  let total_cost := (659.73 : ℝ)
  abs (fencing_rate diameter total_cost - 5) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_approx_five_l433_43377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l433_43322

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the inverse function of f
def f_inv : ℝ → ℝ := sorry

-- State the theorem
theorem problem_statement :
  (∀ x : ℝ, ∃ y : ℝ, f y = x) →  -- f is surjective
  (∀ x y : ℝ, f x = f y → x = y) →  -- f is injective
  (∀ x : ℝ, f (f_inv x) = x) →  -- f_inv is the left inverse of f
  (∀ x : ℝ, f_inv (f x) = x) →  -- f_inv is the right inverse of f
  f 1 = 3 →
  f 2 = 4 →
  f 3 = 6 →
  f 4 = 8 →
  f 5 = 9 →
  f (f 2) + f (f_inv 6) + f_inv (f_inv 4) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l433_43322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l433_43347

def a : ℕ → ℚ
  | 0 => 2  -- Adding the case for 0
  | 1 => 2
  | (n + 2) => a (n + 1) + 1 / ((n + 1) * (n + 2))

theorem a_formula (n : ℕ) (h : n > 0) : a n = 3 - 1 / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l433_43347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_purchase_price_l433_43345

/-- Calculates the purchase price of an item given its sale price and loss percentage. -/
noncomputable def calculate_purchase_price (sale_price : ℝ) (loss_percentage : ℝ) : ℝ :=
  sale_price / (1 - loss_percentage / 100)

/-- Proves that the purchase price of a radio sold for Rs 465.50 with a 5% loss is Rs 490. -/
theorem radio_purchase_price :
  let sale_price : ℝ := 465.50
  let loss_percentage : ℝ := 5
  calculate_purchase_price sale_price loss_percentage = 490 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_purchase_price_l433_43345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l433_43348

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2 * x + 1 else Real.exp (Real.log 3 * x)

-- Define the set of m that satisfy the equation
def solution_set : Set ℝ :=
  {m : ℝ | f (f m) = Real.exp (Real.log 3 * f m)}

-- State the theorem
theorem solution_set_characterization :
  solution_set = Set.Ici 0 ∪ {-1/2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l433_43348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_28_l433_43366

def initial_list : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def first_removal : Nat := 4

def is_valid_removal (list : List Nat) (n : Nat) : Bool :=
  n ∈ list ∧ ∃ m ∈ list, m ≠ n ∧ m ∣ n

def game_step (current_list : List Nat) (player : Bool) : Option (Nat × List Nat) :=
  if player then
    -- Carolyn's turn
    current_list.find? (is_valid_removal current_list) |>.map (fun n => (n, current_list.filter (· ≠ n)))
  else
    -- Paul's turn
    let divisors := current_list.filter (fun d => ∃ n ∈ current_list, n ≠ d ∧ d ∣ n)
    if divisors.isEmpty then none
    else some (0, current_list.filter (fun n => ¬ (n ∈ divisors)))

def play_game (initial : List Nat) (first_move : Nat) : List Nat :=
  let rec aux (current_list : List Nat) (player : Bool) (carolyn_moves : List Nat) (fuel : Nat) : List Nat :=
    match fuel with
    | 0 => carolyn_moves
    | fuel + 1 =>
      if player then
        match game_step current_list player with
        | some (move, new_list) => aux new_list false (move :: carolyn_moves) fuel
        | none => carolyn_moves
      else
        match game_step current_list player with
        | some (_, new_list) => aux new_list true carolyn_moves fuel
        | none => carolyn_moves
  aux (initial.filter (· ≠ first_move)) false [first_move] (initial.length * 2)

theorem carolyn_sum_is_28 :
  (play_game initial_list first_removal).sum = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_28_l433_43366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_relationship_l433_43310

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem log_condition_relationship :
  (∀ x : ℝ, f (f x) > 0 → f x > 0) ∧
  (∃ x : ℝ, f x > 0 ∧ f (f x) ≤ 0) :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_relationship_l433_43310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_is_2x_l433_43383

/-- The remainder when dividing a polynomial by (x-2)(x-5) -/
noncomputable def remainder (p : ℝ → ℝ) : ℝ → ℝ :=
  fun x ↦ p x - (x - 2) * (x - 5) * ((p x - p 2) / (x - 2) - (p 5 - p 2) / (5 - 2))

/-- A polynomial satisfying the given conditions -/
axiom p : ℝ → ℝ

/-- The first condition: p(2) = 4 -/
axiom p_2 : p 2 = 4

/-- The second condition: p(5) = 10 -/
axiom p_5 : p 5 = 10

/-- Theorem stating that the remainder is 2x -/
theorem remainder_is_2x : remainder p = fun x ↦ 2 * x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_is_2x_l433_43383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l433_43326

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a
  else Real.log x / Real.log a

theorem increasing_function_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  (3/2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l433_43326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_lap_times_l433_43337

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℚ
  deriving Repr

/-- Represents two runners on a circular track -/
structure RaceScenario where
  nikita : Runner
  egor : Runner
  meetingCount : ℕ
  deriving Repr

/-- Checks if the race scenario satisfies the given conditions -/
def isValidScenario (scenario : RaceScenario) : Prop :=
  scenario.nikita.lapTime > 30 ∧
  scenario.egor.lapTime = scenario.nikita.lapTime + 12 ∧
  scenario.meetingCount = 7

/-- Theorem stating the correct lap times for Nikita and Egor -/
theorem correct_lap_times (scenario : RaceScenario) 
  (h : isValidScenario scenario) : 
  scenario.nikita.lapTime = 36 ∧ scenario.egor.lapTime = 48 := by
  sorry

#check correct_lap_times

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_lap_times_l433_43337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_length_is_20000_l433_43315

/-- The rise in feet required for the railroad line -/
def rise : ℝ := 800

/-- The initial grade as a percentage -/
def initial_grade : ℝ := 4

/-- The reduced grade as a percentage -/
def reduced_grade : ℝ := 2

/-- Calculate the horizontal length for a given grade percentage -/
noncomputable def horizontal_length (grade_percent : ℝ) : ℝ := rise / (grade_percent / 100)

/-- The additional length of track required to reduce the grade -/
noncomputable def additional_length : ℝ := horizontal_length reduced_grade - horizontal_length initial_grade

theorem additional_length_is_20000 : additional_length = 20000 := by
  -- Unfold definitions
  unfold additional_length
  unfold horizontal_length
  -- Simplify the expression
  simp [rise, initial_grade, reduced_grade]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_length_is_20000_l433_43315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sylvester_theorem_l433_43391

theorem sylvester_theorem (a b : ℕ) (h_coprime : Nat.Coprime a b) :
  -- Part 1: Unique solution for each c
  (∀ c : ℕ, ∃! p : ℕ × ℕ, p.1 < b ∧ a * p.1 + b * p.2 = c) ∧
  -- Part 2: Maximum c with no non-negative solutions
  (∀ k : ℕ, k > a * b - a - b → ∃ x y : ℕ, a * x + b * y = k) ∧
  (¬ ∃ x y : ℕ, a * x + b * y = a * b - a - b) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sylvester_theorem_l433_43391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_time_l433_43335

/-- Represents the time taken for a bus to cover a total distance of 250 km,
    partly at 40 kmph and partly at 60 kmph. -/
noncomputable def total_time (d1 : ℝ) : ℝ :=
  d1 / 40 + (250 - d1) / 60

/-- Theorem stating that the total time taken by the bus is equal to
    the sum of time taken at each speed. -/
theorem bus_journey_time (d1 : ℝ) (h1 : 0 ≤ d1) (h2 : d1 ≤ 250) :
  total_time d1 = d1 / 40 + (250 - d1) / 60 := by
  -- Unfold the definition of total_time
  unfold total_time
  -- The equality holds by definition
  rfl

#check bus_journey_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_time_l433_43335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_larger_segment_approx_80_l433_43398

/-- Triangle with sides a, b, c and altitude h to side c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ

/-- The larger segment of side c when altitude h is dropped --/
noncomputable def largerSegment (t : Triangle) : ℝ :=
  t.c - Real.sqrt (t.a^2 - t.h^2)

theorem triangle_larger_segment_approx_80 :
  ∃ t : Triangle, t.a = 50 ∧ t.b = 90 ∧ t.c = 110 ∧ 
  Int.floor (largerSegment t) = 80 := by
  sorry

#check triangle_larger_segment_approx_80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_larger_segment_approx_80_l433_43398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_exceptional_defeats_all_l433_43313

open Set

universe u

structure Athlete : Type u

-- Define the result of a match between two athletes
axiom defeats : Athlete → Athlete → Prop

-- Define what it means for an athlete to be exceptional
def exceptional (a : Athlete) (athletes : Set Athlete) : Prop :=
  ∀ b ∈ athletes, b ≠ a → defeats a b ∨ ∃ c ∈ athletes, defeats a c ∧ defeats c b

-- The main theorem
theorem unique_exceptional_defeats_all 
  (athletes : Set Athlete) 
  (h_finite : Finite athletes) 
  (h_nonempty : Set.Nonempty athletes)
  (h_all_play : ∀ a b : Athlete, a ∈ athletes → b ∈ athletes → a ≠ b → defeats a b ∨ defeats b a)
  (h_unique : ∃! a : Athlete, a ∈ athletes ∧ exceptional a athletes) :
  ∃ a : Athlete, a ∈ athletes ∧ ∀ b ∈ athletes, b ≠ a → defeats a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_exceptional_defeats_all_l433_43313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l433_43354

noncomputable def h (x : Real) : Real := (2 : Real) ^ x

noncomputable def f (x : Real) : Real := (1 - h x) / (1 + h x)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (h 2 = 4) ∧  -- given condition
  (∀ x, f x = (1 - (2 : Real) ^ x) / (1 + (2 : Real) ^ x)) ∧  -- analytic expression
  (∀ x, f (2*x - 1) > f (x + 1) ↔ x < 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l433_43354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l433_43321

-- Define the complex number z
noncomputable def z (k : ℝ) : ℂ := (1 + k * Complex.I) / (2 - Complex.I)

-- Part I
theorem part_one (k : ℝ) : z k = (1 : ℂ) / 2 → k = -1 / 2 := by
  sorry

-- Part II
theorem part_two (k : ℝ) : (z k).re = 0 ∧ (z k).im ≠ 0 → z k = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l433_43321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l433_43352

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2)
  (scalar * w.1, scalar * w.2)

theorem projection_property :
  let v₁ : ℝ × ℝ := (2, -1)
  let v₂ : ℝ × ℝ := (1, -1/2)
  let v₃ : ℝ × ℝ := (-3, 2)
  let v₄ : ℝ × ℝ := (-16/5, 8/5)
  projection v₁ v₂ = v₂ → projection v₃ v₂ = v₄ :=
by
  sorry

#check projection_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l433_43352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_function_properties_l433_43362

/-- A quadratic function with specific properties -/
structure GoldenFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  symmetric_points : (fun x ↦ a * x^2 + b * x + c) 1 = 4 ∧ (fun x ↦ a * x^2 + b * x + c) (-1) = -4
  axis_right_of_two : -b / (2 * a) > 2

/-- Theorem stating the properties of the golden function -/
theorem golden_function_properties (g : GoldenFunction) :
  g.a + g.c = 0 ∧ g.b = 4 ∧ -1 < g.a ∧ g.a < 0 := by
  sorry

#check golden_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_function_properties_l433_43362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l433_43373

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 3^(x + 1)

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := -1 + Real.log x / Real.log 3

-- Theorem statement
theorem inverse_function_theorem (x : ℝ) (hx : x > 0) :
  f (g x) = x ∧ g (f x) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l433_43373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l433_43357

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

/-- Theorem: The hyperbola x^2/4 - y^2/5 = 1 has an eccentricity of 3/2 -/
theorem hyperbola_eccentricity :
  eccentricity 2 (Real.sqrt 5) = 3/2 := by
  -- Unfold the definition of eccentricity
  unfold eccentricity
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The proof steps would go here, but for now we use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l433_43357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l433_43341

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l433_43341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_sum_l433_43311

/-- Represents a polygon with six vertices -/
structure Hexagon :=
  (P Q R S T U : ℝ × ℝ)

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculates the area of a polygon -/
noncomputable def polygonArea (h : Hexagon) : ℝ := sorry

/-- The main theorem -/
theorem hexagon_side_sum (h : Hexagon) 
  (area_eq : polygonArea h = 65)
  (pq_eq : distance h.P h.Q = 10)
  (qr_eq : distance h.Q h.R = 7)
  (tu_eq : distance h.T h.U = 6) :
  distance h.S h.T + distance h.S h.U = 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_sum_l433_43311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_plus_b_over_3_l433_43361

-- Define the function y in terms of x, a, and b
noncomputable def y (x a b : ℝ) : ℝ := a + b / x

-- State the theorem
theorem value_of_a_plus_b_over_3 (a b : ℝ) :
  y (-2) a b = 3 ∧ y (-6) a b = 7 → a + b / 3 = 13 := by
  intro h
  sorry

#check value_of_a_plus_b_over_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_plus_b_over_3_l433_43361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l433_43324

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The focal length of an ellipse -/
noncomputable def Ellipse.focalLength (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The number of isosceles right triangles with vertex A on the positive y-axis -/
def numIsoscelesRightTriangles (e : Ellipse) : ℕ :=
  sorry

/-- The main theorem about the specific ellipse -/
theorem ellipse_theorem (e : Ellipse)
  (h_focal : e.focalLength = 2 * Real.sqrt 3)
  (h_point : ∃ p : PointOnEllipse e, p.x = 1 ∧ p.y = Real.sqrt 3 / 2) :
  e.a^2 = 4 ∧ e.b^2 = 1 ∧ numIsoscelesRightTriangles e = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l433_43324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_minimizes_sum_squares_to_vertices_special_point_minimizes_sum_squares_to_sides_l433_43384

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  (((t.A).1 + (t.B).1 + (t.C).1) / 3, ((t.A).2 + (t.B).2 + (t.C).2) / 3)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the sum of squares of distances from a point to triangle vertices
noncomputable def sumSquaresDistancesToVertices (t : Triangle) (p : ℝ × ℝ) : ℝ :=
  (distance p t.A)^2 + (distance p t.B)^2 + (distance p t.C)^2

-- Define the distance from a point to a line (side of the triangle)
noncomputable def distanceToLine (p : ℝ × ℝ) (l1 l2 : ℝ × ℝ) : ℝ :=
  sorry -- Actual implementation would go here

-- Define the sum of squares of distances from a point to triangle sides
noncomputable def sumSquaresDistancesToSides (t : Triangle) (p : ℝ × ℝ) : ℝ :=
  (distanceToLine p t.B t.C)^2 + (distanceToLine p t.C t.A)^2 + (distanceToLine p t.A t.B)^2

-- Theorem 1: The centroid minimizes the sum of squares of distances to vertices
theorem centroid_minimizes_sum_squares_to_vertices (t : Triangle) :
  ∀ p, sumSquaresDistancesToVertices t (centroid t) ≤ sumSquaresDistancesToVertices t p := by
  sorry

-- Define a point where distances to sides are proportional to side lengths
noncomputable def special_point (t : Triangle) : ℝ × ℝ :=
  sorry -- Actual implementation would go here

-- Theorem 2: The special point minimizes the sum of squares of distances to sides
theorem special_point_minimizes_sum_squares_to_sides (t : Triangle) :
  ∀ p, sumSquaresDistancesToSides t (special_point t) ≤ sumSquaresDistancesToSides t p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_minimizes_sum_squares_to_vertices_special_point_minimizes_sum_squares_to_sides_l433_43384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l433_43342

/-- The number of pieces in a complete pack of gum -/
def x : ℕ := sorry

/-- The number of pieces of cherry gum Chewbacca has -/
def cherry_gum : ℕ := 28

/-- The number of pieces of grape gum Chewbacca has -/
def grape_gum : ℕ := 36

/-- The ratio equality condition -/
axiom ratio_equality : (cherry_gum - 2 * x) / grape_gum = cherry_gum / (grape_gum + 4 * x)

theorem gum_pack_size : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l433_43342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l433_43301

/-- Predicate to check if a point is the focus of the parabola y² = 8x -/
def is_focus_of_parabola (f : ℝ) : Prop :=
  ∀ (x y : ℝ), y^2 = 8*x → (x - f)^2 + y^2 = (x + f)^2

/-- Predicate to check if a line is the directrix of the parabola y² = 8x -/
def is_directrix_of_parabola (d : ℝ) : Prop :=
  ∀ (x y : ℝ), y^2 = 8*x → abs (x - d) = abs (x - 2)

/-- For a parabola y² = 8x, the distance from its focus to its directrix is 4 -/
theorem parabola_focus_directrix_distance :
  ∃ (f d : ℝ), is_focus_of_parabola f ∧ is_directrix_of_parabola d ∧ abs (f - d) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l433_43301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_stops_in_first_quarter_l433_43363

noncomputable def track_circumference : ℝ := 200
noncomputable def total_distance : ℝ := 15840
noncomputable def quarter_length : ℝ := track_circumference / 4

noncomputable def position_after_run : ℝ := total_distance % track_circumference

theorem runner_stops_in_first_quarter : 
  0 < position_after_run ∧ position_after_run ≤ quarter_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_stops_in_first_quarter_l433_43363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resulting_expr_undefined_at_two_l433_43306

noncomputable def original_expr (x : ℝ) : ℝ := (x + 2) / (x - 2)

noncomputable def resulting_expr (x : ℝ) : ℝ := (original_expr x + 2) / (original_expr x - 2)

theorem resulting_expr_undefined_at_two : 
  ¬ ∃ (y : ℝ), resulting_expr 2 = y :=
by
  intro h
  cases' h with y hy
  simp [resulting_expr, original_expr] at hy
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_resulting_expr_undefined_at_two_l433_43306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l433_43323

noncomputable def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  simp [g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l433_43323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_lateral_face_l433_43330

/-- Represents a regular triangular pyramid. -/
structure RegularTriangularPyramid where
  /-- The dihedral angle at the base of the pyramid. -/
  dihedralAngleAtBase : ℝ
  /-- The lateral surface area of the pyramid. -/
  lateralSurfaceArea : ℝ
  /-- The distance from the center of the base to a lateral face. -/
  distanceCenterToLateralFace : ℝ

/-- In a regular triangular pyramid with dihedral angle α at the base and lateral surface area S,
    the distance from the center of the base to the lateral face is (sin α / 3) * √(S * √3 * cos α). -/
theorem distance_center_to_lateral_face (α S : ℝ) :
  let d := (Real.sin α / 3) * Real.sqrt (S * Real.sqrt 3 * Real.cos α)
  ∃ (pyramid : RegularTriangularPyramid),
    pyramid.dihedralAngleAtBase = α ∧
    pyramid.lateralSurfaceArea = S ∧
    pyramid.distanceCenterToLateralFace = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_lateral_face_l433_43330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_nested_radical_equation_l433_43353

theorem solve_nested_radical_equation :
  ∃ x : ℝ, (Real.sqrt (9 + Real.sqrt (27 + 9*x)) + Real.sqrt (6 + Real.sqrt (9 + 3*x))) = 3 + 3*(Real.sqrt 3) :=
by
  use 48
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_nested_radical_equation_l433_43353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_thousandth_l433_43350

/-- Rounds a real number to the nearest thousandth -/
noncomputable def round_to_thousandth (x : ℝ) : ℝ := 
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

/-- The sum of 78.621 and 34.0568, when rounded to the nearest thousandth, equals 112.678 -/
theorem sum_and_round_to_thousandth : 
  round_to_thousandth (78.621 + 34.0568) = 112.678 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_thousandth_l433_43350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_person_hired_prob_two_people_hired_l433_43303

/-- Represents the probability of selecting k items from n items -/
def probability (n k : ℕ) : ℚ := (n.choose k : ℚ) / (n ^ k : ℚ)

/-- The number of job seekers -/
def total_seekers : ℕ := 5

/-- The number of available positions -/
def available_positions : ℕ := 2

/-- Theorem: Probability of one specific person getting a position -/
theorem prob_one_person_hired :
  probability total_seekers available_positions = 2 / 5 := by sorry

/-- Theorem: Probability of either of two specific people getting a position -/
theorem prob_two_people_hired :
  probability total_seekers available_positions * 2 - 
  (probability total_seekers available_positions)^2 = 7 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_person_hired_prob_two_people_hired_l433_43303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_product_greater_than_one_l433_43369

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |1 - 1/x|

-- State the theorem
theorem function_equality_implies_product_greater_than_one 
  (a b : ℝ) (ha : 0 < a) (hb : a < b) (hf : f a = f b) : a * b > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_product_greater_than_one_l433_43369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_slide_all_off_l433_43325

-- Define a structure for a convex polygon
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool
  nonempty : vertices.length > 0

-- Define a type for the rectangular table
structure RectangularTable where
  width : Real
  height : Real
  positive_dimensions : width > 0 ∧ height > 0

-- Define a function to check if two polygons overlap
noncomputable def polygons_overlap (p1 p2 : ConvexPolygon) : Bool := sorry

-- Define a function to check if a polygon is on the table
noncomputable def polygon_on_table (p : ConvexPolygon) (table : RectangularTable) : Bool := sorry

-- Define a function to check if a polygon can be slid off the table without intersecting others
noncomputable def can_slide_off (p : ConvexPolygon) (others : List ConvexPolygon) (table : RectangularTable) : Bool := sorry

-- Main theorem
theorem can_slide_all_off (polygons : List ConvexPolygon) (table : RectangularTable) 
  (h1 : ∀ p, p ∈ polygons → polygon_on_table p table)
  (h2 : ∀ p1 p2, p1 ∈ polygons → p2 ∈ polygons → p1 ≠ p2 → ¬polygons_overlap p1 p2) :
  ∃ (sequence : List ConvexPolygon), 
    sequence.length = polygons.length ∧
    (∀ p, p ∈ polygons → p ∈ sequence) ∧
    (∀ i, i < sequence.length → can_slide_off (sequence.get ⟨i, sorry⟩) (sequence.drop (i+1)) table) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_slide_all_off_l433_43325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wilsons_initial_balance_l433_43328

/-- Represents the amount of money in Wilson's bank account -/
structure BankAccount where
  balance : Int

/-- Wilson's initial balance two months ago -/
def initial_balance : BankAccount := ⟨0⟩

/-- Deposit made last month -/
def deposit1 : BankAccount := ⟨17⟩

/-- Withdrawal made a few days after the first deposit -/
def withdrawal : BankAccount := ⟨0⟩

/-- Deposit made this month -/
def deposit2 : BankAccount := ⟨21⟩

/-- The difference between the current balance and the initial balance -/
def balance_difference : BankAccount := ⟨16⟩

/-- Addition operation for BankAccount -/
instance : Add BankAccount where
  add a b := ⟨a.balance + b.balance⟩

/-- Subtraction operation for BankAccount -/
instance : Sub BankAccount where
  sub a b := ⟨a.balance - b.balance⟩

theorem wilsons_initial_balance :
  initial_balance = ⟨16⟩ :=
by
  have h1 : initial_balance + deposit1 - withdrawal + deposit2 = initial_balance + balance_difference :=
    sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wilsons_initial_balance_l433_43328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l433_43372

theorem infinitely_many_solutions (a : ℤ) :
  (∃ (S : Set (ℤ × ℤ)), Set.Infinite S ∧ 
    ∀ (p : ℤ × ℤ), p ∈ S → (let (x, y) := p; x^2 + a*x*y + y^2 = 1)) ↔ 
  |a| ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l433_43372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l433_43317

def sequenceA (n : ℕ) : ℚ :=
  n / (2 ^ (n - 1))

theorem sequence_formula (a : ℕ → ℚ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = (n + 1) / (2 * n)) :
  ∀ n : ℕ, n ≥ 1 → a n = sequenceA n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l433_43317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l433_43338

theorem expression_equality : (3 - Real.pi)^0 - 3^(-2 : ℤ) + |Real.sqrt 3 - 2| + 2 * Real.sin (60 * π / 180) = 26 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l433_43338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_jumps_l433_43359

/-- Represents a 3x3 grid --/
def Grid := Fin 3 × Fin 3

/-- The center position of the grid --/
def center : Grid := (1, 1)

/-- Checks if a position is a corner --/
def is_corner (pos : Grid) : Prop :=
  (pos.1 = 0 ∨ pos.1 = 2) ∧ (pos.2 = 0 ∨ pos.2 = 2)

/-- Represents a single jump of the frog --/
def jump (pos : Grid) : Grid :=
  sorry -- Implementation details omitted

/-- The probability of jumping to an adjacent square --/
noncomputable def jump_prob : ℝ := 1 / 3

/-- Expected number of jumps theorem --/
theorem expected_jumps :
  ∃ (E : ℝ), E = 3 ∧
  (∀ (n : ℕ), 
    let f : ℕ → ℝ := λ m => (2 / 3) * (jump_prob ^ (m - 1))
    E = ∑' m, 2 * m * f m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_jumps_l433_43359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_p_q_l433_43367

/-- A quadratic polynomial -/
def q : ℝ → ℝ := sorry

/-- A linear polynomial -/
def p : ℝ → ℝ := sorry

/-- The graph of p(x)/q(x) has a vertical asymptote at x=1 -/
axiom asymptote : ∃ (k : ℝ), ∀ (x : ℝ), x ≠ 1 → |p x / q x| ≤ k * |x - 1|⁻¹

/-- The graph of p(x)/q(x) has a hole at x=0 -/
axiom hole : p 0 = 0 ∧ q 0 = 0

/-- p(1) = 1 -/
axiom p_at_one : p 1 = 1

/-- q(3) = 3 -/
axiom q_at_three : q 3 = 3

/-- The sum of p(x) and q(x) equals (1/2)x^2 + (1/2)x -/
theorem sum_p_q : ∀ (x : ℝ), p x + q x = (1/2) * x^2 + (1/2) * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_p_q_l433_43367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_l433_43351

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = π
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c

-- State the theorem
theorem angle_A_measure (t : Triangle) 
  (h : 1 + Real.tan t.A / Real.tan t.B = 2 * t.c / t.b) : 
  t.A = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_l433_43351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_expansion_l433_43314

theorem cylinder_volume_expansion (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let V_original := π * r^2 * h
  let r_new := 3 * r
  let V_new := π * r_new^2 * h
  V_new = 9 * V_original := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_expansion_l433_43314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_power_criterion_l433_43304

/-- Given integers b and n greater than 1, if for every k > 1 there exists an integer a_k
    such that k divides b - a_k^n, then b is an n-th power. -/
theorem nth_power_criterion (b n : ℕ) (h_b : b > 1) (h_n : n > 1)
  (h : ∀ k : ℕ, k > 1 → ∃ a_k : ℕ, k ∣ (b - a_k^n)) :
  ∃ A : ℕ, b = A^n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_power_criterion_l433_43304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l433_43387

theorem expression_value : 
  Real.rpow 64 (-1/3) + Real.log 0.001 = -11/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l433_43387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_0_to_99_l433_43331

def sum_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_digits (n / 10)

def sum_range_digits (a b : ℕ) : ℕ :=
  (List.range (b - a + 1)).map (fun i => sum_digits (a + i)) |>.sum

theorem sum_digits_0_to_99 :
  (sum_range_digits 0 99 = 900) ∧ (sum_range_digits 18 21 = 24) := by
  sorry

#eval sum_range_digits 0 99
#eval sum_range_digits 18 21

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_0_to_99_l433_43331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_slope_and_intercept_l433_43392

/-- A linear function f: ℝ → ℝ defined as f(x) = 3x + 2 -/
def f (x : ℝ) : ℝ := 3 * x + 2

/-- The slope of a linear function f(x) = mx + b is m -/
def get_slope (f : ℝ → ℝ) : ℝ := sorry

/-- The y-intercept of a linear function f(x) = mx + b is b -/
def get_y_intercept (f : ℝ → ℝ) : ℝ := sorry

theorem linear_function_slope_and_intercept :
  get_slope f = 3 ∧ get_y_intercept f = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_slope_and_intercept_l433_43392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fuse_length_l433_43360

/-- Calculates the minimum fuse length required for safe blasting in a mine. -/
theorem min_fuse_length 
  (safe_distance : ℝ) 
  (fuse_speed : ℝ) 
  (person_speed : ℝ) 
  (h1 : safe_distance ≥ 300)
  (h2 : fuse_speed = 0.8)
  (h3 : person_speed = 5) : 
  safe_distance / person_speed * fuse_speed ≥ 48 := by
  sorry

-- Use 'noncomputable' for the evaluation
noncomputable def min_fuse_length_calc : ℝ :=
  (300 : ℝ) / 5 * 0.8

-- Use '#eval' with 'Float' for approximate computation
#eval (300 : Float) / 5 * 0.8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fuse_length_l433_43360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_increase_theorem_l433_43308

noncomputable def average (scores : List ℝ) : ℝ := scores.sum / scores.length

noncomputable def standardDeviation (scores : List ℝ) : ℝ :=
  Real.sqrt ((scores.map (λ x => (x - average scores)^2)).sum / scores.length)

theorem score_increase_theorem (scores : List ℝ) 
  (h_count : scores.length = 54)
  (h_avg : average scores = 90)
  (h_std : standardDeviation scores = 4) :
  let newScores := scores.map (λ x => x + 5)
  average newScores + standardDeviation newScores = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_increase_theorem_l433_43308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_restricted_domain_full_l433_43309

/-- The function f(x) defined in terms of m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt ((m^2 + m - 2) * x^2 + (m - 1) * x + 4)

/-- Theorem for the value of m when the domain is [-2,1] -/
theorem domain_restricted (m : ℝ) : 
  (∀ x, x ∈ Set.Icc (-2) 1 → f m x ∈ Set.range Real.sqrt) → m = -1 :=
sorry

/-- Theorem for the range of m when the domain is ℝ -/
theorem domain_full (m : ℝ) : 
  (∀ x : ℝ, f m x ∈ Set.range Real.sqrt) ↔ m ∈ Set.Iic (-11/5) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_restricted_domain_full_l433_43309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_32_16_l433_43334

theorem log_32_16 : Real.log 16 / Real.log 32 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_32_16_l433_43334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_for_given_difference_l433_43394

noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem interest_rate_for_given_difference (principal : ℝ) (time : ℝ) (difference : ℝ) :
  principal = 1000 →
  time = 4 →
  difference = 64.10 →
  ∃ (rate : ℝ), rate = 10 ∧ 
    compoundInterest principal rate time - simpleInterest principal rate time = difference :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_for_given_difference_l433_43394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_values_given_cosine_l433_43374

theorem sine_values_given_cosine (θ : ℝ) 
  (h1 : θ > 0) 
  (h2 : θ < π / 2) 
  (h3 : Real.cos (θ + π / 6) = 1 / 3) : 
  Real.sin θ = (2 * Real.sqrt 6 - 1) / 6 ∧ 
  Real.sin (2 * θ + π / 6) = (4 * Real.sqrt 6 + 7) / 18 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_values_given_cosine_l433_43374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fraction_not_in_triangle_l433_43336

/-- A square with side length s -/
structure Square (s : ℝ) where
  side : s > 0

/-- Point P is the midpoint of the bottom side -/
noncomputable def P (s : ℝ) : ℝ × ℝ := (s / 2, 0)

/-- Point Q is the midpoint of the right side -/
noncomputable def Q (s : ℝ) : ℝ × ℝ := (s, s / 2)

/-- Top-left corner of the square -/
noncomputable def TopLeft (s : ℝ) : ℝ × ℝ := (0, s)

/-- Area of the triangle formed by P, Q, and the top-left corner -/
noncomputable def TriangleArea (s : ℝ) : ℝ := (s / 2) * (s / 2) / 2

/-- Area of the square -/
noncomputable def SquareArea (s : ℝ) : ℝ := s * s

/-- Theorem: The fraction of the square's area not in the triangle is 7/8 -/
theorem square_fraction_not_in_triangle (s : ℝ) (sq : Square s) :
  (SquareArea s - TriangleArea s) / SquareArea s = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fraction_not_in_triangle_l433_43336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_unique_property_l433_43393

noncomputable def has_minimum_period_pi (f : ℝ → ℝ) : Prop :=
  ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧ (∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q) ∧ p = Real.pi

def monotonically_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem abs_sin_unique_property :
  has_minimum_period_pi (fun x ↦ |Real.sin x|) ∧
  monotonically_decreasing_on_interval (fun x ↦ |Real.sin x|) (Real.pi/2) Real.pi ∧
  (¬(has_minimum_period_pi Real.cos ∧ monotonically_decreasing_on_interval Real.cos (Real.pi/2) Real.pi)) ∧
  (¬(has_minimum_period_pi Real.tan ∧ monotonically_decreasing_on_interval Real.tan (Real.pi/2) Real.pi)) ∧
  (¬(has_minimum_period_pi (fun x ↦ Real.cos (x/2)) ∧ monotonically_decreasing_on_interval (fun x ↦ Real.cos (x/2)) (Real.pi/2) Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_unique_property_l433_43393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_fold_theorem_l433_43302

-- Define the triangle
structure Triangle :=
  (base : ℝ)
  (area : ℝ)

-- Define the creases
structure Crease :=
  (length : ℝ)

-- Define the problem
def triangular_fold_problem (ABC : Triangle) (DE FG : Crease) : Prop :=
  ABC.base = 15 ∧
  -- DE is parallel to the base (implied by the problem)
  -- FG is parallel to DE (implied by the problem)
  ∃ (area_between_DE_FG : ℝ) (area_below_DE : ℝ),
    area_between_DE_FG = 0.25 * ABC.area ∧
    area_below_DE = 0.36 * ABC.area ∧
    DE.length = 7.5

-- Theorem statement
theorem triangular_fold_theorem (ABC : Triangle) (DE FG : Crease) :
  triangular_fold_problem ABC DE FG :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_fold_theorem_l433_43302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_solution_l433_43327

/-- The number of students in class 4 -/
def num_students : ℕ := sorry

/-- The total amount of money available -/
def total_money : ℕ := sorry

/-- The condition that if each notebook costs 3 yuan, 6 more notebooks can be bought -/
axiom condition_3yuan : total_money = 3 * num_students + 3 * 6

/-- The condition that if each notebook costs 5 yuan, there is a 30-yuan shortfall -/
axiom condition_5yuan : total_money = 5 * num_students - 30

/-- The condition that the sum of money is exactly used to buy one notebook for each student -/
axiom exact_purchase : ∃ (num_3yuan : ℕ), 
  num_3yuan ≤ num_students ∧
  3 * num_3yuan + 5 * (num_students - num_3yuan) = total_money

/-- The theorem stating the correct number of students and 3-yuan notebooks -/
theorem correct_solution : num_students = 24 ∧ ∃ (num_3yuan : ℕ), num_3yuan = 15 := by
  sorry

#check correct_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_solution_l433_43327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_g_minimum_value_l433_43307

/-- The function f(x) = 2x³ + 12x -/
noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 12 * x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := 6 * x^2 + 12

/-- The function g(x) = f(x)/x² -/
noncomputable def g (x : ℝ) : ℝ := f x / x^2

theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

theorem f'_at_one : f' 1 = 18 := by sorry

theorem g_minimum_value : 
  ∀ x > 0, g x ≥ 4 * Real.sqrt 6 ∧ ∃ x > 0, g x = 4 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_g_minimum_value_l433_43307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_is_correct_l433_43380

/-- The volume of a right circular cone with base edge length 2 and side edge length 4√3/3 -/
noncomputable def cone_volume : ℝ :=
  let base_edge := 2
  let side_edge := 4 * Real.sqrt 3 / 3
  2 * Real.sqrt 3 / 3

/-- Theorem stating that the volume of the specified cone is 2√3/3 -/
theorem cone_volume_is_correct : cone_volume = 2 * Real.sqrt 3 / 3 := by
  -- Unfold the definition of cone_volume
  unfold cone_volume
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_is_correct_l433_43380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l433_43344

/-- The area of a triangle ABC with vertices A(2, -1), B(3, 1), and C(2^1999, 2^2000) is 2.5 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, -1)
  let B : ℝ × ℝ := (3, 1)
  let C : ℝ × ℝ := (2^1999, 2^2000)
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  triangle_area = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l433_43344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_max_height_l433_43343

/-- The height of the projectile as a function of time -/
def projectile_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 161

/-- Theorem stating that the height of the projectile is always less than or equal to the maximum height -/
theorem projectile_max_height :
  ∀ t : ℝ, projectile_height t ≤ max_height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_max_height_l433_43343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l433_43318

theorem order_of_expressions : 
  Real.log (1/2) < (1/3 : Real) ^ (0.8 : Real) ∧ (1/3 : Real) ^ (0.8 : Real) < 2 ^ (1/3 : Real) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l433_43318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_relation_l433_43388

/-- A circle in 2D space -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- Two circles are externally tangent -/
def IsExternallyTangent (C₁ C₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ C₁ ∧ p ∈ C₂ ∧ 
  ∀ q : ℝ × ℝ, q ≠ p → (q ∈ C₁ → q ∉ C₂) ∧ (q ∈ C₂ → q ∉ C₁)

/-- A line is tangent to a circle -/
def IsTangentLine (C : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ C ∧ p ∈ l ∧ 
  ∀ q : ℝ × ℝ, q ≠ p → (q ∈ C → q ∉ l) ∧ (q ∈ l → q ∉ C)

/-- Given three circles C₁, C₂, and C₃ with radii r₁, r₂, and r₃ respectively,
    where C₁ and C₂ touch externally, C₃ is externally tangent to both C₁ and C₂,
    and all three circles are tangent to a common line l,
    prove that 1/√r₃ = 1/√r₁ + 1/√r₂ -/
theorem circle_tangency_relation (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₃ > 0) 
  (h_min : r₃ < min r₁ r₂) 
  (h_tangent : ∃ (l : Set (ℝ × ℝ)) (C₁ C₂ C₃ : Set (ℝ × ℝ)), 
    C₁ = Circle (0, 0) r₁ ∧
    C₂ = Circle (r₁ + r₂, 0) r₂ ∧
    IsExternallyTangent C₁ C₂ ∧ 
    IsExternallyTangent C₁ C₃ ∧ 
    IsExternallyTangent C₂ C₃ ∧
    IsTangentLine C₁ l ∧ 
    IsTangentLine C₂ l ∧ 
    IsTangentLine C₃ l) :
  1 / Real.sqrt r₃ = 1 / Real.sqrt r₁ + 1 / Real.sqrt r₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_relation_l433_43388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_orthocenter_triangle_equal_l433_43396

/-- A point is an interior point of a triangle if it's inside the triangle. -/
def is_interior_point (P A B C : ℝ × ℝ) : Prop :=
  sorry

/-- A point is the orthocenter of a triangle if it's the intersection of the three altitudes of the triangle. -/
def is_orthocenter (H P Q R : ℝ × ℝ) : Prop :=
  sorry

/-- The area of a triangle given its three vertices. -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  sorry

/-- Given a triangle ABC with an interior point P, and H_A, H_B, H_C as the orthocenters of triangles PBC, PAC, and PAB respectively, the area of triangle H_A H_B H_C is equal to the area of triangle ABC. -/
theorem area_orthocenter_triangle_equal (A B C P H_A H_B H_C : ℝ × ℝ) : 
  is_interior_point P A B C →
  is_orthocenter H_A P B C →
  is_orthocenter H_B P A C →
  is_orthocenter H_C P A B →
  area_triangle H_A H_B H_C = area_triangle A B C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_orthocenter_triangle_equal_l433_43396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_freshman_correct_l433_43385

/-- The number of freshman volunteers -/
def num_freshmen : ℕ := 2

/-- The number of sophomore volunteers -/
def num_sophomores : ℕ := 4

/-- The total number of volunteers -/
def total_volunteers : ℕ := num_freshmen + num_sophomores

/-- The number of volunteers to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting at least one freshman volunteer -/
def prob_at_least_one_freshman : ℚ := 3 / 5

theorem prob_at_least_one_freshman_correct :
  (Nat.choose total_volunteers num_selected - Nat.choose num_sophomores num_selected : ℚ) /
  (Nat.choose total_volunteers num_selected : ℚ) = prob_at_least_one_freshman := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_freshman_correct_l433_43385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_range_l433_43340

/-- The cost of one book in dollars -/
def book_cost : ℝ := sorry

/-- Eleven books cost less than $15.00 -/
axiom eleven_books_cost : 11 * book_cost < 15

/-- Twelve books cost more than $16.20 -/
axiom twelve_books_cost : 12 * book_cost > 16.20

/-- The cost of one book is between $1.35 and $1.37 -/
theorem book_cost_range : 1.35 < book_cost ∧ book_cost < 1.37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_range_l433_43340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l433_43316

theorem exponential_inequality (a b : ℝ) : a > b → (2 : ℝ)^a > (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l433_43316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_15_value_l433_43339

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2
  | n + 1 => (Real.sqrt (sequence_a n - 1) + 1)^2 + 1

theorem a_15_value : sequence_a 15 = 226 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_15_value_l433_43339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_always_positive_implies_a_range_l433_43300

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x + 3) + 1 / (a * x + 2)

theorem function_always_positive_implies_a_range (a : ℝ) :
  (∀ x ≥ -3, f a x > 0) → 0 < a ∧ a < 2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_always_positive_implies_a_range_l433_43300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_form_l433_43395

-- Define the points
def P : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (3, 4)
def R : ℝ × ℝ := (6, 0)
def S : ℝ × ℝ := (9, 4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter
noncomputable def perimeter : ℝ := 
  distance P Q + distance Q R + distance R S + distance S P

-- Theorem statement
theorem perimeter_form : 
  ∃ (c d : ℕ), perimeter = c * Real.sqrt d ∧ c + d = 114 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_form_l433_43395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_negative_eighteen_l433_43333

/-- A function f(x) defined as x^2 / (Ax^2 + Bx + C) -/
noncomputable def f (A B C : ℤ) : ℝ → ℝ := λ x => x^2 / (A * x^2 + B * x + C)

/-- Theorem stating that if f(x) > 0.5 for all x > 5, then A + B + C = -18 -/
theorem sum_of_coefficients_is_negative_eighteen
  (A B C : ℤ)
  (h : ∀ x > 5, f A B C x > (1/2 : ℝ)) :
  A + B + C = -18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_negative_eighteen_l433_43333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_fraction_used_l433_43349

-- Define the car's properties and travel conditions
def speed : ℝ := 50
def fuel_efficiency : ℝ := 30
def full_tank : ℝ := 15
def travel_time : ℝ := 5

-- Theorem statement
theorem gasoline_fraction_used :
  (speed * travel_time / fuel_efficiency) / full_tank = 5 / 9 := by
  -- Calculate the distance traveled
  have distance : ℝ := speed * travel_time
  
  -- Calculate the gallons of gasoline used
  have gallons_used : ℝ := distance / fuel_efficiency
  
  -- Calculate the fraction of the full tank used
  have fraction_used : ℝ := gallons_used / full_tank
  
  -- Prove that this fraction equals 5/9
  sorry  -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_fraction_used_l433_43349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_boy_one_girl_l433_43386

/-- The probability of having at least one boy and one girl in a family of four children,
    given that the probability of having a boy or a girl is equally likely. -/
theorem prob_at_least_one_boy_one_girl : 
  let n : ℕ := 4  -- number of children
  let p : ℚ := 1/2  -- probability of having a boy (or girl)
  let Prob : ℚ := 1 - 2 * p^n  -- probability of at least one boy and one girl
  Prob = 7/8 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_boy_one_girl_l433_43386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr1_equals_4_plus_sqrt6_expr2_equals_sqrt6_over_2_l433_43365

-- Define the expressions as noncomputable
noncomputable def expr1 : ℝ := Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24
noncomputable def expr2 : ℝ := Real.sqrt 3 * Real.sqrt 2 - Real.sqrt 12 / Real.sqrt 8

-- State the theorems
theorem expr1_equals_4_plus_sqrt6 : expr1 = 4 + Real.sqrt 6 := by sorry

theorem expr2_equals_sqrt6_over_2 : expr2 = (Real.sqrt 6) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr1_equals_4_plus_sqrt6_expr2_equals_sqrt6_over_2_l433_43365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l433_43390

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (m : ℝ), m = Real.tan (π / 6) ∧ m = b / a) →
  Real.sqrt (1 + b^2 / a^2) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l433_43390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_value_max_value_expression_l433_43397

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2)

theorem triangle_angle_B_value (t : Triangle) 
  (h : area t = (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2)) : 
  t.B = π/3 := by sorry

theorem max_value_expression (t : Triangle) 
  (h1 : area t = (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2))
  (h2 : t.b = Real.sqrt 3) : 
  (∀ x : ℝ, (Real.sqrt 3 - 1) * t.a + 2 * t.c ≤ 2 * Real.sqrt 6) ∧ 
  (∃ x : ℝ, (Real.sqrt 3 - 1) * t.a + 2 * t.c = 2 * Real.sqrt 6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_value_max_value_expression_l433_43397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_with_divisibility_l433_43389

/-- 
Theorem: The smallest positive integer M such that among M, M+1, M+2, and M+3, 
one is divisible by 2^3, one by 3^2, one by 5^2, and one by 11^2, is 484.
-/
theorem smallest_m_with_divisibility : ∃ (M : ℕ),
  (M = 484) ∧
  (∀ (k : ℕ), k < M →
    ¬(∃ (a b c d : ℕ), ({a, b, c, d} : Finset ℕ) = {k, k+1, k+2, k+3} ∧
      8 ∣ a ∧ 9 ∣ b ∧ 25 ∣ c ∧ 121 ∣ d)) ∧
  (∃ (a b c d : ℕ), ({a, b, c, d} : Finset ℕ) = {M, M+1, M+2, M+3} ∧
    8 ∣ a ∧ 9 ∣ b ∧ 25 ∣ c ∧ 121 ∣ d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_with_divisibility_l433_43389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l433_43376

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem f_properties :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f (-x) = -f x) ∧  -- f is odd on (-1, 1)
  (f (1/2) = 2/5) ∧  -- f(1/2) = 2/5
  (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f x < f y) ∧  -- f is monotonically increasing on (-1, 1)
  (∀ t, t ∈ Set.Ioo (-1/2 : ℝ) (-1/3) → f (t+1) + f (2*t) < 0) ∧  -- Solution to the inequality
  (∀ t, f (t+1) + f (2*t) < 0 → t ∈ Set.Ioo (-1/2 : ℝ) (-1/3))  -- Completeness of the solution
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l433_43376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_planes_from_three_lines_l433_43319

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line in 3D space
  mk :: -- Constructor

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane in 3D space
  mk :: -- Constructor

/-- Checks if two lines intersect -/
def intersects (l1 l2 : Line3D) : Prop :=
  sorry

/-- Determines the planes formed by three lines -/
def planes_from_lines (l1 l2 l3 : Line3D) : Finset Plane3D :=
  sorry

/-- Main theorem -/
theorem number_of_planes_from_three_lines
  (l1 l2 l3 : Line3D)
  (h : (intersects l1 l2 ∧ intersects l1 l3) ∨ (intersects l2 l1 ∧ intersects l2 l3) ∨ (intersects l3 l1 ∧ intersects l3 l2)) :
  let planes := planes_from_lines l1 l2 l3
  planes.card = 1 ∨ planes.card = 2 ∨ planes.card = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_planes_from_three_lines_l433_43319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_savings_amount_l433_43368

-- Define the earnings of Kimmie
noncomputable def kimmie_earnings : ℚ := 450

-- Define Zahra's earnings as a function of Kimmie's
noncomputable def zahra_earnings (k : ℚ) : ℚ := k - (k / 3)

-- Define the savings rate
noncomputable def savings_rate : ℚ := 1 / 2

-- Define the total savings in the joint account
noncomputable def total_savings (k : ℚ) : ℚ := 
  savings_rate * k + savings_rate * (zahra_earnings k)

-- Theorem statement
theorem joint_savings_amount :
  total_savings kimmie_earnings = 375 := by
  -- Expand the definition of total_savings
  unfold total_savings
  -- Expand the definition of zahra_earnings
  unfold zahra_earnings
  -- Expand the definition of kimmie_earnings
  unfold kimmie_earnings
  -- Expand the definition of savings_rate
  unfold savings_rate
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_savings_amount_l433_43368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_range_l433_43399

-- Define a cube with edge length 1
def Cube : Type := Fin 8 → ℝ × ℝ × ℝ

-- Define a plane passing through a face diagonal
def PlaneThroughFaceDiagonal (c : Cube) : Type := ℝ → ℝ × ℝ × ℝ

-- Define the cross-section area function
noncomputable def CrossSectionArea (c : Cube) (p : PlaneThroughFaceDiagonal c) : ℝ :=
  sorry -- Actual calculation of area would go here

-- Theorem statement
theorem cross_section_area_range (c : Cube) :
  ∀ p : PlaneThroughFaceDiagonal c,
    Real.sqrt 6 / 2 ≤ CrossSectionArea c p ∧ CrossSectionArea c p ≤ Real.sqrt 2 :=
by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_range_l433_43399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_percentage_is_approx_8_03_l433_43375

/-- Represents the shopkeeper's inventory and pricing information -/
structure ShopInventory where
  oranges : Nat
  bananas : Nat
  apples : Nat
  orange_rotten_percent : Float
  banana_rotten_percent : Float
  apple_rotten_percent : Float
  orange_profit_percent : Float
  banana_profit_percent : Float
  apple_profit_percent : Float
  orange_cost : Float
  banana_cost : Float
  apple_cost : Float

/-- Calculates the total profit percentage for the shopkeeper -/
noncomputable def calculateProfitPercentage (inventory : ShopInventory) : Float :=
  let good_oranges := inventory.oranges.toFloat * (1 - inventory.orange_rotten_percent)
  let good_bananas := inventory.bananas.toFloat * (1 - inventory.banana_rotten_percent)
  let good_apples := inventory.apples.toFloat * (1 - inventory.apple_rotten_percent)
  
  let total_cost := inventory.orange_cost * inventory.oranges.toFloat +
                    inventory.banana_cost * inventory.bananas.toFloat +
                    inventory.apple_cost * inventory.apples.toFloat
  
  let total_revenue := inventory.orange_cost * (1 + inventory.orange_profit_percent) * good_oranges +
                       inventory.banana_cost * (1 + inventory.banana_profit_percent) * good_bananas +
                       inventory.apple_cost * (1 + inventory.apple_profit_percent) * good_apples
  
  let profit := total_revenue - total_cost
  (profit / total_cost) * 100

/-- Theorem stating that the shopkeeper's profit percentage is approximately 8.03% -/
theorem shopkeeper_profit_percentage_is_approx_8_03 (inventory : ShopInventory) 
  (h1 : inventory.oranges = 1000)
  (h2 : inventory.bananas = 800)
  (h3 : inventory.apples = 750)
  (h4 : inventory.orange_rotten_percent = 0.12)
  (h5 : inventory.banana_rotten_percent = 0.05)
  (h6 : inventory.apple_rotten_percent = 0.10)
  (h7 : inventory.orange_profit_percent = 0.20)
  (h8 : inventory.banana_profit_percent = 0.25)
  (h9 : inventory.apple_profit_percent = 0.15)
  (h10 : inventory.orange_cost = 2.5)
  (h11 : inventory.banana_cost = 1.5)
  (h12 : inventory.apple_cost = 2.0) :
  Float.abs (calculateProfitPercentage inventory - 8.03) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_percentage_is_approx_8_03_l433_43375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brand_preference_analysis_l433_43329

structure BrandSurvey where
  total_users : Nat
  male_like : Nat
  male_dislike : Nat
  female_like : Nat
  female_dislike : Nat

noncomputable def chi_square (survey : BrandSurvey) : ℝ :=
  let n := survey.total_users
  let a := survey.male_like
  let b := survey.male_dislike
  let c := survey.female_like
  let d := survey.female_dislike
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem brand_preference_analysis (survey : BrandSurvey)
  (h1 : survey.total_users = 100)
  (h2 : survey.male_like = 13)
  (h3 : survey.male_dislike = 27)
  (h4 : survey.female_like = 42)
  (h5 : survey.female_dislike = 18) :
  (survey.female_dislike / (survey.male_dislike + survey.female_dislike) * 5 : ℝ) = 2 ∧
  (6 : ℝ) / 10 = 3 / 5 ∧
  chi_square survey > 6.635 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brand_preference_analysis_l433_43329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_four_theta_l433_43356

theorem cos_four_theta (θ : Real) (h : Real.cos θ = 1/4) : Real.cos (4*θ) = 17/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_four_theta_l433_43356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_F_selects_l433_43370

/-- Two sequences are completely different if they differ at every position -/
def CompletelyDifferent (x y : ℕ → ℕ) : Prop :=
  ∀ n, x n ≠ y n

/-- F is a function from sequences of natural numbers to natural numbers -/
noncomputable def F : (ℕ → ℕ) → ℕ := sorry

/-- F maps completely different sequences to different values -/
axiom F_inj : ∀ x y, CompletelyDifferent x y → F x ≠ F y

/-- F maps constant sequences to their constant value -/
axiom F_const : ∀ k, F (λ _ => k) = k

/-- There exists a position n such that F always selects the n-th element of any sequence -/
theorem exists_n_F_selects : ∃ n, ∀ x : ℕ → ℕ, F x = x n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_F_selects_l433_43370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l433_43371

/-- The radius of the larger circle D -/
noncomputable def R : ℝ := 40

/-- The number of smaller circles -/
def n : ℕ := 8

/-- The radius of each smaller circle -/
noncomputable def r : ℝ := R / (1 + Real.sqrt 2)

/-- The area between the larger circle and the n smaller circles -/
noncomputable def M : ℝ := Real.pi * R^2 - n * Real.pi * r^2

/-- Theorem stating that the floor of M is 1874 -/
theorem area_between_circles : ⌊M⌋ = 1874 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l433_43371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l433_43364

-- Define IsTriangle predicate
def IsTriangle (A B C : Real) : Prop := 
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi

theorem triangle_properties (A B C a b c : Real) : 
  -- Triangle ABC exists with sides a, b, c opposite to angles A, B, C
  IsTriangle A B C → 
  -- Condition 1: sin C * sin(A - B) = sin B * sin(C - A)
  Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A) → 
  -- Condition 2: A = 2B
  A = 2 * B → 
  -- Conclusion 1: C = 5π/8
  C = 5 * Real.pi / 8 ∧ 
  -- Conclusion 2: 2a² = b² + c²
  2 * a^2 = b^2 + c^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l433_43364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_cos_value_triangle_condition_implies_f_range_l433_43346

-- Define vectors m and n
noncomputable def m (x : ℝ) : Fin 2 → ℝ := ![Real.sqrt 3 * Real.sin (x / 4), 1]
noncomputable def n (x : ℝ) : Fin 2 → ℝ := ![Real.cos (x / 4), (Real.cos (x / 4))^2]

-- Define dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define perpendicularity
def perpendicular (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

-- Define function f
noncomputable def f (x : ℝ) : ℝ := dot_product (m x) (n x)

theorem perpendicular_implies_cos_value (x : ℝ) :
  perpendicular (m x) (n x) → Real.cos (2 * Real.pi / 3 - x) = -1/2 := by sorry

theorem triangle_condition_implies_f_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < 2 * Real.pi / 3 →
  (2 * a - c) * Real.cos B = b * Real.cos C →
  1 < f A ∧ f A < 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_cos_value_triangle_condition_implies_f_range_l433_43346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_shoes_cost_proof_l433_43320

/-- The cost of soccer shoes given the cost of socks and Jack's financial situation -/
def soccer_shoes_cost (socks_cost total_has total_needs : ℚ) : ℚ :=
  let total_cost := total_has + total_needs
  total_cost - socks_cost

/-- Proof of the soccer shoes cost -/
theorem soccer_shoes_cost_proof 
  (socks_cost : ℚ) 
  (total_has : ℚ) 
  (total_needs : ℚ) 
  (h1 : socks_cost = 19)
  (h2 : total_has = 40)
  (h3 : total_needs = 71) :
  soccer_shoes_cost socks_cost total_has total_needs = 92 := by
  unfold soccer_shoes_cost
  rw [h1, h2, h3]
  norm_num

#eval soccer_shoes_cost 19 40 71

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_shoes_cost_proof_l433_43320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l433_43312

theorem triangle_angle_inequalities (A B : Real) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) :
  (A > B ↔ Real.cos A < Real.cos B) ∧
  (A > B ↔ Real.sin A > Real.sin B) ∧
  (A > B ↔ Real.cos (2*A) < Real.cos (2*B)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l433_43312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_on_line_l433_43381

noncomputable def proj (a : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dotProduct := a.1 * u.1 + a.2 * u.2
  let normSquared := a.1 * a.1 + a.2 * a.2
  (dotProduct / normSquared * a.1, dotProduct / normSquared * a.2)

theorem vector_on_line (u : ℝ × ℝ) :
  proj (3, 1) u = (3/5, 1/5) → u.2 = -3 * u.1 + 2 := by
  sorry

#check vector_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_on_line_l433_43381
