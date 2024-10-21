import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AD_is_4_l322_32299

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define vectors m and n
variable (m n : V)

-- Define the angle between m and n
noncomputable def angle_m_n : ℝ := Real.pi / 6

-- Define the magnitudes of m and n
noncomputable def mag_m : ℝ := Real.sqrt 3
def mag_n : ℝ := 2

-- Define vectors AB and AC
def AB (m n : V) : V := 2 • m + 2 • n
def AC (m n : V) : V := 2 • m - 6 • n

-- Define D as the midpoint of BC
def D (m n : V) : V := AB m n + (1/2) • (AC m n - AB m n)

-- Theorem to prove
theorem length_AD_is_4 (m n : V) 
  (h1 : ‖m‖ = Real.sqrt 3) 
  (h2 : ‖n‖ = 2) 
  (h3 : inner m n = ‖m‖ * ‖n‖ * Real.cos (Real.pi / 6)) : 
  ‖D m n‖ = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AD_is_4_l322_32299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janos_walked_distance_l322_32224

/-- Represents the scenario of János and his wife's travel --/
structure TravelScenario where
  wife_speed : ℝ
  normal_trip_duration : ℝ
  early_arrival_time : ℝ

/-- Calculates the distance János walked based on the given travel scenario --/
noncomputable def distance_walked (scenario : TravelScenario) : ℝ :=
  scenario.wife_speed * (scenario.normal_trip_duration - scenario.early_arrival_time) / 2

/-- Theorem stating that János walked 3.5 km --/
theorem janos_walked_distance (scenario : TravelScenario) 
  (h1 : scenario.wife_speed = 42)
  (h2 : scenario.normal_trip_duration = 1)
  (h3 : scenario.early_arrival_time = 1/6) :
  distance_walked scenario = 3.5 := by
  sorry

#check distance_walked { wife_speed := 42, normal_trip_duration := 1, early_arrival_time := 1/6 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janos_walked_distance_l322_32224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l322_32230

/-- Represents a plant type -/
inductive Plant
| Basil
| Aloe
| Cactus

/-- Represents a lamp color -/
inductive LampColor
| White
| Red

/-- Represents the collection of plants -/
def plants : Multiset Plant :=
  { Plant.Basil, Plant.Basil, Plant.Aloe, Plant.Cactus }

/-- Represents the collection of lamps -/
def lamps : Multiset LampColor :=
  { LampColor.White, LampColor.White, LampColor.Red, LampColor.Red }

/-- A function that counts the number of valid arrangements -/
noncomputable def countArrangements : ℕ :=
  sorry

/-- Theorem stating that the number of valid arrangements is 22 -/
theorem arrangement_count : countArrangements = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l322_32230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_60_digits_eq_15_l322_32281

/-- The decimal representation of 1/9999 -/
def decimal_rep : ℚ := 1 / 9999

/-- The sequence of digits in the decimal representation of 1/9999 -/
def digit_sequence (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => 1
  | _ => 0  -- This case is technically unreachable, but Lean requires it for completeness

/-- The sum of the first 60 digits after the decimal point in the decimal representation of 1/9999 -/
def sum_60_digits : ℕ := (List.range 60).map digit_sequence |>.sum

theorem sum_60_digits_eq_15 : sum_60_digits = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_60_digits_eq_15_l322_32281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iterated_function_fixed_point_l322_32249

-- Define the function f
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

-- Define the n-th iteration of f
noncomputable def F (a b c d : ℝ) : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => f a b c d (F a b c d n x)

-- Theorem statement
theorem iterated_function_fixed_point 
  (a b c d : ℝ) (n : ℕ) 
  (h1 : f a b c d 0 ≠ 0) 
  (h2 : f a b c d (f a b c d 0) ≠ 0) 
  (h3 : F a b c d n 0 = 0) : 
  ∀ x, F a b c d n x = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_iterated_function_fixed_point_l322_32249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_of_digits_l322_32201

/-- Function to calculate the number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ := 
  if n = 0 then 1 else (Nat.log n 10).succ

/-- Function to calculate the sum of digits for all numbers from 1 to n -/
def sumDigits (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => numDigits (i + 1))

/-- Theorem stating that 295 is the unique natural number n for which 
    the sum of digits of all numbers from 1 to n equals 777 -/
theorem unique_sum_of_digits : ∃! n : ℕ, sumDigits n = 777 ∧ n = 295 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_of_digits_l322_32201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l322_32298

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt 3 / 2) * Real.sin (2 * x) - (1 / 2) * (Real.cos x ^ 2 - Real.sin x ^ 2) - 1

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6)

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  c = Real.sqrt 7 →
  f C = 0 →
  Real.sin B = 3 * Real.sin A →
  g B = 0 →
  (a = 1 ∧ b = 3) ∧
  0 < Real.cos A * 1 + Real.cos B * (Real.sin A - Real.cos A * Real.tan B) ∧
  Real.cos A * 1 + Real.cos B * (Real.sin A - Real.cos A * Real.tan B) ≤ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l322_32298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_symmetry_center_l322_32213

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.cos (2 * x)

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi := by
  sorry

-- Theorem for the symmetry center
theorem symmetry_center :
  ∀ (x : ℝ), f (Real.pi / 4 + x) = -f (Real.pi / 4 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_symmetry_center_l322_32213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l322_32262

-- Define the function f(x) = √(x-1)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1)

-- State the theorem about the domain of f
theorem domain_of_f : Set.Ici 1 = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l322_32262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_sine_intersections_l322_32276

-- Define the arc
noncomputable def Arc (center : ℝ × ℝ) (angle : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ angle ∧
    p.1 = center.1 + (p.2 - center.2) * Real.tan θ ∧
    (p.1 - center.1)^2 + (p.2 - center.2)^2 = ((p.2 - center.2) / Real.cos θ)^2 }

-- Define the sine curve
def SineCurve : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.2 = Real.sin p.1 }

-- Theorem statement
theorem arc_sine_intersections :
  ∀ h : ℝ, ∃ S : Set ℕ, S = {0} ∪ {n : ℕ | n > 0} ∧
  ∀ n ∈ S, ∃ r : ℝ, (Arc (0, h) (30 * π / 180) ∩ SineCurve).ncard = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_sine_intersections_l322_32276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_pie_distribution_l322_32203

theorem cookie_pie_distribution :
  ∀ (num_pies : ℕ) (slices_per_pie : ℕ) (num_classmates : ℕ) (slices_left : ℕ),
    num_pies = 3 →
    slices_per_pie = 10 →
    num_classmates = 24 →
    slices_left = 4 →
    let total_people := num_classmates + 2
    let total_slices := num_pies * slices_per_pie
    let slices_eaten := total_slices - slices_left
    slices_eaten = total_people →
    slices_eaten / total_people = 1 :=
by
  intros num_pies slices_per_pie num_classmates slices_left
  intros h_pies h_slices h_classmates h_left
  intros total_people total_slices slices_eaten
  intro h_eaten
  sorry

#check cookie_pie_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_pie_distribution_l322_32203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_right_l322_32280

/-- The horizontal shift of sin(2x - π/3) compared to sin(2x) -/
noncomputable def horizontal_shift : ℝ := Real.pi / 6

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

/-- The transformed function -/
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem sin_shift_right (x : ℝ) : 
  g x = f (x - horizontal_shift) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_right_l322_32280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_integer_set_l322_32240

theorem exists_special_integer_set :
  ∃ (S : Finset ℤ), 
    Finset.card S = 25 ∧ 
    (∀ x y : ℤ, x ∈ S → y ∈ S → x ≠ y → x ≠ y) ∧
    (Finset.sum S id = 0) ∧
    (∀ T : Finset ℤ, T ⊆ S → Finset.card T = 24 → |Finset.sum T id| > 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_integer_set_l322_32240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_distance_l322_32228

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Curve in 2D space -/
structure Curve where
  equation : ℝ → ℝ → Prop

noncomputable def P : Point := ⟨0, Real.sqrt 3⟩

def C : Curve := ⟨fun x y ↦ x^2 / 5 + y^2 / 15 = 1⟩

def l : Line := sorry

def A : Point := sorry
def B : Point := sorry

theorem intersection_sum_distance (t₁ t₂ : ℝ) (h₁ : t₁ + t₂ = 8) (h₂ : 2 * t₁ * t₂ = 8) :
  let d₁ := Real.sqrt ((A.x - P.x)^2 + (A.y - P.y)^2)
  let d₂ := Real.sqrt ((B.x - P.x)^2 + (B.y - P.y)^2)
  d₁ + d₂ = 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_distance_l322_32228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_by_resistance_l322_32220

/-- The work done by resistance for an object moving according to x = 4t^2 -/
theorem work_done_by_resistance 
  (x : ℝ → ℝ) -- Position function
  (F : ℝ → ℝ) -- Resistance force function
  (h1 : ∀ t, x t = 4 * t^2) -- Position equation
  (h2 : ∃ k, ∀ v, F v = k * v) -- Resistance proportional to velocity
  (h3 : F 10 = 2) -- Given condition for resistance at v = 10 m/s
  : ∫ x in (0 : ℝ)..(2 : ℝ), -F (Real.sqrt (x/4)) = -16/15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_by_resistance_l322_32220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l322_32263

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ :=
  Real.sqrt ((r1 * Real.cos θ1 - r2 * Real.cos θ2)^2 + (r1 * Real.sin θ1 - r2 * Real.sin θ2)^2)

/-- Theorem: The distance between points A(2, π/3) and B(2, 2π/3) in polar coordinates is 2 -/
theorem distance_between_specific_points :
  polar_distance 2 (Real.pi / 3) 2 (2 * Real.pi / 3) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l322_32263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_count_special_permutations_correct_l322_32215

/-- 
Given a natural number n, count_special_permutations n returns the number of permutations 
of {1, 2, ..., n} where each element is either greater than all preceding elements 
or less than all preceding elements.
-/
def count_special_permutations (n : ℕ) : ℕ := 2^(n-1)

/-- 
Theorem stating that count_special_permutations correctly counts the number of special permutations
for any natural number n.
-/
theorem count_special_permutations_correct (n : ℕ) : 
  count_special_permutations n = (Finset.univ.filter (λ p : Equiv.Perm (Fin n) => 
    ∀ i : Fin n, i.val > 0 → 
      (∀ j : Fin n, j.val < i.val → p.toFun j < p.toFun i) ∨ 
      (∀ j : Fin n, j.val < i.val → p.toFun j > p.toFun i)
  )).card := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_count_special_permutations_correct_l322_32215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puck_leave_disk_time_l322_32245

/-- The time for a puck to leave a rotating disk -/
noncomputable def time_to_leave_disk (R : ℝ) (n : ℝ) : ℝ :=
  Real.sqrt 15 / (2 * Real.pi * n)

/-- Theorem: The time for a puck to leave a rotating disk of radius R,
    given an initial distance of R/4 from the center and a rotation frequency of n,
    is equal to sqrt(15) / (2π * n) -/
theorem puck_leave_disk_time (R : ℝ) (n : ℝ) (h₁ : R > 0) (h₂ : n > 0) :
  let initial_distance := R / 4
  let disk_radius := R
  let rotation_frequency := n
  time_to_leave_disk R n = Real.sqrt 15 / (2 * Real.pi * n) := by
  sorry

#check puck_leave_disk_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puck_leave_disk_time_l322_32245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l322_32295

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  B = π / 3 →
  Real.cos A = 4 / 5 →
  b = Real.sqrt 3 →
  Real.sin C = (3 + 4 * Real.sqrt 3) / 10 ∧
  (1 / 2) * a * b * Real.sin C = (36 + 9 * Real.sqrt 3) / 50 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l322_32295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pens_l322_32243

/-- The cost of one pen in rubles -/
def pen_cost : ℕ := sorry

/-- The number of pens Masha bought -/
def masha_pens : ℕ := sorry

/-- The number of pens Olya bought -/
def olya_pens : ℕ := sorry

/-- Axiom: The cost of one pen is greater than 10 -/
axiom pen_cost_gt_10 : pen_cost > 10

/-- Axiom: Masha's total spending is 357 rubles -/
axiom masha_total : masha_pens * pen_cost = 357

/-- Axiom: Olya's total spending is 441 rubles -/
axiom olya_total : olya_pens * pen_cost = 441

/-- Theorem: The total number of pens bought by Masha and Olya is 38 -/
theorem total_pens : masha_pens + olya_pens = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pens_l322_32243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_same_color_l322_32251

/-- Represents a backgammon checker with two sides -/
inductive Checker
| Black
| White

/-- Represents a move in the game -/
structure Move where
  start : Nat
  endPos : Nat

/-- Represents the game state -/
structure GameState where
  checkers : List Checker
  moves : List Move

/-- Defines a valid initial arrangement of checkers -/
def valid_initial_arrangement (checkers : List Checker) : Prop :=
  checkers.length = 2012 ∧
  ∀ i, i < 2011 → checkers[i]? ≠ checkers[i+1]?

/-- Defines a valid move -/
def valid_move (move : Move) (state : GameState) : Prop :=
  move.start < move.endPos ∧ move.endPos < state.checkers.length

/-- Defines the state after a move -/
def apply_move (move : Move) (state : GameState) : GameState :=
  sorry

/-- Checks if all checkers are the same color -/
def all_same_color (state : GameState) : Prop :=
  ∀ i j, i < state.checkers.length → j < state.checkers.length → 
    state.checkers[i]? = state.checkers[j]?

/-- The main theorem to prove -/
theorem min_moves_to_same_color 
  (initial_state : GameState) 
  (h_valid : valid_initial_arrangement initial_state.checkers) :
  ∃ (final_state : GameState),
    final_state.moves.length = 1006 ∧
    all_same_color final_state ∧
    (∀ (state : GameState),
      state.moves.length < 1006 →
      ¬all_same_color state) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_same_color_l322_32251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_buses_2010_year_electric_buses_exceed_third_l322_32284

/-- Represents the number of fuel-powered buses in 2003 -/
def initial_fuel_buses : ℕ := 10000

/-- Represents the number of electric buses introduced in 2004 -/
def initial_electric_buses : ℕ := 128

/-- Represents the annual growth rate of electric buses -/
def growth_rate : ℚ := 3/2

/-- Calculates the number of electric buses introduced in a given year -/
def electric_buses_introduced (year : ℕ) : ℕ :=
  ⌊(initial_electric_buses : ℚ) * growth_rate ^ (year - 1)⌋.toNat

/-- Calculates the total number of electric buses up to a given year -/
def total_electric_buses (year : ℕ) : ℕ :=
  (Finset.range year).sum (λ i => electric_buses_introduced (i + 1))

/-- Theorem for the number of electric buses introduced in 2010 -/
theorem electric_buses_2010 : electric_buses_introduced 7 = 1458 := by sorry

/-- Theorem for the year when electric buses exceed one-third of total buses -/
theorem year_electric_buses_exceed_third :
  (∀ y < 8, 3 * total_electric_buses y ≤ initial_fuel_buses + total_electric_buses y) ∧
  (3 * total_electric_buses 8 > initial_fuel_buses + total_electric_buses 8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_buses_2010_year_electric_buses_exceed_third_l322_32284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_agreement_exists_l322_32236

/-- A type representing an infinite decimal fraction -/
def InfiniteDecimal := ℕ → Fin 10

/-- A function that checks if two infinite decimals agree at a given position -/
def agree (d1 d2 : InfiniteDecimal) (pos : ℕ) : Prop := d1 pos = d2 pos

/-- The theorem statement -/
theorem infinite_agreement_exists (decimals : Fin 11 → InfiniteDecimal) 
  (h : ∀ i j : Fin 11, i ≠ j → decimals i ≠ decimals j) :
  ∃ i j : Fin 11, i ≠ j ∧ Set.Infinite {pos : ℕ | agree (decimals i) (decimals j) pos} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_agreement_exists_l322_32236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_for_beta_f_value_for_alpha_l322_32269

-- Define the function f as noncomputable due to dependency on Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.cos x) ^ 2

-- Statement for the range of f(β)
theorem f_range_for_beta :
  ∀ β : ℝ, β ∈ Set.Icc 0 (Real.pi / 2) →
  ∃ y : ℝ, y ∈ Set.Icc (-2) 1 ∧ f β = y :=
by
  sorry

-- Statement for the value of f(α)
theorem f_value_for_alpha :
  ∀ α : ℝ, Real.tan α = 2 * Real.sqrt 3 →
  f α = 10 / 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_for_beta_f_value_for_alpha_l322_32269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turban_price_l322_32296

def yearly_salary : ℝ := 90
def months_worked : ℝ := 9
def cash_received : ℝ := 65

theorem turban_price : (months_worked / 12) * yearly_salary - cash_received = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turban_price_l322_32296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l322_32231

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f x = f (x + 4)
axiom f_exp : ∀ x : ℝ, x > -2 ∧ x < 0 → f x = 2^x

-- State the theorem
theorem f_difference : f 2012 - f 2011 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l322_32231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l322_32278

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Define the minimum value function g(a)
noncomputable def g (a : ℝ) : ℝ :=
  if a < -1 then 2*a + 3
  else if a ≤ 1 then 2 - a^2
  else 3 - 2*a

-- Theorem statement
theorem min_value_f (a : ℝ) :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, g a ≤ f a x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l322_32278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_percentage_theorem_l322_32242

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a rectangle given two opposite corners -/
noncomputable def rectangleArea (p1 p2 : Point) : ℝ :=
  |p2.x - p1.x| * |p2.y - p1.y|

/-- Calculates the area of a right triangle given three vertices -/
noncomputable def rightTriangleArea (p1 p2 p3 : Point) : ℝ :=
  rectangleArea p1 p3 / 2

/-- The side length of the square PQRS -/
def squareSideLength : ℝ := 6

/-- The total area of square PQRS -/
def totalArea : ℝ := squareSideLength ^ 2

/-- The shaded regions in the square -/
def shadedRegions : List (Point × Point) := [
  ({x := 0, y := 0}, {x := 2, y := 2}),
  ({x := 3, y := 0}, {x := 6, y := 3}),
  ({x := 4, y := 4}, {x := 6, y := 6})
]

/-- The unshaded triangle vertices -/
def unshadedTriangle : (Point × Point × Point) :=
  ({x := 6, y := 0}, {x := 6, y := 3}, {x := 3, y := 3})

/-- Calculates the total shaded area -/
noncomputable def shadedArea : ℝ :=
  (shadedRegions.map (fun (p1, p2) => rectangleArea p1 p2)).sum -
  rightTriangleArea unshadedTriangle.1 unshadedTriangle.2.1 unshadedTriangle.2.2

/-- Theorem: The percentage of the shaded area in square PQRS is equal to (1250/36)% -/
theorem shaded_percentage_theorem :
  (shadedArea / totalArea) * 100 = 1250 / 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_percentage_theorem_l322_32242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l322_32297

theorem triangle_proof (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C →
  -- A, B, C are angles of a triangle
  A + B + C = π →
  -- Area of triangle is 3√3
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 →
  -- b/c = 3√3/4
  b / c = 3 * Real.sqrt 3 / 4 →
  -- Vectors are parallel
  b * Real.sin (2*A) = Real.sqrt 3 * a * Real.sin B →
  -- Prove A = π/6 and a = √7
  A = π/6 ∧ a = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l322_32297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcake_preparation_time_l322_32253

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hMinutes : minutes < 60

/-- Converts Time to hours as a real number -/
noncomputable def timeToHours (t : Time) : ℝ :=
  t.hours + t.minutes / 60

/-- Adds hours to a Time -/
def addHours (t : Time) (h : ℕ) : Time :=
  { hours := (t.hours + h) % 24,
    minutes := t.minutes,
    hMinutes := t.hMinutes }

def startTime : Time := ⟨9, 0, by norm_num⟩
def halfwayTime : Time := ⟨12, 30, by norm_num⟩
def finishTime : Time := ⟨16, 0, by norm_num⟩

theorem cupcake_preparation_time : 
  timeToHours halfwayTime - timeToHours startTime = 
    (timeToHours finishTime - timeToHours startTime) / 2 ∧
  addHours startTime 7 = finishTime :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcake_preparation_time_l322_32253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_draw_probability_second_draw_probability_l322_32252

/-- Represents the color of a ball in the lottery system -/
inductive BallColor
| Black
| Red
| White

/-- Represents the lottery system with its rules and probabilities -/
structure LotterySystem where
  /-- Total number of balls in the system -/
  total_balls : Nat
  /-- Number of black balls -/
  black_balls : Nat
  /-- Number of red balls -/
  red_balls : Nat
  /-- Number of white balls -/
  white_balls : Nat
  /-- Reward for a black ball in the first draw scenario -/
  black_reward_1 : Nat
  /-- Reward for a red ball in the first draw scenario -/
  red_reward_1 : Nat
  /-- Reward for a white ball in the first draw scenario -/
  white_reward_1 : Nat
  /-- Reward for a black ball in the second draw scenario -/
  black_reward_2 : Nat
  /-- Reward for a red ball in the second draw scenario -/
  red_reward_2 : Nat
  /-- Reward for a white ball in the second draw scenario -/
  white_reward_2 : Nat
  /-- Constraint: Total balls is sum of all colored balls -/
  total_constraint : total_balls = black_balls + red_balls + white_balls

/-- Theorem for the first draw scenario -/
theorem first_draw_probability (ls : LotterySystem) 
  (h1 : ls.total_balls = 6)
  (h2 : ls.black_balls = 3)
  (h3 : ls.red_balls = 2)
  (h4 : ls.white_balls = 1)
  (h5 : ls.black_reward_1 = 1)
  (h6 : ls.red_reward_1 = 2)
  (h7 : ls.white_reward_1 = 3) :
  (2 : ℚ) / 5 = 2/5 := by sorry

/-- Theorem for the second draw scenario -/
theorem second_draw_probability (ls : LotterySystem)
  (h1 : ls.total_balls = 6)
  (h2 : ls.black_balls = 3)
  (h3 : ls.red_balls = 2)
  (h4 : ls.white_balls = 1)
  (h5 : ls.black_reward_2 = 5)
  (h6 : ls.red_reward_2 = 10)
  (h7 : ls.white_reward_2 = 5) :
  (8 : ℚ) / 9 = 8/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_draw_probability_second_draw_probability_l322_32252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nine_factorial_over_126_l322_32250

theorem sqrt_nine_factorial_over_126 :
  let nine_factorial : ℕ := 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  let one_twenty_six : ℕ := 2 * 3^2 * 7
  (Real.sqrt ((nine_factorial : ℝ) / one_twenty_six)) = 24 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nine_factorial_over_126_l322_32250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_given_polynomial_l322_32223

/-- The degree of a polynomial is the highest sum of exponents in any term. -/
noncomputable def polynomial_degree (p : MvPolynomial (Fin 2) ℚ) : ℕ := sorry

/-- The given polynomial 7a+2a³b-a²b-5a³ -/
noncomputable def given_polynomial : MvPolynomial (Fin 2) ℚ :=
  let a := MvPolynomial.X 0
  let b := MvPolynomial.X 1
  7*a + 2*a^3*b - a^2*b - 5*a^3

theorem degree_of_given_polynomial :
  polynomial_degree given_polynomial = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_given_polynomial_l322_32223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gwen_average_speed_l322_32289

/-- Represents a segment of Gwen's trip -/
structure TripSegment where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a trip segment -/
noncomputable def time_for_segment (segment : TripSegment) : ℝ :=
  segment.distance / segment.speed

/-- Gwen's trip segments -/
def trip_segments : List TripSegment := [
  ⟨40, 15⟩,
  ⟨60, 30⟩,
  ⟨100, 45⟩,
  ⟨50, 60⟩
]

/-- Theorem stating Gwen's average speed for the entire trip -/
theorem gwen_average_speed :
  let total_distance := (trip_segments.map (λ s => s.distance)).sum
  let total_time := (trip_segments.map time_for_segment).sum
  total_distance / total_time = 4500 / 139 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gwen_average_speed_l322_32289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_seven_last_is_one_fourth_compute_result_l322_32207

/-- Represents an 8-sided die with dynamic relabeling -/
structure DynamicDie :=
  (fixed_sides : Finset Nat)
  (dynamic_side : Nat)

/-- The process of rolling the die until all numbers are rolled -/
def roll_sequence (d : DynamicDie) : List Nat :=
  sorry

/-- The probability of rolling 7 as the last number -/
noncomputable def prob_seven_last (d : DynamicDie) : ℝ :=
  sorry

/-- Initial state of the die -/
def initial_die : DynamicDie :=
  { fixed_sides := {1, 2, 3, 4, 5, 6, 7}, dynamic_side := 1 }

/-- Main theorem: The probability of rolling 7 last is 1/4 -/
theorem prob_seven_last_is_one_fourth :
  prob_seven_last initial_die = 1/4 := by
  sorry

/-- Compute 100a + b where probability is a/b -/
theorem compute_result :
  (100 : ℕ) * (1 : ℕ) + (4 : ℕ) = 104 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_seven_last_is_one_fourth_compute_result_l322_32207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_result_sin_cos_equation_result_l322_32227

-- Part 1
theorem trig_equation_result (α : ℝ) (h : Real.sin (α - 3 * Real.pi) = 2 * Real.cos (α - 4 * Real.pi)) :
  (Real.sin (Real.pi - α) + 5 * Real.cos (2 * Real.pi - α)) / 
  (2 * Real.sin (3 * Real.pi / 2 - α) - Real.sin (-α)) = -3 / 4 := by
  sorry

-- Part 2
theorem sin_cos_equation_result (x : ℝ) (h1 : -Real.pi / 2 < x) (h2 : x < 0) 
    (h3 : Real.sin x + Real.cos x = 1 / 5) :
  Real.sin x - Real.cos x = -7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_result_sin_cos_equation_result_l322_32227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_selection_for_difference_two_l322_32247

theorem minimum_selection_for_difference_two (n : ℕ) (h : n = 20) :
  ∃ k : ℕ, k = 11 ∧
  (∀ S : Finset ℕ, S ⊆ Finset.range n → S.card = k →
    ∃ a b, a ∈ S ∧ b ∈ S ∧ a - b = 2) ∧
  (∀ m : ℕ, m < k →
    ∃ T : Finset ℕ, T ⊆ Finset.range n ∧ T.card = m ∧
    ∀ a b, a ∈ T → b ∈ T → a - b ≠ 2) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_selection_for_difference_two_l322_32247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_l322_32212

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (in_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (intersect_planes : Plane → Plane → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the theorem
theorem geometry_propositions
  (α β : Plane)
  (h_distinct : α ≠ β) :
  -- Proposition 1
  (∀ (l1 l2 m1 m2 : Line),
    in_plane l1 α ∧ in_plane l2 α ∧ in_plane m1 β ∧ in_plane m2 β →
    parallel_lines l1 m1 ∧ parallel_lines l2 m2 →
    parallel_planes α β) ∧
  -- Proposition 2
  (∀ (l m : Line),
    ¬in_plane l α ∧ in_plane m α →
    parallel_lines l m →
    ∃ (γ : Plane), in_plane l γ ∧ parallel_planes α γ) ∧
  -- Proposition 3 (negation)
  (∃ (l m : Line),
    intersect_planes α β l ∧ in_plane m α ∧ perpendicular_lines m l ∧
    ¬perpendicular_planes α β) ∧
  -- Proposition 4 (negation)
  (∃ (l m1 m2 : Line),
    perpendicular_lines l m1 ∧ perpendicular_lines l m2 ∧
    in_plane m1 α ∧ in_plane m2 α ∧
    ¬perpendicular_line_plane l α) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_l322_32212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_equals_one_l322_32290

theorem integral_reciprocal_equals_one (a : ℝ) : 
  (∫ x in (1 : ℝ)..a, 1 / x) = 1 → a > 1 → a = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_equals_one_l322_32290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_isosceles_triangles_l322_32241

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  is_regular : sides = n

/-- A diagonal in a regular polygon -/
structure Diagonal (P : RegularPolygon n) where
  endpoints : Fin n × Fin n
  is_diagonal : endpoints.1 ≠ endpoints.2

/-- A "good" diagonal or side in a regular polygon -/
def is_good (P : RegularPolygon n) (d : Sum (Diagonal P) (Fin n)) : Prop :=
  match d with
  | .inl diagonal => ∃ (k : ℕ), 2 * k + 1 < n ∧ 
      (diagonal.endpoints.2 - diagonal.endpoints.1).val % n = 2 * k + 1 ∨
      (diagonal.endpoints.1 - diagonal.endpoints.2).val % n = 2 * k + 1
  | .inr _ => true

/-- A subdivision of a regular polygon into triangles -/
structure Subdivision (P : RegularPolygon n) where
  diagonals : Fin (n - 3) → Diagonal P
  no_intersections : ∀ i j, i ≠ j → 
    (diagonals i).endpoints.1 ≠ (diagonals j).endpoints.1 ∧
    (diagonals i).endpoints.1 ≠ (diagonals j).endpoints.2 ∧
    (diagonals i).endpoints.2 ≠ (diagonals j).endpoints.1 ∧
    (diagonals i).endpoints.2 ≠ (diagonals j).endpoints.2

/-- An isosceles triangle in the subdivision with two "good" sides -/
def is_isosceles_good (P : RegularPolygon n) (S : Subdivision P) 
  (t : Fin n × Fin n × Fin n) : Prop :=
  (is_good P (.inr t.1) ∧ is_good P (.inr t.2.1)) ∨
  (is_good P (.inr t.2.1) ∧ is_good P (.inr t.2.2)) ∨
  (is_good P (.inr t.2.2) ∧ is_good P (.inr t.1)) ∨
  (is_good P (.inl ⟨(t.1, t.2.1), sorry⟩) ∧ is_good P (.inl ⟨(t.2.1, t.2.2), sorry⟩)) ∨
  (is_good P (.inl ⟨(t.2.1, t.2.2), sorry⟩) ∧ is_good P (.inl ⟨(t.2.2, t.1), sorry⟩)) ∨
  (is_good P (.inl ⟨(t.2.2, t.1), sorry⟩) ∧ is_good P (.inl ⟨(t.1, t.2.1), sorry⟩))

/-- The main theorem -/
theorem max_isosceles_triangles 
  (P : RegularPolygon 2006) 
  (S : Subdivision P) :
  (∃ (triangles : Finset (Fin 2006 × Fin 2006 × Fin 2006)), 
    (∀ t ∈ triangles, is_isosceles_good P S t) ∧ 
    triangles.card = 1003) ∧
  (∀ (triangles : Finset (Fin 2006 × Fin 2006 × Fin 2006)), 
    (∀ t ∈ triangles, is_isosceles_good P S t) → 
    triangles.card ≤ 1003) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_isosceles_triangles_l322_32241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l322_32229

/-- The function f(x) defined as 4x + 2/x for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := 4 * x + 2 / x

/-- Theorem stating that the minimum value of f(x) for x > 0 is 4√2 -/
theorem f_min_value (x : ℝ) (hx : x > 0) : 
  f x ≥ 4 * Real.sqrt 2 ∧ ∃ y > 0, f y = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l322_32229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_intersection_ratio_l322_32246

theorem sine_intersection_ratio : ∃ (p q : ℕ+), 
  (p < q) ∧ 
  (∀ (x : ℝ), Real.sin x = Real.sin (π/3) → 
    ∃ (n : ℤ), x = π/3 + 2*π*n ∨ x = 2*π/3 + 2*π*n) ∧
  (∀ (n : ℤ), (2*π/3 + 2*π*n) - (π/3 + 2*π*n) = π/3) ∧
  (Nat.gcd p.val q.val = 1) ∧
  (p = 1 ∧ q = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_intersection_ratio_l322_32246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l322_32208

-- Define the type of functions from positive reals to positive reals
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x, x > 0 → f x > 0}

-- State the theorem
theorem functional_equation_solutions :
  {f : PositiveRealFunction | ∀ x y, x > 0 → y > 0 → f.val (x^y) = (f.val x) ^ (f.val y)} =
  {f : PositiveRealFunction | f.val = λ x ↦ 1} ∪ {f : PositiveRealFunction | f.val = λ x ↦ x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l322_32208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_li_birthdays_l322_32200

def is_leap_year (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def count_leap_years (start_year end_year : Nat) : Nat :=
  List.range (end_year - start_year + 1)
    |>.map (· + start_year)
    |>.filter is_leap_year
    |>.length

theorem li_birthdays :
  count_leap_years 1944 2011 = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_li_birthdays_l322_32200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_cubic_count_l322_32214

-- Define a root of unity
def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ+, z^(n:ℕ) = 1

-- Define the set of roots of unity that satisfy the cubic equation
def roots_of_unity_cubic (c d : ℤ) : Set ℂ :=
  {z : ℂ | is_root_of_unity z ∧ z^3 + c*z + d = 0}

-- Theorem statement
theorem roots_of_unity_cubic_count :
  ∃ c d : ℤ, ∃ (s : Finset ℂ), s.card = 2 ∧ ∀ z ∈ s, z ∈ roots_of_unity_cubic c d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_cubic_count_l322_32214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_section_area_implies_vertex_angle_l322_32248

/-- Represents a right circular cone -/
structure RightCircularCone where
  /-- The angle at the vertex of the axial section -/
  vertex_angle : ℝ

/-- The area of the maximum section through the vertex of the cone -/
noncomputable def area_max_section (cone : RightCircularCone) : ℝ := sorry

/-- The area of the axial section of the cone -/
noncomputable def area_axial_section (cone : RightCircularCone) : ℝ := sorry

/-- 
Given a right circular cone where the area of the maximum section through the vertex 
is twice the area of the axial section, the angle at the vertex of the axial section 
is 5π/6 radians.
-/
theorem max_section_area_implies_vertex_angle (cone : RightCircularCone) 
  (h : area_max_section cone = 2 * area_axial_section cone) : 
  cone.vertex_angle = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_section_area_implies_vertex_angle_l322_32248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_upper_bound_l322_32239

theorem sin_upper_bound (a : ℝ) : (∀ x : ℝ, Real.sin x ≤ a) → a ≥ 1 := by
  intro h
  have h1 : Real.sin (Real.pi / 2) ≤ a := h (Real.pi / 2)
  rw [Real.sin_pi_div_two] at h1
  exact h1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_upper_bound_l322_32239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l322_32219

open Real MeasureTheory

-- Define the equation
def satisfies_equation (y : ℝ) : Prop :=
  sin y ^ 4 - cos y ^ 4 = 1 / cos y + 1 / sin y

-- Define the set of angles that satisfy the equation
def solution_set : Set ℝ :=
  {y | 0 ≤ y ∧ y < 2 * π ∧ satisfies_equation y}

-- State the theorem
theorem sum_of_solutions :
  ∫ y in solution_set, y = 3 * π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l322_32219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_distance_l322_32273

/-- The straight-line distance between two points on a plane given a path with two segments -/
noncomputable def straightLineDistance (northDistance : ℝ) (eastAngle : ℝ) (secondSegment : ℝ) : ℝ :=
  Real.sqrt (34 + 15 * Real.sqrt 2)

/-- Theorem stating that the straight-line distance for the given path is √(34 + 15√2) miles -/
theorem hiking_distance :
  straightLineDistance 3 (π / 4) 5 = Real.sqrt (34 + 15 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_distance_l322_32273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_l322_32294

/-- Number of people to be seated -/
def n : ℕ := 6

/-- Total number of seats -/
def total_seats : ℕ := 10

/-- Number of empty seats -/
def empty_seats : ℕ := total_seats - n

/-- Number of gaps where empty seats can be inserted -/
def gaps : ℕ := n + 1

/-- Calculates the number of seating arrangements with no two empty seats adjacent -/
def no_adjacent_empty_seats : ℕ := n.factorial * (gaps.choose empty_seats)

/-- Calculates the number of seating arrangements with exactly 3 out of 4 empty seats adjacent -/
def three_adjacent_empty_seats : ℕ := n.factorial * (gaps * (gaps - 1))

/-- Calculates the number of seating arrangements with at most 2 out of 4 empty seats adjacent -/
def at_most_two_adjacent_empty_seats : ℕ := 
  n.factorial * (gaps.choose empty_seats + gaps * (gaps - 1).choose 2 + gaps.choose 2)

theorem seating_arrangements :
  no_adjacent_empty_seats = 25200 ∧
  three_adjacent_empty_seats = 30240 ∧
  at_most_two_adjacent_empty_seats = 115920 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_l322_32294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_alpha_plus_gamma_l322_32282

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function f
noncomputable def f (α γ : ℂ) (z : ℂ) : ℂ := (4 + i) * z^2 + α * z + γ

-- State the theorem
theorem min_abs_alpha_plus_gamma :
  ∃ (α γ : ℂ),
    (f α γ 1).im = 0 ∧
    (f α γ i).im = 0 ∧
    ∀ (β δ : ℂ), (f β δ 1).im = 0 → (f β δ i).im = 0 →
      Complex.abs α + Complex.abs γ ≤ Complex.abs β + Complex.abs δ ∧
      Complex.abs α + Complex.abs γ = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_alpha_plus_gamma_l322_32282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l322_32234

def P : Set ℝ := {x : ℝ | |x - 2| ≤ 1}
def Q : Set ℝ := {x : ℝ | ∃ n : ℕ, x = n}

theorem intersection_of_P_and_Q : P ∩ Q = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l322_32234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_lead_second_race_l322_32268

/-- Represents a runner in the race -/
structure Runner where
  speed : ℚ
  deriving Repr

/-- Represents the race scenario -/
structure RaceScenario where
  h : ℚ  -- race distance
  d : ℚ  -- distance Sunny is ahead in the first race
  sunny : Runner
  windy : Runner
  deriving Repr

/-- Calculates the lead of Sunny in the second race -/
def second_race_lead (scenario : RaceScenario) : ℚ :=
  scenario.d^2 / scenario.h

/-- Theorem stating that Sunny's lead in the second race is d²/h -/
theorem sunny_lead_second_race (scenario : RaceScenario) 
    (h_positive : scenario.h > 0)
    (d_positive : scenario.d > 0)
    (h_greater_d : scenario.h > scenario.d)
    (constant_speed : scenario.sunny.speed = scenario.sunny.speed ∧ 
                      scenario.windy.speed = scenario.windy.speed) :
  second_race_lead scenario = scenario.d^2 / scenario.h := by
  sorry

#eval second_race_lead { h := 100, d := 10, sunny := { speed := 10 }, windy := { speed := 9 } }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_lead_second_race_l322_32268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_area_ratio_l322_32264

-- Define the side lengths of the triangles
def large_side : ℝ := 12
def small_side : ℝ := 6

-- Define the areas of the triangles
noncomputable def area_equilateral (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2

-- Define the area of the trapezoid
noncomputable def area_trapezoid : ℝ := area_equilateral large_side - area_equilateral small_side

-- State the theorem
theorem triangle_trapezoid_area_ratio :
  area_equilateral small_side / area_trapezoid = 1 / 3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_area_ratio_l322_32264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_theorem_l322_32287

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The locus of midpoints of chords passing through a point inside a circle -/
noncomputable def midpointLocus (c : Circle) (q : ℝ × ℝ) : Circle :=
  { center := ((c.center.1 + q.1) / 2, (c.center.2 + q.2) / 2),
    radius := dist c.center q / 2 }

theorem midpoint_locus_theorem (c : Circle) (q : ℝ × ℝ) :
  c.radius = 10 →
  dist c.center q = 8 →
  (midpointLocus c q).radius = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_theorem_l322_32287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l322_32216

theorem log_inequality (a b c : ℝ) : 
  a = Real.log 2 / Real.log 0.2 →
  b = Real.log 3 / Real.log 0.2 →
  c = (2 : ℝ) ^ (0.2 : ℝ) →
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l322_32216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l322_32202

-- Define the parametric equations of the ellipse
noncomputable def ellipse_x (t : ℝ) : ℝ := (3 * (Real.sin t + 2)) / (3 - Real.cos t)
noncomputable def ellipse_y (t : ℝ) : ℝ := (4 * (Real.cos t - 1)) / (3 - Real.cos t)

-- State the theorem
theorem ellipse_equation :
  ∃ (A B C D E F : ℤ),
    (∀ t : ℝ, 25 * (ellipse_x t)^2 - 40 * (ellipse_x t) * (ellipse_y t) - 4 * (ellipse_y t)^2 + 
               80 * (ellipse_x t) - 16 * (ellipse_y t) - 100 = 0) ∧
    (Int.gcd A (Int.gcd B (Int.gcd C (Int.gcd D (Int.gcd E F)))) = 1) ∧
    (Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 265) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l322_32202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_area_area_triangle_area_increasing_area_double_bound_l322_32210

/-- The area of a regular n-sided polygon inscribed in a unit circle -/
noncomputable def f (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * Real.sin (2 * Real.pi / n)

theorem regular_polygon_area (n : ℕ) (h : n ≥ 3) :
  f n = (n : ℝ) / 2 * Real.sin (2 * Real.pi / n) := by
  rfl

theorem area_triangle : f 3 = 3 * Real.sqrt 3 / 4 := by
  sorry

theorem area_increasing (n : ℕ) (h : n ≥ 3) : f n < f (n + 1) := by
  sorry

theorem area_double_bound (n : ℕ) (h : n ≥ 3) : f n < f (2 * n) ∧ f (2 * n) ≤ 2 * f n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_area_area_triangle_area_increasing_area_double_bound_l322_32210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_card_types_count_l322_32257

/-- Represents a rectangular grid paper -/
structure GridPaper where
  rows : Nat
  cols : Nat
  frontColor : String
  backColor : String

/-- Represents a card cut from the grid paper -/
structure Card where
  shape : List (Nat × Nat)  -- List of coordinates representing the shape
  frontColor : String
  backColor : String

/-- Checks if two cards are of the same type -/
def sameCardType (c1 c2 : Card) : Prop :=
  c1.shape = c2.shape ∧ c1.frontColor = c2.frontColor ∧ c1.backColor = c2.backColor

/-- Checks if a card's shape is valid within the grid paper -/
def validCardShape (gp : GridPaper) (c : Card) : Prop :=
  ∀ p : Nat × Nat, p ∈ c.shape → p.1 ≤ gp.rows ∧ p.2 ≤ gp.cols

/-- Checks if two cards form a valid cut of the grid paper -/
def validCut (gp : GridPaper) (c1 c2 : Card) : Prop :=
  validCardShape gp c1 ∧ 
  validCardShape gp c2 ∧ 
  c1.shape.length = c2.shape.length ∧
  (∀ p : Nat × Nat, p ∈ c1.shape ∨ p ∈ c2.shape) ∧
  (∀ p : Nat × Nat, ¬(p ∈ c1.shape ∧ p ∈ c2.shape))

/-- The main theorem stating that there are exactly 8 distinct card types -/
theorem distinct_card_types_count (gp : GridPaper) 
  (h1 : gp.rows = 3) (h2 : gp.cols = 4) 
  (h3 : gp.frontColor ≠ gp.backColor) : 
  ∃! (cardTypes : List Card), 
    (∀ c1 c2 : Card, c1 ∈ cardTypes → c2 ∈ cardTypes → c1 ≠ c2 → ¬(sameCardType c1 c2)) ∧ 
    (∀ c1 c2 : Card, validCut gp c1 c2 → ∃ c : Card, c ∈ cardTypes ∧ sameCardType c c1) ∧
    cardTypes.length = 8 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_card_types_count_l322_32257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_positive_f_monotonicity_negative_g_greater_than_f_plus_two_l322_32254

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- State the theorems to be proved
theorem f_monotonicity_positive (a : ℝ) (ha : a ≥ 0) :
  StrictMono (f a) := by sorry

theorem f_monotonicity_negative (a : ℝ) (ha : a < 0) :
  StrictMonoOn (f a) (Set.Ioo 0 (-1/a)) ∧
  StrictAntiOn (f a) (Set.Ioi (-1/a)) := by sorry

theorem g_greater_than_f_plus_two (x : ℝ) (hx : x > 0) :
  g x > f 0 x + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_positive_f_monotonicity_negative_g_greater_than_f_plus_two_l322_32254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_intervals_l322_32259

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2

def monotonic_increase_interval (k : ℤ) : Set ℝ := 
  Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8)

theorem f_monotonic_increase_intervals :
  ∀ x : ℝ, (∃ k : ℤ, x ∈ monotonic_increase_interval k) ↔ 
    (∀ y : ℝ, y ∈ Set.Ioo x (x + Real.pi / 2) → f y > f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_intervals_l322_32259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_worth_l322_32217

/-- Represents the worth of sales in rupees -/
def S : ℝ := sorry

/-- Old commission rate as a decimal -/
def old_rate : ℝ := 0.05

/-- New commission rate as a decimal -/
def new_rate : ℝ := 0.025

/-- Fixed salary in the new scheme -/
def fixed_salary : ℝ := 1000

/-- Sales threshold for new commission in the new scheme -/
def sales_threshold : ℝ := 4000

/-- Difference in remuneration between new and old schemes -/
def remuneration_difference : ℝ := 600

theorem sales_worth :
  fixed_salary + new_rate * (S - sales_threshold) = old_rate * S + remuneration_difference →
  S = 20000 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_worth_l322_32217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnification_is_1000_l322_32291

/-- The magnification factor of an electron microscope -/
noncomputable def magnification_factor (magnified_diameter actual_diameter : ℝ) : ℝ :=
  magnified_diameter / actual_diameter

/-- Theorem: The magnification factor is 1000 given the specified diameters -/
theorem magnification_is_1000 (magnified_diameter actual_diameter : ℝ) 
  (h1 : magnified_diameter = 5)
  (h2 : actual_diameter = 0.005) :
  magnification_factor magnified_diameter actual_diameter = 1000 := by
  -- Unfold the definition of magnification_factor
  unfold magnification_factor
  -- Substitute the given values
  rw [h1, h2]
  -- Perform the division
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnification_is_1000_l322_32291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_ten_numbers_l322_32261

noncomputable def mean (xs : Finset ℝ) : ℝ := (xs.sum id) / xs.card

noncomputable def variance (xs : Finset ℝ) : ℝ :=
  (xs.sum (λ x => (x - mean xs) ^ 2)) / xs.card

theorem mean_of_ten_numbers
  (xs : Finset ℝ)
  (h_card : xs.card = 10)
  (h_positive : ∀ x ∈ xs, x > 0)
  (h_sum_squares : xs.sum (λ x => x ^ 2) = 370)
  (h_variance : variance xs = 33) :
  mean xs = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_ten_numbers_l322_32261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_six_l322_32260

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the expression
noncomputable def expression : ℝ :=
  |1 + log10 0.001| + 
  Real.sqrt ((log10 (1/3))^2 - 4 * log10 3 + 4) + 
  log10 6 - log10 0.02

-- Theorem statement
theorem expression_equals_six : expression = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_six_l322_32260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_problem_l322_32226

/-- The fraction of cookies remaining after John eats each day -/
def daily_remaining_fraction : ℚ := 7/10

/-- The number of days John has been eating cookies -/
def days : ℕ := 3

/-- The number of cookies remaining after 'days' -/
def remaining_cookies : ℕ := 28

/-- The original number of cookies in the box -/
def original_cookies : ℕ := 82

theorem cookie_problem :
  ⌊(daily_remaining_fraction ^ days) * (original_cookies : ℚ)⌋ = remaining_cookies := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_problem_l322_32226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l322_32221

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l322_32221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_theorem_l322_32258

/-- Two lines in the 2D plane -/
structure TwoLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ

/-- The condition that two lines are parallel -/
def are_parallel (lines : TwoLines) : Prop :=
  ∃ (k : ℝ), ∀ x y, lines.line2 x = k * lines.line1 x + y

/-- The specific lines given in the problem -/
noncomputable def problem_lines (a : ℝ) : TwoLines where
  line1 := λ x => (a - 1) * x - 2
  line2 := λ x => (-3 * x + 1) / (a + 3)

/-- The theorem to be proved -/
theorem parallel_lines_theorem :
  ∀ a : ℝ, are_parallel (problem_lines a) → a = 0 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_theorem_l322_32258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangements_count_l322_32211

/-- The number of ways to arrange n distinct objects into k groups with specified sizes. -/
def multinomial (n : ℕ) (sizes : List ℕ) : ℕ :=
  if sizes.sum = n then
    n.factorial / (sizes.map Nat.factorial).prod
  else
    0

/-- The number of arrangements for 6 students into venues A, B, and C. -/
def studentArrangements : ℕ :=
  multinomial 6 [1, 2, 3]

/-- Theorem stating that the number of arrangements is 60. -/
theorem student_arrangements_count :
  studentArrangements = 60 := by
  -- Unfold the definitions
  unfold studentArrangements
  unfold multinomial
  -- Simplify the if-then-else expression
  simp
  -- Evaluate the factorials and perform the division
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangements_count_l322_32211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_whole_number_difference_l322_32274

theorem greatest_whole_number_difference (x y : ℝ) : 
  7 < x → x < 9 → 9 < y → y < 15 → 
  (∃ (n : ℕ), n > 0 ∧ n = ⌊y - x⌋ ∧ ∀ (m : ℕ), m > 0 → m ≤ ⌊y - x⌋ → m ≤ n) → 
  (∃ (n : ℕ), n > 0 ∧ n = ⌊y - x⌋ ∧ ∀ (m : ℕ), m > 0 → m ≤ ⌊y - x⌋ → m ≤ n) ∧ ⌊y - x⌋ = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_whole_number_difference_l322_32274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_remainder_problem_l322_32233

theorem divisor_remainder_problem : 
  let expression := (3^2020 + 3^2021) * (3^2021 + 3^2022) * (3^2022 + 3^2023) * (3^2023 + 3^2024)
  let num_divisors := Nat.card (Nat.divisors expression)
  num_divisors % 1000 = 783 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_remainder_problem_l322_32233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_palindrome_addition_l322_32232

def is_palindrome (n : ℕ) : Prop :=
  (n.digits 10).reverse = n.digits 10

theorem smallest_palindrome_addition : 
  ∀ x : ℕ, x > 0 ∧ is_palindrome (x + 7593) → x ≥ 74 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_palindrome_addition_l322_32232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l322_32265

theorem sin_cos_difference (θ : ℝ) (h1 : Real.sin θ + Real.cos θ = 1/5) (h2 : θ ∈ Set.Ioo 0 Real.pi) : 
  Real.sin θ - Real.cos θ = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l322_32265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cylinder_ratio_l322_32205

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents a cylinder with a given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the surface area of a sphere -/
noncomputable def sphereSurfaceArea (s : Sphere) : ℝ :=
  4 * Real.pi * s.radius^2

/-- Calculates the surface area of a cylinder -/
noncomputable def cylinderSurfaceArea (c : Cylinder) : ℝ :=
  2 * Real.pi * c.radius^2 + 2 * Real.pi * c.radius * c.height

/-- Calculates the volume of a sphere -/
noncomputable def sphereVolume (s : Sphere) : ℝ :=
  (4 / 3) * Real.pi * s.radius^3

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ :=
  Real.pi * c.radius^2 * c.height

theorem sphere_cylinder_ratio (r : ℝ) (hr : r > 0) :
  let s : Sphere := ⟨r⟩
  let c : Cylinder := ⟨r, 2*r⟩
  (sphereSurfaceArea s) / (cylinderSurfaceArea c) = 2/3 ∧
  (sphereVolume s) / (cylinderVolume c) = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cylinder_ratio_l322_32205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angela_action_figures_l322_32266

def calculate_final_collection (initial : ℕ) (increase : ℚ) 
  (sold_fraction : ℚ) (daughter_fraction : ℚ) (nephew_fraction : ℚ) : ℕ :=
  let new_collection := initial + Int.floor (initial * increase)
  let after_selling := new_collection - Int.floor (new_collection * sold_fraction)
  let after_daughter := after_selling - Int.floor (after_selling * daughter_fraction)
  (after_daughter - Int.ceil (after_daughter * nephew_fraction)).toNat

theorem angela_action_figures :
  calculate_final_collection 24 (83/1000) (3/10) (7/15) (1/4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angela_action_figures_l322_32266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_exponential_l322_32218

-- Define the exponential function type
def IsExponentialFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = a^x

-- Define our specific function
noncomputable def f (x : ℝ) : ℝ := (1/3)^x

-- Theorem statement
theorem f_is_exponential : IsExponentialFunction f := by
  -- Provide the value of a
  use (1/3 : ℝ)
  -- Prove the three conditions
  constructor
  · -- Prove 1/3 > 0
    norm_num
  constructor
  · -- Prove 1/3 ≠ 1
    norm_num
  · -- Prove ∀ x, f x = (1/3)^x
    intro x
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_exponential_l322_32218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_travel_time_l322_32293

/-- The time taken for a vehicle to travel a given distance at a constant speed -/
noncomputable def travelTime (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Theorem: A vehicle traveling at 20 m/s takes 5 seconds to cover 100 m -/
theorem vehicle_travel_time :
  travelTime 100 20 = 5 := by
  -- Unfold the definition of travelTime
  unfold travelTime
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_travel_time_l322_32293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_can_prevent_wolf_escape_l322_32286

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square field -/
structure SquareField where
  side : ℝ
  center : Point
  vertices : Fin 4 → Point

/-- Represents the speed of an animal -/
structure Speed where
  value : ℝ

/-- Represents the wolf -/
structure Wolf where
  position : Point
  speed : Speed

/-- Represents a dog -/
structure Dog where
  position : Point
  speed : Speed

/-- Checks if a point is on the perimeter of the square field -/
def isOnPerimeter (field : SquareField) (p : Point) : Prop :=
  (p.x = 0 ∨ p.x = field.side) ∧ (0 ≤ p.y ∧ p.y ≤ field.side) ∨
  (p.y = 0 ∨ p.y = field.side) ∧ (0 ≤ p.x ∧ p.x ≤ field.side)

/-- Checks if two dogs are at the same point -/
def dogsAtSamePoint (d1 d2 : Dog) : Prop :=
  d1.position = d2.position

/-- Theorem: Dogs can prevent the wolf from escaping -/
theorem dogs_can_prevent_wolf_escape (field : SquareField) (wolf : Wolf) (dogs : Fin 4 → Dog) :
  (∀ i, isOnPerimeter field (dogs i).position) →
  (∀ i, (dogs i).speed.value = 1.5 * wolf.speed.value) →
  (∀ p, isOnPerimeter field p → ∃ i j, i ≠ j ∧ dogsAtSamePoint (dogs i) (dogs j)) →
  (∀ t : ℝ, ∃ p : Point, wolf.position = p → ¬(isOnPerimeter field p)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_can_prevent_wolf_escape_l322_32286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_trajectory_l322_32292

-- Define the lines C₁ and C₂
noncomputable def C₁ (α t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, t * Real.sin α)
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the perpendicular line and point A
noncomputable def perpendicular_line (α : ℝ) : ℝ → ℝ := λ x ↦ -x * (Real.cos α / Real.sin α)
noncomputable def point_A (α : ℝ) : ℝ × ℝ := (Real.sin α ^ 2, -Real.cos α * Real.sin α)

-- Define point P as the midpoint of OA
noncomputable def point_P (α : ℝ) : ℝ × ℝ := ((Real.sin α ^ 2) / 2, -(Real.cos α * Real.sin α) / 2)

-- State the theorem
theorem intersection_and_trajectory :
  (∃ θ₁ θ₂, C₂ θ₁ = C₁ (π/3) 1 ∧ C₂ θ₂ = C₁ (π/3) (-1/Real.sqrt 3)) ∧
  (∀ α, ∃ x y, point_P α = (x, y) ∧ (x - 1/4)^2 + y^2 = 1/16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_trajectory_l322_32292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l322_32238

theorem simplify_expression (k : ℝ) (h : k ≠ 0) : (3 * k^2)^(-2 : ℤ) * (2 * k)^4 = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l322_32238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_function_k_range_l322_32206

open Real

theorem monotonic_increasing_function_k_range 
  (f : ℝ → ℝ) 
  (k : ℝ) 
  (h1 : ∀ x, x > 0 → f x = sin x + log x - k * x)
  (h2 : k > 0)
  (h3 : StrictMonoOn f (Set.Ioo 0 (π / 2))) :
  k ∈ Set.Ioo 0 (2 / π) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_function_k_range_l322_32206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_root_finding_l322_32209

-- Define the function f(x) = 2log₅x - 1
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.log x / Real.log 5) - 1

-- Define the bisection method's next interval
def bisection_next_interval (a b : ℝ) (f : ℝ → ℝ) : Set ℝ :=
  let c := (a + b) / 2
  if f a * f c ≤ 0 then Set.Ioo a c else Set.Ioo c b

-- Theorem statement
theorem bisection_root_finding :
  bisection_next_interval 2 3 f = Set.Ioo 2 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_root_finding_l322_32209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l322_32285

noncomputable def f (x y : ℝ) : ℝ := (x * y) / (2 * x^2 + 3 * y^2)

theorem min_value_of_f :
  ∀ x y : ℝ, 0.5 ≤ x → x ≤ 0.7 → 0.3 ≤ y → y ≤ 0.6 →
  f x y ≥ 1 / (4 * Real.sqrt (3/2)) ∧
  ∃ x₀ y₀ : ℝ, 0.5 ≤ x₀ ∧ x₀ ≤ 0.7 ∧ 0.3 ≤ y₀ ∧ y₀ ≤ 0.6 ∧
  f x₀ y₀ = 1 / (4 * Real.sqrt (3/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l322_32285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_estate_investment_analysis_l322_32275

/-- Represents the financial model of the real estate investment --/
structure RealEstateInvestment where
  initial_investment : ℝ
  annual_rental_income : ℝ
  first_year_renovation_cost : ℝ
  renovation_cost_increase : ℝ

/-- Calculates the net profit after n years --/
noncomputable def net_profit (investment : RealEstateInvestment) (n : ℕ) : ℝ :=
  n * investment.annual_rental_income - 
  (investment.initial_investment + 
   n * investment.first_year_renovation_cost + 
   (n * (n - 1) / 2) * investment.renovation_cost_increase)

/-- Calculates the average annual profit after n years --/
noncomputable def average_annual_profit (investment : RealEstateInvestment) (n : ℕ) : ℝ :=
  (net_profit investment n) / n

theorem real_estate_investment_analysis 
  (investment : RealEstateInvestment) 
  (h_initial : investment.initial_investment = 810000)
  (h_rental : investment.annual_rental_income = 300000)
  (h_reno_first : investment.first_year_renovation_cost = 10000)
  (h_reno_increase : investment.renovation_cost_increase = 20000) :
  (∀ n : ℕ, n < 4 → net_profit investment n ≤ 0) ∧ 
  (net_profit investment 4 > 0) ∧
  (∃ n : ℕ, n = 15 ∧ ∀ m : ℕ, net_profit investment m ≤ net_profit investment n) ∧
  (∃ n : ℕ, n = 9 ∧ ∀ m : ℕ, average_annual_profit investment m ≤ average_annual_profit investment n) ∧
  (net_profit investment 15 = net_profit investment 9 + 460000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_estate_investment_analysis_l322_32275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l322_32235

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > Real.sin x)) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l322_32235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_has_winning_strategy_l322_32272

/-- Represents a position on the chessboard -/
structure Position where
  row : Fin 8
  col : Fin 8
deriving DecidableEq

/-- Represents the state of a square on the chessboard -/
inductive SquareState
  | White
  | Black
deriving DecidableEq

/-- Represents the entire chessboard -/
def Chessboard := Position → SquareState

/-- Checks if two positions are adjacent -/
def are_adjacent (p1 p2 : Position) : Prop :=
  (p1.row = p2.row ∧ (p1.col.val + 1 = p2.col.val ∨ p2.col.val + 1 = p1.col.val)) ∨
  (p1.col = p2.col ∧ (p1.row.val + 1 = p2.row.val ∨ p2.row.val + 1 = p1.row.val))

/-- Represents a move by Player A -/
structure MoveA where
  pos1 : Position
  pos2 : Position
  h : are_adjacent pos1 pos2

/-- Represents a move by Player B -/
structure MoveB where
  pos : Position

/-- Applies Player A's move to the chessboard -/
def apply_move_a (board : Chessboard) (move : MoveA) : Chessboard :=
  fun pos => if pos = move.pos1 ∨ pos = move.pos2 then SquareState.Black else board pos

/-- Applies Player B's move to the chessboard -/
def apply_move_b (board : Chessboard) (move : MoveB) : Chessboard :=
  fun pos => if pos = move.pos then SquareState.White else board pos

/-- Checks if a position is a corner of a 5x5 subrectangle -/
def is_corner_5x5 (pos : Position) : Prop :=
  ∃ (i j : Fin 4), 
    (pos.row = i ∧ pos.col = j) ∨
    (pos.row = i ∧ pos.col = j + 4) ∨
    (pos.row = i + 4 ∧ pos.col = j) ∨
    (pos.row = i + 4 ∧ pos.col = j + 4)

/-- Represents Player B's strategy -/
def StrategyB := Chessboard → MoveB

/-- The winning condition for Player B -/
def winning_condition (strategy : StrategyB) : Prop :=
  ∀ (board : Chessboard) (move_a : MoveA),
    ∃ (pos : Position), is_corner_5x5 pos ∧ 
      (apply_move_b (apply_move_a board move_a) (strategy (apply_move_a board move_a))) pos = SquareState.White

/-- The main theorem: Player B has a winning strategy -/
theorem player_b_has_winning_strategy : 
  ∃ (strategy : StrategyB), winning_condition strategy := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_has_winning_strategy_l322_32272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l322_32237

/-- Given vectors a, b, c in ℝ², and a real number lambda,
    if (a + lambda*b) is parallel to c, then lambda = 1/2 -/
theorem parallel_vectors_lambda (a b c : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 2) →
  b = (1, 0) →
  c = (3, 4) →
  (∃ (k : ℝ), k ≠ 0 ∧ a + lambda • b = k • c) →
  lambda = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l322_32237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_tortoise_meeting_l322_32255

/-- Represents the race between a hare and a tortoise -/
structure Race where
  distance : ℝ
  hare_speed : ℝ
  tortoise_speed : ℝ
  lead_at_finish : ℝ

/-- The meeting point after the hare turns back -/
noncomputable def meeting_point (r : Race) : ℝ :=
  r.lead_at_finish * r.tortoise_speed / (r.hare_speed + r.tortoise_speed)

theorem hare_tortoise_meeting (r : Race) 
  (h_distance : r.distance = 100)
  (h_lead : r.lead_at_finish = 75)
  (h_speed_ratio : r.hare_speed = 4 * r.tortoise_speed)
  (h_positive_speeds : r.hare_speed > 0 ∧ r.tortoise_speed > 0) :
  meeting_point r = 60 := by
  sorry

#check hare_tortoise_meeting

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_tortoise_meeting_l322_32255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_graph_implies_logarithm_sum_l322_32256

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem symmetric_graph_implies_logarithm_sum (x : ℝ) (h : x > 0) :
  (∀ y : ℝ, f (Real.exp y) = y) →  -- Symmetry condition
  f (2 * x) = Real.log 2 + Real.log x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_graph_implies_logarithm_sum_l322_32256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l322_32277

/-- Represents a word and its antonym --/
structure WordAntonym :=
  (word : String)
  (antonym : String)

/-- The solution to our puzzle --/
def solution : WordAntonym := ⟨"seldom", "often"⟩

/-- A theorem stating that our solution is correct --/
theorem solution_is_correct :
  solution.word = "seldom" ∧ solution.antonym = "often" := by
  -- Split the conjunction
  apply And.intro
  -- Prove the first part
  rfl
  -- Prove the second part
  rfl

#check solution_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l322_32277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_interval_l322_32283

theorem divisibility_in_interval (n : ℕ) (S : Finset ℕ) : 
  n > 0 → 
  S.card = n + 1 → 
  (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 2*n) → 
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_interval_l322_32283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_is_1600_l322_32222

/-- Proves that the number of 3-inch-by-9-inch tiles needed to cover a 15-foot-by-20-foot room is 1600 -/
def tiles_needed_for_room : ℕ :=
  let room_length : ℚ := 15
  let room_width : ℚ := 20
  let tile_length : ℚ := 3 / 12  -- 3 inches in feet
  let tile_width : ℚ := 9 / 12   -- 9 inches in feet
  let room_area : ℚ := room_length * room_width
  let tile_area : ℚ := tile_length * tile_width
  let tiles_needed : ℚ := room_area / tile_area
  (tiles_needed.ceil.toNat)

/-- The number of tiles needed is 1600 -/
theorem tiles_needed_is_1600 : tiles_needed_for_room = 1600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_is_1600_l322_32222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l322_32271

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ A < Real.pi)
  (h3 : 0 < B ∧ B < Real.pi)
  (h4 : 0 < C ∧ C < Real.pi)
  (h5 : A + B + C = Real.pi)

-- Define the area function
noncomputable def area (t : Triangle) : ℝ := 
  1 / 2 * t.b * t.c * Real.sin t.A

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : (2 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C) :
  t.A = Real.pi / 3 ∧ 
  (t.a = 3 ∧ t.b = 2 * t.c → area t = 3 * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l322_32271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_distinct_objects_l322_32279

def number_of_ways_to_distribute (n m : ℕ) : ℕ := m^n

theorem distribute_distinct_objects (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  number_of_ways_to_distribute n m = m^n := by
  rfl

#eval number_of_ways_to_distribute 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_distinct_objects_l322_32279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l322_32270

/-- An ellipse with equation x²/16 + y²/7 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 16 + p.2^2 / 7 = 1}

/-- The foci of the ellipse -/
noncomputable def F₁ : ℝ × ℝ := sorry
noncomputable def F₂ : ℝ × ℝ := sorry

/-- A straight line passing through F₁ -/
def Line (A : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • F₁ + t • A}

/-- The perimeter of a triangle given by three points -/
def trianglePerimeter (A B C : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C A

theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (hA : A ∈ Ellipse) 
  (hB : B ∈ Ellipse) 
  (hLine : Line A = Line B) :
  trianglePerimeter A B F₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l322_32270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_minus_x_l322_32244

theorem ceiling_minus_x (p q : ℤ) (h1 : q > 1) (h2 : Nat.gcd p.natAbs q.natAbs = 1) 
  (h3 : ⌈(p : ℚ) / q⌉ - ⌊(p : ℚ) / q⌋ = 1) :
  ∃ r : ℤ, 0 < r ∧ r < q ∧ ⌈(p : ℚ) / q⌉ - (p : ℚ) / q = 1 - (r : ℚ) / q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_minus_x_l322_32244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_swap_iff_multiple_of_three_l322_32225

/-- Represents the color of a cell in the grid -/
inductive CellColor
| White
| Black
| Green

/-- Represents a move that repaints a 2x2 sub-square -/
def repaint (grid : List (List CellColor)) : List (List CellColor) :=
  sorry

/-- Checks if the grid is in a checkerboard pattern -/
def isCheckerboard (grid : List (List CellColor)) : Prop :=
  sorry

/-- Checks if the grid has at least one black corner -/
def hasBlackCorner (grid : List (List CellColor)) : Prop :=
  sorry

/-- Checks if two grids have swapped black and white colors -/
def hasSwappedColors (grid1 grid2 : List (List CellColor)) : Prop :=
  sorry

/-- Main theorem: It's possible to swap black and white in a checkerboard pattern
    if and only if n is a multiple of 3 -/
theorem checkerboard_swap_iff_multiple_of_three (n : Nat) :
  (∃ (initial final : List (List CellColor)),
    initial.length = n ∧
    (∀ row ∈ initial, row.length = n) ∧
    isCheckerboard initial ∧
    hasBlackCorner initial ∧
    isCheckerboard final ∧
    hasSwappedColors initial final ∧
    (∃ (moves : List (List (List CellColor))),
      moves.head? = some initial ∧
      moves.getLast? = some final ∧
      ∀ i, i < moves.length - 1 → repaint (moves.get! i) = moves.get! (i + 1)))
  ↔
  ∃ k : Nat, n = 3 * k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_swap_iff_multiple_of_three_l322_32225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_cut_rectangle_properties_l322_32267

/-- Represents a rectangle with diagonals that intersect at the center forming right angles -/
structure DiagonalCutRectangle where
  width : ℝ
  height : ℝ
  diag_intersect_center : Bool
  diag_right_angle : Bool

/-- Calculates the side length of the inscribed square -/
noncomputable def inscribed_square_side (r : DiagonalCutRectangle) : ℝ :=
  min r.width r.height

/-- Calculates the area of a triangular piece -/
noncomputable def triangular_piece_area (r : DiagonalCutRectangle) : ℝ :=
  (r.width * r.height) / 4

/-- Calculates the area of a five-sided piece -/
noncomputable def five_sided_piece_area (r : DiagonalCutRectangle) : ℝ :=
  (r.width * r.height) / 2 - (inscribed_square_side r)^2 / 4

/-- Calculates the area of the rectangular hole when pieces are rearranged -/
noncomputable def hole_area (r : DiagonalCutRectangle) : ℝ :=
  r.width * r.height - (inscribed_square_side r)^2

theorem diagonal_cut_rectangle_properties (r : DiagonalCutRectangle) 
    (h_width : r.width = 20) (h_height : r.height = 30)
    (h_diag_intersect : r.diag_intersect_center = true)
    (h_diag_right_angle : r.diag_right_angle = true) :
  inscribed_square_side r = 20 ∧
  triangular_piece_area r = 100 ∧
  five_sided_piece_area r = 200 ∧
  hole_area r = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_cut_rectangle_properties_l322_32267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_tangent_circles_area_ratio_l322_32288

-- Define a regular hexagon
structure RegularHexagon :=
  (side_length : ℝ)

-- Define a circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the property of a circle being tangent to a line segment
def is_tangent_to_segment (c : Circle) (a b : ℝ × ℝ) : Prop := sorry

-- Define the property of a circle being tangent to a line
def is_tangent_to_line (c : Circle) (l : Set (ℝ × ℝ)) : Prop := sorry

-- Define the area of a circle
noncomputable def circle_area (c : Circle) : ℝ := Real.pi * c.radius^2

-- Main theorem
theorem hexagon_tangent_circles_area_ratio 
  (hex : RegularHexagon)
  (c1 c2 : Circle)
  (A B C D E F : ℝ × ℝ) :
  hex.side_length = 2 →
  is_tangent_to_segment c1 C D →
  is_tangent_to_line c1 {p | p.1 = D.1 ∧ p.2 = E.2} →
  is_tangent_to_line c1 {p | p.1 = B.1 ∧ p.2 = C.2} →
  is_tangent_to_segment c2 F A →
  is_tangent_to_line c2 {p | p.1 = D.1 ∧ p.2 = E.2} →
  is_tangent_to_line c2 {p | p.1 = B.1 ∧ p.2 = C.2} →
  circle_area c1 / circle_area c2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_tangent_circles_area_ratio_l322_32288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_cost_l322_32204

theorem notebook_cost (total_students : Nat) (total_cost : Nat) 
  (h_total_students : total_students = 42)
  (h_total_cost : total_cost = 2457)
  (s : Nat) (c : Nat) (n : Nat)
  (h_majority : s > total_students / 2)
  (h_same_number : ∀ (student : Nat), student ≤ s → n > 0)
  (h_more_than_two : n > 2)
  (h_cost_gt_number : c > n)
  (h_total_cost_eq : s * c * n = total_cost) :
  c = 19 ∨ c = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_cost_l322_32204
