import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l781_78195

noncomputable def data_set : List ℝ := [9, 12, 10, 9, 11, 9, 10]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ)^2)).sum / xs.length

theorem variance_of_dataset :
  variance data_set = 8/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l781_78195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_even_is_some_odd_l781_78186

theorem negation_of_all_even_is_some_odd :
  (¬ ∀ x : ℕ, x > 0 → Even x) ↔ (∃ x : ℕ, x > 0 ∧ Odd x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_even_is_some_odd_l781_78186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infection_probabilities_l781_78177

/-- Represents the infection status of a person -/
inductive InfectionStatus
  | Infected
  | NotInfected

/-- Represents the source of infection for a person -/
inductive InfectionSource
  | A
  | B
  | C
  | None

/-- Models the infection scenario -/
structure InfectionModel where
  /-- Probability of B, C, and D being infected by A -/
  prob_infected_by_A : ℝ
  /-- B's infection status (always infected by A) -/
  B_status : InfectionStatus
  /-- Probability of C being infected by A or B -/
  prob_C_infected_by_A_or_B : ℝ
  /-- Probability of D being infected by A, B, or C -/
  prob_D_infected_by_A_B_or_C : ℝ

/-- Calculates the probability that exactly one of B, C, and D is infected -/
noncomputable def prob_exactly_one_infected (model : InfectionModel) : ℝ :=
  3 * (1/2) * (1/2)^2

/-- Calculates the expected number of people directly infected by A among B, C, and D -/
noncomputable def expected_infected_by_A (model : InfectionModel) : ℝ :=
  1 * (1/3) + 2 * (1/2) + 3 * (1/6)

/-- Main theorem stating the probabilities and expected value -/
theorem infection_probabilities (model : InfectionModel) 
  (h1 : model.prob_infected_by_A = 1/2)
  (h2 : model.B_status = InfectionStatus.Infected)
  (h3 : model.prob_C_infected_by_A_or_B = 1/2)
  (h4 : model.prob_D_infected_by_A_B_or_C = 1/3) :
  prob_exactly_one_infected model = 3/8 ∧ 
  expected_infected_by_A model = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infection_probabilities_l781_78177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_solution_l781_78144

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

theorem existence_of_solution :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    frac x + ↑(floor y) + frac z = 2.9 ∧
    frac y + ↑(floor z) + frac x = 5.3 ∧
    frac z + ↑(floor x) + frac y = 4.0 := by
  sorry

#check existence_of_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_solution_l781_78144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_triangles_congruent_l781_78172

/-- A structure representing a triangle in a plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The inradius of a triangle -/
noncomputable def inradius (t : Triangle) : ℝ := sorry

/-- A structure representing 4 points in a plane -/
structure FourPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The 4 triangles formed by 4 points -/
def triangles (fp : FourPoints) : Fin 4 → Triangle
  | 0 => ⟨fp.A, fp.B, fp.C⟩
  | 1 => ⟨fp.B, fp.C, fp.D⟩
  | 2 => ⟨fp.C, fp.D, fp.A⟩
  | 3 => ⟨fp.D, fp.A, fp.B⟩

/-- Two triangles are congruent -/
def congruent (t1 t2 : Triangle) : Prop := sorry

theorem four_triangles_congruent (fp : FourPoints) :
  (∀ i j : Fin 4, inradius (triangles fp i) = inradius (triangles fp j)) →
  (∀ i j : Fin 4, congruent (triangles fp i) (triangles fp j)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_triangles_congruent_l781_78172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l781_78152

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O2 (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 16

-- Define the centers and radii
def center_O1 : ℝ × ℝ := (0, 0)
def center_O2 : ℝ × ℝ := (3, -4)
def radius_O1 : ℝ := 1
def radius_O2 : ℝ := 4

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center_O1.1 - center_O2.1)^2 + (center_O1.2 - center_O2.2)^2)

-- Theorem: The circles are externally tangent
theorem circles_externally_tangent :
  distance_between_centers = radius_O1 + radius_O2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l781_78152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_cardinality_l781_78153

/-- The number of subsets of a set S -/
noncomputable def n (S : Finset ℕ) : ℕ := 2^S.card

/-- The theorem statement -/
theorem min_intersection_cardinality
  (A B C : Finset ℕ)
  (h1 : n A + n B + n C = n (A ∪ B ∪ C))
  (h2 : A.card = 100)
  (h3 : B.card = 100) :
  ∃ (D : Finset ℕ), D = A ∩ B ∩ C ∧ D.card ≥ 97 ∧
  ∀ (E : Finset ℕ), E = A ∩ B ∩ C → E.card ≥ D.card :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_cardinality_l781_78153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_f_monotone_increasing_range_l781_78190

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x - x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := x + a / x - 1

/-- Theorem stating the monotone increasing condition -/
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ici 1, MonotoneOn (f a) (Set.Ici 1)) ↔ a ∈ Set.Ici 0 := by sorry

/-- The main theorem -/
theorem f_monotone_increasing_range (a : ℝ) :
  (∀ x ∈ Set.Ici 1, MonotoneOn (f a) (Set.Ici 1)) ↔ a ∈ Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_f_monotone_increasing_range_l781_78190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_freezes_at_negative_ten_l781_78187

/-- The freezing point of water under standard atmospheric pressure in Celsius --/
def water_freezing_point : ℝ := 0

/-- A temperature in Celsius --/
def temperature : ℝ := -10

/-- Represents the state of water --/
inductive WaterState
| Liquid
| Frozen

/-- Represents a water sample --/
structure WaterSample where
  state_at_temp : ℝ → WaterState

/-- Theorem stating that water freezes at -10°C under standard atmospheric pressure --/
theorem water_freezes_at_negative_ten :
  temperature < water_freezing_point → 
  (∀ water_sample : WaterSample, water_sample.state_at_temp temperature = WaterState.Frozen) :=
by
  sorry

#check water_freezes_at_negative_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_freezes_at_negative_ten_l781_78187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_6789_l781_78196

noncomputable def c : ℝ := 4 + Real.sqrt 15
noncomputable def d : ℝ := 4 - Real.sqrt 15

noncomputable def S (n : ℕ) : ℝ := (1/2) * (c^n + d^n)

theorem units_digit_S_6789 : ∃ k : ℤ, S 6789 = k + 4 ∧ k % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_6789_l781_78196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_when_m_2_A_superset_B_iff_l781_78159

-- Define sets A and B
def A : Set ℝ := {x | (1/32 : ℝ) ≤ Real.exp (-x * Real.log 2) ∧ Real.exp (-x * Real.log 2) ≤ 4}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*m*x - 3*m^2 < 0}

-- Theorem 1: Intersection of A and B when m = 2
theorem intersection_A_B_when_m_2 : 
  A ∩ B 2 = {x : ℝ | -2 ≤ x ∧ x < 2} := by sorry

-- Theorem 2: Condition for A to be a superset of B
theorem A_superset_B_iff (m : ℝ) : 
  m > 0 → (A ⊇ B m ↔ m ≤ 2/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_when_m_2_A_superset_B_iff_l781_78159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_A_when_area_maximized_l781_78135

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the semi-perimeter of a triangle -/
noncomputable def semiPerimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

/-- Calculates the area of a triangle using Heron's formula -/
noncomputable def area (t : Triangle) : ℝ :=
  let p := semiPerimeter t
  Real.sqrt (p * (p - t.a) * (p - t.b) * (p - t.c))

/-- Theorem: Maximum value of sin A in triangle ABC with BC = 6 and AB = 2AC -/
theorem max_sin_A_when_area_maximized :
  ∃ (t : Triangle),
    t.c = 6 ∧
    t.a = 2 * t.b ∧
    (∀ (t' : Triangle), t'.c = 6 → t'.a = 2 * t'.b → area t ≥ area t') →
    Real.sin (Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))) = 3/5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_A_when_area_maximized_l781_78135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l781_78149

/-- The area of the region formed by P(a, b) given the specified conditions -/
theorem area_of_region (a b : ℝ) : 
  (∀ x y : ℝ, |x| ≤ 1 → |y| ≤ 1 → a * x - 2 * b * y ≤ 2) →
  (MeasureTheory.volume {p : ℝ × ℝ | ∃ (a b : ℝ), p = (a, b) ∧ 
    (∀ x y : ℝ, |x| ≤ 1 → |y| ≤ 1 → a * x - 2 * b * y ≤ 2)}) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l781_78149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_time_problem_l781_78109

-- Define the function f as noncomputable
noncomputable def f (x a c : ℝ) : ℝ :=
  if x < a then c / Real.sqrt x else c / Real.sqrt a

-- State the theorem
theorem assembly_time_problem (a c : ℝ) 
  (h1 : f 4 a c = 30)
  (h2 : f a a c = 5) :
  c = 60 ∧ a = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_time_problem_l781_78109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_width_always_less_than_length_remaining_piece_not_square_l781_78199

/-- Represents the dimensions of a rectangular piece of cake -/
structure CakePiece where
  length : ℝ
  width : ℝ

/-- Represents a cut made on the cake -/
inductive Cut
  | Horizontal
  | Vertical

/-- The state of the cake after each cut -/
structure CakeState where
  remainder : CakePiece
  lastCut : Cut

/-- Performs a cut on the cake -/
def makeCut (state : CakeState) : CakeState :=
  match state.lastCut with
  | Cut.Horizontal => CakeState.mk (CakePiece.mk state.remainder.width (state.remainder.length - state.remainder.width)) Cut.Vertical
  | Cut.Vertical => CakeState.mk (CakePiece.mk (state.remainder.length - state.remainder.width) state.remainder.width) Cut.Horizontal

/-- The theorem stating that the width is always less than the length after any number of cuts -/
theorem width_always_less_than_length (initialSide : ℝ) (n : ℕ) :
  let initialState := CakeState.mk (CakePiece.mk initialSide initialSide) Cut.Horizontal
  let finalState := (n.iterate makeCut initialState)
  finalState.remainder.width < finalState.remainder.length := by
  sorry

/-- The main theorem stating that the remaining piece cannot be a square -/
theorem remaining_piece_not_square (initialSide : ℝ) (n : ℕ) :
  let initialState := CakeState.mk (CakePiece.mk initialSide initialSide) Cut.Horizontal
  let finalState := (n.iterate makeCut initialState)
  finalState.remainder.width ≠ finalState.remainder.length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_width_always_less_than_length_remaining_piece_not_square_l781_78199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_repair_problem_l781_78128

/-- The total length of the road in meters -/
noncomputable def total_length : ℝ := 225

/-- The length of road repaired in two days in meters -/
noncomputable def repaired_length : ℝ := 135

/-- Fraction of the road repaired on the first day -/
noncomputable def first_day_fraction : ℝ := 1/3

/-- Fraction of the remaining road repaired on the second day -/
noncomputable def second_day_fraction : ℝ := 2/5

theorem road_repair_problem :
  let remaining_after_first_day := total_length * (1 - first_day_fraction)
  let second_day_repair := remaining_after_first_day * second_day_fraction
  total_length * first_day_fraction + second_day_repair = repaired_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_repair_problem_l781_78128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_average_speed_l781_78164

/-- Calculates the average speed given two trips with distances and times -/
noncomputable def average_speed (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) : ℝ :=
  (distance1 + distance2) / (time1 + time2)

/-- Theorem: Linda's average speed for the entire trip is approximately 62.86 miles per hour -/
theorem linda_average_speed :
  let distance1 : ℝ := 450  -- miles
  let time1 : ℝ := 7.5      -- hours (7 hours 30 minutes)
  let distance2 : ℝ := 540  -- miles
  let time2 : ℝ := 8.25     -- hours (8 hours 15 minutes)
  let avg_speed := average_speed distance1 time1 distance2 time2
  abs (avg_speed - 62.86) < 0.01 := by
  sorry

-- Note: We can't use #eval with noncomputable functions, so we'll use #check instead
#check average_speed 450 7.5 540 8.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_average_speed_l781_78164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_at_negative_one_l781_78137

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the first derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Define the second derivative of f
def f'' (a b : ℝ) (x : ℝ) : ℝ := 12 * a * x^2 + 2 * b

-- Theorem statement
theorem second_derivative_at_negative_one 
  (a b c : ℝ) (h : f' a b 1 = 2) : f'' a b (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_at_negative_one_l781_78137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l781_78178

-- Define the sequence
noncomputable def a : ℕ → ℝ
| 0 => 0  -- Adding a case for 0 to avoid missing cases error
| 1 => Real.sqrt 2
| 2 => Real.sqrt 5
| 3 => Real.sqrt 8
| 4 => Real.sqrt 11
| n + 1 => Real.sqrt (3 * n - 1)  -- General formula for n ≥ 5

-- State the theorem
theorem sequence_general_term : ∀ n : ℕ, n > 0 → a n = Real.sqrt (3 * n - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l781_78178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_2cos6_l781_78100

theorem min_sin6_2cos6 :
  ∀ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ (3 + 2 * Real.sqrt 2) / 12 ∧
  ∃ y : ℝ, Real.sin y ^ 6 + 2 * Real.cos y ^ 6 = (3 + 2 * Real.sqrt 2) / 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_2cos6_l781_78100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l781_78189

/-- The time taken for two trains to cross each other -/
noncomputable def time_to_cross (train_length : ℝ) (time1 time2 : ℝ) : ℝ :=
  (2 * train_length) / (train_length / time1 + train_length / time2)

/-- Theorem: Two trains of equal length 120 m, with crossing times of 9 sec and 15 sec respectively, 
    will cross each other in approximately 11.25 seconds when traveling in opposite directions -/
theorem trains_crossing_time :
  let train_length : ℝ := 120
  let time1 : ℝ := 9
  let time2 : ℝ := 15
  let crossing_time := time_to_cross train_length time1 time2
  ∃ ε > 0, abs (crossing_time - 11.25) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l781_78189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l781_78123

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ + Real.pi / 4)

theorem f_monotone_decreasing 
  (ω φ : ℝ) 
  (h_ω : 5/2 < ω ∧ ω < 9/2) 
  (h_φ : 0 < φ ∧ φ < Real.pi) 
  (h_even : ∀ x, f ω φ x = f ω φ (-x)) 
  (h_eq : f ω φ 0 = f ω φ Real.pi) : 
  StrictMonoOn (fun x ↦ f ω φ (-x)) (Set.Ioo 0 (Real.pi/4)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l781_78123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_lele_correct_lele_correct_qiangqiang_first_l781_78120

-- Define the set of children
inductive Child : Type
  | Baby : Child
  | Star : Child
  | Lele : Child
  | Qiangqiang : Child

-- Define the first place
def first_place : Child := Child.Qiangqiang

-- Define the prediction function
def prediction (c : Child) : Prop :=
  match c with
  | Child.Star => (first_place = Child.Lele)
  | Child.Baby => (first_place = Child.Star)
  | Child.Lele => (first_place ≠ Child.Lele)
  | Child.Qiangqiang => (first_place ≠ Child.Qiangqiang)

-- Theorem stating that only Lele's prediction is correct
theorem only_lele_correct :
  (∀ c : Child, c ≠ Child.Lele → ¬(prediction c)) ∧
  (prediction Child.Lele) := by
  sorry

-- Main theorem to prove
theorem lele_correct_qiangqiang_first :
  (prediction Child.Lele) ∧ (first_place = Child.Qiangqiang) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_lele_correct_lele_correct_qiangqiang_first_l781_78120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l781_78106

noncomputable section

open Real

-- Define the original function
def f (x : ℝ) : ℝ := sin (2 * x)

-- Define the shifted function
def g (x : ℝ) : ℝ := f (x + π / 4)

-- Define the stretched function
def h (x : ℝ) : ℝ := g (x / 2)

-- Theorem statement
theorem function_transformation (x : ℝ) : h x = cos x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l781_78106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l781_78140

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^a

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (f a x)

theorem min_value_g (a : ℝ) :
  (f a 2 = 1/2) →
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, g a x ≥ -1) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, g a x = -1) :=
by
  sorry

#check min_value_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l781_78140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_theorem_l781_78122

-- Define the symmetry relations
def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

def symmetric_wrt_x_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, g x = -f x

-- State the theorem
theorem symmetry_theorem (f g : ℝ → ℝ) (a : ℝ) :
  symmetric_wrt_y_eq_x f (λ x => Real.exp x) →
  symmetric_wrt_x_axis f g →
  g a = 1 →
  a = 1 / Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_theorem_l781_78122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l781_78105

theorem trigonometric_identity (α : ℝ) 
  (h1 : Real.sin α * Real.cos α = 1/8)
  (h2 : π < α ∧ α < 3*π/2) : 
  (1 - Real.cos α^2) / (Real.cos (3*π/2 - α) + Real.cos α) + 
  (Real.sin (α - 7*π/2) + Real.sin (2017*π - α)) / (Real.tan α^2 - 1) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l781_78105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_DEF_l781_78181

/-- The diameter of the inscribed circle in a triangle with sides a, b, and c. -/
noncomputable def inscribed_circle_diameter (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (4 * area) / (a + b + c)

/-- Theorem: The diameter of the circle inscribed in triangle DEF is 8√3/3. -/
theorem inscribed_circle_diameter_DEF :
  inscribed_circle_diameter 13 7 10 = 8 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_DEF_l781_78181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l781_78104

theorem constant_term_expansion (a : ℝ) : 
  (∃ k : ℕ, (Nat.choose 6 k) * (-1)^k * a^k = -160) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l781_78104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l781_78130

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) + 2 * Real.sqrt 3 * (Real.sin (ω * x))^2 - Real.sqrt 3

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x + Real.pi / 6) + 1

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def num_zeros_in_interval (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem f_properties (ω : ℝ) (h1 : ω > 0) (h2 : has_period (f ω) Real.pi) :
  (∀ k : ℤ, is_increasing_on (f ω) (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)) ∧
  (num_zeros_in_interval (g ω) 0 (59 * Real.pi / 12) ≥ 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l781_78130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l781_78173

theorem simplify_trig_expression (θ : Real) 
  (h1 : Real.sin θ < 0) 
  (h2 : Real.tan θ > 0) : 
  Real.sqrt (1 - Real.sin θ ^ 2) = -Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l781_78173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l781_78101

-- Define the circle C
def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*a*y + a^2 - 2 = 0

-- Define the fixed point P
def P : ℝ × ℝ := (2, -1)

-- Define the length of PT
noncomputable def PT_length (a : ℝ) : ℝ :=
  Real.sqrt ((a + 1)^2 + 2)

theorem min_tangent_length :
  ∃ (min_length : ℝ), min_length = Real.sqrt 2 ∧
  ∀ (a : ℝ), PT_length a ≥ min_length := by
  sorry

#check min_tangent_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l781_78101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_calculation_l781_78150

/-- Given a river with specified depth, flow rate, and water volume per minute,
    calculate its width. -/
theorem river_width_calculation (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) 
    (h1 : depth = 2)
    (h2 : flow_rate_kmph = 5)
    (h3 : volume_per_minute = 7500) :
    volume_per_minute / (depth * (flow_rate_kmph * 1000 / 60)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_calculation_l781_78150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determinant_zero_l781_78193

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = Real.pi) : 
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![Real.cos A ^ 2, Real.tan A, 1],
    ![Real.cos B ^ 2, Real.tan B, 1],
    ![Real.cos C ^ 2, Real.tan C, 1]
  ]
  Matrix.det M = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determinant_zero_l781_78193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_third_side_l781_78126

theorem greatest_integer_third_side (a b : ℝ) (ha : a = 7) (hb : b = 15) :
  ∃ (c : ℕ), c = 21 ∧ 
  (∀ (x : ℕ), x > c → ¬(a + b > x ∧ x + a > b ∧ x + b > a)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_third_side_l781_78126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l781_78155

/-- Helper function to represent the terminal side of an angle -/
noncomputable def TerminalSide (angle : ℝ) : ℝ × ℝ :=
  (Real.cos angle, Real.sin angle)

/-- For all integers k, the angles 2kπ+π and 4k±π have the same terminal side -/
theorem same_terminal_side (k : ℤ) : 
  TerminalSide (2 * k * Real.pi + Real.pi) = TerminalSide (4 * k * Real.pi + Real.pi) ∧
  TerminalSide (2 * k * Real.pi + Real.pi) = TerminalSide (4 * k * Real.pi - Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l781_78155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_12_neg_gt_neg150_l781_78151

theorem largest_multiple_12_neg_gt_neg150 : 
  ∀ n : ℤ, n > 0 → 12 ∣ n → -n > -150 → n ≤ 144 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_12_neg_gt_neg150_l781_78151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_quad_area_theorem_l781_78114

/-- Represents a parallelogram with side lengths a and b -/
structure Parallelogram (a b : ℝ) where
  area_eq_one : a * b * Real.sin (Real.arccos ((a^2 + b^2) / (2*a*b))) = 1
  a_lt_b : a < b
  b_lt_two_a : b < 2 * a

/-- The area of the quadrilateral formed by the internal angle bisectors of a parallelogram -/
noncomputable def angle_bisector_quad_area (a b : ℝ) (p : Parallelogram a b) : ℝ :=
  (b - a)^2 / (2 * a * b)

/-- Theorem: The area of the quadrilateral formed by the internal angle bisectors
    of a parallelogram with area 1 and side lengths a and b, where a < b < 2a,
    is (b-a)^2 / (2ab) -/
theorem angle_bisector_quad_area_theorem (a b : ℝ) (p : Parallelogram a b) :
  angle_bisector_quad_area a b p = (b - a)^2 / (2 * a * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_quad_area_theorem_l781_78114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_M_l781_78108

def M : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_divisors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_M_l781_78108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_after_n_years_l781_78141

/-- The cost function after n years with initial cost a and yearly reduction rate p% -/
noncomputable def cost_function (a : ℝ) (p : ℝ) (n : ℕ) : ℝ :=
  a * (1 - p / 100) ^ n

/-- Theorem stating that the cost after n years is equal to a(1-p%)^n -/
theorem cost_after_n_years (a : ℝ) (p : ℝ) (n : ℕ) (h1 : a > 0) (h2 : 0 ≤ p ∧ p ≤ 100) :
  cost_function a p n = a * (1 - p / 100) ^ n :=
by
  -- Unfold the definition of cost_function
  unfold cost_function
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_after_n_years_l781_78141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_teachers_l781_78103

theorem stratified_sampling_teachers (total : ℕ) (senior intermediate junior : ℕ) (sample : ℕ)
  (h_total : total = 300)
  (h_senior : senior = 90)
  (h_intermediate : intermediate = 150)
  (h_junior : junior = 60)
  (h_sum : senior + intermediate + junior = total)
  (h_sample : sample = 40) :
  (senior * sample) / total = 12 ∧
  (intermediate * sample) / total = 20 ∧
  sample - (senior * sample) / total - (intermediate * sample) / total = 8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_teachers_l781_78103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l781_78157

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + Real.sqrt 2 * y - 2 * Real.sqrt 2 = 0

-- Define points A, B, M, and N
noncomputable def point_A : ℝ × ℝ := sorry
noncomputable def point_B : ℝ × ℝ := sorry
noncomputable def point_M : ℝ × ℝ := sorry
noncomputable def point_N : ℝ × ℝ := sorry

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_equation_proof :
  -- Conditions
  (ellipse point_A.1 point_A.2) →
  (ellipse point_B.1 point_B.2) →
  (point_A.1 > 0 ∧ point_A.2 > 0) →
  (point_B.1 > 0 ∧ point_B.2 > 0) →
  (point_M.2 = 0) →
  (point_N.1 = 0) →
  (distance point_M point_A = distance point_N point_B) →
  (distance point_M point_N = 2 * Real.sqrt 3) →
  -- Conclusion
  ∀ x y : ℝ, line_l x y ↔ (x + Real.sqrt 2 * y - 2 * Real.sqrt 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l781_78157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l781_78115

/-- Given a function f(x) = A * sin(x + π/3) where x ∈ ℝ, 
    if f(5π/12) = 3√2/2 and f(θ) - f(-θ) = √3 for θ ∈ (0, π/2),
    then A = 3 and f(π/6 - θ) = √6 -/
theorem function_properties (A : ℝ) (θ : ℝ) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = A * Real.sin (x + π/3))
  (h1 : f (5*π/12) = 3*Real.sqrt 2/2)
  (h2 : θ > 0 ∧ θ < π/2)
  (h3 : f θ - f (-θ) = Real.sqrt 3) :
  A = 3 ∧ f (π/6 - θ) = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l781_78115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l781_78142

noncomputable def f (x : ℝ) : ℝ := 3^x - 6/x

theorem range_of_f :
  let S := {y | ∃ x ∈ Set.Icc 1 2, f x = y}
  S = Set.Icc (-3) 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l781_78142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_double_angle_quadratic_l781_78182

theorem cosine_double_angle_quadratic 
  (a b c : ℝ) :
  let f := λ x => a * (Real.cos x)^2 + b * Real.cos x + c
  let g := λ x => a^2 * (Real.cos x)^2 + (2*a^2 + 4*a*c - 2*b^2) * Real.cos x + a^2 + 4*a*c - 2*b^2 + 4*c^2
  (∀ x, f x = 0) → (∀ x, g (2*x) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_double_angle_quadratic_l781_78182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_positive_probability_l781_78127

/-- The lower bound of the interval -/
noncomputable def lower_bound : ℝ := -30

/-- The upper bound of the interval -/
noncomputable def upper_bound : ℝ := 15

/-- The probability of selecting a number from the negative part of the interval -/
noncomputable def prob_negative : ℝ := (0 - lower_bound) / (upper_bound - lower_bound)

/-- The probability of selecting a number from the positive part of the interval -/
noncomputable def prob_positive : ℝ := (upper_bound - 0) / (upper_bound - lower_bound)

/-- The theorem stating that the probability of the product being positive is 5/9 -/
theorem product_positive_probability :
  prob_positive ^ 2 + prob_negative ^ 2 = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_positive_probability_l781_78127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_is_correct_l781_78170

/-- The parabola equation y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- The line y = 2x -/
def line (x y : ℝ) : Prop := y = 2*x

/-- The point symmetric to the focus -/
noncomputable def symmetric_point : ℝ × ℝ := (-3/5, 4/5)

/-- Theorem stating that the symmetric_point is indeed symmetric to the focus about the line -/
theorem symmetric_point_is_correct : 
  let (x₁, y₁) := focus
  let (x₂, y₂) := symmetric_point
  (y₁ + y₂)/2 = 2*((x₁ + x₂)/2) ∧ 
  (y₂ - y₁)/(x₂ - x₁) = -1/(2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_is_correct_l781_78170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l781_78168

-- Define the speeds of the trains in km/hr
noncomputable def speed_faster : ℚ := 45
noncomputable def speed_slower : ℚ := 15

-- Define the time taken for the slower train to pass the driver of the faster train in seconds
noncomputable def passing_time : ℚ := 60

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℚ := 5 / 18

-- Theorem statement
theorem train_length_calculation :
  let relative_speed : ℚ := (speed_faster + speed_slower) * km_hr_to_m_s
  let train_length : ℚ := relative_speed * passing_time
  train_length = 1000 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l781_78168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_triangle_ratio_l781_78175

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x ^ 2 + cos (π / 4 - x) ^ 2 - (1 + sqrt 3) / 2

-- Theorem for the maximum value of f
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (π / 2) ∧ f x = 1 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 (π / 2) → f y ≤ f x :=
sorry

-- Theorem for the triangle problem
theorem triangle_ratio (A B C : ℝ) (hABC : A + B + C = π) (hAB : A < B) (hfA : f A = 1/2) (hfB : f B = 1/2) :
  sin A / sin C = sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_triangle_ratio_l781_78175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_due_approximation_l781_78167

/-- Calculates the sum due given the true discount, interest rate, and time period -/
noncomputable def sum_due (true_discount : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  true_discount / (1 - (1 + interest_rate)^(-time))

/-- Theorem stating that the sum due is approximately 347.15 given the specified conditions -/
theorem sum_due_approximation :
  let td := 168
  let r := 0.18
  let t := 4
  abs (sum_due td r t - 347.15) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_due_approximation_l781_78167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_centroid_l781_78132

/-- Parabola type representing y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point type representing a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if a point lies on a parabola -/
def on_parabola (p : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * p.p * pt.x

/-- Function to calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem statement -/
theorem parabola_focus_centroid (p : Parabola) (A B F : Point) (h_A : on_parabola p A)
    (h_B : on_parabola p B) (h_dist : distance A F + distance B F = 10)
    (h_centroid : F.x = (A.x + B.x) / 3 ∧ F.y = (A.y + B.y) / 3)
    (h_focus : F.x = p.p / 2 ∧ F.y = 0) : p.p = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_centroid_l781_78132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_expression_approx_1_1_l781_78112

-- Define the original expression
noncomputable def original_expression : ℝ := Real.sqrt (Real.rpow 0.001024 (1/3))

-- Define the approximation
def approximation : ℝ := 1.1

-- Define the tolerance for rounding to the nearest tenth
def tolerance : ℝ := 0.05

-- Theorem statement
theorem original_expression_approx_1_1 :
  |original_expression - approximation| < tolerance := by
  sorry

#eval approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_expression_approx_1_1_l781_78112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratification_ratio_rounded_l781_78119

theorem ratification_ratio_rounded : 
  (round ((9 : ℝ) / 13 * 10) / 10 : ℝ) = 0.7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratification_ratio_rounded_l781_78119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_periodic_g_is_even_l781_78184

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin (abs x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (x + 5 * Real.pi / 2)

-- Statement 1: f is not a periodic function
theorem f_not_periodic : ¬∃ (p : ℝ), p ≠ 0 ∧ ∀ (x : ℝ), f (x + p) = f x := by sorry

-- Statement 2: g is an even function
theorem g_is_even : ∀ (x : ℝ), g x = g (-x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_periodic_g_is_even_l781_78184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_at_loss_l781_78198

/-- The cost of two books and their selling conditions -/
def TwoBooks (c1 c2 : ℝ) : Prop :=
  c1 + c2 = 470 ∧
  0.85 * c1 = 1.19 * c2

theorem book_cost_at_loss (c1 c2 : ℝ) (h : TwoBooks c1 c2) :
  abs (c1 - 274.11) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_at_loss_l781_78198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_week_rate_is_18_l781_78169

/-- The daily rate for a student youth hostel stay. -/
noncomputable def daily_rate (days : ℕ) : ℝ → ℝ :=
  fun rate => if days ≤ 7 then rate else 13

/-- The total cost for a stay of given duration at a given first-week rate. -/
noncomputable def total_cost (days : ℕ) (first_week_rate : ℝ) : ℝ :=
  (min days 7 : ℝ) * first_week_rate + (max (days - 7) 0 : ℝ) * 13

/-- Theorem stating that the daily rate for the first week is $18.00 -/
theorem first_week_rate_is_18 :
  ∃ (rate : ℝ), total_cost 23 rate = 334 ∧ rate = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_week_rate_is_18_l781_78169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_three_l781_78163

-- Define the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y - 2 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem tangent_length_is_three :
  ∃ (x y : ℝ), 
    circle_equation x y ∧ 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2).sqrt = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_three_l781_78163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l781_78161

noncomputable def f (x : ℝ) := (1 + 1 / Real.log (Real.sqrt (x^2 + 10) - x)) * 
                                (1 + 2 / Real.log (Real.sqrt (x^2 + 10) - x))

theorem f_lower_bound (x : ℝ) (h1 : 0 < x) (h2 : x < 4.5) : 
  f x > 8 + 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l781_78161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_theorem_l781_78180

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 9
def F₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the trajectory L
def L (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l
def l (m x y : ℝ) : Prop := x = m * y + 1

-- Define the area of triangle ABF₁
noncomputable def area_ABF₁ (m : ℝ) : ℝ := (12 * Real.sqrt (m^2 + 1)) / (3 * m^2 + 4)

theorem moving_circle_theorem :
  ∀ (x y m : ℝ),
  (∃ (R : ℝ), ∀ (x' y' : ℝ), F₁ x' y' → ((x - x')^2 + (y - y')^2 = (R + 1)^2)) →
  (∃ (R : ℝ), ∀ (x' y' : ℝ), F₂ x' y' → ((x - x')^2 + (y - y')^2 = (3 - R)^2)) →
  L x y ∧
  (∀ m, area_ABF₁ m ≤ 3) ∧
  (area_ABF₁ 0 = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_theorem_l781_78180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_special_angle_l781_78145

theorem sin_cos_sum_special_angle (α : Real) :
  (Real.sin (5 * Real.pi / 6), Real.cos (5 * Real.pi / 6)) = (Real.sin α, Real.cos α) →
  Real.sin α + Real.cos α = -Real.sqrt 3 / 2 + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_special_angle_l781_78145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intercept_range_l781_78111

open Set Real

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the y-intercept of the tangent line at point x₀
noncomputable def tangent_intercept (x₀ : ℝ) : ℝ := f x₀ * (1 - x₀)

-- Define the set of all possible y-intercepts
def intercept_set : Set ℝ := {y | ∃ x₀ : ℝ, y = tangent_intercept x₀}

-- Theorem statement
theorem tangent_intercept_range :
  intercept_set = Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intercept_range_l781_78111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_constraint_l781_78129

noncomputable section

/-- A cubic function parameterized by b -/
def f (b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + b * x^2 + (b + 2) * x + 3

/-- The derivative of f with respect to x -/
def f' (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x + (b + 2)

/-- f is monotonically increasing on ℝ -/
def is_monotone_increasing (b : ℝ) : Prop := ∀ x : ℝ, f' b x ≥ 0

theorem monotone_increasing_constraint (b : ℝ) :
  is_monotone_increasing b → -1 ≤ b ∧ b ≤ 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_constraint_l781_78129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_length_locus_theorem_l781_78139

/-- A straight fence represented by a line segment -/
structure Fence where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- A point from which the fence is observed -/
def ObservationPoint : Type := ℝ × ℝ

/-- The angle under which the fence is observed from a point -/
noncomputable def observationAngle (F : Fence) (P : ObservationPoint) : ℝ := sorry

/-- The locus of points from which the fence appears of constant length -/
def constantLengthLocus (F : Fence) (θ : ℝ) : Set ObservationPoint :=
  {P | observationAngle F P = θ}

/-- Two points are symmetric with respect to a line -/
def areSymmetric (P Q : ObservationPoint) (L : Set (ℝ × ℝ)) : Prop := sorry

/-- A set of points forms a circular arc -/
def isCircularArc (S : Set ObservationPoint) : Prop := sorry

theorem constant_length_locus_theorem (F : Fence) (θ : ℝ) :
  ∃ (S₁ S₂ : Set ObservationPoint),
    constantLengthLocus F θ = S₁ ∪ S₂ ∧
    isCircularArc S₁ ∧
    isCircularArc S₂ ∧
    (∀ P Q, P ∈ S₁ → Q ∈ S₂ → areSymmetric P Q {x | x ∈ Set.Icc F.A F.B}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_length_locus_theorem_l781_78139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_24_and_max_monotone_interval_l781_78113

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x) ^ 2

theorem f_value_at_pi_24_and_max_monotone_interval :
  (f (π / 24) = Real.sqrt 2 + 1) ∧
  (∀ m : ℝ, (∀ x y : ℝ, -m ≤ x ∧ x < y ∧ y ≤ m → f x < f y) → m ≤ π / 6) ∧
  (∃ x y : ℝ, -π/6 < x ∧ x < y ∧ y < π/6 ∧ f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_24_and_max_monotone_interval_l781_78113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_ship_speed_is_21_l781_78179

/-- The speed of the faster ship given the conditions of the problem -/
noncomputable def faster_ship_speed (length1 length2 opposite_time same_time : ℝ) : ℝ :=
  let total_length := length1 + length2
  let opposite_speed := total_length / opposite_time
  let same_speed := total_length / same_time
  (opposite_speed + same_speed) / 2

/-- Theorem stating the speed of the faster ship under the given conditions -/
theorem faster_ship_speed_is_21 :
  faster_ship_speed 200 100 10 25 = 21 := by
  -- Unfold the definition of faster_ship_speed
  unfold faster_ship_speed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_ship_speed_is_21_l781_78179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l781_78156

-- Define the necessary structures and functions
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

def CongruentTriangles (t1 t2 : Triangle) : Prop := sorry
def Area (t : Triangle) : ℝ := sorry
def TriangleAnglesEqual (t : Triangle) : Prop := sorry
def ScaleneTriangle (t : Triangle) : Prop := sorry

theorem propositions_truth :
  (∀ x y : ℝ, x = -y → x + y = 0) ∧
  ¬(∀ t1 t2 : Triangle, CongruentTriangles t1 t2 → Area t1 = Area t2) ∧
  (∀ q : ℝ, q ≤ 1 → ∃ x : ℝ, x^2 + 2*x + q = 0) ∧
  ¬(∀ t : Triangle, ¬TriangleAnglesEqual t → ¬ScaleneTriangle t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l781_78156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_equation_l781_78176

/-- The curve C in polar coordinates -/
noncomputable def C (θ : ℝ) : ℝ := 2 * Real.cos θ

theorem polar_to_rectangular_equation :
  ∀ x y θ : ℝ,
  θ ∈ Set.Icc 0 (Real.pi / 2) →
  x = C θ * Real.cos θ →
  y = C θ * Real.sin θ →
  0 ≤ y →
  y ≤ 1 →
  (x - 1)^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_equation_l781_78176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_p_projection_unique_l781_78143

noncomputable def vector_a : ℝ × ℝ := (3, -1)
noncomputable def vector_b : ℝ × ℝ := (-4, 5)
noncomputable def vector_p : ℝ × ℝ := (-15/17, 77/85)

/-- The dot product of two 2D vectors -/
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

/-- The orthogonal projection of vector a onto vector v -/
noncomputable def orthogonal_projection (a v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dot_product a v) / (dot_product v v)
  (scalar * v.1, scalar * v.2)

theorem projection_equals_p :
  ∃ v : ℝ × ℝ, v ≠ (0, 0) ∧
    orthogonal_projection vector_a v = vector_p ∧
    orthogonal_projection vector_b v = vector_p := by
  sorry

theorem projection_unique :
  ∀ q : ℝ × ℝ, (∃ v : ℝ × ℝ, v ≠ (0, 0) ∧
    orthogonal_projection vector_a v = q ∧
    orthogonal_projection vector_b v = q) → q = vector_p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_p_projection_unique_l781_78143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_education_expenses_l781_78110

def monthly_salary (savings : ℚ) (savings_percentage : ℚ) : ℚ :=
  savings / savings_percentage

def total_expenses (rent milk groceries petrol misc : ℚ) : ℚ :=
  rent + milk + groceries + petrol + misc

def education_expenses (salary expenses savings : ℚ) : ℚ :=
  salary - (expenses + savings)

theorem kishore_education_expenses :
  let rent := (5000 : ℚ)
  let milk := (1500 : ℚ)
  let groceries := (4500 : ℚ)
  let petrol := (2000 : ℚ)
  let misc := (700 : ℚ)
  let savings := (1800 : ℚ)
  let savings_percentage := 1/10
  let salary := monthly_salary savings savings_percentage
  let expenses := total_expenses rent milk groceries petrol misc
  education_expenses salary expenses savings = 2500 := by
  sorry

#eval monthly_salary 1800 (1/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_education_expenses_l781_78110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_17_100_l781_78133

/-- Represents a rational number in the interval [0,2) with denominator between 1 and 5 -/
structure RationalInRange where
  num : ℕ
  denom : ℕ
  denom_pos : 0 < denom
  denom_le_five : denom ≤ 5
  frac_lt_two : num < 2 * denom

/-- The set of all possible RationalInRange values -/
def allRationals : Finset RationalInRange := sorry

/-- Checks if (cos(aπ) + i*sin(bπ))^6 is a real number -/
def isReal (a b : RationalInRange) : Prop := sorry

/-- Counts the number of (a,b) pairs where (cos(aπ) + i*sin(bπ))^6 is real -/
def countRealPairs : ℕ := sorry

/-- The main theorem stating the probability -/
theorem probability_is_17_100 : 
  (countRealPairs : ℚ) / (Finset.card allRationals ^ 2 : ℚ) = 17 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_17_100_l781_78133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_black_pairs_custom_deck_l781_78162

/-- A custom deck of cards -/
structure Deck where
  total : ℕ
  black : ℕ
  red : ℕ
  h1 : total = black + red

/-- The expected number of pairs of adjacent black cards in a circular deal -/
def expectedBlackPairs (d : Deck) : ℚ :=
  (d.black : ℚ) * ((d.black - 1) : ℚ) / ((d.total - 1) : ℚ)

/-- Theorem: Expected number of adjacent black pairs in the given deck -/
theorem expected_black_pairs_custom_deck :
  let d : Deck := ⟨104, 60, 44, rfl⟩
  expectedBlackPairs d = 3540 / 103 := by
  sorry

#eval expectedBlackPairs ⟨104, 60, 44, rfl⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_black_pairs_custom_deck_l781_78162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_but_not_opposite_l781_78171

/-- Represents the number of boys selected -/
def num_boys : Fin 4 → Prop := sorry

/-- Represents the number of girls selected -/
def num_girls : Fin 4 → Prop := sorry

/-- Total number of people selected -/
def total_selected : Nat := 3

/-- Total number of boys available -/
def total_boys : Nat := 6

/-- Total number of girls available -/
def total_girls : Nat := 5

/-- Event: At least 2 boys are selected -/
def at_least_two_boys : Prop := ∃ k : Fin 4, k ≥ 2 ∧ num_boys k

/-- Event: At least 2 girls are selected -/
def at_least_two_girls : Prop := ∃ k : Fin 4, k ≥ 2 ∧ num_girls k

/-- Two events are mutually exclusive -/
def mutually_exclusive (e1 e2 : Prop) : Prop :=
  ¬(e1 ∧ e2)

/-- Two events are opposite -/
def opposite (e1 e2 : Prop) : Prop :=
  (e1 ↔ ¬e2) ∧ (e2 ↔ ¬e1)

theorem events_mutually_exclusive_but_not_opposite :
  mutually_exclusive at_least_two_boys at_least_two_girls ∧
  ¬opposite at_least_two_boys at_least_two_girls :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_but_not_opposite_l781_78171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_sum_l781_78194

theorem lcm_ratio_sum (a b : ℕ) : 
  Nat.lcm a b = 60 → a * 3 = b * 2 → a + b = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_sum_l781_78194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_when_e_sqrt3_div2_min_k_for_orthogonal_vectors_l781_78116

/-- An ellipse with semi-major axis a, semi-minor axis b, and right focus at (3,0) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_focus : 3^2 = a^2 - b^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := (3 : ℝ) / e.a

theorem ellipse_equation_when_e_sqrt3_div2 (e : Ellipse) 
  (h_e : eccentricity e = Real.sqrt 3 / 2) :
  ∃ (x y : ℝ), x^2 / 12 + y^2 / 3 = 1 := by sorry

theorem min_k_for_orthogonal_vectors (e : Ellipse)
  (h_e_lower : Real.sqrt 2 / 2 < eccentricity e)
  (h_e_upper : eccentricity e ≤ Real.sqrt 3 / 2) :
  ∃ (k : ℝ), k > 0 ∧ 
    (∀ (k' : ℝ), k' > 0 → 
      (∃ (x₁ y₁ x₂ y₂ : ℝ), 
        x₁^2 / e.a^2 + y₁^2 / e.b^2 = 1 ∧
        x₂^2 / e.a^2 + y₂^2 / e.b^2 = 1 ∧
        y₁ = k' * x₁ ∧ y₂ = k' * x₂ ∧
        (3 - x₁) * (3 - x₂) + y₁ * y₂ = 0) →
      k' ≥ k) ∧
    k = Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_when_e_sqrt3_div2_min_k_for_orthogonal_vectors_l781_78116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l781_78160

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) - Real.cos (ω * x)

def is_symmetry_axis (ω : ℝ) (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi + 3 * Real.pi / 4) / ω

def symmetry_axis_intersection (ω : ℝ) (x : ℝ) : Prop :=
  is_symmetry_axis ω x ∧ f ω x = 0

theorem omega_range (ω : ℝ) :
  (ω > 2/3) →
  (∀ x : ℝ, symmetry_axis_intersection ω x → (x ≤ 2 * Real.pi ∨ x ≥ 3 * Real.pi)) →
  (7/8 ≤ ω ∧ ω ≤ 11/12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l781_78160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_exponential_sum_max_sum_value_max_sum_achievable_l781_78146

theorem max_sum_of_exponential_sum (x y : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^y = 1) : 
  ∀ a b : ℝ, (2 : ℝ)^a + (2 : ℝ)^b = 1 → x + y ≥ a + b :=
by sorry

theorem max_sum_value (x y : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^y = 1) : 
  x + y ≤ -2 :=
by sorry

theorem max_sum_achievable : 
  ∃ x y : ℝ, (2 : ℝ)^x + (2 : ℝ)^y = 1 ∧ x + y = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_exponential_sum_max_sum_value_max_sum_achievable_l781_78146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_wedge_l781_78107

-- Define the circumference of the hemisphere's base
noncomputable def base_circumference : ℝ := 18 * Real.pi

-- Define the number of wedges
def num_wedges : ℕ := 6

-- Theorem statement
theorem volume_of_wedge :
  let radius : ℝ := base_circumference / (2 * Real.pi)
  let hemisphere_volume : ℝ := (2/3) * Real.pi * (radius ^ 3)
  let wedge_volume : ℝ := hemisphere_volume / (num_wedges : ℝ)
  wedge_volume = 81 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_wedge_l781_78107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_ellipse_to_line_l781_78121

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := 7 * x^2 + 4 * y^2 = 28

/-- The line equation -/
def line (x y : ℝ) : Prop := 3 * x - 2 * y - 16 = 0

/-- Distance function from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (3 * x - 2 * y - 16) / Real.sqrt 13

/-- The theorem stating that (3/2, -7/4) is the closest point -/
theorem closest_point_on_ellipse_to_line :
  let x₀ : ℝ := 3/2
  let y₀ : ℝ := -7/4
  ellipse x₀ y₀ ∧
  ∀ x y : ℝ, ellipse x y → distance_to_line x₀ y₀ ≤ distance_to_line x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_ellipse_to_line_l781_78121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l781_78136

/-- Curve C with parametric equations x = 2 - t - t² and y = 2 - 3t + t² -/
def C (t : ℝ) : ℝ × ℝ := (2 - t - t^2, 2 - 3*t + t^2)

/-- Point A where curve C intersects the y-axis -/
def A : ℝ × ℝ := C (-2)

/-- Point B where curve C intersects the x-axis -/
def B : ℝ × ℝ := C 2

/-- The distance between points A and B -/
noncomputable def AB_distance : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The polar coordinate equation of line AB -/
def AB_polar_equation (ρ θ : ℝ) : Prop :=
  3 * ρ * Real.cos θ - ρ * Real.sin θ + 12 = 0

theorem curve_C_properties :
  AB_distance = 4 * Real.sqrt 10 ∧
  ∀ ρ θ, AB_polar_equation ρ θ ↔ 3 * ρ * Real.cos θ - ρ * Real.sin θ + 12 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l781_78136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_positive_l781_78185

noncomputable def f (x : ℝ) := x - Real.sin x

theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁₂ : x₁ + x₂ > 0) (h₂₃ : x₂ + x₃ > 0) (h₁₃ : x₁ + x₃ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_positive_l781_78185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_200_l781_78183

theorem closest_to_200 :
  let result := (2.46 : ℝ) * 8.163 * (5.17 + 4.829)
  ∀ x ∈ ({100, 300, 400, 500} : Set ℝ), |result - 200| < |result - x| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_200_l781_78183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_concentration_calculation_l781_78158

-- Define the initial volume and salt concentration
noncomputable def initial_volume : ℝ := 120
noncomputable def initial_salt_concentration : ℝ := 0.20

-- Define the evaporation rate and additions
noncomputable def evaporation_rate : ℝ := 0.25
noncomputable def water_added : ℝ := 8
noncomputable def salt_added : ℝ := 16

-- Calculate the final salt concentration
noncomputable def final_salt_concentration : ℝ :=
  let initial_salt := initial_volume * initial_salt_concentration
  let volume_after_evaporation := initial_volume * (1 - evaporation_rate)
  let final_volume := volume_after_evaporation + water_added
  let final_salt := initial_salt + salt_added
  (final_salt / final_volume) * 100

-- Theorem statement
theorem salt_concentration_calculation :
  |final_salt_concentration - 40.82| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_concentration_calculation_l781_78158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_l781_78191

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (Real.log x^2 / Real.log a - 1)

-- Define the domain for a > 1
def domain_gt_1 (a : ℝ) : Set ℝ :=
  {x | x ≤ -Real.sqrt a ∨ x ≥ Real.sqrt a}

-- Define the domain for 0 < a < 1
def domain_lt_1 (a : ℝ) : Set ℝ :=
  {x | (x ≥ -Real.sqrt a ∧ x < 0) ∨ (x > 0 ∧ x ≤ Real.sqrt a)}

-- State the theorem
theorem function_domain (a : ℝ) (h : a > 0) :
  (a > 1 → {x : ℝ | f a x = f a x} = domain_gt_1 a) ∧
  (a < 1 → {x : ℝ | f a x = f a x} = domain_lt_1 a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_l781_78191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_reach_all_iff_l781_78117

/-- A knight's move on an infinite chessboard -/
structure KnightMove (p q : ℕ+) where
  horizontal : ℤ
  vertical : ℤ
  is_valid : (horizontal.natAbs = p ∧ vertical.natAbs = q) ∨
             (horizontal.natAbs = q ∧ vertical.natAbs = p)

/-- The property of a knight being able to reach any square from any other square -/
def can_reach_all (p q : ℕ+) : Prop :=
  ∀ (start finish : ℤ × ℤ), ∃ (n : ℕ) (moves : Fin n → KnightMove p q),
    start = finish ∨
    (∃ i : Fin n, 
      (moves i).horizontal + (moves i).vertical = finish.1 - start.1 + finish.2 - start.2) ∧
    (∀ i j : Fin n, i.val + 1 = j.val →
      (moves j).horizontal + (moves j).vertical = 
      (moves i).horizontal + (moves i).vertical)

/-- The main theorem stating the necessary and sufficient conditions for a knight to reach all squares -/
theorem knight_reach_all_iff (p q : ℕ+) :
  can_reach_all p q ↔ Nat.gcd p.val q.val = 1 ∧ (p.val + q.val) % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_reach_all_iff_l781_78117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_special_values_l781_78131

theorem ordering_of_special_values :
  Real.cos (2 * Real.pi / 3) < Real.log 2 / Real.log 3 ∧ Real.log 2 / Real.log 3 < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_special_values_l781_78131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_formula_T_formula_l781_78154

noncomputable def S (n : ℕ) : ℝ := 2 - 1 / (2^(n-1))

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => S (n + 1) - S n

noncomputable def b : ℕ → ℝ
  | 0 => 1
  | n + 1 => b n + 2

noncomputable def c (n : ℕ) : ℝ := b n / a n

noncomputable def T : ℕ → ℝ
  | 0 => c 0
  | n + 1 => T n + c (n + 1)

theorem a_formula (n : ℕ) : a n = 1 / (2^n) := by sorry

theorem b_formula (n : ℕ) : b n = 2 * n + 1 := by sorry

theorem T_formula (n : ℕ) : T n = (2 * n + 1 - 3) * 2^(n + 1) + 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_formula_T_formula_l781_78154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l781_78125

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define properties of the triangle
noncomputable def Triangle.sideLength (t : Triangle) : ℝ × ℝ × ℝ := sorry

noncomputable def Triangle.centroid (t : Triangle) : ℝ × ℝ := sorry

noncomputable def Triangle.circumradius (t : Triangle) : ℝ := sorry

-- Define a point on the circumcircle
noncomputable def pointOnCircumcircle (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem constant_sum_of_squares (t : Triangle) :
  let G := t.centroid
  let P := pointOnCircumcircle t
  let (a, b, c) := t.sideLength
  let R := t.circumradius
  distance P t.A ^ 2 + distance P t.B ^ 2 + distance P t.C ^ 2 - distance P G ^ 2 =
    a ^ 2 + b ^ 2 + c ^ 2 - 2 * R ^ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l781_78125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karen_initial_water_weight_l781_78102

/-- The amount of water Karen drinks per hour -/
noncomputable def water_consumption_rate : ℝ := 2

/-- The ratio of food eaten to water drunk per hour -/
noncomputable def food_consumption_ratio : ℝ := 1/3

/-- The duration of Karen's hike in hours -/
noncomputable def hike_duration : ℝ := 6

/-- The initial amount of food Karen packs -/
noncomputable def initial_food : ℝ := 10

/-- The amount of gear Karen packs -/
noncomputable def gear_weight : ℝ := 20

/-- The total weight Karen is carrying after the hike -/
noncomputable def final_weight : ℝ := 34

/-- Theorem stating that Karen initially packed 20 pounds of water -/
theorem karen_initial_water_weight :
  ∃ (initial_water : ℝ),
    initial_water +
    initial_food +
    gear_weight -
    (water_consumption_rate * hike_duration) -
    (food_consumption_ratio * water_consumption_rate * hike_duration) =
    final_weight ∧
    initial_water = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_karen_initial_water_weight_l781_78102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_asymptote_l781_78165

/-- Given a hyperbola and a parabola intersecting at point P, 
    where F is the focus of the parabola, prove that the 
    equation of the asymptote of the hyperbola is √3x ± y = 0 --/
theorem hyperbola_parabola_intersection_asymptote 
  (m : ℝ) 
  (P : ℝ × ℝ) 
  (F : ℝ × ℝ) :
  (P.1^2 - P.2^2 / m = 1) →  -- P is on the hyperbola
  (P.2^2 = 8 * P.1) →        -- P is on the parabola
  (Real.sqrt ((F.1 - P.1)^2 + (F.2 - P.2)^2) = 5) →  -- |PF| = 5
  (F.1 = 2) →                -- F is the focus of the parabola y^2 = 8x
  (∃ (k : ℝ), k * Real.sqrt 3 * P.1 + P.2 = 0 ∨ k * Real.sqrt 3 * P.1 - P.2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_asymptote_l781_78165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_speed_calculation_l781_78192

/-- The speed of a beetle in km/h, given an ant's walking distance and the beetle's relative speed -/
noncomputable def beetle_speed (ant_distance : ℝ) (ant_time : ℝ) (beetle_relative_speed : ℝ) : ℝ :=
  let beetle_distance := ant_distance * (1 - beetle_relative_speed)
  let beetle_speed_mpm := beetle_distance / ant_time
  (beetle_speed_mpm * 60) / 1000

/-- Theorem stating that under given conditions, the beetle's speed is 0.425 km/h -/
theorem beetle_speed_calculation :
  beetle_speed 500 60 0.15 = 0.425 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_speed_calculation_l781_78192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_cube_root_nine_equals_cube_root_three_l781_78124

theorem sqrt_cube_root_nine_equals_cube_root_three :
  Real.sqrt (9 ^ (1/3)) = 3 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_cube_root_nine_equals_cube_root_three_l781_78124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_roots_l781_78188

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem no_negative_roots (a : ℝ) (h : a > 1) :
  ∀ x : ℝ, x < 0 → f a x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_roots_l781_78188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l781_78147

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

def solution_set : Set ℝ :=
  {x | x = 2 ∨ (3 < x ∧ x < 4)}

theorem equation_solution :
  ∀ x : ℝ, x > 0 → (floor (1/x * (floor x)^2 : ℝ) = 2 ↔ x ∈ solution_set) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l781_78147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_sum_l781_78118

/-- Given two vectors a and b in ℝ³, proves that if they are collinear
    and have specific components, then the sum of their variable components is 6. -/
theorem collinear_vectors_sum (m n : ℝ) :
  let a : ℝ × ℝ × ℝ := (2, 3, m)
  let b : ℝ × ℝ × ℝ := (2*n, 6, 8)
  (∃ (k : ℝ), a = k • b) →
  m + n = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_sum_l781_78118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l781_78174

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) + Real.log (x + 1) / Real.log 10

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l781_78174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_117_l781_78197

theorem divisibility_by_117 (n : ℕ) :
  ∃ k : ℤ, 3^(2*(n+1)) * 5^(2*n) - 3^(3*n+2) * 2^(2*n) = 117 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_117_l781_78197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_validPathCount_l781_78148

/-- Represents a point on the 2D plane -/
structure Point where
  x : Int
  y : Int

/-- Represents the four possible directions -/
inductive Direction
  | Right
  | Up
  | Left
  | Down

/-- Represents a path on the 2D plane -/
structure PathStruct where
  points : List Point
  directions : List Direction

/-- Checks if a point is within the allowed square -/
def isWithinBounds (p : Point) : Bool :=
  p.x.natAbs ≤ 6 ∧ p.y.natAbs ≤ 6

/-- Checks if a path is valid according to the given rules -/
def isValidPath (path : PathStruct) : Bool :=
  sorry

/-- Counts the number of valid paths from (0,0) to (6,6) -/
noncomputable def countValidPaths : Nat :=
  sorry

/-- The main theorem stating the number of valid paths -/
theorem validPathCount : countValidPaths = 131922 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_validPathCount_l781_78148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_value_l781_78166

/-- The first sequence in the nested radicals -/
def a (n : ℕ) : ℕ := 8 + 3 * n

/-- The second sequence in the nested radicals -/
def b (n : ℕ) : ℕ := n^2 + 3 * n

/-- The nested radical expression -/
noncomputable def nestedRadical : ℝ := (11 + 4 * ((14 + 10 * ((17 + 18 * Real.pi) ^ (1/3))) ^ (1/3))) ^ (1/3)

theorem nested_radical_value : nestedRadical = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_value_l781_78166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_coin_tails_up_l781_78138

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails
deriving Repr, DecidableEq

/-- Represents the circle of coins -/
def CoinCircle (n : ℕ) := Fin (2*n+1) → CoinState

/-- The initial state of the coin circle, all heads up -/
def initialState (n : ℕ) : CoinCircle n :=
  λ _ => CoinState.Heads

/-- The position of the coin to be flipped at step k -/
def flipPosition (n k : ℕ) : Fin (2*n+1) :=
  ⟨k * (k + 1) / 2 % (2*n+1), by sorry⟩

/-- Flips the state of a coin -/
def flipCoin : CoinState → CoinState
| CoinState.Heads => CoinState.Tails
| CoinState.Tails => CoinState.Heads

/-- Performs a single flip operation on the coin circle -/
def flipStep (n : ℕ) (k : Fin (2*n+1)) (circle : CoinCircle n) : CoinCircle n :=
  λ i => if i = flipPosition n k then flipCoin (circle i) else circle i

/-- Performs the entire flipping process -/
def flipProcess (n : ℕ) : CoinCircle n → CoinCircle n :=
  λ circle => (List.range (2*n+1)).foldl (λ acc i => flipStep n ⟨i, by sorry⟩ acc) circle

/-- Counts the number of tails-up coins in the circle -/
def countTails (n : ℕ) (circle : CoinCircle n) : ℕ :=
  (List.range (2*n+1)).foldl (λ count i => if circle ⟨i, by sorry⟩ = CoinState.Tails then count + 1 else count) 0

/-- The main theorem: after the flipping process, exactly one coin is tails up -/
theorem one_coin_tails_up (n : ℕ) : countTails n (flipProcess n (initialState n)) = 1 := by
  sorry  -- Proof omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_coin_tails_up_l781_78138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_specific_l781_78134

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : Real × Real × Real :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_specific :
  spherical_to_rectangular 10 (3 * π / 4) (π / 6) =
  (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_specific_l781_78134
