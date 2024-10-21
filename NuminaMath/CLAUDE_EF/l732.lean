import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_percentage_l732_73205

/-- Proves that a price reduction resulting in a 50% sales increase and a 5% total receipts increase must be a 30% price reduction. -/
theorem price_reduction_percentage (P S : ℝ) (hP : P > 0) (hS : S > 0) : 
  let new_sales := 1.5 * S
  let new_receipts := 1.05 * (P * S)
  let price_reduction_factor := new_receipts / (new_sales * P)
  price_reduction_factor = 0.7
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_percentage_l732_73205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_new_average_l732_73248

/-- Represents a batsman's performance statistics -/
structure BatsmanStats where
  initialAverage : ℚ
  initialInnings : ℕ
  improvementAverage : ℚ
  additionalInnings : ℕ

/-- Calculates the new average for a batsman after improvement -/
def newAverage (stats : BatsmanStats) : ℚ :=
  let totalInitialRuns := stats.initialAverage * stats.initialInnings
  let totalImprovedRuns := (stats.initialAverage + stats.improvementAverage) * stats.additionalInnings
  let totalInnings := stats.initialInnings + stats.additionalInnings
  (totalInitialRuns + totalImprovedRuns) / totalInnings

/-- Theorem stating the new average of the batsman after improvement -/
theorem batsman_new_average (stats : BatsmanStats) 
  (h1 : stats.initialAverage = 45)
  (h2 : stats.initialInnings = 40)
  (h3 : stats.improvementAverage = 5)
  (h4 : stats.additionalInnings = 20) :
  newAverage stats = 2800 / 60 := by
  sorry

#eval newAverage { initialAverage := 45, initialInnings := 40, improvementAverage := 5, additionalInnings := 20 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_new_average_l732_73248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_negative_one_l732_73286

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x - 2 else 2^x

-- Theorem statement
theorem f_at_negative_one : f (-1) = (1/2 : ℝ) := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expression
  simp [if_neg (show ¬(-1 ≥ 0) by norm_num)]
  -- Evaluate 2^(-1)
  norm_num

-- The proof is complete, so we don't need 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_negative_one_l732_73286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_team_size_proof_l732_73250

/-- The number of people in the first team -/
def first_team_size : ℕ := 4

/-- The time it takes the first team to complete the job alone (in hours) -/
noncomputable def first_team_time : ℝ := 8

/-- The time it takes both teams to complete the job together (in hours) -/
noncomputable def combined_time : ℝ := 3

/-- The total worker-hours required to complete the job -/
noncomputable def total_worker_hours : ℝ := (first_team_size : ℝ) * first_team_time

/-- The number of people in the second team -/
noncomputable def second_team_size : ℝ := (total_worker_hours / combined_time) - (first_team_size : ℝ)

theorem second_team_size_proof :
  second_team_size = 20 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_team_size_proof_l732_73250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_divisor_digit_sum_l732_73200

def n : ℕ := 8191

theorem greatest_prime_divisor_digit_sum (h : n = 2^13 - 1) :
  (n.factors.maximum?.map (λ p => (p.repr.toList.map (λ c => c.toNat - 48)).sum)).get! = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_divisor_digit_sum_l732_73200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_points_l732_73284

-- Define the conjugate point transformation
def conjugate (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y + 1, x + 1)

-- Define the sequence of points
def A : ℕ → ℝ × ℝ
  | 0 => (3, 1)  -- Define for 0 to handle all natural numbers
  | n + 1 => conjugate (A n)

-- Theorem statement
theorem conjugate_points :
  A 2 = (0, 4) ∧ A 2023 = (-3, 1) := by
  sorry

-- Helper lemmas to prove the cyclic nature
lemma A_cycle (n : ℕ) : A (n + 4) = A n := by
  sorry

lemma A_values :
  A 1 = (3, 1) ∧ A 2 = (0, 4) ∧ A 3 = (-3, 1) ∧ A 4 = (0, -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_points_l732_73284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_difference_l732_73258

/-- Calculates simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating the principal amount given the interest difference -/
theorem principal_from_interest_difference 
  (rate : ℝ) (time : ℝ) (difference : ℝ) (principal : ℝ) :
  rate = 10 ∧ time = 2 ∧ difference = 15 ∧
  compoundInterest principal rate time - simpleInterest principal rate time = difference →
  principal = 1500 := by
  sorry

#check principal_from_interest_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_difference_l732_73258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l732_73242

noncomputable section

/-- The function f(x) = x - 2/x -/
def f (x : ℝ) : ℝ := x - 2/x

/-- The function g(x) = a*cos(πx/2) + 11 - 2a -/
def g (a x : ℝ) : ℝ := a * Real.cos (Real.pi * x / 2) + 11 - 2*a

theorem range_of_a :
  ∀ a : ℝ,
  (a ≠ 0) →
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 0 1, g a x₂ = f x₁) →
  a ∈ Set.Icc 6 10 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l732_73242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_competition_wins_l732_73259

theorem soccer_competition_wins :
  ∃ (wins losses : ℕ),
    wins + losses = 12 ∧
    3 * wins + losses = 28 ∧
    wins = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_competition_wins_l732_73259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cevas_theorem_triangle_concurrency_l732_73290

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line segment
def LineSegment (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t, 1 - t) • p + (t, t) • q}

-- Define a median
def Median (t : Triangle) (v : ℝ × ℝ) : Prop := 
  ∃ m : ℝ × ℝ, m ∈ LineSegment t.B t.C ∧ v ∈ LineSegment t.A m

-- Define an altitude
def Altitude (t : Triangle) (v : ℝ × ℝ) : Prop := 
  ∃ h : ℝ × ℝ, h ∈ LineSegment t.B t.C ∧ v ∈ LineSegment t.A h ∧ 
    (h.1 - t.A.1) * (t.C.1 - t.B.1) + (h.2 - t.A.2) * (t.C.2 - t.B.2) = 0

-- Define an angle bisector
def AngleBisector (t : Triangle) (v : ℝ × ℝ) : Prop := 
  ∃ i : ℝ × ℝ, i ∈ LineSegment t.B t.C ∧ v ∈ LineSegment t.A i ∧
    (i.1 - t.A.1) * (t.B.1 - t.A.1) + (i.2 - t.A.2) * (t.B.2 - t.A.2) =
    (i.1 - t.A.1) * (t.C.1 - t.A.1) + (i.2 - t.A.2) * (t.C.2 - t.A.2)

-- State Ceva's theorem
theorem cevas_theorem (t : Triangle) (D E F : ℝ × ℝ) :
  D ∈ LineSegment t.B t.C ∧ E ∈ LineSegment t.C t.A ∧ F ∈ LineSegment t.A t.B →
  (∃ G : ℝ × ℝ, G ∈ LineSegment t.A D ∧ G ∈ LineSegment t.B E ∧ G ∈ LineSegment t.C F) ↔
  (t.B.1 - D.1) * (t.C.1 - E.1) * (t.A.1 - F.1) = 
  (D.1 - t.C.1) * (E.1 - t.A.1) * (F.1 - t.B.1) :=
by sorry

-- Theorem to prove
theorem triangle_concurrency (t : Triangle) :
  (∃ G : ℝ × ℝ, ∀ v : ℝ × ℝ, Median t v → v ∈ LineSegment t.A G ∨ v ∈ LineSegment t.B G ∨ v ∈ LineSegment t.C G) ∧
  (∃ H : ℝ × ℝ, ∀ v : ℝ × ℝ, Altitude t v → v ∈ LineSegment t.A H ∨ v ∈ LineSegment t.B H ∨ v ∈ LineSegment t.C H) ∧
  (∃ I : ℝ × ℝ, ∀ v : ℝ × ℝ, AngleBisector t v → v ∈ LineSegment t.A I ∨ v ∈ LineSegment t.B I ∨ v ∈ LineSegment t.C I) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cevas_theorem_triangle_concurrency_l732_73290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_worked_67_hours_l732_73252

/-- Represents the compensation structure and work details of a bus driver -/
structure BusDriverCompensation where
  regularRate : ℚ
  passengerBonus : ℚ
  regularHours : ℚ
  firstOvertimeRate : ℚ
  firstOvertimeHours : ℚ
  secondOvertimeRate : ℚ
  totalCompensation : ℚ
  totalPassengers : ℕ

/-- Calculates the total hours worked by the bus driver -/
noncomputable def totalHoursWorked (bdc : BusDriverCompensation) : ℚ :=
  bdc.regularHours + bdc.firstOvertimeHours + 
  (bdc.totalCompensation - bdc.regularRate * bdc.regularHours - 
   bdc.firstOvertimeRate * bdc.firstOvertimeHours - 
   bdc.passengerBonus * bdc.totalPassengers) / bdc.secondOvertimeRate

/-- Theorem stating that given the conditions, the bus driver worked 67 hours -/
theorem bus_driver_worked_67_hours : 
  ∀ (bdc : BusDriverCompensation), 
    bdc.regularRate = 12 ∧ 
    bdc.passengerBonus = 1/2 ∧
    bdc.regularHours = 40 ∧
    bdc.firstOvertimeRate = 18 ∧
    bdc.firstOvertimeHours = 5 ∧
    bdc.secondOvertimeRate = 24 ∧
    bdc.totalCompensation = 1280 ∧
    bdc.totalPassengers = 350 →
    totalHoursWorked bdc = 67 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_worked_67_hours_l732_73252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l732_73228

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then -x + 7 else 3 + (Real.log x) / (Real.log a)

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ y : ℝ, y ≥ 5 → ∃ x : ℝ, f a x = y) ∧
  (∀ x : ℝ, f a x ≥ 5) ↔
  1 < a ∧ a ≤ Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l732_73228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_composition_l732_73261

noncomputable def g (x : ℝ) : ℝ := 5 * x - 3

noncomputable def g_inv (x : ℝ) : ℝ := (x + 3) / 5

theorem g_inverse_composition : g_inv (g_inv 11) = 29 / 25 := by
  -- Unfold the definition of g_inv
  unfold g_inv
  -- Simplify the expression
  simp [add_div, div_div]
  -- Perform numerical calculations
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_composition_l732_73261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_for_given_volume_l732_73233

/-- A cone with an isosceles right triangle as its axis section -/
structure IsoscelesRightCone where
  radius : ℝ
  height : ℝ
  isIsoscelesRight : height = radius

/-- The volume of a cone -/
noncomputable def volume (c : IsoscelesRightCone) : ℝ :=
  (1/3) * Real.pi * c.radius^2 * c.height

/-- The lateral area of a cone -/
noncomputable def lateralArea (c : IsoscelesRightCone) : ℝ :=
  Real.pi * c.radius * Real.sqrt (c.radius^2 + c.height^2)

theorem cone_lateral_area_for_given_volume :
  ∀ (c : IsoscelesRightCone), volume c = (64*Real.pi)/3 → lateralArea c = 16*Real.sqrt 2*Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_for_given_volume_l732_73233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_105_selection_l732_73213

theorem product_105_selection (n : ℕ) :
  n ≥ 7 →
  ∀ (S : Finset ℕ),
  (∀ m, m ∈ S → m ∈ Finset.range 100) →
  S.card = n →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a * b = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_105_selection_l732_73213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_insphere_radius_l732_73241

/-- The radius of the insphere of a triangular pyramid -/
noncomputable def insphere_radius (V S : ℝ) : ℝ := 3 * V / S

/-- Theorem: The radius of the insphere of a triangular pyramid is 3V/S,
    where V is the volume and S is the surface area of the pyramid -/
theorem triangular_pyramid_insphere_radius (V S : ℝ) (h1 : V > 0) (h2 : S > 0) :
  ∃ (R : ℝ), R = insphere_radius V S ∧ R > 0 := by
  use insphere_radius V S
  constructor
  · rfl
  · apply div_pos
    · exact mul_pos (by norm_num) h1
    · exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_insphere_radius_l732_73241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l732_73201

def P : Set ℝ := {x | x^2 - 9 < 0}

def Q : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n ∧ -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_P_and_Q : P ∩ Q = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l732_73201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_equal_sides_necessary_not_sufficient_l732_73273

-- Define the properties of quadrilaterals
structure Quadrilateral where
  -- We'll leave the internal structure abstract for now
  mk :: -- Constructor

-- Define the properties of quadrilaterals
def has_parallel_equal_sides (q : Quadrilateral) : Prop :=
  -- We'll leave this as an axiom for now
  sorry

def is_rectangle (q : Quadrilateral) : Prop :=
  -- We'll leave this as an axiom for now
  sorry

theorem parallel_equal_sides_necessary_not_sufficient :
  (∀ q : Quadrilateral, is_rectangle q → has_parallel_equal_sides q) ∧
  (∃ q : Quadrilateral, has_parallel_equal_sides q ∧ ¬is_rectangle q) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_equal_sides_necessary_not_sufficient_l732_73273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_eq_half_l732_73219

theorem sin_2theta_eq_half (θ : ℝ) (h : Real.tan θ + (Real.tan θ)⁻¹ = 4) : 
  Real.sin (2 * θ) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_eq_half_l732_73219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_unit_vectors_l732_73206

/-- Given vectors a and b in a real inner product space satisfying 
    |a| = |b| = |a + b| = 1, the angle between a and b is 2π/3. -/
theorem angle_between_unit_vectors (V : Type*) 
  [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hab : ‖a + b‖ = 1) :
  Real.arccos (inner a b) = (2 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_unit_vectors_l732_73206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_of_roots_l732_73277

theorem tan_half_sum_of_roots (a : ℝ) (α β : ℝ) : 
  a > 1 →
  (∀ x, x^2 + 4*a*x + 3*a + 1 = 0 ↔ x = Real.tan α ∨ x = Real.tan β) →
  α ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) →
  β ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) →
  Real.tan ((α + β) / 2) = -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_of_roots_l732_73277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_foci_coincide_l732_73289

/-- Given a parabola and a hyperbola with coinciding foci, prove that the parameter of the parabola is 2√14 -/
theorem parabola_hyperbola_foci_coincide (p : ℝ) : 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2/9 - y^2/5 = 1 ∧ x = Real.sqrt 14 ∧ y = 0) →
  p = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_foci_coincide_l732_73289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l732_73223

theorem recurring_decimal_to_fraction : (7/10 : ℚ) + (32/99 : ℚ) = 1013/990 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l732_73223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_millet_majority_on_sunday_l732_73256

/-- Amount of millet in the feeder on day n -/
noncomputable def milletAmount (n : ℕ) : ℝ := (4/3) * (1 - (0.7 ^ n))

/-- Total amount of seeds in the feeder on day n -/
def totalSeeds : ℕ → ℝ := λ _ => 2

theorem millet_majority_on_sunday :
  (∀ k < 5, milletAmount k ≤ totalSeeds k / 2) ∧
  milletAmount 5 > totalSeeds 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_millet_majority_on_sunday_l732_73256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l732_73246

theorem trigonometric_problem (x : ℝ) 
  (h1 : Real.sin x = Real.sqrt 5 / 5) 
  (h2 : 0 < x) (h3 : x < Real.pi / 2) : 
  (Real.cos x = 2 * Real.sqrt 5 / 5) ∧ 
  (Real.tan x = 1 / 2) ∧ 
  ((Real.cos x + 2 * Real.sin x) / (2 * Real.cos x - Real.sin x) = 4 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l732_73246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l732_73299

-- Define the inequality
def inequality (a b x : ℝ) : Prop := x^2 - a*x - x + b < 0

-- Part I
theorem part_one (a b : ℝ) : 
  (∀ x, inequality a b x ↔ -1 < x ∧ x < 1) → a = -1 ∧ b = -1 :=
by sorry

-- Part II
def solution_set (a : ℝ) : Set ℝ :=
  if a = 1 then ∅ 
  else if a < 1 then Set.Ioo a 1
  else Set.Ioo 1 a

theorem part_two (a : ℝ) :
  (∀ x, inequality a a x ↔ x ∈ solution_set a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l732_73299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_construction_l732_73239

/-- Represents a Π-shaped bracket -/
structure PiBracket :=
  (segments : Fin 3 → Unit)

/-- Represents the cube frame -/
structure CubeFrame :=
  (segments : Fin 54 → Unit)
  (vertices : Fin 8 → Unit)
  (center : Unit)

/-- Represents an attempt to construct the cube frame using brackets -/
def FrameConstruction := CubeFrame → List PiBracket

/-- The theorem stating that it's impossible to construct the frame with exactly 18 brackets -/
theorem impossible_construction (frame : CubeFrame) :
  ¬ ∃ (construction : FrameConstruction), (construction frame).length = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_construction_l732_73239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_equals_incenter_distance_l732_73283

-- Define the triangle and points
variable (A B C D O : EuclideanSpace ℝ (Fin 2))
variable (n : ℝ)

-- Define the circumcircle and incircle
variable (circumcircle incircle : Sphere (EuclideanSpace ℝ (Fin 2)) ℝ)

-- State the conditions
variable (h1 : IsAngleBisector A B C D)
variable (h2 : D ∈ circumcircle.sphere)
variable (h3 : O = incircle.center)
variable (h4 : dist O D = n)

-- State the theorem
theorem chord_length_equals_incenter_distance :
  dist C D = n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_equals_incenter_distance_l732_73283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berkeley_class_a_students_l732_73271

theorem berkeley_class_a_students
  (abraham_total : ℕ)
  (abraham_a : ℕ)
  (berkeley_total : ℕ)
  (berkeley_a : ℕ)
  (h1 : abraham_total = 15)
  (h2 : abraham_a = 10)
  (h3 : berkeley_total = 24)
  (h4 : abraham_a * berkeley_total = abraham_total * berkeley_a) :
  berkeley_a = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_berkeley_class_a_students_l732_73271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_midsection_area_l732_73234

/-- Given a frustum with top radius 2 and bottom radius 3, 
    the area of the circle formed by its midsection is 25π/4 -/
theorem frustum_midsection_area : 
  ∀ (top_radius bottom_radius : ℝ),
    top_radius = 2 → 
    bottom_radius = 3 → 
    let midsection_radius := (top_radius + bottom_radius) / 2
    π * midsection_radius^2 = 25 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_midsection_area_l732_73234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_transformation_graph_transformation_l732_73253

open Real

theorem sin_plus_cos_transformation (x : ℝ) : 
  sin x + cos x = sqrt 2 * cos (x - π/4) :=
by
  sorry

theorem graph_transformation (x : ℝ) :
  sqrt 2 * cos (2*x) = sqrt 2 * cos (x - π/4) ↔
  ∃ (y : ℝ), y = x - π/8 ∧ sqrt 2 * cos (2*y) = sqrt 2 * cos (x - π/4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_transformation_graph_transformation_l732_73253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l732_73292

-- Define the profit for each item
def jerseyProfit : ℚ := 5
def tshirtProfit : ℚ := 15
def hatProfit : ℚ := 8
def hoodieProfit : ℚ := 25

-- Define the number of items sold
def jerseysSold : ℚ := 64
def tshirtsSold : ℚ := 20
def hatsSold : ℚ := 30
def hoodiesSold : ℚ := 10

-- Define the discount rate and vendor fee
def discountRate : ℚ := 1/10
def vendorFee : ℚ := 50

-- Theorem statement
theorem total_profit_calculation :
  let totalProfitBeforeDiscount := jerseyProfit * jerseysSold + tshirtProfit * tshirtsSold + 
                                   hatProfit * hatsSold + hoodieProfit * hoodiesSold
  let discountAmount := discountRate * totalProfitBeforeDiscount
  let profitAfterDiscount := totalProfitBeforeDiscount - discountAmount
  profitAfterDiscount - vendorFee = 949 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l732_73292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_g_max_min_on_interval_l732_73247

noncomputable def f (x : ℝ) : ℝ := (1/2) * x + Real.sin x

theorem f_max_min_on_interval :
  ∃ (max min : ℝ), max = π ∧ min = 0 ∧
  (∀ x ∈ Set.Icc 0 (2 * π), f x ≤ max ∧ f x ≥ min) ∧
  (∃ x₁ ∈ Set.Icc 0 (2 * π), f x₁ = max) ∧
  (∃ x₂ ∈ Set.Icc 0 (2 * π), f x₂ = min) :=
by
  sorry

def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x - 2

theorem g_max_min_on_interval :
  ∃ (max min : ℝ), max = 2 ∧ min = -12 ∧
  (∀ x ∈ Set.Icc (-1) 1, g x ≤ max ∧ g x ≥ min) ∧
  (∃ x₁ ∈ Set.Icc (-1) 1, g x₁ = max) ∧
  (∃ x₂ ∈ Set.Icc (-1) 1, g x₂ = min) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_g_max_min_on_interval_l732_73247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_11_value_l732_73231

def sequence_a : ℕ → ℤ
| 0 => 1
| n + 1 => 2 * sequence_a n + 3

theorem a_11_value : sequence_a 10 = 2^11 - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_11_value_l732_73231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l732_73202

open Polynomial

theorem polynomial_remainder : ∃ q : Polynomial ℤ, 
  (5 * X^8 - 2 * X^7 - 8 * X^6 + 3 * X^4 + 5 * X^3 - 13 : Polynomial ℤ) = 
  (3 * X - 9) * q + 23364 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l732_73202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_side_length_l732_73297

/-- The side length of a cube inscribed between a sphere and one face of a larger cube (with side length 2) inscribed in the same sphere is 2/3. -/
theorem inscribed_cube_side_length :
  ∀ (R : ℝ) (s : ℝ),
  R = Real.sqrt 3 →  -- Radius of the sphere
  3 * s^2 + 4 * s - 4 = 0 →  -- Equation derived from geometric constraints
  s > 0 →  -- Side length must be positive
  s = 2/3 := by
  intros R s hR hEq hPos
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_side_length_l732_73297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_distance_l732_73278

/-- An ellipse with eccentricity √6/3 and minor axis length 2 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : b = 1
  h4 : (6 : ℝ) / 9 = 1 - (b / a)^2

/-- A line that intersects the ellipse at two distinct points -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h5 : ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + m ∧ y₂ = k * x₂ + m ∧
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1

/-- The theorem to be proved -/
theorem ellipse_intersection_distance (E : Ellipse) (L : IntersectingLine E) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    y₁ = L.k * x₁ + L.m ∧ y₂ = L.k * x₂ + L.m ∧
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1 ∧
    x₁ * x₂ + y₁ * y₂ = 0) →
  |L.m| / Real.sqrt (1 + L.k^2) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_distance_l732_73278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l732_73280

-- Define the power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_decreasing :
  ∃ α : ℝ, power_function α 2 = Real.sqrt 2 / 2 ∧
  ∀ x y, x < y → x > 0 → y > 0 → power_function α x > power_function α y :=
by
  -- Introduce α = -1/2
  let α := -1/2
  
  -- Show that this α satisfies the conditions
  have h1 : power_function α 2 = Real.sqrt 2 / 2 := by
    -- This step would require some calculation, which we'll skip for now
    sorry
  
  -- Show that the function is decreasing for this α
  have h2 : ∀ x y, x < y → x > 0 → y > 0 → power_function α x > power_function α y := by
    -- This step would require proving the derivative is negative, which we'll skip for now
    sorry
  
  -- Combine the results
  exact ⟨α, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l732_73280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_two_l732_73232

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from the focus to the asymptote of a hyperbola -/
noncomputable def focus_to_asymptote_distance (h : Hyperbola) : ℝ :=
  h.b * Real.sqrt (h.a^2 + h.b^2) / (h.a^2 + h.b^2)

theorem hyperbola_eccentricity_sqrt_two (h : Hyperbola) 
  (h_equal : focus_to_asymptote_distance h = h.a) : 
  eccentricity h = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_two_l732_73232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_B_l732_73214

-- Define set A
def A : Finset ℕ := {1, 2, 3}

-- Define set B
def B : Finset (ℕ × ℕ) := Finset.filter (fun p => p.1 ∈ A ∧ p.2 ∈ A ∧ (p.1 + p.2) ∈ A) (A.product A)

-- Theorem statement
theorem number_of_subsets_of_B : Finset.card (Finset.powerset B) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_B_l732_73214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_24_l732_73210

theorem sum_of_factors_24 : (Finset.filter (λ x => 24 % x = 0) (Finset.range 25)).sum id = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_24_l732_73210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_satisfies_conditions_l732_73207

-- Define the points A, B, and M
def A : Fin 3 → ℝ := ![1, 0, 2]
def B : Fin 3 → ℝ := ![1, -3, 3]
def M : Fin 3 → ℝ := ![0, 0, 7]

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : Fin 3 → ℝ) : ℝ :=
  (p1 0 - p2 0)^2 + (p1 1 - p2 1)^2 + (p1 2 - p2 2)^2

-- Theorem stating that M is on the z-axis and equidistant from A and B
theorem point_M_satisfies_conditions :
  M 0 = 0 ∧ M 1 = 0 ∧ squaredDistance M A = squaredDistance M B := by
  sorry

#eval M 0  -- Should output 0
#eval M 1  -- Should output 0
#eval M 2  -- Should output 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_satisfies_conditions_l732_73207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l732_73254

noncomputable def line_direction : ℝ × ℝ × ℝ := (1, Real.sqrt 2, -1)
def point_P : ℝ × ℝ × ℝ := (-1, 1, -1)
def point_A : ℝ × ℝ × ℝ := (4, 1, -2)

theorem distance_point_to_line :
  let PA : ℝ × ℝ × ℝ := (point_A.1 - point_P.1, point_A.2.1 - point_P.2.1, point_A.2.2 - point_P.2.2)
  let PA_norm := Real.sqrt (PA.1^2 + PA.2.1^2 + PA.2.2^2)
  let m_norm := Real.sqrt (line_direction.1^2 + line_direction.2.1^2 + line_direction.2.2^2)
  let dot_product := PA.1 * line_direction.1 + PA.2.1 * line_direction.2.1 + PA.2.2 * line_direction.2.2
  let cos_angle := dot_product / (PA_norm * m_norm)
  let distance := PA_norm * Real.sqrt (1 - cos_angle^2)
  distance = Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l732_73254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_existence_l732_73288

theorem subset_sum_existence (A B C : Finset ℕ) : 
  (A ∪ B ∪ C = Finset.range 2017 \ {0}) →
  (A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅) →
  (A.card = 672 ∧ B.card = 672 ∧ C.card = 672) →
  ∃ (a b c : ℕ), a ∈ A ∧ b ∈ B ∧ c ∈ C ∧ (a + b = c ∨ b + c = a ∨ c + a = b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_existence_l732_73288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_for_equal_frequency_events_l732_73229

/-- Represents a periodic event with a start time and frequency -/
structure PeriodicEvent where
  start_time : ℚ
  frequency : ℚ

/-- The probability of encountering a specific periodic event
    when choosing a random time, given two periodic events with equal frequencies -/
def probability_of_event (event1 event2 : PeriodicEvent) : ℚ :=
  1 / 2

/-- Theorem stating that the probability of encountering either of two periodic events
    with the same frequency but different start times is equal -/
theorem equal_probability_for_equal_frequency_events
  (event1 event2 : PeriodicEvent)
  (h_freq : event1.frequency = event2.frequency)
  (h_start : event1.start_time ≠ event2.start_time) :
  probability_of_event event1 event2 = 1 / 2 := by
  -- The proof is omitted for now
  sorry

#check equal_probability_for_equal_frequency_events

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_for_equal_frequency_events_l732_73229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l732_73238

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

theorem even_function_condition (ω φ : ℝ) (h : ω > 0) :
  (∀ x, f ω φ x = f ω φ (-x)) ↔ (deriv (f ω φ) 0 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l732_73238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l732_73281

theorem solve_equation (x y : ℝ) (h : 7 * (3 : ℝ)^x = (4 : ℝ)^(y + 3)) :
  y = -3 → x = -Real.log 7 / Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l732_73281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_wins_six_two_one_l732_73296

/-- Represents a wall configuration as a list of integers -/
def WallConfig := List Nat

/-- Represents a player in the game -/
inductive Player
| A
| B

/-- Defines the game rules and winning condition -/
def GameRules (config : WallConfig) (player : Player) : Prop :=
  match player with
  | Player.A => ∃ (move : WallConfig → WallConfig), 
      (∀ c, move c ≠ c) ∧ 
      (∀ c, (move c).length ≤ c.length - 1) ∧
      (∀ c, (move c).length ≥ c.length - 2)
  | Player.B => ∀ (moveA : WallConfig → WallConfig), 
      (moveA config ≠ config) → 
      ((moveA config).length ≤ config.length - 1) → 
      ((moveA config).length ≥ config.length - 2) → 
      ∃ (moveB : WallConfig → WallConfig), 
        (moveB (moveA config) ≠ moveA config) ∧ 
        ((moveB (moveA config)).length ≤ (moveA config).length - 1) ∧ 
        ((moveB (moveA config)).length ≥ (moveA config).length - 2)

/-- Theorem stating that player B has a winning strategy for the initial configuration (6,2,1) -/
theorem b_wins_six_two_one : 
  ∃ (strategy : WallConfig → WallConfig), 
    GameRules [6,2,1] Player.B ∧ 
    (∀ move : WallConfig → WallConfig, GameRules (strategy (move [6,2,1])) Player.B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_wins_six_two_one_l732_73296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l732_73215

-- Define the set of real numbers that satisfy the inequality
def solution_set : Set ℝ := {x | (x - 1) * x ≥ 2}

-- State the theorem
theorem inequality_solution : 
  solution_set = Set.Iic (-1) ∪ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l732_73215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_tetrahedron_volume_ratio_l732_73255

/-- The volume of a regular tetrahedron with edge length a -/
noncomputable def tetrahedron_volume (a : ℝ) : ℝ := (a^3 * Real.sqrt 2) / 12

/-- The volume of a regular octahedron with edge length s -/
noncomputable def octahedron_volume (s : ℝ) : ℝ := (s^3 * Real.sqrt 2) / 3

/-- The edge length of an octahedron formed by joining the centers of adjoining faces of a regular tetrahedron -/
noncomputable def octahedron_edge_length (a : ℝ) : ℝ := (a * Real.sqrt 2) / 2

theorem octahedron_tetrahedron_volume_ratio (a : ℝ) (h : a > 0) :
  octahedron_volume (octahedron_edge_length a) = tetrahedron_volume a := by
  sorry

#check octahedron_tetrahedron_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_tetrahedron_volume_ratio_l732_73255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_iff_m_eq_neg_two_l732_73203

def a (m : ℝ) : ℝ × ℝ × ℝ := (2*m + 1, 3, m - 1)
def b (m : ℝ) : ℝ × ℝ × ℝ := (2, m, -m)

theorem parallel_vectors_iff_m_eq_neg_two :
  ∀ m : ℝ, (∃ k : ℝ, a m = k • b m) ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_iff_m_eq_neg_two_l732_73203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l732_73245

/-- Calculates the simple interest rate given the principal, time, and interest amount -/
noncomputable def simple_interest_rate (principal : ℝ) (time : ℝ) (interest : ℝ) : ℝ :=
  (interest * 100) / (principal * time)

theorem interest_rate_calculation (P : ℝ) :
  P > 0 →
  simple_interest_rate P 8 (P * 5 * 8 / 100) = 5 →
  simple_interest_rate P 5 840 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l732_73245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_division_theorem_l732_73298

-- Define the original cube
def original_cube_edge : ℕ := 4

-- Define the function to calculate the volume of a cube
def cube_volume (edge : ℕ) : ℕ := edge ^ 3

-- Define the total number of smaller cubes
def total_smaller_cubes : ℕ := 57

-- Define the function to check if a list of cubes fills the original cube
def fills_original_cube (cubes : List ℕ) : Prop :=
  (cubes.map cube_volume).sum = cube_volume original_cube_edge ∧
  cubes.all (λ e => e > 0 ∧ e ≤ original_cube_edge) ∧
  cubes.length = total_smaller_cubes ∧
  ∃ (a b : ℕ), a ∈ cubes ∧ b ∈ cubes ∧ a ≠ b

-- Theorem statement
theorem cube_division_theorem :
  ∃ (cubes : List ℕ), fills_original_cube cubes := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_division_theorem_l732_73298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr1_is_complete_square_l732_73243

-- Define the expressions
def expr1 (x : ℝ) := 4 * x^2 - 4 * x + 1
def expr2 (x : ℝ) := 6 * x^2 + 3 * x + 1
def expr3 (x y : ℝ) := x^2 + 4 * x * y + 2 * y^2
def expr4 (x : ℝ) := 9 * x^2 + 18 * x + 1

-- Define a predicate for being a complete square
def is_complete_square (f : ℝ → ℝ) :=
  ∃ a b : ℝ → ℝ, ∀ x, f x = (a x - b x)^2 ∧ (∃ m n : ℝ, a x = m * x + n) ∧ (∃ p q : ℝ, b x = p * x + q)

-- Theorem statement
theorem only_expr1_is_complete_square :
  is_complete_square expr1 ∧
  ¬is_complete_square expr2 ∧
  ¬is_complete_square (λ x => expr3 x x) ∧
  ¬is_complete_square expr4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr1_is_complete_square_l732_73243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l732_73272

def our_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≠ 0 → a n = n * (a (n + 1) - a n)

theorem sequence_properties (a : ℕ → ℝ) (h : our_sequence a) :
  a 2 = 2 ∧ ∀ n : ℕ, n ≠ 0 → a n = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l732_73272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_inequality_solution_l732_73295

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -x^2 + 2*x + 2
  else if x = 0 then 0
  else x^2 + 2*x - 2

-- State the properties of f
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = -x^2 + 2*x + 2) := by sorry

-- State the solution set for f(x) ≤ 1
theorem f_inequality_solution :
  {x : ℝ | f x ≤ 1} = Set.Icc (-3) 0 ∪ Set.Ici (1 + Real.sqrt 2) := by sorry

#check f_properties
#check f_inequality_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_inequality_solution_l732_73295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l732_73285

noncomputable def arithmetic_sequence_sum : ℕ → ℕ
| 0 => 0
| n + 1 => arithmetic_sequence_sum n + (2 * n + 1)

noncomputable def sum_odd (n : ℕ) : ℕ := arithmetic_sequence_sum n

def sum_even (n : ℕ) : ℕ := n * (n + 1)

theorem problem_solution : sum_odd 1013 - sum_even 1012 + 50 = 963 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l732_73285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downloaded_size_correct_l732_73217

/-- Given a file download scenario where:
  * The total file size is 1.5 MB
  * Initially, half the file (0.75 MB) is downloaded
  * The initial estimated remaining time is 2 minutes
  * After t additional minutes, the estimated remaining time is still 2 minutes
  This function calculates the size of the file downloaded after t minutes -/
noncomputable def downloaded_size (t : ℝ) : ℝ :=
  (3 + 1.5 * t) / (4 + t)

/-- Theorem stating that the downloaded_size function correctly represents
    the size of the file downloaded after t minutes under the given conditions -/
theorem downloaded_size_correct (t : ℝ) :
  let total_size : ℝ := 1.5
  let initial_download : ℝ := 0.75
  let initial_time : ℝ := 2
  let remaining_time : ℝ := 2
  downloaded_size t = (3 + 1.5 * t) / (4 + t) ∧
  downloaded_size t ≤ total_size ∧
  downloaded_size 0 = initial_download ∧
  (downloaded_size t - initial_download) / t = (total_size - downloaded_size t) / remaining_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downloaded_size_correct_l732_73217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_periodic_equivalence_main_theorem_l732_73276

noncomputable section

open Real

theorem trigonometric_identities :
  (∀ x, Real.arccos (Real.sin (-x)) = π - Real.arccos (Real.sin x)) ∧
  (∀ x, Real.arccos (Real.sin x) = π / 2 - x) ∧
  (∀ x, Real.arcsin (Real.cos x) = π / 2 - x) := by sorry

theorem periodic_equivalence :
  ∀ x, Real.cos x = Real.cos (x % (2 * π)) := by sorry

theorem main_theorem :
  Real.arccos (Real.sin (-π / 7)) = 9 * π / 14 ∧
  Real.arcsin (Real.cos (33 * π / 5)) = -π / 10 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_periodic_equivalence_main_theorem_l732_73276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_identity_l732_73282

theorem expression_identity : 
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + Real.log 5^64 / Real.log 2) = 
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + Real.log 5^64 / Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_identity_l732_73282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solution_l732_73216

theorem factorial_equation_solution : ∃ n : ℕ, 6 * 8 * 2 * n = Nat.factorial 8 ∧ n = 420 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solution_l732_73216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bounds_l732_73294

theorem angle_sum_bounds (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < Real.pi/2)
  (h_acute_β : 0 < β ∧ β < Real.pi/2)
  (h_acute_γ : 0 < γ ∧ γ < Real.pi/2)
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  3*Real.pi/4 < α + β + γ ∧ α + β + γ < Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bounds_l732_73294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l732_73291

/-- Proposition p: For all x in [1,2], x^2 - a ≥ 0 -/
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

/-- Proposition q: There exists x₀ in ℝ, such that x₀^2 + 2ax₀ + 2 - a = 0 -/
def prop_q (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

/-- The range of real number a for which "Proposition p AND Proposition q" is false -/
def a_range : Set ℝ :=
  Set.Ioo (-2) 1 ∪ Set.Ioi 1

theorem a_range_theorem :
  ∀ a : ℝ, a ∈ a_range ↔ ¬(prop_p a ∧ prop_q a) := by
  sorry

#check a_range_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l732_73291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_HClO4_required_proof_l732_73220

/-- Represents the number of moles of a substance -/
structure Moles where
  value : ℝ
  nonNeg : 0 ≤ value

/-- Addition of moles -/
instance : Add Moles where
  add a b := ⟨a.value + b.value, add_nonneg a.nonNeg b.nonNeg⟩

/-- Conversion from ℝ to Moles -/
def realToMoles (r : ℝ) (h : 0 ≤ r) : Moles := ⟨r, h⟩

/-- Reaction between HClO₄ and NaOH -/
def reaction_NaOH (hclo4 : Moles) (naoh : Moles) : Prop :=
  hclo4.value = naoh.value

/-- Reaction between HClO₄ and KOH -/
def reaction_KOH (hclo4 : Moles) (koh : Moles) : Prop :=
  hclo4.value = koh.value

/-- The total moles of HClO₄ required for both reactions -/
def total_HClO4_required (naoh : Moles) (koh : Moles) : Moles :=
  naoh + koh

/-- Theorem: The total moles of HClO₄ required to react with 1 mole of NaOH and 0.5 moles of KOH is 1.5 moles -/
theorem HClO4_required_proof (naoh koh : Moles) 
  (h1 : naoh.value = 1)
  (h2 : koh.value = 0.5)
  (h3 : reaction_NaOH naoh naoh)
  (h4 : reaction_KOH koh koh) :
  (total_HClO4_required naoh koh).value = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_HClO4_required_proof_l732_73220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_forms_equilateral_triangle_l732_73270

-- Define the hyperbola
def hyperbola (c : ℝ) : Set (ℝ × ℝ) := {p | p.1 * p.2 = c^2}

-- Define a point P on the hyperbola
def P_on_hyperbola (c : ℝ) (P : ℝ × ℝ) : Prop := P ∈ hyperbola c

-- Define the symmetric point Q
def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- Define the circle with center P and radius PQ
def circle_pq (P : ℝ × ℝ) (Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x | (x.1 - P.1)^2 + (x.2 - P.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2}

-- Define the intersection points of the circle and the hyperbola
def intersection_points (c : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  hyperbola c ∩ circle_pq P Q

-- Define an equilateral triangle
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

-- Theorem statement
theorem hyperbola_circle_intersection_forms_equilateral_triangle
  (c : ℝ) (P : ℝ × ℝ) (Q A B C : ℝ × ℝ) :
  P_on_hyperbola c P →
  Q = symmetric_point P →
  {A, B, C, Q} ⊆ intersection_points c P Q →
  is_equilateral A B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_forms_equilateral_triangle_l732_73270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_value_l732_73208

noncomputable def f (l : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) - (x * (1 + l * x)) / (1 + x)

theorem min_lambda_value :
  ∃ l_min : ℝ, l_min = 1/2 ∧
    (∀ l : ℝ, (∀ x : ℝ, x ≥ 0 → f l x ≤ 0) → l ≥ l_min) ∧
    (∀ x : ℝ, x ≥ 0 → f l_min x ≤ 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_value_l732_73208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_successes_in_three_trials_l732_73227

def probability_of_success : ℝ := 0.6
def number_of_trials : ℕ := 3
def number_of_successes : ℕ := 2

theorem exactly_two_successes_in_three_trials :
  (Nat.choose number_of_trials number_of_successes : ℝ) * 
  probability_of_success ^ number_of_successes * 
  (1 - probability_of_success) ^ (number_of_trials - number_of_successes) = 54/125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_successes_in_three_trials_l732_73227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l732_73269

theorem expression_equality : 
  ((-8 : ℝ) ^ (1/3 : ℝ)) + (-1)^(2023 : ℕ) - abs (1 - Real.sqrt 2) + (-Real.sqrt 3)^2 = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l732_73269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_21_l732_73211

def f : ℕ → ℤ
  | 0 => 3  -- We need to define f(0) to cover all natural numbers
  | 1 => 3
  | 2 => 5
  | (n+3) => f (n+2) - 2 * f (n+1) + 2 * (n+3)

theorem f_10_equals_21 : f 10 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_21_l732_73211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_when_f_g_decreasing_l732_73230

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x
def g (a : ℝ) (x : ℝ) : ℝ := a/x

-- Define what it means for a function to be decreasing on an interval
def DecreasingOn (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → h y < h x

-- State the theorem
theorem a_range_when_f_g_decreasing :
  ∀ a : ℝ, 
    (DecreasingOn (f a) 1 2 ∧ DecreasingOn (g a) 1 2) → 
    (0 < a ∧ a ≤ 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_when_f_g_decreasing_l732_73230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_problem_region_l732_73265

/-- The line equation in the form ax + by = c -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A triangular region bounded by coordinate axes and a line -/
structure TriangularRegion where
  line : LineEquation

/-- Calculate the area of a triangular region -/
noncomputable def area (region : TriangularRegion) : ℝ :=
  let x_intercept := region.line.c / region.line.a
  let y_intercept := region.line.c / region.line.b
  x_intercept * y_intercept / 2

/-- The specific triangular region in the problem -/
def problem_region : TriangularRegion :=
  { line := { a := 3, b := 2, c := 12 } }

theorem area_of_problem_region :
  area problem_region = 12 := by
  -- Unfold the definitions
  unfold area
  unfold problem_region
  -- Simplify the expressions
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_problem_region_l732_73265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_alignment_l732_73235

theorem brick_alignment (a b c : ℤ) : 
  (∃ n : ℤ, ∃ seq : List (Fin 3 × Fin 3), 
    let final := seq.foldl (λ acc pair ↦
      match pair with
      | (0, 1) => (acc.1 - 1, acc.2.1 + 1, acc.2.2)
      | (0, 2) => (acc.1 - 1, acc.2.1, acc.2.2 + 1)
      | (1, 0) => (acc.1 + 1, acc.2.1 - 1, acc.2.2)
      | (1, 2) => (acc.1, acc.2.1 - 1, acc.2.2 + 1)
      | (2, 0) => (acc.1 + 1, acc.2.1, acc.2.2 - 1)
      | (2, 1) => (acc.1, acc.2.1 + 1, acc.2.2 - 1)
      | _ => acc
    ) (a, b, c)
    final.1 = n ∧ final.2.1 = n ∧ final.2.2 = n) ↔ 
  (a + b + c) % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_alignment_l732_73235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l732_73260

def A : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem intersection_of_A_and_B :
  (A.image (coe : ℤ → ℝ) ∩ B) = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l732_73260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tied_teams_is_seven_l732_73251

/-- Represents a round-robin tournament. -/
structure Tournament :=
  (num_teams : ℕ)
  (games : List (Fin num_teams × Fin num_teams))
  (winner : Fin num_teams × Fin num_teams → Fin num_teams)

/-- The number of wins for each team in the tournament. -/
def wins (t : Tournament) (team : Fin t.num_teams) : ℕ :=
  (t.games.filter (fun g => t.winner g = team)).length

/-- Predicate to check if a tournament is valid. -/
def valid_tournament (t : Tournament) : Prop :=
  t.num_teams = 8 ∧
  t.games.length = (t.num_teams.choose 2) ∧
  ∀ i j : Fin t.num_teams, i ≠ j → ((i, j) ∈ t.games ∨ (j, i) ∈ t.games) ∧
  ∀ g : Fin t.num_teams × Fin t.num_teams, g ∈ t.games → (t.winner g = g.1 ∨ t.winner g = g.2)

/-- The maximum number of teams that can be tied for the most wins. -/
def max_tied_teams (t : Tournament) : ℕ :=
  let max_wins := (List.finRange t.num_teams).map (wins t) |>.maximum?
  (List.finRange t.num_teams).filter (fun team => wins t team = max_wins.getD 0) |>.length

/-- Theorem stating the maximum number of teams tied for most wins. -/
theorem max_tied_teams_is_seven (t : Tournament) (h : valid_tournament t) :
  max_tied_teams t = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tied_teams_is_seven_l732_73251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l732_73222

-- Define the function f(x) = x^2 + log_10(x) - 3
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x / Real.log 10 - 3

-- State the theorem
theorem zero_in_interval :
  ∃ c ∈ Set.Ioo (3/2 : ℝ) 2, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l732_73222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_theorem_l732_73263

-- Define the floor function
def floor (x : ℚ) : ℤ := Int.floor x

-- Define the product function
def product (k : ℕ) : ℕ :=
  Finset.prod (Finset.range k.succ) (λ i => (floor ((i : ℚ) / 7) + 1).toNat)

-- State the theorem
theorem largest_k_theorem :
  ∃ (k : ℕ), k ≤ 48 ∧
    product k % 13 = 7 ∧
    ∀ (m : ℕ), m > k → m ≤ 48 → product m % 13 ≠ 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_theorem_l732_73263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_axis_l732_73257

theorem equidistant_point_x_axis : ∃ x : ℝ, 
  let point : ℝ × ℝ := (x, 0)
  let A : ℝ × ℝ := (-4, 0)
  let B : ℝ × ℝ := (0, 6)
  (point.1 - A.1)^2 + (point.2 - A.2)^2 = (point.1 - B.1)^2 + (point.2 - B.2)^2 ∧ 
  x = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_axis_l732_73257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l732_73244

noncomputable def f (x : ℝ) : ℝ := Real.cos (x - Real.pi / 2) + Real.sin (x + Real.pi / 3)

theorem f_monotone_increasing :
  ∀ k : ℤ, ∀ x y : ℝ,
    x ∈ Set.Ioo (2 * (k : ℝ) * Real.pi - 2 * Real.pi / 3) (2 * (k : ℝ) * Real.pi + Real.pi / 3) →
    y ∈ Set.Ioo (2 * (k : ℝ) * Real.pi - 2 * Real.pi / 3) (2 * (k : ℝ) * Real.pi + Real.pi / 3) →
    x < y →
    f x < f y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l732_73244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_halving_line_l732_73240

/-- An isosceles triangle with angle α between its equal sides has a line parallel to the base
    that halves both its area and perimeter if and only if α = 2 * arcsin(√2 - 1) -/
theorem isosceles_triangle_halving_line (α : ℝ) :
  (∃ b a : ℝ, b > 0 ∧ a > 0 ∧
    ∃ x l : ℝ, 0 < x ∧ x < b ∧ 0 < l ∧ l < a ∧
      x^2 / b^2 = 1/2 ∧
      2*x + l = b + a/2 ∧
      l/a = x/b ∧
      a = 2*b*Real.sin (α/2)) ↔
  α = 2 * Real.arcsin (Real.sqrt 2 - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_halving_line_l732_73240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_bag_wholesale_cost_l732_73224

/-- The wholesale cost of a sleeping bag given the retail price and profit margin. -/
noncomputable def wholesale_cost (retail_price : ℝ) (profit_margin : ℝ) : ℝ :=
  retail_price / (1 + profit_margin)

/-- Theorem stating the wholesale cost of a sleeping bag given specific conditions. -/
theorem sleeping_bag_wholesale_cost :
  let retail_price : ℝ := 28
  let profit_margin : ℝ := 0.16
  let calculated_wholesale_cost : ℝ := wholesale_cost retail_price profit_margin
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |calculated_wholesale_cost - 24.14| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_bag_wholesale_cost_l732_73224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equations_l732_73268

theorem solution_satisfies_equations :
  ∃ (x y z : ℤ),
    (z : ℚ)^x = (y : ℚ)^(3*x) ∧
    (2 : ℚ)^z = 8 * (8 : ℚ)^x ∧
    x + y + z = 20 ∧
    x = 2 ∧ y = 9 ∧ z = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equations_l732_73268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_count_l732_73212

/-- A polynomial x^2 + ax + b is factorizable if it can be written as (x-r)(x-s) 
    where r and s are distinct non-zero integers -/
def IsFactorizable (a b : ℤ) : Prop :=
  ∃ r s : ℤ, r ≠ s ∧ r ≠ 0 ∧ s ≠ 0 ∧ (∀ x, x^2 + a*x + b = (x - r) * (x - s))

/-- The set of ordered pairs (a,b) satisfying the given conditions -/
def ValidPairs : Set (ℤ × ℤ) :=
  {p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ p.2 > 0 ∧ IsFactorizable p.1 p.2}

/-- The number of valid pairs is finite -/
instance : Fintype ValidPairs := by
  sorry -- We need to prove that ValidPairs is finite, which it is, but the proof is non-trivial

/-- The count of valid pairs -/
theorem valid_pairs_count : Fintype.card ValidPairs = 925 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_count_l732_73212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l732_73249

noncomputable def f (a b x : ℝ) : ℝ := x + a * x^2 + b * Real.log x

theorem function_properties (a b : ℝ) :
  (f a b 1 = 0) →
  ((deriv (f a b)) 1 = 2) →
  (a = -1 ∧ b = 3) ∧
  (∀ x > 0, x - x^2 + 3 * Real.log x ≤ 2 * x - 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l732_73249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_l732_73225

/-- Given points A, B, and C in 3D space, prove that A is equidistant from B and C -/
theorem point_equidistant (A B C : ℝ × ℝ × ℝ) : 
  A = (-21, 0, 0) ∧ B = (-2, -4, -6) ∧ C = (-1, -2, -3) →
  (A.1 - B.1)^2 + (A.2.1 - B.2.1)^2 + (A.2.2 - B.2.2)^2 = 
  (A.1 - C.1)^2 + (A.2.1 - C.2.1)^2 + (A.2.2 - C.2.2)^2 := by
  sorry

#check point_equidistant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_l732_73225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_path_count_l732_73267

/-- Represents the number of unshaded squares in each row of the grid -/
def grid : List Nat := [3, 2, 3, 2, 3]

/-- Calculates the number of paths to reach each unshaded square in a given row -/
def paths_to_row (prev_row : List Nat) (current_row_size : Nat) : List Nat :=
  match prev_row, current_row_size with
  | [], _ => []
  | [x], 1 => [x]
  | [x, y], 2 => [x + y, x + y]
  | [x, y, z], 3 => [y, x + y + z, y]
  | _, _ => []  -- Default case, should not occur in our problem

/-- Calculates the total number of paths from top to bottom row -/
def total_paths : Nat :=
  let init_row := List.replicate (grid.head!) 1
  let final_row := grid.tail.foldl (fun acc n => paths_to_row acc n) init_row
  final_row.sum

theorem coin_path_count : total_paths = 24 := by
  sorry

#eval total_paths  -- This will evaluate and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_path_count_l732_73267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l732_73274

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a + 1)^(1 - x)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, x < y → f a x > f a y) ∧
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, x < y → g a x > g a y) →
  a ∈ Set.Ioo 0 1 ∪ {1} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l732_73274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l732_73293

/-- The length of a train overtaking a motorbike -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) :
  train_speed = 150 →
  motorbike_speed = 90 →
  overtake_time = 12 →
  ∃ (train_length : ℝ), abs (train_length - 200.04) < 0.01 ∧ train_length > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l732_73293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_problem_l732_73287

/-- The time it takes for pipe A to fill the tank -/
noncomputable def fill_time_A : ℝ := 30 / 13

/-- The initial fullness of the tank -/
noncomputable def initial_fullness : ℝ := 1 / 5

/-- The time it takes for pipe B to empty the tank -/
noncomputable def empty_time_B : ℝ := 6

/-- The time it takes to empty or fill the tank with both pipes open -/
noncomputable def both_pipes_time : ℝ := 3.000000000000001

theorem water_tank_problem :
  let fill_rate_A := 1 / fill_time_A
  let empty_rate_B := 1 / empty_time_B
  let combined_rate := fill_rate_A - empty_rate_B
  let tank_to_fill := 1 - initial_fullness
  tank_to_fill / both_pipes_time = combined_rate :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_problem_l732_73287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systems_solutions_l732_73266

-- System 1
def system1 (x y : ℝ) : Prop :=
  x - y - 1 = 0 ∧ 4 * (x - y) - y = 5

-- System 2
def system2 (x y : ℝ) : Prop :=
  2 * x - 3 * y - 2 = 0 ∧ (2 * x - 3 * y + 5) / 7 + 2 * y = 9

-- System 3
def system3 (x y : ℝ) (m : ℕ) : Prop :=
  2 * x + y = -3 * (m : ℝ) + 2 ∧ x + 2 * y = 7

theorem systems_solutions :
  (∃ x y, system1 x y ∧ x = 0 ∧ y = -1) ∧
  (∃ x y, system2 x y ∧ x = 7 ∧ y = 4) ∧
  (∀ m : ℕ, m > 0 → (∃ x y, system3 x y m ∧ x + y > -5/6) ↔ m = 1 ∨ m = 2 ∨ m = 3) := by
  sorry

#check systems_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systems_solutions_l732_73266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_degree_is_four_l732_73275

/-- A polynomial type with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The set of coefficients of a polynomial -/
def coefficientSet (p : IntPolynomial) : Set ℤ :=
  {a : ℤ | ∃ (n : ℕ), a = p.coeff n}

/-- The property that a polynomial satisfies the given conditions -/
def satisfiesCondition (p : IntPolynomial) : Prop :=
  ∃ (b : ℤ), 
    (∃ (x y : ℤ), x ∈ coefficientSet p ∧ y ∈ coefficientSet p ∧ x < b ∧ y > b) ∧
    b ∉ coefficientSet p

/-- The theorem stating that the lowest degree of a polynomial satisfying the condition is 4 -/
theorem lowest_degree_is_four :
  ∃ (p : IntPolynomial), satisfiesCondition p ∧ p.degree = 4 ∧
  ∀ (q : IntPolynomial), satisfiesCondition q → q.degree ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_degree_is_four_l732_73275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_trajectory_l732_73236

-- Define the circle C
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def Line (x y : ℝ) : Prop := (3*x - 4*y + 5 = 0) ∨ (x = 1)

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the distance between A and B
noncomputable def AB_distance : ℝ := 2 * Real.sqrt 3

-- Define the point M on circle C
def M (x y : ℝ) : Prop := Circle x y ∧ y ≠ 0

-- Define the line m parallel to x-axis
def m (y : ℝ) : ℝ → Prop := λ _ => True

-- Define the point N
def N (y : ℝ) : ℝ × ℝ := (0, y)

-- Define the point Q
def Q (x y : ℝ) : Prop := x^2/4 + y^2/16 = 1 ∧ y ≠ 0

theorem line_and_trajectory :
  ∀ (x y : ℝ),
  (∃ (a b : ℝ), Circle a b ∧ Line a b) →
  (Line x y ↔ (3*x - 4*y + 5 = 0 ∨ x = 1)) ∧
  (M x y → Q x (2*y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_trajectory_l732_73236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_y_axis_plane_l732_73226

/-- The angle between the y-axis and a plane with normal vector (1, -√3, 0) is π/3 -/
theorem angle_y_axis_plane (n : ℝ × ℝ × ℝ) (θ : ℝ) : 
  n = (1, -Real.sqrt 3, 0) → 
  θ = Real.arcsin (Real.sqrt 3 / 2) →
  θ = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_y_axis_plane_l732_73226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l732_73209

def A : Set ℤ := {x : ℤ | Real.sqrt (x - 1 : ℝ) ≤ 2}
def B (a : ℝ) : Set ℤ := {x : ℤ | (x : ℝ) ≤ a}

theorem range_of_a (a : ℝ) : A ∩ B a = A → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l732_73209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_b_in_special_triangle_l732_73221

/-- In a triangle ABC, if A = 2B and the ratio of sides a to b is 3:2, then cos B = 3/4 -/
theorem cosine_b_in_special_triangle (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = Real.pi → 
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  a / Real.sin A = b / Real.sin B →  -- Law of sines
  A = 2 * B →
  2 * a = 3 * b →
  Real.cos B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_b_in_special_triangle_l732_73221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_to_cos_product_l732_73218

theorem sin_sum_to_cos_product :
  (∀ x : ℝ, (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (5*x))^2 + (Real.sin (7*x))^2 = 2) ↔
  (∀ x : ℝ, Real.cos (8*x) * Real.cos (4*x) * Real.cos (2*x) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_to_cos_product_l732_73218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_quadrilateral_area_l732_73279

/-- Trisection point on a line segment -/
def trisection_point (A B : ℝ × ℝ) (n : ℕ) : ℝ × ℝ := sorry

/-- Area of a triangle given its vertices -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Area of a quadrilateral given its vertices -/
def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Intersection point of two lines defined by two points each -/
def intersection_point (A B C D : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Given a triangle ABC with area 1 and trisection points, prove the area of the inner quadrilateral -/
theorem inner_quadrilateral_area (A B C : ℝ × ℝ) 
  (h_area : area_triangle A B C = 1) 
  (B₁ B₂ : ℝ × ℝ) 
  (h_B₁ : B₁ = trisection_point A B 1)
  (h_B₂ : B₂ = trisection_point A B 2)
  (C₁ C₂ : ℝ × ℝ)
  (h_C₁ : C₁ = trisection_point A C 1)
  (h_C₂ : C₂ = trisection_point A C 2) :
  area_quadrilateral 
    (intersection_point B₁ C B C₁)
    (intersection_point B₂ C B C₁)
    (intersection_point B₂ C B C₂)
    (intersection_point B₁ C B C₂) = 9 / 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_quadrilateral_area_l732_73279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_face_angle_tangent_l732_73204

/-- A square pyramid with a specific configuration -/
structure SquarePyramid where
  -- The side length of the square base
  base_side : ℝ
  -- The height of the pyramid
  height : ℝ
  -- The acute angle between two adjacent faces
  face_angle : ℝ
  -- Constraint that the base side is positive
  base_positive : 0 < base_side
  -- Constraint that the height is positive
  height_positive : 0 < height
  -- Constraint that the face angle is acute
  face_angle_acute : 0 < face_angle ∧ face_angle < Real.pi / 2

/-- The tangent of the face angle in a specific square pyramid configuration -/
noncomputable def face_angle_tangent (p : SquarePyramid) : ℝ :=
  17 / 144

/-- Theorem stating that for a square pyramid with base side 12 and height 1/2,
    the tangent of the face angle is 17/144 -/
theorem square_pyramid_face_angle_tangent :
  ∀ (p : SquarePyramid), 
    p.base_side = 12 → p.height = 1/2 → 
    Real.tan p.face_angle = face_angle_tangent p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_face_angle_tangent_l732_73204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l732_73237

noncomputable def parabola (x : ℝ) : ℝ := x^2 - 6*x + 11

noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y1) - (y1*x2 + y2*x3 + y3*x1))

theorem max_triangle_area :
  ∃ (p : ℝ), 2 ≤ p ∧ p ≤ 5 ∧
  ∀ (q : ℝ), q = parabola p →
  ∀ (r : ℝ), 2 ≤ r ∧ r ≤ 5 →
  ∀ (s : ℝ), s = parabola r →
  triangle_area 2 0 5 3 r s ≤ 4.5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l732_73237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l732_73264

theorem triangle_angle_inequalities (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Real.sin α * Real.sin β * Real.sin γ ≤ (3 * Real.sqrt 3) / 8 ∧
  Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2) ≤ (3 * Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l732_73264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l732_73262

theorem simplify_trig_expression (x : ℝ) (h : x ≠ π) :
  (Real.sin x) / (1 + Real.cos x) + (1 + Real.cos x) / (Real.sin x) = 2 / Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l732_73262
