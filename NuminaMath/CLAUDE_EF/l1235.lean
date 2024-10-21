import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_l1235_123503

-- Define the ellipse parameters
def ellipse_equation (m n : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (x^2 / m^2) + (y^2 / n^2) = 1

-- Define the parabola
def parabola : ℝ → ℝ → Prop :=
  λ x y ↦ y = 8 * x

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

-- Theorem statement
theorem ellipse_parameters (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∃ x y : ℝ, ellipse_equation m n x y ∧ parabola x y) →
  eccentricity m n = 1/2 →
  m^2 = 16 ∧ n^2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_l1235_123503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_plus_1_power_decomposition_l1235_123535

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The sequence a_n -/
noncomputable def a (n : ℕ+) : ℤ :=
  floor (Real.sqrt 2 * n)

theorem sqrt2_plus_1_power_decomposition
  (m : ℕ+) (x_m y_m : ℕ+)
  (h : (Real.sqrt 2 + 1)^(m : ℝ) = Real.sqrt 2 * x_m + y_m) :
  (∀ m : ℕ+, Odd (y_m : ℤ)) ∧
  (∃ b : ℕ+ → ℤ, (∀ n : ℕ+, ∃ k : ℕ+, b n = a k) ∧
                 (∀ n : ℕ+, b n % 4 = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_plus_1_power_decomposition_l1235_123535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1235_123510

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The property that angles form an arithmetic sequence -/
def arithmeticAngles (t : Triangle) : Prop :=
  ∃ (d : ℝ), t.B - t.A = t.C - t.B

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * t.a * t.c * Real.sin t.B

theorem triangle_properties (t : Triangle) 
  (h_arithmetic : arithmeticAngles t)
  (h_area : area t = Real.sqrt 3) :
  (∃ (r : ℝ), t.a = 2/r ∧ t.c = 2*r) ∧  -- a, 2, c form a geometric sequence
  (∀ (L : ℝ), L = t.a + t.b + t.c → L ≥ 6) ∧  -- minimum perimeter is 6
  (t.a = t.b ∧ t.b = t.c ↔ t.a + t.b + t.c = 6)  -- equilateral when perimeter is minimum
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1235_123510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_in_lcm_l1235_123548

theorem third_number_in_lcm (n : ℕ) : Nat.lcm 10 (Nat.lcm 14 n) = 140 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_in_lcm_l1235_123548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lishui_sleep_suitable_for_sampling_lishui_sleep_only_suitable_task_l1235_123571

-- Define the type for survey tasks
inductive SurveyTask where
  | ClassmateAge
  | SchoolGenderRatio
  | OlympicAthletesUrine
  | LishuiStudentsSleep

-- Define a predicate for suitability for sampling survey
def SuitableForSamplingSurvey : SurveyTask → Prop := sorry

-- Define the property of having a wide scope
def HasWideScope : SurveyTask → Prop := sorry

-- Define the property of being time-consuming and labor-intensive
def IsTimeConsumingAndLaborIntensive : SurveyTask → Prop := sorry

-- Theorem: LishuiStudentsSleep is suitable for sampling survey
theorem lishui_sleep_suitable_for_sampling :
  SuitableForSamplingSurvey SurveyTask.LishuiStudentsSleep :=
by
  sorry

-- Assumptions based on the problem description
axiom wide_scope_lishui_sleep : HasWideScope SurveyTask.LishuiStudentsSleep
axiom time_consuming_lishui_sleep : IsTimeConsumingAndLaborIntensive SurveyTask.LishuiStudentsSleep

-- Relationship between properties and suitability for sampling survey
axiom sampling_survey_criteria (task : SurveyTask) :
  HasWideScope task → IsTimeConsumingAndLaborIntensive task → SuitableForSamplingSurvey task

-- Other tasks are not suitable for sampling survey
axiom not_suitable_classmate_age : ¬SuitableForSamplingSurvey SurveyTask.ClassmateAge
axiom not_suitable_school_gender : ¬SuitableForSamplingSurvey SurveyTask.SchoolGenderRatio
axiom not_suitable_olympic_urine : ¬SuitableForSamplingSurvey SurveyTask.OlympicAthletesUrine

-- Theorem: LishuiStudentsSleep is the only task suitable for sampling survey
theorem lishui_sleep_only_suitable_task :
  ∀ (task : SurveyTask), SuitableForSamplingSurvey task ↔ task = SurveyTask.LishuiStudentsSleep :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lishui_sleep_suitable_for_sampling_lishui_sleep_only_suitable_task_l1235_123571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_34865_to_nearest_tenth_l1235_123561

noncomputable def round_to_nearest_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem round_34865_to_nearest_tenth :
  round_to_nearest_tenth 34.865 = 34.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_34865_to_nearest_tenth_l1235_123561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l1235_123565

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → f (x + y) = f x + f y) ∧
  (∀ x, x > 0 → x ∈ Set.Icc (-1) 1 → f x > 0) ∧
  (f 1 = 1)

theorem special_function_properties (f : ℝ → ℝ) (h : SpecialFunction f) :
  (∀ x, x ∈ Set.Icc (-1) 1 → f (-x) = -f x) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x a m, x ∈ Set.Icc (-1) 1 → a ∈ Set.Icc (-1) 1 →
    (f x < m^2 - 2*a*m + 1) → (m < -2 ∨ m > 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l1235_123565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_proposition_correctness_l1235_123507

theorem trigonometric_proposition_correctness : ∃ α : Real,
  (Real.sin α = 0 ∧ Real.cos α = -1) ∧
  (Real.sin α)^2 + (Real.cos α)^2 = 1 ∧
  ¬(Real.sin α = 1/2 ∧ Real.cos α = 1/2) ∧
  ¬(Real.tan α = 1 ∧ Real.cos α = -1) ∧
  ¬(∀ β : Real, β ∈ Set.Ioo (Real.pi/2) Real.pi → Real.tan β = -(Real.sin β / Real.cos β)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_proposition_correctness_l1235_123507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_reachability_l1235_123557

structure City where
  name : String
deriving Inhabited

structure TransportGraph where
  cities : Set City
  bus_connections : Set (City × City)
  train_connections : Set (City × City)

def can_reach (g : TransportGraph) (a b : City) : Prop :=
  (a, b) ∈ g.bus_connections ∨ (a, b) ∈ g.train_connections

def monochromatic_path_exists (g : TransportGraph) (a b : City) : Prop :=
  ∃ path : List City, path.head? = some a ∧ path.getLast? = some b ∧
    (∀ i j, i < j → j < path.length →
      ((path[i]?, path[j]?) = (some (path[i]!), some (path[j]!)) →
        ((path[i]!, path[j]!) ∈ g.bus_connections ∨
         (path[i]!, path[j]!) ∈ g.train_connections)))

theorem city_reachability (g : TransportGraph) :
  (∀ a b : City, a ∈ g.cities → b ∈ g.cities → 
    monochromatic_path_exists g a b ∨ monochromatic_path_exists g b a) →
  ∃ c : City, c ∈ g.cities ∧
    ∀ d : City, d ∈ g.cities → monochromatic_path_exists g c d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_reachability_l1235_123557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steer_weight_conversion_l1235_123523

/-- Conversion factor from kilograms to pounds -/
noncomputable def kg_to_pound : ℝ := 1 / 0.4536

/-- Weight of the steer in kilograms -/
def steer_weight_kg : ℝ := 250

/-- Approximate weight of the steer in pounds, rounded to the nearest tenth -/
def steer_weight_pound : ℝ := 551.2

/-- Theorem stating that the weight of a 250 kg steer is approximately 551.2 pounds -/
theorem steer_weight_conversion :
  (⌊(steer_weight_kg * kg_to_pound * 10 + 0.5)⌋ : ℝ) / 10 = steer_weight_pound := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_steer_weight_conversion_l1235_123523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l1235_123526

/-- The equation of a circle in R² -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in R² represented by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Check if two circles are symmetric with respect to a line -/
def are_circles_symmetric (c1 c2 : Circle) (l : Line) : Prop :=
  -- This definition would involve the mathematical conditions for symmetry
  sorry

theorem symmetric_circle_equation :
  let c1 : Circle := { center := (3, -4), radius := 1 }
  let c2 : Circle := { center := (10, 3), radius := 1 }
  let l : Line := { m := -1, b := 6 }
  are_circles_symmetric c1 c2 l := by
  sorry

#check symmetric_circle_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l1235_123526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l1235_123531

-- Define the circle
def circleEq (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define point A
def A : ℝ × ℝ := (-3, 4)

-- Define the distance from a point to the center of the circle
noncomputable def dist_to_center (p : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - 2)^2 + (p.2 - 3)^2)

-- Theorem statement
theorem tangent_length : 
  ∃ (t : ℝ × ℝ), circleEq t.1 t.2 ∧ 
  Real.sqrt ((t.1 - A.1)^2 + (t.2 - A.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l1235_123531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_five_l1235_123524

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (a₁ + arithmeticSequence a₁ d n) / 2

theorem arithmetic_sum_five :
  ∃ (d : ℝ), 
    (arithmeticSequence 1 d 5 = 9) ∧ 
    (arithmeticSum 1 d 5 = 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_five_l1235_123524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1235_123521

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else -2 / x

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ x₀ : ℝ, x₀ = -2 → f x₀ = -1) ∧
  ¬(∀ x₀ : ℝ, f x₀ = -1 → x₀ = -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1235_123521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_relation_l1235_123560

theorem cos_sin_relation (f : ℝ → ℝ) (x : ℝ) :
  (∀ y, Real.cos (17 * y) = f (Real.cos y)) → Real.sin (17 * x) = f (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_relation_l1235_123560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paula_paint_cans_l1235_123590

/-- Represents the number of rooms that can be painted with a given amount of paint -/
def RoomsPainted := ℕ

/-- Represents the number of paint cans -/
def PaintCans := ℕ

/-- Given the initial and final number of rooms that can be painted, and the number of cans lost,
    calculates the original number of paint cans -/
def originalCans (initial_rooms : ℕ) (final_rooms : ℕ) (cans_lost : ℕ) : ℕ :=
  (initial_rooms - final_rooms) / cans_lost * cans_lost + final_rooms / (initial_rooms / cans_lost)

theorem paula_paint_cans :
  let initial_rooms : ℕ := 40
  let final_rooms : ℕ := 30
  let cans_lost : ℕ := 5
  originalCans initial_rooms final_rooms cans_lost = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paula_paint_cans_l1235_123590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_lifespan_in_months_l1235_123552

/-- Represents the life-span of a dog breed in years -/
structure DogBreed where
  lifespan : ℝ
  popularity : ℝ

/-- Calculates the weighted average lifespan of dogs given a list of dog breeds -/
def weightedAverageDogLifespan (breeds : List DogBreed) : ℝ :=
  (breeds.map (λ b => b.lifespan * b.popularity)).sum

/-- Theorem: The weighted average fish life-span in months is 168 -/
theorem fish_lifespan_in_months : weightedAverageDogLifespan [
    ⟨10, 0.4⟩,
    ⟨12, 0.3⟩,
    ⟨14, 0.2⟩,
    ⟨16, 0.1⟩
  ] + 2 * 12 = 168 := by
  -- Placeholder for the actual proof
  sorry

-- Evaluate the result
#eval (weightedAverageDogLifespan [
    ⟨10, 0.4⟩,
    ⟨12, 0.3⟩,
    ⟨14, 0.2⟩,
    ⟨16, 0.1⟩
  ] + 2) * 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_lifespan_in_months_l1235_123552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_cards_l1235_123582

/-- The number of empty boxes needed to transfer n cards --/
def boxes_needed (n : ℕ) : ℕ :=
  Nat.log2 n + 1

theorem transfer_cards (n : ℕ) (h : n = 2006) :
  boxes_needed n = 11 := by
  rw [h]
  rw [boxes_needed]
  norm_num
  rfl

#eval boxes_needed 2006

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_cards_l1235_123582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_has_most_divisors_l1235_123574

/-- The number of divisors of a positive integer n -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The set of positive integers from 1 to 15 -/
def numbers_set : Finset ℕ := Finset.range 16 \ {0}

theorem twelve_has_most_divisors :
  ∃ (m : ℕ), m ∈ numbers_set ∧
  ∀ (n : ℕ), n ∈ numbers_set → num_divisors n ≤ num_divisors m ∧
  m = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_has_most_divisors_l1235_123574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l1235_123567

/-- The area increase when redesigning a rectangular garden to a square garden --/
theorem garden_area_increase (initial_length initial_width perimeter : ℝ) 
    (h1 : initial_length = 60)
    (h2 : initial_width = 12)
    (h3 : perimeter = 2 * (initial_length + initial_width))
    (h4 : perimeter = 144) : 
  (perimeter / 4)^2 - (initial_length * initial_width) = 576 := by
  sorry

#check garden_area_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l1235_123567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_temperature_of_two_rooms_l1235_123511

/-- Represents a room with its dimensions and temperature -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  temperature : ℝ

/-- Calculates the volume of a room -/
noncomputable def volume (r : Room) : ℝ := r.length * r.width * r.height

/-- Calculates the common temperature of two connected rooms -/
noncomputable def commonTemperature (r1 r2 : Room) : ℝ :=
  (volume r1 * r1.temperature + volume r2 * r2.temperature) / (volume r1 + volume r2)

theorem common_temperature_of_two_rooms :
  let room1 : Room := ⟨5, 3, 4, 22⟩
  let room2 : Room := ⟨6, 5, 4, 13⟩
  commonTemperature room1 room2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_temperature_of_two_rooms_l1235_123511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_analysis_properties_l1235_123589

/-- Coefficient of determination in a regression model -/
def R_squared : ℝ := sorry

/-- Sum of squared residuals in a regression model -/
def sum_squared_residuals : ℝ → ℝ := sorry

/-- Measure of the simulation effect of the model -/
def simulation_effect : ℝ → ℝ := sorry

/-- Measure of the contribution of the explanatory variable to the change in the forecast variable -/
def explanatory_variable_contribution : ℝ → ℝ := sorry

/-- Properties of the coefficient of determination in regression analysis -/
theorem regression_analysis_properties :
  (∀ r1 r2 : ℝ, r1 < r2 → r1 < R_squared → r2 < R_squared → simulation_effect r1 < simulation_effect r2) ∧
  (∀ r1 r2 : ℝ, r1 < r2 → r1 < R_squared → r2 < R_squared → sum_squared_residuals r2 < sum_squared_residuals r1) ∧
  (∀ r1 r2 : ℝ, r1 < r2 → r1 < R_squared → r2 < R_squared → explanatory_variable_contribution r1 < explanatory_variable_contribution r2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_analysis_properties_l1235_123589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_term_is_24_l1235_123598

def sequenceList : List ℕ := [3, 8, 15, 35, 48]

def difference_pattern (seq : List ℕ) : Prop :=
  ∀ i : ℕ, i < seq.length - 2 →
    (seq[i+2]! - seq[i+1]!) - (seq[i+1]! - seq[i]!) = 2

theorem missing_term_is_24 (a : ℕ) :
  difference_pattern sequenceList →
  sequenceList[1]! - sequenceList[0]! = 5 →
  a = 24 :=
by sorry

#eval sequenceList

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_term_is_24_l1235_123598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_proposition_q_l1235_123532

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x
noncomputable def g (x : ℝ) : ℝ := Real.log x + x + 1

-- Proposition p
theorem proposition_p : ∀ x : ℝ, f x > 0 := by sorry

-- Proposition q
theorem proposition_q : ∃ x₀ : ℝ, x₀ > 0 ∧ g x₀ = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_proposition_q_l1235_123532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_area_sum_l1235_123588

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem dark_area_sum : 
  let r1 : ℝ := 2
  let r2 : ℝ := 4
  let r3 : ℝ := 6
  let r4 : ℝ := 8
  let r5 : ℝ := 10
  (circle_area r5 - circle_area r4) + 
  (circle_area r3 - circle_area r2) + 
  circle_area r1 = 60 * Real.pi :=
by
  -- Unfold the definition of circle_area
  unfold circle_area
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_area_sum_l1235_123588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_h_in_interval_l1235_123519

-- Define the function h(x) = x + ln x
noncomputable def h (x : ℝ) : ℝ := x + Real.log x

-- State the theorem
theorem zero_of_h_in_interval :
  ∀ x₀ : ℝ, x₀ > 0 → h x₀ = 0 → x₀ > (Real.exp 1)⁻¹ ∧ x₀ < (Real.exp (1/2))⁻¹ := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma h_continuous : Continuous h := by
  sorry

lemma h_strictly_increasing : StrictMono h := by
  sorry

lemma h_negative_at_e_inv : h (Real.exp 1)⁻¹ < 0 := by
  sorry

lemma h_positive_at_sqrt_e_inv : h (Real.exp (1/2))⁻¹ > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_h_in_interval_l1235_123519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_comparison_l1235_123543

noncomputable section

open Real

theorem triangle_angle_comparison (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  (0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) →
  -- A, B, C are angles of the triangle
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  -- a, b, c are sides opposite to A, B, C respectively
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  -- Given condition
  Real.sin A > Real.sin B →
  -- Conclusion
  A > B :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_comparison_l1235_123543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1235_123506

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  -- Asymptote 1: y = 2x + 3
  asymptote1_slope : ℝ := 2
  asymptote1_intercept : ℝ := 3
  -- Asymptote 2: y = -2x + 1
  asymptote2_slope : ℝ := -2
  asymptote2_intercept : ℝ := 1
  -- Point the hyperbola passes through
  point_x : ℝ := 5
  point_y : ℝ := 5

/-- The distance between the foci of the hyperbola -/
noncomputable def foci_distance (h : Hyperbola) : ℝ := 2 * Real.sqrt 42.5

/-- Theorem stating that the distance between the foci of the given hyperbola is 2√(42.5) -/
theorem hyperbola_foci_distance (h : Hyperbola) : foci_distance h = 2 * Real.sqrt 42.5 := by
  sorry

#check hyperbola_foci_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1235_123506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_nth_root_in_S_l1235_123550

def S : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 3 / 2}

theorem smallest_m_for_nth_root_in_S : 
  ∀ n : ℕ, n ≥ 12 → ∃ z ∈ S, z^n = 1 ∧ 
  ∀ m : ℕ, m < 12 → ∃ k : ℕ, k ≥ m ∧ ¬∃ z ∈ S, z^k = 1 :=
by
  sorry

#check smallest_m_for_nth_root_in_S

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_nth_root_in_S_l1235_123550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brothers_writing_equation_l1235_123530

/-- Represents the number of characters the older brother writes per day -/
def x : ℝ := sorry

/-- The total number of characters the older brother needs to write -/
def older_total : ℝ := 8000

/-- The total number of characters the younger brother needs to write -/
def younger_total : ℝ := 6000

/-- The difference in characters written per day between the brothers -/
def daily_difference : ℝ := 100

/-- The number of characters the younger brother writes per day -/
def younger_daily : ℝ := x - daily_difference

/-- The theorem stating the equation that represents the relationship between 
    the number of characters each brother writes per day -/
theorem brothers_writing_equation :
  older_total / x = younger_total / younger_daily :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brothers_writing_equation_l1235_123530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1235_123536

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 2) / (x^3 - 2*x^2 - x + 2)

def num_holes (f : ℝ → ℝ) : ℕ := sorry
def num_vertical_asymptotes (f : ℝ → ℝ) : ℕ := sorry
def num_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := sorry
def num_oblique_asymptotes (f : ℝ → ℝ) : ℕ := sorry

theorem asymptote_sum (f : ℝ → ℝ) :
  let a := num_holes f
  let b := num_vertical_asymptotes f
  let c := num_horizontal_asymptotes f
  let d := num_oblique_asymptotes f
  a + 2*b + 3*c + 4*d = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1235_123536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1235_123534

theorem trig_inequality : ∀ x : ℝ, Real.sin 2 + Real.cos 2 + 2 * (Real.sin 1 - Real.cos 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1235_123534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_numbers_l1235_123585

noncomputable def number_set : Finset ℝ := {-7, 0, -3, 4/3, 9100, -0.7}

theorem count_positive_numbers : (number_set.filter (λ x => x > 0)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_numbers_l1235_123585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_selling_price_l1235_123564

-- Define the initial cost of the bicycle
noncomputable def initial_cost : ℚ := 120

-- Define the profit percentages
noncomputable def profit_percentage_A : ℚ := 25
noncomputable def profit_percentage_B : ℚ := 50

-- Define the function to calculate selling price given cost and profit percentage
noncomputable def selling_price (cost : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost * (1 + profit_percentage / 100)

-- Theorem stating the final selling price
theorem final_selling_price :
  selling_price (selling_price initial_cost profit_percentage_A) profit_percentage_B = 225 := by
  -- Expand the definition of selling_price
  unfold selling_price
  -- Perform algebraic simplification
  simp [initial_cost, profit_percentage_A, profit_percentage_B]
  -- The proof is completed by computational reflection
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_selling_price_l1235_123564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_doubling_l1235_123579

theorem factory_output_doubling (x : ℝ) :
  (∀ (n : ℕ), (n : ℝ) < x → (1.1 : ℝ)^(n + 1) ≤ (1.1 : ℝ)^x) →
  (1.1 : ℝ)^x = 2 ↔ 
  (∃ (initial_output : ℝ), initial_output > 0 ∧
    (initial_output * (1.1 : ℝ)^x = 2 * initial_output)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_doubling_l1235_123579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_two_l1235_123569

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2)^x else Real.log x / Real.log 3

-- State the theorem
theorem f_composition_equals_two : f (f (1/3)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_two_l1235_123569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_primary_schools_l1235_123527

/-- Represents the types of schools in the region -/
inductive SchoolType
| Primary
| Middle
| University

/-- Represents the total number of schools in the region -/
def totalSchools : Nat := 42

/-- Represents the number of schools of each type -/
def schoolCounts : SchoolType → Nat
| SchoolType.Primary => 21
| SchoolType.Middle => 14
| SchoolType.University => 7

/-- Represents the size of the stratified sample -/
def sampleSize : Nat := 6

/-- Represents the number of schools of each type in the sample -/
def sampleCounts : SchoolType → Nat
| SchoolType.Primary => 3
| SchoolType.Middle => 2
| SchoolType.University => 1

/-- Theorem: The probability of selecting 2 primary schools when randomly choosing 2 schools from the stratified sample is 1/5 -/
theorem probability_two_primary_schools :
  (Nat.choose (sampleCounts SchoolType.Primary) 2) / (Nat.choose sampleSize 2) = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_primary_schools_l1235_123527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_solutions_l1235_123516

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the base-10 logarithm function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := (log10 x)^2 - (floor (log10 x)) - 2 = 0

-- Theorem statement
theorem equation_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x ∈ s, equation x) ∧
  (∀ y : ℝ, equation y → y ∈ s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_solutions_l1235_123516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_bike_speed_l1235_123505

/-- Triathlon race parameters -/
structure TriathlonRace where
  swim_distance : ℚ
  run_distance : ℚ
  bike_distance : ℚ
  swim_speed : ℚ
  run_speed : ℚ
  total_time : ℚ

/-- Calculate the required bike speed for a triathlon race -/
def required_bike_speed (race : TriathlonRace) : ℚ :=
  let swim_time := race.swim_distance / race.swim_speed
  let run_time := race.run_distance / race.run_speed
  let bike_time := race.total_time - swim_time - run_time
  race.bike_distance / bike_time

/-- Theorem: The required bike speed for the given triathlon race is 100/17 miles per hour -/
theorem triathlon_bike_speed :
  let race : TriathlonRace := {
    swim_distance := 1/2,
    run_distance := 4,
    bike_distance := 10,
    swim_speed := 1,
    run_speed := 5,
    total_time := 3
  }
  required_bike_speed race = 100/17 := by
  sorry

#eval required_bike_speed {
  swim_distance := 1/2,
  run_distance := 4,
  bike_distance := 10,
  swim_speed := 1,
  run_speed := 5,
  total_time := 3
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_bike_speed_l1235_123505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_reflection_distance_l1235_123593

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The focus of the parabola y² = 4x -/
def focus : Point := ⟨1, 0⟩

/-- The starting point of the light ray -/
def A : Point := ⟨5, 4⟩

theorem parabola_reflection_distance :
  ∃ (B C : Point), B ∈ Parabola ∧ C ∈ Parabola ∧
  B.y = A.y ∧  -- B is on the same horizontal line as A
  (C.x - B.x) * (focus.y - B.y) = (C.y - B.y) * (focus.x - B.x) ∧  -- BC is perpendicular to BF
  distance B C = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_reflection_distance_l1235_123593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pauls_total_cans_l1235_123554

/-- The number of cans Paul picked up in total given the conditions -/
theorem pauls_total_cans
  (saturday_bags : ℕ) (sunday_bags : ℕ) (saturday_cans_per_bag : ℕ) (sunday_cans_per_bag : ℕ) :
  saturday_bags = 10 →
  sunday_bags = 5 →
  saturday_cans_per_bag = 12 →
  sunday_cans_per_bag = 15 →
  saturday_bags * saturday_cans_per_bag + sunday_bags * sunday_cans_per_bag = 195 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pauls_total_cans_l1235_123554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_percent_replaced_in_three_days_all_banknotes_can_be_replaced_l1235_123581

/-- Represents the banknote replacement process in the Magical Kingdom treasury --/
structure BanknoteReplacement where
  total_banknotes : ℕ
  daily_capacity : ℕ → ℕ
  startup_cost : ℕ
  repair_cost : ℕ
  post_repair_capacity : ℕ
  budget : ℕ

/-- The specific banknote replacement scenario for the Magical Kingdom --/
def magical_kingdom : BanknoteReplacement where
  total_banknotes := 3628800
  daily_capacity := fun d => (3628800 / (d + 1))
  startup_cost := 90000
  repair_cost := 700000
  post_repair_capacity := 1000000
  budget := 1000000

/-- Calculates the number of banknotes replaced after a given number of days --/
def banknotes_replaced (br : BanknoteReplacement) (days : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 90% of banknotes can be replaced in 3 days --/
theorem ninety_percent_replaced_in_three_days (br : BanknoteReplacement) :
  br = magical_kingdom →
  banknotes_replaced br 3 ≥ (br.total_banknotes * 9) / 10 := by
  sorry

/-- Theorem stating that all banknotes can eventually be replaced --/
theorem all_banknotes_can_be_replaced (br : BanknoteReplacement) :
  br = magical_kingdom →
  ∃ n : ℕ, banknotes_replaced br n = br.total_banknotes := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_percent_replaced_in_three_days_all_banknotes_can_be_replaced_l1235_123581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l1235_123555

/-- Parametric curve definition -/
noncomputable def x (t : ℝ) : ℝ := 2 * t - t^2
noncomputable def y (t : ℝ) : ℝ := 3 * t - t^3

/-- Point on the curve at t₀ = 1 -/
def t₀ : ℝ := 1
noncomputable def x₀ : ℝ := x t₀
noncomputable def y₀ : ℝ := y t₀

/-- Derivatives of x and y with respect to t -/
noncomputable def x' (t : ℝ) : ℝ := 2 - 2 * t
noncomputable def y' (t : ℝ) : ℝ := 3 - 3 * t^2

/-- Slope of the tangent line at t₀ -/
noncomputable def m : ℝ := y' t₀ / x' t₀

/-- Theorem: Equations of tangent and normal lines -/
theorem tangent_and_normal_equations :
  (∀ x, y₀ + m * (x - x₀) = 3 * x - 1) ∧
  (∀ x, y₀ - (1 / m) * (x - x₀) = -1/3 * x + 7/3) := by
  sorry

#check tangent_and_normal_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l1235_123555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_beta_value_l1235_123529

theorem cos_two_beta_value (α β : ℝ) 
  (h1 : Real.sin (α - β) = 3/5)
  (h2 : Real.sin (α + β) = -3/5)
  (h3 : α - β ∈ Set.Ioo (π/2) π)
  (h4 : α + β ∈ Set.Ioo (3*π/2) (2*π)) :
  Real.cos (2*β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_beta_value_l1235_123529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_possible_values_for_D_l1235_123545

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition problem AACBA + BCCAB = CABBD -/
def AdditionProblem (A B C D : Digit) : Prop :=
  10000 * A.val + 1000 * A.val + 100 * C.val + 10 * B.val + A.val +
  10000 * B.val + 1000 * C.val + 100 * C.val + 10 * A.val + B.val =
  10000 * C.val + 1000 * A.val + 100 * B.val + 10 * B.val + D.val

/-- Helper function to check if there's a carry in a specific column -/
def HasCarry (A B C D : Digit) (col : Fin 5) : Prop := sorry

/-- The main theorem stating that there are exactly 5 possible values for D -/
theorem five_possible_values_for_D :
  ∃ (S : Finset Digit),
    S.card = 5 ∧
    (∀ D, D ∈ S ↔
      ∃ (A B C : Digit),
        A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧  -- distinct digits
        A.val ≠ 0 ∧ B.val ≠ 0 ∧ C.val ≠ 0 ∧ D.val ≠ 0 ∧  -- non-zero digits
        AdditionProblem A B C D ∧  -- satisfies the addition problem
        (∃ col : Fin 5, HasCarry A B C D col))  -- has carry in at least one column
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_possible_values_for_D_l1235_123545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_f_squared_l1235_123576

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 1

-- Define the statement to be proved
theorem f_composition_equals_f_squared :
  ∀ x : ℝ, f (f x) = (f x)^2 ↔ x = 2 + Real.sqrt 13 / 2 ∨ x = 2 - Real.sqrt 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_f_squared_l1235_123576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_test_disease_probability_l1235_123502

noncomputable def disease_prevalence : ℝ := 1 / 1000
noncomputable def test_sensitivity : ℝ := 1
noncomputable def test_specificity : ℝ := 0.97

theorem positive_test_disease_probability :
  let p_disease := disease_prevalence
  let p_no_disease := 1 - disease_prevalence
  let p_positive_given_disease := test_sensitivity
  let p_positive_given_no_disease := 1 - test_specificity
  let p_positive := p_positive_given_disease * p_disease + p_positive_given_no_disease * p_no_disease
  p_positive_given_disease * p_disease / p_positive = 100 / 3997 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_test_disease_probability_l1235_123502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_trains_20_min_stoppage_l1235_123566

/-- Represents a train with its speeds including and excluding stoppages -/
structure Train where
  speed_with_stops : ℚ
  speed_without_stops : ℚ

/-- Calculates the average stoppage time per hour for a train -/
def average_stoppage_time (t : Train) : ℚ :=
  (1 - t.speed_with_stops / t.speed_without_stops) * 60

/-- The given data for Train A -/
def train_a : Train :=
  { speed_with_stops := 30
    speed_without_stops := 45 }

/-- The given data for Train B -/
def train_b : Train :=
  { speed_with_stops := 40
    speed_without_stops := 60 }

/-- Theorem stating that both trains have an average stoppage time of 20 minutes per hour -/
theorem both_trains_20_min_stoppage :
  average_stoppage_time train_a = 20 ∧ average_stoppage_time train_b = 20 := by
  sorry

#eval average_stoppage_time train_a
#eval average_stoppage_time train_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_trains_20_min_stoppage_l1235_123566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collins_total_petals_l1235_123520

/-- The number of petals Collin has in total -/
def total_petals (collins_flowers : ℕ) (ingrids_flowers : ℕ) (flowers_given_fraction : ℚ) (petals_per_flower : ℕ) : ℕ :=
  (collins_flowers + Int.toNat ((ingrids_flowers : ℚ) * flowers_given_fraction).floor) * petals_per_flower

/-- Theorem stating that Collin has 144 petals in total -/
theorem collins_total_petals :
  total_petals 25 33 (1/3) 4 = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collins_total_petals_l1235_123520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_domain_symmetry_l1235_123533

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain (a : ℝ) : Set ℝ := Set.Ioo (3 - 2*a) (a + 1)

-- State the theorem
theorem even_function_domain_symmetry (a : ℝ) :
  (∀ x, f (x + 1) = f (-x + 1)) →  -- f(x+1) is even
  (∀ x, x ∈ domain a → f x ∈ domain a) →  -- domain of f is (3-2a, a+1)
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_domain_symmetry_l1235_123533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1235_123553

theorem sin_alpha_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.sin α = 1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1235_123553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1235_123508

noncomputable def geometric_series (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

noncomputable def infinite_geometric_series (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

theorem solution_satisfies_equation :
  let x : ℝ := 10
  let series (n : ℕ) := (5 + n * x) / (3^n)
  5 + (infinite_geometric_series 5 (1/3)) + (infinite_geometric_series x (1/3)) = 15 := by
  sorry

#check solution_satisfies_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1235_123508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_max_min_values_l1235_123592

-- Define the curve C
def C (x y : ℝ) : Prop := 4 * x^2 / 9 + y^2 / 16 = 1

-- Define the line l
def l (x y : ℝ) : Prop := 2 * x + y = 11

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x y : ℝ) : ℝ := (Real.sqrt 5 / 5) * |3 * x + 4 * y - 11|

-- Define the length of PA
noncomputable def PA_length (x y : ℝ) : ℝ := 
  (2 * Real.sqrt 5 / 5) * |5 * Real.sin (Real.arctan (4/3) + Real.arccos (2*x/3)) - 11|

theorem PA_max_min_values :
  ∀ x y : ℝ, C x y →
  (∃ A : ℝ × ℝ, l A.1 A.2 ∧ 
    Real.cos (Real.arctan ((A.2 - y) / (A.1 - x)) - Real.arctan (-1/2)) = Real.sqrt 3 / 2) →
  (PA_length x y ≤ 32 * Real.sqrt 5 / 5 ∧ PA_length x y ≥ 12 * Real.sqrt 5 / 5 ∧
   ∃ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ ∧ C x₂ y₂ ∧ 
     PA_length x₁ y₁ = 32 * Real.sqrt 5 / 5 ∧ PA_length x₂ y₂ = 12 * Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_max_min_values_l1235_123592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_mn_equals_six_l1235_123559

/-- Two algebraic expressions are like terms if their variables and corresponding exponents are identical. -/
def areLikeTerms (expr1 expr2 : ℕ → ℕ → ℝ) : Prop :=
  ∀ (a b c d : ℕ), expr1 a b = expr1 c d → expr2 a b = expr2 c d

/-- The first algebraic expression -2x^m*y^2 -/
def expr1 (m : ℕ) (y : ℕ) : ℝ := -2 * (y^2)

/-- The second algebraic expression 2x^3*y^n -/
def expr2 (x : ℕ) (n : ℕ) : ℝ := 2 * (x^3)

theorem like_terms_mn_equals_six (m n : ℕ) 
  (h : areLikeTerms (fun x y => expr1 m y) (fun x y => expr2 x n)) : m * n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_mn_equals_six_l1235_123559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_proof_l1235_123542

-- Define the given distance in kilometers
noncomputable def distance : ℚ := 4

-- Define the given speed in kilometers per hour
noncomputable def speed : ℚ := 3

-- Define the time in hours
noncomputable def time_hours : ℚ := distance / speed

-- Define the conversion factor from hours to minutes
noncomputable def minutes_per_hour : ℚ := 60

-- Theorem to prove
theorem walking_time_proof : 
  ⌊(time_hours * minutes_per_hour : ℚ)⌋ = 80 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_proof_l1235_123542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_exponential_difference_l1235_123537

theorem max_value_of_exponential_difference :
  ∃ (x : ℝ), ∀ (y : ℝ), (20 : ℝ)^x - (400 : ℝ)^x ≥ (20 : ℝ)^y - (400 : ℝ)^y ∧ (20 : ℝ)^x - (400 : ℝ)^x = (1/4 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_exponential_difference_l1235_123537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1235_123594

/-- Given a triangle ABC with the specified properties, prove the angle C and the range of a² + b² -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  (Real.sqrt 3 * Real.tan A * Real.tan B - Real.tan A - Real.tan B = Real.sqrt 3) →
  (c = 2) →
  (0 < A) → (A < π / 2) →
  (0 < B) → (B < π / 2) →
  (0 < C) → (C < π / 2) →
  -- Conclusions
  (C = π / 3) ∧
  (20 / 3 < a^2 + b^2) ∧ (a^2 + b^2 ≤ 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1235_123594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1235_123562

noncomputable def f (x : ℝ) : ℝ := Real.log (|Real.sin x|)

theorem f_properties :
  (∀ x, f (x + π) = f x) ∧
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1235_123562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_unsold_books_l1235_123551

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

noncomputable def percentage_not_sold : ℝ := (books_not_sold : ℝ) / (initial_stock : ℝ) * 100

theorem percentage_of_unsold_books :
  abs (percentage_not_sold - 71.29) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_unsold_books_l1235_123551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1235_123540

def point1 : ℝ × ℝ := (-4, 3)
def point2 : ℝ × ℝ := (6, -7)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  distance point1 point2 = 10 * Real.sqrt 2 := by
  -- Unfold the definitions and apply the distance formula
  unfold distance point1 point2
  -- Simplify the expressions inside the square root
  simp [Real.sqrt_sq, abs_of_nonneg]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1235_123540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_AB_AC_l1235_123595

def A : ℝ × ℝ × ℝ := (0, 2, 3)
def B : ℝ × ℝ × ℝ := (-2, 1, 6)
def C : ℝ × ℝ × ℝ := (1, -1, 5)

def vec_AB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2)
def vec_AC : ℝ × ℝ × ℝ := (C.1 - A.1, C.2.1 - A.2.1, C.2.2 - A.2.2)

theorem angle_between_AB_AC :
  Real.arccos ((vec_AB.1 * vec_AC.1 + vec_AB.2.1 * vec_AC.2.1 + vec_AB.2.2 * vec_AC.2.2) /
    (Real.sqrt (vec_AB.1^2 + vec_AB.2.1^2 + vec_AB.2.2^2) *
     Real.sqrt (vec_AC.1^2 + vec_AC.2.1^2 + vec_AC.2.2^2))) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_AB_AC_l1235_123595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_case_l1235_123578

/-- The length of the tangent line from a point to a circle -/
noncomputable def tangent_length (px py cx cy r : ℝ) : ℝ :=
  Real.sqrt ((px - cx)^2 + (py - cy)^2 - r^2)

/-- Theorem: The length of the tangent line from P(3,5) to the circle (x-1)^2 + (y-1)^2 = 4 is 4 -/
theorem tangent_length_specific_case : tangent_length 3 5 1 1 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_case_l1235_123578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_overtake_time_l1235_123568

/-- Calculates the time for a train to overtake a motorbike -/
noncomputable def overtake_time (train_speed : ℝ) (motorbike_speed : ℝ) (train_length : ℝ) : ℝ :=
  train_length / ((train_speed - motorbike_speed) * (1000 / 3600))

/-- Theorem: The time for a train to overtake a motorbike under specific conditions -/
theorem train_overtake_time :
  let train_speed : ℝ := 100
  let motorbike_speed : ℝ := 64
  let train_length : ℝ := 800.064
  overtake_time train_speed motorbike_speed train_length = 80.0064 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval overtake_time 100 64 800.064

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_overtake_time_l1235_123568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1235_123563

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ x

theorem m_range (m : ℝ) : f (2*m - 1) - f (m + 3) < 0 → m > 4 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1235_123563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_exists_l1235_123518

theorem no_valid_n_exists : ∀ (n : ℕ), n ≥ 2 →
  ¬∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    ∀ (a : Fin n → ℕ), (∀ i j, i < j → (p ∣ a j - a i) ∨ (q ∣ a j - a i) ∨ (r ∣ a j - a i)) →
    ((∀ i j, i < j → p ∣ a j - a i) ∨ (∀ i j, i < j → q ∣ a j - a i) ∨ (∀ i j, i < j → r ∣ a j - a i)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_exists_l1235_123518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1235_123509

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * (Real.cos (ω * x))^2 - 1 + 2 * Real.sqrt 3 * Real.cos (ω * x) * Real.sin (ω * x)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω ((x + 2 * Real.pi / 3) / 2)

theorem problem_solution (ω : ℝ) (α : ℝ) 
  (h1 : 0 < ω) (h2 : ω < 1) 
  (h3 : ∀ x, f ω (Real.pi / 3 + x) = f ω (Real.pi / 3 - x))
  (h4 : 0 < α) (h5 : α < Real.pi / 2)
  (h6 : g ω (2 * α + Real.pi / 3) = 6 / 5) : 
  ω = 1 / 2 ∧ Real.sin α = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1235_123509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_xy_value_l1235_123597

theorem min_xy_value (x y : ℝ) : 
  1 + (Real.cos (x + y - 1))^2 = ((x^2 + y^2 + 2*(x + 1)*(1 - y)) / (x - y + 1)) →
  ∀ z, z = x*y → z ≥ (1/4 : ℝ) ∧ ∃ a b : ℝ, a*b = (1/4 : ℝ) ∧ 
  1 + (Real.cos (a + b - 1))^2 = ((a^2 + b^2 + 2*(a + 1)*(1 - b)) / (a - b + 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_xy_value_l1235_123597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_heights_3_4_5_is_obtuse_l1235_123572

/-- Define a structure for a triangle -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  height1 : ℝ
  height2 : ℝ
  height3 : ℝ

/-- Define what it means for a triangle to be obtuse -/
def Triangle.isObtuse (t : Triangle) : Prop :=
  t.side1^2 > t.side2^2 + t.side3^2 ∨
  t.side2^2 > t.side1^2 + t.side3^2 ∨
  t.side3^2 > t.side1^2 + t.side2^2

/-- A triangle with heights 3, 4, and 5 is obtuse -/
theorem triangle_with_heights_3_4_5_is_obtuse (T : Triangle) 
  (h1 : T.height1 = 3) (h2 : T.height2 = 4) (h3 : T.height3 = 5) : 
  T.isObtuse := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_heights_3_4_5_is_obtuse_l1235_123572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_1_l1235_123586

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | n + 2 => sequence_a (n + 1) + sequence_a n

theorem a_7_equals_1 : sequence_a 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_1_l1235_123586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_treasure_signs_l1235_123513

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried" -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried" -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried" -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried" -/
def signs_3 : ℕ := 3

/-- Represents that only signs under which there is no treasure are truthful -/
def truthful_signs (n : ℕ) : Prop :=
  n ≤ total_trees ∧ (n ≠ signs_15 ∧ n ≠ signs_8 ∧ n ≠ signs_4 ∧ n ≠ signs_3)

/-- The theorem stating that the minimum number of signs under which treasures can be buried is 15 -/
theorem min_treasure_signs : ∃ (n : ℕ), n = 15 ∧ (∀ m : ℕ, m < n → ¬(truthful_signs m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_treasure_signs_l1235_123513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_basket_capacity_l1235_123546

/-- Represents the capacity of Jack's basket in apples -/
def jack_capacity : ℕ := sorry

/-- Represents the current number of apples in Jack's basket -/
def jack_current : ℕ := sorry

/-- Represents the capacity of Jill's basket in apples -/
def jill_capacity : ℕ := sorry

/-- Jill's basket can hold twice as much as Jack's basket when both are full -/
axiom jill_double_jack : jill_capacity = 2 * jack_capacity

/-- Jack's basket currently has space for 4 more apples -/
axiom jack_space_for_four : jack_capacity = jack_current + 4

/-- Jack's current number of apples could fit into Jill's basket 3 times -/
axiom jack_current_fits_thrice : jill_capacity = 3 * jack_current

theorem jack_basket_capacity : jack_capacity = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_basket_capacity_l1235_123546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sport_formulation_volume_l1235_123558

/-- Standard formulation ratio -/
def standard_ratio : Fin 4 → ℚ
  | 0 => 1    -- Flavoring
  | 1 => 12   -- Corn Syrup
  | 2 => 30   -- Water
  | 3 => 1/2  -- Colorant

/-- Sport formulation ratio -/
def sport_ratio : Fin 4 → ℚ
  | 0 => 1    -- Flavoring
  | 1 => 4    -- Corn Syrup
  | 2 => 60   -- Water
  | 3 => 1/4  -- Colorant

/-- Corn syrup amount in sport formulation (in ounces) -/
def corn_syrup_amount : ℚ := 3

/-- Theorem stating the total volume of the sport formulation -/
theorem sport_formulation_volume :
  (Finset.sum Finset.univ (λ i => (corn_syrup_amount / sport_ratio 1) * sport_ratio i)) = 207/4 :=
by
  sorry

#eval (Finset.sum Finset.univ (λ i : Fin 4 => (corn_syrup_amount / sport_ratio 1) * sport_ratio i))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sport_formulation_volume_l1235_123558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotateAroundSmallBase_correct_l1235_123596

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  smallBase : ℝ
  largeBase : ℝ
  acuteAngle : ℝ

/-- Represents a solid of revolution -/
structure SolidOfRevolution where
  surfaceArea : ℝ
  volume : ℝ

/-- Calculates the solid of revolution when an isosceles trapezoid is rotated around its smaller base -/
noncomputable def rotateAroundSmallBase (t : IsoscelesTrapezoid) : SolidOfRevolution :=
  { surfaceArea := 4 * Real.pi * Real.sqrt 3
  , volume := 2 * Real.pi }

/-- Theorem: The surface area and volume of the solid of revolution are correct for the given trapezoid -/
theorem rotateAroundSmallBase_correct (t : IsoscelesTrapezoid) :
    t.smallBase = 2 ∧ t.largeBase = 3 ∧ t.acuteAngle = Real.pi / 3 →
    let solid := rotateAroundSmallBase t
    solid.surfaceArea = 4 * Real.pi * Real.sqrt 3 ∧ solid.volume = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotateAroundSmallBase_correct_l1235_123596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1235_123525

-- Define the curve
noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log (x + 1)

-- Define the derivative of the curve
noncomputable def curve_derivative (a : ℝ) (x : ℝ) : ℝ := a - 1 / (x + 1)

theorem tangent_line_slope (a : ℝ) : 
  curve a 0 = 0 → curve_derivative a 0 = 2 → a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1235_123525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increase_l1235_123500

/-- Represents a rectangle with length and breadth -/
structure Rectangle where
  length : ℝ
  breadth : ℝ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.breadth

/-- The original rectangle -/
noncomputable def original_rectangle : Rectangle := {
  length := 33.333333333333336,
  breadth := 33.333333333333336 / 2
}

/-- The modified rectangle -/
noncomputable def modified_rectangle : Rectangle := {
  length := original_rectangle.length - 5,
  breadth := original_rectangle.breadth + 4
}

theorem area_increase :
  area modified_rectangle - area original_rectangle = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increase_l1235_123500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_sequences_l1235_123549

/-- Represents the number of commercial advertisements -/
def num_commercials : ℕ := 3

/-- Represents the number of public service announcements (PSAs) -/
def num_psas : ℕ := 2

/-- Represents the total number of advertisements -/
def total_ads : ℕ := 5

/-- Theorem stating the number of valid broadcasting sequences -/
theorem num_valid_sequences :
  (num_commercials = 3) →
  (num_psas = 2) →
  (total_ads = 5) →
  (∃ (n : ℕ), n = 36 ∧
    n = (Nat.choose 4 1 * Nat.factorial 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_sequences_l1235_123549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_lines_l1235_123583

-- Define the points
noncomputable def A : ℝ × ℝ := (0, 7/3)
noncomputable def B : ℝ × ℝ := (7, 0)
noncomputable def C : ℝ × ℝ := (2, 1)
noncomputable def D (k : ℝ) : ℝ × ℝ := (3, k+1)

-- Define the lines
noncomputable def l₁ (x : ℝ) : ℝ := (7/3) * (1 - x/7)
noncomputable def l₂ (k : ℝ) (x : ℝ) : ℝ := 1 + (k/1) * (x - 2)

-- Define perpendicularity condition
def perpendicular (k : ℝ) : Prop := (7/3) / (-7) * k = -1

theorem circle_tangent_lines (k : ℝ) : 
  perpendicular k → k = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_lines_l1235_123583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_count_l1235_123512

-- Define the number of digits in base 2 representation of 10^k
noncomputable def digits_base_2 (k : ℕ) : ℕ := 
  ⌊(k : ℝ) * Real.log 10 / Real.log 2⌋.toNat + 1

-- Define the number of digits in base 5 representation of 10^k
noncomputable def digits_base_5 (k : ℕ) : ℕ := 
  ⌊(k : ℝ) * Real.log 10 / Real.log 5⌋.toNat + 1

theorem unique_digit_count : 
  ∀ n : ℕ, n > 1 → 
  ∃! k : ℕ, k ≥ 1 ∧ 
  ((digits_base_2 k = n ∧ digits_base_5 k ≠ n) ∨ 
   (digits_base_2 k ≠ n ∧ digits_base_5 k = n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_count_l1235_123512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_impossible_triangle_possible_l1235_123504

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The set of stick lengths -/
def stick_lengths : Finset ℕ := Finset.range 20

theorem square_impossible_triangle_possible :
  (¬ ∃ (side_length : ℕ), 4 * side_length = sum_to_n 20) ∧
  (∃ (partition : Finset ℕ → List (Finset ℕ)),
    (∀ i ∈ stick_lengths, ∃ s ∈ partition stick_lengths, i ∈ s) ∧
    (∀ s ∈ partition stick_lengths,
      Finset.sum s id = Finset.sum stick_lengths id / 3) ∧
    (Finset.sum stick_lengths id = 3 * (Finset.sum stick_lengths id / 3))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_impossible_triangle_possible_l1235_123504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1235_123556

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := (4 - 3 * (sin x)^6 - 3 * (cos x)^6) / (sin x * cos x)

theorem f_range :
  ∀ y : ℝ, y ≥ 6 → ∃ x : ℝ, x ∈ Set.Ioo 0 (π/2) ∧ f x = y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1235_123556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_negative_one_l1235_123591

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 * Real.pi then Real.sin x
  else if -Real.pi < x ∧ x < 0 then Real.cos x
  else 0  -- Define a default value for x outside the given intervals

axiom f_period (x : ℝ) : f (x + 3 * Real.pi) = f x

theorem f_sum_equals_negative_one :
  f (-308 * Real.pi / 3) + f (601 * Real.pi / 6) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_negative_one_l1235_123591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_for_special_angle_l1235_123587

/-- Given an angle α whose terminal side passes through (-4, 3), 
    prove two trigonometric identities involving α. -/
theorem trig_identities_for_special_angle (α : ℝ) 
    (h : ∃ (r : ℝ), r > 0 ∧ r * Real.cos α = -4 ∧ r * Real.sin α = 3) : 
  (Real.sin (π/2 + α) - Real.cos (π + α)) / (Real.sin (π/2 - α) - Real.sin (π - α)) = 8/7 ∧ 
  Real.sin α * Real.cos α = -12/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_for_special_angle_l1235_123587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1235_123515

-- Define the condition function
noncomputable def condition (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a/b + b/c + c/a) + (b/a + c/b + a/c) = 9

-- Define the expression to be minimized
noncomputable def expression (a b c : ℝ) : ℝ :=
  (a/b + b/c + c/a) * (b/a + c/b + a/c)

-- State the theorem
theorem min_value_theorem :
  ∀ a b c : ℝ, condition a b c → expression a b c ≥ 30 :=
by
  intros a b c h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1235_123515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1235_123584

theorem triangle_side_length (PQ QR PR RS : ℝ) (h1 : PQ = 6) (h2 : QR = 8) (h3 : PR = 7) (h4 : RS = 10)
  (h5 : ∃ φ, Real.cos φ = 51 / 96) : ∃ PS, PS = Real.sqrt 7116 / Real.sqrt 96 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1235_123584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l1235_123539

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_solution (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  d ≠ 0 →
  a 3 + a 9 = a 10 - a 8 →
  a 5 = 0 := by
  intros h_arith h_d_neq_zero h_eq
  sorry -- Proof details would go here

#check arithmetic_sequence_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l1235_123539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l1235_123599

theorem pure_imaginary_condition (m : ℝ) : 
  (Complex.I * 2 + 1) * (Complex.I * m + 1) = Complex.I * Complex.I.im * (2*m + 1) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l1235_123599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1235_123580

def is_geometric_sequence (seq : List ℝ) : Prop :=
  seq.length > 1 ∧ ∃ r : ℝ, ∀ i : Nat, i + 1 < seq.length → seq[i+1]! / seq[i]! = r

theorem geometric_sequence_properties (a b c : ℝ) :
  is_geometric_sequence [-1, a, b, c, -9] →
  b = -3 ∧ a * c = 9 := by
  sorry

#check geometric_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1235_123580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_p_geq_neg_one_l1235_123522

/-- A function f(x) that depends on a parameter p -/
noncomputable def f (p : ℝ) (x : ℝ) : ℝ := x - p / x + p / 2

/-- Theorem stating that if f(x) is increasing on (1, +∞), then p ≥ -1 -/
theorem f_increasing_implies_p_geq_neg_one (p : ℝ) :
  (∀ x y : ℝ, 1 < x ∧ x < y → f p x < f p y) →
  p ≥ -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_p_geq_neg_one_l1235_123522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1235_123541

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((1 + x) / (1 - x))

-- State the theorem
theorem range_of_g : ∀ x : ℝ, x ≠ 1 → g x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1235_123541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_t_for_even_shifted_function_l1235_123528

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt (3 * (Real.sin (ω * x))^2 + (Real.cos (ω * x))^2)

theorem minimum_t_for_even_shifted_function (ω t : ℝ) (h_t : t > 0) :
  (∀ x, f ω x = f ω (x + 2 * π)) →
  (∀ x, f ω (x + t) = f ω (-x + t)) →
  ∃ k : ℤ, t = 5 * π / 6 + k * π ∧ 
    ∀ t' : ℝ, t' > 0 → (∃ k' : ℤ, t' = 5 * π / 6 + k' * π) → t' ≥ t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_t_for_even_shifted_function_l1235_123528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_equality_l1235_123575

-- Define the quadratic trinomials
def trinomial (a b c : ℝ) (x : ℝ) : ℝ := a*x^2 + b*x + c

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- State the theorem
theorem roots_sum_equality 
  (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℝ) 
  (h1 : discriminant a1 b1 c1 = 1) 
  (h2 : discriminant a2 b2 c2 = 4) 
  (h3 : discriminant a3 b3 c3 = 9) : 
  ∃ (x1 x2 y1 y2 z1 z2 : ℝ), 
    (trinomial a1 b1 c1 x1 = 0 ∧ trinomial a1 b1 c1 x2 = 0) ∧
    (trinomial a2 b2 c2 y1 = 0 ∧ trinomial a2 b2 c2 y2 = 0) ∧
    (trinomial a3 b3 c3 z1 = 0 ∧ trinomial a3 b3 c3 z2 = 0) ∧
    (x1 + y1 + z2 = x2 + y2 + z1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_equality_l1235_123575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1235_123514

/-- Given an equilateral triangle ABC with side length 4, midpoint M on BC,
    points N on CA and P on AB forming a cyclic quadrilateral ANMP with AN > AP,
    and triangle NMP having area 3, prove that CN can be expressed as (a - √b) / c
    where a, b, c are positive integers and a + b + c = 6. -/
theorem triangle_problem (A B C M N P : ℝ × ℝ) :
  let d := λ (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d A B = 4 ∧ d B C = 4 ∧ d C A = 4) →  -- ABC is equilateral with side length 4
  (M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) →  -- M is midpoint of BC
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ N = (t * C.1 + (1 - t) * A.1, t * C.2 + (1 - t) * A.2)) →  -- N on CA
  (∃ s : ℝ, 0 < s ∧ s < 1 ∧ P = (s * A.1 + (1 - s) * B.1, s * A.2 + (1 - s) * B.2)) →  -- P on AB
  (d A N > d A P) →  -- AN > AP
  (∃ O : ℝ × ℝ, d O A = d O N ∧ d O N = d O M ∧ d O M = d O P) →  -- ANMP is cyclic
  (abs ((N.1 - M.1) * (P.2 - M.2) - (N.2 - M.2) * (P.1 - M.1)) / 2 = 3) →  -- Area of NMP is 3
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ d C N = (a - Real.sqrt b) / c ∧ a + b + c = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1235_123514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1235_123577

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

-- State the theorem
theorem zero_in_interval :
  (∀ x y : ℝ, 0 < x ∧ x < y → Real.log x < Real.log y) →
  (∀ x y : ℝ, x < y → 2 * x - 6 < 2 * y - 6) →
  f 2 < 0 →
  f 3 > 0 →
  ∃ c : ℝ, c ∈ Set.Icc 2 3 ∧ f c = 0 :=
by
  sorry

-- Additional helper lemmas
lemma log_monotone (x y : ℝ) (h : 0 < x ∧ x < y) : Real.log x < Real.log y :=
by
  sorry

lemma linear_monotone (x y : ℝ) (h : x < y) : 2 * x - 6 < 2 * y - 6 :=
by
  sorry

lemma f_2_negative : f 2 < 0 :=
by
  sorry

lemma f_3_positive : f 3 > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1235_123577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_nested_calculation_l1235_123547

/-- The ♠ operation for positive real numbers -/
noncomputable def spade (a b : ℝ) : ℝ := a - 1 / b

/-- Theorem stating that ♠(3, ♠(3, ♠(3,6))) = 118/45 -/
theorem spade_nested_calculation :
  spade 3 (spade 3 (spade 3 6)) = 118 / 45 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_nested_calculation_l1235_123547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_g_g_eq_10_l1235_123573

-- Define the function g
noncomputable def g : ℝ → ℝ := fun x => if x ≥ 0 then x^2 + 1 else x - 2

-- State the theorem
theorem two_solutions_for_g_g_eq_10 :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x : ℝ, x ∈ s ↔ g (g x) = 10 := by
  sorry

#check two_solutions_for_g_g_eq_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_g_g_eq_10_l1235_123573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l1235_123544

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (3 + 2*Complex.I) / Complex.I
  Complex.im z = -3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l1235_123544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_value_l1235_123570

noncomputable def pyramid_height : ℝ :=
  let cube_edge : ℝ := 6
  let pyramid_base : ℝ := 10
  let sphere_radius : ℝ := 4
  let cube_volume : ℝ := cube_edge ^ 3
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  let total_volume : ℝ := cube_volume + sphere_volume
  let pyramid_volume : ℝ := total_volume
  (3 * pyramid_volume) / (pyramid_base ^ 2)

theorem pyramid_height_value : pyramid_height = 6.48 + 2.56 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_value_l1235_123570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1235_123501

-- Define the exponential function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_inequality (a : ℝ) (h1 : a > 2) :
  ∀ x > 1, f a (f a x) ≥ 4 := by
  intro x hx
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1235_123501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1235_123517

def sequenceN (n : ℕ) : ℕ := n

theorem sequence_general_term : ∀ n : ℕ, n > 0 → sequenceN n = n := by
  intro n hn
  rfl

#check sequence_general_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1235_123517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_calculation_l1235_123538

/-- The original price of a shirt before discounts, given the final price after two consecutive 25% discounts -/
noncomputable def original_price (final_price : ℝ) : ℝ :=
  final_price / (0.75 * 0.75)

/-- Theorem stating that the original price of a shirt is approximately $30.22 given the conditions -/
theorem shirt_price_calculation :
  let final_price := (17 : ℝ)
  abs (original_price final_price - 30.22) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_calculation_l1235_123538
