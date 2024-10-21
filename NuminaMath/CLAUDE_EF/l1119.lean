import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1119_111927

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := 180

/-- The speed of the train in km/h -/
noncomputable def train_speed_kmh : ℝ := 54

/-- Conversion factor from km/h to m/s -/
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

/-- The speed of the train in m/s -/
noncomputable def train_speed_ms : ℝ := train_speed_kmh * kmh_to_ms

/-- The time it takes for the train to cross a stationary point -/
noncomputable def crossing_time : ℝ := train_length / train_speed_ms

theorem train_crossing_time :
  crossing_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1119_111927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1119_111902

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b * k = a

def function_property (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, is_divisible ((m^2 + n)^2) ((f m)^2 + f n)

theorem function_identity (f : ℕ → ℕ) (h : function_property f) :
  ∀ n : ℕ, f n = n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1119_111902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_speed_at_max_power_l1119_111937

/-- The speed of a sailboat when wind power is maximized -/
theorem sailboat_speed_at_max_power
  (C S ρ v₀ : ℝ)
  (hC : C > 0)
  (hS : S > 0)
  (hρ : ρ > 0)
  (hv₀ : v₀ > 0) :
  ∃ v : ℝ,
    v = v₀ / 3 ∧
    ∀ u : ℝ,
      (fun x => (C * S * ρ * (v₀ - x)^2) / 2 * x) v ≥
      (fun x => (C * S * ρ * (v₀ - x)^2) / 2 * x) u :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_speed_at_max_power_l1119_111937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamb_downhill_time_l1119_111987

/-- Represents a road with three sections -/
structure Road where
  section1 : ℝ
  section2 : ℝ
  section3 : ℝ

/-- Represents the speed of a lamb on each section of the road -/
structure LambSpeed where
  speed1 : ℝ
  speed2 : ℝ
  speed3 : ℝ

/-- Calculates the time taken by the lamb on each section of the road -/
noncomputable def timeTaken (r : Road) (s : LambSpeed) : ℝ × ℝ × ℝ :=
  (r.section1 / s.speed1, r.section2 / s.speed2, r.section3 / s.speed3)

theorem lamb_downhill_time 
  (r : Road) 
  (s : LambSpeed) 
  (h1 : r.section1 / r.section2 = 1 / 2 ∧ r.section2 / r.section3 = 2 / 3) 
  (h2 : s.speed1 / s.speed2 = 3 / 4 ∧ s.speed2 / s.speed3 = 4 / 5) 
  (h3 : (timeTaken r s).1 + (timeTaken r s).2.1 + (timeTaken r s).2.2 = 86) : 
  (timeTaken r s).2.2 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamb_downhill_time_l1119_111987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_interval_inequality_condition_l1119_111925

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

-- Part 1: Extremum in the interval (a, a + 1/2)
theorem extremum_interval (a : ℝ) (ha : a > 0) :
  (∃ x ∈ Set.Ioo a (a + 1/2), ∀ y ∈ Set.Ioo a (a + 1/2), f x ≥ f y) →
  (1/2 < a ∧ a < 1) := by
  sorry

-- Part 2: Inequality condition for x ≥ 1
theorem inequality_condition (k : ℝ) :
  (∀ x ≥ 1, f x ≥ k / (x + 1)) →
  k ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_interval_inequality_condition_l1119_111925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_sixty_degree_angle_l1119_111960

theorem point_on_sixty_degree_angle (a : ℝ) : 
  (4, a) ∈ {p : ℝ × ℝ | p.1 > 0 ∧ p.2 / p.1 = Real.sqrt 3} → a = 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_sixty_degree_angle_l1119_111960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_point_l1119_111998

-- Define the circle and points
variable (S : Set (ℝ × ℝ)) -- Circle S
variable (A B C D J : ℝ × ℝ) -- Points A, B, C, D, J

-- Define the conditions
def is_circle (S : Set (ℝ × ℝ)) : Prop := sorry
def on_circle (p : ℝ × ℝ) (S : Set (ℝ × ℝ)) : Prop := sorry
def is_chord (p q : ℝ × ℝ) (S : Set (ℝ × ℝ)) : Prop := sorry
def on_line_segment (p q r : ℝ × ℝ) : Prop := sorry
def is_midpoint (m p q : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem unique_intersection_point 
  (h_circle : is_circle S)
  (h_AB : is_chord A B S)
  (h_CD : is_chord C D S)
  (h_J : on_line_segment C J D) :
  ∃! X, on_circle X S ∧ 
        (∃ E F, on_line_segment C E D ∧ 
                on_line_segment C F D ∧
                on_line_segment A E X ∧
                on_line_segment B F X ∧
                is_midpoint J E F) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_point_l1119_111998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_location_l1119_111910

/-- The line defined by the equation x - 2y + 6 = 0 -/
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

/-- The region defined by the inequality x - 2y + 6 > 0 -/
def region (x y : ℝ) : Prop := x - 2*y + 6 > 0

/-- Theorem stating that the region is on the lower right side of the line -/
theorem region_location :
  ∀ (x y x' y' : ℝ), region x y → line x' y' → (x > x' ∨ y < y') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_location_l1119_111910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_max_balloons_l1119_111963

/-- Represents the price of a balloon in cents -/
def regular_price : ℕ := 200

/-- Represents Orvin's total money in cents -/
def total_money : ℕ := 40 * regular_price

/-- Represents the price of a pair of balloons under the promotion in cents -/
def promo_pair_price : ℕ := regular_price + regular_price / 2

/-- Represents the maximum number of balloons Orvin can buy -/
def max_balloons : ℕ := 
  let pairs := total_money / promo_pair_price
  let remaining_money := total_money % promo_pair_price
  2 * pairs + (if remaining_money ≥ regular_price then 1 else 0)

theorem orvin_max_balloons : max_balloons = 53 := by
  sorry

#eval max_balloons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_max_balloons_l1119_111963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_successful_trials_l1119_111923

/-- A trial where two dice are thrown -/
structure DiceTrial where
  success : Bool

/-- An experiment consisting of 10 dice trials -/
def Experiment := Fin 10 → DiceTrial

/-- The probability of a successful trial -/
noncomputable def success_probability : ℝ := 5/9

/-- The expected value of successful trials in 10 experiments -/
noncomputable def expected_value (e : Experiment) : ℝ := 
  10 * success_probability

theorem expected_value_of_successful_trials : 
  ∀ (e : Experiment), expected_value e = 50/9 := by
  intro e
  unfold expected_value
  unfold success_probability
  norm_num

#check expected_value_of_successful_trials

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_successful_trials_l1119_111923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_complements_nonempty_l1119_111978

def A : Set ℝ := {x | x > -2}
def B : Set ℝ := {x | x^2 + 2*x - 15 ≥ 0}

theorem intersection_of_complements_nonempty :
  (Set.univ \ A) ∩ (Set.univ \ B) ≠ ∅ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_complements_nonempty_l1119_111978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_numbers_exist_l1119_111976

/-- A function that checks if a natural number uses only the digits in a given set --/
def usesOnlyDigits (n : ℕ) (digits : Finset ℕ) : Prop :=
  ∀ d, d ∈ digits → d < 10 ∧ d ≠ 0

/-- A function that checks if two natural numbers use the same set of digits --/
def haveSameDigits (a b : ℕ) : Prop :=
  ∃ digits : Finset ℕ, usesOnlyDigits a digits ∧ usesOnlyDigits b digits

/-- A function that returns the number of digits in a natural number --/
def digitCount : ℕ → ℕ
  | 0 => 1
  | n => if n < 10 then 1 else 1 + digitCount (n / 10)

theorem similar_numbers_exist : ∃ a b c : ℕ,
  digitCount a = 1995 ∧
  digitCount b = 1995 ∧
  digitCount c = 1995 ∧
  haveSameDigits a b ∧
  haveSameDigits b c ∧
  haveSameDigits a c ∧
  a + b = c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_numbers_exist_l1119_111976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_27_eq_0_l1119_111916

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := x^(1/3) + (Real.log x) / (Real.log (1/3))

-- State the theorem
theorem f_of_27_eq_0 : f 27 = 0 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [Real.rpow_def, Real.log_div]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_27_eq_0_l1119_111916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inverse_intersection_l1119_111997

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

noncomputable def f_inv (a : ℝ) (x : ℝ) : ℝ := a ^ x

theorem log_inverse_intersection (a : ℝ) (h : a > 1) :
  ∃! x : ℝ, x > 0 ∧ f a x = f_inv a x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inverse_intersection_l1119_111997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_digit_is_three_l1119_111945

/-- Represents a digit in the number --/
def Digit := Fin 10

/-- Represents the initial number as a list of digits --/
def InitialNumber : List Digit := List.replicate 100 ⟨9, by norm_num⟩

/-- Represents the allowed operations on the number --/
inductive Operation
  | increaseDecrease (pos : Nat) -- Increase digit at pos by 1, decrease neighbors by 1
  | subtractAdd (pos : Nat) -- Subtract 1 from digit at pos, add 3 to next digit
  | decreaseBySeven (pos : Nat) -- Decrease digit at pos by 7

/-- Applies an operation to a list of digits --/
def applyOperation (digits : List Digit) (op : Operation) : List Digit :=
  sorry -- Implementation details omitted

/-- Checks if a list of digits represents a single-digit number --/
def isSingleDigit (digits : List Digit) : Bool :=
  digits.length = 1

/-- Theorem: The final single-digit number after applying operations is 3 --/
theorem final_digit_is_three :
  ∃ (operations : List Operation),
    let finalDigits := operations.foldl applyOperation InitialNumber
    isSingleDigit finalDigits ∧ finalDigits = [⟨3, by norm_num⟩] :=
  sorry

#eval InitialNumber

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_digit_is_three_l1119_111945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l1119_111913

theorem unique_solution_exponential_equation :
  ∃! (x y : ℝ), (4 : ℝ)^(x^2 + y) + (4 : ℝ)^(x + y^2) = 2 ∧ x = -1/2 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l1119_111913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_work_contradiction_l1119_111915

theorem johns_work_contradiction :
  ¬ ∃ (days_present : ℕ),
    let days_absent := 60 - days_present
    (7 : ℚ) * days_present + (3 : ℚ) * days_absent = 170 ∧
    days_present + days_absent = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_work_contradiction_l1119_111915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_problem_l1119_111991

theorem fraction_equality_problem :
  ∃ a : ℕ, (2018 - a : ℚ) / (2011 - a) = (2054 - a : ℚ) / (2019 - a) ↔ a = 2009 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_problem_l1119_111991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_axes_product_l1119_111943

-- Define the semi-axes lengths a and b
noncomputable def a : ℝ := Real.sqrt 19.5
noncomputable def b : ℝ := Real.sqrt 44.5

-- Define the ellipse and hyperbola equations
def ellipse_equation (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def hyperbola_equation (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- State the theorem
theorem semi_axes_product :
  (∃ (y : ℝ), ellipse_equation 0 y ∧ y = 5) ∧
  (∃ (x : ℝ), hyperbola_equation x 0 ∧ x = 8) →
  abs (a * b) = Real.sqrt 867.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_axes_product_l1119_111943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1119_111970

/-- The constant term in the expansion of (x + 1/x^2)^6 is 15 -/
theorem constant_term_binomial_expansion :
  (Finset.range 7).sum (fun k => (Nat.choose 6 k) * (1 : ℚ)^k * (1 : ℚ)^(6-3*k)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1119_111970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_removed_volume_l1119_111972

/-- The side length of the cube -/
noncomputable def cube_side_length : ℝ := 2

/-- The side length of the equilateral triangle formed by slicing a corner -/
noncomputable def triangle_side_length : ℝ := 2 / 3

/-- The area of the equilateral triangle formed by slicing a corner -/
noncomputable def triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side_length^2

/-- The height of the equilateral triangle formed by slicing a corner -/
noncomputable def triangle_height : ℝ := (Real.sqrt 3 / 3) * triangle_side_length

/-- The height of the tetrahedron formed by slicing a corner -/
noncomputable def tetrahedron_height : ℝ := cube_side_length - triangle_height

/-- The volume of one tetrahedron formed by slicing a corner -/
noncomputable def tetrahedron_volume : ℝ := (1 / 3) * triangle_area * tetrahedron_height

/-- The number of corners in a cube -/
def num_corners : ℕ := 8

/-- The theorem stating the total volume of removed tetrahedra -/
theorem total_removed_volume :
  (num_corners : ℝ) * tetrahedron_volume = (48 * Real.sqrt 3 - 16) / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_removed_volume_l1119_111972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_inverse_sqrt_l1119_111901

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt x

-- State the theorem
theorem derivative_of_inverse_sqrt (x : ℝ) (h : x > 0) :
  deriv f x = -1 / (2 * x^(3/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_inverse_sqrt_l1119_111901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l1119_111981

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  major_axis : ℝ
  minor_axis : ℝ

/-- The focus of an ellipse with the greater x-coordinate -/
noncomputable def focus_with_greater_x (e : Ellipse) : Point :=
  { x := e.center.x + Real.sqrt (e.major_axis^2 / 4 - e.minor_axis^2 / 4),
    y := e.center.y }

theorem focus_coordinates (e : Ellipse) 
  (h1 : e.center = ⟨3, -2⟩) 
  (h2 : e.major_axis = 6) 
  (h3 : e.minor_axis = 4) : 
  focus_with_greater_x e = ⟨3 + Real.sqrt 5, -2⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l1119_111981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_surface_area_l1119_111919

/-- The lateral surface area of a cylinder with base diameter and height both equal to 4 cm is 16π cm² -/
theorem cylinder_lateral_surface_area :
  ∀ (d h : ℝ), 
  d = 4 →
  h = 4 →
  let c := π * d
  let a := c * h
  a = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_surface_area_l1119_111919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_quadratic_roots_l1119_111939

theorem sum_of_squares_quadratic_roots :
  ∀ x : ℝ, x^2 - 10*x + 16 = 0 →
  let r₁ := (10 + Real.sqrt (10^2 - 4*16)) / 2
  let r₂ := (10 - Real.sqrt (10^2 - 4*16)) / 2
  r₁^2 + r₂^2 = 68 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_quadratic_roots_l1119_111939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_journey_time_l1119_111992

/-- Represents the journey of Joe from home to school -/
structure JoeJourney where
  walkTime : ℚ  -- Time taken to walk half the distance
  waitTime : ℚ  -- Time spent waiting
  runSpeed : ℚ → ℚ  -- Function to calculate running speed based on walking speed

/-- Calculates the total time of Joe's journey -/
def totalTime (j : JoeJourney) : ℚ :=
  j.walkTime + j.waitTime + j.walkTime / 2

/-- Theorem stating that Joe's total journey time is 10 minutes -/
theorem joe_journey_time (j : JoeJourney) 
  (h1 : j.walkTime = 4)
  (h2 : j.waitTime = 2) 
  (h3 : j.runSpeed = λ x => 2 * x) : 
  totalTime j = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_journey_time_l1119_111992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_properties_generalized_distribution_function_characterization_l1119_111936

-- Define the indicator function
noncomputable def indicator (p : Prop) [Decidable p] : ℝ := if p then 1 else 0

-- Define the two functions G₁ and G₂
noncomputable def G₁ (x y : ℝ) : ℝ := indicator (x + y ≥ 0)
noncomputable def G₂ (x y : ℝ) : ℝ := ⌊x + y⌋

-- Define right-continuity
def RightContinuous (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y ε, ε > 0 → ∃ δ, δ > 0 ∧ ∀ x' y', x ≤ x' ∧ x' < x + δ ∧ y ≤ y' ∧ y' < y + δ → |f x' y' - f x y| < ε

-- Define increasing with respect to each variable
def IncreasingInEachVariable (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ y₁ y₂, x₁ ≤ x₂ → y₁ ≤ y₂ → f x₁ y₁ ≤ f x₂ y₂

-- Define generalized distribution function
def GeneralizedDistributionFunction (f : ℝ → ℝ → ℝ) : Prop :=
  RightContinuous f ∧
  (∀ a b c d, a ≤ b → c ≤ d → f a b + f c d - f a d - f c b ≥ 0)

theorem G_properties :
  (¬ RightContinuous G₁) ∧
  (¬ RightContinuous G₂) ∧
  IncreasingInEachVariable G₁ ∧
  IncreasingInEachVariable G₂ ∧
  (¬ GeneralizedDistributionFunction G₁) ∧
  (¬ GeneralizedDistributionFunction G₂) :=
by sorry

theorem generalized_distribution_function_characterization
  (f : ℝ → ℝ → ℝ) :
  GeneralizedDistributionFunction f ↔
  RightContinuous f ∧
  (∀ a b c d, a ≤ b → c ≤ d → f a b + f c d - f a d - f c b ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_properties_generalized_distribution_function_characterization_l1119_111936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_beads_cost_l1119_111930

/-- The cost of one set of crystal beads in dollars -/
def crystal_cost : ℚ := 9

/-- The number of crystal bead sets purchased -/
def crystal_sets : ℕ := 1

/-- The number of metal bead sets purchased -/
def metal_sets : ℕ := 2

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 29

/-- The cost of one set of metal beads in dollars -/
noncomputable def metal_cost : ℚ := (total_cost - crystal_cost * crystal_sets) / metal_sets

theorem metal_beads_cost : metal_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_beads_cost_l1119_111930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_exp_gt_log_plus_two_l1119_111994

-- Define the functions
noncomputable def f₁ : ℝ → ℝ := λ x ↦ x
noncomputable def f₂ : ℝ → ℝ := λ x ↦ Real.exp x
noncomputable def f₃ : ℝ → ℝ := λ x ↦ Real.log x

-- Define the interval (1/2, 2]
def I : Set ℝ := Set.Ioc (1/2) 2

-- Statement 1
theorem monotonic_condition (m : ℝ) :
  (∀ x ∈ I, Monotone (λ x ↦ m * f₁ x - f₃ x)) ↔ 
  (m ≤ 1/2 ∨ m ≥ 2) :=
sorry

-- Statement 2
theorem exp_gt_log_plus_two :
  ∀ x > 0, f₂ x > f₃ x + 2 * (deriv f₁ x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_exp_gt_log_plus_two_l1119_111994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_cone_apex_angle_l1119_111905

/-- The apex angle of a cone is the angle between its generatrices in the axial section. -/
def ApexAngle (cone : Type) : ℝ := sorry

/-- A cone with vertex A -/
structure Cone (A : Type) where
  vertex : A
  apexAngle : ℝ

/-- Three cones touch each other externally -/
def TouchExternally (cone1 cone2 cone3 : Cone A) : Prop := sorry

/-- A cone touches another cone internally -/
def TouchInternally (cone1 cone2 : Cone A) : Prop := sorry

theorem third_cone_apex_angle 
  (A : Type) 
  (cone1 cone2 cone3 cone4 : Cone A)
  (h1 : TouchExternally cone1 cone2 cone3)
  (h2 : cone1.apexAngle = π/6)
  (h3 : cone2.apexAngle = π/6)
  (h4 : cone4.apexAngle = π/3)
  (h5 : TouchInternally cone4 cone1)
  (h6 : TouchInternally cone4 cone2)
  (h7 : TouchInternally cone4 cone3) :
  cone3.apexAngle = 2 * Real.arctan (1 / (Real.sqrt 3 + 4)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_cone_apex_angle_l1119_111905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_books_count_l1119_111969

theorem library_books_count : ℕ := by
  -- Define the return rate
  let return_rate : ℝ := 0.8

  -- Define the number of books at the end of the month
  let end_month_books : ℕ := 65

  -- Define the number of books loaned out
  let loaned_books : ℝ := 50.000000000000014

  -- Define the number of books returned
  let returned_books : ℝ := return_rate * loaned_books

  -- The theorem
  have h : ∀ (initial_books : ℕ),
    (initial_books : ℝ) - loaned_books + returned_books = end_month_books →
    initial_books = 75
  
  sorry

  exact 75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_books_count_l1119_111969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_and_intersection_l1119_111952

/-- Hyperbola with center at origin, right focus at (2,0), and eccentricity 2 -/
structure Hyperbola where
  center : ℝ × ℝ := (0, 0)
  right_focus : ℝ × ℝ := (2, 0)
  eccentricity : ℝ := 2

/-- Line with equation x - y + 1 = 0 -/
def line (x y : ℝ) : Prop := x - y + 1 = 0

/-- Check if a point (x, y) is on the hyperbola -/
def on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

theorem hyperbola_equation_and_intersection (C : Hyperbola) :
  (∀ x y : ℝ, on_hyperbola x y ↔ (x^2 - y^2/3 = 1)) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    line x₁ y₁ ∧ line x₂ y₂ ∧
    on_hyperbola x₁ y₁ ∧ on_hyperbola x₂ y₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_and_intersection_l1119_111952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_R_squared_better_fit_l1119_111947

/-- Coefficient of determination (R²) for a regression model -/
noncomputable def R_squared : ℝ → ℝ := sorry

/-- Measure of model fit -/
noncomputable def model_fit : ℝ → ℝ := sorry

/-- Axiom: R² is between 0 and 1 -/
axiom R_squared_range (x : ℝ) : 0 ≤ R_squared x ∧ R_squared x ≤ 1

/-- Axiom: Model fit is a positive real number -/
axiom model_fit_positive (x : ℝ) : model_fit x > 0

/-- Theorem: A larger R² indicates a better fitting effect -/
theorem larger_R_squared_better_fit (x y : ℝ) (h : R_squared x > R_squared y) :
  model_fit x > model_fit y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_R_squared_better_fit_l1119_111947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l1119_111990

/-- Circle C with center (a, b) and radius √2 -/
noncomputable def circle_C (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = 2}

/-- The curve y = 1/x -/
noncomputable def curve (x : ℝ) : ℝ := 1 / x

/-- The line l: x + 2y = 0 -/
def line_l (x y : ℝ) : Prop := x + 2 * y = 0

/-- The statement to be proved -/
theorem circle_and_line_intersection 
  (a b : ℝ) 
  (h1 : a ∈ Set.Icc 1 2) 
  (h2 : b = curve a) : 
  (a * b = 1) ∧ 
  (∃ (min max : ℝ), 
    min = 2 * Real.sqrt 5 / 5 ∧ 
    max = 2 * Real.sqrt 10 / 5 ∧
    ∀ (x y : ℝ), (x, y) ∈ circle_C a b → line_l x y → 
      2 * Real.sqrt ((x - a)^2 + (y - b)^2) ∈ Set.Icc min max) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l1119_111990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_ratio_l1119_111968

-- Define the circles
def circle_C : Set ℝ := sorry
def circle_D : Set ℝ := sorry

-- Define the diameters
def diameter_D : ℝ := 20

-- Define the areas
noncomputable def area_C : ℝ := sorry
noncomputable def area_D : ℝ := sorry
noncomputable def area_between : ℝ := area_D - area_C

-- State the theorem
theorem circle_diameter_ratio :
  circle_C ⊆ circle_D →  -- C is in the interior of D
  diameter_D = 20 →  -- Diameter of D is 20 cm
  area_between / area_C = 7 →  -- Ratio of shaded area to area of C is 7:1
  ∃ (diameter_C : ℝ), diameter_C = 5 * Real.sqrt 2 :=  -- Diameter of C is 5√2 cm
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_ratio_l1119_111968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_candidates_count_l1119_111964

theorem exam_candidates_count :
  ∃ (total : ℕ) (failed_english_percent : ℚ),
    failed_english_percent = 49 / 100 ∧
    ∃ (passed_english_alone : ℕ),
      passed_english_alone = 630 ∧
      (1 - failed_english_percent) * (total : ℚ) = passed_english_alone ∧
      total = 1235 := by
  -- Introduce the variables
  let total : ℕ := 1235
  let failed_english_percent : ℚ := 49 / 100
  let passed_english_alone : ℕ := 630

  -- Prove the existence of these values
  use total, failed_english_percent
  constructor
  · -- Prove failed_english_percent = 49 / 100
    rfl
  use passed_english_alone
  constructor
  · -- Prove passed_english_alone = 630
    rfl
  constructor
  · -- Prove (1 - failed_english_percent) * (total : ℚ) = passed_english_alone
    -- This step requires actual calculation, which we'll skip for now
    sorry
  · -- Prove total = 1235
    rfl

-- Check that the theorem is well-formed
#check exam_candidates_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_candidates_count_l1119_111964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_perfect_square_factors_l1119_111906

/-- The number of positive perfect square integers that are factors of (2^12)(3^18)(7^10) -/
def num_perfect_square_factors : ℕ := 420

/-- The product in question -/
def product : ℕ := 2^12 * 3^18 * 7^10

/-- A function that returns true if a number is a perfect square, false otherwise -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A function that returns true if a number is a factor of the product, false otherwise -/
def is_factor (n : ℕ) : Prop := product % n = 0

/-- Helper function to make is_perfect_square decidable -/
def is_perfect_square_decidable (n : ℕ) : Bool :=
  match Nat.sqrt n with
  | m => m * m = n

/-- Helper function to make is_factor decidable -/
def is_factor_decidable (n : ℕ) : Bool :=
  product % n = 0

theorem count_perfect_square_factors :
  (Finset.filter (λ n => is_perfect_square_decidable n ∧ is_factor_decidable n) (Finset.range (product + 1))).card = num_perfect_square_factors :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_perfect_square_factors_l1119_111906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stair_climbing_even_perfect_square_l1119_111914

def stair_climbing_sequence : ℕ → ℕ
  | 0 => 1  -- Adding a case for 0
  | 1 => 1
  | 2 => 1
  | 3 => 2
  | 4 => 4
  | n + 5 => stair_climbing_sequence (n + 4) + stair_climbing_sequence (n + 2) + stair_climbing_sequence (n + 1)

theorem stair_climbing_even_perfect_square (n : ℕ) (h : n ≥ 1) :
  ∃ m : ℕ, stair_climbing_sequence (2 * n) = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stair_climbing_even_perfect_square_l1119_111914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_formula_a_increasing_l1119_111929

/-- Sequence a_n defined by the given recurrence relation -/
noncomputable def a : ℕ → ℝ → ℝ
  | 0, t => 2 * t - 3
  | n + 1, t => ((2 * t^(n + 2) - 3) * a n t + 2 * (t - 1) * t^(n + 1) - 1) / (a n t + 2 * t^(n + 1) - 1)

/-- General formula for a_n -/
noncomputable def a_formula (n : ℕ) (t : ℝ) : ℝ := 
  if n = 0 then 2 * t - 3 else 2 * (t^n - 1) / n - 1

theorem a_general_formula (t : ℝ) (h : t ≠ 1 ∧ t ≠ -1) :
  ∀ n : ℕ, a n t = a_formula n t := by
  sorry

theorem a_increasing (t : ℝ) (h : t > 0) :
  ∀ n : ℕ, a (n + 1) t > a n t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_formula_a_increasing_l1119_111929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_required_speed_is_27_l1119_111961

-- Define the distance from home to train station
noncomputable def distance : ℝ := 22.5

-- Define the speeds mentioned in the problem
noncomputable def speed_early : ℝ := 30
noncomputable def speed_late : ℝ := 18

-- Define the time differences in hours
noncomputable def time_early : ℝ := 15 / 60
noncomputable def time_late : ℝ := 15 / 60
noncomputable def time_before : ℝ := 10 / 60

-- Define the function to calculate travel time given speed
noncomputable def travel_time (speed : ℝ) : ℝ := distance / speed

-- Theorem to prove
theorem required_speed_is_27 :
  ∃ (speed : ℝ), 
    travel_time speed_early + time_early = travel_time speed_late - time_late ∧
    travel_time speed = 1 - time_before ∧
    speed = 27 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_required_speed_is_27_l1119_111961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkpoints_needed_additional_checkpoints_needed_l1119_111974

/-- Represents the number of security checkpoints -/
def n : ℕ := sorry

/-- Represents the initial number of people in the queue -/
def initial_queue : ℕ := 3000

/-- Represents the time taken to clear the initial queue (in minutes) -/
def clearing_time : ℕ := 20

/-- Represents the number of people a single checkpoint can process per minute -/
def checkpoint_rate : ℕ := 3

/-- Theorem stating the number of checkpoints needed to clear the initial queue -/
theorem checkpoints_needed :
  initial_queue / (clearing_time * checkpoint_rate) = 50 :=
by sorry

/-- Represents the increased arrival rate of visitors (50% increase) -/
def increased_arrival_rate : ℚ := 3/2

/-- Theorem stating the number of additional checkpoints needed when arrival rate increases -/
theorem additional_checkpoints_needed :
  (10 * n : ℕ) * increased_arrival_rate - (10 * n : ℕ) = (3 * n : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkpoints_needed_additional_checkpoints_needed_l1119_111974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ACM_l1119_111942

-- Define the triangles and point M
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def ABC : Triangle := ⟨18, 10, 0⟩  -- We set c to 0 as it's not given and not needed
noncomputable def ADE : Triangle := ⟨14, 4, 0⟩   -- We set c to 0 as it's not given and not needed

-- M is the midpoint of AE, but we don't need to define it explicitly for this statement

-- Define the area function for right triangles
noncomputable def area (t : Triangle) : ℝ := t.a * t.b / 2

-- State the theorem
theorem area_of_ACM : 
  area ABC = 18 * 10 / 2 ∧ 
  area ADE = 14 * 4 / 2 → 
  ∃ (ACM : Triangle), area ACM = 53 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ACM_l1119_111942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_average_rate_of_change_l1119_111979

/-- The average rate of change of a function f on an interval [x₀, x₁] -/
noncomputable def average_rate_of_change (f : ℝ → ℝ) (x₀ x₁ : ℝ) : ℝ :=
  (f x₁ - f x₀) / (x₁ - x₀)

/-- Theorem stating that the ratio of function value increment to independent variable increment
    is the average rate of change on the interval [x₀, x₁] -/
theorem ratio_is_average_rate_of_change (f : ℝ → ℝ) (x₀ x₁ : ℝ) (h : x₀ ≠ x₁) :
  (f x₁ - f x₀) / (x₁ - x₀) = average_rate_of_change f x₀ x₁ := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_average_rate_of_change_l1119_111979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_max_min_distances_l1119_111920

/-- Given a circle and a point P in the same plane, if the maximum distance from P to any point on
    the circle is 6 and the minimum distance is 4, then the radius of the circle is either 1 or 5. -/
theorem circle_radius_from_max_min_distances (C : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  (∃ (center : ℝ × ℝ) (r : ℝ), C = {x : ℝ × ℝ | (x.1 - center.1)^2 + (x.2 - center.2)^2 = r^2}) →
  (∀ x ∈ C, Real.sqrt ((x.1 - P.1)^2 + (x.2 - P.2)^2) ≤ 6) →
  (∃ y ∈ C, Real.sqrt ((y.1 - P.1)^2 + (y.2 - P.2)^2) = 6) →
  (∀ x ∈ C, Real.sqrt ((x.1 - P.1)^2 + (x.2 - P.2)^2) ≥ 4) →
  (∃ y ∈ C, Real.sqrt ((y.1 - P.1)^2 + (y.2 - P.2)^2) = 4) →
  ∃ (center : ℝ × ℝ) (r : ℝ), C = {x : ℝ × ℝ | (x.1 - center.1)^2 + (x.2 - center.2)^2 = r^2} ∧ (r = 1 ∨ r = 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_max_min_distances_l1119_111920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_significant_quality_difference_frequencies_first_class_l1119_111926

/-- Represents the production data for a machine -/
structure MachineData where
  firstClass : ℕ
  secondClass : ℕ

/-- Calculates the K² statistic for comparing two machines -/
noncomputable def calculateKSquared (machineA machineB : MachineData) : ℝ :=
  let n := (machineA.firstClass + machineA.secondClass + machineB.firstClass + machineB.secondClass : ℝ)
  let a := (machineA.firstClass : ℝ)
  let b := (machineA.secondClass : ℝ)
  let c := (machineB.firstClass : ℝ)
  let d := (machineB.secondClass : ℝ)
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for K² at 99% confidence level -/
def criticalValue99Percent : ℝ := 6.635

/-- Theorem stating that there is a significant difference in quality between Machine A and B -/
theorem significant_quality_difference (machineA machineB : MachineData)
  (h1 : machineA.firstClass = 150)
  (h2 : machineA.secondClass = 50)
  (h3 : machineB.firstClass = 120)
  (h4 : machineB.secondClass = 80) :
  calculateKSquared machineA machineB > criticalValue99Percent := by
  sorry

/-- Frequencies of first-class products -/
def frequencyFirstClass (machine : MachineData) : ℚ :=
  (machine.firstClass : ℚ) / (machine.firstClass + machine.secondClass)

/-- Theorem stating the frequencies of first-class products for Machine A and B -/
theorem frequencies_first_class (machineA machineB : MachineData)
  (h1 : machineA.firstClass = 150)
  (h2 : machineA.secondClass = 50)
  (h3 : machineB.firstClass = 120)
  (h4 : machineB.secondClass = 80) :
  frequencyFirstClass machineA = 3/4 ∧ frequencyFirstClass machineB = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_significant_quality_difference_frequencies_first_class_l1119_111926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_s_r_composition_l1119_111971

def r : Finset Int := {-2, -1, 0, 1}
def r_range : Finset Int := {-1, 0, 3, 5}
def s_domain : Finset Int := {0, 1, 2, 3}

def s (x : Int) : Int := x^2 + 2*x + 1

theorem sum_s_r_composition :
  (r_range ∩ s_domain).sum (fun x => s x) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_s_r_composition_l1119_111971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exposed_area_is_4pi_l1119_111995

/-- Represents the dimensions and cuts of a cylindrical container -/
structure CylinderData where
  height : ℝ
  radius : ℝ
  cut1 : ℝ
  cut2 : ℝ
  cut3 : ℝ

/-- Calculates the total exposed surface area of the rearranged cylinder -/
noncomputable def totalExposedArea (c : CylinderData) : ℝ :=
  2 * Real.pi * c.radius * c.height + 2 * Real.pi * c.radius^2

/-- Theorem stating that the total exposed area is 4π for the given cylinder -/
theorem exposed_area_is_4pi (c : CylinderData) 
  (h1 : c.height = 1) 
  (h2 : c.radius = 1) 
  (h3 : c.cut1 = 1/3) 
  (h4 : c.cut2 = 1/4) 
  (h5 : c.cut3 = 1/6) : 
  totalExposedArea c = 4 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exposed_area_is_4pi_l1119_111995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_nine_l1119_111933

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by slope and a point -/
structure Line where
  slope : ℝ
  point : Point

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

theorem triangle_area_is_nine
  (line1 : Line)
  (line2 : Line)
  (h1 : line1.point = ⟨3, 3⟩)
  (h2 : line2.point = ⟨3, 3⟩)
  (h3 : line1.slope = 1)
  (h4 : line2.slope = -1)
  (line3 : Point → Prop)
  (h5 : ∀ p, line3 p ↔ p.x + p.y = 12)
  : ∃ (a b c : Point), 
    (a = line1.point ∨ (a.y - line1.point.y = line1.slope * (a.x - line1.point.x))) ∧ 
    (b = line2.point ∨ (b.y - line2.point.y = line2.slope * (b.x - line2.point.x))) ∧ 
    line3 c ∧
    triangleArea a b c = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_nine_l1119_111933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_tangent_sum_l1119_111984

theorem smallest_angle_tangent_sum (x : ℝ) : 
  (x > 0) → 
  (Real.tan (2 * x) + Real.tan (3 * x) = 1) → 
  (∀ y, y > 0 → Real.tan (2 * y) + Real.tan (3 * y) = 1 → x ≤ y) → 
  x = 9 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_tangent_sum_l1119_111984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_plus_one_div_by_5_pow_l1119_111917

def x : ℕ → ℕ
  | 0 => 2  -- Adding the base case for 0
  | 1 => 2
  | n + 1 => 2 * (x n)^3 + x n

theorem x_squared_plus_one_div_by_5_pow (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, (x n)^2 + 1 = k * 5^n ∧ ¬(∃ m : ℕ, (x n)^2 + 1 = m * 5^(n+1)) :=
by
  -- The proof goes here
  sorry

#check x_squared_plus_one_div_by_5_pow

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_plus_one_div_by_5_pow_l1119_111917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_center_distance_l1119_111999

/-- Two circles with equal radii intersecting such that one center is outside the other -/
structure IntersectingCircles where
  R : ℝ
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h₁ : ‖O₁ - B‖ = R
  h₂ : ‖O₂ - B‖ = R
  h₃ : ‖O₂ - C‖ = R
  h₄ : (O₂.1 - C.1) * (O₂.1 - O₁.1) + (O₂.2 - C.2) * (O₂.2 - O₁.2) = 0  -- O₂C ⟂ O₁O₂

/-- The distance between the centers of two intersecting circles with equal radii -/
def centerDistance (ic : IntersectingCircles) : ℝ :=
  ‖ic.O₁ - ic.O₂‖

/-- The theorem stating that the distance between the centers is R√3 -/
theorem intersecting_circles_center_distance (ic : IntersectingCircles) :
  centerDistance ic = ic.R * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_center_distance_l1119_111999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arccot_sum_equals_two_pi_l1119_111908

theorem arccot_sum_equals_two_pi :
  2 * Real.arctan (-(1/2)⁻¹) + Real.arctan (-2⁻¹) = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arccot_sum_equals_two_pi_l1119_111908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l1119_111909

/-- Given vectors u and v in a real inner product space, with given norms and sum norm,
    prove that the cosine of the angle between them is 13/35. -/
theorem cosine_of_angle_between_vectors
  {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  (u v : E)
  (norm_u : ‖u‖ = 5)
  (norm_v : ‖v‖ = 7)
  (norm_sum : ‖u + v‖ = 10) :
  inner u v / (‖u‖ * ‖v‖) = 13 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l1119_111909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_coloring_theorem_l1119_111950

theorem house_coloring_theorem (n : ℕ) (σ : Equiv.Perm (Fin n)) :
  ∃ (f : Fin n → Fin 3), ∀ i : Fin n, f i ≠ f (σ i) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_coloring_theorem_l1119_111950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_match_target_l1119_111958

/-- Calculates the target number of runs in a cricket match --/
def cricket_target_runs (first_overs : ℕ) (remaining_overs : ℕ) 
  (first_run_rate : ℚ) (required_run_rate : ℚ) : ℕ :=
  let total_runs := (first_overs : ℚ) * first_run_rate + 
                    (remaining_overs : ℚ) * required_run_rate
  (total_runs.num / total_runs.den).natAbs

/-- Proves that the target number of runs is 282 given the specific conditions --/
theorem cricket_match_target : 
  cricket_target_runs 10 40 (32/10) (25/4) = 282 := by
  rfl

#eval cricket_target_runs 10 40 (32/10) (25/4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_match_target_l1119_111958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_bound_l1119_111975

theorem polynomial_roots_bound {F : Type*} [Field F] [DecidableEq F] (P : Polynomial F) :
  (P.roots.toFinset.card : ℕ) ≤ P.degree :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_bound_l1119_111975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_must_be_nine_l1119_111982

def hexagon_numbers : List ℕ := [1, 2, 7, 8, 9, 13, 14]

def is_valid_arrangement (arrangement : List ℕ) : Prop :=
  arrangement.length = 7 ∧
  (∀ n ∈ arrangement, n ∈ hexagon_numbers) ∧
  (∀ triangle : List ℕ,
    triangle.length = 3 →
    (∀ n ∈ triangle, n ∈ arrangement) →
    (triangle.sum % 3 = 0))

theorem center_must_be_nine (arrangement : List ℕ) :
  is_valid_arrangement arrangement →
  arrangement.get? 3 = some 9 :=
by
  intro h
  sorry

#check center_must_be_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_must_be_nine_l1119_111982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1119_111959

/-- The parabola equation: x = -1/8 * y^2 -/
def parabola_equation (x y : ℝ) : Prop := x = -1/8 * y^2

/-- The focus of a parabola is a point on its axis of symmetry -/
def is_focus (f : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (d : ℝ), ∀ (x y : ℝ), parabola x y →
    (x - f.1)^2 + (y - f.2)^2 = (x - d)^2

/-- The theorem stating that the focus of the given parabola is at (-2, 0) -/
theorem parabola_focus :
  is_focus (-2, 0) parabola_equation := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1119_111959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_315_degrees_l1119_111904

theorem cos_315_degrees : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_315_degrees_l1119_111904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_less_than_c_l1119_111918

-- Define the constants a, b, and c as noncomputable
noncomputable def a : ℝ := 2 * Real.sin (1/2)
noncomputable def b : ℝ := 3 * Real.sin (1/3)
noncomputable def c : ℝ := 3 * Real.cos (1/3)

-- State the theorem
theorem a_less_than_b_less_than_c : a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_less_than_c_l1119_111918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_valid_is_206_l1119_111938

def is_valid (n : ℕ) : Bool := n > 0 && n ≤ 500

def find_nth_valid (seq : List ℕ) (n : ℕ) : Option ℕ :=
  let valid_seq := seq.filter is_valid
  valid_seq.get? (n - 1)

theorem fourth_valid_is_206 (random_seq : List ℕ) :
  random_seq.take 6 = [253, 78, 591, 695, 567, 206] →
  find_nth_valid random_seq 4 = some 206 := by
  intro h
  simp [find_nth_valid, is_valid]
  sorry

#eval find_nth_valid [253, 78, 591, 695, 567, 206] 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_valid_is_206_l1119_111938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_positive_f_odd_l1119_111977

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x) - Real.log (1 + x)

-- Theorem for the domain of f(x)
theorem f_domain : Set.Ioo (-1 : ℝ) 1 = {x : ℝ | ∃ y, f x = y} := by sorry

-- Theorem for the solution set of f(x) > 0
theorem f_positive : Set.Ioo (-1 : ℝ) 0 = {x : ℝ | f x > 0} := by sorry

-- Theorem for f(x) being an odd function
theorem f_odd : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_positive_f_odd_l1119_111977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_f_g_l1119_111946

noncomputable section

open Real

-- Define the functions f and g
def f (x : ℝ) : ℝ := (1/2)^x
def g (x : ℝ) : ℝ := -abs x

-- Define the interval (-∞, 0)
def interval : Set ℝ := Set.Iio 0

-- State the theorem
theorem monotonicity_f_g : 
  (∀ x y, x ∈ interval → y ∈ interval → x < y → f y < f x) ∧ 
  (∀ x y, x ∈ interval → y ∈ interval → x < y → g x < g y) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_f_g_l1119_111946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwich_ratio_l1119_111966

theorem sandwich_ratio : 
  ∀ (billy katelyn chloe : ℕ),
  billy = 49 →
  katelyn = billy + 47 →
  billy + katelyn + chloe = 169 →
  (chloe : ℚ) / (katelyn : ℚ) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwich_ratio_l1119_111966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_relation_l1119_111903

/-- Represents a square with a specific division and shading pattern -/
structure DividedSquare where
  totalArea : ℝ
  shadedArea : ℝ

/-- Square I: divided by diagonals into four triangles, each further divided -/
noncomputable def squareI : DividedSquare :=
  { totalArea := 1
  , shadedArea := 1/2 }

/-- Square II: divided into four equal squares, two non-adjacent shaded -/
noncomputable def squareII : DividedSquare :=
  { totalArea := 1
  , shadedArea := 1/2 }

/-- Square III: divided at third points, two smaller squares in middle shaded -/
noncomputable def squareIII : DividedSquare :=
  { totalArea := 1
  , shadedArea := 2/9 }

/-- Theorem stating the relation between shaded areas of the three squares -/
theorem shaded_area_relation :
  squareI.shadedArea = squareII.shadedArea ∧
  squareI.shadedArea ≠ squareIII.shadedArea ∧
  squareII.shadedArea ≠ squareIII.shadedArea := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_relation_l1119_111903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_12_l1119_111957

/-- The distance between two points A and B given a round trip rowing scenario -/
noncomputable def distance_AB (rowing_speed : ℝ) (stream_speed : ℝ) (total_time : ℝ) : ℝ :=
  (total_time * (rowing_speed^2 - stream_speed^2)) / (2 * rowing_speed)

/-- Theorem stating that the distance between A and B is 12 km -/
theorem distance_AB_is_12 :
  distance_AB 5 1 5 = 12 := by
  -- Unfold the definition of distance_AB
  unfold distance_AB
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_12_l1119_111957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_taker_characterization_l1119_111907

def is_zero_taker (m : ℕ) : Prop :=
  ∃ k : ℕ, ∃ n : ℕ,
    k = (10^n + 1)^2 ∧
    m ∣ k ∧
    n ≥ 2021 ∧
    k % 10 ≠ 0

theorem zero_taker_characterization (m : ℕ) (hm : m > 0) :
  is_zero_taker m ↔ ¬(10 ∣ m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_taker_characterization_l1119_111907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_MA_MB_l1119_111922

noncomputable section

/-- The function f(x) = (x^2 + 4) / x -/
def f (x : ℝ) : ℝ := (x^2 + 4) / x

/-- Point M on the graph of f -/
def M (t : ℝ) : ℝ × ℝ := (t, f t)

/-- Vector MA -/
def MA (t : ℝ) : ℝ × ℝ := ((t - (t^2 + 4) / t) / Real.sqrt 2, (t - (t^2 + 4) / t) / Real.sqrt 2)

/-- Vector MB -/
def MB (t : ℝ) : ℝ × ℝ := (t, 0)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem dot_product_MA_MB (t : ℝ) (h : t > 0) : 
  dot_product (MA t) (MB t) = -2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_MA_MB_l1119_111922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_inequality_l1119_111962

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x - x + 1 - a

-- State the theorem
theorem min_a_for_inequality (a : ℤ) : 
  (∃ x : ℝ, x > 1 ∧ f a x + x < (1 - x) / x) ↔ a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_inequality_l1119_111962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_phi_l1119_111985

noncomputable section

/-- The function that represents the original graph --/
def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

/-- The function that represents the shifted graph --/
def g (x φ : ℝ) : ℝ := f (x - Real.pi/8) φ

/-- Symmetry condition for a function about the y-axis --/
def symmetric_about_y_axis (h : ℝ → ℝ) : Prop :=
  ∀ x, h x = h (-x)

theorem unique_phi :
  ∃! φ, 0 < φ ∧ φ < Real.pi ∧ 
    symmetric_about_y_axis (λ x ↦ g x φ) ∧
    φ = 3 * Real.pi / 4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_phi_l1119_111985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_iff_a_range_l1119_111955

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := 3 * x + Real.cos (Real.pi / 2 - 2 * x) - Real.log (Real.sqrt (x^2 + 1) - x) + 3

-- State the theorem
theorem g_inequality_iff_a_range :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → g (a * x - 2 * Real.exp x + 2) < 3) ↔ (a > 0 ∧ a ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_iff_a_range_l1119_111955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_equidistant_l1119_111954

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the point lying on a line
variable (on_line : Point → Line → Prop)

-- Define the midpoint of a line segment
variable (midpoint : Point → Point → Point)

-- Define the intersection of two lines
variable (intersect : Line → Line → Point)

-- Define the equality of distances
variable (dist_eq : Point → Point → Point → Point → Prop)

-- Define a function to create a line from two points
variable (line_from_points : Point → Point → Line)

-- Theorem statement
theorem line_through_point_equidistant (a b l : Line) (A B C A1 B1 : Point) :
  parallel a b →
  on_line A a →
  on_line B b →
  on_line C l →
  on_line A1 l →
  on_line B1 l →
  on_line A1 a →
  on_line B1 b →
  dist_eq A A1 B B1 →
  (parallel l (line_from_points A B) ∨ on_line (midpoint A B) l) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_equidistant_l1119_111954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1119_111921

def f : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n+2 => if n % 2 = 0 then f (n / 2 + 1) else f (n / 2 + 1) + f (n / 2 + 2)

theorem f_property (n : ℕ) :
  (Finset.filter (fun m => m % 2 = 1 ∧ f m = n) (Finset.range (n^2 + 1))).card = Nat.totient n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1119_111921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_inequality_f_positive_l1119_111940

/-- The function f: ℝ → ℝ defined as f(x) = 1/(1+x²) -/
noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

/-- Theorem stating that f satisfies the given inequality for all positive real numbers -/
theorem f_satisfies_inequality (x : ℝ) (hx : x > 0) :
  f (1/x) ≥ 1 - (Real.sqrt (f x * f (1/x))) / x ∧
  1 - (Real.sqrt (f x * f (1/x))) / x ≥ x^2 * f x := by
  sorry

/-- Theorem stating that f is positive for positive real inputs -/
theorem f_positive (x : ℝ) (hx : x > 0) : f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_inequality_f_positive_l1119_111940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_length_is_110_l1119_111941

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  width : ℚ
  length : ℚ
  roadWidth : ℚ
  travelCost : ℚ
  costPerSqm : ℚ

/-- The length of a lawn given its properties -/
def calculateLawnLength (lawn : LawnWithRoads) : ℚ :=
  (lawn.travelCost / lawn.costPerSqm - lawn.roadWidth * (lawn.width - lawn.roadWidth)) / lawn.roadWidth

/-- Theorem stating the length of the lawn under given conditions -/
theorem lawn_length_is_110 (lawn : LawnWithRoads) 
  (h1 : lawn.width = 60)
  (h2 : lawn.roadWidth = 10)
  (h3 : lawn.travelCost = 4800)
  (h4 : lawn.costPerSqm = 3) :
  calculateLawnLength lawn = 110 := by
  sorry

def main : IO Unit := do
  let result := calculateLawnLength { width := 60, length := 0, roadWidth := 10, travelCost := 4800, costPerSqm := 3 }
  IO.println s!"The length of the lawn is {result} meters"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_length_is_110_l1119_111941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_exponent_sum_l1119_111989

-- Define the expression
noncomputable def cube_root_expression (a b c : ℝ) : ℝ := Real.rpow (48 * a^5 * b^8 * c^14) (1/3)

-- Define the simplified expression
noncomputable def simplified_expression (a b c : ℝ) : ℝ := 2 * a * b^2 * c^4 * Real.rpow (48 * a^2 * b^2 * c^2) (1/3)

-- Define the sum of exponents outside the radical
def sum_of_exponents : ℕ := 1 + 2 + 4

-- Theorem statement
theorem simplification_and_exponent_sum :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
  cube_root_expression a b c = simplified_expression a b c ∧
  sum_of_exponents = 7 := by
  sorry

#eval sum_of_exponents

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_exponent_sum_l1119_111989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1119_111980

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := -Real.log x / Real.log 2 + x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  if x ≥ 0 then Real.arccos x + 2 else x^2 + 2*a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x₁ ≥ 2, ∃ x₂, f x₁ = g x₂ a) ↔ a ∈ Set.Iic (1/2) ∪ Set.Icc 1 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1119_111980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1119_111956

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * Real.cos x ^ 2 - 1

def is_axis_of_symmetry (f : ℝ → ℝ) (axis : ℝ → ℝ) : Prop :=
  ∀ x, f (axis x - x) = f (axis x + x)

def is_center_of_symmetry (f : ℝ → ℝ) (center : ℝ × ℝ) : Prop :=
  ∀ x, f (center.1 - x) = 2 * center.2 - f (center.1 + x)

theorem f_properties :
  (∀ k : ℤ, is_axis_of_symmetry f (λ x ↦ k / 2 * Real.pi + Real.pi / 6)) ∧
  (∀ k : ℤ, is_center_of_symmetry f (k / 2 * Real.pi - Real.pi / 12, 0)) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x ≥ -1) ∧
  f (Real.pi / 6) = 2 ∧
  f (-Real.pi / 6) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1119_111956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travis_journey_cost_l1119_111934

/-- Calculates the total cost of a three-leg journey with given costs, discounts, and taxes --/
def totalJourneyCost (costs : Fin 3 → ℝ) (discounts : Fin 3 → ℝ) (taxes : Fin 3 → ℝ) : ℝ :=
  let discountedCosts := λ i => costs i * (1 - discounts i)
  (Finset.sum Finset.univ discountedCosts) + (Finset.sum Finset.univ taxes)

/-- The total cost of Travis's journey to Australia is $2840 --/
theorem travis_journey_cost :
  let costs : Fin 3 → ℝ := ![1500, 800, 1200]
  let discounts : Fin 3 → ℝ := ![0.25, 0.20, 0.35]
  let taxes : Fin 3 → ℝ := ![100, 75, 120]
  totalJourneyCost costs discounts taxes = 2840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travis_journey_cost_l1119_111934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hensel_lemma_generalized_l1119_111967

/-- Given a polynomial P, prime p, and integer a, if P(y) ≡ a (mod p) has a solution y
    such that P'(y) ≢ 0 (mod p), then P(x) ≡ a (mod p^n) has a solution for any n > 0. -/
theorem hensel_lemma_generalized
  (P : ℤ → ℤ) -- polynomial function
  (P' : ℤ → ℤ) -- derivative of P
  (p : ℕ) -- prime number
  (a : ℤ) -- integer
  (hp : Nat.Prime p)
  (hy : ∃ y : ℤ, P y ≡ a [ZMOD p])
  (hpy : ∀ y : ℤ, P y ≡ a [ZMOD p] → ¬(P' y ≡ 0 [ZMOD p])) :
  ∀ n : ℕ, n > 0 → ∃ x : ℤ, P x ≡ a [ZMOD p^n] :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hensel_lemma_generalized_l1119_111967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1119_111953

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

/-- The x-coordinate of the point of tangency -/
def x₀ : ℝ := -1

/-- The y-coordinate of the point of tangency -/
def y₀ : ℝ := -3

/-- The slope of the tangent line -/
noncomputable def m : ℝ := (deriv f) x₀

/-- Theorem: The equation of the tangent line to the curve y = x^3 + 3x^2 - 5
    at the point (-1, -3) is 3x + y + 6 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 3*x + y + 6 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1119_111953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_sum_l1119_111949

noncomputable section

/-- Represents a parabola y = ax^2 -/
structure Parabola where
  a : ℝ
  pos_a : a > 0

/-- Represents a point on the parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.a * x^2

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (0, 1 / (4 * p.a))

/-- A line through the focus intersecting the parabola at two points -/
structure IntersectingLine (p : Parabola) where
  P : ParabolaPoint p
  Q : ParabolaPoint p
  through_focus : ∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧
    (t * P.x + (1 - t) * (focus p).1 = Q.x) ∧
    (t * P.y + (1 - t) * (focus p).2 = Q.y)

/-- Theorem: For any parabola and line through its focus intersecting
    the parabola at P and Q, 1/PF + 1/FQ = 4a -/
theorem parabola_intersection_sum (p : Parabola) (l : IntersectingLine p) :
  let F := focus p
  let PF := Real.sqrt ((l.P.x - F.1)^2 + (l.P.y - F.2)^2)
  let FQ := Real.sqrt ((l.Q.x - F.1)^2 + (l.Q.y - F.2)^2)
  1 / PF + 1 / FQ = 4 * p.a := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_sum_l1119_111949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l1119_111924

open Real

theorem indefinite_integral_proof (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 2) (h3 : x ≠ 0) :
  let F := λ x => x + (1/2) * log (abs (x - 4)) + 4 * log (abs (x - 2)) - (3/2) * log (abs x)
  (deriv F x) = (x^3 - 3*x^2 - 12) / ((x - 4) * (x - 2) * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l1119_111924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_bound_l1119_111983

theorem sin_2x_bound (x : ℝ) (h : Real.sin x > 0.9) : |Real.sin (2 * x)| < 0.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_bound_l1119_111983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_308_607_l1119_111935

/-- Rounds a real number to the nearest integer -/
noncomputable def roundToNearestInt (x : ℝ) : ℤ :=
  if x - ↑⌊x⌋ < 0.5 then ⌊x⌋ else ⌈x⌉

/-- The statement that 308.607 rounded to the nearest whole number is 309 -/
theorem round_308_607 : roundToNearestInt 308.607 = 309 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_308_607_l1119_111935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dormitory_students_theorem_l1119_111931

theorem dormitory_students_theorem (T : ℝ) (hT : T > 0) : 
  let first_year := T / 2
  let second_year := T / 2
  let first_year_undeclared := (4 / 5) * first_year
  let first_year_declared := first_year - first_year_undeclared
  let second_year_declared := (1 / 3) * (first_year_declared / first_year) * second_year
  let second_year_undeclared := second_year - second_year_declared
  second_year_undeclared / T = 29 / 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dormitory_students_theorem_l1119_111931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_iff_x_neq_two_l1119_111988

-- Define the fraction
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 2)

-- Theorem statement
theorem fraction_meaningful_iff_x_neq_two :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x ≠ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_iff_x_neq_two_l1119_111988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_agricultural_product_pricing_l1119_111965

/-- Represents the relationship between selling price and daily sales volume -/
def sales_function (x : ℝ) : ℝ := -30 * x + 1500

/-- Represents the daily sales profit function -/
def profit_function (x : ℝ) : ℝ := sales_function x * (x - 30)

/-- Represents the daily profit function with cost a -/
def total_profit_function (x a : ℝ) : ℝ := sales_function x * (x - 30 - a)

theorem agricultural_product_pricing :
  /- Given data points -/
  (∀ (x n : ℝ), (x = 30 ∧ n = 600) ∨ (x = 35 ∧ n = 450) ∨ (x = 40 ∧ n = 300) ∨
                (x = 45 ∧ n = 150) ∨ (x = 50 ∧ n = 0) → n = sales_function x) →
  /- Purchase price is 30 yuan per kilogram -/
  (∀ x : ℝ, x > 30) →
  /- Maximize daily sales profit -/
  (∃ x : ℝ, ∀ y : ℝ, profit_function x ≥ profit_function y) →
  /- Maximum daily profit conditions -/
  (∃ a : ℝ, a > 0 ∧ 
    (∀ x : ℝ, 40 ≤ x ∧ x ≤ 45 → total_profit_function x a ≤ 2430) ∧
    (∃ x : ℝ, 40 ≤ x ∧ x ≤ 45 ∧ total_profit_function x a = 2430)) →
  /- Conclusions -/
  (∀ x : ℝ, sales_function x = -30 * x + 1500) ∧
  (∃ x : ℝ, x = 40 ∧ ∀ y : ℝ, profit_function x ≥ profit_function y) ∧
  (∃ a : ℝ, a = 2 ∧ 
    (∀ x : ℝ, 40 ≤ x ∧ x ≤ 45 → total_profit_function x a ≤ 2430) ∧
    (∃ x : ℝ, 40 ≤ x ∧ x ≤ 45 ∧ total_profit_function x a = 2430)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_agricultural_product_pricing_l1119_111965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_angles_not_always_equal_l1119_111912

-- Define a structure for lines in a plane
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a structure for angles
structure Angle where
  measure : ℝ

-- Define a function to determine if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a predicate to check if a point is on a line
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

-- Define a function to determine if two angles are corresponding
def are_corresponding (a1 a2 : Angle) (l1 l2 l3 : Line) : Prop :=
  ∃ (point1 point2 : ℝ × ℝ), 
    point1 ≠ point2 ∧
    point_on_line point1 l1 ∧
    point_on_line point2 l1 ∧
    point_on_line point1 l2 ∧
    point_on_line point2 l3

-- State the theorem
theorem corresponding_angles_not_always_equal :
  ¬ (∀ (a1 a2 : Angle) (l1 l2 l3 : Line),
    are_corresponding a1 a2 l1 l2 l3 → a1.measure = a2.measure) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_angles_not_always_equal_l1119_111912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_64_l1119_111911

-- Define the sequence and its partial sums
def S (n : ℕ) : ℕ := 2^n - 1

def a : ℕ → ℕ
  | 0 => 0  -- Define a₀ to be 0
  | 1 => S 1
  | n + 1 => S (n + 1) - S n

-- State the theorem
theorem seventh_term_is_64 : a 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_64_l1119_111911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_withheld_percentage_is_twenty_percent_l1119_111928

/-- Calculates the percentage of pay withheld given the hourly rate, work hours, and amount received if the report is not finished. -/
noncomputable def percentage_withheld (hourly_rate : ℝ) (work_hours : ℝ) (amount_received : ℝ) : ℝ :=
  let full_pay := hourly_rate * work_hours
  let withheld_amount := full_pay - amount_received
  (withheld_amount / full_pay) * 100

/-- Theorem stating that the percentage of pay withheld is 20% under the given conditions. -/
theorem withheld_percentage_is_twenty_percent :
  percentage_withheld 50 10 400 = 20 := by
  unfold percentage_withheld
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_withheld_percentage_is_twenty_percent_l1119_111928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merge_to_two_or_one_pile_merge_two_piles_condition_l1119_111973

/-- Represents a pile of coins -/
structure CoinPile where
  count : ℕ

/-- Represents the state of the coin game -/
structure CoinGame where
  piles : List CoinPile

/-- Represents a move in the coin game -/
inductive Move where
  | double : CoinPile → CoinPile → Move

/-- Function to apply a move to the game state -/
def applyMove (game : CoinGame) (move : Move) : CoinGame :=
  sorry

/-- Theorem: For n ≥ 3 piles, it's always possible to merge into two or one pile -/
theorem merge_to_two_or_one_pile (game : CoinGame) :
  game.piles.length ≥ 3 →
  ∃ (moves : List Move), (List.foldl applyMove game moves).piles.length ≤ 2 :=
by sorry

/-- Theorem: For n = 2 piles, the necessary and sufficient condition to merge into one pile -/
theorem merge_two_piles_condition (r s : ℕ) :
  (∃ (k : ℕ), r + s = 2^k * Nat.gcd r s) ↔
  ∃ (moves : List Move),
    let game := CoinGame.mk [CoinPile.mk r, CoinPile.mk s]
    (List.foldl applyMove game moves).piles.length = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merge_to_two_or_one_pile_merge_two_piles_condition_l1119_111973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_210_degrees_l1119_111993

theorem sin_210_degrees : 
  Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_210_degrees_l1119_111993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_propositions_true_l1119_111996

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Define the given conditions
variable (l : Line) (m : Line) (α : Plane) (β : Plane)
variable (h1 : perpendicular l α)
variable (h2 : contained_in m β)

-- Define the propositions
def prop1 (Line Plane : Type) 
  (parallel_planes : Plane → Plane → Prop) 
  (perpendicular_lines : Line → Line → Prop) 
  (l m : Line) (α β : Plane) : Prop := 
  parallel_planes α β → perpendicular_lines l m

def prop2 (Line Plane : Type) 
  (perpendicular_planes : Plane → Plane → Prop) 
  (parallel_lines : Line → Line → Prop) 
  (l m : Line) (α β : Plane) : Prop := 
  perpendicular_planes α β → parallel_lines l m

def prop3 (Line Plane : Type) 
  (parallel_lines : Line → Line → Prop) 
  (perpendicular_planes : Plane → Plane → Prop) 
  (l m : Line) (α β : Plane) : Prop := 
  parallel_lines l m → perpendicular_planes α β

-- State the theorem
theorem two_propositions_true : 
  ∃ (p q : Prop) (r : Prop), 
    (p = prop1 Line Plane parallel_planes perpendicular_lines l m α β ∧ 
     q = prop3 Line Plane parallel_lines perpendicular_planes l m α β ∧ 
     r = prop2 Line Plane perpendicular_planes parallel_lines l m α β) ∧ 
    p ∧ q ∧ ¬r :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_propositions_true_l1119_111996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_connect_four_sections_cost_l1119_111900

/-- Represents a chain section with a fixed number of rings. -/
structure ChainSection where
  rings : Nat

/-- Represents the cost of operations on rings. -/
structure RingOperationCost where
  openCost : Nat
  joinCost : Nat

/-- Calculates the minimum points required to connect chain sections. -/
def minPointsToConnect (sections : List ChainSection) (costs : RingOperationCost) : Nat :=
  sorry

/-- Theorem stating that connecting 4 sections of 3-ring chains requires 15 points. -/
theorem connect_four_sections_cost (sections : List ChainSection) (costs : RingOperationCost) :
  sections.length = 4 ∧ 
  (∀ s ∈ sections, s.rings = 3) ∧
  costs.openCost = 2 ∧
  costs.joinCost = 3 →
  minPointsToConnect sections costs = 15 :=
by
  sorry

#check connect_four_sections_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_connect_four_sections_cost_l1119_111900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1119_111944

noncomputable def f (x : ℝ) := 3 * Real.cos (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x ≤ 3) ∧
  (∀ x, f x ≥ -3) ∧
  (∀ k : ℤ, f (k * Real.pi - Real.pi / 6) = 3) ∧
  (∀ k : ℤ, f (k * Real.pi + Real.pi / 3) = -3) ∧
  (∀ x, f x = 3 → ∃ k : ℤ, x = k * Real.pi - Real.pi / 6) ∧
  (∀ x, f x = -3 → ∃ k : ℤ, x = k * Real.pi + Real.pi / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1119_111944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l1119_111932

/-- The time it takes for two people to complete a task together, given they can each complete it in a certain number of days individually. -/
noncomputable def combined_time (individual_time : ℚ) : ℚ :=
  individual_time / 2

/-- Theorem: If two people can each complete a task in 6 days individually, they can complete it together in 3 days. -/
theorem combined_work_time :
  combined_time 6 = 3 := by
  unfold combined_time
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l1119_111932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1119_111951

noncomputable def f (x : ℝ) := x^2 - 4*x + 3
noncomputable def g (x : ℝ) := Real.exp (Real.log 3 * x) - 2

def M := {x : ℝ | f (g x) > 0}
def N := {x : ℝ | g x < 2}

theorem intersection_M_N : Set.Iio 1 = M ∩ N := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1119_111951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candies_distribution_l1119_111986

def total_candies : ℕ := 15
def num_children : ℕ := 7

def candies_per_child : ℚ := total_candies / num_children

def round_to_nearest_tenth (x : ℚ) : ℚ :=
  (x * 10).floor / 10

theorem candies_distribution :
  round_to_nearest_tenth candies_per_child = 21/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candies_distribution_l1119_111986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_identities_l1119_111948

noncomputable section

open Real

def I (n : ℤ) : ℝ := ∫ x in (0 : ℝ)..(π / 4), 1 / (cos x) ^ n

theorem integral_identities :
  (I 0 = π / 4) ∧
  (I (-1) = sqrt 2 / 2) ∧
  (I 2 = 1) ∧
  (I 1 = log (sqrt 2 + 1)) ∧
  (∀ n : ℤ, n ≠ 0 → I (n + 2) = (n + 1 : ℝ) / n * I n) ∧
  (I (-3) = sqrt 2 / 3) ∧
  (I (-2) = π / 8) ∧
  (I 3 = 2 * log (sqrt 2 + 1)) ∧
  (∫ x in (0 : ℝ)..1, sqrt (x^2 + 1) = 2 * log (sqrt 2 + 1)) ∧
  (∫ x in (0 : ℝ)..1, 1 / (x^2 + 1)^2 = π / 8) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_identities_l1119_111948
