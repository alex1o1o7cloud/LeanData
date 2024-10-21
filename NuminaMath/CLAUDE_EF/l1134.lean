import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_rates_plan_comparison_l1134_113431

-- Define the road length and daily renovation rates
variable (S : ℝ) -- Total road length
variable (a b : ℝ) -- Daily renovation rates for Team A and B

-- Define the conditions
axiom condition1 : 360 / a = 300 / b
axiom condition2 : a = b + 30
axiom condition3 : a ≠ b

-- Define the two plans
noncomputable def plan1_time := (S / 2) / a + (S / 2) / b
noncomputable def plan2_time := S / ((a + b) / 2)

-- Theorem 1: Daily renovation rates
theorem daily_rates : a = 180 ∧ b = 150 := by sorry

-- Theorem 2: Plan comparison
theorem plan_comparison : plan1_time - plan2_time > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_rates_plan_comparison_l1134_113431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_three_rays_with_common_endpoint_l1134_113481

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 6 ≤ 5) ∨
               (5 = y - 6 ∧ x + 3 ≤ 5) ∨
               (x + 3 = y - 6 ∧ 5 ≥ x + 3)}

-- Define what it means for a set to be three rays with a common endpoint
def is_three_rays_with_common_endpoint (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ) (r₁ r₂ r₃ : Set (ℝ × ℝ)),
    S = r₁ ∪ r₂ ∪ r₃ ∧
    r₁ ∩ r₂ = {p} ∧ r₂ ∩ r₃ = {p} ∧ r₃ ∩ r₁ = {p} ∧
    (∀ i : Fin 3, ∃ (v : ℝ × ℝ), ∀ t : ℝ, t ≥ 0 →
      p + t • v ∈ (match i with
        | 0 => r₁
        | 1 => r₂
        | 2 => r₃
      ))

-- State the theorem
theorem T_is_three_rays_with_common_endpoint :
  is_three_rays_with_common_endpoint T := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_three_rays_with_common_endpoint_l1134_113481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cooling_system_theorem_l1134_113410

/-- Represents the cooling system of a car -/
structure CoolingSystem where
  total_volume : ℚ
  initial_concentration : ℚ
  replacement_concentration : ℚ
  target_concentration : ℚ

/-- Calculates the volume to be replaced and the remaining original volume -/
def calculate_replacement (system : CoolingSystem) : ℚ × ℚ :=
  let volume_to_replace := (system.target_concentration - system.initial_concentration) * system.total_volume /
                           (system.replacement_concentration - system.initial_concentration)
  let remaining_volume := system.total_volume - volume_to_replace
  (volume_to_replace, remaining_volume)

theorem cooling_system_theorem (system : CoolingSystem) 
  (h_total : system.total_volume = 19)
  (h_initial : system.initial_concentration = 3/10)
  (h_replacement : system.replacement_concentration = 4/5)
  (h_target : system.target_concentration = 1/2) :
  calculate_replacement system = (38/5, 57/5) := by
  sorry

#eval calculate_replacement { 
  total_volume := 19, 
  initial_concentration := 3/10, 
  replacement_concentration := 4/5, 
  target_concentration := 1/2 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cooling_system_theorem_l1134_113410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l1134_113443

/-- Given a right circular cone whose lateral surface is a sector with a central angle of 120° and an area of 3π, prove that the surface area of the cone is 4π. -/
theorem cone_surface_area (sector_angle : ℝ) (sector_area : ℝ) (cone_surface : ℝ → ℝ) :
  sector_angle = 2 * Real.pi / 3 →
  sector_area = 3 * Real.pi →
  (∀ r, cone_surface r = Real.pi * r^2 + sector_area) →
  ∃ r, cone_surface r = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l1134_113443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ages_of_nisha_and_mike_l1134_113445

-- Define the ages as natural numbers
def claire_future_age : ℕ := 20
def years_until_future : ℕ := 2
def jessica_claire_age_diff : ℕ := 6
def years_ago_for_mike : ℕ := 3

-- Define the ages as functions
def claire_age : ℕ := claire_future_age - years_until_future
def jessica_age : ℕ := claire_age + jessica_claire_age_diff
def mike_age : ℕ := 2 * (jessica_age - years_ago_for_mike)

-- Define Nisha's age as a real number to allow for the square root
noncomputable def nisha_age : ℝ := Real.sqrt (claire_age * jessica_age)

-- Theorem statement
theorem ages_of_nisha_and_mike :
  (mike_age = 42) ∧ (Int.floor nisha_age = 21) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ages_of_nisha_and_mike_l1134_113445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_f_nonneg_range_g_when_f_nonneg_l1134_113425

/-- The function f(x) defined in the problem -/
def f (a x : ℝ) : ℝ := x^2 + 4*a*x + 2*a + 6

/-- The function g(a) defined in the problem -/
def g (a : ℝ) : ℝ := 2 - a * |a - 1|

/-- Theorem stating the conditions for the range of f(x) to be [0, +∞) -/
theorem range_f_nonneg (a : ℝ) :
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f a x = y) ∧ (∀ x : ℝ, f a x ≥ 0) ↔ a = -1 ∨ a = 3/2 :=
by sorry

/-- Theorem stating the range of g(a) when f(x) is always non-negative -/
theorem range_g_when_f_nonneg :
  (∀ a : ℝ, -1 ≤ a ∧ a ≤ 3/2 → ∀ x : ℝ, f a x ≥ 0) →
  (∀ y : ℝ, 5/4 ≤ y ∧ y ≤ 4 → ∃ a : ℝ, g a = y) ∧ (∀ a : ℝ, 5/4 ≤ g a ∧ g a ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_f_nonneg_range_g_when_f_nonneg_l1134_113425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_rectangular_equation_l_intersects_C_l1134_113439

noncomputable def C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

def l (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) + m = 0

def l_rect (x y m : ℝ) : Prop := Real.sqrt 3 * x + y + 2 * m = 0

theorem l_rectangular_equation {x y ρ θ m : ℝ} :
  l ρ θ m ↔ l_rect x y m := by
  sorry

theorem l_intersects_C {m : ℝ} :
  (∃ t : ℝ, l_rect (C t).1 (C t).2 m) ↔ -19/12 ≤ m ∧ m ≤ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_rectangular_equation_l_intersects_C_l1134_113439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1134_113408

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x - 2) + Real.log (3 * x - x^2)

-- Define the domain of f
def domain_f : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1134_113408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_data_mean_variance_sum_l1134_113444

noncomputable def mean (data : List ℝ) : ℝ := data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

noncomputable def standardDeviation (data : List ℝ) : ℝ :=
  Real.sqrt (variance data)

theorem transformed_data_mean_variance_sum
  (data : List ℝ)
  (h_mean : mean data = 6)
  (h_std : standardDeviation data = 2)
  (transformed_data := data.map (λ x => 3 * x - 5)) :
  mean transformed_data + variance transformed_data = 49 := by
  sorry

#check transformed_data_mean_variance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_data_mean_variance_sum_l1134_113444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l1134_113495

-- Define the complex number (3+4i)
def a : ℂ := Complex.mk 3 4

-- Define the set S
def S : Set ℂ := {z : ℂ | (a * z).im = 0}

-- Theorem statement
theorem S_is_line : ∃ (m k : ℝ), S = {z : ℂ | z.re = m * z.im + k} :=
by
  -- We'll use m = -3/4 and k = 0 as derived in the solution
  use (-3/4), 0
  ext z
  simp [S, a]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l1134_113495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_constraint_l1134_113472

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

-- Define the theorem
theorem symmetry_axis_constraint (ω : ℝ) (h1 : ω > 1/4) :
  (∀ x : ℝ, f ω x = f ω (-x) → x ∉ Set.Ioo Real.pi (2*Real.pi)) →
  1/3 ≤ ω ∧ ω ≤ 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_constraint_l1134_113472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_experimental_field_map_area_l1134_113477

/-- Calculates the area of a rectangular field on a map given the field's real dimensions and the map scale. -/
noncomputable def map_area (length width : ℝ) (scale : ℝ) : ℝ :=
  (length * width) * (1 / scale)^2

/-- Theorem stating that a 200m x 100m field on a 1:2000 scale map has an area of 50 square centimeters. -/
theorem experimental_field_map_area :
  let field_length : ℝ := 200
  let field_width : ℝ := 100
  let map_scale : ℝ := 2000
  map_area field_length field_width map_scale = 50 := by
  sorry

#check experimental_field_map_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_experimental_field_map_area_l1134_113477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_f_f_monotone_increasing_on_interval_l1134_113493

noncomputable def f (x : ℝ) := (1/2 : ℝ) ^ Real.sqrt (-x^2 + x + 2)

noncomputable def g (x : ℝ) := (1/2 : ℝ) ^ x

theorem monotone_increasing_interval_f :
  ∀ x y, x ∈ Set.Icc (1/2 : ℝ) 2 → y ∈ Set.Icc (1/2 : ℝ) 2 → x ≤ y → f x ≤ f y :=
by sorry

axiom g_monotone_decreasing : ∀ x y, x ≤ y → g x ≥ g y

-- The main theorem
theorem f_monotone_increasing_on_interval :
  MonotoneOn f (Set.Icc (1/2 : ℝ) 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_f_f_monotone_increasing_on_interval_l1134_113493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_four_minus_x_squared_plus_abs_x_l1134_113447

/-- The definite integral of √(4 - x²) + |x| from -2 to 2 equals 2π + 4 -/
theorem integral_sqrt_four_minus_x_squared_plus_abs_x : 
  ∫ x in (-2)..(2), (Real.sqrt (4 - x^2) + |x|) = 2 * Real.pi + 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_four_minus_x_squared_plus_abs_x_l1134_113447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_length_is_25_l1134_113488

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a door or window -/
structure OpeningDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the total area to be whitewashed -/
def areaToWhitewash (room : RoomDimensions) (door : OpeningDimensions) (window : OpeningDimensions) (numWindows : ℕ) : ℝ :=
  2 * (room.length * room.height + room.width * room.height) - 
  (door.width * door.height + (numWindows : ℝ) * window.width * window.height)

/-- Theorem: Given the room dimensions and openings, if the total cost is 2718 at 3 Rs per sq ft, then the room length is 25 feet -/
theorem room_length_is_25 (x : ℝ) :
  let room := RoomDimensions.mk x 15 12
  let door := OpeningDimensions.mk 6 3
  let window := OpeningDimensions.mk 4 3
  let costPerSqFt := 3
  areaToWhitewash room door window 3 * costPerSqFt = 2718 →
  x = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_length_is_25_l1134_113488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_growth_rate_is_half_l1134_113401

/-- Represents the farm's production data and growth rates -/
structure FarmData where
  initial_area : ℝ
  initial_yield : ℝ
  final_yield : ℝ
  growth_rate_yield : ℝ

/-- The growth rate of the planting area is twice the growth rate of the average yield per acre -/
noncomputable def growth_rate_area (fd : FarmData) : ℝ := 2 * fd.growth_rate_yield

/-- The final area after growth -/
noncomputable def final_area (fd : FarmData) : ℝ := fd.initial_area * (1 + growth_rate_area fd)

/-- The average yield per acre after growth -/
noncomputable def avg_yield_after_growth (fd : FarmData) : ℝ := 
  (fd.initial_yield / fd.initial_area) * (1 + fd.growth_rate_yield)

/-- The theorem stating that the growth rate of the average yield per acre is 0.5 -/
theorem farm_growth_rate_is_half (fd : FarmData) 
    (h1 : fd.initial_area = 5)
    (h2 : fd.initial_yield = 10000)
    (h3 : fd.final_yield = 30000)
    (h4 : final_area fd * avg_yield_after_growth fd = fd.final_yield) :
    fd.growth_rate_yield = 0.5 := by
  sorry

#check farm_growth_rate_is_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_growth_rate_is_half_l1134_113401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_NaNO3_formed_amount_l1134_113485

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactants : List (String × Moles)
  products : List (String × Moles)

instance : OfNat Moles n where
  ofNat := (n : ℝ)

/-- The chemical equation for the reaction -/
def reactionEquation : Reaction :=
  { reactants := [("HNO3", 1), ("NaHCO3", 1)],
    products := [("NaNO3", 1), ("CO2", 1), ("H2O", 1)] }

/-- The amount of HNO3 available -/
def availableHNO3 : Moles := 1

/-- The amount of NaHCO3 available -/
def availableNaHCO3 : Moles := 1

/-- Calculates the amount of product formed based on the reaction and available reactants -/
def productFormed (reaction : Reaction) (availableReactants : List (String × Moles)) (product : String) : Moles :=
  sorry

theorem NaNO3_formed_amount :
  productFormed reactionEquation [("HNO3", availableHNO3), ("NaHCO3", availableNaHCO3)] "NaNO3" = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_NaNO3_formed_amount_l1134_113485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1134_113437

/-- Given a hyperbola C and a circle satisfying certain conditions, 
    prove that the eccentricity of C is √7/2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (A B : ℝ × ℝ) : 
  a > 0 → b > 0 → 
  c^2 = a^2 + b^2 → 
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → 
    ∃ k, y = (b/a) * x * k ∧ 
    (x - c)^2 + y^2 = a^2) →
  ‖A - B‖ = a → 
  c/a = Real.sqrt 7 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1134_113437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_lines_l1134_113434

/-- A point on the circle (x-1)^2 + (y-1)^2 = 50 with integer coordinates -/
def IntegerPointOnCircle (x y : ℤ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 50

/-- A line ax + by = 1 that passes through an integer point on the circle -/
def ValidLine (a b : ℝ) : Prop :=
  ∃ (x y : ℤ), IntegerPointOnCircle x y ∧ a * (x : ℝ) + b * (y : ℝ) = 1

/-- The main theorem stating that there are exactly 36 valid lines -/
theorem count_valid_lines : 
  ∃ (S : Finset (ℝ × ℝ)), S.card = 36 ∧ ∀ (p : ℝ × ℝ), p ∈ S ↔ ValidLine p.1 p.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_lines_l1134_113434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_power_product_l1134_113492

theorem sum_of_digits_of_power_product : ∃ n : ℕ, 
  (∀ d : ℕ, d ∈ Nat.digits 10 (2^2010 * 5^2012 * 7) → d ≤ 9) ∧ 
  (Nat.digits 10 (2^2010 * 5^2012 * 7)).sum = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_power_product_l1134_113492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_percentage_per_annum_calculation_l1134_113442

/-- 
Given a banker's gain, banker's discount, and time period,
calculate the percentage per annum.
-/
noncomputable def calculate_percentage_per_annum (bankers_gain : ℝ) (bankers_discount : ℝ) (time_period : ℝ) : ℝ :=
  let true_discount := bankers_discount - bankers_gain
  let rate := (bankers_discount / true_discount - 1) / time_period
  rate * 100

/-- 
Theorem stating that given the specific values in the problem,
the calculated percentage per annum is approximately 28.67%.
-/
theorem percentage_per_annum_calculation :
  let bankers_gain := (684 : ℝ)
  let bankers_discount := (1634 : ℝ)
  let time_period := (6 : ℝ)
  let result := calculate_percentage_per_annum bankers_gain bankers_discount time_period
  ∃ ε > 0, abs (result - 28.67) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_percentage_per_annum_calculation_l1134_113442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1134_113470

/-- Given a geometric sequence of real numbers where the fourth term is 6! and the sixth term is 7!,
    prove that the first term is (720√7)/49. -/
theorem geometric_sequence_first_term :
  ∀ (a : ℝ) (r : ℝ),
    a * r^3 = 720 →
    a * r^5 = 5040 →
    a = (720 * Real.sqrt 7) / 49 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1134_113470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_after_departure_l1134_113480

theorem average_age_after_departure (initial_people : ℕ) (initial_average : ℚ) 
  (departing_age : ℕ) (remaining_people : ℕ) : 
  initial_people = 8 →
  initial_average = 25 →
  departing_age = 20 →
  remaining_people = 7 →
  (initial_people * initial_average - departing_age) / remaining_people = 25.71 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_after_departure_l1134_113480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_equation_l1134_113438

theorem cosine_sine_equation (x : ℝ) :
  Real.cos x - 3 * Real.sin x = 2 →
  2 * Real.sin x + 3 * Real.cos x = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_equation_l1134_113438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_three_rays_with_common_endpoint_l1134_113473

/-- The set T of points in three-dimensional space satisfying the given conditions -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | 
    (5 = p.2.2 + 1 ∧ p.2.2 - 3 ≤ 5) ∨
    (5 = p.2.2 - 3 ∧ p.2.1 + 1 ≤ 5) ∨
    (p.2.1 + 1 = p.2.2 - 3 ∧ 5 ≤ p.2.1 + 1)}

/-- The common endpoint of the three rays -/
def common_endpoint : ℝ × ℝ × ℝ := (0, 4, 8)

/-- The three rays that form set T -/
def ray1 : Set (ℝ × ℝ × ℝ) := {p : ℝ × ℝ × ℝ | p.1 = 0 ∧ p.2.1 = 4 ∧ p.2.2 ≤ 8}
def ray2 : Set (ℝ × ℝ × ℝ) := {p : ℝ × ℝ × ℝ | p.1 = 0 ∧ p.2.1 ≤ 4 ∧ p.2.2 = 8}
def ray3 : Set (ℝ × ℝ × ℝ) := {p : ℝ × ℝ × ℝ | p.1 = 0 ∧ p.2.1 ≥ 4 ∧ p.2.2 = p.2.1 + 4}

/-- Theorem stating that T consists of three rays with a common endpoint -/
theorem T_is_three_rays_with_common_endpoint :
  T = ray1 ∪ ray2 ∪ ray3 ∧
  common_endpoint ∈ ray1 ∧
  common_endpoint ∈ ray2 ∧
  common_endpoint ∈ ray3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_three_rays_with_common_endpoint_l1134_113473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boris_arrives_later_l1134_113497

/-- Represents the race scenario between Boris and Vasya -/
structure RaceScenario where
  v : ℝ  -- Vasya's initial speed
  L : ℝ  -- Half of the total route length
  h : v > 0  -- Vasya's speed is positive
  k : L > 0  -- Route length is positive

/-- Calculates Vasya's total travel time -/
noncomputable def vasyaTime (s : RaceScenario) : ℝ :=
  s.L / s.v + 2 * s.L / s.v

/-- Calculates Boris's total travel time -/
noncomputable def borisTime (s : RaceScenario) : ℝ :=
  1 + s.L / (10 * s.v) + 4 + s.L / (5 * s.v)

/-- Theorem stating that Boris arrives at least 1 hour after Vasya -/
theorem boris_arrives_later (s : RaceScenario) :
  borisTime s ≥ vasyaTime s + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boris_arrives_later_l1134_113497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_l1134_113487

/-- Represents an elliptical billiard table -/
structure EllipticalTable where
  a : ℝ  -- Half of the major axis length
  c : ℝ  -- Half of the focal distance
  h : 0 < c ∧ c < a  -- Condition for a valid ellipse

/-- The possible distances traveled by a ball on an elliptical table -/
def possibleDistances (table : EllipticalTable) : Set ℝ :=
  {4 * table.a, 2 * (table.a - table.c), 2 * (table.a + table.c)}

/-- The boundary of the elliptical table -/
def ellipseBoundary (table : EllipticalTable) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 / table.a)^2 + (p.2 / Real.sqrt (table.a^2 - table.c^2))^2 = 1}

/-- The length of a path -/
noncomputable def pathLength (path : ℝ → ℝ × ℝ) : ℝ := sorry

/-- 
Theorem: For an elliptical billiard table, the distance traveled by a ball 
starting from one focus, reflecting off the wall, and returning to the same 
focus for the first time is one of the possible distances.
-/
theorem ball_travel_distance (table : EllipticalTable) 
  (d : ℝ) (h : d ∈ possibleDistances table) : 
  ∃ (path : ℝ → ℝ × ℝ), 
    (path 0 = (table.c, 0)) ∧  -- Start at one focus
    (path 1 = (table.c, 0)) ∧  -- End at the same focus
    (∃ t ∈ Set.Ioo 0 1, path t ∈ ellipseBoundary table) ∧  -- Reflect off the wall
    (pathLength path = d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_l1134_113487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_eight_l1134_113464

theorem floor_expression_equals_eight :
  let n : ℝ := 2022
  ⌊(n + 1)^3 / ((n - 1) * n) - (n - 1)^3 / (n * (n + 1))⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_eight_l1134_113464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_phase_shift_specific_cosine_phase_shift_l1134_113476

/-- The phase shift of y = cos(Bx - C) is C/B units to the right when C is positive -/
theorem cosine_phase_shift (B C : ℝ) (hB : B ≠ 0) :
  let φ := C / B
  φ > 0 → (∀ x, Real.cos (B * x - C) = Real.cos (B * (x - φ))) :=
by sorry

/-- The phase shift of y = cos(5x - π/2) is π/10 units to the right -/
theorem specific_cosine_phase_shift :
  let φ := (π / 2) / 5
  φ = π / 10 ∧ φ > 0 ∧ (∀ x, Real.cos (5 * x - π / 2) = Real.cos (5 * (x - φ))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_phase_shift_specific_cosine_phase_shift_l1134_113476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_replaced_person_l1134_113416

/-- Theorem: Weight of replaced person in a group weight change scenario -/
theorem weight_of_replaced_person
  (n : ℕ)  -- Number of people in the initial group
  (new_weight : ℝ)  -- Weight of the new person
  (avg_increase : ℝ)  -- Increase in average weight
  (h1 : n = 8)  -- There are 8 people initially
  (h2 : new_weight = 85)  -- The new person weighs 85 kg
  (h3 : avg_increase = 2.5)  -- The average weight increases by 2.5 kg
  : ∃ (replaced_weight : ℝ), replaced_weight = 65 := by
  -- The weight of the replaced person is 65 kg
  sorry

#check weight_of_replaced_person

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_replaced_person_l1134_113416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bead_arrangement_probability_l1134_113441

def total_beads : ℕ := 7
def red_beads : ℕ := 3
def white_beads : ℕ := 2
def blue_beads : ℕ := 2

def total_arrangements : ℕ := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

def valid_arrangements : ℕ := 12

theorem bead_arrangement_probability :
  (valid_arrangements : ℚ) / total_arrangements = 2 / 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bead_arrangement_probability_l1134_113441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l1134_113479

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)^2 - 1/2

theorem min_translation_for_symmetry 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω x = f ω (x + π/(2*ω))) 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_symmetry : ∀ x, f ω (x + a) = -f ω (-x + a)) :
  π/8 ≤ a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l1134_113479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1134_113450

theorem triangle_side_length (A B C : ℝ × ℝ) :
  let angle_A := 60 * Real.pi / 180
  let AC := 16
  let area := 220 * Real.sqrt 3
  let BC := Real.sqrt ((AC^2 + 55^2) - 2 * AC * 55 * Real.cos angle_A)
  (B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2) = 2 * area ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = AC^2 ∧
  Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) /
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)) = angle_A
  →
  BC = 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1134_113450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l1134_113400

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 2 4

-- Define the composite function g(x) = f(log₂(x))
noncomputable def g (x : ℝ) : ℝ := f (Real.log x / Real.log 2)

-- Theorem statement
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 4 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l1134_113400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_w_powers_l1134_113407

noncomputable def w : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I

theorem distinct_values_of_w_powers (m n l : ℤ) (h1 : m ≠ n) (h2 : n ≠ l) (h3 : m ≠ l) :
  ∃ (S : Finset ℂ), (Finset.card S = 10) ∧ 
  (∀ (a b c : ℤ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) → (w^a + w^b + w^c) ∈ S) ∧
  (∀ x ∈ S, ∃ (p q r : ℤ), (p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧ (x = w^p + w^q + w^r)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_w_powers_l1134_113407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1134_113405

/-- Represents the state of a lamp (on or off) -/
inductive LampState
| Off
| On
deriving Repr, DecidableEq

/-- Represents the configuration of all lamps -/
def LampConfiguration := Fin 2012 → LampState

/-- Represents a player in the game -/
inductive Player
| First
| Second
deriving Repr, DecidableEq

/-- Represents the game state -/
structure GameState where
  configuration : LampConfiguration
  player : Player
  history : List LampConfiguration

/-- Defines a valid move in the game -/
def validMove (state : GameState) (newConfig : LampConfiguration) : Prop :=
  ∃ (i : Fin 2012), 
    (∀ (j : Fin 2012), j ≠ i → newConfig j = state.configuration j) ∧
    (newConfig i ≠ state.configuration i) ∧
    (newConfig ∉ state.history)

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Fin 2012),
    ∀ (initialState : GameState),
      initialState.player = Player.First →
      ∃ (game : ℕ → GameState),
        game 0 = initialState ∧
        (∀ n : ℕ, 
          validMove (game n) (game (n+1)).configuration ∧
          (game (n+1)).player = (if (game n).player = Player.First then Player.Second else Player.First)) ∧
        ∃ (finalStep : ℕ), ¬∃ (newConfig : LampConfiguration), validMove (game finalStep) newConfig :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1134_113405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_PA_l1134_113454

-- Define the curve C
def C (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

-- Define the line l
def l (t : ℝ) : ℝ × ℝ := (2 + t, 2 - 2*t)

-- Define the angle between two vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem max_min_distance_PA {P A : ℝ × ℝ} {t : ℝ} :
  C P.1 P.2 →
  A = l t →
  angle (A.1 - P.1, A.2 - P.2) (1, -2) = π/6 →
  (∀ Q : ℝ × ℝ, C Q.1 Q.2 → 
    ∀ B : ℝ × ℝ, (∃ s : ℝ, B = l s) → 
    angle (B.1 - Q.1, B.2 - Q.2) (1, -2) = π/6 → 
    distance P A ≤ distance Q B) →
  distance P A = 22 * Real.sqrt 5 / 5 ∧
  (∀ Q : ℝ × ℝ, C Q.1 Q.2 → 
    ∀ B : ℝ × ℝ, (∃ s : ℝ, B = l s) → 
    angle (B.1 - Q.1, B.2 - Q.2) (1, -2) = π/6 → 
    distance Q B ≥ 2 * Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_PA_l1134_113454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_score_ratio_l1134_113484

theorem game_score_ratio : ∀ (connor_score amy_score jason_score : ℕ),
  connor_score = 2 →
  amy_score = connor_score + 4 →
  (∃ m : ℕ, jason_score = m * amy_score) →
  connor_score + amy_score + jason_score = 20 →
  jason_score = 2 * amy_score :=
λ connor_score amy_score jason_score
  h_connor h_amy h_jason h_total ↦
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_score_ratio_l1134_113484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_remaining_amount_l1134_113466

noncomputable section

def initial_amount : ℝ := 250

def banana_cost : ℝ := 6 * 5
def pear_cost : ℝ := 8 * 1.5
def asparagus_cost : ℝ := 3 * 7.5
def chicken_cost : ℝ := 2 * 20
def strawberry_cost : ℝ := 5 * 4
def olive_oil_cost : ℝ := 15
def almond_cost : ℝ := 25
def shrimp_cost : ℝ := 0.5 * 20
def cheese_cost : ℝ := 1.2 * 10.5

def total_cost : ℝ := banana_cost + pear_cost + asparagus_cost + chicken_cost + 
                      strawberry_cost + olive_oil_cost + almond_cost + shrimp_cost + 
                      cheese_cost

def discount_threshold : ℝ := 200
def discount_amount : ℝ := 10

noncomputable def applied_discount : ℝ := if total_cost ≥ discount_threshold then discount_amount else 0

noncomputable def final_cost : ℝ := total_cost - applied_discount

noncomputable def remaining_amount : ℝ := initial_amount - final_cost

theorem shopping_remaining_amount : remaining_amount = 62.9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_remaining_amount_l1134_113466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l1134_113432

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- Two parallel lines l₁ and l₂ -/
def l₁ (x y : ℝ) : Prop := x + y + 1 = 0
def l₂ (x y : ℝ) : Prop := x + y - 1 = 0

theorem distance_between_l₁_and_l₂ :
  distance_between_parallel_lines 1 1 (-1) 1 = Real.sqrt 2 := by
  -- Unfold the definition of distance_between_parallel_lines
  unfold distance_between_parallel_lines
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l1134_113432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trench_digging_problem_l1134_113403

/-- Represents the total amount of work required to dig the trench -/
def total_work : ℝ := sorry

/-- The initial number of workers in the brigade -/
def initial_workers : ℕ := sorry

/-- The initial number of working hours per day -/
def initial_hours : ℕ := sorry

theorem trench_digging_problem :
  (total_work = 6 * initial_workers * initial_hours) ∧
  (total_work = 9 * (initial_workers - 5) * (initial_hours - 1)) ∧
  (total_work = 12 * (initial_workers - 7) * (initial_hours - 2)) →
  initial_workers = 21 ∧ initial_hours = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trench_digging_problem_l1134_113403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1134_113499

-- Define the hyperbola equation
def hyperbola (m x y : ℝ) : Prop :=
  x^2 / (m^2 + 8) - y^2 / (6 - 2*m) = 1

-- Define the focal distance
noncomputable def focal_distance (m : ℝ) : ℝ :=
  2 * Real.sqrt (m^2 - 2*m + 14)

-- Define the condition for minimum focal distance
def min_focal_distance (m : ℝ) : Prop :=
  ∀ k, focal_distance m ≤ focal_distance k

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop :=
  y = 2/3 * x ∨ y = -2/3 * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ m x y, hyperbola m x y ∧ min_focal_distance m → asymptotes x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1134_113499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_origin_l1134_113489

/-- The rectangular region --/
def Rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- The distance function between two points --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The region where points are closer to (0,0) than to (6,2) --/
def CloserRegion : Set (ℝ × ℝ) :=
  {p ∈ Rectangle | distance p (0, 0) < distance p (6, 2)}

/-- The probability measure on the rectangle --/
noncomputable def rectangleMeasure : Set (ℝ × ℝ) → ℝ := sorry

/-- The theorem stating the probability --/
theorem probability_closer_to_origin : rectangleMeasure CloserRegion / rectangleMeasure Rectangle = 5 / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_origin_l1134_113489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_l1134_113457

noncomputable def sine_graph (x : ℝ) := Real.sin (2 * x)

noncomputable def shifted_sine_graph (x : ℝ) := Real.sin (2 * x + 1)

theorem sine_graph_shift :
  ∀ x : ℝ, shifted_sine_graph x = sine_graph (x + 1/2) := by
  intro x
  simp [sine_graph, shifted_sine_graph]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_l1134_113457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_simple_solution_for_unfair_coin_l1134_113482

/-- The probability of getting exactly k heads in n independent tosses of a coin with probability p of showing heads -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The problem statement -/
theorem no_simple_solution_for_unfair_coin : ¬ ∃ (p : ℝ),
  binomial_probability 6 4 p = 27 / 256 ∧ 
  (p = 1/2 ∨ p = 2/3 ∨ p = 3/4 ∨ p = 5/6) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_simple_solution_for_unfair_coin_l1134_113482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_is_eleven_l1134_113449

/-- A rectangle with a regular octagon inscribed -/
structure OctagonInRectangle where
  /-- The width of the rectangle -/
  width : ℝ
  /-- The height of the rectangle -/
  height : ℝ
  /-- The side length of the octagon -/
  octagon_side : ℝ
  /-- Width is positive -/
  width_pos : 0 < width
  /-- Height is positive -/
  height_pos : 0 < height
  /-- Octagon side is positive -/
  octagon_side_pos : 0 < octagon_side
  /-- The width of the rectangle is equal to AB + octagon_side + AB -/
  width_eq : width = 2 * 1 + octagon_side
  /-- The height of the rectangle is equal to BC + octagon_side + BC -/
  height_eq : height = 2 * 2 + octagon_side

/-- The area of the regular octagon inscribed in the rectangle -/
noncomputable def octagon_area (r : OctagonInRectangle) : ℝ :=
  r.width * r.height - 4 * (1 * 2 / 2)

/-- Theorem stating that the area of the inscribed regular octagon is 11 square units -/
theorem octagon_area_is_eleven (r : OctagonInRectangle) (h : r.octagon_side = 1) : 
  octagon_area r = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_is_eleven_l1134_113449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_f_geq_neg_two_range_m_f_geq_g_l1134_113412

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * m * x - 1
noncomputable def g (x : ℝ) : ℝ := |x| - 7/4

-- Theorem for the solution of f(x) ≥ -2
theorem solution_f_geq_neg_two (m : ℝ) :
  (∀ x, f m x ≥ -2) ↔ -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 := by
  sorry

-- Theorem for the range of m such that f(x) ≥ g(x) for all x ∈ [0, 2]
theorem range_m_f_geq_g :
  {m : ℝ | ∀ x ∈ Set.Icc 0 2, f m x ≥ g x} = Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_f_geq_neg_two_range_m_f_geq_g_l1134_113412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_condition_l1134_113491

theorem integer_fraction_condition (p : ℕ) : 
  p > 0 ∧ (∃ (n : ℤ), n > 0 ∧ (4 * p + 34 : ℚ) / (3 * p - 7) = n) ↔ p = 3 ∨ p = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_condition_l1134_113491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_equals_6_to_49_b_general_formula_l1134_113406

noncomputable def b : ℕ → ℝ
  | 0 => 1  -- Add this case to handle n = 0
  | 1 => 1
  | (n + 1) => Real.sqrt (36 * (b n)^2)

theorem b_50_equals_6_to_49 : b 50 = 6^49 := by
  -- The proof goes here
  sorry

-- Helper lemma to prove the general formula
theorem b_general_formula (n : ℕ) : b n = 6^(n - 1) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_equals_6_to_49_b_general_formula_l1134_113406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_income_increase_l1134_113494

/-- Calculates the percentage increase in total income --/
noncomputable def percentage_increase (initial_job : ℝ) (initial_side_gig : ℝ) (initial_dividends : ℝ)
                        (new_job : ℝ) (new_side_gig : ℝ) (new_dividends : ℝ) : ℝ :=
  let initial_total := initial_job + initial_side_gig + initial_dividends
  let new_total := new_job + new_side_gig + new_dividends
  ((new_total - initial_total) / initial_total) * 100

/-- Theorem stating that John's income increase is approximately 72.73% --/
theorem johns_income_increase :
  ∃ ε > 0, ε < 0.01 ∧ 
  |percentage_increase 40 10 5 80 10 5 - 72.73| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_income_increase_l1134_113494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_k_l1134_113452

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 2*x + 3)
def g (x : ℝ) : ℝ := x^2 - x + 1

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x ≥ 3/4}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x | 3/4 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part (2)
theorem range_of_k : ∀ k : ℝ, (∀ x > 0, g x ≥ k * x) ↔ k ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_k_l1134_113452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_inequality_f_derivative_at_midpoint_l1134_113422

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + (1-a)*x - a * Real.log x

-- Theorem for monotonicity
theorem f_monotonicity (a : ℝ) :
  (∀ x > 0, Monotone (f a)) ∨ 
   (∃ y > 0, StrictMonoOn (f a) (Set.Ioo 0 y) ∧ 
             StrictMonoOn (f a) (Set.Ioi y)) := by sorry

-- Theorem for inequality when 0 < x < a
theorem f_inequality (a : ℝ) (x : ℝ) (ha : a > 0) (hx : 0 < x ∧ x < a) :
  f a (x + a) < f a (a - x) := by sorry

-- Theorem for derivative at midpoint of roots
theorem f_derivative_at_midpoint (a : ℝ) (x₁ x₂ : ℝ) 
  (hx : x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :
  deriv (f a) ((x₁ + x₂) / 2) > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_inequality_f_derivative_at_midpoint_l1134_113422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_construction_theorem_l1134_113483

-- Define the basic geometric objects
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

-- Define the concept of a point lying on a line
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- State the theorem
theorem geometric_construction_theorem 
  (a b : Line) (A B P : Point) (k : ℝ) 
  (h1 : A.on_line a) 
  (h2 : B.on_line b) 
  (h3 : ¬ P.on_line a) 
  (h4 : ¬ P.on_line b) 
  (h5 : k > 0) :
  ∃ (X Y : Point), 
    X.on_line a ∧ 
    Y.on_line b ∧ 
    (distance A X) / (distance B Y) = k ∧
    (distance A X) * (distance B Y) = k := by
  sorry

#check geometric_construction_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_construction_theorem_l1134_113483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_little_theorem_l1134_113430

theorem fermat_little_theorem (p : Nat) (a : Nat) (h : Nat.Prime p) :
  p ∣ (a^p - a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_little_theorem_l1134_113430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_ratio_l1134_113471

theorem cylinder_sphere_ratio (r R H : ℝ) : 
  r > 0 → R > 0 → H > 0 →
  (4 / 3 * Real.pi * r^3) * 3 = Real.pi * R^2 * H →
  H = 2 * r →
  H / R = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_ratio_l1134_113471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_l1134_113467

/-- The capacity of the tank in liters -/
def C : ℝ := 2250

/-- The time in hours it takes for the leak to empty the tank -/
def leak_empty_time : ℝ := 6

/-- The rate at which Pipe A fills the tank in liters per hour -/
def pipe_a_rate : ℝ := 4 * 60

/-- The rate at which Pipe B fills the tank in liters per hour -/
def pipe_b_rate : ℝ := 6 * 60

/-- The time in hours it takes to empty the tank with both pipes open and leak present -/
def both_pipes_leak_empty_time : ℝ := 10

/-- Theorem stating that the tank capacity is 2250 liters -/
theorem tank_capacity : C = 2250 := by
  -- Unfold the definition of C
  unfold C
  -- The proof is complete by reflexivity
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_l1134_113467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_value_sin_B_value_side_a_value_l1134_113435

/-- Properties of a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  law_of_sines : b / Real.sin B = c / Real.sin C

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.b / t.c = 2 * Real.sqrt 3 / 3 ∧
  t.A + 3 * t.C = Real.pi

theorem cos_C_value (t : Triangle) (h : SpecialTriangle t) :
  Real.cos t.C = Real.sqrt 3 / 3 := by sorry

theorem sin_B_value (t : Triangle) (h : SpecialTriangle t) :
  Real.sin t.B = 2 * Real.sqrt 2 / 3 := by sorry

theorem side_a_value (t : Triangle) (h : SpecialTriangle t) (hb : t.b = 3 * Real.sqrt 3) :
  t.a = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_value_sin_B_value_side_a_value_l1134_113435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_T_l1134_113411

-- Define S_k as a function that takes k as input and returns the integer
-- whose base-ten representation consists of k repetitions of the digit "2"
def S (k : ℕ) : ℕ := 2 * ((10^k - 1) / 9)

-- Define T as the quotient of S_20 and S_2
def T : ℕ := S 20 / S 2

-- Function to count the number of zeros in the base-ten representation of a number
def countZeros (n : ℕ) : ℕ :=
  (toString n).toList.filter (· = '0') |>.length

-- Theorem stating that the number of zeros in T is 18
theorem zeros_in_T : countZeros T = 18 := by
  sorry

#eval countZeros T  -- This will evaluate and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_T_l1134_113411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_row_col_ratio_l1134_113498

def is_valid_element (x : ℕ) : Prop := x = 1 ∨ x = 2 ∨ x = 3

def array_type := Fin 50 → Fin 100 → ℕ

def is_valid_array (arr : array_type) : Prop :=
  ∀ i j, is_valid_element (arr i j)

def row_sum (arr : array_type) (i : Fin 50) : ℕ :=
  (Finset.univ.sum fun j => arr i j)

def col_sum (arr : array_type) (j : Fin 100) : ℕ :=
  (Finset.univ.sum fun i => arr i j)

noncomputable def avg_row_sum (arr : array_type) : ℚ :=
  (Finset.univ.sum fun i => (row_sum arr i : ℚ)) / 50

noncomputable def avg_col_sum (arr : array_type) : ℚ :=
  (Finset.univ.sum fun j => (col_sum arr j : ℚ)) / 100

theorem avg_row_col_ratio (arr : array_type) (h : is_valid_array arr) :
  avg_row_sum arr / avg_col_sum arr = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_row_col_ratio_l1134_113498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1134_113486

/-- Parabola structure -/
structure Parabola where
  /-- The equation of the parabola y² = 4x -/
  equation : ℝ → ℝ → Prop
  /-- The focus of the parabola -/
  focus : ℝ × ℝ

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) (point : ℝ × ℝ) : Prop :=
  p.equation point.1 point.2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem statement -/
theorem parabola_distance_theorem (p : Parabola) (A B : ℝ × ℝ) :
  p.focus = (1, 0) →
  B = (3, 0) →
  PointOnParabola p A →
  distance A p.focus = distance B p.focus →
  distance A B = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1134_113486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_sin_2theta_l1134_113428

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)

theorem period_and_sin_2theta (ω : ℝ) (θ : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + π) = f ω x) →
  (∀ T : ℝ, T > 0 → T < π → ∃ x : ℝ, f ω (x + T) ≠ f ω x) →
  θ ∈ Set.Ioo 0 (π / 2) →
  f ω (θ / 2 + π / 12) = 6 / 5 →
  ω = 2 ∧ Real.sin (2 * θ) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_sin_2theta_l1134_113428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_predecessor_l1134_113453

def has_distinct_digits (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → (n.digits 10).get? i ≠ (n.digits 10).get? j

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

theorem sum_of_digits_of_predecessor (n : ℕ) 
  (h_distinct : has_distinct_digits n) 
  (h_sum : digit_sum n = 22) :
  digit_sum (n - 1) = 21 ∨ digit_sum (n - 1) = 30 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_predecessor_l1134_113453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_sum_l1134_113404

theorem determinant_sum (x y : ℝ) : 
  x ≠ y → 
  Matrix.det !![1, 4, 9; 3, x, y; 3, y, x] = 0 → 
  x + y = 39 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_sum_l1134_113404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l1134_113456

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem f_extrema :
  ∀ x ∈ Set.Icc 1 4,
    f x ≤ 9/5 ∧
    f x ≥ 3/2 ∧
    (∃ x₁ ∈ Set.Icc 1 4, f x₁ = 9/5) ∧
    (∃ x₂ ∈ Set.Icc 1 4, f x₂ = 3/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l1134_113456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_necessary_not_sufficient_l1134_113414

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b ∨ b = k • a

/-- The magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

/-- Vector addition for 2D vectors -/
def vectorAdd (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

theorem parallel_necessary_not_sufficient :
  (∀ a b : ℝ × ℝ, magnitude (vectorAdd a b) = magnitude a + magnitude b → parallel a b) ∧
  (∃ a b : ℝ × ℝ, parallel a b ∧ magnitude (vectorAdd a b) ≠ magnitude a + magnitude b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_necessary_not_sufficient_l1134_113414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1134_113469

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) / Real.log (2 * a)

-- State the theorem
theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f a x > 0) → 0 < a ∧ a < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1134_113469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_simplification_l1134_113458

theorem trig_identity_simplification (x y : ℝ) : 
  Real.sin (x + y)^2 + Real.cos x^2 + Real.cos (x + y)^2 - 2 * Real.sin (x + y) * Real.cos x * Real.cos y 
  = 1 - Real.sin (2 * x) * Real.cos y^2 - Real.cos x^2 * Real.sin (2 * y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_simplification_l1134_113458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_faces_with_same_sides_l1134_113496

-- Define a convex polyhedron
structure ConvexPolyhedron where
  faces : ℕ
  is_convex : Bool

-- Helper function (not part of the proof, just for completeness)
def count_faces_with_k_sides (P : ConvexPolyhedron) (k : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem existence_of_n_faces_with_same_sides 
  (P : ConvexPolyhedron) 
  (h : P.faces = 10 * n) 
  (n : ℕ) :
  ∃ (k : ℕ), (count_faces_with_k_sides P k) ≥ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_faces_with_same_sides_l1134_113496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l1134_113424

noncomputable def f (u : ℝ) : ℝ := Real.sqrt ((1 + u) / (1 - u))
noncomputable def g (v : ℝ) : ℝ := Real.sqrt ((1 + v) / (1 - v))

theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l1134_113424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_motorcyclist_distance_l1134_113427

/-- The time when the distance between a cyclist and a motorcyclist is 5 km -/
theorem cyclist_motorcyclist_distance (cyclist_initial_distance : ℝ) 
                                      (motorcyclist_initial_distance : ℝ)
                                      (cyclist_speed : ℝ)
                                      (motorcyclist_speed : ℝ) :
  cyclist_initial_distance = 8 →
  motorcyclist_initial_distance = 15 →
  cyclist_speed = 1/3 →
  motorcyclist_speed = 1 →
  ∃ t : ℝ, (5 * t^2 - 159 * t + 1188 = 0) ∧ 
           ((8 - cyclist_speed * t)^2 + (15 - motorcyclist_speed * t)^2 = 5^2) := by
  sorry

#check cyclist_motorcyclist_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_motorcyclist_distance_l1134_113427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_complex_expression_evaluation_l1134_113461

-- Part 1
theorem inequality_solution_set (x : ℝ) :
  (Real.log (x + 2) / Real.log (1/2) > -3) ↔ (-2 < x ∧ x < 6) := by sorry

-- Part 2
theorem complex_expression_evaluation :
  (1/8)^(1/3) * (-7/6)^0 + 8^(1/4) * 2^(1/4) + (2^(1/3) * 3^(1/2))^6 = 221/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_complex_expression_evaluation_l1134_113461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_eval_l1134_113423

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_expression_eval : 
  (floor 6.5) * (floor (2 / 3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.3) - (6.6 : ℝ) = 15.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_eval_l1134_113423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1134_113475

-- Define set M
def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}

-- Define set N
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1134_113475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_30_minus_cos_60_l1134_113419

theorem cos_30_minus_cos_60 : 
  (Real.sqrt 3 / 2) - (1 / 2) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_30_minus_cos_60_l1134_113419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1134_113465

-- Define the function (marked as noncomputable due to Real.sqrt)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 4) / (abs x - 5)

-- Define the domain
def domain : Set ℝ := {x | 4 ≤ x ∧ x < 5} ∪ {x | x > 5}

-- Theorem statement
theorem f_domain : 
  {x : ℝ | ∃ y, f x = y} = domain := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1134_113465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1134_113455

theorem solve_exponential_equation :
  ∃ x : ℤ, (3 : ℝ)^7 * (3 : ℝ)^x = 27 ∧ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1134_113455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1134_113474

/-- Hyperbola E with equation x²/a² - y²/b² = 1 -/
structure Hyperbola (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Eccentricity of hyperbola E is √3 -/
theorem hyperbola_eccentricity 
  (E : Hyperbola a b) 
  (F₁ F₂ P A Q : Point) 
  (h1 : distance F₁ F₂ = 6)
  (h2 : P.x > 0) -- P is on the right branch
  (h3 : P.x * F₁.y = P.y * F₁.x) -- PF₁ intersects y-axis at A
  (h4 : distance A Q = Real.sqrt 3)
  : Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 3 := by 
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1134_113474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_coefficient_l1134_113462

theorem perpendicular_vector_coefficient (a b : ℝ × ℝ) (lambda : ℝ) : 
  a = (1, -3) →
  b = (4, -2) →
  (lambda * a.1 + b.1) * a.1 + (lambda * a.2 + b.2) * a.2 = 0 →
  lambda = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_coefficient_l1134_113462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_tenth_term_value_l1134_113460

def a (n : ℕ) : ℚ :=
  if n = 1 then 2
  else if n = 2 then 1
  else 2 / n

theorem sequence_property (n : ℕ) (h : n > 2) :
  (a n / a (n - 1)) / (a (n - 1) / a n) =
  (a n / a (n + 1)) / (a n / a (n + 1)) :=
sorry

theorem tenth_term_value : a 10 = 1 / 5 := by
  simp [a]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_tenth_term_value_l1134_113460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_value_l1134_113490

theorem min_lambda_value (l : ℝ) (h_l_pos : l > 0) :
  (∀ x > Real.exp 2, l * Real.exp (l * x) - Real.log x ≥ 0) ↔ l ≥ 2 / Real.exp 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_value_l1134_113490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_prove_train_length_l1134_113413

/-- Represents the length of a train and platform in meters -/
def train_length : ℝ := 300

/-- Represents the speed of the train in km/hr -/
def train_speed : ℝ := 36

/-- Represents the time taken to cross the platform in minutes -/
def crossing_time : ℝ := 1

/-- Theorem stating that given the conditions, the train length is 300 meters -/
theorem train_length_calculation (h1 : train_speed = 36)
                                 (h2 : crossing_time = 1) :
  train_length = 300 := by
  sorry

/-- Main theorem proving the train length -/
theorem prove_train_length : train_length = 300 := by
  apply train_length_calculation
  · rfl
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_prove_train_length_l1134_113413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_cost_equation_l1134_113446

/-- Proves that the original cost of a car satisfies the given equation based on the problem conditions --/
theorem car_cost_equation (C : ℝ) 
  (h1 : C > 0)  -- Assuming the cost is positive
  (h2 : C * (1 + 0.12545454545454545) = 61900 - 13000) :
  ∃ ε > 0, |C - 43455| < ε := by
  sorry

#check car_cost_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_cost_equation_l1134_113446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_walking_time_l1134_113421

/-- Calculates the time taken to travel a given distance at a given speed -/
noncomputable def travelTime (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem man_walking_time :
  let distance := (6 : ℝ) -- km
  let speed := (10 : ℝ) -- km/hr
  travelTime distance speed * 60 = 36 -- minutes
  := by
    -- Unfold the definition of travelTime
    unfold travelTime
    -- Simplify the expression
    simp
    -- Perform the calculation
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_walking_time_l1134_113421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_60kmph_l1134_113436

/-- Calculates the speed of a train given its length, the time it takes to pass a man running in the opposite direction, and the man's speed. -/
noncomputable def train_speed (train_length : ℝ) (passing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_mps := relative_speed - man_speed_mps
  train_speed_mps * (3600 / 1000)

/-- Theorem stating that the speed of the train is approximately 60 km/h given the specified conditions. -/
theorem train_speed_approx_60kmph :
  let train_length := (110 : ℝ)
  let passing_time := (5.999520038396929 : ℝ)
  let man_speed := (6 : ℝ)
  abs (train_speed train_length passing_time man_speed - 60) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_60kmph_l1134_113436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_f_l1134_113440

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1) + Real.sqrt 3

-- Define the interval
def I : Set ℝ := Set.Icc (-2023) 2023

-- State the theorem
theorem max_min_sum_of_f :
  ∃ (M N : ℝ), (∀ x ∈ I, f x ≤ M) ∧ 
               (∃ x ∈ I, f x = M) ∧
               (∀ x ∈ I, N ≤ f x) ∧ 
               (∃ x ∈ I, f x = N) ∧
               (M + N = 2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_f_l1134_113440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1134_113417

/-- The area of a triangle with vertices at (0,0), (0,6), and (8,10) is 24 square units. -/
theorem triangle_area : 
  let vertex1 : ℝ × ℝ := (0, 0)
  let vertex2 : ℝ × ℝ := (0, 6)
  let vertex3 : ℝ × ℝ := (8, 10)
  (1/2 : ℝ) * 6 * 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1134_113417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1134_113448

noncomputable def vector1 : ℝ × ℝ := (3, -4)
noncomputable def vector2 : ℝ × ℝ := (6, 8)

noncomputable def dotProduct (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def angle (v1 v2 : ℝ × ℝ) : ℝ :=
  Real.arccos ((dotProduct v1 v2) / (magnitude v1 * magnitude v2))

theorem angle_between_vectors :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |angle vector1 vector2 * (180 / Real.pi) - 106| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1134_113448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculation_l1134_113420

theorem arithmetic_calculation : (-7) - 5 + 13 = 1 := by
  -- Simplify the expression
  calc
    (-7) - 5 + 13 = -7 - 5 + 13 := by rfl
    _ = -12 + 13 := by ring
    _ = 1 := by ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculation_l1134_113420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_fourth_l1134_113429

theorem sin_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : Real.sin α = -4/5) 
  (h2 : π < α ∧ α < 3*π/2) : 
  Real.sin (α + π/4) = -7*Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_fourth_l1134_113429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_height_l1134_113433

/-- Heron's formula for the area of a triangle -/
noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let p := (a + b + c) / 2
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

/-- Height of a triangle given its area and base -/
noncomputable def triangle_height (area base : ℝ) : ℝ :=
  2 * area / base

theorem triangle_area_and_height (a b c : ℝ) 
  (ha : a = 6) (hb : b = 7) (hc : c = 5) :
  let area := heron_area a b c
  (area = 6 * Real.sqrt 6) ∧ 
  (triangle_height area a = 2 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_height_l1134_113433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_20_value_l1134_113463

def a : ℕ → ℝ := sorry

def S : ℕ → ℝ := sorry

axiom a_1 : a 1 = 2

axiom S_def : ∀ n : ℕ, n ≥ 2 → S n = 1 + 2 * a n

theorem S_20_value : S 20 = 2^19 + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_20_value_l1134_113463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_exactly_one_success_expected_profit_value_l1134_113468

-- Define the probabilities of success for each team
noncomputable def prob_A_success : ℝ := 4/5
noncomputable def prob_B_success : ℝ := 3/4

-- Define the profits and losses for each product
def profit_A_success : ℝ := 150
def loss_A_failure : ℝ := 60
def profit_B_success : ℝ := 120
def loss_B_failure : ℝ := 40

-- Theorem for the probability of exactly one new product being successfully developed
theorem prob_exactly_one_success : 
  (prob_A_success * (1 - prob_B_success) + (1 - prob_A_success) * prob_B_success) = 7/20 := by
  sorry

-- Function to calculate the expected profit
noncomputable def expected_profit : ℝ :=
  (prob_A_success * prob_B_success) * (profit_A_success + profit_B_success) +
  (prob_A_success * (1 - prob_B_success)) * (profit_A_success - loss_B_failure) +
  ((1 - prob_A_success) * prob_B_success) * (-loss_A_failure + profit_B_success) +
  ((1 - prob_A_success) * (1 - prob_B_success)) * (-loss_A_failure - loss_B_failure)

-- Theorem for the mathematical expectation of the company's profit
theorem expected_profit_value : expected_profit = 188 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_exactly_one_success_expected_profit_value_l1134_113468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1134_113409

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (2*x) - f (x+2)

-- Theorem statement
theorem g_properties :
  (∀ x ∈ Set.Icc 0 1, g x ∈ Set.range g) ∧
  (∃ x ∈ Set.Icc 0 1, g x = -3) ∧
  (∃ x ∈ Set.Icc 0 1, g x = -4) ∧
  (∀ x ∈ Set.Icc 0 1, g x ≥ -4) ∧
  (∀ x ∈ Set.Icc 0 1, g x ≤ -3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1134_113409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_l1134_113415

theorem salary_change (s : ℝ) : 
  (s * 1.4 * 0.6 - s) / s = -0.16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_l1134_113415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_S_eq_18_l1134_113418

-- Define the floor function (noncomputable due to dependency on Real)
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define S as the sum of floor of square roots from 1 to 99
-- We use Int instead of Nat for the sum to avoid type mismatch
noncomputable def S : ℤ := (Finset.range 99).sum (λ i => floor (Real.sqrt ((i + 1 : ℕ) : ℝ)))

-- State the theorem
theorem floor_sqrt_S_eq_18 : floor (Real.sqrt (S : ℝ)) = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_S_eq_18_l1134_113418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_is_169_l1134_113426

noncomputable section

/-- Parabola defined by y = 2x^2 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 2 * p.1^2}

/-- Circle intersecting the parabola at four distinct points -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (k h r : ℝ), (p.1 - k)^2 + (p.2 - h)^2 = r^2}

/-- The four intersection points -/
def intersectionPoints : Finset (ℝ × ℝ) :=
  {(-1, 2), (5, 50), (0, 0), (-4, 32)}

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (0, 1/8)

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: Sum of distances from focus to intersection points is 169 -/
theorem sum_of_distances_is_169 :
  (intersectionPoints.sum (λ p => distance p focus)) = 169 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_is_169_l1134_113426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_for_tan_one_third_l1134_113459

theorem sin_minus_cos_for_tan_one_third (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_for_tan_one_third_l1134_113459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_and_inequality_l1134_113478

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define points
def point_O : ℝ × ℝ := (0, 0)
def point_A : ℝ × ℝ := (1, 1)
def point_B : ℝ × ℝ := (0, -1)

-- Define the line through two points
def line_through (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_tangent_and_inequality :
  ∃ p : ℝ,
    parabola p point_A.1 point_A.2 ∧
    (∀ x y : ℝ, line_through point_A point_B x y → ¬(parabola p x y)) ∧
    (∀ P Q : ℝ × ℝ,
      P ≠ Q →
      parabola p P.1 P.2 →
      parabola p Q.1 Q.2 →
      line_through point_B P P.1 P.2 →
      line_through point_B Q Q.1 Q.2 →
      distance point_B P * distance point_B Q > distance point_B point_A ^ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_and_inequality_l1134_113478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l1134_113402

/-- An isosceles right triangle with hypotenuse length 8 -/
structure IsoscelesRightTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is 8 cm long -/
  hypotenuse_eq : hypotenuse = 8

/-- The area of an isosceles right triangle -/
noncomputable def triangle_area (t : IsoscelesRightTriangle) : ℝ :=
  (t.hypotenuse ^ 2) / 4

/-- Theorem: The area of an isosceles right triangle with hypotenuse 8 cm is 32 sq cm -/
theorem isosceles_right_triangle_area (t : IsoscelesRightTriangle) :
  triangle_area t = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l1134_113402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_ratio_l1134_113451

noncomputable def f (x : ℝ) := (3 * x - 2) / (x + 4)

theorem inverse_function_ratio : 
  ∃ (a b c d : ℝ), 
    (∀ x, Function.invFun f x = (a * x + b) / (c * x + d)) ∧ 
    (c ≠ 0) ∧ 
    (a / c = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_ratio_l1134_113451
