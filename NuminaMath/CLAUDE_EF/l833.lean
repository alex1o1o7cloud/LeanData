import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1992_power_2_1991_l833_83387

-- Define f₁(k) as the square of the sum of the digits of k in decimal notation
def f1 (k : ℕ) : ℕ :=
  (Nat.digits 10 k).sum ^ 2

-- Define fₙ(k) recursively for n > 1
def f (n : ℕ) (k : ℕ) : ℕ :=
  match n with
  | 0 => k
  | n+1 => f1 (f n k)

-- Theorem statement
theorem f_1992_power_2_1991 : f 1992 (2^1991) = 256 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1992_power_2_1991_l833_83387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_has_small_prime_factor_l833_83398

theorem composite_has_small_prime_factor (N : ℕ) (h : ¬ Prime N) (h_pos : N > 1) :
  ∃ (p : ℕ), Prime p ∧ p ∣ N ∧ p ≤ N.sqrt := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_has_small_prime_factor_l833_83398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_miles_per_tankful_l833_83318

noncomputable def miles_per_tankful_city (highway_miles_per_tankful : ℝ) (city_mpg_difference : ℝ) (city_mpg : ℝ) : ℝ :=
  let highway_mpg := city_mpg + city_mpg_difference
  let tank_size := highway_miles_per_tankful / highway_mpg
  tank_size * city_mpg

theorem car_city_miles_per_tankful 
  (highway_miles_per_tankful : ℝ) 
  (city_mpg_difference : ℝ) 
  (city_mpg : ℝ) 
  (h1 : highway_miles_per_tankful = 462)
  (h2 : city_mpg_difference = 3)
  (h3 : city_mpg = 8) :
  miles_per_tankful_city highway_miles_per_tankful city_mpg_difference city_mpg = 336 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_miles_per_tankful_l833_83318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andys_shift_earnings_l833_83372

/-- Calculates Andy's earnings for a shift at the tennis resort pro shop. -/
def andys_earnings (hourly_rate : ℝ) (hours_worked : ℝ) (racquet_stringing_fee : ℝ) 
  (racquets_strung : ℕ) (grommet_change_fee : ℝ) (grommets_changed : ℕ) 
  (stencil_fee : ℝ) (stencils_painted : ℕ) : ℝ :=
  hourly_rate * hours_worked + 
  racquet_stringing_fee * (racquets_strung : ℝ) + 
  grommet_change_fee * (grommets_changed : ℝ) + 
  stencil_fee * (stencils_painted : ℝ)

/-- Theorem stating Andy's earnings for the given shift. -/
theorem andys_shift_earnings : 
  andys_earnings 9 8 15 7 10 2 1 5 = 202 := by
  -- Unfold the definition of andys_earnings
  unfold andys_earnings
  -- Perform the calculation
  simp [Nat.cast_add, Nat.cast_mul]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andys_shift_earnings_l833_83372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_2alpha_l833_83326

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (π / 2 - α) = 1 / 3) :
  Real.cos (π - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_2alpha_l833_83326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_condition_l833_83375

/-- The eccentricity of a hyperbola with equation x²/a² - y²/4 = 1 -/
noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt (1 + 4 / (a * a))

theorem hyperbola_eccentricity_condition (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 1 → eccentricity a > Real.sqrt 2) ∧
  ∃ a > 0, eccentricity a > Real.sqrt 2 ∧ a ≥ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_condition_l833_83375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_iff_m_range_iff_l833_83311

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |2*x - 1| - |x + 3/2|

-- Theorem for part (1)
theorem f_negative_iff (x : ℝ) : f x < 0 ↔ -1/6 < x ∧ x < 5/2 := by sorry

-- Theorem for part (2)
theorem m_range_iff (m : ℝ) : (∃ x₀ : ℝ, f x₀ + 3*m^2 < 5*m) ↔ -1/3 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_iff_m_range_iff_l833_83311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complementB_l833_83377

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℤ := {x : ℤ | x^2 < 16}

-- Define set B
def B : Set ℝ := {x : ℝ | x - 1 ≤ 0}

-- Define the complement of B in U
def complementB : Set ℝ := U \ B

-- State the theorem
theorem intersection_A_complementB : A ∩ (complementB.preimage Int.cast) = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complementB_l833_83377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_negative_one_l833_83353

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The given function f defined for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := x^2 - 1/x

theorem f_derivative_at_negative_one :
  ∀ f : ℝ → ℝ,
  EvenFunction f →
  (∀ x > 0, f x = x^2 - 1/x) →
  HasDerivAt f (-3) (-1) := by
  sorry

#check f_derivative_at_negative_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_negative_one_l833_83353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_sum_l833_83361

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem tangent_slope_sum (α : ℝ) : 
  (deriv f) 1 = Real.tan α → 0 < α → α < Real.pi / 2 → Real.cos α + Real.sin α = 2 * Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_sum_l833_83361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l833_83310

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*x + y^2 - 8*y + 12 = 0

/-- The line equation -/
def line_equation (y : ℝ) : Prop :=
  y = 3

/-- The area of the circle above the line -/
noncomputable def area_above_line : ℝ :=
  (10 * Real.pi / 3) + Real.sqrt 3

/-- Theorem stating that the area above the line is correct -/
theorem circle_area_above_line :
  ∃ (x y : ℝ), circle_equation x y ∧ line_equation y →
  area_above_line = (10 * Real.pi / 3) + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l833_83310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_zero_and_B_equals_one_l833_83303

-- Define the angles
variable (θ α β γ : ℝ)

-- Define A
noncomputable def A (θ α β γ : ℝ) : ℝ :=
  (Real.sin (θ - β) * Real.sin (θ - γ)) / (Real.sin (α - β) * Real.sin (α - γ)) * Real.sin (2 * (θ - α)) +
  (Real.sin (θ - γ) * Real.sin (θ - α)) / (Real.sin (β - γ) * Real.sin (β - α)) * Real.sin (2 * (θ - β)) +
  (Real.sin (θ - α) * Real.sin (θ - β)) / (Real.sin (γ - α) * Real.sin (γ - β)) * Real.sin (2 * (θ - γ))

-- Define B
noncomputable def B (θ α β γ : ℝ) : ℝ :=
  (Real.sin (θ - β) * Real.sin (θ - γ)) / (Real.sin (α - β) * Real.sin (α - γ)) * Real.cos (2 * (θ - α)) +
  (Real.sin (θ - γ) * Real.sin (θ - α)) / (Real.sin (β - γ) * Real.sin (β - α)) * Real.cos (2 * (θ - β)) +
  (Real.sin (θ - α) * Real.sin (θ - β)) / (Real.sin (γ - α) * Real.sin (γ - β)) * Real.cos (2 * (θ - γ))

-- Theorem statement
theorem A_equals_zero_and_B_equals_one (θ α β γ : ℝ) : A θ α β γ = 0 ∧ B θ α β γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_zero_and_B_equals_one_l833_83303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l833_83379

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Define line l passing through focus and intersecting parabola at A and B
def line_l (k x y : ℝ) : Prop := y = k*(x - 1)

-- Define point A on the parabola
def point_A (k x y : ℝ) : Prop := parabola x y ∧ line_l k x y

-- Define point B on the parabola
def point_B (k x y x_a y_a : ℝ) : Prop := parabola x y ∧ line_l k x y ∧ ¬(x = x_a ∧ y = y_a)

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem parabola_properties :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), point_A k x y ∧ distance x y 1 0 = 4 → (x = 3 ∧ (y = 2 ∨ y = -2))) ∧
  (∃ (x1 y1 x2 y2 : ℝ), point_A k x1 y1 ∧ point_B k x2 y2 x1 y1 ∧ distance x1 y1 x2 y2 = 5 → k = 2 ∨ k = -2) ∧
  (∃ (x y : ℝ), parabola x y ∧ 
    (∀ (x' y' : ℝ), parabola x' y' → 
      distance x y ((2*x - y + 4)/5) ((4*x + 2*y - 8)/5) ≤ 
      distance x' y' ((2*x' - y' + 4)/5) ((4*x' + 2*y' - 8)/5)) →
    x = 0.25 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l833_83379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_success_is_five_twelfths_min_people_for_99_percent_l833_83341

noncomputable def prob_A : ℝ := 1/3
noncomputable def prob_B : ℝ := 1/4

noncomputable def prob_one_success : ℝ := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B

noncomputable def prob_at_least_one (n : ℕ) : ℝ := 1 - (1 - prob_B)^n

theorem prob_one_success_is_five_twelfths : 
  prob_one_success = 5/12 := by sorry

theorem min_people_for_99_percent :
  ∀ n : ℕ, (prob_at_least_one n ≥ 99/100 ∧ 
    ∀ m : ℕ, m < n → prob_at_least_one m < 99/100) → n = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_success_is_five_twelfths_min_people_for_99_percent_l833_83341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_sufficient_not_necessary_for_tan_2alpha_l833_83373

theorem alpha_sufficient_not_necessary_for_tan_2alpha (α : ℝ) : 
  (α = π / 6 → Real.tan (2 * α) = Real.sqrt 3) ∧ 
  ¬(Real.tan (2 * α) = Real.sqrt 3 → α = π / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_sufficient_not_necessary_for_tan_2alpha_l833_83373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_length_l833_83356

-- Define the function f(x) = |log₃(x)|
noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 3|

-- State the theorem
theorem min_interval_length (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc 0 1) →
  (∃ x ∈ Set.Icc a b, f x = 0) →
  (∃ x ∈ Set.Icc a b, f x = 1) →
  b - a ≥ 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_length_l833_83356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l833_83390

/-- Calculates the time (in seconds) for two trains to cross each other -/
noncomputable def trainCrossingTime (trainLength : ℝ) (trainSpeed : ℝ) : ℝ :=
  let relativeSpeed := 2 * trainSpeed * (1000 / 3600)  -- Convert km/hr to m/s and double for relative speed
  let totalDistance := 2 * trainLength  -- Total distance is sum of both train lengths
  totalDistance / relativeSpeed

theorem trains_crossing_time :
  let trainLength : ℝ := 120  -- Length of each train in meters
  let trainSpeed : ℝ := 108   -- Speed of each train in km/hr
  trainCrossingTime trainLength trainSpeed = 4 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l833_83390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_satisfies_conditions_l833_83376

/-- An equilateral hyperbola passing through (3, 1) with axes of symmetry on coordinate axes -/
def equilateral_hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 8}

/-- The point (3, 1) -/
def point_A : ℝ × ℝ := (3, 1)

/-- Theorem stating that the defined hyperbola satisfies all conditions -/
theorem hyperbola_satisfies_conditions :
  (point_A ∈ equilateral_hyperbola) ∧
  (∀ (x y : ℝ), (x, y) ∈ equilateral_hyperbola ↔ x^2 - y^2 = 8) ∧
  (∀ (x : ℝ), (x, 0) ∈ equilateral_hyperbola ↔ x = Real.sqrt 8 ∨ x = -Real.sqrt 8) ∧
  (∀ (y : ℝ), (0, y) ∈ equilateral_hyperbola ↔ y = Real.sqrt 8 ∨ y = -Real.sqrt 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_satisfies_conditions_l833_83376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l833_83343

-- Define polynomials f, g, and h
def f (a : Fin 6 → ℤ) (x : ℤ) : ℤ :=
  (a 5) * x^5 + (a 4) * x^4 + (a 3) * x^3 + (a 2) * x^2 + (a 1) * x + (a 0)

def g (b : Fin 4 → ℤ) (x : ℤ) : ℤ :=
  (b 3) * x^3 + (b 2) * x^2 + (b 1) * x + (b 0)

def h (c : Fin 3 → ℤ) (x : ℤ) : ℤ :=
  (c 2) * x^2 + (c 1) * x + (c 0)

-- Define the theorem
theorem polynomial_equality
  (a : Fin 6 → ℤ) (b : Fin 4 → ℤ) (c : Fin 3 → ℤ)
  (h1 : ∀ i, |a i| ≤ 4)
  (h2 : ∀ i, |b i| ≤ 1)
  (h3 : ∀ i, |c i| ≤ 1)
  (h4 : f a 10 = g b 10 * h c 10) :
  ∀ x, f a x = g b x * h c x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l833_83343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_is_barium_chloride_l833_83306

/-- Represents a chemical element --/
structure Element where
  name : String
  atomic_mass : ℝ

/-- Represents a chemical compound --/
structure Compound where
  elements : List Element
  mass_percentages : List ℝ

/-- Calculates the moles of an element given its mass and atomic mass --/
noncomputable def moles (mass : ℝ) (atomic_mass : ℝ) : ℝ :=
  mass / atomic_mass

/-- Calculates the empirical formula ratio for a two-element compound --/
noncomputable def empirical_formula_ratio (m1 m2 : ℝ) : ℚ × ℚ :=
  let ratio1 := m1 / min m1 m2
  let ratio2 := m2 / min m1 m2
  (↑(round ratio1), ↑(round ratio2))

/-- The main theorem stating that the given compound is BaCl₂ --/
theorem compound_is_barium_chloride (Ba Cl : Element) (compound : Compound) :
  Ba.name = "Barium" →
  Cl.name = "Chlorine" →
  Ba.atomic_mass = 137.33 →
  Cl.atomic_mass = 35.45 →
  compound.elements = [Ba, Cl] →
  compound.mass_percentages = [66.18, 33.82] →
  empirical_formula_ratio
    (moles 66.18 Ba.atomic_mass)
    (moles 33.82 Cl.atomic_mass) = (1, 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_is_barium_chloride_l833_83306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l833_83308

noncomputable def f (x : ℝ) : ℝ := Real.log (|x - 2| + 1)

theorem f_properties :
  (∀ x, f (x + 2) = f (-x + 2)) ∧ 
  (∀ x y, x < y → x < 2 → y < 2 → f x > f y) ∧
  (∀ x y, x < y → x > 2 → y > 2 → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l833_83308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_l833_83345

-- Define the ellipse
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the proposed tangent line to the ellipse
def proposed_ellipse_tangent (x y x₀ y₀ a b : ℝ) : Prop :=
  x₀ * x / a^2 + y₀ * y / b^2 = 1

-- State the theorem
theorem ellipse_tangent (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (ellipse x₀ y₀ a b) →
  (∀ x y, proposed_ellipse_tangent x y x₀ y₀ a b → ellipse x y a b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_l833_83345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l833_83335

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def number_of_circular_arrangements (n : ℕ) : ℕ := factorial (n - 1)

theorem circular_table_seating (n : ℕ) (h : n > 1) :
  (factorial (n - 1)) = number_of_circular_arrangements n := by
  rfl

#eval factorial 9  -- Should output 362880

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l833_83335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_circular_sector_l833_83337

theorem cone_volume_from_circular_sector (r : ℝ) (h : r = 6) :
  let remaining_area_ratio : ℝ := 5/6
  let base_radius : ℝ := r * remaining_area_ratio
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let cone_volume : ℝ := (1/3) * Real.pi * base_radius^2 * cone_height
  cone_volume = (25 * Real.sqrt 11 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_circular_sector_l833_83337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_80_over_31_l833_83317

/-- Represents the hiking scenario with Chantal and Jean --/
structure HikingScenario where
  d : ℝ  -- Distance from trailhead to hilltop
  chantal_ascent_speed : ℝ  -- Chantal's speed while ascending
  chantal_descent_speed : ℝ  -- Chantal's speed while descending
  rest_time : ℝ  -- Chantal's rest time at the hilltop in hours
  meeting_point : ℝ  -- Fraction of the way back where they meet

/-- Calculates Jean's average speed given the hiking scenario --/
noncomputable def jean_average_speed (scenario : HikingScenario) : ℝ :=
  2 * scenario.d / (scenario.d / scenario.chantal_ascent_speed + scenario.rest_time + 
    (2 * scenario.d - scenario.meeting_point * 2 * scenario.d) / scenario.chantal_descent_speed)

/-- Theorem stating that Jean's average speed is 80/31 miles per hour --/
theorem jean_speed_is_80_over_31 (scenario : HikingScenario) 
    (h1 : scenario.chantal_ascent_speed = 5)
    (h2 : scenario.chantal_descent_speed = 4)
    (h3 : scenario.rest_time = 1/4)
    (h4 : scenario.meeting_point = 3/4) :
  jean_average_speed scenario = 80/31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_80_over_31_l833_83317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_speed_approximately_12_l833_83355

-- Define the parameters of the problem
def total_distance : ℚ := 22
def first_part_distance : ℚ := 10
def second_part_distance : ℚ := 12
def second_part_speed : ℚ := 10
def total_average_speed : ℚ := 10.82

-- Define the function to calculate the first part speed
noncomputable def calculate_first_part_speed (d1 d2 v2 v_avg : ℚ) : ℚ :=
  let v1 := (d1 + d2) / (d1 / v_avg + d2 / v2)
  (d1 + d2) / ((d1 / v1) + (d2 / v2))

-- Theorem statement
theorem first_part_speed_approximately_12 :
  ∃ ε : ℚ, ε > 0 ∧ ε < 0.1 ∧ 
  |calculate_first_part_speed first_part_distance second_part_distance second_part_speed total_average_speed - 12| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_speed_approximately_12_l833_83355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_of_n_multiple_of_300_l833_83331

/-- The number of natural-number factors of 2^10 * 3^14 * 5^8 that are multiples of 300 -/
def count_factors : ℕ := 882

/-- The prime factorization of n -/
def n : ℕ := 2^10 * 3^14 * 5^8

/-- Theorem stating that the count of factors of n that are multiples of 300 is correct -/
theorem count_factors_of_n_multiple_of_300 :
  (Finset.filter (fun x => x ∣ n ∧ 300 ∣ x) (Finset.range (n + 1))).card = count_factors :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_of_n_multiple_of_300_l833_83331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_and_b_range_l833_83383

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((1 + a * x) / (1 - 2 * x)) / Real.log 10

theorem odd_function_implies_a_and_b_range (a b : ℝ) 
  (h1 : a ≠ -2)
  (h2 : ∀ x ∈ Set.Ioo (-b) b, f a x = -f a (-x))
  (h3 : ∀ x ∈ Set.Ioo (-b) b, (1 + a * x) / (1 - 2 * x) > 0) :
  a = 2 ∧ 0 < b ∧ b ≤ 1/2 ∧ 1 < a^b ∧ a^b ≤ Real.sqrt 2 :=
by
  sorry

#check odd_function_implies_a_and_b_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_and_b_range_l833_83383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrong_mark_calculation_l833_83357

theorem wrong_mark_calculation (n : ℕ) (initial_avg correct_avg correct_mark : ℝ) : 
  n = 10 ∧ 
  initial_avg = 100 ∧ 
  correct_avg = 92 ∧ 
  correct_mark = 10 → 
  ∃ wrong_mark : ℝ, 
    wrong_mark = 90 ∧ 
    n * initial_avg = (n - 1) * correct_avg + wrong_mark ∧
    n * correct_avg = (n - 1) * correct_avg + correct_mark :=
by
  sorry

#check wrong_mark_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrong_mark_calculation_l833_83357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_to_slope_intercept_form_l833_83392

/-- Given a line equation, prove its slope-intercept form -/
theorem line_equation_to_slope_intercept_form :
  ∃ (m b : ℝ), (∀ x y : ℝ, 3 * (x - 4) + 7 * (y - 14) = 0 ↔ y = m * x + b) ∧ m = -3/7 ∧ b = 110/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_to_slope_intercept_form_l833_83392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_34992_l833_83349

/-- The number of positive factors of 34992 -/
def number_of_factors : ℕ := 40

/-- The number we're analyzing -/
def n : ℕ := 34992

theorem factors_of_34992 : 
  (Finset.filter (λ i : ℕ ↦ n % i = 0) (Finset.range (n + 1))).card = number_of_factors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_34992_l833_83349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_diff_l833_83388

-- Define the quadratic functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the relationship between f and g
axiom g_def : ∀ x, g x = -3 * f (200 - x)

-- Define that the vertex of g is the same as the vertex of f
axiom same_vertex : ∃ v : ℝ, (∀ x, f x ≤ f v) ∧ (∀ x, g x ≤ g v)

-- Define the x-intercepts
noncomputable def a₁ : ℝ := sorry
noncomputable def a₂ : ℝ := sorry
noncomputable def a₃ : ℝ := sorry
noncomputable def a₄ : ℝ := sorry

-- Define the order of x-intercepts
axiom x_intercepts_order : a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄

-- Define the condition for a₃ and a₂
axiom a₃_a₂_diff : a₃ - a₂ = 300

-- The theorem to prove
theorem x_intercepts_diff : a₄ - a₁ = 200 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_diff_l833_83388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l833_83324

def condition_p (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (4*a - 3)*x₁ + 1/4 = 0 ∧ x₂^2 + (4*a - 3)*x₂ + 1/4 = 0

def condition_q (a : ℝ) : Prop :=
  let z : ℂ := Complex.ofReal ((a + 1) / 2) + Complex.I * ((1 - a) / 2)
  Complex.re z > 0 ∧ Complex.im z > 0

def range_a (a : ℝ) : Prop :=
  a ≤ -1 ∨ (a ≥ 1/2 ∧ a ≠ 1)

theorem a_range_theorem (a : ℝ) :
  ((condition_p a ∨ condition_q a) ∧ ¬(condition_p a ∧ condition_q a)) → range_a a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l833_83324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chef_planned_six_cups_l833_83393

/-- A recipe for brownies with butter and coconut oil substitution. -/
structure Recipe where
  butter_per_cup : ℚ  -- ounces of butter per cup of baking mix
  oil_sub_ratio : ℚ   -- ratio of coconut oil to butter substitution

/-- The chef's available ingredients and usage. -/
structure ChefIngredients where
  butter_available : ℚ  -- ounces of butter available
  oil_used : ℚ          -- ounces of coconut oil used

/-- Calculate the total cups of baking mix given a recipe and chef's ingredients. -/
def total_baking_mix (recipe : Recipe) (ingredients : ChefIngredients) : ℚ :=
  ingredients.butter_available / recipe.butter_per_cup +
  (ingredients.oil_used / recipe.oil_sub_ratio) / recipe.butter_per_cup

/-- Theorem stating that the chef planned to use 6 cups of baking mix. -/
theorem chef_planned_six_cups (recipe : Recipe) (ingredients : ChefIngredients) : 
  recipe.butter_per_cup = 2 →
  recipe.oil_sub_ratio = 1 →
  ingredients.butter_available = 4 →
  ingredients.oil_used = 8 →
  total_baking_mix recipe ingredients = 6 := by
  intro h1 h2 h3 h4
  unfold total_baking_mix
  simp [h1, h2, h3, h4]
  norm_num

#eval total_baking_mix ⟨2, 1⟩ ⟨4, 8⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chef_planned_six_cups_l833_83393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l833_83320

theorem trigonometric_equation_solution (x : ℝ) :
  (Real.sin x ≠ 0 ∧ Real.sin (2*x) ≠ 0 ∧ Real.sin (4*x) ≠ 0) →
  (1 / Real.sin x - 1 / Real.sin (2*x) = 2 / Real.sin (4*x)) ↔
  (∃ n : ℤ, x = 2*Real.pi/3 + 2*n*Real.pi ∨ x = -2*Real.pi/3 + 2*n*Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l833_83320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l833_83325

-- Define the points A and B
def A : ℚ × ℚ := (0, 0)
def B : ℚ × ℚ := (30, 10)

-- Define the set of possible points C
def C : Set (ℤ × ℤ) := {p | ∃ (x y : ℤ), p = (x, y) ∧ y = 2 * x - 5}

-- Define the area function for a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Theorem statement
theorem min_triangle_area :
  ∃ (min_area : ℚ), min_area = 15 ∧
    ∀ (c : ℤ × ℤ), c ∈ C →
      triangleArea A B (↑c.1, ↑c.2) ≥ min_area :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l833_83325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_relationship_l833_83336

/-- Represents the dimensions and painting cost of a rectangular box. -/
structure Box where
  width : ℝ
  length : ℝ
  height : ℝ
  floorPaintCost : ℝ
  paintRate : ℝ

/-- Calculates the cost to paint all sides of the box. -/
def paintAllSidesCost (b : Box) : ℝ :=
  2 * (b.length * b.width + b.width * b.height + b.length * b.height) * b.paintRate

/-- Theorem stating the relationship between floor painting cost and all sides painting cost. -/
theorem paint_cost_relationship (b : Box) 
    (h1 : b.length = 3 * b.width)
    (h2 : b.length = 1.5 * b.height)
    (h3 : b.floorPaintCost = 400)
    (h4 : b.paintRate = 3) :
  ∃ (ε : ℝ), abs (paintAllSidesCost b - 2933.04) < ε ∧ ε > 0 := by
  sorry

#check paint_cost_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_relationship_l833_83336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_squares_of_product_eleven_l833_83309

theorem sum_reciprocal_squares_of_product_eleven (a b : ℕ) 
  (h1 : Nat.Coprime a b) 
  (h2 : a * b = 11) : 
  (1 : ℚ) / (a^2 : ℚ) + (1 : ℚ) / (b^2 : ℚ) = 122 / 121 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_squares_of_product_eleven_l833_83309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l833_83396

/-- The set of possible values for 'a' given the conditions of the problem -/
def solution_set : Set ℝ := {-3, -1, 1, 3}

/-- Definition of point A -/
def A : ℝ × ℝ := (1, 0)

/-- Definition of point B -/
def B : ℝ × ℝ := (4, 0)

/-- Definition of the circle with center (a, 0) and radius 1 -/
def circle_eq (a : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - a)^2 + P.2^2 = 1

/-- Definition of the distance ratio condition -/
def distance_ratio (P : ℝ × ℝ) : Prop :=
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = 1/4 * ((P.1 - B.1)^2 + (P.2 - B.2)^2)

/-- The main theorem stating the problem -/
theorem problem_statement (a : ℝ) : 
  (∃! P, circle_eq a P ∧ distance_ratio P) → a ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l833_83396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equality_l833_83322

theorem exponent_equality (m : ℤ) (h1 : ((-2 : ℚ)^(2*m) = (2 : ℚ)^(21-m))) (h2 : m = 7) :
  (2 : ℚ)^(21-m) = (2 : ℚ)^14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equality_l833_83322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_sum_ratio_l833_83368

theorem percentage_sum_ratio (X : ℝ) (h : X > 0) : 
  (0.125 * X + 0.225 * X) / (0.375 * X) * 100 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_sum_ratio_l833_83368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l833_83329

/-- The sum of the infinite series 1 + 2(1/999) + 3(1/999)² + 4(1/999)³ + ... -/
noncomputable def infiniteSeries : ℝ := ∑' n, n * (1 / 999) ^ (n - 1)

/-- Theorem: The sum of the infinite series equals 1000/998 -/
theorem infiniteSeriesSum : infiniteSeries = 1000 / 998 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l833_83329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_divisible_by_five_and_three_l833_83300

theorem percent_divisible_by_five_and_three (n : ℕ) : 
  (↑(Finset.filter (λ x : ℕ ↦ x % 5 = 0 ∧ x % 3 = 0) (Finset.range (n + 1))).card / ↑n) * 100 = (8 / 120) * 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_divisible_by_five_and_three_l833_83300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_series_convergence_l833_83350

open Real
open BigOperators

theorem double_series_convergence :
  (∑' (j : ℕ), ∑' (k : ℕ), (3 : ℝ) ^ (-(2 * k + j + (k + j)^2 : ℤ))) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_series_convergence_l833_83350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_l833_83301

/-- Represents an ellipse with semi-major axis a, semi-minor axis b, and focal distance c -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_c_eq : c^2 = a^2 - b^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- Theorem stating the minimum eccentricity of an ellipse under given conditions -/
theorem min_eccentricity (e : Ellipse) 
  (h_point : ∃ (P : ℝ × ℝ), (P.1^2 / e.a^2) + (P.2^2 / e.b^2) = 1 ∧ 
    (((P.1 + e.c)^2 + P.2^2).sqrt * ((P.1 - e.c)^2 + P.2^2).sqrt = 2 * e.c^2)) :
  eccentricity e ≥ Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_l833_83301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_fifty_l833_83374

theorem ceiling_sum_equals_fifty : 
  ⌈Real.sqrt (16/5 : ℝ)⌉ + ⌈(16/5 : ℝ)⌉ + ⌈((16/5 : ℝ)^2)⌉ + ⌈((16/5 : ℝ)^3)⌉ = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_fifty_l833_83374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_division_ratio_l833_83363

/-- Predicate for a regular hexagon -/
def regular_hexagon (A B C D E F : ℝ × ℝ) : Prop := sorry

/-- Predicate for a point being an internal point of a line segment -/
def is_internal_point (P A B : ℝ × ℝ) : Prop := sorry

/-- The ratio of vectors -/
noncomputable def vector_ratio (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Predicate for three points being collinear -/
def collinear (A B C : ℝ × ℝ) : Prop := sorry

/-- In a regular hexagon ABCDEF, if M divides AC internally, N divides CE internally, 
    AM : AC = CN : CE = r, and B, M, and N are collinear, then r = √3 / 3. -/
theorem hexagon_division_ratio (A B C D E F M N : ℝ × ℝ) (r : ℝ) : 
  regular_hexagon A B C D E F →
  is_internal_point M A C →
  is_internal_point N C E →
  vector_ratio A M A C = r →
  vector_ratio C N C E = r →
  collinear B M N →
  r = Real.sqrt 3 / 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_division_ratio_l833_83363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_l833_83312

theorem inequality_implies_log (x y : ℝ) :
  (2 : ℝ)^x - (2 : ℝ)^y < (3 : ℝ)^(-x) - (3 : ℝ)^(-y) → Real.log (y - x + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_l833_83312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_8pi_l833_83381

-- Define the points C and D
def C : ℝ × ℝ := (-2, 3)
def D : ℝ × ℝ := (2, -1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the circle area function
noncomputable def circleArea (r : ℝ) : ℝ := Real.pi * r^2

-- Theorem statement
theorem circle_area_is_8pi :
  circleArea (distance C D / 2) = 8 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_8pi_l833_83381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l833_83382

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x + 2*x + 1

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := x^2 * Real.exp x + 2*x * Real.exp x + 2

theorem tangent_line_intersection :
  let P : ℝ × ℝ := (0, 1)
  let k : ℝ := f' P.fst
  let b : ℝ := P.snd - k * P.fst
  let x_intersect : ℝ := -b / k
  x_intersect = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l833_83382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l833_83371

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point M
def point_M : ℝ × ℝ := (4, -1)

-- Define a tangent line to the circle
def is_tangent_line (x y a b : ℝ) : Prop := x * a + y * b = 4

-- Define the line passing through two points
def is_on_line_through_points (x y : ℝ) : Prop := 4 * x - y - 4 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_circle x₁ y₁ ∧ 
    is_on_circle x₂ y₂ ∧ 
    is_tangent_line x₁ y₁ point_M.1 point_M.2 ∧
    is_tangent_line x₂ y₂ point_M.1 point_M.2 ∧
    is_on_line_through_points x₁ y₁ ∧
    is_on_line_through_points x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l833_83371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_optimization_l833_83352

-- Define the fuel consumption function
noncomputable def fuel_consumption (x : ℝ) : ℝ := (1/128000) * x^3 - (3/80) * x + 8

-- Define the total fuel consumed for a 100 km trip
noncomputable def total_fuel (x : ℝ) : ℝ := (fuel_consumption x) * (100 / x)

-- State the theorem
theorem fuel_optimization :
  ∀ x : ℝ, 0 < x → x ≤ 120 →
  (total_fuel 40 = 17.5) ∧
  (∀ y : ℝ, 0 < y → y ≤ 120 → total_fuel y ≥ total_fuel 80) ∧
  (total_fuel 80 = 11.25) := by
  sorry

-- Additional lemmas to help with the proof
lemma fuel_consumption_continuous : Continuous fuel_consumption := by
  sorry

lemma total_fuel_continuous : Continuous total_fuel := by
  sorry

lemma total_fuel_differentiable : Differentiable ℝ total_fuel := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_optimization_l833_83352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l833_83394

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the circle (renamed to avoid conflict)
def circle_eq (x y r : ℝ) : Prop := (x-3)^2 + y^2 = r^2

-- Define the foci of the hyperbola
def foci : ℝ × ℝ × ℝ × ℝ := (-2, 0, 2, 0)

-- Define the dot product of vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

-- Define the condition for point P
def condition (x y : ℝ) : Prop :=
  let (f1x, f1y, f2x, f2y) := foci
  dot_product (x - f1x) (y - f1y) (x - f2x) (y - f2y) = 0

-- Main theorem
theorem hyperbola_circle_intersection :
  (∀ r : ℝ, r > 0 → 
    (∃ x y : ℝ, circle_eq x y r ∧ condition x y)) →
  (∀ r : ℝ, r ∈ Set.Icc 1 3 ↔ 
    (∃ x y : ℝ, circle_eq x y r ∧ condition x y)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l833_83394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFGH_is_three_l833_83315

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a quadrilateral in 3D space -/
structure Quadrilateral where
  E : Point3D
  F : Point3D
  G : Point3D
  H : Point3D

/-- The side length of the cube -/
noncomputable def cubeSideLength : ℝ := 3

/-- The quadrilateral EFGH formed by the intersection of the plane with the cube -/
noncomputable def EFGH : Quadrilateral := {
  E := { x := 0, y := 0, z := 0 },
  F := { x := cubeSideLength, y := 0, z := 0 },
  G := { x := cubeSideLength, y := cubeSideLength / 3, z := cubeSideLength },
  H := { x := 0, y := 2 * cubeSideLength / 3, z := cubeSideLength }
}

/-- Calculate the area of the quadrilateral EFGH -/
noncomputable def areaEFGH (q : Quadrilateral) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the area of EFGH is 3 -/
theorem area_EFGH_is_three : areaEFGH EFGH = 3 := by
  sorry

#check area_EFGH_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFGH_is_three_l833_83315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_c_for_sqrt2_frac_equality_condition_l833_83321

/-- The fractional part of a real number x -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- The statement of the theorem -/
theorem max_c_for_sqrt2_frac : 
  (∃ c : ℝ, c = Real.sqrt 2 / 4 ∧ 
   (∀ n : ℕ, frac (n * Real.sqrt 2) ≥ c / n) ∧
   (∀ c' : ℝ, (∀ n : ℕ, frac (n * Real.sqrt 2) ≥ c' / n) → c' ≤ c)) := by
  sorry

/-- For which n the equality holds -/
theorem equality_condition (c : ℝ) (h : c = Real.sqrt 2 / 4) :
  ∀ n : ℕ, frac (n * Real.sqrt 2) = c / n ↔ 
    n * Real.sqrt 2 - ⌊n * Real.sqrt 2⌋ = Real.sqrt 2 / (4 * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_c_for_sqrt2_frac_equality_condition_l833_83321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_six_zero_l833_83378

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem sum_six_zero (seq : ArithmeticSequence) 
    (h : seq.a 2 ^ 2 + seq.a 3 ^ 2 = seq.a 4 ^ 2 + seq.a 5 ^ 2) : 
    sum_n seq 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_six_zero_l833_83378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_2006_is_rhombus_not_rectangle_l833_83346

-- Define a quadrilateral
structure Quadrilateral :=
  (is_rhombus : Bool)
  (is_rectangle : Bool)

-- Define the sequence of quadrilaterals
def quadrilateral_sequence : ℕ → Quadrilateral
  | 0 => { is_rhombus := false, is_rectangle := false }  -- S₀ (added to handle Nat.zero case)
  | 1 => { is_rhombus := false, is_rectangle := false }  -- S₁
  | n+2 => 
    if n % 2 = 0 then
      { is_rhombus := true, is_rectangle := true }  -- Rectangle for odd n > 1
    else
      { is_rhombus := true, is_rectangle := false }  -- Rhombus but not rectangle for even n

-- Theorem statement
theorem s_2006_is_rhombus_not_rectangle :
  let s_2006 := quadrilateral_sequence 2006
  s_2006.is_rhombus ∧ ¬s_2006.is_rectangle := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_2006_is_rhombus_not_rectangle_l833_83346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_coordinates_l833_83319

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := x + y - 6 = 0

-- Theorem statement
theorem tangent_point_coordinates :
  ∃ (x₀ : ℝ),
    (Real.exp x₀ - x₀ = 1) ∧ 
    f x₀ = 1 ∧
    x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_coordinates_l833_83319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_purchase_optimization_l833_83362

/-- Represents the annual purchase and cost scenario for a store --/
structure StorePurchase where
  annual_amount : ℝ
  shipping_cost : ℝ
  storage_cost_rate : ℝ

/-- Calculates the total annual cost for a given purchase amount --/
noncomputable def total_annual_cost (s : StorePurchase) (purchase_amount : ℝ) : ℝ :=
  (s.annual_amount / purchase_amount) * s.shipping_cost + s.storage_cost_rate * purchase_amount

/-- The optimal purchase amount minimizes the total annual cost --/
def is_optimal_purchase (s : StorePurchase) (purchase_amount : ℝ) : Prop :=
  ∀ x > 0, total_annual_cost s purchase_amount ≤ total_annual_cost s x

/-- The purchase amount is within the acceptable range if the total annual cost is at most 5.85 million --/
def is_in_acceptable_range (s : StorePurchase) (purchase_amount : ℝ) : Prop :=
  total_annual_cost s purchase_amount ≤ 5850000

theorem store_purchase_optimization (s : StorePurchase) 
    (h1 : s.annual_amount = 900)
    (h2 : s.shipping_cost = 90000)
    (h3 : s.storage_cost_rate = 90000) : 
    (is_optimal_purchase s 30) ∧ 
    (∀ x, is_in_acceptable_range s x ↔ 20 ≤ x ∧ x ≤ 45) := by
  sorry

#check store_purchase_optimization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_purchase_optimization_l833_83362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_integer_l833_83323

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = 4
  sum_234 : a 2 + a 3 + a 4 = 18
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem ratio_is_integer (seq : ArithmeticSequence) :
  ∀ n : ℕ, n > 0 → (∃ k : ℤ, sum_n seq 5 / sum_n seq n = k) ↔ n = 3 ∨ n = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_integer_l833_83323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l833_83348

/-- Represents a rectangle -/
structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  area : ℝ
  diagonal : ℝ

/-- Defines similarity between two rectangles -/
def Similar (R1 R2 : Rectangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ R2.side1 = k * R1.side1 ∧ R2.side2 = k * R1.side2

/-- Given two similar rectangles R1 and R2, where R1 has one side of 4 units and area of 32 square units,
    and R2 has a diagonal of 20 units, prove that the area of R2 is 160 square units. -/
theorem area_of_similar_rectangle (R1 R2 : Rectangle) 
  (h1 : R1.side1 = 4)
  (h2 : R1.area = 32)
  (h3 : R2.diagonal = 20)
  (h4 : Similar R1 R2) :
  R2.area = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l833_83348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_twelve_terms_l833_83333

def my_sequence (n : ℕ) : ℕ := n.factorial + n

def sum_of_sequence (n : ℕ) : ℕ := (Finset.range n).sum (λ i => my_sequence (i + 1))

theorem units_digit_of_sum_twelve_terms :
  (sum_of_sequence 12) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_twelve_terms_l833_83333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l833_83386

-- Define the fixed point P
def P : ℝ × ℝ := (-1, 2)

-- Define the line that passes through P for all a ∈ ℝ
def line (a x y : ℝ) : Prop := (x + y - 1) - a * (x + 1) = 0

-- Define the condition that the line passes through P for all a
def line_passes_through_P : Prop := ∀ a : ℝ, line a P.1 P.2

-- Define the circle with center P and radius √5
def circle_eq (x y : ℝ) : Prop := (x - P.1)^2 + (y - P.2)^2 = 5

-- State the theorem
theorem circle_equation (h : line_passes_through_P) :
  ∀ x y : ℝ, circle_eq x y ↔ x^2 + y^2 + 2*x - 4*y = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l833_83386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_price_is_one_l833_83391

/-- Represents the sales data for a fruit vendor --/
structure FruitSales where
  apple_price : ℚ
  morning_apples : ℕ
  morning_oranges : ℕ
  afternoon_apples : ℕ
  afternoon_oranges : ℕ
  total_sales : ℚ

/-- Calculates the price of one orange given the sales data --/
def orange_price (sales : FruitSales) : ℚ :=
  let total_apples := sales.morning_apples + sales.afternoon_apples
  let total_oranges := sales.morning_oranges + sales.afternoon_oranges
  let apple_revenue := sales.apple_price * total_apples
  (sales.total_sales - apple_revenue) / total_oranges

/-- Theorem stating that the price of one orange is $1 given the specific sales data --/
theorem orange_price_is_one :
  let sales : FruitSales := {
    apple_price := 3/2,
    morning_apples := 40,
    morning_oranges := 30,
    afternoon_apples := 50,
    afternoon_oranges := 40,
    total_sales := 205
  }
  orange_price sales = 1 := by
    -- The proof goes here
    sorry

#eval orange_price {
  apple_price := 3/2,
  morning_apples := 40,
  morning_oranges := 30,
  afternoon_apples := 50,
  afternoon_oranges := 40,
  total_sales := 205
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_price_is_one_l833_83391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_concurrent_l833_83389

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The vertices of the triangle -/
  vertices : Fin 3 → ℝ × ℝ
  /-- The center of the inscribed circle -/
  incenter : ℝ × ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ

/-- A line in 2D space -/
structure Line where
  /-- A point on the line -/
  point : ℝ × ℝ
  /-- The direction vector of the line -/
  direction : ℝ × ℝ

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p = (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2)

/-- The lines connecting vertices to opposite tangency points -/
def tangentLines (t : TriangleWithInscribedCircle) : Fin 3 → Line :=
  sorry

/-- The theorem stating that the tangent lines are concurrent -/
theorem tangent_lines_concurrent (t : TriangleWithInscribedCircle) : 
  ∃ (p : ℝ × ℝ), ∀ (i : Fin 3), pointOnLine (tangentLines t i) p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_concurrent_l833_83389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_implies_exponent_difference_l833_83397

/-- Two terms are considered "like terms" if they have the same variables with the same exponents -/
def like_terms (term1 term2 : ℤ → ℤ → ℚ) : Prop :=
  ∃ (c1 c2 : ℚ), ∀ (a b : ℤ), term1 a b = c1 * (a : ℚ)^2 * (b : ℚ)^1 ∧ term2 a b = c2 * (a : ℚ)^2 * (b : ℚ)^1

theorem like_terms_implies_exponent_difference
  (m n : ℤ) :
  like_terms (λ a b => -2 * (a : ℚ)^2 * (b : ℚ)^m) (λ a b => 4 * (a : ℚ)^n * (b : ℚ)) →
  m - n = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_implies_exponent_difference_l833_83397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_size_l833_83304

/-- A square on the chessboard -/
structure Square where
  x : ℕ
  y : ℕ

/-- A checker on the chessboard -/
structure Checker where
  position : Square

/-- A chessboard with checkers -/
structure Chessboard where
  size : ℕ
  checkers : Finset Checker

/-- Predicate to check if a checker is on the edge of the board -/
def is_on_edge (c : Checker) (board : Chessboard) : Prop :=
  c.position.x = 0 ∨ c.position.x = board.size - 1 ∨
  c.position.y = 0 ∨ c.position.y = board.size - 1

/-- Predicate to check if a square is a corner of the board -/
def is_corner (s : Square) (board : Chessboard) : Prop :=
  (s.x = 0 ∨ s.x = board.size - 1) ∧ (s.y = 0 ∨ s.y = board.size - 1)

theorem chessboard_size 
  (board : Chessboard)
  (h1 : board.checkers.card = 4)
  (h2 : ∀ c, c ∈ board.checkers → is_on_edge c board)
  (h3 : ∀ s, is_corner s board → ∀ c, c ∈ board.checkers → c.position ≠ s)
  (h4 : ∀ c1 c2, c1 ∈ board.checkers → c2 ∈ board.checkers → c1 ≠ c2 → c1.position ≠ c2.position) :
  board.size * board.size = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_size_l833_83304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_6_smallest_positive_period_range_on_interval_range_bounds_l833_83327

-- Define the function f
noncomputable def f (x : Real) : Real := 2 * Real.sqrt 3 * Real.cos x * Real.sin x + 2 * (Real.cos x) ^ 2

-- Theorem for part 1
theorem f_at_pi_over_6 : f (π / 6) = 3 := by sorry

-- Theorem for part 2
theorem smallest_positive_period : ∃ T : Real, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = π := by sorry

-- Theorem for part 3
theorem range_on_interval : ∀ y ∈ Set.Icc 0 3, ∃ x ∈ Set.Icc 0 (π / 2), f x = y := by sorry

theorem range_bounds : ∀ x ∈ Set.Icc 0 (π / 2), 0 ≤ f x ∧ f x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_6_smallest_positive_period_range_on_interval_range_bounds_l833_83327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_center_of_symmetry_l833_83330

theorem tan_center_of_symmetry :
  ∀ (k : ℤ), ∃ (x y : ℝ),
    x = (2 * k - 1 : ℝ) / 4 ∧
    y = 0 ∧
    (∀ (t : ℝ), Real.tan (π * (x + t) + π / 4) = Real.tan (π * (x - t) + π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_center_of_symmetry_l833_83330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_real_parts_cube_root_l833_83339

theorem product_of_real_parts_cube_root (z : ℂ) : 
  z^3 = 2 + 2*I → 
  (Real.sqrt 2 * Real.cos (π / 12 : ℝ)) * 
  (Real.sqrt 2 * Real.cos ((π / 12 : ℝ) + (2 * π) / 3)) * 
  (Real.sqrt 2 * Real.cos ((π / 12 : ℝ) + (4 * π) / 3)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_real_parts_cube_root_l833_83339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l833_83307

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.tan x - 1) + Real.sqrt (4 - x^2)

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-2) (-Real.pi/2) ∪ Set.Ico (Real.pi/4) (Real.pi/2)

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | Real.tan x - 1 ≥ 0 ∧ 4 - x^2 ≥ 0} = domain_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l833_83307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degrees_of_freedom_uniform_distribution_l833_83384

/-- Represents the number of sample intervals in a Pearson's chi-squared test -/
def s : ℕ := sorry

/-- Represents the number of degrees of freedom in a Pearson's chi-squared test -/
def k : ℕ := sorry

/-- Represents that we are testing a uniform distribution -/
def is_uniform_distribution : Prop := sorry

/-- Theorem stating the relationship between k and s for a Pearson's chi-squared test of uniform distribution -/
theorem degrees_of_freedom_uniform_distribution (h : is_uniform_distribution) : k = s - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degrees_of_freedom_uniform_distribution_l833_83384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_areas_in_circular_disk_l833_83367

-- Define the number of areas as a function of n
def num_areas (n : ℕ) : ℕ := 4 + 6 * n

-- Define the concept of maximum non-overlapping areas
def maximum_non_overlapping_areas (radii : ℕ) (secants : ℕ) : ℕ :=
  sorry -- This would be defined based on the problem's constraints

-- Additional helper definitions if needed
def equally_spaced_radii (n : ℕ) : Prop :=
  sorry -- This would define what it means for radii to be equally spaced

def non_intersecting_secants : Prop :=
  sorry -- This would define what it means for secants to not intersect inside the disk

-- Theorem statement
theorem max_areas_in_circular_disk (n : ℕ) :
  let radii := 3 * n
  let secants := 2
  num_areas n = maximum_non_overlapping_areas radii secants :=
by
  sorry -- Proof to be completed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_areas_in_circular_disk_l833_83367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_12_divisors_l833_83340

theorem smallest_integer_with_12_divisors :
  ∃ n : ℕ, (Nat.divisors n).card = 12 ∧ (∀ m : ℕ, (Nat.divisors m).card = 12 → n ≤ m) ∧ n = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_12_divisors_l833_83340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_line_intersection_l833_83360

/-- Given a dihedral angle with measure α and a line on one face forming an angle β with the edge,
    the angle θ between this line and the other face is equal to arcsin(sin α * sin β). -/
theorem dihedral_angle_line_intersection (α β : Real) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π/2) :
  Real.arcsin (Real.sin α * Real.sin β) = Real.arcsin (Real.sin α * Real.sin β) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_line_intersection_l833_83360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buckingham_palace_visitors_l833_83370

/-- The number of visitors to Buckingham Palace on a given day -/
def visitors_on_day (day : ℕ) : ℕ := 
  if day = 0 then 47118
  else if day = 1 then 191
  else if day = 2 then 457
  else 191

/-- The total number of visitors over a given number of days -/
def total_visitors (days : ℕ) : ℕ :=
  Finset.sum (Finset.range days) (fun i => visitors_on_day (i + 1))

theorem buckingham_palace_visitors :
  visitors_on_day 0 = total_visitors 245 + 57 := by
  sorry

#eval visitors_on_day 0
#eval total_visitors 245 + 57

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buckingham_palace_visitors_l833_83370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l833_83313

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * ω * x) + 2 * (Real.cos (ω * x))^2

theorem function_properties (ω : ℝ) :
  (∀ x : ℝ, f ω (x + π / (2 * ω)) = f ω x) →
  (ω = 1) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (π/6 + k*π) (2*π/3 + k*π), 
    ∀ y ∈ Set.Icc (π/6 + k*π) (2*π/3 + k*π), 
    x ≤ y → f ω y ≤ f ω x) ∧
  (Set.Ioo (1 - Real.sqrt 3) 2 = 
    {m : ℝ | ∃ x ∈ Set.Ioo (-π/4) (π/4), f ω x = m}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l833_83313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mulch_purchase_proof_l833_83338

/-- Calculates the number of tons of mulch purchased given the total cost,
    price per pound, and the number of pounds in a ton. -/
noncomputable def tons_of_mulch_purchased (total_cost : ℝ) (price_per_pound : ℝ) (pounds_per_ton : ℝ) : ℝ :=
  (total_cost / price_per_pound) / pounds_per_ton

/-- Proves that the number of tons of mulch purchased is equal to 3 -/
theorem mulch_purchase_proof :
  tons_of_mulch_purchased 15000 2.5 2000 = 3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval tons_of_mulch_purchased 15000 2.5 2000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mulch_purchase_proof_l833_83338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lathe_defect_probabilities_l833_83316

/-- Represents a lathe used for processing parts -/
structure Lathe where
  defect_rate : ℝ
  proportion : ℝ

/-- The probability of a part being defective -/
def probability_defective (lathes : List Lathe) : ℝ :=
  lathes.foldr (λ l acc => l.defect_rate * l.proportion + acc) 0

/-- The probability of a part not being defective, given it was processed by a specific lathe -/
def probability_not_defective_given_lathe (l : Lathe) : ℝ :=
  1 - l.defect_rate

theorem lathe_defect_probabilities 
  (l1 l2 l3 : Lathe)
  (h1 : l1.defect_rate = 0.06)
  (h2 : l2.defect_rate = 0.05)
  (h3 : l3.defect_rate = 0.02)
  (h4 : l1.proportion = 0.1)
  (h5 : l2.proportion = 0.4)
  (h6 : l3.proportion = 0.5)
  : probability_defective [l1, l2, l3] = 0.036 ∧ 
    probability_not_defective_given_lathe l3 = 0.98 := by
  sorry

#check lathe_defect_probabilities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lathe_defect_probabilities_l833_83316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_distance_sum_on_line_segment_l833_83342

noncomputable def rectangular_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₁ - x₂| + |y₁ - y₂|

noncomputable def line_equation (x : ℝ) : ℝ := -2 * x + Real.sqrt 5

theorem rectangular_distance_sum_on_line_segment :
  let x_min : ℝ := -Real.sqrt 5
  let x_max : ℝ := 2 * Real.sqrt 5
  let min_distance := min (rectangular_distance 0 0 x_min (line_equation x_min))
                          (rectangular_distance 0 0 x_max (line_equation x_max))
  let max_distance := max (rectangular_distance 0 0 x_min (line_equation x_min))
                          (rectangular_distance 0 0 x_max (line_equation x_max))
  min_distance + max_distance = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_distance_sum_on_line_segment_l833_83342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_more_valuable_than_banana_l833_83354

-- Define the variables as real numbers
variable (L B A V : ℝ)

-- Condition 1: 1 lemon and 1 banana can be traded for 2 oranges and 23 cherries
axiom exchange_rate_1 : L + B = 2 * A + 23 * V

-- Condition 2: 3 lemons can be traded for 2 bananas, 2 oranges, and 14 cherries
axiom exchange_rate_2 : 3 * L = 2 * B + 2 * A + 14 * V

-- Theorem: The value of a lemon is greater than the value of a banana
theorem lemon_more_valuable_than_banana : L > B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_more_valuable_than_banana_l833_83354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_exterior_angle_l833_83305

/-- Polygon represents a geometric polygon -/
structure Polygon where
  -- You might want to add more fields here to fully define a polygon

/-- Regular represents that a polygon is regular (all sides equal and all angles equal) -/
def Regular (p : Polygon) : Prop := sorry

/-- NumberOfSides represents the number of sides of a polygon -/
def NumberOfSides (p : Polygon) : ℕ := sorry

/-- ExteriorAngle represents the measure of an exterior angle of a polygon -/
def ExteriorAngle (p : Polygon) : ℝ := sorry

theorem regular_octagon_exterior_angle :
  ∀ (octagon : Polygon), 
    Regular octagon → 
    NumberOfSides octagon = 8 → 
    ExteriorAngle octagon = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_exterior_angle_l833_83305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_relationships_l833_83365

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Theorem statement
theorem geometry_relationships 
  (a b : Line) (α : Plane) :
  (∀ (P : Plane), intersects a α → contained_in b α → ¬ parallel a b) ∧
  (parallel a b → perpendicular b α → perpendicular a α) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_relationships_l833_83365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_l833_83347

/-- The speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- The speed of the man in still water is the average of upstream and downstream speeds -/
def isValidSpeed (s : RowingSpeed) : Prop :=
  s.stillWater = (s.upstream + s.downstream) / 2

/-- The theorem stating the downstream speed given the conditions -/
theorem downstream_speed (s : RowingSpeed) 
  (h1 : s.upstream = 10)
  (h2 : s.stillWater = 15)
  (h3 : isValidSpeed s) :
  s.downstream = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_l833_83347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_2_minus_3i_l833_83364

theorem real_part_of_2_minus_3i :
  (Complex.re (2 - 3 * Complex.I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_2_minus_3i_l833_83364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_configuration_l833_83351

/-- Represents a cuboid in 3D space -/
structure Cuboid where
  x : Set ℝ
  y : Set ℝ
  z : Set ℝ

/-- Represents the configuration of 12 cuboids -/
def CuboidConfiguration := Fin 12 → Cuboid

/-- Determines if two cuboids intersect -/
def intersects (c1 c2 : Cuboid) : Prop :=
  (c1.x ∩ c2.x).Nonempty ∧ (c1.y ∩ c2.y).Nonempty ∧ (c1.z ∩ c2.z).Nonempty

/-- Checks if the configuration satisfies the given intersection conditions -/
def satisfiesConditions (config : CuboidConfiguration) : Prop :=
  ∀ i j : Fin 12, i ≠ j →
    (i = 1 ∧ (j = 2 ∨ j = 12)) ∨
    (i = 2 ∧ (j = 1 ∨ j = 3)) ∨
    (i = 3 ∧ (j = 2 ∨ j = 4)) ∨
    (i = 12 ∧ (j = 11 ∨ j = 1)) ∨
    intersects (config i) (config j)

/-- The main theorem stating that no valid configuration exists -/
theorem no_valid_configuration : ¬∃ (config : CuboidConfiguration), satisfiesConditions config := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_configuration_l833_83351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_transformed_f_l833_83380

-- Define the custom operation
noncomputable def customOp (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := customOp (Real.cos x ^ 2 + Real.sin x) (5/4)

-- State the theorem
theorem max_value_of_transformed_f :
  ∃ (M : ℝ), M = 2 ∧
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    f (x - Real.pi / 2) + 3/4 ≤ M :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_transformed_f_l833_83380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_plus_count_equals_281_l833_83359

/-- The sum of integers from 20 to 30, inclusive -/
def sum_20_to_30 : ℕ := Finset.sum (Finset.range 11) (fun i => i + 20)

/-- The count of even integers from 20 to 30, inclusive -/
def count_even_20_to_30 : ℕ := (Finset.range 11).filter (fun i => Even (i + 20)) |>.card

/-- Theorem stating that the sum of integers from 20 to 30 (inclusive) 
    plus the count of even integers in the same range equals 281 -/
theorem sum_plus_count_equals_281 : sum_20_to_30 + count_even_20_to_30 = 281 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_plus_count_equals_281_l833_83359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_relation_l833_83328

/-- A parabola with equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A hyperbola with equation x^2 - y^2 = 1 -/
def Hyperbola := {point : ℝ × ℝ | point.fst^2 - point.snd^2 = 1}

/-- The focus of a hyperbola -/
noncomputable def hyperbola_focus : ℝ × ℝ := (-Real.sqrt 2, 0)

/-- The axis of a parabola -/
noncomputable def parabola_axis (par : Parabola) : Set ℝ :=
  {x | x = -Real.sqrt 2}

/-- The theorem stating the relationship between the parabola and hyperbola -/
theorem parabola_hyperbola_relation (par : Parabola) :
  hyperbola_focus.fst ∈ parabola_axis par →
  par.p = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_relation_l833_83328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l833_83332

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (2*θ) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l833_83332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_even_and_pi_period_l833_83344

-- Define the absolute sine function
noncomputable def abs_sin (x : ℝ) : ℝ := |Real.sin x|

-- State the theorem
theorem abs_sin_even_and_pi_period : 
  (∀ x, abs_sin x = abs_sin (-x)) ∧ 
  (∀ x, abs_sin (x + Real.pi) = abs_sin x) ∧
  (∀ p, p > 0 ∧ p < Real.pi → ∃ x, abs_sin (x + p) ≠ abs_sin x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_even_and_pi_period_l833_83344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_and_hypotenuse_l833_83334

/-- A right triangle with an altitude --/
structure RightTriangleWithAltitude where
  /-- Point X --/
  X : ℝ × ℝ
  /-- Point Y --/
  Y : ℝ × ℝ
  /-- Point Z --/
  Z : ℝ × ℝ
  /-- Point W on XZ --/
  W : ℝ × ℝ
  /-- XYZ is a right triangle with right angle at Y --/
  right_angle_at_Y : (X - Y) • (Z - Y) = 0
  /-- W is the foot of the altitude from Y to XZ --/
  W_is_foot_of_altitude : ∃ t : ℝ, W = X + t • (Z - X) ∧ (Y - W) • (Z - X) = 0
  /-- Length of XW is 5 --/
  XW_length : Real.sqrt ((X.1 - W.1)^2 + (X.2 - W.2)^2) = 5
  /-- Length of WZ is 7 --/
  WZ_length : Real.sqrt ((W.1 - Z.1)^2 + (W.2 - Z.2)^2) = 7

/-- Main theorem about the area and hypotenuse of the right triangle --/
theorem right_triangle_area_and_hypotenuse 
  (t : RightTriangleWithAltitude) : 
  Real.sqrt (|(t.X.1 - t.Y.1) * (t.Z.2 - t.Y.2) - (t.X.2 - t.Y.2) * (t.Z.1 - t.Y.1)|) / 2 = 6 * Real.sqrt 35 ∧ 
  Real.sqrt ((t.X.1 - t.Z.1)^2 + (t.X.2 - t.Z.2)^2) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_and_hypotenuse_l833_83334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_i_107_l833_83385

theorem conjugate_i_107 : (Complex.I : ℂ)^107 = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_i_107_l833_83385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_distances_l833_83302

def circleEquation (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2

def pointOnCircle (r u v : ℝ) : Prop :=
  circleEquation r u v

theorem circle_points_distances
  (r u v : ℝ)
  (p q : ℕ)
  (m n : ℕ)
  (h_odd : ∃ k : ℕ, r^2 = 2*k + 1)
  (h_prime_p : Nat.Prime p)
  (h_prime_q : Nat.Prime q)
  (h_u : u = p^m)
  (h_v : v = q^n)
  (h_uv : u > v)
  (h_on_circle_p : pointOnCircle r u v)
  (h_on_circle_q : pointOnCircle r (-u) v)
  : abs (r - u) = 1 ∧
    abs (-r - u) = 9 ∧
    abs (v - (-r)) = 8 ∧
    abs (2 * u) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_distances_l833_83302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l833_83395

/-- Given vectors a and b, if a + 2b is parallel to 3a + λb, then λ = 6 -/
theorem parallel_vectors_lambda (a b : ℝ × ℝ) (l : ℝ) : 
  a = (1, 3) → 
  b = (2, 1) → 
  (∃ (k : ℝ), k ≠ 0 ∧ a + 2 • b = k • (3 • a + l • b)) → 
  l = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l833_83395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l833_83314

-- Define the value of cos 30°
noncomputable def cos30 : ℝ := Real.sqrt 3 / 2

-- Define the theorem
theorem calculate_expression : 
  4 * cos30 + (1 - Real.sqrt 2) ^ 0 - Real.sqrt 12 + |(-2)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l833_83314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_edge_bound_l833_83369

/-- A simple graph without self-loops -/
structure MySimpleGraph (V : Type*) where
  adj : V → V → Prop
  sym : ∀ {u v}, adj u v → adj v u
  loopless : ∀ v, ¬adj v v

/-- The number of edges in a simple graph -/
def edge_count {V : Type*} (G : MySimpleGraph V) : ℕ := sorry

/-- The number of vertices in a simple graph -/
noncomputable def vertex_count {V : Type*} (G : MySimpleGraph V) : ℕ := sorry

/-- A cycle of length n in a graph -/
def has_cycle {V : Type*} (G : MySimpleGraph V) (n : ℕ) : Prop := sorry

theorem graph_edge_bound {V : Type*} (G : MySimpleGraph V) (n m : ℕ) 
  (h_vertex_count : vertex_count G = n)
  (h_edge_count : edge_count G = m)
  (h_no_3_cycle : ¬has_cycle G 3)
  (h_no_4_cycle : ¬has_cycle G 4) :
  m ≤ (n * Real.sqrt (n - 1)) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_edge_bound_l833_83369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_spent_percentage_l833_83358

-- Define the initial and remaining amounts
noncomputable def initial_amount : ℚ := 500
noncomputable def remaining_amount : ℚ := 350

-- Define the percentage spent
noncomputable def percentage_spent : ℚ := ((initial_amount - remaining_amount) / initial_amount) * 100

-- Theorem to prove
theorem money_spent_percentage :
  percentage_spent = 30 :=
by
  -- Unfold the definition of percentage_spent
  unfold percentage_spent
  -- Simplify the expression
  simp [initial_amount, remaining_amount]
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_spent_percentage_l833_83358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_equals_seven_l833_83366

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 3 = 1

-- Define the foci F₁ and F₂
noncomputable def F₁ : ℝ × ℝ := sorry
noncomputable def F₂ : ℝ × ℝ := sorry

-- Define a point P on the ellipse
noncomputable def P : ℝ × ℝ := sorry
axiom P_on_ellipse : ellipse P.1 P.2

-- Define the midpoint of PF₁
noncomputable def midpoint_PF₁ : ℝ × ℝ := ((P.1 + F₁.1) / 2, (P.2 + F₁.2) / 2)

-- Axiom: The midpoint of PF₁ is on the y-axis
axiom midpoint_on_yaxis : midpoint_PF₁.1 = 0

-- Define the distances PF₁ and PF₂
noncomputable def PF₁ : ℝ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
noncomputable def PF₂ : ℝ := Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)

-- Define t such that PF₁ = t * PF₂
noncomputable def t : ℝ := PF₁ / PF₂

-- Theorem: Under these conditions, t = 7
theorem t_equals_seven : t = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_equals_seven_l833_83366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l833_83399

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  asymptote1 : ℝ → ℝ
  asymptote2 : ℝ → ℝ
  point : ℝ × ℝ

/-- The distance between the foci of a hyperbola -/
noncomputable def foci_distance (h : Hyperbola) : ℝ :=
  2 * Real.sqrt 24.694

/-- Theorem stating that for a hyperbola with the given properties, 
    the distance between its foci is 2√24.694 -/
theorem hyperbola_foci_distance (h : Hyperbola) 
  (h_asymptote1 : h.asymptote1 = fun x => 2 * x + 3)
  (h_asymptote2 : h.asymptote2 = fun x => 1 - 2 * x)
  (h_point : h.point = (4, 5)) :
  foci_distance h = 2 * Real.sqrt 24.694 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l833_83399
