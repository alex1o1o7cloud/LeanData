import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_problem_l509_50910

/-- The number of which 70 is 56.00000000000001 percent -/
noncomputable def x : ℝ := 70 / 0.5600000000000001

theorem percent_problem : ‖x - 125‖ < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_problem_l509_50910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_mark_l509_50941

theorem new_average_mark (n : ℕ) (total_avg : ℚ) (excluded_n : ℕ) (excluded_avg : ℚ) :
  n = 9 →
  total_avg = 60 →
  excluded_n = 5 →
  excluded_avg = 44 →
  (n * total_avg - excluded_n * excluded_avg) / (n - excluded_n) = 80 := by
  intro hn htotal hexcl_n hexcl_avg
  
  -- Define intermediate calculations
  let remaining_n := n - excluded_n
  let total_marks := n * total_avg
  let excluded_marks := excluded_n * excluded_avg
  let remaining_marks := total_marks - excluded_marks
  let new_avg := remaining_marks / remaining_n

  -- Prove the theorem
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_mark_l509_50941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_equivalence_l509_50906

/-- The function f(t) defined as x*sin(t) + y*cos(t) -/
noncomputable def f (x y t : ℝ) : ℝ := x * Real.sin t + y * Real.cos t

/-- The theorem stating the equivalence of the integral inequality and the domain conditions -/
theorem integral_inequality_equivalence (x y : ℝ) :
  (|∫ t in (-π)..π, f x y t * Real.cos t| ≤ ∫ t in (-π)..π, (f x y t)^2) ↔
  (x^2 + (y - 1/2)^2 ≥ 1/4 ∧ x^2 + (y + 1/2)^2 ≥ 1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_equivalence_l509_50906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_theorem_l509_50916

/-- Represents an alloy with gold, silver, and copper components -/
structure Alloy where
  gold : ℚ
  silver : ℚ
  copper : ℚ

/-- The total weight of an alloy -/
def Alloy.weight (a : Alloy) : ℚ := a.gold + a.silver + a.copper

/-- Creates an alloy given a total weight and ratio of components -/
def createAlloy (weight : ℚ) (gold_ratio silver_ratio copper_ratio : ℚ) : Alloy :=
  let total_ratio := gold_ratio + silver_ratio + copper_ratio
  { gold := (gold_ratio / total_ratio) * weight,
    silver := (silver_ratio / total_ratio) * weight,
    copper := (copper_ratio / total_ratio) * weight }

/-- Mixes multiple alloys -/
def mixAlloys (alloys : List Alloy) : Alloy :=
  { gold := alloys.map (·.gold) |>.sum,
    silver := alloys.map (·.silver) |>.sum,
    copper := alloys.map (·.copper) |>.sum }

theorem alloy_mixture_theorem :
  let alloy1 := createAlloy 195 1 3 5
  let alloy2 := createAlloy 78 3 5 1
  let alloy3 := createAlloy 78 5 1 3
  let mixture := mixAlloys [alloy1, alloy2, alloy3]
  mixture.weight = 351 ∧
  mixture.gold / mixture.silver = 7 / 9 ∧
  mixture.silver / mixture.copper = 9 / 11 := by
  sorry

#check alloy_mixture_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_theorem_l509_50916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_30_l509_50954

def S : Finset Int := {-30, -5, -1, 3, 4, 10}

theorem largest_quotient_is_30 : 
  ∀ a b : Int, a ∈ S → b ∈ S → b ≠ 0 → (a : ℚ) / b ≤ 30 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_30_l509_50954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cagney_lacey_frosting_l509_50926

/-- Represents the cupcake frosting scenario with Cagney and Lacey -/
def CupcakeFrosting (cagney_rate : ℚ) (lacey_rate : ℚ) (error_frequency : ℕ) (error_time : ℚ) (total_time : ℚ) : Prop :=
  let combined_rate : ℚ := 1 / (1 / cagney_rate + 1 / lacey_rate)
  let initial_cupcakes : ℕ := Int.toNat ((total_time / combined_rate).floor)
  let error_count : ℕ := (initial_cupcakes / error_frequency)
  let adjusted_time : ℚ := total_time - (error_count : ℚ) * error_time
  let final_cupcakes : ℕ := Int.toNat ((adjusted_time / combined_rate).floor)
  final_cupcakes = 37

/-- Theorem stating that Cagney and Lacey can frost 37 cupcakes in 10 minutes -/
theorem cagney_lacey_frosting :
  CupcakeFrosting (1/15) (1/40) 10 40 600 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cagney_lacey_frosting_l509_50926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l509_50959

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x + 4 * y^2 + 8 * y + 8 = 0

/-- The distance between foci of the ellipse -/
noncomputable def foci_distance : ℝ := 5/3

/-- Theorem: The distance between the foci of the given ellipse is 5/3 -/
theorem ellipse_foci_distance :
  ∀ x y : ℝ, ellipse_equation x y → foci_distance = 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l509_50959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_squares_difference_l509_50984

theorem consecutive_squares_difference (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (max a (max b c)) > 2022 →
  (a^2 - b^2 = -1 ∧ b^2 - c^2 = 0 ∧ c^2 - a^2 = 1) ∨
  (a^2 - b^2 = -1 ∧ b^2 - c^2 = 1 ∧ c^2 - a^2 = 0) ∨
  (a^2 - b^2 = 0 ∧ b^2 - c^2 = -1 ∧ c^2 - a^2 = 1) ∨
  (a^2 - b^2 = 0 ∧ b^2 - c^2 = 1 ∧ c^2 - a^2 = -1) ∨
  (a^2 - b^2 = 1 ∧ b^2 - c^2 = -1 ∧ c^2 - a^2 = 0) ∨
  (a^2 - b^2 = 1 ∧ b^2 - c^2 = 0 ∧ c^2 - a^2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_squares_difference_l509_50984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_payment_action_mapping_l509_50912

-- Define the payment technologies
inductive PaymentTechnology
| Chip
| MagneticStripe
| Paypass
| CVC

-- Define the actions
inductive PaymentAction
| Tap
| PayOnline
| Swipe
| InsertIntoTerminal

-- Define the mapping function
def payment_action_mapping : PaymentTechnology → PaymentAction
| PaymentTechnology.Chip => PaymentAction.InsertIntoTerminal
| PaymentTechnology.MagneticStripe => PaymentAction.Swipe
| PaymentTechnology.Paypass => PaymentAction.Tap
| PaymentTechnology.CVC => PaymentAction.PayOnline

-- Theorem statement
theorem correct_payment_action_mapping :
  (payment_action_mapping PaymentTechnology.Chip = PaymentAction.InsertIntoTerminal) ∧
  (payment_action_mapping PaymentTechnology.MagneticStripe = PaymentAction.Swipe) ∧
  (payment_action_mapping PaymentTechnology.Paypass = PaymentAction.Tap) ∧
  (payment_action_mapping PaymentTechnology.CVC = PaymentAction.PayOnline) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_payment_action_mapping_l509_50912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_season_rounds_l509_50918

/-- Theorem about the number of rounds in a bowling season -/
theorem bowling_season_rounds : ∃ (total_rounds : ℕ), total_rounds = 10 := by
  -- League record in points per player per round
  let league_record : ℕ := 287

  -- Number of players per team
  let players_per_team : ℕ := 4

  -- Number of rounds played so far
  let rounds_played : ℕ := 9

  -- Total score after 9 rounds
  let current_score : ℕ := 10440

  -- Difference between league record and minimum average needed in final round
  let final_round_diff : ℕ := 27

  -- Calculate the total number of rounds in the season
  let total_rounds : ℕ :=
    let points_per_round := league_record * players_per_team
    let final_round_score := (league_record - final_round_diff) * players_per_team
    let total_points := current_score + final_round_score
    total_points / points_per_round

  -- Prove that the total number of rounds is 10
  have h : total_rounds = 10 := by
    -- The actual proof would go here
    sorry

  -- Conclude the theorem
  exact ⟨total_rounds, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_season_rounds_l509_50918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_north_southland_population_increase_l509_50980

/-- Represents the time interval in hours between births -/
noncomputable def birth_interval : ℝ := 6

/-- Represents the time interval in days between deaths -/
noncomputable def death_interval : ℝ := 2

/-- Represents the average number of days in a year, accounting for leap years -/
noncomputable def avg_days_per_year : ℝ := 365.5

/-- Calculates the average annual population increase in North Southland -/
noncomputable def annual_population_increase : ℝ :=
  ((24 / birth_interval) - (1 / death_interval)) * avg_days_per_year

/-- Rounds a real number to the nearest multiple of 50 -/
noncomputable def round_to_nearest_fifty (x : ℝ) : ℤ :=
  ⌊(x / 50 + 0.5)⌋ * 50

/-- Theorem stating that the average annual population increase in North Southland,
    rounded to the nearest fifty, is 1300 -/
theorem north_southland_population_increase :
  round_to_nearest_fifty annual_population_increase = 1300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_north_southland_population_increase_l509_50980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_cost_l509_50960

/-- Proves that the cost of each adult ticket is $11 --/
theorem adult_ticket_cost (num_adults num_children : ℕ) (child_ticket_cost adult_ticket_cost : ℚ) (extra_cost : ℚ) 
  (h1 : num_adults = 9)
  (h2 : num_children = 7)
  (h3 : child_ticket_cost = 7)
  (h4 : num_adults * adult_ticket_cost = num_children * child_ticket_cost + extra_cost)
  (h5 : extra_cost = 50) :
  adult_ticket_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_cost_l509_50960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kamal_weighted_average_l509_50981

-- Define the marks for each subject
noncomputable def english_marks : ℝ := 96
noncomputable def math_marks : ℝ := 65
noncomputable def physics_marks : ℝ := 82
noncomputable def chemistry_marks : ℝ := 67
noncomputable def biology_marks : ℝ := 85

-- Define the weights for each subject
noncomputable def english_weight : ℝ := 0.25
noncomputable def math_weight : ℝ := 0.20
noncomputable def physics_weight : ℝ := 0.30
noncomputable def chemistry_weight : ℝ := 0.15
noncomputable def biology_weight : ℝ := 0.10

-- Define the function to calculate the weighted average
noncomputable def weighted_average : ℝ :=
  (english_marks * english_weight +
   math_marks * math_weight +
   physics_marks * physics_weight +
   chemistry_marks * chemistry_weight +
   biology_marks * biology_weight) /
  (english_weight + math_weight + physics_weight + chemistry_weight + biology_weight)

-- Theorem statement
theorem kamal_weighted_average :
  weighted_average = 80.15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kamal_weighted_average_l509_50981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l509_50928

def f (m n x : ℝ) : ℝ := |x^2 - 2*m*x + n|

theorem f_decreasing_on_interval (m n : ℝ) (h : m^2 - n ≤ 0) :
  StrictMonoOn f (Set.Iic m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l509_50928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l509_50905

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a * (Real.cos C + Real.sqrt 3 * Real.sin C) = b + c →
  a = Real.sqrt 7 →
  (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 →
  (A = π / 3 ∧ ((b = 2 ∧ c = 3) ∨ (b = 3 ∧ c = 2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l509_50905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l509_50917

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given two non-collinear vectors and vector relationships between points,
    prove that λ = 2 when points A, B, and D are collinear. -/
theorem vector_collinearity (e₁ e₂ : V) (lambda : ℝ) 
  (h_non_collinear : ¬ ∃ (r : ℝ), e₂ = r • e₁)
  (h_AB : e₁ + e₂ = (1 : ℝ) • e₁ + (1 : ℝ) • e₂)
  (h_CB : -lambda • e₁ - 8 • e₂ = (-lambda : ℝ) • e₁ + (-8 : ℝ) • e₂)
  (h_CD : 3 • e₁ - 3 • e₂ = (3 : ℝ) • e₁ + (-3 : ℝ) • e₂)
  (h_collinear : ∃ (m : ℝ), e₁ + e₂ = m • ((3 + lambda) • e₁ + 5 • e₂)) :
  lambda = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l509_50917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_rotation_l509_50948

-- Define the number of sectors
variable (n : ℕ)

-- Define the maximum angular measure of each sector
noncomputable def max_sector_angle (n : ℕ) : ℝ := Real.pi / (n^2 - n + 1 : ℝ)

-- Define a sector as a pair of real numbers representing its start and end angles
def Sector := ℝ × ℝ

-- Define a list of n sectors
def sectors (n : ℕ) : List Sector := sorry

-- Define a predicate to check if a sector's measure is less than the maximum allowed
def valid_sector (s : Sector) (n : ℕ) : Prop :=
  (s.2 - s.1) < max_sector_angle n

-- Define a predicate to check if all sectors are valid
def all_sectors_valid (sectors : List Sector) (n : ℕ) : Prop :=
  ∀ s ∈ sectors, valid_sector s n

-- Define a function to rotate a sector by an angle θ
def rotate_sector (s : Sector) (θ : ℝ) : Sector := sorry

-- Define a predicate to check if a rotated sector overlaps with its original position
def sector_overlaps (s : Sector) (θ : ℝ) : Prop := sorry

-- The main theorem
theorem exists_valid_rotation (n : ℕ) (sectors : List Sector) 
  (h : all_sectors_valid sectors n) :
  ∃ θ : ℝ, ∀ s ∈ sectors, ¬ sector_overlaps s θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_rotation_l509_50948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_z_in_fourth_quadrant_l509_50923

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (2 + Complex.I) * m^2 - m * (1 - Complex.I) - (1 + 2*Complex.I)

-- Define what it means for a complex number to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop := 0 < z.re ∧ z.im < 0

-- State the theorem
theorem m_range_for_z_in_fourth_quadrant :
  ∀ m : ℝ, in_fourth_quadrant (z m) → -2 < m ∧ m < -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_z_in_fourth_quadrant_l509_50923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_used_fraction_is_correct_l509_50967

/-- Represents the fuel efficiency of a car in different terrains -/
structure FuelEfficiency where
  flat : ℚ
  uphill : ℚ
  downhill : ℚ

/-- Represents the duration of travel in different terrains -/
structure TravelDuration where
  flat : ℚ
  uphill : ℚ
  downhill : ℚ

/-- Calculates the fraction of fuel used given the initial conditions -/
noncomputable def fuelUsedFraction (speed : ℚ) (tankCapacity : ℚ) (efficiency : FuelEfficiency) (duration : TravelDuration) : ℚ :=
  let flatDistance := speed * duration.flat
  let uphillDistance := speed * duration.uphill
  let downhillDistance := speed * duration.downhill
  let flatFuelUsed := flatDistance / efficiency.flat
  let uphillFuelUsed := uphillDistance / efficiency.uphill
  let downhillFuelUsed := downhillDistance / efficiency.downhill
  (flatFuelUsed + uphillFuelUsed + downhillFuelUsed) / tankCapacity

/-- Theorem stating the fraction of fuel used under given conditions -/
theorem fuel_used_fraction_is_correct (speed : ℚ) (tankCapacity : ℚ) (efficiency : FuelEfficiency) (duration : TravelDuration)
  (h1 : speed = 50)
  (h2 : tankCapacity = 20)
  (h3 : efficiency.flat = 30)
  (h4 : efficiency.uphill = 24)
  (h5 : efficiency.downhill = 36)
  (h6 : duration.flat = 1)
  (h7 : duration.uphill = 2)
  (h8 : duration.downhill = 2) :
  fuelUsedFraction speed tankCapacity efficiency duration = 431/1000 := by
  sorry

#eval (431 : ℚ) / 1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_used_fraction_is_correct_l509_50967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_monotonic_iff_l509_50942

/-- Sequence a_n defined by the recurrence relation -/
noncomputable def a (a₁ : ℝ) : ℕ → ℝ
| 0 => a₁
| n + 1 => -1/2 * a a₁ n + (1/3)^n

/-- The sequence a_n is monotonic -/
def IsMonotonic (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) ≤ a n) ∨ (∀ n, a (n + 1) ≥ a n)

/-- Theorem: The sequence a_n is monotonic if and only if a₁ = 2/5 -/
theorem a_monotonic_iff (a₁ : ℝ) : IsMonotonic (a a₁) ↔ a₁ = 2/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_monotonic_iff_l509_50942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ellipse_eccentricity_l509_50947

/-- Represents a cylinder with given height and radius -/
structure Cylinder where
  height : ℝ
  radius : ℝ

/-- Represents a sphere inside the cylinder -/
structure Sphere where
  center : ℝ × ℝ × ℝ

/-- Represents an ellipse formed by the intersection of a plane with the cylinder -/
structure Ellipse where
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- The configuration of the problem -/
def problem_setup (c : Cylinder) (s1 s2 : Sphere) : Prop :=
  c.height = 10 ∧ c.radius = 1 ∧
  -- Spheres are tangent to top/bottom and side of cylinder
  (s1.center = (0, 0, 1)) ∧
  (s2.center = (0, 0, 9))

/-- The ellipse formed by the intersection of the tangent plane -/
def intersection_ellipse (c : Cylinder) (s1 s2 : Sphere) : Ellipse :=
  { semi_major_axis := 4,
    semi_minor_axis := 1 }

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (e.semi_major_axis^2 - e.semi_minor_axis^2) / e.semi_major_axis

/-- The main theorem: the eccentricity of the intersection ellipse is √15/4 -/
theorem intersection_ellipse_eccentricity
  (c : Cylinder) (s1 s2 : Sphere) (h : problem_setup c s1 s2) :
  eccentricity (intersection_ellipse c s1 s2) = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ellipse_eccentricity_l509_50947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_below_zero_l509_50934

-- Define a type for temperature
structure Temperature where
  value : Int

-- Define a function to represent temperature notation
def notate (t : Temperature) : String :=
  if t.value > 0 then
    s!"+{t.value}°C"
  else if t.value < 0 then
    s!"-{-t.value}°C"
  else
    "0°C"

-- Axiom: Temperatures above zero are denoted with a positive sign
axiom above_zero (t : Temperature) : t.value > 0 → notate t = s!"+{t.value}°C"

-- Theorem: Temperatures below zero are denoted with a negative sign
theorem below_zero (t : Temperature) : t.value < 0 → notate t = s!"-{-t.value}°C" := by
  sorry

-- Example usage
def temp_above : Temperature := ⟨4⟩
def temp_below : Temperature := ⟨-6⟩

#eval notate temp_above
#eval notate temp_below

end NUMINAMATH_CALUDE_ERRORFEEDBACK_below_zero_l509_50934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_vector_mult_l509_50925

theorem matrix_power_vector_mult (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.mulVec ![4, -3] = ![8, -6]) :
  (B ^ 4).mulVec ![4, -3] = ![64, -48] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_vector_mult_l509_50925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_download_time_90mb_l509_50900

/-- Represents the download time for a file given its size and download rates. -/
noncomputable def downloadTime (fileSize : ℝ) (rate1 : ℝ) (rate2 : ℝ) (threshold : ℝ) : ℝ :=
  let time1 := min threshold fileSize / rate1
  let remainingSize := max 0 (fileSize - threshold)
  let time2 := remainingSize / rate2
  time1 + time2

/-- Proves that the download time for a 90 MB file with given rates is 15 seconds. -/
theorem download_time_90mb : downloadTime 90 5 10 60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_download_time_90mb_l509_50900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_squared_divides_b_n_plus_one_l509_50978

-- Define a function to check if a number is a power of 2
def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

-- Define the main theorem
theorem infinitely_many_n_squared_divides_b_n_plus_one (b : ℕ) (h : b > 2) :
  (∃ f : ℕ → ℕ, StrictMono f ∧ ∀ n, (f n)^2 ∣ b^(f n) + 1) ↔ ¬(isPowerOfTwo (b + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_squared_divides_b_n_plus_one_l509_50978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l509_50993

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between x - y = 0 and x - y + 4 = 0 is 2√2 -/
theorem distance_between_given_lines :
  distance_between_parallel_lines 1 (-1) 0 4 = 2 * Real.sqrt 2 := by
  -- Unfold the definition of distance_between_parallel_lines
  unfold distance_between_parallel_lines
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l509_50993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mack_trip_time_l509_50986

/-- Calculates the time taken for a one-way trip given the total round trip time and speeds -/
noncomputable def one_way_trip_time (total_time : ℝ) (speed_to : ℝ) (speed_from : ℝ) : ℝ :=
  let distance := (total_time * speed_to * speed_from) / (speed_to + speed_from)
  distance / speed_to

/-- Given speeds and total time, the one-way trip time is approximately 1.55 hours -/
theorem mack_trip_time : 
  ∃ ε > 0, |one_way_trip_time 3 58 62 - 1.55| < ε :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mack_trip_time_l509_50986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_percentage_of_front_squat_l509_50955

noncomputable def front_squat_percentage : ℝ := 0.8
noncomputable def original_back_squat : ℝ := 200
noncomputable def back_squat_increase : ℝ := 50
noncomputable def total_weight_three_triples : ℝ := 540

noncomputable def new_back_squat : ℝ := original_back_squat + back_squat_increase
noncomputable def front_squat : ℝ := front_squat_percentage * new_back_squat
noncomputable def weight_one_triple : ℝ := total_weight_three_triples / 3

theorem triple_percentage_of_front_squat :
  (weight_one_triple / front_squat) * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_percentage_of_front_squat_l509_50955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_equation_l509_50919

theorem unique_solution_for_equation :
  ∀ (n : ℕ) (p : ℕ), 
    n > 0 →
    Prime p → 
    (17^n * 2^(n^2) - p = (2^(n^2 + 3) + 2^(n^2) - 1) * n^2) → 
    (n = 1 ∧ p = 17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_equation_l509_50919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_sum_l509_50977

-- Define the complex number 2 + i√2
noncomputable def z : ℂ := 2 + Complex.I * Real.sqrt 2

-- Define the polynomial x^3 + ax + b
def f (a b : ℝ) (x : ℂ) : ℂ := x^3 + a * x + b

theorem root_implies_sum (a b : ℝ) : f a b z = 0 → a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_sum_l509_50977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_space_diagonals_l509_50946

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem: A convex polyhedron Q with 30 vertices, 72 edges, 44 faces
    (34 triangular and 10 quadrilateral) has 343 space diagonals -/
theorem polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 34,
    quadrilateral_faces := 10
  }
  space_diagonals Q = 343 := by
  sorry

#eval space_diagonals {
  vertices := 30,
  edges := 72,
  faces := 44,
  triangular_faces := 34,
  quadrilateral_faces := 10
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_space_diagonals_l509_50946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_l509_50950

/-- If the terminal side of angle α passes through point (3,4), then cos(2α) = -7/25 -/
theorem cos_double_angle_special (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * (Real.cos α) = 3 ∧ r * (Real.sin α) = 4) → 
  Real.cos (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_l509_50950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_is_4_l509_50907

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + Real.exp ((2 - x) * Real.log 2)

-- Theorem stating that the minimum value of f(x) is 4
theorem f_min_value_is_4 : ∀ x : ℝ, f x ≥ 4 ∧ ∃ x₀ : ℝ, f x₀ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_is_4_l509_50907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABF2_area_ABF2_l509_50973

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 7, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 7, 0)

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + Real.sqrt 7

-- Define points A and B as the intersection of line_l and the ellipse
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Theorem for the perimeter of triangle ABF2
theorem perimeter_ABF2 : 
  ∀ (A B : ℝ × ℝ), 
  ellipse A.1 A.2 → ellipse B.1 B.2 → 
  line_l A.1 A.2 → line_l B.1 B.2 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
  Real.sqrt ((A.1 - F2.1)^2 + (A.2 - F2.2)^2) + 
  Real.sqrt ((B.1 - F2.1)^2 + (B.2 - F2.2)^2) = 16 := by sorry

-- Theorem for the area of triangle ABF2
theorem area_ABF2 : 
  ∀ (A B : ℝ × ℝ), 
  ellipse A.1 A.2 → ellipse B.1 B.2 → 
  line_l A.1 A.2 → line_l B.1 B.2 →
  (1/2) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * 
  (|F2.2 - A.2 + F2.1 - A.1| / Real.sqrt 2) = 72 * Real.sqrt 14 / 25 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABF2_area_ABF2_l509_50973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_140_l509_50989

/-- Calculates the length of a bridge given train parameters and passing time. -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  let total_distance : ℝ := train_speed_ms * passing_time
  total_distance - train_length

/-- Proves that the bridge length is 140 meters given the specified conditions. -/
theorem bridge_length_is_140 :
  bridge_length 360 52 34.61538461538461 = 140 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_140_l509_50989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_invariant_l509_50940

-- Define a polygon type
def Polygon := List (ℚ × ℚ)

-- Function to get x-coordinates
def x_coordinates (p : Polygon) : List ℚ := p.map Prod.fst

-- Function to calculate midpoints
def midpoints (p : Polygon) : Polygon :=
  let n := p.length
  List.zipWith (λ a b => ((a.1 + b.1) / 2, (a.2 + b.2) / 2)) p (p.rotateLeft 1)

-- Main theorem
theorem midpoint_sum_invariant (Q : Polygon) :
  Q.length = 39 →
  (x_coordinates Q).sum = 117 →
  let R := midpoints Q
  let S := midpoints R
  let T := midpoints S
  (x_coordinates T).sum = 117 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_invariant_l509_50940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equations_l509_50939

/-- Given a set of equations, prove that x = 2, y = ∛6, z = 6 is a solution --/
theorem solution_satisfies_equations :
  ∃ (x y z : ℝ),
    x = 2 ∧
    y = (6 : ℝ) ^ (1/3 : ℝ) ∧
    z = 6 ∧
    z^x = y^(3*x) ∧
    (2 : ℝ)^z = 3 * 8^x ∧
    x * y * z = 36 :=
by
  -- Introduce the existential variables
  use 2, (6 : ℝ) ^ (1/3 : ℝ), 6
  
  -- Split the goal into individual components
  apply And.intro
  · rfl  -- x = 2
  apply And.intro
  · rfl  -- y = 6^(1/3)
  apply And.intro
  · rfl  -- z = 6
  apply And.intro
  · -- z^x = y^(3*x)
    sorry
  apply And.intro
  · -- (2 : ℝ)^z = 3 * 8^x
    sorry
  · -- x * y * z = 36
    sorry

#check solution_satisfies_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equations_l509_50939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_solves_system_system_solution_in_solution_set_l509_50922

noncomputable def ω : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2

def system (x y z : ℂ) : Prop :=
  x * (x - y) * (x - z) = 3 ∧
  y * (y - x) * (y - z) = 3 ∧
  z * (z - x) * (z - y) = 3

def solution_set : Set (ℂ × ℂ × ℂ) :=
  {(1, ω, ω^2), (1, ω^2, ω), (ω, 1, ω^2), (ω, ω^2, 1), (ω^2, 1, ω), (ω^2, ω, 1)}

theorem solution_set_solves_system :
  ∀ (s : ℂ × ℂ × ℂ), s ∈ solution_set → system s.1 s.2.1 s.2.2 := by
  sorry

theorem system_solution_in_solution_set :
  ∀ (x y z : ℂ), system x y z → (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_solves_system_system_solution_in_solution_set_l509_50922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l509_50998

noncomputable section

-- Define the curve C
def C (α : Real) : Real × Real :=
  (2 * Real.sqrt 3 * Real.cos α, 2 * Real.sin α)

-- Define the point P
def P : Real × Real := (4, 4)

-- Define the line l
def l (x y : Real) : Prop :=
  x - y - 10 = 0

-- Define the midpoint M
def M (α : Real) : Real × Real :=
  ((Real.sqrt 3 * Real.cos α + 2), (Real.sin α + 2))

-- Define the distance function from a point to a line
def distance_to_line (x y : Real) : Real :=
  |Real.sqrt 3 * Real.cos x - Real.sin x - 10| / Real.sqrt 2

-- State the theorem
theorem max_distance_to_line :
  ∃ (α : Real), α ∈ Set.Ioo 0 Real.pi ∧
  ∀ (β : Real), β ∈ Set.Ioo 0 Real.pi →
  distance_to_line (M α).1 (M α).2 ≥ distance_to_line (M β).1 (M β).2 ∧
  distance_to_line (M α).1 (M α).2 = 6 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l509_50998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_lines_l509_50988

/-- The circle in the problem -/
def problem_circle (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

/-- A line with equal intercepts on both axes -/
def equal_intercept_line (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = a * p.1 + b ∧ a ≠ 0 ∧ (a = -1 ∨ a * b + b = 0)}

/-- A line is tangent to the circle if it intersects the circle at exactly one point -/
def is_tangent (l : Set (ℝ × ℝ)) : Prop :=
  ∃! p : ℝ × ℝ, p ∈ l ∧ problem_circle p.1 p.2

theorem four_tangent_lines :
  ∃ (l₁ l₂ l₃ l₄ : Set (ℝ × ℝ)),
    (∀ a b, l₁ = equal_intercept_line a b → is_tangent l₁) ∧
    (∀ a b, l₂ = equal_intercept_line a b → is_tangent l₂) ∧
    (∀ a b, l₃ = equal_intercept_line a b → is_tangent l₃) ∧
    (∀ a b, l₄ = equal_intercept_line a b → is_tangent l₄) ∧
    l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₁ ≠ l₄ ∧ l₂ ≠ l₃ ∧ l₂ ≠ l₄ ∧ l₃ ≠ l₄ ∧
    (∀ l : Set (ℝ × ℝ), (∃ a b, l = equal_intercept_line a b) →
      is_tangent l → (l = l₁ ∨ l = l₂ ∨ l = l₃ ∨ l = l₄)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_lines_l509_50988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l509_50994

def v : Fin 2 → ℚ := ![3, 4]
def w : Fin 2 → ℚ := ![1, 2]

def dot_product (u v : Fin 2 → ℚ) : ℚ :=
  Finset.sum (Finset.range 2) (λ i => u i * v i)

def proj (u v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  λ i => (dot_product u v / dot_product u u) * u i

theorem projection_theorem :
  proj w v = ![11/5, 22/5] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l509_50994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_lambda_l509_50949

theorem equilateral_triangle_lambda (ω : ℂ) (l : ℝ) : 
  Complex.abs ω = 3 →
  l > 1 →
  (let vertices := [ω, ω^2, l • ω]
   ∀ (i j k : Fin 3), i ≠ j → j ≠ k → k ≠ i → 
     Complex.abs (vertices.get i - vertices.get j) = Complex.abs (vertices.get j - vertices.get k)) →
  l = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_lambda_l509_50949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l509_50927

/-- Regular square pyramid with edge length a -/
structure RegularSquarePyramid (a : ℝ) where
  -- All edges have length a
  edge_length : ℝ

/-- Cross-section of a regular square pyramid through two non-adjacent side edges -/
structure CrossSection (p : RegularSquarePyramid a) where
  -- The cross-section is made through two non-adjacent side edges
  non_adjacent_edges : Bool

/-- The area of a cross-section in a regular square pyramid -/
noncomputable def cross_section_area (p : RegularSquarePyramid a) (s : CrossSection p) : ℝ :=
  1/2 * p.edge_length^2

/-- Theorem: The area of a cross-section SAC in a regular square pyramid S-ABCD 
    with all edge lengths equal to a is 1/2 * a^2 -/
theorem cross_section_area_theorem (a : ℝ) (p : RegularSquarePyramid a) (s : CrossSection p) :
  cross_section_area p s = 1/2 * a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l509_50927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_sum_l509_50966

theorem quadratic_coefficient_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (f = λ x ↦ a*x^2 + b*x + c) →
  (a = 2 ∧ b = -3 ∧ c = 4) →
  (∃ (y : ℤ), f 2 = y) →
  2*a - 3*b + 4*c = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_sum_l509_50966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_proof_l509_50983

/-- The radius of a circle inscribed in a sector that is one-third of a circle with radius 4 cm -/
noncomputable def inscribed_circle_radius : ℝ :=
  2 * Real.sqrt 3 - 2

/-- The theorem stating that the radius of the inscribed circle is 2√3 - 2 cm -/
theorem inscribed_circle_radius_proof :
  let sector_radius : ℝ := 4
  let sector_angle : ℝ := 2 * Real.pi / 3  -- One-third of a full circle
  inscribed_circle_radius = 2 * Real.sqrt 3 - 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_proof_l509_50983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l509_50995

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem inequality_proof (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m ≥ n) :
  (2 ^ n) * factorial n ≤ (m + n).factorial / (m - n).factorial ∧ 
  (m + n).factorial / (m - n).factorial ≤ (m^2 + m)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l509_50995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_transformations_l509_50961

/-- Represents a triplet of integers -/
structure Triplet :=
  (a b c : ℤ)

/-- The allowed transformations on a triplet -/
inductive Transform : Triplet → Triplet → Prop
  | permute (t : Triplet) : 
      Transform t ⟨t.c, t.a, t.b⟩  -- Example permutation, others are similar
  | change (t : Triplet) : 
      Transform t ⟨2*t.b + 2*t.c - t.a, t.b, t.c⟩

/-- The relation of being obtainable through a finite number of transformations -/
def Obtainable (start finish : Triplet) : Prop :=
  ∃ (n : ℕ) (chain : ℕ → Triplet), 
    chain 0 = start ∧ 
    chain n = finish ∧ 
    ∀ i : ℕ, i < n → Transform (chain i) (chain (i + 1))

/-- The sum of elements modulo 2 -/
def SumMod2 (t : Triplet) : Fin 2 :=
  (((t.a + t.b + t.c) : ℤ).toNat % 2 : Fin 2)

/-- The quadratic invariant -/
def QuadraticInvariant (t : Triplet) : ℤ :=
  t.a^2 + t.b^2 + t.c^2 - 2*(t.a*t.b + t.b*t.c + t.c*t.a)

theorem triplet_transformations :
  let start := Triplet.mk (-1) 0 1
  ∀ (finish : Triplet), Obtainable start finish →
    (SumMod2 finish = SumMod2 start ∧
     QuadraticInvariant finish = 4 ∧
     (finish = Triplet.mk 1 2024 2034 ∨ finish = Triplet.mk 1 2024 2016 ∨
      finish ≠ Triplet.mk 1 2024 2034 ∧ finish ≠ Triplet.mk 1 2024 2016)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_transformations_l509_50961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l509_50997

/-- Calculates the total profit for a business partnership given initial investments,
    changes after 8 months, and one partner's share of the profit. -/
def calculate_total_profit (a_initial : ℕ) (b_initial : ℕ) (a_change : ℤ) (b_change : ℕ) 
                            (a_profit_share : ℕ) : ℕ :=
  let a_investment_months := (a_initial * 8 : ℤ) + ((a_initial : ℤ) + a_change) * 4
  let b_investment_months := (b_initial * 8 : ℕ) + (b_initial + b_change) * 4
  let total_ratio := a_investment_months + b_investment_months
  let a_ratio := a_investment_months
  ((a_profit_share : ℤ) * total_ratio / a_ratio).toNat

theorem total_profit_calculation : 
  calculate_total_profit 6000 4000 (-1000) 1000 357 = 630 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l509_50997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximation_theorem_l509_50982

noncomputable def approximateAndDivide (α : ℝ) : ℝ :=
  let α₄ := ⌊α * 10000⌋ / 10000
  ⌊(α₄ / α) * 10000⌋ / 10000

def possibleValues : Set ℝ :=
  {x | ∃ k : ℕ, k ≤ 10000 ∧ x = k / 10000}

theorem approximation_theorem :
  ∀ α : ℝ, α > 0 → approximateAndDivide α ∈ possibleValues :=
by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximation_theorem_l509_50982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l509_50920

-- Define the function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x + b / x

-- Define the derivative of f(x)
noncomputable def f_derivative (b : ℝ) (x : ℝ) : ℝ := 1 - b / (x^2)

-- Theorem statement
theorem f_monotone_increasing (b : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, f_derivative b x = 0) →
  StrictMonoOn (f b) (Set.Iic (-2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l509_50920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_angle_measure_l509_50921

/-- The measure of an angle, in degrees -/
def AngleMeasure (P Q R : Point) : ℝ :=
  sorry

/-- Predicate stating that a circle with center O is tangent to each side of triangle ABC -/
def CircleTangentToTriangle (O A B C : Point) : Prop :=
  sorry

/-- Given a triangle ABC with an inscribed circle centered at O,
    if ∠BAC = 50° and ∠BCO = 20°, then ∠ACB = 40°. -/
theorem inscribed_circle_angle_measure (A B C O : Point) (h1 : CircleTangentToTriangle O A B C)
    (h2 : AngleMeasure B A C = 50) (h3 : AngleMeasure B C O = 20) :
    AngleMeasure A C B = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_angle_measure_l509_50921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_gcd_difference_l509_50915

theorem sum_gcd_difference (x y : ℕ) : 
  x + y = 50 → Nat.gcd x y = 5 → (x - y = 20 ∨ y - x = 20 ∨ x - y = 40 ∨ y - x = 40) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_gcd_difference_l509_50915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_eight_initial_l509_50902

/-- Represents the allowed operations on the board -/
inductive BoardOperation : ℕ → ℕ → Prop
  | double_plus_one (x : ℕ) : BoardOperation x (2 * x + 1)
  | fraction (x : ℕ) : BoardOperation x (x / (x + 2))

/-- Represents a sequence of operations on the board -/
def ReachableNumber (start finish : ℕ) : Prop :=
  ∃ (n : ℕ) (seq : Fin (n + 1) → ℕ),
    seq 0 = start ∧
    seq n = finish ∧
    ∀ (i : Fin n), BoardOperation (seq i) (seq i.succ)

/-- The main theorem: if 2008 is reachable, it must have been the starting number -/
theorem two_thousand_eight_initial (start : ℕ) :
  ReachableNumber start 2008 → start = 2008 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_eight_initial_l509_50902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l509_50963

-- Define the sets and conditions
def U : Set ℝ := {x | x ≤ 4}
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

noncomputable def α : ℝ := Real.arctan 3

-- State the theorem
theorem problem_solution :
  (((U \ A) ∩ B) = {-3, -2, 3}) ∧
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) ∧
  (Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α = -4/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l509_50963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_polar_l509_50931

/-- The distance from a point to a line in polar coordinates -/
noncomputable def distance_point_to_line_polar (r : ℝ) (θ : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  (|a * (r * Real.cos θ) + b * (r * Real.sin θ) + c|) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from point A(2, π/2) to the line ρcos(θ + π/4) = √2 is 2√2 -/
theorem distance_to_line_polar : 
  distance_point_to_line_polar 2 (Real.pi / 2) 1 (-1) (-2) = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_polar_l509_50931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_49_l509_50944

def divisors_of_245 : List Nat := [5, 7, 35, 49, 245]

-- Function to check if two numbers have a common factor greater than 1
def has_common_factor_greater_than_one (a b : Nat) : Prop :=
  ∃ (f : Nat), f > 1 ∧ a % f = 0 ∧ b % f = 0

-- Function to check if a list represents a valid circular arrangement
def is_valid_circular_arrangement (lst : List Nat) : Prop :=
  ∀ i : Nat, i < lst.length →
    has_common_factor_greater_than_one (lst.get! i) (lst.get! ((i + 1) % lst.length))

-- Theorem statement
theorem sum_of_adjacent_to_49 (arrangement : List Nat)
  (h1 : arrangement.toFinset = divisors_of_245.toFinset)
  (h2 : is_valid_circular_arrangement arrangement)
  (h3 : 49 ∈ arrangement) :
  ∃ i : Nat, i < arrangement.length ∧ arrangement.get! i = 49 ∧
    arrangement.get! ((i - 1 + arrangement.length) % arrangement.length) +
    arrangement.get! ((i + 1) % arrangement.length) = 280 := by
  sorry

#check sum_of_adjacent_to_49

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_49_l509_50944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_approx_13_l509_50901

/-- The distance to the park in miles -/
noncomputable def distance : ℝ := 8

/-- Anna's travel time in hours -/
noncomputable def anna_time : ℝ := 15 / 60

/-- Ben's travel time in hours -/
noncomputable def ben_time : ℝ := 25 / 60

/-- Anna's average speed in miles per hour -/
noncomputable def anna_speed : ℝ := distance / anna_time

/-- Ben's average speed in miles per hour -/
noncomputable def ben_speed : ℝ := distance / ben_time

/-- The difference in average speeds -/
noncomputable def speed_difference : ℝ := anna_speed - ben_speed

theorem speed_difference_approx_13 : 
  12.5 < speed_difference ∧ speed_difference < 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_approx_13_l509_50901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l509_50913

noncomputable def a : ℝ × ℝ := (-8, 1)
noncomputable def b : ℝ × ℝ := (3, 4)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def scalar_projection (v w : ℝ × ℝ) : ℝ := (dot_product v w) / (magnitude w)

theorem projection_a_onto_b :
  scalar_projection a b = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l509_50913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_symmetry_axis_l509_50975

-- Define the cosine function as noncomputable
noncomputable def f (ω φ x : ℝ) : ℝ := Real.cos (ω * x + φ)

-- State the theorem
theorem cos_symmetry_axis (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < Real.pi) :
  (∀ x, f ω φ x = -f ω φ (-x)) →  -- Odd function
  (∃ A B, f ω φ A = 1 ∧ f ω φ B = -1) →  -- Max and min points exist
  (∃ x, ∀ y, f ω φ (x + y) = f ω φ (x - y)) ∧  -- Symmetry axis exists
  (∃ x, ∀ y, f ω φ (x + y) = f ω φ (x - y) ∧ x = 1)  -- x = 1 is a symmetry axis
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_symmetry_axis_l509_50975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_value_nonnegative_implies_a_range_l509_50972

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (2*a - 1)/x - 2*a * Real.log x

-- Part I: Extremum at x=2 implies a = 3/2
theorem extremum_implies_a_value (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f a x ≥ f a 2) →
  a = 3/2 := by sorry

-- Part II: f(x) ≥ 0 for x ∈ [1, +∞) implies a ≤ 1
theorem nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) →
  a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_value_nonnegative_implies_a_range_l509_50972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_bound_l509_50945

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle defined by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Check if a point is inside or on a rectangle -/
def Point.inRectangle (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem -/
theorem smallest_distance_bound (points : Fin 6 → Point) (r : Rectangle)
    (h_rect : r.width = 2 ∧ r.height = 1)
    (h_points : ∀ i, (points i).inRectangle r) :
    ∃ i j, i ≠ j ∧ distance (points i) (points j) ≤ Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_bound_l509_50945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_factorial_sum_l509_50904

theorem largest_prime_factor_of_factorial_sum : 
  (Nat.factorial 7 + Nat.factorial 8).factors.maximum = some 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_factorial_sum_l509_50904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_explicit_and_m_range_l509_50933

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := lg ((2 * x) / (a * x + b))

-- State the theorem
theorem f_explicit_and_m_range :
  ∀ a b : ℝ,
  (∀ x > 0, f a b x - f a b (1/x) = lg x) →
  f a b 1 = 0 →
  (∀ x > 0, f a b x = lg ((2 * x) / (x + 1))) ∧
  (∀ m : ℝ, (∀ x > 0, f a b x ≠ lg (m + x)) ↔ m ∈ Set.Ioo (3 - 2 * Real.sqrt 2) (3 + 2 * Real.sqrt 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_explicit_and_m_range_l509_50933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l509_50992

-- Define the statements p and q as functions of m
noncomputable def p (m : ℝ) : Prop := ∃ x : ℝ, Real.cos (2 * x) - Real.sin x + 2 ≤ m

def q (m : ℝ) : Prop := ∀ x y : ℝ, 2 ≤ x → x < y → 
  (1/3 : ℝ) ^ (2 * x^2 - m * x + 2) > (1/3 : ℝ) ^ (2 * y^2 - m * y + 2)

-- Define the set of m values that satisfy the conditions
def S : Set ℝ := {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- State the theorem
theorem range_of_m : S = {m : ℝ | m < 0 ∨ m > 8} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l509_50992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_conversion_l509_50938

noncomputable def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := Real.arctan (y / x)
  let φ := Real.arccos (z / ρ)
  (ρ, θ, φ)

theorem rectangular_to_spherical_conversion :
  let (ρ, θ, φ) := rectangular_to_spherical 4 (4 * Real.sqrt 2) 4
  ρ = 8 ∧ θ = Real.arctan (Real.sqrt 2) ∧ φ = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_conversion_l509_50938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_integers_pairs_l509_50909

theorem circle_integers_pairs (nums : List ℕ) (h1 : nums.length = 2009) (h2 : nums.sum = 7036) :
  ∃ (i j : ℕ), i ≠ j ∧
  i < nums.length ∧ j < nums.length ∧
  (nums[i]! + nums[(i + 1) % nums.length]! ≥ 8) ∧
  (nums[j]! + nums[(j + 1) % nums.length]! ≥ 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_integers_pairs_l509_50909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_side_length_l509_50951

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapezium_area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem: Given a trapezium with one side 20 cm, height 10 cm, and area 190 cm², 
    the other side is 18 cm -/
theorem other_side_length (t : Trapezium) 
    (h1 : t.side1 = 20)
    (h2 : t.height = 10)
    (h3 : t.area = 190)
    (h4 : t.area = trapezium_area t) :
    t.side2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_side_length_l509_50951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_510_degrees_third_quadrant_l509_50976

-- Define a function to normalize an angle to the range [0, 360)
noncomputable def normalizeAngle (angle : ℝ) : ℝ :=
  (angle % 360 + 360) % 360

-- Define a function to determine the quadrant of an angle
noncomputable def getQuadrant (angle : ℝ) : ℕ :=
  let normalizedAngle := normalizeAngle angle
  if 0 ≤ normalizedAngle ∧ normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle ∧ normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle ∧ normalizedAngle < 270 then 3
  else 4

-- Theorem statement
theorem negative_510_degrees_third_quadrant :
  getQuadrant (-510) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_510_degrees_third_quadrant_l509_50976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_number_power_l509_50935

/-- A positive real number is "good" if it can be written as √n + √(n+1) for some n ∈ ℕ₊ -/
def is_good_number (x : ℝ) : Prop :=
  ∃ n : ℕ+, x = Real.sqrt n + Real.sqrt (n + 1)

/-- The main theorem: if x is a good number, then xʳ is also a good number for any r ∈ ℕ₊ -/
theorem good_number_power (x : ℝ) (r : ℕ+) (hx : is_good_number x) : is_good_number (x ^ (r : ℕ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_number_power_l509_50935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_bidirectional_crossing_l509_50971

/-- Represents a line on a plane -/
structure Line where
  -- Add necessary fields
  mk :: -- Constructor

/-- Represents a point on a plane -/
structure Point where
  -- Add necessary fields
  mk :: -- Constructor

/-- Represents the direction of movement -/
inductive Direction
  | Left
  | Right

/-- Represents the state of the snail -/
structure SnailState where
  position : Point
  direction : Direction

/-- Checks if three lines are concurrent -/
def areConcurrent (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Checks if a point is an intersection of two lines -/
def isIntersection (p : Point) (l1 l2 : Line) : Prop :=
  sorry

/-- Represents the movement of the snail -/
def snailMove (state : SnailState) (lines : List Line) : SnailState :=
  sorry

/-- Theorem: No line segment can be crossed in both directions -/
theorem no_bidirectional_crossing (n : Nat) (lines : List Line) 
  (h1 : lines.length = n)
  (h2 : ∀ l1 l2 l3, l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → ¬(areConcurrent l1 l2 l3))
  (initialState : SnailState) :
  ¬∃ (p1 p2 : Point) (l : Line), l ∈ lines ∧ 
    (∃ (s1 s2 : SnailState), 
      s1 = snailMove initialState lines ∧
      s2 = snailMove s1 lines ∧
      isIntersection p1 l (Line.mk) ∧
      isIntersection p2 l (Line.mk) ∧
      (s1.position = p1 ∧ s2.position = p2 ∨ s1.position = p2 ∧ s2.position = p1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_bidirectional_crossing_l509_50971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_schedule_l509_50956

/-- Represents a team in the tournament -/
inductive Team : Type
| P | Q | R | S | T | U

/-- Represents a round in the tournament -/
inductive Round : Type
| one | two | three | four | five

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the schedule of matches for the tournament -/
def Schedule := Round → List Match

/-- Checks if a schedule is valid according to the tournament rules -/
def is_valid_schedule (schedule : Schedule) : Prop :=
  (∀ r : Round, (schedule r).length = 3) ∧
  (∀ t1 t2 : Team, t1 ≠ t2 → ∃! r : Round, (Match.mk t1 t2 ∈ schedule r) ∨ (Match.mk t2 t1 ∈ schedule r)) ∧
  (∀ r : Round, ∀ m1 m2 : Match, m1 ∈ schedule r → m2 ∈ schedule r → m1 ≠ m2 → 
    m1.team1 ≠ m2.team1 ∧ m1.team1 ≠ m2.team2 ∧ m1.team2 ≠ m2.team1 ∧ m1.team2 ≠ m2.team2)

theorem tournament_schedule (schedule : Schedule) 
  (h_valid : is_valid_schedule schedule)
  (h_P_Q : Match.mk Team.P Team.Q ∈ schedule Round.one)
  (h_P_T : Match.mk Team.P Team.T ∈ schedule Round.three)
  (h_P_R : Match.mk Team.P Team.R ∈ schedule Round.five)
  (h_T_Q : Match.mk Team.T Team.Q ∈ schedule Round.two)
  (h_T_U : Match.mk Team.T Team.U ∈ schedule Round.four) :
  Match.mk Team.S Team.U ∈ schedule Round.one ∨ Match.mk Team.U Team.S ∈ schedule Round.one :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_schedule_l509_50956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_l509_50962

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point on a circle --/
def PointOnCircle (c : Circle) := { p : ℝ × ℝ // (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 }

/-- The angle bisector from a vertex of a triangle --/
def AngleBisector (A B C : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The median from a vertex of a triangle --/
def Median (A B C : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The altitude from a vertex of a triangle --/
def Altitude (A B C : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Check if a point is on a circle --/
def IsOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Main theorem --/
theorem triangle_construction (c : Circle) (F S M : PointOnCircle c) 
  (hFSM : F ≠ S ∧ S ≠ M ∧ M ≠ F) :
  ∃ (A B C : ℝ × ℝ), 
    IsOnCircle A c ∧ IsOnCircle B c ∧ IsOnCircle C c ∧
    F.val ∈ AngleBisector A B C ∧
    S.val ∈ Median A B C ∧
    M.val ∈ Altitude A B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_l509_50962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_theorem_l509_50943

/-- 
Given an angle α where:
- Its initial side coincides with the positive x-axis
- Its terminal side falls on the line x + 2y = 0

This theorem proves that (sin α + cos α) / (sin α - cos α) = -1/3
-/
theorem angle_ratio_theorem (α : Real) 
  (h : Real.tan α = -1/2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_theorem_l509_50943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_correspondence_l509_50964

/-- Given a ratio of 15 to 5 seconds, find the corresponding number for 600 seconds -/
theorem ratio_correspondence : ∃ x : ℝ, x / 600 = 15 / 5 ∧ x = 1800 := by
  -- Define the original ratio
  let original_ratio : ℝ := 15 / 5
  
  -- Define the target time in seconds (10 minutes = 600 seconds)
  let target_time : ℝ := 600

  -- Prove the theorem
  use 1800
  constructor
  · -- Prove that 1800 / 600 = 15 / 5
    norm_num
  · -- Prove that x = 1800
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_correspondence_l509_50964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l509_50930

/-- The probability of selecting two balls of different colors from a set of 2 yellow balls and 3 red balls -/
theorem different_color_probability : 
  let total_balls : ℕ := 5
  let yellow_balls : ℕ := 2
  let red_balls : ℕ := 3
  (3 : ℚ) / 5 = (yellow_balls * red_balls : ℚ) / (total_balls * (total_balls - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l509_50930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_inequality_l509_50953

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define angle bisector lengths x, y, z
noncomputable def angle_bisector_length (t : Triangle) : ℝ × ℝ × ℝ :=
  let x := Real.sqrt (t.b * t.c * (1 - t.a^2 / (t.b + t.c)^2))
  let y := Real.sqrt (t.a * t.c * (1 - t.b^2 / (t.a + t.c)^2))
  let z := Real.sqrt (t.a * t.b * (1 - t.c^2 / (t.a + t.b)^2))
  (x, y, z)

-- State the theorem
theorem angle_bisector_inequality (t : Triangle) :
  let (x, y, z) := angle_bisector_length t
  1/x + 1/y + 1/z > 1/t.a + 1/t.b + 1/t.c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_inequality_l509_50953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l509_50985

theorem max_value_of_expression (a b c : ℚ) : 
  a ∈ ({2, 3, 6} : Set ℚ) → 
  b ∈ ({2, 3, 6} : Set ℚ) → 
  c ∈ ({2, 3, 6} : Set ℚ) → 
  a ≠ b → b ≠ c → a ≠ c → 
  a / (b / c) ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l509_50985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_price_calculation_l509_50957

theorem coat_price_calculation (original_price discount_rate tax_rate : ℝ) 
  (h1 : original_price = 120)
  (h2 : discount_rate = 0.30)
  (h3 : tax_rate = 0.15) : 
  original_price * (1 - discount_rate) * (1 + tax_rate) = 96.60 := by
  -- Proof steps would go here
  sorry

#eval (120 : ℝ) * (1 - 0.30) * (1 + 0.15)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_price_calculation_l509_50957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tall_students_count_l509_50987

theorem tall_students_count (total : ℕ) (short_ratio : ℚ) (average_height : ℕ) 
  (h1 : total = 400)
  (h2 : short_ratio = 2 / 5)
  (h3 : average_height = 150)
  : total - (short_ratio * ↑total).floor - average_height = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tall_students_count_l509_50987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_problem_quadratic_equation_solution_l509_50911

-- Problem 1
theorem sqrt_problem :
  Real.sqrt 18 - Real.sqrt 24 / Real.sqrt 3 = Real.sqrt 2 := by sorry

-- Problem 2
theorem quadratic_equation_solution :
  ∃ (f : ℝ → ℝ), (f = λ x => x^2 - 4*x - 5) ∧
  (f 5 = 0) ∧ (f (-1) = 0) ∧ (∀ x, f x = 0 → x = 5 ∨ x = -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_problem_quadratic_equation_solution_l509_50911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_ratio_l509_50958

theorem line_segment_ratio (x y z : ℝ) (Q : ℝ) (h1 : x < y) (h2 : y < z) 
  (h3 : x / z = z / (x + y + z)) (h4 : Q = x / z) : 
  Q^(Q^(Q^2 + Q⁻¹) + Q⁻¹) + Q⁻¹ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_ratio_l509_50958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l509_50999

noncomputable section

open Real

def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_given : (cos A) / a + (cos B) / b = (2 * c * cos C) / (a * b)) :
  C = Real.pi / 3 ∧ (a = 2 → c = sqrt 5 → b = 1 + sqrt 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l509_50999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_cube_l509_50952

/-- If a natural number n has exactly 4 divisors, then n^3 has exactly 10 divisors -/
theorem divisors_of_cube (n : ℕ) (h : (Nat.divisors n).card = 4) : (Nat.divisors (n^3)).card = 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_cube_l509_50952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_sphere_l509_50965

/-- A rectangular solid with a square base -/
structure RectangularSolid where
  base_length : ℝ
  height : ℝ

/-- The sum of the lengths of the edges is 24 -/
def edge_sum (s : RectangularSolid) : ℝ := 8 * s.base_length + 4 * s.height

/-- The volume of the rectangular solid -/
def volume (s : RectangularSolid) : ℝ := s.base_length^2 * s.height

/-- The volume of the circumscribed sphere -/
noncomputable def sphere_volume (s : RectangularSolid) : ℝ :=
  (4 / 3) * Real.pi * ((s.base_length^2 + s.base_length^2 + s.height^2) / 4)^(3/2)

/-- The theorem to be proved -/
theorem max_volume_sphere (s : RectangularSolid) :
  edge_sum s = 24 →
  (∀ t : RectangularSolid, edge_sum t = 24 → volume t ≤ volume s) →
  sphere_volume s = 4 * Real.pi * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_sphere_l509_50965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l509_50979

-- Define the given parameters
noncomputable def train_speed_kmh : ℝ := 54
noncomputable def time_pass_platform : ℝ := 50
noncomputable def time_pass_man : ℝ := 20

-- Convert km/h to m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600

-- Define the theorem
theorem platform_length : ∃ (length : ℝ), length = train_speed_ms * time_pass_platform - train_speed_ms * time_pass_man := by
  -- The length of the platform is 450 meters
  use 450
  sorry

#check platform_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l509_50979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_difference_count_l509_50974

theorem set_difference_count {U A B : Finset ℕ} : 
  (Finset.card U = 192) →
  (Finset.card A = 107) →
  (Finset.card B = 49) →
  (Finset.card (A ∩ B) = 23) →
  Finset.card (U \ (A ∪ B)) = 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_difference_count_l509_50974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_ratio_l509_50990

theorem gift_ratio : 
  (14 : ℚ) / (28 : ℚ) = (1 : ℚ) / (2 : ℚ) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_ratio_l509_50990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_donation_theorem_l509_50924

/-- The new average contribution size after John's donation -/
noncomputable def new_average_contribution (initial_contribution : ℝ) : ℝ :=
  (initial_contribution + 100) / 2

/-- The increase in average contribution size -/
noncomputable def average_increase (initial_contribution : ℝ) : ℝ :=
  (new_average_contribution initial_contribution) / (initial_contribution / 1) - 1

theorem john_donation_theorem (initial_contribution : ℝ) :
  average_increase initial_contribution = 0.5 →
  new_average_contribution initial_contribution = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_donation_theorem_l509_50924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_problem_l509_50996

theorem sum_reciprocal_problem : (8 : ℚ) * ((1/3 + 1/4 + 1/12)⁻¹ : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_problem_l509_50996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_4_digit_square_base_8_l509_50937

/-- The largest integer whose square has exactly 4 digits in base 8 -/
def N : ℕ := 63

/-- Converts a natural number to its base 8 representation -/
def to_base_8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 8) :: aux (m / 8)
  aux n |>.reverse

/-- Checks if a number has exactly 4 digits when written in base 8 -/
def has_4_digits_base_8 (n : ℕ) : Prop :=
  (to_base_8 n).length = 4

theorem largest_4_digit_square_base_8 :
  N = 63 ∧ 
  has_4_digits_base_8 (N^2) ∧
  ∀ m : ℕ, m > N → ¬has_4_digits_base_8 (m^2) :=
by
  sorry -- Placeholder for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_4_digit_square_base_8_l509_50937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_iff_m_zero_or_one_l509_50968

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of line l₁: mx + y - 2 = 0 -/
noncomputable def slope_l₁ (m : ℝ) : ℝ := -m

/-- The slope of line l₂: (m + 1)x - 2my + 1 = 0 -/
noncomputable def slope_l₂ (m : ℝ) : ℝ := 
  if m = 0 then 0 else (m + 1) / (-2 * m)

/-- Theorem: Lines l₁ and l₂ are perpendicular if and only if m = 0 or m = 1 -/
theorem lines_perpendicular_iff_m_zero_or_one :
  ∀ m : ℝ, (m = 0 ∨ m = 1) ↔ 
    (perpendicular (slope_l₁ m) (slope_l₂ m) ∨ m = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_iff_m_zero_or_one_l509_50968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_parabola_intersection_l509_50969

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a parabola with focus at (p, 0) -/
structure Parabola where
  p : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The main theorem -/
theorem ellipse_and_parabola_intersection
  (C₁ : Ellipse)
  (C₂ : Parabola)
  (h_focus : C₂.p = 1) -- Focus of parabola at (1, 0)
  (P : Point)
  (h_intersection : P.y^2 = 4 * P.x ∧ P.x^2 / C₁.a^2 + P.y^2 / C₁.b^2 = 1)
  (h_distance : distance P ⟨1, 0⟩ = 5/3)
  : C₁.a^2 = 4 ∧ C₁.b^2 = 3 ∧
    ∀ (R : Point), 4 * R.y^2 + 3 * (R.x^2 + 4 * R.x + 3) = 0 ↔
      ∃ (M N : Point),
        M.x^2 / 4 + M.y^2 / 3 = 1 ∧
        N.x^2 / 4 + N.y^2 / 3 = 1 ∧
        (M.x - 1) + (N.x - 1) = R.x - 1 ∧
        M.y + N.y = R.y ∧
        (R.x + 3) * (M.y - N.y) = (M.x - N.x) * R.y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_parabola_intersection_l509_50969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_win_probability_host_opens_door3_probability_switching_increases_win_probability_l509_50929

-- Define the set of doors
inductive Door : Type
| door1 : Door
| door2 : Door
| door3 : Door

-- Define the game state
structure MontyHallGame where
  prize_door : Door
  initial_choice : Door
  opened_door : Door

-- Define the probability space
def prob_space : Type := Door → ℝ

-- Axioms for the problem
axiom prob_uniform : ∀ (p : prob_space), p Door.door1 + p Door.door2 + p Door.door3 = 1
axiom prob_positive : ∀ (p : prob_space) (d : Door), p d ≥ 0

-- Initial choice probability
noncomputable def initial_win_prob (p : prob_space) : ℝ := p Door.door1

-- Probability of host opening door 3
noncomputable def host_opens_door3_prob (p : prob_space) : ℝ :=
  p Door.door1 * (1/2) + p Door.door2 * 1 + p Door.door3 * 0

-- Probability of winning by switching after host opens door 3
noncomputable def switch_win_prob (p : prob_space) : ℝ :=
  (p Door.door2 * 1) / (p Door.door1 * (1/2) + p Door.door2 * 1)

-- Probability of winning by staying after host opens door 3
noncomputable def stay_win_prob (p : prob_space) : ℝ :=
  (p Door.door1 * (1/2)) / (p Door.door1 * (1/2) + p Door.door2 * 1)

-- Theorem statements
theorem initial_win_probability (p : prob_space) :
  initial_win_prob p = 1/3 := by sorry

theorem host_opens_door3_probability (p : prob_space) :
  host_opens_door3_prob p = 1/2 := by sorry

theorem switching_increases_win_probability (p : prob_space) :
  switch_win_prob p > stay_win_prob p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_win_probability_host_opens_door3_probability_switching_increases_win_probability_l509_50929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_percentage_is_zero_l509_50914

/-- Represents the financial data of Lucius's business --/
structure BusinessData where
  daily_expense : ℚ
  french_fries_price : ℚ
  poutine_price : ℚ
  weekly_earnings_after_expenses : ℚ

/-- Calculates the tax percentage given the business data --/
def calculate_tax_percentage (data : BusinessData) : ℚ :=
  let weekly_expense := data.daily_expense * 7
  let total_revenue := data.weekly_earnings_after_expenses + weekly_expense
  let tax_amount := total_revenue - data.weekly_earnings_after_expenses - weekly_expense
  (tax_amount / total_revenue) * 100

/-- Theorem stating that the tax percentage is 0% for Lucius's business --/
theorem tax_percentage_is_zero (data : BusinessData) 
  (h1 : data.daily_expense = 10)
  (h2 : data.french_fries_price = 12)
  (h3 : data.poutine_price = 8)
  (h4 : data.weekly_earnings_after_expenses = 56) :
  calculate_tax_percentage data = 0 := by
  sorry

#eval calculate_tax_percentage {
  daily_expense := 10,
  french_fries_price := 12,
  poutine_price := 8,
  weekly_earnings_after_expenses := 56
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_percentage_is_zero_l509_50914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_quadratic_roots_l509_50936

theorem smallest_a_for_quadratic_roots : ∃ (b c : ℤ),
  let f := fun (x : ℚ) => (1001000 : ℚ) * x^2 + (b : ℚ) * x + (c : ℚ)
  ∃ (x₁ x₂ : ℚ), x₁ ≠ x₂ ∧ 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≤ 1/1000 ∧ x₂ ≤ 1/1000 ∧
  f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (a : ℕ), a < 1001000 → ¬(∃ (b' c' : ℤ),
    let g := fun (x : ℚ) => (a : ℚ) * x^2 + (b' : ℚ) * x + (c' : ℚ)
    ∃ (y₁ y₂ : ℚ), y₁ ≠ y₂ ∧ 0 < y₁ ∧ 0 < y₂ ∧ y₁ ≤ 1/1000 ∧ y₂ ≤ 1/1000 ∧
    g y₁ = 0 ∧ g y₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_quadratic_roots_l509_50936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l509_50932

/-- Definition of the ellipse -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The right focus of the ellipse -/
def rightFocus : ℝ × ℝ := (1, 0)

/-- The point E where line x = 2 intersects x-axis -/
def E : ℝ × ℝ := (2, 0)

/-- Vector from origin to right focus -/
def OF : ℝ × ℝ := rightFocus

/-- Vector from right focus to E -/
def FE : ℝ × ℝ := (1, 0)

/-- The line x = 2 -/
def lineL : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2}

theorem ellipse_properties :
  ∃ (e : Set (ℝ × ℝ)), e = Ellipse (Real.sqrt 2) 1 ∧
    (∀ (A B : ℝ × ℝ), A ∈ e → B ∈ e → ∃ (k : ℝ), (A.2 - rightFocus.2) = k * (A.1 - rightFocus.1) ∧ 
                                                  (B.2 - rightFocus.2) = k * (B.1 - rightFocus.1) →
      ∀ (C D : ℝ × ℝ), C ∈ lineL → D ∈ lineL → 
        A.2 - D.2 = C.2 - B.2 →  -- AD parallel to BC
        ∃ (N : ℝ × ℝ), N = ((E.1 + rightFocus.1) / 2, 0) ∧  -- Midpoint of EF
                        ∃ (t : ℝ), (1 - t) • A + t • C = N) ∧  -- AC passes through midpoint
    FE = OF ∧
    (Real.sqrt 2 / 2 : ℝ) = (rightFocus.1 : ℝ) / Real.sqrt 2  -- Eccentricity
:= by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l509_50932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_of_cos_l509_50908

-- Define the arcsin function as the inverse of sin
noncomputable def arcsin (x : ℝ) : ℝ := Real.arcsin x

-- State the theorem
theorem arcsin_of_cos (x : ℝ) (h : x ∈ Set.Icc (-Real.pi) Real.pi) :
  arcsin (Real.cos x) = if x ≤ 0 then x + Real.pi/2 else Real.pi/2 - x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_of_cos_l509_50908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_inequality_condition_l509_50970

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)
def g (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Theorem for the tangent line condition
theorem tangent_line_condition (k : ℝ) :
  (∃ t : ℝ, (f t = g k t) ∧ (deriv f t = k)) ↔ k = 2 := by sorry

-- Theorem for the inequality condition
theorem inequality_condition (k : ℝ) :
  (k > 0 ∧ ∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, 0 < x ∧ x < m → |f x - g k x| > 2 * x) ↔ k > 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_inequality_condition_l509_50970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_unique_l509_50903

/-- A linear function passing through (2, 4) with integral from 0 to 1 equal to 3 -/
def LinearFunctionWithConditions (f : ℝ → ℝ) : Prop :=
  (∃ a b : ℝ, ∀ x, f x = a * x + b) ∧ 
  f 2 = 4 ∧ 
  (∫ x in (0:ℝ)..(1:ℝ), f x) = 3

theorem linear_function_unique : 
  ∀ f : ℝ → ℝ, LinearFunctionWithConditions f → f = λ x ↦ (2/3) * x + 8/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_unique_l509_50903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l509_50991

noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (x a : ℝ) : ℝ := 2^x + a

theorem range_of_a :
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂ a) ↔ 
  Set.Iic 1 = {a : ℝ | ∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂ a} :=
by sorry

-- Note: Set.Icc represents a closed interval [a, b]
-- Set.Iic represents an interval (-∞, b]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l509_50991
