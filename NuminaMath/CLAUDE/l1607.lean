import Mathlib

namespace NUMINAMATH_CALUDE_restore_exchange_rate_l1607_160778

/-- The exchange rate between Trade Federation's currency and Naboo's currency -/
structure ExchangeRate :=
  (rate : ℝ)

/-- The money supply of the Trade Federation -/
structure MoneySupply :=
  (supply : ℝ)

/-- The relationship between money supply changes and exchange rate changes -/
def money_supply_effect (ms_change : ℝ) : ℝ := 5 * ms_change

/-- The theorem stating the required change in money supply to restore the exchange rate -/
theorem restore_exchange_rate 
  (initial_rate : ExchangeRate)
  (new_rate : ExchangeRate)
  (money_supply : MoneySupply) :
  initial_rate.rate = 90 →
  new_rate.rate = 100 →
  (∀ (ms_change : ℝ), 
    ExchangeRate.rate (new_rate) * (1 - money_supply_effect ms_change / 100) = 
    ExchangeRate.rate (initial_rate)) →
  ∃ (ms_change : ℝ), ms_change = -2 :=
sorry

end NUMINAMATH_CALUDE_restore_exchange_rate_l1607_160778


namespace NUMINAMATH_CALUDE_x_value_proof_l1607_160720

theorem x_value_proof (x : ℝ) (h : 3/4 - 1/2 = 4/x) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1607_160720


namespace NUMINAMATH_CALUDE_double_reflection_of_D_l1607_160706

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the line y = x - 1 -/
def reflect_line_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 + 1, p.1 - 1)

/-- The composition of two reflections -/
def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_line_y_eq_x_minus_1 (reflect_y_axis p)

theorem double_reflection_of_D :
  double_reflection (7, 0) = (1, -8) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_of_D_l1607_160706


namespace NUMINAMATH_CALUDE_circle_properties_l1607_160753

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y - 4 = 0

-- Theorem statement
theorem circle_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_x = -2 ∧
    center_y = 1 ∧
    radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l1607_160753


namespace NUMINAMATH_CALUDE_prime_sum_power_implies_three_power_l1607_160748

theorem prime_sum_power_implies_three_power (n : ℕ) : 
  Nat.Prime (1 + 2^n + 4^n) → ∃ k : ℕ, n = 3^k :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_power_implies_three_power_l1607_160748


namespace NUMINAMATH_CALUDE_phoebe_peanut_butter_l1607_160780

/-- The number of jars of peanut butter needed for Phoebe and her dog for 30 days -/
def jars_needed (
  phoebe_servings : ℕ)  -- Phoebe's daily servings
  (dog_servings : ℕ)    -- Dog's daily servings
  (days : ℕ)            -- Number of days
  (servings_per_jar : ℕ) -- Servings per jar
  : ℕ :=
  ((phoebe_servings + dog_servings) * days + servings_per_jar - 1) / servings_per_jar

/-- Theorem stating the number of jars needed for Phoebe and her dog for 30 days -/
theorem phoebe_peanut_butter :
  jars_needed 1 1 30 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_phoebe_peanut_butter_l1607_160780


namespace NUMINAMATH_CALUDE_complement_of_A_l1607_160774

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x ≥ 1} ∪ {x : ℝ | x < 0}

theorem complement_of_A : Set.compl A = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1607_160774


namespace NUMINAMATH_CALUDE_distance_traveled_l1607_160757

/-- 
Given a speed of 20 km/hr and a time of 8 hr, prove that the distance traveled is 160 km.
-/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 20) (h2 : time = 8) :
  speed * time = 160 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l1607_160757


namespace NUMINAMATH_CALUDE_profit_maximization_l1607_160769

noncomputable def profit (x : ℝ) : ℝ := 20 - x - 4 / (x + 1)

theorem profit_maximization (a : ℝ) (h_a : a > 0) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ a^2 - 3*a + 3 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ a^2 - 3*a + 3 → profit x ≥ profit y) ∧
  ((a ≥ 2 ∨ 0 < a ∧ a ≤ 1) → x = 1) ∧
  (1 < a ∧ a < 2 → x = a^2 - 3*a + 3) :=
sorry

end NUMINAMATH_CALUDE_profit_maximization_l1607_160769


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l1607_160763

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 7) :
  ∃ (m : ℝ), m = -1 ∧ ∀ (x : ℝ), 8 * x^2 + 6 * x + 5 = 7 → 3 * x + 2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l1607_160763


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_max_perimeter_l1607_160723

theorem smallest_whole_number_above_max_perimeter : ∀ s : ℝ,
  s > 0 →
  s + 7 > 21 →
  s + 21 > 7 →
  7 + 21 > s →
  57 > 7 + 21 + s ∧ 
  ∀ n : ℕ, n < 57 → ∃ s' : ℝ, 
    s' > 0 ∧ 
    s' + 7 > 21 ∧ 
    s' + 21 > 7 ∧ 
    7 + 21 > s' ∧ 
    n ≤ 7 + 21 + s' :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_max_perimeter_l1607_160723


namespace NUMINAMATH_CALUDE_expression_simplification_l1607_160741

theorem expression_simplification (x : ℝ) (h : x^2 - 3*x - 4 = 0) :
  (x / (x + 1) - 2 / (x - 1)) / (1 / (x^2 - 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1607_160741


namespace NUMINAMATH_CALUDE_probability_two_females_l1607_160779

def total_students : ℕ := 5
def female_students : ℕ := 3
def male_students : ℕ := 2
def students_to_select : ℕ := 2

theorem probability_two_females :
  (Nat.choose female_students students_to_select : ℚ) / 
  (Nat.choose total_students students_to_select : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_l1607_160779


namespace NUMINAMATH_CALUDE_vectors_form_basis_l1607_160729

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : 
  LinearIndependent ℝ ![e₁, e₂] ∧ Submodule.span ℝ {e₁, e₂} = ⊤ :=
by sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l1607_160729


namespace NUMINAMATH_CALUDE_tan_theta_value_l1607_160708

theorem tan_theta_value (θ : Real) (x y : Real) : 
  x = -Real.sqrt 3 / 2 ∧ y = 1 / 2 → Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1607_160708


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l1607_160727

theorem maintenance_check_increase (old_time new_time : ℝ) (h1 : old_time = 45) (h2 : new_time = 60) :
  (new_time - old_time) / old_time * 100 = 33.33 := by
sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l1607_160727


namespace NUMINAMATH_CALUDE_calculate_expression_l1607_160798

theorem calculate_expression : 2 * Real.cos (45 * π / 180) - (π - 2023) ^ 0 + |3 - Real.sqrt 2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1607_160798


namespace NUMINAMATH_CALUDE_chess_group_players_l1607_160709

/-- The number of players in the chess group -/
def n : ℕ := 10

/-- The total number of games played -/
def total_games : ℕ := 45

/-- Theorem: Given the conditions, the number of players in the chess group is 10 -/
theorem chess_group_players :
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (game : ℕ), game < total_games) ∧
  (∀ (game : ℕ), game < total_games → ∃! (i j : ℕ), i < n ∧ j < n ∧ i ≠ j) ∧
  (n * (n - 1) / 2 = total_games) →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_chess_group_players_l1607_160709


namespace NUMINAMATH_CALUDE_odd_prime_sqrt_sum_l1607_160716

theorem odd_prime_sqrt_sum (p : ℕ) : 
  Prime p ↔ (∃ m : ℕ, ∃ n : ℕ, Real.sqrt m + Real.sqrt (m + p) = n) ∧ Odd p := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_sqrt_sum_l1607_160716


namespace NUMINAMATH_CALUDE_cows_that_ran_away_l1607_160701

/-- Represents the problem of determining how many cows ran away --/
theorem cows_that_ran_away 
  (initial_cows : ℕ) 
  (feeding_period : ℕ) 
  (days_passed : ℕ) 
  (h1 : initial_cows = 1000)
  (h2 : feeding_period = 50)
  (h3 : days_passed = 10)
  : ∃ (cows_ran_away : ℕ),
    cows_ran_away = 200 ∧ 
    (initial_cows * feeding_period - initial_cows * days_passed) 
    = (initial_cows - cows_ran_away) * feeding_period :=
by sorry


end NUMINAMATH_CALUDE_cows_that_ran_away_l1607_160701


namespace NUMINAMATH_CALUDE_painting_height_l1607_160759

theorem painting_height (wall_height wall_width painting_width : ℝ) 
  (wall_area painting_area : ℝ) (painting_percentage : ℝ) :
  wall_height = 5 →
  wall_width = 10 →
  painting_width = 4 →
  painting_percentage = 0.16 →
  wall_area = wall_height * wall_width →
  painting_area = painting_percentage * wall_area →
  painting_area = painting_width * 2 :=
by
  sorry

end NUMINAMATH_CALUDE_painting_height_l1607_160759


namespace NUMINAMATH_CALUDE_cos_sin_transformation_l1607_160786

theorem cos_sin_transformation (x : ℝ) :
  3 * Real.cos (2 * x) = 3 * Real.sin (2 * (x + π / 12) + π / 3) :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_transformation_l1607_160786


namespace NUMINAMATH_CALUDE_divisible_by_seven_l1607_160746

theorem divisible_by_seven (n : ℕ) : 7 ∣ (3^(2*n+1) + 2^(n+2)) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l1607_160746


namespace NUMINAMATH_CALUDE_heart_value_is_three_l1607_160755

/-- Represents a digit in base 9 and base 10 notation -/
def Heart : ℕ → Prop :=
  λ n => 0 ≤ n ∧ n ≤ 9

theorem heart_value_is_three :
  ∀ h : ℕ,
  Heart h →
  (h * 9 + 6 = h * 10 + 3) →
  h = 3 := by
sorry

end NUMINAMATH_CALUDE_heart_value_is_three_l1607_160755


namespace NUMINAMATH_CALUDE_solution_system_l1607_160750

theorem solution_system (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 88) : 
  x^2 + y^2 = 304/9 := by
sorry

end NUMINAMATH_CALUDE_solution_system_l1607_160750


namespace NUMINAMATH_CALUDE_nephews_count_l1607_160787

/-- The number of nephews Alden and Vihaan have altogether -/
def total_nephews (alden_past : ℕ) (increase : ℕ) : ℕ :=
  let alden_current := 2 * alden_past
  let vihaan := alden_current + increase
  alden_current + vihaan

/-- Theorem stating the total number of nephews Alden and Vihaan have -/
theorem nephews_count : total_nephews 50 60 = 260 := by
  sorry

end NUMINAMATH_CALUDE_nephews_count_l1607_160787


namespace NUMINAMATH_CALUDE_initial_ratio_of_liquids_l1607_160732

/-- Given a mixture of two liquids p and q with total volume 40 liters,
    if adding 15 liters of q results in a ratio of 5:6 for p:q,
    then the initial ratio of p:q was 5:3. -/
theorem initial_ratio_of_liquids (p q : ℝ) : 
  p + q = 40 →
  p / (q + 15) = 5 / 6 →
  p / q = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_of_liquids_l1607_160732


namespace NUMINAMATH_CALUDE_pressure_volume_inverse_proportionality_l1607_160724

/-- Given inverse proportionality of pressure and volume, prove that if the initial pressure is 8 kPa
    at 3.5 liters, then the pressure at 7 liters is 4 kPa. -/
theorem pressure_volume_inverse_proportionality
  (pressure volume : ℝ → ℝ) -- Pressure and volume as functions of time
  (t₀ t₁ : ℝ) -- Initial and final times
  (h_inverse_prop : ∀ t, pressure t * volume t = pressure t₀ * volume t₀) -- Inverse proportionality
  (h_init_volume : volume t₀ = 3.5)
  (h_init_pressure : pressure t₀ = 8)
  (h_final_volume : volume t₁ = 7) :
  pressure t₁ = 4 := by
  sorry

end NUMINAMATH_CALUDE_pressure_volume_inverse_proportionality_l1607_160724


namespace NUMINAMATH_CALUDE_probability_equals_three_over_646_l1607_160713

-- Define the cube
def cube_side_length : ℕ := 5
def total_cubes : ℕ := cube_side_length ^ 3

-- Define the number of cubes with different numbers of painted faces
def cubes_with_three_painted_faces : ℕ := 1
def cubes_with_one_painted_face : ℕ := 36

-- Define the probability calculation function
def probability_one_three_one_face : ℚ :=
  (cubes_with_three_painted_faces * cubes_with_one_painted_face : ℚ) /
  (total_cubes * (total_cubes - 1) / 2)

-- The theorem to prove
theorem probability_equals_three_over_646 :
  probability_one_three_one_face = 3 / 646 := by
  sorry

end NUMINAMATH_CALUDE_probability_equals_three_over_646_l1607_160713


namespace NUMINAMATH_CALUDE_remaining_sample_is_nineteen_l1607_160734

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_samples : List ℕ

/-- Calculates the sampling interval -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.total_students / s.sample_size

/-- Theorem stating that the remaining sample number is 19 -/
theorem remaining_sample_is_nineteen (s : SystematicSampling)
  (h1 : s.total_students = 56)
  (h2 : s.sample_size = 4)
  (h3 : s.known_samples = [5, 33, 47])
  : ∃ (remaining : ℕ), remaining = 19 ∧ remaining ∉ s.known_samples :=
by sorry

end NUMINAMATH_CALUDE_remaining_sample_is_nineteen_l1607_160734


namespace NUMINAMATH_CALUDE_ball_distribution_ways_l1607_160790

/-- Represents the number of ways to distribute balls of a given color --/
def distribute_balls (remaining : ℕ) : ℕ := remaining + 1

/-- Represents the total number of ways to distribute balls between two boys --/
def total_distributions (white : ℕ) (black : ℕ) (red : ℕ) : ℕ :=
  distribute_balls (white - 4) * distribute_balls (red - 4)

theorem ball_distribution_ways :
  let white := 6
  let black := 4
  let red := 8
  total_distributions white black red = 15 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_ways_l1607_160790


namespace NUMINAMATH_CALUDE_sum_mod_twelve_l1607_160765

theorem sum_mod_twelve : (2150 + 2151 + 2152 + 2153 + 2154 + 2155) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_twelve_l1607_160765


namespace NUMINAMATH_CALUDE_perpendicular_segments_s_value_l1607_160766

/-- Given two perpendicular line segments PQ and PR, where P(4, 2), R(0, 1), and Q(2, s),
    prove that s = 10 -/
theorem perpendicular_segments_s_value (P Q R : ℝ × ℝ) (s : ℝ) : 
  P = (4, 2) →
  R = (0, 1) →
  Q = (2, s) →
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0 →
  s = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_segments_s_value_l1607_160766


namespace NUMINAMATH_CALUDE_root_sum_product_l1607_160735

theorem root_sum_product (a b : ℝ) : 
  (a^4 - 6*a - 2 = 0) → 
  (b^4 - 6*b - 2 = 0) → 
  (a ≠ b) →
  (a*b + a + b = 2 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_product_l1607_160735


namespace NUMINAMATH_CALUDE_power_of_half_l1607_160737

theorem power_of_half (some_power k : ℕ) : 
  (1/2 : ℝ)^some_power * (1/81 : ℝ)^k = 1/(18^16 : ℝ) → 
  k = 8 → 
  some_power = 16 := by
sorry

end NUMINAMATH_CALUDE_power_of_half_l1607_160737


namespace NUMINAMATH_CALUDE_weed_difference_l1607_160710

/-- The number of weeds Sarah pulled on each day --/
structure WeedCount where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Sarah's weed-pulling over four days --/
def SarahsWeedPulling (w : WeedCount) : Prop :=
  w.tuesday = 25 ∧
  w.wednesday = 3 * w.tuesday ∧
  w.thursday = w.wednesday / 5 ∧
  w.friday < w.thursday ∧
  w.tuesday + w.wednesday + w.thursday + w.friday = 120

/-- Theorem stating the difference in weeds pulled between Thursday and Friday --/
theorem weed_difference (w : WeedCount) (h : SarahsWeedPulling w) :
  w.thursday - w.friday = 10 := by
  sorry

end NUMINAMATH_CALUDE_weed_difference_l1607_160710


namespace NUMINAMATH_CALUDE_altitude_not_integer_l1607_160751

/-- Represents a right triangle with integer sides -/
structure RightTriangle where
  a : ℕ  -- First leg
  b : ℕ  -- Second leg
  c : ℕ  -- Hypotenuse
  is_right : a^2 + b^2 = c^2  -- Pythagorean theorem

/-- The altitude to the hypotenuse in a right triangle -/
def altitude (t : RightTriangle) : ℚ :=
  (t.a * t.b : ℚ) / t.c

/-- Theorem: In a right triangle with pairwise coprime integer sides, 
    the altitude to the hypotenuse is not an integer -/
theorem altitude_not_integer (t : RightTriangle) 
  (h_coprime : Nat.gcd t.a t.b = 1 ∧ Nat.gcd t.b t.c = 1 ∧ Nat.gcd t.c t.a = 1) : 
  ¬ ∃ (n : ℕ), altitude t = n :=
sorry

end NUMINAMATH_CALUDE_altitude_not_integer_l1607_160751


namespace NUMINAMATH_CALUDE_intersection_probability_odd_polygon_l1607_160740

/-- The probability that two randomly chosen diagonals intersect inside a convex polygon with 2n+1 vertices -/
theorem intersection_probability_odd_polygon (n : ℕ) :
  let vertices := 2 * n + 1
  let diagonals := n * (2 * n + 1) - (2 * n + 1)
  let ways_to_choose_diagonals := (diagonals.choose 2 : ℚ)
  let ways_to_choose_vertices := ((2 * n + 1).choose 4 : ℚ)
  ways_to_choose_vertices / ways_to_choose_diagonals = n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_probability_odd_polygon_l1607_160740


namespace NUMINAMATH_CALUDE_philips_farm_animals_l1607_160792

/-- The number of animals on Philip's farm --/
def total_animals (cows ducks pigs : ℕ) : ℕ := cows + ducks + pigs

/-- Theorem stating the total number of animals on Philip's farm --/
theorem philips_farm_animals :
  ∀ (cows ducks pigs : ℕ),
  cows = 20 →
  ducks = cows + cows / 2 →
  pigs = (cows + ducks) / 5 →
  total_animals cows ducks pigs = 60 := by
sorry

end NUMINAMATH_CALUDE_philips_farm_animals_l1607_160792


namespace NUMINAMATH_CALUDE_sphere_and_cylinder_properties_l1607_160714

/-- Given a sphere with volume 72π cubic inches, prove its surface area and the radius of a cylinder with the same volume and height 4 inches. -/
theorem sphere_and_cylinder_properties :
  ∃ (r : ℝ), 
    (4 / 3 * π * r^3 = 72 * π) ∧ 
    (4 * π * r^2 = 36 * 2^(2/3) * π) ∧
    ∃ (r_cyl : ℝ), 
      (π * r_cyl^2 * 4 = 72 * π) ∧ 
      (r_cyl = 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sphere_and_cylinder_properties_l1607_160714


namespace NUMINAMATH_CALUDE_tan_theta_equals_sqrt_three_over_five_l1607_160718

theorem tan_theta_equals_sqrt_three_over_five (θ : Real) : 
  2 * Real.sin (θ + π/3) = 3 * Real.sin (π/3 - θ) → 
  Real.tan θ = Real.sqrt 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_equals_sqrt_three_over_five_l1607_160718


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l1607_160799

theorem quadratic_perfect_square (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (x + b)^2) ↔ k = 100 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l1607_160799


namespace NUMINAMATH_CALUDE_salary_adjustment_proof_l1607_160796

def initial_salary : ℝ := 2500
def june_raise_percentage : ℝ := 0.15
def june_bonus : ℝ := 300
def july_cut_percentage : ℝ := 0.25

def final_salary : ℝ :=
  (initial_salary * (1 + june_raise_percentage) + june_bonus) * (1 - july_cut_percentage)

theorem salary_adjustment_proof :
  final_salary = 2381.25 := by sorry

end NUMINAMATH_CALUDE_salary_adjustment_proof_l1607_160796


namespace NUMINAMATH_CALUDE_max_planes_from_parallel_lines_max_planes_is_six_l1607_160770

/-- Given four parallel lines, the maximum number of unique planes formed by selecting two lines -/
theorem max_planes_from_parallel_lines : ℕ :=
  -- Define the number of lines
  let num_lines : ℕ := 4

  -- Define the function to calculate combinations
  let combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

  -- Calculate the number of ways to select 2 lines out of 4
  combinations num_lines 2

/-- Proof that the maximum number of planes is 6 -/
theorem max_planes_is_six : max_planes_from_parallel_lines = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_planes_from_parallel_lines_max_planes_is_six_l1607_160770


namespace NUMINAMATH_CALUDE_set_B_proof_l1607_160739

open Set

theorem set_B_proof (U : Set ℕ) (A B : Set ℕ) :
  U = {1, 2, 3, 4, 5, 6, 7, 8, 9} →
  (U \ (A ∪ B)) = {1, 3} →
  ((U \ A) ∩ B) = {2, 4} →
  B = {5, 6, 7, 8, 9} :=
by sorry

end NUMINAMATH_CALUDE_set_B_proof_l1607_160739


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1607_160793

theorem absolute_value_inequality (x : ℝ) : |x - 3| ≥ |x| ↔ x ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1607_160793


namespace NUMINAMATH_CALUDE_debby_vacation_pictures_l1607_160749

/-- Calculates the number of remaining pictures after deletion -/
def remaining_pictures (zoo_pictures museum_pictures deleted_pictures : ℕ) : ℕ :=
  (zoo_pictures + museum_pictures) - deleted_pictures

/-- Theorem: The number of remaining pictures is correct for Debby's vacation -/
theorem debby_vacation_pictures : remaining_pictures 24 12 14 = 22 := by
  sorry

end NUMINAMATH_CALUDE_debby_vacation_pictures_l1607_160749


namespace NUMINAMATH_CALUDE_triangle_properties_l1607_160795

/-- Triangle ABC with given properties -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  h1 : a = 3
  h2 : b = 4
  h3 : B = π/2 + A

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) : Real.tan t.B = -4/3 ∧ t.c = 7/5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1607_160795


namespace NUMINAMATH_CALUDE_true_propositions_l1607_160704

theorem true_propositions :
  (∃ x : ℝ, x^3 < 1) ∧
  (∃ x : ℝ, x^2 + 1 > 0) ∧
  ¬(∃ x : ℚ, x^2 = 2) ∧
  ¬(∃ x : ℕ, x^3 > x^2) :=
by sorry

end NUMINAMATH_CALUDE_true_propositions_l1607_160704


namespace NUMINAMATH_CALUDE_quadratic_rational_root_contradiction_l1607_160730

theorem quadratic_rational_root_contradiction (a b c : ℤ) (h_a_nonzero : a ≠ 0) 
  (h_rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ a * (p / q)^2 + b * (p / q) + c = 0) 
  (h_all_odd : Odd a ∧ Odd b ∧ Odd c) : False :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_contradiction_l1607_160730


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l1607_160784

theorem coconut_grove_problem (x : ℕ) : 
  (∃ (t₄₀ t₁₂₀ t₁₈₀ : ℕ),
    t₄₀ = x + 2 ∧
    t₁₂₀ = x ∧
    t₁₈₀ = x - 2 ∧
    (40 * t₄₀ + 120 * t₁₂₀ + 180 * t₁₈₀) / (t₄₀ + t₁₂₀ + t₁₈₀) = 100) →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l1607_160784


namespace NUMINAMATH_CALUDE_consecutive_sums_not_prime_l1607_160719

theorem consecutive_sums_not_prime (n : ℕ) :
  (∃ k : ℕ, k > 1 ∧ k < 5*n + 10 ∧ (5*n + 10) % k = 0) ∧
  (∃ k : ℕ, k > 1 ∧ k < 5*n^2 + 10 ∧ (5*n^2 + 10) % k = 0) := by
  sorry

#check consecutive_sums_not_prime

end NUMINAMATH_CALUDE_consecutive_sums_not_prime_l1607_160719


namespace NUMINAMATH_CALUDE_sector_radius_l1607_160715

/-- Given a circular sector with area 7 square centimeters and arc length 3.5 cm,
    prove that the radius of the circle is 4 cm. -/
theorem sector_radius (area : ℝ) (arc_length : ℝ) (radius : ℝ) 
    (h_area : area = 7) 
    (h_arc_length : arc_length = 3.5) 
    (h_sector_area : area = (arc_length * radius) / 2) : radius = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l1607_160715


namespace NUMINAMATH_CALUDE_prime_sum_of_squares_l1607_160702

theorem prime_sum_of_squares (k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_form : p = 4 * k + 1) :
  (∃ (x y m : ℕ), x^2 + y^2 = m * p) ∧
  (∀ (x y m : ℕ), x^2 + y^2 = m * p → m > 1 → 
    ∃ (X Y m' : ℕ), X^2 + Y^2 = m' * p ∧ 0 < m' ∧ m' < m) :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_of_squares_l1607_160702


namespace NUMINAMATH_CALUDE_abcd_congruence_l1607_160772

theorem abcd_congruence (a b c d : ℕ) 
  (h1 : a < 7) (h2 : b < 7) (h3 : c < 7) (h4 : d < 7)
  (c1 : (a + 2*b + 3*c + 4*d) % 7 = 1)
  (c2 : (2*a + 3*b + c + 2*d) % 7 = 5)
  (c3 : (3*a + b + 2*c + 3*d) % 7 = 3)
  (c4 : (4*a + 2*b + d + c) % 7 = 2) :
  (a * b * c * d) % 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_abcd_congruence_l1607_160772


namespace NUMINAMATH_CALUDE_absolute_value_equals_negative_l1607_160758

theorem absolute_value_equals_negative (a : ℝ) : 
  (abs a = -a) → a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equals_negative_l1607_160758


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l1607_160767

/-- The number of saltwater aquariums Tyler has -/
def saltwater_aquariums : ℕ := 22

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 46

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := saltwater_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_saltwater_animals = 1012 := by
  sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l1607_160767


namespace NUMINAMATH_CALUDE_triangle_properties_l1607_160777

theorem triangle_properties (a b c A B C S : ℝ) : 
  a = 2 → C = π / 3 → 
  (A = π / 4 → c = Real.sqrt 6) ∧ 
  (S = Real.sqrt 3 → b = 2 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1607_160777


namespace NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l1607_160773

/-- A geometric sequence of positive real numbers -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem ninth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_fifth : a 5 = 32)
  (h_eleventh : a 11 = 2) :
  a 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l1607_160773


namespace NUMINAMATH_CALUDE_soccer_team_beverage_consumption_l1607_160754

theorem soccer_team_beverage_consumption 
  (team_size : ℕ) 
  (total_beverage : ℕ) 
  (h1 : team_size = 36) 
  (h2 : total_beverage = 252) :
  total_beverage / team_size = 7 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_beverage_consumption_l1607_160754


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1607_160705

theorem rectangle_max_area (perimeter : ℝ) (h_perimeter : perimeter = 40) :
  let short_side := perimeter / 6
  let long_side := 2 * short_side
  let area := short_side * long_side
  area = 800 / 9 := by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1607_160705


namespace NUMINAMATH_CALUDE_bryan_total_books_l1607_160783

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 9

/-- The number of books in each bookshelf -/
def books_per_shelf : ℕ := 56

/-- The total number of books Bryan has -/
def total_books : ℕ := num_bookshelves * books_per_shelf

/-- Theorem stating that Bryan has 504 books in total -/
theorem bryan_total_books : total_books = 504 := by sorry

end NUMINAMATH_CALUDE_bryan_total_books_l1607_160783


namespace NUMINAMATH_CALUDE_vehicle_value_last_year_l1607_160752

/-- If a vehicle's value this year is 16000 dollars and is 0.8 times its value last year,
    then its value last year was 20000 dollars. -/
theorem vehicle_value_last_year 
  (value_this_year : ℝ) 
  (value_ratio : ℝ) 
  (h1 : value_this_year = 16000)
  (h2 : value_ratio = 0.8)
  (h3 : value_this_year = value_ratio * value_last_year) : 
  value_last_year = 20000 := by
  sorry


end NUMINAMATH_CALUDE_vehicle_value_last_year_l1607_160752


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_isosceles_triangle_l1607_160775

/-- The radius of the inscribed circle in an isosceles triangle -/
theorem inscribed_circle_radius_isosceles_triangle (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 8) (h3 : EF = 10) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  r = 5 * Real.sqrt 39 / 13 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_isosceles_triangle_l1607_160775


namespace NUMINAMATH_CALUDE_fence_length_l1607_160728

/-- The total length of a fence for a land shaped like a rectangle combined with a semicircle,
    given the dimensions and an opening. -/
theorem fence_length
  (rect_length : ℝ)
  (rect_width : ℝ)
  (semicircle_radius : ℝ)
  (opening_length : ℝ)
  (h1 : rect_length = 20)
  (h2 : rect_width = 14)
  (h3 : semicircle_radius = 7)
  (h4 : opening_length = 3)
  : rect_length * 2 + rect_width + π * semicircle_radius + rect_width - opening_length = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_fence_length_l1607_160728


namespace NUMINAMATH_CALUDE_f_properties_l1607_160756

open Real

/-- The function f defined by the given conditions -/
noncomputable def f (x : ℝ) : ℝ := x / (1 + 2 * x^2)

/-- The theorem stating the properties of f -/
theorem f_properties :
  ∀ α β x y : ℝ,
  (sin (2 * α + β) = 3 * sin β) →
  (tan α = x) →
  (tan β = y) →
  (y = f x) →
  (0 < α) →
  (α < π / 3) →
  (∀ z : ℝ, 0 < z → z < f x → z < sqrt 2 / 4) ∧
  (f x ≤ sqrt 2 / 4) ∧
  (∃ z : ℝ, 0 < z ∧ z < sqrt 2 / 4 ∧ z = f x) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l1607_160756


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1607_160744

theorem absolute_value_inequality (x : ℝ) : 
  |x - x^2 - 2| > x^2 - 3*x - 4 ↔ x > -3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1607_160744


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l1607_160794

theorem common_root_quadratic_equations (p : ℝ) :
  (p > 0 ∧
   ∃ x : ℝ, (3 * x^2 - 4 * p * x + 9 = 0) ∧ (x^2 - 2 * p * x + 5 = 0)) ↔
  p = 3 :=
by sorry

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l1607_160794


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1607_160747

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) / z = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1607_160747


namespace NUMINAMATH_CALUDE_field_trip_students_l1607_160700

theorem field_trip_students (van_capacity : ℕ) (num_adults : ℕ) (num_vans : ℕ) : 
  van_capacity = 5 → num_adults = 5 → num_vans = 6 → 
  (num_vans * van_capacity - num_adults : ℕ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l1607_160700


namespace NUMINAMATH_CALUDE_inequality_proof_l1607_160771

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (a + c) + c / (a + b)) ≥ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1607_160771


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1607_160725

theorem diophantine_equation_solutions : 
  ∀ x y : ℤ, 5 * x^2 + 5 * x * y + 5 * y^2 = 7 * x + 14 * y ↔ 
  (x = -1 ∧ y = 3) ∨ (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1607_160725


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1607_160743

/-- Given a cube with surface area 864 square units, its volume is 1728 cubic units. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  (6 * s^2 = 864) →
  s^3 = 1728 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1607_160743


namespace NUMINAMATH_CALUDE_parking_lot_capacity_l1607_160791

/-- Represents a multi-story parking lot -/
structure ParkingLot where
  levels : Nat
  capacity_per_level : Nat

/-- Calculates the total capacity of a parking lot -/
def total_capacity (p : ParkingLot) : Nat :=
  p.levels * p.capacity_per_level

/-- Theorem stating the total capacity of the specific parking lot -/
theorem parking_lot_capacity :
  ∃ (p : ParkingLot), p.levels = 5 ∧ p.capacity_per_level = 23 + 62 ∧ total_capacity p = 425 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_lot_capacity_l1607_160791


namespace NUMINAMATH_CALUDE_father_daughter_ages_l1607_160717

/-- Represents the ages of a father and daughter at present and in the future. -/
structure FamilyAges where
  daughter_now : ℕ
  father_now : ℕ
  daughter_future : ℕ
  father_future : ℕ

/-- The conditions given in the problem. -/
def age_conditions (ages : FamilyAges) : Prop :=
  ages.father_now = 5 * ages.daughter_now ∧
  ages.daughter_future = ages.daughter_now + 30 ∧
  ages.father_future = ages.father_now + 30 ∧
  ages.father_future = 3 * ages.daughter_future

/-- The theorem stating the solution to the problem. -/
theorem father_daughter_ages :
  ∃ (ages : FamilyAges), age_conditions ages ∧ ages.daughter_now = 30 ∧ ages.father_now = 150 := by
  sorry

end NUMINAMATH_CALUDE_father_daughter_ages_l1607_160717


namespace NUMINAMATH_CALUDE_power_multiplication_simplification_l1607_160726

theorem power_multiplication_simplification :
  let a : ℝ := 0.25
  let b : ℝ := -4
  let n : ℕ := 16
  let m : ℕ := 17
  (a ^ n) * (b ^ m) = -4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_simplification_l1607_160726


namespace NUMINAMATH_CALUDE_max_value_reciprocal_sum_l1607_160760

theorem max_value_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hax : a^x = 3) (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  ∃ (max : ℝ), max = 1 ∧ ∀ (x' y' : ℝ), 
    (∃ (a' b' : ℝ), a' > 1 ∧ b' > 1 ∧ a'^x' = 3 ∧ b'^y' = 3 ∧ a' + b' = 2 * Real.sqrt 3) →
    1/x' + 1/y' ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_reciprocal_sum_l1607_160760


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1607_160788

theorem tan_alpha_plus_pi_fourth (α : Real) (h : Real.tan (α / 2) = 2) :
  Real.tan (α + π / 4) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1607_160788


namespace NUMINAMATH_CALUDE_element_in_complement_l1607_160703

def U : Set Nat := {1,2,3,4,5,6}
def M : Set Nat := {1,5}
def P : Set Nat := {2,4}

theorem element_in_complement : 3 ∈ (U \ (M ∪ P)) := by
  sorry

end NUMINAMATH_CALUDE_element_in_complement_l1607_160703


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_once_l1607_160782

/-- A parabola in the xy-plane defined by y = x^2 + 2x + k -/
def parabola (k : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + k

/-- Condition for a quadratic equation to have exactly one real root -/
def has_one_root (a b c : ℝ) : Prop := b^2 - 4*a*c = 0

/-- Theorem: The parabola y = x^2 + 2x + k intersects the x-axis at only one point if and only if k = 1 -/
theorem parabola_intersects_x_axis_once (k : ℝ) :
  (∃ x : ℝ, parabola k x = 0 ∧ ∀ y : ℝ, parabola k y = 0 → y = x) ↔ k = 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_once_l1607_160782


namespace NUMINAMATH_CALUDE_isosceles_triangle_circumradius_l1607_160781

/-- Given an isosceles triangle with base angle α, if the altitude to the base exceeds
    the radius of the inscribed circle by m, then the radius of the circumscribed circle
    is m / (4 * sin²(α/2)). -/
theorem isosceles_triangle_circumradius (α m : ℝ) (h_α : 0 < α ∧ α < π) (h_m : m > 0) :
  let altitude := inradius + m
  circumradius = m / (4 * Real.sin (α / 2) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_circumradius_l1607_160781


namespace NUMINAMATH_CALUDE_broken_line_rectangle_ratio_l1607_160797

/-- A rectangle with a broken line inside it -/
structure BrokenLineRectangle where
  /-- The shorter side of the rectangle -/
  short_side : ℝ
  /-- The longer side of the rectangle -/
  long_side : ℝ
  /-- The broken line consists of segments equal to the shorter side -/
  segment_length : ℝ
  /-- The short side is positive -/
  short_positive : 0 < short_side
  /-- The long side is longer than the short side -/
  long_longer : short_side < long_side
  /-- The segment length is equal to the shorter side -/
  segment_eq_short : segment_length = short_side
  /-- Adjacent segments of the broken line are perpendicular -/
  segments_perpendicular : True

/-- The ratio of the shorter side to the longer side is 1:2 -/
theorem broken_line_rectangle_ratio (r : BrokenLineRectangle) :
  r.short_side / r.long_side = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_broken_line_rectangle_ratio_l1607_160797


namespace NUMINAMATH_CALUDE_star_3_2_l1607_160722

-- Define the ★ operation
def star (a b : ℝ) : ℝ := a^3 + 3*a^2*b + 3*a*b^2 + b^3

-- Theorem statement
theorem star_3_2 : star 3 2 = 125 := by sorry

end NUMINAMATH_CALUDE_star_3_2_l1607_160722


namespace NUMINAMATH_CALUDE_polynomial_expansion_alternating_sum_l1607_160712

theorem polynomial_expansion_alternating_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ - a₁ + a₂ - a₃ + a₄ = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_alternating_sum_l1607_160712


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1607_160785

theorem smallest_n_congruence (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < n → ¬(629 * m ≡ 1181 * m [ZMOD 35])) ∧ 
  (629 * n ≡ 1181 * n [ZMOD 35]) → 
  n = 35 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1607_160785


namespace NUMINAMATH_CALUDE_max_store_visits_l1607_160721

theorem max_store_visits (total_stores : ℕ) (total_visits : ℕ) (unique_visitors : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 7) (h2 : total_visits = 21) 
  (h3 : unique_visitors = 11) (h4 : double_visitors = 7) 
  (h5 : double_visitors * 2 ≤ total_visits) 
  (h6 : ∀ v, v ≤ unique_visitors → v ≥ 1) : 
  ∃ max_visits : ℕ, max_visits ≤ total_stores ∧ 
  (∀ v, v ≤ unique_visitors → v ≤ max_visits) ∧ max_visits = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_store_visits_l1607_160721


namespace NUMINAMATH_CALUDE_smallest_c_value_l1607_160731

theorem smallest_c_value (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : b - a = c - b)  -- arithmetic progression
  (h4 : a * a = c * b)  -- geometric progression
  : c ≥ 4 ∧ ∃ (a' b' : ℤ), a' < b' ∧ b' < 4 ∧ b' - a' = 4 - b' ∧ a' * a' = 4 * b' := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l1607_160731


namespace NUMINAMATH_CALUDE_unique_solution_trig_equation_l1607_160707

theorem unique_solution_trig_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
    Real.tan ((150 - x) * π / 180) = 
      (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
      (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
    x = 115 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trig_equation_l1607_160707


namespace NUMINAMATH_CALUDE_expression_evaluation_l1607_160789

theorem expression_evaluation :
  ∃ k : ℝ, k > 0 ∧ (3^512 + 7^513)^2 - (3^512 - 7^513)^2 = k * 10^513 ∧ k = 28 * 2.1^512 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1607_160789


namespace NUMINAMATH_CALUDE_initial_paint_amount_l1607_160776

theorem initial_paint_amount (total_needed : ℕ) (bought : ℕ) (still_needed : ℕ) 
  (h1 : total_needed = 70)
  (h2 : bought = 23)
  (h3 : still_needed = 11) :
  total_needed - still_needed - bought = 36 :=
by sorry

end NUMINAMATH_CALUDE_initial_paint_amount_l1607_160776


namespace NUMINAMATH_CALUDE_inequality_proof_l1607_160761

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1607_160761


namespace NUMINAMATH_CALUDE_h2o_formation_in_neutralization_l1607_160762

/-- Represents a chemical substance -/
structure Substance where
  name : String
  moles : ℝ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Substance
  products : List Substance

/-- Given a balanced chemical equation and the amounts of reactants, 
    calculate the amount of a specific product formed -/
def calculateProductAmount (reaction : Reaction) (product : Substance) : ℝ :=
  sorry

theorem h2o_formation_in_neutralization :
  let hch3co2 := Substance.mk "HCH3CO2" 1
  let naoh := Substance.mk "NaOH" 1
  let h2o := Substance.mk "H2O" 1
  let nach3co2 := Substance.mk "NaCH3CO2" 1
  let reaction := Reaction.mk [hch3co2, naoh] [nach3co2, h2o]
  calculateProductAmount reaction h2o = 1 := by
  sorry

end NUMINAMATH_CALUDE_h2o_formation_in_neutralization_l1607_160762


namespace NUMINAMATH_CALUDE_first_number_proof_l1607_160742

theorem first_number_proof (x y : ℕ) (h1 : x + y = 20) (h2 : y = 15) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l1607_160742


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1607_160768

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

-- The sum-to-product identity for sine
axiom sin_sum_to_product_identity (a b : ℝ) : 
  Real.sin a + Real.sin b = 2 * Real.sin ((a + b) / 2) * Real.cos ((a - b) / 2)

end NUMINAMATH_CALUDE_sin_sum_to_product_l1607_160768


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1607_160745

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 20 + 25 + 7 + 15 + y) / 6 = 15 → y = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1607_160745


namespace NUMINAMATH_CALUDE_mango_purchase_l1607_160764

theorem mango_purchase (grapes_kg : ℕ) (grapes_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) :
  grapes_kg = 10 ∧ 
  grapes_rate = 70 ∧ 
  mango_rate = 55 ∧ 
  total_paid = 1195 →
  ∃ (mango_kg : ℕ), mango_kg = 9 ∧ grapes_kg * grapes_rate + mango_kg * mango_rate = total_paid :=
by sorry

end NUMINAMATH_CALUDE_mango_purchase_l1607_160764


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_terms_l1607_160733

theorem arithmetic_sequence_max_terms 
  (a : ℝ) (n : ℕ) 
  (h1 : a^2 + (n - 1) * (a + 2 * (n - 1)) ≤ 100) : n ≤ 8 := by
  sorry

#check arithmetic_sequence_max_terms

end NUMINAMATH_CALUDE_arithmetic_sequence_max_terms_l1607_160733


namespace NUMINAMATH_CALUDE_five_eighteenths_decimal_l1607_160711

theorem five_eighteenths_decimal : 
  (5 : ℚ) / 18 = 0.2777777777777777 :=
by sorry

end NUMINAMATH_CALUDE_five_eighteenths_decimal_l1607_160711


namespace NUMINAMATH_CALUDE_G_is_odd_and_f_neg_b_value_l1607_160736

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x / (Real.exp x + 1)

noncomputable def G (x : ℝ) : ℝ := f x - 1

theorem G_is_odd_and_f_neg_b_value (b : ℝ) (h : f b = 3/2) :
  (∀ x, G (-x) = -G x) ∧ f (-b) = 1/2 := by sorry

end NUMINAMATH_CALUDE_G_is_odd_and_f_neg_b_value_l1607_160736


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1607_160738

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- Our specific arithmetic sequence satisfying a₃ + a₈ = 6 -/
def our_sequence (a : ℕ → ℝ) : Prop :=
  arithmetic_sequence a ∧ a 3 + a 8 = 6

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : our_sequence a) :
  3 * a 2 + a 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1607_160738
