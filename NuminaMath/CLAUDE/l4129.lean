import Mathlib

namespace NUMINAMATH_CALUDE_infinite_equal_pairs_l4129_412922

-- Define the sequence type
def InfiniteSequence := ℤ → ℝ

-- Define the property that each term is 1/4 of the sum of its neighbors
def NeighborSumProperty (a : InfiniteSequence) :=
  ∀ n : ℤ, a n = (1 / 4) * (a (n - 1) + a (n + 1))

-- Define the existence of two equal terms
def HasEqualTerms (a : InfiniteSequence) :=
  ∃ i j : ℤ, i ≠ j ∧ a i = a j

-- Define the existence of infinitely many pairs of equal terms
def HasInfiniteEqualPairs (a : InfiniteSequence) :=
  ∀ N : ℕ, ∃ i j : ℤ, i ≠ j ∧ |i - j| > N ∧ a i = a j

-- The main theorem
theorem infinite_equal_pairs
  (a : InfiniteSequence)
  (h1 : NeighborSumProperty a)
  (h2 : HasEqualTerms a) :
  HasInfiniteEqualPairs a :=
sorry

end NUMINAMATH_CALUDE_infinite_equal_pairs_l4129_412922


namespace NUMINAMATH_CALUDE_total_milk_volume_l4129_412979

-- Define the conversion factor from milliliters to liters
def ml_to_l : ℚ := 1 / 1000

-- Define the volumes of milk in liters
def volume1 : ℚ := 2
def volume2 : ℚ := 750 * ml_to_l
def volume3 : ℚ := 250 * ml_to_l

-- State the theorem
theorem total_milk_volume :
  volume1 + volume2 + volume3 = 3 := by sorry

end NUMINAMATH_CALUDE_total_milk_volume_l4129_412979


namespace NUMINAMATH_CALUDE_lunch_total_amount_l4129_412976

/-- The total amount spent on lunch given the conditions -/
theorem lunch_total_amount (your_spending friend_spending : ℕ) 
  (h1 : friend_spending = 10)
  (h2 : friend_spending = your_spending + 3) : 
  your_spending + friend_spending = 17 := by
  sorry

end NUMINAMATH_CALUDE_lunch_total_amount_l4129_412976


namespace NUMINAMATH_CALUDE_email_sending_ways_l4129_412997

theorem email_sending_ways (email_addresses : ℕ) (emails_to_send : ℕ) : 
  email_addresses = 3 → emails_to_send = 5 → email_addresses ^ emails_to_send = 243 := by
  sorry

end NUMINAMATH_CALUDE_email_sending_ways_l4129_412997


namespace NUMINAMATH_CALUDE_driving_time_to_school_l4129_412992

theorem driving_time_to_school 
  (total_hours : ℕ) 
  (school_days : ℕ) 
  (drives_both_ways : Bool) : 
  total_hours = 50 → 
  school_days = 75 → 
  drives_both_ways = true → 
  (total_hours * 60) / (school_days * 2) = 20 := by
sorry

end NUMINAMATH_CALUDE_driving_time_to_school_l4129_412992


namespace NUMINAMATH_CALUDE_problem_solution_l4129_412952

theorem problem_solution (x y : ℤ) : 
  x > y ∧ y > 0 ∧ x + y + x * y = 152 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4129_412952


namespace NUMINAMATH_CALUDE_intersection_sum_l4129_412937

/-- Given two lines y = mx + 5 and y = 2x + b intersecting at (7, 10),
    prove that the sum of constants b and m is equal to -23/7 -/
theorem intersection_sum (m b : ℚ) : 
  (7 * m + 5 = 10) → (2 * 7 + b = 10) → b + m = -23/7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l4129_412937


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l4129_412935

/-- The eccentricity of the ellipse x^2 + 4y^2 = 4 is √3/2 -/
theorem ellipse_eccentricity : 
  let equation := fun (x y : ℝ) => x^2 + 4*y^2 = 4
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  equation 0 1 ∧ e = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l4129_412935


namespace NUMINAMATH_CALUDE_exists_team_rating_l4129_412923

variable {Team : Type}
variable (d : Team → Team → ℝ)

axiom goal_difference_symmetry :
  ∀ (A B : Team), d A B + d B A = 0

axiom goal_difference_transitivity :
  ∀ (A B C : Team), d A B + d B C + d C A = 0

theorem exists_team_rating :
  ∃ (f : Team → ℝ), ∀ (A B : Team), d A B = f A - f B :=
sorry

end NUMINAMATH_CALUDE_exists_team_rating_l4129_412923


namespace NUMINAMATH_CALUDE_cyclic_trapezoid_area_l4129_412930

/-- Represents a cyclic trapezoid with parallel sides a and b, where a < b -/
structure CyclicTrapezoid where
  a : ℝ
  b : ℝ
  h : a < b

/-- The area of a cyclic trapezoid given the conditions -/
def area (t : CyclicTrapezoid) : Set ℝ :=
  let t₁ := (t.a + t.b) * (t.a - Real.sqrt (2 * t.a^2 - t.b^2)) / 4
  let t₂ := (t.a + t.b) * (t.a + Real.sqrt (2 * t.a^2 - t.b^2)) / 4
  {t₁, t₂}

/-- Theorem stating that the area of the cyclic trapezoid is either t₁ or t₂ -/
theorem cyclic_trapezoid_area (t : CyclicTrapezoid) :
  ∃ A ∈ area t, A = (t.a + t.b) * (t.a - Real.sqrt (2 * t.a^2 - t.b^2)) / 4 ∨
                 A = (t.a + t.b) * (t.a + Real.sqrt (2 * t.a^2 - t.b^2)) / 4 := by
  sorry


end NUMINAMATH_CALUDE_cyclic_trapezoid_area_l4129_412930


namespace NUMINAMATH_CALUDE_caramel_apple_cost_is_25_l4129_412902

/-- The cost of an ice cream cone in cents -/
def ice_cream_cost : ℕ := 15

/-- The additional cost of a caramel apple compared to an ice cream cone in cents -/
def apple_additional_cost : ℕ := 10

/-- The cost of a caramel apple in cents -/
def caramel_apple_cost : ℕ := ice_cream_cost + apple_additional_cost

/-- Theorem: The cost of a caramel apple is 25 cents -/
theorem caramel_apple_cost_is_25 : caramel_apple_cost = 25 := by
  sorry

end NUMINAMATH_CALUDE_caramel_apple_cost_is_25_l4129_412902


namespace NUMINAMATH_CALUDE_distance_before_meeting_l4129_412986

/-- The distance between two boats one minute before they meet -/
theorem distance_before_meeting (v1 v2 d : ℝ) (hv1 : v1 = 4) (hv2 : v2 = 20) (hd : d = 20) :
  let t := d / (v1 + v2)  -- Time to meet
  let distance_per_minute := (v1 + v2) / 60
  (t - 1/60) * (v1 + v2) = 0.4
  := by sorry

end NUMINAMATH_CALUDE_distance_before_meeting_l4129_412986


namespace NUMINAMATH_CALUDE_odd_function_symmetric_behavior_l4129_412938

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def HasMinimumOn (f : ℝ → ℝ) (a b m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

def HasMaximumOn (f : ℝ → ℝ) (a b M : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ M) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = M)

theorem odd_function_symmetric_behavior (f : ℝ → ℝ) :
  IsOdd f →
  IsIncreasingOn f 3 7 →
  HasMinimumOn f 3 7 5 →
  IsIncreasingOn f (-7) (-3) ∧ HasMaximumOn f (-7) (-3) (-5) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_symmetric_behavior_l4129_412938


namespace NUMINAMATH_CALUDE_track_length_is_360_l4129_412982

/-- Represents a circular running track -/
structure Track where
  length : ℝ
  start_points_opposite : Bool
  runners_opposite_directions : Bool

/-- Represents a runner on the track -/
structure Runner where
  speed : ℝ
  distance_to_first_meeting : ℝ
  distance_between_meetings : ℝ

/-- The main theorem statement -/
theorem track_length_is_360 (track : Track) (brenda sally : Runner) : 
  track.start_points_opposite ∧ 
  track.runners_opposite_directions ∧
  brenda.distance_to_first_meeting = 120 ∧
  sally.speed = 2 * brenda.speed ∧
  sally.distance_between_meetings = 180 →
  track.length = 360 := by sorry

end NUMINAMATH_CALUDE_track_length_is_360_l4129_412982


namespace NUMINAMATH_CALUDE_correct_equation_l4129_412929

/-- Represents the problem of sending a letter over a certain distance with two horses of different speeds. -/
def letter_problem (distance : ℝ) (slow_delay : ℝ) (fast_early : ℝ) (speed_ratio : ℝ) :=
  ∀ x : ℝ, x > 3 → (distance / (x + slow_delay)) * speed_ratio = distance / (x - fast_early)

/-- The theorem states that the given equation correctly represents the problem for the specific values mentioned. -/
theorem correct_equation : letter_problem 900 1 3 2 := by sorry

end NUMINAMATH_CALUDE_correct_equation_l4129_412929


namespace NUMINAMATH_CALUDE_regular_octagon_area_1_5_sqrt_2_l4129_412928

/-- The area of a regular octagon with side length s -/
noncomputable def regularOctagonArea (s : ℝ) : ℝ := 2 * s^2 * (1 + Real.sqrt 2)

theorem regular_octagon_area_1_5_sqrt_2 :
  regularOctagonArea (1.5 * Real.sqrt 2) = 9 + 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_area_1_5_sqrt_2_l4129_412928


namespace NUMINAMATH_CALUDE_parabola_translation_l4129_412969

/-- Given two parabolas, prove that one is a translation of the other -/
theorem parabola_translation (x y : ℝ) :
  (y = 2 * x^2) → (y + 1 = 2 * (x - 4)^2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l4129_412969


namespace NUMINAMATH_CALUDE_circle_line_regions_l4129_412966

/-- Represents a configuration of concentric circles and intersecting lines. -/
structure CircleLineConfiguration where
  n : ℕ  -- number of concentric circles
  k : ℕ  -- number of lines through point A
  m : ℕ  -- number of lines through point B

/-- Calculates the maximum number of regions formed by the configuration. -/
def max_regions (config : CircleLineConfiguration) : ℕ :=
  (config.k + 1) * (config.m + 1) * config.n

/-- Calculates the minimum number of regions formed by the configuration. -/
def min_regions (config : CircleLineConfiguration) : ℕ :=
  (config.k + config.m + 1) + config.n - 1

/-- Theorem stating the maximum and minimum number of regions formed. -/
theorem circle_line_regions (config : CircleLineConfiguration) :
  (max_regions config = (config.k + 1) * (config.m + 1) * config.n) ∧
  (min_regions config = (config.k + config.m + 1) + config.n - 1) :=
sorry

end NUMINAMATH_CALUDE_circle_line_regions_l4129_412966


namespace NUMINAMATH_CALUDE_young_bonnet_ratio_l4129_412993

/-- Mrs. Young's bonnet making problem -/
theorem young_bonnet_ratio :
  let monday_bonnets : ℕ := 10
  let thursday_bonnets : ℕ := monday_bonnets + 5
  let friday_bonnets : ℕ := thursday_bonnets - 5
  let total_orphanages : ℕ := 5
  let bonnets_per_orphanage : ℕ := 11
  let total_bonnets : ℕ := total_orphanages * bonnets_per_orphanage
  let tues_wed_bonnets : ℕ := total_bonnets - (monday_bonnets + thursday_bonnets + friday_bonnets)
  tues_wed_bonnets / monday_bonnets = 2 := by
  sorry

end NUMINAMATH_CALUDE_young_bonnet_ratio_l4129_412993


namespace NUMINAMATH_CALUDE_basketball_game_result_l4129_412931

/-- Represents a basketball player with their score and penalties -/
structure Player where
  score : ℕ
  penalties : List ℕ

/-- Calculates the total points for a player after applying penalties -/
def playerPoints (p : Player) : ℤ :=
  p.score - (List.sum p.penalties)

/-- Calculates the total points for a team -/
def teamPoints (team : List Player) : ℤ :=
  List.sum (team.map playerPoints)

theorem basketball_game_result :
  let team_a := [
    Player.mk 12 [1, 2],
    Player.mk 18 [1, 2, 3],
    Player.mk 5 [],
    Player.mk 7 [1, 2],
    Player.mk 6 [1]
  ]
  let team_b := [
    Player.mk 10 [1, 2],
    Player.mk 9 [1],
    Player.mk 12 [],
    Player.mk 8 [1, 2, 3],
    Player.mk 5 [1, 2],
    Player.mk 4 [1]
  ]
  teamPoints team_a - teamPoints team_b = 1 := by
  sorry


end NUMINAMATH_CALUDE_basketball_game_result_l4129_412931


namespace NUMINAMATH_CALUDE_divisor_problem_l4129_412964

theorem divisor_problem (D : ℚ) : D ≠ 0 → (72 / D + 5 = 17) → D = 6 := by sorry

end NUMINAMATH_CALUDE_divisor_problem_l4129_412964


namespace NUMINAMATH_CALUDE_race_distance_l4129_412981

/-- The race problem -/
theorem race_distance (d : ℝ) (a b c : ℝ) : 
  d > 0 ∧ a > b ∧ b > c ∧ a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive distances and speeds
  d / a = (d - 15) / b →  -- A beats B by 15 meters
  d / b = (d - 30) / c →  -- B beats C by 30 meters
  d / a = (d - 40) / c →  -- A beats C by 40 meters
  d = 90 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l4129_412981


namespace NUMINAMATH_CALUDE_abs_neg_half_eq_half_l4129_412990

theorem abs_neg_half_eq_half : |(-1/2 : ℚ)| = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_half_eq_half_l4129_412990


namespace NUMINAMATH_CALUDE_cupcake_distribution_exists_l4129_412974

theorem cupcake_distribution_exists (total_cupcakes : ℕ) 
  (cupcakes_per_cousin : ℕ) (cupcakes_per_friend : ℕ) : 
  total_cupcakes = 42 → cupcakes_per_cousin = 3 → cupcakes_per_friend = 2 →
  ∃ (n : ℕ), ∃ (cousins : ℕ), ∃ (friends : ℕ),
    n = cousins + friends ∧ 
    cousins * cupcakes_per_cousin + friends * cupcakes_per_friend = total_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_cupcake_distribution_exists_l4129_412974


namespace NUMINAMATH_CALUDE_expression_bounds_l4129_412900

theorem expression_bounds (a b c d e : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  let expr := Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
              Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + 
              Real.sqrt (e^2 + (1-a)^2)
  5 / Real.sqrt 2 ≤ expr ∧ expr ≤ 5 ∧ 
  ∃ (a' b' c' d' e' : Real), (0 ≤ a' ∧ a' ≤ 1) ∧ (0 ≤ b' ∧ b' ≤ 1) ∧ 
    (0 ≤ c' ∧ c' ≤ 1) ∧ (0 ≤ d' ∧ d' ≤ 1) ∧ (0 ≤ e' ∧ e' ≤ 1) ∧
    let expr' := Real.sqrt (a'^2 + (1-b')^2) + Real.sqrt (b'^2 + (1-c')^2) + 
                 Real.sqrt (c'^2 + (1-d')^2) + Real.sqrt (d'^2 + (1-e')^2) + 
                 Real.sqrt (e'^2 + (1-a')^2)
    expr' = 5 / Real.sqrt 2 ∧
  ∃ (a'' b'' c'' d'' e'' : Real), (0 ≤ a'' ∧ a'' ≤ 1) ∧ (0 ≤ b'' ∧ b'' ≤ 1) ∧ 
    (0 ≤ c'' ∧ c'' ≤ 1) ∧ (0 ≤ d'' ∧ d'' ≤ 1) ∧ (0 ≤ e'' ∧ e'' ≤ 1) ∧
    let expr'' := Real.sqrt (a''^2 + (1-b'')^2) + Real.sqrt (b''^2 + (1-c'')^2) + 
                  Real.sqrt (c''^2 + (1-d'')^2) + Real.sqrt (d''^2 + (1-e'')^2) + 
                  Real.sqrt (e''^2 + (1-a'')^2)
    expr'' = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l4129_412900


namespace NUMINAMATH_CALUDE_expand_product_l4129_412960

theorem expand_product (x : ℝ) : (x + 3) * (x + 8) = x^2 + 11*x + 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l4129_412960


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l4129_412980

def U : Set ℝ := {x | x ≥ 0}
def A : Set ℝ := {x | x ≥ 1}

theorem complement_of_A_in_U : 
  (U \ A) = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l4129_412980


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l4129_412921

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 4 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 5 * x = y^2) → (∃ (z : ℕ), 4 * x = z^3) → x ≥ n) ∧
  n = 1080 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l4129_412921


namespace NUMINAMATH_CALUDE_second_run_time_l4129_412927

/-- Represents the time in seconds for various parts of the obstacle course challenge -/
structure ObstacleCourseTime where
  totalSecondRun : ℕ
  doorOpenTime : ℕ

/-- Calculates the time for the second run without backpack -/
def secondRunWithoutBackpack (t : ObstacleCourseTime) : ℕ :=
  t.totalSecondRun - t.doorOpenTime

/-- Theorem stating that for the given times, the second run without backpack takes 801 seconds -/
theorem second_run_time (t : ObstacleCourseTime) 
    (h1 : t.totalSecondRun = 874)
    (h2 : t.doorOpenTime = 73) : 
  secondRunWithoutBackpack t = 801 := by
  sorry

end NUMINAMATH_CALUDE_second_run_time_l4129_412927


namespace NUMINAMATH_CALUDE_abs_m_minus_n_l4129_412991

theorem abs_m_minus_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_l4129_412991


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l4129_412987

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a = 4 * b →    -- ratio of angles is 4:1
  b = 36 :=      -- smaller angle is 36°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l4129_412987


namespace NUMINAMATH_CALUDE_difference_of_squares_l4129_412941

theorem difference_of_squares (m : ℝ) : (m + 1) * (m - 1) = m^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4129_412941


namespace NUMINAMATH_CALUDE_expression_value_l4129_412911

theorem expression_value (x y z : ℝ) (hx : x = 1) (hy : y = 1) (hz : z = 3) :
  x^2 * y * z - x * y * z^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4129_412911


namespace NUMINAMATH_CALUDE_f_2019_equals_2_l4129_412959

def f_property (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) + f (x - 2) = 2 * f 2) ∧
  (∀ x, f (x + 1) = -f (-x - 1)) ∧
  (f 1 = 2)

theorem f_2019_equals_2 (f : ℝ → ℝ) (h : f_property f) : f 2019 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_equals_2_l4129_412959


namespace NUMINAMATH_CALUDE_savings_calculation_l4129_412984

def calculate_savings (initial_amount : ℚ) (tax_rate : ℚ) (bike_spending_rate : ℚ) : ℚ :=
  let after_tax := initial_amount * (1 - tax_rate)
  let bike_cost := after_tax * bike_spending_rate
  after_tax - bike_cost

theorem savings_calculation :
  calculate_savings 125 0.2 0.8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l4129_412984


namespace NUMINAMATH_CALUDE_field_area_l4129_412947

/-- A rectangular field with specific properties -/
structure RectangularField where
  breadth : ℝ
  length : ℝ
  length_relation : length = breadth + 30
  perimeter : ℝ
  perimeter_formula : perimeter = 2 * (length + breadth)
  perimeter_value : perimeter = 540

/-- The area of the rectangular field is 18000 square metres -/
theorem field_area (field : RectangularField) : field.length * field.breadth = 18000 := by
  sorry

end NUMINAMATH_CALUDE_field_area_l4129_412947


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l4129_412916

/-- Defines the equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 - 16*y^2 - 8*x + 16*y + 32 = 0

/-- Theorem stating that the conic equation represents a hyperbola -/
theorem conic_is_hyperbola :
  ∃ (h k a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ (x y : ℝ), conic_equation x y ↔ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l4129_412916


namespace NUMINAMATH_CALUDE_inverse_variation_cube_root_l4129_412989

theorem inverse_variation_cube_root (y x : ℝ) (k : ℝ) (h1 : y * x^(1/3) = k) (h2 : 2 * 8^(1/3) = k) :
  8 * x^(1/3) = k → x = 1/8 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_root_l4129_412989


namespace NUMINAMATH_CALUDE_smallest_tablecloth_diameter_l4129_412906

/-- The smallest diameter of a circular tablecloth that can completely cover a square table with sides of 1 meter is √2 meters. -/
theorem smallest_tablecloth_diameter (table_side : ℝ) (h : table_side = 1) :
  let diagonal := Real.sqrt (2 * table_side ^ 2)
  diagonal = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_tablecloth_diameter_l4129_412906


namespace NUMINAMATH_CALUDE_paths_to_2005_l4129_412963

/-- Represents the number of choices for each step in forming the number 2005 -/
structure PathChoices where
  first_zero : Nat
  second_zero : Nat
  final_five : Nat

/-- Calculates the total number of paths to form 2005 -/
def total_paths (choices : PathChoices) : Nat :=
  choices.first_zero * choices.second_zero * choices.final_five

/-- The given choices for each step in forming 2005 -/
def given_choices : PathChoices :=
  { first_zero := 6
  , second_zero := 2
  , final_five := 3 }

/-- Theorem stating that there are 36 different paths to form 2005 -/
theorem paths_to_2005 : total_paths given_choices = 36 := by
  sorry

#eval total_paths given_choices

end NUMINAMATH_CALUDE_paths_to_2005_l4129_412963


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l4129_412903

/-- Given a line L1 with equation 4x - 2y + 1 = 0, prove that the line L2 passing through
    the point (2, -3) and perpendicular to L1 has the equation x + 2y + 4 = 0. -/
theorem perpendicular_line_equation (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 4 * x - 2 * y + 1 = 0
  let P : ℝ × ℝ := (2, -3)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ x + 2 * y + 4 = 0
  (∀ x y, L1 x y → L2 x y → (y - P.2) = -(x - P.1)) →
  (∀ x y, L1 x y → L2 x y → (y - P.2) * (x - P.1) = -1) →
  L2 P.1 P.2 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l4129_412903


namespace NUMINAMATH_CALUDE_number_accurate_to_hundreds_l4129_412901

def number : ℝ := 1.45 * 10^4

def accurate_to_hundreds_place (x : ℝ) : Prop :=
  ∃ n : ℤ, x = (n * 100 : ℝ) ∧ n ≥ 0 ∧ n < 1000

theorem number_accurate_to_hundreds : accurate_to_hundreds_place number := by
  sorry

end NUMINAMATH_CALUDE_number_accurate_to_hundreds_l4129_412901


namespace NUMINAMATH_CALUDE_domain_subset_theorem_l4129_412967

theorem domain_subset_theorem (a : ℝ) : 
  (Set.Ioo a (a + 1) ⊆ Set.Ioo (-1) 1) ↔ a ∈ Set.Icc (-1) 0 := by
sorry

end NUMINAMATH_CALUDE_domain_subset_theorem_l4129_412967


namespace NUMINAMATH_CALUDE_G_minimized_at_three_l4129_412936

/-- The number of devices required for a base-n system -/
noncomputable def G (n : ℕ) (M : ℕ) : ℝ :=
  (n : ℝ) / Real.log n * Real.log (M + 1)

/-- The theorem stating that G is minimized when n = 3 -/
theorem G_minimized_at_three (M : ℕ) :
  ∀ n : ℕ, n ≥ 2 → G 3 M ≤ G n M :=
sorry

end NUMINAMATH_CALUDE_G_minimized_at_three_l4129_412936


namespace NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l4129_412957

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_through_isosceles_triangle_vertices (a b c : ℝ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 4) :
  let r := (a * b * c) / (4 * (1/2 * c * (a^2 - (c/2)^2).sqrt))
  π * r^2 = (13125/1764) * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l4129_412957


namespace NUMINAMATH_CALUDE_sin_cos_tan_relation_l4129_412939

theorem sin_cos_tan_relation (A : Real) (q : Real) 
  (h1 : Real.sin A = 3/5)
  (h2 : Real.cos A / Real.tan A = q/15) :
  q = 16 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_tan_relation_l4129_412939


namespace NUMINAMATH_CALUDE_peters_remaining_money_l4129_412973

-- Define the initial amount and expenses
def initial_amount : ℝ := 500

def first_trip_expenses : ℝ :=
  6 * 2 + 9 * 3 + 5 * 4 + 3 * 5 + 2 * 3.5 + 7 * 4.25 + 4 * 6 + 8 * 5.5

def second_trip_expenses : ℝ :=
  2 * 1.5 + 5 * 2.75

-- Define the theorem
theorem peters_remaining_money :
  initial_amount - (first_trip_expenses + second_trip_expenses) = 304.5 := by
  sorry

end NUMINAMATH_CALUDE_peters_remaining_money_l4129_412973


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l4129_412917

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^1200 - 1) (2^1230 - 1) = 2^30 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l4129_412917


namespace NUMINAMATH_CALUDE_nickels_count_l4129_412949

/-- Proves that given 70 coins consisting of nickels and dimes with a total value of $5.55, the number of nickels is 29. -/
theorem nickels_count (total_coins : ℕ) (total_value : ℚ) (nickels : ℕ) (dimes : ℕ) :
  total_coins = 70 →
  total_value = 555/100 →
  total_coins = nickels + dimes →
  total_value = (5/100 : ℚ) * nickels + (10/100 : ℚ) * dimes →
  nickels = 29 := by
sorry

end NUMINAMATH_CALUDE_nickels_count_l4129_412949


namespace NUMINAMATH_CALUDE_age_when_hired_l4129_412971

/-- Rule of 70 provision for retirement eligibility -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1990

/-- The year the employee became eligible for retirement -/
def retirement_eligibility_year : ℕ := 2009

/-- Years employed before retirement eligibility -/
def years_employed : ℕ := retirement_eligibility_year - hire_year

/-- Theorem stating the employee's age when hired -/
theorem age_when_hired :
  ∃ (age : ℕ), rule_of_70 (age + years_employed) years_employed ∧ age = 51 := by
  sorry

end NUMINAMATH_CALUDE_age_when_hired_l4129_412971


namespace NUMINAMATH_CALUDE_line_symmetry_l4129_412961

-- Define the lines
def original_line (x y : ℝ) : Prop := 2*x - y + 3 = 0
def reference_line (x y : ℝ) : Prop := x - y + 2 = 0
def symmetric_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (l1 l2 l_ref : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x y : ℝ), l1 x y → ∃ (x' y' : ℝ), l2 x' y' ∧
    (x + x') / 2 = (y + y') / 2 + 2 ∧ -- Point on reference line
    (y' - y) = (x' - x) -- Perpendicular to reference line

-- Theorem statement
theorem line_symmetry :
  symmetric_wrt original_line symmetric_line reference_line :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l4129_412961


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l4129_412919

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x + |2*x - 3| > 0} = {x : ℝ | x < 2/3 ∨ x > 2} :=
sorry

-- Part II
theorem range_of_a_part_ii :
  {a : ℝ | ∀ x, f a x < |x - 3|} = {a : ℝ | 1 < a ∧ a < 5} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l4129_412919


namespace NUMINAMATH_CALUDE_expand_product_l4129_412946

theorem expand_product (x : ℝ) : (5*x + 3) * (3*x^2 + 4) = 15*x^3 + 9*x^2 + 20*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l4129_412946


namespace NUMINAMATH_CALUDE_book_pages_calculation_l4129_412954

theorem book_pages_calculation (total_pages : ℕ) : 
  (7 : ℚ) / 13 * total_pages + 
  (5 : ℚ) / 9 * ((6 : ℚ) / 13 * total_pages) + 
  96 = total_pages → 
  total_pages = 468 := by
sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l4129_412954


namespace NUMINAMATH_CALUDE_quadratic_polynomial_discriminant_l4129_412915

-- Define a quadratic polynomial
def QuadraticPolynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic polynomial
def Discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_polynomial_discriminant 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∃! x, QuadraticPolynomial a b c x = x - 2) ∧ 
  (∃! x, QuadraticPolynomial a b c x = 1 - x/2) →
  Discriminant a b c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_discriminant_l4129_412915


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l4129_412907

theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = Real.sqrt 22 ∧
  (∀ x y : ℝ, 2 * x^2 + 3 * y^2 = 1 → x + 2*y ≤ M) ∧
  (∃ x y : ℝ, 2 * x^2 + 3 * y^2 = 1 ∧ x + 2*y = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l4129_412907


namespace NUMINAMATH_CALUDE_ethan_works_five_days_per_week_l4129_412978

/-- Calculates the number of days Ethan works per week given his hourly rate, daily hours, total earnings, and number of weeks worked. -/
def days_worked_per_week (hourly_rate : ℚ) (hours_per_day : ℚ) (total_earnings : ℚ) (num_weeks : ℚ) : ℚ :=
  total_earnings / num_weeks / (hourly_rate * hours_per_day)

/-- Proves that Ethan works 5 days per week given the problem conditions. -/
theorem ethan_works_five_days_per_week :
  days_worked_per_week 18 8 3600 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ethan_works_five_days_per_week_l4129_412978


namespace NUMINAMATH_CALUDE_leaf_movement_l4129_412933

theorem leaf_movement (forward_distance : ℕ) (num_gusts : ℕ) (total_distance : ℕ) 
  (h1 : forward_distance = 5)
  (h2 : num_gusts = 11)
  (h3 : total_distance = 33) :
  ∃ (backward_distance : ℕ), 
    num_gusts * (forward_distance - backward_distance) = total_distance ∧ 
    backward_distance = 2 :=
by sorry

end NUMINAMATH_CALUDE_leaf_movement_l4129_412933


namespace NUMINAMATH_CALUDE_min_value_expression_l4129_412983

theorem min_value_expression (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ 2) (h3 : 2 ≤ b) (h4 : b ≤ c) (h5 : c ≤ 6) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (6/c - 1)^2 ≥ 16 - 8 * Real.sqrt 3 ∧
  ∃ (a₀ b₀ c₀ : ℝ), 1 ≤ a₀ ∧ a₀ ≤ 2 ∧ 2 ≤ b₀ ∧ b₀ ≤ c₀ ∧ c₀ ≤ 6 ∧
    (a₀ - 1)^2 + (b₀/a₀ - 1)^2 + (c₀/b₀ - 1)^2 + (6/c₀ - 1)^2 = 16 - 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4129_412983


namespace NUMINAMATH_CALUDE_lyle_friends_served_l4129_412988

/-- Calculates the maximum number of friends who can have a sandwich and a juice pack. -/
def max_friends_served (sandwich_cost juice_cost total_money : ℚ) : ℕ :=
  let cost_per_person := sandwich_cost + juice_cost
  let total_servings := (total_money / cost_per_person).floor
  (total_servings - 1).natAbs

/-- Proves that Lyle can buy a sandwich and a juice pack for 4 friends. -/
theorem lyle_friends_served :
  max_friends_served 0.30 0.20 2.50 = 4 := by
  sorry

#eval max_friends_served 0.30 0.20 2.50

end NUMINAMATH_CALUDE_lyle_friends_served_l4129_412988


namespace NUMINAMATH_CALUDE_alien_year_conversion_l4129_412914

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The problem statement --/
theorem alien_year_conversion :
  base8ToBase10 [2, 6, 3] = 242 := by
  sorry

end NUMINAMATH_CALUDE_alien_year_conversion_l4129_412914


namespace NUMINAMATH_CALUDE_solve_equation_l4129_412908

theorem solve_equation (x y : ℝ) (h1 : (12 : ℝ)^2 * x^4 / 432 = y) (h2 : y = 432) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4129_412908


namespace NUMINAMATH_CALUDE_race_distance_l4129_412932

theorem race_distance (total : ℝ) (selena : ℝ) (josh : ℝ) 
  (h1 : total = 36)
  (h2 : selena + josh = total)
  (h3 : josh = selena / 2) : 
  selena = 24 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l4129_412932


namespace NUMINAMATH_CALUDE_modular_inverse_of_35_mod_37_l4129_412953

theorem modular_inverse_of_35_mod_37 :
  ∃ x : ℤ, (35 * x) % 37 = 1 ∧ x % 37 = 18 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_35_mod_37_l4129_412953


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_five_l4129_412934

theorem cubic_fraction_equals_five :
  let a : ℚ := 3
  let b : ℚ := 2
  (a^3 + b^3) / (a^2 - 2*a*b + b^2 + a*b) = 5 := by sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_five_l4129_412934


namespace NUMINAMATH_CALUDE_cards_distribution_theorem_l4129_412944

/-- Given a total number of cards and people, calculate how many people receive fewer cards when dealt as evenly as possible. -/
def people_with_fewer_cards (total_cards : ℕ) (num_people : ℕ) (threshold : ℕ) : ℕ :=
  let cards_per_person := total_cards / num_people
  let extra_cards := total_cards % num_people
  if cards_per_person + 1 < threshold then num_people
  else num_people - extra_cards

/-- Theorem stating that when 60 cards are dealt to 9 people as evenly as possible, 3 people will have fewer than 7 cards. -/
theorem cards_distribution_theorem :
  people_with_fewer_cards 60 9 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_theorem_l4129_412944


namespace NUMINAMATH_CALUDE_square_difference_solutions_l4129_412972

/-- A function to check if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m * m

/-- The main theorem stating the solution to the problem -/
theorem square_difference_solutions :
  ∀ a b : ℕ,
    a > 0 ∧ b > 0 →
    is_perfect_square (a^2 - 4*b) ∧ is_perfect_square (b^2 - 4*a) →
    ((a = 4 ∧ b = 4) ∨ (a = 5 ∧ b = 6) ∨ (a = 6 ∧ b = 5)) :=
by sorry

end NUMINAMATH_CALUDE_square_difference_solutions_l4129_412972


namespace NUMINAMATH_CALUDE_solutions_to_z_sixth_eq_neg_eight_l4129_412905

theorem solutions_to_z_sixth_eq_neg_eight :
  {z : ℂ | z^6 = -8} = {1 + I, 1 - I, -1 + I, -1 - I} := by
  sorry

end NUMINAMATH_CALUDE_solutions_to_z_sixth_eq_neg_eight_l4129_412905


namespace NUMINAMATH_CALUDE_cyclist_speed_north_l4129_412962

/-- The speed of the cyclist going north -/
def speed_north : ℝ := 10

/-- The speed of the cyclist going south -/
def speed_south : ℝ := 40

/-- The time taken -/
def time : ℝ := 1

/-- The distance between the cyclists after the given time -/
def distance : ℝ := 50

/-- Theorem stating that the speed of the cyclist going north is 10 km/h -/
theorem cyclist_speed_north : 
  speed_north + speed_south = distance / time :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_north_l4129_412962


namespace NUMINAMATH_CALUDE_remainder_problem_l4129_412965

theorem remainder_problem (N : ℕ) : 
  (∃ R, N = 7 * 5 + R ∧ R < 7) → 
  (∃ Q, N = 11 * Q + 2) → 
  N % 7 = 4 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l4129_412965


namespace NUMINAMATH_CALUDE_num_girls_on_playground_l4129_412955

/-- The number of boys on the playground -/
def num_boys : ℝ := 35.0

/-- The difference between the number of boys and girls -/
def boy_girl_difference : ℝ := 7

/-- Theorem: The number of girls on the playground is 28.0 -/
theorem num_girls_on_playground : 
  ∃ (num_girls : ℝ), num_girls = num_boys - boy_girl_difference := by
  sorry

end NUMINAMATH_CALUDE_num_girls_on_playground_l4129_412955


namespace NUMINAMATH_CALUDE_ratio_problem_l4129_412996

theorem ratio_problem (a b c d e f : ℚ) 
  (h1 : a * b * c / (d * e * f) = 1.875)
  (h2 : a / b = 5 / 2)
  (h3 : b / c = 1 / 2)
  (h4 : c / d = 1)
  (h5 : d / e = 3 / 2) :
  e / f = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l4129_412996


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4129_412918

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h1 : a 4 + a 10 + a 16 = 30) : a 18 - 2 * a 14 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4129_412918


namespace NUMINAMATH_CALUDE_average_of_combined_sets_l4129_412968

theorem average_of_combined_sets :
  let set1_count : ℕ := 60
  let set1_avg : ℚ := 40
  let set2_count : ℕ := 40
  let set2_avg : ℚ := 60
  let total_count : ℕ := set1_count + set2_count
  let total_sum : ℚ := set1_count * set1_avg + set2_count * set2_avg
  total_sum / total_count = 48 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_sets_l4129_412968


namespace NUMINAMATH_CALUDE_sqrt_calculation_l4129_412920

theorem sqrt_calculation :
  (Real.sqrt 80 - Real.sqrt 20 + Real.sqrt 5 = 3 * Real.sqrt 5) ∧
  (2 * Real.sqrt 6 * 3 * Real.sqrt (1/2) / Real.sqrt 3 = 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l4129_412920


namespace NUMINAMATH_CALUDE_infinite_primes_l4129_412943

theorem infinite_primes (S : Finset Nat) (h : ∀ p ∈ S, Nat.Prime p) : 
  ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_l4129_412943


namespace NUMINAMATH_CALUDE_alex_and_carla_weight_l4129_412948

/-- Given the weights of pairs of individuals, prove that Alex and Carla weigh 235 pounds together. -/
theorem alex_and_carla_weight
  (alex_ben : ℝ)
  (ben_carla : ℝ)
  (carla_derek : ℝ)
  (alex_derek : ℝ)
  (h1 : alex_ben = 280)
  (h2 : ben_carla = 235)
  (h3 : carla_derek = 260)
  (h4 : alex_derek = 295) :
  ∃ (a b c d : ℝ),
    a + b = alex_ben ∧
    b + c = ben_carla ∧
    c + d = carla_derek ∧
    a + d = alex_derek ∧
    a + c = 235 := by
  sorry

end NUMINAMATH_CALUDE_alex_and_carla_weight_l4129_412948


namespace NUMINAMATH_CALUDE_industrial_machine_output_l4129_412975

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  totalShirts : ℕ
  workingMinutes : ℕ

/-- Calculate the shirts per minute for a given machine -/
def shirtsPerMinute (machine : ShirtMachine) : ℚ :=
  machine.totalShirts / machine.workingMinutes

theorem industrial_machine_output (machine : ShirtMachine) 
  (h1 : machine.totalShirts = 6)
  (h2 : machine.workingMinutes = 2) : 
  shirtsPerMinute machine = 3 := by
  sorry

end NUMINAMATH_CALUDE_industrial_machine_output_l4129_412975


namespace NUMINAMATH_CALUDE_sarah_amount_l4129_412950

-- Define the total amount Bridge and Sarah have
def total : ℕ := 300

-- Define the difference between Bridget's and Sarah's amounts
def difference : ℕ := 50

-- Theorem to prove
theorem sarah_amount : ∃ (s : ℕ), s + (s + difference) = total ∧ s = 125 := by
  sorry

end NUMINAMATH_CALUDE_sarah_amount_l4129_412950


namespace NUMINAMATH_CALUDE_susan_ate_six_candies_l4129_412925

/-- The number of candies Susan ate during the week -/
def candies_eaten (bought_tuesday bought_thursday bought_friday remaining : ℕ) : ℕ :=
  bought_tuesday + bought_thursday + bought_friday - remaining

/-- Theorem stating that Susan ate 6 candies during the week -/
theorem susan_ate_six_candies :
  candies_eaten 3 5 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_ate_six_candies_l4129_412925


namespace NUMINAMATH_CALUDE_daves_sticks_l4129_412995

theorem daves_sticks (sticks_picked : ℕ) (sticks_left : ℕ) : 
  sticks_left = 4 → 
  sticks_picked - sticks_left = 10 → 
  sticks_picked = 14 := by
  sorry

end NUMINAMATH_CALUDE_daves_sticks_l4129_412995


namespace NUMINAMATH_CALUDE_mikes_seashells_l4129_412956

theorem mikes_seashells (unbroken_seashells broken_seashells : ℕ) 
  (h1 : unbroken_seashells = 2) 
  (h2 : broken_seashells = 4) : 
  unbroken_seashells + broken_seashells = 6 := by
  sorry

end NUMINAMATH_CALUDE_mikes_seashells_l4129_412956


namespace NUMINAMATH_CALUDE_seungchan_book_pages_l4129_412985

/-- The number of pages in Seungchan's children's book -/
def total_pages : ℝ := 250

/-- The fraction of the book Seungchan read until yesterday -/
def read_yesterday : ℝ := 0.2

/-- The fraction of the remaining part Seungchan read today -/
def read_today : ℝ := 0.35

/-- The number of pages left after today's reading -/
def pages_left : ℝ := 130

theorem seungchan_book_pages :
  (1 - read_yesterday) * (1 - read_today) * total_pages = pages_left :=
sorry

end NUMINAMATH_CALUDE_seungchan_book_pages_l4129_412985


namespace NUMINAMATH_CALUDE_ten_thousand_one_divides_repeat_digit_number_l4129_412913

/-- An 8-digit positive integer with the first four digits repeated -/
def RepeatDigitNumber (a b c d : Nat) : Nat :=
  a * 10000000 + b * 1000000 + c * 100000 + d * 10000 +
  a * 1000 + b * 100 + c * 10 + d

/-- Theorem: 10001 is a factor of any 8-digit number with repeated first four digits -/
theorem ten_thousand_one_divides_repeat_digit_number 
  (a b c d : Nat) (ha : a > 0) (hb : b < 10) (hc : c < 10) (hd : d < 10) :
  10001 ∣ RepeatDigitNumber a b c d := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_one_divides_repeat_digit_number_l4129_412913


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_eq_neg_two_l4129_412945

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem stating that if vectors (2, 3m+2) and (m, -1) are perpendicular, then m = -2 -/
theorem perpendicular_vectors_imply_m_eq_neg_two (m : ℝ) :
  perpendicular (2, 3*m+2) (m, -1) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_eq_neg_two_l4129_412945


namespace NUMINAMATH_CALUDE_value_of_C_l4129_412942

theorem value_of_C : ∃ C : ℝ, (4 * C + 3 = 25) ∧ (C = 5.5) := by sorry

end NUMINAMATH_CALUDE_value_of_C_l4129_412942


namespace NUMINAMATH_CALUDE_emissions_2019_safe_m_range_l4129_412924

/-- Represents the carbon emissions of City A over years -/
def CarbonEmissions (m : ℝ) : ℕ → ℝ
  | 0 => 400  -- 2017 emissions
  | n + 1 => 0.9 * CarbonEmissions m n + m

/-- The maximum allowed annual carbon emissions -/
def MaxEmissions : ℝ := 550

/-- Theorem stating the carbon emissions of City A in 2019 -/
theorem emissions_2019 (m : ℝ) (h : m > 0) : 
  CarbonEmissions m 2 = 324 + 1.9 * m := by sorry

/-- Theorem stating the range of m for which emergency measures are never needed -/
theorem safe_m_range : 
  ∀ m : ℝ, (m > 0 ∧ m ≤ 55) ↔ 
    (∀ n : ℕ, CarbonEmissions m n ≤ MaxEmissions) := by sorry

end NUMINAMATH_CALUDE_emissions_2019_safe_m_range_l4129_412924


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l4129_412940

theorem solution_satisfies_equations :
  let x : ℚ := -256 / 29
  let y : ℚ := -37 / 29
  (7 * x - 50 * y = 2) ∧ (3 * y - x = 5) := by
sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l4129_412940


namespace NUMINAMATH_CALUDE_balls_remaining_l4129_412909

/-- The number of baskets --/
def num_baskets : ℕ := 5

/-- The number of tennis balls in each basket --/
def tennis_balls_per_basket : ℕ := 15

/-- The number of soccer balls in each basket --/
def soccer_balls_per_basket : ℕ := 5

/-- The number of students who removed 8 balls each --/
def students_eight : ℕ := 3

/-- The number of students who removed 10 balls each --/
def students_ten : ℕ := 2

/-- The number of balls removed by each student in the first group --/
def balls_removed_eight : ℕ := 8

/-- The number of balls removed by each student in the second group --/
def balls_removed_ten : ℕ := 10

/-- Theorem: The number of balls remaining in the baskets is 56 --/
theorem balls_remaining :
  (num_baskets * (tennis_balls_per_basket + soccer_balls_per_basket)) -
  (students_eight * balls_removed_eight + students_ten * balls_removed_ten) = 56 := by
  sorry

end NUMINAMATH_CALUDE_balls_remaining_l4129_412909


namespace NUMINAMATH_CALUDE_initial_amount_proof_l4129_412926

theorem initial_amount_proof (P : ℚ) : 
  (P * (1 + 1/8) * (1 + 1/8) = 105300) → P = 83200 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l4129_412926


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4129_412910

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ x = 1 ∧ y = 0) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 5) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4129_412910


namespace NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l4129_412970

theorem max_area_inscribed_rectangle (h : Real) (α : Real) (x y : Real) :
  h = 24 → α = π / 3 →
  x > 0 → y > 0 →
  x * y ≤ (3 * Real.sqrt 3) * 12 :=
by sorry

end NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l4129_412970


namespace NUMINAMATH_CALUDE_max_sum_of_distances_l4129_412977

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a triangle is isosceles with AB = AC = b and BC = a -/
def isIsoscelesTriangle (t : Triangle) (a b : ℝ) : Prop :=
  let d (p q : Point) := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)
  d t.A t.B = b ∧ d t.A t.C = b ∧ d t.B t.C = a

/-- Checks if a point is inside a triangle -/
def isInside (P : Point) (t : Triangle) : Prop := sorry

/-- Calculates the sum of distances from a point to each side of a triangle -/
def sumOfDistances (P : Point) (t : Triangle) : ℝ := sorry

/-- The main theorem -/
theorem max_sum_of_distances (t : Triangle) (a b : ℝ) (P : Point) :
  a ≤ b →
  isIsoscelesTriangle t a b →
  isInside P t →
  sumOfDistances P t < 2 * b + a := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_distances_l4129_412977


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l4129_412912

theorem regular_polygon_sides (central_angle : ℝ) : 
  central_angle = 40 → (360 : ℝ) / central_angle = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l4129_412912


namespace NUMINAMATH_CALUDE_complex_number_location_l4129_412951

theorem complex_number_location (z : ℂ) (h : (3 - 2*I)*z = 4 + 3*I) :
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_complex_number_location_l4129_412951


namespace NUMINAMATH_CALUDE_coin_problem_l4129_412994

theorem coin_problem (total_coins : ℕ) (total_value : ℚ) (nickel_value dime_value : ℚ) :
  total_coins = 28 →
  total_value = 260/100 →
  nickel_value = 5/100 →
  dime_value = 10/100 →
  ∃ (nickels dimes : ℕ),
    nickels + dimes = total_coins ∧
    nickels * nickel_value + dimes * dime_value = total_value ∧
    nickels = 4 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l4129_412994


namespace NUMINAMATH_CALUDE_simplify_expression_l4129_412958

theorem simplify_expression :
  ∃ (a b c : ℕ+),
    c.val = 24 ∧
    a.val = 56 ∧
    b.val = 54 ∧
    (∀ (x y z : ℕ+),
      Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) =
      (x.val * Real.sqrt 6 + y.val * Real.sqrt 8) / z.val →
      z.val ≥ c.val) ∧
    Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) =
    (a.val * Real.sqrt 6 + b.val * Real.sqrt 8) / c.val :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4129_412958


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l4129_412904

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  2 ≤ (1 / a + 1 / b) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l4129_412904


namespace NUMINAMATH_CALUDE_acute_triangle_theorem_l4129_412998

theorem acute_triangle_theorem (A B C : Real) (a b c : Real) :
  0 < A → A < π / 2 →
  0 < B → B < π / 2 →
  0 < C → C < π / 2 →
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  Real.sqrt 3 * b * Real.sin A - a * Real.cos B - a = 0 →
  (B = π / 3) ∧ 
  (3 * Real.sqrt 3 / 2 < Real.sin A + Real.sin C ∧ Real.sin A + Real.sin C ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_theorem_l4129_412998


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l4129_412999

theorem units_digit_of_7_power_2023 : (7^2023 : ℕ) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l4129_412999
