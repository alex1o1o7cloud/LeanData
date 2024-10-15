import Mathlib

namespace NUMINAMATH_CALUDE_g_range_l1070_107017

noncomputable def f (a x : ℝ) : ℝ := a^x / (1 + a^x)

noncomputable def g (a x : ℝ) : ℤ := 
  ⌊f a x - 1/2⌋ + ⌊f a (-x) - 1/2⌋

theorem g_range (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  ∀ x : ℝ, g a x ∈ ({0, -1} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_g_range_l1070_107017


namespace NUMINAMATH_CALUDE_circle_center_on_line_ab_range_l1070_107003

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line equation
def line_eq (a b x y : ℝ) : Prop :=
  a*x - b*y + 1 = 0

-- Define the center of the circle
def center (x y : ℝ) : Prop :=
  x = -1 ∧ y = 2

-- Theorem statement
theorem circle_center_on_line_ab_range :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), circle_eq x y ∧ center x y ∧ line_eq a b x y) →
  0 < a * b ∧ a * b ≤ 2 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_circle_center_on_line_ab_range_l1070_107003


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l1070_107030

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (22 : ℝ) / 100 * total_students = (students_more_than_100 : ℝ))
  (h2 : (75 : ℝ) / 100 * (students_on_trip : ℝ) = (students_not_more_than_100 : ℝ))
  (h3 : students_on_trip = students_more_than_100 + students_not_more_than_100) :
  (students_on_trip : ℝ) / total_students = 88 / 100 :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l1070_107030


namespace NUMINAMATH_CALUDE_Z_in_third_quadrant_l1070_107051

-- Define the complex number Z
def Z : ℂ := -1 + (1 - Complex.I)^2

-- Theorem statement
theorem Z_in_third_quadrant : 
  Z.re < 0 ∧ Z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_Z_in_third_quadrant_l1070_107051


namespace NUMINAMATH_CALUDE_m_less_than_n_l1070_107090

theorem m_less_than_n (x : ℝ) : (x + 2) * (x + 3) < 2 * x^2 + 5 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_n_l1070_107090


namespace NUMINAMATH_CALUDE_house_position_l1070_107018

theorem house_position (total_houses : Nat) (product_difference : Nat) : 
  total_houses = 11 → product_difference = 5 → 
  ∃ (position : Nat), position = 4 ∧ 
    (position - 1) * (total_houses - position) = 
    (position - 2) * (total_houses - position + 1) + product_difference := by
  sorry

end NUMINAMATH_CALUDE_house_position_l1070_107018


namespace NUMINAMATH_CALUDE_joao_chocolate_bars_l1070_107032

theorem joao_chocolate_bars (x y z : ℕ) : 
  x + y + z = 30 →
  2 * x + 3 * y + 4 * z = 100 →
  z > x :=
by sorry

end NUMINAMATH_CALUDE_joao_chocolate_bars_l1070_107032


namespace NUMINAMATH_CALUDE_particle_speed_is_sqrt_34_l1070_107016

/-- A particle moves along a path. Its position at time t is (3t + 1, 5t - 2). -/
def particle_position (t : ℝ) : ℝ × ℝ := (3 * t + 1, 5 * t - 2)

/-- The speed of the particle is defined as the distance traveled per unit time. -/
def particle_speed : ℝ := sorry

/-- Theorem: The speed of the particle is √34 units of distance per unit of time. -/
theorem particle_speed_is_sqrt_34 : particle_speed = Real.sqrt 34 := by sorry

end NUMINAMATH_CALUDE_particle_speed_is_sqrt_34_l1070_107016


namespace NUMINAMATH_CALUDE_mika_stickers_bought_l1070_107001

/-- The number of stickers Mika bought from the store -/
def stickers_bought : ℕ := 26

/-- The number of stickers Mika started with -/
def initial_stickers : ℕ := 20

/-- The number of stickers Mika got for her birthday -/
def birthday_stickers : ℕ := 20

/-- The number of stickers Mika gave to her sister -/
def stickers_given : ℕ := 6

/-- The number of stickers Mika used to decorate a greeting card -/
def stickers_used : ℕ := 58

/-- The number of stickers Mika is left with -/
def remaining_stickers : ℕ := 2

theorem mika_stickers_bought : 
  initial_stickers + birthday_stickers + stickers_bought = 
  stickers_given + stickers_used + remaining_stickers :=
sorry

end NUMINAMATH_CALUDE_mika_stickers_bought_l1070_107001


namespace NUMINAMATH_CALUDE_sqrt_221_range_l1070_107082

theorem sqrt_221_range : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_221_range_l1070_107082


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_two_range_of_a_for_inequality_l1070_107058

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 3| - |x + a|

-- Theorem for part I
theorem solution_set_when_a_eq_two :
  {x : ℝ | f 2 x < 1} = {x : ℝ | x > 0} := by sorry

-- Theorem for part II
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x, f a x ≤ 2*a} = {a : ℝ | a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_two_range_of_a_for_inequality_l1070_107058


namespace NUMINAMATH_CALUDE_drive_time_between_towns_l1070_107073

/-- Proves that the time to drive between two towns is 4 hours given the map distance, scale, and average speed. -/
theorem drive_time_between_towns
  (map_distance : ℝ)
  (scale_distance : ℝ)
  (scale_miles : ℝ)
  (average_speed : ℝ)
  (h1 : map_distance = 12)
  (h2 : scale_distance = 0.5)
  (h3 : scale_miles = 10)
  (h4 : average_speed = 60)
  : (map_distance * scale_miles / scale_distance) / average_speed = 4 :=
by
  sorry

#check drive_time_between_towns

end NUMINAMATH_CALUDE_drive_time_between_towns_l1070_107073


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_63_2898_l1070_107064

theorem gcd_lcm_sum_63_2898 : Nat.gcd 63 2898 + Nat.lcm 63 2898 = 182575 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_63_2898_l1070_107064


namespace NUMINAMATH_CALUDE_jack_marbles_shared_l1070_107059

/-- Calculates the number of marbles shared given initial and remaining marbles -/
def marblesShared (initial remaining : ℕ) : ℕ := initial - remaining

/-- Proves that the number of marbles shared is correct for Jack's scenario -/
theorem jack_marbles_shared :
  marblesShared 62 29 = 33 := by
  sorry

end NUMINAMATH_CALUDE_jack_marbles_shared_l1070_107059


namespace NUMINAMATH_CALUDE_min_rain_day4_exceeds_21_inches_l1070_107042

/-- Represents the rainfall and drainage scenario over 4 days -/
structure RainfallScenario where
  capacity : ℝ  -- Total capacity in inches
  drainRate : ℝ  -- Drainage rate in inches per day
  day1Rain : ℝ  -- Rainfall on day 1 in inches
  day2Rain : ℝ  -- Rainfall on day 2 in inches
  day3Rain : ℝ  -- Rainfall on day 3 in inches

/-- Calculates the minimum rainfall on day 4 to cause flooding -/
def minRainDay4ToFlood (scenario : RainfallScenario) : ℝ :=
  scenario.capacity - (scenario.day1Rain + scenario.day2Rain + scenario.day3Rain - 3 * scenario.drainRate)

/-- Theorem stating the minimum rainfall on day 4 to cause flooding is more than 21 inches -/
theorem min_rain_day4_exceeds_21_inches (scenario : RainfallScenario) 
    (h1 : scenario.capacity = 72) -- 6 feet = 72 inches
    (h2 : scenario.drainRate = 3)
    (h3 : scenario.day1Rain = 10)
    (h4 : scenario.day2Rain = 2 * scenario.day1Rain)
    (h5 : scenario.day3Rain = 1.5 * scenario.day2Rain) : 
  minRainDay4ToFlood scenario > 21 := by
  sorry

#eval minRainDay4ToFlood { capacity := 72, drainRate := 3, day1Rain := 10, day2Rain := 20, day3Rain := 30 }

end NUMINAMATH_CALUDE_min_rain_day4_exceeds_21_inches_l1070_107042


namespace NUMINAMATH_CALUDE_porter_buns_problem_l1070_107067

/-- The maximum number of buns that can be transported given the conditions -/
def max_buns_transported (total_buns : ℕ) (capacity : ℕ) (eaten_per_trip : ℕ) : ℕ :=
  total_buns - (2 * (total_buns / capacity) - 1) * eaten_per_trip

/-- Theorem stating that given the specific conditions, the maximum number of buns transported is 191 -/
theorem porter_buns_problem :
  max_buns_transported 200 40 1 = 191 := by
  sorry

end NUMINAMATH_CALUDE_porter_buns_problem_l1070_107067


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l1070_107038

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss (weight_lost : ℝ) (current_weight : ℝ) (loss_per_day : ℝ) :
  weight_lost = 126 →
  current_weight = 66 →
  loss_per_day = 0.5 →
  ∃ (initial_weight : ℝ) (days : ℝ),
    initial_weight = current_weight + weight_lost ∧
    initial_weight = 192 ∧
    days * loss_per_day = weight_lost :=
by sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l1070_107038


namespace NUMINAMATH_CALUDE_triple_hash_40_l1070_107086

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem triple_hash_40 : hash (hash (hash 40)) = 12.56 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_40_l1070_107086


namespace NUMINAMATH_CALUDE_floor_neg_three_point_seven_l1070_107041

-- Define the greatest integer function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem floor_neg_three_point_seven :
  floor (-3.7) = -4 := by sorry

end NUMINAMATH_CALUDE_floor_neg_three_point_seven_l1070_107041


namespace NUMINAMATH_CALUDE_number_of_houses_street_houses_l1070_107043

/-- Given a street with clotheslines, prove the number of houses -/
theorem number_of_houses (children : ℕ) (adults : ℕ) (child_items : ℕ) (adult_items : ℕ) 
  (items_per_line : ℕ) (lines_per_house : ℕ) : ℕ :=
  let total_items := children * child_items + adults * adult_items
  let total_lines := total_items / items_per_line
  total_lines / lines_per_house

/-- Prove that there are 26 houses on the street -/
theorem street_houses : 
  number_of_houses 11 20 4 3 2 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_of_houses_street_houses_l1070_107043


namespace NUMINAMATH_CALUDE_root_existence_condition_l1070_107046

def f (m : ℝ) (x : ℝ) : ℝ := m * x + 6

theorem root_existence_condition (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 3, f m x = 0) ↔ m ≤ -2 ∨ m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_root_existence_condition_l1070_107046


namespace NUMINAMATH_CALUDE_quadrilateral_sides_diagonals_inequality_l1070_107061

/-- Theorem: For any quadrilateral, the sum of the squares of its sides is not less than
    the sum of the squares of its diagonals. -/
theorem quadrilateral_sides_diagonals_inequality 
  (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : 
  (x₂ - x₁)^2 + (y₂ - y₁)^2 + (x₃ - x₂)^2 + (y₃ - y₂)^2 + 
  (x₄ - x₃)^2 + (y₄ - y₃)^2 + (x₄ - x₁)^2 + (y₄ - y₁)^2 ≥ 
  (x₃ - x₁)^2 + (y₃ - y₁)^2 + (x₄ - x₂)^2 + (y₄ - y₂)^2 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_sides_diagonals_inequality_l1070_107061


namespace NUMINAMATH_CALUDE_complex_multiplication_l1070_107020

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 - 2*i) = 2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1070_107020


namespace NUMINAMATH_CALUDE_pqr_sum_bounds_l1070_107096

theorem pqr_sum_bounds (p q r : ℝ) (h : 5 * (p + q + r) = p^2 + q^2 + r^2) :
  let R := p*q + p*r + q*r
  ∃ (N n : ℝ),
    (∀ x y z : ℝ, 5 * (x + y + z) = x^2 + y^2 + z^2 → x*y + x*z + y*z ≤ N) ∧
    (∀ x y z : ℝ, 5 * (x + y + z) = x^2 + y^2 + z^2 → n ≤ x*y + x*z + y*z) ∧
    N = 150 ∧
    n = -12.5 ∧
    N + 15*n = -37.5 :=
by sorry

end NUMINAMATH_CALUDE_pqr_sum_bounds_l1070_107096


namespace NUMINAMATH_CALUDE_marble_distribution_l1070_107009

theorem marble_distribution (total : ℕ) (first second third : ℚ) : 
  total = 78 →
  first = 3 * second + 2 →
  second = third / 2 →
  first + second + third = total →
  (first = 40 ∧ second = 38/3 ∧ third = 76/3) := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l1070_107009


namespace NUMINAMATH_CALUDE_hardware_store_earnings_l1070_107047

/-- Calculates the total earnings of a hardware store for a week given the sales and prices of various items. -/
theorem hardware_store_earnings 
  (graphics_cards_sold : ℕ) (graphics_card_price : ℕ)
  (hard_drives_sold : ℕ) (hard_drive_price : ℕ)
  (cpus_sold : ℕ) (cpu_price : ℕ)
  (ram_pairs_sold : ℕ) (ram_pair_price : ℕ)
  (h1 : graphics_cards_sold = 10)
  (h2 : graphics_card_price = 600)
  (h3 : hard_drives_sold = 14)
  (h4 : hard_drive_price = 80)
  (h5 : cpus_sold = 8)
  (h6 : cpu_price = 200)
  (h7 : ram_pairs_sold = 4)
  (h8 : ram_pair_price = 60) :
  graphics_cards_sold * graphics_card_price +
  hard_drives_sold * hard_drive_price +
  cpus_sold * cpu_price +
  ram_pairs_sold * ram_pair_price = 8960 := by
  sorry

end NUMINAMATH_CALUDE_hardware_store_earnings_l1070_107047


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1070_107045

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt (2 * x - 1)) = 4 → x = 85 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1070_107045


namespace NUMINAMATH_CALUDE_sum_P_Q_equals_52_l1070_107025

theorem sum_P_Q_equals_52 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3)) →
  P + Q = 52 := by
sorry

end NUMINAMATH_CALUDE_sum_P_Q_equals_52_l1070_107025


namespace NUMINAMATH_CALUDE_running_percentage_is_fifty_percent_l1070_107006

/-- Represents a cricket batsman's score -/
structure BatsmanScore where
  total_runs : ℕ
  boundaries : ℕ
  sixes : ℕ

/-- Calculates the percentage of runs made by running between wickets -/
def runningPercentage (score : BatsmanScore) : ℚ :=
  let boundary_runs := 4 * score.boundaries
  let six_runs := 6 * score.sixes
  let running_runs := score.total_runs - (boundary_runs + six_runs)
  (running_runs : ℚ) / score.total_runs * 100

/-- Theorem: The percentage of runs made by running is 50% for the given score -/
theorem running_percentage_is_fifty_percent (score : BatsmanScore) 
    (h_total : score.total_runs = 120)
    (h_boundaries : score.boundaries = 3)
    (h_sixes : score.sixes = 8) : 
  runningPercentage score = 50 := by
  sorry

end NUMINAMATH_CALUDE_running_percentage_is_fifty_percent_l1070_107006


namespace NUMINAMATH_CALUDE_policeman_speed_l1070_107062

/-- Given a chase scenario between a policeman and a thief, this theorem proves
    the speed of the policeman required to catch the thief. -/
theorem policeman_speed (initial_distance : ℝ) (thief_speed : ℝ) (thief_distance : ℝ) :
  initial_distance = 0.15 →
  thief_speed = 8 →
  thief_distance = 0.6 →
  ∃ (policeman_speed : ℝ),
    policeman_speed * (thief_distance / thief_speed) = initial_distance + thief_distance ∧
    policeman_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_policeman_speed_l1070_107062


namespace NUMINAMATH_CALUDE_find_a_l1070_107091

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, (deriv (f a)) x = 3 * a * x^2 + 6 * x) ∧ (deriv (f a)) (-1) = 3 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1070_107091


namespace NUMINAMATH_CALUDE_equation_solution_l1070_107031

theorem equation_solution : ∃ x : ℝ, 300 * x + (12 + 4) * (1 / 8) = 602 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1070_107031


namespace NUMINAMATH_CALUDE_projection_is_regular_polygon_l1070_107044

-- Define the types of polyhedra
inductive Polyhedron
  | Dodecahedron
  | Icosahedron

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  is_regular : Bool

-- Define a projection function
def project (p : Polyhedron) : RegularPolygon :=
  match p with
  | Polyhedron.Dodecahedron => { sides := 10, is_regular := true }
  | Polyhedron.Icosahedron => { sides := 6, is_regular := true }

-- Theorem statement
theorem projection_is_regular_polygon (p : Polyhedron) :
  (project p).is_regular = true :=
by sorry

end NUMINAMATH_CALUDE_projection_is_regular_polygon_l1070_107044


namespace NUMINAMATH_CALUDE_amoeba_count_after_week_l1070_107060

/-- The number of amoebas after n days, given an initial population of 1 and each amoeba splitting into two every day. -/
def amoeba_count (n : ℕ) : ℕ := 2^n

/-- Theorem stating that the number of amoebas after 7 days is 128. -/
theorem amoeba_count_after_week : amoeba_count 7 = 128 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_week_l1070_107060


namespace NUMINAMATH_CALUDE_full_bucket_weight_l1070_107010

/-- Represents the weight of a bucket with water -/
structure BucketWeight where
  empty : ℝ  -- Weight of the empty bucket
  full : ℝ   -- Weight of water when bucket is full

/-- Given conditions about the bucket weights -/
def bucket_conditions (p q : ℝ) (b : BucketWeight) : Prop :=
  b.empty + (3/4 * b.full) = p ∧ b.empty + (1/3 * b.full) = q

/-- Theorem stating the weight of a fully full bucket -/
theorem full_bucket_weight (p q : ℝ) (b : BucketWeight) 
  (h : bucket_conditions p q b) : 
  b.empty + b.full = (8*p - 3*q) / 5 := by
  sorry

end NUMINAMATH_CALUDE_full_bucket_weight_l1070_107010


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1070_107013

/-- Given a point P with coordinates (m+3, m+1) on the x-axis,
    prove that its coordinates are (2, 0) -/
theorem point_on_x_axis (m : ℝ) : 
  let P : ℝ × ℝ := (m + 3, m + 1)
  P.2 = 0 → P = (2, 0) := by
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1070_107013


namespace NUMINAMATH_CALUDE_cube_root_function_l1070_107077

theorem cube_root_function (k : ℝ) :
  (∀ x, x > 0 → ∃ y, y = k * x^(1/3)) →
  (4 * Real.sqrt 3 = k * 64^(1/3)) →
  (2 * Real.sqrt 3 = k * 8^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_function_l1070_107077


namespace NUMINAMATH_CALUDE_sliding_ladder_inequality_l1070_107097

/-- Represents a sliding ladder against a wall -/
structure SlidingLadder where
  length : ℝ
  topSlideDistance : ℝ
  bottomSlipDistance : ℝ

/-- The bottom slip distance is always greater than the top slide distance for a sliding ladder -/
theorem sliding_ladder_inequality (ladder : SlidingLadder) :
  ladder.bottomSlipDistance > ladder.topSlideDistance :=
sorry

end NUMINAMATH_CALUDE_sliding_ladder_inequality_l1070_107097


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1070_107079

-- Define the universe U
def U : Set ℝ := { x | x > -3 }

-- Define set A
def A : Set ℝ := { x | x < -2 ∨ x > 3 }

-- Define set B
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }

-- Theorem statement
theorem intersection_complement_equality :
  A ∩ (U \ B) = { x | -3 < x ∧ x < -2 ∨ x > 4 } := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1070_107079


namespace NUMINAMATH_CALUDE_flower_expenses_l1070_107023

/-- The total expenses for ordering flowers at Parc Municipal -/
theorem flower_expenses : 
  let tulips : ℕ := 250
  let carnations : ℕ := 375
  let roses : ℕ := 320
  let price_per_flower : ℕ := 2
  (tulips + carnations + roses) * price_per_flower = 1890 := by
  sorry

end NUMINAMATH_CALUDE_flower_expenses_l1070_107023


namespace NUMINAMATH_CALUDE_tangent_line_intercept_l1070_107085

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * exp x - 3 * x + 1

theorem tangent_line_intercept (a : ℝ) :
  let f' := fun x => a * exp x - 3
  (f' 0 = 1) →
  (f a 0 = 5) →
  (∃ b, ∀ x, f a 0 + f' 0 * x = x + b) →
  ∃ b, f a 0 + f' 0 * 0 = 0 + b ∧ b = 5 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_intercept_l1070_107085


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1070_107052

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of the 1st, 7th, and 13th terms equals 8 -/
def product_condition (a : ℕ → ℝ) : Prop :=
  a 1 * a 7 * a 13 = 8

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : product_condition a) : 
  a 3 * a 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1070_107052


namespace NUMINAMATH_CALUDE_cuboid_from_rectangular_projections_l1070_107056

/-- Represents a solid object in 3D space -/
structure Solid :=
  (shape : Type)

/-- Represents an orthographic projection (view) of a solid -/
inductive Projection
  | Rectangle
  | Other

/-- Defines the front view of a solid -/
def front_view (s : Solid) : Projection := sorry

/-- Defines the top view of a solid -/
def top_view (s : Solid) : Projection := sorry

/-- Defines the side view of a solid -/
def side_view (s : Solid) : Projection := sorry

/-- Defines a cuboid -/
def is_cuboid (s : Solid) : Prop := sorry

/-- Theorem: If all three orthographic projections of a solid are rectangles, then the solid is a cuboid -/
theorem cuboid_from_rectangular_projections (s : Solid) :
  front_view s = Projection.Rectangle →
  top_view s = Projection.Rectangle →
  side_view s = Projection.Rectangle →
  is_cuboid s :=
sorry

end NUMINAMATH_CALUDE_cuboid_from_rectangular_projections_l1070_107056


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1070_107084

theorem complex_equation_sum (a b : ℝ) :
  (2 : ℂ) - 2 * Complex.I^3 = a + b * Complex.I → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1070_107084


namespace NUMINAMATH_CALUDE_total_students_correct_l1070_107027

/-- The total number of students at the college -/
def total_students : ℝ := 880

/-- The percentage of students enrolled in biology classes -/
def biology_percentage : ℝ := 32.5

/-- The number of students not enrolled in biology classes -/
def non_biology_students : ℕ := 594

/-- Theorem stating that the total number of students is correct given the conditions -/
theorem total_students_correct :
  (1 - biology_percentage / 100) * total_students = non_biology_students :=
sorry

end NUMINAMATH_CALUDE_total_students_correct_l1070_107027


namespace NUMINAMATH_CALUDE_column_of_1985_l1070_107029

/-- The column number (1-indexed) in which a given odd positive integer appears in the arrangement -/
def columnNumber (n : ℕ) : ℕ :=
  (n % 16 + 15) % 16 / 2 + 1

theorem column_of_1985 : columnNumber 1985 = 1 := by sorry

end NUMINAMATH_CALUDE_column_of_1985_l1070_107029


namespace NUMINAMATH_CALUDE_jessica_initial_money_l1070_107071

/-- The amount of money Jessica spent on a cat toy -/
def spent : ℚ := 10.22

/-- The amount of money Jessica has left -/
def left : ℚ := 1.51

/-- Jessica's initial amount of money -/
def initial : ℚ := spent + left

/-- Theorem stating that Jessica's initial amount of money was $11.73 -/
theorem jessica_initial_money : initial = 11.73 := by
  sorry

end NUMINAMATH_CALUDE_jessica_initial_money_l1070_107071


namespace NUMINAMATH_CALUDE_unique_three_config_score_l1070_107022

/-- Represents a quiz score configuration -/
structure QuizScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- The scoring system for the quiz -/
def score (qs : QuizScore) : ℚ :=
  5 * qs.correct + 1.5 * qs.unanswered

/-- Predicate to check if a QuizScore is valid -/
def is_valid_score (qs : QuizScore) : Prop :=
  qs.correct + qs.unanswered + qs.incorrect = 20

/-- Predicate to check if a rational number is a possible quiz score -/
def is_possible_score (s : ℚ) : Prop :=
  ∃ qs : QuizScore, is_valid_score qs ∧ score qs = s

/-- Predicate to check if a rational number has exactly three distinct valid quiz configurations -/
def has_three_configurations (s : ℚ) : Prop :=
  ∃ qs1 qs2 qs3 : QuizScore,
    is_valid_score qs1 ∧ is_valid_score qs2 ∧ is_valid_score qs3 ∧
    score qs1 = s ∧ score qs2 = s ∧ score qs3 = s ∧
    qs1 ≠ qs2 ∧ qs1 ≠ qs3 ∧ qs2 ≠ qs3 ∧
    ∀ qs : QuizScore, is_valid_score qs → score qs = s → (qs = qs1 ∨ qs = qs2 ∨ qs = qs3)

theorem unique_three_config_score :
  ∀ s : ℚ, 0 ≤ s ∧ s ≤ 100 → has_three_configurations s → s = 75 :=
sorry

end NUMINAMATH_CALUDE_unique_three_config_score_l1070_107022


namespace NUMINAMATH_CALUDE_child_ticket_cost_l1070_107037

theorem child_ticket_cost (num_adults num_children : ℕ) (adult_ticket_cost : ℚ) (extra_cost : ℚ) :
  num_adults = 9 →
  num_children = 7 →
  adult_ticket_cost = 11 →
  extra_cost = 50 →
  ∃ (child_ticket_cost : ℚ),
    num_adults * adult_ticket_cost = num_children * child_ticket_cost + extra_cost ∧
    child_ticket_cost = 7 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l1070_107037


namespace NUMINAMATH_CALUDE_prob_at_least_one_one_value_l1070_107054

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single : ℚ := 1 / 6

/-- The probability of not rolling a specific number on a fair six-sided die -/
def prob_not_single : ℚ := 1 - prob_single

/-- The probability of at least one die showing 1 when two fair six-sided dice are rolled once -/
def prob_at_least_one_one : ℚ := 
  prob_single * prob_not_single + 
  prob_not_single * prob_single + 
  prob_single * prob_single

theorem prob_at_least_one_one_value : prob_at_least_one_one = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_one_value_l1070_107054


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1070_107076

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) :
  (∃ r : ℝ, 20 * r = a ∧ a * r = 5/4) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1070_107076


namespace NUMINAMATH_CALUDE_sanchez_problem_l1070_107053

theorem sanchez_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 2)
  (h2 : x.val * y.val = 120) :
  x.val + y.val = 22 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_problem_l1070_107053


namespace NUMINAMATH_CALUDE_sqrt_sin_sum_equals_neg_two_cos_three_l1070_107034

theorem sqrt_sin_sum_equals_neg_two_cos_three :
  Real.sqrt (1 + Real.sin 6) + Real.sqrt (1 - Real.sin 6) = -2 * Real.cos 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sin_sum_equals_neg_two_cos_three_l1070_107034


namespace NUMINAMATH_CALUDE_sum_reciprocals_l1070_107072

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (h1 : ω^4 = 1)
  (h2 : ω ≠ 1)
  (h3 : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l1070_107072


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l1070_107083

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) = (4 * Real.pi * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l1070_107083


namespace NUMINAMATH_CALUDE_largest_divided_by_smallest_l1070_107049

def numbers : List ℝ := [10, 11, 12]

theorem largest_divided_by_smallest : 
  (List.maximum numbers).get! / (List.minimum numbers).get! = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_largest_divided_by_smallest_l1070_107049


namespace NUMINAMATH_CALUDE_smallest_n_for_unique_k_l1070_107092

theorem smallest_n_for_unique_k : ∃ (k : ℤ), (9 : ℚ)/17 < (3 : ℚ)/(3 + k) ∧ (3 : ℚ)/(3 + k) < 8/15 ∧
  ∀ (n : ℕ), n < 3 → ¬(∃! (k : ℤ), (9 : ℚ)/17 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/15) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_unique_k_l1070_107092


namespace NUMINAMATH_CALUDE_problem_statement_l1070_107070

theorem problem_statement (a b : ℝ) (ha : a > 0) (heq : Real.exp a * (1 - Real.log b) = 1) :
  (1 < b ∧ b < Real.exp 1) ∧ (a > Real.log b) ∧ (b - a < 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1070_107070


namespace NUMINAMATH_CALUDE_expected_rainfall_theorem_l1070_107004

/-- The number of days considered in the weather forecast -/
def days : ℕ := 5

/-- The probability of no rain on a given day -/
def prob_no_rain : ℝ := 0.3

/-- The probability of 3 inches of rain on a given day -/
def prob_3_inches : ℝ := 0.5

/-- The probability of 8 inches of rain on a given day -/
def prob_8_inches : ℝ := 0.2

/-- The amount of rainfall (in inches) for the "no rain" scenario -/
def rain_0 : ℝ := 0

/-- The amount of rainfall (in inches) for the "3 inches" scenario -/
def rain_3 : ℝ := 3

/-- The amount of rainfall (in inches) for the "8 inches" scenario -/
def rain_8 : ℝ := 8

/-- The expected total rainfall over the given number of days -/
def expected_total_rainfall : ℝ := days * (prob_no_rain * rain_0 + prob_3_inches * rain_3 + prob_8_inches * rain_8)

theorem expected_rainfall_theorem : expected_total_rainfall = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_rainfall_theorem_l1070_107004


namespace NUMINAMATH_CALUDE_joan_total_cents_l1070_107065

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of quarters Joan has -/
def joan_quarters : ℕ := 6

/-- Theorem: Joan's total cents -/
theorem joan_total_cents : joan_quarters * quarter_value = 150 := by
  sorry

end NUMINAMATH_CALUDE_joan_total_cents_l1070_107065


namespace NUMINAMATH_CALUDE_cos_180_degrees_l1070_107019

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l1070_107019


namespace NUMINAMATH_CALUDE_harolds_remaining_money_l1070_107005

/-- Represents Harold's financial situation and calculates his remaining money --/
def harolds_finances (income rent car_payment groceries : ℚ) : ℚ := 
  let utilities := car_payment / 2
  let total_expenses := rent + car_payment + utilities + groceries
  let remaining := income - total_expenses
  let retirement_savings := remaining / 2
  remaining - retirement_savings

/-- Theorem stating that Harold will have $650.00 left after expenses and retirement savings --/
theorem harolds_remaining_money :
  harolds_finances 2500 700 300 50 = 650 := by
  sorry

end NUMINAMATH_CALUDE_harolds_remaining_money_l1070_107005


namespace NUMINAMATH_CALUDE_same_gender_officers_l1070_107099

theorem same_gender_officers (total_members : Nat) (boys : Nat) (girls : Nat) :
  total_members = 24 →
  boys = 12 →
  girls = 12 →
  boys + girls = total_members →
  (boys * (boys - 1) + girls * (girls - 1) : Nat) = 264 := by
  sorry

end NUMINAMATH_CALUDE_same_gender_officers_l1070_107099


namespace NUMINAMATH_CALUDE_marks_father_gave_85_l1070_107007

/-- The amount of money Mark's father gave him. -/
def fathers_money (num_books : ℕ) (book_price : ℕ) (money_left : ℕ) : ℕ :=
  num_books * book_price + money_left

/-- Theorem stating that Mark's father gave him $85. -/
theorem marks_father_gave_85 :
  fathers_money 10 5 35 = 85 := by
  sorry

end NUMINAMATH_CALUDE_marks_father_gave_85_l1070_107007


namespace NUMINAMATH_CALUDE_modulus_complex_power_eight_l1070_107024

theorem modulus_complex_power_eight :
  Complex.abs ((2 : ℂ) + Complex.I) ^ 8 = 625 := by
  sorry

end NUMINAMATH_CALUDE_modulus_complex_power_eight_l1070_107024


namespace NUMINAMATH_CALUDE_jose_investment_is_45000_l1070_107033

/-- Represents the investment and profit scenario of Tom and Jose --/
structure InvestmentScenario where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment amount based on the given scenario --/
def calculate_jose_investment (scenario : InvestmentScenario) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that Jose's investment is 45000 given the specific scenario --/
theorem jose_investment_is_45000 :
  let scenario : InvestmentScenario := {
    tom_investment := 30000,
    jose_join_delay := 2,
    total_profit := 72000,
    jose_profit := 40000
  }
  calculate_jose_investment scenario = 45000 := by
  sorry

end NUMINAMATH_CALUDE_jose_investment_is_45000_l1070_107033


namespace NUMINAMATH_CALUDE_circle_radius_is_ten_l1070_107015

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 25 = 6*x + 12*y

/-- The radius of the circle -/
def circle_radius : ℝ := 10

theorem circle_radius_is_ten :
  ∃ (center_x center_y : ℝ),
    ∀ (x y : ℝ), circle_equation x y ↔ 
      (x - center_x)^2 + (y - center_y)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_ten_l1070_107015


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1070_107063

/-- Proves that 37/80 is equal to 0.4625 -/
theorem fraction_to_decimal : (37 : ℚ) / 80 = 0.4625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1070_107063


namespace NUMINAMATH_CALUDE_ratio_problem_l1070_107094

theorem ratio_problem (A B C : ℝ) (h1 : A + B + C = 98) (h2 : A / B = 2 / 3) (h3 : B = 30) :
  B / C = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1070_107094


namespace NUMINAMATH_CALUDE_foma_cannot_guarantee_win_l1070_107080

/-- Represents a player in the coin game -/
inductive Player : Type
| Foma : Player
| Yerema : Player

/-- Represents the state of the game -/
structure GameState :=
(coins : List Nat)  -- List of remaining coin values
(foma_coins : Nat)  -- Total value of Foma's coins
(yerema_coins : Nat)  -- Total value of Yerema's coins
(last_selector : Player)  -- Player who made the last selection

/-- Function to determine the next selector based on current game state -/
def next_selector (state : GameState) : Player :=
  if state.foma_coins > state.yerema_coins then Player.Foma
  else if state.yerema_coins > state.foma_coins then Player.Yerema
  else state.last_selector

/-- Theorem stating that Foma cannot guarantee winning -/
theorem foma_cannot_guarantee_win :
  ∀ (initial_state : GameState),
    initial_state.coins = List.range 25
    → initial_state.foma_coins = 0
    → initial_state.yerema_coins = 0
    → initial_state.last_selector = Player.Foma
    → ¬ (∀ (strategy : GameState → Nat),
         ∃ (final_state : GameState),
           final_state.coins = []
           ∧ final_state.foma_coins > final_state.yerema_coins) :=
sorry

end NUMINAMATH_CALUDE_foma_cannot_guarantee_win_l1070_107080


namespace NUMINAMATH_CALUDE_six_arts_arrangement_l1070_107008

def number_of_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  -- n: total number of lectures
  -- k: position limit for the specific lecture (Mathematics)
  -- m: number of lectures that must be adjacent (Archery and Charioteering)
  sorry

theorem six_arts_arrangement : number_of_arrangements 6 3 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_six_arts_arrangement_l1070_107008


namespace NUMINAMATH_CALUDE_same_color_probability_l1070_107089

def red_plates : ℕ := 7
def blue_plates : ℕ := 5
def green_plates : ℕ := 3

def total_plates : ℕ := red_plates + blue_plates + green_plates

def same_color_pairs : ℕ := (red_plates.choose 2) + (blue_plates.choose 2) + (green_plates.choose 2)
def total_pairs : ℕ := total_plates.choose 2

theorem same_color_probability :
  (same_color_pairs : ℚ) / total_pairs = 34 / 105 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l1070_107089


namespace NUMINAMATH_CALUDE_proper_subset_condition_l1070_107074

def A (a : ℝ) : Set ℝ := {1, 4, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

def valid_a : Set ℝ := {-2, -1, 0, 1, 2}

theorem proper_subset_condition (a : ℝ) : 
  (B a ⊂ A a) ↔ a ∈ valid_a := by sorry

end NUMINAMATH_CALUDE_proper_subset_condition_l1070_107074


namespace NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l1070_107039

theorem sum_reciprocals_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 1/y ≥ 2 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ 1/a + 1/b = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l1070_107039


namespace NUMINAMATH_CALUDE_simplify_expression_l1070_107000

theorem simplify_expression (x y : ℝ) : (35*x - 24*y) + (15*x + 40*y) - (25*x - 49*y) = 25*x + 65*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1070_107000


namespace NUMINAMATH_CALUDE_complement_intersection_equals_three_l1070_107069

universe u

def U : Finset (Fin 5) := {0, 1, 2, 3, 4}
def M : Finset (Fin 5) := {0, 1, 2}
def N : Finset (Fin 5) := {2, 3}

theorem complement_intersection_equals_three :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_three_l1070_107069


namespace NUMINAMATH_CALUDE_factorial_ratio_2017_2016_l1070_107068

-- Define factorial operation
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_ratio_2017_2016 :
  factorial 2017 / factorial 2016 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_2017_2016_l1070_107068


namespace NUMINAMATH_CALUDE_exponential_monotonicity_l1070_107075

theorem exponential_monotonicity (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > 1) : c^a > c^b := by
  sorry

end NUMINAMATH_CALUDE_exponential_monotonicity_l1070_107075


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1070_107028

theorem solution_set_of_inequality (x : ℝ) :
  (x - 1) * (2 - x) > 0 ↔ x ∈ Set.Ioo 1 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1070_107028


namespace NUMINAMATH_CALUDE_six_digit_permutations_count_l1070_107055

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 2, 2, 5, 9, and 9 -/
def six_digit_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of such integers is 180 -/
theorem six_digit_permutations_count : six_digit_permutations = 180 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_permutations_count_l1070_107055


namespace NUMINAMATH_CALUDE_investment_after_three_years_l1070_107066

def compound_interest (initial_investment : ℝ) (interest_rate : ℝ) (additional_investment : ℝ) (years : ℕ) : ℝ :=
  let rec helper (n : ℕ) (current_amount : ℝ) : ℝ :=
    if n = 0 then
      current_amount
    else
      helper (n - 1) ((current_amount * (1 + interest_rate)) + additional_investment)
  helper years initial_investment

theorem investment_after_three_years :
  let initial_investment : ℝ := 500
  let interest_rate : ℝ := 0.02
  let additional_investment : ℝ := 500
  let years : ℕ := 3
  compound_interest initial_investment interest_rate additional_investment years = 2060.80 := by
  sorry

end NUMINAMATH_CALUDE_investment_after_three_years_l1070_107066


namespace NUMINAMATH_CALUDE_product_xyz_l1070_107048

theorem product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5)
  (eq2 : y + 1/z = 2)
  (eq3 : z + 1/x = 8/3) :
  x * y * z = (11 + Real.sqrt 117) / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l1070_107048


namespace NUMINAMATH_CALUDE_complex_square_plus_four_l1070_107002

theorem complex_square_plus_four : 
  let i : ℂ := Complex.I
  (2 - 3*i)^2 + 4 = -1 - 12*i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_plus_four_l1070_107002


namespace NUMINAMATH_CALUDE_savings_after_expense_increase_l1070_107098

def monthly_savings (salary : ℝ) (initial_savings_rate : ℝ) (expense_increase_rate : ℝ) : ℝ :=
  let initial_savings := salary * initial_savings_rate
  let initial_expenses := salary - initial_savings
  let new_expenses := initial_expenses * (1 + expense_increase_rate)
  salary - new_expenses

theorem savings_after_expense_increase :
  monthly_savings 1000 0.25 0.1 = 175 := by sorry

end NUMINAMATH_CALUDE_savings_after_expense_increase_l1070_107098


namespace NUMINAMATH_CALUDE_total_octopus_legs_l1070_107011

/-- The number of legs an octopus has -/
def legs_per_octopus : ℕ := 8

/-- The number of octopuses Sawyer saw -/
def octopuses_seen : ℕ := 5

/-- Theorem: The total number of octopus legs Sawyer saw is 40 -/
theorem total_octopus_legs : octopuses_seen * legs_per_octopus = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_octopus_legs_l1070_107011


namespace NUMINAMATH_CALUDE_complete_square_l1070_107088

theorem complete_square (x : ℝ) : (x^2 - 5*x = 31) → ((x - 5/2)^2 = 149/4) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_complete_square_l1070_107088


namespace NUMINAMATH_CALUDE_expression_evaluation_l1070_107040

theorem expression_evaluation (x : ℝ) : x = 2 → 2 * x^2 - 3 * x + 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1070_107040


namespace NUMINAMATH_CALUDE_exists_winning_strategy_l1070_107078

/-- The set of numbers from which the hidden numbers are chosen -/
def S : Set ℕ := Finset.range 250

/-- A strategy is a function that takes the player's number and the history of announcements,
    and returns the next announcement -/
def Strategy := ℕ → List ℕ → ℕ

/-- The game state consists of both players' numbers and the history of announcements -/
structure GameState :=
  (player_a_number : ℕ)
  (player_b_number : ℕ)
  (announcements : List ℕ)

/-- A game is valid if both players' numbers are in S and the sum of announcements is 20 -/
def valid_game (g : GameState) : Prop :=
  g.player_a_number ∈ S ∧ g.player_b_number ∈ S ∧ g.announcements.sum = 20

/-- A strategy is winning if it allows both players to determine each other's number -/
def winning_strategy (strat_a strat_b : Strategy) : Prop :=
  ∀ (g : GameState), valid_game g →
    ∃ (n : ℕ), strat_a g.player_a_number (g.announcements.take n) = g.player_b_number ∧
               strat_b g.player_b_number (g.announcements.take n) = g.player_a_number

/-- There exists a winning strategy for the game -/
theorem exists_winning_strategy : ∃ (strat_a strat_b : Strategy), winning_strategy strat_a strat_b :=
sorry

end NUMINAMATH_CALUDE_exists_winning_strategy_l1070_107078


namespace NUMINAMATH_CALUDE_zero_location_l1070_107057

def has_unique_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

theorem zero_location (f : ℝ → ℝ) :
  has_unique_zero f 0 16 ∧
  has_unique_zero f 0 8 ∧
  has_unique_zero f 0 4 ∧
  has_unique_zero f 0 2 →
  ∀ x, 2 ≤ x ∧ x < 16 → f x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_location_l1070_107057


namespace NUMINAMATH_CALUDE_total_distance_walked_l1070_107036

/-- Given a constant walking pace and duration, calculate the total distance walked. -/
theorem total_distance_walked (pace : ℝ) (duration : ℝ) (total_distance : ℝ) : 
  pace = 2 → duration = 8 → total_distance = pace * duration → total_distance = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l1070_107036


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1070_107050

/-- The area of a square with adjacent points (1,2) and (4,6) on a Cartesian coordinate plane is 25. -/
theorem square_area_from_adjacent_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1070_107050


namespace NUMINAMATH_CALUDE_greater_number_problem_l1070_107087

theorem greater_number_problem (a b : ℕ+) (h1 : a * b = 4107) (h2 : Nat.gcd a b = 37) (h3 : a > b) : a = 111 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l1070_107087


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_l1070_107021

/-- Represents a club with members and their music preferences -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_dislike_both : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club)
  (h1 : c.total_members = 25)
  (h2 : c.left_handed = 10)
  (h3 : c.jazz_lovers = 18)
  (h4 : c.right_handed_dislike_both = 3)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members)
  (h6 : ∀ (m : ℕ), m < c.total_members → 
    (m ∈ (Finset.range c.jazz_lovers) ∨ 
     m ∈ (Finset.range (c.total_members - c.jazz_lovers - c.right_handed_dislike_both))))
  : {x : ℕ // x = 10 ∧ x ≤ c.left_handed ∧ x ≤ c.jazz_lovers} :=
by
  sorry

#check left_handed_jazz_lovers

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_l1070_107021


namespace NUMINAMATH_CALUDE_perimeter_semicircular_arcs_square_l1070_107035

/-- The perimeter of a region bounded by four semicircular arcs, each constructed on the sides of a square with side length √2, is equal to 2π√2. -/
theorem perimeter_semicircular_arcs_square (side_length : ℝ) : 
  side_length = Real.sqrt 2 → 
  (4 : ℝ) * (π / 2 * side_length) = 2 * π * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_semicircular_arcs_square_l1070_107035


namespace NUMINAMATH_CALUDE_geometric_series_equality_l1070_107093

theorem geometric_series_equality (n : ℕ) : n ≥ 1 → (
  let C : ℕ → ℝ := λ k => 512 * (1 - (1 / 2^k))
  let D : ℕ → ℝ := λ k => (2048 / 3) * (1 - (-1 / 2)^k)
  (∀ k < n, C k ≠ D k) ∧ C n = D n → n = 4
) := by sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l1070_107093


namespace NUMINAMATH_CALUDE_product_calculation_l1070_107081

theorem product_calculation : 
  (1 / 3) * 6 * (1 / 12) * 24 * (1 / 48) * 96 * (1 / 192) * 384 = 16 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l1070_107081


namespace NUMINAMATH_CALUDE_amys_garden_space_l1070_107095

/-- Calculates the total square footage of growing space for Amy's garden beds -/
theorem amys_garden_space (small_bed_length small_bed_width : ℝ)
                           (large_bed_length large_bed_width : ℝ)
                           (num_small_beds num_large_beds : ℕ) :
  small_bed_length = 3 →
  small_bed_width = 3 →
  large_bed_length = 4 →
  large_bed_width = 3 →
  num_small_beds = 2 →
  num_large_beds = 2 →
  (num_small_beds : ℝ) * (small_bed_length * small_bed_width) +
  (num_large_beds : ℝ) * (large_bed_length * large_bed_width) = 42 := by
  sorry

#check amys_garden_space

end NUMINAMATH_CALUDE_amys_garden_space_l1070_107095


namespace NUMINAMATH_CALUDE_smallest_q_in_geometric_sequence_l1070_107026

def is_geometric_sequence (p q r : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ q = p * k ∧ r = q * k

theorem smallest_q_in_geometric_sequence (p q r : ℝ) :
  p > 0 → q > 0 → r > 0 →
  is_geometric_sequence p q r →
  p * q * r = 216 →
  q ≥ 6 ∧ ∃ p' q' r' : ℝ, p' > 0 ∧ q' > 0 ∧ r' > 0 ∧
    is_geometric_sequence p' q' r' ∧ p' * q' * r' = 216 ∧ q' = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_q_in_geometric_sequence_l1070_107026


namespace NUMINAMATH_CALUDE_pool_swimmers_l1070_107014

theorem pool_swimmers (total : ℕ) (first_day : ℕ) (second_day_diff : ℕ) :
  total = 246 →
  first_day = 79 →
  second_day_diff = 47 →
  ∃ (third_day : ℕ), 
    total = first_day + (third_day + second_day_diff) + third_day ∧
    third_day = 60 :=
by sorry

end NUMINAMATH_CALUDE_pool_swimmers_l1070_107014


namespace NUMINAMATH_CALUDE_decagon_ratio_l1070_107012

/-- Represents a decagon with the properties described in the problem -/
structure Decagon :=
  (area : ℝ)
  (bisector_line : Set ℝ × Set ℝ)
  (below_area : ℝ)
  (triangle_base : ℝ)
  (xq : ℝ)
  (qy : ℝ)

/-- The theorem corresponding to the problem -/
theorem decagon_ratio (d : Decagon) : 
  d.area = 15 ∧ 
  d.below_area = 7.5 ∧ 
  d.triangle_base = 7 ∧ 
  d.xq + d.qy = 7 →
  d.xq / d.qy = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_decagon_ratio_l1070_107012
