import Mathlib

namespace NUMINAMATH_CALUDE_train_length_l3937_393711

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 15 → ∃ (length : ℝ), abs (length - 250.05) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l3937_393711


namespace NUMINAMATH_CALUDE_limit_example_l3937_393735

open Real

/-- The limit of (9x^2 - 1) / (x + 1/3) as x approaches -1/3 is -6 -/
theorem limit_example : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x + 1/3| → |x + 1/3| < δ → 
    |(9*x^2 - 1) / (x + 1/3) + 6| < ε := by
sorry

end NUMINAMATH_CALUDE_limit_example_l3937_393735


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_squares_l3937_393737

theorem root_sum_reciprocal_squares (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 → 
  b^3 - 6*b^2 + 11*b - 6 = 0 → 
  c^3 - 6*c^2 + 11*c - 6 = 0 → 
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_squares_l3937_393737


namespace NUMINAMATH_CALUDE_gcd_98_63_l3937_393753

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l3937_393753


namespace NUMINAMATH_CALUDE_perpendicular_points_constant_sum_l3937_393791

/-- The curve E in polar coordinates -/
def curve_E (ρ θ : ℝ) : Prop :=
  ρ^2 * (1/3 * Real.cos θ^2 + 1/2 * Real.sin θ^2) = 1

/-- Theorem: For any two perpendicular points on curve E, the sum of reciprocals of their squared distances from the origin is constant -/
theorem perpendicular_points_constant_sum (ρ₁ ρ₂ θ : ℝ) :
  curve_E ρ₁ θ → curve_E ρ₂ (θ + π/2) → 1/ρ₁^2 + 1/ρ₂^2 = 5/6 := by
  sorry

#check perpendicular_points_constant_sum

end NUMINAMATH_CALUDE_perpendicular_points_constant_sum_l3937_393791


namespace NUMINAMATH_CALUDE_total_travel_time_l3937_393785

/-- Prove that the total time traveled is 4 hours -/
theorem total_travel_time (speed : ℝ) (distance_AB : ℝ) (h1 : speed = 60) (h2 : distance_AB = 120) :
  2 * distance_AB / speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_l3937_393785


namespace NUMINAMATH_CALUDE_rectangle_area_is_9000_l3937_393777

/-- A rectangle WXYZ with given coordinates -/
structure Rectangle where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Z : ℝ × ℤ

/-- The area of a rectangle WXYZ -/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- The theorem stating that the area of the given rectangle is 9000 -/
theorem rectangle_area_is_9000 (r : Rectangle) 
  (h1 : r.W = (2, 3))
  (h2 : r.X = (302, 23))
  (h3 : r.Z.1 = 4) :
  rectangleArea r = 9000 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_is_9000_l3937_393777


namespace NUMINAMATH_CALUDE_glycerin_mixture_problem_l3937_393752

theorem glycerin_mixture_problem :
  let total_volume : ℝ := 100
  let final_concentration : ℝ := 0.75
  let solution1_volume : ℝ := 75
  let solution1_concentration : ℝ := 0.30
  let solution2_volume : ℝ := 75
  let solution2_concentration : ℝ := x
  (solution1_volume * solution1_concentration + solution2_volume * solution2_concentration = total_volume * final_concentration) →
  x = 0.70 :=
by sorry

end NUMINAMATH_CALUDE_glycerin_mixture_problem_l3937_393752


namespace NUMINAMATH_CALUDE_road_trip_cost_equalization_l3937_393702

/-- The amount Jamie must give Dana to equalize costs on a road trip -/
theorem road_trip_cost_equalization
  (X Y Z : ℝ)  -- Amounts paid by Alexi, Jamie, and Dana respectively
  (hXY : Y > X)  -- Jamie paid more than Alexi
  (hYZ : Z > Y)  -- Dana paid more than Jamie
  : (X + Z - 2*Y) / 3 = 
    ((X + Y + Z) / 3 - Y)  -- Amount Jamie should give Dana
    := by sorry

end NUMINAMATH_CALUDE_road_trip_cost_equalization_l3937_393702


namespace NUMINAMATH_CALUDE_prize_points_l3937_393764

/-- The number of chocolate bunnies sold -/
def chocolate_bunnies : ℕ := 8

/-- The points per chocolate bunny -/
def points_per_bunny : ℕ := 100

/-- The number of Snickers bars needed -/
def snickers_bars : ℕ := 48

/-- The points per Snickers bar -/
def points_per_snickers : ℕ := 25

/-- The total points needed for the prize -/
def total_points : ℕ := 2000

theorem prize_points :
  chocolate_bunnies * points_per_bunny + snickers_bars * points_per_snickers = total_points :=
by sorry

end NUMINAMATH_CALUDE_prize_points_l3937_393764


namespace NUMINAMATH_CALUDE_mike_toys_total_cost_l3937_393783

def marbles_cost : ℝ := 9.05
def football_cost : ℝ := 4.95
def baseball_cost : ℝ := 6.52
def toy_car_original_cost : ℝ := 5.50
def toy_car_discount_rate : ℝ := 0.10
def puzzle_cost : ℝ := 2.90
def action_figure_cost : ℝ := 8.80

def total_cost : ℝ :=
  marbles_cost +
  football_cost +
  baseball_cost +
  (toy_car_original_cost * (1 - toy_car_discount_rate)) +
  puzzle_cost +
  action_figure_cost

theorem mike_toys_total_cost :
  total_cost = 36.17 := by
  sorry

end NUMINAMATH_CALUDE_mike_toys_total_cost_l3937_393783


namespace NUMINAMATH_CALUDE_eight_mile_taxi_cost_l3937_393792

/-- Calculates the cost of a taxi ride given the base fare, per-mile charge, and distance traveled. -/
def taxi_cost (base_fare : ℝ) (per_mile_charge : ℝ) (distance : ℝ) : ℝ :=
  base_fare + per_mile_charge * distance

/-- Proves that the cost of an 8-mile taxi ride with a base fare of $2.00 and a per-mile charge of $0.30 is equal to $4.40. -/
theorem eight_mile_taxi_cost :
  taxi_cost 2.00 0.30 8 = 4.40 := by
  sorry

end NUMINAMATH_CALUDE_eight_mile_taxi_cost_l3937_393792


namespace NUMINAMATH_CALUDE_triangle_with_arithmetic_sides_l3937_393773

/-- 
A triangle with sides forming an arithmetic sequence with common difference 1 and area 6 
has sides 3, 4, and 5, and one of its angles is a right angle.
-/
theorem triangle_with_arithmetic_sides (a b c : ℝ) (α β γ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  b = a + 1 ∧ c = b + 1 →  -- sides form arithmetic sequence with difference 1
  (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) = 36 →  -- area is 6 (Heron's formula)
  α + β + γ = π →  -- sum of angles is π
  a * a = b * b + c * c - 2 * b * c * Real.cos α →  -- law of cosines for side a
  b * b = a * a + c * c - 2 * a * c * Real.cos β →  -- law of cosines for side b
  c * c = a * a + b * b - 2 * a * b * Real.cos γ →  -- law of cosines for side c
  (a = 3 ∧ b = 4 ∧ c = 5) ∧ γ = π / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_with_arithmetic_sides_l3937_393773


namespace NUMINAMATH_CALUDE_xiaohong_journey_time_l3937_393770

/-- Represents Xiaohong's journey to the meeting venue -/
structure Journey where
  initialSpeed : ℝ
  totalTime : ℝ

/-- The conditions of Xiaohong's journey -/
def journeyConditions (j : Journey) : Prop :=
  j.initialSpeed * 30 + (j.initialSpeed * 1.25) * (j.totalTime - 55) = j.initialSpeed * j.totalTime

/-- Theorem stating that the total time of Xiaohong's journey is 155 minutes -/
theorem xiaohong_journey_time :
  ∃ j : Journey, journeyConditions j ∧ j.totalTime = 155 := by
  sorry


end NUMINAMATH_CALUDE_xiaohong_journey_time_l3937_393770


namespace NUMINAMATH_CALUDE_orange_cost_l3937_393700

theorem orange_cost (calorie_per_orange : ℝ) (total_money : ℝ) (required_calories : ℝ) (money_left : ℝ) :
  calorie_per_orange = 80 →
  total_money = 10 →
  required_calories = 400 →
  money_left = 4 →
  (total_money - money_left) / (required_calories / calorie_per_orange) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_l3937_393700


namespace NUMINAMATH_CALUDE_inequality_proof_l3937_393703

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) :
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3937_393703


namespace NUMINAMATH_CALUDE_triangle_angle_c_l3937_393750

theorem triangle_angle_c (A B C : ℝ) (a b c : ℝ) : 
  A = 80 * π / 180 →
  a^2 = b * (b + c) →
  A + B + C = π →
  a = 2 * Real.sin (A / 2) →
  b = 2 * Real.sin (B / 2) →
  c = 2 * Real.sin (C / 2) →
  C = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l3937_393750


namespace NUMINAMATH_CALUDE_largest_increase_2007_2008_l3937_393751

def students : Fin 6 → ℕ
  | 0 => 50  -- 2003
  | 1 => 58  -- 2004
  | 2 => 65  -- 2005
  | 3 => 75  -- 2006
  | 4 => 80  -- 2007
  | 5 => 100 -- 2008

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

theorem largest_increase_2007_2008 :
  ∀ i : Fin 5, percentageIncrease (students 4) (students 5) ≥ percentageIncrease (students i) (students (i + 1)) :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_2007_2008_l3937_393751


namespace NUMINAMATH_CALUDE_sum_of_divisors_156_l3937_393707

/-- The sum of positive whole number divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive whole number divisors of 156 is 392 -/
theorem sum_of_divisors_156 : sum_of_divisors 156 = 392 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_156_l3937_393707


namespace NUMINAMATH_CALUDE_english_majors_count_l3937_393795

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem english_majors_count (bio_majors : ℕ) (engineers : ℕ) (total_selections : ℕ) :
  bio_majors = 6 →
  engineers = 5 →
  total_selections = 200 →
  ∃ (eng_majors : ℕ), 
    choose eng_majors 3 * choose bio_majors 3 * choose engineers 3 = total_selections ∧
    eng_majors = 3 :=
by sorry

end NUMINAMATH_CALUDE_english_majors_count_l3937_393795


namespace NUMINAMATH_CALUDE_value_of_a_l3937_393748

theorem value_of_a (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 4) (h3 : c^2 / a = 4) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3937_393748


namespace NUMINAMATH_CALUDE_ratio_composition_l3937_393763

theorem ratio_composition (a b c : ℚ) 
  (hab : a / b = 11 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_composition_l3937_393763


namespace NUMINAMATH_CALUDE_min_roots_sum_squared_l3937_393719

/-- Given a quadratic equation x^2 + 2(k+3)x + k^2 + 3 = 0 with real parameter k,
    this function returns the value of (α - 1)^2 + (β - 1)^2,
    where α and β are the roots of the equation. -/
def rootsSumSquared (k : ℝ) : ℝ :=
  2 * (k + 4)^2 - 12

/-- The minimum value of (α - 1)^2 + (β - 1)^2 where α and β are real roots of
    x^2 + 2(k+3)x + k^2 + 3 = 0, and k is a real parameter. -/
theorem min_roots_sum_squared :
  ∃ (m : ℝ), m = 6 ∧ ∀ (k : ℝ), (∀ (x : ℝ), x^2 + 2*(k+3)*x + k^2 + 3 ≥ 0) →
    rootsSumSquared k ≥ m :=
  sorry

end NUMINAMATH_CALUDE_min_roots_sum_squared_l3937_393719


namespace NUMINAMATH_CALUDE_exists_function_with_properties_l3937_393757

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the properties of the function
def PassesThroughPoint (f : RealFunction) : Prop :=
  f (-2) = 1

def IncreasingInSecondQuadrant (f : RealFunction) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₁ < 0 → x₂ < 0 → f x₁ > 0 → f x₂ > 0 → f x₁ < f x₂

-- Theorem statement
theorem exists_function_with_properties :
  ∃ f : RealFunction, PassesThroughPoint f ∧ IncreasingInSecondQuadrant f :=
sorry

end NUMINAMATH_CALUDE_exists_function_with_properties_l3937_393757


namespace NUMINAMATH_CALUDE_tim_additional_water_consumption_l3937_393755

/-- Represents the amount of water Tim drinks -/
structure WaterConsumption where
  bottles_per_day : ℕ
  quarts_per_bottle : ℚ
  total_ounces_per_week : ℕ
  ounces_per_quart : ℕ
  days_per_week : ℕ

/-- Calculates the additional ounces of water Tim drinks daily -/
def additional_daily_ounces (w : WaterConsumption) : ℚ :=
  ((w.total_ounces_per_week : ℚ) - 
   (w.bottles_per_day : ℚ) * w.quarts_per_bottle * (w.ounces_per_quart : ℚ) * (w.days_per_week : ℚ)) / 
  (w.days_per_week : ℚ)

/-- Theorem stating that Tim drinks an additional 20 ounces of water daily -/
theorem tim_additional_water_consumption :
  let w : WaterConsumption := {
    bottles_per_day := 2,
    quarts_per_bottle := 3/2,
    total_ounces_per_week := 812,
    ounces_per_quart := 32,
    days_per_week := 7
  }
  additional_daily_ounces w = 20 := by
  sorry

end NUMINAMATH_CALUDE_tim_additional_water_consumption_l3937_393755


namespace NUMINAMATH_CALUDE_salt_solution_volume_l3937_393794

/-- Given a salt solution with a concentration of 15 grams per 1000 cubic centimeters,
    prove that 0.375 grams of salt corresponds to 25 cubic centimeters of solution. -/
theorem salt_solution_volume (concentration : ℝ) (volume : ℝ) (salt_amount : ℝ) :
  concentration = 15 / 1000 →
  salt_amount = 0.375 →
  volume * concentration = salt_amount →
  volume = 25 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l3937_393794


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3937_393760

def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3937_393760


namespace NUMINAMATH_CALUDE_range_of_f_range_of_f_complete_l3937_393766

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem range_of_f :
  ∀ y ∈ Set.range f,
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ f x = y) →
  2 ≤ y ∧ y ≤ 6 :=
by sorry

theorem range_of_f_complete :
  ∀ y : ℝ, 2 ≤ y ∧ y ≤ 6 →
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ f x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_f_complete_l3937_393766


namespace NUMINAMATH_CALUDE_distance_XT_equals_twenty_l3937_393759

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseLength : ℝ
  height : ℝ

/-- Represents a frustum created by cutting a pyramid -/
structure Frustum where
  basePyramid : SquarePyramid
  volumeRatio : ℝ  -- Ratio of original pyramid volume to smaller pyramid volume

/-- The distance from the center of the frustum's circumsphere to the apex of the original pyramid -/
def distanceXT (f : Frustum) : ℝ := sorry

theorem distance_XT_equals_twenty (f : Frustum) 
  (h1 : f.basePyramid.baseLength = 10)
  (h2 : f.basePyramid.height = 20)
  (h3 : f.volumeRatio = 9) :
  distanceXT f = 20 := by sorry

end NUMINAMATH_CALUDE_distance_XT_equals_twenty_l3937_393759


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l3937_393790

/-- The sum of the squares of the coefficients of the fully simplified expression 
    3(x^3 - 4x + 5) - 5(2x^3 - x^2 + 3x - 2) is equal to 1428 -/
theorem sum_of_squares_of_coefficients : ∃ (a b c d : ℤ),
  (∀ x : ℝ, 3 * (x^3 - 4*x + 5) - 5 * (2*x^3 - x^2 + 3*x - 2) = a*x^3 + b*x^2 + c*x + d) ∧
  a^2 + b^2 + c^2 + d^2 = 1428 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l3937_393790


namespace NUMINAMATH_CALUDE_tan_cot_45_simplification_l3937_393727

theorem tan_cot_45_simplification :
  let tan_45 : ℝ := 1
  let cot_45 : ℝ := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_cot_45_simplification_l3937_393727


namespace NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_tons_l3937_393728

/-- Represents the weight of a packet in pounds -/
def packet_weight : ℚ := 16 + 4 / 16

/-- Represents the number of packets -/
def num_packets : ℕ := 1680

/-- Represents the number of pounds in a ton -/
def pounds_per_ton : ℕ := 2100

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℚ := (num_packets * packet_weight) / pounds_per_ton

theorem gunny_bag_capacity_is_13_tons : gunny_bag_capacity = 13 := by
  sorry

end NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_tons_l3937_393728


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l3937_393767

theorem tan_theta_minus_pi_fourth (θ : Real) (h : Real.tan θ = 3) : 
  Real.tan (θ - π/4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l3937_393767


namespace NUMINAMATH_CALUDE_standard_deck_probability_l3937_393745

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = black_cards + red_cards)

/-- The probability of drawing a black card, then a red card, then a black card -/
def draw_probability (d : Deck) : ℚ :=
  (d.black_cards : ℚ) * d.red_cards * (d.black_cards - 1) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- Theorem stating the probability for a standard 52-card deck -/
theorem standard_deck_probability :
  let d : Deck := ⟨52, 26, 26, rfl⟩
  draw_probability d = 13 / 102 := by
  sorry

end NUMINAMATH_CALUDE_standard_deck_probability_l3937_393745


namespace NUMINAMATH_CALUDE_always_balanced_arrangement_l3937_393758

-- Define the cube type
structure Cube :=
  (blue_faces : Nat)
  (red_faces : Nat)

-- Define the set of 8 cubes
def CubeSet := List Cube

-- Define the property of a valid cube set
def ValidCubeSet (cs : CubeSet) : Prop :=
  cs.length = 8 ∧
  (cs.map (·.blue_faces)).sum = 24 ∧
  (cs.map (·.red_faces)).sum = 24

-- Define the property of a balanced surface
def BalancedSurface (surface_blue : Nat) (surface_red : Nat) : Prop :=
  surface_blue = surface_red ∧ surface_blue + surface_red = 24

-- Main theorem
theorem always_balanced_arrangement (cs : CubeSet) 
  (h : ValidCubeSet cs) : 
  ∃ (surface_blue surface_red : Nat), 
    BalancedSurface surface_blue surface_red :=
sorry

end NUMINAMATH_CALUDE_always_balanced_arrangement_l3937_393758


namespace NUMINAMATH_CALUDE_cube_coloring_count_dodecahedron_coloring_count_l3937_393741

/-- The number of rotational symmetries of a cube -/
def cube_rotations : ℕ := 24

/-- The number of rotational symmetries of a dodecahedron -/
def dodecahedron_rotations : ℕ := 60

/-- The number of faces of a cube -/
def cube_faces : ℕ := 6

/-- The number of faces of a dodecahedron -/
def dodecahedron_faces : ℕ := 12

/-- Calculates the number of geometrically distinct colorings for a polyhedron -/
def distinct_colorings (faces : ℕ) (rotations : ℕ) : ℕ :=
  (Nat.factorial faces) / rotations

theorem cube_coloring_count :
  distinct_colorings cube_faces cube_rotations = 30 := by sorry

theorem dodecahedron_coloring_count :
  distinct_colorings dodecahedron_faces dodecahedron_rotations = 7983360 := by sorry

end NUMINAMATH_CALUDE_cube_coloring_count_dodecahedron_coloring_count_l3937_393741


namespace NUMINAMATH_CALUDE_two_digit_number_divisible_by_8_12_18_l3937_393722

theorem two_digit_number_divisible_by_8_12_18 :
  ∃! n : ℕ, 60 ≤ n ∧ n ≤ 79 ∧ 8 ∣ n ∧ 12 ∣ n ∧ 18 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_divisible_by_8_12_18_l3937_393722


namespace NUMINAMATH_CALUDE_right_triangle_sin_y_l3937_393765

theorem right_triangle_sin_y (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_a : a = 20) (h_b : b = 21) :
  let sin_y := a / c
  sin_y = 20 / 29 := by sorry

end NUMINAMATH_CALUDE_right_triangle_sin_y_l3937_393765


namespace NUMINAMATH_CALUDE_conference_schedule_ways_l3937_393713

/-- Represents the number of lecturers --/
def n : ℕ := 7

/-- Represents the number of lecturers with specific ordering constraints --/
def k : ℕ := 3

/-- Calculates the number of ways to schedule n lecturers with k lecturers having specific ordering constraints --/
def schedule_ways (n : ℕ) (k : ℕ) : ℕ :=
  (n - k + 1) * Nat.factorial (n - k)

/-- Theorem stating that the number of ways to schedule 7 lecturers with 3 having specific ordering constraints is 600 --/
theorem conference_schedule_ways : schedule_ways n k = 600 := by
  sorry

end NUMINAMATH_CALUDE_conference_schedule_ways_l3937_393713


namespace NUMINAMATH_CALUDE_probability_divisible_by_five_l3937_393769

-- Define a three-digit positive integer with a ones digit of 5
def three_digit_ending_in_five (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 10 = 5

-- Define divisibility by 5
def divisible_by_five (n : ℕ) : Prop :=
  n % 5 = 0

-- Theorem statement
theorem probability_divisible_by_five :
  ∀ n : ℕ, three_digit_ending_in_five n → divisible_by_five n :=
by
  sorry

end NUMINAMATH_CALUDE_probability_divisible_by_five_l3937_393769


namespace NUMINAMATH_CALUDE_a_squared_b_gt_ab_squared_iff_one_over_a_lt_one_over_b_l3937_393720

theorem a_squared_b_gt_ab_squared_iff_one_over_a_lt_one_over_b (a b : ℝ) :
  a^2 * b > a * b^2 ↔ 1/a < 1/b :=
by sorry

end NUMINAMATH_CALUDE_a_squared_b_gt_ab_squared_iff_one_over_a_lt_one_over_b_l3937_393720


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3937_393715

theorem no_integer_solutions : ¬∃ (x y z : ℤ),
  (x^2 - 3*x*y + 2*y^2 - z^2 = 27) ∧
  (-x^2 + 6*y*z + 2*z^2 = 52) ∧
  (x^2 + x*y + 8*z^2 = 110) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3937_393715


namespace NUMINAMATH_CALUDE_square_fence_poles_l3937_393723

theorem square_fence_poles (poles_per_side : ℕ) (h : poles_per_side = 27) :
  poles_per_side * 4 - 4 = 104 :=
by sorry

end NUMINAMATH_CALUDE_square_fence_poles_l3937_393723


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l3937_393743

/-- Given a car's speed in the second hour and its average speed over two hours,
    calculate its speed in the first hour. -/
theorem car_speed_first_hour (second_hour_speed : ℝ) (average_speed : ℝ) :
  second_hour_speed = 60 →
  average_speed = 77.5 →
  (second_hour_speed + (average_speed * 2 - second_hour_speed)) / 2 = average_speed →
  average_speed * 2 - second_hour_speed = 95 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_first_hour_l3937_393743


namespace NUMINAMATH_CALUDE_intersection_M_N_l3937_393762

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3937_393762


namespace NUMINAMATH_CALUDE_taylor_score_ratio_l3937_393774

/-- Given the conditions for Taylor's score mixture, prove the ratio of white to black scores -/
theorem taylor_score_ratio :
  ∀ (white black : ℕ),
  white + black = 78 →
  2 * (black - white) = 3 * 4 →
  (white : ℚ) / black = 6 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_taylor_score_ratio_l3937_393774


namespace NUMINAMATH_CALUDE_a_range_l3937_393793

theorem a_range (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_a_range_l3937_393793


namespace NUMINAMATH_CALUDE_existence_of_special_number_l3937_393778

theorem existence_of_special_number : ∃ N : ℕ, 
  (∃ k : ℕ, k < 150 ∧ k + 1 ≤ 150 ∧ ¬(k ∣ N) ∧ ¬((k + 1) ∣ N)) ∧ 
  (∀ m : ℕ, m ≤ 150 → (∃ k : ℕ, k < 150 ∧ k + 1 ≤ 150 ∧ m ≠ k ∧ m ≠ k + 1) → m ∣ N) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l3937_393778


namespace NUMINAMATH_CALUDE_oil_needed_calculation_l3937_393717

structure Vehicle where
  cylinders : ℕ
  oil_per_cylinder : ℕ
  oil_in_engine : ℕ

def additional_oil_needed (v : Vehicle) : ℕ :=
  v.cylinders * v.oil_per_cylinder - v.oil_in_engine

def car : Vehicle := {
  cylinders := 6,
  oil_per_cylinder := 8,
  oil_in_engine := 16
}

def truck : Vehicle := {
  cylinders := 8,
  oil_per_cylinder := 10,
  oil_in_engine := 20
}

def motorcycle : Vehicle := {
  cylinders := 4,
  oil_per_cylinder := 6,
  oil_in_engine := 8
}

theorem oil_needed_calculation :
  additional_oil_needed car = 32 ∧
  additional_oil_needed truck = 60 ∧
  additional_oil_needed motorcycle = 16 := by
  sorry

end NUMINAMATH_CALUDE_oil_needed_calculation_l3937_393717


namespace NUMINAMATH_CALUDE_combination_sum_permutation_ratio_l3937_393744

-- Define combination function
def C (n : ℕ) (r : ℕ) : ℕ := 
  if r ≤ n then (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))
  else 0

-- Define permutation function
def A (n : ℕ) (r : ℕ) : ℕ := 
  if r ≤ n then (Nat.factorial n) / (Nat.factorial (n - r))
  else 0

-- Theorem 1: Combination sum
theorem combination_sum : C 9 2 + C 9 3 = 120 := by sorry

-- Theorem 2: Permutation ratio
theorem permutation_ratio (n m : ℕ) (h : m < n) : 
  (A n m) / (A (n-1) (m-1)) = n := by sorry

end NUMINAMATH_CALUDE_combination_sum_permutation_ratio_l3937_393744


namespace NUMINAMATH_CALUDE_two_person_travel_problem_l3937_393775

/-- The problem of two people traveling between two locations --/
theorem two_person_travel_problem 
  (distance : ℝ) 
  (total_time : ℝ) 
  (speed_difference : ℝ) :
  distance = 25.5 ∧ 
  total_time = 3 ∧ 
  speed_difference = 2 →
  ∃ (speed_A speed_B : ℝ),
    speed_A = 2 * speed_B + speed_difference ∧
    speed_B * total_time + speed_A * total_time = 2 * distance ∧
    speed_A = 12 ∧
    speed_B = 5 := by sorry

end NUMINAMATH_CALUDE_two_person_travel_problem_l3937_393775


namespace NUMINAMATH_CALUDE_digit_150_is_5_l3937_393780

/-- The decimal representation of 31/198 -/
def decimal_rep : ℚ := 31 / 198

/-- The period of the decimal representation -/
def period : ℕ := 6

/-- The nth digit after the decimal point in the decimal representation -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

/-- The 150th digit after the decimal point in the decimal representation of 31/198 is 5 -/
theorem digit_150_is_5 : nth_digit 150 = 5 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_5_l3937_393780


namespace NUMINAMATH_CALUDE_special_triangle_sides_special_triangle_right_l3937_393714

/-- A triangle with sides in arithmetic progression and area 6 -/
structure SpecialTriangle where
  a : ℝ
  area : ℝ
  sides_arithmetic : a > 0 ∧ area = 6 ∧ a * (a + 1) * (a + 2) / 4 = area

/-- The sides of the special triangle are 3, 4, and 5 -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 3 ∧ t.a + 1 = 4 ∧ t.a + 2 = 5 :=
sorry

/-- The special triangle is a right triangle -/
theorem special_triangle_right (t : SpecialTriangle) : 
  t.a ^ 2 + (t.a + 1) ^ 2 = (t.a + 2) ^ 2 :=
sorry

end NUMINAMATH_CALUDE_special_triangle_sides_special_triangle_right_l3937_393714


namespace NUMINAMATH_CALUDE_problem_statement_l3937_393732

-- Define proposition p
def p : Prop := ∀ a : ℝ, a^2 ≥ 0

-- Define function f
def f (x : ℝ) : ℝ := x^2 - x

-- Define proposition q
def q : Prop := ∀ x y : ℝ, 0 < x ∧ x < y → f x < f y

-- Theorem statement
theorem problem_statement : p ∨ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3937_393732


namespace NUMINAMATH_CALUDE_angle_bisector_relation_l3937_393710

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is on the angle bisector of the second and fourth quadrants -/
def isOnAngleBisector (p : Point) : Prop :=
  (p.x < 0 ∧ p.y > 0) ∨ (p.x > 0 ∧ p.y < 0) ∨ p = ⟨0, 0⟩

/-- Theorem stating that for any point on the angle bisector of the second and fourth quadrants, 
    its x-coordinate is the negative of its y-coordinate -/
theorem angle_bisector_relation (p : Point) (h : isOnAngleBisector p) : p.x = -p.y := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_relation_l3937_393710


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3937_393771

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + 9*k = 0) ↔ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3937_393771


namespace NUMINAMATH_CALUDE_positive_integer_solutions_m_value_when_sum_zero_fixed_solution_integer_m_for_integer_x_l3937_393788

-- Define the system of equations
def system (x y m : ℝ) : Prop :=
  x + 2*y - 6 = 0 ∧ x - 2*y + m*x + 5 = 0

-- Theorem 1: Positive integer solutions
theorem positive_integer_solutions :
  ∀ x y : ℤ, x > 0 ∧ y > 0 ∧ x + 2*y - 6 = 0 → (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 1) :=
sorry

-- Theorem 2: Value of m when x + y = 0
theorem m_value_when_sum_zero :
  ∀ x y m : ℝ, system x y m ∧ x + y = 0 → m = -13/6 :=
sorry

-- Theorem 3: Fixed solution regardless of m
theorem fixed_solution :
  ∀ m : ℝ, 0 - 2*2.5 + m*0 + 5 = 0 :=
sorry

-- Theorem 4: Integer values of m for integer x
theorem integer_m_for_integer_x :
  ∀ x : ℤ, ∀ m : ℤ, (∃ y : ℝ, system x y m) → m = -1 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_m_value_when_sum_zero_fixed_solution_integer_m_for_integer_x_l3937_393788


namespace NUMINAMATH_CALUDE_work_hours_ratio_l3937_393796

def total_hours : ℕ := 157
def rebecca_hours : ℕ := 56

theorem work_hours_ratio (thomas_hours toby_hours : ℕ) : 
  thomas_hours + toby_hours + rebecca_hours = total_hours →
  toby_hours = thomas_hours + 10 →
  rebecca_hours = toby_hours - 8 →
  (toby_hours : ℚ) / (thomas_hours : ℚ) = 32 / 27 := by
  sorry

end NUMINAMATH_CALUDE_work_hours_ratio_l3937_393796


namespace NUMINAMATH_CALUDE_conical_tube_surface_area_l3937_393718

/-- The surface area of a conical tube formed by rolling a semicircular paper. -/
theorem conical_tube_surface_area (r : ℝ) (h : r = 2) : 
  (π * r) = Real.pi * 2 := by
  sorry

end NUMINAMATH_CALUDE_conical_tube_surface_area_l3937_393718


namespace NUMINAMATH_CALUDE_meaningful_iff_condition_l3937_393787

def is_meaningful (x : ℝ) : Prop :=
  x ≥ -1 ∧ x ≠ 0

theorem meaningful_iff_condition (x : ℝ) :
  is_meaningful x ↔ (∃ y : ℝ, y^2 = x + 1) ∧ x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_iff_condition_l3937_393787


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l3937_393731

/-- The slope angle of the line x - y + 1 = 0 is 45 degrees -/
theorem slope_angle_of_line (x y : ℝ) : 
  x - y + 1 = 0 → Real.arctan 1 = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l3937_393731


namespace NUMINAMATH_CALUDE_fraction_equality_implies_cross_product_l3937_393705

theorem fraction_equality_implies_cross_product (x y : ℚ) :
  x / 2 = y / 3 → 3 * x = 2 * y ∧ ¬(2 * x = 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_cross_product_l3937_393705


namespace NUMINAMATH_CALUDE_max_rectangles_intersection_l3937_393797

/-- A rectangle in a plane --/
structure Rectangle where
  -- We don't need to define the specifics of a rectangle for this problem

/-- The number of intersection points between two rectangles --/
def intersection_points (r1 r2 : Rectangle) : ℕ := sorry

/-- The maximum number of intersection points between any two rectangles --/
def max_intersection_points : ℕ := sorry

/-- Theorem: The maximum number of intersection points between any two rectangles is 8 --/
theorem max_rectangles_intersection :
  max_intersection_points = 8 := by sorry

end NUMINAMATH_CALUDE_max_rectangles_intersection_l3937_393797


namespace NUMINAMATH_CALUDE_shelter_ratio_l3937_393781

theorem shelter_ratio (initial_cats : ℕ) (initial_dogs : ℕ) (additional_dogs : ℕ) :
  initial_cats = 45 →
  (initial_cats : ℚ) / initial_dogs = 15 / 7 →
  additional_dogs = 12 →
  (initial_cats : ℚ) / (initial_dogs + additional_dogs) = 15 / 11 :=
by sorry

end NUMINAMATH_CALUDE_shelter_ratio_l3937_393781


namespace NUMINAMATH_CALUDE_fruit_boxes_distribution_l3937_393709

/-- Given 22 boxes distributed among 3 types of fruits, 
    prove that there must be at least 8 boxes of one type of fruit. -/
theorem fruit_boxes_distribution (boxes : ℕ) (fruit_types : ℕ) 
  (h1 : boxes = 22) (h2 : fruit_types = 3) : 
  ∃ (type : ℕ), type ≤ fruit_types ∧ 
  ∃ (boxes_of_type : ℕ), boxes_of_type ≥ 8 ∧ 
  boxes_of_type ≤ boxes := by
  sorry

end NUMINAMATH_CALUDE_fruit_boxes_distribution_l3937_393709


namespace NUMINAMATH_CALUDE_shape_cell_count_l3937_393779

theorem shape_cell_count (n : ℕ) : 
  n < 16 ∧ 
  n % 4 = 0 ∧ 
  n % 3 = 0 → 
  n = 12 := by sorry

end NUMINAMATH_CALUDE_shape_cell_count_l3937_393779


namespace NUMINAMATH_CALUDE_power_function_increasing_exponent_l3937_393730

theorem power_function_increasing_exponent (a : ℝ) :
  (∀ x y : ℝ, 0 < x ∧ x < y → x^a < y^a) → a > 0 := by sorry

end NUMINAMATH_CALUDE_power_function_increasing_exponent_l3937_393730


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3937_393726

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)  -- arithmetic sequence
  (S : ℕ → ℚ)  -- sum function
  (h1 : a 1 = 2022)  -- first term
  (h2 : S 20 = 22)  -- sum of first 20 terms
  (h3 : ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1)))  -- sum formula
  (h4 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- common difference property
  : a 2 - a 1 = -20209 / 95 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3937_393726


namespace NUMINAMATH_CALUDE_history_book_pages_l3937_393736

theorem history_book_pages (novel_pages science_pages history_pages : ℕ) : 
  novel_pages = history_pages / 2 →
  science_pages = 4 * novel_pages →
  science_pages = 600 →
  history_pages = 300 := by
sorry

end NUMINAMATH_CALUDE_history_book_pages_l3937_393736


namespace NUMINAMATH_CALUDE_three_dice_same_number_l3937_393746

/-- A standard six-sided die -/
def StandardDie := Fin 6

/-- The probability of a specific outcome on a standard die -/
def prob_specific_outcome : ℚ := 1 / 6

/-- The probability of all three dice showing the same number -/
def prob_all_same : ℚ := 1 / 36

/-- Theorem: The probability of three standard six-sided dice showing the same number
    when tossed simultaneously is 1/36 -/
theorem three_dice_same_number :
  (1 : ℚ) * prob_specific_outcome * prob_specific_outcome = prob_all_same := by
  sorry

end NUMINAMATH_CALUDE_three_dice_same_number_l3937_393746


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3937_393712

-- Problem 1
theorem problem_1 (x y : ℝ) : (x - y)^2 + x * (x + 2*y) = 2*x^2 + y^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  ((-3*x + 4) / (x - 1) + x) / ((x - 2) / (x^2 - x)) = x^2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3937_393712


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3937_393701

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (0 < a ∧ a < b → 1/a > 1/b) ∧
  ∃ a b : ℝ, 1/a > 1/b ∧ ¬(0 < a ∧ a < b) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3937_393701


namespace NUMINAMATH_CALUDE_ratio_problem_l3937_393747

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 7) :
  a / c = 105 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3937_393747


namespace NUMINAMATH_CALUDE_roll_less_than_5_most_likely_l3937_393738

-- Define the probability of an event on a fair die
def prob (n : ℕ) : ℚ := n / 6

-- Define the events
def roll_6 : ℚ := prob 1
def roll_more_than_4 : ℚ := prob 2
def roll_less_than_4 : ℚ := prob 3
def roll_less_than_5 : ℚ := prob 4

-- Theorem statement
theorem roll_less_than_5_most_likely :
  roll_less_than_5 > roll_6 ∧
  roll_less_than_5 > roll_more_than_4 ∧
  roll_less_than_5 > roll_less_than_4 :=
sorry

end NUMINAMATH_CALUDE_roll_less_than_5_most_likely_l3937_393738


namespace NUMINAMATH_CALUDE_binomial_60_3_l3937_393749

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l3937_393749


namespace NUMINAMATH_CALUDE_square_difference_of_roots_l3937_393776

theorem square_difference_of_roots (α β : ℝ) : 
  (α^2 - 2*α - 4 = 0) → (β^2 - 2*β - 4 = 0) → (α - β)^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_roots_l3937_393776


namespace NUMINAMATH_CALUDE_quadrilateral_area_bound_l3937_393768

-- Define a quadrilateral type
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the area function for a quadrilateral
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area_bound (q : Quadrilateral) :
  area q ≤ (1/4 : ℝ) * (q.a + q.c)^2 + q.b * q.d := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_bound_l3937_393768


namespace NUMINAMATH_CALUDE_equality_proof_l3937_393782

theorem equality_proof (a b c p : ℝ) (h : a + b + c = 2 * p) :
  (2 * a * p + b * c) * (2 * b * p + a * c) * (2 * c * p + a * b) =
  (a + b)^2 * (a + c)^2 * (b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_equality_proof_l3937_393782


namespace NUMINAMATH_CALUDE_truck_catches_bus_l3937_393708

-- Define the vehicles
structure Vehicle :=
  (speed : ℝ)

-- Define the initial positions
def initial_position (bus truck car : Vehicle) : Prop :=
  truck.speed > car.speed ∧ bus.speed > truck.speed

-- Define the time when car catches up with truck
def car_catches_truck (t : ℝ) : Prop := t = 10

-- Define the time when car catches up with bus
def car_catches_bus (t : ℝ) : Prop := t = 15

-- Theorem to prove
theorem truck_catches_bus 
  (bus truck car : Vehicle) 
  (h1 : initial_position bus truck car)
  (h2 : car_catches_truck 10)
  (h3 : car_catches_bus 15) :
  ∃ (t : ℝ), t = 15 ∧ 
    (truck.speed * (15 + t) = bus.speed * 15) :=
sorry

end NUMINAMATH_CALUDE_truck_catches_bus_l3937_393708


namespace NUMINAMATH_CALUDE_questionnaire_survey_l3937_393761

theorem questionnaire_survey (a₁ a₂ a₃ a₄ : ℕ) : 
  a₂ = 60 →
  a₁ + a₂ + a₃ + a₄ = 300 →
  ∃ d : ℤ, a₁ = a₂ - d ∧ a₃ = a₂ + d ∧ a₄ = a₂ + 2*d →
  a₄ = 120 := by
  sorry

end NUMINAMATH_CALUDE_questionnaire_survey_l3937_393761


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3937_393754

-- Define the propositions p and q as functions of x
def p (x : ℝ) : Prop := Real.log (x - 1) < 0
def q (x : ℝ) : Prop := |1 - x| < 2

-- State the theorem
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬(p x)) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3937_393754


namespace NUMINAMATH_CALUDE_train_length_calculation_l3937_393740

theorem train_length_calculation (speed1 speed2 speed3 : ℝ) (time1 time2 time3 : ℝ) :
  let length1 := (speed1 / 3600) * time1
  let length2 := (speed2 / 3600) * time2
  let length3 := (speed3 / 3600) * time3
  speed1 = 300 ∧ speed2 = 250 ∧ speed3 = 350 ∧
  time1 = 33 ∧ time2 = 44 ∧ time3 = 28 →
  length1 + length2 + length3 = 8.52741 :=
by sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l3937_393740


namespace NUMINAMATH_CALUDE_smallest_factor_sum_factorization_exists_l3937_393729

theorem smallest_factor_sum (b : ℤ) : 
  (∃ (p q : ℤ), x^2 + b*x + 2007 = (x + p) * (x + q)) →
  b ≥ 232 :=
by sorry

theorem factorization_exists : 
  ∃ (b p q : ℤ), (b = p + q) ∧ (p * q = 2007) ∧ 
  (x^2 + b*x + 2007 = (x + p) * (x + q)) ∧
  (b = 232) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_sum_factorization_exists_l3937_393729


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3937_393742

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (1 : ℝ) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 5 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = 6 ∧ B = 4 ∧ C = -1 ∧ D = 1 ∧ E = 30 ∧ F = 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3937_393742


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficient_l3937_393734

theorem quadratic_equation_coefficient (a : ℝ) : 
  (∀ x, ∃ y, y = (a - 3) * x^2 - 3 * x - 4) → a ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficient_l3937_393734


namespace NUMINAMATH_CALUDE_existence_of_special_subset_l3937_393716

theorem existence_of_special_subset : 
  ∃ X : Set ℕ+, 
    ∀ n : ℕ+, ∃! (pair : ℕ+ × ℕ+), 
      pair.1 ∈ X ∧ pair.2 ∈ X ∧ n = pair.1 - pair.2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_subset_l3937_393716


namespace NUMINAMATH_CALUDE_objects_meeting_probability_l3937_393786

/-- The probability of two objects meeting in a coordinate plane --/
theorem objects_meeting_probability :
  let start_A : ℕ × ℕ := (0, 0)
  let start_B : ℕ × ℕ := (3, 5)
  let steps : ℕ := 5
  let prob_A_right : ℚ := 1/2
  let prob_A_up : ℚ := 1/2
  let prob_B_left : ℚ := 1/2
  let prob_B_down : ℚ := 1/2
  ∃ (meeting_prob : ℚ), meeting_prob = 31/128 := by
  sorry

end NUMINAMATH_CALUDE_objects_meeting_probability_l3937_393786


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_l3937_393739

/-- An unfair eight-sided die with specific probabilities -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  prob_eight : ℚ
  /-- The probability of rolling any number from 1 to 7 -/
  prob_others : ℚ
  /-- The probability of rolling an 8 is 3/7 -/
  h_prob_eight : prob_eight = 3/7
  /-- The probabilities sum to 1 -/
  h_sum_to_one : prob_eight + 7 * prob_others = 1

/-- The expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℚ :=
  d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + 8 * d.prob_eight

/-- Theorem stating the expected value of the unfair die -/
theorem unfair_die_expected_value (d : UnfairDie) :
  expected_value d = 40/7 := by
  sorry

#eval (40 : ℚ) / 7

end NUMINAMATH_CALUDE_unfair_die_expected_value_l3937_393739


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3937_393725

theorem quadratic_equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃ (a b : ℝ), ∀ (x : ℝ),
    x = a * m + b * n →
    (x + m)^2 - (x + n)^2 = (m - n)^2 →
    a = 0 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3937_393725


namespace NUMINAMATH_CALUDE_spinner_product_even_probability_l3937_393756

def spinner1 : Finset Nat := {2, 5, 7, 11}
def spinner2 : Finset Nat := {3, 4, 6, 8, 10}

def isEven (n : Nat) : Bool := n % 2 = 0

theorem spinner_product_even_probability :
  let totalOutcomes := spinner1.card * spinner2.card
  let evenProductOutcomes := (spinner1.card * spinner2.card) - 
    (spinner1.filter (λ x => ¬isEven x)).card * (spinner2.filter (λ x => ¬isEven x)).card
  (evenProductOutcomes : ℚ) / totalOutcomes = 17 / 20 := by
  sorry

end NUMINAMATH_CALUDE_spinner_product_even_probability_l3937_393756


namespace NUMINAMATH_CALUDE_factorization_of_4x2_minus_16y2_l3937_393704

theorem factorization_of_4x2_minus_16y2 (x y : ℝ) : 4 * x^2 - 16 * y^2 = 4 * (x + 2*y) * (x - 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x2_minus_16y2_l3937_393704


namespace NUMINAMATH_CALUDE_N_is_composite_l3937_393706

/-- The number formed by 2n ones -/
def N (n : ℕ) : ℕ := (10^(2*n) - 1) / 9

/-- Theorem: For all natural numbers n ≥ 1, N(n) is composite -/
theorem N_is_composite (n : ℕ) (h : n ≥ 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N n = a * b := by
  sorry

end NUMINAMATH_CALUDE_N_is_composite_l3937_393706


namespace NUMINAMATH_CALUDE_sum_704_159_base12_l3937_393798

/-- Represents a number in base 12 --/
def Base12 : Type := List (Fin 12)

/-- Converts a base 10 number to base 12 --/
def toBase12 (n : ℕ) : Base12 :=
  sorry

/-- Converts a base 12 number to base 10 --/
def toBase10 (b : Base12) : ℕ :=
  sorry

/-- Adds two base 12 numbers --/
def addBase12 (a b : Base12) : Base12 :=
  sorry

/-- Theorem: The sum of 704₁₂ and 159₁₂ in base 12 is 861₁₂ --/
theorem sum_704_159_base12 :
  addBase12 (toBase12 704) (toBase12 159) = toBase12 861 :=
sorry

end NUMINAMATH_CALUDE_sum_704_159_base12_l3937_393798


namespace NUMINAMATH_CALUDE_triangle_equilateral_l3937_393733

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if (a+b+c)(b+c-a) = 3bc and sin A = 2sin B cos C, 
then the triangle is equilateral with A = B = C = 60°
-/
theorem triangle_equilateral (a b c A B C : ℝ) : 
  (a + b + c) * (b + c - a) = 3 * b * c →
  Real.sin A = 2 * Real.sin B * Real.cos C →
  A = 60 * π / 180 ∧ B = 60 * π / 180 ∧ C = 60 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_triangle_equilateral_l3937_393733


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3937_393721

/-- Given a geometric sequence {a_n} with positive terms, where a_1, (1/2)a_3, 2a_2 form an arithmetic sequence,
    prove that (a_8 + a_9) / (a_6 + a_7) = 3 + 2√2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : ∃ d : ℝ, a 1 + d = (1/2) * a 3 ∧ (1/2) * a 3 + d = 2 * a 2) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3937_393721


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3937_393784

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b / (a + b + 2 * c) + b * c / (b + c + 2 * a) + c * a / (c + a + 2 * b)) ≤ (a + b + c) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3937_393784


namespace NUMINAMATH_CALUDE_s_99_digits_l3937_393724

/-- s(n) is an n-digit number formed by attaching the first n perfect squares, in order, into one integer. -/
def s (n : ℕ) : ℕ := sorry

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The theorem states that s(99) has 189 digits -/
theorem s_99_digits : num_digits (s 99) = 189 := by sorry

end NUMINAMATH_CALUDE_s_99_digits_l3937_393724


namespace NUMINAMATH_CALUDE_birth_year_problem_l3937_393799

theorem birth_year_problem :
  ∃! x : ℕ, 1750 < x ∧ x < 1954 ∧
  (7 * x) % 13 = 11 ∧
  (13 * x) % 11 = 7 ∧
  1954 - x = 86 := by
  sorry

end NUMINAMATH_CALUDE_birth_year_problem_l3937_393799


namespace NUMINAMATH_CALUDE_susie_rhode_island_reds_l3937_393789

/-- The number of Rhode Island Reds that Susie has -/
def susie_reds : ℕ := sorry

/-- The number of Golden Comets that Susie has -/
def susie_comets : ℕ := 6

/-- The number of Rhode Island Reds that Britney has -/
def britney_reds : ℕ := 2 * susie_reds

/-- The number of Golden Comets that Britney has -/
def britney_comets : ℕ := susie_comets / 2

/-- The total number of chickens Susie has -/
def susie_total : ℕ := susie_reds + susie_comets

/-- The total number of chickens Britney has -/
def britney_total : ℕ := britney_reds + britney_comets

theorem susie_rhode_island_reds :
  (britney_total = susie_total + 8) → susie_reds = 11 := by
  sorry

end NUMINAMATH_CALUDE_susie_rhode_island_reds_l3937_393789


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3937_393772

theorem product_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 24)
  (first_eq : x = 3 * (y + z))
  (second_eq : y = 6 * z) :
  x * y * z = 126 := by sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3937_393772
